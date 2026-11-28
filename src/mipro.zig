//! `dspy.MIPROv2` (DSPy 3.5.0): bootstrap few-shot candidate sets, propose
//! instructions grounded in a dataset summary, the predictor's demos and a
//! random prompting tip, then search instruction × demo-set combinations with
//! Optuna's TPE sampler (src/tpe.zig), on minibatches with periodic full
//! evaluations. One CPython `random.Random(seed)` stream is shared by every
//! step, in DSPy's order, so the same LM answers give the same program.
//!
//! The program-aware proposer needs the program's code, which DSPy reads with
//! `inspect.getsource`; DSZig takes it as `Options.program_code` and runs
//! without it, like `program_aware_proposer=False`, when none is given.
//! Not supported: Python object deepcopy and LM call counters.
//!
//! Upstream (DSPy 3.5.0): dspy/teleprompt/mipro_optimizer_v2.py

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const signature = @import("signature.zig");
const program_state = @import("state.zig");
const tpe = @import("tpe.zig");
const unicode = @import("unicode.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const Field = signature.Field;
const LM = @import("lm.zig").LM;
const Metric = optimize.Metric;
const Record = value.Record;

pub const bootstrapped_fewshot_examples_in_context = 3;
pub const labeled_fewshot_examples_in_context = 0;
pub const min_minibatch_size = 50;

/// `GroundedProposer` prompting tips, in DSPy's dict order.
pub const tips = [_][]const u8{
    "",
    "Don't be afraid to be creative when creating the new instruction!",
    "Keep the instruction clear and concise.",
    "Make sure your instruction is very informative and descriptive.",
    "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\")",
};

const no_task_demos = "No task demos provided.";

pub const ObservationSummarizer = signature.Signature(struct {
    pub const instructions = "Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.";
    pub const Input = struct { observations: []const u8 };
    pub const Output = struct { summary: []const u8 };
    pub const descriptions = .{
        .observations = "Observations I have made about my dataset",
        .summary = "Two to Three sentence summary of only the most significant highlights of my observations",
    };
});

pub const DatasetDescriptor = signature.Signature(struct {
    pub const instructions = "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. " ++
        "Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. " ++
        "It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative";
    pub const Input = struct { examples: []const u8 };
    pub const Output = struct { observations: []const u8 };
    pub const descriptions = .{
        .examples = "Sample data points from the dataset",
        .observations = "Somethings that holds true for most or all of the data you observed",
    };
});

pub const DatasetDescriptorWithPriorObservations = signature.Signature(struct {
    pub const instructions = "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. " ++
        "I will also provide you with a few observations I have already made.  Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE' " ++
        "Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. " ++
        "It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative";
    pub const Input = struct { examples: []const u8, prior_observations: []const u8 };
    pub const Output = struct { observations: []const u8 };
    pub const descriptions = .{
        .examples = "Sample data points from the dataset",
        .prior_observations = "Some prior observations I made about the data",
        .observations = "Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add",
    };
});

const generate_instruction = "Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task.";
const dataset_description_desc = "A description of the dataset that we are using.";
const task_demos_desc = "Example inputs/outputs of our module.";
const basic_instruction_desc = "Basic instruction.";
const tip_desc = "A suggestion for how to go about generating the new instruction.";
const proposed_instruction_desc = "Propose an instruction that will be used to prompt a Language Model to perform this task.";
const InstructionOutput = struct { proposed_instruction: []const u8 };
const program_code_desc = "Language model program designed to solve a particular task.";
const program_description_desc = "Summary of the task the program is designed to solve, and how it goes about solving it.";
const module_desc = "The module to create an instruction for.";
const module_description_desc = "Description of the module to create an instruction for.";
const pseudocode_desc = "Pseudocode for a language model program designed to solve a particular task.";
const program_example_desc = "An example of the program in use.";
const describe_preamble = "Below is some pseudo-code for a pipeline that solves tasks with calls to language models. ";

pub const DescribeProgram = signature.Signature(struct {
    pub const instructions = describe_preamble ++ "Please describe what type of task this program appears to be designed to solve, and how it appears to work.";
    pub const Input = struct { program_code: []const u8, program_example: []const u8 };
    pub const Output = struct { program_description: []const u8 };
    pub const descriptions = .{
        .program_code = pseudocode_desc,
        .program_example = program_example_desc,
        .program_description = "Describe what task the program is designed to solve, and how it goes about solving this task.",
    };
});

pub const DescribeModule = signature.Signature(struct {
    pub const instructions = describe_preamble ++ "Please describe the purpose of one of the specified module in this pipeline.";
    pub const Input = struct {
        program_code: []const u8,
        program_example: []const u8,
        program_description: []const u8,
        module: []const u8,
    };
    pub const Output = struct { module_description: []const u8 };
    pub const descriptions = .{
        .program_code = pseudocode_desc,
        .program_example = program_example_desc,
        .program_description = program_description_desc,
        .module = "The module in the program that we want to describe.",
        .module_description = "Description of the module's role in the broader program.",
    };
});

/// The program-aware inputs of `generate_instruction_class(program_aware=True)`.
const ProgramInfo = struct {
    program_code: []const u8,
    program_description: []const u8 = "Not available",
    module: []const u8 = "Not provided",
    module_description: []const u8 = "Not provided",
};

const AwareWithSummaryAndTip = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { dataset_description: []const u8, program_code: []const u8, program_description: []const u8, module: []const u8, module_description: []const u8, task_demos: []const u8, basic_instruction: []const u8, tip: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .program_code = program_code_desc, .program_description = program_description_desc, .module = module_desc, .module_description = module_description_desc, .dataset_description = dataset_description_desc, .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .tip = tip_desc, .proposed_instruction = proposed_instruction_desc };
});
const AwareWithSummary = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { dataset_description: []const u8, program_code: []const u8, program_description: []const u8, module: []const u8, module_description: []const u8, task_demos: []const u8, basic_instruction: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .program_code = program_code_desc, .program_description = program_description_desc, .module = module_desc, .module_description = module_description_desc, .dataset_description = dataset_description_desc, .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .proposed_instruction = proposed_instruction_desc };
});
const AwareWithTip = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { program_code: []const u8, program_description: []const u8, module: []const u8, module_description: []const u8, task_demos: []const u8, basic_instruction: []const u8, tip: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .program_code = program_code_desc, .program_description = program_description_desc, .module = module_desc, .module_description = module_description_desc, .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .tip = tip_desc, .proposed_instruction = proposed_instruction_desc };
});
const Aware = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { program_code: []const u8, program_description: []const u8, module: []const u8, module_description: []const u8, task_demos: []const u8, basic_instruction: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .program_code = program_code_desc, .program_description = program_description_desc, .module = module_desc, .module_description = module_description_desc, .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .proposed_instruction = proposed_instruction_desc };
});

/// `generate_instruction_class(...)` with `program_aware=False` and no
/// instruction history (MIPROv2's configuration), with and without the
/// dataset summary and the tip.
const GenerateInstructionWithSummaryAndTip = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { dataset_description: []const u8, task_demos: []const u8, basic_instruction: []const u8, tip: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .dataset_description = dataset_description_desc, .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .tip = tip_desc, .proposed_instruction = proposed_instruction_desc };
});
const GenerateInstructionWithSummary = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { dataset_description: []const u8, task_demos: []const u8, basic_instruction: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .dataset_description = dataset_description_desc, .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .proposed_instruction = proposed_instruction_desc };
});
const GenerateInstructionWithTip = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { task_demos: []const u8, basic_instruction: []const u8, tip: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .tip = tip_desc, .proposed_instruction = proposed_instruction_desc };
});
const GenerateInstruction = signature.Signature(struct {
    pub const instructions = generate_instruction;
    pub const Input = struct { task_demos: []const u8, basic_instruction: []const u8 };
    pub const Output = InstructionOutput;
    pub const descriptions = .{ .task_demos = task_demos_desc, .basic_instruction = basic_instruction_desc, .proposed_instruction = proposed_instruction_desc };
});

/// `dspy.propose.utils.strip_prefix`: drops a leading `Label:` of at most five
/// words (`^[\*\s]*(([\w'\-]+\s+){0,4}[\w'\-]+):\s*`), then surrounding `"`.
pub fn stripPrefix(text: []const u8) []const u8 {
    var i: usize = 0;
    while (i < text.len and (text[i] == '*' or isSpace(text, i))) i += cpLen(text, i);
    var words: usize = 0;
    const rest = while (true) {
        const start = i;
        while (i < text.len and isWord(text, i)) i += cpLen(text, i);
        if (i == start) break text;
        words += 1;
        if (i < text.len and text[i] == ':') {
            i += 1;
            while (i < text.len and isSpace(text, i)) i += cpLen(text, i);
            break text[i..];
        }
        if (words == 5 or i == text.len or !isSpace(text, i)) break text;
        while (i < text.len and isSpace(text, i)) i += cpLen(text, i);
    };
    return std.mem.trim(u8, rest, "\"");
}

/// Byte length of the UTF-8 sequence at `i` (1 for an invalid byte).
fn cpLen(text: []const u8, i: usize) usize {
    const n = std.unicode.utf8ByteSequenceLength(text[i]) catch return 1;
    if (i + n > text.len) return 1;
    _ = std.unicode.utf8Decode(text[i..][0..n]) catch return 1;
    return n;
}

fn codepoint(text: []const u8, i: usize) ?u21 {
    const n = cpLen(text, i);
    if (n == 1 and text[i] >= 0x80) return null;
    return std.unicode.utf8Decode(text[i..][0..n]) catch null;
}

/// `[\w'\-]` with re's Unicode `\w`.
fn isWord(text: []const u8, i: usize) bool {
    const cp = codepoint(text, i) orelse return false;
    return cp == '\'' or cp == '-' or unicode.isWord(cp);
}

/// re's Unicode `\s`.
fn isSpace(text: []const u8, i: usize) bool {
    const cp = codepoint(text, i) orelse return false;
    return unicode.isSpace(cp);
}

/// `order_input_keys_in_string(repr(examples))` for a list of `dspy.Example`.
pub fn examplesRepr(arena: Allocator, examples: []const Example) (Allocator.Error || error{ JsonTooDeep, AdapterParse })![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    writeExamplesRepr(arena, &aw.writer, examples) catch |err| return switch (err) {
        error.OutOfMemory, error.WriteFailed => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
        error.ParseValue => error.AdapterParse,
    };
    return aw.written();
}

fn writeExamplesRepr(arena: Allocator, w: *Writer, examples: []const Example) (pyjson.Error || Allocator.Error)!void {
    try w.writeByte('[');
    for (examples, 0..) |*ex, n| {
        if (n > 0) try w.writeAll(", ");
        try w.writeAll("Example({");
        var it = ex.record.iterator();
        var first = true;
        while (it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, "dspy_")) continue;
            if (!first) try w.writeAll(", ");
            first = false;
            try pyjson.writeReprString(w, entry.key_ptr.*);
            try w.writeAll(": ");
            try pyjson.writeRepr(w, entry.value_ptr.*, 0);
        }
        try w.writeAll("}) (input_keys=");
        if (ex.input_keys.len == 0) {
            try w.writeAll("set()");
        } else {
            const keys = try arena.alloc([]const u8, ex.input_keys.len);
            for (keys, ex.input_keys) |*k, key| {
                var kw: Writer.Allocating = .init(arena);
                try pyjson.writeReprString(&kw.writer, key);
                k.* = kw.written();
            }
            std.sort.insertion([]const u8, keys, {}, stringLess);
            try w.writeByte('{');
            for (keys, 0..) |k, j| {
                if (j > 0) try w.writeAll(", ");
                try w.writeAll(k);
            }
            try w.writeByte('}');
        }
        try w.writeByte(')');
    }
    try w.writeByte(']');
}

fn stringLess(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// `create_dataset_summary(trainset, view_data_batch_size, prompt_model)`.
pub fn datasetSummary(out: Allocator, prompt: Context, trainset: []const Example, batch_size: usize) predict.Error![]const u8 {
    var ctx = prompt;
    ctx.lm_options.temperature = 1.0;
    const first = try (predict.Predict(DatasetDescriptor){}).forward(ctx, .{
        .examples = try examplesRepr(ctx.arena, trainset[0..@min(trainset.len, batch_size)]),
    });
    var observations: std.ArrayList(u8) = .empty;
    try observations.appendSlice(out, first.observations);
    var skips: usize = 0;
    // A zero Python range step skips refinement, not the first or final call.
    var start = batch_size;
    for (0..9) |_| {
        if (batch_size == 0 or start >= trainset.len) break;
        const end = start + @min(batch_size, trainset.len - start);
        const more = (predict.Predict(DatasetDescriptorWithPriorObservations){}).forward(ctx, .{
            .examples = try examplesRepr(ctx.arena, trainset[start..end]),
            .prior_observations = observations.items,
        }) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => break, // DSPy keeps the observations gathered so far.
        };
        start = end;
        const o = more.observations;
        if (o.len >= 8 and std.ascii.eqlIgnoreCase(o[0..8], "COMPLETE")) {
            skips += 1;
            if (skips >= 5) break;
            continue;
        }
        try observations.appendSlice(out, o);
    }
    const summary = try (predict.Predict(ObservationSummarizer){}).forward(ctx, .{ .observations = observations.items });
    return out.dupe(u8, stripPrefix(summary.summary));
}

/// `GenerateModuleInstruction.forward`'s `task_demos`: up to `max_examples`
/// bootstrapped (`augmented`) demos, starting at set `set_index` and wrapping.
pub fn taskDemos(arena: Allocator, state: *const predict.PredictState, sets: []const []const Record, set_index: usize, max_examples: usize) Allocator.Error![]const u8 {
    assert(set_index < sets.len);
    if (set_index == 0) return no_task_demos;
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    var count: usize = 0;
    outer: for (0..sets.len) |offset| {
        for (sets[(set_index + offset) % sets.len]) |*demo| {
            if (demo.get("augmented") == null) continue;
            for (state.fields, state.prefixes, 0..) |f, prefix, j| {
                if (j > 0) w.writeByte('\n') catch return error.OutOfMemory;
                w.print("{s} ", .{prefix}) catch return error.OutOfMemory;
                if (demo.get(f.name)) |v| {
                    pyjson.writeStr(w, v) catch return error.OutOfMemory;
                } else {
                    w.writeAll("None") catch return error.OutOfMemory;
                }
            }
            w.writeAll("\n\n") catch return error.OutOfMemory;
            count += 1;
            if (count >= max_examples) break :outer;
        }
    }
    if (value.pyStrip(aw.written()).len == 0) return no_task_demos;
    return aw.written();
}

pub const MIPROv2 = struct {
    metric: Metric,
    metric_threshold: ?f64 = null,
    max_bootstrapped_demos: usize = 4,
    max_labeled_demos: usize = 4,
    auto: ?Auto = .light,
    num_candidates: ?usize = null,
    /// Manual-mode instruction budget. Null inherits `num_candidates`.
    num_instruct_candidates: ?usize = null,
    /// Manual-mode demo-set budget. Null inherits `num_candidates`.
    num_fewshot_candidates: ?usize = null,
    /// Null inherits the compile context for demo generation and evaluation.
    max_errors: ?usize = null,
    /// DSPy `num_threads` for trial evaluation; null evaluates serially.
    concurrent_evaluation: ?optimize.ConcurrentEvaluation = null,
    seed: u32 = 9,
    init_temperature: f64 = 1.0,
    /// Retain the evaluated native program values and their ranking.
    track_stats: bool = true,
    /// Optional state-only JSON snapshots in a caller-trusted directory.
    log_dir: ?LogDir = null,
    /// Used while bootstrapping demo candidates.
    teacher_settings: optimize.TeacherSettings = .{},
    /// When present, borrow live settings for the duration of compile. Each
    /// teacher attempt reads them anew instead of using teacher_settings.
    teacher_settings_ref: ?*const optimize.TeacherSettings = null,
    /// Borrowed model handles captured by init. Unbound struct literals use
    /// the compile contexts for model selection.
    prompt_model: ?LM = null,
    task_model: ?LM = null,

    /// Resolve both model roles independently, as DSPy does at construction.
    /// The backing LMs must outlive compilation. No other context settings
    /// are captured.
    pub fn init(config: MIPROv2, settings: Context) error{NoLM}!MIPROv2 {
        const prompt_model = config.prompt_model orelse settings.lm;
        const task_model = config.task_model orelse settings.lm;
        if (prompt_model.sameInstance(.unconfigured) or task_model.sameInstance(.unconfigured))
            return error.NoLM;
        var self = config;
        self.prompt_model = prompt_model;
        self.task_model = task_model;
        return self;
    }

    pub const LogDir = struct {
        io: std.Io,
        path: []const u8,
    };

    const Artifact = union(enum) {
        baseline,
        trial: usize,
        full_eval: usize,
    };

    fn saveCandidate(
        destination: ?LogDir,
        out: Allocator,
        scratch: Allocator,
        program: anytype,
        artifact: Artifact,
    ) !?[]const u8 {
        const dir = destination orelse return null;
        const filename = switch (artifact) {
            .baseline => "program_-1.json",
            .trial => |number| try std.fmt.allocPrint(scratch, "program_{d}.json", .{number}),
            .full_eval => |number| try std.fmt.allocPrint(
                scratch,
                "program_{d}_full_eval.json",
                .{number},
            ),
        };
        const path = try std.fs.path.join(out, &.{ dir.path, "evaluated_programs", filename });
        errdefer out.free(path);

        var file = try std.Io.Dir.cwd().createFileAtomic(dir.io, path, .{
            .make_path = true,
            .replace = true,
        });
        defer file.deinit(dir.io);
        var buffer: [4096]u8 = undefined;
        var writer = file.file.writer(dir.io, &buffer);
        try program_state.save(scratch, program, &writer.interface);
        try writer.flush();
        try file.replace(dir.io);
        return path;
    }

    pub const Auto = enum {
        light,
        medium,
        heavy,

        fn n(a: Auto) usize {
            return switch (a) {
                .light => 6,
                .medium => 12,
                .heavy => 18,
            };
        }

        fn valSize(a: Auto) usize {
            return switch (a) {
                .light => 100,
                .medium => 300,
                .heavy => 1000,
            };
        }
    };

    /// `compile(...)` keyword arguments. Null overrides inherit optimizer defaults.
    pub const Options = struct {
        valset: ?[]const Example = null,
        num_trials: ?usize = null,
        max_bootstrapped_demos: ?usize = null,
        max_labeled_demos: ?usize = null,
        seed: ?u32 = null,
        minibatch: bool = true,
        minibatch_size: usize = 35,
        minibatch_full_eval_steps: usize = 5,
        data_aware_proposer: bool = true,
        view_data_batch_size: usize = 10,
        tip_aware_proposer: bool = true,
        fewshot_aware_proposer: bool = true,
        program_aware_proposer: bool = true,
        /// What DSPy's `get_dspy_source_code(program)` returns for the
        /// program; the program-aware proposer runs only when it is set.
        program_code: ?[]const u8 = null,
    };

    /// One entry of DSPy's `trial_logs`: `params` holds the chosen indices
    /// (`{i}_predictor_instruction`, then `{i}_predictor_demos`, per
    /// predictor) and, as in DSPy, is null for the default program's and
    /// the periodic full evaluations.
    pub const TrialLog = struct {
        number: usize,
        params: ?[]const usize,
        score: f64,
        full_eval: bool,
        /// Trial logging mode, distinct from whether the batch covered the valset.
        minibatch_log: bool = false,
        total_eval_calls_so_far: usize = 0,
        /// Path to the state snapshot, when `log_dir` is configured.
        program_path: ?[]const u8 = null,
    };

    pub fn TrackedPrograms(comptime P: type) type {
        return struct {
            /// Native program values in trial-log order, including the baseline.
            /// As with the winning program, opaque borrowed state is not deep copied.
            by_trial: []const P,
            /// Indices into `by_trial` and `Result.trials`, descending by score.
            /// Ties retain evaluation order.
            full: []const usize,
            minibatch: []const usize,
        };
    }

    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            score: f64,
            /// Per predictor; index 0 is the original instruction.
            instruction_candidates: []const []const []const u8,
            /// Per predictor; empty for zero-shot optimization.
            demo_candidates: []const []const []const Record,
            trials: []const TrialLog,
            /// Scores of `candidate_programs` (full evaluations), descending.
            candidates: []const f64,
            tracked_programs: ?TrackedPrograms(P) = null,
        };
    }

    pub const Error = error{ InvalidOptions, EmptyTrainset, MinibatchTooLarge, NoCandidate };

    /// Results use `out`. `prompt` supplies proposal execution settings and
    /// defaults to `ctx`. Bound models override the respective context LMs.
    pub fn compile(
        defaults: MIPROv2,
        comptime P: type,
        out: Allocator,
        context: Context,
        prompt: ?Context,
        student: P,
        teacher: anytype,
        trainset_in: []const Example,
        options: Options,
    ) !Result(P) {
        var ctx = context;
        if (defaults.task_model) |model| ctx.lm = model;
        var prompt_ctx = prompt orelse ctx;
        if (defaults.prompt_model) |model| prompt_ctx.lm = model;
        var self = defaults;
        self.max_bootstrapped_demos = options.max_bootstrapped_demos orelse defaults.max_bootstrapped_demos;
        self.max_labeled_demos = options.max_labeled_demos orelse defaults.max_labeled_demos;
        const zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0;
        if (self.auto == null and (self.num_candidates == null or options.num_trials == null)) return error.InvalidOptions;
        if (self.auto != null and (self.num_candidates != null or options.num_trials != null)) return error.InvalidOptions;
        const seed = options.seed orelse self.seed;
        var rng: pyrandom.Random = .init(seed);

        if (trainset_in.len == 0) return error.EmptyTrainset;
        var trainset = trainset_in;
        var valset: []const Example = undefined;
        if (options.valset) |v| {
            if (v.len == 0) return error.EmptyTrainset;
            valset = v;
        } else {
            if (trainset.len < 2) return error.EmptyTrainset;
            const len_f: f64 = @floatFromInt(trainset.len);
            const size = @min(1000, @max(1, @as(usize, @intFromFloat(@floor(len_f * 0.80)))));
            valset = trainset[trainset.len - size ..];
            trainset = trainset[0 .. trainset.len - size];
        }

        var program = student;
        const n_preds = (try predict.namedPredictors(out, &program)).len;
        var num_trials: usize = undefined;
        var minibatch = options.minibatch;
        var n_instruct: usize = undefined;
        var n_fewshot: usize = undefined;
        if (self.auto) |auto| {
            valset = try createMinibatch(out, &rng, valset, auto.valSize());
            minibatch = valset.len > min_minibatch_size;
            n_instruct = if (zeroshot) auto.n() else auto.n() / 2;
            n_fewshot = auto.n();
            num_trials = trialsFromCandidates(n_preds, zeroshot, auto.n());
        } else {
            n_instruct = self.num_instruct_candidates orelse self.num_candidates.?;
            n_fewshot = self.num_fewshot_candidates orelse self.num_candidates.?;
            num_trials = options.num_trials.?;
        }
        if (minibatch and options.minibatch_size > valset.len) return error.MinibatchTooLarge;
        if (options.minibatch_full_eval_steps == 0) return error.InvalidOptions;
        // A wrong evaluator binding is a compile configuration error, not
        // a failed candidate evaluation to be scored as zero.
        if (self.concurrent_evaluation) |bound| {
            if (bound.program_type != @import("runtime_value.zig").typeIdentity(P))
                return error.ProgramTypeMismatch;
        }

        const demo_sets = try self.demoCandidates(
            P,
            out,
            ctx,
            program,
            teacher,
            trainset,
            n_fewshot,
            zeroshot,
            &rng,
        );
        const instructions = try self.proposeInstructions(P, out, prompt_ctx, &program, trainset, demo_sets, n_instruct, options, &rng);
        const demos: []const []const []const Record = if (zeroshot) &.{} else demo_sets;

        return self.search(P, out, ctx, program, instructions, demos, valset, num_trials, minibatch, options, seed, &rng);
    }

    /// `_set_num_trials_from_num_candidates`.
    fn trialsFromCandidates(n_preds: usize, zeroshot: bool, n: usize) usize {
        const vars: f64 = @floatFromInt(if (zeroshot) n_preds else n_preds * 2);
        const n_f: f64 = @floatFromInt(n);
        return @intFromFloat(@floor(@max(2 * vars * std.math.log2(n_f), 1.5 * n_f)));
    }

    /// `create_n_fewshot_demo_sets` with MIPROv2's shared `rng`.
    fn demoCandidates(
        self: MIPROv2,
        comptime P: type,
        out: Allocator,
        ctx: Context,
        student: P,
        teacher: anytype,
        trainset: []const Example,
        num_sets: usize,
        zeroshot: bool,
        rng: *pyrandom.Random,
    ) ![]const []const []const Record {
        const max_labeled = if (zeroshot) labeled_fewshot_examples_in_context else self.max_labeled_demos;
        const max_bootstrapped = if (zeroshot) bootstrapped_fewshot_examples_in_context else self.max_bootstrapped_demos;
        var probe = student;
        const n_preds = (try predict.namedPredictors(out, &probe)).len;
        const sets = try out.alloc(std.ArrayList([]const Record), n_preds);
        for (sets) |*s| s.* = .empty;

        var seed: i64 = -3;
        while (seed < @as(i64, @intCast(num_sets)) - 3) : (seed += 1) {
            var candidate: P = undefined;
            if (seed == -3) {
                candidate = student;
                try predict.resetPredictors(out, &candidate);
            } else if (seed == -2 and max_labeled > 0) {
                candidate = try (optimize.LabeledFewShot{ .k = max_labeled }).compile(P, out, student, trainset, .{ .sample = true });
            } else {
                const shuffled = try out.dupe(Example, trainset);
                var size = max_bootstrapped;
                if (seed != -1) {
                    rng.shuffle(Example, shuffled);
                    if (max_bootstrapped < 1) return error.InvalidOptions;
                    size = rng.randint(1, max_bootstrapped);
                }
                var bootstrap: optimize.BootstrapFewShot = .{
                    .metric = self.metric,
                    .metric_threshold = self.metric_threshold,
                    .max_bootstrapped_demos = size,
                    .max_labeled_demos = max_labeled,
                    .max_errors = self.max_errors,
                    .teacher_settings = self.teacher_settings,
                    .teacher_settings_ref = self.teacher_settings_ref,
                };
                candidate = (try bootstrap.compile(P, out, ctx, student, teacher, shuffled)).program;
            }
            for (try predict.namedPredictors(out, &candidate), sets) |p, *s| try s.append(out, p.param.predict.demos);
        }
        const result = try out.alloc([]const []const Record, n_preds);
        for (result, sets) |*r, s| r.* = s.items;
        return result;
    }

    /// `GroundedProposer.propose_instructions_for_program`, then index 0 of
    /// every predictor's list is reset to its original instruction.
    fn proposeInstructions(
        self: MIPROv2,
        comptime P: type,
        out: Allocator,
        prompt: Context,
        program: *P,
        trainset: []const Example,
        demo_sets: []const []const []const Record,
        n_instruct: usize,
        options: Options,
        rng: *pyrandom.Random,
    ) ![]const []const []const u8 {
        const summary: ?[]const u8 = if (options.data_aware_proposer)
            datasetSummary(out, prompt, trainset, options.view_data_batch_size) catch |err| switch (err) {
                error.OutOfMemory => return err,
                else => null, // DSPy runs without the data-aware proposer.
            }
        else
            null;
        const preds = try predict.namedPredictors(out, program);
        const num_demos = @max(if (demo_sets.len > 0) demo_sets[0].len else n_instruct, 1);
        const per_pred = @min(n_instruct, num_demos);
        // A nonempty predictor graph needs slot zero for the original instructions.
        if (preds.len != 0 and per_pred == 0) return error.InvalidOptions;
        const result = try out.alloc([]const []const u8, preds.len);
        for (preds, result, 0..) |p, *list, pred_i| {
            const state = p.param.predict;
            const items = try out.alloc([]const u8, per_pred);
            for (items, 0..) |*item, set_i| {
                var tip: []const u8 = "";
                if (options.tip_aware_proposer) tip = tips[rng.randbelow(tips.len)];
                var ctx = prompt;
                ctx.lm_options.rollout_id = .{ .integer = @intCast(rng.randint(0, 1_000_000_000)) };
                ctx.lm_options.temperature = self.init_temperature;
                const demos_text = if (options.fewshot_aware_proposer and demo_sets.len > 0) blk: {
                    const sets = demo_sets[pred_i];
                    if (set_i >= sets.len) return error.InvalidOptions;
                    break :blk try taskDemos(
                        ctx.arena,
                        state,
                        sets,
                        set_i,
                        bootstrapped_fewshot_examples_in_context,
                    );
                } else no_task_demos;
                const info: ?ProgramInfo = if (options.program_aware_proposer)
                    if (options.program_code) |code| try describe(ctx, code, demos_text, state) else null
                else
                    null;
                const proposed = try generate(ctx, summary, tip, demos_text, state.instructions, info);
                item.* = try out.dupe(u8, stripPrefix(stripPrefix(proposed)));
            }
            items[0] = state.instructions;
            list.* = items;
        }
        return result;
    }

    /// `GenerateModuleInstruction.forward`'s program summary. A failed LM
    /// call keeps the defaults for what is not yet known, as DSPy does.
    fn describe(ctx: Context, code: []const u8, task_demos: []const u8, state: *const predict.PredictState) Allocator.Error!ProgramInfo {
        var info: ProgramInfo = .{ .program_code = code };
        const described = (predict.Predict(DescribeProgram){}).forward(ctx, .{
            .program_code = code,
            .program_example = task_demos,
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return info,
        };
        info.program_description = stripPrefix(described.program_description);
        var aw: Writer.Allocating = .init(ctx.arena);
        const w = &aw.writer;
        w.writeAll("Predict(") catch return error.OutOfMemory;
        for (state.fields, 0..) |f, i| {
            if (i == state.input_count) w.writeAll(") -> ") catch return error.OutOfMemory;
            if (i != 0 and i != state.input_count) w.writeAll(", ") catch return error.OutOfMemory;
            w.writeAll(f.name) catch return error.OutOfMemory;
        }
        info.module = aw.written();
        const module = (predict.Predict(DescribeModule){}).forward(ctx, .{
            .program_code = code,
            .program_example = task_demos,
            .program_description = info.program_description,
            .module = info.module,
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return info,
        };
        info.module_description = module.module_description;
        return info;
    }

    fn generate(
        ctx: Context,
        summary: ?[]const u8,
        tip: []const u8,
        task_demos: []const u8,
        basic: []const u8,
        program: ?ProgramInfo,
    ) predict.Error![]const u8 {
        const p = program orelse return generatePlain(ctx, summary, tip, task_demos, basic);
        if (summary) |s| {
            if (tip.len > 0) return (try (predict.Predict(AwareWithSummaryAndTip){}).forward(ctx, .{
                .dataset_description = s,
                .program_code = p.program_code,
                .program_description = p.program_description,
                .module = p.module,
                .module_description = p.module_description,
                .task_demos = task_demos,
                .basic_instruction = basic,
                .tip = tip,
            })).proposed_instruction;
            return (try (predict.Predict(AwareWithSummary){}).forward(ctx, .{
                .dataset_description = s,
                .program_code = p.program_code,
                .program_description = p.program_description,
                .module = p.module,
                .module_description = p.module_description,
                .task_demos = task_demos,
                .basic_instruction = basic,
            })).proposed_instruction;
        }
        if (tip.len > 0) return (try (predict.Predict(AwareWithTip){}).forward(ctx, .{
            .program_code = p.program_code,
            .program_description = p.program_description,
            .module = p.module,
            .module_description = p.module_description,
            .task_demos = task_demos,
            .basic_instruction = basic,
            .tip = tip,
        })).proposed_instruction;
        return (try (predict.Predict(Aware){}).forward(ctx, .{
            .program_code = p.program_code,
            .program_description = p.program_description,
            .module = p.module,
            .module_description = p.module_description,
            .task_demos = task_demos,
            .basic_instruction = basic,
        })).proposed_instruction;
    }

    fn generatePlain(ctx: Context, summary: ?[]const u8, tip: []const u8, task_demos: []const u8, basic: []const u8) predict.Error![]const u8 {
        if (summary) |s| {
            if (tip.len > 0) return (try (predict.Predict(GenerateInstructionWithSummaryAndTip){}).forward(ctx, .{
                .dataset_description = s,
                .task_demos = task_demos,
                .basic_instruction = basic,
                .tip = tip,
            })).proposed_instruction;
            return (try (predict.Predict(GenerateInstructionWithSummary){}).forward(ctx, .{
                .dataset_description = s,
                .task_demos = task_demos,
                .basic_instruction = basic,
            })).proposed_instruction;
        }
        if (tip.len > 0) return (try (predict.Predict(GenerateInstructionWithTip){}).forward(ctx, .{
            .task_demos = task_demos,
            .basic_instruction = basic,
            .tip = tip,
        })).proposed_instruction;
        return (try (predict.Predict(GenerateInstruction){}).forward(ctx, .{
            .task_demos = task_demos,
            .basic_instruction = basic,
        })).proposed_instruction;
    }

    /// `_optimize_prompt_parameters`.
    fn search(
        self: MIPROv2,
        comptime P: type,
        out: Allocator,
        ctx: Context,
        program: P,
        instructions: []const []const []const u8,
        demos: []const []const []const Record,
        valset: []const Example,
        num_trials: usize,
        minibatch: bool,
        options: Options,
        seed: u32,
        rng: *pyrandom.Random,
    ) !Result(P) {
        const n_preds = instructions.len;
        assert(demos.len == 0 or demos.len == n_preds);
        var params: std.ArrayList(tpe.Param) = .empty;
        for (instructions, 0..) |list, i| {
            try params.append(out, .{ .name = try std.fmt.allocPrint(out, "{d}_predictor_instruction", .{i}), .choices = list.len });
            if (demos.len > 0) try params.append(out, .{ .name = try std.fmt.allocPrint(out, "{d}_predictor_demos", .{i}), .choices = demos[i].len });
        }

        const steps = options.minibatch_full_eval_steps;
        const interval = if (minibatch)
            std.math.add(usize, steps, 1) catch return error.InvalidOptions
        else
            0;
        // Include the baseline in the trial numbering even without minibatching.
        const base_count = std.math.add(usize, num_trials, 1) catch return error.InvalidOptions;
        const program_limit = if (minibatch) count: {
            const full_evals = num_trials / steps + @intFromBool(num_trials % steps != 0);
            break :count std.math.add(usize, base_count, full_evals) catch
                return error.InvalidOptions;
        } else base_count;

        const workspace = ctx.workspace orelse return error.WorkspaceRequired;
        var baseline = workspace.open();
        defer baseline.close();

        const default_score = try self.score(P, out, ctx, &program, valset, valset.len, rng);
        var logs: std.ArrayList(TrialLog) = .empty;
        var programs: std.ArrayList(P) = .empty;
        if (self.track_stats) try programs.ensureTotalCapacity(out, program_limit);
        var total_eval_calls = valset.len;
        try logs.append(out, .{
            .number = 1,
            .params = null,
            .score = default_score,
            .full_eval = true,
            .total_eval_calls_so_far = total_eval_calls,
            .program_path = try saveCandidate(self.log_dir, out, baseline.allocator(), &program, .baseline),
        });
        if (self.track_stats) try programs.append(out, program);
        var full_scores: std.ArrayList(f64) = .empty;
        if (!self.track_stats) try full_scores.append(out, default_score);
        var full_count: usize = 1;
        var best = program;
        var best_score = default_score;

        const Combo = struct { params: []const usize, total: f64, count: usize, program: P, evaluated: bool };
        var combos: std.ArrayList(Combo) = .empty;

        // Optuna create_trial rejects NaN for explicitly inserted observations.
        // The baseline's snapshot has already been written at this point.
        if (std.math.isNan(default_score)) return error.NaNStudyObservation;
        var history: std.ArrayList(tpe.Trial) = .empty;
        try history.append(out, .{ .params = try zeros(out, params.items.len), .value = default_score });

        var sampler: tpe.Sampler = .init(seed);

        if (num_trials > 0) {
            for (params.items) |param| {
                if (param.choices == 0) return error.InvalidOptions;
            }
        }
        for (0..num_trials) |_| {
            // Failed objectives still get a study number and a candidate log,
            // but only completed observations count toward TPE startup/fitting.
            const trial_num = logs.items.len + 1;
            const chosen = try out.alloc(usize, params.items.len);
            var scratch = workspace.open();
            defer scratch.close();
            try sampler.sample(scratch.allocator(), params.items, history.items, chosen);

            var candidate = program;
            for (try predict.namedPredictors(out, &candidate), 0..) |p, i| {
                const stride: usize = if (demos.len > 0) 2 else 1;
                try p.param.predict.forkSignatureClass(out);
                p.param.predict.instructions = instructions[i][chosen[i * stride]];
                if (demos.len > 0) p.param.predict.demos = demos[i][chosen[i * stride + 1]];
            }
            const batch = if (minibatch) options.minibatch_size else valset.len;
            const s = try self.score(P, out, ctx, &candidate, valset, batch, rng);
            if (!minibatch and s > best_score) {
                best_score = s;
                best = candidate;
            }
            const full = batch >= valset.len;
            if (full) {
                full_count += 1;
                if (!self.track_stats) try full_scores.append(out, s);
            }
            total_eval_calls = std.math.add(usize, total_eval_calls, batch) catch
                return error.InvalidOptions;
            try logs.append(out, .{
                .number = trial_num,
                .params = chosen,
                .score = s,
                .full_eval = full,
                .minibatch_log = minibatch,
                .total_eval_calls_so_far = total_eval_calls,
                .program_path = try saveCandidate(
                    self.log_dir,
                    out,
                    scratch.allocator(),
                    &candidate,
                    .{ .trial = trial_num },
                ),
            });
            if (self.track_stats) try programs.append(out, candidate);
            if (!std.math.isNan(s)) try history.append(out, .{ .params = chosen, .value = s });
            const combo = for (combos.items) |*c| {
                if (std.mem.eql(usize, c.params, chosen)) break c;
            } else blk: {
                try combos.append(out, .{ .params = chosen, .total = 0, .count = 0, .program = candidate, .evaluated = false });
                break :blk &combos.items[combos.items.len - 1];
            };
            combo.total += s;
            combo.count += 1;

            if (minibatch and (trial_num % interval == 0 or trial_num == program_limit - 1)) {
                var pick: ?*Combo = null;
                var pick_mean: f64 = 0;
                for (combos.items) |*c| {
                    if (c.evaluated) continue;
                    const mean = c.total / @as(f64, @floatFromInt(c.count));
                    if (pick == null or mean > pick_mean) {
                        pick = c;
                        pick_mean = mean;
                    }
                }
                const target = pick orelse return error.NoCandidate;
                const full_score = try self.score(P, out, ctx, &target.program, valset, valset.len, rng);
                if (std.math.isNan(full_score)) return error.NaNStudyObservation;
                full_count += 1;
                if (!self.track_stats) try full_scores.append(out, full_score);
                try history.append(out, .{ .params = target.params, .value = full_score });
                total_eval_calls = std.math.add(usize, total_eval_calls, valset.len) catch
                    return error.InvalidOptions;
                try logs.append(out, .{
                    .number = trial_num + 1,
                    .params = null,
                    .score = full_score,
                    .full_eval = true,
                    .total_eval_calls_so_far = total_eval_calls,
                    .program_path = try saveCandidate(
                        self.log_dir,
                        out,
                        scratch.allocator(),
                        &target.program,
                        .{ .full_eval = trial_num + 1 },
                    ),
                });
                if (self.track_stats) try programs.append(out, target.program);
                target.evaluated = true;
                if (full_score > best_score) {
                    best_score = full_score;
                    best = target.program;
                }
            }
        }
        const tracked = if (self.track_stats)
            try rankPrograms(P, out, programs.items, logs.items, full_count)
        else
            null;
        const ranked_scores = if (tracked) |ranking| ranked: {
            const scores = try out.alloc(f64, full_count);
            for (scores, ranking.full) |*s, index| s.* = logs.items[index].score;
            break :ranked scores;
        } else ranked: {
            std.sort.block(f64, full_scores.items, {}, std.sort.desc(f64));
            break :ranked full_scores.items;
        };
        return .{
            .program = best,
            .score = best_score,
            .instruction_candidates = instructions,
            .demo_candidates = demos,
            .trials = logs.items,
            .candidates = ranked_scores,
            .tracked_programs = tracked,
        };
    }

    fn rankPrograms(
        comptime P: type,
        out: Allocator,
        programs: []const P,
        logs: []const TrialLog,
        full_count: usize,
    ) Allocator.Error!TrackedPrograms(P) {
        assert(programs.len == logs.len);
        const order = try out.alloc(usize, logs.len);
        var full_index: usize = 0;
        var mb_index = full_count;
        for (logs, 0..) |log, index| {
            if (log.full_eval) {
                order[full_index] = index;
                full_index += 1;
            } else {
                order[mb_index] = index;
                mb_index += 1;
            }
        }
        assert(full_index == full_count);
        assert(mb_index == order.len);
        const ByScore = struct {
            fn less(trials: []const TrialLog, left: usize, right: usize) bool {
                return trials[left].score > trials[right].score;
            }
        };
        const full = order[0..full_count];
        const minibatch = order[full_count..];
        std.sort.block(usize, full, logs, ByScore.less);
        std.sort.block(usize, minibatch, logs, ByScore.less);
        return .{ .by_trial = programs, .full = full, .minibatch = minibatch };
    }

    /// `eval_candidate_program`: a minibatch drawn with `rng` unless
    /// `batch_size` covers `valset`; any failure scores 0.0, as in DSPy.
    fn score(self: MIPROv2, comptime P: type, out: Allocator, ctx: Context, program: *const P, valset: []const Example, batch_size: usize, rng: *pyrandom.Random) Allocator.Error!f64 {
        const devset = if (batch_size >= valset.len) valset else try createMinibatch(out, rng, valset, batch_size);
        const max_errors = self.max_errors orelse ctx.max_errors;
        const result = optimize.evaluateWith(P, self.concurrent_evaluation, ctx, program, devset, self.metric, max_errors) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return 0.0,
        };
        return result.score;
    }
};

/// `create_minibatch(trainset, batch_size, rng)`.
fn createMinibatch(out: Allocator, rng: *pyrandom.Random, set: []const Example, batch_size: usize) Allocator.Error![]const Example {
    const k = @min(batch_size, set.len);
    const indices = try out.alloc(usize, k);
    const scratch = try out.alloc(usize, set.len);
    rng.sampleIndices(set.len, indices, scratch);
    const result = try out.alloc(Example, k);
    for (result, indices) |*r, i| r.* = set[i];
    return result;
}

fn zeros(out: Allocator, n: usize) Allocator.Error![]usize {
    const z = try out.alloc(usize, n);
    @memset(z, 0);
    return z;
}

const testing = std.testing;

test "MIPRO zero-size dataset summary skips refinement but uses the initial observations" {
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const lm_mod = @import("lm.zig");
    var history: lm_mod.History = .{ .arena = arena };
    var model: lm_mod.ScriptedLM = .{ .history = &history, .responses = &.{
        "{\"observations\":\"Initial.\"}",
        "{\"summary\":\"Facts.\"}",
    } };
    const sample = try Example.fromJson(arena, "{\"q\":\"unseen row\"}", &.{"q"});
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = model.lm(), .adapter = .json };
    try testing.expectEqualStrings("Facts.", try datasetSummary(arena, ctx, &.{sample}, 0));

    const requests = history.requests.items;
    try testing.expectEqual(@as(usize, 2), requests.len);
    const initial = requests[0].messages;
    const final = requests[1].messages;
    try testing.expect(std.mem.indexOf(u8, initial[initial.len - 1].content.text, "[]") != null);
    try testing.expect(std.mem.indexOf(u8, final[final.len - 1].content.text, "Initial.") != null);
}

test "MIPRO compile demo overrides select zero-shot and seed zero samples independently" {
    const P = predict.Predict(@import("signature.zig").Signature(struct {
        pub const instructions = "base";
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    const metric = struct {
        fn score(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            return 1;
        }
    }.score;
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const lm_mod = @import("lm.zig");
    const answers: [32][]const u8 = @splat("<answer>\n1\n</answer>");
    const proposals = [_][]const u8{
        "<proposed_instruction>\nalpha\n</proposed_instruction>",
        "<proposed_instruction>\nbeta\n</proposed_instruction>",
    };
    var rows: [5]Example = undefined;
    for (&rows, 0..) |*row, i| {
        const json = try std.fmt.allocPrint(arena, "{{\"question\":\"{d}\",\"answer\":1}}", .{i});
        row.* = try Example.fromJson(arena, json, &.{"question"});
    }
    const student: P = .{};
    const optimizer: MIPROv2 = .{ .metric = metric, .auto = null, .num_candidates = 2 };
    for ([_]?u32{ 0, null }, [_][]const u8{ "4", "1" }) |seed, first_sample| {
        var history: lm_mod.History = .{ .arena = arena };
        var task: lm_mod.ScriptedLM = .{ .responses = &answers, .history = &history };
        var prompt: lm_mod.ScriptedLM = .{ .responses = &proposals };
        var unused: lm_mod.ScriptedLM = .{ .responses = &.{} };
        var config = optimizer;
        config.task_model = task.lm();
        const bound = try config.init(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = prompt.lm() });
        const result = try bound.compile(
            P,
            arena,
            .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = unused.lm() },
            null,
            student,
            student,
            rows[0..2],
            .{
                .valset = &rows,
                .num_trials = 3,
                .seed = seed,
                .max_bootstrapped_demos = 0,
                .max_labeled_demos = 0,
                .minibatch_size = 2,
                .minibatch_full_eval_steps = 2,
                .program_aware_proposer = false,
                .data_aware_proposer = false,
                .fewshot_aware_proposer = false,
                .tip_aware_proposer = false,
            },
        );
        try testing.expectEqualSlices(usize, &.{1}, result.trials[1].params.?);
        try testing.expectEqual(proposals.len, prompt.index);
        try testing.expectEqualStrings("beta", result.instruction_candidates[0][1]);
        const request = history.requests.items[7];
        const actual = request.messages[request.messages.len - 1].content.text;
        const expected = try std.fmt.allocPrint(arena, "<question>\n{s}\n</question>", .{first_sample});
        try testing.expect(std.mem.indexOf(u8, actual, expected) != null);
    }

    // Even with valid candidate proposals, an unrepresentable schedule must
    // refuse rather than panic or start evaluating the validation rows.
    for ([_]struct { trials: usize, steps: usize, minibatch: bool }{
        .{ .trials = std.math.maxInt(usize) - 1, .steps = 2, .minibatch = true },
        .{ .trials = 3, .steps = std.math.maxInt(usize), .minibatch = true },
        .{ .trials = std.math.maxInt(usize), .steps = 2, .minibatch = false },
    }) |case| {
        var history: lm_mod.History = .{ .arena = arena };
        var task: lm_mod.ScriptedLM = .{ .responses = &answers, .history = &history };
        var prompt: lm_mod.ScriptedLM = .{ .responses = &proposals };
        try testing.expectError(error.InvalidOptions, optimizer.compile(
            P,
            arena,
            .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = task.lm() },
            .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = prompt.lm() },
            student,
            student,
            rows[0..2],
            .{
                .valset = &rows,
                .num_trials = case.trials,
                .max_bootstrapped_demos = 0,
                .max_labeled_demos = 0,
                .minibatch = case.minibatch,
                .minibatch_size = 2,
                .minibatch_full_eval_steps = case.steps,
                .program_aware_proposer = false,
                .data_aware_proposer = false,
                .fewshot_aware_proposer = false,
                .tip_aware_proposer = false,
            },
        ));
        for (history.requests.items) |request| {
            const input = request.messages[request.messages.len - 1].content.text;
            try testing.expect(std.mem.indexOf(u8, input, "<question>\n2\n</question>") == null);
        }
    }
}

test "MIPRO refuses a mismatched evaluator before using either model" {
    const Program = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    const Callback = struct {
        fn evaluate(_: *anyopaque, _: *const anyopaque, _: []const Example, _: Metric, _: usize, _: ?[]f64) anyerror!optimize.Evaluation {
            return error.UnexpectedEvaluatorDispatch;
        }

        fn score(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const rows = [_]Example{try Example.fromJson(arena, "{\"question\":\"example\",\"answer\":1}", &.{"question"})};
    const lm = @import("lm.zig");
    var task: lm.ScriptedLM = .{ .responses = &.{"<answer>1</answer>"} };
    var prompt: lm.ScriptedLM = .{ .responses = &.{"<proposed_instruction>candidate</proposed_instruction>"} };
    const optimizer: MIPROv2 = .{
        .metric = Callback.score,
        .auto = null,
        .num_candidates = 1,
        .max_bootstrapped_demos = 0,
        .max_labeled_demos = 0,
        .concurrent_evaluation = .{
            .ptr = &memory,
            .program_type = @import("runtime_value.zig").typeIdentity(struct {}),
            .run = Callback.evaluate,
        },
    };
    const result = optimizer.compile(
        Program,
        arena,
        .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = task.lm() },
        .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = prompt.lm() },
        .{},
        null,
        &rows,
        .{
            .valset = &rows,
            .num_trials = 1,
            .minibatch = false,
            .program_aware_proposer = false,
            .data_aware_proposer = false,
            .tip_aware_proposer = false,
            .fewshot_aware_proposer = false,
        },
    );
    try testing.expectError(error.ProgramTypeMismatch, result);
    try testing.expectEqual(@as(usize, 0), task.index);
    try testing.expectEqual(@as(usize, 0), prompt.index);
}

test "MIPRO demo candidates observe teacher setting changes between examples" {
    const lm = @import("lm.zig");
    const P = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    const FirstTeacher = struct {
        settings: *optimize.TeacherSettings,
        next: lm.LM,
        scripted: lm.ScriptedLM,

        fn complete(ptr: *anyopaque, arena: Allocator, request: lm.Request) lm.Error![]const lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.settings.lm = self.next;
            return self.scripted.lm().completeResponses(arena, request);
        }
    };
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const rows = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"one\",\"answer\":1}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\":\"two\",\"answer\":2}", &.{"question"}),
    };
    var task: lm.ScriptedLM = .{ .responses = &.{} };
    var second: lm.ScriptedLM = .{ .responses = &.{"<answer>22</answer>"} };
    var settings: optimize.TeacherSettings = .{};
    var first: FirstTeacher = .{
        .settings = &settings,
        .next = second.lm(),
        .scripted = .{ .responses = &.{"<answer>11</answer>"} },
    };
    settings.lm = .{ .ptr = &first, .vtable = &.{ .complete = FirstTeacher.complete } };
    const optimizer: MIPROv2 = .{
        .metric = struct {
            fn pass(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
                return 1;
            }
        }.pass,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 1,
        .teacher_settings_ref = &settings,
    };
    var rng: pyrandom.Random = .init(0);
    const sets = try optimizer.demoCandidates(P, arena, .{
        .workspace = @import("workspace.zig").forTests(),
        .arena = arena,
        .lm = task.lm(),
    }, .{}, null, &rows, 3, false, &rng);
    try testing.expectEqual(@as(i64, 11), sets[0][2][0].get("answer").?.integer);
    try testing.expectEqual(@as(i64, 22), sets[0][2][1].get("answer").?.integer);
}

test "MIPRO inherits the compile context error limit for demos and trial evaluation" {
    const P = predict.Predict(signature.Signature(struct {
        pub const instructions = "base";
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    const FailingMetric = struct {
        fn score(_: Allocator, example: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) @import("value.zig").ParseError!f64 {
            if (std.mem.eql(u8, example.record.get("question").?.string, "fail"))
                return error.ParseValue;
            return 1;
        }
    };
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const lm = @import("lm.zig");
    const rows = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"fail\",\"answer\":1}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\":\"pass\",\"answer\":1}", &.{"question"}),
    };
    for ([_]bool{ false, true }) |bootstrap| {
        for ([_]?usize{ null, 2 }) |explicit| {
            var task: lm.ScriptedLM = .{ .responses = &(@as([8][]const u8, @splat("<answer>1</answer>"))) };
            var prompt: lm.ScriptedLM = .{ .responses = &.{
                "<proposed_instruction>alpha</proposed_instruction>",
                "<proposed_instruction>beta</proposed_instruction>",
                "<proposed_instruction>gamma</proposed_instruction>",
            } };
            const optimizer: MIPROv2 = .{
                .metric = FailingMetric.score,
                .auto = null,
                .num_candidates = 3,
                .max_bootstrapped_demos = @intFromBool(bootstrap),
                .max_labeled_demos = @intFromBool(bootstrap),
                .max_errors = explicit,
                .teacher_settings = .{ .max_errors = 9 },
            };
            const compiled = optimizer.compile(
                P,
                arena,
                .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = task.lm(), .max_errors = 1 },
                .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = prompt.lm() },
                .{},
                @as(P, .{}),
                if (bootstrap) &rows else rows[1..],
                .{
                    .valset = if (bootstrap) rows[1..] else &rows,
                    .num_trials = 2,
                    .minibatch = false,
                    .program_aware_proposer = false,
                    .data_aware_proposer = false,
                    .tip_aware_proposer = false,
                    .fewshot_aware_proposer = false,
                },
            );
            if (bootstrap and explicit == null) {
                try testing.expectError(error.ParseValue, compiled);
                try testing.expectEqual(@as(usize, 1), task.index);
                continue;
            }
            const result = try compiled;
            const score: f64 = if (bootstrap) 100 else if (explicit != null) 50 else 0;
            try testing.expectEqual(score, result.score);
            for (result.trials) |trial| try testing.expectEqual(score, trial.score);
            try testing.expectEqual(@as(usize, if (!bootstrap and explicit != null) 8 else 5), task.index);
        }
    }
}

test "MIPRO refuses empty candidate spaces without indexing or sampling them" {
    const P = predict.Predict(@import("signature.zig").Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    const metric = struct {
        fn score(
            _: Allocator,
            _: *const Example,
            _: ?*const optimize.RecordPrediction,
            _: ?*const predict.Trace,
        ) value.ParseError!f64 {
            return 1;
        }
    }.score;
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const row = try Example.fromJson(arena, "{\"question\":\"test\",\"answer\":1}", &.{"question"});
    const student: P = .{};
    for ([_]struct { instructions: usize, demos: usize, aware: bool }{
        .{ .instructions = 0, .demos = 0, .aware = true },
        .{ .instructions = 1, .demos = 0, .aware = true },
        .{ .instructions = 1, .demos = 0, .aware = false },
    }) |case| {
        var task: @import("lm.zig").ScriptedLM = .{ .responses = &.{"<answer>\n1\n</answer>"} };
        var prompt: @import("lm.zig").ScriptedLM = .{
            .responses = &.{"<proposed_instruction>\nbetter\n</proposed_instruction>"},
        };
        const optimizer: MIPROv2 = .{
            .metric = metric,
            .auto = null,
            .num_candidates = 1,
            .num_instruct_candidates = case.instructions,
            .num_fewshot_candidates = case.demos,
        };
        try testing.expectError(error.InvalidOptions, optimizer.compile(
            P,
            arena,
            .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = task.lm() },
            .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = prompt.lm() },
            student,
            student,
            &.{row},
            .{
                .valset = &.{row},
                .num_trials = 1,
                .minibatch = false,
                .data_aware_proposer = false,
                .fewshot_aware_proposer = case.aware,
            },
        ));
    }
}

test "MIPRO snapshot serialization failure preserves the previous artifact" {
    const P = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const destination: MIPROv2.LogDir = .{
        .io = testing.io,
        .path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena),
    };

    var program: P = .{};
    program.state.instructions = "original";
    const artifact: MIPROv2.Artifact = .{ .full_eval = 4 };
    const path = (try MIPROv2.saveCandidate(destination, arena, arena, &program, artifact)).?;
    var model: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    program.state.lm = model.lm();
    try testing.expectError(
        error.UnsupportedLMState,
        MIPROv2.saveCandidate(destination, arena, arena, &program, artifact),
    );

    const saved = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, arena, .limited(4096));
    var restored: P = .{};
    try program_state.load(arena, &restored, saved, .{});
    try testing.expectEqualStrings("original", restored.state.instructions);
}

test "MIPRO distinguishes failed NaN objectives from explicit Optuna observations" {
    const P = predict.Predict(signature.Signature(struct {
        pub const instructions = "base";
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    }));
    const metric = struct {
        fn score(_: Allocator, _: *const Example, pred: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            const n = pred.?.get("answer").?.integer;
            return if (n < 0) std.math.nan(f64) else @as(f64, @floatFromInt(n)) / 4;
        }
    }.score;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const row = try Example.fromJson(arena, "{\"question\":\"q\"}", &.{"question"});
    const rows = [_]Example{ row, row };
    const optimizer: MIPROv2 = .{ .metric = metric };
    const nan = "<answer>-1</answer>";
    const low = "<answer>1</answer>";
    const mid = "<answer>2</answer>";
    const high = "<answer>3</answer>";
    const perfect = "<answer>4</answer>";
    // Actual DSPy 3.5 / Optuna 4.5 _optimize_prompt_parameters, seed 9.
    const Scenario = enum { startup, promotion, invalid_baseline, invalid_promotion };
    for ([_]Scenario{ .startup, .promotion, .invalid_baseline, .invalid_promotion }) |case| {
        const responses: []const []const u8 = switch (case) {
            .startup => &.{ low, nan, nan, nan, nan, nan, nan, nan, nan, nan, mid, high, perfect },
            .promotion => &.{ low, low, nan, high, mid, mid, low, perfect, perfect },
            .invalid_baseline => &.{nan},
            .invalid_promotion => &.{ low, low, high, nan, nan },
        };
        var task: @import("lm.zig").ScriptedLM = .{ .responses = responses };
        const minibatch = case == .promotion or case == .invalid_promotion;
        var rng = pyrandom.Random.init(9);
        const run = optimizer.search(P, arena, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = task.lm() }, .{}, &.{&.{ "base", "beta", "gamma" }}, &.{}, rows[0..if (minibatch) @as(usize, 2) else 1], switch (case) {
            .startup => 12,
            .promotion => 3,
            else => 1,
        }, minibatch, .{ .minibatch_size = 1, .minibatch_full_eval_steps = if (case == .invalid_promotion) 1 else 2 }, 9, &rng);
        try testing.expectEqual(responses.len, task.index);
        if (case == .invalid_baseline or case == .invalid_promotion) {
            try testing.expectError(error.NaNStudyObservation, run);
            continue;
        }
        const result = try run;
        try testing.expectEqualStrings("gamma", result.program.state.instructions);
        try testing.expectEqual(@as(f64, 100), result.score);
        if (case == .startup) {
            const choices = [_]usize{ 1, 2, 0, 2, 0, 1, 1, 1, 2, 2, 2, 2 };
            try testing.expectEqual(choices.len + 1, result.trials.len);
            for (result.trials[1..], choices, 2..) |trial, choice, n| {
                try testing.expectEqual(n, trial.number);
                try testing.expectEqual(choice, trial.params.?[0]);
                try testing.expectEqual(n <= 10, std.math.isNan(trial.score));
            }
        } else {
            const tracked = result.tracked_programs.?.by_trial;
            for (result.trials, tracked, [_][]const u8{ "base", "beta", "gamma", "beta", "base", "gamma" }, [_]usize{ 2, 3, 4, 6, 7, 9 }, 1..) |trial, prog, instruction, calls, n| {
                try testing.expectEqual(n, trial.number);
                try testing.expectEqualStrings(instruction, prog.state.instructions);
                try testing.expectEqual(calls, trial.total_eval_calls_so_far);
            }
            try testing.expect(std.math.isNan(result.trials[1].score));
        }
    }
}

test "MIPRO ranks full-set minibatch trials as full candidates and preserves ties" {
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const programs = [_]u8{ 10, 20, 30, 40, 50, 60 };
    const logs = [_]MIPROv2.TrialLog{
        .{ .number = 1, .params = null, .score = 40, .full_eval = true },
        .{ .number = 2, .params = &.{1}, .score = 90, .full_eval = false, .minibatch_log = true },
        .{ .number = 3, .params = &.{2}, .score = 80, .full_eval = true, .minibatch_log = true },
        .{ .number = 4, .params = null, .score = 80, .full_eval = true },
        .{ .number = 5, .params = &.{3}, .score = 90, .full_eval = false, .minibatch_log = true },
        .{ .number = 6, .params = null, .score = 100, .full_eval = true },
    };
    const ranked = try MIPROv2.rankPrograms(u8, backing.allocator(), &programs, &logs, 4);
    try testing.expectEqualSlices(usize, &.{ 5, 2, 3, 0 }, ranked.full);
    try testing.expectEqualSlices(usize, &.{ 1, 4 }, ranked.minibatch);
}

test "stripPrefix matches dspy.propose.utils.strip_prefix" {
    try testing.expectEqualStrings("Answer with ISO dates.", stripPrefix("Proposed Instruction: Answer with ISO dates."));
    try testing.expectEqualStrings("**New Instruction**: \"Do it", stripPrefix("**New Instruction**: \"Do it\""));
    try testing.expectEqualStrings("hi\" there", stripPrefix("  *Instruction:   \"hi\" there\""));
    try testing.expectEqualStrings("Hazlo", stripPrefix("Instrucción propuesta: Hazlo"));
    try testing.expectEqualStrings("go", stripPrefix("*\u{3000}Ⅻ\u{a0}Schritt: go"));
    try testing.expectEqualStrings("a—b: c", stripPrefix("a—b: c"));
    try testing.expectEqualStrings("one two three four five six: x", stripPrefix("one two three four five six: x"));
    try testing.expectEqualStrings("x", stripPrefix("one two three four five: x"));
    try testing.expectEqualStrings("No label here.", stripPrefix("No label here."));
    try testing.expectEqualStrings("quoted", stripPrefix("\"quoted\""));
}

test "examplesRepr matches order_input_keys_in_string(repr(examples))" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const examples = [_]Example{
        try .fromJson(arena, "{\"question\": \"When?\", \"answer\": \"1969-07-20\", \"n\": 3, \"ok\": true, \"l\": [\"a\", \"b'\"]}", &.{ "question", "n" }),
        try .fromJson(arena, "{\"question\": \"it's\"}", &.{}),
    };
    try testing.expectEqualStrings(
        \\[Example({'question': 'When?', 'answer': '1969-07-20', 'n': 3, 'ok': True, 'l': ['a', "b'"]}) (input_keys={'n', 'question'}), Example({'question': "it's"}) (input_keys=set())]
    , try examplesRepr(arena, &examples));
}
