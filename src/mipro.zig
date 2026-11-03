//! `dspy.MIPROv2` (DSPy 3.4.1): bootstrap few-shot candidate sets, propose
//! instructions grounded in a dataset summary, the predictor's demos and a
//! random prompting tip, then search instruction × demo-set combinations with
//! Optuna's TPE sampler (src/tpe.zig), on minibatches with periodic full
//! evaluations. One CPython `random.Random(seed)` stream is shared by every
//! step, in DSPy's order, so the same LM answers give the same program.
//!
//! The program-aware proposer needs the program's code, which DSPy reads with
//! `inspect.getsource`; DSZig takes it as `Options.program_code` and runs
//! without it, like `program_aware_proposer=False`, when none is given.
//! Not supported: `teacher_settings`, `track_stats` program copies and
//! `log_dir`.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const signature = @import("signature.zig");
const tpe = @import("tpe.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const Field = signature.Field;
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
/// `\w` and `\s` are matched over ASCII only.
pub fn stripPrefix(text: []const u8) []const u8 {
    var i: usize = 0;
    while (i < text.len and (text[i] == '*' or isSpace(text[i]))) i += 1;
    var words: usize = 0;
    const rest = while (true) {
        const start = i;
        while (i < text.len and isWord(text[i])) i += 1;
        if (i == start) break text;
        words += 1;
        if (i < text.len and text[i] == ':') {
            i += 1;
            while (i < text.len and isSpace(text[i])) i += 1;
            break text[i..];
        }
        if (words == 5 or i == text.len or !isSpace(text[i])) break text;
        while (i < text.len and isSpace(text[i])) i += 1;
    };
    return std.mem.trim(u8, rest, "\"");
}

fn isWord(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '\'' or c == '-';
}

fn isSpace(c: u8) bool {
    return std.mem.indexOfScalar(u8, value.py_whitespace, c) != null;
}

/// `order_input_keys_in_string(repr(examples))` for a list of `dspy.Example`.
pub fn examplesRepr(arena: Allocator, examples: []const Example) Allocator.Error![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    writeExamplesRepr(arena, &aw.writer, examples) catch |err| switch (err) {
        error.OutOfMemory, error.WriteFailed => return error.OutOfMemory,
        error.JsonTooDeep => unreachable, // Examples come from parsed JSON.
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
    assert(batch_size > 0);
    var ctx = prompt;
    ctx.lm_options.temperature = 1.0;
    const first = try (predict.Predict(DatasetDescriptor){}).forward(ctx, .{
        .examples = try examplesRepr(ctx.arena, trainset[0..@min(trainset.len, batch_size)]),
    });
    var observations: std.ArrayList(u8) = .empty;
    try observations.appendSlice(out, first.observations);
    var skips: usize = 0;
    var calls: usize = 0;
    var b = batch_size;
    while (b < trainset.len) : (b += batch_size) {
        calls += 1;
        if (calls >= 10) break;
        const more = (predict.Predict(DatasetDescriptorWithPriorObservations){}).forward(ctx, .{
            .examples = try examplesRepr(ctx.arena, trainset[b..@min(trainset.len, b + batch_size)]),
            .prior_observations = observations.items,
        }) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => break, // DSPy keeps the observations gathered so far.
        };
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
    max_errors: usize = 10,
    seed: u32 = 9,
    init_temperature: f64 = 1.0,

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

    /// `compile(...)` keyword arguments. `seed` 0 means "use `self.seed`",
    /// as DSPy's `seed or self.seed` does.
    pub const Options = struct {
        valset: ?[]const Example = null,
        num_trials: ?usize = null,
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
    };

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
        };
    }

    pub const Error = error{ InvalidOptions, EmptyTrainset, MinibatchTooLarge, NoCandidate };

    /// Everything the result references lives in `out`. `prompt` is the
    /// `prompt_model` context (defaults to `ctx`, the task model).
    pub fn compile(
        self: MIPROv2,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        prompt: ?Context,
        student: P,
        teacher: ?P,
        trainset_in: []const Example,
        options: Options,
    ) !Result(P) {
        const zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0;
        if (self.auto == null and (self.num_candidates == null or options.num_trials == null)) return error.InvalidOptions;
        if (self.auto != null and (self.num_candidates != null or options.num_trials != null)) return error.InvalidOptions;
        assert(options.minibatch_full_eval_steps > 0);

        const seed = if (options.seed) |s| (if (s == 0) self.seed else s) else self.seed;
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
            n_instruct = self.num_candidates.?;
            n_fewshot = self.num_candidates.?;
            num_trials = options.num_trials.?;
        }
        if (minibatch and options.minibatch_size > valset.len) return error.MinibatchTooLarge;

        const demo_sets = try self.demoCandidates(P, out, gpa, ctx, program, teacher, trainset, n_fewshot, zeroshot, &rng);
        const instructions = try self.proposeInstructions(P, out, prompt orelse ctx, &program, trainset, demo_sets, n_instruct, options, &rng);
        const demos: []const []const []const Record = if (zeroshot) &.{} else demo_sets;

        return self.search(P, out, gpa, ctx, program, instructions, demos, valset, num_trials, minibatch, options, seed, &rng);
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
        gpa: Allocator,
        ctx: Context,
        student: P,
        teacher: ?P,
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
                try predict.resetDemos(out, &candidate);
            } else if (seed == -2 and max_labeled > 0) {
                candidate = try (optimize.LabeledFewShot{ .k = max_labeled }).compile(P, out, student, trainset);
            } else {
                const shuffled = try out.dupe(Example, trainset);
                var size = max_bootstrapped;
                if (seed != -1) {
                    rng.shuffle(Example, shuffled);
                    if (max_bootstrapped < 1) return error.InvalidOptions;
                    size = rng.randint(1, max_bootstrapped);
                }
                const bootstrap: optimize.BootstrapFewShot = .{
                    .metric = self.metric,
                    .metric_threshold = self.metric_threshold,
                    .max_bootstrapped_demos = size,
                    .max_labeled_demos = max_labeled,
                    .max_errors = self.max_errors,
                };
                candidate = (try bootstrap.compile(P, out, gpa, ctx, student, teacher, shuffled)).program;
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
        const result = try out.alloc([]const []const u8, preds.len);
        for (preds, result, 0..) |p, *list, pred_i| {
            const state = p.param.predict;
            const items = try out.alloc([]const u8, per_pred);
            for (items, 0..) |*item, set_i| {
                var tip: []const u8 = "";
                if (options.tip_aware_proposer) tip = tips[rng.randbelow(tips.len)];
                var ctx = prompt;
                ctx.lm_options.rollout_id = @intCast(rng.randint(0, 1_000_000_000));
                ctx.lm_options.temperature = self.init_temperature;
                const demos_text = if (options.fewshot_aware_proposer and demo_sets.len > 0)
                    try taskDemos(ctx.arena, state, demo_sets[pred_i], set_i, bootstrapped_fewshot_examples_in_context)
                else
                    no_task_demos;
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
        gpa: Allocator,
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
        const adjusted = if (minibatch) num_trials + num_trials / steps + 1 + @intFromBool(num_trials % steps != 0) else num_trials;

        const default_score = try self.score(P, out, gpa, ctx, &program, valset, valset.len, rng);
        var history: std.ArrayList(tpe.Trial) = .empty;
        try history.append(out, .{ .params = try zeros(out, params.items.len), .value = default_score });
        var logs: std.ArrayList(TrialLog) = .empty;
        try logs.append(out, .{ .number = 1, .params = null, .score = default_score, .full_eval = true });
        var full_scores: std.ArrayList(f64) = .empty;
        try full_scores.append(out, default_score);
        var best = program;
        var best_score = default_score;

        const Combo = struct { params: []const usize, total: f64, count: usize, program: P, evaluated: bool };
        var combos: std.ArrayList(Combo) = .empty;

        var sampler: tpe.Sampler = .init(seed);
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();

        for (0..num_trials) |_| {
            const trial_num = history.items.len + 1;
            const chosen = try out.alloc(usize, params.items.len);
            _ = scratch.reset(.retain_capacity);
            try sampler.sample(scratch.allocator(), params.items, history.items, chosen);

            var candidate = program;
            for (try predict.namedPredictors(out, &candidate), 0..) |p, i| {
                const stride: usize = if (demos.len > 0) 2 else 1;
                p.param.predict.instructions = instructions[i][chosen[i * stride]];
                if (demos.len > 0) p.param.predict.demos = demos[i][chosen[i * stride + 1]];
            }
            const batch = if (minibatch) options.minibatch_size else valset.len;
            const s = try self.score(P, out, gpa, ctx, &candidate, valset, batch, rng);
            if (!minibatch and s > best_score) {
                best_score = s;
                best = candidate;
            }
            const full = batch >= valset.len;
            if (full) try full_scores.append(out, s);
            try logs.append(out, .{ .number = trial_num, .params = chosen, .score = s, .full_eval = full });
            try history.append(out, .{ .params = chosen, .value = s });
            const combo = for (combos.items) |*c| {
                if (std.mem.eql(usize, c.params, chosen)) break c;
            } else blk: {
                try combos.append(out, .{ .params = chosen, .total = 0, .count = 0, .program = candidate, .evaluated = false });
                break :blk &combos.items[combos.items.len - 1];
            };
            combo.total += s;
            combo.count += 1;

            if (minibatch and (trial_num % (steps + 1) == 0 or trial_num == adjusted - 1)) {
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
                const full_score = try self.score(P, out, gpa, ctx, &target.program, valset, valset.len, rng);
                try full_scores.append(out, full_score);
                try history.append(out, .{ .params = target.params, .value = full_score });
                try logs.append(out, .{ .number = trial_num + 1, .params = null, .score = full_score, .full_eval = true });
                target.evaluated = true;
                if (full_score > best_score) {
                    best_score = full_score;
                    best = target.program;
                }
            }
        }
        std.sort.insertion(f64, full_scores.items, {}, std.sort.desc(f64));
        return .{
            .program = best,
            .score = best_score,
            .instruction_candidates = instructions,
            .demo_candidates = demos,
            .trials = logs.items,
            .candidates = full_scores.items,
        };
    }

    /// `eval_candidate_program`: a minibatch drawn with `rng` unless
    /// `batch_size` covers `valset`; any failure scores 0.0, as in DSPy.
    fn score(self: MIPROv2, comptime P: type, out: Allocator, gpa: Allocator, ctx: Context, program: *const P, valset: []const Example, batch_size: usize, rng: *pyrandom.Random) Allocator.Error!f64 {
        const devset = if (batch_size >= valset.len) valset else try createMinibatch(out, rng, valset, batch_size);
        const result = optimize.evaluate(P, gpa, ctx, program, devset, self.metric, self.max_errors) catch |err| switch (err) {
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

test "stripPrefix matches dspy.propose.utils.strip_prefix" {
    try testing.expectEqualStrings("Answer with ISO dates.", stripPrefix("Proposed Instruction: Answer with ISO dates."));
    try testing.expectEqualStrings("**New Instruction**: \"Do it", stripPrefix("**New Instruction**: \"Do it\""));
    try testing.expectEqualStrings("hi\" there", stripPrefix("  *Instruction:   \"hi\" there\""));
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
