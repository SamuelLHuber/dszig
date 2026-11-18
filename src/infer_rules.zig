//! Native `dspy.teleprompt.infer_rules` (DSPy 3.5.0).
//!
//! Programs follow optimize.zig's value-program convention: `Input`, `Output`,
//! `compiled` and `forward(*const P, Context, Input) !Output`. Returned reachable
//! predictor instructions, descriptions, prefixes and demos belong to `out`
//! (normally an arena); attempt/candidate scratch belongs to `gpa` and is freed.
//! Other user-program members retain their usual value-copy/borrowed lifetimes.
//! Compile borrows mutable student/teacher pointers: the source mutates shared
//! Signature classes even on the caller and earlier winning candidates. Native
//! aliases are propagated across that reachable graph by declared-field identity.
//! Other live instances outside that graph cannot share instructions without a
//! general shared-signature storage API; this is an explicit parity boundary.
//!
//! DSPy `num_threads` maps to `concurrent_evaluation`, a bound startup-created
//! `parallel.Evaluator`; null uses the native sequential evaluator. A missing
//! metric fails every item either way, so that source path stays sequential.
//! Generation LM errors erase provider exception types. Recoverable failures
//! therefore require an explicit classifier backed by the provider's real error
//! metadata; an opaque LMFailure is never assumed to be a context-window error.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyrandom = @import("pyrandom.zig");
const pyjson = @import("pyjson.zig");
const signature = @import("signature.zig");
const unicode = @import("unicode.zig");
const value = @import("value.zig");
const Context = predict.Context;
const Example = optimize.Example;
const Record = value.Record;

pub const rule_instruction_suffix = "\n\nPlease adhere to the following rules when making your prediction:\n";
pub const context_window_failure_message = "Failed to generate natural language rules since a single example couldn't fit in the model's context window.";

pub const CustomRulesInduction = signature.Signature(struct {
    pub const instructions = "Given a set of examples, extract a list of 10 concise and non-redundant natural language rules that provide clear guidance for performing the task. All rules should be actionable for a well-specified scope of examples of this general kind of task.";
    pub const Input = struct { examples_text: []const u8 };
    pub const Output = struct { natural_language_rules: []const u8 };
    pub const descriptions = .{
        .examples_text = "Text containing examples",
        .natural_language_rules = "Induced natural language rules",
    };
});

/// The three accepted source conditions, classified from actual provider/type
/// information, not by guessing from a Zig error name. `.other` is not retried.
/// Classify the thrown error, not its underlying cause: pinned DSPy wraps engine
/// ValueError as LMUnexpectedError, which is NOT an accepted ValueError.
pub const FailureClass = enum { other, value_error, bad_request_error, context_window_exceeded };

pub const FailureClassifier = struct {
    /// Borrowed for the lifetime of the optimizer's calls.
    ptr: ?*anyopaque = null,
    classify: *const fn (?*anyopaque, anyerror) FailureClass,
};

/// Runtime-num_rules CoT. Its private Random(0) stream persists across forward
/// calls, failed attempts, and repeated optimizer compilations. Metadata borrows
/// the `init` output arena; returned text borrows `ctx.arena`. Neither scratch
/// allocators nor request-arena strings are retained in the program.
pub const RulesInductionProgram = struct {
    pub const Input = CustomRulesInduction.Input;
    pub const Output = []const u8;

    rules_induction: predict.ChainOfThought(CustomRulesInduction) = .{},
    num_rules: i64 = 10,
    teacher_settings: optimize.TeacherSettings = .{},
    /// Optional captured settings object. Unlike value settings, pointee changes
    /// are observable; rebinding the optimizer's pointer does not retarget this.
    teacher_settings_ref: ?*const optimize.TeacherSettings = null,
    rng: pyrandom.Random = blk: {
        @setEvalBranchQuota(10_000);
        break :blk pyrandom.Random.init(0);
    },
    compiled: bool = false,

    pub fn init(out: Allocator, num_rules: i64, teacher_settings: optimize.TeacherSettings) Allocator.Error!RulesInductionProgram {
        var program: RulesInductionProgram = .{ .num_rules = num_rules, .teacher_settings = teacher_settings };
        try program.rules_induction.predict.state.forkSignatureClass(out);
        program.rules_induction.predict.state.instructions = try program.instructionText(out);
        return program;
    }

    pub fn instructionText(self: *const RulesInductionProgram, out: Allocator) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(out, "Given a set of examples, extract a list of {d} concise and non-redundant natural language rules that provide clear guidance for performing the task. All rules should be actionable for a well-specified scope of examples of this general kind of task.", .{self.num_rules});
    }

    pub fn forward(self: *RulesInductionProgram, ctx: Context, input: Input) predict.Error!Output {
        var c = ctx;
        const settings = if (self.teacher_settings_ref) |bound| bound.* else self.teacher_settings;
        settings.apply(&c);
        // Python randint(a, b) includes b; randbelow(10**9 + 1) is exact.
        c.lm_options.rollout_id = .{ .integer = @intCast(self.rng.randbelow(1_000_000_001)) };
        c.lm_options.temperature = 1.0;
        const result = try self.rules_induction.forward(c, input);
        return stripRules(result.natural_language_rules);
    }
};

pub const InferRules = struct {
    pub const Options = struct {
        metric: ?optimize.Metric = null,
        num_candidates: isize = 10,
        num_rules: i64 = 10,
        /// DSPy `num_threads`; null evaluates sequentially.
        concurrent_evaluation: ?optimize.ConcurrentEvaluation = null,
        teacher_settings: optimize.TeacherSettings = .{},
        /// Bind an actual mutable settings object to model Python dictionary
        /// aliasing. The pointee must outlive this optimizer. Value settings are
        /// captured by value instead; they do not pretend to be shared dicts.
        /// Rules dereference this object for every call; the existing bootstrap
        /// API receives its current value once at compile entry.
        teacher_settings_ref: ?*const optimize.TeacherSettings = null,
        metric_threshold: ?f64 = null,
        max_bootstrapped_demos: usize = 4,
        max_labeled_demos: usize = 16,
        max_rounds: usize = 1,
        max_errors: usize = 10,
        failure_classifier: ?FailureClassifier = null,
    };

    metric: ?optimize.Metric,
    num_candidates: isize,
    /// Constructor input, not a live reconfiguration of the rules program.
    num_rules: i64,
    concurrent_evaluation: ?optimize.ConcurrentEvaluation,
    teacher_settings: optimize.TeacherSettings,
    teacher_settings_ref: ?*const optimize.TeacherSettings,
    metric_threshold: ?f64,
    max_bootstrapped_demos: usize,
    max_labeled_demos: usize,
    max_rounds: usize,
    max_errors: usize,
    failure_classifier: ?FailureClassifier,
    /// Cause of the source-equivalent assertion/context-window failure.
    last_induction_failure: ?anyerror = null,
    /// Required constructor state; metadata borrows init's output arena for the
    /// optimizer's lifetime, not the latest compile/induction output arena.
    rules_induction_program: RulesInductionProgram,
    student_storage: ?*anyopaque = null,
    student_type: ?*u8 = null,

    /// Eagerly construct the one persistent source RulesInductionProgram.
    /// Reassigning num_rules or the optimizer's settings binding does not rebuild
    /// its signature, overwrite caller edits, retarget its settings, or reseed it.
    pub fn init(out: Allocator, options: Options) Allocator.Error!InferRules {
        var rules = try RulesInductionProgram.init(out, options.num_rules, options.teacher_settings);
        rules.teacher_settings_ref = options.teacher_settings_ref;
        return .{
            .metric = options.metric,
            .num_candidates = options.num_candidates,
            .num_rules = options.num_rules,
            .concurrent_evaluation = options.concurrent_evaluation,
            .teacher_settings = options.teacher_settings,
            .teacher_settings_ref = options.teacher_settings_ref,
            .metric_threshold = options.metric_threshold,
            .max_bootstrapped_demos = options.max_bootstrapped_demos,
            .max_labeled_demos = options.max_labeled_demos,
            .max_rounds = options.max_rounds,
            .max_errors = options.max_errors,
            .failure_classifier = options.failure_classifier,
            .rules_induction_program = rules,
        };
    }

    pub const Candidate = struct {
        instructions: []const []const u8,
        score: f64,
    };

    pub fn Result(comptime P: type) type {
        return struct {
            /// Genuine null when range(num_candidates) is empty or no score
            /// strictly exceeds negative infinity (including all NaN scores).
            program: ?P,
            /// Bootstrapped self.student, including final source alias mutations.
            original_program: P,
            original_instructions: []const []const u8,
            best_score: f64,
            best_candidate: ?usize,
            candidates: []const Candidate,
            bootstrap_report: optimize.BootstrapFewShot.Report,
        };
    }

    /// The latest out-owned bootstrapped student, also after later induction or
    /// evaluation failure. Wrong program types return null. The optimizer borrows
    /// `out` until a new compile clears/replaces this storage. Failures inside
    /// BootstrapFewShot itself cannot expose its prepared student: that shared
    /// API returns no partial state, so this accessor remains null in that case.
    pub fn student(self: *const InferRules, comptime P: type) ?*P {
        if (self.student_type != programIdentity(P)) return null;
        const storage = self.student_storage orelse return null;
        return @ptrCast(@alignCast(storage));
    }

    /// Bootstrap exactly once. Omitted valset splits at floor(len / 2);
    /// explicit empty valset is kept and fails only if a candidate is evaluated.
    /// Original instruction TEXT snapshots survive candidate resets. Signature
    /// state deliberately aliases: later candidates mutate earlier winners and
    /// the caller, just like the pinned Python source. Caller instructions then
    /// borrow `out`; callers must retain it until those instructions are replaced.
    pub fn compile(
        self: *InferRules,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        student_in: *P,
        teacher_in: ?*P,
        trainset_in: []const Example,
        valset_in: ?[]const Example,
    ) !Result(P) {
        self.student_storage = null;
        self.student_type = null;
        const split = trainset_in.len / 2;
        const trainset = if (valset_in == null) trainset_in[0..split] else trainset_in;
        const valset = valset_in orelse trainset_in[split..];
        var bootstrap_arena: std.heap.ArenaAllocator = .init(gpa);
        defer bootstrap_arena.deinit();
        const bootstrap_out = bootstrap_arena.allocator();
        const bootstrap: optimize.BootstrapFewShot = .{
            .metric = self.metric,
            .metric_threshold = self.metric_threshold,
            .max_bootstrapped_demos = self.max_bootstrapped_demos,
            .max_labeled_demos = self.max_labeled_demos,
            .max_rounds = self.max_rounds,
            .max_errors = self.max_errors,
            .teacher_settings = if (self.teacher_settings_ref) |bound| bound.* else self.teacher_settings,
        };
        const bootstrapped = try bootstrap.compile(P, bootstrap_out, gpa, ctx, student_in.*, if (teacher_in) |t| t.* else null, trainset);
        const original = try out.create(P);
        original.* = try ownPredictorState(P, out, bootstrap_out, bootstrapped.program);
        self.student_storage = original;
        self.student_type = programIdentity(P);
        const original_predictors = try predict.namedPredictors(bootstrap_out, original);
        const instructions = try out.alloc([]const u8, original_predictors.len);
        for (instructions, original_predictors) |*text, p| text.* = p.param.predict.instructions;

        var best_score = -std.math.inf(f64);
        var best_program = original.*;
        var best_candidate: ?usize = null;
        const best_predictors = try predict.namedPredictors(bootstrap_out, &best_program);
        const student_predictors = try predict.namedPredictors(bootstrap_out, student_in);
        const teacher_predictors: []const predict.NamedParam = if (teacher_in) |t| try predict.namedPredictors(bootstrap_out, t) else &.{};
        const count: usize = if (self.num_candidates > 0) @intCast(self.num_candidates) else 0;
        const candidates = try out.alloc(Candidate, count);
        for (candidates, 0..) |*entry, candidate_index| {
            var candidate_arena: std.heap.ArenaAllocator = .init(gpa);
            defer candidate_arena.deinit();
            const arena = candidate_arena.allocator();
            var candidate = original.*;
            const predictors = try predict.namedPredictors(arena, &candidate);
            const targets: AliasTargets = .{
                .candidate = predictors,
                .original = original_predictors,
                .best = best_predictors,
                .student = student_predictors,
                .teacher = teacher_predictors,
            };
            for (predictors, instructions) |p, text| setAliasedInstruction(targets, p.param.predict, text);
            for (predictors, instructions) |p, text| {
                const rules = try self.induceNaturalLanguageRules(out, gpa, ctx, p.param.predict, trainset);
                setAliasedInstruction(targets, p.param.predict, text);
                const updated = try std.fmt.allocPrint(out, "{s}{s}{s}", .{ text, rule_instruction_suffix, rules });
                setAliasedInstruction(targets, p.param.predict, updated);
            }
            const score = try self.evaluateProgram(P, gpa, ctx, &candidate, valset);
            const saved = try out.alloc([]const u8, predictors.len);
            for (saved, predictors) |*text, p| text.* = p.param.predict.instructions;
            entry.* = .{ .instructions = saved, .score = score };
            if (score > best_score) {
                best_program = candidate;
                best_score = score;
                best_candidate = candidate_index;
            }
        }
        return .{
            .program = if (best_candidate != null) best_program else null,
            .original_program = original.*,
            .original_instructions = instructions,
            .best_score = best_score,
            .best_candidate = best_candidate,
            .candidates = candidates,
            .bootstrap_report = bootstrapped.report,
        };
    }

    pub fn induceNaturalLanguageRules(
        self: *InferRules,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        predictor: *const predict.PredictState,
        trainset: []const Example,
    ) ![]const u8 {
        var demo_arena: std.heap.ArenaAllocator = .init(gpa);
        defer demo_arena.deinit();
        const demos = try getPredictorDemos(demo_arena.allocator(), trainset, predictor);
        var count = demos.len;
        while (true) {
            var attempt: std.heap.ArenaAllocator = .init(gpa);
            defer attempt.deinit();
            var c = ctx;
            c.arena = attempt.allocator();
            // Formatting is outside the source try/except and cannot trigger
            // context-window reduction, even if a classifier accepts its error.
            const examples_text = try formatExamples(c.arena, demos[0..count], predictor);
            const rules = self.rules_induction_program.forward(c, .{ .examples_text = examples_text }) catch |err| {
                if (err == error.OutOfMemory) return error.OutOfMemory;
                self.last_induction_failure = err;
                const classifier = self.failure_classifier orelse return error.UnacceptedRulesInductionFailure;
                if (classifier.classify(classifier.ptr, err) == .other) return error.UnacceptedRulesInductionFailure;
                if (count <= 1) return error.RulesContextWindowExceeded;
                count -= 1;
                continue;
            };
            return out.dupe(u8, rules);
        }
    }

    pub fn evaluateProgram(
        self: *const InferRules,
        comptime P: type,
        gpa: Allocator,
        ctx: Context,
        program: *const P,
        dataset: []const Example,
    ) !f64 {
        // Evaluate checks this after induction, not at compile entry. Avoid the
        // existing sequential evaluator's nonempty-devset assertion.
        if (dataset.len == 0) return error.EmptyValidationSet;
        if (self.metric) |metric| {
            return (try optimize.evaluateWith(P, self.concurrent_evaluation, gpa, ctx, program, dataset, metric, self.max_errors)).score;
        }
        // Source allows metric=None at construction. It still runs the program
        // for each example, then fails calling the missing metric; failed items
        // score zero until the same max_errors limit is reached.
        const MissingMetric = struct {
            program: *const P,
            pub const Input = P.Input;
            pub const Output = P.Output;
            pub fn forward(wrapper: *const @This(), c: Context, input: Input) anyerror!Output {
                _ = try wrapper.program.forward(c, input);
                return error.MissingMetric;
            }
        };
        const wrapper: MissingMetric = .{ .program = program };
        return (try optimize.evaluate(MissingMetric, gpa, ctx, &wrapper, dataset, unusedMetric, self.max_errors)).score;
    }
};

/// Keep only signature fields, in the Example store's order. Values borrow the
/// input examples until formatted; this intentionally never uses bootstrap demos.
pub fn getPredictorDemos(out: Allocator, trainset: []const Example, predictor: *const predict.PredictState) Allocator.Error![]const Record {
    const demos = try out.alloc(Record, trainset.len);
    for (demos, trainset) |*demo, example| {
        demo.* = .empty;
        var it = example.record.iterator();
        while (it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, "dspy_")) continue;
            if (hasField(predictor.fields, entry.key_ptr.*)) try demo.put(out, entry.key_ptr.*, entry.value_ptr.*);
        }
    }
    return demos;
}

pub fn formatExamples(out: Allocator, demos: []const Record, predictor: *const predict.PredictState) (Allocator.Error || error{ JsonTooDeep, AdapterParse })![]const u8 {
    var text: Writer.Allocating = .init(out);
    for (demos) |demo| {
        text.writer.writeAll("Input Fields:\n") catch return error.OutOfMemory;
        try writeFields(&text.writer, demo, predictor.fields[0..predictor.input_count]);
        text.writer.writeAll("\n\n=========\nOutput Fields:\n") catch return error.OutOfMemory;
        try writeFields(&text.writer, demo, predictor.fields[predictor.input_count..]);
        text.writer.writeAll("\n\n") catch return error.OutOfMemory;
    }
    return text.written();
}

pub fn updateProgramInstructions(out: Allocator, predictor: *predict.PredictState, rules: []const u8) Allocator.Error!void {
    predictor.instructions = try std.fmt.allocPrint(out, "{s}{s}{s}", .{ predictor.instructions, rule_instruction_suffix, rules });
}

fn hasField(fields: []const signature.Field, key: []const u8) bool {
    for (fields) |field| if (std.mem.eql(u8, field.name, key)) return true;
    return false;
}

fn writeFields(writer: *Writer, demo: Record, fields: []const signature.Field) (Allocator.Error || error{ JsonTooDeep, AdapterParse })!void {
    var it = demo.iterator();
    var first = true;
    while (it.next()) |entry| {
        if (!hasField(fields, entry.key_ptr.*)) continue;
        if (!first) writer.writeByte('\n') catch return error.OutOfMemory;
        first = false;
        writer.print("{s}: ", .{entry.key_ptr.*}) catch return error.OutOfMemory;
        pyjson.writeStr(writer, entry.value_ptr.*) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep => error.JsonTooDeep,
            error.ParseValue => error.AdapterParse,
        };
    }
}

fn ownPredictorState(comptime P: type, out: Allocator, scratch: Allocator, input: P) Allocator.Error!P {
    var program = input;
    for (try predict.namedPredictors(scratch, &program)) |p| {
        const state = p.param.predict;
        state.instructions = try out.dupe(u8, state.instructions);
        const descs = try out.alloc([]const u8, state.descs.len);
        for (descs, state.descs) |*dst, src| dst.* = try out.dupe(u8, src);
        state.descs = descs;
        const prefixes = try out.alloc([]const u8, state.prefixes.len);
        for (prefixes, state.prefixes) |*dst, src| dst.* = try out.dupe(u8, src);
        state.prefixes = prefixes;
        const demos = try out.alloc(Record, state.demos.len);
        for (demos, state.demos) |*dst, *src| dst.* = try value.cloneRecord(out, src);
        state.demos = demos;
    }
    return program;
}

fn ProgramIdentity(comptime P: type) type {
    return struct {
        pub const Program = P;
        // Mutable storage gives each P a stable address which constant-data
        // coalescing cannot merge with another program type's identity.
        var token: u8 = 0;
    };
}

fn programIdentity(comptime P: type) *u8 {
    return &ProgramIdentity(P).token;
}

const AliasTargets = struct {
    candidate: []const predict.NamedParam,
    original: []const predict.NamedParam,
    best: []const predict.NamedParam,
    student: []const predict.NamedParam,
    teacher: []const predict.NamedParam,
};

fn setAliasedInstruction(targets: AliasTargets, identity: *const predict.PredictState, text: []const u8) void {
    inline for (.{ "candidate", "original", "best", "student", "teacher" }) |name| {
        for (@field(targets, name)) |target| {
            const state = target.param.predict;
            if (state.signature_class == identity.signature_class) state.instructions = text;
        }
    }
}

fn unusedMetric(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    unreachable;
}

/// CPython str.strip(), including non-ASCII whitespace. Invalid UTF-8 is kept,
/// never silently repaired. No allocation is needed to return the inner slice.
fn stripRules(text: []const u8) []const u8 {
    var begin: usize = 0;
    var end: usize = text.len;
    while (begin < end) {
        const n = std.unicode.utf8ByteSequenceLength(text[begin]) catch break;
        if (n > end - begin) break;
        const cp = std.unicode.utf8Decode(text[begin..][0..n]) catch break;
        if (!unicode.isSpace(cp)) break;
        begin += n;
    }
    while (end > begin) {
        var start = end - 1;
        while (start > begin and (text[start] & 0xc0) == 0x80) start -= 1;
        const n = std.unicode.utf8ByteSequenceLength(text[start]) catch break;
        if (n != end - start) break;
        const cp = std.unicode.utf8Decode(text[start..end]) catch break;
        if (!unicode.isSpace(cp)) break;
        end = start;
    }
    return text[begin..end];
}

const testing = std.testing;
const lm_mod = @import("lm.zig");

const TestFirst = signature.Signature(struct {
    pub const instructions = "Original first.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { draft: []const u8 };
});
const TestSecond = signature.Signature(struct {
    pub const instructions = "Original second.";
    pub const Input = struct { draft: []const u8, question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});
const TestProgram = struct {
    first: predict.Predict(TestFirst) = .{},
    second: predict.Predict(TestSecond) = .{},
    compiled: bool = false,
    pub const Input = TestFirst.Input;
    pub const Output = TestSecond.Output;
    pub fn forward(self: *const TestProgram, ctx: Context, input: Input) predict.Error!Output {
        const draft = try self.first.forward(ctx, input);
        return self.second.forward(ctx, .{ .draft = draft.draft, .question = input.question });
    }
};
const TestShared = signature.Signature(struct {
    pub const instructions = "Shared task.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});
const TestSharedProgram = struct {
    first: predict.Predict(TestShared) = .{},
    second: predict.Predict(TestShared) = .{},
    compiled: bool = false,
    pub const Input = TestShared.Input;
    pub const Output = TestShared.Output;
    pub fn forward(self: *const TestSharedProgram, ctx: Context, input: Input) predict.Error!Output {
        _ = try self.first.forward(ctx, input);
        return self.second.forward(ctx, input);
    }
};

const TestState = struct {
    history: lm_mod.History,
    teacher_calls: std.ArrayList(bool) = .empty,
    induction_count: usize = 0,
    good_rules: []const []const u8 = &.{ "rule-3", "rule-5" },
    failures: []const ?FailureClass = &.{},
    failure_index: usize = 0,
    last_failure: FailureClass = .other,

    fn classify(ptr: ?*anyopaque, err: anyerror) FailureClass {
        const self: *TestState = @ptrCast(@alignCast(ptr.?));
        return if (err == error.LMFailure) self.last_failure else .other;
    }
};

const TestLM = struct {
    state: *TestState,
    teacher: bool = false,

    fn lm(self: *TestLM) lm_mod.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }
    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
        const self: *TestLM = @ptrCast(@alignCast(ptr));
        const state = self.state;
        try state.history.record(request);
        try state.teacher_calls.append(state.history.arena, self.teacher);
        const system = request.messages[0].content.text;
        const text: []const u8 = if (isInduction(request)) blk: {
            if (state.failure_index < state.failures.len) {
                const failure = state.failures[state.failure_index];
                state.failure_index += 1;
                if (failure) |classified| {
                    state.last_failure = classified;
                    return error.LMFailure;
                }
            }
            const index = state.induction_count;
            state.induction_count += 1;
            break :blk try std.fmt.allocPrint(arena, "<plan>Use the examples.</plan>\n<natural_language_rules>\n\u{2003} rule-{d} \u{a0}\n</natural_language_rules>", .{index});
        } else if (std.mem.indexOf(u8, system, "`answer`") == null) blk: {
            const user = request.messages[request.messages.len - 1].content.text;
            const question = between(user, "<question>", "</question>");
            break :blk try std.fmt.allocPrint(arena, "<draft>generated:{s}</draft>", .{question});
        } else blk: {
            var good = self.teacher or std.mem.indexOf(u8, system, "Please adhere") == null;
            for (state.good_rules) |rule| {
                if (std.mem.indexOf(u8, system, rule) != null) good = true;
            }
            break :blk if (good) "<answer>yes</answer>" else "<answer>no</answer>";
        };
        const result = try arena.alloc(lm_mod.Completion, 1);
        result[0] = .{ .text = text };
        return result;
    }
};

fn isInduction(request: lm_mod.Request) bool {
    return std.mem.indexOf(u8, request.messages[0].content.text, "natural_language_rules") != null;
}
fn between(text: []const u8, opening: []const u8, closing: []const u8) []const u8 {
    const start = (std.mem.indexOf(u8, text, opening) orelse return "") + opening.len;
    const end = std.mem.indexOfPos(u8, text, start, closing) orelse return "";
    return value.pyStrip(text[start..end]);
}
fn occurrenceCount(text: []const u8, needle: []const u8) usize {
    var count: usize = 0;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, needle)) |index| {
        count += 1;
        at = index + needle.len;
    }
    return count;
}
fn testExamples(arena: Allocator, count: usize) ![]const Example {
    const examples = try arena.alloc(Example, count);
    for (examples, 0..) |*example, index| {
        const json = try std.fmt.allocPrint(arena, "{{\"answer\":\"yes\",\"ignored\":\"discard\",\"question\":\"q{d}\",\"draft\":\"labeled:{d}\"}}", .{ index, index });
        example.* = try Example.fromJson(arena, json, &.{"question"});
    }
    return examples;
}
fn testMetric(_: Allocator, example: *const Example, maybe_prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    const prediction = maybe_prediction orelse return error.ParseValue;
    return if (try value.pyEqual(example.record.get("answer").?, prediction.get("answer") orelse return 0)) 1 else 0;
}
fn nanMetric(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    return std.math.nan(f64);
}
fn negativeInfinityMetric(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    return -std.math.inf(f64);
}
fn expectedInstruction(out: Allocator, original: []const u8, rule_index: usize) ![]const u8 {
    return std.fmt.allocPrint(out, "{s}\n\nPlease adhere to the following rules when making your prediction:\nrule-{d}", .{ original, rule_index });
}

test "InferRules real-source CoT prompts teacher options runtime count persistent RNG and Unicode strip" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a } };
    var student_lm: TestLM = .{ .state = &state };
    var teacher_lm: TestLM = .{ .state = &state, .teacher = true };
    const ctx: Context = .{ .arena = a, .lm = student_lm.lm(), .adapter = .chat, .lm_options = .{ .temperature = 0.25 } };
    var program = try RulesInductionProgram.init(a, 7, .{ .lm = teacher_lm.lm(), .adapter = .xml });
    for ([_][]const u8{ "first", "second", "third" }, 0..) |text, index| {
        const output = try program.forward(ctx, .{ .examples_text = text });
        try testing.expectEqualStrings(try std.fmt.allocPrint(a, "rule-{d}", .{index}), output);
    }
    const rollouts = [_]i64{ 906691059, 413653999, 813847339 };
    for (state.history.requests.items, state.teacher_calls.items, rollouts) |request, teacher, rollout| {
        try testing.expect(teacher);
        try testing.expectEqual(@as(?f64, 1.0), request.options.temperature);
        try testing.expectEqual(rollout, request.options.rollout_id.?.integer);
        try testing.expectEqual(@as(u32, 1), request.options.n);
    }
    // Byte-identical actual Python XMLAdapter prompt from infer_rules_oracle.py.
    try testing.expectEqualStrings(
        "Your input fields are:\n1. `examples_text` (str): Text containing examples\nYour output fields are:\n1. `plan` (str): A short plan for solving the task.\n2. `natural_language_rules` (str): Induced natural language rules\nOutput format (replace placeholders):\n<plan>\n{plan}\n</plan>\n\n<natural_language_rules>\n{natural_language_rules}\n</natural_language_rules>\nIn adhering to this structure, your objective is: \n        Given a set of examples, extract a list of 7 concise and non-redundant natural language rules that provide clear guidance for performing the task. All rules should be actionable for a well-specified scope of examples of this general kind of task.",
        state.history.requests.items[0].messages[0].content.text,
    );
    try testing.expectEqualStrings(
        "<examples_text>\nfirst\n</examples_text>\n\nOutput format (replace placeholders):\n<plan>\n{plan}\n</plan>\n\n<natural_language_rules>\n{natural_language_rules}\n</natural_language_rules>\n\nUse this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.",
        state.history.requests.items[0].messages[1].content.text,
    );
    try testing.expectEqualStrings("x\u{200b}", stripRules("\u{85}\u{a0}\u{1680}\u{2003}x\u{200b}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}\x1f"));
    const source_spaces = [_]u21{ 9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288 };
    const padding = try unicode.encode(a, &source_spaces);
    try testing.expectEqualStrings("x\u{200b}", stripRules(try std.fmt.allocPrint(a, "{s}x\u{200b}{s}", .{ padding, padding })));
    try testing.expectEqualStrings("\xff", stripRules(" \xff "));
    try testing.expectEqualStrings("x\x80\x80\x80\x80", stripRules("x\x80\x80\x80\x80"));
}

test "InferRules formats original Example store order Python str and incomplete examples" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const Formatting = signature.Signature(struct {
        pub const Input = struct { question: []const u8, nested: []const u8 };
        pub const Output = struct { flag: bool, answer: []const u8 };
    });
    const predictor: predict.Predict(Formatting) = .{};
    const examples = [_]Example{
        try Example.fromJson(a, "{\"answer\":null,\"unused\":\"skip\",\"nested\":{\"yes\":false},\"question\":[true,\"can't\"],\"flag\":true}", &.{}),
        try Example.fromJson(a, "{\"unused\":\"missing all\"}", &.{}),
        try Example.fromJson(a, "{\"flag\":false,\"question\":\"\u{2003} raw \u{a0}\"}", &.{}),
    };
    const demos = try getPredictorDemos(a, &examples, &predictor.state);
    try testing.expectEqual(@as(usize, 4), demos[0].count());
    try testing.expectEqual(@as(usize, 0), demos[1].count());
    try testing.expectEqualStrings(
        "Input Fields:\nnested: {'yes': False}\nquestion: [True, \"can't\"]\n\n=========\nOutput Fields:\nanswer: None\nflag: True\n\n" ++
            "Input Fields:\n\n\n=========\nOutput Fields:\n\n\n" ++
            "Input Fields:\nquestion: \u{2003} raw \u{a0}\n\n=========\nOutput Fields:\nflag: False\n\n",
        try formatExamples(a, demos, &predictor.state),
    );
}

test "InferRules source Example.items hides reserved metadata even when it is a signature field" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const Reserved = signature.Signature(struct {
        pub const Input = struct { question: []const u8, dspy_secret: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const predictor: predict.Predict(Reserved) = .{};
    const examples = [_]Example{try Example.fromJson(a, "{\"dspy_secret\":\"hidden\",\"question\":\"visible\",\"answer\":\"yes\"}", &.{"question"})};
    const demos = try getPredictorDemos(a, &examples, &predictor.state);
    try testing.expectEqual(@as(usize, 2), demos[0].count());
    try testing.expectEqualStrings("Input Fields:\nquestion: visible\n\n=========\nOutput Fields:\nanswer: yes\n\n", try formatExamples(a, demos, &predictor.state));
}

test "InferRules splits once bootstraps once snapshots resets strict ties and final source aliases" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a } };
    var student_lm: TestLM = .{ .state = &state };
    var teacher_lm: TestLM = .{ .state = &state, .teacher = true };
    const ctx: Context = .{ .arena = a, .lm = student_lm.lm() };
    var student: TestProgram = .{};
    var optimizer = try InferRules.init(a, .{
        .metric = testMetric,
        .num_candidates = 3,
        .num_rules = 7,
        .max_bootstrapped_demos = 1,
        .max_labeled_demos = 0,
        .teacher_settings = .{ .lm = teacher_lm.lm(), .adapter = .xml },
    });
    const result = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 5), null);
    try testing.expectEqual(@as(?usize, 1), result.best_candidate);
    try testing.expectEqual(@as(f64, 100), result.best_score);
    try testing.expectEqual(@as(usize, 1), result.bootstrap_report.attempts);
    try testing.expectEqual(@as(usize, 1), result.bootstrap_report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), result.bootstrap_report.validation);
    try testing.expectEqualStrings("Original first.", result.original_instructions[0]);
    try testing.expectEqualStrings("Original second.", result.original_instructions[1]);
    for (result.candidates, [_]f64{ 0, 100, 100 }, 0..) |candidate, score, index| {
        try testing.expectEqual(score, candidate.score);
        try testing.expectEqualStrings(try expectedInstruction(a, "Original first.", index * 2), candidate.instructions[0]);
        try testing.expectEqualStrings(try expectedInstruction(a, "Original second.", index * 2 + 1), candidate.instructions[1]);
    }
    const best = result.program.?;
    try testing.expectEqualStrings(try expectedInstruction(a, "Original first.", 4), best.first.state.instructions);
    try testing.expectEqualStrings(try expectedInstruction(a, "Original second.", 5), best.second.state.instructions);
    try testing.expectEqualStrings(best.first.state.instructions, student.first.state.instructions);
    try testing.expectEqualStrings(best.second.state.instructions, result.original_program.second.state.instructions);
    try testing.expect(!student.compiled);
    try testing.expectEqual(@as(usize, 0), student.first.state.demos.len);
    try testing.expectEqual(@as(usize, 1), best.first.state.demos.len);
    try testing.expectEqualStrings("generated:q0", best.first.state.demos[0].get("draft").?.string);
    var induction_index: usize = 0;
    const rollouts = [_]i64{ 906691059, 413653999, 813847339, 955892128, 451585301, 43469773 };
    for (state.history.requests.items, state.teacher_calls.items) |request, teacher| {
        if (!isInduction(request)) continue;
        const user = request.messages[request.messages.len - 1].content.text;
        try testing.expectEqual(@as(usize, 2), occurrenceCount(user, "Input Fields:"));
        try testing.expect(std.mem.indexOf(u8, user, "labeled:0") != null);
        try testing.expect(std.mem.indexOf(u8, user, "generated:") == null);
        try testing.expect(std.mem.indexOf(u8, user, "q2") == null);
        try testing.expect(teacher);
        try testing.expectEqual(rollouts[induction_index], request.options.rollout_id.?.integer);
        induction_index += 1;
    }
    try testing.expectEqual(@as(usize, 6), induction_index);
    try testing.expectEqual(@as(usize, 26), state.history.requests.items.len);
    const previous = state.history.requests.items.len;
    var second_student: TestProgram = .{};
    _ = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &second_student, null, try testExamples(a, 2), try testExamples(a, 1));
    const next_induction = state.history.requests.items[previous + 2];
    try testing.expect(isInduction(next_induction));
    try testing.expectEqual(@as(i64, 278009742), next_induction.options.rollout_id.?.integer);
}

test "InferRules eagerly captures persistent rules metadata settings binding and RNG across reconfigured compiles" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a } };
    var student_lm: TestLM = .{ .state = &state };
    var teacher_lm: TestLM = .{ .state = &state, .teacher = true };
    var replacement_lm: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = student_lm.lm() };
    var original_settings: optimize.TeacherSettings = .{ .lm = teacher_lm.lm(), .adapter = .xml };
    const replacement_settings: optimize.TeacherSettings = .{ .lm = replacement_lm.lm(), .adapter = .xml };
    var optimizer = try InferRules.init(a, .{
        .metric = testMetric,
        .num_candidates = 1,
        .num_rules = 7,
        .max_bootstrapped_demos = 0,
        .max_labeled_demos = 0,
        .teacher_settings_ref = &original_settings,
    });
    const rules_program = &optimizer.rules_induction_program;
    const signature_fields = rules_program.rules_induction.predict.state.fields.ptr;
    optimizer.num_rules = 99; // Constructor capture has already happened.
    var student: TestProgram = .{};
    _ = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), try testExamples(a, 1));

    optimizer.num_rules = -2;
    optimizer.teacher_settings_ref = &replacement_settings;
    rules_program.rules_induction.predict.state.instructions = "Caller-owned induction instructions.";
    student = .{};
    _ = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), try testExamples(a, 1));

    // Rebinding did not retarget the captured object, but mutating that original
    // object remains visible on the third compile, exactly like the source dict.
    original_settings.lm = replacement_lm.lm();
    student = .{};
    _ = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), try testExamples(a, 1));
    try testing.expect(rules_program == &optimizer.rules_induction_program);
    try testing.expect(signature_fields == rules_program.rules_induction.predict.state.fields.ptr);
    try testing.expect(rules_program.teacher_settings_ref == &original_settings);
    try testing.expectEqual(@as(i64, 7), rules_program.num_rules);
    try testing.expectEqualStrings("Caller-owned induction instructions.", rules_program.rules_induction.predict.state.instructions);
    var index: usize = 0;
    const rollouts = [_]i64{ 906691059, 413653999, 813847339, 955892128, 451585301, 43469773 };
    for (state.history.requests.items, state.teacher_calls.items) |request, teacher| {
        if (!isInduction(request)) continue;
        try testing.expectEqual(index < 4, teacher);
        try testing.expectEqual(rollouts[index], request.options.rollout_id.?.integer);
        const expected: []const u8 = if (index < 2) "a list of 7 concise" else "Caller-owned induction instructions.";
        try testing.expect(std.mem.indexOf(u8, request.messages[0].content.text, expected) != null);
        index += 1;
    }
    try testing.expectEqual(@as(usize, 6), index);

    // Value settings are an explicit captured-value API, not a pretend shared
    // dictionary. Reassigning the optimizer value does not replace rule settings.
    var value_optimizer = try InferRules.init(a, .{ .teacher_settings = .{ .lm = teacher_lm.lm(), .adapter = .xml } });
    value_optimizer.teacher_settings = replacement_settings;
    const predictor: predict.Predict(TestFirst) = .{};
    _ = try value_optimizer.induceNaturalLanguageRules(a, testing.allocator, ctx, &predictor.state, try testExamples(a, 2));
    try testing.expect(state.teacher_calls.items[state.teacher_calls.items.len - 1]);
}

test "InferRules distinct signature earlier winner retains identity but acquires final worse instructions" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a }, .good_rules = &.{"rule-1"} };
    var model: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = model.lm() };
    var student: TestProgram = .{};
    var optimizer = try InferRules.init(a, .{ .metric = testMetric, .num_candidates = 3, .max_bootstrapped_demos = 0, .max_labeled_demos = 0 });
    const result = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), try testExamples(a, 1));
    try testing.expectEqual(@as(?usize, 0), result.best_candidate);
    for (result.candidates, [_]f64{ 100, 0, 0 }) |candidate, score| try testing.expectEqual(score, candidate.score);
    try testing.expectEqualStrings(try expectedInstruction(a, "Original first.", 0), result.candidates[0].instructions[0]);
    try testing.expectEqualStrings(try expectedInstruction(a, "Original second.", 1), result.candidates[0].instructions[1]);
    try testing.expectEqualStrings(try expectedInstruction(a, "Original first.", 4), result.program.?.first.state.instructions);
    try testing.expectEqualStrings(try expectedInstruction(a, "Original second.", 5), result.program.?.second.state.instructions);
    try testing.expectEqualStrings(result.program.?.first.state.instructions, student.first.state.instructions);
    try testing.expectEqualStrings(result.program.?.second.state.instructions, result.original_program.second.state.instructions);
}

test "InferRules shared signature classes mutate caller teacher bootstrap and earlier best after worse candidates" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a }, .good_rules = &.{"rule-1"} };
    var model: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = model.lm() };
    var student: TestSharedProgram = .{};
    var teacher: TestSharedProgram = .{};
    var optimizer = try InferRules.init(a, .{ .metric = testMetric, .num_candidates = 3, .max_bootstrapped_demos = 0, .max_labeled_demos = 0 });
    const result = try optimizer.compile(TestSharedProgram, a, testing.allocator, ctx, &student, &teacher, try testExamples(a, 2), try testExamples(a, 1));
    try testing.expectEqual(@as(?usize, 0), result.best_candidate);
    for (result.candidates, [_]f64{ 100, 0, 0 }, 0..) |candidate, score, index| {
        try testing.expectEqual(score, candidate.score);
        for (candidate.instructions) |text| try testing.expectEqualStrings(try expectedInstruction(a, "Shared task.", index * 2 + 1), text);
    }
    const expected = try expectedInstruction(a, "Shared task.", 5);
    inline for (.{ "first", "second" }) |name| {
        try testing.expectEqualStrings(expected, @field(student, name).state.instructions);
        try testing.expectEqualStrings(expected, @field(teacher, name).state.instructions);
        try testing.expectEqualStrings(expected, @field(result.original_program, name).state.instructions);
        try testing.expectEqualStrings(expected, @field(result.program.?, name).state.instructions);
    }
}

test "InferRules second induction failure retains precisely the preceding source signature mutations" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    inline for (.{ TestProgram, TestSharedProgram }) |P| {
        var state: TestState = .{ .history = .{ .arena = a }, .failures = &.{ null, .other } };
        var model: TestLM = .{ .state = &state };
        const ctx: Context = .{ .arena = a, .lm = model.lm() };
        var student: P = .{};
        var optimizer = try InferRules.init(a, .{ .metric = testMetric, .num_candidates = 2, .max_bootstrapped_demos = 0, .max_labeled_demos = 0 });
        try testing.expectError(error.UnacceptedRulesInductionFailure, optimizer.compile(P, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), try testExamples(a, 1)));
        try testing.expectEqual(@as(usize, 2), state.history.requests.items.len);
        try testing.expectEqual(@as(usize, 1), state.induction_count);
        try testing.expectEqual(@as(?anyerror, error.LMFailure), optimizer.last_induction_failure);
        const retained = optimizer.student(P).?;
        try testing.expect(retained.compiled);
        try testing.expectEqualStrings(student.first.state.instructions, retained.first.state.instructions);
        try testing.expectEqualStrings(student.second.state.instructions, retained.second.state.instructions);
        try testing.expect(optimizer.student(if (P == TestProgram) TestSharedProgram else TestProgram) == null);
        if (P == TestSharedProgram) {
            const expected = try expectedInstruction(a, "Shared task.", 0);
            try testing.expectEqualStrings(expected, student.first.state.instructions);
            try testing.expectEqualStrings(expected, student.second.state.instructions);
        } else {
            try testing.expectEqualStrings(try expectedInstruction(a, "Original first.", 0), student.first.state.instructions);
            try testing.expectEqualStrings("Original second.", student.second.state.instructions);
        }
        for (state.history.requests.items, [_]i64{ 906691059, 413653999 }) |request, rollout| {
            try testing.expect(isInduction(request));
            try testing.expectEqual(rollout, request.options.rollout_id.?.integer);
        }
    }
}

test "InferRules explicit validation empty failure timing negative candidates empty bootstrap and zero limits" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a } };
    var model: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = model.lm() };
    var optimizer = try InferRules.init(a, .{ .metric = testMetric, .num_candidates = 1, .max_bootstrapped_demos = 0, .max_labeled_demos = 0 });
    var student: TestProgram = .{};
    _ = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 3), try testExamples(a, 1));
    try testing.expectEqual(@as(usize, 3), occurrenceCount(state.history.requests.items[0].messages[1].content.text, "Input Fields:"));
    const before = state.history.requests.items.len;
    student = .{};
    try testing.expectError(error.EmptyValidationSet, optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), &.{}));
    try testing.expectEqual(before + 2, state.history.requests.items.len);
    try testing.expect(std.mem.indexOf(u8, student.first.state.instructions, "Please adhere") != null);
    const failed_student = optimizer.student(TestProgram).?;
    try testing.expect(failed_student.compiled);
    try testing.expectEqualStrings(student.first.state.instructions, failed_student.first.state.instructions);
    try testing.expectEqualStrings(student.second.state.instructions, failed_student.second.state.instructions);
    try testing.expect(optimizer.student(TestSharedProgram) == null);
    for ([_]isize{ 0, -2 }) |count| {
        optimizer.num_candidates = count;
        student = .{};
        const result = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), &.{});
        try testing.expect(result.program == null);
        try testing.expect(result.best_candidate == null);
        try testing.expectEqual(-std.math.inf(f64), result.best_score);
    }
    try testing.expectEqual(before + 2, state.history.requests.items.len);
    student = .{};
    try testing.expectError(error.EmptyBootstrapTrainset, optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 1), null));
    // A new failed compile must never expose the previous invocation's student.
    try testing.expect(optimizer.student(TestProgram) == null);
    try testing.expect(optimizer.student(TestSharedProgram) == null);
    optimizer.max_rounds = 0;
    optimizer.max_errors = 0;
    optimizer.max_labeled_demos = 2;
    student = .{};
    const zero = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 3), &.{});
    try testing.expectEqual(@as(usize, 0), zero.bootstrap_report.attempts);
    try testing.expectEqual(@as(usize, 2), zero.original_program.first.state.demos.len);
    try testing.expectEqualStrings("q2", zero.original_program.first.state.demos[0].get("question").?.string);
    try testing.expectEqualStrings("q1", zero.original_program.first.state.demos[1].get("question").?.string);
}

test "InferRules all NaN and negative infinity evaluations return genuine null" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a } };
    var model: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = model.lm() };
    for ([_]optimize.Metric{ nanMetric, negativeInfinityMetric }) |metric| {
        var optimizer = try InferRules.init(a, .{ .metric = metric, .num_candidates = 2, .max_bootstrapped_demos = 0, .max_labeled_demos = 0 });
        var student: TestProgram = .{};
        const result = try optimizer.compile(TestProgram, a, testing.allocator, ctx, &student, null, try testExamples(a, 2), try testExamples(a, 1));
        try testing.expect(result.program == null);
        try testing.expect(result.best_candidate == null);
        try testing.expectEqual(-std.math.inf(f64), result.best_score);
        try testing.expectEqual(@as(usize, 2), result.candidates.len);
    }
}

test "InferRules only classified source errors shrink the final training example and consume fresh rollouts" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a }, .failures = &.{ .value_error, .bad_request_error, .context_window_exceeded } };
    var model: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = model.lm() };
    const predictor: predict.Predict(TestFirst) = .{};
    var optimizer = try InferRules.init(a, .{ .failure_classifier = .{ .ptr = &state, .classify = TestState.classify } });
    const rules = try optimizer.induceNaturalLanguageRules(a, testing.allocator, ctx, &predictor.state, try testExamples(a, 4));
    try testing.expectEqualStrings("rule-0", rules);
    for (state.history.requests.items, [_]usize{ 4, 3, 2, 1 }, [_]i64{ 906691059, 413653999, 813847339, 955892128 }) |request, count, rollout| {
        try testing.expectEqual(count, occurrenceCount(request.messages[1].content.text, "Input Fields:"));
        try testing.expectEqual(rollout, request.options.rollout_id.?.integer);
        try testing.expectEqual(@as(?f64, 1), request.options.temperature);
        if (count < 4) try testing.expect(std.mem.indexOf(u8, request.messages[1].content.text, "q3") == null);
    }
    for ([_]usize{ 0, 1 }) |count| {
        state.failures = &.{.value_error};
        state.failure_index = 0;
        const previous = state.history.requests.items.len;
        try testing.expectError(error.RulesContextWindowExceeded, optimizer.induceNaturalLanguageRules(a, testing.allocator, ctx, &predictor.state, try testExamples(a, count)));
        try testing.expectEqual(previous + 1, state.history.requests.items.len);
    }
    for ([_]bool{ false, true }) |with_classifier| {
        optimizer.failure_classifier = if (with_classifier) .{ .ptr = &state, .classify = TestState.classify } else null;
        state.failures = &.{.other};
        state.failure_index = 0;
        const previous = state.history.requests.items.len;
        try testing.expectError(error.UnacceptedRulesInductionFailure, optimizer.induceNaturalLanguageRules(a, testing.allocator, ctx, &predictor.state, try testExamples(a, 3)));
        try testing.expectEqual(@as(?anyerror, error.LMFailure), optimizer.last_induction_failure);
        try testing.expectEqual(previous + 1, state.history.requests.items.len);
    }
}

test "InferRules missing metric runs prediction before evaluation errors" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var state: TestState = .{ .history = .{ .arena = a } };
    var model: TestLM = .{ .state = &state };
    const ctx: Context = .{ .arena = a, .lm = model.lm() };
    const student: TestProgram = .{};
    var optimizer = try InferRules.init(a, .{});
    const dataset = try testExamples(a, 1);
    try testing.expectEqual(@as(f64, 0), try optimizer.evaluateProgram(TestProgram, testing.allocator, ctx, &student, dataset));
    try testing.expectEqual(@as(usize, 2), state.history.requests.items.len);
    optimizer.max_errors = 1;
    try testing.expectError(error.MissingMetric, optimizer.evaluateProgram(TestProgram, testing.allocator, ctx, &student, dataset));
    try testing.expectEqual(@as(usize, 4), state.history.requests.items.len);
    try testing.expectError(error.EmptyValidationSet, optimizer.evaluateProgram(TestProgram, testing.allocator, ctx, &student, &.{}));
}
