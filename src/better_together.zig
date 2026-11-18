//! BetterTogether composes real native optimizers over a borrowed typed program.
//! Caller-provisioned output backing retains every candidate, including its LM
//! identities. Scratch and startup jobs may be reused only after joined calls.
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const predict = @import("predict.zig");
const optimize = @import("optimize.zig");
const finetune = @import("bootstrap_finetune.zig");
const lm = @import("lm.zig");
const value = @import("value.zig");
const pyrandom = @import("pyrandom.zig");
pub const optimizers = @import("better_together_optimizers.zig");

pub const Limits = struct {
    max_steps: u32,
    max_examples: u32,
    max_predictors: u32,
    max_teachers: u32,
    max_strategy_bytes: u32,
};

/// Outer presence distinguishes an omitted argument from an explicit null.
pub fn Override(comptime T: type) type {
    return struct { value: T };
}

/// Optimizer-specific settings remain typed on each registry entry. Standard
/// compile arguments can be replaced independently for each named optimizer.
/// Unlike Options.teachers, the teacher override reaches the optimizer raw:
/// explicit None, one program, or a list that is never normalized.
pub const CompileArgs = struct {
    key: []const u8,
    trainset: ?[]const optimize.Example = null,
    valset: ?Override(?[]const optimize.Example) = null,
    teacher: ?optimizers.TeacherArgument = null,
};

pub const Options = struct {
    teachers: ?[]const finetune.Teacher = null,
    valset: ?[]const optimize.Example = null,
    valset_ratio: f64 = 0.1,
    seed: ?u32 = null,
    shuffle_trainset_between_steps: bool = true,
    strategy: []const u8 = "p -> w -> p",
    optimizer_compile_args: []const CompileArgs = &.{},
    max_errors: usize = 10,
    concurrent_evaluation: ?optimize.ConcurrentEvaluation = null,
};

/// Fresh model/name slots per weight step; sequential steps may share startup
/// Jobs, but never the retained names borrowed by earlier candidate programs.
pub const Defaults = struct {
    weight_limits: finetune.Limits,
    weight_slot_sets: []const []finetune.Slot,
    trace_options: @import("bootstrap_trace.zig").Options = .{},
    prompt: ?optimize.BootstrapFewShotWithRandomSearch = null,
    /// Source `restrict` / `labeled_sample` for the default prompt optimizer.
    prompt_compile_options: optimize.BootstrapFewShotWithRandomSearch.Options = .{
        .restrict = null,
        .labeled_sample = true,
    },
    weight: ?finetune.BootstrapFinetune = null,
};

pub fn Candidate(comptime P: type) type {
    return struct {
        program: P,
        score: ?f64,
        strategy: []const u8,
    };
}

pub fn Result(comptime P: type) type {
    return struct {
        program: P,
        candidate_programs: []Candidate(P),
        flag_compilation_error_occurred: bool,
    };
}

pub const BetterTogether = struct {
    metric: optimize.Metric,
    limits: Limits,

    pub fn compile(
        self: *const BetterTogether,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        io: std.Io,
        ctx: predict.Context,
        student: *P,
        trainset: []const optimize.Example,
        options: Options,
        defaults: Defaults,
    ) !Result(P) {
        var prompt: optimizers.RandomSearch = .{
            .optimizer = defaults.prompt orelse .{ .metric = self.metric },
            .compile_options = defaults.prompt_compile_options,
        };
        var weight: optimizers.BootstrapFinetune = .{
            .optimizer = defaults.weight orelse .{
                .metric = self.metric,
                .limits = defaults.weight_limits,
            },
            .slot_sets = defaults.weight_slot_sets,
            .trace_options = defaults.trace_options,
        };
        return self.compileWith(P, out, gpa, io, ctx, student, trainset, options, .{
            .p = &prompt,
            .w = &weight,
        });
    }

    /// Registry fields are exact strategy keys; values are borrowed pointers to
    /// typed adapters implementing compile(P, Request(P)) !P. Any native optimizer
    /// can implement that protocol; the strategy is not a p/w-only enum.
    pub fn compileWith(
        self: *const BetterTogether,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        io: std.Io,
        ctx: predict.Context,
        student: *P,
        trainset: []const optimize.Example,
        options: Options,
        registry: anytype,
    ) !Result(P) {
        if (student.compiled) return error.StudentCompiled;
        const initial = try self.predictors(out, student);
        for (initial) |item| if (item.param.predict.lm == null) return error.MissingPredictorLM;
        const teachers = try self.prepareTeachers(P, out, student, initial, options.teachers);
        const sets = try self.prepareDatasets(out, trainset, options);
        const steps = try self.prepareStrategy(out, registry, options.strategy);
        try self.validateArgs(registry, teachers, options.optimizer_compile_args);
        var request: optimizers.Request(P) = .{
            .out = out,
            .gpa = gpa,
            .io = io,
            .ctx = ctx,
            .student = student.*,
            .student_ref = student,
            .trainset = sets.trainset,
            .valset = sets.valset,
            .teacher = if (teachers) |list| .{ .list = list } else .none,
            .step_index = 0,
        };
        var seed = options.seed;
        if (seed == null) {
            var bytes: [4]u8 = undefined;
            try io.randomSecure(&bytes);
            seed = std.mem.readInt(u32, &bytes, .little);
        }
        var rng = pyrandom.Random.init(seed.?);
        const candidates = try out.alloc(Candidate(P), steps.len + 1);
        lifecycle(initial, .launch, &.{}) catch |err| {
            try lifecycle(initial, .kill, &.{});
            return err;
        };
        candidates[0] = self.candidate(P, &request, student.*, "", options) catch |err| {
            try lifecycle(initial, .kill, &.{});
            return err;
        };
        return self.run(P, &request, student, initial, registry, steps, candidates, &rng, options);
    }

    fn run(
        self: *const BetterTogether,
        comptime P: type,
        request: *optimizers.Request(P),
        original: *P,
        initial: []const predict.NamedParam,
        registry: anytype,
        steps: []const []const u8,
        candidates: []Candidate(P),
        rng: *pyrandom.Random,
        options: Options,
    ) !Result(P) {
        var current = original;
        // Cleanup never allocates: discovery OOM cannot strand launched models.
        var current_predictors = initial;
        var count: usize = 1;
        var failed = false;
        for (steps, 0..) |key, index| {
            if (options.shuffle_trainset_between_steps) rng.shuffle(optimize.Example, @constCast(request.trainset));
            request.student = current.*;
            request.student_ref = current;
            request.student.compiled = false;
            current.compiled = false;
            request.step_index = index;
            const prefix = options.strategy[0..prefixLength(steps[0 .. index + 1])];
            const step_result = self.step(P, request, current, original, candidates[0..count], registry, key, options.optimizer_compile_args) catch {
                failed = true;
                break;
            };
            // Publication happens only here, after restoration and launch
            // succeeded: a failed step leaves the old program and its launched
            // models as the ones cleanup sees.
            current_predictors = step_result.predictors;
            current = step_result.program;
            request.student = current.*;
            candidates[count] = self.candidate(P, request, current.*, prefix, options) catch {
                failed = true;
                break;
            };
            count += 1;
        }
        try lifecycle(current_predictors, .kill, &.{});
        assert(count > 0);
        assert(count <= candidates.len);
        const kept = candidates[0..count];
        const no_validation = if (request.valset) |items| items.len == 0 else true;
        // All no-validation scores are null, so stable sorting leaves chronology.
        std.sort.insertion(Candidate(P), kept, {}, candidateGreater(P));
        const best = if (no_validation) &kept[kept.len - 1] else &kept[0];
        best.program.compiled = true;
        return .{
            .program = best.program,
            .candidate_programs = kept,
            .flag_compilation_error_occurred = failed,
        };
    }

    fn StepResult(comptime P: type) type {
        return struct {
            /// The caller's current program for in-place optimizers, otherwise
            /// fresh result-arena storage. Never a stack or request temporary.
            program: *P,
            predictors: []const predict.NamedParam,
        };
    }

    fn step(
        self: *const BetterTogether,
        comptime P: type,
        request: *optimizers.Request(P),
        current: *P,
        original: *P,
        candidates: []Candidate(P),
        registry: anytype,
        key: []const u8,
        overrides: []const CompileArgs,
    ) !StepResult(P) {
        const before = try self.predictors(request.out, current);
        const states = try request.out.alloc(predict.PredictState, before.len);
        const models = try request.out.alloc(?lm.LM, before.len);
        for (states, models, before) |*state, *saved, item| {
            state.* = item.param.predict.*;
            saved.* = state.lm;
        }
        var in_place = false;
        var selected = request.*;
        for (overrides) |args| if (std.mem.eql(u8, args.key, key)) {
            if (args.trainset) |set| selected.trainset = set;
            if (args.valset) |set| selected.valset = set.value;
            if (args.teacher) |teacher| selected.teacher = teacher;
        };
        if (selected.trainset.len > self.limits.max_examples) return error.MetaExampleLimit;
        const graphs: TeacherGraphs = .{ .top = request.teacher, .overrides = overrides };
        // Stable result storage: the previous program, its predictor pointers
        // and its launched models stay intact until this step fully succeeds.
        var compiled: *P = current;
        inline for (@typeInfo(@TypeOf(registry)).@"struct".field_names) |name| {
            if (std.mem.eql(u8, name, key)) {
                const entry = @field(registry, name);
                const T = @typeInfo(@TypeOf(entry)).pointer.child;
                if (@hasDecl(T, "returns_in_place")) in_place = T.returns_in_place;
                if (!in_place) compiled = try request.out.create(P);
                compiled.* = entry.compile(P, selected) catch |err| {
                    try propagateAliases(P, request.out, states, current, before, original, candidates, graphs);
                    return err;
                };
            }
        }
        // In-place optimizers already wrote the actual current program; only
        // value-returning optimizers require separate stable publication storage.
        const target = compiled;
        const after = if (in_place) before else try self.predictors(request.out, target);
        var missing = false;
        for (after) |item| if (item.param.predict.lm == null) {
            missing = true;
        };
        if (missing) {
            for (after[0..@min(after.len, models.len)], models[0..@min(after.len, models.len)]) |item, saved| {
                item.param.predict.lm = saved;
            }
        }
        try propagateAliases(P, request.out, states, target, before, original, candidates, graphs);
        if (modelsChanged(models, after)) {
            lifecycle(after, .launch, &.{}) catch |err| {
                // An in-place program IS current: final cleanup joins it once.
                // A rejected replacement is joined here, except handles it
                // shares with the surviving program, which cleanup still owns.
                if (!in_place) try lifecycle(after, .kill, before);
                return err;
            };
        }
        assert(after.len == 0 or in_place == (after.ptr == before.ptr));
        return .{ .program = target, .predictors = after };
    }

    fn predictors(self: *const BetterTogether, out: Allocator, program: anytype) ![]predict.NamedParam {
        const items = try predict.namedPredictors(out, program);
        if (items.len > self.limits.max_predictors) return error.MetaPredictorLimit;
        return items;
    }

    fn prepareTeachers(
        self: *const BetterTogether,
        comptime P: type,
        out: Allocator,
        student: *P,
        initial: []const predict.NamedParam,
        selected: ?[]const finetune.Teacher,
    ) !?[]const finetune.Teacher {
        const teachers = selected orelse return null;
        if (teachers.len == 0) return null;
        try finetune.validateTeachers(out, initial, teachers, self.limits.max_teachers);
        const prepared = try out.dupe(finetune.Teacher, teachers);
        for (prepared) |*item| if (item.* == .student) {
            item.* = finetune.Teacher.init(P, student);
        };
        return prepared;
    }

    const Datasets = struct { trainset: []const optimize.Example, valset: ?[]const optimize.Example };

    fn prepareDatasets(self: *const BetterTogether, out: Allocator, trainset: []const optimize.Example, options: Options) !Datasets {
        if (trainset.len == 0) return error.EmptyTrainset;
        if (trainset.len > self.limits.max_examples) return error.MetaExampleLimit;
        if (options.valset_ratio < 0 or options.valset_ratio >= 1) return error.InvalidValidationRatio;
        const copy = try out.dupe(optimize.Example, trainset);
        if (options.valset) |set| {
            if (set.len > self.limits.max_examples) return error.MetaExampleLimit;
            if (set.len > 0) return .{ .trainset = copy, .valset = set };
        }
        if (options.valset_ratio == 0) return .{ .trainset = copy, .valset = null };
        if (!std.math.isFinite(options.valset_ratio)) return error.InvalidValidationRatio;
        const count: usize = @intFromFloat(options.valset_ratio * @as(f64, @floatFromInt(copy.len)));
        assert(count < copy.len);
        return .{ .trainset = copy[count..], .valset = copy[0..count] };
    }

    fn prepareStrategy(self: *const BetterTogether, out: Allocator, registry: anytype, strategy: []const u8) ![]const []const u8 {
        if (strategy.len > self.limits.max_strategy_bytes) return error.MetaStrategyByteLimit;
        if (std.mem.trim(u8, strategy, " \t\r\n\x0b\x0c").len == 0) return error.EmptyStrategy;
        var split = std.mem.splitSequence(u8, strategy, " -> ");
        var count: u32 = 0;
        while (split.next()) |key| {
            if (!hasKey(registry, key)) return error.UnknownOptimizer;
            if (count >= self.limits.max_steps) return error.MetaStepLimit;
            count += 1;
        }
        const steps = try out.alloc([]const u8, count);
        split.reset();
        for (steps) |*key| key.* = split.next().?;
        assert(count > 0);
        return steps;
    }

    /// Source validates every override against inspect.signature(compile)
    /// before any model is launched. Adapters without a declaration consume the
    /// whole Request and therefore accept both standard arguments.
    fn validateArgs(self: *const BetterTogether, registry: anytype, teachers: ?[]const finetune.Teacher, overrides: []const CompileArgs) !void {
        if (overrides.len > self.limits.max_steps) return error.MetaArgumentLimit;
        for (overrides, 0..) |args, index| {
            if (!hasKey(registry, args.key)) return error.UnknownOptimizer;
            for (overrides[0..index]) |previous| if (std.mem.eql(u8, previous.key, args.key)) return error.DuplicateOptimizerArgs;
            inline for (@typeInfo(@TypeOf(registry)).@"struct".field_names) |name| {
                if (std.mem.eql(u8, args.key, name)) {
                    const T = @typeInfo(@TypeOf(@field(registry, name))).pointer.child;
                    const accepts_valset: bool = if (@hasDecl(T, "accepts_valset")) T.accepts_valset else true;
                    const accepts_teacher: bool = if (@hasDecl(T, "accepts_teacher")) T.accepts_teacher else true;
                    if (args.valset != null) {
                        if (!accepts_valset) return error.InvalidCompileArgument;
                    }
                    if (args.teacher != null) {
                        if (!accepts_teacher) return error.InvalidCompileArgument;
                    }
                    if (@hasDecl(T, "source_name")) {
                        if (std.mem.eql(u8, T.source_name, "GEPA") and teachers != null) return error.GEPATeacher;
                    }
                }
            }
            if (args.trainset) |set| if (set.len > self.limits.max_examples) return error.MetaExampleLimit;
            if (args.valset) |override| if (override.value) |set| if (set.len > self.limits.max_examples) return error.MetaExampleLimit;
            if (args.teacher) |teacher| switch (teacher) {
                .none => {},
                .program => if (self.limits.max_teachers == 0) return error.MetaTeacherLimit,
                .list => |list| if (list.len > self.limits.max_teachers) return error.MetaTeacherLimit,
            };
        }
    }

    fn candidate(self: *const BetterTogether, comptime P: type, request: *const optimizers.Request(P), program: P, strategy: []const u8, options: Options) !Candidate(P) {
        const score: ?f64 = if (request.valset) |set| if (set.len > 0)
            // Source eval_candidate_program catches Evaluate exceptions as zero.
            (optimize.evaluateWith(P, options.concurrent_evaluation, request.gpa, request.ctx, &program, set, self.metric, options.max_errors) catch optimize.Evaluation{ .score = 0, .sum = 0, .total = set.len, .errors = set.len }).score
        else
            null else null;
        return .{
            .program = try copyProgram(P, request.out, program),
            .score = score,
            .strategy = try request.out.dupe(u8, strategy),
        };
    }
};

fn hasKey(registry: anytype, key: []const u8) bool {
    inline for (@typeInfo(@TypeOf(registry)).@"struct".field_names) |name| {
        if (std.mem.eql(u8, key, name)) return true;
    }
    return false;
}

fn prefixLength(steps: []const []const u8) usize {
    assert(steps.len > 0);
    var length: usize = (steps.len - 1) * " -> ".len;
    for (steps) |step| length += step.len;
    return length;
}

fn candidateGreater(comptime P: type) fn (void, Candidate(P), Candidate(P)) bool {
    return struct {
        fn greater(_: void, left: Candidate(P), right: Candidate(P)) bool {
            return (left.score orelse -std.math.inf(f64)) > (right.score orelse -std.math.inf(f64));
        }
    }.greater;
}

fn copyProgram(comptime P: type, out: Allocator, input: P) !P {
    var program = input;
    const predictors = try predict.namedPredictors(out, &program);
    const sources = try out.alloc(?lm.LM, predictors.len);
    const copies = try out.alloc(?lm.LM, predictors.len);
    for (predictors, 0..) |item, index| {
        const state = item.param.predict;
        sources[index] = state.lm;
        copies[index] = null;
        if (state.lm) |model| {
            var shared: ?lm.LM = null;
            for (sources[0..index], copies[0..index]) |previous, copy| {
                if (previous) |source| if (model.sameInstance(source)) {
                    shared = copy;
                };
            }
            state.lm = shared orelse try model.clone(out);
            copies[index] = state.lm;
        }
        state.instructions = try out.dupe(u8, state.instructions);
        const descriptions = try out.alloc([]const u8, state.descs.len);
        for (descriptions, state.descs) |*dst, src| dst.* = try out.dupe(u8, src);
        state.descs = descriptions;
        const prefixes = try out.alloc([]const u8, state.prefixes.len);
        for (prefixes, state.prefixes) |*dst, src| dst.* = try out.dupe(u8, src);
        state.prefixes = prefixes;
        const demos = try out.alloc(value.Record, state.demos.len);
        for (demos, state.demos) |*dst, *src| dst.* = try value.cloneRecord(out, src);
        state.demos = demos;
        state.decision_fields = try value.cloneValue(out, state.decision_fields);
    }
    return program;
}

/// Every teacher program reachable from this compile call: the prepared
/// top-level list and each per-key override, whichever step it belongs to.
const TeacherGraphs = struct {
    top: optimizers.TeacherArgument,
    overrides: []const CompileArgs,

    fn propagate(
        self: TeacherGraphs,
        out: Allocator,
        source: *const predict.PredictState,
    ) !void {
        try setArgument(out, self.top, source);
        for (self.overrides) |args| {
            if (args.teacher) |teacher| try setArgument(out, teacher, source);
        }
    }

    fn setArgument(
        out: Allocator,
        argument: optimizers.TeacherArgument,
        source: *const predict.PredictState,
    ) !void {
        switch (argument) {
            .none => {},
            .program => |program| try setProgram(out, program, source),
            .list => |list| for (list) |teacher| switch (teacher) {
                // The current student is already a propagation target.
                .student => {},
                .program => |program| try setProgram(out, program, source),
            },
        }
    }

    fn setProgram(
        out: Allocator,
        program: finetune.Teacher.Program,
        source: *const predict.PredictState,
    ) !void {
        for (try program.vtable.predictors(program.ptr, out)) |item| {
            const target = item.param.predict;
            if (target.signature_class != source.signature_class) continue;
            if (std.mem.eql(u8, target.instructions, source.instructions)) continue;
            if (!program.writable) return error.ImmutableTeacherProgram;
            target.instructions = source.instructions;
        }
    }
};

/// Mutable signature classes survive ordinary Module.deepcopy. Propagate only
/// classes already reachable before this step; with_instructions forks identity.
/// `previous` is the program the step started from: it remains current, and
/// observable, whenever a replacement is rejected after compiling.
fn propagateAliases(
    comptime P: type,
    out: Allocator,
    before: []const predict.PredictState,
    current: *P,
    previous: []const predict.NamedParam,
    original: *P,
    candidates: []Candidate(P),
    teachers: TeacherGraphs,
) !void {
    const after = try predict.namedPredictors(out, current);
    for (after) |item| {
        const changed = item.param.predict;
        var shared_class = false;
        for (before) |old| {
            if (old.signature_class == changed.signature_class and
                !std.mem.eql(u8, old.instructions, changed.instructions))
            {
                shared_class = true;
            }
        }
        if (shared_class) {
            setAliasedInstructions(previous, changed);
            setAliasedInstructions(try predict.namedPredictors(out, original), changed);
            for (candidates) |*candidate| {
                setAliasedInstructions(try predict.namedPredictors(out, &candidate.program), changed);
            }
            try teachers.propagate(out, changed);
        }
    }
}

fn setAliasedInstructions(targets: []const predict.NamedParam, source: *const predict.PredictState) void {
    for (targets) |item| {
        if (item.param.predict.signature_class == source.signature_class) {
            item.param.predict.instructions = source.instructions;
        }
    }
}

/// Source compares model-name lists. A native handle may expose no name; two
/// nameless positions then compare by presence and instance identity, so a
/// swapped nameless model cannot skip its launch.
fn modelsChanged(before: []const ?lm.LM, after: []const predict.NamedParam) bool {
    if (before.len != after.len) return true;
    for (before, after) |previous, item| {
        const current = item.param.predict.lm;
        const old_name = if (previous) |model| model.modelName() else null;
        const new_name = if (current) |model| model.modelName() else null;
        if (old_name) |old| {
            if (new_name == null or !std.mem.eql(u8, old, new_name.?)) return true;
        } else {
            if (new_name != null) return true;
            if (previous) |old_model| {
                const new_model = current orelse return true;
                if (!old_model.sameInstance(new_model)) return true;
            } else {
                if (current != null) return true;
            }
        }
    }
    return false;
}

/// Attempts every unique model and reports the first failure. Models also
/// bound in `retained` are skipped: their owner performs that transition.
fn lifecycle(
    predictors: []const predict.NamedParam,
    comptime action: enum { launch, kill },
    retained: []const predict.NamedParam,
) !void {
    var failure: ?anyerror = null;
    for (predictors, 0..) |item, index| {
        const model = item.param.predict.lm orelse continue;
        var seen = false;
        for (predictors[0..index]) |previous| {
            if (previous.param.predict.lm) |other| if (model.sameInstance(other)) {
                seen = true;
            };
        }
        for (retained) |kept| {
            if (kept.param.predict.lm) |other| if (model.sameInstance(other)) {
                seen = true;
            };
        }
        if (!seen) {
            const result = if (action == .launch) model.launch() else model.kill();
            result catch |err| {
                if (failure == null) failure = err;
            };
        }
    }
    if (failure) |err| return err;
}

const testing = std.testing;
const TestSignature = @import("signature.zig").Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: i64 };
});
const TestProgram = struct {
    first: predict.Predict(TestSignature) = .{},
    second: predict.Predict(TestSignature) = .{},
    stage: i64 = 0,
    compiled: bool = false,
    pub const Input = TestSignature.Input;
    pub const Output = TestSignature.Output;
    pub fn forward(self: *const @This(), _: predict.Context, _: Input) !Output {
        return .{ .answer = self.stage };
    }
};
const TestOptimizer = struct {
    delta: i64 = 1,
    fail: bool = false,
    replacement: ?lm.LM = null,
    instruction: ?[]const u8 = null,

    pub fn compile(self: *@This(), comptime P: type, request: optimizers.Request(P)) !P {
        if (self.instruction) |text| {
            request.student_ref.first.state.instructions = text;
            request.student_ref.second.state.instructions = text;
        }
        if (self.fail) return error.TestOptimizerFailure;
        var program = request.student_ref.*;
        program.stage += self.delta;
        if (self.replacement) |model| {
            program.first.state.lm = model;
            program.second.state.lm = null;
        }
        return program;
    }
};
/// Mirrors a Source compile(student, *, trainset, teacher=None) signature.
const TestTeacherProbe = struct {
    pub const accepts_valset = false;
    calls: u32 = 0,
    typed: ?*const TestProgram = null,
    foreign: bool = false,
    instruction: ?[]const u8 = null,

    pub fn compile(self: *@This(), comptime P: type, request: optimizers.Request(P)) !P {
        self.calls += 1;
        switch (request.teacher) {
            .program => |program| {
                self.typed = program.get(P);
                self.foreign = program.get(struct { compiled: bool }) != null;
            },
            .none, .list => {},
        }
        if (self.instruction) |text| {
            request.student_ref.first.state.instructions = text;
            return error.TestOptimizerFailure;
        }
        return request.student_ref.*;
    }
};
fn testMetric(_: Allocator, _: *const optimize.Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    return 1;
}
fn testOptimizer() BetterTogether {
    return .{
        .metric = testMetric,
        .limits = .{
            .max_steps = 8,
            .max_examples = 8,
            .max_predictors = 2,
            .max_teachers = 2,
            .max_strategy_bytes = 256,
        },
    };
}

test "BetterTogether retains first validation tie and distinct memoized LM snapshots" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const out = arena.allocator();
    var model: lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var student: TestProgram = .{};
    try predict.setLM(out, &student, model.lm());
    const examples = [_]optimize.Example{
        try optimize.Example.fromJson(out, "{\"question\":\"q\"}", &.{"question"}),
    };
    var change: TestOptimizer = .{};
    const result = try testOptimizer().compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, &examples, .{ .seed = 9, .strategy = "improve -> improve", .valset = &examples }, .{ .improve = &change });
    try testing.expectEqual(@as(i64, 0), result.program.stage);
    try testing.expect(result.program.compiled);
    try testing.expect(!result.flag_compilation_error_occurred);
    try testing.expectEqual(@as(usize, 3), result.candidate_programs.len);
    for (result.candidate_programs, 0..) |candidate, index| {
        try testing.expectEqual(@as(i64, @intCast(index)), candidate.program.stage);
        const copy = candidate.program.first.state.lm.?;
        try testing.expect(copy.sameInstance(candidate.program.second.state.lm.?));
        try testing.expect(!copy.sameInstance(model.lm()));
        for (result.candidate_programs[0..index]) |earlier| {
            try testing.expect(!copy.sameInstance(earlier.program.first.state.lm.?));
        }
    }
    model.model = "changed after compile";
    try testing.expectEqualStrings("base", result.program.first.state.lm.?.modelName().?);
}

// A raw single-program override must reach the optimizer as that nominal
// program, an override the optimizer's signature lacks must fail before any
// step runs, and a failed step's signature-class write must stay visible in
// both the top-level and the overridden teacher.
test "BetterTogether passes a raw teacher program prevalidates overrides and reaches teacher signature aliases" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const out = arena.allocator();
    var model: lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var student: TestProgram = .{};
    var top_teacher: TestProgram = .{};
    var raw_teacher: TestProgram = .{};
    try predict.setLM(out, &student, model.lm());
    const examples = [_]optimize.Example{
        try optimize.Example.fromJson(out, "{\"question\":\"q\"}", &.{"question"}),
    };
    var probe: TestTeacherProbe = .{};
    const invalid = [_]CompileArgs{.{ .key = "probe", .valset = .{ .value = null } }};
    try testing.expectError(error.InvalidCompileArgument, testOptimizer().compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, &examples, .{ .seed = 9, .strategy = "probe", .valset_ratio = 0, .optimizer_compile_args = &invalid }, .{ .probe = &probe }));
    try testing.expectEqual(@as(u32, 0), probe.calls);

    const oversized_teachers = [_]finetune.Teacher{ .student, .student, .student };
    const oversized_args = [_]CompileArgs{.{ .key = "probe", .teacher = .{ .list = &oversized_teachers } }};
    try testing.expectError(error.MetaTeacherLimit, testOptimizer().compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, &examples, .{ .seed = 9, .strategy = "probe", .valset_ratio = 0, .optimizer_compile_args = &oversized_args }, .{ .probe = &probe }));
    try testing.expectEqual(@as(u32, 0), probe.calls);

    probe.instruction = "teacher visible mutation";
    const teachers = [_]finetune.Teacher{finetune.Teacher.init(TestProgram, &top_teacher)};
    const raw = [_]CompileArgs{.{
        .key = "probe",
        .teacher = .{ .program = finetune.Teacher.init(TestProgram, &raw_teacher).program },
    }};
    const result = try testOptimizer().compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, &examples, .{ .seed = 9, .strategy = "probe", .valset_ratio = 0, .teachers = &teachers, .optimizer_compile_args = &raw }, .{ .probe = &probe });
    try testing.expect(result.flag_compilation_error_occurred);
    try testing.expectEqual(@as(u32, 1), probe.calls);
    try testing.expect(probe.typed == &raw_teacher);
    try testing.expect(!probe.foreign);
    try testing.expectEqualStrings("teacher visible mutation", student.second.state.instructions);
    try testing.expectEqualStrings("teacher visible mutation", top_teacher.first.state.instructions);
    try testing.expectEqualStrings("teacher visible mutation", raw_teacher.second.state.instructions);
}

test "BetterTogether bounds overridden validation before optimizer execution" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const out = arena.allocator();
    var model: lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var student: TestProgram = .{};
    try predict.setLM(out, &student, model.lm());
    const examples = [_]optimize.Example{
        try optimize.Example.fromJson(out, "{\"question\":\"q\"}", &.{"question"}),
        try optimize.Example.fromJson(out, "{\"question\":\"r\"}", &.{"question"}),
    };
    var optimizer = testOptimizer();
    optimizer.limits.max_examples = 1;
    var change: TestOptimizer = .{};
    const arguments = [_]CompileArgs{.{ .key = "change", .valset = .{ .value = &examples } }};
    try testing.expectError(error.MetaExampleLimit, optimizer.compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, examples[0..1], .{ .seed = 9, .strategy = "change", .valset_ratio = 0, .optimizer_compile_args = &arguments }, .{ .change = &change }));
    try testing.expectEqual(@as(i64, 0), student.stage);
}

test "BetterTogether refuses instruction mutation through a const teacher" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const out = arena.allocator();
    var model: lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var student: TestProgram = .{};
    const teacher: TestProgram = .{};
    const instructions = teacher.first.state.instructions;
    try predict.setLM(out, &student, model.lm());
    const examples = [_]optimize.Example{
        try optimize.Example.fromJson(out, "{\"question\":\"q\"}", &.{"question"}),
    };
    const teachers = [_]finetune.Teacher{finetune.Teacher.init(TestProgram, &teacher)};
    var change: TestOptimizer = .{ .instruction = "must not overwrite const teacher" };
    const result = try testOptimizer().compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, &examples, .{ .seed = 9, .strategy = "change", .valset_ratio = 0, .teachers = &teachers }, .{ .change = &change });
    try testing.expect(result.flag_compilation_error_occurred);
    try testing.expectEqual(@as(usize, 1), result.candidate_programs.len);
    try testing.expectEqualStrings(instructions, teacher.first.state.instructions);
}

test "BetterTogether keeps last unvalidated success restores all LMs and preserves shared signature failure mutations" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const out = arena.allocator();
    var model: lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var replacement: lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "other" };
    var student: TestProgram = .{};
    try predict.setLM(out, &student, model.lm());
    const examples = [_]optimize.Example{
        try optimize.Example.fromJson(out, "{\"question\":\"q\"}", &.{"question"}),
    };
    var change: TestOptimizer = .{ .replacement = replacement.lm() };
    var failure: TestOptimizer = .{ .fail = true, .instruction = "shared class mutation" };
    const result = try testOptimizer().compileWith(TestProgram, out, testing.allocator, testing.io, .{ .arena = out }, &student, &examples, .{ .seed = 9, .strategy = "improve -> improve -> fail", .valset_ratio = 0 }, .{ .improve = &change, .fail = &failure });
    try testing.expect(result.flag_compilation_error_occurred);
    try testing.expectEqual(@as(i64, 2), result.program.stage);
    try testing.expectEqualStrings("improve -> improve", result.candidate_programs[2].strategy);
    for (result.candidate_programs) |candidate| {
        try testing.expect(candidate.score == null);
        try testing.expectEqualStrings("base", candidate.program.first.state.lm.?.modelName().?);
        try testing.expect(candidate.program.first.state.lm.?.sameInstance(candidate.program.second.state.lm.?));
        try testing.expectEqualStrings("shared class mutation", candidate.program.first.state.instructions);
    }
    try testing.expectEqualStrings("shared class mutation", student.first.state.instructions);
}
