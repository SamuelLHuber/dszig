//! DSPy 3.5.0 BootstrapFewShotWithOptuna, with Optuna 4.5.0's default
//! independent integer TPE (not categorical/multivariate search).
//!
//! `compile` requires explicit I/O. A null seed obtains real entropy lazily;
//! set a seed only for reproducible experiments. Native evaluation is serial:
//! num_threads=null/1 is supported, any other explicit value is rejected.
//! Integer-TPE numerical parity remains incomplete: see `tpe_int.zig` for the
//! known logarithm-rounding discrepancy that can change a selected candidate.
//!
//! Like existing native optimizers, programs are values: predictor schema
//! fields are immutable comptime data, while instructions, descriptions,
//! prefixes and demos are copied. Custom non-predictor fields retain the
//! program's normal value semantics. `out` MUST be a caller-owned arena;
//! every returned trial/program/demo and parameter name lives in it. Temporary
//! bootstrap, history and TPE workspaces use `gpa` and never escape compile.
//!
//! Source: dspy/teleprompt/teleprompt_optuna.py, DSPy 3.5.0. The source's
//! constructor max_bootstrapped_demos is informational: compile's REQUIRED
//! max_demos alone configures the one bootstrap pass. Every candidate then
//! selects precisely ONE demo per predictor, regardless of that constructor
//! value. Failed evaluation exceptions abort the study; NaN objectives instead
//! mark trials failed and let optimization continue.

// DSPy-derived portions: MIT License.
// Copyright (c) 2023 Stanford Future Data Systems
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const std = @import("std");
const Allocator = std.mem.Allocator;
const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const value = @import("value.zig");
const tpe = @import("tpe_int.zig");
const Record = value.Record;
const Example = optimize.Example;

pub const BootstrapFewShotWithOptuna = struct {
    metric: optimize.Metric,
    teacher_settings: optimize.TeacherSettings = .{},
    /// Source constructor metadata; compile(max_demos) overrides this value.
    max_bootstrapped_demos: usize = 4,
    max_labeled_demos: usize = 16,
    max_rounds: usize = 1,
    num_candidate_programs: usize = 16,
    num_threads: ?usize = null,
    /// Native counterpart of DSPy settings.max_errors, for bootstrap/evaluate.
    max_errors: usize = 10,
    seed: ?u32 = null,

    pub fn Trial(comptime P: type) type {
        return struct {
            number: usize,
            /// Source trial.user_attrs["program"], retained even for NaN.
            program: P,
            /// One entry per `parameters`, in source predictor traversal order.
            params: []const ?i64,
            /// Failed NaN objectives have no trial value in Optuna.
            score: ?f64,
            state: tpe.State,
        };
    }

    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            trials: []const Trial(P),
            parameters: []const tpe.Param,
            best_trial: usize,
            bootstrap_report: optimize.BootstrapFewShot.Report,
        };
    }

    pub fn compile(
        self: BootstrapFewShotWithOptuna,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        io: std.Io,
        ctx: predict.Context,
        student_in: P,
        teacher_in: ?P,
        max_demos: usize,
        trainset: []const Example,
        valset: ?[]const Example,
    ) !Result(P) {
        if (self.num_threads) |threads| if (threads != 1) return error.UnsupportedNumThreads;
        var workspace: std.heap.ArenaAllocator = .init(gpa);
        defer workspace.deinit();
        const arena = workspace.allocator();
        const student = try copyProgram(P, arena, arena, student_in, true);
        const teacher = if (teacher_in) |given|
            try copyProgram(P, arena, arena, given, false)
        else
            try copyProgram(P, arena, arena, student_in, true);
        const val = if (valset) |given| if (given.len > 0) given else trainset else trainset;

        const bootstrap: optimize.BootstrapFewShot = .{
            .metric = self.metric,
            .max_bootstrapped_demos = max_demos,
            .max_labeled_demos = self.max_labeled_demos,
            .max_rounds = self.max_rounds,
            .max_errors = self.max_errors,
            .teacher_settings = self.teacher_settings,
        };
        var compiled = try bootstrap.compile(P, arena, gpa, ctx, student, teacher, trainset);
        const compiled_predictors = try predict.namedPredictors(arena, &compiled.program);
        const parameters = try out.alloc(tpe.Param, compiled_predictors.len);
        for (compiled_predictors, parameters) |p, *param| {
            const count = p.param.predict.demos.len;
            if (@as(u64, count) > @as(u64, @intCast(tpe.max_exact_bound))) return error.UnsupportedIntegerBounds;
            param.* = .{
                .name = try std.fmt.allocPrint(out, "demo_index_for_{s}", .{p.name}),
                .low = 0,
                // Constructing this invalid range is deliberately not an
                // early failure: zero trials must fail at best_trial lookup,
                // whereas a real first trial fails its first suggest_int.
                .high = @as(i64, @intCast(count)) - 1,
            };
        }
        const trials = try out.alloc(Trial(P), self.num_candidate_programs);
        const history = try arena.alloc(tpe.Trial, self.num_candidate_programs);
        var sampler = if (self.seed) |seed| tpe.Sampler.init(seed) else tpe.Sampler.initUnseeded(io);
        var sampling: std.heap.ArenaAllocator = .init(gpa);
        defer sampling.deinit();
        var best: ?usize = null;
        for (trials, 0..) |*trial, number| {
            _ = sampling.reset(.retain_capacity);
            const scratch = sampling.allocator();
            var program = try copyProgram(P, out, scratch, student, true);
            const program_predictors = try predict.namedPredictors(scratch, &program);
            const drawn = try scratch.alloc(i64, parameters.len);
            try sampler.sample(scratch, parameters, history[0..number], drawn);
            const saved = try out.alloc(?i64, drawn.len);
            for (compiled_predictors, program_predictors, drawn, saved) |all, selected, index, *stored| {
                const demos = try out.alloc(Record, 1);
                demos[0] = try value.cloneRecord(out, &all.param.predict.demos[@intCast(index)]);
                selected.param.predict.demos = demos;
                stored.* = index;
            }
            // No catch-and-continue: study.optimize's default catch=() aborts
            // on objective exceptions, including a later failed candidate.
            const evaluation = try optimize.evaluate(P, gpa, ctx, &program, val, self.metric, self.max_errors);
            const failed = std.math.isNan(evaluation.score);
            const state: tpe.State = if (failed) .fail else .complete;
            trial.* = .{ .number = number, .program = program, .params = saved, .score = if (failed) null else evaluation.score, .state = state };
            history[number] = .{ .params = saved, .value = evaluation.score, .state = state };
            if (!failed) {
                if (best == null or evaluation.score > trials[best.?].score.?) best = number;
            }
        }
        // This check intentionally follows bootstrap and all attempted trials:
        // zero trials and all-NaN studies have no best trial in source Optuna.
        const best_trial = best orelse return error.NoCompletedTrials;
        return .{
            .program = trials[best_trial].program,
            .trials = trials,
            .parameters = parameters,
            .best_trial = best_trial,
            .bootstrap_report = compiled.report,
        };
    }
};

fn copyProgram(comptime P: type, out: Allocator, scratch: Allocator, input: P, reset: bool) Allocator.Error!P {
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
        if (reset) {
            state.reset();
        } else {
            const demos = try out.alloc(Record, state.demos.len);
            for (demos, state.demos) |*dst, *src| dst.* = try value.cloneRecord(out, src);
            state.demos = demos;
        }
    }
    return program;
}

const testing = std.testing;
const signature = @import("signature.zig");
const TestSignature = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

// Same deterministic trace-producing value program as the independent Python
// oracle. Its predictor order deliberately differs from lexical name order.
const TestProgram = struct {
    z: predict.Predict(TestSignature) = .{},
    a: predict.Predict(TestSignature) = .{},
    compiled: bool = false,
    fail: bool = false,
    fail_after_first: bool = false,

    pub const Input = TestSignature.Input;
    pub const Output = struct { answer: []const u8, z: i64, a: i64 };

    pub fn forward(self: *const TestProgram, ctx: predict.Context, input: Input) !Output {
        const z = try selected(self.z.state);
        const a = try selected(self.a.state);
        if (self.fail and std.mem.eql(u8, input.question, "fail") and
            (!self.fail_after_first or z != 0 or a != 2)) return error.LMFailure;
        if (ctx.trace) |trace| {
            const inputs = try value.toRecord(ctx.arena, Input, input);
            const outputs = try value.toRecord(ctx.arena, TestSignature.Output, .{ .answer = input.question });
            try trace.record(&self.z.state, &inputs, &outputs);
            try trace.record(&self.a.state, &inputs, &outputs);
        }
        return .{ .answer = input.question, .z = z, .a = a };
    }

    fn selected(state: predict.PredictState) !i64 {
        if (state.demos.len == 0) return -1;
        return std.fmt.parseInt(i64, state.demos[0].get("answer").?.string, 10);
    }
};

fn sourceMetric(_: Allocator, example: *const Example, maybe_prediction: ?*const Record, trace: ?*const predict.Trace) value.ParseError!f64 {
    if (trace != null) return 1;
    const target = std.fmt.parseInt(i64, example.record.get("question").?.string, 10) catch return 0;
    const prediction = maybe_prediction orelse return error.ParseValue;
    return if (@mod(2 * prediction.get("z").?.integer + prediction.get("a").?.integer, 5) == @mod(target, 5)) 1 else 0;
}

fn tiedMetric(_: Allocator, _: *const Example, _: ?*const Record, _: ?*const predict.Trace) value.ParseError!f64 {
    return 1;
}

fn nanMetric(_: Allocator, _: *const Example, _: ?*const Record, trace: ?*const predict.Trace) value.ParseError!f64 {
    return if (trace != null) 1 else std.math.nan(f64);
}

fn mixedNanMetric(_: Allocator, _: *const Example, maybe_prediction: ?*const Record, trace: ?*const predict.Trace) value.ParseError!f64 {
    if (trace != null) return 1;
    const prediction = maybe_prediction orelse return error.ParseValue;
    if (prediction.get("z").?.integer != 0) return 1;
    return std.math.nan(f64);
}

fn teacherMetric(_: Allocator, _: *const Example, maybe_prediction: ?*const Record, trace: ?*const predict.Trace) value.ParseError!f64 {
    if (trace == null) return 1;
    const prediction = maybe_prediction orelse return error.ParseValue;
    return if (prediction.get("z").?.integer == 88 and prediction.get("a").?.integer == 77) 1 else 0;
}

fn resetTeacherMetric(_: Allocator, _: *const Example, maybe_prediction: ?*const Record, trace: ?*const predict.Trace) value.ParseError!f64 {
    if (trace == null) return 1;
    const prediction = maybe_prediction orelse return error.ParseValue;
    return if (prediction.get("z").?.integer == -1 and prediction.get("a").?.integer == -1) 1 else 0;
}

fn testExamples(arena: Allocator, values: []const []const u8) ![]Example {
    const rows = try arena.alloc(Example, values.len);
    for (rows, values) |*row, question| {
        var record: Record = .empty;
        try record.put(arena, "question", .{ .string = try arena.dupe(u8, question) });
        try record.put(arena, "answer", .{ .string = try arena.dupe(u8, question) });
        row.* = .{ .record = record, .input_keys = &.{"question"} };
    }
    return rows;
}

const source_choices = [_][2]i64{
    .{ 0, 2 }, .{ 2, 0 }, .{ 0, 1 }, .{ 2, 1 }, .{ 0, 1 },
    .{ 0, 4 }, .{ 4, 0 }, .{ 3, 2 }, .{ 4, 3 }, .{ 2, 3 },
    .{ 1, 1 }, .{ 1, 4 }, .{ 1, 4 }, .{ 3, 3 }, .{ 3, 1 },
    .{ 1, 4 }, .{ 2, 2 }, .{ 0, 3 }, .{ 3, 0 }, .{ 1, 1 },
};

// Expected values are independent real-DSPy receipts, not generated by native
// code. See tools/dspy_reference/bootstrap_optuna_oracle.py, seed=9.
test "BootstrapOptuna selects one demo in predictor order and retains every trial safely" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var inputs: std.heap.ArenaAllocator = .init(testing.allocator);
    var inputs_live = true;
    defer if (inputs_live) inputs.deinit();
    const arena = inputs.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const valset = try testExamples(arena, &.{ "0", "0", "1" });
    var student: TestProgram = .{};
    student.z.state.demos = &.{(try testExamples(arena, &.{"99"}))[0].record};
    student.z.state.instructions = try arena.dupe(u8, student.z.state.instructions);
    const original = student.z.state.demos[0];
    const optimizer: BootstrapFewShotWithOptuna = .{
        .metric = sourceMetric,
        .seed = 9,
        .max_bootstrapped_demos = 99,
        .max_labeled_demos = 5,
        .num_candidate_programs = 20,
        .num_threads = 1,
    };
    const result = try optimizer.compile(TestProgram, out.allocator(), testing.allocator, testing.io, ctx, student, null, 3, trainset, valset);
    try testing.expect(try value.recordEqual(&original, &student.z.state.demos[0]));
    try testing.expectEqual(@as(usize, 3), result.best_trial);
    try testing.expectEqual(@as(usize, 3), result.bootstrap_report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), silent.index);
    try testing.expect(!result.program.compiled);
    try testing.expectEqualStrings("demo_index_for_z", result.parameters[0].name);
    try testing.expectEqualStrings("demo_index_for_a", result.parameters[1].name);
    // Destroy BOTH original student metadata and training/validation records.
    // No returned trial may retain a pointer into this arena or the compile
    // workspaces that have already been released.
    inputs.deinit();
    inputs_live = false;
    const scores = [_]f64{ 0, 0, 33.33, 66.67, 33.33, 66.67, 0, 0, 33.33, 0, 0, 0, 0, 0, 33.33, 0, 33.33, 0, 66.67, 0 };
    const z_pool = [_][]const u8{ "0", "1", "2", "5", "4" };
    const a_pool = [_][]const u8{ "0", "1", "2", "3", "5" };
    for (result.trials, source_choices, scores, 0..) |trial, chosen, score, i| {
        try testing.expectEqual(i, trial.number);
        try testing.expectEqual(tpe.State.complete, trial.state);
        try testing.expectEqual(score, trial.score.?);
        try testing.expectEqual(@as(?i64, chosen[0]), trial.params[0]);
        try testing.expectEqual(@as(?i64, chosen[1]), trial.params[1]);
        try testing.expectEqual(@as(usize, 1), trial.program.z.state.demos.len);
        try testing.expectEqual(@as(usize, 1), trial.program.a.state.demos.len);
        try testing.expectEqualStrings(z_pool[@intCast(chosen[0])], trial.program.z.state.demos[0].get("answer").?.string);
        try testing.expectEqualStrings(a_pool[@intCast(chosen[1])], trial.program.a.state.demos[0].get("answer").?.string);
        try testing.expectEqual(chosen[0] < 3, trial.program.z.state.demos[0].contains("augmented"));
        try testing.expectEqual(chosen[1] < 3, trial.program.a.state.demos[0].contains("augmented"));
        try testing.expectEqualStrings(TestSignature.instructions, trial.program.z.state.instructions);
    }
    try testing.expectEqualStrings("2", result.program.z.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("1", result.program.a.state.demos[0].get("answer").?.string);
}

test "BootstrapOptuna absent and empty validation both fall back to training" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = sourceMetric, .seed = 9, .max_labeled_demos = 5, .num_candidate_programs = 20 };
    const absent = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, null);
    const empty = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, &.{});
    for (absent.trials, empty.trials, source_choices, 0..) |a, b, choices, i| {
        try testing.expectEqualSlices(?i64, a.params, b.params);
        try testing.expectEqual(@as(?i64, choices[0]), a.params[0]);
        try testing.expectEqual(@as(?i64, choices[1]), a.params[1]);
        const expected: f64 = if (i == 3 or i == 5 or i == 18) 33.33 else 16.67;
        try testing.expectEqual(expected, a.score.?);
        try testing.expectEqual(a.score, b.score);
    }
    try testing.expectEqual(@as(usize, 3), absent.best_trial);
    try testing.expectEqual(absent.best_trial, empty.best_trial);
}

test "BootstrapOptuna ties keep the first program and do not mutate the teacher" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    var teacher: TestProgram = .{};
    teacher.z.state.demos = &.{(try testExamples(arena, &.{"88"}))[0].record};
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = tiedMetric, .seed = 9, .max_labeled_demos = 5, .num_candidate_programs = 20 };
    const result = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, teacher, 3, trainset, null);
    try testing.expectEqualStrings("88", teacher.z.state.demos[0].get("answer").?.string);
    try testing.expectEqual(@as(usize, 0), result.best_trial);
    try testing.expectEqualStrings("0", result.program.z.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("2", result.program.a.state.demos[0].get("answer").?.string);
    const post_startup = [_][2]i64{ .{ 1, 2 }, .{ 1, 0 }, .{ 1, 0 }, .{ 3, 4 }, .{ 3, 3 }, .{ 1, 1 }, .{ 2, 2 }, .{ 0, 0 }, .{ 3, 3 }, .{ 1, 1 } };
    for (result.trials[10..], post_startup) |trial, choices| {
        try testing.expectEqual(@as(?i64, choices[0]), trial.params[0]);
        try testing.expectEqual(@as(?i64, choices[1]), trial.params[1]);
        try testing.expectEqual(@as(?f64, 100), trial.score);
    }
}

test "BootstrapOptuna zero rounds preserves labeled candidates and defaults remain source defaults" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const valset = try testExamples(arena, &.{ "0", "0", "1" });
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = sourceMetric, .seed = 9, .max_labeled_demos = 5, .max_rounds = 0, .num_candidate_programs = 20 };
    const result = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, valset);
    try testing.expectEqual(@as(usize, 0), result.bootstrap_report.attempts);
    try testing.expectEqual(@as(usize, 0), result.bootstrap_report.bootstrapped);
    try testing.expectEqual(@as(usize, 9), result.best_trial);
    try testing.expectEqualStrings("4", result.program.z.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("2", result.program.a.state.demos[0].get("answer").?.string);
    for (result.trials) |trial| {
        try testing.expect(!trial.program.z.state.demos[0].contains("augmented"));
        try testing.expect(!trial.program.a.state.demos[0].contains("augmented"));
    }
    const defaults: BootstrapFewShotWithOptuna = .{ .metric = sourceMetric, .seed = 9 };
    const default_result = try defaults.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, valset);
    try testing.expectEqual(@as(usize, 16), default_result.trials.len);
    try testing.expectEqual(@as(usize, 3), default_result.bootstrap_report.bootstrapped);
    try testing.expectEqual(@as(i64, 5), default_result.parameters[0].high);
    try testing.expectEqual(@as(i64, 5), default_result.parameters[1].high);
    try testing.expectEqual(@as(usize, 0), default_result.best_trial);
    const default_choices = [_][2]i64{
        .{ 0, 3 }, .{ 2, 0 }, .{ 0, 1 }, .{ 2, 1 }, .{ 0, 2 }, .{ 1, 5 },
        .{ 5, 0 }, .{ 4, 3 }, .{ 5, 4 }, .{ 3, 4 }, .{ 1, 3 }, .{ 2, 2 },
        .{ 3, 1 }, .{ 1, 1 }, .{ 0, 2 }, .{ 2, 4 },
    };
    const default_scores = [_]f64{ 66.67, 0, 33.33, 66.67, 0, 66.67, 33.33, 0, 66.67, 0, 0, 33.33, 33.33, 0, 0, 0 };
    for (default_result.trials, default_choices, default_scores) |trial, choices, score| {
        try testing.expectEqual(@as(?i64, choices[0]), trial.params[0]);
        try testing.expectEqual(@as(?i64, choices[1]), trial.params[1]);
        try testing.expectEqual(@as(?f64, score), trial.score);
    }
}

test "BootstrapOptuna source errors precede study lookup and NaN never wins" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = sourceMetric, .seed = 9, .num_candidate_programs = 0 };
    try testing.expectError(error.StudentCompiled, optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{ .compiled = true }, null, 3, &.{}, null));
    var teacher: TestProgram = .{};
    teacher.a.state.instructions = "different";
    try testing.expectError(error.ProgramMismatch, optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, teacher, 3, &.{}, null));
    try testing.expectError(error.EmptyBootstrapTrainset, optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, &.{}, null));
    try testing.expectError(error.NoCompletedTrials, optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, null));
    var empty_demos = optimizer;
    empty_demos.max_labeled_demos = 0;
    try testing.expectError(error.NoCompletedTrials, empty_demos.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 0, trainset, null));
    empty_demos.num_candidate_programs = 1;
    try testing.expectError(error.InvalidBounds, empty_demos.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 0, trainset, null));
    const nan_optimizer: BootstrapFewShotWithOptuna = .{ .metric = nanMetric, .seed = 9, .num_candidate_programs = 3 };
    try testing.expectError(error.NoCompletedTrials, nan_optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, null));
    var unsupported = optimizer;
    unsupported.num_threads = 2;
    try testing.expectError(error.UnsupportedNumThreads, unsupported.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, null));
}

test "BootstrapOptuna catches individual evaluation failures but never swallows an aborted later trial" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const partial_val = try testExamples(arena, &.{"fail"});
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = tiedMetric, .seed = 9, .max_labeled_demos = 5, .num_candidate_programs = 3 };
    const partial = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{ .fail = true }, null, 3, trainset, partial_val);
    try testing.expectEqual(@as(usize, 3), partial.trials.len);
    try testing.expectEqual(@as(usize, 0), partial.best_trial);
    for (partial.trials) |trial| try testing.expectEqual(@as(?f64, 0), trial.score);
    const abort_val = try testExamples(arena, &.{ "fail", "fail", "fail", "fail", "fail", "fail", "fail", "fail", "fail", "fail" });
    try testing.expectError(error.LMFailure, optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{ .fail = true }, null, 3, trainset, abort_val));
    try testing.expectError(error.LMFailure, optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{ .fail = true, .fail_after_first = true }, null, 3, trainset, abort_val));
}

test "BootstrapOptuna keeps supplied teacher demos but resets the implicit teacher" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    var teacher: TestProgram = .{ .compiled = true };
    teacher.z.state.demos = &.{(try testExamples(arena, &.{"88"}))[0].record};
    teacher.a.state.demos = &.{(try testExamples(arena, &.{"77"}))[0].record};
    var student: TestProgram = .{};
    student.z.state.demos = &.{(try testExamples(arena, &.{"99"}))[0].record};
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = teacherMetric, .seed = 9, .max_labeled_demos = 0, .num_candidate_programs = 1 };
    const supplied = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, student, teacher, 1, trainset, null);
    try testing.expectEqual(@as(usize, 1), supplied.bootstrap_report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), supplied.bootstrap_report.attempts);
    try testing.expectEqualStrings("0", supplied.program.z.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("88", teacher.z.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("77", teacher.a.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("99", student.z.state.demos[0].get("answer").?.string);
    var implicit = optimizer;
    implicit.metric = resetTeacherMetric;
    const reset = try implicit.compile(TestProgram, arena, testing.allocator, testing.io, ctx, student, null, 1, trainset, null);
    try testing.expectEqual(@as(usize, 1), reset.bootstrap_report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), reset.bootstrap_report.attempts);
    try testing.expectEqual(@as(?f64, 100), reset.trials[0].score);
}

test "BootstrapOptuna unseeded compilation uses supplied entropy and seeded probes do not" {
    const NoEntropy = struct {
        fn fill(_: ?*anyopaque, _: []u8) std.Io.RandomSecureError!void {
            return error.EntropyUnavailable;
        }
    };
    var vtable = testing.io.vtable.*;
    vtable.randomSecure = NoEntropy.fill;
    const io: std.Io = .{ .userdata = null, .vtable = &vtable };
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = tiedMetric };
    try testing.expectError(error.EntropyUnavailable, optimizer.compile(TestProgram, arena, testing.allocator, io, ctx, .{}, null, 3, trainset, null));
    var seeded = optimizer;
    seeded.seed = 9;
    const reproducible = try seeded.compile(TestProgram, arena, testing.allocator, io, ctx, .{}, null, 3, trainset, null);
    try testing.expectEqual(@as(usize, 16), reproducible.trials.len);
    var singleton = optimizer;
    singleton.max_labeled_demos = 0;
    const fixed = try singleton.compile(TestProgram, arena, testing.allocator, io, ctx, .{}, null, 1, trainset, null);
    try testing.expectEqual(@as(usize, 16), fixed.trials.len);
    var no_trials = optimizer;
    no_trials.num_candidate_programs = 0;
    try testing.expectError(error.NoCompletedTrials, no_trials.compile(TestProgram, arena, testing.allocator, io, ctx, .{}, null, 3, trainset, null));
}

test "BootstrapOptuna retains NaN trial programs without advancing the startup count" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: predict.Context = .{ .arena = arena, .lm = silent.lm() };
    const trainset = try testExamples(arena, &.{ "0", "1", "2", "3", "4", "5" });
    const valset = try testExamples(arena, &.{ "0", "0", "1" });
    const optimizer: BootstrapFewShotWithOptuna = .{ .metric = mixedNanMetric, .seed = 9, .max_labeled_demos = 5, .num_candidate_programs = 30 };
    const result = try optimizer.compile(TestProgram, arena, testing.allocator, testing.io, ctx, .{}, null, 3, trainset, valset);
    try testing.expectEqual(@as(usize, 1), result.best_trial);
    // Ten completed trials are reached at trial 13, not at trial 9.
    const startup_choices = [_][2]i64{
        .{ 0, 2 }, .{ 2, 0 }, .{ 0, 1 }, .{ 2, 1 }, .{ 0, 1 }, .{ 0, 4 },
        .{ 4, 0 }, .{ 3, 2 }, .{ 4, 3 }, .{ 2, 3 }, .{ 1, 3 }, .{ 4, 2 },
        .{ 4, 4 }, .{ 4, 0 },
    };
    for (result.trials, 0..) |trial, i| {
        const expected: [2]i64 = if (i < startup_choices.len) startup_choices[i] else .{ 0, 1 };
        try testing.expectEqual(@as(?i64, expected[0]), trial.params[0]);
        try testing.expectEqual(@as(?i64, expected[1]), trial.params[1]);
        const failed = expected[0] == 0;
        try testing.expectEqual(if (failed) tpe.State.fail else tpe.State.complete, trial.state);
        try testing.expectEqual(if (failed) @as(?f64, null) else @as(?f64, 100), trial.score);
        try testing.expectEqual(@as(usize, 1), trial.program.z.state.demos.len);
        try testing.expectEqual(@as(usize, 1), trial.program.a.state.demos.len);
        if (failed) try testing.expectEqualStrings("0", trial.program.z.state.demos[0].get("answer").?.string);
    }
}
