//! Native `dspy.BestOfN`: independent attempts, typed rewards, selected traces.
//!
//! ```zig
//! const Program = dszig.Predict(MySignature);
//! const Best = dszig.BestOfN(Program);
//! const Rewards = struct {
//!     fn score(_: dszig.Context, _: *const Program.Input, out: *const Program.Output) anyerror!f64 {
//!         return if (out.answer.len != 0) 1.0 else 0.0;
//!     }
//! };
//! var best = Best.init(.{}, .{ .n = 3, .reward_fn = Rewards.score, .threshold = .{ .float = 1.0 } });
//! const result: ?Program.Output = try best.forward(ctx, input);
//! ```
//!
//! Like other native modules, program copies share immutable slices; mutable
//! program state must be stored by value, not behind shared mutable pointers.
//! Attempt programs, outputs, and selected trace predictor pointers live in
//! `ctx.arena`. That arena must outlive consumption of the output and trace.
//! Traces are native record snapshots, not live aliases of Python predictions.
//! `forward` takes a mutable wrapper because DSPy's failure budget persists
//! between calls. Tolerated failures do not print DSPy's stdout diagnostics.
//! Rollout IDs support JSON numbers, booleans, and boxed
//! integers/floats; other Python `__add__` behavior is not modeled and
//! unsupported IDs return `error.UnsupportedRolloutId`.
//!
//! Upstream (DSPy 3.5.0): dspy/predict/best_of_n.py

const std = @import("std");
const lm_mod = @import("lm.zig");
const predict = @import("predict.zig");
const value = @import("value.zig");
const Context = predict.Context;
const Trace = predict.Trace;
const Value = @import("runtime_value.zig").Value;
const Allocator = std.mem.Allocator;

/// The reward receives the original LM/options with the isolated attempt
/// trace. Its predictor calls are not included in the selected trace.
/// Callback and program errors retain their original error identities.
pub fn BestOfN(comptime P: type) type {
    return struct {
        const Self = @This();
        pub const Input = P.Input;
        pub const Prediction = P.Output;
        pub const Output = ?Prediction;
        pub const Reward = *const fn (Context, *const Input, *const Prediction) anyerror!f64;
        pub const Options = struct {
            n: isize,
            reward_fn: Reward,
            /// Numeric scalar; use .number_string for integers beyond i64.
            threshold: Value,
            fail_count: ?i128 = null,
        };

        module: P,
        n: isize,
        reward_fn: Reward,
        threshold: Value,
        fail_count: i128,
        compiled: bool = false,

        pub fn init(module: P, options: Options) Self {
            const count = options.fail_count orelse 0;
            return .{
                .module = module,
                .n = options.n,
                .reward_fn = options.reward_fn,
                .threshold = options.threshold,
                .fail_count = if (count == 0) options.n else count,
            };
        }

        /// After LM setup, no best prediction is a genuine null, including
        /// all NaN/-inf rewards and zero/negative attempt counts. Reaching a
        /// threshold does not itself guarantee that a prediction was selected.
        pub fn forward(self: *Self, ctx: Context, input: Input) anyerror!Output {
            const rollout = try rolloutContext(ctx, &self.module);
            if (self.n <= 0) return null;
            var best: Output = null;
            var best_steps: []const Trace.Step = &.{};
            var best_reward = -std.math.inf(f64);
            for (0..@as(usize, @intCast(self.n))) |idx| {
                // Like LM.copy/deepcopy upstream, setup is outside the
                // caught program/reward region and never consumes budget.
                const module = try ctx.arena.create(P);
                module.* = self.module;
                var scope: predict.PredictRollout = undefined;
                const attempt_ctx = try rollout.attempt(idx, try predict.namedPredictors(ctx.arena, module), &scope);
                const attempt = self.runAttempt(module, attempt_ctx, ctx, &input) catch |err| {
                    // Python's asyncio.CancelledError bypasses `except Exception`.
                    // A native stream sink's .stop also ends the call, not just
                    // the active attempt.
                    if (err == error.Canceled or err == error.StreamStopped) return err;
                    // Deliberately not >=, and deliberately before decrement.
                    // Negative budgets therefore propagate immediately.
                    if (@as(i128, @intCast(idx)) > self.fail_count) return err;
                    self.fail_count -= 1;
                    continue;
                };
                if (attempt.reward > best_reward) {
                    best_reward = attempt.reward;
                    best = attempt.output;
                    best_steps = attempt.steps;
                }
                // DSPy selects the best prediction before comparing its
                // reward to the threshold. A comparison failure spends the
                // same budget as a program or reward failure, but does not
                // undo that selection.
                const reached_threshold = value.floatAtLeast(attempt.reward, self.threshold) catch |err| {
                    if (@as(i128, @intCast(idx)) > self.fail_count) return err;
                    self.fail_count -= 1;
                    continue;
                };
                if (reached_threshold) break;
            }
            if (ctx.trace) |trace| {
                for (best_steps) |step| {
                    // Trace inputs are already serialized. Calling record
                    // again would format native tool catalogs a second time.
                    try trace.steps.append(trace.arena, try step.clone(trace.arena));
                }
            }
            return best;
        }

        const Attempt = struct { output: Prediction, reward: f64, steps: []const Trace.Step };

        fn runAttempt(self: *const Self, module: *P, ctx: Context, reward_ctx: Context, input: *const Input) anyerror!Attempt {
            var trace: Trace = .{ .arena = ctx.arena };
            defer trace.steps.deinit(ctx.arena);
            var isolated = ctx;
            isolated.trace = &trace;
            const output = try module.forward(isolated, input.*);
            // A slice into the list is not a snapshot: reward calls can grow
            // and reallocate it. Copy before invoking the reward callback.
            const steps: []const Trace.Step = if (ctx.trace != null)
                try ctx.arena.dupe(Trace.Step, trace.steps.items)
            else
                &.{};
            var reward_context = reward_ctx;
            reward_context.trace = &trace;
            const reward = try self.reward_fn(reward_context, input, &output);
            return .{ .output = output, .reward = reward, .steps = steps };
        }
    };
}

const RolloutContext = struct {
    base: Context,
    lm: lm_mod.LM,
    start: Value,

    pub fn attempt(
        self: RolloutContext,
        index: usize,
        names: []const predict.NamedParam,
        scope: *predict.PredictRollout,
    ) RolloutError!Context {
        var ctx = self.base;
        const id = try rolloutAt(ctx.arena, self.start, index);
        if (names.len == 0) {
            // Native extension: direct-engine modules have no predictors to
            // bind, so retain their Context-based sampling.
            if (ctx.lm_override) |model| ctx.lm = model;
            ctx.before_settings_rollout = null;
            ctx.lm_options.rollout_id = id;
            ctx.lm_options.temperature = 1.0;
        } else {
            scope.* = .{
                .parent = ctx.predict_rollout,
                .predictors = names,
                .lm = self.lm,
                .id = id,
            };
            ctx.predict_rollout = scope;
        }
        return ctx;
    }
};

/// Resolve the LM before entering the attempt loop, even for an empty range.
/// A settings refresh must not become the starting rollout of a bound LM.
pub fn rolloutContext(base: Context, module: anytype) !RolloutContext {
    const names = try predict.namedPredictors(base.arena, module);
    // Nested retries choose from the outer attempt's copied model. A per-call
    // LM override goes to the predictor only; it never becomes the retry base.
    const enclosing = if (names.len == 0) null else predict.PredictRollout.find(base.predict_rollout, names[0].param.predict);
    const binding = if (enclosing) |scope| scope.lm else try predict.getLMFromPredictors(names);
    const lm = binding orelse base.lm;
    if (names.len != 0 and lm.sameInstance(lm_mod.LM.unconfigured)) return error.NoLM;
    const start = if (enclosing) |scope|
        scope.id
    else if (binding != null and base.before_settings_rollout != null)
        base.before_settings_rollout.?.rollout_id
    else
        base.lm_options.rollout_id;
    return .{ .base = base, .lm = lm, .start = start orelse .{ .integer = 0 } };
}

/// Python numeric addition: bool and integer subclasses have an integral
/// result; true float boxes behave like unboxed floats. Do not coerce objects
/// merely convertible to numbers. Integer arithmetic stays exact past i128.
const RolloutError = Allocator.Error || error{UnsupportedRolloutId};

pub fn rolloutAt(arena: Allocator, start: Value, index: usize) RolloutError!Value {
    const numeric = if (start == .custom) blk: {
        if (start.custom.integerValue()) |integer| break :blk integer;
        const builtin = @import("python_value.zig").view(start) orelse return error.UnsupportedRolloutId;
        if (builtin != .float) return error.UnsupportedRolloutId;
        break :blk Value{ .float = builtin.float };
    } else start;
    return switch (numeric) {
        .integer => |n| integerRollout(arena, n, index),
        .bool => |b| integerRollout(arena, @intFromBool(b), index),
        .float => |n| .{ .float = n + @as(f64, @floatFromInt(index)) },
        .number_string => |text| numberStringRollout(arena, text, index),
        else => error.UnsupportedRolloutId,
    };
}

fn integerRollout(arena: Allocator, start: i128, index: usize) RolloutError!Value {
    const sum = @addWithOverflow(start, @as(i128, @intCast(index)));
    if (sum[1] != 0) {
        return numberStringRollout(arena, try std.fmt.allocPrint(arena, "{d}", .{start}), index);
    }
    if (std.math.cast(i64, sum[0])) |n| return .{ .integer = n };
    return .{ .number_string = try std.fmt.allocPrint(arena, "{d}", .{sum[0]}) };
}

fn numberStringRollout(arena: Allocator, text: []const u8, index: usize) RolloutError!Value {
    if (text.len == 0) return error.UnsupportedRolloutId;
    const negative = text[0] == '-';
    const digits = text[@intFromBool(negative)..];
    if (digits.len == 0) return error.UnsupportedRolloutId;
    var integral = true;
    for (digits) |digit| integral = integral and std.ascii.isDigit(digit);
    if (!integral) {
        // JSON number_string may also carry a decimal/exponent spelling.
        const n = std.fmt.parseFloat(f64, text) catch return error.UnsupportedRolloutId;
        return .{ .float = n + @as(f64, @floatFromInt(index)) };
    }
    if (std.fmt.parseInt(i128, text, 10)) |n| {
        const sum = @addWithOverflow(n, @as(i128, @intCast(index)));
        if (sum[1] == 0) return integerRollout(arena, n, index);
    } else |_| {}

    // The magnitude exceeds i128 (or positive addition overflowed it), so
    // it is greater than any usize delta. Negative values cannot cross zero.
    const storage = try arena.alloc(u8, digits.len + 2);
    var pos = storage.len;
    var remainder: u128 = index;
    var source_pos = digits.len;
    while (source_pos > 0) {
        source_pos -= 1;
        var digit: u128 = digits[source_pos] - '0';
        if (negative) {
            const sub = remainder % 10;
            remainder /= 10;
            if (digit < sub) {
                digit += 10;
                remainder += 1;
            }
            digit -= sub;
        } else {
            remainder += digit;
            digit = remainder % 10;
            remainder /= 10;
        }
        pos -= 1;
        storage[pos] = '0' + @as(u8, @intCast(digit));
    }
    if (remainder != 0) {
        std.debug.assert(!negative and remainder == 1);
        pos -= 1;
        storage[pos] = '1';
    }
    while (pos + 1 < storage.len and storage[pos] == '0') pos += 1;
    if (negative) {
        pos -= 1;
        storage[pos] = '-';
    }
    return .{ .number_string = storage[pos..] };
}

const testing = std.testing;
const TestSignature = @import("signature.zig").Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: usize };
});

const TestProgram = struct {
    const Self = @This();
    pub const Input = struct {
        scores: []const f64,
        program_failures: []const usize = &.{},
        reward_failures: []const usize = &.{},
        reward_trace: bool = false,
        start: f64 = 0,
    };
    pub const Output = struct {
        attempt: usize,
        counter: usize,
        rollout: Value,
        temperature: ?f64,
    };

    predictor: predict.Predict(TestSignature) = .{},
    counter: usize = 0,
    compiled: bool = false,

    pub fn forward(self: *Self, ctx: Context, input: Input) !Output {
        const options = (try self.predictor.state.resolveContext(ctx)).lm_options;
        const rollout = options.rollout_id.?;
        const number = try value.fromValue(ctx.arena, f64, rollout);
        const attempt: usize = @intFromFloat(number - input.start);
        self.counter += 1;
        self.predictor.state.instructions = try std.fmt.allocPrint(ctx.arena, "attempt-{d}", .{attempt});
        if (std.mem.indexOfScalar(usize, input.program_failures, attempt) != null) return error.ProgramFailure;
        var inputs: value.Record = .empty;
        try inputs.put(ctx.arena, "question", .{ .string = "q" });
        var outputs: value.Record = .empty;
        try outputs.put(ctx.arena, "answer", .{ .integer = @intCast(attempt) });
        if (ctx.trace) |trace| try trace.record(&self.predictor.state, &inputs, &outputs);
        return .{
            .attempt = attempt,
            .counter = self.counter,
            .rollout = rollout,
            .temperature = options.temperature,
        };
    }

    fn reward(ctx: Context, input: *const Input, output: *const Output) anyerror!f64 {
        try testing.expectEqual(@as(usize, 1), output.counter);
        try testing.expectEqual(@as(usize, 1), ctx.trace.?.steps.items.len);
        if (input.reward_trace) {
            const reward_predictor: predict.Predict(TestSignature) = .{};
            var record: value.Record = .empty;
            try record.put(ctx.arena, "answer", .{ .integer = 99 });
            const empty: value.Record = .empty;
            for (0..40) |_| try ctx.trace.?.record(&reward_predictor.state, &empty, &record);
        }
        if (std.mem.indexOfScalar(usize, input.reward_failures, output.attempt) != null) return error.RewardFailure;
        return input.scores[output.attempt];
    }

    fn rewardWithPredict(ctx: Context, input: *const Input, output: *const Output) anyerror!f64 {
        const score = try reward(ctx, input, output);
        const independent: predict.Predict(TestSignature) = .{};
        _ = try independent.forward(ctx, .{ .question = "reward" });
        return score;
    }
};

test "BestOfN early stop uses numeric rollout overrides and independent program copies" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var best = BestOfN(TestProgram).init(.{}, .{
        .n = 3,
        .reward_fn = TestProgram.reward,
        .threshold = .{ .float = 0.8 },
    });
    const ctx: Context = .{
        .arena = arena,
        .lm = script.lm(),
        .lm_options = .{ .rollout_id = .{ .integer = 7 }, .temperature = 0.2 },
    };
    const result = (try best.forward(ctx, .{ .scores = &.{ 0.25, 0.8, 1.0 }, .start = 7 })).?;
    try testing.expectEqual(@as(usize, 1), result.attempt);
    try testing.expectEqual(@as(i64, 8), result.rollout.integer);
    try testing.expectEqual(@as(?f64, 1.0), result.temperature);
    try testing.expectEqual(@as(usize, 0), best.module.counter);
    try testing.expectEqual(@as(?f64, 0.2), ctx.lm_options.temperature);
    try testing.expectEqual(@as(i128, 3), best.fail_count);
    try testing.expectEqualStrings("module.predictor", (try predict.namedPredictors(arena, &best))[0].name);
}

test "BestOfN distinguishes exact integer thresholds from rounded binary64" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const QA = predict.Predict(TestSignature);
    const Reward = struct {
        fn score(_: Context, _: *const QA.Input, output: *const QA.Output) anyerror!f64 {
            const scores = [_]f64{ 9_007_199_254_740_992, 9_007_199_254_740_994 };
            return scores[@intCast(output.answer)];
        }
    };
    const cases = [_]struct { threshold: Value, winner: usize, calls: usize }{
        .{ .threshold = .{ .integer = 9_007_199_254_740_993 }, .winner = 1, .calls = 2 },
        .{ .threshold = .{ .float = 9_007_199_254_740_992 }, .winner = 0, .calls = 1 },
    };
    for (cases) |case| {
        var script: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>0</answer>", "<answer>1</answer>" } };
        var trace: Trace = .{ .arena = arena };
        var best = BestOfN(QA).init(.{}, .{ .n = 2, .reward_fn = Reward.score, .threshold = case.threshold });
        const answer = (try best.forward(.{ .arena = arena, .lm = script.lm(), .trace = &trace }, .{ .question = "Q" })).?;
        try testing.expectEqual(case.calls, script.index);
        try testing.expectEqual(case.winner, answer.answer);
        try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
        try testing.expectEqual(@as(i64, @intCast(case.winner)), trace.steps.items[0].outputs.get("answer").?.integer);
    }
}

test "BestOfN first ties and selected trace survive reward trace reallocation and losing attempts" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var trace: Trace = .{ .arena = arena };
    const existing: predict.Predict(TestSignature) = .{};
    const empty: value.Record = .empty;
    try trace.record(&existing.state, &empty, &empty);
    var best = BestOfN(TestProgram).init(.{}, .{
        .n = 3,
        .reward_fn = TestProgram.reward,
        .threshold = .{ .float = std.math.inf(f64) },
    });
    const ctx: Context = .{ .arena = arena, .lm = script.lm(), .trace = &trace };
    const result = (try best.forward(ctx, .{ .scores = &.{ 0.5, 0.5, 0.25 }, .reward_trace = true })).?;
    try testing.expectEqual(@as(usize, 0), result.attempt);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].predictor == &existing.state);
    try testing.expect(trace.steps.items[1].predictor != &best.module.predictor.state);
    try testing.expectEqualStrings("attempt-0", trace.steps.items[1].predictor.instructions);
    try testing.expectEqual(@as(i64, 0), trace.steps.items[1].outputs.get("answer").?.integer);
}

test "BestOfN reward predictors use original LM options and never enter the selected trace" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var script: lm_mod.ScriptedLM = .{ .responses = &.{"<answer>\n9\n</answer>"}, .history = &history };
    var trace: Trace = .{ .arena = arena };
    var best = BestOfN(TestProgram).init(.{}, .{
        .n = 3,
        .reward_fn = TestProgram.rewardWithPredict,
        .threshold = .{ .float = 1.0 },
    });
    const result = (try best.forward(.{
        .arena = arena,
        .lm = script.lm(),
        .trace = &trace,
        .lm_options = .{ .rollout_id = .{ .integer = 7 }, .temperature = 0.35 },
    }, .{ .scores = &.{ 1.0, 0.0, 0.0 }, .start = 7 })).?;
    try testing.expectEqual(@as(?f64, 1.0), result.temperature);
    try testing.expectEqual(@as(usize, 1), history.requests.items.len);
    try testing.expectEqual(@as(?f64, 0.35), history.requests.items[0].options.temperature);
    try testing.expectEqual(@as(i64, 7), history.requests.items[0].options.rollout_id.?.integer);
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    try testing.expectEqual(@as(i64, 0), trace.steps.items[0].outputs.get("answer").?.integer);
}

test "BestOfN bound model receives attempt options instead of bootstrap settings refresh" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var bound: lm_mod.ScriptedLM = .{ .responses = &.{"<answer>\n9\n</answer>"}, .history = &history };
    var settings: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const QA = predict.Predict(TestSignature);
    const Reward = struct {
        fn score(_: Context, _: *const QA.Input, output: *const QA.Output) anyerror!f64 {
            return if (output.answer == 9) 1 else 0;
        }
    };
    var qa: QA = .{};
    qa.state.lm = bound.lm();
    var best = BestOfN(QA).init(qa, .{ .n = 2, .reward_fn = Reward.score, .threshold = .{ .integer = 1 } });
    const output = try best.forward(.{
        .arena = arena,
        .lm = settings.lm(),
        .lm_options = .{ .rollout_id = .{ .string = "bootstrap-refresh" }, .temperature = 0.8 },
        .before_settings_rollout = .{ .rollout_id = .{ .integer = 7 }, .temperature = 0.35 },
    }, .{ .question = "test" });
    try testing.expectEqual(@as(usize, 9), output.?.answer);
    try testing.expectEqual(@as(i64, 7), history.requests.items[0].options.rollout_id.?.integer);
    try testing.expectEqual(@as(?f64, 1.0), history.requests.items[0].options.temperature);
}

test "BestOfN forwards per-call LM choices without using them as the retry base" {
    const QA = predict.Predict(TestSignature);
    const Reward = struct {
        fn score(_: Context, _: *const QA.Input, out: *const QA.Output) anyerror!f64 {
            return @floatFromInt(out.answer);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var override: lm_mod.ScriptedLM = .{
        .responses = &.{ "<answer>7</answer>", "<answer>8</answer>" },
        .history = &history,
    };
    var settings: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var wrapper = BestOfN(QA).init(.{}, .{
        .n = 2,
        .reward_fn = Reward.score,
        .threshold = .{ .integer = 8 },
    });
    var ctx: Context = .{
        .arena = arena,
        .lm_override = override.lm(),
        .lm_options = .{ .rollout_id = .{ .integer = 97 }, .temperature = 0.5 },
    };
    try testing.expectError(error.NoLM, wrapper.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(usize, 0), history.requests.items.len);
    try testing.expectEqual(@as(i128, 2), wrapper.fail_count);
    ctx.lm = settings.lm();
    const result = (try wrapper.forward(ctx, .{ .question = "q" })).?;
    try testing.expectEqual(@as(usize, 8), result.answer);
    try testing.expectEqual(@as(usize, 2), history.requests.items.len);
    for (history.requests.items) |request| {
        try testing.expectEqual(@as(i64, 97), request.options.rollout_id.?.integer);
        try testing.expectEqual(@as(?f64, 0.5), request.options.temperature);
    }
}

test "BestOfN null results preserve NaN negative infinity and empty attempt semantics" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var trace: Trace = .{ .arena = arena };
    const ctx: Context = .{ .arena = arena, .lm = script.lm(), .trace = &trace };
    var best = BestOfN(TestProgram).init(.{}, .{
        .n = 2,
        .reward_fn = TestProgram.reward,
        .threshold = .{ .float = std.math.inf(f64) },
    });
    try testing.expect((try best.forward(ctx, .{ .scores = &.{ std.math.nan(f64), std.math.nan(f64) } })) == null);
    try testing.expect((try best.forward(ctx, .{ .scores = &.{ -std.math.inf(f64), -std.math.inf(f64) } })) == null);
    // Even an immediate -inf threshold break does not select a prediction.
    best.threshold = .{ .float = -std.math.inf(f64) };
    try testing.expect((try best.forward(ctx, .{ .scores = &.{-std.math.inf(f64)} })) == null);
    try testing.expectEqual(@as(usize, 0), trace.steps.items.len);
    best.n = 0;
    var invalid = ctx;
    invalid.lm_options.rollout_id = .{ .string = "bad" };
    try testing.expect((try best.forward(invalid, .{ .scores = &.{} })) == null);
    best.n = -2;
    try testing.expect((try best.forward(invalid, .{ .scores = &.{} })) == null);
    best.n = 1;
    try testing.expectError(error.UnsupportedRolloutId, best.forward(invalid, .{ .scores = &.{1.0} }));
    invalid.lm_options.rollout_id = .null;
    try testing.expectError(error.UnsupportedRolloutId, best.forward(invalid, .{ .scores = &.{1.0} }));
    try testing.expectEqual(@as(i128, 2), best.fail_count);
}

test "BestOfN program and reward failure budgets persist and check the attempt index before decrement" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .arena = arena, .lm = script.lm() };
    for ([_]bool{ false, true }) |reward_fails| {
        var best = BestOfN(TestProgram).init(.{}, .{
            .n = 1,
            .reward_fn = TestProgram.reward,
            .threshold = .{ .float = 1.0 },
            .fail_count = 0,
        });
        const input: TestProgram.Input = .{
            .scores = &.{0.0},
            .program_failures = if (reward_fails) &.{} else &.{0},
            .reward_failures = if (reward_fails) &.{0} else &.{},
        };
        try testing.expect((try best.forward(ctx, input)) == null);
        try testing.expectEqual(@as(i128, 0), best.fail_count);
        try testing.expect((try best.forward(ctx, input)) == null);
        try testing.expectEqual(@as(i128, -1), best.fail_count);
        try testing.expectError(if (reward_fails) error.RewardFailure else error.ProgramFailure, best.forward(ctx, input));
        try testing.expectEqual(@as(i128, -1), best.fail_count);

        best = BestOfN(TestProgram).init(.{}, .{
            .n = 3,
            .reward_fn = TestProgram.reward,
            .threshold = .{ .float = 1.0 },
        });
        const all_fail: TestProgram.Input = .{
            .scores = &.{ 0.0, 0.0, 0.0 },
            .program_failures = if (reward_fails) &.{} else &.{ 0, 1, 2 },
            .reward_failures = if (reward_fails) &.{ 0, 1, 2 } else &.{},
        };
        try testing.expectError(if (reward_fails) error.RewardFailure else error.ProgramFailure, best.forward(ctx, all_fail));
        try testing.expectEqual(@as(i128, 1), best.fail_count);
    }
}

test "BestOfN a final failed program or reward retains the earlier best and its trace" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    for ([_]bool{ false, true }) |reward_fails| {
        var trace: Trace = .{ .arena = arena };
        var best = BestOfN(TestProgram).init(.{}, .{
            .n = 3,
            .reward_fn = TestProgram.reward,
            .threshold = .{ .float = std.math.inf(f64) },
        });
        const result = (try best.forward(.{ .arena = arena, .lm = script.lm(), .trace = &trace }, .{
            .scores = &.{ 0.6, 0.2, 0.0 },
            .program_failures = if (reward_fails) &.{} else &.{2},
            .reward_failures = if (reward_fails) &.{2} else &.{},
        })).?;
        try testing.expectEqual(@as(usize, 0), result.attempt);
        try testing.expectEqual(@as(i128, 2), best.fail_count);
        try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
        try testing.expectEqualStrings("attempt-0", trace.steps.items[0].predictor.instructions);
    }
}

test "BestOfN threshold errors spend the budget without discarding the best result" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var trace: Trace = .{ .arena = arena };
    const ctx: Context = .{ .arena = arena, .lm = script.lm(), .trace = &trace };
    var best = BestOfN(TestProgram).init(.{}, .{
        .n = 2,
        .reward_fn = TestProgram.reward,
        .threshold = .null,
    });
    const input: TestProgram.Input = .{ .scores = &.{ 1.0, 2.0 } };
    const result = (try best.forward(ctx, input)).?;
    try testing.expectEqual(@as(usize, 1), result.attempt);
    try testing.expectEqual(@as(i128, 0), best.fail_count);
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    try testing.expectEqual(@as(i64, 1), trace.steps.items[0].outputs.get("answer").?.integer);

    try testing.expectError(error.InvalidNumber, best.forward(ctx, input));
    try testing.expectEqual(@as(i128, -1), best.fail_count);
    // The failed invocation doesn't publish its partial best trace.
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
}

test "BestOfN rollout arithmetic retains floats bools and exact unbounded integers" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    try testing.expectEqual(@as(f64, 2.5), (try rolloutAt(arena, .{ .float = 1.5 }, 1)).float);
    try testing.expectEqual(@as(i64, 2), (try rolloutAt(arena, .{ .bool = true }, 1)).integer);
    try testing.expectEqualStrings("9223372036854775808", (try rolloutAt(arena, .{ .integer = std.math.maxInt(i64) }, 1)).number_string);
    const cases = [_]struct { start: []const u8, index: usize, expected: []const u8 }{
        .{ .start = "170141183460469231731687303715884105727", .index = 1, .expected = "170141183460469231731687303715884105728" },
        .{ .start = "999999999999999999999999999999999999999999", .index = 1, .expected = "1000000000000000000000000000000000000000000" },
        .{ .start = "-1000000000000000000000000000000000000000000", .index = 1, .expected = "-999999999999999999999999999999999999999999" },
    };
    for (cases) |case| {
        const result = try rolloutAt(arena, .{ .number_string = case.start }, case.index);
        try testing.expectEqualStrings(case.expected, result.number_string);
    }
}

test "BestOfN LM setup errors do not spend the budget or get hidden by an empty range" {
    const QA = predict.Predict(TestSignature);
    const Pair = struct {
        pub const Input = QA.Input;
        pub const Output = QA.Output;
        first: QA = .{},
        second: QA = .{},
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) predict.Error!Output {
            _ = try self.first.forward(ctx, input);
            return self.second.forward(ctx, input);
        }

        fn reward(_: Context, _: *const Input, _: *const Output) anyerror!f64 {
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const ctx: Context = .{ .arena = memory.allocator() };
    var model: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var best = BestOfN(Pair).init(.{}, .{ .n = 1, .reward_fn = Pair.reward, .threshold = .{ .integer = 1 } });
    try testing.expectError(error.NoLM, best.forward(ctx, .{ .question = "q" }));
    for ([_]isize{ 0, -2 }) |n| {
        best.n = n;
        best.module.first.state.lm = null;
        best.module.second.state.lm = null;
        try testing.expectError(error.NoLM, best.forward(ctx, .{ .question = "q" }));
        best.module.first.state.lm = model.lm();
        try testing.expectError(error.MultipleLMs, best.forward(ctx, .{ .question = "q" }));
        best.module.second.state.lm = model.lm();
        try testing.expectEqual(null, try best.forward(ctx, .{ .question = "q" }));
    }
    try testing.expectEqual(@as(i128, 1), best.fail_count);
    try testing.expectEqual(@as(usize, 0), model.index);
}

test "BestOfN uses boxed numeric rollout defaults and stops at the threshold" {
    const QA = predict.Predict(TestSignature);
    const Id = @import("enum.zig").IntEnum("Id", &.{
        .{ .name = "seven", .value = .{ .integer = 7 } },
    });
    const Reward = struct {
        fn score(_: Context, _: *const QA.Input, output: *const QA.Output) anyerror!f64 {
            return @floatFromInt(output.answer);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const cases = [_]struct { start: Value, ids: [2]Value }{
        .{ .start = try Id.member("seven"), .ids = .{ .{ .integer = 7 }, .{ .integer = 8 } } },
        .{ .start = try @import("python_value.zig").float(arena, 7.5), .ids = .{ .{ .float = 7.5 }, .{ .float = 8.5 } } },
    };
    for (cases) |case| {
        var history: lm_mod.History = .{ .arena = arena };
        var model: lm_mod.ScriptedLM = .{
            .responses = &.{ "<answer>7</answer>", "<answer>8</answer>" },
            .history = &history,
        };
        var best = BestOfN(QA).init(.{}, .{ .n = 3, .reward_fn = Reward.score, .threshold = .{ .integer = 8 } });
        const result = try best.forward(.{
            .arena = arena,
            .lm = model.lm(),
            .lm_options = .{ .rollout_id = case.start },
        }, .{ .question = "q" });
        try testing.expectEqual(@as(usize, 8), result.?.answer);
        try testing.expectEqual(@as(usize, 2), model.index);
        try testing.expectEqual(@as(i128, 3), best.fail_count);
        try testing.expectEqual(@as(usize, 2), history.requests.items.len);
        for (history.requests.items, case.ids) |request, expected| {
            const actual = request.options.rollout_id.?;
            try testing.expectEqual(std.meta.activeTag(expected), std.meta.activeTag(actual));
            try testing.expect(try value.pyEqual(expected, actual));
            try testing.expectEqual(@as(?f64, 1), request.options.temperature);
        }
    }
}

test "BestOfN interruption aborts without spending budget or publishing the earlier winner" {
    const QA = predict.Predict(TestSignature);
    inline for (.{ error.Canceled, error.StreamStopped }) |interruption| {
        const Reward = struct {
            fn score(_: Context, _: *const QA.Input, output: *const QA.Output) anyerror!f64 {
                if (output.answer == 8) return interruption;
                return @floatFromInt(output.answer);
            }
        };
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var lm: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>7</answer>", "<answer>8</answer>", "<answer>9</answer>" } };
        var trace: Trace = .{ .arena = arena };
        var best = BestOfN(QA).init(.{}, .{ .n = 3, .threshold = .{ .integer = 9 }, .reward_fn = Reward.score });
        try testing.expectError(interruption, best.forward(
            .{ .arena = arena, .lm = lm.lm(), .trace = &trace },
            .{ .question = "q" },
        ));
        try testing.expectEqual(@as(usize, 2), lm.index);
        try testing.expectEqual(@as(i128, 3), best.fail_count);
        try testing.expectEqual(@as(usize, 0), trace.steps.items.len);
    }
}
