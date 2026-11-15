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
//! var best = Best.init(.{}, .{ .n = 3, .reward_fn = Rewards.score, .threshold = 1.0 });
//! const result: ?Program.Output = try best.forward(ctx, input);
//! ```
//!
//! Like other native modules, program copies share immutable slices; mutable
//! program state must be stored by value, not behind shared mutable pointers.
//! Attempt programs, outputs, and selected trace predictor pointers live in
//! `ctx.arena`. That arena must outlive consumption of the output and trace.
//! `forward` takes a mutable wrapper because DSPy's failure budget persists
//! between calls. Python's nonnumeric rollout addition errors are represented
//! by `error.UnsupportedRolloutId`, not coerced into numeric IDs.
//!
//! Upstream (DSPy 3.5.0): dspy/predict/best_of_n.py

const std = @import("std");
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
            threshold: f64,
            fail_count: ?i128 = null,
        };

        module: P,
        n: isize,
        reward_fn: Reward,
        threshold: f64,
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

        /// No best prediction is a genuine null, including all NaN/-inf
        /// rewards and zero/negative attempt counts. Reaching a threshold
        /// does not itself guarantee that a best prediction was selected.
        pub fn forward(self: *Self, ctx: Context, input: Input) anyerror!Output {
            if (self.n <= 0) return null;
            const start = ctx.lm_options.rollout_id orelse Value{ .integer = 0 };
            var best: Output = null;
            var best_steps: []const Trace.Step = &.{};
            var best_reward = -std.math.inf(f64);
            for (0..@as(usize, @intCast(self.n))) |idx| {
                var attempt_ctx = ctx;
                // Like LM.copy/deepcopy upstream, setup is outside the
                // caught program/reward region and never consumes budget.
                attempt_ctx.lm_options.rollout_id = try rolloutAt(ctx.arena, start, idx);
                attempt_ctx.lm_options.temperature = 1.0;
                const module = try ctx.arena.create(P);
                module.* = self.module;
                const attempt = self.runAttempt(module, attempt_ctx, ctx, &input) catch |err| {
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
                if (attempt.reward >= self.threshold) break;
            }
            if (ctx.trace) |trace| {
                for (best_steps) |step| {
                    // Trace inputs are already serialized. Calling record
                    // again would format native tool catalogs a second time.
                    try trace.steps.append(trace.arena, .{
                        .predictor = step.predictor,
                        .inputs = try value.cloneRecord(trace.arena, &step.inputs),
                        .outputs = try value.cloneRecord(trace.arena, &step.outputs),
                    });
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

/// Python numeric addition: bool is integral; integer arithmetic remains
/// exact past i64/i128 rather than passing through a lossy float conversion.
const RolloutError = Allocator.Error || error{UnsupportedRolloutId};

pub fn rolloutAt(arena: Allocator, start: Value, index: usize) RolloutError!Value {
    return switch (start) {
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
const lm_mod = @import("lm.zig");
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
        const rollout = ctx.lm_options.rollout_id.?;
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
            .temperature = ctx.lm_options.temperature,
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
        .threshold = 0.8,
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
        .threshold = std.math.inf(f64),
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
        .threshold = 1.0,
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
        .threshold = std.math.inf(f64),
    });
    try testing.expect((try best.forward(ctx, .{ .scores = &.{ std.math.nan(f64), std.math.nan(f64) } })) == null);
    try testing.expect((try best.forward(ctx, .{ .scores = &.{ -std.math.inf(f64), -std.math.inf(f64) } })) == null);
    // Even an immediate -inf threshold break does not select a prediction.
    best.threshold = -std.math.inf(f64);
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
            .threshold = 1.0,
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
            .threshold = 1.0,
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
            .threshold = std.math.inf(f64),
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
