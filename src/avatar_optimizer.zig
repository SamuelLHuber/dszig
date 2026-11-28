//! DSPy 3.5 `dspy.teleprompt.avatar_optimizer.AvatarOptimizer` for native
//! `Avatar` programs: score the trainset, show a comparator the positive and
//! negative runs, and turn its feedback into a new actor instruction.
//!
//! Source's constructor cannot run in DSPy 3.5 (it calls the removed
//! `dspy.TypedPredictor`); the behavior here follows its unchanged methods,
//! with `Predict` as the comparator.
//!
//! Intentional differences:
//! - Source samples with the process-global `random`; the caller passes the
//!   generator, so runs are reproducible.
//! - Examples run serially in trainset order, which is the order Source
//!   collects its thread results in.
//! - Source scores any failed example or metric as 0. `OutOfMemory` is
//!   returned instead; every other failure scores 0 as in Source.
//! - `optimize_for` is an enum. Source accepts any string and never applies
//!   an instruction for one that is neither "max" nor "min": `.other`.
//! - An empty trainset is `error.AvatarEmptyTrainset`, not ZeroDivisionError.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const avatar = @import("avatar.zig");
const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyrandom = @import("pyrandom.zig");
const runtime_value = @import("runtime_value.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");

const ActionOutput = avatar.ActionOutput;
const Context = predict.Context;
const Example = optimize.Example;
const Metric = optimize.Metric;

pub const default_max_examples = 10;

/// `EvalResult`. `actions` is null when the example or its metric failed.
pub const EvalResult = struct {
    example: runtime_value.ObjectMap,
    score: f64,
    actions: ?[]const ActionOutput = null,

    pub const model_name = "EvalResult";
    pub const model_module = "dspy.teleprompt.avatar_optimizer";
};

pub const Comparator = signature.Signature(struct {
    pub const instructions =
        "After executing the given actions on user inputs using the given instruction, some inputs have yielded good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation metrics:\n\n" ++
        "Task:\n" ++
        "(1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.\n" ++
        "(2) Then, review the computational logic for any inconsistencies in the previous actions.\n" ++
        "(3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs.";
    pub const Input = struct {
        instruction: []const u8,
        actions: []const []const u8,
        pos_input_with_metrics: []const EvalResult,
        neg_input_with_metrics: []const EvalResult,
    };
    pub const Output = struct { feedback: []const u8 };
    pub const descriptions = .{
        .instruction = "Instruction for the actor to execute the task",
        .actions = "Actions actor can take to complete the task",
        .pos_input_with_metrics = "Positive inputs along with their score on a evaluation metric and actions taken",
        .neg_input_with_metrics = "Negative inputs along with their score on a evaluation metric and actions taken",
        .feedback = "Feedback for the actor to improve the performance of negative inputs",
    };
});

pub const FeedbackBasedInstruction = signature.Signature(struct {
    pub const instructions =
        "There is a task that needs to be completed for which one can use multiple tools to achieve the desired outcome. A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive inputs, and the inputs that did not do well are negative inputs.\n\n" ++
        "You received feedback on how they can better use the tools to improve your performance on the negative inputs. You have been provided with the previous instruction, that they followed to use tools to complete the task, and the feedback on your performance.\n\n" ++
        "Your task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve their performance on the task.\n\n" ++
        "Make sure that the new instruction talks about how to use the tools effectively and should be no more than 3 paragraphs long. The previous instruction contains general guidelines that you must retain in the new instruction.";
    pub const Input = struct { previous_instruction: []const u8, feedback: []const u8 };
    pub const Output = struct { new_instruction: []const u8 };
    pub const descriptions = .{
        .previous_instruction = "Previous instruction for the actor to execute the task",
        .feedback = "Feedback for the actor to improve the performance of negative inputs",
        .new_instruction = "New instruction for the actor to execute the task",
    };
});

pub const AvatarOptimizer = struct {
    metric: Metric,
    max_iters: i64 = 10,
    lower_bound: f64 = 0,
    upper_bound: f64 = 1,
    /// Null and zero both mean `default_max_examples`, as in Source.
    max_positive_inputs: ?usize = null,
    max_negative_inputs: ?usize = null,
    optimize_for: OptimizeFor = .max,

    pub const OptimizeFor = enum { max, min, other };

    pub const Error = error{
        AvatarEmptyTrainset,
        /// No score reached `upper_bound`: lower it or add training data.
        AvatarNoPositiveExamples,
        /// No score fell to `lower_bound`: raise it or add training data.
        AvatarNoNegativeExamples,
    };

    pub const Iteration = struct {
        /// Raw mean of the example scores, not a percentage.
        score: f64,
        /// Counts before sampling.
        positives: usize,
        negatives: usize,
        new_instruction: []const u8,
        applied: bool,
    };

    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            iterations: []const Iteration,
        };
    }

    /// `compile(student, trainset=...)`. The student is copied, never changed.
    /// Applied instructions and the report live in `out`, which must outlive
    /// the returned program; each iteration's scratch comes from `gpa`.
    pub fn compile(
        self: AvatarOptimizer,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        rng: *pyrandom.Random,
        student: P,
        trainset: []const Example,
    ) !Result(P) {
        comptime assert(@hasDecl(P, "is_dszig_avatar"));
        if (self.max_iters <= 0) return .{ .program = student, .iterations = &.{} };
        // Training copies must be callable. Failed actors require an explicit reset.
        if (student.evolved()) return error.AvatarActorEvolved;
        var best = student;
        if (trainset.len == 0) return error.AvatarEmptyTrainset;

        const iterations_max: usize = std.math.cast(usize, self.max_iters) orelse std.math.maxInt(usize);
        var iterations: std.ArrayList(Iteration) = .empty;
        var best_score: f64 = if (self.optimize_for == .max) -999 else 999;
        const comparator: predict.Predict(Comparator) = .{};
        const feedback_instruction: predict.Predict(FeedbackBasedInstruction) = .{};

        for (0..iterations_max) |_| {
            var scratch: std.heap.ArenaAllocator = .init(gpa);
            defer scratch.deinit();
            const arena = scratch.allocator();
            var c = ctx;
            c.arena = arena;

            var positives: std.ArrayList(EvalResult) = .empty;
            var negatives: std.ArrayList(EvalResult) = .empty;
            var total: f64 = 0;
            for (trainset) |*example| {
                const result = try self.processExample(P, c, &best, example);
                total += result.score;
                if (result.score >= self.upper_bound) {
                    try positives.append(arena, result);
                } else if (result.score <= self.lower_bound) {
                    try negatives.append(arena, result);
                }
            }
            const score = total / @as(f64, @floatFromInt(trainset.len));
            if (positives.items.len == 0) return error.AvatarNoPositiveExamples;
            if (negatives.items.len == 0) return error.AvatarNoNegativeExamples;

            // Positives draw from the generator before negatives.
            const shown_positives = try sample(arena, rng, positives.items, self.max_positive_inputs);
            const shown_negatives = try sample(arena, rng, negatives.items, self.max_negative_inputs);

            const instruction = best.actor.state.instructions;
            const tool_strings = try arena.alloc([]const u8, best.tools.len + 1);
            for (best.tools, tool_strings[0..best.tools.len]) |tool, *text| text.* = try tool.str(arena);
            tool_strings[best.tools.len] = try avatar.finish_tool.str(arena);

            const compared = try value.toRecord(arena, Comparator.Input, .{
                .instruction = instruction,
                .actions = tool_strings,
                .pos_input_with_metrics = shown_positives,
                .neg_input_with_metrics = shown_negatives,
            });
            const feedback = try value.fromRecord(arena, Comparator.Output, &(try comparator.state.call(c, &compared)));
            const requested = try value.toRecord(arena, FeedbackBasedInstruction.Input, .{
                .previous_instruction = instruction,
                .feedback = feedback.feedback,
            });
            const proposed = try value.fromRecord(arena, FeedbackBasedInstruction.Output, &(try feedback_instruction.state.call(c, &requested)));

            const applied = switch (self.optimize_for) {
                .max => best_score < score,
                .min => best_score > score,
                .other => false,
            };
            const new_instruction = try out.dupe(u8, proposed.new_instruction);
            if (applied) {
                try best.setInstructions(out, new_instruction);
                best_score = score;
            }
            try iterations.append(out, .{
                .score = score,
                .positives = positives.items.len,
                .negatives = negatives.items.len,
                .new_instruction = new_instruction,
                .applied = applied,
            });
        }
        return .{ .program = best, .iterations = iterations.items };
    }

    /// `process_example(actor, example, return_outputs=True)`.
    fn processExample(self: AvatarOptimizer, comptime P: type, ctx: Context, program: *const P, example: *const Example) Allocator.Error!EvalResult {
        const inputs = try example.inputs(ctx.arena);
        const failed: EvalResult = .{ .example = inputs, .score = 0 };
        var usage: ?@import("prediction.zig").Graph.Value = null;
        const output = optimize.forwardProgram(P, ctx, program, example, &usage) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return failed,
        };
        // The typed output stays for `actions`; the metric gets the same
        // call's Prediction, and only score and actions are retained.
        const prediction = try optimize.RecordPrediction.init(ctx.arena, try value.toRecord(ctx.arena, P.Output, output));
        prediction.setLMUsage(usage);
        const score = self.metric(ctx.arena, example, &prediction, null) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return failed,
        };
        return .{ .example = inputs, .score = score, .actions = output.actions };
    }

    /// `sample(results, limit)` only when there are more than `limit`.
    fn sample(arena: Allocator, rng: *pyrandom.Random, results: []const EvalResult, limit_opt: ?usize) Allocator.Error![]const EvalResult {
        const configured = limit_opt orelse default_max_examples;
        const limit = if (configured == 0) default_max_examples else configured;
        if (results.len <= limit) return results;
        const indices = try arena.alloc(usize, limit);
        const scratch = try arena.alloc(usize, results.len);
        rng.sampleIndices(results.len, indices, scratch);
        const chosen = try arena.alloc(EvalResult, limit);
        for (chosen, indices) |*item, index| item.* = results[index];
        return chosen;
    }
};

const TestQA = signature.Signature(struct {
    pub const instructions = "Answer the question with the tools.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

fn testMetric(_: Allocator, example: *const Example, prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    const got = prediction.?.get("answer") orelse return error.ParseValue;
    const want = example.record.get("answer") orelse return error.ParseValue;
    if (got != .string or want != .string) return error.ParseValue;
    return if (std.mem.eql(u8, got.string, want.string)) 1 else 0;
}

const test_finish = "<action_1>{\"tool_name\": \"Finish\", \"tool_input_query\": \"\"}</action_1>";

fn testTrainset(arena: Allocator) ![2]Example {
    return .{
        try Example.fromJson(arena, "{\"question\": \"q1\", \"answer\": \"a\"}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\": \"q2\", \"answer\": \"b\"}", &.{"question"}),
    };
}

test "AvatarOptimizer applies a new instruction only on a strictly better score" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const P = avatar.Avatar(TestQA);
    const student = try P.init(arena, &.{});
    const trainset = try testTrainset(arena);
    // Each iteration: q1 right, q2 wrong, comparator, instruction.
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        test_finish,                                "<answer>a</answer>",
        test_finish,                                "<answer>wrong</answer>",
        "<feedback>use the tools</feedback>",       "<new_instruction>instruction-1</new_instruction>",
        test_finish,                                "<answer>a</answer>",
        test_finish,                                "<answer>wrong</answer>",
        "<feedback>still use the tools</feedback>", "<new_instruction>instruction-2</new_instruction>",
    } };
    var recording: @import("lm.zig").RecordingLM = .{ .inner = script.lm(), .arena = arena };
    var rng: pyrandom.Random = .init(0);
    const optimizer: AvatarOptimizer = .{ .metric = testMetric, .max_iters = 2 };
    const result = try optimizer.compile(P, arena, testing.allocator, .{ .arena = arena, .lm = recording.lm() }, &rng, student, &trainset);

    try testing.expectEqual(@as(usize, 12), script.index);
    try testing.expectEqual(@as(usize, 2), result.iterations.len);
    try testing.expectEqual(@as(f64, 0.5), result.iterations[0].score);
    try testing.expect(result.iterations[0].applied);
    // The second score ties the first, so its instruction is discarded.
    try testing.expect(!result.iterations[1].applied);
    try testing.expectEqualStrings("instruction-2", result.iterations[1].new_instruction);
    try testing.expectEqualStrings("instruction-1", result.program.actor.state.instructions);
    try testing.expectEqualStrings("instruction-1", result.program.actor_clone.state.instructions);
    try testing.expect(result.program.actor.state.signature_class != student.actor.state.signature_class);
    try testing.expectEqualStrings(avatar.actor_instructions, student.actor.state.instructions);
}

test "AvatarOptimizer scores failures zero and needs both example classes" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const P = avatar.Avatar(TestQA);
    const student = try P.init(arena, &.{});
    const trainset = try testTrainset(arena);
    var rng: pyrandom.Random = .init(0);
    const optimizer: AvatarOptimizer = .{ .metric = testMetric, .max_iters = 1 };

    // No iterations: a copy, no LM call, even without a trainset.
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm() };
    var idle = optimizer;
    idle.max_iters = 0;
    const copy = try idle.compile(P, arena, testing.allocator, ctx, &rng, student, &.{});
    try testing.expectEqual(student.actor.state.signature_class, copy.program.actor.state.signature_class);
    try testing.expectError(error.AvatarEmptyTrainset, optimizer.compile(P, arena, testing.allocator, ctx, &rng, student, &.{}));

    // q2 names a tool that does not exist: the failure scores 0 and is the
    // negative example, so the comparator still runs.
    script = .{ .responses = &.{
        test_finish,                                                                      "<answer>a</answer>",
        "<action_1>{\"tool_name\": \"missing\", \"tool_input_query\": \"x\"}</action_1>", "<feedback>f</feedback>",
        "<new_instruction>n</new_instruction>",
    } };
    var low = optimizer;
    low.optimize_for = .min;
    const minimized = try low.compile(P, arena, testing.allocator, ctx, &rng, student, &trainset);
    try testing.expectEqual(@as(usize, 5), script.index);
    try testing.expect(minimized.iterations[0].applied);
    try testing.expectEqualStrings("n", minimized.program.actor.state.instructions);

    script.index = 0;
    var other = optimizer;
    other.optimize_for = .other;
    const unchanged = try other.compile(P, arena, testing.allocator, ctx, &rng, student, &trainset);
    try testing.expectEqual(@as(usize, 5), script.index);
    try testing.expectEqualStrings(avatar.actor_instructions, unchanged.program.actor.state.instructions);

    // Both right: no negatives, and the comparator is never asked.
    script = .{ .responses = &.{ test_finish, "<answer>a</answer>", test_finish, "<answer>b</answer>" } };
    try testing.expectError(error.AvatarNoNegativeExamples, optimizer.compile(P, arena, testing.allocator, ctx, &rng, student, &trainset));
    try testing.expectEqual(@as(usize, 4), script.index);
    script = .{ .responses = &.{ test_finish, "<answer>x</answer>", test_finish, "<answer>y</answer>" } };
    try testing.expectError(error.AvatarNoPositiveExamples, optimizer.compile(P, arena, testing.allocator, ctx, &rng, student, &trainset));
}

test "AvatarOptimizer preserves a failed actor for idle compile and requires reset for training" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const P = avatar.Avatar(TestQA);
    var student = try P.init(arena, &.{});
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{test_finish} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm() };
    try testing.expectError(error.ScriptExhausted, student.call(ctx, .{ .question = "q1" }, .{}));

    var rng: pyrandom.Random = .init(0);
    const idle: AvatarOptimizer = .{ .metric = testMetric, .max_iters = 0 };
    const active: AvatarOptimizer = .{ .metric = testMetric, .max_iters = 1 };
    const trainset = try testTrainset(arena);
    var copy = (try idle.compile(P, arena, testing.allocator, ctx, &rng, student, &.{})).program;
    try testing.expectError(error.AvatarActorEvolved, copy.call(ctx, .{ .question = "q1" }, .{}));
    try testing.expectError(
        error.AvatarActorEvolved,
        active.compile(P, arena, testing.allocator, ctx, &rng, student, &trainset),
    );

    copy.reset();
    script = .{ .responses = &.{ test_finish, "<answer>a</answer>" } };
    const prediction = try copy.call(ctx, .{ .question = "q1" }, .{});
    try testing.expectEqualStrings("a", prediction.answer);
    try testing.expect(student.evolved());
}

test "AvatarOptimizer samples positives before negatives from one generator" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var results: [5]EvalResult = undefined;
    for (&results, 0..) |*result, index| result.* = .{ .example = .empty, .score = @floatFromInt(index) };
    var rng: pyrandom.Random = .init(0);
    // CPython: random.seed(0); random.sample(range(5), 3) == [3, 4, 0].
    const chosen = try AvatarOptimizer.sample(arena, &rng, &results, 3);
    try testing.expectEqual(@as(usize, 3), chosen.len);
    try testing.expectEqual(@as(f64, 3), chosen[0].score);
    try testing.expectEqual(@as(f64, 4), chosen[1].score);
    try testing.expectEqual(@as(f64, 0), chosen[2].score);
    // Zero and null both mean ten, and a short list is never sampled.
    try testing.expectEqual(@as(usize, 5), (try AvatarOptimizer.sample(arena, &rng, &results, 0)).len);
    try testing.expectEqual(@as(usize, 5), (try AvatarOptimizer.sample(arena, &rng, &results, null)).len);
}
