//! Behavioural properties: optimizer convergence over generations, trace and
//! compile idempotency, leak freedom, and bounded fixed-buffer operation.

const std = @import("std");
const dszig = @import("dszig");
const rag = @import("rag_program");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Mock LM that answers a question correctly only when the prompt carries at
/// least `difficulty` demos: few-shot demos genuinely improve it.
const LearningLM = struct {
    const difficulty = [_]usize{ 0, 0, 1, 2, 3 };

    fn lm(self: *LearningLM) dszig.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(_: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const u8 {
        const messages = request.messages;
        var demos: usize = 0;
        for (messages) |m| if (m.role == .assistant) {
            demos += 1;
        };
        const last = messages[messages.len - 1].content;
        for (rag.train_rows, difficulty) |row, need| {
            if (std.mem.indexOf(u8, last, row[0]) == null) continue;
            const answer = if (demos >= need) row[1] else "unknown";
            return std.fmt.allocPrint(arena, "[[ ## reasoning ## ]]\n{d} demos.\n\n[[ ## answer ## ]]\n{s}\n\n[[ ## completed ## ]]", .{ demos, answer });
        }
        return error.LMFailure;
    }
};

test "BootstrapFewShot scores never drop across three generations" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 4, .max_labeled_demos = 0 };

    var teacher: rag.RAG = .{};
    var scores: [4]f64 = undefined;
    scores[0] = (try dszig.optimize.evaluate(rag.RAG, testing.allocator, ctx, &teacher, trainset, rag.keywordMetric, 10)).score;
    for (scores[1..]) |*score| {
        const result = try optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, teacher, trainset);
        teacher = result.program;
        score.* = (try dszig.optimize.evaluate(rag.RAG, testing.allocator, ctx, &teacher, trainset, rag.keywordMetric, 10)).score;
    }
    try testing.expectEqualSlices(f64, &.{ 40.0, 80.0, 100.0, 100.0 }, &scores);
    for (scores[1..], scores[0 .. scores.len - 1]) |now, before| try testing.expect(now >= before);
}

test "Evaluate matches DSPy scores for the RAG fixture" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const eval_json = @embedFile("fixtures/bootstrap_rag_eval.json");
    const expected = try std.json.parseFromSliceLeaky(std.json.Value, out.allocator(), eval_json, .{});
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const baseline = try dszig.optimize.evaluate(rag.RAG, testing.allocator, ctx, &rag.RAG{}, trainset, rag.keywordMetric, 10);
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset);
    const optimized = try dszig.optimize.evaluate(rag.RAG, testing.allocator, ctx, &result.program, trainset, rag.keywordMetric, 10);
    try testing.expectEqual(expected.object.get("baseline").?.float, baseline.score);
    try testing.expectEqual(expected.object.get("optimized").?.float, optimized.score);
}

test "the same forward pass yields identical traces" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var keyword = rag.keywordLM();
    const program: rag.RAG = .{};
    var traces: [2]dszig.Trace = .{ .{ .arena = arena }, .{ .arena = arena } };
    for (&traces) |*trace| {
        const ctx: dszig.Context = .{ .arena = arena, .lm = keyword.lm(), .retriever = rag.corpus.retriever(), .trace = trace };
        const output = try program.forward(ctx, .{ .question = "What is the capital of France?" });
        try testing.expectEqualStrings("Paris", output.answer);
    }
    try testing.expectEqual(@as(usize, 1), traces[0].steps.items.len);
    try testing.expect(traces[0].eql(&traces[1]));
    try testing.expectEqualStrings("What is the capital of France?", traces[0].steps.items[0].inputs.get("question").?.string);
}

test "compile is deterministic and does not mutate its inputs" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const student: rag.RAG = .{};
    const a = try optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, student, null, trainset);
    const b = try optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, student, null, trainset);
    try testing.expect(!student.compiled);
    try testing.expectEqual(@as(usize, 0), student.generate_answer.predict.state.demos.len);
    try testing.expectEqualStrings(try dszig.state.saveAlloc(out.allocator(), &a.program), try dszig.state.saveAlloc(out.allocator(), &b.program));
    try testing.expectError(error.StudentCompiled, optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, a.program, null, trainset));
}

test "compiled sub-modules are frozen out of the parameter walk" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const Outer = struct {
        inner: rag.RAG = .{},
        compiled: bool = false,
        pub const Input = rag.RAG.Input;
        pub const Output = rag.RAG.Output;
        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) dszig.predict.Error!Output {
            return self.inner.forward(ctx, input);
        }
    };
    var program: Outer = .{};
    try testing.expectEqual(@as(usize, 2), (try dszig.predict.namedParameters(out.allocator(), &program)).len);
    try testing.expectEqualStrings("inner.generate_answer.predict", (try dszig.predict.namedPredictors(out.allocator(), &program))[0].name);
    program.inner.compiled = true;
    try testing.expectEqual(@as(usize, 0), (try dszig.predict.namedParameters(out.allocator(), &program)).len);
}

test "optimize, evaluate and save run in fixed buffers" {
    var out_buf: [256 * 1024]u8 = undefined;
    var scratch_buf: [128 * 1024]u8 align(16) = undefined;
    var out: std.heap.FixedBufferAllocator = .init(&out_buf);
    var scratch: std.heap.FixedBufferAllocator = .init(&scratch_buf);
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), scratch.allocator(), ctx, .{}, null, trainset);
    // Per-attempt scratch is released after every example.
    try testing.expectEqual(@as(usize, 0), scratch.end_index);
    const eval = try dszig.optimize.evaluate(rag.RAG, scratch.allocator(), ctx, &result.program, trainset, rag.keywordMetric, 10);
    try testing.expectEqual(@as(f64, 80.0), eval.score);
    try testing.expectEqual(@as(usize, 0), scratch.end_index);
    _ = try dszig.state.saveAlloc(out.allocator(), &result.program);
}

test "every allocation failure point is handled without leaks" {
    try testing.checkAllAllocationFailures(testing.allocator, compileAndSave, .{});
}

fn compileAndSave(gpa: Allocator) !void {
    var out: std.heap.ArenaAllocator = .init(gpa);
    defer out.deinit();
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), gpa, ctx, .{}, null, trainset);
    _ = try dszig.state.saveAlloc(out.allocator(), &result.program);
}

test "random search scores every seed, ranks them and keeps the best" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = rag.keywordMetric,
        .max_bootstrapped_demos = 3,
        .max_labeled_demos = 2,
        .num_candidate_programs = 3,
    };
    const result = try search.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset, null);
    try testing.expectEqual(@as(usize, 6), result.candidates.len);
    var seen: [6]bool = @splat(false);
    for (result.candidates, 0..) |c, i| {
        seen[@intCast(c.seed + 3)] = true;
        if (i > 0) try testing.expect(result.candidates[i - 1].score >= c.score);
    }
    for (seen) |s| try testing.expect(s);
    const best = try dszig.optimize.evaluate(rag.RAG, testing.allocator, ctx, &result.program, trainset, rag.keywordMetric, 10);
    try testing.expectEqual(result.candidates[0].score, best.score);
}

test "random search stops at stop_at_score" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = rag.keywordMetric,
        .num_candidate_programs = 3,
        .stop_at_score = 0.0,
    };
    const result = try search.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset, null);
    try testing.expectEqual(@as(usize, 1), result.candidates.len);
    try testing.expectEqual(@as(i64, -3), result.candidates[0].seed);
}
