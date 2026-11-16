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

    fn complete(_: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
        const out = try arena.alloc(dszig.lm.Completion, 1);
        out[0] = .{ .text = try reply(arena, request) };
        return out;
    }

    fn reply(arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const u8 {
        const messages = request.messages;
        var demos: usize = 0;
        for (messages) |m| if (m.role == .assistant) {
            demos += 1;
        };
        const last = messages[messages.len - 1].content.text;
        for (rag.train_rows, difficulty) |row, need| {
            if (std.mem.indexOf(u8, last, row[0]) == null) continue;
            const answer = if (demos >= need) row[1] else "unknown";
            return std.fmt.allocPrint(arena, "<plan>\n{d} demos.\n</plan>\n\n<answer>\n{s}\n</answer>", .{ demos, answer });
        }
        return error.LMFailure;
    }
};

test "Evaluate uses Python decimal ties-to-even without intermediate rounding or overflow" {
    const P = struct {
        pub const Input = struct { score: f64 };
        pub const Output = struct {};
        pub fn forward(_: *const @This(), _: dszig.Context, _: Input) !Output {
            return .{};
        }
        fn metric(_: Allocator, example: *const dszig.Example, _: ?*const dszig.value.Record, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            return example.record.get("score").?.float;
        }
    };
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .arena = arena, .lm = unused.lm() };
    // Exact result encodings from actual DSPy 3.5 Evaluate with one example.
    const cases = [_]struct { metric: f64, bits: u64 }{
        .{ .metric = 0.02675, .bits = 0x4005_5c28_f5c2_8f5c },
        .{ .metric = 0.00125, .bits = 0x3fbe_b851_eb85_1eb8 },
        .{ .metric = 0.00375, .bits = 0x3fd8_51eb_851e_b852 },
        .{ .metric = -0.00125, .bits = 0xbfbe_b851_eb85_1eb8 },
        .{ .metric = -0.00375, .bits = 0xbfd8_51eb_851e_b852 },
        .{ .metric = -1e-20, .bits = 0x8000_0000_0000_0000 },
        .{ .metric = 1e300, .bits = 0x7ea2_aa4f_4a40_5be2 },
        .{ .metric = 1e308, .bits = 0x7ff0_0000_0000_0000 },
        .{ .metric = -std.math.inf(f64), .bits = 0xfff0_0000_0000_0000 },
    };
    var example: dszig.Example = .{ .record = .empty, .input_keys = &.{"score"} };
    for (cases) |case| {
        try example.record.put(arena, "score", .{ .float = case.metric });
        const result = try dszig.optimize.evaluate(P, testing.allocator, ctx, &.{}, &.{example}, P.metric, 1);
        try testing.expectEqual(case.bits, @as(u64, @bitCast(result.score)));
    }
    try example.record.put(arena, "score", .{ .float = std.math.nan(f64) });
    const result = try dszig.optimize.evaluate(P, testing.allocator, ctx, &.{}, &.{example}, P.metric, 1);
    try testing.expect(std.math.isNan(result.score));
}

test "native input decoding needs no projection allocation and never admits labels" {
    const P = struct {
        product: *i64,
        pub const Input = struct { number: i64, text: []const u8 };
        pub const Output = struct {};

        pub fn forward(self: *const @This(), _: dszig.Context, input: Input) !Output {
            self.product.* = input.number * @as(i64, @intCast(input.text.len));
            return .{};
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var example = try dszig.Example.fromJson(
        arena,
        "{\"number\":7,\"text\":\"abc\",\"answer\":\"private\"}",
        &.{ "text", "unknown", "number", "text" },
    );
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .arena = testing.failing_allocator, .lm = unused.lm() };
    var product: i64 = -1;
    const program: P = .{ .product = &product };
    _ = try dszig.optimize.runProgram(P, ctx, &program, &example);
    try testing.expectEqual(@as(i64, 21), product);

    // An existing training label is not an input just because the program asks for it.
    example.input_keys = &.{"number"};
    product = -1;
    try testing.expectError(
        error.MissingField,
        dszig.optimize.runProgram(P, ctx, &program, &example),
    );
    try testing.expectEqual(@as(i64, -1), product);

    // Decode in declaration order: an earlier invalid input precedes a later missing role.
    try example.record.put(arena, "number", .{ .bool = true });
    try testing.expectError(
        error.TypeMismatch,
        dszig.optimize.runProgram(P, ctx, &program, &example),
    );
    try testing.expectEqual(@as(i64, -1), product);
}

test "Evaluate counts program and metric failures together and keeps allocation failures fatal" {
    const P = struct {
        calls: *usize,
        pub const Input = struct { question: []const u8 };
        pub const Output = struct {};

        pub fn forward(self: *const @This(), _: dszig.Context, input: Input) !Output {
            self.calls.* += 1;
            if (std.mem.eql(u8, input.question, "program_fail")) return error.LMFailure;
            if (std.mem.eql(u8, input.question, "program_oom")) return error.OutOfMemory;
            return .{};
        }

        fn metric(_: Allocator, example: *const dszig.Example, _: ?*const dszig.value.Record, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const question = example.record.get("question").?.string;
            if (std.mem.eql(u8, question, "metric_fail")) return error.ParseValue;
            if (std.mem.eql(u8, question, "metric_oom")) return error.OutOfMemory;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .arena = arena, .lm = unused.lm() };
    const devset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"metric_fail\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good1\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good2\"}", &.{"question"}),
    };
    var calls: usize = 0;
    const program: P = .{ .calls = &calls };
    const result = try dszig.optimize.evaluate(P, testing.allocator, ctx, &program, &devset, P.metric, 2);
    try testing.expectEqual(@as(f64, 66.67), result.score);
    try testing.expectEqual(@as(f64, 2), result.sum);
    try testing.expectEqual(@as(usize, 3), result.total);
    try testing.expectEqual(@as(usize, 1), result.errors);
    try testing.expectEqual(@as(usize, 3), calls);

    for ([_]usize{ 0, 1 }) |max_errors| {
        calls = 0;
        try testing.expectError(error.ParseValue, dszig.optimize.evaluate(P, testing.allocator, ctx, &program, &devset, P.metric, max_errors));
        try testing.expectEqual(@as(usize, 1), calls);
    }

    const program_failure = try dszig.Example.fromJson(arena, "{\"question\":\"program_fail\"}", &.{"question"});
    calls = 0;
    try testing.expectError(error.ParseValue, dszig.optimize.evaluate(P, testing.allocator, ctx, &program, &.{ program_failure, devset[0], devset[1] }, P.metric, 2));
    try testing.expectEqual(@as(usize, 2), calls);

    for ([_][]const u8{ "metric_oom", "program_oom" }) |question| {
        var failure = try dszig.Example.fromJson(arena, "{\"question\":\"unused\"}", &.{"question"});
        try failure.record.put(arena, "question", .{ .string = question });
        calls = 0;
        try testing.expectError(error.OutOfMemory, dszig.optimize.evaluate(P, testing.allocator, ctx, &program, &.{ failure, devset[1] }, P.metric, 100));
        try testing.expectEqual(@as(usize, 1), calls);
    }
}

test "BootstrapFewShot counts metric failures with program failures after restoring teacher demos" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = dszig.Predict(Sig);
    const Model = struct {
        calls: usize = 0,
        fail_first: bool = false,

        fn complete(ptr: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            if (self.fail_first and self.calls == 1) return error.LMFailure;
            const out = try arena.alloc(dszig.lm.Completion, request.options.n);
            for (out) |*completion| completion.* = .{ .text = "<answer>ok</answer>" };
            return out;
        }
    };
    const Metric = struct {
        fn run(_: Allocator, example: *const dszig.Example, _: ?*const dszig.value.Record, trace: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const steps = trace.?.steps.items;
            // A successful metric must see the teacher's full, restored demos,
            // not the per-example filtered slice used during prediction.
            if (steps.len != 1 or steps[0].predictor.demos.len != 3) return error.ParseValue;
            const question = example.record.get("question").?.string;
            if (std.mem.eql(u8, question, "metric_fail")) return error.ParseValue;
            if (std.mem.eql(u8, question, "metric_oom")) return error.OutOfMemory;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const trainset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"metric_fail\",\"answer\":\"ok\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good1\",\"answer\":\"ok\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good2\",\"answer\":\"ok\"}", &.{"question"}),
    };
    var teacher: P = .{};
    teacher.state.demos = &.{ trainset[0].record, trainset[1].record, trainset[2].record };
    var model: Model = .{};
    const ctx: dszig.Context = .{
        .arena = arena,
        .lm = .{ .ptr = &model, .vtable = &.{ .complete = Model.complete } },
        .adapter = .xml,
    };
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = Metric.run,
        .max_bootstrapped_demos = 3,
        .max_labeled_demos = 0,
        .max_errors = 2,
    };
    const result = try optimizer.compile(P, arena, testing.allocator, ctx, .{}, teacher, &trainset);
    try testing.expectEqual(@as(usize, 3), result.report.attempts);
    try testing.expectEqual(@as(usize, 1), result.report.errors);
    try testing.expectEqual(@as(usize, 2), result.report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), result.report.validation);
    try testing.expectEqual(@as(usize, 3), model.calls);
    try testing.expectEqual(@as(usize, 3), teacher.state.demos.len);
    try testing.expectEqual(@as(usize, 2), result.program.state.demos.len);

    for ([_]usize{ 0, 1 }) |max_errors| {
        model.calls = 0;
        optimizer.max_errors = max_errors;
        try testing.expectError(error.ParseValue, optimizer.compile(P, arena, testing.allocator, ctx, .{}, teacher, &trainset));
        try testing.expectEqual(@as(usize, 1), model.calls);
    }

    model.calls = 0;
    model.fail_first = true;
    optimizer.max_errors = 2;
    try testing.expectError(error.ParseValue, optimizer.compile(P, arena, testing.allocator, ctx, .{}, teacher, &.{ trainset[1], trainset[0], trainset[2] }));
    try testing.expectEqual(@as(usize, 2), model.calls);

    const allocation_failure = try dszig.Example.fromJson(arena, "{\"question\":\"metric_oom\",\"answer\":\"ok\"}", &.{"question"});
    model.calls = 0;
    model.fail_first = false;
    optimizer.max_errors = 100;
    try testing.expectError(error.OutOfMemory, optimizer.compile(P, arena, testing.allocator, ctx, .{}, teacher, &.{ allocation_failure, trainset[1] }));
    try testing.expectEqual(@as(usize, 1), model.calls);
}

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

test "teacher_settings runs bootstrapping under the teacher's LM" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = silent.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{
        .metric = rag.keywordMetric,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 0,
        .teacher_settings = .{ .lm = keyword.lm() },
    };
    const result = try optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset);
    try testing.expectEqual(@as(usize, 2), result.report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), silent.index);
    try testing.expectEqual(@as(usize, 2), result.program.generate_answer.predict.state.demos.len);
}

test "BootstrapFewShot zero rounds retains labeled demos and zero errors propagates first failure" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = dszig.Predict(Sig);
    const FailingLM = struct {
        calls: usize = 0,
        fn complete(ptr: *anyopaque, _: Allocator, _: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            return error.LMFailure;
        }
    };
    var failing: FailingLM = .{};
    const ctx: dszig.Context = .{ .arena = arena, .lm = .{ .ptr = &failing, .vtable = &.{ .complete = FailingLM.complete } } };
    const trainset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"q0\",\"answer\":\"a0\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"q1\",\"answer\":\"a1\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"q2\",\"answer\":\"a2\"}", &.{"question"}),
    };
    var optimizer: dszig.BootstrapFewShot = .{ .max_rounds = 0, .max_errors = 0, .max_labeled_demos = 2 };
    const result = try optimizer.compile(P, arena, testing.allocator, ctx, .{}, null, &trainset);
    try testing.expect(result.program.compiled);
    try testing.expectEqual(@as(usize, 0), result.report.attempts);
    try testing.expectEqual(@as(usize, 0), failing.calls);
    const demos = result.program.state.demos;
    try testing.expectEqual(@as(usize, 2), demos.len);
    // Actual pinned DSPy selects these labels after its shuffle and sample.
    try testing.expectEqualStrings("q2", demos[0].get("question").?.string);
    try testing.expectEqualStrings("a2", demos[0].get("answer").?.string);
    try testing.expectEqualStrings("q1", demos[1].get("question").?.string);
    try testing.expectEqualStrings("a1", demos[1].get("answer").?.string);
    optimizer.max_rounds = 1;
    optimizer.max_labeled_demos = 0;
    try testing.expectError(error.LMFailure, optimizer.compile(P, arena, testing.allocator, ctx, .{}, null, &trainset));
    try testing.expectEqual(@as(usize, 1), failing.calls);
}

test "BootstrapFewShot zero threshold accepts a zero score while null rejects it" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = arena, .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    // The real shared mock answers Germany with Munich, not the gold Berlin.
    const trainset = (try rag.trainset(arena))[3..4];
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = rag.keywordMetric,
        .metric_threshold = 0.0,
        .max_bootstrapped_demos = 1,
        .max_labeled_demos = 0,
    };
    const accepted = try optimizer.compile(rag.RAG, arena, testing.allocator, ctx, .{}, null, trainset);
    try testing.expect(accepted.program.compiled);
    try testing.expectEqual(@as(usize, 1), accepted.report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), accepted.report.validation);
    const demos = accepted.program.generate_answer.predict.state.demos;
    try testing.expectEqual(@as(usize, 1), demos.len);
    try testing.expectEqualStrings("Munich", demos[0].get("answer").?.string);
    try testing.expectEqual(@as(f64, 0.0), try rag.keywordMetric(arena, &trainset[0], &demos[0], null));
    try testing.expectEqualStrings("Not sure.", (try dszig.Reasoning.fromValue(demos[0].get("reasoning").?)).content);

    optimizer.metric_threshold = null;
    const rejected = try optimizer.compile(rag.RAG, arena, testing.allocator, ctx, .{}, null, trainset);
    try testing.expect(rejected.program.compiled);
    try testing.expectEqual(@as(usize, 0), rejected.report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), rejected.report.validation);
    try testing.expectEqual(@as(usize, 0), rejected.program.generate_answer.predict.state.demos.len);
}

test "Evaluate matches DSPy scores for the RAG fixture" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const eval_json = @embedFile("fixtures/bootstrap_rag_eval.json");
    const expected = try std.json.parseFromSliceLeaky(dszig.Value, out.allocator(), eval_json, .{});
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
        try testing.expectEqualStrings("The context says Paris is the capital of France.", output.reasoning.?.content);
    }
    try testing.expectEqual(@as(usize, 1), traces[0].steps.items.len);
    try testing.expect(try traces[0].eql(&traces[1]));
    try testing.expectEqualStrings("What is the capital of France?", traces[0].steps.items[0].inputs.get("question").?.string);
    try testing.expectEqualStrings("The context says Paris is the capital of France.", (try dszig.Reasoning.fromValue(traces[0].steps.items[0].outputs.get("reasoning").?)).content);
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
        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) anyerror!Output {
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
    // Disable address-dependent backing resize so every injected allocation
    // site is deterministic while the complete optimizer/save path still runs.
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), compileAndSave, .{});
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

test "Chat Completions retain native reasoning without inventing absent summaries" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const json =
        \\{"choices":[
        \\  {"message":{"content":"first","reasoning_content":"Actual summary."}},
        \\  {"message":{"content":"second"}},
        \\  {"message":{"content":"third","reasoning_content":null}},
        \\  {"message":{"content":"fourth","reasoning_content":""}},
        \\  {"message":{"content":"fifth","reasoning_content":42}},
        \\  {"message":{"content":"sixth","reasoning_content":[]}},
        \\  {"message":{"content":"seventh","reasoning_content":["bad"]}}
        \\]}
    ;
    const body = try arena.dupe(u8, json);
    const completions = try dszig.lm.OpenAI.parseCompletions(arena, body);
    // Provider response buffers are released after parsing.
    @memset(body, 'x');
    try testing.expectEqual(@as(usize, 7), completions.len);
    try testing.expectEqualStrings("first", completions[0].text.?);
    try testing.expectEqualStrings("Actual summary.", completions[0].reasoning.?);
    try testing.expectEqualStrings("second", completions[1].text.?);
    try testing.expect(completions[1].reasoning == null);
    try testing.expectEqualStrings("third", completions[2].text.?);
    try testing.expect(completions[2].reasoning == null);
    try testing.expectEqualStrings("fourth", completions[3].text.?);
    try testing.expect(completions[3].reasoning == null);
    try testing.expectEqualStrings("42", completions[4].reasoning.?);
    try testing.expect(completions[5].reasoning == null);
    try testing.expectEqualStrings("['bad']", completions[6].reasoning.?);

    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseCompletions(arena, "{\"choices\":[]}"));
}

test "Responses parse assistant output after reasoning and preserve real summary parts" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const json =
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[
        \\    {"type":"summary_text","text":"Locate the capital."},
        \\    {"type":"summary_text","text":"Use the supplied fact."}
        \\  ]},
        \\  {"type":"web_search_call","status":"completed"},
        \\  {"type":"message","role":"assistant","status":"completed","content":[
        \\    {"type":"output_text","text":"<answer>\n"},
        \\    {"type":"output_text","text":"Paris\n</answer>"}
        \\  ]}
        \\]}
    ;
    const body = try arena.dupe(u8, json);
    const completion = try dszig.lm.OpenAI.parseResponsesResponse(arena, body);
    @memset(body, 'x');
    try testing.expectEqualStrings("<answer>\nParis\n</answer>", completion.text.?);
    try testing.expectEqualStrings("Locate the capital.\nUse the supplied fact.", completion.reasoning.?);

    const no_summary = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[]},
        \\  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
        \\]}
    );
    try testing.expectEqualStrings("answer", no_summary.text.?);
    try testing.expect(no_summary.reasoning == null);

    const empty_summary = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[{"type":"summary_text","text":""}]},
        \\  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
        \\]}
    );
    try testing.expect(empty_summary.reasoning == null);
}

test "Responses citation spans count Python characters and affect canonical call identifiers" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const completion = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[
        \\ {"type":"message","content":[{"type":"output_text","text":"A😀B","annotations":[
        \\   {"start_index":"١","end_index":2.9,"url":"https://example.org","title":"Unicode"},
        \\   {"quote":"fact","url":false,"uri":false,"title":"","file_id":0},
        \\   {"start_index":true,"end_index":2}
        \\ ]}]},
        \\ {"type":"function_call","name":"probe","arguments":"{}"}
        \\]}
    );
    try testing.expectEqual(@as(usize, 2), completion.citations.len);
    try testing.expectEqualStrings("😀", completion.citations[0].text.?);
    try testing.expectEqualStrings("https://example.org", completion.citations[0].url.?);
    try testing.expectEqualStrings("Unicode", completion.citations[0].title.?);
    try testing.expectEqualStrings("fact", completion.citations[1].text.?);
    try testing.expectEqualStrings("False", completion.citations[1].url.?);
    try testing.expectEqualStrings("0", completion.citations[1].title.?);
    try testing.expectEqualStrings("call_3", completion.tool_calls[0].id);
    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"text","text":"x","annotations":[{"start_index":Infinity}]}]}]}
    ));
    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"text","text":"x","annotations":[{"quote":"quoted","start_index":Infinity}]}]}]}
    ));
    const quoted = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"text","text":"x","annotations":[{"quote":"\ud800"}]}]}]}
    );
    try testing.expectEqualStrings("\xed\xa0\x80", quoted.citations[0].text.?);
}

test "Chat text blocks preserve canonical indices and reasoning-only responses have no text" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const blocks = try dszig.lm.OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":[{"type":"text","text":""},{"type":"text","text":"b"}],
        \\ "tool_calls":[{"type":"function","function":{"name":"probe","arguments":"{}"}}]}}]}
    );
    try testing.expectEqualStrings("b", blocks[0].text.?);
    try testing.expectEqualStrings("call_2", blocks[0].tool_calls[0].id);
    const reasoning = try dszig.lm.OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":null,"reasoning_content":"","reasoning":"fallback reasoning",
        \\ "tool_calls":[{"type":"function","function":{"name":"probe","arguments":"{}"}}]}}]}
    );
    try testing.expect(reasoning[0].text == null);
    try testing.expectEqualStrings("fallback reasoning", reasoning[0].reasoning.?);
    try testing.expectEqualStrings("call_1", reasoning[0].tool_calls[0].id);
}

test "citation completion clones unwind every failed allocation" {
    const Clone = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const completion = try dszig.lm.cloneCompletion(allocator, .{
                .text = "answer",
                .citations = &.{
                    .{
                        .text = "quoted",
                        .title = "source",
                        .url = "https://example.org",
                        .continuation = &.{.{ .provider = "fixture", .kind = "citation", .data = .{ .string = "state" } }},
                    },
                    .{ .url = "https://example.net" },
                },
            });
            defer dszig.lm.freeCompletion(allocator, completion);
            try testing.expectEqualStrings("quoted", completion.citations[0].text.?);
            try testing.expectEqualStrings("state", completion.citations[0].continuation[0].data.string);
            try testing.expectEqualStrings("https://example.net", completion.citations[1].url.?);
        }
    };
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), Clone.run, .{});
}

test "Responses distinguish absent text from empty output and retain fallback call identifiers" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    // Actual DSPy 3.5 LM HTTP observations, not provider status assumptions.
    const cases = [_]struct { body: []const u8, text: ?[]const u8 }{
        .{ .body = "{\"output\":null}", .text = "" },
        .{ .body = "{\"output\":[],\"output_text\":\"fallback\"}", .text = "fallback" },
        .{ .body = "{\"status\":\"incomplete\",\"output\":[{\"type\":\"message\",\"role\":\"user\",\"status\":\"incomplete\",\"content\":[{\"type\":\"output_text\",\"text\":\"partial\"}]}]}", .text = "partial" },
        .{ .body = "{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"No.\"}]}]}", .text = null },
        .{ .body = "{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"first\"},{\"type\":\"text\",\"text\":\"second\"}]}]}", .text = "firstsecond" },
        .{ .body = "{\"output\":[{\"type\":\"function_call\",\"name\":\"probe\",\"arguments\":\"{}\"}]}", .text = null },
    };
    for (cases) |case| {
        const completion = try dszig.lm.OpenAI.parseResponsesResponse(arena, case.body);
        if (case.text) |text| try testing.expectEqualStrings(text, completion.text.?) else try testing.expect(completion.text == null);
        const restored = try dszig.lm.completionFromValue(arena, try dszig.lm.completionValue(arena, completion));
        if (case.text) |text| try testing.expectEqualStrings(text, restored.text.?) else try testing.expect(restored.text == null);
    }
    const tool_only = try dszig.lm.OpenAI.parseResponsesResponse(arena, cases[cases.len - 1].body);
    try testing.expectEqualStrings("call_0", tool_only.tool_calls[0].id);
    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseResponsesResponse(arena, "{\"error\":{\"code\":\"fixture_error\",\"message\":\"source fixture error\"},\"output_text\":\"not a successful response\"}"));
}

test "provider request bodies select reasoning JSON and completion-count contracts" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var provider: dszig.lm.OpenAI = .{
        .client = undefined,
        .base_url = "http://unused",
        .model = "reasoning-model",
        .api = .responses,
        .max_tokens = 1024,
        .supports_reasoning = true,
        .supports_response_format = true,
    };
    const messages = [_]dszig.adapter.Message{
        .{ .role = .system, .content = .{ .text = "Use the supplied fact." } },
        .{ .role = .user, .content = .{ .text = "Capital?" } },
    };
    const request: dszig.lm.Request = .{
        .messages = &messages,
        .options = .{ .n = 3, .response_format = .json_object, .reasoning_effort = "low", .rollout_id = .{ .integer = 7 } },
    };
    const responses = (try std.json.parseFromSliceLeaky(dszig.Value, arena, try provider.requestBody(arena, request), .{})).object;
    try testing.expectEqualStrings("reasoning-model", responses.get("model").?.string);
    try testing.expectEqual(@as(i64, 1024), responses.get("max_output_tokens").?.integer);
    try testing.expect(!responses.contains("store"));
    try testing.expect(!responses.contains("messages") and !responses.contains("n"));
    try testing.expect(!responses.contains("rollout_id"));
    try testing.expectEqualStrings(messages[0].content.text, responses.get("instructions").?.string);
    const input = responses.get("input").?.array.items;
    try testing.expectEqual(@as(usize, 1), input.len);
    try testing.expectEqualStrings("user", input[0].object.get("role").?.string);
    const content = input[0].object.get("content").?.array.items;
    try testing.expectEqual(@as(usize, 1), content.len);
    try testing.expectEqualStrings("input_text", content[0].object.get("type").?.string);
    try testing.expectEqualStrings(messages[1].content.text, content[0].object.get("text").?.string);
    const reasoning = responses.get("reasoning").?.object;
    try testing.expectEqualStrings("low", reasoning.get("effort").?.string);
    try testing.expectEqualStrings("auto", reasoning.get("summary").?.string);
    try testing.expectEqualStrings("json_object", responses.get("text").?.object.get("format").?.object.get("type").?.string);
    try testing.expect(provider.lm().supports_reasoning);

    for ([_][]const u8{ "none", "off" }) |effort| {
        var disabled = request;
        disabled.options.reasoning_effort = effort;
        const body = (try std.json.parseFromSliceLeaky(dszig.Value, arena, try provider.requestBody(arena, disabled), .{})).object;
        try testing.expectEqualStrings("none", body.get("reasoning").?.object.get("effort").?.string);
        try testing.expect(!body.get("reasoning").?.object.contains("summary"));
    }

    provider.api = .chat_completions;
    const chat = (try std.json.parseFromSliceLeaky(dszig.Value, arena, try provider.requestBody(arena, request), .{})).object;
    try testing.expectEqual(@as(i64, 3), chat.get("n").?.integer);
    try testing.expectEqual(@as(i64, 1024), chat.get("max_tokens").?.integer);
    try testing.expectEqualStrings("low", chat.get("reasoning_effort").?.string);
    try testing.expectEqualStrings("json_object", chat.get("response_format").?.object.get("type").?.string);
    try testing.expectEqual(messages.len, chat.get("messages").?.array.items.len);
    try testing.expect(!chat.contains("input") and !chat.contains("store") and !chat.contains("rollout_id"));

    const ordinary = (try std.json.parseFromSliceLeaky(dszig.Value, arena, try provider.requestBody(arena, .{ .messages = &messages }), .{})).object;
    try testing.expect(!ordinary.contains("reasoning_effort"));
    try testing.expect(!ordinary.contains("response_format") and !ordinary.contains("n"));
}

test "Chat replays assistant reasoning as text without dropping answers or native calls" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const native_calls = [_]dszig.lm.ToolCall{.{ .id = "call_thought", .name = "lookup", .arguments = "{ \"query\": \"Paris\" }" }};
    const Case = struct {
        reasoning: ?[]const u8,
        content: []const u8,
        expected: []const u8,
        calls: []const dszig.lm.ToolCall = &.{},
    };
    const cases = [_]Case{
        .{ .reasoning = "thought", .content = "answer", .expected = "thought\nanswer" },
        .{ .reasoning = "thought", .content = "", .expected = "thought" },
        .{ .reasoning = "", .content = "answer", .expected = "answer" },
        .{ .reasoning = null, .content = "answer", .expected = "answer" },
        .{ .reasoning = "thought", .content = "answer", .expected = "thought\nanswer", .calls = &native_calls },
    };
    const provider: dszig.lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "m" };
    for (cases) |case| {
        const messages = [_]dszig.adapter.Message{.{
            .role = .assistant,
            .content = .{ .text = case.content },
            .reasoning = case.reasoning,
            .tool_calls = case.calls,
        }};
        const body = try provider.requestBody(arena, .{ .messages = &messages });
        const root = try std.json.parseFromSliceLeaky(dszig.Value, arena, body, .{});
        const projected = root.object.get("messages").?.array.items;
        try testing.expectEqual(@as(usize, 1), projected.len);
        const message = projected[0].object;
        try testing.expectEqualStrings("assistant", message.get("role").?.string);
        try testing.expectEqualStrings(case.expected, message.get("content").?.string);
        try testing.expect(!message.contains("reasoning") and !message.contains("reasoning_content"));
        if (case.calls.len == 0) {
            try testing.expect(!message.contains("tool_calls"));
        } else {
            const calls = message.get("tool_calls").?.array.items;
            try testing.expectEqual(case.calls.len, calls.len);
            for (case.calls, calls) |want, got| {
                try testing.expectEqualStrings(want.id, got.object.get("id").?.string);
                try testing.expectEqualStrings("function", got.object.get("type").?.string);
                const function = got.object.get("function").?.object;
                try testing.expectEqualStrings(want.name, function.get("name").?.string);
                try testing.expectEqualStrings(want.arguments, function.get("arguments").?.string);
            }
        }
    }
}

test "recording metadata roundtrip and replay preserve rich completions and exact requests" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const parameters = try std.json.parseFromSliceLeaky(dszig.Value, arena,
        \\{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
    , .{});
    const output_schema = try std.json.parseFromSliceLeaky(dszig.Value, arena,
        \\{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
    , .{});
    const tool_calls = [_]dszig.lm.ToolCall{.{ .id = "call_lookup", .name = "lookup", .arguments = "{ \"query\": \"Paris\" }" }};
    var scripted: dszig.lm.ScriptedLM = .{
        .responses = &.{ "", "{\"answer\":\"Berlin\"}" },
        .reasonings = &.{ "Actual summary.", null },
        .tool_calls = &.{ &tool_calls, &.{} },
        .supports_reasoning = true,
        .supports_response_format = true,
        .supports_response_schema = true,
        .supports_function_calling = true,
    };
    var recording: dszig.lm.RecordingLM = .{ .inner = scripted.lm(), .arena = arena };
    const request: dszig.lm.Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "question" } }},
        .tools = &.{.{ .name = "lookup", .description = "Look up a fact.", .parameters = parameters, .strict = true }},
        .options = .{
            .n = 2,
            .temperature = 0.7,
            .rollout_id = .{ .integer = 4 },
            .reasoning_effort = "low",
            .response_format = .{ .json_schema = .{ .name = "TaskOutput", .schema = output_schema } },
            .tool_choice = .{ .function = "lookup" },
            .parallel_tool_calls = false,
        },
    };
    try testing.expect(recording.lm().supports_reasoning);
    try testing.expect(recording.lm().supports_response_format);
    try testing.expect(recording.lm().supports_response_schema);
    try testing.expect(recording.lm().supports_function_calling);
    const completed = try recording.lm().completeResponses(arena, request);
    try testing.expectEqual(@as(usize, 2), completed.len);
    try testing.expectEqualStrings("Actual summary.", completed[0].reasoning.?);
    try testing.expect(completed[1].reasoning == null);
    try testing.expectEqual(@as(usize, 1), completed[0].tool_calls.len);
    try testing.expectEqualStrings("call_lookup", completed[0].tool_calls[0].id);
    try testing.expectEqualStrings("lookup", completed[0].tool_calls[0].name);
    try testing.expectEqualStrings(tool_calls[0].arguments, completed[0].tool_calls[0].arguments);
    try testing.expectEqual(@as(usize, 0), completed[1].tool_calls.len);
    try testing.expectEqual(@as(usize, 2), recording.calls.items.len);
    try testing.expectEqual(@as(usize, 1), recording.calls.items[0].tools.len);
    try testing.expectEqualStrings("lookup", recording.calls.items[0].tools[0].name);

    var used = [_]bool{ false, false };
    var replay: dszig.lm.ReplayLM = .{
        .calls = recording.calls.items,
        .used = &used,
        .supports_reasoning = true,
        .supports_response_format = true,
        .supports_response_schema = true,
        .supports_function_calling = true,
    };
    try testing.expect(replay.lm().supports_reasoning);
    try testing.expect(replay.lm().supports_response_format);
    try testing.expect(replay.lm().supports_response_schema);
    try testing.expect(replay.lm().supports_function_calling);
    var changed = request;
    changed.options.reasoning_effort = "high";
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.options.response_format = .text;
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.options.tool_choice = .none;
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.options.parallel_tool_calls = true;
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.tools = &.{};
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    try testing.expect(!used[0]);
    const restored_request = try dszig.lm.requestFromValue(arena, try dszig.lm.requestValue(arena, request));
    const replayed = try replay.lm().completeResponses(arena, restored_request);
    try testing.expectEqual(@as(usize, 2), replayed.len);
    for (completed, replayed) |want, got| {
        const restored = try dszig.lm.completionFromValue(arena, try dszig.lm.completionValue(arena, want));
        if (restored.text) |text| try testing.expectEqualStrings(text, got.text.?) else try testing.expect(got.text == null);
        if (restored.reasoning) |summary| try testing.expectEqualStrings(summary, got.reasoning.?) else try testing.expect(got.reasoning == null);
        try testing.expectEqual(restored.tool_calls.len, got.tool_calls.len);
        for (restored.tool_calls, got.tool_calls) |want_call, got_call| {
            try testing.expectEqualStrings(want_call.id, got_call.id);
            try testing.expectEqualStrings(want_call.name, got_call.name);
            try testing.expectEqualStrings(want_call.arguments, got_call.arguments);
        }
    }
    try testing.expect(used[0] and used[1]);
    try testing.expectEqual(@as(usize, 5), replay.misses);
}

test "native function-only completions preserve identifiers and normalize Responses arguments" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const chat_body = try arena.dupe(u8,
        \\{"choices":[{"message":{"content":null,"reasoning_content":"Look up the fact.","tool_calls":[
        \\  {"id":"call_chat","type":"function","function":{"name":"lookup","arguments":"{ \"query\": \"Paris\", \"limit\": 2 }"}}
        \\]}}]}
    );
    const chat = try dszig.lm.OpenAI.parseCompletions(arena, chat_body);
    @memset(chat_body, 'x');
    try testing.expectEqual(@as(usize, 1), chat.len);
    try testing.expect(chat[0].text == null);
    try testing.expectEqualStrings("Look up the fact.", chat[0].reasoning.?);
    try testing.expectEqual(@as(usize, 1), chat[0].tool_calls.len);
    try testing.expectEqualStrings("call_chat", chat[0].tool_calls[0].id);
    try testing.expectEqualStrings("lookup", chat[0].tool_calls[0].name);

    const responses_body = try arena.dupe(u8,
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[{"type":"summary_text","text":"Look up the fact."}]},
        \\  {"type":"function_call","id":"fc_item","call_id":"call_responses","status":"completed",
        \\   "name":"lookup","arguments":"{ \"query\": \"Paris\", \"limit\": 2 }"}
        \\]}
    );
    const responses = try dszig.lm.OpenAI.parseResponsesResponse(arena, responses_body);
    @memset(responses_body, 'x');
    try testing.expect(responses.text == null);
    try testing.expectEqualStrings("Look up the fact.", responses.reasoning.?);
    try testing.expectEqual(@as(usize, 1), responses.tool_calls.len);
    try testing.expectEqualStrings("call_responses", responses.tool_calls[0].id);
    try testing.expectEqualStrings("lookup", responses.tool_calls[0].name);
    try testing.expectEqualStrings("{\"query\": \"Paris\", \"limit\": 2}", responses.tool_calls[0].arguments);
}

test "native schemas tools choices and tool results use each provider's actual wire format" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const parameters = try std.json.parseFromSliceLeaky(dszig.Value, arena,
        \\{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}
    , .{});
    const output_schema = try std.json.parseFromSliceLeaky(dszig.Value, arena,
        \\{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}
    , .{});
    const arguments = "{ \"query\": \"Paris\" }";
    const calls = [_]dszig.lm.ToolCall{.{ .id = "call_lookup", .name = "lookup", .arguments = arguments }};
    const messages = [_]dszig.adapter.Message{
        .{ .role = .user, .content = .{ .text = "Capital?" } },
        .{ .role = .assistant, .content = .{ .text = "" }, .tool_calls = &calls },
        .{ .role = .tool, .content = .{ .text = "Paris" }, .tool_call_id = "call_lookup" },
    };
    const tools = [_]dszig.lm.FunctionTool{.{
        .name = "lookup",
        .description = "Look up a fact.",
        .parameters = parameters,
        .strict = true,
    }};
    const request: dszig.lm.Request = .{
        .messages = &messages,
        .tools = &tools,
        .options = .{
            .response_format = .{ .json_schema = .{ .name = "TaskOutput", .schema = output_schema, .description = "Answer only.", .strict = true } },
            .tool_choice = .{ .function = "lookup" },
            .parallel_tool_calls = false,
        },
    };
    var provider: dszig.lm.OpenAI = .{
        .client = undefined,
        .base_url = "http://unused",
        .model = "native-model",
        .supports_response_format = true,
        .supports_response_schema = true,
        .supports_function_calling = true,
    };
    try testing.expect(provider.lm().supports_response_format);
    try testing.expect(provider.lm().supports_response_schema);
    try testing.expect(provider.lm().supports_function_calling);

    const chat = (try std.json.parseFromSliceLeaky(dszig.Value, arena, try provider.requestBody(arena, request), .{})).object;
    const chat_tools = chat.get("tools").?.array.items;
    try testing.expectEqual(@as(usize, 1), chat_tools.len);
    try testing.expectEqualStrings("function", chat_tools[0].object.get("type").?.string);
    const chat_function = chat_tools[0].object.get("function").?.object;
    try testing.expectEqualStrings("lookup", chat_function.get("name").?.string);
    try testing.expectEqualStrings("Look up a fact.", chat_function.get("description").?.string);
    try testing.expect(chat_function.get("strict").?.bool);
    try testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, parameters, .{}),
        try std.json.Stringify.valueAlloc(arena, chat_function.get("parameters").?, .{}),
    );
    const chat_format = chat.get("response_format").?.object;
    try testing.expectEqualStrings("json_schema", chat_format.get("type").?.string);
    const chat_schema = chat_format.get("json_schema").?.object;
    try testing.expectEqualStrings("TaskOutput", chat_schema.get("name").?.string);
    try testing.expectEqualStrings("Answer only.", chat_schema.get("description").?.string);
    try testing.expect(chat_schema.get("strict").?.bool);
    try testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, output_schema, .{}),
        try std.json.Stringify.valueAlloc(arena, chat_schema.get("schema").?, .{}),
    );
    try testing.expectEqualStrings("function", chat.get("tool_choice").?.object.get("type").?.string);
    try testing.expectEqualStrings("lookup", chat.get("tool_choice").?.object.get("function").?.object.get("name").?.string);
    try testing.expect(!chat.get("parallel_tool_calls").?.bool);
    const chat_messages = chat.get("messages").?.array.items;
    try testing.expectEqual(messages.len, chat_messages.len);
    const chat_call = chat_messages[1].object.get("tool_calls").?.array.items[0].object;
    try testing.expectEqualStrings("call_lookup", chat_call.get("id").?.string);
    try testing.expectEqualStrings("lookup", chat_call.get("function").?.object.get("name").?.string);
    try testing.expectEqualStrings(arguments, chat_call.get("function").?.object.get("arguments").?.string);
    try testing.expectEqualStrings("tool", chat_messages[2].object.get("role").?.string);
    try testing.expectEqualStrings("call_lookup", chat_messages[2].object.get("tool_call_id").?.string);
    try testing.expectEqualStrings("Paris", chat_messages[2].object.get("content").?.string);

    provider.api = .responses;
    const responses = (try std.json.parseFromSliceLeaky(dszig.Value, arena, try provider.requestBody(arena, request), .{})).object;
    const response_tools = responses.get("tools").?.array.items;
    try testing.expectEqual(@as(usize, 1), response_tools.len);
    try testing.expectEqualStrings("function", response_tools[0].object.get("type").?.string);
    try testing.expectEqualStrings("lookup", response_tools[0].object.get("name").?.string);
    try testing.expectEqualStrings("Look up a fact.", response_tools[0].object.get("description").?.string);
    try testing.expect(response_tools[0].object.get("strict").?.bool);
    try testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, parameters, .{}),
        try std.json.Stringify.valueAlloc(arena, response_tools[0].object.get("parameters").?, .{}),
    );
    const response_format = responses.get("text").?.object.get("format").?.object;
    try testing.expectEqualStrings("json_schema", response_format.get("type").?.string);
    try testing.expectEqualStrings("TaskOutput", response_format.get("name").?.string);
    try testing.expectEqualStrings("Answer only.", response_format.get("description").?.string);
    try testing.expect(response_format.get("strict").?.bool);
    try testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, output_schema, .{}),
        try std.json.Stringify.valueAlloc(arena, response_format.get("schema").?, .{}),
    );
    try testing.expectEqualStrings("function", responses.get("tool_choice").?.object.get("type").?.string);
    try testing.expectEqualStrings("lookup", responses.get("tool_choice").?.object.get("name").?.string);
    try testing.expect(!responses.get("parallel_tool_calls").?.bool);
    var call_count: usize = 0;
    var result_count: usize = 0;
    for (responses.get("input").?.array.items) |item| {
        const kind = item.object.get("type") orelse continue;
        if (std.mem.eql(u8, kind.string, "function_call")) {
            call_count += 1;
            try testing.expectEqualStrings("call_lookup", item.object.get("call_id").?.string);
            try testing.expectEqualStrings("lookup", item.object.get("name").?.string);
            try testing.expectEqualStrings(arguments, item.object.get("arguments").?.string);
        } else if (std.mem.eql(u8, kind.string, "function_call_output")) {
            result_count += 1;
            try testing.expectEqualStrings("call_lookup", item.object.get("call_id").?.string);
            try testing.expectEqualStrings("Paris", item.object.get("output").?.string);
        }
    }
    try testing.expectEqual(@as(usize, 1), call_count);
    try testing.expectEqual(@as(usize, 1), result_count);
}

const SubmitDetails = struct { city: []const u8, zip_code: i64 };

const StructuredSubmit = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct {
        answer: []const u8,
        details: SubmitDetails,
        ids: []const i64,
        note: ?[]const u8,
        label: enum { pos, neg },
        count: u8,
    };
});

test "ReAct typed submit returns every nested list nullable enum and bounded output" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{
        \\<next_thought>
        \\Submit the result.
        \\</next_thought>
        \\<tool_calls>
        \\{"tool_calls":[{"name":"submit","args":{"answer":"Paris","details":{"city":"Paris","zip_code":9},"ids":[1,2],"note":null,"label":"pos","count":255}}]}
        \\</tool_calls>
    } };
    var agent = try dszig.ReAct(StructuredSubmit, &.{}).init(arena);
    agent.max_iters = 1;
    const output = try agent.forward(.{ .arena = arena, .lm = scripted.lm() }, .{ .question = "Return the result." });
    try testing.expectEqualStrings("Paris", output.answer);
    try testing.expectEqualStrings("Paris", output.details.city);
    try testing.expectEqual(@as(i64, 9), output.details.zip_code);
    try testing.expectEqualSlices(i64, &.{ 1, 2 }, output.ids);
    try testing.expect(output.note == null);
    try testing.expect(output.label == .pos);
    try testing.expectEqual(@as(u8, 255), output.count);
    try testing.expectEqual(@as(usize, 1), scripted.index);
}

test "ReAct rejects malformed typed final values before returning any output" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var agent = try dszig.ReAct(StructuredSubmit, &.{}).init(arena);
    agent.max_iters = 0;
    const inputs = try dszig.value.toRecord(arena, StructuredSubmit.Input, .{ .question = "Return the result." });
    for ([_][]const u8{
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":256}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":-1}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":1.5}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":42,\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":7}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":\"not a list\",\"note\":null,\"label\":\"pos\",\"count\":7}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"other\",\"count\":7}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\"}",
    }) |args| {
        const completion = try std.mem.concat(arena, u8, &.{
            "<next_thought>\nSubmit the result.\n</next_thought>\n<tool_calls>\n{\"tool_calls\":[{\"name\":\"submit\",\"args\":",
            args,
            "}]}\n</tool_calls>",
        });
        var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{completion} };
        var history: dszig.History = .{};
        try testing.expectError(error.ReActSubmitFailed, agent.runWithOptions(.{ .arena = arena, .lm = scripted.lm() }, &inputs, .{ .history = &history }));
        try testing.expectEqual(@as(usize, 1), scripted.index);
        try testing.expectEqual(@as(usize, 1), history.messages.len);
        const calls = try dszig.react.ToolCalls.fromValue(arena, history.messages[0].get("tool_calls").?);
        const results = calls.tool_call_results.?.tool_call_results;
        try testing.expectEqual(@as(usize, 1), results.len);
        try testing.expect(results[0].is_error);
        try testing.expectEqualStrings("Execution error in submit: InvalidArguments", results[0].value.string);
        try testing.expect(!history.messages[0].contains("answer"));
    }
}

test "ReAct reuses structured History across calls without replacing prior events" {
    const QA = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var requests: dszig.lm.History = .{ .arena = arena };
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{
        "<next_thought>\nFirst result.\n</next_thought>\n<tool_calls>\n{\"tool_calls\":[{\"id\":\"submit-paris\",\"name\":\"submit\",\"args\":{\"answer\":\"Paris\"}}]}\n</tool_calls>",
        "<next_thought>\nSecond result.\n</next_thought>\n<tool_calls>\n{\"tool_calls\":[{\"id\":\"submit-berlin\",\"name\":\"submit\",\"args\":{\"answer\":\"Berlin\"}}]}\n</tool_calls>",
    }, .history = &requests };
    const ctx: dszig.Context = .{ .arena = arena, .lm = scripted.lm() };
    var agent = try dszig.ReAct(QA, &.{}).init(arena);
    agent.max_iters = 1;
    var history: dszig.History = .{};
    const first_inputs = try dszig.value.toRecord(arena, QA.Input, .{ .question = "First question?" });
    const first = try agent.runWithOptions(ctx, &first_inputs, .{ .history = &history });
    try testing.expectEqualStrings("Paris", first.outputs.get("answer").?.string);
    try testing.expectEqual(dszig.react.TerminationReason.submit, first.termination_reason);
    try testing.expectEqual(@as(usize, 1), history.messages.len);
    const saved_first = try std.json.Stringify.valueAlloc(arena, dszig.Value{ .object = history.messages[0] }, .{});

    const second_inputs = try dszig.value.toRecord(arena, QA.Input, .{ .question = "Second question?" });
    const second = try agent.runWithOptions(ctx, &second_inputs, .{ .history = &history });
    try testing.expectEqualStrings("Berlin", second.outputs.get("answer").?.string);
    try testing.expectEqual(dszig.react.TerminationReason.submit, second.termination_reason);
    try testing.expectEqual(@as(usize, 2), history.messages.len);
    try testing.expectEqual(@as(usize, 1), first.history.messages.len);
    try testing.expectEqualStrings(saved_first, try std.json.Stringify.valueAlloc(arena, dszig.Value{ .object = history.messages[0] }, .{}));
    try testing.expectEqualStrings("First question?", history.messages[0].get("question").?.string);
    try testing.expectEqualStrings("Second question?", history.messages[1].get("question").?.string);
    try testing.expectEqualStrings("Paris", history.messages[0].get("answer").?.string);
    try testing.expectEqualStrings("Berlin", history.messages[1].get("answer").?.string);
    const first_calls = try dszig.react.ToolCalls.fromValue(arena, history.messages[0].get("tool_calls").?);
    const second_calls = try dszig.react.ToolCalls.fromValue(arena, history.messages[1].get("tool_calls").?);
    try testing.expectEqualStrings("submit-paris", first_calls.tool_calls[0].id.?);
    try testing.expectEqualStrings("submit-berlin", second_calls.tool_calls[0].id.?);
    try testing.expectEqual(@as(usize, 2), requests.requests.items.len);
    try testing.expect(requests.requests.items[1].messages.len > requests.requests.items[0].messages.len);
    var saw_first = false;
    var saw_second = false;
    var saw_first_answer = false;
    for (requests.requests.items[1].messages) |message| {
        saw_first = saw_first or message.content.contains("First question?");
        saw_second = saw_second or message.content.contains("Second question?");
        saw_first_answer = saw_first_answer or message.content.contains("Paris");
    }
    try testing.expect(saw_first and saw_second and saw_first_answer);
}

test "LM fans out actual n1 candidate requests and replay preserves strict rollout identity" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const messages = [_]dszig.adapter.Message{.{ .role = .user, .content = .{ .text = "Look up the facts." } }};
    const parameters = try std.json.parseFromSliceLeaky(dszig.Value, arena, "{\"type\":\"object\"}", .{});
    const tools = [_]dszig.lm.FunctionTool{.{ .name = "lookup", .parameters = parameters }};
    const texts = [_][]const u8{ "first", "second", "third" };
    const summaries = [_]?[]const u8{ "First summary.", null, "" };
    const first_calls = [_]dszig.lm.ToolCall{.{ .id = "call_first", .name = "lookup", .arguments = "{ \"query\": \"first\" }" }};
    const third_calls = [_]dszig.lm.ToolCall{.{ .id = "call_third", .name = "lookup", .arguments = "{\"query\":\"third\"}" }};
    const native_calls = [_][]const dszig.lm.ToolCall{ &first_calls, &.{}, &third_calls };
    const array_rollout = try std.json.parseFromSliceLeaky(dszig.Value, arena, "[\"session\",7]", .{});
    const bases = [_]?dszig.Value{ null, .{ .integer = 4 }, .{ .string = "base-run" }, array_rollout };
    for (bases) |base| {
        var scripted: dszig.lm.ScriptedLM = .{
            .responses = &texts,
            .reasonings = &summaries,
            .tool_calls = &native_calls,
            .supports_reasoning = true,
            .supports_function_calling = true,
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = scripted.lm(), .arena = arena };
        const request: dszig.lm.Request = .{
            .messages = &messages,
            .tools = &tools,
            .options = .{ .n = 3, .temperature = 0.7, .reasoning_effort = "low", .rollout_id = base },
        };
        const generated = try recording.lm().completeResponses(arena, request);
        try testing.expectEqual(@as(usize, 3), generated.len);
        try testing.expectEqual(@as(usize, 3), scripted.index);
        try testing.expectEqual(@as(usize, 3), recording.calls.items.len);
        try testing.expectEqual(@as(usize, 3), dszig.lm.ReplayLM.completionCount(recording.calls.items));
        for (recording.calls.items, 0..) |call, index| {
            const options = call.options.?;
            try testing.expectEqual(@as(u32, 1), options.n);
            try testing.expectEqual(@as(f64, 0.7), options.temperature.?);
            try testing.expectEqualStrings("low", options.reasoning_effort.?);
            const rollout = options.rollout_id.?.array.items;
            try testing.expectEqual(@as(usize, 3), rollout.len);
            try testing.expectEqualStrings("dspy.adapter.candidate", rollout[0].string);
            try testing.expectEqualStrings(
                try std.json.Stringify.valueAlloc(arena, base orelse .null, .{}),
                try std.json.Stringify.valueAlloc(arena, rollout[1], .{}),
            );
            try testing.expectEqual(@as(i64, @intCast(index)), rollout[2].integer);
            try testing.expectEqual(@as(usize, 1), call.all().len);
            try testing.expectEqualStrings(texts[index], call.completion.?);
            try testing.expectEqualStrings(texts[index], generated[index].text.?);
        }

        var untagged_used = [_]bool{ false, false, false };
        var untagged: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &untagged_used };
        var wrong_request = request;
        wrong_request.options.n = 1;
        try testing.expectError(error.LMFailure, untagged.lm().completeResponses(arena, wrong_request));
        try testing.expectEqualSlices(bool, &.{ false, false, false }, &untagged_used);
        try testing.expectEqual(@as(usize, 1), untagged.misses);

        var used = [_]bool{ false, false, false };
        var replay: dszig.lm.ReplayLM = .{
            .calls = recording.calls.items,
            .used = &used,
            .supports_reasoning = true,
            .supports_function_calling = true,
        };
        const completions = try replay.lm().completeResponses(arena, request);
        try testing.expectEqual(@as(usize, 3), completions.len);
        for (completions, 0..) |got, index| {
            try testing.expectEqualStrings(texts[index], got.text.?);
            if (summaries[index]) |summary| try testing.expectEqualStrings(summary, got.reasoning.?) else try testing.expect(got.reasoning == null);
            try testing.expectEqual(native_calls[index].len, got.tool_calls.len);
            for (native_calls[index], got.tool_calls) |want_call, got_call| {
                try testing.expectEqualStrings(want_call.id, got_call.id);
                try testing.expectEqualStrings(want_call.name, got_call.name);
                try testing.expectEqualStrings(want_call.arguments, got_call.arguments);
            }
        }
        try testing.expect(replay.exhausted());
        try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, request));
        try testing.expectEqual(@as(usize, 1), replay.misses);
    }
}

test "tool JSON repair requires raw ID provenance and preserves legitimate partial_json" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const completions = try dszig.lm.OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":null,"tool_calls":[
        \\ {"id":"real","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"Paris\""}},
        \\ {"type":"function","function":{"name":"lookup","arguments":"{\"query\":\"Paris\""}},
        \\ {"id":"legitimate","type":"function","function":{"name":"lookup","arguments":"{\"partial_json\":\"{bad\"}"}}
        \\]}}]}
    );
    const calls = completions[0].tool_calls;
    try testing.expectEqualStrings("{\"query\": \"Paris\"}", calls[0].arguments);
    try testing.expectEqualStrings("call_1", calls[1].id);
    try testing.expectEqualStrings("{\"partial_json\": \"{\\\"query\\\":\\\"Paris\\\"\"}", calls[1].arguments);
    try testing.expectEqualStrings("{\"partial_json\": \"{bad\"}", calls[2].arguments);
}

test "native provider calls reject unnamed functions" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    for ([_][]const u8{
        "{\"choices\":[{\"message\":{\"content\":null,\"tool_calls\":[{\"id\":\"call\",\"type\":\"function\",\"function\":{\"arguments\":\"{}\"}}]}}]}",
    }) |invalid| {
        try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseCompletions(arena, invalid));
    }
}
