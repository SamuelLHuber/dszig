//! Golden tests against fixtures produced by real DSPy 3.5.0
//! (tools/dspy_reference/gen_fixtures.py). Prompts must match byte for byte,
//! parsed outputs must be equal, and the BootstrapFewShot artifact must be
//! identical to the one DSPy saves for the same program, data and mock LM.

const std = @import("std");
const dszig = @import("dszig");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Value = dszig.Value;
const Record = dszig.value.Record;
const rag = @import("rag_program");

const format_cases = @embedFile("fixtures/format_cases.json");
const bootstrap_rag = @embedFile("fixtures/bootstrap_rag.json");
const bootstrap_multihop = @embedFile("fixtures/bootstrap_multihop.json");
const react_cases = @embedFile("fixtures/react.json");
const ensemble_cases = @embedFile("fixtures/ensemble.json");
const simba_case = @embedFile("fixtures/simba.json");
const gepa_case = @embedFile("fixtures/gepa.json");
const gepa_merge_case = @embedFile("fixtures/gepa_merge.json");
const nested_case = @embedFile("fixtures/nested.json");
const xml_cases = @embedFile("fixtures/xml.json");
const live_suite = @import("live_suite");
const QA = rag.QA;

const Typed = dszig.Signature(struct {
    pub const Input = struct { text: []const u8, tags: []const []const u8 };
    pub const Output = struct {
        flag: bool,
        count: i64,
        ratio: f64,
        label: enum { pos, neg },
        items: []const []const u8,
    };
    pub const descriptions = .{ .tags = "free-form labels", .ratio = "between 0 and 1" };
});

const Indented = dszig.Signature(struct {
    pub const instructions = "Classify the text.\n    Keep nested indentation.\n\nBe brief.";
    pub const Input = struct { text: []const u8 };
    pub const Output = struct { label: enum { pos, neg } };
});

fn findCase(arena: Allocator, name: []const u8) !dszig.ObjectMap {
    const all = try std.json.parseFromSliceLeaky(Value, arena, format_cases, .{});
    for (all.array.items) |c| {
        if (std.mem.eql(u8, c.object.get("name").?.string, name)) return c.object;
    }
    return error.CaseNotFound;
}

fn parseRecords(arena: Allocator, json: []const u8) ![]const Record {
    const v = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
    const out = try arena.alloc(Record, v.array.items.len);
    for (v.array.items, out) |item, *o| o.* = item.object;
    return out;
}

fn compact(arena: Allocator, v: Value) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(arena);
    try dszig.pyjson.writeJson(&aw.writer, v, .compact);
    return aw.written();
}

fn expectMessages(arena: Allocator, expected: []const Value, actual: []const dszig.adapter.Message) !void {
    // DSPy's fixture boundary is prompt_to_openai_messages, not just content.
    // Use the production serializer to compare native calls and tool-result IDs
    // as well, while keeping prompt text and argument strings byte-exact.
    const provider: dszig.lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "golden" };
    const body = try provider.requestBody(arena, .{ .messages = actual });
    const root = try std.json.parseFromSliceLeaky(Value, arena, body, .{});
    const projected = root.object.get("messages").?.array.items;
    try testing.expectEqual(expected.len, projected.len);
    for (expected, projected) |want, got| {
        try testing.expectEqualStrings(want.object.get("role").?.string, got.object.get("role").?.string);
        try testing.expectEqualStrings(want.object.get("content").?.string, got.object.get("content").?.string);
        try testing.expect(try dszig.value.recordEqual(&want.object, &got.object));
    }
}

fn runCase(
    comptime P: type,
    name: []const u8,
    adapter: dszig.Adapter,
    program: *P,
    state: *dszig.predict.PredictState,
    demos_json: []const u8,
    input: P.Input,
) !void {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const case = try findCase(arena, name);
    var history: dszig.lm.History = .{ .arena = arena };
    var script: dszig.lm.ScriptedLM = .{ .responses = &.{case.get("completion").?.string}, .history = &history };
    state.demos = try parseRecords(arena, demos_json);
    const ctx: dszig.Context = .{ .arena = arena, .lm = script.lm(), .adapter = adapter };
    const output = try program.forward(ctx, input);

    const expected = case.get("messages").?.array.items;
    try expectMessages(arena, expected, history.requests.items[0].messages);
    if (comptime @hasField(P.Output, "reasoning")) {
        const expected_reasoning = case.get("outputs").?.object.get("reasoning").?;
        if (expected_reasoning == .null) {
            try testing.expect(output.reasoning == null);
        } else {
            try testing.expectEqualStrings(expected_reasoning.string, output.reasoning.?.content);
        }
    }
    const got_outputs = try dszig.value.toRecord(arena, P.Output, output);
    try testing.expectEqualStrings(
        try compact(arena, case.get("outputs").?),
        try compact(arena, .{ .object = got_outputs }),
    );
}

const qa_demos =
    \\[{"augmented": true, "context": ["a fact"], "question": "q1?", "reasoning": "r1", "answer": "a1"},
    \\ {"question": "q2?", "answer": "a2"}]
;
const qa_input: QA.Input = .{
    .context = &.{ "Paris is the capital of France.", "multi\nline «x»" },
    .question = "What is the capital of France?",
};

test "ChainOfThought ChatAdapter prompt and parse match DSPy" {
    var cot: dszig.ChainOfThought(QA) = .{};
    try runCase(@TypeOf(cot), "cot_qa_chat", .chat, &cot, &cot.predict.state, qa_demos, qa_input);
}

test "ChainOfThought JSONAdapter prompt and parse match DSPy" {
    var cot: dszig.ChainOfThought(QA) = .{};
    try runCase(@TypeOf(cot), "cot_qa_json", .json, &cot, &cot.predict.state, qa_demos, qa_input);
}

test "ChainOfThought default XMLAdapter prompt and Reasoning match DSPy" {
    var cot: dszig.ChainOfThought(QA) = .{};
    try runCase(@TypeOf(cot), "cot_qa_xml", .xml, &cot, &cot.predict.state, qa_demos, qa_input);
}

const typed_demos =
    \\[{"text": "good", "tags": ["x"], "flag": true, "count": 3, "ratio": 0.5, "label": "pos", "items": ["a", "b"]},
    \\ {"text": "bad", "flag": false, "label": "neg"}]
;
const typed_input: Typed.Input = .{ .text = "great movie", .tags = &.{ "review", "film" } };

test "typed fields ChatAdapter match DSPy" {
    var p: dszig.Predict(Typed) = .{};
    try runCase(@TypeOf(p), "typed_chat", .chat, &p, &p.state, typed_demos, typed_input);
}

test "typed fields JSONAdapter match DSPy" {
    var p: dszig.Predict(Typed) = .{};
    try runCase(@TypeOf(p), "typed_json", .json, &p, &p.state, typed_demos, typed_input);
}

test "typed fields XMLAdapter match DSPy" {
    var p: dszig.Predict(Typed) = .{};
    try runCase(@TypeOf(p), "typed_xml", .xml, &p, &p.state, typed_demos, typed_input);
}

test "multi-line instructions match DSPy" {
    var p: dszig.Predict(Indented) = .{};
    try runCase(@TypeOf(p), "indented_chat", .chat, &p, &p.state, "[]", .{ .text = "meh" });
}

test "multi-line XMLAdapter instructions match DSPy" {
    var p: dszig.Predict(Indented) = .{};
    try runCase(@TypeOf(p), "indented_xml", .xml, &p, &p.state, "[]", .{ .text = "meh" });
}

test "BootstrapFewShot RAG artifact is byte-identical to DSPy" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var keyword = rag.keywordLM();
    const retriever = rag.corpus.retriever();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = keyword.lm(), .retriever = retriever };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset);
    try testing.expect(result.program.compiled);
    try testing.expectEqual(@as(usize, 2), result.report.bootstrapped);
    const saved = try dszig.state.saveAlloc(out.allocator(), &result.program);
    try testing.expectEqualStrings(bootstrap_rag, saved);

    var loaded: rag.RAG = .{};
    try dszig.state.load(out.allocator(), &loaded, bootstrap_rag, .{});
    try testing.expectEqualStrings(bootstrap_rag, try dszig.state.saveAlloc(out.allocator(), &loaded));
}

test "BootstrapFewShot multi-hop artifact picks DSPy's same-predictor traces" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var keyword = rag.multihopLM();
    const retriever = rag.corpus.retriever();
    const ctx: dszig.Context = .{ .arena = out.allocator(), .lm = keyword.lm(), .retriever = retriever };
    const trainset = try rag.trainset(out.allocator());
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 4, .max_labeled_demos = 1 };
    const result = try optimizer.compile(rag.MultiHop, out.allocator(), testing.allocator, ctx, .{}, null, trainset);
    try testing.expectEqual(@as(usize, 4), result.report.bootstrapped);
    const saved = try dszig.state.saveAlloc(out.allocator(), &result.program);
    try testing.expectEqualStrings(bootstrap_multihop, saved);
    // DSPy kept the first hop's trace for some examples and the second for others.
    try testing.expect(std.mem.indexOf(u8, saved, "Tokyo Japan capital") != null);
    try testing.expect(std.mem.indexOf(u8, saved, "highest mountain Everest") != null);
}

fn reactSearch(ctx: dszig.Context, _: *const Record) dszig.react.ToolError!Value {
    var list: dszig.Array = .init(ctx.arena);
    try list.append(.{ .string = "Apollo 11 landed on the Moon in 1969." });
    try list.append(.{ .string = "multi\nline «x»" });
    return .{ .array = list };
}

fn reactAdd(_: dszig.Context, args: *const Record) dszig.react.ToolError!Value {
    const a = args.get("a") orelse return error.InvalidArguments;
    const b: i64 = if (args.get("b")) |b| b.integer else 2;
    return .{ .integer = a.integer + b };
}

const search_tool: dszig.Tool = .{
    .name = "search",
    .desc = "Search the corpus.\n    Returns passages.",
    .args = &.{.{ .name = "query", .schema_type = "string" }},
    .func = reactSearch,
};

const add_tool: dszig.Tool = .{
    .name = "add",
    .args = &.{ .{ .name = "a", .schema_type = "integer" }, .{ .name = "b", .schema_type = "integer", .default = "2" } },
    .func = reactAdd,
};

const AgentQA = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

fn checkReact(comptime Agent: type, name: []const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const all = try std.json.parseFromSliceLeaky(Value, arena, react_cases, .{});
    const case = for (all.array.items) |c| {
        if (std.mem.eql(u8, c.object.get("name").?.string, name)) break c.object;
    } else return error.MissingCase;
    const calls = case.get("calls").?.array.items;
    const completions = try arena.alloc([]const u8, calls.len);
    for (calls, completions) |c, *out| out.* = c.object.get("completion").?.string;
    var history: dszig.lm.History = .{ .arena = arena };
    var scripted: dszig.lm.ScriptedLM = .{ .responses = completions, .history = &history };
    const ctx: dszig.Context = .{ .arena = arena, .lm = scripted.lm() };
    var agent = try Agent.init(arena);
    agent.max_iters = @intCast(case.get("max_iters").?.integer);
    var inputs: Record = .empty;
    try inputs.put(arena, "question", .{ .string = "When did Apollo 11 land?" });
    const prediction = try agent.run(ctx, &inputs);
    try testing.expectEqual(calls.len, history.requests.items.len);
    for (calls, history.requests.items) |c, request| {
        const want = c.object.get("messages").?.array.items;
        try expectMessages(arena, want, request.messages);
    }
    try testing.expectEqualStrings(try compact(arena, case.get("history").?), try compact(arena, try prediction.history.toValue(arena)));
    try testing.expectEqualStrings(case.get("termination_reason").?.string, @tagName(prediction.termination_reason));
    try testing.expectEqualStrings(try compact(arena, case.get("outputs").?), try compact(arena, .{ .object = prediction.outputs }));
}

test "ReAct prompts structured history termination and outputs match DSPy" {
    try checkReact(dszig.ReAct(AgentQA, &.{ search_tool, add_tool }), "react_tools");
    try checkReact(dszig.ReAct(AgentQA, &.{search_tool}), "react_max_iters");
    try checkReact(dszig.ReAct(AgentQA, &.{ search_tool, add_tool }), "react_batched");
    try checkReact(dszig.ReAct(AgentQA, &.{search_tool}), "react_empty_tool_calls");
    try checkReact(dszig.ReAct(AgentQA, &.{search_tool}), "react_parse_error");
    try checkReact(dszig.ReAct(AgentQA, &.{search_tool}), "react_unknown_tool");
    try checkReact(dszig.ReAct(AgentQA, &.{search_tool}), "react_zero_iters");
}

test "ReAct records invalid arguments as tool errors before forced typed submit" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{
        "<next_thought>\nx\n</next_thought>\n\n<tool_calls>\n{\"tool_calls\": [{\"name\": \"add\", \"args\": {\"c\": 1}}]}\n</tool_calls>",
        "<next_thought>\ny\n</next_thought>\n\n<tool_calls>\n{\"tool_calls\": [{\"name\": \"add\", \"args\": {\"a\": \"one\"}}]}\n</tool_calls>",
        "<next_thought>\nSubmit the answer.\n</next_thought>\n\n<tool_calls>\n{\"tool_calls\": [{\"name\": \"submit\", \"args\": {\"answer\": \"none\"}}]}\n</tool_calls>",
    } };
    const ctx: dszig.Context = .{ .arena = arena, .lm = scripted.lm() };
    var agent = try dszig.ReAct(AgentQA, &.{add_tool}).init(arena);
    agent.max_iters = 2;
    var inputs: Record = .empty;
    try inputs.put(arena, "question", .{ .string = "?" });
    const prediction = try agent.run(ctx, &inputs);
    try testing.expectEqual(@as(usize, 3), scripted.index);
    try testing.expectEqual(@as(usize, 3), prediction.history.messages.len);
    for (prediction.history.messages[0..2]) |event| {
        const calls = try dszig.react.ToolCalls.fromValue(arena, event.get("tool_calls").?);
        try testing.expectEqual(@as(usize, 1), calls.tool_calls.len);
        try testing.expectEqualStrings("add", calls.tool_calls[0].name);
        const results = calls.tool_call_results.?.tool_call_results;
        try testing.expectEqual(@as(usize, 1), results.len);
        try testing.expectEqualStrings(calls.tool_calls[0].id.?, results[0].call_id.?);
        try testing.expectEqualStrings("add", results[0].name);
        try testing.expect(results[0].is_error);
        try testing.expectEqualStrings("Execution error in add: InvalidArguments", results[0].value.string);
    }
    try testing.expectEqual(dszig.react.TerminationReason.forced_submit, prediction.termination_reason);
    try testing.expectEqualStrings("none", prediction.outputs.get("answer").?.string);
    try testing.expectEqualStrings("none", prediction.history.messages[2].get("answer").?.string);
}

test "normalize_text and dspy.majority match DSPy" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const data = try std.json.parseFromSliceLeaky(Value, arena, ensemble_cases, .{});
    for (data.object.get("normalize").?.array.items) |pair| {
        const got = try dszig.ensemble.normalizeText(arena, pair.array.items[0].string);
        try testing.expectEqualStrings(pair.array.items[1].string, got);
    }
    for (data.object.get("majority").?.array.items) |case| {
        const values = case.object.get("values").?.array.items;
        const records = try arena.alloc(Record, values.len);
        for (values, records) |v, *r| {
            r.* = .empty;
            try r.put(arena, "reasoning", .{ .string = "r" });
            try r.put(arena, "answer", v);
        }
        const winner = try dszig.ensemble.majority(arena, records, "answer");
        try testing.expectEqualStrings(case.object.get("answer").?.string, values[winner].string);
    }
}

const Echo = struct {
    answer: []const u8,
    compiled: bool = false,

    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };

    pub fn forward(self: *const Echo, _: dszig.Context, _: Input) dszig.predict.Error!Output {
        return .{ .answer = self.answer };
    }
};

test "Ensemble reduces with majority and samples like random.sample" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .arena = arena, .lm = scripted.lm() };
    const all: dszig.Ensemble(Echo, 4) = .{ .programs = .{ .{ .answer = "Paris" }, .{ .answer = "Berlin" }, .{ .answer = "berlin!" }, .{ .answer = "Rome" } } };
    try testing.expectEqualStrings("Berlin", (try all.forward(ctx, .{ .question = "?" })).answer);
    // random.seed(0); random.sample(range(4), 2) == [3, 1] in CPython.
    var rng: dszig.pyrandom.Random = .init(0);
    const sampled: dszig.Ensemble(Echo, 4) = .{ .programs = all.programs, .size = 2, .rng = &rng };
    const outputs = try sampled.forwardAll(ctx, .{ .question = "?" });
    try testing.expectEqualStrings("Rome", outputs[0].answer);
    try testing.expectEqualStrings("Berlin", outputs[1].answer);
}

/// tools/dspy_reference/gen_simba.py's MockLM: answers depend on the rollout
/// id, the demos and SIMBA's advice; every request is logged by digest.
const SimbaMock = struct {
    arena: Allocator,
    golds: std.StringHashMapUnmanaged([]const u8) = .empty,
    calls: std.ArrayList(Call) = .empty,

    const advice = "Answer with an ISO date.";
    const feedback = "<discussion>\nThe worse run guessed.\n</discussion>\n\n<module_advice>\n<generate.predict>\n" ++
        advice ++ "\n</generate.predict>\n</module_advice>";
    const Call = struct { rollout: ?Value, temperature: ?f64, digest: [64]u8 };

    fn lm(self: *SimbaMock) dszig.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
        const self: *SimbaMock = @ptrCast(@alignCast(ptr));
        var list: dszig.Array = .init(arena);
        for (request.messages) |m| {
            var o: Record = .empty;
            try o.put(arena, "role", .{ .string = m.role.name() });
            try o.put(arena, "content", m.content.toValue(arena));
            try list.append(.{ .object = o });
        }
        var aw: std.Io.Writer.Allocating = .init(arena);
        dszig.pyjson.writeJson(&aw.writer, .{ .array = list }, .compact) catch return error.OutOfMemory;
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(aw.written(), &hash, .{});
        try self.calls.append(self.arena, .{
            .rollout = if (request.options.rollout_id) |rollout| try dszig.value.cloneValue(self.arena, rollout) else null,
            .temperature = request.options.temperature,
            .digest = std.fmt.bytesToHex(hash, .lower),
        });
        const system = request.messages[0].content.text;
        const user = request.messages[request.messages.len - 1].content.text;
        const out = try arena.alloc(dszig.lm.Completion, 1);
        if (std.mem.indexOf(u8, system, "module_advice") != null) {
            out[0] = .{ .text = feedback };
            return out;
        }
        const marker = "<question>\n";
        const start = (std.mem.indexOf(u8, user, marker) orelse return error.LMFailure) + marker.len;
        const end = std.mem.indexOfPos(u8, user, start, "\n</question>") orelse return error.LMFailure;
        const q = user[start..end];
        var good = std.mem.indexOf(u8, system, advice) != null or (request.messages.len > 2 and q.len % 3 == 0);
        if (request.options.rollout_id) |r| {
            if (r != .integer) return error.LMFailure;
            good = good or q.len % 2 == @as(usize, @intCast(@mod(r.integer, 2)));
        }
        const answer = if (good) self.golds.get(q) orelse return error.LMFailure else "unknown";
        out[0] = .{ .text = try std.fmt.allocPrint(arena, "<plan>\nr\n</plan>\n\n<answer>\n{s}\n</answer>", .{answer}) };
        return out;
    }
};

test "SIMBA program, scores and every request match DSPy" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var mock: SimbaMock = .{ .arena = arena };
    const train = try live_suite.dateExamples(arena, .train);
    const dev = try live_suite.dateExamples(arena, .dev);
    for ([_][]const dszig.Example{ train, dev }) |set| {
        for (set) |ex| try mock.golds.put(arena, ex.record.get("question").?.string, ex.record.get("answer").?.string);
    }
    const ctx: dszig.Context = .{ .arena = arena, .lm = mock.lm(), .lm_options = .{ .temperature = 0.0 } };
    const simba: dszig.SIMBA = .{ .metric = live_suite.exactMetric, .bsize = 4, .num_candidates = 3, .max_steps = 3, .max_demos = 2 };
    const result = try simba.compile(live_suite.DateQA, arena, ctx, .{}, train, .{ .program_code = live_suite.dates_class_source });

    const want = try std.json.parseFromSliceLeaky(Value, arena, simba_case, .{});
    const calls = want.object.get("calls").?.array.items;
    try testing.expectEqual(calls.len, mock.calls.items.len);
    for (calls, mock.calls.items) |w, got| {
        const items = w.array.items;
        try testing.expectEqualStrings(try compact(arena, items[0]), try compact(arena, got.rollout orelse .null));
        try testing.expectEqual(items[1].float, got.temperature.?);
        try testing.expectEqualStrings(items[2].string, &got.digest);
    }
    const scores = want.object.get("scores").?.array.items;
    try testing.expectEqual(scores.len, result.scores.len);
    for (scores, result.scores) |w, got| try testing.expectEqual(w.float, got);
    const train_scores = want.object.get("train_scores").?.array.items;
    for (train_scores, result.train_scores[0..train_scores.len]) |w, got| {
        if (w == .null) try testing.expect(got == null) else try testing.expectEqual(w.float, got.?);
    }
    try testing.expectEqualStrings(want.object.get("artifact").?.string, try dszig.state.saveAlloc(arena, &result.program));
}

test "SIMBA with concurrent rollouts makes DSPy's requests and selects DSPy's program" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    // ArenaAllocator over the thread-safe testing allocator is thread-safe.
    const arena = arena_state.allocator();
    const limits: dszig.parallel.Limits = .{ .workers = 4, .tasks = 16, .attempts = 16, .failures = 1 };
    var mocks: [limits.attempts + 1]SimbaMock = @splat(.{ .arena = arena });
    const train = try live_suite.dateExamples(arena, .train);
    const dev = try live_suite.dateExamples(arena, .dev);
    for (&mocks) |*mock| for ([_][]const dszig.Example{ train, dev }) |set| {
        for (set) |ex| try mock.golds.put(arena, ex.record.get("question").?.string, ex.record.get("answer").?.string);
    };
    var contexts: [limits.attempts]dszig.Context = undefined;
    for (&contexts, mocks[1..]) |*context, *mock| context.* = .{ .arena = arena, .lm = mock.lm() };
    const Pool = dszig.Parallel(dszig.SIMBA.Run, limits);
    const pool = try testing.allocator.create(Pool);
    defer testing.allocator.destroy(pool);
    try pool.init(testing.io, .{});
    defer pool.deinit();
    const ctx: dszig.Context = .{ .arena = arena, .lm = mocks[0].lm(), .lm_options = .{ .temperature = 0.0 } };
    const simba: dszig.SIMBA = .{
        .metric = live_suite.exactMetric,
        .bsize = 4,
        .num_candidates = 3,
        .max_steps = 3,
        .max_demos = 2,
        .rollouts = .{ .runner = pool.runner(), .contexts = &contexts },
    };
    const result = try simba.compile(live_suite.DateQA, arena, ctx, .{}, train, .{ .program_code = live_suite.dates_class_source });

    const want = try std.json.parseFromSliceLeaky(Value, arena, simba_case, .{});
    // Completion order is concurrent; the multiset of requests must match.
    try expectSameCalls(SimbaMock, arena, want.object.get("calls").?.array.items, &mocks);
    const scores = want.object.get("scores").?.array.items;
    try testing.expectEqual(scores.len, result.scores.len);
    for (scores, result.scores) |w, got| try testing.expectEqual(w.float, got);
    const train_scores = want.object.get("train_scores").?.array.items;
    for (train_scores, result.train_scores[0..train_scores.len]) |w, got| {
        if (w == .null) try testing.expect(got == null) else try testing.expectEqual(w.float, got.?);
    }
    try testing.expectEqualStrings(want.object.get("artifact").?.string, try dszig.state.saveAlloc(arena, &result.program));
}

/// Compare DSPy's request log with the union of every mock's log, ignoring
/// the interleaving produced by concurrent completion.
fn expectSameCalls(comptime Mock: type, arena: Allocator, calls: []const Value, mocks: []const Mock) !void {
    var expected: std.ArrayList([]const u8) = .empty;
    for (calls) |w| {
        const items = w.array.items;
        try expected.append(arena, try std.fmt.allocPrint(arena, "{s} {d} {s}", .{ try compact(arena, items[0]), items[1].float, items[2].string }));
    }
    var actual: std.ArrayList([]const u8) = .empty;
    for (mocks) |*mock| for (mock.calls.items) |got| {
        try actual.append(arena, try std.fmt.allocPrint(arena, "{s} {d} {s}", .{ try compact(arena, got.rollout orelse .null), got.temperature.?, &got.digest }));
    };
    const lessThan = struct {
        fn f(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.f;
    std.sort.pdq([]const u8, expected.items, {}, lessThan);
    std.sort.pdq([]const u8, actual.items, {}, lessThan);
    try testing.expectEqual(expected.items.len, actual.items.len);
    for (expected.items, actual.items) |w, got| try testing.expectEqualStrings(w, got);
}

const Address = struct { city: []const u8, zip_code: i64 };

const Person = struct {
    name: []const u8,
    age: i64,
    tags: []const []const u8,
    address: Address,
    role: enum { admin, user },
    score: f64,
    nickname: ?[]const u8 = null,
};

const Extract = dszig.Signature(struct {
    pub const instructions = "Extract people.";
    pub const Input = struct { people: []const Person, note: ?[]const u8 };
    pub const Output = struct {
        person: Person,
        ids: []const i64,
        pairs: []const []const i64,
        maybe: ?[]const u8,
        age: ?i64,
    };
});

test "nested Pydantic-style fields format, parse and validate like DSPy" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const want = try std.json.parseFromSliceLeaky(Value, arena, nested_case, .{});
    const predictor: dszig.Predict(Extract) = .{};
    const view = try predictor.state.view(arena);
    const ada: Person = .{ .name = "Ada", .age = 36, .tags = &.{"math"}, .address = .{ .city = "London", .zip_code = 1 }, .role = .admin, .score = 0.5 };
    const inputs = try dszig.value.toRecord(arena, Extract.Input, .{ .people = &.{ada}, .note = null });
    for (want.object.get("cases").?.array.items) |case| {
        const adapter: dszig.Adapter = if (std.mem.eql(u8, case.object.get("adapter").?.string, "chat")) .chat else .json;
        const messages = try adapter.format(arena, view, &.{}, &inputs);
        const want_messages = case.object.get("messages").?.array.items;
        try expectMessages(arena, want_messages, messages);
    }
    const parsed = try dszig.Adapter.chat.parse(arena, view, want.object.get("completion").?.string);
    try testing.expectEqualStrings(try compact(arena, want.object.get("outputs").?), try compact(arena, .{ .object = parsed }));
    const typed = try dszig.value.fromRecord(arena, Extract.Output, &parsed);
    try testing.expectEqualStrings("Bo", typed.person.name);
    try testing.expectEqual(@as(i64, 9), typed.person.address.zip_code);
    try testing.expect(typed.person.nickname == null and typed.age == null);
    try testing.expectEqualStrings("None", typed.maybe.?);
    try testing.expectEqual(@as(usize, 0), typed.pairs[1].len);
}

const Inventory = struct { items: dszig.Dict(i64), where: dszig.Dict(Address) };

const Tally = dszig.Signature(struct {
    pub const instructions = "Tally things.";
    pub const Input = struct { counts: dszig.Dict(i64), inventory: Inventory };
    pub const Output = struct {
        scores: dszig.Dict(f64),
        groups: dszig.Dict([]const []const u8),
        homes: dszig.Dict(Address),
        deep: dszig.Dict(dszig.Dict(i64)),
    };
});

test "dict[str, T] fields format, parse and validate like DSPy" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const want = try std.json.parseFromSliceLeaky(Value, arena, nested_case, .{});
    const predictor: dszig.Predict(Tally) = .{};
    const view = try predictor.state.view(arena);
    const inputs = try dszig.value.toRecord(arena, Tally.Input, .{
        .counts = .{ .keys = &.{ "b", "a" }, .values = &.{ 2, 1 } },
        .inventory = .{
            .items = .{ .keys = &.{"x"}, .values = &.{1} },
            .where = .{ .keys = &.{"h"}, .values = &.{.{ .city = "Oslo", .zip_code = 5 }} },
        },
    });
    for (want.object.get("dict_cases").?.array.items) |case| {
        const adapter: dszig.Adapter = if (std.mem.eql(u8, case.object.get("adapter").?.string, "chat")) .chat else .json;
        const messages = try adapter.format(arena, view, &.{}, &inputs);
        const want_messages = case.object.get("messages").?.array.items;
        try expectMessages(arena, want_messages, messages);
    }
    const parsed = try dszig.Adapter.chat.parse(arena, view, want.object.get("dict_completion").?.string);
    try testing.expectEqualStrings(try compact(arena, want.object.get("dict_outputs").?), try compact(arena, .{ .object = parsed }));
    const typed = try dszig.value.fromRecord(arena, Tally.Output, &parsed);
    try testing.expectEqual(@as(f64, 2.5), typed.scores.get("a").?);
    try testing.expectEqual(@as(i64, 9), typed.homes.get("bo").?.zip_code);
    try testing.expectEqual(@as(i64, 2), typed.deep.get("k").?.get("z").?);
    try testing.expectEqual(@as(usize, 0), typed.groups.get("y").?.len);
}

const XmlNullable = dszig.Signature(struct {
    pub const instructions = "Return nullable values without conflating null and empty text.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { text: ?[]const u8, number: ?i64, empty: []const u8 };
});

const XmlNestedOptional = dszig.Signature(struct {
    pub const instructions = "Return optional people and dictionary values.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct {
        person: ?Person,
        people: []const ?Person,
        codes: dszig.Dict(?i64),
    };
});

fn xmlNativeRoundTrip(arena: Allocator, name: []const u8, outputs: *const Record) !void {
    inline for (.{
        .{ "primitive", Typed },
        .{ "nullable", XmlNullable },
        .{ "nested_optional", XmlNestedOptional },
        .{ "extract", Extract },
        .{ "tally", Tally },
    }) |schema| {
        if (std.mem.eql(u8, name, schema[0])) {
            const Sig = schema[1];
            const typed = try dszig.value.fromRecord(arena, Sig.Output, outputs);
            const roundtrip = try dszig.value.toRecord(arena, Sig.Output, typed);
            try testing.expectEqualStrings(try compact(arena, .{ .object = outputs.* }), try compact(arena, .{ .object = roundtrip }));
            return;
        }
    }
    if (!std.mem.eql(u8, name, "literal_union")) return error.MissingXmlSchema;
}

fn xmlFixtureView(arena: Allocator, schemas: *const Record, name: []const u8) !dszig.adapter.View {
    inline for (.{
        .{ "primitive", Typed },
        .{ "nullable", XmlNullable },
        .{ "nested_optional", XmlNestedOptional },
        .{ "extract", Extract },
        .{ "tally", Tally },
    }) |schema| {
        if (std.mem.eql(u8, name, schema[0])) {
            const predictor: dszig.Predict(schema[1]) = .{};
            return predictor.state.view(arena);
        }
    }
    if (!std.mem.eql(u8, name, "literal_union")) return error.MissingXmlSchema;

    // Arbitrary Python unions use the existing schema-backed adapter API;
    // no native Zig union signature or compatibility alias is invented.
    const schema = schemas.get(name).?.object;
    const metadata = schema.get("field_metadata").?.object.get("value").?.object;
    const fields = try arena.alloc(dszig.signature.Field, 2);
    fields[0] = Typed.output_fields_list[3];
    fields[1] = .{
        .name = "value",
        .kind = .json,
        .type_name = metadata.get("annotation_name").?.string,
        .desc = metadata.get("description").?.string,
        .prefix = metadata.get("prefix").?.string,
        .schema = try compact(arena, metadata.get("json_schema").?),
    };
    return .{
        .instructions = schema.get("instructions").?.string,
        .inputs = AgentQA.input_fields_list,
        .outputs = fields,
    };
}

fn checkXmlParse(arena: Allocator, view: dszig.adapter.View, case: *const Record) !void {
    const completion = case.get("completion").?.string;
    if (case.get("error")) |expected_error| {
        try testing.expectEqualStrings("AdapterParseError", expected_error.string);
        try testing.expectError(error.AdapterParse, dszig.Adapter.xml.parse(arena, view, completion));
        return;
    }
    const parsed = try dszig.Adapter.xml.parse(arena, view, completion);
    try testing.expectEqualStrings(try compact(arena, case.get("outputs").?), try compact(arena, .{ .object = parsed }));
    try xmlNativeRoundTrip(arena, case.get("schema").?.string, &parsed);
}

test "all independently generated XML formats and parse outcomes match DSPy 3.5" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = (try std.json.parseFromSliceLeaky(Value, arena, xml_cases, .{})).object;
    try testing.expectEqualStrings("3.5.0", fixture.get("dspy").?.string);
    const schemas = fixture.get("schemas").?.object;
    const formats = fixture.get("format_cases").?.array.items;
    try testing.expectEqual(@as(usize, 6), formats.len);
    for (formats) |case| {
        const row = case.object;
        const view = try xmlFixtureView(arena, &schemas, row.get("schema").?.string);
        const inputs = row.get("inputs").?.object;
        const messages = try dszig.Adapter.xml.format(arena, view, &.{}, &inputs);
        const expected = row.get("messages").?.array.items;
        try expectMessages(arena, expected, messages);
        try checkXmlParse(arena, view, &row);
    }
    const parses = fixture.get("parse_cases").?.array.items;
    try testing.expect(parses.len > 0);
    for (parses) |case| {
        const row = case.object;
        const view = try xmlFixtureView(arena, &schemas, row.get("schema").?.string);
        try checkXmlParse(arena, view, &row);
    }
}

/// tools/dspy_reference/gen_gepa.py's MockLM: task answers are right when the
/// instruction carries the advice; reflection proposes a fixed sequence.
const GepaMock = struct {
    arena: Allocator,
    golds: std.StringHashMapUnmanaged([]const u8) = .empty,
    calls: std.ArrayList(SimbaMock.Call) = .empty,
    reflections: usize = 0,

    const advice = "Answer with an ISO date.";
    const keep_advice = "Keep the question.";
    const proposals = [_][]const u8{ "Be brief.", "Think about the calendar. " ++ advice, "Answer politely.", advice ++ " Never guess." };

    fn lm(self: *GepaMock) dszig.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
        const self: *GepaMock = @ptrCast(@alignCast(ptr));
        var list: dszig.Array = .init(arena);
        for (request.messages) |m| {
            var o: Record = .empty;
            try o.put(arena, "role", .{ .string = m.role.name() });
            try o.put(arena, "content", m.content.toValue(arena));
            try list.append(.{ .object = o });
        }
        var aw: std.Io.Writer.Allocating = .init(arena);
        dszig.pyjson.writeJson(&aw.writer, .{ .array = list }, .compact) catch return error.OutOfMemory;
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(aw.written(), &hash, .{});
        try self.calls.append(self.arena, .{
            .rollout = if (request.options.rollout_id) |rollout| try dszig.value.cloneValue(self.arena, rollout) else null,
            .temperature = request.options.temperature,
            .digest = std.fmt.bytesToHex(hash, .lower),
        });
        const system = request.messages[0].content.text;
        const user = request.messages[request.messages.len - 1].content.text;
        const out = try arena.alloc(dszig.lm.Completion, 1);
        const reflection = std.mem.indexOf(u8, system, "examples_with_feedback") != null;
        if (reflection and std.mem.indexOf(u8, user, "Restate the question.") != null) {
            out[0] = .{ .text = "{\"new_instruction\": \"Restate the question. " ++ keep_advice ++ "\"}" };
            return out;
        }
        if (reflection) {
            out[0] = .{ .text = try std.fmt.allocPrint(arena, "{{\"new_instruction\": \"  {s}\\n\"}}", .{proposals[self.reflections % proposals.len]}) };
            self.reflections += 1;
            return out;
        }
        const marker = "<question>\n";
        const start = (std.mem.indexOf(u8, user, marker) orelse return error.LMFailure) + marker.len;
        const end = std.mem.indexOfPos(u8, user, start, "\n</question>") orelse return error.LMFailure;
        const q = user[start..end];
        if (std.mem.indexOf(u8, system, "`restated`") != null) {
            const keep = if (std.mem.indexOf(u8, system, keep_advice) != null) q.len % 5 != 0 else q.len % 2 == 1;
            out[0] = .{ .text = try std.fmt.allocPrint(arena, "<restated>\n{s}{s}\n</restated>", .{ q, if (keep) "" else " Please." }) };
            return out;
        }
        const gold = self.golds.get(q);
        const good = (std.mem.indexOf(u8, system, advice) != null or q.len % 3 == 0) and gold != null;
        const answer = if (good) gold.? else "unknown";
        out[0] = .{ .text = try std.fmt.allocPrint(arena, "<plan>\nr\n</plan>\n\n<answer>\n{s}\n</answer>", .{answer}) };
        return out;
    }
};

fn gepaMetric(
    arena: Allocator,
    example: *const dszig.Example,
    prediction: *const Record,
    trace: ?*const dszig.Trace,
    pred_name: ?[]const u8,
    _: ?*const dszig.Trace.Step,
) dszig.value.ParseError!dszig.gepa.Feedback {
    const score = try live_suite.exactMetric(arena, example, prediction, trace);
    if (pred_name == null or score != 0) return .{ .score = score };
    const gold = example.record.get("answer").?.string;
    return .{ .score = score, .feedback = std.fmt.allocPrint(arena, "Expected {s}.", .{gold}) catch "" };
}

const Restate = dszig.Signature(struct {
    pub const instructions = "Restate the question.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { restated: []const u8 };
});

/// gen_gepa.py's `TwoStep`: two predictors, so GEPA's merge runs.
const TwoStep = struct {
    restate: dszig.Predict(Restate) = .{},
    generate: dszig.ChainOfThought(live_suite.Dates) = .{},
    compiled: bool = false,

    pub const Input = struct { question: []const u8 };
    pub const Output = dszig.ChainOfThought(live_suite.Dates).Output;

    pub fn forward(self: *const TwoStep, ctx: dszig.Context, input: Input) dszig.predict.Error!Output {
        const r = try self.restate.forward(ctx, .{ .question = input.question });
        return self.generate.forward(ctx, .{ .question = r.restated });
    }
};

test "GEPA candidates, scores and every request match DSPy" {
    try checkGepa(live_suite.DateQA, gepa_case, 150, false);
}

test "GEPA merge on a two-predictor program matches DSPy" {
    try checkGepa(TwoStep, gepa_merge_case, 400, false);
}

test "GEPA with concurrent rollouts makes DSPy's requests and selects DSPy's candidates" {
    try checkGepa(live_suite.DateQA, gepa_case, 150, true);
    try checkGepa(TwoStep, gepa_merge_case, 400, true);
}

fn checkGepa(comptime P: type, case: []const u8, budget: usize, concurrent: bool) !void {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    // ArenaAllocator over the thread-safe testing allocator is thread-safe.
    const arena = arena_state.allocator();
    const limits: dszig.parallel.Limits = .{ .workers = 4, .tasks = 8, .attempts = 8, .failures = 1 };
    // mocks[0] serves the reflection LM and serial runs; the rest serve attempts.
    var mocks: [limits.attempts + 1]GepaMock = @splat(.{ .arena = arena });
    const train = try live_suite.dateExamples(arena, .train);
    const dev = try live_suite.dateExamples(arena, .dev);
    for (&mocks) |*mock| for ([_][]const dszig.Example{ train, dev }) |set| {
        for (set) |ex| try mock.golds.put(arena, ex.record.get("question").?.string, ex.record.get("answer").?.string);
    };
    var contexts: [limits.attempts]dszig.Context = undefined;
    for (&contexts, mocks[1..]) |*context, *mock| context.* = .{ .arena = arena, .lm = mock.lm() };
    const Pool = dszig.Parallel(dszig.GEPA.Run, limits);
    const pool = try testing.allocator.create(Pool);
    defer testing.allocator.destroy(pool);
    try pool.init(testing.io, .{});
    defer pool.deinit();
    const ctx: dszig.Context = .{ .arena = arena, .lm = mocks[0].lm(), .lm_options = .{ .temperature = 0.0 } };
    var gepa: dszig.GEPA = .{ .metric = gepaMetric, .max_metric_calls = budget };
    if (concurrent) gepa.rollouts = .{ .runner = pool.runner(), .contexts = &contexts };
    const result = try gepa.compile(P, arena, ctx, .{}, train, .{ .valset = dev });

    const want = try std.json.parseFromSliceLeaky(Value, arena, case, .{});
    const calls = want.object.get("calls").?.array.items;
    if (concurrent) {
        // Completion order is concurrent; the multiset of requests must match.
        try expectSameCalls(GepaMock, arena, calls, &mocks);
    } else {
        try testing.expectEqual(calls.len, mocks[0].calls.items.len);
        for (calls, mocks[0].calls.items) |w, got| {
            const items = w.array.items;
            try testing.expectEqualStrings(try compact(arena, items[0]), try compact(arena, got.rollout orelse .null));
            try testing.expectEqual(items[1].float, got.temperature.?);
            try testing.expectEqualStrings(items[2].string, &got.digest);
        }
    }
    const candidates = want.object.get("candidates").?.array.items;
    try testing.expectEqual(candidates.len, result.candidates.len);
    for (candidates, result.candidates) |w, got| {
        for (w.array.items, got) |a, b| try testing.expectEqualStrings(a.string, b);
    }
    for (want.object.get("parents").?.array.items, result.parents) |w, got| {
        for (w.array.items, got) |a, b| {
            if (a == .null) try testing.expect(b == null) else try testing.expectEqual(@as(usize, @intCast(a.integer)), b.?);
        }
    }
    for (want.object.get("val_aggregate_scores").?.array.items, result.val_aggregate_scores) |w, got| {
        try testing.expectEqual(w.float, got);
    }
    for (want.object.get("discovery_eval_counts").?.array.items, result.discovery_eval_counts) |w, got| {
        try testing.expectEqual(@as(usize, @intCast(w.integer)), got);
    }
    try testing.expectEqual(@as(usize, @intCast(want.object.get("best_idx").?.integer)), result.best_idx);
    try testing.expectEqual(@as(usize, @intCast(want.object.get("total_metric_calls").?.integer)), result.total_metric_calls);
    try testing.expectEqual(@as(usize, @intCast(want.object.get("num_full_val_evals").?.integer)), result.num_full_val_evals);
    try testing.expectEqualStrings(want.object.get("artifact").?.string, try dszig.state.saveAlloc(arena, &result.program));
}
