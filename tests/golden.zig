//! Golden tests against fixtures produced by real DSPy 3.4.1
//! (tools/dspy_reference/gen_fixtures.py). Prompts must match byte for byte,
//! parsed outputs must be equal, and the BootstrapFewShot artifact must be
//! identical to the one DSPy saves for the same program, data and mock LM.

const std = @import("std");
const dszig = @import("dszig");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Record = dszig.value.Record;
const rag = @import("rag_program");

const format_cases = @embedFile("fixtures/format_cases.json");
const bootstrap_rag = @embedFile("fixtures/bootstrap_rag.json");
const bootstrap_multihop = @embedFile("fixtures/bootstrap_multihop.json");
const react_cases = @embedFile("fixtures/react.json");
const ensemble_cases = @embedFile("fixtures/ensemble.json");
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

fn findCase(arena: Allocator, name: []const u8) !std.json.ObjectMap {
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
    const got = history.requests.items[0].messages;
    try testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        try testing.expectEqualStrings(e.object.get("role").?.string, g.role.name());
        try testing.expectEqualStrings(e.object.get("content").?.string, g.content);
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

test "multi-line instructions match DSPy" {
    var p: dszig.Predict(Indented) = .{};
    try runCase(@TypeOf(p), "indented_chat", .chat, &p, &p.state, "[]", .{ .text = "meh" });
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
    try dszig.state.load(out.allocator(), &loaded, bootstrap_rag);
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
    var list: std.json.Array = .init(ctx.arena);
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
    const agent: Agent = .{ .max_iters = @intCast(case.get("max_iters").?.integer) };
    var inputs: Record = .empty;
    try inputs.put(arena, "question", .{ .string = "When did Apollo 11 land?" });
    const prediction = try agent.run(ctx, &inputs);
    try testing.expectEqual(calls.len, history.requests.items.len);
    for (calls, history.requests.items) |c, request| {
        const want = c.object.get("messages").?.array.items;
        try testing.expectEqual(want.len, request.messages.len);
        for (want, request.messages) |w, got| {
            try testing.expectEqualStrings(w.object.get("role").?.string, got.role.name());
            try testing.expectEqualStrings(w.object.get("content").?.string, got.content);
        }
    }
    try testing.expectEqualStrings(try compact(arena, case.get("trajectory").?), try compact(arena, .{ .object = prediction.trajectory }));
    try testing.expectEqualStrings(try compact(arena, case.get("outputs").?), try compact(arena, .{ .object = prediction.outputs }));
}

test "ReAct prompts, trajectory and outputs match DSPy" {
    try checkReact(dszig.ReAct(AgentQA, &.{ search_tool, add_tool }), "react_tools");
    try checkReact(dszig.ReAct(AgentQA, &.{search_tool}), "react_max_iters");
}

test "ReAct turns bad tool arguments into an execution-error observation" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{
        "[[ ## next_thought ## ]]\nx\n\n[[ ## next_tool_name ## ]]\nadd\n\n[[ ## next_tool_args ## ]]\n{\"c\": 1}\n\n[[ ## completed ## ]]",
        "[[ ## next_thought ## ]]\ny\n\n[[ ## next_tool_name ## ]]\nadd\n\n[[ ## next_tool_args ## ]]\n{\"a\": \"one\"}\n\n[[ ## completed ## ]]",
        "[[ ## reasoning ## ]]\nr\n\n[[ ## answer ## ]]\nnone\n\n[[ ## completed ## ]]",
    } };
    const ctx: dszig.Context = .{ .arena = arena, .lm = scripted.lm() };
    const agent: dszig.ReAct(AgentQA, &.{add_tool}) = .{ .max_iters = 2 };
    var inputs: Record = .empty;
    try inputs.put(arena, "question", .{ .string = "?" });
    const prediction = try agent.run(ctx, &inputs);
    try testing.expectEqualStrings("Execution error in add: InvalidArguments", prediction.trajectory.get("observation_0").?.string);
    try testing.expectEqualStrings("Execution error in add: InvalidArguments", prediction.trajectory.get("observation_1").?.string);
    try testing.expectEqualStrings("none", prediction.outputs.get("answer").?.string);
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
