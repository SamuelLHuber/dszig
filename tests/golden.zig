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
