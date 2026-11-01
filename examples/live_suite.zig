//! Live-validation suite: the same cases and RAG BootstrapFewShot pipeline
//! as tools/dspy_reference/live_suite.py. `zig build live` runs it on a real
//! OpenAI model and writes a transcript DSPy replays; tests/live_replay.zig
//! replays DSPy's real-model transcript through it.

const std = @import("std");
const dszig = @import("dszig");
const rag = @import("rag_program");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Record = dszig.value.Record;
const Call = dszig.lm.ReplayLM.Call;

pub const temperature = 0.0;

pub const Typed = dszig.Signature(struct {
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

pub const Indented = dszig.Signature(struct {
    pub const instructions = "Classify the text.\n    Keep nested indentation.\n\nBe brief.";
    pub const Input = struct { text: []const u8 };
    pub const Output = struct { label: enum { pos, neg } };
});

pub const qa_demos =
    \\[{"augmented": true, "context": ["a fact"], "question": "q1?", "reasoning": "r1", "answer": "a1"},
    \\ {"question": "q2?", "answer": "a2"}]
;
pub const qa_input: rag.QA.Input = .{
    .context = &.{ "Paris is the capital of France.", "multi\nline «x»" },
    .question = "What is the capital of France?",
};
pub const typed_demos =
    \\[{"text": "good", "tags": ["x"], "flag": true, "count": 3, "ratio": 0.5, "label": "pos", "items": ["a", "b"]},
    \\ {"text": "bad", "flag": false, "label": "neg"}]
;
pub const typed_input: Typed.Input = .{ .text = "great movie", .tags = &.{ "review", "film" } };

/// Gold answers follow an ISO 8601 convention the instructions never state,
/// so the zero-shot baseline is low and only demos can teach it.
pub const Dates = dszig.Signature(struct {
    pub const instructions = "Answer the question with the date it asks for.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

pub const DateQA = struct {
    generate: dszig.ChainOfThought(Dates) = .{},
    compiled: bool = false,

    pub const Input = struct { question: []const u8 };
    pub const Output = dszig.ChainOfThought(Dates).Output;

    pub fn forward(self: *const DateQA, ctx: dszig.Context, input: Input) dszig.predict.Error!Output {
        return self.generate.forward(ctx, .{ .question = input.question });
    }
};

const dates_json = @embedFile("dates.json");

pub fn dateExamples(arena: Allocator, comptime split: enum { train, dev }) ![]const dszig.Example {
    const Rows = struct { note: []const u8, train: []const [2][]const u8, dev: []const [2][]const u8 };
    const rows = try std.json.parseFromSliceLeaky(Rows, arena, dates_json, .{});
    const selected = @field(rows, @tagName(split));
    const out = try arena.alloc(dszig.Example, selected.len);
    for (selected, out) |row, *ex| {
        var record: Record = .empty;
        try record.put(arena, "question", .{ .string = row[0] });
        try record.put(arena, "answer", .{ .string = row[1] });
        ex.* = .{ .record = record, .input_keys = &.{"question"} };
    }
    return out;
}

/// `pred.answer.strip() == example.answer`.
pub fn exactMetric(_: Allocator, example: *const dszig.Example, prediction: *const Record, _: ?*const dszig.Trace) f64 {
    const want = example.record.get("answer").?.string;
    const got = (prediction.get("answer") orelse return 0).string;
    return if (std.mem.eql(u8, std.mem.trim(u8, got, " \t\r\n"), want)) 1 else 0;
}

pub const case_count = 5;

pub const Case = struct {
    name: []const u8,
    /// Null when the program failed (`err` says why).
    outputs: ?Record,
    err: ?anyerror,
};

/// Evaluate on the devset, BootstrapFewShot on the trainset, evaluate again.
pub const Pipeline = struct {
    baseline: f64,
    optimized: f64,
    /// Compiled program in `program.save()` format.
    artifact: []const u8,
};

pub const Result = struct {
    cases: [case_count]Case,
    rag: Pipeline,
    dates: Pipeline,
};

pub fn run(arena: Allocator, gpa: Allocator, lm: dszig.LM) !Result {
    const base: dszig.Context = .{ .arena = arena, .lm = lm, .lm_options = .{ .temperature = temperature } };
    var result: Result = undefined;
    result.cases = .{
        try runCase(dszig.ChainOfThought(rag.QA), base, .chat, "cot_qa_chat", qa_demos, qa_input),
        try runCase(dszig.ChainOfThought(rag.QA), base, .json, "cot_qa_json", qa_demos, qa_input),
        try runCase(dszig.Predict(Typed), base, .chat, "typed_chat", typed_demos, typed_input),
        try runCase(dszig.Predict(Typed), base, .json, "typed_json", typed_demos, typed_input),
        try runCase(dszig.Predict(Indented), base, .chat, "indented_chat", "[]", .{ .text = "meh" }),
    };

    var ctx = base;
    ctx.retriever = rag.corpus.retriever();
    const rag_set = try rag.trainset(arena);
    const rag_optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    result.rag = try pipeline(rag.RAG, arena, gpa, ctx, rag_optimizer, rag_set, rag_set, rag.keywordMetric);
    const dates_optimizer: dszig.BootstrapFewShot = .{ .metric = exactMetric, .max_bootstrapped_demos = 4, .max_labeled_demos = 8 };
    result.dates = try pipeline(DateQA, arena, gpa, ctx, dates_optimizer, try dateExamples(arena, .train), try dateExamples(arena, .dev), exactMetric);
    return result;
}

fn pipeline(
    comptime P: type,
    arena: Allocator,
    gpa: Allocator,
    ctx: dszig.Context,
    optimizer: dszig.BootstrapFewShot,
    trainset: []const dszig.Example,
    devset: []const dszig.Example,
    metric: dszig.optimize.Metric,
) !Pipeline {
    const baseline = try dszig.optimize.evaluate(P, gpa, ctx, &P{}, devset, metric, 10);
    const compiled = try optimizer.compile(P, arena, gpa, ctx, .{}, null, trainset);
    const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
    return .{
        .baseline = baseline.score,
        .optimized = optimized.score,
        .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
    };
}

fn runCase(
    comptime P: type,
    base: dszig.Context,
    adapter: dszig.Adapter,
    name: []const u8,
    demos_json: []const u8,
    input: P.Input,
) !Case {
    var program: P = .{};
    const state = if (@hasField(P, "predict")) &program.predict.state else &program.state;
    state.demos = try parseRecords(base.arena, demos_json);
    var ctx = base;
    ctx.adapter = adapter;
    const output = program.forward(ctx, input) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return .{ .name = name, .outputs = null, .err = err };
    };
    return .{ .name = name, .outputs = try dszig.value.toRecord(base.arena, P.Output, output), .err = null };
}

fn parseRecords(arena: Allocator, json: []const u8) ![]const Record {
    const v = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
    const out = try arena.alloc(Record, v.array.items.len);
    for (v.array.items, out) |item, *o| o.* = item.object;
    return out;
}

/// Calls of a transcript (`{"calls": [{"messages", "completion"}, ...]}`).
pub fn parseCalls(arena: Allocator, transcript: Value) ![]const Call {
    const items = transcript.object.get("calls").?.array.items;
    const calls = try arena.alloc(Call, items.len);
    for (items, calls) |item, *call| {
        const raw = item.object.get("messages").?.array.items;
        const messages = try arena.alloc(dszig.adapter.Message, raw.len);
        for (raw, messages) |m, *dst| dst.* = .{
            .role = std.meta.stringToEnum(dszig.adapter.Role, m.object.get("role").?.string) orelse return error.BadTranscript,
            .content = m.object.get("content").?.string,
        };
        call.* = .{ .messages = messages, .completion = item.object.get("completion").?.string };
    }
    return calls;
}

/// Transcript JSON in the layout tools/dspy_reference/live.py writes.
pub fn writeTranscript(arena: Allocator, runtime: []const u8, model: []const u8, calls: []const Call, result: Result) ![]const u8 {
    var call_values: std.json.Array = .init(arena);
    for (calls) |call| {
        var messages: std.json.Array = .init(arena);
        for (call.messages) |m| {
            var obj: std.json.ObjectMap = .empty;
            try obj.put(arena, "role", .{ .string = m.role.name() });
            try obj.put(arena, "content", .{ .string = m.content });
            try messages.append(.{ .object = obj });
        }
        var obj: std.json.ObjectMap = .empty;
        try obj.put(arena, "messages", .{ .array = messages });
        try obj.put(arena, "completion", .{ .string = call.completion });
        try call_values.append(.{ .object = obj });
    }
    var cases: std.json.Array = .init(arena);
    for (result.cases) |c| {
        var obj: std.json.ObjectMap = .empty;
        try obj.put(arena, "name", .{ .string = c.name });
        if (c.outputs) |o| try obj.put(arena, "outputs", .{ .object = o });
        if (c.err) |e| try obj.put(arena, "error", .{ .string = @errorName(e) });
        try cases.append(.{ .object = obj });
    }
    var root: std.json.ObjectMap = .empty;
    try root.put(arena, "runtime", .{ .string = runtime });
    try root.put(arena, "model", .{ .string = model });
    try root.put(arena, "calls", .{ .array = call_values });
    try root.put(arena, "cases", .{ .array = cases });
    inline for (.{ "rag", "dates" }) |name| {
        const p = @field(result, name);
        var obj: std.json.ObjectMap = .empty;
        try obj.put(arena, "baseline", .{ .float = p.baseline });
        try obj.put(arena, "optimized", .{ .float = p.optimized });
        try obj.put(arena, "artifact", .{ .string = p.artifact });
        try root.put(arena, name, .{ .object = obj });
    }
    var aw: std.Io.Writer.Allocating = .init(arena);
    dszig.pyjson.writeJson(&aw.writer, .{ .object = root }, .indent2) catch return error.OutOfMemory;
    try aw.writer.writeByte('\n');
    return aw.written();
}
