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

/// Evaluate on the devset, compile on the trainset, evaluate again.
pub const Pipeline = struct {
    baseline: f64,
    optimized: f64,
    /// Compiled program in `program.save()` format.
    artifact: []const u8,
    /// Random-search candidates (`candidate_programs`); empty otherwise.
    candidates: []const dszig.optimize.BootstrapFewShotWithRandomSearch.Candidate = &.{},
    /// MIPROv2 only: instruction candidates, `trial_logs` and the scores of
    /// `candidate_programs`.
    instructions: []const []const []const u8 = &.{},
    trials: []const dszig.MIPROv2.TrialLog = &.{},
    scores: []const f64 = &.{},
    /// SIMBA only: `trial_logs[i].get("train_score")`.
    train_scores: []const ?f64 = &.{},
};

pub const pipeline_names = .{ "rag", "dates", "dates_search", "dates_mipro", "dates_mipro_aware", "rag_react", "dates_copro", "dates_simba", "dates_gepa" };

pub const Result = struct {
    cases: [case_count]Case,
    rag: Pipeline,
    dates: Pipeline,
    dates_search: Pipeline,
    dates_mipro: Pipeline,
    dates_mipro_aware: Pipeline,
    rag_react: Pipeline,
    dates_copro: Pipeline,
    dates_simba: Pipeline,
    dates_gepa: Pipeline,
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
    const dates_train = try dateExamples(arena, .train);
    const dates_dev = try dateExamples(arena, .dev);
    result.dates = try pipeline(DateQA, arena, gpa, ctx, dates_optimizer, dates_train, dates_dev, exactMetric);
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = exactMetric,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 2,
        .num_candidate_programs = 4,
    };
    result.dates_search = try pipeline(DateQA, arena, gpa, ctx, search, dates_train, dates_dev, exactMetric);
    const mipro: dszig.MIPROv2 = .{
        .metric = exactMetric,
        .auto = null,
        .num_candidates = 4,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 2,
    };
    result.dates_mipro = try pipeline(DateQA, arena, gpa, ctx, mipro, dates_train, dates_dev, exactMetric);
    var aware = mipro;
    aware.num_candidates = 2;
    result.dates_mipro_aware = try pipeline(DateQA, arena, gpa, ctx, aware, dates_train, dates_dev, exactMetric);
    const react_optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 0 };
    result.rag_react = try pipeline(RagAgent, arena, gpa, ctx, react_optimizer, rag_set, rag_set, rag.keywordMetric);
    const copro: dszig.COPRO = .{ .metric = exactMetric, .breadth = copro_breadth, .depth = copro_depth };
    result.dates_copro = try pipeline(DateQA, arena, gpa, ctx, copro, dates_train, dates_dev, exactMetric);
    const simba: dszig.SIMBA = .{
        .metric = exactMetric,
        .bsize = simba_bsize,
        .num_candidates = simba_candidates,
        .max_steps = simba_steps,
        .max_demos = simba_demos,
    };
    result.dates_simba = try pipeline(DateQA, arena, gpa, ctx, simba, dates_train, dates_dev, exactMetric);
    const gepa: dszig.GEPA = .{ .metric = gepaMetric, .max_metric_calls = gepa_budget };
    result.dates_gepa = try pipeline(DateQA, arena, gpa, ctx, gepa, dates_train, dates_dev, exactMetric);
    return result;
}

pub const gepa_budget = 60;

/// GEPA feedback: the gold answer when the prediction misses.
pub fn gepaMetric(
    arena: Allocator,
    example: *const dszig.Example,
    prediction: *const Record,
    trace: ?*const dszig.Trace,
    pred_name: ?[]const u8,
    _: ?*const dszig.Trace.Step,
) dszig.gepa.Feedback {
    const score = exactMetric(arena, example, prediction, trace);
    if (pred_name == null or score != 0) return .{ .score = score };
    const gold = example.record.get("answer").?.string;
    return .{ .score = score, .feedback = std.fmt.allocPrint(arena, "Expected {s}.", .{gold}) catch "" };
}

pub const mipro_trials = 12;
pub const mipro_minibatch = 4;
pub const mipro_aware_trials = 4;
pub const copro_breadth = 3;
pub const copro_depth = 2;
/// DSPy's `get_dspy_source_code(DateQA())`, which the program-aware proposer
/// reads; tools/dspy_reference/gen_program_code.py regenerates it.
pub const dates_program_code = @embedFile("dates_program_code.txt");
/// `inspect.getsource(DateQA)`, which SIMBA's rule writer reads.
pub const dates_class_source = @embedFile("dates_class_source.txt");
pub const simba_bsize = 4;
pub const simba_candidates = 2;
pub const simba_steps = 2;
pub const simba_demos = 2;
pub const react_max_iters = 4;

pub const AgentQA = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

fn corpusSearch(ctx: dszig.Context, args: *const Record) dszig.react.ToolError!Value {
    const query = args.get("query") orelse return error.InvalidArguments;
    const retriever = ctx.retriever orelse return error.ToolFailed;
    const passages = try retriever.search(ctx.arena, query.string, 2);
    var list: std.json.Array = .init(ctx.arena);
    for (passages) |p| try list.append(.{ .string = p });
    return .{ .array = list };
}

/// `dspy.ReAct("question -> answer", tools=[corpus_search], max_iters=4)`.
pub const RagAgent = dszig.react.ReActIters(AgentQA, &.{.{
    .name = "corpus_search",
    .desc = "Search the corpus for passages relevant to the query.",
    .args = &.{.{ .name = "query", .schema_type = "string" }},
    .func = corpusSearch,
}}, react_max_iters);
fn pipeline(
    comptime P: type,
    arena: Allocator,
    gpa: Allocator,
    ctx: dszig.Context,
    optimizer: anytype,
    trainset: []const dszig.Example,
    devset: []const dszig.Example,
    metric: dszig.optimize.Metric,
) !Pipeline {
    const baseline = try dszig.optimize.evaluate(P, gpa, ctx, &P{}, devset, metric, 10);
    if (@TypeOf(optimizer) == dszig.MIPROv2) {
        const aware = optimizer.num_candidates == 2;
        const options: dszig.MIPROv2.Options = .{
            .valset = devset,
            .num_trials = if (aware) mipro_aware_trials else mipro_trials,
            .minibatch_size = mipro_minibatch,
            .program_aware_proposer = aware,
            .program_code = dates_program_code,
        };
        const compiled = try optimizer.compile(P, arena, gpa, ctx, null, .{}, null, trainset, options);
        const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
        return .{
            .baseline = baseline.score,
            .optimized = optimized.score,
            .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
            .instructions = compiled.instruction_candidates,
            .trials = compiled.trials,
            .scores = compiled.candidates,
        };
    }
    if (@TypeOf(optimizer) == dszig.SIMBA) {
        const compiled = try optimizer.compile(P, arena, ctx, .{}, trainset, .{ .program_code = dates_class_source });
        const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
        return .{
            .baseline = baseline.score,
            .optimized = optimized.score,
            .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
            .scores = compiled.scores,
            .train_scores = compiled.train_scores,
        };
    }
    if (@TypeOf(optimizer) == dszig.GEPA) {
        const compiled = try optimizer.compile(P, arena, ctx, .{}, trainset, .{ .valset = devset });
        const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
        return .{
            .baseline = baseline.score,
            .optimized = optimized.score,
            .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
            .instructions = compiled.candidates,
            .scores = compiled.val_aggregate_scores,
        };
    }
    if (@TypeOf(optimizer) == dszig.COPRO) {
        const compiled = try optimizer.compile(P, arena, gpa, ctx, null, .{}, trainset);
        const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
        const scores = try arena.alloc(f64, compiled.candidates.len);
        for (scores, compiled.candidates) |*s, c| s.* = c.score;
        return .{
            .baseline = baseline.score,
            .optimized = optimized.score,
            .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
            .scores = scores,
        };
    }
    const search = @TypeOf(optimizer) == dszig.BootstrapFewShotWithRandomSearch;
    const compiled = if (search)
        try optimizer.compile(P, arena, gpa, ctx, .{}, null, trainset, null)
    else
        try optimizer.compile(P, arena, gpa, ctx, .{}, null, trainset);
    const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
    return .{
        .baseline = baseline.score,
        .optimized = optimized.score,
        .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
        .candidates = if (search) compiled.candidates else &.{},
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
        var completions: []const []const u8 = &.{};
        if (item.object.get("completions")) |all| {
            const list = try arena.alloc([]const u8, all.array.items.len);
            for (all.array.items, list) |c, *dst| dst.* = c.string;
            completions = list;
        }
        call.* = .{
            .messages = messages,
            .completion = item.object.get("completion").?.string,
            .completions = completions,
        };
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
        if (call.completions.len > 0) {
            var all: std.json.Array = .init(arena);
            for (call.completions) |c| try all.append(.{ .string = c });
            try obj.put(arena, "completions", .{ .array = all });
        }
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
    inline for (pipeline_names) |name| {
        const p = @field(result, name);
        var obj: std.json.ObjectMap = .empty;
        try obj.put(arena, "baseline", .{ .float = p.baseline });
        try obj.put(arena, "optimized", .{ .float = p.optimized });
        try obj.put(arena, "artifact", .{ .string = p.artifact });
        var candidates: std.json.Array = .init(arena);
        for (p.candidates) |c| {
            var pair: std.json.Array = .init(arena);
            try pair.append(.{ .integer = c.seed });
            try pair.append(.{ .float = c.score });
            try candidates.append(.{ .array = pair });
        }
        try obj.put(arena, "candidates", .{ .array = candidates });
        var instructions: std.json.Array = .init(arena);
        for (p.instructions) |list| {
            var arr: std.json.Array = .init(arena);
            for (list) |text| try arr.append(.{ .string = text });
            try instructions.append(.{ .array = arr });
        }
        try obj.put(arena, "instructions", .{ .array = instructions });
        var trials: std.json.Array = .init(arena);
        for (p.trials) |t| {
            var row: std.json.Array = .init(arena);
            try row.append(.{ .integer = @intCast(t.number) });
            if (t.params) |params| {
                var arr: std.json.Array = .init(arena);
                for (params) |i| try arr.append(.{ .integer = @intCast(i) });
                try row.append(.{ .array = arr });
            } else try row.append(.null);
            try row.append(.{ .float = t.score });
            try row.append(.{ .bool = t.full_eval });
            try trials.append(.{ .array = row });
        }
        try obj.put(arena, "trials", .{ .array = trials });
        var scores: std.json.Array = .init(arena);
        for (p.scores) |x| try scores.append(.{ .float = x });
        try obj.put(arena, "scores", .{ .array = scores });
        var train_scores: std.json.Array = .init(arena);
        for (p.train_scores) |x| try train_scores.append(if (x) |f| .{ .float = f } else .null);
        try obj.put(arena, "train_scores", .{ .array = train_scores });
        try root.put(arena, name, .{ .object = obj });
    }
    var aw: std.Io.Writer.Allocating = .init(arena);
    dszig.pyjson.writeJson(&aw.writer, .{ .object = root }, .indent2) catch return error.OutOfMemory;
    try aw.writer.writeByte('\n');
    return aw.written();
}
