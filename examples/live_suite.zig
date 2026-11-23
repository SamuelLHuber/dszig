//! DSPy 3.5.0 validation suite shared with tools/dspy_reference/live_suite.py.
//! `zig build live` runs every adapter case and optimizer pipeline against a
//! real model and writes a request/response transcript DSPy can replay.
//! tests/live_replay.zig replays independently recorded DSPy 3.5.0 calls.

const std = @import("std");
const dszig = @import("dszig");
const rag = @import("rag_program");
const Allocator = std.mem.Allocator;
const Value = dszig.Value;
const Record = dszig.value.Record;
const Call = dszig.lm.ReplayLM.Call;

pub const target_dspy = "3.5.0";
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
    \\[{"augmented": true, "context": ["a fact"], "question": "q1?", "reasoning": {"content": "r1"}, "answer": "a1"},
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
pub fn exactMetric(_: Allocator, example: *const dszig.Example, prediction: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    const want = example.record.get("answer").?.string;
    const pred = prediction orelse return error.ParseValue;
    const got = (pred.get("answer") orelse return 0).string;
    return if (std.mem.eql(u8, std.mem.trim(u8, got, " \t\r\n"), want)) 1 else 0;
}

pub const case_count = 8;

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
    candidates: []const CandidateSummary = &.{},
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
        try runCase(dszig.ChainOfThought(rag.QA), base, base.adapter, "cot_qa_xml", qa_demos, qa_input),
        try runCase(dszig.Predict(Typed), base, base.adapter, "typed_xml", typed_demos, typed_input),
        try runCase(dszig.Predict(Indented), base, base.adapter, "indented_xml", "[]", .{ .text = "meh" }),
    };

    var ctx = base;
    ctx.retriever = rag.corpus.retriever();
    const rag_set = try rag.trainset(arena);
    const rag_optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    result.rag = try pipeline(rag.RAG, .{}, arena, gpa, ctx, rag_optimizer, rag_set, rag_set, rag.keywordMetric);
    const dates_optimizer: dszig.BootstrapFewShot = .{ .metric = exactMetric, .max_bootstrapped_demos = 4, .max_labeled_demos = 8 };
    const dates_train = try dateExamples(arena, .train);
    const dates_dev = try dateExamples(arena, .dev);
    result.dates = try pipeline(DateQA, .{}, arena, gpa, ctx, dates_optimizer, dates_train, dates_dev, exactMetric);
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = exactMetric,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 2,
        .num_candidate_programs = 4,
    };
    result.dates_search = try pipeline(DateQA, .{}, arena, gpa, ctx, search, dates_train, dates_dev, exactMetric);
    const mipro: dszig.MIPROv2 = .{
        .metric = exactMetric,
        .auto = null,
        .num_candidates = 4,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 2,
    };
    result.dates_mipro = try pipeline(DateQA, .{}, arena, gpa, ctx, mipro, dates_train, dates_dev, exactMetric);
    var aware = mipro;
    aware.num_candidates = 2;
    result.dates_mipro_aware = try pipeline(DateQA, .{}, arena, gpa, ctx, aware, dates_train, dates_dev, exactMetric);
    const react_optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 0 };
    result.rag_react = try pipeline(RagAgent, try RagAgent.init(arena), arena, gpa, ctx, react_optimizer, rag_set, rag_set, rag.keywordMetric);
    const copro: dszig.COPRO = .{ .metric = exactMetric, .breadth = copro_breadth, .depth = copro_depth };
    result.dates_copro = try pipeline(DateQA, .{}, arena, gpa, ctx, copro, dates_train, dates_dev, exactMetric);
    const simba: dszig.SIMBA = .{
        .metric = exactMetric,
        .bsize = simba_bsize,
        .num_candidates = simba_candidates,
        .max_steps = simba_steps,
        .max_demos = simba_demos,
    };
    result.dates_simba = try pipeline(DateQA, .{}, arena, gpa, ctx, simba, dates_train, dates_dev, exactMetric);
    const gepa: dszig.GEPA = .{ .metric = gepaMetric, .max_metric_calls = gepa_budget };
    result.dates_gepa = try pipeline(DateQA, .{}, arena, gpa, ctx, gepa, dates_train, dates_dev, exactMetric);
    return result;
}

pub const gepa_budget = 60;

/// GEPA feedback: the gold answer when the prediction misses.
pub fn gepaMetric(
    arena: Allocator,
    example: *const dszig.Example,
    prediction: *const dszig.optimize.RecordPrediction,
    trace: ?*const dszig.Trace,
    pred_name: ?[]const u8,
    _: ?*const dszig.Trace.Step,
) dszig.value.ParseError!dszig.gepa.Feedback {
    const score = try exactMetric(arena, example, prediction, trace);
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
    const passages = retriever.search(ctx.arena, query.string, 2) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ToolFailed,
    };
    var list: dszig.Array = .init(ctx.arena);
    for (passages) |p| try list.append(.{ .string = p });
    return .{ .array = list };
}

/// Structured history and tool calls, ending in `submit(answer=...)`.
/// `dspy.ReAct("question -> answer", tools=[corpus_search], max_iters=4)`.
pub const RagAgent = dszig.react.ReActIters(AgentQA, &.{.{
    .name = "corpus_search",
    .desc = "Search the corpus for passages relevant to the query.",
    .args = &.{.{ .name = "query", .schema_type = "string" }},
    .func = corpusSearch,
}}, react_max_iters);
fn pipeline(
    comptime P: type,
    student: P,
    arena: Allocator,
    gpa: Allocator,
    ctx: dszig.Context,
    optimizer: anytype,
    trainset: []const dszig.Example,
    devset: []const dszig.Example,
    metric: dszig.optimize.Metric,
) !Pipeline {
    const baseline = try dszig.optimize.evaluate(P, gpa, ctx, &student, devset, metric, 10);
    if (@TypeOf(optimizer) == dszig.MIPROv2) {
        const aware = optimizer.num_candidates == 2;
        const options: dszig.MIPROv2.Options = .{
            .valset = devset,
            .num_trials = if (aware) mipro_aware_trials else mipro_trials,
            .minibatch_size = mipro_minibatch,
            .program_aware_proposer = aware,
            .program_code = dates_program_code,
        };
        const compiled = try optimizer.compile(P, arena, gpa, ctx, null, student, null, trainset, options);
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
        const compiled = try optimizer.compile(P, arena, ctx, student, trainset, .{ .program_code = dates_class_source });
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
        const compiled = try optimizer.compile(P, arena, ctx, student, trainset, .{ .valset = devset });
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
        const compiled = try optimizer.compile(P, arena, gpa, ctx, null, student, trainset);
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
        try optimizer.compile(P, arena, gpa, ctx, student, null, trainset, null, .{ .restrict = null, .labeled_sample = true })
    else bootstrap: {
        var instance = optimizer;
        break :bootstrap try instance.compile(P, arena, gpa, ctx, student, null, trainset);
    };
    const optimized = try dszig.optimize.evaluate(P, gpa, ctx, &compiled.program, devset, metric, 10);
    const candidates = if (search) summaries: {
        const rows = try arena.alloc(CandidateSummary, compiled.candidates.len);
        for (compiled.candidates, rows) |candidate, *row| {
            row.* = .{ .seed = candidate.seed, .score = candidate.score };
        }
        break :summaries rows;
    } else &.{};
    return .{
        .baseline = baseline.score,
        .optimized = optimized.score,
        .artifact = try dszig.state.saveAlloc(arena, &compiled.program),
        .candidates = candidates,
    };
}

const CandidateSummary = struct { seed: i64, score: f64 };

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

/// Reads 3.5.0 request/response transcripts without discarding native reasoning,
/// function calls, or provider continuation state.
/// Historical rows without request options retain `options = null`; they do not
/// silently acquire the current default options.
pub fn parseCalls(arena: Allocator, transcript: Value) ![]const Call {
    const items = transcript.object.get("calls").?.array.items;
    const calls = try arena.alloc(Call, items.len);
    for (items, calls) |item, *call| {
        const message_values = item.object.get("messages").?.array.items;
        const messages = try arena.alloc(dszig.adapter.Message, message_values.len);
        for (message_values, messages) |m, *dst| dst.* = try dszig.lm.messageFromValue(arena, m);
        var completions: []const ?[]const u8 = &.{};
        if (item.object.get("completions")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(?[]const u8, all.array.items.len);
            for (all.array.items, list) |c, *dst| dst.* = switch (c) {
                .null => null,
                .string => |text| text,
                else => return error.BadTranscript,
            };
            completions = list;
        }
        var reasonings: []const ?[]const u8 = &.{};
        if (item.object.get("reasonings")) |all| {
            const list = try arena.alloc(?[]const u8, all.array.items.len);
            for (all.array.items, list) |reasoning, *dst| dst.* = switch (reasoning) {
                .null => null,
                .string => |text| text,
                else => return error.BadTranscript,
            };
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            reasonings = list;
        }
        var tool_calls: []const []const dszig.lm.ToolCall = &.{};
        if (item.object.get("tool_calls")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc([]const dszig.lm.ToolCall, all.array.items.len);
            for (all.array.items, list) |calls_value, *dst| dst.* = try dszig.inference.toolCallsFromValue(arena, calls_value);
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            tool_calls = list;
        }
        var citations: []const []const dszig.inference.CitationPart = &.{};
        if (item.object.get("citations")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc([]const dszig.inference.CitationPart, all.array.items.len);
            for (all.array.items, list) |parts, *dst| dst.* = try dszig.inference.citationsFromValue(arena, parts);
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            citations = list;
        }
        var continuation: []const []const dszig.inference.Continuation = &.{};
        if (item.object.get("continuation")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc([]const dszig.inference.Continuation, all.array.items.len);
            for (all.array.items, list) |state, *dst| dst.* = try dszig.inference.continuationsFromValue(arena, state);
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            continuation = list;
        }
        var provider_data: []const ?Value = &.{};
        if (item.object.get("provider_data")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(?Value, all.array.items.len);
            for (all.array.items, list) |data, *dst| dst.* = switch (data) {
                .null => null,
                .object => data,
                else => return error.BadTranscript,
            };
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            provider_data = list;
        }
        // Transcripts recorded before response metadata have no such key.
        var metadata: []const dszig.lm.ResponseMetadata = &.{};
        if (item.object.get("metadata")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(dszig.lm.ResponseMetadata, all.array.items.len);
            for (all.array.items, list) |entry, *dst| dst.* = try dszig.lm.responseMetadataFromValue(arena, entry);
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            metadata = list;
        }
        var generated_media: []const bool = &.{};
        if (item.object.get("generated_media")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(bool, all.array.items.len);
            for (all.array.items, list) |entry, *dst| {
                if (entry != .bool) return error.BadTranscript;
                dst.* = entry.bool;
            }
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            generated_media = list;
        }
        var logprobs: []const ?[]const dszig.lm.TokenLogprob = &.{};
        if (item.object.get("logprobs")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(?[]const dszig.lm.TokenLogprob, all.array.items.len);
            for (all.array.items, list) |entry, *dst| {
                dst.* = if (entry == .null) null else try dszig.lm.logprobs.fromNativeValue(arena, entry);
            }
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            logprobs = list;
        }
        var logprobs_complete: []const bool = &.{};
        if (item.object.get("logprobs_complete")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(bool, all.array.items.len);
            for (all.array.items, list) |entry, *dst| {
                if (entry != .bool) return error.BadTranscript;
                dst.* = entry.bool;
            }
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            logprobs_complete = list;
        }
        var response_messages: []const ?dszig.sdk_response.Message = &.{};
        if (item.object.get("response_messages")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc(?dszig.sdk_response.Message, all.array.items.len);
            for (all.array.items, list) |entry, *dst| {
                dst.* = if (entry == .null) null else try dszig.sdk_response.Message.fromValue(arena, entry);
            }
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            response_messages = list;
        }
        var adaptations: []const []const dszig.sdk_response.Adaptation = &.{};
        if (item.object.get("adaptations")) |all| {
            if (all != .array) return error.BadTranscript;
            const list = try arena.alloc([]const dszig.sdk_response.Adaptation, all.array.items.len);
            for (all.array.items, list) |entry, *dst|
                dst.* = try dszig.sdk_response.adaptationsFromValue(arena, entry);
            if (list.len != 0 and list.len != (if (completions.len > 0) completions.len else 1)) return error.BadTranscript;
            adaptations = list;
        }
        const options: ?dszig.lm.Options = if (item.object.get("options")) |raw|
            if (raw == .null) null else try dszig.lm.optionsFromValue(arena, raw)
        else
            null;
        const tools = if (item.object.get("tools")) |raw|
            try dszig.inference.toolsFromValue(arena, raw)
        else
            &.{};
        const decision: ?dszig.lm.DecisionRequest = if (item.object.get("decision")) |raw|
            if (raw == .null) null else try dszig.lm.decisionFromValue(arena, raw)
        else
            null;
        call.* = .{
            .messages = messages,
            .completion = switch (item.object.get("completion") orelse return error.BadTranscript) {
                .null => null,
                .string => |text| text,
                else => return error.BadTranscript,
            },
            .completions = completions,
            .reasonings = reasonings,
            .options = options,
            .tools = tools,
            .decision = decision,
            .tool_calls = tool_calls,
            .citations = citations,
            .continuation = continuation,
            .provider_data = provider_data,
            .metadata = metadata,
            .generated_media = generated_media,
            .logprobs = logprobs,
            .logprobs_complete = logprobs_complete,
            .response_messages = response_messages,
            .adaptations = adaptations,
        };
    }
    return calls;
}

/// Transcript JSON in the layout tools/dspy_reference/live.py writes.
/// Capabilities are the actual LM's advertised native features.
pub fn writeTranscript(arena: Allocator, runtime: []const u8, model: []const u8, lm: dszig.LM, calls: []const Call, result: Result) ![]const u8 {
    var call_values: dszig.Array = .init(arena);
    for (calls) |call| {
        var messages: dszig.Array = .init(arena);
        for (call.messages) |m| try messages.append(try dszig.lm.messageValue(arena, m));
        var obj: dszig.ObjectMap = .empty;
        try obj.put(arena, "messages", .{ .array = messages });
        try obj.put(arena, "tools", try dszig.inference.toolsValue(arena, call.tools));
        // Ordinary rows retain the historical layout written by live.py.
        if (call.decision) |decision| try obj.put(arena, "decision", try dszig.lm.decisionValue(arena, decision));
        try obj.put(arena, "completion", if (call.completion) |text| .{ .string = text } else .null);
        if (call.completions.len > 0) {
            var all: dszig.Array = .init(arena);
            for (call.completions) |c| try all.append(if (c) |text| .{ .string = text } else .null);
            try obj.put(arena, "completions", .{ .array = all });
        }
        if (call.reasonings.len > 0) {
            var all: dszig.Array = .init(arena);
            for (call.reasonings) |reasoning| try all.append(if (reasoning) |text| .{ .string = text } else .null);
            try obj.put(arena, "reasonings", .{ .array = all });
        }
        var native_calls: dszig.Array = .init(arena);
        for (call.tool_calls) |calls_value| try native_calls.append(try dszig.inference.toolCallsValue(arena, calls_value));
        try obj.put(arena, "tool_calls", .{ .array = native_calls });
        if (call.citations.len != 0) {
            var citations: dszig.Array = .init(arena);
            for (call.citations) |parts| try citations.append(try dszig.inference.citationsValue(arena, parts));
            try obj.put(arena, "citations", .{ .array = citations });
        }
        var continuation: dszig.Array = .init(arena);
        for (call.continuation) |state| try continuation.append(try dszig.inference.continuationsValue(arena, state));
        try obj.put(arena, "continuation", .{ .array = continuation });
        var provider_data: dszig.Array = .init(arena);
        for (call.provider_data) |data| try provider_data.append(data orelse .null);
        try obj.put(arena, "provider_data", .{ .array = provider_data });
        if (call.metadata.len != 0) {
            var metadata: dszig.Array = .init(arena);
            for (call.metadata) |entry| try metadata.append(try dszig.lm.responseMetadataValue(arena, entry));
            try obj.put(arena, "metadata", .{ .array = metadata });
        }
        if (call.generated_media.len != 0) {
            var flags: dszig.Array = .init(arena);
            for (call.generated_media) |flag| try flags.append(.{ .bool = flag });
            try obj.put(arena, "generated_media", .{ .array = flags });
        }
        if (call.logprobs.len != 0) {
            var scores: dszig.Array = .init(arena);
            for (call.logprobs) |parts| try scores.append(if (parts) |entries| try dszig.lm.logprobs.toNativeValue(arena, entries) else .null);
            try obj.put(arena, "logprobs", .{ .array = scores });
        }
        if (call.logprobs_complete.len != 0) {
            var flags: dszig.Array = .init(arena);
            for (call.logprobs_complete) |flag| try flags.append(.{ .bool = flag });
            try obj.put(arena, "logprobs_complete", .{ .array = flags });
        }
        if (call.response_messages.len != 0) {
            var response_messages: dszig.Array = .init(arena);
            for (call.response_messages) |message|
                try response_messages.append(if (message) |entry| try entry.toValue(arena) else .null);
            try obj.put(arena, "response_messages", .{ .array = response_messages });
        }
        if (call.adaptations.len != 0) {
            var adaptations: dszig.Array = .init(arena);
            for (call.adaptations) |entries|
                try adaptations.append(try dszig.sdk_response.adaptationsValue(arena, entries));
            try obj.put(arena, "adaptations", .{ .array = adaptations });
        }
        if (call.options) |options| try obj.put(arena, "options", try dszig.lm.optionsValue(arena, options));
        try call_values.append(.{ .object = obj });
    }
    var root: dszig.ObjectMap = .empty;
    try root.put(arena, "runtime", .{ .string = runtime });
    try root.put(arena, "target_dspy", .{ .string = target_dspy });
    try root.put(arena, "model", .{ .string = model });
    try root.put(arena, "supports_reasoning", .{ .bool = lm.supports_reasoning });
    try root.put(arena, "supports_response_format", .{ .bool = lm.supports_response_format });
    try root.put(arena, "supports_response_schema", .{ .bool = lm.supports_response_schema });
    try root.put(arena, "supports_function_calling", .{ .bool = lm.supports_function_calling });
    try root.put(arena, "records_tool_result_names", .{ .bool = true });
    if (lm.supports_decision_requests) try root.put(arena, "supports_decision_requests", .{ .bool = true });
    try root.put(arena, "calls", .{ .array = call_values });
    try appendResult(arena, &root, result);
    var aw: std.Io.Writer.Allocating = .init(arena);
    dszig.pyjson.writeJson(&aw.writer, .{ .object = root }, .indent2) catch return error.OutOfMemory;
    try aw.writer.writeByte('\n');
    return aw.written();
}

/// Shared result representation for transcript writing and CLI verification.
fn appendResult(arena: Allocator, root: *dszig.ObjectMap, result: Result) Allocator.Error!void {
    var cases: dszig.Array = .init(arena);
    for (result.cases) |c| {
        var obj: dszig.ObjectMap = .empty;
        try obj.put(arena, "name", .{ .string = c.name });
        if (c.outputs) |o| try obj.put(arena, "outputs", .{ .object = o });
        if (c.err) |e| try obj.put(arena, "error", .{ .string = @errorName(e) });
        try cases.append(.{ .object = obj });
    }
    try root.put(arena, "cases", .{ .array = cases });
    inline for (pipeline_names) |name| {
        const p = @field(result, name);
        var obj: dszig.ObjectMap = .empty;
        try obj.put(arena, "baseline", .{ .float = p.baseline });
        try obj.put(arena, "optimized", .{ .float = p.optimized });
        try obj.put(arena, "artifact", .{ .string = p.artifact });
        var candidates: dszig.Array = .init(arena);
        for (p.candidates) |c| {
            var pair: dszig.Array = .init(arena);
            try pair.append(.{ .integer = c.seed });
            try pair.append(.{ .float = c.score });
            try candidates.append(.{ .array = pair });
        }
        try obj.put(arena, "candidates", .{ .array = candidates });
        var instructions: dszig.Array = .init(arena);
        for (p.instructions) |list| {
            var arr: dszig.Array = .init(arena);
            for (list) |text| try arr.append(.{ .string = text });
            try instructions.append(.{ .array = arr });
        }
        try obj.put(arena, "instructions", .{ .array = instructions });
        var trials: dszig.Array = .init(arena);
        for (p.trials) |t| {
            var row: dszig.Array = .init(arena);
            try row.append(.{ .integer = @intCast(t.number) });
            if (t.params) |params| {
                var arr: dszig.Array = .init(arena);
                for (params) |i| try arr.append(.{ .integer = @intCast(i) });
                try row.append(.{ .array = arr });
            } else try row.append(.null);
            try row.append(.{ .float = t.score });
            try row.append(.{ .bool = t.full_eval });
            try trials.append(.{ .array = row });
        }
        try obj.put(arena, "trials", .{ .array = trials });
        var scores: dszig.Array = .init(arena);
        for (p.scores) |x| try scores.append(.{ .float = x });
        try obj.put(arena, "scores", .{ .array = scores });
        var train_scores: dszig.Array = .init(arena);
        for (p.train_scores) |x| try train_scores.append(if (x) |f| .{ .float = f } else .null);
        try obj.put(arena, "train_scores", .{ .array = train_scores });
        try root.put(arena, name, .{ .object = obj });
    }
}

pub const VerifyTranscriptError = error{
    BadTranscript,
    TargetVersionMismatch,
    CaseCountMismatch,
    CaseNameMismatch,
    CaseOutputMismatch,
    CaseErrorMismatch,
    PipelineArtifactMismatch,
    PipelineBaselineMismatch,
    PipelineOptimizedMismatch,
    PipelineCandidatesMismatch,
    PipelineInstructionsMismatch,
    PipelineTrialsMismatch,
    PipelineScoresMismatch,
    PipelineTrainScoresMismatch,
} || dszig.value.ParseError;

/// Checks the complete suite result against a DSPy 3.5.0 transcript.
/// Exception names are runtime-specific; their presence must agree.
/// Request matching and consumption remain the ReplayLM caller's responsibility.
pub fn verifyTranscript(arena: Allocator, expected: Value, result: Result) VerifyTranscriptError!void {
    if (expected != .object) return error.BadTranscript;
    const target = expected.object.get("target_dspy") orelse return error.BadTranscript;
    if (target != .string or !std.mem.eql(u8, target.string, target_dspy)) return error.TargetVersionMismatch;
    const expected_cases = expected.object.get("cases") orelse return error.BadTranscript;
    if (expected_cases != .array) return error.BadTranscript;
    if (expected_cases.array.items.len != result.cases.len) return error.CaseCountMismatch;

    var actual: dszig.ObjectMap = .empty;
    try appendResult(arena, &actual, result);
    const actual_cases = actual.get("cases").?.array.items;
    for (expected_cases.array.items, actual_cases) |want, got| {
        if (want != .object) return error.BadTranscript;
        const want_name = want.object.get("name") orelse return error.BadTranscript;
        if (!(try dszig.value.pyEqual(want_name, got.object.get("name").?))) return error.CaseNameMismatch;
        const want_outputs = want.object.get("outputs");
        const got_outputs = got.object.get("outputs");
        if ((want_outputs == null) != (got_outputs == null)) return error.CaseOutputMismatch;
        if (want_outputs) |outputs| {
            if (!(try dszig.value.pyEqual(outputs, got_outputs.?))) return error.CaseOutputMismatch;
        }
        if ((want.object.get("error") != null) != (got.object.get("error") != null)) return error.CaseErrorMismatch;
    }

    inline for (pipeline_names) |name| {
        const want = expected.object.get(name) orelse return error.BadTranscript;
        if (want != .object) return error.BadTranscript;
        const got = actual.get(name).?.object;
        const want_artifact = want.object.get("artifact") orelse return error.BadTranscript;
        if (want_artifact != .string) return error.BadTranscript;
        if (!std.mem.eql(u8, want_artifact.string, got.get("artifact").?.string)) return error.PipelineArtifactMismatch;
        inline for (.{
            .{ "baseline", error.PipelineBaselineMismatch },
            .{ "optimized", error.PipelineOptimizedMismatch },
            .{ "candidates", error.PipelineCandidatesMismatch },
            .{ "instructions", error.PipelineInstructionsMismatch },
            .{ "trials", error.PipelineTrialsMismatch },
            .{ "scores", error.PipelineScoresMismatch },
            .{ "train_scores", error.PipelineTrainScoresMismatch },
        }) |field| {
            const expected_field = want.object.get(field[0]) orelse return error.BadTranscript;
            if (!(try dszig.value.pyEqual(expected_field, got.get(field[0]).?))) return field[1];
        }
    }
}
