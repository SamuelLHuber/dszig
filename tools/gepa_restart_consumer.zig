//! Public GEPA.compile consumer for the pinned two-process reference check.
const std = @import("std");
const dz = @import("dszig");
const GEPA = dz.GEPA;
const Sig = dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});
const P = dz.Predict(Sig);
const IntP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: i64 };
}));
const WideIntP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: i128 };
}));
const wide_low = "-9223372036854775809";
const wide_high = "9223372036854775811";
const CodeP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: dz.Code, note: []const u8 };
}));
const JavaCode = dz.CodeWithLanguageIdentity("java", enum { first });
const OtherJavaCode = dz.CodeWithLanguageIdentity("java", enum { second });
const code_classes: []const GEPA.CheckpointCode = &.{
    .bind(JavaCode, "first-java"),
    .bind(OtherJavaCode, "second-java"),
};
const swapped_code_classes: []const GEPA.CheckpointCode = &.{
    .bind(OtherJavaCode, "first-java"),
    .bind(JavaCode, "second-java"),
};
const SpecializedCodeP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: JavaCode, second: OtherJavaCode, note: []const u8 };
}));
const BoxP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: dz.BoundingBox, note: dz.value.Record, boxes: []const dz.BoundingBox };
}));
const ConfiguredNoul = dz.NoulWithCriteria(&.{.{ .value = .{ .bool = true }, .description_json = "\"Meets the restart criterion.\"" }});
const RevisedNoul = dz.NoulWithCriteria(&.{.{ .value = .{ .bool = true }, .description_json = "\"Different criterion.\"" }});
const NoulP = NoulPredict(dz.Noul);
const ConfiguredNoulP = NoulPredict(ConfiguredNoul);
const Rubric = dz.ScoreWithLevels("[\"poor\",\"excellent\"]");
const ScoreP = NoulPredict(Rubric);
const Category = dz.ChoiceWithOptions(&.{
    .{ .value = .{ .integer = 7 }, .description_json = "\"integer option\"" },
    .{ .value = .{ .bool = true }, .description_json = "\"boolean option\"" },
});
const ChoiceP = NoulPredict(Category);

fn NoulPredict(comptime Decision: type) type {
    return dz.Predict(dz.Signature(struct {
        pub const instructions = "Seed.";
        pub const Input = Sig.Input;
        pub const Output = struct { answer: Decision };
        pub const descriptions = .{ .answer = "Whether the case passes." };
    }));
}
const CitationP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: dz.Citations, nested: []const dz.Citations.Citation, note: dz.value.Record };
}));
const citation_json = "{\"cited_text\":\"literal 雪\",\"document_index\":18446744073709551616}";
const citation_xml = "<answer>[]</answer><nested>[" ++ citation_json ++ "]</nested><note>" ++ citation_json ++ "</note>";
const low_citations: []const dz.lm.CitationPart = &.{ .{ .text = "archived µ", .title = "Source" }, .{ .url = "https://example.invalid/topic" } };
const high_citations: []const dz.lm.CitationPart = &.{ .{ .text = "winner 雪", .title = "Source" }, .{ .url = "https://example.invalid/topic" } };
const xml_box_low = boxXml("0.5");
const xml_box_high = boxXml("1.0");

fn boxXml(comptime right: []const u8) []const u8 {
    const box = "{\"top\":0.125,\"bottom\":0.875,\"left\":0.0,\"right\":" ++ right ++ "}";
    return "<answer>" ++ box ++ "</answer><note>" ++ box ++ "</note><boxes>[" ++ box ++ "]</boxes>";
}

const code_low = "print('µ')";
const code_high = "print('winner')";
const xml_code_low = "<answer>```python\n" ++ code_low ++ "\n```</answer><note>" ++ code_low ++ "</note>";
const xml_code_high = "<answer>```python\n" ++ code_high ++ "\n```</answer><note>" ++ code_high ++ "</note>";
const xml_specialized_low = specializedXml(code_low);
const xml_specialized_high = specializedXml(code_high);

fn specializedXml(comptime text: []const u8) []const u8 {
    const block = "```java\n" ++ text ++ "\n```";
    return "<answer>" ++ block ++ "</answer><second>" ++ block ++ "</second><note>" ++ text ++ "</note>";
}

fn DefaultPredict(comptime answer_default: []const u8) type {
    return dz.Predict(dz.Signature(struct {
        pub const instructions = "Seed.";
        pub const Input = Sig.Input;
        pub const Output = struct { answer: []const u8 = answer_default, status: []const u8 };
    }));
}

const DecisionP = dz.Predict(dz.Signature(struct {
    pub const instructions = "Seed.";
    pub const Input = Sig.Input;
    pub const Output = struct { answer: bool };
    pub const descriptions = .{ .answer = "Decide whether the case qualifies." };
}));

const original_criterion = "Exclude broken cases.";
const revised_criterion = "Include broken cases.";

fn AnnotatedDecisionPredict(comptime criterion: []const u8) type {
    const Noul = dz.NoulWithCriteria(&.{.{
        .value = .{ .bool = true },
        .description_json = "\"" ++ criterion ++ "\"",
    }});
    return dz.Predict(dz.Signature(struct {
        pub const instructions = "Seed.";
        pub const Input = Sig.Input;
        pub const Output = struct { answer: bool };
        pub const descriptions = .{ .answer = .{
            .text = "Decide whether the case qualifies.",
            .metadata = &.{Noul},
        } };
    }));
}

const CoT = dz.ChainOfThought(Sig);

const ToolSig = dz.Signature(struct {
    pub const Input = struct { question: []const u8, tools: []const dz.value.Record };
    pub const Output = struct { answer: []const u8, calls: dz.ToolCalls = .{} };
});
const ToolP = dz.Predict(ToolSig);
const tool_fields = blk: {
    var fields = ToolSig.fields[0..ToolSig.fields.len].*;
    fields[1].native_tools = true;
    break :blk fields;
};

const Workload = struct {
    resume_run: bool,
    sparse: bool,
    model: dz.lm.ScriptedLM,

    fn metric(_: std.mem.Allocator, gold: *const dz.Example, pred: *const dz.optimize.RecordPrediction, _: ?*const dz.Trace, _: ?[]const u8, _: ?*const dz.Trace.Step) dz.value.ParseError!dz.gepa.Feedback {
        const question = gold.record.get("question").?.string;
        if (std.mem.eql(u8, question, "nan-case")) return .{ .score = std.math.nan(f64) };
        const answer = pred.get("answer").?;
        if (switch (answer) {
            .string => |text| std.mem.eql(u8, text, "high"),
            .integer => |n| n == 1,
            .number_string => |text| std.mem.eql(u8, text, wide_high),
            .bool => |b| b,
            .custom => |custom| if (custom.get(dz.Noul)) |noul|
                noul.value
            else if (custom.get(ConfiguredNoul)) |noul|
                noul.value
            else if (custom.get(Rubric)) |score|
                score.value > 0.5
            else if (custom.get(Category)) |category|
                category.value == .bool and category.value.bool
            else if (custom.get(dz.BoundingBox)) |box|
                box.right == 1.0
            else if (custom.get(dz.Citations)) |citations|
                citations.len() != 0 and std.mem.eql(u8, citations.citations[0].cited_text, "winner 雪")
            else
                std.mem.eql(u8, (dz.Code.fromValue(answer) catch return error.ParseValue).code, code_high),
            else => unreachable,
        }) return .{ .score = 1 };
        const broken = std.mem.eql(u8, question, "broken-case");
        return .{ .score = if (broken and gold.record.get("training").?.bool) 0.5 else 0 };
    }

    fn sample(ptr: *anyopaque, _: *const GEPA.BatchSample) ![]const usize {
        const self: *const Workload = @ptrCast(@alignCast(ptr));
        return if (self.resume_run) &.{0} else if (self.sparse) &.{2} else &.{1};
    }

    fn seedBatch(_: *anyopaque, _: []const dz.Example) ![]const usize {
        return &.{ 2, 0 };
    }

    fn valBatch(_: *anyopaque, _: *const GEPA.ValidationState) ![]const usize {
        return &.{ 2, 1, 0 };
    }

    fn propose(_: *anyopaque, _: dz.Context, _: *const GEPA.InstructionProposal) ![]const GEPA.Instruction {
        return &.{.{ .name = "self", .text = "Winner." }};
    }

    fn proposeCoT(_: *anyopaque, _: dz.Context, _: *const GEPA.InstructionProposal) ![]const GEPA.Instruction {
        return &.{.{ .name = "predict", .text = "Winner." }};
    }
};

fn checkCodeRecord(comptime specialized: bool, record: *const dz.value.Record) ![]const u8 {
    const answer = record.get("answer") orelse return error.MissingCode;
    if (answer != .custom) return error.MissingCode;
    const code = answer.custom.get(if (specialized) JavaCode else dz.Code) orelse return error.MissingCode;
    if (specialized) {
        const second = record.get("second") orelse return error.MissingCode;
        if (second != .custom) return error.MissingCode;
        const other = second.custom.get(OtherJavaCode) orelse return error.MissingCode;
        if (answer.custom.sameType(second.custom) or !std.mem.eql(u8, code.code, other.code))
            return error.UnexpectedCodeClass;
    }
    const note = record.get("note") orelse return error.MissingNote;
    if (note != .string or !std.mem.eql(u8, code.code, note.string))
        return error.UnexpectedNote;
    return code.code;
}

fn checkBoxRecord(record: *const dz.value.Record) !dz.BoundingBox {
    const answer = record.get("answer") orelse return error.MissingBox;
    const note = record.get("note") orelse return error.MissingBox;
    const boxes = record.get("boxes") orelse return error.MissingBox;
    if (answer != .custom or note != .object or boxes != .array or boxes.array.items.len != 1)
        return error.UnexpectedBoxType;
    const box = (answer.custom.get(dz.BoundingBox) orelse return error.UnexpectedBoxType).*;
    const nested = boxes.array.items[0];
    if (nested != .custom) return error.UnexpectedBoxType;
    const nested_box = (nested.custom.get(dz.BoundingBox) orelse return error.UnexpectedBoxType).*;
    if (!std.meta.eql(box, nested_box) or !std.meta.eql(box, try dz.BoundingBox.fromValue(note)))
        return error.UnexpectedGeometry;
    return box;
}

fn checkNoul(comptime T: type, input: ?dz.Value) !*const T {
    const answer = input orelse return error.MissingNoul;
    if (answer != .custom) return error.UnexpectedNoulType;
    return answer.custom.get(T) orelse error.UnexpectedNoulType;
}

fn checkChoice(input: ?dz.Value) !*const Category {
    const answer = input orelse return error.MissingChoice;
    if (answer != .custom) return error.UnexpectedChoiceType;
    const category = answer.custom.get(Category) orelse return error.UnexpectedChoiceType;
    if (!(category.value == .integer and category.value.integer == 7) and
        !(category.value == .bool and category.value.bool)) return error.UnexpectedChoiceValue;
    const probs = category.probabilities orelse return error.MissingChoiceEvidence;
    if (probs != .object or probs.object.count() != 2 or
        (probs.object.get("7") orelse return error.MissingChoiceEvidence) != .float or
        (probs.object.get("True") orelse return error.MissingChoiceEvidence) != .float)
        return error.UnexpectedChoiceEvidence;
    return category;
}

fn checkScore(input: ?dz.Value) !*const Rubric {
    const answer = input orelse return error.MissingScore;
    if (answer != .custom) return error.UnexpectedScoreType;
    const score = answer.custom.get(Rubric) orelse return error.UnexpectedScoreType;
    const level = (score.level orelse return error.UnexpectedScoreEvidence).value;
    if (level != .integer) return error.UnexpectedScoreEvidence;
    const probs = dz.python_value.view(score.probabilities orelse return error.UnexpectedScoreEvidence) orelse return error.UnexpectedScoreEvidence;
    if (probs != .dictionary or probs.dictionary.len != 2) return error.UnexpectedScoreEvidence;
    for (probs.dictionary, 0..) |entry, i| {
        if (entry.key != .integer or entry.key.integer != i or entry.value != .float)
            return error.UnexpectedScoreEvidence;
    }
    return score;
}

fn checkCitationRecord(record: *const dz.value.Record) !void {
    const answer = record.get("answer") orelse return error.MissingCitations;
    const nested = record.get("nested") orelse return error.MissingCitations;
    const note = record.get("note") orelse return error.MissingCitations;
    if (answer != .custom or nested != .array or nested.array.items.len != 1 or note != .object)
        return error.UnexpectedCitationType;
    const native = answer.custom.get(dz.Citations) orelse return error.UnexpectedCitationType;
    if (native.len() != 2) return error.MissingCitations;
    const item = nested.array.items[0];
    if (item != .custom) return error.UnexpectedCitationType;
    const literal = item.custom.get(dz.Citations.Citation) orelse return error.UnexpectedCitationType;
    const index = literal.document_index orelse return error.MissingCitations;
    if (!std.mem.eql(u8, literal.cited_text, "literal 雪") or index.value != .number_string or
        !std.mem.eql(u8, index.value.number_string, "18446744073709551616"))
        return error.UnexpectedCitationValue;
}

const pair_tool: dz.Tool = .{ .name = "lookup", .func = lookup };

fn lookup(ctx: dz.Context, _: *const dz.value.Record) dz.react.ToolError!dz.Value {
    const row = .{ .pair = [2]i64{ 7, 9 }, .list = @as([]const i64, &.{ 7, 9 }) };
    return dz.value.toValue(ctx.arena, @TypeOf(row), row);
}

fn reactReply(comptime thought: []const u8, comptime name: []const u8, comptime args: []const u8) []const u8 {
    return "<next_thought>" ++ thought ++ "</next_thought><tool_calls>{\"tool_calls\":[{\"id\":\"" ++ name ++
        "-id\",\"name\":\"" ++ name ++ "\",\"args\":" ++ args ++ "}]}</tool_calls>";
}

fn runReactHistory(init: std.process.Init, dir: []const u8, phase: []const u8) !void {
    const start = std.mem.endsWith(u8, phase, "-start");
    const invalid = std.mem.endsWith(u8, phase, "-invalid");
    const native = std.mem.startsWith(u8, phase, "react-history-native-");
    const arena = init.arena.allocator();
    const Agent = dz.ReAct(Sig, &.{pair_tool});
    var student = try Agent.init(arena);
    student.max_iters = 2;
    const inspect = reactReply("Look up the pair.", "lookup", "{}");
    const low = reactReply("Archived.", "submit", "{\"answer\":\"low\"}");
    const rerun = reactReply("Rerun.", "submit", "{\"answer\":\"rerun\"}");
    const high = reactReply("New.", "submit", "{\"answer\":\"high\"}");
    const replies: []const []const u8 = if (start)
        &.{ inspect, low, inspect, low }
    else
        &.{ inspect, rerun, inspect, rerun, inspect, low, inspect, high, inspect, high, inspect, low };
    const native_replies = try arena.alloc([]const u8, replies.len);
    const native_calls = try arena.alloc([]const dz.lm.ToolCall, replies.len);
    if (native) {
        const payload = try dz.json_graph.fromValue(arena, try dz.json_repair.parseJson(arena,
            \\{"item_id":"item_雪","flags":[true,null,3]}
        ));
        const states: []const dz.lm.Continuation = &.{
            .{ .provider = "openai", .kind = "fixture", .data = .borrowed(payload) },
            .{ .provider = "openai", .kind = "empty", .data = .borrowed(try dz.json_graph.fromValue(arena, .{ .object = .empty })) },
        };
        for (replies, native_replies, native_calls, 0..) |reply, *text, *calls, i| {
            text.* = reply[0..std.mem.indexOf(u8, reply, "<tool_calls>").?];
            const call = try arena.create(dz.lm.ToolCall);
            call.* = if (i % 2 == 0)
                .{ .id = "lookup-id", .name = "lookup", .arguments = "{}", .continuation = states }
            else
                .{ .id = "submit-id", .name = "submit", .arguments = reply[std.mem.indexOf(u8, reply, "\"args\":").? + "\"args\":".len .. reply.len - "}]}</tool_calls>".len] };
            calls.* = call[0..1];
        }
    }
    var history: dz.lm.History = .{ .arena = arena };
    var model: dz.lm.ScriptedLM = .{
        .responses = if (native) native_replies else replies,
        .tool_calls = if (native) native_calls else &.{},
        .supports_function_calling = native,
        .history = &history,
    };
    const train = [_]dz.Example{try dz.Example.fromJson(arena, "{\"question\":\"train\"}", &.{"question"})};
    const val = [_]dz.Example{
        try dz.Example.fromJson(arena, "{\"question\":\"left\"}", &.{"question"}),
        try dz.Example.fromJson(arena, "{\"question\":\"right\"}", &.{"question"}),
    };
    const Callbacks = struct {
        fn metric(_: std.mem.Allocator, _: *const dz.Example, pred: *const dz.optimize.RecordPrediction, _: ?*const dz.Trace, _: ?[]const u8, _: ?*const dz.Trace.Step) dz.value.ParseError!dz.gepa.Feedback {
            return .{ .score = if (std.mem.eql(u8, pred.get("answer").?.string, "high")) 1 else 0 };
        }
        fn sample(_: *anyopaque, _: *const GEPA.BatchSample) ![]const usize {
            return &.{0};
        }
        fn propose(_: *anyopaque, _: dz.Context, _: *const GEPA.InstructionProposal) ![]const GEPA.Instruction {
            return &.{.{ .name = "react", .text = "Winner." }};
        }
    };
    const result = (GEPA{
        .metric = Callbacks.metric,
        .max_metric_calls = if (start) 2 else 6,
        .track_best_outputs = true,
        .use_merge = false,
        .batch_sampler = .{ .ptr = &model, .max_batch_size = 1, .next = Callbacks.sample },
        .instruction_proposer = .{ .ptr = &model, .propose = Callbacks.propose },
    }).compile(Agent, arena, .{
        .workspace = &workspace,
        .arena = arena,
        .lm = model.lm(),
        .json_fallback = false,
        .use_native_function_calling = native,
    }, student, &train, .{
        .valset = &val,
        .checkpoint = .{
            .io = init.io,
            .path = try std.fs.path.join(arena, &.{ dir, "native-checkpoint.json" }),
            .revision = "react-history",
        },
    }) catch |err| {
        if (invalid and err == error.InvalidGepaCheckpoint and model.index == 0) return;
        return err;
    };
    if (invalid or model.index != model.responses.len) return error.UnexpectedRestart;
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    var json: std.json.Stringify = .{ .writer = &stdout.interface };
    try json.beginObject();
    try json.objectField("budget");
    try json.write(result.total_metric_calls);
    try json.objectField("best");
    try json.write(result.best_idx);
    try json.objectField("scores");
    try json.write(result.val_subscores);
    try json.objectField("calls");
    try json.beginArray();
    for (history.requests.items, 0..) |request, i| {
        const user = request.messages[request.messages.len - 1].content.text;
        const question = if (i % 2 == 0) for ([_][]const u8{ "left", "right", "train" }) |name| {
            const field = try std.fmt.allocPrint(arena, "<question>\n{s}\n</question>", .{name});
            if (std.mem.indexOf(u8, user, field) != null) break name;
        } else return error.UnexpectedQuestion else "";
        try json.write(.{ std.mem.indexOf(u8, request.messages[0].content.text, "Winner.") != null, question });
    }
    try json.endArray();
    try json.objectField("outputs");
    try json.beginArray();
    for (result.best_outputs_valset.?) |row| {
        try json.beginArray();
        for (row) |entry| {
            const pred = switch (entry.output) {
                .prediction => |p| p,
                else => return error.MissingHistory,
            };
            const conversation = try dz.History.fromValue(arena, pred.get("history").?);
            const first = try dz.ToolCalls.fromValue(arena, conversation.messages[0].get("tool_calls").?);
            const observation = first.tool_call_results.?.tool_call_results[0].value.object;
            const pair = dz.python_value.view(observation.get("pair").?) orelse return error.MissingTuple;
            if (pair != .tuple or observation.get("list").? != .array) return error.UnexpectedObservation;
            const thought = conversation.messages[1].get("next_thought").?;
            if (thought != .custom or thought.custom.get(dz.Reasoning) == null)
                return error.MissingReasoning;
            const projected = try dz.value.serializedValue(arena, .{ .object = pred.fields().* });
            if (native) {
                const replay_history = if (!start and entry.candidate_idx == 0) replay: {
                    const copy = try conversation.clone(arena);
                    const event_calls = copy.messages[0].get("tool_calls").?.object;
                    const call = &event_calls.get("tool_calls").?.array.items[0].object;
                    const empty = &call.get("continuation").?.array.items[1].object;
                    if (!empty.orderedRemove("data")) return error.MissingContinuationData;
                    break :replay copy;
                } else conversation;
                const replay = try replay_history.format(.xml, arena, .{
                    .instructions = "",
                    .inputs = &.{},
                    .outputs = Agent.react_fields[Agent.react_fields.len - 2 ..],
                    .use_native_function_calling = true,
                });
                if (!start and entry.candidate_idx == 0) {
                    if (init.environ_map.get("DSZIG_ORACLE_RESPONSES_URL")) |url|
                        try sendArchivedHistory(init.io, arena, url, replay);
                }
                try json.write(.{ entry.candidate_idx, projected, try nativeHistoryReplay(arena, replay) });
            } else {
                try json.write(.{ entry.candidate_idx, projected });
            }
        }
        try json.endArray();
    }
    try json.endArray();
    try json.endObject();
    try stdout.interface.flush();
}

fn sendArchivedHistory(io: std.Io, arena: std.mem.Allocator, url: []const u8, messages: []const dz.adapter.Message) !void {
    var client: std.http.Client = .{ .io = io, .allocator = arena };
    defer client.deinit();
    var provider: dz.lm.OpenAI = .{
        .client = &client,
        .base_url = url,
        .model = "oracle-model",
        .api = .responses,
        .api_key = "loopback-only",
        .max_retries = 0,
    };
    // Exercise recorded requests too, so a result name lost in serde fails on the wire.
    const request: dz.lm.Request = .{ .messages = messages };
    const restored = try dz.lm.requestFromValue(arena, try dz.lm.requestValue(arena, request));
    for ([_]@FieldType(dz.lm.OpenAI.ResponsesCompat, "tool_result_name"){ .include, .omit }) |policy| {
        provider.responses_compat = .{ .tool_result_name = policy };
        const text = try provider.lm().complete(arena, restored);
        if (!std.mem.eql(u8, text, "replayed")) return error.UnexpectedReplayResponse;
    }
}

fn nativeHistoryReplay(arena: std.mem.Allocator, messages: []const dz.adapter.Message) !struct {
    call: dz.Value,
    result: dz.Value,
    result_id: []const u8,
} {
    if (messages.len < 2) return error.MissingNativeReplay;
    for (messages[0 .. messages.len - 1], messages[1..]) |assistant, result| {
        if (assistant.role != .assistant or assistant.tool_calls.len == 0 or result.role != .tool) continue;
        const calls = try dz.tool.callsFromNative(arena, assistant.tool_calls);
        return .{
            .call = calls.object.get("tool_calls").?.array.items[0],
            .result = try dz.json_repair.parseJson(arena, result.content.text),
            .result_id = result.tool_call_id orelse return error.MissingToolResultId,
        };
    }
    return error.MissingNativeReplay;
}

fn demoRecords(arena: std.mem.Allocator, answer: []const u8) ![]const dz.value.Record {
    const demos = try arena.alloc(dz.value.Record, 1);
    demos[0] = (try dz.Example.fromJson(arena, try std.fmt.allocPrint(arena, "{{\"question\":\"demo-question\",\"answer\":\"{s}\"}}", .{answer}), &.{"question"})).record;
    return demos;
}

/// Fixed scratch for internal scopes, reserved statically.
var workspace_buffer: [16 * 1024 * 1024]u8 = undefined;
var workspace: dz.Workspace = .init(&workspace_buffer);

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) return error.InvalidArguments;
    if (std.mem.startsWith(u8, args[2], "react-history-")) return runReactHistory(init, args[1], args[2]);
    if (std.mem.startsWith(u8, args[2], "merge-")) return runMerge(init, args[1], args[2]);
    if (std.mem.eql(u8, args[2], "field-annotation-changed")) return run(IntP, init, args);
    if (std.mem.startsWith(u8, args[2], "wide-integer-")) return run(WideIntP, init, args);
    if (std.mem.startsWith(u8, args[2], "typed-code-")) return run(CodeP, init, args);
    if (std.mem.startsWith(u8, args[2], "specialized-code-")) return run(SpecializedCodeP, init, args);
    if (std.mem.startsWith(u8, args[2], "bounding-box-")) return run(BoxP, init, args);
    if (std.mem.startsWith(u8, args[2], "citations-")) return run(CitationP, init, args);
    if (std.mem.startsWith(u8, args[2], "typed-noul-")) return run(NoulP, init, args);
    if (std.mem.startsWith(u8, args[2], "configured-noul-")) return run(ConfiguredNoulP, init, args);
    if (std.mem.startsWith(u8, args[2], "typed-score-")) return run(ScoreP, init, args);
    if (std.mem.startsWith(u8, args[2], "typed-choice-")) return run(ChoiceP, init, args);
    if (std.mem.startsWith(u8, args[2], "field-default-")) {
        if (std.mem.eql(u8, args[2], "field-default-changed")) return run(DefaultPredict("high"), init, args);
        return run(DefaultPredict("low"), init, args);
    }
    if (std.mem.startsWith(u8, args[2], "field-decision-") or
        std.mem.startsWith(u8, args[2], "field-native-decision-")) return run(DecisionP, init, args);
    if (std.mem.startsWith(u8, args[2], "field-declared-decision-")) {
        if (std.mem.eql(u8, args[2], "field-declared-decision-changed"))
            return run(AnnotatedDecisionPredict(revised_criterion), init, args);
        return run(AnnotatedDecisionPredict(original_criterion), init, args);
    }
    if (std.mem.startsWith(u8, args[2], "native-tools-")) return run(ToolP, init, args);
    if (std.mem.startsWith(u8, args[2], "native-reasoning-")) return run(CoT, init, args);
    return run(P, init, args);
}

fn run(comptime Program: type, init: std.process.Init, args: []const [:0]const u8) !void {
    const arena = init.arena.allocator();
    const tool_mode = Program == ToolP;
    const reasoning_mode = Program == CoT;
    const reasoning_summary_mode = std.mem.startsWith(u8, args[2], "native-reasoning-summary-");
    const changed_reasoning = std.mem.eql(u8, args[2], "native-reasoning-changed");
    const two_step_mode = std.mem.startsWith(u8, args[2], "two-step-");
    const changed_extractor = std.mem.eql(u8, args[2], "two-step-changed");
    const extraction_options_mode = std.mem.startsWith(u8, args[2], "two-step-options-");
    const changed_extraction_options = std.mem.eql(u8, args[2], "two-step-options-changed");
    const changed_tools = std.mem.eql(u8, args[2], "native-tools-changed");
    const tool_capability_mode = std.mem.startsWith(u8, args[2], "native-tools-capability-");
    const changed_tool_capability = std.mem.eql(u8, args[2], "native-tools-capability-changed");
    const prefix = "sparse-";
    const sparse = std.mem.startsWith(u8, args[2], prefix);
    const override_mode = std.mem.startsWith(u8, args[2], "override-");
    const demo_mode = std.mem.startsWith(u8, args[2], "demo-") or override_mode;
    const demo_phase = args[2][if (override_mode) "override-".len else if (demo_mode) "demo-".len else 0..];
    const changed_demo = demo_mode and std.mem.eql(u8, demo_phase, "changed");
    const removed_override = override_mode and std.mem.eql(u8, demo_phase, "removed");
    const clear_override = override_mode and std.mem.eql(u8, demo_phase, "empty-start");
    const stored_config = std.mem.startsWith(u8, args[2], "stored-config-");
    const config_mode = stored_config or std.mem.startsWith(u8, args[2], "config-");
    const lm_options_mode = std.mem.startsWith(u8, args[2], "lm-options-");
    const temperature_mode = config_mode or lm_options_mode;
    const changed_temperature = temperature_mode and std.mem.endsWith(u8, args[2], "-changed");
    const fallback_mode = std.mem.startsWith(u8, args[2], "fallback-");
    const response_format_mode = std.mem.startsWith(u8, args[2], "response-format-");
    const schema_enabled = std.mem.eql(u8, args[2], "response-format-schema");
    const format_disabled = std.mem.eql(u8, args[2], "response-format-unsupported");
    const adapter_mode = std.mem.startsWith(u8, args[2], "adapter-") or fallback_mode or tool_mode or reasoning_mode or response_format_mode;
    const changed_adapter = std.mem.eql(u8, args[2], "adapter-changed");
    const changed_fallback = std.mem.eql(u8, args[2], "fallback-changed");
    const task_model_mode = std.mem.startsWith(u8, args[2], "task-model-");
    const rollout_model_mode = std.mem.startsWith(u8, args[2], "rollout-model-");
    const field_description_mode = std.mem.startsWith(u8, args[2], "field-description-");
    const changed_field_description = std.mem.eql(u8, args[2], "field-description-changed");
    const field_annotation_mode = std.mem.startsWith(u8, args[2], "field-annotation-");
    const field_default_mode = std.mem.startsWith(u8, args[2], "field-default-");
    const decision_mode = Program == DecisionP or std.mem.startsWith(u8, args[2], "field-declared-decision-");
    const decision_capability_mode = std.mem.startsWith(u8, args[2], "field-native-decision-");
    const changed_decision_capability = std.mem.eql(u8, args[2], "field-native-decision-changed");
    const changed_criterion = decision_mode and !decision_capability_mode and std.mem.endsWith(u8, args[2], "-changed");
    const criterion = if (changed_criterion) revised_criterion else original_criterion;
    const wide_integer_mode = Program == WideIntP;
    const specialized_code_mode = Program == SpecializedCodeP;
    const changed_code_classes = std.mem.eql(u8, args[2], "specialized-code-changed");
    const changed_noul_classes = std.mem.eql(u8, args[2], "configured-noul-changed");
    const changed_score_classes = std.mem.eql(u8, args[2], "typed-score-changed");
    const changed_choice_classes = std.mem.eql(u8, args[2], "typed-choice-changed");
    const invalid_marker = std.mem.eql(u8, args[2], "specialized-code-invalid-marker") or std.mem.eql(u8, args[2], "bounding-box-invalid") or
        std.mem.eql(u8, args[2], "citations-invalid") or std.mem.eql(u8, args[2], "typed-noul-invalid") or std.mem.eql(u8, args[2], "configured-noul-invalid") or
        std.mem.eql(u8, args[2], "typed-score-invalid") or std.mem.eql(u8, args[2], "typed-choice-invalid");
    const code_mode = Program == CodeP or specialized_code_mode;
    const box_mode = Program == BoxP;
    const citation_mode = Program == CitationP;
    const configured_noul = Program == ConfiguredNoulP;
    const noul_mode = Program == NoulP or configured_noul;
    const score_mode = Program == ScoreP;
    const choice_mode = Program == ChoiceP;
    const typed_decision_mode = noul_mode or score_mode or choice_mode;
    const field_value_mode = field_annotation_mode or field_default_mode or decision_mode or wide_integer_mode or code_mode or box_mode or citation_mode or typed_decision_mode;
    const field_identity_mode = field_description_mode or field_value_mode;
    const failure_score_mode = std.mem.startsWith(u8, args[2], "failure-score-");
    const changed_failure_score = std.mem.eql(u8, args[2], "failure-score-changed");
    const changed_task_model = std.mem.eql(u8, args[2], "task-model-changed") or std.mem.eql(u8, args[2], "rollout-model-changed");
    const mode = if (temperature_mode or adapter_mode or task_model_mode or rollout_model_mode or two_step_mode or failure_score_mode or field_identity_mode)
        (if (std.mem.endsWith(u8, args[2], "-start")) "start" else "resume")
    else if (demo_mode)
        (if (std.mem.eql(u8, demo_phase, "start") or clear_override) "start" else "resume")
    else
        args[2][if (sparse) prefix.len else 0..];
    const phase = std.meta.stringToEnum(enum { start, @"resume", fresh, @"dense-start", enable, inspect, @"policy-start", @"policy-resume", @"policy-partial", @"policy-full", @"policy-reordered", @"train-reordered", @"train-relabeled" }, mode) orelse return error.InvalidArguments;
    const legacy = (temperature_mode or adapter_mode or task_model_mode or rollout_model_mode or two_step_mode or failure_score_mode or field_identity_mode) and std.mem.endsWith(u8, args[2], "-legacy");
    const rejects_resume = changed_code_classes or changed_noul_classes or changed_score_classes or changed_choice_classes or invalid_marker or phase == .@"policy-reordered" or phase == .@"train-reordered" or phase == .@"train-relabeled" or changed_demo or removed_override or changed_temperature or changed_adapter or changed_fallback or changed_task_model or changed_tools or changed_tool_capability or changed_reasoning or changed_extractor or changed_extraction_options or changed_failure_score or changed_field_description or Program == IntP or Program == DefaultPredict("high") or changed_criterion or changed_decision_capability or schema_enabled or format_disabled or legacy;
    const policy_switch = std.mem.startsWith(u8, args[2], "policy-");
    const partial_policy = sparse or phase == .@"policy-resume" or phase == .@"policy-partial";
    const resume_run = phase != .start and phase != .@"dense-start" and phase != .@"policy-start" and phase != .@"policy-partial";
    const dir = args[1];
    const low = "<answer>low</answer>";
    const chat_high = "[[ ## answer ## ]]\nhigh\n\n[[ ## completed ## ]]";
    const high = "<answer>high</answer>";
    const json_low = "{\"answer\":\"low\"}";
    const json_high = "{\"answer\":\"high\"}";
    const xml_wide_low = "<answer>" ++ wide_low ++ "</answer>";
    const xml_wide_high = "<answer>" ++ wide_high ++ "</answer>";
    const omitted_answer = "<status>ok</status>";
    const explicit_low = low ++ omitted_answer;
    const explicit_high = high ++ omitted_answer;
    const decision_no = "<answer>{\"noul\":0.1}</answer>";
    const decision_yes = "<answer>{\"noul\":0.9}</answer>";
    const text_tool_high = "<answer>high</answer><calls>{\"tool_calls\":[]}</calls>";
    var history: dz.lm.History = .{ .arena = arena };
    var workload: Workload = .{
        .resume_run = resume_run,
        .sparse = sparse or policy_switch,
        .model = .{
            .model = if (changed_task_model) "task-replacement" else "task-original",
            .history = if (policy_switch or override_mode or temperature_mode or adapter_mode or task_model_mode or rollout_model_mode or two_step_mode or failure_score_mode or field_identity_mode) &history else null,
            .supports_function_calling = tool_mode and !changed_tool_capability,
            .supports_reasoning = reasoning_mode and !changed_reasoning,
            .supports_decision_requests = changed_decision_capability or typed_decision_mode,
            .supports_response_format = response_format_mode and !format_disabled,
            .supports_response_schema = schema_enabled,
            .reasonings = if (!reasoning_summary_mode)
                &.{}
            else if (resume_run)
                &.{ "fresh summary", "fresh summary" }
            else
                &.{ "archived summary µ", "archived summary µ", "archived summary µ", "archived summary µ" },
            .citations = if (!citation_mode) &.{} else if (resume_run)
                &.{ low_citations, low_citations, low_citations, high_citations, high_citations, low_citations }
            else
                &.{ low_citations, low_citations, low_citations, low_citations },
            .responses = switch (phase) {
                .@"policy-start" => &.{ low, low, low, low, low, "archived malformed answer" },
                .@"policy-resume" => &.{ high, high, low, high, "archived malformed answer", high, high },
                .@"policy-partial" => &.{ low, low },
                .@"policy-reordered" => &.{ high, high, high, high },
                .@"train-reordered", .@"train-relabeled" => &.{ low, low, low, high },
                .@"policy-full" => &.{ high, high, high, high, low, high, high, high, "archived malformed answer", high },
                .start, .@"dense-start" => if (typed_decision_mode)
                    &.{ "", "", "", "" }
                else if (decision_mode)
                    &.{ decision_no, decision_no, decision_no, decision_no }
                else if (response_format_mode)
                    &.{ json_low, json_low, json_low, json_low }
                else if (field_default_mode)
                    &.{ explicit_low, omitted_answer, omitted_answer, omitted_answer }
                else if (wide_integer_mode)
                    &.{ xml_wide_low, xml_wide_low, xml_wide_low, xml_wide_low }
                else if (specialized_code_mode)
                    &.{ xml_specialized_low, xml_specialized_low, xml_specialized_low, xml_specialized_low }
                else if (code_mode)
                    &.{ xml_code_low, xml_code_low, xml_code_low, xml_code_low }
                else if (box_mode)
                    &.{ xml_box_low, xml_box_low, xml_box_low, xml_box_low }
                else if (citation_mode)
                    &.{ citation_xml, citation_xml, citation_xml, citation_xml }
                else if (field_annotation_mode)
                    &.{ low, low, low, low }
                else if (failure_score_mode)
                    &.{ low, "malformed seed answer" }
                else if (two_step_mode)
                    &.{ "seed good", "seed broken", "seed broken", "child broken" }
                else
                    &.{ low, low, low, "archived malformed answer" },
                .@"resume" => if (changed_decision_capability or typed_decision_mode)
                    &.{ "", "", "", "", "", "" }
                else if (decision_mode)
                    &.{ decision_no, if (changed_criterion) decision_yes else decision_no, decision_no, decision_yes, decision_yes, if (changed_criterion) decision_yes else decision_no }
                else if (response_format_mode)
                    &.{ json_low, json_low, json_low, json_high, json_high, if (schema_enabled or format_disabled) json_high else json_low }
                else if (field_default_mode)
                    &.{ explicit_low, omitted_answer, explicit_low, explicit_high, explicit_high, omitted_answer }
                else if (wide_integer_mode)
                    &.{ xml_wide_high, xml_wide_high, xml_wide_low, xml_wide_high, xml_wide_high, xml_wide_low }
                else if (specialized_code_mode)
                    &.{ xml_specialized_low, xml_specialized_low, xml_specialized_low, xml_specialized_high, xml_specialized_high, xml_specialized_low }
                else if (code_mode)
                    &.{ xml_code_low, xml_code_low, xml_code_low, xml_code_high, xml_code_high, xml_code_low }
                else if (box_mode)
                    &.{ xml_box_low, xml_box_low, xml_box_low, xml_box_high, xml_box_high, xml_box_low }
                else if (citation_mode)
                    &.{ citation_xml, citation_xml, citation_xml, citation_xml, citation_xml, citation_xml }
                else if (field_annotation_mode)
                    (if (Program == IntP)
                        &.{ "<answer>0</answer>", "<answer>0</answer>", "<answer>0</answer>", "<answer>1</answer>", "<answer>1</answer>", "<answer>0</answer>" }
                    else
                        &.{ low, low, low, high, high, low })
                else if (failure_score_mode)
                    &.{ low, "malformed seed answer", low, high, high, "archived malformed answer" }
                else if (two_step_mode)
                    &.{ "seed good", "seed broken" }
                else if (changed_adapter)
                    &.{ chat_high, chat_high, chat_high, chat_high }
                else if (fallback_mode)
                    (if (changed_fallback) &.{ "malformed seed answer", "{\"answer\":\"high\"}", low } else &.{ "malformed seed answer", low })
                else if (tool_mode)
                    (if (changed_tools or changed_tool_capability) &.{ text_tool_high, text_tool_high } else &.{ low, low })
                else if (reasoning_mode)
                    (if (changed_reasoning) &.{ "<plan>text plan</plan><answer>high</answer>", "<plan>text plan</plan><answer>high</answer>" } else &.{ low, low })
                else if (rollout_model_mode)
                    (if (changed_task_model) &.{ high, high } else &.{ low, low })
                else if (sparse) &.{ low, low, low, high, high } else if (changed_demo or changed_temperature or changed_task_model or changed_field_description) &.{ high, high, high, high } else &.{ low, low, low, high },
                .inspect => &.{ low, low },
                .fresh => if (sparse) &.{ low, low, low, high, "fresh malformed answer", high, high } else &.{ low, low, low, high, high, "fresh malformed answer" },
                // Fresh seed scores differ, but the archive still wins on restart.
                .enable => if (sparse) &.{ high, high, low, high, "fresh malformed answer", high } else &.{ high, high, low, high, "fresh malformed answer" },
            },
        },
    };
    if (changed_decision_capability or typed_decision_mode) {
        const low_evidence = (try dz.Example.fromJson(arena, if (score_mode)
            "{\"typesafe\":{\"answers\":{\"answer\":{\"probabilities\":{\"0\":0.75,\"1\":0.25},\"confidence\":0.875}}}}"
        else if (choice_mode)
            "{\"typesafe\":{\"answers\":{\"answer\":{\"probabilities\":{\"7\":0.75,\"True\":0.25},\"confidence\":0.875}}}}"
        else
            "{\"typesafe\":{\"answers\":{\"answer\":{\"noul\":0.1}}}}", &.{})).record;
        const high_evidence = (try dz.Example.fromJson(arena, if (score_mode)
            "{\"typesafe\":{\"answers\":{\"answer\":{\"probabilities\":{\"0\":0.25,\"1\":0.75},\"confidence\":0.875}}}}"
        else if (choice_mode)
            "{\"typesafe\":{\"answers\":{\"answer\":{\"probabilities\":{\"7\":0.25,\"True\":0.75},\"confidence\":0.875}}}}"
        else
            "{\"typesafe\":{\"answers\":{\"answer\":{\"noul\":0.9}}}}", &.{})).record;
        const data = try arena.alloc(?dz.Value, workload.model.responses.len);
        for (data, 0..) |*entry, i| {
            const positive = if (typed_decision_mode) resume_run and (i == 3 or i == 4) else i != 0 and i != 2;
            entry.* = .{ .object = if (positive) high_evidence else low_evidence };
        }
        workload.model.provider_data = data;
    }
    const questions: []const []const u8 = if (policy_switch)
        &.{ "good-case", "middle-case", "broken-case", "untouched-case" }
    else if (sparse)
        &.{ "good-case", "nan-case", "broken-case", "untouched-case" }
    else
        &.{ "good-case", "broken-case" };
    const train = try rows(arena, questions, true);
    const val = try rows(arena, questions, false);
    if (phase == .@"policy-reordered") std.mem.swap(dz.Example, &val[0], &val[2]);
    if (phase == .@"train-reordered") std.mem.swap(dz.Example, &train[0], &train[1]);
    if (phase == .@"train-relabeled") train[1].record.getPtr("training").?.* = .{ .bool = false };
    if (tool_mode) {
        const catalog = (try dz.Example.fromJson(arena, "{\"tools\":[{\"name\":\"lookup\",\"desc\":\"Look up a value.\",\"args\":{}}]}", &.{"tools"})).record.get("tools").?;
        for ([_][]dz.Example{ train, val }) |dataset| for (dataset) |*row| {
            try row.record.put(arena, "tools", catalog);
            row.input_keys = &.{ "question", "tools" };
        };
    }
    var student: Program = .{};
    if (tool_mode) student.state.fields = &tool_fields;
    var ctx: dz.Context = .{ .workspace = &workspace, .arena = arena, .lm = workload.model.lm(), .adapter = if (response_format_mode) .json else if (changed_adapter) .chat else .xml, .json_fallback = changed_fallback };
    if (tool_mode) ctx.use_native_function_calling = if (tool_capability_mode) true else if (changed_tools) false else if (resume_run) null else true;
    const chat_low = "[[ ## answer ## ]]\nlow\n\n[[ ## completed ## ]]";
    var extraction_lm: dz.lm.ScriptedLM = .{
        .model = if (changed_extractor) "extract-replacement" else "extract-original",
        .history = &history,
        .responses = if (changed_extractor or changed_extraction_options) &.{ chat_high, chat_high } else if (resume_run) &.{ chat_low, chat_low } else &.{ chat_low, chat_low, chat_low, chat_high },
    };
    const two_step: dz.TwoStepAdapter = .{
        .extraction_model = extraction_lm.lm(),
        .extraction_options = .{ .temperature = if (!extraction_options_mode) null else if (changed_extraction_options) 0.75 else 0.25 },
    };
    if (two_step_mode) ctx = two_step.context(ctx);
    var unused_lm: dz.lm.ScriptedLM = .{
        .model = if (std.mem.eql(u8, args[2], "rollout-model-resume")) "unused-replacement" else "unused",
        .responses = &.{},
        .history = &history,
        .supports_function_calling = tool_capability_mode and resume_run,
        .supports_reasoning = reasoning_mode and resume_run,
    };
    const state = if (reasoning_mode) &student.predict.state else &student.state;
    if (Program == DecisionP) {
        var criteria: dz.value.Record = .empty;
        try criteria.put(arena, "true", .{ .string = try arena.dupe(u8, criterion) });
        try state.setCriteria(arena, "answer", .{ .object = criteria });
    }
    if (field_description_mode) {
        const descs = try arena.dupe([]const u8, state.descs);
        descs[1] = try arena.dupe(u8, if (changed_field_description) "Revised answer guidance." else "Original answer guidance.");
        state.descs = descs;
    }
    if (task_model_mode or tool_capability_mode or reasoning_mode or decision_capability_mode or response_format_mode) {
        state.lm = ctx.lm;
        ctx.lm = unused_lm.lm();
    }
    if (demo_mode) {
        state.demos = try demoRecords(arena, if (override_mode) "stored-demo" else if (changed_demo) "changed-demo" else "original-demo");
        if (override_mode and !removed_override)
            ctx.demos_override = if (clear_override) &.{} else try demoRecords(arena, if (changed_demo) "changed-demo" else "original-demo");
    }
    const config_temperature: f64 = if (changed_temperature) 0.75 else 0.25;
    const config: dz.predict.PredictConfig = .{ .temperature = config_temperature };
    if (stored_config) {
        state.config = &config;
    } else if (config_mode) {
        ctx.predict_config = &config;
    } else if (lm_options_mode) {
        ctx.lm_options.temperature = config_temperature;
    }
    // One pool thread keeps the scripted LM and request history single-writer.
    const Pool = dz.Parallel(GEPA.Run, .{ .workers = 1, .tasks = 1, .attempts = 1, .failures = 1 });
    var pool: Pool = undefined;
    const rollout_context = ctx;
    if (rollout_model_mode) {
        try pool.init(init.io, .{});
        ctx.lm = unused_lm.lm();
    }
    defer if (rollout_model_mode) pool.deinit();
    const result = (GEPA{
        .metric = Workload.metric,
        .rollouts = if (rollout_model_mode) .{ .runner = pool.runner(), .contexts = &.{rollout_context} } else null,
        .max_metric_calls = @as(usize, switch (phase) {
            .start, .@"dense-start" => if (failure_score_mode) 2 else 4,
            .@"resume" => if (fallback_mode or tool_mode or reasoning_mode or two_step_mode or rollout_model_mode) 4 else 6,
            .inspect, .@"train-reordered", .@"train-relabeled" => 6,
            .fresh => 8,
            .enable => 7,
            .@"policy-start" => 6,
            .@"policy-resume" => 11,
            .@"policy-partial", .@"policy-reordered" => 2,
            .@"policy-full" => 8,
        }) + @intFromBool(sparse and resume_run),
        .failure_score = if (changed_failure_score) 0.75 else 0.25,
        .track_best_outputs = true,
        .use_merge = false,
        .cache_evaluation = !policy_switch and !failure_score_mode and !field_value_mode and !response_format_mode and phase != .fresh and phase != .@"dense-start",
        .batch_sampler = .{ .ptr = &workload, .max_batch_size = 1, .next = Workload.sample },
        .val_evaluation_policy = if (partial_policy) .{
            .ptr = &workload,
            .max_batch_size = 3,
            .seed_batch = Workload.seedBatch,
            .next_batch = Workload.valBatch,
        } else null,
        .instruction_proposer = .{ .ptr = &workload, .propose = if (reasoning_mode) Workload.proposeCoT else Workload.propose },
    }).compile(Program, arena, ctx, student, train, .{
        .valset = val,
        .log_dir = .{ .io = init.io, .path = dir },
        .checkpoint = .{
            .io = init.io,
            .path = try std.fs.path.join(arena, &.{ dir, "native-checkpoint.json" }),
            .revision = "failed cache row",
            .code_classes = if (changed_code_classes) swapped_code_classes else if (specialized_code_mode) code_classes else &.{},
            .choice_classes = if (changed_choice_classes)
                &.{.bind(Category, "renamed-category")}
            else if (choice_mode)
                &.{.bind(Category, "category")}
            else
                &.{},
            .score_classes = if (changed_score_classes)
                &.{.bind(Rubric, "renamed-rubric")}
            else if (score_mode)
                &.{.bind(Rubric, "rubric")}
            else
                &.{},
            .noul_classes = if (changed_noul_classes)
                &.{.bind(RevisedNoul, "pass")}
            else if (configured_noul)
                &.{.bind(ConfiguredNoul, "pass")}
            else
                &.{},
        },
    }) catch |err| {
        const expected_error = if (invalid_marker) error.InvalidGepaCheckpoint else error.IncompatibleGepaCheckpoint;
        if (!rejects_resume or err != expected_error) return err;
        if (workload.model.index != 0 or extraction_lm.index != 0 or history.requests.items.len != 0) return error.UnexpectedModelCallCount;
        return;
    };
    if (rejects_resume) {
        std.debug.print("incompatible restart made {d} main and {d} extraction calls\n", .{ workload.model.index, extraction_lm.index });
        return error.UnexpectedIncompatibleResume;
    }
    if (workload.model.index != workload.model.responses.len) return error.UnexpectedModelCallCount;
    if ((task_model_mode or rollout_model_mode) and history.requests.items.len != workload.model.index) return error.UnexpectedModelCallCount;
    if (override_mode) for (history.requests.items) |request| {
        var saw_override = false;
        for (request.messages[0 .. request.messages.len - 1]) |message| {
            if (std.mem.indexOf(u8, message.content.text, "<answer>\nstored-demo\n</answer>") != null)
                return error.UnexpectedStoredDemo;
            saw_override = saw_override or std.mem.indexOf(u8, message.content.text, "<answer>\noriginal-demo\n</answer>") != null;
        }
        if (saw_override == clear_override) return error.UnexpectedDemoOverride;
    };
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    var json: std.json.Stringify = .{ .writer = &stdout.interface };
    try json.beginObject();
    if (two_step_mode) {
        try json.objectField("stages");
        try json.beginArray();
        for (history.requests.items) |request| {
            const system = request.messages[0].content.text;
            const user = request.messages[request.messages.len - 1].content.text;
            if (std.mem.indexOf(u8, system, "Your job is to extract") != null) {
                const payload = for ([_][]const u8{ "seed good", "seed broken", "child broken" }) |text| {
                    if (std.mem.indexOf(u8, user, text) != null) break text;
                } else return error.UnexpectedExtractionInput;
                try json.write(.{ "extract", payload });
            } else {
                if (!std.mem.startsWith(u8, user, "question: ")) return error.UnexpectedQuestion;
                const stage = if (std.mem.indexOf(u8, system, "Winner.") != null) "child" else "seed";
                try json.write(.{ stage, user["question: ".len..] });
            }
        }
        try json.endArray();
        if (extraction_options_mode) {
            try json.objectField("extraction_temperatures");
            try json.beginArray();
            for (history.requests.items) |request| {
                if (std.mem.indexOf(u8, request.messages[0].content.text, "Your job is to extract") != null)
                    try json.write(request.options.temperature);
            }
            try json.endArray();
        }
    }
    if (temperature_mode) {
        try json.objectField("temperatures");
        try json.beginArray();
        for (history.requests.items) |request| try json.write(request.options.temperature);
        try json.endArray();
    }
    if (adapter_mode) {
        try json.objectField("adapters");
        try json.beginArray();
        for (history.requests.items) |request| {
            const user = request.messages[request.messages.len - 1].content.text;
            if (std.mem.indexOf(u8, request.messages[0].content.text, "Outputs will be a JSON object") != null) {
                try json.write("json");
            } else if (std.mem.indexOf(u8, user, "[[ ## question ## ]]") != null) {
                try json.write("chat");
            } else if (std.mem.indexOf(u8, user, "<question>") != null) {
                try json.write("xml");
            } else return error.UnexpectedQuestion;
        }
        try json.endArray();
    }
    if (reasoning_mode) {
        try json.objectField("plan_fields");
        try json.beginArray();
        for (history.requests.items) |request|
            try json.write(std.mem.indexOf(u8, request.messages[0].content.text, "<plan>") != null);
        try json.endArray();
    }
    if (tool_mode) {
        try json.objectField("declared_tools");
        try json.beginArray();
        for (history.requests.items) |request| {
            try json.beginArray();
            for (request.tools) |tool| try json.write(tool.name);
            try json.endArray();
        }
        try json.endArray();
    }
    if (rollout_model_mode) {
        try json.objectField("models");
        try json.beginArray();
        for (0..workload.model.index) |_| try json.write(workload.model.model);
        try json.endArray();
    }
    if (field_description_mode) {
        try json.objectField("field_descriptions");
        try json.beginArray();
        for (history.requests.items) |request| {
            const system = request.messages[0].content.text;
            const description = for ([_][]const u8{ "Original answer guidance.", "Revised answer guidance." }) |text| {
                if (std.mem.indexOf(u8, system, text) != null) break text;
            } else return error.MissingFieldDescription;
            try json.write(description);
        }
        try json.endArray();
    }
    if (decision_mode) {
        try json.objectField("decision_criteria");
        try json.beginArray();
        for (history.requests.items) |request| {
            const system = request.messages[0].content.text;
            if (std.mem.indexOf(u8, system, criterion) == null) return error.MissingDecisionCriterion;
            try json.write(criterion);
        }
        try json.endArray();
    }
    if (decision_capability_mode) {
        try json.objectField("decision_routes");
        try json.beginArray();
        for (history.requests.items) |request|
            try json.write(if (request.decision == null) "generative" else "native");
        try json.endArray();
    }
    if (response_format_mode) {
        try json.objectField("response_formats");
        try json.beginArray();
        for (history.requests.items) |request| {
            const format: ?[]const u8 = if (request.options.response_format) |fmt| @tagName(fmt) else null;
            try json.write(format);
        }
        try json.endArray();

        try json.objectField("calls");
        try json.beginArray();
        for (history.requests.items) |request| {
            const system = request.messages[0].content.text;
            const user = request.messages[request.messages.len - 1].content.text;
            const broken = std.mem.indexOf(u8, user, "broken-case") != null;
            if (!broken and std.mem.indexOf(u8, user, "good-case") == null) return error.UnexpectedQuestion;
            try json.write(.{ std.mem.indexOf(u8, system, "Winner.") != null, broken });
        }
        try json.endArray();
    }
    if (policy_switch or reasoning_summary_mode or rollout_model_mode or failure_score_mode or field_identity_mode) {
        try json.objectField("calls");
        try json.beginArray();
        for (history.requests.items) |request| {
            if (typed_decision_mode) {
                const decision = request.decision orelse return error.MissingDecisionRequest;
                const state_value = decision.state.object;
                const instructions = state_value.get("instructions").?.string;
                const question = state_value.get("inputs").?.object.get("question").?.string;
                try json.write(.{ std.mem.indexOf(u8, instructions, "Winner.") != null, std.mem.eql(u8, question, "broken-case") });
                continue;
            }
            const system = request.messages[0].content.text;
            const user = request.messages[request.messages.len - 1].content.text;
            const tag = "<question>\n";
            const first = (std.mem.indexOf(u8, user, tag) orelse return error.UnexpectedQuestion) + tag.len;
            const end = std.mem.indexOfPos(u8, user, first, "\n</question>") orelse return error.UnexpectedQuestion;
            const question = user[first..end];
            const child = std.mem.indexOf(u8, system, "Winner.") != null;
            if (reasoning_summary_mode or rollout_model_mode or failure_score_mode or field_identity_mode)
                try json.write(.{ child, std.mem.eql(u8, question, "broken-case") })
            else
                try json.write(.{ child, question });
        }
        try json.endArray();
    }
    try json.objectField("budget");
    try json.write(result.total_metric_calls);
    try json.objectField("best");
    try json.write(result.best_idx);
    try json.objectField("scores");
    try json.beginArray();
    for (result.val_subscores, 0..) |scores, i| {
        const ids: []const usize = if (result.val_ids) |rows_by_candidate|
            rows_by_candidate[i]
        else if (policy_switch)
            &.{ 0, 1, 2, 3 }
        else
            &.{ 0, 1 };
        if (sparse or policy_switch) {
            if ((partial_policy or phase == .@"policy-full") and result.val_ids == null) return error.MissingValidationIds;
            try json.beginArray();
            for (ids, scores) |id, score| {
                try json.beginArray();
                try json.write(id);
                if (std.math.isNan(score)) try json.write("nan") else try json.write(score);
                try json.endArray();
            }
            try json.endArray();
        } else {
            if (scores.len != 2) return error.UnexpectedValidationCount;
            var ordered: [2]f64 = undefined;
            for (ids, scores) |id, score| ordered[id] = score;
            try json.write(.{ .@"0" = ordered[0], .@"1" = ordered[1] });
        }
    }
    try json.endArray();
    try json.objectField("leaders");
    try json.write(result.per_val_instance_best_candidates);
    if (code_mode) {
        try json.objectField("code_types");
        try json.beginArray();
        for (result.best_outputs_valset.?) |row| {
            try json.beginArray();
            for (row) |entry| {
                const pred = switch (entry.output) {
                    .prediction => |pred| pred,
                    .failed_parse => return error.UnexpectedCodeParseFailure,
                };
                const code = try checkCodeRecord(specialized_code_mode, pred.fields());
                const completions = pred.completions() orelse return error.MissingCompletions;
                for (completions) |*completion| {
                    if (!std.mem.eql(u8, code, try checkCodeRecord(specialized_code_mode, completion)))
                        return error.UnexpectedCodeCompletion;
                }
                try json.write(true);
            }
            try json.endArray();
        }
        try json.endArray();
    }
    if (box_mode) {
        try json.objectField("box_types");
        try json.beginArray();
        for (result.best_outputs_valset.?) |row| {
            try json.beginArray();
            for (row) |entry| {
                const pred = switch (entry.output) {
                    .prediction => |pred| pred,
                    .failed_parse => return error.UnexpectedBoxParseFailure,
                };
                const box = try checkBoxRecord(pred.fields());
                for (pred.completions() orelse return error.MissingCompletions) |*completion| {
                    if (!std.meta.eql(box, try checkBoxRecord(completion))) return error.UnexpectedGeometry;
                }
                try json.write(true);
            }
            try json.endArray();
        }
        try json.endArray();
    }
    if (citation_mode) {
        try json.objectField("citation_types");
        try json.beginArray();
        for (result.best_outputs_valset.?) |row| {
            try json.beginArray();
            for (row) |entry| {
                const pred = switch (entry.output) {
                    .prediction => |pred| pred,
                    .failed_parse => return error.UnexpectedCitationParseFailure,
                };
                try checkCitationRecord(pred.fields());
                for (pred.completions() orelse return error.MissingCompletions) |*completion| {
                    try checkCitationRecord(completion);
                    if (!(try pred.get("answer").?.custom.get(dz.Citations).?.eql(completion.get("answer").?.custom.get(dz.Citations).?.*)))
                        return error.UnexpectedCitationValue;
                }
                try json.write(true);
            }
            try json.endArray();
        }
        try json.endArray();
    }
    if (choice_mode) {
        try json.objectField("choice_types");
        try json.beginArray();
        for (result.best_outputs_valset.?) |row| {
            try json.beginArray();
            for (row) |entry| {
                const pred = switch (entry.output) {
                    .prediction => |pred| pred,
                    .failed_parse => return error.UnexpectedChoiceParseFailure,
                };
                const answer = try checkChoice(pred.get("answer"));
                for (pred.completions() orelse return error.MissingCompletions) |completion| {
                    const other = try checkChoice(completion.get("answer"));
                    if (std.meta.activeTag(answer.value) != std.meta.activeTag(other.value) or
                        !try dz.value.pyEqual(answer.value, other.value) or answer.confidence != other.confidence or
                        !try dz.value.pyEqual(answer.probabilities.?, other.probabilities.?))
                        return error.UnexpectedChoiceEvidence;
                }
                try json.write(true);
            }
            try json.endArray();
        }
        try json.endArray();
    }
    if (score_mode) {
        try json.objectField("score_types");
        try json.beginArray();
        for (result.best_outputs_valset.?) |row| {
            try json.beginArray();
            for (row) |entry| {
                const pred = switch (entry.output) {
                    .prediction => |pred| pred,
                    .failed_parse => return error.UnexpectedScoreParseFailure,
                };
                const answer = try checkScore(pred.get("answer"));
                for (pred.completions() orelse return error.MissingCompletions) |completion| {
                    const other = try checkScore(completion.get("answer"));
                    if (answer.value != other.value or answer.confidence != other.confidence or
                        answer.level.?.value.integer != other.level.?.value.integer or
                        !try dz.value.pyEqual(answer.probabilities.?, other.probabilities.?))
                        return error.UnexpectedScoreEvidence;
                }
                try json.write(true);
            }
            try json.endArray();
        }
        try json.endArray();
    }
    if (noul_mode) {
        try json.objectField("noul_types");
        try json.beginArray();
        for (result.best_outputs_valset.?) |row| {
            try json.beginArray();
            for (row) |entry| {
                const pred = switch (entry.output) {
                    .prediction => |pred| pred,
                    .failed_parse => return error.UnexpectedNoulParseFailure,
                };
                const T = if (configured_noul) ConfiguredNoul else dz.Noul;
                const answer = try checkNoul(T, pred.get("answer"));
                for (pred.completions() orelse return error.MissingCompletions) |completion| {
                    const other = try checkNoul(T, completion.get("answer"));
                    if (answer.value != other.value or answer.confidence != other.confidence or answer.probability != other.probability)
                        return error.UnexpectedNoulEvidence;
                }
                try json.write(true);
            }
            try json.endArray();
        }
        try json.endArray();
    }
    try json.objectField("outputs");
    try json.beginArray();
    for (result.best_outputs_valset.?) |row| {
        try json.beginArray();
        for (row) |entry| {
            try json.beginArray();
            try json.write(entry.candidate_idx);
            switch (entry.output) {
                .prediction => |pred| if (reasoning_summary_mode) {
                    const value = pred.get("reasoning") orelse return error.MissingReasoning;
                    if (value != .custom) return error.MissingReasoning;
                    const typed = value.custom.get(dz.Reasoning) orelse return error.MissingReasoning;
                    if (pred.completions()) |completions| {
                        for (completions) |completion| {
                            const item = completion.get("reasoning") orelse return error.MissingReasoning;
                            if (item != .custom) return error.MissingReasoning;
                            const boxed = item.custom.get(dz.Reasoning) orelse return error.MissingReasoning;
                            if (!std.mem.eql(u8, typed.content, boxed.content)) return error.UnexpectedReasoning;
                        }
                    }
                    try json.write(.{ .reasoning = typed.content, .answer = pred.get("answer").?.string });
                } else {
                    if (wide_integer_mode) {
                        const answer = pred.get("answer") orelse return error.MissingInteger;
                        if (answer != .number_string) return error.UnexpectedIntegerRepresentation;
                        const completions = pred.completions() orelse return error.MissingCompletions;
                        for (completions) |completion| {
                            const item = completion.get("answer") orelse return error.MissingInteger;
                            if (item != .number_string or !std.mem.eql(u8, item.number_string, answer.number_string))
                                return error.UnexpectedIntegerRepresentation;
                        }
                    }
                    try json.write(dz.Value{ .object = pred.fields().* });
                },
                .failed_parse => |failure| try json.write(try std.fmt.allocPrint(arena, "FailedPrediction(completion_text='{s}', format_reward={d})", .{ failure.completion_text, failure.format_reward })),
            }
            try json.endArray();
        }
        try json.endArray();
    }
    try json.endArray();
    try json.endObject();
    try stdout.interface.flush();
}

const MergeWorkload = struct {
    const Left = dz.Predict(dz.Signature(struct {
        pub const instructions = "Left seed.";
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { tag: []const u8 };
    }));
    const Right = dz.Predict(dz.Signature(struct {
        pub const instructions = "Right seed.";
        pub const Input = struct { question: []const u8, tag: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const Program = struct {
        left: Left = .{},
        right: Right = .{},

        pub const Input = Left.Input;
        pub const Output = Right.Output;

        pub fn forward(self: *const @This(), ctx: dz.Context, input: Input) dz.predict.Error!Output {
            const tag = (try self.left.forward(ctx, input)).tag;
            return self.right.forward(ctx, .{ .question = input.question, .tag = tag });
        }
    };

    const Call = struct { usize, bool, bool };
    allocator: std.mem.Allocator,
    cached: bool,
    full: bool,
    reject: bool,
    retry: bool,
    stop_iteration: usize,
    calls: std.ArrayList(Call) = .empty,
    merged_zero_calls: usize = 0,
    saw_decoy: bool = false,
    sample_id: [1]usize = undefined,

    fn complete(ptr: *anyopaque, arena: std.mem.Allocator, request: dz.lm.Request) dz.lm.Error![]const dz.lm.Completion {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const system = request.messages[0].content.text;
        const user = request.messages[request.messages.len - 1].content.text;
        const text = blk: {
            if (std.mem.indexOf(u8, system, "Left seed.") != null) break :blk "<tag>base</tag>";
            if (std.mem.indexOf(u8, system, "Left upgraded.") != null) break :blk "<tag>upgraded</tag>";
            if (std.mem.indexOf(u8, system, "Left decoy.") != null) break :blk "<tag>decoy</tag>";
            const tag = "<question>\n";
            const first = (std.mem.indexOf(u8, user, tag) orelse return error.LMFailure) + tag.len;
            const end = std.mem.indexOfScalarPos(u8, user, first, '\n') orelse return error.LMFailure;
            const id = std.fmt.parseInt(usize, user[first..end], 10) catch return error.LMFailure;
            const left = std.mem.indexOf(u8, user, "<tag>\nupgraded\n</tag>") != null;
            const right = std.mem.indexOf(u8, system, "Right upgraded.") != null;
            try self.calls.append(self.allocator, .{ id, left, right });
            if (left and right) {
                if (self.reject) break :blk "<answer>rejected</answer>";
                if (self.cached and !self.full and id == 0 and !self.retry) {
                    self.merged_zero_calls += 1;
                    break :blk if (self.merged_zero_calls == 4) "<answer>last</answer>" else "<answer>low</answer>";
                }
                break :blk "<answer>merged</answer>";
            }
            if (std.mem.indexOf(u8, user, "<tag>\ndecoy\n</tag>") != null or std.mem.indexOf(u8, system, "Right decoy.") != null) {
                const validation = self.retry and self.full and self.saw_decoy;
                self.saw_decoy = true;
                break :blk if (validation) "<answer>rejected</answer>" else if (id == 3) "<answer>high</answer>" else "<answer>low</answer>";
            }
            break :blk if ((left and id == 1) or (right and id == 2)) "<answer>high</answer>" else "<answer>low</answer>";
        };
        return arena.dupe(dz.lm.Completion, &.{.{ .text = text }});
    }

    fn score(_: std.mem.Allocator, _: *const dz.Example, pred: *const dz.optimize.RecordPrediction, _: ?*const dz.Trace, _: ?[]const u8, _: ?*const dz.Trace.Step) dz.value.ParseError!dz.gepa.Feedback {
        const answer = pred.get("answer").?.string;
        return .{ .score = if (std.mem.eql(u8, answer, "merged")) 1 else if (std.mem.eql(u8, answer, "last")) 0.75 else if (std.mem.eql(u8, answer, "high")) 0.5 else if (std.mem.eql(u8, answer, "rejected")) 0 else 0.125 };
    }

    fn select(_: *anyopaque, _: *const GEPA.CandidateSelection) !usize {
        return 0;
    }

    fn sample(ptr: *anyopaque, info: *const GEPA.BatchSample) ![]const usize {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        // The reference samples by archive size, which differs from the
        // iteration number once the rejected merge adds no candidate.
        self.sample_id[0] = if (self.retry) 3 else info.iteration + 1;
        return &self.sample_id;
    }

    fn seedBatch(_: *anyopaque, _: []const dz.Example) ![]const usize {
        return &.{ 0, 8 };
    }

    fn valBatch(ptr: *anyopaque, state: *const GEPA.ValidationState) ![]const usize {
        const self: *const @This() = @ptrCast(@alignCast(ptr));
        return switch (state.candidates.len) {
            1 => &.{ 8, 0, 1 },
            2 => &.{ 8, 0, 2 },
            3 => if (self.retry) &.{ 8, 0 } else if (self.cached) &.{ 8, 1, 0, 2 } else &.{ 8, 1, 2 },
            4, 5 => if (self.stop_iteration != 4) &.{ 8, 0 } else error.UnexpectedCandidateCount,
            else => error.UnexpectedCandidateCount,
        };
    }

    fn propose(ptr: *anyopaque, _: dz.Context, input: *const GEPA.InstructionProposal) ![]const GEPA.Instruction {
        const self: *const @This() = @ptrCast(@alignCast(ptr));
        if (std.mem.eql(u8, input.components_to_update[0], "left"))
            return if (self.retry) &.{.{ .name = "left", .text = "Left decoy." }} else &.{.{ .name = "left", .text = "Left upgraded." }};
        return if (self.retry) &.{.{ .name = "right", .text = "Right decoy." }} else &.{.{ .name = "right", .text = "Right upgraded." }};
    }

    fn stopAfterRetry(ptr: *anyopaque, state: *const GEPA.StopState) !bool {
        const self: *const @This() = @ptrCast(@alignCast(ptr));
        return if (state.last_iteration) |i| i >= self.stop_iteration else false;
    }
};

fn runMerge(init: std.process.Init, dir: []const u8, phase: []const u8) !void {
    const arena = init.arena.allocator();
    const mode = std.meta.stringToEnum(enum {
        @"merge-start",
        @"merge-resume",
        @"merge-full-resume",
        @"merge-full-enable-cache",
        @"merge-full-disable-cache",
        @"merge-cached-start",
        @"merge-cached-resume",
        @"merge-cached-reject",
        @"merge-cached-retry",
        @"merge-cycle-disable",
        @"merge-cycle-enable",
    }, phase) orelse return error.InvalidArguments;
    const reenable = mode == .@"merge-cycle-enable";
    const pause_uncached = mode == .@"merge-cycle-disable";
    const cached = std.mem.startsWith(u8, phase, "merge-cached-") or mode == .@"merge-full-enable-cache" or reenable;
    const disable_cache = mode == .@"merge-full-disable-cache";
    const full = mode == .@"merge-full-resume" or mode == .@"merge-full-enable-cache" or disable_cache;
    const reject = mode == .@"merge-cached-reject";
    const retry = mode == .@"merge-cached-retry" or disable_cache or pause_uncached or reenable;
    var workload: MergeWorkload = .{
        .allocator = arena,
        .cached = cached,
        .full = full,
        .reject = reject,
        .retry = retry,
        .stop_iteration = if (pause_uncached) 3 else if (reenable) 5 else 4,
    };
    const examples = try rows(arena, &.{ "0", "1", "2", "3", "4", "5", "6", "7", "8" }, false);
    const result = try (GEPA{
        .metric = MergeWorkload.score,
        .max_metric_calls = switch (mode) {
            .@"merge-start" => 7,
            .@"merge-resume" => 20,
            .@"merge-full-resume" => 32,
            .@"merge-full-enable-cache" => 28,
            .@"merge-cached-start" => 6,
            .@"merge-cached-resume" => 17,
            .@"merge-cached-reject" => 15,
            .@"merge-cached-retry", .@"merge-full-disable-cache", .@"merge-cycle-disable", .@"merge-cycle-enable" => 100,
        },
        .stop_callbacks = if (retry) &.{.{ .ptr = &workload, .should_stop = MergeWorkload.stopAfterRetry }} else &.{},
        .cache_evaluation = cached,
        .merge_val_overlap_floor = 2,
        .candidate_selection_strategy = .{ .custom = .{ .ptr = &workload, .select = MergeWorkload.select } },
        .batch_sampler = .{ .ptr = &workload, .max_batch_size = 1, .next = MergeWorkload.sample },
        .val_evaluation_policy = if (full) null else .{
            .ptr = &workload,
            .max_batch_size = 4,
            .seed_batch = MergeWorkload.seedBatch,
            .next_batch = MergeWorkload.valBatch,
        },
        .instruction_proposer = .{ .ptr = &workload, .propose = MergeWorkload.propose },
    }).compile(MergeWorkload.Program, arena, .{
        .workspace = &workspace,
        .arena = arena,
        .lm = .{ .ptr = &workload, .vtable = &.{ .complete = MergeWorkload.complete } },
        .json_fallback = false,
    }, .{}, examples, .{
        .log_dir = .{ .io = init.io, .path = dir },
        .valset = examples,
        .checkpoint = .{ .io = init.io, .path = try std.fs.path.join(arena, &.{ dir, "native-checkpoint.json" }), .revision = "sparse merge" },
    });
    const Score = struct { usize, f64 };
    const scores = try arena.alloc([]const Score, result.candidates.len);
    for (scores, result.val_ids.?, result.val_subscores) |*output, ids, values| {
        const entries = try arena.alloc(Score, ids.len);
        for (entries, ids, values) |*entry, id, score| entry.* = .{ id, score };
        output.* = entries;
    }
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    try std.json.Stringify.value(.{
        .calls = workload.calls.items,
        .budget = result.total_metric_calls,
        .best = result.best_idx,
        .parents = result.parents,
        .scores = scores,
    }, .{}, &stdout.interface);
    try stdout.interface.flush();
}

fn rows(arena: std.mem.Allocator, questions: []const []const u8, training: bool) ![]dz.Example {
    const output = try arena.alloc(dz.Example, questions.len);
    for (output, questions) |*row, question| {
        row.* = try dz.Example.fromJson(arena, try std.fmt.allocPrint(arena, "{{\"question\":\"{s}\",\"training\":{s}}}", .{ question, if (training) "true" else "false" }), &.{"question"});
    }
    return output;
}
