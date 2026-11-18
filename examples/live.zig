//! Runs examples/live_suite.zig against a real OpenAI-compatible model and
//! writes the transcript of every request and real completion, so
//! tools/dspy_reference/check_live.py can replay it through DSPy 3.5.0.
//!
//!     OPENAI_API_KEY=... zig build live -- transcript.json
//!
//! For a Responses endpoint, set OPENAI_BASE_URL, DSZIG_LIVE_MODEL and
//! DSZIG_LIVE_API=responses. DSZIG_LIVE_REASONING=1 explicitly enables native
//! reasoning for a capable model. DSZIG_LIVE_RESPONSE_SCHEMA=1 and
//! DSZIG_LIVE_FUNCTION_CALLING=1 advertise the corresponding native capabilities.
//! CODEX_LB_API_KEY is accepted only with an explicit OPENAI_BASE_URL, so the
//! homelab credential can never fall through to the public OpenAI default.
//! `zig build live -- --smoke [transcript.json]` runs one real CoT request,
//! checks the answer to 2 + 2, and records its actual native metadata.
//! Defaults remain Chat Completions, gpt-4o-mini, and the public OpenAI URL.

const std = @import("std");
const dszig = @import("dszig");
const suite = @import("live_suite");

const Smoke = dszig.Signature(struct {
    pub const instructions = "Answer the arithmetic question. Give the final answer as a single digit.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});
const smoke_question = "What is 2 + 2?";

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const explicit_base_url = init.environ_map.get("OPENAI_BASE_URL");
    const key = init.environ_map.get("OPENAI_API_KEY") orelse homelab: {
        const homelab_key = init.environ_map.get("CODEX_LB_API_KEY") orelse {
            std.log.err("OPENAI_API_KEY or CODEX_LB_API_KEY must be set", .{});
            std.process.exit(2);
        };
        if (explicit_base_url == null) {
            std.log.err("OPENAI_BASE_URL must be set when using CODEX_LB_API_KEY", .{});
            std.process.exit(2);
        }
        break :homelab homelab_key;
    };
    const model = init.environ_map.get("DSZIG_LIVE_MODEL") orelse "gpt-4o-mini";
    const base_url = explicit_base_url orelse "https://api.openai.com/v1";
    const api_name = init.environ_map.get("DSZIG_LIVE_API") orelse "chat_completions";
    const api = std.meta.stringToEnum(dszig.lm.OpenAI.Api, api_name) orelse {
        std.log.err("DSZIG_LIVE_API must be chat_completions or responses", .{});
        std.process.exit(2);
    };
    const reasoning = enabled(init.environ_map.get("DSZIG_LIVE_REASONING"), "DSZIG_LIVE_REASONING");
    const response_schema = enabled(init.environ_map.get("DSZIG_LIVE_RESPONSE_SCHEMA"), "DSZIG_LIVE_RESPONSE_SCHEMA");
    const function_calling = enabled(init.environ_map.get("DSZIG_LIVE_FUNCTION_CALLING"), "DSZIG_LIVE_FUNCTION_CALLING");
    const smoke = args.len > 1 and std.mem.eql(u8, args[1], "--smoke");
    const out_path = if (smoke)
        if (args.len > 2) args[2] else "live_smoke.json"
    else if (args.len > 1)
        args[1]
    else
        "live_dszig.json";

    var client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer client.deinit();
    var openai: dszig.lm.OpenAI = .{
        .client = &client,
        .base_url = base_url,
        .model = model,
        .api = api,
        .api_key = key,
        .supports_reasoning = reasoning,
        .supports_response_schema = response_schema,
        .supports_function_calling = function_calling,
    };
    var buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    const w = &stdout.interface;
    var recorder: dszig.lm.RecordingLM = .{ .inner = openai.lm(), .arena = arena };
    if (smoke) {
        const program: dszig.ChainOfThought(Smoke) = .{};
        const result = try program.forward(.{
            .arena = arena,
            .lm = recorder.lm(),
            // A malformed model output must fail, not silently turn this
            // one-request smoke into an adapter-fallback conversation.
            .json_fallback = false,
        }, .{ .question = smoke_question });
        if (!std.mem.eql(u8, result.answer, "4")) return error.UnexpectedSmokeAnswer;
        if (recorder.calls.items.len != 1) return error.UnexpectedSmokeCallCount;
        const call = &recorder.calls.items[0];
        const json = try smokeTranscript(arena, model, api_name, recorder.lm(), call, result.answer);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        const native_summary = call.reasonings.len == 1 and
            call.reasonings[0] != null and call.reasonings[0].?.len != 0;
        try w.print("model: {s}; api: {s}; answer: {s}; native summary: {s}; wrote {s}\n", .{
            model,
            api_name,
            result.answer,
            if (native_summary) "yes" else "no",
            out_path,
        });
        try w.flush();
        return;
    }
    const result = try suite.run(arena, init.gpa, recorder.lm());
    const json = try suite.writeTranscript(arena, "dszig", model, recorder.lm(), recorder.calls.items, result);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
    for (result.cases) |c| {
        try w.print("{s}: {s}\n", .{ c.name, if (c.err) |e| @errorName(e) else "ok" });
    }
    try w.print("rag: baseline {d:.1}%  optimized {d:.1}%\n", .{ result.rag.baseline, result.rag.optimized });
    try w.print("dates: baseline {d:.1}%  optimized {d:.1}%\n", .{ result.dates.baseline, result.dates.optimized });
    try w.print("dates_search: baseline {d:.1}%  optimized {d:.1}%  candidates", .{ result.dates_search.baseline, result.dates_search.optimized });
    for (result.dates_search.candidates) |c| try w.print(" {d}:{d:.1}", .{ c.seed, c.score });
    try w.writeByte('\n');
    try w.print("dates_mipro: baseline {d:.1}%  optimized {d:.1}%  {d} trials\n", .{ result.dates_mipro.baseline, result.dates_mipro.optimized, result.dates_mipro.trials.len });
    try w.print("{d} real calls to {s}; wrote {s}\n", .{ recorder.calls.items.len, model, out_path });
    try w.flush();
}

/// A smoke transcript is deliberately not a full-suite result. It retains the
/// real request, completion and raw provider evidence without fake case data.
fn smokeTranscript(
    arena: std.mem.Allocator,
    model: []const u8,
    api: []const u8,
    lm: dszig.LM,
    call: *const dszig.lm.ReplayLM.Call,
    answer: []const u8,
) ![]const u8 {
    if (call.all().len != 1) return error.UnexpectedSmokeCallCount;
    var row = (try dszig.lm.requestValue(arena, .{
        .messages = call.messages,
        .options = call.options orelse return error.InvalidSmokeTranscript,
        .tools = call.tools,
        .decision = call.decision,
    })).object;
    try row.put(arena, "completion", if (call.completion) |text| .{ .string = text } else .null);
    var reasonings: dszig.Array = .init(arena);
    for (call.reasonings) |reasoning| try reasonings.append(if (reasoning) |text| .{ .string = text } else .null);
    try row.put(arena, "reasonings", .{ .array = reasonings });
    var tool_calls: dszig.Array = .init(arena);
    for (call.tool_calls) |native_calls| try tool_calls.append(try dszig.inference.toolCallsValue(arena, native_calls));
    try row.put(arena, "tool_calls", .{ .array = tool_calls });
    if (call.citations.len != 0) {
        var citations: dszig.Array = .init(arena);
        for (call.citations) |parts| try citations.append(try dszig.inference.citationsValue(arena, parts));
        try row.put(arena, "citations", .{ .array = citations });
    }
    var continuation: dszig.Array = .init(arena);
    for (call.continuation) |state| try continuation.append(try dszig.inference.continuationsValue(arena, state));
    try row.put(arena, "continuation", .{ .array = continuation });
    var provider_data: dszig.Array = .init(arena);
    for (call.provider_data) |data| try provider_data.append(data orelse .null);
    try row.put(arena, "provider_data", .{ .array = provider_data });
    if (call.metadata.len != 0) {
        var metadata: dszig.Array = .init(arena);
        for (call.metadata) |entry| try metadata.append(try dszig.lm.responseMetadataValue(arena, entry));
        try row.put(arena, "metadata", .{ .array = metadata });
    }
    if (call.generated_media.len != 0) {
        var flags: dszig.Array = .init(arena);
        for (call.generated_media) |flag| try flags.append(.{ .bool = flag });
        try row.put(arena, "generated_media", .{ .array = flags });
    }
    if (call.logprobs.len != 0) {
        var scores: dszig.Array = .init(arena);
        for (call.logprobs) |parts| try scores.append(if (parts) |entries| try dszig.lm.logprobs.toNativeValue(arena, entries) else .null);
        try row.put(arena, "logprobs", .{ .array = scores });
    }
    if (call.logprobs_complete.len != 0) {
        var flags: dszig.Array = .init(arena);
        for (call.logprobs_complete) |flag| try flags.append(.{ .bool = flag });
        try row.put(arena, "logprobs_complete", .{ .array = flags });
    }
    if (call.response_messages.len != 0) {
        var messages: dszig.Array = .init(arena);
        for (call.response_messages) |message|
            try messages.append(if (message) |entry| try entry.toValue(arena) else .null);
        try row.put(arena, "response_messages", .{ .array = messages });
    }
    if (call.adaptations.len != 0) {
        var adaptations: dszig.Array = .init(arena);
        for (call.adaptations) |entries|
            try adaptations.append(try dszig.sdk_response.adaptationsValue(arena, entries));
        try row.put(arena, "adaptations", .{ .array = adaptations });
    }
    var calls: dszig.Array = .init(arena);
    try calls.append(.{ .object = row });

    var root: dszig.ObjectMap = .empty;
    try root.put(arena, "runtime", .{ .string = "dszig" });
    try root.put(arena, "target_dspy", .{ .string = suite.target_dspy });
    try root.put(arena, "scenario", .{ .string = "cot_smoke" });
    try root.put(arena, "model", .{ .string = model });
    try root.put(arena, "api", .{ .string = api });
    try root.put(arena, "supports_reasoning", .{ .bool = lm.supports_reasoning });
    try root.put(arena, "supports_response_format", .{ .bool = lm.supports_response_format });
    try root.put(arena, "supports_response_schema", .{ .bool = lm.supports_response_schema });
    try root.put(arena, "supports_function_calling", .{ .bool = lm.supports_function_calling });
    if (lm.supports_decision_requests) try root.put(arena, "supports_decision_requests", .{ .bool = true });
    try root.put(arena, "question", .{ .string = smoke_question });
    try root.put(arena, "answer", .{ .string = answer });
    try root.put(arena, "calls", .{ .array = calls });
    var writer: std.Io.Writer.Allocating = .init(arena);
    try dszig.pyjson.writeJson(&writer.writer, .{ .object = root }, .indent2);
    return writer.written();
}

fn enabled(raw: ?[]const u8, name: []const u8) bool {
    const flag = raw orelse return false;
    if (std.mem.eql(u8, flag, "0")) return false;
    if (std.mem.eql(u8, flag, "1")) return true;
    std.log.err("{s} must be 0 or 1", .{name});
    std.process.exit(2);
}
