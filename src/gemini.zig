//! Gemini Developer API generateContent transport for text and structured text.
const std = @import("std");
const lm_mod = @import("lm.zig");
const http = @import("http.zig");
const runtime = @import("runtime_value.zig");
const content = @import("openai_content.zig");
const sdk = lm_mod.sdk;
const Allocator = std.mem.Allocator;
const Error = lm_mod.Error;
const Value = runtime.Value;
const Object = runtime.ObjectMap;

pub const Gemini = struct {
    client: *std.http.Client,
    model: []const u8,
    api_key: ?[]const u8 = null,
    base_url: []const u8 = "https://generativelanguage.googleapis.com/v1beta",
    max_tokens: ?u32 = null,
    max_response_bytes: usize = 4 * 1024 * 1024,
    max_retries: u8 = 3,
    retry_base_ms: u32 = 500,
    timeout_ns: ?u64 = null,
    connect_timeout_ns: ?u64 = null,
    header_provider: ?http.HeaderProvider = null,
    adaptations: enum { note, silent, refuse } = .note,
    supports_reasoning: bool = false,

    pub fn lm(self: *Gemini) lm_mod.LM {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = complete,
                .complete_stream = completeStream,
                .complete_canonical = canonical,
                .clone = clone,
                .model_name = modelName,
            },
            .supports_response_format = true,
            .supports_response_schema = true,
            .supports_function_calling = true,
            .supports_reasoning = self.supports_reasoning,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    fn clone(ptr: *anyopaque, arena: Allocator) anyerror!lm_mod.LM {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const model = try arena.dupe(u8, self.model);
        errdefer arena.free(model);
        const base = try arena.dupe(u8, self.base_url);
        errdefer arena.free(base);
        const copy = try arena.create(Gemini);
        copy.* = self.*;
        copy.model = model;
        copy.base_url = base;
        return copy.lm();
    }

    fn predictionRequest(self: *const Gemini, arena: Allocator, request: lm_mod.Request) Error!*const sdk.Request {
        for (request.tools) |tool| if (tool.strict != null) return error.LMFailure;
        return lm_mod.OpenAI.canonicalPredictionRequest(arena, request, self.model, self.max_tokens, null);
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) Error![]const lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const native = try self.predictionRequest(arena, request);
        defer native.release();
        const result = try arena.alloc(lm_mod.Completion, 1);
        result[0] = try self.completeFields(arena, native, .prediction);
        return result;
    }

    fn completeStream(
        ptr: *anyopaque,
        arena: Allocator,
        request: lm_mod.Request,
        sink: StreamSink,
    ) Error!lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const native = try self.predictionRequest(arena, request);
        defer native.release();
        var assembler: lm_mod.OpenAI.OpenAIStream = .{
            .arena = arena,
            .model = native.fields.model,
            .stop = &.{},
        };
        const outcome = try self.runStream(arena, &native.fields, .prediction, sink, &assembler);
        if (outcome == .stopped) return error.StreamStopped;
        var completion = assembler.result orelse return error.LMFailure;
        try lm_mod.publishResponse(arena, &completion, native.fields.model, .borrowed);
        return completion;
    }

    pub fn completeRequest(self: *Gemini, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        return (try canonical(self, arena, request)).response.?;
    }

    fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) Error!lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        return self.completeFields(arena, request, .canonical);
    }

    fn completeFields(
        self: *const Gemini,
        arena: Allocator,
        request: *const sdk.Request,
        mode: RequestMode,
    ) Error!lm_mod.Completion {
        const fields = &request.fields;
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try requestBody(arena, fields, mode, self.adaptations, &records);
        const reply = try self.send(arena, fields.model, body, null);
        defer self.client.allocator.free(reply.body);
        const raw = @import("json_repair.zig").parseJson(arena, reply.body) catch |err|
            return codecError(err);
        var result = try parseResponse(arena, fields.model, raw, records.items);
        try lm_mod.OpenAI.foldJudgments(arena, &result, request);
        try lm_mod.publishResponse(arena, &result, fields.model, .borrowed);
        return result;
    }

    pub const StreamEvent = lm_mod.OpenAI.StreamEvent;
    pub const StreamSink = lm_mod.OpenAI.StreamSink;
    pub const StreamOutcome = lm_mod.OpenAI.StreamOutcome;

    /// Event payloads borrow frame storage during the callback. `.stop` closes the stream.
    pub fn streamRequest(
        self: *Gemini,
        backing: Allocator,
        request: *const sdk.Request,
        sink: StreamSink,
    ) Error!StreamOutcome {
        var scratch: std.heap.ArenaAllocator = .init(backing);
        defer scratch.deinit();
        return self.runStream(scratch.allocator(), &request.fields, .canonical, sink, null);
    }

    fn runStream(
        self: *Gemini,
        arena: Allocator,
        fields: *const sdk.RequestFields,
        mode: RequestMode,
        sink: StreamSink,
        assembler: ?*lm_mod.OpenAI.OpenAIStream,
    ) Error!StreamOutcome {
        const extensions = if (fields.config.extensions) |value|
            value.toValue(arena) catch |err| return codecError(err)
        else
            Value.null;
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        var state: GenerateStream = .{
            .model = fields.model,
            .adaptations = &.{},
            .sink = sink,
            .assembler = assembler,
        };
        if (usesLive(fields.model, extensions)) {
            self.runLive(arena, fields, extensions, &state) catch |err| return codecError(err);
            return state.outcome;
        }
        const body = try requestBody(arena, fields, mode, self.adaptations, &records);
        state.adaptations = records.items;
        var consumer: http.ResponseConsumer = .{
            .context = &state,
            .consume = GenerateStream.consume,
        };
        const reply = try self.send(arena, fields.model, body, &consumer);
        defer self.client.allocator.free(reply.body);
        return state.outcome;
    }

    fn runLive(
        self: *const Gemini,
        arena: Allocator,
        fields: *const sdk.RequestFields,
        extensions: Value,
        state: *GenerateStream,
    ) !void {
        if (self.header_provider != null) return error.LMFailure;
        const uri = try liveEndpoint(arena, self.base_url, self.api_key);
        const payloads = try livePayloads(arena, fields, extensions);
        var timed: @import("timed_io.zig") = .{
            .inner = self.client.io,
            .timeout_ns = self.timeout_ns orelse 600 * std.time.ns_per_s,
            .connect_timeout_ns = self.connect_timeout_ns orelse self.timeout_ns orelse
                5 * std.time.ns_per_s,
        };
        var client = try http.isolatedClient(self.client, timed.io());
        defer client.deinit();
        var socket = try @import("websocket.zig").open(&client, uri);
        defer socket.deinit();
        try socket.sendText(payloads.setup);

        var buffer: std.ArrayList(u8) = .empty;
        defer buffer.deinit(client.allocator);
        var scratch: std.heap.ArenaAllocator = .init(client.allocator);
        defer scratch.deinit();
        var remaining = self.max_response_bytes;
        for (0..8192) |_| {
            const value = try receiveLive(&socket, &scratch, &buffer, &remaining);
            if (value != .object) continue;
            if (value.object.contains("setupComplete")) break;
            if (value.object.contains("error")) return error.LMFailure;
        } else return error.LMFailure;

        for (payloads.input) |payload| try socket.sendText(payload);
        try state.start();
        const usage_allocator = if (state.assembler == null) client.allocator else arena;
        var usage: sdk.Usage = .{};
        defer if (state.assembler == null) usage.free(usage_allocator);
        var saw_tool = false;
        for (0..8192) |_| {
            if (state.outcome == .stopped) return;
            const value = try receiveLive(&socket, &scratch, &buffer, &remaining);
            if (value != .object) continue;
            if (try dispatchLive(scratch.allocator(), usage_allocator, state, value, &usage, &saw_tool)) {
                if (state.outcome != .stopped) socket.finish() catch {
                    // The terminal event already committed the completed turn.
                };
                return;
            }
        }
        return error.LMFailure;
    }

    fn send(
        self: *const Gemini,
        arena: Allocator,
        model: []const u8,
        body: []const u8,
        consumer: ?*http.ResponseConsumer,
    ) Error!http.Response {
        const url = try endpoint(arena, self.base_url, model, consumer != null);
        var headers: [1]std.http.Header = undefined;
        var headers_len: usize = 0;
        if (self.api_key) |key| {
            if (key.len == 0 or std.mem.indexOfAny(u8, key, "\r\n") != null) return error.LMFailure;
            headers[0] = .{ .name = "x-goog-api-key", .value = key };
            headers_len = 1;
        }
        const transport: http.Transport = .{
            .client = self.client,
            .max_response_bytes = self.max_response_bytes,
            .max_retries = self.max_retries,
            .retry_base_ms = self.retry_base_ms,
            .timeout_ns = self.timeout_ns,
            .connect_timeout_ns = self.connect_timeout_ns,
            .header_provider = self.header_provider,
            .success = .ok,
            .failure_error = error.LMFailure,
            .response_limit_error = error.LMFailure,
        };
        return transport.send(arena, .{
            .url = url,
            .method = .POST,
            .body = body,
            .content_type = "application/json",
            .headers = headers[0..headers_len],
            .response_consumer = consumer,
        }) catch |err| return codecError(err);
    }
};

fn usesLive(model: []const u8, extensions: Value) bool {
    const transport = text(get(extensions, "transport"));
    for ([_][]const u8{ "live", "websocket", "ws" }) |name| {
        if (std.ascii.eqlIgnoreCase(transport, name)) return true;
    }
    return hasPhrase(model, "-live") or
        (model.len >= 4 and std.ascii.eqlIgnoreCase(model[model.len - 4 ..], "live"));
}

fn liveEndpoint(arena: Allocator, base_url: []const u8, key: ?[]const u8) !std.Uri {
    var uri = try std.Uri.parse(base_url);
    if (!contains(&.{ "http", "https" }, uri.scheme)) return error.LMFailure;
    if (uri.user != null or uri.password != null) return error.LMFailure;
    uri.fragment = null;
    uri.path = .{ .percent_encoded = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent" };
    uri.query = null;
    if (key) |bytes| {
        if (bytes.len == 0) return error.LMFailure;
        var query: std.Io.Writer.Allocating = .init(arena);
        errdefer query.deinit();
        try query.writer.writeAll("key=");
        const hex = "0123456789ABCDEF";
        for (bytes) |byte| {
            if (std.ascii.isAlphanumeric(byte) or std.mem.indexOfScalar(u8, "-._~", byte) != null) {
                try query.writer.writeByte(byte);
            } else if (byte == ' ') {
                try query.writer.writeByte('+');
            } else {
                try query.writer.writeAll(&.{ '%', hex[byte >> 4], hex[byte & 15] });
            }
        }
        uri.query = .{ .percent_encoded = try query.toOwnedSlice() };
    }
    return uri;
}

const LivePayloads = struct { setup: []const u8, input: []const []const u8 };

fn livePayloads(
    arena: Allocator,
    fields: *const sdk.RequestFields,
    extensions: Value,
) !LivePayloads {
    if (extensions != .object and extensions != .null) return error.LMFailure;
    const audio_native = hasPhrase(fields.model, "live-preview") or
        hasPhrase(fields.model, "native-audio");
    const output = text(get(extensions, "output"));
    const audio = audio_native or std.mem.eql(u8, output, "audio");
    const image = std.mem.eql(u8, output, "image");
    var has_activity = false;
    if (audio_native) {
        for (fields.messages) |message| for (message.parts) |part| {
            if (part == .audio or part == .video) has_activity = true;
        };
    }

    var setup: Object = .empty;
    const model = fields.model;
    try setup.put(arena, "model", str(if (std.mem.startsWith(u8, model, "models/"))
        model
    else
        try std.fmt.allocPrint(arena, "models/{s}", .{model})));
    if (fields.system) |system| {
        const bytes = switch (system) {
            .text => |value| value,
            .parts => |parts| try content.partsText(arena, parts, null),
        };
        if (bytes.len != 0) {
            const parts = try singleton(arena, try textPart(arena, bytes));
            try setup.put(arena, "systemInstruction", try wrapped(arena, "parts", parts));
        }
    }
    const declarations = try functionDeclarations(arena, fields.tools);
    if (declarations.items.len != 0) {
        const group = try wrapped(arena, "functionDeclarations", .{ .array = declarations });
        try setup.put(arena, "tools", try singleton(arena, group));
    }
    if (extensions == .object) {
        var entries = extensions.object.iterator();
        while (entries.next()) |entry| {
            if (contains(&.{ "transport", "prompt_caching", "output" }, entry.key_ptr.*)) continue;
            try setup.put(arena, entry.key_ptr.*, entry.value_ptr.*);
        }
    }
    try liveSetupModalities(arena, &setup, audio, image, has_activity);
    if (audio and !std.mem.eql(u8, output, "audio")) {
        try setup.put(arena, "outputAudioTranscription", .{ .object = .empty });
    }
    return .{
        .setup = try liveJson(arena, try wrapped(arena, "setup", .{ .object = setup })),
        .input = if (audio_native)
            try liveRealtimeFrames(arena, fields.messages, has_activity)
        else
            try liveTurnFrames(arena, fields.messages),
    };
}

fn liveSetupModalities(
    arena: Allocator,
    setup: *Object,
    audio: bool,
    image: bool,
    activity: bool,
) !void {
    var generation = try liveObjectCopy(arena, setup.get("generationConfig"));
    if (audio or image or !generation.contains("responseModalities")) {
        const modality_name = if (audio) "AUDIO" else if (image) "IMAGE" else "TEXT";
        try generation.put(arena, "responseModalities", try singleton(arena, str(modality_name)));
    }
    try setup.put(arena, "generationConfig", .{ .object = generation });
    if (!activity) return;
    var realtime = try liveObjectCopy(arena, setup.get("realtimeInputConfig"));
    var detection = try liveObjectCopy(arena, realtime.get("automaticActivityDetection"));
    try detection.put(arena, "disabled", .{ .bool = true });
    try realtime.put(arena, "automaticActivityDetection", .{ .object = detection });
    try setup.put(arena, "realtimeInputConfig", .{ .object = realtime });
}

fn liveObjectCopy(arena: Allocator, item: ?Value) !Object {
    const value = item orelse return .empty;
    if (value != .object) return error.LMFailure;
    return value.object.clone(arena);
}

fn liveTurnFrames(arena: Allocator, messages: []const sdk.Message) ![]const []const u8 {
    const payloads = try arena.alloc([]const u8, 1);
    if (messages.len == 1 and messages[0].role == .user) text_only: {
        for (messages[0].parts) |part| if (part != .text) break :text_only;
        const bytes = try content.partsText(arena, messages[0].parts, null);
        payloads[0] = try liveFrame(arena, "realtimeInput", "text", str(bytes));
        return payloads;
    }
    var turns: runtime.Array = .init(arena);
    var names: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer names.deinit(arena);
    for (messages) |message| for (message.parts) |part| {
        if (part == .tool_call) try names.put(arena, part.tool_call.id, part.tool_call.name);
    };
    for (messages) |message| {
        var parts: runtime.Array = .init(arena);
        for (message.parts) |part| try parts.append(try encodePart(arena, part, &names));
        var turn: Object = .empty;
        try turn.put(arena, "role", str(if (message.role == .assistant) "model" else "user"));
        try turn.put(arena, "parts", .{ .array = parts });
        try turns.append(.{ .object = turn });
    }
    payloads[0] = try liveContentFrame(arena, turns, true);
    return payloads;
}

fn liveContentFrame(arena: Allocator, turns: runtime.Array, complete: bool) ![]const u8 {
    var object: Object = .empty;
    try object.put(arena, "turns", .{ .array = turns });
    try object.put(arena, "turnComplete", .{ .bool = complete });
    return liveJson(arena, try wrapped(arena, "clientContent", .{ .object = object }));
}

fn liveFrame(arena: Allocator, outer: []const u8, inner: []const u8, value: Value) ![]const u8 {
    return liveJson(arena, try wrapped(arena, outer, try wrapped(arena, inner, value)));
}

fn liveRealtimeFrames(
    arena: Allocator,
    messages: []const sdk.Message,
    activity: bool,
) ![]const []const u8 {
    var parts: runtime.Array = .init(arena);
    var texts: std.ArrayList([]const u8) = .empty;
    var media: std.ArrayList([]const u8) = .empty;
    const names: std.StringHashMapUnmanaged([]const u8) = .empty;
    for (messages) |message| for (message.parts) |part| switch (part) {
        .text => |item| {
            if (item.text.len != 0) try texts.append(
                arena,
                try liveFrame(arena, "realtimeInput", "text", str(item.text)),
            );
        },
        .audio, .video => |item| try media.append(
            arena,
            try liveMediaFrame(arena, if (part == .audio) "audio" else "video", &item),
        ),
        .image, .document, .binary => try parts.append(try encodePart(arena, part, &names)),
        else => return error.LMFailure,
    };
    var frames: std.ArrayList([]const u8) = .empty;
    if (activity) try frames.append(
        arena,
        try liveFrame(arena, "realtimeInput", "activityStart", .{ .object = .empty }),
    );
    if (parts.items.len != 0) {
        var turn: Object = .empty;
        try turn.put(arena, "role", str("user"));
        try turn.put(arena, "parts", .{ .array = parts });
        const turns = try singleton(arena, .{ .object = turn });
        try frames.append(arena, try liveContentFrame(arena, turns.array, false));
    }
    try frames.appendSlice(arena, texts.items);
    try frames.appendSlice(arena, media.items);
    if (activity) try frames.append(
        arena,
        try liveFrame(arena, "realtimeInput", "activityEnd", .{ .object = .empty }),
    );
    if (frames.items.len == 0) try frames.append(
        arena,
        try liveFrame(arena, "realtimeInput", "text", str("")),
    );
    return frames.toOwnedSlice(arena);
}

fn liveMediaFrame(arena: Allocator, name: []const u8, media: *const sdk.MediaPart) ![]const u8 {
    if (media.path != null) return error.LMFailure;
    var bytes = media.data orelse return error.LMFailure;
    var mime = media.media_type;
    const audio = std.mem.eql(u8, name, "audio");
    if (mime.len == 0) mime = if (audio) "audio/pcm" else "video/mp4";
    if (audio and (std.mem.find(u8, mime, "wav") != null or
        std.mem.find(u8, mime, "wave") != null))
    {
        const decoder = std.base64.standard.Decoder;
        const raw = try arena.alloc(u8, try decoder.calcSizeForSlice(bytes));
        try decoder.decode(raw, bytes);
        const pcm = liveWave(raw);
        const encoder = std.base64.standard.Encoder;
        bytes = encoder.encode(try arena.alloc(u8, encoder.calcSize(pcm.bytes.len)), pcm.bytes);
        mime = try std.fmt.allocPrint(arena, "audio/pcm;rate={d}", .{pcm.rate});
    }
    var media_value: Object = .empty;
    try media_value.put(arena, "mimeType", str(mime));
    try media_value.put(arena, "data", str(bytes));
    return liveFrame(arena, "realtimeInput", name, .{ .object = media_value });
}

fn liveWave(raw: []const u8) struct { bytes: []const u8, rate: u32 } {
    if (raw.len < 44 or !std.mem.eql(u8, raw[0..4], "RIFF") or
        !std.mem.eql(u8, raw[8..12], "WAVE"))
    {
        return .{ .bytes = raw, .rate = 16000 };
    }
    const rate = std.mem.readInt(u32, raw[24..28], .little);
    var offset: usize = 12;
    while (raw.len - offset >= 8) {
        const id = raw[offset..][0..4];
        const size = std.mem.readInt(u32, raw[offset + 4 ..][0..4], .little);
        offset += 8;
        if (std.mem.eql(u8, id, "data")) {
            return .{ .bytes = raw[offset..][0..@min(size, raw.len - offset)], .rate = rate };
        }
        if (size > raw.len - offset) break;
        offset += size;
    }
    return .{ .bytes = raw[44..], .rate = rate };
}

fn liveJson(arena: Allocator, value: Value) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(arena);
    errdefer output.deinit();
    try @import("pyjson.zig").writeJson(&output.writer, value, .minified);
    return output.toOwnedSlice();
}

fn receiveLive(
    socket: *@import("websocket.zig"),
    scratch: *std.heap.ArenaAllocator,
    buffer: *std.ArrayList(u8),
    remaining: *usize,
) !Value {
    _ = scratch.reset(.retain_capacity);
    const bytes = try socket.receive(scratch.child_allocator, buffer, remaining.*);
    remaining.* -= bytes.len;
    return @import("json_repair.zig").parseJson(scratch.allocator(), bytes) catch |err| {
        if (err == error.OutOfMemory) return err;
        return .null;
    };
}

fn dispatchLive(
    scratch: Allocator,
    usage_allocator: Allocator,
    state: *GenerateStream,
    value: Value,
    usage: *sdk.Usage,
    saw_tool: *bool,
) !bool {
    if (value.object.get("error")) |detail| {
        try state.emit(.{ .provider_error = try streamError(scratch, detail) });
        return true;
    }
    const calls = get(get(value, "toolCall"), "functionCalls");
    for (elems(calls), 0..) |call, index| {
        if (call != .object) continue;
        const fallback = if (try truthyText(scratch, get(call, "id"))) |id|
            id
        else
            try std.fmt.allocPrint(scratch, "fc_{d}", .{index});
        try emitLiveTool(scratch, usage_allocator, state, call, index, fallback);
        saw_tool.* = true;
        if (state.outcome == .stopped) return true;
    }
    const server = get(value, "serverContent");
    const top_usage = get(value, "usageMetadata");
    const counts = if (top_usage == .object) top_usage else get(server, "usageMetadata");
    if (counts == .object and counts.object.count() != 0) {
        var live_counts = counts;
        if (counts.object.get("responseTokenCount")) |output| {
            live_counts = .{ .object = try counts.object.clone(scratch) };
            try live_counts.object.put(scratch, "candidatesTokenCount", output);
        }
        const parsed = try readUsage(scratch, live_counts);
        const replacement = try parsed.clone(usage_allocator);
        usage.free(usage_allocator);
        usage.* = replacement;
    }
    for (elems(get(get(server, "modelTurn"), "parts")), 0..) |part, index| {
        const call = get(part, "functionCall");
        if (call == .object and !part.object.contains("text")) {
            const id = (try truthyText(scratch, get(call, "id"))) orelse "fc_0";
            try emitLiveTool(scratch, usage_allocator, state, call, index, id);
            saw_tool.* = true;
            if (state.outcome == .stopped) return true;
            continue;
        }
        const data = (try decodeLivePart(scratch, usage_allocator, state, part)) orelse continue;
        var key_buffer: [24]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buffer, "{d}", .{index}) catch unreachable;
        try state.emit(.{ .delta = .{ .part_index = key, .data = data } });
        if (state.outcome == .stopped) return true;
    }
    if (try truthyText(scratch, get(get(server, "outputTranscription"), "text"))) |bytes| {
        const retained = if (state.assembler != null) try usage_allocator.dupe(u8, bytes) else bytes;
        try state.emit(.{ .delta = .{ .part_index = null, .data = .{ .text = .{ .text = retained } } } });
        if (state.outcome == .stopped) return true;
    }
    const complete = get(server, "turnComplete");
    if (!(lm_mod.OpenAI.responseTruthy(complete) catch |err| return codecError(err))) return false;
    try state.emit(.{ .end = .{
        .finish_reason = if (saw_tool.*) .tool_call else .stop,
        .usage = if (usage.isKnown()) usage.* else null,
        .provider_data = null,
    } });
    return true;
}

fn emitLiveTool(
    scratch: Allocator,
    retained: Allocator,
    state: *GenerateStream,
    call: Value,
    index: usize,
    id: []const u8,
) !void {
    const payload_arena = if (state.assembler != null) retained else scratch;
    const args = call.object.get("args") orelse Value{ .object = .empty };
    var output: std.Io.Writer.Allocating = .init(payload_arena);
    errdefer output.deinit();
    try @import("pyjson.zig").writeJsonAscii(&output.writer, args);
    const input = try output.toOwnedSlice();
    const name = (try truthyText(scratch, get(call, "name"))) orelse "tool";
    var key_buffer: [24]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buffer, "{d}", .{index}) catch unreachable;
    try state.emit(.{ .delta = .{
        .part_index = key,
        .data = .{ .tool_call = .{
            .id = if (state.assembler != null) try retained.dupe(u8, id) else id,
            .name = if (state.assembler != null) try retained.dupe(u8, name) else name,
            .input = input,
        } },
    } });
}

fn decodeLivePart(
    scratch: Allocator,
    retained: Allocator,
    state: *const GenerateStream,
    part: Value,
) !?GenerateStream.stream.DeltaData {
    if (part != .object) return null;
    const retain = state.assembler != null;
    if (part.object.get("text")) |value| {
        const bytes = (try truthyText(scratch, value)) orelse "";
        return .{ .text = .{ .text = if (retain) try retained.dupe(u8, bytes) else bytes } };
    }
    const inline_data = get(part, "inlineData");
    const mime = text(get(inline_data, "mimeType"));
    const audio = std.mem.startsWith(u8, mime, "audio/");
    if (!audio and !std.mem.startsWith(u8, mime, "image/")) return null;
    const bytes = text(get(inline_data, "data"));
    const media: GenerateStream.stream.Media = .{
        .data = if (retain) try retained.dupe(u8, bytes) else bytes,
        .media_type = if (retain) try retained.dupe(u8, mime) else mime,
    };
    return if (audio) .{ .audio = media } else .{ .image = media };
}

fn endpoint(arena: Allocator, base_url: []const u8, model: []const u8, stream: bool) Error![]const u8 {
    // Match urllib.quote(safe="/:@") on the model, not on the configured origin.
    var writer: std.Io.Writer.Allocating = .init(arena);
    const out = &writer.writer;
    out.writeAll(std.mem.trimEnd(u8, base_url, "/")) catch return error.OutOfMemory;
    out.writeByte('/') catch return error.OutOfMemory;
    if (!std.mem.startsWith(u8, model, "models/")) out.writeAll("models/") catch return error.OutOfMemory;
    const hex = "0123456789ABCDEF";
    for (model) |byte| {
        if (std.ascii.isAlphanumeric(byte) or std.mem.indexOfScalar(u8, "-._~/:@", byte) != null) {
            out.writeByte(byte) catch return error.OutOfMemory;
        } else {
            out.writeAll(&.{ '%', hex[byte >> 4], hex[byte & 15] }) catch return error.OutOfMemory;
        }
    }
    out.writeAll(if (stream) ":streamGenerateContent?alt=sse" else ":generateContent") catch
        return error.OutOfMemory;
    return writer.written();
}

const GenerateStream = struct {
    model: []const u8,
    adaptations: []const sdk.Adaptation,
    sink: Gemini.StreamSink,
    assembler: ?*lm_mod.OpenAI.OpenAIStream = null,
    outcome: Gemini.StreamOutcome = .exhausted,
    started: bool = false,
    previous: ?Slot = null,
    next_index: usize = 0,

    const Kind = enum { text, thinking, tool_call, audio, image };
    const Slot = struct { index: usize, kind: Kind };
    const stream = @import("openai_stream.zig");

    fn consume(context: *anyopaque, arena: Allocator, reader: *std.Io.Reader, limit: usize) !void {
        const self: *GenerateStream = @ptrCast(@alignCast(context));
        var events = @import("sse.zig").Iterator.init(arena, reader, limit);
        defer events.deinit();
        var frame: std.heap.ArenaAllocator = .init(arena);
        defer frame.deinit();
        while (try events.next()) |bytes| {
            if (bytes.len == 0) continue;
            // Only raw delivery can recycle parsed leaves after each callback.
            const temp = if (self.assembler != null) arena else frame.allocator();
            const value = try @import("json_repair.zig").parseJson(temp, bytes);
            if (value != .object) continue;
            if (try self.dispatch(temp, value)) return;
            if (self.outcome == .stopped) return;
            _ = frame.reset(.retain_capacity);
        }
    }

    fn emit(self: *GenerateStream, event: Gemini.StreamEvent) Error!void {
        if (self.outcome == .stopped) return;
        if (self.sink.on_event(self.sink.context, event) == .stop) {
            self.outcome = .stopped;
            return;
        }
        if (self.assembler) |assembler| try assembler.acceptRetainedEvent(event);
    }

    fn start(self: *GenerateStream) Error!void {
        if (self.started) return;
        self.started = true;
        try self.emit(.{ .start = .{ .id = null, .model = self.model, .adaptations = self.adaptations } });
    }

    fn slot(self: *GenerateStream, index: usize, kind: Kind) usize {
        if (index == 0 and (kind == .text or kind == .thinking)) {
            if (self.previous) |previous| if (previous.kind == kind) return previous.index;
        }
        const result = self.next_index;
        self.next_index += 1;
        self.previous = .{ .index = result, .kind = kind };
        return result;
    }

    fn dispatch(self: *GenerateStream, arena: Allocator, value: Value) Error!bool {
        const candidates = elems(get(value, "candidates"));
        const candidate = if (candidates.len == 0) .null else candidates[0];
        const finish = text(get(candidate, "finishReason"));
        if (value.object.get("error")) |detail| {
            try self.emit(.{ .provider_error = try streamError(arena, detail) });
            return true;
        }
        if (try blockedStreamError(arena, value, candidate)) |detail| {
            try self.emit(.{ .provider_error = detail });
            return true;
        }
        var saw_tool = false;
        var saw_delta = false;
        var scores = try readLogprobs(arena, get(candidate, "logprobsResult"));
        for (elems(get(get(candidate, "content"), "parts")), 0..) |part, index| {
            var decoded = (try decodePart(arena, part)) orelse continue;
            if (decoded.data == .text) {
                decoded.data.text.scores = scores;
                scores = &.{};
            }
            try self.start();
            const position = self.slot(index, decoded.kind);
            var buffer: [24]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{d}", .{position}) catch unreachable;
            try self.emit(.{ .delta = .{ .part_index = key, .data = decoded.data } });
            if (self.outcome == .stopped) return true;
            const continuation = try parseSignature(arena, decoded.signature);
            if (continuation.len != 0) {
                try self.emit(.{ .delta = .{
                    .part_index = key,
                    .data = .{ .continuation = continuation[0] },
                } });
                if (self.outcome == .stopped) return true;
            }
            saw_delta = true;
            saw_tool = saw_tool or decoded.kind == .tool_call;
        }
        const usage = get(value, "usageMetadata");
        if (finish.len != 0 or (!saw_delta and usage != .null)) {
            try self.start();
            try self.emit(.{ .end = .{
                .finish_reason = finishReason(finish, saw_tool),
                .usage = if (usage != .null) try readUsage(arena, usage) else null,
                .provider_data = sdk.PayloadValue.fromBorrowedValue(arena, value) catch |err|
                    return codecError(err),
            } });
        }
        return false;
    }

    const Decoded = struct { kind: Kind, data: stream.DeltaData, signature: Value = .null };

    fn decodePart(arena: Allocator, part: Value) Error!?Decoded {
        const signature = get(part, "thoughtSignature");
        const bytes = get(part, "text");
        const thought = get(part, "thought");
        if (bytes == .string) {
            if (thought == .bool and thought.bool)
                return .{ .kind = .thinking, .data = .{ .thinking = bytes.string }, .signature = signature };
            return .{ .kind = .text, .data = .{ .text = .{ .text = bytes.string } }, .signature = signature };
        }
        const call = get(part, "functionCall");
        if (call == .object) {
            const args = call.object.get("args") orelse Value{ .object = .empty };
            var writer: std.Io.Writer.Allocating = .init(arena);
            @import("pyjson.zig").writeJson(&writer.writer, args, .minified) catch |err|
                return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
            return .{
                .kind = .tool_call,
                .data = .{ .tool_call = .{
                    .input = writer.written(),
                    .id = nonempty(get(call, "id")),
                    .name = nonempty(get(call, "name")),
                } },
                .signature = if (nonempty(signature) != null) signature else get(call, "thoughtSignature"),
            };
        }
        const inline_data = get(part, "inlineData");
        if (inline_data != .object) return null;
        const mime = text(get(inline_data, "mimeType"));
        const media: stream.Media = .{ .media_type = mime, .data = text(get(inline_data, "data")) };
        if (std.mem.startsWith(u8, mime, "image/")) return .{ .kind = .image, .data = .{ .image = media } };
        if (std.mem.startsWith(u8, mime, "audio/")) return .{ .kind = .audio, .data = .{ .audio = media } };
        return null;
    }
};

const StreamError = @import("openai_stream.zig").ErrorDetail;

fn truthyText(arena: Allocator, value: Value) Error!?[]const u8 {
    const truthy = lm_mod.OpenAI.responseTruthy(value) catch |err| return codecError(err);
    if (!truthy) return null;
    if (value == .string) return value.string;
    var output: std.Io.Writer.Allocating = .init(arena);
    errdefer output.deinit();
    @import("pyjson.zig").writeStr(&output.writer, value) catch |err| return codecError(err);
    return try output.toOwnedSlice();
}

fn streamError(arena: Allocator, detail: Value) Error!StreamError {
    const status = try truthyText(arena, get(detail, "status")) orelse
        try truthyText(arena, get(detail, "code")) orelse "provider";
    const message = try truthyText(arena, get(detail, "message")) orelse status;
    return .{
        .code = providerError(status, message),
        .provider_code = status,
        .message = message,
    };
}

fn blockedStreamError(arena: Allocator, frame: Value, candidate: Value) Error!?StreamError {
    const block = try truthyText(arena, get(get(frame, "promptFeedback"), "blockReason"));
    const message = message: {
        if (block) |reason| {
            if (!std.mem.eql(u8, reason, "BLOCK_REASON_UNSPECIFIED")) {
                break :message try std.fmt.allocPrint(
                    arena,
                    "Prompt blocked: {s} (gemini)",
                    .{reason},
                );
            }
        }
        const finish = text(get(candidate, "finishReason"));
        if (!invalidFinish(finish)) return null;
        if (try truthyText(arena, get(candidate, "finishMessage"))) |detail| {
            break :message try std.fmt.allocPrint(arena, "{s} (gemini)", .{detail});
        }
        break :message try std.fmt.allocPrint(arena, "Candidate blocked: {s} (gemini)", .{finish});
    };
    return .{
        .code = .invalid_request,
        .provider_code = "inband_finish_reason",
        .message = message,
    };
}

fn hasPhrase(haystack: []const u8, comptime needle: []const u8) bool {
    comptime std.debug.assert(needle.len != 0);
    if (haystack.len < needle.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

fn providerError(code: []const u8, message: []const u8) @FieldType(StreamError, "code") {
    if (hasPhrase(message, "token")) {
        if (hasPhrase(message, "limit") or hasPhrase(message, "exceed")) return .context_length;
    }
    if (hasPhrase(message, "too long") or hasPhrase(message, "context length")) {
        return .context_length;
    }
    if (std.mem.eql(u8, code, "NOT_FOUND") and hasPhrase(message, "model")) {
        inline for (.{
            "not found",   "not exist",     "not supported",
            "unsupported", "not available", "unknown",
        }) |marker| {
            if (hasPhrase(message, marker)) return .unsupported_model;
        }
    }
    if (contains(&.{ "INVALID_ARGUMENT", "NOT_FOUND" }, code)) return .invalid_request;
    if (contains(&.{ "PERMISSION_DENIED", "UNAUTHENTICATED" }, code)) return .auth;
    if (contains(&.{ "INTERNAL", "UNAVAILABLE" }, code)) return .server;
    if (std.mem.eql(u8, code, "FAILED_PRECONDITION")) return .billing;
    if (std.mem.eql(u8, code, "RESOURCE_EXHAUSTED")) return .rate_limit;
    if (std.mem.eql(u8, code, "DEADLINE_EXCEEDED")) return .timeout;
    return .provider;
}

fn nonempty(value: Value) ?[]const u8 {
    const bytes = text(value);
    return if (bytes.len == 0) null else bytes;
}

fn finishReason(reason: []const u8, saw_tool: bool) sdk.FinishReason {
    if (saw_tool) return .tool_call;
    if (std.ascii.eqlIgnoreCase(reason, "MAX_TOKENS")) return .length;
    return .stop;
}

fn invalidFinish(reason: []const u8) bool {
    return contains(&.{
        "SAFETY",             "RECITATION",              "LANGUAGE",             "BLOCKLIST",                "PROHIBITED_CONTENT",
        "SPII",               "MALFORMED_FUNCTION_CALL", "IMAGE_SAFETY",         "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER",
        "NO_IMAGE",           "IMAGE_RECITATION",        "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS",      "MISSING_THOUGHT_SIGNATURE",
        "MALFORMED_RESPONSE",
    }, reason);
}

const RequestMode = enum { canonical, prediction };

fn requestBody(
    arena: Allocator,
    fields: *const sdk.RequestFields,
    mode: RequestMode,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error![]const u8 {
    const config = &fields.config;
    const extensions = if (config.extensions) |value|
        value.toValue(arena) catch |err| return codecError(err)
    else
        Value{ .object = .empty };
    if (extensions != .object) return error.LMFailure;
    var names: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer names.deinit(arena);
    for (fields.messages) |message| for (message.parts) |part| {
        if (part == .tool_call) try names.put(arena, part.tool_call.id, part.tool_call.name);
    };
    const cache = try cachePrefix(arena, fields, policy, records);
    const messages = fields.messages[cache.suffix..];
    var rows: runtime.Array = .init(arena);
    try rows.ensureTotalCapacity(messages.len);
    for (messages) |message| {
        var row: Object = .empty;
        try row.put(arena, "role", str(if (message.role == .assistant) "model" else "user"));
        const parts = if (message.role == .developer)
            try joinedParts(arena, "[developer]", message.parts)
        else blk: {
            var values: runtime.Array = .init(arena);
            try values.ensureTotalCapacity(message.parts.len);
            for (message.parts) |part| values.appendAssumeCapacity(try encodePart(arena, part, &names));
            break :blk Value{ .array = values };
        };
        try row.put(arena, "parts", parts);
        rows.appendAssumeCapacity(.{ .object = row });
    }
    var body: Object = .empty;
    try body.put(arena, "contents", .{ .array = rows });
    if (cache.resource) |resource| {
        try body.put(arena, "cachedContent", str(resource));
    } else if (fields.system) |system| {
        var row: Object = .empty;
        const parts = switch (system) {
            .text => |bytes| try singleton(arena, try textPart(arena, bytes)),
            .parts => |parts| try joinedParts(arena, null, parts),
        };
        try row.put(arena, "parts", parts);
        try body.put(arena, "systemInstruction", .{ .object = row });
    }
    var generation: Object = .empty;
    inline for (.{ .{ "temperature", "temperature" }, .{ "top_p", "topP" }, .{ "frequency_penalty", "frequencyPenalty" }, .{ "presence_penalty", "presencePenalty" } }) |pair| {
        if (@field(config, pair[0])) |num| {
            const normalized: Value = if (num == @trunc(num) and num >= -0x1p63 and num < 0x1p63)
                .{ .integer = @intFromFloat(num) }
            else
                .{ .float = num };
            try generation.put(arena, pair[1], normalized);
        }
    }
    inline for (.{ .{ "max_tokens", "maxOutputTokens" }, .{ "top_k", "topK" }, .{ "seed", "seed" } }) |pair| {
        const value = @field(config, pair[0]);
        if (value != .null) try generation.put(arena, pair[1], value);
    }
    if (config.stop.len != 0) {
        var values: runtime.Array = .init(arena);
        try values.ensureTotalCapacity(config.stop.len);
        for (config.stop) |stop| values.appendAssumeCapacity(str(stop));
        try generation.put(arena, "stopSequences", .{ .array = values });
    }
    if (config.logprobs != .null) {
        try generation.put(arena, "responseLogprobs", .{ .bool = true });
        const order = @import("value.zig").numericOrder(config.logprobs, .{ .integer = 0 }) catch
            return error.LMFailure;
        if (order == .gt) try generation.put(arena, "logprobs", config.logprobs);
    }
    if (config.response_format) |format| {
        const value = format.toValue(arena) catch |err| return codecError(err);
        try generation.put(arena, "responseMimeType", str("application/json"));
        const kind = text(get(value, "type"));
        if (std.mem.eql(u8, kind, "json_schema")) {
            var schema = get(value, "schema");
            const found = try @import("typesafe.zig").responseJudgments(arena, schema);
            try noteJudgmentProbabilities(arena, config, policy, records, found);
            try encodeJudgmentSchema(arena, &schema, found);
            const field = if (try openApi(arena, schema)) "responseSchema" else "responseJsonSchema";
            try generation.put(arena, field, schema);
        } else if (!std.mem.eql(u8, kind, "json_object")) return error.LMFailure;
    }
    if (config.reasoning) |reasoning| {
        var effective = reasoning;
        // Typed DSPy prediction consumes the provider's visible thought summary.
        if (mode == .prediction and reasoning.effort != .off and reasoning.summary == null)
            effective.summary = .auto;
        const thinking = try thinkingConfig(arena, fields.model, &effective, policy, records);
        try generation.put(arena, "thinkingConfig", thinking);
    }
    const output = text(get(extensions, "output"));
    const requested_modality: ?[]const u8 = if (std.mem.eql(u8, output, "image"))
        "IMAGE"
    else if (std.mem.eql(u8, output, "audio"))
        "AUDIO"
    else
        null;
    if (requested_modality) |name| {
        try generation.put(arena, "responseModalities", try singleton(arena, str(name)));
    }
    if (generation.count() != 0) try body.put(arena, "generationConfig", .{ .object = generation });
    if (cache.resource == null) try addTools(arena, &body, fields, policy, records);
    try applyExtensions(arena, &body, config, &extensions.object, policy, records);
    var writer: std.Io.Writer.Allocating = .init(arena);
    @import("pyjson.zig").writeJson(&writer.writer, .{ .object = body }, .minified) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
    return writer.written();
}

fn applyExtensions(
    arena: Allocator,
    body: *Object,
    config: *const sdk.Config,
    extensions: *const Object,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error!void {
    if (config.store) |store| try body.put(arena, "store", .{ .bool = store });
    if (config.service_tier) |tier| try body.put(arena, "serviceTier", str(tier));
    if (config.user_id) |user| try adaptString(
        arena,
        policy,
        records,
        "config.user_id",
        .dropped,
        "GenerateContent has no end-user attribution field (OpenAI and Anthropic carry it)",
        user,
        null,
    );
    var entries = extensions.iterator();
    while (entries.next()) |entry| {
        const key = entry.key_ptr.*;
        if (contains(&.{ "prompt_caching", "output" }, key)) continue;
        try body.put(arena, key, entry.value_ptr.*);
    }
}

const CachePrefix = struct {
    suffix: usize = 0,
    resource: ?[]const u8 = null,
};

fn cachePrefix(
    arena: Allocator,
    fields: *const sdk.RequestFields,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error!CachePrefix {
    const config = fields.config.cache orelse return .{};
    if (config.mode == .off) return .{};
    if (config.key) |key| {
        try adaptString(arena, policy, records, "config.cache.key", .dropped, "GenerateContent has no cache affinity key; implicit caching applies, and a " ++
            "stored cache (lm.cache(prefix), cache.resource) is the explicit tier", key, null);
    }
    if (config.retention == .long) {
        try adaptString(arena, policy, records, "config.cache.retention", .dropped, "GenerateContent takes no lifetime in-request; it belongs to the stored cache " ++
            "(cache_create(..., ttl_seconds=...) / cache_update)", "long", null);
    }
    const resource = config.resource orelse return .{};
    if (fields.messages.len == 0) return error.LMFailure;
    var result: CachePrefix = .{
        .resource = if (std.mem.startsWith(u8, resource, "cachedContents/")) resource else try std.fmt.allocPrint(arena, "cachedContents/{s}", .{resource}),
    };
    if (config.prefix_until_index != .null) {
        const last = fields.messages.len - 1;
        const order = @import("value.zig").numericOrder(
            config.prefix_until_index,
            .{ .integer = @intCast(last) },
        ) catch return error.LMFailure;
        if (order != .lt) return error.LMFailure;
        if (config.prefix_until_index != .integer) return error.LMFailure;
        result.suffix = std.math.cast(usize, config.prefix_until_index.integer) orelse
            return error.LMFailure;
        result.suffix += 1;
    }
    return result;
}

fn thinkingConfig(
    arena: Allocator,
    model: []const u8,
    reasoning: *const sdk.Reasoning,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error!Value {
    const level = startsIgnoreCase(model, "gemini-3") or startsIgnoreCase(model, "models/gemini-3");
    var effort = reasoning.effort;
    if (effort == .off) {
        if (!level) return wrapped(arena, "thinkingBudget", .{ .integer = 0 });
        const reason = try std.fmt.allocPrint(arena, "{s} cannot disable thinking " ++
            "(the Gemini 3 class honours no off switch); the lowest level was sent and " ++
            "the thinking spend is visible in usage", .{model});
        try adaptString(arena, policy, records, "config.reasoning.effort", .substituted, reason, "off", "minimal");
        effort = .minimal;
    }

    var thinking: Object = .empty;
    if (reasoning.summary) |summary| {
        if (summary != .auto) {
            try adaptString(arena, policy, records, "config.reasoning.summary", .substituted, "GenerateContent has includeThoughts only, no detail levels; 'auto' shows the thoughts", @tagName(summary), "auto");
        }
        try thinking.put(arena, "includeThoughts", .{ .bool = true });
    }
    if (reasoning.thinking_budget != .null) {
        try thinking.put(arena, "thinkingBudget", reasoning.thinking_budget);
    } else {
        var selected = effort orelse return error.LMFailure;
        if (level) {
            if (selected == .xhigh or selected == .max) {
                try adaptString(arena, policy, records, "config.reasoning.effort", .clamped, "the Gemini 3 class has thinkingLevel minimal|low|medium|high; 'high' is the ceiling", @tagName(selected), "high");
                selected = .high;
            }
            try thinking.put(arena, "thinkingLevel", str(@tagName(selected)));
        } else {
            const budget: i64 = switch (selected) {
                .off => unreachable,
                .minimal => 1024,
                .low => 2048,
                .medium => 8192,
                .high => 16384,
                .xhigh => 24576,
                .max => 32768,
            };
            try thinking.put(arena, "thinkingBudget", .{ .integer = budget });
        }
    }
    return .{ .object = thinking };
}

const Judgment = @import("typesafe.zig").Judgment;

fn noteJudgmentProbabilities(
    arena: Allocator,
    config: *const sdk.Config,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
    found: []const Judgment,
) Error!void {
    for (found) |judgment| {
        if (judgment.kind == .box) continue;
        return switch (config.probabilities orelse .off) {
            .off => {},
            .required => error.LMFailure,
            .if_available => adaptString(
                arena,
                policy,
                records,
                "config.probabilities",
                .dropped,
                "this wire cannot measure a distribution over the declared keys; the answer carries the pick only",
                "if_available",
                null,
            ),
        };
    }
}

fn encodeJudgmentSchema(arena: Allocator, schema: *Value, found: []const Judgment) Error!void {
    if (found.len == 0) return;
    const properties = &schema.object.getPtr("properties").?.object;
    for (found) |judgment| {
        if (judgment.kind == .boolean or judgment.kind == .box) continue;
        const property = &properties.getPtr(judgment.name).?.object;
        const value = property.get("anyOf") orelse continue;
        const old_enum = property.get("enum") orelse .null;
        // A recognized explicit enum can coexist with a non-const anyOf.
        // Only const-branch judgments carry per-key labels.
        const branches: []const Value = if (old_enum == .array) &.{} else value.array.items;
        const values = if (old_enum == .array) old_enum.array else from_branches: {
            std.debug.assert(branches.len == judgment.keys.len);
            var items: runtime.Array = .init(arena);
            try items.ensureTotalCapacity(branches.len);
            for (branches) |branch| items.appendAssumeCapacity(get(branch, "const"));
            break :from_branches items;
        };
        _ = property.orderedRemove("anyOf");
        const ordered = judgment.kind == .ordered;
        try property.put(arena, "type", str(if (ordered) "integer" else "string"));
        try property.put(arena, "enum", .{ .array = values });

        if (try judgmentDescription(arena, property.get("description") orelse .null, judgment, branches)) |description|
            try property.put(arena, "description", str(description));
    }
}

fn judgmentDescription(
    arena: Allocator,
    existing: Value,
    judgment: Judgment,
    branches: []const Value,
) Error!?[]const u8 {
    for (branches) |branch| {
        if (text(get(branch, "title")).len != 0) break;
        if (text(get(branch, "description")).len != 0) break;
    } else return null;

    var out: std.Io.Writer.Allocating = .init(arena);
    const head = text(existing);
    if (head.len != 0) {
        out.writer.writeAll(head) catch return error.OutOfMemory;
        out.writer.writeByte(' ') catch return error.OutOfMemory;
    }
    const heading = if (judgment.kind == .ordered) "Levels: " else "Options: ";
    out.writer.writeAll(heading) catch return error.OutOfMemory;
    for (branches, judgment.keys, 0..) |branch, key, index| {
        if (index != 0) out.writer.writeAll("; ") catch return error.OutOfMemory;
        out.writer.writeAll(key) catch return error.OutOfMemory;
        const title = get(branch, "title");
        const description = get(branch, "description");
        if (title != .string and description != .string) continue;
        const label = text(title);
        const details = text(description);
        out.writer.writeAll(" = ") catch return error.OutOfMemory;
        out.writer.writeAll(label) catch return error.OutOfMemory;
        if (label.len != 0 and details.len != 0)
            out.writer.writeAll(": ") catch return error.OutOfMemory;
        out.writer.writeAll(details) catch return error.OutOfMemory;
    }
    return @import("value.zig").pyStrip(out.written());
}

fn startsIgnoreCase(value: []const u8, prefix: []const u8) bool {
    return value.len >= prefix.len and std.ascii.eqlIgnoreCase(value[0..prefix.len], prefix);
}

fn adaptString(
    arena: Allocator,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
    field: []const u8,
    action: sdk.AdaptationAction,
    reason: []const u8,
    asked: []const u8,
    applied: ?[]const u8,
) Error!void {
    switch (policy) {
        .silent => {},
        .refuse => return error.LMFailure,
        .note => try records.append(arena, .{
            .field = field,
            .action = action,
            .reason = reason,
            .asked = sdk.PayloadValue.fromBorrowedValue(arena, str(asked)) catch |err|
                return codecError(err),
            .applied = if (applied) |bytes|
                sdk.PayloadValue.fromBorrowedValue(arena, str(bytes)) catch |err| return codecError(err)
            else
                .{},
        }),
    }
}

fn encodeSignature(
    arena: Allocator,
    row: *Value,
    states: []const sdk.Continuation,
    thinking: bool,
) Error!void {
    for (states) |state| {
        if (!std.mem.eql(u8, state.provider, "gemini")) continue;
        if (!std.mem.eql(u8, state.kind, "thought_signature")) continue;
        const data = state.data.toValue(arena) catch |err| return codecError(err);
        const signature = text(get(data, "value"));
        if (signature.len != 0) {
            if (thinking) try row.object.put(arena, "thought", .{ .bool = true });
            try row.object.put(arena, "thoughtSignature", str(signature));
        }
        return;
    }
}

fn parseSignature(arena: Allocator, value: Value) Error![]const sdk.Continuation {
    if (value == .null) return &.{};
    if (value != .string) return error.LMFailure;
    const data = try wrapped(arena, "value", value);
    const states = try arena.alloc(sdk.Continuation, 1);
    states[0] = .{
        .provider = "gemini",
        .kind = "thought_signature",
        .data = sdk.PayloadValue.fromBorrowedValue(arena, data) catch |err| return codecError(err),
    };
    return states;
}

fn encodePart(arena: Allocator, part: sdk.Part, names: *const std.StringHashMapUnmanaged([]const u8)) Error!Value {
    switch (part) {
        .data => |value| return textPart(arena, try content.dataText(arena, value.value)),
        .text => |value| {
            var row = try textPart(arena, value.text);
            try encodeSignature(arena, &row, value.continuation, false);
            return row;
        },
        .thinking => |value| {
            var row = try textPart(arena, value.text);
            try encodeSignature(arena, &row, value.continuation, true);
            return row;
        },
        .tool_call => |call| {
            var row: Object = .empty;
            try row.put(arena, "name", str(call.name));
            try row.put(arena, "args", call.input.toValue(arena) catch |err| return codecError(err));
            try row.put(arena, "id", str(call.id));
            var encoded = try wrapped(arena, "functionCall", .{ .object = row });
            try encodeSignature(arena, &encoded, call.continuation, false);
            return encoded;
        },
        .image, .audio, .video, .document, .binary => |media| return encodeMedia(arena, &media),
        .tool_result => |result| return encodeToolResult(arena, &result, names),
        else => return error.LMFailure,
    }
}

fn encodeMedia(arena: Allocator, media: *const sdk.MediaPart) Error!Value {
    if (media.path != null or media.continuation.len != 0) return error.LMFailure;
    const mime = if (media.media_type.len != 0) media.media_type else "application/octet-stream";
    var row: Object = .empty;
    try row.put(arena, "mimeType", str(mime));
    if (media.url orelse media.file_id) |uri| {
        try row.put(arena, "fileUri", str(uri));
        return wrapped(arena, "fileData", .{ .object = row });
    }
    try row.put(arena, "data", str(media.data orelse return error.LMFailure));
    return wrapped(arena, "inlineData", .{ .object = row });
}

fn encodeToolResult(
    arena: Allocator,
    result: *const sdk.ToolResultPart,
    names: *const std.StringHashMapUnmanaged([]const u8),
) Error!Value {
    if (result.continuation.len != 0) return error.LMFailure;
    const name = result.name orelse names.get(result.id) orelse return error.LMFailure;
    var attachments: runtime.Array = .init(arena);
    var body: std.Io.Writer.Allocating = .init(arena);
    var has_text = false;
    for (result.content) |part| switch (part) {
        .image, .document => |media| try attachments.append(try encodeMedia(arena, &media)),
        .text => |value| {
            if (value.continuation.len != 0) return error.LMFailure;
            if (has_text) body.writer.writeByte('\n') catch return error.OutOfMemory;
            body.writer.writeAll(value.text) catch return error.OutOfMemory;
            has_text = true;
        },
        else => return error.LMFailure,
    };
    const response = if (result.is_error)
        try wrapped(arena, "error", str(body.written()))
    else if (has_text or attachments.items.len == 0)
        try wrapped(arena, "result", str(body.written()))
    else
        Value{ .object = .empty };
    var row: Object = .empty;
    try row.put(arena, "name", str(name));
    try row.put(arena, "response", response);
    if (result.id.len != 0) try row.put(arena, "id", str(result.id));
    if (attachments.items.len != 0) try row.put(arena, "parts", .{ .array = attachments });
    return wrapped(arena, "functionResponse", .{ .object = row });
}

fn addTools(arena: Allocator, body: *Object, fields: *const sdk.RequestFields, policy: @FieldType(Gemini, "adaptations"), records: *std.ArrayList(sdk.Adaptation)) Error!void {
    const declarations = try functionDeclarations(arena, fields.tools);
    var tools: runtime.Array = .init(arena);
    if (declarations.items.len != 0) try tools.append(try wrapped(arena, "functionDeclarations", .{ .array = declarations }));
    for (fields.tools) |tool| if (tool == .builtin) {
        const builtin = tool.builtin;
        const key = if (std.mem.eql(u8, builtin.name, "web_search")) "googleSearch" else if (std.mem.eql(u8, builtin.name, "code_execution")) "codeExecution" else builtin.name;
        const config: Value = if (builtin.config) |value|
            value.toValue(arena) catch |err| return codecError(err)
        else
            .{ .object = .empty };
        try tools.append(try wrapped(arena, key, config));
    };
    if (tools.items.len != 0) try body.put(arena, "tools", .{ .array = tools });
    const choice = fields.config.tool_choice orelse return;
    if (choice.parallel == false) switch (policy) {
        .refuse => return error.LMFailure,
        .silent => {},
        .note => try records.append(arena, .{
            .field = "config.tool_choice.parallel",
            .action = .dropped,
            .reason = "GenerateContent has no parallel-tool-calls knob and may return several calls (OpenAI and Anthropic carry it)",
            .asked = sdk.PayloadValue.fromBorrowedValue(arena, .{ .bool = false }) catch |err| return codecError(err),
        }),
    };
    var row: Object = .empty;
    try row.put(arena, "mode", str(switch (choice.mode) {
        .none => "NONE",
        .required => "ANY",
        .auto => if (choice.allowed.len == 0) "AUTO" else "VALIDATED",
    }));
    if (choice.allowed.len != 0) {
        for (fields.tools) |tool|
            if (tool == .builtin and contains(choice.allowed, tool.builtin.name)) return error.LMFailure;
        var allowed: runtime.Array = .init(arena);
        try allowed.ensureTotalCapacity(choice.allowed.len);
        for (choice.allowed) |name| allowed.appendAssumeCapacity(str(name));
        try row.put(arena, "allowedFunctionNames", .{ .array = allowed });
    }
    try body.put(arena, "toolConfig", try wrapped(arena, "functionCallingConfig", .{ .object = row }));
}

fn functionDeclarations(arena: Allocator, tools: []const sdk.RequestTool) Error!runtime.Array {
    var declarations: runtime.Array = .init(arena);
    for (tools) |tool| {
        if (tool != .function) continue;
        const function = tool.function;
        const schema: Value = if (function.parameters) |params|
            params.toValue(arena) catch |err| return codecError(err)
        else
            .{ .object = .empty };
        var row: Object = .empty;
        try row.put(arena, "name", str(function.name));
        if (function.description) |description| if (description.len != 0)
            try row.put(arena, "description", str(description));
        const schema_key = if (try openApi(arena, schema)) "parameters" else "parametersJsonSchema";
        try row.put(arena, schema_key, schema);
        try declarations.append(.{ .object = row });
    }
    return declarations;
}

fn wrapped(arena: Allocator, key: []const u8, value: Value) Error!Value {
    var object: Object = .empty;
    try object.put(arena, key, value);
    return .{ .object = object };
}

fn openApi(arena: Allocator, root: Value) Error!bool {
    var pending: std.ArrayList(Value) = .empty;
    defer pending.deinit(arena);
    try pending.append(arena, root);
    while (pending.pop()) |node| {
        if (node == .bool) return false;
        if (node != .object) continue;
        for (node.object.keys(), node.object.values()) |key, value| {
            if (!contains(&schema_fields, key)) return false;
            if (std.mem.eql(u8, key, "type") and value == .array) return false;
            if (std.mem.eql(u8, key, "enum") and value == .array) {
                for (value.array.items) |item| if (item != .string) return false;
            } else if (std.mem.eql(u8, key, "properties") and value == .object) {
                try pending.appendSlice(arena, value.object.values());
            } else if (std.mem.eql(u8, key, "anyOf") and value == .array) {
                try pending.appendSlice(arena, value.array.items);
            } else if (contains(&.{ "items", "anyOf" }, key)) try pending.append(arena, value);
        }
    }
    return true;
}

const schema_fields = [_][]const u8{
    "type",     "format",        "title",         "description", "nullable",  "enum",    "maxItems", "minItems", "properties",
    "required", "minProperties", "maxProperties", "minLength",   "maxLength", "pattern", "example",  "anyOf",    "propertyOrdering",
    "default",  "items",         "minimum",       "maximum",
};

fn parseResponse(arena: Allocator, model: []const u8, raw: Value, records: []const sdk.Adaptation) Error!lm_mod.Completion {
    if (raw != .object or get(raw, "error") != .null) return error.LMFailure;
    const block = text(get(get(raw, "promptFeedback"), "blockReason"));
    if (block.len != 0 and !std.mem.eql(u8, block, "BLOCK_REASON_UNSPECIFIED")) return error.LMFailure;
    const candidates = elems(get(raw, "candidates"));
    const candidate = if (candidates.len == 0) .null else candidates[0];
    const reason = text(get(candidate, "finishReason"));
    if (invalidFinish(reason)) return error.LMFailure;
    const grounding = get(candidate, "groundingMetadata");
    const wire_parts = elems(get(get(candidate, "content"), "parts"));
    const parts = try arena.alloc(sdk.Part, @max(wire_parts.len, 1));
    var visible: std.Io.Writer.Allocating = .init(arena);
    var reasoning: std.Io.Writer.Allocating = .init(arena);
    var calls: std.ArrayList(lm_mod.ToolCall) = .empty;
    var parts_len: usize = 0;
    for (wire_parts) |part| {
        const dest = &parts[parts_len];
        const signature = get(part, "thoughtSignature");
        const value = get(part, "text");
        const thought = get(part, "thought");
        if (value == .string) {
            const states = try parseSignature(arena, signature);
            if (thought == .bool and thought.bool) {
                dest.* = .{ .thinking = .{ .text = value.string, .continuation = states } };
                reasoning.writer.writeAll(value.string) catch return error.OutOfMemory;
            } else {
                dest.* = .{ .text = .{ .text = value.string, .continuation = states } };
                visible.writer.writeAll(value.string) catch return error.OutOfMemory;
            }
            parts_len += 1;
            continue;
        }
        const inline_data = get(part, "inlineData");
        const file_data = get(part, "fileData");
        if (inline_data == .object or file_data == .object) {
            const media = if (inline_data == .object)
                parseMedia(inline_data, .inline_bytes)
            else
                parseMedia(file_data, .uri);
            if (media) |value_part| {
                dest.* = value_part;
                parts_len += 1;
            }
            continue;
        }
        const call = get(part, "functionCall");
        if (call == .null and (get(part, "executableCode") != .null or
            get(part, "codeExecutionResult") != .null)) continue;
        const name = text(get(call, "name"));
        if (name.len == 0) return error.LMFailure;
        const states = try parseSignature(arena, if (text(signature).len != 0)
            signature
        else
            get(call, "thoughtSignature"));
        const wire_id = text(get(call, "id"));
        const call_id = if (wire_id.len != 0) wire_id else try std.fmt.allocPrint(arena, "tool_call_{d}", .{parts_len});
        const args = get(call, "args");
        const input: Value = if (args == .object) args else .{ .object = .empty };
        const parsed = sdk.JsonValue.fromBorrowedValue(arena, input) catch |err| return codecError(err);
        var json: std.Io.Writer.Allocating = .init(arena);
        @import("pyjson.zig").writeJson(&json.writer, input, .minified) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
        dest.* = .{ .tool_call = .{
            .name = name,
            .id = call_id,
            .input = parsed,
            .continuation = states,
        } };
        try calls.append(arena, .{
            .name = name,
            .id = call_id,
            .arguments = json.written(),
            .continuation = states,
        });
        parts_len += 1;
    }
    const citations = try readCitations(arena, grounding, visible.written());
    const final_parts = if (citations.len != 0) with_citations: {
        const combined = try arena.realloc(parts, parts_len + citations.len);
        for (combined[parts_len..], citations) |*slot, citation| slot.* = .{ .citation = citation };
        break :with_citations combined;
    } else without_citations: {
        if (parts_len == 0) {
            parts[0] = .{ .text = .{} };
            parts_len = 1;
        }
        break :without_citations parts[0..parts_len];
    };
    const id = text(get(raw, "responseId"));
    const scores = try readLogprobs(arena, get(candidate, "logprobsResult"));
    return .{
        .logprobs = if (scores.len == 0) null else scores,
        .text = if (visible.written().len == 0 and calls.items.len != 0) null else visible.written(),
        .reasoning = if (reasoning.written().len != 0) reasoning.written() else null,
        .tool_calls = calls.items,
        .adaptations = records,
        .citations = citations,
        .message = .{ .role = .assistant, .parts = final_parts },
        .provider_data = sdk.PayloadValue.fromBorrowedValue(arena, raw) catch |err| return codecError(err),
        .metadata = .{
            .id = if (id.len != 0) id else null,
            .model = model,
            .finish_reason = finishReason(reason, calls.items.len != 0),
            .usage = try readUsage(arena, get(raw, "usageMetadata")),
        },
    };
}

const CitationKey = [3][]const u8;

const CitationContext = struct {
    pub fn hash(_: CitationContext, key: CitationKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (key) |item| {
            const len = item.len;
            hasher.update(std.mem.asBytes(&len));
            hasher.update(item);
        }
        return hasher.final();
    }

    pub fn eql(_: CitationContext, lhs: CitationKey, rhs: CitationKey) bool {
        for (lhs, rhs) |a, b| {
            if (!std.mem.eql(u8, a, b)) return false;
        }
        return true;
    }
};

fn readCitations(arena: Allocator, grounding: Value, answer: []const u8) Error![]const sdk.CitationPart {
    const chunks = elems(get(grounding, "groundingChunks"));
    const supports = elems(get(grounding, "groundingSupports"));
    var seen: std.HashMapUnmanaged(CitationKey, void, CitationContext, 80) = .empty;
    defer seen.deinit(arena);
    var citations: std.ArrayList(sdk.CitationPart) = .empty;
    for (supports) |support| {
        const segment = get(support, "segment");
        const cited = nonempty(get(segment, "text")) orelse try segmentText(segment, answer);
        for (elems(get(support, "groundingChunkIndices"))) |raw_index| {
            var source: Value = .null;
            if (try lm_mod.OpenAI.annotationIndex(raw_index, chunks.len)) |index| {
                if (index < chunks.len) {
                    inline for (.{ "web", "retrievedContext", "googleSearch" }) |name| {
                        const entry = get(chunks[index], name);
                        if (source == .null and try lm_mod.OpenAI.responseTruthy(entry)) source = entry;
                    }
                }
            }
            const url = try citationLabel(arena, source, "uri", "url");
            const title = try citationLabel(arena, source, "title", "name");
            if (url == null and title == null and cited == null) continue;
            const entry = try seen.getOrPut(arena, .{ url orelse "", title orelse "", cited orelse "" });
            if (entry.found_existing) continue;
            try citations.append(arena, .{ .text = cited, .title = title, .url = url });
        }
    }
    return citations.items;
}

fn citationLabel(arena: Allocator, source: Value, primary: []const u8, fallback: []const u8) Error!?[]const u8 {
    const field = get(source, primary);
    const value = if (try lm_mod.OpenAI.responseTruthy(field)) field else get(source, fallback);
    if (!try lm_mod.OpenAI.responseTruthy(value)) return null;
    if (value == .string) return value.string;
    var writer: std.Io.Writer.Allocating = .init(arena);
    @import("pyjson.zig").writeStr(&writer.writer, value) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
    return writer.written();
}

fn segmentText(segment: Value, answer: []const u8) Error!?[]const u8 {
    const common = lm_mod.OpenAI;
    const start = try common.annotationIndex(get(segment, "startIndex"), answer.len) orelse return null;
    const end = try common.annotationIndex(get(segment, "endIndex"), answer.len) orelse return null;
    if (start >= end) return null;
    const first = common.codepointOffset(answer, start) orelse return null;
    const last = common.codepointOffset(answer[first..], end - start) orelse return null;
    return answer[first..][0..last];
}

fn readLogprobs(arena: Allocator, value: Value) Error![]const lm_mod.TokenLogprob {
    const chosen = elems(get(value, "chosenCandidates"));
    const alternatives = elems(get(value, "topCandidates"));
    var scores: std.ArrayList(lm_mod.TokenLogprob) = .empty;
    for (chosen, 0..) |item, index| {
        if (item != .object) continue;
        var top: std.ArrayList(lm_mod.TopLogprob) = .empty;
        if (index < alternatives.len) {
            for (elems(get(alternatives[index], "candidates"))) |alt|
                if (alt == .object) try top.append(arena, try readScore(arena, alt));
        }
        const token = try readScore(arena, item);
        try scores.append(arena, .{
            .token = token.token,
            .token_id = token.token_id,
            .logprob = token.logprob,
            .top = top.items,
        });
    }
    return scores.items;
}

fn readScore(arena: Allocator, item: Value) Error!lm_mod.TopLogprob {
    const numbers = @import("value.zig");
    const raw = get(item, "logProbability");
    const truthy = lm_mod.OpenAI.responseTruthy;
    const score: f64 = if (!(truthy(raw) catch return error.LMFailure))
        0
    else if (raw == .string)
        numbers.floatFromString(arena, raw.string) catch |err| return codecError(err)
    else
        numbers.numericFloat(raw) catch return error.LMFailure;
    const token = get(item, "token");
    const bytes = if (!(truthy(token) catch return error.LMFailure)) "" else if (token == .string)
        token.string
    else formatted: {
        var writer: std.Io.Writer.Allocating = .init(arena);
        @import("pyjson.zig").writeStr(&writer.writer, token) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
        break :formatted writer.written();
    };
    return .{
        .token = bytes,
        .token_id = @import("logprobs.zig").normalizeBorrowedId(arena, get(item, "tokenId")) catch |err|
            return codecError(err),
        .logprob = score,
    };
}

fn parseMedia(value: Value, mode: enum { inline_bytes, uri }) ?sdk.Part {
    const address = text(get(value, if (mode == .inline_bytes) "data" else "fileUri"));
    if (address.len == 0) return null;
    const mime = text(get(value, "mimeType"));
    const media: sdk.MediaPart = .{
        .media_type = if (mime.len == 0) "application/octet-stream" else mime,
        .data = if (mode == .inline_bytes) address else null,
        .url = if (mode == .uri) address else null,
    };
    if (std.mem.startsWith(u8, mime, "image/")) return .{ .image = media };
    if (std.mem.startsWith(u8, mime, "audio/")) return .{ .audio = media };
    return .{ .document = media };
}

fn readUsage(arena: Allocator, usage: Value) Error!sdk.Usage {
    if (usage != .object or usage.object.count() == 0) return .{};
    const common = lm_mod.OpenAI;
    const answer_key = if (usage.object.contains("candidatesTokenCount")) "candidatesTokenCount" else "responseTokenCount";
    const thoughts = try count(arena, usage, "thoughtsTokenCount");
    const prompt = try modality(arena, get(usage, "promptTokensDetails"));
    const tools = try modality(arena, get(usage, "toolUsePromptTokensDetails"));
    const output_details = elems(get(usage, "candidatesTokensDetails"));
    return common.settleUsage(arena, .{
        .input_tokens = try common.countSum(arena, try count(arena, usage, "promptTokenCount"), try count(arena, usage, "toolUsePromptTokenCount")),
        .output_tokens = try common.countSum(arena, try count(arena, usage, answer_key), thoughts),
        .total_tokens = try common.wireCount(arena, get(usage, "totalTokenCount")),
        .cache_read_tokens = try count(arena, usage, "cachedContentTokenCount"),
        .reasoning_tokens = thoughts,
        .input_audio_tokens = if (prompt == .null) tools else if (tools == .null) prompt else try common.countSum(arena, prompt, tools),
        .output_audio_tokens = try modality(arena, if (output_details.len == 0) get(usage, "responseTokensDetails") else get(usage, "candidatesTokensDetails")),
    });
}

fn count(arena: Allocator, object: Value, key: []const u8) Error!Value {
    const value = try lm_mod.OpenAI.wireCount(arena, get(object, key));
    return if (value == .null) .{ .integer = 0 } else value;
}

fn modality(arena: Allocator, details: Value) Error!Value {
    var total: Value = .null;
    for (elems(details)) |row| {
        if (!std.mem.eql(u8, text(get(row, "modality")), "AUDIO")) continue;
        const tokens = try count(arena, row, "tokenCount");
        total = if (total == .null) tokens else try lm_mod.OpenAI.countSum(arena, total, tokens);
    }
    return total;
}

fn joinedParts(arena: Allocator, leading_line: ?[]const u8, parts: anytype) Error!Value {
    const rendered = try content.partsText(arena, parts, leading_line);
    return singleton(arena, try textPart(arena, rendered));
}

fn singleton(arena: Allocator, part: Value) Error!Value {
    var list: runtime.Array = .init(arena);
    try list.append(part);
    return .{ .array = list };
}

fn textPart(arena: Allocator, bytes: []const u8) Error!Value {
    var part: Object = .empty;
    try part.put(arena, "text", str(bytes));
    return .{ .object = part };
}

fn str(bytes: []const u8) Value {
    return .{ .string = bytes };
}
fn text(value: Value) []const u8 {
    return if (value == .string) value.string else "";
}
fn get(value: Value, key: []const u8) Value {
    return if (value == .object) value.object.get(key) orelse .null else .null;
}
fn elems(value: Value) []const Value {
    return if (value == .array) value.array.items else &.{};
}
fn contains(values: []const []const u8, needle: []const u8) bool {
    for (values) |value| if (std.mem.eql(u8, value, needle)) return true;
    return false;
}
fn codecError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
}

test "Gemini separates structured schema dialects and rejects blocked candidates" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const schema = try parse(arena,
        \\{"type":"object","properties":{"answer":{"anyOf":[{"type":"string"},false]}}}
    );
    try std.testing.expect(!try openApi(arena, schema));
    const ordinary = try parse(arena,
        \\{"type":"object","properties":{"answer":{"type":"string","default":{"x":false}}}}
    );
    try std.testing.expect(try openApi(arena, ordinary));
    const blocked = try parse(arena,
        \\{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}
    );
    try std.testing.expectError(error.LMFailure, parseResponse(arena, "fixture", blocked, &.{}));
}

test "Gemini judgment enum takes precedence over non-const branches" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var schema = try @import("json_repair.zig").parseJson(arena,
        \\{"type":"object","properties":{"grade":{
        \\"enum":["ok","no"],"anyOf":[{"type":"string"}]}}}
    );
    const found = try @import("typesafe.zig").responseJudgments(arena, schema);
    try encodeJudgmentSchema(arena, &schema, found);
    const grade = get(get(schema, "properties"), "grade");
    try std.testing.expectEqualStrings("string", text(get(grade, "type")));
    try std.testing.expect(get(grade, "anyOf") == .null);
    try std.testing.expectEqualStrings("ok", text(elems(get(grade, "enum"))[0]));
}

test "Gemini extensions override top-level wire fields without leaking control keys" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena,
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"Answer"}]}],
        \\"config":{"user_id":"actor","max_tokens":24,"extensions":{"output":"audio",
        \\"prompt_caching":true,"generationConfig":{"responseModalities":["TEXT"]}}}}
    ));
    defer request.release();

    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try parse(arena, try requestBody(arena, &request.fields, .canonical, .note, &records));
    const generation = get(body, "generationConfig");
    try std.testing.expectEqualStrings("TEXT", text(elems(get(generation, "responseModalities"))[0]));
    try std.testing.expectEqual(@as(usize, 1), generation.object.count());
    try std.testing.expect(!body.object.contains("output"));
    try std.testing.expect(!body.object.contains("prompt_caching"));
    try std.testing.expectEqualStrings("config.user_id", records.items[0].field);
    try std.testing.expectError(
        error.LMFailure,
        requestBody(arena, &request.fields, .canonical, .refuse, &records),
    );
}

test "Gemini retains tool call position in mixed responses" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[{"text":"Looking up"},{"functionCall":{"name":"weather","args":{"city":"Paris"}}}]},"finishReason":"MAX_TOKENS"}]}
    );
    const result = try parseResponse(arena, "fixture", raw, &.{});
    try std.testing.expectEqual(sdk.FinishReason.tool_call, result.metadata.finish_reason.?);
    try std.testing.expectEqualStrings("tool_call_1", result.message.?.parts[1].tool_call.id);
    try std.testing.expectEqualStrings("tool_call_1", result.tool_calls[0].id);
}

test "Gemini separates answer text from signed and unsigned thought summaries" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[
        \\ {"thought":true,"text":"","thoughtSignature":"opaque"},
        \\ {"thought":true,"text":"Check "},{"thought":true,"text":"the answer"},
        \\ {"text":"Paris","thoughtSignature":"answer-state"}
        \\]}}]}
    );
    const result = try parseResponse(arena, "gemini-3-flash", raw, &.{});
    try std.testing.expectEqualStrings("Check the answer", result.reasoning.?);
    try std.testing.expectEqualStrings("Paris", result.text.?);

    const parts = result.message.?.parts;
    const names: std.StringHashMapUnmanaged([]const u8) = .empty;
    const signed = try encodePart(arena, parts[0], &names);
    const unsigned = try encodePart(arena, parts[1], &names);
    const answer = try encodePart(arena, parts[3], &names);
    try std.testing.expect(get(signed, "thought").bool);
    try std.testing.expectEqualStrings("opaque", text(get(signed, "thoughtSignature")));
    try std.testing.expect(get(unsigned, "thought") == .null);
    try std.testing.expectEqualStrings("answer-state", text(get(answer, "thoughtSignature")));
}

test "Gemini ignores empty media before assigning a synthetic tool ID" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[
        \\ {"inlineData":{"mimeType":"image/png","data":""}},
        \\ {"fileData":{"mimeType":"audio/wav","fileUri":"gs://bucket/audio"}},
        \\ {"functionCall":{"name":"lookup","args":{}}}
        \\]}}]}
    );
    const result = try parseResponse(arena, "fixture", raw, &.{});
    const parts = result.message.?.parts;
    try std.testing.expectEqualStrings("gs://bucket/audio", parts[0].audio.url.?);
    try std.testing.expectEqualStrings("tool_call_1", parts[1].tool_call.id);
}

test "Gemini Live clips a truncated WAV data chunk to the available samples" {
    const header = "RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00" ++
        "\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00";
    const raw = header ++ "data\xff\xff\xff\xff\x01\x02";
    const pcm = liveWave(raw);
    try std.testing.expectEqual(22050, pcm.rate);
    try std.testing.expectEqualStrings("\x01\x02", pcm.bytes);
}

test "Gemini SSE preserves diagnostic precedence and emits no successful terminal event" {
    const Capture = struct {
        detail: ?StreamError = null,
        event_count: usize = 0,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.event_count += 1;
            if (event == .provider_error) self.detail = event.provider_error;
            return .proceed;
        }
    };
    const cases = [_]struct { frame: []const u8, expected: StreamError }{
        .{
            .frame = "{\"error\":{\"status\":\"RESOURCE_EXHAUSTED\",\"message\":\"Token LIMIT\"}}",
            .expected = .{
                .code = .context_length,
                .provider_code = "RESOURCE_EXHAUSTED",
                .message = "Token LIMIT",
            },
        },
        .{
            .frame = "{\"error\":null,\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}",
            .expected = .{
                .code = .provider,
                .provider_code = "provider",
                .message = "provider",
            },
        },
        .{
            .frame = "{\"candidates\":[{\"finishReason\":\"SAFETY\",\"finishMessage\":\"Blocked\"}]}",
            .expected = .{
                .code = .invalid_request,
                .provider_code = "inband_finish_reason",
                .message = "Blocked (gemini)",
            },
        },
    };
    for (cases) |case| {
        var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer scratch.deinit();
        const arena = scratch.allocator();
        var capture: Capture = .{};
        var state: GenerateStream = .{
            .model = "fixture",
            .adaptations = &.{},
            .sink = .{ .context = &capture, .on_event = Capture.onEvent },
        };
        const frame = try @import("json_repair.zig").parseJson(arena, case.frame);
        try std.testing.expect(try state.dispatch(arena, frame));
        try std.testing.expectEqual(1, capture.event_count);
        try std.testing.expectEqualDeep(case.expected, capture.detail.?);
    }
}

test "Gemini SSE joins split text without merging parallel tool calls" {
    const Capture = struct {
        indexes: [4]u8 = undefined,
        used: usize = 0,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (event == .delta) {
                const index = event.delta.part_index.?;
                if (self.used == self.indexes.len or index.len != 1) return .stop;
                self.indexes[self.used] = index[0];
                self.used += 1;
            }
            return .proceed;
        }
    };
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    var capture: Capture = .{};
    var state: GenerateStream = .{
        .model = "gemini-3-flash",
        .adaptations = &.{},
        .sink = .{ .context = &capture, .on_event = Capture.onEvent },
    };
    const frames =
        \\data: {"candidates":[{"content":{"parts":[{"text":"Pa"}]}}]}
        \\
        \\data: {"candidates":[{"content":{"parts":[{"text":"ris"}]}}]}
        \\
        \\data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"a"}}]}}]}
        \\
        \\data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"b"}}]}}]}
        \\
        \\
    ;
    var reader = std.Io.Reader.fixed(frames);
    try GenerateStream.consume(&state, scratch.allocator(), &reader, frames.len);
    try std.testing.expectEqualStrings("0012", capture.indexes[0..capture.used]);
    try std.testing.expectEqual(Gemini.StreamOutcome.exhausted, state.outcome);
}
