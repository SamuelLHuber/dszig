//! Gemini Developer API generateContent transport for text and structured text.
const std = @import("std");
const lm_mod = @import("lm.zig");
const http = @import("http.zig");
const runtime = @import("runtime_value.zig");
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
            .vtable = &.{ .complete = complete, .complete_canonical = canonical, .clone = clone, .model_name = modelName },
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

    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) Error![]const lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        for (request.tools) |tool| if (tool.strict != null) return error.LMFailure;
        const native = try lm_mod.OpenAI.canonicalPredictionRequest(arena, request, self.model, self.max_tokens, null);
        defer native.release();
        const result = try arena.alloc(lm_mod.Completion, 1);
        result[0] = try self.completeFields(arena, &native.fields, .prediction);
        return result;
    }

    pub fn completeRequest(self: *Gemini, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        return (try canonical(self, arena, request)).response.?;
    }

    fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) Error!lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        return self.completeFields(arena, &request.fields, .canonical);
    }

    fn completeFields(
        self: *const Gemini,
        arena: Allocator,
        fields: *const sdk.RequestFields,
        mode: RequestMode,
    ) Error!lm_mod.Completion {
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try requestBody(arena, fields, mode, self.adaptations, &records);
        const reply = try self.send(arena, fields.model, body, null);
        defer self.client.allocator.free(reply.body);
        const raw = @import("json_repair.zig").parseJson(arena, reply.body) catch |err|
            return codecError(err);
        return parseResponse(arena, fields.model, raw, records.items);
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
        const arena = scratch.allocator();
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const fields = &request.fields;
        const body = try requestBody(arena, fields, .canonical, self.adaptations, &records);
        var state: GenerateStream = .{
            .model = fields.model,
            .adaptations = records.items,
            .sink = sink,
        };
        var consumer: http.ResponseConsumer = .{
            .context = &state,
            .consume = GenerateStream.consume,
        };
        const reply = try self.send(arena, fields.model, body, &consumer);
        defer self.client.allocator.free(reply.body);
        return state.outcome;
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
            const temp = frame.allocator();
            const value = try @import("json_repair.zig").parseJson(temp, bytes);
            if (value != .object) continue;
            if (try self.dispatch(temp, value)) return;
            if (self.outcome == .stopped) return;
            _ = frame.reset(.retain_capacity);
        }
    }

    fn emit(self: *GenerateStream, event: Gemini.StreamEvent) void {
        if (self.outcome == .stopped) return;
        if (self.sink.on_event(self.sink.context, event) == .stop) self.outcome = .stopped;
    }

    fn start(self: *GenerateStream) void {
        if (self.started) return;
        self.started = true;
        self.emit(.{ .start = .{ .id = null, .model = self.model, .adaptations = self.adaptations } });
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
        const block = text(get(get(value, "promptFeedback"), "blockReason"));
        if (get(value, "error") != .null) {
            const detail = get(value, "error");
            const status = nonempty(get(detail, "status")) orelse "provider";
            self.emit(.{ .provider_error = .{
                .code = providerError(status),
                .provider_code = status,
                .message = nonempty(get(detail, "message")) orelse status,
            } });
            return true;
        }
        if (invalidFinish(finish) or (block.len != 0 and
            !std.mem.eql(u8, block, "BLOCK_REASON_UNSPECIFIED")))
        {
            self.emit(.{ .provider_error = .{
                .code = .invalid_request,
                .provider_code = "inband_finish_reason",
                .message = if (block.len != 0) block else finish,
            } });
            return true;
        }
        var saw_tool = false;
        var saw_delta = false;
        for (elems(get(get(candidate, "content"), "parts")), 0..) |part, index| {
            const decoded = (try decodePart(arena, part)) orelse continue;
            self.start();
            const position = self.slot(index, decoded.kind);
            var buffer: [24]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{d}", .{position}) catch unreachable;
            self.emit(.{ .delta = .{ .part_index = key, .data = decoded.data } });
            if (self.outcome == .stopped) return true;
            const continuation = try parseSignature(arena, decoded.signature);
            if (continuation.len != 0) {
                self.emit(.{ .delta = .{
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
            self.start();
            self.emit(.{ .end = .{
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

fn providerError(code: []const u8) @FieldType(@import("openai_stream.zig").ErrorDetail, "code") {
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
    if (config.cache != null or config.logprobs != .null or
        config.probabilities != null or config.user_id != null or config.extensions != null)
        return error.LMFailure;
    var names: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer names.deinit(arena);
    for (fields.messages) |message| for (message.parts) |part| {
        if (part == .tool_call) try names.put(arena, part.tool_call.id, part.tool_call.name);
    };
    var rows: runtime.Array = .init(arena);
    try rows.ensureTotalCapacity(fields.messages.len);
    for (fields.messages) |message| {
        var row: Object = .empty;
        try row.put(arena, "role", str(if (message.role == .assistant) "model" else "user"));
        const parts = if (message.role == .developer)
            try joinedParts(arena, "[developer]\n", message.parts)
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
    if (fields.system) |system| {
        var row: Object = .empty;
        const parts = switch (system) {
            .text => |bytes| try singleton(arena, try textPart(arena, bytes)),
            .parts => |parts| try joinedParts(arena, "", parts),
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
    if (config.response_format) |format| {
        const value = format.toValue(arena) catch |err| return codecError(err);
        try generation.put(arena, "responseMimeType", str("application/json"));
        const kind = text(get(value, "type"));
        if (std.mem.eql(u8, kind, "json_schema")) {
            const schema = get(value, "schema");
            try generation.put(arena, if (try openApi(arena, schema)) "responseSchema" else "responseJsonSchema", schema);
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
    if (generation.count() != 0) try body.put(arena, "generationConfig", .{ .object = generation });
    try addTools(arena, &body, fields, policy, records);
    if (config.store) |store| try body.put(arena, "store", .{ .bool = store });
    if (config.service_tier) |tier| try body.put(arena, "serviceTier", str(tier));
    var writer: std.Io.Writer.Allocating = .init(arena);
    @import("pyjson.zig").writeJson(&writer.writer, .{ .object = body }, .minified) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
    return writer.written();
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
        try reasoningAdapt(arena, policy, records, "config.reasoning.effort", .substituted, reason, "off", "minimal");
        effort = .minimal;
    }

    var thinking: Object = .empty;
    if (reasoning.summary) |summary| {
        if (summary != .auto) {
            try reasoningAdapt(arena, policy, records, "config.reasoning.summary", .substituted, "GenerateContent has includeThoughts only, no detail levels; 'auto' shows the thoughts", @tagName(summary), "auto");
        }
        try thinking.put(arena, "includeThoughts", .{ .bool = true });
    }
    if (reasoning.thinking_budget != .null) {
        try thinking.put(arena, "thinkingBudget", reasoning.thinking_budget);
    } else {
        var selected = effort orelse return error.LMFailure;
        if (level) {
            if (selected == .xhigh or selected == .max) {
                try reasoningAdapt(arena, policy, records, "config.reasoning.effort", .clamped, "the Gemini 3 class has thinkingLevel minimal|low|medium|high; 'high' is the ceiling", @tagName(selected), "high");
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

fn startsIgnoreCase(value: []const u8, prefix: []const u8) bool {
    return value.len >= prefix.len and std.ascii.eqlIgnoreCase(value[0..prefix.len], prefix);
}

fn reasoningAdapt(
    arena: Allocator,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
    field: []const u8,
    action: sdk.AdaptationAction,
    reason: []const u8,
    asked: []const u8,
    applied: []const u8,
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
            .applied = sdk.PayloadValue.fromBorrowedValue(arena, str(applied)) catch |err|
                return codecError(err),
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
    var declarations: runtime.Array = .init(arena);
    var tools: runtime.Array = .init(arena);
    for (fields.tools) |tool| switch (tool) {
        .function => |function| {
            const schema: Value = if (function.parameters) |params|
                params.toValue(arena) catch |err| return codecError(err)
            else
                .{ .object = .empty };
            var row: Object = .empty;
            try row.put(arena, "name", str(function.name));
            if (function.description) |description| if (description.len != 0)
                try row.put(arena, "description", str(description));
            try row.put(arena, if (try openApi(arena, schema)) "parameters" else "parametersJsonSchema", schema);
            try declarations.append(.{ .object = row });
        },
        .builtin => {},
    };
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
    if (get(candidate, "groundingMetadata") != .null or get(candidate, "logprobsResult") != .null) return error.LMFailure;
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
    if (parts_len == 0) {
        parts[0] = .{ .text = .{} };
        parts_len = 1;
    }
    const id = text(get(raw, "responseId"));
    var result: lm_mod.Completion = .{
        .text = if (visible.written().len == 0 and calls.items.len != 0) null else visible.written(),
        .reasoning = if (reasoning.written().len != 0) reasoning.written() else null,
        .tool_calls = calls.items,
        .adaptations = records,
        .message = .{ .role = .assistant, .parts = parts[0..parts_len] },
        .provider_data = sdk.PayloadValue.fromBorrowedValue(arena, raw) catch |err| return codecError(err),
        .metadata = .{
            .id = if (id.len != 0) id else null,
            .model = model,
            .finish_reason = finishReason(reason, calls.items.len != 0),
            .usage = try readUsage(arena, get(raw, "usageMetadata")),
        },
    };
    try lm_mod.publishResponse(arena, &result, model, .borrowed);
    return result;
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

fn joinedParts(arena: Allocator, prefix: []const u8, parts: anytype) Error!Value {
    return singleton(arena, try textPart(arena, try joinText(arena, prefix, parts)));
}

fn joinText(arena: Allocator, prefix: []const u8, parts: anytype) Error![]const u8 {
    var writer: std.Io.Writer.Allocating = .init(arena);
    writer.writer.writeAll(prefix) catch return error.OutOfMemory;
    for (parts, 0..) |part, index| {
        if (part != .text or part.text.continuation.len != 0) return error.LMFailure;
        if (index != 0) writer.writer.writeByte('\n') catch return error.OutOfMemory;
        writer.writer.writeAll(part.text.text) catch return error.OutOfMemory;
    }
    return writer.written();
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

test "Gemini retains tool call position in mixed responses" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[{"text":"Looking up"},{"functionCall":{"name":"weather","args":{"city":"Paris"}}}]},"finishReason":"MAX_TOKENS"}]}
    );
    const result = try parseResponse(arena, "fixture", raw, &.{});
    try std.testing.expectEqual(sdk.FinishReason.tool_call, result.metadata.finish_reason.?);
    try std.testing.expectEqualStrings("tool_call_1", result.response.?.fields.message.parts[1].tool_call.id);
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

    const parts = result.response.?.fields.message.parts;
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
    const parts = result.response.?.fields.message.parts;
    try std.testing.expectEqualStrings("gs://bucket/audio", parts[0].audio.url.?);
    try std.testing.expectEqualStrings("tool_call_1", parts[1].tool_call.id);
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
