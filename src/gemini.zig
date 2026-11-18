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

    pub fn lm(self: *Gemini) lm_mod.LM {
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete, .complete_canonical = canonical, .clone = clone, .model_name = modelName },
            .supports_response_format = true,
            .supports_response_schema = true,
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
        if (request.tools.len != 0) return error.LMFailure;
        const native = try lm_mod.OpenAI.canonicalPredictionRequest(arena, request, self.model, self.max_tokens, null);
        defer native.release();
        const result = try arena.alloc(lm_mod.Completion, 1);
        result[0] = try canonical(ptr, arena, native);
        return result;
    }

    pub fn completeRequest(self: *Gemini, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        return (try canonical(self, arena, request)).response.?;
    }

    fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) Error!lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const body = try requestBody(arena, &request.fields);
        const url = try endpoint(arena, self.base_url, request.fields.model);
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
        const reply = transport.send(arena, .{
            .url = url,
            .method = .POST,
            .body = body,
            .content_type = "application/json",
            .headers = headers[0..headers_len],
        }) catch |err| return codecError(err);
        defer self.client.allocator.free(reply.body);
        const raw = @import("json_repair.zig").parseJson(arena, reply.body) catch |err| return codecError(err);
        return parseResponse(arena, request.fields.model, raw);
    }
};

fn endpoint(arena: Allocator, base_url: []const u8, model: []const u8) Error![]const u8 {
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
    out.writeAll(":generateContent") catch return error.OutOfMemory;
    return writer.written();
}

fn requestBody(arena: Allocator, fields: *const sdk.RequestFields) Error![]const u8 {
    const config = &fields.config;
    if (fields.tools.len != 0 or config.tool_choice != null or config.reasoning != null or
        config.cache != null or config.logprobs != .null or config.probabilities != null or
        config.user_id != null or config.extensions != null) return error.LMFailure;
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
            for (message.parts) |part| {
                if (part != .text or part.text.continuation.len != 0) return error.LMFailure;
                values.appendAssumeCapacity(try textPart(arena, part.text.text));
            }
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
    if (generation.count() != 0) try body.put(arena, "generationConfig", .{ .object = generation });
    if (config.store) |store| try body.put(arena, "store", .{ .bool = store });
    if (config.service_tier) |tier| try body.put(arena, "serviceTier", str(tier));
    var writer: std.Io.Writer.Allocating = .init(arena);
    @import("pyjson.zig").writeJson(&writer.writer, .{ .object = body }, .minified) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
    return writer.written();
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

fn parseResponse(arena: Allocator, model: []const u8, raw: Value) Error!lm_mod.Completion {
    if (raw != .object or get(raw, "error") != .null) return error.LMFailure;
    const block = text(get(get(raw, "promptFeedback"), "blockReason"));
    if (block.len != 0 and !std.mem.eql(u8, block, "BLOCK_REASON_UNSPECIFIED")) return error.LMFailure;
    const candidates = elems(get(raw, "candidates"));
    const candidate = if (candidates.len == 0) .null else candidates[0];
    const reason = text(get(candidate, "finishReason"));
    if (contains(&.{ "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER", "NO_IMAGE", "IMAGE_RECITATION", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE" }, reason)) return error.LMFailure;
    if (get(candidate, "groundingMetadata") != .null or get(candidate, "logprobsResult") != .null) return error.LMFailure;
    const wire_parts = elems(get(get(candidate, "content"), "parts"));
    const parts = try arena.alloc(sdk.Part, @max(wire_parts.len, 1));
    var visible: std.Io.Writer.Allocating = .init(arena);
    for (wire_parts, parts[0..wire_parts.len]) |part, *dest| {
        const value = get(part, "text");
        const thought = get(part, "thought");
        if (value != .string or (thought == .bool and thought.bool) or get(part, "thoughtSignature") != .null) return error.LMFailure;
        dest.* = .{ .text = .{ .text = value.string } };
        visible.writer.writeAll(value.string) catch return error.OutOfMemory;
    }
    if (wire_parts.len == 0) parts[0] = .{ .text = .{} };
    const id = text(get(raw, "responseId"));
    var result: lm_mod.Completion = .{
        .text = visible.written(),
        .message = .{ .role = .assistant, .parts = parts },
        .provider_data = sdk.PayloadValue.fromBorrowedValue(arena, raw) catch |err| return codecError(err),
        .metadata = .{
            .id = if (id.len != 0) id else null,
            .model = model,
            .finish_reason = if (std.ascii.eqlIgnoreCase(reason, "MAX_TOKENS")) .length else .stop,
            .usage = try readUsage(arena, get(raw, "usageMetadata")),
        },
    };
    try lm_mod.publishResponse(arena, &result, model, .borrowed);
    return result;
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
    var writer: std.Io.Writer.Allocating = .init(arena);
    writer.writer.writeAll(prefix) catch return error.OutOfMemory;
    for (parts, 0..) |part, index| {
        if (part != .text or part.text.continuation.len != 0) return error.LMFailure;
        if (index != 0) writer.writer.writeByte('\n') catch return error.OutOfMemory;
        writer.writer.writeAll(part.text.text) catch return error.OutOfMemory;
    }
    return singleton(arena, try textPart(arena, writer.written()));
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
    try std.testing.expectError(error.LMFailure, parseResponse(arena, "fixture", blocked));
}
