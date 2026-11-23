//! TypeSafe System One (Jev): one `POST /v1/systemone` per decision request,
//! ported from lm15 `providers/typesafe.py`. Jev generates no text. The
//! response schema's top-level properties must all declare judgments
//! (boolean, string choice, ordered integer score, or bounding box); the
//! native questions travel verbatim, as DSPy's `questions` extension
//! overwrites lm15's derived questions. Answers are validated against the
//! declared judgments and returned raw in `provider_data.typesafe.answers`.
//!
//! Physical execution takes canonical SDK requests, including inline/local
//! PNG and JPEG assets. Native predictions construct that request before the
//! same codec and transport; unsupported media and conversation are refused.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const http = @import("http.zig");
const pyjson = @import("pyjson.zig");
const lm_mod = @import("lm.zig");
const sdk = lm_mod.sdk;
const runtime = @import("runtime_value.zig");
const Value = runtime.Value;
const ObjectMap = runtime.ObjectMap;

const LM = lm_mod.LM;
const Request = lm_mod.Request;
const Completion = lm_mod.Completion;
const Error = lm_mod.Error;

/// Jev's Score ceiling (docs.typesafe.ai/primitives/score).
pub const max_ordered_levels = 10;
/// Jev's Choice ceiling (docs.typesafe.ai/primitives/choice).
pub const max_choice_keys = 255;
/// Jev's Box edges, in Jev's order.
pub const box_edges = [_][]const u8{ "top", "bottom", "left", "right" };
/// Images and Box questions are refused by systemone without this opt-in.
pub const images_header = "TypeSafe-Danger-Zone-Images";

pub const JudgmentKind = enum { boolean, choice, ordered, box };

/// One judgment declared by a top-level response-schema property.
pub const Judgment = struct {
    name: []const u8,
    kind: JudgmentKind,
    /// Declared answer keys: `true`/`false`, choice strings, or `0`..`n-1`.
    /// Empty for a box, which is answered and never scored.
    keys: []const []const u8,
};

/// Wire request bytes plus the experimental-feature header decision.
pub const Wire = struct {
    body: []const u8,
    images_header: bool,
    adaptations: []const sdk.Adaptation,
};

pub const TypeSafe = struct {
    client: *std.http.Client,
    /// Endpoint root, e.g. "http://127.0.0.1:8080"; a trailing slash is
    /// accepted and `/v1/systemone` is appended. There is no public default.
    base_url: []const u8,
    /// e.g. "jev-latest"; a `typesafe:` binding prefix is stripped once.
    model: []const u8,
    /// Bearer token; never logged.
    api_key: ?[]const u8 = null,
    /// Upper bound enforced while reading each response body.
    max_response_bytes: usize = 4 * 1024 * 1024,
    /// Extra attempts after a transport error, HTTP 429 or 5xx.
    max_retries: u8 = 3,
    retry_base_ms: u32 = 500,
    /// Per-network-operation inactivity limit; null disables it.
    timeout_ns: ?u64 = null,
    /// DNS/TCP limit; null inherits timeout_ns.
    connect_timeout_ns: ?u64 = null,
    /// Authentication providers are refreshed on every attempt, not per call.
    header_provider: ?http.HeaderProvider = null,
    /// Local image reads are bounded, unlike Source's unbounded read_bytes.
    max_media_bytes: usize = 16 * 1024 * 1024,
    adaptations: enum { note, silent, refuse } = .note,

    pub fn lm(self: *TypeSafe) LM {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = complete,
                .complete_canonical = completeCanonical,
                .model_name = modelName,
            },
            .supports_decision_requests = true,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *TypeSafe = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    /// Build the physical request without changing the caller's SDK object.
    /// Wire bytes and adaptation views live in the supplied execution arena.
    pub fn requestBody(self: *const TypeSafe, arena: Allocator, request: *const sdk.Request) Error!Wire {
        const fields = request.fields;
        const config = fields.config;
        if (fields.tools.len != 0 or config.tool_choice != null) return error.LMFailure;
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        inline for (.{
            "max_tokens",        "temperature",      "top_p",     "top_k",    "stop",  "seed",
            "frequency_penalty", "presence_penalty", "reasoning", "logprobs", "store", "user_id",
            "service_tier",      "cache",
        }) |name| {
            const asked = try droppedValue(arena, @field(config, name));
            if (asked != .null) {
                if (self.adaptations == .refuse) return error.LMFailure;
                try records.append(arena, .{
                    .field = "config." ++ name,
                    .action = .dropped,
                    .reason = "no such control on the systemone wire (Jev returns decisions, not samples)",
                    .asked = try jsonPayload(arena, asked),
                });
            }
        }
        const schema = try responseSchema(arena, request);
        const found = try judgments(arena, schema);
        const questions = try derivedQuestions(arena, schema, found, &records);
        const state = try sdkState(arena, self.client.io, self.max_media_bytes, request);
        var payload: ObjectMap = .empty;
        try payload.put(arena, "model", .{ .string = wireModel(fields.model) });
        try payload.put(arena, "state", state.value);
        if (state.assets.count() != 0) try payload.put(arena, "assets", .{ .object = state.assets });
        try payload.put(arena, "questions", questions);
        if (config.extensions) |extensions| {
            const extra = extensions.toValue(arena) catch |err| return sdkError(err);
            std.debug.assert(extra == .object);
            var iterator = extra.object.iterator();
            while (iterator.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, "n")) {
                    const value = entry.value_ptr.*;
                    switch (value) {
                        .integer, .float, .bool, .number_string => {
                            const order = @import("value.zig").numericOrder(value, .{ .integer = 1 }) catch return error.LMFailure;
                            if (order == .gt) return error.LMFailure;
                        },
                        else => {},
                    }
                }
                try payload.put(arena, entry.key_ptr.*, entry.value_ptr.*);
            }
        }
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeJson(&aw.writer, .{ .object = payload }, .minified) catch |err| return sdkError(err);
        const body = aw.written();
        if (!std.unicode.utf8ValidateSlice(body)) return error.LMFailure;
        return .{
            .body = body,
            .images_header = payload.contains("assets") or try boxHeader(payload.get("questions").?),
            .adaptations = records.items,
        };
    }

    /// One physical call, returning the actual owning SDK response. Candidate
    /// fanout, public-call cache and history belong outside this engine.
    pub fn completeRequest(self: *TypeSafe, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        const completion = try self.execute(arena, request);
        return completion.response.?;
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *TypeSafe = @ptrCast(@alignCast(ptr));
        const canonical = try self.predictionRequest(arena, request);
        defer canonical.release();
        const out = try arena.alloc(Completion, 1);
        out[0] = try completeCanonical(ptr, arena, canonical);
        return out;
    }

    fn completeCanonical(
        ptr: *anyopaque,
        arena: Allocator,
        request: *const sdk.Request,
    ) Error!Completion {
        const self: *TypeSafe = @ptrCast(@alignCast(ptr));
        return self.execute(arena, request);
    }

    fn predictionRequest(self: *const TypeSafe, arena: Allocator, request: Request) Error!*const sdk.Request {
        const decision = request.decision orelse return error.LMFailure;
        if (request.tools.len != 0 or request.options.tool_choice != null or
            request.options.response_format != null or request.options.n != 1 or
            request.options.max_tokens != null or request.options.stop.len != 0 or
            request.options.top_p != null or request.options.top_k != .null or request.options.seed != .null or
            request.options.frequency_penalty != null or request.options.presence_penalty != null or
            request.options.store != null or request.options.user_id != null or decision.questions != .object) return error.LMFailure;
        const state = try predictionState(request.messages, decision.state);
        const part: sdk.Part = if (request.messages.len == 0)
            .{ .data = .{ .value = try jsonPayload(arena, state) } }
        else
            .{ .text = .{ .text = state.string } };
        var format: ObjectMap = .empty;
        try format.put(arena, "type", .{ .string = "json_schema" });
        try format.put(arena, "schema", decision.response_schema);
        var extensions: ObjectMap = .empty;
        try extensions.put(arena, "questions", decision.questions);
        if (request.options.parallel_tool_calls) |parallel|
            try extensions.put(arena, "parallel_tool_calls", .{ .bool = parallel });
        const reasoning: ?sdk.Reasoning = if (request.options.reasoning_effort) |effort|
            .{ .effort = std.meta.stringToEnum(sdk.ReasoningEffort, effort) orelse return error.LMFailure }
        else
            null;
        return sdk.Request.fromFields(arena, &.{
            .model = self.model,
            .messages = &.{.{ .role = .user, .parts = &.{part} }},
            .config = .{
                .response_format = try jsonPayload(arena, .{ .object = format }),
                .extensions = try jsonPayload(arena, .{ .object = extensions }),
                .temperature = request.options.temperature,
                .logprobs = request.options.logprobs,
                .reasoning = reasoning,
                .service_tier = request.options.service_tier,
            },
        }) catch |err| return sdkError(err);
    }

    fn execute(self: *TypeSafe, arena: Allocator, request: *const sdk.Request) Error!Completion {
        const wire = try self.requestBody(arena, request);
        const url = try std.fmt.allocPrint(arena, "{s}/v1/systemone", .{std.mem.trimEnd(u8, self.base_url, "/")});
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        const headers: []const std.http.Header = if (wire.images_header)
            &.{.{ .name = images_header, .value = "true" }}
        else
            &.{};
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
        const response = transport.send(arena, .{
            .url = url,
            .method = .POST,
            .body = wire.body,
            .content_type = "application/json",
            .authorization = auth,
            .headers = headers,
        }) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
        defer self.client.allocator.free(response.body);
        // Authentication/transport callbacks can mutate opaque SDK payloads.
        // Source reads this same request again at the response boundary.
        var completion = try parseResponse(arena, try responseSchema(arena, request), response.body);
        // lm15 reads the response id only from this header, never the body.
        completion.metadata.id = response.request_id;
        if (completion.metadata.model == null)
            completion.metadata.model = try arena.dupe(u8, wireModel(request.fields.model));
        if (self.adaptations != .silent) completion.adaptations = wire.adaptations;
        try lm_mod.publishResponse(arena, &completion, wireModel(request.fields.model), .borrowed);
        return completion;
    }
};

fn sdkError(err: anyerror) Error {
    return switch (err) {
        error.OutOfMemory, error.WriteFailed => error.OutOfMemory,
        else => error.LMFailure,
    };
}

fn responseSchema(arena: Allocator, request: *const sdk.Request) Error!Value {
    const format = request.fields.config.response_format orelse return .null;
    const fmt = format.toValue(arena) catch |err| return sdkError(err);
    if (fmt != .object or !isString(fmt.object.get("type") orelse .null, "json_schema")) return .null;
    return fmt.object.get("schema") orelse .null;
}

fn scalarValue(input: anytype) Value {
    const T = @TypeOf(input);
    if (T == Value) return input;
    return switch (@typeInfo(T)) {
        .optional => if (input) |value| scalarValue(value) else .null,
        .@"enum" => .{ .string = @tagName(input) },
        .bool => .{ .bool = input },
        .float => .{ .float = input },
        .pointer => .{ .string = input },
        else => @compileError("not an SDK configuration scalar"),
    };
}

/// Source records scalar requests verbatim, tuples using Python repr, and
/// the two configuration dataclasses using their generated field-order repr.
fn droppedValue(arena: Allocator, input: anytype) Error!Value {
    const T = @TypeOf(input);
    if (@typeInfo(T) == .optional) {
        return if (input) |value| droppedValue(arena, value) else .null;
    }
    if (T == sdk.Reasoning or T == sdk.CacheConfig) {
        var aw: Writer.Allocating = .init(arena);
        aw.writer.writeAll(if (T == sdk.Reasoning) "Reasoning(" else "CacheConfig(") catch return error.OutOfMemory;
        inline for (@typeInfo(T).@"struct".field_names, 0..) |name, index| {
            if (index != 0) aw.writer.writeAll(", ") catch return error.OutOfMemory;
            aw.writer.writeAll(name ++ "=") catch return error.OutOfMemory;
            pyjson.writeRepr(&aw.writer, scalarValue(@field(input, name)), 0) catch |err| return sdkError(err);
        }
        aw.writer.writeByte(')') catch return error.OutOfMemory;
        return .{ .string = aw.written() };
    }
    if (T == []const []const u8) {
        if (input.len == 0) return .null;
        var aw: Writer.Allocating = .init(arena);
        aw.writer.writeByte('(') catch return error.OutOfMemory;
        for (input, 0..) |item, index| {
            if (index != 0) aw.writer.writeAll(", ") catch return error.OutOfMemory;
            pyjson.writeReprString(&aw.writer, item) catch return error.OutOfMemory;
        }
        if (input.len == 1) aw.writer.writeByte(',') catch return error.OutOfMemory;
        aw.writer.writeByte(')') catch return error.OutOfMemory;
        return .{ .string = aw.written() };
    }
    return scalarValue(input);
}

fn derivedQuestions(
    arena: Allocator,
    schema: Value,
    found: []const Judgment,
    records: *std.ArrayList(sdk.Adaptation),
) Error!Value {
    var questions: ObjectMap = .empty;
    for (found) |judgment| {
        const prop = schema.object.get("properties").?.object.get(judgment.name).?;
        const description = prop.object.get("description") orelse .null;
        const instruction = if (description == .string and description.string.len != 0) description.string else judgment.name;
        if (description != .string or description.string.len == 0) {
            try records.append(arena, .{
                .field = try std.fmt.allocPrint(arena, "config.response_format.schema.properties.{s}.description", .{judgment.name}),
                .action = .defaulted,
                .reason = "a judgment without a description: the property name goes as the instruction (Jev never sees property names)",
                .applied = .borrowed(.{ .string = judgment.name }),
            });
        }
        var question: ObjectMap = .empty;
        const kind = switch (judgment.kind) {
            .boolean => "noul",
            .box => "box",
            .choice => "choice",
            .ordered => "score",
        };
        try question.put(arena, "type", .{ .string = kind });
        try question.put(arena, "instructions", .{ .string = instruction });
        if (judgment.kind == .choice or judgment.kind == .ordered) {
            const branches = constBranches(prop);
            var choices: ObjectMap = .empty;
            var levels: std.array_list.Managed(Value) = .init(arena);
            for (judgment.keys, 0..) |key, index| {
                const desc = if (branches) |items| items[index].object.get("description") orelse .null else .null;
                if (judgment.kind == .choice) {
                    try choices.put(arena, key, if (desc == .string) desc else .null);
                } else {
                    try levels.append(if (desc == .string and desc.string.len != 0) desc else .{ .string = key });
                }
            }
            try question.put(arena, "criteria", if (judgment.kind == .choice) .{ .object = choices } else .{ .array = levels });
        }
        try questions.put(arena, judgment.name, .{ .object = question });
    }
    return .{ .object = questions };
}

const State = struct { value: Value, assets: ObjectMap };

test "TypeSafe local images enforce an inclusive byte cap without changing the request" {
    const testing = std.testing;
    var directory = testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(testing.io, .{ .sub_path = "image", .data = "abc" });
    const path = try directory.dir.realPathFileAlloc(testing.io, "image", testing.allocator);
    defer testing.allocator.free(path);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const format = try @import("json_repair.zig").parseJson(arena,
        \\{"type":"json_schema","schema":{"type":"object","properties":{"safe":{"type":"boolean","description":"Safe?"}}}}
    );
    const request = try sdk.Request.fromFields(testing.allocator, &.{
        .model = "typesafe:jev-test",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .image = .{ .media_type = "image/png", .path = path } }} }},
        .config = .{ .response_format = try jsonPayload(arena, format) },
    });
    defer request.release();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var provider: TypeSafe = .{ .client = &client, .base_url = "http://127.0.0.1:1", .model = "unused", .max_media_bytes = 3 };
    const wire = try provider.requestBody(arena, request);
    const body = try @import("json_repair.zig").parseJson(arena, wire.body);
    try testing.expect(wire.images_header);
    try testing.expectEqualStrings("jev-test", body.object.get("model").?.string);
    try testing.expectEqualStrings("<lm15-image-1>", body.object.get("state").?.string);
    try testing.expectEqualStrings("YWJj", body.object.get("assets").?.object.get("<lm15-image-1>").?.object.get("data_base64").?.string);
    provider.max_media_bytes = 2;
    try testing.expectError(error.LMFailure, provider.requestBody(arena, request));
    try testing.expectEqualStrings(path, request.fields.messages[0].parts[0].image.path.?);
    try testing.expect(request.fields.messages[0].parts[0].image.data == null);
}

fn imageFormat(media_type: []const u8) Error![]const u8 {
    if (std.ascii.eqlIgnoreCase(media_type, "image/png")) return "png";
    if (std.ascii.eqlIgnoreCase(media_type, "image/jpeg") or
        std.ascii.eqlIgnoreCase(media_type, "image/jpg")) return "jpg";
    return error.LMFailure;
}

fn sdkState(arena: Allocator, io: std.Io, maximum_file_bytes: usize, request: *const sdk.Request) Error!State {
    for (request.fields.messages) |message| {
        for (message.parts) |part| switch (part) {
            .text, .data => {},
            .image => |image| {
                if (image.url != null or image.file_id != null) return error.LMFailure;
                _ = try imageFormat(image.media_type);
            },
            else => return error.LMFailure,
        };
    }
    if (request.fields.system != null or request.fields.messages.len != 1) return error.LMFailure;
    const message = request.fields.messages[0];
    if (message.role != .user) return error.LMFailure;
    var has_image = false;
    for (message.parts) |part| has_image = has_image or part == .image;
    if (message.parts.len != 1 and !has_image) return error.LMFailure;
    var assets: ObjectMap = .empty;
    var elements: std.array_list.Managed(Value) = .init(arena);
    for (message.parts) |part| {
        const element: Value = switch (part) {
            .text => |text| .{ .string = text.text },
            .data => |data| data.value.toValue(arena) catch |err| return sdkError(err),
            .image => |image| blk: {
                const reference = try std.fmt.allocPrint(arena, "<lm15-image-{d}>", .{assets.count() + 1});
                const data = image.data orelse encoded: {
                    const path = image.path orelse return error.LMFailure;
                    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(maximum_file_bytes +| 1)) catch |err| return sdkError(err);
                    defer arena.free(bytes);
                    if (bytes.len > maximum_file_bytes) return error.LMFailure;
                    const encoder = std.base64.standard.Encoder;
                    const groups = bytes.len / 3 + @intFromBool(bytes.len % 3 != 0);
                    const size = std.math.mul(usize, groups, 4) catch return error.OutOfMemory;
                    const output = try arena.alloc(u8, size);
                    break :encoded encoder.encode(output, bytes);
                };
                var asset: ObjectMap = .empty;
                try asset.put(arena, "type", .{ .string = "image" });
                try asset.put(arena, "format", .{ .string = try imageFormat(image.media_type) });
                try asset.put(arena, "data_base64", .{ .string = data });
                try assets.put(arena, reference, .{ .object = asset });
                break :blk .{ .string = reference };
            },
            else => unreachable,
        };
        if (message.parts.len == 1) return .{ .value = element, .assets = assets };
        try elements.append(element);
    }
    return .{ .value = .{ .array = elements }, .assets = assets };
}

/// lm15 `_wire_request`: strip exactly one `typesafe:` binding prefix.
fn wireModel(model: []const u8) []const u8 {
    const colon = std.mem.indexOfScalar(u8, model, ':') orelse return model;
    const head = model[0..colon];
    const rest = model[colon + 1 ..];
    if (rest.len == 0 or head.len != "typesafe".len) return model;
    for (head, "typesafe") |byte, want| {
        if ((if (byte == '_') '-' else byte) != want) return model;
    }
    return rest;
}

/// Lift the native prediction adapter's state before the physical SDK boundary.
fn predictionState(messages: []const @import("adapter.zig").Message, state: Value) Error!Value {
    if (messages.len == 0) return state;
    if (state != .null or messages.len != 1) return error.LMFailure;
    const message = messages[0];
    if (message.role != .user or message.tool_calls.len != 0 or message.tool_call_id != null or
        message.tool_result_name != null or message.reasoning != null or message.continuation.len != 0) return error.LMFailure;
    return switch (message.content) {
        .text => |text| .{ .string = text },
        .parts => |parts| blk: {
            for (parts) |part| {
                if (part != .object) return error.LMFailure;
                if (!isString(part.object.get("type") orelse return error.LMFailure, "text")) return error.LMFailure;
            }
            if (parts.len != 1) return error.LMFailure;
            const text = parts[0].object.get("text") orelse Value{ .string = "" };
            if (text != .string) return error.LMFailure;
            break :blk text;
        },
    };
}

/// lm15 checks `q["type"] == "box"` over the final questions with `any`, so
/// malformed questions after the first box are not inspected.
fn boxHeader(questions: Value) Error!bool {
    if (questions != .object) return error.LMFailure;
    var iterator = questions.object.iterator();
    while (iterator.next()) |entry| {
        const question = entry.value_ptr.*;
        if (question != .object) return error.LMFailure;
        const kind = question.object.get("type") orelse return error.LMFailure;
        if (kind == .string and std.mem.eql(u8, kind.string, "box")) return true;
    }
    return false;
}

/// lm15 `request_judgments` plus the `_questions` refusals: every property
/// must declare a judgment, at least one must exist, and Jev's choice/score
/// ceilings apply.
pub fn judgments(arena: Allocator, schema: Value) Error![]const Judgment {
    if (schema != .object) return error.LMFailure;
    if (schema.object.get("type")) |kind| {
        if (kind != .null and !isString(kind, "object")) return error.LMFailure;
    }
    const properties = schema.object.get("properties") orelse return error.LMFailure;
    if (properties != .object or properties.object.count() == 0) return error.LMFailure;
    const out = try arena.alloc(Judgment, properties.object.count());
    var iterator = properties.object.iterator();
    var index: usize = 0;
    while (iterator.next()) |entry| : (index += 1) {
        const judgment = try judgmentOf(arena, entry.key_ptr.*, entry.value_ptr.*) orelse return error.LMFailure;
        switch (judgment.kind) {
            .choice => if (judgment.keys.len > max_choice_keys) return error.LMFailure,
            .ordered => if (judgment.keys.len > max_ordered_levels) return error.LMFailure,
            .boolean, .box => {},
        }
        out[index] = judgment;
    }
    return out;
}

/// Generic lm15 recognition, without Jev's all-properties or size restrictions.
/// OpenAI folds ordinary structured output only when some judgment is declared.
pub fn hasJudgments(arena: Allocator, schema: Value) Allocator.Error!bool {
    return hasJudgmentKind(arena, schema, false);
}

/// Boxes carry answers but no distribution over declared keys.
pub fn hasMeasurableJudgments(arena: Allocator, schema: Value) Allocator.Error!bool {
    return hasJudgmentKind(arena, schema, true);
}

fn hasJudgmentKind(arena: Allocator, schema: Value, measurable: bool) Allocator.Error!bool {
    if (schema != .object) return false;
    if (schema.object.get("type")) |kind| {
        if (kind != .null and !isString(kind, "object")) return false;
    }
    const properties = schema.object.get("properties") orelse return false;
    if (properties != .object) return false;
    var iterator = properties.object.iterator();
    while (iterator.next()) |entry| {
        if (try judgmentOf(arena, entry.key_ptr.*, entry.value_ptr.*)) |judgment| {
            if (!measurable or judgment.keys.len != 0) return true;
        }
    }
    return false;
}

fn judgmentOf(arena: Allocator, name: []const u8, prop: Value) Allocator.Error!?Judgment {
    if (prop != .object) return null;
    if (isBox(prop)) return .{ .name = name, .kind = .box, .keys = &.{} };
    const kind: Value = prop.object.get("type") orelse .null;
    if (isString(kind, "boolean")) return .{ .name = name, .kind = .boolean, .keys = &.{ "true", "false" } };
    const enum_value: Value = prop.object.get("enum") orelse .null;
    const branches = constBranches(prop);
    var values: std.ArrayList(Value) = .empty;
    if (enum_value == .array and enum_value.array.items.len != 0 and branches == null) {
        try values.appendSlice(arena, enum_value.array.items);
    } else if (branches != null and enum_value == .null) {
        try values.ensureTotalCapacity(arena, branches.?.len);
        for (branches.?) |branch| values.appendAssumeCapacity(branch.object.get("const").?);
    } else return null;

    var all_strings = true;
    var all_integers = true;
    for (values.items) |value| {
        all_strings = all_strings and value == .string and value.string.len != 0;
        all_integers = all_integers and (value == .integer or
            (value == .number_string and std.json.isNumberFormattedLikeAnInteger(value.number_string)));
    }
    if (all_strings) {
        if (kind != .null and !isString(kind, "string")) return null;
        const keys = try arena.alloc([]const u8, values.items.len);
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        try seen.ensureTotalCapacity(arena, @intCast(values.items.len));
        for (values.items, keys) |value, *key| {
            if (seen.getOrPutAssumeCapacity(value.string).found_existing) return null;
            key.* = value.string;
        }
        return .{ .name = name, .kind = .choice, .keys = keys };
    }
    if (all_integers) {
        if (kind != .null and !isString(kind, "integer")) return null;
        if (values.items.len < 2) return null;
        // Oversized integers can never equal their position.
        for (values.items, 0..) |value, i| {
            if (value != .integer or value.integer != @as(i64, @intCast(i))) return null;
        }
        const keys = try arena.alloc([]const u8, values.items.len);
        for (keys, 0..) |*key, i| key.* = try std.fmt.allocPrint(arena, "{d}", .{i});
        return .{ .name = name, .kind = .ordered, .keys = keys };
    }
    return null;
}

/// An object whose properties are exactly the four edges, each a number.
fn isBox(prop: Value) bool {
    if (prop.object.get("type")) |kind| {
        if (!isString(kind, "object")) return false;
    }
    const properties = prop.object.get("properties") orelse return false;
    if (properties != .object or properties.object.count() != box_edges.len) return false;
    for (box_edges) |edge| {
        const schema = properties.object.get(edge) orelse return false;
        if (schema != .object) return false;
        if (!isString(schema.object.get("type") orelse return false, "number")) return false;
    }
    return true;
}

fn constBranches(prop: Value) ?[]const Value {
    const branches = prop.object.get("anyOf") orelse return null;
    if (branches != .array or branches.array.items.len == 0) return null;
    for (branches.array.items) |branch| {
        if (branch != .object or !branch.object.contains("const")) return null;
    }
    return branches.array.items;
}

fn isString(value: Value, expected: []const u8) bool {
    return value == .string and std.mem.eql(u8, value.string, expected);
}

fn jsonPayload(arena: Allocator, input: Value) Error!lm_mod.sdk.JsonValue {
    return lm_mod.sdk.JsonValue.fromBorrowedValue(arena, input) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    };
}

/// `parse_response` recognizes declared judgments afresh, without repeating
/// build-time free-form/size refusals. Mutable schemas can change in transit.
pub fn responseJudgments(arena: Allocator, schema: Value) Allocator.Error![]Judgment {
    if (schema != .object) return &.{};
    if (schema.object.get("type")) |kind|
        if (kind != .null and !isString(kind, "object")) return &.{};
    const properties = schema.object.get("properties") orelse return &.{};
    if (properties != .object) return &.{};
    var found: std.ArrayList(Judgment) = .empty;
    var iterator = properties.object.iterator();
    while (iterator.next()) |entry|
        if (try judgmentOf(arena, entry.key_ptr.*, entry.value_ptr.*)) |judgment|
            try found.append(arena, judgment);
    return found.toOwnedSlice(arena);
}

test "TypeSafe response judgment recognition is distinct from build-time admission" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const schema = try @import("json_repair.zig").parseJson(arena,
        \\{"type":"object","properties":{"flag":{"type":"boolean"},"free":{"type":"string"}}}
    );
    try testing.expectError(error.LMFailure, judgments(arena, schema));
    const response = try parseResponse(arena, schema,
        \\{"answers":{"flag":{"type":"noul","noul":0.8}}}
    );
    try testing.expect(response.message.?.parts[0].data.value.root.object.get("flag").?.bool);
    const empty = try parseResponse(arena, .null,
        \\{"answers":{}}
    );
    try testing.expectEqual(@as(usize, 0), empty.message.?.parts[0].data.value.root.object.count());
    try testing.expect(empty.message.?.parts[0].data.probabilities == null);
    try testing.expect(empty.message.?.parts[0].data.method == null);
}

/// lm15 `parse_response`: validates every declared answer, then returns the
/// raw answers. Jev generates no text, so `text` is null.
pub fn parseResponse(arena: Allocator, response_schema: Value, body: []const u8) Error!Completion {
    const found = try responseJudgments(arena, response_schema);
    const data = @import("json_repair.zig").parseJson(arena, body) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    };
    if (data != .object) return error.LMFailure;
    const answers = data.object.get("answers") orelse return error.LMFailure;
    if (answers != .object or answers.object.count() != found.len) return error.LMFailure;
    for (found) |judgment| {
        const answer = answers.object.get(judgment.name) orelse return error.LMFailure;
        try validateAnswer(judgment, answer);
    }
    // lm15 builds Usage from these two counters only; a refused count fails
    // the response, and an absent or null usage leaves every counter unknown.
    var counts: lm_mod.Usage = .{};
    if (data.object.get("usage")) |reported| {
        if (reported != .null) {
            if (reported != .object) return error.LMFailure;
            counts.input_tokens = reported.object.get("input_tokens") orelse .null;
            counts.output_tokens = reported.object.get("output_tokens") orelse .null;
        }
    }
    const usage = lm_mod.Usage.init(arena, counts) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidUsage => error.LMFailure,
    };
    // The engine applies the SDK's wire-request-model fallback when absent.
    var model_reported: ?[]const u8 = null;
    if (data.object.get("model")) |model| {
        if (model != .null) {
            if (model != .string or model.string.len == 0) return error.LMFailure;
            model_reported = model.string;
        }
    }
    var typesafe: ObjectMap = .empty;
    try typesafe.put(arena, "answers", answers);
    var provider_data: ObjectMap = .empty;
    try provider_data.put(arena, "typesafe", .{ .object = typesafe });
    const parts = try arena.alloc(lm_mod.sdk.Part, 1);
    parts[0] = .{ .data = try judgmentData(arena, found, answers) };
    return .{
        .text = null,
        .message = .{ .role = .assistant, .parts = parts },
        .provider_data = try jsonPayload(arena, .{ .object = provider_data }),
        .metadata = .{ .model = model_reported, .finish_reason = .stop, .usage = usage },
    };
}

/// Project already validated provider answers without changing opaque box numbers.
/// Ordered ties retain the first declared key, just like Python's max().
fn judgmentData(arena: Allocator, found: []const Judgment, answers: Value) Error!lm_mod.sdk.DataPart {
    std.debug.assert(answers == .object);
    std.debug.assert(found.len == answers.object.count());
    var values: ObjectMap = .empty;
    var distributions: ObjectMap = .empty;
    for (found) |judgment| {
        const answer = answers.object.get(judgment.name).?;
        if (judgment.kind == .box) {
            var edges: ObjectMap = .empty;
            for (box_edges) |edge| try edges.put(arena, edge, answer.object.get("box").?.object.get(edge).?);
            try values.put(arena, judgment.name, .{ .object = edges });
            continue;
        }
        var distribution: ObjectMap = .empty;
        if (judgment.kind == .boolean) {
            const p = judgmentProbability(answer.object.get("noul").?);
            try values.put(arena, judgment.name, .{ .bool = p >= 0.5 });
            try distribution.put(arena, "true", .{ .float = p });
            try distribution.put(arena, "false", .{ .float = 1.0 - p });
        } else {
            var maximum: f64 = -1;
            var selected: usize = 0;
            for (judgment.keys, 0..) |key, index| {
                const p = judgmentProbability(answer.object.get("probabilities").?.object.get(key).?);
                try distribution.put(arena, key, .{ .float = p });
                if (p > maximum) {
                    maximum = p;
                    selected = index;
                }
            }
            const value: Value = if (judgment.kind == .choice)
                answer.object.get("choice").?
            else
                .{ .integer = @intCast(selected) };
            try values.put(arena, judgment.name, value);
        }
        try distributions.put(arena, judgment.name, .{ .object = distribution });
    }
    return .{
        .value = try jsonPayload(arena, .{ .object = values }),
        .probabilities = if (distributions.count() != 0)
            try jsonPayload(arena, .{ .object = distributions })
        else
            null,
        .method = if (distributions.count() != 0) .provider_classification else null,
    };
}

fn judgmentProbability(raw: Value) f64 {
    std.debug.assert(raw == .integer or raw == .float);
    const number: f64 = if (raw == .integer) @floatFromInt(raw.integer) else raw.float;
    std.debug.assert(number >= 0 and number <= 1);
    return number;
}

fn validateAnswer(judgment: Judgment, answer: Value) Error!void {
    if (answer != .object) return error.LMFailure;
    const expected = switch (judgment.kind) {
        .boolean => "noul",
        .choice => "choice",
        .ordered => "score",
        .box => "box",
    };
    if (!isString(answer.object.get("type") orelse .null, expected)) return error.LMFailure;
    switch (judgment.kind) {
        .box => {
            const edges = answer.object.get("box") orelse return error.LMFailure;
            if (edges != .object or edges.object.count() != box_edges.len) return error.LMFailure;
            for (box_edges) |edge| try probability(edges.object.get(edge) orelse return error.LMFailure);
        },
        .boolean => try probability(answer.object.get("noul") orelse .null),
        .choice, .ordered => {
            const distribution = answer.object.get("probabilities") orelse return error.LMFailure;
            if (distribution != .object or distribution.object.count() != judgment.keys.len) return error.LMFailure;
            for (judgment.keys) |key| try probability(distribution.object.get(key) orelse return error.LMFailure);
            if (judgment.kind == .choice) {
                const pick = answer.object.get("choice") orelse return error.LMFailure;
                if (pick != .string) return error.LMFailure;
                for (judgment.keys) |key| {
                    if (std.mem.eql(u8, key, pick.string)) break;
                } else return error.LMFailure;
            }
        },
    }
}

/// A finite JSON number in [0, 1]; booleans and oversized numbers refuse.
fn probability(raw: Value) Error!void {
    const ok = switch (raw) {
        .integer => |value| value == 0 or value == 1,
        .float => |value| std.math.isFinite(value) and value >= 0 and value <= 1,
        else => false,
    };
    if (!ok) return error.LMFailure;
}
