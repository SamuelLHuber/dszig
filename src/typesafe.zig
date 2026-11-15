//! TypeSafe System One (Jev): one `POST /v1/systemone` per decision request,
//! ported from lm15 `providers/typesafe.py`. Jev generates no text. The
//! response schema's top-level properties must all declare judgments
//! (boolean, string choice, ordered integer score, or bounding box); the
//! native questions travel verbatim, as DSPy's `questions` extension
//! overwrites lm15's derived questions. Answers are validated against the
//! declared judgments and returned raw in `provider_data.typesafe.answers`.
//!
//! Native canonical content has no image part, so systemone image assets
//! cannot be expressed here; document parts are refused exactly as upstream.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const http = @import("http.zig");
const pyjson = @import("pyjson.zig");
const lm_mod = @import("lm.zig");
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

    pub fn lm(self: *TypeSafe) LM {
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete },
            .supports_decision_requests = true,
        };
    }

    /// The `/v1/systemone` body. Every refusal lm15 raises before the wire
    /// fails with `LMFailure`; knobs lm15 drops with a record are ignored.
    pub fn requestBody(self: *const TypeSafe, arena: Allocator, request: Request) Error!Wire {
        const decision = request.decision orelse return error.LMFailure;
        if (request.tools.len != 0) return error.LMFailure;
        const options = request.options;
        if (options.tool_choice != null) return error.LMFailure;
        // The decision schema is the single response format.
        if (options.response_format != null) return error.LMFailure;
        if (options.n != 1) return error.LMFailure;
        // temperature and reasoning are lm15 `_DROPPED_KNOBS`; rollout_id is
        // DSPy cache identity, never a provider field.
        _ = try judgments(arena, decision.response_schema);
        const state = try wireState(request.messages, decision.state);
        if (decision.questions != .object) return error.LMFailure;

        var payload: ObjectMap = .empty;
        try payload.put(arena, "model", .{ .string = wireModel(self.model) });
        try payload.put(arena, "state", state);
        try payload.put(arena, "questions", decision.questions);
        // Not an lm15 Config field: a provider extension, copied verbatim
        // after `questions` like every other `cfg.extensions` entry.
        if (options.parallel_tool_calls) |parallel| try payload.put(arena, "parallel_tool_calls", .{ .bool = parallel });
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeJson(&aw.writer, .{ .object = payload }, .minified) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep, error.ParseValue => error.LMFailure,
        };
        const body = aw.written();
        // lm15 refuses text with no UTF-8 form (lone surrogates) before HTTP.
        if (!std.unicode.utf8ValidateSlice(body)) return error.LMFailure;
        return .{ .body = body, .images_header = try boxHeader(decision.questions) };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *TypeSafe = @ptrCast(@alignCast(ptr));
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
        const out = try arena.alloc(Completion, 1);
        out[0] = try parseResponse(arena, request.decision.?.response_schema, response.body);
        return out;
    }
};

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

/// The Jev state. Without messages it is the DataPart value DSPy builds for
/// text-only state. A single user message instead follows lm15 `_state`:
/// its one text part is the state; documents and other parts are refused.
fn wireState(messages: []const @import("adapter.zig").Message, state: Value) Error!Value {
    if (messages.len == 0) return state;
    if (state != .null or messages.len != 1) return error.LMFailure;
    const message = messages[0];
    if (message.role != .user or message.tool_calls.len != 0 or message.tool_call_id != null or
        message.reasoning != null or message.continuation.len != 0) return error.LMFailure;
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

/// lm15 `parse_response`: validates every declared answer, then returns the
/// raw answers. Jev generates no text, so `text` is null.
pub fn parseResponse(arena: Allocator, response_schema: Value, body: []const u8) Error!Completion {
    const found = try judgments(arena, response_schema);
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
    if (data.object.get("usage")) |usage| {
        if (usage != .null) {
            if (usage != .object) return error.LMFailure;
            for ([_][]const u8{ "input_tokens", "output_tokens" }) |field| {
                if (!tokenCount(usage.object.get(field) orelse .null)) return error.LMFailure;
            }
        }
    }
    if (data.object.get("model")) |model| {
        if (model != .null and (model != .string or model.string.len == 0)) return error.LMFailure;
    }
    var typesafe: ObjectMap = .empty;
    try typesafe.put(arena, "answers", answers);
    var provider_data: ObjectMap = .empty;
    try provider_data.put(arena, "typesafe", .{ .object = typesafe });
    return .{ .text = null, .provider_data = .{ .object = provider_data } };
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

/// lm15 Usage counters: null, a nonnegative integer, or a same-valued float.
fn tokenCount(raw: Value) bool {
    return switch (raw) {
        .null => true,
        .integer => |value| value >= 0,
        .float => |value| std.math.isFinite(value) and @floor(value) == value and value >= 0,
        .number_string => |text| std.json.isNumberFormattedLikeAnInteger(text) and text[0] != '-',
        else => false,
    };
}
