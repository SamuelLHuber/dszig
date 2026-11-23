//! JSON option ingress for the public string-prompt lane, using the generic Chat vocabulary.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const sdk = @import("sdk_response.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Object = runtime.ObjectMap;
const Error = sdk.Error || error{InvalidPromptOptions};
const invalid = error.InvalidPromptOptions;

pub const Input = struct {
    defaults: Object = .{},
    overrides: Object = .{},
};

const client_keys = .{
    "api_key",                 "api_base",     "base_url", "headers",     "extra_headers",       "timeout", "api_version",
    "azure_ad_token_provider", "organization", "project",  "extra_query", "custom_llm_provider",
};
const reserved = client_keys ++ .{
    "model", "messages", "request", "cache", "rollout_id", "n", "num_generations",
};
const scalars = .{
    "temperature",  "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty",
    "service_tier", "store",
};
const specials = .{
    "max_completion_tokens", "max_tokens",       "stop",                   "logprobs",             "top_logprobs", "response_format",
    "tools",                 "tool_choice",      "parallel_tool_calls",    "safety_identifier",    "user",         "reasoning_effort",
    "reasoning",             "prompt_cache_key", "prompt_cache_retention", "prompt_cache_options",
};
const refused = .{
    "user_id",          "functions",    "function_call", "thinking", "enable_thinking", "chat_template_kwargs",
    "reasoning_format", "prompt_cache",
};
const dropped = .{ "stream", "stream_options" };

pub fn build(
    allocator: Allocator,
    model: []const u8,
    prompt: []const u8,
    input: *const Input,
    text_mode: bool,
    maximum_candidates: u32,
) Error!struct { request: *const sdk.Request, count: u32 } {
    const max_options = 1024;
    if (input.defaults.count() > max_options or input.overrides.count() > max_options) return invalid;
    for (input.overrides.keys()) |key| {
        if (in(key, client_keys) or in(key, .{ "model", "messages", "request" })) return invalid;
    }
    const count = try candidateCount(input, maximum_candidates);
    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var merged: Object = .{};
    for (input.defaults.keys(), input.defaults.values()) |key, value| {
        if (overridden(input, key)) continue;
        try mergeOption(arena, &merged, key, value);
    }
    for (input.overrides.keys(), input.overrides.values()) |key, value| {
        try mergeOption(arena, &merged, key, value);
    }
    var config = try readConfig(arena, &merged, text_mode);
    var extensions: Object = .{};
    for (merged.keys(), merged.values()) |key, value| {
        if (!in(key, scalars ++ specials ++ dropped)) try extensions.put(arena, key, value);
    }
    if (extensions.count() > 0) try config.put(arena, "extensions", .{ .object = extensions });
    const tools = try readTools(arena, merged.get("tools"));
    if (try readChoice(arena, &merged, tools.items)) |choice| {
        try config.put(arena, "tool_choice", choice);
    }
    const part = try object(arena, .{ .type = str("text"), .text = str(prompt) });
    const message = try object(arena, .{ .role = str("user"), .parts = try array(arena, &.{part}) });
    const record = try object(arena, .{
        .model = str(model),
        .messages = try array(arena, &.{message}),
        .config = Value{ .object = config },
        .tools = Value{ .array = tools },
    });
    return .{ .request = try sdk.Request.fromValue(allocator, record), .count = count };
}

fn mergeOption(arena: Allocator, merged: *Object, key: []const u8, value: Value) Error!void {
    if (value == .null or in(key, reserved)) return;
    if (in(key, refused)) return invalid;
    try merged.put(arena, key, value);
}

fn overridden(input: *const Input, key: []const u8) bool {
    if (input.overrides.contains(key)) return true;
    inline for (.{ .{ "max_tokens", "max_completion_tokens" }, .{ "reasoning", "reasoning_effort" } }) |aliases| {
        if (in(key, aliases)) return input.overrides.contains(aliases[0]) or
            input.overrides.contains(aliases[1]);
    }
    return false;
}

fn candidateCount(input: *const Input, maximum: u32) Error!u32 {
    var count = input.overrides.get("n") orelse input.overrides.get("num_generations") orelse .null;
    if (count == .null) {
        for ([_][]const u8{ "n", "num_generations" }) |key| {
            const value = input.defaults.get(key) orelse continue;
            if (value == .null or (value == .integer and value.integer == 0) or
                (value == .bool and !value.bool)) continue;
            count = value;
            break;
        }
    }
    if (count == .null) return 1;
    if (count != .integer or count.integer < 1 or count.integer > maximum) return invalid;
    return @intCast(count.integer);
}

fn readConfig(arena: Allocator, merged: *const Object, text_mode: bool) Error!Object {
    var config: Object = .{};
    inline for (scalars) |key| {
        if (merged.get(key)) |value| try config.put(arena, key, value);
    }
    const max_tokens = merged.get("max_tokens");
    const max_completion = merged.get("max_completion_tokens");
    if (max_tokens != null and max_completion != null) {
        if (!sameInteger(max_tokens.?, max_completion.?)) return invalid;
    }
    if (max_completion orelse max_tokens) |value| try config.put(arena, "max_tokens", value);
    if (merged.get("stop")) |value| try config.put(arena, "stop", switch (value) {
        .string => try array(arena, &.{value}),
        .array => value,
        else => return invalid,
    });
    if (try readLogprobs(merged, text_mode)) |value| try config.put(arena, "logprobs", value);
    const user = merged.get("user");
    const safety_identifier = merged.get("safety_identifier");
    if (user != null and safety_identifier != null) return invalid;
    if (user orelse safety_identifier) |value| try config.put(arena, "user_id", value);
    if (merged.get("response_format")) |value| {
        if (try readFormat(arena, value)) |format| try config.put(arena, "response_format", format);
    }
    if (try readReasoning(arena, merged)) |reasoning| try config.put(arena, "reasoning", reasoning);
    if (try readCache(arena, merged)) |cache| try config.put(arena, "cache", cache);
    return config;
}

fn readLogprobs(merged: *const Object, text_mode: bool) Error!?Value {
    const raw = merged.get("logprobs") orelse .null;
    const top = merged.get("top_logprobs");
    if (text_mode and (raw == .integer or raw == .number_string)) {
        if (top) |value| if (!sameInteger(raw, value)) return invalid;
        return raw;
    }
    if (raw == .bool and raw.bool) return top orelse .{ .integer = 0 };
    if (raw != .null and !(raw == .bool and !raw.bool) and
        !(raw == .integer and raw.integer == 0) and !(raw == .float and raw.float == 0)) return invalid;
    if (top != null) return invalid;
    return null;
}

fn readFormat(arena: Allocator, value: Value) Error!?Value {
    const format = try asObject(value);
    const kind = format.get("type") orelse return invalid;
    if (is(kind, "text") or is(kind, "json_object")) {
        try only(&format, .{"type"});
        return if (is(kind, "text")) null else value;
    }
    if (!is(kind, "json_schema")) return invalid;
    try only(&format, .{ "type", "json_schema" });
    const inner = try asObject(format.get("json_schema") orelse return invalid);
    try only(&inner, .{ "schema", "name", "strict", "description" });
    if (present(&inner, "description") != null) return invalid;
    const schema = inner.get("schema") orelse return invalid;
    _ = try asObject(schema);
    var result: Object = .{};
    try result.put(arena, "type", kind);
    try result.put(arena, "schema", schema);
    if (present(&inner, "name")) |name| {
        if (!is(name, "response")) try result.put(arena, "name", str(try asString(name)));
    }
    if (present(&inner, "strict")) |strict| {
        if (strict != .bool) return invalid;
        try result.put(arena, "strict", strict);
    }
    return .{ .object = result };
}

fn readReasoning(arena: Allocator, merged: *const Object) Error!?Value {
    var effort = merged.get("reasoning_effort") orelse .null;
    var summary: Value = .null;
    if (merged.get("reasoning")) |value| {
        const spec = try asObject(value);
        effort = spec.get("effort") orelse str("medium");
        const enabled = spec.get("enabled") orelse .null;
        if (enabled == .bool and !enabled.bool) effort = str("off");
        summary = spec.get("summary") orelse .null;
    }
    if (effort == .null) {
        if (merged.contains("reasoning")) return invalid;
        return null;
    }
    if (is(effort, "none")) effort = str("off");
    if (!in(try asString(effort), comptime std.meta.fieldNames(sdk.ReasoningEffort).*)) return invalid;
    if (is(effort, "off")) {
        if (summary != .null) return invalid;
        return try object(arena, .{ .effort = effort });
    }
    if (summary == .null) summary = str("auto");
    if (!in(try asString(summary), comptime std.meta.fieldNames(sdk.ReasoningSummary).*)) return invalid;
    return try object(arena, .{ .effort = effort, .summary = summary });
}

fn readCache(arena: Allocator, merged: *const Object) Error!?Value {
    const key = merged.get("prompt_cache_key");
    const retention = merged.get("prompt_cache_retention");
    const options = merged.get("prompt_cache_options");
    if (key == null and retention == null and options == null) return null;
    if (options) |raw| {
        const spec = try asObject(raw);
        try only(&spec, .{ "mode", "ttl" });
        if (!is(spec.get("mode") orelse .null, "explicit")) return invalid;
        if (present(&spec, "ttl") != null or key != null or retention != null) return invalid;
        return try object(arena, .{ .mode = str("off") });
    }
    var cache: Object = .{};
    try cache.put(arena, "mode", str("auto"));
    if (key) |value| try cache.put(arena, "key", value);
    if (retention) |value| {
        if (!is(value, "24h")) return invalid;
        try cache.put(arena, "retention", str("long"));
    }
    return .{ .object = cache };
}

fn readTools(arena: Allocator, raw: ?Value) Error!runtime.Array {
    var tools = runtime.Array.init(arena);
    const value = raw orelse return tools;
    if (value != .array or value.array.items.len > 1024) return invalid;
    var nested = runtime.Array.init(arena);
    for (value.array.items) |item| {
        const tool = try asObject(item);
        const kind = try asString(tool.get("type") orelse return invalid);
        if (kind.len == 0) return invalid;
        const function = tool.get("function");
        if (is(str(kind), "function")) {
            if (function) |fields| {
                try only(&tool, .{ "type", "function" });
                const spec = try asObject(fields);
                try only(&spec, .{ "name", "description", "parameters", "strict" });
                try nested.append(try readFunction(arena, &spec));
            } else {
                try tools.append(try readFunction(arena, &tool));
            }
        } else {
            if (function != null) return invalid;
            var config: Object = .{};
            for (tool.keys(), tool.values()) |key, field| {
                if (!eql(key, "type")) try config.put(arena, key, field);
            }
            var builtin = (try object(arena, .{ .type = str("builtin"), .name = str(kind) })).object;
            if (config.count() > 0) try builtin.put(arena, "config", .{ .object = config });
            try tools.append(.{ .object = builtin });
        }
    }
    try tools.appendSlice(nested.items);
    return tools;
}

fn readFunction(arena: Allocator, spec: *const Object) Error!Value {
    if (spec.get("strict")) |value| if (value == .bool and value.bool) return invalid;
    const name = try asString(spec.get("name") orelse return invalid);
    var tool = (try object(arena, .{ .type = str("function"), .name = str(name) })).object;
    if (present(spec, "description")) |value| try tool.put(arena, "description", value);
    if (present(spec, "parameters")) |value| {
        const parameters = try asObject(value);
        if (!spec.contains("type") or parameters.count() > 0) try tool.put(arena, "parameters", value);
    }
    return .{ .object = tool };
}

fn readChoice(arena: Allocator, merged: *const Object, tools: []const Value) Error!?Value {
    const raw = merged.get("tool_choice");
    const parallel = merged.get("parallel_tool_calls");
    if (raw == null and parallel == null) return null;
    var mode = str("auto");
    var names = runtime.Array.init(arena);
    if (raw) |value| switch (value) {
        .string => mode = value,
        .object => |spec| {
            const kind = spec.get("type") orelse return invalid;
            mode = str("required");
            if (spec.get("function")) |_| {
                try only(&spec, .{ "type", "function" });
                try names.append(try choiceFunction(&spec));
            } else if (is(kind, "allowed_tools")) {
                try only(&spec, .{ "type", "allowed_tools" });
                const allowed = try asObject(spec.get("allowed_tools") orelse return invalid);
                try only(&allowed, .{ "mode", "tools" });
                mode = allowed.get("mode") orelse return invalid;
                const entries = allowed.get("tools") orelse return invalid;
                if (entries != .array or entries.array.items.len == 0 or
                    entries.array.items.len > 1024) return invalid;
                for (entries.array.items) |entry| {
                    const declaration = try asObject(entry);
                    try names.append(try choiceFunction(&declaration));
                }
            } else if (is(kind, "function")) {
                try names.append(str(try asString(spec.get("name") orelse return invalid)));
            } else {
                const name = try asString(kind);
                for (tools) |tool| {
                    if (is(tool.object.get("type").?, "builtin") and
                        is(tool.object.get("name").?, name)) break;
                } else return invalid;
                try names.append(kind);
            }
        },
        else => return invalid,
    };
    if (!in(try asString(mode), .{ "auto", "required", "none" })) return invalid;
    var choice = (try object(arena, .{ .mode = mode })).object;
    if (names.items.len > 0) try choice.put(arena, "allowed", .{ .array = names });
    if (parallel) |value| try choice.put(arena, "parallel", value);
    return .{ .object = choice };
}

fn choiceFunction(spec: *const Object) Error!Value {
    if (!is(spec.get("type") orelse .null, "function")) return invalid;
    const function = try asObject(spec.get("function") orelse return invalid);
    try only(&function, .{"name"});
    return str(try asString(function.get("name") orelse return invalid));
}

fn present(map: *const Object, key: []const u8) ?Value {
    const value = map.get(key) orelse return null;
    return if (value == .null) null else value;
}

fn object(arena: Allocator, fields: anytype) Allocator.Error!Value {
    var map: Object = .{};
    inline for (comptime std.meta.fieldNames(@TypeOf(fields))) |name| {
        try map.put(arena, name, @field(fields, name));
    }
    return .{ .object = map };
}

fn array(arena: Allocator, items: []const Value) Allocator.Error!Value {
    var list = runtime.Array.init(arena);
    try list.appendSlice(items);
    return .{ .array = list };
}

fn asObject(value: Value) Error!Object {
    return if (value == .object) value.object else invalid;
}

fn asString(value: Value) Error![]const u8 {
    return if (value == .string) value.string else invalid;
}

fn str(value: []const u8) Value {
    return .{ .string = value };
}

fn is(value: Value, word: []const u8) bool {
    return value == .string and eql(value.string, word);
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn in(key: []const u8, comptime keys: anytype) bool {
    inline for (keys) |candidate| if (eql(key, candidate)) return true;
    return false;
}

fn only(map: *const Object, comptime keys: anytype) Error!void {
    for (map.keys()) |key| if (!in(key, keys)) return invalid;
}

fn sameInteger(a: Value, b: Value) bool {
    return switch (a) {
        .integer => b == .integer and a.integer == b.integer,
        .number_string => b == .number_string and eql(a.number_string, b.number_string),
        else => false,
    };
}
