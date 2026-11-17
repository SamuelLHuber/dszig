//! DSPy 3.5 tool protocol values and native function-tool boundary conversion.
//! Per-call provider continuation is preserved; malformed JSON repair is not.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const value = @import("value.zig");
const types = @import("types.zig");
const Record = value.Record;
const adapter = @import("adapter.zig");
const inference = @import("inference.zig");
const pyjson = @import("pyjson.zig");

pub const Error = types.Error || error{JsonTooDeep};

pub const ToolCall = struct {
    id: ?[]const u8 = null,
    name: []const u8,
    args: Record = .empty,
    continuation: []const Record = &.{},

    fn fromValue(arena: Allocator, input: Value) types.Error!ToolCall {
        if (input != .object) return error.ParseValue;
        const record = &input.object;
        const function = record.get("function");
        var name = record.get("name");
        var args: Value = .null;
        if (function) |f| {
            if (try truthy(f) and f != .object) return error.ParseValue;
            if (f == .object) {
                if (f.object.get("name")) |n| {
                    if (try truthy(n)) name = n;
                }
                args = f.object.get("arguments") orelse .null;
            }
        } else {
            args = record.get("args") orelse record.get("arguments") orelse .null;
        }
        const tool_name = name orelse return error.ParseValue;
        if (tool_name != .string) return error.ParseValue;
        if (args == .string) {
            args = std.json.parseFromSliceLeaky(Value, arena, args.string, .{}) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseValue,
            };
            if (args != .object) return error.ParseValue;
        } else if (args != .object) args = .{ .object = .empty };

        var id = record.get("id");
        if (id == null or !try truthy(id.?)) id = record.get("call_id");
        const call_id: ?[]const u8 = if (id) |i| switch (i) {
            .null => null,
            .string => |s| s,
            else => return error.ParseValue,
        } else null;
        var continuation: []const Record = &.{};
        if (record.get("continuation")) |states| {
            if (states != .array) return error.ParseValue;
            const out = try arena.alloc(Record, states.array.items.len);
            for (states.array.items, out) |state, *o| {
                if (state != .object) return error.ParseValue;
                o.* = state.object;
            }
            continuation = out;
        }
        return .{ .id = call_id, .name = tool_name.string, .args = args.object, .continuation = continuation };
    }

    pub fn toValue(self: ToolCall, arena: Allocator, protocol_metadata: bool) Allocator.Error!Value {
        var out: Record = .empty;
        try out.ensureTotalCapacity(arena, if (protocol_metadata) 4 else 2);
        if (protocol_metadata) out.putAssumeCapacity("id", if (self.id) |id| .{ .string = id } else .null);
        out.putAssumeCapacity("name", .{ .string = self.name });
        out.putAssumeCapacity("args", .{ .object = self.args });
        if (protocol_metadata and self.continuation.len > 0) {
            var states: @import("runtime_value.zig").Array = .init(arena);
            try states.ensureTotalCapacity(self.continuation.len);
            for (self.continuation) |state| states.appendAssumeCapacity(.{ .object = state });
            out.putAssumeCapacity("continuation", .{ .array = states });
        }
        return .{ .object = out };
    }
};

pub const ToolCallResult = struct {
    call_id: ?[]const u8 = null,
    name: []const u8,
    value: Value,
    is_error: bool = false,

    fn fromValue(input: Value) types.Error!ToolCallResult {
        if (input != .object) return error.ParseValue;
        const name = input.object.get("name") orelse return error.ParseValue;
        const result = input.object.get("value") orelse return error.ParseValue;
        if (name != .string) return error.ParseValue;
        var id: ?[]const u8 = null;
        if (input.object.get("call_id")) |call_id| {
            if (call_id != .null) {
                if (call_id != .string) return error.ParseValue;
                id = call_id.string;
            }
        }
        var is_error = false;
        if (input.object.get("is_error")) |flag| {
            if (flag != .bool) return error.ParseValue;
            is_error = flag.bool;
        }
        return .{ .call_id = id, .name = name.string, .value = result, .is_error = is_error };
    }

    pub fn toValue(self: ToolCallResult, arena: Allocator) Allocator.Error!Value {
        var out: Record = .empty;
        try out.ensureTotalCapacity(arena, 4);
        out.putAssumeCapacity("call_id", if (self.call_id) |id| .{ .string = id } else .null);
        out.putAssumeCapacity("name", .{ .string = self.name });
        out.putAssumeCapacity("value", self.value);
        out.putAssumeCapacity("is_error", .{ .bool = self.is_error });
        return .{ .object = out };
    }
};

pub const ToolCallResults = struct {
    tool_call_results: []const ToolCallResult = &.{},

    pub fn fromValue(arena: Allocator, input: Value) types.Error!ToolCallResults {
        const items: []const Value = switch (input) {
            .array => |a| a.items,
            .object => |o| blk: {
                if (o.get("tool_call_results")) |results| {
                    if (results != .array) return error.ParseValue;
                    break :blk results.array.items;
                }
                if (o.contains("name") and o.contains("value")) {
                    const result = try arena.alloc(ToolCallResult, 1);
                    result[0] = try ToolCallResult.fromValue(input);
                    return .{ .tool_call_results = result };
                }
                return error.ParseValue;
            },
            else => return error.ParseValue,
        };
        const results = try arena.alloc(ToolCallResult, items.len);
        for (items, results) |item, *result| result.* = try ToolCallResult.fromValue(item);
        return .{ .tool_call_results = results };
    }

    pub fn toValue(self: ToolCallResults, arena: Allocator) Allocator.Error!Value {
        var results: @import("runtime_value.zig").Array = .init(arena);
        try results.ensureTotalCapacity(self.tool_call_results.len);
        for (self.tool_call_results) |result| results.appendAssumeCapacity(try result.toValue(arena));
        var out: Record = .empty;
        try out.put(arena, "tool_call_results", .{ .array = results });
        return .{ .object = out };
    }
};

pub const ToolCalls = struct {
    tool_calls: []const ToolCall = &.{},
    tool_call_results: ?ToolCallResults = null,

    pub const description =
        "Tool calls must be a JSON object with `tool_calls`, a list of calls. " ++
        "Each call must include `name` and `args`. " ++
        "Example: {\"tool_calls\": [{\"name\": \"search\", \"args\": {\"query\": \"cats\"}}]}";

    /// DSPy's prompt schema intentionally hides IDs, continuations, and results.
    pub const schema =
        \\{"type": "object", "$defs": {"ToolCall": {"type": "object", "properties": {"args": {"type": "object", "additionalProperties": true, "title": "Args"}, "name": {"type": "string", "title": "Name"}}, "required": ["name", "args"], "title": "ToolCall"}}, "properties": {"tool_calls": {"type": "array", "items": {"$ref": "#/$defs/ToolCall"}, "title": "Tool Calls"}}, "required": ["tool_calls"], "title": "ToolCalls"}
    ;

    pub fn validate(arena: Allocator, input: Value) types.Error!Value {
        return (try fromValue(arena, input)).toValue(arena);
    }

    pub fn fromValue(arena: Allocator, input: Value) types.Error!ToolCalls {
        if (input == .null) return .{};
        var results: ?ToolCallResults = null;
        const items: []const Value = switch (input) {
            .array => |a| blk: {
                for (a.items) |item| if (!isCall(item)) return error.ParseValue;
                break :blk a.items;
            },
            .object => |o| blk: {
                if (o.get("tool_calls")) |calls| {
                    if (calls == .null) return .{};
                    if (calls != .array) return error.ParseValue;
                    if (o.get("tool_call_results")) |r| {
                        if (r != .null) results = try ToolCallResults.fromValue(arena, r);
                    }
                    break :blk calls.array.items;
                }
                if (isCall(input)) {
                    const calls = try arena.alloc(ToolCall, 1);
                    calls[0] = try ToolCall.fromValue(arena, input);
                    return .{ .tool_calls = calls };
                }
                if (o.count() == 0) return .{};
                return error.ParseValue;
            },
            else => return error.ParseValue,
        };
        const calls = try arena.alloc(ToolCall, items.len);
        for (items, calls) |item, *call| call.* = try ToolCall.fromValue(arena, item);
        return .{ .tool_calls = calls, .tool_call_results = results };
    }

    /// Serialization for retained History, not the metadata-free prompt value.
    pub fn toValue(self: ToolCalls, arena: Allocator) Allocator.Error!Value {
        return self.renderValue(arena, true);
    }

    pub fn formatValue(self: ToolCalls, arena: Allocator) Allocator.Error!Value {
        return self.renderValue(arena, false);
    }

    fn renderValue(self: ToolCalls, arena: Allocator, protocol_metadata: bool) Allocator.Error!Value {
        var calls: @import("runtime_value.zig").Array = .init(arena);
        try calls.ensureTotalCapacity(self.tool_calls.len);
        for (self.tool_calls) |call| calls.appendAssumeCapacity(try call.toValue(arena, protocol_metadata));
        var out: Record = .empty;
        try out.put(arena, "tool_calls", .{ .array = calls });
        if (self.tool_call_results) |results| try out.put(arena, "tool_call_results", try results.toValue(arena));
        return .{ .object = out };
    }

    pub fn ensureIds(self: ToolCalls, arena: Allocator, turn_index: usize) Allocator.Error!ToolCalls {
        var missing = false;
        for (self.tool_calls) |call| if (call.id == null) {
            missing = true;
            break;
        };
        if (!missing) return .{ .tool_calls = self.tool_calls };
        const calls = try arena.dupe(ToolCall, self.tool_calls);
        for (calls, 0..) |*call, call_index| {
            if (call.id == null) call.id = try std.fmt.allocPrint(arena, "call_{d}_{d}", .{ turn_index, call_index });
        }
        return .{ .tool_calls = calls };
    }

    pub fn named(self: ToolCalls, arena: Allocator, name: []const u8) Allocator.Error!ToolCalls {
        var count: usize = 0;
        for (self.tool_calls) |call| if (std.mem.eql(u8, call.name, name)) {
            count += 1;
        };
        if (count == self.tool_calls.len) return .{ .tool_calls = self.tool_calls };
        const calls = try arena.alloc(ToolCall, count);
        var index: usize = 0;
        for (self.tool_calls) |call| if (std.mem.eql(u8, call.name, name)) {
            calls[index] = call;
            index += 1;
        };
        return .{ .tool_calls = calls };
    }
};

/// Render Tool records while preserving their already-serialized text values
/// in saved/bootstrap demos. Native declaration conversion remains strict.
pub fn formatCatalogValue(arena: Allocator, input: Value) adapter.Error!Value {
    if (input != .array) return error.AdapterParse;
    var needs_formatting = false;
    for (input.array.items) |item| {
        if (item != .string) {
            needs_formatting = true;
            break;
        }
    }
    if (!needs_formatting) return input;
    var out: @import("runtime_value.zig").Array = .init(arena);
    try out.ensureTotalCapacity(input.array.items.len);
    for (input.array.items) |item| out.appendAssumeCapacity(if (item == .string) item else .{ .string = try formatToolValue(arena, item) });
    return .{ .array = out };
}

pub fn formatToolValue(arena: Allocator, input: Value) adapter.Error![]const u8 {
    if (input != .object) return error.AdapterParse;
    const name = input.object.get("name") orelse return error.AdapterParse;
    const args = input.object.get("args") orelse return error.AdapterParse;
    const desc = input.object.get("desc") orelse Value{ .string = "" };
    if (name != .string or args != .object or desc != .string) return error.AdapterParse;
    var out: std.Io.Writer.Allocating = .init(arena);
    const w = &out.writer;
    w.writeAll(name.string) catch return error.OutOfMemory;
    if (desc.string.len > 0) {
        w.writeAll(", whose description is <desc>") catch return error.OutOfMemory;
        for (desc.string) |c| {
            if (c == '\n') {
                w.writeAll("  ") catch return error.OutOfMemory;
            } else w.writeByte(c) catch return error.OutOfMemory;
        }
        w.writeAll("</desc>.") catch return error.OutOfMemory;
    } else w.writeByte('.') catch return error.OutOfMemory;
    w.writeAll(" It takes arguments ") catch return error.OutOfMemory;
    pyjson.writeRepr(w, args, 0) catch |err| return renderError(err);
    w.writeByte('.') catch return error.OutOfMemory;
    return out.written();
}

/// Match Tool.as_function_tool: argument-local refs move under their property,
/// including JSON-pointer escaping and URL quoting of the property token.
pub fn functionToolsFromValue(arena: Allocator, input: Value) Error![]const inference.FunctionTool {
    if (input != .array) return error.ParseValue;
    const tools = try arena.alloc(inference.FunctionTool, input.array.items.len);
    for (input.array.items, tools) |item, *out| {
        if (item != .object) return error.ParseValue;
        const name = item.object.get("name") orelse return error.ParseValue;
        const args = item.object.get("args") orelse return error.ParseValue;
        const desc = item.object.get("desc") orelse Value{ .string = "" };
        if (name != .string or args != .object or desc != .string) return error.ParseValue;
        var properties: Record = .empty;
        try properties.ensureTotalCapacity(arena, args.object.count());
        var required: @import("runtime_value.zig").Array = .init(arena);
        try required.ensureTotalCapacity(args.object.count());
        for (args.object.keys(), args.object.values()) |arg_name, arg_schema| {
            if (arg_schema != .object) return error.ParseValue;
            const prefix = try propertyPrefix(arena, arg_name);
            properties.putAssumeCapacity(arg_name, try rebaseRefs(arena, arg_schema, prefix, 0));
            if (!arg_schema.object.contains("default")) required.appendAssumeCapacity(.{ .string = arg_name });
        }
        var parameters: Record = .empty;
        try parameters.ensureTotalCapacity(arena, 3);
        parameters.putAssumeCapacity("type", .{ .string = "object" });
        parameters.putAssumeCapacity("properties", .{ .object = properties });
        parameters.putAssumeCapacity("required", .{ .array = required });
        out.* = .{ .name = name.string, .description = desc.string, .parameters = .{ .object = parameters } };
    }
    return tools;
}

/// Native transport IDs must be supplied by the provider, never synthesized.
pub fn callsFromNative(arena: Allocator, input: []const inference.ToolCall) types.Error!Value {
    const calls = try arena.alloc(ToolCall, input.len);
    for (input, calls) |native, *call| {
        if (native.id.len == 0) return error.ParseValue;
        const args = std.json.parseFromSliceLeaky(Value, arena, native.arguments, .{}) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        };
        if (args != .object) return error.ParseValue;
        const continuation = try arena.alloc(Record, native.continuation.len);
        for (native.continuation, continuation) |state, *out| {
            out.* = .empty;
            try out.ensureTotalCapacity(arena, 3);
            out.putAssumeCapacity("provider", .{ .string = state.provider });
            out.putAssumeCapacity("kind", .{ .string = state.kind });
            out.putAssumeCapacity("data", state.data);
        }
        call.* = .{ .id = native.id, .name = native.name, .args = args.object, .continuation = continuation };
    }
    return (ToolCalls{ .tool_calls = calls }).toValue(arena);
}

pub fn callsToNative(arena: Allocator, input: []const ToolCall) Error![]const inference.ToolCall {
    const calls = try arena.alloc(inference.ToolCall, input.len);
    for (input, calls) |call, *native| {
        const id = call.id orelse return error.ParseValue;
        if (id.len == 0) return error.ParseValue;
        var arguments: std.Io.Writer.Allocating = .init(arena);
        pyjson.writeJson(&arguments.writer, .{ .object = call.args }, .minified) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep => error.JsonTooDeep,
            error.ParseValue => error.ParseValue,
        };
        const continuation = try arena.alloc(inference.Continuation, call.continuation.len);
        for (call.continuation, continuation) |state, *out| {
            const provider = state.get("provider") orelse return error.ParseValue;
            const kind = state.get("kind") orelse return error.ParseValue;
            const data = state.get("data") orelse return error.ParseValue;
            if (provider != .string or kind != .string or data != .object) return error.ParseValue;
            if (provider.string.len == 0 or kind.string.len == 0) return error.ParseValue;
            out.* = .{ .provider = provider.string, .kind = kind.string, .data = data };
        }
        native.* = .{ .id = id, .name = call.name, .arguments = arguments.written(), .continuation = continuation };
    }
    return calls;
}

fn propertyPrefix(arena: Allocator, name: []const u8) Allocator.Error![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    out.writer.writeAll("#/properties/") catch return error.OutOfMemory;
    const hex = "0123456789ABCDEF";
    for (name) |c| {
        if (c == '~') {
            out.writer.writeAll("~0") catch return error.OutOfMemory;
        } else if (c == '/') {
            out.writer.writeAll("~1") catch return error.OutOfMemory;
        } else if (std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.') {
            out.writer.writeByte(c) catch return error.OutOfMemory;
        } else {
            out.writer.writeAll(&.{ '%', hex[c >> 4], hex[c & 15] }) catch return error.OutOfMemory;
        }
    }
    return out.written();
}

fn rebaseRefs(arena: Allocator, input: Value, prefix: []const u8, depth: usize) Error!Value {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (input != .object or input.object.contains("$id")) return input;
    var out = try input.object.clone(arena);
    if (out.get("$ref")) |ref| {
        if (ref == .string and (std.mem.eql(u8, ref.string, "#") or std.mem.startsWith(u8, ref.string, "#/"))) {
            try out.put(arena, "$ref", .{ .string = try std.fmt.allocPrint(arena, "{s}{s}", .{ prefix, ref.string[1..] }) });
        }
    }
    for ([_][]const u8{ "$defs", "definitions", "properties", "patternProperties", "dependentSchemas", "dependencies" }) |key| {
        if (input.object.get(key)) |schemas| {
            if (schemas != .object) return error.ParseValue;
            var rebased: Record = .empty;
            try rebased.ensureTotalCapacity(arena, schemas.object.count());
            for (schemas.object.keys(), schemas.object.values()) |name, schema| {
                rebased.putAssumeCapacity(name, try rebaseRefs(arena, schema, prefix, depth + 1));
            }
            try out.put(arena, key, .{ .object = rebased });
        }
    }
    for ([_][]const u8{
        "items",            "additionalItems",       "additionalProperties", "contains", "propertyNames",
        "unevaluatedItems", "unevaluatedProperties", "not",                  "if",       "then",
        "else",             "allOf",                 "anyOf",                "oneOf",    "prefixItems",
    }) |key| {
        if (input.object.get(key)) |schema| {
            const rebased = if (schema == .array) blk: {
                var items: @import("runtime_value.zig").Array = .init(arena);
                try items.ensureTotalCapacity(schema.array.items.len);
                for (schema.array.items) |item| items.appendAssumeCapacity(try rebaseRefs(arena, item, prefix, depth + 1));
                break :blk Value{ .array = items };
            } else try rebaseRefs(arena, schema, prefix, depth + 1);
            try out.put(arena, key, rebased);
        }
    }
    return .{ .object = out };
}

fn renderError(err: pyjson.Error) adapter.Error {
    return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
        error.ParseValue => error.AdapterParse,
    };
}

fn truthy(input: Value) error{ParseValue}!bool {
    return switch (input) {
        .null => false,
        .bool => |b| b,
        .integer => |n| n != 0,
        .float => |n| n != 0,
        .number_string => |s| (std.fmt.parseFloat(f64, s) catch return true) != 0,
        .string => |s| s.len > 0,
        .array => |a| a.items.len > 0,
        .object => |o| o.count() > 0,
        .custom => |custom| try custom.truthy(),
    };
}

fn isCall(input: Value) bool {
    if (input != .object) return false;
    return input.object.contains("function") or
        (input.object.contains("name") and (input.object.contains("args") or input.object.contains("arguments")));
}
