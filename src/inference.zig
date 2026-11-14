//! Provider-neutral native function and structured-output metadata.
//! Tool arguments remain their actual provider JSON text until the caller
//! parses them; transport code never invents call identifiers or arguments.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const MetadataError = error{BadMetadata} || Allocator.Error;

pub const FunctionTool = struct {
    name: []const u8,
    description: []const u8 = "",
    parameters: Value,
    strict: ?bool = null,
};

pub const Continuation = struct {
    provider: []const u8,
    kind: []const u8,
    data: Value,
};

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8,
    continuation: []const Continuation = &.{},
};

pub const JsonSchema = struct {
    name: []const u8,
    schema: Value,
    description: ?[]const u8 = null,
    strict: bool = true,
};

pub const ResponseFormat = union(enum) {
    text,
    json_object,
    json_schema: JsonSchema,
};

pub const ToolChoice = union(enum) {
    auto,
    none,
    required,
    function: []const u8,
    allowed_tools: struct {
        mode: enum { auto, required },
        names: []const []const u8,
    },
};

pub fn cloneTools(arena: Allocator, tools: []const FunctionTool) Allocator.Error![]const FunctionTool {
    const out = try arena.alloc(FunctionTool, tools.len);
    for (tools, out) |tool, *copy| {
        copy.* = .{
            .name = try arena.dupe(u8, tool.name),
            .description = try arena.dupe(u8, tool.description),
            .parameters = try cloneJsonValue(arena, tool.parameters),
            .strict = tool.strict,
        };
    }
    return out;
}

pub fn cloneToolCalls(allocator: Allocator, calls: []const ToolCall) Allocator.Error![]const ToolCall {
    const out = try allocator.alloc(ToolCall, calls.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |call| freeToolCall(allocator, call);
        allocator.free(out);
    }
    for (calls, out) |call, *copy| {
        const id = try allocator.dupe(u8, call.id);
        errdefer allocator.free(id);
        const name = try allocator.dupe(u8, call.name);
        errdefer allocator.free(name);
        const arguments = try allocator.dupe(u8, call.arguments);
        errdefer allocator.free(arguments);
        const continuation = try cloneContinuations(allocator, call.continuation);
        copy.* = .{ .id = id, .name = name, .arguments = arguments, .continuation = continuation };
        initialized += 1;
    }
    return out;
}

fn freeToolCall(allocator: Allocator, call: ToolCall) void {
    allocator.free(call.id);
    allocator.free(call.name);
    allocator.free(call.arguments);
    freeContinuations(allocator, call.continuation);
}

pub fn freeToolCalls(allocator: Allocator, calls: []const ToolCall) void {
    for (calls) |call| freeToolCall(allocator, call);
    allocator.free(calls);
}

/// General-allocator JSON ownership, including cleanup on any failed child
/// clone. Completion caches cannot rely on a surrounding arena to recover it.
pub fn cloneJsonValue(allocator: Allocator, input: Value) Allocator.Error!Value {
    return switch (input) {
        .string => |text| .{ .string = try allocator.dupe(u8, text) },
        .number_string => |text| .{ .number_string = try allocator.dupe(u8, text) },
        .array => |array| blk: {
            var out: std.json.Array = .init(allocator);
            errdefer {
                for (out.items) |item| freeJsonValue(allocator, item);
                out.deinit();
            }
            try out.ensureTotalCapacity(array.items.len);
            for (array.items) |item| out.appendAssumeCapacity(try cloneJsonValue(allocator, item));
            break :blk .{ .array = out };
        },
        .object => |object| blk: {
            var out: std.json.ObjectMap = .empty;
            errdefer {
                var iterator = out.iterator();
                while (iterator.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    freeJsonValue(allocator, entry.value_ptr.*);
                }
                out.deinit(allocator);
            }
            try out.ensureTotalCapacity(allocator, object.count());
            var iterator = object.iterator();
            while (iterator.next()) |entry| {
                const key = try allocator.dupe(u8, entry.key_ptr.*);
                errdefer allocator.free(key);
                const child = try cloneJsonValue(allocator, entry.value_ptr.*);
                out.putAssumeCapacity(key, child);
            }
            break :blk .{ .object = out };
        },
        else => input,
    };
}

pub fn freeJsonValue(allocator: Allocator, input: Value) void {
    switch (input) {
        .string => |text| allocator.free(text),
        .number_string => |text| allocator.free(text),
        .array => |array| {
            var owned = array;
            for (owned.items) |item| freeJsonValue(allocator, item);
            owned.deinit();
        },
        .object => |object| {
            var owned = object;
            var iterator = owned.iterator();
            while (iterator.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                freeJsonValue(allocator, entry.value_ptr.*);
            }
            owned.deinit(allocator);
        },
        else => {},
    }
}

pub fn cloneContinuations(allocator: Allocator, input: []const Continuation) Allocator.Error![]const Continuation {
    const out = try allocator.alloc(Continuation, input.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |state| freeContinuation(allocator, state);
        allocator.free(out);
    }
    for (input, out) |state, *copy| {
        const provider = try allocator.dupe(u8, state.provider);
        errdefer allocator.free(provider);
        const kind = try allocator.dupe(u8, state.kind);
        errdefer allocator.free(kind);
        const data = try cloneJsonValue(allocator, state.data);
        copy.* = .{ .provider = provider, .kind = kind, .data = data };
        initialized += 1;
    }
    return out;
}

fn freeContinuation(allocator: Allocator, state: Continuation) void {
    allocator.free(state.provider);
    allocator.free(state.kind);
    freeJsonValue(allocator, state.data);
}

pub fn freeContinuations(allocator: Allocator, input: []const Continuation) void {
    for (input) |state| freeContinuation(allocator, state);
    allocator.free(input);
}

pub fn continuationsValue(arena: Allocator, input: []const Continuation) Allocator.Error!Value {
    var out: std.json.Array = .init(arena);
    try out.ensureTotalCapacity(input.len);
    for (input) |state| {
        var object: std.json.ObjectMap = .empty;
        try object.put(arena, "provider", .{ .string = state.provider });
        try object.put(arena, "kind", .{ .string = state.kind });
        try object.put(arena, "data", state.data);
        out.appendAssumeCapacity(.{ .object = object });
    }
    return .{ .array = out };
}

pub fn continuationsFromValue(arena: Allocator, input: Value) MetadataError![]const Continuation {
    if (input != .array) return error.BadMetadata;
    const out = try arena.alloc(Continuation, input.array.items.len);
    for (input.array.items, out) |state, *copy| {
        if (state != .object) return error.BadMetadata;
        const data = state.object.get("data") orelse return error.BadMetadata;
        if (data != .object) return error.BadMetadata;
        const provider = try stringField(state, "provider");
        const kind = try stringField(state, "kind");
        if (provider.len == 0 or kind.len == 0) return error.BadMetadata;
        copy.* = .{
            .provider = try arena.dupe(u8, provider),
            .kind = try arena.dupe(u8, kind),
            .data = try cloneJsonValue(arena, data),
        };
    }
    return out;
}

pub fn continuationsEqual(a: []const Continuation, b: []const Continuation) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x.provider, y.provider) or !std.mem.eql(u8, x.kind, y.kind) or
            !jsonEqual(x.data, y.data)) return false;
    }
    return true;
}

pub fn cloneResponseFormat(arena: Allocator, format: ResponseFormat) Allocator.Error!ResponseFormat {
    return switch (format) {
        .text, .json_object => format,
        .json_schema => |schema| .{ .json_schema = .{
            .name = try arena.dupe(u8, schema.name),
            .schema = try cloneJsonValue(arena, schema.schema),
            .description = if (schema.description) |description| try arena.dupe(u8, description) else null,
            .strict = schema.strict,
        } },
    };
}

pub fn cloneToolChoice(arena: Allocator, choice: ToolChoice) Allocator.Error!ToolChoice {
    return switch (choice) {
        .auto, .none, .required => choice,
        .function => |name| .{ .function = try arena.dupe(u8, name) },
        .allowed_tools => |allowed| blk: {
            const names = try arena.alloc([]const u8, allowed.names.len);
            for (names, allowed.names) |*copy, name| copy.* = try arena.dupe(u8, name);
            break :blk .{ .allowed_tools = .{ .mode = allowed.mode, .names = names } };
        },
    };
}

/// Provider-neutral form, before Chat's json_schema wrapper or Responses'
/// text.format wrapper is applied. Used by caches and faithful transcripts.
pub fn responseFormatValue(arena: Allocator, format: ResponseFormat) Allocator.Error!Value {
    var out: std.json.ObjectMap = .empty;
    try out.put(arena, "type", .{ .string = @tagName(format) });
    if (format == .json_schema) {
        const schema = format.json_schema;
        try out.put(arena, "name", .{ .string = schema.name });
        try out.put(arena, "schema", schema.schema);
        if (schema.description) |description| try out.put(arena, "description", .{ .string = description });
        try out.put(arena, "strict", .{ .bool = schema.strict });
    }
    return .{ .object = out };
}

/// Canonical choice semantics; a forced function is required + one allowed
/// name, not an auto choice that the model is permitted to ignore.
pub fn toolChoiceValue(arena: Allocator, choice: ToolChoice) Allocator.Error!Value {
    var out: std.json.ObjectMap = .empty;
    var allowed: std.json.Array = .init(arena);
    const mode: []const u8 = switch (choice) {
        .auto => "auto",
        .none => "none",
        .required => "required",
        .function => |name| blk: {
            try allowed.append(.{ .string = name });
            break :blk "required";
        },
        .allowed_tools => |selection| blk: {
            for (selection.names) |name| try allowed.append(.{ .string = name });
            break :blk @tagName(selection.mode);
        },
    };
    try out.put(arena, "mode", .{ .string = mode });
    try out.put(arena, "allowed", .{ .array = allowed });
    return .{ .object = out };
}

pub fn toolsValue(arena: Allocator, tools: []const FunctionTool) Allocator.Error!Value {
    var out: std.json.Array = .init(arena);
    try out.ensureTotalCapacity(tools.len);
    for (tools) |tool| {
        var object: std.json.ObjectMap = .empty;
        try object.put(arena, "name", .{ .string = tool.name });
        try object.put(arena, "description", .{ .string = tool.description });
        try object.put(arena, "parameters", tool.parameters);
        try object.put(arena, "strict", if (tool.strict) |strict| .{ .bool = strict } else .null);
        out.appendAssumeCapacity(.{ .object = object });
    }
    return .{ .array = out };
}

pub fn toolCallsValue(arena: Allocator, calls: []const ToolCall) Allocator.Error!Value {
    var out: std.json.Array = .init(arena);
    try out.ensureTotalCapacity(calls.len);
    for (calls) |call| {
        var object: std.json.ObjectMap = .empty;
        try object.put(arena, "id", .{ .string = call.id });
        try object.put(arena, "name", .{ .string = call.name });
        try object.put(arena, "arguments", .{ .string = call.arguments });
        try object.put(arena, "continuation", try continuationsValue(arena, call.continuation));
        out.appendAssumeCapacity(.{ .object = object });
    }
    return .{ .array = out };
}

/// Owning canonical metadata readers; returned slices and schemas live in
/// `arena`, independently of the JSON document supplied by the caller.
pub fn responseFormatFromValue(arena: Allocator, input: Value) MetadataError!ResponseFormat {
    if (input != .object) return error.BadMetadata;
    const kind = try stringField(input, "type");
    if (std.mem.eql(u8, kind, "text")) return .text;
    if (std.mem.eql(u8, kind, "json_object")) return .json_object;
    if (!std.mem.eql(u8, kind, "json_schema")) return error.BadMetadata;
    const schema = input.object.get("schema") orelse return error.BadMetadata;
    if (schema != .object) return error.BadMetadata;
    const description = if (input.object.get("description")) |description_value|
        if (description_value == .null) null else try string(description_value)
    else
        null;
    const strict = if (input.object.get("strict")) |strict_value| try boolean(strict_value) else true;
    return .{ .json_schema = .{
        .name = try arena.dupe(u8, try stringField(input, "name")),
        .schema = try cloneJsonValue(arena, schema),
        .description = if (description) |text| try arena.dupe(u8, text) else null,
        .strict = strict,
    } };
}

pub fn toolChoiceFromValue(arena: Allocator, input: Value) MetadataError!ToolChoice {
    if (input != .object) return error.BadMetadata;
    const mode = try stringField(input, "mode");
    const allowed_value = input.object.get("allowed") orelse return error.BadMetadata;
    if (allowed_value != .array) return error.BadMetadata;
    const allowed = allowed_value.array.items;
    if (std.mem.eql(u8, mode, "none")) {
        if (allowed.len != 0) return error.BadMetadata;
        return .none;
    }
    const required = std.mem.eql(u8, mode, "required");
    if (!required and !std.mem.eql(u8, mode, "auto")) return error.BadMetadata;
    if (allowed.len == 0) return if (required) .required else .auto;
    if (required and allowed.len == 1) return .{ .function = try arena.dupe(u8, try string(allowed[0])) };
    const names = try arena.alloc([]const u8, allowed.len);
    for (allowed, names) |name, *copy| copy.* = try arena.dupe(u8, try string(name));
    return .{ .allowed_tools = .{ .mode = if (required) .required else .auto, .names = names } };
}

pub fn toolsFromValue(arena: Allocator, input: Value) MetadataError![]const FunctionTool {
    if (input != .array) return error.BadMetadata;
    const out = try arena.alloc(FunctionTool, input.array.items.len);
    for (input.array.items, out) |tool, *copy| {
        if (tool != .object) return error.BadMetadata;
        const parameters = tool.object.get("parameters") orelse return error.BadMetadata;
        if (parameters != .object) return error.BadMetadata;
        const strict = if (tool.object.get("strict")) |strict_value|
            if (strict_value == .null) null else try boolean(strict_value)
        else
            null;
        copy.* = .{
            .name = try arena.dupe(u8, try stringField(tool, "name")),
            .description = try arena.dupe(u8, if (tool.object.get("description")) |description| try string(description) else ""),
            .parameters = try cloneJsonValue(arena, parameters),
            .strict = strict,
        };
    }
    return out;
}

pub fn toolCallsFromValue(arena: Allocator, input: Value) MetadataError![]const ToolCall {
    if (input != .array) return error.BadMetadata;
    const out = try arena.alloc(ToolCall, input.array.items.len);
    for (input.array.items, out) |call, *copy| {
        if (call != .object) return error.BadMetadata;
        copy.* = .{
            .id = try arena.dupe(u8, try stringField(call, "id")),
            .name = try arena.dupe(u8, try stringField(call, "name")),
            .arguments = try arena.dupe(u8, try stringField(call, "arguments")),
            .continuation = if (call.object.get("continuation")) |state| try continuationsFromValue(arena, state) else &.{},
        };
    }
    return out;
}

pub fn responseFormatEqual(a: ResponseFormat, b: ResponseFormat) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    if (a != .json_schema) return true;
    const x = a.json_schema;
    const y = b.json_schema;
    return x.strict == y.strict and std.mem.eql(u8, x.name, y.name) and
        optionalStringEqual(x.description, y.description) and jsonEqual(x.schema, y.schema);
}

pub fn toolChoiceEqual(a: *const ToolChoice, b: *const ToolChoice) bool {
    if (!std.mem.eql(u8, toolChoiceMode(a), toolChoiceMode(b))) return false;
    const x = toolChoiceNames(a);
    const y = toolChoiceNames(b);
    if (x.len != y.len) return false;
    for (x, y) |first, second| {
        if (!std.mem.eql(u8, first, second)) return false;
    }
    return true;
}

fn toolChoiceMode(choice: *const ToolChoice) []const u8 {
    return switch (choice.*) {
        .auto => "auto",
        .none => "none",
        .required, .function => "required",
        .allowed_tools => |allowed| @tagName(allowed.mode),
    };
}

fn toolChoiceNames(choice: *const ToolChoice) []const []const u8 {
    return switch (choice.*) {
        .function => (&choice.function)[0..1],
        .allowed_tools => choice.allowed_tools.names,
        else => &.{},
    };
}

pub fn toolsEqual(a: []const FunctionTool, b: []const FunctionTool) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x.name, y.name) or !std.mem.eql(u8, x.description, y.description) or
            x.strict != y.strict or !jsonEqual(x.parameters, y.parameters)) return false;
    }
    return true;
}

pub fn toolCallsEqual(a: []const ToolCall, b: []const ToolCall) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x.id, y.id) or !std.mem.eql(u8, x.name, y.name) or
            !std.mem.eql(u8, x.arguments, y.arguments) or
            !continuationsEqual(x.continuation, y.continuation)) return false;
    }
    return true;
}

/// Schema booleans are not numbers: Python-style true == 1 would accept a
/// different native request. Object insertion order does not change a schema.
pub fn jsonEqual(a: Value, b: Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => |x| x == b.bool,
        .integer => |x| x == b.integer,
        .float => |x| x == b.float,
        .string => |x| std.mem.eql(u8, x, b.string),
        .number_string => |x| std.mem.eql(u8, x, b.number_string),
        .array => |x| blk: {
            if (x.items.len != b.array.items.len) break :blk false;
            for (x.items, b.array.items) |first, second| {
                if (!jsonEqual(first, second)) break :blk false;
            }
            break :blk true;
        },
        .object => |x| blk: {
            if (x.count() != b.object.count()) break :blk false;
            var iterator = x.iterator();
            while (iterator.next()) |entry| {
                if (!jsonEqual(entry.value_ptr.*, b.object.get(entry.key_ptr.*) orelse break :blk false)) break :blk false;
            }
            break :blk true;
        },
    };
}

fn optionalStringEqual(a: ?[]const u8, b: ?[]const u8) bool {
    if (a) |text| return std.mem.eql(u8, text, b orelse return false);
    return b == null;
}

fn string(input: Value) MetadataError![]const u8 {
    if (input != .string) return error.BadMetadata;
    return input.string;
}

fn stringField(input: Value, name: []const u8) MetadataError![]const u8 {
    if (input != .object) return error.BadMetadata;
    return string(input.object.get(name) orelse return error.BadMetadata);
}

fn boolean(input: Value) MetadataError!bool {
    if (input != .bool) return error.BadMetadata;
    return input.bool;
}
