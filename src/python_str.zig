//! Python `str`/`repr` for declared native types after JSON lowering.
//!
//! The declared Zig type restores pydantic model names, Code specializations,
//! and field order. It never invents Python objects that the lowered JSON
//! has erased: an untyped `Value`/`Record` is rendered with the ordinary
//! `pyjson.writeRepr`, so dynamic nested models inside History records stay
//! plain dicts (an explicit information-loss boundary).

const std = @import("std");
const signature = @import("signature.zig");
const types = @import("types.zig");
const code = @import("code.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Writer = std.Io.Writer;

pub const Error = Allocator.Error || error{ JsonTooDeep, AdapterParse };
const WriteError = pyjson.Error || error{AdapterParse};
pub const Render = *const fn (Allocator, Value) Error![]const u8;

/// Returns null when the ordinary `pyjson.writeStr` already matches Python for
/// `T`, so scalar and plain-container fields keep the allocation-free path.
pub fn renderer(comptime T: type) ?Render {
    if (comptime !needsRendering(T)) return null;
    return struct {
        fn apply(allocator: Allocator, v: Value) Error![]const u8 {
            var out: Writer.Allocating = .init(allocator);
            defer out.deinit();
            write(T, &out.writer, v, .str, 0) catch |err| return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                error.JsonTooDeep => error.JsonTooDeep,
                error.AdapterParse, error.ParseValue => error.AdapterParse,
            };
            return out.toOwnedSlice();
        }
    }.apply;
}

/// True only for declared types whose Python `str`/`repr` differs from the
/// JSON-lowered `Value` rendering: models, Code, and containers of them.
fn needsRendering(comptime T: type) bool {
    if (comptime @import("decision.zig").isDecision(T) or code.isCode(T) or @import("enum.zig").isEnum(T) or @import("literal.zig").isLiteral(T) or T == @import("integer.zig").Integer or @import("decimal_type.zig").isDecimal(T)) return false;
    if (T == signature.Reasoning or T == @import("file.zig").File or
        T == @import("bounding_box.zig").BoundingBox or comptime @import("citation.zig").isCitation(T) or T == signature.StrDict or T == Value or
        T == @import("runtime_value.zig").ObjectMap or T == []const u8 or T == []u8) return false;
    if (comptime types.isDict(T)) return needsRendering(T.dszig_dict_value);
    return switch (@typeInfo(T)) {
        .optional => |o| needsRendering(o.child),
        .pointer => |p| p.size == .slice and needsRendering(p.child),
        .array => |a| needsRendering(a.child),
        .@"struct" => true,
        else => false,
    };
}

const Mode = enum { str, repr };

fn write(comptime T: type, w: *Writer, v: Value, mode: Mode, depth: u32) WriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (comptime @typeInfo(T) == .optional) {
        if (v == .null) return w.writeAll("None");
        return write(@typeInfo(T).optional.child, w, v, mode, depth);
    }
    if (comptime @typeInfo(T) == .array and needsRendering(T))
        return writeTuple(@typeInfo(T).array.child, w, v, depth);
    if (v == .custom) {
        return if (mode == .str) pyjson.writeStr(w, v) else pyjson.writeRepr(w, v, depth);
    }
    if (comptime !needsRendering(T)) {
        // `str(value)` of a bare str is raw text; everything else is repr.
        if (mode == .str and v == .string) return w.writeAll(v.string);
        return pyjson.writeRepr(w, v, depth);
    }
    if (v == .null) return w.writeAll("None");
    if (comptime types.isDict(T)) return writeDict(T.dszig_dict_value, w, v, depth);
    return switch (@typeInfo(T)) {
        .optional => |o| write(o.child, w, v, mode, depth),
        .pointer => |p| writeList(p.child, w, v, depth),
        .array => |a| writeTuple(a.child, w, v, depth),
        .@"struct" => writeModel(T, w, v, mode, depth),
        else => pyjson.writeRepr(w, v, depth),
    };
}

fn writeList(comptime Child: type, w: *Writer, v: Value, depth: u32) WriteError!void {
    if (v != .array) return pyjson.writeRepr(w, v, depth);
    try w.writeByte('[');
    for (v.array.items, 0..) |item, i| {
        if (i != 0) try w.writeAll(", ");
        try write(Child, w, item, .repr, depth + 1);
    }
    try w.writeByte(']');
}

fn writeTuple(comptime Child: type, w: *Writer, v: Value, depth: u32) WriteError!void {
    const items = if (v == .array) v.array.items else blk: {
        const builtin = @import("python_value.zig").view(v) orelse
            return pyjson.writeRepr(w, v, depth);
        break :blk if (builtin == .tuple) builtin.tuple else return pyjson.writeRepr(w, v, depth);
    };
    try w.writeByte('(');
    for (items, 0..) |item, i| {
        if (i != 0) try w.writeAll(", ");
        try write(Child, w, item, .repr, depth + 1);
    }
    if (items.len == 1) try w.writeByte(',');
    try w.writeByte(')');
}

fn writeDict(comptime Child: type, w: *Writer, v: Value, depth: u32) WriteError!void {
    if (v != .object) return pyjson.writeRepr(w, v, depth);
    try w.writeByte('{');
    for (v.object.keys(), v.object.values(), 0..) |key, item, i| {
        if (i != 0) try w.writeAll(", ");
        try pyjson.writeReprString(w, key);
        try w.writeAll(": ");
        try write(Child, w, item, .repr, depth + 1);
    }
    try w.writeByte('}');
}

/// Pydantic prints every declared field. Lowering may omit defaulted members,
/// so an absent optional renders as `None` and an absent slice as `[]`,
/// matching the declared Zig defaults the pinned models mirror.
fn writeModel(comptime T: type, w: *Writer, v: Value, mode: Mode, depth: u32) WriteError!void {
    if (v != .object) return pyjson.writeRepr(w, v, depth);
    if (mode == .repr) {
        try w.writeAll(comptime types.modelName(T));
        try w.writeByte('(');
    }
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types, 0..) |name, FT, i| {
        if (i != 0) try w.writeAll(if (mode == .repr) ", " else " ");
        try w.writeAll(name ++ "=");
        if (v.object.get(name)) |member| {
            try write(FT, w, member, .repr, depth + 1);
        } else if (comptime @typeInfo(FT) == .optional) {
            try w.writeAll("None");
        } else if (comptime isSlice(FT)) {
            try w.writeAll("[]");
        } else {
            try w.writeAll("None");
        }
    }
    if (mode == .repr) try w.writeByte(')');
}

test "python str preserves declared model representations in nested lists" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const Payload = struct {
        pub const model_name = "Payload";
        name: []const u8,
        count: i64,
    };
    var obj: @import("runtime_value.zig").ObjectMap = .empty;
    try obj.put(arena, "name", .{ .string = "O'Reilly\n" });
    try obj.put(arena, "count", .{ .integer = 3 });
    const payload: Value = .{ .object = obj };

    const render = renderer(Payload).?;
    try std.testing.expectEqualStrings("name=\"O'Reilly\\n\" count=3", try render(arena, payload));
    var models: @import("runtime_value.zig").Array = .init(arena);
    try models.append(payload);
    try std.testing.expectEqualStrings("[Payload(name=\"O'Reilly\\n\", count=3)]", try renderer([]const Payload).?(arena, .{ .array = models }));
}

/// `str(list[dspy.Tool])`: Python's list `str` applies `Tool.__repr__`, which
/// prints name and desc raw and args as a dict repr. Input is the catalog from
/// `Tool.toValue`, whose members are `{name, desc, args}`. Not generic: a Zig
/// `Tool` is a callback-carrying struct, not a pydantic model.
pub fn toolsRender(allocator: Allocator, v: Value) Error![]const u8 {
    var out: Writer.Allocating = .init(allocator);
    defer out.deinit();
    writeTools(&out.writer, v, 0) catch |err| return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
        error.ParseValue => error.AdapterParse,
    };
    return out.toOwnedSlice();
}

fn writeTools(w: *Writer, v: Value, depth: u32) pyjson.Error!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (v != .array) return pyjson.writeRepr(w, v, depth);
    try w.writeByte('[');
    for (v.array.items, 0..) |item, i| {
        if (i != 0) try w.writeAll(", ");
        try writeToolRepr(w, item, depth + 1);
    }
    try w.writeByte(']');
}

fn writeToolRepr(w: *Writer, v: Value, depth: u32) pyjson.Error!void {
    if (v != .object) return pyjson.writeRepr(w, v, depth);
    try w.writeAll("Tool(name=");
    try writeRawMember(w, v.object, "name", depth);
    try w.writeAll(", desc=");
    try writeRawMember(w, v.object, "desc", depth);
    try w.writeAll(", args=");
    try pyjson.writeRepr(w, v.object.get("args") orelse .{ .object = .empty }, depth + 1);
    try w.writeByte(')');
}

/// `__repr__` interpolates `self.name`/`self.desc` with `{}`, so strings are
/// written unquoted and unescaped.
fn writeRawMember(w: *Writer, obj: @import("runtime_value.zig").ObjectMap, key: []const u8, depth: u32) pyjson.Error!void {
    const member = obj.get(key) orelse return w.writeAll("None");
    if (member == .string) return w.writeAll(member.string);
    return pyjson.writeRepr(w, member, depth + 1);
}

test "python str renders dspy.Tool lists with raw name/desc and dict args" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var schema: @import("runtime_value.zig").ObjectMap = .empty;
    try schema.put(arena, "type", .{ .string = "string" });
    var args: @import("runtime_value.zig").ObjectMap = .empty;
    try args.put(arena, "q", .{ .object = schema });
    var tool_obj: @import("runtime_value.zig").ObjectMap = .empty;
    try tool_obj.put(arena, "name", .{ .string = "search" });
    try tool_obj.put(arena, "desc", .{ .string = "Search docs" });
    try tool_obj.put(arena, "args", .{ .object = args });
    var catalog: @import("runtime_value.zig").Array = .init(arena);
    try catalog.append(.{ .object = tool_obj });

    const rendered = try toolsRender(arena, .{ .array = catalog });
    try std.testing.expectEqualStrings("[Tool(name=search, desc=Search docs, args={'q': {'type': 'string'}})]", rendered);
}

fn isSlice(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.size == .slice and T != []const u8 and T != []u8,
        else => false,
    };
}
