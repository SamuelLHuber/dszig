//! Recover Python object identity for Refine's recursive_mask from typed fields.
//! Refine preserves actual Reasoning content; other models and Code are not
//! Python dictionaries/strings before orjson runs. Copy containers only when a member
//! actually needs replacement, leaving prediction and trace records unchanged.

const std = @import("std");
const signature = @import("signature.zig");
const types = @import("types.zig");
const code = @import("code.zig");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
pub const Error = Allocator.Error || error{ JsonTooDeep, ParseValue };
/// Null means no replacement; consumers can preserve the original container.
pub const Mask = *const fn (Allocator, Value) Error!?Value;

pub fn masker(comptime T: type) ?Mask {
    if (comptime !needsMask(T)) return null;
    return struct {
        fn apply(arena: Allocator, v: Value) Error!?Value {
            return replacement(arena, T, v, 0, true);
        }
    }.apply;
}

/// SIMBA's recursive_mask has no special Reasoning exception. Mask actual
/// classes, not fields merely annotated as classes, preserving plain values.
pub fn maskValue(arena: Allocator, v: Value) Error!Value {
    return try replacement(arena, Value, v, 0, false) orelse v;
}

fn needsMask(comptime T: type) bool {
    if (T == signature.StrDict or T == []const u8 or T == []u8) return true;
    if (T == Value or T == @import("runtime_value.zig").ObjectMap) return true;
    if (comptime types.isDict(T)) return needsMask(T.dszig_dict_value);
    return switch (@typeInfo(T)) {
        .optional => |o| needsMask(o.child),
        .pointer => |p| needsMask(p.child),
        .array => |a| needsMask(a.child),
        .@"struct" => true,
        .int => |i| i.bits > 64,
        else => false,
    };
}

fn replacement(arena: Allocator, comptime T: type, v: Value, depth: u32, comptime reasoning_as_text: bool) Error!?Value {
    if (depth >= @import("pyjson.zig").max_depth) return error.JsonTooDeep;
    if (v == .null) return null;
    if (v == .string and !std.unicode.utf8ValidateSlice(v.string))
        return .{ .string = "<non-serializable: str>" };
    if (v == .custom) {
        const custom = v.custom;
        if (custom.vtable.enumeration) |inspect| {
            if (orjsonAccepts(inspect(custom.ptr).value, depth + 1)) return null;
        }
        if (custom.vtable.python) |inspect| {
            switch (inspect(custom.ptr)) {
                .tuple => |items| {
                    var result: ?[]Value = null;
                    for (items, 0..) |item, index| {
                        if (try replacement(arena, Value, item, depth + 1, reasoning_as_text)) |changed| {
                            if (result == null) result = try arena.dupe(Value, items);
                            result.?[index] = changed;
                        }
                    }
                    return if (result) |changed|
                        try @import("python_value.zig").tuple(arena, changed)
                    else
                        null;
                },
                .dictionary => |entries| {
                    var result: ?[]@import("runtime_value.zig").PythonEntry = null;
                    for (entries, 0..) |entry, index| {
                        if (try replacement(arena, Value, entry.value, depth + 1, reasoning_as_text)) |changed| {
                            if (result == null) result = try arena.dupe(@import("runtime_value.zig").PythonEntry, entries);
                            result.?[index].value = changed;
                        }
                    }
                    return if (result) |changed|
                        try @import("python_value.zig").dictionary(arena, changed)
                    else
                        null;
                },
                else => {},
            }
        }
        if (comptime reasoning_as_text) {
            if (custom.get(signature.Reasoning)) |reasoning| return .{ .string = reasoning.content };
        }
        return .{ .string = try std.fmt.allocPrint(arena, "<non-serializable: {s}>", .{custom.vtable.model_name}) };
    }
    if (T == signature.StrDict) return objectReplacement(arena, []const u8, v, depth, reasoning_as_text);
    if (T == @import("integer.zig").Integer) return replacement(arena, Value, v, depth, reasoning_as_text);
    if (T == signature.Reasoning or T == @import("file.zig").File or
        T == @import("bounding_box.zig").BoundingBox or comptime code.isCode(T))
        return replacement(arena, Value, v, depth, reasoning_as_text);
    if (T == Value) return switch (v) {
        .array => arrayReplacement(arena, Value, v, depth, reasoning_as_text),
        .object => objectReplacement(arena, Value, v, depth, reasoning_as_text),
        .number_string => integerReplacement(v),
        else => null,
    };
    if (T == @import("runtime_value.zig").ObjectMap) return objectReplacement(arena, Value, v, depth, reasoning_as_text);
    if (comptime types.isDict(T)) return objectReplacement(arena, T.dszig_dict_value, v, depth, reasoning_as_text);
    return switch (@typeInfo(T)) {
        .optional => |o| replacement(arena, o.child, v, depth, reasoning_as_text),
        .pointer => |p| if (p.child == u8) null else arrayReplacement(arena, p.child, v, depth, reasoning_as_text),
        .array => |a| arrayReplacement(arena, a.child, v, depth, reasoning_as_text),
        .@"struct" => .{ .string = comptime "<non-serializable: " ++ types.modelName(T) ++ ">" },
        .int => integerReplacement(v),
        else => null,
    };
}

fn orjsonAccepts(v: Value, depth: u32) bool {
    if (depth >= @import("pyjson.zig").max_depth) return false;
    return switch (v) {
        .null, .bool, .integer, .float => true,
        .string => |s| std.unicode.utf8ValidateSlice(s),
        .number_string => integerReplacement(v) == null,
        .array => |a| blk: {
            for (a.items) |item| if (!orjsonAccepts(item, depth + 1)) break :blk false;
            break :blk true;
        },
        .object => |o| blk: {
            for (o.keys(), o.values()) |key, item|
                if (!std.unicode.utf8ValidateSlice(key) or !orjsonAccepts(item, depth + 1)) break :blk false;
            break :blk true;
        },
        .custom => |custom| blk: {
            if (custom.vtable.enumeration) |inspect|
                break :blk orjsonAccepts(inspect(custom.ptr).value, depth + 1);
            if (custom.vtable.python) |inspect| {
                switch (inspect(custom.ptr)) {
                    .tuple => |items| {
                        for (items) |item| if (!orjsonAccepts(item, depth + 1)) break :blk false;
                        break :blk true;
                    },
                    else => {},
                }
            }
            break :blk false;
        },
    };
}

fn integerReplacement(v: Value) ?Value {
    if (v != .number_string) return null;
    const text = v.number_string;
    if (text.len == 0 or std.mem.indexOfAny(u8, text, ".eE") != null) return null;
    if (text[0] == '-') {
        _ = std.fmt.parseInt(i64, text, 10) catch return .{ .string = "<non-serializable: int>" };
    } else {
        _ = std.fmt.parseInt(u64, text, 10) catch return .{ .string = "<non-serializable: int>" };
    }
    return null;
}

fn arrayReplacement(arena: Allocator, comptime Child: type, v: Value, depth: u32, comptime reasoning_as_text: bool) Error!?Value {
    if (v != .array) return null;
    var copied: ?@import("runtime_value.zig").Array = null;
    for (v.array.items, 0..) |item, i| {
        if (try replacement(arena, Child, item, depth + 1, reasoning_as_text)) |changed| {
            if (copied == null) {
                copied = .init(arena);
                try copied.?.appendSlice(v.array.items);
            }
            copied.?.items[i] = changed;
        }
    }
    return if (copied) |items| .{ .array = items } else null;
}

fn objectReplacement(arena: Allocator, comptime Child: type, v: Value, depth: u32, comptime reasoning_as_text: bool) Error!?Value {
    if (v != .object) return null;
    var copied: ?@import("runtime_value.zig").ObjectMap = null;
    for (v.object.keys(), v.object.values()) |key, item| {
        if (try replacement(arena, Child, item, depth + 1, reasoning_as_text)) |changed| {
            if (copied == null) copied = try v.object.clone(arena);
            copied.?.getPtr(key).?.* = changed;
        }
    }
    return if (copied) |items| .{ .object = items } else null;
}
