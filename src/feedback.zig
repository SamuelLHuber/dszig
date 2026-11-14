//! Recover Python object identity for Refine's recursive_mask from typed fields.
//! Serialized Reasoning already is its content; models and Code are not Python
//! dictionaries/strings before orjson runs. Copy containers only when a member
//! actually needs replacement, leaving prediction and trace records unchanged.

const std = @import("std");
const signature = @import("signature.zig");
const types = @import("types.zig");
const code = @import("code.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
pub const Error = Allocator.Error || error{JsonTooDeep};
/// Null means no replacement; consumers can preserve the original container.
pub const Mask = *const fn (Allocator, Value) Error!?Value;

pub fn masker(comptime T: type) ?Mask {
    if (comptime !needsMask(T)) return null;
    return struct {
        fn apply(arena: Allocator, v: Value) Error!?Value {
            return replacement(arena, T, v, 0);
        }
    }.apply;
}

fn needsMask(comptime T: type) bool {
    if (T == signature.Reasoning or T == signature.StrDict or T == []const u8 or T == []u8) return false;
    if (T == Value or T == std.json.ObjectMap) return true;
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

fn replacement(arena: Allocator, comptime T: type, v: Value, depth: u32) Error!?Value {
    if (depth >= @import("pyjson.zig").max_depth) return error.JsonTooDeep;
    if (v == .null) return null;
    if (T == signature.Reasoning or T == signature.StrDict) return null;
    if (comptime code.isCode(T)) return .{ .string = "<non-serializable: " ++ T.model_name ++ ">" };
    if (T == Value) return switch (v) {
        .array => arrayReplacement(arena, Value, v, depth),
        .object => objectReplacement(arena, Value, v, depth),
        .number_string => integerReplacement(v),
        else => null,
    };
    if (T == std.json.ObjectMap) return objectReplacement(arena, Value, v, depth);
    if (comptime types.isDict(T)) return objectReplacement(arena, T.dszig_dict_value, v, depth);
    return switch (@typeInfo(T)) {
        .optional => |o| replacement(arena, o.child, v, depth),
        .pointer => |p| if (p.child == u8) null else arrayReplacement(arena, p.child, v, depth),
        .array => |a| arrayReplacement(arena, a.child, v, depth),
        .@"struct" => .{ .string = comptime "<non-serializable: " ++ types.modelName(T) ++ ">" },
        .int => integerReplacement(v),
        else => null,
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

fn arrayReplacement(arena: Allocator, comptime Child: type, v: Value, depth: u32) Error!?Value {
    if (v != .array) return null;
    var copied: ?std.json.Array = null;
    for (v.array.items, 0..) |item, i| {
        if (try replacement(arena, Child, item, depth + 1)) |changed| {
            if (copied == null) {
                copied = .init(arena);
                try copied.?.appendSlice(v.array.items);
            }
            copied.?.items[i] = changed;
        }
    }
    return if (copied) |items| .{ .array = items } else null;
}

fn objectReplacement(arena: Allocator, comptime Child: type, v: Value, depth: u32) Error!?Value {
    if (v != .object) return null;
    var copied: ?std.json.ObjectMap = null;
    for (v.object.keys(), v.object.values()) |key, item| {
        if (try replacement(arena, Child, item, depth + 1)) |changed| {
            if (copied == null) copied = try v.object.clone(arena);
            copied.?.getPtr(key).?.* = changed;
        }
    }
    return if (copied) |items| .{ .object = items } else null;
}
