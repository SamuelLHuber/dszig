//! Nested field types, as DSPy types them with Pydantic: a Zig struct is a
//! `BaseModel` (named by its `model_name` decl or its type name), a slice is
//! `list[T]`, an optional is `Optional[T]`. This file gives their Python
//! annotation names, their Pydantic JSON schemas with keys in DSPy's
//! `move_type_to_front` order, and Pydantic's lax validation, which also
//! normalizes a value to declaration order with defaults filled in.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Error = error{ ParseValue, OutOfMemory };

/// Types that need the JSON-typed field kind.
pub fn isNested(comptime T: type) bool {
    if (T == []const u8 or T == []const []const u8 or T == std.json.ObjectMap) return false;
    return switch (@typeInfo(T)) {
        .optional => true,
        .@"struct" => !@hasField(T, "map"),
        .pointer => |p| p.size == .slice,
        else => false,
    };
}

/// `Optional[str]`: DSPy returns its text unparsed.
pub fn isOptionalStr(comptime T: type) bool {
    return T == ?[]const u8;
}

pub fn modelName(comptime T: type) []const u8 {
    if (@hasDecl(T, "model_name")) return T.model_name;
    const full = @typeName(T);
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return full;
    return full[dot + 1 ..];
}

/// `get_annotation_name(annotation)`.
pub fn pyName(comptime T: type) []const u8 {
    comptime {
        if (T == []const u8) return "str";
        if (T == bool) return "bool";
        if (T == std.json.ObjectMap) return "dict[str, Any]";
        return switch (@typeInfo(T)) {
            .int => "int",
            .float => "float",
            .@"enum" => |e| blk: {
                var s: []const u8 = "Literal[";
                for (e.field_names, 0..) |tag, i| s = s ++ (if (i > 0) ", " else "") ++ "'" ++ tag ++ "'";
                break :blk s ++ "]";
            },
            .optional => |o| "Union[" ++ pyName(o.child) ++ ", NoneType]",
            .pointer => |p| "list[" ++ pyName(p.child) ++ "]",
            .@"struct" => modelName(T),
            else => @compileError("dszig: unsupported nested type " ++ @typeName(T)),
        };
    }
}

const Entry = struct { key: []const u8, val: []const u8 };

fn lessKey(a: []const u8, b: []const u8) bool {
    const at = std.mem.eql(u8, a, "type");
    const bt = std.mem.eql(u8, b, "type");
    if (at != bt) return at;
    return std.mem.lessThan(u8, a, b);
}

fn render(comptime entries: []const Entry) []const u8 {
    comptime {
        var sorted: [entries.len]Entry = entries[0..entries.len].*;
        for (1..sorted.len) |i| {
            var j = i;
            while (j > 0 and lessKey(sorted[j].key, sorted[j - 1].key)) : (j -= 1) {
                const t = sorted[j];
                sorted[j] = sorted[j - 1];
                sorted[j - 1] = t;
            }
        }
        var s: []const u8 = "{";
        for (sorted, 0..) |e, i| s = s ++ (if (i > 0) ", " else "") ++ "\"" ++ e.key ++ "\": " ++ e.val;
        return s ++ "}";
    }
}

fn quote(comptime s: []const u8) []const u8 {
    return "\"" ++ s ++ "\"";
}

/// Python `name.replace("_", " ").title()`.
fn title(comptime name: []const u8) []const u8 {
    comptime {
        var out: [name.len]u8 = undefined;
        var prev_cased = false;
        for (name, 0..) |c0, i| {
            const c = if (c0 == '_') ' ' else c0;
            out[i] = if (std.ascii.isAlphabetic(c)) (if (prev_cased) std.ascii.toLower(c) else std.ascii.toUpper(c)) else c;
            prev_cased = std.ascii.isAlphabetic(c);
        }
        const final = out;
        return std.mem.trim(u8, &final, " ");
    }
}

/// Schema of `T` where it is used (a struct is a `$ref`).
fn entriesOf(comptime T: type) []const Entry {
    comptime {
        if (T == []const u8) return &.{.{ .key = "type", .val = quote("string") }};
        if (T == bool) return &.{.{ .key = "type", .val = quote("boolean") }};
        if (T == std.json.ObjectMap) return &.{ .{ .key = "type", .val = quote("object") }, .{ .key = "additionalProperties", .val = "true" } };
        return switch (@typeInfo(T)) {
            .int => &.{.{ .key = "type", .val = quote("integer") }},
            .float => &.{.{ .key = "type", .val = quote("number") }},
            .@"enum" => |e| blk: {
                var s: []const u8 = "[";
                for (e.field_names, 0..) |tag, i| s = s ++ (if (i > 0) ", " else "") ++ quote(tag);
                break :blk &.{ .{ .key = "type", .val = quote("string") }, .{ .key = "enum", .val = s ++ "]" } };
            },
            .optional => |o| &.{.{ .key = "anyOf", .val = "[" ++ render(entriesOf(o.child)) ++ ", {\"type\": \"null\"}]" }},
            .pointer => |p| &.{ .{ .key = "type", .val = quote("array") }, .{ .key = "items", .val = render(entriesOf(p.child)) } },
            .@"struct" => &.{.{ .key = "$ref", .val = quote("#/$defs/" ++ modelName(T)) }},
            else => @compileError("dszig: unsupported nested type " ++ @typeName(T)),
        };
    }
}

fn hasDefault(comptime S: type, comptime i: usize) bool {
    return @typeInfo(S).@"struct".field_attrs[i].default_value_ptr != null;
}

fn modelEntries(comptime S: type) []const Entry {
    comptime {
        const info = @typeInfo(S).@"struct";
        var props: [info.field_names.len]Entry = undefined;
        var required: []const u8 = "[";
        var n_required = 0;
        for (info.field_names, info.field_types, 0..) |name, FT, i| {
            var e: []const Entry = entriesOf(FT) ++ &[_]Entry{.{ .key = "title", .val = quote(title(name)) }};
            if (@typeInfo(FT) == .@"struct") e = entriesOf(FT);
            if (hasDefault(S, i)) {
                if (@typeInfo(FT) != .optional or @as(*const FT, @ptrCast(@alignCast(info.field_attrs[i].default_value_ptr.?))).* != null) {
                    @compileError("dszig: model field `" ++ name ++ "` may only default to null");
                }
                e = e ++ &[_]Entry{.{ .key = "default", .val = "null" }};
            } else {
                required = required ++ (if (n_required > 0) ", " else "") ++ quote(name);
                n_required += 1;
            }
            props[i] = .{ .key = name, .val = render(e) };
        }
        const final = props;
        return &.{
            .{ .key = "type", .val = quote("object") },
            .{ .key = "properties", .val = render(&final) },
            .{ .key = "required", .val = required ++ "]" },
            .{ .key = "title", .val = quote(modelName(S)) },
        };
    }
}

fn collectDefs(comptime T: type, comptime seen: []const type) []const type {
    comptime {
        var out = seen;
        if (T == std.json.ObjectMap) return out;
        switch (@typeInfo(T)) {
            .optional => |o| out = collectDefs(o.child, out),
            .pointer => |p| if (T != []const u8) {
                out = collectDefs(p.child, out);
            },
            .@"struct" => |s| {
                for (out) |t| if (t == T) return out;
                out = out ++ &[_]type{T};
                for (s.field_types) |FT| out = collectDefs(FT, out);
            },
            else => {},
        }
        return out;
    }
}

/// `json.dumps(_get_json_schema(T))`.
pub fn schema(comptime T: type) []const u8 {
    comptime {
        @setEvalBranchQuota(1_000_000);
        const is_model = @typeInfo(T) == .@"struct";
        const all = collectDefs(T, &.{});
        const defs = if (is_model) all[1..] else all;
        var top: []const Entry = if (is_model) modelEntries(T) else entriesOf(T);
        if (defs.len > 0) {
            var d: [defs.len]Entry = undefined;
            for (defs, 0..) |D, i| {
                for (defs[0..i]) |E| if (std.mem.eql(u8, modelName(D), modelName(E))) @compileError("dszig: two models named " ++ modelName(D));
                d[i] = .{ .key = modelName(D), .val = render(modelEntries(D)) };
            }
            const final = d;
            top = top ++ &[_]Entry{.{ .key = "$defs", .val = render(&final) }};
        }
        return render(top);
    }
}

/// Pydantic lax-mode `TypeAdapter(T).validate_python(v)`, as a JSON value.
pub fn validate(comptime T: type, arena: Allocator, v: Value) Error!Value {
    if (T == []const u8) return if (v == .string) v else error.ParseValue;
    if (T == std.json.ObjectMap) return if (v == .object) v else error.ParseValue;
    if (T == bool) return switch (v) {
        .bool => v,
        .integer => |i| if (i == 0 or i == 1) Value{ .bool = i == 1 } else error.ParseValue,
        .string => |s| boolText(s),
        else => error.ParseValue,
    };
    switch (@typeInfo(T)) {
        .int => return switch (v) {
            .integer => v,
            .float => |f| if (f == @trunc(f) and @abs(f) < 9.2e18) Value{ .integer = @intFromFloat(f) } else error.ParseValue,
            .string => |s| .{ .integer = std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\n\r"), 10) catch return error.ParseValue },
            else => error.ParseValue,
        },
        .float => return switch (v) {
            .float => v,
            .integer => |i| .{ .float = @floatFromInt(i) },
            .string => |s| .{ .float = std.fmt.parseFloat(f64, std.mem.trim(u8, s, " \t\n\r")) catch return error.ParseValue },
            else => error.ParseValue,
        },
        .@"enum" => {
            if (v != .string) return error.ParseValue;
            if (std.meta.stringToEnum(T, v.string) == null) return error.ParseValue;
            return v;
        },
        .optional => |o| return if (v == .null) v else validate(o.child, arena, v),
        .pointer => |p| {
            if (v != .array) return error.ParseValue;
            var out: std.json.Array = .init(arena);
            try out.ensureTotalCapacity(v.array.items.len);
            for (v.array.items) |item| out.appendAssumeCapacity(try validate(p.child, arena, item));
            return .{ .array = out };
        },
        .@"struct" => |s| {
            if (v != .object) return error.ParseValue;
            var out: std.json.ObjectMap = .empty;
            inline for (s.field_names, s.field_types, 0..) |name, FT, i| {
                if (v.object.get(name)) |fv| {
                    try out.put(arena, name, try validate(FT, arena, fv));
                } else if (comptime hasDefault(T, i)) {
                    try out.put(arena, name, .null);
                } else return error.ParseValue;
            }
            return .{ .object = out };
        },
        else => @compileError("dszig: unsupported nested type " ++ @typeName(T)),
    }
}

fn boolText(s: []const u8) Error!Value {
    const yes = [_][]const u8{ "1", "on", "t", "true", "y", "yes" };
    const no = [_][]const u8{ "0", "off", "f", "false", "n", "no" };
    for (yes) |w| if (std.ascii.eqlIgnoreCase(s, w)) return .{ .bool = true };
    for (no) |w| if (std.ascii.eqlIgnoreCase(s, w)) return .{ .bool = false };
    return error.ParseValue;
}

pub fn Validator(comptime T: type) *const fn (Allocator, Value) Error!Value {
    return struct {
        fn run(arena: Allocator, v: Value) Error!Value {
            return validate(T, arena, v);
        }
    }.run;
}
