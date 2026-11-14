//! Field values: DSPy's `format_field_value` / `parse_value`, and the bridge
//! between typed Zig structs and dynamic `std.json.Value` records.
//!
//! Records (inputs, outputs, demos, examples) are `std.json.ObjectMap`s, which
//! preserve insertion order exactly like Python dicts do.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Writer = std.Io.Writer;

const pyjson = @import("pyjson.zig");
const signature = @import("signature.zig");
const types = @import("types.zig");
const code = @import("code.zig");
const Field = signature.Field;

pub const Record = ObjectMap;

pub const ParseError = error{ParseValue} || Allocator.Error;

/// Python `str.strip()` whitespace (ASCII subset).
pub const py_whitespace = " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f";

pub fn pyStrip(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, py_whitespace);
}

/// `format_field_value(field_info, value)`.
pub fn formatFieldValue(w: *Writer, field: Field, v: Value) pyjson.Error!void {
    if (field.reasoning_type and v == .object) {
        if (v.object.get("content")) |content| {
            if (content == .string) return w.writeAll(content.string);
        }
    }
    if (v == .array and field.kind == .str) return formatListBlobs(w, v.array.items);
    switch (v) {
        .array, .object => try pyjson.writeJson(w, v, .compact),
        else => try pyjson.writeStr(w, v),
    }
}

fn formatListBlobs(w: *Writer, items: []const Value) pyjson.Error!void {
    if (items.len == 0) return w.writeAll("N/A");
    if (items.len == 1) return formatBlob(w, items[0]);
    for (items, 0..) |item, i| {
        if (i > 0) try w.writeByte('\n');
        try w.print("[{d}] ", .{i + 1});
        try formatBlob(w, item);
    }
}

fn formatBlob(w: *Writer, v: Value) pyjson.Error!void {
    var buf: [256]u8 = undefined;
    const text: []const u8 = switch (v) {
        .string => |s| s,
        else => blk: {
            var fw: Writer = .fixed(&buf);
            pyjson.writeStr(&fw, v) catch break :blk "";
            break :blk fw.buffered();
        },
    };
    const multiline = std.mem.indexOfScalar(u8, text, '\n') != null or
        std.mem.indexOf(u8, text, "«") != null or std.mem.indexOf(u8, text, "»") != null;
    if (!multiline) {
        try w.writeAll("«");
        try w.writeAll(text);
        return w.writeAll("»");
    }
    try w.writeAll("«««\n    ");
    var it = std.mem.splitScalar(u8, text, '\n');
    var first = true;
    while (it.next()) |line| {
        if (!first) try w.writeAll("\n    ");
        first = false;
        try w.writeAll(line);
    }
    try w.writeAll("\n»»»");
}

/// `parse_value(value, annotation)` for a value taken from a chat section.
pub fn parseText(arena: Allocator, field: Field, text: []const u8) ParseError!Value {
    if (field.code_type) {
        const validate = field.validate orelse return error.ParseValue;
        const candidate = try parseNestedCandidate(arena, text, true);
        return validate(arena, candidate) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            // DSPy retries a direct custom Type against the original string,
            // but does not apply this fallback to Optional/list/dict unions.
            error.ParseValue => validate(arena, .{ .string = text }),
        };
    }
    return switch (field.kind) {
        .str => .{ .string = text },
        .literal => parseLiteralText(field, text),
        .boolean => parseBoolText(text),
        .int => parseIntText(text),
        .float => parseFloatText(text),
        .str_list => parseStrListText(arena, text),
        .json_object => parseObjectText(arena, text),
        .str_dict => strDict(try parseObjectText(arena, text)),
        .json => parseNestedText(arena, field, text),
    };
}

/// `parse_value(value, annotation)` for a value taken from a JSON object.
pub fn parseJsonValue(arena: Allocator, field: Field, v: Value) ParseError!Value {
    if (field.code_type) {
        if (v == .string) return parseText(arena, field, v.string);
        const validate = field.validate orelse return error.ParseValue;
        return validate(arena, v);
    }
    if (field.reasoning_type and field.kind == .str) return types.validate(signature.Reasoning, arena, v);
    if (field.kind == .str) {
        if (v == .string) return v;
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeStr(&aw.writer, v) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep => error.ParseValue,
        };
        return .{ .string = aw.written() };
    }
    if (v == .string) return parseText(arena, field, v.string);
    return switch (field.kind) {
        .str => unreachable,
        .literal => error.ParseValue,
        .boolean => switch (v) {
            .bool => v,
            .integer => |i| if (i == 0 or i == 1) Value{ .bool = i == 1 } else error.ParseValue,
            else => error.ParseValue,
        },
        .int => parseInteger(v),
        .float => switch (v) {
            .float => v,
            .integer => |i| Value{ .float = @floatFromInt(i) },
            .number_string => |s| parseFloatText(s),
            else => error.ParseValue,
        },
        .str_list => switch (v) {
            .array => |a| blk: {
                for (a.items) |item| if (item != .string) break :blk error.ParseValue;
                break :blk v;
            },
            else => error.ParseValue,
        },
        .json_object => if (v == .object) v else error.ParseValue,
        .str_dict => strDict(v),
        .json => if (field.validate) |validate| validate(arena, v) else error.ParseValue,
    };
}

/// `dict[str, Any]` from text: a JSON object (DSPy also repairs malformed
/// JSON with `json_repair`, which DSZig does not).
/// `parse_value(text, annotation)` for a nested type: `Optional[str]` keeps
/// the text; otherwise JSON (DSPy also repairs malformed JSON), then a Python
/// literal, then the text itself, validated against the type.
fn parseNestedText(arena: Allocator, field: Field, text: []const u8) ParseError!Value {
    if (field.optional_str) return .{ .string = text };
    const candidate = try parseNestedCandidate(arena, text, field.code_annotation);
    const validate = field.validate orelse return error.ParseValue;
    return validate(arena, candidate);
}

fn parseNestedCandidate(arena: Allocator, text: []const u8, code_annotation: bool) ParseError!Value {
    const t = pyStrip(text);
    return std.json.parseFromSliceLeaky(Value, arena, t, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => blk: {
            if (code_annotation) {
                const filtered = code.Code.init(text).code;
                if (filtered.ptr != text.ptr or filtered.len != text.len) {
                    if (std.json.parseFromSliceLeaky(Value, arena, filtered, .{})) |candidate| {
                        break :blk candidate;
                    } else |parse_err| if (parse_err == error.OutOfMemory) return error.OutOfMemory;
                }
            }
            break :blk if (std.mem.eql(u8, t, "None"))
                Value.null
            else if (std.mem.eql(u8, t, "True"))
                Value{ .bool = true }
            else if (std.mem.eql(u8, t, "False"))
                Value{ .bool = false }
            else
                Value{ .string = text };
        },
    };
}

fn strDict(v: Value) ParseError!Value {
    if (v != .object) return error.ParseValue;
    for (v.object.values()) |item| if (item != .string) return error.ParseValue;
    return v;
}

fn parseObjectText(arena: Allocator, text: []const u8) ParseError!Value {
    const v = std.json.parseFromSliceLeaky(Value, arena, pyStrip(text), .{}) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.ParseValue,
    };
    if (v != .object) return error.ParseValue;
    return v;
}

fn parseLiteralText(field: Field, text: []const u8) ParseError!Value {
    if (literalMatch(field, text)) |s| return .{ .string = s };
    var v = pyStrip(text);
    const wrappers = [_][]const u8{ "Literal[", "str[" };
    for (wrappers) |wrapper| {
        if (std.mem.startsWith(u8, v, wrapper) and std.mem.endsWith(u8, v, "]")) {
            v = v[std.mem.indexOfScalar(u8, v, '[').? + 1 .. v.len - 1];
            break;
        }
    }
    if (v.len > 1 and v[0] == v[v.len - 1] and (v[0] == '"' or v[0] == '\'')) v = v[1 .. v.len - 1];
    if (literalMatch(field, v)) |s| return .{ .string = s };
    return error.ParseValue;
}

fn literalMatch(field: Field, text: []const u8) ?[]const u8 {
    for (field.literals) |allowed| {
        if (std.mem.eql(u8, allowed, text)) return allowed;
    }
    return null;
}

/// pydantic lax `bool` from str plus JSON `true`/`false`/`0`/`1`.
fn parseBoolText(text: []const u8) ParseError!Value {
    const truthy = [_][]const u8{ "1", "on", "t", "true", "y", "yes" };
    const falsy = [_][]const u8{ "0", "off", "f", "false", "n", "no" };
    const trimmed = pyStrip(text);
    for ([_][]const u8{ text, trimmed }, 0..) |candidate, pass| {
        // json_repair normalises surrounding whitespace only for JSON tokens.
        if (pass == 1 and !(std.mem.eql(u8, trimmed, "true") or std.mem.eql(u8, trimmed, "false") or
            std.mem.eql(u8, trimmed, "0") or std.mem.eql(u8, trimmed, "1"))) break;
        if (candidate.len > 8) continue;
        var lower_buf: [8]u8 = undefined;
        const lower = std.ascii.lowerString(&lower_buf, candidate);
        for (truthy) |t| if (std.mem.eql(u8, lower, t)) return .{ .bool = true };
        for (falsy) |f| if (std.mem.eql(u8, lower, f)) return .{ .bool = false };
    }
    return error.ParseValue;
}

/// Normalize an integer scalar without narrowing exact decimal JSON numbers.
/// Text is borrowed, like parseText; the typed decoder enforces the Zig width.
pub fn parseInteger(v: Value) error{ParseValue}!Value {
    return switch (v) {
        .integer => v,
        .string, .number_string => |s| parseIntText(s),
        .float => |f| floatToInt(f),
        else => error.ParseValue,
    };
}

fn parseIntText(text: []const u8) error{ParseValue}!Value {
    const t = pyStrip(text);
    if (!isJsonNumber(t)) return error.ParseValue;
    if (std.json.isNumberFormattedLikeAnInteger(t)) return Value.parseFromNumberSlice(t);
    const f = std.fmt.parseFloat(f64, t) catch return error.ParseValue;
    return floatToInt(f);
}

fn floatToInt(f: f64) error{ParseValue}!Value {
    if (!std.math.isFinite(f) or @trunc(f) != f) return error.ParseValue;
    // Pydantic's lax float-to-int conversion excludes both i64 endpoints.
    // Integer tokens take the exact path above and are not limited to i64.
    if (f <= -0x1p63 or f >= 0x1p63) return error.ParseValue;
    return .{ .integer = @intFromFloat(f) };
}

fn parseFloatText(text: []const u8) ParseError!Value {
    const t = pyStrip(text);
    if (!isJsonNumber(t)) return error.ParseValue;
    const f = std.fmt.parseFloat(f64, t) catch return error.ParseValue;
    return .{ .float = f };
}

/// JSON number grammar: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
fn isJsonNumber(s: []const u8) bool {
    var i: usize = 0;
    if (i < s.len and s[i] == '-') i += 1;
    if (i >= s.len) return false;
    if (s[i] == '0') {
        i += 1;
    } else if (std.ascii.isDigit(s[i])) {
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    } else return false;
    if (i < s.len and s[i] == '.') {
        i += 1;
        const start = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == start) return false;
    }
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        const start = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == start) return false;
    }
    return i == s.len;
}

/// `list[str]` from a JSON array or a Python literal list of quoted strings.
fn parseStrListText(arena: Allocator, text: []const u8) ParseError!Value {
    const t = pyStrip(text);
    if (std.json.parseFromSliceLeaky(Value, arena, t, .{})) |v| {
        if (v != .array) return error.ParseValue;
        for (v.array.items) |item| if (item != .string) return error.ParseValue;
        return v;
    } else |_| {}
    return parsePyStrList(arena, t);
}

fn parsePyStrList(arena: Allocator, s: []const u8) ParseError!Value {
    var array: std.json.Array = .init(arena);
    var i: usize = 0;
    if (i >= s.len or s[i] != '[') return error.ParseValue;
    i += 1;
    while (true) {
        i = skipSpaces(s, i);
        if (i >= s.len) return error.ParseValue;
        if (s[i] == ']') {
            i += 1;
            break;
        }
        const quote = s[i];
        if (quote != '\'' and quote != '"') return error.ParseValue;
        i += 1;
        var out: std.ArrayList(u8) = .empty;
        while (true) {
            if (i >= s.len) return error.ParseValue;
            const c = s[i];
            i += 1;
            if (c == quote) break;
            if (c == '\n') return error.ParseValue;
            if (c != '\\') {
                try out.append(arena, c);
                continue;
            }
            if (i >= s.len) return error.ParseValue;
            const e = s[i];
            i += 1;
            const decoded: u8 = switch (e) {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\', '\'', '"' => e,
                else => return error.ParseValue,
            };
            try out.append(arena, decoded);
        }
        try array.append(.{ .string = out.items });
        i = skipSpaces(s, i);
        if (i >= s.len) return error.ParseValue;
        if (s[i] == ',') {
            i += 1;
            continue;
        }
        if (s[i] != ']') return error.ParseValue;
        i += 1;
        break;
    }
    if (skipSpaces(s, i) != s.len) return error.ParseValue;
    return .{ .array = array };
}

fn skipSpaces(s: []const u8, start: usize) usize {
    var i = start;
    while (i < s.len and std.mem.indexOfScalar(u8, py_whitespace, s[i]) != null) i += 1;
    return i;
}

// ---------------------------------------------------------------------------
// Typed <-> dynamic bridge.

pub const ConvertError = error{ MissingField, TypeMismatch } || Allocator.Error;

/// Encode a typed struct (signature `Input`/`Output`) as an ordered record.
pub fn toRecord(arena: Allocator, comptime T: type, value: T) Allocator.Error!Record {
    var record: Record = .empty;
    const info = @typeInfo(T).@"struct";
    try record.ensureTotalCapacity(arena, info.field_names.len);
    inline for (info.field_names) |name| {
        record.putAssumeCapacity(name, try toValue(arena, @TypeOf(@field(value, name)), @field(value, name)));
    }
    return record;
}

pub fn toValue(arena: Allocator, comptime T: type, v: T) Allocator.Error!Value {
    if (comptime code.isCode(T)) return v.toValue();
    if (T == signature.Reasoning) return .{ .string = v.content };
    if (T == @import("tool.zig").ToolCalls) return v.toValue(arena);
    if (T == Value) return v;
    if (T == []const u8) return .{ .string = v };
    if (T == []const []const u8) {
        var array: std.json.Array = .init(arena);
        try array.ensureTotalCapacity(v.len);
        for (v) |s| array.appendAssumeCapacity(.{ .string = s });
        return .{ .array = array };
    }
    if (T == bool) return .{ .bool = v };
    if (T == std.json.ObjectMap) return .{ .object = v };
    if (T == signature.StrDict) return .{ .object = v.map };
    if (comptime types.isDict(T)) {
        var map: ObjectMap = .empty;
        try map.ensureTotalCapacity(arena, v.keys.len);
        for (v.keys, v.values) |k, x| map.putAssumeCapacity(k, try toValue(arena, T.dszig_dict_value, x));
        return .{ .object = map };
    }
    return switch (@typeInfo(T)) {
        .int => if (std.math.cast(i64, v)) |i|
            .{ .integer = i }
        else
            .{ .number_string = try std.fmt.allocPrint(arena, "{d}", .{v}) },
        .float => .{ .float = @floatCast(v) },
        .@"enum" => .{ .string = @tagName(v) },
        .optional => |o| if (v) |x| try toValue(arena, o.child, x) else .null,
        .pointer => |p| blk: {
            var array: std.json.Array = .init(arena);
            try array.ensureTotalCapacity(v.len);
            for (v) |item| array.appendAssumeCapacity(try toValue(arena, p.child, item));
            break :blk .{ .array = array };
        },
        .@"struct" => .{ .object = try toRecord(arena, T, v) },
        else => @compileError("dszig: unsupported field type " ++ @typeName(T)),
    };
}

/// Decode an ordered record into a typed struct; every field is required.
pub fn fromRecord(arena: Allocator, comptime T: type, record: *const Record) ConvertError!T {
    var out: T = undefined;
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types) |name, FT| {
        const v = record.get(name) orelse return error.MissingField;
        @field(out, name) = try fromValue(arena, FT, v);
    }
    return out;
}

pub fn fromValue(arena: Allocator, comptime T: type, v: Value) ConvertError!T {
    if (comptime code.isCode(T)) return T.fromValue(v) catch error.TypeMismatch;
    if (T == Value) return v;
    if (T == @import("tool.zig").ToolCalls) return T.fromValue(arena, v) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TypeMismatch,
    };
    if (T == signature.Reasoning) {
        const content = if (v == .object) v.object.get("content") orelse return error.MissingField else v;
        if (content != .string) return error.TypeMismatch;
        return .{ .content = content.string };
    }
    if (T == []const u8) return if (v == .string) v.string else error.TypeMismatch;
    if (T == []const []const u8) {
        if (v != .array) return error.TypeMismatch;
        const out = try arena.alloc([]const u8, v.array.items.len);
        for (v.array.items, out) |item, *o| {
            if (item != .string) return error.TypeMismatch;
            o.* = item.string;
        }
        return out;
    }
    if (T == bool) return if (v == .bool) v.bool else error.TypeMismatch;
    if (T == std.json.ObjectMap) return if (v == .object) v.object else error.TypeMismatch;
    if (T == signature.StrDict) {
        _ = strDict(v) catch return error.TypeMismatch;
        return .{ .map = v.object };
    }
    if (comptime types.isDict(T)) {
        if (v != .object) return error.TypeMismatch;
        const keys = try arena.alloc([]const u8, v.object.count());
        const values = try arena.alloc(T.dszig_dict_value, keys.len);
        for (v.object.keys(), v.object.values(), keys, values) |k, x, *ko, *vo| {
            ko.* = k;
            vo.* = try fromValue(arena, T.dszig_dict_value, x);
        }
        return .{ .keys = keys, .values = values };
    }
    switch (@typeInfo(T)) {
        .int => return switch (v) {
            .integer => |i| std.math.cast(T, i) orelse error.TypeMismatch,
            .number_string => |s| std.fmt.parseInt(T, s, 10) catch error.TypeMismatch,
            else => error.TypeMismatch,
        },
        .float => return switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            .number_string => |s| std.fmt.parseFloat(T, s) catch error.TypeMismatch,
            else => error.TypeMismatch,
        },
        .@"enum" => {
            if (v != .string) return error.TypeMismatch;
            return std.meta.stringToEnum(T, v.string) orelse error.TypeMismatch;
        },
        .optional => |o| return if (v == .null) null else try fromValue(arena, o.child, v),
        .pointer => |p| {
            if (v != .array) return error.TypeMismatch;
            const out = try arena.alloc(p.child, v.array.items.len);
            for (v.array.items, out) |item, *o| o.* = try fromValue(arena, p.child, item);
            return out;
        },
        .@"struct" => |s| {
            if (v != .object) return error.TypeMismatch;
            var out: T = undefined;
            inline for (s.field_names, s.field_types, 0..) |name, FT, i| {
                if (v.object.get(name)) |fv| {
                    @field(out, name) = try fromValue(arena, FT, fv);
                } else if (comptime s.field_attrs[i].default_value_ptr != null) {
                    @field(out, name) = @as(*const FT, @ptrCast(@alignCast(s.field_attrs[i].default_value_ptr.?))).*;
                } else return error.MissingField;
            }
            return out;
        },
        else => @compileError("dszig: unsupported field type " ++ @typeName(T)),
    }
}

/// Python `==` for JSON-loaded values (dict equality ignores key order;
/// `1 == 1.0` and `True == 1` as in Python).
pub fn pyEqual(a: Value, b: Value) bool {
    if (a == .bool) return pyEqual(.{ .integer = if (a.bool) 1 else 0 }, b);
    if (b == .bool) return pyEqual(a, .{ .integer = if (b.bool) 1 else 0 });
    return switch (a) {
        .integer => |i| switch (b) {
            .integer => |j| i == j,
            .float => |f| floatEqualsInt(f, i),
            .number_string => |s| integerStringEqualsValue(s, a),
            else => false,
        },
        .float => |f| switch (b) {
            .integer => |i| floatEqualsInt(f, i),
            .float => |g| f == g,
            .number_string => |s| integerStringEqualsValue(s, a),
            else => false,
        },
        .number_string => |s| integerStringEqualsValue(s, b),
        .null => b == .null,
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .array => |x| blk: {
            if (b != .array or b.array.items.len != x.items.len) break :blk false;
            for (x.items, b.array.items) |p, q| if (!pyEqual(p, q)) break :blk false;
            break :blk true;
        },
        .object => |x| b == .object and recordEqual(&x, &b.object),
        else => false,
    };
}

pub fn recordEqual(a: *const Record, b: *const Record) bool {
    if (a.count() != b.count()) return false;
    var it = a.iterator();
    while (it.next()) |entry| {
        const other = b.get(entry.key_ptr.*) orelse return false;
        if (!pyEqual(entry.value_ptr.*, other)) return false;
    }
    return true;
}

const IntegerText = struct {
    negative: bool,
    digits: []const u8,

    fn parse(text: []const u8) ?IntegerText {
        if (text.len == 0) return null;
        const negative = text[0] == '-';
        const digits = text[@intFromBool(negative)..];
        if (digits.len == 0) return null;
        for (digits) |c| if (!std.ascii.isDigit(c)) return null;
        var start: usize = 0;
        while (start + 1 < digits.len and digits[start] == '0') start += 1;
        const normalized = digits[start..];
        return .{ .negative = negative and !std.mem.eql(u8, normalized, "0"), .digits = normalized };
    }

    fn eql(a: IntegerText, b: IntegerText) bool {
        return a.negative == b.negative and std.mem.eql(u8, a.digits, b.digits);
    }
};

fn integerStringEqualsValue(text: []const u8, v: Value) bool {
    const a = IntegerText.parse(text) orelse return false;
    // Every finite integral f64 fits in u1024. Formatting its exact integer
    // magnitude, rather than rounding the decimal text to f64, preserves Python
    // equality even for integers above 2^53 and beyond native machine words.
    var buffer: [310]u8 = undefined;
    const other = switch (v) {
        .number_string => |s| s,
        .integer => |i| std.fmt.bufPrint(&buffer, "{d}", .{i}) catch unreachable,
        .float => |f| integralFloatText(f, &buffer) orelse return false,
        else => return false,
    };
    return a.eql(IntegerText.parse(other) orelse return false);
}

fn integralFloatText(f: f64, buffer: *[310]u8) ?[]const u8 {
    if (!std.math.isFinite(f) or f != @trunc(f)) return null;
    // Reconstruct binary64's exact integer without unsupported wide casts.
    const bits: u64 = @bitCast(@abs(f));
    const exponent: u11 = @intCast(bits >> 52);
    var magnitude: u1024 = if (exponent == 0) 0 else (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000;
    if (magnitude != 0) {
        if (exponent >= 1075)
            magnitude <<= @intCast(exponent - 1075)
        else
            magnitude >>= @intCast(1075 - exponent);
    }
    return std.fmt.bufPrint(buffer, "{s}{d}", .{ if (f < 0) "-" else "", magnitude }) catch unreachable;
}

/// Python float >= numeric scalar, retaining arbitrary integer boundaries.
pub fn floatAtLeast(f: f64, threshold: Value) error{InvalidNumber}!bool {
    return switch (threshold) {
        .float => |g| f >= g,
        .integer => |i| @as(f128, f) >= @as(f128, @floatFromInt(i)),
        .bool => |b| f >= @as(f64, @floatFromInt(@intFromBool(b))),
        .number_string => |text| blk: {
            const integer = IntegerText.parse(text) orelse {
                const g = std.fmt.parseFloat(f64, text) catch return error.InvalidNumber;
                break :blk f >= g;
            };
            if (std.math.isNan(f)) break :blk false;
            if (std.math.isInf(f)) break :blk f > 0;
            var buffer: [310]u8 = undefined;
            const floor = IntegerText.parse(integralFloatText(@floor(f), &buffer).?).?;
            if (floor.negative != integer.negative) break :blk !floor.negative;
            const magnitude_order = if (floor.digits.len == integer.digits.len)
                std.mem.order(u8, floor.digits, integer.digits)
            else
                std.math.order(floor.digits.len, integer.digits.len);
            break :blk if (floor.negative) magnitude_order != .gt else magnitude_order != .lt;
        },
        else => error.InvalidNumber,
    };
}

fn floatEqualsInt(f: f64, i: i64) bool {
    // Converting the integer to f64 first would merge distinct values above 2^53.
    if (!std.math.isFinite(f) or f != @trunc(f) or f < -0x1p63 or f >= 0x1p63) return false;
    return @as(i64, @intFromFloat(f)) == i;
}

/// Deep copy of a value into `arena` (strings included).
pub fn cloneValue(arena: Allocator, v: Value) Allocator.Error!Value {
    return switch (v) {
        .string => |s| .{ .string = try arena.dupe(u8, s) },
        .number_string => |s| .{ .number_string = try arena.dupe(u8, s) },
        .array => |a| blk: {
            var out: std.json.Array = .init(arena);
            try out.ensureTotalCapacity(a.items.len);
            for (a.items) |item| out.appendAssumeCapacity(try cloneValue(arena, item));
            break :blk .{ .array = out };
        },
        .object => |o| .{ .object = try cloneRecord(arena, &o) },
        else => v,
    };
}

pub fn cloneRecord(arena: Allocator, r: *const Record) Allocator.Error!Record {
    var out: Record = .empty;
    try out.ensureTotalCapacity(arena, r.count());
    var it = r.iterator();
    while (it.next()) |entry| {
        out.putAssumeCapacity(try arena.dupe(u8, entry.key_ptr.*), try cloneValue(arena, entry.value_ptr.*));
    }
    return out;
}

const testing = std.testing;

test "Python numeric equality preserves integer precision" {
    try testing.expect(!pyEqual(.{ .integer = 9_007_199_254_740_993 }, .{ .integer = 9_007_199_254_740_992 }));
    try testing.expect(!pyEqual(.{ .integer = 9_007_199_254_740_993 }, .{ .float = 9_007_199_254_740_992.0 }));
    try testing.expect(!pyEqual(.{ .integer = std.math.maxInt(i64) }, .{ .float = 9_223_372_036_854_775_808.0 }));
    try testing.expect(pyEqual(.{ .integer = std.math.minInt(i64) }, .{ .float = -9_223_372_036_854_775_808.0 }));
    try testing.expect(pyEqual(.{ .bool = true }, .{ .float = 1.0 }));
    try testing.expect(!pyEqual(.{ .string = "1" }, .{ .integer = 1 }));
}

fn testField(kind: signature.Kind) Field {
    return .{ .name = "f", .kind = kind, .type_name = "", .desc = "", .prefix = "F:", .literals = &.{ "x", "y" } };
}

test "parse_value parity with DSPy 3.4.1" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const ok = struct {
        fn check(a: Allocator, kind: signature.Kind, text: []const u8, expected: []const u8) !void {
            const v = try parseText(a, testField(kind), text);
            var buf: [128]u8 = undefined;
            var w: Writer = .fixed(&buf);
            try pyjson.writeStr(&w, v);
            try testing.expectEqualStrings(expected, w.buffered());
        }
        fn fails(a: Allocator, kind: signature.Kind, text: []const u8) !void {
            try testing.expectError(error.ParseValue, parseText(a, testField(kind), text));
        }
    };
    try ok.check(arena, .boolean, "True", "True");
    try ok.check(arena, .boolean, "true", "True");
    try ok.check(arena, .boolean, "yes", "True");
    try ok.fails(arena, .boolean, " no");
    try ok.check(arena, .boolean, "1", "True");
    try ok.check(arena, .boolean, "TRUE", "True");
    try ok.check(arena, .int, "42", "42");
    try ok.check(arena, .int, "42.0", "42");
    try ok.fails(arena, .int, "4.5");
    try ok.fails(arena, .int, "abc");
    try ok.check(arena, .int, " 7 ", "7");
    try ok.check(arena, .float, "3", "3.0");
    try ok.check(arena, .float, "1e3", "1000.0");
    try ok.check(arena, .str_list, "[\"a\", \"b\"]", "['a', 'b']");
    try ok.check(arena, .str_list, "['a', 'b']", "['a', 'b']");
    try ok.fails(arena, .str_list, "a, b");
    try ok.fails(arena, .str_list, "[1, 2]");
    try ok.check(arena, .literal, "'x'", "x");
    try ok.check(arena, .literal, "Literal[y]", "y");
    try ok.fails(arena, .literal, "z");
    try ok.check(arena, .literal, "x ", "x");
}

test "native integer widths preserve exact unquoted decimal values" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    var empty: [0]u8 = .{};
    var fixed: std.heap.FixedBufferAllocator = .init(&empty);
    try testing.expectEqual(@as(i64, 42), (try toValue(fixed.allocator(), u256, 42)).integer);
    try testing.expectEqual(@as(i64, -42), (try toValue(fixed.allocator(), i256, -42)).integer);
    try testing.expectEqual(std.math.minInt(i64), (try toValue(fixed.allocator(), i128, std.math.minInt(i64))).integer);

    inline for (.{ u64, u65, u128, u256, i65, i128, i256 }) |T| {
        inline for (.{ std.math.minInt(T), std.math.maxInt(T) }) |integer| {
            const encoded = try toValue(arena, T, integer);
            try testing.expectEqual(@as(T, integer), try fromValue(arena, T, encoded));
            const expected = try std.fmt.allocPrint(arena, "{d}", .{integer});
            const json = try std.json.Stringify.valueAlloc(arena, encoded, .{});
            try testing.expectEqualStrings(expected, json);
            var prompt: Writer.Allocating = .init(arena);
            try formatFieldValue(&prompt.writer, testField(.int), encoded);
            try testing.expectEqualStrings(expected, prompt.written());
            const parsed = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
            try testing.expectEqual(@as(T, integer), try fromValue(arena, T, try parseJsonValue(arena, testField(.int), parsed)));
            try testing.expectEqual(@as(T, integer), try fromValue(arena, T, try parseText(arena, testField(.int), expected)));
            try testing.expect(pyEqual(encoded, try cloneValue(arena, encoded)));
        }
    }

    inline for (.{
        .{ @as(u64, 10_000_000_000_000_000_000), "10000000000000000000" },
        .{ std.math.maxInt(u64), "18446744073709551615" },
    }) |case| {
        const encoded = try toValue(arena, u64, case[0]);
        try testing.expectEqualStrings(case[1], encoded.number_string);
        var rendered: Writer.Allocating = .init(arena);
        try pyjson.writeJson(&rendered.writer, encoded, .compact);
        try testing.expectEqualStrings(case[1], rendered.written());
        try testing.expectEqual(case[0], try fromValue(arena, u64, try parseInteger(.{ .string = case[1] })));
    }
    try testing.expectError(error.TypeMismatch, fromValue(arena, i64, .{ .number_string = "18446744073709551615" }));
    try testing.expectError(error.TypeMismatch, fromValue(arena, u64, .{ .number_string = "18446744073709551616" }));
    try testing.expectError(error.TypeMismatch, fromValue(arena, u64, .{ .number_string = "-1" }));
    try testing.expectError(error.TypeMismatch, fromValue(arena, u64, .{ .number_string = "1.5" }));
    try testing.expectEqual(@as(i64, 10_000_000_000_000_000), (try parseInteger(.{ .float = 1e16 })).integer);
    try testing.expectError(error.ParseValue, parseInteger(.{ .float = 0x1p63 }));
    try testing.expectError(error.ParseValue, parseInteger(.{ .float = -0x1p63 }));
}

test "nested native integers round trip through typed validation" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Payload = struct {
        id: u64,
        optional: ?u128,
        items: []const i128,
        counts: signature.Dict(u64),
    };
    const original: Payload = .{
        .id = std.math.maxInt(u64),
        .optional = std.math.maxInt(u128),
        .items = &.{ std.math.minInt(i128), std.math.maxInt(i128) },
        .counts = .{ .keys = &.{"large"}, .values = &.{10_000_000_000_000_000_000} },
    };
    const encoded = try toValue(arena, Payload, original);
    const json = try std.json.Stringify.valueAlloc(arena, encoded, .{});
    const parsed = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
    const checked = try types.validate(Payload, arena, parsed);
    const decoded = try fromValue(arena, Payload, checked);
    try testing.expectEqual(original.id, decoded.id);
    try testing.expectEqual(original.optional, decoded.optional);
    try testing.expectEqualSlices(i128, original.items, decoded.items);
    try testing.expectEqual(original.counts.get("large"), decoded.counts.get("large"));
    try testing.expect(pyEqual(encoded, checked));
    try testing.expectError(error.ParseValue, types.validate(u64, arena, .{ .number_string = "18446744073709551616" }));
    try testing.expectError(error.ParseValue, types.validate(i64, arena, encoded.object.get("id").?));
    const signed_text = try types.validate(u64, arena, .{ .string = "+18_446_744_073_709_551_615" });
    try testing.expectEqual(std.math.maxInt(u64), try fromValue(arena, u64, signed_text));
    try testing.expectEqualStrings("18446744073709551615", signed_text.number_string);
}

test "wide integer equality is exact across JSON representations" {
    const maximum: Value = .{ .number_string = "18446744073709551615" };
    try testing.expect(pyEqual(maximum, maximum));
    try testing.expect(!pyEqual(maximum, .{ .number_string = "18446744073709551614" }));
    try testing.expect(!pyEqual(maximum, .{ .float = 0x1p64 }));
    try testing.expect(!pyEqual(.{ .float = 0x1p64 }, maximum));
    try testing.expect(pyEqual(.{ .number_string = "10000000000000000000" }, .{ .float = 1e19 }));
    try testing.expect(pyEqual(.{ .float = -0x1p127 }, .{ .number_string = "-170141183460469231731687303715884105728" }));
    try testing.expect(!pyEqual(.{ .float = -0x1p127 }, .{ .number_string = "-170141183460469231731687303715884105727" }));
    try testing.expect(pyEqual(.{ .number_string = "9223372036854775807" }, .{ .integer = std.math.maxInt(i64) }));
    try testing.expect(pyEqual(.{ .integer = std.math.minInt(i64) }, .{ .number_string = "-9223372036854775808" }));
    try testing.expect(pyEqual(.{ .number_string = "1" }, .{ .bool = true }));
    try testing.expect(!pyEqual(maximum, .{ .string = "18446744073709551615" }));
    try testing.expect(!pyEqual(maximum, .{ .float = std.math.inf(f64) }));
    try testing.expect(!pyEqual(maximum, .{ .float = std.math.nan(f64) }));
    try testing.expect(pyEqual(.{ .number_string = "0" }, .{ .float = -0.0 }));
    try testing.expect(!pyEqual(.{ .number_string = "0" }, .{ .float = 0x1p-1074 }));
    const largest_float_integer: u1024 = @as(u1024, 0x001f_ffff_ffff_ffff) << 971;
    var buffer: [310]u8 = undefined;
    const exact = try std.fmt.bufPrint(&buffer, "{d}", .{largest_float_integer});
    try testing.expect(pyEqual(.{ .number_string = exact }, .{ .float = 0x1.fffffffffffffp1023 }));
    const adjacent = try std.fmt.bufPrint(&buffer, "{d}", .{largest_float_integer + 1});
    try testing.expect(!pyEqual(.{ .number_string = adjacent }, .{ .float = 0x1.fffffffffffffp1023 }));
}

test "typed record round trip" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const T = struct { q: []const u8, ctx: []const []const u8, ok: bool, n: u8, s: f64, e: enum { a, b } };
    const in: T = .{ .q = "hi", .ctx = &.{ "x", "y" }, .ok = true, .n = 3, .s = 0.5, .e = .b };
    const r = try toRecord(arena, T, in);
    try testing.expectEqualStrings("q", r.keys()[0]);
    const back = try fromRecord(arena, T, &r);
    try testing.expectEqualStrings("y", back.ctx[1]);
    try testing.expectEqual(in.e, back.e);
    try testing.expect(recordEqual(&r, &(try cloneRecord(arena, &r))));
}
