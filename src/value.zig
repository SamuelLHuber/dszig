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
    return switch (field.kind) {
        .str => .{ .string = text },
        .literal => parseLiteralText(field, text),
        .boolean => parseBoolText(text),
        .int => parseIntText(text),
        .float => parseFloatText(text),
        .str_list => parseStrListText(arena, text),
    };
}

/// `parse_value(value, annotation)` for a value taken from a JSON object.
pub fn parseJsonValue(arena: Allocator, field: Field, v: Value) ParseError!Value {
    if (field.kind == .str) {
        if (v == .string) return v;
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeStr(&aw.writer, v) catch return error.OutOfMemory;
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
        .int => switch (v) {
            .integer => v,
            .float => |f| floatToInt(f),
            else => error.ParseValue,
        },
        .float => switch (v) {
            .float => v,
            .integer => |i| Value{ .float = @floatFromInt(i) },
            else => error.ParseValue,
        },
        .str_list => switch (v) {
            .array => |a| blk: {
                for (a.items) |item| if (item != .string) break :blk error.ParseValue;
                break :blk v;
            },
            else => error.ParseValue,
        },
    };
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

fn parseIntText(text: []const u8) ParseError!Value {
    const t = pyStrip(text);
    if (!isJsonNumber(t)) return error.ParseValue;
    if (std.fmt.parseInt(i64, t, 10)) |i| return .{ .integer = i } else |_| {}
    const f = std.fmt.parseFloat(f64, t) catch return error.ParseValue;
    return floatToInt(f);
}

fn floatToInt(f: f64) ParseError!Value {
    if (!std.math.isFinite(f) or @trunc(f) != f) return error.ParseValue;
    if (@abs(f) > 9.007199254740992e15) return error.ParseValue;
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
    if (T == []const u8) return .{ .string = v };
    if (T == []const []const u8) {
        var array: std.json.Array = .init(arena);
        try array.ensureTotalCapacity(v.len);
        for (v) |s| array.appendAssumeCapacity(.{ .string = s });
        return .{ .array = array };
    }
    if (T == bool) return .{ .bool = v };
    return switch (@typeInfo(T)) {
        .int => .{ .integer = @intCast(v) },
        .float => .{ .float = @floatCast(v) },
        .@"enum" => .{ .string = @tagName(v) },
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
    switch (@typeInfo(T)) {
        .int => {
            if (v != .integer) return error.TypeMismatch;
            return std.math.cast(T, v.integer) orelse error.TypeMismatch;
        },
        .float => return switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => error.TypeMismatch,
        },
        .@"enum" => {
            if (v != .string) return error.TypeMismatch;
            return std.meta.stringToEnum(T, v.string) orelse error.TypeMismatch;
        },
        else => @compileError("dszig: unsupported field type " ++ @typeName(T)),
    }
}

/// Python `==` for JSON-loaded values (dict equality ignores key order;
/// `1 == 1.0` and `True == 1` as in Python).
pub fn pyEqual(a: Value, b: Value) bool {
    const an = numeric(a);
    const bn = numeric(b);
    if (an != null and bn != null) return an.? == bn.?;
    if (an != null or bn != null) return false;
    return switch (a) {
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

fn numeric(v: Value) ?f64 {
    return switch (v) {
        .bool => |x| if (x) 1 else 0,
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
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
