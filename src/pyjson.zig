//! Python-compatible rendering of native runtime values.
//!
//! DSPy formats prompts with `json.dumps` (separators ", " and ": ",
//! `ensure_ascii=False`), `str()` / `repr()` of Python values, and saves
//! programs with `orjson` (indent 2). Byte-identical prompts and artifacts
//! require reproducing those exact renderings.

const std = @import("std");
const assert = std.debug.assert;
const python_unicode = @import("python_unicode.zig");
const Value = @import("runtime_value.zig").Value;
const Writer = std.Io.Writer;

pub const Style = @import("runtime_value.zig").JsonStyle;

/// Containers nested deeper than this are rejected rather than recursed.
pub const max_depth = 64;

pub const Error = Writer.Error || error{ JsonTooDeep, ParseValue };

pub fn writeJson(w: *Writer, v: Value, style: Style) Error!void {
    try writeJsonDepth(w, v, style, 0, false);
}

/// Plain Python json.dumps defaults: ASCII escapes and spaced separators.
/// Unlike DSPy's custom serializer, plain JSON rejects native class objects.
pub fn writeJsonAscii(w: *Writer, v: Value) Error!void {
    try writeJsonDepth(w, v, .compact, 0, true);
}

/// Python json.dumps(value, indent=2), including default ASCII escaping.
pub fn writeJsonAsciiIndented(w: *Writer, v: Value) Error!void {
    try writeJsonDepth(w, v, .indent2, 0, true);
}

test "Python repr treats a WTF8 surrogate as one nonprintable code point" {
    var buffer: [32]u8 = undefined;
    var writer: Writer = .fixed(&buffer);
    try writeReprString(&writer, "a\xed\xa0\x80b");
    try std.testing.expectEqualStrings("'a\\ud800b'", writer.buffered());
}

test "ASCII JSON preserves lone surrogates and escapes surrogate keys as strings" {
    const cases = [_][2][]const u8{
        .{ "\xed\xa0\x80", "\"\\ud800\"" },
        .{ "\xed\xb0\x80", "\"\\udc00\"" },
        .{ "\xf0\x90\x80\x80", "\"\\ud800\\udc00\"" },
        .{ "\x00\x7f雪", "\"\\u0000\\u007f\\u96ea\"" },
    };
    for (cases) |case| {
        var buffer: [64]u8 = undefined;
        var writer: Writer = .fixed(&buffer);
        try writeJsonAscii(&writer, .{ .string = case[0] });
        try std.testing.expectEqualStrings(case[1], writer.buffered());
    }
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena.allocator(), "\xed\xa0\x80", .{ .string = "\xed\xb0\x80" });
    var buffer: [64]u8 = undefined;
    var writer: Writer = .fixed(&buffer);
    try writeJsonAscii(&writer, .{ .object = object });
    try std.testing.expectEqualStrings("{\"\\ud800\": \"\\udc00\"}", writer.buffered());
    writer.end = 0;
    try std.testing.expectError(error.ParseValue, writeJsonAscii(&writer, .{ .string = "\xff" }));
}

test "builtin float boxes format as JSON numbers in every flavor" {
    const python = @import("python_value.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var items: @import("runtime_value.zig").Array = .init(a);
    try items.append(try python.float(a, 1.5));
    try items.append(try python.float(a, std.math.nan(f64)));
    const list: Value = .{ .array = items };
    var buffer: [64]u8 = undefined;
    var writer: Writer = .fixed(&buffer);
    try writeJsonAscii(&writer, list);
    try std.testing.expectEqualStrings("[1.5, NaN]", writer.buffered());
    writer.end = 0;
    try writeJsonProjection(&writer, list, .compact, 0);
    try std.testing.expectEqualStrings("[1.5, null]", writer.buffered());
    writer.end = 0;
    try writeJsonDepth(&writer, list, .orjson_indent2, 0, false);
    try std.testing.expectEqualStrings("[\n  1.5,\n  null\n]", writer.buffered());
}

pub fn writeJsonDepth(w: *Writer, v: Value, style: Style, depth: u32, comptime ascii: bool) Error!void {
    return writeJsonOptions(w, v, style, depth, ascii, false);
}

/// Stream a nested Pydantic JSON-mode value without allocating a projected
/// container. Unlike plain json.dumps, nonfinite float leaves become null.
pub fn writeJsonProjection(w: *Writer, v: Value, style: Style, depth: u32) Error!void {
    return writeJsonOptions(w, v, style, depth, false, true);
}

fn writeJsonOptions(w: *Writer, v: Value, style: Style, depth: u32, comptime ascii: bool, comptime projection: bool) Error!void {
    if (depth >= max_depth) return error.JsonTooDeep;
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| if (projection and !std.math.isFinite(f))
            try w.writeAll("null")
        else
            try writeFloat(w, f, if (style == .orjson_indent2) .orjson else .json),
        .number_string => |s| {
            try checkIntegerRepr(s);
            if (style == .orjson_indent2 and std.json.isNumberFormattedLikeAnInteger(s)) {
                if (s.len > 0 and s[0] == '-') {
                    _ = std.fmt.parseInt(i64, s, 10) catch return error.ParseValue;
                } else {
                    _ = std.fmt.parseInt(u64, s, 10) catch return error.ParseValue;
                }
            }
            try w.writeAll(s);
        },
        .string => |s| {
            if (style == .orjson_indent2 and !std.unicode.utf8ValidateSlice(s)) return error.ParseValue;
            if (ascii) try writeJsonAsciiString(w, s) else try writeJsonString(w, s);
        },
        .custom => |custom| {
            if (custom.vtable.python) |inspect| {
                const view = inspect(custom.ptr);
                // An actual builtin float is a JSON number in every flavor;
                // the class-object refusals below never apply to it.
                if (view == .float)
                    return writeJsonOptions(w, .{ .float = view.float }, style, depth, ascii, projection);
            }
            if (ascii) {
                const inspect = custom.vtable.python orelse return error.ParseValue;
                return switch (inspect(custom.ptr)) {
                    .tuple => |items| writeJsonArray(w, items, style, depth, true, false),
                    .dictionary => |entries| writePlainDictionary(w, entries, style, depth),
                    else => error.ParseValue,
                };
            }
            if (style == .orjson_indent2) {
                if (custom.vtable.enumeration) |inspect|
                    return writeJsonOptions(w, inspect(custom.ptr).value, style, depth + 1, false, false);
                if (custom.vtable.python) |inspect| {
                    const view = inspect(custom.ptr);
                    if (view == .tuple) return writeJsonArray(w, view.tuple, style, depth, false, false);
                }
                return error.ParseValue;
            }
            try custom.writeSerializedJson(w, style, depth);
        },
        .array => |a| try writeJsonArray(w, a.items, style, depth, ascii, projection),
        .object => |o| {
            if (o.count() == 0) return w.writeAll("{}");
            try w.writeByte('{');
            var it = o.iterator();
            var first = true;
            while (it.next()) |entry| {
                try writeSeparator(w, style, depth + 1, first);
                first = false;
                if (style == .orjson_indent2 and !std.unicode.utf8ValidateSlice(entry.key_ptr.*)) return error.ParseValue;
                if (ascii) try writeJsonAsciiString(w, entry.key_ptr.*) else try writeJsonString(w, entry.key_ptr.*);
                try w.writeAll(if (style == .minified) ":" else ": ");
                try writeJsonOptions(w, entry.value_ptr.*, style, depth + 1, ascii, projection);
            }
            try writeClose(w, style, depth, '}');
        },
    }
}

fn writeJsonArray(w: *Writer, items: []const Value, style: Style, depth: u32, comptime ascii: bool, comptime projection: bool) Error!void {
    if (items.len == 0) return w.writeAll("[]");
    try w.writeByte('[');
    for (items, 0..) |item, index| {
        try writeSeparator(w, style, depth + 1, index == 0);
        try writeJsonOptions(w, item, style, depth + 1, ascii, projection);
    }
    try writeClose(w, style, depth, ']');
}

fn writePlainDictionary(w: *Writer, entries: []const @import("runtime_value.zig").PythonEntry, style: Style, depth: u32) Error!void {
    if (entries.len == 0) return w.writeAll("{}");
    try w.writeByte('{');
    for (entries, 0..) |entry, index| {
        try writeSeparator(w, style, depth + 1, index == 0);
        try writeJsonKey(w, entry.key, style, depth + 1, true);
        try w.writeAll(if (style == .minified) ":" else ": ");
        try writeJsonOptions(w, entry.value, style, depth + 1, true, false);
    }
    try writeClose(w, style, depth, '}');
}

/// Python JSON accepts string, numeric, boolean and None dictionary keys.
/// Keep this policy shared with graph-backed provider wire serialization.
pub fn writeJsonKey(
    w: *Writer,
    key: Value,
    style: Style,
    depth: u32,
    comptime ascii: bool,
) Error!void {
    switch (key) {
        .string => try writeJsonDepth(w, key, style, depth, ascii),
        .null, .bool, .integer, .float, .number_string => {
            try w.writeByte('"');
            try writeJsonOptions(w, key, style, depth, true, false);
            try w.writeByte('"');
        },
        else => return error.ParseValue,
    }
}

fn checkIntegerRepr(text: []const u8) error{ParseValue}!void {
    if (std.json.isNumberFormattedLikeAnInteger(text) and
        text.len - @intFromBool(text.len > 0 and text[0] == '-') > 4300)
        return error.ParseValue;
}

/// CPython permits lone surrogate code points in strings. Keep their WTF-8
/// representation lossless instead of std.json's invalid-UTF8 byte-array path.
fn writeJsonAsciiString(w: *Writer, s: []const u8) Error!void {
    try w.writeByte('"');
    var start: usize = 0;
    var index: usize = 0;
    while (index < s.len) {
        const byte = s[index];
        if (byte >= 0x20 and byte < 0x7f and byte != '"' and byte != '\\') {
            index += 1;
            continue;
        }
        try w.writeAll(s[start..index]);
        const escaped: ?[]const u8 = switch (byte) {
            '"' => "\\\"",
            '\\' => "\\\\",
            8 => "\\b",
            9 => "\\t",
            10 => "\\n",
            12 => "\\f",
            13 => "\\r",
            else => null,
        };
        if (escaped) |text| {
            try w.writeAll(text);
            index += 1;
        } else {
            const count = std.unicode.utf8ByteSequenceLength(byte) catch return error.ParseValue;
            if (count > s.len - index) return error.ParseValue;
            const point = std.unicode.wtf8Decode(s[index..][0..count]) catch return error.ParseValue;
            if (point <= 0xffff) {
                try writeAsciiCodeUnit(w, @intCast(point));
            } else {
                const pair = point - 0x10000;
                try writeAsciiCodeUnit(w, @intCast(0xd800 | (pair >> 10)));
                try writeAsciiCodeUnit(w, @intCast(0xdc00 | (pair & 0x3ff)));
            }
            index += count;
        }
        start = index;
    }
    try w.writeAll(s[start..]);
    try w.writeByte('"');
}

fn writeAsciiCodeUnit(w: *Writer, unit: u16) Writer.Error!void {
    const hex = "0123456789abcdef";
    const escaped = [_]u8{
        '\\',                  'u',            hex[unit >> 12], hex[(unit >> 8) & 15],
        hex[(unit >> 4) & 15], hex[unit & 15],
    };
    try w.writeAll(&escaped);
}

fn writeSeparator(w: *Writer, style: Style, depth: u32, first: bool) Writer.Error!void {
    switch (style) {
        .compact => if (!first) try w.writeAll(", "),
        .minified => if (!first) try w.writeByte(','),
        .indent2, .orjson_indent2 => {
            if (!first) try w.writeByte(',');
            try w.writeByte('\n');
            try w.splatByteAll(' ', depth * 2);
        },
    }
}

fn writeClose(w: *Writer, style: Style, depth: u32, close: u8) Writer.Error!void {
    if (style == .indent2 or style == .orjson_indent2) {
        try w.writeByte('\n');
        try w.splatByteAll(' ', depth * 2);
    }
    try w.writeByte(close);
}

pub fn writeJsonString(w: *Writer, s: []const u8) Writer.Error!void {
    try w.writeByte('"');
    try writeJsonStringContents(w, s);
    try w.writeByte('"');
}

fn writeJsonStringContents(w: *Writer, s: []const u8) Writer.Error!void {
    var start: usize = 0;
    for (s, 0..) |c, i| {
        const escape: ?[]const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            else => null,
        };
        if (escape == null and c >= 0x20) continue;
        try w.writeAll(s[start..i]);
        if (escape) |e| {
            try w.writeAll(e);
        } else {
            try w.print("\\u{x:0>4}", .{c});
        }
        start = i + 1;
    }
    try w.writeAll(s[start..]);
}

/// Unbuffered JSON string-content escaping. Callers own the surrounding quotes;
/// the rendered input can be streamed without allocating a second large string.
pub const StringContentsWriter = struct {
    destination: *Writer,
    writer: Writer = .{ .vtable = &.{ .drain = drain }, .buffer = &.{} },

    fn drain(writer: *Writer, data: []const []const u8, splat: usize) Writer.Error!usize {
        const self: *StringContentsWriter = @alignCast(@fieldParentPtr("writer", writer));
        assert(writer.end == 0);
        assert(data.len != 0);
        var consumed: usize = 0;
        for (data[0 .. data.len - 1]) |part| {
            try writeJsonStringContents(self.destination, part);
            consumed = std.math.add(usize, consumed, part.len) catch return error.WriteFailed;
        }
        const last = data[data.len - 1];
        const repeated = std.math.mul(usize, last.len, splat) catch return error.WriteFailed;
        const total = std.math.add(usize, consumed, repeated) catch return error.WriteFailed;
        for (0..splat) |_| try writeJsonStringContents(self.destination, last);
        return total;
    }
};

pub const FloatFlavor = enum { json, python, orjson };

/// Python `repr(float)`: shortest round-trip digits, positional notation for
/// decimal exponents in [-4, 16), otherwise `d.ddde+XX`.
pub fn writeFloat(w: *Writer, f: f64, flavor: FloatFlavor) Writer.Error!void {
    if (flavor == .orjson and !std.math.isFinite(f)) return w.writeAll("null");
    if (std.math.isNan(f)) return w.writeAll(if (flavor == .json) "NaN" else "nan");
    if (std.math.isInf(f)) {
        if (f < 0) try w.writeByte('-');
        return w.writeAll(if (flavor == .json) "Infinity" else "inf");
    }
    var buf: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&buf, "{e}", .{f}) catch unreachable;
    var rest = sci;
    if (rest[0] == '-') {
        try w.writeByte('-');
        rest = rest[1..];
    }
    const e_index = std.mem.indexOfScalar(u8, rest, 'e').?;
    const exp = std.fmt.parseInt(i32, rest[e_index + 1 ..], 10) catch unreachable;
    var digits_buf: [32]u8 = undefined;
    var n_digits: usize = 0;
    for (rest[0..e_index]) |c| {
        if (c == '.') continue;
        digits_buf[n_digits] = c;
        n_digits += 1;
    }
    const digits = digits_buf[0..n_digits];
    assert(digits.len > 0);
    const decpt: i32 = exp + 1;
    if (decpt > (if (flavor == .orjson) @as(i32, -5) else -4) and decpt <= 16) {
        if (decpt <= 0) {
            try w.writeAll("0.");
            try w.splatByteAll('0', @intCast(-decpt));
            try w.writeAll(digits);
        } else {
            const point: usize = @intCast(decpt);
            if (point >= digits.len) {
                try w.writeAll(digits);
                try w.splatByteAll('0', point - digits.len);
                try w.writeAll(".0");
            } else {
                try w.writeAll(digits[0..point]);
                try w.writeByte('.');
                try w.writeAll(digits[point..]);
            }
        }
    } else {
        try w.writeByte(digits[0]);
        if (digits.len > 1) {
            try w.writeByte('.');
            try w.writeAll(digits[1..]);
        }
        const sign: u8 = if (exp < 0) '-' else '+';
        if (flavor == .orjson) {
            try w.print("e{c}{d}", .{ sign, @abs(exp) });
        } else {
            try w.print("e{c}{d:0>2}", .{ sign, @abs(exp) });
        }
    }
}

/// Python `str(v)` for a JSON-loaded value: strings verbatim, `True`/`None`,
/// containers via `repr`.
pub fn writeStr(w: *Writer, v: Value) Error!void {
    switch (v) {
        .string => |s| try w.writeAll(s),
        .custom => |custom| try custom.writeStr(w),
        else => try writeRepr(w, v, 0),
    }
}

pub fn writeRepr(w: *Writer, v: Value, depth: u32) Error!void {
    if (depth >= max_depth) return error.JsonTooDeep;
    switch (v) {
        .null => try w.writeAll("None"),
        .bool => |b| try w.writeAll(if (b) "True" else "False"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| try writeFloat(w, f, .python),
        .number_string => |s| {
            try checkIntegerRepr(s);
            try w.writeAll(s);
        },
        .string => |s| try writeReprString(w, s),
        .custom => |custom| try custom.vtable.write_repr(w, custom.ptr, depth),
        .array => |a| {
            try w.writeByte('[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try writeRepr(w, item, depth + 1);
            }
            try w.writeByte(']');
        },
        .object => |o| {
            try w.writeByte('{');
            var it = o.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try w.writeAll(", ");
                first = false;
                try writeReprString(w, entry.key_ptr.*);
                try w.writeAll(": ");
                try writeRepr(w, entry.value_ptr.*, depth + 1);
            }
            try w.writeByte('}');
        },
    }
}

/// Python `repr(str)`, including Unicode printability and quote selection.
pub fn writeReprString(w: *Writer, s: []const u8) Writer.Error!void {
    const has_single = std.mem.indexOfScalar(u8, s, '\'') != null;
    const has_double = std.mem.indexOfScalar(u8, s, '"') != null;
    const quote: u8 = if (has_single and !has_double) '"' else '\'';
    try w.writeByte(quote);
    var index: usize = 0;
    while (index < s.len) {
        const c = s[index];
        if (c >= 0x80) {
            const width = std.unicode.utf8ByteSequenceLength(c) catch 1;
            if (width <= s.len - index) {
                const codepoint: ?u21 = std.unicode.wtf8Decode(s[index..][0..width]) catch null;
                if (codepoint) |cp| {
                    if (python_unicode.isPrintable(cp)) {
                        try w.writeAll(s[index..][0..width]);
                    } else if (cp <= 0xff) {
                        try w.print("\\x{x:0>2}", .{cp});
                    } else if (cp <= 0xffff) {
                        try w.print("\\u{x:0>4}", .{cp});
                    } else {
                        try w.print("\\U{x:0>8}", .{cp});
                    }
                    index += width;
                    continue;
                }
            }
        }
        index += 1;
        switch (c) {
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (c == quote) {
                    try w.writeByte('\\');
                    try w.writeByte(c);
                } else if (c < 0x20 or c == 0x7f) {
                    try w.print("\\x{x:0>2}", .{c});
                } else {
                    try w.writeByte(c);
                }
            },
        }
    }
    try w.writeByte(quote);
}

const testing = std.testing;

fn expectFloat(expected: []const u8, f: f64) !void {
    var buf: [64]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try writeFloat(&w, f, .python);
    try testing.expectEqualStrings(expected, w.buffered());
}

test "float repr matches CPython" {
    try expectFloat("1.0", 1.0);
    try expectFloat("0.1", 0.1);
    try expectFloat("1e+16", 1e16);
    try expectFloat("1e+17", 1e17);
    try expectFloat("1.2345678901234568e+17", 123456789012345678.0);
    try expectFloat("0.0001", 1e-4);
    try expectFloat("1e-05", 1e-5);
    try expectFloat("2.5e-07", 2.5e-7);
    try expectFloat("3.14159", 3.14159);
    try expectFloat("1.5e+300", 1.5e300);
    try expectFloat("-0.0", -0.0);
    try expectFloat("100.0", 100.0);
    try expectFloat("0.3333333333333333", 1.0 / 3.0);
    try expectFloat("1234567890123456.0", 1234567890123456.0);
}

test "json dumps compact and indent2 and repr" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(Value, arena.allocator(),
        \\{"a": [1, "x\n\"y", true, null], "b": {}, "c": 2.5, "d": "it's"}
    , .{});
    var buf: [512]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try writeJson(&w, parsed, .compact);
    try testing.expectEqualStrings(
        \\{"a": [1, "x\n\"y", true, null], "b": {}, "c": 2.5, "d": "it's"}
    , w.buffered());
    w = .fixed(&buf);
    try writeJson(&w, parsed, .indent2);
    try testing.expectEqualStrings(
        \\{
        \\  "a": [
        \\    1,
        \\    "x\n\"y",
        \\    true,
        \\    null
        \\  ],
        \\  "b": {},
        \\  "c": 2.5,
        \\  "d": "it's"
        \\}
    , w.buffered());
    w = .fixed(&buf);
    try writeStr(&w, parsed);
    try testing.expectEqualStrings(
        \\{'a': [1, 'x\n"y', True, None], 'b': {}, 'c': 2.5, 'd': "it's"}
    , w.buffered());
}
