//! Python-compatible rendering of `std.json.Value`.
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

fn writeJsonDepth(w: *Writer, v: Value, style: Style, depth: u32, comptime ascii: bool) Error!void {
    if (depth >= max_depth) return error.JsonTooDeep;
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| try writeFloat(w, f, if (style == .orjson_indent2) .orjson else .json),
        .number_string => |s| try w.writeAll(s),
        .string => |s| if (ascii) try std.json.Stringify.value(s, .{ .escape_unicode = true }, w) else try writeJsonString(w, s),
        .custom => |custom| if (ascii) return error.ParseValue else try custom.writeSerializedJson(w, style, depth),
        .array => |a| {
            if (a.items.len == 0) return w.writeAll("[]");
            try w.writeByte('[');
            for (a.items, 0..) |item, i| {
                try writeSeparator(w, style, depth + 1, i == 0);
                try writeJsonDepth(w, item, style, depth + 1, ascii);
            }
            try writeClose(w, style, depth, ']');
        },
        .object => |o| {
            if (o.count() == 0) return w.writeAll("{}");
            try w.writeByte('{');
            var it = o.iterator();
            var first = true;
            while (it.next()) |entry| {
                try writeSeparator(w, style, depth + 1, first);
                first = false;
                if (ascii) try std.json.Stringify.value(entry.key_ptr.*, .{ .escape_unicode = true }, w) else try writeJsonString(w, entry.key_ptr.*);
                try w.writeAll(if (style == .minified) ":" else ": ");
                try writeJsonDepth(w, entry.value_ptr.*, style, depth + 1, ascii);
            }
            try writeClose(w, style, depth, '}');
        },
    }
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
        .number_string => |s| try w.writeAll(s),
        .string => |s| try writeReprString(w, s),
        .custom => |custom| try custom.writeRepr(w),
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
                const codepoint = std.unicode.utf8Decode(s[index..][0..width]) catch null;
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
