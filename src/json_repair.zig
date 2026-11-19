//! Native port of json-repair 0.64.2's default `loads` path, plus CPython
//! json.loads whole-input decoding without repair (`parseJson`).
//!
//! Only DSPy's call is ported: schema, strict, logging, stream-stable and file
//! input are not options. Values and strings belong to the supplied arena.
//! Python strings can hold lone surrogates; they are emitted as WTF-8 (three
//! bytes, ED A0 80 through ED BF BF), never replaced or dropped. Integers
//! outside i64 use canonical decimal Value.number_string. CPython raises
//! ValueError for integer literals over 4300 digits; that is
//! error.IntegerTooLong, distinct from a JSON decode failure (ParseValue).
//! RecursionError is error.JsonTooDeep, at the native bound `max_depth`
//! rather than CPython's stack-dependent recursion limit. Input must be
//! UTF-8 (WTF-8 surrogates accepted); other bytes fail with ParseValue.
//! A non-string key returned through the source's fenced-block key path
//! raises AssertionError upstream and returns ParseValue here.
//!
//! Derived from https://github.com/mangiucugna/json_repair (MIT):
//! Copyright (c) 2023 Stefano Baccianella
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

const std = @import("std");
const runtime = @import("runtime_value.zig");
const unicode = @import("unicode.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;

pub const Error = Allocator.Error || error{ ParseValue, JsonTooDeep, IntegerTooLong };
/// Native nesting/recursion bound shared by both decoders.
pub const max_depth = 512;
const max_int_digits = 4300;
const Context = enum { key, value, array };
const quotes = [_]u21{ '"', '\'', 0x201c, 0x201d };

/// CPython json.loads(input): whole input, no repair. Accepts NaN/Infinity,
/// overflowing exponents (as infinities) and lone surrogate escapes.
pub fn parseJson(arena: Allocator, input: []const u8) Error!Value {
    var parser: Parser = .{ .arena = arena, .text = try Text.init(arena, input) };
    return parser.wholeStrict();
}

/// Ordered lexical values from the same strict decoder. Object delimiters and
/// duplicate keys survive, numeric spellings remain exact, and strings retain
/// the decoder's existing lone-surrogate behavior.
pub const Token = union(enum) {
    object_begin,
    object_end,
    array_begin,
    array_end,
    scalar: Value,
    number: []const u8,
};

pub fn parseJsonTokens(arena: Allocator, input: []const u8) Error![]const Token {
    var parser: Parser = .{ .arena = arena, .text = try Text.init(arena, input) };
    parser.strictWhitespace();
    _ = try parser.strictValue(0, true);
    parser.strictWhitespace();
    if (parser.index != parser.text.len()) return error.ParseValue;
    return parser.tokens.items;
}

/// json_repair.loads(input) with every option left at its default.
pub fn loads(arena: Allocator, input: []const u8) Error!Value {
    var parser: Parser = .{ .arena = arena, .text = try Text.init(arena, input) };
    // repair_json catches JSONDecodeError and ValueError, not RecursionError.
    if (parser.wholeStrict()) |value| return value else |err| switch (err) {
        error.ParseValue, error.IntegerTooLong => {},
        else => return err,
    }
    parser.index = 0;
    return parser.topLevel();
}

// Character offsets, including Python's negative indexing, are significant to
// the heuristics. ASCII inputs stay borrowed; Unicode or source mutation uses
// a code-point buffer. No intermediate std.json tree is constructed.
const Text = union(enum) {
    ascii: []const u8,
    points: []const u21,

    fn init(arena: Allocator, input: []const u8) Error!Text {
        var ascii = true;
        for (input) |byte| ascii = ascii and byte < 0x80;
        if (ascii) return .{ .ascii = input };
        var result: std.ArrayList(u21) = .empty;
        try result.ensureTotalCapacity(arena, input.len);
        var i: usize = 0;
        while (i < input.len) {
            const first = input[i];
            const n = std.unicode.utf8ByteSequenceLength(first) catch return error.ParseValue;
            if (i + n > input.len) return error.ParseValue;
            var cp: u32 = switch (n) {
                1 => first,
                2 => first & 0x1f,
                3 => first & 0x0f,
                4 => first & 7,
                else => unreachable,
            };
            for (input[i + 1 ..][0 .. n - 1]) |byte| {
                if (byte & 0xc0 != 0x80) return error.ParseValue;
                cp = cp * 64 + (byte & 0x3f);
            }
            if ((n == 2 and cp < 0x80) or (n == 3 and cp < 0x800) or
                (n == 4 and cp < 0x10000) or cp > 0x10ffff) return error.ParseValue;
            result.appendAssumeCapacity(@intCast(cp));
            i += n;
        }
        return .{ .points = result.items };
    }

    fn len(self: Text) isize {
        return @intCast(switch (self) {
            .ascii => |s| s.len,
            .points => |s| s.len,
        });
    }

    /// Python str indexing: negative positions count from the end.
    fn at(self: Text, position: isize) ?u21 {
        const actual = if (position < 0) self.len() + position else position;
        if (actual < 0 or actual >= self.len()) return null;
        const i: usize = @intCast(actual);
        return switch (self) {
            .ascii => |s| s[i],
            .points => |s| s[i],
        };
    }

    /// Python slice [first:last] for nonnegative-clamped bounds.
    fn appendRange(self: Text, arena: Allocator, out: *std.ArrayList(u21), first: isize, last: isize) Allocator.Error!void {
        const lo = std.math.clamp(first, 0, self.len());
        const hi = std.math.clamp(last, lo, self.len());
        try out.ensureUnusedCapacity(arena, @intCast(hi - lo));
        var i = lo;
        while (i < hi) : (i += 1) out.appendAssumeCapacity(self.at(i).?);
    }

    fn string(self: Text, arena: Allocator, first: isize, last: isize) Allocator.Error![]const u8 {
        const lo = std.math.clamp(first, 0, self.len());
        const hi = std.math.clamp(last, lo, self.len());
        return switch (self) {
            .ascii => |s| try arena.dupe(u8, s[@intCast(lo)..@intCast(hi)]),
            .points => |s| try encode(arena, s[@intCast(lo)..@intCast(hi)]),
        };
    }

    fn replace(self: *Text, arena: Allocator, first: isize, last: isize, replacement: []const u21) Allocator.Error!void {
        var out: std.ArrayList(u21) = .empty;
        try self.appendRange(arena, &out, 0, first);
        try out.appendSlice(arena, replacement);
        try self.appendRange(arena, &out, last, self.len());
        self.* = .{ .points = out.items };
    }
};

/// UTF-8, or WTF-8 for lone surrogates.
fn encode(arena: Allocator, points: []const u21) Allocator.Error![]const u8 {
    var size: usize = 0;
    for (points) |cp| size += if (cp < 0x80) @as(usize, 1) else if (cp < 0x800) 2 else if (cp < 0x10000) 3 else 4;
    const bytes = try arena.alloc(u8, size);
    var i: usize = 0;
    for (points) |cp| {
        if (cp < 0x80) {
            bytes[i] = @intCast(cp);
            i += 1;
        } else if (cp < 0x800) {
            bytes[i] = @intCast(0xc0 | cp >> 6);
            bytes[i + 1] = @intCast(0x80 | (cp & 63));
            i += 2;
        } else if (cp < 0x10000) {
            bytes[i] = @intCast(0xe0 | cp >> 12);
            bytes[i + 1] = @intCast(0x80 | ((cp >> 6) & 63));
            bytes[i + 2] = @intCast(0x80 | (cp & 63));
            i += 3;
        } else {
            bytes[i] = @intCast(0xf0 | cp >> 18);
            bytes[i + 1] = @intCast(0x80 | ((cp >> 12) & 63));
            bytes[i + 2] = @intCast(0x80 | ((cp >> 6) & 63));
            bytes[i + 3] = @intCast(0x80 | (cp & 63));
            i += 4;
        }
    }
    std.debug.assert(i == size);
    return bytes;
}

const Range = struct { lo: u21, hi: u21 };

fn inRanges(table: []const Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = table.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < table[mid].lo) {
            hi = mid;
        } else if (cp > table[mid].hi) {
            lo = mid + 1;
        } else return true;
    }
    return false;
}

fn oneOf(cp: ?u21, chars: []const u21) bool {
    const c = cp orelse return false;
    return std.mem.indexOfScalar(u21, chars, c) != null;
}
fn quote(cp: ?u21) bool {
    return oneOf(cp, &quotes);
}
fn opener(cp: ?u21) bool {
    return oneOf(cp, &.{ '[', '{', '(' });
}
/// str.isspace.
fn space(cp: ?u21) bool {
    return if (cp) |c| unicode.isSpace(c) else false;
}
/// str.isalnum: re's Unicode \w minus underscore.
fn alnum(cp: ?u21) bool {
    return if (cp) |c| c != '_' and unicode.isWord(c) else false;
}
/// str.isalpha.
fn alpha(cp: ?u21) bool {
    return if (cp) |c| inRanges(&alpha_ranges, c) else false;
}
/// str.isdigit (includes superscripts and other Unicode digits).
fn digit(cp: ?u21) bool {
    return if (cp) |c| inRanges(&digit_ranges, c) else false;
}
fn asciiDigit(cp: ?u21) bool {
    return if (cp) |c| c >= '0' and c <= '9' else false;
}
/// Exact for every comparison against the ASCII literals used here: no
/// non-ASCII code point lowercases to t/r/u/e/f/a/l/s/n/o.
fn lower(cp: ?u21) ?u21 {
    const c = cp orelse return null;
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}
fn matchingQuote(cp: u21) u21 {
    return if (cp == 0x201c) 0x201d else cp;
}
fn closer(cp: u21) u21 {
    return switch (cp) {
        '{' => '}',
        '[' => ']',
        '(' => ')',
        else => unreachable,
    };
}
fn hex(cp: ?u21) ?u21 {
    const c = cp orelse return null;
    if (c >= '0' and c <= '9') return c - '0';
    if (c >= 'a' and c <= 'f') return c - 'a' + 10;
    if (c >= 'A' and c <= 'F') return c - 'A' + 10;
    return null;
}
fn lastIs(items: []const u21, c: u21) bool {
    return items.len != 0 and items[items.len - 1] == c;
}

/// Python truthiness of a parsed value.
fn truthy(value: Value) bool {
    return switch (value) {
        .null => false,
        .bool => |b| b,
        .integer => |n| n != 0,
        .float => |n| n != 0,
        .number_string => true,
        .string => |s| s.len != 0,
        .array => |a| a.items.len != 0,
        .object => |o| o.count() != 0,
        .custom => unreachable,
    };
}
/// ObjectComparer.is_strictly_empty.
fn strictlyEmpty(value: Value) bool {
    return switch (value) {
        .string => |s| s.len == 0,
        .array => |a| a.items.len == 0,
        .object => |o| o.count() == 0,
        else => false,
    };
}
/// ObjectComparer.is_same_object: types and container structure only.
fn sameObject(a: Value, b: Value, depth: usize) Error!bool {
    if (depth >= max_depth) return error.JsonTooDeep;
    const a_int = a == .integer or a == .number_string;
    const b_int = b == .integer or b == .number_string;
    if (a_int or b_int) return a_int and b_int;
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    switch (a) {
        .object => |object| {
            if (object.count() != b.object.count()) return false;
            for (object.keys(), object.values()) |key, item| {
                const other = b.object.get(key) orelse return false;
                if (!try sameObject(item, other, depth + 1)) return false;
            }
        },
        .array => |array| {
            if (array.items.len != b.array.items.len) return false;
            for (array.items, b.array.items) |x, y| if (!try sameObject(x, y, depth + 1)) return false;
        },
        else => {},
    }
    return true;
}

/// Python int(text) for [-]digits text; wide values stay exact decimal text.
fn integer(arena: Allocator, text: []const u8) Error!Value {
    const negative = text.len != 0 and text[0] == '-';
    const digits = text[@intFromBool(negative)..];
    if (digits.len > max_int_digits) return error.IntegerTooLong;
    if (std.fmt.parseInt(i64, text, 10)) |n| return .{ .integer = n } else |_| {}
    var start: usize = 0;
    while (start + 1 < digits.len and digits[start] == '0') start += 1;
    const significant = digits[start..];
    if (significant.len == 1 and significant[0] == '0') return .{ .integer = 0 };
    const result = try arena.alloc(u8, significant.len + @intFromBool(negative));
    if (negative) result[0] = '-';
    @memcpy(result[@intFromBool(negative)..], significant);
    if (std.fmt.parseInt(i64, result, 10)) |n| return .{ .integer = n } else |_| {}
    return .{ .number_string = result };
}

const CacheEntry = struct {
    targets: [6]u21,
    count: usize,
    start: isize,
    match: ?isize,
};

const StringState = struct {
    missing_quotes: bool = false,
    doubled_quotes: bool = false,
    lstring: u21 = '"',
    rstring: u21 = '"',
    /// Count of LOW_SMART_QUOTE_SENTINEL suffixes on rstring_delimiter.
    low_spans: usize = 0,
    acc: std.ArrayList(u21) = .empty,
    unmatched_delimiter: bool = false,
    pending_inline_container: bool = false,
    inline_stack: std.ArrayList(u21) = .empty,
    no_future_delimiter: bool = false,
    comma_quote_tail: struct { index: isize = -1, is_member: bool = false } = .{},
    cache: std.ArrayList(CacheEntry) = .empty,
    unmatched_braces: usize = 0,
    // True after the latest `[` until a `]` or whitespace is appended.
    compact_regex_class: bool = false,

    fn outer(self: *const StringState) u21 {
        return self.rstring;
    }
    fn active(self: *const StringState) u21 {
        return if (self.low_spans != 0) 0 else self.rstring;
    }
    fn inLow(self: *const StringState) bool {
        return self.active() == 0;
    }
    fn last(self: *const StringState) ?u21 {
        return if (self.acc.items.len == 0) null else self.acc.items[self.acc.items.len - 1];
    }
    fn appendContent(self: *StringState, arena: Allocator, content: []const u21) Allocator.Error!void {
        try self.acc.appendSlice(arena, content);
        for (content) |c| self.track(c);
    }
    fn track(self: *StringState, c: u21) void {
        switch (c) {
            '{' => self.unmatched_braces += 1,
            '}' => if (self.unmatched_braces != 0) {
                self.unmatched_braces -= 1;
            },
            else => {},
        }
        switch (c) {
            '[' => self.compact_regex_class = true,
            ']' => self.compact_regex_class = false,
            else => if (self.compact_regex_class and space(c)) {
                self.compact_regex_class = false;
            },
        }
    }
    /// Backslashes do not affect brace or regex-class tracking. Only the
    /// decoded character needs to be tracked; the prefix has not changed.
    fn replaceTrailingBackslash(self: *StringState, decoded: u21) void {
        std.debug.assert(lastIs(self.acc.items, '\\'));
        const position = self.acc.items.len - 1;
        self.acc.items[position] = decoded;
        self.track(decoded);
    }
};

const Literal = union(enum) { none, value: Value };

const Parser = struct {
    arena: Allocator,
    text: Text,
    index: isize = 0,
    context: [max_depth]Context = undefined,
    contexts: usize = 0,
    deferred: [max_depth]Context = undefined,
    deferred_count: usize = 0,
    calls: usize = 0,
    tried_suffix: bool = false,
    tokens: std.ArrayList(Token) = .empty,

    fn char(self: *const Parser, offset: isize) ?u21 {
        return self.text.at(self.index + offset);
    }
    fn current(self: *const Parser) ?Context {
        return if (self.contexts == 0) null else self.context[self.contexts - 1];
    }
    fn contains(self: *const Parser, context: Context) bool {
        return std.mem.indexOfScalar(Context, self.context[0..self.contexts], context) != null;
    }
    fn push(self: *Parser, context: Context) Error!void {
        if (self.contexts == max_depth) return error.JsonTooDeep;
        self.context[self.contexts] = context;
        self.contexts += 1;
    }
    fn pop(self: *Parser) void {
        if (self.contexts != 0) self.contexts -= 1;
    }
    fn defer_(self: *Parser, context: Context) Error!void {
        if (self.deferred_count == max_depth) return error.JsonTooDeep;
        self.deferred[self.deferred_count] = context;
        self.deferred_count += 1;
    }
    fn enter(self: *Parser) Error!void {
        if (self.calls == max_depth) return error.JsonTooDeep;
        self.calls += 1;
    }
    fn leave(self: *Parser) void {
        self.calls -= 1;
    }
    fn skipWhitespace(self: *Parser) void {
        while (space(self.char(0))) self.index += 1;
    }
    fn scroll(self: *const Parser, offset: isize) isize {
        var i = offset;
        if (self.char(i) == null) return i;
        while (self.index + i >= 0 and space(self.char(i))) i += 1;
        return i;
    }
    fn scrollBack(self: *const Parser, offset: isize) isize {
        var i = offset;
        if (self.char(i) == null) return i;
        while (self.index + i >= 0 and space(self.char(i))) {
            i -= 1;
            if (self.char(i) == null) return i;
        }
        return i;
    }
    /// skip_to_character: first target not escaped by an odd backslash run.
    fn skip(self: *const Parser, targets: []const u21, offset: isize) isize {
        var i = self.index + offset;
        var backslashes: usize = 0;
        while (i < self.text.len()) : (i += 1) {
            const c = self.text.at(i) orelse break;
            if (c == '\\') {
                backslashes += 1;
                continue;
            }
            if (oneOf(c, targets) and backslashes % 2 == 0) return i - self.index;
            backslashes = 0;
        }
        return self.text.len() - self.index;
    }
    fn skip1(self: *const Parser, target: u21, offset: isize) isize {
        return self.skip(&.{target}, offset);
    }
    fn starts(self: *const Parser, offset: isize, literal: []const u8) bool {
        for (literal, 0..) |c, i| if (self.char(offset + @as(isize, @intCast(i))) != c) return false;
        return true;
    }
    fn onlyWhitespaceUntil(self: *const Parser, end: isize) bool {
        var j: isize = 1;
        while (j < end) : (j += 1) {
            const c = self.char(j);
            if (c != null and !space(c)) return false;
        }
        return true;
    }

    // ---- CPython json (C scanner) decoding ----

    fn strictWhitespace(self: *Parser) void {
        while (oneOf(self.char(0), &.{ ' ', '\t', '\r', '\n' })) self.index += 1;
    }
    fn wholeStrict(self: *Parser) Error!Value {
        self.index = 0;
        self.strictWhitespace();
        const value = try self.strictValue(0, false);
        self.strictWhitespace();
        if (self.index != self.text.len()) return error.ParseValue;
        return value;
    }
    fn strictValue(self: *Parser, depth: usize, comptime tokens: bool) Error!Value {
        if (depth >= max_depth) return error.JsonTooDeep;
        const c = self.char(0) orelse return error.ParseValue;
        switch (c) {
            '"' => return self.strictScalar(tokens, .{ .string = try self.strictString() }),
            '{' => return self.strictObject(depth, tokens),
            '[' => return self.strictArray(depth, tokens),
            else => {},
        }
        if (self.starts(0, "null")) return self.strictScalar(tokens, self.takeLiteral(4, .null));
        if (self.starts(0, "true")) return self.strictScalar(tokens, self.takeLiteral(4, .{ .bool = true }));
        if (self.starts(0, "false")) return self.strictScalar(tokens, self.takeLiteral(5, .{ .bool = false }));
        if (self.starts(0, "NaN")) return self.strictScalar(tokens, self.takeLiteral(3, .{ .float = std.math.nan(f64) }));
        if (self.starts(0, "Infinity")) return self.strictScalar(tokens, self.takeLiteral(8, .{ .float = std.math.inf(f64) }));
        if (self.starts(0, "-Infinity")) return self.strictScalar(tokens, self.takeLiteral(9, .{ .float = -std.math.inf(f64) }));
        // NUMBER_RE: (-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?
        const start = self.index;
        if (self.char(0) == '-') self.index += 1;
        if (self.char(0) == '0') {
            self.index += 1;
        } else if (asciiDigit(self.char(0))) {
            while (asciiDigit(self.char(0))) self.index += 1;
        } else return error.ParseValue;
        var floating = false;
        if (self.char(0) == '.' and asciiDigit(self.char(1))) {
            floating = true;
            self.index += 1;
            while (asciiDigit(self.char(0))) self.index += 1;
        }
        if (oneOf(self.char(0), &.{ 'e', 'E' })) {
            const sign: isize = @intFromBool(oneOf(self.char(1), &.{ '+', '-' }));
            if (asciiDigit(self.char(1 + sign))) {
                floating = true;
                self.index += 1 + sign;
                while (asciiDigit(self.char(0))) self.index += 1;
            }
        }
        const number = try self.text.string(self.arena, start, self.index);
        if (tokens) {
            try self.tokens.append(self.arena, .{ .number = number });
            return .null;
        }
        if (floating) return .{ .float = std.fmt.parseFloat(f64, number) catch unreachable };
        return integer(self.arena, number);
    }

    fn strictScalar(self: *Parser, comptime tokens: bool, value: Value) Allocator.Error!Value {
        if (tokens) try self.tokens.append(self.arena, .{ .scalar = value });
        return value;
    }

    fn strictObject(self: *Parser, depth: usize, comptime tokens: bool) Error!Value {
        self.index += 1;
        var object: runtime.ObjectMap = .empty;
        if (tokens) try self.tokens.append(self.arena, .object_begin);
        self.strictWhitespace();
        if (self.char(0) != '}') {
            while (true) {
                if (self.char(0) != '"') return error.ParseValue;
                const key = try self.strictString();
                if (tokens) try self.tokens.append(self.arena, .{ .scalar = .{ .string = key } });
                self.strictWhitespace();
                if (self.char(0) != ':') return error.ParseValue;
                self.index += 1;
                self.strictWhitespace();
                const value = try self.strictValue(depth + 1, tokens);
                if (!tokens) try object.put(self.arena, key, value);
                self.strictWhitespace();
                if (self.char(0) == '}') break;
                if (self.char(0) != ',') return error.ParseValue;
                self.index += 1;
                self.strictWhitespace();
            }
        }
        self.index += 1;
        if (tokens) try self.tokens.append(self.arena, .object_end);
        return .{ .object = object };
    }

    fn strictArray(self: *Parser, depth: usize, comptime tokens: bool) Error!Value {
        self.index += 1;
        var array: runtime.Array = .init(self.arena);
        if (tokens) try self.tokens.append(self.arena, .array_begin);
        self.strictWhitespace();
        if (self.char(0) != ']') {
            while (true) {
                const value = try self.strictValue(depth + 1, tokens);
                if (!tokens) try array.append(value);
                self.strictWhitespace();
                if (self.char(0) == ']') break;
                if (self.char(0) != ',') return error.ParseValue;
                self.index += 1;
                self.strictWhitespace();
            }
        }
        self.index += 1;
        if (tokens) try self.tokens.append(self.arena, .array_end);
        return .{ .array = array };
    }
    fn takeLiteral(self: *Parser, size: isize, value: Value) Value {
        self.index += size;
        return value;
    }
    fn strictString(self: *Parser) Error![]const u8 {
        self.index += 1;
        const start = self.index;
        while (self.char(0)) |c| {
            if (c == '"') {
                const result = try self.text.string(self.arena, start, self.index);
                self.index += 1;
                return result;
            }
            if (c < 0x20) return error.ParseValue;
            if (c == '\\') break;
            self.index += 1;
        }
        if (self.char(0) == null) return error.ParseValue;
        var out: std.ArrayList(u21) = .empty;
        try self.text.appendRange(self.arena, &out, start, self.index);
        while (self.char(0)) |c| {
            self.index += 1;
            if (c == '"') return encode(self.arena, out.items);
            if (c < 0x20) return error.ParseValue;
            if (c != '\\') {
                try out.append(self.arena, c);
                continue;
            }
            const escaped = self.char(0) orelse return error.ParseValue;
            self.index += 1;
            const decoded: u21 = switch (escaped) {
                '"', '\\', '/' => escaped,
                'b' => 8,
                'f' => 12,
                'n' => 10,
                'r' => 13,
                't' => 9,
                'u' => blk: {
                    var cp = try self.strictHex4();
                    // An invalid second escape is an error, not a lone high surrogate.
                    if (cp >= 0xd800 and cp <= 0xdbff and self.starts(0, "\\u")) {
                        self.index += 2;
                        const low = try self.strictHex4();
                        if (low >= 0xdc00 and low <= 0xdfff) {
                            cp = 0x10000 + (cp - 0xd800) * 1024 + (low - 0xdc00);
                        } else self.index -= 6;
                    }
                    break :blk cp;
                },
                else => return error.ParseValue,
            };
            try out.append(self.arena, decoded);
        }
        return error.ParseValue;
    }
    fn strictHex4(self: *Parser) Error!u21 {
        var cp: u21 = 0;
        for (0..4) |_| {
            cp = cp * 16 + (hex(self.char(0)) orelse return error.ParseValue);
            self.index += 1;
        }
        return cp;
    }
    /// JSONDecoder().raw_decode at the current index (no leading whitespace).
    fn rawDecode(self: *Parser) Error!Value {
        return self.strictValue(0, false);
    }

    // ---- json_parser.py ----

    fn topLevel(self: *Parser) Error!Value {
        const first = try self.parseValue();
        if (self.index >= self.text.len()) return first;
        var results: runtime.Array = .init(self.arena);
        try results.append(first);
        while (self.index < self.text.len()) {
            self.contexts = 0;
            self.deferred_count = 0;
            const comma = self.nextTopLevelIsCommaSeparated();
            const element = try self.parseValue();
            if (truthy(element)) {
                const last = results.items[results.items.len - 1];
                if (!comma and try sameObject(last, element, 0)) {
                    _ = results.pop();
                } else if (!truthy(last)) {
                    _ = results.pop();
                }
                try results.append(element);
            } else self.index += 1;
        }
        return if (results.items.len == 1) results.items[0] else .{ .array = results };
    }
    fn nextTopLevelIsCommaSeparated(self: *const Parser) bool {
        if (self.char(self.scroll(0)) == ',') return true;
        var i = self.index - 1;
        while (i >= 0 and space(self.text.at(i))) i -= 1;
        return i >= 0 and self.text.at(i) == ',';
    }
    fn initialContainerHasNonCommaTrailing(self: *const Parser) Allocator.Error!bool {
        if (self.index != 0) return true;
        var stack: std.ArrayList(u21) = .empty;
        var in_string = false;
        var escaped = false;
        var i: isize = 0;
        while (i < self.text.len()) : (i += 1) {
            const c = self.text.at(i).?;
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') {
                in_string = true;
            } else if (c == '{' or c == '[') {
                try stack.append(self.arena, c);
            } else if (c == '}' or c == ']') {
                const open = stack.pop() orelse return false;
                if (c != closer(open)) return false;
                if (stack.items.len == 0) {
                    var trailing = i + 1;
                    while (trailing < self.text.len() and space(self.text.at(trailing))) trailing += 1;
                    return trailing < self.text.len() and self.text.at(trailing) != ',';
                }
            }
        }
        return false;
    }
    fn tryValidSuffix(self: *Parser) Error!?Value {
        if (self.tried_suffix or self.contexts != 0) return null;
        self.tried_suffix = true;
        const save = self.index;
        // Only JSONDecodeError is caught here; ValueError propagates.
        return self.rawDecode() catch |err| switch (err) {
            error.ParseValue => {
                self.index = save;
                return null;
            },
            else => return err,
        };
    }
    fn missingOpeningObject(self: *Parser, delimiter: u21) Error!?Value {
        if (self.scroll(-self.index) != 0) return null;
        const quote_end = self.skip1(delimiter, 1);
        if (self.char(quote_end) == null) return null;
        if (self.char(self.scroll(quote_end + 1)) != ':') return null;
        if (self.char(self.scrollBack(self.text.len() - self.index - 1)) != '}') return null;
        var candidate = self.text;
        try candidate.replace(self.arena, 0, 0, &.{'{'});
        var parser: Parser = .{ .arena = self.arena, .text = candidate };
        // ValueError (including oversized integers) leaves the input to repair.
        const value = parser.rawDecode() catch |err| switch (err) {
            error.ParseValue, error.IntegerTooLong => return null,
            else => return err,
        };
        parser.strictWhitespace();
        if (parser.index != parser.text.len()) return null;
        self.index = self.text.len();
        return value;
    }
    fn parseValue(self: *Parser) Error!Value {
        try self.enter();
        defer self.leave();
        const deferred_count = self.deferred_count;
        self.deferred_count = 0;
        for (self.deferred[0..deferred_count]) |context| try self.push(context);
        defer {
            for (0..deferred_count) |_| self.pop();
        }
        var check_initial_quote = self.index == 0 and self.contexts == 0;
        while (self.char(0)) |c| {
            if ((c == '{' or c == '[') and try self.initialContainerHasNonCommaTrailing()) {
                if (try self.tryValidSuffix()) |value| return value;
            }
            if (c == '{') {
                self.index += 1;
                return self.parseObject();
            }
            if (c == '[') {
                self.index += 1;
                return self.parseArray(']');
            }
            if (c == '(') {
                if (self.contexts != 0 or self.topLevelParenthesizedCanStartValue()) return self.parseParenthesized();
                self.index += 1;
                continue;
            }
            if (self.contexts != 0 and (quote(c) or alpha(c))) return self.parseString();
            if (self.contexts != 0 and (digit(c) or c == '-' or c == '.')) return self.parseNumber();
            if (c == '#' or c == '/') return self.parseComment();
            if (quote(c) and check_initial_quote) {
                check_initial_quote = false;
                if (try self.missingOpeningObject(c)) |value| return value;
            }
            self.index += 1;
        }
        return .{ .string = "" };
    }
    fn parseParenthesized(self: *Parser) Error!Value {
        const explicit = self.parenthesizedIsExplicitTuple();
        self.index += 1;
        const values = try self.parseArray(')');
        if (explicit or values.array.items.len != 1) return values;
        return values.array.items[0];
    }

    // ---- parser_parenthesized.py ----

    fn parenthesizedIsExplicitTuple(self: *const Parser) bool {
        var i = self.index + 1;
        var parens: usize = 0;
        var squares: usize = 0;
        var braces: usize = 0;
        var in_quote: ?u21 = null;
        var backslashes: usize = 0;
        var saw = false;
        while (i < self.text.len()) : (i += 1) {
            const c = self.text.at(i).?;
            if (c == '\\') {
                backslashes += 1;
                continue;
            }
            if (in_quote) |q| {
                if (c == q and backslashes % 2 == 0) in_quote = null;
                backslashes = 0;
                continue;
            }
            const top = parens == 0 and squares == 0 and braces == 0;
            if (quote(c) and backslashes % 2 == 0) {
                in_quote = c;
                saw = saw or top;
                backslashes = 0;
                continue;
            }
            backslashes = 0;
            if (!space(c) and c != ',' and c != ')' and top) saw = true;
            switch (c) {
                '(' => parens += 1,
                ')' => {
                    if (top) return !saw;
                    if (parens > 0) parens -= 1;
                },
                '[' => squares += 1,
                ']' => if (squares > 0) {
                    squares -= 1;
                },
                '{' => braces += 1,
                '}' => if (braces > 0) {
                    braces -= 1;
                },
                ',' => if (top) return true,
                else => {},
            }
        }
        return !saw;
    }
    fn topLevelParenthesizedCanStartValue(self: *const Parser) bool {
        var back = self.index - 1;
        while (back >= 0) : (back -= 1) {
            const c = self.text.at(back).?;
            if (c == '\n' or c == '\r') break;
            if (!space(c)) return false;
        }
        const first_offset = self.scroll(1);
        const first = self.char(first_offset) orelse return false;
        const lowered_prefix = struct {
            fn is(p: *const Parser, offset: isize, word: []const u8) bool {
                for (word, 0..) |w, k| if (lower(p.char(offset + @as(isize, @intCast(k)))) != w) return false;
                return true;
            }
        }.is;
        if (!oneOf(first, &.{ ')', '{', '[', '(', '"', '\'', 0x201c, 0x201d, '-', '.' }) and !digit(first) and
            !lowered_prefix(self, first_offset, "true") and !lowered_prefix(self, first_offset, "null") and
            !lowered_prefix(self, first_offset, "none") and !lowered_prefix(self, first_offset, "false")) return false;
        var i = self.index + 1;
        var parens: isize = 0;
        var squares: usize = 0;
        var braces: usize = 0;
        var in_quote: ?u21 = null;
        var backslashes: usize = 0;
        while (i < self.text.len()) : (i += 1) {
            const c = self.text.at(i).?;
            if (c == '\\') {
                backslashes += 1;
                continue;
            }
            if (in_quote) |q| {
                if (c == q and backslashes % 2 == 0) in_quote = null;
                backslashes = 0;
                continue;
            }
            if (quote(c) and backslashes % 2 == 0) {
                in_quote = c;
                backslashes = 0;
                continue;
            }
            backslashes = 0;
            switch (c) {
                '(' => parens += 1,
                ')' => {
                    if (parens == 0 and squares == 0 and braces == 0) {
                        i += 1;
                        while (i < self.text.len()) : (i += 1) {
                            const trailer = self.text.at(i).?;
                            if (trailer == '\n' or trailer == '\r') return true;
                            if (!space(trailer)) return false;
                        }
                        return true;
                    }
                    // The source decrements without a lower bound.
                    parens -= 1;
                },
                '[' => squares += 1,
                ']' => if (squares > 0) {
                    squares -= 1;
                },
                '{' => braces += 1,
                '}' => if (braces > 0) {
                    braces -= 1;
                },
                else => {},
            }
        }
        return true;
    }

    // ---- parse_comment.py ----

    fn parseComment(self: *Parser) Error!Value {
        try self.enter();
        defer self.leave();
        while (true) {
            const c = self.char(0);
            if (c == '#') {
                while (self.char(0)) |ch| {
                    if (ch == '\n' or ch == '\r' or (ch == ']' and self.contains(.array)) or
                        (ch == '}' and self.contains(.value)) or (ch == ':' and self.contains(.key))) break;
                    self.index += 1;
                }
            } else if (c == '/') {
                const next = self.char(1);
                if (next == '/') {
                    self.index += 2;
                    while (self.char(0)) |ch| {
                        if (ch == '\n' or ch == '\r') break;
                        self.index += 1;
                    }
                } else if (next == '*') {
                    self.index += 2;
                    // The accumulated comment starts with "/*", so "/*/" closes.
                    var previous: u21 = '*';
                    while (self.char(0)) |ch| {
                        self.index += 1;
                        if (previous == '*' and ch == '/') break;
                        previous = ch;
                    }
                } else self.index += 1;
            }
            if (self.contexts != 0) return .{ .string = "" };
            self.skipWhitespace();
            if (!oneOf(self.char(0), &.{ '#', '/' })) return self.parseValue();
        }
    }

    // ---- parse_number.py ----

    fn parseNumber(self: *Parser) Error!Value {
        const start = self.index;
        var number: std.ArrayList(u8) = .empty;
        const is_array = self.current() == .array;
        while (self.char(0)) |c| {
            const exponent_sign = c == '+' and number.items.len != 0 and
                (number.items[number.items.len - 1] == 'e' or number.items[number.items.len - 1] == 'E');
            const number_char = oneOf(c, &.{ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e', 'E', '/', ',', '_' });
            if (!(number_char or exponent_sign) or (is_array and c == ',')) break;
            if (c != '_') try number.append(self.arena, @intCast(c));
            self.index += 1;
        }
        if (alpha(self.char(0))) {
            self.index = start;
            return self.parseString();
        }
        if (number.items.len != 0 and std.mem.indexOfScalar(u8, "+-eE/,", number.items[number.items.len - 1]) != null) {
            number.items.len -= 1;
            self.index -= 1;
        }
        const text = number.items;
        if (std.mem.indexOfScalar(u8, text, ',') != null) return .{ .string = text };
        if (std.mem.indexOfAny(u8, text, ".eE") != null) {
            const parsed_number = @import("value.zig").floatFromString(self.arena, text) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.InvalidNumber => .{ .string = text },
            };
            return .{ .float = parsed_number };
        }
        const digits_start: usize = @intFromBool(text.len != 0 and text[0] == '-');
        if (digits_start == text.len) return .{ .string = text };
        for (text[digits_start..]) |c| if (!std.ascii.isDigit(c)) return .{ .string = text };
        return integer(self.arena, text) catch |err| switch (err) {
            error.IntegerTooLong => return .{ .string = text },
            else => return err,
        };
    }

    // ---- parse_array.py ----

    fn parseArray(self: *Parser, closing: u21) Error!Value {
        try self.enter();
        defer self.leave();
        const array_is_object_value = self.current() == .value;
        var closed_before_parent_member = false;
        var array: runtime.Array = .init(self.arena);
        try self.push(.array);
        defer self.pop();
        self.skipWhitespace();
        var c = self.char(0);
        while (c != null and c.? != closing and c.? != '}') {
            var value: Value = undefined;
            if (quote(c)) {
                var i = self.skip1(c.?, 1);
                i = self.scroll(i + 1);
                if (self.char(i) == ':') {
                    if (array_is_object_value and array.items.len != 0 and allScalars(array.items)) {
                        closed_before_parent_member = true;
                        break;
                    }
                    value = try self.parseObject();
                } else value = try self.parseString();
            } else value = try self.parseValue();
            if (strictlyEmpty(value) and !oneOf(self.char(0), &.{ closing, ',' })) {
                self.index += 1;
            } else if (value == .string and std.mem.eql(u8, value.string, "...") and self.char(-1) == '.') {
                // A stray ellipsis is ignored.
            } else try array.append(value);
            c = self.char(0);
            while (c != null and c.? != closing and (space(c) or c.? == ',')) {
                self.index += 1;
                c = self.char(0);
            }
        }
        if (!closed_before_parent_member) self.index += 1;
        return .{ .array = array };
    }

    // ---- parse_object.py ----

    fn parseObject(self: *Parser) Error!Value {
        try self.enter();
        defer self.leave();
        var object: runtime.ObjectMap = .empty;
        const start_index = self.index;
        const parsing_object_value = self.current() == .value;
        while ((self.char(0) orelse '}') != '}') {
            self.skipWhitespace();
            if (self.char(0) == ':') self.index += 1;
            const key, const rollback_index = try self.parseObjectKey(&object);
            if (self.contains(.array) and object.contains(key) and !parsing_object_value and
                self.shouldSplitDuplicateObject(rollback_index))
            {
                self.index = rollback_index - 1;
                try self.text.replace(self.arena, self.index + 1, self.index + 1, &.{'{'});
                break;
            }
            self.skipWhitespace();
            if ((self.char(0) orelse '}') == '}') continue;
            self.skipWhitespace();
            self.index += 1;
            const value = try self.parseObjectValue();
            try object.put(self.arena, key, value);
            if (oneOf(self.char(0), &.{ ',', '\'', '"' })) self.index += 1;
            if (self.char(0) == ']' and self.contains(.array)) {
                self.index -= 1;
                break;
            }
            self.skipWhitespace();
        }
        self.index += 1;
        if (try self.repairEmptyObject(object, start_index)) |value| return value;
        return self.completeObject(object);
    }
    fn parseObjectKey(self: *Parser, object: *runtime.ObjectMap) Error!struct { []const u8, isize } {
        var key: []const u8 = "";
        var rollback_index = self.index;
        try self.push(.key);
        defer self.pop();
        while (self.char(0) != null) {
            rollback_index = self.index;
            if (self.char(0) == '[' and key.len == 0 and try self.mergeObjectArrayContinuation(object)) continue;
            const raw = try self.parseString();
            if (raw != .string) return error.ParseValue;
            key = raw.string;
            if (key.len == 0) self.skipWhitespace();
            if (key.len != 0 or oneOf(self.char(0), &.{ ':', '}' })) break;
        }
        return .{ key, rollback_index };
    }
    fn mergeObjectArrayContinuation(self: *Parser, object: *runtime.ObjectMap) Error!bool {
        if (object.count() == 0) return false;
        const previous_key = object.keys()[object.count() - 1];
        if (previous_key.len == 0) return false;
        const previous = object.getPtr(previous_key).?;
        if (previous.* != .array) return false;
        self.index += 1;
        const parsed = try self.parseArray(']');
        const new_array = parsed.array.items;
        const rows = &previous.array;
        var expected: usize = 0;
        var first_length: ?usize = null;
        var uniform = true;
        for (rows.items) |item| {
            if (item != .array) continue;
            if (first_length) |n| {
                uniform = uniform and n == item.array.items.len;
            } else first_length = item.array.items.len;
        }
        if (first_length != null and uniform) expected = first_length.?;
        if (expected != 0) {
            var tail_start = rows.items.len;
            while (tail_start > 0 and rows.items[tail_start - 1] != .array) tail_start -= 1;
            const tail_len = rows.items.len - tail_start;
            if (tail_len != 0 and tail_len % expected == 0) {
                const tail = try self.arena.dupe(Value, rows.items[tail_start..]);
                rows.items.len = tail_start;
                var i: usize = 0;
                while (i < tail.len) : (i += expected) {
                    var row: runtime.Array = .init(self.arena);
                    try row.appendSlice(tail[i..][0..expected]);
                    try rows.append(.{ .array = row });
                }
            }
            if (new_array.len != 0) {
                if (allArrays(new_array)) {
                    try rows.appendSlice(new_array);
                } else try rows.append(parsed);
            }
        } else if (new_array.len == 1 and new_array[0] == .array) {
            try rows.appendSlice(new_array[0].array.items);
        } else try rows.appendSlice(new_array);
        self.skipWhitespace();
        if (self.char(0) == ',') self.index += 1;
        self.skipWhitespace();
        return true;
    }
    fn shouldSplitDuplicateObject(self: *const Parser, rollback_index: isize) bool {
        var lookback = rollback_index - self.index - 1;
        var previous = self.char(lookback);
        while (space(previous)) {
            lookback -= 1;
            previous = self.char(lookback);
        }
        const key_start = self.char(rollback_index - self.index);
        const next = self.char(self.scroll(0));
        return !(quote(key_start) and previous == ',' and next == ':');
    }
    fn parseObjectValue(self: *Parser) Error!Value {
        try self.push(.value);
        defer self.pop();
        self.skipWhitespace();
        if (oneOf(self.char(0), &.{ ',', '}' })) return .{ .string = "" };
        return self.parseValue();
    }
    fn repairEmptyObject(self: *Parser, object: runtime.ObjectMap, start_index: isize) Error!?Value {
        if (object.count() != 0 or self.index - start_index <= 2) return null;
        var attempted: std.ArrayList(u21) = .empty;
        try self.text.appendRange(self.arena, &attempted, start_index - 1, self.index + 1);
        var body: []const u21 = attempted.items[@min(1, attempted.items.len)..];
        if (lastIs(body, '}')) body = body[0 .. body.len - 1];
        body = lstrip(body);
        if (body.len == 0) return null;
        if ((startsWith(body, &.{ '\\', '"' }) and containsSlice(body, &.{ '\\', '"', ':' })) or
            (startsWith(body, &.{ '\\', '\'' }) and containsSlice(body, &.{ '\\', '\'', ':' })))
        {
            const once = try replaceEscape(self.arena, attempted.items, '"');
            const normalized = try replaceEscape(self.arena, once, '\'');
            try self.text.replace(self.arena, start_index - 1, self.index + 1, normalized);
            self.index = start_index;
            try self.push(.key);
            const repaired = self.parseObject();
            self.pop();
            const value = try repaired;
            try self.defer_(.key);
            return value;
        }
        const stripped = lstrip(try stripComments(self.arena, body));
        if (stripped.len == 0) return null;
        var in_quote: ?u21 = null;
        var backslashes: usize = 0;
        for (stripped) |c| {
            if (c == '\\') {
                backslashes += 1;
                continue;
            }
            if (in_quote) |q| {
                if (c == q and backslashes % 2 == 0) in_quote = null;
            } else if (quote(c) and backslashes % 2 == 0) {
                in_quote = c;
            } else if (c == ':' and backslashes % 2 == 0) return null;
            backslashes = 0;
        }
        self.index = start_index;
        try self.push(.key);
        const repaired = self.parseArray(']');
        self.pop();
        const value = try repaired;
        try self.defer_(.key);
        return value;
    }
    fn completeObject(self: *Parser, object_in: runtime.ObjectMap) Error!Value {
        var object = object_in;
        if (self.contexts != 0) {
            const cur = self.current().?;
            if (self.char(0) == '}' and cur != .key and cur != .value) self.index += 1;
            return .{ .object = object };
        }
        self.skipWhitespace();
        if (self.char(0) == ',') {
            self.index += 1;
            self.skipWhitespace();
            if (quote(self.char(0))) {
                const additional = try self.parseObject();
                if (additional == .object) {
                    for (additional.object.keys(), additional.object.values()) |key, value| try object.put(self.arena, key, value);
                }
            }
        }
        return .{ .object = object };
    }

    // ---- parse_string.py and parse_string_helpers ----

    fn parseString(self: *Parser) Error!Value {
        try self.enter();
        defer self.leave();
        var state: StringState = .{};
        if (try self.prepareStringEntry(&state)) |direct| return direct;
        const c = try self.scanStringBody(&state);
        return .{ .string = try encode(self.arena, self.finalizeString(&state, c)) };
    }
    fn simpleQuotedString(self: *Parser) Error!?Value {
        if (self.char(0) != '"') return null;
        const start = self.index + 1;
        var end = start;
        while (end < self.text.len() and self.text.at(end).? != '"') end += 1;
        if (end >= self.text.len()) return null;
        var i = start;
        while (i < end) : (i += 1) if (oneOf(self.text.at(i), &.{ '\\', '\n', '\r' })) return null;
        var next = end + 1;
        while (next < self.text.len() and space(self.text.at(next))) next += 1;
        const next_char = if (next < self.text.len()) self.text.at(next) else null;
        if (self.current()) |cur| switch (cur) {
            .key => if (next_char != ':') return null,
            .value => if (next_char != null and !oneOf(next_char, &.{ ',', '}' })) return null,
            .array => if (next_char != null and !oneOf(next_char, &.{ ',', ']' })) return null,
        } else if (next_char != null) return null;
        const value = try self.text.string(self.arena, start, end);
        self.index = end + 1;
        return .{ .string = value };
    }
    fn parseBooleanOrNull(self: *Parser) Literal {
        const c = lower(self.char(0));
        const start = self.index;
        const literals = [_]struct { []const u8, Value }{
            .{ "true", .{ .bool = true } },
            .{ "false", .{ .bool = false } },
            .{ "null", .null },
            .{ "none", .null },
        };
        for (literals) |entry| {
            const word, const value = entry;
            if (c != word[0]) continue;
            self.index = start;
            var matched = true;
            for (word) |expected| {
                if (lower(self.char(0)) != expected) {
                    matched = false;
                    break;
                }
                self.index += 1;
            }
            if (matched) {
                const next = self.char(0);
                if (next == null or space(next) or oneOf(next, &.{ ',', ')', ']', '}' })) return .{ .value = value };
            }
        }
        self.index = start;
        return .none;
    }
    /// parse_json_llm_block; null is the source's `False` sentinel, which a
    /// parsed JSON false also matches (`is not False`).
    fn llmBlock(self: *Parser) Error!?Value {
        if (!self.starts(0, "```json")) return null;
        const i = self.skip1('`', 7);
        if (!self.starts(i, "```")) return null;
        self.index += 7;
        const value = try self.parseValue();
        if (value == .bool and !value.bool) return null;
        return value;
    }
    fn prepareStringEntry(self: *Parser, state: *StringState) Error!?Value {
        var c = self.char(0);
        if (c == '#' or c == '/') return try self.parseComment();
        while (c != null and !quote(c) and !alnum(c)) {
            self.index += 1;
            c = self.char(0);
        }
        if (c == null) return .{ .string = "" };
        if (try self.simpleQuotedString()) |value| return value;
        if (c.? == '\'') {
            state.lstring = '\'';
            state.rstring = '\'';
        } else if (c.? == 0x201c) {
            state.lstring = 0x201c;
            state.rstring = 0x201d;
        } else if (alnum(c)) {
            if (oneOf(lower(c), &.{ 't', 'f', 'n' }) and self.current() != .key) {
                switch (self.parseBooleanOrNull()) {
                    .value => |value| return value,
                    .none => {},
                }
            }
            state.missing_quotes = true;
        }
        if (!state.missing_quotes) self.index += 1;
        if (self.char(0) == '`') {
            if (try self.llmBlock()) |value| return value;
        }
        if (self.char(0) != state.lstring) return null;
        const cur = self.current();
        if ((cur == .key and self.char(1) == ':') or
            (cur == .value and oneOf(self.char(1), &.{ ',', '}' })) or
            (cur == .array and oneOf(self.char(1), &.{ ',', ']' })))
        {
            self.index += 1;
            return .{ .string = "" };
        }
        if (self.char(1) == state.lstring) return .{ .string = "" };
        var i = self.skip1(state.rstring, 1);
        if (self.char(i + 1) == state.rstring) {
            state.doubled_quotes = true;
            self.index += 1;
            return null;
        }
        if (self.char(i) == state.rstring) {
            const outer_quote = self.skip1(state.rstring, i + 1);
            const after = self.char(self.scroll(outer_quote + 1));
            if (cur == .value and !self.onlyWhitespaceUntil(i) and self.char(outer_quote) == state.rstring and
                (after == null or oneOf(after, &.{ ',', '}' })))
            {
                try state.appendContent(self.arena, &.{state.lstring});
                state.unmatched_delimiter = true;
                self.index += 1;
                return null;
            }
        }
        i = self.scroll(1);
        const next = self.char(i);
        if (quote(next) or oneOf(next, &.{ '{', '[' })) {
            self.index += 1;
            return .{ .string = "" };
        }
        if (!oneOf(next, &.{ ',', ']', '}' })) self.index += 1;
        return null;
    }
    fn appendLiteral(self: *Parser, state: *StringState, c: u21) Error!?u21 {
        try state.appendContent(self.arena, &.{c});
        self.index += 1;
        return self.char(0);
    }
    fn cachedSkip(self: *Parser, state: *StringState, targets: []const u21, offset: isize) Error!isize {
        std.debug.assert(targets.len <= 6);
        const start = self.index + offset;
        var entry: ?*CacheEntry = null;
        for (state.cache.items) |*cached| {
            if (std.mem.eql(u21, cached.targets[0..cached.count], targets)) entry = cached;
        }
        if (entry) |cached| {
            if (cached.match) |match| {
                if (cached.start <= start and start <= match) return match - self.index;
            } else if (start >= cached.start) return self.text.len() - self.index;
        }
        const found = self.skip(targets, offset);
        var record: CacheEntry = .{ .targets = undefined, .count = targets.len, .start = start, .match = null };
        @memcpy(record.targets[0..targets.len], targets);
        if (self.char(found) == null) {
            try storeCache(self.arena, state, entry, record);
            return found;
        }
        const match = self.index + found;
        if (match == 0 or self.text.at(match - 1) != '\\') {
            record.match = match;
            try storeCache(self.arena, state, entry, record);
        }
        return found;
    }
    fn normalizeEscape(self: *Parser, state: *StringState, c: u21) Error!struct { bool, ?u21 } {
        const active = state.active();
        if (state.inLow() and c == '"') {
            state.replaceTrailingBackslash(c);
            state.low_spans -= 1;
            self.index += 1;
            return .{ true, self.char(0) };
        }
        if (c == '\\') {
            const run_start = self.index - 1;
            var run_end = self.index + 1;
            while (run_end < self.text.len() and self.text.at(run_end).? == '\\') run_end += 1;
            const run_length: usize = @intCast(run_end - run_start);
            if (run_length % 2 == 0 and self.char(run_end - self.index) != active) {
                // Keep the already-buffered backslash. Adding more does not
                // affect brace or regex-class tracking.
                std.debug.assert(lastIs(state.acc.items, '\\'));
                try state.acc.appendNTimes(self.arena, '\\', run_length / 2 - 1);
                self.index = run_end;
                return .{ true, self.char(0) };
            }
        }
        if (oneOf(c, &.{ active, 't', 'n', 'r', 'b', '\\' })) {
            const decoded: u21 = switch (c) {
                't' => '\t',
                'n' => '\n',
                'r' => '\r',
                'b' => 8,
                else => c,
            };
            state.replaceTrailingBackslash(decoded);
            self.index += 1;
            var next = self.char(0);
            while (next != null and lastIs(state.acc.items, '\\') and oneOf(next, &.{ active, '\\' })) {
                state.replaceTrailingBackslash(next.?);
                self.index += 1;
                next = self.char(0);
            }
            return .{ true, next };
        }
        if (c == 'u' or c == 'x') {
            const count: isize = if (c == 'u') 4 else 2;
            var cp: u21 = 0;
            var k: isize = 1;
            while (k <= count) : (k += 1) {
                const h = hex(self.char(k)) orelse return .{ false, c };
                cp = cp * 16 + h;
            }
            state.replaceTrailingBackslash(cp);
            self.index += 1 + count;
            return .{ true, self.char(0) };
        }
        if (c == 0x201e or (quote(c) and c != active)) {
            state.replaceTrailingBackslash(c);
            self.index += 1;
            return .{ true, self.char(0) };
        }
        return .{ false, c };
    }
    fn bareKeyFollowedByColon(self: *const Parser, start: isize) bool {
        var i = start;
        const first = self.char(i);
        if (!(alnum(first) or first == '_')) return false;
        while (alnum(self.char(i)) or oneOf(self.char(i), &.{ '_', '-' })) i += 1;
        return self.char(self.scroll(i)) == ':';
    }
    fn scrollCommentPrefixedMemberStart(self: *const Parser, start: isize) isize {
        var i = self.scroll(start);
        while (true) {
            const c = self.char(i);
            if (c == '#') {
                while (self.char(i) != null and !oneOf(self.char(i), &.{ '\n', '\r' })) i += 1;
                i = self.scroll(i);
                continue;
            }
            if (c == '/') {
                const next = self.char(i + 1);
                if (next == '/') {
                    i += 2;
                    while (self.char(i) != null and !oneOf(self.char(i), &.{ '\n', '\r' })) i += 1;
                    i = self.scroll(i);
                    continue;
                }
                if (next == '*') {
                    i += 2;
                    while (true) {
                        const ch = self.char(i) orelse return i;
                        if (ch == '*' and self.char(i + 1) == '/') {
                            i += 2;
                            break;
                        }
                        i += 1;
                    }
                    i = self.scroll(i);
                    continue;
                }
            }
            return i;
        }
    }
    fn objectMemberStartsAt(self: *const Parser, start: isize) bool {
        const member = self.char(start) orelse return false;
        if (member == '}') return false;
        if (quote(member)) {
            const end_quote = matchingQuote(member);
            const key_end = self.skip1(end_quote, start + 1);
            if (self.char(key_end) != end_quote) return false;
            return self.char(self.scroll(key_end + 1)) == ':';
        }
        if (alnum(member) or member == '_') return self.bareKeyFollowedByColon(start);
        return false;
    }
    fn quotedObjectMemberFollows(self: *const Parser, quote_index: isize) bool {
        const comma = self.scroll(quote_index + 1);
        if (self.char(comma) != ',') return false;
        return self.objectMemberStartsAt(self.scrollCommentPrefixedMemberStart(comma + 1));
    }
    fn postFenceContainerStartsNextMember(self: *const Parser, container_end: isize) bool {
        const after_index = self.scroll(container_end);
        const after = self.char(after_index);
        if (after == null or after.? == '}') return true;
        if (after.? != ',') return false;
        const next = self.scrollCommentPrefixedMemberStart(after_index + 1);
        return self.char(next) == null or self.char(next) == '}' or self.objectMemberStartsAt(next);
    }
    fn startsNestedInlineContainer(self: *const Parser, at: isize) bool {
        const open = self.char(at);
        var back = at - 1;
        while (back >= 0) : (back -= 1) {
            const previous = self.char(back) orelse return true;
            if (space(previous)) continue;
            if (opener(previous)) return true;
            if (previous != ',' and previous != ':') return false;
            const next_index = self.scroll(at + 1);
            const next = self.char(next_index);
            if (open == '[' or open == '(') {
                return oneOf(next, &.{ ']', ')', '"', '\'', 0x201c, 0x201d, '-', '[', '{', '(', 't', 'f', 'n' }) or digit(next);
            }
            if (open != '{') return false;
            if (next == '}' or quote(next)) return true;
            return previous == ':' and self.bareKeyFollowedByColon(next_index);
        }
        return true;
    }
    fn skipInlineContainer(self: *Parser, at: isize) Error!?isize {
        const open = self.char(at);
        if (!opener(open)) return at;
        var stack: std.ArrayList(u21) = .empty;
        try stack.append(self.arena, closer(open.?));
        var i = at + 1;
        while (stack.items.len != 0) : (i += 1) {
            const c = self.char(i) orelse return null;
            if (quote(c)) {
                const end_quote = matchingQuote(c);
                i = self.skip1(end_quote, i + 1);
                if (self.char(i) != end_quote) return null;
            } else if (opener(c) and self.startsNestedInlineContainer(i)) {
                try stack.append(self.arena, closer(c));
            } else if (c == stack.items[stack.items.len - 1]) {
                _ = stack.pop();
                if (stack.items.len == 0) return i + 1;
            }
        }
        unreachable;
    }
    const FenceQuoteLookahead = struct {
        start: isize = -1,
        end: isize = -1,
    };
    fn braceBeforeFenceBelongsToString(self: *Parser, state: *StringState, fence: isize, lookahead: *FenceQuoteLookahead) Error!bool {
        var quote_search = fence + 3;
        const next_content = self.scrollCommentPrefixedMemberStart(quote_search);
        var keep_container = false;
        if (opener(self.char(next_content))) {
            if (try self.skipInlineContainer(next_content)) |container_end| {
                if (self.postFenceContainerStartsNextMember(container_end)) return false;
                keep_container = true;
                quote_search = container_end;
            }
        }
        // Each search starts after a backtick or a container closer, so it
        // cannot split a backslash run. Reuse the first qualifying quote (or
        // EOF) within the searched interval. A later fence can start before
        // that interval if the previous fence skipped a container.
        const start = self.index + quote_search;
        if (start < lookahead.start or start > lookahead.end) {
            const outer = state.outer();
            var offset = self.skip1(outer, quote_search);
            while (self.char(offset) == outer) {
                const after = self.char(self.scroll(offset + 1));
                if (after == null or oneOf(after, &.{ ',', '}', ']' })) break;
                offset = self.skip1(outer, offset + 1);
            }
            lookahead.* = .{ .start = start, .end = self.index + offset };
        }
        if (lookahead.end == self.text.len()) return false;
        if (keep_container) state.pending_inline_container = true;
        return true;
    }
    fn bareMemberHasRecoverableValue(self: *Parser, state: *StringState, value_offset: isize) Error!bool {
        const start = self.scroll(value_offset);
        const value = self.char(start);
        if (quote(value) or oneOf(value, &.{ '{', '[', '-' })) return true;
        if (digit(value)) return true;
        for ([_][]const u8{ "true", "false", "null" }) |word| {
            if (self.starts(start, word)) {
                const end = self.char(start + @as(isize, @intCast(word.len)));
                if (end == null or space(end) or oneOf(end, &.{ ',', '}', ']' })) return true;
            }
        }
        const end = try self.cachedSkip(state, &.{ '"', '\'', 0x201c, 0x201d, '}' }, start);
        return self.char(end) == '}';
    }
    const Comma = enum { container, member, string, string_no_future_delimiter };
    fn classifyCommaQuoteTail(self: *const Parser, state: *StringState, end_quote: isize) Comma {
        if (self.char(end_quote) == null) return .string;
        // Many embedded commas can look ahead to the same quote. The
        // whitespace and colon check after it depend only on its position.
        const index = self.index + end_quote;
        if (state.comma_quote_tail.index != index) {
            state.comma_quote_tail = .{
                .index = index,
                .is_member = self.char(self.scroll(end_quote + 1)) == ':',
            };
        }
        return if (state.comma_quote_tail.is_member) .member else .string;
    }
    fn classifyObjectValueComma(self: *Parser, state: *StringState) Error!Comma {
        const next_index = self.scroll(1);
        const next = self.char(next_index) orelse return .member;
        if (next == '}') return .member;
        if (quote(next)) {
            return self.classifyCommaQuoteTail(state, self.skip1(next, next_index + 1));
        }
        if (next == '`') {
            var i = next_index + 1;
            while (alnum(self.char(i)) or oneOf(self.char(i), &.{ '_', '-' })) i += 1;
            return if (self.char(self.scroll(i)) == ':') .member else .string;
        }
        if (alnum(next) or next == '_') {
            var i = next_index;
            while (alnum(self.char(i)) or oneOf(self.char(i), &.{ '_', '-' })) i += 1;
            i = self.scroll(i);
            if (self.char(i) == ':' and try self.bareMemberHasRecoverableValue(state, i + 1)) return .member;
        }
        if (next == '{' or next == '[') return .container;
        const special_index = try self.cachedSkip(state, &.{ '"', '\'', 0x201c, 0x201d, '{', '[' }, next_index);
        const special = self.char(special_index) orelse return .string_no_future_delimiter;
        if (special == '{' or special == '[') return .string;
        return self.classifyCommaQuoteTail(state, try self.cachedSkip(state, &.{special}, special_index + 1));
    }
    const Delimiter = struct { handled: bool, char: ?u21, stop: bool };
    fn handleRightDelimiter(self: *Parser, state: *StringState, c: u21) Error!Delimiter {
        const outer = state.outer();
        const cur = self.current();
        if (state.doubled_quotes and self.char(1) == outer) {
            self.index += 1;
            return .{ .handled = true, .char = c, .stop = false };
        }
        if (state.missing_quotes and cur == .value) {
            var i: isize = 1;
            var next = self.char(i);
            while (next != null and next.? != outer and next.? != state.lstring) {
                i += 1;
                next = self.char(i);
            }
            if (next != null) {
                i = self.scroll(i + 1);
                if (self.char(i) == ':') {
                    self.index -= 1;
                    return .{ .handled = false, .char = self.char(0), .stop = true };
                }
            }
            return .{ .handled = false, .char = c, .stop = false };
        }
        if (state.unmatched_delimiter) {
            state.unmatched_delimiter = false;
            return .{ .handled = true, .char = try self.appendLiteral(state, c), .stop = false };
        }
        var i: isize = 1;
        var next = self.char(i);
        var check_comma = true;
        while (next != null and next.? != outer and next.? != state.lstring) {
            const n = next.?;
            if (check_comma and alpha(n)) check_comma = false;
            if ((self.contains(.key) and (n == ':' or n == '}')) or
                (self.contains(.value) and n == '}') or
                (self.contains(.array) and (n == ']' or n == ',')) or
                (check_comma and cur == .value and n == ',')) break;
            i += 1;
            next = self.char(i);
        }
        if (next == ',' and cur == .value) {
            i = self.skip1(outer, i + 1);
            i = self.scroll(i + 1);
            if (oneOf(self.char(i), &.{ '}', ',' })) {
                return .{ .handled = true, .char = try self.appendLiteral(state, c), .stop = false };
            }
        } else if (next == outer and self.char(i - 1) != '\\') {
            if (self.onlyWhitespaceUntil(i) and !(cur == .value and self.quotedObjectMemberFollows(i))) {
                return .{ .handled = false, .char = c, .stop = true };
            }
            if (cur == .value) {
                if (self.quotedObjectMemberFollows(i)) {
                    return .{ .handled = true, .char = try self.appendLiteral(state, c), .stop = false };
                }
                i = self.skip1(outer, i + 1) + 1;
                next = self.char(i);
                while (next != null and next.? != ':') {
                    if (oneOf(next, &.{ ',', ']', '}' }) or (next.? == outer and self.char(i - 1) != '\\')) break;
                    i += 1;
                    next = self.char(i);
                }
                if (next != ':') {
                    state.unmatched_delimiter = !state.unmatched_delimiter;
                    return .{ .handled = true, .char = try self.appendLiteral(state, c), .stop = false };
                }
            } else if (cur == .array) {
                var even = next == outer;
                while (next == outer) {
                    i = self.skip(&.{ outer, ']' }, i + 1);
                    next = self.char(i);
                    if (next != outer) {
                        even = false;
                        break;
                    }
                    i = self.skip(&.{ outer, ']' }, i + 1);
                    next = self.char(i);
                }
                if (even) {
                    state.unmatched_delimiter = !state.unmatched_delimiter;
                    return .{ .handled = true, .char = try self.appendLiteral(state, c), .stop = false };
                }
                return .{ .handled = false, .char = c, .stop = true };
            } else if (cur == .key) {
                return .{ .handled = true, .char = try self.appendLiteral(state, c), .stop = false };
            }
        }
        return .{ .handled = false, .char = c, .stop = false };
    }
    fn braceMayEndValueBeforeQuote(self: *const Parser, opening_quote: u21, closing_offset: isize) bool {
        const start = self.scroll(closing_offset + 1);
        const next = self.char(start);
        if (next == null or oneOf(next, &.{ ',', '}' })) return false;
        const opening_offset = self.skip1(opening_quote, start);
        if (self.char(opening_offset) == null) return false;
        const after = self.char(self.scroll(opening_offset + 1));
        return after == null or after == ':';
    }
    fn scanStringBody(self: *Parser, state: *StringState) Error!?u21 {
        const outer = state.outer();
        var no_future_colon = false;
        var next_closing_brace: isize = -1;
        var next_array_string_quote: isize = -1;
        var next_regex_class_close: isize = -1;
        var key_opening_quote: isize = -1;
        var key_closing_quote: struct { index: isize = -1, ends_key: bool = false } = .{};
        var quote_tail: struct { index: isize = -1, may_end_value: bool = false } = .{};
        var fence_quotes: FenceQuoteLookahead = .{};
        var c = self.char(0);
        while (c != null and (c.? != outer or state.inLow())) {
            const ch = c.?;
            const cur = self.current();
            if (state.missing_quotes) {
                if (cur == .key and (ch == ':' or space(ch))) break;
                if (cur == .array and (ch == ']' or ch == ',')) break;
            }
            if (ch == 0x201e and !lastIs(state.acc.items, '\\')) {
                state.low_spans += 1;
                c = try self.appendLiteral(state, ch);
                continue;
            }
            if (state.inLow() and ch == 0x201d) {
                state.low_spans -= 1;
                c = try self.appendLiteral(state, ch);
                continue;
            }
            if ((state.pending_inline_container or
                (cur == .value and ch == '{' and self.char(-1) != '\\' and self.bareKeyFollowedByColon(self.scroll(1)))) and
                opener(ch) and !lastIs(state.acc.items, '\\'))
            {
                if (try self.skipInlineContainer(0)) |end| {
                    state.pending_inline_container = false;
                    state.inline_stack.clearRetainingCapacity();
                    var content: std.ArrayList(u21) = .empty;
                    try self.text.appendRange(self.arena, &content, self.index, self.index + end);
                    try state.appendContent(self.arena, content.items);
                    self.index += end;
                    c = self.char(0);
                    continue;
                }
            }
            if (cur == .value and ch == ',' and !state.pending_inline_container and state.inline_stack.items.len == 0) {
                const classification: Comma = if (state.no_future_delimiter) .string else try self.classifyObjectValueComma(state);
                if (classification == .member) break;
                if (classification == .string_no_future_delimiter) state.no_future_delimiter = true;
                state.pending_inline_container = classification == .container;
                c = try self.appendLiteral(state, ch);
                continue;
            }
            if (try self.updateInlineContainerStack(state, ch)) {
                c = try self.appendLiteral(state, ch);
                continue;
            }
            if (cur == .value and ch == '}' and !lastIs(state.acc.items, outer)) {
                if (state.unmatched_braces != 0) {
                    c = try self.appendLiteral(state, ch);
                    continue;
                }
                var missing = true;
                self.skipWhitespace();
                if (self.char(1) == '\\') missing = false;
                const quote_offset = try self.cachedSkip(state, &.{outer}, 1);
                if (self.char(quote_offset) != null) {
                    // Braces that see the same quote also see the same tail.
                    // Cache its classification, not the cursor-dependent check above.
                    const quote_index = self.index + quote_offset;
                    if (quote_tail.index != quote_index) {
                        quote_tail = .{
                            .index = quote_index,
                            .may_end_value = self.braceMayEndValueBeforeQuote(state.lstring, quote_offset),
                        };
                    }
                    missing = missing and quote_tail.may_end_value;
                } else {
                    // The search starts after `}` (or subsequent whitespace),
                    // never inside a backslash run. As the cursor advances,
                    // a colon miss stays valid and the next brace stays valid
                    // until the search start passes it.
                    if (!no_future_colon) {
                        if (self.char(self.skip1(':', 1)) != null) break;
                        no_future_colon = true;
                    }
                    const i = self.scroll(1);
                    const start = self.index + i;
                    if (start > next_closing_brace) {
                        next_closing_brace = self.index + self.skip1('}', i);
                    }
                    if (next_closing_brace - start > 1) missing = false;
                }
                if (missing) break;
            }
            if (ch == ']' and self.contains(.array) and !lastIs(state.acc.items, outer)) {
                // A search starting at `]` never starts inside a backslash
                // run, so later brackets before this quote see the same match.
                if (self.index > next_array_string_quote) {
                    const offset = self.skip1(outer, 0);
                    if (self.char(offset) == null) break;
                    next_array_string_quote = self.index + offset;
                }
            }
            if (cur == .value and ch == '}') {
                const i = self.scroll(1);
                const next = self.char(i);
                if (next == '`' and self.char(i + 1) == '`' and self.char(i + 2) == '`') {
                    if (try self.braceBeforeFenceBelongsToString(state, i, &fence_quotes)) {
                        c = try self.appendLiteral(state, ch);
                        continue;
                    }
                    break;
                }
                if (next == null) break;
            }
            try state.appendContent(self.arena, &.{ch});
            self.index += 1;
            c = self.char(0);
            if (c == null) break;
            if (lastIs(state.acc.items, '\\')) {
                const handled, const next_char = try self.normalizeEscape(state, c.?);
                c = next_char;
                if (handled) continue;
                if (c == null) break;
            }
            if (c.? == ':' and !state.missing_quotes and self.current() == .key) {
                // Both searches start outside a backslash run. Several opening
                // smart quotes can share one closing quote (or the same EOF).
                if (self.index >= key_opening_quote) {
                    key_opening_quote = self.index + self.skip1(state.lstring, 1);
                }
                if (key_opening_quote == self.text.len()) break;
                if (key_opening_quote >= key_closing_quote.index) {
                    const second = self.skip1(outer, key_opening_quote - self.index + 1);
                    key_closing_quote = .{
                        .index = self.index + second,
                        .ends_key = self.char(second) != null and
                            oneOf(self.char(self.scroll(second + 1)), &.{ ',', '}' }),
                    };
                }
                if (key_closing_quote.ends_key) break;
            }
            if (state.inLow() and c.? == '"') {
                state.low_spans -= 1;
                c = try self.appendLiteral(state, c.?);
                continue;
            }
            if (c.? == outer and self.current() == .value and state.compact_regex_class) {
                // Each search starts after a quote, outside any backslash run.
                // Reuse the same unescaped bracket (or EOF) until we pass it.
                if (self.index >= next_regex_class_close) {
                    next_regex_class_close = self.index + self.skip1(']', 1);
                }
                if (next_regex_class_close < self.text.len()) {
                    c = try self.appendLiteral(state, c.?);
                    continue;
                }
            }
            if (c.? == outer and state.acc.items.len != 0 and !lastIs(state.acc.items, '\\')) {
                const result = try self.handleRightDelimiter(state, c.?);
                c = result.char;
                if (result.stop) break;
                if (result.handled) continue;
            }
        }
        return c;
    }
    /// update_inline_container_stack; returns whether to keep the character.
    fn updateInlineContainerStack(self: *Parser, state: *StringState, c: u21) Allocator.Error!bool {
        const stack = &state.inline_stack;
        if (c == '{' or c == '[') {
            if (state.pending_inline_container) {
                try stack.append(self.arena, c);
                state.pending_inline_container = false;
                return false;
            }
            if (stack.items.len != 0) try stack.append(self.arena, c);
        }
        if (stack.items.len != 0) {
            const top = stack.items[stack.items.len - 1];
            if ((c == '}' and top == '{') or (c == ']' and top == '[')) {
                _ = stack.pop();
                return true;
            }
        }
        return false;
    }
    fn finalizeString(self: *Parser, state: *StringState, c: ?u21) []const u21 {
        const outer = state.outer();
        if (c != null and state.missing_quotes and self.current() == .key and space(c)) {
            self.skipWhitespace();
            if (!oneOf(self.char(0), &.{ ':', ',' })) return &.{};
        }
        var acc: []const u21 = state.acc.items;
        if (c != outer) {
            acc = rstrip(acc);
        } else self.index += 1;
        if (state.missing_quotes or lastIs(acc, '\n')) acc = rstrip(acc);
        return acc;
    }
};

fn storeCache(arena: Allocator, state: *StringState, existing: ?*CacheEntry, record: CacheEntry) Allocator.Error!void {
    if (existing) |entry| entry.* = record else try state.cache.append(arena, record);
}
fn allScalars(items: []const Value) bool {
    for (items) |item| if (item == .array or item == .object) return false;
    return true;
}
fn allArrays(items: []const Value) bool {
    for (items) |item| if (item != .array) return false;
    return true;
}
fn lstrip(items: []const u21) []const u21 {
    var start: usize = 0;
    while (start < items.len and unicode.isSpace(items[start])) start += 1;
    return items[start..];
}
fn rstrip(items: []const u21) []const u21 {
    var end = items.len;
    while (end > 0 and unicode.isSpace(items[end - 1])) end -= 1;
    return items[0..end];
}
fn startsWith(items: []const u21, prefix: []const u21) bool {
    return std.mem.startsWith(u21, items, prefix);
}
fn containsSlice(items: []const u21, needle: []const u21) bool {
    return std.mem.indexOf(u21, items, needle) != null;
}
/// str.replace('\\' + q, q), scanning left to right without overlap.
fn replaceEscape(arena: Allocator, items: []const u21, q: u21) Allocator.Error![]const u21 {
    var out: std.ArrayList(u21) = .empty;
    try out.ensureTotalCapacity(arena, items.len);
    var i: usize = 0;
    while (i < items.len) {
        if (items[i] == '\\' and i + 1 < items.len and items[i + 1] == q) {
            out.appendAssumeCapacity(q);
            i += 2;
        } else {
            out.appendAssumeCapacity(items[i]);
            i += 1;
        }
    }
    return out.items;
}
/// _strip_comments_for_empty_object_classification.
fn stripComments(arena: Allocator, body: []const u21) Allocator.Error![]const u21 {
    var out: std.ArrayList(u21) = .empty;
    try out.ensureTotalCapacity(arena, body.len);
    var in_quote: ?u21 = null;
    var backslashes: usize = 0;
    var i: usize = 0;
    while (i < body.len) {
        const c = body[i];
        const next: ?u21 = if (i + 1 < body.len) body[i + 1] else null;
        if (c == '\\') {
            backslashes += 1;
            out.appendAssumeCapacity(c);
            i += 1;
            continue;
        }
        if (in_quote) |q| {
            out.appendAssumeCapacity(c);
            if (c == q and backslashes % 2 == 0) in_quote = null;
            backslashes = 0;
            i += 1;
            continue;
        }
        if (quote(c) and backslashes % 2 == 0) {
            in_quote = c;
            out.appendAssumeCapacity(c);
            backslashes = 0;
            i += 1;
            continue;
        }
        backslashes = 0;
        if (c == '#' or (c == '/' and next == '/')) {
            i += if (c == '/') 2 else 1;
            while (i < body.len and body[i] != '\n' and body[i] != '\r') i += 1;
            continue;
        }
        if (c == '/' and next == '*') {
            i += 2;
            while (i + 1 < body.len and !(body[i] == '*' and body[i + 1] == '/')) i += 1;
            i = @min(i + 2, body.len);
            continue;
        }
        out.appendAssumeCapacity(c);
        i += 1;
    }
    return out.items;
}

// Generated by tool_argument_repair_oracle.py --zig-unicode-tables from CPython 3.11.17 (Unicode 14.0.0).
const alpha_ranges = [_]Range{
    .{ .lo = 0x41, .hi = 0x5a },
    .{ .lo = 0x61, .hi = 0x7a },
    .{ .lo = 0xaa, .hi = 0xaa },
    .{ .lo = 0xb5, .hi = 0xb5 },
    .{ .lo = 0xba, .hi = 0xba },
    .{ .lo = 0xc0, .hi = 0xd6 },
    .{ .lo = 0xd8, .hi = 0xf6 },
    .{ .lo = 0xf8, .hi = 0x2c1 },
    .{ .lo = 0x2c6, .hi = 0x2d1 },
    .{ .lo = 0x2e0, .hi = 0x2e4 },
    .{ .lo = 0x2ec, .hi = 0x2ec },
    .{ .lo = 0x2ee, .hi = 0x2ee },
    .{ .lo = 0x370, .hi = 0x374 },
    .{ .lo = 0x376, .hi = 0x377 },
    .{ .lo = 0x37a, .hi = 0x37d },
    .{ .lo = 0x37f, .hi = 0x37f },
    .{ .lo = 0x386, .hi = 0x386 },
    .{ .lo = 0x388, .hi = 0x38a },
    .{ .lo = 0x38c, .hi = 0x38c },
    .{ .lo = 0x38e, .hi = 0x3a1 },
    .{ .lo = 0x3a3, .hi = 0x3f5 },
    .{ .lo = 0x3f7, .hi = 0x481 },
    .{ .lo = 0x48a, .hi = 0x52f },
    .{ .lo = 0x531, .hi = 0x556 },
    .{ .lo = 0x559, .hi = 0x559 },
    .{ .lo = 0x560, .hi = 0x588 },
    .{ .lo = 0x5d0, .hi = 0x5ea },
    .{ .lo = 0x5ef, .hi = 0x5f2 },
    .{ .lo = 0x620, .hi = 0x64a },
    .{ .lo = 0x66e, .hi = 0x66f },
    .{ .lo = 0x671, .hi = 0x6d3 },
    .{ .lo = 0x6d5, .hi = 0x6d5 },
    .{ .lo = 0x6e5, .hi = 0x6e6 },
    .{ .lo = 0x6ee, .hi = 0x6ef },
    .{ .lo = 0x6fa, .hi = 0x6fc },
    .{ .lo = 0x6ff, .hi = 0x6ff },
    .{ .lo = 0x710, .hi = 0x710 },
    .{ .lo = 0x712, .hi = 0x72f },
    .{ .lo = 0x74d, .hi = 0x7a5 },
    .{ .lo = 0x7b1, .hi = 0x7b1 },
    .{ .lo = 0x7ca, .hi = 0x7ea },
    .{ .lo = 0x7f4, .hi = 0x7f5 },
    .{ .lo = 0x7fa, .hi = 0x7fa },
    .{ .lo = 0x800, .hi = 0x815 },
    .{ .lo = 0x81a, .hi = 0x81a },
    .{ .lo = 0x824, .hi = 0x824 },
    .{ .lo = 0x828, .hi = 0x828 },
    .{ .lo = 0x840, .hi = 0x858 },
    .{ .lo = 0x860, .hi = 0x86a },
    .{ .lo = 0x870, .hi = 0x887 },
    .{ .lo = 0x889, .hi = 0x88e },
    .{ .lo = 0x8a0, .hi = 0x8c9 },
    .{ .lo = 0x904, .hi = 0x939 },
    .{ .lo = 0x93d, .hi = 0x93d },
    .{ .lo = 0x950, .hi = 0x950 },
    .{ .lo = 0x958, .hi = 0x961 },
    .{ .lo = 0x971, .hi = 0x980 },
    .{ .lo = 0x985, .hi = 0x98c },
    .{ .lo = 0x98f, .hi = 0x990 },
    .{ .lo = 0x993, .hi = 0x9a8 },
    .{ .lo = 0x9aa, .hi = 0x9b0 },
    .{ .lo = 0x9b2, .hi = 0x9b2 },
    .{ .lo = 0x9b6, .hi = 0x9b9 },
    .{ .lo = 0x9bd, .hi = 0x9bd },
    .{ .lo = 0x9ce, .hi = 0x9ce },
    .{ .lo = 0x9dc, .hi = 0x9dd },
    .{ .lo = 0x9df, .hi = 0x9e1 },
    .{ .lo = 0x9f0, .hi = 0x9f1 },
    .{ .lo = 0x9fc, .hi = 0x9fc },
    .{ .lo = 0xa05, .hi = 0xa0a },
    .{ .lo = 0xa0f, .hi = 0xa10 },
    .{ .lo = 0xa13, .hi = 0xa28 },
    .{ .lo = 0xa2a, .hi = 0xa30 },
    .{ .lo = 0xa32, .hi = 0xa33 },
    .{ .lo = 0xa35, .hi = 0xa36 },
    .{ .lo = 0xa38, .hi = 0xa39 },
    .{ .lo = 0xa59, .hi = 0xa5c },
    .{ .lo = 0xa5e, .hi = 0xa5e },
    .{ .lo = 0xa72, .hi = 0xa74 },
    .{ .lo = 0xa85, .hi = 0xa8d },
    .{ .lo = 0xa8f, .hi = 0xa91 },
    .{ .lo = 0xa93, .hi = 0xaa8 },
    .{ .lo = 0xaaa, .hi = 0xab0 },
    .{ .lo = 0xab2, .hi = 0xab3 },
    .{ .lo = 0xab5, .hi = 0xab9 },
    .{ .lo = 0xabd, .hi = 0xabd },
    .{ .lo = 0xad0, .hi = 0xad0 },
    .{ .lo = 0xae0, .hi = 0xae1 },
    .{ .lo = 0xaf9, .hi = 0xaf9 },
    .{ .lo = 0xb05, .hi = 0xb0c },
    .{ .lo = 0xb0f, .hi = 0xb10 },
    .{ .lo = 0xb13, .hi = 0xb28 },
    .{ .lo = 0xb2a, .hi = 0xb30 },
    .{ .lo = 0xb32, .hi = 0xb33 },
    .{ .lo = 0xb35, .hi = 0xb39 },
    .{ .lo = 0xb3d, .hi = 0xb3d },
    .{ .lo = 0xb5c, .hi = 0xb5d },
    .{ .lo = 0xb5f, .hi = 0xb61 },
    .{ .lo = 0xb71, .hi = 0xb71 },
    .{ .lo = 0xb83, .hi = 0xb83 },
    .{ .lo = 0xb85, .hi = 0xb8a },
    .{ .lo = 0xb8e, .hi = 0xb90 },
    .{ .lo = 0xb92, .hi = 0xb95 },
    .{ .lo = 0xb99, .hi = 0xb9a },
    .{ .lo = 0xb9c, .hi = 0xb9c },
    .{ .lo = 0xb9e, .hi = 0xb9f },
    .{ .lo = 0xba3, .hi = 0xba4 },
    .{ .lo = 0xba8, .hi = 0xbaa },
    .{ .lo = 0xbae, .hi = 0xbb9 },
    .{ .lo = 0xbd0, .hi = 0xbd0 },
    .{ .lo = 0xc05, .hi = 0xc0c },
    .{ .lo = 0xc0e, .hi = 0xc10 },
    .{ .lo = 0xc12, .hi = 0xc28 },
    .{ .lo = 0xc2a, .hi = 0xc39 },
    .{ .lo = 0xc3d, .hi = 0xc3d },
    .{ .lo = 0xc58, .hi = 0xc5a },
    .{ .lo = 0xc5d, .hi = 0xc5d },
    .{ .lo = 0xc60, .hi = 0xc61 },
    .{ .lo = 0xc80, .hi = 0xc80 },
    .{ .lo = 0xc85, .hi = 0xc8c },
    .{ .lo = 0xc8e, .hi = 0xc90 },
    .{ .lo = 0xc92, .hi = 0xca8 },
    .{ .lo = 0xcaa, .hi = 0xcb3 },
    .{ .lo = 0xcb5, .hi = 0xcb9 },
    .{ .lo = 0xcbd, .hi = 0xcbd },
    .{ .lo = 0xcdd, .hi = 0xcde },
    .{ .lo = 0xce0, .hi = 0xce1 },
    .{ .lo = 0xcf1, .hi = 0xcf2 },
    .{ .lo = 0xd04, .hi = 0xd0c },
    .{ .lo = 0xd0e, .hi = 0xd10 },
    .{ .lo = 0xd12, .hi = 0xd3a },
    .{ .lo = 0xd3d, .hi = 0xd3d },
    .{ .lo = 0xd4e, .hi = 0xd4e },
    .{ .lo = 0xd54, .hi = 0xd56 },
    .{ .lo = 0xd5f, .hi = 0xd61 },
    .{ .lo = 0xd7a, .hi = 0xd7f },
    .{ .lo = 0xd85, .hi = 0xd96 },
    .{ .lo = 0xd9a, .hi = 0xdb1 },
    .{ .lo = 0xdb3, .hi = 0xdbb },
    .{ .lo = 0xdbd, .hi = 0xdbd },
    .{ .lo = 0xdc0, .hi = 0xdc6 },
    .{ .lo = 0xe01, .hi = 0xe30 },
    .{ .lo = 0xe32, .hi = 0xe33 },
    .{ .lo = 0xe40, .hi = 0xe46 },
    .{ .lo = 0xe81, .hi = 0xe82 },
    .{ .lo = 0xe84, .hi = 0xe84 },
    .{ .lo = 0xe86, .hi = 0xe8a },
    .{ .lo = 0xe8c, .hi = 0xea3 },
    .{ .lo = 0xea5, .hi = 0xea5 },
    .{ .lo = 0xea7, .hi = 0xeb0 },
    .{ .lo = 0xeb2, .hi = 0xeb3 },
    .{ .lo = 0xebd, .hi = 0xebd },
    .{ .lo = 0xec0, .hi = 0xec4 },
    .{ .lo = 0xec6, .hi = 0xec6 },
    .{ .lo = 0xedc, .hi = 0xedf },
    .{ .lo = 0xf00, .hi = 0xf00 },
    .{ .lo = 0xf40, .hi = 0xf47 },
    .{ .lo = 0xf49, .hi = 0xf6c },
    .{ .lo = 0xf88, .hi = 0xf8c },
    .{ .lo = 0x1000, .hi = 0x102a },
    .{ .lo = 0x103f, .hi = 0x103f },
    .{ .lo = 0x1050, .hi = 0x1055 },
    .{ .lo = 0x105a, .hi = 0x105d },
    .{ .lo = 0x1061, .hi = 0x1061 },
    .{ .lo = 0x1065, .hi = 0x1066 },
    .{ .lo = 0x106e, .hi = 0x1070 },
    .{ .lo = 0x1075, .hi = 0x1081 },
    .{ .lo = 0x108e, .hi = 0x108e },
    .{ .lo = 0x10a0, .hi = 0x10c5 },
    .{ .lo = 0x10c7, .hi = 0x10c7 },
    .{ .lo = 0x10cd, .hi = 0x10cd },
    .{ .lo = 0x10d0, .hi = 0x10fa },
    .{ .lo = 0x10fc, .hi = 0x1248 },
    .{ .lo = 0x124a, .hi = 0x124d },
    .{ .lo = 0x1250, .hi = 0x1256 },
    .{ .lo = 0x1258, .hi = 0x1258 },
    .{ .lo = 0x125a, .hi = 0x125d },
    .{ .lo = 0x1260, .hi = 0x1288 },
    .{ .lo = 0x128a, .hi = 0x128d },
    .{ .lo = 0x1290, .hi = 0x12b0 },
    .{ .lo = 0x12b2, .hi = 0x12b5 },
    .{ .lo = 0x12b8, .hi = 0x12be },
    .{ .lo = 0x12c0, .hi = 0x12c0 },
    .{ .lo = 0x12c2, .hi = 0x12c5 },
    .{ .lo = 0x12c8, .hi = 0x12d6 },
    .{ .lo = 0x12d8, .hi = 0x1310 },
    .{ .lo = 0x1312, .hi = 0x1315 },
    .{ .lo = 0x1318, .hi = 0x135a },
    .{ .lo = 0x1380, .hi = 0x138f },
    .{ .lo = 0x13a0, .hi = 0x13f5 },
    .{ .lo = 0x13f8, .hi = 0x13fd },
    .{ .lo = 0x1401, .hi = 0x166c },
    .{ .lo = 0x166f, .hi = 0x167f },
    .{ .lo = 0x1681, .hi = 0x169a },
    .{ .lo = 0x16a0, .hi = 0x16ea },
    .{ .lo = 0x16f1, .hi = 0x16f8 },
    .{ .lo = 0x1700, .hi = 0x1711 },
    .{ .lo = 0x171f, .hi = 0x1731 },
    .{ .lo = 0x1740, .hi = 0x1751 },
    .{ .lo = 0x1760, .hi = 0x176c },
    .{ .lo = 0x176e, .hi = 0x1770 },
    .{ .lo = 0x1780, .hi = 0x17b3 },
    .{ .lo = 0x17d7, .hi = 0x17d7 },
    .{ .lo = 0x17dc, .hi = 0x17dc },
    .{ .lo = 0x1820, .hi = 0x1878 },
    .{ .lo = 0x1880, .hi = 0x1884 },
    .{ .lo = 0x1887, .hi = 0x18a8 },
    .{ .lo = 0x18aa, .hi = 0x18aa },
    .{ .lo = 0x18b0, .hi = 0x18f5 },
    .{ .lo = 0x1900, .hi = 0x191e },
    .{ .lo = 0x1950, .hi = 0x196d },
    .{ .lo = 0x1970, .hi = 0x1974 },
    .{ .lo = 0x1980, .hi = 0x19ab },
    .{ .lo = 0x19b0, .hi = 0x19c9 },
    .{ .lo = 0x1a00, .hi = 0x1a16 },
    .{ .lo = 0x1a20, .hi = 0x1a54 },
    .{ .lo = 0x1aa7, .hi = 0x1aa7 },
    .{ .lo = 0x1b05, .hi = 0x1b33 },
    .{ .lo = 0x1b45, .hi = 0x1b4c },
    .{ .lo = 0x1b83, .hi = 0x1ba0 },
    .{ .lo = 0x1bae, .hi = 0x1baf },
    .{ .lo = 0x1bba, .hi = 0x1be5 },
    .{ .lo = 0x1c00, .hi = 0x1c23 },
    .{ .lo = 0x1c4d, .hi = 0x1c4f },
    .{ .lo = 0x1c5a, .hi = 0x1c7d },
    .{ .lo = 0x1c80, .hi = 0x1c88 },
    .{ .lo = 0x1c90, .hi = 0x1cba },
    .{ .lo = 0x1cbd, .hi = 0x1cbf },
    .{ .lo = 0x1ce9, .hi = 0x1cec },
    .{ .lo = 0x1cee, .hi = 0x1cf3 },
    .{ .lo = 0x1cf5, .hi = 0x1cf6 },
    .{ .lo = 0x1cfa, .hi = 0x1cfa },
    .{ .lo = 0x1d00, .hi = 0x1dbf },
    .{ .lo = 0x1e00, .hi = 0x1f15 },
    .{ .lo = 0x1f18, .hi = 0x1f1d },
    .{ .lo = 0x1f20, .hi = 0x1f45 },
    .{ .lo = 0x1f48, .hi = 0x1f4d },
    .{ .lo = 0x1f50, .hi = 0x1f57 },
    .{ .lo = 0x1f59, .hi = 0x1f59 },
    .{ .lo = 0x1f5b, .hi = 0x1f5b },
    .{ .lo = 0x1f5d, .hi = 0x1f5d },
    .{ .lo = 0x1f5f, .hi = 0x1f7d },
    .{ .lo = 0x1f80, .hi = 0x1fb4 },
    .{ .lo = 0x1fb6, .hi = 0x1fbc },
    .{ .lo = 0x1fbe, .hi = 0x1fbe },
    .{ .lo = 0x1fc2, .hi = 0x1fc4 },
    .{ .lo = 0x1fc6, .hi = 0x1fcc },
    .{ .lo = 0x1fd0, .hi = 0x1fd3 },
    .{ .lo = 0x1fd6, .hi = 0x1fdb },
    .{ .lo = 0x1fe0, .hi = 0x1fec },
    .{ .lo = 0x1ff2, .hi = 0x1ff4 },
    .{ .lo = 0x1ff6, .hi = 0x1ffc },
    .{ .lo = 0x2071, .hi = 0x2071 },
    .{ .lo = 0x207f, .hi = 0x207f },
    .{ .lo = 0x2090, .hi = 0x209c },
    .{ .lo = 0x2102, .hi = 0x2102 },
    .{ .lo = 0x2107, .hi = 0x2107 },
    .{ .lo = 0x210a, .hi = 0x2113 },
    .{ .lo = 0x2115, .hi = 0x2115 },
    .{ .lo = 0x2119, .hi = 0x211d },
    .{ .lo = 0x2124, .hi = 0x2124 },
    .{ .lo = 0x2126, .hi = 0x2126 },
    .{ .lo = 0x2128, .hi = 0x2128 },
    .{ .lo = 0x212a, .hi = 0x212d },
    .{ .lo = 0x212f, .hi = 0x2139 },
    .{ .lo = 0x213c, .hi = 0x213f },
    .{ .lo = 0x2145, .hi = 0x2149 },
    .{ .lo = 0x214e, .hi = 0x214e },
    .{ .lo = 0x2183, .hi = 0x2184 },
    .{ .lo = 0x2c00, .hi = 0x2ce4 },
    .{ .lo = 0x2ceb, .hi = 0x2cee },
    .{ .lo = 0x2cf2, .hi = 0x2cf3 },
    .{ .lo = 0x2d00, .hi = 0x2d25 },
    .{ .lo = 0x2d27, .hi = 0x2d27 },
    .{ .lo = 0x2d2d, .hi = 0x2d2d },
    .{ .lo = 0x2d30, .hi = 0x2d67 },
    .{ .lo = 0x2d6f, .hi = 0x2d6f },
    .{ .lo = 0x2d80, .hi = 0x2d96 },
    .{ .lo = 0x2da0, .hi = 0x2da6 },
    .{ .lo = 0x2da8, .hi = 0x2dae },
    .{ .lo = 0x2db0, .hi = 0x2db6 },
    .{ .lo = 0x2db8, .hi = 0x2dbe },
    .{ .lo = 0x2dc0, .hi = 0x2dc6 },
    .{ .lo = 0x2dc8, .hi = 0x2dce },
    .{ .lo = 0x2dd0, .hi = 0x2dd6 },
    .{ .lo = 0x2dd8, .hi = 0x2dde },
    .{ .lo = 0x2e2f, .hi = 0x2e2f },
    .{ .lo = 0x3005, .hi = 0x3006 },
    .{ .lo = 0x3031, .hi = 0x3035 },
    .{ .lo = 0x303b, .hi = 0x303c },
    .{ .lo = 0x3041, .hi = 0x3096 },
    .{ .lo = 0x309d, .hi = 0x309f },
    .{ .lo = 0x30a1, .hi = 0x30fa },
    .{ .lo = 0x30fc, .hi = 0x30ff },
    .{ .lo = 0x3105, .hi = 0x312f },
    .{ .lo = 0x3131, .hi = 0x318e },
    .{ .lo = 0x31a0, .hi = 0x31bf },
    .{ .lo = 0x31f0, .hi = 0x31ff },
    .{ .lo = 0x3400, .hi = 0x4dbf },
    .{ .lo = 0x4e00, .hi = 0xa48c },
    .{ .lo = 0xa4d0, .hi = 0xa4fd },
    .{ .lo = 0xa500, .hi = 0xa60c },
    .{ .lo = 0xa610, .hi = 0xa61f },
    .{ .lo = 0xa62a, .hi = 0xa62b },
    .{ .lo = 0xa640, .hi = 0xa66e },
    .{ .lo = 0xa67f, .hi = 0xa69d },
    .{ .lo = 0xa6a0, .hi = 0xa6e5 },
    .{ .lo = 0xa717, .hi = 0xa71f },
    .{ .lo = 0xa722, .hi = 0xa788 },
    .{ .lo = 0xa78b, .hi = 0xa7ca },
    .{ .lo = 0xa7d0, .hi = 0xa7d1 },
    .{ .lo = 0xa7d3, .hi = 0xa7d3 },
    .{ .lo = 0xa7d5, .hi = 0xa7d9 },
    .{ .lo = 0xa7f2, .hi = 0xa801 },
    .{ .lo = 0xa803, .hi = 0xa805 },
    .{ .lo = 0xa807, .hi = 0xa80a },
    .{ .lo = 0xa80c, .hi = 0xa822 },
    .{ .lo = 0xa840, .hi = 0xa873 },
    .{ .lo = 0xa882, .hi = 0xa8b3 },
    .{ .lo = 0xa8f2, .hi = 0xa8f7 },
    .{ .lo = 0xa8fb, .hi = 0xa8fb },
    .{ .lo = 0xa8fd, .hi = 0xa8fe },
    .{ .lo = 0xa90a, .hi = 0xa925 },
    .{ .lo = 0xa930, .hi = 0xa946 },
    .{ .lo = 0xa960, .hi = 0xa97c },
    .{ .lo = 0xa984, .hi = 0xa9b2 },
    .{ .lo = 0xa9cf, .hi = 0xa9cf },
    .{ .lo = 0xa9e0, .hi = 0xa9e4 },
    .{ .lo = 0xa9e6, .hi = 0xa9ef },
    .{ .lo = 0xa9fa, .hi = 0xa9fe },
    .{ .lo = 0xaa00, .hi = 0xaa28 },
    .{ .lo = 0xaa40, .hi = 0xaa42 },
    .{ .lo = 0xaa44, .hi = 0xaa4b },
    .{ .lo = 0xaa60, .hi = 0xaa76 },
    .{ .lo = 0xaa7a, .hi = 0xaa7a },
    .{ .lo = 0xaa7e, .hi = 0xaaaf },
    .{ .lo = 0xaab1, .hi = 0xaab1 },
    .{ .lo = 0xaab5, .hi = 0xaab6 },
    .{ .lo = 0xaab9, .hi = 0xaabd },
    .{ .lo = 0xaac0, .hi = 0xaac0 },
    .{ .lo = 0xaac2, .hi = 0xaac2 },
    .{ .lo = 0xaadb, .hi = 0xaadd },
    .{ .lo = 0xaae0, .hi = 0xaaea },
    .{ .lo = 0xaaf2, .hi = 0xaaf4 },
    .{ .lo = 0xab01, .hi = 0xab06 },
    .{ .lo = 0xab09, .hi = 0xab0e },
    .{ .lo = 0xab11, .hi = 0xab16 },
    .{ .lo = 0xab20, .hi = 0xab26 },
    .{ .lo = 0xab28, .hi = 0xab2e },
    .{ .lo = 0xab30, .hi = 0xab5a },
    .{ .lo = 0xab5c, .hi = 0xab69 },
    .{ .lo = 0xab70, .hi = 0xabe2 },
    .{ .lo = 0xac00, .hi = 0xd7a3 },
    .{ .lo = 0xd7b0, .hi = 0xd7c6 },
    .{ .lo = 0xd7cb, .hi = 0xd7fb },
    .{ .lo = 0xf900, .hi = 0xfa6d },
    .{ .lo = 0xfa70, .hi = 0xfad9 },
    .{ .lo = 0xfb00, .hi = 0xfb06 },
    .{ .lo = 0xfb13, .hi = 0xfb17 },
    .{ .lo = 0xfb1d, .hi = 0xfb1d },
    .{ .lo = 0xfb1f, .hi = 0xfb28 },
    .{ .lo = 0xfb2a, .hi = 0xfb36 },
    .{ .lo = 0xfb38, .hi = 0xfb3c },
    .{ .lo = 0xfb3e, .hi = 0xfb3e },
    .{ .lo = 0xfb40, .hi = 0xfb41 },
    .{ .lo = 0xfb43, .hi = 0xfb44 },
    .{ .lo = 0xfb46, .hi = 0xfbb1 },
    .{ .lo = 0xfbd3, .hi = 0xfd3d },
    .{ .lo = 0xfd50, .hi = 0xfd8f },
    .{ .lo = 0xfd92, .hi = 0xfdc7 },
    .{ .lo = 0xfdf0, .hi = 0xfdfb },
    .{ .lo = 0xfe70, .hi = 0xfe74 },
    .{ .lo = 0xfe76, .hi = 0xfefc },
    .{ .lo = 0xff21, .hi = 0xff3a },
    .{ .lo = 0xff41, .hi = 0xff5a },
    .{ .lo = 0xff66, .hi = 0xffbe },
    .{ .lo = 0xffc2, .hi = 0xffc7 },
    .{ .lo = 0xffca, .hi = 0xffcf },
    .{ .lo = 0xffd2, .hi = 0xffd7 },
    .{ .lo = 0xffda, .hi = 0xffdc },
    .{ .lo = 0x10000, .hi = 0x1000b },
    .{ .lo = 0x1000d, .hi = 0x10026 },
    .{ .lo = 0x10028, .hi = 0x1003a },
    .{ .lo = 0x1003c, .hi = 0x1003d },
    .{ .lo = 0x1003f, .hi = 0x1004d },
    .{ .lo = 0x10050, .hi = 0x1005d },
    .{ .lo = 0x10080, .hi = 0x100fa },
    .{ .lo = 0x10280, .hi = 0x1029c },
    .{ .lo = 0x102a0, .hi = 0x102d0 },
    .{ .lo = 0x10300, .hi = 0x1031f },
    .{ .lo = 0x1032d, .hi = 0x10340 },
    .{ .lo = 0x10342, .hi = 0x10349 },
    .{ .lo = 0x10350, .hi = 0x10375 },
    .{ .lo = 0x10380, .hi = 0x1039d },
    .{ .lo = 0x103a0, .hi = 0x103c3 },
    .{ .lo = 0x103c8, .hi = 0x103cf },
    .{ .lo = 0x10400, .hi = 0x1049d },
    .{ .lo = 0x104b0, .hi = 0x104d3 },
    .{ .lo = 0x104d8, .hi = 0x104fb },
    .{ .lo = 0x10500, .hi = 0x10527 },
    .{ .lo = 0x10530, .hi = 0x10563 },
    .{ .lo = 0x10570, .hi = 0x1057a },
    .{ .lo = 0x1057c, .hi = 0x1058a },
    .{ .lo = 0x1058c, .hi = 0x10592 },
    .{ .lo = 0x10594, .hi = 0x10595 },
    .{ .lo = 0x10597, .hi = 0x105a1 },
    .{ .lo = 0x105a3, .hi = 0x105b1 },
    .{ .lo = 0x105b3, .hi = 0x105b9 },
    .{ .lo = 0x105bb, .hi = 0x105bc },
    .{ .lo = 0x10600, .hi = 0x10736 },
    .{ .lo = 0x10740, .hi = 0x10755 },
    .{ .lo = 0x10760, .hi = 0x10767 },
    .{ .lo = 0x10780, .hi = 0x10785 },
    .{ .lo = 0x10787, .hi = 0x107b0 },
    .{ .lo = 0x107b2, .hi = 0x107ba },
    .{ .lo = 0x10800, .hi = 0x10805 },
    .{ .lo = 0x10808, .hi = 0x10808 },
    .{ .lo = 0x1080a, .hi = 0x10835 },
    .{ .lo = 0x10837, .hi = 0x10838 },
    .{ .lo = 0x1083c, .hi = 0x1083c },
    .{ .lo = 0x1083f, .hi = 0x10855 },
    .{ .lo = 0x10860, .hi = 0x10876 },
    .{ .lo = 0x10880, .hi = 0x1089e },
    .{ .lo = 0x108e0, .hi = 0x108f2 },
    .{ .lo = 0x108f4, .hi = 0x108f5 },
    .{ .lo = 0x10900, .hi = 0x10915 },
    .{ .lo = 0x10920, .hi = 0x10939 },
    .{ .lo = 0x10980, .hi = 0x109b7 },
    .{ .lo = 0x109be, .hi = 0x109bf },
    .{ .lo = 0x10a00, .hi = 0x10a00 },
    .{ .lo = 0x10a10, .hi = 0x10a13 },
    .{ .lo = 0x10a15, .hi = 0x10a17 },
    .{ .lo = 0x10a19, .hi = 0x10a35 },
    .{ .lo = 0x10a60, .hi = 0x10a7c },
    .{ .lo = 0x10a80, .hi = 0x10a9c },
    .{ .lo = 0x10ac0, .hi = 0x10ac7 },
    .{ .lo = 0x10ac9, .hi = 0x10ae4 },
    .{ .lo = 0x10b00, .hi = 0x10b35 },
    .{ .lo = 0x10b40, .hi = 0x10b55 },
    .{ .lo = 0x10b60, .hi = 0x10b72 },
    .{ .lo = 0x10b80, .hi = 0x10b91 },
    .{ .lo = 0x10c00, .hi = 0x10c48 },
    .{ .lo = 0x10c80, .hi = 0x10cb2 },
    .{ .lo = 0x10cc0, .hi = 0x10cf2 },
    .{ .lo = 0x10d00, .hi = 0x10d23 },
    .{ .lo = 0x10e80, .hi = 0x10ea9 },
    .{ .lo = 0x10eb0, .hi = 0x10eb1 },
    .{ .lo = 0x10f00, .hi = 0x10f1c },
    .{ .lo = 0x10f27, .hi = 0x10f27 },
    .{ .lo = 0x10f30, .hi = 0x10f45 },
    .{ .lo = 0x10f70, .hi = 0x10f81 },
    .{ .lo = 0x10fb0, .hi = 0x10fc4 },
    .{ .lo = 0x10fe0, .hi = 0x10ff6 },
    .{ .lo = 0x11003, .hi = 0x11037 },
    .{ .lo = 0x11071, .hi = 0x11072 },
    .{ .lo = 0x11075, .hi = 0x11075 },
    .{ .lo = 0x11083, .hi = 0x110af },
    .{ .lo = 0x110d0, .hi = 0x110e8 },
    .{ .lo = 0x11103, .hi = 0x11126 },
    .{ .lo = 0x11144, .hi = 0x11144 },
    .{ .lo = 0x11147, .hi = 0x11147 },
    .{ .lo = 0x11150, .hi = 0x11172 },
    .{ .lo = 0x11176, .hi = 0x11176 },
    .{ .lo = 0x11183, .hi = 0x111b2 },
    .{ .lo = 0x111c1, .hi = 0x111c4 },
    .{ .lo = 0x111da, .hi = 0x111da },
    .{ .lo = 0x111dc, .hi = 0x111dc },
    .{ .lo = 0x11200, .hi = 0x11211 },
    .{ .lo = 0x11213, .hi = 0x1122b },
    .{ .lo = 0x11280, .hi = 0x11286 },
    .{ .lo = 0x11288, .hi = 0x11288 },
    .{ .lo = 0x1128a, .hi = 0x1128d },
    .{ .lo = 0x1128f, .hi = 0x1129d },
    .{ .lo = 0x1129f, .hi = 0x112a8 },
    .{ .lo = 0x112b0, .hi = 0x112de },
    .{ .lo = 0x11305, .hi = 0x1130c },
    .{ .lo = 0x1130f, .hi = 0x11310 },
    .{ .lo = 0x11313, .hi = 0x11328 },
    .{ .lo = 0x1132a, .hi = 0x11330 },
    .{ .lo = 0x11332, .hi = 0x11333 },
    .{ .lo = 0x11335, .hi = 0x11339 },
    .{ .lo = 0x1133d, .hi = 0x1133d },
    .{ .lo = 0x11350, .hi = 0x11350 },
    .{ .lo = 0x1135d, .hi = 0x11361 },
    .{ .lo = 0x11400, .hi = 0x11434 },
    .{ .lo = 0x11447, .hi = 0x1144a },
    .{ .lo = 0x1145f, .hi = 0x11461 },
    .{ .lo = 0x11480, .hi = 0x114af },
    .{ .lo = 0x114c4, .hi = 0x114c5 },
    .{ .lo = 0x114c7, .hi = 0x114c7 },
    .{ .lo = 0x11580, .hi = 0x115ae },
    .{ .lo = 0x115d8, .hi = 0x115db },
    .{ .lo = 0x11600, .hi = 0x1162f },
    .{ .lo = 0x11644, .hi = 0x11644 },
    .{ .lo = 0x11680, .hi = 0x116aa },
    .{ .lo = 0x116b8, .hi = 0x116b8 },
    .{ .lo = 0x11700, .hi = 0x1171a },
    .{ .lo = 0x11740, .hi = 0x11746 },
    .{ .lo = 0x11800, .hi = 0x1182b },
    .{ .lo = 0x118a0, .hi = 0x118df },
    .{ .lo = 0x118ff, .hi = 0x11906 },
    .{ .lo = 0x11909, .hi = 0x11909 },
    .{ .lo = 0x1190c, .hi = 0x11913 },
    .{ .lo = 0x11915, .hi = 0x11916 },
    .{ .lo = 0x11918, .hi = 0x1192f },
    .{ .lo = 0x1193f, .hi = 0x1193f },
    .{ .lo = 0x11941, .hi = 0x11941 },
    .{ .lo = 0x119a0, .hi = 0x119a7 },
    .{ .lo = 0x119aa, .hi = 0x119d0 },
    .{ .lo = 0x119e1, .hi = 0x119e1 },
    .{ .lo = 0x119e3, .hi = 0x119e3 },
    .{ .lo = 0x11a00, .hi = 0x11a00 },
    .{ .lo = 0x11a0b, .hi = 0x11a32 },
    .{ .lo = 0x11a3a, .hi = 0x11a3a },
    .{ .lo = 0x11a50, .hi = 0x11a50 },
    .{ .lo = 0x11a5c, .hi = 0x11a89 },
    .{ .lo = 0x11a9d, .hi = 0x11a9d },
    .{ .lo = 0x11ab0, .hi = 0x11af8 },
    .{ .lo = 0x11c00, .hi = 0x11c08 },
    .{ .lo = 0x11c0a, .hi = 0x11c2e },
    .{ .lo = 0x11c40, .hi = 0x11c40 },
    .{ .lo = 0x11c72, .hi = 0x11c8f },
    .{ .lo = 0x11d00, .hi = 0x11d06 },
    .{ .lo = 0x11d08, .hi = 0x11d09 },
    .{ .lo = 0x11d0b, .hi = 0x11d30 },
    .{ .lo = 0x11d46, .hi = 0x11d46 },
    .{ .lo = 0x11d60, .hi = 0x11d65 },
    .{ .lo = 0x11d67, .hi = 0x11d68 },
    .{ .lo = 0x11d6a, .hi = 0x11d89 },
    .{ .lo = 0x11d98, .hi = 0x11d98 },
    .{ .lo = 0x11ee0, .hi = 0x11ef2 },
    .{ .lo = 0x11fb0, .hi = 0x11fb0 },
    .{ .lo = 0x12000, .hi = 0x12399 },
    .{ .lo = 0x12480, .hi = 0x12543 },
    .{ .lo = 0x12f90, .hi = 0x12ff0 },
    .{ .lo = 0x13000, .hi = 0x1342e },
    .{ .lo = 0x14400, .hi = 0x14646 },
    .{ .lo = 0x16800, .hi = 0x16a38 },
    .{ .lo = 0x16a40, .hi = 0x16a5e },
    .{ .lo = 0x16a70, .hi = 0x16abe },
    .{ .lo = 0x16ad0, .hi = 0x16aed },
    .{ .lo = 0x16b00, .hi = 0x16b2f },
    .{ .lo = 0x16b40, .hi = 0x16b43 },
    .{ .lo = 0x16b63, .hi = 0x16b77 },
    .{ .lo = 0x16b7d, .hi = 0x16b8f },
    .{ .lo = 0x16e40, .hi = 0x16e7f },
    .{ .lo = 0x16f00, .hi = 0x16f4a },
    .{ .lo = 0x16f50, .hi = 0x16f50 },
    .{ .lo = 0x16f93, .hi = 0x16f9f },
    .{ .lo = 0x16fe0, .hi = 0x16fe1 },
    .{ .lo = 0x16fe3, .hi = 0x16fe3 },
    .{ .lo = 0x17000, .hi = 0x187f7 },
    .{ .lo = 0x18800, .hi = 0x18cd5 },
    .{ .lo = 0x18d00, .hi = 0x18d08 },
    .{ .lo = 0x1aff0, .hi = 0x1aff3 },
    .{ .lo = 0x1aff5, .hi = 0x1affb },
    .{ .lo = 0x1affd, .hi = 0x1affe },
    .{ .lo = 0x1b000, .hi = 0x1b122 },
    .{ .lo = 0x1b150, .hi = 0x1b152 },
    .{ .lo = 0x1b164, .hi = 0x1b167 },
    .{ .lo = 0x1b170, .hi = 0x1b2fb },
    .{ .lo = 0x1bc00, .hi = 0x1bc6a },
    .{ .lo = 0x1bc70, .hi = 0x1bc7c },
    .{ .lo = 0x1bc80, .hi = 0x1bc88 },
    .{ .lo = 0x1bc90, .hi = 0x1bc99 },
    .{ .lo = 0x1d400, .hi = 0x1d454 },
    .{ .lo = 0x1d456, .hi = 0x1d49c },
    .{ .lo = 0x1d49e, .hi = 0x1d49f },
    .{ .lo = 0x1d4a2, .hi = 0x1d4a2 },
    .{ .lo = 0x1d4a5, .hi = 0x1d4a6 },
    .{ .lo = 0x1d4a9, .hi = 0x1d4ac },
    .{ .lo = 0x1d4ae, .hi = 0x1d4b9 },
    .{ .lo = 0x1d4bb, .hi = 0x1d4bb },
    .{ .lo = 0x1d4bd, .hi = 0x1d4c3 },
    .{ .lo = 0x1d4c5, .hi = 0x1d505 },
    .{ .lo = 0x1d507, .hi = 0x1d50a },
    .{ .lo = 0x1d50d, .hi = 0x1d514 },
    .{ .lo = 0x1d516, .hi = 0x1d51c },
    .{ .lo = 0x1d51e, .hi = 0x1d539 },
    .{ .lo = 0x1d53b, .hi = 0x1d53e },
    .{ .lo = 0x1d540, .hi = 0x1d544 },
    .{ .lo = 0x1d546, .hi = 0x1d546 },
    .{ .lo = 0x1d54a, .hi = 0x1d550 },
    .{ .lo = 0x1d552, .hi = 0x1d6a5 },
    .{ .lo = 0x1d6a8, .hi = 0x1d6c0 },
    .{ .lo = 0x1d6c2, .hi = 0x1d6da },
    .{ .lo = 0x1d6dc, .hi = 0x1d6fa },
    .{ .lo = 0x1d6fc, .hi = 0x1d714 },
    .{ .lo = 0x1d716, .hi = 0x1d734 },
    .{ .lo = 0x1d736, .hi = 0x1d74e },
    .{ .lo = 0x1d750, .hi = 0x1d76e },
    .{ .lo = 0x1d770, .hi = 0x1d788 },
    .{ .lo = 0x1d78a, .hi = 0x1d7a8 },
    .{ .lo = 0x1d7aa, .hi = 0x1d7c2 },
    .{ .lo = 0x1d7c4, .hi = 0x1d7cb },
    .{ .lo = 0x1df00, .hi = 0x1df1e },
    .{ .lo = 0x1e100, .hi = 0x1e12c },
    .{ .lo = 0x1e137, .hi = 0x1e13d },
    .{ .lo = 0x1e14e, .hi = 0x1e14e },
    .{ .lo = 0x1e290, .hi = 0x1e2ad },
    .{ .lo = 0x1e2c0, .hi = 0x1e2eb },
    .{ .lo = 0x1e7e0, .hi = 0x1e7e6 },
    .{ .lo = 0x1e7e8, .hi = 0x1e7eb },
    .{ .lo = 0x1e7ed, .hi = 0x1e7ee },
    .{ .lo = 0x1e7f0, .hi = 0x1e7fe },
    .{ .lo = 0x1e800, .hi = 0x1e8c4 },
    .{ .lo = 0x1e900, .hi = 0x1e943 },
    .{ .lo = 0x1e94b, .hi = 0x1e94b },
    .{ .lo = 0x1ee00, .hi = 0x1ee03 },
    .{ .lo = 0x1ee05, .hi = 0x1ee1f },
    .{ .lo = 0x1ee21, .hi = 0x1ee22 },
    .{ .lo = 0x1ee24, .hi = 0x1ee24 },
    .{ .lo = 0x1ee27, .hi = 0x1ee27 },
    .{ .lo = 0x1ee29, .hi = 0x1ee32 },
    .{ .lo = 0x1ee34, .hi = 0x1ee37 },
    .{ .lo = 0x1ee39, .hi = 0x1ee39 },
    .{ .lo = 0x1ee3b, .hi = 0x1ee3b },
    .{ .lo = 0x1ee42, .hi = 0x1ee42 },
    .{ .lo = 0x1ee47, .hi = 0x1ee47 },
    .{ .lo = 0x1ee49, .hi = 0x1ee49 },
    .{ .lo = 0x1ee4b, .hi = 0x1ee4b },
    .{ .lo = 0x1ee4d, .hi = 0x1ee4f },
    .{ .lo = 0x1ee51, .hi = 0x1ee52 },
    .{ .lo = 0x1ee54, .hi = 0x1ee54 },
    .{ .lo = 0x1ee57, .hi = 0x1ee57 },
    .{ .lo = 0x1ee59, .hi = 0x1ee59 },
    .{ .lo = 0x1ee5b, .hi = 0x1ee5b },
    .{ .lo = 0x1ee5d, .hi = 0x1ee5d },
    .{ .lo = 0x1ee5f, .hi = 0x1ee5f },
    .{ .lo = 0x1ee61, .hi = 0x1ee62 },
    .{ .lo = 0x1ee64, .hi = 0x1ee64 },
    .{ .lo = 0x1ee67, .hi = 0x1ee6a },
    .{ .lo = 0x1ee6c, .hi = 0x1ee72 },
    .{ .lo = 0x1ee74, .hi = 0x1ee77 },
    .{ .lo = 0x1ee79, .hi = 0x1ee7c },
    .{ .lo = 0x1ee7e, .hi = 0x1ee7e },
    .{ .lo = 0x1ee80, .hi = 0x1ee89 },
    .{ .lo = 0x1ee8b, .hi = 0x1ee9b },
    .{ .lo = 0x1eea1, .hi = 0x1eea3 },
    .{ .lo = 0x1eea5, .hi = 0x1eea9 },
    .{ .lo = 0x1eeab, .hi = 0x1eebb },
    .{ .lo = 0x20000, .hi = 0x2a6df },
    .{ .lo = 0x2a700, .hi = 0x2b738 },
    .{ .lo = 0x2b740, .hi = 0x2b81d },
    .{ .lo = 0x2b820, .hi = 0x2cea1 },
    .{ .lo = 0x2ceb0, .hi = 0x2ebe0 },
    .{ .lo = 0x2f800, .hi = 0x2fa1d },
    .{ .lo = 0x30000, .hi = 0x3134a },
};
const digit_ranges = [_]Range{
    .{ .lo = 0x30, .hi = 0x39 },
    .{ .lo = 0xb2, .hi = 0xb3 },
    .{ .lo = 0xb9, .hi = 0xb9 },
    .{ .lo = 0x660, .hi = 0x669 },
    .{ .lo = 0x6f0, .hi = 0x6f9 },
    .{ .lo = 0x7c0, .hi = 0x7c9 },
    .{ .lo = 0x966, .hi = 0x96f },
    .{ .lo = 0x9e6, .hi = 0x9ef },
    .{ .lo = 0xa66, .hi = 0xa6f },
    .{ .lo = 0xae6, .hi = 0xaef },
    .{ .lo = 0xb66, .hi = 0xb6f },
    .{ .lo = 0xbe6, .hi = 0xbef },
    .{ .lo = 0xc66, .hi = 0xc6f },
    .{ .lo = 0xce6, .hi = 0xcef },
    .{ .lo = 0xd66, .hi = 0xd6f },
    .{ .lo = 0xde6, .hi = 0xdef },
    .{ .lo = 0xe50, .hi = 0xe59 },
    .{ .lo = 0xed0, .hi = 0xed9 },
    .{ .lo = 0xf20, .hi = 0xf29 },
    .{ .lo = 0x1040, .hi = 0x1049 },
    .{ .lo = 0x1090, .hi = 0x1099 },
    .{ .lo = 0x1369, .hi = 0x1371 },
    .{ .lo = 0x17e0, .hi = 0x17e9 },
    .{ .lo = 0x1810, .hi = 0x1819 },
    .{ .lo = 0x1946, .hi = 0x194f },
    .{ .lo = 0x19d0, .hi = 0x19da },
    .{ .lo = 0x1a80, .hi = 0x1a89 },
    .{ .lo = 0x1a90, .hi = 0x1a99 },
    .{ .lo = 0x1b50, .hi = 0x1b59 },
    .{ .lo = 0x1bb0, .hi = 0x1bb9 },
    .{ .lo = 0x1c40, .hi = 0x1c49 },
    .{ .lo = 0x1c50, .hi = 0x1c59 },
    .{ .lo = 0x2070, .hi = 0x2070 },
    .{ .lo = 0x2074, .hi = 0x2079 },
    .{ .lo = 0x2080, .hi = 0x2089 },
    .{ .lo = 0x2460, .hi = 0x2468 },
    .{ .lo = 0x2474, .hi = 0x247c },
    .{ .lo = 0x2488, .hi = 0x2490 },
    .{ .lo = 0x24ea, .hi = 0x24ea },
    .{ .lo = 0x24f5, .hi = 0x24fd },
    .{ .lo = 0x24ff, .hi = 0x24ff },
    .{ .lo = 0x2776, .hi = 0x277e },
    .{ .lo = 0x2780, .hi = 0x2788 },
    .{ .lo = 0x278a, .hi = 0x2792 },
    .{ .lo = 0xa620, .hi = 0xa629 },
    .{ .lo = 0xa8d0, .hi = 0xa8d9 },
    .{ .lo = 0xa900, .hi = 0xa909 },
    .{ .lo = 0xa9d0, .hi = 0xa9d9 },
    .{ .lo = 0xa9f0, .hi = 0xa9f9 },
    .{ .lo = 0xaa50, .hi = 0xaa59 },
    .{ .lo = 0xabf0, .hi = 0xabf9 },
    .{ .lo = 0xff10, .hi = 0xff19 },
    .{ .lo = 0x104a0, .hi = 0x104a9 },
    .{ .lo = 0x10a40, .hi = 0x10a43 },
    .{ .lo = 0x10d30, .hi = 0x10d39 },
    .{ .lo = 0x10e60, .hi = 0x10e68 },
    .{ .lo = 0x11052, .hi = 0x1105a },
    .{ .lo = 0x11066, .hi = 0x1106f },
    .{ .lo = 0x110f0, .hi = 0x110f9 },
    .{ .lo = 0x11136, .hi = 0x1113f },
    .{ .lo = 0x111d0, .hi = 0x111d9 },
    .{ .lo = 0x112f0, .hi = 0x112f9 },
    .{ .lo = 0x11450, .hi = 0x11459 },
    .{ .lo = 0x114d0, .hi = 0x114d9 },
    .{ .lo = 0x11650, .hi = 0x11659 },
    .{ .lo = 0x116c0, .hi = 0x116c9 },
    .{ .lo = 0x11730, .hi = 0x11739 },
    .{ .lo = 0x118e0, .hi = 0x118e9 },
    .{ .lo = 0x11950, .hi = 0x11959 },
    .{ .lo = 0x11c50, .hi = 0x11c59 },
    .{ .lo = 0x11d50, .hi = 0x11d59 },
    .{ .lo = 0x11da0, .hi = 0x11da9 },
    .{ .lo = 0x16a60, .hi = 0x16a69 },
    .{ .lo = 0x16ac0, .hi = 0x16ac9 },
    .{ .lo = 0x16b50, .hi = 0x16b59 },
    .{ .lo = 0x1d7ce, .hi = 0x1d7ff },
    .{ .lo = 0x1e140, .hi = 0x1e149 },
    .{ .lo = 0x1e2f0, .hi = 0x1e2f9 },
    .{ .lo = 0x1e950, .hi = 0x1e959 },
    .{ .lo = 0x1f100, .hi = 0x1f10a },
    .{ .lo = 0x1fbf0, .hi = 0x1fbf9 },
};

const testing = std.testing;

fn testArena() std.heap.ArenaAllocator {
    return .init(testing.allocator);
}

test "parseJson keeps CPython json numeric and string semantics without repair" {
    var memory = testArena();
    defer memory.deinit();
    const arena = memory.allocator();
    try testing.expect(std.math.isNan((try parseJson(arena, "NaN")).float));
    try testing.expect(std.math.isNegativeInf((try parseJson(arena, "-Infinity")).float));
    try testing.expect(std.math.isPositiveInf((try parseJson(arena, "1e400")).float));
    try testing.expectEqualStrings("18446744073709551616", (try parseJson(arena, "18446744073709551616")).number_string);
    try testing.expectEqual(@as(i64, std.math.minInt(i64)), (try parseJson(arena, "-9223372036854775808")).integer);
    try testing.expectEqualStrings("\xed\xa0\x80", (try parseJson(arena, "\"\\ud800\"")).string);
    try testing.expectEqualStrings("\xf0\x9f\x98\x80", (try parseJson(arena, "\"\\ud83d\\ude00\"")).string);
    try testing.expectError(error.ParseValue, parseJson(arena, "\"\\ud83d\\uZZZZ\""));
    try testing.expectError(error.ParseValue, parseJson(arena, "{\"a\": 1"));
    try testing.expectError(error.ParseValue, parseJson(arena, "{\"a\": 1} x"));
    try testing.expectError(error.ParseValue, parseJson(arena, "nan"));
    const ones: [4301]u8 = @splat('1');
    try testing.expectError(error.IntegerTooLong, parseJson(arena, &ones));
    try testing.expectEqual(@as(usize, 4300), (try parseJson(arena, ones[0..4300])).number_string.len);
    const object = try parseJson(arena, "{\"a\": 1, \"b\": 2, \"a\": 3}");
    try testing.expectEqualStrings("a", object.object.keys()[0]);
    try testing.expectEqual(@as(i64, 3), object.object.get("a").?.integer);
}

test "loads repairs incomplete objects while keeping exact values" {
    var memory = testArena();
    defer memory.deinit();
    const arena = memory.allocator();
    const repaired = try loads(arena, "{\"query\": \"Paris\", \"limit\": 123456789012345678901234567890");
    try testing.expectEqualStrings("Paris", repaired.object.get("query").?.string);
    try testing.expectEqualStrings("123456789012345678901234567890", repaired.object.get("limit").?.number_string);
    const unicode_value = try loads(arena, "{'é': '雪😀'");
    try testing.expectEqualStrings("雪😀", unicode_value.object.get("é").?.string);
}

test "loads oversized integers degrade the same way as the source parser" {
    var memory = testArena();
    defer memory.deinit();
    const arena = memory.allocator();
    // json.loads raises ValueError, which repair_json catches; parse_number's
    // int() also raises, so the digits survive as a string.
    const ones: [4301]u8 = @splat('1');
    const text = try std.mem.concat(arena, u8, &.{ "{\"a\": ", &ones, ", \"b\": 1" });
    const repaired = try loads(arena, text);
    try testing.expectEqual(@as(usize, 4301), repaired.object.get("a").?.string.len);
}

test "loads bounds nesting with an explicit error" {
    var memory = testArena();
    defer memory.deinit();
    const arena = memory.allocator();
    const opening: [max_depth + 8]u8 = @splat('[');
    const closing: [max_depth + 8]u8 = @splat(']');
    try testing.expectError(error.JsonTooDeep, loads(arena, &opening));
    const nested = try std.mem.concat(arena, u8, &.{ &opening, &closing });
    try testing.expectError(error.JsonTooDeep, parseJson(arena, nested));
}
