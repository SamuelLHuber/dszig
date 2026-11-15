//! Field values: DSPy's `format_field_value` / `parse_value`, and the bridge
//! between typed Zig structs and native runtime values, including genuine
//! boxed model and builtin identities.
//!
//! Records (inputs, outputs, demos, examples) preserve insertion order exactly
//! like Python dicts do.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const ObjectMap = @import("runtime_value.zig").ObjectMap;
const Writer = std.Io.Writer;

const pyjson = @import("pyjson.zig");
const signature = @import("signature.zig");
const types = @import("types.zig");
const code = @import("code.zig");
const citation = @import("citation.zig");
const decision = @import("decision.zig");
const Field = signature.Field;

pub const Record = ObjectMap;

pub const ParseError = error{ParseValue} || Allocator.Error;

/// ASCII Python whitespace for byte-oriented token grammars.
pub const py_whitespace = " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f";

pub fn pyStrip(s: []const u8) []const u8 {
    return @import("unicode.zig").strip(s, false);
}

/// `format_field_value(field_info, value)`.
pub fn formatFieldValue(w: *Writer, field: Field, v: Value) pyjson.Error!void {
    if (v == .custom and v.custom.vtable.decision != null)
        return v.custom.writeSerializedJson(w, .compact, 0);
    if (v == .array and field.kind == .str and !field.reasoning_type and
        !field.code_type and !field.optional_str) return formatListBlobs(w, v.array.items);
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
    if (field.parse_value) |parse| return parse(arena, .{ .string = text });
    if (field.kind == .str and !field.reasoning_type and !field.code_type)
        return .{ .string = text };
    if (field.optional_str) return .{ .string = text };
    var source = text;
    if (field.kind == .literal) {
        if (literalMatch(field, source)) return .{ .string = source };
        source = literalText(source);
        if (literalMatch(field, source)) return .{ .string = source };
    }
    if (field.kind == .int) {
        const token = pyStrip(source);
        if (isJsonNumber(token) and std.json.isNumberFormattedLikeAnInteger(token)) {
            if (std.fmt.parseInt(i64, token, 10)) |integer|
                return .{ .integer = integer }
            else |_| {}
        }
    }
    const candidate = try parseCandidate(arena, source);
    return validateCandidate(arena, field, candidate) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.ParseValue => if (field.dspy_type or field.code_type or
            (field.reasoning_type and field.kind == .str))
            validateCandidate(arena, field, .{ .string = source })
        else
            error.ParseValue,
    };
}

/// `parse_value(value, annotation)` preserves the original Python type until
/// the annotation's validator chooses a conversion.
pub fn parseJsonValue(arena: Allocator, field: Field, v: Value) ParseError!Value {
    if (field.parse_value) |parse| return parse(arena, v);
    if (field.kind == .str and !field.reasoning_type and !field.code_type) {
        if (v == .string) return v;
        var out: Writer.Allocating = .init(arena);
        pyjson.writeStr(&out.writer, v) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep, error.ParseValue => error.ParseValue,
        };
        return .{ .string = out.written() };
    }
    if (field.kind == .literal) {
        for (field.literals) |allowed| if (pyEqual(v, .{ .string = allowed })) return v;
    }
    if (v == .string) return parseText(arena, field, v.string);
    return validateCandidate(arena, field, v);
}

fn validateCandidate(arena: Allocator, field: Field, v: Value) ParseError!Value {
    if (field.code_type) return (field.validate orelse return error.ParseValue)(arena, v);
    if (field.reasoning_type and field.kind == .str) return types.validate(signature.Reasoning, arena, v);
    return switch (field.kind) {
        .str => unreachable,
        .literal => if (v == .string and literalMatch(field, v.string)) v else error.ParseValue,
        .boolean => types.validate(bool, arena, v),
        .int => coerceInteger(arena, v),
        .float => types.validate(f64, arena, v),
        .str_list => types.validate([]const []const u8, arena, v),
        .json_object => types.validate(ObjectMap, arena, v),
        .str_dict => types.validate(signature.Dict([]const u8), arena, v),
        .json => (field.validate orelse return error.ParseValue)(arena, v),
    };
}

fn parseCandidate(arena: Allocator, source: []const u8) ParseError!Value {
    const candidate = @import("json_repair.zig").loads(arena, source) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.ParseValue,
    };
    if (candidate == .string and candidate.string.len == 0 and source.len != 0) {
        return @import("python_literal.zig").parse(arena, source) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.JsonTooDeep, error.LiteralEvaluation => error.ParseValue,
            error.ParseValue, error.IntegerTooLong => .{ .string = source },
        };
    }
    return candidate;
}

fn literalText(text: []const u8) []const u8 {
    var result = pyStrip(text);
    for ([_][]const u8{ "Literal[", "str[" }) |prefix| {
        if (std.mem.startsWith(u8, result, prefix) and std.mem.endsWith(u8, result, "]")) {
            result = result[prefix.len .. result.len - 1];
            break;
        }
    }
    if (result.len > 1 and result[0] == result[result.len - 1] and
        (result[0] == '"' or result[0] == '\''))
        result = result[1 .. result.len - 1];
    return result;
}

fn literalMatch(field: Field, text: []const u8) bool {
    for (field.literals) |allowed| if (std.mem.eql(u8, allowed, text)) return true;
    return false;
}

/// Allocation-free exact numeric normalization. Pydantic string/bytes coercion
/// is separate because removing separators may require owned storage.
pub fn parseInteger(v: Value) error{ParseValue}!Value {
    return switch (v) {
        .integer => v,
        .bool => |b| .{ .integer = @intFromBool(b) },
        .string, .number_string => |s| blk: {
            if (!isJsonNumber(s) or !std.json.isNumberFormattedLikeAnInteger(s)) break :blk error.ParseValue;
            break :blk Value.parseFromNumberSlice(s);
        },
        .float => |f| floatToInt(f),
        else => error.ParseValue,
    };
}

/// Pydantic's lax integer strings allow signs, zero prefixes, underscores and
/// a zero-only fractional suffix, but not exponent strings or non-ASCII digits.
pub fn coerceInteger(arena: Allocator, input: Value) ParseError!Value {
    var v = input;
    if (v == .custom) {
        if (v.custom.vtable.enumeration) |inspect| return inspect(v.custom.ptr).value;
    }
    if (v == .custom) {
        const view = if (v.custom.vtable.python) |inspect| inspect(v.custom.ptr) else return error.ParseValue;
        v = switch (view) {
            .bytes => |data| .{ .string = data },
            else => return error.ParseValue,
        };
    }
    if (v != .string) return parseInteger(v);
    const source = @import("unicode.zig").strip(v.string, true);
    if (source.len == 0) return error.ParseValue;
    var start: usize = 0;
    const negative = source[0] == '-';
    if (source[0] == '+' or negative) start = 1;
    var end = start;
    var digits: usize = 0;
    var separated = false;
    while (end < source.len and source[end] != '.') : (end += 1) {
        if (source[end] == '_') {
            if (end == start or end + 1 == source.len or
                !std.ascii.isDigit(source[end - 1]) or !std.ascii.isDigit(source[end + 1]))
                return error.ParseValue;
            separated = true;
        } else if (std.ascii.isDigit(source[end])) {
            digits += 1;
        } else return error.ParseValue;
    }
    if (digits == 0 or digits > 4300) return error.ParseValue;
    if (end != source.len) {
        if (end + 1 == source.len) return error.ParseValue;
        for (source[end + 1 ..], 0..) |byte, offset| {
            if (byte == '_') {
                const index = end + 1 + offset;
                if (offset == 0 or index + 1 == source.len or source[index - 1] != '0' or source[index + 1] != '0')
                    return error.ParseValue;
            } else if (byte != '0') return error.ParseValue;
        }
    }
    var first = start;
    while (first < end and (source[first] == '0' or source[first] == '_')) first += 1;
    if (first == end) return .{ .integer = 0 };
    if (std.fmt.parseInt(i64, source[0..end], 10)) |integer|
        return .{ .integer = integer }
    else |_| {}
    const magnitude = source[first..end];
    if (!separated) {
        if (!negative) return Value.parseFromNumberSlice(magnitude);
        if (first == start) return Value.parseFromNumberSlice(source[0..end]);
    }
    const count = std.mem.count(u8, magnitude, "_");
    const canonical = try arena.alloc(u8, @intFromBool(negative) + magnitude.len - count);
    var index: usize = 0;
    if (negative) {
        canonical[0] = '-';
        index = 1;
    }
    for (magnitude) |byte| if (byte != '_') {
        canonical[index] = byte;
        index += 1;
    };
    return Value.parseFromNumberSlice(canonical);
}

fn floatToInt(f: f64) error{ParseValue}!Value {
    if (!std.math.isFinite(f) or @trunc(f) != f) return error.ParseValue;
    if (f <= -0x1p63 or f >= 0x1p63) return error.ParseValue;
    return .{ .integer = @intFromFloat(f) };
}

/// JSON number grammar: -?(0|[1-9]\\d*)(\\.\\d+)?([eE][+-]?\\d+)?
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
    if (comptime code.isCode(T) or @import("enum.zig").isEnum(T) or @import("literal.zig").isLiteral(T) or T == @import("integer.zig").Integer) return v.toRuntime(arena);
    if (comptime decision.isDecision(T) or citation.isCitation(T) or T == @import("file.zig").File or T == @import("bounding_box.zig").BoundingBox or T == signature.Reasoning)
        return v.toRuntime(arena);
    if (T == @import("tool.zig").ToolCalls) return v.toValue(arena);
    if (T == Value) return v;
    if (T == []const u8) return .{ .string = v };
    if (T == []const []const u8) {
        var array: @import("runtime_value.zig").Array = .init(arena);
        try array.ensureTotalCapacity(v.len);
        for (v) |s| array.appendAssumeCapacity(.{ .string = s });
        return .{ .array = array };
    }
    if (T == bool) return .{ .bool = v };
    if (T == @import("runtime_value.zig").ObjectMap) return .{ .object = v };
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
            var array: @import("runtime_value.zig").Array = .init(arena);
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
    if (comptime decision.isDecision(T)) return T.fromValue(arena, v) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TypeMismatch,
    };
    if (comptime citation.isCitation(T)) return T.fromValue(arena, v) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TypeMismatch,
    };
    if (comptime code.isCode(T)) return T.fromValue(v) catch error.TypeMismatch;
    if (T == @import("bounding_box.zig").BoundingBox) return T.fromValue(v) catch error.TypeMismatch;
    if (T == @import("file.zig").File) return T.fromValue(arena, v) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TypeMismatch,
    };
    if (T == Value) return v;
    if (T == @import("integer.zig").Integer) return T.fromValue(v);
    if (comptime @import("enum.zig").isEnum(T) or @import("literal.zig").isLiteral(T)) return T.fromValue(v) catch error.TypeMismatch;
    if (T == @import("tool.zig").ToolCalls) return T.fromValue(arena, v) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TypeMismatch,
    };
    if (T == signature.Reasoning) return T.fromValue(v) catch error.TypeMismatch;
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
    if (T == @import("runtime_value.zig").ObjectMap) return if (v == .object) v.object else error.TypeMismatch;
    if (T == signature.StrDict) {
        if (v != .object) return error.TypeMismatch;
        for (v.object.values()) |item| if (item != .string) return error.TypeMismatch;
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
    if (std.meta.activeTag(a) != std.meta.activeTag(b) or
        (a == .custom and b == .custom and !a.custom.sameType(b.custom)))
    {
        if (a == .custom) {
            if (a.custom.vtable.eql_value) |equal| return equal(a.custom.ptr, b);
        }
        if (b == .custom) {
            if (b.custom.vtable.eql_value) |equal| return equal(b.custom.ptr, a);
        }
    }
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
        .custom => |custom| b == .custom and custom.eql(b.custom),
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

    fn fromValue(v: Value, buffer: *[20]u8) ?IntegerText {
        return parse(switch (v) {
            .bool => |b| if (b) "1" else "0",
            .integer => |i| std.fmt.bufPrint(buffer, "{d}", .{i}) catch unreachable,
            .number_string => |text| text,
            else => return null,
        });
    }

    fn order(a: IntegerText, b: IntegerText) std.math.Order {
        if (a.negative != b.negative) return if (a.negative) .lt else .gt;
        const magnitude = if (a.digits.len == b.digits.len)
            std.mem.order(u8, a.digits, b.digits)
        else
            std.math.order(a.digits.len, b.digits.len);
        return if (a.negative) magnitude.invert() else magnitude;
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
            break :blk floor.order(integer) != .lt;
        },
        else => error.InvalidNumber,
    };
}

/// Python arithmetic's implicit float conversion, including integer overflow.
pub fn numericFloat(v: Value) error{InvalidNumber}!f64 {
    return switch (v) {
        .bool => |b| if (b) 1 else 0,
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |text| blk: {
            const f = std.fmt.parseFloat(f64, text) catch return error.InvalidNumber;
            if (IntegerText.parse(text) != null and !std.math.isFinite(f)) return error.InvalidNumber;
            break :blk f;
        },
        else => error.InvalidNumber,
    };
}

/// Python numeric ordering. NaNs are unordered; integers never round to float.
pub fn numericOrder(a: Value, b: Value) error{InvalidNumber}!?std.math.Order {
    var a_buffer: [20]u8 = undefined;
    var b_buffer: [20]u8 = undefined;
    const a_integer = IntegerText.fromValue(a, &a_buffer);
    const b_integer = IntegerText.fromValue(b, &b_buffer);
    if (a_integer) |integer| {
        if (b_integer) |other| return integer.order(other);
        const f = try numericFloat(b);
        if (std.math.isNan(f)) return null;
        if (pyEqual(a, .{ .float = f })) return .eq;
        return if (try floatAtLeast(f, a)) .lt else .gt;
    }
    const f = try numericFloat(a);
    if (b_integer != null) {
        if (std.math.isNan(f)) return null;
        if (pyEqual(.{ .float = f }, b)) return .eq;
        return if (try floatAtLeast(f, b)) .gt else .lt;
    }
    const g = try numericFloat(b);
    if (std.math.isNan(f) or std.math.isNan(g)) return null;
    return std.math.order(f, g);
}

/// Python numeric multiplication. Integer-only products remain exact; float
/// products first convert BOTH operands, even when the other operand is zero.
/// Zero/one integer products borrow unchanged numeric text without allocation.
pub fn numericProduct(arena: Allocator, a: Value, b: Value) (Allocator.Error || error{InvalidNumber})!Value {
    var a_buffer: [20]u8 = undefined;
    var b_buffer: [20]u8 = undefined;
    if (IntegerText.fromValue(a, &a_buffer)) |a_integer| {
        if (IntegerText.fromValue(b, &b_buffer)) |b_integer| {
            if (std.mem.eql(u8, a_integer.digits, "0") or std.mem.eql(u8, b_integer.digits, "0"))
                return .{ .integer = 0 };
            if (!a_integer.negative and std.mem.eql(u8, a_integer.digits, "1"))
                return if (b == .bool) .{ .integer = @intFromBool(b.bool) } else b;
            if (!b_integer.negative and std.mem.eql(u8, b_integer.digits, "1"))
                return if (a == .bool) .{ .integer = @intFromBool(a.bool) } else a;
            const a_small: ?i64 = switch (a) {
                .integer => |i| i,
                .number_string => |text| std.fmt.parseInt(i64, text, 10) catch null,
                else => null,
            };
            const b_small: ?i64 = switch (b) {
                .integer => |i| i,
                .number_string => |text| std.fmt.parseInt(i64, text, 10) catch null,
                else => null,
            };
            if (a_small != null and b_small != null) {
                const product = @as(i128, a_small.?) * @as(i128, b_small.?);
                if (std.math.cast(i64, product)) |i| return .{ .integer = i };
                return .{ .number_string = try std.fmt.allocPrint(arena, "{d}", .{product}) };
            }
            var left = try std.math.big.int.Managed.init(arena);
            defer left.deinit();
            var right = try std.math.big.int.Managed.init(arena);
            defer right.deinit();
            var product = try std.math.big.int.Managed.init(arena);
            defer product.deinit();
            try setBigInteger(&left, a);
            try setBigInteger(&right, b);
            try product.mul(&left, &right);
            return .{ .number_string = try product.toConst().toStringAlloc(arena, 10, .lower) };
        }
    }
    const left = try numericFloat(a);
    const right = try numericFloat(b);
    return .{ .float = left * right };
}

fn setBigInteger(out: *std.math.big.int.Managed, v: Value) (Allocator.Error || error{InvalidNumber})!void {
    switch (v) {
        .integer => |i| try out.set(i),
        .bool => |b| try out.set(@intFromBool(b)),
        .number_string => |text| out.setString(10, text) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidNumber,
        },
        else => return error.InvalidNumber,
    }
}

fn floatEqualsInt(f: f64, i: i64) bool {
    // Converting the integer to f64 first would merge distinct values above 2^53.
    if (!std.math.isFinite(f) or f != @trunc(f) or f < -0x1p63 or f >= 0x1p63) return false;
    return @as(i64, @intFromFloat(f)) == i;
}

/// `serialize_for_json`: only actual custom objects become their serialized
/// values. Ordinary dictionaries with identical keys stay dictionaries.
pub fn serializedValue(arena: Allocator, input: Value) ParseError!Value {
    const result = serializeDepth(arena, input, 0) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.ParseValue => error.ParseValue,
        error.PythonSerializeFallback => blk: {
            var out: Writer.Allocating = .init(arena);
            pyjson.writeStr(&out.writer, input) catch |write_err| return switch (write_err) {
                error.WriteFailed => error.OutOfMemory,
                error.JsonTooDeep, error.ParseValue => error.ParseValue,
            };
            break :blk .{ .string = out.written() };
        },
    };
    return result.value;
}

/// Strict JSON-mode projection for nested custom serializers. Only the public
/// top-level boundary may replace the whole original value after a fallback.
pub fn serializedValueDepth(arena: Allocator, input: Value, depth: u32) @import("runtime_value.zig").SerializationError!Value {
    return (try serializeDepth(arena, input, depth)).value;
}

test "DSPy JSON mode keeps a root nonfinite float but projects nested floats to null" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try serializedValue(a, .{ .float = std.math.inf(f64) });
    try std.testing.expect(root == .float and std.math.isPositiveInf(root.float));
    var items: @import("runtime_value.zig").Array = .init(a);
    try items.append(.{ .float = std.math.inf(f64) });
    try items.append(.{ .float = std.math.nan(f64) });
    try items.append(.{ .float = 2.5 });
    const nested = try serializedValue(a, .{ .array = items });
    try std.testing.expect(nested.array.items[0] == .null);
    try std.testing.expect(nested.array.items[1] == .null);
    try std.testing.expectEqual(@as(f64, 2.5), nested.array.items[2].float);
}
const Serialized = struct { value: Value, changed: bool };

fn serializeDepth(arena: Allocator, input: Value, depth: u32) @import("runtime_value.zig").SerializationError!Serialized {
    if (depth >= pyjson.max_depth) return error.ParseValue;
    return switch (input) {
        .float => |f| if (depth > 0 and !std.math.isFinite(f))
            .{ .value = .null, .changed = true }
        else
            .{ .value = input, .changed = false },
        .custom => |custom| blk: {
            if (custom.vtable.python) |inspect| {
                if (inspect(custom.ptr) == .ellipsis)
                    break :blk if (depth == 0)
                        .{ .value = input, .changed = false }
                    else
                        .{ .value = .{ .string = "Ellipsis" }, .changed = true };
            }
            break :blk .{
                .value = try custom.vtable.serialized(arena, custom.ptr, depth),
                .changed = true,
            };
        },
        .array => |array| blk: {
            var out: ?@import("runtime_value.zig").Array = null;
            for (array.items, 0..) |item, index| {
                const serialized = try serializeDepth(arena, item, depth + 1);
                if (out == null and serialized.changed) {
                    out = .init(arena);
                    try out.?.ensureTotalCapacity(array.items.len);
                    out.?.appendSliceAssumeCapacity(array.items[0..index]);
                }
                if (out) |*result| result.appendAssumeCapacity(serialized.value);
            }
            break :blk if (out) |result|
                .{ .value = .{ .array = result }, .changed = true }
            else
                .{ .value = input, .changed = false };
        },
        .object => |object| blk: {
            var out: ?ObjectMap = null;
            for (object.keys(), object.values(), 0..) |key, item, index| {
                const serialized = try serializeDepth(arena, item, depth + 1);
                if (out == null and serialized.changed) {
                    out = .empty;
                    try out.?.ensureTotalCapacity(arena, object.count());
                    for (object.keys()[0..index], object.values()[0..index]) |before_key, before_value|
                        out.?.putAssumeCapacity(before_key, before_value);
                }
                if (out) |*result| result.putAssumeCapacity(key, serialized.value);
            }
            break :blk if (out) |result|
                .{ .value = .{ .object = result }, .changed = true }
            else
                .{ .value = input, .changed = false };
        },
        else => .{ .value = input, .changed = false },
    };
}

/// Deep copy of a value into `arena` (strings included).
pub fn cloneValue(arena: Allocator, v: Value) Allocator.Error!Value {
    return switch (v) {
        .custom => |custom| .{ .custom = try custom.clone(arena) },
        .string => |s| .{ .string = try arena.dupe(u8, s) },
        .number_string => |s| .{ .number_string = try arena.dupe(u8, s) },
        .array => |a| blk: {
            var out: @import("runtime_value.zig").Array = .init(arena);
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

test "field annotations do not turn dictionaries or class-annotated lists into text blobs" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var dictionary: Record = .empty;
    try dictionary.put(arena, "content", .{ .string = "not a class" });
    var list: @import("runtime_value.zig").Array = .init(arena);
    try list.append(.{ .string = "first" });
    try list.append(.{ .string = "second" });
    var fields: [4]Field = @splat(testField(.str));
    fields[0].reasoning_type = true;
    fields[1].code_type = true;
    fields[2].optional_str = true;
    var output: Writer.Allocating = .init(arena);
    for (fields, 0..) |field, index| {
        output.clearRetainingCapacity();
        try formatFieldValue(&output.writer, field, .{ .object = dictionary });
        try std.testing.expectEqualStrings("{\"content\": \"not a class\"}", output.written());
        output.clearRetainingCapacity();
        try formatFieldValue(&output.writer, field, .{ .array = list });
        try std.testing.expectEqualStrings(
            if (index == 3) "[1] «first»\n[2] «second»" else "[\"first\", \"second\"]",
            output.written(),
        );
    }
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
