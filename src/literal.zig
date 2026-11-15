//! Explicit typing.Literal annotations. These retain the allowed scalar or
//! genuine Enum member or bytes, and DSPy's membership shortcut preserves input type.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const value = @import("value.zig");
const Value = runtime.Value;
const Allocator = std.mem.Allocator;

pub fn isLiteral(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "dszig_literal_values");
}

fn repr(comptime v: Value, comptime as_str: bool) []const u8 {
    comptime {
        const size = switch (v) {
            .string, .number_string => |s| s.len * 10 + 64,
            .custom => |c| blk: {
                if (c.vtable.enumeration) |inspect| {
                    const member = inspect(c.ptr);
                    break :blk member.class_name.len + member.member_name.len + 1;
                }
                break :blk @import("python_value.zig").view(v).?.bytes.len * 4 + 3;
            },
            else => 128,
        };
        var buffer: [size]u8 = undefined;
        var writer: std.Io.Writer = .fixed(&buffer);
        if (as_str) @import("pyjson.zig").writeStr(&writer, v) catch @compileError("dszig: invalid Literal value") else @import("pyjson.zig").writeRepr(&writer, v, 0) catch @compileError("dszig: invalid Literal value");
        const result = buffer[0..writer.buffered().len].*;
        return &result;
    }
}

fn smallInt(v: Value) ?i64 {
    return switch (v) {
        .integer => |i| i,
        .number_string => |s| std.fmt.parseInt(i64, s, 10) catch null,
        else => null,
    };
}

fn primitive(v: Value) bool {
    return v == .bool or v == .string or smallInt(v) != null;
}

/// Literal arguments are flattened explicitly. Equal values of different
/// Python types remain separate (Literal[1, True] has distinct fast buckets).
/// Enum members retain their class identity rather than becoming their values.
/// Bytes arguments use python_value.staticBytes. Dynamic annotations remain
/// unsupported; non-UTF-8 bytes validate but cannot produce a JSON schema.
pub fn Literal(comptime arguments: []const Value) type {
    const allowed = comptime blk: {
        @setEvalBranchQuota(100_000);
        var result: [arguments.len]Value = undefined;
        var len: usize = 0;
        for (arguments) |v| {
            switch (v) {
                .null, .bool, .integer, .float, .string => {},
                .number_string => |s| if (!std.json.isNumberFormattedLikeAnInteger(s))
                    @compileError("dszig: Literal number_string must be an integer"),
                .custom => |c| {
                    if (c.vtable.enumeration == null) {
                        const builtin = @import("python_value.zig").view(v) orelse
                            @compileError("dszig: Literal custom arguments must be genuine Enum members or bytes");
                        if (builtin != .bytes)
                            @compileError("dszig: Literal custom arguments must be genuine Enum members or bytes");
                    }
                },
                else => @compileError("dszig: Literal supports scalars, Enum members and bytes only"),
            }
            for (result[0..len]) |previous| {
                const same_kind = (smallInt(v) != null and smallInt(previous) != null) or std.meta.activeTag(v) == std.meta.activeTag(previous);
                if (same_kind and value.pyEqual(v, previous)) break;
            } else {
                result[len] = v;
                len += 1;
            }
        }
        if (len == 0) @compileError("dszig: Literal must have at least one value");
        const final = result[0..len].*;
        break :blk final;
    };
    return struct {
        const Self = @This();
        pub const dszig_literal_values = allowed;
        pub const annotation_name = blk: {
            var text: []const u8 = "Literal[";
            for (allowed, 0..) |v, i| text = text ++ (if (i == 0) "" else ", ") ++ repr(v, v == .custom);
            break :blk text ++ "]";
        };
        pub const display_values = blk: {
            var result: [allowed.len][]const u8 = undefined;
            for (allowed, 0..) |v, i| result[i] = repr(v, true);
            break :blk result;
        };
        value: Value,

        pub fn validate(_: Allocator, input: Value) runtime.CustomError!Value {
            return validateValues(&allowed, input);
        }

        pub fn parseValue(arena: Allocator, input: Value) runtime.CustomError!Value {
            return parseValues(arena, &allowed, input);
        }

        /// Decode an already validated prediction, retaining DSPy's shortcut
        /// type (e.g. True for Literal[1]) rather than validating it twice.
        pub fn fromValue(input: Value) error{ParseValue}!Self {
            for (allowed) |v| if (value.pyEqual(v, input)) return .{ .value = input };
            return error.ParseValue;
        }
        pub fn toRuntime(self: Self, _: Allocator) Allocator.Error!Value {
            return self.value;
        }
    };
}

/// Runtime schema-backed literals (XML leaves) use the same lookup as typed
/// annotations. No coercion occurs before exact bool/int/string selection.
pub fn validateValues(allowed: []const Value, input: Value) error{ParseValue}!Value {
    if (input == .bool or input == .string) {
        var i = allowed.len;
        while (i > 0) {
            i -= 1;
            const v = allowed[i];
            if (std.meta.activeTag(v) == std.meta.activeTag(input) and value.pyEqual(v, input)) return v;
        }
    }
    const has_int = for (allowed) |v| {
        if (smallInt(v) != null) break true;
    } else false;
    if (has_int and (input == .integer or input == .number_string)) {
        const integer = smallInt(input) orelse return error.ParseValue;
        var i = allowed.len;
        while (i > 0) {
            i -= 1;
            const v = allowed[i];
            if (smallInt(v)) |expected| if (expected == integer) return v;
        }
    }
    var i = allowed.len;
    while (i > 0) {
        i -= 1;
        const v = allowed[i];
        if ((!primitive(v) or v == .bool) and value.pyEqual(v, input)) return v;
    }
    i = allowed.len;
    while (i > 0) {
        i -= 1;
        const v = allowed[i];
        if (primitive(v) and value.pyEqual(v, input)) return v;
    }
    return error.ParseValue;
}

fn identity(_: Allocator, input: Value) runtime.CustomError!Value {
    return input;
}

/// DSPy's membership shortcut, wrapper stripping and repair-first fallback.
pub fn parseValues(arena: Allocator, allowed: []const Value, input: Value) runtime.CustomError!Value {
    for (allowed) |v| if (value.pyEqual(v, input)) return input;
    if (input != .string) return validateValues(allowed, input);
    var text = value.pyStrip(input.string);
    for ([_][]const u8{ "Literal[", "str[" }) |prefix| {
        if (std.mem.startsWith(u8, text, prefix) and std.mem.endsWith(u8, text, "]")) {
            text = text[prefix.len .. text.len - 1];
            break;
        }
    }
    if (text.len > 1 and text[0] == text[text.len - 1] and (text[0] == '\'' or text[0] == '"')) text = text[1 .. text.len - 1];
    const stripped: Value = .{ .string = text };
    for (allowed) |v| if (value.pyEqual(v, stripped)) return stripped;
    const candidate = try value.parseText(arena, .{
        .name = "literal",
        .kind = .json,
        .type_name = "Literal",
        .desc = "",
        .prefix = "",
        .validate = identity,
    }, text);
    return validateValues(allowed, candidate);
}

test "Bytes Literal keeps bytes distinct from text and owning clones free normally" {
    const python = @import("python_value.zig");
    const a = std.testing.allocator;
    const binary = comptime python.staticBytes("hi");
    const Binary = Literal(&.{binary});
    try std.testing.expectError(error.ParseValue, Binary.validate(a, .{ .string = "hi" }));
    var scratch: std.heap.ArenaAllocator = .init(a);
    defer scratch.deinit();
    const parsed = try Binary.parseValue(scratch.allocator(), .{ .string = "b'hi'" });
    try std.testing.expectEqualStrings("hi", python.view(parsed).?.bytes);
    const Mixed = Literal(&.{ binary, .{ .string = "hi" } });
    try std.testing.expectEqualStrings("hi", (try Mixed.validate(a, .{ .string = "hi" })).string);
    try std.testing.expectEqualStrings("{\"type\": \"string\", \"enum\": [\"hi\", \"hi\"]}", @import("types.zig").schema(Mixed));
    const Invalid = Literal(&.{comptime python.staticBytes("\xff")});
    try std.testing.expectEqualStrings("\xff", python.view(try Invalid.validate(a, comptime python.staticBytes("\xff"))).?.bytes);
    const owned = try @import("inference.zig").cloneJsonValue(a, binary);
    defer @import("inference.zig").freeJsonValue(a, owned);
    binary.custom.deinit(a);
    try std.testing.expectEqualStrings("hi", python.view(owned).?.bytes);
    const copied = try @import("inference.zig").cloneJsonValue(a, owned);
    defer @import("inference.zig").freeJsonValue(a, copied);
    try std.testing.expectEqualStrings("hi", python.view(copied).?.bytes);
}

test "Enum Literal preserves member identity and rejects equal projected values" {
    const enums = @import("enum.zig");
    const entries = &[_]enums.Entry{
        .{ .name = "red", .value = .{ .string = "red" } },
        .{ .name = "alias", .value = .{ .string = "red" } },
    };
    const Left = enums.EnumWithIdentity("Color", entries, struct {});
    const Right = enums.EnumWithIdentity("Color", entries, struct {});
    const red = comptime Left.member("red") catch unreachable;
    const T = Literal(&.{ red, comptime Left.member("alias") catch unreachable });
    const a = std.testing.allocator;
    try std.testing.expect((try T.validate(a, red)).custom.ptr == red.custom.ptr);
    try std.testing.expect((try T.parseValue(a, red)).custom.ptr == red.custom.ptr);
    try std.testing.expectError(error.ParseValue, T.validate(a, try Right.member("red")));
    try std.testing.expectError(error.ParseValue, T.parseValue(a, .{ .string = "red" }));
    try std.testing.expectError(error.ParseValue, T.parseValue(a, .{ .string = "Color.red" }));
    const Mixed = Literal(&.{ red, .{ .string = "red" } });
    try std.testing.expect((try Mixed.parseValue(a, red)).custom.ptr == red.custom.ptr);
    try std.testing.expectEqualStrings("red", (try Mixed.parseValue(a, .{ .string = "red" })).string);
    const Wrapped = enums.Enum("Wrapped", &.{.{ .name = "member", .value = red }});
    const Nested = Literal(&.{comptime Wrapped.member("member") catch unreachable});
    try std.testing.expectEqualStrings("{\"type\": \"string\", \"const\": \"red\"}", @import("types.zig").schema(Nested));
}

test "Mixed Literal retains DSPy membership type while Pydantic selects exact buckets" {
    const T = Literal(&.{ .{ .integer = 1 }, .{ .bool = true } });
    const a = std.testing.allocator;
    try std.testing.expect((try T.validate(a, .{ .float = 1 })).bool);
    try std.testing.expectEqual(@as(i64, 1), (try T.validate(a, .{ .integer = 1 })).integer);
    try std.testing.expectEqual(@as(f64, 1), (try T.parseValue(a, .{ .float = 1 })).float);
    try std.testing.expect((try T.parseValue(a, .{ .string = "Literal[True]" })).bool);
    try std.testing.expectError(error.ParseValue, T.validate(a, .{ .string = "1" }));
    try std.testing.expectError(error.ParseValue, T.parseValue(a, .{ .integer = 2 }));
    const Numeric = Literal(&.{ .{ .integer = 1 }, .{ .float = 1 } });
    try std.testing.expectEqual(@as(f64, 1), (try Numeric.validate(a, .{ .bool = true })).float);
    try std.testing.expect((try Numeric.parseValue(a, .{ .bool = true })).bool);
}

test "Nested XML Literal validates numeric equality after leaf parsing" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const T = Literal(&.{ .{ .integer = 1 }, .{ .bool = true }, .{ .string = "ready" }, .null });
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { choices: []const T, optional: ?T };
    });
    const adapter = @import("adapter.zig");
    const outputs = try adapter.Adapter.xml.parse(a, .{
        .instructions = "",
        .inputs = Sig.input_fields_list,
        .outputs = Sig.output_fields_list,
    }, "<choices><item>1.0</item><item>true</item><item>ready</item><item>null</item></choices><optional>null</optional>");
    const typed = try value.fromRecord(a, Sig.Output, &outputs);
    try std.testing.expectEqual(@as(usize, 4), typed.choices.len);
    try std.testing.expect(typed.choices[0].value.bool);
    try std.testing.expect(typed.choices[1].value.bool);
    try std.testing.expectEqualStrings("ready", typed.choices[2].value.string);
    try std.testing.expect(typed.choices[3].value == .null);
    try std.testing.expect(typed.optional == null);
}

test "Literal large integers respect Pydantic fast bucket overflow but not DSPy membership" {
    const big: Value = .{ .number_string = "100000000000000000000000000000000000000000000000000" };
    const Mixed = Literal(&.{ .{ .integer = 0 }, big });
    const Only = Literal(&.{big});
    const a = std.testing.allocator;
    try std.testing.expectError(error.ParseValue, Mixed.validate(a, big));
    try std.testing.expectEqualStrings(big.number_string, (try Mixed.parseValue(a, big)).number_string);
    try std.testing.expectEqualStrings(big.number_string, (try Only.validate(a, big)).number_string);
}
