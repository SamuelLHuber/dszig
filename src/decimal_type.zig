//! First-class storage for a validated Python Decimal annotation.
//! The runtime box retains exact coefficient, quantum and object identity;
//! Pydantic coercion is distinct from Decimal's context-sensitive constructors.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const decimals = @import("decimal.zig");
const python = @import("python_value.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;

pub const Decimal = struct {
    value: Value,

    /// Decode an already validated result without running field validation a
    /// second time. A genuine nonfinite Decimal can be stored, but cannot pass
    /// this annotation's ordinary finite-only validator.
    pub fn fromValue(input: Value) error{ParseValue}!Decimal {
        const builtin = python.view(input) orelse return error.ParseValue;
        if (builtin != .decimal) return error.ParseValue;
        return .{ .value = input };
    }

    /// This view borrows the genuine immutable runtime value. Owning copies use
    /// the runtime Custom.clone protocol, not Python deepcopy identity.
    pub fn toRuntime(self: Decimal, _: Allocator) Allocator.Error!Value {
        return self.value;
    }

    pub fn coerce(allocator: Allocator, input: Value) runtime.CustomError!Decimal {
        return .{ .value = try validate(allocator, input) };
    }

    /// TypeAdapter(Decimal).validate_python with Pydantic's default lax policy.
    /// Existing finite Decimal instances are retained; bool, bytes and ordinary
    /// Enum members are not numerical surrogates for this annotation.
    pub fn validate(allocator: Allocator, input: Value) runtime.CustomError!Value {
        return switch (input) {
            .custom => blk: {
                const builtin = python.view(input) orelse return error.ParseValue;
                if (builtin == .decimal) {
                    if (builtin.decimal.kind != .finite) return error.ParseValue;
                    break :blk input;
                }
                if (builtin != .tuple) return error.ParseValue;
                const result = try python.decimalFromTuple(allocator, input);
                if (python.view(result).?.decimal.kind != .finite) {
                    result.custom.deinit(allocator);
                    return error.ParseValue;
                }
                break :blk result;
            },
            .string => |text| parseFinite(allocator, text),
            .integer => |number| blk: {
                var buffer: [21]u8 = undefined;
                const text = std.fmt.bufPrint(&buffer, "{d}", .{number}) catch unreachable;
                break :blk parseFinite(allocator, text);
            },
            .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text))
                parseFinite(allocator, text)
            else
                fromFloatString(allocator, std.fmt.parseFloat(f64, text) catch return error.ParseValue),
            .float => |number| fromFloatString(allocator, number),
            else => error.ParseValue,
        };
    }

    /// TypeAdapter(Decimal).validate_json. Decimal accepts JSON strings and
    /// numbers even in strict mode. JSON float coercion uses shortest plain
    /// decimal text, unlike validate_python's Python float repr.
    pub fn validateJson(allocator: Allocator, text: []const u8) runtime.CustomError!Value {
        const input = @import("json_repair.zig").parseJson(allocator, text) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        };
        return switch (input) {
            .float => |number| fromJsonFloat(allocator, number),
            .number_string => |number| if (std.json.isNumberFormattedLikeAnInteger(number))
                parseFinite(allocator, number)
            else
                fromJsonFloat(allocator, std.fmt.parseFloat(f64, number) catch return error.ParseValue),
            else => validate(allocator, input),
        };
    }

    fn fromJsonFloat(allocator: Allocator, number: f64) runtime.CustomError!Value {
        if (!std.math.isFinite(number)) return error.ParseValue;
        var repr_buffer: [64]u8 = undefined;
        var repr_writer: std.Io.Writer = .fixed(&repr_buffer);
        pyjson.writeFloat(&repr_writer, number, .python) catch unreachable;
        const representation = repr_writer.buffered();
        const negative = representation[0] == '-';
        const unsigned = representation[@intFromBool(negative)..];
        const exponent_at = std.mem.indexOfAny(u8, unsigned, "eE") orelse unsigned.len;
        const mantissa = unsigned[0..exponent_at];
        const exponent: i16 = if (exponent_at < unsigned.len)
            std.fmt.parseInt(i16, unsigned[exponent_at + 1 ..], 10) catch unreachable
        else
            0;
        var digits: [24]u8 = undefined;
        var length: usize = 0;
        var before_point: i16 = 0;
        var after_point = false;
        for (mantissa) |digit| {
            if (digit == '.') {
                after_point = true;
                continue;
            }
            digits[length] = digit;
            length += 1;
            if (!after_point) before_point += 1;
        }
        while (length > 1 and digits[length - 1] == '0') length -= 1;
        const point: i16 = before_point + exponent;
        // The longest plain finite binary64 spelling fits comfortably: the
        // smallest subnormal needs "0.", 323 zeroes and its final digit.
        var output: [384]u8 = undefined;
        var writer: std.Io.Writer = .fixed(&output);
        if (negative) writer.writeByte('-') catch unreachable;
        if (length == 1 and digits[0] == '0') {
            writer.writeByte('0') catch unreachable;
        } else if (point <= 0) {
            writer.writeAll("0.") catch unreachable;
            writer.splatByteAll('0', @intCast(-point)) catch unreachable;
            writer.writeAll(digits[0..length]) catch unreachable;
        } else if (point >= @as(i16, @intCast(length))) {
            writer.writeAll(digits[0..length]) catch unreachable;
            writer.splatByteAll('0', @as(usize, @intCast(point)) - length) catch unreachable;
        } else {
            const split: usize = @intCast(point);
            writer.writeAll(digits[0..split]) catch unreachable;
            writer.writeByte('.') catch unreachable;
            writer.writeAll(digits[split..length]) catch unreachable;
        }
        return parseFinite(allocator, writer.buffered());
    }

    fn parseFinite(allocator: Allocator, text: []const u8) runtime.CustomError!Value {
        // Pydantic constructs Decimal from text: syntax failures signal
        // InvalidOperation, unlike merely refusing an existing nonfinite value.
        const data = decimals.parse(allocator, text) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ParseValue => {
                @import("decimal_environment.zig").getContext().signal(.invalid_operation) catch return error.ParseValue;
                return error.ParseValue;
            },
        };
        errdefer data.deinit(allocator);
        if (data.kind != .finite) return error.ParseValue;
        return python.decimalFromOwnedMetadata(allocator, data);
    }

    fn fromFloatString(allocator: Allocator, number: f64) runtime.CustomError!Value {
        if (!std.math.isFinite(number)) return error.ParseValue;
        // Pydantic uses Decimal(str(float)), not Decimal(float)'s exact binary
        // expansion. Every Python binary64 repr fits in this fixed buffer.
        var buffer: [64]u8 = undefined;
        var writer: std.Io.Writer = .fixed(&buffer);
        pyjson.writeFloat(&writer, number, .python) catch unreachable;
        return parseFinite(allocator, writer.buffered());
    }
};

fn expectDecimalText(expected: []const u8, actual: Value) !void {
    const builtin = python.view(actual) orelse return error.TestUnexpectedResult;
    try std.testing.expect(builtin == .decimal);
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try builtin.decimal.write(&out.writer, true);
    try std.testing.expectEqualStrings(expected, out.written());
}

test "Decimal JSON float coercion differs from Python float repr without losing quoted quantum" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const cases = [_]struct { json: []const u8, expected: []const u8 }{
        .{ .json = "0.00", .expected = "0" },
        .{ .json = "-0.00", .expected = "-0" },
        .{ .json = "1.00", .expected = "1" },
        .{ .json = "\"1.00\"", .expected = "1.00" },
        .{ .json = "1e30", .expected = "1000000000000000000000000000000" },
        .{ .json = "9223372036854775807.1", .expected = "9223372036854776000" },
        .{ .json = "5e-324", .expected = "5E-324" },
        .{ .json = "18446744073709551617", .expected = "18446744073709551617" },
    };
    for (cases) |case| try expectDecimalText(case.expected, try Decimal.validateJson(a, case.json));
    try expectDecimalText("1.0", try Decimal.validate(a, .{ .float = 1.0 }));
    for ([_][]const u8{ "true", "null", "[]", "{}", "1e1000", "1 trailing" }) |invalid|
        try std.testing.expectError(error.ParseValue, Decimal.validateJson(a, invalid));
}

test "Decimal typed storage retains genuine identity without a second validation pass" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const original = try python.decimal(a, "-0.00");
    const checked = try Decimal.coerce(a, original);
    try std.testing.expectEqual(original.custom.ptr, checked.value.custom.ptr);
    try std.testing.expectEqual(original.custom.ptr, (try checked.toRuntime(a)).custom.ptr);
    const decoded = try Decimal.fromValue(original);
    try std.testing.expectEqual(original.custom.ptr, decoded.value.custom.ptr);
    try expectDecimalText("-0.00", checked.value);

    for ([_][]const u8{ "NaN123", "-sNaN42", "Infinity", "-Infinity" }) |text| {
        const nonfinite = try python.decimal(a, text);
        try std.testing.expectError(error.ParseValue, Decimal.validate(a, nonfinite));
        const stored = try Decimal.fromValue(nonfinite);
        try std.testing.expectEqual(nonfinite.custom.ptr, stored.value.custom.ptr);
    }
    try std.testing.expectError(error.ParseValue, Decimal.fromValue(.{ .string = "1.00" }));
    try std.testing.expectError(error.ParseValue, Decimal.fromValue(.{ .integer = 1 }));
}

test "Decimal coercion preserves precision and uses Python float strings" {
    var local: @import("decimal_environment.zig").LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { input: Value, expected: []const u8 }{
        .{ .input = .{ .float = 0.1 }, .expected = "0.1" },
        .{ .input = .{ .float = -0.0 }, .expected = "-0.0" },
        .{ .input = .{ .float = std.math.pi }, .expected = "3.141592653589793" },
        .{ .input = .{ .float = 5e-324 }, .expected = "5E-324" },
        .{ .input = .{ .float = 1e100 }, .expected = "1E+100" },
        .{ .input = .{ .integer = std.math.minInt(i64) }, .expected = "-9223372036854775808" },
        .{ .input = .{ .number_string = "18446744073709551617" }, .expected = "18446744073709551617" },
        .{ .input = .{ .string = "1.2300" }, .expected = "1.2300" },
        .{ .input = .{ .string = "1e309" }, .expected = "1E+309" },
        .{
            .input = .{ .string = "9007199254740993.00000000000000000001" },
            .expected = "9007199254740993.00000000000000000001",
        },
    };
    for (cases) |case| try expectDecimalText(case.expected, (try Decimal.coerce(a, case.input)).value);

    const E = @import("enum.zig").Enum("NotADecimal", &.{.{ .name = "one", .value = .{ .integer = 1 } }});
    for ([_]Value{
        .null,
        .{ .bool = true },
        .{ .float = std.math.nan(f64) },
        .{ .float = std.math.inf(f64) },
        .{ .number_string = "1e309" },
        .{ .string = "not a decimal" },
        .{ .string = "NaN" },
        .{ .string = "Infinity" },
        python.staticBytes("1"),
        try E.member("one"),
    }) |input| try std.testing.expectError(error.ParseValue, Decimal.coerce(a, input));
}

test "Decimal annotations validate nested containers and parse scalar signature outputs" {
    const types = @import("types.zig");
    const values = @import("value.zig");
    const Payload = struct {
        total: Decimal,
        entries: []const ?Decimal,
        lookup: types.Dict(Decimal),
    };
    const S = @import("signature.zig").Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { amount: Decimal };
    });
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = try std.json.parseFromSliceLeaky(Value, a,
        \\{"total":"1.2300","entries":[null,"9007199254740993"],"lookup":{"tiny":"1e-1000"}}
    , .{});
    const checked = try types.validate(Payload, a, input);
    try expectDecimalText("1.2300", checked.object.get("total").?);
    const entries = checked.object.get("entries").?.array.items;
    try std.testing.expect(entries[0] == .null);
    try expectDecimalText("9007199254740993", entries[1]);
    try expectDecimalText("1E-1000", checked.object.get("lookup").?.object.get("tiny").?);
    try std.testing.expect((try types.validate(?Decimal, a, .null)) == .null);

    try std.testing.expectEqualStrings(
        "{\"anyOf\": [{\"type\": \"number\"}, {\"type\": \"string\"}]}",
        types.schema(Decimal),
    );
    const model_schema = try std.json.parseFromSliceLeaky(Value, a, types.schema(Payload), .{});
    try std.testing.expect(model_schema.object.get("$defs") == null);
    try expectDecimalText("1.2300", try values.parseText(a, S.output_fields_list[0], "\"1.2300\""));
    try expectDecimalText("1.23", try values.parseText(a, S.output_fields_list[0], "1.2300"));
    try std.testing.expectError(error.ParseValue, values.parseText(a, S.output_fields_list[0], "true"));
}

test "Decimal lax bool uses float rounding and underflow rather than truthiness" {
    const types = @import("types.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { text: []const u8, expected: ?bool }{
        .{ .text = "-0.00", .expected = false },
        .{ .text = "1.0000", .expected = true },
        .{ .text = "0.9999999999999999999999999999", .expected = true },
        .{ .text = "1.0000000000000000000000000001", .expected = true },
        .{ .text = "1e-1000", .expected = false },
        .{ .text = "-1e-1000", .expected = false },
        .{ .text = "-1e-28", .expected = null },
        .{ .text = "0.5", .expected = null },
        .{ .text = "NaN", .expected = null },
        .{ .text = "sNaN", .expected = null },
        .{ .text = "Infinity", .expected = null },
    };
    for (cases) |case| {
        const input = try python.decimal(a, case.text);
        if (case.expected) |expected| {
            try std.testing.expectEqual(expected, (try types.validate(bool, a, input)).bool);
        } else {
            try std.testing.expectError(error.ParseValue, types.validate(bool, a, input));
        }
    }
}

test "Decimal malformed strings signal InvalidOperation with either trap setting" {
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    for ([_]bool{ true, false }) |trapped| {
        local.context.traps = if (trapped) .initOne(.invalid_operation) else .empty;
        for ([_][]const u8{ "not-a-decimal", "1.2.3", "1E1000000000000000000" }) |text| {
            local.context.clearFlags();
            try std.testing.expectError(error.ParseValue, Decimal.validate(a, .{ .string = text }));
            try std.testing.expect(local.context.flags.contains(.invalid_operation));
            try std.testing.expect(!local.context.flags.contains(.float_operation));
        }
        for ([_][]const u8{ "NaN", "sNaN", "Infinity" }) |text| {
            local.context.clearFlags();
            try std.testing.expectError(error.ParseValue, Decimal.validate(a, .{ .string = text }));
            try std.testing.expect(!local.context.flags.contains(.invalid_operation));
        }
    }
}

test "Decimal validation does not inherit float constructor traps or precision" {
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    local.context.clearFlags();
    local.context.traps = .initMany(&.{ .invalid_operation, .float_operation, .division_by_zero, .overflow });

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectDecimalText("0.1", try Decimal.validate(a, .{ .float = 0.1 }));
    try std.testing.expect(!local.context.flags.contains(.float_operation));
    try std.testing.expect(!local.context.flags.contains(.invalid_operation));

    const signaling = try python.decimal(a, "sNaN");
    try std.testing.expectError(error.ParseValue, Decimal.validate(a, signaling));
    try std.testing.expect(!local.context.flags.contains(.invalid_operation));

    local.context.precision = 3;
    try expectDecimalText("3.141592653589793", try Decimal.validate(a, .{ .float = std.math.pi }));
    try std.testing.expect(!local.context.flags.contains(.inexact));
    try std.testing.expect(!local.context.flags.contains(.rounded));
}

test "Decimal validation accepts genuine constructor tuples but not root lists" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const coefficient = try python.tuple(a, &.{
        .{ .integer = 0 },
        .{ .integer = 0 },
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 0 },
    });
    const input = try python.tuple(a, &.{
        .{ .integer = 0 },
        coefficient,
        .{ .integer = -3 },
    });
    try expectDecimalText("0.120", try Decimal.validate(a, input));

    const empty = try python.tuple(a, &.{});
    const zero = try python.tuple(a, &.{
        .{ .integer = 1 },
        empty,
        .{ .integer = -2 },
    });
    try expectDecimalText("-0.00", try Decimal.validate(a, zero));

    var digits: runtime.Array = .init(a);
    try digits.appendSlice(&.{
        .{ .bool = true },
        .{ .bool = false },
        .{ .integer = 1 },
    });
    const from_list_digits = try python.tuple(a, &.{
        .{ .integer = 0 },
        .{ .array = digits },
        .{ .integer = 0 },
    });
    try expectDecimalText("101", try Decimal.validate(a, from_list_digits));

    const boolean_sign = try python.tuple(a, &.{
        .{ .bool = true },
        try python.tuple(a, &.{ .{ .integer = 1 }, .{ .integer = 2 } }),
        .{ .integer = -1 },
    });
    try expectDecimalText("-1.2", try Decimal.validate(a, boolean_sign));
    const boolean_exponent = try python.tuple(a, &.{
        .{ .integer = 0 },
        try python.tuple(a, &.{.{ .integer = 1 }}),
        .{ .bool = true },
    });
    try expectDecimalText("1E+1", try Decimal.validate(a, boolean_exponent));

    for ([_][]const u8{ "F", "n", "N" }) |kind| {
        const nonfinite = try python.tuple(a, &.{
            .{ .integer = 0 },
            coefficient,
            .{ .string = kind },
        });
        try std.testing.expectError(error.ParseValue, Decimal.validate(a, nonfinite));
    }
    var root_list: runtime.Array = .init(a);
    try root_list.appendSlice(&.{
        .{ .integer = 0 },
        coefficient,
        .{ .integer = -3 },
    });
    try std.testing.expectError(error.ParseValue, Decimal.validate(a, .{ .array = root_list }));
}

test "Decimal coercion transfers coefficient ownership across allocation failures" {
    const Fixture = struct {
        fn run(a: Allocator) !void {
            const checked = try Decimal.coerce(a, .{ .string = "12345678901234567890.12300" });
            defer checked.value.custom.deinit(a);
            try std.testing.expectEqual(@as(i64, -5), python.view(checked.value).?.decimal.exponent);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Fixture.run, .{});
}
