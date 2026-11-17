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

pub const Options = struct {
    allow_inf_nan: bool = false,
};

pub const Decimal = DecimalWithOptions(.{});

/// Recognize only this annotation family, without inspecting arbitrary types.
pub fn isDecimal(comptime T: type) bool {
    return T == Decimal or T == DecimalWithOptions(.{ .allow_inf_nan = true });
}

/// Configure Pydantic's annotation policy, not the physical Decimal class.
/// Every variant borrows or constructs the same genuine immutable runtime boxes.
pub fn DecimalWithOptions(comptime options: Options) type {
    return struct {
        const Self = @This();
        pub const dszig_decimal_options: Options = options;
        value: Value,

        /// Decode an already validated result without running field validation a
        /// second time. Nonfinite values can be stored even when the configured
        /// field validator would refuse them.
        pub fn fromValue(input: Value) error{ParseValue}!Self {
            const builtin = python.view(input) orelse return error.ParseValue;
            if (builtin != .decimal) return error.ParseValue;
            return .{ .value = input };
        }

        /// This view borrows the genuine immutable runtime value. Owning copies use
        /// the runtime Custom.clone protocol, not Python deepcopy identity.
        pub fn toRuntime(self: Self, _: Allocator) Allocator.Error!Value {
            return self.value;
        }

        pub fn coerce(allocator: Allocator, input: Value) runtime.CustomError!Self {
            return .{ .value = try validate(allocator, input) };
        }

        /// TypeAdapter(Decimal).validate_python with the configured nonfinite policy.
        /// Existing accepted Decimal instances are retained, including signaling NaNs
        /// without signaling or quieting them. Bool, bytes and ordinary Enum members
        /// are not numerical surrogates for this annotation.
        pub fn validate(allocator: Allocator, input: Value) runtime.CustomError!Value {
            return switch (input) {
                .custom => blk: {
                    if (input.custom.integerValue()) |integer| return validate(allocator, integer);
                    const builtin = python.view(input) orelse return error.ParseValue;
                    if (builtin == .decimal) {
                        if (!options.allow_inf_nan and builtin.decimal.kind != .finite) return error.ParseValue;
                        break :blk input;
                    }
                    if (builtin == .float) break :blk try fromFloatString(allocator, builtin.float);
                    if (builtin != .tuple) return error.ParseValue;
                    break :blk try fromTuple(allocator, input);
                },
                .string => |text| parseValidated(allocator, text),
                .integer => |number| blk: {
                    var buffer: [21]u8 = undefined;
                    const text = std.fmt.bufPrint(&buffer, "{d}", .{number}) catch unreachable;
                    break :blk parseValidated(allocator, text);
                },
                .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text))
                    parseValidated(allocator, text)
                else
                    fromFloatString(allocator, std.fmt.parseFloat(f64, text) catch return error.ParseValue),
                .float => |number| fromFloatString(allocator, number),
                else => error.ParseValue,
            };
        }

        /// TypeAdapter(Decimal).validate_python(..., strict=True). Reject other
        /// runtime kinds before attempting conversion, including invalid text that
        /// would otherwise signal InvalidOperation in the lax constructor.
        pub fn validateStrict(allocator: Allocator, input: Value) runtime.CustomError!Value {
            const builtin = python.view(input) orelse return error.ParseValue;
            if (builtin != .decimal) return error.ParseValue;
            return validate(allocator, input);
        }

        /// TypeAdapter(Decimal).validate_json with Pydantic's default lax policy.
        /// Three-part JSON constructor arrays are accepted only in this lax mode.
        pub fn validateJson(allocator: Allocator, text: []const u8) runtime.CustomError!Value {
            return validateJsonMode(allocator, text, false);
        }

        /// Strict JSON still accepts strings and numbers, unlike strict Python.
        /// Constructor arrays are refused before conversion can signal a condition.
        pub fn validateJsonStrict(allocator: Allocator, text: []const u8) runtime.CustomError!Value {
            return validateJsonMode(allocator, text, true);
        }

        fn validateJsonMode(allocator: Allocator, text: []const u8, comptime strict: bool) runtime.CustomError!Value {
            // The parser borrows text but owns decoded strings and constructor
            // arrays. Decimal conversion owns independent metadata in allocator.
            var scratch: std.heap.ArenaAllocator = .init(allocator);
            defer scratch.deinit();
            const input = @import("json_repair.zig").parseJson(scratch.allocator(), text) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseValue,
            };
            return switch (input) {
                .array => |array| if (!strict and array.items.len == 3)
                    fromTuple(allocator, input)
                else
                    error.ParseValue,
                .float => |number| fromJsonFloat(allocator, number),
                .number_string => |number| if (std.json.isNumberFormattedLikeAnInteger(number))
                    parseValidated(allocator, number)
                else
                    fromJsonFloat(allocator, std.fmt.parseFloat(f64, number) catch return error.ParseValue),
                else => validate(allocator, input),
            };
        }

        /// Pydantic 2.14.2's default schema generator adds no Decimal pattern.
        /// Its validation mode accepts a JSON number or a string for either policy.
        pub fn schemaValidation() []const u8 {
            return "{\"anyOf\": [{\"type\": \"number\"}, {\"type\": \"string\"}]}";
        }

        /// Decimal serialization uses str(value), including permitted nonfinite
        /// values. The pinned default generator returns exactly this string schema.
        pub fn schemaSerialization() []const u8 {
            return "{\"type\": \"string\"}";
        }

        fn fromTuple(allocator: Allocator, input: Value) runtime.CustomError!Value {
            const result = try python.decimalFromTuple(allocator, input);
            if (!options.allow_inf_nan and python.view(result).?.decimal.kind != .finite) {
                result.custom.deinit(allocator);
                return error.ParseValue;
            }
            return result;
        }

        /// JSON float coercion uses shortest plain decimal text, unlike Python's
        /// str(float) spelling. Nonfinite floats obey the same annotation policy.
        fn fromJsonFloat(allocator: Allocator, number: f64) runtime.CustomError!Value {
            if (!std.math.isFinite(number)) return fromFloatString(allocator, number);
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
            return parseValidated(allocator, writer.buffered());
        }

        fn parseValidated(allocator: Allocator, text: []const u8) runtime.CustomError!Value {
            // Pydantic calls Decimal(text) before applying its nonfinite policy.
            // A trapped syntax failure is refused; an untrapped one produces a fresh
            // quiet NaN and is accepted only when allow_inf_nan is enabled.
            const data = decimals.parse(allocator, text) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.ParseValue => {
                    @import("decimal_environment.zig").getContext().signal(.invalid_operation) catch return error.ParseValue;
                    if (options.allow_inf_nan)
                        return python.decimalFromMetadata(allocator, .{ .kind = .quiet_nan, .digits = "" });
                    return error.ParseValue;
                },
            };
            errdefer data.deinit(allocator);
            if (!options.allow_inf_nan and data.kind != .finite) return error.ParseValue;
            return python.decimalFromOwnedMetadata(allocator, data);
        }

        fn fromFloatString(allocator: Allocator, number: f64) runtime.CustomError!Value {
            if (!options.allow_inf_nan and !std.math.isFinite(number)) return error.ParseValue;
            // Pydantic uses Decimal(str(float)), not Decimal(float)'s exact binary
            // expansion. Every Python binary64 repr fits in this fixed buffer.
            var buffer: [64]u8 = undefined;
            var writer: std.Io.Writer = .fixed(&buffer);
            pyjson.writeFloat(&writer, number, .python) catch unreachable;
            return parseValidated(allocator, writer.buffered());
        }
    };
}

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

test "strict Decimal validation preserves genuine identity and refuses coercion without signaling" {
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const genuine = try python.decimal(a, "1.2300");
    const nonfinite = try python.decimal(a, "sNaN");
    local.context.clearFlags();
    local.context.flags.insert(.inexact);
    try std.testing.expect((try Decimal.validateStrict(a, genuine)).custom.ptr == genuine.custom.ptr);
    try std.testing.expectError(error.ParseValue, Decimal.validateStrict(a, nonfinite));
    for ([_]Value{
        .{ .string = "1.2300" },
        .{ .string = "not-a-decimal" },
        .{ .integer = 1 },
        .{ .number_string = "9007199254740993" },
        .{ .float = 1.25 },
        .{ .bool = true },
        .null,
    }) |input| {
        try std.testing.expectError(error.ParseValue, Decimal.validateStrict(a, input));
    }
    try std.testing.expectEqual(@as(usize, 1), local.context.flags.count());
    try std.testing.expect(local.context.flags.contains(.inexact));
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

test "Decimal annotation options expose exact validation and serialization schemas" {
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
    try std.testing.expect(Decimal == DecimalWithOptions(.{}));
    try std.testing.expect(isDecimal(Decimal));
    try std.testing.expect(isDecimal(Allow));
    try std.testing.expect(!Decimal.dszig_decimal_options.allow_inf_nan);
    try std.testing.expect(Allow.dszig_decimal_options.allow_inf_nan);
    inline for (.{ bool, i64, type, void, anyerror, ?Decimal, *const Decimal, []const Decimal, [1]Decimal, fn () void, enum { decimal }, struct { value: Value }, union { value: Value } }) |T| {
        try std.testing.expect(!isDecimal(T));
    }
    const validation = comptime Allow.schemaValidation();
    const serialization = comptime Allow.schemaSerialization();
    try std.testing.expectEqualStrings(
        "{\"anyOf\": [{\"type\": \"number\"}, {\"type\": \"string\"}]}",
        validation,
    );
    try std.testing.expectEqualStrings("{\"type\": \"string\"}", serialization);
    try std.testing.expectEqualStrings(validation, comptime Decimal.schemaValidation());
    try std.testing.expectEqualStrings(serialization, comptime Decimal.schemaSerialization());
}

test "configured Decimal retains nonfinite identity and signaling NaN metadata without signaling" {
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    local.context.traps = .initMany(&.{ .invalid_operation, .float_operation });
    local.context.clearFlags();
    local.context.flags.insert(.inexact);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ordinary = try python.decimal(a, "0");

    for ([_][]const u8{
        "NaN", "-NaN", "NaN123", "sNaN", "-sNaN", "sNaN42", "Infinity", "-Infinity", "-0.00", "1.2300",
    }) |text| {
        const genuine = try python.decimal(a, text);
        try std.testing.expect((try Allow.validate(a, genuine)).custom.ptr == genuine.custom.ptr);
        try std.testing.expect((try Allow.validateStrict(a, genuine)).custom.ptr == genuine.custom.ptr);
        const stored = try Allow.fromValue(genuine);
        try std.testing.expect((try stored.toRuntime(a)).custom.ptr == genuine.custom.ptr);
        const checked = try Allow.coerce(a, genuine);
        try std.testing.expect(checked.value.custom.ptr == genuine.custom.ptr);
        const decoded_default = try Decimal.fromValue(genuine);
        try std.testing.expect((try decoded_default.toRuntime(a)).custom.ptr == genuine.custom.ptr);
        const parsed = try Allow.validate(a, .{ .string = text });
        try expectDecimalText(text, parsed);
        try std.testing.expect(parsed.custom.vtable == ordinary.custom.vtable);
        if (python.view(genuine).?.decimal.kind != .finite) {
            try std.testing.expectError(error.ParseValue, Decimal.validate(a, genuine));
            try std.testing.expectError(error.ParseValue, Decimal.validateStrict(a, genuine));
        }
    }
    try std.testing.expectEqual(@as(usize, 1), local.context.flags.count());
    try std.testing.expect(local.context.flags.contains(.inexact));
}

test "configured Decimal preserves constructor syntax signals and untrapped quiet NaN results" {
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
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
            if (trapped) {
                try std.testing.expectError(error.ParseValue, Allow.validate(a, .{ .string = text }));
            } else {
                try expectDecimalText("NaN", try Allow.validate(a, .{ .string = text }));
            }
            try std.testing.expect(local.context.flags.contains(.invalid_operation));
            try std.testing.expect(!local.context.flags.contains(.float_operation));

            local.context.clearFlags();
            try std.testing.expectError(error.ParseValue, Decimal.validate(a, .{ .string = text }));
            try std.testing.expect(local.context.flags.contains(.invalid_operation));

            const json_text = try std.fmt.allocPrint(a, "\"{s}\"", .{text});
            local.context.clearFlags();
            if (trapped) {
                try std.testing.expectError(error.ParseValue, Allow.validateJson(a, json_text));
            } else {
                try expectDecimalText("NaN", try Allow.validateJson(a, json_text));
            }
            try std.testing.expect(local.context.flags.contains(.invalid_operation));
            try std.testing.expect(!local.context.flags.contains(.float_operation));

            local.context.clearFlags();
            try std.testing.expectError(error.ParseValue, Decimal.validateJson(a, json_text));
            try std.testing.expect(local.context.flags.contains(.invalid_operation));

            local.context.clearFlags();
            if (trapped) {
                try std.testing.expectError(error.ParseValue, Allow.validateJsonStrict(a, json_text));
            } else {
                try expectDecimalText("NaN", try Allow.validateJsonStrict(a, json_text));
            }
            try std.testing.expect(local.context.flags.contains(.invalid_operation));

            local.context.clearFlags();
            try std.testing.expectError(error.ParseValue, Decimal.validateJsonStrict(a, json_text));
            try std.testing.expect(local.context.flags.contains(.invalid_operation));

            local.context.clearFlags();
            try std.testing.expectError(error.ParseValue, Allow.validateStrict(a, .{ .string = text }));
            try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
        }
    }
}

test "configured Decimal tuple coercion retains nonfinite sign payload and trap behavior" {
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const coefficient = try python.tuple(a, &.{
        .{ .integer = 0 }, .{ .integer = 4 }, .{ .integer = 2 },
    });
    const cases = [_]struct { token: []const u8, expected: []const u8 }{
        .{ .token = "F", .expected = "-Infinity" },
        .{ .token = "n", .expected = "-NaN42" },
        .{ .token = "N", .expected = "-sNaN42" },
    };
    for ([_]bool{ true, false }) |trapped| {
        local.context.traps = if (trapped) .initOne(.invalid_operation) else .empty;
        for (cases) |case| {
            const input = try python.tuple(a, &.{
                .{ .integer = 1 }, coefficient, .{ .string = case.token },
            });
            local.context.clearFlags();
            try expectDecimalText(case.expected, try Allow.validate(a, input));
            try std.testing.expectError(error.ParseValue, Decimal.validate(a, input));
            try std.testing.expectError(error.ParseValue, Allow.validateStrict(a, input));
            try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
        }

        const outside_range = try python.tuple(a, &.{
            .{ .integer = 0 }, coefficient, .{ .integer = decimals.max_exponent + 1 },
        });
        local.context.clearFlags();
        if (trapped) {
            try std.testing.expectError(error.ParseValue, Allow.validate(a, outside_range));
        } else {
            try expectDecimalText("NaN", try Allow.validate(a, outside_range));
        }
        try std.testing.expect(local.context.flags.contains(.invalid_operation));
        local.context.clearFlags();
        try std.testing.expectError(error.ParseValue, Allow.validateStrict(a, outside_range));
        try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
    }
}

test "configured Decimal Python and JSON float coercion obey the nonfinite policy" {
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    local.context.traps = .initMany(&.{ .invalid_operation, .float_operation });
    local.context.clearFlags();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const floats = [_]struct { number: f64, expected: []const u8 }{
        .{ .number = std.math.inf(f64), .expected = "Infinity" },
        .{ .number = -std.math.inf(f64), .expected = "-Infinity" },
        .{ .number = std.math.nan(f64), .expected = "NaN" },
        .{ .number = @bitCast(@as(u64, 0xfff8000000000042)), .expected = "NaN" },
    };
    for (floats) |case| {
        try expectDecimalText(case.expected, try Allow.validate(a, .{ .float = case.number }));
        try std.testing.expectError(error.ParseValue, Decimal.validate(a, .{ .float = case.number }));
        try std.testing.expectError(error.ParseValue, Allow.validateStrict(a, .{ .float = case.number }));
    }
    const json_cases = [_]struct { text: []const u8, expected: []const u8 }{
        .{ .text = "NaN", .expected = "NaN" },
        .{ .text = "Infinity", .expected = "Infinity" },
        .{ .text = "-Infinity", .expected = "-Infinity" },
        .{ .text = "1e1000", .expected = "Infinity" },
        .{ .text = "-1e1000", .expected = "-Infinity" },
        .{ .text = "\"-NaN123\"", .expected = "-NaN123" },
        .{ .text = "\"-sNaN42\"", .expected = "-sNaN42" },
    };
    for (json_cases) |case| {
        try expectDecimalText(case.expected, try Allow.validateJson(a, case.text));
        try expectDecimalText(case.expected, try Allow.validateJsonStrict(a, case.text));
        try std.testing.expectError(error.ParseValue, Decimal.validateJson(a, case.text));
        try std.testing.expectError(error.ParseValue, Decimal.validateJsonStrict(a, case.text));
    }
    try expectDecimalText("Infinity", try Allow.validate(a, .{ .number_string = "1e1000" }));
    try expectDecimalText("1E+1000", try Allow.validateJson(a, "\"1e1000\""));
    try expectDecimalText("1", try Allow.validateJson(a, "1.0"));
    try expectDecimalText("1", try Allow.validateJsonStrict(a, "1.0"));
    try expectDecimalText("1.0", try Allow.validate(a, .{ .float = 1.0 }));
    try expectDecimalText("9007199254740993", try Allow.validateJson(a, "9007199254740993"));
    for ([_][]const u8{ "true", "null", "[]", "{}", "1 trailing" }) |text|
        try std.testing.expectError(error.ParseValue, Allow.validateJson(a, text));
    try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
}

test "Decimal annotations coerce actual integer subclasses exactly but not plain enums or strict inputs" {
    const enums = @import("enum.zig");
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
    const power309 = "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    const E = enums.IntEnum("DecimalIntegerSubclass", &.{
        .{ .name = "zero", .value = .{ .integer = 0 } },
        .{ .name = "one", .value = .{ .integer = 1 } },
        .{ .name = "three", .value = .{ .integer = 3 } },
        .{ .name = "negative", .value = .{ .integer = -1 } },
        .{ .name = "wide", .value = .{ .number_string = "9007199254740993" } },
        .{ .name = "beyond_u64", .value = .{ .number_string = "18446744073709551616" } },
        .{ .name = "huge", .value = .{ .number_string = "18446744073709551617" } },
        .{ .name = "power309", .value = .{ .number_string = power309 } },
    });
    const Plain = enums.Enum("DecimalPlainEnum", &.{.{ .name = "one", .value = .{ .integer = 1 } }});
    var local: @import("decimal_environment.zig").LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    local.context.traps = .initMany(&.{
        .invalid_operation, .float_operation, .division_by_zero, .overflow,
        .underflow,         .subnormal,       .inexact,          .rounded,
        .clamped,
    });
    local.context.clearFlags();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { name: []const u8, expected: []const u8 }{
        .{ .name = "zero", .expected = "0" },
        .{ .name = "one", .expected = "1" },
        .{ .name = "three", .expected = "3" },
        .{ .name = "negative", .expected = "-1" },
        .{ .name = "wide", .expected = "9007199254740993" },
        .{ .name = "beyond_u64", .expected = "18446744073709551616" },
        .{ .name = "huge", .expected = "18446744073709551617" },
        .{ .name = "power309", .expected = power309 },
    };
    for (cases) |case| {
        const member = try E.member(case.name);
        try expectDecimalText(case.expected, try Decimal.validate(a, member));
        try expectDecimalText(case.expected, try Allow.validate(a, member));
        try std.testing.expectError(error.ParseValue, Decimal.validateStrict(a, member));
        try std.testing.expectError(error.ParseValue, Allow.validateStrict(a, member));
    }
    for ([_]Value{ try Plain.member("one"), .{ .bool = true }, python.staticBytes("1") }) |input| {
        try std.testing.expectError(error.ParseValue, Decimal.validate(a, input));
        try std.testing.expectError(error.ParseValue, Allow.validate(a, input));
    }
    try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
}

test "configured Decimal owns nonfinite and untrapped syntax results across allocation failures" {
    const Fixture = struct {
        fn run(a: Allocator) !void {
            const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
            var local: @import("decimal_environment.zig").LocalContext = .{};
            local.enter(null);
            defer local.deinit();
            local.context.traps = .empty;
            for ([_][]const u8{ "-sNaN42", "Infinity", "not-a-decimal" }) |text| {
                const checked = try Allow.coerce(a, .{ .string = text });
                defer checked.value.custom.deinit(a);
                try std.testing.expect(python.view(checked.value).?.decimal.kind != .finite);
            }
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Fixture.run, .{});
}

test "Decimal JSON results own their metadata after parser storage is released across allocation failures" {
    const Fixture = struct {
        fn run(a: Allocator) !void {
            const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
            const nonfinite = try Allow.validateJson(a, "[1,[4,2],\"N\"]");
            defer nonfinite.custom.deinit(a);
            const signaling = python.view(nonfinite).?.decimal;
            try std.testing.expect(signaling.kind == .signaling_nan and signaling.negative);
            try std.testing.expectEqualStrings("42", signaling.digits);

            const finite = try Decimal.validateJson(a, "[0,[1,2],-1]");
            defer finite.custom.deinit(a);
            const data = python.view(finite).?.decimal;
            try std.testing.expectEqualStrings("12", data.digits);
            try std.testing.expectEqual(@as(i64, -1), data.exponent);

            const escaped = try Allow.validateJsonStrict(a, "\"\\u0073NaN42\"");
            defer escaped.custom.deinit(a);
            const escaped_data = python.view(escaped).?.decimal;
            try std.testing.expect(escaped_data.kind == .signaling_nan);
            try std.testing.expectEqualStrings("42", escaped_data.digits);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Fixture.run, .{});
}

test "Decimal JSON constructor arrays are lax-only with exact nonfinite metadata and traps" {
    const Allow = DecimalWithOptions(.{ .allow_inf_nan = true });
    const environment = @import("decimal_environment.zig");
    var local: environment.LocalContext = .{};
    local.enter(null);
    defer local.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { text: []const u8, expected: []const u8, finite: bool }{
        .{ .text = "[0,[1,2],-1]", .expected = "1.2", .finite = true },
        .{ .text = "[1,[0,4,2],-3]", .expected = "-0.042", .finite = true },
        .{ .text = "[1,[0,4,2],\"n\"]", .expected = "-NaN42", .finite = false },
        .{ .text = "[1,[0,4,2],\"N\"]", .expected = "-sNaN42", .finite = false },
        .{ .text = "[0,[],\"n\"]", .expected = "NaN", .finite = false },
        .{ .text = "[0,[1,2],\"F\"]", .expected = "Infinity", .finite = false },
    };
    const outside_range = try std.fmt.allocPrint(a, "[0,[4,2],{d}]", .{decimals.max_exponent + 1});
    for ([_]bool{ true, false }) |trapped| {
        local.context.traps = if (trapped) .initOne(.invalid_operation) else .empty;
        for (cases) |case| {
            local.context.clearFlags();
            try expectDecimalText(case.expected, try Allow.validateJson(a, case.text));
            if (case.finite) {
                try expectDecimalText(case.expected, try Decimal.validateJson(a, case.text));
            } else {
                try std.testing.expectError(error.ParseValue, Decimal.validateJson(a, case.text));
            }
            try std.testing.expectError(error.ParseValue, Allow.validateJsonStrict(a, case.text));
            try std.testing.expectError(error.ParseValue, Decimal.validateJsonStrict(a, case.text));
            try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
        }

        local.context.clearFlags();
        if (trapped) {
            try std.testing.expectError(error.ParseValue, Allow.validateJson(a, outside_range));
        } else {
            try expectDecimalText("NaN", try Allow.validateJson(a, outside_range));
        }
        try std.testing.expect(local.context.flags.contains(.invalid_operation));

        local.context.clearFlags();
        try std.testing.expectError(error.ParseValue, Decimal.validateJson(a, outside_range));
        try std.testing.expect(local.context.flags.contains(.invalid_operation));

        local.context.clearFlags();
        try std.testing.expectError(error.ParseValue, Allow.validateJsonStrict(a, outside_range));
        try std.testing.expectError(error.ParseValue, Decimal.validateJsonStrict(a, outside_range));
        try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());

        for ([_][]const u8{ "[]", "[0,[1]]", "[2,[1],0]", "[0,[10],0]" }) |text| {
            try std.testing.expectError(error.ParseValue, Allow.validateJson(a, text));
            try std.testing.expectError(error.ParseValue, Allow.validateJsonStrict(a, text));
            try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
        }
    }
}
