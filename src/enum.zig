//! Explicit Python Enum classes, distinct from Zig enums (DSPy Literal).
//! Class-owned members are immutable singleton boxes. Runtime-created Python
//! classes, other Enum mixins and arbitrary mutable member values are not
//! represented by these comptime factories.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const value = @import("value.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Writer = std.Io.Writer;

pub const Entry = struct { name: []const u8, value: Value };

fn jsonType(v: Value) ?[]const u8 {
    return switch (v) {
        .bool => "boolean",
        .integer, .number_string => "integer",
        .float => "number",
        .string => "string",
        .array => "array",
        .object => "object",
        .custom => |c| if (c.vtable.enumeration) |inspect|
            jsonType(inspect(c.ptr).value)
        else if (@import("python_value.zig").view(v)) |builtin|
            if (builtin == .bytes) "string" else null
        else
            null,
        else => null,
    };
}

pub fn isEnum(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "dszig_enum_entries");
}

/// Repeated factories with identical arguments denote the same native class.
/// Supply a distinct identity tag when modeling independent Python classes.
pub fn Enum(comptime name: []const u8, comptime entries: []const Entry) type {
    return EnumWithIdentity(name, entries, void);
}

test "Enum resolution preserves value-before-name priority and canonical aliases" {
    const Collision = Enum("Collision", &.{
        .{ .name = "first", .value = .{ .string = "second" } },
        .{ .name = "second", .value = .{ .string = "value2" } },
        .{ .name = "alias", .value = .{ .string = "second" } },
    });
    const a = std.testing.allocator;
    const first = try Collision.member("first");
    const alias = try Collision.member("alias");
    const resolved = try Collision.parseValue(a, .{ .string = "second" });
    try std.testing.expect(first.custom.ptr == alias.custom.ptr);
    try std.testing.expect(first.custom.ptr == resolved.custom.ptr);
    try std.testing.expectError(error.ParseValue, Collision.validate(a, .{ .string = "first" }));
    try std.testing.expectError(error.ParseValue, Collision.parseValue(a, first));
    const validated = try Collision.validate(a, first);
    try std.testing.expect(validated.custom.ptr == first.custom.ptr);
    const cloned = try value.cloneValue(a, alias);
    defer cloned.custom.deinit(a);
    try std.testing.expect(cloned.custom.ptr == first.custom.ptr);
}

test "Enum schemas classify genuine bytes and recursively wrapped member values" {
    const binary = @import("python_value.zig").staticBytes;
    const Binary = Enum("Bytes", &.{
        .{ .name = "one", .value = binary("one") },
        .{ .name = "two", .value = binary("two") },
    });
    const MixedText = Enum("MixedText", &.{
        .{ .name = "binary", .value = binary("one") },
        .{ .name = "text", .value = .{ .string = "one" } },
    });
    const Wrapped = Enum("Wrapped", &.{.{ .name = "member", .value = comptime Binary.member("one") catch unreachable }});
    const schema = @import("types.zig").schema;
    const a = std.testing.allocator;
    try std.testing.expectEqualStrings("{\"type\": \"string\", \"enum\": [\"one\", \"two\"], \"title\": \"Bytes\"}", schema(Binary));
    try std.testing.expectEqualStrings("{\"type\": \"string\", \"enum\": [\"one\", \"one\"], \"title\": \"MixedText\"}", schema(MixedText));
    try std.testing.expectEqualStrings("{\"type\": \"string\", \"enum\": [\"one\"], \"title\": \"Wrapped\"}", schema(Wrapped));
    const member = try Binary.validate(a, comptime binary("one"));
    try std.testing.expect(member.custom.ptr == (try Binary.member("one")).custom.ptr);
    try std.testing.expectError(error.ParseValue, Binary.validate(a, .{ .string = "one" }));
}

test "Independent same-name Enum classes cannot validate or compare as one class" {
    const entries = &[_]Entry{.{ .name = "red", .value = .{ .string = "red" } }};
    const Left = EnumWithIdentity("Color", entries, struct {});
    const Right = EnumWithIdentity("Color", entries, struct {});
    const left = try Left.member("red");
    const right = try Right.member("red");
    try std.testing.expect(!try value.pyEqual(left, right));
    try std.testing.expect(!try value.pyEqual(left, .{ .string = "red" }));
    try std.testing.expectError(error.ParseValue, Left.validate(std.testing.allocator, right));
}

test "Pydantic Enum fast integer lookup overflow differs from DSPy value resolution" {
    const big: Value = .{ .number_string = "100000000000000000000000000000000000000000000000000" };
    const Mixed = Enum("Mixed", &.{
        .{ .name = "zero", .value = .{ .integer = 0 } },
        .{ .name = "big", .value = big },
    });
    const OnlyBig = Enum("OnlyBig", &.{.{ .name = "big", .value = big }});
    const a = std.testing.allocator;
    try std.testing.expectError(error.ParseValue, Mixed.validate(a, big));
    const resolved = try Mixed.parseValue(a, big);
    try std.testing.expect(resolved.custom.ptr == (try Mixed.member("big")).custom.ptr);
    const validated = try OnlyBig.validate(a, big);
    try std.testing.expect(validated.custom.ptr == (try OnlyBig.member("big")).custom.ptr);
}

test "long integer enum values retain distinct members and exact aliases" {
    const zeros: [1000]u8 = @splat('0');
    const first = "1" ++ zeros;
    const next = "1" ++ zeros[0..999].* ++ "1";
    const Long = Enum("Long", &.{
        .{ .name = "zero", .value = .{ .integer = 0 } },
        .{ .name = "first", .value = .{ .number_string = first } },
        .{ .name = "next", .value = .{ .number_string = next } },
        .{ .name = "alias", .value = .{ .number_string = first } },
    });
    const member = try Long.member("first");
    const alias = try Long.member("alias");
    const other = try Long.member("next");
    try std.testing.expect(member.custom.ptr == alias.custom.ptr);
    try std.testing.expect(member.custom.ptr != other.custom.ptr);
    try std.testing.expect(std.mem.eql(u8, first, Long.dszig_enum_values[3].number_string));
    const parsed = try Long.parseValue(std.testing.allocator, .{ .number_string = next });
    try std.testing.expect(parsed.custom.ptr == other.custom.ptr);
}

test "IntEnum exact numeric equality does not erase nominal validation or alias identity" {
    const entries = &[_]Entry{
        .{ .name = "zero", .value = .{ .integer = 0 } },
        .{ .name = "one", .value = .{ .integer = 1 } },
        .{ .name = "alias", .value = .{ .integer = 1 } },
        .{ .name = "wide", .value = .{ .integer = 9007199254740993 } },
        .{ .name = "beyond", .value = .{ .number_string = "18446744073709551616" } },
    };
    const Left = IntEnumWithIdentity("Number", entries, struct {});
    const Right = IntEnumWithIdentity("Number", entries, struct {});
    const left = try Left.member("one");
    const right = try Right.member("one");
    const wide = try Left.member("wide");
    const beyond = try Left.member("beyond");
    try std.testing.expect(try value.pyEqual(left, right));
    try std.testing.expect(!left.custom.sameType(right.custom));
    try std.testing.expect((try Left.member("alias")).custom.ptr == left.custom.ptr);
    try std.testing.expect(!((try Left.member("zero")).custom.truthy()));
    try std.testing.expectError(error.ParseValue, Left.validateStrict(std.testing.allocator, right));
    try std.testing.expect((try Left.validate(std.testing.allocator, right)).custom.ptr == left.custom.ptr);
    try std.testing.expect(!try value.pyEqual(wide, .{ .float = 9007199254740992.0 }));
    try std.testing.expect(try value.pyEqual(beyond, .{ .number_string = "18446744073709551616" }));
    try std.testing.expectEqual(try @import("python_value.zig").pythonHash(left, 1), try @import("python_value.zig").pythonHash(.{ .integer = 1 }, 1));
    const left_metadata = left.custom.vtable.enumeration.?(left.custom.ptr);
    const right_metadata = right.custom.vtable.enumeration.?(right.custom.ptr);
    try std.testing.expect(left_metadata.class_identity != right_metadata.class_identity);
}

test "IntEnum declaration coercion truncates floats and canonicalizes Unicode integer strings" {
    const Converted = IntEnum("Converted", &.{
        .{ .name = "positive", .value = .{ .float = 1.9 } },
        .{ .name = "negative", .value = .{ .float = -1.9 } },
        .{ .name = "alias", .value = .{ .string = " +００１ " } },
        .{ .name = "zero", .value = .{ .float = -0.0 } },
        .{ .name = "wide", .value = .{ .string = "\u{2002}٩_٠٠٧_١٩٩_٢٥٤_٧٤٠_٩٩٣\u{2002}" } },
        .{ .name = "float_wide", .value = .{ .float = 0x1p64 } },
    });
    const positive = try Converted.member("positive");
    try std.testing.expectEqual(@as(i64, 1), positive.custom.integerValue().?.integer);
    try std.testing.expectEqual(@as(i64, -1), (try Converted.member("negative")).custom.integerValue().?.integer);
    try std.testing.expectEqual(@as(i64, 0), (try Converted.member("zero")).custom.integerValue().?.integer);
    try std.testing.expect((try Converted.member("alias")).custom.ptr == positive.custom.ptr);
    const wide = try Converted.member("wide");
    try std.testing.expect(try value.pyEqual(wide, .{ .integer = 9007199254740993 }));
    try std.testing.expect(!try value.pyEqual(wide, .{ .float = 9007199254740992.0 }));
    try std.testing.expectEqualStrings("18446744073709551616", (try Converted.member("float_wide")).custom.integerValue().?.number_string);
}

pub fn EnumWithIdentity(comptime name: []const u8, comptime entries: []const Entry, comptime Identity: type) type {
    return EnumClass(name, entries, Identity, false);
}

/// Genuine immutable integer subclasses with class-owned singleton members.
/// Declaration-time builtin floats truncate toward zero; decimal text/bytes,
/// bools and integer subclasses normalize before canonical alias resolution.
pub fn IntEnum(comptime name: []const u8, comptime entries: []const Entry) type {
    return IntEnumWithIdentity(name, entries, void);
}

pub fn IntEnumWithIdentity(comptime name: []const u8, comptime entries: []const Entry, comptime Identity: type) type {
    const normalized = comptime blk: {
        @setEvalBranchQuota(1_000_000);
        var result: [entries.len]Entry = undefined;
        for (entries, 0..) |entry, index| {
            const input = if (entry.value == .custom)
                declarationCustom(entry.value)
            else
                entry.value;
            const integer: Value = switch (input) {
                .bool => |boolean| .{ .integer = @intFromBool(boolean) },
                .integer => input,
                .number_string => |text| number: {
                    if (!value.isJsonNumber(text) or !std.json.isNumberFormattedLikeAnInteger(text))
                        @compileError("dszig: IntEnum values must be exact integers");
                    break :number Value.parseFromNumberSlice(text);
                },
                .float => |number| declarationFloat(number),
                .string => |text| declarationString(text),
                else => @compileError("dszig: IntEnum declaration does not support this integer conversion"),
            };
            result[index] = .{ .name = entry.name, .value = integer };
        }
        break :blk result;
    };
    return EnumClass(name, &normalized, Identity, true);
}

fn declarationCustom(comptime input: Value) Value {
    if (input.custom.integerValue()) |integer| return integer;
    const builtin = @import("python_value.zig").view(input) orelse
        @compileError("dszig: IntEnum values must support builtin integer conversion");
    if (builtin != .bytes)
        @compileError("dszig: IntEnum declaration does not support this integer conversion");
    // int(bytes) accepts ASCII digits/whitespace, not str's Unicode digits or
    // Unicode whitespace. Reject non-ASCII before sharing the decimal parser.
    for (builtin.bytes) |byte| {
        if (byte >= 0x80) @compileError("dszig: invalid IntEnum integer bytes");
    }
    return declarationString(builtin.bytes);
}

test "IntEnum byte declarations canonicalize ASCII values without losing wide integers" {
    const binary = @import("python_value.zig").staticBytes;
    const Converted = IntEnum("ByteConverted", &.{
        .{ .name = "two", .value = binary(" \t+0_2\r\n") },
        .{ .name = "alias", .value = .{ .integer = 2 } },
        .{ .name = "negative", .value = binary("-3") },
        .{ .name = "wide", .value = binary("18446744073709551616") },
    });
    const two = try Converted.member("two");
    const alias = try Converted.member("alias");
    try std.testing.expect(two.custom.ptr == alias.custom.ptr);
    try std.testing.expectEqual(@as(i64, 2), two.custom.integerValue().?.integer);
    try std.testing.expectEqual(@as(i64, -3), (try Converted.member("negative")).custom.integerValue().?.integer);
    try std.testing.expectEqualStrings("18446744073709551616", (try Converted.member("wide")).custom.integerValue().?.number_string);
}

fn declarationFloat(comptime number: f64) Value {
    if (!std.math.isFinite(number))
        @compileError("dszig: IntEnum cannot convert a nonfinite float to an integer");
    const bits: u64 = @bitCast(number);
    const exponent = @as(i32, @intCast((bits >> 52) & 0x7ff)) - 1023;
    if (exponent < 0) return .{ .integer = 0 };
    const significand = (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000;
    const magnitude: comptime_int = if (exponent >= 52)
        @as(comptime_int, significand) << (exponent - 52)
    else
        significand >> (52 - exponent);
    const integer = if (bits >> 63 != 0) -magnitude else magnitude;
    return Value.parseFromNumberSlice(std.fmt.comptimePrint("{d}", .{integer}));
}

fn declarationString(comptime text: []const u8) Value {
    const unicode = @import("unicode.zig");
    // CPython's integer parser excludes the ASCII information separators
    // U+001C..U+001F even though str.isspace() includes them.
    const source = unicode.strip(text, true);
    if (source.len == 0) @compileError("dszig: invalid IntEnum integer string");
    const negative = source[0] == '-';
    const start: usize = @intFromBool(negative or source[0] == '+');
    var buffer: [text.len]u8 = undefined;
    var length: usize = 0;
    var offset = start;
    var previous_digit = false;
    while (offset < source.len) {
        const width = std.unicode.utf8ByteSequenceLength(source[offset]) catch
            @compileError("dszig: invalid IntEnum integer string");
        if (width > source.len - offset)
            @compileError("dszig: invalid IntEnum integer string");
        const cp = std.unicode.utf8Decode(source[offset..][0..width]) catch
            @compileError("dszig: invalid IntEnum integer string");
        offset += width;
        if (cp == '_') {
            if (!previous_digit or offset == source.len)
                @compileError("dszig: invalid IntEnum integer separator");
            previous_digit = false;
            continue;
        }
        const digit = unicode.decimalDigit(cp) orelse
            @compileError("dszig: invalid IntEnum integer digit");
        buffer[length] = '0' + digit;
        length += 1;
        previous_digit = true;
    }
    if (length == 0) @compileError("dszig: invalid IntEnum integer string");
    if (length > 4300)
        @compileError("dszig: IntEnum integer string exceeds CPython's default 4300-digit limit");
    var first: usize = 0;
    while (first < length and buffer[first] == '0') first += 1;
    if (first == length) return .{ .integer = 0 };
    const frozen = buffer;
    const magnitude = frozen[first..length];
    return Value.parseFromNumberSlice(if (negative) "-" ++ magnitude else magnitude);
}

fn EnumClass(comptime name: []const u8, comptime entries: []const Entry, comptime Identity: type, comptime integer: bool) type {
    comptime {
        for (entries, 0..) |entry, index| {
            if (entry.name.len == 0) @compileError("dszig: enum member name must not be empty");
            for (entries[0..index]) |previous| if (std.mem.eql(u8, previous.name, entry.name))
                @compileError("dszig: duplicate enum member name " ++ entry.name);
        }
    }
    return struct {
        const Self = @This();
        pub const model_name = name;
        pub const class_identity = Identity;
        pub const dszig_enum_entries = entries;
        pub const dszig_int_enum = integer;
        index: usize,

        fn canonical(comptime index: usize) usize {
            return comptime resolved: {
                // Lazy enum tables and member lookups evaluate independently
                // of the caller's quota, including exact long-integer equality.
                @setEvalBranchQuota(1_000_000);
                for (entries[0..index], 0..) |previous, i| {
                    const same = value.pyEqual(previous.value, entries[index].value) catch |err|
                        @compileError("dszig: enum alias equality failed: " ++ @errorName(err));
                    if (same) break :resolved canonical(i);
                }
                break :resolved index;
            };
        }

        pub const dszig_enum_values = blk: {
            var result: [entries.len]Value = undefined;
            for (entries, 0..) |_, i| result[i] = entries[canonical(i)].value;
            break :blk result;
        };

        pub const dszig_enum_json_type = blk: {
            if (dszig_enum_values.len == 0) break :blk null;
            const first = jsonType(dszig_enum_values[0]) orelse break :blk null;
            for (dszig_enum_values) |v| {
                const kind = jsonType(v) orelse break :blk null;
                if (!std.mem.eql(u8, first, kind)) break :blk null;
            }
            break :blk @as(?[]const u8, first);
        };

        const members = blk: {
            var result: [entries.len]Self = undefined;
            for (entries, 0..) |_, i| result[i] = .{ .index = canonical(i) };
            break :blk result;
        };

        fn boxed(index: usize) Value {
            return .{ .custom = .{ .ptr = &members[index], .vtable = &vtable } };
        }

        pub fn member(member_name: []const u8) error{ParseValue}!Value {
            inline for (entries, 0..) |entry, index| if (std.mem.eql(u8, member_name, entry.name)) return boxed(canonical(index));
            return error.ParseValue;
        }

        /// DSPy checks member values before names, without reparsing strings.
        pub fn parseValue(_: Allocator, input: Value) runtime.CustomError!Value {
            inline for (entries, 0..) |entry, index| {
                if (comptime canonical(index) != index) continue;
                if (try value.pyEqual(entry.value, input)) return boxed(index);
            }
            if (input == .string) return member(input.string);
            return error.ParseValue;
        }

        /// Pydantic validation accepts existing members or their values, not
        /// arbitrary member names (DSPy's direct enum parser adds name lookup).
        pub fn validate(arena: Allocator, input: Value) runtime.CustomError!Value {
            if (input == .custom and input.custom.get(Self) != null) return input;
            if (integer) return validateInteger(arena, input, false);
            // Pydantic core 2.50.2 LiteralLookup::validate converts exact
            // integer inputs to i64 before falling back to Python equality
            // whenever this enum has a fast i64 bucket. Preserve its overflow
            // refusal, even if a later arbitrary-width member would match.
            if (input == .number_string and std.json.isNumberFormattedLikeAnInteger(input.number_string)) {
                const has_i64 = comptime blk: {
                    for (dszig_enum_values) |v| {
                        if (v == .integer) break :blk true;
                        if (v == .number_string and std.json.isNumberFormattedLikeAnInteger(v.number_string)) {
                            _ = std.fmt.parseInt(i64, v.number_string, 10) catch continue;
                            break :blk true;
                        }
                    }
                    break :blk false;
                };
                if (has_i64) _ = std.fmt.parseInt(i64, input.number_string, 10) catch return error.ParseValue;
            }
            inline for (entries, 0..) |entry, index| {
                if (comptime canonical(index) != index) continue;
                if (try value.pyEqual(entry.value, input)) return boxed(index);
            }
            return error.ParseValue;
        }

        pub fn validateStrict(_: Allocator, input: Value) runtime.CustomError!Value {
            if (input == .custom and input.custom.get(Self) != null) return input;
            return error.ParseValue;
        }

        /// JSON strictness is distinct from Python's instance-only strict
        /// mode: the final Enum constructor still uses numeric equality.
        pub fn validateJson(arena: Allocator, text: []const u8) runtime.CustomError!Value {
            const input = try parseJson(arena, text);
            return validate(arena, input);
        }

        pub fn validateJsonStrict(arena: Allocator, text: []const u8) runtime.CustomError!Value {
            const input = try parseJson(arena, text);
            if (integer) return validateInteger(arena, input, true);
            return validate(arena, input);
        }

        fn parseJson(arena: Allocator, text: []const u8) runtime.CustomError!Value {
            return @import("json_repair.zig").parseJson(arena, text) catch |err| switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseValue,
            };
        }

        fn validateInteger(arena: Allocator, input: Value, comptime strict: bool) runtime.CustomError!Value {
            // Pydantic's IntEnum lookup coerces to int, then extracts i64,
            // but only when this class has an i64-valued lookup bucket.
            // Conversion errors fall back to Enum(value); extraction overflow
            // does not. The fallback also preserves Decimal sNaN signaling.
            const has_i64 = comptime blk: {
                for (dszig_enum_values) |v| if (v == .integer) break :blk true;
                break :blk false;
            };
            if (has_i64) {
                if (@import("python_value.zig").view(input)) |builtin| {
                    if (builtin == .decimal and builtin.decimal.kind == .finite) {
                        // Reject out-of-range Decimal values without expanding
                        // an arbitrarily large positive exponent into storage.
                        const lower = value.numericOrder(input, .{ .integer = std.math.minInt(i64) }) catch return error.ParseValue;
                        const upper = value.numericOrder(input, .{ .integer = std.math.maxInt(i64) }) catch return error.ParseValue;
                        if (lower == .lt or upper == .gt) return error.ParseValue;
                    }
                }
                var scratch: std.heap.ArenaAllocator = .init(arena);
                defer scratch.deinit();
                const parsed: ?Value = if (strict)
                    if (input == .integer or input == .number_string) input else null
                else
                    value.coerceInteger(scratch.allocator(), input) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.ParseValue => null,
                    };
                if (parsed) |number| {
                    // The lax int validator can return an Enum's actual value
                    // without revalidation. Its subsequent PyInt extraction
                    // still accepts bool and genuine integer subclasses.
                    const base = if (number == .custom) number.custom.integerValue() orelse number else number;
                    const small: ?i64 = switch (base) {
                        .bool => |candidate| @intFromBool(candidate),
                        .integer => |candidate| candidate,
                        .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text))
                            std.fmt.parseInt(i64, text, 10) catch return error.ParseValue
                        else
                            null,
                        else => null,
                    };
                    if (small) |candidate| {
                        inline for (entries, 0..) |entry, index| {
                            if (comptime canonical(index) != index) continue;
                            if (try value.pyEqual(entry.value, .{ .integer = candidate })) return boxed(index);
                        }
                    }
                }
            }
            inline for (entries, 0..) |entry, index| {
                if (comptime canonical(index) != index) continue;
                if (try value.pyEqual(entry.value, input)) return boxed(index);
            }
            return error.ParseValue;
        }

        pub fn fromValue(input: Value) error{ParseValue}!Self {
            if (input != .custom) return error.ParseValue;
            return (input.custom.get(Self) orelse return error.ParseValue).*;
        }

        pub fn toRuntime(self: Self, _: Allocator) Allocator.Error!Value {
            if (self.index >= members.len) unreachable;
            return boxed(members[self.index].index);
        }

        const vtable: runtime.CustomVTable = .{
            .type_name = @typeName(Self),
            .type_identity = runtime.typeIdentity(Self),
            .model_name = name,
            .hashable = true,
            .enumeration = inspect,
            .integer_value = if (integer) integerValue else null,
            .float_value = if (integer) floatValue else null,
            .raw_value = rawValue,
            .serialized = serialized,
            .write_str = writeStr,
            .write_repr = writeRepr,
            .write_json = writeJson,
            .write_serialized_json = writeSerializedJson,
            .clone = clone,
            .deinit = deinit,
            .eql = equal,
        };

        fn instance(ptr: *const anyopaque) *const Self {
            return @ptrCast(@alignCast(ptr));
        }
        fn inspect(ptr: *const anyopaque) runtime.EnumValue {
            const entry = entries[instance(ptr).index];
            return .{ .class_identity = runtime.typeIdentity(Self), .class_name = name, .member_name = entry.name, .value = entry.value };
        }
        fn integerValue(ptr: *const anyopaque) Value {
            return entries[instance(ptr).index].value;
        }
        fn floatValue(ptr: *const anyopaque) runtime.CustomError!f64 {
            return value.numericFloat(integerValue(ptr)) catch error.ParseValue;
        }
        fn rawValue(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return boxed(instance(ptr).index);
        }
        fn serialized(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
            return value.serializedValueDepth(arena, entries[instance(ptr).index].value, depth + 1);
        }
        fn writeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            if (integer) return pyjson.writeStr(writer, integerValue(ptr));
            try writer.print("{s}.{s}", .{ name, entries[instance(ptr).index].name });
        }
        fn writeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
            try writer.writeByte('<');
            try writer.print("{s}.{s}", .{ name, entries[instance(ptr).index].name });
            try writer.writeAll(": ");
            try pyjson.writeRepr(writer, entries[instance(ptr).index].value, depth + 1);
            try writer.writeByte('>');
        }
        fn writeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
            try writer.write(entries[instance(ptr).index].value);
        }
        fn writeSerializedJson(writer: *Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            try pyjson.writeJsonProjection(writer, entries[instance(ptr).index].value, style, @intCast(depth + 1));
        }
        fn clone(_: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
            // Python Enum members are immutable class-owned singletons;
            // deepcopy preserves the member, including its value's identity.
            var result = boxed(instance(ptr).index).custom;
            result.owns_data = true;
            return result;
        }
        fn deinit(_: Allocator, _: *const anyopaque, _: bool) void {}
        fn equal(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
            return instance(left).index == instance(right).index;
        }
    };
}
