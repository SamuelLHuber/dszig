//! Explicit Python Enum classes, distinct from Zig enums (DSPy Literal).
//! Class-owned members are immutable singleton boxes. Runtime-created Python
//! classes, Enum mixins and arbitrary mutable member values are not represented
//! by these comptime factories.
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
    try std.testing.expect(!value.pyEqual(left, right));
    try std.testing.expect(!value.pyEqual(left, .{ .string = "red" }));
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

pub fn EnumWithIdentity(comptime name: []const u8, comptime entries: []const Entry, comptime Identity: type) type {
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
        index: usize,

        fn canonical(comptime index: usize) usize {
            return comptime resolved: {
                for (entries[0..index], 0..) |previous, i| if (value.pyEqual(previous.value, entries[index].value)) break :resolved canonical(i);
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
                if (value.pyEqual(entry.value, input)) return boxed(index);
            }
            if (input == .string) return member(input.string);
            return error.ParseValue;
        }

        /// Pydantic validation accepts existing members or their values, not
        /// arbitrary member names (DSPy's direct enum parser adds name lookup).
        pub fn validate(_: Allocator, input: Value) runtime.CustomError!Value {
            if (input == .custom and input.custom.get(Self) != null) return input;
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
                if (value.pyEqual(entry.value, input)) return boxed(index);
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
            .model_name = name,
            .hashable = true,
            .enumeration = inspect,
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

        fn instance(ptr: *const anyopaque) *const Self { return @ptrCast(@alignCast(ptr)); }
        fn inspect(ptr: *const anyopaque) runtime.EnumValue {
            const entry = entries[instance(ptr).index];
            return .{ .class_identity = @typeName(Self), .class_name = name, .member_name = entry.name, .value = entry.value };
        }
        fn rawValue(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return boxed(instance(ptr).index);
        }
        fn serialized(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
            return value.serializedValueDepth(arena, entries[instance(ptr).index].value, depth + 1);
        }
        fn writeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            try writer.print("{s}.{s}", .{ name, entries[instance(ptr).index].name });
        }
        fn writeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
            try writer.writeByte('<');
            try writeStr(writer, ptr);
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
        fn equal(left: *const anyopaque, right: *const anyopaque) bool {
            return instance(left).index == instance(right).index;
        }
    };
}
