//! Width-independent storage for a Python int annotation's validated result.
//! Pydantic's lax Enum path returns the member's value without revalidating its
//! type, so an int-annotated result can genuinely be a string/list/None. Keep
//! that result intact rather than silently imposing an extra native validator.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const values = @import("value.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;

test "Integer annotation preserves Pydantic's unvalidated Enum member value" {
    const E = @import("enum.zig").Enum("Unusual", &.{
        .{ .name = "text", .value = .{ .string = "red" } },
        .{ .name = "nothing", .value = .null },
        .{ .name = "fraction", .value = .{ .float = 2.5 } },
    });
    const a = std.testing.allocator;
    const text = try Integer.coerce(a, try E.member("text"));
    try std.testing.expect(text.value == .string);
    try std.testing.expectEqualStrings("red", text.value.string);
    try std.testing.expect((try Integer.coerce(a, try E.member("nothing"))).value == .null);
    const fraction = try Integer.coerce(a, try E.member("fraction"));
    try std.testing.expect(fraction.value == .float);
    try std.testing.expectEqual(@as(f64, 2.5), fraction.value.float);
    const decoded = try values.fromValue(a, Integer, text.value);
    try std.testing.expectEqualStrings("red", decoded.value.string);
}

test "Integer normalization retains decimal precision beyond native integer widths" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const result = try Integer.coerce(arena.allocator(), .{
        .string = " -000_12345678901234567890123456789012345678901234567890.00 ",
    });
    try std.testing.expect(result.value == .number_string);
    try std.testing.expectEqualStrings("-12345678901234567890123456789012345678901234567890", result.value.number_string);
}

pub const Integer = struct {
    value: Value,

    pub fn init(number: i64) Integer {
        return .{ .value = .{ .integer = number } };
    }

    /// Pydantic-compatible coercion, including its actual Enum-value bypass.
    /// Large decimal data is borrowed unless normalization requires allocation.
    pub fn coerce(arena: Allocator, input: Value) values.ParseError!Integer {
        return .{ .value = try values.coerceInteger(arena, input) };
    }

    /// A typed decoder receives an already validated field result. This is
    /// storage, not a second validation pass that would reject Source results.
    pub fn fromValue(input: Value) Integer {
        return .{ .value = input };
    }

    pub fn toRuntime(self: Integer, _: Allocator) Allocator.Error!Value {
        return self.value;
    }
};
