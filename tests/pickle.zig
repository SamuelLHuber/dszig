//! `pickle.zig` against DSPy 3.4.1: `Hasher.hash(tuple(demos))` and the trace
//! `BootstrapFewShot` keeps, for the cases in tools/dspy_reference/gen_pickle.py.

const std = @import("std");
const dszig = @import("dszig");
const testing = std.testing;

const fixture = @embedFile("fixtures/pickle.json");

test "multi-hop trace pick matches DSPy's pickle hash case by case" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, fixture, .{});
    const cases = root.object.get("cases").?.array.items;
    try testing.expectEqual(@as(usize, 25), cases.len);
    for (cases) |case| {
        const items = case.object.get("demos").?.array.items;
        const demos = try arena.alloc(dszig.value.Record, items.len);
        for (demos, items) |*d, item| {
            d.* = .empty;
            try d.put(arena, "augmented", .{ .bool = true });
            var it = item.object.iterator();
            while (it.next()) |e| try d.put(arena, e.key_ptr.*, e.value_ptr.*);
        }
        const hex = try dszig.pickle.hashDemos(testing.allocator, demos);
        try testing.expectEqualStrings(case.object.get("sha256").?.string, &hex);
        const want: usize = @intCast(case.object.get("pick").?.integer);
        try testing.expectEqual(want, try dszig.pickle.pickTrace(testing.allocator, demos));
    }
}
