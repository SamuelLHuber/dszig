//! Positional multi-trace selections recorded by the actual DSPy 3.5.0
//! BootstrapFewShot optimizer (tools/dspy_reference/gen_trace_selection.py).
//! The generator runs repeated predictors, rather than duplicating the RNG.

const std = @import("std");
const dszig = @import("dszig");
const testing = std.testing;

const fixture_json = @embedFile("fixtures/trace_selection.json");

test "BootstrapFewShot positional trace picks match actual DSPy 3.5" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = (try std.json.parseFromSliceLeaky(dszig.Value, arena, fixture_json, .{})).object;
    try testing.expectEqualStrings("3.5.0", fixture.get("dspy").?.string);
    const cases = fixture.get("cases").?.array.items;
    try testing.expectEqual(@as(usize, 144), cases.len);
    for (cases) |case| {
        const row = case.object;
        const predictor_name = row.get("predictor_name").?.string;
        const prior_trace_count: usize = @intCast(row.get("prior_trace_count").?.integer);
        const trace_count: usize = @intCast(row.get("trace_count").?.integer);
        const expected: usize = @intCast(row.get("pick").?.integer);
        const selected = try dszig.optimize.tracePick(arena, predictor_name, prior_trace_count, trace_count);
        try testing.expectEqual(expected, selected);
        try testing.expect(selected < trace_count);
    }
}
