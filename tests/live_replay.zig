//! Replays the transcript of real gpt-4o-mini calls recorded by DSPy 3.4.1
//! (tools/dspy_reference/live.py) through DSZig. Every request DSZig sends
//! must byte-match a recorded DSPy request, every recorded call must be
//! consumed, and outputs, scores and the compiled artifact must equal DSPy's.

const std = @import("std");
const dszig = @import("dszig");
const suite = @import("live_suite");
const testing = std.testing;
const Value = std.json.Value;

const transcript_json = @embedFile("fixtures/live/dspy_gpt-4o-mini.json");

fn compact(arena: std.mem.Allocator, v: Value) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(arena);
    try dszig.pyjson.writeJson(&aw.writer, v, .compact);
    return aw.written();
}

test "DSZig reproduces DSPy's real gpt-4o-mini run from its transcript" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const transcript = try std.json.parseFromSliceLeaky(Value, arena, transcript_json, .{});
    const calls = try suite.parseCalls(arena, transcript);
    const used = try arena.alloc(bool, calls.len);
    @memset(used, false);
    var replay: dszig.lm.ReplayLM = .{ .calls = calls, .used = used };

    const result = try suite.run(arena, testing.allocator, replay.lm());
    try testing.expectEqual(@as(usize, 0), replay.misses);
    try testing.expect(replay.exhausted());

    const want_cases = transcript.object.get("cases").?.array.items;
    try testing.expectEqual(want_cases.len, result.cases.len);
    for (want_cases, result.cases) |want, got| {
        try testing.expectEqualStrings(want.object.get("name").?.string, got.name);
        try testing.expectEqual(want.object.get("error") != null, got.err != null);
        if (want.object.get("outputs")) |outputs| {
            try testing.expectEqualStrings(try compact(arena, outputs), try compact(arena, .{ .object = got.outputs.? }));
        }
    }
    const mine_json = try suite.writeTranscript(arena, "dszig", "replay", calls, result);
    const mine_root = try std.json.parseFromSliceLeaky(std.json.Value, arena, mine_json, .{});
    inline for (suite.pipeline_names) |name| {
        const want = transcript.object.get(name).?.object;
        const got = @field(result, name);
        const mine = mine_root.object.get(name).?.object;
        try testing.expectEqual(want.get("baseline").?.float, got.baseline);
        try testing.expectEqual(want.get("optimized").?.float, got.optimized);
        try testing.expectEqualStrings(want.get("artifact").?.string, got.artifact);
        const want_candidates = want.get("candidates").?.array.items;
        try testing.expectEqual(want_candidates.len, got.candidates.len);
        for (want_candidates, got.candidates) |pair, c| {
            try testing.expectEqual(pair.array.items[0].integer, c.seed);
            try testing.expectEqual(pair.array.items[1].float, c.score);
        }
        inline for (.{ "instructions", "trials", "scores", "train_scores" }) |key| {
            try testing.expectEqualStrings(try compact(arena, want.get(key).?), try compact(arena, mine.get(key).?));
        }
    }
}
