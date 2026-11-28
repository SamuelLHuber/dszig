//! Replays the full deterministic suite recorded independently by DSPy 3.5.0
//! (tools/dspy_reference/gen_suite_replay.py). This is mock-provider parity,
//! not live-model evidence. Every request must byte-match DSPy's recording,
//! every call must be consumed, and outputs, scores and artifacts must agree.
//! check.sh replays the real-model recordings under fixtures/live/.

const std = @import("std");
const dszig = @import("dszig");
const suite = @import("live_suite");
const testing = std.testing;
const Value = dszig.Value;

const transcript_json = @embedFile("fixtures/dspy_3_5_0_mock_suite.json");

fn compact(arena: std.mem.Allocator, v: Value) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(arena);
    try dszig.pyjson.writeJson(&aw.writer, v, .compact);
    return aw.written();
}

test "DSZig reproduces DSPy 3.5.0's independently recorded deterministic suite" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const transcript = try std.json.parseFromSliceLeaky(Value, arena, transcript_json, .{});
    try testing.expectEqualStrings("3.5.0", transcript.object.get("target_dspy").?.string);
    try testing.expectEqualStrings("dspy 3.5.0 deterministic mock", transcript.object.get("runtime").?.string);
    const supports_reasoning = transcript.object.get("supports_reasoning").?.bool;
    const calls = try suite.parseCalls(arena, transcript);
    try testing.expect(transcript.object.get("records_tool_result_names").?.bool);
    // Current-version parity may not use ReplayLM's historical options bypass.
    for (calls) |call| try testing.expect(call.options != null);
    const used = try arena.alloc(bool, dszig.lm.ReplayLM.completionCount(calls));
    @memset(used, false);
    var replay: dszig.lm.ReplayLM = .{
        .calls = calls,
        .used = used,
        .supports_reasoning = supports_reasoning,
        .supports_response_format = transcript.object.get("supports_response_format").?.bool,
        .supports_response_schema = transcript.object.get("supports_response_schema").?.bool,
        .supports_function_calling = transcript.object.get("supports_function_calling").?.bool,
        .supports_decision_requests = if (transcript.object.get("supports_decision_requests")) |value| value.bool else false,
    };

    const result = try suite.run(arena, testing.allocator, dszig.workspace.forTests(), replay.lm(), try .fromTranscript(transcript));
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
    const mine_json = try suite.writeTranscript(arena, "dszig", "replay", replay.lm(), calls, result, .{});
    const mine_root = try std.json.parseFromSliceLeaky(dszig.Value, arena, mine_json, .{});
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
