//! Runs examples/live_suite.zig against a real OpenAI-compatible model and
//! writes the transcript of every request and real completion, so
//! tools/dspy_reference/check_live.py can replay it through DSPy 3.4.1.
//!
//!     OPENAI_API_KEY=... zig build live -- transcript.json
//!
//! Optional: DSZIG_LIVE_MODEL (default gpt-4o-mini), OPENAI_BASE_URL.

const std = @import("std");
const dszig = @import("dszig");
const suite = @import("live_suite");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const key = init.environ_map.get("OPENAI_API_KEY") orelse {
        std.log.err("OPENAI_API_KEY is not set", .{});
        std.process.exit(2);
    };
    const model = init.environ_map.get("DSZIG_LIVE_MODEL") orelse "gpt-4o-mini";
    const base_url = init.environ_map.get("OPENAI_BASE_URL") orelse "https://api.openai.com/v1";
    const out_path = if (args.len > 1) args[1] else "live_dszig.json";

    var client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer client.deinit();
    var openai: dszig.lm.OpenAI = .{ .client = &client, .base_url = base_url, .model = model, .api_key = key };
    var recorder: dszig.lm.RecordingLM = .{ .inner = openai.lm(), .arena = arena };
    const result = try suite.run(arena, init.gpa, recorder.lm());
    const json = try suite.writeTranscript(arena, "dszig", model, recorder.calls.items, result);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });

    var buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    const w = &stdout.interface;
    for (result.cases) |c| {
        try w.print("{s}: {s}\n", .{ c.name, if (c.err) |e| @errorName(e) else "ok" });
    }
    try w.print("{d} real calls to {s}; baseline {d:.1}%  optimized {d:.1}%; wrote {s}\n", .{
        recorder.calls.items.len, model, result.baseline, result.optimized, out_path,
    });
    try w.flush();
}
