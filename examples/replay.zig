//! Replay a DSPy 3.5 transcript through the native Zig suite and verify every
//! request, completion use, output, score, optimizer trial and saved artifact.
//! This performs no network calls; the transcript retains its original provenance.
const std = @import("std");
const dszig = @import("dszig");
const suite = @import("live_suite");

fn capability(transcript: dszig.Value, name: []const u8) !bool {
    const value = transcript.object.get(name) orelse return error.MissingTranscriptCapabilities;
    if (value != .bool) return error.InvalidTranscript;
    return value.bool;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 2) {
        std.log.err("usage: replay <DSPy-3.5-transcript.json>", .{});
        return error.InvalidArguments;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], arena, .limited(64 * 1024 * 1024));
    const transcript = try std.json.parseFromSliceLeaky(dszig.Value, arena, bytes, .{});
    if (transcript != .object) return error.InvalidTranscript;
    const target = transcript.object.get("target_dspy") orelse return error.MissingTranscriptTarget;
    if (target != .string or !std.mem.eql(u8, target.string, suite.target_dspy)) return error.IncompatibleTranscriptTarget;
    const calls = try suite.parseCalls(arena, transcript);
    for (calls) |call| {
        if (call.options == null) return error.MissingTranscriptRequestOptions;
    }
    const used = try arena.alloc(bool, dszig.lm.ReplayLM.completionCount(calls));
    @memset(used, false);
    var replay: dszig.lm.ReplayLM = .{
        .calls = calls,
        .used = used,
        .supports_reasoning = try capability(transcript, "supports_reasoning"),
        .supports_response_format = try capability(transcript, "supports_response_format"),
        .supports_response_schema = try capability(transcript, "supports_response_schema"),
        .supports_function_calling = try capability(transcript, "supports_function_calling"),
    };
    const result = try suite.run(arena, init.gpa, replay.lm());
    if (!replay.exhausted()) return error.TranscriptNotExhausted;
    try suite.verifyTranscript(arena, transcript, result);
    var buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    try stdout.interface.print("DSPy {s} replay: {d} requests consumed; {d} adapter cases and {d} pipelines match\n", .{
        suite.target_dspy, calls.len, result.cases.len, suite.pipeline_names.len,
    });
    try stdout.interface.flush();
}
