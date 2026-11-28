//! Benchmarks the native suite replaying a full-suite transcript. The DSPy
//! counterpart is tools/dspy_reference/bench.py on the same transcript, and
//! tools/dspy_reference/bench_report.py compares the two.
//!
//! Each repetition runs the eight adapter cases and nine optimizer pipelines
//! against a replay LM, so no network or model latency is included: the
//! numbers are framework overhead (prompt formatting, parsing, optimizer
//! logic, evaluation, serialization and the replay lookup). A timestamp is
//! taken as each LM call starts. Prints JSON: per-repetition totals,
//! per-call start offsets, peak requested and reserved heap bytes and peak RSS.
//!
//!     zig build bench -Doptimize=ReleaseFast -- transcript.json [reps]
const std = @import("std");
const builtin = @import("builtin");
const dszig = @import("dszig");
const suite = @import("live_suite");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Live and peak bytes of every allocation counted against it.
const Meter = struct {
    live: usize = 0,
    peak: usize = 0,
};

/// Counts every allocation made through it against a shared meter.
const Counting = struct {
    child: Allocator,
    meter: *Meter,

    fn allocator(self: *Counting) Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn grow(self: *Counting, n: usize) void {
        self.meter.live += n;
        self.meter.peak = @max(self.meter.peak, self.meter.live);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        const p = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.grow(len);
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        if (!self.child.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.meter.live -= memory.len;
        self.grow(new_len);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        const p = self.child.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;
        self.meter.live -= memory.len;
        self.grow(new_len);
        return p;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        self.child.rawFree(memory, alignment, ret_addr);
        self.meter.live -= memory.len;
    }
};

/// Records when each LM call starts, then forwards it.
const TimedLM = struct {
    inner: dszig.LM,
    io: std.Io,
    start: std.Io.Timestamp,
    stamps: []u64,
    count: usize = 0,

    fn lm(self: *TimedLM) dszig.LM {
        var out = self.inner;
        out.ptr = self;
        out.vtable = &.{ .complete = complete };
        return out;
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
        const self: *TimedLM = @ptrCast(@alignCast(ptr));
        if (self.count < self.stamps.len) self.stamps[self.count] = elapsedNs(self.io, self.start);
        self.count += 1;
        return self.inner.vtable.complete(self.inner.ptr, arena, request);
    }
};

fn elapsedNs(io: std.Io, start: std.Io.Timestamp) u64 {
    return @intCast(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
}

fn capability(transcript: dszig.Value, name: []const u8) !bool {
    const value = transcript.object.get(name) orelse return false;
    if (value != .bool) return error.InvalidTranscript;
    return value.bool;
}

fn seconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2 or args.len > 3) {
        std.log.err("usage: bench <DSPy-3.5-transcript.json> [reps]", .{});
        return error.InvalidArguments;
    }
    const reps = if (args.len == 3) try std.fmt.parseInt(usize, args[2], 10) else 20;
    if (reps == 0 or reps > 1000) return error.InvalidArguments;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .limited(64 * 1024 * 1024));
    const transcript = try std.json.parseFromSliceLeaky(dszig.Value, arena, bytes, .{});
    if (transcript != .object) return error.InvalidTranscript;
    const target = transcript.object.get("target_dspy") orelse return error.MissingTranscriptTarget;
    if (target != .string or !std.mem.eql(u8, target.string, suite.target_dspy)) return error.IncompatibleTranscriptTarget;
    const calls = try suite.parseCalls(arena, transcript);
    const n = dszig.lm.ReplayLM.completionCount(calls);

    // Reserved bytes include the suite arena's chunks, which grow
    // geometrically and so jump in steps. Requested bytes are what the suite
    // asked for, the quantity tracemalloc reports for DSPy, as an upper bound:
    // scratch arenas DSZig builds on `gpa` count their whole chunks.
    var reserved: Meter = .{};
    var requested: Meter = .{};
    var reserving: Counting = .{ .child = init.gpa, .meter = &reserved };
    var gpa: Counting = .{ .child = reserving.allocator(), .meter = &requested };
    const totals = try arena.alloc(u64, reps);
    const stamps = try arena.alloc(u64, reps * n);
    // The suite arena is reserved once at startup and never grows: a run
    // that outgrows it fails with OutOfMemory instead of reserving more.
    const arena_buffer = try reserving.allocator().alloc(u8, suite.arena_bytes_max);
    defer reserving.allocator().free(arena_buffer);
    const workspace_buffer = try reserving.allocator().alloc(u8, suite.workspace_bytes_max);
    defer reserving.allocator().free(workspace_buffer);
    var workspace: dszig.Workspace = .init(workspace_buffer);
    var arena_used_peak: usize = 0;
    for (0..reps) |rep| {
        var run_fixed: std.heap.FixedBufferAllocator = .init(arena_buffer);
        const run_arena = run_fixed.threadSafeAllocator();
        defer arena_used_peak = @max(arena_used_peak, run_fixed.end_index);
        const used = try run_arena.alloc(bool, n);
        @memset(used, false);
        var replay: dszig.lm.ReplayLM = .{
            .calls = calls,
            .used = used,
            .supports_reasoning = try capability(transcript, "supports_reasoning"),
            .supports_response_format = try capability(transcript, "supports_response_format"),
            .supports_response_schema = try capability(transcript, "supports_response_schema"),
            .supports_function_calling = try capability(transcript, "supports_function_calling"),
            .supports_decision_requests = try capability(transcript, "supports_decision_requests"),
        };
        const start = std.Io.Clock.awake.now(io);
        var timed: TimedLM = .{ .inner = replay.lm(), .io = io, .start = start, .stamps = stamps[rep * n ..][0..n] };
        var arena_requests: Counting = .{ .child = run_arena, .meter = &requested };
        // Dropping the fixed buffer's contents bypasses the meter.
        const live_before = requested.live;
        defer requested.live = live_before;
        const result = try suite.run(arena_requests.allocator(), gpa.allocator(), &workspace, timed.lm(), try .fromTranscript(transcript));
        totals[rep] = elapsedNs(io, start);
        if (timed.count != n) return error.CallCountMismatch;
        if (!replay.exhausted()) return error.TranscriptNotExhausted;
        try suite.verifyTranscript(arena, transcript, result);
    }
    const usage = std.posix.getrusage(std.posix.rusage.SELF);

    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buffer);
    const w = &stdout.interface;
    try w.print("{{\"runtime\": \"dszig\", \"optimize\": \"{s}\", \"calls\": {d}, \"reps\": {d}, \"total_seconds\": [", .{
        @tagName(builtin.mode), n, reps,
    });
    for (totals, 0..) |t, i| try w.print("{s}{e}", .{ if (i == 0) "" else ", ", seconds(t) });
    try w.writeAll("], \"call_seconds\": [");
    for (0..reps) |rep| {
        try w.writeAll(if (rep == 0) "[" else ", [");
        for (stamps[rep * n ..][0..n], 0..) |s, i| try w.print("{s}{e}", .{ if (i == 0) "" else ", ", seconds(s) });
        try w.writeAll("]");
    }
    try w.print("], \"heap_peak_bytes\": {d}, \"heap_reserved_peak_bytes\": {d}, \"arena_bytes_max\": {d}, \"arena_used_peak_bytes\": {d}, \"maxrss_kb\": {d}}}\n", .{
        requested.peak, reserved.peak, suite.arena_bytes_max, arena_used_peak, usage.maxrss,
    });
    try w.flush();
}
