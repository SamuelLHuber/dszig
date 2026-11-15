//! Native ParallelExecutor, the execution core of `dspy.Parallel` and
//! DSPy's threaded Evaluate.
//!
//! Outcomes keep input order. A failed item records its error; once
//! `max_errors` failures accumulate, items not yet started are skipped and
//! `execute` returns `error.ExecutionCancelled`, where DSPy raises. One thread
//! runs sequentially on the caller, exactly like DSPy's sequential path; more
//! threads run as `std.Io.Group` workers that claim the next index atomically.
//! Outcome storage is caller-provided, so the executor allocates nothing.
//! Not ported: straggler resubmission (`timeout`, `straggler_limit`), progress
//! bars and SIGINT handling.
//!
//! Upstream (DSPy 3.5.0): dspy/utils/parallelizer.py, dspy/predict/parallel.py

const std = @import("std");
const assert = std.debug.assert;

pub const Options = struct {
    /// DSPy `settings.num_threads` default.
    num_threads: usize = 8,
    /// DSPy `settings.max_errors` default.
    max_errors: usize = 10,
};

pub fn Outcome(comptime R: type) type {
    return union(enum) { unfinished, ok: R, failed: anyerror };
}

pub const Error = error{ExecutionCancelled} || std.Io.Cancelable;

/// Calls `function(context, index, &items[index])` for every item.
/// `function` must be safe to call concurrently when `num_threads > 1`.
pub fn execute(
    comptime T: type,
    comptime R: type,
    io: std.Io,
    options: Options,
    context: anytype,
    comptime function: fn (@TypeOf(context), usize, *const T) anyerror!R,
    items: []const T,
    outcomes: []Outcome(R),
) Error!void {
    assert(options.num_threads > 0);
    assert(items.len == outcomes.len);
    @memset(outcomes, .unfinished);
    const Shared = struct {
        context: @TypeOf(context),
        items: []const T,
        outcomes: []Outcome(R),
        max_errors: usize,
        next: std.atomic.Value(usize) = .init(0),
        errors: std.atomic.Value(usize) = .init(0),
        cancelled: std.atomic.Value(bool) = .init(false),

        fn runOne(self: *@This(), index: usize) void {
            self.outcomes[index] = if (function(self.context, index, &self.items[index])) |result|
                .{ .ok = result }
            else |err| failed: {
                if (self.errors.fetchAdd(1, .acq_rel) + 1 >= self.max_errors) self.cancelled.store(true, .release);
                break :failed .{ .failed = err };
            };
        }

        fn worker(self: *@This()) void {
            while (!self.cancelled.load(.acquire)) {
                const index = self.next.fetchAdd(1, .monotonic);
                if (index >= self.items.len) return;
                self.runOne(index);
            }
        }
    };
    var shared: Shared = .{ .context = context, .items = items, .outcomes = outcomes, .max_errors = options.max_errors };
    if (options.num_threads == 1) {
        for (0..items.len) |index| {
            if (shared.cancelled.load(.acquire)) break;
            shared.runOne(index);
        }
    } else {
        var group: std.Io.Group = .init;
        defer group.cancel(io);
        for (0..@min(options.num_threads, items.len)) |_| {
            group.concurrent(io, Shared.worker, .{&shared}) catch group.async(io, Shared.worker, .{&shared});
        }
        try group.await(io);
    }
    if (shared.cancelled.load(.acquire)) return error.ExecutionCancelled;
}

fn squareOrFail(fail_mod: usize, index: usize, item: *const usize) anyerror!usize {
    assert(index == item.*);
    if (item.* % fail_mod == fail_mod - 1) return error.ItemFailed;
    return item.* * item.*;
}

test "parallel executor keeps input order across threads and cancels sequentially at max_errors" {
    var items: [64]usize = undefined;
    for (&items, 0..) |*item, i| item.* = i;
    var outcomes: [64]Outcome(usize) = undefined;
    try execute(usize, usize, std.testing.io, .{ .num_threads = 6, .max_errors = 100 }, @as(usize, 7), squareOrFail, &items, &outcomes);
    for (outcomes, 0..) |outcome, i| {
        if (i % 7 == 6) try std.testing.expectEqual(error.ItemFailed, outcome.failed) else try std.testing.expectEqual(i * i, outcome.ok);
    }
    try std.testing.expectError(error.ExecutionCancelled, execute(usize, usize, std.testing.io, .{ .num_threads = 1, .max_errors = 2 }, @as(usize, 3), squareOrFail, items[0..10], outcomes[0..10]));
    try std.testing.expectEqual(@as(usize, 16), outcomes[4].ok);
    try std.testing.expect(outcomes[5] == .failed);
    try std.testing.expect(outcomes[6] == .unfinished);
}
