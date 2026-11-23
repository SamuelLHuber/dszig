//! Allocation-free scheduling shared by the real workers and deterministic simulations.
const std = @import("std");
const assert = std.debug.assert;

pub const Limits = struct {
    workers: u16,
    tasks: u32,
    attempts: u32,
    failures: u32,
};

pub const Options = struct {
    max_errors: u32 = 10,
    return_failed_tasks: bool = false,
    /// Treat a task's Canceled error as a request to cancel the whole batch.
    cancel_on_task_canceled: bool = false,
    /// Zero disables speculative retries, not execution deadlines.
    straggler_timeout_ns: u64 = 120 * std.time.ns_per_s,
    straggler_limit: u32 = 3,
};

pub fn Scheduler(comptime limits: Limits) type {
    comptime {
        assert(limits.workers > 0);
        assert(limits.workers < std.math.maxInt(u16));
        assert(limits.tasks > 0);
        assert(limits.attempts >= limits.tasks);
    }
    return struct {
        task_count: u32 = 0,
        attempt_count: u32 = 0,
        next_queued: u32 = 0,
        running_count: u32 = 0,
        completed_count: u32 = 0,
        error_count: u32 = 0,
        winner_count: u32 = 0,
        failure_count: u32 = 0,
        terminal: Terminal = .idle,
        options: Options = .{},
        attempts: [limits.attempts]Attempt = undefined,
        winners: [limits.tasks]?u32 = undefined,
        errors: [limits.tasks]?anyerror = undefined,
        failed_indices: [limits.attempts]u32 = undefined,

        pub const Terminal = enum { idle, running, success, exhausted, error_limit, canceled, capacity };
        pub const Phase = enum { queued, running, completed };
        pub const Attempt = struct {
            task: u32,
            ordinal: u32,
            phase: Phase = .queued,
            started_ns: u64 = 0,
            resubmitted: bool = false,
        };
        const Self = @This();

        pub fn begin(self: *Self, count: u32, options: Options) void {
            assert(count <= limits.tasks);
            assert(self.running_count == 0);
            self.task_count = count;
            self.attempt_count = 0;
            self.next_queued = 0;
            self.completed_count = 0;
            self.error_count = 0;
            self.winner_count = 0;
            self.failure_count = 0;
            self.options = options;
            self.terminal = if (count == 0) .success else .running;
            @memset(self.winners[0..count], null);
            @memset(self.errors[0..count], null);
            for (0..count) |index| self.enqueue(@intCast(index), 0);
            assert(self.attempt_count == count);
        }

        fn enqueue(self: *Self, task: u32, ordinal: u32) void {
            assert(task < self.task_count);
            assert(self.attempt_count < limits.attempts);
            self.attempts[self.attempt_count] = .{ .task = task, .ordinal = ordinal };
            self.attempt_count += 1;
        }

        pub fn start(self: *Self, now_ns: u64) ?u32 {
            if (self.terminal != .running) return null;
            if (self.next_queued == self.attempt_count) return null;
            const index = self.next_queued;
            const attempt = &self.attempts[index];
            assert(attempt.phase == .queued);
            assert(self.running_count < limits.workers);
            attempt.phase = .running;
            attempt.started_ns = now_ns;
            self.next_queued += 1;
            self.running_count += 1;
            return index;
        }

        pub fn complete(self: *Self, index: u32, failure: ?anyerror) void {
            assert(index < self.attempt_count);
            assert(self.running_count > 0);
            const attempt = &self.attempts[index];
            assert(attempt.phase == .running);
            attempt.phase = .completed;
            self.running_count -= 1;
            self.completed_count += 1;
            // Terminal decisions are irrevocable. Late losing callbacks are joined,
            // but cannot retroactively change a result already accepted by the owner.
            if (self.terminal != .running) return;
            if (failure) |err| {
                if (self.options.cancel_on_task_canceled and err == error.Canceled) {
                    self.cancel();
                    return;
                }
                self.error_count += 1;
                if (self.winners[attempt.task] == null) {
                    assert(self.failure_count < limits.attempts);
                    self.failed_indices[self.failure_count] = attempt.task;
                    self.failure_count += 1;
                    self.errors[attempt.task] = err;
                }
                if (self.error_count >= self.options.max_errors) self.terminal = .error_limit;
            } else if (self.winners[attempt.task] == null) {
                self.winners[attempt.task] = index;
                self.winner_count += 1;
                if (self.winner_count == self.task_count) self.terminal = .success;
            }
            if (self.terminal == .running and self.completed_count == self.attempt_count) {
                self.terminal = .exhausted;
            }
            assert(self.completed_count <= self.attempt_count);
            assert(self.winner_count <= self.task_count);
        }

        /// DSPy retries futures, not task indices: a duplicate can itself retry.
        /// Capacity exhaustion is explicit instead of spawning an unbounded chain.
        pub fn retry(self: *Self, now_ns: u64) void {
            if (self.delay(now_ns) != 0) return;
            const count = self.attempt_count;
            for (self.attempts[0..count]) |*attempt| {
                if (attempt.phase != .running or attempt.resubmitted) continue;
                assert(now_ns >= attempt.started_ns);
                if (now_ns - attempt.started_ns < self.options.straggler_timeout_ns) continue;
                if (self.attempt_count == limits.attempts) {
                    self.terminal = .capacity;
                    return;
                }
                attempt.resubmitted = true;
                self.enqueue(attempt.task, attempt.ordinal + 1);
            }
            assert(self.attempt_count <= limits.attempts);
        }

        pub fn delay(self: *const Self, now_ns: u64) ?u64 {
            if (self.terminal != .running) return null;
            if (limits.workers == 1 or self.options.straggler_timeout_ns == 0) return null;
            const pending = self.attempt_count - self.completed_count;
            if (pending > self.options.straggler_limit) return null;
            var minimum: ?u64 = null;
            for (self.attempts[0..self.attempt_count]) |attempt| {
                if (attempt.phase != .running or attempt.resubmitted) continue;
                assert(now_ns >= attempt.started_ns);
                const elapsed = now_ns - attempt.started_ns;
                const remaining = self.options.straggler_timeout_ns -|
                    elapsed;
                minimum = if (minimum) |value| @min(value, remaining) else remaining;
            }
            return minimum;
        }

        pub fn cancel(self: *Self) void {
            if (self.terminal == .running) self.terminal = .canceled;
            assert(self.terminal != .running);
        }
    };
}
