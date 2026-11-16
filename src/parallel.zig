//! Bounded native parallel calls. Initialize in place; never move a live pool.
//! Worker threads are created only at initialization and joined at deinitialization.
//! Every forward joins its running callbacks before returning or reusing storage.
const std = @import("std");
const assert = std.debug.assert;
const scheduling = @import("parallel_state.zig");
const Context = @import("predict.zig").Context;
const optimize = @import("optimize.zig");
const value = @import("value.zig");

pub const Limits = scheduling.Limits;
pub const Options = scheduling.Options;
pub const Scheduler = scheduling.Scheduler;

pub const Cancellation = struct {
    flag: *const std.atomic.Value(bool),

    pub fn requested(self: Cancellation) bool {
        return self.flag.load(.acquire);
    }

    pub fn check(self: Cancellation) error{Canceled}!void {
        if (self.requested()) return error.Canceled;
    }
};

pub const Attempt = struct {
    index: u32,
    task_index: u32,
    ordinal: u32,
    worker_index: u16,
    cancellation: Cancellation,
};

/// Task descriptors and result slots depend only on Result, so components that
/// do not choose a pool's compile-time limits can still build and read them.
pub fn TaskFor(comptime Result: type) type {
    return struct {
        ptr: *anyopaque,
        /// Must initialize output on success and cooperate with cancellation.
        /// Attempt.index selects private scratch/context/output-child storage.
        invoke: *const fn (*anyopaque, Attempt, *Result) anyerror!void,
    };
}

pub fn SlotFor(comptime Result: type) type {
    return union(enum) { failed, value: *const Result };
}

/// Limits-independent handle to a startup-created pool, for optimizers that
/// accept DSPy-style `num_threads`. Slots are valid until the next forward.
pub fn Runner(comptime Result: type) type {
    return struct {
        ptr: *anyopaque,
        tasks: u32,
        attempts: u32,
        forward: *const fn (*anyopaque, []const TaskFor(Result), Options) anyerror![]const SlotFor(Result),
    };
}

/// Optimizer `num_threads` for retained rollouts (SIMBA, GEPA): a pool runner
/// and per-attempt contexts supplying LM and retriever pointees.
pub fn Rollouts(comptime Result: type) type {
    return struct {
        runner: Runner(Result),
        contexts: []const Context,
        options: Options = .{ .straggler_timeout_ns = 0 },

        /// Runs tasks in pool-sized chunks, copying results in task order.
        /// Callers' callbacks absorb program and metric failures, so a failed
        /// slot means exhausted memory.
        pub fn run(self: @This(), tasks: []const TaskFor(Result), results: []Result) !void {
            assert(tasks.len == results.len);
            assert(self.contexts.len >= self.runner.attempts);
            assert(self.runner.tasks > 0);
            var start: usize = 0;
            while (start < tasks.len) {
                const count = @min(self.runner.tasks, tasks.len - start);
                const slots = try self.runner.forward(self.runner.ptr, tasks[start..][0..count], self.options);
                for (slots, results[start..][0..count]) |slot, *result| switch (slot) {
                    .failed => return error.OutOfMemory,
                    .value => |output| result.* = output.*,
                };
                start += count;
            }
        }

        /// Attempt context with this rollout's settings and the attempt's own
        /// LM (unless shared) and retriever.
        pub fn context(self: @This(), rollout: Context, attempt: Attempt, shared_lm: bool) Context {
            const own = self.contexts[attempt.index];
            var c = rollout;
            if (!shared_lm) c.lm = own.lm;
            c.retriever = own.retriever;
            return c;
        }
    };
}

/// Result can be any native type, including optional values or a heterogeneous
/// tagged union. Callback output storage is stable until the next forward.
/// Callback-owned children and program/context pointees are borrowed, not cloned.
pub fn Parallel(comptime Result: type, comptime limits: Limits) type {
    const State = Scheduler(limits);
    return struct {
        io: std.Io,
        address: *const Self,
        mutex: std.Io.Mutex = .init,
        work: std.Io.Condition = .init,
        changed: std.Io.Condition = .init,
        live: bool = true,
        running: bool = false,
        workers_spawned: u16 = 0,
        workers: [limits.workers]std.Thread = undefined,
        cancellation: std.atomic.Value(bool) = .init(false),
        scheduler: State = .{},
        tasks: []const Task = &.{},
        outputs: [limits.attempts]Result = undefined,
        results: [limits.tasks]Slot = undefined,
        failures: Failures = .{},

        pub const Task = TaskFor(Result);
        pub const Slot = SlotFor(Result);
        pub const Failure = struct { task: Task, err: anyerror };
        pub const Failures = struct {
            buffer: [limits.failures]Failure = undefined,
            len: u32 = 0,

            /// The handle, unlike a captured slice length, observes later appends.
            pub fn items(self: *const Failures) []const Failure {
                assert(self.len <= limits.failures);
                return self.buffer[0..self.len];
            }
        };
        pub const Report = struct {
            results: []const Slot,
            failures: *const Failures,
        };
        pub const Error = error{ TaskCapacity, AttemptCapacity, FailureCapacity, ExecutionCanceled, Canceled };
        const Self = @This();

        /// Owns OS threads, but never owns task frames, contexts or child results.
        /// Initialization failure joins every thread already created.
        pub fn init(self: *Self, io: std.Io, config: std.Thread.SpawnConfig) !void {
            self.* = .{ .io = io, .address = self };
            errdefer self.deinit();
            for (0..limits.workers) |index| {
                self.workers[index] = try std.Thread.spawn(config, worker, .{ self, @as(u16, @intCast(index)) });
                self.workers_spawned += 1;
            }
            assert(self.workers_spawned == limits.workers);
        }

        pub fn deinit(self: *Self) void {
            assert(self.address == self);
            self.mutex.lockUncancelable(self.io);
            assert(!self.running);
            assert(self.live);
            self.live = false;
            self.work.broadcast(self.io);
            self.mutex.unlock(self.io);
            for (self.workers[0..self.workers_spawned]) |thread| thread.join();
            self.workers_spawned = 0;
        }

        /// May be called from another thread while forward is running.
        /// Cancellation is cooperative; a non-cooperating callback delays return.
        pub fn cancel(self: *Self) void {
            assert(self.address == self);
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.running) {
                self.scheduler.cancel();
                self.cancellation.store(true, .release);
                self.changed.signal(self.io);
            }
        }

        pub fn runner(self: *Self) Runner(Result) {
            assert(self.address == self);
            return .{ .ptr = self, .tasks = limits.tasks, .attempts = limits.attempts, .forward = forwardErased };
        }

        fn forwardErased(ptr: *anyopaque, tasks: []const Task, options: Options) anyerror![]const Slot {
            const self: *Self = @ptrCast(@alignCast(ptr));
            assert(!options.return_failed_tasks);
            return (try self.forward(tasks, options)).results;
        }

        pub fn clearFailures(self: *Self) void {
            assert(self.address == self);
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            assert(!self.running);
            self.failures.len = 0;
        }

        /// Task descriptors are borrowed for this call. Failed descriptors retained
        /// in the report borrow their frames until clearFailures/deinit.
        pub fn forward(self: *Self, tasks: []const Task, options: Options) Error!Report {
            assert(self.address == self);
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            assert(self.live);
            assert(!self.running);
            if (tasks.len > limits.tasks) return error.TaskCapacity;
            self.running = true;
            defer self.running = false;
            self.tasks = tasks;
            self.cancellation.store(false, .release);
            self.scheduler.begin(@intCast(tasks.len), options);
            self.work.broadcast(self.io);
            self.awaitCompletion();
            assert(self.scheduler.running_count == 0);
            switch (self.scheduler.terminal) {
                .success, .exhausted => {},
                .canceled => return error.Canceled,
                .error_limit => return error.ExecutionCanceled,
                .capacity => return error.AttemptCapacity,
                .idle, .running => unreachable,
            }
            if (options.return_failed_tasks) try self.retainFailures();
            for (self.scheduler.winners[0..tasks.len], 0..) |winner, index| {
                self.results[index] = if (winner) |id|
                    .{ .value = &self.outputs[id] }
                else
                    .failed;
            }
            return .{ .results = self.results[0..tasks.len], .failures = &self.failures };
        }

        fn awaitCompletion(self: *Self) void {
            while (self.scheduler.terminal == .running) {
                const count = self.scheduler.attempt_count;
                self.scheduler.retry(now(self.io));
                if (self.scheduler.attempt_count != count) self.work.broadcast(self.io);
                if (self.scheduler.terminal != .running) break;
                const timeout: std.Io.Timeout = if (self.scheduler.delay(now(self.io))) |ns|
                    .{ .duration = .{ .raw = .fromNanoseconds(ns), .clock = .awake } }
                else
                    .none;
                self.changed.waitTimeout(self.io, &self.mutex, timeout) catch |err| switch (err) {
                    error.Timeout => {},
                    error.Canceled => self.scheduler.cancel(),
                };
            }
            self.cancellation.store(true, .release);
            // No queued callback may start after the terminal decision. Running
            // callbacks retain exclusive output/scratch slots until this barrier.
            while (self.scheduler.running_count != 0) {
                self.changed.waitUncancelable(self.io, &self.mutex);
            }
            assert(self.scheduler.terminal != .running);
            assert(self.scheduler.running_count == 0);
        }

        fn retainFailures(self: *Self) Error!void {
            const count = self.scheduler.failure_count;
            assert(self.failures.len <= limits.failures);
            if (count > limits.failures - self.failures.len) return error.FailureCapacity;
            const destination = self.failures.buffer[self.failures.len..][0..count];
            for (self.scheduler.failed_indices[0..count], destination) |index, *failure| {
                failure.* = .{
                    .task = self.tasks[index],
                    .err = self.scheduler.errors[index].?,
                };
            }
            self.failures.len += count;
            assert(self.failures.len <= limits.failures);
        }

        fn worker(self: *Self, worker_index: u16) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            // Persistent lifecycle loop: only deinit ends a worker. Each callback
            // and each queue are independently bounded by the caller's contract.
            while (self.live) {
                const id = self.scheduler.start(now(self.io)) orelse {
                    self.work.waitUncancelable(self.io, &self.mutex);
                    continue;
                };
                const scheduled = self.scheduler.attempts[id];
                const task = self.tasks[scheduled.task];
                const attempt: Attempt = .{
                    .index = id,
                    .task_index = scheduled.task,
                    .ordinal = scheduled.ordinal,
                    .worker_index = worker_index,
                    .cancellation = .{ .flag = &self.cancellation },
                };
                self.changed.signal(self.io);
                self.mutex.unlock(self.io);
                const failure: ?anyerror = if (task.invoke(task.ptr, attempt, &self.outputs[id]))
                    null
                else |err|
                    err;
                self.mutex.lockUncancelable(self.io);
                self.scheduler.complete(id, failure);
                if (self.scheduler.terminal != .running) self.cancellation.store(true, .release);
                self.changed.signal(self.io);
            }
            assert(!self.live);
        }

        /// Typed native equivalent of binding a module and its arguments.
        /// Provide independent per-attempt contexts; copying Context alone does
        /// not isolate an allocator, mutable LM, trace or retriever pointee.
        pub fn Program(comptime P: type) type {
            if (P.Output != Result) @compileError("Parallel result must match the native program Output");
            return struct {
                program: *const P,
                arguments: Arguments,
                contexts: []const Context,
                pub const Arguments = union(enum) {
                    input: P.Input,
                    example: *const optimize.Example,
                };
                const Call = @This();

                pub fn task(self: *Call) Task {
                    assert(self.contexts.len >= limits.attempts);
                    return .{ .ptr = self, .invoke = invoke };
                }

                fn invoke(ptr: *anyopaque, attempt: Attempt, output: *Result) anyerror!void {
                    const self: *Call = @ptrCast(@alignCast(ptr));
                    assert(attempt.index < self.contexts.len);
                    try attempt.cancellation.check();
                    const ctx = self.contexts[attempt.index];
                    output.* = switch (self.arguments) {
                        .input => |input| try self.program.forward(ctx, input),
                        .example => |example| blk: {
                            if (@typeInfo(P.Input) != .@"struct") return error.TypeMismatch;
                            break :blk try optimize.forwardProgram(P, ctx, self.program, example);
                        },
                    };
                }
            };
        }
    };
}

/// Native `dspy.Evaluate(num_threads=limits.workers)`: concurrent program and
/// metric calls whose results are summed in devset order, using failure_score
/// for failed slots (zero by default).
/// Initialize in place; never move a live evaluator. Per-attempt prediction
/// records, typed outputs and scratch allocators are created at initialization.
/// Out-of-memory aborts the run, as in serial `optimize.evaluate`.
pub fn Evaluator(comptime P: type, comptime limits: Limits) type {
    const Pool = Parallel(f64, limits);
    const Storage = value.RecordStorage(@typeInfo(P.Output).@"struct".field_names.len);
    return struct {
        pool: Pool,
        storage: [limits.attempts]Storage,
        outputs: [limits.attempts]P.Output,
        scratch: [limits.attempts]std.heap.FixedBufferAllocator,
        calls: [limits.tasks]Call,
        tasks: [limits.tasks]Pool.Task,
        program: *const P,
        metric: optimize.Metric,
        contexts: []const Context,
        out_of_memory: std.atomic.Value(bool),
        bound_contexts: []const Context,
        bound_options: Options,
        /// DSPy Evaluate.failure_score; configure between joined evaluations.
        failure_score: f64,

        const Self = @This();
        const Call = struct { evaluator: *Self, example: *const optimize.Example };

        /// Borrows one scratch backing slice per attempt until deinit.
        pub fn init(self: *Self, io: std.Io, config: std.Thread.SpawnConfig, backing: []const []u8) !void {
            assert(backing.len >= limits.attempts);
            for (&self.storage) |*storage| try storage.init();
            for (&self.scratch, backing[0..limits.attempts]) |*scratch, bytes| scratch.* = .init(bytes);
            self.out_of_memory = .init(false);
            self.failure_score = 0;
            try self.pool.init(io, config);
        }

        pub fn deinit(self: *Self) void {
            self.pool.deinit();
        }

        /// Bind contexts/options for optimizer use (`concurrent_evaluation`).
        /// Each optimizer evaluate call overrides options.max_errors with the
        /// optimizer's own budget. contexts must outlive the returned handle.
        pub fn concurrent(self: *Self, contexts: []const Context, options: Options) optimize.ConcurrentEvaluation {
            assert(contexts.len >= limits.attempts);
            self.bound_contexts = contexts;
            self.bound_options = options;
            return .{ .ptr = self, .program_type = @typeName(P), .run = runBound };
        }

        fn runBound(
            ptr: *anyopaque,
            program: *const anyopaque,
            devset: []const optimize.Example,
            metric: optimize.Metric,
            max_errors: usize,
        ) anyerror!optimize.Evaluation {
            const self: *Self = @ptrCast(@alignCast(ptr));
            var options = self.bound_options;
            options.max_errors = std.math.cast(u32, max_errors) orelse std.math.maxInt(u32);
            const typed: *const P = @ptrCast(@alignCast(program));
            return self.evaluate(self.bound_contexts, typed, devset, metric, options);
        }

        /// contexts supply independent per-attempt LM/retriever pointees; their
        /// arena and trace are replaced. program and metric must be thread-safe.
        /// Predictions borrow attempt storage, valid only during the metric call.
        pub fn evaluate(
            self: *Self,
            contexts: []const Context,
            program: *const P,
            devset: []const optimize.Example,
            metric: optimize.Metric,
            options: Options,
        ) !optimize.Evaluation {
            assert(devset.len > 0);
            assert(contexts.len >= limits.attempts);
            assert(!options.return_failed_tasks);
            if (devset.len > limits.tasks) return error.TaskCapacity;
            self.program = program;
            self.metric = metric;
            self.contexts = contexts;
            self.out_of_memory.store(false, .release);
            for (devset, self.calls[0..devset.len], self.tasks[0..devset.len]) |*example, *call, *task| {
                call.* = .{ .evaluator = self, .example = example };
                task.* = .{ .ptr = call, .invoke = invoke };
            }
            const report = self.pool.forward(self.tasks[0..devset.len], options) catch |err| {
                if (self.out_of_memory.load(.acquire)) return error.OutOfMemory;
                return err;
            };
            if (self.out_of_memory.load(.acquire)) return error.OutOfMemory;
            // Sum after the join in devset order, not completion order, as DSPy does.
            var sum: f64 = 0;
            var errors: usize = 0;
            for (report.results) |slot| switch (slot) {
                .failed => {
                    errors += 1;
                    sum += self.failure_score;
                },
                .value => |score| sum += score.*,
            };
            return optimize.summarize(sum, devset.len, errors);
        }

        fn invoke(ptr: *anyopaque, attempt: Attempt, score: *f64) anyerror!void {
            const call: *Call = @ptrCast(@alignCast(ptr));
            const self = call.evaluator;
            assert(attempt.index < limits.attempts);
            try attempt.cancellation.check();
            const scratch = &self.scratch[attempt.index];
            scratch.reset();
            var ctx = self.contexts[attempt.index];
            ctx.arena = scratch.allocator();
            ctx.trace = null;
            const prediction = &self.storage[attempt.index].record;
            score.* = scoreExample(self, ctx, call.example, &self.outputs[attempt.index], prediction) catch |err| {
                if (err == error.OutOfMemory) {
                    self.out_of_memory.store(true, .release);
                    self.pool.cancel();
                }
                return err;
            };
        }

        fn scoreExample(
            self: *const Self,
            ctx: Context,
            example: *const optimize.Example,
            output: *P.Output,
            prediction: *value.Record,
        ) !f64 {
            try optimize.runProgramInto(P, ctx, self.program, example, output, prediction);
            return self.metric(ctx.arena, example, prediction, null);
        }
    };
}

fn now(io: std.Io) u64 {
    const ns = std.Io.Clock.awake.now(io).nanoseconds;
    assert(ns >= 0);
    assert(ns <= std.math.maxInt(u64));
    return @intCast(ns);
}
