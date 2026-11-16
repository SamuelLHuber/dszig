const std = @import("std");
const testing = std.testing;
const dszig = @import("dszig");
const parallel = dszig.parallel;

const Pool = dszig.Parallel(i64, .{ .workers = 2, .tasks = 2, .attempts = 4, .failures = 4 });
const Number = struct {
    number: i64,

    fn invoke(ptr: *anyopaque, _: parallel.Attempt, output: *i64) !void {
        const self: *Number = @ptrCast(@alignCast(ptr));
        if (self.number == -1) return error.FirstNegative;
        if (self.number < 0) return error.SecondNegative;
        output.* = self.number * 2 + 1;
    }

    fn task(self: *Number) Pool.Task {
        return .{ .ptr = self, .invoke = invoke };
    }
};

const Ordered = struct {
    entered: std.Io.Event = .unset,
    released: std.Io.Event = .unset,
    sequence: std.atomic.Value(u32) = .init(0),
    order: [2]u32 = undefined,
    io: std.Io,
    first: bool,
    shared: ?*Ordered = null,

    fn invoke(ptr: *anyopaque, _: parallel.Attempt, output: *i64) !void {
        const self: *Ordered = @ptrCast(@alignCast(ptr));
        const shared = self.shared orelse self;
        if (self.first) {
            shared.entered.set(self.io);
            try shared.released.waitTimeout(self.io, .{ .duration = .{
                .raw = .fromSeconds(5),
                .clock = .awake,
            } });
        } else {
            try shared.entered.waitTimeout(self.io, .{ .duration = .{
                .raw = .fromSeconds(5),
                .clock = .awake,
            } });
        }
        const position = shared.sequence.fetchAdd(1, .monotonic);
        shared.order[position] = if (self.first) 1 else 2;
        output.* = if (self.first) 3 else 5;
        if (!self.first) shared.released.set(self.io);
    }
};

test "parallel preserves input order and accumulates aliased failed-task reports" {
    var pool: Pool = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var first: Ordered = .{ .io = testing.io, .first = true };
    var second: Ordered = .{ .io = testing.io, .first = false, .shared = &first };
    const tasks = [_]Pool.Task{
        .{ .ptr = &first, .invoke = Ordered.invoke },
        .{ .ptr = &second, .invoke = Ordered.invoke },
    };
    const result = try pool.forward(&tasks, .{ .straggler_timeout_ns = 0 });
    try testing.expectEqual(@as(i64, 3), result.results[0].value.*);
    try testing.expectEqual(@as(i64, 5), result.results[1].value.*);
    try testing.expectEqualSlices(u32, &.{ 2, 1 }, &first.order);
    var a: Number = .{ .number = -1 };
    var b: Number = .{ .number = 2 };
    const first_fail = try pool.forward(&.{ a.task(), b.task() }, .{
        .max_errors = 3,
        .return_failed_tasks = true,
        .straggler_timeout_ns = 0,
    });
    try testing.expect(first_fail.results[0] == .failed);
    try testing.expectEqual(@as(i64, 5), first_fail.results[1].value.*);
    try testing.expectEqual(error.FirstNegative, first_fail.failures.items()[0].err);
    var c: Number = .{ .number = -2 };
    b.number = 3;
    const second_fail = try pool.forward(&.{ c.task(), b.task() }, .{
        .max_errors = 3,
        .return_failed_tasks = true,
        .straggler_timeout_ns = 0,
    });
    try testing.expectEqual(@as(i64, 7), second_fail.results[1].value.*);
    try testing.expectEqual(@as(usize, 2), first_fail.failures.items().len);
    try testing.expect(first_fail.failures == second_fail.failures);
    try testing.expect(first_fail.failures.items()[0].task.ptr == @as(*anyopaque, &a));
    try testing.expect(first_fail.failures.items()[1].task.ptr == @as(*anyopaque, &c));
    try testing.expectEqual(error.SecondNegative, first_fail.failures.items()[1].err);
}

const Straggler = struct {
    joined: std.atomic.Value(bool) = .init(false),
    io: std.Io,

    fn invoke(ptr: *anyopaque, attempt: parallel.Attempt, output: *i64) !void {
        const self: *Straggler = @ptrCast(@alignCast(ptr));
        if (attempt.ordinal != 0) {
            output.* = 142;
            return;
        }
        defer self.joined.store(true, .release);
        for (0..5000) |_| {
            try attempt.cancellation.check();
            try std.Io.sleep(self.io, .fromMilliseconds(1), .awake);
        }
        return error.CancellationNotObserved;
    }
};

test "winning duplicate cancels and joins the original before storage reuse" {
    var pool: Pool = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var frame: Straggler = .{ .io = testing.io };
    const tasks = [_]Pool.Task{.{ .ptr = &frame, .invoke = Straggler.invoke }};
    const report = try pool.forward(&tasks, .{ .straggler_timeout_ns = 1 });
    try testing.expectEqual(@as(i64, 142), report.results[0].value.*);
    try testing.expect(frame.joined.load(.acquire));
    try testing.expectEqual(@as(u32, 0), pool.scheduler.running_count);
    var number: Number = .{ .number = 21 };
    const next = try pool.forward(&.{number.task()}, .{ .straggler_timeout_ns = 0 });
    try testing.expectEqual(@as(i64, 43), next.results[0].value.*);
}

test "zero error budget stops queued calls and allows a subsequent run" {
    const Serial = dszig.Parallel(i64, .{ .workers = 1, .tasks = 2, .attempts = 2, .failures = 1 });
    var pool: Serial = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var a: Number = .{ .number = -1 };
    var b: Number = .{ .number = -2 };
    const tasks = [_]Serial.Task{
        .{ .ptr = &a, .invoke = Number.invoke }, .{ .ptr = &b, .invoke = Number.invoke },
    };
    try testing.expectError(error.ExecutionCanceled, pool.forward(&tasks, .{ .max_errors = 0 }));
    try testing.expectEqual(@as(u32, 1), pool.scheduler.next_queued);
    try testing.expectEqual(@as(u32, 1), pool.scheduler.error_count);
    try testing.expectEqual(@as(u32, 0), pool.failures.len);
    a.number = 4;
    const result = try pool.forward(tasks[0..1], .{});
    try testing.expectEqual(@as(i64, 9), result.results[0].value.*);
}

test "successful null differs from a failed result and report capacity is atomic" {
    const Optional = dszig.Parallel(?i64, .{ .workers = 1, .tasks = 2, .attempts = 2, .failures = 1 });
    const Callback = struct {
        fail: bool,
        fn invoke(ptr: *anyopaque, _: parallel.Attempt, output: *?i64) !void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (self.fail) return error.ExpectedFailure;
            output.* = null;
        }
    };
    var pool: Optional = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var good: Callback = .{ .fail = false };
    var bad: Callback = .{ .fail = true };
    const tasks = [_]Optional.Task{
        .{ .ptr = &good, .invoke = Callback.invoke }, .{ .ptr = &bad, .invoke = Callback.invoke },
    };
    const report = try pool.forward(&tasks, .{ .return_failed_tasks = true });
    try testing.expect(report.results[0] == .value);
    try testing.expectEqual(@as(?i64, null), report.results[0].value.*);
    try testing.expect(report.results[1] == .failed);
    try testing.expectError(error.FailureCapacity, pool.forward(&tasks, .{ .return_failed_tasks = true }));
    try testing.expectEqual(@as(usize, 1), report.failures.items().len);
    pool.clearFailures();
    try testing.expectEqual(@as(usize, 0), report.failures.items().len);
    try testing.expectEqual(@as(usize, 0), (try pool.forward(&.{}, .{})).results.len);
    try testing.expectError(error.TaskCapacity, pool.forward(&.{ tasks[0], tasks[1], tasks[0] }, .{}));
}

test "attempt exhaustion and explicit cancellation join running callbacks" {
    const Bounded = dszig.Parallel(i64, .{
        .workers = 3,
        .tasks = 1,
        .attempts = 3,
        .failures = 0,
    });
    const Blocked = struct {
        io: std.Io,
        exited: std.atomic.Value(u32) = .init(0),
        fn invoke(ptr: *anyopaque, attempt: parallel.Attempt, _: *i64) !void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            defer _ = self.exited.fetchAdd(1, .release);
            for (0..5000) |_| {
                try attempt.cancellation.check();
                try std.Io.sleep(self.io, .fromMilliseconds(1), .awake);
            }
            return error.CancellationNotObserved;
        }
    };
    var bounded: Bounded = undefined;
    try bounded.init(testing.io, .{});
    defer bounded.deinit();
    var blocked: Blocked = .{ .io = testing.io };
    const tasks = [_]Bounded.Task{.{ .ptr = &blocked, .invoke = Blocked.invoke }};
    try testing.expectError(error.AttemptCapacity, bounded.forward(&tasks, .{
        .straggler_timeout_ns = 1,
    }));
    try testing.expectEqual(@as(u32, 3), blocked.exited.load(.acquire));
    try testing.expectEqual(@as(u32, 0), bounded.scheduler.running_count);

    const Canceler = struct {
        pool: *Pool,
        fn invoke(ptr: *anyopaque, _: parallel.Attempt, _: *i64) !void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.pool.cancel();
            return error.Canceled;
        }
    };
    var pool: Pool = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var canceler: Canceler = .{ .pool = &pool };
    try testing.expectError(error.Canceled, pool.forward(&.{
        .{ .ptr = &canceler, .invoke = Canceler.invoke },
    }, .{ .straggler_timeout_ns = 0 }));
    var number: Number = .{ .number = 9 };
    const report = try pool.forward(&.{number.task()}, .{ .straggler_timeout_ns = 0 });
    try testing.expectEqual(@as(i64, 19), report.results[0].value.*);
}

const Arguments = struct {
    const Typed = struct {
        pub const Input = struct { number: i64 };
        pub const Output = i64;
        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            return input.number * 2 + 1;
        }
    };
    const Whole = struct {
        pub const Input = *const dszig.Example;
        pub const Output = i64;
        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            return input.record.get("number").?.integer * 2 + input.record.get("label").?.integer;
        }
    };
    const Tuple = struct {
        pub const Input = struct { i64, i64 };
        pub const Output = i64;
        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            return input[0] * 2 + input[1];
        }
    };
};

test "native program binding preserves typed, Example-input, whole-Example and tuple arguments" {
    const Programs = dszig.Parallel(i64, .{ .workers = 2, .tasks = 4, .attempts = 4, .failures = 4 });
    var pool: Programs = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var scratch = std.heap.FixedBufferAllocator.init(&.{});
    var model: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const context: dszig.Context = .{ .arena = scratch.allocator(), .lm = model.lm() };
    const contexts: [4]dszig.Context = @splat(context);
    var example: dszig.Example = .{ .record = .empty, .input_keys = &.{"number"} };
    defer example.record.deinit(testing.allocator);
    try example.record.put(testing.allocator, "number", .{ .integer = 7 });
    try example.record.put(testing.allocator, "label", .{ .integer = 999 });
    const typed: Arguments.Typed = .{};
    const whole: Arguments.Whole = .{};
    const tuple: Arguments.Tuple = .{};
    var direct: Programs.Program(Arguments.Typed) = .{
        .program = &typed,
        .arguments = .{ .input = .{ .number = 7 } },
        .contexts = &contexts,
    };
    var selected: Programs.Program(Arguments.Typed) = .{
        .program = &typed,
        .arguments = .{ .example = &example },
        .contexts = &contexts,
    };
    var all: Programs.Program(Arguments.Whole) = .{
        .program = &whole,
        .arguments = .{ .input = &example },
        .contexts = &contexts,
    };
    var positional: Programs.Program(Arguments.Tuple) = .{
        .program = &tuple,
        .arguments = .{ .input = .{ 7, 3 } },
        .contexts = &contexts,
    };
    const report = try pool.forward(&.{
        direct.task(), selected.task(), all.task(), positional.task(),
    }, .{ .straggler_timeout_ns = 0 });
    for (report.results, [_]i64{ 15, 15, 1013, 17 }) |slot, expected| {
        try testing.expectEqual(expected, slot.value.*);
    }
    try testing.expectEqual(@as(usize, 0), scratch.end_index);
    example.input_keys = &.{};
    const refused = try pool.forward(&.{selected.task()}, .{
        .straggler_timeout_ns = 0,
        .return_failed_tasks = true,
    });
    try testing.expect(refused.results[0] == .failed);
    try testing.expectEqual(error.MissingField, refused.failures.items()[0].err);
}

test "shared typed Predict uses independent per-attempt LM and scratch state" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    const P = dszig.Predict(Sig);
    const Predictions = dszig.Parallel(P.Output, .{
        .workers = 4,
        .tasks = 4,
        .attempts = 4,
        .failures = 0,
    });
    var pool: Predictions = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    const program: P = .{};
    var buffers: [4][32768]u8 = undefined;
    var scratch: [4]std.heap.FixedBufferAllocator = undefined;
    var models: [4]dszig.lm.ScriptedLM = undefined;
    var contexts: [4]dszig.Context = undefined;
    var calls: [4]Predictions.Program(P) = undefined;
    var tasks: [4]Predictions.Task = undefined;
    const answers = [_][]const u8{
        "{\"answer\":2}", "{\"answer\":4}", "{\"answer\":6}", "{\"answer\":8}",
    };
    for (0..4) |index| {
        scratch[index] = .init(&buffers[index]);
        models[index] = .{ .responses = answers[index .. index + 1] };
        contexts[index] = .{
            .arena = scratch[index].allocator(),
            .lm = models[index].lm(),
            .adapter = .json,
        };
        calls[index] = .{
            .program = &program,
            .arguments = .{ .input = .{ .question = "Double." } },
            .contexts = &contexts,
        };
        tasks[index] = calls[index].task();
    }
    const report = try pool.forward(&tasks, .{ .straggler_timeout_ns = 0 });
    for (report.results, models, 0..) |slot, model, index| {
        try testing.expectEqual(@as(i64, @intCast((index + 1) * 2)), slot.value.answer);
        try testing.expectEqual(@as(usize, 1), model.index);
        try testing.expect(scratch[index].end_index <= buffers[index].len);
    }
}

const State = parallel.Scheduler(.{ .workers = 3, .tasks = 2, .attempts = 5, .failures = 5 });

test "simulation preserves duplicate recovery and losing-error cancellation" {
    var state: State = .{};
    state.begin(2, .{ .max_errors = 3, .straggler_timeout_ns = 10 });
    try testing.expectEqual(@as(?u32, 0), state.start(0));
    try testing.expectEqual(@as(?u32, 1), state.start(0));
    state.retry(10);
    try testing.expectEqual(@as(?u32, 2), state.start(10));
    state.complete(2, null);
    state.complete(0, error.LosingFailure);
    state.complete(1, error.PrimaryFailure);
    try testing.expectEqual(@as(?u32, 3), state.start(10));
    state.complete(3, null);
    try testing.expectEqual(.success, state.terminal);
    try testing.expectEqualSlices(?u32, &.{ 2, 3 }, state.winners[0..2]);
    try testing.expectEqual(@as(u32, 2), state.error_count);
    try testing.expectEqualSlices(u32, &.{1}, state.failed_indices[0..state.failure_count]);
    state.begin(2, .{ .max_errors = 1, .straggler_timeout_ns = 10 });
    _ = state.start(0);
    _ = state.start(0);
    state.retry(10);
    _ = state.start(10);
    state.complete(2, null);
    state.complete(0, error.LosingFailure);
    try testing.expectEqual(.error_limit, state.terminal);
    state.complete(1, null);
    try testing.expectEqual(@as(?u32, null), state.winners[1]);
    try testing.expectEqual(@as(u32, 0), state.running_count);
}

test "simulation retries duplicate futures until explicit attempt exhaustion" {
    const Bounded = parallel.Scheduler(.{ .workers = 3, .tasks = 1, .attempts = 3, .failures = 3 });
    var state: Bounded = .{};
    state.begin(1, .{ .straggler_timeout_ns = 10 });
    _ = state.start(0);
    state.retry(9);
    try testing.expectEqual(@as(u32, 1), state.attempt_count);
    state.retry(10);
    _ = state.start(10);
    state.retry(20);
    _ = state.start(20);
    try testing.expectEqual(@as(u32, 2), state.attempts[2].ordinal);
    state.retry(30);
    try testing.expectEqual(.capacity, state.terminal);
    for (0..3) |index| state.complete(@intCast(index), null);
    try testing.expectEqual(@as(u32, 0), state.running_count);
    try testing.expectEqual(@as(?u32, null), state.winners[0]);
}

test "seeded scheduling properties match an independent completion-order error oracle" {
    const Many = parallel.Scheduler(.{ .workers = 4, .tasks = 16, .attempts = 16, .failures = 16 });
    var rng = std.Random.DefaultPrng.init(0x5eed_7061_7261);
    const random = rng.random();
    for (0..10000) |_| {
        var state: Many = .{};
        const count = random.intRangeAtMost(u32, 0, 16);
        const budget = random.intRangeAtMost(u32, 0, 17);
        state.begin(count, .{ .max_errors = budget, .straggler_timeout_ns = 0 });
        var active: [4]u32 = undefined;
        var active_count: u32 = 0;
        var next: u32 = 0;
        var errors: u32 = 0;
        var successes: u32 = 0;
        var canceled = false;
        var expected: [16]?u32 = @splat(null);
        for (0..count) |_| {
            while (active_count < 4 and next < count) : (next += 1) {
                try testing.expectEqual(@as(?u32, next), state.start(0));
                active[active_count] = next;
                active_count += 1;
            }
            const position = random.uintLessThan(u32, active_count);
            const id = active[position];
            active_count -= 1;
            active[position] = active[active_count];
            const fails = random.boolean();
            state.complete(id, if (fails) error.GeneratedFailure else null);
            if (fails) {
                errors += 1;
                canceled = errors >= @max(1, budget);
            } else {
                expected[id] = id;
                successes += 1;
            }
            if (canceled) break;
        }
        for (active[0..active_count]) |id| state.complete(id, error.LateFailure);
        try testing.expectEqualSlices(?u32, expected[0..count], state.winners[0..count]);
        try testing.expectEqual(errors, state.error_count);
        try testing.expectEqual(@as(u32, 0), state.running_count);
        const terminal: Many.Terminal = if (canceled)
            .error_limit
        else if (successes == count)
            .success
        else
            .exhausted;
        try testing.expectEqual(terminal, state.terminal);
    }
}

const Scored = struct {
    gate: *std.Io.Event,
    io: std.Io,
    pub const Input = struct { name: []const u8, wait: bool };
    pub const Output = struct { label: []const u8 };

    pub fn forward(self: *const Scored, ctx: dszig.Context, input: Input) !Output {
        if (std.mem.eql(u8, input.name, "program_fail")) return error.LMFailure;
        // Exhausts the attempt's fixed scratch backing.
        if (std.mem.eql(u8, input.name, "oom")) _ = try ctx.arena.alloc(u8, 4096);
        if (input.wait) try self.gate.waitTimeout(self.io, .{ .duration = .{
            .raw = .fromSeconds(5),
            .clock = .awake,
        } });
        const label = try std.fmt.allocPrint(ctx.arena, "label-{s}", .{input.name});
        if (std.mem.eql(u8, input.name, "last")) self.gate.set(self.io);
        return .{ .label = label };
    }

    fn metric(_: std.mem.Allocator, example: *const dszig.Example, prediction: ?*const dszig.value.Record, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
        const name = example.record.get("name").?.string;
        if (std.mem.eql(u8, name, "metric_fail")) return error.ParseValue;
        const label = prediction.?.get("label").?.string;
        if (!std.mem.eql(u8, label["label-".len..], name)) return error.ParseValue;
        return example.record.get("score").?.float;
    }
};

test "parallel Evaluate sums out-of-order completions in devset order like serial Evaluate" {
    const Evaluator = parallel.Evaluator(Scored, .{ .workers = 2, .tasks = 4, .attempts = 4, .failures = 1 });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const contexts: [4]dszig.Context = @splat(.{ .arena = arena, .lm = unused.lm() });
    var buffers: [4][256]u8 = undefined;
    const backing = [_][]u8{ &buffers[0], &buffers[1], &buffers[2], &buffers[3] };
    var evaluator: Evaluator = undefined;
    try evaluator.init(testing.io, .{}, &backing);
    defer evaluator.deinit();

    // The first example finishes last; float addition order changes the sum.
    const keys: []const []const u8 = &.{ "name", "wait" };
    const devset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"name\":\"first\",\"wait\":true,\"score\":1e16}", keys),
        try dszig.Example.fromJson(arena, "{\"name\":\"second\",\"wait\":false,\"score\":1.0}", keys),
        try dszig.Example.fromJson(arena, "{\"name\":\"third\",\"wait\":false,\"score\":-1e16}", keys),
        try dszig.Example.fromJson(arena, "{\"name\":\"last\",\"wait\":false,\"score\":1.0}", keys),
    };
    const options: parallel.Options = .{ .straggler_timeout_ns = 0 };
    var gate: std.Io.Event = .unset;
    const program: Scored = .{ .gate = &gate, .io = testing.io };
    const concurrent = try evaluator.evaluate(&contexts, &program, &devset, Scored.metric, options);
    gate.set(testing.io);
    const serial = try dszig.optimize.evaluate(Scored, testing.allocator, contexts[0], &program, &devset, Scored.metric, 10);
    try testing.expectEqual(@as(u64, @bitCast(serial.sum)), @as(u64, @bitCast(concurrent.sum)));
    try testing.expectEqual(@as(f64, 1), concurrent.sum);
    try testing.expectEqual(serial.score, concurrent.score);
    try testing.expectEqual(@as(usize, 0), concurrent.errors);

    var failing = devset;
    failing[0] = try dszig.Example.fromJson(arena, "{\"name\":\"program_fail\",\"wait\":false,\"score\":1.0}", keys);
    failing[2] = try dszig.Example.fromJson(arena, "{\"name\":\"third\",\"wait\":false,\"score\":2.0}", keys);
    const recovered = try evaluator.evaluate(&contexts, &program, &failing, Scored.metric, options);
    try testing.expectEqual(@as(f64, 4), recovered.sum);
    try testing.expectEqual(@as(f64, 100), recovered.score);
    try testing.expectEqual(@as(usize, 1), recovered.errors);

    // Program and metric failures both contribute the configured value.
    evaluator.failure_score = -0.5;
    failing[2] = try dszig.Example.fromJson(arena, "{\"name\":\"metric_fail\",\"wait\":false,\"score\":2.0}", keys);
    const penalized = try evaluator.evaluate(&contexts, &program, &failing, Scored.metric, options);
    try testing.expectEqual(@as(f64, 1), penalized.sum);
    try testing.expectEqual(@as(f64, 25), penalized.score);
    try testing.expectEqual(@as(usize, 2), penalized.errors);
    try testing.expectError(error.ExecutionCanceled, evaluator.evaluate(&contexts, &program, &failing, Scored.metric, .{
        .max_errors = 1,
        .straggler_timeout_ns = 0,
    }));

    failing[0] = try dszig.Example.fromJson(arena, "{\"name\":\"oom\",\"wait\":false,\"score\":1.0}", keys);
    try testing.expectError(error.OutOfMemory, evaluator.evaluate(&contexts, &program, &failing, Scored.metric, .{
        .max_errors = 100,
        .straggler_timeout_ns = 0,
    }));
    // Reuse after an aborted run; each attempt's scratch was reset.
    gate = .unset;
    const reused = try evaluator.evaluate(&contexts, &program, &devset, Scored.metric, options);
    try testing.expectEqual(@as(f64, 1), reused.sum);
    const overflow = [_]dszig.Example{ devset[1], devset[1], devset[1], devset[1], devset[1] };
    try testing.expectError(error.TaskCapacity, evaluator.evaluate(&contexts, &program, &overflow, Scored.metric, options));
}
