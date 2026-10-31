//! Evaluation and few-shot optimizers: `Evaluate`, `LabeledFewShot`,
//! `BootstrapFewShot` (ports of DSPy 3.4.1 `dspy.evaluate` and
//! `dspy.teleprompt`). Random choices use the CPython `random.Random` port,
//! so a Zig compile selects the same demos as DSPy for the same inputs.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const predict = @import("predict.zig");
const pyrandom = @import("pyrandom.zig");
const value = @import("value.zig");
const Context = predict.Context;
const Trace = predict.Trace;
const Record = value.Record;

/// `dspy.Example(...).with_inputs(...)`.
pub const Example = struct {
    record: Record,
    input_keys: []const []const u8,

    pub fn isInput(ex: *const Example, key: []const u8) bool {
        for (ex.input_keys) |k| if (std.mem.eql(u8, k, key)) return true;
        return false;
    }

    /// `example.inputs()`: input keys in store order.
    pub fn inputs(ex: *const Example, arena: Allocator) Allocator.Error!Record {
        var out: Record = .empty;
        var it = ex.record.iterator();
        while (it.next()) |entry| {
            if (ex.isInput(entry.key_ptr.*)) try out.put(arena, entry.key_ptr.*, entry.value_ptr.*);
        }
        return out;
    }

    /// Build from a JSON object literal, e.g. `{"question": "...", "answer": "..."}`.
    pub fn fromJson(arena: Allocator, json: []const u8, input_keys: []const []const u8) !Example {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
        if (v != .object) return error.ExpectedObject;
        return .{ .record = v.object, .input_keys = input_keys };
    }
};

/// `metric(example, prediction, trace=None)`; booleans map to 1.0 / 0.0.
pub const Metric = *const fn (arena: Allocator, example: *const Example, prediction: *const Record, trace: ?*const Trace) f64;

pub const Evaluation = struct {
    /// `round(100 * sum / n, 2)` as reported by `dspy.Evaluate`.
    score: f64,
    sum: f64,
    total: usize,
    errors: usize,
};

/// Run `program` over `devset`. Failed examples score 0; once `max_errors`
/// failures accumulate the last error is returned, like DSPy.
pub fn evaluate(
    comptime P: type,
    gpa: Allocator,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    max_errors: usize,
) !Evaluation {
    assert(devset.len > 0);
    var sum: f64 = 0;
    var errors: usize = 0;
    for (devset) |*example| {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        var c = ctx;
        c.arena = scratch.allocator();
        c.trace = null;
        const prediction = runProgram(P, c, program, example) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            errors += 1;
            if (errors >= max_errors) return err;
            continue;
        };
        sum += metric(c.arena, example, &prediction, null);
    }
    const ratio = 100.0 * sum / @as(f64, @floatFromInt(devset.len));
    return .{ .score = @round(ratio * 100.0) / 100.0, .sum = sum, .total = devset.len, .errors = errors };
}

fn runProgram(comptime P: type, ctx: Context, program: *const P, example: *const Example) !Record {
    const inputs = try example.inputs(ctx.arena);
    const input = try value.fromRecord(ctx.arena, P.Input, &inputs);
    const output = try program.forward(ctx, input);
    return value.toRecord(ctx.arena, P.Output, output);
}

/// `rng.sample(population, k)` over records.
fn sampleRecords(arena: Allocator, rng: *pyrandom.Random, population: []const Record, k: usize) Allocator.Error![]Record {
    const indices = try arena.alloc(usize, k);
    const scratch = try arena.alloc(usize, population.len);
    rng.sampleIndices(population.len, indices, scratch);
    const out = try arena.alloc(Record, k);
    for (out, indices) |*o, i| o.* = try value.cloneRecord(arena, &population[i]);
    return out;
}

fn exampleRecords(arena: Allocator, examples: []const Example) Allocator.Error![]Record {
    const out = try arena.alloc(Record, examples.len);
    for (out, examples) |*o, *e| o.* = e.record;
    return out;
}

/// `dspy.LabeledFewShot(k)`.
pub const LabeledFewShot = struct {
    k: usize = 16,

    /// Returns a reset copy of `student` with `k` sampled demos per predictor.
    /// Demos are deep-copied into `out`.
    pub fn compile(self: LabeledFewShot, comptime P: type, out: Allocator, student: P, trainset: []const Example) Allocator.Error!P {
        var program = student;
        try predict.resetDemos(out, &program);
        if (trainset.len == 0) return program;
        const population = try exampleRecords(out, trainset);
        var rng = pyrandom.Random.init(0);
        for (try predict.namedPredictors(out, &program)) |p| {
            p.param.predict.demos = try sampleRecords(out, &rng, population, @min(self.k, trainset.len));
        }
        return program;
    }
};

/// `dspy.BootstrapFewShot`.
pub const BootstrapFewShot = struct {
    metric: ?Metric = null,
    metric_threshold: ?f64 = null,
    max_bootstrapped_demos: usize = 4,
    max_labeled_demos: usize = 16,
    max_rounds: usize = 1,
    /// DSPy `settings.max_errors` default.
    max_errors: usize = 10,

    pub const Report = struct {
        bootstrapped: usize,
        attempts: usize,
        errors: usize,
        validation: usize,
    };

    pub fn Result(comptime P: type) type {
        return struct { program: P, report: Report };
    }

    /// Compile `student` (must be uncompiled). Result demos live in `out`;
    /// per-attempt scratch comes from `gpa` and is freed before returning.
    pub fn compile(
        self: BootstrapFewShot,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        student_in: P,
        teacher_in: ?P,
        trainset: []const Example,
    ) !Result(P) {
        assert(self.max_rounds > 0);
        assert(self.max_errors > 0);
        if (student_in.compiled) return error.StudentCompiled;
        var student = student_in;
        try predict.resetDemos(out, &student);
        var teacher = teacher_in orelse student_in;
        if (self.max_labeled_demos > 0 and !teacher.compiled) {
            teacher = try (LabeledFewShot{ .k = self.max_labeled_demos }).compile(P, out, teacher, trainset);
        }
        const student_preds = try predict.namedPredictors(out, &student);
        const teacher_preds = try predict.namedPredictors(out, &teacher);
        if (student_preds.len != teacher_preds.len) return error.ProgramMismatch;
        for (student_preds, teacher_preds) |s, t| {
            if (!std.mem.eql(u8, s.name, t.name)) return error.ProgramMismatch;
            if (s.param.predict.fields.ptr != t.param.predict.fields.ptr) return error.ProgramMismatch;
        }

        const name2traces = try out.alloc(std.ArrayList(Record), teacher_preds.len);
        for (name2traces) |*l| l.* = .empty;
        const bootstrapped = try out.alloc(bool, trainset.len);
        @memset(bootstrapped, false);
        var n_bootstrapped: usize = 0;
        var attempts: usize = 0;
        var errors: usize = 0;

        for (trainset, 0..) |*example, example_idx| {
            if (n_bootstrapped >= self.max_bootstrapped_demos) break;
            var round_idx: usize = 0;
            while (round_idx < self.max_rounds) : (round_idx += 1) {
                attempts += 1;
                const ok = try self.bootstrapOne(P, out, gpa, ctx, &teacher, teacher_preds, example, round_idx, name2traces, &errors);
                if (ok) {
                    bootstrapped[example_idx] = true;
                    n_bootstrapped += 1;
                    break;
                }
            }
        }

        var validation: std.ArrayList(Record) = .empty;
        for (trainset, bootstrapped) |*e, b| if (!b) try validation.append(out, e.record);
        var shuffle_rng = pyrandom.Random.init(0);
        shuffle_rng.shuffle(Record, validation.items);

        var rng = pyrandom.Random.init(0);
        for (student_preds, name2traces) |p, traces| {
            const augmented = traces.items[0..@min(traces.items.len, self.max_bootstrapped_demos)];
            const want = if (self.max_labeled_demos > augmented.len) self.max_labeled_demos - augmented.len else 0;
            const sampled = try sampleRecords(out, &rng, validation.items, @min(want, validation.items.len));
            const demos = try out.alloc(Record, augmented.len + sampled.len);
            @memcpy(demos[0..augmented.len], augmented);
            @memcpy(demos[augmented.len..], sampled);
            p.param.predict.demos = demos;
        }
        student.compiled = true;
        return .{ .program = student, .report = .{
            .bootstrapped = n_bootstrapped,
            .attempts = attempts,
            .errors = errors,
            .validation = validation.items.len,
        } };
    }

    fn bootstrapOne(
        self: BootstrapFewShot,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        teacher: *P,
        teacher_preds: []const predict.NamedParam,
        example: *const Example,
        round_idx: usize,
        name2traces: []std.ArrayList(Record),
        errors: *usize,
    ) !bool {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();
        var trace: Trace = .{ .arena = arena };
        var c = ctx;
        c.arena = arena;
        c.trace = &trace;
        if (round_idx > 0) {
            c.lm_options.temperature = 1.0;
            c.lm_options.rollout_id = @intCast(round_idx);
        }

        const cached = try arena.alloc([]const Record, teacher_preds.len);
        for (teacher_preds, cached) |p, *cache| {
            cache.* = p.param.predict.demos;
            var kept: std.ArrayList(Record) = .empty;
            for (cache.*) |*demo| {
                if (!value.recordEqual(demo, &example.record)) try kept.append(arena, demo.*);
            }
            p.param.predict.demos = kept.items;
        }
        const run = runProgram(P, c, teacher, example);
        for (teacher_preds, cached) |p, cache| p.param.predict.demos = cache;
        const prediction = run catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            errors.* += 1;
            if (errors.* >= self.max_errors) return err;
            return false;
        };

        if (self.metric) |metric| {
            const score = metric(arena, example, &prediction, &trace);
            const threshold = self.metric_threshold orelse 0;
            const success = if (threshold != 0) score >= threshold else score != 0;
            if (!success) return false;
        }

        const local = try arena.alloc(?Record, teacher_preds.len);
        @memset(local, null);
        for (trace.steps.items) |step| {
            const index = for (teacher_preds, 0..) |p, i| {
                if (p.param.predict == step.predictor) break i;
            } else continue;
            var demo: Record = .empty;
            try demo.put(arena, "augmented", .{ .bool = true });
            var it_in = step.inputs.iterator();
            while (it_in.next()) |e| try demo.put(arena, e.key_ptr.*, e.value_ptr.*);
            var it_out = step.outputs.iterator();
            while (it_out.next()) |e| try demo.put(arena, e.key_ptr.*, e.value_ptr.*);
            // DSPy picks one of several same-predictor traces with a
            // cloudpickle-derived seed; DSZig keeps the most recent one.
            local[index] = demo;
        }
        for (local, name2traces) |maybe, *traces| {
            if (maybe) |demo| try traces.append(out, try value.cloneRecord(out, &demo));
        }
        return true;
    }
};
