//! Evaluation and few-shot optimizers: `Evaluate`, `LabeledFewShot`,
//! `BootstrapFewShot` (ports of DSPy 3.5.0 `dspy.evaluate` and
//! `dspy.teleprompt`). Random choices use the CPython `random.Random` port,
//! so a Zig compile selects the same demos as DSPy for the same inputs.
//!
//! Upstream (DSPy 3.5.0): dspy/evaluate/evaluate.py, dspy/teleprompt/bootstrap.py, dspy/teleprompt/random_search.py, dspy/teleprompt/vanilla.py

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
        const v = try std.json.parseFromSliceLeaky(@import("runtime_value.zig").Value, arena, json, .{});
        if (v != .object) return error.ExpectedObject;
        return .{ .record = v.object, .input_keys = input_keys };
    }
};

/// `metric(example, prediction, trace=None)`; booleans map to 1.0 / 0.0.
/// SIMBA passes null after program failure, as Source passes None. Other
/// scoring paths pass a successful prediction. Comparison failures enter the
/// optimizer's existing error policy.
pub const Metric = *const fn (arena: Allocator, example: *const Example, prediction: ?*const Record, trace: ?*const Trace) value.ParseError!f64;

pub const Evaluation = struct {
    /// `round(100 * sum / n, 2)` as reported by `dspy.Evaluate`.
    score: f64,
    sum: f64,
    total: usize,
    errors: usize,
};

/// Run `program` and `metric` over `devset`. Failed examples score 0; once
/// `max_errors` failures accumulate, return an error.
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
    // Static outer storage survives each scratch reset without consuming the
    // per-example backing budget. Retained optimizer runs use runProgram.
    var storage: value.RecordStorage(@typeInfo(P.Output).@"struct".field_names.len) = undefined;
    try storage.init();
    var output: P.Output = undefined;
    var sum: f64 = 0;
    var errors: usize = 0;
    for (devset) |*example| {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        var c = ctx;
        c.arena = scratch.allocator();
        c.trace = null;
        const score: anyerror!f64 = if (runProgramInto(P, c, program, example, &output, &storage.record))
            metric(c.arena, example, &storage.record, null)
        else |err|
            err;
        sum += score catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            errors += 1;
            if (errors >= max_errors) return err;
            continue;
        };
    }
    return summarize(sum, devset.len, errors);
}

/// Optimizer counterpart of DSPy `num_threads`: a startup-created
/// `parallel.Evaluator` bound to one program type, per-attempt contexts and
/// options (see `parallel.Evaluator.concurrent`). null keeps serial `evaluate`.
pub const ConcurrentEvaluation = struct {
    ptr: *anyopaque,
    program_type: *const anyopaque,
    run: *const fn (
        ptr: *anyopaque,
        program: *const anyopaque,
        devset: []const Example,
        metric: Metric,
        max_errors: usize,
    ) anyerror!Evaluation,
};

/// Evaluate serially or with the configured concurrent evaluator. The
/// concurrent path replaces ctx with its own bound per-attempt contexts.
pub fn evaluateWith(
    comptime P: type,
    concurrent: ?ConcurrentEvaluation,
    gpa: Allocator,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    max_errors: usize,
) !Evaluation {
    const bound = concurrent orelse return evaluate(P, gpa, ctx, program, devset, metric, max_errors);
    if (bound.program_type != @import("runtime_value.zig").typeIdentity(P)) return error.ProgramTypeMismatch;
    return bound.run(bound.ptr, program, devset, metric, max_errors);
}

/// Report an ordered devset score sum as `dspy.Evaluate` does.
pub fn summarize(sum: f64, total: usize, errors: usize) Evaluation {
    assert(total > 0);
    assert(errors <= total);
    const ratio = 100.0 * sum / @as(f64, @floatFromInt(total));
    return .{ .score = roundScore(ratio), .sum = sum, .total = total, .errors = errors };
}

/// Python round(x, 2): round the exact binary value to cents, ties to even.
/// Floating multiplication by 100 would round once before the tie decision.
fn roundScore(score: f64) f64 {
    if (!std.math.isFinite(score) or @abs(score) >= 0x1.0p46) return score;
    const bits: u64 = @bitCast(score);
    const negative = bits >> 63 != 0;
    const exponent: i32 = @as(i32, @intCast((bits >> 52) & 0x7ff)) - 1023;
    // Subnormals and sufficiently small normals round to signed zero.
    if (exponent <= -12) return if (negative) -@as(f64, 0) else 0;
    const mantissa = (bits & 0x000f_ffff_ffff_ffff) | (@as(u64, 1) << 52);
    const scaled = mantissa * 100;
    const shift: u6 = @intCast(52 - exponent);
    var cents = scaled >> shift;
    const remainder = scaled & ((@as(u64, 1) << shift) - 1);
    const half = @as(u64, 1) << (shift - 1);
    if (remainder > half or (remainder == half and cents & 1 != 0)) cents += 1;
    // Below 2^46, cents fits in 53 bits, so its float conversion is exact.
    const rounded = @as(f64, @floatFromInt(cents)) / 100.0;
    return if (negative) -rounded else rounded;
}

/// Materialize a prediction that can be retained for ctx.arena's lifetime.
pub fn runProgram(comptime P: type, ctx: Context, program: *const P, example: *const Example) !Record {
    return value.toRecord(ctx.arena, P.Output, try forwardProgram(P, ctx, program, example));
}

/// Execute into caller-provisioned outer storage. Values borrow program output
/// or ctx.arena until its reset; callers must copy any result they retain across
/// reuse. Input/program failure also clears stale predictions.
/// output must remain alive and unmoved until prediction is discarded or reused.
pub fn runProgramInto(
    comptime P: type,
    ctx: Context,
    program: *const P,
    example: *const Example,
    output: *P.Output,
    prediction: *Record,
) !void {
    assert(prediction.capacity() >= @typeInfo(P.Output).@"struct".field_names.len);
    output.* = forwardProgram(P, ctx, program, example) catch |err| {
        prediction.clearRetainingCapacity();
        return err;
    };
    try value.toRecordInto(ctx.arena, P.Output, output, prediction);
}

/// Execute a typed program directly from borrowed Example input roles, without
/// materializing either an input dictionary or an outer prediction dictionary.
pub fn forwardProgram(comptime P: type, ctx: Context, program: *const P, example: *const Example) !P.Output {
    // Decode only declared input roles, directly from borrowed training storage.
    // Interleave role checks and conversion to preserve declaration-order failures.
    var input: P.Input = undefined;
    const fields = @typeInfo(P.Input).@"struct";
    inline for (fields.field_names, fields.field_types) |name, T| {
        if (!example.isInput(name)) return error.MissingField;
        const item = example.record.get(name) orelse return error.MissingField;
        @field(input, name) = try value.fromValue(ctx.arena, T, item);
    }
    return program.forward(ctx, input);
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
    // A signed wide count preserves Python negative slicing and all native usize counts.
    k: i128 = 16,

    pub const Options = struct {
        sample: bool = true,
    };

    /// Returns a reset copy. Selected demos are owned by `out`, including prefix mode.
    /// Empty datasets return before validating k, matching Source reset_copy.
    pub fn compile(
        self: LabeledFewShot,
        comptime P: type,
        out: Allocator,
        student: P,
        trainset: []const Example,
        options: Options,
    ) !P {
        var program = student;
        try predict.resetPredictors(out, &program);
        if (trainset.len == 0) return program;
        const predictors = try predict.namedPredictors(out, &program);
        if (options.sample) {
            // Source reaches sample's count validation only with a predictor.
            if (predictors.len == 0) return program;
            if (self.k < 0) return error.InvalidSampleCount;
            var rng = pyrandom.Random.init(0);
            const population = try exampleRecords(out, trainset);
            const count: usize = @intCast(@min(self.k, trainset.len));
            for (predictors) |p| {
                p.param.predict.demos = try sampleRecords(out, &rng, population, count);
            }
        } else {
            const count = self.prefixCount(trainset.len);
            for (predictors) |p| {
                const demos = try out.alloc(Record, count);
                for (demos, trainset[0..count]) |*demo, *example| {
                    demo.* = try value.cloneRecord(out, &example.record);
                }
                p.param.predict.demos = demos;
            }
        }
        return program;
    }

    fn prefixCount(self: LabeledFewShot, length: usize) usize {
        if (self.k >= 0) return @intCast(@min(self.k, length));
        // Avoid negating minInt; Python's negative stop clamps at zero.
        const magnitude: u128 = @as(u128, @intCast(-(self.k + 1))) + 1;
        if (magnitude >= length) return 0;
        return length - @as(usize, @intCast(magnitude));
    }
};

/// `teacher_settings`: the `dspy.context(...)` the teacher runs under.
pub const TeacherSettings = struct {
    lm: ?@FieldType(Context, "lm") = null,
    adapter: ?@FieldType(Context, "adapter") = null,

    pub fn apply(self: TeacherSettings, c: *Context) void {
        if (self.lm) |l| c.lm = l;
        if (self.adapter) |a| c.adapter = a;
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
    teacher_settings: TeacherSettings = .{},

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
        teacher_in: anytype,
        trainset: []const Example,
    ) !Result(P) {
        if (student_in.compiled) return error.StudentCompiled;
        var student = student_in;
        try predict.resetPredictors(out, &student);
        var teacher = (try teacherProgram(P, teacher_in)) orelse student_in;
        if (self.max_labeled_demos > 0 and !teacher.compiled) {
            teacher = try (LabeledFewShot{ .k = self.max_labeled_demos }).compile(P, out, teacher, trainset, .{ .sample = true });
        }
        const student_preds = try predict.namedPredictors(out, &student);
        const teacher_preds = try predict.namedPredictors(out, &teacher);
        if (student_preds.len != teacher_preds.len) return error.ProgramMismatch;
        for (student_preds, teacher_preds) |s, t| {
            if (!std.mem.eql(u8, s.name, t.name)) return error.ProgramMismatch;
            if (!predict.sameSignature(s.param.predict, t.param.predict)) return error.ProgramMismatch;
        }
        // DSPy's bootstrap loop has no final example index for an empty set.
        // Preserve that failure instead of silently returning an untrained copy.
        if (trainset.len == 0) return error.EmptyBootstrapTrainset;

        const name2traces = try out.alloc(std.ArrayList(Record), teacher_preds.len);
        for (name2traces) |*l| l.* = .empty;
        const bootstrapped = try out.alloc(bool, trainset.len);
        @memset(bootstrapped, false);
        var n_bootstrapped: usize = 0;
        var attempts: usize = 0;
        var errors: usize = 0;
        var storage: value.RecordStorage(@typeInfo(P.Output).@"struct".field_names.len) = undefined;
        try storage.init();
        var output: P.Output = undefined;

        for (trainset, 0..) |*example, example_idx| {
            if (n_bootstrapped >= self.max_bootstrapped_demos) break;
            var round_idx: usize = 0;
            while (round_idx < self.max_rounds) : (round_idx += 1) {
                attempts += 1;
                const ok = try self.bootstrapOne(P, out, gpa, ctx, &teacher, teacher_preds, example, round_idx, name2traces, &errors, &output, &storage.record);
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
        output: *P.Output,
        prediction: *Record,
    ) !bool {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();
        var trace: Trace = .{ .arena = arena };
        var c = ctx;
        c.arena = arena;
        c.trace = &trace;
        self.teacher_settings.apply(&c);
        if (round_idx > 0) {
            c.lm_options.temperature = 1.0;
            c.lm_options.rollout_id = .{ .integer = @intCast(round_idx) };
        }

        const run = blk: {
            const cached = try arena.alloc([]const Record, teacher_preds.len);
            for (teacher_preds, cached) |p, *cache| cache.* = p.param.predict.demos;
            defer {
                for (teacher_preds, cached) |p, cache| p.param.predict.demos = cache;
            }
            for (teacher_preds, cached) |p, cache| {
                var kept: std.ArrayList(Record) = .empty;
                for (cache) |*demo| {
                    if (!(try value.recordEqual(demo, &example.record))) try kept.append(arena, demo.*);
                }
                p.param.predict.demos = kept.items;
            }
            break :blk runProgramInto(P, c, teacher, example, output, prediction);
        };
        const success: anyerror!bool = if (run) metric_result: {
            if (self.metric) |metric| {
                const score = metric(arena, example, prediction, &trace) catch |err| break :metric_result err;
                break :metric_result if (self.metric_threshold) |threshold| score >= threshold else score != 0;
            }
            break :metric_result true;
        } else |err| err;
        const accepted = success catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            errors.* += 1;
            if (errors.* >= self.max_errors) return err;
            return false;
        };
        if (!accepted) return false;

        const local = try arena.alloc(std.ArrayList(Record), teacher_preds.len);
        for (local) |*l| l.* = .empty;
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
            try local[index].append(arena, demo);
        }
        // DSPy 3.5 seeds by predictor position so tool-containing traces do
        // not need to be picklable.
        for (local, name2traces, teacher_preds) |demos, *traces, predictor| {
            if (demos.items.len == 0) continue;
            const pick = try tracePick(arena, predictor.name, traces.items.len, demos.items.len);
            try traces.append(out, try value.cloneRecord(out, &demos.items[pick]));
        }
        return true;
    }
};

/// Select one multi-use predictor trace exactly as DSPy 3.5 does.
pub fn tracePick(arena: Allocator, predictor_name: []const u8, prior_count: usize, count: usize) Allocator.Error!usize {
    assert(count > 0);
    if (count == 1) return 0;
    const seed = try std.fmt.allocPrint(arena, "{s}:{d}", .{ predictor_name, prior_count });
    var rng = pyrandom.Random.initStr(seed);
    return if (rng.random() < 0.5) rng.randbelow(count - 1) else count - 1;
}

fn teacherProgram(comptime P: type, teacher: anytype) !?P {
    return switch (@typeInfo(@TypeOf(teacher))) {
        .null => null,
        .optional => teacher,
        .@"struct" => teacher,
        .pointer => error.OptimizerTeacherListUnsupported,
        else => @compileError("teacher must be a program, optional program, or teacher list"),
    };
}

/// `dspy.BootstrapFewShotWithRandomSearch`: evaluates seeds -3 (no demos),
/// -2 (LabeledFewShot), -1 (BootstrapFewShot) and 0.. (shuffled trainset,
/// `randint(1, max_bootstrapped_demos)` demos) on `valset` and keeps the
/// first best-scoring candidate.
pub const BootstrapFewShotWithRandomSearch = struct {
    metric: Metric,
    metric_threshold: ?f64 = null,
    max_bootstrapped_demos: usize = 4,
    max_labeled_demos: usize = 16,
    max_rounds: usize = 1,
    num_candidate_programs: usize = 16,
    max_errors: usize = 10,
    stop_at_score: ?f64 = null,
    /// DSPy `num_threads`; null evaluates candidates serially.
    concurrent_evaluation: ?ConcurrentEvaluation = null,
    teacher_settings: TeacherSettings = .{},

    /// `restrict` is a borrowed compile-local seed set. Duplicates and seeds
    /// outside the range are ignored; candidates always follow Source's
    /// ascending seed range. `labeled_sample` only affects seed -2.
    pub const Options = struct {
        restrict: ?[]const i64 = null,
        labeled_sample: bool = true,
    };

    pub const Candidate = struct { seed: i64, score: f64 };

    /// `candidates` is `best_program.candidate_programs` without the
    /// programs: sorted by score, descending, stable.
    pub fn Result(comptime P: type) type {
        return struct { program: P, candidates: []Candidate };
    }

    /// `valset` defaults to `trainset`, as in DSPy. Result demos live in
    /// `out`; every candidate is built in its own arena from `gpa`.
    pub fn compile(
        self: BootstrapFewShotWithRandomSearch,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        student: P,
        teacher: anytype,
        trainset: []const Example,
        valset: ?[]const Example,
        options: Options,
    ) !Result(P) {
        const end = std.math.cast(i64, self.num_candidate_programs) orelse
            return error.CandidateCapacityExceeded;
        const range_count = std.math.add(usize, self.num_candidate_programs, 3) catch
            return error.CandidateCapacityExceeded;
        // Source rejects a restrict set matching no seed before any candidate work.
        const candidate_count_max = if (options.restrict) |seeds| count: {
            if (!matchesAny(end, seeds)) return error.RestrictMatchesNoSeed;
            break :count @min(range_count, seeds.len);
        } else range_count;
        assert(candidate_count_max > 0);
        const val = if (valset) |set| if (set.len > 0) set else trainset else trainset;
        const candidates = try out.alloc(Candidate, candidate_count_max);
        var count: usize = 0;
        var best: P = undefined;
        var best_score: f64 = 0;
        var best_arena: ?std.heap.ArenaAllocator = null;
        defer if (best_arena) |*a| a.deinit();

        var seed: i64 = -3;
        while (seed < end) : (seed += 1) {
            if (!selected(seed, options.restrict)) continue;
            var arena_state: std.heap.ArenaAllocator = .init(gpa);
            var kept = false;
            defer if (!kept) arena_state.deinit();
            const program = try self.candidate(P, arena_state.allocator(), gpa, ctx, student, teacher, trainset, seed, options.labeled_sample);
            const result = try evaluateWith(P, self.concurrent_evaluation, gpa, ctx, &program, val, self.metric, self.max_errors);
            if (count == 0 or result.score > best_score) {
                best_score = result.score;
                if (best_arena) |*a| a.deinit();
                best_arena = arena_state;
                kept = true;
                best = program;
            }
            assert(count < candidates.len);
            candidates[count] = .{ .seed = seed, .score = result.score };
            count += 1;
            if (self.stop_at_score) |stop| if (result.score >= stop) break;
        }
        assert(count > 0);

        var program = best;
        for (try predict.namedPredictors(out, &program)) |p| {
            const demos = try out.alloc(Record, p.param.predict.demos.len);
            for (demos, p.param.predict.demos) |*d, *src| d.* = try value.cloneRecord(out, src);
            p.param.predict.demos = demos;
        }
        std.sort.insertion(Candidate, candidates[0..count], {}, scoreGreater);
        return .{ .program = program, .candidates = candidates[0..count] };
    }

    fn selected(seed: i64, restrict: ?[]const i64) bool {
        const seeds = restrict orelse return true;
        return std.mem.indexOfScalar(i64, seeds, seed) != null;
    }

    /// One pass over the borrowed set: true when any seed is in `-3..end`.
    fn matchesAny(end: i64, seeds: []const i64) bool {
        assert(end >= 0);
        for (seeds) |seed| {
            if (seed >= -3) {
                if (seed < end) return true;
            }
        }
        return false;
    }

    fn scoreGreater(_: void, a: Candidate, b: Candidate) bool {
        return a.score > b.score;
    }

    fn candidate(
        self: BootstrapFewShotWithRandomSearch,
        comptime P: type,
        arena: Allocator,
        gpa: Allocator,
        ctx: Context,
        student: P,
        teacher: anytype,
        trainset: []const Example,
        seed: i64,
        labeled_sample: bool,
    ) !P {
        assert(seed >= -3);
        if (seed == -3) {
            var program = student;
            try predict.resetPredictors(arena, &program);
            return program;
        }
        if (seed == -2) return (LabeledFewShot{ .k = self.max_labeled_demos }).compile(P, arena, student, trainset, .{ .sample = labeled_sample });
        const shuffled = try arena.dupe(Example, trainset);
        var size = self.max_bootstrapped_demos;
        if (seed >= 0) {
            // Fixed seeds permit zero bootstrapping; only a shuffled seed
            // consumes randint(1, max_bootstrapped_demos).
            if (self.max_bootstrapped_demos == 0) return error.InvalidBootstrapRange;
            var shuffle_rng = pyrandom.Random.init(@intCast(seed));
            shuffle_rng.shuffle(Example, shuffled);
            var size_rng = pyrandom.Random.init(@intCast(seed));
            size = 1 + size_rng.randbelow(self.max_bootstrapped_demos);
        }
        const optimizer: BootstrapFewShot = .{
            .metric = self.metric,
            .metric_threshold = self.metric_threshold,
            .max_bootstrapped_demos = size,
            .max_labeled_demos = self.max_labeled_demos,
            .max_rounds = self.max_rounds,
            .max_errors = self.max_errors,
            .teacher_settings = self.teacher_settings,
        };
        // Only bootstrap consumes the teacher. Earlier reset/labeled candidates
        // may satisfy stop_at_score before an invalid Source list is encountered.
        return (try optimizer.compile(P, arena, gpa, ctx, student, teacher, shuffled)).program;
    }
};
