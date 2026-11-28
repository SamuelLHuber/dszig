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
const Graph = @import("prediction.zig").Graph;
const Value = @import("runtime_value.zig").Value;

/// The Prediction a program run hands to metrics: `fields()` is the output
/// record and `getLMUsage()` the usage of that one invocation. Usage is not
/// part of the record, so demos, feedback, training data and traces built
/// from `fields()` serialize output fields only.
pub const RecordPrediction = predict.Prediction(Record);

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

pub const TraceDemoError = Allocator.Error || error{ DuplicateTraceDemoKeyword, UnsupportedTraceDemoBase };

/// Bind `Example(augmented=True, **inputs, **outputs)` before looking up
/// the predictor. Only keyword arguments collide; copied base fields do not.
/// The demo owns its outer map; unmodified fields borrow from the trace.
pub fn traceDemo(arena: Allocator, step: Trace.Step, input_field_maxlen: usize) TraceDemoError!Record {
    for ([_][]const u8{ "augmented", "self" }) |key| {
        if (step.inputs.contains(key) or step.outputs.contains(key)) return error.DuplicateTraceDemoKeyword;
    }
    for (step.outputs.keys()) |key| {
        if (step.inputs.contains(key)) return error.DuplicateTraceDemoKeyword;
    }

    const base = if (step.inputs.get("base")) |item|
        try truncateTraceInput(arena, item, input_field_maxlen)
    else
        step.outputs.get("base") orelse .null;
    var demo: Record = .empty;
    errdefer demo.deinit(arena);
    switch (base) {
        .object => |record| for (record.keys(), record.values()) |key, item| {
            try demo.put(arena, key, item);
        },
        .custom => {
            // Example ignores non-dicts other than an actual Example, which
            // has no runtime Value representation. A boxed dict may contain
            // non-string keys, which the native demo Record cannot represent.
            if (@import("python_value.zig").view(base)) |boxed| {
                if (boxed == .dictionary) {
                    for (boxed.dictionary) |entry| {
                        if (entry.key != .string) return error.UnsupportedTraceDemoBase;
                        try demo.put(arena, entry.key.string, entry.value);
                    }
                }
            }
        },
        else => {},
    }

    try demo.put(arena, "augmented", .{ .bool = true });
    for (step.inputs.keys(), step.inputs.values()) |key, item| {
        if (std.mem.eql(u8, key, "base")) continue;
        try demo.put(arena, key, try truncateTraceInput(arena, item, input_field_maxlen));
    }
    for (step.outputs.keys(), step.outputs.values()) |key, item| {
        if (std.mem.eql(u8, key, "base")) continue;
        try demo.put(arena, key, item);
    }
    return demo;
}

fn truncateTraceInput(arena: Allocator, item: Value, maxlen: usize) Allocator.Error!Value {
    if (maxlen == 0) return item;
    var rendered: std.Io.Writer.Allocating = .init(arena);
    defer rendered.deinit();
    const text = if (item == .string) item.string else blk: {
        @import("pyjson.zig").writeStr(&rendered.writer, item) catch return error.OutOfMemory;
        break :blk rendered.written();
    };
    const len = std.unicode.utf8CountCodepoints(text) catch text.len;
    if (len <= maxlen) return item;
    var view = std.unicode.Utf8View.initUnchecked(text).iterator();
    var end: usize = 0;
    for (0..maxlen) |_| end += (view.nextCodepointSlice() orelse break).len;
    return .{ .string = try std.fmt.allocPrint(arena, "{s}\n\t\t... <TRUNCATED FOR BREVITY>", .{text[0..end]}) };
}

/// `metric(example, prediction, trace=None)`; booleans map to 1.0 / 0.0.
/// SIMBA passes null after program failure, as Source passes None. Other
/// scoring paths pass a successful prediction. Comparison failures enter the
/// optimizer's existing error policy.
///
/// The prediction may borrow per-attempt storage: a metric that keeps it
/// beyond its own return must `clone` it.
pub const Metric = *const fn (arena: Allocator, example: *const Example, prediction: ?*const RecordPrediction, trace: ?*const Trace) value.ParseError!f64;

pub const Evaluation = struct {
    /// `round(100 * sum / n, 2)` as reported by `dspy.Evaluate`.
    score: f64,
    sum: f64,
    total: usize,
    errors: usize,
};

pub const EvaluationRow = struct {
    example: *const Example,
    /// Null for either a program failure or a metric failure.
    prediction: ?RecordPrediction,
    score: f64,
};

pub const DetailedEvaluation = struct {
    summary: Evaluation,
    rows: []const EvaluationRow,

    /// Write the DSPy evaluation table as plain JSON. The caller supplies the
    /// metric's column name and owns the destination and its flushing.
    pub fn writeJson(
        self: DetailedEvaluation,
        gpa: Allocator,
        writer: *std.Io.Writer,
        metric_name: []const u8,
    ) !void {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        try writer.writeByte('[');
        for (self.rows, 0..) |row, index| {
            if (index != 0) try writer.writeAll(", ");
            _ = scratch.reset(.retain_capacity);
            const record = try evaluationOutput(scratch.allocator(), row, metric_name);
            try @import("pyjson.zig").writeJsonDepth(writer, .{ .object = record }, .compact, 1, true);
        }
        try writer.writeByte(']');
    }

    /// Write a CSV table, with columns in first-seen order across all rows.
    pub fn writeCsv(
        self: DetailedEvaluation,
        gpa: Allocator,
        writer: *std.Io.Writer,
        metric_name: []const u8,
    ) !void {
        var table: std.heap.ArenaAllocator = .init(gpa);
        defer table.deinit();
        const arena = table.allocator();
        const records = try arena.alloc(Record, self.rows.len);
        var columns: Record = .empty;
        for (self.rows, records) |row, *record| {
            record.* = try evaluationOutput(arena, row, metric_name);
            for (record.keys()) |key| try columns.put(arena, key, .null);
        }
        for (columns.keys(), 0..) |key, index| {
            if (index != 0) try writer.writeByte(',');
            try writeCsvCell(writer, key, columns.count() == 1);
        }
        try writer.writeAll("\r\n");
        var cell: std.Io.Writer.Allocating = .init(gpa);
        defer cell.deinit();
        for (records) |*record| {
            for (columns.keys(), 0..) |key, index| {
                if (index != 0) try writer.writeByte(',');
                const item = record.get(key) orelse .null;
                const text = switch (item) {
                    .null => "",
                    .string => |text| text,
                    else => blk: {
                        cell.clearRetainingCapacity();
                        try @import("pyjson.zig").writeStr(&cell.writer, item);
                        break :blk cell.written();
                    },
                };
                try writeCsvCell(writer, text, columns.count() == 1);
            }
            try writer.writeAll("\r\n");
        }
    }
};

fn evaluationOutput(arena: Allocator, row: EvaluationRow, metric_name: []const u8) !Record {
    const example = &row.example.record;
    const prediction = if (row.prediction) |pred| pred.fields() else &Record.empty;
    var result: Record = .empty;
    for (example.keys(), example.values()) |key, item| {
        const name = if (prediction.contains(key))
            try std.fmt.allocPrint(arena, "example_{s}", .{key})
        else
            key;
        try result.put(arena, name, item);
    }
    for (prediction.keys(), prediction.values()) |key, item| {
        const name = if (example.contains(key))
            try std.fmt.allocPrint(arena, "pred_{s}", .{key})
        else
            key;
        try result.put(arena, name, item);
    }
    try result.put(arena, metric_name, .{ .float = row.score });
    return result;
}

fn writeCsvCell(writer: *std.Io.Writer, cell: []const u8, only_column: bool) !void {
    if ((cell.len != 0 or !only_column) and std.mem.indexOfAny(u8, cell, ",\"\r\n") == null)
        return writer.writeAll(cell);
    try writer.writeByte('"');
    var start: usize = 0;
    while (std.mem.indexOfScalarPos(u8, cell, start, '"')) |index| {
        try writer.writeAll(cell[start .. index + 1]);
        try writer.writeByte('"');
        start = index + 1;
    }
    try writer.writeAll(cell[start..]);
    try writer.writeByte('"');
}

pub const EvaluationOptions = struct {
    /// Null inherits the evaluation context. Explicit zero aborts on the first error.
    max_errors: ?usize = null,
    failure_score: f64 = 0,
};

/// Serial evaluation retaining predictions in devset order. Predictions and
/// rows live in `out` (an arena); each row borrows its example from `devset`.
pub fn evaluateDetailed(
    comptime P: type,
    out: Allocator,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    options: EvaluationOptions,
) !DetailedEvaluation {
    if (devset.len == 0) return error.EmptyDevset;
    const rows = try out.alloc(EvaluationRow, devset.len);
    const summary = try evaluateScores(P, ctx, program, devset, metric, options, .{
        .rows = .{ .out = out, .items = rows },
    });
    return .{ .summary = summary, .rows = rows };
}

const EvaluationRetention = struct {
    subscores: ?[]f64 = null,
    rows: ?struct { out: Allocator, items: []EvaluationRow } = null,
};

/// Run `program` and `metric` over `devset`. Failed examples score 0; once
/// `max_errors` failures accumulate, return an error. An empty devset returns
/// `error.EmptyDevset` before invoking the program or metric.
pub fn evaluate(
    comptime P: type,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    max_errors: usize,
) !Evaluation {
    return evaluateScores(P, ctx, program, devset, metric, .{ .max_errors = max_errors }, .{});
}

fn evaluateScores(
    comptime P: type,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    options: EvaluationOptions,
    retention: EvaluationRetention,
) !Evaluation {
    if (devset.len == 0) return error.EmptyDevset;
    if (retention.subscores) |scores| assert(scores.len == devset.len);
    if (retention.rows) |rows| assert(rows.items.len == devset.len);
    // Static outer storage survives each scratch reset without consuming the
    // per-example backing budget. Retained optimizer runs use runProgram.
    var storage: value.RecordStorage(@typeInfo(P.Output).@"struct".field_names.len) = undefined;
    try storage.init();
    var state: RecordPrediction.State = .{ .output = storage.record };
    var output: P.Output = undefined;
    var sum: f64 = 0;
    var errors: usize = 0;
    const max_errors = options.max_errors orelse ctx.max_errors;
    const workspace = ctx.workspace orelse return error.WorkspaceRequired;
    for (devset, 0..) |*example, index| {
        var scratch = workspace.open();
        defer scratch.close();
        var c = ctx;
        c.arena = scratch.allocator();
        c.trace = null;
        const prediction = runProgramInto(P, c, program, example, &output, &state);
        const scored = (if (prediction) |pred|
            metric(c.arena, example, &pred, null)
        else |err|
            err) catch |err| blk: {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            errors += 1;
            if (errors >= max_errors) return err;
            break :blk null;
        };
        const score = scored orelse options.failure_score;
        sum += score;
        if (retention.subscores) |scores| scores[index] = score;
        if (retention.rows) |rows| rows.items[index] = .{
            .example = example,
            .prediction = if (scored != null) try (prediction catch unreachable).clone(rows.out) else null,
            .score = score,
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
    /// Active attempt contexts, for optimizers that must validate LM bindings.
    contexts: ?*const fn (ptr: *anyopaque) []const Context = null,
    run: *const fn (
        ptr: *anyopaque,
        program: *const anyopaque,
        devset: []const Example,
        metric: Metric,
        max_errors: usize,
        subscores: ?[]f64,
    ) anyerror!Evaluation,
};

/// Evaluate serially or with the configured concurrent evaluator. The
/// concurrent path replaces ctx with its own bound per-attempt contexts.
pub fn evaluateWith(
    comptime P: type,
    concurrent: ?ConcurrentEvaluation,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    max_errors: usize,
) !Evaluation {
    return evaluateWithScores(P, concurrent, ctx, program, devset, metric, max_errors, null);
}

fn evaluateWithScores(
    comptime P: type,
    concurrent: ?ConcurrentEvaluation,
    ctx: Context,
    program: *const P,
    devset: []const Example,
    metric: Metric,
    max_errors: usize,
    subscores: ?[]f64,
) !Evaluation {
    const bound = concurrent orelse return evaluateScores(
        P,
        ctx,
        program,
        devset,
        metric,
        .{ .max_errors = max_errors },
        .{ .subscores = subscores },
    );
    if (bound.program_type != @import("runtime_value.zig").typeIdentity(P))
        return error.ProgramTypeMismatch;
    return bound.run(bound.ptr, program, devset, metric, max_errors, subscores);
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

/// A Source Prediction without fields is falsy and never receives usage.
fn carriesUsage(comptime P: type) bool {
    return @typeInfo(P.Output).@"struct".field_names.len != 0;
}

/// Run one Module call and materialize its Prediction, retainable for
/// ctx.arena's lifetime: the instance, its record and its usage snapshot are
/// all allocated there. Usage follows `predict.callProgram`: set after a
/// successful automatic scope, null when tracking is off or a tracker is
/// already active. A failed run returns no Prediction.
pub fn runProgram(comptime P: type, ctx: Context, program: *const P, example: *const Example) !RecordPrediction {
    if (comptime @hasDecl(P, "forwardPrediction")) {
        const prediction = try predict.callProgram(program, ctx, try exampleInput(P, ctx.arena, example));
        const result = try RecordPrediction.init(ctx.arena, try value.toRecord(ctx.arena, P.Output, prediction.fields().*));
        result.state.outputs = try completionRecords(P, ctx.arena, prediction);
        result.state.token_scores = prediction.state.token_scores;
        result.setLMUsage(prediction.getLMUsage());
        return result;
    }
    var usage: ?Graph.Value = null;
    const output = try forwardProgram(P, ctx, program, example, if (comptime carriesUsage(P)) &usage else null);
    const prediction = try RecordPrediction.init(ctx.arena, try value.toRecord(ctx.arena, P.Output, output));
    prediction.setLMUsage(usage);
    return prediction;
}

/// Execute into caller-provisioned storage and return a Prediction borrowing
/// it. Nothing is allocated for the instance or its outer record; with usage
/// tracking off (or a tracker already active) the usage is null and costs
/// nothing, otherwise its snapshot is allocated in ctx.arena.
///
/// `state.output` must already hold capacity for every output field, for
/// example the record of an initialized `value.RecordStorage` adopted once
/// (`.{ .output = storage.record }`) and addressed only through the state
/// afterwards. `state` and `output` must stay alive and unmoved.
///
/// The returned Prediction and all its copies are valid until the next run
/// into `state` or the reset of ctx.arena, whichever comes first: record
/// values borrow `output` or ctx.arena, and the usage graph lives in
/// ctx.arena. `clone` what must outlive that. Input or program failure
/// returns no Prediction and clears the state's record and usage.
pub fn runProgramInto(
    comptime P: type,
    ctx: Context,
    program: *const P,
    example: *const Example,
    output: *P.Output,
    state: *RecordPrediction.State,
) !RecordPrediction {
    assert(state.output.capacity() >= @typeInfo(P.Output).@"struct".field_names.len);
    state.outputs = null;
    state.token_scores = null;
    state.usage = null;
    errdefer state.output.clearRetainingCapacity();
    if (comptime @hasDecl(P, "forwardPrediction")) {
        const prediction = try predict.callProgram(program, ctx, try exampleInput(P, ctx.arena, example));
        output.* = prediction.fields().*;
        try value.toRecordInto(ctx.arena, P.Output, output, &state.output);
        state.outputs = try completionRecords(P, ctx.arena, prediction);
        state.token_scores = prediction.state.token_scores;
        state.usage = prediction.getLMUsage();
        return RecordPrediction.borrowed(state);
    }
    var usage: ?Graph.Value = null;
    output.* = try forwardProgram(P, ctx, program, example, if (comptime carriesUsage(P)) &usage else null);
    try value.toRecordInto(ctx.arena, P.Output, output, &state.output);
    state.usage = usage;
    return RecordPrediction.borrowed(state);
}

/// Preserve a canonical predictor's completion metadata at the record boundary;
/// plain typed forwards have no such metadata and keep the prepared fast path.
fn completionRecords(
    comptime P: type,
    arena: Allocator,
    prediction: predict.Prediction(P.Output),
) !?[]Record {
    const source = prediction.completions() orelse return null;
    const records = try arena.alloc(Record, source.len);
    for (source, records) |row, *record| record.* = try value.toRecord(arena, P.Output, row);
    return records;
}

/// Execute a typed program directly from borrowed Example input roles, without
/// materializing either an input dictionary or an outer prediction dictionary.
/// `usage` is the capture of `predict.forwardScoped`: null declares that the
/// caller's result cannot carry it.
pub fn forwardProgram(
    comptime P: type,
    ctx: Context,
    program: *const P,
    example: *const Example,
    usage: ?*?Graph.Value,
) !P.Output {
    if (usage) |target| target.* = null;
    return predict.forwardScoped(program, ctx, try exampleInput(P, ctx.arena, example), usage);
}

/// Decode typed keyword arguments directly from Example input roles.
/// Omitted arguments take the program input struct's declared defaults.
pub fn exampleInput(comptime P: type, arena: Allocator, example: *const Example) !P.Input {
    // Interleave role checks and conversion to preserve declaration-order failures.
    var input: P.Input = undefined;
    const fields = @typeInfo(P.Input).@"struct";
    inline for (fields.field_names, fields.field_types, fields.field_attrs) |name, T, attrs| {
        const item = if (example.isInput(name)) example.record.get(name) else null;
        if (item) |arg| {
            @field(input, name) = try value.fromValue(arena, T, arg);
        } else if (comptime attrs.default_value_ptr) |default| {
            @field(input, name) = @as(*const T, @ptrCast(@alignCast(default))).*;
        } else return error.MissingField;
    }
    return input;
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
    max_errors: ?usize = null,
    /// Null inherits the caller limit; zero and negative values disable tracing.
    max_trace_size: ?@FieldType(Context, "max_trace_size") = null,

    pub fn apply(self: TeacherSettings, c: *Context) void {
        if (self.lm) |l| c.lm = l;
        if (self.adapter) |a| c.adapter = a;
        if (self.max_errors) |limit| c.max_errors = limit;
        if (self.max_trace_size) |limit| c.max_trace_size = limit;
    }
};

/// `dspy.BootstrapFewShot`.
pub const BootstrapFewShot = struct {
    metric: ?Metric = null,
    metric_threshold: ?f64 = null,
    max_bootstrapped_demos: i128 = 4,
    max_labeled_demos: i128 = 16,
    max_rounds: usize = 1,
    /// Null inherits the compile context's limit, not the teacher's override.
    max_errors: ?usize = null,
    teacher_settings: TeacherSettings = .{},
    /// When set, overrides value settings. Read once per attempt, including retries.
    /// The pointee must outlive compile and must not be concurrently mutated.
    teacher_settings_ref: ?*const TeacherSettings = null,
    /// Cumulative teacher and metric failures across serial compiles.
    error_count: usize = 0,

    pub const Report = struct {
        bootstrapped: usize,
        attempts: usize,
        /// Failures during this compile only.
        errors: usize,
        validation: usize,
    };

    pub fn Result(comptime P: type) type {
        return struct { program: P, report: Report };
    }

    /// Compile `student` (must be uncompiled). A distinct native teacher type
    /// is allowed if its ordered predictor names and signatures match the
    /// student's; the metric receives the teacher's own prediction and trace.
    /// Result demos live in `out`; per-attempt scratch comes from `gpa`.
    pub fn compile(
        self: *BootstrapFewShot,
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
        const T = teacherType(P, @TypeOf(teacher_in));
        if (try teacherProgram(T, teacher_in)) |teacher| {
            return self.compilePreparedTeacher(P, T, out, gpa, ctx, student, teacher, trainset);
        }
        return self.compilePreparedTeacher(P, P, out, gpa, ctx, student, student_in, trainset);
    }

    fn compilePreparedTeacher(
        self: *BootstrapFewShot,
        comptime P: type,
        comptime T: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        student_in: P,
        teacher_in: T,
        trainset: []const Example,
    ) !Result(P) {
        var student = student_in;
        var teacher = teacher_in;
        const teacher_compiled = if (comptime @hasField(T, "compiled")) teacher.compiled else false;
        if (self.max_labeled_demos != 0 and !teacher_compiled) {
            teacher = try (LabeledFewShot{ .k = self.max_labeled_demos }).compile(T, out, teacher, trainset, .{ .sample = true });
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
        var storage: value.RecordStorage(@typeInfo(T.Output).@"struct".field_names.len) = undefined;
        try storage.init();
        var state: RecordPrediction.State = .{ .output = storage.record };
        var output: T.Output = undefined;
        var teacher_memory: std.heap.ArenaAllocator = .init(gpa);
        defer teacher_memory.deinit();

        for (trainset, 0..) |*example, example_idx| {
            if (n_bootstrapped >= self.max_bootstrapped_demos) break;
            var round_idx: usize = 0;
            while (round_idx < self.max_rounds) : (round_idx += 1) {
                attempts += 1;
                const ok = try self.bootstrapOne(
                    T,
                    out,
                    gpa,
                    teacher_memory.allocator(),
                    ctx,
                    &teacher,
                    teacher_preds,
                    example,
                    round_idx,
                    name2traces,
                    &errors,
                    &output,
                    &state,
                );
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
            // A nonpositive bootstrap budget never produces any traces.
            const augmented = traces.items[0..@intCast(@min(traces.items.len, @max(0, self.max_bootstrapped_demos)))];
            const want = @min(self.max_labeled_demos -| augmented.len, validation.items.len);
            const sampled = try sampleRecords(out, &rng, validation.items, @intCast(@max(0, want)));
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
        self: *BootstrapFewShot,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        teacher_arena: Allocator,
        ctx: Context,
        teacher: *P,
        teacher_preds: []const predict.NamedParam,
        example: *const Example,
        round_idx: usize,
        name2traces: []std.ArrayList(Record),
        errors: *usize,
        output: *P.Output,
        state: *RecordPrediction.State,
    ) !bool {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();
        var trace: Trace = .{ .arena = arena };
        // Source catches the failure after exiting the teacher settings scope.
        const max_errors = self.max_errors orelse ctx.max_errors;
        var c = ctx;
        c.arena = arena;
        c.trace = &trace;
        const settings = if (self.teacher_settings_ref) |bound| bound.* else self.teacher_settings;
        settings.apply(&c);
        if (round_idx > 0) {
            if (c.before_settings_rollout == null) {
                c.before_settings_rollout = .{
                    .temperature = c.lm_options.temperature,
                    .rollout_id = c.lm_options.rollout_id,
                };
            }
            c.lm_options.temperature = 1.0;
            c.lm_options.rollout_id = .{ .integer = @intCast(round_idx) };
        }

        const run = blk: {
            const cached = try arena.alloc([]const Record, teacher_preds.len);
            for (teacher_preds, cached) |p, *cache| cache.* = p.param.predict.demos;
            defer {
                for (teacher_preds, cached) |p, cache| p.param.predict.demos = cache;
            }
            const prediction: anyerror!RecordPrediction = attempt: {
                for (teacher_preds, cached) |p, cache| {
                    p.param.predict.demos = filterTeacherDemos(arena, cache, &example.record) catch |err|
                        break :attempt err;
                }
                break :attempt runProgramInto(P, c, teacher, example, output, state);
            };
            if (prediction) |_| {} else |err| {
                // Source restores only after a successful teacher call.
                // Preserve completed filters for the next attempt even if a
                // later predictor's comparison failed.
                if (err != error.OutOfMemory and err != error.Canceled and
                    self.error_count < max_errors -| 1)
                {
                    for (teacher_preds, cached) |p, *cache| {
                        const filtered = p.param.predict.demos;
                        if (filtered.len < cache.len) {
                            cache.* = try teacher_arena.dupe(Record, filtered);
                        }
                    }
                }
            }
            break :blk prediction;
        };
        // The Prediction borrows this attempt's scratch. Demos are built from
        // trace steps, so only output fields are ever retained.
        const success: anyerror!bool = if (run) |prediction| metric_result: {
            if (self.metric) |metric| {
                const score = metric(arena, example, &prediction, &trace) catch |err| break :metric_result err;
                break :metric_result if (self.metric_threshold) |threshold| score >= threshold else score != 0;
            }
            break :metric_result true;
        } else |err| err;
        const accepted = success catch |err| {
            if (err == error.OutOfMemory or err == error.Canceled) return err;
            errors.* +|= 1;
            self.error_count +|= 1;
            if (self.error_count >= max_errors) return err;
            return false;
        };
        if (!accepted) return false;

        const local = try arena.alloc(std.ArrayList(Record), teacher_preds.len);
        for (local) |*l| l.* = .empty;
        for (trace.steps.items) |step| {
            const demo = try traceDemo(arena, step, 0);
            const index = for (teacher_preds, 0..) |p, i| {
                if (p.param.predict == step.predictor) break i;
            } else continue;
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

fn filterTeacherDemos(arena: Allocator, demos: []const Record, example: *const Record) ![]const Record {
    const first = for (demos, 0..) |*demo, index| {
        if (try value.recordEqual(demo, example)) break index;
    } else return demos;
    var kept: std.ArrayList(Record) = .empty;
    try kept.ensureTotalCapacity(arena, demos.len - 1);
    kept.appendSliceAssumeCapacity(demos[0..first]);
    for (demos[first + 1 ..]) |*demo| {
        if (!(try value.recordEqual(demo, example))) kept.appendAssumeCapacity(demo.*);
    }
    return kept.items;
}

/// Select one multi-use predictor trace exactly as DSPy 3.5 does.
pub fn tracePick(arena: Allocator, predictor_name: []const u8, prior_count: usize, count: usize) Allocator.Error!usize {
    assert(count > 0);
    if (count == 1) return 0;
    const seed = try std.fmt.allocPrint(arena, "{s}:{d}", .{ predictor_name, prior_count });
    var rng = pyrandom.Random.initStr(seed);
    return if (rng.random() < 0.5) rng.randbelow(count - 1) else count - 1;
}

// Source matches the predictors, not the nominal type of the teacher module.
// Keep coercing same-type struct literals and rejecting lists at the same point
// as before; an absent optional heterogeneous teacher still means the student.
fn teacherType(comptime P: type, comptime Argument: type) type {
    const T = switch (@typeInfo(Argument)) {
        .optional => |info| info.child,
        else => Argument,
    };
    if (@typeInfo(T) == .@"struct" and @hasDecl(T, "Input") and @hasDecl(T, "Output")) return T;
    return P;
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
    max_bootstrapped_demos: i128 = 4,
    max_labeled_demos: i128 = 16,
    max_rounds: usize = 1,
    /// Upper bound of DSPy's seed range, excluding this value.
    num_candidate_programs: i128 = 16,
    /// Null inherits the compile context for both bootstrapping and evaluation.
    max_errors: ?usize = null,
    stop_at_score: ?f64 = null,
    /// DSPy `num_threads`; null evaluates candidates serially.
    concurrent_evaluation: ?ConcurrentEvaluation = null,
    teacher_settings: TeacherSettings = .{},
    /// Borrow live settings across candidates and teacher attempts. The pointee
    /// must outlive compile and must not be concurrently mutated.
    teacher_settings_ref: ?*const TeacherSettings = null,

    /// `restrict` is a borrowed compile-local seed set. Duplicates and seeds
    /// outside the range are ignored; candidates always follow Source's
    /// ascending seed range without modifying the supplied slice.
    /// `labeled_sample` only affects seed -2.
    pub const Options = struct {
        restrict: ?[]const i64 = null,
        labeled_sample: bool = true,
    };

    pub fn Candidate(comptime P: type) type {
        return struct {
            seed: i64,
            score: f64,
            subscores: []const f64,
            program: P,

            fn higher(_: void, left: @This(), right: @This()) bool {
                return left.score > right.score;
            }
        };
    }

    /// Candidates are runnable programs in stable descending score order.
    pub fn Result(comptime P: type) type {
        return struct { program: P, candidates: []const Candidate(P) };
    }

    /// `valset` defaults to `trainset`. Candidate state and scores live in `out`.
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
        const end = self.num_candidate_programs;
        const seed_scratch = if (options.restrict) |seeds| try collectRestrictedSeeds(gpa, seeds, end) else null;
        defer if (seed_scratch) |seeds| gpa.free(seeds);
        const restricted = if (seed_scratch) |seeds| sortUniqueSeeds(seeds) else null;
        const candidate_count: i128 = if (restricted) |seeds| seeds.len else @max(0, end +| 3);
        if (candidate_count == 0) return error.NoCandidates;
        const max_errors = self.max_errors orelse ctx.max_errors;
        const val = if (valset) |set| if (set.len > 0) set else trainset else trainset;
        var candidates: std.ArrayList(Candidate(P)) = .empty;
        errdefer candidates.deinit(out);
        // A stop score can end even a very large search at the first seed.
        // Reserve only for candidates we actually reach in that case.
        if (self.stop_at_score == null) {
            if (restricted == null and end > @as(i128, std.math.maxInt(i64)) + 1)
                return error.CandidateCapacityExceeded;
            const capacity = std.math.cast(usize, candidate_count) orelse return error.CandidateCapacityExceeded;
            try candidates.ensureTotalCapacityPrecise(out, capacity);
        }
        var best: usize = 0;

        while (candidates.items.len < candidate_count) {
            const count = candidates.items.len;
            const seed = if (restricted) |seeds| seeds[count] else std.math.cast(i64, @as(i128, count) - 3) orelse return error.CandidateCapacityExceeded;
            try candidates.ensureUnusedCapacity(out, 1);
            const program = try self.candidate(
                P,
                out,
                gpa,
                ctx,
                student,
                teacher,
                trainset,
                seed,
                options.labeled_sample,
            );
            const subscores = try out.alloc(f64, val.len);
            const result = try evaluateWithScores(
                P,
                self.concurrent_evaluation,
                ctx,
                &program,
                val,
                self.metric,
                max_errors,
                subscores,
            );
            if (count == 0 or result.score > candidates.items[best].score) best = count;
            candidates.appendAssumeCapacity(.{
                .seed = seed,
                .score = result.score,
                .program = program,
                .subscores = subscores,
            });
            if (self.stop_at_score) |stop| if (result.score >= stop) break;
        }
        assert(candidates.items.len > 0);

        const program = candidates.items[best].program;
        std.sort.insertion(Candidate(P), candidates.items, {}, Candidate(P).higher);
        return .{ .program = program, .candidates = try candidates.toOwnedSlice(out) };
    }

    fn collectRestrictedSeeds(gpa: Allocator, seeds: []const i64, end: i128) ![]i64 {
        var count: usize = 0;
        for (seeds) |seed| {
            if (seed >= -3 and seed < end) count += 1;
        }
        // Preserve this error before allocating scratch or building any candidate.
        if (count == 0) return error.RestrictMatchesNoSeed;
        const scratch = try gpa.alloc(i64, count);
        var next: usize = 0;
        for (seeds) |seed| {
            if (seed < -3 or seed >= end) continue;
            scratch[next] = seed;
            next += 1;
        }
        return scratch;
    }

    fn sortUniqueSeeds(seeds: []i64) []const i64 {
        assert(seeds.len > 0);
        std.sort.heap(i64, seeds, {}, std.sort.asc(i64));
        var unique: usize = 1;
        for (seeds[1..]) |seed| {
            if (seed == seeds[unique - 1]) continue;
            seeds[unique] = seed;
            unique += 1;
        }
        return seeds[0..unique];
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
        const shuffled = if (seed >= 0) try gpa.dupe(Example, trainset) else null;
        defer if (shuffled) |rows| gpa.free(rows);
        var size = self.max_bootstrapped_demos;
        if (seed >= 0) {
            // Only a shuffled seed consumes randint(1, max_bootstrapped_demos).
            if (self.max_bootstrapped_demos <= 0) return error.InvalidBootstrapRange;
            var shuffle_rng = pyrandom.Random.initWide(@intCast(seed));
            shuffle_rng.shuffle(Example, shuffled.?);
            var size_rng = pyrandom.Random.initWide(@intCast(seed));
            size = 1 + @as(i128, @intCast(size_rng.randbelowWide(@intCast(self.max_bootstrapped_demos))));
        }
        var optimizer: BootstrapFewShot = .{
            .metric = self.metric,
            .metric_threshold = self.metric_threshold,
            .max_bootstrapped_demos = size,
            .max_labeled_demos = self.max_labeled_demos,
            .max_rounds = self.max_rounds,
            .max_errors = self.max_errors,
            .teacher_settings = self.teacher_settings,
            .teacher_settings_ref = self.teacher_settings_ref,
        };
        // Only bootstrap consumes the teacher. Earlier reset/labeled candidates
        // may satisfy stop_at_score before an invalid Source list is encountered.
        return (try optimizer.compile(P, arena, gpa, ctx, student, teacher, shuffled orelse trainset)).program;
    }
};
