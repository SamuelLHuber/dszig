//! Native `dspy.teleprompt.bootstrap_trace.bootstrap_trace_data` (DSPy 3.5).
//!
//! Source patches `program.forward` and runs `dspy.Evaluate`; the native helper
//! runs the borrowed program unmodified and attaches a per-attempt trace and
//! parse-failure capture through `Context` instead.
//!
//! Source arguments without a native parameter:
//! - `capture_failed_parses` is unused upstream: adapter parse failures are
//!   always captured, as they are here.
//! - `num_threads` is `Options.rollouts` (a startup-created pool); null is serial.
//! - Progress display has no native counterpart.
//!
//! Source warnings go to `std.log` on the calling thread, in dataset order.
//! ParallelExecutor's per-item error log is not reproduced.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const optimize = @import("optimize.zig");
const parallel = @import("parallel.zig");
const predict = @import("predict.zig");
const value = @import("value.zig");
const Context = predict.Context;
const Example = optimize.Example;
const PredictState = predict.PredictState;
const Record = value.Record;

const log = std.log.scoped(.dspy_bootstrap_trace);

/// Process-local source of `call_id`, drawn only when callbacks are present.
var next_call_id: std.atomic.Value(u64) = .init(1);

/// `FailedPrediction`. A parse failure carries the adapter's completion text
/// and a format reward. A captured crash carries the native error name (Source:
/// `f"{type(e).__name__}: {e}"`; there is no Python class or message) and no
/// format reward.
pub const FailedPrediction = struct {
    completion_text: []const u8,
    format_reward: ?f64 = null,
};

/// The program's own result: Source stores either the `dspy.Prediction` the
/// Module call returned or a `FailedPrediction`. A success keeps the genuine
/// instance, including the usage of that one invocation, exactly as the
/// metric saw it.
pub const Prediction = union(enum) {
    success: optimize.RecordPrediction,
    failed: FailedPrediction,
};

/// A predictor's result inside a trace. Native trace steps record output
/// fields only, so this is deliberately the field projection that demos and
/// fine-tuning data serialize: a step never carries usage.
pub const StepPrediction = union(enum) {
    success: Record,
    failed: FailedPrediction,
};

/// One `(predictor, inputs, prediction)` trace tuple.
pub const TraceStep = struct {
    predictor: *const PredictState,
    inputs: Record,
    prediction: StepPrediction,
};

/// `TraceData`. `score` is null exactly where Source omits the key (no metric).
pub const TraceData = struct {
    example_ind: usize,
    example: *const Example,
    prediction: Prediction,
    trace: []const TraceStep,
    score: ?f64,
};

pub const Stage = enum { program, metric };

/// A row whose `(prediction, trace)` Source cannot unpack: Evaluate replaced a
/// raised program or metric error with an empty Prediction.
pub const Unhandled = struct {
    example_ind: usize,
    stage: Stage,
    /// The native cause. `FailedPredictorNotFound` and `ZeroExpectedOutputs`
    /// stand for the ValueError and ZeroDivisionError Source raises while
    /// building a failed prediction.
    err: anyerror,
};

/// Lets the caller reclaim attempt scratch between examples.
pub const ScratchReset = struct {
    ptr: *anyopaque,
    /// Called on the calling thread only, after every callback has joined and
    /// the attempt's retained data has been copied out. Serial runs pass 0;
    /// concurrent runs pass every pool attempt index after each joined chunk.
    reset: *const fn (ptr: *anyopaque, attempt_index: u32) void,
};

/// `Evaluate.__call__` inputs seen by `on_evaluate_start`.
pub const EvaluateStart = struct {
    /// Process-unique and nonzero; repeated in the matching `EvaluateEnd`.
    /// Source uses a UUID string.
    call_id: u64,
    /// The borrowed `*const P` passed to the helper.
    program: *const anyopaque,
    dataset: []const Example,
    metric: ?optimize.Metric,
    /// Whether a pool runs the examples (Source `num_threads`).
    concurrent: bool,
    failure_score: f64,
    callback_metadata: ?*const Record,
};

/// Source `EvaluationResult`. Borrowed for the duration of the callback.
pub const EvaluateResult = struct {
    /// `errors` counts fallback rows, which add `failure_score` to `sum`. Other
    /// rows add the metric score (1 without a metric) or the failed
    /// prediction's format reward.
    evaluation: optimize.Evaluation,
    /// Captured rows in dataset order; fallback rows are absent.
    rows: []const TraceData,
    first_unhandled: ?Unhandled,
};

/// `on_evaluate_end(outputs, exception)`: exactly one of the fields is set.
pub const EvaluateEnd = struct {
    call_id: u64,
    result: ?EvaluateResult,
    err: ?anyerror,
};

/// Borrowed native counterpart of a `BaseCallback` handling Evaluate. Both
/// handlers run on the calling thread: start before any example, end after
/// every attempt has joined. As in Source, a handler error is logged as a
/// warning and otherwise ignored.
pub const Callback = struct {
    ptr: *anyopaque,
    on_evaluate_start: *const fn (ptr: *anyopaque, start: EvaluateStart) anyerror!void,
    on_evaluate_end: *const fn (ptr: *anyopaque, end: EvaluateEnd) anyerror!void,
};

/// One attempt's result. Children borrow the attempt scratch arena, so a pool
/// of these needs no per-program type and can be created at startup.
pub const Run = union(enum) {
    row: Row,
    unhandled: struct { stage: Stage, err: anyerror },
    /// Exhausted memory or cancellation: aborts the whole helper.
    fatal: anyerror,

    pub const Row = struct {
        prediction: Prediction,
        steps: []const predict.Trace.Step,
        /// Set for a parse failure: the predictor of the appended failed step.
        failed_predictor: ?*const PredictState,
        score: ?f64,
    };
};

/// `parallel.Parallel(Run, limits).runner()` plus per-attempt contexts. Each
/// context supplies that attempt's own scratch arena, LM and retriever; all
/// other settings come from the `ctx` argument. The program and metric must be
/// thread-safe. Speculative retries (`options.straggler_timeout_ns != 0`)
/// re-run a slow example, including its LM calls.
pub const Rollouts = parallel.Rollouts(Run);

pub const Options = struct {
    /// Receives each successful prediction and a null trace, as under Evaluate.
    /// Failed predictions bypass it and score their format reward.
    metric: ?optimize.Metric = null,
    /// Whether an unhandled row fails the helper with
    /// `error.UnpackPredictionAndTrace` (after every example has run, as in
    /// Source) or is skipped.
    raise_on_error: bool = true,
    failure_score: f64 = 0,
    format_failure_score: f64 = -1,
    /// Capture program errors other than parse failures as failed predictions.
    /// LM errors (`LMFailure`, `ScriptExhausted`) are never captured.
    capture_crashes: bool = false,
    /// Warn for every captured parse failure.
    log_format_failures: bool = false,
    /// Notified in order around the evaluation. A fallback row does not fail
    /// the evaluation: `UnpackPredictionAndTrace` is returned after the end
    /// handlers ran with a result. `EmptyDataset`, `OutOfMemory` and `Canceled`
    /// reach them as `err`.
    callbacks: []const Callback = &.{},
    /// Passed to `on_evaluate_start`; a non-empty record is also debug-logged.
    callback_metadata: ?*const Record = null,
    rollouts: ?Rollouts = null,
    scratch_reset: ?ScratchReset = null,
    /// Set to the first unhandled row in dataset order, or to null.
    unhandled: ?*?Unhandled = null,
};

/// Runs `program` over `dataset` and returns one row per example whose
/// prediction or failure was captured, in dataset order.
///
/// Ownership: rows, traces, predictions and completion texts are deep copies
/// in `out`, which is used only on the calling thread. Rows borrow `dataset`
/// (the example pointer and a failed step's top-level inputs) and the program's
/// predictors. `ctx.arena` is the serial attempt scratch and may be reset once
/// the helper returns. `program` is never modified.
///
/// Errors: `EmptyDataset`; `UnpackPredictionAndTrace` (see `raise_on_error`);
/// `OutOfMemory` and `Canceled` from any attempt, which are never captured.
pub fn bootstrapTraceData(
    comptime P: type,
    out: Allocator,
    ctx: Context,
    program: *const P,
    dataset: []const Example,
    options: Options,
) ![]TraceData {
    if (options.unhandled) |slot| slot.* = null;
    const call_id: u64 = if (options.callbacks.len > 0) next_call_id.fetchAdd(1, .monotonic) else 0;
    // An identifier is never reused: exhaustion must precede wraparound.
    assert(call_id != std.math.maxInt(u64));
    notifyStart(&options, .{
        .call_id = call_id,
        .program = program,
        .dataset = dataset,
        .metric = options.metric,
        .concurrent = options.rollouts != null,
        .failure_score = options.failure_score,
        .callback_metadata = options.callback_metadata,
    });
    const collector = evaluate(P, out, ctx, program, dataset, &options) catch |err| {
        notifyEnd(&options, .{ .call_id = call_id, .result = null, .err = err });
        return err;
    };
    const rows = collector.rows[0..collector.count];
    if (options.callbacks.len > 0) notifyEnd(&options, .{
        .call_id = call_id,
        .result = .{
            .evaluation = optimize.summarize(collector.sum, dataset.len, collector.unhandled_count),
            .rows = rows,
            .first_unhandled = collector.first_unhandled,
        },
        .err = null,
    });
    if (options.unhandled) |slot| slot.* = collector.first_unhandled;
    // Source unpacks rows after Evaluate returned: one warning per fallback
    // row reached, and raising stops at the first.
    const warnings = if (options.raise_on_error) @min(collector.unhandled_count, 1) else collector.unhandled_count;
    for (0..warnings) |_| log.warn("Failed to unpack prediction and trace. This is likely due to the LLM response not following dspy formatting.", .{});
    if (collector.first_unhandled != null and options.raise_on_error) return error.UnpackPredictionAndTrace;
    return rows;
}

fn notifyStart(options: *const Options, start: EvaluateStart) void {
    for (options.callbacks) |callback| {
        callback.on_evaluate_start(callback.ptr, start) catch |err| {
            log.warn("Error when calling callback on_evaluate_start: {s}", .{@errorName(err)});
        };
    }
}

fn notifyEnd(options: *const Options, end: EvaluateEnd) void {
    assert((end.result == null) != (end.err == null));
    for (options.callbacks) |callback| {
        callback.on_evaluate_end(callback.ptr, end) catch |err| {
            log.warn("Error when applying callback's end handler on function __call__: {s}.", .{@errorName(err)});
        };
    }
}

/// The patched program under `Evaluate`: every example runs, serially or on
/// the pool, before any fallback row is reported.
fn evaluate(
    comptime P: type,
    out: Allocator,
    ctx: Context,
    program: *const P,
    dataset: []const Example,
    options: *const Options,
) !Collector {
    if (dataset.len == 0) return error.EmptyDataset;
    if (options.callback_metadata) |metadata| {
        if (metadata.count() > 0) log.debug("Evaluate is called with callback metadata: {d} entries", .{metadata.count()});
    }
    // Lookup only compares predictor identity; nothing is written through it.
    const predictors = try predict.namedPredictors(out, @constCast(program));
    var collector: Collector = .{
        .out = out,
        .rows = try out.alloc(TraceData, dataset.len),
        .options = options,
    };
    const job: Job(P) = .{ .options = options, .predictors = predictors, .ctx = ctx, .program = program };
    if (options.rollouts) |rollouts| {
        try runConcurrent(P, &job, rollouts, dataset, &collector);
    } else {
        for (dataset, 0..) |*example, index| {
            try collector.consume(index, example, job.attempt(ctx, example));
            if (options.scratch_reset) |scratch| scratch.reset(scratch.ptr, 0);
        }
    }
    assert(collector.count + collector.unhandled_count <= dataset.len);
    return collector;
}

/// Retains joined attempt results in dataset order, on the calling thread.
const Collector = struct {
    out: Allocator,
    rows: []TraceData,
    count: usize = 0,
    first_unhandled: ?Unhandled = null,
    unhandled_count: usize = 0,
    /// Evaluate's score sum, accumulated in dataset order.
    sum: f64 = 0,
    options: *const Options,

    fn consume(self: *Collector, index: usize, example: *const Example, run: Run) !void {
        assert(self.count <= index);
        const options = self.options;
        switch (run) {
            .fatal => |err| return err,
            .unhandled => |failure| {
                if (self.first_unhandled == null) {
                    self.first_unhandled = .{ .example_ind = index, .stage = failure.stage, .err = failure.err };
                }
                self.unhandled_count += 1;
                self.sum += options.failure_score;
            },
            .row => |row| {
                switch (row.prediction) {
                    .success => self.sum += row.score orelse 1,
                    .failed => |failed| {
                        self.sum += failed.format_reward orelse options.format_failure_score;
                        self.warn(row.failed_predictor != null, failed);
                    },
                }
                // Rows after a raising failure are never returned; only an end
                // handler can still observe them.
                const raising = self.first_unhandled != null and options.raise_on_error;
                if (raising and options.callbacks.len == 0) return;
                self.rows[self.count] = try retain(self.out, index, example, row);
                self.count += 1;
            },
        }
    }

    fn warn(self: *const Collector, parse_failure: bool, failed: FailedPrediction) void {
        if (!parse_failure) {
            log.warn("Program raised on an example; capturing it as a failed prediction: {s}", .{failed.completion_text});
        } else if (self.options.log_format_failures) {
            log.warn("Failed to parse output for example. This is likely due to the LLM response not following the adapter's formatting.", .{});
        }
    }
};

fn retain(out: Allocator, index: usize, example: *const Example, row: Run.Row) value.ConvertError!TraceData {
    if (row.failed_predictor != null) assert(row.prediction == .failed);
    const extra: usize = if (row.failed_predictor != null) 1 else 0;
    const trace = try out.alloc(TraceStep, row.steps.len + extra);
    for (trace[0..row.steps.len], row.steps) |*step, *source| step.* = .{
        .predictor = source.predictor,
        .inputs = try value.cloneRecord(out, &source.inputs),
        .prediction = .{ .success = try value.cloneRecord(out, &source.outputs) },
    };
    // The attempt's Prediction lives in attempt scratch: an owning clone
    // keeps fields and usage (and objects they share) past its reset.
    const prediction: Prediction = switch (row.prediction) {
        .success => |source| .{ .success = try source.clone(out) },
        .failed => |failed| .{ .failed = .{
            .completion_text = try out.dupe(u8, failed.completion_text),
            .format_reward = failed.format_reward,
        } },
    };
    // Source records the top-level program kwargs, not the failing predictor's
    // own inputs, and the same FailedPrediction the row returns.
    if (row.failed_predictor) |predictor| trace[row.steps.len] = .{
        .predictor = predictor,
        .inputs = try example.inputs(out),
        .prediction = .{ .failed = prediction.failed },
    };
    return .{ .example_ind = index, .example = example, .prediction = prediction, .trace = trace, .score = row.score };
}

fn Job(comptime P: type) type {
    return struct {
        options: *const Options,
        predictors: []const predict.NamedParam,
        ctx: Context,
        program: *const P,

        const Self = @This();

        /// `process_item` under the patched forward. Allocates only from
        /// `ctx.arena`, which must be exclusive to this attempt.
        fn attempt(job: *const Self, ctx: Context, example: *const Example) Run {
            var trace: predict.Trace = .{ .arena = ctx.arena };
            var capture: predict.ParseFailure = .{};
            var c = ctx;
            c.trace = &trace;
            c.parse_failure = &capture;
            const prediction = optimize.runProgram(P, c, job.program, example) catch |err|
                return job.failed(err, &capture, trace.steps.items);
            const score: ?f64 = if (job.options.metric) |metric|
                metric(ctx.arena, example, &prediction, null) catch |err| switch (err) {
                    error.OutOfMemory => return .{ .fatal = err },
                    else => return .{ .unhandled = .{ .stage = .metric, .err = err } },
                }
            else
                null;
            return .{ .row = .{
                .prediction = .{ .success = prediction },
                .steps = trace.steps.items,
                .failed_predictor = null,
                .score = score,
            } };
        }

        fn failed(job: *const Self, err: anyerror, capture: *const predict.ParseFailure, steps: []const predict.Trace.Step) Run {
            const options = job.options;
            switch (err) {
                error.OutOfMemory, error.Canceled => return .{ .fatal = err },
                error.LMFailure, error.ScriptExhausted => return .{ .unhandled = .{ .stage = .program, .err = err } },
                error.AdapterParse => if (capture.signature_class) |class| return job.parseFailure(class, capture, steps),
                else => {},
            }
            if (!options.capture_crashes) return .{ .unhandled = .{ .stage = .program, .err = err } };
            return .{ .row = .{
                .prediction = .{ .failed = .{ .completion_text = @errorName(err) } },
                .steps = steps,
                .failed_predictor = null,
                .score = if (options.metric != null) options.format_failure_score else null,
            } };
        }

        fn parseFailure(job: *const Self, class: *const anyopaque, capture: *const predict.ParseFailure, steps: []const predict.Trace.Step) Run {
            const options = job.options;
            const present: usize = if (capture.failure.parsed_result) |parsed| parsed.count() else 0;
            const expected: usize = capture.expected_count;
            // First predictor with the failed signature class, in
            // `program.predictors()` order.
            const predictor: *const PredictState = for (job.predictors) |named| {
                assert(named.param == .predict);
                if (named.param.predict.signature_class == class) break named.param.predict;
            } else return .{ .unhandled = .{ .stage = .program, .err = error.FailedPredictorNotFound } };
            var reward = options.format_failure_score;
            if (present > 0) {
                if (expected == 0) return .{ .unhandled = .{ .stage = .program, .err = error.ZeroExpectedOutputs } };
                const fraction = @as(f64, @floatFromInt(present)) / @as(f64, @floatFromInt(expected));
                reward = options.format_failure_score + (options.failure_score - options.format_failure_score) * fraction;
            }
            return .{ .row = .{
                .prediction = .{ .failed = .{ .completion_text = capture.failure.completion_text, .format_reward = reward } },
                .steps = steps,
                .failed_predictor = predictor,
                .score = if (options.metric != null) reward else null,
            } };
        }
    };
}

/// Pool-sized chunks: every callback joins before its results are copied into
/// `out`, and before any attempt scratch is reset or reused.
fn runConcurrent(
    comptime P: type,
    job: *const Job(P),
    rollouts: Rollouts,
    dataset: []const Example,
    collector: *Collector,
) !void {
    const runner = rollouts.runner;
    assert(runner.tasks > 0);
    assert(rollouts.contexts.len >= runner.attempts);
    const Call = struct {
        job: *const Job(P),
        contexts: []const Context,
        example: *const Example,

        fn invoke(ptr: *anyopaque, attempt: parallel.Attempt, run: *Run) anyerror!void {
            const call: *const @This() = @ptrCast(@alignCast(ptr));
            assert(attempt.index < call.contexts.len);
            try attempt.cancellation.check();
            const own = call.contexts[attempt.index];
            var c = call.job.ctx;
            c.arena = own.arena;
            c.lm = own.lm;
            c.retriever = own.retriever;
            run.* = call.job.attempt(c, call.example);
        }
    };
    const chunk = @min(runner.tasks, dataset.len);
    const calls = try collector.out.alloc(Call, chunk);
    const tasks = try collector.out.alloc(parallel.TaskFor(Run), chunk);
    for (calls, tasks) |*call, *task| task.* = .{ .ptr = call, .invoke = Call.invoke };
    var start: usize = 0;
    while (start < dataset.len) {
        const count = @min(chunk, dataset.len - start);
        for (calls[0..count], dataset[start..][0..count]) |*call, *example| {
            call.* = .{ .job = job, .contexts = rollouts.contexts, .example = example };
        }
        const slots = try runner.forward(runner.ptr, tasks[0..count], rollouts.options);
        assert(slots.len == count);
        for (slots, 0..) |slot, offset| switch (slot) {
            // Callbacks absorb every error except a cancellation request.
            .failed => return error.Canceled,
            .value => |run| try collector.consume(start + offset, &dataset[start + offset], run.*),
        };
        if (job.options.scratch_reset) |scratch| {
            for (0..runner.attempts) |index| scratch.reset(scratch.ptr, @intCast(index));
        }
        start += count;
    }
    assert(start == dataset.len);
}
