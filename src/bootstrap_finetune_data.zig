//! DSPy 3.5 BootstrapFinetune's `_prepare_finetune_data` and
//! `build_call_data_from_trace`, using the actual adapter training formatter.
//!
//! Each trace step owns the choice of predictor signature and demonstrations;
//! `pred_ind` selects its position, not its predictor identity. Containers use
//! the caller's arena; formatted leaves may borrow trace/predictor storage.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const adapter_mod = @import("adapter.zig");
const bootstrap_trace = @import("bootstrap_trace.zig");
const pyrandom = @import("pyrandom.zig");
const training = @import("training.zig");
const value = @import("value.zig");
const Value = @import("runtime_value.zig").Value;

pub const Limits = struct {
    /// Maximum emitted training records in one prepared group.
    max_rows: u32,
    /// All visited steps in metric-retained rows, even unselected positions.
    max_trace_steps: u32,
};

pub const Prepared = struct {
    data: []Value,
    format: training.Format,
};

/// Infer the format even when no rows survive, then format selected calls and
/// shuffle with a fresh Python-compatible Random(0), independently per group.
/// Zero limits permit empty groups. Resource failures are explicit rather than
/// truncating training data: FinetuneRowLimit and FinetuneTraceStepLimit.
/// A selected FailedPrediction is not a dictionary upstream: Chat/XML raise
/// on its missing `get` method when the signature has declared output fields.
pub fn prepare(
    arena: Allocator,
    io: std.Io,
    rows: []const bootstrap_trace.TraceData,
    pred_ind: ?usize,
    adapter: adapter_mod.Adapter,
    metric_present: bool,
    exclude_demos: bool,
    limits: Limits,
) !Prepared {
    // JSON inherits ChatAdapter, but TwoStepAdapter does not. In particular,
    // JSON's unimplemented formatter must not reject an empty prepared group.
    const format: training.Format = switch (adapter) {
        .chat, .xml, .json => .chat,
        .two_step => return error.CannotInferFinetuneFormat,
    };
    const count = try preparedCount(rows, pred_ind, metric_present, limits);
    const data = try arena.alloc(Value, count);
    var written: usize = 0;
    for (rows) |*row| {
        if (!try retainRow(row.score, metric_present)) continue;
        for (row.trace, 0..) |*step, trace_pred_ind| {
            if (pred_ind) |selected| {
                if (trace_pred_ind != selected) continue;
            }
            assert(written < data.len);
            data[written] = try prepareCall(arena, io, step, adapter, exclude_demos);
            written += 1;
        }
    }
    assert(written == data.len);
    assert(data.len <= limits.max_rows);
    var random = pyrandom.Random.init(0);
    random.shuffle(Value, data);
    return .{ .data = data, .format = format };
}

fn preparedCount(
    rows: []const bootstrap_trace.TraceData,
    pred_ind: ?usize,
    metric_present: bool,
    limits: Limits,
) !u32 {
    // The optimizer bounds input examples and teachers. This limit applies to
    // emitted records, so filtered rows and empty traces consume no capacity.
    var visited: u32 = 0;
    var selected_count: u32 = 0;
    for (rows) |*row| {
        if (!try retainRow(row.score, metric_present)) continue;
        // Subtract before adding so even adversarial slice lengths cannot
        // overflow either the per-group count or an intermediate sum.
        assert(visited <= limits.max_trace_steps);
        if (row.trace.len > limits.max_trace_steps - visited) {
            return error.FinetuneTraceStepLimit;
        }
        visited += @intCast(row.trace.len);
        const selected: usize = if (pred_ind) |index|
            @intFromBool(index < row.trace.len)
        else
            row.trace.len;
        assert(selected_count <= limits.max_rows);
        if (selected > limits.max_rows - selected_count) return error.FinetuneRowLimit;
        selected_count += @intCast(selected);
    }
    assert(selected_count <= visited);
    assert(selected_count <= limits.max_rows);
    return selected_count;
}

fn retainRow(score: ?f64, metric_present: bool) !bool {
    // Native TraceData null means the score KEY is absent, not Python None.
    // With a metric Source indexes that key; without one it never reads it.
    if (!metric_present) return true;
    const number = score orelse return error.MissingFinetuneScore;
    // Negative values and NaN are truthy, unlike either signed zero.
    return number != 0;
}

fn prepareCall(
    arena: Allocator,
    io: std.Io,
    step: *const bootstrap_trace.TraceStep,
    adapter: adapter_mod.Adapter,
    exclude_demos: bool,
) !Value {
    assert(adapter != .two_step);
    const view = try step.predictor.view(arena);
    const demos = if (exclude_demos) &.{} else step.predictor.demos;
    switch (step.prediction) {
        .success => |outputs| {
            return adapter.formatFinetuneData(arena, io, view, demos, &step.inputs, &outputs);
        },
        .failed => {
            // JSON raises NotImplementedError before looking at outputs.
            // Chat/XML render the prompt first, then try outputs.get once per
            // declared output field. Preserve prompt errors before this error;
            // never turn the dataclass attributes into invented answer keys.
            if (adapter != .json and view.outputs.len != 0) {
                _ = try adapter.format(arena, view, demos, &step.inputs);
                return error.FailedPredictionNotMapping;
            }
            const outputs: value.Record = .empty;
            assert(adapter == .json or view.outputs.len == 0);
            return adapter.formatFinetuneData(arena, io, view, demos, &step.inputs, &outputs);
        },
    }
}

test "empty finetune groups infer chat for JSON but refuse TwoStep" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const limits: Limits = .{ .max_rows = 0, .max_trace_steps = 0 };
    inline for (.{ .chat, .xml, .json }) |adapter| {
        const result = try prepare(
            memory.allocator(),
            std.testing.io,
            &.{},
            null,
            adapter,
            true,
            false,
            limits,
        );
        try std.testing.expectEqual(training.Format.chat, result.format);
        try std.testing.expectEqual(@as(usize, 0), result.data.len);
    }
    try std.testing.expectError(error.CannotInferFinetuneFormat, prepare(
        memory.allocator(),
        std.testing.io,
        &.{},
        null,
        .two_step,
        false,
        false,
        limits,
    ));
}

test "failed finetune outputs are nonmapping except for zero-output signatures" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const fields = [_]@import("signature.zig").Field{.{
        .name = "completion_text",
        .kind = .str,
        .type_name = "str",
        .desc = "${completion_text}",
        .prefix = "Completion Text:",
    }};
    const identity: u8 = 0;
    var predictor: @import("predict.zig").PredictState = .{
        .signature_class = &identity,
        .fields = &fields,
        .input_count = 0,
        .instructions = "",
        .descs = &.{"${completion_text}"},
        .prefixes = &.{"Completion Text:"},
    };
    const step: bootstrap_trace.TraceStep = .{
        .predictor = &predictor,
        .inputs = .empty,
        .prediction = .{ .failed = .{ .completion_text = "not an answer" } },
    };
    const example: @import("optimize.zig").Example = .{
        .record = .empty,
        .input_keys = &.{},
    };
    const rows = [_]bootstrap_trace.TraceData{.{
        .example_ind = 0,
        .example = &example,
        .prediction = .{ .failed = step.prediction.failed },
        .trace = &.{step},
        .score = null,
    }};
    const limits: Limits = .{ .max_rows = 1, .max_trace_steps = 1 };
    // A dataclass attribute named exactly like an output is still not a key.
    inline for (.{ .chat, .xml }) |adapter| {
        try std.testing.expectError(error.FailedPredictionNotMapping, prepare(
            memory.allocator(),
            std.testing.io,
            &rows,
            null,
            adapter,
            false,
            false,
            limits,
        ));
    }
    try std.testing.expectError(error.FinetuneNotImplemented, prepare(
        memory.allocator(),
        std.testing.io,
        &rows,
        null,
        .json,
        false,
        false,
        limits,
    ));
    predictor.fields = &.{};
    predictor.descs = &.{};
    predictor.prefixes = &.{};
    inline for (.{ .chat, .xml }) |adapter| {
        const result = try prepare(
            memory.allocator(),
            std.testing.io,
            &rows,
            null,
            adapter,
            false,
            false,
            limits,
        );
        try std.testing.expectEqual(@as(usize, 1), result.data.len);
        try std.testing.expectEqual(@as(usize, 1), result.data[0].object.count());
        const messages = result.data[0].object.get("messages").?.array.items;
        try std.testing.expectEqualStrings(
            "assistant",
            messages[messages.len - 1].object.get("role").?.string,
        );
    }
}
