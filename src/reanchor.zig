//! `dspy.experimental.ReAnchor` (DSPy 3.5.0 `teleprompt/reanchor`): fit every
//! Noul threshold, Score cut, and Choice weight of a program's predictors
//! against the metric, keeping a fitted setting only when it beats the
//! original behavior under the same fold check.
//!
//! Native boundaries, all deliberate:
//! - The student copy shares its borrowed LM handles, as every native program
//!   copy does; only `decision_fields` is copied and changed.
//! - The evidence-observation pass runs serially in `ctx` so the scoped
//!   `DecisionEvidence` collector sees every call. Scoring passes use the
//!   configured `concurrent_evaluation` when present.
//! - `require_cache` checks the effective LM and predictor cache override.
//!   Concurrent scoring must use the same LM for each decision predictor as
//!   serial evidence collection, with caching enabled in both paths.
//! - Evidence and parameter numbers must be ints, floats, or builtin float
//!   objects. Decimal or arbitrary-precision integers fail with
//!   `error.UnsupportedNumber` rather than being approximated.
//! - Optional `log_dir` writes `report.json` atomically. `Report.writeJson`
//!   also emits the report to a caller-owned writer.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const decision_state = @import("decision_state.zig");
const lm_mod = @import("lm.zig");
const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const python = @import("python_value.zig");
const runtime = @import("runtime_value.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const ObjectMap = runtime.ObjectMap;
const PredictState = predict.PredictState;
const Value = runtime.Value;

/// Candidate settings per parameter; more distinct values thin to quantiles.
pub const max_candidates = 40;
/// The smallest and largest Choice multiplier tried.
pub const weight_range = [2]f64{ 1e-3, 1e3 };
/// Training-set parts the fold check holds out in turn.
pub const fold_count = 5;

pub const Parameter = enum { threshold, weights, cuts };

pub const FoldCheck = struct { passed: usize = 0, failed: usize = 0 };

/// Source `observed`: `{calls, distinct, min, max, candidates}` for
/// thresholds and cuts (only `calls` without evidence), `{calls,
/// candidates}` for weights.
pub const Observed = struct {
    calls: usize,
    distinct: ?usize = null,
    min: ?Value = null,
    max: ?Value = null,
    candidates: usize,
};

/// One output's report row. `value` is null exactly when `skipped` is set.
pub const Row = struct {
    predictor: []const u8,
    field: []const u8,
    parameter: Parameter,
    value: ?Value,
    train_score: f64,
    train_score_at_start: f64,
    observed: Observed,
    fold_check: FoldCheck,
    train_score_original: f64,
    skipped: ?[]const u8 = null,
};

pub const Report = struct {
    train_score_before: f64,
    train_score: f64,
    fitted: []const Row,
    val_score_before: ?f64 = null,
    val_score: ?f64 = null,

    /// Source's report dictionary, keys in Source order.
    pub fn toValue(self: *const Report, arena: Allocator) Allocator.Error!Value {
        var rows: runtime.Array = .init(arena);
        try rows.ensureTotalCapacity(self.fitted.len);
        for (self.fitted) |*row| rows.appendAssumeCapacity(try rowValue(arena, row));
        var report: ObjectMap = .empty;
        try report.put(arena, "train_score_before", .{ .float = self.train_score_before });
        try report.put(arena, "train_score", .{ .float = self.train_score });
        try report.put(arena, "fitted", .{ .array = rows });
        if (self.val_score_before) |score| try report.put(arena, "val_score_before", .{ .float = score });
        if (self.val_score) |score| try report.put(arena, "val_score", .{ .float = score });
        return .{ .object = report };
    }

    /// `json.dumps(report, indent=2)`, the Source `log_dir/report.json` text.
    pub fn writeJson(self: *const Report, arena: Allocator, writer: *std.Io.Writer) !void {
        try pyjson.writeJsonAsciiIndented(writer, try self.toValue(arena));
    }
};

fn rowValue(arena: Allocator, row: *const Row) Allocator.Error!Value {
    var observed: ObjectMap = .empty;
    try observed.put(arena, "calls", .{ .integer = @intCast(row.observed.calls) });
    if (row.observed.distinct) |distinct| try observed.put(arena, "distinct", .{ .integer = @intCast(distinct) });
    if (row.observed.min) |min| try observed.put(arena, "min", min);
    if (row.observed.max) |max| try observed.put(arena, "max", max);
    try observed.put(arena, "candidates", .{ .integer = @intCast(row.observed.candidates) });
    var check: ObjectMap = .empty;
    try check.put(arena, "passed", .{ .integer = @intCast(row.fold_check.passed) });
    try check.put(arena, "failed", .{ .integer = @intCast(row.fold_check.failed) });
    var out: ObjectMap = .empty;
    try out.put(arena, "predictor", .{ .string = row.predictor });
    try out.put(arena, "field", .{ .string = row.field });
    try out.put(arena, "parameter", .{ .string = @tagName(row.parameter) });
    if (row.value) |setting| try out.put(arena, "value", setting);
    try out.put(arena, "train_score", .{ .float = row.train_score });
    try out.put(arena, "train_score_at_start", .{ .float = row.train_score_at_start });
    try out.put(arena, "observed", .{ .object = observed });
    try out.put(arena, "fold_check", .{ .object = check });
    try out.put(arena, "train_score_original", .{ .float = row.train_score_original });
    if (row.skipped) |reason| try out.put(arena, "skipped", .{ .string = reason });
    return .{ .object = out };
}

pub const ReAnchor = struct {
    /// Per-example metric to maximize; every value must be finite.
    metric: optimize.Metric,
    /// Synchronizes the evidence collector.
    io: std.Io,
    /// DSPy `num_threads`: scoring passes use this bound evaluator.
    concurrent_evaluation: ?optimize.ConcurrentEvaluation = null,
    /// Require every decision predictor's bound or settings LM to cache.
    require_cache: bool = true,
    /// Optional caller-trusted directory for report.json. An empty path disables logging.
    log_dir: ?[]const u8 = null,
    /// Published after training scoring even if validation or logging then fails.
    /// Lives in compile's `out`, including on those failures.
    report: ?Report = null,
    /// Predictor names of the last `error.UncachedPredictors`, in `out`.
    uncached: []const []const u8 = &.{},

    /// Return a calibrated copy; the student is unchanged. Results and the
    /// report live in `out`; `gpa` backs temporary work and is fully freed.
    pub fn compile(
        self: *ReAnchor,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        student: *const P,
        trainset: []const Example,
        valset: ?[]const Example,
    ) !P {
        if (trainset.len == 0) return error.EmptyTrainset;
        var work_memory: std.heap.ArenaAllocator = .init(gpa);
        defer work_memory.deinit();
        const work = work_memory.allocator();

        var program = student.*;
        const all = try predict.namedPredictors(work, &program);
        for (all) |item| {
            const state = item.param.predict;
            state.decision_fields = try value.cloneValue(work, state.decision_fields);
        }
        var found: std.ArrayList(predict.NamedParam) = .empty;
        for (all) |item| {
            if (hasDecisionOutput(item.param.predict)) try found.append(work, item);
        }
        if (found.items.len == 0) return error.NoDecisionPredictors;
        if (self.concurrent_evaluation) |bound| {
            if (bound.program_type != runtime.typeIdentity(P)) return error.ProgramTypeMismatch;
        }
        if (self.require_cache) {
            const scoring_contexts = if (self.concurrent_evaluation) |bound|
                (bound.contexts orelse return error.UnverifiableEvaluationContexts)(bound.ptr)
            else
                &.{};
            var uncached: std.ArrayList([]const u8) = .empty;
            for (found.items) |item| {
                const state = item.param.predict;
                const model = state.effectiveLM(ctx);
                if (model.sameInstance(lm_mod.LM.unconfigured)) return error.NoLM;
                var uses_cache = model.cachesResponses() and (state.configuredOptions(ctx).cache orelse true);
                for (scoring_contexts) |scoring_ctx| {
                    if (!model.sameInstance(state.effectiveLM(scoring_ctx))) return error.EvaluationLMMismatch;
                    if (state.configuredOptions(scoring_ctx).cache == false) uses_cache = false;
                }
                if (!uses_cache) try uncached.append(out, try out.dupe(u8, item.name));
            }
            if (uncached.items.len != 0) {
                self.uncached = uncached.items;
                return error.UncachedPredictors;
            }
        }

        const runner: Runner(P) = .{ .self = self, .gpa = gpa, .ctx = ctx, .program = &program };
        const before = pyRound(try runner.run(work, trainset), 4);
        const has_valset = if (valset) |examples| examples.len != 0 else false;
        const val_before = if (has_valset) pyRound(try runner.run(work, valset.?), 4) else null;
        const fitted = try calibrate(P, runner, out, work, found.items, trainset);
        self.report = .{
            .train_score_before = before,
            .train_score = pyRound(try runner.run(work, trainset), 4),
            .fitted = fitted,
        };
        if (has_valset) {
            const val_after = pyRound(try runner.run(work, valset.?), 4);
            self.report.?.val_score_before = val_before;
            self.report.?.val_score = val_after;
        }
        try self.saveReport(work);
        for (all) |item| {
            const state = item.param.predict;
            state.decision_fields = try value.cloneValue(out, state.decision_fields);
        }
        if (@hasField(P, "compiled")) program.compiled = true;
        return program;
    }

    fn saveReport(self: *const ReAnchor, work: Allocator) !void {
        const dir = self.log_dir orelse return;
        if (dir.len == 0) return;
        const path = try std.fs.path.join(work, &.{ dir, "report.json" });
        var file = try std.Io.Dir.cwd().createFileAtomic(self.io, path, .{
            .make_path = true,
            .replace = true,
        });
        defer file.deinit(self.io);
        var buffer: [4096]u8 = undefined;
        var writer = file.file.writer(self.io, &buffer);
        try self.report.?.writeJson(work, &writer.interface);
        try writer.flush();
        try file.replace(self.io);
    }
};

fn hasDecisionOutput(state: *const PredictState) bool {
    for (state.fields[state.input_count..]) |field| {
        if (field.decision != null) return true;
    }
    return false;
}

fn Runner(comptime P: type) type {
    return struct {
        self: *const ReAnchor,
        gpa: Allocator,
        ctx: Context,
        program: *const P,

        const Self = @This();

        /// Source `scores`: any program or metric error fails the pass.
        fn scores(runner: Self, arena: Allocator, examples: []const Example) ![]f64 {
            const result = try arena.alloc(f64, examples.len);
            if (runner.self.concurrent_evaluation) |bound| {
                _ = try bound.run(bound.ptr, runner.program, examples, runner.self.metric, 1, result);
            } else try serialScores(P, runner.gpa, runner.ctx, runner.program, examples, runner.self.metric, result);
            for (result) |score| if (!std.math.isFinite(score)) return error.NonFiniteMetric;
            return result;
        }

        /// Source `run`: `math.fsum(scores) / len(scores)`.
        fn run(runner: Self, arena: Allocator, examples: []const Example) !f64 {
            const values = try runner.scores(arena, examples);
            return try fsum(values) / @as(f64, @floatFromInt(values.len));
        }

        /// The evidence each training call decodes `field` from, in one
        /// serial pass at the current setting.
        fn observe(runner: Self, arena: Allocator, state: *const PredictState, field: []const u8, trainset: []const Example) ![]const Value {
            var collector: predict.DecisionEvidence = .{ .arena = arena, .io = runner.self.io };
            var ctx = runner.ctx;
            ctx.decision_evidence = &collector;
            const result = try arena.alloc(f64, trainset.len);
            try serialScores(P, runner.gpa, ctx, runner.program, trainset, runner.self.metric, result);
            for (result) |score| if (!std.math.isFinite(score)) return error.NonFiniteMetric;
            var evidence: std.ArrayList(Value) = .empty;
            for (collector.items.items) |entry| {
                if (entry.predictor == state and std.mem.eql(u8, entry.output, field)) try evidence.append(arena, entry.evidence);
            }
            return evidence.items;
        }
    };
}

fn serialScores(
    comptime P: type,
    gpa: Allocator,
    ctx: Context,
    program: *const P,
    examples: []const Example,
    metric: optimize.Metric,
    result: []f64,
) !void {
    assert(result.len == examples.len);
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const detailed = try optimize.evaluateDetailed(P, scratch.allocator(), ctx, program, examples, metric, .{ .max_errors = 1 });
    for (result, detailed.rows) |*score, row| score.* = row.score;
}

const skipped_reason = "fitted behavior did not beat the original";

fn calibrate(
    comptime P: type,
    runner: Runner(P),
    out: Allocator,
    work: Allocator,
    found: []const predict.NamedParam,
    trainset: []const Example,
) ![]const Row {
    var rows: std.ArrayList(Row) = .empty;
    for (found) |item| {
        const state = item.param.predict;
        if (state.decision_fields != .object) return error.InvalidDecision;
        const view = try state.view(work);
        for (view.outputs) |field| {
            const binding = field.decision orelse continue;
            const original: ?Value = if (state.decision_fields.object.get(field.name)) |entry|
                try value.cloneValue(work, entry)
            else
                null;
            const before = try runner.scores(work, trainset);
            try state.decision_fields.object.put(work, field.name, try effective(work, state, view, field.name));
            const evidence = try runner.observe(work, state, field.name, trainset);
            const base = try runner.scores(work, trainset);
            const config = &state.decision_fields.object.getPtr(field.name).?.object;
            const fit = switch (binding.model.kind) {
                .noul => try fitThreshold(P, runner, work, config, evidence, trainset, base),
                .choice => try fitWeights(P, runner, work, config, binding.model.labels, evidence, trainset, base),
                .score => try fitCuts(P, runner, work, config, binding.model.labels.len, evidence, trainset, base),
            };
            var row: Row = .{
                .predictor = try out.dupe(u8, item.name),
                .field = try out.dupe(u8, field.name),
                .parameter = switch (binding.model.kind) {
                    .noul => .threshold,
                    .choice => .weights,
                    .score => .cuts,
                },
                .value = try value.cloneValue(out, fit.best),
                .train_score = try mean(fit.values),
                .train_score_at_start = try mean(base),
                .observed = .{
                    .calls = fit.observed.calls,
                    .distinct = fit.observed.distinct,
                    .min = fit.observed.min,
                    .max = fit.observed.max,
                    .candidates = fit.observed.candidates,
                },
                .fold_check = fit.check,
                .train_score_original = try mean(before),
            };
            const only: [1]Tried = .{.{ .scores = fit.values, .key = .{}, .setting = .null }};
            if ((try select(work, before, &only)).kept == null) {
                if (original) |entry| {
                    try state.decision_fields.object.put(work, field.name, entry);
                } else {
                    const removed = state.decision_fields.object.orderedRemove(field.name);
                    assert(removed);
                }
                row.value = null;
                row.skipped = skipped_reason;
                row.train_score = try mean(before);
            }
            try rows.append(out, row);
        }
    }
    return rows.items;
}

/// Source `effective`: stored overrides on top of the type's defaults.
fn effective(arena: Allocator, state: *const PredictState, view: @import("adapter.zig").View, field: []const u8) !Value {
    var configs = try value.cloneValue(arena, state.decision_fields);
    if (configs != .object) return error.InvalidDecision;
    if (!configs.object.contains(field)) try configs.object.put(arena, field, .{ .object = .empty });
    const snapshot = try decision_state.State.init(arena, view, configs, false, null);
    const config = snapshot.fields.get(field) orelse return error.InvalidDecision;
    if (config != .object) return error.InvalidDecision;
    return config;
}

/// A Python int or float, with its original representation for reporting.
const Num = struct {
    f: f64,
    value: Value,

    fn float(f: f64) Num {
        return .{ .f = f, .value = .{ .float = f } };
    }
};

fn number(v: Value) error{UnsupportedNumber}!Num {
    if (python.view(v)) |builtin| {
        if (builtin == .float) return .{ .f = builtin.float, .value = v };
        return error.UnsupportedNumber;
    }
    return switch (v) {
        .integer => |i| .{ .f = @floatFromInt(i), .value = v },
        .float => |f| .{ .f = f, .value = v },
        else => error.UnsupportedNumber,
    };
}

/// Python `round(number, 4)`: ints stay ints.
fn roundedValue(n: Num) Value {
    return if (n.value == .integer) n.value else .{ .float = pyRound(n.f, 4) };
}

const Summary = struct {
    calls: usize,
    distinct: ?usize = null,
    min: ?Value = null,
    max: ?Value = null,
    candidates: usize = 0,
};

/// Source `_summary`, before `candidates` is added.
fn summary(arena: Allocator, values: []const Num) Allocator.Error!Summary {
    const distinct = try sortedSet(arena, values);
    if (distinct.len == 0) return .{ .calls = 0 };
    return .{
        .calls = values.len,
        .distinct = distinct.len,
        .min = roundedValue(distinct[0]),
        .max = roundedValue(distinct[distinct.len - 1]),
    };
}

/// `sorted(set(values))`: the first occurrence represents equal numbers.
fn sortedSet(arena: Allocator, values: []const Num) Allocator.Error![]Num {
    var distinct: std.ArrayList(Num) = .empty;
    outer: for (values) |item| {
        for (distinct.items) |seen| if (seen.f == item.f) continue :outer;
        try distinct.append(arena, item);
    }
    std.sort.insertion(Num, distinct.items, {}, struct {
        fn less(_: void, a: Num, b: Num) bool {
            return a.f < b.f;
        }
    }.less);
    return distinct.items;
}

const Fit = struct {
    best: Value,
    values: []const f64,
    observed: Summary,
    check: FoldCheck,
};

/// Tie-break key; weights and cuts use only `width`.
const Key = struct {
    width: f64 = 0,
    distance: f64 = 0,
    t: f64 = 0,

    fn order(a: Key, b: Key) std.math.Order {
        inline for (.{ "width", "distance", "t" }) |name| {
            const x = @field(a, name);
            const y = @field(b, name);
            if (x < y) return .lt;
            if (x > y) return .gt;
        }
        return .eq;
    }
};

const Tried = struct {
    scores: []const f64,
    key: Key,
    setting: Value,
};

fn fitThreshold(
    comptime P: type,
    runner: Runner(P),
    work: Allocator,
    config: *ObjectMap,
    evidence: []const Value,
    trainset: []const Example,
    base: []const f64,
) !Fit {
    const start = try number(config.get("threshold") orelse return error.InvalidDecision);
    const probabilities = try work.alloc(Num, evidence.len);
    const raw = try work.alloc(f64, evidence.len);
    for (probabilities, raw, evidence) |*p, *f, e| {
        if (e != .object) return error.InvalidDecision;
        p.* = try number(e.object.get("noul") orelse return error.InvalidDecision);
        f.* = p.f;
    }
    var candidates: std.ArrayList(struct { setting: Value, t: f64, width: f64 }) = .empty;
    for (try gaps(work, raw, 0.0, 1.0, false)) |gap| {
        try candidates.append(work, .{ .setting = .{ .float = gap.x }, .t = gap.x, .width = gap.width });
    }
    const distinct = try sortedSet(work, probabilities);
    if (distinct.len > 1) {
        for (distinct[0 .. distinct.len - 1], distinct[1..]) |lower, upper| {
            // No interior float exists, but p >= upper still separates the pair.
            if (std.math.nextAfter(f64, lower.f, upper.f) == upper.f)
                try candidates.append(work, .{ .setting = upper.value, .t = upper.f, .width = upper.f - lower.f });
        }
    }
    if (std.mem.indexOfScalar(f64, raw, 0.0) != null) {
        // P(True) >= threshold makes zero a distinct outcome.
        try candidates.append(work, .{ .setting = .{ .float = 0.0 }, .t = 0.0, .width = 0.0 });
    }
    const tried = try work.alloc(Tried, candidates.items.len);
    for (tried, candidates.items) |*entry, candidate| {
        try config.put(work, "threshold", candidate.setting);
        entry.* = .{
            .scores = try runner.scores(work, trainset),
            .key = .{ .width = candidate.width, .distance = -@abs(candidate.t - start.f), .t = candidate.t },
            .setting = candidate.setting,
        };
    }
    const selection = try select(work, base, tried);
    const best = if (selection.kept) |index| tried[index].setting else start.value;
    try config.put(work, "threshold", best);
    var observed = try summary(work, probabilities);
    observed.candidates = tried.len;
    return .{
        .best = best,
        .values = if (selection.kept) |index| tried[index].scores else base,
        .observed = observed,
        .check = .{ .passed = @intFromBool(selection.kept != null), .failed = @intFromBool(selection.refused) },
    };
}

fn fitWeights(
    comptime P: type,
    runner: Runner(P),
    work: Allocator,
    config: *ObjectMap,
    labels: []const []const u8,
    evidence: []const Value,
    trainset: []const Example,
    base: []const f64,
) !Fit {
    const configured = config.get("weights") orelse return error.InvalidDecision;
    if (configured != .object) return error.InvalidDecision;
    var best: ObjectMap = .empty;
    for (labels) |label| {
        try best.put(work, label, if (configured.object.get(label)) |weight| (try number(weight)).value else .{ .float = 1.0 });
    }
    var best_scores = base;
    var candidates: usize = 0;
    var check: FoldCheck = .{};
    var flips: std.ArrayList(f64) = .empty;
    for (labels) |label| {
        flips.clearRetainingCapacity();
        for (evidence) |e| {
            if (e != .object) return error.InvalidDecision;
            const p = e.object.get("probabilities") orelse return error.InvalidDecision;
            if (p != .object) return error.InvalidDecision;
            var rival: ?f64 = null;
            for (labels) |other| {
                if (std.mem.eql(u8, other, label)) continue;
                const weighted = (try number(best.get(other).?)).f * (try number(p.object.get(other) orelse return error.InvalidDecision)).f;
                if (rival == null or weighted > rival.?) rival = weighted;
            }
            // Source max() over no rivals raises.
            const strongest = rival orelse return error.InvalidDecision;
            const own = (try number(p.object.get(label) orelse return error.InvalidDecision)).f;
            if (own > 0 and strongest > 0) try flips.append(work, strongest / own);
        }
        var low = weight_range[0];
        var high = weight_range[1];
        if (flips.items.len != 0) {
            low = @max(low, std.mem.min(f64, flips.items) / 4);
            high = @min(high, std.mem.max(f64, flips.items) * 4);
        }
        var tried: std.ArrayList(Tried) = .empty;
        const current = (try number(best.get(label).?)).f;
        for (try gaps(work, flips.items, low, high, true)) |gap| {
            if (gap.x == current) continue;
            var candidate = try best.clone(work);
            try candidate.put(work, label, .{ .float = gap.x });
            try config.put(work, "weights", .{ .object = candidate });
            try tried.append(work, .{
                .scores = try runner.scores(work, trainset),
                .key = .{ .width = gap.width },
                .setting = .{ .object = candidate },
            });
        }
        candidates += tried.items.len;
        const selection = try select(work, best_scores, tried.items);
        check.passed += @intFromBool(selection.kept != null);
        check.failed += @intFromBool(selection.refused);
        if (selection.kept) |index| {
            best_scores = tried.items[index].scores;
            best = tried.items[index].setting.object;
        }
    }
    try config.put(work, "weights", .{ .object = best });
    return .{
        .best = .{ .object = best },
        .values = best_scores,
        .observed = .{ .calls = evidence.len, .candidates = candidates },
        .check = check,
    };
}

fn fitCuts(
    comptime P: type,
    runner: Runner(P),
    work: Allocator,
    config: *ObjectMap,
    levels: usize,
    evidence: []const Value,
    trainset: []const Example,
    base: []const f64,
) !Fit {
    assert(levels >= 1);
    const top: f64 = @floatFromInt(levels - 1);
    const means = try work.alloc(Num, evidence.len);
    const raw = try work.alloc(f64, evidence.len);
    for (means, raw, evidence) |*m, *f, e| {
        if (e != .object) return error.InvalidDecision;
        const distribution = python.view(e.object.get("probabilities") orelse return error.InvalidDecision) orelse
            return error.InvalidDecision;
        if (distribution != .dictionary) return error.InvalidDecision;
        // Builtin sum() adds left to right from integer zero.
        var weighted: f64 = 0;
        var mass: f64 = 0;
        for (distribution.dictionary) |entry| {
            if (entry.key != .integer) return error.InvalidDecision;
            const p = (try number(entry.value)).f;
            weighted += @as(f64, @floatFromInt(entry.key.integer)) * p;
            mass += p;
        }
        if (mass == 0) return error.ZeroProbabilityTotal;
        m.* = .float(weighted / mass);
        f.* = m.f;
    }
    const configured = config.get("cuts") orelse return error.InvalidDecision;
    if (configured != .array) return error.InvalidDecision;
    var best = try work.alloc(Value, configured.array.items.len);
    for (best, configured.array.items) |*dst, cut| dst.* = (try number(cut)).value;
    var best_scores = base;
    var candidates: usize = 0;
    var check: FoldCheck = .{};
    for (0..best.len) |i| {
        const below: f64 = if (i != 0) (try number(best[i - 1])).f else 0;
        const above: f64 = if (i + 1 < best.len) (try number(best[i + 1])).f else top;
        const current = (try number(best[i])).f;
        var tried: std.ArrayList(Tried) = .empty;
        for (try gaps(work, raw, below, above, false)) |gap| {
            if (gap.x == current) continue;
            const candidate = try work.dupe(Value, best);
            candidate[i] = .{ .float = gap.x };
            try config.put(work, "cuts", try arrayValue(work, candidate));
            try tried.append(work, .{
                .scores = try runner.scores(work, trainset),
                .key = .{ .width = gap.width },
                .setting = try arrayValue(work, candidate),
            });
        }
        candidates += tried.items.len;
        const selection = try select(work, best_scores, tried.items);
        check.passed += @intFromBool(selection.kept != null);
        check.failed += @intFromBool(selection.refused);
        if (selection.kept) |index| {
            best_scores = tried.items[index].scores;
            best = tried.items[index].setting.array.items;
        }
    }
    const setting = try arrayValue(work, best);
    try config.put(work, "cuts", setting);
    var observed = try summary(work, means);
    observed.candidates = candidates;
    return .{ .best = setting, .values = best_scores, .observed = observed, .check = check };
}

fn arrayValue(arena: Allocator, items: []const Value) Allocator.Error!Value {
    var array: runtime.Array = .init(arena);
    try array.appendSlice(items);
    return .{ .array = array };
}

pub const Gap = struct { x: f64, width: f64 };

/// Source `_gaps`: midpoints (and widths) of the gaps between neighbouring
/// distinct values strictly inside `(lo, hi)`, closed by `lo` and `hi`, with
/// more than `max_candidates - 1` values thinned to quantiles.
pub fn gaps(arena: Allocator, values: []const f64, lo: f64, hi: f64, geometric: bool) Allocator.Error![]const Gap {
    var inner: std.ArrayList(f64) = .empty;
    for (values) |v| if (lo < v and v < hi) try inner.append(arena, v);
    std.sort.pdq(f64, inner.items, {}, std.sort.asc(f64));
    inner.items.len = dedupSorted(inner.items);
    if (inner.items.len > max_candidates - 1) {
        const step = @as(f64, @floatFromInt(inner.items.len - 1)) / @as(f64, max_candidates - 2);
        var thinned: [max_candidates - 1]f64 = undefined;
        for (&thinned, 0..) |*item, i| item.* = inner.items[roundHalfEven(@as(f64, @floatFromInt(i)) * step)];
        const count = dedupSorted(&thinned);
        @memcpy(inner.items[0..count], thinned[0..count]);
        inner.items.len = count;
    }
    var result: std.ArrayList(Gap) = .empty;
    var a = lo;
    for (0..inner.items.len + 1) |i| {
        const b = if (i < inner.items.len) inner.items[i] else hi;
        const middle: f64 = if (geometric) @sqrt(a * b) else (a + b) / 2;
        const width: f64 = if (geometric) @log(b / a) else b - a;
        // Adjacent floats can round the midpoint onto a boundary.
        if (a < middle and middle < b) try result.append(arena, .{ .x = tidy(middle, a, b), .width = width });
        a = b;
    }
    return result.items;
}

fn dedupSorted(items: []f64) usize {
    if (items.len == 0) return 0;
    var count: usize = 1;
    for (items[1..]) |item| {
        if (item == items[count - 1]) continue;
        items[count] = item;
        count += 1;
    }
    return count;
}

/// Python `round(x)` for a non-negative float index.
fn roundHalfEven(x: f64) usize {
    assert(x >= 0 and std.math.isFinite(x));
    const floor = @floor(x);
    const diff = x - floor;
    var result: usize = @intFromFloat(floor);
    if (diff > 0.5 or (diff == 0.5 and result % 2 == 1)) result += 1;
    return result;
}

/// Source `_tidy`: the shortest decimal strictly between `lo` and `hi`,
/// rounded from `x`.
pub fn tidy(x: f64, lo: f64, hi: f64) f64 {
    for (1..12) |digits| {
        const rounded = pyRound(x, @intCast(digits));
        if (lo < rounded and rounded < hi) return rounded;
    }
    return x;
}

/// Python `round(x, digits)` for a float: the exact binary value rounded
/// half-even to `digits` decimals, then correctly rounded back to binary.
pub fn pyRound(x: f64, digits: u4) f64 {
    assert(digits <= 11);
    if (!std.math.isFinite(x) or x == 0) return x;
    const bits: u64 = @bitCast(x);
    const exponent_bits: u11 = @truncate(bits >> 52);
    const fraction: u64 = bits & ((@as(u64, 1) << 52) - 1);
    const mantissa: u64 = if (exponent_bits == 0) fraction else fraction | (@as(u64, 1) << 52);
    const exponent: i32 = (if (exponent_bits == 0) @as(i32, 1) else @as(i32, exponent_bits)) - 1075;
    // An integral value is unchanged at any positive number of digits.
    if (exponent >= 0) return x;
    const scaled: u128 = @as(u128, mantissa) * std.math.pow(u128, 10, digits);
    const shift: u32 = @intCast(-exponent);
    var quotient: u128 = 0;
    if (shift < 128) {
        const s: u7 = @intCast(shift);
        quotient = scaled >> s;
        const remainder = scaled - (quotient << s);
        const half = @as(u128, 1) << (s - 1);
        if (remainder > half or (remainder == half and quotient & 1 == 1)) quotient += 1;
    }
    var buffer: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}e-{d}", .{ quotient, digits }) catch unreachable;
    const magnitude = std.fmt.parseFloat(f64, text) catch unreachable;
    return std.math.copysign(magnitude, x);
}

pub const FsumError = error{ FsumOverflow, FsumInvalid };

/// Python `math.fsum`, including its intermediate-overflow error.
pub fn fsum(values: []const f64) FsumError!f64 {
    var sum: Fsum = .{};
    for (values) |item| try sum.add(item);
    return sum.result();
}

/// CPython's streaming Shewchuk summation behind `math.fsum`.
const Fsum = struct {
    /// Nonoverlapping partials occupy distinct binary64 exponent ranges, so
    /// at most ceil(2098 / 53) of them coexist.
    partials: [48]f64 = undefined,
    n: usize = 0,
    special: f64 = 0,
    inf_sum: f64 = 0,

    fn add(self: *Fsum, item: f64) FsumError!void {
        var x = item;
        var i: usize = 0;
        for (self.partials[0..self.n]) |partial| {
            var y = partial;
            if (@abs(x) < @abs(y)) std.mem.swap(f64, &x, &y);
            const hi = x + y;
            const lo = y - (hi - x);
            if (lo != 0) {
                self.partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        self.n = i;
        if (x == 0) return;
        if (!std.math.isFinite(x)) {
            if (std.math.isFinite(item)) return error.FsumOverflow;
            if (std.math.isInf(item)) self.inf_sum += item;
            self.special += item;
            self.n = 0;
        } else {
            assert(self.n < self.partials.len);
            self.partials[self.n] = x;
            self.n += 1;
        }
    }

    fn result(self: *const Fsum) FsumError!f64 {
        if (self.special != 0) {
            if (std.math.isNan(self.inf_sum)) return error.FsumInvalid;
            return self.special;
        }
        var n = self.n;
        var hi: f64 = 0;
        if (n > 0) {
            n -= 1;
            hi = self.partials[n];
            var lo: f64 = 0;
            while (n > 0) {
                const x = hi;
                n -= 1;
                const y = self.partials[n];
                hi = x + y;
                lo = y - (hi - x);
                if (lo != 0) break;
            }
            if (n > 0 and ((lo < 0 and self.partials[n - 1] < 0) or (lo > 0 and self.partials[n - 1] > 0))) {
                const y = lo * 2;
                const x = hi + y;
                if (y == x - hi) hi = x;
            }
        }
        return hi;
    }
};

fn mean(values: []const f64) !f64 {
    assert(values.len != 0);
    return pyRound(try fsum(values) / @as(f64, @floatFromInt(values.len)), 4);
}

/// Source `_folds`: example indexes split into up to `fold_count` parts.
pub fn folds(arena: Allocator, n: usize) Allocator.Error![]const []const usize {
    const order = try arena.alloc(usize, n);
    for (order, 0..) |*item, i| item.* = i;
    var random: pyrandom.Random = .init(0);
    random.shuffle(usize, order);
    const k = @min(fold_count, n);
    const parts = try arena.alloc([]const usize, k);
    for (parts, 0..) |*part, i| {
        const items = try arena.alloc(usize, (n - i + k - 1) / k);
        for (items, 0..) |*item, j| item.* = order[i + j * k];
        part.* = items;
    }
    return parts;
}

/// fsum of `values` over the rows not excluded, in index order.
fn total(values: []const f64, excluded: []const bool) FsumError!f64 {
    assert(values.len == excluded.len);
    var sum: Fsum = .{};
    for (values, excluded) |item, skip| {
        if (!skip) try sum.add(item);
    }
    return sum.result();
}

const Selection = struct { kept: ?usize, refused: bool };

/// Source `_pick`: the best entry on the included rows, when it beats `start` there.
fn pick(start: []const f64, tried: []const Tried, excluded: []const bool) !?usize {
    var best: ?usize = null;
    var best_total: f64 = 0;
    for (tried, 0..) |entry, index| {
        const sum = try total(entry.scores, excluded);
        if (best == null or sum > best_total or (sum == best_total and entry.key.order(tried[best.?].key) == .gt)) {
            best = index;
            best_total = sum;
        }
    }
    const index = best orelse return null;
    return if (best_total > try total(start, excluded)) index else null;
}

/// Source `_select`: the entry to keep, and whether the fold check refused
/// an entry that scored best on the whole training set.
fn select(arena: Allocator, start: []const f64, tried: []const Tried) !Selection {
    const excluded = try arena.alloc(bool, start.len);
    @memset(excluded, false);
    const best = try pick(start, tried, excluded) orelse return .{ .kept = null, .refused = false };
    var held: f64 = 0;
    var held_start: f64 = 0;
    for (try folds(arena, start.len)) |fold| {
        @memset(excluded, false);
        for (fold) |index| excluded[index] = true;
        const chosen = try pick(start, tried, excluded);
        const source = if (chosen) |index| tried[index].scores else start;
        var fold_sum: Fsum = .{};
        var fold_start: Fsum = .{};
        for (fold) |index| {
            try fold_sum.add(source[index]);
            try fold_start.add(start[index]);
        }
        held += try fold_sum.result();
        held_start += try fold_start.result();
    }
    return if (held > held_start) .{ .kept = best, .refused = false } else .{ .kept = null, .refused = true };
}

const testing = std.testing;

test "numeric helpers match Source round, fsum, _tidy, _gaps, and _folds" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    // Oracles printed by the pinned DSPy 3.5.0 calibrate module.
    const rounds = [_]struct { f64, u4, f64 }{
        .{ 2.675, 2, 2.67 },     .{ 0.30000000000000004, 4, 0.3 },
        .{ 1e-300, 4, 0.0 },     .{ 0.125, 2, 0.12 },
        .{ 0.12345, 4, 0.1235 }, .{ 123456.78915, 4, 123456.7891 },
    };
    for (rounds) |case| try testing.expectEqual(case[2], pyRound(case[0], case[1]));
    const negative_zero = pyRound(-0.00004, 4);
    try testing.expect(negative_zero == 0 and std.math.signbit(negative_zero));
    var tenths: [10]f64 = @splat(0.1);
    try testing.expectEqual(@as(f64, 1.0), try fsum(&tenths));
    try testing.expectEqual(@as(f64, 2.0), try fsum(&.{ 1, 1e100, 1, -1e100 }));
    try testing.expectError(error.FsumOverflow, fsum(&.{ 1e308, 1e308, -1e308 }));
    try testing.expectEqual(@as(f64, 0.2), try mean(&.{ 0.1, 0.2, 0.3 }));

    try testing.expectEqual(@as(f64, 0.45), tidy(0.45, 0.4, 0.5));
    try testing.expectEqual(@as(f64, 0.12345679), tidy(0.123456789, 0.1234567, 0.1234568));
    try testing.expectEqual(@as(f64, 0.5000000000000001), tidy(0.5000000000000001, 0.5, 0.5000000000000002));

    const plain = try gaps(arena, &.{ 0.2, 0.4, 0.4, 0.9, 0.0, 1.0 }, 0.0, 1.0, false);
    const expected_plain = [_]Gap{ .{ .x = 0.1, .width = 0.2 }, .{ .x = 0.3, .width = 0.2 }, .{ .x = 0.7, .width = 0.5 }, .{ .x = 0.95, .width = 0.09999999999999998 } };
    try testing.expectEqualSlices(Gap, &expected_plain, plain);
    var many: [96]f64 = undefined;
    for (&many, 1..) |*item, i| item.* = @as(f64, @floatFromInt(i)) / 97;
    const thinned = try gaps(arena, &many, 0.0, 1.0, false);
    const expected_thinned = [_]f64{ 0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.23, 0.25, 0.28, 0.3, 0.33, 0.36, 0.4, 0.41, 0.43, 0.46, 0.5, 0.52, 0.54, 0.56, 0.6, 0.62, 0.64, 0.66, 0.7, 0.72, 0.74, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 0.93, 0.95, 0.97, 0.99 };
    try testing.expectEqual(expected_thinned.len, thinned.len);
    for (expected_thinned, thinned) |x, gap| try testing.expectEqual(x, gap.x);
    try testing.expectEqual(@as(f64, 0.010309278350515427), thinned[thinned.len - 1].width);
    const geometric = try gaps(arena, &.{ 0.5, 2.0, 8.0, 0.01 }, 0.002, 40.0, true);
    const expected_geometric = [_]f64{ 0.004, 0.1, 1.0, 4.0, 17.9 };
    const expected_widths = [_]f64{ 1.6094379124341003, 3.912023005428146, 1.3862943611198906, 1.3862943611198906, 1.6094379124341003 };
    try testing.expectEqual(expected_geometric.len, geometric.len);
    for (expected_geometric, expected_widths, geometric) |x, width, gap| {
        try testing.expectEqual(x, gap.x);
        try testing.expectApproxEqRel(width, gap.width, 1e-15);
    }

    const fold_cases = [_]struct { usize, []const []const usize }{
        .{ 1, &.{&.{0}} },
        .{ 3, &.{ &.{0}, &.{2}, &.{1} } },
        .{ 7, &.{ &.{ 4, 3 }, &.{ 2, 6 }, &.{1}, &.{0}, &.{5} } },
        .{ 12, &.{ &.{ 1, 2, 11 }, &.{ 9, 3, 6 }, &.{ 8, 7 }, &.{ 5, 4 }, &.{ 10, 0 } } },
    };
    for (fold_cases) |case| {
        const actual = try folds(arena, case[0]);
        try testing.expectEqual(case[1].len, actual.len);
        for (case[1], actual) |want, got| try testing.expectEqualSlices(usize, want, got);
    }
}

test "selection keeps only gains that survive Source's fold check" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const start = [_]f64{ 0, 1, 0, 0, 1, 0, 1 };
    const a = [_]f64{ 1, 1, 0, 1, 1, 0, 1 };
    const b = [_]f64{ 1, 1, 1, 1, 1, 1, 0 };
    const both = [_]Tried{
        .{ .scores = &a, .key = .{ .width = 0.1 }, .setting = .{ .string = "a" } },
        .{ .scores = &b, .key = .{ .width = 0.2 }, .setting = .{ .string = "b" } },
    };
    try testing.expectEqual(Selection{ .kept = 1, .refused = false }, try select(arena, &start, &both));
    const zeros: [7]f64 = @splat(0);
    const one = [_]f64{ 1, 0, 0, 0, 0, 0, 0 };
    const single = [_]Tried{.{ .scores = &one, .key = .{ .width = 0.1 }, .setting = .null }};
    try testing.expectEqual(Selection{ .kept = null, .refused = true }, try select(arena, &zeros, &single));
    const same = [_]Tried{.{ .scores = &start, .key = .{ .width = 1 }, .setting = .null }};
    try testing.expectEqual(Selection{ .kept = null, .refused = false }, try select(arena, &start, &same));

    const values = [_]Num{ .float(0.123456), .{ .f = 1, .value = .{ .integer = 1 } }, .float(0.5), .float(1.0) };
    const observed = try summary(arena, &values);
    try testing.expectEqual(@as(usize, 4), observed.calls);
    try testing.expectEqual(@as(?usize, 3), observed.distinct);
    try testing.expectEqual(@as(f64, 0.1235), observed.min.?.float);
    try testing.expectEqual(@as(i64, 1), observed.max.?.integer);
    try testing.expectEqual(@as(usize, 0), (try summary(arena, &.{})).calls);
}

const signature = @import("signature.zig");
const decision = @import("decision.zig");
const json_repair = @import("json_repair.zig");

const NoulSig = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: decision.Noul };
    pub const descriptions = .{ .answer = "Is it yes?" };
});
const Category = decision.ChoiceWithOptions(&.{
    .{ .value = .{ .string = "billing" }, .description_json = "\"Payment issue\"" },
    .{ .value = .{ .string = "technical" }, .description_json = "\"Product bug\"" },
});
const ChoiceSig = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: Category };
    pub const descriptions = .{ .answer = "Category." };
});
const Rating = decision.ScoreWithLevels("[\"poor\", \"fair\", \"good\"]");
const ScoreSig = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: Rating };
    pub const descriptions = .{ .answer = "Rating." };
});

fn noulMetric(_: Allocator, example: *const Example, prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    const answer = prediction.?.get("answer").?.custom.get(decision.Noul).?;
    return @floatFromInt(@intFromBool(answer.value == example.record.get("label").?.bool));
}

fn choiceMetric(_: Allocator, example: *const Example, prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    const answer = prediction.?.get("answer").?.custom.get(Category).?;
    return @floatFromInt(@intFromBool(std.mem.eql(u8, answer.value.string, example.record.get("label").?.string)));
}

fn scoreMetric(_: Allocator, example: *const Example, prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
    const answer = prediction.?.get("answer").?.custom.get(Rating).?;
    return @floatFromInt(@intFromBool(answer.level.?.value.integer == example.record.get("label").?.integer));
}

fn examplesFrom(arena: Allocator, rows: []const []const u8) ![]Example {
    const result = try arena.alloc(Example, rows.len);
    for (result, rows) |*example, row| example.* = try Example.fromJson(arena, row, &.{"question"});
    return result;
}

fn compileCase(
    comptime Sig: type,
    arena: Allocator,
    optimizer: *ReAnchor,
    rules: []const lm_mod.KeywordLM.Rule,
    fields_json: ?[]const u8,
    trainset: []const Example,
    valset: ?[]const Example,
) !struct { student: predict.Predict(Sig), program: predict.Predict(Sig) } {
    var keyword: lm_mod.KeywordLM = .{ .rules = rules, .default = "<answer>{}</answer>" };
    var cache: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = keyword.lm() };
    defer cache.deinit();
    var student: predict.Predict(Sig) = .{};
    if (fields_json) |json| student.state.decision_fields = try json_repair.parseJson(arena, json);
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = cache.lm() };
    const program = try optimizer.compile(predict.Predict(Sig), arena, testing.allocator, ctx, &student, trainset, valset);
    return .{ .student = student, .program = program };
}

fn expectJson(arena: Allocator, expected: []const u8, actual: Value) !void {
    var text: std.Io.Writer.Allocating = .init(arena);
    try pyjson.writeJsonAscii(&text.writer, actual);
    try testing.expectEqualStrings(expected, text.written());
}

fn expectReport(arena: Allocator, expected: []const u8, report: Report) !void {
    var text: std.Io.Writer.Allocating = .init(arena);
    try report.writeJson(arena, &text.writer);
    try testing.expectEqualStrings(expected, text.written());
}

test "ReAnchor creates and replaces reports and preserves them on log failure" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const base = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    const logs = try std.fs.path.join(arena, &.{ base, "nested", "logs" });
    const trainset = try examplesFrom(arena, &.{
        "{\"question\":\"alpha\",\"label\":false}",
        "{\"question\":\"echo\",\"label\":true}",
    });
    var optimizer: ReAnchor = .{ .metric = noulMetric, .io = testing.io, .log_dir = logs };
    const report_path = "nested/logs/report.json";
    for ([_]?[]const Example{ trainset[0..1], trainset[0..0] }) |valset| {
        _ = try compileCase(NoulSig, arena, &optimizer, &noul_rules, null, trainset, valset);
        const saved = try temporary.dir.readFileAlloc(testing.io, report_path, arena, .limited(4096));
        var expected: std.Io.Writer.Allocating = .init(arena);
        try optimizer.report.?.writeJson(arena, &expected.writer);
        try testing.expectEqualStrings(expected.written(), saved);
        try testing.expectEqual(valset.?.len != 0, optimizer.report.?.val_score != null);
    }

    // Reuse the existing report file as a would-be directory. A completed
    // validation report is observable even though no compiled program returns.
    optimizer.log_dir = try std.fs.path.join(arena, &.{ base, report_path });
    try testing.expectError(
        error.NotDir,
        compileCase(NoulSig, arena, &optimizer, &noul_rules, null, trainset, trainset[0..1]),
    );
    try testing.expectEqual(@as(?f64, 1.0), optimizer.report.?.val_score);
}

const noul_rules = [_]lm_mod.KeywordLM.Rule{
    .{ .needle = "alpha", .response = "<answer>{\"noul\": 0.1}</answer>" },
    .{ .needle = "bravo", .response = "<answer>{\"noul\": 0.2}</answer>" },
    .{ .needle = "charlie", .response = "<answer>{\"noul\": 0.3}</answer>" },
    .{ .needle = "delta", .response = "<answer>{\"noul\": 0.4}</answer>" },
    .{ .needle = "echo", .response = "<answer>{\"noul\": 0.6}</answer>" },
    .{ .needle = "foxtrot", .response = "<answer>{\"noul\": 0.7}</answer>" },
    .{ .needle = "golf", .response = "<answer>{\"noul\": 0.8}</answer>" },
};

test "threshold calibration matches Source and leaves the student unchanged" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const rules = [_]lm_mod.KeywordLM.Rule{
        .{ .needle = "q0", .response = "<answer>{\"noul\": 0.2}</answer>" },
        .{ .needle = "q1", .response = "<answer>{\"noul\": 0.35}</answer>" },
        .{ .needle = "q2", .response = "<answer>{\"noul\": 0.4}</answer>" },
        .{ .needle = "q3", .response = "<answer>{\"noul\": 0.45}</answer>" },
        .{ .needle = "q4", .response = "<answer>{\"noul\": 0.6}</answer>" },
        .{ .needle = "q5", .response = "<answer>{\"noul\": 0.7}</answer>" },
        .{ .needle = "q6", .response = "<answer>{\"noul\": 0.3}</answer>" },
    };
    const trainset = try examplesFrom(arena, &.{
        "{\"question\": \"q0\", \"label\": false}", "{\"question\": \"q1\", \"label\": true}",
        "{\"question\": \"q2\", \"label\": true}",  "{\"question\": \"q3\", \"label\": true}",
        "{\"question\": \"q4\", \"label\": true}",  "{\"question\": \"q5\", \"label\": true}",
        "{\"question\": \"q6\", \"label\": false}",
    });
    var optimizer: ReAnchor = .{ .metric = noulMetric, .io = testing.io };
    const result = try compileCase(NoulSig, arena, &optimizer, &rules, null, trainset, null);
    try expectJson(arena, "{}", result.student.state.decision_fields);
    try expectJson(arena, "{\"answer\": {\"threshold\": 0.32}}", result.program.state.decision_fields);
    try testing.expect(result.program.compiled);
    try expectReport(arena,
        \\{
        \\  "train_score_before": 0.5714,
        \\  "train_score": 1.0,
        \\  "fitted": [
        \\    {
        \\      "predictor": "self",
        \\      "field": "answer",
        \\      "parameter": "threshold",
        \\      "value": 0.32,
        \\      "train_score": 1.0,
        \\      "train_score_at_start": 0.5714,
        \\      "observed": {
        \\        "calls": 7,
        \\        "distinct": 7,
        \\        "min": 0.2,
        \\        "max": 0.7,
        \\        "candidates": 8
        \\      },
        \\      "fold_check": {
        \\        "passed": 1,
        \\        "failed": 0
        \\      },
        \\      "train_score_original": 0.5714
        \\    }
        \\  ]
        \\}
    , optimizer.report.?);
}

test "unimproved threshold removes the absent field entry Source added" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const trainset = try examplesFrom(arena, &.{
        "{\"question\": \"alpha\", \"label\": true}",   "{\"question\": \"bravo\", \"label\": false}",
        "{\"question\": \"charlie\", \"label\": true}", "{\"question\": \"delta\", \"label\": false}",
        "{\"question\": \"echo\", \"label\": true}",    "{\"question\": \"foxtrot\", \"label\": false}",
        "{\"question\": \"golf\", \"label\": true}",
    });
    var optimizer: ReAnchor = .{ .metric = noulMetric, .io = testing.io };
    const result = try compileCase(NoulSig, arena, &optimizer, &noul_rules, null, trainset, null);
    try expectJson(arena, "{}", result.program.state.decision_fields);
    const row = optimizer.report.?.fitted[0];
    try testing.expect(row.value == null);
    try testing.expectEqualStrings(skipped_reason, row.skipped.?);
    try testing.expectEqual(FoldCheck{ .passed = 0, .failed = 0 }, row.fold_check);
    try testing.expectEqual(@as(f64, 0.5714), row.train_score);
    try testing.expectEqual(@as(usize, 8), row.observed.candidates);
}

test "fold-check refusal restores the original configured entry" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const trainset = try examplesFrom(arena, &.{
        "{\"question\": \"alpha\", \"label\": false}",   "{\"question\": \"bravo\", \"label\": false}",
        "{\"question\": \"charlie\", \"label\": false}", "{\"question\": \"delta\", \"label\": true}",
        "{\"question\": \"echo\", \"label\": true}",     "{\"question\": \"foxtrot\", \"label\": true}",
        "{\"question\": \"golf\", \"label\": true}",
    });
    var optimizer: ReAnchor = .{ .metric = noulMetric, .io = testing.io };
    const result = try compileCase(NoulSig, arena, &optimizer, &noul_rules, "{\"answer\": {\"threshold\": 0.55}}", trainset, null);
    try expectJson(arena, "{\"answer\": {\"threshold\": 0.55}}", result.program.state.decision_fields);
    try expectJson(arena, "{\"answer\": {\"threshold\": 0.55}}", result.student.state.decision_fields);
    try expectReport(arena,
        \\{
        \\  "train_score_before": 0.8571,
        \\  "train_score": 0.8571,
        \\  "fitted": [
        \\    {
        \\      "predictor": "self",
        \\      "field": "answer",
        \\      "parameter": "threshold",
        \\      "train_score": 0.8571,
        \\      "train_score_at_start": 0.8571,
        \\      "observed": {
        \\        "calls": 7,
        \\        "distinct": 7,
        \\        "min": 0.1,
        \\        "max": 0.8,
        \\        "candidates": 8
        \\      },
        \\      "fold_check": {
        \\        "passed": 0,
        \\        "failed": 1
        \\      },
        \\      "train_score_original": 0.8571,
        \\      "skipped": "fitted behavior did not beat the original"
        \\    }
        \\  ]
        \\}
    , optimizer.report.?);
}

test "Choice weights and validation scores match Source" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const rules = [_]lm_mod.KeywordLM.Rule{
        .{ .needle = "alpha", .response = "<answer>{\"probabilities\": {\"billing\": 0.9, \"technical\": 0.1}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "bravo", .response = "<answer>{\"probabilities\": {\"billing\": 0.7, \"technical\": 0.3}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "charlie", .response = "<answer>{\"probabilities\": {\"billing\": 0.6, \"technical\": 0.4}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "delta", .response = "<answer>{\"probabilities\": {\"billing\": 0.55, \"technical\": 0.45}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "echo", .response = "<answer>{\"probabilities\": {\"billing\": 0.3, \"technical\": 0.7}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "foxtrot", .response = "<answer>{\"probabilities\": {\"billing\": 0.2, \"technical\": 0.8}, \"confidence\": 0.5}</answer>" },
    };
    const trainset = try examplesFrom(arena, &.{
        "{\"question\": \"alpha\", \"label\": \"billing\"}",     "{\"question\": \"bravo\", \"label\": \"billing\"}",
        "{\"question\": \"charlie\", \"label\": \"technical\"}", "{\"question\": \"delta\", \"label\": \"technical\"}",
        "{\"question\": \"echo\", \"label\": \"technical\"}",    "{\"question\": \"foxtrot\", \"label\": \"technical\"}",
    });
    var optimizer: ReAnchor = .{ .metric = choiceMetric, .io = testing.io };
    const result = try compileCase(ChoiceSig, arena, &optimizer, &rules, null, trainset, &.{ trainset[0], trainset[4] });
    try expectJson(arena, "{\"answer\": {\"weights\": {\"billing\": 0.5, \"technical\": 1.0}}}", result.program.state.decision_fields);
    try expectReport(arena,
        \\{
        \\  "train_score_before": 0.6667,
        \\  "train_score": 1.0,
        \\  "fitted": [
        \\    {
        \\      "predictor": "self",
        \\      "field": "answer",
        \\      "parameter": "weights",
        \\      "value": {
        \\        "billing": 0.5,
        \\        "technical": 1.0
        \\      },
        \\      "train_score": 1.0,
        \\      "train_score_at_start": 0.6667,
        \\      "observed": {
        \\        "calls": 6,
        \\        "candidates": 14
        \\      },
        \\      "fold_check": {
        \\        "passed": 1,
        \\        "failed": 0
        \\      },
        \\      "train_score_original": 0.6667
        \\    }
        \\  ],
        \\  "val_score_before": 1.0,
        \\  "val_score": 1.0
        \\}
    , optimizer.report.?);
}

test "Score cuts match Source" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const rules = [_]lm_mod.KeywordLM.Rule{
        .{ .needle = "alpha", .response = "<answer>{\"probabilities\": {\"0\": 0.8, \"1\": 0.2, \"2\": 0.0}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "bravo", .response = "<answer>{\"probabilities\": {\"0\": 0.5, \"1\": 0.4, \"2\": 0.1}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "charlie", .response = "<answer>{\"probabilities\": {\"0\": 0.2, \"1\": 0.6, \"2\": 0.2}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "delta", .response = "<answer>{\"probabilities\": {\"0\": 0.1, \"1\": 0.5, \"2\": 0.4}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "echo", .response = "<answer>{\"probabilities\": {\"0\": 0.0, \"1\": 0.3, \"2\": 0.7}, \"confidence\": 0.5}</answer>" },
        .{ .needle = "foxtrot", .response = "<answer>{\"probabilities\": {\"0\": 0.0, \"1\": 0.1, \"2\": 0.9}, \"confidence\": 0.5}</answer>" },
    };
    const trainset = try examplesFrom(arena, &.{
        "{\"question\": \"alpha\", \"label\": 0}",   "{\"question\": \"bravo\", \"label\": 0}",
        "{\"question\": \"charlie\", \"label\": 0}", "{\"question\": \"delta\", \"label\": 1}",
        "{\"question\": \"echo\", \"label\": 2}",    "{\"question\": \"foxtrot\", \"label\": 2}",
    });
    var optimizer: ReAnchor = .{ .metric = scoreMetric, .io = testing.io };
    const result = try compileCase(ScoreSig, arena, &optimizer, &rules, null, trainset, null);
    try expectJson(arena, "{\"answer\": {\"cuts\": [1.1, 1.5]}}", result.program.state.decision_fields);
    const row = optimizer.report.?.fitted[0];
    try testing.expectEqual(Parameter.cuts, row.parameter);
    try testing.expectEqual(@as(f64, 0.6667), optimizer.report.?.train_score_before);
    try testing.expectEqual(@as(f64, 1.0), optimizer.report.?.train_score);
    try testing.expectEqual(@as(?usize, 6), row.observed.distinct);
    try testing.expectEqual(@as(f64, 0.2), row.observed.min.?.float);
    try testing.expectEqual(@as(f64, 1.9), row.observed.max.?.float);
    try testing.expectEqual(@as(usize, 8), row.observed.candidates);
    try testing.expectEqual(FoldCheck{ .passed = 1, .failed = 0 }, row.fold_check);
}

test "compile refuses empty training sets, decision-free students, and uncached models" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const trainset = try examplesFrom(arena, &.{"{\"question\": \"alpha\", \"label\": true}"});
    var keyword: lm_mod.KeywordLM = .{ .rules = &noul_rules, .default = "<answer>{}</answer>" };
    var optimizer: ReAnchor = .{ .metric = noulMetric, .io = testing.io };
    const student: predict.Predict(NoulSig) = .{};
    const P = predict.Predict(NoulSig);
    try testing.expectError(error.EmptyTrainset, optimizer.compile(P, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = keyword.lm() }, &student, &.{}, null));
    try testing.expectError(error.NoLM, optimizer.compile(P, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena }, &student, trainset, null));
    try testing.expectError(error.UncachedPredictors, optimizer.compile(P, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = keyword.lm() }, &student, trainset, null));
    try testing.expectEqual(@as(usize, 1), optimizer.uncached.len);
    try testing.expectEqualStrings("self", optimizer.uncached[0]);
    var disabled: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = keyword.lm(), .max_entries = 0 };
    defer disabled.deinit();
    try testing.expectError(error.UncachedPredictors, optimizer.compile(P, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = disabled.lm() }, &student, trainset, null));
    // A bound cached model satisfies the check without a settings model.
    var cached: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = keyword.lm() };
    defer cached.deinit();
    var bound = student;
    bound.state.lm = cached.lm();
    bound.state.config = &.{ .cache = false };
    try testing.expectError(error.UncachedPredictors, optimizer.compile(P, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena }, &bound, trainset, null));
    _ = try optimizer.compile(P, arena, testing.allocator, .{
        .workspace = @import("workspace.zig").forTests(),
        .arena = arena,
        .predict_config = &.{ .cache = @as(?bool, null) },
    }, &bound, trainset, null);
    try testing.expect(optimizer.report != null);

    const Plain = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    try testing.expectError(error.NoDecisionPredictors, optimizer.compile(Plain, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = keyword.lm() }, &Plain{}, trainset, null));
    optimizer.require_cache = false;
    _ = try optimizer.compile(P, arena, testing.allocator, .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = keyword.lm() }, &student, trainset, null);
}

test "ReAnchor rejects inconsistent scoring LMs before collecting evidence or scoring" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const P = predict.Predict(NoulSig);
    const parallel = @import("parallel.zig");
    const E = parallel.Evaluator(P, .{ .workers = 1, .tasks = 1, .attempts = 1, .failures = 1 });
    const trainset = try examplesFrom(arena, &.{"{\"question\": \"alpha\", \"label\": false}"});
    const response = "<answer>{\"noul\": 0.1}</answer>";
    var source: lm_mod.ScriptedLM = .{ .responses = &.{response} };
    var other: lm_mod.ScriptedLM = .{ .responses = &(@as([32][]const u8, @splat(response))) };
    var cached: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = source.lm() };
    defer cached.deinit();
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = cached.lm() };
    var scoring: [1]Context = .{.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = other.lm() }};
    var evaluator: E = undefined;
    const scratch = try arena.alloc(u8, 64 * 1024);
    try evaluator.init(testing.io, .{}, &.{scratch});
    defer evaluator.deinit();
    var optimizer: ReAnchor = .{
        .metric = noulMetric,
        .io = testing.io,
        .concurrent_evaluation = evaluator.concurrent(&scoring, .{ .straggler_timeout_ns = 0 }),
    };

    try testing.expectError(error.EvaluationLMMismatch, optimizer.compile(P, arena, testing.allocator, ctx, &P{}, trainset, null));
    try testing.expectEqual(@as(usize, 0), source.index);
    try testing.expectEqual(@as(usize, 0), other.index);

    // An explicit override wins even when the predictor has a different LM.
    var student: P = .{};
    student.state.lm = other.lm();
    scoring[0] = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm_override = cached.lm() };
    const evidence_ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm_override = cached.lm() };
    _ = try optimizer.compile(P, arena, testing.allocator, evidence_ctx, &student, trainset, null);
    try testing.expectEqual(@as(f64, 1), optimizer.report.?.train_score);
    try testing.expectEqual(@as(usize, 1), source.index);
    try testing.expectEqual(@as(usize, 0), other.index);

    optimizer.require_cache = false;
    scoring[0].lm_override = other.lm();
    _ = try optimizer.compile(P, arena, testing.allocator, evidence_ctx, &student, trainset, null);
    try testing.expect(other.index > 0);
}
