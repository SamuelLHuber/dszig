//! `dspy.GEPA` (DSPy 3.4.1 on gepa 0.1.4): reflective prompt evolution.
//! Each iteration picks a parent from the per-example Pareto front, runs it
//! on a minibatch with traces, asks a reflection LM (`ProposeInstruction`,
//! JSONAdapter) for a better instruction from the metric's textual
//! feedback, and keeps the child when its minibatch score sum strictly
//! improves; kept children are scored on the full valset. Merge combines two
//! Pareto programs that improved different predictors of a common ancestor.
//! Python's `random.Random(seed)` is ported (one stream for the engine, one
//! for the DSPy adapter), so the same LM answers give the same program.
//!
//! Not supported: custom proposers, `dspy.Flex` code components, frontier
//! types other than `"instance"`, `"epsilon_greedy"`/`"top_k_pareto"`,
//! `add_format_failure_as_feedback`, `log_dir`. DSPy `num_threads` maps to
//! `rollouts`: each minibatch/valset evaluation then runs on a native pool,
//! results in example order. CPython iterates small-int sets in ascending
//! order; DSZig assumes it always does.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const optimize = @import("optimize.zig");
const parallel = @import("parallel.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const Record = value.Record;
const Trace = predict.Trace;
const Value = @import("runtime_value.zig").Value;

/// What a GEPA metric returns: DSPy's float or `dspy.Prediction(score, feedback)`.
pub const Feedback = struct { score: f64, feedback: ?[]const u8 = null };

/// `metric(gold, pred, trace, pred_name, pred_trace)`. `pred_name` and
/// `pred_step` are set only when GEPA asks for feedback on one predictor.
/// `prediction` is the program's retained Prediction, with the usage of its
/// invocation; a failed evaluation row passes an empty one, as Source does.
pub const FeedbackMetric = *const fn (
    arena: Allocator,
    example: *const Example,
    prediction: *const optimize.RecordPrediction,
    trace: ?*const Trace,
    pred_name: ?[]const u8,
    pred_step: ?*const Trace.Step,
) value.ParseError!Feedback;

pub const ProposeInstruction = signature.Signature(struct {
    pub const instructions = PROPOSE_INSTRUCTIONS;
    pub const Input = struct { current_instruction: []const u8, examples_with_feedback: []const @import("runtime_value.zig").ObjectMap };
    pub const Output = struct { new_instruction: []const u8 };
    pub const descriptions = .{
        .current_instruction = "The instructions I provided to the assistant.",
        .examples_with_feedback = "Task inputs provided to the assistant, the assistant's response for each, and feedback on how the response could be better.",
        .new_instruction = "The new instruction for the assistant.",
    };
});

pub const GEPA = struct {
    metric: FeedbackMetric,
    auto: ?Auto = null,
    max_full_evals: ?usize = null,
    max_metric_calls: ?usize = null,
    reflection_minibatch_size: usize = 3,
    candidate_selection_strategy: Selection = .pareto,
    skip_perfect_score: bool = true,
    component_selector: ComponentSelector = .round_robin,
    use_merge: bool = true,
    max_merge_invocations: usize = 5,
    failure_score: f64 = 0.0,
    perfect_score: f64 = 1.0,
    seed: u32 = 0,
    /// Retain the predictions attaining the best score for each validation row.
    track_best_outputs: bool = false,
    /// DSPy `num_threads`: a `parallel.Parallel(GEPA.Run, limits).runner()`
    /// plus per-attempt contexts. Program and metric must be thread-safe, and
    /// `out`, which retains predictions and traces, must then be thread-safe
    /// (an ArenaAllocator over a thread-safe child is). null runs serially.
    rollouts: ?Rollouts = null,

    pub const Rollouts = parallel.Rollouts(Run);

    pub const Auto = enum { light, medium, heavy };
    pub const Selection = enum { pareto, current_best };
    pub const ComponentSelector = enum { round_robin, all };

    pub const Options = struct {
        /// Defaults to the trainset, as in DSPy.
        valset: ?[]const Example = null,
        /// DSPy's `reflection_lm` (defaults to `ctx`).
        reflection: ?Context = null,
    };

    pub const Error = error{ ZeroWeights, NoBudget };

    pub const BestOutput = struct {
        candidate_idx: usize,
        prediction: optimize.RecordPrediction,
    };

    /// `DspyGEPAResult` (`track_stats=True`) plus the optimized program.
    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            /// Instructions per predictor (DSPy `named_predictors()` order).
            candidates: []const []const []const u8,
            parents: []const []const ?usize,
            val_aggregate_scores: []const f64,
            val_subscores: []const []const f64,
            /// Indexed by validation row; tied candidates are in discovery order.
            per_val_instance_best_candidates: []const []const usize,
            /// Indexed by validation row, in candidate discovery order.
            best_outputs_valset: ?[]const []const BestOutput,
            discovery_eval_counts: []const usize,
            best_idx: usize,
            total_metric_calls: usize,
            num_full_val_evals: usize,
            max_metric_calls: usize,
        };
    }

    const merge_overlap_floor = 5;
    const merge_subsample = 5;
    const max_attempts = 10;

    /// The reflective dataset serializes trace-step records and feedback
    /// only, so a prediction's usage reaches the metric and nothing else.
    pub const Run = struct { prediction: ?optimize.RecordPrediction, trace: *const Trace, score: f64 };

    const Desc = struct { id1: usize, id2: usize, picks: []const usize };

    fn Engine(comptime P: type) type {
        return struct {
            const Self = @This();
            gepa: GEPA,
            out: Allocator,
            ctx: Context,
            reflection: Context,
            student: *const P,
            names: []const []const u8,
            trainset: []const Example,
            valset: []const Example,
            rng: pyrandom.Random,
            adapter_rng: pyrandom.Random,
            candidates: std.ArrayList([]const []const u8) = .empty,
            subscores: std.ArrayList([]const f64) = .empty,
            parents: std.ArrayList([]const ?usize) = .empty,
            discovery: std.ArrayList(usize) = .empty,
            next_pred: std.ArrayList(usize) = .empty,
            front_score: []f64,
            fronts: []std.ArrayList(usize),
            best_outputs: ?[]std.ArrayList(BestOutput) = null,
            total_evals: usize = 0,
            full_evals: usize = 0,
            i: usize = 0,
            // EpochShuffledBatchSampler
            shuffled: std.ArrayList(usize) = .empty,
            epoch: ?usize = null,
            sampler_iteration: ?usize = null,
            calls_in_iteration: usize = 0,
            // MergeProposer
            merges_due: usize = 0,
            merges_tested: usize = 0,
            last_iter_found_new_program: bool = false,
            triplets: std.ArrayList([3]usize) = .empty,
            descs: std.ArrayList(Desc) = .empty,

            fn average(e: *const Self, idx: usize) f64 {
                const list = e.subscores.items[idx];
                if (list.len == 0) return -std.math.inf(f64);
                var sum: f64 = 0;
                for (list) |x| sum += x;
                return sum / @as(f64, @floatFromInt(list.len));
            }

            fn averages(e: *const Self) Allocator.Error![]f64 {
                const list = try e.out.alloc(f64, e.candidates.items.len);
                for (list, 0..) |*a, idx| a.* = e.average(idx);
                return list;
            }

            fn build(e: *const Self, candidate: []const []const u8) Allocator.Error!*P {
                const copy = try e.out.create(P);
                copy.* = e.student.*;
                const preds = try predict.namedPredictors(e.out, copy);
                assert(preds.len == candidate.len);
                for (preds, candidate) |p, text| {
                    try p.param.predict.forkSignatureClass(e.out);
                    p.param.predict.instructions = text;
                }
                return copy;
            }

            fn failedRun(e: *const Self, trace: *Trace, capture: bool) Allocator.Error!Run {
                // Evaluate substitutes an empty Prediction, while bootstrap
                // trace capture drops failed rows and their trajectories.
                trace.steps.clearRetainingCapacity();
                return .{
                    .prediction = if (capture) null else try optimize.RecordPrediction.init(e.out, .empty),
                    .trace = trace,
                    .score = e.gepa.failure_score,
                };
            }

            fn runOne(e: *const Self, ctx: Context, program: *const P, example: *const Example, capture: bool) Allocator.Error!Run {
                const trace = try e.out.create(Trace);
                trace.* = .{ .arena = e.out };
                var c = ctx;
                c.arena = e.out;
                c.trace = if (capture) trace else null;
                const prediction = optimize.runProgram(P, c, program, example) catch |err| {
                    if (err == error.OutOfMemory) return error.OutOfMemory;
                    return e.failedRun(trace, capture);
                };
                const feedback = e.gepa.metric(e.out, example, &prediction, if (capture) trace else null, null, null) catch |err| {
                    if (err == error.OutOfMemory) return error.OutOfMemory;
                    return e.failedRun(trace, capture);
                };
                return .{ .prediction = prediction, .trace = trace, .score = feedback.score };
            }

            fn evalIds(e: *Self, candidate: []const []const u8, set: []const Example, ids: []const usize, capture: bool) ![]Run {
                const program = try e.build(candidate);
                const runs = try e.out.alloc(Run, ids.len);
                if (e.gepa.rollouts) |rollouts| {
                    const Call = struct {
                        engine: *const Self,
                        program: *const P,
                        example: *const Example,
                        capture: bool,

                        fn invoke(ptr: *anyopaque, attempt: parallel.Attempt, run: *Run) anyerror!void {
                            const call: *const @This() = @ptrCast(@alignCast(ptr));
                            try attempt.cancellation.check();
                            const en = call.engine;
                            const c = try en.gepa.rollouts.?.context(en.ctx, attempt, false);
                            run.* = try en.runOne(c, call.program, call.example, call.capture);
                        }
                    };
                    const calls = try e.out.alloc(Call, ids.len);
                    const tasks = try e.out.alloc(parallel.TaskFor(Run), ids.len);
                    for (calls, tasks, ids) |*call, *task, id| {
                        call.* = .{ .engine = e, .program = program, .example = &set[id], .capture = capture };
                        task.* = .{ .ptr = call, .invoke = Call.invoke };
                    }
                    try rollouts.run(tasks, runs);
                } else {
                    for (runs, ids) |*r, id| r.* = try e.runOne(e.ctx, program, &set[id], capture);
                }
                e.total_evals += ids.len;
                return runs;
            }

            fn addProgram(e: *Self, candidate: []const []const u8, parent_ids: []const ?usize) !void {
                const all = try e.out.alloc(usize, e.valset.len);
                for (all, 0..) |*a, k| a.* = k;
                const discovered = e.total_evals;
                const runs = try e.evalIds(candidate, e.valset, all, false);
                e.full_evals += 1;
                const idx = e.candidates.items.len;
                const scores = try e.out.alloc(f64, runs.len);
                for (scores, runs) |*s, r| s.* = r.score;
                var next: usize = 0;
                for (parent_ids) |p| if (p) |pi| {
                    next = @max(next, e.next_pred.items[pi]);
                };
                try e.candidates.append(e.out, candidate);
                try e.subscores.append(e.out, scores);
                try e.parents.append(e.out, parent_ids);
                try e.discovery.append(e.out, discovered);
                try e.next_pred.append(e.out, next);
                for (runs, 0..) |run, k| {
                    const improves = idx == 0 or run.score > e.front_score[k];
                    if (!improves and run.score != e.front_score[k]) continue;
                    if (improves) {
                        e.front_score[k] = run.score;
                        e.fronts[k].clearRetainingCapacity();
                    }
                    try e.fronts[k].append(e.out, idx);
                    if (e.best_outputs) |outputs| {
                        // Full evaluation substitutes an empty prediction on failure.
                        const prediction = run.prediction.?;
                        if (improves) outputs[k].clearRetainingCapacity();
                        try outputs[k].append(e.out, .{ .candidate_idx = idx, .prediction = prediction });
                    }
                }
            }

            /// `remove_dominated_programs`: returns the dominated flags.
            fn dominated(e: *const Self) Allocator.Error![]bool {
                const n = e.candidates.items.len;
                const flags = try e.out.alloc(bool, n);
                @memset(flags, false);
                var programs: std.ArrayList(usize) = .empty;
                for (e.fronts) |front| for (front.items) |p| {
                    if (std.mem.indexOfScalar(usize, programs.items, p) == null) try programs.append(e.out, p);
                };
                const scores = try e.averages();
                std.sort.insertion(usize, programs.items, scores, struct {
                    fn less(s: []const f64, a: usize, b: usize) bool {
                        return s[a] < s[b];
                    }
                }.less);
                var found = true;
                while (found) {
                    found = false;
                    for (programs.items) |y| {
                        if (flags[y]) continue;
                        if (e.isDominated(y, flags)) {
                            flags[y] = true;
                            found = true;
                            break;
                        }
                    }
                }
                return flags;
            }

            fn isDominated(e: *const Self, y: usize, flags: []const bool) bool {
                for (e.fronts) |front| {
                    if (std.mem.indexOfScalar(usize, front.items, y) == null) continue;
                    const covered = for (front.items) |p| {
                        if (p != y and !flags[p]) break true;
                    } else false;
                    if (!covered) return false;
                }
                return true;
            }

            /// `select_program_candidate_from_pareto_front`.
            fn selectPareto(e: *Self) Allocator.Error!usize {
                const flags = try e.dominated();
                var order: std.ArrayList(usize) = .empty;
                var freq: std.ArrayList(usize) = .empty;
                for (e.fronts) |front| for (front.items) |p| {
                    if (flags[p]) continue;
                    if (std.mem.indexOfScalar(usize, order.items, p)) |k| {
                        freq.items[k] += 1;
                    } else {
                        try order.append(e.out, p);
                        try freq.append(e.out, 1);
                    }
                };
                var sampling: std.ArrayList(usize) = .empty;
                for (order.items, freq.items) |p, f| for (0..f) |_| try sampling.append(e.out, p);
                assert(sampling.items.len > 0);
                return sampling.items[e.rng.randbelow(sampling.items.len)];
            }

            fn selectBest(e: *const Self) usize {
                var best: usize = 0;
                for (0..e.candidates.items.len) |idx| {
                    if (e.average(idx) > e.average(best)) best = idx;
                }
                return best;
            }

            /// `EpochShuffledBatchSampler.next_minibatch_ids`.
            fn nextMinibatch(e: *Self) Allocator.Error![]const usize {
                const mb = e.gepa.reflection_minibatch_size;
                if (e.sampler_iteration == e.i) {
                    e.calls_in_iteration += 1;
                } else {
                    e.sampler_iteration = e.i;
                    e.calls_in_iteration = 0;
                }
                var base = e.i * mb;
                const curr_epoch: usize = if (e.epoch == null) 0 else base / @max(e.shuffled.items.len, 1);
                if (e.shuffled.items.len == 0 or e.epoch == null or curr_epoch > e.epoch.?) {
                    e.epoch = curr_epoch;
                    try e.reshuffle();
                }
                base = (base + e.calls_in_iteration * mb) % e.shuffled.items.len;
                const end = @min(base + mb, e.shuffled.items.len);
                return e.shuffled.items[base..end];
            }

            fn reshuffle(e: *Self) Allocator.Error!void {
                const n = e.trainset.len;
                const mb = e.gepa.reflection_minibatch_size;
                e.shuffled = .empty;
                for (0..n) |k| try e.shuffled.append(e.out, k);
                e.rng.shuffle(usize, e.shuffled.items);
                const first = try e.out.dupe(usize, e.shuffled.items);
                const counts = try e.out.alloc(usize, n);
                @memset(counts, 1);
                const rem = n % mb;
                if (rem == 0) return;
                for (0..mb - rem) |_| {
                    // `Counter.most_common()[::-1][0]`: the least frequent id,
                    // latest in first-appearance order.
                    var least: usize = std.math.maxInt(usize);
                    for (counts) |c| least = @min(least, c);
                    var k = n;
                    const pick = while (k > 0) {
                        k -= 1;
                        if (counts[first[k]] == least) break first[k];
                    } else unreachable;
                    try e.shuffled.append(e.out, pick);
                    counts[pick] += 1;
                }
            }

            /// One `ReflectiveMutationProposer.propose` + acceptance.
            fn reflect(e: *Self) !void {
                const parent = switch (e.gepa.candidate_selection_strategy) {
                    .pareto => try e.selectPareto(),
                    .current_best => e.selectBest(),
                };
                const ids = try e.nextMinibatch();
                const parent_candidate = e.candidates.items[parent];
                const before = try e.evalIds(parent_candidate, e.trainset, ids, true);
                if (e.gepa.skip_perfect_score) {
                    const perfect = for (before) |r| {
                        if (r.score < e.gepa.perfect_score) break false;
                    } else true;
                    if (perfect) return;
                }
                var components: std.ArrayList(usize) = .empty;
                switch (e.gepa.component_selector) {
                    .round_robin => {
                        const pid = e.next_pred.items[parent];
                        e.next_pred.items[parent] = (pid + 1) % e.names.len;
                        try components.append(e.out, pid);
                    },
                    .all => for (0..e.names.len) |k| try components.append(e.out, k),
                }
                const datasets = try e.reflectiveDataset(parent_candidate, before, ids, components.items);
                var any = false;
                for (datasets) |d| any = any or d != null;
                if (!any) return;
                const texts = (try e.proposeTexts(parent_candidate, components.items, datasets)) orelse
                    (try e.proposeTexts(parent_candidate, components.items, datasets)) orelse return;
                const child = try e.out.dupe([]const u8, parent_candidate);
                var changed = false;
                for (components.items, texts) |k, t| if (t) |text| {
                    child[k] = text;
                    changed = true;
                };
                if (!changed) return;
                const after = try e.evalIds(child, e.trainset, ids, true);
                if (sumScores(after) > sumScores(before)) {
                    const parent_ids = try e.out.alloc(?usize, 1);
                    parent_ids[0] = parent;
                    try e.addProgram(child, parent_ids);
                    if (e.gepa.use_merge) {
                        e.last_iter_found_new_program = true;
                        if (e.merges_tested < e.gepa.max_merge_invocations) e.merges_due += 1;
                    }
                }
            }

            /// `DspyAdapter.make_reflective_dataset`; `null` where no example
            /// produced a usable trace for that predictor.
            fn reflectiveDataset(e: *Self, candidate: []const []const u8, runs: []const Run, ids: []const usize, components: []const usize) value.ParseError![]?[]const @import("runtime_value.zig").ObjectMap {
                const program = try e.build(candidate);
                const preds = try predict.namedPredictors(e.out, program);
                const result = try e.out.alloc(?[]const @import("runtime_value.zig").ObjectMap, components.len);
                for (components, result) |k, *slot| {
                    const module = preds[k].param.predict;
                    var items: std.ArrayList(@import("runtime_value.zig").ObjectMap) = .empty;
                    for (runs, ids) |run, id| {
                        const prediction = run.prediction orelse continue;
                        var matching: std.ArrayList(*const Trace.Step) = .empty;
                        for (run.trace.steps.items) |*step| if (predict.sameSignature(step.predictor, module)) try matching.append(e.out, step);
                        if (matching.items.len == 0) continue;
                        const step = matching.items[e.adapter_rng.randbelow(matching.items.len)];
                        const fb = try e.gepa.metric(e.out, &e.trainset[id], &prediction, run.trace, e.names[k], step);
                        var item: @import("runtime_value.zig").ObjectMap = .empty;
                        try item.put(e.out, "Inputs", .{ .object = step.inputs });
                        try item.put(e.out, "Generated Outputs", .{ .object = step.outputs });
                        const text = fb.feedback orelse try std.fmt.allocPrint(e.out, "This trajectory got a score of {s}.", .{try pyFloat(e.out, fb.score)});
                        try item.put(e.out, "Feedback", .{ .string = text });
                        try items.append(e.out, item);
                    }
                    slot.* = if (items.items.len == 0) null else items.items;
                }
                return result;
            }

            /// `InstructionProposer.__call__`; `null` when a call fails.
            fn proposeTexts(e: *Self, candidate: []const []const u8, components: []const usize, datasets: []const ?[]const @import("runtime_value.zig").ObjectMap) Allocator.Error!?[]?[]const u8 {
                const texts = try e.out.alloc(?[]const u8, components.len);
                const proposer: predict.Predict(ProposeInstruction) = .{};
                for (components, datasets, texts) |k, dataset, *t| {
                    t.* = null;
                    const examples = dataset orelse continue;
                    var c = e.reflection;
                    c.arena = e.out;
                    c.trace = null;
                    c.adapter = .json;
                    const out = proposer.forward(c, .{ .current_instruction = candidate[k], .examples_with_feedback = examples }) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return null,
                    };
                    const draft = value.pyStrip(out.new_instruction);
                    if (draft.len == 0) return null;
                    t.* = draft;
                }
                return texts;
            }

            /// `MergeProposer.propose` + the engine's merge acceptance.
            /// Returns whether a merge was evaluated (the iteration ends).
            fn merge(e: *Self) !bool {
                const flags = try e.dominated();
                var dominators: std.ArrayList(usize) = .empty;
                for (0..e.candidates.items.len) |p| {
                    if (flags[p]) continue;
                    const on_front = for (e.fronts) |front| {
                        if (std.mem.indexOfScalar(usize, front.items, p) != null) break true;
                    } else false;
                    if (on_front) try dominators.append(e.out, p);
                }
                const agg = try e.averages();
                const attempt = (try e.attemptMerge(dominators.items, agg)) orelse return false;
                try e.triplets.append(e.out, .{ attempt.id1, attempt.id2, attempt.ancestor });
                const ids = try e.mergeSubsample(e.subscores.items[attempt.id1], e.subscores.items[attempt.id2]);
                if (ids.len == 0) return false;
                var sum1: f64 = 0;
                var sum2: f64 = 0;
                for (ids) |id| {
                    sum1 += e.subscores.items[attempt.id1][id];
                    sum2 += e.subscores.items[attempt.id2][id];
                }
                const runs = try e.evalIds(attempt.candidate, e.valset, ids, false);
                if (sumScores(runs) >= @max(sum1, sum2)) {
                    const parent_ids = try e.out.alloc(?usize, 2);
                    parent_ids[0] = attempt.id1;
                    parent_ids[1] = attempt.id2;
                    try e.addProgram(attempt.candidate, parent_ids);
                    e.merges_due -= 1;
                    e.merges_tested += 1;
                }
                return true;
            }

            const Attempt = struct { candidate: []const []const u8, id1: usize, id2: usize, ancestor: usize };

            /// `sample_and_attempt_merge_programs_by_common_predictors`.
            fn attemptMerge(e: *Self, merge_candidates: []const usize, agg: []const f64) !?Attempt {
                if (merge_candidates.len < 2) return null;
                if (e.parents.items.len < 3) return null;
                attempts: for (0..max_attempts) |_| {
                    const pair = (try e.commonAncestorPair(merge_candidates, agg)) orelse continue;
                    const id1, const id2, const ancestor = pair;
                    for (e.triplets.items) |t| if (t[0] == id1 and t[1] == id2 and t[2] == ancestor) continue :attempts;
                    const cands = e.candidates.items;
                    const merged = try e.out.dupe([]const u8, cands[ancestor]);
                    const picks = try e.out.alloc(usize, merged.len);
                    for (merged, picks, 0..) |*text, *pick, k| {
                        const anc = cands[ancestor][k];
                        const p1 = cands[id1][k];
                        const p2 = cands[id2][k];
                        const eq1 = std.mem.eql(u8, anc, p1);
                        const eq2 = std.mem.eql(u8, anc, p2);
                        const same = std.mem.eql(u8, p1, p2);
                        if ((eq1 or eq2) and !same) {
                            pick.* = if (eq1) id2 else id1;
                        } else if (!eq1 and !eq2) {
                            pick.* = if (agg[id1] > agg[id2]) id1 else if (agg[id2] > agg[id1]) id2 else if (e.rng.randbelow(2) == 0) id1 else id2;
                        } else {
                            pick.* = id1;
                        }
                        text.* = cands[pick.*][k];
                    }
                    for (e.descs.items) |d| {
                        if (d.id1 == id1 and d.id2 == id2 and std.mem.eql(usize, d.picks, picks)) continue :attempts;
                    }
                    if (e.valset.len < merge_overlap_floor) continue;
                    try e.descs.append(e.out, .{ .id1 = id1, .id2 = id2, .picks = picks });
                    return .{ .candidate = merged, .id1 = id1, .id2 = id2, .ancestor = ancestor };
                }
                return null;
            }

            /// `find_common_ancestor_pair`.
            fn commonAncestorPair(e: *Self, merge_candidates: []const usize, agg: []const f64) !?[3]usize {
                const n = e.candidates.items.len;
                for (0..max_attempts) |_| {
                    var chosen: [2]usize = undefined;
                    const scratch = try e.out.alloc(usize, merge_candidates.len);
                    e.rng.sampleIndices(merge_candidates.len, &chosen, scratch);
                    var i = merge_candidates[chosen[0]];
                    var j = merge_candidates[chosen[1]];
                    if (i == j) continue;
                    if (j < i) std.mem.swap(usize, &i, &j);
                    const anc_i = try e.ancestors(i);
                    const anc_j = try e.ancestors(j);
                    if (anc_i[j] or anc_j[i]) continue;
                    var common: std.ArrayList(usize) = .empty;
                    for (0..n) |a| {
                        if (!anc_i[a] or !anc_j[a]) continue;
                        const done = for (e.triplets.items) |t| {
                            if (t[0] == i and t[1] == j and t[2] == a) break true;
                        } else false;
                        if (done) continue;
                        if (agg[a] > agg[i] or agg[a] > agg[j]) continue;
                        if (!e.desirable(a, i, j)) continue;
                        try common.append(e.out, a);
                    }
                    if (common.items.len == 0) continue;
                    // `rng.choices(common, k=1, weights=agg)`.
                    const cum = try e.out.alloc(f64, common.items.len);
                    var acc: f64 = 0;
                    for (cum, common.items) |*c, a| {
                        acc += agg[a];
                        c.* = acc;
                    }
                    const total = cum[cum.len - 1] + 0.0;
                    if (!(total > 0.0)) return error.ZeroWeights;
                    const x = e.rng.random() * total;
                    var lo: usize = 0;
                    var hi: usize = cum.len - 1;
                    while (lo < hi) {
                        const mid = (lo + hi) / 2;
                        if (x < cum[mid]) hi = mid else lo = mid + 1;
                    }
                    return .{ i, j, common.items[lo] };
                }
                return null;
            }

            fn ancestors(e: *const Self, node: usize) Allocator.Error![]bool {
                const found = try e.out.alloc(bool, e.candidates.items.len);
                @memset(found, false);
                var stack: std.ArrayList(usize) = .empty;
                try stack.append(e.out, node);
                while (stack.pop()) |cur| {
                    for (e.parents.items[cur]) |p| if (p) |pi| {
                        if (found[pi]) continue;
                        found[pi] = true;
                        try stack.append(e.out, pi);
                    };
                }
                return found;
            }

            fn desirable(e: *const Self, ancestor: usize, id1: usize, id2: usize) bool {
                const cands = e.candidates.items;
                for (cands[ancestor], cands[id1], cands[id2]) |anc, p1, p2| {
                    if ((std.mem.eql(u8, anc, p1) or std.mem.eql(u8, anc, p2)) and !std.mem.eql(u8, p1, p2)) return true;
                }
                return false;
            }

            /// `select_eval_subsample_for_merged_program`.
            fn mergeSubsample(e: *Self, scores1: []const f64, scores2: []const f64) Allocator.Error![]const usize {
                var buckets: [3]std.ArrayList(usize) = .{ .empty, .empty, .empty };
                for (scores1, scores2, 0..) |a, b, id| {
                    const k: usize = if (a > b) 0 else if (b > a) 1 else 2;
                    try buckets[k].append(e.out, id);
                }
                const n_each = @max(1, std.math.divCeil(usize, merge_subsample, 3) catch unreachable);
                var selected: std.ArrayList(usize) = .empty;
                for (&buckets) |bucket| {
                    if (selected.items.len >= merge_subsample) break;
                    const take = @min(bucket.items.len, n_each, merge_subsample - selected.items.len);
                    if (take > 0) try e.sampleInto(&selected, bucket.items, take);
                }
                const remaining = merge_subsample - selected.items.len;
                if (remaining > 0) {
                    var unused: std.ArrayList(usize) = .empty;
                    for (0..scores1.len) |id| {
                        if (std.mem.indexOfScalar(usize, selected.items, id) == null) try unused.append(e.out, id);
                    }
                    if (unused.items.len >= remaining) {
                        try e.sampleInto(&selected, unused.items, remaining);
                    } else if (scores1.len > 0) {
                        for (0..remaining) |_| {
                            const r = e.rng.random() * @as(f64, @floatFromInt(scores1.len));
                            try selected.append(e.out, @intFromFloat(@floor(r)));
                        }
                    }
                }
                return selected.items[0..@min(selected.items.len, merge_subsample)];
            }

            fn sampleInto(e: *Self, selected: *std.ArrayList(usize), population: []const usize, k: usize) Allocator.Error!void {
                const chosen = try e.out.alloc(usize, k);
                const scratch = try e.out.alloc(usize, population.len);
                e.rng.sampleIndices(population.len, chosen, scratch);
                for (chosen) |c| try selected.append(e.out, population[c]);
            }
        };
    }

    /// `GEPA.auto_budget`.
    pub fn autoBudget(num_preds: usize, num_candidates: usize, valset_size: usize) usize {
        const minibatch_size = 35;
        const full_eval_steps = 5;
        const n_f: f64 = @floatFromInt(num_candidates);
        const trials_f = @max(2.0 * @as(f64, @floatFromInt(num_preds * 2)) * std.math.log2(n_f), 1.5 * n_f);
        const trials: usize = @intFromFloat(@trunc(trials_f));
        var total = valset_size + num_candidates * 5 + trials * minibatch_size;
        if (trials == 0) return total;
        const periodic_fulls = (trials + 1) / full_eval_steps + 1;
        const extra_final: usize = @intFromBool(trials < full_eval_steps);
        total += (periodic_fulls + extra_final) * valset_size;
        return total;
    }

    pub fn compile(
        self: GEPA,
        comptime P: type,
        out: Allocator,
        ctx: Context,
        student: P,
        trainset: []const Example,
        options: Options,
    ) !Result(P) {
        assert(trainset.len > 0);
        assert(self.reflection_minibatch_size > 0);
        // Source computes its budget from the supplied set before applying
        // `valset or trainset`; an empty sampled holdout still evaluates trainset.
        const valset = if (options.valset) |set| if (set.len > 0) set else trainset else trainset;
        assert(valset.len > 0);
        const owned = try out.create(P);
        owned.* = student;
        const preds = try predict.namedPredictors(out, owned);
        assert(preds.len > 0);
        const max_calls: usize = if (self.auto) |a|
            autoBudget(preds.len, switch (a) {
                .light => 6,
                .medium => 12,
                .heavy => 18,
            }, if (options.valset) |set| set.len else trainset.len)
        else if (self.max_full_evals) |k|
            k * (trainset.len + if (options.valset) |v| v.len else 0)
        else
            self.max_metric_calls orelse return error.NoBudget;

        const names = try out.alloc([]const u8, preds.len);
        const seed_candidate = try out.alloc([]const u8, preds.len);
        for (preds, names, seed_candidate) |p, *n, *s| {
            n.* = p.name;
            s.* = p.param.predict.instructions;
        }
        const fronts = try out.alloc(std.ArrayList(usize), valset.len);
        for (fronts) |*f| f.* = .empty;
        var e: Engine(P) = .{
            .gepa = self,
            .out = out,
            .ctx = ctx,
            .reflection = options.reflection orelse ctx,
            .student = owned,
            .names = names,
            .trainset = trainset,
            .valset = valset,
            .rng = .init(self.seed),
            .adapter_rng = .init(self.seed),
            .front_score = try out.alloc(f64, valset.len),
            .fronts = fronts,
        };
        if (self.track_best_outputs) {
            const outputs = try out.alloc(std.ArrayList(BestOutput), valset.len);
            @memset(outputs, .empty);
            e.best_outputs = outputs;
        }
        try e.addProgram(seed_candidate, &.{null});
        var started = false;
        while (e.total_evals < max_calls) {
            if (started) e.i += 1;
            started = true;
            if (self.use_merge) {
                if (e.merges_due > 0 and e.last_iter_found_new_program) {
                    const merged = try e.merge();
                    e.last_iter_found_new_program = false;
                    if (merged) continue;
                }
                e.last_iter_found_new_program = false;
            }
            try e.reflect();
        }

        var best: usize = 0;
        for (0..e.candidates.items.len) |idx| {
            if (e.average(idx) > e.average(best)) best = idx;
        }
        const program = try e.build(e.candidates.items[best]);
        const per_instance = try out.alloc([]const usize, e.fronts.len);
        for (per_instance, e.fronts) |*row, front| row.* = front.items;
        const best_outputs: ?[]const []const BestOutput = if (e.best_outputs) |outputs| rows: {
            const rows = try out.alloc([]const BestOutput, outputs.len);
            for (rows, outputs) |*row, outputs_for_row| row.* = outputs_for_row.items;
            break :rows rows;
        } else null;
        return .{
            .program = program.*,
            .candidates = e.candidates.items,
            .parents = e.parents.items,
            .val_aggregate_scores = try e.averages(),
            .val_subscores = e.subscores.items,
            .per_val_instance_best_candidates = per_instance,
            .best_outputs_valset = best_outputs,
            .discovery_eval_counts = e.discovery.items,
            .best_idx = best,
            .total_metric_calls = e.total_evals,
            .num_full_val_evals = e.full_evals,
            .max_metric_calls = max_calls,
        };
    }
};

test "GEPA scoring failures preserve capture-specific outputs while feedback errors propagate" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = predict.Predict(Sig);
    const FailingMetric = struct {
        fn failure(_: Allocator, _: *const Example, _: *const optimize.RecordPrediction, _: ?*const Trace, _: ?[]const u8, _: ?*const Trace.Step) value.ParseError!Feedback {
            return error.ParseValue;
        }

        fn feedbackFailure(_: Allocator, _: *const Example, _: *const optimize.RecordPrediction, _: ?*const Trace, pred_name: ?[]const u8, _: ?*const Trace.Step) value.ParseError!Feedback {
            if (pred_name != null) return error.ParseValue;
            return .{ .score = 1 };
        }

        fn allocationFailure(_: Allocator, _: *const Example, _: *const optimize.RecordPrediction, _: ?*const Trace, _: ?[]const u8, _: ?*const Trace.Step) value.ParseError!Feedback {
            return error.OutOfMemory;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{
        .responses = &.{
            "<answer>ok</answer>", "<answer>ok</answer>", "<answer>ok</answer>",
            "<answer>ok</answer>", "<answer>ok</answer>",
        },
    };
    const ctx: Context = .{ .arena = arena, .lm = script.lm(), .adapter = .xml };
    const examples = [_]Example{try Example.fromJson(arena, "{\"question\":\"q\",\"answer\":\"ok\"}", &.{"question"})};
    var program: P = .{};
    var front_scores = [_]f64{0};
    var fronts = [_]std.ArrayList(usize){.empty};
    var engine: GEPA.Engine(P) = .{
        .gepa = .{ .metric = FailingMetric.failure, .failure_score = 0.125 },
        .out = arena,
        .ctx = ctx,
        .reflection = ctx,
        .student = &program,
        .names = &.{"self"},
        .trainset = &examples,
        .valset = &examples,
        .rng = .init(0),
        .adapter_rng = .init(0),
        .front_score = &front_scores,
        .fronts = &fronts,
    };

    for ([_]bool{ false, true }) |capture| {
        const run = try engine.runOne(engine.ctx, &program, &examples[0], capture);
        try std.testing.expectEqual(@as(f64, 0.125), run.score);
        if (capture) {
            try std.testing.expect(run.prediction == null);
        } else {
            try std.testing.expectEqual(@as(usize, 0), run.prediction.?.fields().count());
        }
        try std.testing.expectEqual(@as(usize, 0), run.trace.steps.items.len);
    }
    try std.testing.expectEqual(@as(usize, 2), script.index);

    var program_failure: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    engine.ctx.lm = program_failure.lm();
    for ([_]bool{ false, true }) |capture| {
        const run = try engine.runOne(engine.ctx, &program, &examples[0], capture);
        try std.testing.expectEqual(@as(f64, 0.125), run.score);
        if (capture) {
            try std.testing.expect(run.prediction == null);
        } else {
            try std.testing.expectEqual(@as(usize, 0), run.prediction.?.fields().count());
        }
        try std.testing.expectEqual(@as(usize, 0), run.trace.steps.items.len);
    }

    engine.ctx = ctx;
    engine.gepa.metric = FailingMetric.feedbackFailure;
    const successful = try engine.runOne(engine.ctx, &program, &examples[0], true);
    try std.testing.expectEqualStrings("ok", successful.prediction.?.get("answer").?.string);
    try std.testing.expectEqual(@as(usize, 1), successful.trace.steps.items.len);
    try std.testing.expectError(error.ParseValue, engine.reflectiveDataset(&.{program.state.instructions}, &.{successful}, &.{0}, &.{0}));

    engine.gepa.metric = FailingMetric.allocationFailure;
    for ([_]bool{ false, true }) |capture| {
        try std.testing.expectError(error.OutOfMemory, engine.runOne(engine.ctx, &program, &examples[0], capture));
    }
    try std.testing.expectEqual(@as(usize, 5), script.index);
}

test "GEPA empty validation evaluates the entire training set" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Metric = struct {
        fn score(_: Allocator, example: *const Example, prediction: *const optimize.RecordPrediction, _: ?*const Trace, _: ?[]const u8, _: ?*const Trace.Step) value.ParseError!Feedback {
            return .{ .score = @floatFromInt(@intFromBool(std.mem.eql(
                u8,
                example.record.get("answer").?.string,
                prediction.get("answer").?.string,
            ))) };
        }
    };
    var script: @import("lm.zig").ScriptedLM = .{
        .responses = &.{ "<answer>ok</answer>", "<answer>wrong</answer>" },
    };
    const examples = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"q\",\"answer\":\"ok\"}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\":\"r\",\"answer\":\"ok\"}", &.{"question"}),
    };
    const optimizer: GEPA = .{ .metric = Metric.score, .max_metric_calls = 1 };
    const result = try optimizer.compile(predict.Predict(Sig), arena, .{ .arena = arena, .lm = script.lm() }, .{}, &examples, .{ .valset = &.{} });
    try std.testing.expect(result.best_outputs_valset == null);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0 }, result.val_subscores[0]);
    try std.testing.expectEqual(@as(f64, 0.5), result.val_aggregate_scores[0]);
    try std.testing.expectEqual(@as(usize, 2), result.total_metric_calls);
    try std.testing.expectEqual(@as(usize, 1), result.max_metric_calls);
}

test "GEPA best outputs keep ties and replace an older frontier without extra evaluations" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const P = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const Metric = struct {
        fn score(_: Allocator, _: *const Example, pred: *const optimize.RecordPrediction, _: ?*const Trace, _: ?[]const u8, _: ?*const Trace.Step) value.ParseError!Feedback {
            return .{ .score = if (pred.get("answer").?.string[0] == 'h') 1 else 0 };
        }
    };
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        "<answer>low0</answer>",  "<answer>high0</answer>",
        "<answer>high1</answer>", "<answer>high1</answer>",
        "<answer>high2</answer>", "<answer>low2</answer>",
    } };
    const ctx: Context = .{ .arena = arena, .lm = script.lm(), .json_fallback = false };
    const examples = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"q\"}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\":\"r\"}", &.{"question"}),
    };
    const program: P = .{};
    var front_scores: [2]f64 = undefined;
    var fronts: [2]std.ArrayList(usize) = @splat(.empty);
    var outputs: [2]std.ArrayList(GEPA.BestOutput) = @splat(.empty);
    var engine: GEPA.Engine(P) = .{
        .gepa = .{ .metric = Metric.score, .track_best_outputs = true },
        .out = arena,
        .ctx = ctx,
        .reflection = ctx,
        .student = &program,
        .names = &.{"self"},
        .trainset = &examples,
        .valset = &examples,
        .rng = .init(0),
        .adapter_rng = .init(0),
        .front_score = &front_scores,
        .fronts = &fronts,
        .best_outputs = &outputs,
    };
    try engine.addProgram(&.{"first"}, &.{null});
    try engine.addProgram(&.{"second"}, &.{0});
    try engine.addProgram(&.{"third"}, &.{1});
    const expected = [_][]const usize{ &.{ 1, 2 }, &.{ 0, 1 } };
    const answers = [_][]const []const u8{ &.{ "high1", "high2" }, &.{ "high0", "high1" } };
    for (fronts, outputs, expected, answers) |front, row, ids, texts| {
        try std.testing.expectEqualSlices(usize, ids, front.items);
        try std.testing.expectEqual(ids.len, row.items.len);
        for (row.items, ids, texts) |entry, id, text| {
            try std.testing.expectEqual(id, entry.candidate_idx);
            try std.testing.expectEqualStrings(text, entry.prediction.get("answer").?.string);
        }
    }
    try std.testing.expectEqual(@as(usize, 6), script.index);
}

fn sumScores(runs: []const GEPA.Run) f64 {
    var sum: f64 = 0;
    for (runs) |r| sum += r.score;
    return sum;
}

/// Python `str(float)`.
fn pyFloat(out: Allocator, x: f64) Allocator.Error![]const u8 {
    var aw: Writer.Allocating = .init(out);
    pyjson.writeStr(&aw.writer, .{ .float = x }) catch return error.OutOfMemory;
    return aw.written();
}

/// `ProposeInstruction.__doc__` (inspect.cleandoc), copied from DSPy 3.4.1.
const PROPOSE_INSTRUCTIONS =
    \\I provided an assistant with instructions to perform a task for me. You are given those instructions, along with examples of different task inputs provided to the assistant, the assistant's response for each of them, and some feedback on how the assistant's response could be better.
    \\
    \\Your task is to write a new instruction for the assistant.
    \\
    \\Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.
    \\
    \\Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.
;
