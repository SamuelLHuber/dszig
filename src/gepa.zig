//! `dspy.GEPA` (DSPy 3.5.0 on gepa 0.1.4): reflective prompt evolution.
//! Each iteration picks a parent with the configured selector, runs it
//! on a minibatch with traces, asks a reflection LM (`ProposeInstruction`,
//! JSONAdapter) for a better instruction from the metric's textual
//! feedback, and applies the configured acceptance criterion to the minibatch
//! scores. Accepted children are scored on the full valset. Merge combines two
//! Pareto programs that improved different predictors of a common ancestor.
//! Python's `random.Random(seed)` is ported (one stream for the engine, one
//! for the DSPy adapter), so the same LM answers give the same program.
//!
//! Not supported: `dspy.Flex` code components or `log_dir`.
//! DSPy `num_threads` maps to
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
const History = @import("history.zig").History;

pub const ObjectiveScore = struct { name: []const u8, score: f64 };
pub const ObjectiveLeaders = struct { name: []const u8, candidates: []const usize };

/// What a GEPA metric returns: DSPy's score, feedback and named objectives.
pub const Feedback = struct {
    score: f64,
    feedback: ?[]const u8 = null,
    /// Unique names in metric order. Return static or metric-arena storage.
    objective_scores: []const ObjectiveScore = &.{},
};

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

const CompressInstruction = signature.Signature(struct {
    pub const instructions =
        \\Shorten the instruction to fit within the stated limit. Preserve every
        \\strategy, rule, and constraint; cut redundancy and verbosity. Do not add
        \\new content. Output only the shortened instruction.
    ;
    pub const Input = struct { instruction: []const u8, length_limit: []const u8 };
    pub const Output = struct { shortened_instruction: []const u8 };
    pub const descriptions = .{ .instruction = "", .length_limit = "", .shortened_instruction = "" };
});

const ProposalControls = signature.Signature(struct {
    pub const Input = struct {
        reference_skills: []const u8,
        additional_instructions: []const u8,
        length_limit: []const u8,
    };
    pub const Output = struct { unused: []const u8 };
    pub const descriptions = .{
        .reference_skills = "Trusted reference material chosen by the user. Draw on it when writing the new instruction.",
        .additional_instructions = "Additional requirements from the user for the new instruction. Follow them.",
        .length_limit = "Hard limit on the length of the new instruction.",
    };
});

/// Loaded reference material. Applications resolve files and metadata at startup.
pub const ReflectionSkill = struct {
    name: []const u8,
    content: []const u8,
    description: []const u8 = "",
};

/// Options for the built-in reflection LM proposer. The caller owns the strings.
pub const ConfiguredInstructionProposer = struct {
    base_instructions: []const u8 = "",
    additional_instructions: []const u8 = "",
    skills: []const ReflectionSkill = &.{},
    max_chars: ?u32 = null,
    /// Compact tool observations in typed History fields, not ordinary JSON records.
    truncate_history_outputs: bool = false,
    adapter: @import("adapter.zig").Adapter = .json,

    /// Prepare a typed History for insertion into a standalone proposal record.
    /// Unchanged fields borrow from `history`; the input itself is never mutated.
    pub fn prepareHistory(
        options: ConfiguredInstructionProposer,
        arena: Allocator,
        history: History,
    ) value.ParseError!Value {
        if (!options.truncate_history_outputs) return history.toValue(arena);
        var messages: ?[]Record = null;
        for (history.messages, 0..) |message, index| {
            var copied = false;
            for (message.keys(), message.values()) |name, item| {
                if (item != .object or !item.object.contains("tool_calls") or
                    !item.object.contains("tool_call_results")) continue;
                const compacted = try compactToolObservations(arena, item) orelse continue;
                if (messages == null) messages = try arena.dupe(Record, history.messages);
                const output = &messages.?[index];
                if (!copied) output.* = try message.clone(arena);
                copied = true;
                try output.put(arena, name, compacted);
            }
        }
        return (History{ .messages = messages orelse history.messages }).toValue(arena);
    }

    pub fn propose(
        options: ConfiguredInstructionProposer,
        context: Context,
        current_instruction: []const u8,
        examples: []const Record,
    ) (predict.Error || error{ InvalidInstructionLimit, InvalidInstruction, InvalidReflectionSkill })![]const u8 {
        const max_chars = options.max_chars;
        if (max_chars == 0) return error.InvalidInstructionLimit;
        var ctx = context;
        ctx.adapter = options.adapter;
        ctx.trace = null;
        const length_limit = if (max_chars) |max|
            try std.fmt.allocPrint(ctx.arena, "The new instruction must be at most {d} characters.", .{max})
        else
            "";
        const draft = try options.generateDraft(ctx, current_instruction, examples, length_limit);
        if (draft.len == 0) return error.InvalidInstruction;
        const max = max_chars orelse return draft;
        if (try fitsInstruction(draft, max)) return draft;
        const compressor: predict.Predict(CompressInstruction) = .{};
        const output = try compressor.forward(ctx, .{ .instruction = draft, .length_limit = length_limit });
        const shortened = value.pyStrip(output.shortened_instruction);
        if (shortened.len == 0 or !(try fitsInstruction(shortened, max)))
            return error.InvalidInstruction;
        return shortened;
    }

    fn fitsInstruction(text: []const u8, max: u32) error{InvalidInstruction}!bool {
        const count = std.unicode.utf8CountCodepoints(text) catch return error.InvalidInstruction;
        return count <= max;
    }

    fn renderSkills(
        skills: []const ReflectionSkill,
        arena: Allocator,
    ) (Allocator.Error || error{InvalidReflectionSkill})![]const u8 {
        if (skills.len == 0) return "";
        if (skills.len > 64) return error.InvalidReflectionSkill;
        var bytes_left: usize = 256 * 1024;
        var output: std.Io.Writer.Allocating = .init(arena);
        defer output.deinit();
        for (skills, 0..) |skill, index| {
            for ([_][]const u8{ skill.name, skill.description, skill.content }) |text| {
                if (text.len > bytes_left) return error.InvalidReflectionSkill;
                bytes_left -= text.len;
                if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidReflectionSkill;
            }
            const content = value.pyStrip(skill.content);
            if (skill.name.len == 0 or content.len == 0) return error.InvalidReflectionSkill;
            writeSkill(&output.writer, skill, content, index != 0) catch return error.OutOfMemory;
        }
        return output.toOwnedSlice();
    }

    fn writeSkill(
        writer: *std.Io.Writer,
        skill: ReflectionSkill,
        content: []const u8,
        separator: bool,
    ) std.Io.Writer.Error!void {
        const repr = @import("pyjson.zig").writeReprString;
        if (separator) try writer.writeAll("\n\n");
        try writer.writeAll("<skill name=");
        try repr(writer, skill.name);
        if (skill.description.len > 0) {
            try writer.writeAll(" description=");
            try repr(writer, skill.description);
        }
        try writer.writeAll(">\n");
        try writer.writeAll(content);
        try writer.writeAll("\n</skill>");
    }

    fn generateDraft(
        options: ConfiguredInstructionProposer,
        ctx: Context,
        current_instruction: []const u8,
        examples: []const Record,
        length_limit: []const u8,
    ) (predict.Error || error{InvalidReflectionSkill})![]const u8 {
        const controls_count = ProposalControls.fields.len - 1;
        const capacity = controls_count + ProposeInstruction.fields.len;
        var fields: [capacity]signature.Field = undefined;
        var field_count: usize = 0;
        var inputs: Record = .empty;
        const skills = try renderSkills(options.skills, ctx.arena);
        const additional = value.pyStrip(options.additional_instructions);
        const controls = [controls_count][]const u8{ skills, additional, length_limit };
        for (ProposalControls.fields[0..controls_count], controls) |field, text| {
            if (text.len == 0) continue;
            fields[field_count] = field;
            field_count += 1;
            try inputs.put(ctx.arena, field.name, .{ .string = text });
        }
        @memcpy(fields[field_count..][0..ProposeInstruction.fields.len], ProposeInstruction.fields);
        field_count += ProposeInstruction.fields.len;
        var descs: [capacity][]const u8 = undefined;
        var prefixes: [capacity][]const u8 = undefined;
        for (fields[0..field_count], descs[0..field_count], prefixes[0..field_count]) |field, *desc, *prefix| {
            desc.* = field.desc;
            prefix.* = field.prefix;
        }
        const rows = try ctx.arena.alloc(Value, examples.len);
        for (examples, rows) |example, *row| row.* = .{ .object = example };
        try inputs.put(ctx.arena, "current_instruction", .{ .string = current_instruction });
        try inputs.put(ctx.arena, "examples_with_feedback", .{ .array = .fromOwnedSlice(ctx.arena, rows) });
        const proposer: predict.Predict(ProposeInstruction) = .{};
        var state = proposer.state;
        state.fields = fields[0..field_count];
        state.descs = descs[0..field_count];
        state.prefixes = prefixes[0..field_count];
        state.input_count = field_count - 1;
        const base = value.pyStrip(options.base_instructions);
        if (base.len > 0) state.instructions = base;
        const output = try state.call(ctx, &inputs);
        const text = output.get("new_instruction") orelse return error.AdapterParse;
        if (text != .string) return error.AdapterParse;
        return value.pyStrip(text.string);
    }
};

test "blank reflection skills are refused before a model request" {
    var model: @import("lm.zig").ScriptedLM = .{
        .responses = &.{"{\"new_instruction\":\"should not be requested\"}"},
    };
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const options: ConfiguredInstructionProposer = .{
        .skills = &.{.{ .name = "Blank", .content = "\u{2003}\n" }},
    };
    try std.testing.expectError(error.InvalidReflectionSkill, options.propose(
        .{ .arena = arena_state.allocator(), .lm = model.lm() },
        "Original.",
        &.{},
    ));
    try std.testing.expectEqual(@as(usize, 0), model.index);
}

pub const GEPA = struct {
    metric: FeedbackMetric,
    instruction_proposer: ?InstructionProposer = null,
    /// Used when `instruction_proposer` is null.
    proposal_options: ConfiguredInstructionProposer = .{},
    auto: ?Auto = null,
    max_full_evals: ?usize = null,
    max_metric_calls: ?usize = null,
    /// Checked in caller order after seed evaluation and between iterations.
    stop_callbacks: []const StopCallback = &.{},
    /// Defaults to three with the built-in sampler; leave null with `batch_sampler`.
    reflection_minibatch_size: ?usize = null,
    batch_sampler: ?BatchSampler = null,
    val_evaluation_policy: ?ValidationPolicy = null,
    merge_val_overlap_floor: usize = 5,
    candidate_selection_strategy: Selection = .pareto,
    frontier_type: Frontier = .instance,
    acceptance_criterion: Acceptance = .strict_improvement,
    skip_perfect_score: bool = true,
    /// Use genuine predictor parse failures as reflection examples.
    add_format_failure_as_feedback: bool = false,
    component_selector: ComponentSelector = .round_robin,
    use_merge: bool = true,
    max_merge_invocations: usize = 5,
    failure_score: f64 = 0.0,
    perfect_score: f64 = 1.0,
    seed: u32 = 0,
    /// Retain the predictions attaining the best score for each validation row.
    track_best_outputs: bool = false,
    /// Warn once per compile if predictor and program scores disagree.
    warn_on_score_mismatch: bool = true,
    /// DSPy `num_threads`: a `parallel.Parallel(GEPA.Run, limits).runner()`
    /// plus per-attempt contexts. Program and metric must be thread-safe, and
    /// `out`, which retains predictions and traces, must then be thread-safe
    /// (an ArenaAllocator over a thread-safe child is). null runs serially.
    rollouts: ?Rollouts = null,

    pub const Rollouts = parallel.Rollouts(Run);

    /// Choose training indices in evaluation order. Repeated indices are allowed.
    /// The returned slice must remain valid until the next sampler call or compile exit.
    pub const BatchSampler = struct {
        ptr: *anyopaque,
        max_batch_size: usize,
        next: *const fn (*anyopaque, *const BatchSample) anyerror![]const usize,
    };

    pub const BatchSample = struct {
        trainset: []const Example,
        parent_idx: usize,
        iteration: usize,
        total_metric_calls: usize,
    };

    pub const Auto = enum { light, medium, heavy };
    pub const Selection = union(enum) {
        pareto,
        current_best,
        epsilon_greedy,
        top_k_pareto,
        custom: CustomCandidateSelector,
    };

    pub const CustomCandidateSelector = struct {
        ptr: *anyopaque,
        select: *const fn (*anyopaque, *const CandidateSelection) anyerror!usize,
    };

    /// Borrowed for one callback before each reflection minibatch.
    /// Candidate instructions follow `names`; indices follow discovery order.
    pub const CandidateSelection = struct {
        names: []const []const u8,
        candidates: []const []const []const u8,
        val_subscores: []const []const f64,
        val_ids: ?[]const []const usize,
        parents: []const []const ?usize,
        discovery_eval_counts: []const usize,
        iteration: usize,
        total_metric_calls: usize,
        num_full_val_evals: usize,
        frontier_type: Frontier,
        frontier_count: usize,
        engine: *const anyopaque,
        get_front: *const fn (*const anyopaque, usize) []const usize,

        /// Validation leader indices in the active frontier's native order.
        pub fn front(self: *const CandidateSelection, index: usize) ?[]const usize {
            if (index >= self.frontier_count) return null;
            return self.get_front(self.engine, index);
        }
    };
    pub const Frontier = enum { instance, objective, hybrid, cartesian };
    pub const Acceptance = union(enum) {
        strict_improvement,
        improvement_or_equal,
        custom: CustomAcceptance,
    };

    pub const CustomAcceptance = struct {
        ptr: *anyopaque,
        should_accept: *const fn (*anyopaque, *const AcceptanceInput) anyerror!bool,
    };

    /// Borrowed for one callback after both minibatch runs and before validation.
    /// Candidate instructions follow `names`; run entries follow `minibatch_ids`.
    pub const AcceptanceInput = struct {
        names: []const []const u8,
        child: []const []const u8,
        parent_idx: usize,
        iteration: usize,
        total_metric_calls: usize,
        candidates: []const []const []const u8,
        val_subscores: []const []const f64,
        val_ids: ?[]const []const usize,
        parents: []const []const ?usize,
        trainset: []const Example,
        minibatch_ids: []const usize,
        before: []const Run,
        after: []const Run,
    };
    pub const ComponentSelector = union(enum) {
        round_robin,
        all,
        custom: CustomComponentSelector,
    };

    /// Candidate instructions follow `names`; runs follow `minibatch_ids`.
    /// All slices and their contents are borrowed for the selector invocation.
    pub const ComponentSelection = struct {
        names: []const []const u8,
        candidates: []const []const []const u8,
        val_subscores: []const []const f64,
        val_ids: ?[]const []const usize,
        parents: []const []const ?usize,
        parent_idx: usize,
        total_metric_calls: usize,
        trainset: []const Example,
        minibatch_ids: []const usize,
        runs: []const Run,
    };

    /// Return distinct predictor names in reflection order, using static
    /// storage or the provided scratch arena. An empty slice skips reflection.
    pub const CustomComponentSelector = struct {
        ptr: *anyopaque,
        select: *const fn (
            *anyopaque,
            Context,
            *const ComponentSelection,
        ) anyerror![]const []const u8,
    };

    /// Read-only snapshot borrowed for the duration of a stop callback.
    pub const StopState = struct {
        last_iteration: ?usize,
        total_metric_calls: usize,
        num_full_val_evals: usize,
        max_metric_calls: usize,
        candidates: []const []const []const u8,
        val_subscores: []const []const f64,
        val_ids: ?[]const []const usize,
        parents: []const []const ?usize,
        discovery_eval_counts: []const usize,
    };

    pub const StopCallback = struct {
        ptr: *anyopaque,
        should_stop: *const fn (*anyopaque, *const StopState) anyerror!bool,
    };

    pub const Options = struct {
        /// Defaults to the trainset, as in DSPy.
        valset: ?[]const Example = null,
        /// DSPy's `reflection_lm` (defaults to `ctx`).
        reflection: ?Context = null,
    };

    pub const Error = error{
        InvalidValidationBatch,
        ZeroWeights,
        NoBudget,
        UnknownInstructionComponent,
        InvalidComponentSelection,
        InvalidCandidateSelection,
        InvalidBatchSamplerConfiguration,
        InvalidBatchSample,
    };

    pub const Instruction = struct {
        name: []const u8,
        text: []const u8,
    };

    pub const ReflectionDataset = struct {
        name: []const u8,
        examples: []const Record,
    };

    pub const InstructionProposal = struct {
        candidate: []const Instruction,
        reflective_dataset: []const ReflectionDataset,
        components_to_update: []const []const u8,
    };

    /// Called once per reflection task, with a retry on non-allocation failure.
    /// Inputs are borrowed. Return static data or allocate updates in ctx.arena;
    /// GEPA copies accepted text before releasing that scratch arena.
    /// An empty slice skips evaluation; any existing component can be updated.
    pub const InstructionProposer = struct {
        ptr: *anyopaque,
        propose: *const fn (*anyopaque, Context, *const InstructionProposal) anyerror![]const Instruction,
    };

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
            /// Null with full validation. Otherwise, row IDs in score order.
            val_ids: ?[]const []const usize,
            val_subscores: []const []const f64,
            /// Indexed by candidate; each objective averages only rows that supplied it.
            val_aggregate_subscores: ?[]const []const ObjectiveScore,
            per_objective_best_candidates: ?[]const ObjectiveLeaders,
            objective_pareto_front: ?[]const ObjectiveScore,
            /// Indexed by validation row; tied candidates are in discovery order.
            per_val_instance_best_candidates: []const []const usize,
            /// Indexed by validation row, in candidate discovery order.
            best_outputs_valset: ?[]const []const BestOutput,
            discovery_eval_counts: []const usize,
            best_idx: usize,
            total_metric_calls: usize,
            num_full_val_evals: usize,
            max_metric_calls: usize,
            seed: u32,
        };
    }

    /// Batch selection from a fixed ordered validation set. Returned IDs must
    /// stay readable through the selected batch's evaluation.
    pub const ValidationPolicy = struct {
        ptr: *anyopaque,
        /// Custom batches must be nonempty and fit this limit.
        max_batch_size: usize,
        seed_batch: ?*const fn (*anyopaque, []const Example) anyerror![]const usize = null,
        next_batch: *const fn (*anyopaque, *const ValidationState) anyerror![]const usize,
    };

    /// Borrowed during the callback. Score rows follow their matching ID rows.
    pub const ValidationState = struct {
        valset: []const Example,
        candidates: []const []const []const u8,
        val_ids: []const []const usize,
        val_subscores: []const []const f64,
        parents: []const []const ?usize,
        iteration: usize,
        total_metric_calls: usize,
    };
    const merge_subsample = 5;
    const max_attempts = 10;

    /// The reflective dataset serializes trace-step records and feedback
    /// only, so a prediction's usage reaches the metric and nothing else.
    pub const Run = struct {
        prediction: ?optimize.RecordPrediction,
        trace: *const Trace,
        score: f64,
        format_failure: ?FormatFailure = null,
        objective_scores: []const ObjectiveScore = &.{},
    };

    pub const FormatFailure = struct {
        predictor: *const predict.PredictState,
        inputs: Record,
        text: []const u8,
    };

    const Desc = struct { id1: usize, id2: usize, picks: []const usize };

    const Leader = struct {
        score: ?f64 = null,
        candidates: std.ArrayList(usize) = .empty,

        fn update(leader: *Leader, out: Allocator, candidate: usize, score: f64) !void {
            const previous = leader.score orelse -std.math.inf(f64);
            if (candidate == 0 or score > previous) {
                leader.score = score;
                leader.candidates.clearRetainingCapacity();
            } else if (score != previous) return;
            try leader.candidates.append(out, candidate);
        }
    };

    const Objectives = struct {
        named: std.StringArrayHashMapUnmanaged(Leader) = .empty,
        cartesian: std.AutoArrayHashMapUnmanaged([2]usize, Leader) = .empty,
        aggregates: std.ArrayList([]const ObjectiveScore) = .empty,
        totals: std.StringArrayHashMapUnmanaged(Total) = .empty,

        const Total = struct { sum: f64 = 0, count: usize = 0, last_row: ?usize = null };

        fn update(
            state: *Objectives,
            out: Allocator,
            mode: Frontier,
            candidate: usize,
            runs: []const Run,
            ids: ?[]const usize,
        ) !void {
            assert(ids == null or ids.?.len == runs.len);
            const totals = &state.totals;
            defer totals.clearRetainingCapacity();
            for (runs, 0..) |run, row| for (run.objective_scores) |entry| {
                const total = try totals.getOrPut(out, entry.name);
                if (!total.found_existing) total.value_ptr.* = .{};
                if (total.value_ptr.last_row == row) return error.DuplicateObjective;
                total.value_ptr.last_row = row;
                total.value_ptr.sum += entry.score;
                total.value_ptr.count += 1;
            };
            if (totals.count() == 0) {
                if (mode != .instance) return error.MissingObjectiveScores;
                if (state.aggregates.items.len != 0) try state.aggregates.append(out, &.{});
                return;
            }
            if (state.aggregates.items.len == 0) try state.aggregates.appendNTimes(out, &.{}, candidate);
            assert(state.aggregates.items.len == candidate);
            const aggregate = try out.alloc(ObjectiveScore, totals.count());
            for (aggregate, totals.keys(), totals.values()) |*entry, name, total| {
                const score = total.sum / @as(f64, @floatFromInt(total.count));
                const named = try state.named.getOrPut(out, name);
                if (!named.found_existing) {
                    named.key_ptr.* = try out.dupe(u8, name);
                    named.value_ptr.* = .{};
                }
                try named.value_ptr.update(out, candidate, score);
                entry.* = .{ .name = named.key_ptr.*, .score = score };
            }
            try state.aggregates.append(out, aggregate);
            if (mode == .cartesian) {
                for (runs, 0..) |run, row| for (run.objective_scores) |entry| {
                    const id = if (ids) |rows| rows[row] else row;
                    const key = [2]usize{ id, state.named.getIndex(entry.name).? };
                    const front = try state.cartesian.getOrPut(out, key);
                    if (!front.found_existing) front.value_ptr.* = .{};
                    try front.value_ptr.update(out, candidate, entry.score);
                };
            }
        }

        fn results(state: *const Objectives, out: Allocator) !struct {
            leaders: ?[]const ObjectiveLeaders,
            scores: ?[]const ObjectiveScore,
        } {
            var leaders: std.ArrayList(ObjectiveLeaders) = .empty;
            var scores: std.ArrayList(ObjectiveScore) = .empty;
            for (state.named.keys(), state.named.values()) |name, front| {
                if (front.candidates.items.len > 0) try leaders.append(out, .{
                    .name = name,
                    .candidates = front.candidates.items,
                });
                if (front.score) |score| try scores.append(out, .{ .name = name, .score = score });
            }
            return .{
                .leaders = if (leaders.items.len == 0) null else leaders.items,
                .scores = if (scores.items.len == 0) null else scores.items,
            };
        }
    };

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
            all_val_ids: []const usize = &.{},
            val_ids: std.ArrayList([]const usize) = .empty,
            val_indices: std.ArrayList(std.AutoArrayHashMapUnmanaged(usize, usize)) = .empty,
            val_front_order: std.ArrayList(usize) = .empty,
            parents: std.ArrayList([]const ?usize) = .empty,
            discovery: std.ArrayList(usize) = .empty,
            next_pred: std.ArrayList(usize) = .empty,
            front_score: []f64,
            fronts: []std.ArrayList(usize),
            objectives: Objectives = .{},
            best_outputs: ?[]std.ArrayList(BestOutput) = null,
            total_evals: usize = 0,
            full_evals: usize = 0,
            i: usize = 0,
            score_mismatch_warned: bool = false,
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

            fn shouldStop(e: *const Self, last_iteration: ?usize, max_calls: usize) !bool {
                if (e.gepa.stop_callbacks.len > 0) {
                    const state: StopState = .{
                        .last_iteration = last_iteration,
                        .total_metric_calls = e.total_evals,
                        .num_full_val_evals = e.full_evals,
                        .max_metric_calls = max_calls,
                        .candidates = e.candidates.items,
                        .val_subscores = e.subscores.items,
                        .val_ids = e.validationIds(),
                        .parents = e.parents.items,
                        .discovery_eval_counts = e.discovery.items,
                    };
                    for (e.gepa.stop_callbacks) |callback| {
                        if (try callback.should_stop(callback.ptr, &state)) return true;
                    }
                }
                return e.total_evals >= max_calls;
            }

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

            const Built = struct { program: *P, predictors: []const predict.NamedParam };

            fn build(e: *const Self, candidate: []const []const u8) Allocator.Error!Built {
                const copy = try e.out.create(P);
                copy.* = e.student.*;
                const preds = try predict.namedPredictors(e.out, copy);
                assert(preds.len == candidate.len);
                for (preds, candidate) |p, text| {
                    try p.param.predict.forkSignatureClass(e.out);
                    p.param.predict.instructions = text;
                }
                return .{ .program = copy, .predictors = preds };
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

            fn runOne(
                e: *const Self,
                ctx: Context,
                built: *const Built,
                example: *const Example,
                capture: bool,
            ) Allocator.Error!Run {
                const trace = try e.out.create(Trace);
                trace.* = .{ .arena = e.out };
                var c = ctx;
                c.arena = e.out;
                c.trace = if (capture) trace else null;
                var failure: predict.ParseFailure = .{};
                const reflect_failure = capture and e.gepa.add_format_failure_as_feedback;
                if (reflect_failure) c.parse_failure = &failure;
                const prediction = optimize.runProgram(P, c, built.program, example) catch |err| {
                    if (err == error.OutOfMemory) return error.OutOfMemory;
                    var run = try e.failedRun(trace, capture);
                    if (reflect_failure and err == error.AdapterParse) {
                        run.format_failure = try e.formatFailure(built.predictors, example, &failure);
                    }
                    return run;
                };
                const feedback = e.gepa.metric(e.out, example, &prediction, if (capture) trace else null, null, null) catch |err| {
                    if (err == error.OutOfMemory) return error.OutOfMemory;
                    return e.failedRun(trace, capture);
                };
                return .{
                    .prediction = prediction,
                    .trace = trace,
                    .score = feedback.score,
                    .objective_scores = feedback.objective_scores,
                };
            }

            fn formatFailure(
                e: *const Self,
                predictors: []const predict.NamedParam,
                example: *const Example,
                failure: *const predict.ParseFailure,
            ) Allocator.Error!?FormatFailure {
                const class = failure.signature_class orelse return null;
                for (predictors) |named| {
                    const predictor = named.param.predict;
                    if (predictor.signature_class != class) continue;
                    return .{
                        .predictor = predictor,
                        .inputs = try example.inputs(e.out),
                        .text = failure.failure.completion_text,
                    };
                }
                return null;
            }

            fn evalIds(e: *Self, candidate: []const []const u8, set: []const Example, ids: []const usize, capture: bool) ![]Run {
                const built = try e.build(candidate);
                const runs = try e.out.alloc(Run, ids.len);
                if (e.gepa.rollouts) |rollouts| {
                    const Call = struct {
                        engine: *const Self,
                        built: *const Built,
                        example: *const Example,
                        capture: bool,

                        fn invoke(ptr: *anyopaque, attempt: parallel.Attempt, run: *Run) anyerror!void {
                            const call: *const @This() = @ptrCast(@alignCast(ptr));
                            try attempt.cancellation.check();
                            const en = call.engine;
                            const c = try en.gepa.rollouts.?.context(en.ctx, attempt, false);
                            run.* = try en.runOne(c, call.built, call.example, call.capture);
                        }
                    };
                    const calls = try e.out.alloc(Call, ids.len);
                    const tasks = try e.out.alloc(parallel.TaskFor(Run), ids.len);
                    for (calls, tasks, ids) |*call, *task, id| {
                        call.* = .{ .engine = e, .built = &built, .example = &set[id], .capture = capture };
                        task.* = .{ .ptr = call, .invoke = Call.invoke };
                    }
                    try rollouts.run(tasks, runs);
                } else {
                    for (runs, ids) |*r, id| r.* = try e.runOne(e.ctx, &built, &set[id], capture);
                }
                e.total_evals += ids.len;
                return runs;
            }

            fn validationIds(e: *const Self) ?[]const []const usize {
                return if (e.gepa.val_evaluation_policy != null) e.val_ids.items else null;
            }

            fn validationBatch(e: *const Self) ![]const usize {
                const policy = e.gepa.val_evaluation_policy orelse return e.all_val_ids;
                const ids = if (e.candidates.items.len == 0)
                    if (policy.seed_batch) |seed| try seed(policy.ptr, e.valset) else return e.all_val_ids
                else
                    try policy.next_batch(policy.ptr, &.{
                        .valset = e.valset,
                        .candidates = e.candidates.items,
                        .val_ids = e.val_ids.items,
                        .val_subscores = e.subscores.items,
                        .parents = e.parents.items,
                        .iteration = e.i,
                        .total_metric_calls = e.total_evals,
                    });
                if (ids.len == 0 or ids.len > policy.max_batch_size)
                    return error.InvalidValidationBatch;
                for (ids) |id| if (id >= e.valset.len) return error.InvalidValidationBatch;
                return ids;
            }

            fn addProgram(e: *Self, candidate: []const []const u8, parent_ids: []const ?usize) !void {
                const requested = try e.validationBatch();
                const discovered = e.total_evals;
                var runs = try e.evalIds(candidate, e.valset, requested, false);
                const sparse = e.gepa.val_evaluation_policy != null;
                const ids = if (sparse) try e.retainValidationRows(requested, &runs) else requested;
                const idx = e.candidates.items.len;
                // Source seeds the metric counter from the number of unique scored IDs.
                // Subsequent evaluations count every rollout sent to the adapter.
                if (idx == 0) e.total_evals = ids.len;
                e.full_evals += 1;
                try e.objectives.update(e.out, e.gepa.frontier_type, idx, runs, if (sparse) ids else null);
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
                for (runs, ids) |run, k| {
                    const first = e.fronts[k].items.len == 0;
                    const improves = first or run.score > e.front_score[k];
                    if (first and sparse) try e.val_front_order.append(e.out, k);
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

            fn retainValidationRows(e: *Self, ids: []const usize, runs: *[]Run) ![]const usize {
                assert(ids.len == runs.len);
                // The first occurrence determines order; the last rollout supplies its value.
                var indices: std.AutoArrayHashMapUnmanaged(usize, usize) = .empty;
                for (ids, 0..) |id, i| try indices.put(e.out, id, i);
                for (indices.values(), 0..) |*last, i| {
                    assert(last.* >= i);
                    runs.*[i] = runs.*[last.*];
                    last.* = i;
                }
                const retained = indices.keys();
                runs.* = runs.*[0..retained.len];
                try e.val_ids.append(e.out, retained);
                if (e.gepa.use_merge) try e.val_indices.append(e.out, indices);
                return retained;
            }

            fn instanceFrontCount(e: *const Self) usize {
                return if (e.gepa.val_evaluation_policy != null)
                    e.val_front_order.items.len
                else
                    e.fronts.len;
            }

            fn instanceFront(e: *const Self, index: usize) []const usize {
                const id = if (e.gepa.val_evaluation_policy != null)
                    e.val_front_order.items[index]
                else
                    index;
                return e.fronts[id].items;
            }

            fn activeFrontCount(e: *const Self) usize {
                return switch (e.gepa.frontier_type) {
                    .instance => e.instanceFrontCount(),
                    .objective => e.objectives.named.count(),
                    .hybrid => e.instanceFrontCount() + e.objectives.named.count(),
                    .cartesian => e.objectives.cartesian.count(),
                };
            }

            fn activeFront(e: *const Self, index: usize) []const usize {
                assert(index < e.activeFrontCount());
                return switch (e.gepa.frontier_type) {
                    .instance => e.instanceFront(index),
                    .objective => e.objectives.named.values()[index].candidates.items,
                    .hybrid => if (index < e.instanceFrontCount())
                        e.instanceFront(index)
                    else
                        e.objectives.named.values()[index - e.instanceFrontCount()].candidates.items,
                    .cartesian => e.objectives.cartesian.values()[index].candidates.items,
                };
            }

            /// `remove_dominated_programs`: returns the dominated flags.
            fn dominated(
                e: *const Self,
                eligible: ?[]const usize,
                ranked_scores: ?[]const f64,
            ) Allocator.Error![]bool {
                const n = e.candidates.items.len;
                const flags = try e.out.alloc(bool, n);
                @memset(flags, eligible != null);
                if (eligible) |ids| for (ids) |id| {
                    flags[id] = false;
                };
                var programs: std.ArrayList(usize) = .empty;
                for (0..e.activeFrontCount()) |index| for (e.activeFront(index)) |p| {
                    if (flags[p]) continue;
                    if (std.mem.indexOfScalar(usize, programs.items, p) == null)
                        try programs.append(e.out, p);
                };
                if (programs.items.len == 0) return flags;
                const scores = ranked_scores orelse try e.averages();
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
                for (0..e.activeFrontCount()) |index| {
                    const front = e.activeFront(index);
                    if (std.mem.indexOfScalar(usize, front, y) == null) continue;
                    const covered = for (front) |p| {
                        if (p != y and !flags[p]) break true;
                    } else false;
                    if (!covered) return false;
                }
                return true;
            }

            /// `select_program_candidate_from_pareto_front`.
            fn selectPareto(
                e: *Self,
                eligible: ?[]const usize,
                ranked_scores: ?[]const f64,
            ) Allocator.Error!usize {
                const flags = try e.dominated(eligible, ranked_scores);
                var order: std.ArrayList(usize) = .empty;
                var freq: std.ArrayList(usize) = .empty;
                for (0..e.activeFrontCount()) |index| for (e.activeFront(index)) |p| {
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
                if (sampling.items.len == 0) {
                    const ranked = eligible orelse unreachable;
                    assert(ranked.len > 0);
                    return ranked[0];
                }
                return sampling.items[e.rng.randbelow(sampling.items.len)];
            }

            fn selectParent(e: *Self) !usize {
                assert(e.candidates.items.len > 0);
                return switch (e.gepa.candidate_selection_strategy) {
                    .pareto => e.selectPareto(null, null),
                    .current_best => e.selectBest(),
                    .epsilon_greedy => if (e.rng.random() < 0.1)
                        e.rng.randbelow(e.candidates.items.len)
                    else
                        e.selectBest(),
                    .top_k_pareto => e.selectTopPareto(),
                    .custom => |selector| e.selectCustomParent(selector),
                };
            }

            fn selectCustomParent(e: *const Self, selector: CustomCandidateSelector) !usize {
                const selection: CandidateSelection = .{
                    .names = e.names,
                    .candidates = e.candidates.items,
                    .val_subscores = e.subscores.items,
                    .val_ids = e.validationIds(),
                    .parents = e.parents.items,
                    .discovery_eval_counts = e.discovery.items,
                    .iteration = e.i,
                    .total_metric_calls = e.total_evals,
                    .num_full_val_evals = e.full_evals,
                    .frontier_type = e.gepa.frontier_type,
                    .frontier_count = e.activeFrontCount(),
                    .engine = e,
                    .get_front = selectionFront,
                };
                const index = try selector.select(selector.ptr, &selection);
                if (index >= e.candidates.items.len) return error.InvalidCandidateSelection;
                return index;
            }

            fn selectionFront(ptr: *const anyopaque, index: usize) []const usize {
                const e: *const Self = @ptrCast(@alignCast(ptr));
                return e.activeFront(index);
            }

            // Stable top five; equal scores retain discovery order.
            fn selectTopPareto(e: *Self) Allocator.Error!usize {
                var top: [5]usize = undefined;
                const scores = try e.averages();
                var len: usize = 0;
                for (scores, 0..) |score, id| {
                    var slot: usize = 0;
                    while (slot < len) : (slot += 1) {
                        if (score > scores[top[slot]]) break;
                    }
                    if (slot == top.len) continue;
                    len = @min(len + 1, top.len);
                    std.mem.copyBackwards(usize, top[slot + 1 .. len], top[slot .. len - 1]);
                    top[slot] = id;
                }
                assert(len == @min(top.len, e.candidates.items.len));
                return e.selectPareto(top[0..len], scores);
            }

            fn selectBest(e: *const Self) usize {
                var best: usize = 0;
                for (0..e.candidates.items.len) |idx| {
                    if (e.average(idx) > e.average(best)) best = idx;
                }
                return best;
            }

            /// `EpochShuffledBatchSampler.next_minibatch_ids`.
            fn nextMinibatch(e: *Self, parent: usize) ![]const usize {
                if (e.gepa.batch_sampler) |sampler| {
                    const sample: BatchSample = .{
                        .trainset = e.trainset,
                        .parent_idx = parent,
                        .iteration = e.i,
                        .total_metric_calls = e.total_evals,
                    };
                    const ids = try sampler.next(sampler.ptr, &sample);
                    if (ids.len == 0 or ids.len > sampler.max_batch_size)
                        return error.InvalidBatchSample;
                    for (ids) |id| if (id >= e.trainset.len) return error.InvalidBatchSample;
                    return ids;
                }
                const mb = e.gepa.reflection_minibatch_size orelse 3;
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
                const mb = e.gepa.reflection_minibatch_size orelse 3;
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

            fn selectComponents(
                e: *Self,
                parent: usize,
                ids: []const usize,
                runs: []const Run,
                single: *[1]usize,
            ) ![]const usize {
                assert(e.names.len > 0);
                assert(ids.len == runs.len);
                switch (e.gepa.component_selector) {
                    .round_robin => {
                        const index = e.next_pred.items[parent];
                        e.next_pred.items[parent] = (index + 1) % e.names.len;
                        single[0] = index;
                        return single;
                    },
                    .all => {
                        const selected = try e.out.alloc(usize, e.names.len);
                        for (selected, 0..) |*index, i| index.* = i;
                        return selected;
                    },
                    .custom => |selector| {
                        var scratch = std.heap.ArenaAllocator.init(e.out);
                        defer scratch.deinit();
                        var ctx = e.ctx;
                        ctx.arena = scratch.allocator();
                        ctx.trace = null;
                        const input: ComponentSelection = .{
                            .names = e.names,
                            .candidates = e.candidates.items,
                            .val_subscores = e.subscores.items,
                            .val_ids = e.validationIds(),
                            .parents = e.parents.items,
                            .parent_idx = parent,
                            .total_metric_calls = e.total_evals,
                            .trainset = e.trainset,
                            .minibatch_ids = ids,
                            .runs = runs,
                        };
                        const names = try selector.select(selector.ptr, ctx, &input);
                        if (names.len > e.names.len) return error.InvalidComponentSelection;
                        const selected = try e.out.alloc(usize, names.len);
                        for (names, selected, 0..) |name, *slot, i| {
                            slot.* = for (e.names, 0..) |known, index| {
                                if (std.mem.eql(u8, known, name)) break index;
                            } else return error.InvalidComponentSelection;
                            if (std.mem.indexOfScalar(usize, selected[0..i], slot.*) != null)
                                return error.InvalidComponentSelection;
                        }
                        return selected;
                    },
                }
            }

            /// One `ReflectiveMutationProposer.propose` + acceptance.
            fn reflect(e: *Self) !void {
                const parent = try e.selectParent();
                const ids = try e.nextMinibatch(parent);
                const parent_candidate = e.candidates.items[parent];
                const before = try e.evalIds(parent_candidate, e.trainset, ids, true);
                if (e.gepa.skip_perfect_score) {
                    const perfect = for (before) |r| {
                        if (r.score < e.gepa.perfect_score) break false;
                    } else true;
                    if (perfect) return;
                }
                var single: [1]usize = undefined;
                const components = try e.selectComponents(parent, ids, before, &single);
                if (components.len == 0) return;
                const datasets = try e.reflectiveDataset(
                    parent_candidate,
                    before,
                    ids,
                    components,
                );
                for (datasets) |dataset| {
                    if (dataset != null) break;
                } else return;
                const proposed = if (e.gepa.instruction_proposer) |proposer|
                    try e.proposeCustomChild(proposer, parent_candidate, components, datasets)
                else
                    try e.proposeBuiltinChild(parent_candidate, components, datasets);
                const child = proposed orelse return;
                const after = try e.evalIds(child, e.trainset, ids, true);
                const accepted = switch (e.gepa.acceptance_criterion) {
                    .strict_improvement => sumScores(after) > sumScores(before),
                    .improvement_or_equal => sumScores(after) >= sumScores(before),
                    .custom => |callback| try callback.should_accept(callback.ptr, &.{
                        .names = e.names,
                        .child = child,
                        .parent_idx = parent,
                        .iteration = e.i,
                        .total_metric_calls = e.total_evals,
                        .candidates = e.candidates.items,
                        .val_subscores = e.subscores.items,
                        .val_ids = e.validationIds(),
                        .parents = e.parents.items,
                        .trainset = e.trainset,
                        .minibatch_ids = ids,
                        .before = before,
                        .after = after,
                    }),
                };
                if (accepted) {
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
            fn reflectiveDataset(
                e: *Self,
                candidate: []const []const u8,
                runs: []const Run,
                ids: []const usize,
                components: []const usize,
            ) @import("adapter.zig").Error![]?[]const Record {
                assert(runs.len == ids.len);
                const preds = (try e.build(candidate)).predictors;
                const result = try e.out.alloc(?[]const Record, components.len);
                for (components, result) |k, *slot| {
                    const module = preds[k].param.predict;
                    var items: std.ArrayList(Record) = .empty;
                    for (runs, ids) |run, id| {
                        if (run.format_failure) |failure| {
                            if (predict.sameSignature(failure.predictor, module)) {
                                try items.append(e.out, try e.failedReflectionExample(failure, module));
                            }
                            continue;
                        }
                        const prediction = run.prediction orelse continue;
                        var matching: std.ArrayList(*const Trace.Step) = .empty;
                        for (run.trace.steps.items) |*step| if (predict.sameSignature(step.predictor, module)) try matching.append(e.out, step);
                        if (matching.items.len == 0) continue;
                        const step = matching.items[e.adapter_rng.randbelow(matching.items.len)];
                        const fb = try e.gepa.metric(e.out, &e.trainset[id], &prediction, run.trace, e.names[k], step);
                        if (e.gepa.warn_on_score_mismatch and !e.score_mismatch_warned and
                            fb.score != run.score)
                        {
                            std.log.scoped(.gepa).warn(
                                "Predictor feedback score differs from program score; " ++
                                    "GEPA uses the program score for selection.",
                                .{},
                            );
                            e.score_mismatch_warned = true;
                        }
                        var item: Record = .empty;
                        const custom = e.gepa.instruction_proposer != null;
                        const inputs = if (custom)
                            try customInputs(e.out, step)
                        else
                            try builtinInputs(e.out, e.gepa.proposal_options, step);
                        const outputs = if (custom)
                            try customOutputs(e.out, step)
                        else
                            try builtinOutputs(e.out, e.gepa.proposal_options, step);
                        try item.put(e.out, "Inputs", .{ .object = inputs });
                        try item.put(e.out, "Generated Outputs", .{ .object = outputs });
                        const text = fb.feedback orelse try std.fmt.allocPrint(e.out, "This trajectory got a score of {s}.", .{try pyFloat(e.out, fb.score)});
                        try item.put(e.out, "Feedback", .{ .string = text });
                        try items.append(e.out, item);
                    }
                    slot.* = if (items.items.len == 0) null else items.items;
                }
                return result;
            }

            fn failedReflectionExample(
                e: *const Self,
                failure: FormatFailure,
                module: *const predict.PredictState,
            ) @import("adapter.zig").Error!Record {
                const step: Trace.Step = .{
                    .predictor = failure.predictor,
                    .inputs = failure.inputs,
                    .outputs = .empty,
                };
                const inputs = if (e.gepa.instruction_proposer != null)
                    try customInputs(e.out, &step)
                else
                    try builtinInputs(e.out, e.gepa.proposal_options, &step);
                const text = try std.fmt.allocPrint(
                    e.out,
                    "Couldn't parse the output as per the expected output format. " ++
                        "The model's raw response was:\n```\n{s}\n```\n\n",
                    .{failure.text},
                );
                const view = try module.view(e.out);
                const messages = try e.ctx.adapter.format(e.out, view, &.{}, &.empty);
                var feedback: Writer.Allocating = .init(e.out);
                defer feedback.deinit();
                const w = &feedback.writer;
                w.writeAll("Your output failed to parse. Follow this structure:\n") catch
                    return error.OutOfMemory;
                for (messages) |message| {
                    if (message.content != .text) return error.AdapterParse;
                    w.print("{s}: {s}\n", .{ @tagName(message.role), message.content.text }) catch
                        return error.OutOfMemory;
                }
                var item: Record = .empty;
                try item.put(e.out, "Inputs", .{ .object = inputs });
                try item.put(e.out, "Generated Outputs", .{ .string = text });
                try item.put(e.out, "Feedback", .{ .string = try feedback.toOwnedSlice() });
                return item;
            }

            fn proposeCustomChild(
                e: *Self,
                proposer: InstructionProposer,
                candidate: []const []const u8,
                components: []const usize,
                datasets: []const ?[]const Record,
            ) !?[]const []const u8 {
                assert(candidate.len == e.names.len);
                assert(components.len == datasets.len);
                var scratch = std.heap.ArenaAllocator.init(e.out);
                defer scratch.deinit();
                const arena = scratch.allocator();
                const named = try arena.alloc(Instruction, candidate.len);
                for (e.names, candidate, named) |name, text, *entry| {
                    entry.* = .{ .name = name, .text = text };
                }
                const selected = try arena.alloc([]const u8, components.len);
                const entries = try arena.alloc(ReflectionDataset, components.len);
                var dataset_count: usize = 0;
                for (components, datasets, selected) |index, dataset, *name| {
                    name.* = e.names[index];
                    if (dataset) |examples| {
                        entries[dataset_count] = .{ .name = name.*, .examples = examples };
                        dataset_count += 1;
                    }
                }
                const input: InstructionProposal = .{
                    .candidate = named,
                    .reflective_dataset = entries[0..dataset_count],
                    .components_to_update = selected,
                };
                var context = e.reflection;
                context.arena = arena;
                context.trace = null;
                const updates = (try invokeProposer(proposer, context, &input)) orelse
                    (try invokeProposer(proposer, context, &input)) orelse return null;
                if (updates.len == 0) return null;
                const child = try e.out.dupe([]const u8, candidate);
                for (updates) |update| {
                    const index = for (e.names, 0..) |name, index| {
                        if (std.mem.eql(u8, name, update.name)) break index;
                    } else return error.UnknownInstructionComponent;
                    child[index] = try e.out.dupe(u8, update.text);
                }
                return child;
            }

            fn proposeBuiltinChild(
                e: *Self,
                candidate: []const []const u8,
                components: []const usize,
                datasets: []const ?[]const Record,
            ) Allocator.Error!?[]const []const u8 {
                const texts = (try e.proposeTexts(candidate, components, datasets)) orelse
                    (try e.proposeTexts(candidate, components, datasets)) orelse return null;
                const child = try e.out.dupe([]const u8, candidate);
                // At least one dataset reached a successful built-in proposal.
                for (components, texts) |index, text| {
                    if (text) |instruction| child[index] = instruction;
                }
                return child;
            }

            /// `InstructionProposer.__call__`; `null` when a call fails.
            fn proposeTexts(
                e: *Self,
                candidate: []const []const u8,
                components: []const usize,
                datasets: []const ?[]const Record,
            ) Allocator.Error!?[]?[]const u8 {
                const texts = try e.out.alloc(?[]const u8, components.len);
                for (components, datasets, texts) |k, dataset, *t| {
                    t.* = null;
                    const examples = dataset orelse continue;
                    var c = e.reflection;
                    c.arena = e.out;
                    const draft = e.gepa.proposal_options.propose(c, candidate[k], examples) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return null,
                    };
                    t.* = draft;
                }
                return texts;
            }

            /// `MergeProposer.propose` + the engine's merge acceptance.
            /// Returns whether a merge was evaluated (the iteration ends).
            fn merge(e: *Self) !bool {
                const flags = try e.dominated(null, null);
                var dominators: std.ArrayList(usize) = .empty;
                for (0..e.candidates.items.len) |p| {
                    if (flags[p]) continue;
                    const on_front = for (0..e.activeFrontCount()) |index| {
                        if (std.mem.indexOfScalar(usize, e.activeFront(index), p) != null) break true;
                    } else false;
                    if (on_front) try dominators.append(e.out, p);
                }
                const agg = try e.averages();
                const attempt = (try e.attemptMerge(dominators.items, agg)) orelse return false;
                try e.triplets.append(e.out, .{ attempt.id1, attempt.id2, attempt.ancestor });
                const common = try e.commonValidationIds(attempt.id1, attempt.id2);
                const ids = try e.mergeSubsample(attempt.id1, attempt.id2, common);
                if (ids.len == 0) return false;
                var sum1: f64 = 0;
                var sum2: f64 = 0;
                for (ids) |id| {
                    sum1 += e.validationScore(attempt.id1, id).?;
                    sum2 += e.validationScore(attempt.id2, id).?;
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
                    if (!e.hasValidationOverlap(id1, id2)) continue;
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

            fn validationScore(e: *const Self, candidate: usize, id: usize) ?f64 {
                const offset = if (e.gepa.val_evaluation_policy != null)
                    e.val_indices.items[candidate].get(id) orelse return null
                else
                    id;
                return e.subscores.items[candidate][offset];
            }

            fn hasValidationOverlap(e: *const Self, left: usize, right: usize) bool {
                const floor = e.gepa.merge_val_overlap_floor;
                if (e.gepa.val_evaluation_policy == null) return e.valset.len >= floor;
                if (floor == 0) return true;
                var count: usize = 0;
                for (e.val_ids.items[left]) |id| {
                    if (!e.val_indices.items[right].contains(id)) continue;
                    count += 1;
                    if (count == floor) return true;
                }
                return false;
            }

            fn commonValidationIds(e: *const Self, left: usize, right: usize) ![]const usize {
                if (e.gepa.val_evaluation_policy == null) return e.all_val_ids;
                var common: std.ArrayList(usize) = .empty;
                for (e.val_ids.items[left]) |id| {
                    if (e.val_indices.items[right].contains(id)) try common.append(e.out, id);
                }
                std.mem.sortUnstable(usize, common.items, {}, std.sort.asc(usize));
                return common.items;
            }

            /// `select_eval_subsample_for_merged_program`.
            fn mergeSubsample(
                e: *Self,
                left: usize,
                right: usize,
                common: []const usize,
            ) Allocator.Error![]const usize {
                var buckets: [3]std.ArrayList(usize) = .{ .empty, .empty, .empty };
                for (common) |id| {
                    const a = e.validationScore(left, id).?;
                    const b = e.validationScore(right, id).?;
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
                    for (common) |id| {
                        if (std.mem.indexOfScalar(usize, selected.items, id) == null) try unused.append(e.out, id);
                    }
                    if (unused.items.len >= remaining) {
                        try e.sampleInto(&selected, unused.items, remaining);
                    } else if (common.len > 0) {
                        for (0..remaining) |_| {
                            const r = e.rng.random() * @as(f64, @floatFromInt(common.len));
                            try selected.append(e.out, common[@intFromFloat(@floor(r))]);
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
        const budget_count = @as(u2, @intFromBool(self.auto != null)) +
            @as(u2, @intFromBool(self.max_full_evals != null)) +
            @as(u2, @intFromBool(self.max_metric_calls != null));
        if (budget_count == 0) return error.NoBudget;
        if (budget_count != 1) return error.ConflictingBudgets;
        assert(trainset.len > 0);
        if (self.val_evaluation_policy) |policy| {
            if (policy.max_batch_size == 0) return error.InvalidValidationBatch;
        }
        if (self.batch_sampler) |sampler| {
            if (self.reflection_minibatch_size != null or sampler.max_batch_size == 0)
                return error.InvalidBatchSamplerConfiguration;
        } else {
            assert((self.reflection_minibatch_size orelse 3) > 0);
        }
        if (self.instruction_proposer == null and self.proposal_options.max_chars == 0)
            return error.InvalidInstructionLimit;
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
        else if (self.max_full_evals) |k| blk: {
            const val_count = if (options.valset) |v| v.len else 0;
            const total = std.math.add(usize, trainset.len, val_count) catch
                return error.BudgetOverflow;
            break :blk std.math.mul(usize, k, total) catch return error.BudgetOverflow;
        } else self.max_metric_calls.?;

        const names = try out.alloc([]const u8, preds.len);
        const seed_candidate = try out.alloc([]const u8, preds.len);
        for (preds, names, seed_candidate) |p, *n, *s| {
            n.* = p.name;
            s.* = p.param.predict.instructions;
        }
        const all_val_ids = try out.alloc(usize, valset.len);
        for (all_val_ids, 0..) |*id, k| id.* = k;
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
            .all_val_ids = all_val_ids,
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
        while (!(try e.shouldStop(if (started) e.i else null, max_calls))) {
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
        const program = (try e.build(e.candidates.items[best])).program;
        const per_instance = try out.alloc([]const usize, e.fronts.len);
        for (per_instance, e.fronts) |*row, front| row.* = front.items;
        const best_outputs: ?[]const []const BestOutput = if (e.best_outputs) |outputs| rows: {
            const rows = try out.alloc([]const BestOutput, outputs.len);
            for (rows, outputs) |*row, outputs_for_row| row.* = outputs_for_row.items;
            break :rows rows;
        } else null;
        const objectives = try e.objectives.results(out);
        return .{
            .program = program.*,
            .candidates = e.candidates.items,
            .parents = e.parents.items,
            .val_aggregate_scores = try e.averages(),
            .val_ids = e.validationIds(),
            .val_subscores = e.subscores.items,
            .val_aggregate_subscores = if (e.objectives.aggregates.items.len == 0)
                null
            else
                e.objectives.aggregates.items,
            .per_objective_best_candidates = objectives.leaders,
            .objective_pareto_front = objectives.scores,
            .per_val_instance_best_candidates = per_instance,
            .best_outputs_valset = best_outputs,
            .discovery_eval_counts = e.discovery.items,
            .best_idx = best,
            .total_metric_calls = e.total_evals,
            .num_full_val_evals = e.full_evals,
            .max_metric_calls = max_calls,
            .seed = self.seed,
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

    const built: GEPA.Engine(P).Built = .{ .program = &program, .predictors = &.{} };
    for ([_]bool{ false, true }) |capture| {
        const run = try engine.runOne(engine.ctx, &built, &examples[0], capture);
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
        const run = try engine.runOne(engine.ctx, &built, &examples[0], capture);
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
    const successful = try engine.runOne(engine.ctx, &built, &examples[0], true);
    try std.testing.expectEqualStrings("ok", successful.prediction.?.get("answer").?.string);
    try std.testing.expectEqual(@as(usize, 1), successful.trace.steps.items.len);
    try std.testing.expectError(error.ParseValue, engine.reflectiveDataset(&.{program.state.instructions}, &.{successful}, &.{0}, &.{0}));

    engine.gepa.metric = FailingMetric.allocationFailure;
    for ([_]bool{ false, true }) |capture| {
        try std.testing.expectError(error.OutOfMemory, engine.runOne(engine.ctx, &built, &examples[0], capture));
    }
    try std.testing.expectEqual(@as(usize, 5), script.index);
}

test "GEPA custom selection can choose off frontier without advancing the shared RNG" {
    const Selector = struct {
        fn select(ptr: *anyopaque, state: *const GEPA.CandidateSelection) !usize {
            const index: *const usize = @ptrCast(@alignCast(ptr));
            try std.testing.expectEqualSlices(usize, &.{1}, state.front(0).?);
            try std.testing.expectEqual(@as(?[]const usize, null), state.front(1));
            return index.*;
        }
    };
    var index: usize = 0;
    var candidates = [_][]const []const u8{ &.{"seed"}, &.{"child"} };
    var leaders = [_]usize{1};
    var fronts = [_]std.ArrayList(usize){.fromOwnedSlice(&leaders)};
    var engine: GEPA.Engine(void) = .{
        .gepa = .{
            .metric = undefined,
            .candidate_selection_strategy = .{ .custom = .{
                .ptr = &index,
                .select = Selector.select,
            } },
        },
        .out = std.testing.failing_allocator,
        .ctx = undefined,
        .reflection = undefined,
        .student = undefined,
        .names = &.{"self"},
        .trainset = &.{},
        .valset = &.{},
        .rng = .init(13),
        .adapter_rng = .init(0),
        .candidates = .fromOwnedSlice(&candidates),
        .front_score = &.{},
        .fronts = &fronts,
    };
    var expected_rng = engine.rng;
    try std.testing.expectEqual(@as(usize, 0), try engine.selectParent());
    index = candidates.len;
    try std.testing.expectError(error.InvalidCandidateSelection, engine.selectParent());
    try std.testing.expectEqual(expected_rng.nextU32(), engine.rng.nextU32());
}

// Pinned GEPA selectors, including the shared RNG position after each draw.
test "GEPA candidate selection preserves exploration, filtered frontiers, and RNG state" {
    const cases = [_]struct {
        strategy: GEPA.Selection,
        seed: u32,
        scores: []const f64,
        fronts: []const []const usize,
        draws: [12]usize,
        rng_after: [12]u32,
    }{
        .{
            .strategy = .epsilon_greedy,
            .seed = 31,
            .scores = &.{ 0.2, 0.8, 0.8 },
            .fronts = &.{ &.{1}, &.{2} },
            .draws = .{ 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 },
            .rng_after = .{ 3281255694, 604968615, 185835555, 2298718664, 3055913604, 599541520, 4216431881, 142230572, 586600295, 2300649693, 1923075604, 1771975039 },
        },
        .{
            .strategy = .epsilon_greedy,
            .seed = 0,
            .scores = &.{0.5},
            .fronts = &.{&.{0}},
            .draws = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            .rng_after = .{ 3255389356, 1806341205, 1112038970, 2195908194, 1739178872, 3366389305, 1302718217, 2046968324, 2505606783, 3900315155, 2167613558, 1210484339 },
        },
        .{
            .strategy = .top_k_pareto,
            .seed = 7,
            .scores = &.{ 0.2, 0.8, 0.8, 0.6, 0.5, 0.6, 0.1 },
            .fronts = &.{ &.{ 0, 1, 2 }, &.{ 1, 6 }, &.{ 3, 5 }, &.{4}, &.{6} },
            .draws = .{ 5, 1, 4, 1, 1, 1, 5, 1, 1, 1, 1, 4 },
            .rng_after = .{ 4071050724, 1695753998, 2795742288, 311111475, 3527346212, 1570621944, 2503055453, 3907149204, 161042648, 369140570, 1862494042, 1796035739 },
        },
        .{
            .strategy = .top_k_pareto,
            .seed = 15,
            .scores = &.{ 0.2, 0.8, 0.8, 0.6, 0.5, 0.6, 0.1 },
            .fronts = &.{&.{ 0, 6 }},
            .draws = .{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            .rng_after = .{ 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452, 4145683452 },
        },
    };
    for (cases) |case| {
        var storage = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer storage.deinit();
        const arena = storage.allocator();
        const candidates = try arena.alloc([]const []const u8, case.scores.len);
        @memset(candidates, &.{});
        const scores = try arena.alloc([]const f64, case.scores.len);
        for (scores, 0..) |*row, idx| row.* = case.scores[idx..][0..1];
        const fronts = try arena.alloc(std.ArrayList(usize), case.fronts.len);
        for (fronts, case.fronts) |*front, ids| {
            front.* = .empty;
            try front.appendSlice(arena, ids);
        }
        var engine: GEPA.Engine(void) = .{
            .gepa = .{ .metric = undefined, .candidate_selection_strategy = case.strategy },
            .out = arena,
            .ctx = undefined,
            .reflection = undefined,
            .student = undefined,
            .names = &.{},
            .trainset = &.{},
            .valset = &.{},
            .rng = .init(case.seed),
            .adapter_rng = .init(0),
            .candidates = .fromOwnedSlice(candidates),
            .subscores = .fromOwnedSlice(scores),
            .front_score = &.{},
            .fronts = fronts,
        };
        for (case.draws, case.rng_after) |draw, rng_after| {
            try std.testing.expectEqual(draw, try engine.selectParent());
            var snapshot = engine.rng;
            try std.testing.expectEqual(rng_after, snapshot.nextU32());
        }
    }
}

// Frozen gepa 0.1.4 state and selectors, seed 17. Sparse keys arrive in later candidates.
test "GEPA objective frontiers select different parents without changing the RNG clock" {
    const rows: [4][3][]const ObjectiveScore = .{
        .{ &.{ .{ .name = "q", .score = 3 }, .{ .name = "cost", .score = 0 } }, &.{.{ .name = "q", .score = 0 }}, &.{.{ .name = "ok", .score = 1 }} },
        .{ &.{ .{ .name = "q", .score = 1 }, .{ .name = "cost", .score = 1 } }, &.{ .{ .name = "q", .score = 4 }, .{ .name = "cost", .score = 2 } }, &.{.{ .name = "ok", .score = 1 }} },
        .{ &.{ .{ .name = "q", .score = 0 }, .{ .name = "cost", .score = 2 } }, &.{.{ .name = "q", .score = 0 }}, &.{.{ .name = "ok", .score = 1 }} },
        .{ &.{ .{ .name = "q", .score = 2 }, .{ .name = "cost", .score = 3 } }, &.{ .{ .name = "q", .score = 2 }, .{ .name = "cost", .score = -1 } }, &.{.{ .name = "ok", .score = 2 }} },
    };
    const scores: []const []const f64 = &.{ &.{ 1, 0, 0.5 }, &.{ 0, 1, 0.5 }, &.{ 0.8, 0.8, 0.8 }, &.{ 0, 0, 1 } };
    const cases = [_]struct {
        mode: GEPA.Frontier,
        leaders: []const usize,
        draws: [16]usize,
        next: [16]u32,
    }{
        .{ .mode = .instance, .leaders = &.{ 0, 1, 3 }, .draws = .{ 3, 1, 1, 1, 1, 0, 3, 3, 3, 3, 1, 0, 0, 0, 1, 3 }, .next = .{ 1778837931, 3464710496, 4125293655, 1243931546, 750280511, 3290406397, 3024601294, 2320078575, 2840618605, 1195188523, 473142372, 3940602561, 1069030695, 1650002322, 3501197581, 1802024078 } },
        .{ .mode = .objective, .leaders = &.{ 1, 2, 3 }, .draws = .{ 3, 2, 2, 2, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 2, 3 }, .next = .{ 1778837931, 3464710496, 4125293655, 1243931546, 750280511, 3290406397, 3024601294, 2320078575, 2840618605, 1195188523, 473142372, 3940602561, 1069030695, 1650002322, 3501197581, 1802024078 } },
        .{ .mode = .hybrid, .leaders = &.{ 0, 1, 3, 1, 2, 3 }, .draws = .{ 3, 3, 1, 1, 1, 1, 2, 2, 3, 2, 1, 0, 0, 1, 3, 2 }, .next = .{ 1778837931, 3464710496, 4125293655, 1243931546, 750280511, 3290406397, 3024601294, 2320078575, 2840618605, 1195188523, 473142372, 3940602561, 1069030695, 1650002322, 3501197581, 1802024078 } },
        .{ .mode = .cartesian, .leaders = &.{ 0, 3, 1, 3, 1 }, .draws = .{ 1, 1, 3, 3, 3, 3, 1, 3, 0, 0, 3, 1, 1, 3, 1, 3 }, .next = .{ 1778837931, 3464710496, 4125293655, 1243931546, 750280511, 3290406397, 2840618605, 473142372, 3940602561, 1069030695, 1650002322, 3501197581, 1083980018, 3692404330, 3446169948, 2751033102 } },
    };
    var storage = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    for (cases) |case| {
        var fronts: [3]std.ArrayList(usize) = .{ .fromOwnedSlice(try arena.dupe(usize, &.{0})), .fromOwnedSlice(try arena.dupe(usize, &.{1})), .fromOwnedSlice(try arena.dupe(usize, &.{3})) };
        var engine: GEPA.Engine(void) = .{
            .gepa = .{ .metric = undefined, .frontier_type = case.mode },
            .out = arena,
            .ctx = undefined,
            .reflection = undefined,
            .student = undefined,
            .names = &.{},
            .trainset = &.{},
            .valset = &.{},
            .rng = .init(17),
            .adapter_rng = .init(0),
            .front_score = &.{},
            .fronts = &fronts,
        };
        try engine.candidates.appendNTimes(arena, &.{}, rows.len);
        try engine.subscores.appendSlice(arena, scores);
        for (rows, 0..) |entries, i| {
            var runs: [3]GEPA.Run = undefined;
            for (&runs, entries, scores[i]) |*run, row, score| run.* = .{ .prediction = null, .trace = undefined, .score = score, .objective_scores = row };
            try engine.objectives.update(arena, case.mode, i, &runs, null);
        }
        try std.testing.expectEqual(case.leaders.len, engine.activeFrontCount());
        for (case.leaders, 0..) |leader, i| try std.testing.expectEqualSlices(usize, &.{leader}, engine.activeFront(i));
        for ([_]GEPA.Selection{ .pareto, .top_k_pareto }) |selection| {
            engine.gepa.candidate_selection_strategy = selection;
            engine.rng = .init(17);
            for (case.draws, case.next) |draw, word| {
                try std.testing.expectEqual(draw, try engine.selectParent());
                var snapshot = engine.rng;
                try std.testing.expectEqual(word, snapshot.nextU32());
            }
        }
    }
}

test "GEPA objective batches require scores only for non-instance frontiers" {
    var storage = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const empty: GEPA.Run = .{ .prediction = null, .trace = undefined, .score = 1 };
    var state: GEPA.Objectives = .{};
    try std.testing.expectError(error.MissingObjectiveScores, state.update(arena, .hybrid, 0, &.{empty}, null));
    try state.update(arena, .instance, 0, &.{empty}, null);
    var scored = empty;
    scored.objective_scores = &.{.{ .name = "quality", .score = 0.5 }};
    try state.update(arena, .instance, 1, &.{ scored, empty }, null);
    try state.update(arena, .instance, 2, &.{scored}, null);
    try state.update(arena, .instance, 3, &.{empty}, null);
    const result = try state.results(arena);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, result.leaders.?[0].candidates);
    try std.testing.expectEqual(@as(f64, 0.5), state.aggregates.items[1][0].score);
    try std.testing.expectEqual(@as(usize, 0), state.aggregates.items[0].len);
    try std.testing.expectEqual(@as(usize, 0), state.aggregates.items[3].len);
    scored.objective_scores = &.{ scored.objective_scores[0], scored.objective_scores[0] };
    try std.testing.expectError(error.DuplicateObjective, state.update(arena, .instance, 4, &.{scored}, null));
}

test "GEPA requires one budget and evaluates the seed even with a zero budget" {
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
    const Program = predict.Predict(Sig);
    const ctx: Context = .{ .arena = arena, .lm = script.lm() };
    const options: GEPA.Options = .{ .valset = &.{} };
    const invalid: []const GEPA = &.{
        .{ .metric = Metric.score, .auto = .light, .max_full_evals = 0 },
        .{ .metric = Metric.score, .auto = .light, .max_metric_calls = 0 },
        .{ .metric = Metric.score, .max_full_evals = 0, .max_metric_calls = 0 },
        .{ .metric = Metric.score, .auto = .light, .max_full_evals = 0, .max_metric_calls = 0 },
    };
    for (invalid) |optimizer| {
        try std.testing.expectError(
            error.ConflictingBudgets,
            optimizer.compile(Program, arena, ctx, .{}, &examples, options),
        );
    }
    const missing: GEPA = .{ .metric = Metric.score };
    try std.testing.expectError(
        error.NoBudget,
        missing.compile(Program, arena, ctx, .{}, &examples, options),
    );
    const overflowing: GEPA = .{ .metric = Metric.score, .max_full_evals = std.math.maxInt(usize) };
    try std.testing.expectError(
        error.BudgetOverflow,
        overflowing.compile(Program, arena, ctx, .{}, &examples, options),
    );
    const optimizer: GEPA = .{ .metric = Metric.score, .max_metric_calls = 0 };
    const result = try optimizer.compile(Program, arena, ctx, .{}, &examples, options);
    try std.testing.expect(result.best_outputs_valset == null);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0 }, result.val_subscores[0]);
    try std.testing.expectEqual(@as(f64, 0.5), result.val_aggregate_scores[0]);
    try std.testing.expectEqual(@as(usize, 2), result.total_metric_calls);
    try std.testing.expectEqual(@as(usize, 0), result.max_metric_calls);
}

test "GEPA stops after seed evaluation before reflecting or calling later stoppers" {
    const Sig = signature.Signature(struct {
        pub const instructions = "Original.";
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Callbacks = struct {
        fn metric(
            _: Allocator,
            _: *const Example,
            _: *const optimize.RecordPrediction,
            _: ?*const Trace,
            _: ?[]const u8,
            _: ?*const Trace.Step,
        ) value.ParseError!Feedback {
            return .{ .score = 0 };
        }

        fn afterSeed(_: *anyopaque, state: *const GEPA.StopState) !bool {
            try std.testing.expectEqual(@as(?usize, null), state.last_iteration);
            try std.testing.expectEqual(@as(usize, 1), state.total_metric_calls);
            return true;
        }

        fn unexpected(_: *anyopaque, _: *const GEPA.StopState) !bool {
            return error.CallbackShouldHaveShortCircuited;
        }
    };
    var storage: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    var script: @import("lm.zig").ScriptedLM = .{
        .responses = &.{ "{\"answer\":\"seed\"}", "{\"answer\":\"would reflect\"}" },
    };
    const example = try Example.fromJson(arena, "{\"question\":\"q\"}", &.{"question"});
    const optimizer: GEPA = .{
        .metric = Callbacks.metric,
        .max_metric_calls = 8,
        .stop_callbacks = &.{
            .{ .ptr = &script, .should_stop = Callbacks.afterSeed },
            .{ .ptr = &script, .should_stop = Callbacks.unexpected },
        },
    };
    const result = try optimizer.compile(predict.Predict(Sig), arena, .{
        .arena = arena,
        .lm = script.lm(),
        .adapter = .json,
    }, .{}, &.{example}, .{});
    try std.testing.expectEqual(@as(usize, 1), script.index);
    try std.testing.expectEqual(@as(usize, 1), result.total_metric_calls);
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
        .all_val_ids = &.{ 0, 1 },
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

test "GEPA custom dataset separates History and stringifies ordinary values" {
    var storage = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {
            question: bool,
            prior: History,
            thought: signature.Reasoning,
        };
        pub const Output = struct { answer: bool };
    });
    const predictor: predict.Predict(Sig) = .{};
    var event: Record = .empty;
    try event.put(arena, "answer", .{ .bool = false });
    const history: History = .{ .messages = &.{event} };
    var trace: Trace = .{ .arena = arena };
    var inputs: Record = .empty;
    try inputs.put(arena, "question", .{ .bool = true });
    try inputs.put(arena, "prior", try history.toValue(arena));
    try inputs.put(arena, "thought", .{ .string = "plan" });
    var outputs: Record = .empty;
    try outputs.put(arena, "answer", .{ .bool = false });
    try trace.record(&predictor.state, &inputs, &outputs);

    const step = &trace.steps.items[0];
    const projected = try customInputs(arena, step);
    try std.testing.expectEqualStrings("True", projected.get("question").?.string);
    try std.testing.expectEqualStrings("plan", projected.get("thought").?.string);
    try std.testing.expectEqualStrings(
        "```json\n  0: {'answer': False}\n```",
        projected.get("Context").?.string,
    );
    try std.testing.expect(projected.get("prior") == null);
    const generated = try customOutputs(arena, step);
    try std.testing.expectEqualStrings("False", generated.get("answer").?.string);
}

fn invokeProposer(
    proposer: GEPA.InstructionProposer,
    context: Context,
    input: *const GEPA.InstructionProposal,
) Allocator.Error!?[]const GEPA.Instruction {
    return proposer.propose(proposer.ptr, context, input) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

fn compactToolObservations(arena: Allocator, item: Value) value.ParseError!?Value {
    const ToolCalls = @import("tool.zig").ToolCalls;
    var calls = try ToolCalls.fromValue(arena, item);
    const results = calls.tool_call_results orelse return null;
    var replacements: ?[]@import("tool.zig").ToolCallResult = null;
    for (results.tool_call_results, 0..) |observation, index| {
        const text = try reflectionString(arena, null, observation.value);
        const limit = 500;
        if (text.len <= limit) continue;
        const view = std.unicode.Utf8View.init(text) catch return error.ParseValue;
        var iterator = view.iterator();
        for (0..limit) |_| if (iterator.nextCodepointSlice() == null) break;
        const prefix_len = iterator.i;
        if (prefix_len == text.len) continue;
        const cut = std.unicode.utf8CountCodepoints(text[prefix_len..]) catch unreachable;
        var cut_buffer: [32]u8 = undefined;
        var total_buffer: [32]u8 = undefined;
        if (replacements == null) replacements = try arena.dupe(
            @import("tool.zig").ToolCallResult,
            results.tool_call_results,
        );
        replacements.?[index].value = .{ .string = try std.fmt.allocPrint(
            arena,
            "{s} [{s} of {s} characters cut]",
            .{ text[0..prefix_len], groupedDecimal(&cut_buffer, cut), groupedDecimal(&total_buffer, cut + limit) },
        ) };
    }
    calls.tool_call_results = .{ .tool_call_results = replacements orelse return null };
    return try calls.toValue(arena);
}

fn groupedDecimal(buffer: *[32]u8, number: usize) []const u8 {
    var remaining = number;
    var pos: usize = buffer.len;
    var digits: usize = 0;
    while (true) {
        pos -= 1;
        buffer[pos] = '0' + @as(u8, @intCast(remaining % 10));
        remaining /= 10;
        if (remaining == 0) return buffer[pos..];
        digits += 1;
        if (digits % 3 == 0) {
            pos -= 1;
            buffer[pos] = ',';
        }
    }
}

test "GEPA reflects truncated Unicode tool results without changing history" {
    var storage: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const text = try arena.alloc(u8, 2 * 1501);
    for (0..1501) |i| @memcpy(text[2 * i ..][0..2], "é");
    const ToolCalls = @import("tool.zig").ToolCalls;
    const calls: ToolCalls = .{
        .tool_calls = &.{.{ .id = "one", .name = "lookup", .args = .empty }},
        .tool_call_results = .{ .tool_call_results = &.{
            .{ .call_id = "one", .name = "lookup", .value = .{ .string = text }, .is_error = true },
            .{ .name = "exact", .value = .{ .string = text[0..1000] } },
        } },
    };
    var event: Record = .empty;
    try event.put(arena, "actions", try calls.toValue(arena));
    try event.put(arena, "notes", .{ .string = text });
    const history: History = .{ .messages = &.{event} };
    const options: ConfiguredInstructionProposer = .{ .truncate_history_outputs = true };
    const prepared = try History.fromValue(arena, try options.prepareHistory(arena, history));
    const actual = try ToolCalls.fromValue(arena, prepared.messages[0].get("actions").?);
    const original = try ToolCalls.fromValue(arena, history.messages[0].get("actions").?);
    const observations = actual.tool_call_results.?.tool_call_results;
    const compact = observations[0].value.string;
    try std.testing.expectEqualStrings(text[0..1000], compact[0..1000]);
    try std.testing.expectEqualStrings(" [1,001 of 1,501 characters cut]", compact[1000..]);
    try std.testing.expect(observations[0].is_error);
    try std.testing.expectEqualStrings(text[0..1000], observations[1].value.string);
    try std.testing.expectEqualStrings(text, prepared.messages[0].get("notes").?.string);
    try std.testing.expectEqualStrings(text, original.tool_call_results.?.tool_call_results[0].value.string);
}

fn isHistory(field: signature.Field) bool {
    return field.dspy_type and std.mem.eql(u8, field.type_name, "History");
}

fn prepareHistoryValue(
    arena: Allocator,
    options: ConfiguredInstructionProposer,
    item: Value,
) value.ParseError!Value {
    if (!options.truncate_history_outputs) return item;
    const history = History.fromValue(arena, item) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.ParseValue,
    };
    return options.prepareHistory(arena, history);
}

fn builtinInputs(
    arena: Allocator,
    options: ConfiguredInstructionProposer,
    step: *const Trace.Step,
) value.ParseError!Record {
    const fields = step.predictor.fields[0..step.predictor.input_count];
    var history_key: ?[]const u8 = null;
    var inputs: Record = .empty;
    for (fields) |field| {
        if (!isHistory(field)) continue;
        const input = step.inputs.get(field.name) orelse continue;
        if (input == .null) continue;
        if (history_key != null) return error.ParseValue;
        history_key = field.name;
        try inputs.put(arena, "Context", try prepareHistoryValue(arena, options, input));
    }
    const key = history_key orelse return step.inputs;
    for (step.inputs.keys(), step.inputs.values()) |name, input| {
        if (std.mem.eql(u8, key, name)) continue;
        try inputs.put(arena, name, input);
    }
    return inputs;
}

fn builtinOutputs(
    arena: Allocator,
    options: ConfiguredInstructionProposer,
    step: *const Trace.Step,
) value.ParseError!Record {
    if (!options.truncate_history_outputs) return step.outputs;
    var outputs = step.outputs;
    var copied = false;
    for (step.predictor.fields[step.predictor.input_count..]) |field| {
        if (!isHistory(field)) continue;
        const output = step.outputs.get(field.name) orelse continue;
        if (output == .null) continue;
        if (!copied) outputs = try step.outputs.clone(arena);
        copied = true;
        try outputs.put(arena, field.name, try prepareHistoryValue(arena, options, output));
    }
    return outputs;
}

fn customInputs(arena: Allocator, step: *const Trace.Step) value.ParseError!Record {
    const fields = step.predictor.fields[0..step.predictor.input_count];
    var history_key: ?[]const u8 = null;
    var inputs: Record = .empty;
    for (fields) |field| {
        if (!field.dspy_type or !std.mem.eql(u8, field.type_name, "History")) continue;
        const input = step.inputs.get(field.name) orelse continue;
        if (input == .null) continue;
        if (history_key != null) return error.ParseValue;
        history_key = field.name;
        try inputs.put(arena, "Context", .{ .string = try historyText(arena, input) });
    }
    for (step.inputs.keys(), step.inputs.values()) |key, input| {
        if (history_key) |name| {
            if (std.mem.eql(u8, name, key)) continue;
        }
        const field = findField(fields, key);
        const projected = if (field != null and field.?.dspy_type and input != .null)
            input
        else
            Value{ .string = try reflectionString(arena, field, input) };
        try inputs.put(arena, key, projected);
    }
    return inputs;
}

fn customOutputs(arena: Allocator, step: *const Trace.Step) value.ParseError!Record {
    const fields = step.predictor.fields[step.predictor.input_count..];
    var outputs: Record = .empty;
    for (step.outputs.keys(), step.outputs.values()) |key, output| {
        const text = try reflectionString(arena, findField(fields, key), output);
        try outputs.put(arena, key, .{ .string = text });
    }
    return outputs;
}

fn findField(fields: []const signature.Field, name: []const u8) ?signature.Field {
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, name)) return field;
    }
    return null;
}

fn reflectionString(
    arena: Allocator,
    field: ?signature.Field,
    item: Value,
) value.ParseError![]const u8 {
    if (field) |declared| {
        if (declared.python_str) |render| return render(arena, item) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        };
    }
    if (item == .string) return item.string;
    var writer: Writer.Allocating = .init(arena);
    defer writer.deinit();
    pyjson.writeStr(&writer.writer, item) catch |err| return reflectionWriteError(err);
    return writer.toOwnedSlice();
}

fn historyText(arena: Allocator, item: Value) value.ParseError![]const u8 {
    const history = History.fromValue(arena, item) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.ParseValue,
    };
    var writer: Writer.Allocating = .init(arena);
    defer writer.deinit();
    const w = &writer.writer;
    w.writeAll("```json\n") catch return error.OutOfMemory;
    for (history.messages, 0..) |message, index| {
        w.print("  {d}: ", .{index}) catch return error.OutOfMemory;
        pyjson.writeRepr(w, .{ .object = message }, 0) catch |err| {
            return reflectionWriteError(err);
        };
        w.writeByte('\n') catch return error.OutOfMemory;
    }
    w.writeAll("```") catch return error.OutOfMemory;
    return writer.toOwnedSlice();
}

fn reflectionWriteError(err: pyjson.Error) value.ParseError {
    return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        else => error.ParseValue,
    };
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
