//! Modules: `Predict`, `ChainOfThought`, `Retrieve`, the explicit execution
//! `Context`, the call `Trace`, and DSPy-named parameter discovery.
//!
//! A program is a plain struct with a `compiled: bool` field, `Input` and
//! `Output` decls, and `pub fn forward(self: *const P, ctx: Context, input:
//! P.Input) !P.Output`. Nothing is global: allocator, LM, adapter, retriever
//! and trace are all passed in `Context`.
//!
//! Upstream (DSPy 3.5.0): dspy/dsp/utils/settings.py, dspy/predict/chain_of_thought.py, dspy/predict/predict.py, dspy/primitives/base_module.py, dspy/primitives/example.py, dspy/primitives/module.py, dspy/primitives/prediction.py, dspy/retrievers/retrieve.py

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const adapter_mod = @import("adapter.zig");
const lm_mod = @import("lm.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const Workspace = @import("workspace.zig").Workspace;
const tool_mod = @import("tool.zig");
const inference = @import("inference.zig");
const usage_mod = @import("usage_tracker.zig");
const prediction_mod = @import("prediction.zig");
const Graph = usage_mod.Graph;
pub const Prediction = prediction_mod.Prediction;
const Adapter = adapter_mod.Adapter;
const stream = @import("predict_stream.zig");
const status = @import("stream_status.zig");
const Field = signature.Field;
const Record = value.Record;

pub const Error = adapter_mod.Error || lm_mod.Error || value.ConvertError ||
    @import("decision_state.zig").Error || stream.Error || error{
    NoRetriever,
    InvalidCompletionCount,
    MissingToolInput,
    InconsistentCompletionLogprobs,
    ConflictingCompletionLogprobs,
};

/// The final failed completion parse of one predictor call, as DSPy's
/// AdapterParseError exposes it. Contents borrow the call's arena.
pub const ParseFailure = struct {
    /// Prepared signature identity; non-null exactly when a failure is captured.
    signature_class: ?*const anyopaque = null,
    /// Output field count of the failed prepared signature.
    expected_count: u32 = 0,
    failure: adapter_mod.ParseFailure = .{},

    var derived_signature_token: u8 = 0;

    /// DSPy's decision evidence translation hands the chat adapter a signature
    /// class derived per call, so its parse errors name a class that no
    /// predictor owns. This identity equals no `PredictState.signature_class`.
    pub const derived_signature: *const anyopaque = &derived_signature_token;

    /// Predictor calls empty the capture themselves. Call this after handling
    /// an error when a later `error.AdapterParse` may come from code that
    /// makes no predictor call; the error value alone cannot tell them apart.
    pub fn reset(capture: *ParseFailure) void {
        capture.* = .{};
    }
};

/// A partial configuration of native generation options. For nullable
/// options the outer null means inherit; a present inner null clears the value.
pub const PredictConfig = struct {
    temperature: ??f64 = null,
    /// Nucleus sampling in [0, 1]; an explicit inner null clears it.
    top_p: ??f64 = null,
    /// Positive integer sampling cutoff; an explicit Value.null clears it.
    top_k: ?@import("runtime_value.zig").Value = null,
    /// Repetition penalties in [-2, 2]; an explicit inner null clears either.
    frequency_penalty: ??f64 = null,
    presence_penalty: ??f64 = null,
    /// Best-effort integer sampling seed; an explicit Value.null clears it.
    seed: ?@import("runtime_value.zig").Value = null,
    rollout_id: ??@import("runtime_value.zig").Value = null,
    response_format: ??lm_mod.ResponseFormat = null,
    tool_choice: ??lm_mod.ToolChoice = null,
    parallel_tool_calls: ??bool = null,
    logprobs: ?@import("runtime_value.zig").Value = null,
    n: ?u32 = null,
    reasoning_effort: ??[]const u8 = null,
    prediction: ??@import("runtime_value.zig").Value = null,
    /// False bypasses an installed response cache. An explicit inner null
    /// clears an inherited override and uses the LM's default.
    cache: ??bool = null,
    /// Provider-side response storage; an explicit inner null clears it.
    store: ??bool = null,
    /// Opaque end-user attribution; an explicit inner null clears it.
    user_id: ??[]const u8 = null,
    /// Provider-owned service tier; an explicit inner null clears it.
    service_tier: ??[]const u8 = null,
    /// A present inner null removes the provider's configured output cap.
    max_tokens: ??u32 = null,
    /// Null inherits the stop list; an empty slice clears it.
    stop: ?[]const []const u8 = null,

    fn apply(config: *const PredictConfig, options: *lm_mod.Options) void {
        inline for (@typeInfo(PredictConfig).@"struct".field_names) |name| {
            const override = @field(config.*, name);
            if (comptime std.mem.eql(u8, name, "max_tokens")) {
                if (override != null) options.max_tokens = override;
            } else if (override) |setting| {
                @field(options.*, name) = setting;
            }
        }
        if (config.reasoning_effort != null) options.reasoning_effort_present = true;
    }
};

pub const Context = struct {
    /// Per-call scratch: prompts, completions, parsed outputs.
    arena: Allocator,
    /// Caller-reserved fixed scratch for internal per-example and per-step
    /// scopes (TigerStyle static allocation). Owned by one thread at a time.
    workspace: ?*Workspace = null,
    /// Settings-level fallback; bound predictors can run without one.
    lm: lm_mod.LM = lm_mod.LM.unconfigured,
    /// Privileged per-call override. Explicit null selects the settings LM.
    lm_override: ?lm_mod.LM = null,
    lm_override_present: bool = false,
    /// Borrowed for the call: null uses stored demonstrations; empty clears them.
    demos_override: ?[]const Record = null,
    adapter: Adapter = .xml,
    /// Borrowed extraction configuration when adapter is .two_step.
    two_step: ?*const TwoStepAdapter = null,
    /// When set, successful predictor calls can append deep-copied steps.
    trace: ?*Trace = null,
    /// Nonpositive disables predictor recording. When at or above a positive
    /// limit, a predictor append evicts exactly one oldest step.
    max_trace_size: i64 = 10_000,
    retriever: ?Retriever = null,
    /// Original sampling options when an optimizer refreshes the settings LM.
    /// An explicitly selected LM does not inherit that refresh.
    before_settings_rollout: ?struct {
        temperature: ?f64,
        rollout_id: ?@import("runtime_value.zig").Value,
    } = null,
    /// Sampling defaults of a retry's copied LM, scoped to its predictors.
    predict_rollout: ?*const PredictRollout = null,
    lm_options: lm_mod.Options = .{},
    /// Borrowed overrides for a predictor invocation, after its stored config.
    predict_config: ?*const PredictConfig = null,
    /// Synchronous raw events from physical adapter generations. Cache hits
    /// and non-streaming engines emit none. Concurrent calls need disjoint or
    /// synchronized sinks. A stop aborts the active prediction.
    stream_sink: ?lm_mod.OpenAI.StreamSink = null,
    /// Scoped by callStream; callers normally leave this null.
    stream_session: ?*stream.Session = null,
    /// Synchronous status observer for module, adapter LM, and tool calls.
    status_sink: ?status.Sink = null,
    /// ChatAdapter and XMLAdapter retry once with JSONAdapter on parse failure.
    json_fallback: bool = true,
    /// DSPy settings-level error budget, inherited by BootstrapFewShot.
    max_errors: usize = 10,
    /// Implicit XML and JSON enable native History; explicit Chat opts in.
    use_native_function_calling: ?bool = null,
    /// Scoped adapter wrappers prepare a signature and kwargs before formatting.
    preparation: ?Preparation = null,
    /// Independent scoped evidence collectors share this handle with workers.
    decision_evidence: ?*DecisionEvidence = null,
    /// Opt-in parse diagnostics. Each predictor call and adapter attempt
    /// empties it first, so only a final `error.AdapterParse` from parsing a
    /// completion leaves it set. One capture per concurrently running context.
    parse_failure: ?*ParseFailure = null,
    /// Effective execution-local tracker; workers share its synchronized
    /// collector, never a mutable settings slot.
    usage_tracker: ?*usage_mod.UsageTracker = null,
    usage_failure: ?*lm_mod.AccountingFailure = null,
    /// Enables automatic Module-call scopes with caller-provisioned resources.
    track_usage: ?UsageTracking = null,

    /// The adapter policy before negotiating the selected LM's capabilities.
    pub fn nativeFunctionCallingEnabled(ctx: Context) bool {
        return ctx.use_native_function_calling orelse (ctx.adapter == .xml or ctx.adapter == .json);
    }
};

pub const UsageTracking = struct {
    storage: *usage_mod.Storage,
    limits: usage_mod.UsageTracker.Limits,
};

/// Source record_evidence entries are captured before decision decoding.
/// The collector and its allocator must outlive every context using it.
pub const DecisionEvidence = struct {
    arena: Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    items: std.ArrayList(Entry) = .empty,

    pub const Entry = struct {
        predictor: *const PredictState,
        output: []const u8,
        evidence: @import("runtime_value.zig").Value,
    };

    fn record(self: *DecisionEvidence, predictor: *const PredictState, answers: *const Record) Allocator.Error!void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (answers.keys(), answers.values()) |name, answer| {
            try self.items.append(self.arena, .{
                .predictor = predictor,
                .output = try self.arena.dupe(u8, name),
                .evidence = try value.cloneValue(self.arena, answer),
            });
        }
    }
};

/// DSPy 3.5's two-stage adapter (deprecated upstream, but still supported).
/// LM handles are borrowed; both models must outlive every execution context.
/// Extraction has its own LM defaults, never the main predictor's options.
pub const TwoStepAdapter = struct {
    extraction_model: lm_mod.LM,
    extraction_options: lm_mod.Options = .{},

    pub fn context(self: *const TwoStepAdapter, base: Context) Context {
        var ctx = base;
        ctx.adapter = .two_step;
        ctx.two_step = self;
        return ctx;
    }

    pub fn parse(self: *const TwoStepAdapter, base: Context, original: adapter_mod.View, completion: []const u8) Error!Record {
        var instructions: std.Io.Writer.Allocating = .init(base.arena);
        const w = &instructions.writer;
        w.writeAll("The input is a text that should contain all the necessary information to produce the fields ") catch return error.OutOfMemory;
        for (original.outputs, 0..) |field, index| {
            if (index > 0) w.writeAll(", ") catch return error.OutOfMemory;
            w.print("`{s}`", .{field.name}) catch return error.OutOfMemory;
        }
        w.writeAll(".             Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field.") catch return error.OutOfMemory;
        var extractor: adapter_mod.View = .{
            .instructions = instructions.written(),
            .inputs = &.{.{ .name = "text", .kind = .str, .type_name = "str", .desc = "${text}", .prefix = "Text:" }},
            .outputs = original.outputs,
        };
        // Python's {'text': InputField(), **outputs} preserves the first key's
        // position when an output named text overwrites the extraction input.
        if (original.findOutput("text")) |text_output| {
            extractor.inputs = &.{};
            if (!std.mem.eql(u8, original.outputs[0].name, "text")) {
                const fields = try base.arena.alloc(Field, original.outputs.len);
                fields[0] = text_output;
                var index: usize = 1;
                for (original.outputs) |field| {
                    if (std.mem.eql(u8, field.name, "text")) continue;
                    fields[index] = field;
                    index += 1;
                }
                extractor.outputs = fields;
            }
        }
        var inputs: Record = .empty;
        try inputs.put(base.arena, "text", .{ .string = completion });
        const ctx: Context = .{
            .arena = base.arena,
            .lm = self.extraction_model,
            .adapter = .chat,
            .lm_options = self.extraction_options,
            .usage_tracker = base.usage_tracker,
            .usage_failure = base.usage_failure,
            .track_usage = base.track_usage,
        };
        // Extraction failures are reported against the outer signature and the
        // main completion by the caller, so this context captures nothing.
        assert(ctx.parse_failure == null);
        const parsed = runAdapterMany(ctx, null, null, .chat, extractor, &.{}, &inputs, null) catch |err| switch (err) {
            error.AdapterParse => try runAdapterMany(ctx, null, null, .json, extractor, &.{}, &inputs, null),
            else => return err,
        };
        if (parsed.records.len == 0) return error.AdapterParse;
        return parsed.records[0];
    }
};

/// Adapter-call state before native field adaptation and JSON fallback.
pub const PreparedCall = struct {
    signature_class: *const anyopaque,
    view: adapter_mod.View,
    inputs: *const Record,
};

/// A borrowed, synchronous adapter wrapper. Its storage must outlive the call.
/// Wrappers explicitly invoke their captured parent to preserve nesting order.
pub const Preparation = struct {
    ptr: *const anyopaque,
    prepare_fn: *const fn (*const anyopaque, Allocator, PreparedCall) Error!PreparedCall,

    pub fn apply(self: Preparation, arena: Allocator, call: PreparedCall) Error!PreparedCall {
        return self.prepare_fn(self.ptr, arena, call);
    }
};

const CompletionLogprobs = @import("prediction.zig").CompletionLogprobs;

pub const Trace = struct {
    arena: Allocator,
    steps: std.ArrayList(Step) = .empty,
    /// Front removals by predictor recording, modulo usize. Consumers that
    /// track positions can distinguish an append from an overwrite at capacity.
    evicted_steps: usize = 0,

    pub const Step = struct {
        predictor: *const PredictState,
        inputs: Record,
        outputs: Record,
        /// Provider metadata, deliberately outside the parsed output fields.
        token_scores: ?[]const ?CompletionLogprobs = null,

        /// Null means no score or coverage information, or out of range.
        pub fn completionLogprobs(step: Step, index: usize) ?CompletionLogprobs {
            return CompletionLogprobs.at(step.token_scores, index);
        }

        /// Copy an already formatted trace step without reformatting tool catalogs.
        pub fn clone(step: Step, arena: Allocator) Allocator.Error!Step {
            return .{
                .predictor = step.predictor,
                .inputs = try value.cloneRecord(arena, &step.inputs),
                .outputs = try value.cloneRecord(arena, &step.outputs),
                .token_scores = try CompletionLogprobs.cloneRows(arena, step.token_scores),
            };
        }
    };

    pub fn record(t: *Trace, predictor: *const PredictState, inputs: *const Record, outputs: *const Record) adapter_mod.Error!void {
        try t.recordLimited(predictor, inputs, outputs, &.{}, std.math.maxInt(u64), false);
    }

    fn recordLimited(t: *Trace, predictor: *const PredictState, inputs: *const Record, outputs: *const Record, responses: []const lm_mod.Completion, limit: u64, omit_trace_control: bool) adapter_mod.Error!void {
        if (limit == 0) return;
        var serialized_inputs = if (omit_trace_control)
            try cloneTraceInputs(t.arena, inputs)
        else
            try value.cloneRecord(t.arena, inputs);
        for (predictor.fields[0..predictor.input_count]) |f| {
            if (!f.native_tools or (omit_trace_control and std.mem.eql(u8, f.name, "_trace"))) continue;
            const catalog = inputs.get(f.name) orelse continue;
            try serialized_inputs.put(t.arena, f.name, try tool_mod.formatCatalogValue(t.arena, catalog));
        }
        const step: Step = .{
            .predictor = predictor,
            .inputs = serialized_inputs,
            .outputs = try value.cloneRecord(t.arena, outputs),
            .token_scores = try predictionScores(t.arena, responses, true),
        };
        if (t.steps.items.len >= limit) {
            _ = t.steps.orderedRemove(0);
            t.steps.appendAssumeCapacity(step);
            t.evicted_steps +%= 1;
        } else {
            try t.steps.append(t.arena, step);
        }
    }

    fn cloneTraceInputs(arena: Allocator, inputs: *const Record) Allocator.Error!Record {
        const runtime = @import("runtime_value.zig");
        var clone: runtime.CloneContext = .init(arena);
        defer clone.deinit();
        var result: Record = .empty;
        errdefer runtime.freeValue(arena, .{ .object = result });
        try result.ensureTotalCapacity(arena, inputs.count() - 1);
        for (inputs.keys(), inputs.values()) |key, item| {
            if (std.mem.eql(u8, key, "_trace")) continue;
            const owned_key = try arena.dupe(u8, key);
            errdefer arena.free(owned_key);
            result.putAssumeCapacity(owned_key, try clone.cloneValue(item));
        }
        return result;
    }

    /// Structural equality (predictor identity, inputs, outputs, order).
    pub fn eql(a: *const Trace, b: *const Trace) value.ParseError!bool {
        if (a.steps.items.len != b.steps.items.len) return false;
        for (a.steps.items, b.steps.items) |x, y| {
            if (x.predictor != y.predictor) return false;
            if (!(try orderedRecordEqual(&x.inputs, &y.inputs))) return false;
            if (!(try orderedRecordEqual(&x.outputs, &y.outputs))) return false;
        }
        return true;
    }
};

fn orderedRecordEqual(a: *const Record, b: *const Record) value.ParseError!bool {
    if (a.count() != b.count()) return false;
    for (a.keys(), b.keys()) |ka, kb| if (!std.mem.eql(u8, ka, kb)) return false;
    return value.recordEqual(a, b);
}

/// Type-erased, mutable state of one predictor: what DSPy saves and what
/// optimizers change. Slices point to static or caller-arena memory and are
/// replaced, never mutated in place, so copying a program is a cheap clone.
pub const PredictState = struct {
    /// Class identity survives program copies; structural equality is separate.
    signature_class: *const anyopaque,
    fields: []const Field,
    input_count: usize,
    instructions: []const u8,
    descs: []const []const u8,
    prefixes: []const []const u8,
    demos: []const Record = &.{},
    /// Borrowed immutable config, retained across resets and program copies.
    config: ?*const PredictConfig = null,
    /// Borrowed model identity survives program copies; reset clears it.
    lm: ?lm_mod.LM = null,
    /// Persisted Predict.fields overrides, distinct from signature metadata.
    decision_fields: @import("runtime_value.zig").Value = .{ .object = .empty },

    pub fn hasDecisionFields(state: *const PredictState) error{ParseValue}!bool {
        return switch (state.decision_fields) {
            .null => false,
            .bool => |b| b,
            .integer => |n| n != 0,
            .float => |n| n != 0,
            .number_string => |text| (std.fmt.parseFloat(f64, text) catch return true) != 0,
            .string => |text| text.len != 0,
            .array => |items| items.items.len != 0,
            .object => |items| items.count() != 0,
            .custom => |custom| try custom.truthy(),
        };
    }

    pub fn getCriteria(state: *const PredictState, arena: Allocator, name: []const u8) Error!@import("runtime_value.zig").Value {
        var configs = try value.cloneValue(arena, state.decision_fields);
        if (configs != .object) return error.InvalidDecision;
        if (!configs.object.contains(name)) try configs.object.put(arena, name, .{ .object = .empty });
        const snapshot = try @import("decision_state.zig").State.init(arena, try state.view(arena), configs, false, null);
        return snapshot.getCriteria(arena, name);
    }

    pub fn setCriteria(state: *PredictState, arena: Allocator, name: []const u8, criteria: @import("runtime_value.zig").Value) Error!void {
        var configs = try value.cloneValue(arena, state.decision_fields);
        if (configs != .object) return error.InvalidDecision;
        if (!configs.object.contains(name)) try configs.object.put(arena, name, .{ .object = .empty });
        const config = configs.object.getPtr(name).?;
        if (config.* != .object) return error.InvalidDecision;
        try config.object.put(arena, "criteria", try value.cloneValue(arena, criteria));
        _ = try @import("decision_state.zig").State.init(arena, try state.view(arena), configs, false, null);
        state.decision_fields = configs;
    }

    pub fn view(state: *const PredictState, arena: Allocator) Allocator.Error!adapter_mod.View {
        assert(state.descs.len == state.fields.len);
        const fields = try arena.dupe(Field, state.fields);
        for (fields, state.descs, state.prefixes) |*f, d, p| {
            f.desc = d;
            f.prefix = p;
        }
        return .{
            .instructions = state.instructions,
            .inputs = fields[0..state.input_count],
            .outputs = fields[state.input_count..],
        };
    }

    pub fn reset(state: *PredictState) void {
        state.demos = &.{};
        state.lm = null;
    }

    pub fn effectiveLM(state: *const PredictState, base: Context) lm_mod.LM {
        if (base.lm_override_present or base.lm_override != null) return base.lm_override orelse base.lm;
        if (PredictRollout.find(base.predict_rollout, state)) |rollout| return rollout.lm;
        return state.lm orelse base.lm;
    }

    /// Resolve a predictor's LM and sampling defaults before applying config.
    pub fn resolveContext(state: *const PredictState, base: Context) Error!Context {
        var ctx = base;
        const override = base.lm_override_present or base.lm_override != null;
        const rollout = if (override) null else PredictRollout.find(base.predict_rollout, state);
        const explicit_lm = if (override) base.lm_override else if (rollout) |scope| scope.lm else state.lm;
        ctx.lm = explicit_lm orelse base.lm;
        if (explicit_lm != null) {
            if (base.before_settings_rollout) |before| {
                ctx.lm_options.temperature = before.temperature;
                ctx.lm_options.rollout_id = before.rollout_id;
            }
        }
        if (rollout) |scope| {
            ctx.lm_options.temperature = 1;
            ctx.lm_options.rollout_id = scope.id;
        }
        ctx.before_settings_rollout = null;
        if (ctx.lm.sameInstance(lm_mod.LM.unconfigured)) return error.NoLM;
        return ctx;
    }

    /// Signature.with_instructions/with_updated_fields/load_state return a new
    /// class even when the resulting metadata is structurally unchanged.
    pub fn forkSignatureClass(state: *PredictState, arena: Allocator) Allocator.Error!void {
        const identity = try arena.create(u8);
        identity.* = 0;
        state.signature_class = identity;
    }

    /// Run one LM call: format, complete, parse (with JSON fallback), trace.
    pub fn call(state: *const PredictState, ctx_in: Context, inputs: *const Record) Error!Record {
        return (try state.callConfigured(ctx_in, inputs, null)).records[0];
    }

    /// `Predict(..., n=n)`: one LM call for `n` completions, all parsed with
    /// the same adapter (DSPy falls back to JSON for the whole call). Like
    /// DSPy, a temperature that is unset or at most 0.15 becomes 0.7.
    pub fn callMany(state: *const PredictState, ctx_in: Context, inputs: *const Record, n: u32) Error![]const Record {
        assert(n >= 1);
        return (try state.callConfigured(ctx_in, inputs, n)).records;
    }

    pub fn configuredOptions(state: *const PredictState, ctx: Context) lm_mod.Options {
        var options = ctx.lm_options;
        if (state.config) |config| config.apply(&options);
        if (ctx.predict_config) |config| config.apply(&options);
        return options;
    }

    fn callConfigured(state: *const PredictState, ctx_in: Context, inputs: *const Record, count: ?u32) Error!ParsedBatch {
        if (ctx_in.parse_failure) |capture| capture.* = .{};
        var ctx = try state.resolveContext(ctx_in);
        const base_temperature = ctx.lm_options.temperature orelse 0;
        ctx.lm_options = state.configuredOptions(ctx);
        if (count) |n| ctx.lm_options.n = n;
        if (ctx.lm_options.n == 0) return error.InvalidCompletionCount;
        const temperature = ctx.lm_options.temperature orelse 0;
        const sampling_temperature = if (temperature == 0) base_temperature else temperature;
        if (ctx.lm_options.n > 1 and sampling_temperature <= 0.15) ctx.lm_options.temperature = 0.7;
        var defaulted_inputs: Record = undefined;
        const prepared = try state.prepare(&ctx, inputs, &defaulted_inputs);
        const outputs = try runPredict(state, ctx, prepared);
        if (prepared.view.findOutput("logprobs") != null) {
            // DSPy overwrites the parsed field even for a reported-empty
            // score list. A typed native output cannot change shape at runtime.
            for (outputs.responses) |response| {
                if (response.logprobs != null) return error.ConflictingCompletionLogprobs;
            }
        } else if (outputs.responses.len > 1) {
            // DSPy's Completions refuses a ragged new logprobs column.
            const reported = outputs.responses[0].logprobs != null;
            for (outputs.responses[1..]) |response| {
                if ((response.logprobs != null) != reported) return error.InconsistentCompletionLogprobs;
            }
        }
        const trace_control = prepared.inputs.get("_trace");
        // DSPy tests the control after inference, even with no active trace.
        if (trace_control == null or try traceTruthy(trace_control.?)) {
            if (ctx.trace) |t| try t.recordLimited(state, prepared.inputs, &outputs.records[0], outputs.responses, @intCast(@max(0, ctx.max_trace_size)), trace_control != null);
        }
        return outputs;
    }

    fn prepare(state: *const PredictState, ctx: *Context, inputs: *const Record, defaulted: *Record) Error!PreparedCall {
        const view_ = try state.view(ctx.arena);
        const hint = predictionHint(inputs);
        if (hint) |arg| ctx.lm_options.prediction = arg;
        var missing_defaults: usize = 0;
        for (view_.inputs) |field| {
            if (field.default_value == null) continue;
            if (!inputs.contains(field.name) or (hint != null and isPrediction(field.name)))
                missing_defaults += 1;
        }
        const rebuild = hint != null or missing_defaults != 0;
        if (rebuild) {
            defaulted.* = .empty;
            try defaulted.ensureTotalCapacity(
                ctx.arena,
                inputs.count() - @intFromBool(hint != null) + missing_defaults,
            );
            for (inputs.keys(), inputs.values()) |key, item| {
                if (hint != null and isPrediction(key)) continue;
                defaulted.putAssumeCapacity(key, item);
            }
            for (view_.inputs) |field| {
                const default = field.default_value orelse continue;
                if (!defaulted.contains(field.name))
                    defaulted.putAssumeCapacity(field.name, try default(ctx.arena));
            }
        }
        const call_: PreparedCall = .{
            .signature_class = state.signature_class,
            .view = view_,
            .inputs = if (rebuild) defaulted else inputs,
        };
        return if (ctx.preparation) |wrapper| wrapper.apply(ctx.arena, call_) else call_;
    }

    fn traceTruthy(flag: @import("runtime_value.zig").Value) value.ParseError!bool {
        return switch (flag) {
            .null => false,
            .bool => |b| b,
            .integer => |n| n != 0,
            .float => |n| n != 0,
            .number_string => |text| (std.fmt.parseFloat(f64, text) catch return true) != 0,
            .string => |s| s.len != 0,
            .array => |a| a.items.len != 0,
            .object => |o| o.count() != 0,
            .custom => |c| c.truthy(),
        };
    }

    fn predictionHint(inputs: *const Record) ?@import("runtime_value.zig").Value {
        const hint = inputs.get("prediction") orelse return null;
        if (hint != .object) return null;
        const kind = hint.object.get("type") orelse return null;
        if (kind != .string or !std.mem.eql(u8, kind.string, "content")) return null;
        if (!hint.object.contains("content")) return null;
        return hint;
    }

    fn isPrediction(name: []const u8) bool {
        return std.mem.eql(u8, name, "prediction");
    }
};

const ParsedBatch = struct {
    records: []const Record,
    responses: []const lm_mod.Completion = &.{},
};

fn runWithFallback(ctx: Context, predictor: *const PredictState, signature_class: *const anyopaque, view_: adapter_mod.View, demos: []const Record, inputs: *const Record) Error!ParsedBatch {
    var published = false;
    return runAdapterMany(ctx, predictor, signature_class, ctx.adapter, view_, demos, inputs, &published) catch |err| switch (err) {
        error.AdapterParse => if (!published and (ctx.adapter == .chat or ctx.adapter == .xml) and ctx.json_fallback)
            try runAdapterMany(ctx, predictor, signature_class, .json, view_, demos, inputs, null)
        else
            return err,
        else => return err,
    };
}

fn runPredict(predictor: *const PredictState, ctx: Context, prepared: PreparedCall) Error!ParsedBatch {
    const native = ctx.lm.supports_decision_requests;
    var active = native or predictor.decision_fields != .object;
    if (predictor.decision_fields == .object) active = active or predictor.decision_fields.object.count() != 0;
    for (prepared.view.outputs) |field| {
        if (field.decision) |binding| active = active or binding.automatic;
    }
    const demos = ctx.demos_override orelse predictor.demos;
    if (!active) return runWithFallback(ctx, predictor, prepared.signature_class, prepared.view, demos, prepared.inputs);
    if (native) if (ctx.stream_session) |session| {
        if (session.selects(predictor)) return error.UnsupportedStreamField;
    };
    const declared = if (predictor.decision_fields == .object and predictor.decision_fields.object.count() != 0)
        try predictor.view(ctx.arena)
    else
        null;
    const snapshot = try @import("decision_state.zig").State.init(ctx.arena, prepared.view, predictor.decision_fields, native, declared);
    var questions: Record = .empty;
    for (snapshot.types) |entry| try questions.put(ctx.arena, entry.field.name, try snapshot.question(ctx.arena, entry.field.name));
    const batch: ParsedBatch = if (native)
        .{ .records = try runSystemOne(ctx, prepared, demos, snapshot, &questions) }
    else blk: {
        var wire = prepared.view;
        const fields = try ctx.arena.dupe(Field, wire.outputs);
        for (fields) |*field| {
            const binding = field.decision orelse continue;
            if (!questions.contains(field.name)) continue;
            var description: std.Io.Writer.Allocating = .init(ctx.arena);
            description.writer.writeByte('\n') catch return error.OutOfMemory;
            @import("pyjson.zig").writeJson(&description.writer, questions.get(field.name).?, .indent2) catch |err| return adapter_mod.mapWriteError(err);
            field.* = .{
                .name = field.name,
                .kind = .json,
                .type_name = binding.evidence_name,
                .desc = description.written(),
                .prefix = field.prefix,
                .schema = binding.evidence_schema,
                .xml_schema = binding.evidence_schema,
                .validate = binding.validate_evidence,
                .xml_validate = binding.validate_evidence,
                .xml_literal = true,
                .type_description = if (binding.rich) "" else field.type_description,
                .type_description_name = if (binding.rich) "" else field.type_description_name,
            };
        }
        wire.outputs = fields;
        if (demos.len != 0) {
            const examples = try decisionDemos(ctx.arena, prepared.view, demos);
            var instructions: std.Io.Writer.Allocating = .init(ctx.arena);
            instructions.writer.writeAll(wire.instructions) catch return error.OutOfMemory;
            instructions.writer.writeAll("\n\nTask examples (labels or evidence):\n") catch return error.OutOfMemory;
            @import("pyjson.zig").writeJsonAscii(&instructions.writer, .{ .array = examples }) catch |err| return adapter_mod.mapWriteError(err);
            wire.instructions = instructions.written();
        }
        // with_updated_fields creates a new class whenever evidence is requested.
        const wire_class = if (snapshot.types.len != 0) ParseFailure.derived_signature else prepared.signature_class;
        break :blk try runWithFallback(ctx, predictor, wire_class, wire, &.{}, prepared.inputs);
    };
    const outputs = try ctx.arena.alloc(Record, batch.records.len);
    for (batch.records, outputs) |completion, *output| {
        var answers: Record = .empty;
        output.* = .empty;
        try output.ensureTotalCapacity(ctx.arena, completion.count());
        // Retain completion order and extras without cloning evidence that the
        // typed decision/box result immediately replaces.
        for (completion.keys(), completion.values()) |name, raw| {
            var replaced = false;
            for (snapshot.types) |entry| {
                if (std.mem.eql(u8, name, entry.field.name)) {
                    replaced = true;
                    break;
                }
            }
            if (native and !replaced) {
                if (prepared.view.findOutput(name)) |field| replaced = field.native_bounding_box;
            }
            output.putAssumeCapacity(try ctx.arena.dupe(u8, name), if (replaced) .null else try value.cloneValue(ctx.arena, raw));
        }
        if (native) for (prepared.view.outputs) |field| {
            if (!field.native_bounding_box) continue;
            const answer = completion.get(field.name) orelse return error.InvalidDecision;
            if (answer != .object) return error.InvalidDecision;
            const raw_box = answer.object.get("box") orelse return error.InvalidDecision;
            const parsed_box = @import("types.zig").validate(@import("bounding_box.zig").BoundingBox, ctx.arena, raw_box) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => {
                    try captureBoxFailure(ctx, prepared, completion);
                    return error.AdapterParse;
                },
            };
            try output.put(ctx.arena, field.name, parsed_box);
            try answers.put(ctx.arena, field.name, try value.cloneValue(ctx.arena, answer));
        };
        for (snapshot.types) |entry| {
            const raw_answer = completion.get(entry.field.name) orelse return error.InvalidDecision;
            const answer = if (native) raw_answer else entry.binding.evidence_value(ctx.arena, raw_answer) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.AdapterParse,
            };
            if (entry.binding.model.kind == .score) {
                if (answer != .object) return error.InvalidDecision;
                const probabilities = answer.object.get("probabilities") orelse return error.InvalidDecision;
                if (probabilities != .object or probabilities.object.count() != entry.binding.model.labels.len) return error.InvalidDecision;
                for (probabilities.object.keys()) |label| {
                    var found = false;
                    for (entry.binding.model.labels) |expected| found = found or std.mem.eql(u8, label, expected);
                    if (!found) return error.InvalidDecision;
                }
            }
            var copy = try value.cloneValue(ctx.arena, answer);
            if (entry.binding.model.kind == .score) {
                const probabilities = copy.object.get("probabilities").?;
                const pairs = try ctx.arena.alloc(@import("runtime_value.zig").PythonEntry, probabilities.object.count());
                for (probabilities.object.keys(), probabilities.object.values(), pairs) |label, probability_, *pair| {
                    pair.* = .{ .key = .{ .integer = std.fmt.parseInt(i64, label, 10) catch return error.InvalidDecision }, .value = probability_ };
                }
                const dictionary = @import("python_value.zig").dictionary(ctx.arena, pairs) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidDecision,
                };
                try copy.object.put(ctx.arena, "probabilities", dictionary);
            }
            try answers.put(ctx.arena, entry.field.name, copy);
        }
        if (ctx.decision_evidence) |collector| try collector.record(predictor, &answers);
        const decoded = try snapshot.decode(ctx.arena, &answers);
        for (decoded.keys(), decoded.values()) |name, result| try output.put(ctx.arena, name, result);
    }
    return .{ .records = outputs, .responses = batch.responses };
}

/// DecisionAdapter's only AdapterParseError: an invalid native bounding box.
/// It names the call signature and `json.dumps` of the answers, with no
/// partial result.
fn captureBoxFailure(ctx: Context, prepared: PreparedCall, answers: Record) Error!void {
    const capture = ctx.parse_failure orelse return;
    assert(capture.signature_class == null);
    var text: std.Io.Writer.Allocating = .init(ctx.arena);
    @import("pyjson.zig").writeJsonAscii(&text.writer, .{ .object = answers }) catch |err| return adapter_mod.mapWriteError(err);
    capture.* = .{
        .signature_class = prepared.signature_class,
        .expected_count = @intCast(prepared.view.outputs.len),
        .failure = .{ .completion_text = text.written() },
    };
}

fn decisionDemos(arena: Allocator, view_: adapter_mod.View, demos: []const Record) Error!@import("runtime_value.zig").Array {
    var examples: @import("runtime_value.zig").Array = .init(arena);
    try examples.ensureTotalCapacity(demos.len);
    for (demos) |demo| {
        var filtered: Record = .empty;
        for (demo.keys(), demo.values()) |name, entry| {
            if (view_.hasField(name)) try filtered.put(arena, name, entry);
        }
        try examples.append(value.serializedValue(arena, .{ .object = filtered }) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidDecision,
        });
    }
    return examples;
}

fn runSystemOne(ctx: Context, prepared: PreparedCall, demos: []const Record, snapshot: @import("decision_state.zig").State, questions: *Record) Error![]const Record {
    const options = ctx.lm_options;
    if (options.temperature != null or options.top_p != null or options.top_k != .null or options.seed != .null or
        options.frequency_penalty != null or options.presence_penalty != null or
        options.rollout_id != null or options.n != 1 or
        options.reasoning_effort != null or options.reasoning_effort_present or options.response_format != null or
        options.tool_choice != null or options.parallel_tool_calls != null or
        options.logprobs != .null or options.prediction != null or options.max_tokens != null or
        options.stop.len != 0 or options.store != null or options.user_id != null) return error.InvalidDecision;
    var state: Record = .empty;
    try state.put(ctx.arena, "instructions", .{ .string = prepared.view.instructions });
    try state.put(ctx.arena, "input_fields", .{ .string = try adapter_mod.fieldDescriptionString(ctx.arena, prepared.view.inputs) });
    var inputs: Record = .empty;
    for (prepared.inputs.keys(), prepared.inputs.values()) |name, entry| {
        if (prepared.view.findInput(name) != null) try inputs.put(ctx.arena, name, entry);
    }
    try state.put(ctx.arena, "inputs", .{ .object = inputs });
    if (demos.len != 0) try state.put(ctx.arena, "demos", .{ .array = try decisionDemos(ctx.arena, prepared.view, demos) });
    var properties: Record = .empty;
    for (snapshot.types) |entry| {
        var property: Record = .empty;
        const kind = entry.binding.model.kind;
        try property.put(ctx.arena, "type", .{ .string = switch (kind) {
            .noul => "boolean",
            .score => "integer",
            .choice => "string",
        } });
        try property.put(ctx.arena, "description", .{ .string = entry.field.name });
        if (kind == .score) {
            var branches: @import("runtime_value.zig").Array = .init(ctx.arena);
            for (entry.binding.model.labels, 0..) |label, i| {
                var branch: Record = .empty;
                try branch.put(ctx.arena, "const", .{ .integer = @intCast(i) });
                try branch.put(ctx.arena, "description", .{ .string = label });
                try branches.append(.{ .object = branch });
            }
            try property.put(ctx.arena, "anyOf", .{ .array = branches });
        } else if (kind == .choice) {
            const question = questions.get(entry.field.name).?;
            const criteria = question.object.get("criteria") orelse return error.InvalidDecision;
            if (criteria != .object) return error.InvalidDecision;
            var labels: @import("runtime_value.zig").Array = .init(ctx.arena);
            for (criteria.object.keys()) |label| {
                if (label.len == 0) return error.InvalidDecision;
                try labels.append(.{ .string = label });
            }
            try property.put(ctx.arena, "enum", .{ .array = labels });
        }
        try properties.put(ctx.arena, entry.field.name, .{ .object = property });
    }
    for (prepared.view.outputs) |field| {
        if (!field.native_bounding_box) continue;
        const default_description = try std.fmt.allocPrint(ctx.arena, "${{{s}}}", .{field.name});
        const description = if (field.desc.len == 0 or std.mem.eql(u8, field.desc, default_description)) field.name else field.desc;
        var question: Record = .empty;
        try question.put(ctx.arena, "type", .{ .string = "box" });
        try question.put(ctx.arena, "instructions", .{ .string = description });
        try questions.put(ctx.arena, field.name, .{ .object = question });
        var property: Record = .empty;
        try property.put(ctx.arena, "type", .{ .string = "object" });
        try property.put(ctx.arena, "description", .{ .string = description });
        var edges: Record = .empty;
        var required: @import("runtime_value.zig").Array = .init(ctx.arena);
        const edge_descriptions = [_][]const u8{
            "Top edge, as a fraction of the image's height from the top, from 0 to 1",
            "Bottom edge, as a fraction of the image's height from the top, from 0 to 1",
            "Left edge, as a fraction of the image's width from the left, from 0 to 1",
            "Right edge, as a fraction of the image's width from the left, from 0 to 1",
        };
        for ([_][]const u8{ "top", "bottom", "left", "right" }, edge_descriptions) |edge, edge_description| {
            var schema: Record = .empty;
            try schema.put(ctx.arena, "type", .{ .string = "number" });
            try schema.put(ctx.arena, "description", .{ .string = edge_description });
            try edges.put(ctx.arena, edge, .{ .object = schema });
            try required.append(.{ .string = edge });
        }
        try property.put(ctx.arena, "properties", .{ .object = edges });
        try property.put(ctx.arena, "required", .{ .array = required });
        try property.put(ctx.arena, "additionalProperties", .{ .bool = false });
        try properties.put(ctx.arena, field.name, .{ .object = property });
    }
    var response_schema: Record = .empty;
    try response_schema.put(ctx.arena, "type", .{ .string = "object" });
    try response_schema.put(ctx.arena, "properties", .{ .object = properties });
    // SystemOne has its own generation configuration and adaptation policy.
    const completions = try ctx.lm.completeResponses(ctx.arena, .{
        .messages = &.{},
        .options = .{ .cache = options.cache, .service_tier = options.service_tier },
        .usage_tracker = ctx.usage_tracker,
        .usage_failure = ctx.usage_failure,
        .decision = .{
            .state = value.serializedValue(ctx.arena, .{ .object = state }) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidDecision,
            },
            .questions = .{ .object = questions.* },
            .response_schema = .{ .object = response_schema },
        },
    });
    if (completions.len != 1) return error.InvalidDecision;
    const payload = completions[0].provider_data orelse return error.InvalidDecision;
    const provider_data = payload.toValue(ctx.arena) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidDecision,
    };
    if (provider_data != .object) return error.InvalidDecision;
    const typesafe = provider_data.object.get("typesafe") orelse return error.InvalidDecision;
    if (typesafe != .object) return error.InvalidDecision;
    const answers = typesafe.object.get("answers") orelse return error.InvalidDecision;
    if (answers != .object) return error.InvalidDecision;
    const result = try ctx.arena.alloc(Record, 1);
    result[0] = answers.object;
    return result;
}

fn adapterCompletions(
    ctx: Context,
    predictor: ?*const PredictState,
    kind: Adapter,
    original: adapter_mod.View,
    wire: []const Field,
    native: stream.NativeFields,
    request: lm_mod.Request,
    published: ?*bool,
) Error![]const lm_mod.Completion {
    const session = ctx.stream_session orelse return completeWithStatus(ctx, request);
    const target = predictor orelse return completeWithStatus(ctx, request);
    if (!session.selects(target)) return completeWithStatus(ctx, request);
    if (session.in_flight.swap(true, .acquire)) return error.StreamSessionBusy;
    defer session.in_flight.store(false, .release);
    var attempt: stream.Attempt = .{ .session = session, .target = target, .published = published };
    try attempt.prepare(ctx.arena, kind, original.outputs, wire, native);
    var routed = request;
    routed.stream_sink = attempt.sink();
    const result = completeWithStatus(ctx, routed);
    if (attempt.failure) |err| return err;
    return result;
}

fn completeWithStatus(ctx: Context, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
    const sink = ctx.status_sink orelse return ctx.lm.completeResponses(ctx.arena, request);
    sink.notify(.{ .lm_start = .{ .instance = ctx.lm, .request = &request } });
    const result = ctx.lm.completeResponses(ctx.arena, request);
    if (result) |output| {
        sink.notify(.{ .lm_end = .{ .output = output } });
    } else |failure| {
        sink.notify(.{ .lm_end = .{ .failure = failure } });
    }
    return result;
}

/// `signature_class` identifies the prepared signature for `ctx.parse_failure`.
fn runAdapterMany(ctx: Context, predictor: ?*const PredictState, signature_class: ?*const anyopaque, adapter: Adapter, original: adapter_mod.View, demos: []const Record, inputs: *const Record, published: ?*bool) Error!ParsedBatch {
    // An earlier attempt's failure must not outlive this attempt's outcome.
    if (ctx.parse_failure) |capture| capture.* = .{};
    if (adapter == .two_step and ctx.two_step == null) return error.AdapterParse;
    var options = ctx.lm_options;
    var request_tools: []const inference.FunctionTool = &.{};
    var tool_input: ?Field = null;
    var tool_output: ?Field = null;
    for (original.inputs) |f| if (f.native_tools) {
        tool_input = f;
        break;
    };
    for (original.outputs) |f| if (f.native_tool_calls) {
        tool_output = f;
        break;
    };
    // DSPy's implicit XML default enables native History rendering even if
    // this LM cannot execute native function declarations.
    const native_policy = ctx.nativeFunctionCallingEnabled();
    // This is a signature configuration error, not a completion parse error:
    // reject it even for an LM without native function-calling capability.
    if (native_policy and tool_output != null and tool_input == null) return error.MissingToolInput;
    const use_native_tools = native_policy and ctx.lm.supports_function_calling and tool_output != null;
    // With native calling enabled, only a declared ToolCalls output makes
    // adapter preprocessing conditional on the LM's tool capability.
    if (!native_policy or (tool_output != null and !ctx.lm.supports_function_calling)) {
        options.tool_choice = null;
        options.parallel_tool_calls = null;
    }
    const effort = options.reasoning_effort;
    // DSPy 3.5's adapter only treats null and "none" as disabling native
    // signature adaptation. "off" still removes the field from the prompt
    // and is passed through to disable reasoning at the provider boundary.
    const native = ctx.lm.supports_reasoning and
        !(options.reasoning_effort_present and effort == null) and
        !(effort != null and std.mem.eql(u8, effort.?, "none"));
    var needs_preprocess = false;
    for (original.outputs) |f| needs_preprocess = needs_preprocess or f.native_reasoning;
    var processed = original;
    processed.use_native_function_calling = native_policy;
    if (use_native_tools) {
        const field = tool_input.?;
        request_tools = tool_mod.functionToolsFromValue(ctx.arena, inputs.get(field.name) orelse return error.AdapterParse) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.AdapterParse,
        };
        var fields: std.ArrayList(Field) = .empty;
        try fields.ensureTotalCapacity(ctx.arena, original.inputs.len);
        for (original.inputs) |f| if (!std.mem.eql(u8, f.name, field.name)) {
            fields.appendAssumeCapacity(f);
        };
        processed.inputs = fields.items;
    }
    var processed_demos = demos;
    var history_renames: std.ArrayList(Adapter.HistoryRename) = .empty;
    const has_history = if (needs_preprocess and !native) blk: {
        for (processed.inputs) |field| {
            if (std.mem.eql(u8, field.type_name, "History") and inputs.contains(field.name)) break :blk true;
        }
        break :blk false;
    } else false;
    if (needs_preprocess or use_native_tools) {
        var fields: std.ArrayList(Field) = .empty;
        try fields.ensureTotalCapacity(ctx.arena, original.outputs.len);
        for (original.outputs) |f| {
            if (use_native_tools and std.mem.eql(u8, f.name, tool_output.?.name)) continue;
            if (f.native_reasoning and native) {
                if (options.reasoning_effort == null) options.reasoning_effort = "low";
                continue;
            }
            var wire = f;
            if (f.native_reasoning) {
                wire.name = try fallbackName(ctx.arena, original, f);
                if (has_history and !std.mem.eql(u8, wire.name, f.name)) {
                    try history_renames.append(ctx.arena, .{ .from = f.name, .to = wire.name });
                }
            }
            fields.appendAssumeCapacity(wire);
        }
        processed.outputs = fields.items;
        const mapped = try ctx.arena.alloc(Record, demos.len);
        for (demos, mapped) |demo, *dst| {
            dst.* = demo;
            for (original.outputs) |f| {
                if (!f.native_reasoning or native) continue;
                const name = try fallbackName(ctx.arena, original, f);
                if (std.mem.eql(u8, name, f.name)) continue;
                if (dst.*.get(f.name)) |v| {
                    // Copy only records whose keys actually change.
                    if (dst.*.keys().ptr == demo.keys().ptr) dst.* = try value.cloneRecord(ctx.arena, &demo);
                    _ = dst.*.orderedRemove(f.name);
                    const text = if (v == .object) v.object.get("content") orelse v else v;
                    try dst.*.put(ctx.arena, name, text);
                }
            }
        }
        processed_demos = mapped;
    }
    if (adapter == .json and ctx.lm.supports_response_format) {
        options.response_format = try adapter_mod.responseFormat(ctx.arena, processed, ctx.lm.supports_response_schema);
        if (options.response_format.? == .json_schema) {
            options.response_format.?.json_schema.schema = @import("inference.zig").closeObjectSchemas(ctx.arena, options.response_format.?.json_schema.schema) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.JsonTooDeep => error.JsonTooDeep,
                error.InvalidSchema => error.AdapterParse,
            };
        }
    }
    // Explicit-null intent has now been resolved into the effective signature.
    options.reasoning_effort_present = false;
    const messages = try adapter.formatWithHistoryRenames(ctx.arena, processed, processed_demos, inputs, history_renames.items);
    const completions = try adapterCompletions(
        ctx,
        predictor,
        adapter,
        original,
        processed.outputs,
        .{ .reasoning = native, .tools = use_native_tools },
        .{
            .messages = messages,
            .options = options,
            .tools = request_tools,
            .stream_sink = ctx.stream_sink,
            .usage_tracker = ctx.usage_tracker,
            .usage_failure = ctx.usage_failure,
        },
        published,
    );
    const outputs = try ctx.arena.alloc(Record, completions.len);
    var has_citation_field = false;
    for (original.outputs) |field| has_citation_field = has_citation_field or field.native_citations;
    for (completions, outputs) |c, *o| {
        // A response can carry native tool calls even if this request did not
        // declare tools. Only the first declared ToolCalls output receives them.
        const has_tool_calls = tool_output != null and c.tool_calls.len > 0;
        if (!has_tool_calls and (c.text == null or c.text.?.len == 0)) return emptyResponse(ctx, signature_class, original);
        const parsed = if (has_tool_calls) blk: {
            const text = c.text orelse break :blk Record.empty;
            if (text.len == 0 or processed.outputs.len == 0) break :blk Record.empty;
            // A tolerated failure beside native tool calls is never captured.
            break :blk parseCompletion(ctx, adapter, processed, text, null) catch |err| switch (err) {
                error.AdapterParse => Record.empty,
                else => return err,
            };
        } else blk: {
            const text = c.text orelse return emptyResponse(ctx, signature_class, original);
            const capture = ctx.parse_failure orelse break :blk try parseCompletion(ctx, adapter, processed, text, null);
            break :blk parseCompletion(ctx, adapter, processed, text, &capture.failure) catch |err| {
                if (err == error.AdapterParse) try finishParseFailure(ctx.arena, capture, signature_class, original, native);
                return err;
            };
        };
        if (!needs_preprocess and !use_native_tools and !has_tool_calls and !has_citation_field and adapter != .two_step) {
            o.* = parsed;
            continue;
        }
        o.* = .empty;
        try o.ensureTotalCapacity(ctx.arena, original.outputs.len);
        // DSPy `_call_postprocess` appends fields removed for native
        // features (native reasoning, native tool calls) after the parse.
        for ([_]bool{ false, true }) |removed_pass| for (original.outputs) |f| {
            const removed = (f.native_reasoning and native) or
                (use_native_tools and std.mem.eql(u8, f.name, tool_output.?.name));
            if (removed != removed_pass) continue;
            const is_tool_output = if (tool_output) |field| std.mem.eql(u8, f.name, field.name) else false;
            const v = if (f.native_citations and c.citations.len != 0)
                try (try @import("citation.zig").Citations.fromParts(ctx.arena, c.citations)).toRuntime(ctx.arena)
            else if (is_tool_output and has_tool_calls)
                tool_mod.callsFromNative(ctx.arena, c.tool_calls) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.AdapterParse,
                }
            else if (f.native_reasoning and native)
                if (c.reasoning) |r| if (r.len > 0) try (signature.Reasoning{ .content = r }).toRuntime(ctx.arena) else .null else .null
            else
                parsed.get(if (f.native_reasoning) try fallbackName(ctx.arena, original, f) else f.name) orelse
                    (try signature.outputDefault(ctx.arena, f)) orelse
                    if (has_tool_calls or (use_native_tools and is_tool_output)) @as(@import("runtime_value.zig").Value, .null) else return error.AdapterParse;
            o.putAssumeCapacity(f.name, v);
        };
    }
    return .{ .records = outputs, .responses = completions };
}

fn parseCompletion(ctx: Context, adapter: Adapter, view_: adapter_mod.View, text: []const u8, failure: ?*adapter_mod.ParseFailure) Error!Record {
    if (adapter != .two_step) return adapter.parseWithFailure(ctx.arena, view_, text, failure);
    // DSPy rewraps an extraction parse error around the main completion,
    // without the extractor's partial result.
    return ctx.two_step.?.parse(ctx, view_, text) catch |err| {
        if (err == error.AdapterParse) if (failure) |f| {
            f.* = .{ .completion_text = text };
        };
        return err;
    };
}

/// DSPy rejects a null or empty response before any parsing.
fn emptyResponse(ctx: Context, signature_class: ?*const anyopaque, original: adapter_mod.View) error{AdapterParse} {
    if (ctx.parse_failure) |capture| {
        assert(signature_class != null);
        capture.* = .{ .signature_class = signature_class, .expected_count = @intCast(original.outputs.len) };
    }
    return error.AdapterParse;
}

/// Mirrors `_call_postprocess`: report the original signature and restore
/// the original names of reasoning fallback fields in the partial result.
fn finishParseFailure(
    arena: Allocator,
    capture: *ParseFailure,
    signature_class: ?*const anyopaque,
    original: adapter_mod.View,
    native: bool,
) Allocator.Error!void {
    assert(signature_class != null);
    assert(capture.signature_class == null);
    errdefer capture.* = .{};
    if (capture.failure.parsed_result) |*partial| {
        for (original.outputs) |f| {
            if (!f.native_reasoning or native) continue;
            const name = try fallbackName(arena, original, f);
            if (std.mem.eql(u8, name, f.name)) continue;
            const entry = partial.fetchOrderedRemove(name) orelse continue;
            try partial.put(arena, f.name, entry.value);
        }
    }
    capture.signature_class = signature_class;
    capture.expected_count = @intCast(original.outputs.len);
}

fn fallbackName(arena: Allocator, view_: adapter_mod.View, f: Field) Allocator.Error![]const u8 {
    const base = f.reasoning_fallback orelse return f.name;
    if (!view_.hasField(base)) return base;
    var suffix: usize = 2;
    while (suffix <= view_.inputs.len + view_.outputs.len + 2) : (suffix += 1) {
        const name = try std.fmt.allocPrint(arena, "{s}_{d}", .{ base, suffix });
        if (!view_.hasField(name)) return name;
    }
    unreachable;
}

fn predictionScores(arena: Allocator, responses: []const lm_mod.Completion, comptime owned: bool) Allocator.Error!?[]const ?CompletionLogprobs {
    for (responses) |response| {
        if (response.logprobs != null or !response.logprobs_complete) break;
    } else return null;

    const scores = try arena.alloc(?CompletionLogprobs, responses.len);
    var written: usize = 0;
    errdefer if (owned) CompletionLogprobs.freeRows(arena, scores, written);
    for (responses, scores) |response, *score| {
        score.* = if (response.logprobs != null or !response.logprobs_complete) blk: {
            const row: CompletionLogprobs = .{
                .tokens = response.logprobs orelse &.{},
                .complete = response.logprobs_complete,
                .reported = response.logprobs != null,
            };
            break :blk if (owned) try row.clone(arena) else row;
        } else null;
        if (owned) written += 1;
    }
    return scores;
}

/// `dspy.Predict(Signature)`.
pub fn Predict(comptime Sig: type) type {
    comptime assert(signature.isSignature(Sig));
    return struct {
        const Self = @This();
        pub const Signature = Sig;
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        pub const is_dszig_predict = true;

        const default_descs = blk: {
            var d: [Sig.fields.len][]const u8 = undefined;
            for (Sig.fields, 0..) |f, i| d[i] = f.desc;
            break :blk d;
        };
        const default_prefixes = blk: {
            var p: [Sig.fields.len][]const u8 = undefined;
            for (Sig.fields, 0..) |f, i| p[i] = f.prefix;
            break :blk p;
        };

        var signature_class_token: u8 = 0;

        state: PredictState = .{
            .signature_class = &signature_class_token,
            .fields = Sig.fields,
            .input_count = Sig.input_fields_list.len,
            .instructions = Sig.instructions,
            .descs = &default_descs,
            .prefixes = &default_prefixes,
        },
        compiled: bool = false,

        pub fn getCriteria(self: *const Self, arena: Allocator, name: []const u8) Error!@import("runtime_value.zig").Value {
            return self.state.getCriteria(arena, name);
        }

        pub fn setCriteria(self: *Self, arena: Allocator, name: []const u8, criteria: @import("runtime_value.zig").Value) Error!void {
            try self.state.setCriteria(arena, name, criteria);
        }

        pub fn forward(self: *const Self, ctx: Context, input: Input) Error!Output {
            const inputs = try value.toRecord(ctx.arena, Input, input);
            const outputs = try self.state.call(ctx, &inputs);
            return value.fromRecord(ctx.arena, Output, &outputs);
        }

        /// `forward` as a genuine Prediction retaining every parsed completion
        /// of the one public LM call (the resolved `n`). No Module-call
        /// scope: usage stays unset, like a direct Source `forward`.
        pub fn forwardPrediction(self: *const Self, ctx: Context, input: Input) Error!Prediction(Output) {
            const inputs = try value.toRecord(ctx.arena, Input, input);
            const batch = try self.state.callConfigured(ctx, &inputs, null);
            const scores = try predictionScores(ctx.arena, batch.responses, false);
            const outputs = try ctx.arena.alloc(Output, batch.records.len);
            for (batch.records, outputs) |*record, *output|
                output.* = try value.fromRecord(ctx.arena, Output, record);
            const prediction = try Prediction(Output).fromCompletions(ctx.arena, outputs);
            prediction.state.token_scores = scores;
            return prediction;
        }

        /// `Module.__call__`: see `callProgram`.
        pub fn call(self: *const Self, ctx: Context, input: Input) Error!Prediction(Output) {
            return callProgram(self, ctx, input);
        }
    };
}

/// A predictor over a comptime field list, for modules that build their own
/// signatures (`ReAct`).
pub fn PredictFields(comptime fields: []const Field, comptime input_count: usize, comptime instructions: []const u8) type {
    comptime assert(input_count < fields.len);
    return struct {
        pub const is_dszig_predict = true;

        const default_descs = blk: {
            var d: [fields.len][]const u8 = undefined;
            for (fields, 0..) |f, i| d[i] = f.desc;
            break :blk d;
        };
        const default_prefixes = blk: {
            var p: [fields.len][]const u8 = undefined;
            for (fields, 0..) |f, i| p[i] = f.prefix;
            break :blk p;
        };

        var signature_class_token: u8 = 0;

        state: PredictState = .{
            .signature_class = &signature_class_token,
            .fields = fields,
            .input_count = input_count,
            .instructions = instructions,
            .descs = &default_descs,
            .prefixes = &default_prefixes,
        },
        compiled: bool = false,
    };
}

/// DSPy 3.5 CoT uses native reasoning, or an explicit plan on other models.
pub fn ChainOfThought(comptime Sig: type) type {
    return ChainOfThoughtWithOptions(Sig, .{});
}

pub const ChainOfThoughtOptions = struct {
    /// Explicit annotation requests a normal output, like rationale_field_type=str.
    rationale_field_type: ?type = null,
    rationale_description: ?[]const u8 = null,
};

pub fn ChainOfThoughtWithOptions(comptime Sig: type, comptime options: ChainOfThoughtOptions) type {
    const use_default = options.rationale_field_type == null;
    const Base = Sig.PrependOutput("reasoning", options.rationale_field_type orelse ?signature.Reasoning);
    const extended_fields = comptime blk: {
        var fields = Base.fields[0..Base.fields.len].*;
        const f = &fields[Sig.input_fields_list.len];
        f.desc = options.rationale_description orelse if (use_default) "A short plan for solving the task." else "${reasoning}";
        if (use_default) {
            f.kind = .str;
            f.type_name = "str";
            f.schema = "";
            f.python_str = null;
            f.xml_schema = "";
            f.validate = null;
            f.xml_validate = null;
            f.optional_str = false;
            f.xml_literal = false;
            f.nullable = false;
            f.reasoning_type = true;
            f.dspy_type = true;
            f.native_reasoning = true;
            f.reasoning_fallback = "plan";
        }
        break :blk fields;
    };
    return struct {
        const Self = @This();
        pub const Extended = struct {
            pub const Input = Sig.Input;
            pub const Output = Base.Output;
            pub const fields: []const Field = &extended_fields;
            pub const input_fields_list = fields[0..Sig.input_fields_list.len];
            pub const output_fields_list = fields[Sig.input_fields_list.len..];
            pub const instructions = Sig.instructions;
            pub const is_dszig_signature = true;
        };
        pub const Input = Sig.Input;
        pub const Output = Extended.Output;

        predict: Predict(Extended) = .{},
        compiled: bool = false,

        pub fn forward(self: *const Self, ctx: Context, input: Input) Error!Output {
            return self.predict.forward(ctx, input);
        }

        /// Source `forward` invokes the inner predictor as a Module call, so
        /// a direct use scopes and attaches usage on that child itself.
        pub fn forwardPrediction(self: *const Self, ctx: Context, input: Input) Error!Prediction(Output) {
            return self.predict.call(ctx, input);
        }

        /// `Module.__call__`: returns the inner predictor's own instance.
        pub fn call(self: *const Self, ctx: Context, input: Input) Error!Prediction(Output) {
            return callProgram(self, ctx, input);
        }
    };
}

fn ProgramOf(comptime Pointer: type) type {
    return switch (@typeInfo(Pointer)) {
        .pointer => |pointer| pointer.child,
        else => @compileError("dszig: a module call takes a program pointer, found " ++ @typeName(Pointer)),
    };
}

fn InputOf(comptime Pointer: type) type {
    const P = ProgramOf(Pointer);
    if (!@hasDecl(P, "Input")) @compileError("dszig: program " ++ @typeName(P) ++ " declares no Input");
    return P.Input;
}

/// Return type of the invoked method: `forwardPrediction` when a Module call
/// asks for it and the program has one, the typed `forward` otherwise.
fn Forwarded(comptime Pointer: type, comptime prediction: bool) type {
    const P = ProgramOf(Pointer);
    const program: Pointer = undefined;
    const ctx: Context = undefined;
    const input: P.Input = undefined;
    if (prediction and @hasDecl(P, "forwardPrediction")) return @TypeOf(program.forwardPrediction(ctx, input));
    return @TypeOf(program.forward(ctx, input));
}

fn Payload(comptime Result: type) type {
    return switch (@typeInfo(Result)) {
        .error_union => |info| info.payload,
        else => Result,
    };
}

fn Failure(comptime Result: type) type {
    return switch (@typeInfo(Result)) {
        .error_union => |info| info.error_set,
        else => error{},
    };
}

fn invoke(
    comptime prediction: bool,
    program: anytype,
    ctx: Context,
    input: InputOf(@TypeOf(program)),
) Failure(Forwarded(@TypeOf(program), prediction))!Payload(Forwarded(@TypeOf(program), prediction)) {
    if (comptime prediction and @hasDecl(ProgramOf(@TypeOf(program)), "forwardPrediction")) return program.forwardPrediction(ctx, input);
    return program.forward(ctx, input);
}

/// One automatic Module-call scope. It has no parent: only an explicit
/// `UsageScope` rolls up. Context is passed by value, so its private tracker
/// never changes the caller's slot and needs no restoration or rollup.
/// Retained tracker references stay valid until Storage reset. Totals are
/// snapshotted into ctx.arena, and only after a successful program call.
fn invokeScoped(
    comptime prediction: bool,
    program: anytype,
    ctx: Context,
    input: InputOf(@TypeOf(program)),
    tracking: UsageTracking,
    usage: ?*Graph.Value,
) (Failure(Forwarded(@TypeOf(program), prediction)) || usage_mod.Error)!Payload(Forwarded(@TypeOf(program), prediction)) {
    assert(ctx.usage_tracker == null);
    const tracker = try usage_mod.UsageTracker.create(tracking.storage, tracking.limits);
    var scoped = ctx;
    scoped.usage_tracker = tracker;
    const output = try invoke(prediction, program, scoped, input);
    // Source computes totals even when the returned value cannot carry them.
    // Skipping this would suppress invalid-merge and resource failures.
    const totals = try tracker.getTotalTokens(ctx.arena);
    if (usage) |target| target.* = totals;
    return output;
}

/// Reflected field structs become Predictions. Native value models keep their
/// own protocols, and scalars, slices and maps stay bare, like the Source
/// outputs that cannot carry usage.
fn isFields(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct" or info.@"struct".is_tuple or prediction_mod.isPrediction(T)) return false;
    if (T == @import("runtime_value.zig").ObjectMap or T == signature.StrDict or @import("types.zig").isDict(T)) return false;
    return !(@hasDecl(T, "toRuntime") or @hasDecl(T, "toValue") or @hasDecl(T, "fromValue"));
}

fn Wrapped(comptime T: type) type {
    if (prediction_mod.isPrediction(T)) return T;
    if (@typeInfo(T) == .optional) return ?Wrapped(@typeInfo(T).optional.child);
    return if (isFields(T)) Prediction(T) else T;
}

/// Result type of `callProgram` for a program output `T`. Only a tuple's
/// first element is transformed; its other elements keep their types.
pub fn CallResult(comptime T: type) type {
    const info = @typeInfo(T);
    if (info != .@"struct" or !info.@"struct".is_tuple or info.@"struct".field_types.len == 0) return Wrapped(T);
    const fields = info.@"struct".field_types;
    if (Wrapped(fields[0]) == fields[0]) return T;
    var types = fields[0..fields.len].*;
    types[0] = Wrapped(fields[0]);
    return @Tuple(&types);
}

fn wrapOne(comptime T: type, arena: Allocator, output: T) Allocator.Error!Wrapped(T) {
    if (comptime Wrapped(T) == T) return output;
    if (comptime @typeInfo(T) == .optional) {
        return if (output) |item| try wrapOne(@typeInfo(T).optional.child, arena, item) else null;
    }
    return Prediction(T).init(arena, output);
}

fn wrap(comptime T: type, arena: Allocator, output: T) Allocator.Error!CallResult(T) {
    const Result = CallResult(T);
    if (comptime Result == T) return output;
    if (comptime Result == Wrapped(T)) return wrapOne(T, arena, output);
    var result: Result = undefined;
    inline for (@typeInfo(T).@"struct".field_types, 0..) |Item, index| {
        result[index] = if (index == 0) try wrapOne(Item, arena, output[0]) else output[index];
    }
    return result;
}

/// Type that Source `_set_lm_usage` inspects: the output or a tuple's first item.
fn UsageTarget(comptime Result: type) type {
    const info = @typeInfo(Result);
    if (info == .@"struct" and info.@"struct".is_tuple and info.@"struct".field_types.len != 0) return info.@"struct".field_types[0];
    return Result;
}

/// A Source Prediction without fields is falsy and never receives usage.
fn carriesUsage(comptime T: type) bool {
    if (@typeInfo(T) == .optional) return carriesUsage(@typeInfo(T).optional.child);
    if (!prediction_mod.isPrediction(T)) return false;
    const info = @typeInfo(T.Fields);
    return info != .@"struct" or info.@"struct".field_types.len != 0;
}

fn attach(comptime T: type, arena: Allocator, target: T, usage: Graph.Value) Allocator.Error!void {
    if (comptime @typeInfo(T) == .optional) {
        if (target) |item| try attach(@typeInfo(T).optional.child, arena, item, usage);
    } else {
        if (comptime T.Fields == @import("runtime_value.zig").ObjectMap) {
            if (target.fields().count() == 0) return;
        }
        target.setLMUsage(try Graph.clone(arena, usage));
    }
}

/// Canonical `Module.__call__` for any native program pointer. A reflected
/// output struct becomes a `Prediction`; a genuine Prediction is returned as
/// the same instance; an optional keeps null; a tuple transforms only its
/// first element; every other output stays bare.
///
/// With `ctx.track_usage` set and no active `ctx.usage_tracker`, the program
/// runs under one automatic scope whose totals are attached to the returned
/// Prediction (or a tuple's first) after success. Any active tracker, however
/// it was configured, suppresses both the scope and the attachment.
pub fn callProgram(
    program: anytype,
    ctx: Context,
    input: InputOf(@TypeOf(program)),
) (Failure(Forwarded(@TypeOf(program), true)) || usage_mod.Error)!CallResult(Payload(Forwarded(@TypeOf(program), true))) {
    const sink = ctx.status_sink orelse return callProgramCore(program, ctx, input);
    sink.notify(.{ .module_start = .{
        .instance = status.View.borrow(ProgramOf(@TypeOf(program)), program),
        .input = status.View.borrow(@TypeOf(input), &input),
    } });
    const result = callProgramCore(program, ctx, input);
    if (result) |output| {
        sink.notify(.{ .module_end = .{ .output = status.View.borrow(@TypeOf(output), &output) } });
    } else |failure| {
        sink.notify(.{ .module_end = .{ .failure = failure } });
    }
    return result;
}

fn callProgramCore(
    program: anytype,
    ctx: Context,
    input: InputOf(@TypeOf(program)),
) (Failure(Forwarded(@TypeOf(program), true)) || usage_mod.Error)!CallResult(Payload(Forwarded(@TypeOf(program), true))) {
    const Output = Payload(Forwarded(@TypeOf(program), true));
    const Result = CallResult(Output);
    const tracking = ctx.track_usage orelse return wrap(Output, ctx.arena, try invoke(true, program, ctx, input));
    if (ctx.usage_tracker != null) return wrap(Output, ctx.arena, try invoke(true, program, ctx, input));
    if (comptime carriesUsage(UsageTarget(Result))) {
        var usage: Graph.Value = undefined;
        const result = try wrap(Output, ctx.arena, try invokeScoped(true, program, ctx, input, tracking, &usage));
        if (comptime UsageTarget(Result) == Result) {
            try attach(Result, ctx.arena, result, usage);
        } else try attach(UsageTarget(Result), ctx.arena, result[0], usage);
        return result;
    } else {
        return wrap(Output, ctx.arena, try invokeScoped(true, program, ctx, input, tracking, null));
    }
}

/// Run with synchronous field routing. The typed result stays available even
/// when `emit_prediction` is false (the stream would omit the final result).
pub fn callStream(
    program: anytype,
    ctx: Context,
    input: InputOf(@TypeOf(program)),
    session: *stream.Session,
    include_prediction: bool,
) (Failure(Forwarded(@TypeOf(program), true)) || usage_mod.Error || stream.Error)!struct {
    prediction: CallResult(Payload(Forwarded(@TypeOf(program), true))),
    emit_prediction: bool,
} {
    if (ctx.stream_session != null) return error.StreamSessionBusy;
    if (session.bindings.len != 0 and ctx.stream_sink != null) return error.ConflictingStreamSinks;
    if (session.active.swap(true, .acquire)) return error.StreamSessionBusy;
    defer session.active.store(false, .release);
    defer session.clear();
    var scoped = ctx;
    scoped.stream_session = session;
    const result = try callProgram(program, scoped, input);
    return .{
        .prediction = result,
        .emit_prediction = try session.finish(include_prediction),
    };
}

/// Module-call scoping around the typed `forward`, for execution entry points
/// that build their own Prediction around reflected output fields
/// (evaluation). Nested Module calls see an active tracker
/// exactly as under `callProgram`.
///
/// `usage` receives what `callProgram` would attach: an owned snapshot of
/// this invocation's totals in ctx.arena after a successful automatic scope,
/// and null when tracking is off, a tracker is already active, or the call
/// fails. A caller passing null declares that its result cannot carry usage;
/// the scope still runs and its totals are still validated.
pub fn forwardScoped(
    program: anytype,
    ctx: Context,
    input: InputOf(@TypeOf(program)),
    usage: ?*?Graph.Value,
) (Failure(Forwarded(@TypeOf(program), false)) || usage_mod.Error)!Payload(Forwarded(@TypeOf(program), false)) {
    if (usage) |target| target.* = null;
    const tracking = ctx.track_usage orelse return program.forward(ctx, input);
    if (ctx.usage_tracker != null) return program.forward(ctx, input);
    var captured: Graph.Value = undefined;
    const output = try invokeScoped(false, program, ctx, input, tracking, if (usage != null) &captured else null);
    if (usage) |target| target.* = try Graph.clone(ctx.arena, captured);
    return output;
}

pub const Retriever = struct {
    ptr: *const anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        search: *const fn (ptr: *const anyopaque, arena: Allocator, query: []const u8, k: i64) anyerror![]const []const u8,
    };

    pub fn search(r: Retriever, arena: Allocator, query: []const u8, k: i64) anyerror![]const []const u8 {
        return r.vtable.search(r.ptr, arena, query, k);
    }
};

/// Deterministic lexical retriever: ranks passages by the number of distinct
/// lowercase query terms they contain; ties keep corpus order.
pub const InMemoryRetriever = struct {
    passages: []const []const u8,

    pub fn retriever(self: *const InMemoryRetriever) Retriever {
        return .{ .ptr = self, .vtable = &.{ .search = search } };
    }

    fn search(ptr: *const anyopaque, arena: Allocator, query: []const u8, k: i64) Allocator.Error![]const []const u8 {
        const self: *const InMemoryRetriever = @ptrCast(@alignCast(ptr));
        const Scored = struct { index: usize, score: usize };
        const scored = try arena.alloc(Scored, self.passages.len);
        for (self.passages, scored, 0..) |p, *s, i| s.* = .{ .index = i, .score = overlap(query, p) };
        std.sort.insertion(Scored, scored, {}, struct {
            fn lessThan(_: void, a: Scored, b: Scored) bool {
                return a.score > b.score or (a.score == b.score and a.index < b.index);
            }
        }.lessThan);
        const n: usize = if (k < 0)
            scored.len - @as(usize, @intCast(@min(@as(u64, scored.len), @abs(k))))
        else
            @intCast(@min(@as(u64, scored.len), @as(u64, @intCast(k))));
        const out = try arena.alloc([]const u8, n);
        for (out, scored[0..n]) |*o, s| o.* = self.passages[s.index];
        return out;
    }

    fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
        if (needle.len > haystack.len) return false;
        var i: usize = 0;
        while (i + needle.len <= haystack.len) : (i += 1) {
            if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
        }
        return false;
    }

    fn overlap(query: []const u8, passage: []const u8) usize {
        var score: usize = 0;
        var terms = std.mem.tokenizeAny(u8, query, " \t\n.,;:!?\"'()");
        var seen: [32][]const u8 = undefined;
        var n_seen: usize = 0;
        outer: while (terms.next()) |term| {
            if (term.len < 3) continue;
            for (seen[0..n_seen]) |s| if (std.ascii.eqlIgnoreCase(s, term)) continue :outer;
            if (n_seen < seen.len) {
                seen[n_seen] = term;
                n_seen += 1;
            }
            if (containsIgnoreCase(passage, term)) score += 1;
        }
        return score;
    }
};

/// `dspy.Retrieve(k)`.
pub const Retrieve = struct {
    k: i64 = 3,

    pub const Input = []const u8;
    pub const Output = []const []const u8;
    pub const Fields = struct { passages: Output };

    pub fn forwardPrediction(self: *const Retrieve, ctx: Context, query: Input) anyerror!Prediction(Fields) {
        return Prediction(Fields).init(ctx.arena, .{ .passages = try self.forward(ctx, query) });
    }

    pub fn call(self: *const Retrieve, ctx: Context, query: Input) anyerror!Prediction(Fields) {
        return callProgram(self, ctx, query);
    }

    pub fn forward(self: *const Retrieve, ctx: Context, query: []const u8) anyerror![]const []const u8 {
        const r = ctx.retriever orelse return error.NoRetriever;
        return r.search(ctx.arena, query, self.k);
    }
};

test "canonical empty record Prediction remains without usage" {
    const P = struct {
        pub const Input = void;
        pub const Output = Prediction(Record);

        pub fn forward(_: *const @This(), ctx: Context, _: Input) Allocator.Error!Output {
            return Output.init(ctx.arena, .empty);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var storage: usage_mod.Storage = .{ .arena = arena, .io = std.testing.io };
    const result = try callProgram(&P{}, .{
        .arena = arena,
        .track_usage = .{ .storage = &storage, .limits = .{ .models = 0, .entries = 0 } },
    }, {});
    try std.testing.expect(result.getLMUsage() == null);
}

pub const Param = union(enum) {
    predict: *PredictState,
    retrieve: *Retrieve,
};

pub const NamedParam = struct { name: []const u8, param: Param };

/// A retry gives the copied module a new LM configuration, without mutating
/// its shared transport or replacing ambient settings for unrelated calls.
pub const PredictRollout = struct {
    parent: ?*const PredictRollout,
    predictors: []const NamedParam,
    lm: lm_mod.LM,
    id: @import("runtime_value.zig").Value,

    pub fn find(first: ?*const PredictRollout, state: *const PredictState) ?*const PredictRollout {
        var scope = first;
        while (scope) |entry| : (scope = entry.parent) {
            for (entry.predictors) |named| {
                if (named.param.predict == state) return entry;
            }
        }
        return null;
    }
};

/// DSPy's Signature.equals compares instructions and named field metadata,
/// not annotation types, field order, or signature class identity.
pub fn sameSignature(a: *const PredictState, b: *const PredictState) bool {
    if (a == b) return true;
    if (a.fields.len != b.fields.len or a.input_count != b.input_count) return false;
    if (!std.mem.eql(u8, a.instructions, b.instructions)) return false;
    if (a.descs.len != a.fields.len or b.descs.len != b.fields.len or
        a.prefixes.len != a.fields.len or b.prefixes.len != b.fields.len) return false;
    for (a.fields, 0..) |field, i| {
        const j = if (a.fields.ptr == b.fields.ptr) i else for (b.fields, 0..) |other, index| {
            if (std.mem.eql(u8, field.name, other.name)) break index;
        } else return false;
        if ((i < a.input_count) != (j < b.input_count)) return false;
        if (!std.mem.eql(u8, a.descs[i], b.descs[j])) return false;
        if (!std.mem.eql(u8, a.prefixes[i], b.prefixes[j])) return false;
    }
    return true;
}

/// `module.named_parameters()`: DSPy names (`self`, `predict`,
/// `generate_answer.predict`, `hops[0]`), skipping compiled sub-modules.
pub fn namedParameters(arena: Allocator, program: anytype) Allocator.Error![]NamedParam {
    const P = @typeInfo(@TypeOf(program)).pointer.child;
    // Wrappers replacing a module's forward method retain that module's
    // parameter namespace instead of exposing their capture/storage fields.
    if (@hasDecl(P, "dszig_parameter_target")) {
        return namedParameters(arena, &@field(program, P.dszig_parameter_target));
    }
    var out: std.ArrayList(NamedParam) = .empty;
    if (@hasDecl(P, "is_dszig_predict")) {
        try out.append(arena, .{ .name = "self", .param = .{ .predict = &program.state } });
    } else {
        try walk(arena, &out, "", program);
    }
    return out.items;
}

/// `module.named_predictors()`.
pub fn namedPredictors(arena: Allocator, program: anytype) Allocator.Error![]NamedParam {
    const all = try namedParameters(arena, program);
    var n: usize = 0;
    for (all) |p| {
        if (p.param == .predict) {
            all[n] = p;
            n += 1;
        }
    }
    return all[0..n];
}

fn walk(arena: Allocator, out: *std.ArrayList(NamedParam), prefix: []const u8, module: anytype) Allocator.Error!void {
    const M = @typeInfo(@TypeOf(module)).pointer.child;
    inline for (@typeInfo(M).@"struct".field_names) |field_name| {
        const name = try joinName(arena, prefix, field_name);
        try visit(arena, out, name, &@field(module, field_name));
    }
}

fn visit(arena: Allocator, out: *std.ArrayList(NamedParam), name: []const u8, ptr: anytype) Allocator.Error!void {
    const T = @typeInfo(@TypeOf(ptr)).pointer.child;
    if (T == Retrieve) return out.append(arena, .{ .name = name, .param = .{ .retrieve = ptr } });
    switch (@typeInfo(T)) {
        .@"struct" => {
            if (@hasDecl(T, "is_dszig_predict")) {
                return out.append(arena, .{ .name = name, .param = .{ .predict = &ptr.state } });
            }
            if (!isModule(T)) return;
            if (@hasDecl(T, "dszig_parameter_target")) {
                const Target = discoveryTargetType(T);
                if (!@hasDecl(Target, "is_dszig_predict") and ptr.compiled) return;
                return visit(arena, out, name, &@field(ptr, T.dszig_parameter_target));
            }
            if (ptr.compiled) return;
            try walk(arena, out, name, ptr);
        },
        .array => for (ptr, 0..) |*item, i| {
            try visit(arena, out, try std.fmt.allocPrint(arena, "{s}[{d}]", .{ name, i }), item);
        },
        else => {},
    }
}

fn discoveryTargetType(comptime T: type) type {
    if (@hasDecl(T, "dszig_parameter_target")) {
        return discoveryTargetType(@FieldType(T, T.dszig_parameter_target));
    }
    return T;
}

fn isModule(comptime T: type) bool {
    return @hasField(T, "compiled") and @hasDecl(T, "forward");
}

fn joinName(arena: Allocator, prefix: []const u8, name: []const u8) Allocator.Error![]const u8 {
    if (prefix.len == 0) return name;
    return std.fmt.allocPrint(arena, "{s}.{s}", .{ prefix, name });
}

/// Reset every reachable predictor (DSPy `reset_copy` minus the copy:
/// programs are values, so `var copy = program;` is the copy).
pub fn resetPredictors(arena: Allocator, program: anytype) Allocator.Error!void {
    for (try namedPredictors(arena, program)) |p| p.param.predict.reset();
}

/// Bind every reachable predictor, preserving the caller-owned LM identity.
pub fn setLM(arena: Allocator, program: anytype, model: ?lm_mod.LM) Allocator.Error!void {
    for (try namedPredictors(arena, program)) |p| p.param.predict.lm = model;
}

/// Empty modules and mixed bound/unbound models have no unique LM.
pub fn getLM(arena: Allocator, program: anytype) (Allocator.Error || error{MultipleLMs})!?lm_mod.LM {
    const predictors = try namedPredictors(arena, program);
    if (predictors.len == 0) return error.MultipleLMs;
    return getLMFromPredictors(predictors);
}

/// Unique binding for a previously enumerated program. An empty traversal
/// carries no binding (some native programs call engines directly).
pub fn getLMFromPredictors(predictors: []const NamedParam) error{MultipleLMs}!?lm_mod.LM {
    if (predictors.len == 0) return null;
    const first = predictors[0].param.predict.lm;
    for (predictors[1..]) |p| {
        const next = p.param.predict.lm;
        if ((first == null) != (next == null)) return error.MultipleLMs;
        if (first) |model| if (!model.sameInstance(next.?)) return error.MultipleLMs;
    }
    return first;
}

const testing = std.testing;

test "Predict cache overrides bypass storage and preserve the previous response" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const input: Sig.Input = .{ .question = "q" };
    const responses = &.{ "<answer>call 1</answer>", "<answer>call 2</answer>", "<answer>call 3</answer>" };

    {
        var engine: lm_mod.ScriptedLM = .{ .responses = responses };
        var cache: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = engine.lm() };
        defer cache.deinit();
        const program: Predict(Sig) = .{};
        var ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = cache.lm() };
        try testing.expectEqualStrings("call 1", (try program.call(ctx, input)).fields().answer);
        ctx.predict_config = &.{ .cache = false };
        try testing.expectEqualStrings("call 2", (try program.forward(ctx, input)).answer);
        ctx.predict_config = null;
        try testing.expectEqualStrings("call 1", (try program.forward(ctx, input)).answer);
        try testing.expectEqual(@as(usize, 2), engine.index);
    }

    {
        var engine: lm_mod.ScriptedLM = .{ .responses = responses };
        var cache: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = engine.lm() };
        defer cache.deinit();
        var program: Predict(Sig) = .{};
        program.state.config = &.{ .cache = false };
        var ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = cache.lm() };
        try testing.expectEqualStrings("call 1", (try program.forward(ctx, input)).answer);
        ctx.predict_config = &.{ .cache = true };
        try testing.expectEqualStrings("call 2", (try program.forward(ctx, input)).answer);
        try testing.expectEqualStrings("call 2", (try program.forward(ctx, input)).answer);
        ctx.predict_config = null;
        try testing.expectEqualStrings("call 3", (try program.forward(ctx, input)).answer);
        ctx.predict_config = &.{ .cache = @as(?bool, null) };
        try testing.expectEqualStrings("call 2", (try program.forward(ctx, input)).answer);
        try testing.expectEqual(@as(usize, 3), engine.index);
    }
}

test "per-call candidate count overrides configured fanout without leaking into later calls" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var model: lm_mod.KeywordLM = .{
        .rules = &.{},
        .default = "<answer>Paris</answer>",
        .history = &history,
    };
    var program: Predict(Sig) = .{};
    program.state.config = &.{ .n = 2, .temperature = 0.1 };
    var ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = model.lm(), .predict_config = &.{ .n = 1 } };
    const input: Sig.Input = .{ .question = "Capital of France?" };
    _ = try program.forwardPrediction(ctx, input);
    try testing.expectEqual(@as(usize, 1), history.requests.items.len);
    try testing.expectEqual(@as(?f64, 0.1), history.requests.items[0].options.temperature);
    ctx.predict_config = null;
    _ = try program.forwardPrediction(ctx, input);
    try testing.expectEqual(@as(usize, 3), history.requests.items.len);
    for (history.requests.items[1..]) |request|
        try testing.expectEqual(@as(?f64, 0.7), request.options.temperature);
}

test "a zero-shot predictor invocation does not discard stored demos" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var model: lm_mod.KeywordLM = .{
        .rules = &.{},
        .default = "<answer>Paris</answer>",
        .history = &history,
    };
    var stored: Record = .empty;
    try stored.put(arena, "question", .{ .string = "A stored question?" });
    try stored.put(arena, "answer", .{ .string = "Berlin" });
    var program: Predict(Sig) = .{};
    program.state.demos = &.{stored};
    const input: Sig.Input = .{ .question = "Capital of France?" };
    const zero_shot = try program.forward(.{ .arena = arena, .lm = model.lm(), .demos_override = &.{} }, input);
    const few_shot = try program.forward(.{ .arena = arena, .lm = model.lm() }, input);
    try testing.expectEqualStrings("Paris", zero_shot.answer);
    try testing.expectEqualStrings("Paris", few_shot.answer);
    for (history.requests.items[0].messages) |message|
        try testing.expect(std.mem.indexOf(u8, message.content.text, "A stored question?") == null);
    try testing.expect(std.mem.indexOf(u8, history.requests.items[1].messages[1].content.text, "A stored question?") != null);
}

test "record prediction renders and traces omitted defaults without replacing explicit null" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { locale: []const u8 = "en-GB" };
        pub const Output = struct { answer: i64 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var model: lm_mod.KeywordLM = .{ .rules = &.{}, .default = "<answer>1</answer>", .history = &history };
    var trace: Trace = .{ .arena = arena };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = model.lm(), .trace = &trace };
    const program: Predict(Sig) = .{};
    var input: Record = .empty;

    _ = try program.state.call(ctx, &input);
    try testing.expect(!input.contains("locale"));
    try testing.expectEqualStrings("en-GB", trace.steps.items[0].inputs.get("locale").?.string);
    const messages = history.requests.items[0].messages;
    try testing.expect(std.mem.indexOf(u8, messages[messages.len - 1].content.text, "<locale>\nen-GB\n</locale>") != null);

    try input.put(arena, "locale", .null);
    _ = try program.state.call(ctx, &input);
    try testing.expect(trace.steps.items[1].inputs.get("locale").? == .null);
}

test "bound LM precedence includes explicit null and fanout and resets without changing copies" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var bound: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>1</answer>", "<answer>4</answer>", "<answer>5</answer>", "<answer>6</answer>" } };
    var fallback: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>2</answer>", "<answer>2</answer>" } };
    var override: lm_mod.ScriptedLM = .{ .responses = &.{"<answer>3</answer>"} };
    var program: Predict(Sig) = .{};
    program.state.lm = bound.lm();
    const input: Sig.Input = .{ .question = "q" };
    try testing.expectEqual(@as(i64, 1), (try program.forward(.{ .arena = arena, .lm = fallback.lm() }, input)).answer);
    try testing.expectEqual(@as(i64, 3), (try program.forward(.{ .arena = arena, .lm_override = override.lm() }, input)).answer);
    try testing.expectEqual(@as(i64, 2), (try program.forward(.{ .arena = arena, .lm = fallback.lm(), .lm_override_present = true }, input)).answer);
    const inputs = try value.toRecord(arena, Sig.Input, input);
    const many = try program.state.callMany(.{ .arena = arena }, &inputs, 2);
    try testing.expectEqual(@as(i64, 4), many[0].get("answer").?.integer);
    try testing.expectEqual(@as(i64, 5), many[1].get("answer").?.integer);
    const copy = program;
    program.state.reset();
    var capture: ParseFailure = .{ .expected_count = 99 };
    try testing.expectError(error.NoLM, program.forward(.{ .arena = arena, .parse_failure = &capture }, input));
    try testing.expectEqual(@as(u32, 0), capture.expected_count);
    try testing.expectEqual(@as(i64, 6), (try copy.forward(.{ .arena = arena }, input)).answer);
    try testing.expectEqual(@as(i64, 2), (try program.forward(.{ .arena = arena, .lm = fallback.lm() }, input)).answer);
}

fn testPromptTokens(arena: Allocator, usage: ?Graph.Value) !i64 {
    const total = try Graph.toValue(arena, usage.?);
    return total.object.get("configured").?.object.get("prompt_tokens").?.integer;
}

test "module calls scope usage onto the returned prediction instance unless a tracker is active" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    const Child = Predict(Sig);
    const Pair = struct {
        child: Child = .{},
        seen: *?Prediction(Sig.Output),
        compiled: bool = false,

        pub const Input = Sig.Input;
        pub const Output = struct { Prediction(Sig.Output), []const u8 };

        pub fn forward(self: *const @This(), ctx: Context, input: Input) Error!Output {
            const first = try self.child.call(ctx, input);
            self.seen.* = first;
            return .{ first, "extra" };
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const one: lm_mod.ResponseMetadata = .{ .usage = .{ .input_tokens = .{ .integer = 1 } } };
    var engine: lm_mod.ScriptedLM = .{
        .responses = &.{ "<answer>1</answer>", "<answer>2</answer>", "<answer>3</answer>", "<answer>4</answer>", "<answer>5</answer>", "<answer>6</answer>" },
        .model = "configured",
        .metadata = &.{ one, one, one, one, one, one },
    };
    var storage: usage_mod.Storage = .{ .arena = arena, .io = testing.io };
    const plain: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = engine.lm() };
    var tracked = plain;
    tracked.track_usage = .{ .storage = &storage, .limits = .{ .models = 1, .entries = 2 } };
    const input: Sig.Input = .{ .question = "q" };

    // The outer call's totals land on the returned child instance, so a copy
    // taken before the tuple was returned observes them too.
    var seen: ?Prediction(Sig.Output) = null;
    const pair: Pair = .{ .seen = &seen };
    const tuple = try callProgram(&pair, tracked, input);
    try testing.expect(tuple[0].fields() == seen.?.fields());
    try testing.expectEqual(@as(i64, 1), tuple[0].fields().answer);
    try testing.expectEqualStrings("extra", tuple[1]);
    try testing.expectEqual(@as(i64, 1), try testPromptTokens(arena, seen.?.getLMUsage()));
    const retained = seen.?;

    // A direct forward is no Module call, but its child call scopes itself.
    const direct = try pair.forward(tracked, input);
    try testing.expectEqual(@as(i64, 1), try testPromptTokens(arena, direct[0].getLMUsage()));
    try testing.expect(direct[0].fields() != retained.fields());

    // Any active tracker suppresses the automatic scope and the attachment.
    var tracker: usage_mod.UsageTracker = undefined;
    try tracker.init(&storage, .{ .models = 1, .entries = 1 });
    var explicit = tracked;
    explicit.usage_tracker = &tracker;
    const suppressed = try pair.child.call(explicit, input);
    try testing.expect(suppressed.getLMUsage() == null);
    try testing.expectEqual(@as(i64, 1), try testPromptTokens(arena, try tracker.getTotalTokens(arena)));
    try testing.expect((try pair.child.call(plain, input)).getLMUsage() == null);

    // One public call keeps every completion and one combined usage entry.
    var many = tracked;
    many.lm_options.n = 2;
    const group = try pair.child.call(many, input);
    try testing.expectEqual(@as(usize, 2), group.completions().?.len);
    try testing.expectEqual(@as(i64, 5), group.fields().answer);
    try testing.expectEqual(@as(i64, 6), group.completions().?[1].answer);
    try testing.expectEqual(@as(i64, 2), try testPromptTokens(arena, group.getLMUsage()));
    try testing.expectEqual(@as(usize, 6), engine.index);

    // A failed call returns no prediction, and earlier ones keep their own state.
    try testing.expectError(error.ScriptExhausted, pair.child.call(tracked, input));
    try testing.expectEqual(@as(i64, 1), try testPromptTokens(arena, retained.getLMUsage()));
    const copy = try group.clone(arena);
    copy.fields().answer = 9;
    copy.setLMUsage(null);
    try testing.expectEqual(@as(usize, 2), copy.completions().?.len);
    try testing.expectEqual(@as(i64, 5), group.fields().answer);
    try testing.expectEqual(@as(i64, 5), copy.completions().?[0].answer);
    try testing.expectEqual(@as(i64, 2), try testPromptTokens(arena, group.getLMUsage()));
}

test "automatic scope propagates invalid totals even when its result cannot carry usage" {
    const Program = struct {
        pub const Input = struct {};
        pub const Output = i64;
        pub fn forward(_: *const @This(), ctx: Context, _: Input) anyerror!Output {
            if (ctx.usage_tracker) |tracker| {
                try tracker.addUsage("manual", try Graph.parse(ctx.arena, "{\"field\":{\"count\":1}}"));
                try tracker.addUsage("manual", try Graph.parse(ctx.arena, "{\"field\":false}"));
            }
            return 17;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var storage: usage_mod.Storage = .{ .arena = arena, .io = testing.io };
    const ctx: Context = .{
        .arena = arena,
        .track_usage = .{ .storage = &storage, .limits = .{ .models = 1, .entries = 2 } },
    };
    const program: Program = .{};
    try testing.expectError(error.ParseValue, callProgram(&program, ctx, .{}));
}

test "module LM aggregation distinguishes shared identities mixed bindings and no predictors" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var first: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var second: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var pair: struct { a: Predict(Sig) = .{}, b: Predict(Sig) = .{}, retrieve: Retrieve = .{} } = .{};
    try testing.expect((try getLM(arena, &pair)) == null);
    try setLM(arena, &pair, first.lm());
    try testing.expect((try getLM(arena, &pair)).?.sameInstance(first.lm()));
    pair.b.state.lm = second.lm();
    try testing.expectError(error.MultipleLMs, getLM(arena, &pair));
    pair.b.state.lm = null;
    try testing.expectError(error.MultipleLMs, getLM(arena, &pair));
    try resetPredictors(arena, &pair);
    try testing.expect((try getLM(arena, &pair)) == null);
    var empty: struct {} = .{};
    try testing.expectError(error.MultipleLMs, getLM(arena, &empty));
}

test "TwoStepAdapter isolates extraction options and retries only the extraction stage" {
    const CountSignature = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var main_lm: lm_mod.ScriptedLM = .{ .responses = &.{"There are five."} };
    var extraction_history: lm_mod.History = .{ .arena = arena };
    var extraction_lm: lm_mod.ScriptedLM = .{
        .responses = &.{ "not structured", "{\"answer\":5}" },
        .history = &extraction_history,
    };
    var trace: Trace = .{ .arena = arena };
    const two_step: TwoStepAdapter = .{
        .extraction_model = extraction_lm.lm(),
        .extraction_options = .{ .temperature = 0.125 },
    };
    const ctx = two_step.context(.{
        .arena = arena,
        .lm = main_lm.lm(),
        .trace = &trace,
        .lm_options = .{ .temperature = 0.9, .rollout_id = .{ .integer = 42 } },
    });
    const program: Predict(CountSignature) = .{};
    const result = try program.forward(ctx, .{ .question = "How many?" });
    try testing.expectEqual(@as(i64, 5), result.answer);
    try testing.expectEqual(@as(usize, 1), main_lm.index);
    try testing.expectEqual(@as(usize, 2), extraction_lm.index);
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    for (extraction_history.requests.items) |request| {
        try testing.expectEqual(@as(?f64, 0.125), request.options.temperature);
        try testing.expect(request.options.rollout_id == null);
        try testing.expectEqual(@as(u32, 1), request.options.n);
    }
}

test "TwoStepAdapter text output collision preserves original output order" {
    const CollisionSignature = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8, text: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var main_lm: lm_mod.ScriptedLM = .{ .responses = &.{"freeform"} };
    var extraction_lm: lm_mod.ScriptedLM = .{
        .responses = &.{"[[ ## text ## ]]\nverbatim\n\n[[ ## answer ## ]]\nyes"},
    };
    const two_step: TwoStepAdapter = .{ .extraction_model = extraction_lm.lm() };
    var trace: Trace = .{ .arena = arena };
    const program: Predict(CollisionSignature) = .{};
    const result = try program.forward(two_step.context(.{
        .arena = arena,
        .lm = main_lm.lm(),
        .trace = &trace,
    }), .{ .question = "q" });
    try testing.expectEqualStrings("yes", result.answer);
    try testing.expectEqualStrings("verbatim", result.text);
    try testing.expectEqualStrings("answer", trace.steps.items[0].outputs.keys()[0]);
    try testing.expectEqualStrings("text", trace.steps.items[0].outputs.keys()[1]);
}

test "TwoStepAdapter parse failure never retries the main LM" {
    const FailureSignature = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var main_lm: lm_mod.ScriptedLM = .{ .responses = &.{ "main", "must not run" } };
    var extraction_lm: lm_mod.ScriptedLM = .{ .responses = &.{ "bad chat", "bad json" } };
    const two_step: TwoStepAdapter = .{ .extraction_model = extraction_lm.lm() };
    var trace: Trace = .{ .arena = arena };
    const ctx = two_step.context(.{ .arena = arena, .lm = main_lm.lm(), .trace = &trace });
    const program: Predict(FailureSignature) = .{};
    try testing.expectError(error.AdapterParse, program.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(usize, 1), main_lm.index);
    try testing.expectEqual(@as(usize, 2), extraction_lm.index);
    try testing.expectEqual(@as(usize, 0), trace.steps.items.len);
}

const QA = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

const Inner = struct {
    generate_answer: ChainOfThought(QA) = .{},
    compiled: bool = false,
    pub const Input = QA.Input;
    pub fn forward(_: *const @This(), _: Context, _: Input) Error!void {}
};

const Outer = struct {
    retrieve: Retrieve = .{},
    hops: [2]Predict(QA) = .{ .{}, .{} },
    inner: Inner = .{},
    frozen: Inner = .{ .compiled = true },
    compiled: bool = false,
};

test "named parameters follow DSPy naming and freeze rules" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var program: Outer = .{};
    const params = try namedParameters(arena.allocator(), &program);
    const expected = [_][]const u8{ "retrieve", "hops[0]", "hops[1]", "inner.generate_answer.predict" };
    try testing.expectEqual(expected.len, params.len);
    for (expected, params) |e, p| try testing.expectEqualStrings(e, p.name);
    var p: Predict(QA) = .{};
    try testing.expectEqualStrings("self", (try namedParameters(arena.allocator(), &p))[0].name);
}

test "predict and chain of thought run through the adapter with trace" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{
        "<plan>\nParis is the capital.\n</plan>\n\n<answer>\nParis\n</answer>",
        "not parseable",
        "{\"plan\": \"r\", \"answer\": \"Lyon\"}",
    } };
    var trace: Trace = .{ .arena = arena.allocator() };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena.allocator(), .lm = script.lm(), .trace = &trace };
    const cot: ChainOfThought(QA) = .{};
    const out = try cot.forward(ctx, .{ .question = "Capital of France?" });
    try testing.expectEqualStrings("Paris", out.answer);
    try testing.expectEqualStrings("Paris is the capital.", out.reasoning.?.content);
    const fallback = try cot.forward(ctx, .{ .question = "x" });
    try testing.expectEqualStrings("Lyon", fallback.answer);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].predictor == &cot.predict.state);
}

test "Predict preserves explicit tool policy without a ToolCalls output" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm_mod.History = .{ .arena = arena };
    var model: lm_mod.KeywordLM = .{
        .rules = &.{},
        .default = "<answer>ok</answer>",
        .history = &history,
    };
    const program: Predict(Sig) = .{};
    var ctx: Context = .{
        .arena = arena,
        .lm = model.lm(),
        .json_fallback = false,
        .use_native_function_calling = true,
        .lm_options = .{ .tool_choice = .none, .parallel_tool_calls = false },
    };
    _ = try program.forward(ctx, .{ .question = "q" });
    const enabled = history.requests.items[0].options;
    try testing.expectEqual(@as(?lm_mod.ToolChoice, .none), enabled.tool_choice);
    try testing.expectEqual(@as(?bool, false), enabled.parallel_tool_calls);

    ctx.use_native_function_calling = false;
    _ = try program.forward(ctx, .{ .question = "q" });
    const disabled = history.requests.items[1].options;
    try testing.expectEqual(null, disabled.tool_choice);
    try testing.expectEqual(null, disabled.parallel_tool_calls);
}

test "Predict consumes returned tool calls independently of request negotiation" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 = 17, calls: tool_mod.ToolCalls };
    });
    const program: Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const returned: []const inference.ToolCall = &.{.{ .id = "call-native", .name = "search", .arguments = "{\"query\":\"q\"}" }};
    for ([_]struct { Adapter, []const u8, i64 }{
        .{ .chat, "", 17 },
        .{ .xml, "<answer>5</answer>", 17 },
        .{ .json, "{\"answer\":5,\"calls\":null}", 5 },
    }) |case| {
        var script: lm_mod.ScriptedLM = .{ .responses = &.{case[1]}, .tool_calls = &.{returned} };
        var ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm(), .adapter = case[0], .use_native_function_calling = true };
        const input: Sig.Input = .{ .question = "q" };
        try testing.expectError(error.MissingToolInput, program.forward(ctx, input));
        try testing.expectEqual(0, script.index);

        ctx.use_native_function_calling = false;
        const view_ = try program.state.view(arena);
        if (case[0] == .json) {
            const text_only = try case[0].parse(arena, view_, case[1]);
            try testing.expectEqual(0, (try tool_mod.ToolCalls.fromValue(arena, text_only.get("calls").?)).tool_calls.len);
        } else try testing.expectError(error.AdapterParse, case[0].parse(arena, view_, case[1]));

        const result = try program.forward(ctx, input);
        try testing.expectEqual(case[2], result.answer);
        try testing.expectEqual(1, result.calls.tool_calls.len);
        const call = result.calls.tool_calls[0];
        try testing.expectEqualStrings("call-native", call.id.?);
        try testing.expectEqualStrings("search", call.name);
        try testing.expectEqualStrings("q", call.args.get("query").?.string);
        try testing.expectEqual(1, script.index);
    }
}

test "native negotiation leaves a second ToolCalls output in the text signature" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { tools: []const []const u8 };
        pub const Output = struct { calls: tool_mod.ToolCalls, spare: tool_mod.ToolCalls };
    });
    const fields = comptime blk: {
        var f = Sig.fields[0..Sig.fields.len].*;
        f[0].native_tools = true;
        break :blk f;
    };
    const program: PredictFields(&fields, 1, "Choose calls.") = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const returned: []const inference.ToolCall = &.{.{ .id = "id", .name = "search", .arguments = "{}" }};
    var script: lm_mod.ScriptedLM = .{
        .responses = &.{ "", "<spare>{\"tool_calls\":[]}</spare>" },
        .tool_calls = &.{ returned, returned },
        .supports_function_calling = true,
    };
    const inputs = try value.toRecord(arena, Sig.Input, .{ .tools = &.{} });
    const rows = try program.state.callMany(.{ .arena = arena, .lm = script.lm() }, &inputs, 2);
    try testing.expectEqual(2, rows.len);
    for (rows) |row| {
        const calls = try tool_mod.ToolCalls.fromValue(arena, row.get("calls").?);
        try testing.expectEqual(1, calls.tool_calls.len);
        try testing.expectEqualStrings("id", calls.tool_calls[0].id.?);
    }
    try testing.expect(rows[0].get("spare").? == .null);
    try testing.expect(rows[1].get("spare").? == .object);
    try testing.expectEqual(0, (try tool_mod.ToolCalls.fromValue(arena, rows[1].get("spare").?)).tool_calls.len);
}

test "Predict consumes the per-call trace control only after inference" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>A</answer>",
        "<answer>B</answer>",
        "<answer>C</answer>",
        "<answer>D</answer>",
    } };
    var trace: Trace = .{ .arena = arena };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm(), .trace = &trace };
    const program: Predict(QA) = .{};
    var input: Record = .empty;
    try input.put(arena, "question", .{ .string = "q" });
    for ([_]?@import("runtime_value.zig").Value{ null, .{ .bool = false }, .{ .string = "" }, .{ .integer = 1 } }, "ABCD", [_]usize{ 1, 1, 1, 2 }) |control, answer, retained| {
        if (control) |flag| try input.put(arena, "_trace", flag);
        const result = try program.state.call(ctx, &input);
        try testing.expectEqualStrings(&.{answer}, result.get("answer").?.string);
        try testing.expectEqual(retained, trace.steps.items.len);
        try testing.expectEqual(control != null, input.contains("_trace"));
    }
    try testing.expectEqual(@as(usize, 4), script.index);
    try testing.expectEqualStrings("D", trace.steps.items[1].outputs.get("answer").?.string);
    try testing.expect(!trace.steps.items[1].inputs.contains("_trace"));
    try testing.expectEqualStrings("q", trace.steps.items[1].inputs.get("question").?.string);
}

test "Predict evicts one oldest trace step only after a successful parse" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>A</answer>",
        "<answer>B</answer>",
        "<answer>C</answer>",
        "<answer>D</answer>",
        "<answer>E</answer>",
        "<answer>F</answer>",
        "malformed",
    } };
    var trace: Trace = .{ .arena = arena };
    var ctx: Context = .{
        .arena = arena,
        .lm = script.lm(),
        .trace = &trace,
        .json_fallback = false,
    };
    const program: Predict(QA) = .{};
    for ([_]i64{ 2, 2, 2, 1, 0, -1 }, "ABCDEF", [_][]const u8{
        "A", "AB", "BC", "CD", "CD", "CD",
    }) |limit, answer, retained| {
        ctx.max_trace_size = limit;
        const result = try program.forward(ctx, .{ .question = "question" });
        try testing.expectEqualStrings(&.{answer}, result.answer);
        try testing.expectEqual(retained.len, trace.steps.items.len);
        for (trace.steps.items, retained) |step, expected|
            try testing.expectEqualStrings(&.{expected}, step.outputs.get("answer").?.string);
    }
    ctx.max_trace_size = 1;
    try testing.expectError(error.AdapterParse, program.forward(ctx, .{ .question = "question" }));
    try testing.expectEqual(@as(usize, 7), script.index);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expectEqualStrings("C", trace.steps.items[0].outputs.get("answer").?.string);
    try testing.expectEqualStrings("D", trace.steps.items[1].outputs.get("answer").?.string);
}

test "typed Predict retains wide integer inputs and outputs across adapters" {
    const Wide = signature.Signature(struct {
        pub const Input = struct { n: u64, nested: struct { id: u64 } };
        pub const Output = struct { answer: u64 };
    });
    const InspectingLM = struct {
        response: []const u8,

        fn complete(ptr: *anyopaque, a: Allocator, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const text = request.messages[request.messages.len - 1].content.text;
            if (std.mem.indexOf(u8, text, "10000000000000000000") == null or
                std.mem.indexOf(u8, text, "18446744073709551615") == null or
                std.mem.indexOf(u8, text, "\"18446744073709551615\"") != null or
                std.mem.indexOf(u8, text, "\"10000000000000000000\"") != null) return error.LMFailure;
            const out = try a.alloc(lm_mod.Completion, 1);
            out[0] = .{ .text = self.response };
            return out;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    inline for (.{
        .{ Adapter.chat, "[[ ## answer ## ]]\n18446744073709551615\n\n[[ ## completed ## ]]" },
        .{ Adapter.json, "{\"answer\":18446744073709551615}" },
        .{ Adapter.xml, "<answer>18446744073709551615</answer>" },
    }) |case| {
        var model: InspectingLM = .{ .response = case[1] };
        const predict: Predict(Wide) = .{};
        const out = try predict.forward(.{
            .arena = memory.allocator(),
            .lm = .{ .ptr = &model, .vtable = &.{ .complete = InspectingLM.complete } },
            .adapter = case[0],
            .json_fallback = false,
        }, .{ .n = 10_000_000_000_000_000_000, .nested = .{ .id = std.math.maxInt(u64) } });
        try testing.expectEqual(std.math.maxInt(u64), out.answer);
    }
}

test "typed Predict preserves custom nullable Reasoning null and empty content" {
    const Custom = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { analysis: ?signature.Reasoning, answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    // Optional custom Reasoning remains a text output even on a reasoning LM:
    // upstream native adaptation only handles a direct Reasoning annotation.
    inline for (.{ false, true }) |supports_native| {
        inline for (.{ false, true }) |disabled| {
            inline for (.{
                .{ Adapter.json, "{\"analysis\":null,\"answer\":\"ok\"}", @as(?[]const u8, null) },
                .{ Adapter.json, "{\"analysis\":\"\",\"answer\":\"ok\"}", @as(?[]const u8, "") },
                .{ Adapter.json, "{\"analysis\":{\"content\":\"why\"},\"answer\":\"ok\"}", @as(?[]const u8, "why") },
                .{ Adapter.xml, "<analysis /><answer>ok</answer>", @as(?[]const u8, null) },
                .{ Adapter.xml, "<analysis></analysis><answer>ok</answer>", @as(?[]const u8, "") },
                .{ Adapter.xml, "<analysis><content>why</content></analysis><answer>ok</answer>", @as(?[]const u8, "why") },
            }) |case| {
                var script: lm_mod.ScriptedLM = .{ .responses = &.{case[1]} };
                var lm = script.lm();
                lm.supports_reasoning = supports_native;
                var trace: Trace = .{ .arena = memory.allocator() };
                const predict: Predict(Custom) = .{};
                const out = try predict.forward(.{
                    .arena = memory.allocator(),
                    .lm = lm,
                    .adapter = case[0],
                    .json_fallback = false,
                    .lm_options = if (disabled) .{ .reasoning_effort = "none" } else .{},
                    .trace = &trace,
                }, .{ .question = "Explain?" });
                try testing.expectEqualStrings("ok", out.answer);
                if (case[2]) |expected| {
                    try testing.expectEqualStrings(expected, out.analysis.?.content);
                    try testing.expectEqualStrings(expected, (try signature.Reasoning.fromValue(trace.steps.items[0].outputs.get("analysis").?)).content);
                } else {
                    try testing.expect(out.analysis == null);
                    try testing.expect(trace.steps.items[0].outputs.get("analysis").? == .null);
                }
            }
        }
    }
}

test "default CoT fallback validates required Reasoning without changing native null injection" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    inline for (.{
        .{ Adapter.json, "{\"plan\":null,\"answer\":\"ok\"}", @as(?[]const u8, null) },
        .{ Adapter.json, "{\"plan\":{},\"answer\":\"ok\"}", @as(?[]const u8, null) },
        .{ Adapter.json, "{\"plan\":7,\"answer\":\"ok\"}", @as(?[]const u8, null) },
        .{ Adapter.json, "{\"plan\":\"\",\"answer\":\"ok\"}", @as(?[]const u8, "") },
        .{ Adapter.json, "{\"plan\":{\"content\":\"why\"},\"answer\":\"ok\"}", @as(?[]const u8, "why") },
        .{ Adapter.xml, "<plan /><answer>ok</answer>", @as(?[]const u8, null) },
        .{ Adapter.xml, "<plan></plan><answer>ok</answer>", @as(?[]const u8, null) },
        .{ Adapter.xml, "<plan><content></content></plan><answer>ok</answer>", @as(?[]const u8, "") },
    }) |case| {
        var script: lm_mod.ScriptedLM = .{ .responses = &.{case[1]} };
        const cot: ChainOfThought(QA) = .{};
        const ctx: Context = .{
            .arena = memory.allocator(),
            .lm = script.lm(),
            .adapter = case[0],
            .json_fallback = false,
        };
        if (case[2]) |expected| {
            const out = try cot.forward(ctx, .{ .question = "Explain?" });
            try testing.expectEqualStrings(expected, out.reasoning.?.content);
            try testing.expectEqualStrings("ok", out.answer);
        } else {
            try testing.expectError(error.AdapterParse, cot.forward(ctx, .{ .question = "Explain?" }));
        }
    }
}

test "default XML CoT distinguishes off from none for native reasoning adaptation" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const cot: ChainOfThought(QA) = .{};

    const Case = struct {
        options: lm_mod.Options,
        wire_effort: ?[]const u8,
        wants_plan: bool,
    };
    for ([_]Case{
        .{ .options = .{ .reasoning_effort = "off" }, .wire_effort = "off", .wants_plan = false },
        .{ .options = .{ .reasoning_effort = "none" }, .wire_effort = "none", .wants_plan = true },
        .{ .options = .{ .reasoning_effort_present = true }, .wire_effort = null, .wants_plan = true },
        .{ .options = .{}, .wire_effort = "low", .wants_plan = false },
    }) |case| {
        var history: lm_mod.History = .{ .arena = arena };
        var model: lm_mod.KeywordLM = .{
            .rules = &.{.{ .needle = "<plan>", .response = "<plan>fallback</plan><answer>ok</answer>" }},
            .default = "<answer>ok</answer>",
            .history = &history,
        };
        var lm = model.lm();
        lm.supports_reasoning = true;
        const result = try cot.forward(.{ .arena = arena, .lm = lm, .lm_options = case.options }, .{ .question = "Explain?" });

        try testing.expectEqual(@as(usize, 1), history.requests.items.len);
        const request = history.requests.items[0];
        const user_text = request.messages[request.messages.len - 1].content.text;
        try testing.expectEqual(case.wants_plan, std.mem.indexOf(u8, user_text, "<plan>") != null);
        try testing.expectEqual(case.wire_effort != null, request.options.reasoning_effort != null);
        if (case.wire_effort) |effort| try testing.expectEqualStrings(effort, request.options.reasoning_effort.?);
        try testing.expectEqualStrings("ok", result.answer);
        if (case.wants_plan) {
            try testing.expectEqualStrings("fallback", result.reasoning.?.content);
        } else {
            try testing.expectEqual(@as(?signature.Reasoning, null), result.reasoning);
        }
    }
}

test "native reasoning preserves absent summaries and records actual content" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const Native = struct {
        summary: ?[]const u8 = null,

        fn complete(ptr: *anyopaque, a: Allocator, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (request.options.reasoning_effort == null or
                !std.mem.eql(u8, request.options.reasoning_effort.?, "low")) return error.LMFailure;
            // Native reasoning must never be requested as a text output as well.
            if (std.mem.indexOf(u8, request.messages[0].content.text, "`reasoning`") != null or
                std.mem.indexOf(u8, request.messages[0].content.text, "`plan`") != null) return error.LMFailure;
            const out = try a.alloc(lm_mod.Completion, 1);
            out[0] = .{ .text = "<answer>\nParis\n</answer>", .reasoning = self.summary };
            return out;
        }
    };
    var native: Native = .{};
    var trace: Trace = .{ .arena = arena.allocator() };
    const ctx: Context = .{
        .arena = arena.allocator(),
        .lm = .{ .ptr = &native, .vtable = &.{ .complete = Native.complete }, .supports_reasoning = true },
        .trace = &trace,
    };
    const cot: ChainOfThought(QA) = .{};
    const without = try cot.forward(ctx, .{ .question = "Capital?" });
    try testing.expectEqualStrings("Paris", without.answer);
    try testing.expectEqual(@as(?signature.Reasoning, null), without.reasoning);
    try testing.expect(trace.steps.items[0].outputs.get("reasoning").? == .null);
    native.summary = "Paris is France's capital.";
    const with = try cot.forward(ctx, .{ .question = "Capital?" });
    try testing.expectEqualStrings(native.summary.?, with.reasoning.?.content);
    try testing.expectEqualStrings(native.summary.?, (try signature.Reasoning.fromValue(trace.steps.items[1].outputs.get("reasoning").?)).content);
    // DSPy appends natively returned reasoning after the parsed fields.
    try testing.expectEqualStrings("answer", trace.steps.items[1].outputs.keys()[0]);
    try testing.expectEqualStrings("reasoning", trace.steps.items[1].outputs.keys()[1]);
    native.summary = "";
    const empty = try cot.forward(ctx, .{ .question = "Capital?" });
    try testing.expectEqual(@as(?signature.Reasoning, null), empty.reasoning);
    try testing.expect(trace.steps.items[2].outputs.get("reasoning").? == .null);
}

test "CoT plan fallback avoids task field collisions and honors disabled native reasoning" {
    const WithPlan = signature.Signature(struct {
        pub const Input = struct { question: []const u8, plan: []const u8 };
        pub const Output = struct { answer: []const u8, plan_2: []const u8 };
    });
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{
        "<plan_3>\nCheck the facts.\n</plan_3>\n\n<answer>\nParis\n</answer>\n\n<plan_2>\nDone.\n</plan_2>",
    } };
    var lm = script.lm();
    lm.supports_reasoning = true;
    const cot: ChainOfThought(WithPlan) = .{};
    const out = try cot.forward(.{
        .arena = arena.allocator(),
        .lm = lm,
        .lm_options = .{ .reasoning_effort = "none" },
    }, .{ .question = "Capital?", .plan = "Use evidence." });
    try testing.expectEqualStrings("Check the facts.", out.reasoning.?.content);
    try testing.expectEqualStrings("Done.", out.plan_2);
    try testing.expectEqualStrings("Paris", out.answer);
}

test "CoT keeps prior rationale in History when the fallback name collides" {
    const History = @import("history.zig").History;
    const Conversation = signature.Signature(struct {
        pub const Input = struct { question: []const u8, plan: []const u8, history: History };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var event: Record = .empty;
    try event.put(arena, "question", .{ .string = "old question" });
    try event.put(arena, "plan", .{ .string = "old task plan" });
    try event.put(arena, "reasoning", .{ .string = "PRIOR-731" });
    try event.put(arena, "answer", .{ .string = "old answer" });
    var requests: lm_mod.History = .{ .arena = arena };
    var model: lm_mod.KeywordLM = .{
        .rules = &.{.{
            .needle = "<plan_2>\nPRIOR-731\n</plan_2>",
            .response = "<plan_2>\nnew rationale\n</plan_2>\n<answer>\nsaw-prior\n</answer>",
            .scope = .any,
        }},
        .default = "<plan_2>\nnew rationale\n</plan_2>\n<answer>\nlost-prior\n</answer>",
        .history = &requests,
    };
    const cot: ChainOfThought(Conversation) = .{};
    const out = try cot.forward(.{ .arena = arena, .lm = model.lm(), .json_fallback = false }, .{
        .question = "new question",
        .plan = "new task plan",
        .history = .{ .messages = &.{event} },
    });
    try testing.expectEqualStrings("saw-prior", out.answer);
    try testing.expectEqual(@as(usize, 1), requests.requests.items.len);
    try testing.expectEqualStrings("PRIOR-731", event.get("reasoning").?.string);
    try testing.expect(!event.contains("plan_2"));
}

test "signature equality compares named roles and metadata rather than order or annotations" {
    const A = signature.Signature(struct {
        pub const instructions = "identity";
        pub const Input = struct { question: []const u8, count: i64 };
        pub const Output = struct { answer: []const u8 };
    });
    const B = signature.Signature(struct {
        pub const instructions = "identity";
        pub const Input = struct { count: []const u8, question: i64 };
        pub const Output = struct { answer: f64 };
    });
    const C = signature.Signature(struct {
        pub const instructions = "identity";
        pub const Input = struct { question: []const u8, answer: []const u8 };
        pub const Output = struct { count: i64 };
    });
    const a: Predict(A) = .{};
    var b: Predict(B) = .{};
    const c: Predict(C) = .{};
    try testing.expect(sameSignature(&a.state, &b.state));
    try testing.expect(sameSignature(&b.state, &a.state));
    try testing.expect(!sameSignature(&a.state, &c.state));
    b.state.instructions = "changed";
    try testing.expect(!sameSignature(&a.state, &b.state));
    b.state.instructions = "identity";
    b.state.descs = &.{ "changed", b.state.descs[1], b.state.descs[2] };
    try testing.expect(!sameSignature(&a.state, &b.state));
}

test "published fields prevent retrying a broken stream with JSON" {
    const S = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8, count: i64 };
    });
    const Model = struct {
        body: []const u8,
        calls: usize = 0,
        published: usize = 0,

        fn complete(_: *anyopaque, _: Allocator, _: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
            return error.LMFailure;
        }

        fn completeStream(ptr: *anyopaque, _: Allocator, _: lm_mod.Request, sink: lm_mod.OpenAI.StreamSink) lm_mod.Error!lm_mod.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            const text = if (self.calls == 1) self.body else "{\"answer\":\"recovered\",\"count\":7}";
            if (sink.on_event(sink.context, .{ .delta = .{
                .part_index = "0",
                .data = .{ .text = .{ .text = text } },
            } }) == .stop) return error.LMFailure;
            return .{ .text = text };
        }

        fn receive(ptr: *anyopaque, fragment: stream.Fragment) lm_mod.OpenAI.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.published += fragment.text.len;
            return .proceed;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var program: Predict(S) = .{};
    for ([_][]const u8{ "unstructured text", "<answer>\nalready sent\n</answer>\n" }, [_]bool{ false, true }) |body, publishes| {
        var model: Model = .{ .body = body };
        const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = .{
            .ptr = &model,
            .vtable = &.{ .complete = Model.complete, .complete_stream = Model.completeStream },
        } };
        var bindings: [1]stream.Binding = undefined;
        var pending: [128]u8 = undefined;
        var output: [128]u8 = undefined;
        var fields: [2]lm_mod.XmlStreamListener.Field = undefined;
        var session = try stream.Session.init(arena, &program, &.{.{
            .field_name = "answer",
            .pending_storage = &pending,
            .output_storage = &output,
            .xml_fields = &fields,
        }}, &bindings, .{ .context = &model, .on_fragment = Model.receive });
        if (publishes) {
            try testing.expectError(error.AdapterParse, callStream(&program, ctx, .{}, &session, true));
            try testing.expectEqual(@as(usize, "already sent".len), model.published);
            try testing.expectEqual(@as(usize, 1), model.calls);
        } else {
            const recovered = try callStream(&program, ctx, .{}, &session, true);
            try testing.expectEqualStrings("recovered", recovered.prediction.fields().answer);
            try testing.expectEqual(@as(i64, 7), recovered.prediction.fields().count);
            try testing.expectEqual(@as(usize, 2), model.calls);
        }
    }
}

test "in-memory retriever ranks by overlap" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const corpus: InMemoryRetriever = .{ .passages = &.{ "Berlin is in Germany.", "Paris is the capital of France.", "France has wine." } };
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena.allocator(), .lm = script.lm(), .retriever = corpus.retriever() };
    const r: Retrieve = .{ .k = 2 };
    const got = try r.forward(ctx, "What is the capital of France?");
    try testing.expectEqualStrings("Paris is the capital of France.", got[0]);
    try testing.expectEqualStrings("France has wine.", got[1]);
    var storage: usage_mod.Storage = .{ .arena = arena.allocator(), .io = testing.io };
    var tracked = ctx;
    tracked.track_usage = .{ .storage = &storage, .limits = .{ .models = 0, .entries = 0 } };
    const prediction = try callProgram(&r, tracked, "What is the capital of France?");
    try testing.expectEqualStrings(got[0], prediction.fields().passages[0]);
    try testing.expectEqualStrings(got[1], prediction.fields().passages[1]);
    try testing.expectEqual(@as(usize, 0), prediction.getLMUsage().?.object.count());
}
