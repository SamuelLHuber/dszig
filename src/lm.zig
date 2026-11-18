//! Language-model interface. Programs never name a model: they call through
//! `LM`, so a scripted mock, a keyword mock, or an OpenAI-compatible HTTP
//! endpoint can be swapped without touching program code.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const adapter = @import("adapter.zig");
const pyjson = @import("pyjson.zig");
const http = @import("http.zig");
pub const inference = @import("inference.zig");
pub const sdk = @import("sdk_response.zig");
pub const public_history = @import("lm_history.zig");
pub const ChatStreamListener = @import("chat_stream_listener.zig").ChatStreamListener;
pub const JsonStreamListener = @import("json_stream_listener.zig").JsonStreamListener;
pub const XmlStreamListener = @import("xml_stream_listener.zig").XmlStreamListener;

test {
    _ = @import("json_stream_listener.zig");
    _ = @import("xml_stream_listener.zig");
}
const prompt_options = @import("lm_prompt_options.zig");
/// Canonical SDK token usage; every unknown counter stays null, never zero.
pub const Usage = @import("lm_usage.zig").Usage;
pub const logprobs = @import("logprobs.zig");
pub const TokenLogprob = logprobs.TokenLogprob;
pub const TopLogprob = logprobs.TopLogprob;
const accounting = @import("usage_tracker.zig");
const Message = adapter.Message;
pub const FunctionTool = inference.FunctionTool;
pub const ToolCall = inference.ToolCall;
pub const CitationPart = inference.CitationPart;
pub const Continuation = inference.Continuation;
pub const ResponseFormat = inference.ResponseFormat;
pub const ToolChoice = inference.ToolChoice;
/// TypeSafe System One decision backend (`POST /v1/systemone`).
pub const TypeSafe = @import("typesafe.zig").TypeSafe;
pub const Anthropic = @import("anthropic.zig").Anthropic;
pub const Gemini = @import("gemini.zig").Gemini;

pub const Options = struct {
    temperature: ?f64 = null,
    /// DSPy cache identity; candidate groups use a tagged JSON-array tuple.
    rollout_id: ?@import("runtime_value.zig").Value = null,
    response_format: ?ResponseFormat = null,
    tool_choice: ?ToolChoice = null,
    parallel_tool_calls: ?bool = null,
    /// SDK nonnegative integer: null is off, zero requests chosen tokens;
    /// positive values request alternatives without imposing provider caps.
    logprobs: @import("runtime_value.zig").Value = .null,
    /// Number of completions (OpenAI `n`).
    n: u32 = 1,
    /// Native reasoning effort; an absent option leaves provider defaults.
    reasoning_effort: ?[]const u8 = null,
    /// Distinguishes an explicit null from an unset option for preprocessing.
    /// A nonnull effort is inherently present, regardless of this marker.
    reasoning_effort_present: bool = false,
};

/// A typed System One judgment request. `state` is the provider state value,
/// `questions` the native question objects, and `response_schema` the JSON
/// schema whose top-level properties declare the judgments' answer spaces.
pub const DecisionRequest = struct {
    state: @import("runtime_value.zig").Value,
    questions: @import("runtime_value.zig").Value,
    response_schema: @import("runtime_value.zig").Value,
};

pub const Request = struct {
    /// Empty only for a decision request whose state is `decision.state`.
    messages: []const Message,
    options: Options = .{},
    tools: []const FunctionTool = &.{},
    /// Observe physical generations on engines with streaming support.
    /// Borrowed for this call, never serialized, hashed, or retained.
    stream_sink: ?OpenAI.StreamSink = null,
    /// Present only for LMs advertising `supports_decision_requests`.
    decision: ?DecisionRequest = null,
    /// Execution-local collector, never serialized, hashed or retained.
    usage_tracker: ?*accounting.UsageTracker = null,
    /// Optional caller-owned capture of secondary accounting failures. The
    /// engine error remains primary; one capture per concurrent public call.
    usage_failure: ?*AccountingFailure = null,
    /// Stack-local recording chain; only actual completed engine requests
    /// notify it. Cache hits preserve responses without inventing requests.
    recordings: ?*const RecordingLM.Pending = null,
    /// One stack-local public-call finalizer across transparent wrappers.
    execution: ?*LM.Execution = null,
};

pub fn cloneDecision(arena: Allocator, decision: DecisionRequest) Allocator.Error!DecisionRequest {
    return .{
        .state = try inference.cloneJsonValue(arena, decision.state),
        .questions = try inference.cloneJsonValue(arena, decision.questions),
        .response_schema = try inference.cloneJsonValue(arena, decision.response_schema),
    };
}

pub fn decisionEqual(a: ?DecisionRequest, b: ?DecisionRequest) @import("runtime_value.zig").CustomError!bool {
    const x = a orelse return b == null;
    const y = b orelse return false;
    return try inference.jsonEqual(x.state, y.state) and try inference.jsonEqual(x.questions, y.questions) and
        try inference.jsonEqual(x.response_schema, y.response_schema);
}

/// Borrowed metadata view; callers own `decision` for the view's lifetime.
pub fn decisionValue(arena: Allocator, decision: DecisionRequest) Allocator.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "state", decision.state);
    try object.put(arena, "questions", decision.questions);
    try object.put(arena, "response_schema", decision.response_schema);
    return .{ .object = object };
}

/// Owning reader; all three keys are required, though any may be JSON null.
pub fn decisionFromValue(arena: Allocator, input: @import("runtime_value.zig").Value) inference.MetadataError!DecisionRequest {
    if (input != .object) return error.BadMetadata;
    return cloneDecision(arena, .{
        .state = input.object.get("state") orelse return error.BadMetadata,
        .questions = input.object.get("questions") orelse return error.BadMetadata,
        .response_schema = input.object.get("response_schema") orelse return error.BadMetadata,
    });
}

pub const Error = error{ NoLM, LMFailure, ScriptExhausted, MissingUsageModel, StreamStopped } ||
    @import("runtime_value.zig").CustomError || accounting.Error || @import("call_result.zig").Error;

pub const AccountingFailure = struct {
    first: ?Error = null,
    count: u32 = 0,
};

/// Persistent configuration is separate from transport clients and credentials.
pub const StateError = error{ InvalidLMState, UnsupportedLMState, MissingLMLoader } || Allocator.Error;

fn jsonPayload(arena: Allocator, input: @import("runtime_value.zig").Value) Error!sdk.JsonValue {
    return sdk.JsonValue.fromBorrowedValue(arena, input) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    };
}

/// One provider completion and its optional, genuinely returned reasoning.
pub const Completion = struct {
    /// Null means no TextPart was returned; it is distinct from an empty text.
    text: ?[]const u8,
    reasoning: ?[]const u8 = null,
    tool_calls: []const ToolCall = &.{},
    citations: []const CitationPart = &.{},
    continuation: []const Continuation = &.{},
    /// Actual provider JSON, retained separately from canonical continuation.
    provider_data: ?sdk.JsonValue = null,
    /// Canonical response facts; raw provider numbers stay in `provider_data`.
    metadata: ResponseMetadata = .{},
    /// A validated generated media part was returned, independently of text
    /// and audio token usage. Payloads remain in `provider_data`.
    generated_media: bool = false,
    /// Null means scores were not reported; an empty sequence is distinct.
    logprobs: ?[]const TokenLogprob = null,
    /// False when local text editing could not retain every original score.
    logprobs_complete: bool = true,
    /// Unpublished wire artifact. A provider consumes it when publishing its
    /// canonical response; parser-only callers need no configured model.
    message: ?sdk.Message = null,
    /// The actual object returned by the engine, not an LM reconstruction.
    response: ?*const sdk.Response = null,
    adaptations: []const sdk.Adaptation = &.{},
    /// Priced once, before recording and cache publication.
    billing: ?@import("lm_cost.zig").Estimate = null,
    cache_hit: bool = false,

    /// Persist the canonical artifact, not the object's ownership header or
    /// the now-consumed wire slot. This keeps the native cache format stable.
    pub fn jsonStringify(self: Completion, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        inline for (@typeInfo(Completion).@"struct".field_names) |name| {
            if (comptime !std.mem.eql(u8, name, "response")) {
                try stream.objectField(name);
                if (comptime std.mem.eql(u8, name, "message")) {
                    try stream.write(completionMessage(&self));
                } else {
                    try stream.write(@field(self, name));
                }
            }
        }
        try stream.endObject();
    }
};

/// The five lm15 finish reasons.
pub const FinishReason = sdk.FinishReason;

/// Canonical facts a provider actually reported about one response. Every
/// field defaults to unknown: engines without the information leave it absent
/// rather than inventing an identity, a finish reason or token counts.
pub const ResponseMetadata = struct {
    id: ?[]const u8 = null,
    model: ?[]const u8 = null,
    finish_reason: ?FinishReason = null,
    usage: Usage = .{},

    /// False for legacy completions, whose serialized shape has no metadata.
    pub fn known(metadata: ResponseMetadata) bool {
        if (metadata.id != null) return true;
        if (metadata.model != null) return true;
        if (metadata.finish_reason != null) return true;
        return usageKnown(metadata.usage);
    }
};

const usage_counters = [_][]const u8{
    "input_tokens",       "output_tokens",    "total_tokens",       "cache_read_tokens",
    "cache_write_tokens", "reasoning_tokens", "input_audio_tokens", "output_audio_tokens",
};

comptime {
    // Every counter must be inspected, or a reported count could be dropped.
    assert(@typeInfo(Usage).@"struct".field_names.len == usage_counters.len);
}

fn usageKnown(usage: Usage) bool {
    inline for (usage_counters) |name| {
        if (@field(usage, name) != .null) return true;
    }
    return false;
}

/// Owning copy which borrows nothing from `metadata`; unwinds on failure.
pub fn cloneResponseMetadata(allocator: Allocator, metadata: ResponseMetadata) Allocator.Error!ResponseMetadata {
    const id = if (metadata.id) |value| try allocator.dupe(u8, value) else null;
    errdefer if (id) |value| allocator.free(value);
    const model = if (metadata.model) |value| try allocator.dupe(u8, value) else null;
    errdefer if (model) |value| allocator.free(value);
    return .{
        .id = id,
        .model = model,
        .finish_reason = metadata.finish_reason,
        .usage = try metadata.usage.clone(allocator),
    };
}

pub fn freeResponseMetadata(allocator: Allocator, metadata: ResponseMetadata) void {
    metadata.usage.free(allocator);
    if (metadata.model) |model| allocator.free(model);
    if (metadata.id) |id| allocator.free(id);
}

/// Sparse borrowed view: only known fields appear, so unknown never reads as
/// zero. Callers own `metadata` for the view's lifetime.
pub fn responseMetadataValue(arena: Allocator, metadata: ResponseMetadata) Allocator.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    if (metadata.id) |id| try object.put(arena, "id", .{ .string = id });
    if (metadata.model) |model| try object.put(arena, "model", .{ .string = model });
    if (metadata.finish_reason) |reason| try object.put(arena, "finish_reason", .{ .string = @tagName(reason) });
    if (usageKnown(metadata.usage)) try object.put(arena, "usage", try metadata.usage.toValue(arena));
    return .{ .object = object };
}

/// Owning reader for `responseMetadataValue`. Absent fields stay unknown;
/// unrecognized fields refuse rather than being silently dropped.
pub fn responseMetadataFromValue(arena: Allocator, input: @import("runtime_value.zig").Value) inference.MetadataError!ResponseMetadata {
    if (input != .object) return error.BadMetadata;
    var out: ResponseMetadata = .{};
    var recognized: usize = 0;
    if (input.object.get("id")) |id| {
        if (id != .null) out.id = try arena.dupe(u8, try metadataIdentity(id));
        recognized += 1;
    }
    if (input.object.get("model")) |model| {
        if (model != .null) out.model = try arena.dupe(u8, try metadataIdentity(model));
        recognized += 1;
    }
    if (input.object.get("finish_reason")) |reason| {
        if (reason != .null) out.finish_reason = std.meta.stringToEnum(FinishReason, try metadataString(reason)) orelse return error.BadMetadata;
        recognized += 1;
    }
    if (input.object.get("usage")) |usage| {
        const borrowed = Usage.fromValue(arena, usage) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.InvalidUsage => error.BadMetadata,
        };
        out.usage = try borrowed.clone(arena);
        recognized += 1;
    }
    if (recognized != input.object.count()) return error.BadMetadata;
    return out;
}

/// An unknown identity is an absent field, so a present one is never empty.
fn metadataIdentity(input: @import("runtime_value.zig").Value) inference.MetadataError![]const u8 {
    const text = try metadataString(input);
    if (text.len == 0) return error.BadMetadata;
    return text;
}

/// Engine-boundary validation of caller-supplied metadata. The result may
/// borrow from `raw`; engines clone it before returning a completion.
fn checkedMetadata(arena: Allocator, raw: ResponseMetadata) Error!ResponseMetadata {
    if (raw.id) |id| {
        if (id.len == 0) return error.LMFailure;
    }
    if (raw.model) |model| {
        if (model.len == 0) return error.LMFailure;
    }
    var out = raw;
    out.usage = try checkedUsage(arena, raw.usage);
    return out;
}

/// SDK Usage construction; a refused count is a failed response.
fn checkedUsage(arena: Allocator, raw: Usage) Error!Usage {
    return Usage.init(arena, raw) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidUsage => error.LMFailure,
    };
}

/// lm15 Response.provider_data permits only an optional JSON object, without
/// coercing builtin classes into JSON. Like the other runtime traversals this
/// native check is bounded by pyjson.max_depth; SDK exception classes differ.
fn providerJsonValue(input: @import("runtime_value.zig").Value, depth: usize) bool {
    if (depth >= pyjson.max_depth) return false;
    return switch (input) {
        .null, .bool, .integer, .string => true,
        .float => |number| std.math.isFinite(number),
        .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text))
            true
        else
            std.math.isFinite(std.fmt.parseFloat(f64, text) catch return false),
        .array => |items| blk: {
            for (items.items) |item| if (!providerJsonValue(item, depth + 1)) break :blk false;
            break :blk true;
        },
        .object => |object| blk: {
            for (object.values()) |item| if (!providerJsonValue(item, depth + 1)) break :blk false;
            break :blk true;
        },
        .custom => if (@import("python_value.zig").view(input)) |builtin|
            builtin == .float and std.math.isFinite(builtin.float)
        else
            false,
    };
}

/// Publish once, inside the engine. Wire parsers borrow some immutable leaves;
/// owning clones instead release their replaced fields. The remaining canonical
/// fields become borrowed views of the class, which alone releases its storage.
pub fn publishResponse(
    allocator: Allocator,
    completion: *Completion,
    model: []const u8,
    ownership: enum { borrowed, owned },
) Error!void {
    assert(completion.response == null);
    const message = completion.message orelse return;
    const fields: sdk.ResponseFields = .{
        .id = completion.metadata.id,
        .model = completion.metadata.model orelse model,
        .message = message,
        .finish_reason = completion.metadata.finish_reason orelse
            if (completion.tool_calls.len != 0) .tool_call else .stop,
        .usage = completion.metadata.usage,
        .logprobs = completion.logprobs,
        .logprobs_complete = completion.logprobs_complete,
        .provider_data = completion.provider_data,
        .adaptations = completion.adaptations,
    };
    const object = sdk.Response.fromFields(allocator, &fields) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    };
    if (ownership == .owned) {
        message.free(allocator);
        freeResponseMetadata(allocator, completion.metadata);
        if (completion.provider_data) |data| data.free();
        if (completion.logprobs) |items| logprobs.free(allocator, items);
        sdk.freeAdaptations(allocator, completion.adaptations);
    }
    bindResponse(completion, object);
}

fn bindResponse(completion: *Completion, object: *const sdk.Response) void {
    assert(completion.response == null);
    completion.response = object;
    completion.message = null;
    completion.provider_data = object.fields.provider_data;
    completion.metadata = .{
        .id = object.fields.id,
        .model = object.fields.model,
        .finish_reason = object.fields.finish_reason,
        .usage = object.fields.usage,
    };
    completion.logprobs = object.fields.logprobs;
    completion.logprobs_complete = object.fields.logprobs_complete;
    completion.adaptations = object.fields.adaptations;
    assert(completion.message == null);
}

fn completionMessage(completion: *const Completion) ?sdk.Message {
    if (completion.response) |object| return object.fields.message;
    return completion.message;
}

/// Owning clones for long-lived histories and caches. Completion cloning also
/// unwinds partial allocations, so `freeCompletion` works with a general GPA.
pub fn cloneCompletion(allocator: Allocator, completion: Completion) Allocator.Error!Completion {
    var context: sdk.Response.CloneContext = .init(allocator);
    defer context.deinit();
    return cloneCompletionWithContext(allocator, completion, &context);
}

/// A whole response group shares one context to preserve repeated objects and
/// cross-response payload aliases. Keep earlier results alive until deinit.
pub fn cloneCompletionWithContext(
    allocator: Allocator,
    completion: Completion,
    context: *sdk.Response.CloneContext,
) Allocator.Error!Completion {
    const response = if (completion.response) |object| try context.clone(object) else null;
    defer if (response) |object| object.release();
    return cloneCompletionWithResponse(allocator, completion, response);
}

/// Copy native completion fields while retaining a separately owned SDK object.
/// Canonical cache reads reconstruct each response occurrence from its record;
/// they must not deep-copy that response again or coalesce repeated headers.
pub fn cloneCompletionWithResponse(
    allocator: Allocator,
    completion: Completion,
    response: ?*const sdk.Response,
) Allocator.Error!Completion {
    const text = if (completion.text) |value| try allocator.dupe(u8, value) else null;
    errdefer if (text) |value| allocator.free(value);
    const reasoning = if (completion.reasoning) |summary| try allocator.dupe(u8, summary) else null;
    errdefer if (reasoning) |summary| allocator.free(summary);
    const tool_calls = try inference.cloneToolCalls(allocator, completion.tool_calls);
    errdefer inference.freeToolCalls(allocator, tool_calls);
    const citations = try inference.cloneCitations(allocator, completion.citations);
    errdefer inference.freeCitations(allocator, citations);
    const continuation = try inference.cloneContinuations(allocator, completion.continuation);
    errdefer inference.freeContinuations(allocator, continuation);
    const artifacts = try (sdk.Artifacts{
        .message = completion.message,
        .provider_data = if (response == null) completion.provider_data else null,
    }).clone(allocator);
    errdefer artifacts.free(allocator);
    const scores = if (response == null and completion.logprobs != null)
        try logprobs.clone(allocator, completion.logprobs.?)
    else
        null;
    errdefer if (scores) |items| logprobs.free(allocator, items);
    const adaptations = try sdk.cloneAdaptations(allocator, if (response == null) completion.adaptations else &.{});
    errdefer sdk.freeAdaptations(allocator, adaptations);
    const billing = if (completion.billing) |bill| try cloneBilling(allocator, bill) else null;
    errdefer if (billing) |bill| freeBilling(allocator, bill);
    var result: Completion = .{
        .text = text,
        .reasoning = reasoning,
        .tool_calls = tool_calls,
        .citations = citations,
        .continuation = continuation,
        .provider_data = artifacts.provider_data,
        .metadata = if (response == null) try cloneResponseMetadata(allocator, completion.metadata) else .{},
        .generated_media = completion.generated_media,
        .logprobs = scores,
        .logprobs_complete = completion.logprobs_complete,
        .message = artifacts.message,
        .adaptations = adaptations,
        .billing = billing,
        .cache_hit = completion.cache_hit,
    };
    if (response) |object| bindResponse(&result, object.retain());
    return result;
}

pub fn freeCompletion(allocator: Allocator, completion: Completion) void {
    if (completion.billing) |bill| freeBilling(allocator, bill);
    if (completion.response) |object| {
        assert(completion.message == null);
        object.release();
    } else {
        if (completion.message) |message| message.free(allocator);
        sdk.freeAdaptations(allocator, completion.adaptations);
        if (completion.logprobs) |items| logprobs.free(allocator, items);
        freeResponseMetadata(allocator, completion.metadata);
        if (completion.provider_data) |data| data.free();
    }
    if (completion.text) |text| allocator.free(text);
    if (completion.reasoning) |reasoning| allocator.free(reasoning);
    inference.freeContinuations(allocator, completion.continuation);
    inference.freeToolCalls(allocator, completion.tool_calls);
    inference.freeCitations(allocator, completion.citations);
}

fn cloneBilling(allocator: Allocator, bill: @import("lm_cost.zig").Estimate) Allocator.Error!@import("lm_cost.zig").Estimate {
    const cost = try inference.cloneJsonValue(allocator, bill.cost);
    errdefer inference.freeJsonValue(allocator, cost);
    return .{ .cost = cost, .details = try inference.cloneJsonValue(allocator, bill.details) };
}

fn freeBilling(allocator: Allocator, bill: @import("lm_cost.zig").Estimate) void {
    inference.freeJsonValue(allocator, bill.cost);
    inference.freeJsonValue(allocator, bill.details);
}

pub fn cloneOptions(arena: Allocator, options: Options) Allocator.Error!Options {
    var out = options;
    if (options.rollout_id) |id| out.rollout_id = try inference.cloneJsonValue(arena, id);
    out.logprobs = try inference.cloneJsonValue(arena, options.logprobs);
    if (options.reasoning_effort) |effort| out.reasoning_effort = try arena.dupe(u8, effort);
    if (options.response_format) |format| out.response_format = try inference.cloneResponseFormat(arena, format);
    if (options.tool_choice) |choice| out.tool_choice = try inference.cloneToolChoice(arena, choice);
    return out;
}

pub fn cloneMessage(arena: Allocator, message: Message) Allocator.Error!Message {
    return .{
        .role = message.role,
        .content = try message.content.clone(arena),
        .reasoning = if (message.reasoning) |reasoning| try arena.dupe(u8, reasoning) else null,
        .tool_calls = try inference.cloneToolCalls(arena, message.tool_calls),
        .tool_call_id = if (message.tool_call_id) |id| try arena.dupe(u8, id) else null,
        .continuation = try inference.cloneContinuations(arena, message.continuation),
    };
}

/// Canonical metadata, before a provider dialect adds its wire wrappers.
/// All knobs remain visible, including explicitly absent values.
pub fn optionsValue(arena: Allocator, options: Options) Allocator.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "temperature", if (options.temperature) |temperature| .{ .float = temperature } else .null);
    try object.put(arena, "rollout_id", options.rollout_id orelse .null);
    try object.put(arena, "n", .{ .integer = options.n });
    try object.put(arena, "reasoning_effort", if (options.reasoning_effort) |effort| .{ .string = effort } else .null);
    try object.put(arena, "reasoning_effort_present", .{ .bool = options.reasoning_effort_present });
    try object.put(arena, "response_format", if (options.response_format) |format| try inference.responseFormatValue(arena, format) else .null);
    try object.put(arena, "tool_choice", if (options.tool_choice) |choice| try inference.toolChoiceValue(arena, choice) else .null);
    try object.put(arena, "parallel_tool_calls", if (options.parallel_tool_calls) |parallel| .{ .bool = parallel } else .null);
    try object.put(arena, "logprobs", options.logprobs);
    return .{ .object = object };
}

pub fn messageValue(arena: Allocator, message: Message) sdk.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "role", .{ .string = message.role.name() });
    try object.put(arena, "content", message.content.toValue(arena));
    try object.put(arena, "reasoning", if (message.reasoning) |reasoning| .{ .string = reasoning } else .null);
    try object.put(arena, "tool_calls", try inference.toolCallsValue(arena, message.tool_calls));
    try object.put(arena, "tool_call_id", if (message.tool_call_id) |id| .{ .string = id } else .null);
    try object.put(arena, "continuation", try inference.continuationsValue(arena, message.continuation));
    return .{ .object = object };
}

pub fn requestValue(arena: Allocator, request: Request) sdk.Error!@import("runtime_value.zig").Value {
    var messages: @import("runtime_value.zig").Array = .init(arena);
    try messages.ensureTotalCapacity(request.messages.len);
    for (request.messages) |message| messages.appendAssumeCapacity(try messageValue(arena, message));
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "messages", .{ .array = messages });
    try object.put(arena, "options", try optionsValue(arena, request.options));
    try object.put(arena, "tools", try inference.toolsValue(arena, request.tools));
    // Absent for ordinary requests, keeping historical metadata unchanged.
    if (request.decision) |decision| try object.put(arena, "decision", try decisionValue(arena, decision));
    return .{ .object = object };
}

pub fn completionValue(arena: Allocator, completion: Completion) sdk.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "text", if (completion.text) |text| .{ .string = text } else .null);
    try object.put(arena, "reasoning", if (completion.reasoning) |reasoning| .{ .string = reasoning } else .null);
    try object.put(arena, "tool_calls", try inference.toolCallsValue(arena, completion.tool_calls));
    try object.put(arena, "citations", try inference.citationsValue(arena, completion.citations));
    try object.put(arena, "continuation", try inference.continuationsValue(arena, completion.continuation));
    try object.put(arena, "provider_data", if (completion.provider_data) |data| try data.toValue(arena) else .null);
    if (completion.billing) |bill| {
        var billing: @import("runtime_value.zig").ObjectMap = .empty;
        try billing.put(arena, "cost", bill.cost);
        try billing.put(arena, "details", bill.details);
        try object.put(arena, "billing", .{ .object = billing });
    }
    if (completion.cache_hit) try object.put(arena, "cache_hit", .{ .bool = true });
    if (completion.generated_media) try object.put(arena, "generated_media", .{ .bool = true });
    if (completion.logprobs) |items| try object.put(arena, "logprobs", try logprobs.toNativeValue(arena, items));
    if (!completion.logprobs_complete) try object.put(arena, "logprobs_complete", .{ .bool = false });
    if (completionMessage(&completion)) |message| try object.put(arena, "message", try message.toValue(arena));
    if (completion.adaptations.len != 0)
        try object.put(arena, "adaptations", try sdk.adaptationsValue(arena, completion.adaptations));
    // Legacy completions carry no metadata and keep their original shape.
    if (completion.metadata.known()) try object.put(arena, "metadata", try responseMetadataValue(arena, completion.metadata));
    return .{ .object = object };
}

/// Owning readers accept extra row fields, allowing a transcript call to carry
/// completion evidence alongside its canonical request without dropping it.
pub fn optionsFromValue(arena: Allocator, input: @import("runtime_value.zig").Value) inference.MetadataError!Options {
    if (input != .object) return error.BadMetadata;
    var out: Options = .{};
    if (input.object.get("temperature")) |temperature| {
        out.temperature = switch (temperature) {
            .null => null,
            .float => temperature.float,
            .integer => @floatFromInt(temperature.integer),
            else => return error.BadMetadata,
        };
    }
    if (input.object.get("rollout_id")) |id| {
        if (id != .null) out.rollout_id = try inference.cloneJsonValue(arena, id);
    }
    if (input.object.get("n")) |n| {
        out.n = try metadataU32(n);
        if (out.n == 0) return error.BadMetadata;
    }
    if (input.object.get("reasoning_effort")) |effort| {
        if (effort != .null) out.reasoning_effort = try arena.dupe(u8, try metadataString(effort));
    }
    if (input.object.get("reasoning_effort_present")) |present| {
        if (present != .bool) return error.BadMetadata;
        out.reasoning_effort_present = present.bool;
    }
    if (input.object.get("response_format")) |format| {
        if (format != .null) out.response_format = try inference.responseFormatFromValue(arena, format);
    }
    if (input.object.get("tool_choice")) |choice| {
        if (choice != .null) out.tool_choice = try inference.toolChoiceFromValue(arena, choice);
    }
    if (input.object.get("logprobs")) |scores| {
        const normalized = Usage.init(arena, .{ .input_tokens = scores }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.InvalidUsage => return error.BadMetadata,
        };
        out.logprobs = try inference.cloneJsonValue(arena, normalized.input_tokens);
    }
    if (input.object.get("parallel_tool_calls")) |parallel| {
        if (parallel != .null) {
            if (parallel != .bool) return error.BadMetadata;
            out.parallel_tool_calls = parallel.bool;
        }
    }
    return out;
}

pub fn messageFromValue(arena: Allocator, input: @import("runtime_value.zig").Value) inference.MetadataError!Message {
    if (input != .object) return error.BadMetadata;
    const role = try metadataString(input.object.get("role") orelse return error.BadMetadata);
    const content = input.object.get("content") orelse return error.BadMetadata;
    var out: Message = .{
        .role = std.meta.stringToEnum(adapter.Role, role) orelse return error.BadMetadata,
        .content = adapter.Content.fromValue(arena, content) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.BadMetadata,
        },
    };
    if (input.object.get("reasoning")) |reasoning| {
        if (reasoning != .null) out.reasoning = try arena.dupe(u8, try metadataString(reasoning));
    }
    if (input.object.get("tool_calls")) |calls| out.tool_calls = try inference.toolCallsFromValue(arena, calls);
    if (input.object.get("tool_call_id")) |id| {
        if (id != .null) out.tool_call_id = try arena.dupe(u8, try metadataString(id));
    }
    if (input.object.get("continuation")) |continuation| out.continuation = try inference.continuationsFromValue(arena, continuation);
    return out;
}

pub fn requestFromValue(arena: Allocator, input: @import("runtime_value.zig").Value) inference.MetadataError!Request {
    if (input != .object) return error.BadMetadata;
    const raw_messages = input.object.get("messages") orelse return error.BadMetadata;
    const decision: ?DecisionRequest = if (input.object.get("decision")) |raw|
        if (raw == .null) null else try decisionFromValue(arena, raw)
    else
        null;
    if (raw_messages != .array or (raw_messages.array.items.len == 0 and decision == null)) return error.BadMetadata;
    const messages = try arena.alloc(Message, raw_messages.array.items.len);
    for (raw_messages.array.items, messages) |message, *copy| copy.* = try messageFromValue(arena, message);
    var out: Request = .{ .messages = messages, .decision = decision };
    if (input.object.get("options")) |options| {
        if (options != .null) out.options = try optionsFromValue(arena, options);
    }
    if (input.object.get("tools")) |tools| out.tools = try inference.toolsFromValue(arena, tools);
    return out;
}

pub fn completionFromValue(arena: Allocator, input: @import("runtime_value.zig").Value) inference.MetadataError!Completion {
    if (input != .object) return error.BadMetadata;
    const text = input.object.get("text") orelse return error.BadMetadata;
    var out: Completion = .{ .text = if (text == .null) null else try arena.dupe(u8, try metadataString(text)) };
    errdefer freeCompletion(arena, out);
    if (input.object.get("message")) |message| {
        if (message != .null) out.message = sdk.Message.fromValue(arena, message) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.BadMetadata,
        };
    }
    if (input.object.get("adaptations")) |adaptations| {
        out.adaptations = sdk.adaptationsFromValue(arena, adaptations) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.BadMetadata,
        };
    }
    if (input.object.get("citations")) |citations| out.citations = try inference.citationsFromValue(arena, citations);
    if (input.object.get("reasoning")) |reasoning| {
        if (reasoning != .null) out.reasoning = try arena.dupe(u8, try metadataString(reasoning));
    }
    if (input.object.get("tool_calls")) |calls| out.tool_calls = try inference.toolCallsFromValue(arena, calls);
    if (input.object.get("continuation")) |continuation| out.continuation = try inference.continuationsFromValue(arena, continuation);
    if (input.object.get("provider_data")) |data| {
        if (data != .null) {
            if (data != .object) return error.BadMetadata;
            out.provider_data = sdk.JsonValue.fromValue(arena, data) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.BadMetadata,
            };
        }
    }
    if (input.object.get("metadata")) |metadata| {
        if (metadata != .null) out.metadata = try responseMetadataFromValue(arena, metadata);
    }
    if (input.object.get("generated_media")) |media| {
        if (media != .bool) return error.BadMetadata;
        out.generated_media = media.bool;
    }
    if (input.object.get("logprobs")) |scores| {
        if (scores != .null) out.logprobs = logprobs.fromNativeValue(arena, scores) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.InvalidLogprobs => return error.BadMetadata,
        };
    }
    if (input.object.get("logprobs_complete")) |complete| {
        if (complete != .bool) return error.BadMetadata;
        out.logprobs_complete = complete.bool;
    }
    if (input.object.get("billing")) |bill| {
        if (bill != .null) {
            if (bill != .object) return error.BadMetadata;
            out.billing = try cloneBilling(arena, .{
                .cost = bill.object.get("cost") orelse return error.BadMetadata,
                .details = bill.object.get("details") orelse return error.BadMetadata,
            });
        }
    }
    if (input.object.get("cache_hit")) |hit| {
        if (hit != .bool) return error.BadMetadata;
        out.cache_hit = hit.bool;
    }
    if (out.metadata.model) |model| {
        publishResponse(arena, &out, model, .owned) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.BadMetadata,
        };
    }
    return out;
}

fn metadataString(input: @import("runtime_value.zig").Value) inference.MetadataError![]const u8 {
    if (input != .string) return error.BadMetadata;
    return input.string;
}

fn metadataU32(input: @import("runtime_value.zig").Value) inference.MetadataError!u32 {
    if (input != .integer or input.integer < 0 or input.integer > 0xffffffff) return error.BadMetadata;
    return @intCast(input.integer);
}

fn optionalStringEqual(a: ?[]const u8, b: ?[]const u8) bool {
    if (a) |text| return std.mem.eql(u8, text, b orelse return false);
    return b == null;
}

pub const LM = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    supports_reasoning: bool = false,
    supports_response_format: bool = false,
    supports_response_schema: bool = false,
    supports_function_calling: bool = false,
    /// Accepts `Request.decision` System One judgments instead of chat prompts.
    supports_decision_requests: bool = false,
    pricing: ?Pricing = null,
    /// Client output mode, independently of the provider's HTTP wire dialect.
    model_type: @import("call_result.zig").ModelType = .auto,

    /// Startup-owned pricing resources. Advisory storage must be disjoint from
    /// generation storage and outlive every call using this handle.
    pub const Pricing = struct {
        catalogue: *const @import("model_prices.zig").Catalogue,
        arena: Allocator,
        provider: ?[]const u8 = null,
        namespaces: ?[]const []const u8 = null,
    };

    pub const CanonicalCall = struct {
        use_cache: bool = true,
        rollout_id: ?@import("runtime_value.zig").Value = null,
        usage_tracker: ?*accounting.UsageTracker = null,
        history: ?public_history.Capture = null,
        maximum_file_bytes: usize = 16 * 1024 * 1024,
    };

    pub const PromptCall = struct {
        pub const max_candidates = 1024;
        config: sdk.Config = .{},
        n: u32 = 1,
        execution: CanonicalCall = .{},
    };

    pub const PromptOptions = prompt_options.Input;

    const Execution = struct {
        pricing: ?Pricing = null,
        model_type: @import("call_result.zig").ModelType = .auto,
        want_result: bool = false,
        canonical: ?*const sdk.Request = null,
        use_cache: bool = true,
        result: ?@import("call_result.zig").CallResult = null,
        engine: ?LM = null,
        completions: []const Completion = &.{},
        written: u32 = 0,
    };

    pub const VTable = struct {
        /// Returns `request.options.n` completions, allocated in `arena`.
        complete: *const fn (ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion,
        /// Optional streaming equivalent of one physical adapter generation.
        complete_stream: ?*const fn (
            ptr: *anyopaque,
            arena: Allocator,
            request: Request,
            sink: OpenAI.StreamSink,
        ) Error!Completion = null,
        /// One public SDK request at the engine boundary; no adapter defaults.
        complete_canonical: ?*const fn (
            ptr: *anyopaque,
            arena: Allocator,
            request: *const sdk.Request,
        ) Error!Completion = null,
        /// Transparent wrappers delegate exactly one whole public group.
        /// Final accounting runs after all wrappers, not inside cache storage.
        handles_candidate_groups: bool = false,
        /// Custom native engines explicitly opt into persistent state.
        dump_state: ?*const fn (ptr: *anyopaque, arena: Allocator) StateError!@import("runtime_value.zig").Value = null,
        /// A genuine provider starts work in the caller's startup-created slot.
        finetune: ?*const fn (ptr: *anyopaque, job: *@import("training.zig").Job, request: @import("training.zig").Request) anyerror!void = null,
        /// API providers need no local process. Native engines opt into lifecycle
        /// hooks explicitly; kill must return only after their workers have joined.
        launch: ?*const fn (ptr: *anyopaque) anyerror!void = null,
        kill: ?*const fn (ptr: *anyopaque) anyerror!void = null,
        model_name: ?*const fn (ptr: *anyopaque) []const u8 = null,
        /// Ordinary Module.deepcopy clones model instances; native engines must
        /// provide a real snapshot rather than silently sharing mutable state.
        clone: ?*const fn (ptr: *anyopaque, arena: Allocator) anyerror!LM = null,
    };

    /// Explicit application resources, never reconstructed from serialized pointers.
    /// The callback must return handles whose storage outlives the loaded program.
    pub const StateLoader = struct {
        ptr: *anyopaque,
        restore_engine: *const fn (ptr: *anyopaque, arena: Allocator, state: @import("runtime_value.zig").Value) StateError!LM,

        pub fn restore(self: StateLoader, arena: Allocator, state: @import("runtime_value.zig").Value) StateError!LM {
            if (state != .object) return error.InvalidLMState;
            const mode: ?@import("call_result.zig").ModelType = if (state.object.get("model_type")) |encoded| blk: {
                if (encoded != .string) return error.InvalidLMState;
                break :blk std.meta.stringToEnum(@import("call_result.zig").ModelType, encoded.string) orelse
                    return error.InvalidLMState;
            } else null;
            var restored = try self.restore_engine(self.ptr, arena, state);
            if (mode) |value| restored.model_type = value;
            return restored;
        }
    };

    var unconfigured_token: u8 = 0;
    pub const unconfigured: LM = .{
        .ptr = &unconfigured_token,
        .vtable = &.{ .complete = unconfiguredComplete },
    };

    fn unconfiguredComplete(_: *anyopaque, _: Allocator, _: Request) Error![]const Completion {
        return error.NoLM;
    }

    pub fn sameInstance(a: LM, b: LM) bool {
        return a.ptr == b.ptr and a.vtable == b.vtable;
    }

    pub fn clone(model: LM, arena: Allocator) !LM {
        const copy = model.vtable.clone orelse return error.UnsupportedLMCopy;
        var result = try copy(model.ptr, arena);
        result.pricing = model.pricing;
        result.model_type = model.model_type;
        return result;
    }

    pub fn modelName(model: LM) ?[]const u8 {
        const name = model.vtable.model_name orelse return null;
        return name(model.ptr);
    }

    pub fn launch(model: LM) !void {
        if (model.vtable.launch) |start| try start(model.ptr);
    }

    pub fn kill(model: LM) !void {
        if (model.vtable.kill) |stop| try stop(model.ptr);
    }

    pub fn finetune(model: LM, job: *@import("training.zig").Job, request: @import("training.zig").Request) !void {
        const start = model.vtable.finetune orelse return error.UnsupportedFinetuning;
        return start(model.ptr, job, request);
    }

    pub fn dumpState(lm: LM, arena: Allocator) StateError!@import("runtime_value.zig").Value {
        const dump = lm.vtable.dump_state orelse return error.UnsupportedLMState;
        var state = try dump(lm.ptr, arena);
        if (state != .object) return error.InvalidLMState;
        try state.object.put(arena, "model_type", .{ .string = @tagName(lm.model_type) });
        return state;
    }

    pub fn complete(lm: LM, arena: Allocator, request: Request) Error![]const u8 {
        assert(request.options.n == 1);
        return (try lm.completeResponses(arena, request))[0].text orelse error.LMFailure;
    }

    /// Text-only convenience; use `completeResponses` to retain native reasoning.
    pub fn completeMany(lm: LM, arena: Allocator, request: Request) Error![]const []const u8 {
        const completions = try lm.completeResponses(arena, request);
        const out = try arena.alloc([]const u8, completions.len);
        for (out, completions) |*text, completion| text.* = completion.text orelse return error.LMFailure;
        return out;
    }

    /// Owns canonical SDK responses and the Source convenience-output view.
    /// Like completeResponses, generation scratch storage belongs to `arena`.
    /// Custom engines must provide their actual SDK response object, not a
    /// reconstruction from joined text or unpublished wire artifacts.
    pub fn callResult(lm: LM, arena: Allocator, request: Request) Error!@import("call_result.zig").CallResult {
        assert(request.execution == null);
        var execution: Execution = .{
            .pricing = lm.pricing,
            .model_type = lm.model_type,
            .want_result = true,
        };
        _ = try lm.execute(arena, request, &execution);
        return execution.result.?;
    }

    /// Execute one normalized request through the public cache and accounting
    /// boundary. No adapter options or engine defaults replace Request.config.
    /// The backing allocators of retained SDK objects must outlive their
    /// cache/history entries, even after the caller frees its CallResult.
    pub fn callCanonicalResult(
        lm: LM,
        arena: Allocator,
        io: std.Io,
        request: *const sdk.Request,
        options: CanonicalCall,
    ) (Error || public_history.Error)!@import("call_result.zig").CallResult {
        if (options.history) |capture| try capture.validate();
        const configured = lm.modelName() orelse return error.LMFailure;
        if (!std.mem.eql(u8, configured, request.fields.model)) return error.LMFailure;
        const snapshot = request.snapshot(arena, io, options.maximum_file_bytes) catch |err|
            return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
        defer snapshot.release();
        return lm.callSdkGroup(arena, io, snapshot, options, 1, null);
    }

    /// Execute a string as one SDK user message. Config is already normalized;
    /// unlike the adapter lane, this does not apply legacy generation defaults.
    pub fn callPromptResult(
        lm: LM,
        arena: Allocator,
        io: std.Io,
        prompt: []const u8,
        options: PromptCall,
    ) (Error || public_history.Error)!@import("call_result.zig").CallResult {
        if (options.n == 0 or options.n > PromptCall.max_candidates) return error.LMFailure;
        if (!std.unicode.utf8ValidateSlice(prompt)) return error.LMFailure;
        if (options.execution.history) |capture| try capture.validate();
        const configured = lm.modelName() orelse return error.LMFailure;
        const request = sdk.Request.fromFields(arena, &.{
            .model = configured,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = prompt } }} }},
            .config = options.config,
        }) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
        defer request.release();
        return lm.callSdkGroup(
            arena,
            io,
            request,
            options.execution,
            options.n,
            request.fields.messages[0].parts[0].text.text,
        );
    }

    /// Merge JSON generation defaults and overrides using generic Chat spellings.
    /// Connection, caching and accounting remain controlled by the native LM and execution options.
    pub fn callPromptOptionsResult(
        lm: LM,
        arena: Allocator,
        io: std.Io,
        prompt: []const u8,
        options: *const PromptOptions,
        execution: CanonicalCall,
    ) (Error || public_history.Error)!@import("call_result.zig").CallResult {
        if (!std.unicode.utf8ValidateSlice(prompt)) return error.LMFailure;
        if (execution.history) |capture| try capture.validate();
        const configured = lm.modelName() orelse return error.LMFailure;
        const prepared = prompt_options.build(
            arena,
            configured,
            prompt,
            options,
            lm.model_type == .text,
            PromptCall.max_candidates,
        ) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
        defer prepared.request.release();
        return lm.callSdkGroup(
            arena,
            io,
            prepared.request,
            execution,
            prepared.count,
            prepared.request.fields.messages[0].parts[0].text.text,
        );
    }

    fn callSdkGroup(
        lm: LM,
        arena: Allocator,
        io: std.Io,
        snapshot: *const sdk.Request,
        options: CanonicalCall,
        count: u32,
        prompt: ?[]const u8,
    ) (Error || public_history.Error)!@import("call_result.zig").CallResult {
        assert(count > 0);
        assert(prompt != null or count == 1);
        var execution: Execution = .{
            .pricing = lm.pricing,
            .model_type = lm.model_type,
            .want_result = true,
            .canonical = snapshot,
            .use_cache = options.use_cache,
        };
        var cache_completions: []const Completion = &.{};
        defer {
            const temporary = if (execution.engine == null)
                cache_completions
            else
                execution.completions[0..execution.written];
            for (temporary) |completion| {
                if (completion.response) |response| response.release();
            }
        }
        const completions = try lm.execute(arena, .{
            .messages = &.{},
            .options = .{
                .rollout_id = options.rollout_id,
                .logprobs = snapshot.fields.config.logprobs,
                .n = count,
            },
            .usage_tracker = options.usage_tracker,
        }, &execution);
        if (execution.engine == null) cache_completions = completions;
        const result = execution.result.?;
        errdefer result.free(arena);
        if (options.history) |capture| try capture.publish(io, snapshot, prompt, &result, lm.model_type);
        return result;
    }

    fn resultsFromCompletions(lm: LM, arena: Allocator, completions: []const Completion, options: Options) Error!@import("call_result.zig").CallResult {
        const result_module = @import("call_result.zig");
        assert(completions.len > 0);
        if (completions.len == 1) return lm.candidateResult(arena, &completions[0], options);
        const candidates = try arena.alloc(result_module.CallResult, completions.len);
        defer arena.free(candidates);
        var initialized: usize = 0;
        defer for (candidates[0..initialized]) |*candidate| candidate.free(arena);
        for (completions, candidates) |*completion, *candidate| {
            candidate.* = try lm.candidateResult(arena, completion, options);
            initialized += 1;
        }
        var result = try result_module.combine(arena, candidates, lm.model_type);
        result.cache_hit = completions[0].cache_hit;
        return result;
    }

    fn candidateResult(lm: LM, arena: Allocator, completion: *const Completion, options: Options) Error!@import("call_result.zig").CallResult {
        const object = completion.response orelse return error.LMFailure;
        var result = try @import("call_result.zig").CallResult.native(arena, object, .{
            .model_type = lm.model_type,
            .logprobs = options.logprobs != .null,
        });
        errdefer result.free(arena);
        if (completion.billing) |bill| {
            const cost = sdk.PayloadValue.fromPythonValue(arena, bill.cost) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidCallResult,
            };
            errdefer cost.free();
            const details = sdk.PayloadValue.fromPythonValue(arena, bill.details) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidCallResult,
            };
            result.cost.free();
            result.cost = cost;
            result.cost_details.free();
            result.cost_details = details;
        }
        if (completion.cache_hit) {
            result.usage.root.object.clearRetainingCapacity();
            result.cache_hit = true;
        }
        return result;
    }

    /// All `request.options.n` completions; a provider that returns another
    /// count fails with `LMFailure`.
    pub fn completeResponses(lm: LM, arena: Allocator, request: Request) Error![]const Completion {
        assert(request.options.n >= 1);
        if (request.execution != null) return lm.dispatch(arena, request);
        assert(request.messages.len > 0 or request.decision != null);
        var execution: Execution = .{ .pricing = lm.pricing, .model_type = lm.model_type };
        return lm.execute(arena, request, &execution);
    }

    fn execute(lm: LM, arena: Allocator, request: Request, execution: *Execution) Error![]const Completion {
        assert(request.messages.len > 0 or request.decision != null or execution.canonical != null);
        assert(request.options.n >= 1);
        assert(request.execution == null);
        if (request.usage_failure) |failure| failure.* = .{};
        errdefer if (execution.result) |*result| result.free(arena);
        var delegated = request;
        delegated.execution = execution;
        delegated.options.logprobs = (try checkedUsage(arena, .{ .input_tokens = request.options.logprobs })).input_tokens;
        const completions = lm.dispatch(arena, delegated) catch |err| {
            if (execution.engine) |engine|
                engine.accountFailure(arena, request, execution.completions[0..execution.written]);
            return err;
        };
        // Cache storage and recording have completed. A hit has no physical
        // completions, so its original SDK metadata never incurs another charge.
        if (execution.engine) |engine|
            try engine.accountSuccess(arena, request.usage_tracker, execution.completions[0..execution.written]);
        return completions;
    }

    fn dispatch(lm: LM, arena: Allocator, request: Request) Error![]const Completion {
        if (request.decision != null) {
            // System One has no canonical multiple-response representation.
            if (!lm.supports_decision_requests or request.options.n != 1) return error.LMFailure;
        }
        if (lm.vtable.handles_candidate_groups) {
            const completions = try lm.vtable.complete(lm.ptr, arena, request);
            if (completions.len != request.options.n) return error.LMFailure;
            try lm.finalizeResult(arena, request, completions);
            return completions;
        }
        return lm.executeCandidates(arena, request);
    }

    fn executeCandidates(lm: LM, arena: Allocator, request: Request) Error![]const Completion {
        const execution = request.execution.?;
        assert(execution.engine == null);
        execution.engine = lm;
        // Engines may return immutable headers. Own the execution headers before
        // attaching advisory facts; never mutate a custom engine's storage.
        const out = try arena.alloc(Completion, request.options.n);
        execution.completions = out;
        var written: u32 = 0;
        for (out, 0..) |*completion, index| {
            var candidate = request;
            candidate.options.n = 1;
            var identity = [_]@import("runtime_value.zig").Value{
                .{ .string = "dspy.adapter.candidate" },
                request.options.rollout_id orelse .null,
                .{ .integer = @intCast(index) },
            };
            var rollout: @import("runtime_value.zig").Array = .init(arena);
            rollout.items = &identity;
            rollout.capacity = identity.len;
            if (execution.canonical) |canonical| {
                completion.* = try lm.completeSdkCandidate(arena, canonical, @intCast(index));
            } else {
                if (request.options.n > 1) candidate.options.rollout_id = .{ .array = rollout };
                completion.* = try lm.completeCandidate(arena, candidate);
            }
            completion.cache_hit = false;
            if (execution.pricing) |pricing| completion.billing = lm.price(pricing, completion);
            written += 1;
            execution.written = written;
            try notifyRecordings(candidate, out[index..][0..1]);
        }
        assert(written == out.len);
        try lm.finalizeResult(arena, request, out);
        return out;
    }

    fn completeSdkCandidate(lm: LM, arena: Allocator, base: *const sdk.Request, index: u32) Error!Completion {
        const complete_canonical = lm.vtable.complete_canonical orelse return error.LMFailure;
        if (index == 0 or base.fields.config.seed == .null)
            return complete_canonical(lm.ptr, arena, base);

        var scratch: std.heap.ArenaAllocator = .init(arena);
        defer scratch.deinit();
        const seed = try candidateSeed(scratch.allocator(), base.fields.config.seed, index);
        const candidate = base.withSeed(arena, seed) catch |err|
            return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
        defer candidate.release();
        return complete_canonical(lm.ptr, arena, candidate);
    }

    fn candidateSeed(arena: Allocator, seed: @import("runtime_value.zig").Value, index: u32) Error!@import("runtime_value.zig").Value {
        if (seed == .integer) {
            const sum = @addWithOverflow(seed.integer, @as(i64, index));
            if (sum[1] == 0) return .{ .integer = sum[0] };
        }
        return @import("value.zig").numericSum(arena, seed, .{ .integer = index }) catch |err|
            return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
    }

    fn finalizeResult(lm: LM, arena: Allocator, request: Request, completions: []const Completion) Error!void {
        const execution = request.execution.?;
        if (!execution.want_result or execution.result != null) return;
        var client = lm;
        client.model_type = execution.model_type;
        // Physical groups finalize before cache publication. A cached group
        // finalizes before returning through any enclosing cache wrapper.
        execution.result = try client.resultsFromCompletions(arena, completions, request.options);
    }

    fn price(lm: LM, pricing: Pricing, completion: *const Completion) ?@import("lm_cost.zig").Estimate {
        const model = lm.modelName() orelse return null;
        const facts: @import("lm_cost.zig").Facts = .{
            .provider = pricing.provider,
            .requested_model = model,
            .response_model = completion.metadata.model,
            .usage = completion.metadata.usage,
            .provider_data = if (completion.provider_data) |data| data.toValue(pricing.arena) catch return null else null,
            .namespaces = pricing.namespaces,
            .generated_media = completion.generated_media,
        };
        // Pricing failure cannot invalidate generation or required recording.
        return @import("lm_cost.zig").estimate(pricing.arena, pricing.catalogue, &facts) catch null;
    }

    fn completeCandidate(lm: LM, arena: Allocator, request: Request) Error!Completion {
        assert(request.options.n == 1);
        var physical = request;
        // A custom engine can delegate internally without charging or recording
        // the same public completion twice. These pointers never reach engines.
        physical.usage_tracker = null;
        physical.usage_failure = null;
        physical.recordings = null;
        physical.execution = null;
        const sink = physical.stream_sink;
        physical.stream_sink = null;
        if (sink) |observer| {
            if (lm.vtable.complete_stream) |stream| {
                return stream(lm.ptr, arena, physical, observer);
            }
        }
        const completions = try lm.vtable.complete(lm.ptr, arena, physical);
        if (completions.len != 1) return error.LMFailure;
        return completions[0];
    }

    fn notifyRecordings(request: Request, completions: []const Completion) Error!void {
        var current = request.recordings;
        var depth: u32 = 0;
        while (current) |pending| : (depth += 1) {
            // Recording wrappers are a trusted, acyclic stack-local chain.
            assert(depth < 1000);
            try pending.recorder.record(request, completions);
            current = pending.parent;
        }
    }

    fn accountSuccess(lm: LM, arena: Allocator, tracker: ?*accounting.UsageTracker, completions: []const Completion) Error!void {
        const target = tracker orelse return;
        var total: accounting.Graph.Value = .null;
        for (completions) |completion| {
            const usage = try completion.metadata.usage.dspyValue(arena);
            if (usage.object.count() == 0) continue;
            total = try accounting.merge(arena, total, try accounting.Graph.fromValue(arena, usage));
        }
        if (total == .null) return;
        try target.addOwnedUsage(lm.modelName() orelse return error.MissingUsageModel, total);
    }

    fn accountFailure(lm: LM, arena: Allocator, request: Request, completions: []const Completion) void {
        // A secondary accounting error must not replace the engine/recording
        // failure. Still attempt every retained success, in candidate order.
        if (request.usage_tracker == null) return;
        for (completions) |completion| {
            lm.accountSuccess(arena, request.usage_tracker, (&completion)[0..1]) catch |err| {
                if (request.usage_failure) |failure| {
                    if (failure.first == null) failure.first = err;
                    failure.count +|= 1;
                }
            };
        }
    }
};

/// Returns canned responses in order; fails once they run out.
pub const ScriptedLM = struct {
    responses: []const []const u8,
    /// Configured accounting identity, independent of response wire metadata.
    model: []const u8 = "scripted",
    /// Optional native summaries, parallel to `responses`.
    reasonings: []const ?[]const u8 = &.{},
    tool_calls: []const []const ToolCall = &.{},
    citations: []const []const CitationPart = &.{},
    continuation: []const []const Continuation = &.{},
    provider_data: []const ?@import("runtime_value.zig").Value = &.{},
    /// Optional canonical metadata, parallel to `responses` and independent
    /// of `provider_data`; nothing is derived from the raw provider JSON.
    metadata: []const ResponseMetadata = &.{},
    /// Validated generated-media facts, parallel to `responses`.
    generated_media: []const bool = &.{},
    /// Optional canonical scores and completeness, parallel to `responses`.
    logprobs: []const ?[]const TokenLogprob = &.{},
    logprobs_complete: []const bool = &.{},
    response_messages: []const ?sdk.Message = &.{},
    adaptations: []const []const sdk.Adaptation = &.{},
    supports_reasoning: bool = false,
    supports_response_format: bool = false,
    supports_response_schema: bool = false,
    supports_function_calling: bool = false,
    supports_decision_requests: bool = false,
    index: usize = 0,
    /// Optional transcript of every request (deep-copied into its arena).
    history: ?*History = null,

    pub fn lm(self: *ScriptedLM) LM {
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete, .model_name = modelName },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_format,
            .supports_response_schema = self.supports_response_schema,
            .supports_function_calling = self.supports_function_calling,
            .supports_decision_requests = self.supports_decision_requests,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *ScriptedLM = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    /// The next `n` canned responses.
    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *ScriptedLM = @ptrCast(@alignCast(ptr));
        if (self.history) |h| try h.record(request);
        const n = request.options.n;
        if (self.responses.len - self.index < n) return error.ScriptExhausted;
        if (self.reasonings.len != 0 and self.reasonings.len != self.responses.len) return error.LMFailure;
        if (self.tool_calls.len != 0 and self.tool_calls.len != self.responses.len) return error.LMFailure;
        if (self.citations.len != 0 and self.citations.len != self.responses.len) return error.LMFailure;
        if (self.continuation.len != 0 and self.continuation.len != self.responses.len) return error.LMFailure;
        if (self.provider_data.len != 0 and self.provider_data.len != self.responses.len) return error.LMFailure;
        if (self.metadata.len != 0 and self.metadata.len != self.responses.len) return error.LMFailure;
        if (self.generated_media.len != 0 and self.generated_media.len != self.responses.len) return error.LMFailure;
        if (self.logprobs.len != 0 and self.logprobs.len != self.responses.len) return error.LMFailure;
        if (self.logprobs_complete.len != 0 and self.logprobs_complete.len != self.responses.len) return error.LMFailure;
        if (self.response_messages.len != 0 and self.response_messages.len != self.responses.len) return error.LMFailure;
        if (self.adaptations.len != 0 and self.adaptations.len != self.responses.len) return error.LMFailure;
        const out = try arena.alloc(Completion, n);
        for (out, self.responses[self.index..][0..n], self.index..) |*o, r, i| {
            const reasoning = if (self.reasonings.len != 0) self.reasonings[i] else null;
            const provider_data = if (self.provider_data.len != 0) self.provider_data[i] else null;
            if (provider_data) |data| {
                if (data != .null and (data != .object or !providerJsonValue(data, 0))) return error.LMFailure;
            }
            o.* = try cloneCompletion(arena, .{
                .text = r,
                .reasoning = reasoning,
                .tool_calls = if (self.tool_calls.len != 0) self.tool_calls[i] else &.{},
                .citations = if (self.citations.len != 0) self.citations[i] else &.{},
                .continuation = if (self.continuation.len != 0) self.continuation[i] else &.{},
                .provider_data = if (provider_data) |data| try jsonPayload(arena, data) else null,
                .metadata = if (self.metadata.len != 0) try checkedMetadata(arena, self.metadata[i]) else .{},
                .generated_media = self.generated_media.len != 0 and self.generated_media[i],
                .message = if (self.response_messages.len != 0) self.response_messages[i] else null,
                .adaptations = if (self.adaptations.len != 0) self.adaptations[i] else &.{},
            });
            if (self.logprobs.len != 0) {
                if (self.logprobs[i]) |scores| o.logprobs = logprobs.checked(arena, scores) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.InvalidLogprobs => return error.LMFailure,
                };
            }
            o.logprobs_complete = self.logprobs_complete.len == 0 or self.logprobs_complete[i];
            try publishResponse(arena, o, self.model, .owned);
        }
        self.index += n;
        return out;
    }
};

/// Keyword-rule mock: the first rule whose needle occurs in its scope wins.
/// Deterministic and stateless, so it gives identical answers to Zig and to
/// the Python mirror used for cross-runtime optimizer checks.
pub const KeywordLM = struct {
    rules: []const Rule,
    default: []const u8,
    history: ?*History = null,

    pub const Scope = enum {
        /// Content of the final user message (the current request).
        last_user,
        /// Every message, demos included.
        any,
    };

    pub const Rule = struct { needle: []const u8, response: []const u8, scope: Scope = .last_user };

    pub fn lm(self: *KeywordLM) LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    pub fn respond(self: *const KeywordLM, messages: []const Message) []const u8 {
        assert(messages.len > 0);
        const last = messages[messages.len - 1].content;
        for (self.rules) |rule| {
            switch (rule.scope) {
                .last_user => if (last.contains(rule.needle)) return rule.response,
                .any => for (messages) |m| {
                    if (m.content.contains(rule.needle)) return rule.response;
                },
            }
        }
        return self.default;
    }

    /// `n` copies of the matching rule's response.
    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *KeywordLM = @ptrCast(@alignCast(ptr));
        if (request.decision != null) return error.LMFailure;
        if (self.history) |h| try h.record(request);
        const out = try arena.alloc(Completion, request.options.n);
        const response = try arena.dupe(u8, self.respond(request.messages));
        for (out) |*o| o.* = .{ .text = response };
        return out;
    }
};

/// Replays a recorded transcript (e.g. real-model calls captured from DSPy).
/// Each request must byte-match an unused recorded request (roles and
/// contents); the first such call's completion is returned. A request with
/// no recorded twin fails with `LMFailure`, so a replay that completes with
/// `exhausted()` proves both runtimes sent identical prompts.
pub const ReplayLM = struct {
    calls: []const Call,
    /// The replay client's configured identity, not a recorded wire model.
    model: []const u8 = "replay",
    /// Flat flags for every completion in `calls`, not one flag per batch.
    /// Size with `completionCount` to support both split and grouped requests.
    used: []bool,
    misses: usize = 0,
    history: ?*History = null,
    supports_reasoning: bool = false,
    supports_response_format: bool = false,
    supports_response_schema: bool = false,
    supports_function_calling: bool = false,
    supports_decision_requests: bool = false,

    pub const Call = struct {
        messages: []const Message,
        /// Null only for historical transcripts that did not record options.
        options: ?Options = null,
        tools: []const FunctionTool = &.{},
        decision: ?DecisionRequest = null,
        completion: ?[]const u8,
        /// Every completion of an `n > 1` call (`completion` is the first);
        /// empty for single-completion calls. Null entries preserve absent text.
        completions: []const ?[]const u8 = &.{},
        /// Native summaries, parallel to `all()`; empty when none were returned.
        reasonings: []const ?[]const u8 = &.{},
        /// Native calls for each completion, parallel to `all()`.
        tool_calls: []const []const ToolCall = &.{},
        citations: []const []const CitationPart = &.{},
        continuation: []const []const Continuation = &.{},
        provider_data: []const ?@import("runtime_value.zig").Value = &.{},
        /// Canonical metadata, parallel to `all()`; empty for transcripts
        /// which recorded none, whose completions then stay unknown.
        metadata: []const ResponseMetadata = &.{},
        /// Generated-media facts, parallel to `all()`.
        generated_media: []const bool = &.{},
        /// Canonical token telemetry, parallel to `all()`.
        logprobs: []const ?[]const TokenLogprob = &.{},
        logprobs_complete: []const bool = &.{},
        response_messages: []const ?sdk.Message = &.{},
        adaptations: []const []const sdk.Adaptation = &.{},
        /// Advisory billing for this recorded physical request, not SDK usage
        /// or a replay charge. Null means no estimate was retained.
        billing: ?@import("lm_cost.zig").Estimate = null,

        pub fn all(call: *const Call) []const ?[]const u8 {
            if (call.completions.len > 0) return call.completions;
            return (&call.completion)[0..1];
        }
    };

    pub fn completionCount(calls: []const Call) usize {
        var count: usize = 0;
        for (calls) |*call| count += call.all().len;
        return count;
    }

    pub fn lm(self: *ReplayLM) LM {
        assert(completionCount(self.calls) == self.used.len);
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete, .model_name = modelName },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_format,
            .supports_response_schema = self.supports_response_schema,
            .supports_function_calling = self.supports_function_calling,
            .supports_decision_requests = self.supports_decision_requests,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *ReplayLM = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    pub fn exhausted(self: *const ReplayLM) bool {
        for (self.used) |u| if (!u) return false;
        return self.misses == 0;
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *ReplayLM = @ptrCast(@alignCast(ptr));
        if (self.history) |h| try h.record(request);
        const out = try arena.alloc(Completion, request.options.n);
        var written: usize = 0;
        var offset: usize = 0;
        for (self.calls) |*call| {
            const recorded = call.all();
            const first = offset;
            offset += recorded.len;
            if (!(try sameMessages(call.messages, request.messages))) continue;
            if (!(try inference.toolsEqual(call.tools, request.tools))) continue;
            if (!(try decisionEqual(call.decision, request.decision))) continue;
            if (call.options) |options| {
                if (!(try sameOptions(options, request.options))) continue;
            }
            if (call.reasonings.len != 0 and call.reasonings.len != recorded.len) return error.LMFailure;
            if (call.tool_calls.len != 0 and call.tool_calls.len != recorded.len) return error.LMFailure;
            if (call.citations.len != 0 and call.citations.len != recorded.len) return error.LMFailure;
            if (call.continuation.len != 0 and call.continuation.len != recorded.len) return error.LMFailure;
            if (call.provider_data.len != 0 and call.provider_data.len != recorded.len) return error.LMFailure;
            if (call.metadata.len != 0 and call.metadata.len != recorded.len) return error.LMFailure;
            if (call.generated_media.len != 0 and call.generated_media.len != recorded.len) return error.LMFailure;
            if (call.logprobs.len != 0 and call.logprobs.len != recorded.len) return error.LMFailure;
            if (call.logprobs_complete.len != 0 and call.logprobs_complete.len != recorded.len) return error.LMFailure;
            if (call.response_messages.len != 0 and call.response_messages.len != recorded.len) return error.LMFailure;
            if (call.adaptations.len != 0 and call.adaptations.len != recorded.len) return error.LMFailure;
            for (recorded, 0..) |text, i| {
                if (self.used[first + i]) continue;
                out[written] = try cloneCompletion(arena, .{
                    .text = text,
                    .reasoning = if (call.reasonings.len != 0) call.reasonings[i] else null,
                    .tool_calls = if (call.tool_calls.len != 0) call.tool_calls[i] else &.{},
                    .citations = if (call.citations.len != 0) call.citations[i] else &.{},
                    .continuation = if (call.continuation.len != 0) call.continuation[i] else &.{},
                    .provider_data = if (call.provider_data.len != 0 and call.provider_data[i] != null)
                        try jsonPayload(arena, call.provider_data[i].?)
                    else
                        null,
                    .metadata = if (call.metadata.len != 0) try checkedMetadata(arena, call.metadata[i]) else .{},
                    .generated_media = call.generated_media.len != 0 and call.generated_media[i],
                    .message = if (call.response_messages.len != 0) call.response_messages[i] else null,
                    .adaptations = if (call.adaptations.len != 0) call.adaptations[i] else &.{},
                    .billing = if (i == 0) call.billing else null,
                });
                if (call.logprobs.len != 0) {
                    if (call.logprobs[i]) |scores| out[written].logprobs = logprobs.checked(arena, scores) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.InvalidLogprobs => return error.LMFailure,
                    };
                }
                out[written].logprobs_complete = call.logprobs_complete.len == 0 or call.logprobs_complete[i];
                try publishResponse(arena, &out[written], self.model, .owned);
                self.used[first + i] = true;
                written += 1;
                if (written == out.len) return out;
            }
        }
        self.misses += 1;
        return error.LMFailure;
    }

    fn sameOptions(a: Options, b: Options) @import("runtime_value.zig").CustomError!bool {
        if (a.temperature != b.temperature or
            a.parallel_tool_calls != b.parallel_tool_calls or a.n != b.n or
            a.reasoning_effort_present != b.reasoning_effort_present) return false;
        if (!(try inference.jsonEqual(a.rollout_id orelse .null, b.rollout_id orelse .null))) return false;
        if (!(try inference.jsonEqual(a.logprobs, b.logprobs))) return false;
        if (a.response_format) |format| {
            if (!(try inference.responseFormatEqual(format, b.response_format orelse return false))) return false;
        } else if (b.response_format != null) return false;
        if (a.tool_choice) |choice| {
            const other = b.tool_choice orelse return false;
            if (!inference.toolChoiceEqual(&choice, &other)) return false;
        } else if (b.tool_choice != null) return false;
        if (a.reasoning_effort) |effort| {
            return std.mem.eql(u8, effort, b.reasoning_effort orelse return false);
        }
        return b.reasoning_effort == null;
    }

    fn sameMessages(a: []const Message, b: []const Message) @import("runtime_value.zig").CustomError!bool {
        if (a.len != b.len) return false;
        for (a, b) |x, y| {
            if (x.role != y.role or !(try adapter.Content.eql(x.content, y.content))) return false;
            if (!optionalStringEqual(x.tool_call_id, y.tool_call_id)) return false;
            if (!optionalStringEqual(x.reasoning, y.reasoning)) return false;
            if (!(try inference.toolCallsEqual(x.tool_calls, y.tool_calls))) return false;
            if (!(try inference.continuationsEqual(x.continuation, y.continuation))) return false;
        }
        return true;
    }
};

/// Wraps another LM and keeps every request with its completion, e.g. to
/// capture real-model traces for later `ReplayLM` runs.
pub const RecordingLM = struct {
    inner: LM,
    /// Owns the recorded copies.
    arena: Allocator,
    calls: std.ArrayList(ReplayLM.Call) = .empty,

    const Pending = struct { recorder: *RecordingLM, parent: ?*const Pending };

    pub fn lm(self: *RecordingLM) LM {
        return .{
            .ptr = self,
            .vtable = if (self.inner.vtable.model_name == null)
                &.{ .complete = complete, .handles_candidate_groups = true }
            else
                &.{ .complete = complete, .handles_candidate_groups = true, .model_name = modelName },
            .supports_reasoning = self.inner.supports_reasoning,
            .supports_response_format = self.inner.supports_response_format,
            .supports_response_schema = self.inner.supports_response_schema,
            .supports_function_calling = self.inner.supports_function_calling,
            .supports_decision_requests = self.inner.supports_decision_requests,
            .pricing = self.inner.pricing,
            .model_type = self.inner.model_type,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *RecordingLM = @ptrCast(@alignCast(ptr));
        return self.inner.modelName().?;
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *RecordingLM = @ptrCast(@alignCast(ptr));
        if (request.execution) |execution| {
            if (execution.canonical != null) return error.LMFailure;
        }
        const pending: Pending = .{ .recorder = self, .parent = request.recordings };
        var delegated = request;
        delegated.recordings = &pending;
        return self.inner.completeResponses(arena, delegated);
    }

    fn record(self: *RecordingLM, request: Request, completions: []const Completion) Error!void {
        assert(completions.len == 1);
        const messages = try self.arena.alloc(Message, request.messages.len);
        for (request.messages, messages) |src, *dst| dst.* = try cloneMessage(self.arena, src);
        const kept = try self.arena.alloc(?[]const u8, completions.len);
        var has_reasoning = false;
        for (kept, completions) |*k, c| {
            k.* = if (c.text) |text| try self.arena.dupe(u8, text) else null;
            has_reasoning = has_reasoning or c.reasoning != null;
        }
        const reasonings: []?[]const u8 = if (has_reasoning) try self.arena.alloc(?[]const u8, completions.len) else &.{};
        for (reasonings, completions[0..reasonings.len]) |*kept_reasoning, c| {
            kept_reasoning.* = if (c.reasoning) |text| try self.arena.dupe(u8, text) else null;
        }
        var has_tool_calls = false;
        for (completions) |completion| has_tool_calls = has_tool_calls or completion.tool_calls.len != 0;
        const tool_calls: [][]const ToolCall = if (has_tool_calls) try self.arena.alloc([]const ToolCall, completions.len) else &.{};
        for (tool_calls, completions[0..tool_calls.len]) |*kept_calls, completion| {
            kept_calls.* = try inference.cloneToolCalls(self.arena, completion.tool_calls);
        }
        var has_citations = false;
        for (completions) |completion| has_citations = has_citations or completion.citations.len != 0;
        const citations: [][]const CitationPart = if (has_citations) try self.arena.alloc([]const CitationPart, completions.len) else &.{};
        for (citations, completions[0..citations.len]) |*kept_citations, completion| {
            kept_citations.* = try inference.cloneCitations(self.arena, completion.citations);
        }
        var has_continuation = false;
        for (completions) |completion| has_continuation = has_continuation or completion.continuation.len != 0;
        const continuation: [][]const Continuation = if (has_continuation) try self.arena.alloc([]const Continuation, completions.len) else &.{};
        for (continuation, completions[0..continuation.len]) |*kept_state, completion| {
            kept_state.* = try inference.cloneContinuations(self.arena, completion.continuation);
        }
        var has_provider_data = false;
        for (completions) |completion| has_provider_data = has_provider_data or completion.provider_data != null;
        const provider_data: []?@import("runtime_value.zig").Value = if (has_provider_data) try self.arena.alloc(?@import("runtime_value.zig").Value, completions.len) else &.{};
        for (provider_data, completions[0..provider_data.len]) |*kept_data, completion| {
            kept_data.* = if (completion.provider_data) |data| data.toValue(self.arena) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.LMFailure,
            } else null;
        }
        var has_metadata = false;
        for (completions) |completion| has_metadata = has_metadata or completion.metadata.known();
        const metadata: []ResponseMetadata = if (has_metadata) try self.arena.alloc(ResponseMetadata, completions.len) else &.{};
        for (metadata, completions[0..metadata.len]) |*kept_metadata, completion| {
            kept_metadata.* = try cloneResponseMetadata(self.arena, completion.metadata);
        }
        var has_media = false;
        for (completions) |completion| has_media = has_media or completion.generated_media;
        const generated_media: []bool = if (has_media) try self.arena.alloc(bool, completions.len) else &.{};
        for (generated_media, completions[0..generated_media.len]) |*kept_media, completion| {
            kept_media.* = completion.generated_media;
        }
        var has_scores = false;
        var incomplete_scores = false;
        for (completions) |completion| {
            has_scores = has_scores or completion.logprobs != null;
            incomplete_scores = incomplete_scores or !completion.logprobs_complete;
        }
        const token_scores: []?[]const TokenLogprob = if (has_scores) try self.arena.alloc(?[]const TokenLogprob, completions.len) else &.{};
        for (token_scores, completions[0..token_scores.len]) |*kept_scores, completion| {
            kept_scores.* = if (completion.logprobs) |scores| try logprobs.clone(self.arena, scores) else null;
        }
        const scores_complete: []bool = if (incomplete_scores) try self.arena.alloc(bool, completions.len) else &.{};
        for (scores_complete, completions[0..scores_complete.len]) |*kept_complete, completion| {
            kept_complete.* = completion.logprobs_complete;
        }
        var has_message = false;
        var has_adaptations = false;
        for (completions) |completion| {
            has_message = has_message or completionMessage(&completion) != null;
            has_adaptations = has_adaptations or completion.adaptations.len != 0;
        }
        const response_messages: []?sdk.Message = if (has_message)
            try self.arena.alloc(?sdk.Message, completions.len)
        else
            &.{};
        for (response_messages, completions[0..response_messages.len]) |*kept_message, completion| {
            kept_message.* = if (completionMessage(&completion)) |message| try message.clone(self.arena) else null;
        }
        const adaptations: [][]const sdk.Adaptation = if (has_adaptations)
            try self.arena.alloc([]const sdk.Adaptation, completions.len)
        else
            &.{};
        for (adaptations, completions[0..adaptations.len]) |*kept_adaptations, completion| {
            kept_adaptations.* = try sdk.cloneAdaptations(self.arena, completion.adaptations);
        }
        try self.calls.append(self.arena, .{
            .messages = messages,
            .options = try cloneOptions(self.arena, request.options),
            .tools = try inference.cloneTools(self.arena, request.tools),
            .decision = if (request.decision) |decision| try cloneDecision(self.arena, decision) else null,
            .completion = kept[0],
            .completions = if (kept.len > 1) kept else &.{},
            .reasonings = reasonings,
            .tool_calls = tool_calls,
            .citations = citations,
            .continuation = continuation,
            .provider_data = provider_data,
            .metadata = metadata,
            .generated_media = generated_media,
            .logprobs = token_scores,
            .logprobs_complete = scores_complete,
            .response_messages = response_messages,
            .adaptations = adaptations,
            // Known billing is required evidence, not another price estimate.
            .billing = if (completions[0].billing) |bill| try cloneBilling(self.arena, bill) else null,
        });
    }
};

/// Transcript of LM requests, for golden comparisons and inspection.
pub const History = struct {
    arena: Allocator,
    requests: std.ArrayList(Request) = .empty,

    pub fn record(h: *History, request: Request) Allocator.Error!void {
        const messages = try h.arena.alloc(Message, request.messages.len);
        for (request.messages, messages) |src, *dst| dst.* = try cloneMessage(h.arena, src);
        try h.requests.append(h.arena, .{
            .messages = messages,
            .options = try cloneOptions(h.arena, request.options),
            .tools = try inference.cloneTools(h.arena, request.tools),
            .decision = if (request.decision) |decision| try cloneDecision(h.arena, decision) else null,
        });
    }

    pub fn last(h: *const History) ?Request {
        if (h.requests.items.len == 0) return null;
        return h.requests.items[h.requests.items.len - 1];
    }
};

/// Project a completed SDK response onto the same events as live streaming.
pub const responseToEvents = @import("openai_stream.zig").responseToEvents;

pub const MaterializedStream = @import("openai_stream.zig").Materialized;

/// Fold a borrowed sequence of typed canonical events; no HTTP or cache.
pub fn materializeStream(
    backing: Allocator,
    request: *const sdk.Request,
    events: []const OpenAI.StreamEvent,
) Error!MaterializedStream {
    return OpenAI.OpenAIStream.materialize(backing, request, events);
}

pub const StreamSource = @import("openai_stream.zig").Source;
pub const StreamSourceResult = @import("openai_stream.zig").SourceResult;

/// Assemble a synchronous event source and call close exactly once.
/// An observer may stop before the end without pulling another event.
pub fn materializeStreamSource(
    backing: Allocator,
    request: *const sdk.Request,
    source: StreamSource,
    observer: ?OpenAI.StreamSink,
) StreamSourceResult {
    return OpenAI.OpenAIStream.materializeSource(backing, request, source, observer);
}

test "Azure OpenAI v1 keeps gateway prefixes and refuses ambiguous endpoint URLs" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cases = [_]struct { endpoint: []const u8, base: []const u8 }{
        .{ .endpoint = "https://account.services.ai.azure.com", .base = "https://account.services.ai.azure.com/openai/v1" },
        .{ .endpoint = " http://localhost:1234/gateway//openai/ ", .base = "http://localhost:1234/gateway/openai/v1" },
        .{ .endpoint = "https://account.openai.azure.com/openai/v1/", .base = "https://account.openai.azure.com/openai/v1" },
    };
    for (cases) |case| {
        try testing.expectEqualStrings(case.base, try OpenAI.azureBaseUrl(arena, case.endpoint));
    }
    for ([_][]const u8{
        "ftp://azure.invalid",
        "https:///path",
        "https://user@azure.invalid",
        "https://azure.invalid?api-version=2024",
        "https://azure.invalid/#fragment",
    }) |endpoint| {
        try testing.expectError(error.InvalidAzureEndpoint, OpenAI.azureBaseUrl(arena, endpoint));
    }
}

/// OpenAI-compatible Chat Completions and Responses clients on `std.http`.
pub const OpenAI = struct {
    client: *std.http.Client,
    /// e.g. "https://api.openai.com/v1"; a trailing slash is accepted.
    base_url: []const u8,
    model: []const u8,
    /// Training configuration is borrowed until persisted or startup-restored.
    finetuning_model: ?[]const u8 = null,
    train_kwargs: @import("runtime_value.zig").Value = .{ .object = .empty },
    launch_kwargs: @import("runtime_value.zig").Value = .{ .object = .empty },
    api: Api = .chat_completions,
    /// Explicit capabilities, not inferred from model-name substrings.
    supports_reasoning: bool = false,
    supports_response_format: bool = true,
    supports_response_schema: bool = false,
    supports_function_calling: bool = false,
    /// Bearer token; never logged.
    api_key: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    /// Upper bound enforced while reading each response body.
    max_response_bytes: usize = 4 * 1024 * 1024,
    /// Local-media reads have an explicit bound, unlike Source read_bytes.
    max_media_bytes: usize = 16 * 1024 * 1024,
    adaptations: @import("openai_request.zig").Adaptations = .note,
    /// Chat servers such as xAI report reasoning outside completion_tokens.
    /// Responses always uses its own inclusive output_tokens counter.
    chat_reasoning_usage: @FieldType(@import("openai_request.zig").Compat, "reasoning_usage") = .in_completion,
    /// When bound, its model policy replaces the standalone Chat usage knob.
    chat_compat: ?ChatCompat = null,
    /// Partial bound Responses policy, borrowing caller-owned graph roots.
    /// Request overrides are applied afterward; Chat ignores this policy.
    responses_compat: ?ResponsesCompat = null,
    /// Extra attempts after a transport error, HTTP 429 or 5xx (DSPy's
    /// `num_retries`). Backoff doubles from `retry_base_ms`.
    max_retries: u8 = 3,
    retry_base_ms: u32 = 500,
    /// Per-network-operation inactivity limit; null disables it.
    timeout_ns: ?u64 = null,
    /// DNS/TCP limit; null inherits timeout_ns.
    connect_timeout_ns: ?u64 = null,
    /// Authentication providers are refreshed on every attempt, not per call.
    header_provider: ?http.HeaderProvider = null,

    pub const Api = enum { chat_completions, responses };
    pub const ResponsesCompat = @import("openai_request.zig").ResponsesCompat;
    pub const ChatCompat = @import("openai_request.zig").ChatCompat;

    pub const AzureOptions = struct {
        endpoint: []const u8,
        deployment: []const u8,
        api: Api = .responses,
        credential: union(enum) {
            api_key: []const u8,
            /// Supplies Authorization per attempt. Token acquisition is caller-owned.
            entra: http.HeaderProvider,
        },
    };

    /// Bind the Azure OpenAI v1 generation endpoint. The arena and borrowed
    /// deployment/credentials must outlive the provider, including LM copies.
    pub fn initAzure(
        arena: Allocator,
        client: *std.http.Client,
        options: AzureOptions,
    ) (Allocator.Error || error{ InvalidAzureEndpoint, InvalidAzureCredential, EmptyDeployment })!OpenAI {
        if (options.deployment.len == 0) return error.EmptyDeployment;
        if (options.credential == .api_key) {
            const key = options.credential.api_key;
            if (key.len == 0 or std.mem.indexOfAny(u8, key, "\r\n") != null)
                return error.InvalidAzureCredential;
        }
        const base_url = try azureBaseUrl(arena, options.endpoint);
        errdefer arena.free(base_url);
        const provider: http.HeaderProvider = switch (options.credential) {
            .entra => |provider| provider,
            .api_key => |key| blk: {
                const header = try arena.create(std.http.Header);
                header.* = .{ .name = "api-key", .value = key };
                break :blk .{ .context = header, .prepare = azureKeyHeader };
            },
        };
        return .{
            .client = client,
            .base_url = base_url,
            .model = options.deployment,
            .api = options.api,
            .header_provider = provider,
        };
    }

    fn azureBaseUrl(arena: Allocator, endpoint: []const u8) ![]const u8 {
        const trimmed = std.mem.trim(u8, endpoint, " \t\r\n");
        const uri = std.Uri.parse(trimmed) catch return error.InvalidAzureEndpoint;
        const scheme_end = std.mem.indexOf(u8, trimmed, "://") orelse
            return error.InvalidAzureEndpoint;
        if (!std.ascii.eqlIgnoreCase(uri.scheme, "http") and
            !std.ascii.eqlIgnoreCase(uri.scheme, "https")) return error.InvalidAzureEndpoint;
        if (uri.host == null or uri.host.?.isEmpty() or uri.user != null or uri.password != null or
            uri.query != null or uri.fragment != null) return error.InvalidAzureEndpoint;
        for (trimmed) |byte| {
            if (byte <= ' ' or byte == 0x7f or byte == '\\') return error.InvalidAzureEndpoint;
        }
        const authority_start = scheme_end + 3;
        const authority_end = std.mem.indexOfScalarPos(u8, trimmed, authority_start, '/') orelse
            trimmed.len;
        const path = trimmed[authority_end..];
        // Find the longest overlap with the two-segment Azure door, ignoring
        // redundant slashes without decoding the caller's path segments.
        var segments = std.mem.tokenizeScalar(u8, path, '/');
        var previous: []const u8 = "";
        var last: []const u8 = "";
        var path_len: usize = 0;
        while (segments.next()) |segment| {
            path_len += 1 + segment.len;
            previous = last;
            last = segment;
        }
        const suffix: []const u8 = if (std.mem.eql(u8, last, "openai"))
            "/v1"
        else if (std.mem.eql(u8, previous, "openai") and std.mem.eql(u8, last, "v1"))
            ""
        else
            "/openai/v1";

        const len = std.math.add(usize, authority_end + path_len, suffix.len) catch
            return error.InvalidAzureEndpoint;
        const output = try arena.alloc(u8, len);
        var writer: Writer = .fixed(output);
        writer.writeAll(if (uri.scheme.len == 5) "https://" else "http://") catch unreachable;
        writer.writeAll(trimmed[authority_start..authority_end]) catch unreachable;
        segments.reset();
        while (segments.next()) |segment| {
            writer.writeByte('/') catch unreachable;
            writer.writeAll(segment) catch unreachable;
        }
        writer.writeAll(suffix) catch unreachable;
        assert(writer.end == len);
        return output;
    }

    fn azureKeyHeader(
        context: *anyopaque,
        arena: Allocator,
        headers: []const std.http.Header,
    ) Allocator.Error![]const std.http.Header {
        const header: *const std.http.Header = @ptrCast(@alignCast(context));
        if (headers.len == 0) return @as(*const [1]std.http.Header, @ptrCast(header));
        for (headers) |extra| {
            if (std.ascii.eqlIgnoreCase(extra.name, header.name)) return headers;
        }
        const combined = try arena.alloc(std.http.Header, headers.len + 1);
        @memcpy(combined[0..headers.len], headers);
        combined[headers.len] = header.*;
        return combined;
    }

    /// Bind the Chat preset and its root atomically, never reusing a root
    /// previously selected for a different provider.
    pub fn bindChatPreset(self: *OpenAI, name: []const u8, base_url: ?[]const u8) error{
        UnknownChatPreset,
        MissingChatPresetBaseUrl,
    }!void {
        const wire = @import("openai_request.zig");
        const preset = try wire.chatPreset(name);
        const cloud = wire.ResponsesPreset.default_openai_url;
        const requested_url = base_url orelse cloud;
        const url = if (std.mem.eql(u8, requested_url, cloud))
            try preset.defaultUrl()
        else
            requested_url;
        self.api = .chat_completions;
        self.base_url = url;
        self.chat_compat = preset.policy().*;
    }

    /// Bind the Responses policy and its address together without consulting
    /// the old client address. Like Source, the exact OpenAI default URL
    /// selects the preset default. A non-default override is borrowed.
    pub fn bindResponsesPreset(self: *OpenAI, name: []const u8, base_url: ?[]const u8) error{
        UnknownResponsesPreset,
        MissingResponsesPresetBaseUrl,
    }!void {
        const Preset = @import("openai_request.zig").ResponsesPreset;
        const preset = try Preset.fromName(name);
        const requested_url = base_url orelse Preset.default_openai_url;
        const url = if (std.mem.eql(u8, requested_url, Preset.default_openai_url))
            try preset.defaultUrl()
        else
            requested_url;
        self.api = .responses;
        self.base_url = url;
        self.responses_compat = preset.policy().*;
    }

    /// Startup loader with explicitly supplied trust, endpoint and credentials.
    /// State loading never reads secrets or imports executable code.
    pub const StateLoader = struct {
        client: *std.http.Client,
        base_url: []const u8,
        api_key: ?[]const u8 = null,
        header_provider: ?http.HeaderProvider = null,

        pub fn loader(self: *StateLoader) LM.StateLoader {
            return .{ .ptr = self, .restore_engine = restore };
        }

        fn restore(ptr: *anyopaque, arena: Allocator, input: @import("runtime_value.zig").Value) StateError!LM {
            const self: *StateLoader = @ptrCast(@alignCast(ptr));
            if (input != .object) return error.InvalidLMState;
            const class = input.object.get("__lm_class__") orelse return error.InvalidLMState;
            if (class != .string) return error.InvalidLMState;
            if (!std.mem.eql(u8, class.string, "dszig.lm.OpenAI")) return error.UnsupportedLMState;
            const model = input.object.get("model") orelse return error.InvalidLMState;
            if (model != .string) return error.InvalidLMState;
            const target = try arena.create(OpenAI);
            target.* = .{
                .client = self.client,
                .base_url = "",
                .model = "",
                .api_key = self.api_key,
                .header_provider = self.header_provider,
            };
            inline for (persistent_fields) |name| {
                const T = @FieldType(OpenAI, name);
                var item = input.object.get(name);
                if (comptime std.mem.eql(u8, name, "base_url")) {
                    if (item == null) item = .{ .string = self.base_url };
                }
                if (item) |encoded| {
                    @field(target, name) = if (comptime T == []const u8) blk: {
                        if (encoded != .string) return error.InvalidLMState;
                        break :blk try arena.dupe(u8, encoded.string);
                    } else if (comptime T == ?[]const u8) blk: {
                        if (encoded == .null) break :blk null;
                        if (encoded != .string) return error.InvalidLMState;
                        break :blk try arena.dupe(u8, encoded.string);
                    } else if (comptime T == ?ResponsesCompat or T == ?ChatCompat) blk: {
                        if (encoded == .null) break :blk null;
                        const Policy = @typeInfo(T).optional.child;
                        const borrowed = Policy.fromValue(arena, encoded) catch |err|
                            return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidLMState;
                        break :blk try borrowed.clone(arena);
                    } else if (comptime T == @import("runtime_value.zig").Value)
                        try cloneConfigObject(arena, encoded)
                    else
                        @import("value.zig").fromValue(arena, T, encoded) catch |err| return switch (err) {
                            error.OutOfMemory => error.OutOfMemory,
                            else => error.InvalidLMState,
                        };
                }
            }
            return target.lm();
        }
    };

    // An allowlist prevents future runtime resources or credentials from
    // becoming persistent merely because someone adds an OpenAI field.
    const persistent_fields = .{
        "base_url",                 "model",                    "api",                       "supports_reasoning",
        "supports_response_format", "supports_response_schema", "supports_function_calling", "max_tokens",
        "max_response_bytes",       "max_retries",              "retry_base_ms",             "timeout_ns",
        "connect_timeout_ns",       "finetuning_model",         "train_kwargs",              "launch_kwargs",
        "max_media_bytes",          "adaptations",              "chat_reasoning_usage",      "responses_compat",
        "chat_compat",
    };

    fn dumpState(ptr: *anyopaque, arena: Allocator) StateError!@import("runtime_value.zig").Value {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        var state: @import("runtime_value.zig").ObjectMap = .empty;
        try state.ensureTotalCapacity(arena, persistent_fields.len + 1);
        state.putAssumeCapacity("__lm_class__", .{ .string = "dszig.lm.OpenAI" });
        inline for (persistent_fields) |name| {
            const T = @FieldType(OpenAI, name);
            const encoded = if (comptime T == ?ResponsesCompat or T == ?ChatCompat) blk: {
                const policy = @field(self, name) orelse break :blk @import("runtime_value.zig").Value.null;
                break :blk policy.toValue(arena) catch |err|
                    return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidLMState;
            } else if (comptime @FieldType(OpenAI, name) == @import("runtime_value.zig").Value)
                try cloneConfigObject(arena, @field(self, name))
            else
                try @import("value.zig").toValue(arena, @FieldType(OpenAI, name), @field(self, name));
            state.putAssumeCapacity(name, encoded);
        }
        return .{ .object = state };
    }

    fn cloneConfigObject(arena: Allocator, input: @import("runtime_value.zig").Value) StateError!@import("runtime_value.zig").Value {
        if (input != .object) return error.InvalidLMState;
        if (!providerJsonValue(input, 0)) return error.InvalidLMState;
        return inference.cloneJsonValue(arena, input);
    }

    /// No allocation or ownership transfer: the caller retains the model text
    /// and all resources/options borrowed from the original client.
    pub fn copyWithModel(self: *const OpenAI, target: *OpenAI, model: []const u8) void {
        target.* = self.*;
        target.model = model;
    }

    pub fn lm(self: *OpenAI) LM {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = complete,
                .complete_stream = completeStream,
                .complete_canonical = completeCanonical,
                .dump_state = dumpState,
                .finetune = startFinetune,
                .model_name = modelName,
                .clone = clone,
            },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_format,
            .supports_response_schema = self.supports_response_schema,
            .supports_function_calling = self.supports_function_calling,
        };
    }

    fn clone(ptr: *anyopaque, arena: Allocator) anyerror!LM {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        const copy = try arena.create(OpenAI);
        copy.* = self.*;
        copy.model = try arena.dupe(u8, self.model);
        copy.base_url = try arena.dupe(u8, self.base_url);
        if (self.finetuning_model) |name| copy.finetuning_model = try arena.dupe(u8, name);
        copy.train_kwargs = try @import("value.zig").cloneValue(arena, self.train_kwargs);
        copy.launch_kwargs = try @import("value.zig").cloneValue(arena, self.launch_kwargs);
        if (self.responses_compat) |policy| copy.responses_compat = try policy.clone(arena);
        if (self.chat_compat) |policy| copy.chat_compat = try policy.clone(arena);
        return copy.lm();
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    fn startFinetune(ptr: *anyopaque, job: *@import("training.zig").Job, request: @import("training.zig").Request) anyerror!void {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        try job.start(self, request);
    }

    /// Build a physical wire request from the caller's actual SDK request.
    /// Bound-model defaults belong only to the native prediction adapter.
    pub fn requestBody(self: *const OpenAI, arena: Allocator, request: *const sdk.Request) Error!@import("openai_request.zig").Wire {
        const prepared = try self.prepareRequest(arena, request, false);
        return .{ .body = try prepared.body(arena), .adaptations = prepared.adaptations };
    }

    fn chatCompatibility(self: *const OpenAI, model: []const u8) Error!@import("openai_request.zig").Compat {
        const wire = @import("openai_request.zig");
        if (self.chat_compat) |*bound| {
            const partial = try bound.forModel(wire.wireModel(model, .chat));
            return partial.resolve();
        }
        var compat = wire.Compat.defaults(.chat);
        compat.reasoning_usage = self.chat_reasoning_usage;
        return compat;
    }

    fn prepareRequest(self: *const OpenAI, arena: Allocator, request: *const sdk.Request, stream: bool) Error!@import("openai_request.zig").Prepared {
        const wire = @import("openai_request.zig");
        const dialect: wire.Dialect = if (self.api == .responses) .responses else .chat;
        return wire.prepare(arena, self.client.io, request, .{
            .dialect = dialect,
            .compat = if (dialect == .chat)
                try self.chatCompatibility(request.fields.model)
            else
                try wire.compatibility(
                    dialect,
                    if (self.responses_compat) |*policy| policy else null,
                    request.fields.config.extensions,
                ),
            .adaptations = self.adaptations,
            .max_media_bytes = self.max_media_bytes,
            .stream = stream,
        });
    }

    pub const StreamEvent = @import("openai_stream.zig").Event;
    pub const StreamSink = @import("openai_stream.zig").Sink;
    pub const StreamOutcome = @import("openai_stream.zig").Outcome;

    /// Stream one canonical OpenAI request without cache, history, or
    /// response assembly. Events are borrowed for the duration of the callback.
    /// Returning `.stop` closes the stream without emitting further events.
    /// `.exhausted` does not imply a terminal frame was received.
    pub fn streamRequest(self: *OpenAI, backing: Allocator, request: *const sdk.Request, sink: StreamSink) Error!StreamOutcome {
        var scratch = std.heap.ArenaAllocator.init(backing);
        defer scratch.deinit();
        const stream = try self.runStream(scratch.allocator(), request, sink, false);
        return stream.outcome;
    }

    const StreamRun = struct {
        outcome: StreamOutcome,
        completion: ?Completion,
        model: []const u8,
        adaptations: []const sdk.Adaptation,
    };

    fn runStream(
        self: *OpenAI,
        arena: Allocator,
        request: *const sdk.Request,
        sink: StreamSink,
        assemble: bool,
    ) Error!StreamRun {
        const wire_module = @import("openai_request.zig");
        const chat_compat = if (self.api == .chat_completions)
            try self.chatCompatibility(request.fields.model)
        else
            null;
        const dialect: wire_module.Dialect = if (chat_compat == null) .responses else .chat;
        var wire = if (chat_compat) |compat|
            try wire_module.prepare(arena, self.client.io, request, .{
                .dialect = dialect,
                .compat = compat,
                .adaptations = self.adaptations,
                .max_media_bytes = self.max_media_bytes,
                .stream = true,
            })
        else
            try self.prepareRequest(arena, request, true);
        var stream: OpenAIStream = .{
            .arena = arena,
            .model = wire_module.wireModel(request.fields.model, dialect),
            .stop = if (dialect == .responses) request.fields.config.stop else &.{},
            .chat_reasoning_usage = if (chat_compat) |compat| compat.reasoning_usage else null,
            .adaptations = if (self.adaptations == .silent) &.{} else wire.adaptations,
            .sink = sink,
            .assemble_events = assemble,
        };
        var consumer: http.ResponseConsumer = .{ .context = &stream, .consume = OpenAIStream.consume };
        const path = if (dialect == .responses) "responses" else "chat/completions";
        const url = try std.fmt.allocPrint(arena, "{s}/{s}", .{ std.mem.trimEnd(u8, self.base_url, "/"), path });
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        const response = self.sendRequest(arena, .{
            .url = url,
            .method = .POST,
            .body_provider = .{ .context = &wire, .prepare = wire_module.Prepared.prepareBody },
            .response_consumer = &consumer,
            .content_type = "application/json",
            .authorization = auth,
        }) catch |err| return sdkCodecError(err);
        defer self.client.allocator.free(response.body);
        return .{
            .outcome = stream.outcome,
            .completion = stream.result,
            .model = stream.model,
            .adaptations = stream.adaptations,
        };
    }

    /// Execute one SDK request. Candidate scoring may use several HTTP calls;
    /// public candidate fanout, cache and history remain outside this engine.
    pub fn completeRequest(self: *OpenAI, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        const completion = try self.execute(arena, request);
        return completion.response.?;
    }

    fn predictionRequest(self: *const OpenAI, arena: Allocator, request: Request) Error!*const sdk.Request {
        return canonicalPredictionRequest(arena, request, self.model, self.max_tokens, self);
    }

    pub fn canonicalPredictionRequest(
        arena: Allocator,
        request: Request,
        model: []const u8,
        max_tokens: ?u32,
        openai: ?*const OpenAI,
    ) Error!*const sdk.Request {
        if (request.decision != null or request.options.n != 1) return error.LMFailure;
        const runtime = @import("runtime_value.zig");
        var fields: runtime.ObjectMap = .empty;
        try fields.put(arena, "model", .{ .string = model });
        const has_system = request.messages.len != 0 and request.messages[0].role == .system;
        if (has_system) try fields.put(arena, "system", nativeContent(arena, request.messages[0].content));
        var messages: runtime.Array = .init(arena);
        try messages.ensureTotalCapacity(request.messages.len - @intFromBool(has_system));
        const turns = request.messages[@intFromBool(has_system)..];
        const attachment: ?AttachmentKind = if (openai) |self| switch (self.api) {
            .responses => if (hasToolAttachments(turns, .audio)) .audio else null,
            .chat_completions => if (hasToolAttachments(turns, .image) and
                (try self.chatCompatibility(self.model)).content_policy.tool_result_media == .reject)
                .image
            else
                null,
        } else null;
        if (attachment) |kind| {
            try appendToolAttachments(arena, &messages, turns, kind);
        } else {
            for (turns) |*message| try messages.append(try nativeMessage(arena, message));
        }
        try fields.put(arena, "messages", .{ .array = messages });
        var tools: runtime.Array = .init(arena);
        for (request.tools) |tool| {
            var function: runtime.ObjectMap = .empty;
            try function.put(arena, "type", .{ .string = "function" });
            try function.put(arena, "name", .{ .string = tool.name });
            try function.put(arena, "description", .{ .string = tool.description });
            try function.put(arena, "parameters", tool.parameters);
            try tools.append(.{ .object = function });
        }
        try fields.put(arena, "tools", .{ .array = tools });
        var config: runtime.ObjectMap = .empty;
        if (max_tokens) |limit| try config.put(arena, "max_tokens", .{ .integer = limit });
        if (request.options.temperature) |temperature| try config.put(arena, "temperature", .{ .float = temperature });
        if (request.options.logprobs != .null) try config.put(arena, "logprobs", request.options.logprobs);
        if (request.options.reasoning_effort) |effort| {
            var reasoning: runtime.ObjectMap = .empty;
            try reasoning.put(arena, "effort", .{ .string = if (std.mem.eql(u8, effort, "none")) "off" else effort });
            try config.put(arena, "reasoning", .{ .object = reasoning });
        }
        if (request.options.response_format) |format| {
            if (format != .text) {
                var encoded: runtime.ObjectMap = .empty;
                try encoded.put(arena, "type", .{ .string = @tagName(format) });
                if (format == .json_schema) {
                    try encoded.put(arena, "name", .{ .string = format.json_schema.name });
                    try encoded.put(arena, "schema", format.json_schema.schema);
                    try encoded.put(arena, "strict", .{ .bool = format.json_schema.strict });
                    if (format.json_schema.description) |description|
                        try encoded.put(arena, "description", .{ .string = description });
                }
                try config.put(arena, "response_format", .{ .object = encoded });
            }
        }
        if (request.options.tool_choice != null or request.options.parallel_tool_calls != null) {
            const choice = request.options.tool_choice orelse .auto;
            var selection: runtime.ObjectMap = .empty;
            const mode: []const u8 = switch (choice) {
                .auto, .none, .required => @tagName(choice),
                .function => "required",
                .allowed_tools => |allowed| @tagName(allowed.mode),
            };
            try selection.put(arena, "mode", .{ .string = mode });
            var allowed: runtime.Array = .init(arena);
            switch (choice) {
                .function => |name| try allowed.append(.{ .string = name }),
                .allowed_tools => |subset| for (subset.names) |name| {
                    try allowed.append(.{ .string = name });
                },
                else => {},
            }
            try selection.put(arena, "allowed", .{ .array = allowed });
            if (request.options.parallel_tool_calls) |parallel|
                try selection.put(arena, "parallel", .{ .bool = parallel });
            try config.put(arena, "tool_choice", .{ .object = selection });
        }
        if (openai) |self| try self.nativeToolOverrides(arena, request.tools, &config);
        try fields.put(arena, "config", .{ .object = config });
        return sdk.Request.fromValue(arena, .{ .object = fields }) catch |err| return sdkCodecError(err);
    }

    const AttachmentKind = enum {
        audio,
        image,

        fn matches(kind: AttachmentKind, part: @import("runtime_value.zig").Value) bool {
            if (part != .object) return false;
            const tag = part.object.get("type") orelse return false;
            return tag == .string and std.mem.eql(u8, tag.string, @tagName(kind));
        }

        fn placeholder(kind: AttachmentKind) []const u8 {
            return switch (kind) {
                .audio => "[audio attached in the next message]",
                .image => "[image attached in the next message]",
            };
        }
    };

    fn hasToolAttachments(messages: []const Message, kind: AttachmentKind) bool {
        for (messages) |message| {
            if (message.role != .tool or message.content != .parts) continue;
            for (message.content.parts) |part| if (kind.matches(part)) return true;
        }
        return false;
    }

    fn appendToolAttachments(
        arena: Allocator,
        rows: *@import("runtime_value.zig").Array,
        messages: []const Message,
        kind: AttachmentKind,
    ) Error!void {
        const runtime = @import("runtime_value.zig");
        var attachments: runtime.Array = .init(arena);
        var calls: []const inference.ToolCall = &.{};
        for (messages) |*message| {
            if (message.role == .tool) {
                try rows.append(try toolAttachment(arena, &attachments, message, calls, kind));
                continue;
            }
            try flushToolAttachments(arena, rows, &attachments);
            calls = if (message.role == .assistant) message.tool_calls else &.{};
            try rows.append(try nativeMessage(arena, message));
        }
        try flushToolAttachments(arena, rows, &attachments);
    }

    fn flushToolAttachments(
        arena: Allocator,
        rows: *@import("runtime_value.zig").Array,
        attachments: *@import("runtime_value.zig").Array,
    ) Error!void {
        if (attachments.items.len == 0) return;
        var row: @import("runtime_value.zig").ObjectMap = .empty;
        try row.put(arena, "role", .{ .string = "user" });
        try row.put(arena, "parts", .{ .array = attachments.* });
        try rows.append(.{ .object = row });
        attachments.* = .init(arena);
    }

    fn toolAttachment(
        arena: Allocator,
        attachments: *@import("runtime_value.zig").Array,
        message: *const Message,
        calls: []const inference.ToolCall,
        kind: AttachmentKind,
    ) Error!@import("runtime_value.zig").Value {
        const runtime = @import("runtime_value.zig");
        if (message.content != .parts) return nativeMessage(arena, message);
        for (message.content.parts) |part| {
            if (kind.matches(part)) break;
        } else return nativeMessage(arena, message);
        const id = message.tool_call_id orelse return error.LMFailure;
        const name = for (calls) |call| {
            if (std.mem.eql(u8, call.id, id)) break call.name;
        } else "";
        const label = if (name.len == 0)
            try std.fmt.allocPrint(arena, "Attachments from tool call {s}:", .{id})
        else
            try std.fmt.allocPrint(arena, "Attachments from tool call {s} ({s}):", .{ id, name });
        try attachments.append(try nativeTextPart(arena, label));
        const placeholder = try nativeTextPart(arena, kind.placeholder());
        var result: runtime.Array = .init(arena);
        try result.ensureTotalCapacity(message.content.parts.len);
        for (message.content.parts) |part| {
            if (kind.matches(part)) {
                try attachments.append(part);
                result.appendAssumeCapacity(placeholder);
            } else {
                result.appendAssumeCapacity(part);
            }
        }
        var text_result = message.*;
        text_result.content = .{ .parts = result.items };
        return nativeMessage(arena, &text_result);
    }

    fn nativeTextPart(arena: Allocator, text: []const u8) Error!@import("runtime_value.zig").Value {
        var fields: @import("runtime_value.zig").ObjectMap = .empty;
        try fields.put(arena, "type", .{ .string = "text" });
        try fields.put(arena, "text", .{ .string = text });
        return .{ .object = fields };
    }

    fn nativeContent(arena: Allocator, input: adapter.Content) @import("runtime_value.zig").Value {
        return switch (input) {
            .text => |text| .{ .string = text },
            .parts => |parts| blk: {
                var borrowed: @import("runtime_value.zig").Array = .init(arena);
                borrowed.items = @constCast(parts);
                borrowed.capacity = parts.len;
                break :blk .{ .array = borrowed };
            },
        };
    }

    fn nativeMessage(arena: Allocator, message: *const Message) Error!@import("runtime_value.zig").Value {
        const runtime = @import("runtime_value.zig");
        var row: runtime.ObjectMap = .empty;
        try row.put(arena, "role", .{ .string = if (message.role == .system) "developer" else message.role.name() });
        var parts: runtime.Array = .init(arena);
        if (message.role == .tool) {
            var result: runtime.ObjectMap = .empty;
            try result.put(arena, "type", .{ .string = "tool_result" });
            try result.put(arena, "id", .{ .string = message.tool_call_id orelse return error.LMFailure });
            try result.put(arena, "content", try nativeContentParts(arena, message.content));
            try parts.append(.{ .object = result });
        } else {
            if (message.tool_call_id != null or (message.role != .assistant and message.tool_calls.len != 0)) return error.LMFailure;
            if (message.continuation.len > 1 and message.reasoning != null and message.reasoning.?.len != 0) return error.LMFailure;
            if (message.reasoning != null or message.continuation.len != 0) {
                if (message.role != .assistant) return error.LMFailure;
                // The legacy response flattens part-local states. Without a
                // summary, each state can be replayed as its own thinking part;
                // grouping them would let the codec select only the first.
                for (0..@max(message.continuation.len, 1)) |index| {
                    const thinking: sdk.Part = .{ .thinking = .{
                        .text = message.reasoning orelse "",
                        .continuation = if (message.continuation.len == 0)
                            &.{}
                        else
                            message.continuation[index .. index + 1],
                    } };
                    try parts.append(thinking.toValue(arena) catch |err| return sdkCodecError(err));
                }
            }
            const visible = try nativeContentParts(arena, message.content);
            try parts.appendSlice(visible.array.items);
            for (message.tool_calls) |call| {
                const parsed = try parseJson(arena, call.arguments);
                var part: runtime.ObjectMap = .empty;
                try part.put(arena, "type", .{ .string = "tool_call" });
                try part.put(arena, "id", .{ .string = call.id });
                try part.put(arena, "name", .{ .string = call.name });
                try part.put(arena, "input", parsed);
                if (call.continuation.len != 0) {
                    const canonical: sdk.Part = .{ .tool_call = .{
                        .id = call.id,
                        .name = call.name,
                        .input = try jsonPayload(arena, parsed),
                        .continuation = call.continuation,
                    } };
                    try parts.append(canonical.toValue(arena) catch |err| return sdkCodecError(err));
                } else try parts.append(.{ .object = part });
            }
        }
        try row.put(arena, "parts", .{ .array = parts });
        return .{ .object = row };
    }

    fn nativeContentParts(arena: Allocator, input: adapter.Content) Error!@import("runtime_value.zig").Value {
        if (input == .parts) return nativeContent(arena, input);
        var parts: @import("runtime_value.zig").Array = .init(arena);
        try parts.append(try nativeTextPart(arena, input.text));
        return .{ .array = parts };
    }

    fn nativeToolOverrides(self: *const OpenAI, arena: Allocator, tools: []const FunctionTool, config: *@import("runtime_value.zig").ObjectMap) Error!void {
        // Per-tool strictness predates the SDK's single compatibility policy.
        // Preserve explicit native controls through its canonical wire hatch.
        const explicit = for (tools) |tool| {
            if (tool.strict != null) break true;
        } else false;
        if (!explicit) return;
        const runtime = @import("runtime_value.zig");
        var entries: runtime.Array = .init(arena);
        for (tools) |tool| {
            var fields: runtime.ObjectMap = .empty;
            if (self.api == .responses) try fields.put(arena, "type", .{ .string = "function" });
            try fields.put(arena, "name", .{ .string = tool.name });
            try fields.put(arena, "description", .{ .string = tool.description });
            try fields.put(arena, "parameters", tool.parameters);
            if (tool.strict) |strict| try fields.put(arena, "strict", .{ .bool = strict });
            if (self.api == .chat_completions) {
                var wrapper: runtime.ObjectMap = .empty;
                try wrapper.put(arena, "type", .{ .string = "function" });
                try wrapper.put(arena, "function", .{ .object = fields });
                try entries.append(.{ .object = wrapper });
            } else try entries.append(.{ .object = fields });
        }
        var extensions: runtime.ObjectMap = .empty;
        try extensions.put(arena, "tools", .{ .array = entries });
        try config.put(arena, "extensions", .{ .object = extensions });
    }

    fn sdkCodecError(err: anyerror) Error {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
    }

    /// Shared provider encoding for HTTP requests and native training records.
    /// Containers use arena; ID, name and serialized arguments remain borrowed.
    pub fn toolCallValue(arena: Allocator, api: Api, call: ToolCall) Error!@import("runtime_value.zig").Value {
        // OpenAI's function-call mapping has no native per-call opaque state.
        // Preserve it in metadata, but refuse a wire that cannot carry it.
        if (call.continuation.len != 0 or call.id.len == 0 or call.name.len == 0) return error.LMFailure;
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        if (api == .responses) {
            try object.put(arena, "type", .{ .string = "function_call" });
            try object.put(arena, "call_id", .{ .string = call.id });
            try object.put(arena, "name", .{ .string = call.name });
            try object.put(arena, "arguments", .{ .string = call.arguments });
        } else {
            var function: @import("runtime_value.zig").ObjectMap = .empty;
            try function.put(arena, "name", .{ .string = call.name });
            try function.put(arena, "arguments", .{ .string = call.arguments });
            try object.put(arena, "id", .{ .string = call.id });
            try object.put(arena, "type", .{ .string = "function" });
            try object.put(arena, "function", .{ .object = function });
        }
        return .{ .object = object };
    }

    /// Send a JSON request through the same bounded, retrying transport used by
    /// generation. `path` is relative to `base_url`; response values own their
    /// strings in `arena`. Additional headers support compatible provider APIs.
    pub fn postJson(self: *OpenAI, arena: Allocator, path: []const u8, body: @import("runtime_value.zig").Value, headers: []const std.http.Header) anyerror!@import("runtime_value.zig").Value {
        const url = try std.fmt.allocPrint(arena, "{s}/{s}", .{
            std.mem.trimEnd(u8, self.base_url, "/"),
            std.mem.trimStart(u8, path, "/"),
        });
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        var encoded: Writer.Allocating = .init(arena);
        pyjson.writeJson(&encoded.writer, body, .compact) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep, error.ParseValue => error.LMFailure,
        };
        const response = try self.fetchWithRetries(arena, url, auth, encoded.written(), headers);
        defer self.client.allocator.free(response.body);
        return parseJson(arena, response.body);
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        const canonical = try self.predictionRequest(arena, request);
        defer canonical.release();
        const out = try arena.alloc(Completion, 1);
        out[0] = try completeCanonical(ptr, arena, canonical);
        return out;
    }

    fn completeStream(
        ptr: *anyopaque,
        arena: Allocator,
        request: Request,
        sink: StreamSink,
    ) Error!Completion {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        const canonical = try self.predictionRequest(arena, request);
        defer canonical.release();
        const stream = try self.runStream(arena, canonical, sink, true);
        if (stream.outcome == .stopped) return error.StreamStopped;
        var completion = stream.completion orelse return error.LMFailure;
        try foldJudgments(arena, &completion, canonical);
        completion.adaptations = stream.adaptations;
        try publishResponse(arena, &completion, stream.model, .borrowed);
        return completion;
    }

    fn completeCanonical(
        ptr: *anyopaque,
        arena: Allocator,
        request: *const sdk.Request,
    ) Error!Completion {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        return self.execute(arena, request);
    }

    const judgments = @import("openai_judgment.zig");

    fn execute(self: *OpenAI, arena: Allocator, request: *const sdk.Request) Error!Completion {
        if (self.api == .chat_completions and request.fields.config.probabilities != null and
            request.fields.config.probabilities.? != .off)
        {
            const compat = try self.chatCompatibility(request.fields.model);
            if (compat.token_scoring == .logprob_token_ids) {
                if (try judgments.Format.fromRequest(arena, request)) |format|
                    return self.executeScoring(arena, request, compat, format);
            }
        }
        var wire = try self.prepareRequest(arena, request, false);
        const path = if (self.api == .responses) "responses" else "chat/completions";
        const url = try std.fmt.allocPrint(arena, "{s}/{s}", .{ std.mem.trimEnd(u8, self.base_url, "/"), path });
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        var input: http.Request = .{
            .url = url,
            .method = .POST,
            .body_provider = .{ .context = &wire, .prepare = @import("openai_request.zig").Prepared.prepareBody },
            .content_type = "application/json",
            .authorization = auth,
        };
        const streaming = for (wire.adaptations) |adaptation| {
            if (adaptation.action == .client_side and std.mem.eql(u8, adaptation.field, "config.stop")) break true;
        } else false;
        var stream: OpenAIStream = .{
            .arena = arena,
            .model = @import("openai_request.zig").wireModel(request.fields.model, .responses),
            .stop = request.fields.config.stop,
        };
        var consumer: http.ResponseConsumer = .{ .context = &stream, .consume = OpenAIStream.consume };
        if (streaming) {
            _ = input.prepare(arena, self.header_provider) catch |err| return sdkCodecError(err);
            wire = try self.prepareRequest(arena, request, true);
            stream.model = @import("openai_request.zig").wireModel(request.fields.model, .responses);
            stream.stop = request.fields.config.stop;
            input.response_consumer = &consumer;
        }
        const response = self.sendRequest(arena, input) catch |err| return sdkCodecError(err);
        defer self.client.allocator.free(response.body);
        var completion = if (streaming)
            stream.result orelse return error.LMFailure
        else if (self.api == .responses)
            try parseResponsesResponse(arena, response.body)
        else
            try parseCanonicalChat(
                arena,
                response.body,
                (try self.chatCompatibility(request.fields.model)).reasoning_usage,
            );
        const dialect: @import("openai_request.zig").Dialect = if (self.api == .responses) .responses else .chat;
        const model = @import("openai_request.zig").wireModel(request.fields.model, dialect);
        if (completion.metadata.model == null) completion.metadata.model = try arena.dupe(u8, model);
        try foldJudgments(arena, &completion, request);
        if (self.adaptations != .silent) completion.adaptations = wire.adaptations;
        try publishResponse(arena, &completion, model, .borrowed);
        return completion;
    }

    fn executeScoring(
        self: *OpenAI,
        arena: Allocator,
        request: *const sdk.Request,
        compat: @import("openai_request.zig").Compat,
        format: judgments.Format,
    ) Error!Completion {
        const wire = @import("openai_request.zig");
        const content = @import("openai_content.zig");
        const model = wire.wireModel(request.fields.model, .chat);
        const conversation_options: content.Options = .{
            .dialect = .chat,
            .policy = compat.content_policy,
            .max_media_bytes = self.max_media_bytes,
        };
        try judgments.validate(request);
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        try judgments.adaptations(arena, self.adaptations, &records, format, request.fields.config);
        var generated: sdk.Request = .{ .fields = request.fields, .allocator = arena };
        generated.fields.config.probabilities = .off;
        if (format.mixed()) {
            const preflight = try self.prepareRequest(arena, &generated, false);
            try records.appendSlice(arena, preflight.adaptations);
        }
        // Preflight is credential-free. Rebuild each physical message payload
        // separately, since credential hooks can mutate shared payload nodes.
        _ = try content.messages(arena, self.client.io, request, conversation_options);

        const base = std.mem.trimEnd(u8, self.base_url, "/");
        const tokenize_base = if (std.mem.endsWith(u8, base, "/v1")) base[0 .. base.len - 3] else base;
        const tokenize_url = try std.fmt.allocPrint(arena, "{s}/tokenize", .{tokenize_base});
        var inputs: std.ArrayList(http.Request) = .empty;
        for (format.measured) |judgment| {
            const ask = try judgments.question(arena, format.schema, judgment);
            try inputs.append(arena, try self.prepareTokenize(arena, request, conversation_options, tokenize_url, model, ask, "Answer:", true));
            for (judgment.keys) |key| {
                const answer = try std.fmt.allocPrint(arena, "Answer: {s}", .{key});
                try inputs.append(arena, try self.prepareTokenize(arena, request, conversation_options, tokenize_url, model, ask, answer, true));
                try inputs.append(arena, try self.prepareTokenize(arena, request, conversation_options, tokenize_url, model, ask, answer, false));
            }
        }
        // Source prepares all tokenization requests before the first send.
        // A malformed first tokenize reply must not shift the auth clock.
        const ends = try arena.alloc([]const usize, format.measured.len);
        var batch: judgments.Batch = .{ .arena = arena };
        var next: usize = 0;
        for (format.measured, ends) |judgment, *end| {
            const prefix = try self.readTokenization(arena, inputs.items[next], &batch);
            next += 1;
            const replies = try arena.alloc(judgments.Replies, judgment.keys.len);
            for (replies) |*reply| {
                reply.open = try self.readTokenization(arena, inputs.items[next], &batch);
                reply.closed = try self.readTokenization(arena, inputs.items[next + 1], &batch);
                next += 2;
            }
            end.* = try batch.add(prefix, replies);
        }
        assert(next == inputs.items.len);

        const score_url = try std.fmt.allocPrint(arena, "{s}/completions", .{base});
        const score_payload = try batch.payload(model);
        const score_request = try self.prepareScoringPost(arena, score_url, score_payload);
        const scored = try self.readScoringPost(arena, score_request);
        const measured = try batch.readScores(scored);
        const usage_raw = scored.object.get("usage") orelse .null;
        if (usage_raw != .null) {
            if (usage_raw != .object) return error.LMFailure;
            for ([_][]const u8{ "prompt_tokens_details", "completion_tokens_details" }) |field| {
                const details = usage_raw.object.get(field) orelse .null;
                if (details != .null and details != .object) return error.LMFailure;
            }
        }
        const score_model = scored.object.get("model") orelse .null;
        if (score_model != .null and (score_model != .string or score_model.string.len == 0)) return error.LMFailure;
        const usage_compat = if (self.chat_compat) |*bound| resolve: {
            const partial = if (score_model == .string) try bound.forModel(score_model.string) else bound.*;
            break :resolve try partial.resolve();
        } else try self.chatCompatibility(model);
        const usage = try chatUsage(arena, scored, usage_compat.reasoning_usage);
        var completion: Completion = undefined;
        if (measured) {
            const folded = try batch.fold(format.measured, ends, next);
            const parts = try arena.alloc(sdk.Part, 1);
            parts[0] = .{ .data = folded.data };
            completion = .{
                .text = null,
                .message = .{ .role = .assistant, .parts = parts },
                .metadata = .{
                    .model = if (score_model == .string) score_model.string else model,
                    .finish_reason = .stop,
                    .usage = usage,
                },
                .provider_data = try judgments.json(arena, folded.provider),
            };
            if (format.mixed()) {
                const result = try self.generateScoringAnswer(arena, &generated, format, false);
                var generated_response = result.completion;
                // Freeze the nested generated receipt before overwriting its
                // DataPart. Adaptations appear only on the outer response.
                try publishResponse(arena, &generated_response, model, .borrowed);
                const receipt = generated_response.response.?.toValue(arena) catch |err| return sdkCodecError(err);
                const index = dataIndex(result.parts);
                var merged = result.parts[index].data.value.toValue(arena) catch |err| return sdkCodecError(err);
                const picks = folded.data.value.toValue(arena) catch |err| return sdkCodecError(err);
                for (picks.object.keys(), picks.object.values()) |key, value| try merged.object.put(arena, key, value);
                var part = folded.data;
                part.value = try judgments.json(arena, merged);
                part.continuation = result.parts[index].data.continuation;
                result.parts[index] = .{ .data = part };
                completion.message = result.completion.message;
                completion.metadata.id = result.completion.metadata.id;
                completion.metadata.usage = try sumScoringUsage(arena, usage, result.completion.metadata.usage);
                var provider = folded.provider;
                try provider.object.put(arena, "scoring_usage", try usage.toValue(arena));
                try provider.object.put(arena, "generated_response", receipt);
                completion.provider_data = try judgments.json(arena, provider);
            }
        } else {
            if (request.fields.config.probabilities == .required) return error.LMFailure;
            try judgments.adapt(arena, self.adaptations, &records, "config.probabilities", .dropped, "the server accepted the request and returned no log-probs for the requested token ids (logprob_token_ids ignored); " ++
                "answered by structured output instead", .{ .string = "if_available" });
            const result = try self.generateScoringAnswer(arena, &generated, format, true);
            try judgments.appendUnique(arena, &records, result.adaptations);
            completion = result.completion;
            completion.metadata.usage = try sumScoringUsage(arena, usage, completion.metadata.usage);
            var provider = if (completion.provider_data) |data|
                data.toValue(arena) catch |err| return sdkCodecError(err)
            else
                try judgments.object(arena, &.{});
            try provider.object.put(arena, "scoring_usage", try usage.toValue(arena));
            completion.provider_data = try judgments.json(arena, provider);
        }
        if (self.adaptations != .silent) completion.adaptations = records.items;
        try publishResponse(arena, &completion, model, .borrowed);
        return completion;
    }

    fn dataIndex(parts: []const sdk.Part) usize {
        for (parts, 0..) |part, i| if (part == .data) return i;
        unreachable;
    }

    fn sumScoringUsage(arena: Allocator, a: Usage, b: Usage) Error!Usage {
        var result: Usage = .{};
        inline for (@typeInfo(Usage).@"struct".field_names) |field| {
            const left = @field(a, field);
            const right = @field(b, field);
            if (left != .null and right != .null)
                @field(result, field) = try countSum(arena, left, right);
        }
        return result;
    }

    fn prepareTokenize(self: *OpenAI, arena: Allocator, request: *const sdk.Request, options: @import("openai_content.zig").Options, url: []const u8, model: []const u8, ask: []const u8, answer: []const u8, open: bool) Error!http.Request {
        const conversation = try @import("openai_content.zig").messages(arena, self.client.io, request, options);
        return self.prepareScoringPost(arena, url, try judgments.tokenize(arena, model, conversation, ask, answer, open));
    }

    fn prepareScoringPost(self: *OpenAI, arena: Allocator, url: []const u8, payload: @import("runtime_value.zig").Value) Error!http.Request {
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        const input: http.Request = .{
            .url = url,
            .method = .POST,
            .content_type = "application/json",
            .authorization = auth,
        };
        var prepared = input.prepare(arena, self.header_provider) catch |err| return sdkCodecError(err);
        var encoded: Writer.Allocating = .init(arena);
        pyjson.writeJson(&encoded.writer, payload, .minified) catch |err| return sdkCodecError(if (err == error.WriteFailed) error.OutOfMemory else err);
        prepared.body = encoded.written();
        return prepared;
    }

    fn readScoringPost(self: *OpenAI, arena: Allocator, prepared: http.Request) Error!@import("runtime_value.zig").Value {
        const reply = self.sendWithHeaders(arena, prepared, null) catch |err| return sdkCodecError(err);
        defer self.client.allocator.free(reply.body);
        return parseJson(arena, reply.body);
    }

    fn readTokenization(self: *OpenAI, arena: Allocator, prepared: http.Request, batch: *judgments.Batch) Error![]const usize {
        return batch.readTokens(try self.readScoringPost(arena, prepared));
    }

    const GeneratedAnswer = struct {
        completion: Completion,
        parts: []sdk.Part,
        adaptations: []const sdk.Adaptation,
    };

    fn generateScoringAnswer(self: *OpenAI, arena: Allocator, request: *const sdk.Request, format: judgments.Format, fallback: bool) Error!GeneratedAnswer {
        const wire_mod = @import("openai_request.zig");
        var wire = try self.prepareRequest(arena, request, false);
        const url = try std.fmt.allocPrint(arena, "{s}/chat/completions", .{std.mem.trimEnd(u8, self.base_url, "/")});
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        const reply = self.sendRequest(arena, .{
            .url = url,
            .method = .POST,
            .body_provider = .{ .context = &wire, .prepare = wire_mod.Prepared.prepareBody },
            .content_type = "application/json",
            .authorization = auth,
        }) catch |err| return sdkCodecError(err);
        defer self.client.allocator.free(reply.body);
        const parsed = try parseJson(arena, reply.body);
        const choices = try chatChoices(parsed, true);
        if (choices.len != 1 or choices[0] != .object or
            !stringEquals(choices[0].object.get("finish_reason") orelse .null, "stop")) return error.LMFailure;
        var out: [1]Completion = undefined;
        try parseChatChoices(arena, parsed, choices, &out, true, (try self.chatCompatibility(request.fields.model)).reasoning_usage);
        var completion = out[0];
        const parts = try judgments.generatedParts(arena, format, completion.message.?, fallback);
        completion.message.?.parts = parts;
        if (completion.metadata.model == null) completion.metadata.model = wire_mod.wireModel(request.fields.model, .chat);
        return .{ .completion = completion, .parts = parts, .adaptations = wire.adaptations };
    }

    pub const OpenAIStream = @import("openai_stream.zig").Reader(.{
        .parse = parseJson,
        .text = responseText,
        .string = responseStringValue,
        .identity = responseIdentity,
        .usage = responsesUsage,
        .finish = responsesFinishReason,
        .chat_usage = chatUsage,
        .chat_finish = chatFinishReason,
        .scores = wireLogprobs,
        .citation = responseCitation,
        .equals = stringEquals,
        .truthy = responseTruthy,
        .finite = finiteJson,
        .payload = jsonPayload,
        .arguments = responseArguments,
        .json = streamJson,
        .project = streamProjection,
        .fold = foldJudgments,
    });

    fn streamProjection(arena: Allocator, completion: *Completion) Error!void {
        assert(completion.message != null);
        var text: TextParts = .{ .joined = .init(arena) };
        var reasoning: TextParts = .{ .joined = .init(arena) };
        for (completion.message.?.parts) |part| switch (part) {
            .text => |visible| try text.add(visible.text, ""),
            .thinking => |summary| try reasoning.add(summary.text, ""),
            else => {},
        };
        completion.text = text.value();
        completion.reasoning = reasoning.value();
    }

    fn streamJson(arena: Allocator, value: @import("runtime_value.zig").Value) Error![]const u8 {
        var out: Writer.Allocating = .init(arena);
        pyjson.writeJson(&out.writer, value, .compact) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            else => error.LMFailure,
        };
        return out.written();
    }

    /// Request-aware SDK adaptation, not JSON repair or answer validation.
    /// Preserve every other part and the replaced text's opaque continuation.
    pub fn foldJudgments(arena: Allocator, completion: *Completion, request: *const sdk.Request) Error!void {
        // Transport and credential callbacks may have changed opaque schemas.
        // Read the same SDK request again, never its stale wire projection.
        const configured = request.fields.config.response_format orelse return;
        const format = configured.toValue(arena) catch |err| return sdkCodecError(err);
        if (!stringEquals(format.object.get("type") orelse .null, "json_schema")) return;
        if (!try @import("typesafe.zig").hasJudgments(arena, format.object.get("schema") orelse .null)) return;
        var message = completion.message orelse return error.LMFailure;
        var text_index: ?usize = null;
        for (message.parts, 0..) |part, index| {
            if (part != .text) continue;
            if (text_index != null) return;
            text_index = index;
        }
        const index = text_index orelse return;
        const text = message.parts[index].text;
        const parsed = @import("json_repair.zig").parseJson(arena, text.text) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => {},
        };
        if (parsed != .object) return;
        // json.loads permits nonfinite literals, but SDK DataPart construction
        // refuses them. This is a provider error, not an unchanged text answer.
        if (!finiteJson(parsed)) return error.LMFailure;
        const parts = try arena.dupe(sdk.Part, message.parts);
        parts[index] = .{ .data = .{
            .value = try jsonPayload(arena, parsed),
            .continuation = text.continuation,
        } };
        message.parts = parts;
        completion.message = message;
    }

    fn fetchWithRetries(self: *OpenAI, arena: Allocator, url: []const u8, auth: ?[]const u8, body: []const u8, headers: []const std.http.Header) anyerror!http.Response {
        return self.sendRequest(arena, .{
            .url = url,
            .method = .POST,
            .body = body,
            .content_type = "application/json",
            .authorization = auth,
            .headers = headers,
        });
    }

    fn sendRequest(self: *OpenAI, arena: Allocator, input: http.Request) anyerror!http.Response {
        return self.sendWithHeaders(arena, input, self.header_provider);
    }

    fn sendWithHeaders(self: *OpenAI, arena: Allocator, input: http.Request, header_provider: ?http.HeaderProvider) anyerror!http.Response {
        const transport: http.Transport = .{
            .client = self.client,
            .max_response_bytes = self.max_response_bytes,
            .max_retries = self.max_retries,
            .retry_base_ms = self.retry_base_ms,
            .timeout_ns = self.timeout_ns,
            .connect_timeout_ns = self.connect_timeout_ns,
            .header_provider = header_provider,
            .success = .ok,
            .failure_error = error.LMFailure,
            .response_limit_error = error.LMFailure,
        };
        return transport.send(arena, input);
    }

    fn parseJson(arena: Allocator, body: []const u8) Error!@import("runtime_value.zig").Value {
        // The HTTP body is freed after parsing. Escaped and unescaped strings
        // must both belong to the caller's arena, not to the transport buffer.
        return @import("json_repair.zig").parseJson(arena, body) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
    }

    /// Extract `choices[i].message.content` and optional native reasoning.
    pub fn parseCompletions(arena: Allocator, body: []const u8) Error![]const Completion {
        const parsed = try parseJson(arena, body);
        const choices = try chatChoices(parsed, false);
        const out = try arena.alloc(Completion, choices.len);
        try parseChatChoices(arena, parsed, choices, out, false, .in_completion);
        return out;
    }

    fn parseCanonicalChat(arena: Allocator, body: []const u8, reasoning_usage: @FieldType(OpenAI, "chat_reasoning_usage")) Error!Completion {
        const parsed = try parseJson(arena, body);
        const choices = try chatChoices(parsed, true);
        var out: [1]Completion = undefined;
        try parseChatChoices(arena, parsed, choices, &out, true, reasoning_usage);
        return out[0];
    }

    fn chatChoices(parsed: @import("runtime_value.zig").Value, comptime canonical: bool) Error![]const @import("runtime_value.zig").Value {
        if (parsed != .object or !finiteJson(parsed)) return error.LMFailure;
        if (parsed.object.get("error")) |failure| if (failure == .object) return error.LMFailure;
        const choices = parsed.object.get("choices") orelse .null;
        const items = if (!try responseTruthy(choices))
            &.{}
        else if (choices == .array)
            choices.array.items
        else
            return error.LMFailure;
        if (canonical) {
            if (items.len > 1) return error.LMFailure;
        } else if (items.len == 0) return error.LMFailure;
        return items;
    }

    fn parseChatChoices(arena: Allocator, parsed: @import("runtime_value.zig").Value, choices: []const @import("runtime_value.zig").Value, out: []Completion, comptime canonical: bool, reasoning_usage: @FieldType(OpenAI, "chat_reasoning_usage")) Error!void {
        const id = try responseIdentity(arena, parsed.object.get("id"));
        const model = try responseIdentity(arena, parsed.object.get("model"));
        const usage = try chatUsage(arena, parsed, reasoning_usage);
        for (out, 0..) |*completion, choice_index| {
            var unmapped: Unmapped = .{ .records = .init(arena) };
            const raw_choice = if (choice_index < choices.len) choices[choice_index] else @import("runtime_value.zig").Value{ .object = .empty };
            const choice = if (raw_choice == .object) raw_choice else if (canonical) empty: {
                try unmapped.addChat(choice_index, "", null, .{ .string = jsonTypeName(raw_choice) });
                break :empty @import("runtime_value.zig").Value{ .object = .empty };
            } else return error.LMFailure;
            const raw_message = choice.object.get("message") orelse .null;
            const message = if (raw_message == .object) raw_message else if (canonical)
                @import("runtime_value.zig").Value{ .object = .empty }
            else
                return error.LMFailure;
            completion.* = .{ .text = null };
            var parts: std.ArrayList(sdk.Part) = .empty;
            const reasoning = try annotationAlternative(message, &.{ "reasoning_content", "reasoning" });
            if (try responseTruthy(reasoning)) {
                completion.reasoning = try responseStringValue(arena, reasoning);
                try parts.append(arena, .{ .thinking = .{ .text = completion.reasoning.? } });
            }
            var text: TextParts = .{ .joined = .init(arena) };
            try parseChatText(arena, message.object.get("content"), &text, &parts, &unmapped, choice_index);
            if (message.object.get("refusal")) |refusal| {
                if (try responseTruthy(refusal)) {
                    try parts.append(arena, .{ .refusal = .{ .text = try responseStringValue(arena, refusal) } });
                }
            }
            completion.tool_calls = try parseChatTools(arena, message.object.get("tool_calls"), &parts, &unmapped, choice_index);
            if (parts.items.len == 0) {
                try text.add("", "");
                try parts.append(arena, .{ .text = .{ .text = "" } });
            }
            completion.message = .{ .role = .assistant, .parts = parts.items };
            const raw_scores = choice.object.get("logprobs") orelse .null;
            if (raw_scores == .object) {
                const scores = try wireLogprobs(arena, raw_scores.object.get("content") orelse .null);
                if (scores.len != 0) completion.logprobs = scores;
            }
            completion.text = text.value();
            const raw_finish = choice.object.get("finish_reason") orelse .null;
            const finish = chatFinishReason(raw_finish, completion.tool_calls.len != 0);
            if (completion.tool_calls.len == 0 and raw_finish != .null and !stringEquals(raw_finish, "") and
                finish == .stop and !stringEquals(raw_finish, "stop"))
                try unmapped.addChat(choice_index, "finish_reason", null, raw_finish);
            completion.provider_data = try jsonPayload(arena, try unmapped.attach(parsed));
            completion.metadata = .{
                .id = id,
                .model = model,
                .finish_reason = finish,
                // SDK choice=i ingestion retains the response-wide usage for
                // each selected choice. The LM boundary rejects an unexpected
                // multi-choice body rather than billing these as separate calls.
                .usage = usage,
            };
        }
    }

    fn parseChatText(arena: Allocator, raw: ?@import("runtime_value.zig").Value, text: *TextParts, parts: *std.ArrayList(sdk.Part), unmapped: *Unmapped, choice_index: usize) Error!void {
        const content = raw orelse return;
        if (content == .string and content.string.len != 0) {
            try text.add(content.string, "");
            try parts.append(arena, .{ .text = .{ .text = content.string } });
        } else if (content == .array) {
            for (content.array.items, 0..) |part, content_index| {
                if (part != .object or !stringEquals(part.object.get("type") orelse .null, "text")) {
                    try unmapped.addChat(choice_index, "message.content", content_index, if (part == .object) part.object.get("type") orelse .null else .{ .string = jsonTypeName(part) });
                    continue;
                }
                const visible = try responseText(arena, part.object.get("text"));
                try text.add(visible, "");
                try parts.append(arena, .{ .text = .{ .text = visible } });
            }
        } else if (content != .null and content != .string) {
            try unmapped.addChat(choice_index, "message.content", null, .{ .string = jsonTypeName(content) });
        }
    }

    fn parseChatTools(arena: Allocator, raw: ?@import("runtime_value.zig").Value, parts: *std.ArrayList(sdk.Part), unmapped: *Unmapped, choice_index: usize) Error![]const ToolCall {
        const calls = raw orelse return &.{};
        if (calls != .array) {
            if (calls != .object and calls != .string and try responseTruthy(calls)) return error.LMFailure;
            return &.{};
        }
        var native: std.ArrayList(ToolCall) = .empty;
        for (calls.array.items, 0..) |call, call_index| {
            if (call != .object) {
                try unmapped.addChat(choice_index, "message.tool_calls", call_index, .{ .string = jsonTypeName(call) });
                continue;
            }
            const kind = call.object.get("type") orelse .null;
            if (try responseTruthy(kind) and !stringEquals(kind, "function")) {
                try unmapped.addChat(choice_index, "message.tool_calls", call_index, kind);
                continue;
            }
            const function = call.object.get("function") orelse return error.LMFailure;
            if (function != .object) return error.LMFailure;
            const name = try responseText(arena, function.object.get("name"));
            if (name.len == 0) return error.LMFailure;
            const raw_id = call.object.get("id");
            const call_id = try responseText(arena, raw_id);
            const input = try responseInput(arena, function.object.get("arguments"), call_id.len != 0 and raw_id != null and raw_id.? == .string);
            const resolved_id = if (call_id.len != 0) call_id else try std.fmt.allocPrint(arena, "call_{d}", .{parts.items.len});
            try native.append(arena, .{
                .id = resolved_id,
                .name = name,
                .arguments = try responseArguments(arena, input),
            });
            try parts.append(arena, .{ .tool_call = .{ .id = resolved_id, .name = name, .input = try jsonPayload(arena, input) } });
        }
        return native.items;
    }

    /// Text-only Chat Completions parsing, matching `LM.completeMany`.
    pub fn parseResponse(arena: Allocator, body: []const u8) Error![]const []const u8 {
        const completions = try parseCompletions(arena, body);
        const out = try arena.alloc([]const u8, completions.len);
        for (out, completions) |*text, completion| text.* = completion.text orelse return error.LMFailure;
        return out;
    }

    /// Normalize visible text without conflating absent TextParts with "".
    /// lm15 ignores root/item statuses and roles here; refusal, reasoning and
    /// tool-only outputs have null text rather than a fabricated empty answer.
    pub fn parseResponsesResponse(arena: Allocator, body: []const u8) Error!Completion {
        const parsed = try parseJson(arena, body);
        if (parsed != .object) return error.LMFailure;
        if (!finiteJson(parsed)) return error.LMFailure;
        if (parsed.object.get("error")) |failure| {
            if (failure == .object) return error.LMFailure;
        }
        const output = parsed.object.get("output") orelse .null;
        if (output != .array and output != .null) return error.LMFailure;
        const items: []const @import("runtime_value.zig").Value = if (output == .array) output.array.items else &.{};
        var parts: std.ArrayList(sdk.Part) = .empty;
        var generated_media = false;
        var scores: std.ArrayList(TokenLogprob) = .empty;
        var text: TextParts = .{ .joined = .init(arena) };
        var reasoning: TextParts = .{ .joined = .init(arena) };
        var calls: std.ArrayList(ToolCall) = .empty;
        var citations: std.ArrayList(CitationPart) = .empty;
        var continuation: std.ArrayList(Continuation) = .empty;
        var unmapped: Unmapped = .{ .records = .init(arena) };
        for (items, 0..) |item, item_index| {
            if (item != .object) {
                try unmapped.add(item_index, null, .{ .string = jsonTypeName(item) });
                continue;
            }
            const kind = item.object.get("type") orelse .null;
            if (stringEquals(kind, "reasoning")) {
                const summary = try reasoningText(arena, item);
                if (summary) |visible| try reasoning.add(visible, "");
                var state: @import("runtime_value.zig").ObjectMap = .empty;
                for ([_][]const u8{ "id", "encrypted_content" }) |name| {
                    if (item.object.get(name)) |field| {
                        if (try responseTruthy(field))
                            try state.put(arena, name, .{ .string = try responseStringValue(arena, field) });
                    }
                }
                const item_continuation: []const Continuation = if (state.count() != 0)
                    try arena.dupe(Continuation, &.{.{
                        .provider = "openai",
                        .kind = "reasoning_item",
                        .data = try jsonPayload(arena, .{ .object = state }),
                    }})
                else
                    &.{};
                try continuation.appendSlice(arena, item_continuation);
                if (state.count() != 0 or summary != null) {
                    try parts.append(arena, .{ .thinking = .{
                        .text = summary orelse "",
                        .continuation = item_continuation,
                    } });
                }
            } else if (stringEquals(kind, "function_call")) {
                const call_id = try responseText(arena, item.object.get("call_id"));
                const item_id = if (call_id.len == 0) try responseText(arena, item.object.get("id")) else "";
                const input = try responseInput(arena, item.object.get("arguments"), blk: {
                    const raw_id = if (call_id.len != 0) item.object.get("call_id") else item.object.get("id");
                    break :blk raw_id != null and raw_id.? == .string and (call_id.len != 0 or item_id.len != 0);
                });
                const resolved_id = if (call_id.len != 0) call_id else if (item_id.len != 0) item_id else try std.fmt.allocPrint(arena, "call_{d}", .{parts.items.len});
                const name = try responseString(item, "name", true);
                try calls.append(arena, .{
                    .id = resolved_id,
                    .name = name,
                    .arguments = try responseArguments(arena, input),
                });
                try parts.append(arena, .{ .tool_call = .{ .id = resolved_id, .name = name, .input = try jsonPayload(arena, input) } });
            } else if (stringEquals(kind, "message")) {
                const content = item.object.get("content") orelse .null;
                if (content == .null) continue;
                if (content != .array) return error.LMFailure;
                for (content.array.items, 0..) |part, content_index| {
                    if (part != .object) {
                        try unmapped.add(item_index, content_index, .{ .string = jsonTypeName(part) });
                        continue;
                    }
                    const part_type = part.object.get("type") orelse .null;
                    if (stringEquals(part_type, "output_image") or stringEquals(part_type, "output_audio")) {
                        if (try responseGeneratedMedia(arena, part, stringEquals(part_type, "output_audio"))) |media| {
                            generated_media = true;
                            try parts.append(arena, media);
                        }
                    } else if (stringEquals(part_type, "refusal")) {
                        const refusal = try responseText(arena, part.object.get("refusal"));
                        const refused = if (refusal.len != 0) refusal else try responseText(arena, part.object.get("text"));
                        if (refused.len == 0) {
                            try text.add("", "");
                            try parts.append(arena, .{ .text = .{ .text = "" } });
                        } else {
                            try parts.append(arena, .{ .refusal = .{ .text = refused } });
                        }
                    } else if (stringEquals(part_type, "output_text") or stringEquals(part_type, "text")) {
                        try scores.appendSlice(arena, try wireLogprobs(arena, part.object.get("logprobs") orelse .null));
                        const visible = try responseText(arena, part.object.get("text"));
                        try text.add(visible, "");
                        try parts.append(arena, .{ .text = .{ .text = visible } });
                        if (part.object.get("annotations")) |annotations| {
                            if (annotations == .array) {
                                for (annotations.array.items) |annotation| {
                                    if (annotation != .object) continue;
                                    if (try responseCitation(arena, annotation, visible)) |citation| {
                                        try citations.append(arena, citation);
                                        try parts.append(arena, .{ .citation = citation });
                                    }
                                }
                            } else if (annotations != .object and annotations != .string and try responseTruthy(annotations)) {
                                return error.LMFailure;
                            }
                        }
                    } else {
                        try unmapped.add(item_index, content_index, part_type);
                    }
                }
            } else if (!providerExecutedItem(kind)) {
                if (kind == .array or kind == .object) return error.LMFailure;
                try unmapped.add(item_index, null, kind);
            }
        }
        if (parts.items.len == 0) {
            const fallback = try responseText(arena, parsed.object.get("output_text"));
            try text.add(fallback, "");
            try parts.append(arena, .{ .text = .{ .text = fallback } });
        }
        const answer = text.value();
        return .{
            .text = answer,
            .message = .{ .role = .assistant, .parts = parts.items },
            .reasoning = reasoning.value(),
            .generated_media = generated_media,
            .logprobs = if (scores.items.len != 0) scores.items else null,
            .tool_calls = calls.items,
            .citations = citations.items,
            .continuation = continuation.items,
            .provider_data = try jsonPayload(arena, try unmapped.attach(parsed)),
            .metadata = .{
                .id = try responseIdentity(arena, parsed.object.get("id")),
                .model = try responseIdentity(arena, parsed.object.get("model")),
                .finish_reason = try responsesFinishReason(arena, parsed, calls.items.len != 0),
                .usage = try responsesUsage(arena, parsed),
            },
        };
    }

    const Unmapped = struct {
        records: @import("runtime_value.zig").Array,

        fn add(self: *Unmapped, item: usize, content: ?usize, kind: @import("runtime_value.zig").Value) Error!void {
            const arena = self.records.allocator;
            const path = if (content) |index|
                try std.fmt.allocPrint(arena, "output[{d}].content[{d}]", .{ item, index })
            else
                try std.fmt.allocPrint(arena, "output[{d}]", .{item});
            try self.addPath(path, kind);
        }

        fn addChat(self: *Unmapped, choice: usize, field: []const u8, item: ?usize, kind: @import("runtime_value.zig").Value) Error!void {
            const arena = self.records.allocator;
            const path = if (item) |index|
                try std.fmt.allocPrint(arena, "choices[{d}].{s}[{d}]", .{ choice, field, index })
            else if (field.len != 0)
                try std.fmt.allocPrint(arena, "choices[{d}].{s}", .{ choice, field })
            else
                try std.fmt.allocPrint(arena, "choices[{d}]", .{choice});
            try self.addPath(path, kind);
        }

        fn addPath(self: *Unmapped, path: []const u8, kind: @import("runtime_value.zig").Value) Error!void {
            const arena = self.records.allocator;
            var record: @import("runtime_value.zig").ObjectMap = .empty;
            try record.put(arena, "path", .{ .string = path });
            try record.put(arena, "type", .{ .string = if (try responseTruthy(kind))
                try responseStringValue(arena, kind)
            else
                "<missing>" });
            try self.records.append(.{ .object = record });
        }

        fn attach(self: *const Unmapped, parsed: @import("runtime_value.zig").Value) Allocator.Error!@import("runtime_value.zig").Value {
            if (self.records.items.len == 0) return parsed;
            const arena = self.records.allocator;
            var object: @import("runtime_value.zig").ObjectMap = .empty;
            try object.ensureTotalCapacity(arena, parsed.object.count() +
                @intFromBool(!parsed.object.contains("_lm15_unmapped")));
            for (parsed.object.keys(), parsed.object.values()) |key, value|
                object.putAssumeCapacity(key, value);
            object.putAssumeCapacity("_lm15_unmapped", .{ .array = self.records });
            return .{ .object = object };
        }
    };

    fn jsonTypeName(value: @import("runtime_value.zig").Value) []const u8 {
        return switch (value) {
            .null => "NoneType",
            .bool => "bool",
            .integer => "int",
            .float => "float",
            .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text)) "int" else "float",
            .string => "str",
            .array => "list",
            .object => "dict",
            .custom => unreachable, // HTTP JSON does not construct native classes.
        };
    }

    fn providerExecutedItem(kind: @import("runtime_value.zig").Value) bool {
        inline for (.{ "web_search_call", "file_search_call", "code_interpreter_call", "computer_call", "computer_use_call" }) |name| {
            if (stringEquals(kind, name)) return true;
        }
        return false;
    }

    /// Both OpenAI dialects use the same chosen-token and alternative wire
    /// shape. Malformed entries are skipped, but a present invalid score or
    /// byte sequence refuses the response rather than silently dropping data.
    fn wireLogprobs(arena: Allocator, input: @import("runtime_value.zig").Value) Error![]const TokenLogprob {
        if (input != .array) return &.{};
        var scores: std.ArrayList(TokenLogprob) = .empty;
        for (input.array.items) |entry| {
            if (!wireScoreEntry(entry)) continue;
            const chosen = try wireTopLogprob(arena, entry);
            var top: std.ArrayList(TopLogprob) = .empty;
            const alternatives = entry.object.get("top_logprobs") orelse .null;
            if (alternatives == .array) {
                for (alternatives.array.items) |alternative| {
                    if (!wireScoreEntry(alternative)) continue;
                    try top.append(arena, try wireTopLogprob(arena, alternative));
                }
            } else if (alternatives != .object and alternatives != .string) {
                if (try responseTruthy(alternatives)) return error.LMFailure;
            }
            try scores.append(arena, .{
                .token = chosen.token,
                .logprob = chosen.logprob,
                .bytes = chosen.bytes,
                .top = top.items,
            });
        }
        return scores.items;
    }

    fn wireScoreEntry(input: @import("runtime_value.zig").Value) bool {
        if (input != .object) return false;
        return input.object.contains("token") and input.object.contains("logprob");
    }

    fn wireTopLogprob(arena: Allocator, input: @import("runtime_value.zig").Value) Error!TopLogprob {
        const raw = input.object.get("logprob").?;
        const score = if (raw == .string)
            @import("value.zig").floatFromString(arena, raw.string) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.InvalidNumber => return error.LMFailure,
            }
        else
            @import("value.zig").numericFloat(raw) catch return error.LMFailure;
        const bytes = input.object.get("bytes") orelse .null;
        const sequence = if (bytes == .array) bytes.array.items else null;
        logprobs.validateBytes(sequence) catch return error.LMFailure;
        return .{
            .token = try responseStringValue(arena, input.object.get("token").?),
            .logprob = score,
            .bytes = sequence,
        };
    }

    /// Match the SDK's truthy payload fallback and shared media shape policy
    /// before counting a part. A reported cost cannot rescue malformed media.
    fn responseGeneratedMedia(arena: Allocator, part: @import("runtime_value.zig").Value, audio: bool) Error!?sdk.Part {
        const Value = @import("runtime_value.zig").Value;
        var payload: Value = .null;
        if (audio) {
            const nested = part.object.get("audio") orelse .null;
            if (nested == .object) payload = nested.object.get("data") orelse .null;
        }
        if (!(try responseTruthy(payload))) payload = part.object.get("b64_json") orelse .null;
        if (!audio and !(try responseTruthy(payload))) payload = part.object.get("image_base64") orelse .null;
        if (!(try responseTruthy(payload))) return null;
        const encoded = try responseStringValue(arena, payload);
        @import("file.zig").validateBase64(encoded) catch return error.LMFailure;
        return if (audio)
            .{ .audio = .{ .media_type = "audio/wav", .data = encoded } }
        else
            .{ .image = .{ .media_type = "image/png", .data = encoded } };
    }

    /// lm15 reads `str(x) if x else None`. Model fallback is applied at the
    /// engine boundary, where the actual request model is available.
    fn responseIdentity(arena: Allocator, raw: ?@import("runtime_value.zig").Value) Error!?[]const u8 {
        const value = raw orelse return null;
        if (!(try responseTruthy(value))) return null;
        return try responseStringValue(arena, value);
    }

    /// lm15 `_FINISH_REASON_MAP`: a tool call wins, and an absent, empty or
    /// unmapped reason reads as stop.
    fn chatFinishReason(raw: @import("runtime_value.zig").Value, has_tool_call: bool) FinishReason {
        if (has_tool_call) return .tool_call;
        if (raw != .string) return .stop;
        const mapped = [_]struct { []const u8, FinishReason }{
            .{ "stop", .stop },
            .{ "length", .length },
            .{ "tool_calls", .tool_call },
            .{ "function_call", .tool_call },
            .{ "content_filter", .content_filter },
        };
        for (mapped) |entry| {
            if (std.mem.eql(u8, raw.string, entry[0])) return entry[1];
        }
        return .stop;
    }

    /// lm15 `_finish_from_status`. Case folding is ASCII, not Python's Unicode
    /// `str.lower`; the provider vocabulary compared against is ASCII.
    fn responsesFinishReason(arena: Allocator, data: @import("runtime_value.zig").Value, has_tool_call: bool) Error!FinishReason {
        assert(data == .object);
        if (has_tool_call) return .tool_call;
        const status = try loweredText(arena, data.object.get("status") orelse .null);
        const details = data.object.get("incomplete_details") orelse .null;
        const reason = if (details == .object)
            try loweredText(arena, details.object.get("reason") orelse .null)
        else
            "";
        if (std.mem.eql(u8, status, "incomplete")) {
            if (std.mem.indexOf(u8, reason, "token") != null) return .length;
        }
        if (std.mem.indexOf(u8, reason, "content_filter") != null) return .content_filter;
        if (std.mem.indexOf(u8, reason, "safety") != null) return .content_filter;
        return .stop;
    }

    /// Python `str(value or "").lower()`.
    fn loweredText(arena: Allocator, value: @import("runtime_value.zig").Value) Error![]const u8 {
        if (!(try responseTruthy(value))) return "";
        return std.ascii.allocLowerString(arena, try responseStringValue(arena, value));
    }

    /// MAP-19: only separate-reasoning servers add reasoning to completion.
    /// A missing separate count leaves output unknown unless total proves zero.
    fn chatUsage(arena: Allocator, data: @import("runtime_value.zig").Value, reasoning_usage: @FieldType(OpenAI, "chat_reasoning_usage")) Error!Usage {
        assert(data == .object);
        const raw = data.object.get("usage") orelse .null;
        if (!(try responseTruthy(raw))) return .{};
        if (raw != .object) return error.LMFailure;
        const prompt_details = try usageDetails(raw, &.{"prompt_tokens_details"});
        const completion_details = try usageDetails(raw, &.{"completion_tokens_details"});
        // Some servers report the cached count flat on usage instead.
        var cached = detail(prompt_details, "cached_tokens");
        if (cached == .null) cached = raw.object.get("cached_tokens") orelse .null;
        const prompt = try wireCount(arena, raw.object.get("prompt_tokens") orelse .null);
        const completion = try wireCount(arena, raw.object.get("completion_tokens") orelse .null);
        const total = try wireCount(arena, raw.object.get("total_tokens") orelse .null);
        const reasoning = try wireCount(arena, detail(completion_details, "reasoning_tokens"));
        var output = completion;
        if (reasoning_usage == .separate and completion != .null) {
            if (reasoning != .null) {
                output = try countSum(arena, completion, reasoning);
            } else {
                output = .null;
                if (total != .null and prompt != .null) {
                    const sum = try countSum(arena, prompt, completion);
                    const order = @import("value.zig").numericOrder(total, sum) catch return error.LMFailure;
                    if ((order orelse return error.LMFailure) == .eq) output = completion;
                }
            }
        }
        return settleUsage(arena, .{
            .input_tokens = prompt,
            .output_tokens = output,
            .total_tokens = total,
            .cache_read_tokens = try wireCount(arena, cached),
            .cache_write_tokens = try wireCount(arena, detail(prompt_details, "cache_write_tokens")),
            .reasoning_tokens = reasoning,
            .input_audio_tokens = try wireCount(arena, detail(prompt_details, "audio_tokens")),
            .output_audio_tokens = try wireCount(arena, detail(completion_details, "audio_tokens")),
        });
    }

    /// MAP-19 for Responses: every counter keeps its reported value.
    fn responsesUsage(arena: Allocator, data: @import("runtime_value.zig").Value) Error!Usage {
        assert(data == .object);
        const raw = data.object.get("usage") orelse .null;
        if (raw != .object) return .{};
        const input_details = try usageDetails(raw, &.{ "input_tokens_details", "input_token_details" });
        const output_details = try usageDetails(raw, &.{ "output_tokens_details", "output_token_details" });
        return settleUsage(arena, .{
            .input_tokens = try wireCount(arena, raw.object.get("input_tokens") orelse .null),
            .output_tokens = try wireCount(arena, raw.object.get("output_tokens") orelse .null),
            .total_tokens = try wireCount(arena, raw.object.get("total_tokens") orelse .null),
            .cache_read_tokens = try wireCount(arena, detail(input_details, "cached_tokens")),
            .cache_write_tokens = try wireCount(arena, detail(input_details, "cache_write_tokens")),
            .reasoning_tokens = try wireCount(arena, detail(output_details, "reasoning_tokens")),
            .input_audio_tokens = try wireCount(arena, detail(input_details, "audio_tokens")),
            .output_audio_tokens = try wireCount(arena, detail(output_details, "audio_tokens")),
        });
    }

    /// Python `usage.get(a) or usage.get(b) or {}`, where a non-dict is `{}`.
    fn usageDetails(usage: @import("runtime_value.zig").Value, comptime names: []const []const u8) error{ParseValue}!?@import("runtime_value.zig").Value {
        const chosen = try annotationAlternative(usage, names);
        return if (chosen == .object) chosen else null;
    }

    fn detail(details: ?@import("runtime_value.zig").Value, name: []const u8) @import("runtime_value.zig").Value {
        const object = details orelse return .null;
        return object.object.get(name) orelse .null;
    }

    /// Wire counts may be negative: settle_usage first withholds an inconsistent
    /// whole, then SDK Usage validates the counters which remain. Refusing here
    /// changes that observable order. Bool and nonintegral values still refuse.
    pub fn wireCount(arena: Allocator, raw: @import("runtime_value.zig").Value) Error!@import("runtime_value.zig").Value {
        switch (raw) {
            .null, .integer => return raw,
            .number_string => |digits| {
                if (!std.json.isNumberFormattedLikeAnInteger(digits)) return error.LMFailure;
                return raw;
            },
            .float => |number| {
                if (!std.math.isFinite(number) or number != @trunc(number)) return error.LMFailure;
                if (number >= -9223372036854775808.0 and number < 9223372036854775808.0)
                    return .{ .integer = @intFromFloat(number) };
                var buffer: [310]u8 = undefined;
                const digits = @import("value.zig").integralFloatText(number, &buffer).?;
                return .{ .number_string = try arena.dupe(u8, digits) };
            },
            .bool, .string, .array, .object, .custom => return error.LMFailure,
        }
    }

    /// lm15 `settle_usage` (MAP-19 rule 3): withhold what the provider's
    /// numbers do not determine instead of failing or estimating. A part
    /// larger than its whole withholds the whole; a total which is not
    /// input + output withholds both and keeps the provider's total.
    pub fn settleUsage(arena: Allocator, counts: Usage) Error!Usage {
        var usage = counts;
        if (usage.input_tokens != .null) {
            const cached = if (usage.cache_read_tokens == .null and usage.cache_write_tokens == .null)
                @import("runtime_value.zig").Value{ .integer = 0 }
            else if (usage.cache_read_tokens == .null)
                usage.cache_write_tokens
            else if (usage.cache_write_tokens == .null)
                usage.cache_read_tokens
            else
                try countSum(arena, usage.cache_read_tokens, usage.cache_write_tokens);
            if (try countExceeds(cached, usage.input_tokens)) {
                usage.input_tokens = .null;
            } else if (try countExceeds(usage.input_audio_tokens, usage.input_tokens)) {
                usage.input_tokens = .null;
            }
        }
        if (usage.output_tokens != .null) {
            if (try countExceeds(usage.reasoning_tokens, usage.output_tokens)) {
                usage.output_tokens = .null;
            } else if (try countExceeds(usage.output_audio_tokens, usage.output_tokens)) {
                usage.output_tokens = .null;
            }
        }
        if (usage.total_tokens != .null and usage.input_tokens != .null and usage.output_tokens != .null) {
            const sum = try countSum(arena, usage.input_tokens, usage.output_tokens);
            if (!(try countExceeds(sum, usage.total_tokens)) and !(try countExceeds(usage.total_tokens, sum))) {
                // The provider's total is exactly input + output.
            } else {
                usage.input_tokens = .null;
                usage.output_tokens = .null;
            }
        }
        return checkedUsage(arena, usage);
    }

    /// Exact integer comparison; an unknown part exceeds nothing.
    fn countExceeds(part: @import("runtime_value.zig").Value, whole: @import("runtime_value.zig").Value) Error!bool {
        assert(whole != .null);
        if (part == .null) return false;
        const order = @import("value.zig").numericOrder(part, whole) catch return error.LMFailure;
        return (order orelse return error.LMFailure) == .gt;
    }

    /// Exact integer sum; counters never round through binary64.
    pub fn countSum(arena: Allocator, a: @import("runtime_value.zig").Value, b: @import("runtime_value.zig").Value) Error!@import("runtime_value.zig").Value {
        return @import("value.zig").numericSum(arena, a, b) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
    }

    pub fn responseTruthy(value: @import("runtime_value.zig").Value) error{ParseValue}!bool {
        return switch (value) {
            .null => false,
            .bool => |b| b,
            .integer => |n| n != 0,
            .float => |n| n != 0,
            .string => |s| s.len != 0,
            .array => |a| a.items.len != 0,
            .object => |o| o.count() != 0,
            .number_string => true,
            .custom => |c| try c.truthy(),
        };
    }

    fn annotationString(arena: Allocator, value: @import("runtime_value.zig").Value) Error!?[]const u8 {
        if (value == .null or (value == .string and value.string.len == 0)) return null;
        return try responseStringValue(arena, value);
    }

    /// Python `a or b or c` keeps the last operand even when it is false.
    fn annotationAlternative(annotation: @import("runtime_value.zig").Value, comptime names: []const []const u8) error{ParseValue}!@import("runtime_value.zig").Value {
        inline for (names, 0..) |name, i| {
            const value = annotation.object.get(name) orelse .null;
            if (i + 1 == names.len or try responseTruthy(value)) return value;
        }
        unreachable;
    }

    /// Convert only indices which can address this text. Python int truncates
    /// floats and admits Unicode decimal digits; Pydantic integer coercion does
    /// neither, so it is not the right validator here. No big integer allocation
    /// is needed when an index already lies outside the byte-length upper bound.
    pub fn annotationIndex(value: @import("runtime_value.zig").Value, bound: usize) Error!?usize {
        switch (value) {
            .integer => |n| return if (n >= 0 and @as(u64, @intCast(n)) <= bound) @intCast(n) else null,
            .float => |n| {
                if (std.math.isInf(n)) return error.LMFailure; // Source catches ValueError, not OverflowError.
                if (std.math.isNan(n)) return null;
                const truncated = @trunc(n);
                return if (truncated >= 0 and truncated <= @as(f64, @floatFromInt(bound))) @intFromFloat(truncated) else null;
            },
            .number_string => |s| {
                const n = std.fmt.parseInt(usize, s, 10) catch return null;
                return if (n <= bound) n else null;
            },
            .string => |raw| {
                const source = @import("unicode.zig").strip(raw, true);
                if (source.len == 0) return null;
                const negative = source[0] == '-';
                var i: usize = @intFromBool(source[0] == '-' or source[0] == '+');
                var n: usize = 0;
                var digits: usize = 0;
                var previous_digit = false;
                while (i < source.len) {
                    if (source[i] == '_') {
                        if (!previous_digit) return null;
                        previous_digit = false;
                        i += 1;
                        continue;
                    }
                    const size = std.unicode.utf8ByteSequenceLength(source[i]) catch return null;
                    if (size > source.len - i) return null;
                    const cp = std.unicode.utf8Decode(source[i..][0..size]) catch return null;
                    const digit = @import("unicode.zig").decimalDigit(cp) orelse return null;
                    const product = @mulWithOverflow(n, 10);
                    if (product[1] != 0) return null;
                    const sum = @addWithOverflow(product[0], digit);
                    if (sum[1] != 0 or sum[0] > bound) return null;
                    n = sum[0];
                    digits += 1;
                    if (digits > 4300) return null;
                    previous_digit = true;
                    i += size;
                }
                return if (previous_digit and (!negative or n == 0)) n else null;
            },
            else => return null,
        }
    }

    pub fn codepointOffset(text: []const u8, index: usize) ?usize {
        var i: usize = 0;
        var count: usize = 0;
        while (count < index and i < text.len) : (count += 1) {
            const size = std.unicode.utf8ByteSequenceLength(text[i]) catch return null;
            if (size > text.len - i) return null;
            i += size;
        }
        return if (count == index) i else null;
    }

    fn responseCitation(arena: Allocator, annotation: @import("runtime_value.zig").Value, text: []const u8) Error!?CitationPart {
        const url = try annotationString(arena, try annotationAlternative(annotation, &.{ "url", "uri" }));
        const title = try annotationString(arena, try annotationAlternative(annotation, &.{ "title", "filename", "file_id" }));
        var cited: ?[]const u8 = null;
        for ([_][]const u8{ "text", "snippet", "cited_text", "quote" }) |name| {
            if (try annotationString(arena, annotation.object.get(name) orelse .null)) |s| {
                cited = s;
                break;
            }
        }
        if (cited == null) {
            const start = try annotationIndex(annotation.object.get("start_index") orelse .null, text.len);
            const end = try annotationIndex(annotation.object.get("end_index") orelse .null, text.len);
            if (start != null and end != null and start.? < end.?) {
                if (codepointOffset(text, start.?)) |first| {
                    if (codepointOffset(text, end.?)) |last| cited = text[first..last];
                }
            }
        }
        if (url == null and title == null and cited == null) return null;
        return .{ .text = cited, .title = title, .url = url };
    }

    /// lm15's str(value or ""), retaining strings without allocation.
    pub fn responseText(arena: Allocator, input: ?@import("runtime_value.zig").Value) Error![]const u8 {
        const value = input orelse return "";
        if (!try responseTruthy(value)) return "";
        return responseStringValue(arena, value);
    }

    fn responseStringValue(arena: Allocator, value: @import("runtime_value.zig").Value) Error![]const u8 {
        if (value == .string) return value.string;
        var out: Writer.Allocating = .init(arena);
        defer out.deinit();
        @import("pyjson.zig").writeStr(&out.writer, value) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            else => error.LMFailure,
        };
        return out.toOwnedSlice();
    }

    fn responseInput(arena: Allocator, input: ?@import("runtime_value.zig").Value, correlated_raw_id: bool) Error!@import("runtime_value.zig").Value {
        const Value = @import("runtime_value.zig").Value;
        var normalized: Value = .{ .object = .empty };
        if (input) |raw| {
            if (raw == .object) {
                normalized = raw;
            } else if (raw == .string and raw.string.len != 0) {
                const repair = @import("json_repair.zig");
                const decoded = repair.parseJson(arena, raw.string) catch |err| blk: {
                    if (err == error.OutOfMemory) return error.OutOfMemory;
                    // DSPy's repair boundary rechecks the original JSON and only
                    // catches JSONDecodeError, not CPython's integer-size error.
                    if (correlated_raw_id and err == error.IntegerTooLong) return error.LMFailure;
                    if (correlated_raw_id and err == error.ParseValue) {
                        const repaired = repair.loads(arena, raw.string) catch |failure| return switch (failure) {
                            error.OutOfMemory => error.OutOfMemory,
                            else => error.LMFailure,
                        };
                        if (repaired == .object) break :blk repaired;
                    }
                    var partial: @import("runtime_value.zig").ObjectMap = .empty;
                    try partial.put(arena, "partial_json", raw);
                    break :blk Value{ .object = partial };
                };
                if (decoded == .object) {
                    normalized = decoded;
                } else {
                    try normalized.object.put(arena, "value", decoded);
                }
            }
        }
        if (!finiteJson(normalized)) return error.LMFailure;
        return normalized;
    }

    fn responseArguments(arena: Allocator, normalized: @import("runtime_value.zig").Value) Error![]const u8 {
        std.debug.assert(normalized == .object);
        if (normalized.object.count() == 0) return "{}";
        var out: Writer.Allocating = .init(arena);
        defer out.deinit();
        @import("pyjson.zig").writeJsonAscii(&out.writer, normalized) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            else => error.LMFailure,
        };
        return out.toOwnedSlice();
    }

    // Both ToolCallPart input and Response.provider_data require finite JSON.
    fn finiteJson(value: @import("runtime_value.zig").Value) bool {
        switch (value) {
            .float => |number| return std.math.isFinite(number),
            .custom => {
                if (@import("python_value.zig").view(value)) |builtin| {
                    if (builtin == .float) return std.math.isFinite(builtin.float);
                }
            },
            .array => |items| for (items.items) |item| {
                if (!finiteJson(item)) return false;
            },
            .object => |object| for (object.values()) |item| {
                if (!finiteJson(item)) return false;
            },
            else => {},
        }
        return true;
    }

    fn reasoningText(arena: Allocator, item: @import("runtime_value.zig").Value) Error!?[]const u8 {
        if (item.object.get("summary")) |summary| {
            if (summary == .array) {
                var parts: TextParts = .{ .joined = .init(arena) };
                for (summary.array.items) |part| {
                    const value = if (part == .object) part.object.get("text") orelse .null else part;
                    try parts.add(try responseStringValue(arena, value), "\n");
                }
                const joined = parts.value() orelse return null;
                return if (joined.len == 0) null else joined;
            }
            const text = try responseText(arena, summary);
            if (text.len != 0) return text;
        }
        const text = try responseText(arena, item.object.get("text"));
        return if (text.len == 0) null else text;
    }

    fn responseString(object: @import("runtime_value.zig").Value, name: []const u8, nonempty: bool) Error![]const u8 {
        if (object != .object) return error.LMFailure;
        const text = object.object.get(name) orelse return error.LMFailure;
        if (text != .string or (nonempty and text.string.len == 0)) return error.LMFailure;
        return text.string;
    }

    fn stringEquals(value: @import("runtime_value.zig").Value, expected: []const u8) bool {
        return value == .string and std.mem.eql(u8, value.string, expected);
    }

    /// Avoid copying the overwhelmingly common single text/summary part.
    const TextParts = struct {
        first: ?[]const u8 = null,
        count: usize = 0,
        joined: Writer.Allocating,

        fn add(parts: *TextParts, text: []const u8, separator: []const u8) Allocator.Error!void {
            if (parts.count == 0) {
                parts.first = text;
            } else {
                if (parts.count == 1) parts.joined.writer.writeAll(parts.first.?) catch return error.OutOfMemory;
                parts.joined.writer.writeAll(separator) catch return error.OutOfMemory;
                parts.joined.writer.writeAll(text) catch return error.OutOfMemory;
            }
            parts.count += 1;
        }

        fn value(parts: *const TextParts) ?[]const u8 {
            return if (parts.count <= 1) parts.first else parts.joined.writer.buffered();
        }
    };
};

const testing = std.testing;

test "scripted response construction rejects non JSON provider data" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const Value = @import("runtime_value.zig").Value;
    const python = @import("python_value.zig");
    const messages = [_]Message{.{ .role = .user, .content = .{ .text = "question" } }};
    const invalid = [_]Value{
        try python.bytearray(arena, "1"),
        try python.tuple(arena, &.{.{ .string = "1" }}),
        .{ .float = std.math.nan(f64) },
    };
    for (invalid) |child| {
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.put(arena, "nested", child);
        const metadata = [_]?Value{.{ .object = object }};
        var scripted: ScriptedLM = .{ .responses = &.{"answer"}, .provider_data = &metadata };
        try testing.expectError(error.LMFailure, scripted.lm().complete(arena, .{ .messages = &messages }));
    }
    const null_metadata = [_]?Value{.null};
    var with_null: ScriptedLM = .{ .responses = &.{"answer"}, .provider_data = &null_metadata };
    try testing.expectEqualStrings("answer", try with_null.lm().complete(arena, .{ .messages = &messages }));
}

// Builds {"hyperparameters": {"n_epochs": 3, "tags": [tag]}} entirely in `arena`,
// so a restored copy can be distinguished from a view into the source arena.
fn testNestedConfig(arena: Allocator, tag: []const u8) !@import("runtime_value.zig").Value {
    var tags: @import("runtime_value.zig").Array = .init(arena);
    try tags.append(.{ .string = try arena.dupe(u8, tag) });
    var inner: @import("runtime_value.zig").ObjectMap = .empty;
    try inner.put(arena, try arena.dupe(u8, "n_epochs"), .{ .integer = 3 });
    try inner.put(arena, try arena.dupe(u8, "tags"), .{ .array = tags });
    var outer: @import("runtime_value.zig").ObjectMap = .empty;
    try outer.put(arena, try arena.dupe(u8, "hyperparameters"), .{ .object = inner });
    return .{ .object = outer };
}

test "OpenAI provider configuration survives destruction of the source arena" {
    // Dump and restore through the public LM state interfaces, free every byte
    // of the source, then read the nested configuration from the restored model.
    var source_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer source_memory.deinit();
    var target_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer target_memory.deinit();
    const source = source_memory.allocator();
    const target = target_memory.allocator();

    var defaults: OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "m" };
    const default_state = try defaults.lm().dumpState(source);
    try testing.expect(default_state.object.get("finetuning_model").? == .null);
    try testing.expectEqual(@as(usize, 0), default_state.object.get("train_kwargs").?.object.count());
    try testing.expectEqual(@as(usize, 0), default_state.object.get("launch_kwargs").?.object.count());

    var original: OpenAI = .{
        .client = undefined,
        .base_url = "https://saved.invalid/v1",
        .model = "base-model",
        .api_key = "must-not-be-saved",
        .finetuning_model = try source.dupe(u8, "tunable-model"),
        .train_kwargs = try testNestedConfig(source, "train"),
        .launch_kwargs = try testNestedConfig(source, "launch"),
        .chat_reasoning_usage = .separate,
    };
    const state = try original.lm().dumpState(source);
    try testing.expect(state.object.get("api_key") == null);
    try testing.expect(state.object.get("client") == null);
    try testing.expect(state.object.get("header_provider") == null);
    // The dump is a clone: mutating the model afterwards must not alter it.
    const dumped_tags = state.object.get("train_kwargs").?.object.get("hyperparameters").?.object.get("tags").?;
    try testing.expect(dumped_tags.array.items.ptr !=
        original.train_kwargs.object.get("hyperparameters").?.object.get("tags").?.array.items.ptr);

    var loader: OpenAI.StateLoader = .{ .client = undefined, .base_url = "https://trusted.invalid/v1", .api_key = "injected" };
    const interface = loader.loader();
    const restored_lm = try interface.restore(target, state);
    var without_model = try inference.cloneJsonValue(source, state);
    try testing.expect(without_model.object.orderedRemove("finetuning_model"));
    try testing.expect(without_model.object.orderedRemove("train_kwargs"));
    const sparse_lm = try interface.restore(target, without_model);
    _ = source_memory.reset(.free_all);

    const restored: *OpenAI = @ptrCast(@alignCast(restored_lm.ptr));
    try testing.expectEqualStrings("tunable-model", restored.finetuning_model.?);
    try testing.expectEqualStrings("injected", restored.api_key.?);
    const counted = try OpenAI.parseCanonicalChat(target,
        \\{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":3}}}
    , restored.chat_reasoning_usage);
    try expectCount("10", counted.metadata.usage.output_tokens);
    inline for (.{ "train", "launch" }) |tag| {
        const config = @field(restored, tag ++ "_kwargs");
        try testing.expectEqual(@as(usize, 1), config.object.count());
        const inner = config.object.get("hyperparameters").?;
        try testing.expectEqual(@as(i64, 3), inner.object.get("n_epochs").?.integer);
        const tags = inner.object.get("tags").?.array.items;
        try testing.expectEqual(@as(usize, 1), tags.len);
        try testing.expectEqualStrings(tag, tags[0].string);
    }
    // Absent keys keep the Source defaults rather than failing the load.
    const sparse: *OpenAI = @ptrCast(@alignCast(sparse_lm.ptr));
    try testing.expect(sparse.finetuning_model == null);
    try testing.expectEqual(@as(usize, 0), sparse.train_kwargs.object.count());
    try testing.expectEqual(@as(usize, 1), sparse.launch_kwargs.object.count());
}

test "OpenAI Responses preset binding cannot reuse an unrelated endpoint" {
    var provider: OpenAI = .{
        .client = undefined,
        .base_url = "https://unrelated.invalid/v1",
        .model = "fixture",
    };
    const refused = "https://unrelated.invalid/v1";
    try testing.expectError(
        error.MissingResponsesPresetBaseUrl,
        provider.bindResponsesPreset("DEEPSEE\u{212a}", "https://api.openai.com/v1"),
    );
    try testing.expectEqualStrings(refused, provider.base_url);
    try testing.expectEqual(OpenAI.Api.chat_completions, provider.api);
    try testing.expect(provider.responses_compat == null);
    try testing.expectError(
        error.UnknownResponsesPreset,
        provider.bindResponsesPreset(" CHAT ", "http://localhost:9999/v1"),
    );
    try provider.bindResponsesPreset("LM.Studio", null);
    try testing.expectEqualStrings("http://localhost:1234/v1", provider.base_url);
    try testing.expectEqual(
        try OpenAI.ResponsesCompat.preset("ollama"),
        try OpenAI.ResponsesCompat.preset("LM Studio"),
    );
    try testing.expect(
        try OpenAI.ResponsesCompat.preset("vllm") != try OpenAI.ResponsesCompat.preset("sglang"),
    );
    try provider.bindResponsesPreset("dashscope-qwen", "http://localhost:9999/v1");
    try testing.expectEqualStrings("http://localhost:9999/v1", provider.base_url);
    try testing.expectEqual(
        .qwen,
        (try provider.responses_compat.?.resolve()).reasoning_format,
    );
}

test "OpenAI bound Responses policy owns cloned routing and restored controls" {
    var source_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer source_memory.deinit();
    var target_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer target_memory.deinit();
    const source = source_memory.allocator();
    const target = target_memory.allocator();
    var client: std.http.Client = .{ .allocator = target, .io = testing.io };
    defer client.deinit();
    const graph = @import("json_graph.zig");
    const parsed = try graph.parse(source,
        \\{"order":["fast"]}
    );
    const dictionary = try source.create(graph.Dictionary);
    dictionary.* = .{ .entries = .init(source) };
    try dictionary.entries.append(.{
        .key = .{ .string = try source.dupe(u8, "order") },
        .value = parsed.object.get("order").?,
    });
    const routing: graph.Value = .{ .dictionary = dictionary };
    var original: OpenAI = .{
        .client = &client,
        .base_url = "http://unused",
        .model = "fixture",
        .api = .responses,
        .responses_compat = try OpenAI.ResponsesCompat.init(.{
            .max_output_tokens_field = .max_tokens,
            .routing = routing,
            .extensions = routing,
        }),
    };
    const state = try original.lm().dumpState(source);
    const cloned_lm = try original.lm().clone(target);
    try dictionary.entries.items[0].value.array.append(.{ .string = "source mutation" });
    var loader: OpenAI.StateLoader = .{ .client = &client, .base_url = "http://unused" };
    const restored_lm = try loader.loader().restore(target, state);
    _ = source_memory.reset(.free_all);

    const cloned: *OpenAI = @ptrCast(@alignCast(cloned_lm.ptr));
    const restored: *OpenAI = @ptrCast(@alignCast(restored_lm.ptr));
    const cloned_order = cloned.responses_compat.?.routing.?.dictionary.entries.items[0].value.array;
    try cloned_order.append(.{ .string = "backup" });
    const alias = cloned.responses_compat.?.extensions.?.dictionary.entries.items[0].value.array.items;
    try testing.expectEqualStrings("backup", alias[1].string);
    const request = try sdk.Request.fromFields(target, &.{
        .model = "fixture",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "next" } }} }},
        .config = .{ .max_tokens = .{ .integer = 16 } },
    });
    defer request.release();
    for ([_]*OpenAI{ restored, cloned }, [_]usize{ 1, 2 }) |provider, count| {
        const wire = try provider.requestBody(target, request);
        const payload = try OpenAI.parseJson(target, wire.body);
        try testing.expectEqual(@as(i64, 16), payload.object.get("max_tokens").?.integer);
        try testing.expect(!payload.object.contains("max_output_tokens"));
        const order = payload.object.get("provider").?.object.get("order").?.array.items;
        try testing.expectEqual(count, order.len);
        try testing.expectEqualStrings("fast", order[0].string);
    }
}

test "OpenAI Chat model policy survives source destruction and uses first prefix" {
    var source_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer source_memory.deinit();
    var target_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer target_memory.deinit();
    const source = source_memory.allocator();
    const target = target_memory.allocator();
    var client: std.http.Client = .{ .allocator = target, .io = testing.io };
    defer client.deinit();
    const policy = try OpenAI.parseJson(source,
        \\{"max_tokens_field":"max_tokens","routing":{"order":["fast"]},
        \\ "model_overrides":[["family",{"max_tokens_field":"auto","stream_usage":"omit"}],
        \\ ["family-special",{"max_tokens_field":"invalid"}],["bad",{"stream_usage":"invalid"}]]}
    );
    var original: OpenAI = .{
        .client = &client,
        .base_url = "http://unused",
        .model = "bad",
        .chat_compat = try OpenAI.ChatCompat.fromValue(source, policy),
    };
    try testing.expectError(error.MissingChatPresetBaseUrl, original.bindChatPreset("bedrock", null));
    try testing.expectEqualStrings("http://unused", original.base_url);
    const state = try original.lm().dumpState(source);
    const cloned_lm = try original.lm().clone(target);
    var loader: OpenAI.StateLoader = .{ .client = &client, .base_url = "http://unused" };
    const restored_lm = try loader.loader().restore(target, state);
    _ = source_memory.reset(.free_all);

    for ([_]LM{ cloned_lm, restored_lm }) |handle| {
        const provider: *OpenAI = @ptrCast(@alignCast(handle.ptr));
        try testing.expectError(error.LMFailure, provider.chatCompatibility("bad"));
        const request = try sdk.Request.fromFields(target, &.{
            .model = "family-special",
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "next" } }} }},
            .config = .{ .max_tokens = .{ .integer = 16 } },
        });
        defer request.release();
        const prepared = try provider.prepareRequest(target, request, true);
        const payload = try OpenAI.parseJson(target, try prepared.body(target));
        try testing.expectEqual(@as(i64, 16), payload.object.get("max_completion_tokens").?.integer);
        try testing.expect(!payload.object.contains("max_tokens"));
        try testing.expect(payload.object.get("stream").?.bool);
        try testing.expect(!payload.object.contains("stream_options"));
        const order = payload.object.get("provider").?.object.get("order").?.array.items;
        try testing.expectEqualStrings("fast", order[0].string);
    }
}

test "OpenAI training configuration rejects non object and non JSON values" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Value = @import("runtime_value.zig").Value;
    const python = @import("python_value.zig");
    var nested_tuple: @import("runtime_value.zig").ObjectMap = .empty;
    try nested_tuple.put(arena, "nested", try python.tuple(arena, &.{.{ .string = "1" }}));
    var nested_nan: @import("runtime_value.zig").ObjectMap = .empty;
    try nested_nan.put(arena, "nested", .{ .float = std.math.nan(f64) });
    const invalid = [_]Value{
        .null,
        .{ .string = "{}" },
        .{ .array = .init(arena) },
        .{ .object = nested_tuple },
        .{ .object = nested_nan },
    };
    var loader: OpenAI.StateLoader = .{ .client = undefined, .base_url = "http://unused" };
    const interface = loader.loader();
    inline for (.{ "train_kwargs", "launch_kwargs" }) |name| {
        for (invalid) |value| {
            var model: OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "m" };
            @field(model, name) = value;
            try testing.expectError(error.InvalidLMState, model.lm().dumpState(arena));

            var valid: OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "m" };
            var state = try valid.lm().dumpState(arena);
            _ = try interface.restore(arena, state);
            try state.object.put(arena, name, value);
            try testing.expectError(error.InvalidLMState, interface.restore(arena, state));
        }
    }
    var valid: OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "m" };
    var state = try valid.lm().dumpState(arena);
    try state.object.put(arena, "finetuning_model", .{ .integer = 1 });
    try testing.expectError(error.InvalidLMState, interface.restore(arena, state));
}

test "public prompt ingress removes defaults under either max token alias" {
    const allocator = testing.allocator;
    const Map = @FieldType(LM.PromptOptions, "defaults");
    var defaults: Map = .empty;
    defer defaults.deinit(allocator);
    try defaults.put(allocator, "max_tokens", .{ .integer = 25 });
    try defaults.put(allocator, "store", .{ .bool = false });
    var overrides: Map = .empty;
    defer overrides.deinit(allocator);
    try overrides.put(allocator, "max_completion_tokens", .null);

    const input: LM.PromptOptions = .{ .defaults = defaults, .overrides = overrides };
    const prepared = try prompt_options.build(allocator, "mock", "prompt", &input, false, 8);
    defer prepared.request.release();
    try testing.expectEqual(.null, prepared.request.fields.config.max_tokens);
    try testing.expectEqual(@as(?bool, false), prepared.request.fields.config.store);

    try overrides.put(allocator, "max_completion_tokens", .{ .float = 25 });
    try overrides.put(allocator, "max_tokens", .{ .integer = 25 });
    const conflict: LM.PromptOptions = .{ .defaults = defaults, .overrides = overrides };
    try testing.expectError(
        error.InvalidPromptOptions,
        prompt_options.build(allocator, "mock", "prompt", &conflict, false, 8),
    );
}

test "Chat prediction puts tool images after all tool outputs in the turn" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    const model: OpenAI = .{
        .client = &client,
        .base_url = "https://example.invalid/v1",
        .model = "gpt-4o-mini",
    };
    const url = "https://example.invalid/sample.png";
    const image = try @import("image.zig").Image.fromString(arena, url);
    const marker = try image.serialize(arena);
    const messages: []const Message = &.{
        .{ .role = .assistant, .content = .{ .text = "" }, .tool_calls = &.{
            .{ .id = "visual", .name = "camera", .arguments = "{}" },
            .{ .id = "plain", .name = "note", .arguments = "{}" },
        } },
        .{
            .role = .tool,
            .content = try adapter.renderedContent(arena, marker),
            .tool_call_id = "visual",
        },
        .{ .role = .tool, .content = .{ .text = "unchanged" }, .tool_call_id = "plain" },
        .{ .role = .user, .content = .{ .text = "finish" } },
    };
    const canonical = try model.predictionRequest(arena, .{ .messages = messages });
    defer canonical.release();
    const wire = try model.requestBody(arena, canonical);
    const body = try OpenAI.parseJson(arena, wire.body);
    const rows = body.object.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 5), rows.len);
    try testing.expectEqualStrings("visual", rows[1].object.get("tool_call_id").?.string);
    try testing.expectEqualStrings(
        "[image attached in the next message]",
        rows[1].object.get("content").?.string,
    );
    try testing.expectEqualStrings("plain", rows[2].object.get("tool_call_id").?.string);
    try testing.expectEqualStrings("unchanged", rows[2].object.get("content").?.string);
    try testing.expectEqualStrings("user", rows[3].object.get("role").?.string);
    const attachments = rows[3].object.get("content").?.array.items;
    try testing.expectEqual(@as(usize, 2), attachments.len);
    try testing.expectEqualStrings(
        "Attachments from tool call visual (camera):",
        attachments[0].object.get("text").?.string,
    );
    try testing.expectEqualStrings(
        url,
        attachments[1].object.get("image_url").?.object.get("url").?.string,
    );
    try testing.expectEqualStrings("finish", rows[4].object.get("content").?.string);
}

test "Responses moves tool audio into a user message without moving tool images" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    const model: OpenAI = .{
        .client = &client,
        .base_url = "https://example.invalid/v1",
        .model = "gpt-4o-mini",
        .api = .responses,
    };
    const media = [_]@import("runtime_value.zig").Value{
        try OpenAI.parseJson(arena, "{\"type\":\"audio\",\"media_type\":\"audio/wav\",\"data\":\"UklGRg==\"}"),
        try OpenAI.parseJson(arena, "{\"type\":\"image\",\"media_type\":\"image/png\",\"url\":\"https://example.invalid/image.png\"}"),
    };
    const messages: []const Message = &.{
        .{ .role = .assistant, .content = .{ .text = "" }, .tool_calls = &.{
            .{ .id = "media", .name = "capture", .arguments = "{}" },
        } },
        .{ .role = .tool, .tool_call_id = "media", .content = .{ .parts = &media } },
        .{ .role = .user, .content = .{ .text = "summarize" } },
    };
    const canonical = try model.predictionRequest(arena, .{ .messages = messages });
    defer canonical.release();
    const wire = try model.requestBody(arena, canonical);
    const body = try OpenAI.parseJson(arena, wire.body);
    const rows = body.object.get("input").?.array.items;
    try testing.expectEqual(@as(usize, 5), rows.len);
    try testing.expectEqualStrings("media", rows[2].object.get("call_id").?.string);
    const output = rows[2].object.get("output").?.array.items;
    try testing.expectEqualStrings(
        "[audio attached in the next message]",
        output[0].object.get("text").?.string,
    );
    try testing.expectEqualStrings("input_image", output[1].object.get("type").?.string);
    try testing.expectEqualStrings("user", rows[3].object.get("role").?.string);
    const attachment = rows[3].object.get("content").?.array.items;
    try testing.expectEqualStrings(
        "Attachments from tool call media (capture):",
        attachment[0].object.get("text").?.string,
    );
    try testing.expectEqualStrings("input_audio", attachment[1].object.get("type").?.string);
    try testing.expectEqualStrings("UklGRg==", attachment[1].object.get("audio").?.string);
}

test "Chat canonical parts preserve text boundaries across roles" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var model: OpenAI = .{ .client = &client, .base_url = "https://example.invalid/v1", .model = "text" };
    const field: @import("signature.zig").Field = .{
        .name = "question",
        .kind = .str,
        .type_name = "str",
        .desc = "",
        .prefix = "",
    };
    const parts = [_]@import("runtime_value.zig").Value{
        try adapter.formatFieldContent(arena, field, .{ .string = "é\"" }),
        try adapter.formatFieldContent(arena, field, .{ .string = "" }),
    };
    const messages = [_]Message{
        .{ .role = .user, .content = .{ .parts = &parts } },
        .{ .role = .assistant, .content = .{ .parts = &parts } },
        .{ .role = .assistant, .content = .{ .parts = &parts }, .reasoning = "think" },
        .{ .role = .tool, .content = .{ .parts = &parts }, .tool_call_id = "call" },
    };
    const canonical = try model.predictionRequest(arena, .{ .messages = &messages });
    defer canonical.release();
    const wire = try model.requestBody(arena, canonical);
    const body = try OpenAI.parseJson(arena, wire.body);
    const rows = body.object.get("messages").?.array.items;
    const user = rows[0].object.get("content").?.array.items;
    try testing.expectEqual(@as(usize, 2), user.len);
    try testing.expectEqualStrings("text", user[0].object.get("type").?.string);
    try testing.expectEqualStrings("é\"", user[0].object.get("text").?.string);
    try testing.expectEqualStrings("", user[1].object.get("text").?.string);
    try testing.expectEqualStrings("é\"\n", rows[1].object.get("content").?.string);
    try testing.expectEqualStrings("think\né\"\n", rows[2].object.get("content").?.string);
    try testing.expectEqualStrings("é\"\n", rows[3].object.get("content").?.string);
    try testing.expectEqualStrings("call", rows[3].object.get("tool_call_id").?.string);

    const invalid = [_][]const u8{
        \\{"type":"document","media_type":"application/pdf","path":"/must-not-read"}
        ,
        \\{"type":"text","text":true}
        ,
        \\{"type":"image"}
    };
    for (invalid) |json| {
        const part = try OpenAI.parseJson(arena, json);
        const bad_messages = [_]Message{.{ .role = .user, .content = .{ .parts = &.{part} } }};
        try testing.expectError(error.LMFailure, model.lm().completeResponses(arena, .{ .messages = &bad_messages }));
    }
}

test "keyword LM scopes and Chat completion order" {
    var kw: KeywordLM = .{
        .rules = &.{
            .{ .needle = "capital", .response = "A" },
            .{ .needle = "hint", .response = "B", .scope = .any },
        },
        .default = "C",
    };
    const msgs = [_]Message{ .{ .role = .system, .content = .{ .text = "hint" } }, .{ .role = .user, .content = .{ .text = "q" } } };
    try testing.expectEqualStrings("B", kw.respond(&msgs));
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const content = try OpenAI.parseResponse(arena.allocator(),
        \\{"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    );
    try testing.expectEqual(@as(usize, 1), content.len);
    try testing.expectEqualStrings("hi", content[0]);
    const many = try OpenAI.parseResponse(arena.allocator(),
        \\{"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]}
    );
    try testing.expectEqualStrings("b", many[1]);
}

test "replay equality failure leaves the recorded completion available" {
    var local: @import("decimal_environment.zig").LocalContext = .{};
    local.enter(&.{});
    defer local.deinit();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const signaling = try @import("python_value.zig").decimal(a, "sNaN");
    const ordinary = try @import("python_value.zig").decimal(a, "1");
    const messages = [_]Message{.{ .role = .user, .content = .{ .text = "q" } }};
    var used = [_]bool{false};
    var replay: ReplayLM = .{
        .calls = &.{.{ .messages = &messages, .options = .{ .rollout_id = ordinary }, .completion = "available" }},
        .used = &used,
    };
    const model = replay.lm();
    try testing.expectError(error.ParseValue, model.complete(a, .{
        .messages = &messages,
        .options = .{ .rollout_id = signaling },
    }));
    try testing.expect(!used[0]);
    try testing.expectEqual(@as(usize, 0), replay.misses);
    try testing.expectEqualStrings("available", try model.complete(a, .{
        .messages = &messages,
        .options = .{ .rollout_id = ordinary },
    }));
    try testing.expect(replay.exhausted());
}

test "replay LM matches recorded requests exactly and reports misses" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = [_]Message{.{ .role = .user, .content = .{ .text = "a" } }};
    const b = [_]Message{.{ .role = .user, .content = .{ .text = "b" } }};
    var used = [_]bool{ false, false };
    var replay: ReplayLM = .{
        .calls = &.{ .{ .messages = &a, .completion = "A1" }, .{ .messages = &a, .completion = "A2" } },
        .used = &used,
    };
    const replay_lm = replay.lm();
    try testing.expectEqualStrings("A1", try replay_lm.complete(arena.allocator(), .{ .messages = &a }));
    try testing.expect(!replay.exhausted());
    try testing.expectError(error.LMFailure, replay_lm.complete(arena.allocator(), .{ .messages = &b }));
    try testing.expectEqualStrings("A2", try replay_lm.complete(arena.allocator(), .{ .messages = &a }));
    try testing.expectError(error.LMFailure, replay_lm.complete(arena.allocator(), .{ .messages = &a }));
    try testing.expectEqual(@as(usize, 2), replay.misses);
    try testing.expect(!replay.exhausted());

    var inner_used = [_]bool{false};
    var inner: ReplayLM = .{ .calls = &.{.{ .messages = &b, .completion = "B" }}, .used = &inner_used };
    var recorder: RecordingLM = .{ .inner = inner.lm(), .arena = arena.allocator() };
    try testing.expectEqualStrings("B", try recorder.lm().complete(arena.allocator(), .{ .messages = &b }));
    try testing.expect(inner.exhausted());
    try testing.expectEqual(@as(usize, 1), recorder.calls.items.len);
    try testing.expectEqualStrings("b", recorder.calls.items[0].messages[0].content.text);
}

test "usage accounting combines successful groups but retains failed candidates separately" {
    for ([_]bool{ false, true }) |fail_third| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var storage: accounting.Storage = .{ .arena = arena, .io = testing.io };
        var tracker: accounting.UsageTracker = undefined;
        try tracker.init(&storage, .{ .models = 1, .entries = 3 });
        try tracker.addUsage("configured", try accounting.Graph.parse(arena, "{\"prompt_tokens\":1e16}"));
        var engine: ScriptedLM = .{
            .responses = &.{ "first", "second" },
            .model = "configured",
            .metadata = &.{
                .{ .model = "wire", .usage = .{ .input_tokens = .{ .integer = 1 } } },
                .{ .model = "wire", .usage = .{ .input_tokens = .{ .integer = 1 } } },
            },
        };
        const request: Request = .{
            .messages = &.{.{ .role = .user, .content = .{ .text = "group" } }},
            .options = .{ .n = if (fail_third) 3 else 2 },
            .usage_tracker = &tracker,
        };
        if (fail_third) {
            try testing.expectError(error.ScriptExhausted, engine.lm().completeResponses(arena, request));
        } else {
            _ = try engine.lm().completeResponses(arena, request);
        }
        const total = try accounting.Graph.toValue(arena, try tracker.getTotalTokens(arena));
        try testing.expect(total.object.get("wire") == null);
        const count = total.object.get("configured").?.object.get("prompt_tokens").?.float;
        try testing.expectEqual(if (fail_third) @as(f64, 1e16) else 10000000000000002.0, count);
    }
}

test "failed engine keeps primary error and captures every secondary usage refusal" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var storage: accounting.Storage = .{ .arena = arena, .io = testing.io };
    var tracker: accounting.UsageTracker = undefined;
    try tracker.init(&storage, .{ .models = 1, .entries = 0 });
    var engine: ScriptedLM = .{
        .responses = &.{ "first", "second" },
        .metadata = &.{
            .{ .usage = .{ .input_tokens = .{ .integer = 1 } } },
            .{ .usage = .{ .input_tokens = .{ .integer = 1 } } },
        },
    };
    var secondary: AccountingFailure = .{};
    const request: Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "failure" } }},
        .options = .{ .n = 3 },
        .usage_tracker = &tracker,
        .usage_failure = &secondary,
    };
    try testing.expectError(error.ScriptExhausted, engine.lm().completeResponses(arena, request));
    try testing.expectEqual(error.CapacityExceeded, secondary.first.?);
    try testing.expectEqual(@as(u32, 2), secondary.count);
    engine.index = 0;
    var success = request;
    success.options.n = 2;
    try testing.expectError(error.CapacityExceeded, engine.lm().completeResponses(arena, success));
    try testing.expect(secondary.first == null);
    try testing.expectEqual(@as(u32, 0), secondary.count);
}

test "retained engine usage owns wide counters after call storage is destroyed" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var storage: accounting.Storage = .{ .arena = arena, .io = testing.io };
    var tracker: accounting.UsageTracker = undefined;
    try tracker.init(&storage, .{ .models = 1, .entries = 1 });
    {
        var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
        defer scratch.deinit();
        const transient = scratch.allocator();
        const digits = try transient.dupe(u8, "184467440737095516170000000000000000003");
        var engine: ScriptedLM = .{
            .model = "configured",
            .responses = &.{"answer"},
            .metadata = &.{.{ .usage = .{ .input_tokens = .{ .number_string = digits } } }},
        };
        _ = try engine.lm().completeResponses(transient, .{
            .messages = &.{.{ .role = .user, .content = .{ .text = "wide" } }},
            .usage_tracker = &tracker,
        });
        @memset(digits, '0');
    }
    const total = try accounting.Graph.toValue(arena, try tracker.getTotalTokens(arena));
    try expectCount("184467440737095516170000000000000000003", total.object.get("configured").?.object.get("prompt_tokens").?);
}

/// Counters are opaque canonical integers, so compare by exact numeric order.
fn expectCount(expected: []const u8, actual: @import("runtime_value.zig").Value) !void {
    try testing.expect(actual != .null);
    const order = try @import("value.zig").numericOrder(.{ .number_string = expected }, actual);
    try testing.expectEqual(std.math.Order.eq, order.?);
}

test "Chat separate reasoning usage preserves exact sums and unknown counters" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const cases = [_]struct { []const u8, ?[]const u8, ?[]const u8, ?[]const u8 }{
        .{
            \\{"choices":[],"usage":{"prompt_tokens":9007199254740993,"completion_tokens":18446744073709551617,"total_tokens":36902495346673844229,"completion_tokens_details":{"reasoning_tokens":18446744073709551619}}}
            ,
            "9007199254740993",
            "36893488147419103236",
            "36902495346673844229",
        },
        .{
            \\{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}
            ,
            "5",
            null,
            null,
        },
        .{
            \\{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}
            ,
            "5",
            "7",
            "12",
        },
        .{
            \\{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":15}}
            ,
            "5",
            null,
            "15",
        },
        .{
            \\{"choices":[],"usage":{"completion_tokens":7,"total_tokens":7}}
            ,
            null,
            null,
            "7",
        },
        .{
            \\{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12,"completion_tokens_details":{"reasoning_tokens":3}}}
            ,
            null,
            null,
            "12",
        },
        .{
            \\{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":-2,"total_tokens":8,"completion_tokens_details":{"reasoning_tokens":5}}}
            ,
            "5",
            null,
            "8",
        },
    };
    inline for (cases) |row| {
        const completion = try OpenAI.parseCanonicalChat(arena, row[0], .separate);
        inline for (.{ "input_tokens", "output_tokens", "total_tokens" }, 1..) |name, index| {
            const actual = @field(completion.metadata.usage, name);
            if (row[index]) |expected| {
                try expectCount(expected, actual);
            } else {
                try testing.expect(actual == .null);
            }
        }
    }
}

test "Chat body metadata keeps reported counts and withholds undetermined ones" {
    // Decodes real Chat bodies through the public parser. The raw provider
    // JSON must stay byte-equivalent to the body while the canonical usage
    // follows lm15 MAP-19: detail counters are parts, never added on top.
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const body =
        \\{"id":"chatcmpl-1","model":"served-model","choices":[
        \\{"message":{"content":"one"},"finish_reason":"length"},
        \\{"message":{"content":"two"},"finish_reason":"provider_specific"}],
        \\"usage":{"prompt_tokens":11,"completion_tokens":7.0,"total_tokens":18,
        \\"prompt_tokens_details":{"cached_tokens":4},
        \\"completion_tokens_details":{"reasoning_tokens":3}}}
    ;
    const completions = try OpenAI.parseCompletions(arena, body);
    try testing.expectEqual(@as(usize, 2), completions.len);
    const first = completions[0].metadata;
    try testing.expectEqualStrings("chatcmpl-1", first.id.?);
    try testing.expectEqualStrings("served-model", first.model.?);
    try testing.expectEqual(FinishReason.length, first.finish_reason.?);
    try expectCount("11", first.usage.input_tokens);
    try expectCount("7", first.usage.output_tokens);
    try expectCount("18", first.usage.total_tokens);
    try expectCount("4", first.usage.cache_read_tokens);
    try expectCount("3", first.usage.reasoning_tokens);
    try testing.expect(first.usage.cache_write_tokens == .null);
    try testing.expect(first.usage.input_audio_tokens == .null);
    // Explicit choice ingestion retains the same reported response-wide usage.
    try testing.expectEqual(FinishReason.stop, completions[1].metadata.finish_reason.?);
    try testing.expectEqualStrings("chatcmpl-1", completions[1].metadata.id.?);
    try expectCount("11", completions[1].metadata.usage.input_tokens);
    try expectCount("7", completions[1].metadata.usage.output_tokens);
    const raw = try OpenAI.parseJson(arena, body);
    try testing.expect(try inference.jsonEqual(raw, try completions[0].provider_data.?.toValue(arena)));

    // A cached count above the prompt count withholds input only.
    const part_over_whole = (try OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":5,
        \\"completion_tokens":7,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":9}}}
    ))[0].metadata;
    try testing.expect(part_over_whole.id == null);
    try testing.expect(part_over_whole.model == null);
    try testing.expectEqual(FinishReason.stop, part_over_whole.finish_reason.?);
    try testing.expect(part_over_whole.usage.input_tokens == .null);
    try expectCount("7", part_over_whole.usage.output_tokens);
    try expectCount("12", part_over_whole.usage.total_tokens);
    try expectCount("9", part_over_whole.usage.cache_read_tokens);

    // A total which is not input + output keeps only the provider's total.
    const total_mismatch = (try OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":"x"}}],
        \\"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":99}}
    ))[0].metadata;
    try testing.expect(total_mismatch.usage.input_tokens == .null);
    try testing.expect(total_mismatch.usage.output_tokens == .null);
    try expectCount("99", total_mismatch.usage.total_tokens);

    // A count beyond 64 bits stays exact, and unknown is not zero.
    const wide = (try OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"total_tokens":18446744073709551617}}
    ))[0].metadata;
    try expectCount("18446744073709551617", wide.usage.total_tokens);
    try testing.expect(wide.usage.input_tokens == .null);

    const absent = (try OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":"x"}}],"usage":null}
    ))[0].metadata;
    try testing.expect(!usageKnown(absent.usage));

    // wire_count accepts the negative wire integer; settle_usage withholds an
    // input smaller than its known cache sum (zero even with no cache details).
    const negative_input = (try OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":-1}}
    ))[0].metadata;
    try testing.expect(negative_input.usage.input_tokens == .null);
    try testing.expect(!usageKnown(negative_input.usage));

    const refused = [_][]const u8{
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"completion_tokens":true}}
        ,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"completion_tokens":1.5}}
        ,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"completion_tokens":-1}}
        ,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"total_tokens":"7"}}
        ,
        \\{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens_details":{"cached_tokens":[1]}}}
        ,
        \\{"choices":[{"message":{"content":"x"}}],"usage":"12 tokens"}
    };
    for (refused) |malformed| {
        try testing.expectError(error.LMFailure, OpenAI.parseCompletions(arena, malformed));
    }
}

test "Responses body metadata maps status, tool calls and detail counters" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const body =
        \\{"id":"resp_1","model":"served-model","status":"incomplete",
        \\"incomplete_details":{"reason":"max_output_tokens"},
        \\"output":[{"type":"message","content":[{"type":"output_text","text":"cut"}]}],
        \\"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,
        \\"input_tokens_details":{"cached_tokens":2},
        \\"output_tokens_details":{"reasoning_tokens":1}}}
    ;
    const truncated = try OpenAI.parseResponsesResponse(arena, body);
    try testing.expectEqualStrings("resp_1", truncated.metadata.id.?);
    try testing.expectEqualStrings("served-model", truncated.metadata.model.?);
    try testing.expectEqual(FinishReason.length, truncated.metadata.finish_reason.?);
    try expectCount("10", truncated.metadata.usage.input_tokens);
    try expectCount("4", truncated.metadata.usage.output_tokens);
    try expectCount("14", truncated.metadata.usage.total_tokens);
    try expectCount("2", truncated.metadata.usage.cache_read_tokens);
    try expectCount("1", truncated.metadata.usage.reasoning_tokens);
    try testing.expect(try inference.jsonEqual(try OpenAI.parseJson(arena, body), try truncated.provider_data.?.toValue(arena)));

    const filtered = try OpenAI.parseResponsesResponse(arena,
        \\{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}
    );
    try testing.expectEqual(FinishReason.content_filter, filtered.metadata.finish_reason.?);
    try testing.expect(filtered.metadata.id == null);
    try testing.expect(filtered.metadata.model == null);
    try testing.expect(!usageKnown(filtered.metadata.usage));

    // A tool call wins over the status, as in the SDK.
    const called = try OpenAI.parseResponsesResponse(arena,
        \\{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
        \\"output":[{"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"}]}
    );
    try testing.expectEqual(FinishReason.tool_call, called.metadata.finish_reason.?);

    try testing.expectError(error.LMFailure, OpenAI.parseResponsesResponse(arena,
        \\{"output":[],"usage":{"output_tokens_details":{"reasoning_tokens":0.5}}}
    ));
}

test "scripted metadata survives recording, replay and serialization without the source arena" {
    // Scripted -> Recording -> Replay -> value round trip. The scripted
    // source strings are overwritten and freed after recording, so any alias
    // into them would be observed as corrupted metadata.
    const Value = @import("runtime_value.zig").Value;
    var kept_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer kept_state.deinit();
    const kept = kept_state.allocator();
    const messages = [_]Message{.{ .role = .user, .content = .{ .text = "question" } }};

    var recorder: RecordingLM = undefined;
    {
        var source_state: std.heap.ArenaAllocator = .init(testing.allocator);
        defer source_state.deinit();
        const source = source_state.allocator();
        const id = try source.dupe(u8, "resp-source");
        const model = try source.dupe(u8, "served-model");
        const wide = try source.dupe(u8, "18446744073709551617");
        var raw_usage: @import("runtime_value.zig").ObjectMap = .empty;
        try raw_usage.put(source, "prompt_tokens", .{ .integer = 999 });
        var raw: @import("runtime_value.zig").ObjectMap = .empty;
        try raw.put(source, "usage", .{ .object = raw_usage });
        const provider_data = [_]?Value{ .{ .object = raw }, null };
        const metadata = [_]ResponseMetadata{
            .{
                .id = id,
                .model = model,
                .finish_reason = .content_filter,
                .usage = .{ .input_tokens = .{ .integer = 3 }, .cache_read_tokens = .{ .integer = 0 }, .total_tokens = .{ .number_string = wide } },
            },
            .{},
        };
        var scripted: ScriptedLM = .{
            .responses = &.{ "first", "legacy" },
            .provider_data = &provider_data,
            .metadata = &metadata,
        };
        recorder = .{ .inner = scripted.lm(), .arena = kept };
        _ = try recorder.lm().completeResponses(source, .{ .messages = &messages, .options = .{ .n = 2 } });
        @memset(id, '!');
        @memset(model, '!');
        @memset(wide, '9');
    }

    var used = [_]bool{ false, false };
    var replay: ReplayLM = .{ .calls = recorder.calls.items, .used = &used };
    const replayed = try replay.lm().completeResponses(kept, .{ .messages = &messages, .options = .{ .n = 2 } });
    try testing.expect(replay.exhausted());
    for ([_]Completion{ replayed[0], try completionFromValue(kept, try completionValue(kept, replayed[0])) }) |completion| {
        try testing.expectEqualStrings("resp-source", completion.metadata.id.?);
        try testing.expectEqualStrings("served-model", completion.metadata.model.?);
        try testing.expectEqual(FinishReason.content_filter, completion.metadata.finish_reason.?);
        try expectCount("3", completion.metadata.usage.input_tokens);
        // A reported zero is a count; an unreported counter stays unknown.
        try expectCount("0", completion.metadata.usage.cache_read_tokens);
        try expectCount("18446744073709551617", completion.metadata.usage.total_tokens);
        try testing.expect(completion.metadata.usage.output_tokens == .null);
        // Metadata is never derived from, or written into, raw provider JSON.
        try testing.expectEqual(@as(i64, 999), completion.provider_data.?.root.object.get("usage").?.object.get("prompt_tokens").?.integer);
        try testing.expectEqual(@as(usize, 1), completion.provider_data.?.root.object.count());
    }
    // The second completion had none: it stays unknown and keeps the legacy shape.
    try testing.expect(!replayed[1].metadata.known());
}

test "metadata boundaries refuse malformed counts, identities and lengths" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const messages = [_]Message{.{ .role = .user, .content = .{ .text = "question" } }};

    const refused = [_]ResponseMetadata{
        .{ .id = "" },
        .{ .model = "" },
        .{ .usage = .{ .input_tokens = .{ .bool = true } } },
        .{ .usage = .{ .output_tokens = .{ .integer = -1 } } },
        .{ .usage = .{ .total_tokens = .{ .float = 1.5 } } },
        // An explicit total must be input + output.
        .{ .usage = .{ .input_tokens = .{ .integer = 3 }, .output_tokens = .{ .integer = 1 }, .total_tokens = .{ .integer = 9 } } },
        // A part never exceeds its whole.
        .{ .usage = .{ .output_tokens = .{ .integer = 1 }, .reasoning_tokens = .{ .integer = 2 } } },
    };
    for (refused) |metadata| {
        var scripted: ScriptedLM = .{ .responses = &.{"answer"}, .metadata = &.{metadata} };
        try testing.expectError(error.LMFailure, scripted.lm().completeResponses(arena, .{ .messages = &messages }));
        var used = [_]bool{false};
        var replay: ReplayLM = .{
            .calls = &.{.{ .messages = &messages, .completion = "answer", .metadata = &.{metadata} }},
            .used = &used,
        };
        try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, .{ .messages = &messages }));
    }

    var mismatched: ScriptedLM = .{ .responses = &.{"answer"}, .metadata = &.{ .{}, .{} } };
    try testing.expectError(error.LMFailure, mismatched.lm().completeResponses(arena, .{ .messages = &messages }));
    var used = [_]bool{false};
    var replay: ReplayLM = .{
        .calls = &.{.{ .messages = &messages, .completion = "answer", .metadata = &.{ .{}, .{} } }},
        .used = &used,
    };
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, .{ .messages = &messages }));

    const malformed = [_][]const u8{
        \\{"text":"x","metadata":"stop"}
        ,
        \\{"text":"x","metadata":{"finish_reason":"done"}}
        ,
        \\{"text":"x","metadata":{"id":""}}
        ,
        \\{"text":"x","metadata":{"model":7}}
        ,
        \\{"text":"x","metadata":{"usage":{"input_tokens":true}}}
        ,
        \\{"text":"x","metadata":{"usage":{"input_tokens":2,"cache_read_tokens":3}}}
        ,
        \\{"text":"x","metadata":{"cost":1}}
    };
    for (malformed) |text| {
        try testing.expectError(error.BadMetadata, completionFromValue(arena, try OpenAI.parseJson(arena, text)));
    }
    // Absent, null and empty metadata all mean unknown.
    for ([_][]const u8{
        \\{"text":"x"}
        ,
        \\{"text":"x","metadata":null}
        ,
        \\{"text":"x","metadata":{}}
    }) |text| {
        const completion = try completionFromValue(arena, try OpenAI.parseJson(arena, text));
        try testing.expect(!completion.metadata.known());
    }
}

fn cloneThenFreeCompletion(allocator: Allocator, completion: Completion) !void {
    const copy = try cloneCompletion(allocator, completion);
    defer freeCompletion(allocator, copy);
    try testing.expect(copy.metadata.id.?.ptr != completion.metadata.id.?.ptr);
    try testing.expectEqualStrings(completion.metadata.id.?, copy.metadata.id.?);
    try expectCount("18446744073709551617", copy.metadata.usage.total_tokens);
    const data = completionMessage(&copy).?.parts[0].data.value.root;
    try testing.expect(data.object == copy.provider_data.?.root.object);
    try testing.expect(data.object != completion.provider_data.?.root.object);
}

test "completion clone with metadata unwinds every allocation failure" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    var raw: @import("runtime_value.zig").ObjectMap = .empty;
    try raw.put(arena, "kept", .{ .string = "raw" });
    const payload = try jsonPayload(arena, .{ .object = raw });
    const completion: Completion = .{
        .text = "answer",
        .provider_data = payload,
        .message = .{ .role = .assistant, .parts = &.{.{ .data = .{ .value = payload } }} },
        .metadata = try checkedMetadata(arena, .{
            .id = "resp-1",
            .model = "served-model",
            .finish_reason = .stop,
            .usage = .{ .total_tokens = .{ .number_string = "18446744073709551617" } },
        }),
    };
    try testing.checkAllAllocationFailures(testing.allocator, cloneThenFreeCompletion, .{completion});
    var published = try cloneCompletion(testing.allocator, completion);
    defer freeCompletion(testing.allocator, published);
    try publishResponse(testing.allocator, &published, "served-model", .owned);
    try testing.expect(published.message == null);
    try testing.expect(published.response != null);
    try testing.checkAllAllocationFailures(testing.allocator, cloneThenFreeCompletion, .{published});
}

test "provider publication owns borrowed wire continuation leaves" {
    const response = block: {
        var source: std.heap.ArenaAllocator = .init(testing.allocator);
        defer source.deinit();
        var wire = try OpenAI.parseResponsesResponse(source.allocator(),
            \\{"output":[{"type":"reasoning","id":"reason-1","encrypted_content":"opaque","summary":[{"text":"plan"}]}]}
        );
        try publishResponse(testing.allocator, &wire, "configured-model", .borrowed);
        try testing.expect(wire.message == null);
        break :block wire.response.?;
    };
    defer response.release();
    const thinking = response.fields.message.parts[0].thinking;
    try testing.expectEqualStrings("plan", thinking.text);
    try testing.expectEqualStrings("openai", thinking.continuation[0].provider);
    try testing.expectEqualStrings("reasoning_item", thinking.continuation[0].kind);
    try testing.expectEqualStrings("opaque", thinking.continuation[0].data.root.object.get("encrypted_content").?.string);
    try testing.expectEqualStrings("configured-model", response.fields.model);
}

test "provider artifacts retain adjacent text and part-local continuation before tool ids" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const chat = (try OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"reasoning_content":"plan","content":[{"type":"text","text":"A"},{"type":"text","text":""},{"type":"text","text":"B"}],"refusal":"no","tool_calls":[{"function":{"name":"lookup","arguments":"{\"key\":[1,true]}"}}]}}]}
    ))[0];
    const chat_parts = chat.message.?.parts;
    try testing.expectEqual(@as(usize, 6), chat_parts.len);
    try testing.expectEqualStrings("plan", chat_parts[0].thinking.text);
    try testing.expectEqualStrings("A", chat_parts[1].text.text);
    try testing.expectEqualStrings("", chat_parts[2].text.text);
    try testing.expectEqualStrings("B", chat_parts[3].text.text);
    try testing.expectEqualStrings("no", chat_parts[4].refusal.text);
    try testing.expectEqualStrings("call_5", chat_parts[5].tool_call.id);
    const input = chat_parts[5].tool_call.input.root.object.get("key").?.array.items;
    try testing.expectEqual(@as(i64, 1), input[0].integer);
    try testing.expect(input[1].bool);
    const response = try OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"output_text","text":"A"},{"type":"output_text","text":"B"}]},{"type":"reasoning","id":"reason-1","encrypted_content":"opaque","summary":[{"text":"one"},{"text":"two"}]},{"type":"function_call","name":"lookup","arguments":"{}"},{"type":"message","content":[{"type":"output_image","b64_json":"YQ=="},{"type":"refusal","refusal":""},{"type":"output_audio","audio":{"data":"Yg=="}}]}]}
    );
    const message = response.message.?;
    try testing.expectEqual(@as(usize, 0), message.continuation.len);
    try testing.expectEqual(@as(usize, 7), message.parts.len);
    try testing.expectEqualStrings("A", message.parts[0].text.text);
    try testing.expectEqualStrings("B", message.parts[1].text.text);
    try testing.expectEqualStrings("one\ntwo", message.parts[2].thinking.text);
    try testing.expectEqual(@as(usize, 1), message.parts[2].thinking.continuation.len);
    try testing.expectEqualStrings("opaque", message.parts[2].thinking.continuation[0].data.root.object.get("encrypted_content").?.string);
    try testing.expectEqualStrings("call_3", message.parts[3].tool_call.id);
    try testing.expectEqualStrings("YQ==", message.parts[4].image.data.?);
    try testing.expectEqualStrings("", message.parts[5].text.text);
    try testing.expectEqualStrings("audio/wav", message.parts[6].audio.media_type);
}

test "Responses retain ordered unmapped blocks without classifying executed tools" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const response = try OpenAI.parseResponsesResponse(arena,
        \\{"_lm15_unmapped":[{"path":"old","type":"old"}],"output":[null,{"type":"web_search_call"},{"type":"message","content":[42,{"type":["future"]},{"type":"output_text","text":"visible"}]},{"type":"later"}]}
    );
    try testing.expectEqualStrings("visible", response.text.?);
    const records = response.provider_data.?.root.object.get("_lm15_unmapped").?.array.items;
    const paths = [_][]const u8{ "output[0]", "output[2].content[0]", "output[2].content[1]", "output[3]" };
    const kinds = [_][]const u8{ "NoneType", "int", "['future']", "later" };
    try testing.expectEqual(paths.len, records.len);
    for (records, paths, kinds) |record, path, kind| {
        try testing.expectEqualStrings(path, record.object.get("path").?.string);
        try testing.expectEqualStrings(kind, record.object.get("type").?.string);
    }
    const unchanged = try OpenAI.parseResponsesResponse(arena,
        \\{"_lm15_unmapped":[{"path":"old","type":"old"}],"output":[{"type":"file_search_call"}]}
    );
    try testing.expectEqualStrings("old", unchanged.provider_data.?.root.object.get("_lm15_unmapped").?.array.items[0].object.get("type").?.string);
}

test "Responses compatibility distinguishes direct and nested empty overrides" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    const provider: OpenAI = .{ .client = &client, .base_url = "https://example.invalid/v1", .model = "unused", .api = .responses };
    const extensions = [_][]const u8{
        \\{"openai_responses_compat":{},"openai_compat":{"developer_role":"system"}}
        ,
        \\{"compat":{"openai_responses":{},"openai":{"developer_role":"system"}}}
        ,
        \\{"openai_responses_compat":false,"openai_compat":{"developer_role":"system"}}
    };
    for (extensions, [_][]const u8{ "developer", "system", "developer" }) |extra, role| {
        const input = try OpenAI.parseJson(arena,
            \\{"model":"configured","messages":[{"role":"developer","parts":[{"type":"text","text":"policy"}]}],"config":{}}
        );
        try input.object.getPtr("config").?.object.put(arena, "extensions", try OpenAI.parseJson(arena, extra));
        const request = try sdk.Request.fromValue(arena, input);
        defer request.release();
        const wire = try provider.requestBody(arena, request);
        const payload = try OpenAI.parseJson(arena, wire.body);
        try testing.expectEqualStrings(role, payload.object.get("input").?.array.items[0].object.get("role").?.string);
    }
    // Valid Chat policy words must not escape the Responses domain.
    const patch = try OpenAI.parseJson(arena,
        \\{"openai_responses_compat":{"reasoning_format":"kimi"}}
    );
    const invalid = try sdk.Request.fromFields(arena, &.{
        .model = "configured",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "next" } }} }},
        .config = .{ .extensions = try sdk.JsonValue.fromBorrowedValue(arena, patch) },
    });
    defer invalid.release();
    try testing.expectError(error.LMFailure, provider.requestBody(arena, invalid));
}

test "native Responses replay retains separate opaque reasoning items" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const first = try OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"reasoning","id":"first","encrypted_content":"opaque-1"},{"type":"reasoning","id":"second","encrypted_content":"opaque-2"},{"type":"message","content":[{"type":"output_text","text":"answer"}]}]}
    );
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    const provider: OpenAI = .{ .client = &client, .base_url = "https://example.invalid/v1", .model = "gpt-5.6-sol", .api = .responses };
    var messages = [_]Message{.{
        .role = .assistant,
        .content = .{ .text = first.text.? },
        .continuation = first.continuation,
    }};
    const request = try provider.predictionRequest(arena, .{ .messages = &messages });
    defer request.release();
    const wire = try provider.requestBody(arena, request);
    const payload = try OpenAI.parseJson(arena, wire.body);
    const items = payload.object.get("input").?.array.items;
    try testing.expectEqual(@as(usize, 3), items.len);
    for (items[0..2], [_][]const u8{ "first", "second" }, [_][]const u8{ "opaque-1", "opaque-2" }) |item, id, encrypted| {
        try testing.expectEqualStrings("reasoning", item.object.get("type").?.string);
        try testing.expectEqualStrings(id, item.object.get("id").?.string);
        try testing.expectEqualStrings(encrypted, item.object.get("encrypted_content").?.string);
        try testing.expectEqual(@as(usize, 0), item.object.get("summary").?.array.items.len);
    }
    try testing.expectEqualStrings("answer", items[2].object.get("content").?.array.items[0].object.get("text").?.string);
    // A flattened summary cannot be assigned to one of multiple states.
    messages[0].reasoning = "ambiguous summary";
    try testing.expectError(error.LMFailure, provider.predictionRequest(arena, .{ .messages = &messages }));
}

test "Responses reasoning continuation stringifies truthy state and omits falsy state" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const response = try OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"reasoning","summary":"first","id":42,"encrypted_content":{"key":1}},{"type":"reasoning","summary":"second","id":0,"encrypted_content":[]} ,{"type":"message","content":[{"type":"output_text","text":"visible"}]}]}
    );
    const parts = response.message.?.parts;
    const state = parts[0].thinking.continuation[0].data.root.object;
    try testing.expectEqualStrings("42", state.get("id").?.string);
    try testing.expectEqualStrings("{'key': 1}", state.get("encrypted_content").?.string);
    try testing.expectEqual(@as(usize, 0), parts[1].thinking.continuation.len);
    try testing.expectEqualStrings("firstsecond", response.reasoning.?);
    try testing.expectEqualStrings("visible", parts[2].text.text);
}

test "judgment folding requires one strict object text and retains continuation" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const request = try sdk.Request.fromValue(arena, try OpenAI.parseJson(arena,
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"state"}]}],"config":{"response_format":{"type":"json_schema","name":"judgment","schema":{"properties":{"pick":{"type":"boolean"},"ordinary":{"type":"string"}}}}}}
    ));
    defer request.release();
    const continuation = [_]Continuation{.{
        .provider = "fixture",
        .kind = "opaque",
        .data = try jsonPayload(arena, try OpenAI.parseJson(arena, "{\"opaque\":\"state\"}")),
    }};
    const original = [_]sdk.Part{
        .{ .thinking = .{ .text = "plan" } },
        .{ .text = .{ .text = "{\"not_declared\":1}", .continuation = &continuation } },
        .{ .refusal = .{ .text = "notice" } },
    };
    var completion: Completion = .{ .text = original[1].text.text, .message = .{ .role = .assistant, .parts = &original } };
    try OpenAI.foldJudgments(arena, &completion, request);
    const parts = completion.message.?.parts;
    try testing.expectEqualStrings("plan", parts[0].thinking.text);
    try testing.expectEqual(@as(i64, 1), parts[1].data.value.root.object.get("not_declared").?.integer);
    try testing.expectEqualStrings("state", parts[1].data.continuation[0].data.root.object.get("opaque").?.string);
    try testing.expectEqualStrings("notice", parts[2].refusal.text);
    try testing.expect(original[1] == .text);
    for ([_][]const u8{ "{'pick':true}", "{\"pick\":", "[true]", "null" }) |text| {
        const unchanged = [_]sdk.Part{.{ .text = .{ .text = text } }};
        completion.message = .{ .role = .assistant, .parts = &unchanged };
        try OpenAI.foldJudgments(arena, &completion, request);
        try testing.expectEqualStrings(text, completion.message.?.parts[0].text.text);
    }
    const multiple = [_]sdk.Part{ original[1], .{ .text = .{ .text = "" } } };
    completion.message = .{ .role = .assistant, .parts = &multiple };
    try OpenAI.foldJudgments(arena, &completion, request);
    try testing.expect(completion.message.?.parts[0] == .text);
    const nonfinite = [_]sdk.Part{.{ .text = .{ .text = "{\"pick\":Infinity}" } }};
    completion.message = .{ .role = .assistant, .parts = &nonfinite };
    try testing.expectError(error.LMFailure, OpenAI.foldJudgments(arena, &completion, request));
}

test "TypeSafe artifacts preserve distributions and opaque box numeric kinds" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const schema = try OpenAI.parseJson(arena,
        \\{"type":"object","properties":{"flag":{"type":"boolean"},"box":{"type":"object","properties":{"top":{"type":"number"},"bottom":{"type":"number"},"left":{"type":"number"},"right":{"type":"number"}}}}}
    );
    const response = try @import("typesafe.zig").parseResponse(arena, schema,
        \\{"answers":{"flag":{"type":"noul","noul":0.8},"box":{"type":"box","box":{"top":0,"bottom":1,"left":0.2,"right":0.8}}}}
    );
    const data = response.message.?.parts[0].data;
    try testing.expect(data.value.root.object.get("flag").?.bool);
    try testing.expectEqual(@as(i64, 0), data.value.root.object.get("box").?.object.get("top").?.integer);
    try testing.expectEqual(@as(i64, 1), data.value.root.object.get("box").?.object.get("bottom").?.integer);
    const distribution = data.probabilities.?.root.object.get("flag").?.object;
    try testing.expectEqual(@as(f64, 0.8), distribution.get("true").?.float);
    try testing.expectEqual(@as(f64, 0.19999999999999996), distribution.get("false").?.float);
    try testing.expect(data.probabilities.?.root.object.get("box") == null);
    try testing.expectEqual(sdk.JudgmentMethod.provider_classification, data.method.?);
    const box_only = try OpenAI.parseJson(arena,
        \\{"type":"object","properties":{"box":{"type":"object","properties":{"top":{"type":"number"},"bottom":{"type":"number"},"left":{"type":"number"},"right":{"type":"number"}}}}}
    );
    const box_response = try @import("typesafe.zig").parseResponse(arena, box_only,
        \\{"answers":{"box":{"type":"box","box":{"top":0,"bottom":1,"left":0.2,"right":0.8}}}}
    );
    try testing.expect(box_response.message.?.parts[0].data.probabilities == null);
    try testing.expect(box_response.message.?.parts[0].data.method == null);
}

test "Responses media counts canonical parts and rejects malformed payloads before reported billing" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const response = try OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"output_image","b64_json":"data:image/png;base64,AQ=="},{"type":"output_audio","audio":{"data":"Ag=="}}]},{"type":"function_call","name":"run","arguments":"{}"}]}
    );
    try testing.expect(response.generated_media);
    try testing.expect(response.text == null);
    try testing.expectEqualStrings("call_2", response.tool_calls[0].id);
    for ([_][]const u8{
        \\{"output":[{"type":"message","content":[{"type":"output_image","b64_json":1}]}],"_hidden_params":{"response_cost":0}}
        ,
        \\{"output":[{"type":"message","content":[{"type":"output_audio","audio":{"data":"!!!!"}}]}]}
    }) |body| try testing.expectError(error.LMFailure, OpenAI.parseResponsesResponse(arena, body));
    const empty = try OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"output_audio","audio":{"data":""}}]}]}
    );
    try testing.expect(!empty.generated_media);
    try testing.expectEqualStrings("", empty.text.?);
}

test "Chat field listener bounds pending text and held whitespace independently" {
    var marker: [32]u8 = undefined;
    var pending: [32]u8 = undefined;
    var output: [4]u8 = undefined;
    var listener = try ChatStreamListener.init(.{
        .wire_field = "answer",
        .marker_storage = &marker,
        .pending_storage = &pending,
        .output_storage = &output,
    });
    const oversized: [33]u8 = @splat('[');
    try testing.expectError(error.BufferTooSmall, listener.receive(&oversized));
    _ = try listener.receive("[[ ## answer ## ]]");
    const first = (try listener.receive("x  ")).?;
    try testing.expectEqualStrings("x", first.text);
    _ = try listener.receive("  ");
    try testing.expectError(error.BufferTooSmall, listener.receive("y"));
    _ = try listener.receive("[[ ## answer ## ]]");
    const recovered = (try listener.receive("yes")).?;
    try testing.expectEqualStrings("yes", recovered.text);
    const last = (try listener.receive("[[ ## completed ## ]]")).?;
    try testing.expectEqualStrings("", last.text);
    try testing.expect(last.is_last_chunk);
}

test "Responses streamed stop preserves whole scores and discards later queued parts" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const bytes =
        \\data: {"type":"response.created","response":{"id":"stream-id","model":"served"}}
        \\
        \\data: {"type":"response.output_text.delta","delta":"aS","logprobs":[{"token":"a","logprob":-0.2,"bytes":[97]},{"token":"","logprob":-0.4,"bytes":[]},{"token":"S","logprob":-0.3,"bytes":[83]}]}
        \\
        \\data: {"type":"response.reasoning_text.delta","output_index":1,"delta":"after stop boundary"}
        \\
        \\data: {"type":"response.output_text.delta","output_index":2,"delta":"TOP suffix"}
        \\
        \\data: {"type":"response.completed","response":{"usage":{"input_tokens":3}}}
        \\
        \\data: [DONE]
        \\
    ;
    var input: std.Io.Reader = .fixed(bytes);
    var stream: OpenAI.OpenAIStream = .{ .arena = arena, .model = "fallback", .stop = &.{"STOP"} };
    try OpenAI.OpenAIStream.consume(&stream, arena, &input, bytes.len);
    const result = stream.result.?;
    try testing.expectEqualStrings("a", result.text.?);
    try testing.expectEqual(@as(usize, 1), result.message.?.parts.len);
    try testing.expectEqualStrings("a", result.message.?.parts[0].text.text);
    try testing.expectEqualStrings("stream-id", result.metadata.id.?);
    try testing.expectEqualStrings("served", result.metadata.model.?);
    try testing.expectEqual(FinishReason.stop, result.metadata.finish_reason.?);
    try testing.expect(result.metadata.usage.input_tokens == .null);
    try testing.expect(result.provider_data == null);
    try testing.expectEqualStrings("a", result.logprobs.?[0].token);
    try testing.expectEqual(@as(usize, 1), result.logprobs.?.len);
    try testing.expect(result.logprobs_complete);
    try testing.expect(input.seek < bytes.len);
}

test "Responses audio recovers padded chunks but refuses empty final media" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const rows = [_]struct { chunks: []const []const u8, wanted: ?[]const u8 }{
        .{ .chunks = &.{ "AQ", "A Q==ignored", "A", "東京" }, .wanted = "AQE=" },
        .{ .chunks = &.{ "A", "東京" }, .wanted = null },
    };
    for (rows) |row| {
        var wire: Writer.Allocating = .init(arena);
        for (row.chunks) |chunk| {
            const frame = try std.json.Stringify.valueAlloc(arena, .{ .type = "response.output_audio.delta", .delta = chunk }, .{});
            try wire.writer.print("data: {s}\n\n", .{frame});
        }
        try wire.writer.writeAll("data: [DONE]\n\n");
        var input: std.Io.Reader = .fixed(wire.written());
        var stream: OpenAI.OpenAIStream = .{ .arena = arena, .model = "fixture", .stop = &.{"NEVER"} };
        if (row.wanted) |expected| {
            try OpenAI.OpenAIStream.consume(&stream, arena, &input, wire.written().len);
            const audio = stream.result.?.message.?.parts[0].audio;
            try testing.expectEqualStrings(expected, audio.data.?);
            try testing.expectEqualStrings("audio/wav", audio.media_type);
            try testing.expect(stream.result.?.generated_media);
        } else {
            try testing.expectError(error.LMFailure, OpenAI.OpenAIStream.consume(&stream, arena, &input, wire.written().len));
            try testing.expect(stream.result == null);
        }
    }
    const empty_image = "data: {\"type\":\"response.output_image.delta\",\"delta\":\"\"}\n\ndata: [DONE]\n\n";
    var input: std.Io.Reader = .fixed(empty_image);
    var stream: OpenAI.OpenAIStream = .{ .arena = arena, .model = "fixture", .stop = &.{"NEVER"} };
    try testing.expectError(error.LMFailure, OpenAI.OpenAIStream.consume(&stream, arena, &input, empty_image.len));
    try testing.expect(stream.result == null);
    try testing.expect(input.seek < empty_image.len);
}

test "Responses final copies compare exact ordered JSON and refuse missing stream end" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const rows = [_]struct { first: []const u8, final: []const u8, accepted: bool, expected_input: []const u8 = "{}" }{
        .{ .first = "{\"n\":1}", .final = "{ \"n\":1.0 }", .accepted = true, .expected_input = "{\"n\":1}" },
        .{ .first = "{\"a\":1,\"a\":2}", .final = "{ \"a\":1,\"a\":2 }", .accepted = true, .expected_input = "{\"a\":2}" },
        .{ .first = "{\"n\":NaN}", .final = "{ \"n\": NaN }", .accepted = true, .expected_input = "{\"partial_json\":\"{\\\"n\\\":NaN}\"}" },
        .{ .first = "{\"n\":1e999}", .final = "{\"n\":1e999}", .accepted = false },
        .{ .first = "{\"n\":0.1}", .final = "{\"n\":0.10000000000000001}", .accepted = false },
        .{ .first = "{\"a\":1,\"b\":2}", .final = "{\"b\":2,\"a\":1}", .accepted = false },
    };
    for (rows) |row| {
        const first = try std.json.Stringify.valueAlloc(arena, .{
            .type = "response.output_item.added",
            .item = .{ .type = "function_call", .name = "lookup", .arguments = row.first },
        }, .{});
        const final = try std.json.Stringify.valueAlloc(arena, .{
            .type = "response.output_item.done",
            .item = .{ .type = "function_call", .name = "lookup", .arguments = row.final },
        }, .{});
        const bytes = try std.fmt.allocPrint(arena, "data: {s}\n\ndata: {s}\n\ndata: [DONE]\n\n", .{ first, final });
        var input: std.Io.Reader = .fixed(bytes);
        var stream: OpenAI.OpenAIStream = .{ .arena = arena, .model = "fixture", .stop = &.{"NEVER"} };
        if (row.accepted) {
            try OpenAI.OpenAIStream.consume(&stream, arena, &input, bytes.len);
            try testing.expectEqualStrings("lookup", stream.result.?.tool_calls[0].name);
            try testing.expectEqual(FinishReason.tool_call, stream.result.?.metadata.finish_reason.?);
            const actual = try stream.result.?.message.?.parts[0].tool_call.input.toValue(arena);
            const expected = try OpenAI.parseJson(arena, row.expected_input);
            try testing.expect(try @import("value.zig").pyEqual(actual, expected));
        } else {
            try testing.expectError(error.LMFailure, OpenAI.OpenAIStream.consume(&stream, arena, &input, bytes.len));
            try testing.expect(stream.result == null);
        }
    }
    const truncated = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"unfinished\"}\n\n";
    var input: std.Io.Reader = .fixed(truncated);
    var stream: OpenAI.OpenAIStream = .{ .arena = arena, .model = "fixture", .stop = &.{"NEVER"} };
    try testing.expectError(error.LMFailure, OpenAI.OpenAIStream.consume(&stream, arena, &input, truncated.len));
    try testing.expect(stream.result == null);
}

test "canonical Chat keeps empty and unknown replies but refuses multiple messages" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const multiple =
        \\{"choices":[{"message":{"content":"first"}},{"message":{"content":"second"}}]}
    ;
    try testing.expectError(error.LMFailure, OpenAI.parseCanonicalChat(arena, multiple, .in_completion));
    const selected = try OpenAI.parseCompletions(arena, multiple);
    try testing.expectEqualStrings("first", selected[0].text.?);
    try testing.expectEqualStrings("second", selected[1].text.?);

    const empty = try OpenAI.parseCanonicalChat(arena,
        \\{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}
    , .in_completion);
    try testing.expectEqualStrings("", empty.text.?);
    try testing.expectEqual(FinishReason.stop, empty.metadata.finish_reason.?);
    try expectCount("3", empty.metadata.usage.input_tokens);

    const unknown = try OpenAI.parseCanonicalChat(arena,
        \\{"choices":[{"message":{"content":[null,{"type":"future"},{"type":"text","text":"visible"}],"tool_calls":[null,{"type":"future_tool"}]},"finish_reason":"future_finish"}]}
    , .in_completion);
    try testing.expectEqualStrings("visible", unknown.text.?);
    try testing.expectEqual(FinishReason.stop, unknown.metadata.finish_reason.?);
    const records = unknown.provider_data.?.root.object.get("_lm15_unmapped").?.array.items;
    const paths = [_][]const u8{
        "choices[0].message.content[0]",    "choices[0].message.content[1]",
        "choices[0].message.tool_calls[0]", "choices[0].message.tool_calls[1]",
        "choices[0].finish_reason",
    };
    const kinds = [_][]const u8{ "NoneType", "future", "NoneType", "future_tool", "future_finish" };
    try testing.expectEqual(paths.len, records.len);
    for (records, paths, kinds) |record, path, kind| {
        try testing.expectEqualStrings(path, record.object.get("path").?.string);
        try testing.expectEqualStrings(kind, record.object.get("type").?.string);
    }
}

test "recorded billing uses actual response usage and cannot exhaust generation storage" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const catalogue = try @import("model_prices.zig").Catalogue.init(arena,
        \\{"openai/test":{"litellm_provider":"openai","input_cost_per_token":0.00001,"output_cost_per_token":0.00003}}
    , .{});
    const completion = (try OpenAI.parseCompletions(arena,
        \\{"model":"test","usage":{"prompt_tokens":3,"completion_tokens":7},"choices":[{"message":{"content":"retained"},"finish_reason":"stop"}]}
    ))[0];
    var billing_bytes: [4096]u8 = undefined;
    var billing_storage: std.heap.FixedBufferAllocator = .init(&billing_bytes);
    var script: ScriptedLM = .{
        .model = "test",
        .responses = &.{ "retained", "retained after pricing exhaustion" },
        .metadata = &.{ completion.metadata, completion.metadata },
        .provider_data = &.{
            try completion.provider_data.?.toValue(arena),
            try completion.provider_data.?.toValue(arena),
        },
    };
    var model = script.lm();
    model.pricing = .{
        .catalogue = &catalogue,
        .arena = billing_storage.allocator(),
        .provider = "openai",
    };
    var recorder: RecordingLM = .{ .inner = model, .arena = arena };
    const request: Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "bill canonical SDK usage" } }},
    };
    try testing.expectEqualStrings("retained", try recorder.lm().complete(arena, request));
    const bill = recorder.calls.items[0].billing.?;
    // Actual pinned Source oracle: provider/'openai'/None, three inputs/seven outputs.
    try testing.expectEqual(@as(u64, 0x3f2f75104d551d69), @as(u64, @bitCast(bill.cost.float)));
    try testing.expectEqualStrings("estimate", bill.details.object.get("kind").?.string);
    _ = try billing_storage.allocator().alloc(u8, billing_bytes.len - billing_storage.end_index);
    try testing.expectEqualStrings("retained after pricing exhaustion", try recorder.lm().complete(arena, request));
    try testing.expectEqual(@as(usize, 2), recorder.calls.items.len);
    try testing.expect(recorder.calls.items[1].billing == null);
    // The earlier bill remains independently retained despite later advisory OOM.
    try testing.expectEqual(@as(u64, 0x3f2f75104d551d69), @as(u64, @bitCast(recorder.calls.items[0].billing.?.cost.float)));
}

test "canonical stream rejects trailing events without changing the owned partial response" {
    const allocator = testing.allocator;
    const request = try sdk.Request.fromFields(allocator, &.{
        .model = "fixture",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "ask" } }} }},
    });
    defer request.release();
    var fragment = [_]u8{ 'o', 'k' };
    const events = [_]OpenAI.StreamEvent{
        .{ .delta = .{ .part_index = "0", .data = .{ .text = .{ .text = &fragment } } } },
        .{ .end = .{ .finish_reason = .length, .usage = null, .provider_data = null } },
        .{ .provider_error = .{ .code = .server, .provider_code = "server", .message = "late" } },
    };
    const result = try materializeStream(allocator, request, &events);
    defer result.release();
    try testing.expect(result == .after_end);
    fragment[0] = 'x';
    try testing.expectEqualStrings("ok", result.after_end.fields.message.parts[0].text.text);
    try testing.expectEqual(FinishReason.length, result.after_end.fields.finish_reason);
}
