//! Language-model interface. Programs never name a model: they call through
//! `LM`, so a scripted mock, a keyword mock, or an OpenAI-compatible HTTP
//! endpoint can be swapped without touching program code.
//!
//! Upstream (DSPy 3.5.0): dspy/clients/base_lm.py, dspy/clients/lm.py, dspy/utils/exceptions.py

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const adapter = @import("adapter.zig");
const pyjson = @import("pyjson.zig");
const http = @import("http.zig");
pub const inference = @import("inference.zig");
/// Canonical SDK token usage; every unknown counter stays null, never zero.
pub const Usage = @import("lm_usage.zig").Usage;
const Message = adapter.Message;
pub const FunctionTool = inference.FunctionTool;
pub const ToolCall = inference.ToolCall;
pub const CitationPart = inference.CitationPart;
pub const Continuation = inference.Continuation;
pub const ResponseFormat = inference.ResponseFormat;
pub const ToolChoice = inference.ToolChoice;
/// TypeSafe System One decision backend (`POST /v1/systemone`).
pub const TypeSafe = @import("typesafe.zig").TypeSafe;

pub const Options = struct {
    temperature: ?f64 = null,
    /// DSPy cache identity; candidate groups use a tagged JSON-array tuple.
    rollout_id: ?@import("runtime_value.zig").Value = null,
    response_format: ?ResponseFormat = null,
    tool_choice: ?ToolChoice = null,
    parallel_tool_calls: ?bool = null,
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
    /// Present only for LMs advertising `supports_decision_requests`.
    decision: ?DecisionRequest = null,
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

pub const Error = error{ NoLM, LMFailure, ScriptExhausted } || @import("runtime_value.zig").CustomError;

/// Persistent configuration is separate from transport clients and credentials.
pub const StateError = error{ InvalidLMState, UnsupportedLMState, MissingLMLoader } || Allocator.Error;

/// One provider completion and its optional, genuinely returned reasoning.
pub const Completion = struct {
    /// Null means no TextPart was returned; it is distinct from an empty text.
    text: ?[]const u8,
    reasoning: ?[]const u8 = null,
    tool_calls: []const ToolCall = &.{},
    citations: []const CitationPart = &.{},
    continuation: []const Continuation = &.{},
    /// Actual provider JSON, retained separately from canonical continuation.
    provider_data: ?@import("runtime_value.zig").Value = null,
    /// Canonical response facts; raw provider numbers stay in `provider_data`.
    metadata: ResponseMetadata = .{},
};

/// The five lm15 finish reasons.
pub const FinishReason = enum { stop, length, tool_call, content_filter, @"error" };

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

/// Owning clones for long-lived histories and caches. Completion cloning also
/// unwinds partial allocations, so `freeCompletion` works with a general GPA.
pub fn cloneCompletion(allocator: Allocator, completion: Completion) Allocator.Error!Completion {
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
    const provider_data = if (completion.provider_data) |data| try inference.cloneJsonValue(allocator, data) else null;
    errdefer if (provider_data) |data| inference.freeJsonValue(allocator, data);
    return .{
        .text = text,
        .reasoning = reasoning,
        .tool_calls = tool_calls,
        .citations = citations,
        .continuation = continuation,
        .provider_data = provider_data,
        .metadata = try cloneResponseMetadata(allocator, completion.metadata),
    };
}

pub fn freeCompletion(allocator: Allocator, completion: Completion) void {
    freeResponseMetadata(allocator, completion.metadata);
    if (completion.provider_data) |data| inference.freeJsonValue(allocator, data);
    if (completion.text) |text| allocator.free(text);
    if (completion.reasoning) |reasoning| allocator.free(reasoning);
    inference.freeContinuations(allocator, completion.continuation);
    inference.freeToolCalls(allocator, completion.tool_calls);
    inference.freeCitations(allocator, completion.citations);
}

pub fn cloneOptions(arena: Allocator, options: Options) Allocator.Error!Options {
    var out = options;
    if (options.rollout_id) |id| out.rollout_id = try inference.cloneJsonValue(arena, id);
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
    return .{ .object = object };
}

pub fn messageValue(arena: Allocator, message: Message) Allocator.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "role", .{ .string = message.role.name() });
    try object.put(arena, "content", message.content.toValue(arena));
    try object.put(arena, "reasoning", if (message.reasoning) |reasoning| .{ .string = reasoning } else .null);
    try object.put(arena, "tool_calls", try inference.toolCallsValue(arena, message.tool_calls));
    try object.put(arena, "tool_call_id", if (message.tool_call_id) |id| .{ .string = id } else .null);
    try object.put(arena, "continuation", try inference.continuationsValue(arena, message.continuation));
    return .{ .object = object };
}

pub fn requestValue(arena: Allocator, request: Request) Allocator.Error!@import("runtime_value.zig").Value {
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

pub fn completionValue(arena: Allocator, completion: Completion) Allocator.Error!@import("runtime_value.zig").Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "text", if (completion.text) |text| .{ .string = text } else .null);
    try object.put(arena, "reasoning", if (completion.reasoning) |reasoning| .{ .string = reasoning } else .null);
    try object.put(arena, "tool_calls", try inference.toolCallsValue(arena, completion.tool_calls));
    try object.put(arena, "citations", try inference.citationsValue(arena, completion.citations));
    try object.put(arena, "continuation", try inference.continuationsValue(arena, completion.continuation));
    try object.put(arena, "provider_data", completion.provider_data orelse .null);
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
    if (input.object.get("citations")) |citations| out.citations = try inference.citationsFromValue(arena, citations);
    if (input.object.get("reasoning")) |reasoning| {
        if (reasoning != .null) out.reasoning = try arena.dupe(u8, try metadataString(reasoning));
    }
    if (input.object.get("tool_calls")) |calls| out.tool_calls = try inference.toolCallsFromValue(arena, calls);
    if (input.object.get("continuation")) |continuation| out.continuation = try inference.continuationsFromValue(arena, continuation);
    if (input.object.get("provider_data")) |data| {
        if (data != .null) {
            if (data != .object) return error.BadMetadata;
            out.provider_data = try inference.cloneJsonValue(arena, data);
        }
    }
    if (input.object.get("metadata")) |metadata| {
        if (metadata != .null) out.metadata = try responseMetadataFromValue(arena, metadata);
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

    pub const VTable = struct {
        /// Returns `request.options.n` completions, allocated in `arena`.
        complete: *const fn (ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion,
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
        restore: *const fn (ptr: *anyopaque, arena: Allocator, state: @import("runtime_value.zig").Value) StateError!LM,
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
        return copy(model.ptr, arena);
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
        return dump(lm.ptr, arena);
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

    /// All `request.options.n` completions; a provider that returns another
    /// count fails with `LMFailure`.
    pub fn completeResponses(lm: LM, arena: Allocator, request: Request) Error![]const Completion {
        assert(request.messages.len > 0 or request.decision != null);
        assert(request.options.n >= 1);
        if (request.decision != null) {
            // System One has no canonical multiple-response representation.
            if (!lm.supports_decision_requests or request.options.n != 1) return error.LMFailure;
        }
        if (request.options.n == 1) {
            const completions = try lm.vtable.complete(lm.ptr, arena, request);
            if (completions.len != 1) return error.LMFailure;
            return completions;
        }
        const out = try arena.alloc(Completion, request.options.n);
        for (out, 0..) |*completion, index| {
            var candidate = request;
            candidate.options.n = 1;
            // Borrowed only for this synchronous callback; histories/caches
            // clone request metadata before retaining it.
            var identity = [_]@import("runtime_value.zig").Value{
                .{ .string = "dspy.adapter.candidate" },
                request.options.rollout_id orelse .null,
                .{ .integer = @intCast(index) },
            };
            var rollout: @import("runtime_value.zig").Array = .init(arena);
            rollout.items = &identity;
            rollout.capacity = identity.len;
            candidate.options.rollout_id = .{ .array = rollout };
            const completions = try lm.vtable.complete(lm.ptr, arena, candidate);
            if (completions.len != 1) return error.LMFailure;
            completion.* = completions[0];
        }
        return out;
    }
};

/// Returns canned responses in order; fails once they run out.
pub const ScriptedLM = struct {
    responses: []const []const u8,
    /// Optional native summaries, parallel to `responses`.
    reasonings: []const ?[]const u8 = &.{},
    tool_calls: []const []const ToolCall = &.{},
    citations: []const []const CitationPart = &.{},
    continuation: []const []const Continuation = &.{},
    provider_data: []const ?@import("runtime_value.zig").Value = &.{},
    /// Optional canonical metadata, parallel to `responses` and independent
    /// of `provider_data`; nothing is derived from the raw provider JSON.
    metadata: []const ResponseMetadata = &.{},
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
            .vtable = &.{ .complete = complete },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_format,
            .supports_response_schema = self.supports_response_schema,
            .supports_function_calling = self.supports_function_calling,
            .supports_decision_requests = self.supports_decision_requests,
        };
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
                .provider_data = provider_data,
                .metadata = if (self.metadata.len != 0) try checkedMetadata(arena, self.metadata[i]) else .{},
            });
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
            .vtable = &.{ .complete = complete },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_format,
            .supports_response_schema = self.supports_response_schema,
            .supports_function_calling = self.supports_function_calling,
            .supports_decision_requests = self.supports_decision_requests,
        };
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
            for (recorded, 0..) |text, i| {
                if (self.used[first + i]) continue;
                out[written] = try cloneCompletion(arena, .{
                    .text = text,
                    .reasoning = if (call.reasonings.len != 0) call.reasonings[i] else null,
                    .tool_calls = if (call.tool_calls.len != 0) call.tool_calls[i] else &.{},
                    .citations = if (call.citations.len != 0) call.citations[i] else &.{},
                    .continuation = if (call.continuation.len != 0) call.continuation[i] else &.{},
                    .provider_data = if (call.provider_data.len != 0) call.provider_data[i] else null,
                    .metadata = if (call.metadata.len != 0) try checkedMetadata(arena, call.metadata[i]) else .{},
                });
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

    pub fn lm(self: *RecordingLM) LM {
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete },
            .supports_reasoning = self.inner.supports_reasoning,
            .supports_response_format = self.inner.supports_response_format,
            .supports_response_schema = self.inner.supports_response_schema,
            .supports_function_calling = self.inner.supports_function_calling,
            .supports_decision_requests = self.inner.supports_decision_requests,
        };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *RecordingLM = @ptrCast(@alignCast(ptr));
        const completions = try self.inner.completeResponses(arena, request);
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
            kept_data.* = if (completion.provider_data) |data| try inference.cloneJsonValue(self.arena, data) else null;
        }
        var has_metadata = false;
        for (completions) |completion| has_metadata = has_metadata or completion.metadata.known();
        const metadata: []ResponseMetadata = if (has_metadata) try self.arena.alloc(ResponseMetadata, completions.len) else &.{};
        for (metadata, completions[0..metadata.len]) |*kept_metadata, completion| {
            kept_metadata.* = try cloneResponseMetadata(self.arena, completion.metadata);
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
        });
        return completions;
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

    /// Startup loader with explicitly supplied trust, endpoint and credentials.
    /// State loading never reads secrets or imports executable code.
    pub const StateLoader = struct {
        client: *std.http.Client,
        base_url: []const u8,
        api_key: ?[]const u8 = null,
        header_provider: ?http.HeaderProvider = null,

        pub fn loader(self: *StateLoader) LM.StateLoader {
            return .{ .ptr = self, .restore = restore };
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
    };

    fn dumpState(ptr: *anyopaque, arena: Allocator) StateError!@import("runtime_value.zig").Value {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        var state: @import("runtime_value.zig").ObjectMap = .empty;
        try state.ensureTotalCapacity(arena, persistent_fields.len + 1);
        state.putAssumeCapacity("__lm_class__", .{ .string = "dszig.lm.OpenAI" });
        inline for (persistent_fields) |name| {
            const encoded = if (comptime @FieldType(OpenAI, name) == @import("runtime_value.zig").Value)
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

    /// A Responses request represents one completion; `complete` issues
    /// independent requests for `n > 1`, since Responses has no `n` parameter.
    pub fn requestBody(self: *const OpenAI, arena: Allocator, request: Request) Error![]const u8 {
        if (request.decision != null) return error.LMFailure;
        const has_system = self.api == .responses and request.messages.len != 0 and request.messages[0].role == .system;
        const messages = try self.wireMessages(arena, request.messages[@intFromBool(has_system)..]);
        var body: @import("runtime_value.zig").ObjectMap = .empty;
        try body.put(arena, "model", .{ .string = self.model });
        try body.put(arena, if (self.api == .responses) "input" else "messages", .{ .array = messages });
        if (self.api == .responses) {
            try body.put(arena, "stream", .{ .bool = false });
            if (has_system) {
                const system = request.messages[0];
                if (system.content != .text or system.tool_calls.len != 0 or system.tool_call_id != null) return error.LMFailure;
                if (system.content.text.len != 0) try body.put(arena, "instructions", .{ .string = system.content.text });
            }
        }
        if (request.options.temperature) |t| try body.put(arena, "temperature", .{ .float = t });
        if (self.max_tokens) |n| {
            try body.put(arena, if (self.api == .responses) "max_output_tokens" else "max_tokens", .{ .integer = n });
        }
        if (self.api == .chat_completions and request.options.n > 1) {
            try body.put(arena, "n", .{ .integer = request.options.n });
        }
        if (request.options.response_format) |format| {
            const encoded = try self.wireResponseFormat(arena, format);
            if (self.api == .responses) {
                var text: @import("runtime_value.zig").ObjectMap = .empty;
                try text.put(arena, "format", encoded);
                try body.put(arena, "text", .{ .object = text });
            } else {
                try body.put(arena, "response_format", encoded);
            }
        }
        if (request.tools.len != 0) {
            var tools: @import("runtime_value.zig").Array = .init(arena);
            try tools.ensureTotalCapacity(request.tools.len);
            for (request.tools) |tool| {
                if (tool.name.len == 0 or tool.parameters != .object) return error.LMFailure;
                var function: @import("runtime_value.zig").ObjectMap = .empty;
                try function.put(arena, "name", .{ .string = tool.name });
                if (tool.description.len != 0) try function.put(arena, "description", .{ .string = tool.description });
                try function.put(arena, "parameters", tool.parameters);
                if (tool.strict) |strict| {
                    try function.put(arena, "strict", .{ .bool = strict });
                } else if (self.api == .responses) {
                    // Responses otherwise defaults to strict tools; DSPy's
                    // FunctionTool parameters do not request that coercion.
                    try function.put(arena, "strict", .{ .bool = false });
                }
                if (self.api == .responses) {
                    try function.put(arena, "type", .{ .string = "function" });
                    tools.appendAssumeCapacity(.{ .object = function });
                } else {
                    var wrapper: @import("runtime_value.zig").ObjectMap = .empty;
                    try wrapper.put(arena, "type", .{ .string = "function" });
                    try wrapper.put(arena, "function", .{ .object = function });
                    tools.appendAssumeCapacity(.{ .object = wrapper });
                }
            }
            try body.put(arena, "tools", .{ .array = tools });
        }
        if (request.options.tool_choice) |choice| try body.put(arena, "tool_choice", try self.wireToolChoice(arena, choice));
        if (request.options.parallel_tool_calls) |parallel| try body.put(arena, "parallel_tool_calls", .{ .bool = parallel });
        if (request.options.reasoning_effort) |effort| {
            const disabled = std.mem.eql(u8, effort, "none") or std.mem.eql(u8, effort, "off");
            const wire_effort = if (disabled) "none" else effort;
            if (self.api == .responses) {
                var reasoning: @import("runtime_value.zig").ObjectMap = .empty;
                try reasoning.put(arena, "effort", .{ .string = wire_effort });
                if (!disabled) try reasoning.put(arena, "summary", .{ .string = "auto" });
                try body.put(arena, "reasoning", .{ .object = reasoning });
            } else {
                try body.put(arena, "reasoning_effort", .{ .string = wire_effort });
            }
        }
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeJson(&aw.writer, .{ .object = body }, .minified) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep, error.ParseValue => error.LMFailure,
        };
        return aw.written();
    }

    fn wireMessages(self: *const OpenAI, arena: Allocator, input: []const Message) Error!@import("runtime_value.zig").Array {
        var out: @import("runtime_value.zig").Array = .init(arena);
        for (input) |message| {
            if (message.continuation.len != 0) {
                if (self.api != .responses or message.role != .assistant) return error.LMFailure;
                // Flattened summary text cannot be assigned to multiple
                // distinct ThinkingParts without inventing their boundaries.
                if (message.continuation.len > 1 and
                    message.reasoning != null and message.reasoning.?.len != 0) return error.LMFailure;
                for (message.continuation) |state| try out.append(try wireReasoningItem(arena, state, message.reasoning));
            }
            if (message.role != .assistant and message.tool_calls.len != 0) return error.LMFailure;
            if (message.role != .tool and message.tool_call_id != null) return error.LMFailure;
            var object: @import("runtime_value.zig").ObjectMap = .empty;
            if (self.api == .responses and message.role == .tool) {
                const id = message.tool_call_id orelse return error.LMFailure;
                if (id.len == 0) return error.LMFailure;
                try object.put(arena, "type", .{ .string = "function_call_output" });
                try object.put(arena, "call_id", .{ .string = id });
                try object.put(arena, "output", if (message.content == .text)
                    .{ .string = message.content.text }
                else
                    .{ .array = try self.responseContent(arena, .user, message.content) });
                try out.append(.{ .object = object });
                continue;
            }
            if (self.api == .chat_completions or !message.content.isEmpty() or message.tool_calls.len == 0 or
                (message.reasoning != null and message.continuation.len == 0))
            {
                try object.put(arena, "role", .{ .string = message.role.name() });
                if (self.api == .responses) {
                    var parts = try self.responseContent(arena, message.role, message.content);
                    if (message.role == .assistant and message.reasoning != null and
                        message.reasoning.?.len != 0 and message.continuation.len == 0)
                    {
                        var thinking: @import("runtime_value.zig").ObjectMap = .empty;
                        try thinking.put(arena, "type", .{ .string = "output_text" });
                        try thinking.put(arena, "text", .{ .string = message.reasoning.? });
                        try parts.insert(0, .{ .object = thinking });
                    }
                    if (parts.items.len == 0) continue;
                    try object.put(arena, "content", .{ .array = parts });
                } else if (self.api == .chat_completions and message.role == .assistant and
                    message.reasoning != null and message.reasoning.?.len != 0)
                {
                    // lm15's default Chat policy replays visible thinking as
                    // assistant text rather than dropping it or sending a
                    // provider-specific reasoning_content request field.
                    const reasoning = message.reasoning.?;
                    if (message.content != .text) return error.LMFailure;
                    const content = if (message.content.text.len == 0)
                        reasoning
                    else
                        try std.fmt.allocPrint(arena, "{s}\n{s}", .{ reasoning, message.content.text });
                    try object.put(arena, "content", .{ .string = content });
                } else {
                    // lm15 Chat currently rejects DocumentPart before HTTP,
                    // although the SDK-compatible display serializer accepts it.
                    if (message.content != .text) return error.LMFailure;
                    try object.put(arena, "content", .{ .string = message.content.text });
                }
                if (message.role == .tool) {
                    const id = message.tool_call_id orelse return error.LMFailure;
                    if (id.len == 0) return error.LMFailure;
                    try object.put(arena, "tool_call_id", .{ .string = id });
                }
                if (self.api == .chat_completions and message.tool_calls.len != 0) {
                    var calls: @import("runtime_value.zig").Array = .init(arena);
                    try calls.ensureTotalCapacity(message.tool_calls.len);
                    for (message.tool_calls) |call| calls.appendAssumeCapacity(try toolCallValue(arena, self.api, call));
                    try object.put(arena, "tool_calls", .{ .array = calls });
                }
                try out.append(.{ .object = object });
            }
            if (self.api == .responses) {
                for (message.tool_calls) |call| try out.append(try toolCallValue(arena, self.api, call));
            }
        }
        return out;
    }

    fn responseContent(self: *const OpenAI, arena: Allocator, role: adapter.Role, content: adapter.Content) Error!@import("runtime_value.zig").Array {
        var out: @import("runtime_value.zig").Array = .init(arena);
        if (content == .text) {
            var text: @import("runtime_value.zig").ObjectMap = .empty;
            try text.put(arena, "type", .{ .string = if (role == .assistant) "output_text" else "input_text" });
            try text.put(arena, "text", .{ .string = content.text });
            try out.append(.{ .object = text });
            return out;
        }
        for (content.parts) |part| {
            if (part != .object) return error.LMFailure;
            const kind = part.object.get("type") orelse return error.LMFailure;
            if (kind != .string) return error.LMFailure;
            var block: @import("runtime_value.zig").ObjectMap = .empty;
            if (std.mem.eql(u8, kind.string, "text")) {
                const text = part.object.get("text") orelse @import("runtime_value.zig").Value{ .string = "" };
                if (text != .string) return error.LMFailure;
                try block.put(arena, "type", .{ .string = if (role == .assistant) "output_text" else "input_text" });
                try block.put(arena, "text", text);
            } else if (std.mem.eql(u8, kind.string, "document")) {
                // Public source build_request preflight rejects assistant
                // media, even though its private serializer would drop it.
                if (role == .assistant) return error.LMFailure;
                try block.put(arena, "type", .{ .string = "input_file" });
                if (part.object.get("url")) |url| {
                    if (url != .string) return error.LMFailure;
                    try block.put(arena, "file_url", url);
                } else if (part.object.get("file_id")) |file_id| {
                    if (file_id != .string) return error.LMFailure;
                    try block.put(arena, "file_id", file_id);
                } else {
                    const media_type = part.object.get("media_type") orelse return error.LMFailure;
                    if (media_type != .string or media_type.string.len == 0) return error.LMFailure;
                    const data_uri = if (part.object.get("data")) |data| blk: {
                        if (data != .string) return error.LMFailure;
                        break :blk try std.fmt.allocPrint(arena, "data:{s};base64,{s}", .{ media_type.string, data.string });
                    } else if (part.object.get("path")) |path| blk: {
                        if (path != .string) return error.LMFailure;
                        const file = @import("file.zig").File.fromPath(arena, self.client.io, path.string, .{ .mime_type = media_type.string }) catch |err| return switch (err) {
                            error.OutOfMemory => error.OutOfMemory,
                            else => error.LMFailure,
                        };
                        break :blk file.file_data orelse return error.LMFailure;
                    } else return error.LMFailure;
                    const subtype = if (std.mem.indexOfScalar(u8, media_type.string, '/')) |slash|
                        media_type.string[slash + 1 ..]
                    else
                        media_type.string;
                    const extension = subtype[0 .. std.mem.indexOfScalar(u8, subtype, '+') orelse subtype.len];
                    try block.put(arena, "filename", .{ .string = try std.fmt.allocPrint(arena, "file.{s}", .{if (extension.len == 0) "bin" else extension}) });
                    try block.put(arena, "file_data", .{ .string = data_uri });
                }
            } else return error.LMFailure;
            try out.append(.{ .object = block });
        }
        return out;
    }

    fn wireReasoningItem(arena: Allocator, state: Continuation, summary: ?[]const u8) Error!@import("runtime_value.zig").Value {
        if (!std.mem.eql(u8, state.provider, "openai") or
            !std.mem.eql(u8, state.kind, "reasoning_item") or state.data != .object) return error.LMFailure;
        if (state.data.object.count() == 0) return error.LMFailure;
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.put(arena, "type", .{ .string = "reasoning" });
        var fields = state.data.object.iterator();
        while (fields.next()) |field| {
            if (!std.mem.eql(u8, field.key_ptr.*, "id") and
                !std.mem.eql(u8, field.key_ptr.*, "encrypted_content")) return error.LMFailure;
            const text = field.value_ptr.*;
            if (text != .string or text.string.len == 0) return error.LMFailure;
            try object.put(arena, field.key_ptr.*, text);
        }
        var parts: @import("runtime_value.zig").Array = .init(arena);
        if (summary) |text| {
            if (text.len != 0) {
                var part: @import("runtime_value.zig").ObjectMap = .empty;
                try part.put(arena, "type", .{ .string = "summary_text" });
                try part.put(arena, "text", .{ .string = text });
                try parts.append(.{ .object = part });
            }
        }
        try object.put(arena, "summary", .{ .array = parts });
        return .{ .object = object };
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

    fn wireResponseFormat(self: *const OpenAI, arena: Allocator, format: ResponseFormat) Error!@import("runtime_value.zig").Value {
        if (format != .json_schema) {
            var object: @import("runtime_value.zig").ObjectMap = .empty;
            try object.put(arena, "type", .{ .string = @tagName(format) });
            return .{ .object = object };
        }
        const schema = format.json_schema;
        if (schema.name.len == 0 or schema.schema != .object) return error.LMFailure;
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.put(arena, "name", .{ .string = schema.name });
        try object.put(arena, "schema", schema.schema);
        if (schema.description) |description| try object.put(arena, "description", .{ .string = description });
        try object.put(arena, "strict", .{ .bool = schema.strict });
        if (self.api == .responses) {
            try object.put(arena, "type", .{ .string = "json_schema" });
            return .{ .object = object };
        }
        var wrapper: @import("runtime_value.zig").ObjectMap = .empty;
        try wrapper.put(arena, "type", .{ .string = "json_schema" });
        try wrapper.put(arena, "json_schema", .{ .object = object });
        return .{ .object = wrapper };
    }

    fn wireNamedTool(self: *const OpenAI, arena: Allocator, name: []const u8) Error!@import("runtime_value.zig").Value {
        if (name.len == 0) return error.LMFailure;
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.put(arena, "type", .{ .string = "function" });
        if (self.api == .responses) {
            try object.put(arena, "name", .{ .string = name });
        } else {
            var function: @import("runtime_value.zig").ObjectMap = .empty;
            try function.put(arena, "name", .{ .string = name });
            try object.put(arena, "function", .{ .object = function });
        }
        return .{ .object = object };
    }

    fn wireToolChoice(self: *const OpenAI, arena: Allocator, choice: ToolChoice) Error!@import("runtime_value.zig").Value {
        return switch (choice) {
            .auto => .{ .string = "auto" },
            .none => .{ .string = "none" },
            .required => .{ .string = "required" },
            .function => |name| try self.wireNamedTool(arena, name),
            .allowed_tools => |allowed| blk: {
                if (allowed.names.len == 0) break :blk .{ .string = @tagName(allowed.mode) };
                var tools: @import("runtime_value.zig").Array = .init(arena);
                try tools.ensureTotalCapacity(allowed.names.len);
                for (allowed.names) |name| tools.appendAssumeCapacity(try self.wireNamedTool(arena, name));
                var selection: @import("runtime_value.zig").ObjectMap = .empty;
                try selection.put(arena, "mode", .{ .string = @tagName(allowed.mode) });
                try selection.put(arena, "tools", .{ .array = tools });
                if (self.api == .responses) {
                    try selection.put(arena, "type", .{ .string = "allowed_tools" });
                    break :blk .{ .object = selection };
                }
                var wrapper: @import("runtime_value.zig").ObjectMap = .empty;
                try wrapper.put(arena, "type", .{ .string = "allowed_tools" });
                try wrapper.put(arena, "allowed_tools", .{ .object = selection });
                break :blk .{ .object = wrapper };
            },
        };
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
        // Chat dialects have no System One judgment wire; never drop it silently.
        if (request.decision != null) return error.LMFailure;
        const body = try self.requestBody(arena, request);
        const path = if (self.api == .responses) "responses" else "chat/completions";
        const url = try std.fmt.allocPrint(arena, "{s}/{s}", .{ std.mem.trimEnd(u8, self.base_url, "/"), path });
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        if (self.api == .chat_completions) {
            const response = self.fetchWithRetries(arena, url, auth, body, &.{}) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.LMFailure,
            };
            defer self.client.allocator.free(response.body);
            const completions = try parseCompletions(arena, response.body);
            // The parser allocated these completions in this call's arena;
            // its public const view does not make that owned storage immutable.
            for (@constCast(completions)) |*completion| {
                if (completion.metadata.model == null) completion.metadata.model = try arena.dupe(u8, self.model);
            }
            return completions;
        }
        const out = try arena.alloc(Completion, request.options.n);
        for (out) |*completion| {
            const response = self.fetchWithRetries(arena, url, auth, body, &.{}) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.LMFailure,
            };
            defer self.client.allocator.free(response.body);
            completion.* = try parseResponsesResponse(arena, response.body);
            if (completion.metadata.model == null) completion.metadata.model = try arena.dupe(u8, self.model);
        }
        return out;
    }

    fn fetchWithRetries(self: *OpenAI, arena: Allocator, url: []const u8, auth: ?[]const u8, body: []const u8, headers: []const std.http.Header) anyerror!http.Response {
        const transport: http.Transport = .{
            .client = self.client,
            .max_response_bytes = self.max_response_bytes,
            .max_retries = self.max_retries,
            .retry_base_ms = self.retry_base_ms,
            .timeout_ns = self.timeout_ns,
            .connect_timeout_ns = self.connect_timeout_ns,
            .header_provider = self.header_provider,
            .success = .ok,
            .failure_error = error.LMFailure,
            .response_limit_error = error.LMFailure,
        };
        return transport.send(arena, .{
            .url = url,
            .method = .POST,
            .body = body,
            .content_type = "application/json",
            .authorization = auth,
            .headers = headers,
        });
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
        if (parsed != .object) return error.LMFailure;
        if (!finiteJson(parsed)) return error.LMFailure;
        const choices = parsed.object.get("choices") orelse return error.LMFailure;
        if (choices != .array or choices.array.items.len == 0) return error.LMFailure;
        const out = try arena.alloc(Completion, choices.array.items.len);
        const id = try responseIdentity(arena, parsed.object.get("id"));
        const model = try responseIdentity(arena, parsed.object.get("model"));
        const usage = try chatUsage(arena, parsed);
        for (choices.array.items, out) |choice, *completion| {
            if (choice != .object) return error.LMFailure;
            const message = choice.object.get("message") orelse return error.LMFailure;
            if (message != .object) return error.LMFailure;
            completion.* = .{ .text = null, .provider_data = parsed };
            var part_index: usize = 0;
            const reasoning = try annotationAlternative(message, &.{ "reasoning_content", "reasoning" });
            if (try responseTruthy(reasoning)) {
                completion.reasoning = try responseStringValue(arena, reasoning);
                part_index += 1;
            }
            var text: TextParts = .{ .joined = .init(arena) };
            if (message.object.get("content")) |content| {
                if (content == .string and content.string.len != 0) {
                    try text.add(content.string, "");
                    part_index += 1;
                } else if (content == .array) {
                    for (content.array.items) |part| {
                        if (part != .object) continue;
                        if (!stringEquals(part.object.get("type") orelse .null, "text")) continue;
                        try text.add(try responseText(arena, part.object.get("text")), "");
                        part_index += 1;
                    }
                }
            }
            if (message.object.get("refusal")) |refusal| {
                if (try responseTruthy(refusal)) part_index += 1;
            }
            if (message.object.get("tool_calls")) |calls| {
                if (calls == .array) {
                    var native: std.ArrayList(ToolCall) = .empty;
                    for (calls.array.items) |call| {
                        if (call != .object) continue;
                        const kind = call.object.get("type") orelse .null;
                        if (try responseTruthy(kind) and !stringEquals(kind, "function")) continue;
                        const function = call.object.get("function") orelse return error.LMFailure;
                        if (function != .object) return error.LMFailure;
                        const name = try responseText(arena, function.object.get("name"));
                        if (name.len == 0) return error.LMFailure;
                        const raw_id = call.object.get("id");
                        const call_id = try responseText(arena, raw_id);
                        try native.append(arena, .{
                            .id = if (call_id.len != 0) call_id else try std.fmt.allocPrint(arena, "call_{d}", .{part_index}),
                            .name = name,
                            .arguments = try responseArguments(arena, function.object.get("arguments"), call_id.len != 0 and raw_id != null and raw_id.? == .string),
                        });
                        part_index += 1;
                    }
                    completion.tool_calls = native.items;
                } else if (calls != .object and calls != .string and try responseTruthy(calls)) return error.LMFailure;
            }
            if (part_index == 0) try text.add("", "");
            completion.text = text.value();
            completion.metadata = .{
                .id = id,
                .model = model,
                .finish_reason = chatFinishReason(choice.object.get("finish_reason") orelse .null, completion.tool_calls.len != 0),
                // SDK choice=i ingestion retains the response-wide usage for
                // each selected choice. The LM boundary rejects an unexpected
                // multi-choice body rather than billing these as separate calls.
                .usage = usage,
            };
        }
        return out;
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
        var part_count: usize = 0;
        var text: TextParts = .{ .joined = .init(arena) };
        var reasoning: TextParts = .{ .joined = .init(arena) };
        var calls: std.ArrayList(ToolCall) = .empty;
        var citations: std.ArrayList(CitationPart) = .empty;
        var continuation: std.ArrayList(Continuation) = .empty;
        for (items) |item| {
            if (item != .object) continue;
            const kind = item.object.get("type") orelse continue;
            if (stringEquals(kind, "reasoning")) {
                const summary = try reasoningText(arena, item);
                if (summary) |visible| try reasoning.add(visible, "");
                var state: @import("runtime_value.zig").ObjectMap = .empty;
                for ([_][]const u8{ "id", "encrypted_content" }) |name| {
                    if (item.object.get(name)) |field| {
                        if (field == .null) continue;
                        if (field != .string) return error.LMFailure;
                        if (field.string.len != 0) try state.put(arena, name, field);
                    }
                }
                if (state.count() != 0) try continuation.append(arena, .{
                    .provider = "openai",
                    .kind = "reasoning_item",
                    .data = .{ .object = state },
                });
                if (state.count() != 0 or summary != null) part_count += 1;
            } else if (stringEquals(kind, "function_call")) {
                const call_id = try responseText(arena, item.object.get("call_id"));
                const item_id = if (call_id.len == 0) try responseText(arena, item.object.get("id")) else "";
                try calls.append(arena, .{
                    .id = if (call_id.len != 0) call_id else if (item_id.len != 0) item_id else try std.fmt.allocPrint(arena, "call_{d}", .{part_count}),
                    .name = try responseString(item, "name", true),
                    .arguments = try responseArguments(arena, item.object.get("arguments"), blk: {
                        const raw_id = if (call_id.len != 0) item.object.get("call_id") else item.object.get("id");
                        break :blk raw_id != null and raw_id.? == .string and (call_id.len != 0 or item_id.len != 0);
                    }),
                });
                part_count += 1;
            } else if (stringEquals(kind, "message")) {
                const content = item.object.get("content") orelse .null;
                if (content == .null) continue;
                if (content != .array) return error.LMFailure;
                for (content.array.items) |part| {
                    if (part != .object) continue;
                    const part_type = part.object.get("type") orelse continue;
                    if (stringEquals(part_type, "refusal")) {
                        const refusal = try responseText(arena, part.object.get("refusal"));
                        const refused = if (refusal.len != 0) refusal else try responseText(arena, part.object.get("text"));
                        if (refused.len == 0) try text.add("", "");
                        part_count += 1;
                    } else if (stringEquals(part_type, "output_text") or stringEquals(part_type, "text")) {
                        const visible = try responseText(arena, part.object.get("text"));
                        try text.add(visible, "");
                        part_count += 1;
                        if (part.object.get("annotations")) |annotations| {
                            if (annotations == .array) {
                                for (annotations.array.items) |annotation| {
                                    if (annotation != .object) continue;
                                    if (try responseCitation(arena, annotation, visible)) |citation| {
                                        try citations.append(arena, citation);
                                        part_count += 1;
                                    }
                                }
                            } else if (annotations != .object and annotations != .string and try responseTruthy(annotations)) {
                                return error.LMFailure;
                            }
                        }
                    }
                }
            }
        }
        if (part_count == 0) try text.add(try responseText(arena, parsed.object.get("output_text")), "");
        const answer = text.value();
        return .{
            .text = answer,
            .reasoning = reasoning.value(),
            .tool_calls = calls.items,
            .citations = citations.items,
            .continuation = continuation.items,
            .provider_data = parsed,
            .metadata = .{
                .id = try responseIdentity(arena, parsed.object.get("id")),
                .model = try responseIdentity(arena, parsed.object.get("model")),
                .finish_reason = try responsesFinishReason(arena, parsed, calls.items.len != 0),
                .usage = try responsesUsage(arena, parsed),
            },
        };
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

    /// MAP-19 for Chat Completions under the default `in_completion` dialect:
    /// prompt and completion counts already include their detail counters.
    /// The xAI `separate` reasoning dialect has no native configuration.
    fn chatUsage(arena: Allocator, data: @import("runtime_value.zig").Value) Error!Usage {
        assert(data == .object);
        const raw = data.object.get("usage") orelse .null;
        if (!(try responseTruthy(raw))) return .{};
        if (raw != .object) return error.LMFailure;
        const prompt_details = try usageDetails(raw, &.{"prompt_tokens_details"});
        const completion_details = try usageDetails(raw, &.{"completion_tokens_details"});
        // Some servers report the cached count flat on usage instead.
        var cached = detail(prompt_details, "cached_tokens");
        if (cached == .null) cached = raw.object.get("cached_tokens") orelse .null;
        return settleUsage(arena, .{
            .input_tokens = try wireCount(arena, raw.object.get("prompt_tokens") orelse .null),
            .output_tokens = try wireCount(arena, raw.object.get("completion_tokens") orelse .null),
            .total_tokens = try wireCount(arena, raw.object.get("total_tokens") orelse .null),
            .cache_read_tokens = try wireCount(arena, cached),
            .cache_write_tokens = try wireCount(arena, detail(prompt_details, "cache_write_tokens")),
            .reasoning_tokens = try wireCount(arena, detail(completion_details, "reasoning_tokens")),
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
    fn wireCount(arena: Allocator, raw: @import("runtime_value.zig").Value) Error!@import("runtime_value.zig").Value {
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
    fn settleUsage(arena: Allocator, counts: Usage) Error!Usage {
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
    fn countSum(arena: Allocator, a: @import("runtime_value.zig").Value, b: @import("runtime_value.zig").Value) Error!@import("runtime_value.zig").Value {
        return @import("value.zig").numericSum(arena, a, b) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
    }

    fn responseTruthy(value: @import("runtime_value.zig").Value) error{ParseValue}!bool {
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
    fn annotationIndex(value: @import("runtime_value.zig").Value, bound: usize) Error!?usize {
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

    fn codepointOffset(text: []const u8, index: usize) ?usize {
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
    fn responseText(arena: Allocator, input: ?@import("runtime_value.zig").Value) Error![]const u8 {
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

    fn responseArguments(arena: Allocator, input: ?@import("runtime_value.zig").Value, correlated_raw_id: bool) Error![]const u8 {
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

test "OpenAI training configuration survives destruction of the source arena" {
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
    const restored_lm = try interface.restore(interface.ptr, target, state);
    var without_model = try inference.cloneJsonValue(source, state);
    try testing.expect(without_model.object.orderedRemove("finetuning_model"));
    try testing.expect(without_model.object.orderedRemove("train_kwargs"));
    const sparse_lm = try interface.restore(interface.ptr, target, without_model);
    _ = source_memory.reset(.free_all);

    const restored: *OpenAI = @ptrCast(@alignCast(restored_lm.ptr));
    try testing.expectEqualStrings("tunable-model", restored.finetuning_model.?);
    try testing.expectEqualStrings("injected", restored.api_key.?);
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
            _ = try interface.restore(interface.ptr, arena, state);
            try state.object.put(arena, name, value);
            try testing.expectError(error.InvalidLMState, interface.restore(interface.ptr, arena, state));
        }
    }
    var valid: OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "m" };
    var state = try valid.lm().dumpState(arena);
    try state.object.put(arena, "finetuning_model", .{ .integer = 1 });
    try testing.expectError(error.InvalidLMState, interface.restore(interface.ptr, arena, state));
}

test "OpenAI model copy shares resources and sends the substituted model" {
    // The copy must reach the real request encoder with the new model while
    // the original keeps its own, and neither owns separate resources.
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var original: OpenAI = .{
        .client = &client,
        .base_url = "https://example.invalid/v1",
        .model = "base-model",
        .api = .responses,
        .api_key = "credential",
        .max_tokens = 41,
        .finetuning_model = "tunable-model",
        .train_kwargs = try testNestedConfig(arena, "train"),
    };
    const trained_name: []const u8 = "ft:base-model:org::job";
    var trained: OpenAI = undefined;
    original.copyWithModel(&trained, trained_name);

    try testing.expect(trained.model.ptr == trained_name.ptr);
    try testing.expectEqual(original.client, trained.client);
    try testing.expect(trained.base_url.ptr == original.base_url.ptr);
    try testing.expect(trained.api_key.?.ptr == original.api_key.?.ptr);
    try testing.expect(trained.finetuning_model.?.ptr == original.finetuning_model.?.ptr);
    try testing.expectEqual(original.api, trained.api);
    try testing.expectEqual(original.train_kwargs.object.keys().ptr, trained.train_kwargs.object.keys().ptr);

    const request: Request = .{ .messages = &.{.{ .role = .user, .content = .{ .text = "q" } }} };
    const trained_body = try OpenAI.parseJson(arena, try trained.requestBody(arena, request));
    try testing.expectEqualStrings(trained_name, trained_body.object.get("model").?.string);
    try testing.expectEqual(@as(i64, 41), trained_body.object.get("max_output_tokens").?.integer);
    try testing.expect(trained_body.object.get("train_kwargs") == null);
    const original_body = try OpenAI.parseJson(arena, try original.requestBody(arena, request));
    try testing.expectEqualStrings("base-model", original_body.object.get("model").?.string);
    try testing.expectEqualStrings("base-model", original.model);
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

/// Counters are opaque canonical integers, so compare by exact numeric order.
fn expectCount(expected: []const u8, actual: @import("runtime_value.zig").Value) !void {
    try testing.expect(actual != .null);
    const order = try @import("value.zig").numericOrder(.{ .number_string = expected }, actual);
    try testing.expectEqual(std.math.Order.eq, order.?);
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
    try testing.expect(try inference.jsonEqual(raw, completions[0].provider_data.?));

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
    try testing.expect(try inference.jsonEqual(try OpenAI.parseJson(arena, body), truncated.provider_data.?));

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
        try testing.expectEqual(@as(i64, 999), completion.provider_data.?.object.get("usage").?.object.get("prompt_tokens").?.integer);
        try testing.expectEqual(@as(usize, 1), completion.provider_data.?.object.count());
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
}

test "completion clone with metadata unwinds every allocation failure" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    var raw: @import("runtime_value.zig").ObjectMap = .empty;
    try raw.put(arena, "kept", .{ .string = "raw" });
    const completion: Completion = .{
        .text = "answer",
        .provider_data = .{ .object = raw },
        .metadata = try checkedMetadata(arena, .{
            .id = "resp-1",
            .model = "served-model",
            .finish_reason = .stop,
            .usage = .{ .total_tokens = .{ .number_string = "18446744073709551617" } },
        }),
    };
    try testing.checkAllAllocationFailures(testing.allocator, cloneThenFreeCompletion, .{completion});
}
