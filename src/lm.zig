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
const Message = adapter.Message;
pub const FunctionTool = inference.FunctionTool;
pub const ToolCall = inference.ToolCall;
pub const Continuation = inference.Continuation;
pub const ResponseFormat = inference.ResponseFormat;
pub const ToolChoice = inference.ToolChoice;

pub const Options = struct {
    temperature: ?f64 = null,
    /// DSPy cache identity; candidate groups use a tagged JSON-array tuple.
    rollout_id: ?std.json.Value = null,
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

pub const Request = struct {
    messages: []const Message,
    options: Options = .{},
    tools: []const FunctionTool = &.{},
};

pub const Error = error{ LMFailure, ScriptExhausted } || Allocator.Error;

/// One provider completion and its optional, genuinely returned reasoning.
pub const Completion = struct {
    text: []const u8,
    reasoning: ?[]const u8 = null,
    tool_calls: []const ToolCall = &.{},
    continuation: []const Continuation = &.{},
    /// Actual provider JSON, retained separately from canonical continuation.
    provider_data: ?std.json.Value = null,
};

/// Owning clones for long-lived histories and caches. Completion cloning also
/// unwinds partial allocations, so `freeCompletion` works with a general GPA.
pub fn cloneCompletion(allocator: Allocator, completion: Completion) Allocator.Error!Completion {
    const text = try allocator.dupe(u8, completion.text);
    errdefer allocator.free(text);
    const reasoning = if (completion.reasoning) |summary| try allocator.dupe(u8, summary) else null;
    errdefer if (reasoning) |summary| allocator.free(summary);
    const tool_calls = try inference.cloneToolCalls(allocator, completion.tool_calls);
    errdefer inference.freeToolCalls(allocator, tool_calls);
    const continuation = try inference.cloneContinuations(allocator, completion.continuation);
    errdefer inference.freeContinuations(allocator, continuation);
    const provider_data = if (completion.provider_data) |data| try inference.cloneJsonValue(allocator, data) else null;
    return .{
        .text = text,
        .reasoning = reasoning,
        .tool_calls = tool_calls,
        .continuation = continuation,
        .provider_data = provider_data,
    };
}

pub fn freeCompletion(allocator: Allocator, completion: Completion) void {
    if (completion.provider_data) |data| inference.freeJsonValue(allocator, data);
    allocator.free(completion.text);
    if (completion.reasoning) |reasoning| allocator.free(reasoning);
    inference.freeContinuations(allocator, completion.continuation);
    inference.freeToolCalls(allocator, completion.tool_calls);
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
        .content = try arena.dupe(u8, message.content),
        .reasoning = if (message.reasoning) |reasoning| try arena.dupe(u8, reasoning) else null,
        .tool_calls = try inference.cloneToolCalls(arena, message.tool_calls),
        .tool_call_id = if (message.tool_call_id) |id| try arena.dupe(u8, id) else null,
        .continuation = try inference.cloneContinuations(arena, message.continuation),
    };
}

/// Canonical metadata, before a provider dialect adds its wire wrappers.
/// All knobs remain visible, including explicitly absent values.
pub fn optionsValue(arena: Allocator, options: Options) Allocator.Error!std.json.Value {
    var object: std.json.ObjectMap = .empty;
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

pub fn messageValue(arena: Allocator, message: Message) Allocator.Error!std.json.Value {
    var object: std.json.ObjectMap = .empty;
    try object.put(arena, "role", .{ .string = message.role.name() });
    try object.put(arena, "content", .{ .string = message.content });
    try object.put(arena, "reasoning", if (message.reasoning) |reasoning| .{ .string = reasoning } else .null);
    try object.put(arena, "tool_calls", try inference.toolCallsValue(arena, message.tool_calls));
    try object.put(arena, "tool_call_id", if (message.tool_call_id) |id| .{ .string = id } else .null);
    try object.put(arena, "continuation", try inference.continuationsValue(arena, message.continuation));
    return .{ .object = object };
}

pub fn requestValue(arena: Allocator, request: Request) Allocator.Error!std.json.Value {
    var messages: std.json.Array = .init(arena);
    try messages.ensureTotalCapacity(request.messages.len);
    for (request.messages) |message| messages.appendAssumeCapacity(try messageValue(arena, message));
    var object: std.json.ObjectMap = .empty;
    try object.put(arena, "messages", .{ .array = messages });
    try object.put(arena, "options", try optionsValue(arena, request.options));
    try object.put(arena, "tools", try inference.toolsValue(arena, request.tools));
    return .{ .object = object };
}

pub fn completionValue(arena: Allocator, completion: Completion) Allocator.Error!std.json.Value {
    var object: std.json.ObjectMap = .empty;
    try object.put(arena, "text", .{ .string = completion.text });
    try object.put(arena, "reasoning", if (completion.reasoning) |reasoning| .{ .string = reasoning } else .null);
    try object.put(arena, "tool_calls", try inference.toolCallsValue(arena, completion.tool_calls));
    try object.put(arena, "continuation", try inference.continuationsValue(arena, completion.continuation));
    try object.put(arena, "provider_data", completion.provider_data orelse .null);
    return .{ .object = object };
}

/// Owning readers accept extra row fields, allowing a transcript call to carry
/// completion evidence alongside its canonical request without dropping it.
pub fn optionsFromValue(arena: Allocator, input: std.json.Value) inference.MetadataError!Options {
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

pub fn messageFromValue(arena: Allocator, input: std.json.Value) inference.MetadataError!Message {
    if (input != .object) return error.BadMetadata;
    const role = try metadataString(input.object.get("role") orelse return error.BadMetadata);
    const content = try metadataString(input.object.get("content") orelse return error.BadMetadata);
    var out: Message = .{
        .role = std.meta.stringToEnum(adapter.Role, role) orelse return error.BadMetadata,
        .content = try arena.dupe(u8, content),
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

pub fn requestFromValue(arena: Allocator, input: std.json.Value) inference.MetadataError!Request {
    if (input != .object) return error.BadMetadata;
    const raw_messages = input.object.get("messages") orelse return error.BadMetadata;
    if (raw_messages != .array or raw_messages.array.items.len == 0) return error.BadMetadata;
    const messages = try arena.alloc(Message, raw_messages.array.items.len);
    for (raw_messages.array.items, messages) |message, *copy| copy.* = try messageFromValue(arena, message);
    var out: Request = .{ .messages = messages };
    if (input.object.get("options")) |options| {
        if (options != .null) out.options = try optionsFromValue(arena, options);
    }
    if (input.object.get("tools")) |tools| out.tools = try inference.toolsFromValue(arena, tools);
    return out;
}

pub fn completionFromValue(arena: Allocator, input: std.json.Value) inference.MetadataError!Completion {
    if (input != .object) return error.BadMetadata;
    var out: Completion = .{ .text = try arena.dupe(u8, try metadataString(input.object.get("text") orelse return error.BadMetadata)) };
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
    return out;
}

fn metadataString(input: std.json.Value) inference.MetadataError![]const u8 {
    if (input != .string) return error.BadMetadata;
    return input.string;
}

fn metadataU32(input: std.json.Value) inference.MetadataError!u32 {
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

    pub const VTable = struct {
        /// Returns `request.options.n` completions, allocated in `arena`.
        complete: *const fn (ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion,
    };

    pub fn complete(lm: LM, arena: Allocator, request: Request) Error![]const u8 {
        assert(request.options.n == 1);
        return (try lm.completeResponses(arena, request))[0].text;
    }

    /// Text-only convenience; use `completeResponses` to retain native reasoning.
    pub fn completeMany(lm: LM, arena: Allocator, request: Request) Error![]const []const u8 {
        const completions = try lm.completeResponses(arena, request);
        const out = try arena.alloc([]const u8, completions.len);
        for (out, completions) |*text, completion| text.* = completion.text;
        return out;
    }

    /// All `request.options.n` completions; a provider that returns another
    /// count fails with `LMFailure`.
    pub fn completeResponses(lm: LM, arena: Allocator, request: Request) Error![]const Completion {
        assert(request.messages.len > 0);
        assert(request.options.n >= 1);
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
            var identity = [_]std.json.Value{
                .{ .string = "dspy.adapter.candidate" },
                request.options.rollout_id orelse .null,
                .{ .integer = @intCast(index) },
            };
            var rollout: std.json.Array = .init(arena);
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
    continuation: []const []const Continuation = &.{},
    provider_data: []const ?std.json.Value = &.{},
    supports_reasoning: bool = false,
    supports_response_format: bool = false,
    supports_response_schema: bool = false,
    supports_function_calling: bool = false,
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
        if (self.continuation.len != 0 and self.continuation.len != self.responses.len) return error.LMFailure;
        if (self.provider_data.len != 0 and self.provider_data.len != self.responses.len) return error.LMFailure;
        const out = try arena.alloc(Completion, n);
        for (out, self.responses[self.index..][0..n], self.index..) |*o, r, i| {
            const reasoning = if (self.reasonings.len != 0) self.reasonings[i] else null;
            o.* = try cloneCompletion(arena, .{
                .text = r,
                .reasoning = reasoning,
                .tool_calls = if (self.tool_calls.len != 0) self.tool_calls[i] else &.{},
                .continuation = if (self.continuation.len != 0) self.continuation[i] else &.{},
                .provider_data = if (self.provider_data.len != 0) self.provider_data[i] else null,
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
                .last_user => if (std.mem.indexOf(u8, last, rule.needle) != null) return rule.response,
                .any => for (messages) |m| {
                    if (std.mem.indexOf(u8, m.content, rule.needle) != null) return rule.response;
                },
            }
        }
        return self.default;
    }

    /// `n` copies of the matching rule's response.
    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *KeywordLM = @ptrCast(@alignCast(ptr));
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

    pub const Call = struct {
        messages: []const Message,
        /// Null only for historical transcripts that did not record options.
        options: ?Options = null,
        tools: []const FunctionTool = &.{},
        completion: []const u8,
        /// Every completion of an `n > 1` call (`completion` is the first);
        /// empty for single-completion calls.
        completions: []const []const u8 = &.{},
        /// Native summaries, parallel to `all()`; empty when none were returned.
        reasonings: []const ?[]const u8 = &.{},
        /// Native calls for each completion, parallel to `all()`.
        tool_calls: []const []const ToolCall = &.{},
        continuation: []const []const Continuation = &.{},
        provider_data: []const ?std.json.Value = &.{},

        pub fn all(call: *const Call) []const []const u8 {
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
            if (!sameMessages(call.messages, request.messages)) continue;
            if (!inference.toolsEqual(call.tools, request.tools)) continue;
            if (call.options) |options| {
                if (!sameOptions(options, request.options)) continue;
            }
            if (call.reasonings.len != 0 and call.reasonings.len != recorded.len) return error.LMFailure;
            if (call.tool_calls.len != 0 and call.tool_calls.len != recorded.len) return error.LMFailure;
            if (call.continuation.len != 0 and call.continuation.len != recorded.len) return error.LMFailure;
            if (call.provider_data.len != 0 and call.provider_data.len != recorded.len) return error.LMFailure;
            for (recorded, 0..) |text, i| {
                if (self.used[first + i]) continue;
                out[written] = try cloneCompletion(arena, .{
                    .text = text,
                    .reasoning = if (call.reasonings.len != 0) call.reasonings[i] else null,
                    .tool_calls = if (call.tool_calls.len != 0) call.tool_calls[i] else &.{},
                    .continuation = if (call.continuation.len != 0) call.continuation[i] else &.{},
                    .provider_data = if (call.provider_data.len != 0) call.provider_data[i] else null,
                });
                self.used[first + i] = true;
                written += 1;
                if (written == out.len) return out;
            }
        }
        self.misses += 1;
        return error.LMFailure;
    }

    fn sameOptions(a: Options, b: Options) bool {
        if (a.temperature != b.temperature or
            a.parallel_tool_calls != b.parallel_tool_calls or a.n != b.n or
            a.reasoning_effort_present != b.reasoning_effort_present) return false;
        if (!inference.jsonEqual(a.rollout_id orelse .null, b.rollout_id orelse .null)) return false;
        if (a.response_format) |format| {
            if (!inference.responseFormatEqual(format, b.response_format orelse return false)) return false;
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

    fn sameMessages(a: []const Message, b: []const Message) bool {
        if (a.len != b.len) return false;
        for (a, b) |x, y| {
            if (x.role != y.role or !std.mem.eql(u8, x.content, y.content)) return false;
            if (!optionalStringEqual(x.tool_call_id, y.tool_call_id)) return false;
            if (!optionalStringEqual(x.reasoning, y.reasoning)) return false;
            if (!inference.toolCallsEqual(x.tool_calls, y.tool_calls)) return false;
            if (!inference.continuationsEqual(x.continuation, y.continuation)) return false;
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
        };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *RecordingLM = @ptrCast(@alignCast(ptr));
        const completions = try self.inner.completeResponses(arena, request);
        const messages = try self.arena.alloc(Message, request.messages.len);
        for (request.messages, messages) |src, *dst| dst.* = try cloneMessage(self.arena, src);
        const kept = try self.arena.alloc([]const u8, completions.len);
        var has_reasoning = false;
        for (kept, completions) |*k, c| {
            k.* = try self.arena.dupe(u8, c.text);
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
        var has_continuation = false;
        for (completions) |completion| has_continuation = has_continuation or completion.continuation.len != 0;
        const continuation: [][]const Continuation = if (has_continuation) try self.arena.alloc([]const Continuation, completions.len) else &.{};
        for (continuation, completions[0..continuation.len]) |*kept_state, completion| {
            kept_state.* = try inference.cloneContinuations(self.arena, completion.continuation);
        }
        var has_provider_data = false;
        for (completions) |completion| has_provider_data = has_provider_data or completion.provider_data != null;
        const provider_data: []?std.json.Value = if (has_provider_data) try self.arena.alloc(?std.json.Value, completions.len) else &.{};
        for (provider_data, completions[0..provider_data.len]) |*kept_data, completion| {
            kept_data.* = if (completion.provider_data) |data| try inference.cloneJsonValue(self.arena, data) else null;
        }
        try self.calls.append(self.arena, .{
            .messages = messages,
            .options = try cloneOptions(self.arena, request.options),
            .tools = try inference.cloneTools(self.arena, request.tools),
            .completion = kept[0],
            .completions = if (kept.len > 1) kept else &.{},
            .reasonings = reasonings,
            .tool_calls = tool_calls,
            .continuation = continuation,
            .provider_data = provider_data,
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

    pub fn lm(self: *OpenAI) LM {
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_format,
            .supports_response_schema = self.supports_response_schema,
            .supports_function_calling = self.supports_function_calling,
        };
    }

    /// A Responses request represents one completion; `complete` issues
    /// independent requests for `n > 1`, since Responses has no `n` parameter.
    pub fn requestBody(self: *const OpenAI, arena: Allocator, request: Request) Error![]const u8 {
        const messages = try self.wireMessages(arena, request.messages);
        var body: std.json.ObjectMap = .empty;
        try body.put(arena, "model", .{ .string = self.model });
        try body.put(arena, if (self.api == .responses) "input" else "messages", .{ .array = messages });
        if (request.options.temperature) |t| try body.put(arena, "temperature", .{ .float = t });
        if (self.max_tokens) |n| {
            try body.put(arena, if (self.api == .responses) "max_output_tokens" else "max_tokens", .{ .integer = n });
        }
        if (self.api == .responses) {
            try body.put(arena, "store", .{ .bool = false });
        } else if (request.options.n > 1) {
            try body.put(arena, "n", .{ .integer = request.options.n });
        }
        if (request.options.response_format) |format| {
            const encoded = try self.wireResponseFormat(arena, format);
            if (self.api == .responses) {
                var text: std.json.ObjectMap = .empty;
                try text.put(arena, "format", encoded);
                try body.put(arena, "text", .{ .object = text });
            } else {
                try body.put(arena, "response_format", encoded);
            }
        }
        if (request.tools.len != 0) {
            var tools: std.json.Array = .init(arena);
            try tools.ensureTotalCapacity(request.tools.len);
            for (request.tools) |tool| {
                if (tool.name.len == 0 or tool.parameters != .object) return error.LMFailure;
                var function: std.json.ObjectMap = .empty;
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
                    var wrapper: std.json.ObjectMap = .empty;
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
                var reasoning: std.json.ObjectMap = .empty;
                try reasoning.put(arena, "effort", .{ .string = wire_effort });
                if (!disabled) try reasoning.put(arena, "summary", .{ .string = "auto" });
                try body.put(arena, "reasoning", .{ .object = reasoning });
            } else {
                try body.put(arena, "reasoning_effort", .{ .string = wire_effort });
            }
        }
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeJson(&aw.writer, .{ .object = body }, .compact) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep => error.LMFailure,
        };
        return aw.written();
    }

    fn wireMessages(self: *const OpenAI, arena: Allocator, input: []const Message) Error!std.json.Array {
        var out: std.json.Array = .init(arena);
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
            var object: std.json.ObjectMap = .empty;
            if (self.api == .responses and message.role == .tool) {
                const id = message.tool_call_id orelse return error.LMFailure;
                if (id.len == 0) return error.LMFailure;
                try object.put(arena, "type", .{ .string = "function_call_output" });
                try object.put(arena, "call_id", .{ .string = id });
                try object.put(arena, "output", .{ .string = message.content });
                try out.append(.{ .object = object });
                continue;
            }
            if (self.api == .chat_completions or message.content.len != 0 or message.tool_calls.len == 0 or
                (message.reasoning != null and message.continuation.len == 0))
            {
                try object.put(arena, "role", .{ .string = message.role.name() });
                if (self.api == .responses and message.role == .assistant and
                    message.reasoning != null and message.continuation.len == 0)
                {
                    // lm15 replays visible thinking without native state as
                    // assistant text; it is never silently discarded.
                    var parts: std.json.Array = .init(arena);
                    for ([_][]const u8{ message.reasoning.?, message.content }) |text| {
                        if (text.len == 0) continue;
                        var part: std.json.ObjectMap = .empty;
                        try part.put(arena, "type", .{ .string = "output_text" });
                        try part.put(arena, "text", .{ .string = text });
                        try parts.append(.{ .object = part });
                    }
                    try object.put(arena, "content", .{ .array = parts });
                } else if (self.api == .chat_completions and message.role == .assistant and
                    message.reasoning != null and message.reasoning.?.len != 0)
                {
                    // lm15's default Chat policy replays visible thinking as
                    // assistant text rather than dropping it or sending a
                    // provider-specific reasoning_content request field.
                    const reasoning = message.reasoning.?;
                    const content = if (message.content.len == 0)
                        reasoning
                    else
                        try std.fmt.allocPrint(arena, "{s}\n{s}", .{ reasoning, message.content });
                    try object.put(arena, "content", .{ .string = content });
                } else {
                    try object.put(arena, "content", .{ .string = message.content });
                }
                if (message.role == .tool) {
                    const id = message.tool_call_id orelse return error.LMFailure;
                    if (id.len == 0) return error.LMFailure;
                    try object.put(arena, "tool_call_id", .{ .string = id });
                }
                if (self.api == .chat_completions and message.tool_calls.len != 0) {
                    var calls: std.json.Array = .init(arena);
                    try calls.ensureTotalCapacity(message.tool_calls.len);
                    for (message.tool_calls) |call| calls.appendAssumeCapacity(try self.wireToolCall(arena, call));
                    try object.put(arena, "tool_calls", .{ .array = calls });
                }
                try out.append(.{ .object = object });
            }
            if (self.api == .responses) {
                for (message.tool_calls) |call| try out.append(try self.wireToolCall(arena, call));
            }
        }
        return out;
    }

    fn wireReasoningItem(arena: Allocator, state: Continuation, summary: ?[]const u8) Error!std.json.Value {
        if (!std.mem.eql(u8, state.provider, "openai") or
            !std.mem.eql(u8, state.kind, "reasoning_item") or state.data != .object) return error.LMFailure;
        if (state.data.object.count() == 0) return error.LMFailure;
        var object: std.json.ObjectMap = .empty;
        try object.put(arena, "type", .{ .string = "reasoning" });
        var fields = state.data.object.iterator();
        while (fields.next()) |field| {
            if (!std.mem.eql(u8, field.key_ptr.*, "id") and
                !std.mem.eql(u8, field.key_ptr.*, "encrypted_content")) return error.LMFailure;
            const text = field.value_ptr.*;
            if (text != .string or text.string.len == 0) return error.LMFailure;
            try object.put(arena, field.key_ptr.*, text);
        }
        var parts: std.json.Array = .init(arena);
        if (summary) |text| {
            if (text.len != 0) {
                var part: std.json.ObjectMap = .empty;
                try part.put(arena, "type", .{ .string = "summary_text" });
                try part.put(arena, "text", .{ .string = text });
                try parts.append(.{ .object = part });
            }
        }
        try object.put(arena, "summary", .{ .array = parts });
        return .{ .object = object };
    }

    fn wireToolCall(self: *const OpenAI, arena: Allocator, call: ToolCall) Error!std.json.Value {
        // OpenAI's function-call mapping has no native per-call opaque state.
        // Preserve it in metadata, but refuse a wire that cannot carry it.
        if (call.continuation.len != 0 or call.id.len == 0 or call.name.len == 0) return error.LMFailure;
        var object: std.json.ObjectMap = .empty;
        if (self.api == .responses) {
            try object.put(arena, "type", .{ .string = "function_call" });
            try object.put(arena, "call_id", .{ .string = call.id });
            try object.put(arena, "name", .{ .string = call.name });
            try object.put(arena, "arguments", .{ .string = call.arguments });
        } else {
            var function: std.json.ObjectMap = .empty;
            try function.put(arena, "name", .{ .string = call.name });
            try function.put(arena, "arguments", .{ .string = call.arguments });
            try object.put(arena, "id", .{ .string = call.id });
            try object.put(arena, "type", .{ .string = "function" });
            try object.put(arena, "function", .{ .object = function });
        }
        return .{ .object = object };
    }

    fn wireResponseFormat(self: *const OpenAI, arena: Allocator, format: ResponseFormat) Error!std.json.Value {
        if (format != .json_schema) {
            var object: std.json.ObjectMap = .empty;
            try object.put(arena, "type", .{ .string = @tagName(format) });
            return .{ .object = object };
        }
        const schema = format.json_schema;
        if (schema.name.len == 0 or schema.schema != .object) return error.LMFailure;
        var object: std.json.ObjectMap = .empty;
        try object.put(arena, "name", .{ .string = schema.name });
        try object.put(arena, "schema", schema.schema);
        if (schema.description) |description| try object.put(arena, "description", .{ .string = description });
        try object.put(arena, "strict", .{ .bool = schema.strict });
        if (self.api == .responses) {
            try object.put(arena, "type", .{ .string = "json_schema" });
            return .{ .object = object };
        }
        var wrapper: std.json.ObjectMap = .empty;
        try wrapper.put(arena, "type", .{ .string = "json_schema" });
        try wrapper.put(arena, "json_schema", .{ .object = object });
        return .{ .object = wrapper };
    }

    fn wireNamedTool(self: *const OpenAI, arena: Allocator, name: []const u8) Error!std.json.Value {
        if (name.len == 0) return error.LMFailure;
        var object: std.json.ObjectMap = .empty;
        try object.put(arena, "type", .{ .string = "function" });
        if (self.api == .responses) {
            try object.put(arena, "name", .{ .string = name });
        } else {
            var function: std.json.ObjectMap = .empty;
            try function.put(arena, "name", .{ .string = name });
            try object.put(arena, "function", .{ .object = function });
        }
        return .{ .object = object };
    }

    fn wireToolChoice(self: *const OpenAI, arena: Allocator, choice: ToolChoice) Error!std.json.Value {
        return switch (choice) {
            .auto => .{ .string = "auto" },
            .none => .{ .string = "none" },
            .required => .{ .string = "required" },
            .function => |name| try self.wireNamedTool(arena, name),
            .allowed_tools => |allowed| blk: {
                if (allowed.names.len == 0) break :blk .{ .string = @tagName(allowed.mode) };
                var tools: std.json.Array = .init(arena);
                try tools.ensureTotalCapacity(allowed.names.len);
                for (allowed.names) |name| tools.appendAssumeCapacity(try self.wireNamedTool(arena, name));
                var selection: std.json.ObjectMap = .empty;
                try selection.put(arena, "mode", .{ .string = @tagName(allowed.mode) });
                try selection.put(arena, "tools", .{ .array = tools });
                if (self.api == .responses) {
                    try selection.put(arena, "type", .{ .string = "allowed_tools" });
                    break :blk .{ .object = selection };
                }
                var wrapper: std.json.ObjectMap = .empty;
                try wrapper.put(arena, "type", .{ .string = "allowed_tools" });
                try wrapper.put(arena, "allowed_tools", .{ .object = selection });
                break :blk .{ .object = wrapper };
            },
        };
    }

    /// Send a JSON request through the same bounded, retrying transport used by
    /// generation. `path` is relative to `base_url`; response values own their
    /// strings in `arena`. Additional headers support compatible provider APIs.
    pub fn postJson(self: *OpenAI, arena: Allocator, path: []const u8, body: std.json.Value, headers: []const std.http.Header) anyerror!std.json.Value {
        const url = try std.fmt.allocPrint(arena, "{s}/{s}", .{
            std.mem.trimEnd(u8, self.base_url, "/"),
            std.mem.trimStart(u8, path, "/"),
        });
        const auth = if (self.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
        var encoded: Writer.Allocating = .init(arena);
        pyjson.writeJson(&encoded.writer, body, .compact) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep => error.LMFailure,
        };
        const response = try self.fetchWithRetries(arena, url, auth, encoded.written(), headers);
        defer self.client.allocator.free(response.body);
        return parseJson(arena, response.body);
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const Completion {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
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
            return parseCompletions(arena, response.body);
        }
        const out = try arena.alloc(Completion, request.options.n);
        for (out) |*completion| {
            const response = self.fetchWithRetries(arena, url, auth, body, &.{}) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.LMFailure,
            };
            defer self.client.allocator.free(response.body);
            completion.* = try parseResponsesResponse(arena, response.body);
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

    fn parseJson(arena: Allocator, body: []const u8) Error!std.json.Value {
        // The HTTP body is freed after parsing. Escaped and unescaped strings
        // must both belong to the caller's arena, not to the transport buffer.
        return std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{ .allocate = .alloc_always }) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
    }

    /// Extract `choices[i].message.content` and optional native reasoning.
    pub fn parseCompletions(arena: Allocator, body: []const u8) Error![]const Completion {
        const parsed = try parseJson(arena, body);
        if (parsed != .object) return error.LMFailure;
        const choices = parsed.object.get("choices") orelse return error.LMFailure;
        if (choices != .array or choices.array.items.len == 0) return error.LMFailure;
        const out = try arena.alloc(Completion, choices.array.items.len);
        for (choices.array.items, out) |choice, *completion| {
            if (choice != .object) return error.LMFailure;
            const message = choice.object.get("message") orelse return error.LMFailure;
            if (message != .object) return error.LMFailure;
            completion.* = .{ .text = "", .provider_data = parsed };
            if (message.object.get("tool_calls")) |calls| {
                if (calls != .array) return error.LMFailure;
                const native = try arena.alloc(ToolCall, calls.array.items.len);
                for (calls.array.items, native) |call, *copy| {
                    if (call != .object or !stringEquals(call.object.get("type") orelse return error.LMFailure, "function")) return error.LMFailure;
                    const function = call.object.get("function") orelse return error.LMFailure;
                    if (function != .object) return error.LMFailure;
                    copy.* = .{
                        .id = try responseString(call, "id", true),
                        .name = try responseString(function, "name", true),
                        .arguments = try responseString(function, "arguments", false),
                    };
                }
                completion.tool_calls = native;
            }
            if (message.object.get("content")) |content| {
                switch (content) {
                    .string => completion.text = content.string,
                    .null => if (completion.tool_calls.len == 0) return error.LMFailure,
                    else => return error.LMFailure,
                }
            } else if (completion.tool_calls.len == 0) return error.LMFailure;
            if (message.object.get("reasoning_content")) |reasoning| {
                switch (reasoning) {
                    .null => {},
                    .string => completion.reasoning = reasoning.string,
                    else => return error.LMFailure,
                }
            }
        }
        return out;
    }

    /// Text-only Chat Completions parsing, matching `LM.completeMany`.
    pub fn parseResponse(arena: Allocator, body: []const u8) Error![]const []const u8 {
        const completions = try parseCompletions(arena, body);
        const out = try arena.alloc([]const u8, completions.len);
        for (out, completions) |*text, completion| text.* = completion.text;
        return out;
    }

    /// A Responses output may begin with reasoning or other non-message items.
    /// Completed assistant output_text or actual function calls are answers;
    /// refusal/incomplete responses never masquerade as successful model text.
    pub fn parseResponsesResponse(arena: Allocator, body: []const u8) Error!Completion {
        const parsed = try parseJson(arena, body);
        if (parsed != .object) return error.LMFailure;
        const status = parsed.object.get("status") orelse return error.LMFailure;
        if (!stringEquals(status, "completed")) return error.LMFailure;
        const output = parsed.object.get("output") orelse return error.LMFailure;
        if (output != .array) return error.LMFailure;
        var text: TextParts = .{ .joined = .init(arena) };
        var reasoning: TextParts = .{ .joined = .init(arena) };
        var calls: std.ArrayList(ToolCall) = .empty;
        var continuation: std.ArrayList(Continuation) = .empty;
        for (output.array.items) |item| {
            if (item != .object) return error.LMFailure;
            const kind = item.object.get("type") orelse return error.LMFailure;
            if (stringEquals(kind, "reasoning")) {
                if (try reasoningText(arena, item)) |summary| try reasoning.add(summary, "");
                var state: std.json.ObjectMap = .empty;
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
            } else if (stringEquals(kind, "function_call")) {
                if (item.object.get("status")) |call_status| {
                    if (!stringEquals(call_status, "completed")) return error.LMFailure;
                }
                try calls.append(arena, .{
                    .id = try responseString(item, "call_id", true),
                    .name = try responseString(item, "name", true),
                    .arguments = try responseString(item, "arguments", false),
                });
            } else if (stringEquals(kind, "message")) {
                const role = item.object.get("role") orelse return error.LMFailure;
                if (!stringEquals(role, "assistant")) return error.LMFailure;
                if (item.object.get("status")) |message_status| {
                    if (!stringEquals(message_status, "completed")) return error.LMFailure;
                }
                const content = item.object.get("content") orelse return error.LMFailure;
                if (content != .array) return error.LMFailure;
                for (content.array.items) |part| {
                    if (part != .object) return error.LMFailure;
                    const part_type = part.object.get("type") orelse return error.LMFailure;
                    if (stringEquals(part_type, "refusal")) return error.LMFailure;
                    if (!stringEquals(part_type, "output_text")) continue;
                    const value = part.object.get("text") orelse return error.LMFailure;
                    if (value != .string) return error.LMFailure;
                    try text.add(value.string, "");
                }
            }
        }
        const answer = text.value() orelse "";
        if (answer.len == 0 and calls.items.len == 0) return error.LMFailure;
        return .{
            .text = answer,
            .reasoning = reasoning.value(),
            .tool_calls = calls.items,
            .continuation = continuation.items,
            .provider_data = parsed,
        };
    }

    fn reasoningText(arena: Allocator, item: std.json.Value) Error!?[]const u8 {
        if (item.object.get("summary")) |summary| {
            if (summary == .array) {
                var parts: TextParts = .{ .joined = .init(arena) };
                for (summary.array.items) |part| {
                    if (part != .object) return error.LMFailure;
                    if (!stringEquals(part.object.get("type") orelse return error.LMFailure, "summary_text")) continue;
                    try parts.add(try responseString(part, "text", false), "\n");
                }
                return parts.value();
            }
            if (summary == .string) {
                if (summary.string.len != 0) return summary.string;
            } else if (summary != .null) return error.LMFailure;
        }
        if (item.object.get("text")) |text| {
            if (text != .string) return error.LMFailure;
            if (text.string.len != 0) return text.string;
        }
        return null;
    }

    fn responseString(object: std.json.Value, name: []const u8, nonempty: bool) Error![]const u8 {
        if (object != .object) return error.LMFailure;
        const text = object.object.get(name) orelse return error.LMFailure;
        if (text != .string or (nonempty and text.string.len == 0)) return error.LMFailure;
        return text.string;
    }

    fn stringEquals(value: std.json.Value, expected: []const u8) bool {
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

test "keyword LM scopes and OpenAI wire format" {
    var kw: KeywordLM = .{
        .rules = &.{
            .{ .needle = "capital", .response = "A" },
            .{ .needle = "hint", .response = "B", .scope = .any },
        },
        .default = "C",
    };
    const msgs = [_]Message{ .{ .role = .system, .content = "hint" }, .{ .role = .user, .content = "q" } };
    try testing.expectEqualStrings("B", kw.respond(&msgs));
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var client: OpenAI = .{ .client = undefined, .base_url = "http://x", .model = "m" };
    const body = try client.requestBody(arena.allocator(), .{ .messages = &msgs, .options = .{ .temperature = 0.0 } });
    try testing.expectEqualStrings(
        \\{"model": "m", "messages": [{"role": "system", "content": "hint"}, {"role": "user", "content": "q"}], "temperature": 0.0}
    , body);
    const content = try OpenAI.parseResponse(arena.allocator(),
        \\{"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    );
    try testing.expectEqual(@as(usize, 1), content.len);
    try testing.expectEqualStrings("hi", content[0]);
    const many = try OpenAI.parseResponse(arena.allocator(),
        \\{"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]}
    );
    try testing.expectEqualStrings("b", many[1]);
    const body_n = try client.requestBody(arena.allocator(), .{ .messages = &msgs, .options = .{ .n = 3 } });
    try testing.expect(std.mem.endsWith(u8, body_n, "\"n\": 3}"));
}

test "replay LM matches recorded requests exactly and reports misses" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = [_]Message{.{ .role = .user, .content = "a" }};
    const b = [_]Message{.{ .role = .user, .content = "b" }};
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
    try testing.expectEqualStrings("b", recorder.calls.items[0].messages[0].content);
}
