//! Language-model interface. Programs never name a model: they call through
//! `LM`, so a scripted mock, a keyword mock, or an OpenAI-compatible HTTP
//! endpoint can be swapped without touching program code.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const adapter = @import("adapter.zig");
const pyjson = @import("pyjson.zig");
const Message = adapter.Message;

pub const Options = struct {
    temperature: ?f64 = null,
    /// DSPy `rollout_id`: distinguishes otherwise-identical retries.
    rollout_id: ?u32 = null,
    /// Ask the provider for a JSON object response (JSONAdapter).
    json_mode: bool = false,
};

pub const Request = struct {
    messages: []const Message,
    options: Options = .{},
};

pub const Error = error{ LMFailure, ScriptExhausted } || Allocator.Error;

pub const LM = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Returns the completion text, allocated in `arena`.
        complete: *const fn (ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8,
    };

    pub fn complete(lm: LM, arena: Allocator, request: Request) Error![]const u8 {
        assert(request.messages.len > 0);
        return lm.vtable.complete(lm.ptr, arena, request);
    }
};

/// Returns canned responses in order; fails once they run out.
pub const ScriptedLM = struct {
    responses: []const []const u8,
    index: usize = 0,
    /// Optional transcript of every request (deep-copied into its arena).
    history: ?*History = null,

    pub fn lm(self: *ScriptedLM) LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8 {
        const self: *ScriptedLM = @ptrCast(@alignCast(ptr));
        if (self.history) |h| try h.record(request);
        if (self.index >= self.responses.len) return error.ScriptExhausted;
        defer self.index += 1;
        return arena.dupe(u8, self.responses[self.index]);
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

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8 {
        const self: *KeywordLM = @ptrCast(@alignCast(ptr));
        if (self.history) |h| try h.record(request);
        return arena.dupe(u8, self.respond(request.messages));
    }
};

/// Replays a recorded transcript (e.g. real-model calls captured from DSPy).
/// Each request must byte-match an unused recorded request (roles and
/// contents); the first such call's completion is returned. A request with
/// no recorded twin fails with `LMFailure`, so a replay that completes with
/// `exhausted()` proves both runtimes sent identical prompts.
pub const ReplayLM = struct {
    calls: []const Call,
    /// Parallel to `calls`; set once a call has been replayed.
    used: []bool,
    misses: usize = 0,
    history: ?*History = null,

    pub const Call = struct { messages: []const Message, completion: []const u8 };

    pub fn lm(self: *ReplayLM) LM {
        assert(self.calls.len == self.used.len);
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    pub fn exhausted(self: *const ReplayLM) bool {
        for (self.used) |u| if (!u) return false;
        return self.misses == 0;
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8 {
        const self: *ReplayLM = @ptrCast(@alignCast(ptr));
        if (self.history) |h| try h.record(request);
        for (self.calls, self.used) |call, *used| {
            if (used.* or !sameMessages(call.messages, request.messages)) continue;
            used.* = true;
            return arena.dupe(u8, call.completion);
        }
        self.misses += 1;
        return error.LMFailure;
    }

    fn sameMessages(a: []const Message, b: []const Message) bool {
        if (a.len != b.len) return false;
        for (a, b) |x, y| {
            if (x.role != y.role or !std.mem.eql(u8, x.content, y.content)) return false;
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
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8 {
        const self: *RecordingLM = @ptrCast(@alignCast(ptr));
        const completion = try self.inner.complete(arena, request);
        const messages = try self.arena.alloc(Message, request.messages.len);
        for (request.messages, messages) |src, *dst| {
            dst.* = .{ .role = src.role, .content = try self.arena.dupe(u8, src.content) };
        }
        try self.calls.append(self.arena, .{ .messages = messages, .completion = try self.arena.dupe(u8, completion) });
        return completion;
    }
};

/// Transcript of LM requests, for golden comparisons and inspection.
pub const History = struct {
    arena: Allocator,
    requests: std.ArrayList(Request) = .empty,

    pub fn record(h: *History, request: Request) Allocator.Error!void {
        const messages = try h.arena.alloc(Message, request.messages.len);
        for (request.messages, messages) |src, *dst| {
            dst.* = .{ .role = src.role, .content = try h.arena.dupe(u8, src.content) };
        }
        try h.requests.append(h.arena, .{ .messages = messages, .options = request.options });
    }

    pub fn last(h: *const History) ?Request {
        if (h.requests.items.len == 0) return null;
        return h.requests.items[h.requests.items.len - 1];
    }
};

/// Minimal OpenAI-compatible `/chat/completions` client on `std.http`.
pub const OpenAI = struct {
    client: *std.http.Client,
    /// e.g. "https://api.openai.com/v1" (no trailing slash).
    base_url: []const u8,
    model: []const u8,
    /// Bearer token; never logged.
    api_key: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    /// Upper bound on accepted response bodies.
    max_response_bytes: usize = 4 * 1024 * 1024,
    /// Extra attempts after a transport error, HTTP 429 or 5xx (DSPy's
    /// `num_retries`). Backoff doubles from `retry_base_ms`.
    max_retries: u8 = 3,
    retry_base_ms: u32 = 500,

    pub fn lm(self: *OpenAI) LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    /// Request body as sent on the wire (exposed for tests).
    pub fn requestBody(self: *const OpenAI, arena: Allocator, request: Request) Error![]const u8 {
        var messages: std.json.Array = .init(arena);
        for (request.messages) |m| {
            var obj: std.json.ObjectMap = .empty;
            try obj.put(arena, "role", .{ .string = m.role.name() });
            try obj.put(arena, "content", .{ .string = m.content });
            try messages.append(.{ .object = obj });
        }
        var body: std.json.ObjectMap = .empty;
        try body.put(arena, "model", .{ .string = self.model });
        try body.put(arena, "messages", .{ .array = messages });
        if (request.options.temperature) |t| try body.put(arena, "temperature", .{ .float = t });
        if (self.max_tokens) |n| try body.put(arena, "max_tokens", .{ .integer = n });
        if (request.options.json_mode) {
            var format: std.json.ObjectMap = .empty;
            try format.put(arena, "type", .{ .string = "json_object" });
            try body.put(arena, "response_format", .{ .object = format });
        }
        var aw: Writer.Allocating = .init(arena);
        pyjson.writeJson(&aw.writer, .{ .object = body }, .compact) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.JsonTooDeep => error.LMFailure,
        };
        return aw.written();
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8 {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        const body = try self.requestBody(arena, request);
        const url = try std.fmt.allocPrint(arena, "{s}/chat/completions", .{self.base_url});
        var auth_buf: ?[]const u8 = null;
        if (self.api_key) |key| auth_buf = try std.fmt.allocPrint(arena, "Bearer {s}", .{key});
        var attempt: u8 = 0;
        while (true) : (attempt += 1) {
            var response: Writer.Allocating = .init(arena);
            const status: ?std.http.Status = if (self.client.fetch(.{
                .location = .{ .url = url },
                .method = .POST,
                .payload = body,
                .response_writer = &response.writer,
                .headers = .{
                    .content_type = .{ .override = "application/json" },
                    .authorization = if (auth_buf) |a| .{ .override = a } else .default,
                },
            })) |result| result.status else |_| null;
            if (status == .ok) {
                if (response.written().len > self.max_response_bytes) return error.LMFailure;
                return parseResponse(arena, response.written());
            }
            if (!retryable(status) or attempt >= self.max_retries) return error.LMFailure;
            const delay_ms = @as(i64, self.retry_base_ms) << @intCast(attempt);
            std.Io.sleep(self.client.io, .fromMilliseconds(delay_ms), .awake) catch return error.LMFailure;
        }
    }

    /// Transport failures (null), rate limits and server errors.
    pub fn retryable(status: ?std.http.Status) bool {
        const s = status orelse return true;
        return s == .too_many_requests or @backingInt(s) >= 500;
    }

    /// Extract `choices[0].message.content`.
    pub fn parseResponse(arena: Allocator, body: []const u8) Error![]const u8 {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return error.LMFailure;
        if (parsed != .object) return error.LMFailure;
        const choices = parsed.object.get("choices") orelse return error.LMFailure;
        if (choices != .array or choices.array.items.len == 0) return error.LMFailure;
        const first = choices.array.items[0];
        if (first != .object) return error.LMFailure;
        const message = first.object.get("message") orelse return error.LMFailure;
        if (message != .object) return error.LMFailure;
        const content = message.object.get("content") orelse return error.LMFailure;
        if (content != .string) return error.LMFailure;
        return content.string;
    }
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
    try testing.expectEqualStrings("hi", content);
    try testing.expect(OpenAI.retryable(null));
    try testing.expect(OpenAI.retryable(.too_many_requests));
    try testing.expect(OpenAI.retryable(.service_unavailable));
    try testing.expect(!OpenAI.retryable(.unauthorized));
    try testing.expect(!OpenAI.retryable(.bad_request));
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
