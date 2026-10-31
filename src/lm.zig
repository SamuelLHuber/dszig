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
        pyjson.writeJson(&aw.writer, .{ .object = body }, .compact) catch return error.OutOfMemory;
        return aw.written();
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) Error![]const u8 {
        const self: *OpenAI = @ptrCast(@alignCast(ptr));
        const body = try self.requestBody(arena, request);
        const url = try std.fmt.allocPrint(arena, "{s}/chat/completions", .{self.base_url});
        var auth_buf: ?[]const u8 = null;
        if (self.api_key) |key| auth_buf = try std.fmt.allocPrint(arena, "Bearer {s}", .{key});
        var response: Writer.Allocating = .init(arena);
        const result = self.client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .response_writer = &response.writer,
            .headers = .{
                .content_type = .{ .override = "application/json" },
                .authorization = if (auth_buf) |a| .{ .override = a } else .default,
            },
        }) catch return error.LMFailure;
        if (result.status != .ok) return error.LMFailure;
        if (response.written().len > self.max_response_bytes) return error.LMFailure;
        return parseResponse(arena, response.written());
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
}
