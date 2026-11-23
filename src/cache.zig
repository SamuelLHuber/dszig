//! `CachedLM`: DSPy's response cache (`dspy.cache`) as an LM wrapper. A
//! request is keyed by the SHA-256 of its messages and options (temperature,
//! `n`, `rollout_id`, JSON mode), so like DSPy a different `rollout_id`
//! bypasses earlier entries. Entries live in memory, least recently used
//! evicted first; `dump`/`load` persist them as JSON. Failed calls are not
//! cached. DSPy's on-disk `diskcache` format and key hashing are not shared.
//!
//! Upstream (DSPy 3.5.0): dspy/clients/__init__.py

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Value = @import("runtime_value.zig").Value;

const adapter = @import("adapter.zig");
const lm_mod = @import("lm.zig");
const pyjson = @import("pyjson.zig");

const LM = lm_mod.LM;
const Request = lm_mod.Request;
const Completion = lm_mod.Completion;
const sdk = @import("sdk_response.zig");
const results = @import("call_result.zig");

pub const Key = [64]u8;

pub const CachedLM = struct {
    gpa: Allocator,
    inner: LM,
    /// `memory_max_entries`.
    max_entries: usize = 1_000_000,
    entries: std.AutoArrayHashMapUnmanaged(Key, Entry) = .empty,
    hits: usize = 0,
    misses: usize = 0,

    const Entry = struct {
        completions: []const Completion,
        /// Public calls store Source's canonical record before any deep copy.
        /// Native text-only calls need no public output/metadata projection.
        record: ?sdk.PayloadValue = null,
    };

    pub fn deinit(self: *CachedLM) void {
        for (self.entries.values()) |entry| self.freeEntry(entry);
        self.entries.deinit(self.gpa);
        self.* = undefined;
    }

    pub fn lm(self: *CachedLM) LM {
        return .{
            .ptr = self,
            .vtable = if (self.inner.vtable.model_name == null)
                &.{ .complete = complete, .handles_candidate_groups = true, .caches_responses = cachesResponses }
            else
                &.{ .complete = complete, .handles_candidate_groups = true, .model_name = modelName, .caches_responses = cachesResponses },
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
        const self: *CachedLM = @ptrCast(@alignCast(ptr));
        return self.inner.modelName().?;
    }

    fn cachesResponses(ptr: *anyopaque) bool {
        const self: *CachedLM = @ptrCast(@alignCast(ptr));
        return self.max_entries > 0;
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) lm_mod.Error![]const Completion {
        const self: *CachedLM = @ptrCast(@alignCast(ptr));
        if (request.execution) |execution| {
            if (!execution.use_cache) return self.inner.completeResponses(arena, request);
        }
        const k = try key(arena, request);
        if (self.entries.fetchOrderedRemove(k)) |hit| {
            // Re-insert at the end: most recently used.
            self.entries.putAssumeCapacity(k, hit.value);
            self.hits += 1;
            if (hit.value.record) |record| {
                // Source stores the record itself and deep-copies it on get.
                // Clone the entire graph once, then reconstruct every SDK
                // response occurrence separately, retaining shared payloads.
                var context: sdk.Response.CloneContext = .init(arena);
                defer context.deinit();
                const copied = try context.clonePayload(record);
                defer copied.free();
                const loaded = try results.CallResult.load(arena, copied);
                var transferred = false;
                defer if (!transferred) loaded.free(arena);
                if (loaded.responses.len != hit.value.completions.len) return error.InvalidCallResult;
                const completions = try dupeCompletions(arena, hit.value.completions, loaded.responses);
                for (completions) |*completion| completion.cache_hit = true;
                if (request.execution) |execution| if (execution.want_result and execution.result == null) {
                    execution.result = loaded;
                    transferred = true;
                };
                return completions;
            }
            const completions = try dupeCompletions(arena, hit.value.completions, null);
            for (completions) |*completion| completion.cache_hit = true;
            return completions;
        }
        self.misses += 1;
        const completions = try self.inner.completeResponses(arena, request);
        const record = if (request.execution) |execution|
            if (execution.result) |result| try result.dump(self.gpa) else null
        else
            null;
        defer if (record) |value| value.free();
        try self.insert(k, completions, record);
        return completions;
    }

    fn insert(self: *CachedLM, k: Key, completions: []const Completion, record: ?sdk.PayloadValue) results.Error!void {
        assert(self.max_entries > 0);
        // Own immutable SDK headers in cache storage, but do not copy the
        // record's opaque mutable graph until lookup.
        const loaded = if (record) |value| try results.CallResult.load(self.gpa, value) else null;
        defer if (loaded) |value| value.free(self.gpa);
        if (loaded) |value| if (value.responses.len != completions.len) return error.InvalidCallResult;
        const owned = try dupeCompletions(self.gpa, completions, if (loaded) |value| value.responses else null);
        errdefer self.freeCompletions(owned);
        try self.entries.ensureUnusedCapacity(self.gpa, 1);
        if (self.entries.count() >= self.max_entries) {
            self.freeEntry(self.entries.values()[0]);
            self.entries.orderedRemoveAt(0);
        }
        self.entries.putAssumeCapacity(k, .{
            .completions = owned,
            .record = if (record) |value| value.retain() else null,
        });
    }

    fn freeEntry(self: *CachedLM, entry: Entry) void {
        self.freeCompletions(entry.completions);
        if (entry.record) |record| record.free();
    }

    fn freeCompletions(self: *CachedLM, completions: []const Completion) void {
        for (completions) |c| lm_mod.freeCompletion(self.gpa, c);
        self.gpa.free(completions);
    }

    /// Native completion carriers and optional canonical records, LRU first.
    pub fn dump(self: *const CachedLM, w: *Writer) (Allocator.Error || pyjson.Error)!void {
        var scratch: std.heap.ArenaAllocator = .init(self.gpa);
        defer scratch.deinit();
        try w.writeByte('{');
        for (self.entries.keys(), self.entries.values(), 0..) |k, entry, i| {
            if (i > 0) try w.writeAll(", ");
            try pyjson.writeJsonString(w, &k);
            try w.writeAll(": {\"completions\": [");
            for (entry.completions, 0..) |c, j| {
                if (j > 0) try w.writeAll(", ");
                try std.json.Stringify.value(c, .{}, w);
            }
            try w.writeAll("], \"record\": ");
            if (entry.record) |record| {
                // SDK responses already use the native completion codec above,
                // including nonfinite score tags. Persist public snapshots only;
                // JSON itself does not retain graph identity.
                try w.writeByte('{');
                var written: usize = 0;
                for (record.root.object.keys(), record.root.object.values()) |name, value| {
                    if (std.mem.eql(u8, name, "responses")) continue;
                    if (written != 0) try w.writeAll(", ");
                    try pyjson.writeJsonString(w, name);
                    try w.writeAll(": ");
                    const view = (sdk.PayloadValue.borrowed(value)).toValue(scratch.allocator()) catch |err| return switch (err) {
                        error.OutOfMemory => error.OutOfMemory,
                        else => error.ParseValue,
                    };
                    try pyjson.writeJsonProjection(w, view, .compact, 0);
                    written += 1;
                }
                try w.writeByte('}');
            } else try w.writeAll("null");
            try w.writeByte('}');
        }
        try w.writeByte('}');
    }

    /// Add the entries of a `dump`.
    pub fn load(self: *CachedLM, json: []const u8) (Allocator.Error || error{InvalidCache})!void {
        var scratch: std.heap.ArenaAllocator = .init(self.gpa);
        defer scratch.deinit();
        const root = std.json.parseFromSliceLeaky(Value, scratch.allocator(), json, .{}) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidCache,
        };
        if (root != .object) return error.InvalidCache;
        var it = root.object.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.len != 64 or entry.value_ptr.* != .object) return error.InvalidCache;
            const encoded = entry.value_ptr.object.get("completions") orelse return error.InvalidCache;
            if (encoded != .array) return error.InvalidCache;
            const items = encoded.array.items;
            if (items.len == 0) return error.InvalidCache;
            const completions = try scratch.allocator().alloc(Completion, items.len);
            for (items, completions) |item, *c| {
                c.* = lm_mod.completionFromValue(scratch.allocator(), item) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidCache,
                };
            }
            if (self.entries.contains(entry.key_ptr.*[0..64].*)) continue;
            const encoded_record = entry.value_ptr.object.get("record") orelse return error.InvalidCache;
            const record = if (encoded_record != .null) block: {
                if (encoded_record != .object) return error.InvalidCache;
                var responses: @import("runtime_value.zig").Array = .init(scratch.allocator());
                try responses.ensureTotalCapacity(completions.len);
                for (completions) |completion| {
                    const response = completion.response orelse return error.InvalidCache;
                    responses.appendAssumeCapacity(response.toValue(scratch.allocator()) catch |err| return switch (err) {
                        error.OutOfMemory => error.OutOfMemory,
                        else => error.InvalidCache,
                    });
                }
                var fields = encoded_record.object;
                try fields.put(scratch.allocator(), "responses", .{ .array = responses });
                break :block sdk.PayloadValue.fromPythonValue(self.gpa, .{ .object = fields }) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidCache,
                };
            } else null;
            defer if (record) |value| value.free();
            self.insert(entry.key_ptr.*[0..64].*, completions, record) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidCache,
            };
        }
    }
};

fn dupeCompletions(a: Allocator, completions: []const Completion, responses: ?[]const *const sdk.Response) Allocator.Error![]Completion {
    if (responses) |objects| assert(objects.len == completions.len);
    var context: @import("sdk_response.zig").Response.CloneContext = .init(a);
    defer context.deinit();
    const out = try a.alloc(Completion, completions.len);
    var done: usize = 0;
    errdefer {
        for (out[0..done]) |c| lm_mod.freeCompletion(a, c);
        a.free(out);
    }
    for (completions, out, 0..) |c, *o, index| {
        o.* = if (responses) |objects|
            try lm_mod.cloneCompletionWithResponse(a, c, objects[index])
        else
            try lm_mod.cloneCompletionWithContext(a, c, &context);
        done += 1;
    }
    return out;
}

/// Hex SHA-256 of the request as canonical JSON.
pub fn key(arena: Allocator, request: Request) Allocator.Error!Key {
    var aw: Writer.Allocating = .init(arena);
    writeCanonical(&aw.writer, request) catch return error.OutOfMemory;
    var hash: [32]u8 = undefined;
    var hasher: std.crypto.hash.sha2.Sha256 = .init(.{});
    hasher.update(aw.written());
    // JSON cannot contain this raw separator. Preserve existing single-call
    // keys while separating canonical groups with different candidate counts.
    if (request.execution) |execution| {
        if (execution.canonical != null and request.options.n != 1) {
            var count: [5]u8 = .{ 0, 0, 0, 0, 0 };
            std.mem.writeInt(u32, count[1..], request.options.n, .little);
            hasher.update(&count);
        }
    }
    hasher.final(&hash);
    return std.fmt.bytesToHex(hash, .lower);
}

fn writeCanonical(w: *Writer, request: Request) Writer.Error!void {
    if (request.execution) |execution| {
        if (execution.canonical) |canonical| return std.json.Stringify.value(.{
            .sdk_request = canonical,
            .model_type = execution.model_type,
            .rollout_id = request.options.rollout_id,
        }, .{}, w);
    }
    // Stream every typed field, including native tools and structured schemas.
    // Cache hashes are local to DSZig and are not DSPy's diskcache format.
    // Ordinary requests keep their historical key bytes; decisions add a key.
    if (request.decision == null) return std.json.Stringify.value(.{
        .messages = request.messages,
        .options = request.options,
        .tools = request.tools,
    }, .{}, w);
    try std.json.Stringify.value(.{
        .messages = request.messages,
        .options = request.options,
        .tools = request.tools,
        .decision = request.decision,
    }, .{}, w);
}

const testing = std.testing;

test "CachedLM serves repeats from memory and keys on options" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scripted: lm_mod.ScriptedLM = .{ .responses = &.{ "a", "b", "c", "d", "e" } };
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = scripted.lm() };
    defer cache.deinit();
    const messages = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "hi" } }};
    const base: Request = .{ .messages = &messages, .options = .{ .temperature = 0.0 } };
    const lm = cache.lm();
    try testing.expectEqualStrings("a", try lm.complete(arena, base));
    try testing.expectEqualStrings("a", try lm.complete(arena, base));
    var rollout = base;
    rollout.options.rollout_id = .{ .integer = 1 };
    try testing.expectEqualStrings("b", try lm.complete(arena, rollout));
    try testing.expectEqualStrings("b", try lm.complete(arena, rollout));
    var hot = base;
    hot.options.temperature = 0.7;
    try testing.expectEqualStrings("c", try lm.complete(arena, hot));
    var many = base;
    many.options.n = 2;
    const pair = try lm.completeMany(arena, many);
    try testing.expectEqualStrings("d", pair[0]);
    try testing.expectEqualStrings("e", pair[1]);
    const again = try lm.completeMany(arena, many);
    try testing.expectEqualStrings("e", again[1]);
}

test "CachedLM retries a failed group without reusing a successful prefix" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scripted: lm_mod.ScriptedLM = .{ .responses = &.{"abandoned"} };
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = scripted.lm() };
    defer cache.deinit();
    const messages = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "group" } }};
    const request: Request = .{ .messages = &messages, .options = .{ .n = 2 } };
    try testing.expectError(error.ScriptExhausted, cache.lm().completeResponses(arena, request));
    scripted.responses = &.{ "abandoned", "retry first", "retry second" };
    const retried = try cache.lm().completeMany(arena, request);
    try testing.expectEqualStrings("retry first", retried[0]);
    try testing.expectEqualStrings("retry second", retried[1]);
    const cached = try cache.lm().completeMany(arena, request);
    try testing.expectEqualStrings("retry first", cached[0]);
    try testing.expectEqualStrings("retry second", cached[1]);
}

test "successful responses stay cached when final usage accounting refuses" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const usage = @import("usage_tracker.zig");
    var storage: usage.Storage = .{ .arena = arena, .io = testing.io };
    var tracker: usage.UsageTracker = undefined;
    try tracker.init(&storage, .{ .models = 0, .entries = 0 });
    var engine: lm_mod.ScriptedLM = .{
        .responses = &.{ "first", "second" },
        .metadata = &.{
            .{ .usage = .{ .input_tokens = .{ .integer = 1 } } },
            .{ .usage = .{ .input_tokens = .{ .integer = 1 } } },
        },
    };
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = engine.lm() };
    defer cache.deinit();
    const request: Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "cached before accounting" } }},
        .options = .{ .n = 2 },
        .usage_tracker = &tracker,
    };
    try testing.expectError(error.CapacityExceeded, cache.lm().completeResponses(arena, request));
    const hit = try cache.lm().completeResponses(arena, request);
    try testing.expectEqualStrings("first", hit[0].text.?);
    try testing.expectEqualStrings("second", hit[1].text.?);
    try testing.expectEqual(@as(usize, 2), engine.index);
    try testing.expectEqual(@as(u32, 0), tracker.entry_count);
}

test "CachedLM evicts the least recently used entry and round-trips dump/load" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scripted: lm_mod.ScriptedLM = .{ .responses = &.{ "1", "2", "3", "4" } };
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = scripted.lm(), .max_entries = 2 };
    defer cache.deinit();
    const lm = cache.lm();
    const m1 = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "one" } }};
    const m2 = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "two" } }};
    const m3 = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "three" } }};
    _ = try lm.complete(arena, .{ .messages = &m1 });
    _ = try lm.complete(arena, .{ .messages = &m2 });
    _ = try lm.complete(arena, .{ .messages = &m1 }); // m1 becomes most recent
    _ = try lm.complete(arena, .{ .messages = &m3 }); // evicts m2
    try testing.expectEqual(@as(usize, 2), cache.entries.count());
    try testing.expectEqualStrings("4", try lm.complete(arena, .{ .messages = &m2 }));

    var aw: Writer.Allocating = .init(arena);
    try cache.dump(&aw.writer);
    var failing: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var restored: CachedLM = .{ .gpa = testing.allocator, .inner = failing.lm() };
    defer restored.deinit();
    try restored.load(aw.written());
    try testing.expectEqualStrings("4", try restored.lm().complete(arena, .{ .messages = &m2 }));
    try testing.expectEqualStrings("3", try restored.lm().complete(arena, .{ .messages = &m3 }));
    try testing.expectError(error.InvalidCache, restored.load("{\"x\": []}"));
}

test "invalid combined cost cannot publish through fresh or nested caches" {
    const Native = struct {
        fn complete(_: *anyopaque, arena: Allocator, _: Request) lm_mod.Error![]const Completion {
            const out = try arena.alloc(Completion, 1);
            out[0] = try lm_mod.cloneCompletion(arena, .{
                .text = "answer",
                .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "answer" } }} },
                .metadata = .{ .model = "test", .finish_reason = .stop },
                .billing = .{
                    .cost = .{ .string = "not a numeric cost" },
                    .details = .{ .object = .empty },
                },
            });
            try lm_mod.publishResponse(arena, &out[0], "test", .owned);
            return out;
        }
    };
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    var token: u8 = 0;
    var inner: CachedLM = .{
        .gpa = testing.allocator,
        .inner = .{ .ptr = &token, .vtable = &.{ .complete = Native.complete } },
    };
    defer inner.deinit();
    const request: Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "invalid combined price" } }},
        .options = .{ .n = 2 },
    };
    try testing.expectError(error.InvalidCost, inner.lm().callResult(arena, request));
    try testing.expectEqual(@as(usize, 0), inner.entries.count());
    // An existing legacy cache entry is not proof that public finalization
    // succeeds. Refuse before an enclosing cache can publish that old entry.
    _ = try inner.lm().completeResponses(arena, request);
    var outer: CachedLM = .{ .gpa = testing.allocator, .inner = inner.lm() };
    defer outer.deinit();
    try testing.expectError(error.InvalidCost, outer.lm().callResult(arena, request));
    try testing.expectEqual(@as(usize, 0), outer.entries.count());
    try testing.expectEqual(@as(usize, 1), inner.entries.count());
}

test "cached public results retain grouped cost without charging original usage again" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const catalogue = try @import("model_prices.zig").Catalogue.init(arena,
        \\{"openai/test":{"litellm_provider":"openai","input_cost_per_token":0.00001,"output_cost_per_token":0.00003}}
    , .{});
    var pricing_bytes: [8192]u8 = undefined;
    var pricing: std.heap.FixedBufferAllocator = .init(&pricing_bytes);
    var source: std.heap.ArenaAllocator = .init(testing.allocator);
    const completion = (try lm_mod.OpenAI.parseCompletions(source.allocator(),
        \\{"model":"test","usage":{"prompt_tokens":3,"completion_tokens":7},"choices":[{"message":{"content":"retained"},"finish_reason":"stop"}]}
    ))[0];
    var scripted: lm_mod.ScriptedLM = .{
        .model = "test",
        .responses = &.{ "retained", "retained" },
        .metadata = &.{ completion.metadata, completion.metadata },
        .response_messages = &.{ completion.message, completion.message },
    };
    var model = scripted.lm();
    model.pricing = .{ .catalogue = &catalogue, .arena = pricing.allocator(), .provider = "openai" };
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = model };
    defer cache.deinit();
    const request: Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "two candidates" } }},
        .options = .{ .n = 2 },
    };
    const first = try cache.lm().callResult(source.allocator(), request);
    const expected = first.cost.root.float;
    try testing.expect(!first.cache_hit);
    const public_usage = try first.usage.toValue(arena);
    try testing.expectEqual(@as(i64, 6), public_usage.object.get("prompt_tokens").?.integer);
    source.deinit();
    // The cached cost cannot be recomputed: its dedicated pricing storage is full.
    _ = try pricing.allocator().alloc(u8, pricing_bytes.len - pricing.end_index);
    const hit = try cache.lm().callResult(arena, request);
    try testing.expect(hit.cache_hit);
    try testing.expectEqual(@as(usize, 0), hit.usage.root.object.count());
    try testing.expectEqual(expected, hit.cost.root.float);
    try testing.expectEqual(@as(usize, 2), hit.responses.len);
    try testing.expectEqual(@as(i64, 3), hit.responses[0].fields.usage.input_tokens.integer);
    try testing.expectEqual(@as(usize, 2), hit.cost_details.root.object.get("candidates").?.array.items.len);
    try testing.expectEqual(@as(usize, 2), scripted.index);
}

test "CachedLM preserves native summaries and calls across eviction-independent persistence" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const Native = struct {
        calls: usize = 0,

        fn complete(ptr: *anyopaque, _: Allocator, _: Request) lm_mod.Error![]const Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            return &.{.{
                .text = "",
                .reasoning = "Search before answering.",
                .tool_calls = &.{.{ .id = "call_1", .name = "search", .arguments = "{\"query\":\"Paris\"}" }},
            }};
        }
    };
    var native: Native = .{};
    var cache: CachedLM = .{
        .gpa = testing.allocator,
        .inner = .{ .ptr = &native, .vtable = &.{ .complete = Native.complete }, .supports_function_calling = true },
    };
    defer cache.deinit();
    const messages = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "Find the capital." } }};
    const request: Request = .{ .messages = &messages, .options = .{ .reasoning_effort = "low" } };
    _ = try cache.lm().completeResponses(arena, request);
    const hit = (try cache.lm().completeResponses(arena, request))[0];
    try testing.expectEqualStrings("Search before answering.", hit.reasoning.?);
    try testing.expectEqualStrings("{\"query\":\"Paris\"}", hit.tool_calls[0].arguments);
    try testing.expectEqual(@as(usize, 1), native.calls);

    var writer: Writer.Allocating = .init(arena);
    try cache.dump(&writer.writer);
    var empty: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var restored: CachedLM = .{ .gpa = testing.allocator, .inner = empty.lm() };
    defer restored.deinit();
    try restored.load(writer.written());
    const persisted = (try restored.lm().completeResponses(arena, request))[0];
    try testing.expectEqualStrings("Search before answering.", persisted.reasoning.?);
    try testing.expectEqualStrings("call_1", persisted.tool_calls[0].id);
    try testing.expectEqualStrings("search", persisted.tool_calls[0].name);
    try testing.expectEqualStrings("{\"query\":\"Paris\"}", persisted.tool_calls[0].arguments);
    var changed = request;
    changed.options.reasoning_effort = "high";
    try testing.expectError(error.ScriptExhausted, restored.lm().completeResponses(arena, changed));
}

test "CachedLM persists unknown, reported-empty and nonfinite token scores without loss" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const payload: u64 = 0x7ff80000000000a5;
    const top = [_]lm_mod.TopLogprob{.{ .token = "other", .logprob = -std.math.inf(f64) }};
    const scores = [_]lm_mod.TokenLogprob{.{ .token = "chosen", .logprob = @bitCast(payload), .top = &top }};
    var script: lm_mod.ScriptedLM = .{
        .responses = &.{ "unknown", "empty", "scored" },
        .logprobs = &.{ null, &.{}, &scores },
        .logprobs_complete = &.{ true, true, false },
    };
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = script.lm() };
    defer cache.deinit();
    const messages = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "score" } }};
    const request: Request = .{ .messages = &messages, .options = .{ .n = 3, .logprobs = .{ .integer = 0 } } };
    _ = try cache.lm().completeResponses(arena, request);
    var writer: Writer.Allocating = .init(arena);
    try cache.dump(&writer.writer);

    var empty: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var restored: CachedLM = .{ .gpa = testing.allocator, .inner = empty.lm() };
    defer restored.deinit();
    try restored.load(writer.written());
    const hit = try restored.lm().completeResponses(arena, request);
    try testing.expect(hit[0].logprobs == null);
    try testing.expect(hit[1].logprobs != null);
    try testing.expectEqual(@as(usize, 0), hit[1].logprobs.?.len);
    try testing.expectEqual(payload, @as(u64, @bitCast(hit[2].logprobs.?[0].logprob)));
    try testing.expectEqual(-std.math.inf(f64), hit[2].logprobs.?[0].top[0].logprob);
    try testing.expect(!hit[2].logprobs_complete);
    var alternatives = request;
    alternatives.options.logprobs = .{ .integer = 1 };
    try testing.expectError(error.ScriptExhausted, restored.lm().completeResponses(arena, alternatives));
}

test "canonical cache records snapshot outputs and copy opaque graphs only on lookup" {
    const Probe = struct {
        calls: usize = 0,
        response: *const sdk.Response,

        fn complete(ptr: *anyopaque, arena: Allocator, _: Request) lm_mod.Error![]const Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            const out = try arena.alloc(Completion, 1);
            out[0] = .{ .text = null, .response = self.response };
            return out;
        }

        fn run(gpa: Allocator) !void {
            const graph = @import("json_graph.zig");
            const owner = try @import("request_cache.zig").Payload.create(gpa);
            defer owner.release();
            const fields = try owner.allocator().create(graph.Record);
            fields.* = .empty;
            const items = try owner.allocator().create(graph.Array);
            items.* = .init(owner.allocator());
            try fields.put(owner.allocator(), "items", .{ .array = items });
            owner.root = .{ .object = fields };
            const response = try sdk.Response.fromFields(gpa, &.{
                .model = "cache-alias-response",
                .finish_reason = .stop,
                .message = .{ .role = .assistant, .parts = &.{.{ .tool_call = .{
                    .id = "call",
                    .name = "tool",
                    .input = .{ .root = owner.root, .storage = owner },
                } }} },
                .usage = .{ .input_tokens = .{ .integer = 2 }, .output_tokens = .{ .integer = 3 } },
            });
            defer response.release();
            var probe: @This() = .{ .response = response };
            var cache: CachedLM = .{ .gpa = gpa, .inner = .{
                .ptr = &probe,
                .vtable = &.{ .complete = complete },
                .supports_function_calling = true,
            } };
            defer cache.deinit();
            const request: Request = .{
                .messages = &.{.{ .role = .user, .content = .{ .text = "shared cache payload" } }},
                .options = .{ .n = 2 },
            };
            {
                var execution: std.heap.ArenaAllocator = .init(gpa);
                defer execution.deinit();
                const first = try cache.lm().callResult(execution.allocator(), request);
                defer first.free(execution.allocator());
                try testing.expect(first.responses[0] == response);
                try testing.expect(first.responses[1] == response);
            }
            // The execution allocator has gone away. The SDK owner is still
            // alive and mutations made before lookup belong in the next copy.
            try items.append(.{ .integer = 7 });
            var execution: std.heap.ArenaAllocator = .init(gpa);
            defer execution.deinit();
            const arena = execution.allocator();
            const hit = try cache.lm().callResult(arena, request);
            defer hit.free(arena);
            try testing.expectEqual(@as(usize, 2), probe.calls);
            try testing.expect(hit.responses[0] != hit.responses[1]);
            try testing.expect(hit.responses[0] != response);
            const left = hit.responses[0].fields.message.parts[0].tool_call.input.root;
            const right = hit.responses[1].fields.message.parts[0].tool_call.input.root;
            try testing.expect(left.object == right.object);
            try testing.expect(left.object != fields);
            try testing.expectEqual(@as(i64, 7), left.object.get("items").?.array.items[0].integer);
            const arguments = hit.outputs.root.array.items[0].object.get("tool_calls").?.array.items[0]
                .object.get("function").?.object.get("arguments").?.string;
            try testing.expectEqualStrings("{\"items\": []}", arguments);
            try testing.expectEqual(@as(usize, 0), hit.usage.root.object.count());
            try testing.expectEqual(@as(i64, 2), hit.responses[0].fields.usage.input_tokens.integer);
            try left.object.get("items").?.array.append(.{ .integer = 8 });
            const again = try cache.lm().callResult(arena, request);
            defer again.free(arena);
            try testing.expectEqual(@as(usize, 1), again.responses[0].fields.message.parts[0]
                .tool_call.input.root.object.get("items").?.array.items.len);

            var bytes: [16 * 1024]u8 = undefined;
            var encoded: Writer = .fixed(&bytes);
            try cache.dump(&encoded);
            var empty: lm_mod.ScriptedLM = .{ .responses = &.{} };
            var restored: CachedLM = .{ .gpa = gpa, .inner = empty.lm() };
            defer restored.deinit();
            try restored.load(encoded.buffered());
            const persisted = try restored.lm().callResult(arena, request);
            defer persisted.free(arena);
            try testing.expectEqual(@as(i64, 7), persisted.responses[0].fields.message.parts[0]
                .tool_call.input.root.object.get("items").?.array.items[0].integer);
            try testing.expectEqualStrings(arguments, persisted.outputs.root.array.items[0]
                .object.get("tool_calls").?.array.items[0].object.get("function").?.object.get("arguments").?.string);
        }
    };
    var stable = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable.allocator(), Probe.run, .{});
}

test "public canonical retries replay only the failed candidate, before cache and accounting" {
    const Mode = enum { recover, disabled, exhaust, terminal, abort_backoff };
    const Engine = struct {
        mode: Mode,
        seeds: [4]i64 = undefined,
        calls: usize = 0,
        backoffs: u8 = 0,
        failed_request: ?*const sdk.Request = null,

        fn complete(_: *anyopaque, _: Allocator, _: Request) lm_mod.Error![]const Completion {
            unreachable;
        }

        fn name(_: *anyopaque) []const u8 {
            return "retry-model";
        }

        fn backoff(ptr: *anyopaque, request: *const sdk.Request, number: u8) lm_mod.Error!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            assert(request == self.failed_request.?);
            self.backoffs += 1;
            assert(number == self.backoffs);
            if (self.mode == .abort_backoff) return error.StreamStopped;
        }

        fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) lm_mod.Error!Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const seed = request.fields.config.seed.integer;
            self.seeds[self.calls] = seed;
            self.calls += 1;
            if (seed == 41) {
                if (self.failed_request) |first| assert(request == first);
                self.failed_request = request;
                if (self.mode == .terminal) return error.LMFailure;
                if (self.mode != .recover or self.calls == 2) return error.RetryableLMFailure;
            }
            const text = if (seed == 40) "first" else "second";
            var result: Completion = .{
                .text = text,
                .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = text } }} },
                .metadata = .{ .usage = .{ .input_tokens = .{ .integer = 1 } } },
            };
            try lm_mod.publishResponse(arena, &result, request.fields.model, .borrowed);
            return result;
        }
    };
    for ([_]Mode{ .recover, .disabled, .exhaust, .terminal, .abort_backoff }) |mode| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        const usage = @import("usage_tracker.zig");
        var storage: usage.Storage = .{ .arena = arena, .io = testing.io };
        var tracker: usage.UsageTracker = undefined;
        try tracker.init(&storage, .{ .models = 1, .entries = 1 });
        var engine: Engine = .{ .mode = mode };
        var vtable: LM.VTable = .{
            .complete = Engine.complete,
            .complete_canonical = Engine.canonical,
            .model_name = Engine.name,
        };
        if (mode != .disabled) vtable.canonical_retry = .{ .max_retries = 2, .backoff = Engine.backoff };
        var cache: CachedLM = .{ .gpa = testing.allocator, .inner = .{ .ptr = &engine, .vtable = &vtable } };
        defer cache.deinit();
        const options: LM.PromptCall = .{
            .n = 2,
            .config = .{ .seed = .{ .integer = 40 } },
            .execution = .{ .usage_tracker = &tracker },
        };
        if (mode == .recover) {
            for ([_]bool{ false, true }) |hit| {
                const result = try cache.lm().callPromptResult(arena, testing.io, "prompt", options);
                defer result.free(arena);
                try testing.expectEqual(hit, result.cache_hit);
                try testing.expectEqual(2, result.responses.len);
                try testing.expectEqualStrings("first", result.outputs.root.array.items[0].string);
                try testing.expectEqualStrings("second", result.outputs.root.array.items[1].string);
            }
        } else {
            const expected: lm_mod.Error = switch (mode) {
                .recover => unreachable,
                .disabled, .exhaust => error.RetryableLMFailure,
                .terminal => error.LMFailure,
                .abort_backoff => error.StreamStopped,
            };
            try testing.expectError(expected, cache.lm().callPromptResult(arena, testing.io, "prompt", options));
            try testing.expectEqual(0, cache.entries.count());
        }
        const seeds: []const i64 = switch (mode) {
            .recover => &.{ 40, 41, 41 },
            .exhaust => &.{ 40, 41, 41, 41 },
            else => &.{ 40, 41 },
        };
        try testing.expectEqualSlices(i64, seeds, engine.seeds[0..engine.calls]);
        const waits: u8 = switch (mode) {
            .recover, .abort_backoff => 1,
            .exhaust => 2,
            else => 0,
        };
        try testing.expectEqual(waits, engine.backoffs);
        try testing.expectEqual(1, cache.misses);
        try testing.expectEqual(1, tracker.entry_count);
        const total = try usage.Graph.toValue(arena, try tracker.getTotalTokens(arena));
        const tokens = total.object.get("retry-model").?.object.get("prompt_tokens").?.integer;
        try testing.expectEqual(@as(i64, if (mode == .recover) 2 else 1), tokens);
    }
}

test "canonical cache separates SDK configuration and honors public bypass" {
    const Engine = struct {
        calls: u32 = 0,

        fn complete(_: *anyopaque, _: Allocator, _: Request) lm_mod.Error![]const Completion {
            return error.LMFailure;
        }

        fn name(_: *anyopaque) []const u8 {
            return "cache-model";
        }

        fn canonical(
            ptr: *anyopaque,
            arena: Allocator,
            request: *const sdk.Request,
        ) lm_mod.Error!Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            const body = try std.fmt.allocPrint(arena, "call {d}", .{self.calls});
            var result: Completion = .{
                .text = body,
                .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = body } }} },
                .metadata = .{ .usage = .{ .input_tokens = .{ .integer = 2 } } },
            };
            try lm_mod.publishResponse(arena, &result, request.fields.model, .borrowed);
            return result;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var engine: Engine = .{};
    var cache: CachedLM = .{ .gpa = testing.allocator, .inner = .{
        .ptr = &engine,
        .vtable = &.{
            .complete = Engine.complete,
            .complete_canonical = Engine.canonical,
            .model_name = Engine.name,
        },
    } };
    defer cache.deinit();
    var fields: sdk.RequestFields = .{
        .model = "cache-model",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "same" } }} }},
    };
    const base = try sdk.Request.fromFields(arena, &fields);
    defer base.release();
    fields.config.temperature = 0.5;
    const hot = try sdk.Request.fromFields(arena, &fields);
    defer hot.release();
    const cases = [_]struct { request: *const sdk.Request, cache: bool, text: []const u8, hit: bool }{
        .{ .request = base, .cache = true, .text = "call 1", .hit = false },
        .{ .request = hot, .cache = true, .text = "call 2", .hit = false },
        .{ .request = base, .cache = false, .text = "call 3", .hit = false },
        .{ .request = base, .cache = true, .text = "call 1", .hit = true },
    };
    const history = lm_mod.public_history;
    var global_slots: [history.global_capacity]*history.Entry = undefined;
    var client_slots: [1]*history.Entry = undefined;
    var global = history.Owner.init(&global_slots);
    defer global.clear();
    var owner = history.Owner.init(&client_slots);
    defer owner.clear();
    for (cases) |case| {
        const result = try cache.lm().callCanonicalResult(arena, testing.io, case.request, .{
            .use_cache = case.cache,
            .history = .{ .allocator = testing.allocator, .global = &global, .client = &owner, .owner_limit = 1 },
        });
        defer result.free(arena);
        try testing.expectEqualStrings(case.text, result.outputs.root.array.items[0].string);
        try testing.expectEqual(case.hit, result.cache_hit);
        try testing.expectEqual(case.hit, result.usage.root.object.get("prompt_tokens") == null);
        try testing.expectEqual(@as(i64, 2), result.responses[0].fields.usage.input_tokens.integer);
        const entry = owner.at(0).?;
        try testing.expect(entry.responses[0] == result.responses[0]);
        try testing.expect(entry.usage.root.object == result.usage.root.object);
        try testing.expectEqual(case.hit, entry.usage.root.object.get("prompt_tokens") == null);
    }
    const prompt_calls = [_]struct { count: u32, first: []const u8, hit: bool }{
        .{ .count = 2, .first = "call 4", .hit = false },
        .{ .count = 1, .first = "call 1", .hit = true },
        .{ .count = 3, .first = "call 6", .hit = false },
        .{ .count = 2, .first = "call 4", .hit = true },
    };
    for (prompt_calls) |case| {
        const result = try cache.lm().callPromptResult(arena, testing.io, "same", .{ .n = case.count });
        defer result.free(arena);
        try testing.expectEqual(case.hit, result.cache_hit);
        try testing.expectEqual(case.count, result.responses.len);
        try testing.expectEqualStrings(case.first, result.outputs.root.array.items[0].string);
    }
    try testing.expectEqual(@as(u32, 8), engine.calls);
    owner.clear();
    try testing.expectEqualStrings("call 1", global.at(3).?.outputs.root.array.items[0].string);
}

test "CachedLM survives every allocation failure" {
    const messages = [_]adapter.Message{.{ .role = .user, .content = .{ .text = "hi" } }};
    // In-place arena growth makes fault sites address-dependent on some allocators.
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), struct {
        fn run(gpa: Allocator, msgs: []const adapter.Message) !void {
            var arena_state: std.heap.ArenaAllocator = .init(gpa);
            defer arena_state.deinit();
            var scripted: lm_mod.ScriptedLM = .{ .responses = &.{ "x", "y" } };
            var cache: CachedLM = .{ .gpa = gpa, .inner = scripted.lm() };
            defer cache.deinit();
            _ = try cache.lm().complete(arena_state.allocator(), .{ .messages = msgs });
            _ = try cache.lm().complete(arena_state.allocator(), .{ .messages = msgs });
            try cache.load("{\"0000000000000000000000000000000000000000000000000000000000000000\": {\"completions\": [{\"text\": \"z\", \"reasoning\": \"native\", \"tool_calls\": []}], \"record\": null}}");
        }
    }.run, .{&messages});
}
