//! `CachedLM`: DSPy's response cache (`dspy.cache`) as an LM wrapper. A
//! request is keyed by the SHA-256 of its messages and options (temperature,
//! `n`, `rollout_id`, JSON mode), so like DSPy a different `rollout_id`
//! bypasses earlier entries. Entries live in memory, least recently used
//! evicted first; `dump`/`load` persist them as JSON. Failed calls are not
//! cached. DSPy's on-disk `diskcache` format and key hashing are not shared.

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

pub const Key = [64]u8;

pub const CachedLM = struct {
    gpa: Allocator,
    inner: LM,
    /// `memory_max_entries`.
    max_entries: usize = 1_000_000,
    entries: std.AutoArrayHashMapUnmanaged(Key, []const Completion) = .empty,
    hits: usize = 0,
    misses: usize = 0,

    pub fn deinit(self: *CachedLM) void {
        for (self.entries.values()) |completions| self.freeCompletions(completions);
        self.entries.deinit(self.gpa);
        self.* = undefined;
    }

    pub fn lm(self: *CachedLM) LM {
        return .{
            .ptr = self,
            .vtable = &.{ .complete = complete },
            .supports_reasoning = self.inner.supports_reasoning,
            .supports_response_format = self.inner.supports_response_format,
            .supports_response_schema = self.inner.supports_response_schema,
            .supports_function_calling = self.inner.supports_function_calling,
        };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: Request) lm_mod.Error![]const Completion {
        const self: *CachedLM = @ptrCast(@alignCast(ptr));
        const k = try key(arena, request);
        if (self.entries.fetchOrderedRemove(k)) |hit| {
            // Re-insert at the end: most recently used.
            self.entries.putAssumeCapacity(k, hit.value);
            self.hits += 1;
            return dupeCompletions(arena, hit.value);
        }
        self.misses += 1;
        const completions = try self.inner.completeResponses(arena, request);
        try self.insert(k, completions);
        return completions;
    }

    fn insert(self: *CachedLM, k: Key, completions: []const Completion) Allocator.Error!void {
        assert(self.max_entries > 0);
        const owned = try dupeCompletions(self.gpa, completions);
        errdefer self.freeCompletions(owned);
        try self.entries.ensureUnusedCapacity(self.gpa, 1);
        if (self.entries.count() >= self.max_entries) {
            self.freeCompletions(self.entries.values()[0]);
            self.entries.orderedRemoveAt(0);
        }
        self.entries.putAssumeCapacity(k, owned);
    }

    fn freeCompletions(self: *CachedLM, completions: []const Completion) void {
        for (completions) |c| lm_mod.freeCompletion(self.gpa, c);
        self.gpa.free(completions);
    }

    /// `{"<key>": [{"text": "...", "reasoning": null}, ...]}`, LRU first.
    pub fn dump(self: *const CachedLM, w: *Writer) pyjson.Error!void {
        try w.writeByte('{');
        for (self.entries.keys(), self.entries.values(), 0..) |k, completions, i| {
            if (i > 0) try w.writeAll(", ");
            try pyjson.writeJsonString(w, &k);
            try w.writeAll(": [");
            for (completions, 0..) |c, j| {
                if (j > 0) try w.writeAll(", ");
                try std.json.Stringify.value(c, .{}, w);
            }
            try w.writeByte(']');
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
            if (entry.key_ptr.len != 64 or entry.value_ptr.* != .array) return error.InvalidCache;
            const items = entry.value_ptr.array.items;
            if (items.len == 0) return error.InvalidCache;
            const completions = try scratch.allocator().alloc(Completion, items.len);
            for (items, completions) |item, *c| {
                c.* = lm_mod.completionFromValue(scratch.allocator(), item) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidCache,
                };
            }
            if (self.entries.contains(entry.key_ptr.*[0..64].*)) continue;
            try self.insert(entry.key_ptr.*[0..64].*, completions);
        }
    }
};

fn dupeCompletions(a: Allocator, completions: []const Completion) Allocator.Error![]const Completion {
    const out = try a.alloc(Completion, completions.len);
    var done: usize = 0;
    errdefer {
        for (out[0..done]) |c| lm_mod.freeCompletion(a, c);
        a.free(out);
    }
    for (completions, out) |c, *o| {
        o.* = try lm_mod.cloneCompletion(a, c);
        done += 1;
    }
    return out;
}

/// Hex SHA-256 of the request as canonical JSON.
pub fn key(arena: Allocator, request: Request) Allocator.Error!Key {
    var aw: Writer.Allocating = .init(arena);
    writeCanonical(&aw.writer, request) catch return error.OutOfMemory;
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(aw.written(), &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

fn writeCanonical(w: *Writer, request: Request) Writer.Error!void {
    // Stream every typed field, including native tools and structured schemas.
    // Cache hashes are local to DSZig and are not DSPy's diskcache format.
    try std.json.Stringify.value(request, .{}, w);
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
    try testing.expectEqual(@as(usize, 4), cache.hits);
    try testing.expectEqual(@as(usize, 5), cache.misses);
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
            try cache.load("{\"0000000000000000000000000000000000000000000000000000000000000000\": [{\"text\": \"z\", \"reasoning\": \"native\", \"tool_calls\": []}]}");
        }
    }.run, .{&messages});
}
