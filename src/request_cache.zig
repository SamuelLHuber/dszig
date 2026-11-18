//! Native counterpart of the in-memory layer of DSPy's `Cache`: an LRU map
//! from request key to response graph. `put` stores the caller's payload
//! itself (no copy), so later caller mutations of nested containers stay
//! visible to future hits, as with upstream `memory_cache[key] = value`.
//! `get` returns a fresh deep copy (`copy.deepcopy`) that the caller owns.
//! Payloads are reference counted: results keep their payloads alive after
//! replacement, eviction or cache destruction. DSPy's disk cache, pickle
//! format and key hashing are not provided.
//!
//! The mutex serializes the entry map only. Payload graphs are unsynchronized:
//! callers must not mutate a cached graph while another thread reads it.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;

const json_graph = @import("json_graph.zig");

pub const Key = [32]u8;

/// Heap-stable mutable graph. Its arena owns new nodes; immutable dependencies
/// retain any shallow views into older payloads.
pub const Payload = struct {
    gpa: Allocator,
    arena_state: std.heap.ArenaAllocator,
    references: std.atomic.Value(usize),
    root: json_graph.Value = .null,
    // Immutable older owners make dependency cycles impossible. The link is
    // used only after the last reference, for allocation-free iterative release.
    dependencies: []const *Payload = &.{},
    release_next: ?*Payload = null,

    /// One reference, owned by the caller.
    pub fn create(gpa: Allocator) Allocator.Error!*Payload {
        const payload = try gpa.create(Payload);
        payload.* = .{ .gpa = gpa, .arena_state = .init(gpa), .references = .init(1) };
        return payload;
    }

    /// A new owner of shallow graph views into existing, independently owned
    /// payloads. Adopts the gpa-allocated parent slice on success only; parents
    /// themselves remain borrowed throughout construction.
    pub fn createDependent(gpa: Allocator, parents: []const *Payload) Allocator.Error!*Payload {
        const payload = try create(gpa);
        payload.attachDependencies(parents);
        return payload;
    }

    /// Finish an unpublished payload whose dependency set depends on its
    /// projected leaves. Adopts the gpa-allocated slice without allocation.
    pub fn attachDependencies(self: *Payload, parents: []const *Payload) void {
        assert(self.references.load(.monotonic) == 1);
        assert(self.dependencies.len == 0);
        for (parents) |parent| {
            assert(parent != self);
            parent.retain();
        }
        self.dependencies = parents;
    }

    /// Allocator for nodes of `root`; freed by the final `release`.
    pub fn allocator(self: *Payload) Allocator {
        return self.arena_state.allocator();
    }

    pub fn retain(self: *Payload) void {
        const previous = self.references.fetchAdd(1, .monotonic);
        assert(previous > 0);
    }

    pub fn release(self: *Payload) void {
        const previous = self.references.fetchSub(1, .acq_rel);
        assert(previous > 0);
        if (previous != 1) return;
        var pending: ?*Payload = self;
        while (pending) |current| {
            pending = current.release_next;
            for (current.dependencies) |parent| {
                assert(parent != current);
                const count = parent.references.fetchSub(1, .acq_rel);
                assert(count > 0);
                if (count == 1) {
                    parent.release_next = pending;
                    pending = parent;
                }
            }
            const gpa = current.gpa;
            gpa.free(current.dependencies);
            current.arena_state.deinit();
            gpa.destroy(current);
        }
    }

    /// New payload (one reference) holding a deep copy of `root` that keeps
    /// shared containers and cycles.
    pub fn clone(self: *Payload, gpa: Allocator) Allocator.Error!*Payload {
        const out = try create(gpa);
        errdefer out.release();
        out.root = try json_graph.cloneUsing(out.allocator(), gpa, self.root);
        return out;
    }
};

test "shallow payload ownership releases deep dependency chains without recursion" {
    const allocator = std.testing.allocator;
    var current = try Payload.create(allocator);
    errdefer current.release();
    current.root = .{ .integer = 7 };
    for (0..10000) |_| {
        const parents = try allocator.alloc(*Payload, 1);
        parents[0] = current;
        const next = Payload.createDependent(allocator, parents) catch |err| {
            allocator.free(parents);
            return err;
        };
        next.root = current.root;
        current.release();
        current = next;
    }
    try std.testing.expectEqual(@as(i64, 7), current.root.integer);
    current.release();
}

pub const Cache = struct {
    gpa: Allocator,
    io: Io,
    /// `enable_memory_cache`: when false, `get` misses and `put` stores nothing.
    enabled: bool = true,
    /// `memory_max_entries`; must be positive when storing.
    max_entries: usize = 1_000_000,
    mutex: Io.Mutex = .init,
    /// Least recently used first; each value holds one payload reference.
    entries: std.AutoArrayHashMapUnmanaged(Key, *Payload) = .empty,

    pub const GetError = Allocator.Error || Io.Cancelable;
    pub const PutError = Allocator.Error || Io.Cancelable || error{InvalidMaxEntries};

    /// Releases the cache's references; outstanding payloads stay valid.
    pub fn deinit(self: *Cache) void {
        for (self.entries.values()) |payload| payload.release();
        self.entries.deinit(self.gpa);
        self.* = undefined;
    }

    /// Owned deep copy of the cached payload (caller releases), or null when
    /// disabled, absent, or cached with a null root. Any hit, including a null
    /// root, becomes most recently used.
    pub fn get(self: *Cache, key: Key) GetError!?*Payload {
        const payload = hit: {
            try self.mutex.lock(self.io);
            defer self.mutex.unlock(self.io);
            if (!self.enabled) return null;
            const entry = self.entries.fetchOrderedRemove(key) orelse return null;
            self.entries.putAssumeCapacity(key, entry.value);
            if (entry.value.root == .null) return null;
            entry.value.retain();
            break :hit entry.value;
        };
        defer payload.release();
        return try payload.clone(self.gpa);
    }

    /// Store `payload` itself (adding a reference; the caller keeps its own).
    /// Replacing a key makes it most recently used; a full cache evicts the
    /// least recently used entries first.
    pub fn put(self: *Cache, key: Key, payload: *Payload) PutError!void {
        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.enabled) return;
        if (self.max_entries == 0) return error.InvalidMaxEntries;
        const existing = self.entries.getIndex(key);
        if (existing == null and self.entries.count() < self.max_entries)
            try self.entries.ensureUnusedCapacity(self.gpa, 1);
        payload.retain();
        if (existing) |index| {
            self.entries.values()[index].release();
            self.entries.orderedRemoveAt(index);
        }
        while (self.entries.count() >= self.max_entries) {
            self.entries.values()[0].release();
            self.entries.orderedRemoveAt(0);
        }
        self.entries.putAssumeCapacity(key, payload);
    }
};

const testing = std.testing;

fn testPayload(gpa: Allocator, json: []const u8) !*Payload {
    const payload = try Payload.create(gpa);
    errdefer payload.release();
    payload.root = try json_graph.parse(payload.allocator(), json);
    return payload;
}

fn testKey(byte: u8) Key {
    return @splat(byte);
}

fn testTags(root: json_graph.Value) *json_graph.Array {
    return root.array.items[0].object.get("meta").?.object.get("tags").?.array;
}

test "raw put keeps caller mutations visible while hits are isolated copies" {
    var cache: Cache = .{ .gpa = testing.allocator, .io = testing.io };
    defer cache.deinit();
    const response = try testPayload(testing.allocator, "[{\"long_text\":\"server\",\"meta\":{\"tags\":[\"a\"]}}]");
    defer response.release();
    try cache.put(testKey(1), response);

    // Appending to the stored nested list after `put` reaches later hits.
    for (0..32) |_| try testTags(response.root).append(.{ .string = "b" });

    const hit = (try cache.get(testKey(1))).?;
    defer hit.release();
    try testing.expectEqual(@as(usize, 33), testTags(hit.root).items.len);
    try testing.expectEqualStrings("server", hit.root.array.items[0].object.get("long_text").?.string);
    try testTags(hit.root).append(.{ .string = "hit" });
    try hit.root.array.items[0].object.put(hit.allocator(), "long_text", .{ .string = "changed" });

    const again = (try cache.get(testKey(1))).?;
    defer again.release();
    try testing.expectEqual(@as(usize, 33), testTags(again.root).items.len);
    try testing.expectEqualStrings("server", again.root.array.items[0].object.get("long_text").?.string);
    try testing.expectEqual(@as(usize, 33), testTags(response.root).items.len);
}

test "leases outlive replacement, eviction and cache destruction" {
    var cache: Cache = .{ .gpa = testing.allocator, .io = testing.io, .max_entries = 1 };
    const network = try testPayload(testing.allocator, "{\"v\":\"net\"}");
    defer network.release();

    try cache.put(testKey(1), network);
    try cache.put(testKey(2), network); // evicts key 1
    try testing.expectEqual(@as(?*Payload, null), try cache.get(testKey(1)));

    const replacement = try testPayload(testing.allocator, "{\"v\":\"new\"}");
    try cache.put(testKey(2), replacement);
    replacement.release(); // cache is now the only owner
    try testing.expectEqualStrings("net", network.root.object.get("v").?.string);

    const hit = (try cache.get(testKey(2))).?;
    defer hit.release();
    cache.deinit();
    try testing.expectEqualStrings("new", hit.root.object.get("v").?.string);
    try testing.expectEqualStrings("net", network.root.object.get("v").?.string);
}

test "LRU order follows null-root hits and replacements" {
    var cache: Cache = .{ .gpa = testing.allocator, .io = testing.io, .max_entries = 2 };
    defer cache.deinit();
    const empty = try Payload.create(testing.allocator);
    defer empty.release();
    const value = try testPayload(testing.allocator, "\"x\"");
    defer value.release();

    try cache.put(testKey(1), empty);
    try cache.put(testKey(2), value);
    try testing.expectEqual(@as(?*Payload, null), try cache.get(testKey(1))); // touches key 1
    try cache.put(testKey(3), value); // evicts key 2
    try testing.expectEqual(@as(?*Payload, null), try cache.get(testKey(2)));
    try testing.expect(cache.entries.contains(testKey(1)));

    try cache.put(testKey(1), value); // replacement touches key 1
    try cache.put(testKey(4), value); // evicts key 3
    try testing.expectEqual(@as(?*Payload, null), try cache.get(testKey(3)));
    const hit = (try cache.get(testKey(1))).?;
    defer hit.release();
    try testing.expectEqualStrings("x", hit.root.string);
}

test "disabled cache stores nothing and zero capacity is rejected" {
    const value = try testPayload(testing.allocator, "[1]");
    defer value.release();

    var disabled: Cache = .{ .gpa = testing.allocator, .io = testing.io, .enabled = false, .max_entries = 0 };
    defer disabled.deinit();
    try disabled.put(testKey(1), value);
    try testing.expectEqual(@as(usize, 0), disabled.entries.count());
    try testing.expectEqual(@as(?*Payload, null), try disabled.get(testKey(1)));

    var zero: Cache = .{ .gpa = testing.allocator, .io = testing.io, .max_entries = 0 };
    defer zero.deinit();
    try testing.expectError(error.InvalidMaxEntries, zero.put(testKey(1), value));
    try testing.expectEqual(@as(usize, 0), zero.entries.count());
}

test "cached graph copies keep sharing and cycles under every allocation failure" {
    // In-place arena growth makes fault sites address-dependent on some allocators.
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), struct {
        fn run(gpa: Allocator) !void {
            var cache: Cache = .{ .gpa = gpa, .io = testing.io, .max_entries = 1 };
            defer cache.deinit();
            const source = try testPayload(gpa, "{\"shared\":[1,12345678901234567890],\"text\":\"x\"}");
            defer source.release();
            const record = source.root.object;
            try record.put(source.allocator(), "again", record.get("shared").?);
            try record.put(source.allocator(), "self", source.root);
            try cache.put(testKey(1), source);
            try cache.put(testKey(2), source);

            const hit = (try cache.get(testKey(2))) orelse return error.TestUnexpectedResult;
            defer hit.release();
            const copied = hit.root.object;
            try testing.expect(copied != record);
            try testing.expect(copied.get("shared").?.array == copied.get("again").?.array);
            try testing.expect(copied.get("shared").?.array != record.get("shared").?.array);
            try testing.expect(copied.get("self").?.object == copied);
            try testing.expectEqualStrings("12345678901234567890", copied.get("shared").?.array.items[1].number_string);
            try testing.expectEqualStrings("x", copied.get("text").?.string);
        }
    }.run, .{});
}
