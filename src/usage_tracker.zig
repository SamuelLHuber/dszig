//! DSPy usage entries, not SDK counter validation. Root entries are shallow
//! snapshots; ordinary nested graph values remain borrowed and mutable.
//! Related trackers share Storage, including its allocator lock and lifetime.
//! Model serializers must not reenter this storage. Input graph mutation still
//! requires external synchronization. All storage may use a startup buffer.
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");
const python = @import("python_value.zig");
const numeric = @import("value.zig");
pub const Graph = @import("json_graph.zig");
const Value = Graph.Value;
const Key = runtime.Value;
pub const Error = Graph.ConvertError || error{CapacityExceeded};
const absent = std.math.maxInt(u32);
const work_max = 1 << 20;
const depth_max = 1000;

/// Share this between nested scopes so their allocator and entry snapshots
/// are serialized together. It must outlive every tracker and borrowed result.
pub const Storage = struct {
    arena: Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
};

pub const UsageTracker = struct {
    storage: *Storage,
    models: []Model,
    entries: []Entry,
    model_indices: std.StringHashMapUnmanaged(u32) = .empty,
    model_count: u32 = 0,
    entry_count: u32 = 0,

    pub const Limits = struct { models: u32, entries: u32 };
    pub const Model = struct { name: []const u8, first: u32 = absent, last: u32 = absent };
    pub const Entry = struct { data: Value, next: u32 = absent };

    /// Initialize in place. Capacities are exact and never grow. Entry graphs
    /// and names live in storage.arena; resetting it invalidates all trackers.
    pub fn init(self: *UsageTracker, storage: *Storage, limits: Limits) Error!void {
        if (limits.models > work_max or limits.entries > work_max) return error.CapacityExceeded;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        try self.initLocked(storage, limits);
    }

    fn initLocked(self: *UsageTracker, storage: *Storage, limits: Limits) Error!void {
        assert(limits.models <= work_max);
        assert(limits.entries <= work_max);
        self.* = .{
            .storage = storage,
            .models = try storage.arena.alloc(Model, limits.models),
            .entries = try storage.arena.alloc(Entry, limits.entries),
        };
        try self.model_indices.ensureTotalCapacity(storage.arena, limits.models);
        assert(self.model_count == 0);
        assert(self.entry_count == 0);
    }

    /// Arena-owned instance for automatic Module scopes. Unlike an in-place
    /// caller-owned header, its identity remains valid until storage reset.
    pub fn create(storage: *Storage, limits: Limits) Error!*UsageTracker {
        if (limits.models > work_max or limits.entries > work_max) return error.CapacityExceeded;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        return createLocked(storage, limits);
    }

    fn createLocked(storage: *Storage, limits: Limits) Error!*UsageTracker {
        const result = try storage.arena.create(UsageTracker);
        try result.initLocked(storage, limits);
        return result;
    }

    /// Snapshot an inherited tracker for one parallel task, never a worker's
    /// cumulative state. Entries are independent, but aliases within the
    /// snapshot survive across models and entries. Immutable names and the
    /// allocator lock retain the shared storage lifetime. No automatic rollup
    /// reconnects this fork to its original. Its header remains address-stable
    /// until storage reset, including when a task retains it for observation.
    pub fn fork(self: *UsageTracker) Error!*UsageTracker {
        const storage = self.storage;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        assert(self.model_count <= self.models.len);
        assert(self.entry_count <= self.entries.len);
        const result = try createLocked(storage, .{
            .models = @intCast(self.models.len),
            .entries = @intCast(self.entries.len),
        });
        @memcpy(result.models[0..self.model_count], self.models[0..self.model_count]);
        for (result.models[0..self.model_count], 0..) |model_data, index|
            result.model_indices.putAssumeCapacity(model_data.name, @intCast(index));
        var custom_cloner: runtime.CloneContext = .init(storage.arena);
        defer custom_cloner.deinit();
        var cloner = Graph.CloneContext.init(storage.arena, storage.arena, &custom_cloner);
        defer cloner.deinit();
        for (self.entries[0..self.entry_count], result.entries[0..self.entry_count]) |entry, *target| {
            target.* = .{
                .data = try cloner.clone(entry.data),
                .next = entry.next,
            };
        }
        result.model_count = self.model_count;
        result.entry_count = self.entry_count;
        assert(result.model_indices.count() == result.model_count);
        assert(result.storage == self.storage);
        return result;
    }

    /// Only direct genuine BaseModel values receive default Python-mode
    /// model_dump. Other nested objects are not recursively flattened here.
    /// Like Source, a nonempty invalid entry can leave an empty model bucket.
    pub fn addUsage(self: *UsageTracker, name: []const u8, input: Value) Error!void {
        const storage = self.storage;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        try self.addUsageLocked(name, input);
    }

    /// Retain an independent snapshot of ephemeral engine usage. Unlike the
    /// explicit shallow collector API, no call-arena storage may escape here.
    pub fn addOwnedUsage(self: *UsageTracker, name: []const u8, input: Value) Error!void {
        const storage = self.storage;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        if (try length(input) == 0) return;
        const model_index = try self.model(name);
        if (self.entry_count == self.entries.len) return error.CapacityExceeded;
        if (!isDictionary(input)) return error.ParseValue;
        if (try dictionaryLength(input) > work_max) return error.ParseValue;
        const owned = try Graph.clone(storage.arena, input);
        switch (owned) {
            .object => |record| {
                for (record.values()) |*item| item.* = try flattenModel(storage.arena, item.*);
            },
            .dictionary => |dictionary| {
                for (dictionary.entries.items) |*entry|
                    entry.value = try flattenModel(storage.arena, entry.value);
            },
            else => unreachable,
        }
        self.append(model_index, owned);
        assert(self.entry_count <= self.entries.len);
    }

    fn addUsageLocked(self: *UsageTracker, name: []const u8, input: Value) Error!void {
        const storage = self.storage;
        if (try length(input) == 0) return;
        const model_index = try self.model(name);
        if (self.entry_count == self.entries.len) return error.CapacityExceeded;
        self.append(model_index, try flattenEntry(storage.arena, input));
        assert(self.entry_count <= self.entries.len);
    }

    /// Recompute in entry order. The result owns fresh dictionary headers in
    /// arena but borrows ordinary nested containers/leaves from recorded data.
    /// A result is not a deep snapshot; use Graph.clone when that is required.
    pub fn getTotalTokens(self: *UsageTracker, arena: Allocator) Error!Value {
        const storage = self.storage;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        const result = try arena.create(Graph.Record);
        result.* = .empty;
        var work: Work = .{};
        for (self.models[0..self.model_count]) |model_data| {
            try work.tick();
            var total: Value = .null;
            var current = model_data.first;
            var count: u32 = 0;
            while (current != absent) : (count += 1) {
                assert(count < self.entry_count);
                const entry = self.entries[current];
                total = try mergeUsing(arena, &work, total, entry.data);
                current = entry.next;
            }
            if (total == .null) total = .{ .dictionary = (try Indexed.init(arena, 0)).data };
            try result.put(arena, model_data.name, total);
        }
        assert(result.count() == self.model_count);
        return .{ .object = result };
    }

    /// Copy entry headers ONCE, grouped by model, not a live child list. Both
    /// trackers must share storage: deep-copying here would destroy aliases.
    /// Capacity refusal occurs before publication, never drops recorded usage.
    pub fn rollUp(self: *UsageTracker, parent: *UsageTracker) Error!void {
        assert(self != parent);
        assert(self.storage == parent.storage);
        const storage = self.storage;
        storage.mutex.lockUncancelable(storage.io);
        defer storage.mutex.unlock(storage.io);
        if (self.entry_count > parent.entries.len - parent.entry_count) return error.CapacityExceeded;
        var new_models: u32 = 0;
        for (self.models[0..self.model_count]) |model_data| {
            if (parent.findModel(model_data.name) == null) new_models += 1;
        }
        if (new_models > parent.models.len - parent.model_count) return error.CapacityExceeded;
        // Reserve model names before appending any entry. Their bytes already
        // belong to the shared lifetime; rollup needs no additional allocation.
        for (self.models[0..self.model_count]) |model_data| {
            const target = parent.findModel(model_data.name) orelse blk: {
                const next = parent.model_count;
                parent.models[next] = .{ .name = model_data.name };
                parent.model_indices.putAssumeCapacity(model_data.name, next);
                parent.model_count += 1;
                break :blk next;
            };
            var current = model_data.first;
            var count: u32 = 0;
            while (current != absent) : (count += 1) {
                assert(count < self.entry_count);
                parent.append(target, self.entries[current].data);
                current = self.entries[current].next;
            }
        }
        assert(parent.entry_count <= parent.entries.len);
    }

    fn flattenModel(arena: Allocator, item: Value) Error!Value {
        if (item == .custom and item.custom.vtable.model_dump != null)
            return Graph.fromValue(arena, try item.custom.modelDump(arena));
        return item;
    }

    fn findModel(self: *const UsageTracker, name: []const u8) ?u32 {
        assert(self.model_count <= self.models.len);
        assert(self.model_indices.count() == self.model_count);
        return self.model_indices.get(name);
    }

    fn model(self: *UsageTracker, name: []const u8) Error!u32 {
        if (self.findModel(name)) |index| return index;
        if (self.model_count == self.models.len) return error.CapacityExceeded;
        const index = self.model_count;
        self.models[index] = .{ .name = try self.storage.arena.dupe(u8, name) };
        self.model_indices.putAssumeCapacity(self.models[index].name, index);
        self.model_count += 1;
        assert(self.model_count <= self.models.len);
        return index;
    }

    fn append(self: *UsageTracker, model_index: u32, data: Value) void {
        assert(model_index < self.model_count);
        assert(self.entry_count < self.entries.len);
        const index = self.entry_count;
        self.entries[index] = .{ .data = data };
        const model_data = &self.models[model_index];
        if (model_data.last == absent) model_data.first = index else self.entries[model_data.last].next = index;
        model_data.last = index;
        self.entry_count += 1;
    }
};

/// Explicit native settings slot, local to its execution context. Always call
/// finish, including on a body error; it restores the slot even if rollup fails.
/// The scope is immovable after init. Nested contexts borrow the tracker, not
/// this mutable slot; parallel Module bindings fork it separately per task.
/// Child additions after finish no longer reach parent.
pub const Scope = struct {
    tracker: UsageTracker,
    active: *?*UsageTracker,
    parent: ?*UsageTracker,
    finished: bool,

    pub fn init(self: *Scope, storage: *Storage, active: *?*UsageTracker, limits: UsageTracker.Limits) Error!void {
        if (active.*) |parent| assert(parent.storage == storage);
        try self.tracker.init(storage, limits);
        self.active = active;
        self.parent = active.*;
        self.finished = false;
        active.* = &self.tracker;
        assert(active.* == &self.tracker);
    }

    /// Execute a native body and roll up on both success and error. As in a
    /// Python finally, a rollup failure takes precedence over the body's error.
    pub fn run(self: *Scope, comptime body: anytype, args: anytype) RunResult(body, @TypeOf(args)) {
        assert(!self.finished);
        assert(self.active.* == &self.tracker);
        const result = @call(.auto, body, args);
        try self.finish();
        return result;
    }

    fn RunResult(comptime body: anytype, comptime Args: type) type {
        const Result = @TypeOf(@call(.auto, body, @as(Args, undefined)));
        return switch (@typeInfo(Result)) {
            .error_union => |info| (Error || info.error_set)!info.payload,
            else => Error!Result,
        };
    }

    pub fn finish(self: *Scope) Error!void {
        assert(!self.finished);
        assert(self.active.* == &self.tracker);
        defer {
            self.active.* = self.parent;
            self.finished = true;
        }
        if (self.parent) |parent| try self.tracker.rollUp(parent);
    }
};

const Work = struct {
    remaining: u32 = work_max,
    fn tick(self: *Work) Error!void {
        if (self.remaining == 0) return error.ParseValue;
        self.remaining -= 1;
    }
};

/// Ordered dictionary plus a fallible Python-key index. Hash buckets avoid
/// repeatedly comparing every unrelated key; collision walks share the budget.
const Indexed = struct {
    data: *Graph.Dictionary,
    heads: std.AutoHashMapUnmanaged(i64, u32) = .empty,
    next: []u32,

    fn init(arena: Allocator, capacity: u32) Error!Indexed {
        if (capacity > work_max) return error.ParseValue;
        const data = try arena.create(Graph.Dictionary);
        data.* = .{ .entries = try .initCapacity(arena, capacity) };
        var result: Indexed = .{ .data = data, .next = try arena.alloc(u32, capacity) };
        try result.heads.ensureTotalCapacity(arena, capacity);
        assert(result.data.entries.items.len == 0);
        return result;
    }

    fn find(self: *const Indexed, work: *Work, key: Key, hash: i64) Error!?u32 {
        var index = self.heads.get(hash) orelse return null;
        var count: u32 = 0;
        while (index != absent) : (count += 1) {
            try work.tick();
            assert(count < self.data.entries.items.len);
            if (try python.containerEqual(self.data.entries.items[index].key, key)) return index;
            index = self.next[index];
        }
        return null;
    }

    fn put(self: *Indexed, arena: Allocator, work: *Work, key: Key, item: Value) Error!u32 {
        const hash = try python.pythonHash(key, 0);
        if (try self.find(work, key, hash)) |index| {
            self.data.entries.items[index].value = item;
            return index;
        }
        const index: u32 = @intCast(self.data.entries.items.len);
        assert(index < self.next.len);
        self.next[index] = self.heads.get(hash) orelse absent;
        self.data.entries.appendAssumeCapacity(.{ .key = key, .value = item });
        self.heads.putAssumeCapacity(hash, index);
        _ = arena;
        return index;
    }
};

const Frame = struct {
    result: Indexed,
    right: Value,
    next: u32 = 0,
    parent_slot: u32 = absent,
};

/// Source add_usage preprocessing without collector state. Empty sized inputs
/// are ignored; nonempty entries require .items(), and only direct BaseModel
/// values are flattened. Ordinary nested graphs remain shallow aliases.
pub fn prepareEntry(arena: Allocator, input: Value) Error!?Value {
    if (try length(input) == 0) return null;
    return try flattenEntry(arena, input);
}

fn flattenEntry(arena: Allocator, input: Value) Error!Value {
    if (!isDictionary(input)) return error.ParseValue;
    const size = try dictionaryLength(input);
    if (size > work_max) return error.ParseValue;
    const data = try arena.create(Graph.Dictionary);
    data.* = .{ .entries = try .initCapacity(arena, size) };
    var index: u32 = 0;
    while (try dictionaryEntry(input, index)) |entry| : (index += 1) {
        data.entries.appendAssumeCapacity(.{
            .key = entry.key,
            .value = try UsageTracker.flattenModel(arena, entry.value),
        });
    }
    return .{ .dictionary = data };
}

/// Source merge ordering matters: dictionary recursion precedes numeric/None
/// handling, and len(left) is evaluated before len(right). This accepts iterable
/// pairs only in dict(...) positions, never in the right .items() position.
/// The fixed stack and shared work budget also bound cyclic/adversarial graphs.
pub fn merge(arena: Allocator, left: Value, right: Value) Error!Value {
    var work: Work = .{};
    return mergeUsing(arena, &work, left, right);
}

fn mergeUsing(arena: Allocator, work: *Work, left: Value, right: Value) Error!Value {
    var frames: [depth_max + 1]Frame = undefined;
    var count: u32 = 1;
    frames[0] = try enter(arena, work, left, right);
    while (count > 0) {
        try work.tick();
        const frame = &frames[count - 1];
        const incoming = (try dictionaryEntry(frame.right, frame.next)) orelse {
            const result: Value = .{ .dictionary = frame.result.data };
            const slot = frame.parent_slot;
            count -= 1;
            if (count == 0) return result;
            frames[count - 1].result.data.entries.items[slot].value = result;
            continue;
        };
        frame.next += 1;
        const hash = try python.pythonHash(incoming.key, 0);
        const found = try frame.result.find(work, incoming.key, hash);
        const current: Value = if (found) |index| frame.result.data.entries.items[index].value else .null;
        if (isDictionary(current) or isDictionary(incoming.value)) {
            if (count == frames.len) return error.NestingTooDeep;
            var child = try enter(arena, work, current, incoming.value);
            child.parent_slot = found orelse try frame.result.put(arena, work, incoming.key, .null);
            frames[count] = child;
            count += 1;
        } else if (current == .null) {
            if (incoming.value != .null) _ = try frame.result.put(arena, work, incoming.key, incoming.value);
        } else if (summable(current) and summable(incoming.value)) {
            const sum = numeric.numericSum(arena, try scalar(current), try scalar(incoming.value)) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.ParseValue,
            };
            frame.result.data.entries.items[found.?].value = try Graph.fromValue(arena, sum);
        }
    }
    unreachable;
}

fn enter(arena: Allocator, work: *Work, left: Value, right: Value) Error!Frame {
    try work.tick();
    if (left == .null or try length(left) == 0) {
        return .{ .result = try copyDictionary(arena, work, if (try Graph.truthy(right)) right else .null, 0), .right = .null };
    }
    if (right == .null or try length(right) == 0) {
        return .{ .result = try copyDictionary(arena, work, if (try Graph.truthy(left)) left else .null, 0), .right = .null };
    }
    const result = try copyDictionary(arena, work, left, try dictionaryLength(right));
    return .{ .result = result, .right = right };
}

fn copyDictionary(arena: Allocator, work: *Work, input: Value, extra: u32) Error!Indexed {
    const size = if (input == .null) 0 else try length(input);
    if (size > work_max -| extra) return error.ParseValue;
    var result = try Indexed.init(arena, size + extra);
    if (input == .null) return result;
    if (isDictionary(input)) {
        var index: u32 = 0;
        while (try dictionaryEntry(input, index)) |entry| : (index += 1) {
            try work.tick();
            _ = try result.put(arena, work, entry.key, entry.value);
        }
    } else {
        var iterator: Iterator = .{ .value = input };
        while (try iterator.next(arena)) |item| {
            try work.tick();
            var pair: Iterator = .{ .value = item };
            const key = (try pair.next(arena)) orelse return error.ParseValue;
            const value = (try pair.next(arena)) orelse return error.ParseValue;
            if (try pair.next(arena) != null) return error.ParseValue;
            _ = try result.put(arena, work, try Graph.toValue(arena, key), value);
        }
    }
    return result;
}

fn isDictionary(input: Value) bool {
    return input == .object or input == .dictionary;
}

fn dictionaryLength(input: Value) Error!u32 {
    return switch (input) {
        .null => 0,
        .object => |items| @intCast(items.count()),
        .dictionary => |items| std.math.cast(u32, items.entries.items.len) orelse error.ParseValue,
        else => error.ParseValue,
    };
}

fn dictionaryEntry(input: Value, index: u32) Error!?Graph.Dictionary.Entry {
    if (index >= try dictionaryLength(input)) return null;
    return switch (input) {
        .object => |items| .{ .key = .{ .string = items.keys()[index] }, .value = items.values()[index] },
        .dictionary => |items| items.entries.items[index],
        else => unreachable,
    };
}

fn length(input: Value) Error!u32 {
    const size: usize = switch (input) {
        .object => |items| items.count(),
        .dictionary => |items| items.entries.items.len,
        .array => |items| items.items.len,
        .sequence => |items| items.items.items.len,
        .string => |text| text.len, // Only emptiness/capacity uses this upper bound.
        .custom => switch (python.view(.{ .custom = input.custom }) orelse return error.ParseValue) {
            .bytes => |bytes| bytes.len,
            .bytearray => |bytes| bytes.len,
            else => return error.ParseValue,
        },
        else => return error.ParseValue,
    };
    return std.math.cast(u32, size) orelse error.ParseValue;
}

fn scalar(input: Value) Error!Key {
    return switch (input) {
        .integer => |item| .{ .integer = item },
        .number_string => |item| .{ .number_string = item },
        .float => |item| .{ .float = item },
        .custom => |item| .{ .custom = item },
        else => error.ParseValue,
    };
}

fn summable(input: Value) bool {
    return switch (input) {
        .integer, .number_string, .float => true,
        .custom => |item| blk: {
            if (item.integerValue() != null) break :blk true;
            const builtin = python.view(.{ .custom = item }) orelse break :blk false;
            break :blk builtin == .float;
        },
        else => false,
    };
}

/// Iterate borrowed graph values, never roundtrip whole pairs through runtime:
/// that would detach mutable pair values. Runtime dictionary keys are immutable.
const Iterator = struct {
    value: Value,
    index: u32 = 0,

    fn next(self: *Iterator, arena: Allocator) Error!?Value {
        const index = self.index;
        if (index >= try length(self.value)) return null;
        self.index += 1;
        return switch (self.value) {
            .object, .dictionary => try Graph.fromValue(arena, (try dictionaryEntry(self.value, index)).?.key),
            .array => |items| items.items[index],
            .sequence => |items| items.items.items[index],
            .string => |text| blk: {
                const width = std.unicode.utf8ByteSequenceLength(text[index]) catch return error.ParseValue;
                if (width > text.len - index) return error.ParseValue;
                self.index = index + width;
                break :blk .{ .string = text[index..self.index] };
            },
            .custom => |item| blk: {
                const builtin = python.view(.{ .custom = item }).?;
                const bytes = if (builtin == .bytes) builtin.bytes else builtin.bytearray;
                break :blk .{ .integer = bytes[index] };
            },
            else => unreachable,
        };
    }
};

test "usage snapshots retain nested aliases and mixed counts preserve provider flags" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var storage: Storage = .{ .arena = a, .io = std.testing.io };
    var tracker: UsageTracker = undefined;
    try tracker.init(&storage, .{ .models = 2, .entries = 3 });
    const first = try Graph.parse(a, "{\"detail\":{\"count\":2,\"items\":[1]},\"flag\":true,\"count\":9007199254740993}");
    try tracker.addUsage("configured/model", first);
    try first.object.put(a, "replaced", .{ .integer = 9 });
    const second = try Graph.parse(a, "{\"flag\":2,\"count\":2}");
    try tracker.addUsage("configured/model", second);
    const snapshot = try tracker.getTotalTokens(a);
    const observed = try Graph.toValue(a, snapshot);
    const fields = observed.object.get("configured/model").?.object;
    try std.testing.expectEqual(@as(i64, 9007199254740995), fields.get("count").?.integer);
    try std.testing.expect(fields.get("flag").?.bool);
    try std.testing.expect(fields.get("replaced") == null);
    const nested = first.object.get("detail").?;
    const total = snapshot.object.get("configured/model").?.dictionary;
    try std.testing.expect(total.entries.items[0].value.object == nested.object);
    try nested.object.get("items").?.array.append(.{ .integer = 3 });
    try nested.object.put(a, "count", .{ .integer = 5 });
    const later = try Graph.toValue(a, try tracker.getTotalTokens(a));
    const detail = later.object.get("configured/model").?.object.get("detail").?.object;
    try std.testing.expectEqual(@as(i64, 5), detail.get("count").?.integer);
    try std.testing.expectEqual(@as(i64, 3), detail.get("items").?.array.items[1].integer);
}

test "task tracker forks isolate entries while preserving cross-model mutable aliases" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var storage: Storage = .{ .arena = a, .io = std.testing.io };
    var parent: UsageTracker = undefined;
    try parent.init(&storage, .{ .models = 3, .entries = 6 });
    const shared = try Graph.parse(a, "{\"items\":[1]}");
    const bytes = try python.bytearray(a, "a");
    const input = try Graph.parse(a, "{}");
    try input.object.put(a, "nested", shared);
    try input.object.put(a, "mutable", .{ .custom = bytes.custom });
    try parent.addUsage("first", input);
    try parent.addUsage("second", input);
    try parent.addUsage("first", input);
    try std.testing.expectError(error.ParseValue, parent.addUsage("empty", try Graph.parse(a, "[1]")));

    const first = try parent.fork();
    const second = try parent.fork();
    try std.testing.expect(first != second);
    try std.testing.expect(first != &parent);
    try std.testing.expectEqual(parent.model_count, first.model_count);
    try std.testing.expectEqualStrings("empty", first.models[2].name);
    try std.testing.expectEqual(parent.models[2].first, first.models[2].first);
    const nested = first.entries[0].data.dictionary.entries.items[0].value;
    const mutable = first.entries[0].data.dictionary.entries.items[1].value;
    try std.testing.expect(nested.object != shared.object);
    try std.testing.expect(mutable.custom.ptr != bytes.custom.ptr);
    for (first.entries[0..first.entry_count]) |entry| {
        try std.testing.expect(entry.data.dictionary.entries.items[0].value.object == nested.object);
        try std.testing.expect(entry.data.dictionary.entries.items[1].value.custom.ptr == mutable.custom.ptr);
    }
    try nested.object.get("items").?.array.append(.{ .integer = 2 });
    try python.bytearrayAppend(a, .{ .custom = mutable.custom }, 'b');
    try std.testing.expectEqualStrings("ab", python.view(.{ .custom = mutable.custom }).?.bytearray);
    try std.testing.expectEqualStrings("a", python.view(bytes).?.bytearray);
    try std.testing.expectEqual(@as(usize, 1), shared.object.get("items").?.array.items.len);
    const other = second.entries[0].data.dictionary.entries.items[0].value;
    try std.testing.expect(other.object != nested.object);
    try std.testing.expectEqual(@as(usize, 1), other.object.get("items").?.array.items.len);
    try first.addUsage("first", try Graph.parse(a, "{\"count\":7}"));
    try std.testing.expectEqual(@as(u32, 3), parent.entry_count);
    try std.testing.expectEqual(@as(u32, 3), second.entry_count);
    try std.testing.expectEqual(@as(u32, 4), first.entry_count);
    try std.testing.expectEqual(absent, parent.entries[2].next);
    try std.testing.expectEqual(@as(u32, 3), first.entries[2].next);
}

test "scope capacity failure restores settings without publishing partial child entries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var storage: Storage = .{ .arena = a, .io = std.testing.io };
    var active: ?*UsageTracker = null;
    var parent: Scope = undefined;
    try parent.init(&storage, &active, .{ .models = 1, .entries = 1 });
    try parent.tracker.addUsage("parent", try Graph.parse(a, "{\"count\":2}"));
    var child: Scope = undefined;
    try child.init(&storage, &active, .{ .models = 2, .entries = 2 });
    try child.tracker.addUsage("parent", try Graph.parse(a, "{\"count\":3}"));
    try child.tracker.addUsage("child", try Graph.parse(a, "{\"count\":7}"));
    try std.testing.expectError(error.CapacityExceeded, child.finish());
    try std.testing.expect(active == &parent.tracker);
    const total = try Graph.toValue(a, try parent.tracker.getTotalTokens(a));
    try std.testing.expectEqual(@as(i64, 2), total.object.get("parent").?.object.get("count").?.integer);
    try std.testing.expect(total.object.get("child") == null);
    const retained = try Graph.toValue(a, try child.tracker.getTotalTokens(a));
    try std.testing.expectEqual(@as(i64, 3), retained.object.get("parent").?.object.get("count").?.integer);
    try std.testing.expectEqual(@as(i64, 7), retained.object.get("child").?.object.get("count").?.integer);
    try parent.finish();
    try std.testing.expect(active == null);
}

test "usage merge keeps iterable-pair values borrowed and bounds cyclic dictionary recursion" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pairs = try Graph.parse(a, "[[\"nested\",[1]]]");
    const result = try merge(a, pairs, try Graph.parse(a, "{\"other\":2}"));
    const borrowed = result.dictionary.entries.items[0].value;
    try std.testing.expect(borrowed.array == pairs.array.items[0].array.items[1].array);
    try pairs.array.items[0].array.items[1].array.append(.{ .integer = 3 });
    try std.testing.expectEqual(@as(i64, 3), borrowed.array.items[1].integer);
    try std.testing.expectError(error.ParseValue, merge(a, try Graph.parse(a, "{\"other\":2}"), pairs));
    const cycle = try Graph.parse(a, "{}");
    try cycle.object.put(a, "cycle", cycle);
    try std.testing.expectError(error.NestingTooDeep, merge(a, cycle, cycle));
    var exhausted: Work = .{ .remaining = 1 };
    try std.testing.expectError(error.ParseValue, mergeUsing(a, &exhausted, pairs, .null));
}
