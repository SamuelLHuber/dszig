//! Mutable JSON object graph with Python reference semantics. Containers are
//! pointers, so copying a `Value` aliases the same list or dictionary like a
//! Python shallow copy, including appends that reallocate a list's storage.
//! `std.json.Value` stores container metadata by value and cannot express
//! that sharing. `clone` mirrors `copy.deepcopy`: shared containers stay
//! shared and cycles stay cycles inside the copy. Graphs are not synchronized;
//! concurrent mutation needs external serialization.
//! Native Python scalar boxes, tuples, sets and arbitrary-key dictionaries
//! retain their kinds too. Scalar leaves imported by `fromValue` are borrowed;
//! `clone` owns independent copies in the destination arena.

const std = @import("std");
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");

pub const Value = union(enum) {
    null,
    bool: bool,
    integer: i64,
    float: f64,
    /// Lossless integer text outside `i64`.
    number_string: []const u8,
    string: []const u8,
    array: *Array,
    object: *Record,
    /// Genuine noncontainer objects, never a JSON projection.
    custom: runtime.Custom,
    sequence: *Sequence,
    dictionary: *Dictionary,
};

pub const Array = std.array_list.Managed(Value);
pub const Record = std.StringArrayHashMapUnmanaged(Value);

/// Python tuples and sets need their own kind even when empty. The graph owns
/// the item headers; copying a graph value always aliases the same container.
pub const Sequence = struct {
    kind: enum { tuple, set, frozen_set },
    hash_seed: u32 = 0,
    items: Array,
};

/// Unlike a JSON object, a Python dict can retain arbitrary hashable keys.
/// Keys are immutable runtime values; values use pointer-container semantics.
/// Construct entries through Python's dictionary protocol before importing,
/// or maintain unique, hashable keys when mutating this low-level graph.
pub const Dictionary = struct {
    entries: std.array_list.Managed(Entry),
    pub const Entry = struct { key: runtime.Value, value: Value };
};

/// Reuse across roots when genuine Python containers occur in several fields.
pub const ImportContext = @import("python_graph_conversion.zig").ImportContext;

/// Snapshot runtime container headers into the graph, borrowing scalar leaves
/// and immutable keys. Genuine Python container aliases retain one header;
/// by-value runtime arrays/objects do not carry recoverable object identity.
/// This is not deepcopy or a lease on the source storage. Build/mutate graph
/// containers directly when later appends must stay shared.
pub fn fromValue(arena: Allocator, input: runtime.Value) ConvertError!Value {
    return @import("python_graph_conversion.zig").fromValue(arena, input);
}

pub const ParseError = std.json.Scanner.AllocError || error{UnexpectedToken};

/// Parse `bytes` into `arena`, which then owns every container, key and
/// string. Integers beyond `i64` remain exact `number_string` text. Duplicate
/// keys keep their first position with the last value, like Python `dict`.
/// Nesting depth uses heap stacks, never native recursion.
pub fn parse(arena: Allocator, bytes: []const u8) ParseError!Value {
    var scanner: std.json.Scanner = .initCompleteInput(arena, bytes);
    defer scanner.deinit();

    const Frame = struct { container: Value, key: ?[]const u8 = null };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(arena);

    while (true) {
        const value: Value = switch (try scanner.nextAllocMax(arena, .alloc_always, bytes.len)) {
            .array_begin => {
                const array = try arena.create(Array);
                array.* = .init(arena);
                try stack.append(arena, .{ .container = .{ .array = array } });
                continue;
            },
            .object_begin => {
                const record = try arena.create(Record);
                record.* = .empty;
                try stack.append(arena, .{ .container = .{ .object = record } });
                continue;
            },
            .array_end, .object_end => (stack.pop() orelse return error.UnexpectedToken).container,
            .allocated_string => |text| string: {
                if (stack.items.len > 0) {
                    const top = &stack.items[stack.items.len - 1];
                    if (top.container == .object and top.key == null) {
                        top.key = text;
                        continue;
                    }
                }
                break :string .{ .string = text };
            },
            .allocated_number => |text| number(text),
            .true => .{ .bool = true },
            .false => .{ .bool = false },
            .null => .null,
            else => return error.UnexpectedToken,
        };

        if (stack.items.len == 0) {
            const tail = scanner.next() catch |err| switch (err) {
                error.BufferUnderrun => unreachable, // complete input
                else => |e| return e,
            };
            if (tail != .end_of_document) return error.UnexpectedToken;
            return value;
        }
        const top = &stack.items[stack.items.len - 1];
        switch (top.container) {
            .array => |array| try array.append(value),
            .object => |record| {
                try record.put(arena, top.key orelse return error.UnexpectedToken, value);
                top.key = null;
            },
            else => unreachable,
        }
    }
}

fn number(text: []const u8) Value {
    return switch (@import("runtime_value.zig").Value.parseFromNumberSlice(text)) {
        .integer => |i| .{ .integer = i },
        .float => |f| .{ .float = f },
        .number_string => |s| if (std.mem.indexOfAny(u8, s, ".eE") != null)
            .{ .float = std.fmt.parseFloat(f64, s) catch unreachable }
        else
            .{ .number_string = s },
        else => unreachable,
    };
}

/// `copy.deepcopy` into `arena`: every string, key and container is new, while
/// containers referenced several times (or cyclically) map to one copy.
/// On failure, partial allocations remain owned by `arena`.
pub fn clone(arena: Allocator, input: Value) Allocator.Error!Value {
    return cloneUsing(arena, arena, input);
}

/// `clone` keeping its temporary identity memo and work stack in `scratch`
/// (freed before returning) instead of the destination arena.
pub fn cloneUsing(arena: Allocator, scratch: Allocator, input: Value) Allocator.Error!Value {
    var custom_cloner: runtime.CloneContext = .init(arena);
    defer custom_cloner.deinit();
    return cloneWithRuntimeContext(arena, scratch, input, &custom_cloner);
}

/// Share the native-object identity memo with an enclosing owning clone.
/// Graph container identity is local to this graph traversal; genuine custom
/// objects shared with typed fields keep the same copy across both protocols.
/// Its destination allocator must be arena. The caller retains and
/// deinitializes custom_cloner.
pub fn cloneWithRuntimeContext(
    arena: Allocator,
    scratch: Allocator,
    input: Value,
    custom_cloner: *runtime.CloneContext,
) Allocator.Error!Value {
    var cloner = CloneContext.init(arena, scratch, custom_cloner);
    defer cloner.deinit();
    return cloner.clone(input);
}

/// An owning clone spanning several roots. Keep both this context and the
/// caller-owned runtime context alive to preserve container and custom-object
/// aliases between roots. Sources must remain alive and unchanged throughout.
/// After any clone failure, deinitialize rather than reuse the partial memo.
pub const CloneContext = struct {
    cloner: Cloner,

    pub fn init(
        arena: Allocator,
        scratch: Allocator,
        custom_cloner: *runtime.CloneContext,
    ) CloneContext {
        std.debug.assert(custom_cloner.allocator.ptr == arena.ptr);
        std.debug.assert(custom_cloner.allocator.vtable == arena.vtable);
        return .{ .cloner = .{
            .target = arena,
            .scratch = scratch,
            .custom_cloner = custom_cloner,
        } };
    }

    pub fn deinit(self: *CloneContext) void {
        self.cloner.deinit();
    }

    pub fn clone(self: *CloneContext, input: Value) Allocator.Error!Value {
        std.debug.assert(self.cloner.pending.items.len == 0);
        const root = try self.cloner.copy(input);
        while (self.cloner.pending.pop()) |job| try self.cloner.fill(job);
        std.debug.assert(self.cloner.pending.items.len == 0);
        return root;
    }
};

const Cloner = struct {
    target: Allocator,
    scratch: Allocator,
    custom_cloner: *runtime.CloneContext,
    /// Source container address to its copy.
    memo: std.AutoHashMapUnmanaged(usize, Value) = .empty,
    /// Copies allocated but not yet filled; filling iteratively keeps native
    /// stack use constant for arbitrarily deep graphs.
    pending: std.ArrayList(Job) = .empty,

    const Job = struct { source: Value, copy: Value };

    fn deinit(self: *Cloner) void {
        self.memo.deinit(self.scratch);
        self.pending.deinit(self.scratch);
    }

    fn copy(self: *Cloner, v: Value) Allocator.Error!Value {
        return switch (v) {
            .string => |s| .{ .string = try self.target.dupe(u8, s) },
            .number_string => |s| .{ .number_string = try self.target.dupe(u8, s) },
            .array => |array| self.container(v, @intFromPtr(array)),
            .object => |record| self.container(v, @intFromPtr(record)),
            .sequence => |sequence| self.container(v, @intFromPtr(sequence)),
            .dictionary => |dictionary| self.container(v, @intFromPtr(dictionary)),
            .custom => |custom| .{ .custom = try self.custom_cloner.cloneCustom(custom) },
            .null, .bool, .integer, .float => v,
        };
    }

    fn container(self: *Cloner, source: Value, address: usize) Allocator.Error!Value {
        if (self.memo.get(address)) |existing| return existing;
        try self.memo.ensureUnusedCapacity(self.scratch, 1);
        try self.pending.ensureUnusedCapacity(self.scratch, 1);
        std.debug.assert(source == .array or source == .object or
            source == .sequence or source == .dictionary);
        const out: Value = switch (source) {
            .array => array: {
                const array = try self.target.create(Array);
                array.* = .init(self.target);
                break :array .{ .array = array };
            },
            .object => object: {
                const record = try self.target.create(Record);
                record.* = .empty;
                break :object .{ .object = record };
            },
            .sequence => |sequence| sequence: {
                const out = try self.target.create(Sequence);
                out.* = .{
                    .kind = sequence.kind,
                    .hash_seed = sequence.hash_seed,
                    .items = .init(self.target),
                };
                break :sequence .{ .sequence = out };
            },
            .dictionary => dictionary: {
                const out = try self.target.create(Dictionary);
                out.* = .{ .entries = .init(self.target) };
                break :dictionary .{ .dictionary = out };
            },
            else => unreachable,
        };
        self.memo.putAssumeCapacityNoClobber(address, out);
        self.pending.appendAssumeCapacity(.{ .source = source, .copy = out });
        return out;
    }

    fn fill(self: *Cloner, job: Job) Allocator.Error!void {
        switch (job.source) {
            .array => |source| {
                const out = job.copy.array;
                try out.ensureTotalCapacity(source.items.len);
                for (source.items) |item| out.appendAssumeCapacity(try self.copy(item));
            },
            .object => |source| {
                const out = job.copy.object;
                try out.ensureTotalCapacity(self.target, source.count());
                for (source.keys(), source.values()) |key, item| {
                    const owned_key = try self.target.dupe(u8, key);
                    out.putAssumeCapacity(owned_key, try self.copy(item));
                }
            },
            .sequence => |source| {
                const out = job.copy.sequence;
                try out.items.ensureTotalCapacity(source.items.items.len);
                for (source.items.items) |item|
                    out.items.appendAssumeCapacity(try self.copy(item));
            },
            .dictionary => |source| {
                const out = job.copy.dictionary;
                try out.entries.ensureTotalCapacity(source.entries.items.len);
                for (source.entries.items) |entry| out.entries.appendAssumeCapacity(.{
                    .key = try self.custom_cloner.cloneValue(entry.key),
                    .value = try self.copy(entry.value),
                });
            },
            else => unreachable,
        }
    }
};

/// Containers `toValue` may nest, roughly CPython's default recursion limit
/// that bounds `json.dumps`.
pub const max_conversion_depth = 1000;

pub const ConvertError = Allocator.Error || error{ CyclicValue, NestingTooDeep, ParseValue };

/// Convert acyclic graph containers to runtime values without erasing Python
/// kinds. Strings are copied; custom leaves and non-string keys are borrowed.
/// Cycles, excessive nesting and unhashable Python keys fail explicitly.
pub fn toValue(arena: Allocator, input: Value) ConvertError!runtime.Value {
    return @import("python_graph_conversion.zig").toValue(arena, input);
}

/// Python bool(), including failures from genuine class protocols.
pub fn truthy(input: Value) error{ParseValue}!bool {
    return switch (input) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        // NaN is truthy in Python, as `nan != 0` is here.
        .float => |f| f != 0,
        .number_string => |text| numberTextTruthy(text),
        .string => |s| s.len > 0,
        .array => |array| array.items.len > 0,
        .object => |record| record.count() > 0,
        .sequence => |sequence| sequence.items.items.len > 0,
        .dictionary => |dictionary| dictionary.entries.items.len > 0,
        .custom => |custom| try custom.truthy(),
    };
}

fn numberTextTruthy(text: []const u8) bool {
    if (!std.json.isNumberFormattedLikeAnInteger(text)) {
        if (std.fmt.parseFloat(f64, text)) |f| return f != 0 else |_| {}
    }
    const mantissa_end = std.mem.indexOfAny(u8, text, "eE") orelse text.len;
    for (text[0..mantissa_end]) |c| if (c >= '1' and c <= '9') return true;
    return false;
}

const testing = std.testing;

test "parse owns strings, keeps exact integers and Python duplicate keys" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var buffer = "{\"text\":\"caf\\u00e9\",\"x\":1,\"big\":-123456789012345678901234567890,\"half\":0.5,\"x\":2}".*;
    const root = try parse(arena, &buffer);
    @memset(&buffer, 'z');

    try testing.expectEqual(@as(usize, 4), root.object.count());
    try testing.expectEqualStrings("text", root.object.keys()[0]);
    try testing.expectEqualStrings("x", root.object.keys()[1]);
    try testing.expectEqualStrings("café", root.object.get("text").?.string);
    try testing.expectEqual(@as(i64, 2), root.object.get("x").?.integer);
    try testing.expectEqualStrings("-123456789012345678901234567890", root.object.get("big").?.number_string);
    try testing.expectEqual(@as(f64, 0.5), root.object.get("half").?.float);
    // requests' actual JSON backend parses overflowing exponents as floats,
    // while arbitrarily large integer tokens stay integers.
    try testing.expectEqual(std.math.inf(f64), (try parse(arena, "1e400")).float);
    try testing.expectEqual(-std.math.inf(f64), (try parse(arena, "-1e400")).float);

    for ([_][]const u8{ "[1] 2", "{\"a\":}", "[1,", "" }) |bad| {
        if (parse(arena, bad)) |_| return error.TestUnexpectedResult else |_| {}
    }
}

test "array aliases see appends across reallocation while clones stay isolated" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const root = try parse(arena, "{\"rows\":[{\"meta\":{\"tags\":[\"a\"]}}]}");
    const copied = try clone(arena, root);
    const tags = struct {
        fn of(v: Value) *Array {
            return v.object.get("rows").?.array.items[0].object.get("meta").?.object.get("tags").?.array;
        }
    }.of;

    // Copied values share containers, so appends through any copy are seen by all.
    const row_copy: Value = root.object.get("rows").?.array.items[0];
    const alias = row_copy.object.get("meta").?.object.get("tags").?.array;
    for (0..64) |i| try alias.append(.{ .integer = @intCast(i) });

    try testing.expectEqual(@as(usize, 65), tags(root).items.len);
    try testing.expectEqual(@as(i64, 63), tags(root).items[64].integer);
    try testing.expectEqual(@as(usize, 1), tags(copied).items.len);

    try tags(copied).append(.{ .string = "copy" });
    try testing.expectEqual(@as(usize, 65), tags(root).items.len);
    try testing.expectEqualStrings("a", tags(copied).items[0].string);
}

test "clone preserves shared containers and cycles" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const shared = try arena.create(Array);
    shared.* = .init(arena);
    try shared.append(.{ .string = "x" });
    const record = try arena.create(Record);
    record.* = .empty;
    try record.put(arena, "first", .{ .array = shared });
    try record.put(arena, "second", .{ .array = shared });
    try record.put(arena, "self", .{ .object = record });
    try shared.append(.{ .object = record });

    const copied = (try clone(arena, .{ .object = record })).object;
    const first = copied.get("first").?.array;
    try testing.expect(copied != record);
    try testing.expect(first != shared);
    try testing.expect(first == copied.get("second").?.array);
    try testing.expect(copied.get("self").?.object == copied);
    try testing.expect(first.items[1].object == copied);
    try testing.expect(first.items[0].string.ptr != shared.items[0].string.ptr);

    try first.append(.{ .integer = 1 });
    try testing.expectEqual(@as(usize, 3), copied.get("second").?.array.items.len);
    try testing.expectEqual(@as(usize, 2), shared.items.len);
}

test "deep graphs parse and clone iteratively; conversion rejects cycles and depth" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const deep = 100_000;
    const text = try arena.alloc(u8, 2 * deep);
    @memset(text[0..deep], '[');
    @memset(text[deep..], ']');
    const nested = try clone(arena, try parse(arena, text));
    try testing.expectError(error.NestingTooDeep, toValue(arena, nested));

    const limit = max_conversion_depth;
    const at_limit = try arena.alloc(u8, 2 * limit);
    @memset(at_limit[0..limit], '[');
    @memset(at_limit[limit..], ']');
    _ = try toValue(arena, try parse(arena, at_limit));
    const over = try arena.alloc(u8, 2 * (limit + 1));
    @memset(over[0 .. limit + 1], '[');
    @memset(over[limit + 1 ..], ']');
    try testing.expectError(error.NestingTooDeep, toValue(arena, try parse(arena, over)));

    const cyclic = try parse(arena, "{\"items\":[]}");
    try cyclic.object.get("items").?.array.append(cyclic);
    try testing.expectError(error.CyclicValue, toValue(arena, cyclic));

    // Shared acyclic containers expand; numbers and strings survive exactly.
    const dag = try parse(arena, "{\"shared\":[18446744073709551616,\"s\"]}");
    try dag.object.put(arena, "again", dag.object.get("shared").?);
    const converted = try toValue(arena, dag);
    try testing.expectEqualStrings("18446744073709551616", converted.object.get("again").?.array.items[0].number_string);
    try testing.expectEqualStrings("s", converted.object.get("shared").?.array.items[1].string);
}

test "Python graph snapshots retain tuple dict scalar kinds and cloned mutable aliases" {
    const python = @import("python_value.zig");
    var source_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer source_state.deinit();
    const source = source_state.allocator();
    var copy_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer copy_state.deinit();
    const arena = copy_state.allocator();

    const bytes = try python.bytes(source, "\xff");
    const key = try python.tuple(source, &.{bytes});
    const dictionary = try python.dictionary(source, &.{.{
        .key = key,
        .value = try python.decimal(source, "1.25"),
    }});
    var list: runtime.Array = .init(source);
    try list.append(.{ .integer = 1 });
    const tuple = try python.tuple(source, &.{ .{ .array = list }, dictionary, bytes });
    const snapshot = try fromValue(source, tuple);
    const shared = snapshot.sequence.items.items[0];
    try snapshot.sequence.items.append(shared);
    const copy = try clone(arena, snapshot);
    const copy_list = copy.sequence.items.items[0].array;
    try testing.expect(copy_list == copy.sequence.items.items[3].array);
    try testing.expect(copy_list != shared.array);

    // Appending across reallocations must stay visible through tuple aliases.
    for (0..64) |i| try shared.array.append(.{ .integer = @intCast(i) });
    try testing.expectEqual(@as(usize, 65), snapshot.sequence.items.items[3].array.items.len);
    try testing.expectEqual(@as(usize, 1), copy_list.items.len);
    source_state.deinit();
    source_state = .init(testing.allocator);

    const restored = try toValue(arena, copy);
    const items = python.view(restored).?.tuple;
    try testing.expectEqual(@as(i64, 1), items[0].array.items[0].integer);
    const entries = python.view(items[1]).?.dictionary;
    const cloned_key_bytes = python.view(entries[0].key).?.tuple[0];
    try testing.expectEqualStrings("\xff", python.view(cloned_key_bytes).?.bytes);
    var decimal_text: std.Io.Writer.Allocating = .init(arena);
    try entries[0].value.custom.writeStr(&decimal_text.writer);
    try testing.expectEqualStrings("1.25", decimal_text.written());
    try testing.expectEqualStrings("\xff", python.view(items[2]).?.bytes);
    try testing.expect(try python.containerEqual(python.view(entries[0].key).?.tuple[0], items[2]));
    try testing.expect(!items[2].custom.owns_box);
}

test "Python graph conversions retain seeded set order and refuse cross-kind cycles" {
    const python = @import("python_value.zig");
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try python.setWithHashSeed(arena, &.{
        .{ .string = "alpha" }, .{ .string = "beta" },
        .{ .string = "gamma" }, .{ .string = "delta" },
    }, 321);
    const snapshot = try fromValue(arena, set);
    const restored = try toValue(arena, snapshot);
    const before = python.view(set).?.set;
    const after = python.view(restored).?.set;
    for (before, after) |a, b| try testing.expectEqualStrings(a.string, b.string);
    try testing.expect(try python.containerEqual(set, restored));
    snapshot.sequence.kind = .frozen_set;
    const frozen = try toValue(arena, snapshot);
    for (before, python.view(frozen).?.frozen_set) |a, b|
        try testing.expectEqualStrings(a.string, b.string);
    try testing.expect(try python.containerEqual(set, frozen));
    try testing.expectError(error.ParseValue, python.setFromOwnedIterationOrder(
        arena,
        try arena.dupe(runtime.Value, &.{ .{ .integer = 1 }, .{ .bool = true } }),
        321,
        false,
    ));

    const invalid = try arena.create(Sequence);
    invalid.* = .{ .kind = .set, .items = .init(arena) };
    const list = try arena.create(Array);
    list.* = .init(arena);
    try invalid.items.append(.{ .array = list });
    try testing.expectError(error.ParseValue, toValue(arena, .{ .sequence = invalid }));

    // A tuple -> dict -> tuple cycle must survive deepcopy, but cannot be
    // materialized as ordinary runtime container headers.
    const dictionary = try arena.create(Dictionary);
    dictionary.* = .{ .entries = .init(arena) };
    invalid.kind = .tuple;
    invalid.items.clearRetainingCapacity();
    try invalid.items.append(.{ .dictionary = dictionary });
    try dictionary.entries.append(.{ .key = .{ .integer = 7 }, .value = .{ .sequence = invalid } });
    const copied = try clone(arena, .{ .sequence = invalid });
    const back_reference = copied.sequence.items.items[0].dictionary.entries.items[0].value;
    try testing.expect(back_reference.sequence == copied.sequence);
    try testing.expectError(error.CyclicValue, toValue(arena, copied));
}

test "truthy follows Python truth for decoded JSON" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const cases = [_]struct { []const u8, bool }{
        .{ "null", false }, .{ "false", false }, .{ "true", true },
        .{ "0", false },    .{ "-0", false },    .{ "-1", true },
        .{ "0.0", false },  .{ "-0.0", false },  .{ "1e-400", false },
        .{ "1e400", true }, .{ "0.5", true },    .{ "123456789012345678901234567890", true },
        .{ "\"\"", false }, .{ "\"0\"", true },  .{ "[]", false },
        .{ "[0]", true },   .{ "{}", false },    .{ "{\"\":null}", true },
    };
    for (cases) |case| try testing.expectEqual(case[1], try truthy(try parse(arena, case[0])));
    try testing.expect(try truthy(.{ .float = std.math.nan(f64) }));
}
