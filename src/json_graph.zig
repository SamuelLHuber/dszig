//! Mutable JSON object graph with Python reference semantics. Containers are
//! pointers, so copying a `Value` aliases the same list or dictionary like a
//! Python shallow copy, including appends that reallocate a list's storage.
//! `std.json.Value` stores container metadata by value and cannot express
//! that sharing. `clone` mirrors `copy.deepcopy`: shared containers stay
//! shared and cycles stay cycles inside the copy. Graphs are not synchronized;
//! concurrent mutation needs external serialization.

const std = @import("std");
const Allocator = std.mem.Allocator;

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
};

pub const Array = std.array_list.Managed(Value);
pub const Record = std.StringArrayHashMapUnmanaged(Value);

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
    return switch (std.json.Value.parseFromNumberSlice(text)) {
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
    var cloner: Cloner = .{ .target = arena, .scratch = scratch };
    defer cloner.deinit();
    const root = try cloner.copy(input);
    while (cloner.pending.pop()) |job| try cloner.fill(job);
    return root;
}

const Cloner = struct {
    target: Allocator,
    scratch: Allocator,
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
            .null, .bool, .integer, .float => v,
        };
    }

    fn container(self: *Cloner, source: Value, address: usize) Allocator.Error!Value {
        const entry = try self.memo.getOrPut(self.scratch, address);
        if (entry.found_existing) return entry.value_ptr.*;
        try self.pending.ensureUnusedCapacity(self.scratch, 1);
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
            else => unreachable,
        };
        entry.value_ptr.* = out;
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
            else => unreachable,
        }
    }
};

/// Containers `toValue` may nest, roughly CPython's default recursion limit
/// that bounds `json.dumps`.
pub const max_conversion_depth = 1000;

pub const ConvertError = Allocator.Error || error{ CyclicValue, NestingTooDeep };

/// Convert acyclic data to `std.json.Value` for the existing serializers,
/// copying strings into `arena`. Shared acyclic containers are expanded; a
/// cycle or more than `max_conversion_depth` nested containers fails.
pub fn toValue(arena: Allocator, input: Value) ConvertError!std.json.Value {
    var active: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer active.deinit(arena);
    return convert(arena, &active, input, 0);
}

fn convert(
    arena: Allocator,
    active: *std.AutoHashMapUnmanaged(usize, void),
    v: Value,
    depth: usize,
) ConvertError!std.json.Value {
    return switch (v) {
        .null => .null,
        .bool => |b| .{ .bool = b },
        .integer => |i| .{ .integer = i },
        .float => |f| .{ .float = f },
        .number_string => |s| .{ .number_string = try arena.dupe(u8, s) },
        .string => |s| .{ .string = try arena.dupe(u8, s) },
        .array => |array| array: {
            try enter(arena, active, @intFromPtr(array), depth);
            defer _ = active.remove(@intFromPtr(array));
            var out: std.json.Array = .init(arena);
            try out.ensureTotalCapacity(array.items.len);
            for (array.items) |item| out.appendAssumeCapacity(try convert(arena, active, item, depth + 1));
            break :array .{ .array = out };
        },
        .object => |record| object: {
            try enter(arena, active, @intFromPtr(record), depth);
            defer _ = active.remove(@intFromPtr(record));
            var out: std.json.ObjectMap = .empty;
            try out.ensureTotalCapacity(arena, record.count());
            for (record.keys(), record.values()) |key, item| {
                out.putAssumeCapacity(try arena.dupe(u8, key), try convert(arena, active, item, depth + 1));
            }
            break :object .{ .object = out };
        },
    };
}

fn enter(arena: Allocator, active: *std.AutoHashMapUnmanaged(usize, void), address: usize, depth: usize) ConvertError!void {
    const entry = try active.getOrPut(arena, address);
    if (entry.found_existing) return error.CyclicValue;
    if (depth >= max_conversion_depth) return error.NestingTooDeep;
}

/// Python `bool()` of the decoded JSON value.
pub fn truthy(input: Value) bool {
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
    for (cases) |case| try testing.expectEqual(case[1], truthy(try parse(arena, case[0])));
    try testing.expect(truthy(.{ .float = std.math.nan(f64) }));
}
