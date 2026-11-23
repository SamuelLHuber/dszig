//! Arena-backed, mixed-borrowed conversion at the runtime/graph boundary.
//! Explicit DFS frames bound native stack use; no JSON serialization is involved.
const std = @import("std");
const Allocator = std.mem.Allocator;
const graph = @import("json_graph.zig");
const runtime = @import("runtime_value.zig");
const python = @import("python_value.zig");
const Error = graph.ConvertError;

pub const max_steps = 1 << 20;

const FromFrame = struct {
    source: runtime.Value,
    target: *graph.Value,
    entered: bool = false,
    index: usize = 0,
    count: usize = 0,
};

/// Snapshots genuine container identities across one or more runtime roots.
/// Keep source boxes alive while importing. Scalar leaves remain borrowed.
/// Discard the context and destination arena on any conversion failure.
pub const ImportContext = struct {
    arena: Allocator,
    memo: std.AutoHashMapUnmanaged(Key, graph.Value) = .empty,

    const Key = struct {
        ptr: *const anyopaque,
        vtable: *const runtime.CustomVTable,
    };

    pub fn init(arena: Allocator) ImportContext {
        return .{ .arena = arena };
    }

    pub fn deinit(self: *ImportContext) void {
        self.memo.deinit(self.arena);
        self.* = undefined;
    }

    pub fn fromValue(self: *ImportContext, input: runtime.Value) Error!graph.Value {
        var result: graph.Value = undefined;
        var stack: [graph.max_conversion_depth + 1]FromFrame = undefined;
        stack[0] = .{ .source = input, .target = &result };
        var depth: usize = 0;
        var steps: u32 = 0;
        while (true) {
            const frame = &stack[depth];
            if (!frame.entered) {
                if (steps == max_steps) return error.ParseValue;
                steps += 1;
                frame.count = try fromEnter(self, frame.source, frame.target);
                if (containerAddress(frame.target.*) != 0 and depth >= graph.max_conversion_depth)
                    return error.NestingTooDeep;
                frame.entered = true;
            }
            if (frame.index < frame.count) {
                if (depth == graph.max_conversion_depth) return error.NestingTooDeep;
                const child = fromChild(frame, frame.index);
                frame.index += 1;
                depth += 1;
                stack[depth] = child;
                continue;
            }
            if (depth == 0) return result;
            depth -= 1;
        }
    }
};

pub fn fromValue(arena: Allocator, input: runtime.Value) Error!graph.Value {
    var context: ImportContext = .init(arena);
    defer context.deinit();
    return context.fromValue(input);
}

fn borrowed(input: runtime.Value) runtime.Value {
    return if (input == .custom) .{ .custom = input.custom.borrowed() } else input;
}

fn fromEnter(context: *ImportContext, source: runtime.Value, target: *graph.Value) Error!usize {
    const arena = context.arena;
    switch (source) {
        .null => target.* = .null,
        .bool => |b| target.* = .{ .bool = b },
        .integer => |n| target.* = .{ .integer = n },
        .float => |n| target.* = .{ .float = n },
        .number_string => |text| target.* = .{ .number_string = text },
        .string => |text| target.* = .{ .string = text },
        .array => |array| {
            const out = try arena.create(graph.Array);
            out.* = .init(arena);
            try out.resize(array.items.len);
            target.* = .{ .array = out };
            return array.items.len;
        },
        .object => |object| {
            const out = try arena.create(graph.Record);
            out.* = .empty;
            try out.ensureTotalCapacity(arena, object.count());
            for (object.keys()) |key| out.putAssumeCapacity(key, .null);
            target.* = .{ .object = out };
            return object.count();
        },
        .custom => |custom| {
            if (python.view(source)) |payload| {
                const key: ImportContext.Key = .{ .ptr = custom.ptr, .vtable = custom.vtable };
                switch (payload) {
                    .tuple, .set, .frozen_set, .dictionary => {
                        if (context.memo.get(key)) |existing| {
                            target.* = existing;
                            return 0;
                        }
                    },
                    else => {},
                }
                switch (payload) {
                    .tuple, .set, .frozen_set => |items| {
                        const out = try arena.create(graph.Sequence);
                        out.* = .{
                            .kind = switch (payload) {
                                .tuple => .tuple,
                                .set => .set,
                                .frozen_set => .frozen_set,
                                else => unreachable,
                            },
                            .hash_seed = python.sequenceHashSeed(source),
                            .items = .init(arena),
                        };
                        try out.items.resize(items.len);
                        target.* = .{ .sequence = out };
                        try context.memo.putNoClobber(arena, key, target.*);
                        return items.len;
                    },
                    .dictionary => |pairs| {
                        const out = try arena.create(graph.Dictionary);
                        out.* = .{ .entries = .init(arena) };
                        try out.entries.ensureTotalCapacity(pairs.len);
                        for (pairs) |pair| out.entries.appendAssumeCapacity(.{
                            .key = borrowed(pair.key),
                            .value = .null,
                        });
                        target.* = .{ .dictionary = out };
                        try context.memo.putNoClobber(arena, key, target.*);
                        return pairs.len;
                    },
                    else => {},
                }
            }
            target.* = .{ .custom = custom.borrowed() };
        },
    }
    return 0;
}

fn fromChild(frame: *const FromFrame, index: usize) FromFrame {
    std.debug.assert(index < frame.count);
    return switch (frame.source) {
        .array => |array| .{
            .source = array.items[index],
            .target = &frame.target.array.items[index],
        },
        .object => |object| .{
            .source = object.values()[index],
            .target = &frame.target.object.values()[index],
        },
        .custom => switch (python.view(frame.source).?) {
            .tuple, .set, .frozen_set => |items| .{
                .source = items[index],
                .target = &frame.target.sequence.items.items[index],
            },
            .dictionary => |pairs| .{
                .source = pairs[index].value,
                .target = &frame.target.dictionary.entries.items[index].value,
            },
            else => unreachable,
        },
        else => unreachable,
    };
}

const ToFrame = struct {
    source: graph.Value,
    target: *runtime.Value,
    entered: bool = false,
    index: usize = 0,
    count: usize = 0,
    pairs: []runtime.PythonEntry = &.{},
};

pub fn toValue(arena: Allocator, input: graph.Value) Error!runtime.Value {
    var result: runtime.Value = undefined;
    var stack: [graph.max_conversion_depth + 1]ToFrame = undefined;
    stack[0] = .{ .source = input, .target = &result };
    var depth: usize = 0;
    var steps: u32 = 0;
    while (true) {
        const frame = &stack[depth];
        if (!frame.entered) {
            if (steps == max_steps) return error.ParseValue;
            steps += 1;
            const address = containerAddress(frame.source);
            if (address != 0) {
                for (stack[0..depth]) |ancestor|
                    if (containerAddress(ancestor.source) == address) return error.CyclicValue;
                if (depth >= graph.max_conversion_depth) return error.NestingTooDeep;
            }
            try toEnter(arena, frame);
            frame.entered = true;
        }
        if (frame.index < frame.count) {
            if (depth == graph.max_conversion_depth) return error.NestingTooDeep;
            const child = toChild(frame, frame.index);
            frame.index += 1;
            depth += 1;
            stack[depth] = child;
            continue;
        }
        try toFinish(arena, frame);
        if (depth == 0) return result;
        depth -= 1;
    }
}

fn containerAddress(input: graph.Value) usize {
    return switch (input) {
        .array => |p| @intFromPtr(p),
        .object => |p| @intFromPtr(p),
        .sequence => |p| @intFromPtr(p),
        .dictionary => |p| @intFromPtr(p),
        else => 0,
    };
}

fn toEnter(arena: Allocator, frame: *ToFrame) Error!void {
    frame.target.* = switch (frame.source) {
        .null => .null,
        .bool => |b| .{ .bool = b },
        .integer => |n| .{ .integer = n },
        .float => |n| .{ .float = n },
        .number_string => |text| .{ .number_string = try arena.dupe(u8, text) },
        .string => |text| .{ .string = try arena.dupe(u8, text) },
        .custom => |custom| .{ .custom = custom.borrowed() },
        .array, .sequence => array: {
            frame.count = switch (frame.source) {
                .array => |items| items.items.len,
                .sequence => |sequence| sequence.items.items.len,
                else => unreachable,
            };
            var out: runtime.Array = .init(arena);
            out.items = try arena.alloc(runtime.Value, frame.count);
            out.capacity = frame.count;
            break :array .{ .array = out };
        },
        .object => |object| object: {
            var out: runtime.ObjectMap = .empty;
            try out.ensureTotalCapacity(arena, object.count());
            for (object.keys()) |key| out.putAssumeCapacity(try arena.dupe(u8, key), .null);
            frame.count = object.count();
            break :object .{ .object = out };
        },
        .dictionary => |dictionary| dictionary: {
            frame.pairs = try arena.alloc(runtime.PythonEntry, dictionary.entries.items.len);
            for (dictionary.entries.items, frame.pairs) |entry, *out| out.* = .{
                .key = if (entry.key == .string)
                    .{ .string = try arena.dupe(u8, entry.key.string) }
                else
                    borrowed(entry.key),
                .value = .null,
            };
            frame.count = frame.pairs.len;
            break :dictionary .null;
        },
    };
}

fn toChild(frame: *ToFrame, index: usize) ToFrame {
    std.debug.assert(index < frame.count);
    return switch (frame.source) {
        .array => |array| .{
            .source = array.items[index],
            .target = &frame.target.array.items[index],
        },
        .sequence => |sequence| .{
            .source = sequence.items.items[index],
            .target = &frame.target.array.items[index],
        },
        .object => |object| .{
            .source = object.values()[index],
            .target = &frame.target.object.values()[index],
        },
        .dictionary => |dictionary| .{
            .source = dictionary.entries.items[index].value,
            .target = &frame.pairs[index].value,
        },
        else => unreachable,
    };
}

fn toFinish(arena: Allocator, frame: *ToFrame) Error!void {
    switch (frame.source) {
        .sequence => |sequence| {
            const items = frame.target.array.items;
            const result = switch (sequence.kind) {
                .tuple => try python.tupleFromOwnedSlice(arena, items),
                .set => try python.setFromOwnedIterationOrder(arena, items, sequence.hash_seed, false),
                .frozen_set => try python.setFromOwnedIterationOrder(
                    arena,
                    items,
                    sequence.hash_seed,
                    true,
                ),
            };
            frame.target.* = result;
        },
        .dictionary => frame.target.* = try python.dictionary(arena, frame.pairs),
        else => {},
    }
}
