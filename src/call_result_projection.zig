//! Source attributes/plain conversions, not BaseModel model_dump serialization.
//! Each traversed container occurrence is fresh; noncontainer leaves borrow the
//! input owner. Explicit frames bound cycles and native stack consumption.
const std = @import("std");
const graph = @import("json_graph.zig");
const runtime = @import("runtime_value.zig");
const python = @import("python_value.zig");
const Allocator = std.mem.Allocator;
const Error = runtime.CustomError;
pub const Mode = enum { attributes, plain };
const Input = union(enum) { graph: graph.Value, value: runtime.Value };
const Frame = struct {
    input: Input,
    output: *graph.Value,
    entered: bool = false,
    cursor: usize = 0,
    count: usize = 0,
};

pub fn project(arena: Allocator, input: graph.Value, mode: Mode) Error!graph.Value {
    return (try projectOwned(arena, input, mode)).value;
}

pub const Projection = struct { value: graph.Value, borrows_payload: bool };

/// Plain projections own immutable text and key bytes. Only untouched
/// noncontainer leaves require the original payload's lifetime.
pub fn projectOwned(arena: Allocator, input: graph.Value, mode: Mode) Error!Projection {
    var borrows_payload = false;
    var result: graph.Value = undefined;
    var stack: [graph.max_conversion_depth + 1]Frame = undefined;
    stack[0] = .{ .input = .{ .graph = input }, .output = &result };
    var depth: usize = 0;
    for (0..@import("python_graph_conversion.zig").max_steps) |_| {
        const frame = &stack[depth];
        if (!frame.entered) {
            frame.count = try enter(arena, frame, mode, &borrows_payload);
            frame.entered = true;
        }
        if (frame.cursor < frame.count) {
            if (depth == graph.max_conversion_depth) return error.ParseValue;
            const next = child(frame, frame.cursor);
            frame.cursor += 1;
            depth += 1;
            stack[depth] = next;
        } else {
            if (depth == 0) return .{ .value = result, .borrows_payload = borrows_payload };
            depth -= 1;
        }
    }
    return error.ParseValue;
}

fn array(arena: Allocator, output: *graph.Value, count: usize) Error!usize {
    const items = try arena.create(graph.Array);
    items.* = .init(arena);
    try items.resize(count);
    output.* = .{ .array = items };
    return count;
}

fn object(arena: Allocator, output: *graph.Value, keys: []const []const u8, own_keys: bool) Error!usize {
    const fields = try arena.create(graph.Record);
    fields.* = .empty;
    try fields.ensureTotalCapacity(arena, keys.len);
    for (keys) |key| fields.putAssumeCapacity(if (own_keys) try arena.dupe(u8, key) else key, .null);
    output.* = .{ .object = fields };
    return keys.len;
}

fn dictionary(arena: Allocator, output: *graph.Value, entries: anytype, own_keys: bool, borrows: *bool) Error!usize {
    const fields = try arena.create(graph.Dictionary);
    fields.* = .{ .entries = .init(arena) };
    try fields.entries.ensureTotalCapacity(entries.len);
    for (entries) |entry| {
        const key = switch (entry.key) {
            .string => |text| runtime.Value{ .string = if (own_keys) try arena.dupe(u8, text) else text },
            .number_string => |text| runtime.Value{ .number_string = if (own_keys) try arena.dupe(u8, text) else text },
            .custom => block: {
                borrows.* = true;
                break :block entry.key;
            },
            else => entry.key,
        };
        fields.entries.appendAssumeCapacity(.{ .key = key, .value = .null });
    }
    output.* = .{ .dictionary = fields };
    return entries.len;
}

fn enter(arena: Allocator, frame: *Frame, mode: Mode, borrows: *bool) Error!usize {
    switch (frame.input) {
        .graph => |input| switch (input) {
            .array => |items| return array(arena, frame.output, items.items.len),
            .object => |fields| return object(arena, frame.output, fields.keys(), mode == .plain),
            .dictionary => |fields| return dictionary(arena, frame.output, fields.entries.items, mode == .plain, borrows),
            .sequence => |sequence| if (mode == .plain and sequence.kind == .tuple) {
                return array(arena, frame.output, sequence.items.items.len);
            },
            .custom => |custom| if (mode == .plain and custom.vtable.model_dump != null) {
                frame.input = .{ .value = try modelDictionary(arena, custom) };
                return enterValue(arena, frame, borrows);
            },
            else => {},
        },
        .value => return enterValue(arena, frame, borrows),
    }
    frame.output.* = switch (frame.input.graph) {
        .string => |text| .{ .string = if (mode == .plain) try arena.dupe(u8, text) else text },
        .number_string => |text| .{ .number_string = if (mode == .plain) try arena.dupe(u8, text) else text },
        .sequence, .custom => block: {
            borrows.* = true;
            break :block frame.input.graph;
        },
        else => frame.input.graph,
    };
    return 0;
}

fn enterValue(arena: Allocator, frame: *Frame, borrows: *bool) Error!usize {
    // Runtime children arise only from dict(BaseModel), hence plain mode.
    var input = frame.input.value;
    if (input == .custom and input.custom.vtable.model_dump != null) {
        input = try modelDictionary(arena, input.custom);
        frame.input = .{ .value = input };
    }
    switch (input) {
        .array => |items| return array(arena, frame.output, items.items.len),
        .object => |fields| return object(arena, frame.output, fields.keys(), true),
        .custom => |custom| {
            if (python.view(input)) |payload| switch (payload) {
                .tuple => |items| return array(arena, frame.output, items.len),
                .dictionary => |entries| return dictionary(arena, frame.output, entries, true, borrows),
                else => {},
            };
            borrows.* = true;
            frame.output.* = .{ .custom = custom.borrowed() };
        },
        .string => |text| frame.output.* = .{ .string = try arena.dupe(u8, text) },
        .number_string => |text| frame.output.* = .{ .number_string = try arena.dupe(u8, text) },
        inline else => |value, tag| frame.output.* = @unionInit(graph.Value, @tagName(tag), value),
    }
    return 0;
}

fn child(frame: *const Frame, index: usize) Frame {
    std.debug.assert(index < frame.count);
    const input: Input = switch (frame.input) {
        .graph => |value| .{ .graph = switch (value) {
            .array => |items| items.items[index],
            .object => |fields| fields.values()[index],
            .dictionary => |fields| fields.entries.items[index].value,
            .sequence => |sequence| sequence.items.items[index],
            else => unreachable,
        } },
        .value => |value| .{ .value = switch (value) {
            .array => |items| items.items[index],
            .object => |fields| fields.values()[index],
            .custom => switch (python.view(value).?) {
                .tuple => |items| items[index],
                .dictionary => |entries| entries[index].value,
                else => unreachable,
            },
            else => unreachable,
        } },
    };
    const output = switch (frame.output.*) {
        .array => |items| &items.items[index],
        .object => |fields| &fields.values()[index],
        .dictionary => |fields| &fields.entries.items[index].value,
        else => unreachable,
    };
    return .{ .input = input, .output = output };
}

fn modelDictionary(arena: Allocator, custom: runtime.Custom) Error!runtime.Value {
    if (custom.vtable.iterate) |iterate| {
        const items = try iterate(arena, custom.ptr);
        const pairs = try arena.alloc(runtime.PythonEntry, items.len);
        for (items, pairs) |item, *pair| {
            const values = try iterablePair(arena, item);
            pair.* = .{ .key = values[0], .value = values[1] };
        }
        return python.dictionary(arena, pairs);
    }
    // Models without an overridden iterator use BaseModel's field-pair
    // iteration, represented by their raw attribute dictionary.
    const fields = try custom.vtable.raw_value(arena, custom.ptr);
    if (fields != .object) return error.ParseValue;
    return fields;
}

fn iterablePair(arena: Allocator, input: runtime.Value) Error![2]runtime.Value {
    switch (input) {
        .array => |items| if (items.items.len == 2) return .{ items.items[0], items.items[1] },
        .object => |fields| if (fields.count() == 2) return .{
            .{ .string = fields.keys()[0] }, .{ .string = fields.keys()[1] },
        },
        .string => |text| {
            const view = std.unicode.Wtf8View.init(text) catch return error.ParseValue;
            var iterator = view.iterator();
            const first = iterator.nextCodepointSlice() orelse return error.ParseValue;
            const second = iterator.nextCodepointSlice() orelse return error.ParseValue;
            if (iterator.nextCodepointSlice() != null) return error.ParseValue;
            return .{ .{ .string = first }, .{ .string = second } };
        },
        .custom => |custom| {
            if (custom.vtable.iterate) |iterate| {
                const items = try iterate(arena, custom.ptr);
                if (items.len == 2) return .{ items[0], items[1] };
            } else if (python.view(input)) |payload| switch (payload) {
                .tuple, .set, .frozen_set => |items| if (items.len == 2) return .{ items[0], items[1] },
                .dictionary => |entries| if (entries.len == 2) return .{ entries[0].key, entries[1].key },
                .bytes, .bytearray => |bytes| if (bytes.len == 2) return .{
                    .{ .integer = bytes[0] }, .{ .integer = bytes[1] },
                },
                else => {},
            };
        },
        else => {},
    }
    return error.ParseValue;
}
