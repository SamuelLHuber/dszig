//! SDK payloads use the existing mutable pointer-container graph. JSON fields
//! validate their contents; Python Any fields retain the full graph protocol.
//! Borrowed construction is explicit; owning SDK clones share one identity memo
//! across all fields and keep their nodes alive independently of the caller.
const std = @import("std");
const graph = @import("json_graph.zig");
const runtime = @import("runtime_value.zig");
const pyjson = @import("pyjson.zig");
const Payload = @import("request_cache.zig").Payload;
const Allocator = std.mem.Allocator;
pub const Error = Allocator.Error || error{InvalidResponse};

pub const Value = struct {
    root: graph.Value = .null,
    storage: ?*Payload = null,

    /// The caller owns the graph until the enclosing SDK object is cloned.
    pub fn borrowed(root: graph.Value) Value {
        return .{ .root = root };
    }

    pub fn fromValue(allocator: Allocator, input: runtime.Value) Error!Value {
        try validate(input, 0);
        return fromPythonValue(allocator, input);
    }

    /// Python Any input, not a JSON projection. Preserve genuine custom
    /// objects and container kinds through the existing runtime/graph cloners.
    pub fn fromPythonValue(allocator: Allocator, input: runtime.Value) Error!Value {
        switch (input) {
            inline .null, .bool, .integer, .float => |value, tag| {
                const root = @unionInit(graph.Value, @tagName(tag), value);
                return .borrowed(root);
            },
            else => {},
        }
        const storage = try Payload.create(allocator);
        errdefer storage.release();
        // Flat arrays/objects have no recoverable identity; genuine Python
        // boxes do. Clone their leaves once, then import memoized headers into
        // the independently owned arena.
        const arena = storage.allocator();
        var cloner: runtime.CloneContext = .init(arena);
        defer cloner.deinit();
        const copied = try cloner.cloneValue(input);
        storage.root = graph.fromValue(arena, copied) catch |err| return convertError(err);
        return .{ .root = storage.root, .storage = storage };
    }

    /// Wire/runtime values have already erased container identity. This
    /// borrowed conversion is for parsed input, not an alias-recovery API.
    pub fn fromBorrowedValue(arena: Allocator, input: runtime.Value) Error!Value {
        try validate(input, 0);
        return .borrowed(graph.fromValue(arena, input) catch |err| return convertError(err));
    }

    /// A second lease on the same mutable graph, not an independent copy.
    pub fn retain(self: Value) Value {
        if (self.storage) |storage| storage.retain();
        return self;
    }

    pub fn free(self: Value) void {
        if (self.storage) |storage| storage.release();
    }

    pub fn toValue(self: Value, arena: Allocator) Error!runtime.Value {
        return graph.toValue(arena, self.root) catch |err| return convertError(err);
    }

    /// Python json.dumps separators and numeric spelling, streamed directly
    /// from graph nodes without constructing a temporary flat JSON tree.
    pub fn writeJson(self: Value, writer: *std.Io.Writer) pyjson.Error!void {
        return writePython(self.root, writer, 0, false, .compact);
    }

    /// Plain Python json.dumps spelling for public tool-call arguments.
    pub fn writeJsonAscii(self: Value, writer: *std.Io.Writer) pyjson.Error!void {
        return writePython(self.root, writer, 0, true, .compact);
    }

    /// Provider payloads use compact separators without Python's spaces.
    pub fn writeJsonMinified(self: Value, writer: *std.Io.Writer) pyjson.Error!void {
        return writePython(self.root, writer, 0, false, .minified);
    }

    pub fn jsonStringify(self: Value, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try write(self.root, stream, 0);
    }
};

/// Initialized in place: the graph cloner borrows the runtime cloner's address.
/// Deinitialize after success or failure; a failed memo must not be reused.
pub const CloneContext = struct {
    allocator: Allocator,
    storage: ?*Payload = null,
    runtime_cloner: runtime.CloneContext = undefined,
    graph_cloner: graph.CloneContext = undefined,

    pub fn deinit(self: *CloneContext) void {
        if (self.storage) |storage| {
            self.graph_cloner.deinit();
            self.runtime_cloner.deinit();
            storage.release();
        }
    }

    pub fn clone(self: *CloneContext, input: Value) Allocator.Error!Value {
        switch (input.root) {
            .null, .bool, .integer, .float => return .borrowed(input.root),
            else => {},
        }
        if (self.storage == null) {
            const storage = try Payload.create(self.allocator);
            self.storage = storage;
            self.runtime_cloner = .init(storage.allocator());
            self.graph_cloner = .init(storage.allocator(), self.allocator, &self.runtime_cloner);
        }
        const storage = self.storage.?;
        const root = try self.graph_cloner.clone(input.root);
        storage.retain();
        return .{ .root = root, .storage = storage };
    }
};

// One bounded, allocation-free traversal serves validation and both JSON
// writers. Container identity stays in the graph; this is only a token view.
fn Walker(comptime Node: type, comptime python_containers: bool) type {
    return struct {
        pending: ?Node,
        frames: [pyjson.max_depth]Frame = undefined,
        depth: u32 = 0,
        limit: u32 = pyjson.max_depth,
        const Self = @This();
        const Frame = struct { node: Node, index: u32 = 0 };
        const Token = union(enum) {
            scalar: Node,
            array_begin,
            array_end,
            object_begin,
            object_end,
            separator,
            key: struct { text: []const u8, first: bool },
            python_key: struct { value: runtime.Value, first: bool },
        };

        fn containerCount(node: Node) ?usize {
            return switch (node) {
                .array => node.array.items.len,
                .object => node.object.count(),
                else => if (comptime python_containers) switch (node) {
                    .dictionary => node.dictionary.entries.items.len,
                    .sequence => if (node.sequence.kind == .tuple) node.sequence.items.items.len else null,
                    else => null,
                } else null,
            };
        }

        fn arrayLike(node: Node) bool {
            if (node == .array) return true;
            return if (comptime python_containers) node == .sequence else false;
        }

        fn arrayChild(node: Node, index: u32) Node {
            if (node == .array) return node.array.items[index];
            if (comptime python_containers) return node.sequence.items.items[index];
            unreachable;
        }

        fn next(self: *Self) error{ InvalidResponse, JsonTooDeep }!?Token {
            std.debug.assert(self.depth <= self.frames.len);
            std.debug.assert(self.limit <= self.frames.len);
            while (true) {
                if (self.pending) |node| {
                    self.pending = null;
                    if (self.depth >= self.limit) return error.JsonTooDeep;
                    if (containerCount(node)) |count| {
                        if (count > std.math.maxInt(u32)) return error.InvalidResponse;
                        self.frames[self.depth] = .{ .node = node };
                        self.depth += 1;
                        return if (arrayLike(node)) .array_begin else .object_begin;
                    }
                    switch (node) {
                        .null, .bool, .integer, .float, .number_string, .string => return .{ .scalar = node },
                        else => return error.InvalidResponse,
                    }
                }
                if (self.depth == 0) return null;
                const frame = &self.frames[self.depth - 1];
                const count = containerCount(frame.node).?;
                std.debug.assert(frame.index <= count);
                if (frame.index == count) {
                    self.depth -= 1;
                    return if (arrayLike(frame.node)) .array_end else .object_end;
                }
                const index = frame.index;
                frame.index += 1;
                if (arrayLike(frame.node)) {
                    self.pending = arrayChild(frame.node, index);
                    if (index != 0) return .separator;
                } else if (frame.node == .object) {
                    self.pending = frame.node.object.values()[index];
                    return .{ .key = .{
                        .text = frame.node.object.keys()[index],
                        .first = index == 0,
                    } };
                } else if (comptime python_containers) {
                    const entry = frame.node.dictionary.entries.items[index];
                    self.pending = entry.value;
                    return .{ .python_key = .{ .value = entry.key, .first = index == 0 } };
                } else unreachable;
            }
        }
    };
}

pub fn validate(input: anytype, depth: u32) Error!void {
    if (depth >= pyjson.max_depth) return error.InvalidResponse;
    switch (input) {
        .array, .object => {},
        else => return validateScalar(input),
    }
    var walker: Walker(@TypeOf(input), false) = .{
        .pending = input,
        .limit = pyjson.max_depth - depth,
    };
    while (walker.next() catch return error.InvalidResponse) |token|
        if (token == .scalar) try validateScalar(token.scalar);
}

fn validateScalar(input: anytype) Error!void {
    switch (input) {
        .null, .bool, .integer, .string => {},
        .float => |number| if (!std.math.isFinite(number)) return error.InvalidResponse,
        .number_string => |text| {
            if (!@import("value.zig").isJsonNumber(text)) return error.InvalidResponse;
            if (!std.json.isNumberFormattedLikeAnInteger(text)) {
                const number = std.fmt.parseFloat(f64, text) catch return error.InvalidResponse;
                if (!std.math.isFinite(number)) return error.InvalidResponse;
            }
        },
        else => return error.InvalidResponse,
    }
}

fn convertError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidResponse;
}

fn writePython(input: graph.Value, writer: *std.Io.Writer, depth: u32, comptime ascii: bool, comptime style: pyjson.Style) pyjson.Error!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    var walker: Walker(graph.Value, true) = .{ .pending = input, .limit = pyjson.max_depth - depth };
    while (walker.next() catch |err| return switch (err) {
        error.JsonTooDeep => error.JsonTooDeep,
        else => error.ParseValue,
    }) |token| switch (token) {
        .array_begin => try writer.writeByte('['),
        .array_end => try writer.writeByte(']'),
        .object_begin => try writer.writeByte('{'),
        .object_end => try writer.writeByte('}'),
        .separator => try writer.writeAll(if (style == .minified) "," else ", "),
        .key => |key| {
            if (!key.first) try writer.writeAll(if (style == .minified) "," else ", ");
            try pyjson.writeJsonDepth(writer, .{ .string = key.text }, style, 0, ascii);
            try writer.writeAll(if (style == .minified) ":" else ": ");
        },
        .python_key => |key| {
            if (!key.first) try writer.writeAll(if (style == .minified) "," else ", ");
            try pyjson.writeJsonKey(writer, key.value, style, depth + walker.depth, ascii);
            try writer.writeAll(if (style == .minified) ":" else ": ");
        },
        .scalar => |node| switch (node) {
            inline .null, .bool, .integer, .float, .number_string, .string => |value, tag| {
                const scalar = @unionInit(runtime.Value, @tagName(tag), value);
                try pyjson.writeJsonDepth(writer, scalar, style, 0, ascii);
            },
            else => unreachable,
        },
    };
}

fn write(input: graph.Value, stream: *std.json.Stringify, depth: u32) std.Io.Writer.Error!void {
    const max_depth = pyjson.max_depth;
    if (depth >= max_depth) return error.WriteFailed;
    var walker: Walker(graph.Value, false) = .{ .pending = input, .limit = max_depth - depth };
    while (walker.next() catch return error.WriteFailed) |token| switch (token) {
        .array_begin => try stream.beginArray(),
        .array_end => try stream.endArray(),
        .object_begin => try stream.beginObject(),
        .object_end => try stream.endObject(),
        .separator => {},
        .key => |key| try stream.objectField(key.text),
        .python_key => unreachable,
        .scalar => |node| switch (node) {
            .null => try stream.write(null),
            .number_string => |value| try stream.print("{s}", .{value}),
            inline .bool, .integer, .float, .string => |value| try stream.write(value),
            else => unreachable,
        },
    };
}

test "SDK Python JSON preserves dictionary keys and tuples without admitting sets" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const array = try graph.parse(arena,
        \\["値",1]
    );
    const sequence = try arena.create(graph.Sequence);
    sequence.* = .{ .kind = .tuple, .items = array.array.* };
    const dictionary = try arena.create(graph.Dictionary);
    dictionary.* = .{ .entries = .init(arena) };
    try dictionary.entries.append(.{ .key = .{ .integer = 1 }, .value = .{ .string = "one" } });
    try dictionary.entries.append(.{ .key = .null, .value = .{ .string = "none" } });
    try dictionary.entries.append(.{ .key = .{ .string = "東京" }, .value = .{ .sequence = sequence } });
    const payload: Value = .borrowed(.{ .dictionary = dictionary });
    var output: std.Io.Writer.Allocating = .init(arena);
    try payload.writeJsonMinified(&output.writer);
    try testing.expectEqualStrings(
        \\{"1":"one","null":"none","東京":["値",1]}
    , output.written());
    // Python-compatible emission is not permission to pass Python-only
    // container kinds through the finite-JSON admission API.
    try testing.expectError(error.InvalidResponse, validate(payload.root, 0));
    sequence.kind = .set;
    try testing.expectError(error.ParseValue, payload.writeJsonMinified(&output.writer));
}

test "SDK JSON streams nested containers and refuses depth overflow and cycles" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const text =
        \\{"a":[true,1.25,null,{"x":"å"}],"b":[],"c":{}}
    ;
    const root = try graph.parse(arena, text);
    const payload: Value = .borrowed(root);
    var python: std.Io.Writer.Allocating = .init(testing.allocator);
    defer python.deinit();
    try payload.writeJson(&python.writer);
    const expected = (try std.json.parseFromSlice(runtime.Value, arena, text, .{})).value;
    const decoded = (try std.json.parseFromSlice(runtime.Value, arena, python.written(), .{})).value;
    try testing.expect(try @import("inference.zig").jsonEqual(expected, decoded));
    const native = try std.json.Stringify.valueAlloc(testing.allocator, payload, .{});
    defer testing.allocator.free(native);
    const native_decoded = (try std.json.parseFromSlice(runtime.Value, arena, native, .{})).value;
    try testing.expect(try @import("inference.zig").jsonEqual(expected, native_decoded));
    var deep: [129]u8 = undefined;
    @memset(deep[0..64], '[');
    deep[64] = '0';
    @memset(deep[65..], ']');
    const allowed = try graph.parse(arena, deep[1..128]);
    try validate(allowed, 0);
    try validate(try graph.toValue(arena, allowed), 0);
    const denied = try graph.parse(arena, &deep);
    try testing.expectError(error.InvalidResponse, validate(denied, 0));
    const denied_flat = (try std.json.parseFromSlice(runtime.Value, arena, &deep, .{})).value;
    try testing.expectError(error.InvalidResponse, validate(denied_flat, 0));
    const cycle = try arena.create(graph.Array);
    cycle.* = .init(arena);
    try cycle.append(.{ .array = cycle });
    const recursive: Value = .borrowed(.{ .array = cycle });
    try testing.expectError(error.InvalidResponse, validate(recursive.root, 0));
    try testing.expectError(error.JsonTooDeep, recursive.writeJson(&python.writer));
    try testing.expectError(error.WriteFailed, std.json.Stringify.value(recursive, .{}, &python.writer));
}
