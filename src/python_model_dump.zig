//! Default Python-mode BaseModel projection. This is deliberately not JSON:
//! bytes, Enum, Decimal and unknown objects retain their native identities.
//! Containers are snapshots, not aliases of mutable input container headers.
//! All storage comes from the caller's arena (including on failure). Use a
//! startup-backed allocator when projection must not allocate from the heap.

const std = @import("std");
const runtime = @import("runtime_value.zig");
const python = @import("python_value.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Error = runtime.CustomError;

/// A finite native work budget, independent of available arena memory.
/// Depth includes model serializers, so a serializer returning itself refuses.
pub const max_steps = 1 << 20;

const Frame = struct {
    output: *Value,
    depth: u32,
    cursor: usize = 0,
    state: union(enum) {
        value: Value,
        array: []const Value,
        object: runtime.ObjectMap,
        sequence: struct { original: Value, input: []const Value, output: []Value },
        dictionary: struct { input: []const runtime.PythonEntry, output: []runtime.PythonEntry },
    },
};
const Child = struct { input: Value, output: *Value };

/// Internal recursive-model projection implemented with a fixed-depth stack.
/// A public BaseModel entry point must first check CustomVTable.model_dump.
pub fn project(arena: Allocator, input: Value) Error!Value {
    var result: Value = undefined;
    var stack: [pyjson.max_depth]Frame = undefined;
    stack[0] = .{ .output = &result, .depth = 0, .state = .{ .value = input } };
    var count: usize = 1;
    for (0..max_steps) |_| {
        std.debug.assert(count > 0);
        std.debug.assert(count <= stack.len);
        const frame = &stack[count - 1];
        if (frame.depth >= pyjson.max_depth) return error.ParseValue;
        if (frame.state == .value) {
            if (try enter(arena, frame)) {
                count -= 1;
                if (count == 0) return result;
            }
        } else if (nextChild(frame)) |child| {
            if (count == stack.len) return error.ParseValue;
            stack[count] = .{
                .output = child.output,
                .depth = frame.depth + 1,
                .state = .{ .value = child.input },
            };
            count += 1;
        } else {
            try finish(arena, frame);
            count -= 1;
            if (count == 0) return result;
        }
    }
    return error.ParseValue;
}

/// True means a leaf has been published; false keeps working on this frame.
fn enter(arena: Allocator, frame: *Frame) Error!bool {
    std.debug.assert(frame.state == .value);
    std.debug.assert(frame.depth < pyjson.max_depth);
    const input = frame.state.value;
    switch (input) {
        .custom => |custom| {
            if (custom.vtable.model_dump) |dump| {
                frame.state = .{ .value = try dump(arena, custom.ptr) };
                frame.depth += 1;
                return false;
            }
            if (python.view(input)) |payload| switch (payload) {
                .tuple, .set, .frozen_set => |items| {
                    frame.state = .{ .sequence = .{
                        .original = input,
                        .input = items,
                        .output = try arena.alloc(Value, items.len),
                    } };
                    return false;
                },
                .dictionary => |entries| {
                    frame.state = .{ .dictionary = .{
                        .input = entries,
                        .output = try arena.alloc(runtime.PythonEntry, entries.len),
                    } };
                    return false;
                },
                else => {},
            };
        },
        .array => |array| {
            var output: runtime.Array = .init(arena);
            try output.resize(array.items.len);
            frame.output.* = .{ .array = output };
            frame.state = .{ .array = array.items };
            return false;
        },
        .object => |object| {
            var output: runtime.ObjectMap = .empty;
            try output.ensureTotalCapacity(arena, object.count());
            for (object.keys()) |key| output.putAssumeCapacity(key, .null);
            frame.output.* = .{ .object = output };
            frame.state = .{ .object = object };
            return false;
        },
        else => {},
    }
    frame.output.* = input;
    return true;
}

fn nextChild(frame: *Frame) ?Child {
    const index = frame.cursor;
    const child: Child = switch (frame.state) {
        .array => |items| if (index < items.len)
            .{ .input = items[index], .output = &frame.output.array.items[index] }
        else
            return null,
        .object => |object| if (index < object.count())
            .{ .input = object.values()[index], .output = &frame.output.object.values()[index] }
        else
            return null,
        .sequence => |sequence| if (index < sequence.input.len)
            .{ .input = sequence.input[index], .output = &sequence.output[index] }
        else
            return null,
        .dictionary => |dictionary| blk: {
            const pair = index / 2;
            if (pair >= dictionary.input.len) return null;
            break :blk if (index % 2 == 0)
                .{ .input = dictionary.input[pair].key, .output = &dictionary.output[pair].key }
            else
                .{ .input = dictionary.input[pair].value, .output = &dictionary.output[pair].value };
        },
        .value => unreachable,
    };
    frame.cursor += 1;
    return child;
}

fn finish(arena: Allocator, frame: *Frame) Error!void {
    switch (frame.state) {
        .sequence => |sequence| {
            std.debug.assert(frame.cursor == sequence.input.len);
            frame.output.* = try python.sequenceWithItems(arena, sequence.original, sequence.output);
        },
        .dictionary => |dictionary| {
            std.debug.assert(frame.cursor / 2 == dictionary.input.len);
            frame.output.* = try python.dictionary(arena, dictionary.output);
        },
        .array, .object => {},
        .value => unreachable,
    }
}

test "Python model dumps snapshot containers without JSON-coercing native leaves" {
    const Noul = @import("decision.zig").Noul;
    const Reasoning = @import("reasoning.zig").Reasoning;
    const Code = @import("code.zig").Code;
    const Label = @import("enum.zig").Enum("Label", &.{
        .{ .name = "one", .value = .{ .integer = 1 } },
    });
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const nested = try (Reasoning{ .content = "nested" }).toRuntime(a);
    const code = try (Code{ .code = "x" }).toRuntime(a);
    const tuple = try python.tuple(a, try a.dupe(Value, &.{ nested, code }));
    const binary = try python.bytes(a, "\xff");
    const decimal = try python.decimal(a, "1.25");
    const label = try Label.member("one");
    var list: runtime.Array = .init(a);
    try list.append(try (Reasoning{ .content = "before" }).toRuntime(a));
    var payload: runtime.ObjectMap = .empty;
    try payload.put(a, "bytes", binary);
    try payload.put(a, "tuple", tuple);
    try payload.put(a, "decimal", decimal);
    try payload.put(a, "enum", label);
    try payload.put(a, "list", .{ .array = list });
    var fields: runtime.ObjectMap = .empty;
    try fields.put(a, "value", .{ .object = payload });
    try fields.put(a, "confidence", .{ .float = 0.75 });
    const model = try Noul.modelConstruct(a, .{ .object = fields });
    const result = try model.custom.modelDump(a);
    @constCast(nested.custom.get(Reasoning).?).content = "changed";
    list.items[0] = .null;
    const values = result.object.get("value").?.object;
    try std.testing.expect(python.sameIdentity(binary, values.get("bytes").?));
    try std.testing.expect(python.sameIdentity(decimal, values.get("decimal").?));
    try std.testing.expect(python.sameIdentity(label, values.get("enum").?));
    const projected_tuple = python.view(values.get("tuple").?).?.tuple;
    try std.testing.expectEqualStrings("nested", projected_tuple[0].string);
    try std.testing.expectEqualStrings("x", projected_tuple[1].string);
    try std.testing.expectEqualStrings("before", values.get("list").?.array.items[0].string);
    try std.testing.expectEqual(@as(f64, 0.75), result.object.get("confidence").?.float);
    try std.testing.expect(!result.object.contains("probability"));
}

test "Python model dumps keep nonfinite fields and refuse recursive models and nonmodels" {
    const BoundingBox = @import("bounding_box.zig").BoundingBox;
    const Noul = @import("decision.zig").Noul;
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const model = try (BoundingBox{
        .top = std.math.nan(f64),
        .left = 0,
        .bottom = 1,
        .right = 1,
    }).toRuntime(a);
    const dumped = try model.custom.modelDump(a);
    try std.testing.expect(std.math.isNan(dumped.object.get("top").?.float));
    const encoded = try model.custom.serialize(a);
    try std.testing.expect(encoded.object.get("top").? == .null);
    const binary = try python.bytes(a, "\xff");
    try std.testing.expectError(error.ParseValue, binary.custom.modelDump(a));
    const cycle = try Noul.modelConstruct(a, .{ .object = .empty });
    try Noul.setField(cycle, "value", cycle);
    try std.testing.expectError(error.ParseValue, cycle.custom.modelDump(a));
}
