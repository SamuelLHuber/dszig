//! Genuine invocation results. Copies alias one arena-owned instance; clone
//! creates independent fields, completions and usage using native clone protocols.
const std = @import("std");
const value = @import("value.zig");
const runtime = @import("runtime_value.zig");
pub const Graph = @import("json_graph.zig");

test "prediction clone preserves mutable native aliases across fields completions and usage" {
    const python = @import("python_value.zig");
    const Fields = struct { meta: runtime.Value, extra: runtime.Value };
    var owned: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer owned.deinit();
    const target = owned.allocator();
    var copy: Prediction(Fields) = undefined;
    {
        var borrowed: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer borrowed.deinit();
        const source = borrowed.allocator();
        const bytes = try python.bytearray(source, "a");
        const rows = try source.alloc(Fields, 1);
        rows[0] = .{ .meta = bytes, .extra = bytes };
        const prediction = try Prediction(Fields).fromCompletions(source, rows);
        prediction.setLMUsage(.{ .custom = bytes.custom });
        copy = try prediction.clone(target);
        try std.testing.expect(copy.fields().meta.custom.ptr != bytes.custom.ptr);
    }
    const leaf = copy.fields().meta;
    try std.testing.expect(copy.fields().extra.custom.ptr == leaf.custom.ptr);
    try std.testing.expect(copy.completions().?[0].meta.custom.ptr == leaf.custom.ptr);
    try std.testing.expect(copy.getLMUsage().?.custom.ptr == leaf.custom.ptr);
    try python.bytearrayAppend(target, leaf, 'b');
    try std.testing.expectEqualStrings("ab", python.view(copy.fields().extra).?.bytearray);
    try std.testing.expectEqualStrings("ab", python.view(copy.completions().?[0].meta).?.bytearray);
    try std.testing.expectEqualStrings("ab", python.view(.{ .custom = copy.getLMUsage().?.custom }).?.bytearray);
}

test "record completion header replacement preserves independent rows and shared nested values" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const python = @import("python_value.zig");
    const bytes = try python.bytearray(arena, "a");
    const rows = try arena.alloc(runtime.ObjectMap, 1);
    rows[0] = .empty;
    try rows[0].put(arena, "first", bytes);
    try rows[0].put(arena, "second", bytes);
    const prediction = try Prediction(runtime.ObjectMap).fromCompletions(arena, rows);
    prediction.fields().getPtr("first").?.* = .{ .integer = 5 };
    try std.testing.expectEqual(@as(i64, 5), prediction.get("first").?.integer);
    try std.testing.expect(rows[0].get("first").?.custom.ptr == bytes.custom.ptr);
    try python.bytearrayAppend(arena, rows[0].get("first").?, 'b');
    try std.testing.expectEqualStrings("ab", python.view(prediction.get("second").?).?.bytearray);
}

pub fn isPrediction(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => @hasDecl(T, "is_dszig_prediction") and T.is_dszig_prediction,
        else => false,
    };
}

pub fn Prediction(comptime Output: type) type {
    return struct {
        const Self = @This();
        pub const is_dszig_prediction = true;
        pub const Fields = Output;
        /// One instance's storage. `init`/`fromCompletions` own it in an
        /// arena; `borrowed` wraps one the caller provisions and keeps stable.
        pub const State = struct {
            output: Output,
            outputs: ?[]Output = null,
            usage: ?Graph.Value = null,
        };
        state: *State,

        /// Handle onto caller-provisioned state, without allocating. The
        /// state, its fields and its usage graph must stay alive and unmoved
        /// while any copy of the handle is used; every copy aliases this one
        /// instance, so rewriting the state rewrites what they all observe.
        /// `clone` is the way to keep a result across such a rewrite.
        pub fn borrowed(state: *State) Self {
            return .{ .state = state };
        }

        /// Adopt output fields already owned by arena (as parsed outputs are).
        pub fn init(arena: std.mem.Allocator, output: Output) std.mem.Allocator.Error!Self {
            const state = try arena.create(State);
            state.* = .{ .output = output };
            return .{ .state = state };
        }

        /// Adopt all parsed completions without copying their fields.
        pub fn fromCompletions(arena: std.mem.Allocator, outputs: []Output) std.mem.Allocator.Error!Self {
            std.debug.assert(outputs.len > 0);
            const state = try arena.create(State);
            state.* = .{
                .output = if (Output == runtime.ObjectMap) try outputs[0].clone(arena) else outputs[0],
                .outputs = outputs,
            };
            return .{ .state = state };
        }

        pub fn fields(self: Self) *Output {
            return &self.state.output;
        }

        /// Null for ordinary construction; populated only by fromCompletions.
        /// Replacing a field does not rewrite the original completion rows.
        pub fn completions(self: Self) ?[]const Output {
            return self.state.outputs;
        }

        /// Record Predictions only: `prediction[key]`, null for a missing key.
        pub fn get(self: Self, key: []const u8) ?runtime.Value {
            if (Output != runtime.ObjectMap) @compileError("dszig: Prediction.get needs record fields; use fields() for " ++ @typeName(Output));
            return self.state.output.get(key);
        }

        pub fn getLMUsage(self: Self) ?Graph.Value {
            return self.state.usage;
        }

        /// Adopt an optional graph owned for at least this instance's lifetime.
        pub fn setLMUsage(self: Self, usage: ?Graph.Value) void {
            self.state.usage = usage;
        }

        pub fn clone(self: Self, arena: std.mem.Allocator) value.ConvertError!Self {
            var cloner: runtime.CloneContext = .init(arena);
            defer cloner.deinit();
            const result = try init(arena, try cloneFields(&cloner, arena, self.state.output));
            if (self.state.outputs) |source_outputs| {
                const outputs = try arena.alloc(Output, source_outputs.len);
                for (source_outputs, outputs) |source, *target|
                    target.* = try cloneFields(&cloner, arena, source);
                result.state.outputs = outputs;
            }
            if (self.state.usage) |usage|
                result.setLMUsage(try Graph.cloneWithRuntimeContext(arena, arena, usage, &cloner));
            return result;
        }

        fn cloneFields(cloner: *runtime.CloneContext, arena: std.mem.Allocator, source: Output) value.ConvertError!Output {
            const encoded = try value.toValue(arena, Output, source);
            return value.fromValue(arena, Output, try cloner.cloneValue(encoded));
        }
    };
}
