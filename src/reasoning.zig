//! DSPy 3.5 Reasoning retains model identity even though its prompt dump is text.
//! Typed values borrow content; runtime boxes own only their box until cloned.
//!
//! Upstream (DSPy 3.5.0): dspy/adapters/types/reasoning.py
const std = @import("std");
const runtime = @import("runtime_value.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Writer = std.Io.Writer;

pub const Reasoning = struct {
    content: []const u8,

    pub const Error = error{InvalidReasoning};

    /// Explicit typed validation accepts actual instances, strings, and objects
    /// with string content. Extra object fields are ignored, not retained.
    /// Returning the typed view borrows content; types.validate preserves the
    /// original runtime box when the input is already an actual Reasoning.
    pub fn fromValue(input: Value) Error!Reasoning {
        return switch (input) {
            .custom => |custom| (custom.get(Reasoning) orelse return error.InvalidReasoning).*,
            .string => |content| .{ .content = content },
            .object => |object| blk: {
                const content = object.get("content") orelse return error.InvalidReasoning;
                if (content != .string) return error.InvalidReasoning;
                break :blk .{ .content = content.string };
            },
            else => error.InvalidReasoning,
        };
    }

    /// Type.format(), Type.serialize_model(), and non-round-trip model dumps
    /// all expose bare content, including an explicitly empty string.
    pub fn format(self: Reasoning) []const u8 {
        return self.content;
    }

    /// A round-trip dump preserves the real model fields. The object storage
    /// belongs to allocator; its field name and content remain borrowed.
    pub fn toValue(self: Reasoning, allocator: Allocator) Allocator.Error!Value {
        var object: runtime.ObjectMap = .empty;
        try object.ensureTotalCapacity(allocator, 1);
        object.putAssumeCapacity("content", .{ .string = self.content });
        return .{ .object = object };
    }

    /// Box an actual class instance without converting it into a string or
    /// adopting its content. An owning clone deep-copies both box and content.
    pub fn toRuntime(self: Reasoning, allocator: Allocator) Allocator.Error!Value {
        const boxed = try allocator.create(Reasoning);
        boxed.* = self;
        return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
    }

    /// The caller retains both the typed slot and its borrowed content.
    pub fn borrowedRuntime(self: *const Reasoning) Value {
        return .{ .custom = .{
            .ptr = self,
            .vtable = &runtime_vtable,
            .owns_box = false,
        } };
    }

    pub fn writeRepr(self: Reasoning, writer: *Writer) Writer.Error!void {
        try pyjson.writeReprString(writer, self.content);
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(Reasoning),
        .type_identity = runtime.typeIdentity(Reasoning),
        .model_name = "Reasoning",
        // The actual tool boundary emits a Type's serialized string bare;
        // nested containers still JSON-encode that same serialized content.
        .is_type = true,
        .bool_value = runtimeTruthy,
        .iterate = iterateRuntime,
        .raw_value = rawRuntimeValue,
        .serialized = serializedRuntimeValue,
        .write_str = writeRuntimeStr,
        .write_repr = writeRuntimeRepr,
        .write_json = writeRuntimeJson,
        .write_serialized_json = writeRuntimeSerializedJson,
        .clone = cloneRuntime,
        .deinit = deinitRuntime,
        .eql = equalRuntime,
        .eql_value = equalRuntimeValue,
    };

    fn runtimeReasoning(ptr: *const anyopaque) *const Reasoning {
        return @ptrCast(@alignCast(ptr));
    }

    fn runtimeTruthy(ptr: *const anyopaque) error{ParseValue}!bool {
        return runtimeReasoning(ptr).content.len != 0;
    }

    fn iterateRuntime(allocator: Allocator, ptr: *const anyopaque) runtime.CustomError![]Value {
        const content = runtimeReasoning(ptr).content;
        const view = std.unicode.Wtf8View.init(content) catch return error.ParseValue;
        var count: usize = 0;
        for (content) |byte| {
            if (byte & 0xc0 != 0x80) count += 1;
        }
        const result = try allocator.alloc(Value, count);
        var iterator = view.iterator();
        for (result) |*slot| slot.* = .{ .string = iterator.nextCodepointSlice().? };
        return result;
    }

    fn rawRuntimeValue(allocator: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return runtimeReasoning(ptr).toValue(allocator);
    }

    fn serializedRuntimeValue(_: Allocator, ptr: *const anyopaque, _: u32) runtime.CustomError!Value {
        return .{ .string = runtimeReasoning(ptr).format() };
    }

    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        try writer.writeAll(runtimeReasoning(ptr).content);
    }

    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, _: u32) runtime.CustomWriteError!void {
        try runtimeReasoning(ptr).writeRepr(writer);
    }

    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(runtimeReasoning(ptr).*);
    }

    fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, _: usize) runtime.CustomWriteError!void {
        try pyjson.writeJsonString(writer, runtimeReasoning(ptr).content);
    }

    fn cloneRuntime(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const allocator = context.allocator;
        const copy = try allocator.create(Reasoning);
        errdefer allocator.destroy(copy);
        copy.* = .{ .content = try allocator.dupe(u8, runtimeReasoning(ptr).content) };
        return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
    }

    fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        const reasoning = runtimeReasoning(ptr);
        if (owns_data) allocator.free(reasoning.content);
        allocator.destroy(reasoning);
    }

    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
        return std.mem.eql(u8, runtimeReasoning(left).content, runtimeReasoning(right).content);
    }

    fn equalRuntimeValue(ptr: *const anyopaque, other: Value) runtime.CustomError!bool {
        return other == .string and std.mem.eql(u8, runtimeReasoning(ptr).content, other.string);
    }
};

const testing = std.testing;

test "Reasoning validation retains actual identity and explicitly converts strings and content objects" {
    const types = @import("types.zig");
    const value = @import("value.zig");
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const actual = try (Reasoning{ .content = "  雪\n  " }).toRuntime(arena);
    const validated = try types.validate(Reasoning, arena, actual);
    try testing.expectEqual(actual.custom.ptr, validated.custom.ptr);
    const typed = try Reasoning.fromValue(actual);
    try testing.expectEqualStrings("  雪\n  ", typed.content);
    try testing.expectEqual(actual.custom.get(Reasoning).?.content.ptr, typed.content.ptr);
    const converted = try value.fromValue(arena, Reasoning, actual);
    try testing.expectEqual(typed.content.ptr, converted.content.ptr);

    const dictionary = try std.json.parseFromSliceLeaky(Value, arena, "{\"content\":\"  雪\\n  \",\"extra\":{\"content\":\"ignored\"}}", .{});
    const from_dictionary = try types.validate(Reasoning, arena, dictionary);
    try testing.expect(from_dictionary == .custom);
    try testing.expect(from_dictionary.custom.ptr != actual.custom.ptr);
    try testing.expect(try value.pyEqual(actual, from_dictionary));
    try testing.expectEqual(@as(usize, 2), dictionary.object.count());
    const raw = try from_dictionary.custom.rawValue(arena);
    try testing.expectEqual(@as(usize, 1), raw.object.count());
    try testing.expect(raw.object.get("extra") == null);
    try testing.expect(!try value.pyEqual(actual, dictionary));
    for ([_][]const u8{ "", "text", "雪😀e\u{301}" }) |content| {
        const plain: Value = .{ .string = content };
        const from_string = try types.validate(Reasoning, arena, plain);
        try testing.expect(from_string == .custom);
        try testing.expectEqualStrings(content, from_string.custom.get(Reasoning).?.content);
        try testing.expect(plain == .string);
        const fields = try (Reasoning{ .content = content }).toValue(arena);
        const from_fields = try Reasoning.fromValue(fields);
        try testing.expectEqualStrings(content, from_fields.content);
    }
    try testing.expect((try types.validate(?Reasoning, arena, .null)) == .null);

    var items: runtime.Array = .init(arena);
    try items.append(actual);
    try items.append(.{ .string = "new" });
    try items.append(actual);
    const list = try types.validate([]const Reasoning, arena, .{ .array = items });
    try testing.expectEqual(actual.custom.ptr, list.array.items[0].custom.ptr);
    try testing.expectEqual(actual.custom.ptr, list.array.items[2].custom.ptr);
    try testing.expectEqualStrings("new", list.array.items[1].custom.get(Reasoning).?.content);
    try testing.expect(items.items[1] == .string);
}

test "Reasoning rejects null, invalid content and unrelated native instances" {
    const types = @import("types.zig");
    const File = @import("file.zig").File;
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    for ([_][]const u8{
        "null",               "false",               "42",               "1.5",              "[]",                                     "{}", "{\"other\":\"text\"}",
        "{\"content\":null}", "{\"content\":false}", "{\"content\":42}", "{\"content\":[]}", "{\"content\":{\"content\":\"nested\"}}",
    }) |json| {
        const input = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
        try testing.expectError(error.InvalidReasoning, Reasoning.fromValue(input));
        try testing.expectError(error.ParseValue, types.validate(Reasoning, arena, input));
    }
    const file = try (try File.fromFileId("file-id", null)).toRuntime(arena);
    try testing.expectError(error.InvalidReasoning, Reasoning.fromValue(file));
    try testing.expectError(error.ParseValue, types.validate(Reasoning, arena, file));
    const nested_model = try (Reasoning{ .content = "nested" }).toRuntime(arena);
    var fields: runtime.ObjectMap = .empty;
    try fields.put(arena, "content", nested_model);
    try testing.expectError(error.InvalidReasoning, Reasoning.fromValue(.{ .object = fields }));
}

test "Reasoning equality is symmetric with strings but not lookalike dictionaries" {
    const value = @import("value.zig");
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    for ([_][]const u8{ "", "think", "雪😀\n" }) |content| {
        const typed: Reasoning = .{ .content = content };
        const actual = try typed.toRuntime(arena);
        const equal = try typed.toRuntime(arena);
        const different = try (Reasoning{ .content = "not equal" }).toRuntime(arena);
        const plain: Value = .{ .string = content };
        const dictionary = try typed.toValue(arena);
        try testing.expect(try value.pyEqual(actual, actual));
        try testing.expect(try value.pyEqual(actual, equal));
        try testing.expect(try value.pyEqual(equal, actual));
        try testing.expect(!try value.pyEqual(actual, different));
        try testing.expect(try value.pyEqual(actual, plain));
        try testing.expect(try value.pyEqual(plain, actual));
        try testing.expect(!try value.pyEqual(actual, dictionary));
        try testing.expect(!try value.pyEqual(dictionary, actual));
        for ([_]Value{ .null, .{ .bool = false }, .{ .integer = 0 }, .{ .string = "not equal" } }) |other| {
            try testing.expect(!try value.pyEqual(actual, other));
            try testing.expect(!try value.pyEqual(other, actual));
        }
        var actual_list: runtime.Array = .init(arena);
        try actual_list.append(actual);
        var string_list: runtime.Array = .init(arena);
        try string_list.append(plain);
        try testing.expect(try value.pyEqual(.{ .array = actual_list }, .{ .array = string_list }));
        try testing.expect(try value.pyEqual(.{ .array = string_list }, .{ .array = actual_list }));
    }
}

test "Reasoning raw dumps retain fields while normal nested dumps contain content" {
    const value = @import("value.zig");
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const reasoning: Reasoning = .{ .content = "think 雪" };
    const actual = try reasoning.toRuntime(arena);
    const dictionary = try reasoning.toValue(arena);
    try testing.expectEqualStrings("{\"content\":\"think 雪\"}", try std.json.Stringify.valueAlloc(arena, actual, .{}));
    try testing.expect(try value.pyEqual(dictionary, try actual.custom.rawValue(arena)));
    const normal = try value.serializedValue(arena, actual);
    try testing.expect(normal == .string);
    try testing.expectEqualStrings(reasoning.content, normal.string);
    try testing.expect(actual == .custom);

    var items: runtime.Array = .init(arena);
    try items.append(actual);
    try items.append(.{ .string = reasoning.content });
    try items.append(dictionary);
    try items.append(.null);
    var fields: runtime.ObjectMap = .empty;
    try fields.put(arena, "values", .{ .array = items });
    const nested: Value = .{ .object = fields };
    try testing.expectEqualStrings(
        "{\"values\":[{\"content\":\"think 雪\"},\"think 雪\",{\"content\":\"think 雪\"},null]}",
        try std.json.Stringify.valueAlloc(arena, nested, .{}),
    );
    const serialized = try value.serializedValue(arena, nested);
    const serialized_items = serialized.object.get("values").?.array.items;
    try testing.expect(serialized_items[0] == .string);
    try testing.expect(serialized_items[1] == .string);
    try testing.expect(serialized_items[2] == .object);
    try testing.expect(try value.pyEqual(dictionary, serialized_items[2]));
    try testing.expect(items.items[0] == .custom);
    var streamed: Writer.Allocating = .init(arena);
    try pyjson.writeJson(&streamed.writer, nested, .minified);
    try testing.expectEqualStrings(
        "{\"values\":[\"think 雪\",\"think 雪\",{\"content\":\"think 雪\"},null]}",
        streamed.written(),
    );
    try testing.expectEqualStrings(try std.json.Stringify.valueAlloc(arena, serialized, .{}), streamed.written());
}

test "Reasoning str stays raw and repr follows Python content quoting and Unicode escaping" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const Case = struct { content: []const u8, repr: []const u8 };
    const cases = [_]Case{
        .{ .content = "", .repr = "''" },
        .{ .content = "plain", .repr = "'plain'" },
        .{ .content = "a'b", .repr = "\"a'b\"" },
        .{ .content = "a\"b", .repr = "'a\"b'" },
        .{ .content = "a'\"b\\", .repr = "'a\\'\"b\\\\'" },
        .{ .content = "雪😀e\u{301}", .repr = "'雪😀e\u{301}'" },
        .{ .content = "\x00\x08\x0c\n\r\t\x1f\x7f\u{2028}雪😀", .repr = "'\\x00\\x08\\x0c\\n\\r\\t\\x1f\\x7f\\u2028雪😀'" },
    };
    for (cases) |case| {
        const actual = try (Reasoning{ .content = case.content }).toRuntime(arena);
        var text: Writer.Allocating = .init(arena);
        try pyjson.writeStr(&text.writer, actual);
        try testing.expectEqualStrings(case.content, text.written());
        var repr: Writer.Allocating = .init(arena);
        try pyjson.writeRepr(&repr.writer, actual, 0);
        try testing.expectEqualStrings(case.repr, repr.written());
        var list: runtime.Array = .init(arena);
        try list.append(actual);
        var nested: Writer.Allocating = .init(arena);
        try pyjson.writeStr(&nested.writer, .{ .array = list });
        try testing.expectEqualStrings(try std.fmt.allocPrint(arena, "[{s}]", .{case.repr}), nested.written());
        var json: Writer.Allocating = .init(arena);
        try pyjson.writeJson(&json.writer, actual, .compact);
        const decoded = try std.json.parseFromSliceLeaky(Value, arena, json.written(), .{});
        try testing.expect(decoded == .string);
        try testing.expectEqualStrings(case.content, decoded.string);
    }
}

test "Reasoning owning nested clones survive source storage mutation and release" {
    const inference = @import("inference.zig");
    const value = @import("value.zig");
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);
    const allocator = debug_allocator.allocator();
    var source: std.heap.ArenaAllocator = .init(testing.allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const expected = "clone 雪😀\n\"'\\\x00";
    const content = try source.allocator().dupe(u8, expected);
    const reasoning: Reasoning = .{ .content = content };
    var items: runtime.Array = .init(source.allocator());
    try items.append(try reasoning.toRuntime(source.allocator()));
    try items.append(try reasoning.toValue(source.allocator()));
    try items.append(.{ .string = content });
    var fields: runtime.ObjectMap = .empty;
    try fields.put(source.allocator(), "values", .{ .array = items });
    const owned = try inference.cloneJsonValue(allocator, .{ .object = fields });
    defer inference.freeJsonValue(allocator, owned);
    const copies = owned.object.get("values").?.array.items;
    try testing.expect(copies[0] == .custom);
    try testing.expect(copies[0].custom.ptr != items.items[0].custom.ptr);
    try testing.expect(copies[0].custom.get(Reasoning).?.content.ptr != content.ptr);
    @memset(content, 'x');
    try testing.expectEqualStrings(expected, copies[0].custom.get(Reasoning).?.content);
    source.deinit();
    source_live = false;

    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const cloned = try Reasoning.fromValue(copies[0]);
    try testing.expectEqualStrings(expected, cloned.content);
    try testing.expectEqualStrings(expected, copies[1].object.get("content").?.string);
    try testing.expectEqualStrings(expected, copies[2].string);
    try testing.expect(try value.pyEqual(copies[0], copies[2]));
    try testing.expect(!try value.pyEqual(copies[0], copies[1]));
    const serialized = try value.serializedValue(arena, owned);
    try testing.expectEqualStrings(expected, serialized.object.get("values").?.array.items[0].string);
    var rendered: Writer.Allocating = .init(arena);
    try pyjson.writeStr(&rendered.writer, copies[0]);
    try testing.expectEqualStrings(expected, rendered.written());
    const second_copy = try inference.cloneJsonValue(allocator, copies[0]);
    defer inference.freeJsonValue(allocator, second_copy);
    try testing.expect(try value.pyEqual(second_copy, copies[0]));
}

test "Reasoning clone allocation failures preserve the borrowed source" {
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn clone(allocator: Allocator) !void {
            const inference = @import("inference.zig");
            const original = try (Reasoning{ .content = "borrowed 雪" }).toRuntime(testing.allocator);
            defer original.custom.deinit(testing.allocator);
            const copy = try inference.cloneJsonValue(allocator, original);
            defer inference.freeJsonValue(allocator, copy);
            try testing.expectEqualStrings("borrowed 雪", copy.custom.get(Reasoning).?.content);
            try testing.expectEqualStrings("borrowed 雪", original.custom.get(Reasoning).?.content);
        }
    }.clone, .{});
}
