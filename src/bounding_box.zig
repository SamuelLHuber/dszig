//! DSPy 3.5's axis-aligned normalized image rectangle.
//! Construct with init or fromValue to enforce finite edges and positive area.
const std = @import("std");
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");
const Value = runtime.Value;
const Writer = std.Io.Writer;
const pyjson = @import("pyjson.zig");

pub const BoundingBox = struct {
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,

    pub const Error = error{InvalidBoundingBox};

    pub const edge_descriptions = [_][]const u8{
        "Top edge as a fraction of image height, from 0 to 1.",
        "Bottom edge as a fraction of image height, from 0 to 1.",
        "Left edge as a fraction of image width, from 0 to 1.",
        "Right edge as a fraction of image width, from 0 to 1.",
    };
    pub const schema_description =
        "An axis-aligned rectangle in normalized image coordinates.\n\n" ++
        "Edges are finite numbers in [0, 1], measured from the top-left corner.\n" ++
        "Multiply left/right by the image width and top/bottom by its height to\n" ++
        "obtain pixel coordinates. Use the dimensions of the upright image actually\n" ++
        "sent to the model, not those of an original image before resizing/cropping.\n\n" ++
        "Boxes have positive area: top < bottom and left < right. Invalid outputs\n" ++
        "are rejected, never reordered or clamped. Provider confidence is not part\n" ++
        "of the geometry; native TypeSafe evidence remains in the LM response.";

    pub fn init(top: f64, bottom: f64, left: f64, right: f64) Error!BoundingBox {
        const box: BoundingBox = .{ .top = top, .bottom = bottom, .left = left, .right = right };
        for ([_]f64{ top, bottom, left, right }) |coordinate| {
            if (!(coordinate >= 0 and coordinate <= 1)) return error.InvalidBoundingBox;
        }
        if (!(top < bottom and left < right)) return error.InvalidBoundingBox;
        return box;
    }

    /// Pydantic strict float fields accept JSON integers, but not bool or text.
    pub fn fromValue(value: Value) Error!BoundingBox {
        return decode(value, false);
    }

    /// XMLAdapter explicitly overrides model strictness for decoded containers.
    pub fn fromXmlValue(value: Value) Error!BoundingBox {
        return decode(value, true);
    }

    fn decode(value: Value, comptime lax: bool) Error!BoundingBox {
        if (value == .custom) {
            // Pydantic reuses an existing instance without field validators,
            // but its after-model validator still checks this exact ordering.
            const box = (value.custom.get(BoundingBox) orelse return error.InvalidBoundingBox).*;
            if (box.top >= box.bottom or box.left >= box.right) return error.InvalidBoundingBox;
            return box;
        }
        if (value != .object or value.object.count() != 4) return error.InvalidBoundingBox;
        return init(
            try edge(value.object.get("top") orelse return error.InvalidBoundingBox, lax),
            try edge(value.object.get("bottom") orelse return error.InvalidBoundingBox, lax),
            try edge(value.object.get("left") orelse return error.InvalidBoundingBox, lax),
            try edge(value.object.get("right") orelse return error.InvalidBoundingBox, lax),
        );
    }

    fn edge(value: Value, comptime lax: bool) Error!f64 {
        return switch (value) {
            .float => |number| number,
            .integer => |number| @floatFromInt(number),
            .number_string => |text| std.fmt.parseFloat(f64, text) catch error.InvalidBoundingBox,
            .string => |text| if (lax) std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\n\r")) catch error.InvalidBoundingBox else error.InvalidBoundingBox,
            .bool => |boolean| if (lax) @floatFromInt(@intFromBool(boolean)) else error.InvalidBoundingBox,
            else => error.InvalidBoundingBox,
        };
    }

    /// DSPy's format(): standard JSON spacing and binary64 float rendering.
    pub fn format(self: BoundingBox, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        inline for (.{ "top", "bottom", "left", "right" }, 0..) |name, index| {
            try writer.writeAll(if (index == 0) "{\"" ++ name ++ "\": " else ", \"" ++ name ++ "\": ");
            try pyjson.writeFloat(writer, @field(self, name), .json);
        }
        try writer.writeByte('}');
    }

    /// Preserve actual model identity in erased records without changing toValue.
    pub fn toRuntime(self: BoundingBox, allocator: Allocator) Allocator.Error!Value {
        const boxed = try allocator.create(BoundingBox);
        boxed.* = self;
        return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(BoundingBox),
        .type_identity = runtime.typeIdentity(BoundingBox),
        .model_name = "BoundingBox",
        .is_type = true,
        .hashable = true,
        .raw_value = rawRuntimeValue,
        .serialized = serializedRuntimeValue,
        .write_str = writeRuntimeStr,
        .write_repr = writeRuntimeRepr,
        .write_json = writeRuntimeJson,
        .write_serialized_json = writeRuntimeSerializedJson,
        .clone = cloneRuntime,
        .deinit = deinitRuntime,
        .eql = equalRuntime,
    };

    fn runtimeBox(ptr: *const anyopaque) *const BoundingBox {
        return @ptrCast(@alignCast(ptr));
    }

    fn rawRuntimeValue(allocator: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return runtimeBox(ptr).toValue(allocator);
    }

    fn serializedRuntimeValue(allocator: Allocator, ptr: *const anyopaque, _: u32) runtime.CustomError!Value {
        // serialize_for_json uses Pydantic's JSON mode: unlike format() and
        // Python-mode round trips, unvalidated nonfinite fields become null.
        var value = try runtimeBox(ptr).toValue(allocator);
        inline for (.{ "top", "bottom", "left", "right" }) |name| {
            const edge_value = value.object.getPtr(name).?;
            if (!std.math.isFinite(edge_value.float)) edge_value.* = .null;
        }
        return value;
    }

    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        try writeModelDisplay(writer, runtimeBox(ptr).*, false);
    }

    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, _: u32) runtime.CustomWriteError!void {
        try writeModelDisplay(writer, runtimeBox(ptr).*, true);
    }

    fn writeModelDisplay(writer: *Writer, box: BoundingBox, comptime repr: bool) Writer.Error!void {
        if (repr) try writer.writeAll("BoundingBox(");
        inline for (.{ "top", "bottom", "left", "right" }, 0..) |name, index| {
            if (index != 0) try writer.writeAll(if (repr) ", " else " ");
            try writer.writeAll(name ++ "=");
            try pyjson.writeFloat(writer, @field(box, name), .python);
        }
        if (repr) try writer.writeByte(')');
    }

    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(runtimeBox(ptr).*);
    }

    fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
        if (depth >= pyjson.max_depth - 1) return error.JsonTooDeep;
        const box = runtimeBox(ptr);
        try writer.writeByte('{');
        inline for (.{ "top", "bottom", "left", "right" }, 0..) |name, index| {
            switch (style) {
                .compact => if (index != 0) try writer.writeAll(", "),
                .minified => if (index != 0) try writer.writeByte(','),
                .indent2, .orjson_indent2 => {
                    if (index != 0) try writer.writeByte(',');
                    try writer.writeByte('\n');
                    try writer.splatByteAll(' ', (depth + 1) * 2);
                },
            }
            try writer.writeAll("\"" ++ name);
            try writer.writeAll(if (style == .minified) "\":" else "\": ");
            const coordinate = @field(box.*, name);
            if (std.math.isFinite(coordinate)) {
                try pyjson.writeFloat(writer, coordinate, if (style == .orjson_indent2) .orjson else .json);
            } else {
                try writer.writeAll("null");
            }
        }
        if (style == .indent2 or style == .orjson_indent2) {
            try writer.writeByte('\n');
            try writer.splatByteAll(' ', depth * 2);
        }
        try writer.writeByte('}');
    }

    fn cloneRuntime(allocator: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const copy = try allocator.create(BoundingBox);
        copy.* = runtimeBox(ptr).*;
        return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
    }

    fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, _: bool) void {
        allocator.destroy(runtimeBox(ptr));
    }

    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
        // Python model equality compares field dictionaries, whose same-object
        // fast path remains reflexive even for a forged NaN field.
        if (left == right) return true;
        const a = runtimeBox(left);
        const b = runtimeBox(right);
        inline for (.{ "top", "bottom", "left", "right" }) |name| {
            if (@field(a.*, name) != @field(b.*, name)) return false;
        }
        return true;
    }

    /// Canonical structured model dump, not a media marker or encoded string.
    pub fn toValue(self: BoundingBox, arena: Allocator) Allocator.Error!Value {
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.ensureTotalCapacity(arena, 4);
        object.putAssumeCapacity("top", .{ .float = self.top });
        object.putAssumeCapacity("bottom", .{ .float = self.bottom });
        object.putAssumeCapacity("left", .{ .float = self.left });
        object.putAssumeCapacity("right", .{ .float = self.right });
        return .{ .object = object };
    }
};

test "unvalidated box JSON mode nulls nonfinite fields without changing raw formatting" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const box: BoundingBox = .{ .top = std.math.nan(f64), .bottom = 0.5, .left = 0, .right = 1 };
    const actual = try box.toRuntime(arena);
    const validated = try @import("types.zig").validate(BoundingBox, arena, actual);
    try std.testing.expectEqual(actual.custom.ptr, validated.custom.ptr);
    try std.testing.expect(try @import("value.zig").pyEqual(actual, actual));
    const raw = try actual.custom.rawValue(arena);
    try std.testing.expect(std.math.isNan(raw.object.get("top").?.float));
    const serialized = try actual.custom.serialize(arena);
    try std.testing.expect(serialized.object.get("top").? == .null);
    var json: Writer.Allocating = .init(arena);
    try pyjson.writeJson(&json.writer, actual, .compact);
    try std.testing.expectEqualStrings("{\"top\": null, \"bottom\": 0.5, \"left\": 0.0, \"right\": 1.0}", json.written());
    var formatted: Writer.Allocating = .init(arena);
    try box.format(&formatted.writer);
    try std.testing.expectEqualStrings("{\"top\": NaN, \"bottom\": 0.5, \"left\": 0.0, \"right\": 1.0}", formatted.written());
}

test "bounding boxes reject nonfinite out-of-range and nonpositive geometry" {
    const Box = BoundingBox;
    const valid = try Box.init(0, 1, 0, 1);
    try std.testing.expectEqual(@as(f64, 1), valid.bottom);
    for ([_]f64{ -0.001, 1.001, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }) |edge| {
        try std.testing.expectError(error.InvalidBoundingBox, Box.init(edge, 0.875, 0.25, 0.75));
        try std.testing.expectError(error.InvalidBoundingBox, Box.init(0.125, edge, 0.25, 0.75));
        try std.testing.expectError(error.InvalidBoundingBox, Box.init(0.125, 0.875, edge, 0.75));
        try std.testing.expectError(error.InvalidBoundingBox, Box.init(0.125, 0.875, 0.25, edge));
    }
    try std.testing.expectError(error.InvalidBoundingBox, Box.init(0.5, 0.5, 0, 1));
    try std.testing.expectError(error.InvalidBoundingBox, Box.init(0, 1, 0.5, 0.5));
    try std.testing.expectError(error.InvalidBoundingBox, Box.init(0.9, 0.1, 0, 1));
    try std.testing.expectError(error.InvalidBoundingBox, Box.init(0, 1, 0.9, 0.1));
}

test "strict box fields reject coercion missing and extra fields" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const cases = [_][]const u8{
        "{\"top\":\"0.125\",\"bottom\":0.875,\"left\":0.25,\"right\":0.75}",
        "{\"top\":false,\"bottom\":0.875,\"left\":0.25,\"right\":0.75}",
        "{\"bottom\":0.875,\"left\":0.25,\"right\":0.75}",
        "{\"top\":0.125,\"bottom\":0.875,\"left\":0.25,\"right\":0.75,\"confidence\":1}",
    };
    for (cases) |json| {
        const parsed = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
        try std.testing.expectError(error.InvalidBoundingBox, BoundingBox.fromValue(parsed));
    }
    const parsed = try std.json.parseFromSliceLeaky(Value, arena, "{\"top\":0,\"bottom\":1,\"left\":0,\"right\":1}", .{});
    const box = try BoundingBox.fromValue(parsed);
    try std.testing.expectEqual(@as(f64, 0), box.top);
    try std.testing.expectEqual(@as(f64, 1), box.right);
}

test "XML container validation overrides strict box edges but literal JSON does not" {
    const signature = @import("signature.zig");
    const adapter = @import("adapter.zig");
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { region: BoundingBox };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parsed = try adapter.Adapter.xml.parse(arena, view, "<region><top>0.125</top><bottom>0.875</bottom><left>0.25</left><right>0.75</right></region>");
    const box = try BoundingBox.fromValue(parsed.get("region").?);
    try std.testing.expectEqual(@as(f64, 0.125), box.top);
    try std.testing.expectEqual(@as(f64, 0.75), box.right);
    try std.testing.expectError(error.AdapterParse, adapter.Adapter.json.parse(arena, view, "{\"region\":{\"top\":\"0.125\",\"bottom\":0.875,\"left\":0.25,\"right\":0.75}}"));
    try std.testing.expectError(error.AdapterParse, adapter.Adapter.xml.parse(arena, view, "<region>{\"top\":\"0.125\",\"bottom\":0.875,\"left\":0.25,\"right\":0.75}</region>"));
    try std.testing.expectError(error.AdapterParse, adapter.Adapter.xml.parse(arena, view, "<region><top>false</top><bottom>0.875</bottom><left>0.25</left><right>0.75</right></region>"));
}

test "BoundingBox runtime identity stays structured and distinct from raw dictionaries" {
    const value = @import("value.zig");
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const box = try BoundingBox.init(1.0 / 3.0, 0.5, 1e-5, 0.30000000000000004);
    const actual = try box.toRuntime(arena);
    const dictionary = try box.toValue(arena);
    try std.testing.expect(actual.custom.get(BoundingBox) != null);
    try std.testing.expect(dictionary == .object);
    try std.testing.expect(!try value.pyEqual(actual, dictionary));
    try std.testing.expect(!try value.pyEqual(dictionary, actual));
    try std.testing.expect(try value.pyEqual(actual, try box.toRuntime(arena)));
    // Pydantic compares float fields with Python ==, so signed zero is equal.
    try std.testing.expect(try value.pyEqual(
        try (try BoundingBox.init(-0.0, 1, 0, 1)).toRuntime(arena),
        try (try BoundingBox.init(0, 1, 0, 1)).toRuntime(arena),
    ));
    try std.testing.expect(!try value.pyEqual(actual, try (try BoundingBox.init(0, 0.5, 1e-5, 0.1 + 0.2)).toRuntime(arena)));
    try std.testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, dictionary, .{}),
        try std.json.Stringify.valueAlloc(arena, actual, .{}),
    );

    const fields = "top=0.3333333333333333 bottom=0.5 left=1e-05 right=0.30000000000000004";
    const repr = "BoundingBox(top=0.3333333333333333, bottom=0.5, left=1e-05, right=0.30000000000000004)";
    const json = "{\"top\": 0.3333333333333333, \"bottom\": 0.5, \"left\": 1e-05, \"right\": 0.30000000000000004}";
    var text: Writer.Allocating = .init(arena);
    try pyjson.writeStr(&text.writer, actual);
    try std.testing.expectEqualStrings(fields, text.written());
    text.clearRetainingCapacity();
    try pyjson.writeRepr(&text.writer, actual, 0);
    try std.testing.expectEqualStrings(repr, text.written());
    text.clearRetainingCapacity();
    try box.format(&text.writer);
    try std.testing.expectEqualStrings(json, text.written());

    var items: runtime.Array = .init(arena);
    try items.append(actual);
    try items.append(dictionary);
    var nested: runtime.ObjectMap = .empty;
    try nested.put(arena, "boxes", .{ .array = items });
    const input: Value = .{ .object = nested };
    text.clearRetainingCapacity();
    try pyjson.writeRepr(&text.writer, input, 0);
    try std.testing.expectEqualStrings(
        "{'boxes': [" ++ repr ++ ", {'top': 0.3333333333333333, 'bottom': 0.5, 'left': 1e-05, 'right': 0.30000000000000004}]}",
        text.written(),
    );
    // Type inheritance does not turn the wrapped geometry dump into text:
    // ordinary serialization and tool results JSON-encode the structure.
    const serialized = try value.serializedValue(arena, input);
    const serialized_items = serialized.object.get("boxes").?.array.items;
    try std.testing.expect(serialized_items[0] == .object);
    try std.testing.expect(try value.pyEqual(dictionary, serialized_items[0]));
    try std.testing.expect(serialized_items[1] == .object);
    try std.testing.expectEqual(actual.custom.ptr, items.items[0].custom.ptr);
    text.clearRetainingCapacity();
    try pyjson.writeJson(&text.writer, actual, .compact);
    try std.testing.expectEqualStrings(json, text.written());
    try std.testing.expect(actual.custom.vtable.is_type);
}

test "BoundingBox actual instances skip field validation but recheck source ordering" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const valid = try (try BoundingBox.init(0.125, 0.875, 0.25, 0.75)).toRuntime(arena);
    try std.testing.expectEqual(@as(f64, 0.875), (try BoundingBox.fromValue(valid)).bottom);
    try std.testing.expectEqual(@as(f64, 0.875), (try BoundingBox.fromXmlValue(valid)).bottom);
    // Like model_construct(top=2, bottom=3): field validators do not rerun.
    const unchecked_range = try (BoundingBox{ .top = 2, .bottom = 3, .left = 0, .right = 1 }).toRuntime(arena);
    try std.testing.expectEqual(@as(f64, 3), (try BoundingBox.fromValue(unchecked_range)).bottom);
    try std.testing.expectEqual(@as(f64, 3), (try BoundingBox.fromXmlValue(unchecked_range)).bottom);
    const reversed = try (BoundingBox{ .top = 0.9, .bottom = 0.1, .left = 0, .right = 1 }).toRuntime(arena);
    try std.testing.expectError(error.InvalidBoundingBox, BoundingBox.fromValue(reversed));
    try std.testing.expectError(error.InvalidBoundingBox, BoundingBox.fromXmlValue(reversed));
    const other = try (@import("file.zig").File{ .file_id = "file-1" }).toRuntime(arena);
    try std.testing.expectError(error.InvalidBoundingBox, BoundingBox.fromValue(other));
    // A lookalike dictionary is still checked strictly field by field.
    try std.testing.expectError(error.InvalidBoundingBox, BoundingBox.fromValue(try (BoundingBox{ .top = 2, .bottom = 3, .left = 0, .right = 1 }).toValue(arena)));
}

test "BoundingBox typed validation retains instances and boxes dictionaries" {
    const value = @import("value.zig");
    const types = @import("types.zig");
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const box = try BoundingBox.init(0.125, 0.875, 0.25, 0.75);
    const actual = try box.toRuntime(arena);
    const validated = try types.validate(BoundingBox, arena, actual);
    try std.testing.expectEqual(actual.custom.ptr, validated.custom.ptr);
    const dictionary = try box.toValue(arena);
    const coerced = try types.validate(BoundingBox, arena, dictionary);
    try std.testing.expect(coerced == .custom);
    try std.testing.expect(coerced.custom.ptr != actual.custom.ptr);
    try std.testing.expect(try value.pyEqual(actual, coerced));
    try std.testing.expect(dictionary == .object);
    try std.testing.expect(!try value.pyEqual(dictionary, coerced));
    try std.testing.expectEqual(box, try value.fromValue(arena, BoundingBox, actual));
}

test "BoundingBox runtime clone owns its box after source arena release" {
    const value = @import("value.zig");
    const inference = @import("inference.zig");
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);
    const allocator = debug_allocator.allocator();
    var source: std.heap.ArenaAllocator = .init(std.testing.allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const box = try BoundingBox.init(0.125, 0.875, 0.25, 0.75);
    var boxes: runtime.Array = .init(source.allocator());
    try boxes.append(try box.toRuntime(source.allocator()));
    try boxes.append(try box.toValue(source.allocator()));
    var nested: runtime.ObjectMap = .empty;
    try nested.put(source.allocator(), "boxes", .{ .array = boxes });
    const owned = try inference.cloneJsonValue(allocator, .{ .object = nested });
    defer inference.freeJsonValue(allocator, owned);
    const owned_boxes = owned.object.get("boxes").?.array.items;
    try std.testing.expect(owned_boxes[0].custom.ptr != boxes.items[0].custom.ptr);
    try std.testing.expect(owned_boxes[0].custom.owns_data);
    try std.testing.expect(!boxes.items[0].custom.owns_data);
    source.deinit();
    source_live = false;

    var check: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer check.deinit();
    const arena = check.allocator();
    try std.testing.expectEqual(box, owned_boxes[0].custom.get(BoundingBox).?.*);
    try std.testing.expect(try value.pyEqual(try box.toRuntime(arena), owned_boxes[0]));
    try std.testing.expect(owned_boxes[1] == .object);
    try std.testing.expect(!try value.pyEqual(owned_boxes[0], owned_boxes[1]));
}
