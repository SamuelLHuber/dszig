//! DSPy 3.5's axis-aligned normalized image rectangle.
//! Construct with init or fromValue to enforce finite edges and positive area.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

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
        const pyjson = @import("pyjson.zig");
        inline for (.{ "top", "bottom", "left", "right" }, 0..) |name, index| {
            try writer.writeAll(if (index == 0) "{\"" ++ name ++ "\": " else ", \"" ++ name ++ "\": ");
            try pyjson.writeFloat(writer, @field(self, name), .json);
        }
        try writer.writeByte('}');
    }

    /// Canonical structured model dump, not a media marker or encoded string.
    pub fn toValue(self: BoundingBox, arena: Allocator) Allocator.Error!Value {
        var object: std.json.ObjectMap = .empty;
        try object.ensureTotalCapacity(arena, 4);
        object.putAssumeCapacity("top", .{ .float = self.top });
        object.putAssumeCapacity("bottom", .{ .float = self.bottom });
        object.putAssumeCapacity("left", .{ .float = self.left });
        object.putAssumeCapacity("right", .{ .float = self.right });
        return .{ .object = object };
    }
};

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
