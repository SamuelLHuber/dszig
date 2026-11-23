const dszig = @import("dszig");
const Choice = dszig.ChoiceWithOptions(&.{.{ .value = .{ .integer = 1 }, .description_json = "\"one\"" }});
const Sig = dszig.Signature(struct {
    pub const Input = struct { q: []const u8 };
    pub const Output = struct { answer: dszig.Literal(&.{.{ .bool = true }}) };
    pub const descriptions = .{ .answer = .{ .text = "Answer.", .metadata = &.{Choice} } };
});
test {
    _ = Sig;
}
