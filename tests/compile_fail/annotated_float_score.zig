const dszig = @import("dszig");
const Sig = dszig.Signature(struct {
    pub const Input = struct { q: []const u8 };
    pub const Output = struct { answer: f64 };
    pub const descriptions = .{ .answer = .{ .text = "Answer.", .metadata = &.{dszig.Score} } };
});
test {
    _ = Sig;
}
