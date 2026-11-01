const dszig = @import("dszig");
const Sig = dszig.Signature(struct {
    pub const Input = struct { q: []const u8 };
    pub const Output = struct { answer: []const u8 };
    pub const descriptions = .{ .answr = "typo" };
});
test {
    _ = Sig;
}
