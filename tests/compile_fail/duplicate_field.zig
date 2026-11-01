const dszig = @import("dszig");
const Sig = dszig.Signature(struct {
    pub const Input = struct { q: []const u8 };
    pub const Output = struct { q: []const u8 };
});
test {
    _ = Sig;
}
