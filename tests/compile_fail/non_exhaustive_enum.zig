const dszig = @import("dszig");
const Sig = dszig.Signature(struct {
    pub const Input = struct { q: []const u8 };
    pub const Output = struct { label: enum(u8) { a, b, _ } };
});
test {
    _ = Sig;
}
