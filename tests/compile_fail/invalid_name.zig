const dszig = @import("dszig");
const Sig = dszig.Signature(struct {
    pub const Input = struct { @"bad-name": []const u8 };
    pub const Output = struct { a: []const u8 };
});
test {
    _ = Sig;
}
