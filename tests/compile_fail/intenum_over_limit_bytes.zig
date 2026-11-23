const dszig = @import("dszig");
const digits: [4301]u8 = @splat('1');
const Number = dszig.IntEnum("InvalidBytes", &.{.{
    .name = "value",
    .value = dszig.python_value.staticBytes(&digits),
}});
test {
    _ = Number;
}
