const dszig = @import("dszig");
const Number = dszig.IntEnum("InvalidBytes", &.{.{
    .name = "value",
    .value = dszig.python_value.staticBytes("\xd9\xa1"),
}});
test {
    _ = Number;
}
