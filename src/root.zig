//! DSZig: DSPy in Zig.

pub const pyjson = @import("pyjson.zig");
pub const pyrandom = @import("pyrandom.zig");
pub const signature = @import("signature.zig");
pub const Signature = signature.Signature;
pub const value = @import("value.zig");
pub const adapter = @import("adapter.zig");
pub const Adapter = adapter.Adapter;
pub const lm = @import("lm.zig");
pub const LM = lm.LM;
pub const predict = @import("predict.zig");
pub const Context = predict.Context;
pub const Trace = predict.Trace;
pub const Predict = predict.Predict;
pub const ChainOfThought = predict.ChainOfThought;
pub const Retrieve = predict.Retrieve;
pub const optimize = @import("optimize.zig");
pub const Example = optimize.Example;
pub const BootstrapFewShot = optimize.BootstrapFewShot;
pub const LabeledFewShot = optimize.LabeledFewShot;
pub const state = @import("state.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
