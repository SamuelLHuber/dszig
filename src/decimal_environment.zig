//! Thread-local current Decimal context, separate from the pure math engine.
//! Context pointers are borrowed: setContext's argument must remain alive until
//! replaced. LocalContext must have a stable address between enter and deinit.
const std = @import("std");
const engine = @import("decimal_context.zig");
pub const Context = engine.Context;
pub const Rounding = engine.Rounding;
pub const Signal = engine.Signal;
pub const Signals = engine.Signals;

threadlocal var default_context: Context = .{};
threadlocal var current_context: ?*Context = null;

pub fn getContext() *Context {
    return current_context orelse &default_context;
}

pub fn setContext(context: *Context) void {
    current_context = context;
}

/// A localcontext() copies the selected context, including its sticky flags,
/// then restores the preceding context object even when the body fails. Held
/// pointers to the outer context continue to identify that unchanged object.
pub const LocalContext = struct {
    context: Context = .{},
    previous: ?*Context = null,

    pub fn enter(self: *LocalContext, source: ?*const Context) void {
        std.debug.assert(self.previous == null);
        const previous = getContext();
        self.context = (source orelse previous).*;
        self.previous = previous;
        current_context = &self.context;
    }

    pub fn deinit(self: *LocalContext) void {
        current_context = self.previous orelse unreachable;
        self.previous = null;
    }
};
