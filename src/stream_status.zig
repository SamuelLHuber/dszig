//! Synchronous DSPy program status messages. Callback views are borrowed only
//! while the callback runs; no graph conversion or output allocation is needed.
const std = @import("std");
const lm = @import("lm.zig");
const value = @import("value.zig");
const Value = @import("runtime_value.zig").Value;

fn Identity(comptime T: type) type {
    return struct {
        // Its mutable address distinguishes even zero-sized and layout-equal types.
        var token: u8 = @truncate(@typeName(T).len);
    };
}

pub const View = struct {
    pointer: *const anyopaque,
    identity: *const u8,

    pub fn borrow(comptime T: type, pointer: *const T) View {
        return .{ .pointer = @ptrCast(pointer), .identity = &Identity(T).token };
    }

    pub fn as(self: View, comptime T: type) ?*const T {
        if (self.identity != &Identity(T).token) return null;
        return @ptrCast(@alignCast(self.pointer));
    }
};

pub const Event = union(enum) {
    module_start: struct { instance: View, input: View },
    module_end: union(enum) { output: View, failure: anyerror },
    lm_start: struct { instance: lm.LM, request: *const lm.Request },
    lm_end: union(enum) { output: []const lm.Completion, failure: lm.Error },
    tool_start: struct { name: []const u8, input: *const value.Record },
    tool_end: union(enum) { output: Value, failure: anyerror },
};

/// Concatenated segments form one status message. The split allows a tool
/// name of any length without formatting into scratch or allocating.
pub const Message = struct {
    prefix: []const u8 = "",
    text: []const u8,
    suffix: []const u8 = "",

    pub fn write(self: Message, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.writeAll(self.prefix);
        try writer.writeAll(self.text);
        try writer.writeAll(self.suffix);
    }
};

pub const Sink = struct {
    context: *anyopaque,
    on_message: *const fn (*anyopaque, Message) void,
    /// Null selects DSPy's default tool-only provider. A custom provider
    /// returns null or an empty message to suppress this event.
    provide: ?*const fn (*anyopaque, Event) ?Message = null,

    pub fn notify(self: Sink, event: Event) void {
        switch (event) {
            .tool_start => |tool| if (std.mem.eql(u8, tool.name, "finish")) return,
            .tool_end => |end| if (end == .output and end.output == .string and
                std.mem.eql(u8, end.output.string, "Completed.")) return,
            else => {},
        }
        const message = (if (self.provide) |provide| provide(self.context, event) else default(event)) orelse return;
        if (message.prefix.len == 0 and message.text.len == 0 and message.suffix.len == 0) return;
        self.on_message(self.context, message);
    }
};

fn default(event: Event) ?Message {
    return switch (event) {
        .tool_start => |tool| .{ .prefix = "Calling tool ", .text = tool.name, .suffix = "..." },
        .tool_end => .{ .text = "Tool calling finished! Querying the LLM with tool calling results..." },
        else => null,
    };
}
