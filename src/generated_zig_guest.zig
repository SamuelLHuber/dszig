//! The generated program receives only bounded input and capability RPC.
//! This module is not a sandbox: the worker MUST install the kernel policy
//! before calling generated code. Even forged frames are checked by the host.
const standard = @import("std");
const linux = standard.os.linux;
const protocol = @import("generated_zig_protocol.zig");

/// Pure algorithms, formatting and JSON may use caller-provisioned fixed storage.
/// OS calls reachable through std remain subject to the worker's syscall policy.
pub const std = standard;
pub const bytes_max = protocol.payload_bytes_max;
pub const Error = protocol.Error || error{
    GuestReadFailed,
    GuestWriteFailed,
    GuestEndOfStream,
    GuestInterrupted,
    UnexpectedFrame,
    HostRejected,
    AlreadyStarted,
    NotStarted,
    AlreadyFinished,
};

/// Construct in place before applying the worker's execution policy. No guest
/// method allocates, maps memory, starts threads or opens a file or socket.
pub const Context = struct {
    input_buffer: [bytes_max]u8 = undefined,
    reply_buffer: [bytes_max]u8 = undefined,
    input_bytes: u32 = 0,
    reply_bytes: u32 = 0,
    started: bool = false,
    finished: bool = false,

    pub fn start(context: *Context) Error!void {
        if (context.started) return error.AlreadyStarted;
        const header = try receiveHeader();
        if (header.kind != .input) return error.UnexpectedFrame;
        try readAll(context.input_buffer[0..header.payload_bytes]);
        context.input_bytes = header.payload_bytes;
        context.started = true;
    }

    pub fn input(context: *const Context) Error![]const u8 {
        if (!context.started) return error.NotStarted;
        return context.input_buffer[0..context.input_bytes];
    }

    /// Reply bytes borrow fixed Context storage until the next tool call.
    /// The argument is sent completely before that storage is overwritten.
    pub fn tool(context: *Context, name: []const u8, argument: []const u8) Error![]const u8 {
        if (!context.started) return error.NotStarted;
        if (context.finished) return error.AlreadyFinished;
        if (name.len == 0) return error.InvalidToolName;
        if (name.len > protocol.tool_name_bytes_max) return error.InvalidToolName;
        if (argument.len == 0) return error.InvalidToolCall;
        const payload_bytes = standard.math.add(usize, 2 + name.len, argument.len) catch
            return error.FrameCapacityExceeded;
        if (payload_bytes > bytes_max) return error.FrameCapacityExceeded;
        try sendHeader(.tool_call, @intCast(payload_bytes));
        var name_length: [2]u8 = undefined;
        standard.mem.writeInt(u16, &name_length, @intCast(name.len), .big);
        try writeAll(&name_length);
        try writeAll(name);
        try writeAll(argument);
        const response = try receiveHeader();
        context.reply_bytes = 0;
        if (response.kind != .tool_reply and response.kind != .failure) return error.UnexpectedFrame;
        try readAll(context.reply_buffer[0..response.payload_bytes]);
        context.reply_bytes = response.payload_bytes;
        if (response.kind == .failure) return error.HostRejected;
        return context.reply_buffer[0..context.reply_bytes];
    }

    /// Sending Finish does not publish a result: the host waits for successful
    /// process exit and complete process-tree cleanup before retaining output.
    pub fn finish(context: *Context, result: []const u8) Error!void {
        if (!context.started) return error.NotStarted;
        if (context.finished) return error.AlreadyFinished;
        if (result.len > bytes_max) return error.FrameCapacityExceeded;
        try sendHeader(.finish, @intCast(result.len));
        try writeAll(result);
        context.finished = true;
    }

    pub fn lastReply(context: *const Context) []const u8 {
        return context.reply_buffer[0..context.reply_bytes];
    }
};

fn receiveHeader() Error!protocol.Header {
    var bytes: [protocol.header_bytes]u8 = undefined;
    try readAll(&bytes);
    return protocol.Header.decode(&bytes);
}

fn sendHeader(kind: protocol.Kind, payload_bytes: u32) Error!void {
    const header = try (protocol.Header{ .kind = kind, .payload_bytes = payload_bytes }).encode();
    try writeAll(&header);
}

fn readAll(destination: []u8) Error!void {
    var offset: usize = 0;
    var interruptions: u8 = 0;
    while (offset < destination.len) {
        const result = linux.syscall3(.read, 0, @intFromPtr(destination[offset..].ptr), destination.len - offset);
        if (interrupted(result)) {
            if (interruptions == 64) return error.GuestInterrupted;
            interruptions += 1;
            continue;
        }
        if (result > destination.len - offset) return error.GuestReadFailed;
        if (result == 0) return error.GuestEndOfStream;
        offset += result;
    }
    standard.debug.assert(offset == destination.len);
}

fn writeAll(source: []const u8) Error!void {
    var offset: usize = 0;
    var interruptions: u8 = 0;
    while (offset < source.len) {
        const result = linux.syscall3(.write, 1, @intFromPtr(source[offset..].ptr), source.len - offset);
        if (interrupted(result)) {
            if (interruptions == 64) return error.GuestInterrupted;
            interruptions += 1;
            continue;
        }
        if (result > source.len - offset) return error.GuestWriteFailed;
        if (result == 0) return error.GuestWriteFailed;
        offset += result;
    }
    standard.debug.assert(offset == source.len);
}

fn interrupted(result: usize) bool {
    return result == 0 -% @as(usize, @backingInt(linux.E.INTR));
}
