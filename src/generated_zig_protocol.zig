//! Bounded binary framing shared by the generated-Zig guest and its host.
//! Payloads are opaque bytes here; the host independently validates capability
//! names and JSON values before calling any registered tool or publishing output.
const std = @import("std");
const assert = std.debug.assert;

pub const version: u8 = 1;
pub const header_bytes: u32 = 8;
pub const payload_bytes_max: u32 = 64 * 1024;
pub const tool_name_bytes_max: u16 = 128;

pub const Kind = enum(u8) {
    input = 1,
    tool_call = 2,
    tool_reply = 3,
    finish = 4,
    failure = 5,
};

pub const Error = error{
    InvalidFrameVersion,
    InvalidFrameKind,
    InvalidFrameReserved,
    FrameCapacityExceeded,
    InvalidToolName,
    InvalidToolCall,
};

pub const Header = struct {
    kind: Kind,
    payload_bytes: u32,

    pub fn encode(header: Header) Error![header_bytes]u8 {
        if (header.payload_bytes > payload_bytes_max) return error.FrameCapacityExceeded;
        var bytes: [header_bytes]u8 = @splat(0);
        bytes[0] = version;
        bytes[1] = @backingInt(header.kind);
        std.mem.writeInt(u32, bytes[4..8], header.payload_bytes, .big);
        assert(bytes[2] == 0);
        assert(bytes[3] == 0);
        return bytes;
    }

    pub fn decode(bytes: *const [header_bytes]u8) Error!Header {
        if (bytes[0] != version) return error.InvalidFrameVersion;
        if (bytes[2] != 0) return error.InvalidFrameReserved;
        if (bytes[3] != 0) return error.InvalidFrameReserved;
        const kind = std.enums.fromInt(Kind, bytes[1]) orelse return error.InvalidFrameKind;
        const payload_bytes = std.mem.readInt(u32, bytes[4..8], .big);
        if (payload_bytes > payload_bytes_max) return error.FrameCapacityExceeded;
        return .{ .kind = kind, .payload_bytes = payload_bytes };
    }
};

pub const ToolCall = struct {
    name: []const u8,
    argument: []const u8,

    /// Both slices borrow the received frame until its backing is reused.
    pub fn decode(payload: []const u8) Error!ToolCall {
        if (payload.len < 2) return error.InvalidToolCall;
        if (payload.len > payload_bytes_max) return error.FrameCapacityExceeded;
        const name_bytes = std.mem.readInt(u16, payload[0..2], .big);
        if (name_bytes == 0) return error.InvalidToolName;
        if (name_bytes > tool_name_bytes_max) return error.InvalidToolName;
        const argument_start = 2 + @as(u32, name_bytes);
        if (argument_start >= payload.len) return error.InvalidToolCall;
        const name = payload[2..argument_start];
        try validateName(name);
        return .{ .name = name, .argument = payload[argument_start..] };
    }

    pub fn encode(call: ToolCall, destination: []u8) Error![]const u8 {
        try validateName(call.name);
        if (call.argument.len == 0) return error.InvalidToolCall;
        const needed = std.math.add(usize, 2 + call.name.len, call.argument.len) catch
            return error.FrameCapacityExceeded;
        if (needed > payload_bytes_max) return error.FrameCapacityExceeded;
        if (needed > destination.len) return error.FrameCapacityExceeded;
        std.mem.writeInt(u16, destination[0..2], @intCast(call.name.len), .big);
        @memcpy(destination[2..][0..call.name.len], call.name);
        @memcpy(destination[2 + call.name.len ..][0..call.argument.len], call.argument);
        assert(needed <= destination.len);
        return destination[0..needed];
    }
};

fn validateName(name: []const u8) Error!void {
    if (name.len == 0) return error.InvalidToolName;
    if (name.len > tool_name_bytes_max) return error.InvalidToolName;
}
