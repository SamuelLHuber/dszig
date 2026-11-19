//! Uncompressed RFC 6455 client over the standard HTTP client's connection.
const std = @import("std");
const Allocator = std.mem.Allocator;
const base64 = std.base64.standard.Encoder;

request: std.http.Client.Request,

const Socket = @This();
const Opcode = enum(u4) { continuation = 0, text = 1, binary = 2, close = 8, ping = 9, pong = 10, _ };

pub fn open(client: *std.http.Client, uri: std.Uri) !Socket {
    var nonce: [16]u8 = undefined;
    try client.io.randomSecure(&nonce);
    var key_buffer: [24]u8 = undefined;
    const key = base64.encode(&key_buffer, &nonce);
    var request = try client.request(.GET, uri, .{
        .keep_alive = false,
        .redirect_behavior = .unhandled,
        .headers = .{ .connection = .{ .override = "Upgrade" }, .accept_encoding = .omit },
        .extra_headers = &.{
            .{ .name = "Upgrade", .value = "websocket" },
            .{ .name = "Sec-WebSocket-Version", .value = "13" },
            .{ .name = "Sec-WebSocket-Key", .value = key },
        },
    });
    errdefer request.deinit();
    request.connection.?.closing = true;
    try request.sendBodiless();
    const response = try request.receiveHead(&.{});
    try checkUpgrade(response.head, key);
    // Request options above borrow handshake-local data. Nothing uses them again.
    request.extra_headers = &.{};
    return .{ .request = request };
}

fn checkUpgrade(head: std.http.Client.Response.Head, key: []const u8) !void {
    if (head.status != .switching_protocols) return error.InvalidUpgrade;
    var sha = std.crypto.hash.Sha1.init(.{});
    sha.update(key);
    sha.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    const hash = sha.finalResult();
    var accept_buffer: [28]u8 = undefined;
    const accept = base64.encode(&accept_buffer, &hash);
    var upgrade = false;
    var connection = false;
    var accepted = false;
    var headers = head.iterateHeaders();
    while (headers.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "upgrade")) {
            upgrade = upgrade or headerToken(header.value, "websocket");
        } else if (std.ascii.eqlIgnoreCase(header.name, "connection")) {
            connection = connection or headerToken(header.value, "upgrade");
        } else if (std.ascii.eqlIgnoreCase(header.name, "sec-websocket-accept")) {
            if (accepted or !std.mem.eql(u8, header.value, accept)) return error.InvalidUpgrade;
            accepted = true;
        } else if (std.ascii.eqlIgnoreCase(header.name, "sec-websocket-extensions") or
            std.ascii.eqlIgnoreCase(header.name, "sec-websocket-protocol"))
        {
            return error.InvalidUpgrade;
        }
    }
    if (!upgrade or !connection or !accepted) return error.InvalidUpgrade;
}

fn headerToken(value: []const u8, token: []const u8) bool {
    var items = std.mem.splitScalar(u8, value, ',');
    while (items.next()) |item| {
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, item, " \t"), token)) return true;
    }
    return false;
}

pub fn deinit(self: *Socket) void {
    self.request.connection.?.closing = true;
    self.request.deinit();
    self.* = undefined;
}

pub fn sendText(self: *Socket, bytes: []const u8) !void {
    std.debug.assert(std.unicode.utf8ValidateSlice(bytes));
    try self.send(.text, bytes);
}

pub fn finish(self: *Socket) !void {
    try self.send(.close, &.{ 3, 232 }); // Normal closure, without waiting for the peer.
}

fn send(self: *Socket, opcode: Opcode, bytes: []const u8) !void {
    var mask: [4]u8 = undefined;
    try self.request.client.io.randomSecure(&mask);
    const connection = self.request.connection.?;
    const writer = connection.writer();
    try writer.writeByte(0x80 | @as(u8, @backingInt(opcode)));
    if (bytes.len < 126) {
        try writer.writeByte(0x80 | @as(u8, @intCast(bytes.len)));
    } else if (bytes.len <= std.math.maxInt(u16)) {
        try writer.writeByte(0x80 | 126);
        try writer.writeInt(u16, @intCast(bytes.len), .big);
    } else {
        try writer.writeByte(0x80 | 127);
        try writer.writeInt(u64, bytes.len, .big);
    }
    try writer.writeAll(&mask);
    var buffer: [1024]u8 = undefined;
    var offset: usize = 0;
    while (offset < bytes.len) {
        const count = @min(buffer.len, bytes.len - offset);
        for (buffer[0..count], bytes[offset..][0..count], 0..) |*target, byte, i| {
            target.* = byte ^ mask[(offset + i) % mask.len];
        }
        try writer.writeAll(buffer[0..count]);
        offset += count;
    }
    try connection.flush();
}

const Frame = struct {
    final: bool,
    opcode: Opcode,
    size: u64,

    fn read(reader: *std.Io.Reader) !Frame {
        var header: [2]u8 = undefined;
        try reader.readSliceAll(&header);
        if (header[0] & 0x70 != 0 or header[1] & 0x80 != 0) return error.InvalidFrame;
        const opcode: Opcode = @fromBackingInt(@intCast(header[0] & 0x0f));
        const final = header[0] & 0x80 != 0;
        const short = header[1] & 0x7f;
        if (@backingInt(opcode) >= 8 and (!final or short > 125)) return error.InvalidFrame;
        const size: u64 = switch (short) {
            126 => try reader.takeInt(u16, .big),
            127 => try reader.takeInt(u64, .big),
            else => short,
        };
        if (short == 126 and size < 126) return error.InvalidFrame;
        if (short == 127 and (size <= std.math.maxInt(u16) or size >> 63 != 0)) {
            return error.InvalidFrame;
        }
        return .{ .final = final, .opcode = opcode, .size = size };
    }
};

/// Reuses message storage. The result is valid until the next receive.
pub fn receive(self: *Socket, allocator: Allocator, buffer: *std.ArrayList(u8), limit: usize) ![]const u8 {
    buffer.clearRetainingCapacity();
    const reader = self.request.connection.?.reader();
    var message_opcode: ?Opcode = null;
    for (0..8192) |_| {
        const frame = try Frame.read(reader);
        switch (frame.opcode) {
            .close, .ping, .pong => {
                var control: [125]u8 = undefined;
                const bytes = control[0..@intCast(frame.size)];
                try reader.readSliceAll(bytes);
                switch (frame.opcode) {
                    .ping => try self.send(.pong, bytes),
                    .close => {
                        if (bytes.len == 1) return error.InvalidFrame;
                        try self.send(.close, bytes);
                        return error.PeerClosed;
                    },
                    else => {},
                }
                continue;
            },
            .text, .binary => {
                if (message_opcode != null) return error.InvalidFrame;
                message_opcode = frame.opcode;
            },
            .continuation => if (message_opcode == null) return error.InvalidFrame,
            else => return error.InvalidFrame,
        }
        if (frame.size > limit - buffer.items.len) return error.MessageTooLarge;
        const tail = try buffer.addManyAsSlice(allocator, @intCast(frame.size));
        try reader.readSliceAll(tail);
        if (frame.final) {
            if (message_opcode == .text and !std.unicode.utf8ValidateSlice(buffer.items)) {
                return error.InvalidFrame;
            }
            return buffer.items;
        }
    }
    return error.FrameLimit;
}

test "WebSocket rejects noncanonical lengths and fragmented control frames" {
    const cases = [_][]const u8{
        &.{ 0x81, 126, 0, 125 },
        &.{ 0x81, 127, 0x80, 0, 0, 0, 0, 0, 0, 0 },
        &.{ 0x09, 0 },
        &.{ 0x81, 0x80 },
        &.{ 0xc1, 0 },
    };
    for (cases) |bytes| {
        var reader = std.Io.Reader.fixed(bytes);
        try std.testing.expectError(error.InvalidFrame, Frame.read(&reader));
    }
    var reader = std.Io.Reader.fixed(&.{ 0x02, 126, 0, 126 });
    const frame = try Frame.read(&reader);
    try std.testing.expect(!frame.final);
    try std.testing.expectEqual(Opcode.binary, frame.opcode);
    try std.testing.expectEqual(126, frame.size);
}
