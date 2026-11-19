//! Uncompressed RFC 6455 client over the standard HTTP client's connection.
const std = @import("std");
const Allocator = std.mem.Allocator;
const base64 = std.base64.standard.Encoder;

request: ?std.http.Client.Request,
close_sent: bool = false,

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
        .handle_continue = false, // Count 100 Continue toward the same handshake limit.
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
    try receiveUpgrade(&request, key);
    // Request options above borrow handshake-local data. Nothing uses them again.
    request.extra_headers = &.{};
    return .{ .request = request };
}

fn receiveUpgrade(request: *std.http.Client.Request, key: []const u8) !void {
    const max_interim_responses = 16;
    for (0..max_interim_responses + 1) |_| {
        const response = try request.receiveHead(&.{});
        const status = response.head.status;
        if (status.class() != .informational) return error.InvalidUpgrade;
        // RFC 9110 §8.6 and RFC 9112 §6.1 forbid body framing headers on
        // every 1xx response, including the 101 that ends HTTP framing.
        var headers = response.head.iterateHeaders();
        while (headers.next()) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, "content-length") or
                std.ascii.eqlIgnoreCase(header.name, "transfer-encoding"))
                return error.InvalidUpgrade;
        }
        if (status == .switching_protocols) return checkUpgrade(response.head, key);
    }
    return error.InvalidUpgrade;
}

fn checkUpgrade(head: std.http.Client.Response.Head, key: []const u8) !void {
    // The WebSocket Upgrade handshake runs over HTTP/1.1, not HTTP/1.0.
    if (head.version != .@"HTTP/1.1" or head.status != .switching_protocols) return error.InvalidUpgrade;
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
            // RFC 6455 §4.1 requires the whole Upgrade value to match.
            // Unlike Connection, it is not a search for one token in a list.
            if (upgrade or !std.ascii.eqlIgnoreCase(std.mem.trim(u8, header.value, " \t"), "websocket"))
                return error.InvalidUpgrade;
            upgrade = true;
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

fn disconnect(self: *Socket) void {
    if (self.request) |*request| {
        request.connection.?.closing = true;
        request.deinit();
        self.request = null;
    }
}

pub fn deinit(self: *Socket) void {
    self.disconnect();
    self.* = undefined;
}

pub fn isOpen(self: *const Socket) bool {
    return self.request != null and !self.close_sent;
}

pub fn sendText(self: *Socket, bytes: []const u8) !void {
    if (!self.isOpen()) return error.PeerClosed;
    std.debug.assert(std.unicode.utf8ValidateSlice(bytes));
    self.send(.text, bytes) catch |err| {
        // A failed write can leave a partial frame on the wire. Neither a
        // fresh message nor a Close frame can safely follow it.
        self.disconnect();
        return err;
    };
}

pub fn finish(self: *Socket) !void {
    if (self.request == null) return;
    try self.sendClose(&.{ 3, 232 }); // Normal closure, without waiting for the peer.
}

fn failConnection(self: *Socket, status: u16) void {
    // RFC 6455 §7.1.7: send a best-effort Close and stop reading.
    var payload: [2]u8 = undefined;
    std.mem.writeInt(u16, &payload, status, .big);
    self.sendClose(&payload) catch {};
    self.disconnect();
}

fn failFrame(self: *Socket, status: u16) error{InvalidFrame} {
    self.failConnection(status);
    return error.InvalidFrame;
}

fn sendClose(self: *Socket, payload: []const u8) !void {
    if (self.close_sent) return;
    // A failed write can still have sent part of the frame. Never retry it.
    self.close_sent = true;
    try self.send(.close, payload);
}

fn send(self: *Socket, opcode: Opcode, bytes: []const u8) !void {
    var mask: [4]u8 = undefined;
    const request = &self.request.?;
    try request.client.io.randomSecure(&mask);
    const connection = request.connection.?;
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

pub const Message = struct {
    bytes: []const u8,
    binary: bool,
};

/// Reuses message storage. The result is valid until the next receive.
pub fn receive(self: *Socket, allocator: Allocator, buffer: *std.ArrayList(u8), limit: usize) !Message {
    if (!self.isOpen()) return error.PeerClosed;
    return self.receiveMessage(allocator, buffer, limit) catch |err| {
        // A failed receive can consume part of a frame or fragmented message.
        // Its parser state is gone, so a subsequent receive cannot resume it.
        self.disconnect();
        return err;
    };
}

fn receiveMessage(self: *Socket, allocator: Allocator, buffer: *std.ArrayList(u8), limit: usize) !Message {
    buffer.clearRetainingCapacity();
    const reader = self.request.?.connection.?.reader();
    var message_opcode: ?Opcode = null;
    for (0..8192) |_| {
        const frame = Frame.read(reader) catch |err| switch (err) {
            error.InvalidFrame => return self.failFrame(1002),
            else => return err,
        };
        switch (frame.opcode) {
            .close, .ping, .pong => {
                var control: [125]u8 = undefined;
                const bytes = control[0..@intCast(frame.size)];
                try reader.readSliceAll(bytes);
                switch (frame.opcode) {
                    .ping => try self.send(.pong, bytes),
                    .close => {
                        if (invalidCloseStatus(bytes)) |status| return self.failFrame(status);
                        // RFC 6455 §5.5.1 permits closing TCP after replying.
                        // Do not retain the connection if the reply fails either.
                        defer self.disconnect();
                        try self.sendClose(bytes);
                        return error.PeerClosed;
                    },
                    else => {},
                }
                continue;
            },
            .text, .binary => {
                if (message_opcode != null) return self.failFrame(1002);
                message_opcode = frame.opcode;
            },
            .continuation => if (message_opcode == null) return self.failFrame(1002),
            else => return self.failFrame(1002),
        }
        if (frame.size > limit - buffer.items.len) {
            // The unread payload prevents resuming at a frame boundary.
            self.failConnection(1009);
            return error.MessageTooLarge;
        }
        const tail = try buffer.addManyAsSlice(allocator, @intCast(frame.size));
        try reader.readSliceAll(tail);
        if (frame.final) {
            if (message_opcode == .text and !std.unicode.utf8ValidateSlice(buffer.items)) {
                return self.failFrame(1007);
            }
            return .{ .bytes = buffer.items, .binary = message_opcode == .binary };
        }
    }
    return error.FrameLimit;
}

fn invalidCloseStatus(bytes: []const u8) ?u16 {
    if (bytes.len == 0) return null;
    if (bytes.len == 1) return 1002;
    // RFC 6455 §7.4 and the IANA WebSocket Close Code registry.
    switch (std.mem.readInt(u16, bytes[0..2], .big)) {
        1000...1003, 1007...1014, 3000...4999 => {},
        else => return 1002,
    }
    if (!std.unicode.utf8ValidateSlice(bytes[2..])) return 1007;
    return null;
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
