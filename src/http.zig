//! Shared standard-library HTTP transport for generation and retrieval.
const std = @import("std");
const Allocator = std.mem.Allocator;
const TimedIo = @import("timed_io.zig");

/// Request-arena-owned headers are refreshed before every network attempt.
pub const HeaderProvider = struct {
    context: *anyopaque,
    prepare: *const fn (*anyopaque, Allocator, []const std.http.Header) anyerror![]const std.http.Header,
};

pub const Response = struct {
    status: std.http.Status,
    /// Owned by the underlying HTTP client's allocator.
    body: []u8,
};

pub const Request = struct {
    url: []const u8,
    method: std.http.Method = .GET,
    body: ?[]const u8 = null,
    content_type: ?[]const u8 = null,
    authorization: ?[]const u8 = null,
    headers: []const std.http.Header = &.{},
};

pub const Transport = struct {
    client: *std.http.Client,
    max_response_bytes: usize = 4 * 1024 * 1024,
    max_retries: u8 = 0,
    retry_base_ms: u32 = 500,
    timeout_ns: ?u64 = null,
    connect_timeout_ns: ?u64 = null,
    header_provider: ?HeaderProvider = null,
    /// Generation accepts only HTTP 200; callers inspecting status use .any.
    success: enum { ok, any } = .any,
    failure_error: anyerror = error.HttpFailure,
    response_limit_error: anyerror = error.ResponseTooLarge,

    pub fn send(self: *const Transport, arena: Allocator, input: Request) anyerror!Response {
        if (self.timeout_ns == null and self.connect_timeout_ns == null)
            return self.sendRetried(arena, input, null);
        var timed: TimedIo = .{
            .inner = self.client.io,
            .timeout_ns = self.timeout_ns,
            .connect_timeout_ns = self.connect_timeout_ns orelse self.timeout_ns,
        };
        // Connections retain their Io. The isolated client closes before this
        // stack adapter expires, without changing the caller's connection pool.
        var isolated: std.http.Client = .{
            .allocator = self.client.allocator,
            .io = timed.io(),
            .tls_buffer_size = self.client.tls_buffer_size,
            .ssl_key_log = self.client.ssl_key_log,
            .read_buffer_size = self.client.read_buffer_size,
            .write_buffer_size = self.client.write_buffer_size,
            .http_proxy = self.client.http_proxy,
            .https_proxy = self.client.https_proxy,
        };
        defer isolated.deinit();
        if (!std.http.Client.disable_tls) {
            try self.client.ca_bundle_lock.lockShared(self.client.io);
            defer self.client.ca_bundle_lock.unlockShared(self.client.io);
            isolated.now = self.client.now;
            if (self.client.now != null) {
                try isolated.ca_bundle.bytes.appendSlice(isolated.allocator, self.client.ca_bundle.bytes.items);
                try isolated.ca_bundle.map.ensureTotalCapacityContext(isolated.allocator, self.client.ca_bundle.map.count(), .{ .cb = &isolated.ca_bundle });
                var entries = self.client.ca_bundle.map.iterator();
                while (entries.next()) |entry| {
                    isolated.ca_bundle.map.putAssumeCapacityContext(entry.key_ptr.*, entry.value_ptr.*, .{ .cb = &isolated.ca_bundle });
                }
            }
        }
        var scoped = self.*;
        scoped.client = &isolated;
        return scoped.sendRetried(arena, input, &timed);
    }

    fn sendRetried(self: *const Transport, arena: Allocator, input: Request, timed: ?*TimedIo) anyerror!Response {
        var attempt: u8 = 0;
        while (true) : (attempt += 1) {
            var prepared = input;
            if (self.header_provider) |provider|
                prepared.headers = try provider.prepare(provider.context, arena, input.headers);
            if (timed) |timer| timer.resetFailure();
            var last_error = self.failure_error;
            const response: ?Response = self.fetch(prepared) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.StreamTooLong => return self.response_limit_error,
                else => failed: {
                    const cause = if (timed) |timer| timer.failure() else null;
                    if (cause) |failure| {
                        if (failure == error.Canceled) return error.Canceled;
                    } else if (err == error.Canceled) return error.Canceled;
                    last_error = cause orelse self.failure_error;
                    break :failed null;
                },
            };
            const status: ?std.http.Status = if (response) |r| r.status else null;
            if (response) |r| {
                if (self.success == .any or r.status == .ok) return r;
                self.client.allocator.free(r.body);
            }
            if (!retryable(status) or attempt >= self.max_retries) return last_error;
            const delay_ms = @as(i64, self.retry_base_ms) << @as(u6, @intCast(@min(attempt, 31)));
            try std.Io.sleep(self.client.io, .fromMilliseconds(delay_ms), .awake);
        }
    }

    fn fetch(self: *const Transport, input: Request) !Response {
        var selected_headers: std.http.Client.Request.Headers = .{
            .content_type = if (input.content_type) |kind| .{ .override = kind } else .default,
            .authorization = if (input.authorization) |token| .{ .override = token } else .default,
        };
        for (input.headers) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, "content-type")) {
                selected_headers.content_type = .omit;
            } else if (std.ascii.eqlIgnoreCase(header.name, "authorization")) {
                selected_headers.authorization = .omit;
            } else if (std.ascii.eqlIgnoreCase(header.name, "host")) {
                selected_headers.host = .omit;
            } else if (std.ascii.eqlIgnoreCase(header.name, "user-agent")) {
                selected_headers.user_agent = .omit;
            } else if (std.ascii.eqlIgnoreCase(header.name, "accept-encoding")) {
                selected_headers.accept_encoding = .omit;
            } else if (std.ascii.eqlIgnoreCase(header.name, "connection")) {
                selected_headers.connection = .omit;
            }
        }
        var request = try self.client.request(input.method, try std.Uri.parse(input.url), .{
            .redirect_behavior = .unhandled,
            .extra_headers = input.headers,
            .headers = selected_headers,
        });
        defer request.deinit();
        return self.transferRequest(&request, input.body) catch |err| {
            if (request.connection) |connection| {
                if (connection.stream_reader.err) |read_error| {
                    if (read_error == error.Canceled) return error.Canceled;
                }
                if (connection.stream_writer.err) |write_error| {
                    if (write_error == error.Canceled) return error.Canceled;
                }
            }
            return err;
        };
    }

    fn transferRequest(self: *const Transport, request: *std.http.Client.Request, body: ?[]const u8) !Response {
        if (body) |bytes| {
            request.transfer_encoding = .{ .content_length = bytes.len };
            var payload = try request.sendBodyUnflushed(&.{});
            try payload.writer.writeAll(bytes);
            try payload.end();
            try request.connection.?.flush();
        } else {
            try request.sendBodiless();
        }
        var response = try request.receiveHead(&.{});
        const status = response.head.status;
        const decompress_buffer: []u8 = switch (response.head.content_encoding) {
            .identity => &.{},
            .zstd => try self.client.allocator.alloc(u8, std.compress.zstd.default_window_len),
            .deflate, .gzip => try self.client.allocator.alloc(u8, std.compress.flate.max_window_len),
            .compress => return error.UnsupportedCompressionMethod,
        };
        defer self.client.allocator.free(decompress_buffer);
        var transfer_buffer: [64]u8 = undefined;
        var decompress: std.http.Decompress = undefined;
        const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);
        return .{
            .status = status,
            .body = try reader.allocRemaining(self.client.allocator, .limited(self.max_response_bytes)),
        };
    }
};

/// Transport failures, HTTP 429 and server errors are retryable.
pub fn retryable(status: ?std.http.Status) bool {
    const s = status orelse return true;
    return s == .too_many_requests or @backingInt(s) >= 500;
}

test "HTTP retry classification preserves terminal client errors" {
    try std.testing.expect(retryable(null));
    try std.testing.expect(retryable(.too_many_requests));
    try std.testing.expect(retryable(.service_unavailable));
    try std.testing.expect(!retryable(.unauthorized));
    try std.testing.expect(!retryable(.bad_request));
}
