//! Shared standard-library HTTP transport for generation and retrieval.
const std = @import("std");
const Allocator = std.mem.Allocator;
const TimedIo = @import("timed_io.zig");
const unicode = @import("unicode.zig");

/// Request-arena-owned headers are refreshed before each retry attempt.
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
    max_redirects: u8 = 0,
    normalize_url: bool = false,

    const WireResponse = struct { response: Response, location: ?[]const u8 };

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
            const response: ?Response = self.fetch(arena, prepared) catch |err| switch (err) {
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

    fn fetch(self: *const Transport, arena: Allocator, input: Request) !Response {
        var current = input;
        if (self.normalize_url) current.url = try requoteUri(arena, current.url);
        var redirects: u8 = 0;
        while (true) {
            const wire = try self.fetchOnce(arena, current);
            const location = wire.location orelse return wire.response;
            // The body has been consumed before opening another request.
            defer self.client.allocator.free(wire.response.body);
            if (redirects >= self.max_redirects) return error.TooManyHttpRedirects;
            const base = try std.Uri.parse(current.url);
            const target = if (self.normalize_url) try requoteUri(arena, location) else location;
            const auxiliary = try arena.alloc(u8, current.url.len + 2 * target.len + 1);
            @memcpy(auxiliary[0..target.len], target);
            var remaining = auxiliary;
            var resolved = try base.resolveInPlace(target.len, &remaining);
            if (resolved.fragment == null or componentBytes(resolved.fragment.?).len == 0)
                resolved.fragment = base.fragment;
            const remove_auth = stripAuth(base, resolved);
            const remove_body = wire.response.status != .temporary_redirect and
                wire.response.status != .permanent_redirect;
            current.headers = try redirectedHeaders(arena, current.headers, remove_auth, remove_body);
            if (remove_auth) current.authorization = null;
            if ((wire.response.status == .see_other or wire.response.status == .found) and current.method != .HEAD or
                wire.response.status == .moved_permanently and current.method == .POST)
                current.method = .GET;
            if (remove_body) {
                current.body = null;
                current.content_type = null;
            }
            current.url = try std.fmt.allocPrint(arena, "{f}", .{resolved});
            redirects += 1;
        }
    }

    fn fetchOnce(self: *const Transport, arena: Allocator, input: Request) !WireResponse {
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
        return self.transferRequest(arena, &request, input.body) catch |err| {
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

    fn transferRequest(self: *const Transport, arena: Allocator, request: *std.http.Client.Request, body: ?[]const u8) !WireResponse {
        if (body != null or request.method.requestHasBody()) {
            const bytes = body orelse "";
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
        const location = if (self.max_redirects > 0 and isRedirect(status))
            if (response.head.location) |text| if (text.len > 0) try arena.dupe(u8, text) else null else null
        else
            null;
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
            .response = .{
                .status = status,
                .body = try reader.allocRemaining(self.client.allocator, .limited(self.max_response_bytes)),
            },
            .location = location,
        };
    }
};

/// Requests-compatible URI quoting, including Unicode decimal percent pairs.
/// Unchanged input is borrowed; changed text belongs to `arena`.
pub fn requoteUri(arena: Allocator, input: []const u8) (Allocator.Error || error{InvalidUtf8})![]const u8 {
    if (!std.unicode.utf8ValidateSlice(input)) return error.InvalidUtf8;
    var allow_percent = true;
    for (input, 0..) |byte, i| {
        if (byte != '%') continue;
        const pair = percentPair(input[i + 1 ..]) orelse continue;
        if (alphaNumeric(pair.first) and alphaNumeric(pair.second) and
            (hexValue(pair.first) == null or hexValue(pair.second) == null))
        {
            allow_percent = false;
            break;
        }
    }

    var size: usize = 0;
    var changed = false;
    var i: usize = 0;
    while (i < input.len) {
        if (allow_percent and input[i] == '%') {
            if (unreservedEscape(input[i + 1 ..])) |decoded| {
                size += 1;
                i += 1 + decoded.bytes;
                changed = true;
                continue;
            }
        }
        const safe = safeUriByte(input[i], allow_percent);
        size += if (safe) @as(usize, 1) else 3;
        changed = changed or !safe;
        i += 1;
    }
    if (!changed) return input;
    const out = try arena.alloc(u8, size);
    const hex = "0123456789ABCDEF";
    i = 0;
    var written: usize = 0;
    while (i < input.len) {
        if (allow_percent and input[i] == '%') {
            if (unreservedEscape(input[i + 1 ..])) |decoded| {
                out[written] = decoded.byte;
                written += 1;
                i += 1 + decoded.bytes;
                continue;
            }
        }
        const byte = input[i];
        if (safeUriByte(byte, allow_percent)) {
            out[written] = byte;
            written += 1;
        } else {
            out[written..][0..3].* = .{ '%', hex[byte >> 4], hex[byte & 15] };
            written += 3;
        }
        i += 1;
    }
    return out;
}

const PercentPair = struct { first: u21, second: u21, bytes: usize };
fn percentPair(text: []const u8) ?PercentPair {
    var iterator = std.unicode.Utf8View.initUnchecked(text).iterator();
    const first = iterator.nextCodepoint() orelse return null;
    const second = iterator.nextCodepoint() orelse return null;
    return .{ .first = first, .second = second, .bytes = iterator.i };
}

const DecodedEscape = struct { byte: u8, bytes: usize };
fn unreservedEscape(text: []const u8) ?DecodedEscape {
    const pair = percentPair(text) orelse return null;
    const high = hexValue(pair.first) orelse return null;
    const low = hexValue(pair.second) orelse return null;
    const byte = high * 16 + low;
    if (!unreserved(byte)) return null;
    return .{ .byte = byte, .bytes = pair.bytes };
}

fn alphaNumeric(cp: u21) bool {
    return cp != '_' and unicode.isWord(cp);
}

fn hexValue(cp: u21) ?u8 {
    if (cp >= 'a' and cp <= 'f') return @intCast(cp - 'a' + 10);
    if (cp >= 'A' and cp <= 'F') return @intCast(cp - 'A' + 10);
    return unicode.decimalDigit(cp);
}

fn unreserved(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or std.mem.indexOfScalar(u8, "-._~", byte) != null;
}

fn safeUriByte(byte: u8, allow_percent: bool) bool {
    return unreserved(byte) or (allow_percent and byte == '%') or
        std.mem.indexOfScalar(u8, "!#$&'()*+,/:;=?@[]", byte) != null;
}

fn isRedirect(status: std.http.Status) bool {
    return switch (status) {
        .moved_permanently, .found, .see_other, .temporary_redirect, .permanent_redirect => true,
        else => false,
    };
}

fn componentBytes(component: std.Uri.Component) []const u8 {
    return switch (component) {
        .raw, .percent_encoded => |text| text,
    };
}

/// Match requests' origin/port rules without carrying credentials to a new
/// host or a downgrade. The conventional HTTP-to-HTTPS upgrade keeps auth.
fn stripAuth(old: std.Uri, new: std.Uri) bool {
    const old_host = if (old.host) |host| componentBytes(host) else "";
    const new_host = if (new.host) |host| componentBytes(host) else "";
    if (!std.ascii.eqlIgnoreCase(old_host, new_host)) return true;
    const old_http = std.ascii.eqlIgnoreCase(old.scheme, "http");
    const new_https = std.ascii.eqlIgnoreCase(new.scheme, "https");
    if (old_http and (old.port == null or old.port == 80) and new_https and
        (new.port == null or new.port == 443)) return false;
    if (!std.ascii.eqlIgnoreCase(old.scheme, new.scheme)) return true;
    const default_port: ?u16 = if (old_http) 80 else if (new_https) 443 else null;
    if ((old.port == null or old.port == default_port) and
        (new.port == null or new.port == default_port)) return false;
    return old.port != new.port;
}

fn redirectedHeaders(arena: Allocator, headers: []const std.http.Header, remove_auth: bool, remove_body: bool) Allocator.Error![]const std.http.Header {
    var count: usize = 0;
    for (headers) |header| {
        if (!dropRedirectHeader(header.name, remove_auth, remove_body)) count += 1;
    }
    if (count == headers.len) return headers;
    const filtered = try arena.alloc(std.http.Header, count);
    var index: usize = 0;
    for (headers) |header| {
        if (dropRedirectHeader(header.name, remove_auth, remove_body)) continue;
        filtered[index] = header;
        index += 1;
    }
    return filtered;
}

fn dropRedirectHeader(name: []const u8, remove_auth: bool, remove_body: bool) bool {
    return std.ascii.eqlIgnoreCase(name, "cookie") or
        (remove_auth and std.ascii.eqlIgnoreCase(name, "authorization")) or
        (remove_body and (std.ascii.eqlIgnoreCase(name, "content-type") or
            std.ascii.eqlIgnoreCase(name, "content-length") or
            std.ascii.eqlIgnoreCase(name, "transfer-encoding")));
}

/// Transport failures, HTTP 429 and server errors are retryable.
pub fn retryable(status: ?std.http.Status) bool {
    const s = status orelse return true;
    return s == .too_many_requests or @backingInt(s) >= 500;
}

test "URI quoting preserves requests Unicode and invalid percent rules" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const cases = .{
        .{ "http://x/%7e/%2f/a b?x=ä#🙂", "http://x/~/%2f/a%20b?x=%C3%A4#%F0%9F%99%82" },
        .{ "http://x/%GG/a%7E", "http://x/%25GG/a%257E" },
        .{ "http://x/%４１/%١f", "http://x/A/%%D9%A1f" },
        .{ "http://x/%éA/%7E", "http://x/%25%C3%A9A/%257E" },
        .{ "http://x/%_A/%", "http://x/%_A/%" },
        .{ "http://x/%A%41/%1/%%", "http://x/%AA/%1/%%" },
        .{ "http://x/\x00\x1f\x7f[]~!", "http://x/%00%1F%7F[]~!" },
    };
    inline for (cases) |case|
        try std.testing.expectEqualStrings(case[1], try requoteUri(arena.allocator(), case[0]));
}

test "redirect authorization follows host and default-port boundaries" {
    const cases = .{
        .{ "http://example.org/a", "https://example.org/b", false },
        .{ "http://example.org:80/a", "https://example.org:443/b", false },
        .{ "https://example.org/a", "https://EXAMPLE.ORG:443/b", false },
        .{ "http://example.org:8080/a", "http://example.org:8080/b", false },
        .{ "https://example.org/a", "http://example.org/b", true },
        .{ "http://example.org/a", "http://example.org:8080/b", true },
        .{ "https://example.org/a", "https://other.example.org/b", true },
    };
    inline for (cases) |case|
        try std.testing.expectEqual(case[2], stripAuth(try std.Uri.parse(case[0]), try std.Uri.parse(case[1])));
}

test "HTTP retry classification preserves terminal client errors" {
    try std.testing.expect(retryable(null));
    try std.testing.expect(retryable(.too_many_requests));
    try std.testing.expect(retryable(.service_unavailable));
    try std.testing.expect(!retryable(.unauthorized));
    try std.testing.expect(!retryable(.bad_request));
}
