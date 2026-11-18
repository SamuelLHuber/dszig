//! Shared transport for explicit Image and Audio URL constructors.
const std = @import("std");
const http = @import("http.zig");

pub const Options = struct {
    /// Inclusive size of the decompressed response in bytes.
    max_bytes: usize,
    /// Per-network-operation inactivity limit, not a whole-download deadline.
    timeout_ns: u64 = 30 * std.time.ns_per_s,
    max_redirects: u8 = 5,
};

/// `scratch` owns the MIME header; `client.allocator` owns the body. Only call
/// for trusted URLs and redirect targets. Retains the client's TLS trust.
pub fn download(
    scratch: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    options: Options,
) !http.Response {
    const uri = std.Uri.parse(url) catch return error.InvalidMediaUrl;
    if (!std.ascii.eqlIgnoreCase(uri.scheme, "http") and
        !std.ascii.eqlIgnoreCase(uri.scheme, "https")) return error.InvalidMediaUrl;
    if (uri.host == null or uri.user != null or uri.password != null)
        return error.InvalidMediaUrl;

    const transport: http.Transport = .{
        .client = client,
        .max_response_bytes = options.max_bytes,
        .timeout_ns = options.timeout_ns,
        .max_redirects = options.max_redirects,
        .normalize_url = true,
    };
    // Feed the original URL to Transport; it also quotes redirect locations.
    const response = try transport.send(scratch, .{ .url = url, .capture_content_type = true });
    errdefer client.allocator.free(response.body);
    const status = @backingInt(response.status);
    if (status < 200 or status >= 400) return error.HttpFailure;
    return response;
}
