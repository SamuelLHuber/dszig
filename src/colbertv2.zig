//! DSPy 3.5 ColBERTv2's stateless GET/POST retrieval and memory-cache semantics.
//! HTTP/TLS, inactivity timeouts, response bounds and redirects stay in http.zig.
//! Cache keys are native identities, not Python diskcache-compatible hashes.
//!
//! Upstream (DSPy 3.5.0): dspy/dsp/colbertv2.py
const assert = std.debug.assert;
const Workspace = @import("workspace.zig").Workspace;
const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const graph = @import("json_graph.zig");
const request_cache = @import("request_cache.zig");
const http = @import("http.zig");
const predict = @import("predict.zig");

pub const Port = union(enum) {
    string: []const u8,
    integer: i64,
};

pub const Options = struct {
    url: []const u8 = "http://0.0.0.0",
    port: ?Port = null,
    post_requests: bool = false,
    max_response_bytes: usize = 4 * 1024 * 1024,
    /// requests uses a ten-second per-operation connect/read timeout.
    timeout_ns: ?u64 = 10_000_000_000,
};

pub const CacheLayer = enum(u8) { outer, inner };

pub const Result = struct {
    /// One owned lease, independent of cache replacement/eviction/deinit.
    /// Use payload.allocator() for mutations allocating into nested containers.
    payload: *request_cache.Payload,
    /// New shallow dictionaries: top-level changes do not modify cached rows.
    /// Nested graph containers remain shared on a first result or inner hit.
    passages: []graph.Record,

    pub fn deinit(self: *Result) void {
        self.payload.release();
        self.* = undefined;
    }

    /// The slice belongs to arena; its graph values borrow this result's lease.
    /// POST deliberately does not synthesize absent long_text or require strings.
    pub fn simplify(self: *const Result, arena: Allocator) ![]graph.Value {
        const values = try arena.alloc(graph.Value, self.passages.len);
        errdefer arena.free(values);
        for (self.passages, values) |row, *value|
            value.* = row.get("long_text") orelse return error.MissingLongText;
        return values;
    }

    /// Consumes the caller's lease only on success.
    fn fromPayload(payload: *request_cache.Payload) !Result {
        if (payload.root != .array) return error.InvalidColBERTResponse;
        const arena = payload.allocator();
        const rows = payload.root.array.items;
        const passages = try arena.alloc(graph.Record, rows.len);
        for (rows, passages) |row, *passage| {
            if (row != .object) return error.InvalidColBERTResponse;
            passage.* = .empty;
            try passage.ensureTotalCapacity(arena, row.object.count());
            var entries = row.object.iterator();
            while (entries.next()) |entry|
                passage.putAssumeCapacity(entry.key_ptr.*, entry.value_ptr.*);
        }
        return .{ .payload = payload, .passages = passages };
    }
};

pub const ColBERTv2 = struct {
    client: *std.http.Client,
    cache: *request_cache.Cache,
    /// Fixed scratch for request URLs, bodies and responses: one per
    /// concurrent caller, at most 64, reserved by the caller.
    workspaces: []Workspace,
    busy: std.atomic.Value(u64) = .init(0),
    /// Borrowed unless init appends a truthy port into its supplied arena.
    url: []const u8 = "http://0.0.0.0",
    post_requests: bool = false,
    max_response_bytes: usize = 4 * 1024 * 1024,
    timeout_ns: ?u64 = 10_000_000_000,

    pub fn init(arena: Allocator, client: *std.http.Client, cache: *request_cache.Cache, workspaces: []Workspace, options: Options) !ColBERTv2 {
        if (workspaces.len == 0 or workspaces.len > 64) return error.InvalidWorkspaces;
        var url = options.url;
        if (options.port) |port| switch (port) {
            .string => |text| {
                if (text.len != 0) url = try std.fmt.allocPrint(arena, "{s}:{s}", .{ url, text });
            },
            .integer => |number| {
                if (number != 0) url = try std.fmt.allocPrint(arena, "{s}:{d}", .{ url, number });
            },
        };
        return .{
            .client = client,
            .cache = cache,
            .workspaces = workspaces,
            .url = url,
            .post_requests = options.post_requests,
            .max_response_bytes = options.max_response_bytes,
            .timeout_ns = options.timeout_ns,
        };
    }

    pub fn forward(self: *ColBERTv2, query: []const u8, k: i64) !Result {
        const outer_key = self.cacheKey(query, k, .outer);
        if (try self.cache.get(outer_key)) |payload| {
            errdefer payload.release();
            return Result.fromPayload(payload);
        }
        const inner_key = self.cacheKey(query, k, .inner);
        if (try self.cache.get(inner_key)) |payload| {
            errdefer payload.release();
            // The inner cache returned a deep clone. Only the new outer entry
            // shares this returned graph; the inner's original stays isolated.
            try self.cache.put(outer_key, payload);
            return Result.fromPayload(payload);
        }
        // This assertion lives inside the decorated source function. Seeded
        // cache hits, including k > 100, bypass it; POST has no such assertion.
        if (!self.post_requests and k > 100) return error.KTooLarge;
        const payload = try self.fetch(query, k);
        errdefer payload.release();
        // Both source decorators store the same raw return object on a miss.
        // A first shallow result's nested mutations therefore reach both tiers.
        try self.cache.put(inner_key, payload);
        try self.cache.put(outer_key, payload);
        return Result.fromPayload(payload);
    }

    /// Adapt to Retrieve/RAG's string-only interface. Unlike forward/simplify,
    /// this boundary rejects non-string long_text rather than stringifying it.
    pub fn retriever(self: *ColBERTv2) predict.Retriever {
        return .{ .ptr = self, .vtable = &.{ .search = search } };
    }

    fn search(ptr: *const anyopaque, arena: Allocator, query: []const u8, k: i64) anyerror![]const []const u8 {
        const state: *const ColBERTv2 = @ptrCast(@alignCast(ptr));
        const self = @constCast(state);
        var result = try self.forward(query, k);
        defer result.deinit();
        const passages = try arena.alloc([]const u8, result.passages.len);
        var completed: usize = 0;
        errdefer {
            for (passages[0..completed]) |passage| arena.free(passage);
            arena.free(passages);
        }
        // Read simplify's long_text values directly, avoiding a temporary
        // []graph.Value. Returned bytes must outlive the result's payload.
        for (result.passages, passages) |row, *passage| {
            const text = row.get("long_text") orelse return error.MissingLongText;
            if (text != .string) return error.NonStringLongText;
            passage.* = try arena.dupe(u8, text.string);
            completed += 1;
        }
        return passages;
    }

    /// Exact original URL/query/signed k, method and decorator layer identity.
    /// Length-prefix fields avoid delimiter ambiguity without allocating a key.
    pub fn cacheKey(self: *const ColBERTv2, query: []const u8, k: i64, layer: CacheLayer) request_cache.Key {
        var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
        hash.update("dszig.colbertv2.v1\x00");
        hash.update(&.{ @backingInt(layer), @intFromBool(self.post_requests) });
        hashString(&hash, self.url);
        hashString(&hash, query);
        var signed_k: [8]u8 = undefined;
        std.mem.writeInt(i64, &signed_k, k, .little);
        hash.update(&signed_k);
        var key: request_cache.Key = undefined;
        hash.final(&key);
        return key;
    }

    /// More concurrent fetches than workspaces is `error.WorkspacesBusy`.
    fn acquire(self: *ColBERTv2) error{WorkspacesBusy}!u6 {
        assert(self.workspaces.len > 0 and self.workspaces.len <= 64);
        const all: u64 = if (self.workspaces.len == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(self.workspaces.len)) - 1;
        var busy = self.busy.load(.monotonic);
        while (true) {
            const free = all & ~busy;
            if (free == 0) return error.WorkspacesBusy;
            const slot: u6 = @intCast(@ctz(free));
            busy = self.busy.cmpxchgWeak(busy, busy | (@as(u64, 1) << slot), .acquire, .monotonic) orelse return slot;
        }
    }

    fn releaseSlot(self: *ColBERTv2, slot: u6) void {
        const previous = self.busy.fetchAnd(~(@as(u64, 1) << slot), .release);
        assert(previous & (@as(u64, 1) << slot) != 0);
    }

    fn fetch(self: *ColBERTv2, query: []const u8, k: i64) !*request_cache.Payload {
        const slot = try self.acquire();
        defer self.releaseSlot(slot);
        var scratch = self.workspaces[slot].open();
        defer scratch.close();
        const arena = scratch.allocator();
        const url = if (self.post_requests) self.url else try getRequestUrl(arena, self.url, query, k);
        const body = if (self.post_requests) try postRequestBody(arena, query, k) else null;
        const transport: http.Transport = .{
            .client = self.client,
            .max_response_bytes = self.max_response_bytes,
            .max_retries = 0,
            .max_redirects = 30,
            .normalize_url = true,
            .timeout_ns = self.timeout_ns,
            .success = .any,
        };
        const response = try transport.send(arena, .{
            .url = url,
            .method = if (self.post_requests) .POST else .GET,
            .body = body,
            .content_type = if (self.post_requests) "application/json; charset=utf-8" else null,
        });
        defer self.client.allocator.free(response.body);
        const status = @backingInt(response.status);
        // requests.raise_for_status rejects only the 4xx and 5xx classes.
        if (status >= 400 and status < 600) return error.HttpFailure;
        return parsePayload(self.client.allocator, response.body, self.post_requests, k);
    }
};

fn hashString(hash: *std.crypto.hash.sha2.Sha256, text: []const u8) void {
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, @intCast(text.len), .little);
    hash.update(&length);
    hash.update(text);
}

fn parsePayload(gpa: Allocator, bytes: []const u8, post_requests: bool, k: i64) !*request_cache.Payload {
    const payload = try request_cache.Payload.create(gpa);
    errdefer payload.release();
    const arena = payload.allocator();
    const response = graph.parse(arena, bytes) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidColBERTResponse,
    };
    if (response != .object) return error.InvalidColBERTResponse;
    if (response.object.get("error")) |server_error| {
        if (try graph.truthy(server_error)) return error.ColBERTServerError;
    }
    const topk = response.object.get("topk") orelse return error.InvalidColBERTResponse;
    if (topk != .array) return error.InvalidColBERTResponse;
    const first_length = sliceLength(topk.array.items.len, k);
    if (!post_requests) {
        // Validate and transform EVERY row in the first slice, even those the
        // second negative slice drops. The parsed response is private, so its
        // row maps can be reused before the public shallow copies are made.
        for (topk.array.items[0..first_length]) |row| {
            if (row != .object) return error.InvalidColBERTResponse;
            const text = row.object.get("text") orelse return error.MissingText;
            try row.object.put(arena, "long_text", text);
        }
    }
    topk.array.items.len = if (post_requests) first_length else sliceLength(first_length, k);
    payload.root = topk;
    return payload;
}

/// Python [:k], without negating minInt(i64) or narrowing a large list length.
fn sliceLength(length: usize, k: i64) usize {
    if (k >= 0) {
        const positive: u64 = @intCast(k);
        if (positive >= length) return length;
        return @intCast(positive);
    }
    const magnitude = @as(u64, @intCast(-(k + 1))) + 1;
    if (magnitude >= length) return 0;
    return length - @as(usize, @intCast(magnitude));
}

fn getRequestUrl(arena: Allocator, url: []const u8, query: []const u8, k: i64) ![]const u8 {
    _ = try std.unicode.Utf8View.init(query);
    var encoded: Writer.Allocating = .init(arena);
    writeGetUrl(&encoded.writer, url, query, k) catch return error.OutOfMemory;
    return encoded.written();
}

fn writeGetUrl(writer: *Writer, url: []const u8, query: []const u8, k: i64) Writer.Error!void {
    const fragment_at = std.mem.indexOfScalar(u8, url, '#') orelse url.len;
    const prefix = url[0..fragment_at];
    try writer.writeAll(prefix);
    if (std.mem.indexOfScalar(u8, prefix, '?')) |question_at| {
        if (question_at + 1 < prefix.len) try writer.writeByte('&');
    } else try writer.writeByte('?');
    try writer.writeAll("query=");
    // urllib.parse.urlencode uses quote_plus with UTF-8 and uppercase hex.
    const hex = "0123456789ABCDEF";
    for (query) |byte| switch (byte) {
        'a'...'z', 'A'...'Z', '0'...'9', '_', '.', '-', '~' => try writer.writeByte(byte),
        ' ' => try writer.writeByte('+'),
        else => try writer.writeAll(&.{ '%', hex[byte >> 4], hex[byte & 15] }),
    };
    try writer.print("&k={d}", .{k});
    try writer.writeAll(url[fragment_at..]);
}

fn postRequestBody(arena: Allocator, query: []const u8, k: i64) ![]const u8 {
    _ = try std.unicode.Utf8View.init(query);
    var encoded: Writer.Allocating = .init(arena);
    writePostBody(&encoded.writer, query, k) catch return error.OutOfMemory;
    return encoded.written();
}

fn writePostBody(writer: *Writer, query: []const u8, k: i64) Writer.Error!void {
    // requests' json= payload uses json.dumps' defaults: ASCII escapes and
    // comma/colon spaces, insertion order query then k, lowercase UTF-16 hex.
    try writer.writeAll("{\"query\": \"");
    var codepoints = std.unicode.Utf8View.initUnchecked(query).iterator();
    while (codepoints.nextCodepoint()) |cp| switch (cp) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0x08 => try writer.writeAll("\\b"),
        0x0c => try writer.writeAll("\\f"),
        else => {
            if (cp >= 0x20 and cp <= 0x7e) {
                try writer.writeByte(@intCast(cp));
            } else if (cp <= 0xffff) {
                try writer.print("\\u{x:0>4}", .{cp});
            } else {
                const pair = cp - 0x10000;
                try writer.print("\\u{x:0>4}\\u{x:0>4}", .{
                    @as(u16, @intCast(0xd800 + (pair >> 10))),
                    @as(u16, @intCast(0xdc00 + (pair & 0x3ff))),
                });
            }
        },
    };
    try writer.print("\", \"k\": {d}}}", .{k});
}

const testing = std.testing;

test "ColBERT wire encoding matches actual pinned requests receipts" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const query = "café 雪 😀 /?&=+%~\"\\\n\t\x00\x1f\x7f\xe2\x80\xa8";
    const url = try getRequestUrl(arena, "http://example.invalid/rows?existing=one%20two#fragment", query, -1);
    try testing.expectEqualStrings(
        "http://example.invalid/rows?existing=one%20two&query=caf%C3%A9+%E9%9B%AA+%F0%9F%98%80+%2F%3F%26%3D%2B%25~%22%5C%0A%09%00%1F%7F%E2%80%A8&k=-1#fragment",
        url,
    );
    try testing.expectEqualStrings(
        "{\"query\": \"caf\\u00e9 \\u96ea \\ud83d\\ude00 /?&=+%~\\\"\\\\\\n\\t\\u0000\\u001f\\u007f\\u2028\", \"k\": -1}",
        try postRequestBody(arena, query, -1),
    );
    try testing.expectEqualStrings("http://example.invalid/?query=&k=0", try getRequestUrl(arena, "http://example.invalid/?", "", 0));
}

test "ColBERT GET slices twice and POST preserves arbitrary long_text" {
    const bytes =
        \\{"topk":[{"text":{"tokens":[1]},"long_text":23},{"text":false,"long_text":null},{"text":["nested"],"long_text":["server"]},{"text":"delta"},{"text":"epsilon"}]}
    ;
    const get = try parsePayload(testing.allocator, bytes, false, -1);
    defer get.release();
    var get_result = try Result.fromPayload(get);
    // fromPayload consumes get's existing lease; retain for the two owners here.
    get.retain();
    defer get_result.deinit();
    try testing.expectEqual(@as(usize, 3), get_result.passages.len);
    try testing.expect(!get_result.passages[1].get("long_text").?.bool);
    const post = try parsePayload(testing.allocator, bytes, true, -1);
    defer post.release();
    var post_result = try Result.fromPayload(post);
    post.retain();
    defer post_result.deinit();
    try testing.expectEqual(@as(usize, 4), post_result.passages.len);
    try testing.expectEqual(@as(i64, 23), post_result.passages[0].get("long_text").?.integer);
    try testing.expectError(error.MissingLongText, post_result.simplify(testing.allocator));
    try testing.expectEqual(@as(usize, 0), sliceLength(5, std.math.minInt(i64)));
    try testing.expectEqual(@as(usize, 5), sliceLength(5, std.math.maxInt(i64)));
    try testing.expectError(error.MissingText, parsePayload(testing.allocator,
        \\{"topk":[{"text":"alpha"},{"text":"beta"},{"text":"gamma"},{"long_text":"dropped by second slice"},{"text":"epsilon"}]}
    , false, -1));
}

test "concurrent fetches take distinct fixed workspaces" {
    var buffers: [2][64]u8 = undefined;
    var workspaces: [2]Workspace = .{ .init(&buffers[0]), .init(&buffers[1]) };
    var colbert: ColBERTv2 = .{ .client = undefined, .cache = undefined, .workspaces = &workspaces };
    const first = try colbert.acquire();
    const second = try colbert.acquire();
    try std.testing.expect(first != second);
    try std.testing.expectError(error.WorkspacesBusy, colbert.acquire());
    colbert.releaseSlot(first);
    try std.testing.expectEqual(first, try colbert.acquire());
    colbert.releaseSlot(first);
    colbert.releaseSlot(second);
}
