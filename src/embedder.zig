//! Native `dspy.Embedder`: custom Zig callables and OpenAI/Azure HTTP embeddings.
//!
//! Inputs, constructor kwargs, the HTTP client, and callback context are borrowed.
//! Keep them alive until `deinit`; do not copy an Embedder after it starts caching.
//! The persistent allocator owns the per-instance, in-memory batch cache. Every
//! result is contiguous float32 storage owned by the caller's output arena,
//! independent of that cache. Concurrent calls require external synchronization.
//!
//! ```zig
//! const embedding = @import("embedder.zig");
//! const Local = struct {
//!     fn run(_: *anyopaque, arena: std.mem.Allocator, texts: []const []const u8,
//!         _: @import("value.zig").Record) anyerror!embedding.Batch
//!     {
//!         const values = try arena.alloc(f32, texts.len);
//!         for (texts, values) |text, *out| out.* = @floatFromInt(text.len);
//!         return .{ .float32 = .{ .values = values, .rows = texts.len, .columns = 1 } };
//!     }
//! };
//! var context: u8 = 0;
//! var local = embedding.Embedder.initCustom(gpa,
//!     .{ .context = &context, .embed = Local.run }, .{});
//! defer local.deinit();
//! const vectors = try local.embed(arena, &.{ "hello", "world" }, .{});
//! _ = vectors.row(0);
//!
//! var kwargs: @import("value.zig").Record = .empty;
//! try kwargs.put(arena, "api_base", .{ .string = "http://127.0.0.1:8000/v1" });
//! try kwargs.put(arena, "api_key", .{ .string = "explicit-key" });
//! try kwargs.put(arena, "dimensions", .{ .integer = 2 });
//! var hosted = try embedding.Embedder.initHosted(gpa,
//!     .{ .client = &http_client, .model = "openai/text-embedding-3-small" },
//!     .{ .kwargs = kwargs });
//! defer hosted.deinit();
//! const actual = try hosted.embed(arena, &.{ "hello", "world" }, .{});
//! _ = actual;
//! ```
//!
//! Hosted kwargs mirror the reference's SDK client/request split: api_key,
//! base_url/api_base, organization, project, timeout, max_retries,
//! default_headers/default_query, and Azure azure_endpoint/api_version/
//! azure_ad_token are client options. Request options are dimensions, user,
//! encoding_format, extra_headers, extra_query, and extra_body. Unknown options
//! fail rather than disappear. Python objects are not JSON options; use the
//! native Azure token-provider field for a callable credential. No process
//! environment is read: supply only intended values in HostedOptions.environment.
//! Timeout is numeric seconds per network operation; null disables it. The SDK
//! default is 600 seconds with a 5-second connect timeout. Native retry counts
//! are bounded by the shared transport's integer type; invalid/out-of-range
//! options fail explicitly. Cache hashes are native, not DSPy's Python
//! disk-cache serialization.

const std = @import("std");
const lm = @import("lm.zig");
const value = @import("value.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Record = value.Record;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Key = [32]u8;

pub const Matrix = struct {
    values: []const f32,
    rows: usize,
    columns: usize,

    pub fn row(self: Matrix, index: usize) []const f32 {
        std.debug.assert(index < self.rows);
        return self.values[index * self.columns ..][0..self.columns];
    }
};

const empty_matrix: Matrix = .{ .values = &.{}, .rows = 0, .columns = 0 };

/// Custom results are converted to float32 only after successful batches have
/// been cached, just like DSPy's `_postprocess`. JSON accepts rectangular rows
/// of numbers, numeric strings, booleans, and null (NumPy converts null to NaN).
pub const Batch = union(enum) {
    float32: Matrix,
    float64: struct { values: []const f64, rows: usize, columns: usize },
    json: Value,
};

pub const CustomFn = *const fn (*anyopaque, Allocator, []const []const u8, Record) anyerror!Batch;
pub const Custom = struct { context: *anyopaque, embed: CustomFn };

pub const CallOptions = struct {
    batch_size: ?isize = null,
    caching: ?bool = null,
    kwargs: Record = .empty,
};

pub const Options = struct {
    batch_size: isize = 200,
    caching: bool = true,
    kwargs: Record = .empty,
    /// LRU batch entries; zero disables storage without changing callback kwargs.
    cache_capacity: usize = 1_000_000,
};

pub const EnvironmentEntry = struct { name: []const u8, value: []const u8 };

pub const TokenProvider = struct {
    context: *anyopaque,
    /// Called for each HTTP attempt of an uncached Azure batch, including
    /// retries. Tokens must be nonempty and remain valid for the request arena.
    /// Callback errors retain their identity.
    get: *const fn (*anyopaque, Allocator) anyerror![]const u8,
};

pub const HostedOptions = struct {
    client: *std.http.Client,
    model: []const u8,
    environment: []const EnvironmentEntry = &.{},
    azure_ad_token_provider: ?TokenProvider = null,
    max_response_bytes: usize = 4 * 1024 * 1024,
    retry_base_ms: u32 = 500,
};

const HostedModel = struct { azure: bool, name: []const u8 };
const Backend = union(enum) { custom: Custom, hosted: HostedOptions };
const CacheEntry = struct { memory: std.heap.ArenaAllocator, batch: Batch };

pub const Embedder = struct {
    allocator: Allocator,
    backend: Backend,
    options: Options,
    entries: std.AutoArrayHashMapUnmanaged(Key, CacheEntry) = .empty,
    hits: usize = 0,
    misses: usize = 0,

    pub fn initCustom(allocator: Allocator, backend: Custom, options: Options) Embedder {
        return .{ .allocator = allocator, .backend = .{ .custom = backend }, .options = options };
    }

    pub fn initHosted(allocator: Allocator, hosted: HostedOptions, options: Options) !Embedder {
        const model = try hostedModel(hosted.model);
        if (!model.azure and hosted.azure_ad_token_provider != null) return error.InvalidClientOption;
        return .{ .allocator = allocator, .backend = .{ .hosted = hosted }, .options = options };
    }

    pub fn deinit(self: *Embedder) void {
        for (self.entries.values()) |*entry| entry.memory.deinit();
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn embed(self: *Embedder, arena: Allocator, inputs: []const []const u8, options: CallOptions) anyerror!Matrix {
        // The native Matrix contract makes the Python empty (0,) array explicit.
        if (inputs.len == 0) return empty_matrix;
        var memory: std.heap.ArenaAllocator = .init(self.allocator);
        defer memory.deinit();
        const scratch = memory.allocator();
        const kwargs = try self.mergedKwargs(scratch, options.kwargs);
        const requested = options.batch_size orelse 0;
        const batch_size = if (requested == 0) self.options.batch_size else requested;
        if (batch_size == 0) return error.InvalidBatchSize;
        // Python range(0, len(inputs), negative_step) has no iterations.
        if (batch_size < 0) return empty_matrix;
        const size: usize = @intCast(batch_size);
        const count = inputs.len / size + @intFromBool(inputs.len % size != 0);
        const batches = try scratch.alloc(Batch, count);
        const caching = (options.caching orelse self.options.caching) and self.options.cache_capacity != 0;
        var retired: std.ArrayList(std.heap.ArenaAllocator) = .empty;
        defer for (retired.items) |*entry| entry.deinit();
        var start: usize = 0;
        for (batches) |*batch| {
            const end = start + @min(size, inputs.len - start);
            const texts = inputs[start..end];
            // Custom callbacks and native token providers can mutate shared
            // nested kwargs. Snapshot each batch before executing it, as DSPy's
            // request_cache does; a once-per-call hash would alias later batches.
            const key = if (caching) batchKey(try self.baseCacheHash(scratch, kwargs), texts) else null;
            if (key) |k| {
                if (self.entries.fetchOrderedRemove(k)) |hit| {
                    self.entries.putAssumeCapacity(k, hit.value);
                    self.hits += 1;
                    batch.* = hit.value.batch;
                    start = end;
                    continue;
                }
                self.misses += 1;
            }
            batch.* = switch (self.backend) {
                .custom => |custom| blk: {
                    var batch_kwargs: Record = .empty;
                    try mergeRecord(scratch, &batch_kwargs, kwargs);
                    break :blk try custom.embed(custom.context, scratch, texts, batch_kwargs);
                },
                .hosted => |hosted| try computeHosted(scratch, hosted, kwargs, texts),
            };
            // A later backend or postprocessing failure must retain this batch.
            if (key) |k| try self.insert(scratch, &retired, k, batch.*);
            start = end;
        }
        return postprocess(arena, batches);
    }

    /// Python's string-input path returns the first vector, not a 1-row Matrix.
    pub fn embedOne(self: *Embedder, arena: Allocator, input: []const u8, options: CallOptions) anyerror![]const f32 {
        const matrix = try self.embed(arena, &.{input}, options);
        if (matrix.rows == 0) return error.EmptyEmbedding;
        return matrix.row(0);
    }

    fn mergedKwargs(self: *const Embedder, arena: Allocator, call: Record) !Record {
        var merged: Record = .empty;
        try mergeRecord(arena, &merged, self.options.kwargs);
        try mergeRecord(arena, &merged, call);
        if (self.backend == .custom) return merged;
        const hosted = self.backend.hosted;
        const model = try hostedModel(hosted.model);
        const endpoints: []const []const u8 = if (model.azure)
            &.{ "api_base", "base_url", "azure_endpoint" }
        else
            &.{ "api_base", "base_url" };
        var call_endpoint = false;
        for (endpoints) |name| if (nonnull(call.get(name))) {
            call_endpoint = true;
            break;
        };
        if (call_endpoint) {
            for (endpoints) |name| if (!call.contains(name)) {
                _ = merged.orderedRemove(name);
            };
        }
        if (nonnull(merged.get("api_base"))) {
            for (endpoints[1..]) |name| if (nonnull(merged.get(name))) return error.ConflictingEndpoints;
        }
        if (merged.fetchOrderedRemove("api_base")) |entry| {
            if (entry.value != .null) try merged.put(arena, if (model.azure) "azure_endpoint" else "base_url", entry.value);
        }
        if (model.azure) {
            try environmentDefault(arena, &merged, "api_version", envOr(hosted.environment, "OPENAI_API_VERSION", "AZURE_API_VERSION"));
            const provider = try nativeTokenProvider(hosted, merged);
            if (!nonnull(merged.get("azure_ad_token")) and provider == null and env(hosted.environment, "AZURE_OPENAI_AD_TOKEN") == null) {
                try environmentDefault(arena, &merged, "api_key", envOr(hosted.environment, "AZURE_OPENAI_API_KEY", "AZURE_API_KEY"));
            }
            if (!truthy(merged.get("base_url"))) {
                try environmentDefault(arena, &merged, "azure_endpoint", envOr(hosted.environment, "AZURE_OPENAI_ENDPOINT", "AZURE_API_BASE"));
            }
        } else {
            const configured = envOr(hosted.environment, "OPENAI_BASE_URL", "OPENAI_API_BASE") orelse "";
            const endpoint = if (configured.len == 0) "https://api.openai.com/v1" else configured;
            try environmentDefault(arena, &merged, "base_url", endpoint);
            try environmentDefault(arena, &merged, "api_key", env(hosted.environment, "OPENAI_API_KEY"));
        }
        if (!merged.contains("encoding_format")) try merged.put(arena, "encoding_format", .{ .string = "float" });
        return merged;
    }

    fn baseCacheHash(self: *const Embedder, arena: Allocator, kwargs: Record) !Sha256 {
        var filtered: Record = .empty;
        for (kwargs.keys(), kwargs.values()) |name, item| {
            if (!credentialKey(name)) try filtered.put(arena, name, item);
        }
        var encoded: std.Io.Writer.Allocating = .init(arena);
        try writeCanonical(arena, &encoded.writer, .{ .object = filtered }, 0);
        var hash = Sha256.init(.{});
        switch (self.backend) {
            .custom => hash.update("custom\x00"),
            .hosted => |hosted| {
                hash.update("hosted\x00");
                hashPart(&hash, hosted.model);
            },
        }
        hashPart(&hash, encoded.written());
        return hash;
    }

    fn insert(self: *Embedder, scratch: Allocator, retired: *std.ArrayList(std.heap.ArenaAllocator), key: Key, batch: Batch) !void {
        var owned: std.heap.ArenaAllocator = .init(self.allocator);
        errdefer owned.deinit();
        const copy = try cloneBatch(owned.allocator(), batch);
        try self.entries.ensureUnusedCapacity(self.allocator, 1);
        if (self.entries.count() >= self.options.cache_capacity) {
            // An earlier hit in this call may still borrow the evicted batch.
            // Retire its owner until postprocessing rather than copying values.
            try retired.append(scratch, self.entries.values()[0].memory);
            self.entries.orderedRemoveAt(0);
        }
        self.entries.putAssumeCapacity(key, .{ .memory = owned, .batch = copy });
    }
};

fn hostedModel(model: []const u8) !HostedModel {
    if (std.mem.indexOfScalar(u8, model, '/')) |slash| {
        const provider = model[0..slash];
        const name = model[slash + 1 ..];
        if (name.len == 0) return error.InvalidHostedModel;
        if (std.mem.eql(u8, provider, "openai")) return .{ .azure = false, .name = name };
        if (std.mem.eql(u8, provider, "azure")) return .{ .azure = true, .name = name };
        return error.InvalidHostedModel;
    }
    return .{ .azure = false, .name = model };
}

fn env(environment: []const EnvironmentEntry, name: []const u8) ?[]const u8 {
    // A list is convenient for literals; later entries have normal map precedence.
    var i = environment.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, environment[i].name, name)) return environment[i].value;
    }
    return null;
}

fn envOr(environment: []const EnvironmentEntry, first: []const u8, second: []const u8) ?[]const u8 {
    if (env(environment, first)) |s| if (s.len != 0) return s;
    return env(environment, second);
}

fn nonnull(item: ?Value) bool {
    return if (item) |v| v != .null else false;
}

fn truthy(item: ?Value) bool {
    const v = item orelse return false;
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .string, .number_string => |s| s.len != 0,
        .array => |a| a.items.len != 0,
        .object => |o| o.count() != 0,
        .custom => |custom| custom.truthy(),
    };
}

fn mergeRecord(arena: Allocator, output: *Record, input: Record) Allocator.Error!void {
    for (input.keys(), input.values()) |name, item| try output.put(arena, name, item);
}

fn environmentDefault(arena: Allocator, kwargs: *Record, name: []const u8, fallback: ?[]const u8) !void {
    if (!truthy(kwargs.get(name))) try kwargs.put(arena, name, if (fallback) |s| .{ .string = s } else .null);
}

fn nativeTokenProvider(hosted: HostedOptions, kwargs: Record) !?TokenProvider {
    if (kwargs.get("azure_ad_token_provider")) |item| {
        if (item != .null) return error.InvalidClientOption;
        return null;
    }
    return hosted.azure_ad_token_provider;
}

fn nullableString(kwargs: Record, name: []const u8) !?[]const u8 {
    const item = kwargs.get(name) orelse return null;
    return switch (item) {
        .null => null,
        .string => |s| s,
        else => error.InvalidClientOption,
    };
}

fn objectOption(kwargs: Record, name: []const u8) !Record {
    const item = kwargs.get(name) orelse return .empty;
    return switch (item) {
        .null => .empty,
        .object => |o| o,
        else => error.InvalidClientOption,
    };
}

fn credentialKey(name: []const u8) bool {
    return std.mem.eql(u8, name, "api_key") or std.mem.eql(u8, name, "azure_ad_token") or std.mem.eql(u8, name, "azure_ad_token_provider");
}

fn hashPart(hash: *Sha256, text: []const u8) void {
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, @intCast(text.len), .little);
    hash.update(&length);
    hash.update(text);
}

fn batchKey(base: Sha256, inputs: []const []const u8) Key {
    var hash = base;
    for (inputs) |input| hashPart(&hash, input);
    var key: Key = undefined;
    hash.final(&key);
    return key;
}

fn writeCanonical(arena: Allocator, writer: *std.Io.Writer, item: Value, depth: usize) anyerror!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    switch (item) {
        .object => |object| {
            const names = try arena.dupe([]const u8, object.keys());
            std.mem.sort([]const u8, names, {}, struct {
                fn less(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.less);
            try writer.writeByte('{');
            for (names, 0..) |name, i| {
                if (i != 0) try writer.writeByte(',');
                try pyjson.writeJsonString(writer, name);
                try writer.writeByte(':');
                try writeCanonical(arena, writer, object.get(name).?, depth + 1);
            }
            try writer.writeByte('}');
        },
        .array => |array| {
            try writer.writeByte('[');
            for (array.items, 0..) |child, i| {
                if (i != 0) try writer.writeByte(',');
                try writeCanonical(arena, writer, child, depth + 1);
            }
            try writer.writeByte(']');
        },
        .float => |number| if (std.math.isFinite(number)) try pyjson.writeJson(writer, item, .minified) else try writer.writeAll("null"),
        else => try pyjson.writeJson(writer, item, .minified),
    }
}

fn cloneBatch(arena: Allocator, batch: Batch) Allocator.Error!Batch {
    return switch (batch) {
        .float32 => |matrix| .{ .float32 = .{ .values = try arena.dupe(f32, matrix.values), .rows = matrix.rows, .columns = matrix.columns } },
        .float64 => |matrix| .{ .float64 = .{ .values = try arena.dupe(f64, matrix.values), .rows = matrix.rows, .columns = matrix.columns } },
        .json => |item| .{ .json = try value.cloneValue(arena, item) },
    };
}

const Shape = struct { rows: usize, columns: usize };

fn matrixLength(rows: usize, columns: usize) !usize {
    return std.math.mul(usize, rows, columns) catch return error.InvalidEmbeddingShape;
}

fn shape(batch: Batch) !Shape {
    return switch (batch) {
        .float32 => |m| blk: {
            if (try matrixLength(m.rows, m.columns) != m.values.len) return error.InvalidEmbeddingShape;
            break :blk .{ .rows = m.rows, .columns = m.columns };
        },
        .float64 => |m| blk: {
            if (try matrixLength(m.rows, m.columns) != m.values.len) return error.InvalidEmbeddingShape;
            break :blk .{ .rows = m.rows, .columns = m.columns };
        },
        .json => |item| blk: {
            if (item != .array) return error.InvalidEmbeddingShape;
            var columns: ?usize = null;
            for (item.array.items) |row| {
                if (row != .array) return error.InvalidEmbeddingShape;
                if (columns) |n| {
                    if (row.array.items.len != n) return error.InvalidEmbeddingShape;
                } else columns = row.array.items.len;
            }
            break :blk .{ .rows = item.array.items.len, .columns = columns orelse 0 };
        },
    };
}

fn numericFloat(item: Value) !f32 {
    return switch (item) {
        .integer => |n| @floatFromInt(n),
        .float => |n| @floatCast(n),
        .number_string, .string => |s| std.fmt.parseFloat(f32, std.mem.trim(u8, s, " \t\r\n\x0b\x0c")) catch return error.InvalidEmbeddingNumber,
        .bool => |b| @floatFromInt(@intFromBool(b)),
        .null => std.math.nan(f32),
        else => error.InvalidEmbeddingNumber,
    };
}

fn postprocess(arena: Allocator, batches: []const Batch) !Matrix {
    var rows: usize = 0;
    var columns: ?usize = null;
    for (batches) |batch| {
        const s = try shape(batch);
        if (s.rows == 0) continue;
        if (columns) |n| {
            if (s.columns != n) return error.InvalidEmbeddingShape;
        } else columns = s.columns;
        rows = std.math.add(usize, rows, s.rows) catch return error.InvalidEmbeddingShape;
    }
    if (rows == 0) return empty_matrix;
    const width = columns.?;
    const output = try arena.alloc(f32, try matrixLength(rows, width));
    errdefer arena.free(output);
    var offset: usize = 0;
    for (batches) |batch| switch (batch) {
        .float32 => |matrix| {
            @memcpy(output[offset..][0..matrix.values.len], matrix.values);
            offset += matrix.values.len;
        },
        .float64 => |matrix| {
            for (matrix.values) |number| {
                output[offset] = @floatCast(number);
                offset += 1;
            }
        },
        .json => |item| for (item.array.items) |row| {
            for (row.array.items) |number| {
                output[offset] = try numericFloat(number);
                offset += 1;
            }
        },
    };
    return .{ .values = output, .rows = rows, .columns = width };
}

const HostedRequest = struct {
    transport: lm.OpenAI,
    path: []const u8,
    body: Value,
    headers: []const std.http.Header,
};

fn clientKey(name: []const u8, azure: bool) bool {
    for ([_][]const u8{ "api_key", "base_url", "organization", "project", "timeout", "max_retries", "default_headers", "default_query" }) |key| {
        if (std.mem.eql(u8, name, key)) return true;
    }
    if (azure) for ([_][]const u8{ "azure_endpoint", "api_version", "azure_ad_token", "azure_ad_token_provider" }) |key| {
        if (std.mem.eql(u8, name, key)) return true;
    };
    return false;
}

fn prepareRequest(arena: Allocator, hosted: HostedOptions, kwargs: Record, inputs: []const []const u8) anyerror!HostedRequest {
    const model = try hostedModel(hosted.model);
    var body: Record = .empty;
    try body.put(arena, "model", .{ .string = model.name });
    var input: @import("runtime_value.zig").Array = .init(arena);
    try input.ensureTotalCapacity(inputs.len);
    for (inputs) |text| input.appendAssumeCapacity(.{ .string = text });
    try body.put(arena, "input", .{ .array = input });
    for (kwargs.keys(), kwargs.values()) |name, item| {
        if (clientKey(name, model.azure)) continue;
        if (std.mem.eql(u8, name, "model") or std.mem.eql(u8, name, "input")) return error.DuplicateRequestArgument;
        if (std.mem.eql(u8, name, "dimensions") or std.mem.eql(u8, name, "user") or std.mem.eql(u8, name, "encoding_format")) {
            try body.put(arena, name, item);
        } else if (!std.mem.eql(u8, name, "extra_headers") and !std.mem.eql(u8, name, "extra_query") and !std.mem.eql(u8, name, "extra_body")) {
            return error.UnknownRequestOption;
        }
    }
    try mergeRecord(arena, &body, try objectOption(kwargs, "extra_body"));

    var query: Record = .empty;
    try mergeRecord(arena, &query, try objectOption(kwargs, "default_query"));
    var endpoint = try nullableString(kwargs, "base_url");
    var path: []const u8 = "embeddings";
    if (model.azure) {
        const version = try nullableString(kwargs, "api_version") orelse return error.MissingApiVersion;
        try query.put(arena, "api-version", .{ .string = version });
        const azure_endpoint = try nullableString(kwargs, "azure_endpoint");
        if (endpoint != null and azure_endpoint != null) return error.ConflictingEndpoints;
        if (endpoint == null) endpoint = try std.fmt.allocPrint(arena, "{s}/openai", .{std.mem.trimEnd(u8, azure_endpoint orelse return error.MissingEndpoint, "/")});
        const parsed = std.Uri.parse(endpoint.?) catch return error.InvalidEndpoint;
        const endpoint_path = try parsed.path.toRawMaybeAlloc(arena);
        if (std.mem.indexOf(u8, endpoint_path, "/deployments") == null) {
            const deployment = body.get("model") orelse return error.InvalidHostedModel;
            if (deployment != .null) {
                const name = switch (deployment) {
                    .string => |s| s,
                    .bool => |b| if (b) "true" else "false",
                    else => blk: {
                        var text: std.Io.Writer.Allocating = .init(arena);
                        try pyjson.writeStr(&text.writer, deployment);
                        break :blk text.written();
                    },
                };
                const escaped = try escapeDeployment(arena, name);
                path = try std.fmt.allocPrint(arena, "deployments/{s}/embeddings", .{escaped});
            }
        }
    }
    try mergeRecord(arena, &query, try objectOption(kwargs, "extra_query"));
    const joined = try endpointPath(arena, endpoint orelse return error.MissingEndpoint, path, query);
    const base_url = joined.base_url;
    path = joined.path;

    var headers: std.ArrayList(std.http.Header) = .empty;
    try setHeader(arena, &headers, "Accept", "application/json");
    try setHeader(arena, &headers, "Content-Type", "application/json");
    const organization = (try nullableString(kwargs, "organization")) orelse env(hosted.environment, "OPENAI_ORG_ID");
    const project = (try nullableString(kwargs, "project")) orelse env(hosted.environment, "OPENAI_PROJECT_ID");
    if (organization) |s| try setHeader(arena, &headers, "OpenAI-Organization", s);
    if (project) |s| try setHeader(arena, &headers, "OpenAI-Project", s);

    var api_key = try nullableString(kwargs, "api_key");
    var token: ?[]const u8 = null;
    var provider: ?TokenProvider = null;
    if (model.azure) {
        token = try nullableString(kwargs, "azure_ad_token");
        provider = try nativeTokenProvider(hosted, kwargs);
        if (@as(usize, @intFromBool(api_key != null)) + @intFromBool(token != null) + @intFromBool(provider != null) > 1) return error.ConflictingCredentials;
        if (api_key == null and token == null and provider == null) {
            token = env(hosted.environment, "AZURE_OPENAI_AD_TOKEN");
            if (token == null) api_key = env(hosted.environment, "AZURE_OPENAI_API_KEY");
        }
        if (token == null and provider == null and (api_key == null or api_key.?.len == 0)) return error.MissingCredentials;
    } else if (api_key == null or api_key.?.len == 0) return error.MissingCredentials;
    if (token) |s| {
        try setHeader(arena, &headers, "Authorization", try std.fmt.allocPrint(arena, "Bearer {s}", .{s}));
    } else if (api_key) |s| {
        if (model.azure) {
            try setHeader(arena, &headers, "api-key", s);
        } else {
            try setHeader(arena, &headers, "Authorization", try std.fmt.allocPrint(arena, "Bearer {s}", .{s}));
        }
    }
    if (env(hosted.environment, "OPENAI_CUSTOM_HEADERS")) |custom| {
        var lines = std.mem.splitScalar(u8, custom, '\n');
        while (lines.next()) |line| {
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                try setHeader(arena, &headers, std.mem.trim(u8, line[0..colon], " \t\r"), std.mem.trim(u8, line[colon + 1 ..], " \t\r"));
            }
        }
    }
    try mergeHeaders(arena, &headers, try objectOption(kwargs, "default_headers"));
    const extra_headers = try objectOption(kwargs, "extra_headers");
    try mergeHeaders(arena, &headers, extra_headers);
    // Azure injects credentials into request options before default headers are
    // merged. Thus extra headers win, but a default auth header cannot replace
    // the selected Azure credential. OpenAI defaults can override its bearer.
    if (model.azure) {
        if (token) |s| {
            if (!hasHeader(extra_headers, "Authorization")) try setHeader(arena, &headers, "Authorization", try std.fmt.allocPrint(arena, "Bearer {s}", .{s}));
        } else if (api_key) |s| {
            if (!hasHeader(extra_headers, "api-key")) try setHeader(arena, &headers, "api-key", s);
        }
    }

    var transport: lm.OpenAI = .{
        .client = hosted.client,
        .base_url = base_url,
        .model = model.name,
        .api_key = null,
        .max_retries = 2,
        .retry_base_ms = hosted.retry_base_ms,
        .max_response_bytes = hosted.max_response_bytes,
    };
    if (kwargs.get("max_retries")) |retries| {
        const count: i64 = switch (retries) {
            .integer => |n| n,
            .bool => |b| @intFromBool(b),
            else => return error.InvalidClientOption,
        };
        if (count < 0 or count > std.math.maxInt(@TypeOf(transport.max_retries))) return error.InvalidClientOption;
        transport.max_retries = @intCast(count);
    }
    try applyTimeout(&transport, kwargs.get("timeout"));
    if (provider) |p| {
        const context = try arena.create(TokenHeaders);
        context.* = .{ .provider = p, .explicit_authorization = hasHeader(extra_headers, "Authorization") };
        transport.header_provider = .{ .context = context, .prepare = TokenHeaders.prepare };
    } else try validateHeaders(headers.items);
    return .{ .transport = transport, .path = path, .body = .{ .object = body }, .headers = try headers.toOwnedSlice(arena) };
}

fn computeHosted(arena: Allocator, hosted: HostedOptions, kwargs: Record, inputs: []const []const u8) anyerror!Batch {
    var request = try prepareRequest(arena, hosted, kwargs, inputs);
    const response = try request.transport.postJson(arena, request.path, request.body, request.headers);
    return .{ .json = try responseEmbeddings(arena, response) };
}

fn applyTimeout(transport: *lm.OpenAI, option: ?Value) !void {
    transport.timeout_ns = 600 * std.time.ns_per_s;
    transport.connect_timeout_ns = 5 * std.time.ns_per_s;
    const item = option orelse return;
    transport.connect_timeout_ns = null;
    if (item == .null) {
        transport.timeout_ns = null;
        return;
    }
    const seconds: f64 = switch (item) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        .bool => |b| @floatFromInt(@intFromBool(b)),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch return error.InvalidClientOption,
        else => return error.InvalidClientOption,
    };
    if (!std.math.isFinite(seconds)) return error.InvalidClientOption;
    const nanoseconds = @ceil(@max(seconds, 0) * std.time.ns_per_s);
    if (nanoseconds >= 0x1p64) return error.InvalidClientOption;
    transport.timeout_ns = @intFromFloat(nanoseconds);
}

const TokenHeaders = struct {
    provider: TokenProvider,
    explicit_authorization: bool,

    fn prepare(ptr: *anyopaque, arena: Allocator, base: []const std.http.Header) anyerror![]const std.http.Header {
        const self: *const TokenHeaders = @ptrCast(@alignCast(ptr));
        // The SDK still calls/validates the provider when extra_headers selects
        // a different Authorization value; it just does not overwrite that value.
        const token = try self.provider.get(self.provider.context, arena);
        if (token.len == 0) return error.InvalidToken;
        if (self.explicit_authorization) {
            try validateHeaders(base);
            return base;
        }
        var headers: std.ArrayList(std.http.Header) = .empty;
        try headers.appendSlice(arena, base);
        try setHeader(arena, &headers, "Authorization", try std.fmt.allocPrint(arena, "Bearer {s}", .{token}));
        try validateHeaders(headers.items);
        return headers.toOwnedSlice(arena);
    }
};

fn hasHeader(headers: Record, name: []const u8) bool {
    for (headers.keys()) |key| if (std.ascii.eqlIgnoreCase(key, name)) return true;
    return false;
}

fn validHeaderName(name: []const u8) bool {
    if (name.len == 0) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "!#$%&'*+-.^_`|~", c) != null)) return false;
    return true;
}

fn validateHeaders(headers: []const std.http.Header) !void {
    // Validate only effective values: the SDK merges auth, organization,
    // defaults, and extra headers before the HTTP layer validates them.
    for (headers) |header| {
        if (!validHeaderName(header.name) or std.mem.indexOfAny(u8, header.value, "\r\n") != null) return error.InvalidHeader;
    }
}

fn setHeader(arena: Allocator, headers: *std.ArrayList(std.http.Header), name: []const u8, content: []const u8) !void {
    for (headers.items) |*header| if (std.ascii.eqlIgnoreCase(header.name, name)) {
        header.* = .{ .name = name, .value = content };
        return;
    };
    try headers.append(arena, .{ .name = name, .value = content });
}

fn mergeHeaders(arena: Allocator, headers: *std.ArrayList(std.http.Header), input: Record) !void {
    for (input.keys(), input.values()) |name, item| {
        if (item != .string) return error.InvalidHeader;
        try setHeader(arena, headers, name, item.string);
    }
}

fn escapeDeployment(arena: Allocator, name: []const u8) ![]const u8 {
    // SDK path_template rejects literal/encoded dot segments after quoting.
    if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return error.InvalidHostedModel;
    var writer: std.Io.Writer.Allocating = .init(arena);
    try std.Uri.Component.percentEncode(&writer.writer, name, struct {
        fn valid(c: u8) bool {
            return std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "-._~!$&'()*+,;=:@", c) != null;
        }
    }.valid);
    return writer.written();
}

const EndpointPath = struct { base_url: []const u8, path: []const u8 };

fn endpointPath(arena: Allocator, endpoint: []const u8, path: []const u8, query: Record) !EndpointPath {
    const uri = std.Uri.parse(endpoint) catch return error.InvalidEndpoint;
    if (uri.host == null or (!std.ascii.eqlIgnoreCase(uri.scheme, "http") and !std.ascii.eqlIgnoreCase(uri.scheme, "https"))) return error.InvalidEndpoint;
    const without_fragment = endpoint[0 .. std.mem.indexOfScalar(u8, endpoint, '#') orelse endpoint.len];
    if (uri.query) |base_query| {
        // SDK _prepare_url appends the relative raw path after an existing
        // query, not before it. Preserve this observable (unusual) URL rule.
        const raw = switch (base_query) {
            .raw, .percent_encoded => |s| s,
        };
        const combined = try std.fmt.allocPrint(arena, "{s}{s}", .{ raw, path });
        const base = without_fragment[0..std.mem.indexOfScalar(u8, without_fragment, '?').?];
        if (query.count() == 0) return .{
            .base_url = base,
            .path = try std.fmt.allocPrint(arena, "?{s}", .{combined}),
        };
        var merged: Record = .empty;
        var fields = std.mem.splitScalar(u8, combined, '&');
        while (fields.next()) |field| {
            if (field.len == 0) continue;
            const equal = std.mem.indexOfScalar(u8, field, '=') orelse field.len;
            const key = try queryUnescape(arena, field[0..equal]);
            const content = if (equal == field.len) "" else try queryUnescape(arena, field[equal + 1 ..]);
            if (!merged.contains(key)) try merged.put(arena, key, .{ .string = content });
        }
        try mergeRecord(arena, &merged, query);
        return .{ .base_url = base, .path = try pathWithQuery(arena, "", merged) };
    }
    return .{ .base_url = without_fragment, .path = try pathWithQuery(arena, path, query) };
}

fn queryUnescape(arena: Allocator, text: []const u8) ![]const u8 {
    const copy = try arena.dupe(u8, text);
    for (copy) |*c| if (c.* == '+') {
        c.* = ' ';
    };
    return std.Uri.percentDecodeBackwards(copy, copy);
}

fn pathWithQuery(arena: Allocator, path: []const u8, query: Record) ![]const u8 {
    var writer: std.Io.Writer.Allocating = .init(arena);
    try writer.writer.writeAll(path);
    var first = true;
    for (query.keys(), query.values()) |name, item| try writeQuery(arena, &writer.writer, name, item, &first, 0);
    return writer.written();
}

fn writeQuery(arena: Allocator, writer: *std.Io.Writer, name: []const u8, item: Value, first: *bool, depth: usize) anyerror!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    switch (item) {
        .null => {},
        .array => |array| {
            const key = try std.fmt.allocPrint(arena, "{s}[]", .{name});
            for (array.items) |child| try writeQuery(arena, writer, key, child, first, depth + 1);
        },
        .object => |object| for (object.keys(), object.values()) |key, child| {
            try writeQuery(arena, writer, try std.fmt.allocPrint(arena, "{s}[{s}]", .{ name, key }), child, first, depth + 1);
        },
        else => {
            try writer.writeByte(if (first.*) '?' else '&');
            first.* = false;
            try queryEscape(writer, name);
            try writer.writeByte('=');
            const text = switch (item) {
                .string, .number_string => |s| s,
                .bool => |b| if (b) "true" else "false",
                else => blk: {
                    var buffer: std.Io.Writer.Allocating = .init(arena);
                    try pyjson.writeStr(&buffer.writer, item);
                    break :blk buffer.written();
                },
            };
            try queryEscape(writer, text);
        },
    }
}

fn queryEscape(writer: *std.Io.Writer, text: []const u8) !void {
    for (text) |c| {
        if (std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "-._~", c) != null) {
            try writer.writeByte(c);
        } else if (c == ' ') {
            try writer.writeByte('+');
        } else try writer.print("%{X:0>2}", .{c});
    }
}

fn responseEmbeddings(arena: Allocator, response: Value) !Value {
    if (response != .object) return error.InvalidEmbeddingResponse;
    const data = response.object.get("data") orelse return error.InvalidEmbeddingResponse;
    if (data != .array) return error.InvalidEmbeddingResponse;
    for (data.array.items) |item| {
        if (item != .object) return error.InvalidEmbeddingResponse;
        const index = item.object.get("index") orelse return error.InvalidEmbeddingResponse;
        if (index != .integer or !item.object.contains("embedding")) return error.InvalidEmbeddingResponse;
    }
    // The response is private, arena-owned JSON. Stable in-place sorting avoids
    // another row array while preserving Python's order for duplicate indexes.
    std.mem.sort(Value, data.array.items, {}, struct {
        fn less(_: void, a: Value, b: Value) bool {
            return a.object.get("index").?.integer < b.object.get("index").?.integer;
        }
    }.less);
    var output: @import("runtime_value.zig").Array = .init(arena);
    try output.ensureTotalCapacity(data.array.items.len);
    for (data.array.items) |row| {
        var embedding = row.object.get("embedding").?;
        if (embedding == .string) {
            const length = try decodeBase64(embedding.string, null);
            if (length % @sizeOf(f32) != 0) return error.InvalidEmbeddingShape;
            var numbers: @import("runtime_value.zig").Array = .init(arena);
            try numbers.resize(length / @sizeOf(f32));
            _ = try decodeBase64Output(embedding.string, .{ .numbers = numbers.items });
            embedding = .{ .array = numbers };
        }
        if (embedding != .array) return error.InvalidEmbeddingResponse;
        output.appendAssumeCapacity(embedding);
    }
    return .{ .array = output };
}

/// CPython's default non-validating b64decode ignores nonalphabet ASCII and
/// leading/excess padding, but still rejects incomplete final quartets.
fn decodeBase64(text: []const u8, destination: ?[]u8) !usize {
    return decodeBase64Output(text, if (destination) |bytes| .{ .bytes = bytes } else .count);
}

const Base64Output = union(enum) { count, bytes: []u8, numbers: []Value };

fn decodeBase64Output(text: []const u8, destination: Base64Output) !usize {
    // Python validates that the complete input string is ASCII before decoding,
    // even if a padded quartet would otherwise terminate decoding early.
    for (text) |c| if (c > 127) return error.InvalidBase64;
    var chunk: [4]u8 = undefined;
    var position: u3 = 0;
    var previous: u8 = 0;
    var pads: u3 = 0;
    var length: usize = 0;
    for (text) |c| {
        if (c == '=') {
            if (position >= 2) {
                pads += 1;
                if (@as(u4, position) + pads >= 4) return length;
            }
            continue;
        }
        const digit: u8 = switch (c) {
            'A'...'Z' => c - 'A',
            'a'...'z' => c - 'a' + 26,
            '0'...'9' => c - '0' + 52,
            '+' => 62,
            '/' => 63,
            else => continue,
        };
        pads = 0;
        const byte: ?u8 = switch (position) {
            0 => null,
            1 => (previous << 2) | (digit >> 4),
            2 => (previous << 4) | (digit >> 2),
            3 => (previous << 6) | digit,
            else => unreachable,
        };
        if (byte) |b| {
            switch (destination) {
                .count => {},
                .bytes => |output| {
                    if (length >= output.len) return error.InvalidBase64;
                    output[length] = b;
                },
                .numbers => |output| {
                    chunk[length % 4] = b;
                    if (length % 4 == 3) {
                        if (length / 4 >= output.len) return error.InvalidBase64;
                        const number: f32 = @bitCast(std.mem.readInt(u32, &chunk, .little));
                        output[length / 4] = .{ .float = number };
                    }
                },
            }
            length += 1;
        }
        previous = digit;
        position = if (position == 3) 0 else position + 1;
    }
    if (position != 0) return error.InvalidBase64;
    return length;
}

const testing = std.testing;

fn testRecord(arena: Allocator, json: []const u8) !Record {
    const item = try std.json.parseFromSliceLeaky(Value, arena, json, .{});
    if (item != .object) return error.ExpectedObject;
    return item.object;
}

fn headerValue(headers: []const std.http.Header, name: []const u8) ?[]const u8 {
    for (headers) |header| if (std.ascii.eqlIgnoreCase(header.name, name)) return header.value;
    return null;
}

const TestCustom = struct {
    calls: usize = 0,
    sizes: [8]usize = @splat(0),
    fail_second_once: bool = false,
    mode: enum { numeric, ragged, wrong_rows, zero_columns, bad_values, mutate } = .numeric,

    fn run(ptr: *anyopaque, arena: Allocator, inputs: []const []const u8, kwargs: Record) anyerror!Batch {
        const self: *TestCustom = @ptrCast(@alignCast(ptr));
        if (self.calls < self.sizes.len) self.sizes[self.calls] = inputs.len;
        self.calls += 1;
        if (self.fail_second_once and std.mem.eql(u8, inputs[0], "second")) {
            self.fail_second_once = false;
            return error.LaterBatchFailed;
        }
        switch (self.mode) {
            .ragged => return .{ .json = try std.json.parseFromSliceLeaky(Value, arena, "[[1], [2, 3]]", .{}) },
            .wrong_rows => return .{ .float32 = .{ .values = &.{ 1, 2 }, .rows = 2, .columns = 1 } },
            .zero_columns => return .{ .float32 = .{ .values = &.{}, .rows = inputs.len, .columns = 0 } },
            .bad_values => return .{ .float32 = .{ .values = &.{1}, .rows = 1, .columns = 2 } },
            .mutate => {
                try testing.expectEqual(@as(i64, 1), kwargs.get("mutable").?.integer);
                var local = kwargs;
                try local.put(arena, "mutable", .{ .integer = 99 });
            },
            .numeric => {},
        }
        const numbers = try arena.alloc(f64, inputs.len * 2);
        for (inputs, 0..) |text, i| {
            numbers[i * 2] = @floatFromInt(text.len);
            numbers[i * 2 + 1] = 16_777_217;
        }
        return .{ .float64 = .{ .values = numbers, .rows = inputs.len, .columns = 2 } };
    }
};

test "Embedder batches 200 by default and postprocesses custom float64 to float32" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{};
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{ .caching = false });
    defer model.deinit();
    const inputs: [401][]const u8 = @splat("hello");
    const output = try model.embed(arena, &inputs, .{});
    try testing.expectEqual(@as(usize, 3), state.calls);
    try testing.expectEqualSlices(usize, &.{ 200, 200, 1 }, state.sizes[0..3]);
    try testing.expectEqual(@as(usize, 401), output.rows);
    try testing.expectEqual(@as(usize, 2), output.columns);
    try testing.expectEqualSlices(f32, &.{ 5, 16_777_216 }, output.row(400));
    const one = try model.embedOne(arena, "abc", .{});
    try testing.expectEqualSlices(f32, &.{ 3, 16_777_216 }, one);
}

test "Embedder call batch zero falls back while negative batches and empty inputs do not call" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{};
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{ .batch_size = 2, .caching = false });
    defer model.deinit();
    _ = try model.embed(arena, &.{ "a", "bb", "ccc" }, .{ .batch_size = 0 });
    try testing.expectEqualSlices(usize, &.{ 2, 1 }, state.sizes[0..2]);
    const empty = try model.embed(arena, &.{}, .{});
    try testing.expectEqual(@as(usize, 0), empty.rows);
    try testing.expectEqual(@as(usize, 0), empty.columns);
    try testing.expectEqual(@as(usize, 0), empty.values.len);
    const negative = try model.embed(arena, &.{"no call"}, .{ .batch_size = -1 });
    try testing.expectEqual(@as(usize, 0), negative.rows);
    try testing.expectError(error.EmptyEmbedding, model.embedOne(arena, "no call", .{ .batch_size = -1 }));
    try testing.expectEqual(@as(usize, 2), state.calls);
    model.options.batch_size = 0;
    try testing.expectError(error.InvalidBatchSize, model.embed(arena, &.{"a"}, .{}));
}

test "Embedder permits custom row count changes and zero columns but rejects invalid storage" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{ .mode = .wrong_rows };
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{ .caching = false });
    defer model.deinit();
    const different = try model.embed(arena, &.{"one input"}, .{});
    try testing.expectEqual(@as(usize, 2), different.rows);
    state.mode = .zero_columns;
    const zero = try model.embed(arena, &.{ "a", "b" }, .{});
    try testing.expectEqual(@as(usize, 2), zero.rows);
    try testing.expectEqual(@as(usize, 0), zero.columns);
    try testing.expectEqual(@as(usize, 0), zero.row(1).len);
    state.mode = .bad_values;
    try testing.expectError(error.InvalidEmbeddingShape, model.embed(arena, &.{"a"}, .{}));
    try testing.expectError(error.InvalidEmbeddingShape, matrixLength(std.math.maxInt(usize), 2));
    const json = try std.json.parseFromSliceLeaky(Value, arena, "[[1, true, null, \"2.5\"]]", .{});
    const numeric = try postprocess(arena, &.{.{ .json = json }});
    try testing.expectEqual(@as(f32, 1), numeric.values[0]);
    try testing.expectEqual(@as(f32, 1), numeric.values[1]);
    try testing.expect(std.math.isNan(numeric.values[2]));
    try testing.expectEqual(@as(f32, 2.5), numeric.values[3]);
    const bad = try std.json.parseFromSliceLeaky(Value, arena, "[[\"not numeric\"]]", .{});
    try testing.expectError(error.InvalidEmbeddingNumber, postprocess(arena, &.{.{ .json = bad }}));
}

test "Embedder cache merges kwargs canonically ignores named credentials and respects call caching" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{};
    const defaults = try testRecord(arena, "{\"nested\":{\"z\":2,\"a\":1},\"user\":\"default\"}");
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{ .kwargs = defaults });
    defer model.deinit();
    const first = try model.embed(arena, &.{"same"}, .{ .kwargs = try testRecord(arena, "{\"api_key\":\"one\",\"azure_ad_token\":\"one\",\"user\":\"call\"}") });
    const repeat = try model.embed(arena, &.{"same"}, .{ .kwargs = try testRecord(arena, "{\"nested\":{\"a\":1,\"z\":2},\"azure_ad_token\":\"two\",\"api_key\":\"two\",\"user\":\"call\"}") });
    try testing.expectEqualSlices(f32, first.values, repeat.values);
    try testing.expectEqual(@as(usize, 1), state.calls);
    _ = try model.embed(arena, &.{"same"}, .{ .kwargs = try testRecord(arena, "{\"user\":\"changed\"}") });
    _ = try model.embed(arena, &.{"same"}, .{ .caching = false, .kwargs = try testRecord(arena, "{\"user\":\"call\"}") });
    _ = try model.embed(arena, &.{"same"}, .{ .kwargs = try testRecord(arena, "{\"user\":\"call\"}") });
    try testing.expectEqual(@as(usize, 3), state.calls);
    model.options.caching = false;
    _ = try model.embed(arena, &.{"same"}, .{ .caching = true, .kwargs = try testRecord(arena, "{\"user\":\"call\"}") });
    try testing.expectEqual(@as(usize, 3), state.calls);
    try testing.expectEqual(@as(usize, 3), model.hits);
}

test "Embedder successful batches remain cached when a later batch fails" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{ .fail_second_once = true };
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{ .batch_size = 1 });
    defer model.deinit();
    try testing.expectError(error.LaterBatchFailed, model.embed(arena, &.{ "first", "second" }, .{}));
    const output = try model.embed(arena, &.{ "first", "second" }, .{});
    try testing.expectEqual(@as(usize, 3), state.calls);
    try testing.expectEqualSlices(f32, &.{ 5, 16_777_216, 6, 16_777_216 }, output.values);
    try testing.expectEqual(@as(usize, 1), model.hits);
}

test "Embedder caches successful raw batches before rectangular postprocessing errors" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{ .mode = .ragged };
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{});
    defer model.deinit();
    try testing.expectError(error.InvalidEmbeddingShape, model.embed(arena, &.{"ragged"}, .{}));
    try testing.expectError(error.InvalidEmbeddingShape, model.embed(arena, &.{"ragged"}, .{}));
    try testing.expectEqual(@as(usize, 1), state.calls);
}

test "Embedder cache owns results and retains current-call hits through LRU eviction" {
    var first_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    var state: TestCustom = .{};
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{ .batch_size = 1, .cache_capacity = 1 });
    defer model.deinit();
    _ = try model.embed(first_memory.allocator(), &.{"a"}, .{});
    first_memory.deinit();
    var second_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer second_memory.deinit();
    const output = try model.embed(second_memory.allocator(), &.{ "a", "bbb" }, .{});
    try testing.expectEqual(@as(usize, 2), state.calls);
    try testing.expectEqual(@as(usize, 1), model.entries.count());
    try testing.expectEqualSlices(f32, &.{ 1, 16_777_216, 3, 16_777_216 }, output.values);
}

test "Embedder callback top-level kwargs mutations do not poison another batch" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var state: TestCustom = .{ .mode = .mutate };
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = TestCustom.run }, .{
        .batch_size = 1,
        .kwargs = try testRecord(arena, "{\"mutable\":1}"),
    });
    defer model.deinit();
    _ = try model.embed(arena, &.{ "a", "bb" }, .{});
    _ = try model.embed(arena, &.{ "a", "bb" }, .{});
    try testing.expectEqual(@as(usize, 2), state.calls);
    try testing.expectEqual(@as(i64, 1), model.options.kwargs.get("mutable").?.integer);
}

test "Embedder snapshots nested mutable kwargs separately for each cached batch" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Mutable = struct {
        calls: usize = 0,

        fn run(ptr: *anyopaque, output: Allocator, inputs: []const []const u8, kwargs: Record) !Batch {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            const settings = kwargs.get("settings").?.object;
            const counter = settings.getPtr("n").?;
            counter.* = .{ .integer = counter.integer + 1 };
            const values = try output.alloc(f32, inputs.len);
            @memset(values, @as(f32, @floatFromInt(counter.integer)));
            return .{ .float32 = .{ .values = values, .rows = inputs.len, .columns = 1 } };
        }
    };
    var state: Mutable = .{};
    const settings = try testRecord(arena, "{\"n\":0}");
    var defaults: Record = .empty;
    try defaults.put(arena, "settings", .{ .object = settings });
    var model = Embedder.initCustom(testing.allocator, .{ .context = &state, .embed = Mutable.run }, .{
        .batch_size = 1,
        .kwargs = defaults,
    });
    defer model.deinit();
    const first = try model.embed(arena, &.{ "first", "second" }, .{});
    try testing.expectEqualSlices(f32, &.{ 1, 2 }, first.values);
    settings.getPtr("n").?.* = .{ .integer = 0 };
    const second = try model.embed(arena, &.{ "first", "second" }, .{});
    try testing.expectEqualSlices(f32, &.{ 1, 1 }, second.values);
    try testing.expectEqual(@as(usize, 3), state.calls);
    try testing.expectEqual(@as(i64, 1), settings.get("n").?.integer);
}

test "Embedder hosted model validation endpoint aliases and explicit environment precedence" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    for ([_][]const u8{ "cohere/model", "openai/", "azure/", "/name" }) |name| {
        try testing.expectError(error.InvalidHostedModel, Embedder.initHosted(testing.allocator, .{ .client = &client, .model = name }, .{}));
    }
    var model = try Embedder.initHosted(testing.allocator, .{
        .client = &client,
        .model = "text-embedding-3-small",
        .environment = &.{
            .{ .name = "OPENAI_BASE_URL", .value = "https://first.invalid/v1" },
            .{ .name = "OPENAI_API_BASE", .value = "https://second.invalid/v1" },
            .{ .name = "OPENAI_API_KEY", .value = "explicit-environment" },
        },
    }, .{ .kwargs = try testRecord(arena, "{\"api_base\":\"https://default.invalid/v1\",\"user\":\"default\"}") });
    defer model.deinit();
    const merged = try model.mergedKwargs(arena, try testRecord(arena, "{\"base_url\":\"https://call.invalid/v1\",\"user\":\"call\"}"));
    try testing.expectEqualStrings("https://call.invalid/v1", merged.get("base_url").?.string);
    try testing.expect(!merged.contains("api_base"));
    try testing.expectEqualStrings("call", merged.get("user").?.string);
    try testing.expectEqualStrings("explicit-environment", merged.get("api_key").?.string);
    try testing.expectEqualStrings("float", merged.get("encoding_format").?.string);
    try testing.expectError(error.ConflictingEndpoints, model.mergedKwargs(arena, try testRecord(arena, "{\"api_base\":\"https://a.invalid\",\"base_url\":\"https://b.invalid\"}")));
    const explicit_null = try model.mergedKwargs(arena, try testRecord(arena, "{\"encoding_format\":null}"));
    try testing.expect(explicit_null.get("encoding_format").? == .null);
    const replaced = try model.mergedKwargs(arena, try testRecord(arena, "{\"api_base\":null}"));
    try testing.expectEqualStrings("https://first.invalid/v1", replaced.get("base_url").?.string);
}

test "Embedder OpenAI request forwards fields merges client options and case-insensitive request headers" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var model = try Embedder.initHosted(testing.allocator, .{ .client = &client, .model = "openai/native-model" }, .{ .kwargs = try testRecord(arena,
        \\{"api_key":"dummy","base_url":"https://openai.invalid/v1","dimensions":2,"user":"default",
        \\ "organization":"org","project":"project","default_headers":{"Authorization":"Bearer default","X-Default":"one"},
        \\ "default_query":{"keep":"yes","replace":"default","drop":null},"max_retries":0}
    ) });
    defer model.deinit();
    const kwargs = try model.mergedKwargs(arena, try testRecord(arena,
        \\{"user":"call","encoding_format":"base64","extra_headers":{"authorization":"Bearer request","Content-Type":"application/custom+json"},
        \\ "extra_query":{"replace":"call","repeat":["x","y"],"nested":{"a":1},"flag":true},"extra_body":{"custom":{"flag":true},"dimensions":3}}
    ));
    const request = try prepareRequest(arena, model.backend.hosted, kwargs, &.{ "a", "b" });
    try testing.expectEqualStrings("https://openai.invalid/v1", request.transport.base_url);
    try testing.expectEqualStrings("embeddings?keep=yes&replace=call&repeat%5B%5D=x&repeat%5B%5D=y&nested%5Ba%5D=1&flag=true", request.path);
    try testing.expectEqualStrings("native-model", request.body.object.get("model").?.string);
    try testing.expectEqualStrings("a", request.body.object.get("input").?.array.items[0].string);
    try testing.expectEqualStrings("call", request.body.object.get("user").?.string);
    try testing.expectEqual(@as(i64, 3), request.body.object.get("dimensions").?.integer);
    try testing.expectEqualStrings("base64", request.body.object.get("encoding_format").?.string);
    try testing.expectEqualStrings("Bearer request", headerValue(request.headers, "Authorization").?);
    try testing.expectEqualStrings("application/custom+json", headerValue(request.headers, "content-type").?);
    try testing.expectEqualStrings("org", headerValue(request.headers, "openai-organization").?);
    try testing.expectEqualStrings("project", headerValue(request.headers, "OpenAI-Project").?);
    try testing.expect(request.transport.api_key == null);
    try testing.expectEqual(@as(@TypeOf(request.transport.max_retries), 0), request.transport.max_retries);
    try testing.expect(!request.body.object.contains("max_retries"));
    try testing.expect(!request.body.object.contains("extra_headers"));
    try testing.expectError(error.UnknownRequestOption, prepareRequest(arena, model.backend.hosted, try model.mergedKwargs(arena, try testRecord(arena, "{\"unknown\":true}")), &.{"x"}));
    try testing.expectError(error.DuplicateRequestArgument, prepareRequest(arena, model.backend.hosted, try model.mergedKwargs(arena, try testRecord(arena, "{\"input\":[\"other\"]}")), &.{"x"}));
}

test "Embedder validates final headers after SDK authorization and organization precedence" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var openai = try Embedder.initHosted(testing.allocator, .{ .client = &client, .model = "m" }, .{
        .kwargs = try testRecord(arena, "{\"api_key\":\"dummy\",\"organization\":\"bad\\norganization\",\"default_headers\":{\"Authorization\":\"bad\\nheader\"}}"),
    });
    defer openai.deinit();
    const kwargs = try openai.mergedKwargs(arena, try testRecord(arena, "{\"extra_headers\":{\"authorization\":\"Bearer final\",\"OpenAI-Organization\":\"final-org\"}}"));
    const request = try prepareRequest(arena, openai.backend.hosted, kwargs, &.{"x"});
    try testing.expectEqualStrings("Bearer final", headerValue(request.headers, "Authorization").?);
    try testing.expectEqualStrings("final-org", headerValue(request.headers, "OpenAI-Organization").?);
    const invalid = try openai.mergedKwargs(arena, try testRecord(arena, "{\"extra_headers\":{\"authorization\":\"bad\\nfinal\"}}"));
    try testing.expectError(error.InvalidHeader, prepareRequest(arena, openai.backend.hosted, invalid, &.{"x"}));

    var azure = try Embedder.initHosted(testing.allocator, .{ .client = &client, .model = "azure/m" }, .{
        .kwargs = try testRecord(arena, "{\"api_key\":\"bad\\nkey\",\"azure_endpoint\":\"https://azure.invalid\",\"api_version\":\"v\"}"),
    });
    defer azure.deinit();
    const azure_kwargs = try azure.mergedKwargs(arena, try testRecord(arena, "{\"extra_headers\":{\"api-key\":\"final-key\"}}"));
    const azure_request = try prepareRequest(arena, azure.backend.hosted, azure_kwargs, &.{"x"});
    try testing.expectEqualStrings("final-key", headerValue(azure_request.headers, "api-key").?);
}

test "Embedder Azure deployment version auth and alias precedence match hosted reference" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var model = try Embedder.initHosted(testing.allocator, .{
        .client = &client,
        .model = "azure/deploy/name",
        .environment = &.{
            .{ .name = "AZURE_OPENAI_ENDPOINT", .value = "https://azure.invalid" },
            .{ .name = "AZURE_API_KEY", .value = "fallback-key" },
            .{ .name = "OPENAI_API_VERSION", .value = "first-version" },
            .{ .name = "AZURE_API_VERSION", .value = "second-version" },
            .{ .name = "OPENAI_ORG_ID", .value = "env-org" },
        },
    }, .{ .kwargs = try testRecord(arena, "{\"default_query\":{\"api-version\":\"ignored\",\"keep\":1},\"default_headers\":{\"api-key\":\"default-key\"}}") });
    defer model.deinit();
    const kwargs = try model.mergedKwargs(arena, try testRecord(arena, "{\"extra_query\":{\"api-version\":\"call-version\"},\"extra_headers\":{\"api-key\":\"call-key\"}}"));
    const request = try prepareRequest(arena, model.backend.hosted, kwargs, &.{ "a", "b" });
    try testing.expectEqualStrings("https://azure.invalid/openai", request.transport.base_url);
    try testing.expectEqualStrings("deployments/deploy%2Fname/embeddings?api-version=call-version&keep=1", request.path);
    try testing.expectEqualStrings("deploy/name", request.body.object.get("model").?.string);
    try testing.expectEqualStrings("call-key", headerValue(request.headers, "api-key").?);
    try testing.expect(headerValue(request.headers, "Authorization") == null);
    try testing.expectEqualStrings("env-org", headerValue(request.headers, "OpenAI-Organization").?);
    const defaults_request = try prepareRequest(arena, model.backend.hosted, try model.mergedKwargs(arena, .empty), &.{"a"});
    try testing.expectEqualStrings("fallback-key", headerValue(defaults_request.headers, "api-key").?);
    const base = try model.mergedKwargs(arena, try testRecord(arena, "{\"base_url\":\"https://call.invalid/openai/deployments/fixed\"}"));
    try testing.expect(!base.contains("azure_endpoint"));
    const fixed = try prepareRequest(arena, model.backend.hosted, base, &.{"a"});
    try testing.expectEqualStrings("embeddings?api-version=first-version&keep=1", fixed.path);
    try testing.expectError(error.ConflictingEndpoints, model.mergedKwargs(arena, try testRecord(arena, "{\"api_base\":\"https://a.invalid\",\"azure_endpoint\":\"https://b.invalid\"}")));
    const conflicting = try model.mergedKwargs(arena, try testRecord(arena, "{\"base_url\":\"https://a.invalid\",\"azure_endpoint\":\"https://b.invalid\"}"));
    try testing.expectError(error.ConflictingEndpoints, prepareRequest(arena, model.backend.hosted, conflicting, &.{"a"}));
}

test "Embedder Azure native token provider and environment token preserve credential precedence" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    const Provider = struct {
        calls: usize = 0,
        fn get(ptr: *anyopaque, output: Allocator) ![]const u8 {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            return std.fmt.allocPrint(output, "native-provider-token-{d}", .{self.calls});
        }
    };
    var provider: Provider = .{};
    var model = try Embedder.initHosted(testing.allocator, .{
        .client = &client,
        .model = "azure/deploy",
        .environment = &.{
            .{ .name = "AZURE_OPENAI_ENDPOINT", .value = "https://azure.invalid" },
            .{ .name = "AZURE_OPENAI_API_KEY", .value = "environment-key" },
            .{ .name = "AZURE_OPENAI_AD_TOKEN", .value = "environment-token" },
            .{ .name = "AZURE_API_VERSION", .value = "v" },
        },
        .azure_ad_token_provider = .{ .context = &provider, .get = Provider.get },
    }, .{});
    defer model.deinit();
    const kwargs = try model.mergedKwargs(arena, .empty);
    try testing.expect(!kwargs.contains("api_key"));
    const request = try prepareRequest(arena, model.backend.hosted, kwargs, &.{"a"});
    try testing.expectEqual(@as(usize, 0), provider.calls);
    const hook = request.transport.header_provider.?;
    const first_attempt = try hook.prepare(hook.context, arena, request.headers);
    const retry = try hook.prepare(hook.context, arena, request.headers);
    try testing.expectEqualStrings("Bearer native-provider-token-1", headerValue(first_attempt, "Authorization").?);
    try testing.expectEqualStrings("Bearer native-provider-token-2", headerValue(retry, "Authorization").?);
    try testing.expectEqual(@as(usize, 2), provider.calls);
    const overridden = try model.mergedKwargs(arena, try testRecord(arena, "{\"extra_headers\":{\"authorization\":\"Bearer explicit\"}}"));
    const overridden_request = try prepareRequest(arena, model.backend.hosted, overridden, &.{"a"});
    const overridden_hook = overridden_request.transport.header_provider.?;
    const overridden_headers = try overridden_hook.prepare(overridden_hook.context, arena, overridden_request.headers);
    try testing.expectEqualStrings("Bearer explicit", headerValue(overridden_headers, "Authorization").?);
    try testing.expectEqual(@as(usize, 3), provider.calls);
    const disabled = try model.mergedKwargs(arena, try testRecord(arena, "{\"azure_ad_token_provider\":null}"));
    const environment_request = try prepareRequest(arena, model.backend.hosted, disabled, &.{"a"});
    try testing.expectEqualStrings("Bearer environment-token", headerValue(environment_request.headers, "Authorization").?);
    const explicit = try model.mergedKwargs(arena, try testRecord(arena, "{\"azure_ad_token_provider\":null,\"api_key\":\"explicit-key\"}"));
    const explicit_request = try prepareRequest(arena, model.backend.hosted, explicit, &.{"a"});
    try testing.expectEqualStrings("explicit-key", headerValue(explicit_request.headers, "api-key").?);
    try testing.expect(headerValue(explicit_request.headers, "Authorization") == null);
    const conflict = try model.mergedKwargs(arena, try testRecord(arena, "{\"api_key\":\"explicit-key\"}"));
    try testing.expectError(error.ConflictingCredentials, prepareRequest(arena, model.backend.hosted, conflict, &.{"a"}));
}

test "Embedder hosted timeout and endpoint-query semantics follow SDK native options" {
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var transport: lm.OpenAI = .{ .client = &client, .base_url = "https://example.invalid/v1", .model = "m" };
    try applyTimeout(&transport, null);
    try testing.expectEqual(@as(?u64, 600_000_000_000), transport.timeout_ns);
    try testing.expectEqual(@as(?u64, 5_000_000_000), transport.connect_timeout_ns);
    try applyTimeout(&transport, .{ .float = 0.125 });
    try testing.expectEqual(@as(?u64, 125_000_000), transport.timeout_ns);
    try testing.expect(transport.connect_timeout_ns == null);
    try applyTimeout(&transport, .null);
    try testing.expect(transport.timeout_ns == null and transport.connect_timeout_ns == null);
    try applyTimeout(&transport, .{ .bool = true });
    try testing.expectEqual(@as(?u64, 1_000_000_000), transport.timeout_ns);
    try applyTimeout(&transport, .{ .integer = -1 });
    try testing.expectEqual(@as(?u64, 0), transport.timeout_ns);
    try testing.expectError(error.InvalidClientOption, applyTimeout(&transport, .{ .object = .empty }));
    try testing.expectError(error.InvalidClientOption, applyTimeout(&transport, .{ .float = std.math.inf(f64) }));

    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const joined = try endpointPath(arena, "https://openai.invalid/v1?old=value", "embeddings", try testRecord(arena, "{\"extra\":1}"));
    try testing.expectEqualStrings("https://openai.invalid/v1", joined.base_url);
    try testing.expectEqualStrings("?old=valueembeddings&extra=1", joined.path);
}

test "Embedder hosted responses sort indexes stably and decode explicit little-endian base64" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const response = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"data":[{"index":1,"embedding":"AABAQAAAkEA="},{"index":0,"embedding":[1,2]},{"index":1,"embedding":[5,6]}]}
    , .{});
    const ordered = try responseEmbeddings(arena, response);
    const matrix = try postprocess(arena, &.{.{ .json = ordered }});
    try testing.expectEqual(@as(usize, 3), matrix.rows);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4.5, 5, 6 }, matrix.values);
    for ([_][]const u8{ "AACAPw==", "AA CA\nPw==", "?AACAPw==!", "====AACAPw======" }) |text| {
        var bytes: [4]u8 = undefined;
        try testing.expectEqual(@as(usize, 4), try decodeBase64(text, &bytes));
        try testing.expectEqual(@as(f32, 1), @as(f32, @bitCast(std.mem.readInt(u32, &bytes, .little))));
    }
    for ([_][]const u8{ "AACAPw", "AA=A", "A", "AACAPw==é" }) |text| try testing.expectError(error.InvalidBase64, decodeBase64(text, null));
    const invalid = try std.json.parseFromSliceLeaky(Value, arena, "{\"data\":[{\"index\":0,\"embedding\":\"AA==\"}]}", .{});
    try testing.expectError(error.InvalidEmbeddingShape, responseEmbeddings(arena, invalid));
    const missing = try std.json.parseFromSliceLeaky(Value, arena, "{\"data\":[{\"embedding\":[1]}]}", .{});
    try testing.expectError(error.InvalidEmbeddingResponse, responseEmbeddings(arena, missing));
    const empty = try std.json.parseFromSliceLeaky(Value, arena, "{\"data\":[]}", .{});
    const empty_result = try postprocess(arena, &.{.{ .json = try responseEmbeddings(arena, empty) }});
    try testing.expectEqual(@as(usize, 0), empty_result.rows);
}
