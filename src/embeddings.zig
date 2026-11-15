//! Native DSPy 3.5 `Embeddings` and `EmbeddingsWithScores` brute-force retrieval.
//!
//! The Embedder and constructor corpus strings/slice are borrowed. The retriever
//! owns its corpus vectors and loaded configuration until `deinit`; do not copy
//! an initialized retriever. Public k/normalize/corpus/corpus_embeddings fields
//! remain live. Changing normalize only affects subsequent query embeddings.
//! Manually replaced corpus slices/matrices are borrowed; update rank/layout/
//! byte-order metadata too when replacing a loaded array with a different kind.
//! Result slices belong to the caller's output arena; passage strings borrow
//! the current corpus and must not outlive it (including a subsequent load).
//! Calls require external synchronization; batchForward explicitly embeds the
//! whole query list, without Python Unbatchify's background thread/scheduler.
//!
//! Constructor/query vectors use the existing Embedder's contiguous float32
//! boundary. Arbitrary Python callables retaining float64/object/complex arrays
//! are not this typed API. Norms use NumPy's float32 pairwise square reduction;
//! scores use its four-lane NEON einsum kernel, including explicit fused products.
//! Public scores widen those f32 results to f64, matching NumPy tolist()'s Python
//! floats and their decimal-threshold comparisons, without recomputing in f64.
//! Sorting reuses NumPy's scalar unstable argsort, NOT KNN's reversed ordering.
//! Other NumPy CPU-dispatched sort/einsum kernels need not be bitwise identical.
//!
//! save/load/fromSaved take explicit std.Io and DSPy's directory layout. NPY
//! versions 1-3 support canonical NumPy float32 headers, either byte order,
//! C/Fortran storage, and scalar/vector/matrix shapes. Non-matrix arrays can
//! load/save but retrieval rejects them, as einsum does. Higher ranks and other
//! dtypes fail explicitly.
//! Headers are parsed as data, never evaluated; source's 10,000-byte header
//! bound is retained. Unsupported FAISS construction or indexed saves return
//! UnsupportedFaiss, never silently substitute an approximate or brute index.
//!
//! Upstream (DSPy 3.5.0): dspy/retrievers/embeddings.py

const std = @import("std");
const builtin = @import("builtin");
const embedding = @import("embedder.zig");
const ranking = @import("knn.zig");
const Allocator = std.mem.Allocator;
const Matrix = embedding.Matrix;
const native_endian = builtin.cpu.arch.endian();

pub const Options = struct {
    k: isize = 5,
    cache: bool = false,
    brute_force_threshold: isize = 20_000,
    normalize: bool = true,
};

pub const Prediction = struct {
    passages: []const []const u8,
    indices: []const usize,
};

pub const ScoredPrediction = struct {
    passages: []const []const u8,
    indices: []const usize,
    scores: []const f64,
};

/// DSPy's _batch_forward returns scores for both retriever classes.
pub const BatchPrediction = ScoredPrediction;
pub const ArrayRank = enum { scalar, vector, matrix };

pub const Embeddings = Retriever(false);
pub const EmbeddingsWithScores = Retriever(true);

fn Retriever(comptime with_scores: bool) type {
    return struct {
        const Self = @This();
        pub const Output = if (with_scores) ScoredPrediction else Prediction;

        allocator: Allocator,
        configuration_memory: std.heap.ArenaAllocator,
        vector_memory: std.heap.ArenaAllocator,
        embedder: *embedding.Embedder,
        k: isize,
        normalize: bool,
        corpus: []const []const u8,
        corpus_embeddings: Matrix,
        /// Embedder represents Python's empty rank-one array as rows=0.
        corpus_rank: ArrayRank = .matrix,
        corpus_byte_order: std.builtin.Endian = native_endian,
        corpus_fortran_order: bool = false,

        pub fn init(allocator: Allocator, corpus: []const []const u8, embedder: *embedding.Embedder, options: Options) anyerror!Self {
            if (options.cache) return error.UnsupportedCache;
            var vectors: std.heap.ArenaAllocator = .init(allocator);
            errdefer vectors.deinit();
            const matrix = try embedder.embed(vectors.allocator(), corpus, .{});
            try validateMatrix(matrix);
            if (options.normalize) {
                if (matrix.rows == 0) return error.InvalidEmbeddingShape;
                normalizeRows(matrix);
            }
            if (options.brute_force_threshold <= 0 or corpus.len >= @as(usize, @intCast(options.brute_force_threshold))) {
                if (matrix.rows == 0) return error.InvalidEmbeddingShape;
                return error.UnsupportedFaiss;
            }
            return .{
                .allocator = allocator,
                .configuration_memory = .init(allocator),
                .vector_memory = vectors,
                .embedder = embedder,
                .k = options.k,
                .normalize = options.normalize,
                .corpus = corpus,
                .corpus_embeddings = matrix,
                .corpus_rank = if (matrix.rows == 0) .vector else .matrix,
            };
        }

        pub fn deinit(self: *Self) void {
            self.configuration_memory.deinit();
            self.vector_memory.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, out: Allocator, query: []const u8) anyerror!Output {
            const results = try self.batchForward(out, &.{query});
            if (results.len == 0) return error.MissingQueryEmbedding;
            const result = results[0];
            return if (with_scores) result else .{ .passages = result.passages, .indices = result.indices };
        }

        /// A malformed backend may emit a different number of query/corpus rows
        /// than inputs. Retain that behavior; only selected corpus indices must
        /// be in bounds. Zero/negative k uses [:k], not KNN's [-k:][::-1].
        pub fn batchForward(self: *Self, out: Allocator, queries: []const []const u8) anyerror![]const BatchPrediction {
            var memory: std.heap.ArenaAllocator = .init(self.allocator);
            defer memory.deinit();
            const scratch = memory.allocator();
            const query = try self.embedder.embed(scratch, queries, .{});
            try validateMatrix(query);
            if (query.rows == 0) return error.InvalidEmbeddingShape;
            if (self.normalize) normalizeRows(query);
            // DSPy assigns ascontiguousarray before einsum, even when a later
            // dimension check fails. Native matrices are already contiguous.
            self.corpus_fortran_order = false;
            if (self.corpus_rank != .matrix) return error.InvalidEmbeddingShape;
            const corpus = self.corpus_embeddings;
            try validateMatrix(corpus);
            if (query.columns != corpus.columns and query.columns != 1 and corpus.columns != 1)
                return error.EmbeddingDimensionMismatch;
            const count = selectedCount(self.k, corpus.rows);
            const results = try out.alloc(BatchPrediction, query.rows);
            const negative_scores = try scratch.alloc(f32, if (count == 0) 0 else corpus.rows);
            const order = try scratch.alloc(usize, negative_scores.len);
            for (results, 0..) |*result, q| {
                for (negative_scores, 0..) |*score, c| score.* = -einsum(query.row(q), corpus.row(c));
                ranking.argsort(negative_scores, order);
                const passages = try out.alloc([]const u8, count);
                const indices = try out.alloc(usize, count);
                const scores = try out.alloc(f64, count);
                for (passages, indices, scores, order[0..count]) |*passage, *index, *score, selected| {
                    if (selected >= self.corpus.len) return error.CorpusIndexOutOfBounds;
                    passage.* = self.corpus[selected];
                    index.* = selected;
                    score.* = @as(f32, -negative_scores[selected]);
                }
                result.* = .{ .passages = passages, .indices = indices, .scores = scores };
            }
            return results;
        }

        pub fn save(self: *const Self, io: std.Io, path: []const u8) anyerror!void {
            const directory = try std.Io.Dir.cwd().createDirPathOpen(io, path, .{});
            defer directory.close(io);
            try validateMatrix(self.corpus_embeddings);
            const config = try directory.createFile(io, "config.json", .{});
            defer config.close(io);
            var buffer: [4096]u8 = undefined;
            var writer = config.writer(io, &buffer);
            try std.json.Stringify.value(.{
                .k = self.k,
                .normalize = self.normalize,
                .corpus = self.corpus,
                .has_faiss_index = false,
            }, .{}, &writer.interface);
            try writer.interface.flush();
            const array = try directory.createFile(io, "corpus_embeddings.npy", .{});
            defer array.close(io);
            try writeNpy(array, io, self.corpus_embeddings, self.corpus_rank, self.corpus_byte_order, self.corpus_fortran_order);
        }

        /// No embedding calls. Required-file/config failures leave the instance
        /// unchanged; after valid config, later array/index failures preserve
        /// DSPy's observable new k/normalize/corpus/embedder configuration.
        pub fn load(self: *Self, io: std.Io, path: []const u8, embedder: *embedding.Embedder) anyerror!*Self {
            const directory = try std.Io.Dir.cwd().openDir(io, path, .{});
            defer directory.close(io);
            try directory.access(io, "config.json", .{});
            try directory.access(io, "corpus_embeddings.npy", .{});
            var configuration: std.heap.ArenaAllocator = .init(self.allocator);
            defer configuration.deinit();
            const config_bytes = try directory.readFileAlloc(io, "config.json", configuration.allocator(), .unlimited);
            const config = try parseConfig(configuration.allocator(), config_bytes);
            std.mem.swap(std.heap.ArenaAllocator, &self.configuration_memory, &configuration);
            self.k = config.k;
            self.normalize = config.normalize;
            self.corpus = config.corpus;
            self.embedder = embedder;

            var vectors: std.heap.ArenaAllocator = .init(self.allocator);
            defer vectors.deinit();
            const bytes = try directory.readFileAllocOptions(io, "corpus_embeddings.npy", vectors.allocator(), .unlimited, .of(f32), null);
            const array = try readNpy(vectors.allocator(), bytes);
            std.mem.swap(std.heap.ArenaAllocator, &self.vector_memory, &vectors);
            self.corpus_embeddings = array.matrix;
            self.corpus_rank = array.rank;
            self.corpus_byte_order = array.byte_order;
            self.corpus_fortran_order = array.fortran_order;
            if (config.has_faiss_index) return error.UnsupportedFaiss;
            return self;
        }

        /// Unlike init followed by load, this never embeds an unnecessary corpus.
        pub fn fromSaved(allocator: Allocator, io: std.Io, path: []const u8, embedder: *embedding.Embedder) anyerror!Self {
            var self: Self = .{
                .allocator = allocator,
                .configuration_memory = .init(allocator),
                .vector_memory = .init(allocator),
                .embedder = embedder,
                .k = 5,
                .normalize = true,
                .corpus = &.{},
                .corpus_embeddings = .{ .values = &.{}, .rows = 0, .columns = 0 },
            };
            errdefer self.deinit();
            _ = try self.load(io, path, embedder);
            return self;
        }
    };
}

fn validateMatrix(matrix: Matrix) error{InvalidEmbeddingShape}!void {
    const count = std.math.mul(usize, matrix.rows, matrix.columns) catch return error.InvalidEmbeddingShape;
    if (count != matrix.values.len) return error.InvalidEmbeddingShape;
}

fn selectedCount(k: isize, count: usize) usize {
    if (k >= 0) return @min(@as(usize, @intCast(k)), count);
    return count - @min(@abs(k), count);
}

fn normalizeRows(matrix: Matrix) void {
    @setFloatMode(.strict);
    for (0..matrix.rows) |row_index| {
        const row = @constCast(matrix.row(row_index));
        const norm = @sqrt(pairwiseSquares(row));
        // @max has different NaN handling from np.maximum; preserve NaNs.
        const denominator: f32 = if (norm < 1e-10) 1e-10 else norm;
        for (row) |*component| component.* /= denominator;
    }
}

// The numerical reductions below translate NumPy v2.4.6's loops_utils.h.src
// and einsum_sumprod.c.src / NEON arithmetic.h. Copyright (c) 2005-2025,
// NumPy Developers; einsum Copyright (c) 2011 Mark Wiebe and The University
// of British Columbia. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the NumPy Developers nor the names of any contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

fn pairwiseSquares(row: []const f32) f32 {
    @setFloatMode(.strict);
    if (row.len < 8) {
        var sum: f32 = -0.0;
        for (row) |x| sum += x * x;
        return sum;
    }
    if (row.len <= 128) {
        var lanes: [8]f32 = undefined;
        for (&lanes, row[0..8]) |*lane, x| lane.* = x * x;
        var i: usize = 8;
        while (i < row.len - row.len % 8) : (i += 8) {
            for (&lanes, row[i..][0..8]) |*lane, x| lane.* += x * x;
        }
        var sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
            ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while (i < row.len) : (i += 1) sum += row[i] * row[i];
        return sum;
    }
    const middle = (row.len / 2) - (row.len / 2) % 8;
    return pairwiseSquares(row[0..middle]) + pairwiseSquares(row[middle..]);
}

fn einsum(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.strict);
    std.debug.assert(a.len == b.len or a.len == 1 or b.len == 1);
    if (a.len == 0 or b.len == 0) return 0;
    if (a.len != b.len) {
        const product: f32 = if (a.len == 1) a[0] * sumArray(b) else sumArray(a) * b[0];
        return 0.0 + product;
    }
    var lanes: [4]f32 = @splat(0);
    var i: usize = 0;
    while (i + 16 <= a.len) : (i += 16) {
        for (&lanes, 0..) |*lane, j| {
            lane.* = @mulAdd(f32, a[i + j + 12], b[i + j + 12], lane.*);
            lane.* = @mulAdd(f32, a[i + j + 8], b[i + j + 8], lane.*);
            lane.* = @mulAdd(f32, a[i + j + 4], b[i + j + 4], lane.*);
            lane.* = @mulAdd(f32, a[i + j], b[i + j], lane.*);
        }
    }
    while (i < a.len) : (i += 4) {
        for (&lanes, 0..) |*lane, j| {
            const x: f32 = if (i + j < a.len) a[i + j] else 0;
            const y: f32 = if (i + j < b.len) b[i + j] else 0;
            lane.* = @mulAdd(f32, x, y, lane.*);
        }
    }
    return 0.0 + ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
}

fn sumArray(row: []const f32) f32 {
    @setFloatMode(.strict);
    var lanes: [4]f32 = @splat(0);
    var i: usize = 0;
    while (i + 16 <= row.len) : (i += 16) {
        for (&lanes, 0..) |*lane, j| {
            const a01 = row[i + j] + row[i + j + 4];
            const a23 = row[i + j + 8] + row[i + j + 12];
            lane.* = (a01 + a23) + lane.*;
        }
    }
    while (i < row.len) : (i += 4) {
        for (&lanes, 0..) |*lane, j| lane.* = (if (i + j < row.len) row[i + j] else @as(f32, 0)) + lane.*;
    }
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}

const Config = struct {
    k: isize,
    normalize: bool,
    corpus: []const []const u8,
    has_faiss_index: bool,
};

fn parseConfig(arena: Allocator, bytes: []const u8) !Config {
    const config = try std.json.parseFromSliceLeaky(@import("runtime_value.zig").Value, arena, bytes, .{ .duplicate_field_behavior = .use_last });
    if (config != .object) return error.InvalidConfig;
    const k = config.object.get("k") orelse return error.MissingConfigField;
    const normalize = config.object.get("normalize") orelse return error.MissingConfigField;
    const corpus = config.object.get("corpus") orelse return error.MissingConfigField;
    const has_index = config.object.get("has_faiss_index") orelse return error.MissingConfigField;
    if (k != .integer or normalize != .bool or corpus != .array or has_index != .bool) return error.InvalidConfig;
    const count = std.math.cast(isize, k.integer) orelse return error.InvalidConfig;
    const passages = try arena.alloc([]const u8, corpus.array.items.len);
    for (passages, corpus.array.items) |*passage, item| {
        if (item != .string) return error.InvalidConfig;
        passage.* = item.string;
    }
    return .{ .k = count, .normalize = normalize.bool, .corpus = passages, .has_faiss_index = has_index.bool };
}

const Npy = struct {
    matrix: Matrix,
    rank: ArrayRank,
    byte_order: std.builtin.Endian,
    fortran_order: bool,
};

const Header = struct {
    descr: []const u8,
    fortran_order: bool,
    dimensions: [2]usize,
    rank: ArrayRank,
};

const HeaderParser = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn whitespace(self: *HeaderParser) void {
        while (self.pos < self.bytes.len and std.ascii.isWhitespace(self.bytes[self.pos])) self.pos += 1;
    }

    fn consume(self: *HeaderParser, byte: u8) bool {
        self.whitespace();
        if (self.pos == self.bytes.len or self.bytes[self.pos] != byte) return false;
        self.pos += 1;
        return true;
    }

    fn expect(self: *HeaderParser, byte: u8) !void {
        if (!self.consume(byte)) return error.InvalidNpyHeader;
    }

    fn string(self: *HeaderParser) ![]const u8 {
        self.whitespace();
        if (self.pos == self.bytes.len) return error.InvalidNpyHeader;
        const quote = self.bytes[self.pos];
        if (quote != '\'' and quote != '"') return error.InvalidNpyHeader;
        self.pos += 1;
        const start = self.pos;
        while (self.pos < self.bytes.len and self.bytes[self.pos] != quote) : (self.pos += 1) {
            if (self.bytes[self.pos] == '\\' or self.bytes[self.pos] < 0x20) return error.InvalidNpyHeader;
        }
        if (self.pos == self.bytes.len) return error.InvalidNpyHeader;
        const result = self.bytes[start..self.pos];
        self.pos += 1;
        return result;
    }

    fn boolean(self: *HeaderParser) !bool {
        self.whitespace();
        for ([_][]const u8{ "True", "False" }, [_]bool{ true, false }) |text, result| {
            if (std.mem.startsWith(u8, self.bytes[self.pos..], text)) {
                self.pos += text.len;
                return result;
            }
        }
        return error.InvalidNpyHeader;
    }

    fn shape(self: *HeaderParser, dimensions: *[2]usize) !ArrayRank {
        try self.expect('(');
        if (self.consume(')')) return .scalar;
        var rank: usize = 0;
        while (true) {
            if (rank == dimensions.len) return error.UnsupportedNpyShape;
            self.whitespace();
            const start = self.pos;
            while (self.pos < self.bytes.len and std.ascii.isDigit(self.bytes[self.pos])) self.pos += 1;
            if (self.pos == start) return error.InvalidNpyHeader;
            dimensions[rank] = std.fmt.parseInt(usize, self.bytes[start..self.pos], 10) catch return error.InvalidNpyShape;
            rank += 1;
            if (self.consume(')')) {
                if (rank == 1) return error.InvalidNpyHeader;
                break;
            }
            try self.expect(',');
            if (self.consume(')')) break;
        }
        return if (rank == 1) .vector else .matrix;
    }

    fn parse(self: *HeaderParser) !Header {
        var descr: ?[]const u8 = null;
        var fortran_order: ?bool = null;
        var rank: ?ArrayRank = null;
        var dimensions: [2]usize = .{ 1, 1 };
        try self.expect('{');
        while (!self.consume('}')) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "descr")) {
                descr = try self.string();
            } else if (std.mem.eql(u8, key, "fortran_order")) {
                fortran_order = try self.boolean();
            } else if (std.mem.eql(u8, key, "shape")) {
                rank = try self.shape(&dimensions);
            } else return error.InvalidNpyHeader;
            if (self.consume('}')) break;
            try self.expect(',');
        }
        self.whitespace();
        if (self.pos != self.bytes.len) return error.InvalidNpyHeader;
        return .{
            .descr = descr orelse return error.InvalidNpyHeader,
            .fortran_order = fortran_order orelse return error.InvalidNpyHeader,
            .dimensions = dimensions,
            .rank = rank orelse return error.InvalidNpyHeader,
        };
    }
};

fn readNpy(arena: Allocator, bytes: []align(@alignOf(f32)) u8) !Npy {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.InvalidNpyHeader;
    const major = bytes[6];
    if (bytes[7] != 0 or major < 1 or major > 3) return error.UnsupportedNpyVersion;
    const prefix: usize = if (major == 1) 10 else 12;
    if (bytes.len < prefix) return error.InvalidNpyHeader;
    const header_length: usize = if (major == 1) std.mem.readInt(u16, bytes[8..10], .little) else std.mem.readInt(u32, bytes[8..12], .little);
    if (header_length > 10_000) return error.NpyHeaderTooLarge;
    if (header_length > bytes.len - prefix) return error.InvalidNpyHeader;
    var parser: HeaderParser = .{ .bytes = bytes[prefix..][0..header_length] };
    const header = try parser.parse();
    var byte_order = native_endian;
    if (std.mem.eql(u8, header.descr, "<f4")) {
        byte_order = .little;
    } else if (std.mem.eql(u8, header.descr, ">f4")) {
        byte_order = .big;
    } else if (!std.mem.eql(u8, header.descr, "=f4") and !std.mem.eql(u8, header.descr, "|f4") and !std.mem.eql(u8, header.descr, "f4") and !std.mem.eql(u8, header.descr, "float32")) return error.UnsupportedNpyDtype;
    const rows = header.dimensions[0];
    const columns = header.dimensions[1];
    const count = std.math.mul(usize, rows, columns) catch return error.InvalidNpyShape;
    const length = std.math.mul(usize, count, @sizeOf(f32)) catch return error.InvalidNpyShape;
    const payload = bytes[prefix + header_length ..];
    if (payload.len < length) return error.TruncatedNpyData;
    const aligned = @intFromPtr(payload.ptr) % @alignOf(f32) == 0;
    const transpose = header.rank == .matrix and header.fortran_order and rows > 1 and columns > 1;
    const values = if (aligned and !transpose) blk: {
        const output: []f32 = std.mem.bytesAsSlice(f32, @as([]align(@alignOf(f32)) u8, @alignCast(payload[0..length])));
        if (byte_order != native_endian) {
            for (output) |*component| component.* = @bitCast(@byteSwap(@as(u32, @bitCast(component.*))));
        }
        break :blk output;
    } else blk: {
        const output = try arena.alloc(f32, count);
        for (output, 0..) |*component, i| {
            const source_index = if (transpose) (i % columns) * rows + i / columns else i;
            component.* = @bitCast(std.mem.readInt(u32, payload[source_index * 4 ..][0..4], byte_order));
        }
        break :blk output;
    };
    return .{
        .matrix = .{ .values = values, .rows = rows, .columns = columns },
        .rank = header.rank,
        .byte_order = byte_order,
        .fortran_order = transpose,
    };
}

fn writeNpy(file: std.Io.File, io: std.Io, matrix: Matrix, rank: ArrayRank, byte_order: std.builtin.Endian, fortran_order: bool) !void {
    if ((rank == .scalar and matrix.values.len != 1) or
        (rank == .vector and matrix.values.len != matrix.rows)) return error.InvalidEmbeddingShape;
    var shape_buffer: [64]u8 = undefined;
    const shape = switch (rank) {
        .scalar => "()",
        .vector => try std.fmt.bufPrint(&shape_buffer, "({d},)", .{matrix.rows}),
        .matrix => try std.fmt.bufPrint(&shape_buffer, "({d}, {d})", .{ matrix.rows, matrix.columns }),
    };
    // NumPy writes fortran_order only when the array is F- but not C-contiguous,
    // i.e. a matrix with more than one row and column.
    const transpose = rank == .matrix and fortran_order and matrix.rows > 1 and matrix.columns > 1;
    var header_buffer: [256]u8 = undefined;
    const dictionary = try std.fmt.bufPrint(&header_buffer, "{{'descr': '{c}f4', 'fortran_order': {s}, 'shape': {s}, }}", .{
        @as(u8, if (byte_order == .little) '<' else '>'),
        if (transpose) "True" else "False",
        shape,
    });
    const header_length = std.mem.alignForward(usize, 10 + dictionary.len + 1, 64) - 10;
    @memset(header_buffer[dictionary.len .. header_length - 1], ' ');
    header_buffer[header_length - 1] = '\n';
    var prefix = [_]u8{ 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0 };
    std.mem.writeInt(u16, prefix[8..10], @intCast(header_length), .little);
    try file.writeStreamingAll(io, &prefix);
    try file.writeStreamingAll(io, header_buffer[0..header_length]);
    if (!transpose and byte_order == native_endian) {
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(matrix.values));
        return;
    }
    var buffer: [4096]u8 = undefined;
    var start: usize = 0;
    while (start < matrix.values.len) {
        const count = @min(buffer.len / 4, matrix.values.len - start);
        for (0..count) |j| {
            const i = start + j;
            const source = if (transpose) (i % matrix.rows) * matrix.columns + i / matrix.rows else i;
            std.mem.writeInt(u32, buffer[j * 4 ..][0..4], @bitCast(matrix.values[source]), byte_order);
        }
        try file.writeStreamingAll(io, buffer[0 .. count * 4]);
        start += count;
    }
}

const testing = std.testing;
const Record = @import("value.zig").Record;

const TestBackend = struct {
    arena: Allocator,
    corpus: Matrix,
    query: Matrix,
    first_is_corpus: bool = true,
    calls: std.ArrayList([]const []const u8) = .empty,

    fn embed(context: *anyopaque, _: Allocator, texts: []const []const u8, _: Record) anyerror!embedding.Batch {
        const self: *@This() = @ptrCast(@alignCast(context));
        const first = self.first_is_corpus and self.calls.items.len == 0;
        const saved = try self.arena.alloc([]const u8, texts.len);
        for (saved, texts) |*out, text| out.* = try self.arena.dupe(u8, text);
        try self.calls.append(self.arena, saved);
        return .{ .float32 = if (first) self.corpus else self.query };
    }

    fn model(self: *TestBackend) embedding.Embedder {
        return embedding.Embedder.initCustom(testing.allocator, .{ .context = self, .embed = embed }, .{ .caching = false });
    }
};

fn expectBits(expected: []const u32, actual: []const f32) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |bits, number| try testing.expectEqual(bits, @as(u32, @bitCast(number)));
}

fn expectScoreBits(expected: []const u32, actual: []const f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |bits, number| {
        const widened: f64 = @as(f32, @bitCast(bits));
        try testing.expectEqual(@as(u64, @bitCast(widened)), @as(u64, @bitCast(number)));
    }
}

fn fromBits(comptime bits: anytype) [bits.len]f32 {
    var numbers: [bits.len]f32 = undefined;
    inline for (bits, 0..) |word, i| numbers[i] = @bitCast(@as(u32, word));
    return numbers;
}

test "Embeddings embeds its corpus once and batches query text together" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 3, 0, 0, 4, 0, 0, -3, 0 }, .rows = 4, .columns = 2 },
        .query = .{ .values = &.{ 3, 4, 0, 0 }, .rows = 2, .columns = 2 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try Embeddings.init(testing.allocator, &.{ "east", "north", "zero", "west" }, &model, .{});
    defer retriever.deinit();
    const first = try retriever.batchForward(arena, &.{ "diagonal", "origin" });
    try testing.expectEqualSlices(usize, &.{ 1, 0, 2, 3 }, first[0].indices);
    try expectScoreBits(&.{ 1061997773, 1058642330, 0, 3206125978 }, first[0].scores);
    try testing.expectEqualStrings("north", first[0].passages[0]);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, first[1].indices);
    backend.query = .{ .values = &.{ 3, 4 }, .rows = 1, .columns = 2 };
    const single = try retriever.forward(arena, "again");
    try testing.expectEqualSlices(usize, &.{ 1, 0, 2, 3 }, single.indices);
    try testing.expectEqual(@as(usize, 3), backend.calls.items.len);
    try testing.expectEqual(@as(usize, 4), backend.calls.items[0].len);
    try testing.expectEqual(@as(usize, 2), backend.calls.items[1].len);
    try testing.expectEqualStrings("diagonal", backend.calls.items[1][0]);
    try testing.expectEqualStrings("origin", backend.calls.items[1][1]);
    try testing.expectEqualStrings("again", backend.calls.items[2][0]);
}

test "EmbeddingsWithScores observes runtime k normalization corpus and vector changes" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 3, 0, 0, 4 }, .rows = 2, .columns = 2 },
        .query = .{ .values = &.{ 3, 4 }, .rows = 1, .columns = 2 },
    };
    var corpus = [_][]const u8{ "east", "north" };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &corpus, &model, .{ .k = 1 });
    defer retriever.deinit();
    const first = try retriever.forward(arena, "single");
    try testing.expectEqualSlices(usize, &.{1}, first.indices);
    try expectScoreBits(&.{1061997773}, first.scores);
    // Python receives float64 values from NumPy's float32 tolist(), so this
    // score is above the decimal threshold rather than equal to rounded f32.
    try testing.expect(first.scores[0] > 0.8);
    try testing.expect(first.scores[0] != 0.8);
    retriever.k = -1;
    retriever.normalize = false;
    corpus[1] = "renamed";
    const second = try retriever.forward(arena, "mutated");
    try testing.expectEqualStrings("renamed", second.passages[0]);
    try testing.expectEqualSlices(f64, &.{4}, second.scores);
    retriever.corpus_embeddings = .{ .values = &.{ 4, 0, 0, 1 }, .rows = 2, .columns = 2 };
    const third = try retriever.forward(arena, "replaced");
    try testing.expectEqualSlices(usize, &.{0}, third.indices);
    try testing.expectEqualSlices(f64, &.{12}, third.scores);
    try testing.expectEqual(@as(usize, 4), backend.calls.items.len);
}

test "Embeddings ranks the Embedder float32 conversion rather than original float64 values" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const Backend = struct {
        calls: usize = 0,

        fn embed(context: *anyopaque, _: Allocator, _: []const []const u8, _: Record) anyerror!embedding.Batch {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.calls += 1;
            return if (self.calls == 1)
                .{ .float64 = .{ .values = &.{ 1.0 + 0x1p-25, 1.0 }, .rows = 2, .columns = 1 } }
            else
                .{ .float32 = .{ .values = &.{1}, .rows = 1, .columns = 1 } };
        }
    };
    var backend: Backend = .{};
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = Backend.embed }, .{ .caching = false });
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{ "rounded", "one" }, &model, .{ .normalize = false });
    defer retriever.deinit();
    const result = try retriever.forward(memory.allocator(), "query");
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, result.indices);
    try testing.expectEqualSlices(f64, &.{ 1, 1 }, result.scores);
    try testing.expectEqual(@as(usize, 2), backend.calls);
}

test "Embeddings uses signed Python stop slicing including zero and minimum signed k" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 3, 0, 0, 4, 0, 0, -3, 0 }, .rows = 4, .columns = 2 },
        .query = .{ .values = &.{ 3, 4 }, .rows = 1, .columns = 2 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{ "east", "north", "zero", "west" }, &model, .{ .normalize = false });
    defer retriever.deinit();
    for ([_]isize{ 5, 2, 0, -1, -4, -9, 99, std.math.minInt(isize) }, [_]usize{ 4, 2, 0, 3, 0, 0, 4, 0 }) |k, count| {
        retriever.k = k;
        const result = try retriever.forward(arena, "query");
        try testing.expectEqualSlices(usize, (&[_]usize{ 1, 0, 2, 3 })[0..count], result.indices);
        try testing.expectEqualSlices(f64, (&[_]f64{ 16, 9, 0, -9 })[0..count], result.scores);
    }
}

test "Embeddings preserves unstable ties rather than reversing KNN order" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const values: [17]f32 = @splat(1);
    const corpus: [17][]const u8 = @splat("same-score");
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &values, .rows = 17, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &corpus, &model, .{ .k = 17, .normalize = false });
    defer retriever.deinit();
    const result = try retriever.forward(arena, "query");
    try testing.expectEqualSlices(usize, &.{ 0, 14, 13, 12, 11, 10, 9, 15, 8, 6, 5, 4, 3, 2, 1, 7, 16 }, result.indices);
    for (result.scores) |score| try testing.expectEqual(@as(f64, 1), score);
}

test "Embeddings puts NaNs after finite scores with NumPy unstable tie order" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var values: [33]f32 = undefined;
    for (&values, 0..) |*number, i| number.* = if (i % 7 == 0) std.math.nan(f32) else @floatFromInt(i % 3);
    const corpus: [33][]const u8 = @splat("passage");
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &values, .rows = 33, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &corpus, &model, .{ .k = 33, .normalize = false });
    defer retriever.deinit();
    const result = try retriever.forward(arena, "query");
    try testing.expectEqualSlices(usize, &.{
        32, 29, 2,  26, 5,  23, 8,  20, 11, 17, 25, 22, 19, 31, 16, 13, 1,
        10, 4,  15, 30, 27, 3,  24, 6,  9,  18, 12, 7,  21, 28, 14, 0,
    }, result.indices);
    for (result.scores[0..10]) |score| try testing.expectEqual(@as(f64, 2), score);
    for (result.scores[28..]) |score| try testing.expect(std.math.isNan(score));
}

test "Embeddings normalization keeps zero sign tiny rows and nonfinite propagation" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{
            .values = &.{ 0, -0.0, std.math.inf(f32), 1, std.math.nan(f32), 2, 1e-30, 0, 1e30, 1e30 },
            .rows = 5,
            .columns = 2,
        },
        .query = .{ .values = &.{ 1, 0 }, .rows = 1, .columns = 2 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{ "zero", "inf", "nan", "tiny", "overflow" }, &model, .{});
    defer retriever.deinit();
    try testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(retriever.corpus_embeddings.values[1])));
    try testing.expect(std.math.isNan(retriever.corpus_embeddings.values[2]));
    try testing.expect(std.math.isNan(retriever.corpus_embeddings.values[4]));
    try testing.expect(std.math.isNan(retriever.corpus_embeddings.values[5]));
    try expectBits(&.{ 507307272, 0, 0, 0 }, retriever.corpus_embeddings.values[6..]);
    const result = try retriever.forward(arena, "query");
    try testing.expectEqualSlices(usize, &.{ 3, 0, 4, 1, 2 }, result.indices);
    try expectScoreBits(&.{ 507307272, 0, 0 }, result.scores[0..3]);
    try testing.expect(std.math.isNan(result.scores[3]) and std.math.isNan(result.scores[4]));
}

test "Embeddings float32 pairwise normalization matches actual NumPy 17-component vectors" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const input = fromBits(.{
        3248250212, 1091591770, 1084070025, 1099608923, 3240553022, 3222679056,
        3229095719, 1080888462, 3213095501, 1078648211, 3224254039, 1074424164,
        1086320176, 3205974498, 1092348598, 1043679439, 1096248928,
    });
    var backend: TestBackend = .{
        .arena = memory.allocator(),
        .corpus = .{ .values = &input, .rows = 1, .columns = 17 },
        .query = .{ .values = &input, .rows = 1, .columns = 17 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try Embeddings.init(testing.allocator, &.{"random-vector"}, &model, .{});
    defer retriever.deinit();
    try expectBits(&.{
        3205271567, 1048692576, 1041083227, 1056529929, 3197504503, 3179741374,
        3185509575, 1037375445, 3169663334, 1035361592, 3181157190, 1031330062,
        1043105980, 3163029473, 1049372920, 1000535752, 1052879087,
    }, retriever.corpus_embeddings.values);
}

test "Embeddings einsum cancellation changes a real neighbor relative to serial KNN dot" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 1e8, 1, -1e8, 1, 0.25, 0, 0, 0 }, .rows = 2, .columns = 4 },
        .query = .{ .values = &.{ 1, 1, 1, 1 }, .rows = 1, .columns = 4 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{ "cancelled", "quarter" }, &model, .{ .normalize = false });
    defer retriever.deinit();
    const result = try retriever.forward(arena, "query");
    try testing.expectEqualSlices(usize, &.{ 1, 0 }, result.indices);
    try testing.expectEqualSlices(f64, &.{ 0.25, 0 }, result.scores);
}

test "Embeddings fused einsum products match independent NumPy bits across batch shapes" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const query = comptime fromBits(.{
        1045859780, 3199286342, 3211786652, 3214638273, 1070271861, 3214556062,
        3212843186, 3202662348, 1063428715, 1053955964, 3217028444, 1057960827,
        3218259486, 3203273240, 1059952204, 3210747562, 3198474379, 3214186853,
        3186768458, 3207962045, 3215712075, 1007275330, 3192018774, 1075311955,
        3214343613, 3221704478, 1062002762, 3200805104, 3217491188, 3209557834,
        1065977696, 1051192018, 1062937285,
    });
    const corpus = comptime fromBits(.{
        3194449915, 3195077628, 1075391745, 3204779445, 1007910885, 3206786066,
        1041506932, 3218772395, 3207729791, 3213442972, 3206297955, 1062075927,
        3130408543, 3207869012, 1057757694, 1062353878, 3201099470, 1049718803,
        3202294577, 1044917309, 1065394332, 3205345358, 1047837119, 1075516033,
        3190703525, 3212808388, 3208154859, 1053335992, 1027201770, 1057718627,
        3159685432, 1060112396, 3207641931,
    });
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &corpus, .rows = 1, .columns = 33 },
        .query = .{ .values = &query, .rows = 1, .columns = 33 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{"random"}, &model, .{ .normalize = false });
    defer retriever.deinit();
    const single = try retriever.forward(arena, "single");
    try expectScoreBits(&.{1083962968}, single.scores);
    const double_query = query ++ query;
    const double_corpus = corpus ++ corpus;
    backend.query = .{ .values = &double_query, .rows = 2, .columns = 33 };
    retriever.corpus_embeddings = .{ .values = &double_corpus, .rows = 2, .columns = 33 };
    retriever.corpus = &.{ "first", "second" };
    const batch = try retriever.batchForward(arena, &.{ "first-query", "second-query" });
    try testing.expectEqual(@as(usize, 2), batch.len);
    for (batch) |result| {
        try testing.expectEqualSlices(usize, &.{ 0, 1 }, result.indices);
        try expectScoreBits(&.{ 1083962968, 1083962968 }, result.scores);
    }
}

test "Embeddings contracted dimensions broadcast only when one width is one" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 1, 2 }, .rows = 1, .columns = 2 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{"two"}, &model, .{});
    defer retriever.deinit();
    const normalized = try retriever.forward(arena, "scalar");
    try expectScoreBits(&.{1068219106}, normalized.scores);
    backend.query = .{ .values = &.{ 1, 2, 3 }, .rows = 1, .columns = 3 };
    try testing.expectError(error.EmbeddingDimensionMismatch, retriever.forward(arena, "mismatch"));
    retriever.k = 0;
    try testing.expectError(error.EmbeddingDimensionMismatch, retriever.forward(arena, "still-mismatch"));
    retriever.normalize = false;
    retriever.k = 5;
    retriever.corpus_embeddings = .{ .values = &.{2}, .rows = 1, .columns = 1 };
    backend.query = .{ .values = &.{ 1, 2 }, .rows = 1, .columns = 2 };
    const broadcast = try retriever.forward(arena, "wide");
    try testing.expectEqualSlices(f64, &.{6}, broadcast.scores);
    backend.query = .{ .values = &.{}, .rows = 1, .columns = 0 };
    const empty_width = try retriever.forward(arena, "empty-vector");
    try testing.expectEqualSlices(f64, &.{0}, empty_width.scores);
}

test "Embeddings retains backend row counts and only bounds-checks selected passages" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 1, 3 }, .rows = 2, .columns = 1 },
        .query = .{ .values = &.{ 1, 2 }, .rows = 2, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{"only"}, &model, .{ .normalize = false });
    defer retriever.deinit();
    try testing.expectError(error.CorpusIndexOutOfBounds, retriever.batchForward(arena, &.{"one-query"}));
    retriever.k = 0;
    const empty = try retriever.batchForward(arena, &.{"one-query"});
    try testing.expectEqual(@as(usize, 2), empty.len);
    for (empty) |result| try testing.expectEqual(@as(usize, 0), result.indices.len);
    retriever.corpus_embeddings = .{ .values = &.{1}, .rows = 1, .columns = 1 };
    retriever.corpus = &.{ "first", "unused-second" };
    retriever.k = 5;
    const extra_queries = try retriever.batchForward(arena, &.{"one-query"});
    try testing.expectEqual(@as(usize, 2), extra_queries.len);
    try testing.expectEqualSlices(f64, &.{1}, extra_queries[0].scores);
    try testing.expectEqualSlices(f64, &.{2}, extra_queries[1].scores);
    const single = try retriever.forward(arena, "one-query");
    try testing.expectEqualSlices(f64, &.{1}, single.scores);
    backend.query = .{ .values = &.{}, .rows = 0, .columns = 0 };
    try testing.expectError(error.InvalidEmbeddingShape, retriever.forward(arena, "missing-row"));
}

test "Embeddings rejects caching before embedding and FAISS only after corpus embedding" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    var backend: TestBackend = .{
        .arena = memory.allocator(),
        .corpus = .{ .values = &.{1}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    try testing.expectError(error.UnsupportedCache, Embeddings.init(testing.allocator, &.{"one"}, &model, .{ .cache = true }));
    try testing.expectEqual(@as(usize, 0), backend.calls.items.len);
    try testing.expectError(error.UnsupportedFaiss, Embeddings.init(testing.allocator, &.{"one"}, &model, .{ .brute_force_threshold = 1 }));
    try testing.expectEqual(@as(usize, 1), backend.calls.items.len);
}

test "Embeddings distinguishes empty rank-one arrays from zero-width matrix rows" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{}, .rows = 2, .columns = 0 },
        .query = .{ .values = &.{}, .rows = 1, .columns = 0 },
        .first_is_corpus = false,
    };
    var model = backend.model();
    defer model.deinit();
    try testing.expectError(error.InvalidEmbeddingShape, Embeddings.init(testing.allocator, &.{}, &model, .{}));
    try testing.expectEqual(@as(usize, 0), backend.calls.items.len);
    var empty = try Embeddings.init(testing.allocator, &.{}, &model, .{ .normalize = false });
    defer empty.deinit();
    try testing.expectError(error.InvalidEmbeddingShape, empty.forward(arena, "query"));
    try testing.expectError(error.InvalidEmbeddingShape, Embeddings.init(testing.allocator, &.{}, &model, .{ .normalize = false, .brute_force_threshold = 0 }));
    backend.calls.clearRetainingCapacity();
    backend.first_is_corpus = true;
    var zero_width = try EmbeddingsWithScores.init(testing.allocator, &.{ "first", "second" }, &model, .{ .k = 2 });
    defer zero_width.deinit();
    const result = try zero_width.forward(arena, "query");
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, result.indices);
    try testing.expectEqualSlices(f64, &.{ 0, 0 }, result.scores);
    const before = backend.calls.items.len;
    try testing.expectError(error.InvalidEmbeddingShape, zero_width.batchForward(arena, &.{}));
    try testing.expectEqual(before, backend.calls.items.len);
}

// Actual np.lib.format.write_array receipts, NumPy 2.4.6, arrays [[1,2,3],[4,5,6]].
const numpy_npy1_hex = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c2033292c207d202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000803f0000004000004040000080400000a0400000c040";
const numpy_npy2_fortran_big_hex = "934e554d50590200740000007b276465736372273a20273e6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028322c2033292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a3f800000408000004000000040a000004040000040c00000";
const numpy_npy3_hex = "934e554d50590300740000007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c2033292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000803f0000004000004040000080400000a0400000c040";

test "Embeddings save writes independently produced NumPy NPY bytes" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 1, 2, 3, 4, 5, 6 }, .rows = 2, .columns = 3 },
        .query = .{ .values = &.{ 1, 1, 1 }, .rows = 1, .columns = 3 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try EmbeddingsWithScores.init(testing.allocator, &.{ "first", "second" }, &model, .{ .normalize = false, .k = -1 });
    defer retriever.deinit();
    try retriever.save(testing.io, path);
    var expected: [152]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected, numpy_npy1_hex);
    const actual = try temporary.dir.readFileAlloc(testing.io, "corpus_embeddings.npy", arena, .unlimited);
    try testing.expectEqualSlices(u8, &expected, actual);
    const config_bytes = try temporary.dir.readFileAlloc(testing.io, "config.json", arena, .unlimited);
    const config = try std.json.parseFromSliceLeaky(@import("runtime_value.zig").Value, arena, config_bytes, .{});
    try testing.expectEqual(@as(i64, -1), config.object.get("k").?.integer);
    try testing.expect(!config.object.get("normalize").?.bool);
    try testing.expect(!config.object.get("has_faiss_index").?.bool);
    try testing.expectEqualStrings("second", config.object.get("corpus").?.array.items[1].string);
}

test "Embeddings saves degenerate NumPy Fortran headers canonically" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    // Inputs are genuine NumPy 1.x files with fortran_order=True; outputs are DSPy's np.save results.
    const cases = [_]struct { input: []const u8, expected: []const u8, rows: usize, columns: usize, names: []const []const u8 }{
        .{
            .input = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028312c2033292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a000000000000803f00000040",
            .expected = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028312c2033292c207d202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a000000000000803f00000040",
            .rows = 1,
            .columns = 3,
            .names = &.{"degenerate-row-0"},
        },
        .{
            .input = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028332c2031292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a000000000000803f00000040",
            .expected = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028332c2031292c207d202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a000000000000803f00000040",
            .rows = 3,
            .columns = 1,
            .names = &.{ "degenerate-column-0", "degenerate-column-1", "degenerate-column-2" },
        },
    };
    for (cases) |case| {
        var input_buffer: [256]u8 = undefined;
        var expected_buffer: [256]u8 = undefined;
        const input = try std.fmt.hexToBytes(&input_buffer, case.input);
        const expected = try std.fmt.hexToBytes(&expected_buffer, case.expected);
        const values = [_]f32{ 0, 1, 2 };
        var backend: TestBackend = .{
            .arena = arena,
            .corpus = .{ .values = &values, .rows = case.rows, .columns = case.columns },
            .query = .{ .values = &.{ 1, 1, 1 }, .rows = 1, .columns = 3 },
        };
        var model = backend.model();
        defer model.deinit();
        var retriever = try EmbeddingsWithScores.init(testing.allocator, case.names, &model, .{ .normalize = false, .k = -1 });
        defer retriever.deinit();
        try retriever.save(testing.io, path);
        try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = input });
        var loaded = try EmbeddingsWithScores.fromSaved(testing.allocator, testing.io, path, &model);
        defer loaded.deinit();
        try testing.expectEqualSlices(f32, &values, loaded.corpus_embeddings.values);
        try loaded.save(testing.io, path);
        const saved = try temporary.dir.readFileAlloc(testing.io, "corpus_embeddings.npy", arena, .unlimited);
        try testing.expectEqualSlices(u8, expected, saved);
    }
}

test "Embeddings loads genuine NumPy versions and endian Fortran arrays without recomputing" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"k\":2,\"normalize\":false,\"corpus\":[\"first\",\"second\"],\"has_faiss_index\":false}" });
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{999}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{ 1, 1, 1 }, .rows = 1, .columns = 3 },
        .first_is_corpus = false,
    };
    var model = backend.model();
    defer model.deinit();
    for ([_][]const u8{ numpy_npy1_hex, numpy_npy2_fortran_big_hex, numpy_npy3_hex }) |hex| {
        var bytes: [152]u8 = undefined;
        _ = try std.fmt.hexToBytes(&bytes, hex);
        try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = &bytes });
        const before = backend.calls.items.len;
        var loaded = try EmbeddingsWithScores.fromSaved(testing.allocator, testing.io, path, &model);
        defer loaded.deinit();
        try testing.expectEqual(before, backend.calls.items.len);
        try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, loaded.corpus_embeddings.values);
        const result = try loaded.forward(arena, "query");
        try testing.expectEqualSlices(usize, &.{ 1, 0 }, result.indices);
        try testing.expectEqualSlices(f64, &.{ 15, 6 }, result.scores);
        try testing.expectEqualStrings("second", result.passages[0]);
        try testing.expectEqual(before + 1, backend.calls.items.len);
    }
}

test "Embeddings load replaces an existing retriever and fromSaved retains normalized vectors" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{ 3, 0, 0, 4, 0, 0 }, .rows = 3, .columns = 2 },
        .query = .{ .values = &.{ 3, 4 }, .rows = 1, .columns = 2 },
    };
    var model = backend.model();
    defer model.deinit();
    var original = try Embeddings.init(testing.allocator, &.{ "east", "north", "zero" }, &model, .{ .k = -1 });
    defer original.deinit();
    try original.save(testing.io, path);
    var new_backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{999}, .rows = 1, .columns = 1 },
        .query = backend.query,
        .first_is_corpus = false,
    };
    var new_model = new_backend.model();
    defer new_model.deinit();
    var loaded = try EmbeddingsWithScores.fromSaved(testing.allocator, testing.io, path, &new_model);
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 0), new_backend.calls.items.len);
    const result = try loaded.forward(arena, "query");
    try testing.expectEqualSlices(usize, &.{ 1, 0 }, result.indices);
    try expectScoreBits(&.{ 1061997773, 1058642330 }, result.scores);
    original.k = 0;
    original.normalize = false;
    try testing.expect(try original.load(testing.io, path, &new_model) == &original);
    try testing.expectEqual(@as(isize, -1), original.k);
    try testing.expect(original.normalize);
    try testing.expect(original.embedder == &new_model);
    try testing.expectEqual(@as(usize, 1), backend.calls.items.len);
    try testing.expectEqual(@as(usize, 1), new_backend.calls.items.len);
    try testing.expectEqualStrings("north", (try original.forward(arena, "query")).passages[0]);
}

test "Embeddings missing files and invalid config retain prior fields while corrupt NPY applies config" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{1}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    var retriever = try Embeddings.init(testing.allocator, &.{"old"}, &model, .{ .k = 3 });
    defer retriever.deinit();
    try testing.expectError(error.FileNotFound, retriever.load(testing.io, path, &model));
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "invalid-json" });
    // Source checks the array's existence before parsing the invalid config.
    try testing.expectError(error.FileNotFound, retriever.load(testing.io, path, &model));
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = "invalid-npy" });
    for ([_][]const u8{
        "{\"normalize\":true,\"corpus\":[],\"has_faiss_index\":false}",
        "{\"k\":2,\"corpus\":[],\"has_faiss_index\":false}",
        "{\"k\":2,\"normalize\":true,\"has_faiss_index\":false}",
        "{\"k\":2,\"normalize\":true,\"corpus\":[]}",
    }) |config| {
        try temporary.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = config });
        try testing.expectError(error.MissingConfigField, retriever.load(testing.io, path, &model));
        try testing.expectEqual(@as(isize, 3), retriever.k);
        try testing.expectEqualStrings("old", retriever.corpus[0]);
    }
    try temporary.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data = "{\"k\":-2,\"normalize\":false,\"corpus\":[\"new\"],\"has_faiss_index\":false}",
    });
    try testing.expectError(error.InvalidNpyHeader, retriever.load(testing.io, path, &model));
    try testing.expectEqual(@as(isize, -2), retriever.k);
    try testing.expect(!retriever.normalize);
    try testing.expectEqualStrings("new", retriever.corpus[0]);
    try testing.expectEqualSlices(f32, &.{1}, retriever.corpus_embeddings.values);
    try testing.expectEqual(@as(usize, 1), backend.calls.items.len);
}

test "Embeddings rejects unsupported saved dtype index and truncated payload without embedding" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{1}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{ 1, 1, 1 }, .rows = 1, .columns = 3 },
    };
    var model = backend.model();
    defer model.deinit();
    const config = "{\"k\":5,\"normalize\":false,\"corpus\":[\"first\",\"second\"],\"has_faiss_index\":false}";
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = config });
    var bytes: [152]u8 = undefined;
    _ = try std.fmt.hexToBytes(&bytes, numpy_npy1_hex);
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = bytes[0..151] });
    try testing.expectError(error.TruncatedNpyData, Embeddings.fromSaved(testing.allocator, testing.io, path, &model));
    // The descriptor byte comes from the independent NumPy header, not native encoding.
    const dtype = std.mem.indexOf(u8, &bytes, "<f4").?;
    bytes[dtype + 2] = '8';
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = &bytes });
    try testing.expectError(error.UnsupportedNpyDtype, Embeddings.fromSaved(testing.allocator, testing.io, path, &model));
    bytes[dtype + 2] = '4';
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = &bytes });
    try temporary.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data = "{\"k\":5,\"normalize\":false,\"corpus\":[\"first\",\"second\"],\"has_faiss_index\":true}",
    });
    try testing.expectError(error.UnsupportedFaiss, Embeddings.fromSaved(testing.allocator, testing.io, path, &model));
    try testing.expectEqual(@as(usize, 0), backend.calls.items.len);
}

test "Embeddings preserves NumPy empty vector empty matrix and scalar persistence shapes" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    try temporary.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data = "{\"k\":-1,\"normalize\":false,\"corpus\":[\"unused\"],\"has_faiss_index\":false}",
    });
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{999}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{ 3, 4 }, .rows = 1, .columns = 2 },
        .first_is_corpus = false,
    };
    var model = backend.model();
    defer model.deinit();
    const Case = struct { rank: ArrayRank, hex: []const u8 };
    // Actual np.save(np.zeros(shape, np.float32)) bytes from the pinned oracle.
    const cases = [_]Case{
        .{
            .rank = .vector,
            .hex = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028302c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a",
        },
        .{
            .rank = .matrix,
            .hex = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028302c2032292c207d202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a",
        },
        .{
            .rank = .scalar,
            .hex = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a00000000",
        },
    };
    for (cases) |case| {
        var buffer: [132]u8 = undefined;
        const bytes = try std.fmt.hexToBytes(&buffer, case.hex);
        try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = bytes });
        const before = backend.calls.items.len;
        var loaded = try EmbeddingsWithScores.fromSaved(testing.allocator, testing.io, path, &model);
        defer loaded.deinit();
        try testing.expectEqual(before, backend.calls.items.len);
        try testing.expectEqual(case.rank, loaded.corpus_rank);
        try loaded.save(testing.io, path);
        const saved = try temporary.dir.readFileAlloc(testing.io, "corpus_embeddings.npy", arena, .unlimited);
        try testing.expectEqualSlices(u8, bytes, saved);
        if (case.rank == .matrix) {
            const result = try loaded.forward(arena, "query");
            try testing.expectEqual(@as(usize, 0), result.indices.len);
            try testing.expectEqual(@as(usize, 0), result.passages.len);
            try testing.expectEqual(@as(usize, 0), result.scores.len);
        } else {
            try testing.expectError(error.InvalidEmbeddingShape, loaded.forward(arena, "query"));
        }
    }
}

test "Embeddings rejects oversized or unsupported NPY headers as data" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(testing.io, ".", arena);
    try temporary.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data = "{\"k\":5,\"normalize\":false,\"corpus\":[\"first\",\"second\"],\"has_faiss_index\":false}",
    });
    var backend: TestBackend = .{
        .arena = arena,
        .corpus = .{ .values = &.{1}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = backend.model();
    defer model.deinit();
    var bytes: [152]u8 = undefined;
    _ = try std.fmt.hexToBytes(&bytes, numpy_npy1_hex);
    bytes[6] = 4;
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = &bytes });
    try testing.expectError(error.UnsupportedNpyVersion, Embeddings.fromSaved(testing.allocator, testing.io, path, &model));
    bytes[6] = 1;
    std.mem.writeInt(u16, bytes[8..10], 10_001, .little);
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "corpus_embeddings.npy", .data = &bytes });
    try testing.expectError(error.NpyHeaderTooLarge, Embeddings.fromSaved(testing.allocator, testing.io, path, &model));
    const expression = "{'descr': '<f4', 'fortran_order': False, 'shape': (1 + 1, 3), }\n";
    var prefix = [_]u8{ 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0 };
    std.mem.writeInt(u16, prefix[8..10], expression.len, .little);
    {
        const file = try temporary.dir.createFile(testing.io, "corpus_embeddings.npy", .{});
        defer file.close(testing.io);
        try file.writeStreamingAll(testing.io, &prefix);
        try file.writeStreamingAll(testing.io, expression);
    }
    try testing.expectError(error.InvalidNpyHeader, Embeddings.fromSaved(testing.allocator, testing.io, path, &model));
    try testing.expectEqual(@as(usize, 0), backend.calls.items.len);
}
