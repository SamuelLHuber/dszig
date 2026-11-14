//! Native `dspy.KNN`: cached training embeddings and raw float32 dot scores.
//!
//! `out` owns the training vectors; it must outlive the retriever. Training
//! examples and the Embedder are borrowed, and selected Examples retain their
//! original records. Each call's text, scores, indices, and result slice belong
//! to its caller arena. No training embeddings are recomputed during a call.
//!
//! Sorting follows NumPy 2.4.6's scalar `aquicksort_`, including its unstable
//! ties, NaNs-last comparison, insertion threshold, and heapsort fallback. This
//! is the locked Darwin/arm64 oracle's argsort kernel; NumPy's x86 SIMD dispatch
//! can order ties differently. Dot accumulation follows NumPy's non-BLAS FLOAT
//! kernel with strict f32 arithmetic. BLAS kernels are not bitwise portable:
//! the locked Apple Accelerate backend returns 2 for [1e8,1,-1e8,1] dot ones,
//! versus this scalar kernel's 1, and even changes accumulation with row count.
//! Consequently cancellation-sensitive rankings are not a BLAS-equivalence
//! claim. Neither normalization nor an invented stable tie-break is applied.
//!
//! Sources: DSPy 3.5.0 predict/knn.py; NumPy v2.4.6
//! numpy/_core/src/npysort/{quicksort.cpp,npysort_heapsort.h,npysort_common.h}
//! and numpy/_core/src/multiarray/arraytypes.c.src (FLOAT_dot fallback).

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const embedding = @import("embedder.zig");
const optimize = @import("optimize.zig");
const pyjson = @import("pyjson.zig");
const value = @import("value.zig");
const Example = optimize.Example;
const Matrix = embedding.Matrix;
const Record = value.Record;

pub const KNN = struct {
    k: isize,
    trainset: []const Example,
    embedding: *embedding.Embedder,
    trainset_vectors: Matrix,

    pub fn init(
        out: Allocator,
        k: isize,
        trainset: []const Example,
        vectorizer: *embedding.Embedder,
    ) anyerror!KNN {
        const texts = try out.alloc([]const u8, trainset.len);
        for (trainset, texts) |*example, *text| {
            text.* = try recordText(out, &example.record, example);
        }
        const vectors = try vectorizer.embed(out, texts, .{});
        try validateMatrix(vectors);
        return .{ .k = k, .trainset = trainset, .embedding = vectorizer, .trainset_vectors = vectors };
    }

    /// The input Record's insertion order is Python's keyword-argument order.
    /// A malformed backend is not assumed to return one row per input: DSPy
    /// permits fewer training rows and only fails if a selected index cannot
    /// index the original trainset. Multiple query rows can squeeze to a vector
    /// when there is one training row; a genuine 2-D index is an error instead.
    pub fn call(self: *const KNN, arena: Allocator, kwargs: Record) anyerror![]const Example {
        const text = try recordText(arena, &kwargs, null);
        const query = try self.embedding.embed(arena, &.{text}, .{});
        try validateMatrix(query);
        try validateMatrix(self.trainset_vectors);
        const training = self.trainset_vectors;
        // Embedder._postprocess builds np.array([]), a rank-one empty array,
        // whenever a backend emits no rows. Its dot behavior differs from a
        // nonempty matrix whose vectors merely have zero columns.
        const training_width = if (training.rows == 0) 0 else training.columns;
        const query_width = if (query.rows == 0) 0 else query.columns;
        if (training_width != query_width) return error.EmbeddingDimensionMismatch;
        if (training.rows > 1 and query.rows > 1) {
            if (selectedCount(self.k, training.rows) == 0) return &.{};
            return error.NonScalarNeighborIndex;
        }
        const count = if (training.rows == 0)
            @max(query.rows, 1)
        else if (query.rows == 0)
            training.rows
        else
            @max(training.rows, query.rows);
        const scores = try arena.alloc(f32, count);
        if (training.rows == 0 or query.rows == 0) {
            @memset(scores, 0);
        } else {
            for (scores, 0..) |*score, i| {
                const train_row = if (training.rows == 1) 0 else i;
                const query_row = if (query.rows == 1) 0 else i;
                score.* = dot(training.row(train_row), query.row(query_row));
            }
        }
        const indices = try arena.alloc(usize, count);
        argsort(scores, indices);
        const selected = try arena.alloc(Example, selectedCount(self.k, count));
        for (selected, 0..) |*example, i| {
            const index = indices[count - 1 - i];
            if (index >= self.trainset.len) return error.NeighborIndexOutOfBounds;
            example.* = self.trainset[index];
        }
        return selected;
    }
};

fn validateMatrix(matrix: Matrix) error{InvalidEmbeddingShape}!void {
    const size = std.math.mul(usize, matrix.rows, matrix.columns) catch
        return error.InvalidEmbeddingShape;
    if (size != matrix.values.len) return error.InvalidEmbeddingShape;
}

fn recordText(arena: Allocator, record: *const Record, example: ?*const Example) ![]const u8 {
    var writer: std.Io.Writer.Allocating = .init(arena);
    var first = true;
    var iterator = record.iterator();
    while (iterator.next()) |entry| {
        if (example) |ex| {
            // Example.items() hides internal fields even if with_inputs marks
            // them as inputs. Query kwargs are an ordinary dict, not Example.
            if (std.mem.startsWith(u8, entry.key_ptr.*, "dspy_")) continue;
            if (!ex.isInput(entry.key_ptr.*)) continue;
        }
        if (!first) try writer.writer.writeAll(" | ");
        first = false;
        try writer.writer.writeAll(entry.key_ptr.*);
        try writer.writer.writeAll(": ");
        try pyjson.writeStr(&writer.writer, entry.value_ptr.*);
    }
    return writer.written();
}

fn selectedCount(k: isize, count: usize) usize {
    if (k == 0) return count;
    if (k > 0) return @min(@as(usize, @intCast(k)), count);
    return count - @min(@abs(k), count);
}

fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    @setFloatMode(.strict);
    var sum: f32 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

// The indexed introsort/heapsort below are translations of NumPy v2.4.6.
// Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
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

fn less(a: f32, b: f32) bool {
    @setFloatMode(.strict);
    return a < b or (std.math.isNan(b) and !std.math.isNan(a));
}

fn argsort(scores: []const f32, indices: []usize) void {
    assert(scores.len == indices.len);
    for (indices, 0..) |*index, i| index.* = i;
    if (indices.len < 2) return;
    const Partition = struct { left: usize, right: usize, depth: i32 };
    var stack: [@bitSizeOf(usize)]Partition = undefined;
    var stack_len: usize = 0;
    var part: Partition = .{
        .left = 0,
        .right = indices.len - 1,
        .depth = 2 * @as(i32, @intCast(std.math.log2_int(usize, indices.len))),
    };
    while (true) {
        if (part.depth < 0) {
            heapArgsort(scores, indices[part.left .. part.right + 1]);
        } else {
            while (part.right - part.left > 15) {
                const pivot = partition(scores, indices, part.left, part.right);
                part.depth -= 1;
                assert(stack_len < stack.len);
                if (pivot - part.left < part.right - pivot) {
                    stack[stack_len] = .{ .left = pivot + 1, .right = part.right, .depth = part.depth };
                    part.right = pivot - 1;
                } else {
                    stack[stack_len] = .{ .left = part.left, .right = pivot - 1, .depth = part.depth };
                    part.left = pivot + 1;
                }
                stack_len += 1;
            }
            insertionArgsort(scores, indices, part.left, part.right);
        }
        if (stack_len == 0) break;
        stack_len -= 1;
        part = stack[stack_len];
    }
}

fn partition(scores: []const f32, indices: []usize, left: usize, right: usize) usize {
    assert(right - left > 15);
    const middle = left + (right - left) / 2;
    if (less(scores[indices[middle]], scores[indices[left]]))
        std.mem.swap(usize, &indices[middle], &indices[left]);
    if (less(scores[indices[right]], scores[indices[middle]]))
        std.mem.swap(usize, &indices[right], &indices[middle]);
    if (less(scores[indices[middle]], scores[indices[left]]))
        std.mem.swap(usize, &indices[middle], &indices[left]);
    const pivot = scores[indices[middle]];
    var i = left;
    var j = right - 1;
    std.mem.swap(usize, &indices[middle], &indices[j]);
    while (true) {
        i += 1;
        while (less(scores[indices[i]], pivot)) i += 1;
        j -= 1;
        while (less(pivot, scores[indices[j]])) j -= 1;
        if (i >= j) break;
        std.mem.swap(usize, &indices[i], &indices[j]);
    }
    std.mem.swap(usize, &indices[i], &indices[right - 1]);
    assert(i > left and i < right);
    return i;
}

fn insertionArgsort(scores: []const f32, indices: []usize, left: usize, right: usize) void {
    var i = left + 1;
    while (i <= right) : (i += 1) {
        const index = indices[i];
        const score = scores[index];
        var j = i;
        while (j > left and less(score, scores[indices[j - 1]])) : (j -= 1) {
            indices[j] = indices[j - 1];
        }
        indices[j] = index;
    }
}

fn heapArgsort(scores: []const f32, indices: []usize) void {
    var n = indices.len;
    var root = n / 2;
    while (root > 0) : (root -= 1) {
        heapSift(scores, indices, root, n, indices[root - 1]);
    }
    while (n > 1) {
        const temporary = indices[n - 1];
        indices[n - 1] = indices[0];
        n -= 1;
        heapSift(scores, indices, 1, n, temporary);
    }
}

fn heapSift(scores: []const f32, indices: []usize, root: usize, count: usize, temporary: usize) void {
    var parent = root;
    while (parent <= count / 2) {
        var child = parent * 2;
        if (child < count and less(scores[indices[child - 1]], scores[indices[child]])) child += 1;
        if (!less(scores[temporary], scores[indices[child - 1]])) break;
        indices[parent - 1] = indices[child - 1];
        parent = child;
    }
    indices[parent - 1] = temporary;
}

const testing = std.testing;

const TestEmbedding = struct {
    log_arena: Allocator,
    training: Matrix,
    query: Matrix,
    skip_training: bool = false,
    calls: std.ArrayList([]const []const u8) = .empty,

    fn embed(ptr: *anyopaque, arena: Allocator, inputs: []const []const u8, _: Record) anyerror!embedding.Batch {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const first = self.calls.items.len == 0;
        const copy = try self.log_arena.alloc([]const u8, inputs.len);
        for (inputs, copy) |text, *saved| saved.* = try self.log_arena.dupe(u8, text);
        try self.calls.append(self.log_arena, copy);
        const matrix = if (first and !self.skip_training) self.training else self.query;
        return .{ .float32 = .{
            .values = try arena.dupe(f32, matrix.values),
            .rows = matrix.rows,
            .columns = matrix.columns,
        } };
    }
};

test "KNN text uses store order, Python str, input filtering, and original Example records" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const trainset = [_]Example{
        try Example.fromJson(arena, "{\"answer\":\"excluded\",\"z\":true,\"a\":[1,\"two\",null],\"unused\":\"skip\",\"nested\":{\"yes\":false},\"number\":1e20}", &.{ "number", "nested", "a", "z" }),
        try Example.fromJson(arena, "{\"answer\":\"other\",\"z\":false,\"a\":[]}", &.{ "a", "z" }),
    };
    var backend: TestEmbedding = .{
        .log_arena = arena,
        .training = .{ .values = &.{ 1, 1, 2, 1 }, .rows = 2, .columns = 2 },
        .query = .{ .values = &.{ 1, 1 }, .rows = 1, .columns = 2 },
    };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = TestEmbedding.embed }, .{ .caching = false });
    defer model.deinit();
    const knn = try KNN.init(arena, 0, &trainset, &model);
    const query = try Example.fromJson(arena, "{\"number\":1e-7,\"nested\":{\"yes\":null},\"z\":false,\"a\":[true,\"can't\"]}", &.{});
    const result = try knn.call(arena, query.record);
    _ = try knn.call(arena, query.record);
    try testing.expectEqual(@as(usize, 3), backend.calls.items.len);
    try testing.expectEqualStrings("z: True | a: [1, 'two', None] | nested: {'yes': False} | number: 1e+20", backend.calls.items[0][0]);
    try testing.expectEqualStrings("z: False | a: []", backend.calls.items[0][1]);
    try testing.expectEqualStrings("number: 1e-07 | nested: {'yes': None} | z: False | a: [True, \"can't\"]", backend.calls.items[1][0]);
    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expect(result[0].record.keys().ptr == trainset[1].record.keys().ptr);
    try testing.expect(result[1].record.keys().ptr == trainset[0].record.keys().ptr);
}

test "KNN hides reserved Example metadata in training text but preserves all query kwargs" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const trainset = [_]Example{
        try Example.fromJson(arena, "{\"answer\":\"output\",\"dspy_metadata\":\"hidden\",\"question\":\"visible\"}", &.{ "dspy_metadata", "question" }),
    };
    var backend: TestEmbedding = .{
        .log_arena = arena,
        .training = .{ .values = &.{1}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = TestEmbedding.embed }, .{ .caching = false });
    defer model.deinit();
    const knn = try KNN.init(arena, 1, &trainset, &model);
    const query = try Example.fromJson(arena, "{\"dspy_metadata\":\"query-visible\",\"question\":\"query\"}", &.{});
    const selected = try knn.call(arena, query.record);
    try testing.expectEqualStrings("question: visible", backend.calls.items[0][0]);
    try testing.expectEqualStrings("dspy_metadata: query-visible | question: query", backend.calls.items[1][0]);
    try testing.expectEqualStrings("hidden", selected[0].record.get("dspy_metadata").?.string);
    try testing.expect(selected[0].record.keys().ptr == trainset[0].record.keys().ptr);
}

test "KNN Python str escapes nonprintable Unicode only inside nested repr strings" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const trainset = [_]Example{
        try Example.fromJson(arena, "{\"values\":[\"\\u0085\",\"\\u00a0\",\"\\u200b\",\"\\u2028\",\"\\u2029\",\"\\ufeff\",\"日本語\",\"é\"]}", &.{"values"}),
    };
    var backend: TestEmbedding = .{
        .log_arena = arena,
        .training = .{ .values = &.{1}, .rows = 1, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = TestEmbedding.embed }, .{ .caching = false });
    defer model.deinit();
    const knn = try KNN.init(arena, 1, &trainset, &model);
    const query = try Example.fromJson(arena, "{\"values\":[\"\\u200b\",\"日本語\"],\"text\":\"\\u200b\"}", &.{});
    _ = try knn.call(arena, query.record);
    try testing.expectEqualStrings(
        "values: ['\\x85', '\\xa0', '\\u200b', '\\u2028', '\\u2029', '\\ufeff', '日本語', 'é']",
        backend.calls.items[0][0],
    );
    try testing.expectEqualStrings("values: ['\\u200b', '日本語'] | text: \u{200b}", backend.calls.items[1][0]);
}

test "KNN raw dots are not cosine scores and Embedder casts both operands to float32" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const trainset = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"zero\",\"index\":0}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\":\"one\",\"index\":1}", &.{"question"}),
    };
    var backend: TestEmbedding = .{
        .log_arena = arena,
        .training = .{ .values = &.{ 10, 0, 1, 1 }, .rows = 2, .columns = 2 },
        .query = .{ .values = &.{ 1, 1 }, .rows = 1, .columns = 2 },
    };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = TestEmbedding.embed }, .{ .caching = false });
    defer model.deinit();
    const knn = try KNN.init(arena, 1, &trainset, &model);
    const result = try knn.call(arena, trainset[0].record);
    try testing.expectEqual(@as(i64, 0), result[0].record.get("index").?.integer);
    const Rounded = struct {
        calls: usize = 0,
        fn embed(ptr: *anyopaque, _: Allocator, _: []const []const u8, _: Record) anyerror!embedding.Batch {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            const epsilon = 0x1p-25;
            return .{ .float64 = if (self.calls == 1)
                .{ .values = &.{ 1 + epsilon, -1, 0, 0 }, .rows = 2, .columns = 2 }
            else
                .{ .values = &.{ 1 + epsilon, 1 }, .rows = 1, .columns = 2 } };
        }
    };
    var rounded_backend: Rounded = .{};
    var rounded_model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &rounded_backend, .embed = Rounded.embed }, .{ .caching = false });
    defer rounded_model.deinit();
    const rounded = try KNN.init(arena, 1, &trainset, &rounded_model);
    const rounded_result = try rounded.call(arena, trainset[0].record);
    try testing.expectEqual(@as(i64, 1), rounded_result[0].record.get("index").?.integer);
    // The actual locked BLAS oracle gives 2 here. This explicitly tests the
    // documented source non-BLAS kernel rather than concealing that boundary.
    try testing.expectEqual(@as(f32, 1), dot(&.{ 1e8, 1, -1e8, 1 }, &.{ 1, 1, 1, 1 }));
}

test "KNN NumPy argsort reproduces unstable large ties, signed zeros, and NaNs" {
    var indices: [64]usize = undefined;
    var scores: [64]f32 = @splat(1);
    argsort(scores[0..17], indices[0..17]);
    try testing.expectEqualSlices(usize, &.{ 0, 14, 13, 12, 11, 10, 9, 15, 8, 6, 5, 4, 3, 2, 1, 7, 16 }, indices[0..17]);
    argsort(&scores, &indices);
    try testing.expectEqualSlices(usize, &.{ 0, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 33, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 48, 32, 31, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 62, 63 }, &indices);
    for (&scores, 0..) |*score, i| score.* = @floatFromInt((i * 7) % 5);
    argsort(scores[0..33], indices[0..33]);
    try testing.expectEqualSlices(usize, &.{ 0, 30, 5, 25, 10, 20, 15, 28, 3, 23, 8, 18, 13, 26, 21, 31, 16, 11, 6, 1, 19, 9, 24, 4, 29, 14, 17, 12, 22, 7, 27, 2, 32 }, indices[0..33]);
    for (&scores, 0..) |*score, i| {
        score.* = if (i % 7 == 0) std.math.nan(f32) else @as(f32, @floatFromInt(i % 3)) - 1;
    }
    argsort(scores[0..33], indices[0..33]);
    try testing.expectEqualSlices(usize, &.{ 18, 30, 3, 15, 6, 27, 24, 9, 12, 16, 31, 13, 10, 19, 4, 1, 25, 22, 26, 29, 23, 32, 20, 17, 11, 8, 5, 2, 14, 7, 28, 21, 0 }, indices[0..33]);
    for (&scores, 0..) |*score, i| score.* = if (i % 2 == 0) 0.0 else -0.0;
    argsort(scores[0..17], indices[0..17]);
    try testing.expectEqualSlices(usize, &.{ 0, 14, 13, 12, 11, 10, 9, 15, 8, 6, 5, 4, 3, 2, 1, 7, 16 }, indices[0..17]);
    for (&indices, 0..) |*index, i| index.* = i;
    heapArgsort(&scores, &indices);
    for (indices[0..63], 1..) |index, expected| try testing.expectEqual(expected, index);
    // This organ-pipe input exhausts the source introsort depth budget; the
    // 140..158 interval is handled by indexed heapsort, not insertion sort.
    var pipe_scores: [257]f32 = undefined;
    var pipe_indices: [pipe_scores.len]usize = undefined;
    for (&pipe_scores, 0..) |*score, i| {
        score.* = @floatFromInt(@min(i, pipe_scores.len - 1 - i));
    }
    argsort(&pipe_scores, &pipe_indices);
    try testing.expectEqualSlices(usize, &.{
        70, 186, 185, 71,  72, 184, 73,  183, 182,
        74, 181, 75,  180, 76, 77,  179, 78,  178,
    }, pipe_indices[140..158]);
    // Hash every source-sorted index, not just the tied fallback interval.
    // Encoding is explicitly u64 little-endian, independent of host usize.
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    for (pipe_indices) |index| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(u64, &encoded, @intCast(index), .little);
        hasher.update(&encoded);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    try testing.expectEqualStrings(
        "ea0b85244a3366a58540ee95cda4d39ab687122da0eb13353f256cfb0d2080d5",
        &hex,
    );
    try testing.expectEqual(@as(usize, 0), indices[63]);
}

test "KNN selection preserves Python negative-zero and negative-k slicing" {
    const ascending = [_]usize{ 6, 3, 4, 0, 2, 5, 1, 7 };
    const scores = [_]f32{ 2, std.math.nan(f32), 2, -0.0, 0.0, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    var indices: [scores.len]usize = undefined;
    argsort(&scores, &indices);
    try testing.expectEqualSlices(usize, &ascending, &indices);
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const examples = try arena.alloc(Example, scores.len);
    for (examples, 0..) |*example, i| {
        var record: Record = .empty;
        try record.put(arena, "index", .{ .integer = @intCast(i) });
        example.* = .{ .record = record, .input_keys = &.{"index"} };
    }
    var backend: TestEmbedding = .{
        .log_arena = arena,
        .training = .{ .values = &scores, .rows = scores.len, .columns = 1 },
        .query = .{ .values = &.{1}, .rows = 1, .columns = 1 },
    };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = TestEmbedding.embed }, .{ .caching = false });
    defer model.deinit();
    var knn = try KNN.init(arena, 0, examples, &model);
    const Case = struct { k: isize, count: usize };
    for ([_]Case{
        .{ .k = 0, .count = 8 },  .{ .k = 1, .count = 1 },  .{ .k = 3, .count = 3 },
        .{ .k = -1, .count = 7 }, .{ .k = -3, .count = 5 }, .{ .k = -8, .count = 0 },
        .{ .k = -9, .count = 0 }, .{ .k = 9, .count = 8 },  .{ .k = std.math.minInt(isize), .count = 0 },
    }) |case| {
        try testing.expectEqual(case.count, selectedCount(case.k, scores.len));
        knn.k = case.k;
        const selected = try knn.call(arena, .empty);
        try testing.expectEqual(case.count, selected.len);
        for (selected, 0..) |example, i| {
            const expected = ascending[scores.len - 1 - i];
            try testing.expectEqual(@as(i64, @intCast(expected)), example.record.get("index").?.integer);
        }
    }
}

test "KNN empty and malformed backend shapes follow dot squeeze and original-list indexing" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const examples = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"zero\",\"index\":0}", &.{"question"}),
        try Example.fromJson(arena, "{\"question\":\"one\",\"index\":1}", &.{"question"}),
    };
    const empty: Matrix = .{ .values = &.{}, .rows = 0, .columns = 0 };
    const zero_two: Matrix = .{ .values = &.{}, .rows = 2, .columns = 0 };
    const zero_one: Matrix = .{ .values = &.{}, .rows = 1, .columns = 0 };
    const one: Matrix = .{ .values = &.{1}, .rows = 1, .columns = 1 };
    const two: Matrix = .{ .values = &.{ 1, 2 }, .rows = 2, .columns = 1 };
    const Case = struct {
        training: Matrix,
        query: Matrix,
        train_count: usize = 2,
        k: isize = 1,
        expected: []const usize = &.{},
        failure: ?anyerror = null,
    };
    for ([_]Case{
        .{ .training = empty, .query = one, .train_count = 0, .failure = error.EmbeddingDimensionMismatch },
        .{ .training = empty, .query = zero_one, .train_count = 0, .failure = error.NeighborIndexOutOfBounds },
        .{ .training = empty, .query = zero_one, .train_count = 0, .k = -1 },
        .{ .training = zero_two, .query = zero_one, .k = 0, .expected = &.{ 1, 0 } },
        .{ .training = two, .query = zero_one, .failure = error.EmbeddingDimensionMismatch },
        .{ .training = .{ .values = &.{ 1, 2, 3 }, .rows = 3, .columns = 1 }, .query = one, .failure = error.NeighborIndexOutOfBounds },
        .{ .training = one, .query = one, .expected = &.{0} },
        .{ .training = two, .query = two, .failure = error.NonScalarNeighborIndex },
        .{ .training = two, .query = two, .k = -2 },
        .{ .training = one, .query = two, .k = 0, .expected = &.{ 1, 0 } },
        .{ .training = two, .query = empty, .failure = error.EmbeddingDimensionMismatch },
        .{ .training = zero_two, .query = empty, .k = 0, .expected = &.{ 1, 0 } },
        .{ .training = empty, .query = zero_one, .k = 0, .expected = &.{0} },
    }) |case| {
        var backend: TestEmbedding = .{
            .log_arena = arena,
            .training = case.training,
            .query = case.query,
            .skip_training = case.train_count == 0,
        };
        var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &backend, .embed = TestEmbedding.embed }, .{ .caching = false });
        defer model.deinit();
        const knn = try KNN.init(arena, case.k, examples[0..case.train_count], &model);
        const result = knn.call(arena, examples[0].record);
        if (case.failure) |failure| {
            try testing.expectError(failure, result);
        } else {
            const selected = try result;
            try testing.expectEqual(case.expected.len, selected.len);
            for (selected, case.expected) |example, index| {
                try testing.expectEqual(@as(i64, @intCast(index)), example.record.get("index").?.integer);
            }
        }
    }
    try testing.expectError(error.InvalidEmbeddingShape, validateMatrix(.{ .values = &.{1}, .rows = 2, .columns = 1 }));
    try testing.expectError(error.InvalidEmbeddingShape, validateMatrix(.{ .values = &.{}, .rows = std.math.maxInt(usize), .columns = 2 }));
}
