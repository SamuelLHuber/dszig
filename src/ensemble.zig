//! `dspy.Ensemble` and `dspy.majority` (DSPy 3.5.0).
//!
//! `Ensemble(P, n)` runs every program (or, with `size`, a `random.sample`
//! of them drawn from `rng`, which stands in for Python's global `random`)
//! and reduces the outputs with `majority` on the last output field.
//! `forwardAll` returns unreduced outputs; `EnsembleWithReducer` lets a
//! compiled ensemble return a caller-defined result type.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const predict = @import("predict.zig");
const pyrandom = @import("pyrandom.zig");
const unicode = @import("unicode.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Record = value.Record;
const Value = @import("runtime_value.zig").Value;

/// `dspy.evaluate.normalize_text`: NFD, `str.lower`, drop
/// `string.punctuation`, replace the words a/an/the (re's Unicode `\b`)
/// with a space, then `" ".join(s.split())`.
pub fn normalizeText(arena: Allocator, s: []const u8) Allocator.Error![]const u8 {
    const lowered = try unicode.lower(arena, try unicode.nfd(arena, try unicode.decode(arena, s)));
    var kept: std.ArrayList(u21) = .empty;
    try kept.ensureTotalCapacity(arena, lowered.len);
    for (lowered) |cp| {
        if (cp < 0x80 and std.ascii.isPrint(@intCast(cp)) and !std.ascii.isAlphanumeric(@intCast(cp)) and cp != ' ') continue;
        kept.appendAssumeCapacity(cp);
    }
    const text = kept.items;
    var out: std.ArrayList(u21) = .empty;
    try out.ensureTotalCapacity(arena, text.len);
    var i: usize = 0;
    while (i < text.len) {
        // Articles never span whitespace, so drop them word by word.
        while (i < text.len and unicode.isSpace(text[i])) i += 1;
        const start = i;
        while (i < text.len and !unicode.isSpace(text[i])) i += 1;
        if (start == i) break;
        var j = start;
        var pending_space = out.items.len > 0;
        while (j < i) {
            const run_start = j;
            const word_run = unicode.isWord(text[j]);
            while (j < i and unicode.isWord(text[j]) == word_run) j += 1;
            const run = text[run_start..j];
            if (word_run and isArticle(run)) {
                pending_space = out.items.len > 0;
                continue;
            }
            if (pending_space) out.appendAssumeCapacity(' ');
            pending_space = false;
            out.appendSliceAssumeCapacity(run);
        }
    }
    return unicode.encode(arena, out.items);
}

fn isArticle(run: []const u21) bool {
    const articles = [_][]const u21{ &.{'a'}, &.{ 'a', 'n' }, &.{ 't', 'h', 'e' } };
    for (articles) |a| {
        if (std.mem.eql(u21, run, a)) return true;
    }
    return false;
}

/// `dspy.majority(completions, field=field)`: the index of the first
/// completion whose normalized `field` is the most common normalized value
/// (ties go to the value seen first; empty normalizations are ignored
/// unless every one is empty). Non-string scalars keep Python equality.
/// DSPy's winning-NaN quirk is retained: the final equality search finds
/// no match, so it returns the last completion instead.
pub const Error = value.ParseError || error{ UnhashableVote, MissingVoteField, EmptyVotes };

pub const MajorityOptions = struct {
    /// Null or empty selects the last key of the first row.
    field: ?[]const u8 = null,
    normalize: Normalization = .text,
};

pub const Normalization = union(enum) {
    text,
    identity,
    custom: struct {
        context: ?*anyopaque = null,
        run: *const fn (?*anyopaque, Allocator, Value) anyerror!Value,
    },
};

const VoteKey = struct {
    value: Value,
    // Inline scalars have no Python object identity. Only shared Record
    // field storage represents identity; equal NaN bits must not merge.
    identity: ?*const Value,

    fn equal(a: VoteKey, b: VoteKey) value.ParseError!bool {
        if (a.identity != null and a.identity == b.identity) return true;
        // Boxed immutable scalars carry their Python identity across copies.
        if (a.value == .custom and b.value == .custom) {
            if (@import("python_value.zig").sameIdentity(a.value, b.value)) return true;
        }
        return value.pyEqual(a.value, b.value);
    }
};

pub fn majority(arena: Allocator, completions: []const Record, field: []const u8) Error!usize {
    return majorityImpl(arena, completions, field, false, .text);
}

pub fn majorityWithOptions(arena: Allocator, completions: []const Record, options: MajorityOptions) anyerror!usize {
    return majorityImpl(arena, completions, options.field orelse "", true, options.normalize);
}

fn majorityImpl(
    arena: Allocator,
    completions: []const Record,
    requested_field: []const u8,
    comptime allow_custom: bool,
    mode: Normalization,
) (if (allow_custom) anyerror else Error)!usize {
    if (completions.len == 0) return error.EmptyVotes;
    const field = if (requested_field.len != 0) requested_field else field: {
        const names = completions[0].keys();
        if (names.len == 0) return error.MissingVoteField;
        break :field names[names.len - 1];
    };
    const keys = try arena.alloc(VoteKey, completions.len);
    for (completions, keys) |*c, *k| {
        const original = c.getPtr(field) orelse return error.MissingVoteField;
        if (!allow_custom) {
            k.* = .{ .value = try normalizedKey(arena, original.*), .identity = original };
        } else {
            k.* = switch (mode) {
                .text => .{ .value = try normalizedKey(arena, original.*), .identity = original },
                .identity => .{ .value = original.*, .identity = original },
                .custom => |callback| .{ .value = try callback.run(callback.context, arena, original.*), .identity = null },
            };
        }
    }
    for (keys) |k| try requireHashable(k.value);
    var any_kept = false;
    for (keys) |k| any_kept = any_kept or k.value != .null;
    var best: ?usize = null;
    var best_count: usize = 0;
    for (keys, 0..) |k, i| {
        if (any_kept and k.value == .null) continue;
        if (try firstIndex(keys[0..i], k, any_kept) != null) continue;
        var count: usize = 0;
        for (keys, 0..) |other, j| {
            if (i == j or try k.equal(other)) count += 1;
        }
        if (count > best_count) {
            best = i;
            best_count = count;
        }
    }
    // best already names the first occurrence. DSPy's final search uses
    // equality alone: a non-reflexive winning NaN falls through to the last row.
    const winner = best.?;
    if (allow_custom and mode == .custom) {
        const callback = mode.custom;
        for (completions, 0..) |*row, i| {
            const original = row.getPtr(field) orelse return error.MissingVoteField;
            const normalized = try callback.run(callback.context, arena, original.*);
            if (try value.pyEqual(normalized, keys[winner].value)) return i;
        }
        return completions.len - 1;
    }
    return if (try value.pyEqual(keys[winner].value, keys[winner].value)) winner else completions.len - 1;
}

fn firstIndex(keys: []const VoteKey, k: VoteKey, skip_null: bool) value.ParseError!?usize {
    for (keys, 0..) |other, i| {
        if (skip_null and other.value == .null) continue;
        if (try k.equal(other)) return i;
    }
    return null;
}

/// `default_normalize`: text normalization only applies to strings.
fn normalizedKey(arena: Allocator, v: Value) Error!Value {
    return switch (v) {
        .string => |s| blk: {
            const n = try normalizeText(arena, s);
            break :blk if (n.len == 0) .null else .{ .string = n };
        },
        else => v,
    };
}

fn requireHashable(v: Value) error{UnhashableVote}!void {
    switch (v) {
        .array, .object => return error.UnhashableVote,
        .custom => |custom| if (!custom.vtable.hashable) return error.UnhashableVote,
        else => {},
    }
}

test "majority re-runs user normalization in row order to find the winning output" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var rows: [3]Record = @splat(.empty);
    for (&rows, [_][]const u8{ "The cat", "dog", "cat" }) |*row, text| {
        try row.put(arena, "answer", .{ .string = text });
    }
    const Callback = struct {
        fn normalize(ptr: ?*anyopaque, a: Allocator, v: Value) anyerror!Value {
            const calls: *usize = @ptrCast(@alignCast(ptr.?));
            calls.* += 1;
            // The first row helps decide the winner, but drops out on the
            // second pass. DSPy searches again, rather than reusing that row.
            if (calls.* == 4) return .null;
            return normalizedKey(a, v);
        }
    };
    var calls: usize = 0;
    try std.testing.expectEqual(@as(usize, 2), try majorityWithOptions(arena, &rows, .{
        .normalize = .{ .custom = .{ .context = &calls, .run = Callback.normalize } },
    }));
    try std.testing.expectEqual(@as(usize, 6), calls);
    try std.testing.expectEqual(@as(usize, 0), try majorityWithOptions(arena, &rows, .{ .normalize = .identity }));
    try std.testing.expectError(error.EmptyVotes, majorityWithOptions(arena, &.{}, .{}));
}

test "majority rejects Reasoning but accepts frozen File and BoundingBox models" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const Reasoning = @import("signature.zig").Reasoning;
    const File = @import("file.zig").File;
    const Box = @import("bounding_box.zig").BoundingBox;
    const models = [_]Value{
        try (Reasoning{ .content = "vote" }).toRuntime(arena),
        try (try File.fromFileId("file-vote", null)).toRuntime(arena),
        try (try Box.init(0, 1, 0, 1)).toRuntime(arena),
    };
    var rows: [2]Record = @splat(.empty);
    for (models, 0..) |model, index| {
        try rows[0].put(arena, "answer", model);
        try rows[1].put(arena, "answer", model);
        if (index == 0) {
            try std.testing.expectError(error.UnhashableVote, majority(arena, &rows, "answer"));
        } else {
            try std.testing.expectEqual(@as(usize, 0), try majority(arena, &rows, "answer"));
        }
    }
}

test "majority distinguishes text from numbers and groups Python numeric equals" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var records: [3]Record = @splat(.empty);
    try records[0].put(arena, "answer", .{ .string = "1" });
    try records[1].put(arena, "answer", .{ .integer = 1 });
    try records[2].put(arena, "answer", .{ .bool = true });
    try std.testing.expectEqual(@as(usize, 1), try majority(arena, &records, "answer"));
    try records[0].put(arena, "answer", .{ .integer = 9_007_199_254_740_993 });
    try records[1].put(arena, "answer", .{ .integer = 9_007_199_254_740_992 });
    try records[2].put(arena, "answer", .{ .float = 9_007_199_254_740_992.0 });
    try std.testing.expectEqual(@as(usize, 1), try majority(arena, &records, "answer"));
}

test "majority matches DSPy NaN result fallback and ordinary first ties" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const nan: Value = .{ .float = std.math.nan(f64) };
    const seven: Value = .{ .integer = 7 };
    const cases = [_]struct { votes: []const Value, winner: usize }{
        .{ .votes = &.{nan}, .winner = 0 },
        .{ .votes = &.{ nan, nan }, .winner = 1 },
        .{ .votes = &.{ nan, .null }, .winner = 1 },
        .{ .votes = &.{ nan, nan, .null }, .winner = 2 },
        .{ .votes = &.{ nan, seven }, .winner = 1 },
        .{ .votes = &.{ seven, nan }, .winner = 0 },
        .{ .votes = &.{ nan, seven, seven }, .winner = 1 },
        .{ .votes = &.{ nan, nan, seven }, .winner = 2 },
        .{ .votes = &.{ .null, nan, seven }, .winner = 2 },
        .{ .votes = &.{ .null, .{ .string = "the" }, .null }, .winner = 0 },
        .{ .votes = &.{ .{ .integer = 8 }, seven, seven, .{ .integer = 8 } }, .winner = 0 },
    };
    for (cases) |case| {
        const records = try arena.alloc(Record, case.votes.len);
        for (records, case.votes, 0..) |*record, vote, i| {
            record.* = .empty;
            try record.put(arena, "answer", vote);
            try record.put(arena, "marker", .{ .integer = @intCast(i) });
        }
        const winner = try majority(arena, records, "answer");
        try std.testing.expectEqual(case.winner, winner);
        try std.testing.expectEqual(@as(i64, @intCast(case.winner)), records[winner].get("marker").?.integer);
    }
}

test "majority groups one boxed NaN across records but not separately minted NaNs" {
    const python = @import("python_value.zig");
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const shared = try python.float(arena, std.math.nan(f64));
    const seven: Value = .{ .integer = 7 };
    const eight: Value = .{ .integer = 8 };
    // Three votes for one NaN object beat two sevens; the equality-only
    // result search then misses the NaN and returns the last completion.
    const same = [_]Value{ seven, seven, shared, shared, shared, eight };
    // Distinct NaN objects with identical bits each vote once: seven wins.
    const distinct = [_]Value{
        seven,                                      seven,
        try python.float(arena, std.math.nan(f64)), try python.float(arena, std.math.nan(f64)),
        try python.float(arena, std.math.nan(f64)), eight,
    };
    const cases = [_]struct { votes: []const Value, winner: usize }{
        .{ .votes = &same, .winner = 5 },
        .{ .votes = &distinct, .winner = 0 },
    };
    for (cases) |case| {
        const records = try arena.alloc(Record, case.votes.len);
        for (records, case.votes) |*record, vote| {
            record.* = .empty;
            try record.put(arena, "answer", vote);
        }
        try std.testing.expectEqual(case.winner, try majority(arena, records, "answer"));
    }
    // A boxed finite float still votes with its numeric equals.
    const records = try arena.alloc(Record, 3);
    for (records, [_]Value{ eight, try python.float(arena, 7), seven }) |*record, vote| {
        record.* = .empty;
        try record.put(arena, "answer", vote);
    }
    try std.testing.expectEqual(@as(usize, 1), try majority(arena, records, "answer"));
}

test "majority counts aliased NaN identity but not separate equal NaN bits" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var records: [4]Record = @splat(.empty);
    for (&records, 0..) |*record, i| {
        const answer: Value = switch (i) {
            0 => .{ .integer = 7 },
            3 => .{ .integer = 8 },
            else => .{ .float = std.math.nan(f64) },
        };
        try record.put(arena, "answer", answer);
        try record.put(arena, "marker", .{ .integer = @intCast(i) });
    }
    // Separate NaNs each receive one vote, so the first ordinary key wins.
    var winner = try majority(arena, &records, "answer");
    try std.testing.expectEqual(@as(usize, 0), winner);
    try std.testing.expectEqual(@as(i64, 0), records[winner].get("marker").?.integer);
    // Shared field storage is representable identity: the NaN now wins the
    // count, then DSPy's equality-only result search falls through to row 3.
    records[2] = records[1];
    winner = try majority(arena, &records, "answer");
    try std.testing.expectEqual(@as(usize, 3), winner);
    try std.testing.expectEqual(@as(i64, 3), records[winner].get("marker").?.integer);
}

pub fn Ensemble(comptime P: type, comptime n: usize) type {
    const ProgramError = @typeInfo(@typeInfo(@TypeOf(P.forward)).@"fn".return_type.?).error_union.error_set;
    return struct {
        const Self = @This();
        pub const Input = P.Input;
        pub const Output = P.Output;
        pub const ForwardError = ProgramError || Allocator.Error || error{ InvalidSampleSize, MissingEnsembleRng };

        programs: [n]P,
        /// Null and zero run every program without advancing the RNG.
        /// A positive size samples without replacement.
        size: ?i128 = null,
        /// Required only for positive `size`.
        rng: ?*pyrandom.Random = null,
        compiled: bool = false,

        /// `reduce_fn=dspy.majority`.
        pub fn forward(self: *const Self, ctx: Context, input: Input) (ForwardError || predict.Error || Error || error{EmptyEnsemble})!Output {
            const output_fields = @typeInfo(Output).@"struct".field_names;
            comptime assert(output_fields.len > 0);
            const outputs = try self.forwardAll(ctx, input);
            if (outputs.len == 0) return error.EmptyEnsemble;
            const records = try ctx.arena.alloc(Record, outputs.len);
            for (outputs, records) |o, *r| r.* = try value.toRecord(ctx.arena, Output, o);
            return outputs[try majority(ctx.arena, records, output_fields[output_fields.len - 1])];
        }

        /// `reduce_fn=None`: every sampled program's output, in sample order.
        pub fn forwardAll(self: *const Self, ctx: Context, input: Input) ForwardError![]const Output {
            const k = std.math.cast(usize, self.size orelse 0) orelse return error.InvalidSampleSize;
            if (k == 0) {
                const outputs = try ctx.arena.alloc(Output, n);
                for (&self.programs, outputs) |*program, *out| out.* = try program.forward(ctx, input);
                return outputs;
            }
            if (n == 0 or k > n) return error.InvalidSampleSize;
            const rng = self.rng orelse return error.MissingEnsembleRng;
            const indices = try ctx.arena.alloc(usize, k);
            var scratch: [n]usize = undefined;
            const outputs = try ctx.arena.alloc(Output, k);
            rng.sampleIndices(n, indices, &scratch);
            for (indices, outputs) |idx, *out| out.* = try self.programs[idx].forward(ctx, input);
            return outputs;
        }
    };
}

/// `reduce_fn` receives all outputs (even an empty list), in execution order.
pub fn EnsembleWithReducer(comptime P: type, comptime n: usize, comptime reduce: anytype) type {
    const Members = Ensemble(P, n);
    const ReduceResult = @typeInfo(@TypeOf(reduce)).@"fn".return_type.?;
    const reduce_info = @typeInfo(ReduceResult);
    const fallible = reduce_info == .error_union;
    return struct {
        pub const Input = P.Input;
        pub const Output = if (fallible) reduce_info.error_union.payload else ReduceResult;
        pub const ForwardError = Members.ForwardError || if (fallible) reduce_info.error_union.error_set else error{};

        ensemble: Members,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) ForwardError!Output {
            const outputs = try self.ensemble.forwardAll(ctx, input);
            return reduce(ctx, outputs);
        }
    };
}
