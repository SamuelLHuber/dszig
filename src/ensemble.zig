//! `dspy.Ensemble` and `dspy.majority` (DSPy 3.4.1).
//!
//! `Ensemble(P, n)` runs every program (or, with `size`, a `random.sample`
//! of them drawn from `rng`, which stands in for Python's global `random`)
//! and reduces the outputs with `majority` on the last output field.
//! `normalize_text` is exact for ASCII text; DSZig does not apply Unicode
//! NFD normalization or Unicode lowercasing to non-ASCII bytes.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Record = value.Record;
const Value = std.json.Value;

/// `dspy.evaluate.normalize_text`: lowercase, drop ASCII punctuation, drop
/// the words a/an/the, collapse whitespace.
pub fn normalizeText(arena: Allocator, s: []const u8) Allocator.Error![]const u8 {
    var kept: std.ArrayList(u8) = .empty;
    try kept.ensureTotalCapacity(arena, s.len);
    for (s) |c| {
        if (c < 0x80 and std.ascii.isPrint(c) and !std.ascii.isAlphanumeric(c) and c != ' ') continue;
        kept.appendAssumeCapacity(std.ascii.toLower(c));
    }
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(arena, kept.items.len);
    var i: usize = 0;
    const text = kept.items;
    while (i < text.len) {
        // Split into whitespace-separated words, then drop article word runs.
        while (i < text.len and isSpace(text[i])) i += 1;
        const start = i;
        while (i < text.len and !isSpace(text[i])) i += 1;
        if (start == i) break;
        const word = try dropArticles(arena, text[start..i]);
        var it = std.mem.tokenizeAny(u8, word, " ");
        while (it.next()) |part| {
            if (out.items.len > 0) out.appendAssumeCapacity(' ');
            out.appendSliceAssumeCapacity(part);
        }
    }
    return out.items;
}

fn isSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r') or (c >= 0x1c and c <= 0x1f);
}

fn isWord(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c >= 0x80;
}

/// `re.sub(r"\b(a|an|the)\b", " ", word)` within one whitespace-free word.
fn dropArticles(arena: Allocator, word: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(arena, word.len);
    var i: usize = 0;
    while (i < word.len) {
        const start = i;
        const word_run = isWord(word[i]);
        while (i < word.len and isWord(word[i]) == word_run) i += 1;
        const run = word[start..i];
        const article = word_run and (std.mem.eql(u8, run, "a") or std.mem.eql(u8, run, "an") or std.mem.eql(u8, run, "the"));
        if (article) out.appendAssumeCapacity(' ') else out.appendSliceAssumeCapacity(run);
    }
    return out.items;
}

/// `dspy.majority(completions, field=field)`: the index of the first
/// completion whose normalized `field` is the most common normalized value
/// (ties go to the value seen first; empty normalizations are ignored
/// unless every one is empty). Non-string values compare by their JSON.
pub fn majority(arena: Allocator, completions: []const Record, field: []const u8) Allocator.Error!usize {
    assert(completions.len > 0);
    const keys = try arena.alloc(?[]const u8, completions.len);
    for (completions, keys) |*c, *k| k.* = try normalizedKey(arena, c.get(field) orelse .null);
    var any_kept = false;
    for (keys) |k| any_kept = any_kept or k != null;
    var best: ?usize = null;
    var best_count: usize = 0;
    for (keys, 0..) |k, i| {
        if (any_kept and k == null) continue;
        if (firstIndex(keys[0..i], k, any_kept) != null) continue;
        var count: usize = 0;
        for (keys) |other| {
            if (eqlKey(k, other)) count += 1;
        }
        if (count > best_count) {
            best = i;
            best_count = count;
        }
    }
    return best.?;
}

fn firstIndex(keys: []const ?[]const u8, k: ?[]const u8, skip_null: bool) ?usize {
    for (keys, 0..) |other, i| {
        if (skip_null and other == null) continue;
        if (eqlKey(k, other)) return i;
    }
    return null;
}

fn eqlKey(a: ?[]const u8, b: ?[]const u8) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.mem.eql(u8, a.?, b.?);
}

/// `default_normalize`: `normalize_text(s) or None`.
fn normalizedKey(arena: Allocator, v: Value) Allocator.Error!?[]const u8 {
    if (v == .string) {
        const n = try normalizeText(arena, v.string);
        return if (n.len == 0) null else n;
    }
    var aw: std.Io.Writer.Allocating = .init(arena);
    pyjson.writeJson(&aw.writer, v, .compact) catch return error.OutOfMemory;
    return aw.written();
}

pub fn Ensemble(comptime P: type, comptime n: usize) type {
    comptime assert(n > 0);
    const output_fields = @typeInfo(P.Output).@"struct".field_names;
    comptime assert(output_fields.len > 0);
    return struct {
        const Self = @This();
        pub const Input = P.Input;
        pub const Output = P.Output;

        programs: [n]P,
        /// Run a random sample of this many programs per call.
        size: ?usize = null,
        /// Required with `size`.
        rng: ?*pyrandom.Random = null,
        compiled: bool = false,

        /// `reduce_fn=dspy.majority`.
        pub fn forward(self: *const Self, ctx: Context, input: Input) predict.Error!Output {
            const outputs = try self.forwardAll(ctx, input);
            const records = try ctx.arena.alloc(Record, outputs.len);
            for (outputs, records) |o, *r| r.* = try value.toRecord(ctx.arena, Output, o);
            return outputs[try majority(ctx.arena, records, output_fields[output_fields.len - 1])];
        }

        /// `reduce_fn=None`: every sampled program's output, in sample order.
        pub fn forwardAll(self: *const Self, ctx: Context, input: Input) predict.Error![]const Output {
            const k = self.size orelse n;
            assert(k > 0 and k <= n);
            const indices = try ctx.arena.alloc(usize, k);
            if (self.size == null) {
                for (indices, 0..) |*idx, i| idx.* = i;
            } else {
                var scratch: [n]usize = undefined;
                self.rng.?.sampleIndices(n, indices, &scratch);
            }
            const outputs = try ctx.arena.alloc(Output, k);
            for (indices, outputs) |idx, *o| o.* = try self.programs[idx].forward(ctx, input);
            return outputs;
        }
    };
}
