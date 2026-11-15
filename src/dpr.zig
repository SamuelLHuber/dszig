//! DSPy's `DPR_normalize`: NFD, then SimpleTokenizer's
//! `([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])`, then `str.lower` on each token.
//! The character classes in dpr_unicode.zig are the pinned regex package's
//! actual matches, which use newer Unicode data than CPython's NFD/lower.
//!
//! Upstream (DSPy 3.5.0): dspy/dsp/utils/dpr.py

const std = @import("std");
const Allocator = std.mem.Allocator;

const tables = @import("dpr_unicode.zig");
const unicode = @import("unicode.zig");

const Class = enum { excluded, word, other };

fn inRanges(table: []const tables.Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = table.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < table[mid].lo) {
            hi = mid;
        } else if (cp > table[mid].hi) {
            lo = mid + 1;
        } else return true;
    }
    return false;
}

fn classify(cp: u21) Class {
    if (inRanges(&tables.word, cp)) return .word;
    if (inRanges(&tables.other, cp)) return .other;
    return .excluded;
}

/// Lowercasing is per token, so final sigma only sees that token's context.
fn token(arena: Allocator, cps: []const u21) Allocator.Error![]const u8 {
    return unicode.encode(arena, try unicode.lower(arena, cps));
}

pub fn normalize(arena: Allocator, text: []const u8) Allocator.Error![]const []const u8 {
    const cps = try unicode.nfd(arena, try unicode.decode(arena, text));
    var tokens: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < cps.len) {
        switch (classify(cps[i])) {
            .excluded => i += 1,
            .other => {
                try tokens.append(arena, try token(arena, cps[i..][0..1]));
                i += 1;
            },
            .word => {
                const start = i;
                i += 1;
                while (i < cps.len and classify(cps[i]) == .word) i += 1;
                try tokens.append(arena, try token(arena, cps[start..i]));
            },
        }
    }
    return tokens.items;
}

test "DPR_normalize token boundaries" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const Case = struct { text: []const u8, tokens: []const []const u8 };
    // Expected tokens observed from pinned DSPy 3.5 DPR_normalize.
    const cases = [_]Case{
        .{ .text = "", .tokens = &.{} },
        .{ .text = "Café", .tokens = &.{"cafe\u{301}"} },
        .{ .text = "A\u{301}b", .tokens = &.{"a\u{301}b"} },
        .{ .text = "ΟΔΥΣΣΕΥΣ", .tokens = &.{"οδυσσευς"} },
        // Whole-text lowering would keep medial σ before U+200B and Β.
        .{ .text = "ΑΣ\u{200b}Β", .tokens = &.{ "ας", "β" } },
        .{ .text = "ΑΣ-Β", .tokens = &.{ "ας", "-", "β" } },
        .{ .text = "x--y€", .tokens = &.{ "x", "-", "-", "y", "€" } },
        .{ .text = "١٢abc_3", .tokens = &.{ "١٢abc", "_", "3" } },
        .{ .text = "a\u{a0}b\u{2028}c\td", .tokens = &.{ "a", "b", "c", "d" } },
        .{ .text = "a\x00b\u{ad}c\u{200d}d", .tokens = &.{ "a", "b", "c", "d" } },
        .{ .text = "x\u{378}y", .tokens = &.{ "x", "y" } },
        // CPython 3.11 calls both Cn; pinned regex makes them word and other.
        .{ .text = "x\u{897}y", .tokens = &.{"x\u{897}y"} },
        .{ .text = "x\u{1fae9}y", .tokens = &.{ "x", "\u{1fae9}", "y" } },
        .{ .text = "İ", .tokens = &.{"i\u{307}"} },
        .{ .text = "한글", .tokens = &.{"\u{1112}\u{1161}\u{11ab}\u{1100}\u{1173}\u{11af}"} },
        // Lone surrogates are category C.
        .{ .text = "\xed\xa0\x80x", .tokens = &.{"x"} },
    };
    for (cases) |case| {
        const got = try normalize(arena, case.text);
        try std.testing.expectEqual(case.tokens.len, got.len);
        for (case.tokens, got) |want, actual| try std.testing.expectEqualStrings(want, actual);
    }
}
