//! The Unicode steps of DSPy's `normalize_text`, exactly as CPython 3.11
//! performs them: `unicodedata.normalize("NFD", ...)`, `str.lower` (with the
//! final-sigma rule), re's `\w` and `str.split` whitespace. The tables in
//! unicode_tables.zig are generated from Python by gen_unicode.py.

const std = @import("std");
const Allocator = std.mem.Allocator;

const tables = @import("unicode_tables.zig");

const hangul_s = 0xAC00;
const hangul_l = 0x1100;
const hangul_v = 0x1161;
const hangul_t = 0x11A7;
const hangul_t_count = 28;
const hangul_n_count = 21 * hangul_t_count;
const hangul_s_count = 19 * hangul_n_count;
const capital_sigma = 0x3A3;

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

fn mapped(table: []const tables.Mapping, pool: []const u21, cp: u21) ?[]const u21 {
    var lo: usize = 0;
    var hi: usize = table.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < table[mid].cp) {
            hi = mid;
        } else if (cp > table[mid].cp) {
            lo = mid + 1;
        } else return pool[table[mid].start..][0..table[mid].len];
    }
    return null;
}

pub fn combiningClass(cp: u21) u8 {
    var lo: usize = 0;
    var hi: usize = tables.ccc.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < tables.ccc[mid].lo) {
            hi = mid;
        } else if (cp > tables.ccc[mid].hi) {
            lo = mid + 1;
        } else return tables.ccc[mid].ccc;
    }
    return 0;
}

pub fn isWord(cp: u21) bool {
    return inRanges(&tables.word, cp);
}

pub fn isSpace(cp: u21) bool {
    return inRanges(&tables.space, cp);
}

/// Rust/Pydantic White_Space omits U+001C..U+001F from Python whitespace.
/// Invalid UTF-8/WTF-8 ends trimming without replacing the original bytes.
pub fn strip(text: []const u8, pydantic: bool) []const u8 {
    var start: usize = 0;
    var end = text.len;
    while (start < end) {
        const len = std.unicode.utf8ByteSequenceLength(text[start]) catch break;
        if (len > end - start) break;
        const cp = std.unicode.utf8Decode(text[start..][0..len]) catch break;
        if (!isSpace(cp) or (pydantic and cp >= 0x1c and cp <= 0x1f)) break;
        start += len;
    }
    while (end > start) {
        var first = end - 1;
        while (first > start and text[first] & 0xc0 == 0x80) first -= 1;
        const cp = std.unicode.utf8Decode(text[first..end]) catch break;
        if (!isSpace(cp) or (pydantic and cp >= 0x1c and cp <= 0x1f)) break;
        end = first;
    }
    return text[start..end];
}

/// UTF-8 to code points; invalid bytes become U+FFFD.
pub fn decode(arena: Allocator, s: []const u8) Allocator.Error![]u21 {
    var out: std.ArrayList(u21) = .empty;
    try out.ensureTotalCapacity(arena, s.len);
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch {
            out.appendAssumeCapacity(0xFFFD);
            i += 1;
            continue;
        };
        if (i + n > s.len) {
            out.appendAssumeCapacity(0xFFFD);
            i += 1;
            continue;
        }
        const cp = std.unicode.utf8Decode(s[i..][0..n]) catch {
            out.appendAssumeCapacity(0xFFFD);
            i += 1;
            continue;
        };
        out.appendAssumeCapacity(cp);
        i += n;
    }
    return out.items;
}

pub fn encode(arena: Allocator, cps: []const u21) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(arena, cps.len * 4);
    for (cps) |cp| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &buf) catch std.unicode.utf8Encode(0xFFFD, &buf) catch unreachable;
        out.appendSliceAssumeCapacity(buf[0..n]);
    }
    return out.items;
}

/// `unicodedata.normalize("NFD", s)`.
pub fn nfd(arena: Allocator, cps: []const u21) Allocator.Error![]u21 {
    var out: std.ArrayList(u21) = .empty;
    try out.ensureTotalCapacity(arena, cps.len);
    for (cps) |cp| {
        if (cp >= hangul_s and cp < hangul_s + hangul_s_count) {
            const index = cp - hangul_s;
            try out.append(arena, hangul_l + index / hangul_n_count);
            try out.append(arena, hangul_v + (index % hangul_n_count) / hangul_t_count);
            if (index % hangul_t_count != 0) try out.append(arena, hangul_t + index % hangul_t_count);
        } else if (mapped(&tables.nfd, &tables.nfd_pool, cp)) |d| {
            try out.appendSlice(arena, d);
        } else {
            try out.append(arena, cp);
        }
    }
    // Canonical ordering: stable sort each run of non-starters by class.
    const items = out.items;
    var i: usize = 1;
    while (i < items.len) : (i += 1) {
        const c = combiningClass(items[i]);
        if (c == 0) continue;
        var j = i;
        while (j > 0) : (j -= 1) {
            const prev = combiningClass(items[j - 1]);
            if (prev == 0 or prev <= c) break;
            std.mem.swap(u21, &items[j], &items[j - 1]);
        }
    }
    return items;
}

/// `str.lower`, including CPython's final-sigma rule.
pub fn lower(arena: Allocator, cps: []const u21) Allocator.Error![]u21 {
    var out: std.ArrayList(u21) = .empty;
    try out.ensureTotalCapacity(arena, cps.len);
    for (cps, 0..) |cp, i| {
        if (cp == capital_sigma) {
            try out.append(arena, if (finalSigma(cps, i)) 0x3C2 else 0x3C3);
        } else if (mapped(&tables.lower, &tables.lower_pool, cp)) |l| {
            try out.appendSlice(arena, l);
        } else {
            try out.append(arena, cp);
        }
    }
    return out.items;
}

/// Compile-time `str.lower` for static type metadata, without an allocator.
/// Shares the runtime mapping and context rule, without normalizing the text.
pub fn lowerComptime(comptime text: []const u8) []const u8 {
    return comptime result: {
        @setEvalBranchQuota(1_000_000);
        var codepoints: [text.len]u21 = undefined;
        var count: usize = 0;
        var iterator = (std.unicode.Utf8View.init(text) catch @compileError("dszig: lowercase text must be valid UTF-8")).iterator();
        while (iterator.nextCodepoint()) |cp| {
            codepoints[count] = cp;
            count += 1;
        }
        const cps = codepoints[0..count];
        var out: []const u8 = "";
        for (cps, 0..) |cp, i| {
            const lowered: []const u21 = if (cp == capital_sigma)
                &.{if (finalSigma(cps, i)) 0x3c2 else 0x3c3}
            else
                mapped(&tables.lower, &tables.lower_pool, cp) orelse &.{cp};
            for (lowered) |lower_cp| {
                var buffer: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(lower_cp, &buffer) catch unreachable;
                const encoded = buffer[0..len].*;
                out = out ++ encoded;
            }
        }
        break :result out;
    };
}

fn finalSigma(cps: []const u21, i: usize) bool {
    var j = i;
    while (j > 0 and inRanges(&tables.case_ignorable, cps[j - 1])) j -= 1;
    if (j == 0 or !inRanges(&tables.cased, cps[j - 1])) return false;
    j = i + 1;
    while (j < cps.len and inRanges(&tables.case_ignorable, cps[j])) j += 1;
    return j == cps.len or !inRanges(&tables.cased, cps[j]);
}

test "compile-time lower shares Unicode expansion and final-sigma rules without NFD" {
    inline for (.{
        .{ "JAVA", "java" },
        .{ "Café", "café" },
        .{ "İΣ", "i\u{307}ς" },
        .{ "ΣAΣ", "σaς" },
        .{ "AΣ\u{301}B", "aσ\u{301}b" },
        .{ "AΣ\u{301}", "aς\u{301}" },
    }) |case| {
        try std.testing.expectEqualStrings(case[1], lowerComptime(case[0]));
    }
}

test "NFD, lower and final sigma" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cases = [_][2][]const u8{
        .{ "Café", "cafe\u{301}" },
        .{ "ΟΔΥΣΣΕΥΣ", "οδυσσευς" },
        .{ "İ", "i\u{307}" },
        .{ "한", "\u{1112}\u{1161}\u{11ab}" },
        .{ "x\u{301}\u{323}", "x\u{323}\u{301}" },
    };
    for (cases) |c| {
        const got = try encode(arena, try lower(arena, try nfd(arena, try decode(arena, c[0]))));
        try std.testing.expectEqualStrings(c[1], got);
    }
}
