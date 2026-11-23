//! Native ports of the capability patterns in the bundled price catalogue.
//!
//! The catalogue's `fallback_generalizations` rules name Python `re` patterns.
//! There is no general regular-expression engine here: every pattern text the
//! bundled snapshot (LiteLLM 1.108.3) carries is ported by hand, and identify()
//! recognises a pattern only by its exact text. Any other text is unknown, so
//! the catalogue refuses it instead of guessing what it matches.
//!
//! Each port answers what `re.search(pattern, model) is not None` answers for
//! a Python `str`: `\d` is every Unicode decimal digit, the bracketed classes
//! are ASCII, `.` stops at a line feed and `$` also matches before one final
//! line feed. Matching borrows the model text, allocates nothing and does not
//! recurse; every loop advances through the text or through a fixed table.

const std = @import("std");
const assert = std.debug.assert;
const unicode = @import("unicode.zig");

pub const Pattern = enum(u8) {
    bedrock_claude_prefix,
    claude_bare_id,
    claude_family,
    claude_version_from_4_6,
    claude_version_4_6,
    claude_fable_or_mythos,
    claude_version_from_4_8,
    claude_version_from_4_5,
    wandb_prefix,
    openai_reasoning_family,
    gemini_chat,
};

/// Exact pattern texts, indexed by @intFromEnum(Pattern). Two of the twelve
/// bundled rules share one text, so eleven distinct patterns remain.
const sources = [_][]const u8{
    "^(?:[a-z-]+\\.)?anthropic\\.claude-",
    "^claude-[a-z]+-\\d+(?:[-.]\\d+)?(?:-\\d{8})?$",
    "claude-[a-z]+-\\d+(?:[-.]\\d+)?(?:-\\d{8})?",
    "claude-[a-z]+-(?:4[-._](?:[6-9]|[1-9]\\d)(?!\\d)|[5-9](?!\\d)(?:[-._]\\d{1,2}(?!\\d))?)",
    "claude-[a-z]+-4[-._]6(?!\\d)",
    "claude-(?:fable|mythos)-",
    "claude-[a-z]+-(?:4[-._](?:[89]|[1-9]\\d)(?!\\d)|[5-9](?!\\d)(?:[-._]\\d{1,2}(?!\\d))?)",
    "claude-[a-z]+-(?:4[-._](?:[5-9]|[1-9]\\d)(?!\\d)|[5-9](?!\\d)(?:[-._]\\d{1,2}(?!\\d))?)",
    "^wandb/",
    "^(?!.*search-api)(?:[a-z0-9_.-]+/)*(?:ft:)?(?:o[1-9]\\d*(?![a-z0-9])|gpt-[5-9](?:\\.\\d+)?" ++
        "(?![0-9.])|(?:gpt-\\d+(?:\\.\\d+)?(?:-[a-z0-9]+)*-)?(?:codex|deep-research|chat-latest)" ++
        "(?![a-z0-9]))",
    "gemini-(?!.*(?:-tts|-image|-live|-audio|-embedding|-computer-use|-robotics|-transcribe" ++
        "|-translate))(?:2[.-][5-9]|[3-9](?:[.-]\\d{1,2})?)-(?:pro|flash)(?:-lite)?(?![a-z])",
};

comptime {
    assert(sources.len == std.enums.values(Pattern).len);
}

/// The pattern text exactly as the catalogue spells it.
pub fn source(pattern: Pattern) []const u8 {
    return sources[@backingInt(pattern)];
}

/// Recognises a catalogue pattern by its exact text, or null when unknown.
pub fn identify(text: []const u8) ?Pattern {
    for (sources, 0..) |known, index| {
        if (std.mem.eql(u8, known, text)) return @fromBackingInt(@intCast(index));
    }
    return null;
}

/// `re.search(source(pattern), model) is not None`.
pub fn search(pattern: Pattern, model: []const u8) bool {
    return switch (pattern) {
        .bedrock_claude_prefix => bedrockClaudePrefix(model),
        .claude_bare_id => claudeBareId(model),
        .claude_family => claudeFamily(model),
        .claude_version_from_4_6 => claudeVersion(model, '6'),
        .claude_version_4_6 => claudeVersionExact(model),
        .claude_fable_or_mythos => std.mem.indexOf(u8, model, "claude-fable-") != null or
            std.mem.indexOf(u8, model, "claude-mythos-") != null,
        .claude_version_from_4_8 => claudeVersion(model, '8'),
        .claude_version_from_4_5 => claudeVersion(model, '5'),
        .wandb_prefix => std.mem.startsWith(u8, model, "wandb/"),
        .openai_reasoning_family => openaiReasoningFamily(model),
        .gemini_chat => geminiChat(model),
    };
}

/// Byte length of the Unicode decimal digit starting at `index`, or zero.
/// Malformed UTF-8 and lone surrogates are not digits.
fn digitBytes(text: []const u8, index: usize) usize {
    if (index >= text.len) return 0;
    const first = text[index];
    if (first < 0x80) return @intFromBool(std.ascii.isDigit(first));
    const length = std.unicode.utf8ByteSequenceLength(first) catch return 0;
    if (length > text.len - index) return 0;
    const scalar = std.unicode.utf8Decode(text[index..][0..length]) catch return 0;
    return if (unicode.decimalDigit(scalar) == null) 0 else length;
}

/// End of the longest `\d*` run starting at `index`.
fn digitsEnd(text: []const u8, index: usize) usize {
    var end = index;
    while (true) {
        const length = digitBytes(text, end);
        if (length == 0) return end;
        end += length;
    }
}

fn byteAt(text: []const u8, index: usize) ?u8 {
    return if (index < text.len) text[index] else null;
}

fn isLower(byte: ?u8) bool {
    const b = byte orelse return false;
    return b >= 'a' and b <= 'z';
}

fn isLowerOrDigit(byte: ?u8) bool {
    const b = byte orelse return false;
    return (b >= 'a' and b <= 'z') or (b >= '0' and b <= '9');
}

fn inRange(byte: ?u8, low: u8, high: u8) bool {
    const b = byte orelse return false;
    return b >= low and b <= high;
}

fn isOneOf(byte: ?u8, comptime set: []const u8) bool {
    const b = byte orelse return false;
    return std.mem.indexOfScalar(u8, set, b) != null;
}

fn startsAt(text: []const u8, index: usize, prefix: []const u8) bool {
    if (index > text.len) return false;
    return std.mem.startsWith(u8, text[index..], prefix);
}

/// `^(?:[a-z-]+\.)?anthropic\.claude-`. The dot is outside the class, so the
/// optional geo segment is exactly the leading class run.
fn bedrockClaudePrefix(model: []const u8) bool {
    const stem = "anthropic.claude-";
    if (std.mem.startsWith(u8, model, stem)) return true;
    var run: usize = 0;
    while (run < model.len) : (run += 1) {
        const b = model[run];
        if (!((b >= 'a' and b <= 'z') or b == '-')) break;
    }
    if (run == 0) return false;
    if (byteAt(model, run) != '.') return false;
    return startsAt(model, run + 1, stem);
}

/// After `claude-[a-z]+-` at `start`: the index following that hyphen, or null.
/// The hyphen is outside the class, so the family run is the longest one.
fn claudeFamilyEnd(model: []const u8, start: usize) ?usize {
    const stem = "claude-";
    assert(startsAt(model, start, stem));
    var index = start + stem.len;
    const family = index;
    while (isLower(byteAt(model, index))) index += 1;
    if (index == family) return null;
    if (byteAt(model, index) != '-') return null;
    return index + 1;
}

/// `$`: the end of the text, or just before its one final line feed.
fn atEnd(text: []const u8, index: usize) bool {
    if (index == text.len) return true;
    return index + 1 == text.len and text[index] == '\n';
}

/// `(?:-\d{8})?$` at `index`.
fn claudeDateThenEnd(model: []const u8, index: usize) bool {
    if (atEnd(model, index)) return true;
    if (byteAt(model, index) != '-') return false;
    var end = index + 1;
    for (0..8) |_| {
        const length = digitBytes(model, end);
        if (length == 0) return false;
        end += length;
    }
    return atEnd(model, end);
}

/// `^claude-[a-z]+-\d+(?:[-.]\d+)?(?:-\d{8})?$`. Nothing that may follow a
/// digit run is itself a digit, so each run is the longest one; only the
/// optional minor group has two readings.
fn claudeBareId(model: []const u8) bool {
    if (!std.mem.startsWith(u8, model, "claude-")) return false;
    const major = claudeFamilyEnd(model, 0) orelse return false;
    const major_end = digitsEnd(model, major);
    if (major_end == major) return false;
    if (isOneOf(byteAt(model, major_end), "-.")) {
        const minor_end = digitsEnd(model, major_end + 1);
        if (minor_end > major_end + 1) {
            if (claudeDateThenEnd(model, minor_end)) return true;
        }
    }
    return claudeDateThenEnd(model, major_end);
}

/// `claude-[a-z]+-\d+(?:[-.]\d+)?(?:-\d{8})?`, unanchored: both tails are
/// optional, so one digit after the family decides the match.
fn claudeFamily(model: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, model, from, "claude-")) |start| : (from = start + 1) {
        const major = claudeFamilyEnd(model, start) orelse continue;
        if (digitBytes(model, major) != 0) return true;
    }
    return false;
}

/// `claude-[a-z]+-(?:4[-._](?:[M-9]|[1-9]\d)(?!\d)|[5-9](?!\d)(?:…)?)` where
/// `minor_min` is M. The group after a bare major is optional and cannot fail.
fn claudeVersion(model: []const u8, comptime minor_min: u8) bool {
    comptime assert(minor_min >= '1' and minor_min <= '9');
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, model, from, "claude-")) |start| : (from = start + 1) {
        const major = claudeFamilyEnd(model, start) orelse continue;
        if (byteAt(model, major) == '4') {
            if (isOneOf(byteAt(model, major + 1), "-._")) {
                const minor = major + 2;
                if (inRange(byteAt(model, minor), minor_min, '9')) {
                    if (digitBytes(model, minor + 1) == 0) return true;
                }
                if (inRange(byteAt(model, minor), '1', '9')) {
                    const second = digitBytes(model, minor + 1);
                    if (second != 0) {
                        if (digitBytes(model, minor + 1 + second) == 0) return true;
                    }
                }
            }
        }
        if (inRange(byteAt(model, major), '5', '9')) {
            if (digitBytes(model, major + 1) == 0) return true;
        }
    }
    return false;
}

/// `claude-[a-z]+-4[-._]6(?!\d)`.
fn claudeVersionExact(model: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, model, from, "claude-")) |start| : (from = start + 1) {
        const major = claudeFamilyEnd(model, start) orelse continue;
        if (byteAt(model, major) != '4') continue;
        if (!isOneOf(byteAt(model, major + 1), "-._")) continue;
        if (byteAt(model, major + 2) != '6') continue;
        if (digitBytes(model, major + 3) == 0) return true;
    }
    return false;
}

/// `(?:codex|deep-research|chat-latest)(?![a-z0-9])` at `index`.
fn openaiLine(model: []const u8, index: usize) bool {
    for ([_][]const u8{ "codex", "deep-research", "chat-latest" }) |line| {
        if (startsAt(model, index, line)) return !isLowerOrDigit(byteAt(model, index + line.len));
    }
    return false;
}

/// `o[1-9]\d*(?![a-z0-9])` at `index`. Backtracking may stop before a
/// non-ASCII digit, which the ASCII lookahead class does not contain.
fn openaiOSeries(model: []const u8, index: usize) bool {
    if (byteAt(model, index) != 'o') return false;
    if (!inRange(byteAt(model, index + 1), '1', '9')) return false;
    var end = index + 2;
    while (inRange(byteAt(model, end), '0', '9')) end += 1;
    return !isLower(byteAt(model, end));
}

/// `gpt-[5-9](?:\.\d+)?(?![0-9.])` at `index`.
fn openaiGptMajor(model: []const u8, index: usize) bool {
    if (!startsAt(model, index, "gpt-")) return false;
    const major = index + "gpt-".len;
    if (!inRange(byteAt(model, major), '5', '9')) return false;
    if (byteAt(model, major + 1) != '.') return !inRange(byteAt(model, major + 1), '0', '9');
    // The lookahead refuses the dot itself, so the minor group is required.
    const first = digitBytes(model, major + 2);
    if (first == 0) return false;
    var end = major + 2 + first;
    while (inRange(byteAt(model, end), '0', '9')) end += 1;
    return byteAt(model, end) != '.';
}

/// `gpt-\d+(?:\.\d+)?(?:-[a-z0-9]+)*-(?:codex|…)(?![a-z0-9])` at `index`. A
/// hyphen follows every run, so each digit run and each segment is the
/// longest one, and only the number of segments varies.
fn openaiGptLine(model: []const u8, index: usize) bool {
    if (!startsAt(model, index, "gpt-")) return false;
    const major = index + "gpt-".len;
    var end = digitsEnd(model, major);
    if (end == major) return false;
    if (byteAt(model, end) == '.') {
        const minor_end = digitsEnd(model, end + 1);
        if (minor_end == end + 1) return false;
        end = minor_end;
    }
    while (byteAt(model, end) == '-') {
        if (openaiLine(model, end + 1)) return true;
        var segment_end = end + 1;
        while (isLowerOrDigit(byteAt(model, segment_end))) segment_end += 1;
        if (segment_end == end + 1) return false;
        end = segment_end;
    }
    return false;
}

/// The alternatives after the namespaces and the optional `ft:` prefix.
fn openaiFamilyAt(model: []const u8, index: usize) bool {
    if (openaiOSeries(model, index)) return true;
    if (openaiGptMajor(model, index)) return true;
    if (openaiGptLine(model, index)) return true;
    return openaiLine(model, index);
}

/// The OpenAI reasoning-family pattern. `(?:[a-z0-9_.-]+/)*` may stop after
/// any number of leading namespace segments, so every such stop is tried.
fn openaiReasoningFamily(model: []const u8) bool {
    const line_end = std.mem.indexOfScalar(u8, model, '\n') orelse model.len;
    if (std.mem.indexOf(u8, model[0..line_end], "search-api") != null) return false;
    var start: usize = 0;
    while (true) {
        if (openaiFamilyAt(model, start)) return true;
        if (startsAt(model, start, "ft:")) {
            if (openaiFamilyAt(model, start + "ft:".len)) return true;
        }
        var end = start;
        while (isLowerOrDigit(byteAt(model, end)) or isOneOf(byteAt(model, end), "_.-")) end += 1;
        if (end == start) return false;
        if (byteAt(model, end) != '/') return false;
        start = end + 1;
    }
}

/// `-(?:pro|flash)(?:-lite)?(?![a-z])` at `index`. A following `-lite` starts
/// with a hyphen, which already satisfies the lookahead without it.
fn geminiTier(model: []const u8, index: usize) bool {
    if (byteAt(model, index) != '-') return false;
    for ([_][]const u8{ "pro", "flash" }) |tier| {
        if (startsAt(model, index + 1, tier)) return !isLower(byteAt(model, index + 1 + tier.len));
    }
    return false;
}

/// The Gemini chat pattern, unanchored.
fn geminiChat(model: []const u8) bool {
    const excluded = [_][]const u8{
        "-tts",          "-image",    "-live",       "-audio",     "-embedding",
        "-computer-use", "-robotics", "-transcribe", "-translate",
    };
    const stem = "gemini-";
    var from: usize = 0;
    occurrence: while (std.mem.indexOfPos(u8, model, from, stem)) |start| : (from = start + 1) {
        const version = start + stem.len;
        const line_end = std.mem.indexOfScalarPos(u8, model, version, '\n') orelse model.len;
        for (excluded) |line| {
            if (std.mem.indexOf(u8, model[version..line_end], line) != null) continue :occurrence;
        }
        if (byteAt(model, version) == '2') {
            if (isOneOf(byteAt(model, version + 1), ".-")) {
                if (inRange(byteAt(model, version + 2), '5', '9')) {
                    if (geminiTier(model, version + 3)) return true;
                }
            }
        }
        if (inRange(byteAt(model, version), '3', '9')) {
            if (isOneOf(byteAt(model, version + 1), ".-")) {
                const first = digitBytes(model, version + 2);
                if (first != 0) {
                    const second = digitBytes(model, version + 2 + first);
                    if (second != 0) {
                        if (geminiTier(model, version + 2 + first + second)) return true;
                    }
                    if (geminiTier(model, version + 2 + first)) return true;
                }
            }
            if (geminiTier(model, version + 1)) return true;
        }
    }
    return false;
}

const testing = std.testing;

fn expectSearch(pattern: Pattern, expected: bool, model: []const u8) !void {
    testing.expectEqual(expected, search(pattern, model)) catch |err| {
        std.debug.print("pattern {s} model {s}\n", .{ @tagName(pattern), model });
        return err;
    };
}

test "Claude id patterns keep their anchors, version bounds and digit lookaheads" {
    try expectSearch(.bedrock_claude_prefix, true, "anthropic.claude-3-haiku-20240307-v1:0");
    try expectSearch(.bedrock_claude_prefix, true, "us-gov.anthropic.claude-x");
    try expectSearch(.bedrock_claude_prefix, false, "us.eu.anthropic.claude-x");
    try expectSearch(.bedrock_claude_prefix, false, "bedrock/anthropic.claude-x");
    try expectSearch(.bedrock_claude_prefix, false, ".anthropic.claude-x");

    try expectSearch(.claude_bare_id, true, "claude-newfamily-5");
    try expectSearch(.claude_bare_id, true, "claude-opus-4-1-20250805");
    try expectSearch(.claude_bare_id, true, "claude-opus-4-20250514");
    try expectSearch(.claude_bare_id, true, "claude-opus-4.5\n");
    try expectSearch(.claude_bare_id, false, "claude-opus-4-1-2025080");
    try expectSearch(.claude_bare_id, false, "claude-opus-4-1-202508055");
    try expectSearch(.claude_bare_id, false, "claude-opus-4-1-latest");
    try expectSearch(.claude_bare_id, false, "claude-opus-4\n\n");
    try expectSearch(.claude_bare_id, false, "anthropic/claude-opus-4");
    try expectSearch(.claude_bare_id, false, "claude-3-haiku");

    try expectSearch(.claude_family, true, "vertex_ai/claude-opus-4@20250514");
    try expectSearch(.claude_family, true, "claude-claude-opus-4");
    try expectSearch(.claude_family, true, "claude-opus-\xd9\xa3");
    try expectSearch(.claude_bare_id, true, "claude-newfamily-\xd9\xa5");
    try expectSearch(.claude_family, false, "claude-3-5-sonnet");
    try expectSearch(.claude_family, false, "claude-opus-latest");
    try expectSearch(.claude_family, false, "claude--4");

    try expectSearch(.claude_version_from_4_6, true, "claude-opus-4-6");
    try expectSearch(.claude_version_from_4_6, true, "claude-opus-4.10");
    try expectSearch(.claude_version_from_4_6, true, "claude-fable-5");
    try expectSearch(.claude_version_from_4_6, true, "claude-fable-5-20270101");
    try expectSearch(.claude_version_from_4_6, false, "claude-opus-4-5");
    try expectSearch(.claude_version_from_4_6, false, "claude-opus-4-20250514");
    try expectSearch(.claude_version_from_4_6, false, "claude-opus-41");
    try expectSearch(.claude_version_from_4_6, false, "claude-opus-4-600");
    // A non-ASCII decimal digit is `\d`: it completes the two-digit minor.
    try expectSearch(.claude_version_from_4_6, true, "claude-opus-4-6\xd9\xa3");
    try expectSearch(.claude_version_from_4_6, false, "claude-opus-4-6\xd9\xa3\xd9\xa3");
    try expectSearch(.claude_version_from_4_6, false, "team-sonnet-5-1");
    try expectSearch(.claude_version_from_4_8, true, "claude-opus-4_8");
    try expectSearch(.claude_version_from_4_8, true, "claude-opus-4-1\xd9\xa3");
    try expectSearch(.claude_version_from_4_8, false, "claude-opus-4-7");
    try expectSearch(.claude_version_from_4_5, true, "us.anthropic.claude-sonnet-4-5-20250929-v1:0");
    try expectSearch(.claude_version_from_4_5, false, "claude-sonnet-4-4");

    try expectSearch(.claude_version_4_6, true, "claude-sonnet-4-6-20260219");
    try expectSearch(.claude_version_4_6, false, "claude-sonnet-4-60");
    try expectSearch(.claude_version_4_6, false, "claude-sonnet-4-7");
    try expectSearch(.claude_fable_or_mythos, true, "bedrock/claude-mythos-preview");
    try expectSearch(.claude_fable_or_mythos, false, "claude-fable");
}

test "OpenAI reasoning pattern follows namespaces, prefixes and ASCII lookaheads" {
    const pattern: Pattern = .openai_reasoning_family;
    try expectSearch(pattern, true, "o1");
    try expectSearch(pattern, true, "o3-pro");
    try expectSearch(pattern, true, "azure/eu/o4-mini");
    try expectSearch(pattern, true, "ft:o4-mini-2025-04-16:org::id");
    try expectSearch(pattern, true, "o1\xd9\xa3x");
    try expectSearch(pattern, true, "o1\xd9\xa1");
    try expectSearch(pattern, true, "gpt-5.\xd9\xa1");
    try expectSearch(pattern, false, "gpt-5.\xd9\xa1.");
    try expectSearch(pattern, false, "o1x");
    try expectSearch(pattern, false, "o0");
    try expectSearch(pattern, false, "omni-moderation-latest");
    try expectSearch(pattern, false, "Azure/o3");

    try expectSearch(pattern, true, "gpt-5");
    try expectSearch(pattern, true, "gpt-5.5-cyber");
    try expectSearch(pattern, true, "gpt-6-astra");
    try expectSearch(pattern, true, "openai/gpt-5-mini");
    try expectSearch(pattern, false, "gpt-4o");
    try expectSearch(pattern, false, "gpt-55");
    try expectSearch(pattern, false, "gpt-5.");
    try expectSearch(pattern, false, "gpt-5.1.2");
    try expectSearch(pattern, false, "gpt-5-search-api");
    try expectSearch(pattern, true, "gpt-5\nsearch-api");
    try expectSearch(pattern, false, "x\n/gpt-5");

    try expectSearch(pattern, true, "codex-mini-latest");
    try expectSearch(pattern, true, "gpt-4.1-codex");
    try expectSearch(pattern, true, "gpt-4-turbo-preview-deep-research");
    try expectSearch(pattern, true, "gpt-4-chat-latest-2025");
    try expectSearch(pattern, true, "gpt-4-deep-researchx-codex");
    try expectSearch(pattern, false, "gpt-4o-chat-latest");
    try expectSearch(pattern, false, "gpt-4-codexes");
    try expectSearch(pattern, false, "gpt-4-Turbo-codex");
    try expectSearch(pattern, false, "chatgpt-4o-latest");
    try expectSearch(pattern, false, "my codex");
}

test "Gemini and W&B patterns keep their version, mode and namespace boundaries" {
    const pattern: Pattern = .gemini_chat;
    try expectSearch(pattern, true, "gemini-2.5-pro");
    try expectSearch(pattern, true, "vertex_ai/gemini-2.5-flash-lite-preview-06-17");
    try expectSearch(pattern, true, "databricks-gemini-3-1-pro");
    try expectSearch(pattern, true, "gemini-3-flash");
    try expectSearch(pattern, true, "gemini-3.12-pro");
    try expectSearch(pattern, false, "gemini-3.123-pro");
    try expectSearch(pattern, false, "gemini-2.0-flash");
    try expectSearch(pattern, false, "gemini-2-pro");
    try expectSearch(pattern, false, "gemini-2.5-prox");
    try expectSearch(pattern, false, "gemini-2.5-flash-image");
    try expectSearch(pattern, false, "gemini-2.5-pro-preview-tts");
    try expectSearch(pattern, false, "gemini-3.8-live");
    try expectSearch(pattern, true, "gemini-2.5-pro\n-live");
    try expectSearch(pattern, true, "gemini-2.5-flash-image/gemini-2.5-pro");
    try expectSearch(pattern, false, "gemini-1.5-pro");

    try expectSearch(.wandb_prefix, true, "wandb/meta-llama/Llama-3.1-8B-Instruct");
    try expectSearch(.wandb_prefix, false, "x/wandb/model");
}
