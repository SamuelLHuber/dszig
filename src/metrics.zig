//! DSPy 3.5 answer metrics. Text normalization is shared with majority;
//! passage matching uses DPR's token boundaries, not substring search.
//! String inputs use UTF-8/WTF-8; returned text and scratch storage use arena.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Record = @import("value.zig").Record;
const Example = @import("optimize.zig").Example;
const Value = @import("runtime_value.zig").Value;
const dpr = @import("dpr.zig");

pub const normalizeText = @import("ensemble.zig").normalizeText;
pub const Error = Allocator.Error || error{ EmptyAnswers, InvalidAnswer, MissingAnswer, InvalidContext };

/// `em_score`: two empty normalized strings match, unlike token F1.
pub fn emScore(arena: Allocator, prediction: []const u8, truth: []const u8) Allocator.Error!bool {
    return std.mem.eql(u8, try normalizeText(arena, prediction), try normalizeText(arena, truth));
}

const Overlap = struct {
    common: usize,
    predicted: usize,
    expected: usize,

    fn f1(self: Overlap) f64 {
        if (self.common == 0) return 0;
        const precision = @as(f64, @floatFromInt(self.common)) / @as(f64, @floatFromInt(self.predicted));
        const recall = @as(f64, @floatFromInt(self.common)) / @as(f64, @floatFromInt(self.expected));
        // Keep Python's operation order; the algebraically reduced expression
        // can yield different binary64 results.
        return (2 * precision * recall) / (precision + recall);
    }
};

/// Both arguments have already been normalized, so tokens are ASCII-space
/// separated even when their content is Unicode. Counter intersection counts
/// repeats only up to the reference multiplicity.
fn overlap(arena: Allocator, prediction: []const u8, truth: []const u8) Allocator.Error!Overlap {
    var counts: std.StringHashMapUnmanaged(usize) = .empty;
    defer counts.deinit(arena);
    var result: Overlap = .{ .common = 0, .predicted = 0, .expected = 0 };
    var expected = std.mem.tokenizeScalar(u8, truth, ' ');
    while (expected.next()) |token| {
        const entry = try counts.getOrPut(arena, token);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
        result.expected += 1;
    }
    var predicted = std.mem.tokenizeScalar(u8, prediction, ' ');
    while (predicted.next()) |token| {
        result.predicted += 1;
        if (counts.getPtr(token)) |remaining| {
            if (remaining.* > 0) {
                remaining.* -= 1;
                result.common += 1;
            }
        }
    }
    return result;
}

pub fn f1Score(arena: Allocator, prediction: []const u8, truth: []const u8) Allocator.Error!f64 {
    return (try overlap(arena, try normalizeText(arena, prediction), try normalizeText(arena, truth))).f1();
}

fn isSpecialLabel(text: []const u8) bool {
    return std.mem.eql(u8, text, "yes") or std.mem.eql(u8, text, "no") or std.mem.eql(u8, text, "noanswer");
}

pub fn hotpotF1Score(arena: Allocator, prediction: []const u8, truth: []const u8) Allocator.Error!f64 {
    const p = try normalizeText(arena, prediction);
    const t = try normalizeText(arena, truth);
    if ((isSpecialLabel(p) or isSpecialLabel(t)) and !std.mem.eql(u8, p, t)) return 0;
    return (try overlap(arena, p, t)).f1();
}

pub fn precisionScore(arena: Allocator, prediction: []const u8, truth: []const u8) Allocator.Error!f64 {
    const counts = try overlap(arena, try normalizeText(arena, prediction), try normalizeText(arena, truth));
    if (counts.common == 0) return 0;
    return @as(f64, @floatFromInt(counts.common)) / @as(f64, @floatFromInt(counts.predicted));
}

/// `EM`: Python max over every reference, including its empty-list error.
pub fn em(arena: Allocator, prediction: []const u8, answers: []const []const u8) Error!bool {
    if (answers.len == 0) return error.EmptyAnswers;
    const normalized = try normalizeText(arena, prediction);
    var best = false;
    for (answers) |answer| {
        const score = std.mem.eql(u8, normalized, try normalizeText(arena, answer));
        best = best or score;
    }
    return best;
}

pub fn f1(arena: Allocator, prediction: []const u8, answers: []const []const u8) Error!f64 {
    return bestF1(arena, prediction, answers, false);
}

pub fn hotpotF1(arena: Allocator, prediction: []const u8, answers: []const []const u8) Error!f64 {
    return bestF1(arena, prediction, answers, true);
}

fn bestF1(arena: Allocator, prediction: []const u8, answers: []const []const u8, hotpot: bool) Error!f64 {
    if (answers.len == 0) return error.EmptyAnswers;
    const normalized = try normalizeText(arena, prediction);
    var best: f64 = 0;
    for (answers) |answer| {
        const truth = try normalizeText(arena, answer);
        if (hotpot and (isSpecialLabel(normalized) or isSpecialLabel(truth)) and !std.mem.eql(u8, normalized, truth)) continue;
        const score = (try overlap(arena, normalized, truth)).f1();
        if (score > best) best = score;
    }
    return best;
}

/// `_answer_match`: frac >= 1 selects exact match; it does NOT require F1
/// to exceed frac. Negative/NaN thresholds retain ordinary Python comparisons.
pub fn answerMatch(arena: Allocator, prediction: []const u8, answers: []const []const u8, frac: f64) Error!bool {
    if (frac >= 1) return em(arena, prediction, answers);
    return try f1(arena, prediction, answers) >= frac;
}

/// `_passage_match`: contiguous DPR tokens, including an empty answer's
/// zero-length match in any passage (but never in an empty passage list).
pub fn passageMatch(arena: Allocator, passages: []const []const u8, answers: []const []const u8) Allocator.Error!bool {
    if (passages.len == 0) return false;
    const targets = try passageTargets(arena, answers);
    for (passages) |passage| {
        if (try passageMatchesTargets(arena, passage, targets)) return true;
    }
    return false;
}

const Tokens = []const []const u8;

fn passageTargets(arena: Allocator, answers: []const []const u8) Allocator.Error![]const Tokens {
    const targets = try arena.alloc(Tokens, answers.len);
    for (answers, targets) |answer, *target| {
        target.* = try dpr.normalize(arena, try normalizeText(arena, answer));
    }
    return targets;
}

fn passageMatchesTargets(arena: Allocator, passage: []const u8, targets: []const Tokens) Allocator.Error!bool {
    if (targets.len == 0) return false;
    const tokens = try dpr.normalize(arena, try normalizeText(arena, passage));
    for (targets) |target| {
        if (containsTokens(tokens, target)) return true;
    }
    return false;
}

fn containsTokens(tokens: []const []const u8, target: []const []const u8) bool {
    if (target.len > tokens.len) return false;
    for (0..tokens.len - target.len + 1) |start| {
        var matches = true;
        for (target, tokens[start..][0..target.len]) |wanted, actual| {
            if (!std.mem.eql(u8, wanted, actual)) {
                matches = false;
                break;
            }
        }
        if (matches) return true;
    }
    return false;
}

fn answerStrings(arena: Allocator, answer: Value) Error![]const []const u8 {
    if (answer == .string) {
        const result = try arena.alloc([]const u8, 1);
        result[0] = answer.string;
        return result;
    }
    return stringList(arena, answer, error.InvalidAnswer);
}

fn stringList(arena: Allocator, list: Value, invalid: Error) Error![]const []const u8 {
    if (list != .array) return invalid;
    const result = try arena.alloc([]const u8, list.array.items.len);
    for (list.array.items, result) |item, *text| {
        if (item != .string) return invalid;
        text.* = item.string;
    }
    return result;
}

/// Record boundary corresponding to `answer_exact_match(example, pred, frac)`.
/// Trace is not consulted by DSPy's metric. Callers can adapt the bool to
/// optimize.Metric's numeric result without losing validation errors here.
pub fn answerExactMatch(arena: Allocator, example: *const Example, prediction: *const Record, frac: f64) Error!bool {
    const answer = example.record.get("answer") orelse return error.MissingAnswer;
    if (answer != .string and answer != .array) return error.InvalidAnswer;
    const predicted = prediction.get("answer") orelse return error.MissingAnswer;
    if (predicted != .string) return error.InvalidAnswer;
    return answerMatch(arena, predicted.string, try answerStrings(arena, answer), frac);
}

/// Runtime `_passage_match`: both arguments follow Python iteration, even
/// when they are strings or dictionaries rather than the hinted list[str].
pub fn passageMatchValue(arena: Allocator, passages: Value, answers: Value) Error!bool {
    return passageMatchRuntime(arena, passages, answers, false);
}

/// Record boundary corresponding to `answer_passage_match(example, pred)`.
/// Unlike direct `_passage_match`, an Example.answer string is ONE reference.
pub fn answerPassageMatch(arena: Allocator, example: *const Example, prediction: *const Record) Error!bool {
    const answer = example.record.get("answer") orelse return error.MissingAnswer;
    if (answer != .string and answer != .array) return error.InvalidAnswer;
    const passages = prediction.get("context") orelse return error.InvalidContext;
    return passageMatchRuntime(arena, passages, answer, answer == .string);
}

const RuntimeIterator = struct {
    value: Value,
    index: usize = 0,
    custom_items: []const Value = &.{},
    invalid: Error,

    fn init(arena: Allocator, iterable: Value, invalid: Error) Error!RuntimeIterator {
        var result: RuntimeIterator = .{ .value = iterable, .invalid = invalid };
        switch (iterable) {
            .array, .object, .string => {},
            .custom => {
                const iterate = iterable.custom.vtable.iterate orelse return invalid;
                result.custom_items = iterate(arena, iterable.custom.ptr) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return invalid,
                };
            },
            else => return invalid,
        }
        return result;
    }

    fn next(self: *RuntimeIterator) Error!?Value {
        switch (self.value) {
            .array, .custom => {
                const items = if (self.value == .array) self.value.array.items else self.custom_items;
                if (self.index == items.len) return null;
                const item = items[self.index];
                self.index += 1;
                return item;
            },
            .object => {
                const keys = self.value.object.keys();
                if (self.index == keys.len) return null;
                const key = keys[self.index];
                self.index += 1;
                return .{ .string = key };
            },
            .string => {
                const text = self.value.string;
                if (self.index == text.len) return null;
                const len = std.unicode.utf8ByteSequenceLength(text[self.index]) catch return self.invalid;
                if (len > text.len - self.index) return self.invalid;
                const character = text[self.index..][0..len];
                _ = std.unicode.wtf8Decode(character) catch return self.invalid;
                self.index += len;
                return .{ .string = character };
            },
            else => unreachable,
        }
    }
};

fn passageMatchRuntime(arena: Allocator, passages: Value, answers: Value, single_answer: bool) Error!bool {
    var context = try RuntimeIterator.init(arena, passages, error.InvalidContext);
    // Do not inspect any answer-list members or even require answers to be
    // iterable until the first passage exists.
    var current = (try context.next()) orelse return false;
    var single: [1]Tokens = undefined;
    var targets: []const Tokens = undefined;
    if (single_answer) {
        single[0] = try dpr.normalize(arena, try normalizeText(arena, answers.string));
        targets = &single;
    } else {
        var references = try RuntimeIterator.init(arena, answers, error.InvalidAnswer);
        var collected: std.ArrayList(Tokens) = .empty;
        while (try references.next()) |answer| {
            if (answer != .string) return error.InvalidAnswer;
            try collected.append(arena, try dpr.normalize(arena, try normalizeText(arena, answer.string)));
        }
        targets = collected.items;
    }
    while (true) {
        if (current != .string) return error.InvalidContext;
        if (try passageMatchesTargets(arena, current.string, targets)) return true;
        current = (try context.next()) orelse return false;
    }
}

test "token metrics count repeats, empty answers and HotPot labels" {
    var state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    try std.testing.expect(try emScore(arena, "The!", "a"));
    try std.testing.expectEqual(@as(f64, 0), try f1Score(arena, "The!", "a"));
    try std.testing.expectEqual(@as(f64, 0.5), try f1Score(arena, "cat cat cat", "cat"));
    try std.testing.expectEqual(@as(f64, 0.5), try precisionScore(arena, "cat cat dog dog", "cat dog"));
    try std.testing.expectEqual(@as(f64, 0), try hotpotF1Score(arena, "yes", "yes maybe"));
    try std.testing.expectEqual(@as(f64, 0.5), try f1Score(arena, "yes", "yes maybe maybe"));
    try std.testing.expect(try answerMatch(arena, "Paris", &.{"The Paris!"}, 2));
    try std.testing.expect(!try answerMatch(arena, "Paris", &.{"Paris"}, std.math.nan(f64)));
    try std.testing.expectError(error.EmptyAnswers, em(arena, "Paris", &.{}));
    try std.testing.expectError(error.EmptyAnswers, f1(arena, "Paris", &.{}));
}

test "passage metrics match complete tokens and normalized empty answers" {
    var state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    try std.testing.expect(try passageMatch(arena, &.{"The Eiffel Tower is in Paris."}, &.{"Eiffel Tower"}));
    try std.testing.expect(!try passageMatch(arena, &.{"Parisian"}, &.{"Paris"}));
    try std.testing.expect(try passageMatch(arena, &.{""}, &.{"the"}));
    try std.testing.expect(!try passageMatch(arena, &.{}, &.{"the"}));
    var context = @import("runtime_value.zig").Array.init(arena);
    try context.append(.{ .string = "red" });
    // Direct _passage_match iterates an answers string as characters, while
    // the record wrapper treats its answer string as a single reference.
    try std.testing.expect(!try passageMatchValue(arena, .{ .array = context }, .{ .string = "red" }));
    try std.testing.expect(try passageMatchValue(arena, .{ .string = "red" }, .{ .string = "red" }));
    try std.testing.expect(!try passageMatchValue(arena, .{ .array = context }, .{ .string = "" }));
    try std.testing.expect(try passageMatchValue(arena, .{ .string = "\xed\xa0\x80" }, .{ .string = "\xed\xa0\x80" }));
    try std.testing.expect(!try passageMatchValue(arena, .{ .array = @import("runtime_value.zig").Array.init(arena) }, .null));
    try std.testing.expect(!try passageMatch(arena, &.{"Paris"}, &.{}));
}

test "record metrics retain context iteration and lazy passage validation" {
    var state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    var example: Example = .{ .record = .empty, .input_keys = &.{} };
    var prediction: Record = .empty;
    try example.record.put(arena, "answer", .{ .string = "x" });
    // Source iterates a context string character by character, not as one
    // passage: x matches inside xyz, but xy does not.
    try prediction.put(arena, "context", .{ .string = "xyz" });
    try std.testing.expect(try answerPassageMatch(arena, &example, &prediction));
    try example.record.put(arena, "answer", .{ .string = "xy" });
    try std.testing.expect(!try answerPassageMatch(arena, &example, &prediction));

    var invalid_answers = @import("runtime_value.zig").Array.init(arena);
    try invalid_answers.append(.{ .integer = 3 });
    try example.record.put(arena, "answer", .{ .array = invalid_answers });
    try prediction.put(arena, "context", .{ .array = @import("runtime_value.zig").Array.init(arena) });
    try std.testing.expect(!try answerPassageMatch(arena, &example, &prediction));
    try prediction.put(arena, "context", .{ .string = "x" });
    try std.testing.expectError(error.InvalidAnswer, answerPassageMatch(arena, &example, &prediction));

    try example.record.put(arena, "answer", .{ .string = "Paris" });
    var passages = @import("runtime_value.zig").Array.init(arena);
    try passages.append(.{ .string = "The Paris!" });
    try passages.append(.{ .integer = 3 });
    try prediction.put(arena, "context", .{ .array = passages });
    try std.testing.expect(try answerPassageMatch(arena, &example, &prediction));
    try prediction.put(arena, "answer", .{ .string = "Paris" });
    try std.testing.expect(try answerExactMatch(arena, &example, &prediction, 1));
}
