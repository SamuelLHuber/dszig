//! Shared by the examples and golden tests. Zig mirror of tools/dspy_reference/reference.py (RAG program, corpus,
//! trainset, keyword LM rules, metric).

const std = @import("std");
const dszig = @import("dszig");
const Allocator = std.mem.Allocator;

pub const QA = dszig.Signature(struct {
    pub const instructions = "Answer questions with short factoid answers.";
    pub const Input = struct { context: []const []const u8, question: []const u8 };
    pub const Output = struct { answer: []const u8 };
    pub const descriptions = .{ .context = "may contain relevant facts", .answer = "often between 1 and 5 words" };
});

pub const RAG = struct {
    retrieve: dszig.Retrieve = .{ .k = 2 },
    generate_answer: dszig.ChainOfThought(QA) = .{},
    compiled: bool = false,

    pub const Input = struct { question: []const u8 };
    pub const Output = dszig.ChainOfThought(QA).Output;

    pub fn forward(self: *const RAG, ctx: dszig.Context, input: Input) anyerror!Output {
        const context = try self.retrieve.forward(ctx, input.question);
        return self.generate_answer.forward(ctx, .{ .context = context, .question = input.question });
    }
};

pub const corpus: dszig.predict.InMemoryRetriever = .{ .passages = &.{
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Mount Everest is the highest mountain on Earth.",
    "The Pacific is the largest ocean.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Tokyo is the capital of Japan.",
} };

pub const train_rows = [_][2][]const u8{
    .{ "What is the capital of France?", "Paris" },
    .{ "What is the highest mountain?", "Everest" },
    .{ "What is the largest ocean?", "Pacific" },
    .{ "What is the capital of Germany?", "Berlin" },
    .{ "At what temperature does water boil?", "100 degrees" },
};

pub fn trainset(arena: Allocator) ![]const dszig.Example {
    const out = try arena.alloc(dszig.Example, train_rows.len);
    for (train_rows, out) |row, *ex| {
        var record: dszig.value.Record = .empty;
        try record.put(arena, "question", .{ .string = row[0] });
        try record.put(arena, "answer", .{ .string = row[1] });
        ex.* = .{ .record = record, .input_keys = &.{"question"} };
    }
    return out;
}

fn xml(comptime reasoning: []const u8, comptime answer: []const u8) []const u8 {
    return "<plan>\n" ++ reasoning ++ "\n</plan>\n\n<answer>\n" ++ answer ++ "\n</answer>";
}

pub const rules = [_]dszig.lm.KeywordLM.Rule{
    .{ .needle = "What is the capital of France?", .response = xml("The context says Paris is the capital of France.", "Paris") },
    .{ .needle = "What is the highest mountain?", .response = xml("Everest is the highest mountain.", "Mount Everest") },
    .{ .needle = "What is the largest ocean?", .response = xml("The Pacific is the largest ocean.", "The Pacific") },
    .{ .needle = "What is the capital of Germany?", .response = xml("Not sure.", "Munich") },
    .{ .needle = "At what temperature does water boil?", .response = xml("Water boils at 100 degrees Celsius.", "100 degrees Celsius") },
};

pub fn keywordLM() dszig.lm.KeywordLM {
    return .{ .rules = &rules, .default = xml("No idea.", "unknown") };
}

/// `example.answer.lower() in pred.answer.lower()`.
pub fn keywordMetric(arena: Allocator, example: *const dszig.Example, prediction: *const dszig.value.Record, _: ?*const dszig.Trace) f64 {
    const want = example.record.get("answer").?.string;
    const got = (prediction.get("answer") orelse return 0).string;
    const w = std.ascii.allocLowerString(arena, want) catch return 0;
    const g = std.ascii.allocLowerString(arena, got) catch return 0;
    return if (std.mem.indexOf(u8, g, w) != null) 1 else 0;
}

pub const Hop = dszig.Signature(struct {
    pub const instructions = "Write a search query that finds the missing facts.";
    pub const Input = struct { context: []const []const u8, question: []const u8 };
    pub const Output = struct { query: []const u8 };
});

/// Runs `hop` twice per question, so BootstrapFewShot must pick one of two
/// same-predictor traces (seeded by the pickle hash of both).
pub const MultiHop = struct {
    retrieve: dszig.Retrieve = .{ .k = 1 },
    hop: dszig.Predict(Hop) = .{},
    generate_answer: dszig.ChainOfThought(QA) = .{},
    compiled: bool = false,

    pub const Input = struct { question: []const u8 };
    pub const Output = dszig.ChainOfThought(QA).Output;

    pub fn forward(self: *const MultiHop, ctx: dszig.Context, input: Input) anyerror!Output {
        var context: []const []const u8 = &.{};
        var query = input.question;
        for (0..2) |_| {
            context = try std.mem.concat(ctx.arena, []const u8, &.{ context, try self.retrieve.forward(ctx, query) });
            query = (try self.hop.forward(ctx, .{ .context = context, .question = input.question })).query;
        }
        context = try std.mem.concat(ctx.arena, []const u8, &.{ context, try self.retrieve.forward(ctx, query) });
        return self.generate_answer.forward(ctx, .{ .context = context, .question = input.question });
    }
};

fn hopReply(comptime query: []const u8) []const u8 {
    return "<query>\n" ++ query ++ "\n</query>";
}

/// The second hop sees the Tokyo passage the first hop's query retrieved.
pub const multihop_rules = [_]dszig.lm.KeywordLM.Rule{
    .{ .needle = "\"Tokyo is the capital of Japan.\"]\n</context>", .response = hopReply("highest mountain Everest") },
    .{ .needle = "<query>", .response = hopReply("Tokyo Japan capital") },
} ++ rules;

pub fn multihopLM() dszig.lm.KeywordLM {
    return .{ .rules = &multihop_rules, .default = xml("No idea.", "unknown") };
}
