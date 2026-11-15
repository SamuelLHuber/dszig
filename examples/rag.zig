//! A retrieval-augmented ChainOfThought program: `Retrieve` pulls passages,
//! `ChainOfThought(QA)` answers. Prints the exact prompt DSPy would send and
//! the parsed answer. The LM is a deterministic keyword mock; swap in
//! `dszig.lm.OpenAI` without touching the program.

const std = @import("std");
const dszig = @import("dszig");
const rag = @import("rag_program");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    var history: dszig.lm.History = .{ .arena = arena };
    var keyword = rag.keywordLM();
    keyword.history = &history;
    var trace: dszig.Trace = .{ .arena = arena };
    const ctx: dszig.Context = .{
        .arena = arena,
        .lm = keyword.lm(),
        .retriever = rag.corpus.retriever(),
        .trace = &trace,
    };

    const program: rag.RAG = .{};
    const output = try program.forward(ctx, .{ .question = "What is the capital of France?" });

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    const w = &stdout.interface;
    for (history.requests.items[0].messages) |m| {
        try w.print("--- {s} ---\n", .{m.role.name()});
        switch (m.content) {
            .text => |text| try w.writeAll(text),
            .parts => try dszig.pyjson.writeJson(w, m.content.toValue(arena), .indent2),
        }
        try w.writeByte('\n');
    }
    try w.print("=== reasoning: {s}\n=== answer: {s}\n=== trace steps: {d}\n", .{
        if (output.reasoning) |reasoning| reasoning.content else "(none)",
        output.answer,
        trace.steps.items.len,
    });
    try w.flush();
}
