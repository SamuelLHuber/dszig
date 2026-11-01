//! Five-example BootstrapFewShot over the RAG program with a boolean keyword
//! metric. Prints baseline and optimized scores; with a path argument, also
//! writes the compiled program in DSPy's `program.save()` JSON format.
//!
//!     zig build bootstrap -- rag.json

const std = @import("std");
const dszig = @import("dszig");
const rag = @import("rag_program");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .arena = arena, .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(arena);

    const baseline = try dszig.optimize.evaluate(rag.RAG, init.gpa, ctx, &rag.RAG{}, trainset, rag.keywordMetric, 10);
    const optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, arena, init.gpa, ctx, .{}, null, trainset);
    const optimized = try dszig.optimize.evaluate(rag.RAG, init.gpa, ctx, &result.program, trainset, rag.keywordMetric, 10);

    var buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    const w = &stdout.interface;
    try w.print("Bootstrapped {d} full traces after {d} attempts ({d} errors).\n", .{
        result.report.bootstrapped, result.report.attempts, result.report.errors,
    });
    try w.print("baseline {d:.1}%  optimized {d:.1}%\n", .{ baseline.score, optimized.score });
    for (result.program.generate_answer.predict.state.demos) |demo| {
        try w.print("  demo: {s}\n", .{demo.get("question").?.string});
    }
    if (args.len > 1) {
        const json = try dszig.state.saveAlloc(arena, &result.program);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[1], .data = json });
        try w.print("saved {s} ({d} bytes)\n", .{ args[1], json.len });
    }
    try w.flush();
}
