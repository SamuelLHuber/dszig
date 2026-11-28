//! Native `dspy.KNNFewShot`: retrieve and bootstrap anew on every forward.
//!
//! The visible program is the student's reset copy, not the execution target.
//! DSPy's replacement bound method closes over the original student, teacher,
//! and optimizer. Here an arena-owned pointer capture preserves that behavior
//! without exposing captured predictors to discovery. The static transparent
//! parameter target keeps the visible student's original predictor names.
//!
//! `init`'s arena owns training vectors. `compile`'s output arena owns its
//! capture, but no compile-time Context is retained. The optimizer, Embedder,
//! training examples, original student, and optional teacher are borrowed and
//! must outlive every forward. Their captured addresses must remain stable.
//! Optimizer changes are observed at each call, as with the source closure.
//! Separate compile calls have
//! separate captures and visible state. As elsewhere in the native program
//! model, copied programs share immutable slices; mutable state belongs by
//! value, not behind aliased pointers. Each compiled per-call execution program
//! is arena-allocated so returned trace predictor pointers remain valid.
//!
//! Source: DSPy 3.5.0 teleprompt/knn_fewshot.py. Numerical portability limits of
//! the retrieval kernel are stated in knn.zig; no optimizer engine is duplicated.
//!
//! Upstream (DSPy 3.5.0): dspy/teleprompt/knn_fewshot.py

const std = @import("std");
const Allocator = std.mem.Allocator;
const predict = @import("predict.zig");
const optimize = @import("optimize.zig");
const embedding = @import("embedder.zig");
const value = @import("value.zig");
const KNN = @import("knn.zig").KNN;
const Context = predict.Context;
const Workspace = @import("workspace.zig").Workspace;
const Example = optimize.Example;
const BootstrapFewShot = optimize.BootstrapFewShot;

pub fn KNNFewShot(comptime P: type) type {
    return struct {
        const Self = @This();

        knn: KNN,
        bootstrap: BootstrapFewShot,
        /// Fixed scratch each forward reserves from its arena for the
        /// per-neighborhood bootstrap, so the bootstrap's nested scopes never
        /// share the caller's workspace with the arena that owns its demos.
        workspace_bytes_max: usize = 1024 * 1024,

        pub const Program = ProgramWithTeacher(P);

        pub fn ProgramWithTeacher(comptime Teacher: type) type {
            return struct {
                const Compiled = @This();
                const Captured = struct {
                    optimizer: *const Self,
                    gpa: Allocator,
                    original: *const P,
                    teacher: ?*const Teacher,
                };

                pub const Input = P.Input;
                pub const Output = P.Output;
                pub const dszig_parameter_target = "visible";

                visible: P,
                compiled: bool,
                captured: *const Captured,

                pub fn forward(self: *const Compiled, ctx: Context, input: Input) anyerror!Output {
                    const kwargs = try value.toRecord(ctx.arena, Input, input);
                    const neighbors = try self.captured.optimizer.knn.call(ctx.arena, kwargs);
                    // DSPy creates a fresh BootstrapFewShot for each neighborhood.
                    var bootstrap = self.captured.optimizer.bootstrap;
                    bootstrap.error_count = 0;
                    const nested_buffer = try ctx.arena.alloc(u8, self.captured.optimizer.workspace_bytes_max);
                    var nested: Workspace = .init(nested_buffer);
                    var bootstrap_ctx = ctx;
                    bootstrap_ctx.workspace = &nested;
                    const result = try bootstrap.compile(
                        P,
                        ctx.arena,
                        bootstrap_ctx,
                        self.captured.original.*,
                        if (self.captured.teacher) |teacher| teacher.* else null,
                        neighbors,
                    );
                    const program = try ctx.arena.create(P);
                    program.* = result.program;

                    return program.forward(ctx, input);
                }
            };
        }

        pub fn init(
            out: Allocator,
            k: isize,
            trainset: []const Example,
            vectorizer: *embedding.Embedder,
            bootstrap: BootstrapFewShot,
        ) anyerror!Self {
            return .{
                .knn = try KNN.init(out, k, trainset, vectorizer),
                .bootstrap = bootstrap,
            };
        }

        /// Context is deliberately not captured: DSPy's closure uses the
        /// settings active at each forward, not those active during compile.
        /// The pointer arguments likewise retain the actual originals, so
        /// later original/teacher changes are observable but visible-copy
        /// changes do not change the captured execution target.
        pub fn compile(
            self: *const Self,
            out: Allocator,
            gpa: Allocator,
            ctx: Context,
            student: *const P,
            teacher: ?*const P,
        ) anyerror!Program {
            return self.compileWithTeacher(P, out, gpa, ctx, student, teacher);
        }

        /// A different teacher program type is valid when its named predictors
        /// and signatures match the student at query time.
        pub fn compileWithTeacher(
            self: *const Self,
            comptime Teacher: type,
            out: Allocator,
            gpa: Allocator,
            ctx: Context,
            student: *const P,
            teacher: ?*const Teacher,
        ) anyerror!ProgramWithTeacher(Teacher) {
            _ = ctx;
            var visible = student.*;
            try predict.resetPredictors(out, &visible);
            const captured = try out.create(ProgramWithTeacher(Teacher).Captured);
            captured.* = .{
                .optimizer = self,
                .gpa = gpa,
                .original = student,
                .teacher = teacher,
            };
            return .{ .visible = visible, .compiled = visible.compiled, .captured = captured };
        }
    };
}

const testing = std.testing;
const lm_mod = @import("lm.zig");
const Record = value.Record;
const TestSignature = @import("signature.zig").Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

const TestEvents = struct {
    arena: Allocator,
    embeddings: std.ArrayList([]const []const u8) = .empty,
    forwards: std.ArrayList(Forward) = .empty,
    calls: std.ArrayList(Call) = .empty,

    const Forward = struct { marker: []const u8, counter: usize, question: []const u8 };
    const Call = struct { lm: []const u8, questions: []const []const u8, answers: []const []const u8 };

    fn embed(ptr: *anyopaque, arena: Allocator, inputs: []const []const u8, _: Record) anyerror!embedding.Batch {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const texts = try self.arena.alloc([]const u8, inputs.len);
        const vectors = try arena.alloc(f32, inputs.len * 2);
        for (inputs, texts, 0..) |text, *saved, i| {
            saved.* = try self.arena.dupe(u8, text);
            const red = std.mem.indexOf(u8, text, "red") != null;
            vectors[i * 2] = if (red) 1 else 0;
            vectors[i * 2 + 1] = if (red) 0 else 1;
        }
        try self.embeddings.append(self.arena, texts);
        return .{ .float32 = .{ .values = vectors, .rows = inputs.len, .columns = 2 } };
    }
};

const TestProgram = testProgram(false);
const TestTeacher = testProgram(true);

fn testProgram(comptime is_teacher: bool) type {
    return struct {
        pub const teacher = is_teacher;
        pub const Input = TestSignature.Input;
        pub const Output = TestSignature.Output;

        qa: predict.Predict(TestSignature) = .{},
        marker: []const u8,
        counter: usize = 7,
        events: *TestEvents,
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) anyerror!Output {
            try self.events.forwards.append(self.events.arena, .{
                .marker = try self.events.arena.dupe(u8, self.marker),
                .counter = self.counter,
                .question = try self.events.arena.dupe(u8, input.question),
            });
            return self.qa.forward(ctx, .{
                .question = try std.fmt.allocPrint(ctx.arena, "{s}|{s}", .{ input.question, self.marker }),
            });
        }
    };
}

const TestLM = struct {
    label: []const u8,
    events: *TestEvents,

    fn lm(self: *@This()) lm_mod.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        var questions: std.ArrayList([]const u8) = .empty;
        var answers: std.ArrayList([]const u8) = .empty;
        for (request.messages) |message| {
            switch (message.role) {
                .user => try questions.append(self.events.arena, try self.events.arena.dupe(u8, try xmlField(message.content.text, "<question>", "</question>"))),
                .assistant => try answers.append(self.events.arena, try self.events.arena.dupe(u8, try xmlField(message.content.text, "<answer>", "</answer>"))),
                else => {},
            }
        }
        if (questions.items.len == 0) return error.LMFailure;
        try self.events.calls.append(self.events.arena, .{
            .lm = self.label,
            .questions = questions.items,
            .answers = answers.items,
        });
        const result = try arena.alloc(lm_mod.Completion, 1);
        result[0] = .{ .text = try std.fmt.allocPrint(arena, "<answer>\n{s}:{s}:{d}\n</answer>", .{ self.label, questions.items[questions.items.len - 1], answers.items.len }) };
        return result;
    }
};

fn xmlField(text: []const u8, opening: []const u8, closing: []const u8) error{LMFailure}![]const u8 {
    const start = (std.mem.indexOf(u8, text, opening) orelse return error.LMFailure) + opening.len;
    const end = start + (std.mem.indexOf(u8, text[start..], closing) orelse return error.LMFailure);
    return std.mem.trim(u8, text[start..end], value.py_whitespace);
}

fn testMetric(_: Allocator, example: *const Example, maybe_prediction: ?*const optimize.RecordPrediction, trace: ?*const predict.Trace) value.ParseError!f64 {
    const captured = trace orelse return 0;
    if (captured.steps.items.len != 1) return 0;
    const input = captured.steps.items[0].inputs.get("question") orelse return 0;
    const expected = example.record.get("question") orelse return 0;
    const prediction = maybe_prediction orelse return error.ParseValue;
    const answer = prediction.get("answer") orelse return 0;
    return if (std.mem.startsWith(u8, input.string, expected.string) and
        std.mem.startsWith(u8, answer.string, "teacher:")) 1 else 0;
}

fn testTrainset(arena: Allocator) ![]const Example {
    const examples = try arena.alloc(Example, 2);
    examples[0] = try Example.fromJson(arena, "{\"question\":\"red-train\",\"answer\":\"red-gold\"}", &.{"question"});
    examples[1] = try Example.fromJson(arena, "{\"question\":\"blue-train\",\"answer\":\"blue-gold\"}", &.{"question"});
    return examples;
}

fn testDemo(arena: Allocator, question: []const u8) ![]const Record {
    const demos = try arena.alloc(Record, 1);
    demos[0] = .empty;
    try demos[0].put(arena, "question", .{ .string = question });
    try demos[0].put(arena, "answer", .{ .string = question });
    return demos;
}

test "KNNFewShot uses original captures, real bootstrap demos, teacher settings, and per-call context" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var events: TestEvents = .{ .arena = arena };
    var teacher_lm: TestLM = .{ .label = "teacher", .events = &events };
    var student_lm: TestLM = .{ .label = "student", .events = &events };
    var unused_lm: lm_mod.ScriptedLM = .{ .responses = &.{} };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &events, .embed = TestEvents.embed }, .{ .caching = false });
    defer model.deinit();
    const Optimizer = KNNFewShot(TestProgram);
    const optimizer = try Optimizer.init(arena, 1, try testTrainset(arena), &model, .{
        .metric = testMetric,
        .metric_threshold = 1,
        .max_bootstrapped_demos = 1,
        .max_labeled_demos = 0,
        .teacher_settings = .{ .lm = teacher_lm.lm() },
    });
    var student: TestProgram = .{ .marker = "original", .events = &events };
    student.qa.state.demos = try testDemo(arena, "old");
    var teacher: TestTeacher = .{ .marker = "teacher", .events = &events, .compiled = true };
    teacher.qa.state.demos = try testDemo(arena, "teacher-old");
    const compile_ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = unused_lm.lm() };
    var program = try optimizer.compileWithTeacher(TestTeacher, arena, testing.allocator, compile_ctx, &student, &teacher);
    try testing.expectEqual(@as(usize, 1), events.embeddings.items.len);
    try testing.expectEqual(@as(usize, 0), events.calls.items.len);
    try testing.expectEqualStrings("original", program.visible.marker);
    try testing.expectEqual(@as(usize, 7), program.visible.counter);
    try testing.expectEqual(@as(usize, 0), program.visible.qa.state.demos.len);
    try testing.expect(!program.compiled);
    const named = try predict.namedPredictors(arena, &program);
    try testing.expectEqual(@as(usize, 1), named.len);
    try testing.expectEqualStrings("qa", named[0].name);
    try testing.expect(named[0].param.predict == &program.visible.qa.state);
    student.marker = "captured-mutated";
    teacher.marker = "teacher-mutated";
    program.visible.marker = "visible-mutated";
    program.visible.qa.state.demos = try testDemo(arena, "visible-only");
    var trace: predict.Trace = .{ .arena = arena };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = student_lm.lm(), .trace = &trace };
    const red = try program.forward(ctx, .{ .question = "red-query" });
    const blue = try program.forward(ctx, .{ .question = "blue-query" });
    try testing.expectEqualStrings("student:red-query|captured-mutated:1", red.answer);
    try testing.expectEqualStrings("student:blue-query|captured-mutated:1", blue.answer);
    try expectCallSequence(&events, &trace);
    try testing.expectEqualStrings("visible-only", program.visible.qa.state.demos[0].get("question").?.string);
    try testing.expectEqualStrings("old", student.qa.state.demos[0].get("question").?.string);
    try testing.expectEqualStrings("teacher-old", teacher.qa.state.demos[0].get("question").?.string);
    try testing.expectEqual(@as(usize, 7), program.visible.counter);
    try testing.expect(!program.compiled and !program.visible.compiled);
    try testing.expectEqual(@as(usize, 0), unused_lm.index);
    try expectIndependentCapture(Optimizer, &optimizer, arena, ctx, &events);
    // Serial evaluation runs forward with its arena in a workspace scope.
    var scope = ctx.workspace.?.open();
    defer scope.close();
    var scoped = ctx;
    scoped.arena = scope.allocator();
    scoped.trace = null;
    const scoped_red = try program.forward(scoped, .{ .question = "red-query" });
    try testing.expectEqualStrings("student:red-query|captured-mutated:1", scoped_red.answer);
}

fn expectCallSequence(events: *const TestEvents, trace: *const predict.Trace) !void {
    try testing.expectEqual(@as(usize, 3), events.embeddings.items.len);
    try testing.expectEqualStrings("question: red-train", events.embeddings.items[0][0]);
    try testing.expectEqualStrings("question: blue-train", events.embeddings.items[0][1]);
    try testing.expectEqualStrings("question: red-query", events.embeddings.items[1][0]);
    try testing.expectEqualStrings("question: blue-query", events.embeddings.items[2][0]);
    try testing.expectEqual(@as(usize, 4), events.forwards.items.len);
    const markers = [_][]const u8{ "teacher-mutated", "captured-mutated", "teacher-mutated", "captured-mutated" };
    const questions = [_][]const u8{ "red-train", "red-query", "blue-train", "blue-query" };
    for (events.forwards.items, markers, questions) |event, marker, question| {
        try testing.expectEqualStrings(marker, event.marker);
        try testing.expectEqualStrings(question, event.question);
        try testing.expectEqual(@as(usize, 7), event.counter);
    }
    try testing.expectEqual(@as(usize, 4), events.calls.items.len);
    for (events.calls.items, 0..) |call, i| {
        try testing.expectEqualStrings(if (i % 2 == 0) "teacher" else "student", call.lm);
        try testing.expectEqual(@as(usize, 2), call.questions.len);
        try testing.expectEqual(@as(usize, 1), call.answers.len);
    }
    try testing.expectEqualStrings("teacher-old", events.calls.items[0].questions[0]);
    try testing.expectEqualStrings("red-train|teacher-mutated", events.calls.items[1].questions[0]);
    try testing.expectEqualStrings("teacher:red-train|teacher-mutated:1", events.calls.items[1].answers[0]);
    try testing.expectEqualStrings("blue-train|teacher-mutated", events.calls.items[3].questions[0]);
    try testing.expectEqualStrings("teacher:blue-train|teacher-mutated:1", events.calls.items[3].answers[0]);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    for (trace.steps.items, [_][]const u8{ "red-query|captured-mutated", "blue-query|captured-mutated" }) |step, question| {
        try testing.expectEqualStrings(question, step.inputs.get("question").?.string);
        try testing.expectEqual(@as(usize, 1), step.predictor.demos.len);
        try testing.expect(step.predictor.demos[0].get("augmented").?.bool);
    }
    try testing.expect(trace.steps.items[0].predictor != trace.steps.items[1].predictor);
}

fn expectIndependentCapture(
    comptime Optimizer: type,
    optimizer: *const Optimizer,
    arena: Allocator,
    ctx: Context,
    events: *TestEvents,
) !void {
    var original: TestProgram = .{ .marker = "second", .events = events };
    original.qa.state.demos = try testDemo(arena, "second-old");
    var independent = try optimizer.compile(arena, testing.allocator, ctx, &original, null);
    original.marker = "second-captured";
    independent.visible.marker = "second-visible";
    const output = try independent.forward(ctx, .{ .question = "red-query" });
    try testing.expectEqualStrings("student:red-query|second-captured:1", output.answer);
    try testing.expectEqual(@as(usize, 0), independent.visible.qa.state.demos.len);
    try testing.expectEqual(@as(usize, 6), events.forwards.items.len);
    try testing.expectEqualStrings("second-captured", events.forwards.items[4].marker);
    try testing.expectEqualStrings("second-captured", events.forwards.items[5].marker);
    try testing.expectEqualStrings("second-old", events.calls.items[4].questions[0]);
    try testing.expectEqualStrings("red-train|second-captured", events.calls.items[5].questions[0]);
}

test "KNNFewShot transparent discovery keeps direct Predict self and nested original names" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var events: TestEvents = .{ .arena = arena };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &events, .embed = TestEvents.embed }, .{ .caching = false });
    defer model.deinit();
    var unused_lm: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = unused_lm.lm() };
    const P = predict.Predict(TestSignature);
    const optimizer = try KNNFewShot(P).init(arena, 1, try testTrainset(arena), &model, .{});
    var student: P = .{};
    student.state.demos = try testDemo(arena, "old");
    var compiled = try optimizer.compile(arena, testing.allocator, ctx, &student, null);
    const root = try predict.namedPredictors(arena, &compiled);
    try testing.expectEqual(@as(usize, 1), root.len);
    try testing.expectEqualStrings("self", root[0].name);
    try testing.expect(root[0].param.predict == &compiled.visible.state);
    const Outer = struct {
        adapted: @TypeOf(compiled),
        compiled: bool = false,
        pub const Input = P.Input;
        pub const Output = P.Output;
        pub fn forward(self: *const @This(), c: Context, input: Input) anyerror!Output {
            return self.adapted.forward(c, input);
        }
    };
    var outer: Outer = .{ .adapted = compiled };
    const nested = try predict.namedPredictors(arena, &outer);
    try testing.expectEqual(@as(usize, 1), nested.len);
    try testing.expectEqualStrings("adapted", nested[0].name);
    try testing.expect(nested[0].param.predict == &outer.adapted.visible.state);
    student.compiled = true;
    compiled = try optimizer.compile(arena, testing.allocator, ctx, &student, null);
    try testing.expect(compiled.compiled and compiled.visible.compiled);
    // A root Parameter remains discoverable even when its compiled flag is
    // set; a nested ordinary compiled module is frozen by predict.visit.
    const still_root = try predict.namedPredictors(arena, &compiled);
    try testing.expectEqual(@as(usize, 1), still_root.len);
    outer.adapted = compiled;
    const still_nested = try predict.namedPredictors(arena, &outer);
    try testing.expectEqual(@as(usize, 1), still_nested.len);
    try testing.expectEqualStrings("adapted", still_nested[0].name);
    try testing.expectError(error.StudentCompiled, compiled.forward(ctx, .{ .question = "red-query" }));
}

test "KNNFewShot propagates source bootstrap failure for an empty negative-k neighborhood" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var events: TestEvents = .{ .arena = arena };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &events, .embed = TestEvents.embed }, .{ .caching = false });
    defer model.deinit();
    var unused_lm: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = unused_lm.lm() };
    const P = predict.Predict(TestSignature);
    const optimizer = try KNNFewShot(P).init(arena, -2, try testTrainset(arena), &model, .{
        .max_bootstrapped_demos = 0,
        .max_labeled_demos = 0,
    });
    var student: P = .{};
    const program = try optimizer.compile(arena, testing.allocator, ctx, &student, null);
    // Actual DSPy raises UnboundLocalError inside BootstrapFewShot._bootstrap,
    // after preparing and mapping the original student/teacher, before any LM.
    try testing.expectError(error.EmptyBootstrapTrainset, program.forward(ctx, .{ .question = "red-query" }));
    try testing.expectEqual(@as(usize, 0), unused_lm.index);
    try testing.expectEqual(@as(usize, 2), events.embeddings.items.len);
}

test "KNNFewShot observes live optimizer options and retrieval state after compile" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var events: TestEvents = .{ .arena = arena };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &events, .embed = TestEvents.embed }, .{ .caching = false });
    defer model.deinit();
    var student_lm: TestLM = .{ .label = "student", .events = &events };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = student_lm.lm() };
    var optimizer = try KNNFewShot(TestProgram).init(arena, 1, try testTrainset(arena), &model, .{
        .max_bootstrapped_demos = 0,
        .max_labeled_demos = 1,
    });
    var student: TestProgram = .{ .marker = "live", .events = &events };
    const program = try optimizer.compile(arena, testing.allocator, ctx, &student, null);
    const with_demo = try program.forward(ctx, .{ .question = "red-query" });
    try testing.expectEqualStrings("student:red-query|live:1", with_demo.answer);
    try testing.expectEqualStrings("red-train", events.calls.items[0].questions[0]);
    optimizer.bootstrap.max_labeled_demos = 0;
    const without_demo = try program.forward(ctx, .{ .question = "red-query" });
    try testing.expectEqualStrings("student:red-query|live:0", without_demo.answer);
    try testing.expectEqual(@as(usize, 1), events.calls.items[1].questions.len);
    optimizer.knn.k = -2;
    try testing.expectError(error.EmptyBootstrapTrainset, program.forward(ctx, .{ .question = "red-query" }));
    try testing.expectEqual(@as(usize, 2), events.calls.items.len);
    try testing.expectEqual(@as(usize, 4), events.embeddings.items.len);
    try testing.expectEqual(@as(usize, 0), program.visible.qa.state.demos.len);
}

test "KNNFewShot validates live teacher metadata before empty-neighborhood bootstrap failure" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var events: TestEvents = .{ .arena = arena };
    var model = embedding.Embedder.initCustom(testing.allocator, .{ .context = &events, .embed = TestEvents.embed }, .{ .caching = false });
    defer model.deinit();
    var unused_lm: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = unused_lm.lm() };
    var optimizer = try KNNFewShot(TestProgram).init(arena, 1, try testTrainset(arena), &model, .{
        .max_bootstrapped_demos = 0,
        .max_labeled_demos = 0,
    });
    var student: TestProgram = .{ .marker = "student", .events = &events };
    var teacher: TestTeacher = .{ .marker = "teacher", .events = &events };
    const program = try optimizer.compileWithTeacher(TestTeacher, arena, testing.allocator, ctx, &student, &teacher);
    const Mutation = enum { instructions, desc, prefix };
    for ([_]Mutation{ .instructions, .desc, .prefix }) |mutation| {
        teacher.qa.state = student.qa.state;
        var descs = [_][]const u8{ student.qa.state.descs[0], student.qa.state.descs[1] };
        var prefixes = [_][]const u8{ student.qa.state.prefixes[0], student.qa.state.prefixes[1] };
        descs[0] = "A changed teacher description.";
        prefixes[0] = "Changed question:";
        switch (mutation) {
            .instructions => teacher.qa.state.instructions = "A changed teacher instruction.",
            .desc => teacher.qa.state.descs = &descs,
            .prefix => teacher.qa.state.prefixes = &prefixes,
        }
        // Runtime metadata can differ while the comptime field slice is shared.
        // The borrowed teacher is deliberately changed after wrapper compile.
        try testing.expect(student.qa.state.fields.ptr == teacher.qa.state.fields.ptr);
        for ([_]isize{ 1, -2 }) |k| {
            optimizer.knn.k = k;
            try testing.expectError(error.ProgramMismatch, program.forward(ctx, .{ .question = "red-query" }));
        }
    }
    // Signature equality compares text, not slice identity. An equivalent
    // teacher with separately allocated metadata reaches the source empty-set
    // failure, instead of being rejected by predictor mapping.
    teacher.qa.state = student.qa.state;
    teacher.qa.state.instructions = try arena.dupe(u8, student.qa.state.instructions);
    const descs = try arena.alloc([]const u8, student.qa.state.descs.len);
    const prefixes = try arena.alloc([]const u8, student.qa.state.prefixes.len);
    for (student.qa.state.descs, descs) |text, *copy| copy.* = try arena.dupe(u8, text);
    for (student.qa.state.prefixes, prefixes) |text, *copy| copy.* = try arena.dupe(u8, text);
    teacher.qa.state.descs = descs;
    teacher.qa.state.prefixes = prefixes;
    optimizer.knn.k = -2;
    try testing.expectError(error.EmptyBootstrapTrainset, program.forward(ctx, .{ .question = "red-query" }));
    try testing.expectEqual(@as(usize, 0), unused_lm.index);
    try testing.expectEqual(@as(usize, 0), events.forwards.items.len);
    try testing.expectEqualStrings(student.qa.state.instructions, program.visible.qa.state.instructions);
}
