//! Compares native `LabeledFewShot` / `BootstrapFewShotWithRandomSearch` compile
//! options against an actual DSPy 3.5 receipt written by
//! tools/dspy_reference/fewshot_options_oracle.py. Only the public `dszig` API is
//! used. Expectations come from the receipt file, never from this program.
//!
//! Counts are reported separately and are not interchangeable:
//! - matched: a Source value equal to the native value;
//! - refusals: Source raised and native returned the mapped error category
//!   (Python exception classes and messages are not reproduced);
//! - ownership differences: Source aliases the caller's Example objects where
//!   native owns copies, a deliberate difference that is never a match;
//! - mismatched / unsupported: failures, the process exits with an error.
const std = @import("std");
const dszig = @import("dszig");
const Allocator = std.mem.Allocator;
const Value = dszig.Value;
const Record = dszig.value.Record;
const Example = dszig.Example;
const RandomSearch = dszig.BootstrapFewShotWithRandomSearch;
const optimizers = dszig.better_together.optimizers;

const Signature = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

var forward_calls: u32 = 0;
var metric_calls: u32 = 0;

/// Native mirror of the oracle's DemoProgram.
const Program = struct {
    z: dszig.Predict(Signature) = .{},
    a: dszig.Predict(Signature) = .{},
    compiled: bool = false,

    pub const Input = Signature.Input;
    pub const Output = struct { answer: []const u8, z: i64, a: i64, nz: i64, na: i64 };

    pub fn forward(self: *const Program, ctx: dszig.Context, input: Input) !Output {
        forward_calls += 1;
        if (ctx.trace) |trace| {
            const inputs = try dszig.value.toRecord(ctx.arena, Input, input);
            const outputs = try dszig.value.toRecord(ctx.arena, Signature.Output, .{ .answer = input.question });
            try trace.record(&self.z.state, &inputs, &outputs);
            try trace.record(&self.a.state, &inputs, &outputs);
        }
        return .{
            .answer = input.question,
            .z = try firstAnswer(self.z.state.demos),
            .a = try firstAnswer(self.a.state.demos),
            .nz = @intCast(self.z.state.demos.len),
            .na = @intCast(self.a.state.demos.len),
        };
    }

    fn firstAnswer(demos: []const Record) !i64 {
        if (demos.len == 0) return -1;
        return std.fmt.parseInt(i64, demos[0].get("answer").?.string, 10);
    }
};

/// Native mirror of the oracle's EmptyProgram: no predictors.
const Bare = struct {
    compiled: bool = false,

    pub const Input = Signature.Input;
    pub const Output = Signature.Output;

    pub fn forward(_: *const Bare, _: dszig.Context, input: Input) !Output {
        forward_calls += 1;
        return .{ .answer = input.question };
    }
};

fn metric(_: Allocator, example: *const Example, prediction: ?*const dszig.optimize.RecordPrediction, trace: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    metric_calls += 1;
    if (trace != null) return 1;
    const row = prediction orelse return error.ParseValue;
    const question = std.fmt.parseInt(i64, example.record.get("question").?.string, 10) catch return error.ParseValue;
    const sum = 2 * row.get("z").?.integer + row.get("a").?.integer + row.get("nz").?.integer + question;
    return @floatFromInt(@mod(sum, 7));
}

fn tiedMetric(_: Allocator, _: *const Example, _: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    metric_calls += 1;
    return 1;
}

fn teacherMetric(_: Allocator, _: *const Example, prediction: ?*const dszig.optimize.RecordPrediction, trace: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    metric_calls += 1;
    const row = prediction orelse return error.ParseValue;
    if (trace == null) return @floatFromInt(row.get("nz").?.integer);
    return if (row.get("z").?.integer == 88 and row.get("a").?.integer == 77) 1 else 0;
}

const Tally = struct {
    matched: u32 = 0,
    refusals: u32 = 0,
    ownership_differences: u32 = 0,
    mismatched: u32 = 0,
    unsupported: u32 = 0,
    writer: *std.Io.Writer,

    fn check(self: *Tally, ok: bool, case: []const u8, what: []const u8) !void {
        if (ok) {
            self.matched += 1;
        } else {
            self.mismatched += 1;
            try self.writer.print("MISMATCH {s}: {s}\n", .{ case, what });
        }
    }

    /// Source raised: native must return exactly the mapped error category.
    fn refusal(self: *Tally, source_type: []const u8, native: ?[]const u8, case: []const u8) !void {
        const expected = nativeError(source_type) orelse {
            self.unsupported += 1;
            try self.writer.print("UNSUPPORTED {s}: Source {s}\n", .{ case, source_type });
            return;
        };
        const actual = native orelse "success";
        if (std.mem.eql(u8, expected, actual)) {
            self.refusals += 1;
        } else {
            self.mismatched += 1;
            try self.writer.print("MISMATCH {s}: Source {s}, native {s}\n", .{ case, source_type, actual });
        }
    }
};

/// Error categories only; Python classes and messages are not reproduced.
fn nativeError(source_type: []const u8) ?[]const u8 {
    const pairs = [_][2][]const u8{
        .{ "LabeledValueError", "InvalidSampleCount" },
        .{ "RandomRangeValueError", "InvalidBootstrapRange" },
        .{ "ValueError", "RestrictMatchesNoSeed" },
        .{ "AttributeError", "OptimizerTeacherListUnsupported" },
        .{ "UnboundLocalError", "EmptyBootstrapTrainset" },
    };
    for (pairs) |pair| {
        if (std.mem.eql(u8, pair[0], source_type)) return pair[1];
    }
    return null;
}

fn field(row: Value, name: []const u8) !Value {
    if (row != .object) return error.InvalidReceipt;
    return row.object.get(name) orelse error.InvalidReceipt;
}

fn text(row: Value, name: []const u8) ![]const u8 {
    const item = try field(row, name);
    if (item != .string) return error.InvalidReceipt;
    return item.string;
}

fn flag(row: Value, name: []const u8) !bool {
    const item = try field(row, name);
    if (item != .bool) return error.InvalidReceipt;
    return item.bool;
}

fn integer(item: Value) !i64 {
    if (item != .integer) return error.InvalidReceipt;
    return item.integer;
}

fn count(row: Value, name: []const u8) !usize {
    return std.math.cast(usize, try integer(try field(row, name))) orelse error.InvalidReceipt;
}

fn number(item: Value) !f64 {
    return switch (item) {
        .float => |float| float,
        .integer => |int| @floatFromInt(int),
        else => error.InvalidReceipt,
    };
}

fn sourceErrorType(row: Value) !?[]const u8 {
    const item = try field(row, "error");
    if (item == .null) return null;
    return try text(item, "type");
}

fn examplesOf(arena: Allocator, values: []const i64) ![]Example {
    const rows = try arena.alloc(Example, values.len);
    for (rows, values) |*row, item| {
        const answer = try std.fmt.allocPrint(arena, "{d}", .{item});
        var record: Record = .empty;
        try record.put(arena, "question", .{ .string = answer });
        try record.put(arena, "answer", .{ .string = answer });
        row.* = .{ .record = record, .input_keys = &.{"question"} };
    }
    return rows;
}

fn examplesTo(arena: Allocator, length: usize) ![]Example {
    const values = try arena.alloc(i64, length);
    for (values, 0..) |*item, index| item.* = @intCast(index);
    return examplesOf(arena, values);
}

fn integers(arena: Allocator, item: Value) ![]const i64 {
    if (item != .array) return error.InvalidReceipt;
    const values = try arena.alloc(i64, item.array.items.len);
    for (values, item.array.items) |*target, source| target.* = try integer(source);
    return values;
}

fn presetDemo(arena: Allocator, answer: []const u8) ![]const Record {
    const demos = try arena.alloc(Record, 1);
    demos[0] = .empty;
    try demos[0].put(arena, "question", .{ .string = "preset" });
    try demos[0].put(arena, "answer", .{ .string = answer });
    return demos;
}

fn studentProgram(arena: Allocator) !Program {
    var student: Program = .{};
    student.z.state.demos = try presetDemo(arena, "99");
    return student;
}

fn sameAnswers(expected: Value, demos: []const Record) bool {
    if (expected != .array) return false;
    if (expected.array.items.len != demos.len) return false;
    for (expected.array.items, demos) |want, demo| {
        if (want != .string) return false;
        const got = demo.get("answer") orelse return false;
        if (got != .string) return false;
        if (!std.mem.eql(u8, want.string, got.string)) return false;
    }
    return true;
}

fn checkDemos(tally: *Tally, case: []const u8, expected: Value, program: *const Program) !void {
    try tally.check(sameAnswers(try field(expected, "z"), program.z.state.demos), case, "z demos");
    try tally.check(sameAnswers(try field(expected, "a"), program.a.state.demos), case, "a demos");
}

fn labeledRow(arena: Allocator, tally: *Tally, row: Value, case: []const u8) !void {
    const optimizer: dszig.LabeledFewShot = .{ .k = try std.fmt.parseInt(i128, try text(row, "k"), 10) };
    const options: dszig.LabeledFewShot.Options = .{ .sample = try flag(row, "sample") };
    const trainset = try examplesTo(arena, try count(row, "train"));
    const source_error = try sourceErrorType(row);
    if (try flag(row, "empty_program")) {
        const failure: ?[]const u8 = if (optimizer.compile(Bare, arena, .{}, trainset, options)) |_| null else |err| @errorName(err);
        if (source_error) |_| {
            try tally.refusal("LabeledValueError", failure, case);
        } else {
            try tally.check(failure == null, case, "accepted without predictors");
        }
        return;
    }
    const student = try studentProgram(arena);
    const compiled = optimizer.compile(Program, arena, student, trainset, options) catch |err| {
        if (source_error) |_| {
            try tally.refusal("LabeledValueError", @errorName(err), case);
        } else {
            try tally.check(false, case, @errorName(err));
        }
        return;
    };
    if (source_error) |_| return tally.refusal("LabeledValueError", null, case);
    try checkDemos(tally, case, try field(row, "demos"), &compiled);
    try tally.check(sameAnswers(try field(row, "original_demos"), student.z.state.demos), case, "caller demos");
    const aliases = try field(row, "prefix_aliases_trainset");
    if (aliases == .bool) {
        // Native compiled demos are owned copies in both modes.
        if (aliases.bool) tally.ownership_differences += 1;
    }
}

const SearchOutcome = struct {
    failure: ?[]const u8 = null,
    program: Program = .{},
    candidates: []const RandomSearch.Candidate = &.{},
};

fn searchOutcome(result: anytype) SearchOutcome {
    const compiled = result catch |err| return .{ .failure = @errorName(err) };
    return .{ .program = compiled.program, .candidates = compiled.candidates };
}

fn searchConfig(row: Value) !RandomSearch {
    const name = try text(row, "metric");
    const stop = try field(row, "stop_at_score");
    return .{
        .metric = if (std.mem.eql(u8, name, "metric"))
            metric
        else if (std.mem.eql(u8, name, "tied"))
            tiedMetric
        else if (std.mem.eql(u8, name, "teacher"))
            teacherMetric
        else
            return error.InvalidReceipt,
        .max_bootstrapped_demos = try count(row, "max_bootstrapped_demos"),
        .max_labeled_demos = try count(row, "max_labeled_demos"),
        .num_candidate_programs = try count(row, "num_candidate_programs"),
        .stop_at_score = if (stop == .null) null else try number(stop),
    };
}

fn searchOptions(arena: Allocator, row: Value) !RandomSearch.Options {
    const restrict = try field(row, "restrict");
    return .{
        .restrict = if (restrict == .null) null else try integers(arena, restrict),
        .labeled_sample = try flag(row, "labeled_sample"),
    };
}

fn searchRun(arena: Allocator, gpa: Allocator, row: Value) !SearchOutcome {
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .arena = arena, .lm = silent.lm() };
    const search = try searchConfig(row);
    const options = try searchOptions(arena, row);
    const student = try studentProgram(arena);
    const trainset = try examplesTo(arena, try count(row, "train"));
    const val = try field(row, "val");
    const valset: ?[]const Example = if (val == .null) null else try examplesOf(arena, try integers(arena, val));
    const kind = try text(row, "teacher");
    forward_calls = 0;
    metric_calls = 0;
    if (std.mem.eql(u8, kind, "none")) {
        return searchOutcome(search.compile(Program, arena, gpa, ctx, student, null, trainset, valset, options));
    }
    if (std.mem.eql(u8, kind, "program")) {
        var teacher: Program = .{ .compiled = true };
        teacher.z.state.demos = try presetDemo(arena, "88");
        teacher.a.state.demos = try presetDemo(arena, "77");
        return searchOutcome(search.compile(Program, arena, gpa, ctx, student, teacher, trainset, valset, options));
    }
    const list: []const Program = &.{.{}};
    if (std.mem.eql(u8, kind, "list")) {
        return searchOutcome(search.compile(Program, arena, gpa, ctx, student, list, trainset, valset, options));
    }
    if (std.mem.eql(u8, kind, "empty_list")) {
        return searchOutcome(search.compile(Program, arena, gpa, ctx, student, list[0..0], trainset, valset, options));
    }
    return error.InvalidReceipt;
}

fn sameCandidates(expected: Value, actual: []const RandomSearch.Candidate) !bool {
    if (expected != .array) return error.InvalidReceipt;
    if (expected.array.items.len != actual.len) return false;
    for (expected.array.items, actual) |want, got| {
        if (try integer(try field(want, "seed")) != got.seed) return false;
        if (try number(try field(want, "score")) != got.score) return false;
    }
    return true;
}

fn searchRow(arena: Allocator, gpa: Allocator, tally: *Tally, row: Value) !void {
    const case = try text(row, "name");
    const outcome = try searchRun(arena, gpa, row);
    // Calls made before a refusal are part of Source's observable ordering.
    try tally.check(try count(row, "forward_calls") == forward_calls, case, "forward calls");
    try tally.check(try count(row, "metric_calls") == metric_calls, case, "metric calls");
    if (try sourceErrorType(row)) |source_type| {
        const source_error = try field(row, "error");
        const random_range = std.mem.eql(u8, source_type, "ValueError") and
            std.mem.startsWith(u8, try text(source_error, "message"), "empty range for randrange()");
        return tally.refusal(if (random_range) "RandomRangeValueError" else source_type, outcome.failure, case);
    }
    if (outcome.failure) |name| return tally.check(false, case, name);
    try tally.check(try sameCandidates(try field(row, "candidates"), outcome.candidates), case, "candidates");
    try checkDemos(tally, case, try field(row, "best_demos"), &outcome.program);
}

const meta: dszig.BetterTogether = .{
    .metric = metric,
    .limits = .{ .max_steps = 2, .max_examples = 16, .max_predictors = 2, .max_teachers = 2, .max_strategy_bytes = 64 },
};

fn metaOptions(strategy: []const u8) dszig.better_together.Options {
    // No validation split and no shuffle: the single step sees the Source trainset.
    return .{ .seed = 0, .valset_ratio = 0, .shuffle_trainset_between_steps = false, .strategy = strategy };
}

/// The same Source rows through a BetterTogether registry key's compile options.
fn metaLabeledRow(arena: Allocator, init: std.process.Init, tally: *Tally, row: Value, case: []const u8) !void {
    if (try flag(row, "empty_program")) return;
    const train = try count(row, "train");
    // BetterTogether itself refuses an empty trainset before any optimizer.
    if (train == 0) return;
    var model: dszig.lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var student = try studentProgram(arena);
    try dszig.predict.setLM(arena, &student, model.lm());
    var labeled: optimizers.LabeledFewShot = .{
        .optimizer = .{ .k = try std.fmt.parseInt(i128, try text(row, "k"), 10) },
        .compile_options = .{ .sample = try flag(row, "sample") },
    };
    const result = try meta.compileWith(Program, arena, init.gpa, init.io, .{ .arena = arena, .lm = model.lm() }, &student, try examplesTo(arena, train), metaOptions("labeled"), .{ .labeled = &labeled });
    if (try sourceErrorType(row)) |_| {
        // The step's error is recorded by BetterTogether, not propagated.
        return tally.check(result.flag_compilation_error_occurred, case, "BetterTogether step refusal");
    }
    try tally.check(!result.flag_compilation_error_occurred, case, "BetterTogether step success");
    try checkDemos(tally, case, try field(row, "demos"), &result.program);
}

/// Default BetterTogether prompt step with typed prompt compile options. Only
/// rows without validation data and without a teacher are comparable: the
/// default strategy's own candidate selection then keeps the last program.
fn metaSearchRow(arena: Allocator, init: std.process.Init, tally: *Tally, row: Value) !void {
    if ((try field(row, "val")) != .null) return;
    if (!std.mem.eql(u8, try text(row, "teacher"), "none")) return;
    if (try sourceErrorType(row)) |_| return;
    const case = try text(row, "name");
    var model: dszig.lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    var student = try studentProgram(arena);
    try dszig.predict.setLM(arena, &student, model.lm());
    const defaults: dszig.better_together.Defaults = .{
        .weight_limits = .{ .max_teachers = 1, .max_examples = 16, .max_predictors = 2, .data = .{ .max_rows = 16, .max_trace_steps = 32 } },
        .weight_slot_sets = &.{},
        .prompt = try searchConfig(row),
        .prompt_compile_options = try searchOptions(arena, row),
    };
    const result = try meta.compile(Program, arena, init.gpa, init.io, .{ .arena = arena, .lm = model.lm() }, &student, try examplesTo(arena, try count(row, "train")), metaOptions("p"), defaults);
    try tally.check(!result.flag_compilation_error_occurred, case, "default prompt step success");
    try checkDemos(tally, case, try field(row, "best_demos"), &result.program);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 2) {
        std.log.err("usage: fewshot_options_consumer <fewshot-options-source.json>", .{});
        return error.InvalidArguments;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], arena, .limited(16 * 1024 * 1024));
    const receipt = try std.json.parseFromSliceLeaky(Value, arena, bytes, .{});
    if (!std.mem.eql(u8, try text(receipt, "dspy"), "3.5.0")) return error.IncompatibleReceipt;
    const labeled = try field(receipt, "labeled");
    const search = try field(receipt, "search");
    if (labeled != .array or search != .array) return error.InvalidReceipt;

    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    var direct: Tally = .{ .writer = &stdout.interface };
    var composed: Tally = .{ .writer = &stdout.interface };
    for (labeled.array.items, 0..) |row, index| {
        const case = try std.fmt.allocPrint(arena, "labeled[{d}] k={s} sample={}", .{ index, try text(row, "k"), try flag(row, "sample") });
        try labeledRow(arena, &direct, row, case);
        try metaLabeledRow(arena, init, &composed, row, case);
    }
    for (search.array.items) |row| {
        try searchRow(arena, init.gpa, &direct, row);
        try metaSearchRow(arena, init, &composed, row);
    }
    for ([_]struct { name: []const u8, tally: *const Tally }{
        .{ .name = "direct", .tally = &direct },
        .{ .name = "BetterTogether", .tally = &composed },
    }) |section| {
        try stdout.interface.print("{s}: {d} matched, {d} refusal categories matched, {d} ownership differences, {d} mismatched, {d} unsupported\n", .{
            section.name,             section.tally.matched,
            section.tally.refusals,   section.tally.ownership_differences,
            section.tally.mismatched, section.tally.unsupported,
        });
    }
    try stdout.interface.print("rows: {d} labeled, {d} search\n", .{ labeled.array.items.len, search.array.items.len });
    try stdout.interface.flush();
    if (direct.mismatched + direct.unsupported + composed.mismatched + composed.unsupported > 0) return error.SourceMismatch;
}
