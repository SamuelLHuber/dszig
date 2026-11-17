//! Native `dspy.MultiChainComparison` (DSPy 3.5.0).
//!
//! ```zig
//! const Compare = MultiChainComparison(QA, 2);
//! const compare: Compare = .{ .temperature = 0.9 };
//! const result = try compare.forward(ctx, completions, .{ .question = "Q" });
//! ```
//!
//! `completions` are ordered `value.Record`s. The original signature's last
//! output supplies each proposed answer. `Input` remains the original typed
//! input; `Output` is the comparison signature's output, including `rationale`.
//! Context LM options take precedence over the module's temperature (0.7).
//! The embedded `predict` is a normal typed predictor, visible to discovery,
//! state serialization, demos, optimizer edits, and tracing.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Field = signature.Field;
const Value = @import("runtime_value.zig").Value;
const Record = value.Record;

const lm = @import("lm.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const signature = @import("signature.zig");
const unicode = @import("unicode.zig");
const value = @import("value.zig");

pub const Error = predict.Error || error{AttemptCountMismatch};

const FieldAttributes = @TypeOf(@typeInfo(struct { field: u8 }).@"struct".field_attrs[0]);
const rationale_prefix = "Accurate Reasoning: Thank you everyone. Let's now holistically";

/// Compare exactly `M` proposals using a synthesized signature and typed Predict.
/// An explicit null temperature leaves the LM default untouched. A nonnull
/// `ctx.lm_options.temperature` overrides this module's configured temperature.
/// Empty proposals are valid when `M == 0`, as in upstream DSPy.
pub fn MultiChainComparison(comptime Sig: type, comptime M: usize) type {
    comptime {
        if (!signature.isSignature(Sig)) @compileError("dszig: MultiChainComparison requires a signature");
        if (Sig.output_fields_list.len == 0) @compileError("dszig: MultiChainComparison requires an original output field");
    }
    const ExtendedSig = comparisonSignature(Sig, M);
    return struct {
        const Self = @This();
        pub const Extended = ExtendedSig;
        pub const Input = Sig.Input;
        pub const Output = Extended.Output;
        pub const attempt_count = M;
        pub const last_key = Sig.output_fields_list[Sig.output_fields_list.len - 1].name;

        predict: predict.Predict(Extended) = .{},
        temperature: ?f64 = 0.7,
        compiled: bool = false,

        pub fn forward(self: *const Self, ctx_in: predict.Context, completions: []const Record, input: Input) Error!Output {
            var attempts: [M][]const u8 = undefined;
            // Upstream formats every proposal before asserting the count. A
            // malformed extra proposal must therefore not become a count error.
            for (completions, 0..) |*completion, i| {
                const attempt = try formatAttempt(ctx_in.arena, completion, last_key);
                if (i < M) attempts[i] = attempt;
            }
            if (completions.len != M) return error.AttemptCountMismatch;

            var kwargs: Extended.Input = undefined;
            inline for (@typeInfo(Input).@"struct".field_names) |name| {
                @field(kwargs, name) = @field(input, name);
            }
            inline for (0..M) |i| {
                const name = comptime attemptName(i);
                if (comptime !@hasField(Input, name)) @field(kwargs, name) = attempts[i];
            }
            var ctx = ctx_in;
            if (ctx.lm_options.temperature == null) ctx.lm_options.temperature = self.temperature;
            return self.predict.forward(ctx, kwargs);
        }
    };
}

fn attemptName(comptime i: usize) [:0]const u8 {
    return std.fmt.comptimePrint("reasoning_attempt_{d}", .{i + 1});
}

fn attemptIndex(comptime name: []const u8, comptime M: usize) ?usize {
    for (0..M) |i| if (std.mem.eql(u8, name, attemptName(i))) return i;
    return null;
}

fn findField(comptime fields: []const Field, comptime name: []const u8) ?Field {
    for (fields) |field| if (std.mem.eql(u8, field.name, name)) return field;
    return null;
}

fn stringField(comptime name: []const u8, comptime prefix: []const u8, comptime desc: []const u8) Field {
    return .{ .name = name, .kind = .str, .type_name = "str", .prefix = prefix, .desc = desc };
}

fn attemptField(comptime i: usize) Field {
    return stringField(attemptName(i), std.fmt.comptimePrint("Student Attempt #{d}:", .{i + 1}), "${reasoning attempt}");
}

fn comparisonInputCount(comptime Sig: type, comptime M: usize) usize {
    var n: usize = 0;
    for (Sig.input_fields_list) |field| if (!std.mem.eql(u8, field.name, "rationale")) {
        n += 1;
    };
    for (0..M) |i| {
        const name = attemptName(i);
        if (!@hasField(Sig.Input, name) and !@hasField(Sig.Output, name)) n += 1;
    }
    return n;
}

fn comparisonInputs(comptime Sig: type, comptime M: usize) [comparisonInputCount(Sig, M)]Field {
    var fields: [comparisonInputCount(Sig, M)]Field = undefined;
    var n: usize = 0;
    for (Sig.input_fields_list) |field| {
        if (std.mem.eql(u8, field.name, "rationale")) continue;
        fields[n] = if (attemptIndex(field.name, M)) |i| attemptField(i) else field;
        n += 1;
    }
    for (0..M) |i| {
        const name = attemptName(i);
        if (@hasField(Sig.Input, name) or @hasField(Sig.Output, name)) continue;
        fields[n] = attemptField(i);
        n += 1;
    }
    return fields;
}

fn comparisonOutputs(comptime Sig: type, comptime M: usize) [Sig.output_fields_list.len + @intFromBool(!@hasField(Sig.Output, "rationale"))]Field {
    var fields: [Sig.output_fields_list.len + @intFromBool(!@hasField(Sig.Output, "rationale"))]Field = undefined;
    fields[0] = findField(Sig.output_fields_list, "rationale") orelse stringField("rationale", rationale_prefix, "${corrected reasoning}");
    var n: usize = 1;
    // Signature.insert builds dict(input_fields + output_fields). An attempt
    // colliding with an output keeps that output's metadata but moves it to the
    // front. Successive appends reverse the order of those colliding outputs.
    for (0..M) |offset| {
        if (findField(Sig.output_fields_list, attemptName(M - offset - 1))) |field| {
            fields[n] = field;
            n += 1;
        }
    }
    for (Sig.output_fields_list) |field| {
        if (std.mem.eql(u8, field.name, "rationale") or attemptIndex(field.name, M) != null) continue;
        fields[n] = field;
        n += 1;
    }
    return fields;
}

fn structFieldIndex(comptime T: type, comptime name: []const u8) usize {
    for (@typeInfo(T).@"struct".field_names, 0..) |field_name, i| {
        if (std.mem.eql(u8, name, field_name)) return i;
    }
    unreachable;
}

fn inputCarrier(comptime Sig: type, comptime M: usize) type {
    comptime {
        const info = @typeInfo(Sig.Input).@"struct";
        var count = M;
        for (info.field_names) |name| if (attemptIndex(name, M) == null) {
            count += 1;
        };
        var names: [count][:0]const u8 = undefined;
        var types: [count]type = undefined;
        var attrs: [count]FieldAttributes = undefined;
        for (0..M) |i| {
            const name = attemptName(i);
            names[i] = name;
            if (@hasField(Sig.Input, name)) {
                const original = structFieldIndex(Sig.Input, name);
                types[i] = info.field_types[original];
                attrs[i] = info.field_attrs[original];
            } else {
                types[i] = []const u8;
                attrs[i] = .{};
            }
        }
        var n = M;
        for (info.field_names, info.field_types, info.field_attrs) |name, T, attr| {
            if (attemptIndex(name, M) != null) continue;
            names[n] = name;
            types[n] = T;
            attrs[n] = attr;
            n += 1;
        }
        return @Struct(.auto, null, &names, &types, &attrs);
    }
}

fn outputStruct(comptime Sig: type, comptime fields: []const Field) type {
    comptime {
        const info = @typeInfo(Sig.Output).@"struct";
        var names: [fields.len][:0]const u8 = undefined;
        var types: [fields.len]type = undefined;
        var attrs: [fields.len]FieldAttributes = undefined;
        for (fields, 0..) |field, i| {
            if (@hasField(Sig.Output, field.name)) {
                const original = structFieldIndex(Sig.Output, field.name);
                names[i] = info.field_names[original];
                types[i] = info.field_types[original];
                attrs[i] = info.field_attrs[original];
            } else {
                names[i] = "rationale";
                types[i] = []const u8;
                attrs[i] = .{};
            }
        }
        return @Struct(.auto, null, &names, &types, &attrs);
    }
}

fn comparisonSignature(comptime Sig: type, comptime M: usize) type {
    const inputs = comptime comparisonInputs(Sig, M);
    const outputs = comptime comparisonOutputs(Sig, M);
    const all = inputs ++ outputs;
    const InputT = inputCarrier(Sig, M);
    const OutputT = outputStruct(Sig, &outputs);
    return struct {
        // This carrier models Python kwargs, not just signature inputs. Thus
        // output-name collisions remain in traces but are ignored by adapters,
        // and original typed inputs override generated attempts without coercion.
        pub const Input = InputT;
        pub const Output = OutputT;
        pub const fields: []const Field = &all;
        pub const input_fields_list: []const Field = &inputs;
        pub const output_fields_list: []const Field = &outputs;
        pub const instructions = Sig.instructions;
        pub const is_dszig_signature = true;
    };
}

fn truthy(v: Value) error{ParseValue}!bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |n| n != 0,
        .float => |n| n != 0,
        .number_string => |s| (std.fmt.parseFloat(f64, s) catch return true) != 0,
        .string => |s| s.len > 0,
        .array => |a| a.items.len > 0,
        .object => |o| o.count() > 0,
        .custom => |custom| try custom.truthy(),
    };
}

/// Python strip is Unicode-aware; value.pyStrip intentionally covers ASCII.
/// Scan only the edges so ordinary strings need neither copying nor decoding.
fn strip(text: []const u8) []const u8 {
    var start: usize = 0;
    while (start < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[start]) catch break;
        if (len > text.len - start) break;
        const cp = std.unicode.utf8Decode(text[start..][0..len]) catch break;
        if (!unicode.isSpace(cp)) break;
        start += len;
    }
    var end = text.len;
    while (end > start) {
        var i = end - 1;
        while (i > start and text[i] & 0xc0 == 0x80) i -= 1;
        const cp = std.unicode.utf8Decode(text[i..end]) catch break;
        if (!unicode.isSpace(cp)) break;
        end = i;
    }
    return text[start..end];
}

fn firstLine(arena: Allocator, v: Value) Error![]const u8 {
    const text = if (v == .string) v.string else blk: {
        var rendered: std.Io.Writer.Allocating = .init(arena);
        pyjson.writeStr(&rendered.writer, v) catch |err| switch (err) {
            error.WriteFailed => return error.OutOfMemory,
            error.JsonTooDeep => return error.JsonTooDeep,
            error.ParseValue => return error.AdapterParse,
        };
        break :blk rendered.written();
    };
    const trimmed = strip(text);
    const end = std.mem.indexOfScalar(u8, trimmed, '\n') orelse trimmed.len;
    return strip(trimmed[0..end]);
}

fn formatAttempt(arena: Allocator, completion: *const Record, last_key: []const u8) Error![]const u8 {
    // `get` uses reasoning only when rationale is absent, not when it is null
    // or otherwise false. The subsequent `or ""` is a separate operation.
    const raw = completion.get("rationale") orelse completion.get("reasoning") orelse .null;
    const rationale = if (try truthy(raw)) try firstLine(arena, raw) else "";
    const answer = try firstLine(arena, completion.get(last_key) orelse return error.MissingField);
    return if (rationale.len > 0)
        std.fmt.allocPrint(arena, "«I'm trying to {s} I'm not sure but my prediction is {s}»", .{ rationale, answer })
    else
        std.fmt.allocPrint(arena, "«My prediction is {s}»", .{answer});
}

const testing = std.testing;

fn completionRecord(arena: Allocator, json: []const u8) !Record {
    const parsed = try std.json.parseFromSliceLeaky(Value, arena, json, .{ .allocate = .alloc_always });
    return parsed.object;
}

const OracleBase = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { analysis: []const u8, answer: i64 };
    pub const instructions = "Preserve the exact task instructions.";
    pub const descriptions = .{
        .question = "Original input description",
        .analysis = "Original analysis description",
        .answer = "Original answer description",
    };
});

const oracle_fields = blk: {
    var fields = OracleBase.fields[0..OracleBase.fields.len].*;
    fields[0].prefix = "ASK:";
    fields[1].prefix = "Analysis custom:";
    fields[2].prefix = "Answer custom:";
    break :blk fields;
};

const OracleSignature = struct {
    pub const Input = OracleBase.Input;
    pub const Output = OracleBase.Output;
    pub const fields: []const Field = &oracle_fields;
    pub const input_fields_list = fields[0..1];
    pub const output_fields_list = fields[1..];
    pub const instructions = OracleBase.instructions;
    pub const is_dszig_signature = true;
};

test "comparison preserves original metadata and synthesizes exact attempt and rationale fields" {
    const Compare = MultiChainComparison(OracleSignature, 2);
    const compare: Compare = .{};
    const fields = Compare.Extended.fields;
    try testing.expectEqualStrings(OracleSignature.instructions, compare.predict.state.instructions);
    try testing.expectEqual(@as(usize, 3), compare.predict.state.input_count);
    try testing.expectEqual(@as(usize, 6), fields.len);
    try testing.expectEqualDeep(OracleSignature.fields[0], fields[0]);
    try testing.expectEqualDeep(OracleSignature.fields[1], fields[4]);
    try testing.expectEqualDeep(OracleSignature.fields[2], fields[5]);
    try testing.expectEqualStrings("answer", Compare.last_key);
    inline for (0..2) |i| {
        const field = fields[i + 1];
        try testing.expectEqualStrings(attemptName(i), field.name);
        try testing.expectEqualStrings(std.fmt.comptimePrint("Student Attempt #{d}:", .{i + 1}), field.prefix);
        try testing.expectEqualStrings("${reasoning attempt}", field.desc);
        try testing.expectEqual(signature.Kind.str, field.kind);
        try testing.expect(field.default_value == null);
    }
    try testing.expectEqualStrings("rationale", fields[3].name);
    try testing.expectEqualStrings(rationale_prefix, fields[3].prefix);
    try testing.expectEqualStrings("${corrected reasoning}", fields[3].desc);
    try testing.expectEqual(@as(?f64, 0.7), compare.temperature);
}

test "comparison formatting matches Python fallback truthiness conversion strip and literal newline" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const cases = .{
        .{ "{\"rationale\":\" \\n use first \\nsecond\",\"reasoning\":\"ignored\",\"answer\":\" \\n 42 \\n99\"}", "«I'm trying to use first I'm not sure but my prediction is 42»" },
        .{ "{\"reasoning\":\" fallback \\nsecond\",\"answer\":true}", "«I'm trying to fallback I'm not sure but my prediction is True»" },
        .{ "{\"rationale\":null,\"reasoning\":\"not used\",\"answer\":null}", "«My prediction is None»" },
        .{ "{\"rationale\":\"\",\"reasoning\":\"not used\",\"answer\":false}", "«My prediction is False»" },
        .{ "{\"rationale\":false,\"answer\":0}", "«My prediction is 0»" },
        .{ "{\"rationale\":0.0,\"answer\":-0.0}", "«My prediction is -0.0»" },
        .{ "{\"rationale\":[],\"answer\":[true,null,\"x\"]}", "«My prediction is [True, None, 'x']»" },
        .{ "{\"rationale\":{},\"answer\":{}}", "«My prediction is {}»" },
        .{ "{\"rationale\":{\"k\":true},\"answer\":{\"x\":[1,null]}}", "«I'm trying to {'k': True} I'm not sure but my prediction is {'x': [1, None]}»" },
        .{ "{\"rationale\":\"\\r first\\rsecond\\nthird\",\"answer\":\"a\\rb\\nc\"}", "«I'm trying to first\rsecond I'm not sure but my prediction is a\rb»" },
        .{ "{\"rationale\":\"\\u2003\\u00a0why\\u2003\\nsecond\",\"answer\":\"\\u3000yes\\u0085\\nsecond\"}", "«I'm trying to why I'm not sure but my prediction is yes»" },
        .{ "{\"rationale\":\" \\n\\t\\nsecond\",\"answer\":\"\\n\\n\\t\"}", "«I'm trying to second I'm not sure but my prediction is »" },
        .{ "{\"rationale\":\" \\n \\t\",\"answer\":18446744073709551615}", "«My prediction is 18446744073709551615»" },
    };
    inline for (cases) |case| {
        const record = try completionRecord(arena, case[0]);
        try testing.expectEqualStrings(case[1], try formatAttempt(arena, &record, "answer"));
    }
    var nan_record: Record = .empty;
    try nan_record.put(arena, "rationale", .{ .float = std.math.nan(f64) });
    try nan_record.put(arena, "answer", .{ .float = std.math.inf(f64) });
    try testing.expectEqualStrings("«I'm trying to nan I'm not sure but my prediction is inf»", try formatAttempt(arena, &nan_record, "answer"));
    var zero_record: Record = .empty;
    try zero_record.put(arena, "rationale", .{ .number_string = "0" });
    try zero_record.put(arena, "answer", .{ .number_string = "18446744073709551615" });
    try testing.expectEqualStrings("«My prediction is 18446744073709551615»", try formatAttempt(arena, &zero_record, "answer"));
}

test "comparison requires exact count and preserves malformed proposal error precedence" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{ .responses = &.{}, .history = &history };
    var trace: predict.Trace = .{ .arena = arena };
    const ctx: predict.Context = .{ .arena = arena, .lm = script.lm(), .trace = &trace };
    const compare: MultiChainComparison(OracleSignature, 2) = .{};
    var records = [_]Record{
        try completionRecord(arena, "{\"answer\":1}"),
        try completionRecord(arena, "{\"answer\":2}"),
        try completionRecord(arena, "{\"answer\":3}"),
    };
    try testing.expectError(error.AttemptCountMismatch, compare.forward(ctx, records[0..0], .{ .question = "Q" }));
    try testing.expectError(error.AttemptCountMismatch, compare.forward(ctx, records[0..1], .{ .question = "Q" }));
    try testing.expectError(error.AttemptCountMismatch, compare.forward(ctx, &records, .{ .question = "Q" }));
    records[2] = .empty;
    try testing.expectError(error.MissingField, compare.forward(ctx, &records, .{ .question = "Q" }));
    try testing.expectError(error.MissingField, compare.forward(ctx, records[2..], .{ .question = "Q" }));
    try testing.expectEqual(@as(usize, 0), history.requests.items.len);
    try testing.expectEqual(@as(usize, 0), trace.steps.items.len);
}

test "comparison uses typed Predict with captured upstream prompt temperature trace and discovery" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const response = "<rationale>combined</rationale><analysis>analysis</analysis><answer>42</answer>";
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{ .responses = &.{ response, response, response, response }, .history = &history };
    var trace: predict.Trace = .{ .arena = arena };
    const ctx: predict.Context = .{ .arena = arena, .lm = script.lm(), .trace = &trace, .json_fallback = false };
    var compare: MultiChainComparison(OracleSignature, 2) = .{};
    const completions = [_]Record{
        try completionRecord(arena, "{\"rationale\":\"One\\nignored\",\"analysis\":\"not the answer\",\"answer\":41}"),
        try completionRecord(arena, "{\"reasoning\":\"Two\",\"analysis\":\"also ignored\",\"answer\":42}"),
    };
    const out = try compare.forward(ctx, &completions, .{ .question = "Q" });
    try testing.expectEqualStrings("combined", out.rationale);
    try testing.expectEqualStrings("analysis", out.analysis);
    try testing.expectEqual(@as(i64, 42), out.answer);
    try testing.expectEqual(@as(?f64, 0.7), history.requests.items[0].options.temperature);
    const request = history.requests.items[0];
    try testing.expectEqual(@as(usize, 2), request.messages.len);
    try testing.expectEqualStrings(
        "Your input fields are:\n" ++
            "1. `question` (str): Original input description\n" ++
            "2. `reasoning_attempt_1` (str): ${reasoning attempt}\n" ++
            "3. `reasoning_attempt_2` (str): ${reasoning attempt}\n" ++
            "Your output fields are:\n" ++
            "1. `rationale` (str): ${corrected reasoning}\n" ++
            "2. `analysis` (str): Original analysis description\n" ++
            "3. `answer` (int): Original answer description\n" ++
            "Output format (replace placeholders):\n" ++
            "<rationale>\n{rationale}\n</rationale>\n\n" ++
            "<analysis>\n{analysis}\n</analysis>\n\n" ++
            "<answer>\n{answer}\n</answer>\n" ++
            "In adhering to this structure, your objective is: \n" ++
            "        Preserve the exact task instructions.",
        request.messages[0].content.text,
    );
    try testing.expectEqualStrings(
        "<question>\nQ\n</question>\n\n" ++
            "<reasoning_attempt_1>\n«I'm trying to One I'm not sure but my prediction is 41»\n</reasoning_attempt_1>\n\n" ++
            "<reasoning_attempt_2>\n«I'm trying to Two I'm not sure but my prediction is 42»\n</reasoning_attempt_2>\n\n" ++
            "Output format (replace placeholders):\n" ++
            "<rationale>\n{rationale}\n</rationale>\n\n" ++
            "<analysis>\n{analysis}\n</analysis>\n\n" ++
            "<answer>\n{answer}\n</answer>\n\n" ++
            "Use this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.",
        request.messages[1].content.text,
    );
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    const step = trace.steps.items[0];
    try testing.expect(step.predictor == &compare.predict.state);
    try testing.expectEqualStrings("reasoning_attempt_1", step.inputs.keys()[0]);
    try testing.expectEqualStrings("reasoning_attempt_2", step.inputs.keys()[1]);
    try testing.expectEqualStrings("question", step.inputs.keys()[2]);
    try testing.expectEqualStrings("«I'm trying to Two I'm not sure but my prediction is 42»", step.inputs.get("reasoning_attempt_2").?.string);
    const predictors = try predict.namedPredictors(arena, &compare);
    try testing.expectEqual(@as(usize, 1), predictors.len);
    try testing.expectEqualStrings("predict", predictors[0].name);
    try testing.expect(predictors[0].param.predict == &compare.predict.state);

    compare.temperature = 0.91;
    _ = try compare.forward(ctx, &completions, .{ .question = "Q" });
    try testing.expectEqual(@as(?f64, 0.91), history.requests.items[1].options.temperature);
    var overridden = ctx;
    overridden.lm_options.temperature = 0.0;
    overridden.lm_options.rollout_id = .{ .integer = 17 };
    _ = try compare.forward(overridden, &completions, .{ .question = "Q" });
    try testing.expectEqual(@as(?f64, 0.0), history.requests.items[2].options.temperature);
    try testing.expectEqual(@as(i64, 17), history.requests.items[2].options.rollout_id.?.integer);
    compare.temperature = null;
    _ = try compare.forward(ctx, &completions, .{ .question = "Q" });
    try testing.expect(history.requests.items[3].options.temperature == null);
}

test "comparison input collisions override attempts without coercion and retain ignored kwargs in trace" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8, reasoning_attempt_1: i64 = 17, rationale: i64 = 11 };
        pub const Output = struct { answer: []const u8 };
        pub const descriptions = .{ .reasoning_attempt_1 = "Old attempt", .rationale = "Old rationale input" };
    });
    const Compare = MultiChainComparison(Sig, 1);
    try testing.expectEqual(@as(usize, 2), Compare.Extended.input_fields_list.len);
    try testing.expectEqualStrings("question", Compare.Extended.input_fields_list[0].name);
    try testing.expectEqualStrings("reasoning_attempt_1", Compare.Extended.input_fields_list[1].name);
    try testing.expectEqual(signature.Kind.str, Compare.Extended.input_fields_list[1].kind);
    try testing.expectEqualStrings("${reasoning attempt}", Compare.Extended.input_fields_list[1].desc);
    try testing.expect(Compare.Extended.input_fields_list[1].default_value == null);
    try testing.expectEqualStrings("rationale", Compare.Extended.output_fields_list[0].name);
    try testing.expectEqual(signature.Kind.str, Compare.Extended.output_fields_list[0].kind);
    try testing.expectEqualStrings(rationale_prefix, Compare.Extended.output_fields_list[0].prefix);
    try testing.expect(@FieldType(Compare.Extended.Input, "reasoning_attempt_1") == i64);
    try testing.expect(@FieldType(Compare.Extended.Input, "rationale") == i64);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{ .responses = &.{"<rationale>combined</rationale><answer>ok</answer>"}, .history = &history };
    var trace: predict.Trace = .{ .arena = arena };
    const compare: Compare = .{};
    const records = [_]Record{try completionRecord(arena, "{\"answer\":\"generated\"}")};
    const out = try compare.forward(.{ .arena = arena, .lm = script.lm(), .trace = &trace }, &records, .{ .question = "Q" });
    try testing.expectEqualStrings("ok", out.answer);
    const inputs = trace.steps.items[0].inputs;
    try testing.expectEqual(@as(i64, 17), inputs.get("reasoning_attempt_1").?.integer);
    try testing.expectEqual(@as(i64, 11), inputs.get("rationale").?.integer);
    const prompt = history.last().?.messages[1].content.text;
    try testing.expect(std.mem.indexOf(u8, prompt, "<reasoning_attempt_1>\n17\n</reasoning_attempt_1>") != null);
    try testing.expect(std.mem.indexOf(u8, prompt, "generated") == null);
    try testing.expect(std.mem.indexOf(u8, prompt, "<rationale>\n11\n</rationale>") == null);
}

test "comparison output collisions preserve metadata and reproduce upstream reverse promotion" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct {
            first: []const u8,
            reasoning_attempt_2: i64,
            middle: []const u8,
            reasoning_attempt_1: bool,
            last: []const u8,
        };
        pub const descriptions = .{ .reasoning_attempt_2 = "Original attempt output" };
    });
    const Compare = MultiChainComparison(Sig, 3);
    const expected = [_][]const u8{ "rationale", "reasoning_attempt_2", "reasoning_attempt_1", "first", "middle", "last" };
    for (expected, Compare.Extended.output_fields_list) |name, field| try testing.expectEqualStrings(name, field.name);
    try testing.expectEqualDeep(Sig.output_fields_list[1], Compare.Extended.output_fields_list[1]);
    try testing.expectEqualDeep(Sig.output_fields_list[3], Compare.Extended.output_fields_list[2]);
    try testing.expectEqualStrings("last", Compare.last_key);
    try testing.expectEqual(@as(usize, 2), Compare.Extended.input_fields_list.len);
    try testing.expectEqualStrings("reasoning_attempt_3", Compare.Extended.input_fields_list[1].name);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{ .responses = &.{"<rationale>combined</rationale><reasoning_attempt_2>2</reasoning_attempt_2><reasoning_attempt_1>true</reasoning_attempt_1><first>f</first><middle>m</middle><last>final</last>"}, .history = &history };
    var trace: predict.Trace = .{ .arena = arena };
    const compare: Compare = .{};
    const records = [_]Record{
        try completionRecord(arena, "{\"last\":\"a\"}"),
        try completionRecord(arena, "{\"last\":\"b\"}"),
        try completionRecord(arena, "{\"last\":\"c\"}"),
    };
    const out = try compare.forward(.{ .arena = arena, .lm = script.lm(), .trace = &trace }, &records, .{ .question = "Q" });
    try testing.expectEqual(@as(i64, 2), out.reasoning_attempt_2);
    try testing.expect(out.reasoning_attempt_1);
    try testing.expectEqualStrings("final", out.last);
    const inputs = trace.steps.items[0].inputs;
    try testing.expectEqualStrings("«My prediction is a»", inputs.get("reasoning_attempt_1").?.string);
    try testing.expectEqualStrings("«My prediction is b»", inputs.get("reasoning_attempt_2").?.string);
    const prompt = history.last().?.messages[1].content.text;
    try testing.expect(std.mem.indexOf(u8, prompt, "«My prediction is a»") == null);
    try testing.expect(std.mem.indexOf(u8, prompt, "«My prediction is b»") == null);
    try testing.expect(std.mem.indexOf(u8, prompt, "«My prediction is c»") != null);
}

test "comparison existing rationale output keeps its type default metadata and original last key" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8, rationale: i64 = 9 };
        pub const descriptions = .{ .rationale = "Original rationale" };
    });
    const Compare = MultiChainComparison(Sig, 1);
    try testing.expectEqualDeep(Sig.output_fields_list[1], Compare.Extended.output_fields_list[0]);
    try testing.expect(@FieldType(Compare.Output, "rationale") == i64);
    try testing.expectEqualStrings("rationale", Compare.last_key);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: lm.ScriptedLM = .{ .responses = &.{"<answer>ok</answer>"} };
    var trace: predict.Trace = .{ .arena = arena };
    const compare: Compare = .{};
    const records = [_]Record{try completionRecord(arena, "{\"answer\":\"ignored\",\"rationale\":7}")};
    const out = try compare.forward(.{ .arena = arena, .lm = script.lm(), .trace = &trace }, &records, .{ .question = "Q" });
    try testing.expectEqual(@as(i64, 9), out.rationale);
    try testing.expectEqualStrings("«I'm trying to 7 I'm not sure but my prediction is 7»", trace.steps.items[0].inputs.get("reasoning_attempt_1").?.string);
}

test "comparison selects original last output even when that output is promoted by an attempt collision" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { first: []const u8, reasoning_attempt_1: i64 };
    });
    const Compare = MultiChainComparison(Sig, 1);
    try testing.expectEqualStrings("reasoning_attempt_1", Compare.last_key);
    try testing.expectEqualStrings("first", Compare.Extended.output_fields_list[2].name);
    try testing.expectEqual(@as(usize, 1), Compare.Extended.input_fields_list.len);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{
        .responses = &.{"<rationale>combined</rationale><reasoning_attempt_1>7</reasoning_attempt_1><first>first output</first>"},
        .history = &history,
    };
    var trace: predict.Trace = .{ .arena = arena };
    const records = [_]Record{try completionRecord(arena, "{\"first\":\"not the selected answer\",\"reasoning_attempt_1\":42}")};
    const compare: Compare = .{};
    const out = try compare.forward(.{
        .arena = arena,
        .lm = script.lm(),
        .trace = &trace,
        .json_fallback = false,
    }, &records, .{ .question = "Q" });
    try testing.expectEqual(@as(i64, 7), out.reasoning_attempt_1);
    try testing.expectEqualStrings("«My prediction is 42»", trace.steps.items[0].inputs.get("reasoning_attempt_1").?.string);
    const prompt = history.last().?.messages[1].content.text;
    try testing.expect(std.mem.indexOf(u8, prompt, "«My prediction is 42»") == null);
    try testing.expect(std.mem.indexOf(u8, prompt, "not the selected answer") == null);
}

test "comparison state roundtrip preserves optimizer metadata demos and cheap-copy isolation" {
    const state = @import("state.zig");
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
        pub const instructions = "Original instructions.";
    });
    const Compare = MultiChainComparison(Sig, 1);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var compare: Compare = .{};
    compare.predict.state.instructions = "Optimizer instructions.";
    compare.predict.state.descs = &.{
        "${question}", "Optimizer attempt", "${corrected reasoning}", "${answer}",
    };
    compare.predict.state.prefixes = &.{
        "Question:", "Optimizer attempt:", rationale_prefix, "Answer:",
    };
    const demo = try completionRecord(arena, "{\"question\":\"demo Q\",\"reasoning_attempt_1\":\"demo attempt\",\"rationale\":\"demo rationale\",\"answer\":\"demo answer\"}");
    compare.predict.state.demos = &.{demo};
    const dumped = try state.dumpState(arena, &compare);
    const saved = dumped.object.get("predict").?.object;
    const saved_signature = saved.get("signature").?.object;
    const saved_fields = saved_signature.get("fields").?.array.items;
    try testing.expectEqual(@as(usize, 4), saved_fields.len);
    try testing.expectEqualStrings("Optimizer instructions.", saved_signature.get("instructions").?.string);
    try testing.expectEqualStrings("Optimizer attempt:", saved_fields[1].object.get("prefix").?.string);
    try testing.expectEqualStrings("Optimizer attempt", saved_fields[1].object.get("description").?.string);
    try testing.expectEqualStrings(rationale_prefix, saved_fields[2].object.get("prefix").?.string);

    var restored: Compare = .{};
    try state.loadState(arena, &restored, dumped, .{});
    try testing.expectEqualStrings(compare.predict.state.instructions, restored.predict.state.instructions);
    try testing.expectEqualDeep(compare.predict.state.descs, restored.predict.state.descs);
    try testing.expectEqualDeep(compare.predict.state.prefixes, restored.predict.state.prefixes);
    try testing.expect(restored.predict.state.fields.ptr == Compare.Extended.fields.ptr);
    try testing.expectEqual(@as(usize, 1), restored.predict.state.demos.len);
    try testing.expect(try value.recordEqual(&demo, &restored.predict.state.demos[0]));

    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{
        .responses = &.{"<rationale>combined</rationale><answer>ok</answer>"},
        .history = &history,
    };
    var trace: predict.Trace = .{ .arena = arena };
    const records = [_]Record{try completionRecord(arena, "{\"answer\":\"candidate\"}")};
    const out = try restored.forward(.{
        .arena = arena,
        .lm = script.lm(),
        .trace = &trace,
        .json_fallback = false,
    }, &records, .{ .question = "Q" });
    try testing.expectEqualStrings("ok", out.answer);
    const messages = history.last().?.messages;
    try testing.expectEqual(@as(usize, 4), messages.len);
    try testing.expect(std.mem.indexOf(u8, messages[0].content.text, "Optimizer instructions.") != null);
    try testing.expect(std.mem.indexOf(u8, messages[1].content.text, "demo Q") != null);
    try testing.expect(std.mem.indexOf(u8, messages[1].content.text, "demo attempt") != null);
    try testing.expectEqual(@import("adapter.zig").Role.assistant, messages[2].role);
    try testing.expect(std.mem.indexOf(u8, messages[2].content.text, "demo rationale") != null);
    try testing.expect(trace.steps.items[0].predictor == &restored.predict.state);

    var copy = restored;
    copy.predict.state.reset();
    try testing.expectEqual(@as(usize, 0), copy.predict.state.demos.len);
    try testing.expectEqual(@as(usize, 1), restored.predict.state.demos.len);
}

test "comparison zero attempts accepts empty original input and retains output defaults and state edits" {
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 = "default" };
        pub const instructions = "Original instructions.";
    });
    const Compare = MultiChainComparison(Sig, 0);
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{ .responses = &.{"<rationale>combined</rationale>"}, .history = &history };
    var compare: Compare = .{};
    compare.predict.state.instructions = "Optimizer instructions.";
    compare.predict.state.descs = &.{ "Optimizer rationale", "Optimizer answer" };
    compare.predict.state.prefixes = &.{ "Optimizer rationale:", "Optimizer answer:" };
    const out = try compare.forward(.{ .arena = arena, .lm = script.lm() }, &.{}, .{});
    try testing.expectEqualStrings("default", out.answer);
    const system = history.last().?.messages[0].content.text;
    try testing.expect(std.mem.indexOf(u8, system, "Optimizer instructions.") != null);
    try testing.expect(std.mem.indexOf(u8, system, "Optimizer rationale") != null);
    try testing.expect(std.mem.indexOf(u8, system, "Optimizer answer") != null);
}
