//! DSPy 3.5 Refine: typed attempts, source-backed feedback, scoped adapter hints.
//! The caller supplies program/reward source text explicitly: Zig does not have
//! Python's inspect.getsource. All borrowed source slices must outlive Self.
//! Attempt programs, signatures, advice, outputs and trace snapshots live in the
//! call arena. Mutable program data must be stored by value, like BestOfN.

const std = @import("std");
const predict = @import("predict.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const feedback = @import("feedback.zig");
const pyjson = @import("pyjson.zig");
const best_of_n = @import("best_of_n.zig");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Record = value.Record;
const Context = predict.Context;
const Trace = predict.Trace;

pub const OfferFeedback = signature.Signature(struct {
    pub const instructions =
        "In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if\n" ++
        "any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if\n" ++
        "it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.\n" ++
        "The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you\n" ++
        "to avoid the same mistake on the same or similar inputs.";
    pub const Input = struct {
        program_code: []const u8,
        modules_defn: []const u8,
        program_inputs: []const u8,
        program_trajectory: []const u8,
        program_outputs: []const u8,
        reward_code: []const u8,
        target_threshold: f64,
        reward_value: f64,
        module_names: []const []const u8,
    };
    pub const Output = struct { discussion: []const u8, advice: signature.StrDict };
    pub const descriptions = .{
        .program_code = "The code of the program that we are analyzing",
        .modules_defn = "The definition of each module in the program, including its I/O",
        .program_inputs = "The inputs to the program that we are analyzing",
        .program_trajectory = "The trajectory of the program's execution, showing each module's I/O",
        .program_outputs = "The outputs of the program that we are analyzing",
        .reward_code = "The code of the reward function that we are analyzing",
        .target_threshold = "The target threshold for the reward function",
        .reward_value = "The reward value assigned to the program's outputs",
        .module_names = "The names of the modules in the program, for which we seek advice",
        .discussion = "Discussing blame of where each module went wrong, if it did",
        .advice = "For each module, describe very concretely, in this order: the specific scenarios in which it has made " ++
            "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of" ++
            "scenario in the future. If the module is not to blame, write N/A.",
    };
});

pub fn Refine(comptime P: type) type {
    return struct {
        const Self = @This();
        pub const Input = P.Input;
        pub const Prediction = P.Output;
        pub const Output = ?Prediction;
        pub const Reward = *const fn (Context, *const Input, *const Prediction) anyerror!f64;
        pub const Options = struct {
            n: isize,
            reward_fn: Reward,
            threshold: ?Value,
            fail_count: ?i128 = null,
            module_code: []const u8,
            reward_code: []const u8,
        };

        module: P,
        n: isize,
        reward_fn: Reward,
        threshold: ?Value,
        fail_count: i128,
        module_code: []const u8,
        reward_code: []const u8,
        compiled: bool = false,

        pub fn init(module: P, options: Options) Self {
            const count = options.fail_count orelse 0;
            return .{
                .module = module,
                .n = options.n,
                .reward_fn = options.reward_fn,
                .threshold = options.threshold,
                .fail_count = if (count == 0) options.n else count,
                .module_code = options.module_code,
                .reward_code = options.reward_code,
            };
        }

        const Selection = struct {
            output: Output = null,
            steps: []const Trace.Step = &.{},
            reward: f64 = -std.math.inf(f64),
        };

        pub fn forward(self: *Self, ctx: Context, input: Input) anyerror!Output {
            if (self.n <= 0) return null;
            const start = ctx.lm_options.rollout_id orelse Value{ .integer = 0 };
            var best: Selection = .{};
            var advice: Record = .empty;
            for (0..@as(usize, @intCast(self.n))) |index| {
                var attempt_ctx = ctx;
                // LM.copy and program.deepcopy are outside the caught region.
                attempt_ctx.lm_options.rollout_id = try best_of_n.rolloutAt(ctx.arena, start, index);
                attempt_ctx.lm_options.temperature = 1.0;
                const module = try ctx.arena.create(P);
                module.* = self.module;
                const names = try predict.namedPredictors(ctx.arena, module);
                const stop = self.runIteration(ctx, attempt_ctx, module, names, &input, index, &best, &advice) catch |err| {
                    if (@as(i128, @intCast(index)) > self.fail_count) return err;
                    self.fail_count -= 1;
                    continue;
                };
                if (stop) break;
            }
            // Feedback runs on the outer trace naturally. Only the winning
            // program trace is appended here; reward calls remain isolated.
            if (ctx.trace) |trace| {
                for (best.steps) |step| try trace.steps.append(trace.arena, .{
                    .predictor = step.predictor,
                    .inputs = try value.cloneRecord(trace.arena, &step.inputs),
                    .outputs = try value.cloneRecord(trace.arena, &step.outputs),
                });
            }
            return best.output;
        }

        fn runIteration(
            self: *Self,
            ctx: Context,
            attempt_ctx: Context,
            module: *P,
            names: []const predict.NamedParam,
            input: *const Input,
            index: usize,
            best: *Selection,
            advice: *Record,
        ) anyerror!bool {
            var trace: Trace = .{ .arena = ctx.arena };
            defer trace.steps.deinit(ctx.arena);
            var isolated = attempt_ctx;
            isolated.trace = &trace;
            const wrapper: HintWrapper = .{ .parent = ctx.preparation, .names = names, .advice = advice.* };
            if (advice.count() != 0) isolated.preparation = wrapper.preparation();
            const output = try module.forward(isolated, input.*);
            // Snapshot before reward callbacks can append/reallocate the trace.
            const steps = try ctx.arena.dupe(Trace.Step, trace.steps.items);
            var reward_context = ctx;
            reward_context.trace = &trace;
            const reward = try self.reward_fn(reward_context, input, &output);
            if (reward > best.reward) best.* = .{ .output = output, .steps = steps, .reward = reward };
            if (self.threshold) |threshold| {
                if (threshold != .null and try value.floatAtLeast(reward, threshold)) return true;
            }
            if (index + 1 == @as(usize, @intCast(self.n))) return true;
            // Preserve an already-selected best and prior advice if feedback
            // fails. That failure consumes the same persistent attempt budget.
            advice.* = try self.offerAdvice(ctx, names, steps, input, &output, reward);
            return false;
        }

        fn offerAdvice(
            self: *const Self,
            ctx: Context,
            names: []const predict.NamedParam,
            steps: []const Trace.Step,
            input: *const Input,
            output: *const Prediction,
            reward: f64,
        ) anyerror!Record {
            const arena = ctx.arena;
            var trajectory: @import("runtime_value.zig").Array = .init(arena);
            try trajectory.ensureTotalCapacity(steps.len);
            for (steps) |step| {
                const name = for (names) |named| {
                    if (named.param.predict == step.predictor) break named.name;
                } else return error.UnknownTracePredictor;
                var row: Record = .empty;
                try row.put(arena, "module_name", .{ .string = name });
                try row.put(arena, "inputs", .{ .object = try maskedFields(arena, step.predictor.fields[0..step.predictor.input_count], &step.inputs) });
                try row.put(arena, "outputs", .{ .object = try maskedFields(arena, step.predictor.fields[step.predictor.input_count..], &step.outputs) });
                trajectory.appendAssumeCapacity(.{ .object = row });
            }
            var module_names: @import("runtime_value.zig").Array = .init(arena);
            try module_names.ensureTotalCapacity(names.len);
            for (names) |named| module_names.appendAssumeCapacity(.{ .string = named.name });
            var inputs: Record = .empty;
            try inputs.put(arena, "program_code", .{ .string = self.module_code });
            try inputs.put(arena, "modules_defn", .{ .string = try @import("simba.zig").inspectModules(arena, names) });
            try inputs.put(arena, "program_inputs", try dumps(arena, .{ .object = try typedRecord(arena, Input, input.*) }));
            try inputs.put(arena, "program_trajectory", try dumps(arena, .{ .array = trajectory }));
            try inputs.put(arena, "program_outputs", try dumps(arena, .{ .object = try predictionRecord(arena, Prediction, output.*) }));
            try inputs.put(arena, "reward_code", .{ .string = self.reward_code });
            const threshold = self.threshold orelse .null;
            const masked_threshold = (try feedback.masker(Value).?(arena, threshold)) orelse threshold;
            try inputs.put(arena, "target_threshold", try dumps(arena, masked_threshold));
            try inputs.put(arena, "reward_value", try dumps(arena, .{ .float = reward }));
            try inputs.put(arena, "module_names", try dumps(arena, .{ .array = module_names }));
            const adviser = try arena.create(predict.Predict(OfferFeedback));
            adviser.* = .{};
            const result = try adviser.state.call(ctx, &inputs);
            return result.get("advice").?.object;
        }
    };
}

const hint_field: signature.Field = .{
    .name = "hint_",
    .kind = .str,
    .type_name = "str",
    .desc = "A hint to the module from an earlier run",
    .prefix = "Hint :",
};

const HintWrapper = struct {
    parent: ?predict.Preparation,
    names: []const predict.NamedParam,
    advice: Record,

    fn preparation(self: *const HintWrapper) predict.Preparation {
        return .{ .ptr = self, .prepare_fn = prepare };
    }

    fn prepare(ptr: *const anyopaque, arena: Allocator, call: predict.PreparedCall) predict.Error!predict.PreparedCall {
        const self: *const HintWrapper = @ptrCast(@alignCast(ptr));
        var name: ?[]const u8 = null;
        for (self.names) |named| {
            if (named.param.predict.signature_class == call.signature_class) name = named.name;
        }
        const hint = if (name) |key| self.advice.get(key) orelse Value{ .string = "N/A" } else Value{ .string = "N/A" };
        const inputs = try arena.create(Record);
        inputs.* = try call.inputs.clone(arena);
        try inputs.put(arena, "hint_", hint);
        var view = call.view;
        // Signature.append merges through a dict. Existing input names retain
        // their position; an existing output wins and moves before other outputs.
        const existing_input = for (view.inputs, 0..) |field, i| {
            if (std.mem.eql(u8, field.name, "hint_")) break i;
        } else null;
        if (existing_input) |index| {
            const fields = try arena.dupe(signature.Field, view.inputs);
            fields[index] = hint_field;
            view.inputs = fields;
        } else if (for (view.outputs, 0..) |field, i| {
            if (std.mem.eql(u8, field.name, "hint_")) break i;
        } else null) |index| {
            const fields = try arena.alloc(signature.Field, view.outputs.len);
            fields[0] = view.outputs[index];
            @memcpy(fields[1 .. index + 1], view.outputs[0..index]);
            @memcpy(fields[index + 1 ..], view.outputs[index + 1 ..]);
            view.outputs = fields;
        } else {
            const fields = try arena.alloc(signature.Field, view.inputs.len + 1);
            @memcpy(fields[0..view.inputs.len], view.inputs);
            fields[view.inputs.len] = hint_field;
            view.inputs = fields;
        }
        // The per-call field storage already supplies a unique fresh identity.
        const prepared: predict.PreparedCall = .{ .signature_class = view.inputs.ptr, .view = view, .inputs = inputs };
        return if (self.parent) |parent| parent.apply(arena, prepared) else prepared;
    }
};

fn typedRecord(arena: Allocator, comptime T: type, v: T) (feedback.Error || value.ConvertError)!Record {
    var record = try value.toRecord(arena, T, v);
    inline for (@typeInfo(T).@"struct".field_names, @typeInfo(T).@"struct".field_types) |name, FT| {
        if (comptime feedback.masker(FT)) |mask| {
            if (try mask(arena, record.get(name).?)) |changed| record.getPtr(name).?.* = changed;
        }
    }
    return record;
}

fn predictionRecord(arena: Allocator, comptime T: type, v: T) (feedback.Error || value.ConvertError || error{InvalidPrediction})!Record {
    if (@typeInfo(T) == .optional) {
        return if (v) |prediction| predictionRecord(arena, @typeInfo(T).optional.child, prediction) else error.InvalidPrediction;
    }
    return typedRecord(arena, T, v);
}

fn maskedFields(arena: Allocator, fields: []const signature.Field, record: *const Record) feedback.Error!Record {
    var result = record.*;
    var copied = false;
    for (fields) |field| {
        const mask = field.refine_mask orelse continue;
        const original = record.get(field.name) orelse continue;
        if (try mask(arena, original)) |changed| {
            if (!copied) {
                result = try record.clone(arena);
                copied = true;
            }
            result.getPtr(field.name).?.* = changed;
        }
    }
    return result;
}

fn dumps(arena: Allocator, v: Value) (Allocator.Error || error{ JsonTooDeep, AdapterParse })!Value {
    var text: std.Io.Writer.Allocating = .init(arena);
    pyjson.writeJson(&text.writer, v, .orjson_indent2) catch |err| return switch (err) {
        error.JsonTooDeep => error.JsonTooDeep,
        error.WriteFailed => error.OutOfMemory,
        error.ParseValue => error.AdapterParse,
    };
    return .{ .string = text.written() };
}

const testing = std.testing;
const lm_mod = @import("lm.zig");
const TestSignature = signature.Signature(struct {
    pub const instructions = "Answer the question.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});
const SeparateSignature = signature.Signature(struct {
    pub const instructions = "Answer the question.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

fn TestPair(comptime separate: bool) type {
    return struct {
        pub const Input = TestSignature.Input;
        pub const Output = TestSignature.Output;
        first: predict.Predict(TestSignature) = .{},
        last: predict.Predict(if (separate) SeparateSignature else TestSignature) = .{},
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) predict.Error!Output {
            const first = try self.first.forward(ctx, input);
            const last = try self.last.forward(ctx, .{ .question = first.answer });
            return .{ .answer = last.answer };
        }
    };
}

fn testScore(_: Context, _: *const TestSignature.Input, output: *const TestSignature.Output) anyerror!f64 {
    return std.fmt.parseFloat(f64, output.answer);
}

const test_advice =
    "<discussion>Try a concrete correction.</discussion>\n" ++
    "<advice>{\"first\":\"first-specific\",\"last\":\"last-specific\"}</advice>";

test "Refine maps actual signature classes with last-name precedence and retains feedback traces" {
    inline for (.{ false, true }) |separate| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var history: lm_mod.History = .{ .arena = arena };
        var lm: lm_mod.ScriptedLM = .{
            .responses = &.{
                "<answer>7</answer>", "<answer>7</answer>", test_advice,
                "<answer>8</answer>", "<answer>8</answer>", test_advice,
                "<answer>9</answer>", "<answer>9</answer>",
            },
            .history = &history,
        };
        var trace: Trace = .{ .arena = arena };
        const ctx: Context = .{
            .arena = arena,
            .lm = lm.lm(),
            .trace = &trace,
            .lm_options = .{ .rollout_id = .{ .integer = 7 }, .temperature = 0.25 },
        };
        var refined = Refine(TestPair(separate)).init(.{}, .{
            .n = 3,
            .reward_fn = testScore,
            .threshold = .{ .float = 100 },
            .module_code = "class Pair(dspy.Module): ...",
            .reward_code = "def score(inputs, prediction): return float(prediction.answer)",
        });
        const result = (try refined.forward(ctx, .{ .question = "original-question" })).?;
        try testing.expectEqualStrings("9", result.answer);
        try testing.expectEqual(@as(usize, 4), trace.steps.items.len);
        try testing.expect(trace.steps.items[0].outputs.contains("advice"));
        try testing.expect(trace.steps.items[1].outputs.contains("advice"));
        try testing.expectEqualStrings(if (separate) "first-specific" else "last-specific", trace.steps.items[2].inputs.get("hint_").?.string);
        try testing.expectEqualStrings("last-specific", trace.steps.items[3].inputs.get("hint_").?.string);
        try testing.expectEqualStrings("9", trace.steps.items[3].outputs.get("answer").?.string);
        try testing.expectEqualStrings("7.0", trace.steps.items[0].inputs.get("reward_value").?.string);
        try testing.expectEqualStrings("100.0", trace.steps.items[0].inputs.get("target_threshold").?.string);
        try testing.expectEqualStrings("[\n  \"first\",\n  \"last\"\n]", trace.steps.items[0].inputs.get("module_names").?.string);
        for (history.requests.items, 0..) |request, i| {
            const feedback_call = i == 2 or i == 5;
            try testing.expectEqual(if (feedback_call) @as(f64, 0.25) else 1.0, request.options.temperature.?);
            try testing.expectEqual(@as(i64, if (feedback_call) 7 else if (i < 2) 7 else if (i < 5) 8 else 9), request.options.rollout_id.?.integer);
        }
        try testing.expectEqual(@as(usize, 1), refined.module.first.state.input_count);
        try testing.expectEqual(@as(usize, 1), refined.module.last.state.input_count);
    }
}

test "Refine masks ReAct custom objects in feedback without changing stored traces" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Agent = @import("react.zig").ReAct(TestSignature, &.{});
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        "<next_thought>ready</next_thought><tool_calls>{\"tool_calls\":[{\"name\":\"submit\",\"args\":{\"answer\":\"7\"}}]}</tool_calls>",
        "<discussion>Improve.</discussion><advice>{}</advice>",
        "<next_thought>ready</next_thought><tool_calls>{\"tool_calls\":[{\"name\":\"submit\",\"args\":{\"answer\":\"8\"}}]}</tool_calls>",
    } };
    var trace: Trace = .{ .arena = arena };
    var refined = Refine(Agent).init(try Agent.init(arena), .{
        .n = 2,
        .reward_fn = struct {
            fn score(_: Context, _: *const Agent.Input, output: *const Agent.Output) anyerror!f64 {
                return std.fmt.parseFloat(f64, output.answer);
            }
        }.score,
        .threshold = .{ .integer = 100 },
        .module_code = "dspy.ReAct(QA, tools=[])",
        .reward_code = "def score(inputs, prediction): return float(prediction.answer)",
    });
    const result = (try refined.forward(.{ .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
    try testing.expectEqualStrings("8", result.answer);
    try testing.expectEqualStrings("100", trace.steps.items[0].inputs.get("target_threshold").?.string);
    const program_outputs = try std.json.parseFromSliceLeaky(Value, arena, trace.steps.items[0].inputs.get("program_outputs").?.string, .{});
    try testing.expectEqualStrings("<non-serializable: History>", program_outputs.object.get("history").?.string);
    try testing.expectEqualStrings("submit", program_outputs.object.get("termination_reason").?.string);
    try testing.expectEqual(@import("react.zig").TerminationReason.submit, result.termination_reason);
    try testing.expectEqualStrings("8", result.history.messages[0].get("answer").?.string);
    const trajectory = try std.json.parseFromSliceLeaky(Value, arena, trace.steps.items[0].inputs.get("program_trajectory").?.string, .{});
    const step = trajectory.array.items[0].object;
    const inputs = step.get("inputs").?.object;
    const outputs = step.get("outputs").?.object;
    try testing.expect(inputs.get("history").? == .string);
    try testing.expectEqualStrings("<non-serializable: History>", inputs.get("history").?.string);
    try testing.expectEqualStrings("<non-serializable: Tool>", inputs.get("tools").?.array.items[0].string);
    try testing.expectEqualStrings("<non-serializable: ToolCalls>", outputs.get("tool_calls").?.string);
    try testing.expectEqualStrings("ready", outputs.get("next_thought").?.string);
    try testing.expect(trace.steps.items[1].inputs.get("history").? == .object);
    try testing.expect(trace.steps.items[1].outputs.get("tool_calls").? == .object);
}

test "Refine assigns separate advice to independent ReAct instances" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Agent = @import("react.zig").ReAct(TestSignature, &.{});
    const Pair = struct {
        pub const Input = TestSignature.Input;
        pub const Output = Agent.Output;
        first: Agent,
        last: Agent,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) @import("react.zig").Error!Output {
            const first = try self.first.forward(ctx, input);
            return self.last.forward(ctx, .{ .question = first.answer });
        }
    };
    const seven = "<next_thought>ready</next_thought><tool_calls>{\"tool_calls\":[{\"name\":\"submit\",\"args\":{\"answer\":\"7\"}}]}</tool_calls>";
    const eight = "<next_thought>ready</next_thought><tool_calls>{\"tool_calls\":[{\"name\":\"submit\",\"args\":{\"answer\":\"8\"}}]}</tool_calls>";
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        seven,                                                                                                                     seven,
        "<discussion>Improve.</discussion><advice>{\"first.react\":\"first-specific\",\"last.react\":\"last-specific\"}</advice>", eight,
        eight,
    } };
    var trace: Trace = .{ .arena = arena };
    var refined = Refine(Pair).init(.{ .first = try Agent.init(arena), .last = try Agent.init(arena) }, .{
        .n = 2,
        .reward_fn = struct {
            fn score(_: Context, _: *const Pair.Input, output: *const Pair.Output) anyerror!f64 {
                return std.fmt.parseFloat(f64, output.answer);
            }
        }.score,
        .threshold = .{ .integer = 100 },
        .module_code = "AgentPair",
        .reward_code = "score",
    });
    const result = (try refined.forward(.{ .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
    try testing.expectEqualStrings("8", result.answer);
    try testing.expectEqualStrings("first-specific", trace.steps.items[1].inputs.get("hint_").?.string);
    try testing.expectEqualStrings("last-specific", trace.steps.items[2].inputs.get("hint_").?.string);
}

test "Refine retains exact integer thresholds beyond binary64 and binary128 precision" {
    inline for (.{ false, true }) |negative| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var lm: lm_mod.ScriptedLM = .{ .responses = &.{
            "<answer>7</answer>",
            "<discussion>Continue.</discussion><advice>{}</advice>",
            "<answer>8</answer>",
        } };
        var trace: Trace = .{ .arena = arena };
        var refined = Refine(predict.Predict(TestSignature)).init(.{}, .{
            .n = 2,
            .reward_fn = struct {
                fn score(_: Context, _: *const TestSignature.Input, output: *const TestSignature.Output) anyerror!f64 {
                    return if (std.mem.eql(u8, output.answer, "7"))
                        (if (negative) -0x1p200 else 0x1p200)
                    else
                        0x1p201;
                }
            }.score,
            .threshold = .{ .number_string = if (negative)
                "-1606938044258990275541962092341162602522202993782792835301377"
            else
                "1606938044258990275541962092341162602522202993782792835301377" },
            .module_code = "Predict(QA)",
            .reward_code = "large_score",
        });
        const result = (try refined.forward(.{ .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
        try testing.expectEqualStrings(if (negative) "7" else "8", result.answer);
        if (!negative) {
            try testing.expectEqualStrings("\"<non-serializable: int>\"", trace.steps.items[0].inputs.get("target_threshold").?.string);
        }
    }
}

fn nanScore(_: Context, _: *const TestSignature.Input, _: *const TestSignature.Output) anyerror!f64 {
    return std.math.nan(f64);
}

test "Refine preserves null selection for NaN and empty advice does not install hints" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>7</answer>",                               "<answer>7</answer>",
        "<discussion>none</discussion><advice>{}</advice>", "<answer>8</answer>",
        "<answer>8</answer>",
    } };
    var trace: Trace = .{ .arena = arena };
    const ctx: Context = .{ .arena = arena, .lm = lm.lm(), .trace = &trace };
    var refined = Refine(TestPair(false)).init(.{}, .{
        .n = 2,
        .reward_fn = nanScore,
        .threshold = null,
        .module_code = "Pair",
        .reward_code = "nan_score",
    });
    try testing.expectEqual(null, try refined.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].outputs.contains("advice"));
    try testing.expectEqualStrings("null", trace.steps.items[0].inputs.get("reward_value").?.string);
    try testing.expectEqualStrings("null", trace.steps.items[0].inputs.get("target_threshold").?.string);
}

const FailingProgram = struct {
    pub const Input = TestSignature.Input;
    pub const Output = TestSignature.Output;
    pub fn forward(_: *const @This(), _: Context, _: Input) error{ProgramFailure}!Output {
        return error.ProgramFailure;
    }
};

test "Refine failure budgets persist and zero means the constructor attempt count" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .arena = memory.allocator(), .lm = lm.lm() };
    var refined = Refine(FailingProgram).init(.{}, .{
        .n = 4,
        .reward_fn = testScore,
        .threshold = .{ .float = 100 },
        .fail_count = 0,
        .module_code = "FailingPair",
        .reward_code = "score",
    });
    try testing.expectError(error.ProgramFailure, refined.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(i128, 1), refined.fail_count);
    try testing.expectError(error.ProgramFailure, refined.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(i128, 0), refined.fail_count);
    refined.fail_count = -1;
    try testing.expectError(error.ProgramFailure, refined.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(i128, -1), refined.fail_count);
    refined.n = 0;
    try testing.expectEqual(null, try refined.forward(ctx, .{ .question = "q" }));
}

test "Refine feedback failure keeps the selected result without leaking failed attempt traces" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>7</answer>", "invalid-feedback" } };
    var trace: Trace = .{ .arena = arena };
    const ctx: Context = .{ .arena = arena, .lm = lm.lm(), .trace = &trace, .json_fallback = false };
    var refined = Refine(predict.Predict(TestSignature)).init(.{}, .{
        .n = 2,
        .reward_fn = testScore,
        .threshold = .{ .float = 100 },
        .module_code = "Predict(QA)",
        .reward_code = "score",
    });
    try testing.expectEqualStrings("7", (try refined.forward(ctx, .{ .question = "q" })).?.answer);
    try testing.expectEqual(@as(i128, 0), refined.fail_count);
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    try testing.expectEqualStrings("7", trace.steps.items[0].outputs.get("answer").?.string);
}

test "Refine recursive masking retains Reasoning content but masks nested Code models and large integers" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Code = @import("code.zig").Code;
    const Model = struct {
        pub const model_name = "QA";
        question: []const u8,
        answer: []const u8,
    };
    const Inputs = struct {
        reasoning: []const signature.Reasoning,
        code: []const Code,
        model: []const Model,
        large: u256,
    };
    const record = try typedRecord(arena, Inputs, .{
        .reasoning = &.{.{ .content = "reason" }},
        .code = &.{Code.init("x = 1")},
        .model = &.{.{ .question = "q", .answer = "a" }},
        .large = @as(u256, 1) << 128,
    });
    try testing.expectEqualStrings("reason", record.get("reasoning").?.array.items[0].string);
    try testing.expectEqualStrings("<non-serializable: Code>", record.get("code").?.array.items[0].string);
    try testing.expectEqualStrings("<non-serializable: QA>", record.get("model").?.array.items[0].string);
    try testing.expectEqualStrings("<non-serializable: int>", record.get("large").?.string);
    try testing.expectEqualStrings("0.00001", (try dumps(arena, .{ .float = 1e-5 })).string);
    try testing.expectEqualStrings("1e-6", (try dumps(arena, .{ .float = 1e-6 })).string);
    try testing.expectEqualStrings("1e+16", (try dumps(arena, .{ .float = 1e16 })).string);
    try testing.expectEqualStrings("-0.0", (try dumps(arena, .{ .float = -0.0 })).string);
}
