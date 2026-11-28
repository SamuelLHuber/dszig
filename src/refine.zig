//! DSPy 3.5 Refine: typed attempts, source-backed feedback, scoped adapter hints.
//! The caller supplies program/reward source text explicitly: Zig does not have
//! Python's inspect.getsource. All borrowed source slices must outlive Self.
//! Attempt programs, signatures, advice, outputs and trace snapshots live in the
//! call arena. Mutable program data must be stored by value, like BestOfN.
//! Feedback uses typed or dynamic record fields (`void` input means no kwargs).
//! Runtime lists/tuples and native arrays/slices of `Value` can supply result
//! fields when they contain string-keyed pairs.
//! Scalar results can win, but cannot supply feedback fields. Native scalar
//! inputs likewise work until feedback needs keyword names that they lack.
//! Canonical Prediction metadata and Python's general iteration protocol
//! are not used.
//! Trace rows remain snapshots, and tolerated failures affect the budget
//! without printing DSPy's stdout diagnostics.
//!
//! Upstream (DSPy 3.5.0): dspy/predict/refine.py

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
            const rollout = try best_of_n.rolloutContext(ctx, &self.module);
            if (self.n <= 0) return null;
            var best: Selection = .{};
            var advice: Record = .empty;
            for (0..@as(usize, @intCast(self.n))) |index| {
                // LM.copy and program.deepcopy are outside the caught region.
                const module = try ctx.arena.create(P);
                module.* = self.module;
                const names = try predict.namedPredictors(ctx.arena, module);
                var scope: predict.PredictRollout = undefined;
                var attempt_ctx = try rollout.attempt(index, names, &scope);
                const has_advice = advice.count() != 0;
                const classes = if (has_advice) try SignatureName.capture(ctx.arena, names) else &.{};
                const wrapper: HintWrapper = .{ .parent = ctx.preparation, .names = classes, .advice = advice };
                if (has_advice) attempt_ctx.preparation = wrapper.preparation();
                const stop = self.runIteration(ctx, attempt_ctx, module, names, &input, index, &best, &advice) catch |err| {
                    // Cancellation and a native stream sink's .stop bypass
                    // the retry budget, including during feedback.
                    if (err == error.Canceled or err == error.StreamStopped) return err;
                    if (@as(i128, @intCast(index)) > self.fail_count) return err;
                    self.fail_count -= 1;
                    continue;
                };
                if (stop) break;
            }
            // Feedback runs on the outer trace naturally. Only the winning
            // program trace is appended here; reward calls remain isolated.
            if (ctx.trace) |trace| {
                for (best.steps) |step| try trace.steps.append(trace.arena, try step.clone(trace.arena));
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
            advice.* = try self.offerAdvice(ctx, module, names, steps, input, &output, reward);
            return false;
        }

        fn offerAdvice(
            self: *const Self,
            ctx: Context,
            module: *P,
            names: []const predict.NamedParam,
            steps: []const Trace.Step,
            input: *const Input,
            output: *const Prediction,
            reward: f64,
        ) anyerror!Record {
            const arena = ctx.arena;
            // Source re-enumerates for modules_defn after forward and reward.
            // Trajectory labels and module_names still use the original map.
            const modules_defn = try @import("simba.zig").inspectModules(arena, try predict.namedPredictors(arena, module));
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
            try inputs.put(arena, "modules_defn", .{ .string = modules_defn });
            const program_inputs: Record = if (Input == void) .empty else try typedRecord(arena, Input, input.*);
            try inputs.put(arena, "program_inputs", try dumps(arena, .{ .object = program_inputs }));
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
            // DSPy invokes a fresh, unbound Predict with only feedback kwargs.
            // Keep the ambient adapter, trace, and settings LM/options, but
            // don't pass the wrapped program's privileged predictor arguments.
            var adviser_ctx = ctx;
            adviser_ctx.lm_override = null;
            adviser_ctx.lm_override_present = false;
            adviser_ctx.demos_override = null;
            adviser_ctx.predict_config = null;
            const result = try adviser.state.call(adviser_ctx, &inputs);
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

const SignatureName = struct {
    class: *const anyopaque,
    name: []const u8,

    fn capture(arena: Allocator, names: []const predict.NamedParam) Allocator.Error![]SignatureName {
        // DSPy's signature2name keys are captured before the program runs.
        // A predictor assigning itself a fresh signature during forward does
        // not move its advice to that class. Last name wins on shared classes.
        const classes = try arena.alloc(SignatureName, names.len);
        for (names, classes) |named, *entry| {
            entry.* = .{ .class = named.param.predict.signature_class, .name = named.name };
        }
        return classes;
    }
};

const HintWrapper = struct {
    parent: ?predict.Preparation,
    names: []const SignatureName,
    advice: Record,

    fn preparation(self: *const HintWrapper) predict.Preparation {
        return .{ .ptr = self, .prepare_fn = prepare };
    }

    fn prepare(ptr: *const anyopaque, arena: Allocator, call: predict.PreparedCall) predict.Error!predict.PreparedCall {
        const self: *const HintWrapper = @ptrCast(@alignCast(ptr));
        var name: ?[]const u8 = null;
        var remaining = self.names.len;
        while (remaining > 0) {
            remaining -= 1;
            const named = self.names[remaining];
            if (named.class == call.signature_class) {
                name = named.name;
                break;
            }
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
    // A runtime object represents the same keyword dictionary as a Record.
    // Do not stringify scalar inputs or invent a keyword for them.
    if (T == Value) {
        if (v != .object) return error.TypeMismatch;
        return typedRecord(arena, Record, v.object);
    }
    if (comptime T == Record or T == signature.StrDict or @import("types.zig").isDict(T)) {
        // A typed dictionary's entries are its fields, not its Zig storage
        // (`map` or `keys`/`values`). Preserve value types for recursive masking.
        const object = try value.toValue(arena, T, v);
        const mask = comptime feedback.masker(T);
        return (if (mask) |apply| (try apply(arena, object)) orelse object else object).object;
    }
    // A native program may take a scalar instead of a keyword record. Only
    // feedback needs named inputs; do not guess a key or fail compilation.
    if (@typeInfo(T) != .@"struct") return error.TypeMismatch;
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
    // These typed scalars have struct storage in Zig, but `value` and
    // `index` are not Python prediction fields. Only feedback needs this
    // projection; the reward and selection keep the original result type.
    if (comptime T == @import("integer.zig").Integer or @import("literal.zig").isLiteral(T) or
        @import("enum.zig").isEnum(T) or @import("decimal_type.zig").isDecimal(T))
        return predictionRecord(arena, Value, try value.toValue(arena, T, v));
    // DSPy uses dict(outputs) only when offering feedback. A scalar can still
    // win an attempt; fail at feedback time, not when compiling the wrapper.
    if (T == Value) {
        if (v == .object) return typedRecord(arena, Record, v.object);
        const pairs = feedbackSequence(v) orelse return error.InvalidPrediction;
        return feedbackPairRecord(arena, pairs);
    }
    // These native containers already hold runtime pairs; borrow the slice
    // instead of boxing an extra list or tuple just to read its elements.
    const typed_pairs: ?[]const Value = switch (@typeInfo(T)) {
        .array => |info| if (info.child == Value) &v else null,
        .pointer => |info| if (info.size == .slice and info.child == Value) v else null,
        else => null,
    };
    if (typed_pairs) |pairs| return feedbackPairRecord(arena, pairs);
    if (@typeInfo(T) != .@"struct") return error.InvalidPrediction;
    // Completion rows and usage attached to a canonical Prediction are not
    // `dict(outputs)` fields.
    if (comptime @import("prediction.zig").isPrediction(T)) return predictionRecord(arena, T.Fields, v.fields().*);
    return typedRecord(arena, T, v);
}

fn feedbackPairRecord(arena: Allocator, entries: []const Value) (feedback.Error || error{InvalidPrediction})!Record {
    var fields: Record = .empty;
    errdefer fields.deinit(arena);
    for (entries) |entry| {
        const pair = feedbackSequence(entry) orelse return error.InvalidPrediction;
        // orjson refuses non-string dictionary keys. For duplicate string
        // keys, dict() keeps the first insertion position and the last value.
        if (pair.len != 2 or pair[0] != .string) return error.InvalidPrediction;
        try fields.put(arena, pair[0].string, pair[1]);
    }
    // Mask only the final values, after duplicate resolution. This record is
    // newly owned, so no copy is needed and the original pairs stay intact.
    const mask = feedback.masker(Value).?;
    for (fields.values()) |*item| {
        if (try mask(arena, item.*)) |changed| item.* = changed;
    }
    return fields;
}

fn feedbackSequence(v: Value) ?[]const Value {
    if (v == .array) return v.array.items;
    const builtin = @import("python_value.zig").view(v) orelse return null;
    return if (builtin == .tuple) builtin.tuple else null;
}

fn maskedFields(arena: Allocator, fields: []const signature.Field, record: *const Record) feedback.Error!Record {
    const runtime_mask = feedback.masker(Value).?;
    var masked: ?Record = null;
    var matched: usize = 0;
    for (fields) |field| {
        const item = record.get(field.name) orelse continue;
        matched += 1;
        // Declared metadata can recover a model flattened into a container,
        // but an actual scalar is still a scalar. In particular, the adapter
        // can replace a model-typed `hint_` input with an advice string.
        const mask = switch (item) {
            .array, .object, .custom => field.refine_mask orelse runtime_mask,
            else => runtime_mask,
        };
        if (try mask(arena, item)) |changed| {
            if (masked == null) masked = try record.clone(arena);
            masked.?.getPtr(field.name).?.* = changed;
        }
    }
    // Source masks the whole trace dictionary, not just signature fields.
    // Extra kwargs and adapter-injected hints have only runtime type identity.
    if (matched < record.count()) {
        extra: for (record.keys(), record.values()) |name, item| {
            for (fields) |field| {
                if (std.mem.eql(u8, field.name, name)) continue :extra;
            }
            if (try runtime_mask(arena, item)) |changed| {
                if (masked == null) masked = try record.clone(arena);
                masked.?.getPtr(name).?.* = changed;
            }
        }
    }
    return masked orelse record.*;
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
    const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
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
    const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
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
        const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
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
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace };
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
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = memory.allocator(), .lm = lm.lm() };
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
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace, .json_fallback = false };
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

test "Refine uses the original signature map and serializes canonical prediction fields" {
    const Program = struct {
        pub const Input = void;
        pub const Output = predict.Prediction(Record);

        qa: predict.Predict(TestSignature) = .{},
        auxiliary: TestPair(false) = .{},
        compiled: bool = false,

        pub fn forward(self: *@This(), ctx: Context, _: Input) !Output {
            self.auxiliary.compiled = true;
            try self.qa.state.forkSignatureClass(ctx.arena);
            self.qa.state.instructions = "Use the current instructions.";
            const answer = try self.qa.forward(ctx, .{ .question = "q" });
            const completions = try ctx.arena.alloc(Record, 2);
            completions[0] = .empty;
            completions[1] = .empty;
            try completions[0].put(ctx.arena, "answer", .{ .string = answer.answer });
            try completions[0].put(ctx.arena, "reasoning", try value.toValue(ctx.arena, signature.Reasoning, .{ .content = "a plan" }));
            try completions[1].put(ctx.arena, "answer", .{ .string = "secondary" });
            return Output.fromCompletions(ctx.arena, completions);
        }

        fn reward(_: Context, _: *const Input, output: *const Output) anyerror!f64 {
            return std.fmt.parseFloat(f64, output.get("answer").?.string);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var trace: Trace = .{ .arena = arena };
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>7</answer>",
        "<discussion>Try again.</discussion><advice>{\"qa\":\"specific advice\"}</advice>",
        "<answer>8</answer>",
    } };
    var refined = Refine(Program).init(.{}, .{
        .n = 2,
        .reward_fn = Program.reward,
        .threshold = .{ .integer = 8 },
        .module_code = "Program",
        .reward_code = "reward",
    });
    const original_class = refined.module.qa.state.signature_class;
    const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, {})).?;
    try testing.expectEqualStrings("8", result.get("answer").?.string);
    try testing.expectEqual(@as(usize, 3), lm.index);
    try testing.expectEqual(@as(i128, 2), refined.fail_count);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    const feedback_inputs = &trace.steps.items[0].inputs;
    try testing.expectEqualStrings("{}", feedback_inputs.get("program_inputs").?.string);
    try testing.expectEqualStrings(
        "{\n  \"answer\": \"7\",\n  \"reasoning\": \"a plan\"\n}",
        feedback_inputs.get("program_outputs").?.string,
    );
    const definitions = feedback_inputs.get("modules_defn").?.string;
    try testing.expect(std.mem.indexOf(u8, definitions, "Use the current instructions.") != null);
    try testing.expect(std.mem.indexOf(u8, definitions, "auxiliary.") == null);
    try testing.expectEqualStrings(
        "[\n  \"qa\",\n  \"auxiliary.first\",\n  \"auxiliary.last\"\n]",
        feedback_inputs.get("module_names").?.string,
    );
    try testing.expectEqualStrings("N/A", trace.steps.items[1].inputs.get("hint_").?.string);
    try testing.expectEqual(original_class, refined.module.qa.state.signature_class);
}

test "Refine resolves the LM before returning for an empty range" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    var refined = Refine(predict.Predict(TestSignature)).init(.{}, .{
        .n = 0,
        .fail_count = 2,
        .reward_fn = testScore,
        .threshold = null,
        .module_code = "Predict",
        .reward_code = "score",
    });
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = memory.allocator() };
    try testing.expectError(error.NoLM, refined.forward(ctx, .{ .question = "q" }));
    try testing.expectEqual(@as(i128, 2), refined.fail_count);
}

test "Refine masks undeclared trace kwargs without rewriting the selected trace" {
    const QA = predict.Predict(TestSignature);
    const Code = @import("code.zig").Code;
    const Program = struct {
        pub const Input = QA.Input;
        pub const Output = QA.Output;

        qa: QA = .{},
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) !Output {
            var kwargs: Record = .empty;
            try kwargs.put(ctx.arena, "question", .{ .string = input.question });
            try kwargs.put(ctx.arena, "extra_code", try Code.init("x = 1").toRuntime(ctx.arena));
            try kwargs.put(ctx.arena, "extra_reasoning", try value.toValue(ctx.arena, signature.Reasoning, .{ .content = "a plan" }));
            const result = try self.qa.state.call(ctx, &kwargs);
            return .{ .answer = result.get("answer").?.string };
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var trace: Trace = .{ .arena = arena };
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>7</answer>",
        "<discussion>Try again.</discussion><advice>{}</advice>",
        "<answer>8</answer>",
    } };
    var refined = Refine(Program).init(.{}, .{
        .n = 2,
        .reward_fn = testScore,
        .threshold = .{ .integer = 8 },
        .module_code = "Program",
        .reward_code = "score",
    });
    const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .question = "q" })).?;
    try testing.expectEqualStrings("8", result.answer);
    try testing.expectEqual(@as(usize, 3), lm.index);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    const trajectory = trace.steps.items[0].inputs.get("program_trajectory").?.string;
    try testing.expect(std.mem.indexOf(u8, trajectory, "\"extra_code\": \"<non-serializable: Code>\"") != null);
    try testing.expect(std.mem.indexOf(u8, trajectory, "\"extra_reasoning\": \"a plan\"") != null);
    const selected = trace.steps.items[1].inputs;
    try testing.expectEqualStrings("x = 1", selected.get("extra_code").?.custom.get(Code).?.code);
    try testing.expectEqualStrings("a plan", selected.get("extra_reasoning").?.custom.get(signature.Reasoning).?.content);
}

test "Refine masks the actual hint value rather than the overwritten field annotation" {
    const HintModel = struct {
        pub const model_name = "HintModel";
        text: []const u8,
    };
    const Sig = signature.Signature(struct {
        pub const instructions = "Answer the question.";
        pub const Input = struct { question: []const u8, hint_: HintModel };
        pub const Output = TestSignature.Output;
    });
    const Reward = struct {
        fn score(_: Context, _: *const Sig.Input, output: *const Sig.Output) anyerror!f64 {
            return std.fmt.parseFloat(f64, output.answer);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var trace: Trace = .{ .arena = arena };
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>7</answer>",
        "<discussion>Try again.</discussion><advice>{\"self\":\"repair once\"}</advice>",
        "<answer>8</answer>",
        "<discussion>Try again.</discussion><advice>{\"self\":\"repair once more\"}</advice>",
        "<answer>9</answer>",
    } };
    var refined = Refine(predict.Predict(Sig)).init(.{}, .{
        .n = 3,
        .reward_fn = Reward.score,
        .threshold = .{ .integer = 9 },
        .module_code = "Predict",
        .reward_code = "score",
    });
    const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{
        .question = "q",
        .hint_ = .{ .text = "original" },
    })).?;
    try testing.expectEqualStrings("9", result.answer);
    try testing.expectEqual(@as(usize, 5), lm.index);
    try testing.expectEqual(@as(usize, 3), trace.steps.items.len);
    const first = trace.steps.items[0].inputs.get("program_trajectory").?.string;
    const second = trace.steps.items[1].inputs.get("program_trajectory").?.string;
    try testing.expect(std.mem.indexOf(u8, first, "\"hint_\": \"<non-serializable: HintModel>\"") != null);
    try testing.expect(std.mem.indexOf(u8, second, "\"hint_\": \"repair once\"") != null);
    try testing.expectEqualStrings("repair once more", trace.steps.items[2].inputs.get("hint_").?.string);
}

test "Refine adviser uses settings rather than the task predictor overrides" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var task_history: lm_mod.History = .{ .arena = arena };
    var adviser_history: lm_mod.History = .{ .arena = arena };
    var task_lm: lm_mod.ScriptedLM = .{
        .responses = &.{ "<answer>7</answer>", "<answer>8</answer>" },
        .history = &task_history,
    };
    var settings_lm: lm_mod.ScriptedLM = .{
        .responses = &.{"<discussion>Try again.</discussion><advice>{}</advice>"},
        .history = &adviser_history,
    };
    var demo: Record = .empty;
    try demo.put(arena, "question", .{ .string = "task-demo-question" });
    try demo.put(arena, "answer", .{ .string = "task-demo-answer" });
    try demo.put(arena, "program_code", .{ .string = "task-demo-program" });
    try demo.put(arena, "discussion", .{ .string = "task-demo-discussion" });
    try demo.put(arena, "advice", .{ .object = .empty });
    var trace: Trace = .{ .arena = arena };
    var refined = Refine(predict.Predict(TestSignature)).init(.{}, .{
        .n = 2,
        .reward_fn = testScore,
        .threshold = .{ .integer = 8 },
        .module_code = "Predict",
        .reward_code = "score",
    });
    const result = (try refined.forward(.{
        .arena = arena,
        .lm = settings_lm.lm(),
        .lm_override = task_lm.lm(),
        .lm_override_present = true,
        .lm_options = .{ .rollout_id = .{ .integer = 97 }, .temperature = 0.25 },
        .before_settings_rollout = .{ .rollout_id = .{ .integer = 7 }, .temperature = 0.25 },
        .predict_config = &.{ .temperature = 0.6 },
        .demos_override = &.{demo},
        .trace = &trace,
    }, .{ .question = "q" })).?;
    try testing.expectEqualStrings("8", result.answer);
    try testing.expectEqual(@as(usize, 2), task_history.requests.items.len);
    try testing.expectEqual(@as(usize, 1), adviser_history.requests.items.len);
    for (task_history.requests.items) |request| {
        try testing.expectEqual(@as(?f64, 0.6), request.options.temperature);
        try testing.expectEqual(@as(i64, 7), request.options.rollout_id.?.integer);
        try testing.expect(for (request.messages) |message| {
            if (std.mem.indexOf(u8, message.content.text, "task-demo-question") != null) break true;
        } else false);
    }
    const adviser_request = adviser_history.requests.items[0];
    try testing.expectEqual(@as(?f64, 0.25), adviser_request.options.temperature);
    try testing.expectEqual(@as(i64, 97), adviser_request.options.rollout_id.?.integer);
    for (adviser_request.messages) |message| {
        try testing.expect(std.mem.indexOf(u8, message.content.text, "task-demo-") == null);
    }
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].outputs.contains("advice"));
    try testing.expectEqualStrings("8", trace.steps.items[1].outputs.get("answer").?.string);
}

test "BestOfN keeps the ambient adviser and reward options in nested Refine" {
    for ([_]bool{ false, true }) |task_bound| try checkNestedRollout(task_bound);
}

fn checkNestedRollout(task_bound: bool) !void {
    const QA = predict.Predict(TestSignature);
    const Inner = Refine(QA);
    const Reward = struct {
        fn inner(ctx: Context, input: *const QA.Input, output: *const QA.Output) anyerror!f64 {
            const judge: QA = .{};
            _ = try judge.forward(ctx, .{ .question = "judge" });
            return testScore(ctx, input, output);
        }

        fn score(ctx: Context, input: *const Inner.Input, output: *const Inner.Output) anyerror!f64 {
            return if (output.*) |prediction| testScore(ctx, input, &prediction) else 0;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var task_history: lm_mod.History = .{ .arena = arena };
    var adviser_history: lm_mod.History = .{ .arena = arena };
    var task: lm_mod.ScriptedLM = .{
        .responses = &.{ "<answer>7</answer>", "<answer>8</answer>" },
        .history = &task_history,
    };
    var settings: lm_mod.ScriptedLM = .{
        .responses = if (task_bound) &.{
            "<answer>97</answer>",
            "<discussion>Try again.</discussion><advice>{}</advice>",
            "<answer>97</answer>",
        } else &.{
            "<answer>7</answer>",
            "<answer>97</answer>",
            "<discussion>Try again.</discussion><advice>{}</advice>",
            "<answer>8</answer>",
            "<answer>97</answer>",
        },
        .history = &adviser_history,
    };
    var qa: QA = .{};
    if (task_bound) qa.state.lm = task.lm();
    const inner = Inner.init(qa, .{
        .n = 2,
        .reward_fn = Reward.inner,
        .threshold = .{ .integer = 8 },
        .module_code = "Predict",
        .reward_code = "score",
    });
    var outer = best_of_n.BestOfN(Inner).init(inner, .{
        .n = 2,
        .reward_fn = Reward.score,
        .threshold = .{ .integer = 8 },
    });
    var trace: Trace = .{ .arena = arena };
    const result = (try outer.forward(.{
        .arena = arena,
        .lm = settings.lm(),
        .lm_options = .{ .rollout_id = .{ .integer = 97 }, .temperature = 0.25 },
        .before_settings_rollout = .{ .rollout_id = .{ .integer = 7 }, .temperature = 0.25 },
        .trace = &trace,
    }, .{ .question = "q" })).?.?;
    try testing.expectEqualStrings("8", result.answer);
    var tasks: [2]lm_mod.Request = undefined;
    var helpers: [3]lm_mod.Request = undefined;
    if (task_bound) {
        try testing.expectEqual(@as(usize, 2), task_history.requests.items.len);
        try testing.expectEqual(@as(usize, 3), adviser_history.requests.items.len);
        tasks = task_history.requests.items[0..2].*;
        helpers = adviser_history.requests.items[0..3].*;
    } else {
        try testing.expectEqual(@as(usize, 5), adviser_history.requests.items.len);
        const requests = adviser_history.requests.items;
        tasks = .{ requests[0], requests[3] };
        helpers = .{ requests[1], requests[2], requests[4] };
    }
    for (tasks, 0..) |request, index| {
        try testing.expectEqual(@as(i64, @as(i64, if (task_bound) 7 else 97) + @as(i64, @intCast(index))), request.options.rollout_id.?.integer);
        try testing.expectEqual(@as(?f64, 1), request.options.temperature);
    }
    for (helpers, 0..) |request, index| {
        try testing.expectEqual(@as(i64, 97), request.options.rollout_id.?.integer);
        try testing.expectEqual(@as(?f64, 0.25), request.options.temperature);
        const prompt = request.messages[request.messages.len - 1].content.text;
        try testing.expectEqual(index != 1, std.mem.indexOf(u8, prompt, "<question>\njudge\n</question>") != null);
    }
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].outputs.contains("advice"));
    try testing.expectEqualStrings("q", trace.steps.items[1].inputs.get("question").?.string);
    try testing.expectEqualStrings("8", trace.steps.items[1].outputs.get("answer").?.string);
}

test "Refine interruption keeps prior feedback but does not publish a selected attempt" {
    const QA = predict.Predict(TestSignature);
    inline for (.{ error.Canceled, error.StreamStopped }) |interruption| {
        const Reward = struct {
            fn score(ctx: Context, input: *const QA.Input, output: *const QA.Output) anyerror!f64 {
                if (std.mem.eql(u8, output.answer, "8")) return interruption;
                return testScore(ctx, input, output);
            }
        };
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var lm: lm_mod.ScriptedLM = .{ .responses = &.{
            "<answer>7</answer>",
            "<discussion>Try again.</discussion><advice>{}</advice>",
            "<answer>8</answer>",
            "<discussion>Should not be called.</discussion><advice>{}</advice>",
            "<answer>9</answer>",
        } };
        var trace: Trace = .{ .arena = arena };
        var refine = Refine(QA).init(.{}, .{
            .n = 3,
            .threshold = .{ .integer = 9 },
            .reward_fn = Reward.score,
            .module_code = "Predict",
            .reward_code = "score",
        });
        try testing.expectError(interruption, refine.forward(
            .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace },
            .{ .question = "q" },
        ));
        try testing.expectEqual(@as(usize, 3), lm.index);
        try testing.expectEqual(@as(i128, 3), refine.fail_count);
        try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
        try testing.expect(trace.steps.items[0].outputs.contains("advice"));
    }
}

test "Refine selects scalar results and rejects them only when feedback needs fields" {
    const QA = predict.Predict(TestSignature);
    const Integer = @import("integer.zig").Integer;
    const Decimal = @import("decimal_type.zig").Decimal;
    const Literal = @import("literal.zig").Literal(&.{ .{ .integer = 7 }, .{ .integer = 8 } });
    const Grade = @import("enum.zig").IntEnum("Grade", &.{
        .{ .name = "SEVEN", .value = .{ .integer = 7 } },
        .{ .name = "EIGHT", .value = .{ .integer = 8 } },
    });
    inline for (.{ i64, Integer, Decimal, Literal, Grade }) |Result| {
        const Program = struct {
            pub const Input = QA.Input;
            pub const Output = Result;

            qa: QA = .{},
            compiled: bool = false,

            pub fn forward(self: *const @This(), ctx: Context, input: Input) !Output {
                const prediction = try self.qa.forward(ctx, input);
                const number = try std.fmt.parseInt(i64, prediction.answer, 10);
                const raw: Value = .{ .integer = number };
                if (Output == Integer) return Integer.init(number);
                if (Output == Decimal) return Decimal.coerce(ctx.arena, raw);
                if (Output == Literal) return Literal.fromValue(raw);
                if (Output == Grade) return Grade.fromValue(try Grade.member(if (number == 7) "SEVEN" else "EIGHT"));
                return number;
            }

            fn score(ctx: Context, _: *const Input, output: *const Output) anyerror!f64 {
                const result = try value.toValue(ctx.arena, Output, output.*);
                return if (try value.numericOrder(result, .{ .integer = 7 }) == .eq) 7 else 8;
            }
        };
        for ([_]i64{ 7, 8 }) |threshold| {
            var memory: std.heap.ArenaAllocator = .init(testing.allocator);
            defer memory.deinit();
            const arena = memory.allocator();
            var lm: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>7</answer>", "<answer>8</answer>" } };
            var trace: Trace = .{ .arena = arena };
            var refined = Refine(Program).init(.{}, .{
                .n = 3,
                .reward_fn = Program.score,
                .threshold = .{ .integer = threshold },
                .module_code = "ScalarProgram",
                .reward_code = "score",
            });
            const selected = try refined.forward(.{
                .arena = arena,
                .lm = lm.lm(),
                .lm_options = .{ .rollout_id = .{ .integer = 7 } },
                .trace = &trace,
            }, .{ .question = "q" });
            const retried = threshold == 8;
            const result = try value.toValue(arena, Result, selected.?);
            try testing.expectEqual(@as(?std.math.Order, .eq), try value.numericOrder(result, .{ .integer = threshold }));
            try testing.expectEqual(@as(usize, if (retried) 2 else 1), lm.index);
            try testing.expectEqual(@as(i128, if (retried) 2 else 3), refined.fail_count);
            try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
            try testing.expectEqualStrings(if (retried) "8" else "7", trace.steps.items[0].outputs.get("answer").?.string);
        }
    }
}

test "Refine projects runtime dictionary fields and pairs without changing the selected result" {
    const QA = predict.Predict(TestSignature);
    const python = @import("python_value.zig");
    const Form = enum { object, list, tuple, invalid_key };
    const Program = struct {
        pub const Input = Value;
        pub const Output = Value;

        qa: QA = .{},
        form: Form,
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) !Output {
            const prediction = try self.qa.state.call(ctx, &input.object);
            const number = try std.fmt.parseInt(i64, prediction.get("answer").?.string, 10);
            const answer: Value = .{ .integer = number };
            if (number != 7) return answer;
            const reasoning = try value.toValue(ctx.arena, signature.Reasoning, .{ .content = "a plan" });
            if (self.form == .object) {
                var record: Record = .empty;
                try record.put(ctx.arena, "answer", answer);
                try record.put(ctx.arena, "reasoning", reasoning);
                return .{ .object = record };
            }
            const first_key: Value = if (self.form == .invalid_key) answer else .{ .string = "answer" };
            const first = try pair(ctx.arena, &.{ first_key, .{ .integer = 0 } });
            const middle: Value = .{ .array = .fromOwnedSlice(ctx.arena, try ctx.arena.dupe(Value, &.{ .{ .string = "reasoning" }, reasoning })) };
            const last = try pair(ctx.arena, &.{ .{ .string = "answer" }, answer });
            const entries = try ctx.arena.dupe(Value, &.{ first, middle, last });
            return if (self.form == .tuple)
                python.tupleFromOwnedSlice(ctx.arena, entries)
            else
                .{ .array = .fromOwnedSlice(ctx.arena, entries) };
        }

        fn pair(arena: Allocator, entries: []const Value) !Value {
            return python.tupleFromOwnedSlice(arena, try arena.dupe(Value, entries));
        }

        fn score(_: Context, _: *const Input, output: *const Output) anyerror!f64 {
            return if (output.* == .integer) @floatFromInt(output.integer) else 7;
        }
    };
    for ([_]Form{ .object, .list, .tuple, .invalid_key }) |form| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        const has_feedback = form != .invalid_key;
        var lm: lm_mod.ScriptedLM = .{ .responses = if (has_feedback) &.{
            "<answer>7</answer>",
            "<discussion>Try again.</discussion><advice>{}</advice>",
            "<answer>8</answer>",
        } else &.{ "<answer>7</answer>", "<answer>8</answer>" } };
        var trace: Trace = .{ .arena = arena };
        var refined = Refine(Program).init(.{ .form = form }, .{
            .n = 2,
            .reward_fn = Program.score,
            .threshold = .{ .integer = 8 },
            .module_code = "DynamicProgram",
            .reward_code = "score",
        });
        var kwargs: Record = .empty;
        try kwargs.put(arena, "question", .{ .string = "q" });
        try kwargs.put(arena, "extra_code", try @import("code.zig").Code.init("x = 1").toRuntime(arena));
        const selected = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .object = kwargs })).?;
        try testing.expectEqual(@as(i64, 8), selected.integer);
        try testing.expectEqual(@as(usize, if (has_feedback) 3 else 2), lm.index);
        try testing.expectEqual(@as(i128, if (has_feedback) 2 else 1), refined.fail_count);
        try testing.expectEqual(@as(usize, if (has_feedback) 2 else 1), trace.steps.items.len);
        if (has_feedback) {
            try testing.expectEqualStrings(
                "{\n  \"question\": \"q\",\n  \"extra_code\": \"<non-serializable: Code>\"\n}",
                trace.steps.items[0].inputs.get("program_inputs").?.string,
            );
            try testing.expectEqualStrings(
                "{\n  \"answer\": 7,\n  \"reasoning\": \"a plan\"\n}",
                trace.steps.items[0].inputs.get("program_outputs").?.string,
            );
        }
        try testing.expectEqualStrings("8", trace.steps.items[trace.steps.items.len - 1].outputs.get("answer").?.string);
    }
}

test "Refine needs named inputs only for feedback" {
    const QA = predict.Predict(TestSignature);
    const Program = struct {
        pub const Input = []const u8;
        pub const Output = QA.Output;

        qa: QA = .{},

        pub fn forward(self: *const @This(), ctx: Context, question: Input) !Output {
            return self.qa.forward(ctx, .{ .question = question });
        }

        fn score(_: Context, _: *const Input, output: *const Output) anyerror!f64 {
            return std.fmt.parseFloat(f64, output.answer);
        }
    };
    for ([_]i64{ 7, 8 }) |threshold| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var lm: lm_mod.ScriptedLM = .{ .responses = &.{ "<answer>7</answer>", "<answer>8</answer>" } };
        var trace: Trace = .{ .arena = arena };
        var refined = Refine(Program).init(.{}, .{
            .n = 3,
            .threshold = .{ .integer = threshold },
            .reward_fn = Program.score,
            .module_code = "ScalarInput",
            .reward_code = "score",
        });
        const result = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, "q")).?;
        const retried = threshold == 8;
        try testing.expectEqualStrings(if (retried) "8" else "7", result.answer);
        try testing.expectEqual(@as(usize, if (retried) 2 else 1), lm.index);
        try testing.expectEqual(@as(i128, if (retried) 2 else 3), refined.fail_count);
        try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
        try testing.expectEqualStrings("q", trace.steps.items[0].inputs.get("question").?.string);
        try testing.expectEqualStrings(result.answer, trace.steps.items[0].outputs.get("answer").?.string);
    }
}

test "Refine accepts native arrays and slices of runtime feedback pairs" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const pair: Value = .{ .array = .fromOwnedSlice(arena, try arena.dupe(Value, &.{
        .{ .string = "answer" },
        .{ .integer = 7 },
    })) };
    const pairs = [_]Value{pair};
    inline for (.{ @TypeOf(pairs), []const Value }) |Result| {
        const result: Result = if (Result == @TypeOf(pairs)) pairs else &pairs;
        const record = try predictionRecord(arena, Result, result);
        try testing.expectEqual(@as(usize, 1), record.count());
        try testing.expectEqual(@as(i64, 7), record.get("answer").?.integer);
    }
}

test "Refine sends typed dictionary entries rather than storage fields to feedback" {
    const QA = predict.Predict(TestSignature);
    const Code = @import("code.zig").Code;
    const Program = struct {
        pub const Input = signature.StrDict;
        pub const Output = @import("types.zig").Dict(Code);

        qa: QA = .{},
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: Context, input: Input) !Output {
            const answer = try self.qa.forward(ctx, .{ .question = input.map.get("question").?.string });
            const code = Code.init(try std.fmt.allocPrint(ctx.arena, "x = {s}", .{answer.answer}));
            return .{ .keys = &.{"implementation"}, .values = try ctx.arena.dupe(Code, &.{code}) };
        }

        fn score(_: Context, _: *const Input, output: *const Output) anyerror!f64 {
            return std.fmt.parseFloat(f64, output.get("implementation").?.code["x = ".len..]);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var lm: lm_mod.ScriptedLM = .{ .responses = &.{
        "<answer>7</answer>",
        "<discussion>Try again.</discussion><advice>{}</advice>",
        "<answer>8</answer>",
    } };
    var trace: Trace = .{ .arena = arena };
    var refined = Refine(Program).init(.{}, .{
        .n = 2,
        .reward_fn = Program.score,
        .threshold = .{ .integer = 8 },
        .module_code = "DictProgram",
        .reward_code = "score",
    });
    var kwargs: Record = .empty;
    try kwargs.put(arena, "question", .{ .string = "q" });
    const winner = (try refined.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = lm.lm(), .trace = &trace }, .{ .map = kwargs })).?;
    try testing.expectEqualStrings("x = 8", winner.get("implementation").?.code);
    try testing.expectEqual(@as(usize, 3), lm.index);
    try testing.expectEqual(@as(i128, 2), refined.fail_count);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expectEqualStrings("{\n  \"question\": \"q\"\n}", trace.steps.items[0].inputs.get("program_inputs").?.string);
    try testing.expectEqualStrings(
        "{\n  \"implementation\": \"<non-serializable: Code>\"\n}",
        trace.steps.items[0].inputs.get("program_outputs").?.string,
    );
}
