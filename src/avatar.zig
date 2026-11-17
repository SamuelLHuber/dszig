//! DSPy 3.5 `dspy.predict.avatar.Avatar` for native Zig programs: an actor
//! predictor chooses one `Action` per call, each tool result grows the actor
//! signature by `action_N`/`result_N`, and `Finish` swaps the next action for
//! the task's own outputs.
//!
//! Source's constructor cannot run in DSPy 3.5 (it calls the removed
//! `dspy.TypedPredictor`); the behavior here follows its unchanged methods.
//!
//! Intentional differences:
//! - Source never bounds the loop. `Options.actions_max` bounds tool actions;
//!   one more decision must be `Finish`, else `error.AvatarActionsExhausted`
//!   is returned before that tool runs. No final answer is fabricated.
//! - Every evolved signature layout is static memory. Evolved layouts carry a
//!   static class identity per layout, not a new class per call.
//! - A typed `Input` supplies every input; Source omits absent keyword inputs.
//! - A second call on an actor still evolved by a failure is refused with
//!   `error.AvatarActorEvolved` before any LM call; `reset` restores it.
//! - Tools are borrowed native callbacks, not deep-copied with the program.
//! - Source's constructor `max_iters` and `verbose` are unused or print only,
//!   so they have no field here.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const runtime_value = @import("runtime_value.zig");
const Value = runtime_value.Value;
const predict = @import("predict.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const Context = predict.Context;
const Field = signature.Field;
const Record = value.Record;

pub const finish_name = "Finish";
pub const finish_desc = "returns the final output and finishes the task";
pub const finish_result = "Gathered all information needed to finish the task.";

pub const actor_instructions =
    "You will be given `Tools` which will be a list of tools to use to accomplish the `Goal`. Given the user query, your task is to decide which tool to use and what input values to provide.\n\n" ++
    "You will output action needed to accomplish the `Goal`. `Action` should have a tool to use and the input query to pass to the tool.\n\n" ++
    "Note: You can opt to use no tools and provide the final answer directly. You can also one tool multiple times with different input queries if applicable.";

pub const ToolError = error{ OutOfMemory, ToolFailed };

pub const Error = predict.Error || ToolError || error{
    /// No tool has the requested name; no tool ran.
    AvatarUnknownTool,
    /// `tool_name` or `tool_input_query` is not a string. A named tool has
    /// already run with the raw query, as in Source.
    AvatarInvalidAction,
    /// The tool returned something other than a string.
    AvatarInvalidToolOutput,
    /// The named tool has no runner.
    AvatarToolMissing,
    AvatarActionsExhausted,
    AvatarActorEvolved,
    /// The actor answered without the expected output field.
    AvatarMissingOutput,
    AvatarTooManyTools,
};

/// `tool.tool.run(query)`. The query is the action's raw value, which is a
/// string only when the model produced one.
pub const Runner = struct {
    ptr: *anyopaque,
    run_fn: *const fn (ptr: *anyopaque, ctx: Context, query: Value) ToolError!Value,

    pub fn run(runner: Runner, ctx: Context, query: Value) ToolError!Value {
        return runner.run_fn(runner.ptr, ctx, query);
    }
};

/// `dspy.predict.avatar.Tool`. A returned string must outlive `ctx.arena`.
pub const Tool = struct {
    tool: ?Runner,
    name: []const u8,
    desc: ?[]const u8,
    input_type: ?[]const u8 = null,

    /// `str(tool)`, usable as `{f}`.
    pub fn format(tool: Tool, w: *Writer) Writer.Error!void {
        try w.writeAll(tool.name);
        if (tool.input_type) |input_type| {
            if (input_type.len != 0) try w.print("(valid_input: {s})", .{input_type});
        }
        try w.writeAll(": ");
        try w.writeAll(tool.desc orelse "None");
    }

    pub fn str(tool: Tool, arena: Allocator) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(arena, "{f}", .{tool});
    }
};

pub const finish_tool: Tool = .{ .tool = null, .name = finish_name, .desc = finish_desc };

/// `dspy.predict.avatar.Action`: both fields are `Any`.
pub const Action = struct {
    tool_name: Value,
    tool_input_query: Value,

    pub const model_name = "Action";
    pub const model_module = "dspy.predict.avatar";
    /// `json.dumps(_get_json_schema(Action))`, with Source's field descriptions.
    pub const schema = "{\"type\": \"object\", \"properties\": {" ++
        "\"tool_input_query\": {\"description\": \"Query to pass as input to the tool.\", \"title\": \"Tool Input Query\"}, " ++
        "\"tool_name\": {\"description\": \"Name of the tool to use.\", \"title\": \"Tool Name\"}}, " ++
        "\"required\": [\"tool_name\", \"tool_input_query\"], \"title\": \"Action\"}";
    pub const xml_schema = "{\"type\": \"object\", \"properties\": {" ++
        "\"tool_name\": {\"description\": \"Name of the tool to use.\", \"title\": \"Tool Name\"}, " ++
        "\"tool_input_query\": {\"description\": \"Query to pass as input to the tool.\", \"title\": \"Tool Input Query\"}}, " ++
        "\"required\": [\"tool_name\", \"tool_input_query\"], \"title\": \"Action\"}";
};

/// `dspy.predict.avatar.ActionOutput`: three strict strings.
pub const ActionOutput = struct {
    tool_name: []const u8,
    tool_input_query: []const u8,
    tool_output: []const u8,

    pub const model_name = "ActionOutput";
    pub const model_module = "dspy.predict.avatar";
};

pub const RunOptions = struct {
    /// Source's `max_iters` keyword. Null, zero and positive values never stop
    /// Source's loop; a negative value skips it, so the call cannot produce
    /// the task outputs. `Options.actions_max` is the only real bound.
    max_iters: ?i64 = null,
};

pub const Options = struct {
    /// Tool actions one call may take before it must finish.
    actions_max: usize = 8,
    tools_max: usize = 32,
};

/// The generic model protocol supplies Action's validators and renderers; only
/// its schema needs Source's per-field descriptions.
const ActorBase = signature.Make(
    struct { goal: []const u8, tools: []const []const u8 },
    struct { action_1: Action },
    actor_instructions,
    &.{
        .{ .name = "goal", .text = "Task to be accomplished." },
        .{ .name = "tools", .text = "list of tools to use" },
        .{ .name = "action_1", .text = "1st action to take." },
    },
);

/// Source's `get_number_with_suffix`: 11 is `11th`, 21 is `21th`.
fn ordinal(comptime number: usize) []const u8 {
    return switch (number) {
        1 => "1st",
        2 => "2nd",
        3 => "3rd",
        else => std.fmt.comptimePrint("{d}th", .{number}),
    };
}

fn isIndexed(comptime name: []const u8, comptime stem: []const u8) bool {
    if (!std.mem.startsWith(u8, name, stem) or name.len == stem.len) return false;
    for (name[stem.len..]) |byte| if (!std.ascii.isDigit(byte)) return false;
    return true;
}

const Layout = struct {
    fields: []const Field,
    input_count: usize,
    descs: []const []const u8,
    prefixes: []const []const u8,
};

fn makeLayout(comptime fields: []const Field, comptime input_count: usize) Layout {
    comptime {
        assert(input_count < fields.len);
        var descs: [fields.len][]const u8 = undefined;
        var prefixes: [fields.len][]const u8 = undefined;
        for (fields, 0..) |field, index| {
            descs[index] = field.desc;
            prefixes[index] = field.prefix;
        }
        const final_descs = descs;
        const final_prefixes = prefixes;
        return .{ .fields = fields, .input_count = input_count, .descs = &final_descs, .prefixes = &final_prefixes };
    }
}

pub fn Avatar(comptime Sig: type) type {
    return AvatarWithOptions(Sig, .{});
}

pub fn AvatarWithOptions(comptime Sig: type, comptime options: Options) type {
    comptime assert(signature.isSignature(Sig));
    comptime assert(options.actions_max >= 1);
    comptime assert(options.actions_max <= 1024);
    comptime assert(options.tools_max >= 1);
    const n_in = Sig.input_fields_list.len;
    const n_out = Sig.output_fields_list.len;
    comptime {
        for (Sig.input_fields_list ++ Sig.output_fields_list) |field| {
            for ([_][]const u8{ "goal", "tools", "actions", "max_iters" }) |reserved| {
                if (std.mem.eql(u8, field.name, reserved)) @compileError("dszig: Avatar reserves the field `" ++ reserved ++ "`");
            }
            if (isIndexed(field.name, "action_") or isIndexed(field.name, "result_")) {
                @compileError("dszig: Avatar reserves the field `" ++ field.name ++ "`");
            }
        }
    }
    // The last decision may only finish, so there is one more action field
    // than there are tool actions.
    const decisions = options.actions_max + 1;
    const prefix_count = 2 + n_in;
    const master_fields = comptime blk: {
        @setEvalBranchQuota(1_000_000);
        var fields: [prefix_count + 2 * decisions]Field = undefined;
        fields[0] = ActorBase.fields[0];
        fields[1] = ActorBase.fields[1];
        // Source appends the task inputs in reverse declaration order.
        for (0..n_in) |index| fields[2 + index] = Sig.input_fields_list[n_in - 1 - index];
        for (1..decisions + 1) |number| {
            var action = ActorBase.fields[2];
            action.name = std.fmt.comptimePrint("action_{d}", .{number});
            action.prefix = std.fmt.comptimePrint("Action {d}:", .{number});
            action.desc = if (number == 1) "1st action to take." else ordinal(number) ++ " action to taken";
            action.schema = Action.schema;
            action.xml_schema = Action.xml_schema;
            fields[prefix_count + 2 * (number - 1)] = action;
            fields[prefix_count + 2 * (number - 1) + 1] = .{
                .name = std.fmt.comptimePrint("result_{d}", .{number}),
                .kind = .str,
                .type_name = "str",
                .desc = ordinal(number) ++ " result",
                .prefix = std.fmt.comptimePrint("Result {d}:", .{number}),
            };
        }
        break :blk fields;
    };
    const master_layout = comptime makeLayout(&master_fields, master_fields.len - 1);
    // After `Finish` at decision k: every action/result up to k, then the
    // task outputs. Unlike a step layout this is not a prefix of the master.
    const final_layout_table = comptime blk: {
        @setEvalBranchQuota(1_000_000);
        var layouts: [decisions]Layout = undefined;
        for (0..decisions) |index| {
            const input_count = prefix_count + 2 * (index + 1);
            const fields = master_fields[0..input_count].* ++ Sig.output_fields_list[0..n_out].*;
            layouts[index] = makeLayout(&fields, input_count);
        }
        break :blk layouts;
    };

    const FinalOutput = comptime blk: {
        const info = @typeInfo(Sig.Output).@"struct";
        const names = info.field_names[0..info.field_names.len].* ++ [_][:0]const u8{"actions"};
        const field_types = info.field_types[0..info.field_types.len].* ++ [_]type{[]const ActionOutput};
        const attrs = info.field_attrs[0..info.field_attrs.len].* ++ [_]@TypeOf(info.field_attrs[0]){.{}};
        break :blk @Struct(.auto, null, &names, &field_types, &attrs);
    };

    return struct {
        const Self = @This();
        pub const Signature = Sig;
        pub const Input = Sig.Input;
        pub const Output = FinalOutput;
        pub const is_dszig_avatar = true;
        pub const actions_max = options.actions_max;
        pub const tools_max = options.tools_max;
        pub const actor_fields: []const Field = master_fields[0 .. prefix_count + 1];
        pub const Actor = predict.PredictFields(actor_fields, prefix_count, actor_instructions);

        const final_layouts = final_layout_table;
        /// Class identities of evolved layouts: steps 2.., then finals.
        var evolved_classes: [2 * decisions]u8 = @splat(0);

        /// The predictor every call runs and evolves.
        actor: Actor = .{},
        /// What `actor` becomes again after a successful call.
        actor_clone: Actor = .{},
        /// Borrowed; `Finish` is implicit and always last.
        tools: []const Tool = &.{},
        compiled: bool = false,

        /// Each program constructs a fresh actor signature class, shared by
        /// `actor`, `actor_clone` and program copies. The arena and `tools`
        /// must outlive the program and its copies.
        pub fn init(arena: Allocator, tools: []const Tool) (Allocator.Error || error{AvatarTooManyTools})!Self {
            if (tools.len > tools_max) return error.AvatarTooManyTools;
            var self: Self = .{ .tools = tools };
            try self.actor.state.forkSignatureClass(arena);
            self.actor_clone = self.actor;
            return self;
        }

        /// A failed call leaves `actor` evolved by its completed steps.
        pub fn evolved(self: *const Self) bool {
            return self.actor.state.fields.len != actor_fields.len;
        }

        /// `self.actor = deepcopy(self.actor_clone)`.
        pub fn reset(self: *Self) void {
            self.actor = self.actor_clone;
        }

        /// `signature.with_instructions(text)` on both actors, as the
        /// optimizer applies it. `text` must outlive the program.
        pub fn setInstructions(self: *Self, arena: Allocator, text: []const u8) Allocator.Error!void {
            assert(!self.evolved());
            var state = self.actor.state;
            try state.forkSignatureClass(arena);
            state.instructions = text;
            self.actor.state = state;
            self.actor_clone = self.actor;
        }

        /// One independent call: the program copy that evolves is discarded.
        pub fn forward(self: *const Self, ctx: Context, input: Input) Error!Output {
            var copy = self.*;
            return copy.call(ctx, input, .{});
        }

        /// Source's `forward(**kwargs)` on this instance. Results borrow
        /// `ctx.arena`. On failure `actor` keeps the signature its completed
        /// steps produced; only success restores it from `actor_clone`.
        pub fn call(self: *Self, ctx: Context, input: Input, run: RunOptions) Error!Output {
            if (self.evolved()) return error.AvatarActorEvolved;
            if (self.tools.len > tools_max) return error.AvatarTooManyTools;

            var args: Record = .empty;
            try args.ensureTotalCapacity(ctx.arena, prefix_count + 2 * decisions);
            args.putAssumeCapacity("goal", .{ .string = Sig.instructions });
            var names: runtime_value.Array = .init(ctx.arena);
            try names.ensureTotalCapacity(self.tools.len + 1);
            for (self.tools) |tool| names.appendAssumeCapacity(.{ .string = tool.name });
            names.appendAssumeCapacity(.{ .string = finish_name });
            args.putAssumeCapacity("tools", .{ .array = names });
            const inputs = try value.toRecord(ctx.arena, Input, input);
            inline for (Sig.input_fields_list) |field| args.putAssumeCapacity(field.name, inputs.get(field.name).?);

            const actions = try ctx.arena.alloc(ActionOutput, actions_max);
            var action_count: usize = 0;

            if (run.max_iters == null or run.max_iters.? >= 0) {
                var number: usize = 1;
                while (true) : (number += 1) {
                    assert(number <= decisions);
                    assert(action_count == number - 1);
                    const predicted = try self.actor.state.call(ctx, &args);
                    const action = predicted.get(master_fields[actionIndex(number)].name) orelse return error.AvatarMissingOutput;
                    if (action != .object) return error.AvatarInvalidAction;
                    const tool_name = action.object.get("tool_name") orelse return error.AvatarInvalidAction;
                    const query = action.object.get("tool_input_query") orelse return error.AvatarInvalidAction;
                    const finished = tool_name == .string and std.mem.eql(u8, tool_name.string, finish_name);
                    if (finished) {
                        self.setFinal(number);
                        args.putAssumeCapacity(master_fields[actionIndex(number)].name, action);
                        args.putAssumeCapacity(master_fields[actionIndex(number) + 1].name, .{ .string = finish_result });
                        break;
                    }
                    if (action_count == actions_max) return error.AvatarActionsExhausted;
                    actions[action_count] = try self.runTool(ctx, tool_name, query);
                    self.setStep(number + 1);
                    args.putAssumeCapacity(master_fields[actionIndex(number)].name, action);
                    args.putAssumeCapacity(master_fields[actionIndex(number) + 1].name, .{ .string = actions[action_count].tool_output });
                    action_count += 1;
                }
            }

            const final = try self.actor.state.call(ctx, &args);
            self.reset();
            // A skipped loop asked for `action_1`, never for the task outputs.
            if (run.max_iters != null and run.max_iters.? < 0) return error.AvatarMissingOutput;
            var output: Output = undefined;
            inline for (Sig.output_fields_list, @typeInfo(Sig.Output).@"struct".field_types) |field, T| {
                const item = final.get(field.name) orelse return error.AvatarMissingOutput;
                @field(output, field.name) = try value.fromValue(ctx.arena, T, item);
            }
            output.actions = actions[0..action_count];
            return output;
        }

        fn actionIndex(number: usize) usize {
            assert(number >= 1);
            assert(number <= decisions);
            return prefix_count + 2 * (number - 1);
        }

        /// The first matching tool wins. Source builds ActionOutput after the
        /// tool ran, so a named tool sees even a non-string query.
        fn runTool(self: *const Self, ctx: Context, tool_name: Value, query: Value) Error!ActionOutput {
            var output: ?Value = null;
            if (tool_name == .string) {
                for (self.tools) |tool| {
                    if (!std.mem.eql(u8, tool.name, tool_name.string)) continue;
                    const runner = tool.tool orelse return error.AvatarToolMissing;
                    output = try runner.run(ctx, query);
                    break;
                }
            }
            if (tool_name != .string or query != .string) return error.AvatarInvalidAction;
            const result = output orelse return error.AvatarUnknownTool;
            if (result != .string) return error.AvatarInvalidToolOutput;
            return .{ .tool_name = tool_name.string, .tool_input_query = query.string, .tool_output = result.string };
        }

        /// The actor now asks for decision `number`, given every earlier one.
        fn setStep(self: *Self, number: usize) void {
            assert(number >= 2);
            assert(number <= decisions);
            const end = actionIndex(number) + 1;
            const state = &self.actor.state;
            state.fields = master_fields[0..end];
            state.input_count = end - 1;
            state.descs = master_layout.descs[0..end];
            state.prefixes = master_layout.prefixes[0..end];
            state.signature_class = &evolved_classes[number - 2];
        }

        /// `Finish` at decision `number`: the actor now asks for the outputs.
        fn setFinal(self: *Self, number: usize) void {
            assert(number >= 1);
            assert(number <= decisions);
            const layout = final_layouts[number - 1];
            const state = &self.actor.state;
            state.fields = layout.fields;
            state.input_count = layout.input_count;
            state.descs = layout.descs;
            state.prefixes = layout.prefixes;
            state.signature_class = &evolved_classes[decisions - 1 + number];
        }
    };
}

const TestQA = signature.Signature(struct {
    pub const instructions = "Answer the question with the tools.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
    pub const descriptions = .{ .question = "user question", .answer = "short answer" };
});

const TestSearch = struct {
    calls: usize = 0,
    last: Value = .null,
    reply: Value = .null,

    fn runner(self: *TestSearch) Runner {
        return .{ .ptr = self, .run_fn = run };
    }

    fn run(ptr: *anyopaque, ctx: Context, query: Value) ToolError!Value {
        const self: *TestSearch = @ptrCast(@alignCast(ptr));
        self.calls += 1;
        self.last = query;
        if (self.reply != .null) return self.reply;
        if (query != .string) return .{ .string = "found" };
        return .{ .string = try std.fmt.allocPrint(ctx.arena, "found:{s}", .{query.string}) };
    }
};

fn testAction(comptime number: usize, comptime name: []const u8, comptime query_json: []const u8) []const u8 {
    return std.fmt.comptimePrint("<action_{d}>{{\"tool_name\": \"{s}\", \"tool_input_query\": {s}}}</action_{d}>", .{ number, name, query_json, number });
}

test "Tool strings spell absent descriptions and input types as Source does" {
    const testing = std.testing;
    var buffer: [128]u8 = undefined;
    try testing.expectEqualStrings("lookup(valid_input: text): look things up", try std.fmt.bufPrint(&buffer, "{f}", .{Tool{ .tool = null, .name = "lookup", .desc = "look things up", .input_type = "text" }}));
    try testing.expectEqualStrings("lookup: None", try std.fmt.bufPrint(&buffer, "{f}", .{Tool{ .tool = null, .name = "lookup", .desc = null, .input_type = "" }}));
    try testing.expectEqualStrings("Finish: returns the final output and finishes the task", try std.fmt.bufPrint(&buffer, "{f}", .{finish_tool}));
}

test "Avatar evolves the actor per tool result and restores it only on success" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var search: TestSearch = .{};
    const tools = [_]Tool{.{ .tool = search.runner(), .name = "search", .desc = "find facts" }};
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        testAction(1, "search", "\"alpha\""),
        testAction(2, "search", "\"beta\""),
        testAction(3, "Finish", "\"ready\""),
        "<answer>done</answer>",
    } };
    var recording: @import("lm.zig").RecordingLM = .{ .inner = script.lm(), .arena = arena };
    const P = Avatar(TestQA);
    var program = try P.init(arena, &tools);
    // An instruction set on the actor alone does not survive a success.
    program.actor.state.instructions = "ACTOR ONLY";
    const output = try program.call(.{ .arena = arena, .lm = recording.lm() }, .{ .question = "Q1" }, .{ .max_iters = 1 });

    try testing.expectEqualStrings("done", output.answer);
    // The budget of one never stopped the loop, and Finish is not an action.
    try testing.expectEqual(@as(usize, 2), output.actions.len);
    try testing.expectEqualStrings("search", output.actions[1].tool_name);
    try testing.expectEqualStrings("beta", output.actions[1].tool_input_query);
    try testing.expectEqualStrings("found:beta", output.actions[1].tool_output);
    try testing.expectEqual(@as(usize, 2), search.calls);
    try testing.expectEqual(@as(usize, 4), recording.calls.items.len);

    try testing.expect(!program.evolved());
    try testing.expectEqualStrings(actor_instructions, program.actor.state.instructions);
    try testing.expectEqual(program.actor_clone.state.signature_class, program.actor.state.signature_class);
    const other = try P.init(arena, &tools);
    try testing.expect(other.actor.state.signature_class != program.actor.state.signature_class);
    try testing.expect(predict.sameSignature(&other.actor.state, &program.actor.state));
    const predictors = try predict.namedPredictors(arena, &program);
    try testing.expectEqual(@as(usize, 2), predictors.len);
    try testing.expectEqualStrings("actor_clone", predictors[1].name);
}

test "Avatar failures keep completed evolution and refuse reuse until reset" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var search: TestSearch = .{};
    const tools = [_]Tool{
        .{ .tool = search.runner(), .name = "search", .desc = "find facts" },
        .{ .tool = null, .name = "broken", .desc = null },
    };
    const P = Avatar(TestQA);

    // An unknown tool after one completed step: nothing ran for it.
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{ testAction(1, "search", "\"alpha\""), testAction(2, "missing", "\"beta\"") } };
    var program = try P.init(arena, &tools);
    program.actor.state.instructions = "ACTOR ONLY";
    const ctx: Context = .{ .arena = arena, .lm = script.lm() };
    try testing.expectError(error.AvatarUnknownTool, program.call(ctx, .{ .question = "Q" }, .{}));
    try testing.expectEqual(@as(usize, 1), search.calls);
    try testing.expect(program.evolved());
    try testing.expectEqual(P.actor_fields.len + 2, program.actor.state.fields.len);
    try testing.expectEqualStrings("action_2", program.actor.state.fields[program.actor.state.input_count].name);
    try testing.expectEqualStrings("ACTOR ONLY", program.actor.state.instructions);
    try testing.expect(program.actor.state.signature_class != program.actor_clone.state.signature_class);
    try testing.expectError(error.AvatarActorEvolved, program.call(ctx, .{ .question = "Q" }, .{}));
    try testing.expectEqual(@as(usize, 2), script.index);
    program.reset();
    try testing.expect(!program.evolved());

    // A non-string query reaches the tool before the action is rejected, and
    // a failed first step leaves the actor unevolved.
    script = .{ .responses = &.{testAction(1, "search", "{\"k\": [1, 2]}")} };
    try testing.expectError(error.AvatarInvalidAction, program.call(ctx, .{ .question = "Q" }, .{}));
    try testing.expectEqual(@as(usize, 2), search.calls);
    try testing.expect(search.last == .object);
    try testing.expect(!program.evolved());

    script = .{ .responses = &.{testAction(1, "search", "\"alpha\"")} };
    search.reply = .{ .integer = 7 };
    try testing.expectError(error.AvatarInvalidToolOutput, program.call(ctx, .{ .question = "Q" }, .{}));
    search.reply = .null;

    script = .{ .responses = &.{testAction(1, "broken", "\"alpha\"")} };
    try testing.expectError(error.AvatarToolMissing, program.call(ctx, .{ .question = "Q" }, .{}));

    // A failed final call keeps the finished signature, outputs included.
    script = .{ .responses = &.{testAction(1, "Finish", "7")} };
    try testing.expectError(error.ScriptExhausted, program.call(ctx, .{ .question = "Q" }, .{}));
    try testing.expectEqualStrings("answer", program.actor.state.fields[program.actor.state.fields.len - 1].name);
    try testing.expectEqual(P.actor_fields.len + 1, program.actor.state.input_count);
    program.reset();

    // A negative budget skips the loop: one call, restored, no outputs.
    script = .{ .responses = &.{testAction(1, "search", "\"alpha\"")} };
    try testing.expectError(error.AvatarMissingOutput, program.call(ctx, .{ .question = "Q" }, .{ .max_iters = -1 }));
    try testing.expectEqual(@as(usize, 1), script.index);
    try testing.expectEqual(@as(usize, 3), search.calls);
    try testing.expect(!program.evolved());
}

test "Avatar refuses an action beyond its capacity before running the tool" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var search: TestSearch = .{};
    const tools = [_]Tool{.{ .tool = search.runner(), .name = "search", .desc = "find facts" }};
    const P = AvatarWithOptions(TestQA, .{ .actions_max = 1, .tools_max = 1 });
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{ testAction(1, "search", "\"alpha\""), testAction(2, "search", "\"beta\""), "<answer>invented</answer>" } };
    const program = try P.init(arena, &tools);
    try testing.expectError(error.AvatarActionsExhausted, program.forward(.{ .arena = arena, .lm = script.lm() }, .{ .question = "Q" }));
    try testing.expectEqual(@as(usize, 1), search.calls);
    try testing.expectEqual(@as(usize, 2), script.index);
    // forward ran on a copy, so the shared program is untouched.
    try testing.expect(!program.evolved());

    // At capacity, finishing is still possible.
    script = .{ .responses = &.{ testAction(1, "search", "\"alpha\""), testAction(2, "Finish", "\"\""), "<answer>done</answer>" } };
    const output = try program.forward(.{ .arena = arena, .lm = script.lm() }, .{ .question = "Q" });
    try testing.expectEqual(@as(usize, 1), output.actions.len);
    try testing.expectEqualStrings("done", output.answer);
    const two = [_]Tool{ tools[0], tools[0] };
    try testing.expectError(error.AvatarTooManyTools, P.init(arena, &two));
}
