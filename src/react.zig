//! DSPy 3.5 ReAct's text-tool path: one discoverable predictor, reusable
//! structured History, batched tool calls/results, and a typed `submit` tool.
//! Exhausted turns, malformed responses, empty calls, and context-window errors
//! get one final submit attempt, not a separate extraction predictor. All tools
//! remain native Zig callbacks. Async execution, JSON repair, and Python
//! traceback text are not supported.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Value = @import("runtime_value.zig").Value;
const adapter = @import("adapter.zig");
const predict = @import("predict.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const types = @import("types.zig");
const tool_protocol = @import("tool.zig");
const Context = predict.Context;
const Field = signature.Field;
const Record = value.Record;

pub const History = @import("history.zig").History;
pub const ToolCall = tool_protocol.ToolCall;
pub const ToolCalls = tool_protocol.ToolCalls;
pub const ToolCallResult = tool_protocol.ToolCallResult;
pub const ToolCallResults = tool_protocol.ToolCallResults;
pub const ToolError = error{ OutOfMemory, ToolFailed, InvalidArguments };
pub const Error = predict.Error || error{ ReActNoSubmit, ReActSubmitFailed, InvalidMaxIters };
pub const TerminationReason = enum { submit, forced_submit };

pub const RunOptions = struct {
    /// Appended in place, including failed final submit events. New events use
    /// Context.arena; clone the History before releasing that arena.
    history: ?*History = null,
    max_iters: ?i128 = null,
};

pub const Arg = struct {
    name: []const u8,
    /// Basic JSON-schema type, or `Any` for an unconstrained argument.
    schema_type: []const u8 = "Any",
    /// JSON or Python scalar repr of the callback's default. Rich defaults can
    /// be supplied in schema instead. The callback supplies omitted defaults.
    default: ?[]const u8 = null,
    description: ?[]const u8 = null,
    /// A complete schema and matching native validator, normally from typed().
    schema: []const u8 = "",
    validate: ?*const fn (Allocator, Value) types.Error!Value = null,

    pub fn typed(comptime name: []const u8, comptime T: type) Arg {
        return .{
            .name = name,
            .schema_type = schemaType(T),
            .schema = typeSchema(T),
            .validate = TypedValidator(T).run,
        };
    }
};

pub const Tool = struct {
    name: []const u8,
    desc: []const u8 = "",
    args: []const Arg = &.{},
    func: *const fn (ctx: Context, args: *const Record) ToolError!Value,
    has_kwargs: bool = false,

    /// Shared native invocation, including typed argument normalization.
    pub fn call(comptime self: Tool, ctx: Context, input: *const Record) ToolError!Value {
        const sink = ctx.status_sink orelse return self.callCore(ctx, input);
        sink.notify(.{ .tool_start = .{ .name = self.name, .input = input } });
        const result = self.callCore(ctx, input);
        if (result) |output| {
            sink.notify(.{ .tool_end = .{ .output = output } });
        } else |failure| {
            sink.notify(.{ .tool_end = .{ .failure = failure } });
        }
        return result;
    }

    fn callCore(comptime self: Tool, ctx: Context, input: *const Record) ToolError!Value {
        const normalized = try normalizeArgs(self, ctx.arena, input);
        return self.func(ctx, &normalized);
    }

    /// `str(dspy.Tool)`, including full typed argument schemas.
    pub fn format(comptime self: Tool, arena: Allocator) adapter.Error![]const u8 {
        return tool_protocol.formatToolValue(arena, try self.toValue(arena));
    }

    /// Shared typed metadata: rendered as Tool strings for text adapters, or
    /// converted to FunctionTool parameters for a capable native LM.
    pub fn toValue(comptime self: Tool, arena: Allocator) adapter.Error!Value {
        var args: Record = .empty;
        try args.ensureTotalCapacity(arena, self.args.len);
        inline for (self.args) |arg| {
            var schema: Record = .empty;
            if (arg.schema.len > 0) {
                const parsed = std.json.parseFromSliceLeaky(Value, arena, arg.schema, .{}) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.AdapterParse,
                };
                if (parsed != .object) return error.AdapterParse;
                schema = parsed.object;
            } else if (!std.mem.eql(u8, arg.schema_type, "Any")) {
                try schema.put(arena, "type", .{ .string = arg.schema_type });
            }
            if (arg.default) |default| try schema.put(arena, "default", try parseDefault(arena, default));
            if (arg.description) |description| try schema.put(arena, "description", .{ .string = description });
            args.putAssumeCapacity(arg.name, .{ .object = schema });
        }
        var out: Record = .empty;
        try out.ensureTotalCapacity(arena, 3);
        out.putAssumeCapacity("name", .{ .string = self.name });
        out.putAssumeCapacity("desc", .{ .string = self.desc });
        out.putAssumeCapacity("args", .{ .object = args });
        return .{ .object = out };
    }
};

pub fn ReAct(comptime Sig: type, comptime user_tools: []const Tool) type {
    return ReActIters(Sig, user_tools, 20);
}

pub fn ReActIters(comptime Sig: type, comptime user_tools: []const Tool, comptime default_max_iters: i128) type {
    comptime assert(signature.isSignature(Sig));
    const n_in = Sig.input_fields_list.len;
    comptime {
        for (Sig.input_fields_list) |field| {
            for ([_][]const u8{ "history", "tools", "turn_budget" }) |reserved| {
                if (std.mem.eql(u8, field.name, reserved)) @compileError("dszig: ReAct reserves the input field `" ++ reserved ++ "`");
            }
        }
        for (Sig.output_fields_list) |field| {
            for ([_][]const u8{ "history", "termination_reason" }) |reserved| {
                if (std.mem.eql(u8, field.name, reserved)) @compileError("dszig: ReAct reserves the output field `" ++ reserved ++ "`");
            }
        }
    }
    const tools = comptime blk: {
        var unique: []const Tool = &.{};
        for (user_tools, 0..) |tool, index| {
            if (std.mem.eql(u8, tool.name, "submit")) @compileError("dszig: ReAct reserves the tool name `submit`");
            for (tool.args) |arg| {
                if (arg.schema.len > 0 and arg.validate == null) @compileError("dszig: a tool argument schema needs its native validator; use Arg.typed");
            }
            var seen = false;
            for (user_tools[0..index]) |previous| {
                if (std.mem.eql(u8, previous.name, tool.name)) seen = true;
            }
            if (seen) continue;
            var selected = tool;
            for (user_tools[index + 1 ..]) |later| {
                if (std.mem.eql(u8, later.name, tool.name)) selected = later;
            }
            unique = unique ++ [_]Tool{selected};
        }
        break :blk unique ++ [_]Tool{submitTool(Sig)};
    };
    const instructions = comptime blk: {
        var names: []const u8 = "";
        for (tools, 0..) |tool, index| names = names ++ (if (index > 0) ", " else "") ++ "`" ++ tool.name ++ "`";
        break :blk value.pyStrip(Sig.instructions ++ "\n" ++
            "You are an Agent. Use the supplied tools to produce " ++ backticked(Sig.output_fields_list) ++ " from " ++ backticked(Sig.input_fields_list) ++ ".\n" ++
            "Call tools when more information is needed.\n" ++
            "When the final answer is ready, call `submit` with " ++ backticked(Sig.output_fields_list) ++ ".\n" ++
            "The available tools are: " ++ names ++ ".");
    };
    const optional_inputs = comptime blk: {
        var fields: [n_in]Field = undefined;
        const input_types = @typeInfo(Sig.Input).@"struct".field_types;
        for (Sig.input_fields_list, input_types, 0..) |field, T, index| {
            const Optional = if (@typeInfo(T) == .optional) T else ?T;
            fields[index] = field;
            fields[index].kind = .json;
            fields[index].type_name = types.pyName(Optional);
            fields[index].schema = typeSchema(Optional);
            fields[index].validate = TypedValidator(Optional).run;
            fields[index].xml_validate = types.XmlValidator(Optional);
            fields[index].python_str = @import("python_str.zig").renderer(Optional);
            fields[index].refine_mask = @import("feedback.zig").masker(Optional);
            fields[index].optional_str = types.isOptionalStr(Optional);
            fields[index].nullable = true;
            fields[index].code_type = false;
            fields[index].dspy_type = false;
            fields[index].parse_value = null;
            fields[index].native_reasoning = false;
            fields[index].native_citations = false;
            fields[index].schema_definition_name = "";
            fields[index].xml_literal = field.xml_literal or Optional == ?signature.Reasoning;
        }
        break :blk fields;
    };

    const FinalOutput = comptime blk: {
        const info = @typeInfo(Sig.Output).@"struct";
        const names = info.field_names[0..info.field_names.len].* ++ [_][:0]const u8{ "history", "termination_reason" };
        const field_types = info.field_types[0..info.field_types.len].* ++ [_]type{ History, TerminationReason };
        const attrs = info.field_attrs[0..info.field_attrs.len].* ++ [_]@TypeOf(info.field_attrs[0]){ .{}, .{} };
        break :blk @Struct(.auto, null, &names, &field_types, &attrs);
    };

    return struct {
        const Self = @This();
        pub const Input = Sig.Input;
        pub const Output = FinalOutput;
        pub const all_tools: []const Tool = &tools;
        pub const react_instructions = instructions;
        const react_field_array = optional_inputs ++ [_]Field{
            .{ .name = "history", .kind = .json, .type_name = "History", .desc = "${history}", .prefix = "History:", .schema = History.schema, .validate = History.validate, .dspy_type = true, .refine_mask = @import("feedback.zig").masker(History), .python_str = @import("python_str.zig").renderer(History) },
            .{ .name = "tools", .kind = .json, .type_name = "list[Tool]", .desc = "${tools}", .prefix = "Tools:", .schema = "{\"type\": \"array\", \"items\": {\"type\": \"string\"}}", .xml_literal = true, .native_tools = true, .refine_mask = @import("feedback.zig").masker([]const Tool), .python_str = @import("python_str.zig").toolsRender },
            .{ .name = "turn_budget", .kind = .str, .type_name = "str", .desc = "Your current turn and how many turns remain before you must submit.", .prefix = "Turn Budget:" },
            .{ .name = "next_thought", .kind = .str, .type_name = "str", .desc = "${next_thought}", .prefix = "Next Thought:", .reasoning_type = true, .native_reasoning = true, .dspy_type = true, .refine_mask = @import("feedback.zig").masker(signature.Reasoning), .python_str = @import("python_str.zig").renderer(signature.Reasoning) },
            .{ .name = "tool_calls", .kind = .json, .type_name = "ToolCalls", .desc = "${tool_calls}", .prefix = "Tool Calls:", .schema = ToolCalls.schema, .validate = ToolCalls.validate, .dspy_type = true, .xml_literal = true, .native_tool_calls = true, .type_description = ToolCalls.description, .refine_mask = @import("feedback.zig").masker(ToolCalls), .python_str = @import("python_str.zig").renderer(ToolCalls) },
        };
        pub const react_fields: []const Field = &react_field_array;

        react: predict.PredictFields(react_fields, n_in + 3, instructions),
        /// Nonpositive limits skip tool turns and go directly to forced submit.
        max_iters: i128 = default_max_iters,
        compiled: bool = false,

        /// Each agent constructs a fresh derived signature class. Copies retain
        /// that identity; the arena must outlive the agent and its copies.
        pub fn init(arena: Allocator) Allocator.Error!Self {
            var self: Self = .{ .react = .{} };
            try self.react.state.forkSignatureClass(arena);
            return self;
        }

        pub const Prediction = struct {
            outputs: Record,
            history: History,
            termination_reason: TerminationReason,
        };

        pub fn forward(self: *const Self, ctx: Context, input: Input) Error!Output {
            const inputs = try value.toRecord(ctx.arena, Input, input);
            const prediction = try self.run(ctx, &inputs);
            const submitted = try value.fromRecord(ctx.arena, Sig.Output, &prediction.outputs);
            var output: Output = undefined;
            inline for (Sig.output_fields_list) |field| @field(output, field.name) = @field(submitted, field.name);
            output.history = prediction.history;
            output.termination_reason = prediction.termination_reason;
            return output;
        }

        /// Dynamic inputs may include `history` and `max_iters`, just like DSPy.
        pub fn run(self: *const Self, ctx: Context, inputs: *const Record) Error!Prediction {
            return self.runWithOptions(ctx, inputs, .{});
        }

        pub fn runWithOptions(self: *const Self, ctx: Context, inputs: *const Record, options: RunOptions) Error!Prediction {
            const limit = options.max_iters orelse try iterationLimit(inputs, self.max_iters);
            const max_iters = std.math.cast(usize, @max(0, limit)) orelse return error.InvalidMaxIters;
            var local_history: History = if (options.history == null) blk: {
                const supplied = inputs.get("history") orelse break :blk .{};
                if (supplied == .null) break :blk .{};
                break :blk try History.fromValue(ctx.arena, supplied);
            } else .{};
            const history = options.history orelse &local_history;
            var trace_history = TraceHistory.init(ctx.trace);
            var events: std.ArrayList(Record) = .empty;
            try events.appendSlice(ctx.arena, history.messages);
            var pending: Record = .empty;
            inline for (Sig.input_fields_list) |field| {
                if (!std.mem.eql(u8, field.name, "max_iters")) {
                    if (inputs.get(field.name)) |item| try pending.put(ctx.arena, field.name, item);
                }
            }
            const catalog = try toolCatalog(ctx.arena);
            turns: for (0..max_iters) |turn_index| {
                const turn_inputs = try withBudget(ctx.arena, &pending, try turnBudget(ctx.arena, turn_index, max_iters));
                const step_inputs = try callInputs(ctx.arena, history.*, catalog, &turn_inputs);
                const pred = self.react.state.call(ctx, &step_inputs) catch |err| switch (err) {
                    error.AdapterParse, error.ContextWindowExceeded => break :turns,
                    else => return err,
                };
                var calls = ToolCalls.fromValue(ctx.arena, pred.get("tool_calls") orelse .null) catch |err| switch (err) {
                    error.ParseValue => break :turns,
                    error.OutOfMemory => return error.OutOfMemory,
                };
                if (calls.tool_calls.len == 0) break :turns; // empty_tool_calls.
                calls = try calls.ensureIds(ctx.arena, turn_index);
                const execution = try execute(ctx, calls);
                const event = try historyEvent(ctx.arena, &turn_inputs, &pred, calls, execution);
                try appendEvent(ctx.arena, &events, history, event);
                try trace_history.append(ctx, &self.react.state, &event);
                pending = .empty;
                if (execution.final_outputs) |outputs| return .{
                    .outputs = outputs,
                    .history = history.*,
                    .termination_reason = .submit,
                };
            }
            // Recovery retains history and the as-yet-unpublished inputs.
            return self.forcedSubmit(ctx, history, &events, &pending, catalog, limit, &trace_history);
        }

        fn forcedSubmit(self: *const Self, ctx_in: Context, history: *History, events: *std.ArrayList(Record), pending: *const Record, catalog: Value, turn_index: i128, trace_history: *TraceHistory) Error!Prediction {
            var ctx = ctx_in;
            ctx.lm_options.reasoning_effort = null;
            ctx.lm_options.reasoning_effort_present = true;
            ctx.lm_options.tool_choice = .{ .function = "submit" };
            const turn_inputs = try withBudget(ctx.arena, pending, "No turns remain. Call `submit` now with your final outputs.");
            const step_inputs = try callInputs(ctx.arena, history.*, catalog, &turn_inputs);
            const pred = try self.react.state.call(ctx, &step_inputs);
            var calls = ToolCalls.fromValue(ctx.arena, pred.get("tool_calls") orelse .null) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.ParseValue => error.AdapterParse,
            };
            calls = try calls.ensureIds(ctx.arena, turn_index);
            calls = try calls.named(ctx.arena, "submit");
            if (calls.tool_calls.len == 0) return error.ReActNoSubmit;
            const execution = try execute(ctx, calls);
            const event = try historyEvent(ctx.arena, &turn_inputs, &pred, calls, execution);
            try appendEvent(ctx.arena, events, history, event);
            try trace_history.append(ctx, &self.react.state, &event);
            return .{
                .outputs = execution.final_outputs orelse return error.ReActSubmitFailed,
                .history = history.*,
                .termination_reason = .forced_submit,
            };
        }

        fn toolCatalog(arena: Allocator) adapter.Error!Value {
            var catalog: @import("runtime_value.zig").Array = .init(arena);
            try catalog.ensureTotalCapacity(tools.len);
            inline for (tools) |tool| catalog.appendAssumeCapacity(try tool.toValue(arena));
            return .{ .array = catalog };
        }

        const Execution = struct { results: ToolCallResults, final_outputs: ?Record = null };

        fn execute(ctx: Context, calls: ToolCalls) Allocator.Error!Execution {
            const results = try ctx.arena.alloc(ToolCallResult, calls.tool_calls.len);
            var final_outputs: ?Record = null;
            for (calls.tool_calls, results) |call, *result| {
                result.* = .{ .call_id = call.id, .name = call.name, .value = undefined };
                const observed = try observe(ctx, call);
                result.value = observed.value;
                result.is_error = observed.is_error;
                if (!observed.is_error and std.mem.eql(u8, call.name, "submit") and observed.value == .object) final_outputs = observed.value.object;
            }
            return .{ .results = .{ .tool_call_results = results }, .final_outputs = final_outputs };
        }

        const Observation = struct { value: Value, is_error: bool = false };

        fn observe(ctx: Context, call: ToolCall) Allocator.Error!Observation {
            inline for (tools) |tool| {
                if (std.mem.eql(u8, tool.name, call.name)) {
                    const result = tool.call(ctx, &call.args);
                    return .{ .value = result catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return .{
                            .value = .{ .string = try std.fmt.allocPrint(ctx.arena, "Execution error in {s}: {s}", .{ call.name, @errorName(err) }) },
                            .is_error = true,
                        },
                    } };
                }
            }
            return .{ .value = .{ .string = try std.fmt.allocPrint(ctx.arena, "Unknown tool: {s}", .{call.name}) }, .is_error = true };
        }

        fn historyEvent(arena: Allocator, turn_inputs: *const Record, pred: *const Record, calls: ToolCalls, execution: Execution) Allocator.Error!Record {
            var event = try turn_inputs.clone(arena);
            if (pred.get("next_thought")) |thought| {
                if (thought != .null) {
                    // Keep the actual class; JSON serialization invokes its
                    // content serializer without changing stored trace values.
                    try event.put(arena, "next_thought", thought);
                }
            }
            var recorded_calls = calls;
            if (execution.results.tool_call_results.len > 0) recorded_calls.tool_call_results = execution.results;
            try event.put(arena, "tool_calls", try recorded_calls.toValue(arena));
            if (execution.final_outputs) |outputs| {
                for (outputs.keys(), outputs.values()) |name, item| try event.put(arena, name, item);
            }
            return event;
        }
    };
}

fn submitTool(comptime Sig: type) Tool {
    const args = comptime blk: {
        var out: [Sig.output_fields_list.len]Arg = undefined;
        const output_types = @typeInfo(Sig.Output).@"struct".field_types;
        for (Sig.output_fields_list, output_types, 0..) |field, T, index| {
            out[index] = Arg.typed(field.name, T);
            out[index].description = field.desc;
        }
        const final = out;
        break :blk &final;
    };
    return .{
        .name = "submit",
        .desc = "Submit the final outputs for the task.",
        .args = args,
        .has_kwargs = true,
        .func = struct {
            fn call(ctx: Context, input: *const Record) ToolError!Value {
                var outputs: Record = .empty;
                try outputs.ensureTotalCapacity(ctx.arena, Sig.output_fields_list.len);
                inline for (Sig.output_fields_list) |field| outputs.putAssumeCapacity(field.name, input.get(field.name) orelse return error.InvalidArguments);
                return .{ .object = outputs };
            }
        }.call,
    };
}

fn normalizeArgs(comptime tool: Tool, arena: Allocator, input: *const Record) ToolError!Record {
    var out: Record = .empty;
    try out.ensureTotalCapacity(arena, input.count());
    next: for (input.keys(), input.values()) |name, item| {
        inline for (tool.args) |arg| {
            if (std.mem.eql(u8, arg.name, name)) {
                if (!schemaAccepts(arg.schema_type, item)) return error.InvalidArguments;
                const parsed = if (arg.validate) |validate| validate(arena, item) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    error.ParseValue => error.InvalidArguments,
                } else try normalizeBasicArg(arg.schema_type, item);
                out.putAssumeCapacity(name, parsed);
                continue :next;
            }
        }
        if (!tool.has_kwargs) return error.InvalidArguments;
        out.putAssumeCapacity(name, item);
    }
    inline for (tool.args) |arg| {
        if (arg.default == null and !out.contains(arg.name)) {
            const has_default = if (arg.schema.len > 0) blk: {
                const schema = std.json.parseFromSliceLeaky(Value, arena, arg.schema, .{}) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidArguments,
                };
                break :blk schema == .object and schema.object.contains("default");
            } else false;
            if (!has_default) return error.InvalidArguments;
        }
    }
    return out;
}

fn normalizeBasicArg(schema_type: []const u8, item: Value) ToolError!Value {
    if (std.mem.eql(u8, schema_type, "integer") and item == .float) {
        if (item.float < -9223372036854775808.0 or item.float >= 9223372036854775808.0) return error.InvalidArguments;
        return .{ .integer = @intFromFloat(item.float) };
    }
    if (std.mem.eql(u8, schema_type, "integer") and item == .number_string)
        return value.parseInteger(item) catch error.InvalidArguments;
    if (std.mem.eql(u8, schema_type, "number")) return switch (item) {
        .integer => |integer| .{ .float = @floatFromInt(integer) },
        .number_string => |text| .{ .float = std.fmt.parseFloat(f64, text) catch return error.InvalidArguments },
        else => item,
    };
    return item;
}

fn schemaAccepts(schema_type: []const u8, item: Value) bool {
    if (std.mem.eql(u8, schema_type, "string")) return item == .string;
    if (std.mem.eql(u8, schema_type, "integer")) return switch (item) {
        .integer => true,
        .float => |float| std.math.isFinite(float) and float == @trunc(float),
        .number_string => blk: {
            _ = value.parseInteger(item) catch break :blk false;
            break :blk true;
        },
        else => false,
    };
    if (std.mem.eql(u8, schema_type, "number")) return item == .integer or item == .float or item == .number_string;
    if (std.mem.eql(u8, schema_type, "boolean")) return item == .bool;
    if (std.mem.eql(u8, schema_type, "array")) return item == .array;
    if (std.mem.eql(u8, schema_type, "object")) return item == .object;
    return true;
}

fn schemaType(comptime T: type) []const u8 {
    if (T == []const u8) return "string";
    if (T == bool) return "boolean";
    if (T == Record or T == signature.StrDict or types.isDict(T)) return "object";
    return switch (@typeInfo(T)) {
        .int => "integer",
        .float => "number",
        .@"enum" => "string",
        .pointer => "array",
        .@"struct" => "object",
        .optional => "Any",
        else => @compileError("dszig: unsupported tool argument type " ++ @typeName(T)),
    };
}

fn typeSchema(comptime T: type) []const u8 {
    if (T == signature.StrDict) return types.schema(types.Dict([]const u8));
    if (@typeInfo(T) == .optional and @typeInfo(T).optional.child == signature.StrDict) return types.schema(?types.Dict([]const u8));
    return types.schema(T);
}

fn TypedValidator(comptime T: type) type {
    return struct {
        fn run(arena: Allocator, input: Value) types.Error!Value {
            // Tool validates a schema only when its root has a `type` key.
            // Optional roots use Pydantic's lax conversion, matching upstream.
            if (@typeInfo(T) != .optional and !(try strictAccepts(T, input))) return error.ParseValue;
            if (T == signature.StrDict) return input;
            if (@typeInfo(T) == .optional and @typeInfo(T).optional.child == signature.StrDict) {
                if (input != .null and !(try strictAccepts(signature.StrDict, input))) return error.ParseValue;
                return input;
            }
            const parsed = try types.validate(T, arena, input);
            if (!nativeFits(T, parsed)) return error.ParseValue;
            return parsed;
        }
    };
}

fn strictAccepts(comptime T: type, input: Value) value.ParseError!bool {
    if (T == Value) return true;
    if (T == @import("integer.zig").Integer) return schemaAccepts("integer", input);
    if (comptime @import("enum.zig").isEnum(T)) {
        if (T.dszig_enum_json_type) |kind| {
            if (!schemaAccepts(kind, input)) return false;
            for (T.dszig_enum_values) |member| if (try value.pyEqual(member, input)) return true;
            return false;
        }
        return true;
    }
    if (comptime @import("code.zig").isCode(T) or T == signature.Reasoning or
        T == @import("file.zig").File or T == @import("bounding_box.zig").BoundingBox)
    {
        _ = T.fromValue(input) catch return false;
        return true;
    }
    if (T == []const u8) return input == .string;
    if (T == bool) return input == .bool;
    if (T == Record) return input == .object;
    if (T == signature.StrDict) {
        if (input != .object) return false;
        for (input.object.values()) |item| if (item != .string) return false;
        return true;
    }
    if (comptime types.isDict(T)) {
        if (input != .object) return false;
        for (input.object.values()) |item| if (!(try strictAccepts(T.dszig_dict_value, item))) return false;
        return true;
    }
    return switch (@typeInfo(T)) {
        .int => schemaAccepts("integer", input),
        .float => schemaAccepts("number", input),
        .@"enum" => input == .string and std.meta.stringToEnum(T, input.string) != null,
        .optional => |optional| input == .null or try strictAccepts(optional.child, input),
        .pointer => |pointer| blk: {
            if (input != .array) break :blk false;
            for (input.array.items) |item| if (!(try strictAccepts(pointer.child, item))) break :blk false;
            break :blk true;
        },
        .@"struct" => |info| blk: {
            if (input != .object) break :blk false;
            inline for (info.field_names, info.field_types, 0..) |name, FT, index| {
                if (input.object.get(name)) |item| {
                    if (!(try strictAccepts(FT, item))) break :blk false;
                } else if (info.field_attrs[index].default_value_ptr == null) break :blk false;
            }
            break :blk true;
        },
        else => @compileError("dszig: unsupported tool argument type " ++ @typeName(T)),
    };
}

/// JSON schemas describe Python integers, whereas native callbacks/output
/// structs can use narrower integer storage. Reject overflow before submission.
fn nativeFits(comptime T: type, input: Value) bool {
    if (T == Value) return true;
    if (T == @import("integer.zig").Integer) return true;
    if (comptime @import("enum.zig").isEnum(T)) {
        _ = T.fromValue(input) catch return false;
        return true;
    }
    if (comptime @import("code.zig").isCode(T) or T == signature.Reasoning or
        T == @import("file.zig").File or T == @import("bounding_box.zig").BoundingBox)
    {
        _ = T.fromValue(input) catch return false;
        return true;
    }
    if (T == []const u8 or T == bool or T == Record or T == signature.StrDict) return true;
    if (comptime types.isDict(T)) {
        for (input.object.values()) |item| if (!nativeFits(T.dszig_dict_value, item)) return false;
        return true;
    }
    return switch (@typeInfo(T)) {
        .int => switch (input) {
            .integer => |integer| std.math.cast(T, integer) != null,
            .number_string => |text| blk: {
                _ = std.fmt.parseInt(T, text, 10) catch break :blk false;
                break :blk true;
            },
            else => false,
        },
        .float, .@"enum" => true,
        .optional => |optional| input == .null or nativeFits(optional.child, input),
        .pointer => |pointer| blk: {
            for (input.array.items) |item| if (!nativeFits(pointer.child, item)) break :blk false;
            break :blk true;
        },
        .@"struct" => |info| blk: {
            inline for (info.field_names, info.field_types) |name, FT| {
                if (!nativeFits(FT, input.object.get(name).?)) break :blk false;
            }
            break :blk true;
        },
        else => @compileError("dszig: unsupported tool argument type " ++ @typeName(T)),
    };
}

fn backticked(comptime fields: []const Field) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (fields, 0..) |field, index| out = out ++ (if (index > 0) ", " else "") ++ "`" ++ field.name ++ "`";
        return out;
    }
}

fn iterationLimit(inputs: *const Record, default: i128) Error!i128 {
    const supplied = inputs.get("max_iters") orelse return default;
    return switch (supplied) {
        .integer => supplied.integer,
        else => error.InvalidMaxIters,
    };
}

fn turnBudget(arena: Allocator, turn_index: usize, max_iters: usize) Allocator.Error![]const u8 {
    assert(turn_index < max_iters);
    const turn = turn_index + 1;
    const remaining = max_iters - turn;
    return if (remaining == 0)
        std.fmt.allocPrint(arena, "Turn {d} of {d} (this is your last turn).", .{ turn, max_iters })
    else
        std.fmt.allocPrint(arena, "Turn {d} of {d} ({d} more after this one).", .{ turn, max_iters, remaining });
}

fn withBudget(arena: Allocator, pending: *const Record, budget: []const u8) Allocator.Error!Record {
    var out = try pending.clone(arena);
    try out.put(arena, "turn_budget", .{ .string = budget });
    return out;
}

fn callInputs(arena: Allocator, history: History, catalog: Value, turn_inputs: *const Record) Allocator.Error!Record {
    var out: Record = .empty;
    try out.ensureTotalCapacity(arena, turn_inputs.count() + 2);
    out.putAssumeCapacity("history", try history.toValue(arena));
    out.putAssumeCapacity("tools", catalog);
    for (turn_inputs.keys(), turn_inputs.values()) |name, item| out.putAssumeCapacity(name, item);
    return out;
}

/// DSPy traces retain the same mutable History object passed to every turn.
/// Link only this run's matching predictor steps, and copy each appended event
/// once into the trace arena rather than repeatedly cloning its whole prefix.
const TraceHistory = struct {
    next_step: usize = 0,
    evicted_steps: usize = 0,
    shared: ?Value = null,

    fn init(trace: ?*const predict.Trace) TraceHistory {
        const active = trace orelse return .{};
        return .{ .next_step = active.steps.items.len, .evicted_steps = active.evicted_steps };
    }

    fn append(self: *TraceHistory, ctx: Context, predictor: *const predict.PredictState, event: *const Record) Allocator.Error!void {
        const trace = ctx.trace orelse return;
        const first_new = self.next_step -| (trace.evicted_steps -% self.evicted_steps);
        assert(first_new <= trace.steps.items.len);
        for (trace.steps.items[first_new..]) |*step| {
            if (step.predictor != predictor) continue;
            const history = step.inputs.getPtr("history") orelse continue;
            if (self.shared == null) self.shared = history.*;
            history.* = self.shared.?;
        }
        self.next_step = trace.steps.items.len;
        self.evicted_steps = trace.evicted_steps;
        if (self.shared) |*shared| {
            assert(shared.* == .object);
            const messages = shared.object.getPtr("messages").?;
            assert(messages.* == .array);
            try messages.array.append(.{ .object = try value.cloneRecord(trace.arena, event) });
        }
    }
};

fn appendEvent(arena: Allocator, events: *std.ArrayList(Record), history: *History, event: Record) Allocator.Error!void {
    try events.append(arena, event);
    history.messages = events.items;
}

fn writeText(w: *Writer, text: []const u8) Allocator.Error!void {
    w.writeAll(text) catch return error.OutOfMemory;
}

test "ReAct forward retains final history and termination metadata" {
    const testing = std.testing;
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 };
    });
    inline for (.{ false, true }) |forced| {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        const submit = "{\"next_thought\":\"done\",\"tool_calls\":{\"tool_calls\":[{\"name\":\"submit\",\"args\":{\"answer\":\"retained\"}}]}}";
        var script: @import("lm.zig").ScriptedLM = .{ .responses = if (forced)
            &.{ "{\"next_thought\":\"retry\",\"tool_calls\":{\"tool_calls\":[]}}", submit }
        else
            &.{submit} };
        const agent = try ReActIters(Sig, &.{}, 1).init(arena);
        const output = try agent.forward(.{ .arena = arena, .lm = script.lm(), .adapter = .json, .json_fallback = false }, .{});
        const record = try value.toRecord(arena, @TypeOf(output), output);
        try testing.expectEqualStrings("retained", record.get("answer").?.string);
        try testing.expect(record.contains("history"));
        try testing.expectEqualStrings(if (forced) "forced_submit" else "submit", record.get("termination_reason").?.string);
        const event = record.get("history").?.object.get("messages").?.array.items[0].object;
        try testing.expectEqualStrings("retained", event.get("answer").?.string);
    }
}

test "ReAct links final History into the surviving trace step at capacity" {
    const testing = std.testing;
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        \\{"next_thought":"inspect","tool_calls":{"tool_calls":[{"name":"unknown","args":{}}]}}
        ,
        \\{"next_thought":"submit","tool_calls":{"tool_calls":[{"name":"submit","args":{"answer":"done"}}]}}
        ,
    } };
    var trace: predict.Trace = .{ .arena = arena };
    const agent = try ReActIters(Sig, &.{}, 2).init(arena);
    const result = try agent.forward(.{
        .arena = arena,
        .lm = script.lm(),
        .adapter = .json,
        .json_fallback = false,
        .trace = &trace,
        .max_trace_size = 1,
    }, .{});
    try testing.expectEqual(TerminationReason.submit, result.termination_reason);
    try testing.expectEqualStrings("done", result.answer);
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
    const history = trace.steps.items[0].inputs.get("history").?.object;
    const messages = history.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 2), messages.len);
    try testing.expectEqualStrings("done", messages[1].object.get("answer").?.string);
}

test "ReAct submit validates Code models but retains custom string results" {
    const testing = std.testing;
    const Code = @import("code.zig").Code;
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { result: Code, snippets: []const Code };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        "{\"next_thought\":\"Submit code.\",\"tool_calls\":{\"tool_calls\":[{\"name\":\"submit\",\"args\":" ++
            "{\"result\":{\"code\":\"```python\\nrun()\\n```\"},\"snippets\":[{\"code\":\"```x```\"}]}}]}}",
    } };
    const agent = try ReActIters(Sig, &.{}, 1).init(arena);
    const inputs: Record = .empty;
    const prediction = try agent.run(.{ .arena = arena, .lm = script.lm(), .adapter = .json, .json_fallback = false }, &inputs);
    const out = try value.fromRecord(arena, Sig.Output, &prediction.outputs);
    try testing.expectEqualStrings("run()", out.result.code);
    try testing.expectEqualStrings("x", out.snippets[0].code);
    try testing.expectEqual(TerminationReason.submit, prediction.termination_reason);
    const invalid = try std.json.parseFromSliceLeaky(Value, arena, "{\"result\":{\"code\":7},\"snippets\":[]}", .{});
    const submit: Tool = comptime .{
        .name = "submit",
        .args = &.{ Arg.typed("result", Code), Arg.typed("snippets", []const Code) },
        .func = struct {
            fn call(_: Context, args: *const Record) ToolError!Value {
                return args.get("result") orelse error.InvalidArguments;
            }
        }.call,
    };
    try testing.expectError(error.InvalidArguments, normalizeArgs(submit, arena, &invalid.object));
}

test "ReAct typed tools and submit preserve exact wide integers" {
    const testing = std.testing;
    const Payload = struct { id: u128 };
    const Callback = struct {
        fn typed(ctx: Context, args: *const Record) ToolError!Value {
            const Args = struct { count: u64, payload: Payload, ratio: f64 };
            const decoded = value.fromRecord(ctx.arena, Args, args) catch return error.InvalidArguments;
            if (decoded.count != std.math.maxInt(u64) or decoded.payload.id != std.math.maxInt(u128) or
                decoded.ratio != 1e19) return error.InvalidArguments;
            return value.toValue(ctx.arena, u64, decoded.count);
        }

        fn basic(_: Context, args: *const Record) ToolError!Value {
            const count = args.get("count") orelse return error.InvalidArguments;
            const ratio = args.get("ratio") orelse return error.InvalidArguments;
            if (count != .number_string or !std.mem.eql(u8, count.number_string, "18446744073709551615") or
                ratio != .float or ratio.float != 1e19) return error.InvalidArguments;
            return count;
        }
    };
    const typed: Tool = comptime .{
        .name = "typed",
        .args = &.{ Arg.typed("count", u64), Arg.typed("payload", Payload), Arg.typed("ratio", f64) },
        .func = Callback.typed,
    };
    const basic: Tool = comptime .{
        .name = "basic",
        .args = &.{ .{ .name = "count", .schema_type = "integer" }, .{ .name = "ratio", .schema_type = "number" } },
        .func = Callback.basic,
    };
    const Sig = signature.Signature(struct {
        pub const Input = struct { seed: u64 };
        pub const Output = struct { answer: u64, payload: Payload, numbers: []const u64, ratio: f64 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        "{\"next_thought\":\"Check values.\",\"tool_calls\":{\"tool_calls\":[" ++
            "{\"name\":\"typed\",\"args\":{\"count\":18446744073709551615,\"payload\":{\"id\":340282366920938463463374607431768211455},\"ratio\":10000000000000000000}}," ++
            "{\"name\":\"basic\",\"args\":{\"count\":18446744073709551615,\"ratio\":10000000000000000000}}]}}",
        "{\"next_thought\":\"Submit.\",\"tool_calls\":{\"tool_calls\":[" ++
            "{\"name\":\"submit\",\"args\":{\"answer\":18446744073709551615,\"payload\":{\"id\":340282366920938463463374607431768211455}," ++
            "\"numbers\":[10000000000000000000,18446744073709551615],\"ratio\":10000000000000000000}}]}}",
    } };
    const agent = try ReActIters(Sig, &.{ typed, basic }, 2).init(arena);
    const inputs = try value.toRecord(arena, Sig.Input, .{ .seed = std.math.maxInt(u64) });
    const prediction = try agent.run(.{ .arena = arena, .lm = script.lm(), .adapter = .json, .json_fallback = false }, &inputs);
    const out = try value.fromRecord(arena, Sig.Output, &prediction.outputs);
    try testing.expectEqual(std.math.maxInt(u64), out.answer);
    try testing.expectEqual(std.math.maxInt(u128), out.payload.id);
    try testing.expectEqualSlices(u64, &.{ 10_000_000_000_000_000_000, std.math.maxInt(u64) }, out.numbers);
    try testing.expectEqual(@as(f64, 1e19), out.ratio);
    try testing.expectEqual(TerminationReason.submit, prediction.termination_reason);
    try testing.expectEqual(@as(usize, 2), prediction.history.messages.len);
    const calls = try ToolCalls.fromValue(arena, prediction.history.messages[0].get("tool_calls").?);
    const results = calls.tool_call_results.?.tool_call_results;
    try testing.expectEqual(@as(usize, 2), results.len);
    for (results) |result| {
        try testing.expect(!result.is_error);
        try testing.expectEqualStrings("18446744073709551615", result.value.number_string);
    }
    var invalid: Record = .empty;
    try invalid.put(arena, "count", .{ .number_string = "18446744073709551616" });
    try invalid.put(arena, "payload", try value.toValue(arena, Payload, .{ .id = std.math.maxInt(u128) }));
    try invalid.put(arena, "ratio", .{ .number_string = "10000000000000000000" });
    try testing.expectError(error.InvalidArguments, normalizeArgs(typed, arena, &invalid));
}

fn parseDefault(arena: Allocator, text: []const u8) adapter.Error!Value {
    if (std.json.parseFromSliceLeaky(Value, arena, text, .{})) |parsed| return parsed else |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
    }
    if (std.mem.eql(u8, text, "None")) return .null;
    if (std.mem.eql(u8, text, "True")) return .{ .bool = true };
    if (std.mem.eql(u8, text, "False")) return .{ .bool = false };
    if (text.len < 2 or (text[0] != '\'' and text[0] != '"') or text[0] != text[text.len - 1]) return error.AdapterParse;
    var out: Writer.Allocating = .init(arena);
    var index: usize = 1;
    while (index < text.len - 1) : (index += 1) {
        const c = text[index];
        if (c != '\\') {
            out.writer.writeByte(c) catch return error.OutOfMemory;
            continue;
        }
        index += 1;
        if (index >= text.len - 1) return error.AdapterParse;
        const escaped = text[index];
        switch (escaped) {
            '\\', '\'', '"' => out.writer.writeByte(escaped) catch return error.OutOfMemory,
            'n' => try writeText(&out.writer, "\n"),
            'r' => try writeText(&out.writer, "\r"),
            't' => try writeText(&out.writer, "\t"),
            'b' => try writeText(&out.writer, "\x08"),
            'f' => try writeText(&out.writer, "\x0c"),
            'v' => try writeText(&out.writer, "\x0b"),
            'a' => try writeText(&out.writer, "\x07"),
            '\n' => {},
            'x', 'u', 'U' => {
                const digits: usize = switch (escaped) {
                    'x' => 2,
                    'u' => 4,
                    else => 8,
                };
                if (text.len - 1 - index <= digits) return error.AdapterParse;
                const codepoint = std.fmt.parseInt(u21, text[index + 1 ..][0..digits], 16) catch return error.AdapterParse;
                var buffer: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(codepoint, &buffer) catch return error.AdapterParse;
                try writeText(&out.writer, buffer[0..len]);
                index += digits;
            },
            '0'...'7' => {
                var codepoint: u21 = escaped - '0';
                var count: usize = 1;
                while (count < 3 and index + 1 < text.len - 1 and text[index + 1] >= '0' and text[index + 1] <= '7') : (count += 1) {
                    index += 1;
                    codepoint = codepoint * 8 + text[index] - '0';
                }
                var buffer: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(codepoint, &buffer) catch return error.AdapterParse;
                try writeText(&out.writer, buffer[0..len]);
            },
            else => try writeText(&out.writer, &.{ '\\', escaped }),
        }
    }
    return .{ .string = out.written() };
}
