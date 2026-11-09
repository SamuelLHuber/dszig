//! `dspy.ReAct` (DSPy 3.4.1): a tool-using agent over any signature. Each
//! step `react` picks `next_thought`, `next_tool_name` and `next_tool_args`
//! given the trajectory so far; the tool's result is appended as an
//! observation until `finish` or `max_iters`. `extract`, a ChainOfThought
//! over the inputs and the trajectory, then produces the outputs.
//!
//! Tools are comptime: their names form the `next_tool_name` Literal and
//! their descriptions the instructions, exactly as DSPy prints `str(Tool)`.
//! Not supported: trajectory truncation on context-window errors, async
//! tools, Pydantic-typed tool arguments, and Python's traceback text in
//! `Execution error in <tool>: ...` observations (DSZig gives the error name).

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const adapter = @import("adapter.zig");
const predict = @import("predict.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Field = signature.Field;
const Record = value.Record;
const Value = std.json.Value;

pub const ToolError = error{ OutOfMemory, ToolFailed, InvalidArguments };

pub const Arg = struct {
    name: []const u8,
    /// JSON-schema `type` (`string`, `integer`, `number`, `boolean`, ...).
    schema_type: []const u8,
    /// Python repr of the default value, e.g. `2` or `'x'`.
    default: ?[]const u8 = null,
    description: ?[]const u8 = null,
};

pub const Tool = struct {
    name: []const u8,
    /// Empty means no description, like a function without a docstring.
    desc: []const u8 = "",
    args: []const Arg = &.{},
    func: *const fn (ctx: Context, args: *const Record) ToolError!Value,

    /// `str(dspy.Tool)`.
    pub fn describe(comptime tool: Tool) []const u8 {
        comptime {
            const desc = if (tool.desc.len > 0) ", whose description is <desc>" ++ tool.desc ++ "</desc>." else ".";
            return tool.name ++ replaceNewlines(desc) ++ " It takes arguments " ++ argsRepr(tool.args) ++ ".";
        }
    }
};

fn replaceNewlines(comptime s: []const u8) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (s) |c| out = out ++ if (c == '\n') "  " else &[_]u8{c};
        return out;
    }
}

/// Python `repr(dict)` of the tool's JSON schemas,
/// e.g. `{'a': {'type': 'integer'}, 'b': {'type': 'integer', 'default': 2}}`.
fn argsRepr(comptime args: []const Arg) []const u8 {
    comptime {
        var out: []const u8 = "{";
        for (args, 0..) |arg, i| {
            if (i > 0) out = out ++ ", ";
            out = out ++ pyRepr(arg.name) ++ ": {'type': " ++ pyRepr(arg.schema_type);
            if (arg.default) |d| out = out ++ ", 'default': " ++ d;
            if (arg.description) |d| out = out ++ ", 'description': " ++ pyRepr(d);
            out = out ++ "}";
        }
        return out ++ "}";
    }
}

/// Python `repr(str)` for printable text.
fn pyRepr(comptime s: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(100_000);
        const quote: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and
            std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
        var out: []const u8 = &[_]u8{quote};
        for (s) |c| {
            out = out ++ switch (c) {
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                else => blk: {
                    if (c < 0x20 or c == 0x7f) @compileError("dszig: tool text has a control character");
                    if (c == quote) break :blk &[_]u8{ '\\', c };
                    break :blk &[_]u8{c};
                },
            };
        }
        return out ++ &[_]u8{quote};
    }
}

fn backticked(comptime fields: []const Field) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (fields, 0..) |f, i| {
            if (i > 0) out = out ++ ", ";
            out = out ++ "`" ++ f.name ++ "`";
        }
        return out;
    }
}

fn makeField(comptime name: []const u8, comptime kind: signature.Kind, comptime type_name: []const u8, comptime literals: []const []const u8) Field {
    return .{
        .name = name,
        .kind = kind,
        .type_name = type_name,
        .desc = "${" ++ name ++ "}",
        .prefix = signature.inferPrefix(name) ++ ":",
        .literals = literals,
    };
}

const trajectory_field = makeField("trajectory", .str, "str", &.{});

fn finishResult(_: Context, _: *const Record) ToolError!Value {
    return .{ .string = "Completed." };
}

pub fn ReAct(comptime Sig: type, comptime user_tools: []const Tool) type {
    return ReActIters(Sig, user_tools, 20);
}

/// `ReAct` whose `max_iters` defaults to `default_max_iters`.
pub fn ReActIters(comptime Sig: type, comptime user_tools: []const Tool, comptime default_max_iters: usize) type {
    comptime assert(signature.isSignature(Sig));
    const n_in = Sig.input_fields_list.len;
    const output_names = backticked(Sig.fields[n_in..]);
    const tools = comptime blk: {
        for (user_tools, 0..) |t, i| {
            if (std.mem.eql(u8, t.name, "finish")) @compileError("dszig: ReAct reserves the tool name `finish`");
            for (user_tools[0..i]) |u| {
                if (std.mem.eql(u8, t.name, u.name)) @compileError("dszig: duplicate ReAct tool `" ++ t.name ++ "`");
            }
        }
        break :blk user_tools ++ [_]Tool{.{
            .name = "finish",
            .desc = "Marks the task as complete. That is, signals that all information for producing the outputs, i.e. " ++
                output_names ++ ", are now available to be extracted.",
            .func = finishResult,
        }};
    };
    const tool_names = comptime blk: {
        var names: [tools.len][]const u8 = undefined;
        for (tools, 0..) |t, i| names[i] = t.name;
        const final = names;
        break :blk &final;
    };
    const literal_type = comptime blk: {
        var s: []const u8 = "Literal[";
        for (tool_names, 0..) |name, i| s = s ++ (if (i > 0) ", " else "") ++ pyRepr(name);
        break :blk s ++ "]";
    };
    const instructions = comptime blk: {
        var s: []const u8 = Sig.instructions ++ "\n\n" ++
            "You are an Agent. In each episode, you will be given the fields " ++ backticked(Sig.fields[0..n_in]) ++
            " as input. And you can see your past trajectory so far.\n" ++
            "Your goal is to use one or more of the supplied tools to collect any necessary information for producing " ++
            output_names ++ ".\n\n" ++
            "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.\n" ++
            "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n\n" ++
            "When writing next_thought, you may reason about the current situation and plan for future steps.\n" ++
            "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n";
        for (tools, 0..) |t, i| s = s ++ std.fmt.comptimePrint("\n({d}) ", .{i + 1}) ++ t.describe();
        break :blk s ++ "\nWhen providing `next_tool_args`, the value inside the field must be in JSON format";
    };
    const CoT = Sig.PrependOutput("reasoning", []const u8);

    return struct {
        const Self = @This();
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        pub const all_tools: []const Tool = tools;
        pub const react_instructions = instructions;
        pub const react_fields: []const Field = Sig.fields[0..n_in] ++ [_]Field{
            trajectory_field,
            makeField("next_thought", .str, "str", &.{}),
            makeField("next_tool_name", .literal, literal_type, tool_names),
            makeField("next_tool_args", .json_object, "dict[str, Any]", &.{}),
        };
        pub const extract_fields: []const Field = CoT.fields[0..n_in] ++ [_]Field{trajectory_field} ++ CoT.fields[n_in..];

        react: predict.PredictFields(react_fields, n_in + 1, instructions) = .{},
        extract: Extract = .{},
        max_iters: usize = default_max_iters,
        compiled: bool = false,

        /// `dspy.ChainOfThought(fallback_signature)`; its predictor is
        /// `extract.predict`, as in DSPy.
        pub const Extract = struct {
            predict: predict.PredictFields(extract_fields, n_in + 1, Sig.instructions) = .{},
            compiled: bool = false,

            pub fn forward(self: *const Extract, ctx: Context, inputs: *const Record) predict.Error!Record {
                return self.predict.state.call(ctx, inputs);
            }
        };

        /// `dspy.Prediction(trajectory=..., **extract)`.
        pub const Prediction = struct {
            /// `thought_i`, `tool_name_i`, `tool_args_i`, `observation_i`.
            trajectory: Record,
            /// `reasoning` followed by the signature's outputs.
            outputs: Record,
        };

        pub fn forward(self: *const Self, ctx: Context, input: Input) predict.Error!Output {
            const inputs = try value.toRecord(ctx.arena, Input, input);
            const prediction = try self.run(ctx, &inputs);
            return value.fromRecord(ctx.arena, Output, &prediction.outputs);
        }

        pub fn run(self: *const Self, ctx: Context, inputs: *const Record) predict.Error!Prediction {
            var trajectory: Record = .empty;
            for (0..self.max_iters) |idx| {
                const step_inputs = try withTrajectory(ctx.arena, inputs, &trajectory);
                const pred = try self.react.state.call(ctx, &step_inputs);
                const name = pred.get("next_tool_name").?.string;
                const args = pred.get("next_tool_args").?;
                assert(args == .object);
                try trajectory.put(ctx.arena, try key(ctx.arena, "thought", idx), pred.get("next_thought").?);
                try trajectory.put(ctx.arena, try key(ctx.arena, "tool_name", idx), .{ .string = name });
                try trajectory.put(ctx.arena, try key(ctx.arena, "tool_args", idx), args);
                try trajectory.put(ctx.arena, try key(ctx.arena, "observation", idx), try observe(ctx, name, &args.object));
                if (std.mem.eql(u8, name, "finish")) break;
            }
            const final_inputs = try withTrajectory(ctx.arena, inputs, &trajectory);
            const outputs = try self.extract.predict.state.call(ctx, &final_inputs);
            return .{ .trajectory = trajectory, .outputs = outputs };
        }

        fn observe(ctx: Context, name: []const u8, args: *const Record) Allocator.Error!Value {
            inline for (tools) |tool| {
                if (std.mem.eql(u8, tool.name, name)) {
                    const result = if (validArgs(tool, args)) tool.func(ctx, args) else error.InvalidArguments;
                    return result catch |err| switch (err) {
                        error.OutOfMemory => error.OutOfMemory,
                        else => .{ .string = try std.fmt.allocPrint(ctx.arena, "Execution error in {s}: {s}", .{ name, @errorName(err) }) },
                    };
                }
            }
            unreachable; // `next_tool_name` parses only to a tool name.
        }
    };
}

/// `Tool._validate_and_parse_args`: every argument must be declared and of
/// its schema's basic type.
fn validArgs(comptime tool: Tool, args: *const Record) bool {
    var it = args.iterator();
    next: while (it.next()) |entry| {
        inline for (tool.args) |arg| {
            if (std.mem.eql(u8, arg.name, entry.key_ptr.*)) {
                if (!schemaAccepts(arg.schema_type, entry.value_ptr.*)) return false;
                continue :next;
            }
        }
        return false;
    }
    return true;
}

fn schemaAccepts(comptime schema_type: []const u8, v: Value) bool {
    if (comptime std.mem.eql(u8, schema_type, "string")) return v == .string;
    if (comptime std.mem.eql(u8, schema_type, "integer")) return v == .integer;
    if (comptime std.mem.eql(u8, schema_type, "number")) return v == .integer or v == .float;
    if (comptime std.mem.eql(u8, schema_type, "boolean")) return v == .bool;
    if (comptime std.mem.eql(u8, schema_type, "array")) return v == .array;
    if (comptime std.mem.eql(u8, schema_type, "object")) return v == .object;
    return true;
}

fn key(arena: Allocator, comptime kind: []const u8, idx: usize) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(arena, kind ++ "_{d}", .{idx});
}

/// The inputs plus `trajectory`, rendered like DSPy's `_format_trajectory`.
fn withTrajectory(arena: Allocator, inputs: *const Record, trajectory: *const Record) predict.Error!Record {
    const fields = try arena.alloc(Field, trajectory.count());
    for (fields, trajectory.keys()) |*f, k| f.* = .{ .name = k, .kind = .str, .type_name = "str", .desc = "", .prefix = "" };
    var out = try inputs.clone(arena);
    try out.put(arena, "trajectory", .{ .string = try adapter.formatInputs(arena, fields, trajectory) });
    return out;
}
