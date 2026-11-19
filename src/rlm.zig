//! Synchronous recursive-LM orchestration for a caller-supplied runtime.
//! The runtime is trusted host code; this module never evaluates model text itself.
const std = @import("std");
const adapter = @import("adapter.zig");
const history_mod = @import("history.zig");
const predict = @import("predict.zig");
const lm = @import("lm.zig");
const value = @import("value.zig");
const runtime_value = @import("runtime_value.zig");
const pyjson = @import("pyjson.zig");
const Field = @import("signature.zig").Field;
const Allocator = std.mem.Allocator;
const Record = value.Record;
const Value = runtime_value.Value;
const History = history_mod.History;

pub const Runtime = struct {
    ptr: *anyopaque,
    /// Execution is borrowed for this serial callback only.
    execute: *const fn (*anyopaque, []const u8, *Execution) anyerror!Result,
    close: *const fn (*anyopaque) void,

    pub const Execution = struct {
        variables: *const Record,
        model: lm.LM,
        ctx: *const predict.Context,
        calls_remaining: u32,

        /// Dispatch one raw text prompt. A dispatched failure still consumes a call.
        /// The returned text is owned by the invocation arena.
        pub fn query(self: *Execution, prompt: []const u8) ![]const u8 {
            if (prompt.len == 0) return error.EmptySubModelPrompt;
            if (self.calls_remaining == 0) return error.SubModelCallLimit;
            self.calls_remaining -= 1;
            return self.model.complete(self.ctx.arena, .{
                .messages = &.{.{ .role = .user, .content = .{ .text = prompt } }},
                .usage_tracker = self.ctx.usage_tracker,
                .usage_failure = self.ctx.usage_failure,
            });
        }
    };

    /// Result data must survive until the next execute or close, whichever is first.
    pub const Result = union(enum) {
        observation: []const u8,
        execution_error: []const u8,
        submit: Value,
    };
};

pub const Factory = struct {
    ptr: *anyopaque,
    /// Open must clean up internally if it fails. Every success receives one close.
    open: *const fn (*anyopaque) anyerror!Runtime,
};

pub const Entry = struct {
    reasoning: []const u8,
    code: []const u8,
    output: []const u8,
    max_output_chars: i64,

    fn tagged(entry: Entry, arena: Allocator) !Value {
        var record: Record = .empty;
        try record.put(arena, "__dspy_repl__", text("entry"));
        try record.put(arena, "reasoning", text(entry.reasoning));
        try record.put(arena, "code", text(entry.code));
        try record.put(arena, "output", text(entry.output));
        try record.put(arena, "max_output_chars", .{ .integer = entry.max_output_chars });
        return .{ .object = record };
    }
};

pub const HistoryProcessor = struct {
    ptr: *anyopaque,
    /// Draft records are private to this invocation and must not escape the callback.
    /// Replacement records must remain readable until copied just after return.
    process: *const fn (*anyopaque, Allocator, []Record) anyerror!Edit,

    pub const Edit = union(enum) {
        keep,
        truncate: usize,
        replace: History,
    };
};

pub const ForwardOptions = struct {
    history: History = .{},
    history_processor: ?HistoryProcessor = null,
};

const TurnHistory = struct {
    storage: []Record,
    len: usize,

    fn clone(arena: Allocator, source: History, spare: usize) !TurnHistory {
        const count = try std.math.add(usize, source.messages.len, spare);
        const storage = try arena.alloc(Record, count);
        for (source.messages, storage[0..source.messages.len]) |*event, *copy| {
            copy.* = try value.cloneRecord(arena, event);
        }
        return .{ .storage = storage, .len = source.messages.len };
    }

    fn view(self: TurnHistory) History {
        return .{ .messages = self.storage[0..self.len] };
    }

    fn append(self: *TurnHistory, event: Record) void {
        std.debug.assert(self.len < self.storage.len);
        self.storage[self.len] = event;
        self.len += 1;
    }

    fn apply(self: *TurnHistory, arena: Allocator, processor: HistoryProcessor) !void {
        // Allow this turn's event and a possible extraction event after it.
        const spare = 2;
        var candidate = try clone(arena, self.view(), spare);
        const edit = processor.process(processor.ptr, arena, candidate.storage[0..candidate.len]) catch |err| switch (err) {
            error.OutOfMemory, error.Canceled => return err,
            else => {
                std.log.scoped(.dszig_rlm).warn("History processor rolled back: {s}", .{@errorName(err)});
                return;
            },
        };
        switch (edit) {
            .keep => {},
            .truncate => |len| {
                if (len > candidate.len) return error.InvalidHistoryLength;
                candidate.len = len;
            },
            .replace => |replacement| candidate = try clone(arena, replacement, spare),
        }
        self.* = candidate;
    }
};

pub const Options = struct {
    /// Full action objective for the supplied runtime and its capabilities.
    action_instructions: []const u8,
    max_iters: u32 = 20,
    max_llm_calls: u32 = 50,
    sub_lm: ?lm.LM = null,
    max_output_chars: i64 = 10_000,
    history_processor: ?HistoryProcessor = null,
};

/// Bounds history preallocation and the number of model-driven executions.
pub const iterations_max: u32 = 1024;

const variable_field: Field = stringField("variables_info", "Metadata about the variables available in the REPL");
const action_fields = [_]Field{
    .{ .name = "history", .kind = .json, .type_name = "History", .desc = "", .prefix = "History:", .schema = History.schema, .validate = History.validate, .dspy_type = true },
    variable_field,
    stringField("iteration", "Current iteration number (1-indexed) out of max_iters"),
    .{ .name = "reasoning", .kind = .str, .type_name = "str", .desc = "A short plan for the next action.", .prefix = "Reasoning:", .reasoning_type = true, .dspy_type = true, .native_reasoning = true, .reasoning_fallback = "plan" },
    stringField("code", "Python code to execute. Use markdown code block format: ```python\\n<code>\\n```"),
};

pub fn CustomRLM(comptime Sig: type) type {
    const extract_fields = [_]Field{
        variable_field,
        .{ .name = "repl_history", .kind = .str, .type_name = "REPLHistory", .desc = "Your REPL interactions so far", .prefix = "Repl History:" },
    } ++ Sig.output_fields_list[0..Sig.output_fields_list.len].*;
    const extract_instructions = (if (Sig.instructions.len == 0) "" else "The trajectory was generated with the following objective: \n" ++ Sig.instructions ++ "\n\n\n") ++
        "Based on the REPL trajectory, extract the final outputs now.\n\n" ++
        "            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs.";
    return struct {
        const Self = @This();
        pub const Input = Sig.Input;
        pub const Output = struct {
            outputs: Sig.Output,
            history: History,
            final_reasoning: []const u8,
            trajectory: []const Entry,
        };

        generate_action: predict.PredictFields(&action_fields, 3, ""),
        extract: predict.PredictFields(&extract_fields, 2, extract_instructions) = .{},
        factory: Factory,
        max_iters: u32,
        max_llm_calls: u32,
        sub_lm: ?lm.LM,
        max_output_chars: i64,
        history_processor: ?HistoryProcessor,
        compiled: bool = false,

        pub fn init(arena: Allocator, factory: Factory, options: Options) !Self {
            if (options.max_iters > iterations_max) return error.TooManyIterations;
            inline for (Sig.fields) |field| {
                for ([_][]const u8{ "history", "variables_info", "iteration", "repl_entry", "repl_history" }) |name| {
                    if (std.mem.eql(u8, field.name, name)) return error.ReservedRLMField;
                }
            }
            var self: Self = .{
                .generate_action = .{},
                .factory = factory,
                .max_iters = options.max_iters,
                .max_llm_calls = options.max_llm_calls,
                .sub_lm = options.sub_lm,
                .max_output_chars = options.max_output_chars,
                .history_processor = options.history_processor,
            };
            try self.generate_action.state.forkSignatureClass(arena);
            try self.extract.state.forkSignatureClass(arena);
            self.generate_action.state.instructions = try arena.dupe(u8, options.action_instructions);
            return self;
        }

        pub fn forward(self: *const Self, ctx: predict.Context, input: Input) anyerror!Output {
            return self.forwardWithHistory(ctx, input, .{});
        }

        pub fn forwardWithHistory(self: *const Self, ctx: predict.Context, input: Input, prior: History) anyerror!Output {
            return self.forwardWithOptions(ctx, input, .{ .history = prior });
        }

        pub fn forwardWithOptions(
            self: *const Self,
            ctx: predict.Context,
            input: Input,
            options: ForwardOptions,
        ) anyerror!Output {
            const arena = ctx.arena;
            if (self.max_iters > iterations_max) return error.TooManyIterations;
            const variables = try value.toRecord(arena, Input, input);
            const info = try variableInfo(arena, Sig.input_fields_list, &variables);
            const entries = try arena.alloc(Entry, self.max_iters);
            var history = try TurnHistory.clone(arena, options.history, entries.len + 1);
            const processor = options.history_processor orelse self.history_processor;
            var execution: Runtime.Execution = .{
                .variables = &variables,
                .model = self.sub_lm orelse ctx.lm,
                .ctx = &ctx,
                .calls_remaining = self.max_llm_calls,
            };
            const runtime = try self.factory.open(self.factory.ptr);
            defer runtime.close(runtime.ptr);

            for (entries, 0..) |*entry, index| {
                if (processor) |hook| try history.apply(arena, hook);
                const iteration = text(try std.fmt.allocPrint(arena, "{d}/{d}", .{ index + 1, self.max_iters }));
                var action_inputs: Record = .empty;
                try action_inputs.put(arena, "history", try history.view().toValue(arena));
                try action_inputs.put(arena, "variables_info", info);
                try action_inputs.put(arena, "iteration", iteration);
                const action = try self.generate_action.state.call(ctx, &action_inputs);
                const reason = action.get("reasoning") orelse .null;
                entry.* = .{
                    .reasoning = try reasoningText(reason),
                    .code = (action.get("code") orelse return error.MissingField).string,
                    .output = undefined,
                    .max_output_chars = self.max_output_chars,
                };
                const result = try executeAction(arena, runtime, &execution, entry);
                const final = try processResult(arena, Sig.output_fields_list, result, entry);
                var event = if (index == 0) try variables.clone(arena) else Record.empty;
                try event.put(arena, "variables_info", info);
                try event.put(arena, "iteration", iteration);
                try event.put(arena, "repl_entry", try entry.tagged(arena));
                if (final) |outputs| {
                    try appendOutputs(arena, &event, &outputs);
                    history.append(event);
                    return .{
                        .outputs = try value.fromRecord(arena, Sig.Output, &outputs),
                        .history = history.view(),
                        .final_reasoning = entry.reasoning,
                        .trajectory = entries[0 .. index + 1],
                    };
                }
                history.append(event);
            }
            return self.extractFinal(ctx, info, &history, entries);
        }

        fn extractFinal(self: *const Self, ctx: predict.Context, info: Value, history: *TurnHistory, entries: []const Entry) !Output {
            const arena = ctx.arena;
            var inputs: Record = .empty;
            try inputs.put(arena, "variables_info", info);
            try inputs.put(arena, "repl_history", text(try formatTrajectory(arena, entries)));
            const outputs = try self.extract.state.call(ctx, &inputs);
            var tag: Record = .empty;
            try tag.put(arena, "__dspy_repl__", text("extract_fallback"));
            var event: Record = .empty;
            try event.put(arena, "repl_entry", .{ .object = tag });
            try appendOutputs(arena, &event, &outputs);
            history.append(event);
            return .{
                .outputs = try value.fromRecord(arena, Sig.Output, &outputs),
                .history = history.view(),
                .final_reasoning = "Extract forced final output",
                .trajectory = entries,
            };
        }
    };
}

fn stringField(name: []const u8, desc: []const u8) Field {
    return .{ .name = name, .kind = .str, .type_name = "str", .desc = desc, .prefix = "" };
}

fn text(bytes: []const u8) Value {
    return .{ .string = bytes };
}

fn appendOutputs(arena: Allocator, event: *Record, outputs: *const Record) !void {
    for (outputs.keys(), outputs.values()) |name, field| try event.put(arena, name, field);
}

fn reasoningText(reason: Value) ![]const u8 {
    if (reason == .null) return "";
    return (try @import("signature.zig").Reasoning.fromValue(reason)).content;
}

fn variableInfo(arena: Allocator, fields: []const Field, variables: *const Record) !Value {
    var items = runtime_value.Array.init(arena);
    try items.ensureTotalCapacity(fields.len);
    for (fields) |field| {
        const val = variables.get(field.name) orelse return error.MissingField;
        const rendered = try variableText(arena, val);
        const length = try std.unicode.utf8CountCodepoints(rendered);
        const preview = try variablePreview(arena, rendered, length);
        const description = if (field.desc.len == 0 or std.mem.startsWith(u8, field.desc, "${"))
            ""
        else
            try std.fmt.allocPrint(arena, "Description: {s}\n", .{field.desc});
        items.appendAssumeCapacity(text(try std.fmt.allocPrint(
            arena,
            "Variable: `{s}` (access it in your code)\nType: {s}\n{s}Total length: {d} characters\nPreview:\n```\n{s}\n```",
            .{ field.name, field.type_name, description, length, preview },
        )));
    }
    return .{ .array = items };
}

fn variableText(arena: Allocator, input: Value) ![]const u8 {
    const serialized = try value.serializedValue(arena, input);
    if (serialized == .string) return serialized.string;
    var buffer: std.Io.Writer.Allocating = .init(arena);
    switch (serialized) {
        .object, .array => try pyjson.writeJsonAsciiIndented(&buffer.writer, serialized),
        else => try pyjson.writeStr(&buffer.writer, serialized),
    }
    return buffer.written();
}

fn variablePreview(arena: Allocator, rendered: []const u8, length: usize) ![]const u8 {
    const half = 500;
    if (length <= 2 * half) return rendered;
    var chars = (try std.unicode.Utf8View.init(rendered)).iterator();
    for (0..half) |_| _ = chars.nextCodepointSlice().?;
    const head_end = chars.i;
    for (half..length - half) |_| _ = chars.nextCodepointSlice().?;
    return std.mem.concat(arena, u8, &.{ rendered[0..head_end], "...", rendered[chars.i..] });
}

fn executeAction(
    arena: Allocator,
    runtime: Runtime,
    execution: *Runtime.Execution,
    entry: *Entry,
) !Runtime.Result {
    const whitespace = " \t\n\r\x0b\x0c";
    const trimmed = std.mem.trim(u8, entry.code, whitespace);
    entry.code = trimmed;
    if (std.mem.indexOf(u8, trimmed, "```")) |first| {
        const fence = trimmed[first + 3 ..];
        if (std.mem.indexOfScalar(u8, fence, '\n')) |line_end| {
            var words = std.mem.tokenizeAny(u8, fence[0..line_end], whitespace);
            if (words.next()) |language| {
                if (!std.ascii.eqlIgnoreCase(language, "python") and !std.ascii.eqlIgnoreCase(language, "py")) {
                    return .{ .execution_error = try std.fmt.allocPrint(
                        arena,
                        "Expected Python code but got ```{s} fence. Write Python code, not {s}.",
                        .{ language, language },
                    ) };
                }
            }
            const body = fence[line_end + 1 ..];
            entry.code = std.mem.trim(u8, body[0 .. std.mem.indexOf(u8, body, "```") orelse body.len], whitespace);
        }
    }
    return runtime.execute(runtime.ptr, entry.code, execution);
}

fn processResult(arena: Allocator, fields: []const Field, result: Runtime.Result, entry: *Entry) !?Record {
    switch (result) {
        .observation => |observation| entry.output = if (observation.len == 0)
            "(no output - did you forget to print?)"
        else
            try arena.dupe(u8, observation),
        .execution_error => |detail| entry.output = try std.fmt.allocPrint(arena, "[Error] {s}", .{detail}),
        .submit => |submitted| {
            const outputs = try parseSubmit(arena, fields, submitted, entry) orelse return null;
            var buffer: std.Io.Writer.Allocating = .init(arena);
            try buffer.writer.writeAll("FINAL: ");
            try pyjson.writeRepr(&buffer.writer, .{ .object = outputs }, 0);
            entry.output = buffer.written();
            return outputs;
        },
    }
    return null;
}

fn parseSubmit(arena: Allocator, fields: []const Field, submitted: Value, entry: *Entry) !?Record {
    if (submitted != .object) {
        entry.output = "[Error] FINAL must return a dictionary";
        return null;
    }
    var missing = runtime_value.Array.init(arena);
    var names: std.ArrayList([]const u8) = .empty;
    for (fields) |field| {
        try names.append(arena, field.name);
        if (!submitted.object.contains(field.name)) try missing.append(text(field.name));
    }
    if (missing.items.len != 0) {
        std.mem.sort(Value, missing.items, {}, struct {
            fn less(_: void, a: Value, b: Value) bool {
                return std.mem.order(u8, a.string, b.string) == .lt;
            }
        }.less);
        var buffer: std.Io.Writer.Allocating = .init(arena);
        try buffer.writer.writeAll("[Error] Missing output fields: ");
        try pyjson.writeRepr(&buffer.writer, .{ .array = missing }, 0);
        try buffer.writer.print(". Use SUBMIT({s})", .{try std.mem.join(arena, ", ", names.items)});
        entry.output = buffer.written();
        return null;
    }
    var outputs: Record = .empty;
    for (fields) |field| {
        const parsed = adapter.parseFieldValue(arena, field, submitted.object.get(field.name).?) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            entry.output = try std.fmt.allocPrint(arena, "[Type Error] {s}: expected {s}", .{ field.name, field.type_name });
            return null;
        };
        try outputs.put(arena, field.name, try value.cloneValue(arena, parsed));
    }
    return outputs;
}

fn formatTrajectory(arena: Allocator, entries: []const Entry) ![]const u8 {
    if (entries.len == 0) return "You have not interacted with the REPL environment yet.";
    var buffer: std.Io.Writer.Allocating = .init(arena);
    for (entries, 1..) |entry, step| {
        if (step != 1) try buffer.writer.writeByte('\n');
        try buffer.writer.print("=== Step {d} ===\n", .{step});
        if (entry.reasoning.len != 0) try buffer.writer.print("Reasoning: {s}\n", .{entry.reasoning});
        try buffer.writer.print("Code:\n```python\n{s}\n```\n{s}", .{
            entry.code,
            try history_mod.truncateReplOutput(arena, entry.output, entry.max_output_chars),
        });
    }
    return history_mod.escapeReplText(arena, buffer.written(), false);
}
