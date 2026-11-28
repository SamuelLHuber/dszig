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
const signature = @import("signature.zig");
const types = @import("types.zig");
const unicode = @import("unicode.zig");
const whitespace = @import("xml_unicode.zig");
const Field = signature.Field;
const Allocator = std.mem.Allocator;
const Record = value.Record;
const Value = runtime_value.Value;
const History = history_mod.History;

pub const Runtime = struct {
    ptr: *anyopaque,
    /// Execution is borrowed for this serial callback only.
    execute: *const fn (*anyopaque, []const u8, *Execution) anyerror!Result,
    close: *const fn (*anyopaque) void,

    pub const BatchConcurrency = struct {
        io: std.Io,
        /// Must support concurrent allocations and retain the returned text until consumed.
        arena: Allocator,
    };

    pub const Execution = struct {
        variables: *const Record,
        model: lm.LM,
        ctx: *const predict.Context,
        calls_remaining: u32,

        /// Dispatch one raw text prompt. A dispatched failure still consumes a call.
        /// The returned text is owned by the invocation arena.
        pub fn query(self: *Execution, prompt: []const u8) ![]const u8 {
            if (prompt.len == 0) return error.EmptySubModelPrompt;
            try self.reserve(1);
            return self.dispatch(self.ctx.arena, .{ .text = prompt }, self.ctx.usage_failure);
        }

        /// Dispatch one ordered list of canonical text/media parts without
        /// rendering markers or allowing the LM to read a local file.
        pub fn queryParts(self: *Execution, parts: []const Value) ![]const u8 {
            try validateParts(parts);
            try self.reserve(1);
            return self.dispatch(self.ctx.arena, .{ .parts = parts }, self.ctx.usage_failure);
        }

        /// Validate the entire mixed batch before charging or dispatching.
        /// As with queryBatch, empty text is allowed, but empty part lists are not.
        pub fn queryContentBatch(
            self: *Execution,
            prompts: []const adapter.Content,
            concurrency: ?BatchConcurrency,
        ) ![]const []const u8 {
            for (prompts) |prompt| switch (prompt) {
                .text => {},
                .parts => |parts| try validateParts(parts),
            };
            return self.batch(adapter.Content, prompts, concurrency);
        }

        fn validateParts(parts: []const Value) error{ EmptySubModelPrompt, InvalidSubModelPart, SubModelHostFile }!void {
            if (parts.len == 0) return error.EmptySubModelPrompt;
            for (parts) |part| {
                if (part != .object) return error.InvalidSubModelPart;
                if ((part.object.get("path") orelse .null) != .null) return error.SubModelHostFile;
                const kind = part.object.get("type") orelse return error.InvalidSubModelPart;
                if (kind != .string) return error.InvalidSubModelPart;
                if (std.mem.eql(u8, kind.string, "text")) {
                    if ((part.object.get("text") orelse Value{ .string = "" }) != .string)
                        return error.InvalidSubModelPart;
                    continue;
                }
                if (!std.mem.eql(u8, kind.string, "image") and
                    !std.mem.eql(u8, kind.string, "audio") and
                    !std.mem.eql(u8, kind.string, "document")) return error.InvalidSubModelPart;
                const mime = part.object.get("media_type") orelse return error.InvalidSubModelPart;
                if (mime != .string or mime.string.len == 0) return error.InvalidSubModelPart;
                var sources: u8 = 0;
                for ([_][]const u8{ "data", "url", "file_id" }) |key| {
                    const source = part.object.get(key) orelse continue;
                    if (source == .null) continue;
                    if (source != .string or source.string.len == 0) return error.InvalidSubModelPart;
                    sources += 1;
                    if (std.mem.eql(u8, key, "data"))
                        @import("file.zig").validateBase64(source.string) catch return error.InvalidSubModelPart;
                }
                if (sources != 1) return error.InvalidSubModelPart;
            }
        }

        fn asContent(prompt: anytype) adapter.Content {
            return if (@TypeOf(prompt) == adapter.Content) prompt else .{ .text = prompt };
        }

        /// Reserve the entire text batch before dispatch; retain input order. Empty
        /// strings are valid here. Only LMFailure becomes an in-band observation.
        /// Concurrency is explicit: the model and its dependencies must be safe
        /// for concurrent calls. All workers finish before this method returns.
        pub fn queryBatch(
            self: *Execution,
            prompts: []const []const u8,
            concurrency: ?BatchConcurrency,
        ) ![]const []const u8 {
            return self.batch([]const u8, prompts, concurrency);
        }

        fn batch(
            self: *Execution,
            comptime Prompt: type,
            prompts: []const Prompt,
            concurrency: ?BatchConcurrency,
        ) ![]const []const u8 {
            if (prompts.len == 0) return &.{};
            if (prompts.len > self.calls_remaining) return error.SubModelCallLimit;
            const outputs = try self.ctx.arena.alloc([]const u8, prompts.len);
            errdefer self.ctx.arena.free(outputs);
            try self.reserve(@intCast(prompts.len));

            var first_error: ?lm.Error = null;
            if (concurrency) |parallel| {
                var start: usize = 0;
                while (start < prompts.len) {
                    const end = start + @min(batch_workers, prompts.len - start);
                    try self.dispatchWave(Prompt, prompts[start..end], outputs[start..end], parallel, &first_error);
                    start = end;
                }
            } else {
                for (prompts, outputs) |prompt, *output| {
                    collect(self.dispatch(self.ctx.arena, asContent(prompt), self.ctx.usage_failure), output, &first_error);
                }
            }
            if (first_error) |err| return err;
            return outputs;
        }

        const batch_workers = 8;

        const BatchTask = struct {
            result: lm.Error![]const u8 = undefined,
            failure: lm.AccountingFailure = .{},

            fn run(
                task: *BatchTask,
                execution: *const Execution,
                arena: Allocator,
                prompt: adapter.Content,
            ) std.Io.Cancelable!void {
                task.result = execution.dispatch(
                    arena,
                    prompt,
                    if (execution.ctx.usage_failure != null) &task.failure else null,
                );
            }
        };

        fn dispatchWave(
            self: *const Execution,
            comptime Prompt: type,
            prompts: []const Prompt,
            outputs: [][]const u8,
            parallel: BatchConcurrency,
            first_error: *?lm.Error,
        ) !void {
            std.debug.assert(prompts.len <= batch_workers);
            std.debug.assert(prompts.len == outputs.len);
            var tasks: [batch_workers]BatchTask = @splat(.{});
            var group: std.Io.Group = .init;
            var started: usize = 0;
            defer {
                // Join before reading tasks, including when dispatch fails partway.
                group.cancel(parallel.io);
                if (self.ctx.usage_failure) |failure| {
                    for (tasks[0..started]) |task| {
                        if (failure.first == null) failure.first = task.failure.first;
                        failure.count +|= task.failure.count;
                    }
                }
            }
            for (prompts, tasks[0..prompts.len]) |prompt, *task| {
                try group.concurrent(parallel.io, BatchTask.run, .{ task, self, parallel.arena, asContent(prompt) });
                started += 1;
            }
            try group.await(parallel.io);
            for (tasks[0..started], outputs) |task, *output| collect(task.result, output, first_error);
        }

        fn collect(result: lm.Error![]const u8, output: *[]const u8, first_error: *?lm.Error) void {
            output.* = result catch |err| switch (err) {
                error.LMFailure => "[ERROR] LMFailure",
                else => {
                    if (first_error.* == null) first_error.* = err;
                    return;
                },
            };
        }

        fn reserve(self: *Execution, count: u32) error{SubModelCallLimit}!void {
            if (count > self.calls_remaining) return error.SubModelCallLimit;
            self.calls_remaining -= count;
        }

        fn dispatch(
            self: *const Execution,
            arena: Allocator,
            prompt: adapter.Content,
            failure: ?*lm.AccountingFailure,
        ) lm.Error![]const u8 {
            return self.model.complete(arena, .{
                .messages = &.{.{ .role = .user, .content = prompt }},
                .usage_tracker = self.ctx.usage_tracker,
                .usage_failure = failure,
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
    /// Use this factory for this invocation only, without changing the action objective.
    interpreter_factory: ?Factory = null,
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
    /// Reject overlong model actions before fence normalization.
    max_action_bytes: ?usize = null,
    /// Return an error rather than asking the model to invent final outputs.
    require_submit: bool = false,
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
    // DSPy's signature field merge keeps the output when it has this name.
    const extract_history_input = !@hasField(Sig.Output, "repl_history");
    const extract_inputs = if (extract_history_input) [_]Field{
        variable_field,
        .{ .name = "repl_history", .kind = .str, .type_name = "REPLHistory", .desc = "Your REPL interactions so far", .prefix = "Repl History:" },
    } else [_]Field{variable_field};
    const extract_fields = extract_inputs ++ if (extract_history_input)
        Sig.output_fields_list[0..Sig.output_fields_list.len].*
    else fields: {
        var outputs: [Sig.output_fields_list.len]Field = undefined;
        var next: usize = 1;
        for (Sig.output_fields_list) |field| {
            if (std.mem.eql(u8, field.name, "repl_history")) {
                outputs[0] = field;
            } else {
                outputs[next] = field;
                next += 1;
            }
        }
        break :fields outputs;
    };
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
        extract: predict.PredictFields(&extract_fields, extract_inputs.len, extract_instructions) = .{},
        factory: Factory,
        max_iters: u32,
        max_llm_calls: u32,
        sub_lm: ?lm.LM,
        max_output_chars: i64,
        history_processor: ?HistoryProcessor,
        max_action_bytes: ?usize,
        require_submit: bool,
        compiled: bool = false,

        pub fn init(arena: Allocator, factory: Factory, options: Options) !Self {
            if (options.max_iters > iterations_max) return error.TooManyIterations;
            inline for (Sig.input_fields_list) |field| {
                if (comptime reservedInput(field.name)) return error.ReservedRLMField;
            }
            inline for (Sig.output_fields_list) |field| {
                if (comptime reservedOutput(field.name)) return error.ReservedRLMField;
            }
            var self: Self = .{
                .generate_action = .{},
                .factory = factory,
                .max_iters = options.max_iters,
                .max_llm_calls = options.max_llm_calls,
                .sub_lm = options.sub_lm,
                .max_output_chars = options.max_output_chars,
                .history_processor = options.history_processor,
                .max_action_bytes = options.max_action_bytes,
                .require_submit = options.require_submit,
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
            const info = try variableInfo(Sig, arena, &variables);
            const entries = try arena.alloc(Entry, self.max_iters);
            var history = try TurnHistory.clone(arena, options.history, entries.len + 1);
            const processor = options.history_processor orelse self.history_processor;
            var execution: Runtime.Execution = .{
                .variables = &variables,
                .model = self.sub_lm orelse ctx.lm,
                .ctx = &ctx,
                .calls_remaining = self.max_llm_calls,
            };
            const factory = options.interpreter_factory orelse self.factory;
            const runtime = try factory.open(factory.ptr);
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
                const over_limit = if (self.max_action_bytes) |limit| entry.code.len > limit else false;
                const result: Runtime.Result = if (over_limit) rejected: {
                    // Do not echo an oversized model response back in the trace.
                    entry.code = "";
                    break :rejected .{ .execution_error = "Python action exceeds the configured source limit." };
                } else try executeAction(arena, runtime, &execution, entry);
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
            if (self.require_submit) return error.RLMSubmissionRequired;
            return self.extractFinal(ctx, info, &history, entries);
        }

        fn extractFinal(self: *const Self, ctx: predict.Context, info: Value, history: *TurnHistory, entries: []const Entry) !Output {
            const arena = ctx.arena;
            var inputs: Record = .empty;
            try inputs.put(arena, "variables_info", info);
            if (extract_history_input) try inputs.put(arena, "repl_history", text(try formatTrajectory(arena, entries)));
            const outputs = try self.extract.state.call(ctx, &inputs);
            var tag: Record = .empty;
            try tag.put(arena, "__dspy_repl__", text("extract_fallback"));
            var event: Record = .empty;
            try event.put(arena, "repl_entry", .{ .object = tag });
            inline for (Sig.output_fields_list) |field| {
                try event.put(arena, field.name, outputs.get(field.name) orelse return error.MissingField);
            }
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

fn reservedHistoryField(name: []const u8) bool {
    for ([_][]const u8{ "history", "variables_info", "iteration", "repl_entry" }) |reserved| {
        if (std.mem.eql(u8, name, reserved)) return true;
    }
    return false;
}

fn reservedInput(name: []const u8) bool {
    if (reservedHistoryField(name)) return true;
    if (std.mem.startsWith(u8, name, "_dspy") or std.mem.startsWith(u8, name, "__dspy")) return true;
    for ([_][]const u8{ "llm_query", "llm_query_batched", "SUBMIT", "print", "dspy", "history_processor", "interpreter_factory" }) |reserved| {
        if (std.mem.eql(u8, name, reserved)) return true;
    }
    return false;
}

fn reservedOutput(name: []const u8) bool {
    if (reservedHistoryField(name)) return true;
    for ([_][]const u8{ "final_reasoning", "trajectory" }) |reserved| {
        if (std.mem.eql(u8, name, reserved)) return true;
    }
    return false;
}

test "RLM rejects inputs and results that collide with runtime-owned names" {
    const testing = std.testing;
    const unopened = struct {
        fn open(_: *anyopaque) anyerror!Runtime {
            return error.UnexpectedRuntimeOpen;
        }
    };
    var context: u8 = 0;
    const factory: Factory = .{ .ptr = &context, .open = unopened.open };
    inline for (.{
        struct {
            pub const Input = struct { interpreter_factory: i64 };
            pub const Output = struct { answer: i64 };
        },
        struct {
            pub const Input = struct { history_processor: i64 };
            pub const Output = struct { answer: i64 };
        },
        struct {
            pub const Input = struct { llm_query: i64 };
            pub const Output = struct { answer: i64 };
        },
        struct {
            pub const Input = struct { dspy: i64 };
            pub const Output = struct { answer: i64 };
        },
        struct {
            pub const Input = struct { question: i64 };
            pub const Output = struct { final_reasoning: i64 };
        },
        struct {
            pub const Input = struct { question: i64 };
            pub const Output = struct { trajectory: i64 };
        },
    }) |Pair| {
        const Program = CustomRLM(signature.Signature(Pair));
        try testing.expectError(error.ReservedRLMField, Program.init(testing.failing_allocator, factory, .{ .action_instructions = "test" }));
    }
}

test "RLM extraction preserves an output named repl_history" {
    const testing = std.testing;
    const Dummy = struct {
        fn open(ptr: *anyopaque) anyerror!Runtime {
            return .{ .ptr = ptr, .execute = execute, .close = close };
        }
        fn execute(_: *anyopaque, _: []const u8, _: *Runtime.Execution) anyerror!Runtime.Result {
            return error.UnexpectedAction;
        }
        fn close(_: *anyopaque) void {}
    };
    var storage: std.heap.ArenaAllocator = .init(testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const Program = CustomRLM(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { first: []const u8, repl_history: i64 };
    }));
    var context: u8 = 0;
    const program = try Program.init(arena, .{ .ptr = &context, .open = Dummy.open }, .{ .action_instructions = "test", .max_iters = 0 });
    var history: lm.History = .{ .arena = arena };
    var script: lm.ScriptedLM = .{
        .history = &history,
        .responses = &.{"<repl_history>\n42\n</repl_history>\n\n<first>\nresult\n</first>"},
    };
    const result = try program.forward(.{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm(), .json_fallback = false }, .{ .question = "input" });
    const system_prompt = history.requests.items[0].messages[0].content.text;
    const repl_history_position = std.mem.indexOf(u8, system_prompt, "1. `repl_history`") orelse return error.WrongExtractionFieldOrder;
    const first_position = std.mem.indexOf(u8, system_prompt, "2. `first`") orelse return error.WrongExtractionFieldOrder;
    try testing.expect(repl_history_position < first_position);
    try testing.expectEqual(@as(i64, 42), result.outputs.repl_history);
    const event = &result.history.messages[0];
    try testing.expectEqualStrings("first", event.keys()[1]);
    try testing.expectEqual(@as(i64, 42), event.get("repl_history").?.integer);
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

fn variableInfo(comptime Sig: type, arena: Allocator, variables: *const Record) !Value {
    const fields = Sig.input_fields_list;
    const preview_types = comptime blk: {
        var metadata: [fields.len]struct {
            name: []const u8,
            dict: bool,
            literal: bool,
        } = undefined;
        for (fields, 0..) |field, index| {
            const T = @FieldType(Sig.Input, field.name);
            const Base = if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
            metadata[index] = .{
                .name = if (Base == T) field.type_name else types.pyName(Base),
                .dict = Base == signature.StrDict or Base == runtime_value.ObjectMap or types.isDict(Base),
                .literal = @typeInfo(Base) == .@"enum" or @import("literal.zig").isLiteral(Base),
            };
        }
        break :blk metadata;
    };
    var items = runtime_value.Array.init(arena);
    try items.ensureTotalCapacity(fields.len);
    for (fields, preview_types) |field, preview_type| {
        const val = variables.get(field.name) orelse return error.MissingField;
        // REPLVariable describes the current value, not its signature annotation.
        const type_name = switch (val) {
            .null => "NoneType",
            .array => "list",
            .object => if (preview_type.dict) "dict" else preview_type.name,
            .string => if (preview_type.literal) "str" else preview_type.name,
            .integer, .number_string => if (preview_type.literal) "int" else preview_type.name,
            .float => if (preview_type.literal) "float" else preview_type.name,
            .bool => if (preview_type.literal) "bool" else preview_type.name,
            else => preview_type.name,
        };
        const rendered = try variableText(arena, val);
        const length = try std.unicode.utf8CountCodepoints(rendered);
        const preview = try variablePreview(arena, rendered, length);
        const description = if (field.desc.len == 0 or std.mem.startsWith(u8, field.desc, "${"))
            ""
        else
            try std.fmt.allocPrint(arena, "Description: {s}\n", .{field.desc});
        var length_buffer: [51]u8 = undefined;
        items.appendAssumeCapacity(text(try std.fmt.allocPrint(
            arena,
            "Variable: `{s}` (access it in your code)\nType: {s}\n{s}Total length: {s} characters\nPreview:\n```\n{s}\n```",
            .{ field.name, type_name, description, pyjson.groupedDecimal(&length_buffer, length), preview },
        )));
    }
    return .{ .array = items };
}

test "RLM variable previews group Unicode counts and cut only beyond one thousand code points" {
    var storage: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { note: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    var input: [2000]u8 = undefined;
    for (0..1000) |i| @memcpy(input[2 * i ..][0..2], "é");
    const longer = try std.mem.concat(arena, u8, &.{ input[0..1000], "X", input[1000..] });
    var vars: Record = .empty;
    for ([_][]const u8{ &input, longer }, [_][]const u8{ "1,000", "1,001" }) |raw, count| {
        try vars.put(arena, "note", text(raw));
        const data = try variableInfo(Sig, arena, &vars);
        const expected = try std.mem.concat(arena, u8, &.{
            "Variable: `note` (access it in your code)\nType: str\nTotal length: ",
            count,
            " characters\nPreview:\n```\n",
            input[0..1000],
            if (raw.len > input.len) "..." else "",
            input[1000..],
            "\n```",
        });
        try std.testing.expectEqualStrings(expected, data.array.items[0].string);
    }
}

test "RLM distinguishes dictionary values from model records in variable previews" {
    var storage: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const Count = struct { value: i64 };
    const Sig = signature.Signature(struct {
        pub const Input = struct { counts: signature.Dict(i64), model: Count };
        pub const Output = struct { answer: i64 };
    });
    const vars = try value.toRecord(arena, Sig.Input, .{
        .counts = .{ .keys = &.{"value"}, .values = &.{2} },
        .model = .{ .value = 2 },
    });
    const info = try variableInfo(Sig, arena, &vars);
    try std.testing.expect(std.mem.indexOf(u8, info.array.items[0].string, "\nType: dict\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, info.array.items[1].string, "\nType: Count\n") != null);
}

test "RLM variable previews describe optional and literal input values" {
    var storage: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer storage.deinit();
    const arena = storage.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct { maybe: ?i64, label: enum { A, B }, enabled: bool };
        pub const Output = struct { answer: i64 };
    });
    const expected = [_][]const u8{
        "Variable: `maybe` (access it in your code)\nType: NoneType\nTotal length: 4 characters\nPreview:\n```\nNone\n```",
        "Variable: `maybe` (access it in your code)\nType: int\nTotal length: 2 characters\nPreview:\n```\n12\n```",
    };
    for ([_]?i64{ null, 12 }, expected) |maybe, preview| {
        const vars = try value.toRecord(arena, Sig.Input, .{ .maybe = maybe, .label = .A, .enabled = false });
        const info = try variableInfo(Sig, arena, &vars);
        try std.testing.expectEqualStrings(preview, info.array.items[0].string);
        try std.testing.expectEqualStrings(
            "Variable: `label` (access it in your code)\nType: str\nTotal length: 1 characters\nPreview:\n```\nA\n```",
            info.array.items[1].string,
        );
    }
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
    var code = whitespace.strip(entry.code);
    const marker = "```";
    if (std.mem.indexOf(u8, code, marker) != null) {
        code = try normalizeFenceLines(arena, code);
        // Drop matching bare decorative lines, without stripping blanks between
        // the pairs. An explicit language line is handled below instead.
        while (std.mem.indexOfScalar(u8, code, '\n')) |first| {
            if (!std.mem.eql(u8, whitespace.strip(code[0..first]), marker)) break;
            const last = std.mem.lastIndexOfScalar(u8, code, '\n').?;
            if (!std.mem.eql(u8, whitespace.strip(code[last + 1 ..]), marker)) break;
            code = if (first == last) "" else code[first + 1 .. last];
        }
        code = whitespace.strip(code);
        if (std.mem.indexOf(u8, code, marker)) |start| {
            const fence = code[start + marker.len ..];
            if (std.mem.indexOfScalar(u8, fence, '\n')) |line_end| {
                const label = whitespace.trimStart(fence[0..line_end]);
                var chars = (try std.unicode.Utf8View.init(label)).iterator();
                while (chars.i < label.len and whitespace.whitespaceEnd(label, chars.i) == chars.i) {
                    _ = chars.nextCodepoint();
                }
                const tag = label[0..chars.i];
                const accepted = tag.len == 0 or
                    std.ascii.eqlIgnoreCase(tag, "python") or std.ascii.eqlIgnoreCase(tag, "py") or
                    std.ascii.eqlIgnoreCase(tag, "python3") or std.ascii.eqlIgnoreCase(tag, "py3");
                if (!accepted) {
                    const language = try unicode.encode(arena, try unicode.lower(arena, try unicode.decode(arena, tag)));
                    // Match DSPy: the rejected action is reported as received,
                    // rather than publishing a partly normalized action.
                    return .{ .execution_error = try std.fmt.allocPrint(
                        arena,
                        "Expected Python code but got ```{s} fence. Write Python code, not {s}.",
                        .{ language, language },
                    ) };
                }
                const body = fence[line_end + 1 ..];
                code = whitespace.strip(body[0 .. std.mem.indexOf(u8, body, marker) orelse body.len]);
            }
        }
    }
    entry.code = code;
    return runtime.execute(runtime.ptr, entry.code, execution);
}

/// `"\n".join(code.splitlines())` for already stripped action text. Ordinary
/// LF-only actions borrow their input; other Python line separators shrink.
fn normalizeFenceLines(arena: Allocator, code: []const u8) ![]const u8 {
    var lines: adapter.LineIterator = .{ .text = code };
    var previous: usize = 0;
    while (lines.next()) |line| {
        const end = previous + line.len;
        if (lines.index != end and !std.mem.eql(u8, code[end..lines.index], "\n")) {
            const buffer = try arena.alloc(u8, code.len);
            @memcpy(buffer[0..end], code[0..end]);
            var used = end;
            while (lines.next()) |next| {
                buffer[used] = '\n';
                used += 1;
                @memcpy(buffer[used..][0..next.len], next);
                used += next.len;
            }
            return buffer[0..used];
        }
        previous = lines.index;
    }
    return code;
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

fn submittedTypeName(submitted: Value) []const u8 {
    return switch (submitted) {
        .null => "NoneType",
        .bool => "bool",
        .integer => "int",
        .number_string => |number| if (std.json.isNumberFormattedLikeAnInteger(number)) "int" else "float",
        .float => "float",
        .string => "str",
        .array => "list",
        .object => "dict",
        .custom => |custom| custom.vtable.model_name,
    };
}

fn parseSubmit(arena: Allocator, fields: []const Field, submitted: Value, entry: *Entry) !?Record {
    if (submitted != .object) {
        var buffer: std.Io.Writer.Allocating = .init(arena);
        try buffer.writer.print("[Error] FINAL returned {s}, expected dict with fields: [", .{submittedTypeName(submitted)});
        for (fields, 0..) |field, i| {
            if (i != 0) try buffer.writer.writeAll(", ");
            try pyjson.writeRepr(&buffer.writer, text(field.name), 0);
        }
        try buffer.writer.writeByte(']');
        entry.output = buffer.written();
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
    var diagnostics: std.Io.Writer.Allocating = .init(arena);
    for (fields) |field| {
        const raw = submitted.object.get(field.name).?;
        const parsed = adapter.parseFieldValue(arena, field, raw) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            try diagnostics.writer.writeAll(if (diagnostics.written().len == 0) "[Type Error] " else "; ");
            try diagnostics.writer.print("{s}: expected {s}, got {s}: {s}", .{
                field.name, field.type_name, submittedTypeName(raw), @errorName(err),
            });
            continue;
        };
        if (diagnostics.written().len == 0) try outputs.put(arena, field.name, parsed);
    }
    if (diagnostics.written().len != 0) {
        entry.output = diagnostics.written();
        return null;
    }
    for (outputs.values()) |*output| output.* = try value.cloneValue(arena, output.*);
    return outputs;
}

test "RLM submission observation covers every invalid output in signature order" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { z_count: i64, a_count: i64 };
    });
    var entry: Entry = undefined;

    var raw: Record = .empty;
    try raw.put(arena, "a_count", text("invalid"));
    try raw.put(arena, "z_count", text("invalid"));
    try std.testing.expectEqual(null, try parseSubmit(arena, Sig.output_fields_list, .{ .object = raw }, &entry));
    try std.testing.expect(std.mem.startsWith(u8, entry.output, "[Type Error] z_count:"));
    try std.testing.expect(std.mem.indexOf(u8, entry.output, "; a_count:") != null);
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
