//! Bridge from typed RLM calls to the fixed-storage Python guest. The RLM
//! controller and provider retain their normal caller-supplied allocation/I/O.
const std = @import("std");
const py = @import("bounded_python.zig");
const rlm = @import("rlm.zig");
const value = @import("runtime_value.zig");

pub const Session = struct {
    machine: py.Machine,
    options: py.Options,
    active: std.atomic.Value(bool),
    allow_query: bool,

    /// Initialize in place. Keep this address stable until the last program
    /// borrowing its factory is done. Use one session per concurrent invocation.
    pub fn init(self: *Session, options: py.Options, allow_query: bool) void {
        self.machine.init();
        self.options = options;
        self.active = .init(false);
        self.allow_query = allow_query;
    }

    pub fn factory(self: *Session) rlm.Factory {
        return .{ .ptr = self, .open = open };
    }

    fn open(ptr: *anyopaque) anyerror!rlm.Runtime {
        const self: *Session = @ptrCast(@alignCast(ptr));
        if (self.active.cmpxchgStrong(false, true, .acquire, .monotonic) != null)
            return error.BoundedPythonSessionInUse;
        self.machine.reset();
        return .{ .ptr = self, .execute = execute, .close = close };
    }

    fn close(ptr: *anyopaque) void {
        const self: *Session = @ptrCast(@alignCast(ptr));
        std.debug.assert(self.active.load(.monotonic));
        self.machine.reset();
        self.active.store(false, .release);
    }

    fn execute(ptr: *anyopaque, source: []const u8, execution: *rlm.Runtime.Execution) anyerror!rlm.Runtime.Result {
        const self: *Session = @ptrCast(@alignCast(ptr));
        std.debug.assert(self.active.load(.monotonic));
        var bindings: [py.bindings_max]py.Binding = undefined;
        const count = execution.variables.count();
        if (count > bindings.len) return error.TooManyPythonInputs;
        for (execution.variables.keys(), execution.variables.values(), bindings[0..count]) |key, input, *binding| {
            binding.* = .{ .name = key, .value = switch (input) {
                .null => .none,
                .bool => .{ .boolean = input.bool },
                .integer => .{ .integer = input.integer },
                .string => .{ .string = input.string },
                else => return error.UnsupportedPythonInput,
            } };
        }
        var query: Query = .{ .execution = execution };
        const host: ?py.Host = if (self.allow_query) .{ .ptr = &query, .query = Query.call } else null;
        const result = self.machine.run(source, bindings[0..count], self.options, host) catch |err| {
            if (query.failure) |failure| {
                // An exhausted sub-LM quota is a recoverable model action;
                // transport, cancellation and accounting failures are not.
                if (failure == error.SubModelCallLimit or failure == error.EmptySubModelPrompt)
                    return .{ .execution_error = @errorName(failure) };
                return failure;
            }
            return .{ .execution_error = @errorName(err) };
        };
        return switch (result) {
            .observation => |output| .{ .observation = output },
            .submit => |fields| block: {
                const arena = execution.ctx.arena;
                var record: value.ObjectMap = .empty;
                try record.ensureTotalCapacity(arena, fields.len);
                for (fields) |field| record.putAssumeCapacity(field.name, switch (field.value) {
                    .none => .null,
                    .boolean => |b| .{ .bool = b },
                    .integer => |n| .{ .integer = n },
                    .string => |s| .{ .string = s },
                });
                break :block .{ .submit = .{ .object = record } };
            },
        };
    }

    const Query = struct {
        execution: *rlm.Runtime.Execution,
        failure: ?anyerror = null,

        fn call(ptr: *anyopaque, prompt: []const u8) anyerror![]const u8 {
            const self: *Query = @ptrCast(@alignCast(ptr));
            return self.execution.query(prompt) catch |err| {
                self.failure = err;
                return error.GuestHostFailed;
            };
        }
    };
};

const profile =
    "Write Python source, not Zig, pseudocode, an expression-only answer, or JSON. " ++
    "Use one statement per line and indent a while body with four spaces. " ++
    "Use scalar variables, simple assignment (x = x + 1), signed 64-bit integer arithmetic, " ++
    "comparisons, parentheses, string literals, None, True, and False. " ++
    "Use print(value) for a short observation and finish with a SUBMIT call using the exact output names below as Python keywords. " ++
    "State persists between actions. No imports, files, network, classes, collections, or arbitrary functions. " ++
    "Keep source and execution small; syntax, storage, and execution limits are enforced by the guest.";

/// Start an RLM program that can finish only by executing a valid typed
/// SUBMIT. `options.action_instructions` describes the task and output fields.
pub fn initProgram(comptime Sig: type, arena: std.mem.Allocator, session: *Session, options: rlm.Options) !rlm.CustomRLM(Sig) {
    var bounded = options;
    bounded.require_submit = true;
    bounded.max_action_bytes = @min(options.max_action_bytes orelse std.math.maxInt(usize), py.source_bytes_max + 64);
    const queries = if (session.allow_query)
        "\nllm_query(text) is available under the configured sub-model call budget."
    else
        "\nNo sub-model queries are available.";
    const submit_args = comptime blk: {
        var args: []const u8 = "";
        for (Sig.output_fields_list, 0..) |field, index| {
            args = args ++ (if (index == 0) "" else ", ") ++ field.name ++ "=" ++ field.name;
        }
        break :blk args;
    };
    bounded.action_instructions = try std.fmt.allocPrint(arena, "{s}\n\n{s}\nAfter assigning the output variables, end the Python code with SUBMIT({s}). Assignment or print alone never completes the task.{s}", .{
        options.action_instructions, profile, submit_args, queries,
    });
    return rlm.CustomRLM(Sig).init(arena, session.factory(), bounded);
}

test "bounded Python RLM preserves valid turns and requires a typed guest submit" {
    const lm = @import("lm.zig");
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { n: i64 };
        pub const Output = struct { answer: i64 };
    });
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const session = try arena.create(Session);
    session.init(.{ .fuel = 4096 }, false);

    var program = try initProgram(Sig, arena, session, .{
        .action_instructions = "Compute answer by executing Python.",
        .max_iters = 3,
    });
    var model: lm.ScriptedLM = .{ .responses = &.{
        "<plan>Save the total</plan>\n<code>\n```python\ntotal = n + 1\nprint(total)\n```\n</code>",
        "<plan>Try an unsupported action</plan>\n<code>\n```python\ntotal = 99\nimport os\n```\n</code>",
        "<plan>Submit the existing total</plan>\n<code>\n```python\nSUBMIT(answer=total)\n```\n</code>",
    } };
    const result = try program.forward(.{ .arena = arena, .lm = model.lm(), .json_fallback = false }, .{ .n = 6 });
    try std.testing.expectEqual(@as(i64, 7), result.outputs.answer);
    try std.testing.expectEqualStrings("7\n", result.trajectory[0].output);
    try std.testing.expect(std.mem.startsWith(u8, result.trajectory[1].output, "[Error] "));
    try std.testing.expectEqual(@as(usize, 3), model.index);

    program.max_iters = 1;
    var no_submit: lm.ScriptedLM = .{ .responses = &.{
        "<plan>Try state from the preceding session</plan>\n<code>\n```python\nSUBMIT(answer=total)\n```\n</code>",
        "<answer>99</answer>",
    } };
    try std.testing.expectError(error.RLMSubmissionRequired, program.forward(.{
        .arena = arena,
        .lm = no_submit.lm(),
        .json_fallback = false,
    }, .{ .n = 100 }));
    try std.testing.expectEqual(@as(usize, 1), no_submit.index);
}
