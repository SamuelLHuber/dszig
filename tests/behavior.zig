//! Behavioural properties: optimizer convergence over generations, trace and
//! compile idempotency, leak freedom, and bounded fixed-buffer operation.

const std = @import("std");
const dszig = @import("dszig");
const rag = @import("rag_program");
const testing = std.testing;
const Allocator = std.mem.Allocator;

test "custom RLM retains observations across runtime buffer reuse and closes on failure" {
    const Runtime = struct {
        buffer: [5]u8 = "first".*,
        calls: u32 = 0,
        closes: u32 = 0,
        fail: bool = false,

        fn open(ptr: *anyopaque) !dszig.rlm.Runtime {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls = 0;
            self.buffer = "first".*;
            return .{ .ptr = ptr, .execute = execute, .close = close };
        }

        fn execute(
            ptr: *anyopaque,
            code: []const u8,
            execution: *dszig.rlm.Runtime.Execution,
        ) !dszig.rlm.Runtime.Result {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            if (self.fail) return error.RuntimeDisconnected;
            try testing.expectError(error.SubModelCallLimit, execution.query("unavailable"));
            if (self.calls == 1) {
                try testing.expectEqualStrings("inspect()", code);
            } else {
                try testing.expectEqualStrings("verify()", code);
                self.buffer = "later".*;
            }
            return .{ .observation = &self.buffer };
        }

        fn close(ptr: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.closes += 1;
            self.buffer = "ended".*;
        }
    };
    const Sig = dszig.Signature(struct {
        pub const Input = struct { order: i64 };
        pub const Output = struct { answer: i64 };
    });
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    var runtime: Runtime = .{};
    const program = try dszig.CustomRLM(Sig).init(
        arena,
        .{ .ptr = &runtime, .open = Runtime.open },
        .{
            .max_iters = 2,
            .max_llm_calls = 0,
            .action_instructions = "Check the order with the supplied runtime.",
        },
    );
    var model: dszig.lm.ScriptedLM = .{
        .responses = &.{
            "<plan>Inspect.</plan><code>```\r\n```Py3\r\ninspect()\r\n```\r\n```</code>",
            "<plan>Verify.</plan><code>verify()</code>",
            "<answer>17</answer>",
        },
    };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = model.lm() };
    const Processor = struct {
        fn trim(_: *anyopaque, _: Allocator, draft: []dszig.value.Record) !dszig.rlm.HistoryProcessor.Edit {
            if (draft.len == 0) return .keep;
            return .{ .truncate = 0 };
        }
    };
    const result = try program.forwardWithOptions(ctx, .{ .order = 10 }, .{
        .history_processor = .{ .ptr = &runtime, .process = Processor.trim },
    });
    try testing.expectEqual(@as(i64, 17), result.outputs.answer);
    try testing.expectEqual(@as(usize, 2), result.history.messages.len);
    try testing.expectEqualStrings("2/2", result.history.messages[0].get("iteration").?.string);
    const fallback = result.history.messages[1].get("repl_entry").?.object;
    try testing.expectEqualStrings("extract_fallback", fallback.get("__dspy_repl__").?.string);
    try testing.expectEqualStrings("first", result.trajectory[0].output);
    try testing.expectEqualStrings("later", result.trajectory[1].output);
    try testing.expectEqual(@as(u32, 1), runtime.closes);

    runtime.fail = true;
    model.index = 0;
    try testing.expectError(error.RuntimeDisconnected, program.forward(ctx, .{ .order = 10 }));
    try testing.expectEqual(@as(usize, 1), model.index);
    try testing.expectEqual(@as(u32, 2), runtime.closes);
}

test "RLM batch reserves atomically and isolates only provider failures" {
    const Model = struct {
        calls: u32 = 0,

        fn complete(
            ptr: *anyopaque,
            arena: Allocator,
            request: dszig.lm.Request,
        ) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            const content = request.messages[0].content;
            if (content == .parts) {
                if (content.parts.len != 3) return error.ScriptExhausted;
                const expected = [_][]const u8{ "text", "image", "text" };
                for (content.parts, expected) |part, kind| {
                    if (!std.mem.eql(u8, part.object.get("type").?.string, kind)) return error.ScriptExhausted;
                }
                return arena.dupe(dszig.lm.Completion, &.{.{ .text = "media" }});
            }
            const prompt = content.text;
            if (std.mem.eql(u8, prompt, "provider")) return error.LMFailure;
            if (std.mem.eql(u8, prompt, "contract")) return error.NoLM;
            return arena.dupe(dszig.lm.Completion, &.{.{ .text = "verified" }});
        }
    };
    var backing: std.heap.ArenaAllocator = .init(testing.allocator);
    defer backing.deinit();
    var model: Model = .{};
    const ctx: dszig.Context = .{
        .arena = backing.allocator(),
        .lm = .{ .ptr = &model, .vtable = &.{ .complete = Model.complete } },
    };
    var execution: dszig.rlm.Runtime.Execution = .{
        .variables = &dszig.value.Record.empty,
        .model = ctx.lm,
        .ctx = &ctx,
        .calls_remaining = 4,
    };
    const local = try std.json.parseFromSliceLeaky(dszig.runtime_value.Value, ctx.arena,
        \\{"type":"image","media_type":"image/png","path":"/must-not-read"}
    , .{});
    const malformed = try std.json.parseFromSliceLeaky(dszig.runtime_value.Value, ctx.arena,
        \\{"type":"image","media_type":"image/png","data":"not base64!"}
    , .{});
    try testing.expectError(error.SubModelHostFile, execution.queryContentBatch(&.{
        .{ .text = "valid first" }, .{ .parts = &.{local} },
    }, null));
    try testing.expectError(error.InvalidSubModelPart, execution.queryContentBatch(&.{
        .{ .parts = &.{malformed} },
    }, null));
    try testing.expectError(error.EmptySubModelPrompt, execution.queryParts(&.{}));
    try testing.expectEqual(@as(u32, 4), execution.calls_remaining);
    try testing.expectError(error.SubModelCallLimit, execution.queryBatch(&.{
        "blank", "provider", "extra", "extra", "extra",
    }, null));
    try testing.expectEqual(@as(u32, 0), model.calls);
    const parts = try std.json.parseFromSliceLeaky(dszig.Value, ctx.arena,
        \\[{"type":"text","text":""},{"type":"image","media_type":"image/png","data":"eA=="},{"type":"text","text":"end"}]
    , .{});
    const outputs = try execution.queryContentBatch(&.{
        .{ .text = "" }, .{ .parts = parts.array.items }, .{ .text = "provider" },
    }, null);
    try testing.expectEqualStrings("verified", outputs[0]);
    try testing.expectEqualStrings("media", outputs[1]);
    try testing.expectEqualStrings("[ERROR] LMFailure", outputs[2]);

    try testing.expectError(error.NoLM, execution.queryBatch(&.{"contract"}, null));
    try testing.expectEqual(@as(u32, 4), model.calls);
    try testing.expectError(error.SubModelCallLimit, execution.query("over budget"));
}

const BootstrapTraceSignature = dszig.Signature(struct {
    pub const Input = struct { token: []const u8 };
    pub const Output = struct { first: i64, second: i64, third: i64 = 9 };
});

const BootstrapTracePair = struct {
    compiled: bool = false,
    left: dszig.Predict(BootstrapTraceSignature) = .{},
    right: dszig.Predict(BootstrapTraceSignature) = .{},
    pub const Input = struct { question: []const u8 };
    pub const Output = BootstrapTraceSignature.Output;

    pub fn forward(self: *const @This(), ctx: dszig.Context, _: Input) !Output {
        _ = try self.left.forward(ctx, .{ .token = "good" });
        return self.right.forward(ctx, .{ .token = "bad" });
    }
};

fn bootstrapTraceMetric(
    _: Allocator,
    _: *const dszig.Example,
    prediction: ?*const dszig.optimize.RecordPrediction,
    trace: ?*const dszig.Trace,
) dszig.value.ParseError!f64 {
    if (trace != null or prediction == null) return error.ParseValue;
    return @as(f64, @floatFromInt(prediction.?.get("first").?.integer)) / 10;
}

test "trace bootstrap retains preceding calls and attributes failure by signature class" {
    var retained: std.heap.ArenaAllocator = .init(testing.allocator);
    defer retained.deinit();
    const out = retained.allocator();
    const example = try dszig.Example.fromJson(out, "{\"question\":\"outer\"}", &.{"question"});
    var program: BootstrapTracePair = .{};
    const responses = [_][2][]const u8{
        .{
            "[[ ## first ## ]]\n1\n[[ ## second ## ]]\n2",
            "[[ ## second ## ]]\n2",
        },
        .{ "{\"first\":1,\"second\":2}", "{\"second\":\"2\"}" },
        .{ "<first>1</first><second>2</second>", "<second>2</second>" },
    };
    for ([_]dszig.Adapter{ .chat, .json, .xml }, responses) |adapter, bodies| {
        const result = block: {
            var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
            defer scratch.deinit();
            var lm: dszig.lm.ScriptedLM = .{ .responses = &bodies };
            break :block try dszig.bootstrap_trace.bootstrapTraceData(
                BootstrapTracePair,
                out,
                .{
                    .arena = scratch.allocator(),
                    .lm = lm.lm(),
                    .adapter = adapter,
                    .json_fallback = false,
                },
                &program,
                &.{example},
                .{ .metric = bootstrapTraceMetric, .failure_score = 0.5 },
            );
        };
        const row = result[0];
        try testing.expectEqual(@as(usize, 0), row.example_ind);
        // One supplied field plus the default gives 2/3 partial format credit.
        try testing.expectEqual(@as(?f64, 0), row.score);
        try testing.expectEqualStrings(bodies[1], row.prediction.failed.completion_text);
        try testing.expectEqual(@as(?f64, 0), row.prediction.failed.format_reward);
        try testing.expectEqual(@as(usize, 2), row.trace.len);
        try testing.expect(row.trace[0].predictor == &program.left.state);
        try testing.expectEqual(@as(i64, 9), row.trace[0].prediction.success.get("third").?.integer);
        // The right call failed, but DSPy selects the first equal signature class.
        try testing.expect(row.trace[1].predictor == &program.left.state);
        try testing.expectEqualStrings("outer", row.trace[1].inputs.get("question").?.string);
        try testing.expect(!row.trace[1].inputs.contains("token"));
        try testing.expectEqualStrings(bodies[1], row.trace[1].prediction.failed.completion_text);
    }
}

test "Chat prediction preserves bracketed text that is not a Unicode field header" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const intro = "Deployment is approved.";
    const caveat = "Only after the backup finishes.";
    // Pinned DSPy ChatAdapter recognizes precomposed é as a word character,
    // but not the combining mark in e + U+0301 or the warning symbol.
    for ([_][]const u8{ "⚠", "é", "e\u{301}" }, [_]bool{ false, true, false }) |marker, is_header| {
        const line = try std.fmt.allocPrint(arena, "[[ ## {s} ## ]]", .{marker});
        const completion = try std.fmt.allocPrint(arena, "[[ ## answer ## ]]\n{s}\n{s}\n{s}\n[[ ## completed ## ]]", .{ intro, line, caveat });
        var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{completion} };
        const result = try program.forward(
            .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = scripted.lm(), .adapter = .chat, .json_fallback = false },
            .{ .question = "Can I deploy?" },
        );
        const expected = if (is_header) intro else try std.mem.join(arena, "\n", &.{ intro, line, caveat });
        try testing.expectEqualStrings(expected, result.answer);
        try testing.expectEqual(@as(usize, 1), scripted.index);
    }
}

test "prediction and traces retain ordered provider token scores without adding demo fields" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Engine = struct {
        index: usize = 0,

        fn complete(ptr: *anyopaque, arena: Allocator, _: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const rows: []const dszig.lm.Completion = &.{
                .{ .text = "bad XML", .logprobs = &.{.{ .token = "bad", .logprob = -1 }} },
                .{ .text = "<answer>discarded</answer>" },
                .{ .text = "{\"answer\":\"4\"}", .logprobs = &.{.{
                    .token = "4",
                    .logprob = -0.25,
                    .bytes = &.{.{ .integer = 52 }},
                    .token_id = .{ .integer = 91 },
                    .top = &.{.{ .token = "5", .logprob = -2, .token_id = .{ .integer = 92 } }},
                }}, .logprobs_complete = false },
                .{ .text = "{\"answer\":\"6\"}", .logprobs = &.{.{ .token = "6", .logprob = -0.5 }} },
                .{ .text = "<answer>4</answer>", .logprobs = &.{} },
                .{ .text = "<answer>4</answer>" },
                .{ .text = "<answer>4</answer>", .logprobs = &.{} },
                .{ .text = "<answer>6</answer>" },
            };
            if (self.index >= rows.len) return error.LMFailure;
            const row = rows[self.index];
            self.index += 1;
            const out = try arena.alloc(dszig.lm.Completion, 1);
            out[0] = try dszig.lm.cloneCompletion(arena, row);
            return out;
        }
    };
    var engine: Engine = .{};
    const model: dszig.lm.LM = .{ .ptr = &engine, .vtable = &.{ .complete = Engine.complete } };
    var retained: std.heap.ArenaAllocator = .init(testing.allocator);
    defer retained.deinit();
    var cloned: dszig.Prediction(Sig.Output) = undefined;
    var trace: dszig.Trace = .{ .arena = retained.allocator() };
    const program: dszig.Predict(Sig) = .{};
    {
        var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
        defer scratch.deinit();
        const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = scratch.allocator(), .lm = model, .lm_options = .{ .n = 2 }, .trace = &trace };
        const prediction = try program.call(ctx, .{ .question = "q" });
        try testing.expectEqualStrings("4", prediction.fields().answer);
        try testing.expectEqualStrings("6", prediction.completions().?[1].answer);
        try testing.expectEqualStrings("4", prediction.completionLogprobs(0).?.tokens[0].token);
        cloned = try prediction.clone(retained.allocator());

        var single = ctx;
        single.lm_options.n = 1;
        const empty = try program.call(single, .{ .question = "q" });
        try testing.expectEqual(@as(usize, 0), empty.completionLogprobs(0).?.tokens.len);
        try testing.expect(empty.completionLogprobs(0).?.reported);
        const absent = try program.call(single, .{ .question = "q" });
        try testing.expect(absent.completionLogprobs(0) == null);

        try testing.expectError(error.InconsistentCompletionLogprobs, program.call(ctx, .{ .question = "q" }));
        try testing.expectEqual(@as(usize, 3), trace.steps.items.len);
    }
    try testing.expectEqual(@as(usize, 8), engine.index);
    const first = cloned.completionLogprobs(0).?;
    try testing.expect(!first.complete);
    try testing.expectEqual(@as(usize, 1), first.tokens.len);
    try testing.expectEqualStrings("4", first.tokens[0].token);
    try testing.expectEqual(@as(f64, -0.25), first.tokens[0].logprob);
    try testing.expectEqual(@as(i64, 52), first.tokens[0].bytes.?[0].integer);
    try testing.expectEqual(@as(i64, 91), first.tokens[0].token_id.integer);
    try testing.expectEqualStrings("5", first.tokens[0].top[0].token);
    try testing.expectEqual(@as(f64, -2), first.tokens[0].top[0].logprob);
    try testing.expectEqual(@as(i64, 92), first.tokens[0].top[0].token_id.integer);
    const second = cloned.completionLogprobs(1).?;
    try testing.expect(second.complete);
    try testing.expectEqualStrings("6", second.tokens[0].token);
    try testing.expectEqual(@as(f64, -0.5), second.tokens[0].logprob);

    const step = trace.steps.items[0];
    const demo = try dszig.optimize.traceDemo(retained.allocator(), step, 1_000);
    try testing.expectEqualStrings("4", demo.get("answer").?.string);
    try testing.expect(!demo.contains("logprobs"));
    try testing.expectEqualDeep(first, step.completionLogprobs(0).?);
    try testing.expectEqualDeep(second, step.completionLogprobs(1).?);
    try testing.expect(step.completionLogprobs(2) == null);
    try testing.expectEqual(@as(usize, 0), trace.steps.items[1].completionLogprobs(0).?.tokens.len);
    try testing.expect(trace.steps.items[2].completionLogprobs(0) == null);

    engine.index = 0;
    const example = try dszig.Example.fromJson(retained.allocator(), "{\"question\":\"q\"}", &.{"question"});
    const captured = block: {
        var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
        defer scratch.deinit();
        break :block try dszig.bootstrap_trace.bootstrapTraceData(
            @TypeOf(program),
            retained.allocator(),
            .{ .workspace = dszig.workspace.forTests(), .arena = scratch.allocator(), .lm = model, .lm_options = .{ .n = 2 } },
            &program,
            &.{example},
            .{},
        );
    };
    const retained_step = captured[0].trace[0];
    try testing.expectEqualStrings("4", retained_step.prediction.success.get("answer").?.string);
    try testing.expect(!retained_step.prediction.success.contains("logprobs"));
    try testing.expectEqualDeep(first, retained_step.completionLogprobs(0).?);
    try testing.expectEqualDeep(second, retained_step.completionLogprobs(1).?);
}

test "Predict output caps and nucleus sampling override defaults without leaking across calls or cache entries" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, responses: bool, cap: ?u32, top_p: ?f64, answer: []const u8) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings(if (responses) "/v1/responses" else "/v1/chat/completions", request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer parsed.deinit();
            const wire = parsed.value.object;
            const key = if (responses) "max_output_tokens" else "max_completion_tokens";
            try testing.expect(!wire.contains(if (responses) "max_completion_tokens" else "max_output_tokens"));
            try testing.expect(!wire.contains("max_tokens"));
            if (top_p) |expected| {
                const actual = wire.get("top_p") orelse return error.MissingTopP;
                try testing.expectEqual(expected, switch (actual) {
                    .float => actual.float,
                    .integer => @as(f64, @floatFromInt(actual.integer)),
                    else => return error.InvalidWireTopP,
                });
            } else {
                try testing.expect(!wire.contains("top_p"));
            }
            if (cap) |limit| {
                try testing.expectEqual(@as(std.json.Value, .{ .integer = limit }), wire.get(key) orelse return error.MissingTokenCap);
            } else {
                try testing.expect(!wire.contains(key));
            }
            var reply_buf: [512]u8 = undefined;
            const reply = if (responses)
                try std.fmt.bufPrint(&reply_buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{s}</answer>\"}}]}}]}}", .{answer})
            else
                try std.fmt.bufPrint(&reply_buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{answer});
            try request.respond(reply, .{ .keep_alive = false });
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()});
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .max_tokens = 48 };
    var sampling = plain;
    sampling.state.config = &.{ .top_p = 0.5 };
    const inputs: Sig.Input = .{ .question = "q" };
    for ([_]bool{ false, true }) |responses| {
        var provider: dszig.lm.OpenAI = .{
            .client = &client,
            .base_url = base_url,
            .model = "gpt-4o-mini",
            .api = if (responses) .responses else .chat_completions,
            .max_tokens = 96,
            .max_retries = 0,
            .timeout_ns = 2 * std.time.ns_per_s,
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = provider.lm(), .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = 10 };
        defer cache.deinit();
        const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .json_fallback = false };
        const Case = struct {
            program: *const dszig.Predict(Sig),
            context_cap: ??u32 = null,
            context_top_p: ?f64 = null,
            override: ?*const dszig.PredictConfig = null,
            wire_cap: ?u32 = 96,
            wire_top_p: ?f64 = null,
            answer: []const u8,
        };
        const cases = [_]Case{
            .{ .program = &configured, .context_cap = 64, .override = &.{ .max_tokens = 24 }, .wire_cap = 24, .answer = "call" },
            .{ .program = &configured, .context_cap = 64, .wire_cap = 48, .answer = "predictor" },
            .{ .program = &configured, .override = &.{ .max_tokens = @as(?u32, null) }, .wire_cap = null, .answer = "cleared" },
            .{ .program = &plain, .context_cap = 64, .wire_cap = 64, .answer = "context" },
            .{ .program = &sampling, .context_top_p = 0.75, .override = &.{ .top_p = @as(?f64, null) }, .answer = "provider" },
            .{ .program = &sampling, .context_top_p = 0.75, .override = &.{ .top_p = 0 }, .wire_top_p = 0, .answer = "call_top_p" },
            .{ .program = &sampling, .context_top_p = 0.75, .wire_top_p = 0.5, .answer = "stored_top_p" },
            .{ .program = &plain, .context_top_p = 0.75, .wire_top_p = 0.75, .answer = "context_top_p" },
            .{ .program = &plain, .override = &.{ .top_p = 1 }, .wire_top_p = 1, .answer = "upper_top_p" },
        };
        for (cases) |case| {
            var call_ctx = ctx;
            call_ctx.lm_options.max_tokens = case.context_cap;
            call_ctx.lm_options.top_p = case.context_top_p;
            call_ctx.predict_config = case.override;
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, responses, case.wire_cap, case.wire_top_p, case.answer });
            defer serving.cancel(io) catch {};
            const answer = try case.program.call(call_ctx, inputs);
            try serving.await(io);
            try testing.expectEqualStrings(case.answer, answer.fields().answer);
            // No listening task on repeat: a wrong cache identity either returns
            // another case's answer or waits for a request that should not occur.
            try testing.expectEqualStrings(case.answer, (try case.program.call(call_ctx, inputs)).fields().answer);
        }

        try testing.expectEqualStrings("provider", (try plain.call(ctx, inputs)).fields().answer);
        try testing.expectEqual(cases.len, recording.calls.items.len);
        for (recording.calls.items) |*call| {
            const metadata = try dszig.lm.optionsValue(arena, call.options.?);
            call.options = try dszig.lm.optionsFromValue(arena, metadata);
        }
        var used: [cases.len]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
        // Replay out of order, so matching only the prompt will fail.
        var i = cases.len;
        while (i != 0) {
            i -= 1;
            const case = cases[i];
            var replay_ctx = ctx;
            replay_ctx.lm = replay.lm();
            replay_ctx.lm_options.max_tokens = case.context_cap;
            replay_ctx.lm_options.top_p = case.context_top_p;
            replay_ctx.predict_config = case.override;
            try testing.expectEqualStrings(case.answer, (try case.program.call(replay_ctx, inputs)).fields().answer);
        }
        try testing.expect(replay.exhausted());
        try testing.expectEqual(@as(usize, 0), replay.misses);
    }
}

test "Predict seed overrides advance each candidate and distinguish cache and replay requests" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Expected = struct { seed: ?[]const u8, answer: []const u8 };
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, responses: bool, expected: []const Expected) !void {
            for (expected) |row| {
                const socket = try listener.accept(io);
                defer socket.close(io);
                var read_buf: [4096]u8 = undefined;
                var write_buf: [1024]u8 = undefined;
                var input = socket.reader(io, &read_buf);
                var output = socket.writer(io, &write_buf);
                var server = std.http.Server.init(&input.interface, &output.interface);
                var request = try server.receiveHead();
                try testing.expectEqualStrings(if (responses) "/v1/responses" else "/v1/chat/completions", request.head.target);
                var body_buf: [4096]u8 = undefined;
                const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
                defer testing.allocator.free(body);
                const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{ .parse_numbers = false });
                defer parsed.deinit();
                const wire = parsed.value.object;
                try testing.expect(!wire.contains("n"));
                if (!responses and row.seed != null) {
                    const seed = wire.get("seed") orelse return error.MissingWireSeed;
                    try testing.expectEqual(.number_string, std.meta.activeTag(seed));
                    try testing.expectEqualStrings(row.seed.?, seed.number_string);
                } else {
                    try testing.expect(!wire.contains("seed"));
                }
                var reply_buf: [512]u8 = undefined;
                const reply = if (responses)
                    try std.fmt.bufPrint(&reply_buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{s}</answer>\"}}]}}]}}", .{row.answer})
                else
                    try std.fmt.bufPrint(&reply_buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{row.answer});
                try request.respond(reply, .{ .keep_alive = false });
            }
        }

        fn call(program: *const dszig.Predict(Sig), ctx: dszig.Context, expected: []const Expected) !void {
            const prediction = try program.call(ctx, .{ .question = "q" });
            try testing.expectEqualStrings(expected[0].answer, prediction.fields().answer);
            if (expected.len > 1) {
                const candidates = prediction.completions() orelse return error.MissingCandidates;
                try testing.expectEqual(expected.len, candidates.len);
                for (expected, candidates) |row, result| try testing.expectEqualStrings(row.answer, result.answer);
            }
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()});
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .seed = .{ .integer = 41 } };
    const Case = struct {
        program: *const dszig.Predict(Sig),
        config: ?*const dszig.PredictConfig = null,
        expected: []const Expected,
    };
    const cases = [_]Case{
        .{ .program = &configured, .config = &.{ .seed = .{ .integer = 0 }, .n = 2 }, .expected = &.{ .{ .seed = "0", .answer = "zero" }, .{ .seed = "1", .answer = "one" } } },
        .{ .program = &configured, .expected = &.{.{ .seed = "41", .answer = "stored" }} },
        .{ .program = &configured, .config = &.{ .seed = @as(dszig.Value, .null) }, .expected = &.{.{ .seed = null, .answer = "cleared" }} },
        .{ .program = &plain, .expected = &.{.{ .seed = "73", .answer = "context" }} },
        .{ .program = &configured, .config = &.{ .seed = .{ .float = -3 }, .n = 1 }, .expected = &.{.{ .seed = "-3", .answer = "negative" }} },
        .{ .program = &configured, .config = &.{ .seed = .{ .integer = std.math.maxInt(i64) }, .n = 2 }, .expected = &.{ .{ .seed = "9223372036854775807", .answer = "max" }, .{ .seed = "9223372036854775808", .answer = "wide" } } },
    };
    const physical_count = 8;
    for ([_]bool{ false, true }) |responses| {
        var provider: dszig.lm.OpenAI = .{
            .client = &client,
            .base_url = base_url,
            .model = "gpt-4o-mini",
            .api = if (responses) .responses else .chat_completions,
            .max_retries = 0,
            .timeout_ns = 2 * std.time.ns_per_s,
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = provider.lm(), .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = cases.len };
        defer cache.deinit();
        var ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .json_fallback = false };
        ctx.lm_options.seed = .{ .integer = 73 };
        var physical: usize = 0;
        for (cases) |case| {
            ctx.predict_config = case.config;
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, responses, case.expected });
            defer serving.cancel(io) catch {};
            try Fixture.call(case.program, ctx, case.expected);
            try serving.await(io);
            try Fixture.call(case.program, ctx, case.expected);
            for (case.expected) |row| {
                const recorded = &recording.calls.items[physical];
                physical += 1;
                if (responses and row.seed != null) {
                    try testing.expectEqual(@as(usize, 1), recorded.adaptations.len);
                    try testing.expectEqual(@as(usize, 1), recorded.adaptations[0].len);
                    const adaptation = recorded.adaptations[0][0];
                    try testing.expectEqualStrings("config.seed", adaptation.field);
                    try testing.expectEqual(.dropped, adaptation.action);
                    const asked = try adaptation.asked.toValue(arena);
                    switch (asked) {
                        .integer => try testing.expectEqual(try std.fmt.parseInt(i64, row.seed.?, 10), asked.integer),
                        .number_string => try testing.expectEqualStrings(row.seed.?, asked.number_string),
                        else => return error.InvalidSeedAdaptation,
                    }
                } else {
                    try testing.expectEqual(@as(usize, 0), recorded.adaptations.len);
                }
            }
        }
        try testing.expectEqual(@as(usize, physical_count), recording.calls.items.len);
        for ([_]dszig.Value{ .{ .bool = true }, .{ .float = 1.5 } }) |invalid| {
            ctx.predict_config = &.{ .seed = invalid, .n = 2 };
            try testing.expectError(error.LMFailure, configured.call(ctx, .{ .question = "q" }));
        }

        for (recording.calls.items) |*recorded| {
            const metadata = try dszig.lm.optionsValue(arena, recorded.options.?);
            const encoded = try std.json.Stringify.valueAlloc(arena, metadata, .{});
            const decoded = try std.json.parseFromSliceLeaky(dszig.Value, arena, encoded, .{});
            recorded.options = try dszig.lm.optionsFromValue(arena, decoded);
        }
        var used: [physical_count]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
        ctx.lm = replay.lm();
        var i = cases.len;
        while (i > 0) {
            i -= 1;
            ctx.predict_config = cases[i].config;
            try Fixture.call(cases[i].program, ctx, cases[i].expected);
        }
        try testing.expect(replay.exhausted());
        try testing.expectEqual(@as(usize, 0), replay.misses);
    }
}

test "Predict repetition penalties preserve override, cache and replay identity across OpenAI dialects" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Case = struct {
        config: ?*const dszig.PredictConfig = null,
        frequency: ?f64,
        presence: ?f64,
        answer: []const u8,
    };
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, responses: bool, case: Case) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings(if (responses) "/v1/responses" else "/v1/chat/completions", request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer parsed.deinit();
            inline for (.{ "frequency_penalty", "presence_penalty" }, .{ case.frequency, case.presence }) |name, expected| {
                const wire = parsed.value.object.get(name);
                if (!responses and expected != null) {
                    const number = wire orelse return error.MissingWirePenalty;
                    const actual: f64 = switch (number) {
                        .integer => @floatFromInt(number.integer),
                        .float => number.float,
                        else => return error.InvalidWirePenalty,
                    };
                    try testing.expectEqual(expected.?, actual);
                } else try testing.expect(wire == null);
            }
            var reply_buf: [512]u8 = undefined;
            const reply = if (responses)
                try std.fmt.bufPrint(&reply_buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{s}</answer>\"}}]}}]}}", .{case.answer})
            else
                try std.fmt.bufPrint(&reply_buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{case.answer});
            try request.respond(reply, .{ .keep_alive = false });
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()});
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .frequency_penalty = 0, .presence_penalty = 2 };
    const clear: ?f64 = null;
    const cases = [_]Case{
        .{ .frequency = 0.25, .presence = -0.5, .answer = "context" },
        .{ .frequency = 0, .presence = 2, .answer = "stored" },
        .{ .config = &.{ .frequency_penalty = -2 }, .frequency = -2, .presence = 2, .answer = "frequency" },
        .{ .config = &.{ .presence_penalty = clear }, .frequency = 0, .presence = null, .answer = "presence" },
        .{ .config = &.{ .frequency_penalty = clear, .presence_penalty = clear }, .frequency = null, .presence = null, .answer = "cleared" },
    };
    const inputs: Sig.Input = .{ .question = "q" };
    for ([_]bool{ false, true }) |responses| {
        var provider: dszig.lm.OpenAI = .{
            .client = &client,
            .base_url = base_url,
            .model = "gpt-4o-mini",
            .api = if (responses) .responses else .chat_completions,
            .max_retries = 0,
            .timeout_ns = 2 * std.time.ns_per_s,
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = provider.lm(), .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = cases.len };
        defer cache.deinit();
        var ctx: dszig.Context = .{
            .arena = arena,
            .lm = cache.lm(),
            .lm_options = .{ .frequency_penalty = 0.25, .presence_penalty = -0.5 },
            .json_fallback = false,
        };
        for (cases, 0..) |case, index| {
            const program = if (index == 0) &plain else &configured;
            ctx.predict_config = case.config;
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, responses, case });
            defer serving.cancel(io) catch {};
            try testing.expectEqualStrings(case.answer, (try program.call(ctx, inputs)).fields().answer);
            try serving.await(io);
            try testing.expectEqualStrings(case.answer, (try program.call(ctx, inputs)).fields().answer);
            const recorded = &recording.calls.items[index];
            var expected_drops: usize = 0;
            if (responses) {
                inline for (.{ "frequency_penalty", "presence_penalty" }, .{ case.frequency, case.presence }) |name, expected| {
                    if (expected) |number| {
                        if (recorded.adaptations.len == 0 or expected_drops >= recorded.adaptations[0].len) return error.MissingPenaltyDrop;
                        const drop = recorded.adaptations[0][expected_drops];
                        try testing.expectEqualStrings("config." ++ name, drop.field);
                        try testing.expectEqual(.dropped, drop.action);
                        const asked = try drop.asked.toValue(arena);
                        try testing.expectEqual(number, if (asked == .integer) @as(f64, @floatFromInt(asked.integer)) else asked.float);
                        expected_drops += 1;
                    }
                }
            }
            try testing.expectEqual(expected_drops, if (recorded.adaptations.len == 0) 0 else recorded.adaptations[0].len);
            const metadata = try dszig.lm.optionsValue(arena, recorded.options.?);
            const encoded = try std.json.Stringify.valueAlloc(arena, metadata, .{});
            const decoded = try std.json.parseFromSliceLeaky(dszig.Value, arena, encoded, .{});
            recorded.options = try dszig.lm.optionsFromValue(arena, decoded);
        }
        try testing.expectEqual(cases.len, recording.calls.items.len);
        var used: [cases.len]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
        ctx.lm = replay.lm();
        var index = cases.len;
        while (index != 0) {
            index -= 1;
            const case = cases[index];
            ctx.predict_config = case.config;
            const program = if (index == 0) &plain else &configured;
            try testing.expectEqualStrings(case.answer, (try program.call(ctx, inputs)).fields().answer);
        }
        try testing.expect(replay.exhausted());
        try testing.expectEqual(@as(usize, 0), replay.misses);

        ctx.lm = provider.lm();
        for ([_]f64{ 2.01, std.math.nan(f64), std.math.inf(f64) }) |invalid| {
            ctx.predict_config = &.{ .frequency_penalty = invalid };
            try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
        }
        if (responses) {
            provider.adaptations = .refuse;
            ctx.predict_config = &.{ .frequency_penalty = clear, .presence_penalty = 0 };
            try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
        }
    }
}

test "typed Predict top_k reaches hosted wire dialects and preserves cache and replay identity" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Dialect = enum { anthropic, gemini, chat, responses };
    const Case = struct {
        config: ?*const dszig.PredictConfig = null,
        cutoff: ?[]const u8,
        answer: []const u8,
    };
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, dialect: Dialect, row: Case) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            const path = switch (dialect) {
                .anthropic => "/v1/messages",
                .gemini => "/v1/models/gemini-3-flash:generateContent",
                .chat => "/v1/chat/completions",
                .responses => "/v1/responses",
            };
            try testing.expectEqualStrings(path, request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{ .parse_numbers = false });
            defer parsed.deinit();
            const wire = parsed.value.object;
            const cutoff = if (dialect == .gemini)
                (wire.get("generationConfig") orelse return error.MissingGenerationConfig).object.get("topK")
            else
                wire.get("top_k");
            if (dialect == .chat or dialect == .responses or row.cutoff == null) {
                try testing.expect(cutoff == null);
            } else {
                const number = cutoff orelse return error.MissingWireTopK;
                try testing.expectEqual(.number_string, std.meta.activeTag(number));
                try testing.expectEqualStrings(row.cutoff.?, number.number_string);
            }
            var buf: [512]u8 = undefined;
            const reply = switch (dialect) {
                .anthropic => try std.fmt.bufPrint(&buf, "{{\"content\":[{{\"type\":\"text\",\"text\":\"<answer>{s}</answer>\"}}],\"stop_reason\":\"end_turn\"}}", .{row.answer}),
                .gemini => try std.fmt.bufPrint(&buf, "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"<answer>{s}</answer>\"}}]}},\"finishReason\":\"STOP\"}}]}}", .{row.answer}),
                .chat => try std.fmt.bufPrint(&buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{row.answer}),
                .responses => try std.fmt.bufPrint(&buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{s}</answer>\"}}]}}]}}", .{row.answer}),
            };
            try request.respond(reply, .{ .keep_alive = false });
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()});
    const timeout = 2 * std.time.ns_per_s;
    var anthropic: dszig.lm.Anthropic = .{ .client = &client, .model = "claude-sonnet-4-5", .base_url = base_url, .max_tokens = 64, .max_retries = 0, .timeout_ns = timeout };
    var gemini: dszig.lm.Gemini = .{ .client = &client, .model = "gemini-3-flash", .base_url = base_url, .max_tokens = 64, .max_retries = 0, .timeout_ns = timeout };
    var openai: dszig.lm.OpenAI = .{ .client = &client, .model = "gpt-4o-mini", .base_url = base_url, .max_retries = 0, .timeout_ns = timeout };
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .top_k = .{ .integer = 7 } };
    const rows = [_]Case{
        .{ .cutoff = "3", .answer = "context" },
        .{ .cutoff = "7", .answer = "stored" },
        .{ .config = &.{ .top_k = .{ .float = 1 } }, .cutoff = "1", .answer = "call" },
        .{ .config = &.{ .top_k = @as(dszig.Value, .null) }, .cutoff = null, .answer = "cleared" },
        .{ .config = &.{ .top_k = .{ .number_string = "9223372036854775808" } }, .cutoff = "9223372036854775808", .answer = "wide" },
    };
    const inputs: Sig.Input = .{ .question = "q" };
    for ([_]Dialect{ .anthropic, .gemini, .chat, .responses }) |dialect| {
        const drops = dialect == .chat or dialect == .responses;
        openai.api = if (dialect == .responses) .responses else .chat_completions;
        openai.adaptations = .note;
        const lm = switch (dialect) {
            .anthropic => anthropic.lm(),
            .gemini => gemini.lm(),
            .chat, .responses => openai.lm(),
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = lm, .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = rows.len };
        defer cache.deinit();
        var ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .lm_options = .{ .top_k = .{ .integer = 3 } }, .json_fallback = false };
        for (rows, 0..) |row, i| {
            const program = if (i == 0) &plain else &configured;
            ctx.predict_config = row.config;
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, dialect, row });
            defer serving.cancel(io) catch {};
            try testing.expectEqualStrings(row.answer, (try program.call(ctx, inputs)).fields().answer);
            try serving.await(io);
            try testing.expectEqualStrings(row.answer, (try program.call(ctx, inputs)).fields().answer);
            try testing.expectEqual(i + 1, recording.calls.items.len);
            const recorded = &recording.calls.items[i];
            if (drops and row.cutoff != null) {
                try testing.expectEqual(@as(usize, 1), recorded.adaptations.len);
                try testing.expectEqual(@as(usize, 1), recorded.adaptations[0].len);
                const adaptation = recorded.adaptations[0][0];
                try testing.expectEqualStrings("config.top_k", adaptation.field);
                try testing.expectEqual(.dropped, adaptation.action);
                const asked = try adaptation.asked.toValue(arena);
                try testing.expectEqualStrings(row.cutoff.?, try std.json.Stringify.valueAlloc(arena, asked, .{}));
            } else try testing.expectEqual(@as(usize, 0), recorded.adaptations.len);
            const metadata = try dszig.lm.optionsValue(arena, recorded.options.?);
            const encoded = try std.json.Stringify.valueAlloc(arena, metadata, .{});
            recorded.options = try dszig.lm.optionsFromValue(arena, try std.json.parseFromSliceLeaky(dszig.Value, arena, encoded, .{}));
        }
        var used: [rows.len]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
        ctx.lm = replay.lm();
        var i = rows.len;
        while (i > 0) {
            i -= 1;
            ctx.predict_config = rows[i].config;
            const program = if (i == 0) &plain else &configured;
            try testing.expectEqualStrings(rows[i].answer, (try program.call(ctx, inputs)).fields().answer);
        }
        try testing.expect(replay.exhausted());
        try testing.expectEqual(@as(usize, 0), replay.misses);
        ctx.lm = recording.lm();
        for ([_]dszig.Value{ .{ .integer = 0 }, .{ .integer = -1 }, .{ .float = 1.5 }, .{ .bool = true } }) |invalid| {
            ctx.predict_config = &.{ .top_k = invalid };
            try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
        }
        if (drops) {
            openai.adaptations = .refuse;
            ctx.predict_config = null;
            try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
        }
        try testing.expectEqual(rows.len, recording.calls.items.len);
    }
}

test "typed Predict user attribution follows provider policy and separates cache and replay" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Dialect = enum { chat, chat_compat, responses, anthropic, gemini };
    const Row = struct {
        user: ?[]const u8,
        answer: []const u8,
        override: ?*const dszig.PredictConfig = null,
    };
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, dialect: Dialect, row: Row) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings(switch (dialect) {
                .chat, .chat_compat => "/v1/chat/completions",
                .responses => "/v1/responses",
                .anthropic => "/v1/messages",
                .gemini => "/v1/models/gemini-3-flash:generateContent",
            }, request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer parsed.deinit();
            var buf: [512]u8 = undefined;
            const reply = switch (dialect) {
                .chat, .chat_compat => try std.fmt.bufPrint(&buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{row.answer}),
                .responses => try std.fmt.bufPrint(&buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{s}</answer>\"}}]}}]}}", .{row.answer}),
                .anthropic => try std.fmt.bufPrint(&buf, "{{\"content\":[{{\"type\":\"text\",\"text\":\"<answer>{s}</answer>\"}}],\"stop_reason\":\"end_turn\"}}", .{row.answer}),
                .gemini => try std.fmt.bufPrint(&buf, "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"<answer>{s}</answer>\"}}]}},\"finishReason\":\"STOP\"}}]}}", .{row.answer}),
            };
            try request.respond(reply, .{ .keep_alive = false });
            const field: ?[]const u8 = switch (dialect) {
                .chat => "user",
                .chat_compat => "user_id",
                .responses => "safety_identifier",
                .anthropic => "metadata",
                .gemini => null,
            };
            const wire = parsed.value.object;
            for ([_][]const u8{ "user", "user_id", "safety_identifier", "metadata" }) |key| {
                const actual = wire.get(key);
                if (row.user != null and field != null and std.mem.eql(u8, key, field.?)) {
                    const value = actual orelse return error.MissingAttribution;
                    const user = if (dialect == .anthropic) blk: {
                        try testing.expectEqual(.object, std.meta.activeTag(value));
                        break :blk value.object.get("user_id") orelse return error.MissingAttribution;
                    } else value;
                    try testing.expectEqual(.string, std.meta.activeTag(user));
                    try testing.expectEqualStrings(row.user.?, user.string);
                } else try testing.expect(actual == null);
            }
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()});
    const timeout = 2 * std.time.ns_per_s;
    var anthropic: dszig.lm.Anthropic = .{ .client = &client, .model = "claude-sonnet-4-5", .base_url = base_url, .max_tokens = 64, .max_retries = 0, .timeout_ns = timeout };
    var gemini: dszig.lm.Gemini = .{ .client = &client, .model = "gemini-3-flash", .base_url = base_url, .max_tokens = 64, .max_retries = 0, .timeout_ns = timeout };
    var openai: dszig.lm.OpenAI = .{ .client = &client, .model = "gpt-4o-mini", .base_url = base_url, .max_retries = 0, .timeout_ns = timeout };
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .user_id = "actor-stored" };
    const rows = [_]Row{
        .{ .user = "actor-context", .answer = "context" },
        .{ .user = "actor-stored", .answer = "stored" },
        .{ .user = "actor-β", .answer = "override", .override = &.{ .user_id = "actor-β" } },
        .{ .user = null, .answer = "cleared", .override = &.{ .user_id = @as(?[]const u8, null) } },
    };
    const inputs: Sig.Input = .{ .question = "q" };
    for (std.enums.values(Dialect)) |dialect| {
        openai.api = if (dialect == .responses) .responses else .chat_completions;
        openai.chat_compat = if (dialect == .chat_compat) .{ .user_field = .user_id } else null;
        gemini.adaptations = .note;
        const lm = switch (dialect) {
            .chat, .chat_compat, .responses => openai.lm(),
            .anthropic => anthropic.lm(),
            .gemini => gemini.lm(),
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = lm, .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = rows.len };
        defer cache.deinit();
        var ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .lm_options = .{ .user_id = "actor-context" }, .json_fallback = false };
        for (rows, 0..) |row, i| {
            const program = if (i == 0) &plain else &configured;
            ctx.predict_config = row.override;
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, dialect, row });
            defer serving.cancel(io) catch {};
            try testing.expectEqualStrings(row.answer, (try program.call(ctx, inputs)).fields().answer);
            try serving.await(io);
            // No server is serving this call. Different users must not share a cached answer.
            try testing.expectEqualStrings(row.answer, (try program.call(ctx, inputs)).fields().answer);
            const recorded = &recording.calls.items[i];
            if (dialect == .gemini and row.user != null) {
                try testing.expectEqual(@as(usize, 1), recorded.adaptations[0].len);
                const adaptation = recorded.adaptations[0][0];
                try testing.expectEqualStrings("config.user_id", adaptation.field);
                try testing.expectEqual(.dropped, adaptation.action);
                const asked = try adaptation.asked.toValue(arena);
                try testing.expectEqualStrings(row.user.?, asked.string);
            } else try testing.expectEqual(@as(usize, 0), recorded.adaptations.len);
            const metadata = try dszig.lm.optionsValue(arena, recorded.options.?);
            const encoded = try std.json.Stringify.valueAlloc(arena, metadata, .{});
            recorded.options = try dszig.lm.optionsFromValue(arena, try std.json.parseFromSliceLeaky(dszig.Value, arena, encoded, .{}));
        }
        var used: [rows.len]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
        ctx.lm = replay.lm();
        var i = rows.len;
        while (i > 0) {
            i -= 1;
            ctx.predict_config = rows[i].override;
            const program = if (i == 0) &plain else &configured;
            try testing.expectEqualStrings(rows[i].answer, (try program.call(ctx, inputs)).fields().answer);
        }
        try testing.expect(replay.exhausted());

        ctx.lm = lm;
        ctx.predict_config = &.{ .user_id = "" };
        try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
        if (dialect == .gemini) {
            gemini.adaptations = .refuse;
            ctx.predict_config = null;
            try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
        }
    }
}

test "typed Predict service tiers reach generation providers and separate cache and replay" {
    const Text = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Decision = dszig.Signature(struct {
        pub const Input = Text.Input;
        pub const Output = struct { answer: dszig.Noul };
        pub const descriptions = .{ .answer = "Is the answer affirmative?" };
    });
    const Dialect = enum { chat, responses, anthropic, gemini, typesafe };
    const Row = struct {
        tier: ?[]const u8,
        answer: u8,
        override: ?*const dszig.PredictConfig = null,
    };
    const rows = [_]Row{
        .{ .tier = "standard", .answer = 6 },
        .{ .tier = "priority", .answer = 7 },
        .{ .tier = "future-tier", .answer = 8, .override = &.{ .service_tier = "future-tier" } },
        .{ .tier = null, .answer = 9, .override = &.{ .service_tier = @as(?[]const u8, null) } },
    };
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, dialect: Dialect, row: Row) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings(switch (dialect) {
                .chat => "/v1/chat/completions",
                .responses => "/v1/responses",
                .anthropic => "/v1/messages",
                .gemini => "/v1/models/gemini-3-flash:generateContent",
                .typesafe => "/v1/systemone",
            }, request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer parsed.deinit();
            var buf: [512]u8 = undefined;
            const reply = switch (dialect) {
                .chat => try std.fmt.bufPrint(&buf, "{{\"model\":\"gpt-4o-mini\",\"choices\":[{{\"message\":{{\"content\":\"<answer>{d}</answer>\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":2,\"completion_tokens\":1}}{s}}}", .{
                    row.answer, if (row.answer == 8) ",\"service_tier\":\"default\"" else "",
                }),
                .responses => try std.fmt.bufPrint(&buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{d}</answer>\"}}]}}]}}", .{row.answer}),
                .anthropic => try std.fmt.bufPrint(&buf, "{{\"content\":[{{\"type\":\"text\",\"text\":\"<answer>{d}</answer>\"}}],\"stop_reason\":\"end_turn\"}}", .{row.answer}),
                .gemini => try std.fmt.bufPrint(&buf, "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"<answer>{d}</answer>\"}}]}},\"finishReason\":\"STOP\"}}]}}", .{row.answer}),
                .typesafe => try std.fmt.bufPrint(&buf, "{{\"answers\":{{\"answer\":{{\"type\":\"noul\",\"noul\":0.{d}}}}}}}", .{row.answer}),
            };
            try request.respond(reply, .{ .keep_alive = false });
            const wire = parsed.value.object;
            const key = if (dialect == .gemini) "serviceTier" else "service_tier";
            const other = if (dialect == .gemini) "service_tier" else "serviceTier";
            if (row.tier != null and dialect != .typesafe) {
                const tier = wire.get(key) orelse return error.MissingServiceTier;
                try testing.expectEqual(.string, std.meta.activeTag(tier));
                try testing.expectEqualStrings(row.tier.?, tier.string);
            } else try testing.expect(!wire.contains(key));
            try testing.expect(!wire.contains(other));
            if (dialect == .gemini and wire.get("generationConfig") != null)
                try testing.expect(!wire.get("generationConfig").?.object.contains("serviceTier"));
        }

        fn answer(ctx: dszig.Context, dialect: Dialect, i: usize) !u8 {
            if (dialect == .typesafe) {
                var program: dszig.Predict(Decision) = .{};
                if (i != 0) program.state.config = &.{ .service_tier = "priority" };
                const result = (try program.call(ctx, .{ .question = "q" })).fields().answer;
                try testing.expect(result.value);
                const probability = result.probability orelse return error.MissingProbability;
                return @intFromFloat(@round(probability * 10));
            }
            var program: dszig.Predict(Text) = .{};
            if (i != 0) program.state.config = &.{ .service_tier = "priority" };
            return std.fmt.parseInt(u8, (try program.call(ctx, .{ .question = "q" })).fields().answer, 10);
        }

        fn headers(ptr: *anyopaque, _: std.mem.Allocator, original: []const std.http.Header) anyerror![]const std.http.Header {
            const calls: *usize = @ptrCast(@alignCast(ptr));
            calls.* += 1;
            return original;
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    var billing = std.heap.ArenaAllocator.init(testing.allocator);
    defer billing.deinit();
    const arena = scratch.allocator();
    const catalogue = try dszig.model_prices.Catalogue.init(arena,
        \\{"openai/gpt-4o-mini":{"litellm_provider":"openai","input_cost_per_token":1,
        \\"output_cost_per_token":2,"input_cost_per_token_priority":4,"output_cost_per_token_priority":8}}
    , .{});
    const root = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}", .{listener.socket.address.getPort()});
    const base = try std.fmt.allocPrint(arena, "{s}/v1", .{root});
    const timeout = 2 * std.time.ns_per_s;
    var header_calls: usize = 0;
    var anthropic: dszig.lm.Anthropic = .{ .client = &client, .model = "claude-sonnet-4-5", .base_url = base, .max_tokens = 64, .max_retries = 0, .timeout_ns = timeout };
    var gemini: dszig.lm.Gemini = .{ .client = &client, .model = "gemini-3-flash", .base_url = base, .max_retries = 0, .timeout_ns = timeout };
    var openai: dszig.lm.OpenAI = .{ .client = &client, .model = "gpt-4o-mini", .base_url = base, .max_retries = 0, .timeout_ns = timeout };
    var typesafe: dszig.lm.TypeSafe = .{ .client = &client, .model = "fixture", .base_url = root, .max_retries = 0, .timeout_ns = timeout, .header_provider = .{ .context = &header_calls, .prepare = Fixture.headers } };

    for (std.enums.values(Dialect)) |dialect| {
        openai.api = if (dialect == .responses) .responses else .chat_completions;
        var lm = switch (dialect) {
            .chat, .responses => openai.lm(),
            .anthropic => anthropic.lm(),
            .gemini => gemini.lm(),
            .typesafe => typesafe.lm(),
        };
        if (dialect == .chat)
            lm.pricing = .{ .catalogue = &catalogue, .arena = billing.allocator(), .provider = "openai" };
        var recording: dszig.lm.RecordingLM = .{ .inner = lm, .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = rows.len };
        defer cache.deinit();
        var ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .lm_options = .{ .service_tier = "standard" }, .json_fallback = false };
        for (rows, 0..) |row, i| {
            ctx.predict_config = row.override;
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, dialect, row });
            defer serving.cancel(io) catch {};
            try testing.expectEqual(row.answer, try Fixture.answer(ctx, dialect, i));
            try serving.await(io);
            try testing.expectEqual(row.answer, try Fixture.answer(ctx, dialect, i));
            const recorded = &recording.calls.items[i];
            if (dialect == .typesafe and row.tier != null) {
                const adaptation = recorded.adaptations[0][0];
                try testing.expectEqualStrings("config.service_tier", adaptation.field);
                try testing.expectEqual(.dropped, adaptation.action);
                try testing.expectEqualStrings(row.tier.?, (try adaptation.asked.toValue(arena)).string);
            } else try testing.expectEqual(@as(usize, 0), recorded.adaptations.len);
            if (dialect == .chat) {
                const cost = recorded.billing.?.cost;
                try testing.expectEqual(@as(f64, if (i == 1) 16 else 4), cost.float);
            }
            const metadata = try dszig.lm.optionsValue(arena, recorded.options.?);
            const encoded = try std.json.Stringify.valueAlloc(arena, metadata, .{});
            recorded.options = try dszig.lm.optionsFromValue(arena, try std.json.parseFromSliceLeaky(dszig.Value, arena, encoded, .{}));
        }
        var used: [rows.len]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used, .supports_decision_requests = lm.supports_decision_requests };
        ctx.lm = replay.lm();
        var i = rows.len;
        while (i > 0) {
            i -= 1;
            ctx.predict_config = rows[i].override;
            try testing.expectEqual(rows[i].answer, try Fixture.answer(ctx, dialect, i));
        }
        try testing.expect(replay.exhausted());
        ctx.lm = lm;
        ctx.predict_config = &.{ .service_tier = "" };
        try testing.expectError(error.LMFailure, Fixture.answer(ctx, dialect, 1));
        if (dialect == .typesafe) {
            typesafe.adaptations = .refuse;
            ctx.predict_config = null;
            try testing.expectError(error.LMFailure, Fixture.answer(ctx, dialect, 1));
            try testing.expectEqual(rows.len, header_calls);
        }
    }
}

test "typed Predict store follows provider policy over HTTP and distinguishes cache and replay requests" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Store = enum { disabled, enabled, omitted };
    const Dialect = enum { chat, responses, anthropic, gemini };
    const Fixture = struct {
        fn prepare(context: *anyopaque, _: std.mem.Allocator, headers: []const std.http.Header) anyerror![]const std.http.Header {
            const calls: *usize = @ptrCast(@alignCast(context));
            calls.* += 1;
            return headers;
        }

        fn serve(io: std.Io, listener: *std.Io.net.Server, dialect: Dialect, expected: Store) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings(switch (dialect) {
                .chat => "/v1/chat/completions",
                .responses => "/v1/responses",
                .anthropic => "/v1/messages",
                .gemini => "/v1/models/gemini-3-flash:generateContent",
            }, request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer parsed.deinit();
            const store = parsed.value.object.get("store");
            var buf: [256]u8 = undefined;
            const reply = switch (dialect) {
                .chat => try std.fmt.bufPrint(&buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{@tagName(expected)}),
                .responses => try std.fmt.bufPrint(&buf, "{{\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"text\":\"<answer>{s}</answer>\"}}]}}]}}", .{@tagName(expected)}),
                .anthropic => try std.fmt.bufPrint(&buf, "{{\"content\":[{{\"type\":\"text\",\"text\":\"<answer>{s}</answer>\"}}],\"stop_reason\":\"end_turn\"}}", .{@tagName(expected)}),
                .gemini => try std.fmt.bufPrint(&buf, "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"<answer>{s}</answer>\"}}]}},\"finishReason\":\"STOP\"}}]}}", .{@tagName(expected)}),
            };
            try request.respond(reply, .{ .keep_alive = false });
            const wire_store: ?std.json.Value = if (dialect == .anthropic or expected == .omitted)
                null
            else
                .{ .bool = expected == .enabled };
            try testing.expectEqual(wire_store, store);
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()});
    const timeout = 2 * std.time.ns_per_s;
    var openai: dszig.lm.OpenAI = .{ .client = &client, .model = "gpt-4o-mini", .base_url = base_url, .max_retries = 0, .timeout_ns = timeout };
    var anthropic: dszig.lm.Anthropic = .{ .client = &client, .model = "claude-sonnet-4-5", .base_url = base_url, .max_tokens = 64, .max_retries = 0, .timeout_ns = timeout };
    var gemini: dszig.lm.Gemini = .{ .client = &client, .model = "gemini-3-flash", .base_url = base_url, .max_retries = 0, .timeout_ns = timeout };
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .store = true };
    const Row = struct {
        program: *const dszig.Predict(Sig),
        override: ?*const dszig.PredictConfig = null,
        expected: Store,
    };
    const rows = [_]Row{
        .{ .program = &plain, .expected = .disabled },
        .{ .program = &configured, .expected = .enabled },
        .{ .program = &configured, .override = &.{ .store = @as(?bool, null) }, .expected = .omitted },
    };
    const inputs: Sig.Input = .{ .question = "q" };
    for (std.enums.values(Dialect)) |dialect| {
        openai.api = if (dialect == .responses) .responses else .chat_completions;
        const provider = switch (dialect) {
            .chat, .responses => openai.lm(),
            .anthropic => anthropic.lm(),
            .gemini => gemini.lm(),
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = provider, .arena = arena };
        var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = rows.len };
        defer cache.deinit();
        var ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .lm_options = .{ .store = false }, .json_fallback = false };
        for (rows) |row| {
            ctx.predict_config = row.override;
            errdefer std.debug.print("store loopback failed ({s}, {s})\n", .{ @tagName(dialect), @tagName(row.expected) });
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, dialect, row.expected });
            defer serving.cancel(io) catch {};
            try testing.expectEqualStrings(@tagName(row.expected), (try row.program.call(ctx, inputs)).fields().answer);
            try serving.await(io);
            const recorded = recording.calls.items[recording.calls.items.len - 1];
            if (dialect == .anthropic and row.expected != .omitted) {
                try testing.expectEqual(@as(usize, 1), recorded.adaptations.len);
                try testing.expectEqual(@as(usize, 1), recorded.adaptations[0].len);
                const adaptation = recorded.adaptations[0][0];
                try testing.expectEqualStrings("config.store", adaptation.field);
                const enabled = row.expected == .enabled;
                const action: @TypeOf(adaptation.action) = if (enabled) .dropped else .satisfied;
                try testing.expectEqual(action, adaptation.action);
                try testing.expectEqual(dszig.Value{ .bool = enabled }, try adaptation.asked.toValue(arena));
                try testing.expectEqual(dszig.Value.null, try adaptation.applied.toValue(arena));
            } else try testing.expectEqual(@as(usize, 0), recorded.adaptations.len);
        }
        // No fixture serves these calls: they must hit distinct cache entries.
        for (rows) |row| {
            ctx.predict_config = row.override;
            errdefer std.debug.print("store cache failed ({s}, {s})\n", .{ @tagName(dialect), @tagName(row.expected) });
            try testing.expectEqualStrings(@tagName(row.expected), (try row.program.call(ctx, inputs)).fields().answer);
        }
        for (recording.calls.items) |*recorded| {
            const metadata = try dszig.lm.optionsValue(arena, recorded.options.?);
            const encoded = try std.json.Stringify.valueAlloc(arena, metadata, .{});
            recorded.options = try dszig.lm.optionsFromValue(arena, try std.json.parseFromSliceLeaky(dszig.Value, arena, encoded, .{}));
        }
        var used: [rows.len]bool = @splat(false);
        var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
        ctx.lm = replay.lm();
        var i = rows.len;
        while (i > 0) {
            i -= 1;
            const row = rows[i];
            ctx.predict_config = row.override;
            errdefer std.debug.print("store replay failed ({s}, {s})\n", .{ @tagName(dialect), @tagName(row.expected) });
            try testing.expectEqualStrings(@tagName(row.expected), (try row.program.call(ctx, inputs)).fields().answer);
        }
        try testing.expect(replay.exhausted());
    }

    // Bypass the cache: policy changes are checked on real provider calls.
    anthropic.adaptations = .refuse;
    errdefer std.debug.print("store Anthropic refuse-policy check failed\n", .{});
    var header_calls: usize = 0;
    anthropic.header_provider = .{ .context = &header_calls, .prepare = Fixture.prepare };
    var strict: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = anthropic.lm(), .json_fallback = false };
    try testing.expectError(error.LMFailure, configured.call(strict, inputs));
    try testing.expectEqual(@as(usize, 0), header_calls);
    strict.predict_config = &.{ .store = false };
    var serving = try io.concurrent(Fixture.serve, .{ io, &listener, .anthropic, .disabled });
    defer serving.cancel(io) catch {};
    try testing.expectEqualStrings("disabled", (try configured.call(strict, inputs)).fields().answer);
    try serving.await(io);
    try testing.expectEqual(@as(usize, 1), header_calls);
}

test "typed Predict publishes Anthropic streaming store adaptations and refuses before transport" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Fixture = struct {
        const Event = dszig.lm.Anthropic.StreamEvent;
        const Sink = dszig.lm.Anthropic.StreamSink;
        const Answer = "<answer>streamed</answer>";
        const Observation = struct {
            arena: std.mem.Allocator,
            expected: ?bool,
            next: usize = 0,
            failure: ?anyerror = null,

            fn observe(ptr: *anyopaque, event: Event) Sink.Action {
                const self: *@This() = @ptrCast(@alignCast(ptr));
                self.check(event) catch |err| {
                    self.failure = err;
                    return .stop;
                };
                self.next += 1;
                return .proceed;
            }

            fn check(self: @This(), event: Event) !void {
                switch (self.next) {
                    0 => {
                        try testing.expect(event == .start);
                        const records = event.start.adaptations;
                        if (self.expected) |enabled| {
                            try testing.expectEqual(@as(usize, 1), records.len);
                            const record = records[0];
                            try testing.expectEqualStrings("config.store", record.field);
                            const action: @TypeOf(record.action) = if (enabled) .dropped else .satisfied;
                            try testing.expectEqual(action, record.action);
                            try testing.expectEqual(dszig.Value{ .bool = enabled }, try record.asked.toValue(self.arena));
                            try testing.expectEqual(dszig.Value.null, try record.applied.toValue(self.arena));
                        } else try testing.expectEqual(@as(usize, 0), records.len);
                    },
                    1 => {
                        try testing.expect(event == .delta and event.delta.data == .text);
                        try testing.expectEqualStrings(Answer, event.delta.data.text.text);
                    },
                    2 => {
                        try testing.expect(event == .end);
                        try testing.expectEqual(.stop, event.end.finish_reason);
                    },
                    else => return error.UnexpectedStreamEvent,
                }
            }
        };

        fn prepare(ptr: *anyopaque, _: std.mem.Allocator, headers: []const std.http.Header) anyerror![]const std.http.Header {
            const count: *usize = @ptrCast(@alignCast(ptr));
            count.* += 1;
            return headers;
        }

        fn serve(io: std.Io, listener: *std.Io.net.Server) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings("/v1/messages", request.head.target);
            var body_buffer: [4096]u8 = undefined;
            const reader = try request.readerExpectContinue(&body_buffer);
            const body = try reader.allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const wire = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer wire.deinit();
            try testing.expectEqual(std.json.Value{ .bool = true }, wire.value.object.get("stream") orelse return error.NotStreaming);
            try testing.expect(!wire.value.object.contains("store"));
            try request.respond(
                \\data: {"type":"message_start","message":{"id":"msg_store","model":"claude-sonnet-4-5","content":[]}}
                \\
                \\data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"<answer>streamed</answer>"}}
                \\
                \\data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}
                \\
                \\data: {"type":"message_stop"}
                \\
                \\
            , .{ .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "text/event-stream" }} });
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var header_calls: usize = 0;
    var provider: dszig.lm.Anthropic = .{
        .client = &client,
        .model = "claude-sonnet-4-5",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()}),
        .max_tokens = 64,
        .max_retries = 0,
        .timeout_ns = 2 * std.time.ns_per_s,
        .header_provider = .{ .context = &header_calls, .prepare = Fixture.prepare },
    };
    var configured: dszig.Predict(Sig) = .{};
    configured.state.config = &.{ .store = true };
    const plain: dszig.Predict(Sig) = .{};
    const inputs: Sig.Input = .{ .question = "q" };
    var observation: Fixture.Observation = .{ .arena = arena, .expected = null };
    var ctx: dszig.Context = .{
        .arena = arena,
        .lm = provider.lm(),
        .lm_options = .{ .store = false },
        .stream_sink = .{ .context = &observation, .on_event = Fixture.Observation.observe },
        .json_fallback = false,
    };

    provider.adaptations = .refuse;
    try testing.expectError(error.LMFailure, configured.call(ctx, inputs));
    try testing.expectEqual(@as(usize, 0), header_calls);
    try testing.expectEqual(@as(usize, 0), observation.next);
    try testing.expect(observation.failure == null);

    const rows = [_]struct {
        policy: @TypeOf(provider.adaptations) = .note,
        stored: bool = true,
        override: ?*const dszig.PredictConfig = null,
        expected: ?bool,
    }{
        .{ .stored = false, .expected = false },
        .{ .expected = true },
        .{ .override = &.{ .store = @as(?bool, null) }, .expected = null },
        .{ .policy = .silent, .expected = null },
        .{ .policy = .refuse, .override = &.{ .store = false }, .expected = false },
    };
    for (rows, 1..) |row, calls| {
        provider.adaptations = row.policy;
        ctx.predict_config = row.override;
        observation = .{ .arena = arena, .expected = row.expected };
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener });
        defer serving.cancel(io) catch {};
        const program = if (row.stored) &configured else &plain;
        const result = program.call(ctx, inputs);
        if (observation.failure) |err| return err;
        try testing.expectEqualStrings("streamed", (try result).fields().answer);
        try serving.await(io);
        try testing.expectEqual(@as(usize, 3), observation.next);
        try testing.expectEqual(calls, header_calls);
    }
}

test "typed Predict stop overrides distinguish Chat cache and replay requests" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, expected: []const []const u8, answer: []const u8) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buf: [4096]u8 = undefined;
            var write_buf: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buf);
            var output = socket.writer(io, &write_buf);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings("/v1/chat/completions", request.head.target);
            var body_buf: [4096]u8 = undefined;
            const body = try (try request.readerExpectContinue(&body_buf)).allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer parsed.deinit();
            const stop = parsed.value.object.get("stop");
            if (expected.len == 0) {
                try testing.expect(stop == null);
            } else {
                const values = (stop orelse return error.MissingStop).array.items;
                try testing.expectEqual(expected.len, values.len);
                for (values, expected) |value, sequence| try testing.expectEqualStrings(sequence, value.string);
            }
            var reply_buf: [512]u8 = undefined;
            try request.respond(try std.fmt.bufPrint(&reply_buf, "{{\"choices\":[{{\"message\":{{\"content\":\"<answer>{s}</answer>\"}},\"finish_reason\":\"stop\"}}]}}", .{answer}), .{ .keep_alive = false });
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var provider: dszig.lm.OpenAI = .{
        .client = &client,
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()}),
        .model = "gpt-4o-mini",
        .max_retries = 0,
        .timeout_ns = 2 * std.time.ns_per_s,
    };
    var recording: dszig.lm.RecordingLM = .{ .inner = provider.lm(), .arena = arena };
    var cache: dszig.CachedLM = .{ .gpa = testing.allocator, .inner = recording.lm(), .max_entries = 8 };
    defer cache.deinit();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = cache.lm(), .lm_options = .{ .stop = &.{"context"} }, .json_fallback = false };
    const plain: dszig.Predict(Sig) = .{};
    var configured = plain;
    configured.state.config = &.{ .stop = &.{ "first", "second" } };
    const cases = [_]struct {
        program: *const dszig.Predict(Sig),
        config: ?*const dszig.PredictConfig = null,
        stop: []const []const u8,
        answer: []const u8,
    }{
        .{ .program = &configured, .config = &.{ .stop = &.{ "second", "first" } }, .stop = &.{ "second", "first" }, .answer = "call" },
        .{ .program = &configured, .stop = &.{ "first", "second" }, .answer = "stored" },
        .{ .program = &configured, .config = &.{ .stop = &.{} }, .stop = &.{}, .answer = "cleared" },
        .{ .program = &plain, .stop = &.{"context"}, .answer = "context" },
    };
    for (cases) |case| {
        var call_ctx = ctx;
        call_ctx.predict_config = case.config;
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, case.stop, case.answer });
        defer serving.cancel(io) catch {};
        try testing.expectEqualStrings(case.answer, (try case.program.call(call_ctx, .{ .question = "q" })).fields().answer);
        try serving.await(io);
        try testing.expectEqualStrings(case.answer, (try case.program.call(call_ctx, .{ .question = "q" })).fields().answer);
    }

    try testing.expectEqual(cases.len, recording.calls.items.len);
    for (recording.calls.items) |*call| {
        const serialized = try std.json.Stringify.valueAlloc(arena, try dszig.lm.optionsValue(arena, call.options.?), .{});
        const metadata = try std.json.parseFromSlice(dszig.Value, arena, serialized, .{ .allocate = .alloc_always });
        call.options = try dszig.lm.optionsFromValue(arena, metadata.value);
    }
    var used: [cases.len]bool = @splat(false);
    var replay: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &used };
    for (0..cases.len) |i| {
        const case = cases[cases.len - 1 - i];
        var call_ctx = ctx;
        call_ctx.lm = replay.lm();
        call_ctx.predict_config = case.config;
        try testing.expectEqualStrings(case.answer, (try case.program.call(call_ctx, .{ .question = "q" })).fields().answer);
    }
    try testing.expect(replay.exhausted());
}

test "typed Predict retains incomplete score coverage from a Responses stop cut" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Fixture = struct {
        const prefix = "<answer>four</answer>";
        const Case = enum { split_token, whole_token, unscored };

        fn serve(io: std.Io, listener: *std.Io.net.Server, scenario: Case) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try testing.expectEqualStrings("/v1/responses", request.head.target);
            var body_buffer: [4096]u8 = undefined;
            const reader = try request.readerExpectContinue(&body_buffer);
            const body = try reader.allocRemaining(testing.allocator, .limited(4096));
            defer testing.allocator.free(body);
            const wire = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
            defer wire.deinit();
            try testing.expect(wire.value.object.get("stream").?.bool);
            try testing.expectEqual(@as(i64, 0), wire.value.object.get("top_logprobs").?.integer);
            try testing.expect(!wire.value.object.contains("stop"));

            // No terminal event: only the stop cutter can successfully finish.
            try request.respond(switch (scenario) {
                .split_token =>
                \\data: {"type":"response.output_text.delta","delta":"<answer>four</answer>ST","logprobs":[{"token":"<answer>four</answer>ST","logprob":-0.25}]}
                \\
                \\data: {"type":"response.output_text.delta","delta":"OP","logprobs":[{"token":"OP","logprob":-0.25}]}
                \\
                \\
                ,
                .whole_token =>
                \\data: {"type":"response.output_text.delta","delta":"<answer>four</answer>STOP","logprobs":[{"token":"<answer>four</answer>","logprob":-0.25},{"token":"STOP","logprob":-0.25}]}
                \\
                \\
                ,
                .unscored =>
                \\data: {"type":"response.output_text.delta","delta":"<answer>four</answer>ST"}
                \\
                \\data: {"type":"response.output_text.delta","delta":"OP"}
                \\
                \\
                ,
            }, .{ .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "text/event-stream" }} });
        }
    };
    const io = testing.io;
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = io };
    defer client.deinit();
    var retained = std.heap.ArenaAllocator.init(testing.allocator);
    defer retained.deinit();
    const out = retained.allocator();
    var provider: dszig.lm.OpenAI = .{
        .client = &client,
        .model = "gpt-4o-mini",
        .api = .responses,
        .base_url = try std.fmt.allocPrint(out, "http://127.0.0.1:{d}/v1", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const program: dszig.Predict(Sig) = .{};
    var trace: dszig.Trace = .{ .arena = out };
    for ([_]Fixture.Case{ .split_token, .whole_token, .unscored }) |scenario| {
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, scenario });
        defer serving.cancel(io) catch {};
        const prediction = blk: {
            var scratch = std.heap.ArenaAllocator.init(testing.allocator);
            defer scratch.deinit();
            const result = try program.call(.{
                .arena = scratch.allocator(),
                .lm = provider.lm(),
                .lm_options = .{ .logprobs = .{ .integer = 0 }, .stop = &.{"never"} },
                .predict_config = &.{ .stop = &.{"STOP"} },
                .trace = &trace,
                .json_fallback = false,
            }, .{ .question = "Two plus two?" });
            break :blk try result.clone(out);
        };
        try serving.await(io);
        try testing.expectEqualStrings("four", prediction.fields().answer);
        const scores = prediction.completionLogprobs(0);
        const traced = trace.steps.items[trace.steps.items.len - 1].completionLogprobs(0);
        switch (scenario) {
            .split_token => {
                try testing.expect(scores != null);
                try testing.expect(!scores.?.complete);
                try testing.expect(!scores.?.reported);
                try testing.expectEqual(@as(usize, 0), scores.?.tokens.len);
            },
            .whole_token => {
                try testing.expect(scores.?.complete);
                try testing.expect(scores.?.reported);
                try testing.expectEqual(@as(usize, 1), scores.?.tokens.len);
                try testing.expectEqualStrings(Fixture.prefix, scores.?.tokens[0].token);
            },
            .unscored => try testing.expect(scores == null),
        }
        try testing.expectEqualDeep(scores, traced);
    }
}

test "typed Predict rejects provider scores that collide with a declared logprobs output" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { logprobs: []const u8 };
    });
    const Engine = struct {
        rows: []const ?[]const dszig.lm.TokenLogprob,
        index: usize = 0,

        fn complete(ptr: *anyopaque, arena: Allocator, _: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (self.index == self.rows.len) return error.LMFailure;
            const scores = self.rows[self.index];
            self.index += 1;
            const out = try arena.alloc(dszig.lm.Completion, 1);
            out[0] = .{ .text = "<logprobs>declared</logprobs>", .logprobs = scores };
            return out;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var trace: dszig.Trace = .{ .arena = arena };
    const score: dszig.lm.TokenLogprob = .{ .token = "a", .logprob = -0.25 };
    var engine: Engine = .{ .rows = &.{ null, &.{}, null, &.{score} } };
    var ctx: dszig.Context = .{
        .arena = arena,
        .lm = .{ .ptr = &engine, .vtable = &.{ .complete = Engine.complete } },
        .trace = &trace,
    };
    const program: dszig.Predict(Sig) = .{};
    const input: Sig.Input = .{ .question = "q" };
    const absent = try program.call(ctx, input);
    try testing.expectEqualStrings("declared", absent.fields().logprobs);
    try testing.expect(absent.completionLogprobs(0) == null);
    try testing.expectEqualStrings("declared", trace.steps.items[0].outputs.get("logprobs").?.string);

    try testing.expectError(error.ConflictingCompletionLogprobs, program.forward(ctx, input));
    ctx.lm_options.n = 2;
    try testing.expectError(error.ConflictingCompletionLogprobs, program.call(ctx, input));
    try testing.expectEqual(engine.rows.len, engine.index);
    try testing.expectEqual(@as(usize, 1), trace.steps.items.len);
}

test "Chat prediction slices inline headers in Python character units" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    for ([_][]const u8{ "", " ", "\u{2003}" }, [_][]const u8{ "OK", "] OK", "] OK" }) |prefix, answer| {
        const completion = try std.fmt.allocPrint(arena, "{s}[[ ## answer ## ]] OK", .{prefix});
        var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{completion} };
        const result = try program.forward(
            .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = scripted.lm(), .adapter = .chat, .json_fallback = false },
            .{ .question = "Status?" },
        );
        try testing.expectEqualStrings(answer, result.answer);
    }
}

test "empty predictor completions fail before applying output defaults" {
    const S = dszig.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: i64 = 9 };
    });
    const program: dszig.Predict(S) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    for ([_]dszig.Adapter{ .chat, .json, .xml }) |adapter| {
        var lm: dszig.lm.ScriptedLM = .{ .responses = &.{""} };
        var capture: dszig.predict.ParseFailure = .{};
        try testing.expectError(error.AdapterParse, program.forward(
            .{
                .arena = memory.allocator(),
                .lm = lm.lm(),
                .adapter = adapter,
                .json_fallback = false,
                .parse_failure = &capture,
            },
            .{},
        ));
        try testing.expect(capture.signature_class == program.state.signature_class);
        try testing.expect(capture.failure.parsed_result == null);
        try testing.expectEqualStrings("", capture.failure.completion_text);
    }
}

test "JSON prediction recovers the first balanced object after an unmatched prefix" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const opens = try arena.alloc(u8, 128 * 1024);
    @memset(opens, '{');

    // The full response is a valid array, not an object. The fallback regex
    // treats braces inside the first string as literal brace characters.
    const prefixed = try std.fmt.allocPrint(arena, "[\"{s}\",{{\"answer\":\"Paris\"}},{{\"answer\":\"later\"}}]", .{opens});
    const nested = "[{\"meta\":{\"answer\":\"inner\"},\"answer\":\"outer\"},{\"answer\":\"later\"}]";
    const unmatched = try std.fmt.allocPrint(arena, "[\"{s}\"]", .{opens});
    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{ prefixed, nested, unmatched } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const input: Sig.Input = .{ .question = "Where?" };

    try testing.expectEqualStrings("Paris", (try program.forward(ctx, input)).answer);
    try testing.expectEqualStrings("outer", (try program.forward(ctx, input)).answer);
    try testing.expectError(error.AdapterParse, program.forward(ctx, input));
}

test "JSON prediction repairs a long truncated string with mixed escapes" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const count = 32 * 1024;
    const prefix = "{\"answer\":\"";
    const encoded = "x\\n\\u007b\\u007d\\u005b\\u005d\\\\\\\\z";
    const decoded = "x\n{}[]\\\\z";
    const tail = "雪";
    const completion = try arena.alloc(u8, prefix.len + count * encoded.len + tail.len);
    @memcpy(completion[0..prefix.len], prefix);
    for (0..count) |i| @memcpy(completion[prefix.len + i * encoded.len ..][0..encoded.len], encoded);
    @memcpy(completion[completion.len - tail.len ..], tail);

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{completion} };
    const result = try program.forward(.{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json }, .{ .question = "Decode" });
    try testing.expectEqual(count * decoded.len + tail.len, result.answer.len);
    for (0..count) |i| try testing.expectEqualStrings(decoded, result.answer[i * decoded.len ..][0..decoded.len]);
    try testing.expectEqualStrings(tail, result.answer[count * decoded.len ..]);
}

test "JSON prediction repairs a truncated string with many closing braces" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const prefix = "{\"answer\":\"";
    const block = "}xy";
    const body = try arena.alloc(u8, 65536 * block.len + "prefix雪".len);
    @memcpy(body[0.."prefix".len], "prefix");
    for (0..65536) |i| @memcpy(body["prefix".len + i * block.len ..][0..block.len], block);
    @memcpy(body[body.len - "雪".len ..], "雪");
    const completion = try std.mem.concat(arena, u8, &.{ prefix, body });

    // Source Predict keeps a distant brace and respects escape parity. A
    // nearby brace or a later unescaped colon ends the unterminated value.
    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        prefix ++ "prefix}x\\:y}x\\\\}xy雪",
        prefix ++ "prefix}x}x}x",
        prefix ++ "prefix}xy}xy: tail",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const input: Sig.Input = .{ .question = "Decode" };
    for ([_][]const u8{ body, "prefix}x\\:y}x\\}xy雪", "prefix", "prefix" }) |expected| {
        try testing.expectEqualStrings(expected, (try program.forward(ctx, input)).answer);
    }
}

test "JSON prediction repairs a missing object close after a long quoted value" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const prefix = "{\"answer\":\"";
    const block = "}xy";
    const count = 32 * 1024;
    const head = "prefix\n";
    const body = try arena.alloc(u8, head.len + count * block.len + "雪".len);
    @memcpy(body[0..head.len], head);
    for (0..count) |i| @memcpy(body[head.len + i * block.len ..][0..block.len], block);
    @memcpy(body[body.len - "雪".len ..], "雪");
    const whitespace = try arena.alloc(u8, count);
    @memset(whitespace, ' ');
    // The escape keeps this out of the simple quoted-string fast path.
    const completion = try std.mem.concat(arena, u8, &.{ prefix, "prefix\\n", body[head.len..], "\"", whitespace });

    // DSPy keeps the braces if the quote is followed by EOF or `}`.
    // A later quoted token followed by a colon can instead end the value.
    const short = "prefix}xy}xy}xy}xy雪";
    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        prefix ++ short ++ "\"\u{2003}junk",
        prefix ++ short ++ "\"\u{2003}junk\"\u{2003}:",
        prefix ++ short ++ "\"\u{2003}junk\"\u{2003},",
        prefix ++ short ++ "\"\u{2003}}",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const input: Sig.Input = .{ .question = "Decode" };
    for ([_][]const u8{ body, short, "prefix", short ++ "\"\u{2003}junk\"", short }) |expected| {
        try testing.expectEqualStrings(expected, (try program.forward(ctx, input)).answer);
    }
}

test "JSON prediction repairs a quoted array item with many closing brackets" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const []const u8 };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const prefix = "{\"answer\":[\"";
    const block = "]xy";
    const count = 32 * 1024;
    const brackets = try arena.alloc(u8, count * block.len);
    for (0..count) |i| @memcpy(brackets[i * block.len ..][0..block.len], block);
    const expected = try std.mem.concat(arena, u8, &.{ "prefix\n", brackets, "雪" });
    // The escape bypasses the simple string fast path; both container closes
    // are missing, as in a completion truncated just after the quote.
    const completion = try std.mem.concat(arena, u8, &.{ prefix, "prefix\\n", brackets, "雪\"" });

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        prefix ++ "left]",
        prefix ++ "left]x\\\"",
        prefix ++ "left]x\\\"y]z\"",
        prefix ++ "left]x\\\\\"",
        prefix ++ "left]x\",\"right]z\"",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const input: Sig.Input = .{ .question = "Decode" };
    for ([_][]const []const u8{
        &.{expected},
        &.{"left"},
        &.{"left"},
        &.{"left]x\"y]z"},
        &.{"left]x\""},
        &.{ "left]x", "right]z" },
    }) |want| {
        const actual = (try program.forward(ctx, input)).answer;
        try testing.expectEqual(want.len, actual.len);
        for (want, actual) |w, a| try testing.expectEqualStrings(w, a);
    }
}

test "JSON prediction repairs escaped keys containing many colons" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: dszig.Dict(i64) };
    });
    const program: dszig.Predict(Sig) = .{};
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const block = ":xy";
    const count = 32 * 1024;
    const repeated = try arena.alloc(u8, count * block.len);
    for (0..count) |i| @memcpy(repeated[i * block.len ..][0..block.len], block);
    const key = try std.mem.concat(arena, u8, &.{ "prefix\n", repeated, "雪" });
    // An escaped key bypasses the simple string scan; both object closes are missing.
    const completion = try std.mem.concat(arena, u8, &.{ "{\"answer\":{\"prefix\\n", repeated, "雪\":42" });

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        "{\"answer\":{\"a:b\":1,\"c:d\":2",
        "{'answer':{'a:b':3,'c:d':4",
        "{\"answer\":{\"a:b\\\"c:d\":5",
        "{\"answer\":{\"a:b\" \"oops\" , \"tail\":6",
        "{\"answer\":{\"a:b:7",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const input: Sig.Input = .{ .question = "Decode" };
    for ([_]dszig.Dict(i64){
        .{ .keys = &.{key}, .values = &.{42} },
        .{ .keys = &.{ "a:b", "c:d" }, .values = &.{ 1, 2 } },
        .{ .keys = &.{ "a:b", "c:d" }, .values = &.{ 3, 4 } },
        .{ .keys = &.{"a:b\"c:d"}, .values = &.{5} },
    }) |want| {
        const actual = (try program.forward(ctx, input)).answer;
        try testing.expectEqual(want.keys.len, actual.keys.len);
        for (want.keys, want.values) |k, v| try testing.expectEqual(@as(?i64, v), actual.get(k));
    }
    try testing.expectError(error.AdapterParse, program.forward(ctx, input));
    try testing.expectError(error.AdapterParse, program.forward(ctx, input));
}

test "JSON prediction repairs keys with many opening smart quotes" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: dszig.Dict(i64) };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const block = ":x“";
    const repeated = try arena.alloc(u8, 32 * 1024 * block.len);
    var i: usize = 0;
    while (i < repeated.len) : (i += block.len) @memcpy(repeated[i..][0..block.len], block);
    const key = try std.mem.concat(arena, u8, &.{ "prefix", repeated, "雪" });
    const completion = try std.mem.concat(arena, u8, &.{ "{\"answer\":{“", key, "”:42" });

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        "{\"answer\":{“a:b“c:d\\”e:f“g”:5",
        "{\"answer\":{“a:b„inner“c”d:e“f”:9",
        "{\"answer\":{“a:b“c”:1,“d:e“f”:2",
        "{\"answer\":{“a:b“c:d:7",
        "{\"answer\":{“a:b“c:d\\\\”:4",
        "{\"answer\":{“a:b“c”\u{2003}}",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const input: Sig.Input = .{ .question = "Decode" };
    const program: dszig.Predict(Sig) = .{};
    for ([_]dszig.Dict(i64){
        .{ .keys = &.{key}, .values = &.{42} },
        .{ .keys = &.{"a:b“c:d”e:f“g"}, .values = &.{5} },
        .{ .keys = &.{"a:b„inner“c”d:e“f"}, .values = &.{9} },
        .{ .keys = &.{ "a:b“c", "d:e“f" }, .values = &.{ 1, 2 } },
    }) |want| {
        const actual = (try program.forward(ctx, input)).answer;
        try testing.expectEqual(want.keys.len, actual.keys.len);
        for (want.keys, want.values) |k, v| try testing.expectEqual(@as(?i64, v), actual.get(k));
    }
    for (0..3) |_| try testing.expectError(error.AdapterParse, program.forward(ctx, input));
}

test "JSON prediction repairs repeated bare quotes in regex character classes" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const count = 32 * 1024;
    const expected = try arena.alloc(u8, 2 * count + "[z]雪".len);
    expected[0] = '[';
    for (0..count) |i| @memcpy(expected[1 + 2 * i ..][0..2], "a\"");
    @memcpy(expected[1 + 2 * count ..], "z]雪");
    const completion = try std.mem.concat(arena, u8, &.{ "{\"answer\":\"", expected, "\"" });

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        "{\"answer\":\"[a\u{2003}\"b]\"",
        "{\"answer\":\"[a [b\"c]\"",
        "{\"answer\":\"[a\"b\\]c\"d]\"",
        "{\"answer\":\"[a\"b\\\\]c\"",
        "{\"answer\":\"[a\"b\"",
        "{\"answer\":\"[a\"b][c\"d]雪\"",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const program: dszig.Predict(Sig) = .{};
    for ([_][]const u8{
        expected,
        "[a\u{2003}\"b]\"",
        "[a [b\"c]",
        "[a\"b\\]c\"d]\"",
        "[a\"b\\]c",
        "[a\"b\"",
        "[a\"b][c\"d]雪",
    }) |want| {
        try testing.expectEqualStrings(want, (try program.forward(ctx, .{ .question = "Decode" })).answer);
    }
}

test "JSON prediction repairs embedded commas before a shared quoted suffix" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const count = 32 * 1024;
    const commas = try arena.alloc(u8, 2 * count);
    const spaces = try arena.alloc(u8, "\u{2003}".len * count);
    for (0..count) |i| {
        @memcpy(commas[2 * i ..][0..2], ",!");
        @memcpy(spaces[3 * i ..][0..3], "\u{2003}");
    }
    const completion = try std.mem.concat(arena, u8, &.{ "{\"answer\":\"seed\\t", commas, " \"note\"", spaces, "}" });
    const expected = try std.mem.concat(arena, u8, &.{ "seed\t", commas, " \"note\"" });

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        "{\"answer\":\"seed\\t,! \"next\"\u{2003}: 7}",
        "{\"answer\":\"seed\\t,! \"note\"\u{2003}",
        "{\"answer\":\"seed\\t,! \"n\\\"ote\"\u{2003}}",
        "{\"answer\":\"seed\\t,! \"note\"\u{2003},! 'detail'\u{2003}}",
        "{\"answer\":\"seed\\t,\"next\"\u{2003}: 7}",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const program: dszig.Predict(Sig) = .{};
    for ([_][]const u8{
        expected,
        "seed",
        "seed\t,! \"note\"",
        "seed\t,! ",
        "seed\t,! \"note\"\u{2003},! 'detail'",
        "seed",
    }) |want| {
        try testing.expectEqualStrings(want, (try program.forward(ctx, .{ .question = "Decode" })).answer);
    }
}

test "JSON prediction preserves repeated inline fences in a malformed value" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    const snippet = "}\n```txt\nx\n";
    const count = 32 * 1024;
    const fences = try arena.alloc(u8, snippet.len * count);
    for (0..count) |i| @memcpy(fences[i * snippet.len ..][0..snippet.len], snippet);
    const completion = try std.mem.concat(arena, u8, &.{ "{\"answer\":\"seed\\t", fences, "end\"\u{2003}}" });
    const expected = try std.mem.concat(arena, u8, &.{ "seed\t", fences, "end" });

    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        completion,
        "{\"answer\":\"seed\\t}\n``` wrapper",
        "{\"answer\":\"seed\\t}\n```txt\nx\\\" tail\"}",
        "{\"answer\":\"seed\\t}\n```[1] text\"}",
        "{\"answer\":\"seed\\t}\n```[1], \"extra\": 7}",
        "{\"answer\":\"seed\\t}\n```txt\n\"bare\"more\n}\n```txt\nend\"}",
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const program: dszig.Predict(Sig) = .{};
    for ([_][]const u8{
        expected,
        "seed",
        "seed\t}\n```txt\nx\" tail",
        "seed\t}\n```[1] text",
        "seed",
        "seed\t}\n```txt\n\"bare\"more\n}\n```txt\nend",
    }) |want| {
        try testing.expectEqualStrings(want, (try program.forward(ctx, .{ .question = "Decode" })).answer);
    }
}

test "typed JSON prediction follows DSPy nested integer string coercion" {
    const Sig = dszig.Signature(struct {
        pub const instructions = "Return the numbers.";
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { numbers: []const dszig.Integer };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const zeroes: [4301]u8 = @splat('0');
    const max_digits: [4300]u8 = @splat('1');
    const padded = try std.fmt.allocPrint(arena, "{{\"numbers\":[\"{s}42\",\"-001_000.00\",\"{s}{s}\"]}}", .{ &zeroes, &zeroes, &max_digits });
    const too_long = try std.fmt.allocPrint(arena, "{{\"numbers\":[\"{s}1\"]}}", .{&max_digits});
    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
        "{\"numbers\":[\"1.0_0\"]}",
        padded,
        too_long,
    } };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json };
    const program: dszig.Predict(Sig) = .{};
    try testing.expectError(error.AdapterParse, program.forward(ctx, .{ .question = "Malformed fraction" }));
    const numbers = (try program.forward(ctx, .{ .question = "Valid padded numbers" })).numbers;
    try testing.expectEqual(@as(usize, 3), numbers.len);
    try testing.expectEqual(@as(i64, 42), numbers[0].value.integer);
    try testing.expectEqual(@as(i64, -1000), numbers[1].value.integer);
    try testing.expectEqualStrings(&max_digits, numbers[2].value.number_string);
    try testing.expectError(error.AdapterParse, program.forward(ctx, .{ .question = "Too many significant digits" }));
}

test "JSON parse failure retains converted predecessors and raw invalid siblings" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var failure: dszig.adapter.ParseFailure = .{};
    const view: dszig.adapter.View = .{
        .instructions = "",
        .inputs = BootstrapTraceSignature.input_fields_list,
        .outputs = BootstrapTraceSignature.output_fields_list,
    };
    try testing.expectError(error.AdapterParse, dszig.Adapter.json.parseWithFailure(
        arena,
        view,
        "{\"second\":\"2\",\"first\":\"wrong\",\"third\":\"3\",\"extra\":0}",
        &failure,
    ));
    const partial = failure.parsed_result.?;
    try testing.expectEqual(@as(usize, 3), partial.count());
    for ([_][]const u8{ "second", "first", "third" }, partial.keys()) |expected, actual| {
        try testing.expectEqualStrings(expected, actual);
    }
    try testing.expectEqual(@as(i64, 2), partial.get("second").?.integer);
    try testing.expectEqualStrings("wrong", partial.get("first").?.string);
    try testing.expectEqualStrings("3", partial.get("third").?.string);
    // A successful parse must not leak this failed completion into a later call.
    _ = try dszig.Adapter.json.parseWithFailure(
        arena,
        view,
        "{\"first\":4,\"second\":5}",
        &failure,
    );
    try testing.expect(failure.parsed_result == null);
    try testing.expectEqualStrings("", failure.completion_text);
}

const BootstrapTraceSingle = struct {
    predict: dszig.Predict(BootstrapTraceSignature) = .{},
    pub const Input = struct { question: []const u8 };
    pub const Output = BootstrapTraceSignature.Output;

    pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) !Output {
        return self.predict.forward(ctx, .{ .token = input.question });
    }
};

const BootstrapTraceLM = struct {
    fn complete(
        _: *anyopaque,
        arena: Allocator,
        request: dszig.lm.Request,
    ) dszig.lm.Error![]const dszig.lm.Completion {
        const text = request.messages[request.messages.len - 1].content.text;
        const body = if (std.mem.indexOf(u8, text, "q0") != null)
            "{\"first\":1,\"second\":2}"
        else if (std.mem.indexOf(u8, text, "q1") != null)
            "{\"second\":7}"
        else if (std.mem.indexOf(u8, text, "q2") != null)
            "{\"first\":8,\"second\":9}"
        else
            "{\"second\":\"2\",\"first\":\"wrong\",\"third\":\"3\"}";
        const output = try arena.alloc(dszig.lm.Completion, 1);
        output[0] = .{ .text = try arena.dupe(u8, body) };
        return output;
    }

    fn lm(self: *@This()) dszig.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }
};

test "concurrent trace bootstrap retains ordered failures across joined scratch reuse" {
    const Pool = dszig.Parallel(dszig.bootstrap_trace.Run, .{
        .workers = 2,
        .tasks = 2,
        .attempts = 2,
        .failures = 2,
    });
    const Scratch = struct {
        allocators: [2]std.heap.FixedBufferAllocator,

        fn reset(ptr: *anyopaque, index: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.allocators[index].reset();
        }
    };
    var retained: std.heap.ArenaAllocator = .init(testing.allocator);
    defer retained.deinit();
    const out = retained.allocator();
    var pool: Pool = undefined;
    try pool.init(testing.io, .{});
    defer pool.deinit();
    var buffers: [2][131072]u8 = undefined;
    var scratch: Scratch = undefined;
    var models: [2]BootstrapTraceLM = .{ .{}, .{} };
    var contexts: [2]dszig.Context = undefined;
    for (&contexts, &scratch.allocators, &buffers, &models) |*ctx, *allocator, *buffer, *model| {
        allocator.* = .init(buffer);
        ctx.* = .{ .workspace = dszig.workspace.forTests(), .arena = allocator.allocator(), .lm = model.lm(), .adapter = .json };
    }
    const dataset = [_]dszig.Example{
        try dszig.Example.fromJson(out, "{\"question\":\"q0\"}", &.{"question"}),
        try dszig.Example.fromJson(out, "{\"question\":\"q1\"}", &.{"question"}),
        try dszig.Example.fromJson(out, "{\"question\":\"q2\"}", &.{"question"}),
        try dszig.Example.fromJson(out, "{\"question\":\"q3\"}", &.{"question"}),
    };
    var program: BootstrapTraceSingle = .{};
    const result = try dszig.bootstrap_trace.bootstrapTraceData(
        BootstrapTraceSingle,
        out,
        contexts[0],
        &program,
        &dataset,
        .{
            .metric = bootstrapTraceMetric,
            .failure_score = 0.5,
            .rollouts = .{ .runner = pool.runner(), .contexts = &contexts },
            .scratch_reset = .{ .ptr = &scratch, .reset = Scratch.reset },
        },
    );
    for (&buffers, scratch.allocators) |*buffer, allocator| {
        try testing.expectEqual(@as(usize, 0), allocator.end_index);
        @memset(buffer, 0xff);
    }
    try testing.expectEqual(@as(u32, 0), pool.scheduler.running_count);
    try testing.expectEqual(@as(usize, 4), result.len);
    for (result, [_]f64{ 0.1, 0, 0.8, 0.5 }, 0..) |row, score, index| {
        try testing.expectEqual(index, row.example_ind);
        try testing.expectEqual(@as(?f64, score), row.score);
    }
    try testing.expectEqual(@as(i64, 1), result[0].prediction.success.get("first").?.integer);
    try testing.expectEqual(@as(i64, 8), result[2].prediction.success.get("first").?.integer);
    try testing.expectEqualStrings("q0", result[0].trace[0].inputs.get("token").?.string);
    try testing.expectEqualStrings("q2", result[2].trace[0].inputs.get("token").?.string);
    try testing.expectEqualStrings("{\"second\":7}", result[1].prediction.failed.completion_text);
    try testing.expectEqualStrings("q1", result[1].trace[0].inputs.get("question").?.string);
    try testing.expectEqual(@as(?f64, 0.5), result[3].prediction.failed.format_reward);
}

test "trace metric failure reports successful evaluation before unpacking fails" {
    const Hooks = struct {
        call_id: u64 = 0,
        ended: bool = false,
        evaluation: ?dszig.optimize.Evaluation = null,

        fn start(ptr: *anyopaque, event: dszig.bootstrap_trace.EvaluateStart) !void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.call_id = event.call_id;
            return error.CallbackStartFailure;
        }

        fn end(ptr: *anyopaque, event: dszig.bootstrap_trace.EvaluateEnd) !void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.ended = event.call_id == self.call_id and event.err == null;
            if (event.result) |result| self.evaluation = result.evaluation;
            return error.CallbackEndFailure;
        }

        fn metric(
            _: Allocator,
            _: *const dszig.Example,
            _: ?*const dszig.optimize.RecordPrediction,
            _: ?*const dszig.Trace,
        ) dszig.value.ParseError!f64 {
            return error.ParseValue;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var lm: dszig.lm.ScriptedLM = .{ .responses = &.{"{\"first\":1,\"second\":2}"} };
    var program: BootstrapTraceSingle = .{};
    var hooks: Hooks = .{};
    var failure: ?dszig.bootstrap_trace.Unhandled = null;
    const example = try dszig.Example.fromJson(arena, "{\"question\":\"metric\"}", &.{"question"});
    try testing.expectError(error.UnpackPredictionAndTrace, dszig.bootstrap_trace.bootstrapTraceData(
        BootstrapTraceSingle,
        arena,
        .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm(), .adapter = .json },
        &program,
        &.{example},
        .{
            .metric = Hooks.metric,
            .failure_score = 0.5,
            .capture_crashes = true,
            .unhandled = &failure,
            .callbacks = &.{.{
                .ptr = &hooks,
                .on_evaluate_start = Hooks.start,
                .on_evaluate_end = Hooks.end,
            }},
        },
    ));
    try testing.expect(hooks.ended);
    try testing.expectEqual(@as(f64, 50), hooks.evaluation.?.score);
    try testing.expectEqual(@as(usize, 1), hooks.evaluation.?.errors);
    try testing.expectEqual(dszig.bootstrap_trace.Stage.metric, failure.?.stage);
    try testing.expectEqual(error.ParseValue, failure.?.err);
}

const FixedTupleSignature = dszig.Signature(struct {
    pub const Input = struct { source: [2]i64 };
    pub const Output = struct {
        matrix: [2][2]i64,
        snippets: [2]dszig.Code,
        optional: ?[2]i64,
        empty: [0]bool,
    };
});

test "fixed tuple Predict preserves nested values and rejects incorrect cardinality" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const body =
        \\{"matrix":[["1",true],[3,4]],"snippets":["first()","second()"],"optional":[5,6],"empty":[]}
    ;
    const completions = [_][]const u8{
        \\[[ ## matrix ## ]]
        \\[["1",true],[3,4]]
        \\[[ ## snippets ## ]]
        \\["first()","second()"]
        \\[[ ## optional ## ]]
        \\[5,6]
        \\[[ ## empty ## ]]
        \\[]
        \\[[ ## completed ## ]]
        ,
        body,
        \\<matrix>[["1",true],[3,4]]</matrix>
        \\<snippets>["first()","second()"]</snippets>
        \\<optional>[5,6]</optional>
        \\<empty>[]</empty>
        ,
    };
    const predict: dszig.Predict(FixedTupleSignature) = .{};
    for ([_]dszig.Adapter{ .chat, .json, .xml }, completions) |adapter, completion| {
        var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{completion} };
        const output = try predict.forward(
            .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = scripted.lm(), .adapter = adapter },
            .{ .source = .{ 10, 20 } },
        );
        try testing.expectEqual([2][2]i64{ .{ 1, 1 }, .{ 3, 4 } }, output.matrix);
        try testing.expectEqualStrings("first()", output.snippets[0].code);
        try testing.expectEqualStrings("second()", output.snippets[1].code);
        try testing.expectEqual([2]i64{ 5, 6 }, output.optional.?);
    }
    const validate = FixedTupleSignature.output_fields_list[0].validate.?;
    for ([_][]const u8{ "[]", "[[1,2]]", "[[1,2],[3,4],[5,6]]", "[[1],[2,3]]" }) |text| {
        const input = try std.json.parseFromSliceLeaky(dszig.Value, arena, text, .{});
        try testing.expectError(error.ParseValue, validate(arena, input));
        try testing.expectError(error.TypeMismatch, dszig.value.fromValue(arena, [2][2]i64, input));
    }
    const empty = try dszig.value.toValue(arena, [0]bool, .{});
    try testing.expectEqual([0]bool{}, try dszig.value.fromValue(arena, [0]bool, empty));
    const pair = try dszig.value.toValue(arena, [2]i64, .{ 7, 8 });
    const sentinel = try dszig.value.fromValue(arena, [2:99]i64, pair);
    try testing.expectEqual(@as(i64, 99), sentinel[2]);
}

test "fixed model tuples preserve typed feedback and owning clones after source destruction" {
    const Point = struct {
        pub const model_name = "Point";
        x: i64,
        name: []const u8,
    };
    const S = dszig.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { points: ?[1][1]Point };
    });
    const clone = block: {
        var source: std.heap.ArenaAllocator = .init(testing.allocator);
        defer source.deinit();
        const arena = source.allocator();
        const name = try arena.dupe(u8, "A");
        const runtime = try dszig.value.toValue(arena, ?[1][1]Point, .{.{.{ .x = 3, .name = name }}});
        const field = S.output_fields_list[0];
        try testing.expectEqualStrings(
            "((Point(x=3, name='A'),),)",
            try field.python_str.?(arena, runtime),
        );
        const masked = (try field.refine_mask.?(arena, runtime)).?;
        const outer = dszig.python_value.view(masked).?.tuple;
        const inner = dszig.python_value.view(outer[0]).?.tuple;
        try testing.expectEqualStrings("<non-serializable: Point>", inner[0].string);
        // Cloning must own every tuple, dictionary and string child.
        const owned = try runtime.custom.clone(testing.allocator);
        name[0] = 'Z';
        break :block owned;
    };
    defer clone.deinit(testing.allocator);
    const outer = dszig.python_value.view(.{ .custom = clone }).?.tuple;
    const inner = dszig.python_value.view(outer[0]).?.tuple;
    try testing.expectEqualStrings("A", inner[0].object.get("name").?.string);
    try testing.expectEqual(@as(i64, 3), inner[0].object.get("x").?.integer);
}

/// Mock LM that answers a question correctly only when the prompt carries at
/// least `difficulty` demos: few-shot demos genuinely improve it.
const LearningLM = struct {
    const difficulty = [_]usize{ 0, 0, 1, 2, 3 };

    fn lm(self: *LearningLM) dszig.LM {
        return .{ .ptr = self, .vtable = &.{ .complete = complete } };
    }

    fn complete(_: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
        const out = try arena.alloc(dszig.lm.Completion, 1);
        out[0] = .{ .text = try reply(arena, request) };
        return out;
    }

    fn reply(arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const u8 {
        const messages = request.messages;
        var demos: usize = 0;
        for (messages) |m| if (m.role == .assistant) {
            demos += 1;
        };
        const last = messages[messages.len - 1].content.text;
        for (rag.train_rows, difficulty) |row, need| {
            if (std.mem.indexOf(u8, last, row[0]) == null) continue;
            const answer = if (demos >= need) row[1] else "unknown";
            return std.fmt.allocPrint(arena, "<plan>\n{d} demos.\n</plan>\n\n<answer>\n{s}\n</answer>", .{ demos, answer });
        }
        return error.LMFailure;
    }
};

test "Evaluate uses Python decimal ties-to-even without intermediate rounding or overflow" {
    const P = struct {
        pub const Input = struct { score: f64 };
        pub const Output = struct {};
        pub fn forward(_: *const @This(), _: dszig.Context, _: Input) !Output {
            return .{};
        }
        fn metric(_: Allocator, example: *const dszig.Example, _: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            return example.record.get("score").?.float;
        }
    };
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = unused.lm() };
    // Exact result encodings from actual DSPy 3.5 Evaluate with one example.
    const cases = [_]struct { metric: f64, bits: u64 }{
        .{ .metric = 0.02675, .bits = 0x4005_5c28_f5c2_8f5c },
        .{ .metric = 0.00125, .bits = 0x3fbe_b851_eb85_1eb8 },
        .{ .metric = 0.00375, .bits = 0x3fd8_51eb_851e_b852 },
        .{ .metric = -0.00125, .bits = 0xbfbe_b851_eb85_1eb8 },
        .{ .metric = -0.00375, .bits = 0xbfd8_51eb_851e_b852 },
        .{ .metric = -1e-20, .bits = 0x8000_0000_0000_0000 },
        .{ .metric = 1e300, .bits = 0x7ea2_aa4f_4a40_5be2 },
        .{ .metric = 1e308, .bits = 0x7ff0_0000_0000_0000 },
        .{ .metric = -std.math.inf(f64), .bits = 0xfff0_0000_0000_0000 },
    };
    var example: dszig.Example = .{ .record = .empty, .input_keys = &.{"score"} };
    for (cases) |case| {
        try example.record.put(arena, "score", .{ .float = case.metric });
        const result = try dszig.optimize.evaluate(P, ctx, &.{}, &.{example}, P.metric, 1);
        try testing.expectEqual(case.bits, @as(u64, @bitCast(result.score)));
    }
    try example.record.put(arena, "score", .{ .float = std.math.nan(f64) });
    const result = try dszig.optimize.evaluate(P, ctx, &.{}, &.{example}, P.metric, 1);
    try testing.expect(std.math.isNan(result.score));
}

test "native input decoding needs no projection allocation and never admits labels" {
    const P = struct {
        product: *i64,
        pub const Input = struct { number: i64, text: []const u8, scale: i64 = 2 };
        pub const Output = struct {};

        pub fn forward(self: *const @This(), _: dszig.Context, input: Input) !Output {
            self.product.* = input.number * @as(i64, @intCast(input.text.len)) * input.scale;
            return .{};
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var example = try dszig.Example.fromJson(
        arena,
        "{\"number\":7,\"text\":\"abc\",\"scale\":5,\"answer\":\"private\"}",
        &.{ "text", "unknown", "number", "text" },
    );
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = testing.failing_allocator, .lm = unused.lm() };
    var product: i64 = -1;
    const program: P = .{ .product = &product };
    // Input decoding itself allocates nothing; a retained Prediction would.
    _ = try dszig.optimize.forwardProgram(P, ctx, &program, &example, null);
    try testing.expectEqual(@as(i64, 42), product);
    example.input_keys = &.{ "text", "number", "scale" };
    _ = try dszig.optimize.forwardProgram(P, ctx, &program, &example, null);
    try testing.expectEqual(@as(i64, 105), product);

    // An existing training label is not an input just because the program asks for it.
    example.input_keys = &.{"number"};
    product = -1;
    try testing.expectError(
        error.MissingField,
        dszig.optimize.forwardProgram(P, ctx, &program, &example, null),
    );
    try testing.expectEqual(@as(i64, -1), product);

    // Decode in declaration order: an earlier invalid input precedes a later missing role.
    try example.record.put(arena, "number", .{ .bool = true });
    try testing.expectError(
        error.TypeMismatch,
        dszig.optimize.forwardProgram(P, ctx, &program, &example, null),
    );
    try testing.expectEqual(@as(i64, -1), product);
}

test "prepared native predictions preserve mutable metrics and retained results" {
    const P = struct {
        pub const Input = struct { number: i64, text: []const u8 };
        pub const Output = struct { answer: i64, echo: []const u8 };

        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            if (input.number < 0) return error.LMFailure;
            return .{
                .answer = input.number * @as(i64, @intCast(input.text.len)),
                .echo = input.text,
            };
        }

        fn metric(_: Allocator, _: *const dszig.Example, prediction: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const record = prediction orelse return error.ParseValue;
            const answer = record.get("answer").?.integer;
            record.fields().getPtr("answer").?.* = .{ .integer = -100 };
            return if (answer == 21) 1 else 0;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var example = try dszig.Example.fromJson(
        arena,
        "{\"number\":7,\"text\":\"abc\",\"answer\":999}",
        &.{ "number", "text" },
    );
    var state: dszig.optimize.RecordPrediction.State = .{ .output = .empty };
    try state.output.ensureTotalCapacity(testing.allocator, 2);
    defer state.output.deinit(testing.allocator);
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = testing.failing_allocator, .lm = unused.lm() };
    const program: P = .{};
    var output: P.Output = undefined;

    // The actual runner must use only provisioned outer storage for scalar
    // fields: neither the Prediction instance nor its record allocates.
    const prediction = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try testing.expect(prediction.fields() == &state.output);
    try testing.expect(prediction.getLMUsage() == null);
    try testing.expect(prediction.completions() == null);
    try testing.expectEqual(@as(i64, 21), prediction.get("answer").?.integer);
    try testing.expectEqualStrings("abc", prediction.get("echo").?.string);
    const retained = try dszig.optimize.runProgram(
        P,
        .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = unused.lm() },
        &program,
        &example,
    );
    const metric: dszig.optimize.Metric = P.metric;
    try testing.expectEqual(@as(f64, 1), try metric(ctx.arena, &example, &prediction, null));
    try testing.expectEqual(@as(i64, -100), prediction.get("answer").?.integer);
    try testing.expectEqual(@as(i64, 999), example.record.get("answer").?.integer);

    try example.record.put(arena, "number", .{ .integer = 8 });
    try example.record.put(arena, "text", .{ .string = "xy" });
    // The next attempt rewrites the one borrowed instance every copy aliases.
    const next = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try testing.expect(next.fields() == prediction.fields());
    try testing.expectEqual(@as(i64, 16), prediction.get("answer").?.integer);
    try testing.expectEqualStrings("xy", prediction.get("echo").?.string);
    try testing.expectEqual(@as(i64, 21), retained.get("answer").?.integer);
    try testing.expectEqualStrings("abc", retained.get("echo").?.string);
    try testing.expectEqual(@as(usize, 0), unused.index);
}

test "prepared prediction failures clear stale records and partial conversions" {
    const P = struct {
        pub const Input = struct { number: i64 };
        pub const Output = struct { answer: i64 };

        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            if (input.number < 0) return error.LMFailure;
            return .{ .answer = input.number * 2 };
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var example = try dszig.Example.fromJson(arena, "{\"number\":7}", &.{"number"});
    var state: dszig.optimize.RecordPrediction.State = .{ .output = .empty };
    try state.output.ensureTotalCapacity(testing.allocator, 2);
    defer state.output.deinit(testing.allocator);
    const prediction = &state.output;
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = testing.failing_allocator, .lm = unused.lm() };
    const program: P = .{};
    var output: P.Output = undefined;

    _ = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try example.record.put(arena, "number", .{ .bool = true });
    try testing.expectError(error.TypeMismatch, dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state));
    try testing.expectEqual(@as(usize, 0), prediction.count());
    try example.record.put(arena, "number", .{ .integer = 7 });
    _ = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try example.record.put(arena, "number", .{ .integer = -1 });
    try testing.expectError(error.LMFailure, dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state));
    try testing.expectEqual(@as(usize, 0), prediction.count());

    // A later conversion failure must discard already-written earlier fields.
    const Nested = struct { answer: i64, notes: []const []const u8 };
    try testing.expectError(error.OutOfMemory, dszig.value.toRecordInto(
        testing.failing_allocator,
        Nested,
        &.{ .answer = 5, .notes = &.{"note"} },
        prediction,
    ));
    try testing.expectEqual(@as(usize, 0), prediction.count());
    try example.record.put(arena, "number", .{ .integer = 8 });
    _ = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try testing.expectEqual(@as(i64, 16), prediction.get("answer").?.integer);
}

fn metricPromptTokens(arena: Allocator, prediction: dszig.optimize.RecordPrediction) !?i64 {
    const usage = prediction.getLMUsage() orelse return null;
    const total = try dszig.prediction.Graph.toValue(arena, usage);
    return total.object.get("configured").?.object.get("prompt_tokens").?.integer;
}

test "evaluation metrics observe each run's LM usage and retained clones survive prepared reuse" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    const P = dszig.Predict(Sig);
    const Seen = struct {
        var tokens: [2]i64 = @splat(-1);
        var count: usize = 0;

        fn metric(arena: Allocator, _: *const dszig.Example, prediction: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const run = prediction orelse return error.ParseValue;
            const completions = run.completions() orelse return error.ParseValue;
            if (completions.len != 1 or completions[0].get("answer").?.integer != run.get("answer").?.integer)
                return error.ParseValue;
            const prompt = (metricPromptTokens(arena, run.*) catch return error.ParseValue) orelse return error.ParseValue;
            tokens[count] = prompt;
            count += 1;
            // The score depends on both the usage and the fields of this run.
            return @floatFromInt(prompt * run.get("answer").?.integer);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();

    const counts = [_]i64{ 3, 5, 7, 11, 13, 17, 19 };
    var metadata: [counts.len]dszig.lm.ResponseMetadata = undefined;
    for (&metadata, counts) |*item, count| item.* = .{ .usage = .{ .input_tokens = .{ .integer = count } } };
    var engine: dszig.lm.ScriptedLM = .{
        .responses = &.{
            "<answer>1</answer>", "<answer>2</answer>", "<answer>3</answer>", "<answer>4</answer>",
            "<answer>5</answer>", "<answer>6</answer>", "<answer>7</answer>",
        },
        .model = "configured",
        .metadata = &metadata,
    };
    var storage: dszig.usage_tracker.Storage = .{ .arena = arena, .io = testing.io };
    const plain: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = scratch.allocator(), .lm = engine.lm() };
    var tracked = plain;
    tracked.track_usage = .{ .storage = &storage, .limits = .{ .models = 1, .entries = 2 } };
    const program: P = .{};
    const devset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"a\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"b\"}", &.{"question"}),
    };

    // Every evaluated example hands the metric its own invocation's usage.
    const evaluation = try dszig.optimize.evaluate(P, tracked, &program, &devset, Seen.metric, 1);
    try testing.expectEqual(@as(usize, 2), Seen.count);
    try testing.expectEqualSlices(i64, &.{ 3, 5 }, &Seen.tokens);
    try testing.expectEqual(@as(f64, 3 * 1 + 5 * 2), evaluation.sum);
    try testing.expectEqual(@as(usize, 0), evaluation.errors);

    // A clone outlives both the scratch reset and the next prepared attempt.
    var record_storage: dszig.value.RecordStorage(1) = undefined;
    try record_storage.init();
    var state: dszig.optimize.RecordPrediction.State = .{ .output = record_storage.record };
    var output: P.Output = undefined;
    const first = try dszig.optimize.runProgramInto(P, tracked, &program, &devset[0], &output, &state);
    try testing.expectEqual(@as(?i64, 7), try metricPromptTokens(arena, first));
    first.fields().getPtr("answer").?.* = .{ .integer = 99 };
    try testing.expectEqual(@as(i64, 3), first.completions().?[0].get("answer").?.integer);
    first.fields().getPtr("answer").?.* = .{ .integer = 3 };
    const retained = try first.clone(arena);
    try testing.expect(retained.fields() != first.fields());
    _ = scratch.reset(.free_all);
    const second = try dszig.optimize.runProgramInto(P, tracked, &program, &devset[1], &output, &state);
    try testing.expect(second.fields() == first.fields());
    try testing.expectEqual(@as(i64, 4), first.get("answer").?.integer);
    try testing.expectEqual(@as(?i64, 11), try metricPromptTokens(arena, first));
    try testing.expectEqual(@as(i64, 3), retained.get("answer").?.integer);
    try testing.expectEqual(@as(i64, 3), retained.completions().?[0].get("answer").?.integer);
    try testing.expectEqual(@as(?i64, 7), try metricPromptTokens(arena, retained));

    // Tracking off, or any already-active tracker, leaves the run without usage.
    const untracked = try dszig.optimize.runProgramInto(P, plain, &program, &devset[0], &output, &state);
    try testing.expectEqual(@as(i64, 5), untracked.get("answer").?.integer);
    try testing.expect(untracked.getLMUsage() == null);
    var tracker: dszig.UsageTracker = undefined;
    try tracker.init(&storage, .{ .models = 1, .entries = 1 });
    var explicit = tracked;
    explicit.usage_tracker = &tracker;
    const suppressed = try dszig.optimize.runProgram(P, explicit, &program, &devset[0]);
    try testing.expectEqual(@as(i64, 6), suppressed.get("answer").?.integer);
    try testing.expect(suppressed.getLMUsage() == null);
    const charged = try dszig.prediction.Graph.toValue(arena, try tracker.getTotalTokens(arena));
    try testing.expectEqual(@as(i64, 17), charged.object.get("configured").?.object.get("prompt_tokens").?.integer);

    // A failed run leaves neither the previous usage nor its record behind.
    const last = try dszig.optimize.runProgramInto(P, tracked, &program, &devset[1], &output, &state);
    try testing.expectEqual(@as(?i64, 19), try metricPromptTokens(arena, last));
    try testing.expectError(
        error.ScriptExhausted,
        dszig.optimize.runProgramInto(P, tracked, &program, &devset[1], &output, &state),
    );
    try testing.expect(state.usage == null);
    try testing.expect(state.outputs == null);
    try testing.expectEqual(@as(usize, 0), state.output.count());
    try testing.expectEqual(@as(?i64, 7), try metricPromptTokens(arena, retained));
    try testing.expectEqual(counts.len, engine.index);
}

test "Evaluate borrows stable native model outputs without allocating boxes" {
    const P = struct {
        pub const Input = struct { number: i64, text: []const u8 };
        pub const Output = struct {
            code: dszig.Code,
            reasoning: dszig.Reasoning,
            file: dszig.File,
            box: dszig.BoundingBox,
            citations: dszig.Citations,
            citation: dszig.Citations.Citation,
            decision: dszig.decision.Noul,
            optional_code: ?dszig.Code,
            absent_code: ?dszig.Code,
        };

        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            return .{
                .code = .init(input.text),
                .reasoning = .{ .content = input.text },
                .file = .{ .file_id = input.text },
                .box = .{
                    .top = @as(f64, @floatFromInt(input.number)) / 100,
                    .bottom = 1,
                    .left = 0,
                    .right = 1,
                },
                .citations = .{ .citations = &.{} },
                .citation = .{ .cited_text = input.text },
                .decision = .{ .confidence = 1, .value = input.number > 0 },
                .optional_code = if (input.number == 7) .init(input.text) else null,
                .absent_code = null,
            };
        }

        fn metric(arena: Allocator, example: *const dszig.Example, prediction: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const record = prediction orelse return error.ParseValue;
            const text = example.record.get("text").?.string;
            if (!std.mem.eql(u8, text, record.get("code").?.custom.get(dszig.Code).?.code)) return 0;
            if (!std.mem.eql(u8, text, record.get("reasoning").?.custom.get(dszig.Reasoning).?.content)) return 0;
            if (!std.mem.eql(u8, text, record.get("file").?.custom.get(dszig.File).?.file_id.?)) return 0;
            const top = @as(f64, @floatFromInt(example.record.get("number").?.integer)) / 100;
            if (top != record.get("box").?.custom.get(dszig.BoundingBox).?.top) return 0;
            if (record.get("citations").?.custom.get(dszig.Citations).?.citations.len != 0) return 0;
            if (!std.mem.eql(u8, text, record.get("citation").?.custom.get(dszig.Citations.Citation).?.cited_text)) return 0;
            if (!record.get("decision").?.custom.get(dszig.decision.Noul).?.value) return 0;
            const optional = record.get("optional_code").?;
            if (example.record.get("number").?.integer == 7) {
                if (!std.mem.eql(u8, text, optional.custom.get(dszig.Code).?.code)) return 0;
            } else if (optional != .null) return 0;
            if (record.get("absent_code").? != .null) return 0;
            // A metric may release a borrowed handle, but cannot destroy its frame.
            for (record.fields().values()) |item| {
                if (item == .custom) item.custom.deinit(arena);
            }
            if (!std.mem.eql(u8, text, record.get("code").?.custom.get(dszig.Code).?.code)) return 0;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const examples = [_]dszig.Example{
        try dszig.Example.fromJson(memory.allocator(), "{\"number\":7,\"text\":\"first\"}", &.{ "number", "text" }),
        try dszig.Example.fromJson(memory.allocator(), "{\"number\":21,\"text\":\"second\"}", &.{ "number", "text" }),
    };
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const program: P = .{};
    const result = try dszig.optimize.evaluate(
        P,
        .{ .workspace = dszig.workspace.forTests(), .arena = testing.failing_allocator, .lm = unused.lm() },
        &program,
        &examples,
        P.metric,
        1,
    );
    try testing.expectEqual(@as(f64, 100), result.score);
    try testing.expectEqual(@as(usize, 0), result.errors);
}

test "borrowed model clones survive typed slot reuse and preserve projected classes" {
    const Java = dszig.CodeWithLanguage("java");
    var source: std.heap.ArenaAllocator = .init(testing.allocator);
    const text = try source.allocator().dupe(u8, "first");
    const original = try Java.init(text).toRuntime(source.allocator());
    var projected = try dszig.Code.fromValue(original);
    const borrowed = projected.borrowedRuntime();
    const owned = try borrowed.custom.clone(testing.allocator);
    defer owned.deinit(testing.allocator);

    try testing.expect(!borrowed.custom.owns_box);
    try testing.expectEqual(original.custom.ptr, borrowed.custom.ptr);
    try testing.expectEqualStrings("java", borrowed.custom.vtable.code.?.language);
    borrowed.custom.deinit(testing.failing_allocator);
    // The borrowed projection cannot destroy the original specialized object.
    try testing.expectEqualStrings("first", original.custom.get(Java).?.code);
    @memcpy(text, "other");
    projected = .init("replacement");
    source.deinit();

    try testing.expectEqualStrings("java", owned.vtable.code.?.language);
    try testing.expectEqualStrings("first", owned.get(Java).?.code);
    try testing.expect(owned.get(dszig.Code) == null);
    try testing.expect(owned.owns_box);
    try testing.expect(owned.owns_data);
}

test "cloned borrowed model fields own their data after frame and source destruction" {
    const Models = struct {
        code: dszig.Code,
        reasoning: dszig.Reasoning,
        file: dszig.File,
        box: dszig.BoundingBox,
        citation: dszig.Citations.Citation,
        citations: dszig.Citations,
        decision: dszig.decision.Noul,
    };
    var source: std.heap.ArenaAllocator = .init(testing.allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const text = try source.allocator().dupe(u8, "source");
    var children = [_]dszig.Citations.Citation{.{ .cited_text = text }};
    var models: Models = .{
        .code = .init(text),
        .reasoning = .{ .content = text },
        .file = .{ .file_id = text },
        .box = .{ .top = 0, .bottom = 1, .left = 0, .right = 1 },
        .citation = .{ .cited_text = text },
        .citations = .{ .citations = &children },
        .decision = .{ .confidence = 1, .value = true },
    };
    var storage: dszig.value.RecordStorage(7) = undefined;
    try storage.init();
    try dszig.value.toRecordInto(testing.failing_allocator, Models, &models, &storage.record);
    var clones: [7]dszig.runtime_value.Custom = undefined;
    var initialized: usize = 0;
    defer for (clones[0..initialized]) |item| item.deinit(testing.allocator);
    for (storage.record.values(), &clones) |item, *clone| {
        clone.* = try item.custom.clone(testing.allocator);
        initialized += 1;
        item.custom.deinit(testing.failing_allocator);
    }

    models.code = .init("changed");
    models.reasoning.content = "changed";
    models.file.file_id = "changed";
    models.box.top = 0.25;
    models.citation.cited_text = "changed";
    children[0].cited_text = "changed";
    models.decision.value = false;
    @memset(text, 'x');
    source.deinit();
    source_live = false;

    try testing.expectEqualStrings("source", clones[0].get(dszig.Code).?.code);
    try testing.expectEqualStrings("source", clones[1].get(dszig.Reasoning).?.content);
    try testing.expectEqualStrings("source", clones[2].get(dszig.File).?.file_id.?);
    try testing.expectEqual(@as(f64, 0), clones[3].get(dszig.BoundingBox).?.top);
    try testing.expectEqualStrings("source", clones[4].get(dszig.Citations.Citation).?.cited_text);
    try testing.expectEqualStrings("source", clones[5].get(dszig.Citations).?.citations[0].cited_text);
    try testing.expect(clones[6].get(dszig.decision.Noul).?.value);
}

test "prepared Decimal and raw scalar outputs borrow boxes while clones retain ownership" {
    const Fields = struct {
        amount: dszig.Decimal,
        optional: ?dszig.Decimal,
        raw: dszig.Value,
    };
    var source: std.heap.ArenaAllocator = .init(testing.allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const original = try dszig.Decimal.coerce(source.allocator(), .{ .string = "12.50" });
    var fields: Fields = .{ .amount = original, .optional = original, .raw = original.value };
    var storage: dszig.value.RecordStorage(3) = undefined;
    try storage.init();
    try dszig.value.toRecordInto(testing.failing_allocator, Fields, &fields, &storage.record);
    var clones: [3]dszig.runtime_value.Custom = undefined;
    var count: usize = 0;
    defer for (clones[0..count]) |clone| clone.deinit(testing.allocator);
    for (storage.record.values()) |item| {
        try testing.expect(!item.custom.owns_box);
        try testing.expect(!item.custom.owns_data);
        clones[count] = try item.custom.clone(testing.allocator);
        count += 1;
        item.custom.deinit(testing.failing_allocator);
    }
    const decimal = dszig.python_value.view(original.value).?.decimal;
    try testing.expectEqualStrings("1250", decimal.digits);
    // Reuse the frame and clear an optional without touching retained clones.
    fields.optional = null;
    try dszig.value.toRecordInto(testing.failing_allocator, Fields, &fields, &storage.record);
    try testing.expect(storage.record.get("optional").? == .null);
    source.deinit();
    source_live = false;
    for (clones) |clone| {
        try testing.expect(clone.owns_box);
        try testing.expect(clone.owns_data);
        const retained = dszig.python_value.view(.{ .custom = clone }).?.decimal;
        try testing.expectEqualStrings("1250", retained.digits);
        try testing.expectEqual(@as(i64, -2), retained.exponent);
    }
}

test "prepared native output storage supports indexed fields and reuse" {
    const P = struct {
        pub const Input = struct { number: i64 };
        pub const Output = struct {
            a: i64,
            b: i64,
            c: i64,
            d: i64,
            e: i64,
            f: i64,
            g: i64,
            h: i64,
            i: i64,
        };

        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            var output: Output = undefined;
            inline for (@typeInfo(Output).@"struct".field_names, 0..) |name, index| {
                @field(output, name) = input.number + @as(i64, @intCast(index));
            }
            return output;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var example = try dszig.Example.fromJson(arena, "{\"number\":21}", &.{"number"});
    var storage: dszig.value.RecordStorage(9) = undefined;
    try storage.init();
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = testing.failing_allocator, .lm = unused.lm() };
    const program: P = .{};
    var output: P.Output = undefined;

    // The state adopts the startup storage's record and is its only header.
    var state: dszig.optimize.RecordPrediction.State = .{ .output = storage.record };
    const prediction = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try testing.expectEqual(@as(i64, 21), prediction.get("a").?.integer);
    try testing.expectEqual(@as(i64, 29), prediction.get("i").?.integer);
    try testing.expectEqual(@as(usize, 9), prediction.fields().count());
    try example.record.put(arena, "number", .{ .integer = -21 });
    _ = try dszig.optimize.runProgramInto(P, ctx, &program, &example, &output, &state);
    try testing.expectEqual(@as(i64, -21), prediction.get("a").?.integer);
    try testing.expectEqual(@as(i64, -13), prediction.get("i").?.integer);
}

test "Evaluate counts program and metric failures together and keeps allocation failures fatal" {
    const P = struct {
        calls: *usize,
        pub const Input = struct { question: []const u8 };
        pub const Output = struct {};

        pub fn forward(self: *const @This(), _: dszig.Context, input: Input) !Output {
            self.calls.* += 1;
            if (std.mem.eql(u8, input.question, "program_fail")) return error.LMFailure;
            if (std.mem.eql(u8, input.question, "program_oom")) return error.OutOfMemory;
            return .{};
        }

        fn metric(_: Allocator, example: *const dszig.Example, _: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const question = example.record.get("question").?.string;
            if (std.mem.eql(u8, question, "metric_fail")) return error.ParseValue;
            if (std.mem.eql(u8, question, "metric_oom")) return error.OutOfMemory;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var unused: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = unused.lm() };
    const devset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"metric_fail\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good1\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good2\"}", &.{"question"}),
    };
    var calls: usize = 0;
    const program: P = .{ .calls = &calls };
    const result = try dszig.optimize.evaluate(P, ctx, &program, &devset, P.metric, 2);
    try testing.expectEqual(@as(f64, 66.67), result.score);
    try testing.expectEqual(@as(f64, 2), result.sum);
    try testing.expectEqual(@as(usize, 3), result.total);
    try testing.expectEqual(@as(usize, 1), result.errors);
    try testing.expectEqual(@as(usize, 3), calls);

    for ([_]usize{ 0, 1 }) |max_errors| {
        calls = 0;
        try testing.expectError(error.ParseValue, dszig.optimize.evaluate(P, ctx, &program, &devset, P.metric, max_errors));
        try testing.expectEqual(@as(usize, 1), calls);
    }

    const program_failure = try dszig.Example.fromJson(arena, "{\"question\":\"program_fail\"}", &.{"question"});
    calls = 0;
    try testing.expectError(error.ParseValue, dszig.optimize.evaluate(P, ctx, &program, &.{ program_failure, devset[0], devset[1] }, P.metric, 2));
    try testing.expectEqual(@as(usize, 2), calls);

    for ([_][]const u8{ "metric_oom", "program_oom" }) |question| {
        var failure = try dszig.Example.fromJson(arena, "{\"question\":\"unused\"}", &.{"question"});
        try failure.record.put(arena, "question", .{ .string = question });
        calls = 0;
        try testing.expectError(error.OutOfMemory, dszig.optimize.evaluate(P, ctx, &program, &.{ failure, devset[1] }, P.metric, 100));
        try testing.expectEqual(@as(usize, 1), calls);
    }
}

test "BootstrapFewShot counts metric failures with program failures after restoring teacher demos" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = dszig.Predict(Sig);
    const Model = struct {
        calls: usize = 0,
        fail_first: bool = false,

        fn complete(ptr: *anyopaque, arena: Allocator, request: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            if (self.fail_first and self.calls == 1) return error.LMFailure;
            const out = try arena.alloc(dszig.lm.Completion, request.options.n);
            for (out) |*completion| completion.* = .{ .text = "<answer>ok</answer>" };
            return out;
        }
    };
    const Metric = struct {
        var expected_demos: usize = 3;

        fn run(_: Allocator, example: *const dszig.Example, _: ?*const dszig.optimize.RecordPrediction, trace: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            const steps = trace.?.steps.items;
            // The current example is restored before the metric, but a prior
            // failed teacher call has already pruned its own example.
            testing.expectEqual(@as(usize, 1), steps.len) catch unreachable;
            const demos = steps[0].predictor.demos;
            const question = example.record.get("question").?.string;
            testing.expectEqual(expected_demos, demos.len) catch unreachable;
            if (expected_demos == 2) {
                testing.expectEqualStrings("metric_fail", demos[0].get("question").?.string) catch unreachable;
                testing.expectEqualStrings("good2", demos[1].get("question").?.string) catch unreachable;
            }
            if (std.mem.eql(u8, question, "metric_fail")) return error.ParseValue;
            if (std.mem.eql(u8, question, "metric_oom")) return error.OutOfMemory;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const trainset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"metric_fail\",\"answer\":\"ok\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good1\",\"answer\":\"ok\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"good2\",\"answer\":\"ok\"}", &.{"question"}),
    };
    var teacher: P = .{};
    teacher.state.demos = &.{ trainset[0].record, trainset[1].record, trainset[2].record };
    var model: Model = .{};
    const ctx: dszig.Context = .{
        .workspace = dszig.workspace.forTests(),
        .arena = arena,
        .lm = .{ .ptr = &model, .vtable = &.{ .complete = Model.complete } },
        .adapter = .xml,
    };
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = Metric.run,
        .max_bootstrapped_demos = 3,
        .max_labeled_demos = 0,
        .max_errors = 2,
    };
    const result = try optimizer.compile(P, arena, ctx, .{}, teacher, &trainset);
    try testing.expectEqual(@as(usize, 3), result.report.attempts);
    try testing.expectEqual(@as(usize, 1), result.report.errors);
    try testing.expectEqual(@as(usize, 2), result.report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), result.report.validation);
    try testing.expectEqual(@as(usize, 3), model.calls);
    try testing.expectEqual(@as(usize, 3), teacher.state.demos.len);
    try testing.expectEqual(@as(usize, 2), result.program.state.demos.len);

    for ([_]usize{ 0, 1 }) |max_errors| {
        model.calls = 0;
        optimizer.max_errors = max_errors;
        try testing.expectError(error.ParseValue, optimizer.compile(P, arena, ctx, .{}, teacher, &trainset));
        try testing.expectEqual(@as(usize, 1), model.calls);
    }

    model.calls = 0;
    model.fail_first = true;
    Metric.expected_demos = 2;
    optimizer.max_errors = optimizer.error_count + 2;
    try testing.expectError(error.ParseValue, optimizer.compile(P, arena, ctx, .{}, teacher, &.{ trainset[1], trainset[0], trainset[2] }));
    try testing.expectEqual(@as(usize, 2), model.calls);

    model.calls = 0;
    model.fail_first = false;
    Metric.expected_demos = 3;
    const recovered = try optimizer.compile(
        P,
        arena,
        ctx,
        .{},
        teacher,
        &.{trainset[2]},
    );
    try testing.expectEqual(@as(usize, 1), recovered.report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), recovered.report.errors);
    try testing.expectEqual(@as(usize, 1), model.calls);
    try testing.expectEqual(@as(usize, 5), optimizer.error_count);

    const allocation_failure = try dszig.Example.fromJson(arena, "{\"question\":\"metric_oom\",\"answer\":\"ok\"}", &.{"question"});
    model.calls = 0;
    model.fail_first = false;
    Metric.expected_demos = 3;
    optimizer.max_errors = 100;
    try testing.expectEqual(@as(usize, 5), optimizer.error_count);
    try testing.expectError(error.OutOfMemory, optimizer.compile(P, arena, ctx, .{}, teacher, &.{ allocation_failure, trainset[1] }));
    try testing.expectEqual(@as(usize, 1), model.calls);
    try testing.expectEqual(@as(usize, 5), optimizer.error_count);
}

test "BootstrapFewShot propagates cancellation without spending the retry budget" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Teacher = struct {
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        predictor: dszig.Predict(Sig) = .{},
        calls: *usize,

        pub fn forward(self: *const @This(), _: dszig.Context, _: Input) error{Canceled}!Output {
            self.calls.* += 1;
            return error.Canceled;
        }
    };
    const Student = struct {
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        predictor: dszig.Predict(Sig) = .{},
        compiled: bool = false,

        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) !Output {
            return self.predictor.forward(ctx, input);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var calls: usize = 0;
    var mock: dszig.lm.ReplayLM = .{ .calls = &.{}, .used = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = mock.lm() };
    const row = try dszig.Example.fromJson(arena, "{\"question\":\"q\",\"answer\":\"a\"}", &.{"question"});
    var optimizer: dszig.BootstrapFewShot = .{
        .max_rounds = 2,
        .max_errors = 10,
        .max_labeled_demos = 0,
    };
    try testing.expectError(error.Canceled, optimizer.compile(
        Student,
        arena,
        ctx,
        .{},
        Teacher{ .calls = &calls },
        &.{ row, row },
    ));
    try testing.expectEqual(@as(usize, 1), calls);
    try testing.expectEqual(@as(usize, 0), optimizer.error_count);
}

test "BootstrapFewShot counts failed demo comparisons and retains earlier filters" {
    const P = dszig.Predict(dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const Program = struct {
        first: P = .{},
        second: P = .{},
        compiled: bool = false,
        calls: ?*usize = null,
        pub const Input = P.Input;
        pub const Output = P.Output;

        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) !Output {
            self.calls.?.* += 1;
            try testing.expectEqualStrings("good", input.question);
            try testing.expectEqual(@as(usize, 0), self.first.state.demos.len);
            try testing.expectEqual(@as(usize, 2), self.second.state.demos.len);
            const inputs = try dszig.value.toRecord(ctx.arena, Input, input);
            const answer: Output = .{ .answer = "ok" };
            const outputs = try dszig.value.toRecord(ctx.arena, Output, answer);
            try ctx.trace.?.record(&self.second.state, &inputs, &outputs);
            return answer;
        }
    };

    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var bad = try dszig.Example.fromJson(arena, "{\"question\":\"bad\"}", &.{"question"});
    try bad.record.put(arena, "metadata", try dszig.python_value.decimal(arena, "sNaN"));
    var distinct = try dszig.Example.fromJson(arena, "{\"question\":\"bad\"}", &.{"question"});
    try distinct.record.put(arena, "metadata", try dszig.python_value.decimal(arena, "sNaN"));
    const good = try dszig.Example.fromJson(arena, "{\"question\":\"good\"}", &.{"question"});
    const first_demos = [_]dszig.value.Record{bad.record};
    const second_demos = [_]dszig.value.Record{ bad.record, distinct.record };
    var calls: usize = 0;
    var teacher: Program = .{ .calls = &calls };
    teacher.first.state.demos = &first_demos;
    teacher.second.state.demos = &second_demos;
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena };
    var optimizer: dszig.BootstrapFewShot = .{ .max_labeled_demos = 0, .max_errors = 2 };
    const result = try optimizer.compile(Program, arena, ctx, .{}, teacher, &.{ bad, good });
    try testing.expectEqual(@as(usize, 1), calls);
    try testing.expectEqual(@as(usize, 2), result.report.attempts);
    try testing.expectEqual(@as(usize, 1), result.report.errors);
    try testing.expectEqual(@as(usize, 1), result.report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), result.report.validation);
    try testing.expectEqualStrings("good", result.program.second.state.demos[0].get("question").?.string);
    try testing.expectEqual(@as(usize, 1), teacher.first.state.demos.len);
    try testing.expectEqual(@as(usize, 2), teacher.second.state.demos.len);

    optimizer = .{ .max_labeled_demos = 0, .max_errors = 1 };
    calls = 0;
    try testing.expectError(error.ParseValue, optimizer.compile(Program, arena, ctx, .{}, teacher, &.{ bad, good }));
    try testing.expectEqual(@as(usize, 0), calls);
    try testing.expectEqual(@as(usize, 1), optimizer.error_count);
}

test "BootstrapFewShot binds trace demos before mapping predictors and outside the error budget" {
    const P = dszig.Predict(dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const Program = struct {
        predictor: P = .{},
        compiled: bool = false,
        inputs_json: []const u8 = "{}",
        outputs_json: []const u8 = "{}",
        mapped: bool = true,
        boxed_base: bool = false,

        const unmapped: P = .{};
        pub const Input = P.Input;
        pub const Output = P.Output;

        pub fn forward(self: *const @This(), ctx: dszig.Context, _: Input) !Output {
            const inputs = try dszig.Example.fromJson(ctx.arena, self.inputs_json, &.{});
            var outputs = try dszig.Example.fromJson(ctx.arena, self.outputs_json, &.{});
            if (self.boxed_base) {
                const base = try dszig.python_value.dictionary(ctx.arena, &.{
                    .{ .key = .{ .integer = 1 }, .value = .null },
                });
                try outputs.record.put(ctx.arena, "base", base);
            }
            const state = if (self.mapped) &self.predictor.state else &unmapped.state;
            try ctx.trace.?.record(state, &inputs.record, &outputs.record);
            return .{ .answer = "answer" };
        }

        fn reject(_: Allocator, _: *const dszig.Example, _: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
            return 0;
        }
    };

    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = script.lm() };
    const row = try dszig.Example.fromJson(arena, "{\"question\":\"q\"}", &.{"question"});
    var optimizer: dszig.BootstrapFewShot = .{ .max_labeled_demos = 0, .max_errors = 10 };

    const accepted = try optimizer.compile(Program, arena, ctx, .{}, Program{
        .inputs_json = "{\"base\":{\"inherited\":7,\"answer\":\"stale\",\"question\":\"old\",\"augmented\":false},\"question\":\"q\"}",
        .outputs_json = "{\"answer\":\"answer\",\"extra\":8}",
    }, &.{row});
    try testing.expectEqual(@as(usize, 1), accepted.report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), accepted.program.predictor.state.demos.len);
    const demo = accepted.program.predictor.state.demos[0];
    try testing.expectEqual(@as(usize, 5), demo.count());
    for (demo.keys(), [_][]const u8{ "inherited", "answer", "question", "augmented", "extra" }) |key, expected| {
        try testing.expectEqualStrings(expected, key);
    }
    try testing.expectEqual(@as(i64, 7), demo.get("inherited").?.integer);
    try testing.expectEqualStrings("q", demo.get("question").?.string);
    try testing.expectEqualStrings("answer", demo.get("answer").?.string);
    try testing.expect(demo.get("augmented").?.bool);

    for ([_]Program{
        .{ .outputs_json = "{\"augmented\":false}" },
        .{ .outputs_json = "{\"augmented\":false}", .mapped = false },
        .{ .inputs_json = "{\"question\":\"q\"}", .outputs_json = "{\"question\":\"q\"}" },
        .{ .inputs_json = "{\"base\":null}", .outputs_json = "{\"base\":{}}" },
        .{ .inputs_json = "{\"self\":null}" },
    }) |teacher| {
        try testing.expectError(error.DuplicateTraceDemoKeyword, optimizer.compile(
            Program,
            arena,
            ctx,
            .{},
            teacher,
            &.{row},
        ));
        try testing.expectEqual(@as(usize, 0), optimizer.error_count);
    }
    try testing.expectError(error.UnsupportedTraceDemoBase, optimizer.compile(
        Program,
        arena,
        ctx,
        .{},
        Program{ .boxed_base = true },
        &.{row},
    ));
    try testing.expectEqual(@as(usize, 0), optimizer.error_count);

    optimizer.metric = Program.reject;
    const rejected = try optimizer.compile(Program, arena, ctx, .{}, Program{
        .outputs_json = "{\"augmented\":false}",
        .mapped = false,
    }, &.{row});
    try testing.expectEqual(@as(usize, 1), rejected.report.attempts);
    try testing.expectEqual(@as(usize, 0), rejected.report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), rejected.report.errors);
    try testing.expectEqual(@as(usize, 0), rejected.program.predictor.state.demos.len);
}

test "BootstrapFewShot scores never drop across three generations" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    var optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 4, .max_labeled_demos = 0 };

    var teacher: rag.RAG = .{};
    var scores: [4]f64 = undefined;
    scores[0] = (try dszig.optimize.evaluate(rag.RAG, ctx, &teacher, trainset, rag.keywordMetric, 10)).score;
    for (scores[1..]) |*score| {
        const result = try optimizer.compile(rag.RAG, out.allocator(), ctx, .{}, teacher, trainset);
        teacher = result.program;
        score.* = (try dszig.optimize.evaluate(rag.RAG, ctx, &teacher, trainset, rag.keywordMetric, 10)).score;
    }
    try testing.expectEqualSlices(f64, &.{ 40.0, 80.0, 100.0, 100.0 }, &scores);
    for (scores[1..], scores[0 .. scores.len - 1]) |now, before| try testing.expect(now >= before);
}

test "teacher_settings runs bootstrapping under the teacher's LM" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = silent.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = rag.keywordMetric,
        .max_bootstrapped_demos = 2,
        .max_labeled_demos = 0,
        .teacher_settings = .{ .lm = keyword.lm() },
    };
    const result = try optimizer.compile(rag.RAG, out.allocator(), ctx, .{}, null, trainset);
    try testing.expectEqual(@as(usize, 2), result.report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), silent.index);
    try testing.expectEqual(@as(usize, 2), result.program.generate_answer.predict.state.demos.len);
}

test "BootstrapFewShot zero rounds retains labeled demos and zero errors propagates first failure" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = dszig.Predict(Sig);
    const FailingLM = struct {
        calls: usize = 0,
        fn complete(ptr: *anyopaque, _: Allocator, _: dszig.lm.Request) dszig.lm.Error![]const dszig.lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            return error.LMFailure;
        }
    };
    var failing: FailingLM = .{};
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = .{ .ptr = &failing, .vtable = &.{ .complete = FailingLM.complete } } };
    const trainset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"q0\",\"answer\":\"a0\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"q1\",\"answer\":\"a1\"}", &.{"question"}),
        try dszig.Example.fromJson(arena, "{\"question\":\"q2\",\"answer\":\"a2\"}", &.{"question"}),
    };
    var optimizer: dszig.BootstrapFewShot = .{ .max_rounds = 0, .max_errors = 0, .max_labeled_demos = 2 };
    const result = try optimizer.compile(P, arena, ctx, .{}, null, &trainset);
    try testing.expect(result.program.compiled);
    try testing.expectEqual(@as(usize, 0), result.report.attempts);
    try testing.expectEqual(@as(usize, 0), failing.calls);
    const demos = result.program.state.demos;
    try testing.expectEqual(@as(usize, 2), demos.len);
    // Actual pinned DSPy selects these labels after its shuffle and sample.
    try testing.expectEqualStrings("q2", demos[0].get("question").?.string);
    try testing.expectEqualStrings("a2", demos[0].get("answer").?.string);
    try testing.expectEqualStrings("q1", demos[1].get("question").?.string);
    try testing.expectEqualStrings("a1", demos[1].get("answer").?.string);
    optimizer.max_rounds = 1;
    optimizer.max_labeled_demos = 0;
    try testing.expectError(error.LMFailure, optimizer.compile(P, arena, ctx, .{}, null, &trainset));
    try testing.expectEqual(@as(usize, 1), failing.calls);

    // A teacher-local override must not leak to the outer failure handler.
    var compile_ctx = ctx;
    compile_ctx.max_errors = 2;
    failing.calls = 0;
    var inherited: dszig.BootstrapFewShot = .{
        .max_labeled_demos = 0,
        .teacher_settings = .{ .max_errors = 1 },
    };
    try testing.expectError(
        error.LMFailure,
        inherited.compile(P, arena, compile_ctx, .{}, null, &trainset),
    );
    try testing.expectEqual(@as(usize, 2), inherited.error_count);
    try testing.expectEqual(@as(usize, 2), failing.calls);
}

test "BootstrapFewShot zero threshold accepts a zero score while null rejects it" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    // The real shared mock answers Germany with Munich, not the gold Berlin.
    const trainset = (try rag.trainset(arena))[3..4];
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = rag.keywordMetric,
        .metric_threshold = 0.0,
        .max_bootstrapped_demos = 1,
        .max_labeled_demos = 0,
    };
    const accepted = try optimizer.compile(rag.RAG, arena, ctx, .{}, null, trainset);
    try testing.expect(accepted.program.compiled);
    try testing.expectEqual(@as(usize, 1), accepted.report.bootstrapped);
    try testing.expectEqual(@as(usize, 0), accepted.report.validation);
    const demos = accepted.program.generate_answer.predict.state.demos;
    try testing.expectEqual(@as(usize, 1), demos.len);
    try testing.expectEqualStrings("Munich", demos[0].get("answer").?.string);
    const demo_prediction = try dszig.optimize.RecordPrediction.init(arena, demos[0]);
    try testing.expectEqual(@as(f64, 0.0), try rag.keywordMetric(arena, &trainset[0], &demo_prediction, null));
    try testing.expectEqualStrings("Not sure.", (try dszig.Reasoning.fromValue(demos[0].get("reasoning").?)).content);

    optimizer.metric_threshold = null;
    const rejected = try optimizer.compile(rag.RAG, arena, ctx, .{}, null, trainset);
    try testing.expect(rejected.program.compiled);
    try testing.expectEqual(@as(usize, 0), rejected.report.bootstrapped);
    try testing.expectEqual(@as(usize, 1), rejected.report.validation);
    try testing.expectEqual(@as(usize, 0), rejected.program.generate_answer.predict.state.demos.len);
}

test "BootstrapFewShot retries the settings model without retuning a bound model" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Program = struct {
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        compiled: bool = false,
        fallback: dszig.Predict(Sig) = .{},
        bound: dszig.Predict(Sig) = .{},

        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) !Output {
            _ = try self.fallback.forward(ctx, input);
            return self.bound.forward(ctx, input);
        }
    };
    const Metric = struct {
        threadlocal var calls: usize = 0;

        fn score(
            _: Allocator,
            _: *const dszig.Example,
            _: ?*const dszig.optimize.RecordPrediction,
            _: ?*const dszig.Trace,
        ) dszig.value.ParseError!f64 {
            calls += 1;
            return if (calls == 2) 1 else 0;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var fallback: dszig.lm.ScriptedLM = .{ .responses = &.{ "<answer>fallback</answer>", "<answer>fallback</answer>" } };
    var bound: dszig.lm.ScriptedLM = .{ .responses = &.{ "<answer>bound</answer>", "<answer>bound</answer>" } };
    var fallback_tape: dszig.lm.RecordingLM = .{ .inner = fallback.lm(), .arena = arena };
    var bound_tape: dszig.lm.RecordingLM = .{ .inner = bound.lm(), .arena = arena };
    var teacher: Program = .{};
    teacher.bound.state.lm = bound_tape.lm();
    const train = [_]dszig.Example{
        try .fromJson(arena, "{\"question\":\"q\",\"answer\":\"label\"}", &.{"question"}),
    };
    var optimizer: dszig.BootstrapFewShot = .{ .metric = Metric.score, .max_rounds = 2, .max_labeled_demos = 0 };
    Metric.calls = 0;
    const result = try optimizer.compile(Program, arena, .{
        .workspace = dszig.workspace.forTests(),
        .arena = arena,
        .lm = fallback_tape.lm(),
        .lm_options = .{ .temperature = 0 },
    }, .{}, teacher, &train);
    try testing.expectEqual(@as(usize, 2), result.report.attempts);
    try testing.expectEqual(@as(usize, 2), fallback_tape.calls.items.len);
    try testing.expectEqual(@as(usize, 2), bound_tape.calls.items.len);
    const retry = fallback_tape.calls.items[1].options.?;
    try testing.expectEqual(@as(?f64, 1), retry.temperature);
    try testing.expectEqual(@as(i64, 1), retry.rollout_id.?.integer);
    for (bound_tape.calls.items) |call| {
        try testing.expectEqual(@as(?f64, 0), call.options.?.temperature);
        try testing.expect(call.options.?.rollout_id == null);
    }
    try testing.expectEqualStrings("fallback", result.program.fallback.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("bound", result.program.bound.state.demos[0].get("answer").?.string);
}

test "BootstrapFewShot scopes trace retention to teacher attempts" {
    const Metric = struct {
        threadlocal var observed: [2]u8 = undefined;
        threadlocal var count: usize = undefined;

        fn score(
            _: Allocator,
            _: *const dszig.Example,
            _: ?*const dszig.optimize.RecordPrediction,
            trace: ?*const dszig.Trace,
        ) dszig.value.ParseError!f64 {
            const steps = trace.?.steps.items;
            if (steps.len > observed.len) return error.ParseValue;
            for (steps, observed[0..steps.len]) |step, *token|
                token.* = step.inputs.get("token").?.string[0];
            count = steps.len;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const row = try dszig.Example.fromJson(arena, "{\"question\":\"q\"}", &.{"question"});
    const cases = [_]struct { caller: i64, teacher: ?i64, expected: []const u8 }{
        .{ .caller = 2, .teacher = null, .expected = "gb" },
        .{ .caller = 2, .teacher = 1, .expected = "b" },
        .{ .caller = 2, .teacher = 0, .expected = "" },
        .{ .caller = 2, .teacher = -1, .expected = "" },
        .{ .caller = 0, .teacher = 2, .expected = "gb" },
    };
    for (cases) |case| {
        var lm: dszig.lm.ScriptedLM = .{ .responses = &.{
            "<first>1</first>\n<second>2</second>",
            "<first>3</first>\n<second>4</second>",
            "<first>5</first>\n<second>6</second>",
            "<first>7</first>\n<second>8</second>",
        } };
        var trace: dszig.Trace = .{ .arena = arena };
        const ctx: dszig.Context = .{
            .workspace = dszig.workspace.forTests(),
            .arena = arena,
            .lm = lm.lm(),
            .trace = &trace,
            .max_trace_size = case.caller,
        };
        var optimizer: dszig.BootstrapFewShot = .{
            .metric = Metric.score,
            .max_bootstrapped_demos = 1,
            .max_labeled_demos = 0,
            .teacher_settings = .{ .max_trace_size = case.teacher },
        };
        Metric.count = 0;
        const result = try optimizer.compile(BootstrapTracePair, arena, ctx, .{}, null, &.{row});
        try testing.expectEqual(@as(usize, 1), result.report.bootstrapped);
        try testing.expectEqual(@as(usize, 2), lm.index);
        try testing.expectEqualStrings(case.expected, Metric.observed[0..Metric.count]);
        try testing.expectEqual(@as(usize, 0), trace.steps.items.len);
        const expected_left = std.mem.indexOfScalar(u8, case.expected, 'g') != null;
        const expected_right = std.mem.indexOfScalar(u8, case.expected, 'b') != null;
        const left = result.program.left.state.demos;
        const right = result.program.right.state.demos;
        try testing.expectEqual(@as(usize, @intFromBool(expected_left)), left.len);
        try testing.expectEqual(@as(usize, @intFromBool(expected_right)), right.len);
        if (expected_left) try testing.expectEqual(@as(i64, 1), left[0].get("first").?.integer);
        if (expected_right) try testing.expectEqual(@as(i64, 3), right[0].get("first").?.integer);

        const held_out = try result.program.forward(ctx, .{ .question = "held out" });
        try testing.expectEqual(@as(i64, 7), held_out.first);
        try testing.expectEqual(@as(usize, @intCast(case.caller)), trace.steps.items.len);
    }
}

test "BootstrapFewShot reads borrowed settings for retries and later examples" {
    const Sig = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const Program = struct {
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        compiled: bool = false,
        predict: dszig.Predict(Sig) = .{},

        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) !Output {
            return self.predict.forward(ctx, input);
        }
    };
    const Metric = struct {
        threadlocal var settings: *dszig.optimize.TeacherSettings = undefined;
        threadlocal var next: dszig.lm.LM = undefined;

        fn score(
            _: std.mem.Allocator,
            _: *const dszig.Example,
            prediction: ?*const dszig.optimize.RecordPrediction,
            _: ?*const dszig.Trace,
        ) dszig.value.ParseError!f64 {
            const answer = prediction.?.get("answer").?.string;
            if (std.mem.eql(u8, answer, "retry")) {
                settings.lm = next;
                return 0;
            }
            if (std.mem.eql(u8, answer, "accepted")) settings.* = .{};
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var initial: dszig.lm.ScriptedLM = .{ .responses = &.{"<answer>retry</answer>"} };
    var retry: dszig.lm.ScriptedLM = .{ .responses = &.{"<answer>accepted</answer>"} };
    var outer: dszig.lm.ScriptedLM = .{ .responses = &.{
        "<answer>inherited</answer>",
        "<answer>inherited</answer>",
        "<answer>inherited</answer>",
    } };
    var settings: dszig.optimize.TeacherSettings = .{ .lm = initial.lm() };
    Metric.settings = &settings;
    Metric.next = retry.lm();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = outer.lm() };
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = Metric.score,
        .max_rounds = 2,
        .max_labeled_demos = 0,
        .teacher_settings_ref = &settings,
    };
    const train = [_]dszig.Example{
        try .fromJson(arena, "{\"question\":\"one\",\"answer\":\"label\"}", &.{"question"}),
        try .fromJson(arena, "{\"question\":\"two\",\"answer\":\"label\"}", &.{"question"}),
    };
    const result = try optimizer.compile(Program, arena, ctx, .{}, null, &train);
    try testing.expectEqual(@as(usize, 3), result.report.attempts);
    try testing.expectEqual(@as(usize, 0), result.report.errors);
    const demos = result.program.predict.state.demos;
    try testing.expectEqual(@as(usize, 2), demos.len);
    try testing.expectEqualStrings("one", demos[0].get("question").?.string);
    try testing.expectEqualStrings("accepted", demos[0].get("answer").?.string);
    try testing.expectEqualStrings("two", demos[1].get("question").?.string);
    try testing.expectEqualStrings("inherited", demos[1].get("answer").?.string);

    initial.index = 0;
    retry.index = 0;
    outer.index = 0;
    settings.lm = initial.lm();
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = Metric.score,
        .max_rounds = 2,
        .max_bootstrapped_demos = 1,
        .max_labeled_demos = 0,
        .num_candidate_programs = 1,
        .teacher_settings_ref = &settings,
    };
    const ranked = try search.compile(Program, arena, testing.allocator, ctx, .{}, null, &train, train[0..1], .{ .restrict = &.{ -1, 0 } });
    try testing.expectEqualStrings("accepted", ranked.candidates[0].program.predict.state.demos[0].get("answer").?.string);
    try testing.expectEqualStrings("inherited", ranked.candidates[1].program.predict.state.demos[0].get("answer").?.string);

    initial.index = 0;
    retry.index = 0;
    outer.index = 0;
    settings.lm = initial.lm();
    const optuna: dszig.BootstrapFewShotWithOptuna = .{
        .metric = Metric.score,
        .max_rounds = 2,
        .max_labeled_demos = 0,
        .num_candidate_programs = 1,
        .seed = 1,
        .teacher_settings_ref = &settings,
    };
    const sampled = try optuna.compile(Program, arena, testing.io, ctx, .{}, null, 2, &train, train[0..1]);
    try testing.expectEqual(@as(i64, 1), sampled.parameters[0].high);
    try testing.expectEqualStrings("accepted", sampled.program.predict.state.demos[0].get("answer").?.string);
}

test "Evaluate matches DSPy scores for the RAG fixture" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const eval_json = @embedFile("fixtures/bootstrap_rag_eval.json");
    const expected = try std.json.parseFromSliceLeaky(dszig.Value, out.allocator(), eval_json, .{});
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const baseline = try dszig.optimize.evaluate(rag.RAG, ctx, &rag.RAG{}, trainset, rag.keywordMetric, 10);
    var optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), ctx, .{}, null, trainset);
    const optimized = try dszig.optimize.evaluate(rag.RAG, ctx, &result.program, trainset, rag.keywordMetric, 10);
    try testing.expectEqual(expected.object.get("baseline").?.float, baseline.score);
    try testing.expectEqual(expected.object.get("optimized").?.float, optimized.score);
}

test "the same forward pass yields identical traces" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var keyword = rag.keywordLM();
    const program: rag.RAG = .{};
    var traces: [2]dszig.Trace = .{ .{ .arena = arena }, .{ .arena = arena } };
    for (&traces) |*trace| {
        const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = keyword.lm(), .retriever = rag.corpus.retriever(), .trace = trace };
        const output = try program.forward(ctx, .{ .question = "What is the capital of France?" });
        try testing.expectEqualStrings("Paris", output.answer);
        try testing.expectEqualStrings("The context says Paris is the capital of France.", output.reasoning.?.content);
    }
    try testing.expectEqual(@as(usize, 1), traces[0].steps.items.len);
    try testing.expect(try traces[0].eql(&traces[1]));
    try testing.expectEqualStrings("What is the capital of France?", traces[0].steps.items[0].inputs.get("question").?.string);
    try testing.expectEqualStrings("The context says Paris is the capital of France.", (try dszig.Reasoning.fromValue(traces[0].steps.items[0].outputs.get("reasoning").?)).content);
}

test "compile is deterministic and does not mutate its inputs" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    var optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const student: rag.RAG = .{};
    const a = try optimizer.compile(rag.RAG, out.allocator(), ctx, student, null, trainset);
    const b = try optimizer.compile(rag.RAG, out.allocator(), ctx, student, null, trainset);
    try testing.expect(!student.compiled);
    try testing.expectEqual(@as(usize, 0), student.generate_answer.predict.state.demos.len);
    try testing.expectEqualStrings(try dszig.state.saveAlloc(out.allocator(), &a.program), try dszig.state.saveAlloc(out.allocator(), &b.program));
    try testing.expectError(error.StudentCompiled, optimizer.compile(rag.RAG, out.allocator(), ctx, a.program, null, trainset));
}

test "compiled sub-modules are frozen out of the parameter walk" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const Outer = struct {
        inner: rag.RAG = .{},
        compiled: bool = false,
        pub const Input = rag.RAG.Input;
        pub const Output = rag.RAG.Output;
        pub fn forward(self: *const @This(), ctx: dszig.Context, input: Input) anyerror!Output {
            return self.inner.forward(ctx, input);
        }
    };
    var program: Outer = .{};
    try testing.expectEqual(@as(usize, 2), (try dszig.predict.namedParameters(out.allocator(), &program)).len);
    try testing.expectEqualStrings("inner.generate_answer.predict", (try dszig.predict.namedPredictors(out.allocator(), &program))[0].name);
    program.inner.compiled = true;
    try testing.expectEqual(@as(usize, 0), (try dszig.predict.namedParameters(out.allocator(), &program)).len);
}

test "optimize, evaluate and save run in fixed buffers" {
    var out_buf: [256 * 1024]u8 = undefined;
    var scratch_buf: [128 * 1024]u8 align(16) = undefined;
    var out: std.heap.FixedBufferAllocator = .init(&out_buf);
    var workspace: dszig.Workspace = .init(&scratch_buf);
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .workspace = &workspace, .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    var optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), ctx, .{}, null, trainset);
    // Per-attempt scratch is released after every example.
    try testing.expectEqual(@as(usize, 0), workspace.used());
    const eval = try dszig.optimize.evaluate(rag.RAG, ctx, &result.program, trainset, rag.keywordMetric, 10);
    try testing.expectEqual(@as(f64, 80.0), eval.score);
    try testing.expectEqual(@as(usize, 0), workspace.used());
    _ = try dszig.state.saveAlloc(out.allocator(), &result.program);
}

test "every allocation failure point is handled without leaks" {
    // Disable address-dependent backing resize so every injected allocation
    // site is deterministic while the complete optimizer/save path still runs.
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), compileAndSave, .{});
}

fn compileAndSave(gpa: Allocator) !void {
    var out: std.heap.ArenaAllocator = .init(gpa);
    defer out.deinit();
    var keyword = rag.keywordLM();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = keyword.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    var optimizer: dszig.BootstrapFewShot = .{ .metric = rag.keywordMetric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3 };
    const result = try optimizer.compile(rag.RAG, out.allocator(), ctx, .{}, null, trainset);
    _ = try dszig.state.saveAlloc(out.allocator(), &result.program);
}

test "random search scores every seed, ranks them and keeps the best" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = rag.keywordMetric,
        .max_bootstrapped_demos = 3,
        .max_labeled_demos = 2,
        .num_candidate_programs = 3,
    };
    const result = try search.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset, null, .{ .restrict = null, .labeled_sample = true });
    try testing.expectEqual(@as(usize, 6), result.candidates.len);
    var seen: [6]bool = @splat(false);
    for (result.candidates, 0..) |c, i| {
        seen[@intCast(c.seed + 3)] = true;
        if (i > 0) try testing.expect(result.candidates[i - 1].score >= c.score);
    }
    for (seen) |s| try testing.expect(s);
    const best = try dszig.optimize.evaluate(rag.RAG, ctx, &result.program, trainset, rag.keywordMetric, 10);
    try testing.expectEqual(result.candidates[0].score, best.score);
}

test "detailed evaluation preserves successful predictions after scratch reuse" {
    const P = struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };

        pub fn forward(_: *const @This(), ctx: dszig.Context, input: Input) !Output {
            if (std.mem.eql(u8, input.question, "program_error")) return error.BadQuestion;
            return .{ .answer = try ctx.arena.dupe(u8, input.question) };
        }

        fn metric(
            _: std.mem.Allocator,
            example: *const dszig.Example,
            _: ?*const dszig.optimize.RecordPrediction,
            _: ?*const dszig.Trace,
        ) dszig.value.ParseError!f64 {
            if (std.mem.eql(u8, example.record.get("question").?.string, "metric_error"))
                return error.ParseValue;
            return 1;
        }
    };
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const inputs = &.{"question"};
    const devset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"question\":\"first\"}", inputs),
        try dszig.Example.fromJson(arena, "{\"question\":\"program_error\"}", inputs),
        try dszig.Example.fromJson(arena, "{\"question\":\"metric_error\"}", inputs),
        try dszig.Example.fromJson(arena, "{\"question\":\"last\"}", inputs),
    };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = undefined, .max_errors = 2 };
    try testing.expectError(error.ParseValue, dszig.optimize.evaluateDetailed(
        P,
        arena,
        ctx,
        &.{},
        &devset,
        P.metric,
        .{ .failure_score = -0.25 },
    ));
    const result = try dszig.optimize.evaluateDetailed(
        P,
        arena,
        ctx,
        &.{},
        &devset,
        P.metric,
        .{ .failure_score = -0.25, .max_errors = 3 },
    );
    try testing.expectEqual(@as(f64, 37.5), result.summary.score);
    try testing.expectEqual(@as(usize, 2), result.summary.errors);
    try testing.expectEqualStrings("first", result.rows[0].prediction.?.get("answer").?.string);
    try testing.expectEqualStrings("last", result.rows[3].prediction.?.get("answer").?.string);
    for (result.rows[1..3]) |row| {
        try testing.expect(row.prediction == null);
        try testing.expectEqual(@as(f64, -0.25), row.score);
    }
    const limits: dszig.parallel.Limits = .{ .workers = 2, .tasks = 4, .attempts = 4, .failures = 4 };
    const E = dszig.parallel.Evaluator(P, limits);
    const evaluator = try testing.allocator.create(E);
    defer testing.allocator.destroy(evaluator);
    const buffers = try arena.alloc([4096]u8, limits.attempts);
    var backing: [limits.attempts][]u8 = undefined;
    for (buffers, &backing) |*buffer, *slot| slot.* = buffer;
    const concurrent = blk: {
        try evaluator.init(testing.io, .{}, &backing);
        defer evaluator.deinit();
        evaluator.failure_score = -0.25;
        break :blk try evaluator.evaluateDetailed(arena, &@as(
            [limits.attempts]dszig.Context,
            @splat(ctx),
        ), &.{}, &devset, P.metric, .{ .max_errors = 3, .straggler_timeout_ns = 0 });
    };
    @memset(std.mem.sliceAsBytes(buffers), 0xaa);
    try testing.expectEqual(result.summary, concurrent.summary);
    var bytes: std.Io.Writer.Allocating = .init(testing.allocator);
    defer bytes.deinit();
    try concurrent.writeCsv(dszig.workspace.forTests(), &bytes.writer, "reward");
    try testing.expectEqualStrings(
        "question,answer,reward\r\n" ++
            "first,first,1.0\r\n" ++
            "program_error,,-0.25\r\n" ++
            "metric_error,,-0.25\r\n" ++
            "last,last,1.0\r\n",
        bytes.written(),
    );
}

test "random search with concurrent evaluation ranks the same candidates as serial evaluation" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    var search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = rag.keywordMetric,
        .max_bootstrapped_demos = 3,
        .max_labeled_demos = 2,
        .num_candidate_programs = 3,
    };
    const serial = try search.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset, null, .{ .restrict = null, .labeled_sample = true });

    const limits: dszig.parallel.Limits = .{ .workers = 3, .tasks = 8, .attempts = 8, .failures = 1 };
    const Evaluator = dszig.parallel.Evaluator(rag.RAG, limits);
    var learners: [limits.attempts]LearningLM = @splat(.{});
    var contexts: [limits.attempts]dszig.Context = undefined;
    for (&contexts, &learners) |*context, *model| context.* = .{ .workspace = dszig.workspace.forTests(), .arena = undefined, .lm = model.lm(), .retriever = rag.corpus.retriever() };
    const buffers = try out.allocator().alloc([65536]u8, limits.attempts);
    var backing: [limits.attempts][]u8 = undefined;
    for (&backing, buffers) |*slice, *buffer| slice.* = buffer;
    const evaluator = try testing.allocator.create(Evaluator);
    defer testing.allocator.destroy(evaluator);
    try evaluator.init(testing.io, .{}, &backing);
    defer evaluator.deinit();
    try testing.expectError(error.EmptyDevset, dszig.optimize.evaluate(
        rag.RAG,
        ctx,
        &.{},
        &.{},
        rag.keywordMetric,
        10,
    ));
    try testing.expectError(error.EmptyDevset, evaluator.evaluate(
        &contexts,
        &.{},
        &.{},
        rag.keywordMetric,
        .{},
    ));
    search.concurrent_evaluation = evaluator.concurrent(&contexts, .{ .straggler_timeout_ns = 0 });
    try testing.expectError(error.EmptyDevset, search.compile(
        rag.RAG,
        out.allocator(),
        testing.allocator,
        ctx,
        .{},
        null,
        &.{},
        null,
        .{ .restrict = &.{-3} },
    ));
    const concurrent = try search.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset, null, .{ .restrict = null, .labeled_sample = true });

    try testing.expectEqual(serial.candidates.len, concurrent.candidates.len);
    for (serial.candidates, concurrent.candidates) |expected, actual| {
        try testing.expectEqual(expected.seed, actual.seed);
        try testing.expectEqual(expected.score, actual.score);
        try testing.expectEqualSlices(f64, expected.subscores, actual.subscores);
    }
    const Other = struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct {};
        pub fn forward(_: *const @This(), _: dszig.Context, _: Input) !Output {
            return .{};
        }
    };
    try testing.expectError(error.ProgramTypeMismatch, dszig.optimize.evaluateWith(
        Other,
        search.concurrent_evaluation,
        ctx,
        &.{},
        trainset,
        rag.keywordMetric,
        10,
    ));
}

test "random search stops at stop_at_score" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var learner: LearningLM = .{};
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = out.allocator(), .lm = learner.lm(), .retriever = rag.corpus.retriever() };
    const trainset = try rag.trainset(out.allocator());
    const search: dszig.BootstrapFewShotWithRandomSearch = .{
        .metric = rag.keywordMetric,
        // The seed range is huge, but a passing first candidate should stop
        // without reserving space proportional to the whole range.
        .num_candidate_programs = 1 << 100,
        .stop_at_score = 0.0,
    };
    const result = try search.compile(rag.RAG, out.allocator(), testing.allocator, ctx, .{}, null, trainset, null, .{ .restrict = null, .labeled_sample = true });
    try testing.expectEqual(@as(usize, 1), result.candidates.len);
    try testing.expectEqual(@as(i64, -3), result.candidates[0].seed);
}

const FewShotOptionsSignature = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

/// Native mirror of DemoProgram in tools/dspy_reference/fewshot_options_oracle.py.
/// Every expected demo, score and call count in the tests below is that probe's
/// actual DSPy 3.5 receipt, never a value generated by native code.
const FewShotOptionsProgram = struct {
    z: dszig.Predict(FewShotOptionsSignature) = .{},
    a: dszig.Predict(FewShotOptionsSignature) = .{},
    compiled: bool = false,

    var forward_calls: u32 = 0;

    pub const Input = FewShotOptionsSignature.Input;
    pub const Output = struct { answer: []const u8, z: i64, a: i64, nz: i64, na: i64 };

    pub fn forward(self: *const FewShotOptionsProgram, ctx: dszig.Context, input: Input) !Output {
        forward_calls += 1;
        if (ctx.trace) |trace| {
            const inputs = try dszig.value.toRecord(ctx.arena, Input, input);
            const outputs = try dszig.value.toRecord(ctx.arena, FewShotOptionsSignature.Output, .{ .answer = input.question });
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

    fn firstAnswer(demos: []const dszig.value.Record) !i64 {
        if (demos.len == 0) return -1;
        return std.fmt.parseInt(i64, demos[0].get("answer").?.string, 10);
    }
};

fn fewShotOptionsMetric(_: Allocator, example: *const dszig.Example, prediction: ?*const dszig.optimize.RecordPrediction, trace: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    if (trace != null) return 1;
    const row = prediction orelse return error.ParseValue;
    const question = std.fmt.parseInt(i64, example.record.get("question").?.string, 10) catch return error.ParseValue;
    const sum = 2 * row.get("z").?.integer + row.get("a").?.integer + row.get("nz").?.integer + question;
    return @floatFromInt(@mod(sum, 7));
}

fn fewShotOptionsTiedMetric(_: Allocator, _: *const dszig.Example, _: ?*const dszig.optimize.RecordPrediction, _: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    return 1;
}

/// Bootstrap accepts a trace only when the supplied teacher actually ran.
fn fewShotOptionsTeacherMetric(_: Allocator, _: *const dszig.Example, prediction: ?*const dszig.optimize.RecordPrediction, trace: ?*const dszig.Trace) dszig.value.ParseError!f64 {
    const row = prediction orelse return error.ParseValue;
    if (trace == null) return @floatFromInt(row.get("nz").?.integer);
    return if (row.get("z").?.integer == 88 and row.get("a").?.integer == 77) 1 else 0;
}

/// Examples `question == answer == str(i)` for `i` in `0..count`.
fn fewShotOptionsExamples(arena: Allocator, count: usize) ![]dszig.Example {
    const rows = try arena.alloc(dszig.Example, count);
    for (rows, 0..) |*row, index| {
        const text = try std.fmt.allocPrint(arena, "{d}", .{index});
        var record: dszig.value.Record = .empty;
        try record.put(arena, "question", .{ .string = text });
        try record.put(arena, "answer", .{ .string = text });
        row.* = .{ .record = record, .input_keys = &.{"question"} };
    }
    return rows;
}

fn fewShotOptionsDemo(arena: Allocator, answer: []const u8) ![]const dszig.value.Record {
    const demos = try arena.alloc(dszig.value.Record, 1);
    demos[0] = .empty;
    try demos[0].put(arena, "question", .{ .string = "preset" });
    try demos[0].put(arena, "answer", .{ .string = answer });
    return demos;
}

/// The student's preset demo must be discarded by every successful compile.
fn fewShotOptionsStudent(arena: Allocator) !FewShotOptionsProgram {
    var student: FewShotOptionsProgram = .{};
    student.z.state.demos = try fewShotOptionsDemo(arena, "99");
    return student;
}

fn expectFewShotDemos(program: *const FewShotOptionsProgram, z: []const []const u8, a: []const []const u8) !void {
    try testing.expectEqual(z.len, program.z.state.demos.len);
    try testing.expectEqual(a.len, program.a.state.demos.len);
    for (z, program.z.state.demos) |expected, demo| try testing.expectEqualStrings(expected, demo.get("answer").?.string);
    for (a, program.a.state.demos) |expected, demo| try testing.expectEqualStrings(expected, demo.get("answer").?.string);
}

const FewShotCandidate = struct { seed: i64, score: f64 };

fn expectFewShotCandidates(expected: []const FewShotCandidate, actual: anytype) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| {
        try testing.expectEqual(want.seed, got.seed);
        try testing.expectEqual(want.score, got.score);
    }
}

fn fewShotOptionsSearch(metric: dszig.optimize.Metric) dszig.BootstrapFewShotWithRandomSearch {
    return .{ .metric = metric, .max_bootstrapped_demos = 2, .max_labeled_demos = 3, .num_candidate_programs = 4 };
}

test "random search inherits the outer budget for bootstrap and candidate evaluation" {
    const Metric = struct {
        fn score(
            _: Allocator,
            example: *const dszig.Example,
            _: ?*const dszig.optimize.RecordPrediction,
            _: ?*const dszig.Trace,
        ) dszig.value.ParseError!f64 {
            const question = example.record.get("question").?.string;
            if (std.mem.eql(u8, question, "0")) return error.ParseValue;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const P = FewShotOptionsProgram;
    const trainset = try fewShotOptionsExamples(arena, 2);
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .max_errors = 1 };
    var search = fewShotOptionsSearch(Metric.score);
    search.teacher_settings.max_errors = 9;
    for ([_]i64{ -3, -1 }) |seed| {
        const options: dszig.BootstrapFewShotWithRandomSearch.Options = .{ .restrict = &.{seed} };
        const devset = if (seed == -3) trainset else trainset[1..];
        search.max_errors = null;
        P.forward_calls = 0;
        try testing.expectError(
            error.ParseValue,
            search.compile(P, arena, testing.allocator, ctx, .{}, null, trainset, devset, options),
        );
        try testing.expectEqual(@as(u32, 1), P.forward_calls);

        search.max_errors = 2;
        P.forward_calls = 0;
        const result = try search.compile(
            P,
            arena,
            testing.allocator,
            ctx,
            .{},
            null,
            trainset,
            devset,
            options,
        );
        const score: f64 = if (seed == -3) 50 else 100;
        try expectFewShotCandidates(&.{.{ .seed = seed, .score = score }}, result.candidates);
        try testing.expectEqual(@as(u32, if (seed == -3) 2 else 3), P.forward_calls);
    }
}

test "LabeledFewShot sample shares one RNG across predictors and prefix mode owns the trainset head" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    var inputs: std.heap.ArenaAllocator = .init(testing.allocator);
    var inputs_live = true;
    defer if (inputs_live) inputs.deinit();
    const student = try fewShotOptionsStudent(inputs.allocator());
    const trainset = try fewShotOptionsExamples(inputs.allocator(), 6);
    const optimizer: dszig.LabeledFewShot = .{ .k = 3 };

    const sampled = try optimizer.compile(FewShotOptionsProgram, out.allocator(), student, trainset, .{ .sample = true });
    const prefix = try optimizer.compile(FewShotOptionsProgram, out.allocator(), student, trainset, .{ .sample = false });
    // Compiled demos must not borrow the caller's trainset or student.
    inputs.deinit();
    inputs_live = false;
    // The second predictor continues the first predictor's random.Random(0) stream.
    try expectFewShotDemos(&sampled, &.{ "3", "5", "0" }, &.{ "2", "4", "3" });
    try expectFewShotDemos(&prefix, &.{ "0", "1", "2" }, &.{ "0", "1", "2" });
}

test "LabeledFewShot negative k slices like Python in prefix mode and fails only when sampling draws" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const student = try fewShotOptionsStudent(arena);
    const trainset = try fewShotOptionsExamples(arena, 6);
    const all = [_][]const u8{ "0", "1", "2", "3", "4", "5" };
    const prefix_cases = [_]struct { k: i128, count: usize }{
        .{ .k = 0, .count = 0 },
        .{ .k = 1, .count = 1 },
        .{ .k = 6, .count = 6 },
        .{ .k = 16, .count = 6 },
        .{ .k = 1 << 63, .count = 6 },
        .{ .k = -1, .count = 5 },
        .{ .k = -2, .count = 4 },
        .{ .k = -5, .count = 1 },
        .{ .k = -6, .count = 0 },
        .{ .k = -7, .count = 0 },
        .{ .k = -100, .count = 0 },
        .{ .k = std.math.minInt(i128), .count = 0 },
    };
    for (prefix_cases) |case| {
        const optimizer: dszig.LabeledFewShot = .{ .k = case.k };
        const compiled = try optimizer.compile(FewShotOptionsProgram, arena, student, trainset, .{ .sample = false });
        try expectFewShotDemos(&compiled, all[0..case.count], all[0..case.count]);
    }
    for ([_]i128{ -1, -6, -7, std.math.minInt(i128) }) |k| {
        const optimizer: dszig.LabeledFewShot = .{ .k = k };
        try testing.expectError(error.InvalidSampleCount, optimizer.compile(FewShotOptionsProgram, arena, student, trainset, .{ .sample = true }));
        // Source returns the reset copy before reaching rng.sample for an empty trainset.
        const reset = try optimizer.compile(FewShotOptionsProgram, arena, student, &.{}, .{ .sample = true });
        try expectFewShotDemos(&reset, &.{}, &.{});
    }
    const wide: dszig.LabeledFewShot = .{ .k = 1 << 63 };
    const sampled = try wide.compile(FewShotOptionsProgram, arena, student, trainset, .{ .sample = true });
    try expectFewShotDemos(&sampled, &.{ "3", "5", "0", "1", "2", "4" }, &.{ "2", "3", "5", "4", "0", "1" });

    // Source validates the count inside its predictor loop: no predictor, no failure.
    const Bare = struct {
        compiled: bool = false,
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
        pub fn forward(_: *const @This(), _: dszig.Context, input: Input) !Output {
            return .{ .answer = input.question };
        }
    };
    const negative: dszig.LabeledFewShot = .{ .k = -1 };
    _ = try negative.compile(Bare, arena, .{}, trainset, .{ .sample = true });
}

// Actual DSPy 3.5 zero-bootstrap observations: fixed seeds remain usable,
// while a random seed consumes an empty randint range only when reached.
test "random search validates zero bootstrap range only for consumed random seeds" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = silent.lm() };
    const student = try fewShotOptionsStudent(arena);
    const trainset = try fewShotOptionsExamples(arena, 6);
    const valset = try fewShotOptionsExamples(arena, 5);
    const P = FewShotOptionsProgram;
    var search = fewShotOptionsSearch(fewShotOptionsMetric);
    search.max_bootstrapped_demos = 0;

    const fixed = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ -3, -2, -1 },
        .labeled_sample = true,
    });
    try expectFewShotCandidates(&.{
        .{ .seed = -3, .score = 320 },
        .{ .seed = -2, .score = 320 },
        .{ .seed = -1, .score = 320 },
    }, fixed.candidates);
    try expectFewShotDemos(&fixed.program, &.{}, &.{});
    P.forward_calls = 0;
    const bootstrap = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{-1},
        .labeled_sample = true,
    });
    try expectFewShotDemos(&bootstrap.program, &.{ "0", "3", "4" }, &.{ "1", "5", "0" });
    try testing.expectEqual(@as(u32, 5), P.forward_calls);

    search.stop_at_score = 0;
    const stopped = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = null,
        .labeled_sample = true,
    });
    try expectFewShotCandidates(&.{.{ .seed = -3, .score = 320 }}, stopped.candidates);
    search.stop_at_score = null;
    P.forward_calls = 0;
    try testing.expectError(error.InvalidBootstrapRange, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{0},
        .labeled_sample = true,
    }));
    try testing.expectEqual(@as(u32, 0), P.forward_calls);
    try testing.expectError(error.InvalidBootstrapRange, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ -3, -2, -1, 0 },
        .labeled_sample = true,
    }));
    try testing.expectEqual(@as(u32, 15), P.forward_calls);
    P.forward_calls = 0;
    try testing.expectError(error.RestrictMatchesNoSeed, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{},
        .labeled_sample = true,
    }));
    try testing.expectEqual(@as(u32, 0), P.forward_calls);
}

test "random search restrict is a seed set applied before any candidate is built" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = silent.lm() };
    const student = try fewShotOptionsStudent(arena);
    const trainset = try fewShotOptionsExamples(arena, 6);
    const valset = try fewShotOptionsExamples(arena, 5);
    const P = FewShotOptionsProgram;
    var search = fewShotOptionsSearch(fewShotOptionsMetric);

    // Duplicates and out-of-range seeds are ignored; order follows the seed range.
    P.forward_calls = 0;
    const filtered = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ 3, -3, 3, -3, 99, -99, 4, -4 },
        .labeled_sample = true,
    });
    try expectFewShotCandidates(&.{ .{ .seed = 3, .score = 360 }, .{ .seed = -3, .score = 320 } }, filtered.candidates);
    try expectFewShotDemos(&filtered.program, &.{ "0", "1", "4" }, &.{ "0", "5", "2" });
    try testing.expectEqual(@as(u32, 11), P.forward_calls);

    // No matching seed fails before any program runs, including an empty set.
    P.forward_calls = 0;
    for ([_][]const i64{ &.{}, &.{ 4, -4, 100 } }) |restrict| {
        try testing.expectError(error.RestrictMatchesNoSeed, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
            .restrict = restrict,
            .labeled_sample = true,
        }));
    }
    try testing.expectEqual(@as(u32, 0), P.forward_calls);

    // The early stop sees only selected seeds: seed 2 is the first to reach 360.
    const stopped = block: {
        var stopping = search;
        stopping.stop_at_score = 360;
        break :block try stopping.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
            .restrict = &.{ 3, 2, 0, -3 },
            .labeled_sample = true,
        });
    };
    try expectFewShotCandidates(&.{
        .{ .seed = 2, .score = 400 },
        .{ .seed = -3, .score = 320 },
        .{ .seed = 0, .score = 300 },
    }, stopped.candidates);
    try expectFewShotDemos(&stopped.program, &.{ "2", "0", "5" }, &.{ "2", "4", "3" });

    // Equal scores keep seed order and the first selected candidate wins.
    search.metric = fewShotOptionsTiedMetric;
    const tied = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ 2, 0, -2, 3 },
        .labeled_sample = true,
    });
    try expectFewShotCandidates(&.{
        .{ .seed = -2, .score = 100 },
        .{ .seed = 0, .score = 100 },
        .{ .seed = 2, .score = 100 },
        .{ .seed = 3, .score = 100 },
    }, tied.candidates);
    try expectFewShotDemos(&tied.program, &.{ "3", "5", "0" }, &.{ "2", "4", "3" });

    // DSPy's range(-3, n) also excludes fixed seeds when n is negative.
    search.metric = fewShotOptionsMetric;
    for ([_]i128{ -1, -2 }, [_]usize{ 2, 1 }) |limit, count| {
        search.num_candidate_programs = limit;
        P.forward_calls = 0;
        const negative = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{});
        try expectFewShotCandidates((&[_]FewShotCandidate{
            .{ .seed = -3, .score = 320 },
            .{ .seed = -2, .score = 320 },
        })[0..count], negative.candidates);
        try expectFewShotDemos(&negative.program, &.{}, &.{});
        try testing.expectEqual(count * valset.len, P.forward_calls);
    }
    search.num_candidate_programs = -3;
    P.forward_calls = 0;
    try testing.expectError(error.NoCandidates, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{}));
    try testing.expectError(error.RestrictMatchesNoSeed, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{ .restrict = &.{-3} }));
    try testing.expectEqual(@as(u32, 0), P.forward_calls);

    // With no shuffled seeds only -3..-1 can match.
    search.metric = fewShotOptionsMetric;
    search.num_candidate_programs = 0;
    try testing.expectError(error.RestrictMatchesNoSeed, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{0},
        .labeled_sample = true,
    }));
    const fixed = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ -1, -2, -3, 0 },
        .labeled_sample = true,
    });
    try expectFewShotCandidates(&.{
        .{ .seed = -1, .score = 360 },
        .{ .seed = -3, .score = 320 },
        .{ .seed = -2, .score = 320 },
    }, fixed.candidates);
}

test "random search evaluates sparse late seeds in range order without changing the caller's list" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena };
    const P = FewShotOptionsProgram;
    const student = try fewShotOptionsStudent(arena);
    const trainset = try fewShotOptionsExamples(arena, 6);
    const valset = trainset[0..1];

    const end = std.math.maxInt(i64);
    const earlier = end - (1 << 40);
    const latest = end - 1;
    const original = [_]i64{ latest, end, earlier, -3, latest, -4, earlier, -3 };
    var seeds = original;
    const options: dszig.BootstrapFewShotWithRandomSearch.Options = .{ .restrict = &seeds };
    var search = fewShotOptionsSearch(fewShotOptionsTeacherMetric);
    search.num_candidate_programs = end;
    search.max_rounds = 0;
    search.max_labeled_demos = 1;

    P.forward_calls = 0;
    try testing.expectError(error.RestrictMatchesNoSeed, search.compile(P, arena, testing.failing_allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ end, -4 },
    }));
    try testing.expectError(error.OutOfMemory, search.compile(P, arena, testing.failing_allocator, ctx, student, null, trainset, valset, options));
    try testing.expectEqual(@as(u32, 0), P.forward_calls);
    try testing.expectEqualSlices(i64, &original, &seeds);

    // The reset candidate scores 0; the two late candidates tie at 100.
    // Ranking keeps the tie in evaluation order, not the caller's seed order.
    const ranked = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, options);
    try expectFewShotCandidates(&.{
        .{ .seed = earlier, .score = 100 },
        .{ .seed = latest, .score = 100 },
        .{ .seed = -3, .score = 0 },
    }, ranked.candidates);
    // DSPy uses the full integer seed for shuffling, not just its low word.
    try expectFewShotDemos(&ranked.candidates[0].program, &.{"5"}, &.{"5"});
    try expectFewShotDemos(&ranked.candidates[1].program, &.{"1"}, &.{"1"});
    try expectFewShotDemos(&ranked.program, &.{"5"}, &.{"5"});
    try testing.expectEqual(@as(u32, 3), P.forward_calls);
    try testing.expectEqualSlices(i64, &original, &seeds);

    search.stop_at_score = 100;
    P.forward_calls = 0;
    const stopped = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, options);
    try expectFewShotCandidates(&.{
        .{ .seed = earlier, .score = 100 },
        .{ .seed = -3, .score = 0 },
    }, stopped.candidates);
    try expectFewShotDemos(&stopped.program, &.{"5"}, &.{"5"});
    try testing.expectEqual(@as(u32, 2), P.forward_calls);
    try testing.expectEqualSlices(i64, &original, &seeds);

    // A bound larger than i64 includes the largest representable native seed.
    search.num_candidate_programs = 1 << 100;
    search.stop_at_score = null;
    P.forward_calls = 0;
    const wide = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{
        .restrict = &.{ end, -3 },
    });
    try expectFewShotCandidates(&.{
        .{ .seed = end, .score = 100 },
        .{ .seed = -3, .score = 0 },
    }, wide.candidates);
    try testing.expectEqual(@as(u32, 2), P.forward_calls);
    try testing.expectError(error.CandidateCapacityExceeded, search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{}));
    try testing.expectEqual(@as(u32, 2), P.forward_calls);
}

test "random search labeled_sample changes only the labeled seed" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = silent.lm() };
    const student = try fewShotOptionsStudent(arena);
    const trainset = try fewShotOptionsExamples(arena, 6);
    const valset = try fewShotOptionsExamples(arena, 5);
    const P = FewShotOptionsProgram;
    const search = fewShotOptionsSearch(fewShotOptionsMetric);

    const sampled = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{ .restrict = null, .labeled_sample = true });
    try expectFewShotCandidates(&.{
        .{ .seed = 1, .score = 400 },  .{ .seed = 2, .score = 400 },
        .{ .seed = -1, .score = 360 }, .{ .seed = 3, .score = 360 },
        .{ .seed = -3, .score = 320 }, .{ .seed = -2, .score = 320 },
        .{ .seed = 0, .score = 300 },
    }, sampled.candidates);
    // Only seed -2 moves: its prefix demos score 360 instead of 320.
    const prefix = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{ .restrict = null, .labeled_sample = false });
    try expectFewShotCandidates(&.{
        .{ .seed = 1, .score = 400 },  .{ .seed = 2, .score = 400 },
        .{ .seed = -2, .score = 360 }, .{ .seed = -1, .score = 360 },
        .{ .seed = 3, .score = 360 },  .{ .seed = -3, .score = 320 },
        .{ .seed = 0, .score = 300 },
    }, prefix.candidates);
    try expectFewShotDemos(&prefix.program, &.{ "2", "1", "4" }, &.{ "2", "0", "3" });

    const labeled = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{ .restrict = &.{-2}, .labeled_sample = false });
    try expectFewShotCandidates(&.{.{ .seed = -2, .score = 360 }}, labeled.candidates);
    try expectFewShotDemos(&labeled.program, &.{ "0", "1", "2" }, &.{ "0", "1", "2" });

    // Bootstrap seeds still sample their labeled demos when labeled_sample is false.
    const bootstrap = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{ .restrict = &.{ -3, -1, 1 }, .labeled_sample = false });
    try expectFewShotCandidates(&.{
        .{ .seed = 1, .score = 400 },
        .{ .seed = -1, .score = 360 },
        .{ .seed = -3, .score = 320 },
    }, bootstrap.candidates);
    try expectFewShotDemos(&bootstrap.program, &.{ "2", "1", "4" }, &.{ "2", "0", "3" });

    // A missing or empty valset evaluates on the trainset.
    const fallback = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, &.{}, .{ .restrict = &.{ -3, -2 }, .labeled_sample = false });
    try expectFewShotCandidates(&.{ .{ .seed = -2, .score = 316.67 }, .{ .seed = -3, .score = 300 } }, fallback.candidates);
}

test "random search consumes its teacher only at a selected bootstrap seed" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var silent: dszig.lm.ScriptedLM = .{ .responses = &.{} };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = silent.lm() };
    const student = try fewShotOptionsStudent(arena);
    const trainset = try fewShotOptionsExamples(arena, 6);
    const valset = try fewShotOptionsExamples(arena, 5);
    const P = FewShotOptionsProgram;
    var search = fewShotOptionsSearch(fewShotOptionsMetric);
    const teacher_list: []const P = &.{.{}};
    const teacher_none: []const P = &.{};

    // A teacher list is a Source failure only where BootstrapFewShot deep-copies it.
    P.forward_calls = 0;
    const unconsumed = try search.compile(P, arena, testing.allocator, ctx, student, teacher_list, trainset, valset, .{ .restrict = &.{ -3, -2 }, .labeled_sample = true });
    try expectFewShotCandidates(&.{ .{ .seed = -3, .score = 320 }, .{ .seed = -2, .score = 320 } }, unconsumed.candidates);
    try testing.expectEqual(@as(u32, 10), P.forward_calls);
    const empty = try search.compile(P, arena, testing.allocator, ctx, student, teacher_none, trainset, valset, .{ .restrict = &.{-2}, .labeled_sample = false });
    try expectFewShotDemos(&empty.program, &.{ "0", "1", "2" }, &.{ "0", "1", "2" });

    // Seed -2 is fully evaluated before seed -1 meets the list.
    P.forward_calls = 0;
    try testing.expectError(error.OptimizerTeacherListUnsupported, search.compile(P, arena, testing.allocator, ctx, student, teacher_list, trainset, valset, .{ .restrict = &.{ -2, -1 }, .labeled_sample = true }));
    try testing.expectEqual(@as(u32, 5), P.forward_calls);
    P.forward_calls = 0;
    try testing.expectError(error.OptimizerTeacherListUnsupported, search.compile(P, arena, testing.allocator, ctx, student, teacher_list, trainset, valset, .{ .restrict = &.{1}, .labeled_sample = true }));
    try testing.expectEqual(@as(u32, 0), P.forward_calls);

    // An early stop at the zero-shot seed never reaches the list either.
    search.stop_at_score = 0;
    P.forward_calls = 0;
    const early = try search.compile(P, arena, testing.allocator, ctx, student, teacher_list, trainset, valset, .{ .restrict = null, .labeled_sample = true });
    try expectFewShotCandidates(&.{.{ .seed = -3, .score = 320 }}, early.candidates);
    try testing.expectEqual(@as(u32, 5), P.forward_calls);

    // A real teacher is run by the selected bootstrap seeds: only its preset
    // demos satisfy the trace metric, so the first two examples bootstrap.
    search.stop_at_score = null;
    search.metric = fewShotOptionsTeacherMetric;
    var teacher: P = .{ .compiled = true };
    teacher.z.state.demos = try fewShotOptionsDemo(arena, "88");
    teacher.a.state.demos = try fewShotOptionsDemo(arena, "77");
    P.forward_calls = 0;
    const taught = try search.compile(P, arena, testing.allocator, ctx, student, teacher, trainset, valset, .{ .restrict = &.{ -1, 0 }, .labeled_sample = true });
    try expectFewShotCandidates(&.{ .{ .seed = -1, .score = 300 }, .{ .seed = 0, .score = 300 } }, taught.candidates);
    try expectFewShotDemos(&taught.program, &.{ "0", "1", "5" }, &.{ "0", "1", "5" });
    try testing.expectEqual(@as(u32, 14), P.forward_calls);
    // Without it the student copy teaches and nothing bootstraps.
    P.forward_calls = 0;
    const untaught = try search.compile(P, arena, testing.allocator, ctx, student, null, trainset, valset, .{ .restrict = &.{ -1, 0 }, .labeled_sample = true });
    try expectFewShotDemos(&untaught.program, &.{ "0", "3", "4" }, &.{ "1", "5", "0" });
    try testing.expectEqual(@as(u32, 22), P.forward_calls);
}

test "BetterTogether keys and default prompt carry typed compile options" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const P = FewShotOptionsProgram;
    const optimizers = dszig.better_together.optimizers;
    var model: dszig.lm.OpenAI = .{ .client = undefined, .base_url = "http://unused", .model = "base" };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = model.lm() };
    const trainset = try fewShotOptionsExamples(arena, 6);
    const meta: dszig.BetterTogether = .{
        .metric = fewShotOptionsMetric,
        .limits = .{ .max_steps = 4, .max_examples = 8, .max_predictors = 2, .max_teachers = 2, .max_strategy_bytes = 64 },
    };
    const options: dszig.better_together.Options = .{
        .seed = 9,
        .valset_ratio = 0,
        .shuffle_trainset_between_steps = false,
        .strategy = "labeled -> search",
    };

    // Each registry key owns its compile options: Python-sliced prefix labels,
    // then a search restricted to its own prefix-labeled seed.
    var labeled: optimizers.LabeledFewShot = .{ .optimizer = .{ .k = -2 }, .compile_options = .{ .sample = false } };
    var search: optimizers.RandomSearch = .{
        .optimizer = fewShotOptionsSearch(fewShotOptionsMetric),
        .compile_options = .{ .restrict = &.{ -2, 7 }, .labeled_sample = false },
    };
    var student = try fewShotOptionsStudent(arena);
    try dszig.predict.setLM(arena, &student, model.lm());
    const keyed = try meta.compileWith(P, arena, testing.allocator, testing.io, ctx, &student, trainset, options, .{
        .labeled = &labeled,
        .search = &search,
    });
    try testing.expect(!keyed.flag_compilation_error_occurred);
    try testing.expectEqual(@as(usize, 3), keyed.candidate_programs.len);
    try expectFewShotDemos(&keyed.candidate_programs[1].program, &.{ "0", "1", "2", "3" }, &.{ "0", "1", "2", "3" });
    try expectFewShotDemos(&keyed.program, &.{ "0", "1", "2" }, &.{ "0", "1", "2" });

    // The default prompt optimizer receives the same typed options.
    var prompt_student = try fewShotOptionsStudent(arena);
    try dszig.predict.setLM(arena, &prompt_student, model.lm());
    var prompt_options = options;
    prompt_options.strategy = "p";
    var defaults: dszig.better_together.Defaults = .{
        .weight_limits = .{ .max_teachers = 1, .max_examples = 8, .max_predictors = 2, .data = .{ .max_rows = 8, .max_trace_steps = 16 } },
        .weight_slot_sets = &.{},
        .prompt = fewShotOptionsSearch(fewShotOptionsMetric),
        .prompt_compile_options = .{ .restrict = &.{-2}, .labeled_sample = false },
    };
    const prompted = try meta.compile(P, arena, testing.allocator, testing.io, ctx, &prompt_student, trainset, prompt_options, defaults);
    try testing.expect(!prompted.flag_compilation_error_occurred);
    try expectFewShotDemos(&prompted.program, &.{ "0", "1", "2" }, &.{ "0", "1", "2" });

    // A restriction matching no seed fails that step before any candidate runs.
    defaults.prompt_compile_options = .{ .restrict = &.{4}, .labeled_sample = false };
    P.forward_calls = 0;
    const refused = try meta.compile(P, arena, testing.allocator, testing.io, ctx, &prompt_student, trainset, prompt_options, defaults);
    try testing.expect(refused.flag_compilation_error_occurred);
    try testing.expectEqual(@as(usize, 1), refused.candidate_programs.len);
    try testing.expectEqual(@as(u32, 0), P.forward_calls);
}

test "Chat Completions retain native reasoning without inventing absent summaries" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const json =
        \\{"choices":[
        \\  {"message":{"content":"first","reasoning_content":"Actual summary."}},
        \\  {"message":{"content":"second"}},
        \\  {"message":{"content":"third","reasoning_content":null}},
        \\  {"message":{"content":"fourth","reasoning_content":""}},
        \\  {"message":{"content":"fifth","reasoning_content":42}},
        \\  {"message":{"content":"sixth","reasoning_content":[]}},
        \\  {"message":{"content":"seventh","reasoning_content":["bad"]}}
        \\]}
    ;
    const body = try arena.dupe(u8, json);
    const completions = try dszig.lm.OpenAI.parseCompletions(arena, body);
    // Provider response buffers are released after parsing.
    @memset(body, 'x');
    try testing.expectEqual(@as(usize, 7), completions.len);
    try testing.expectEqualStrings("first", completions[0].text.?);
    try testing.expectEqualStrings("Actual summary.", completions[0].reasoning.?);
    try testing.expectEqualStrings("second", completions[1].text.?);
    try testing.expect(completions[1].reasoning == null);
    try testing.expectEqualStrings("third", completions[2].text.?);
    try testing.expect(completions[2].reasoning == null);
    try testing.expectEqualStrings("fourth", completions[3].text.?);
    try testing.expect(completions[3].reasoning == null);
    try testing.expectEqualStrings("42", completions[4].reasoning.?);
    try testing.expect(completions[5].reasoning == null);
    try testing.expectEqualStrings("['bad']", completions[6].reasoning.?);

    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseCompletions(arena, "{\"choices\":[]}"));
}

test "Responses parse assistant output after reasoning and preserve real summary parts" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const json =
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[
        \\    {"type":"summary_text","text":"Locate the capital."},
        \\    {"type":"summary_text","text":"Use the supplied fact."}
        \\  ]},
        \\  {"type":"web_search_call","status":"completed"},
        \\  {"type":"message","role":"assistant","status":"completed","content":[
        \\    {"type":"output_text","text":"<answer>\n"},
        \\    {"type":"output_text","text":"Paris\n</answer>"}
        \\  ]}
        \\]}
    ;
    const body = try arena.dupe(u8, json);
    const completion = try dszig.lm.OpenAI.parseResponsesResponse(arena, body);
    @memset(body, 'x');
    try testing.expectEqualStrings("<answer>\nParis\n</answer>", completion.text.?);
    try testing.expectEqualStrings("Locate the capital.\nUse the supplied fact.", completion.reasoning.?);

    const no_summary = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[]},
        \\  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
        \\]}
    );
    try testing.expectEqualStrings("answer", no_summary.text.?);
    try testing.expect(no_summary.reasoning == null);

    const empty_summary = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[{"type":"summary_text","text":""}]},
        \\  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
        \\]}
    );
    try testing.expect(empty_summary.reasoning == null);
}

test "Responses citation spans count Python characters and affect canonical call identifiers" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const completion = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[
        \\ {"type":"message","content":[{"type":"output_text","text":"A😀B","annotations":[
        \\   {"start_index":"١","end_index":2.9,"url":"https://example.org","title":"Unicode"},
        \\   {"quote":"fact","url":false,"uri":false,"title":"","file_id":0},
        \\   {"start_index":true,"end_index":2}
        \\ ]}]},
        \\ {"type":"function_call","name":"probe","arguments":"{}"}
        \\]}
    );
    try testing.expectEqual(@as(usize, 2), completion.citations.len);
    try testing.expectEqualStrings("😀", completion.citations[0].text.?);
    try testing.expectEqualStrings("https://example.org", completion.citations[0].url.?);
    try testing.expectEqualStrings("Unicode", completion.citations[0].title.?);
    try testing.expectEqualStrings("fact", completion.citations[1].text.?);
    try testing.expectEqualStrings("False", completion.citations[1].url.?);
    try testing.expectEqualStrings("0", completion.citations[1].title.?);
    try testing.expectEqualStrings("call_3", completion.tool_calls[0].id);
    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"text","text":"x","annotations":[{"start_index":Infinity}]}]}]}
    ));
    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"text","text":"x","annotations":[{"quote":"quoted","start_index":Infinity}]}]}]}
    ));
    const quoted = try dszig.lm.OpenAI.parseResponsesResponse(arena,
        \\{"output":[{"type":"message","content":[{"type":"text","text":"x","annotations":[{"quote":"\ud800"}]}]}]}
    );
    try testing.expectEqualStrings("\xed\xa0\x80", quoted.citations[0].text.?);
}

test "Chat text blocks preserve canonical indices and reasoning-only responses have no text" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const blocks = try dszig.lm.OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":[{"type":"text","text":""},{"type":"text","text":"b"}],
        \\ "tool_calls":[{"type":"function","function":{"name":"probe","arguments":"{}"}}]}}]}
    );
    try testing.expectEqualStrings("b", blocks[0].text.?);
    try testing.expectEqualStrings("call_2", blocks[0].tool_calls[0].id);
    const reasoning = try dszig.lm.OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":null,"reasoning_content":"","reasoning":"fallback reasoning",
        \\ "tool_calls":[{"type":"function","function":{"name":"probe","arguments":"{}"}}]}}]}
    );
    try testing.expect(reasoning[0].text == null);
    try testing.expectEqualStrings("fallback reasoning", reasoning[0].reasoning.?);
    try testing.expectEqualStrings("call_1", reasoning[0].tool_calls[0].id);
}

test "citation completion clones unwind every failed allocation" {
    const Clone = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const payload = try dszig.sdk_response.JsonValue.fromValue(allocator, .{ .object = .empty });
            defer payload.free();
            try payload.root.object.put(payload.storage.?.allocator(), "opaque", .{ .string = "state" });
            const completion = try dszig.lm.cloneCompletion(allocator, .{
                .text = "answer",
                .citations = &.{
                    .{
                        .text = "quoted",
                        .title = "source",
                        .url = "https://example.org",
                        .continuation = &.{.{
                            .provider = "fixture",
                            .kind = "citation",
                            .data = payload,
                        }},
                    },
                    .{ .url = "https://example.net" },
                },
            });
            defer dszig.lm.freeCompletion(allocator, completion);
            try testing.expectEqualStrings("quoted", completion.citations[0].text.?);
            try payload.root.object.put(payload.storage.?.allocator(), "opaque", .{ .string = "changed" });
            try testing.expectEqualStrings("state", completion.citations[0].continuation[0].data.root.object.get("opaque").?.string);
            try testing.expectEqualStrings("https://example.net", completion.citations[1].url.?);
        }
    };
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), Clone.run, .{});
}

test "Responses distinguish absent text from empty output and retain fallback call identifiers" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    // Actual DSPy 3.5 LM HTTP observations, not provider status assumptions.
    const cases = [_]struct { body: []const u8, text: ?[]const u8 }{
        .{ .body = "{\"output\":null}", .text = "" },
        .{ .body = "{\"output\":[],\"output_text\":\"fallback\"}", .text = "fallback" },
        .{ .body = "{\"status\":\"incomplete\",\"output\":[{\"type\":\"message\",\"role\":\"user\",\"status\":\"incomplete\",\"content\":[{\"type\":\"output_text\",\"text\":\"partial\"}]}]}", .text = "partial" },
        .{ .body = "{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"No.\"}]}]}", .text = null },
        .{ .body = "{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"first\"},{\"type\":\"text\",\"text\":\"second\"}]}]}", .text = "firstsecond" },
        .{ .body = "{\"output\":[{\"type\":\"function_call\",\"name\":\"probe\",\"arguments\":\"{}\"}]}", .text = null },
    };
    for (cases) |case| {
        const completion = try dszig.lm.OpenAI.parseResponsesResponse(arena, case.body);
        if (case.text) |text| try testing.expectEqualStrings(text, completion.text.?) else try testing.expect(completion.text == null);
        const restored = try dszig.lm.completionFromValue(arena, try dszig.lm.completionValue(arena, completion));
        if (case.text) |text| try testing.expectEqualStrings(text, restored.text.?) else try testing.expect(restored.text == null);
    }
    const tool_only = try dszig.lm.OpenAI.parseResponsesResponse(arena, cases[cases.len - 1].body);
    try testing.expectEqualStrings("call_0", tool_only.tool_calls[0].id);
    try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseResponsesResponse(arena, "{\"error\":{\"code\":\"fixture_error\",\"message\":\"source fixture error\"},\"output_text\":\"not a successful response\"}"));
}

test "Chat replays assistant reasoning as text without dropping answers or native calls" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const native_calls = [_]dszig.lm.ToolCall{.{ .id = "call_thought", .name = "lookup", .arguments = "{ \"query\": \"Paris\" }" }};
    const Case = struct {
        reasoning: ?[]const u8,
        content: []const u8,
        expected: []const u8,
        calls: []const dszig.lm.ToolCall = &.{},
    };
    const cases = [_]Case{
        .{ .reasoning = "thought", .content = "answer", .expected = "thought\nanswer" },
        .{ .reasoning = "", .content = "answer", .expected = "answer" },
        .{ .reasoning = null, .content = "answer", .expected = "answer" },
        .{ .reasoning = "thought", .content = "answer", .expected = "thought\nanswer", .calls = &native_calls },
    };
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    const provider: dszig.lm.OpenAI = .{ .client = &client, .base_url = "http://unused", .model = "wrong-bound-model" };
    for (cases) |case| {
        var parts: std.ArrayList(dszig.lm.sdk.Part) = .empty;
        if (case.reasoning) |reasoning| try parts.append(arena, .{ .thinking = .{ .text = reasoning } });
        try parts.append(arena, .{ .text = .{ .text = case.content } });
        for (case.calls) |call| {
            const input = try std.json.parseFromSliceLeaky(dszig.Value, arena, call.arguments, .{});
            try parts.append(arena, .{ .tool_call = .{
                .id = call.id,
                .name = call.name,
                .input = try dszig.lm.sdk.JsonValue.fromBorrowedValue(arena, input),
            } });
        }
        const request = try dszig.lm.sdk.Request.fromFields(arena, &.{
            .model = "m",
            .messages = &.{.{ .role = .assistant, .parts = parts.items }},
        });
        defer request.release();
        const wire = try provider.requestBody(arena, request);
        const root = try std.json.parseFromSliceLeaky(dszig.Value, arena, wire.body, .{});
        const projected = root.object.get("messages").?.array.items;
        try testing.expectEqual(@as(usize, 1), projected.len);
        const message = projected[0].object;
        try testing.expectEqualStrings("assistant", message.get("role").?.string);
        try testing.expectEqualStrings(case.expected, message.get("content").?.string);
        try testing.expect(!message.contains("reasoning") and !message.contains("reasoning_content"));
        if (case.calls.len == 0) {
            try testing.expect(!message.contains("tool_calls"));
        } else {
            const calls = message.get("tool_calls").?.array.items;
            try testing.expectEqual(case.calls.len, calls.len);
            for (case.calls, calls) |want, got| {
                try testing.expectEqualStrings(want.id, got.object.get("id").?.string);
                try testing.expectEqualStrings("function", got.object.get("type").?.string);
                const function = got.object.get("function").?.object;
                try testing.expectEqualStrings(want.name, function.get("name").?.string);
                const expected = try std.json.parseFromSliceLeaky(dszig.Value, arena, want.arguments, .{});
                const actual = try std.json.parseFromSliceLeaky(dszig.Value, arena, function.get("arguments").?.string, .{});
                try testing.expect(try dszig.value.recordEqual(&expected.object, &actual.object));
            }
        }
    }
}

test "typed prediction uses DSPy Unicode field names and strict schema titles" {
    const Task = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { @"éclair": []const u8, answer_text: []const u8, @"ßeta": []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const cases = [_]struct { adapter: dszig.Adapter, completion: []const u8 }{
        .{ .adapter = .xml, .completion = "<éclair>\npastry\n</éclair>\n<answer_text>\nplain\n</answer_text>\n<ßeta>\nsharp\n</ßeta>" },
        .{ .adapter = .chat, .completion = "[[ ## éclair ## ]]\npastry\n[[ ## answer_text ## ]]\nplain\n[[ ## ßeta ## ]]\nsharp\n[[ ## completed ## ]]" },
        .{ .adapter = .json, .completion = "{\"éclair\":\"pastry\",\"answer_text\":\"plain\",\"ßeta\":\"sharp\"}" },
    };
    for (cases) |case| {
        var model: dszig.lm.ScriptedLM = .{
            .responses = &.{case.completion},
            .supports_response_format = true,
            .supports_response_schema = true,
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = model.lm(), .arena = arena };
        const program: dszig.Predict(Task) = .{};
        const result = try program.forward(.{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = recording.lm(), .adapter = case.adapter }, .{ .question = "q" });
        try testing.expectEqualStrings("pastry", result.@"éclair");
        try testing.expectEqualStrings("plain", result.answer_text);
        try testing.expectEqualStrings("sharp", result.@"ßeta");
        try testing.expectEqual(@as(usize, 1), recording.calls.items.len);
        if (case.adapter == .json) {
            const format = recording.calls.items[0].options.?.response_format.?;
            try testing.expectEqualStrings("DSPyProgramOutputs", format.json_schema.name);
            const properties = format.json_schema.schema.object.get("properties").?.object;
            for ([_][]const u8{ "éclair", "answer_text", "ßeta" }, [_][]const u8{ "Éclair", "Answer Text", "Sseta" }) |key, title| {
                const property = properties.get(key).?.object;
                try testing.expectEqualStrings(title, property.get("title").?.string);
                try testing.expectEqualStrings("string", property.get("type").?.string);
            }
        }
    }
}

test "recording metadata roundtrip and replay preserve rich completions and exact requests" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const parameters = try std.json.parseFromSliceLeaky(dszig.Value, arena,
        \\{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
    , .{});
    const output_schema = try std.json.parseFromSliceLeaky(dszig.Value, arena,
        \\{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
    , .{});
    const tool_calls = [_]dszig.lm.ToolCall{.{ .id = "call_lookup", .name = "lookup", .arguments = "{ \"query\": \"Paris\" }" }};
    var scripted: dszig.lm.ScriptedLM = .{
        .responses = &.{ "", "{\"answer\":\"Berlin\"}" },
        .reasonings = &.{ "Actual summary.", null },
        .tool_calls = &.{ &tool_calls, &.{} },
        .supports_reasoning = true,
        .supports_response_format = true,
        .supports_response_schema = true,
        .supports_function_calling = true,
    };
    var recording: dszig.lm.RecordingLM = .{ .inner = scripted.lm(), .arena = arena };
    const request: dszig.lm.Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "question" } }},
        .tools = &.{.{ .name = "lookup", .description = "Look up a fact.", .parameters = parameters, .strict = true }},
        .options = .{
            .n = 2,
            .temperature = 0.7,
            .rollout_id = .{ .integer = 4 },
            .reasoning_effort = "low",
            .response_format = .{ .json_schema = .{ .name = "TaskOutput", .schema = output_schema } },
            .tool_choice = .{ .function = "lookup" },
            .parallel_tool_calls = false,
        },
    };
    try testing.expect(recording.lm().supports_reasoning);
    try testing.expect(recording.lm().supports_response_format);
    try testing.expect(recording.lm().supports_response_schema);
    try testing.expect(recording.lm().supports_function_calling);
    const completed = try recording.lm().completeResponses(arena, request);
    try testing.expectEqual(@as(usize, 2), completed.len);
    try testing.expectEqualStrings("Actual summary.", completed[0].reasoning.?);
    try testing.expect(completed[1].reasoning == null);
    try testing.expectEqual(@as(usize, 1), completed[0].tool_calls.len);
    try testing.expectEqualStrings("call_lookup", completed[0].tool_calls[0].id);
    try testing.expectEqualStrings("lookup", completed[0].tool_calls[0].name);
    try testing.expectEqualStrings(tool_calls[0].arguments, completed[0].tool_calls[0].arguments);
    try testing.expectEqual(@as(usize, 0), completed[1].tool_calls.len);
    try testing.expectEqual(@as(usize, 2), recording.calls.items.len);
    try testing.expectEqual(@as(usize, 1), recording.calls.items[0].tools.len);
    try testing.expectEqualStrings("lookup", recording.calls.items[0].tools[0].name);

    var used = [_]bool{ false, false };
    var replay: dszig.lm.ReplayLM = .{
        .calls = recording.calls.items,
        .used = &used,
        .supports_reasoning = true,
        .supports_response_format = true,
        .supports_response_schema = true,
        .supports_function_calling = true,
    };
    try testing.expect(replay.lm().supports_reasoning);
    try testing.expect(replay.lm().supports_response_format);
    try testing.expect(replay.lm().supports_response_schema);
    try testing.expect(replay.lm().supports_function_calling);
    var changed = request;
    changed.options.reasoning_effort = "high";
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.options.response_format = .text;
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.options.tool_choice = .none;
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.options.parallel_tool_calls = true;
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    changed = request;
    changed.tools = &.{};
    try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, changed));
    try testing.expect(!used[0]);
    const restored_request = try dszig.lm.requestFromValue(arena, try dszig.lm.requestValue(arena, request));
    const replayed = try replay.lm().completeResponses(arena, restored_request);
    try testing.expectEqual(@as(usize, 2), replayed.len);
    for (completed, replayed) |want, got| {
        const restored = try dszig.lm.completionFromValue(arena, try dszig.lm.completionValue(arena, want));
        if (restored.text) |text| try testing.expectEqualStrings(text, got.text.?) else try testing.expect(got.text == null);
        if (restored.reasoning) |summary| try testing.expectEqualStrings(summary, got.reasoning.?) else try testing.expect(got.reasoning == null);
        try testing.expectEqual(restored.tool_calls.len, got.tool_calls.len);
        for (restored.tool_calls, got.tool_calls) |want_call, got_call| {
            try testing.expectEqualStrings(want_call.id, got_call.id);
            try testing.expectEqualStrings(want_call.name, got_call.name);
            try testing.expectEqualStrings(want_call.arguments, got_call.arguments);
        }
    }
    try testing.expect(used[0] and used[1]);
    try testing.expectEqual(@as(usize, 5), replay.misses);
}

test "native function-only completions preserve identifiers and normalize Responses arguments" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const chat_body = try arena.dupe(u8,
        \\{"choices":[{"message":{"content":null,"reasoning_content":"Look up the fact.","tool_calls":[
        \\  {"id":"call_chat","type":"function","function":{"name":"lookup","arguments":"{ \"query\": \"Paris\", \"limit\": 2 }"}}
        \\]}}]}
    );
    const chat = try dszig.lm.OpenAI.parseCompletions(arena, chat_body);
    @memset(chat_body, 'x');
    try testing.expectEqual(@as(usize, 1), chat.len);
    try testing.expect(chat[0].text == null);
    try testing.expectEqualStrings("Look up the fact.", chat[0].reasoning.?);
    try testing.expectEqual(@as(usize, 1), chat[0].tool_calls.len);
    try testing.expectEqualStrings("call_chat", chat[0].tool_calls[0].id);
    try testing.expectEqualStrings("lookup", chat[0].tool_calls[0].name);

    const responses_body = try arena.dupe(u8,
        \\{"status":"completed","output":[
        \\  {"type":"reasoning","summary":[{"type":"summary_text","text":"Look up the fact."}]},
        \\  {"type":"function_call","id":"fc_item","call_id":"call_responses","status":"completed",
        \\   "name":"lookup","arguments":"{ \"query\": \"Paris\", \"limit\": 2 }"}
        \\]}
    );
    const responses = try dszig.lm.OpenAI.parseResponsesResponse(arena, responses_body);
    @memset(responses_body, 'x');
    try testing.expect(responses.text == null);
    try testing.expectEqualStrings("Look up the fact.", responses.reasoning.?);
    try testing.expectEqual(@as(usize, 1), responses.tool_calls.len);
    try testing.expectEqualStrings("call_responses", responses.tool_calls[0].id);
    try testing.expectEqualStrings("lookup", responses.tool_calls[0].name);
    try testing.expectEqualStrings("{\"query\": \"Paris\", \"limit\": 2}", responses.tool_calls[0].arguments);
}

const SubmitDetails = struct { city: []const u8, zip_code: i64 };

const StructuredSubmit = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct {
        answer: []const u8,
        details: SubmitDetails,
        ids: []const i64,
        note: ?[]const u8,
        label: enum { pos, neg },
        count: u8,
    };
});

test "ReAct typed submit returns every nested list nullable enum and bounded output" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{
        \\<next_thought>
        \\Submit the result.
        \\</next_thought>
        \\<tool_calls>
        \\{"tool_calls":[{"name":"submit","args":{"answer":"Paris","details":{"city":"Paris","zip_code":9},"ids":[1,2],"note":null,"label":"pos","count":255}}]}
        \\</tool_calls>
    } };
    var agent = try dszig.ReAct(StructuredSubmit, &.{}).init(arena);
    agent.max_iters = 1;
    const output = try agent.forward(.{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = scripted.lm() }, .{ .question = "Return the result." });
    try testing.expectEqualStrings("Paris", output.answer);
    try testing.expectEqualStrings("Paris", output.details.city);
    try testing.expectEqual(@as(i64, 9), output.details.zip_code);
    try testing.expectEqualSlices(i64, &.{ 1, 2 }, output.ids);
    try testing.expect(output.note == null);
    try testing.expect(output.label == .pos);
    try testing.expectEqual(@as(u8, 255), output.count);
    try testing.expectEqual(@as(usize, 1), scripted.index);
}

test "ReAct rejects malformed typed final values before returning any output" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var agent = try dszig.ReAct(StructuredSubmit, &.{}).init(arena);
    agent.max_iters = 0;
    const inputs = try dszig.value.toRecord(arena, StructuredSubmit.Input, .{ .question = "Return the result." });
    for ([_][]const u8{
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":256}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":-1}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":1.5}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":42,\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\",\"count\":7}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":\"not a list\",\"note\":null,\"label\":\"pos\",\"count\":7}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"other\",\"count\":7}",
        "{\"answer\":\"Paris\",\"details\":{\"city\":\"Paris\",\"zip_code\":9},\"ids\":[1,2],\"note\":null,\"label\":\"pos\"}",
    }) |args| {
        const completion = try std.mem.concat(arena, u8, &.{
            "<next_thought>\nSubmit the result.\n</next_thought>\n<tool_calls>\n{\"tool_calls\":[{\"name\":\"submit\",\"args\":",
            args,
            "}]}\n</tool_calls>",
        });
        var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{completion} };
        var history: dszig.History = .{};
        try testing.expectError(error.ReActSubmitFailed, agent.runWithOptions(.{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = scripted.lm() }, &inputs, .{ .history = &history }));
        try testing.expectEqual(@as(usize, 1), scripted.index);
        try testing.expectEqual(@as(usize, 1), history.messages.len);
        const calls = try dszig.react.ToolCalls.fromValue(arena, history.messages[0].get("tool_calls").?);
        const results = calls.tool_call_results.?.tool_call_results;
        try testing.expectEqual(@as(usize, 1), results.len);
        try testing.expect(results[0].is_error);
        try testing.expectEqualStrings("Execution error in submit: InvalidArguments", results[0].value.string);
        try testing.expect(!history.messages[0].contains("answer"));
    }
}

test "ReAct reuses structured History across calls without replacing prior events" {
    const QA = dszig.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    var requests: dszig.lm.History = .{ .arena = arena };
    var scripted: dszig.lm.ScriptedLM = .{ .responses = &.{
        "<next_thought>\nFirst result.\n</next_thought>\n<tool_calls>\n{\"tool_calls\":[{\"id\":\"submit-paris\",\"name\":\"submit\",\"args\":{\"answer\":\"Paris\"}}]}\n</tool_calls>",
        "<next_thought>\nSecond result.\n</next_thought>\n<tool_calls>\n{\"tool_calls\":[{\"id\":\"submit-berlin\",\"name\":\"submit\",\"args\":{\"answer\":\"Berlin\"}}]}\n</tool_calls>",
    }, .history = &requests };
    const ctx: dszig.Context = .{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = scripted.lm() };
    var agent = try dszig.ReAct(QA, &.{}).init(arena);
    agent.max_iters = 1;
    var history: dszig.History = .{};
    const first_inputs = try dszig.value.toRecord(arena, QA.Input, .{ .question = "First question?" });
    const first = try agent.runWithOptions(ctx, &first_inputs, .{ .history = &history });
    try testing.expectEqualStrings("Paris", first.outputs.get("answer").?.string);
    try testing.expectEqual(dszig.react.TerminationReason.submit, first.termination_reason);
    try testing.expectEqual(@as(usize, 1), history.messages.len);
    const saved_first = try std.json.Stringify.valueAlloc(arena, dszig.Value{ .object = history.messages[0] }, .{});

    const second_inputs = try dszig.value.toRecord(arena, QA.Input, .{ .question = "Second question?" });
    const second = try agent.runWithOptions(ctx, &second_inputs, .{ .history = &history });
    try testing.expectEqualStrings("Berlin", second.outputs.get("answer").?.string);
    try testing.expectEqual(dszig.react.TerminationReason.submit, second.termination_reason);
    try testing.expectEqual(@as(usize, 2), history.messages.len);
    try testing.expectEqual(@as(usize, 1), first.history.messages.len);
    try testing.expectEqualStrings(saved_first, try std.json.Stringify.valueAlloc(arena, dszig.Value{ .object = history.messages[0] }, .{}));
    try testing.expectEqualStrings("First question?", history.messages[0].get("question").?.string);
    try testing.expectEqualStrings("Second question?", history.messages[1].get("question").?.string);
    try testing.expectEqualStrings("Paris", history.messages[0].get("answer").?.string);
    try testing.expectEqualStrings("Berlin", history.messages[1].get("answer").?.string);
    const first_calls = try dszig.react.ToolCalls.fromValue(arena, history.messages[0].get("tool_calls").?);
    const second_calls = try dszig.react.ToolCalls.fromValue(arena, history.messages[1].get("tool_calls").?);
    try testing.expectEqualStrings("submit-paris", first_calls.tool_calls[0].id.?);
    try testing.expectEqualStrings("submit-berlin", second_calls.tool_calls[0].id.?);
    try testing.expectEqual(@as(usize, 2), requests.requests.items.len);
    try testing.expect(requests.requests.items[1].messages.len > requests.requests.items[0].messages.len);
    var saw_first = false;
    var saw_second = false;
    var saw_first_answer = false;
    for (requests.requests.items[1].messages) |message| {
        saw_first = saw_first or message.content.contains("First question?");
        saw_second = saw_second or message.content.contains("Second question?");
        saw_first_answer = saw_first_answer or message.content.contains("Paris");
    }
    try testing.expect(saw_first and saw_second and saw_first_answer);
}

test "LM fans out actual n1 candidate requests and replay preserves strict rollout identity" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    const messages = [_]dszig.adapter.Message{.{ .role = .user, .content = .{ .text = "Look up the facts." } }};
    const parameters = try std.json.parseFromSliceLeaky(dszig.Value, arena, "{\"type\":\"object\"}", .{});
    const tools = [_]dszig.lm.FunctionTool{.{ .name = "lookup", .parameters = parameters }};
    const texts = [_][]const u8{ "first", "second", "third" };
    const summaries = [_]?[]const u8{ "First summary.", null, "" };
    const first_calls = [_]dszig.lm.ToolCall{.{ .id = "call_first", .name = "lookup", .arguments = "{ \"query\": \"first\" }" }};
    const third_calls = [_]dszig.lm.ToolCall{.{ .id = "call_third", .name = "lookup", .arguments = "{\"query\":\"third\"}" }};
    const native_calls = [_][]const dszig.lm.ToolCall{ &first_calls, &.{}, &third_calls };
    const array_rollout = try std.json.parseFromSliceLeaky(dszig.Value, arena, "[\"session\",7]", .{});
    const bases = [_]?dszig.Value{ null, .{ .integer = 4 }, .{ .string = "base-run" }, array_rollout };
    for (bases) |base| {
        var scripted: dszig.lm.ScriptedLM = .{
            .responses = &texts,
            .reasonings = &summaries,
            .tool_calls = &native_calls,
            .supports_reasoning = true,
            .supports_function_calling = true,
        };
        var recording: dszig.lm.RecordingLM = .{ .inner = scripted.lm(), .arena = arena };
        const request: dszig.lm.Request = .{
            .messages = &messages,
            .tools = &tools,
            .options = .{ .n = 3, .temperature = 0.7, .reasoning_effort = "low", .rollout_id = base },
        };
        const generated = try recording.lm().completeResponses(arena, request);
        try testing.expectEqual(@as(usize, 3), generated.len);
        try testing.expectEqual(@as(usize, 3), scripted.index);
        try testing.expectEqual(@as(usize, 3), recording.calls.items.len);
        try testing.expectEqual(@as(usize, 3), dszig.lm.ReplayLM.completionCount(recording.calls.items));
        for (recording.calls.items, 0..) |call, index| {
            const options = call.options.?;
            try testing.expectEqual(@as(u32, 1), options.n);
            try testing.expectEqual(@as(f64, 0.7), options.temperature.?);
            try testing.expectEqualStrings("low", options.reasoning_effort.?);
            const rollout = options.rollout_id.?.array.items;
            try testing.expectEqual(@as(usize, 3), rollout.len);
            try testing.expectEqualStrings("dspy.adapter.candidate", rollout[0].string);
            try testing.expectEqualStrings(
                try std.json.Stringify.valueAlloc(arena, base orelse .null, .{}),
                try std.json.Stringify.valueAlloc(arena, rollout[1], .{}),
            );
            try testing.expectEqual(@as(i64, @intCast(index)), rollout[2].integer);
            try testing.expectEqual(@as(usize, 1), call.all().len);
            try testing.expectEqualStrings(texts[index], call.completion.?);
            try testing.expectEqualStrings(texts[index], generated[index].text.?);
        }

        var untagged_used = [_]bool{ false, false, false };
        var untagged: dszig.lm.ReplayLM = .{ .calls = recording.calls.items, .used = &untagged_used };
        var wrong_request = request;
        wrong_request.options.n = 1;
        try testing.expectError(error.LMFailure, untagged.lm().completeResponses(arena, wrong_request));
        try testing.expectEqualSlices(bool, &.{ false, false, false }, &untagged_used);
        try testing.expectEqual(@as(usize, 1), untagged.misses);

        var used = [_]bool{ false, false, false };
        var replay: dszig.lm.ReplayLM = .{
            .calls = recording.calls.items,
            .used = &used,
            .supports_reasoning = true,
            .supports_function_calling = true,
        };
        const completions = try replay.lm().completeResponses(arena, request);
        try testing.expectEqual(@as(usize, 3), completions.len);
        for (completions, 0..) |got, index| {
            try testing.expectEqualStrings(texts[index], got.text.?);
            if (summaries[index]) |summary| try testing.expectEqualStrings(summary, got.reasoning.?) else try testing.expect(got.reasoning == null);
            try testing.expectEqual(native_calls[index].len, got.tool_calls.len);
            for (native_calls[index], got.tool_calls) |want_call, got_call| {
                try testing.expectEqualStrings(want_call.id, got_call.id);
                try testing.expectEqualStrings(want_call.name, got_call.name);
                try testing.expectEqualStrings(want_call.arguments, got_call.arguments);
            }
        }
        try testing.expect(replay.exhausted());
        try testing.expectError(error.LMFailure, replay.lm().completeResponses(arena, request));
        try testing.expectEqual(@as(usize, 1), replay.misses);
    }
}

test "tool JSON repair requires raw ID provenance and preserves legitimate partial_json" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const completions = try dszig.lm.OpenAI.parseCompletions(arena,
        \\{"choices":[{"message":{"content":null,"tool_calls":[
        \\ {"id":"real","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"Paris\""}},
        \\ {"type":"function","function":{"name":"lookup","arguments":"{\"query\":\"Paris\""}},
        \\ {"id":"legitimate","type":"function","function":{"name":"lookup","arguments":"{\"partial_json\":\"{bad\"}"}}
        \\]}}]}
    );
    const calls = completions[0].tool_calls;
    try testing.expectEqualStrings("{\"query\": \"Paris\"}", calls[0].arguments);
    try testing.expectEqualStrings("call_1", calls[1].id);
    try testing.expectEqualStrings("{\"partial_json\": \"{\\\"query\\\":\\\"Paris\\\"\"}", calls[1].arguments);
    try testing.expectEqualStrings("{\"partial_json\": \"{bad\"}", calls[2].arguments);
}

test "native provider calls reject unnamed functions" {
    var out: std.heap.ArenaAllocator = .init(testing.allocator);
    defer out.deinit();
    const arena = out.allocator();
    for ([_][]const u8{
        "{\"choices\":[{\"message\":{\"content\":null,\"tool_calls\":[{\"id\":\"call\",\"type\":\"function\",\"function\":{\"arguments\":\"{}\"}}]}}]}",
    }) |invalid| {
        try testing.expectError(error.LMFailure, dszig.lm.OpenAI.parseCompletions(arena, invalid));
    }
}

const FinetuneSignature = dszig.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8, count: i64 };
});

const finetune_view: dszig.adapter.View = .{
    .instructions = "Answer the question.",
    .inputs = FinetuneSignature.input_fields_list,
    .outputs = FinetuneSignature.output_fields_list,
};

const finetune_adapters = [_]dszig.Adapter{ .chat, .xml };

fn finetuneMessages(training: dszig.Value) ![]const dszig.Value {
    try testing.expectEqual(@as(usize, 1), training.object.count());
    return training.object.get("messages").?.array.items;
}

fn expectFinetuneMessage(message: dszig.Value, role: []const u8, content: []const u8) !void {
    try testing.expectEqual(@as(usize, 2), message.object.count());
    try testing.expectEqualStrings(role, message.object.get("role").?.string);
    try testing.expectEqualStrings(content, message.object.get("content").?.string);
}

// Expected strings are DSPy 3.5.0 `format_finetune_data` results for the same
// signature and records: a training answer renders every declared output, and
// an absent one is Python None rather than the demo placeholder or "".
test "finetune data renders missing and null training outputs as None" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var inputs: dszig.value.Record = .empty;
    try inputs.put(arena, "question", .{ .string = "Why?" });
    var complete: dszig.value.Record = .empty;
    try complete.put(arena, "answer", .{ .string = "because\n<yes>" });
    try complete.put(arena, "count", .{ .integer = 3 });
    var missing: dszig.value.Record = .empty;
    try missing.put(arena, "count", .{ .integer = 3 });
    var explicit_null: dszig.value.Record = .empty;
    try explicit_null.put(arena, "answer", .null);
    try explicit_null.put(arena, "count", .null);
    const absent: dszig.value.Record = .empty;

    const users = [finetune_adapters.len][]const u8{
        "[[ ## question ## ]]\nWhy?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, then `[[ ## count ## ]]` (must be formatted as a valid Python int), and then ending with the marker for `[[ ## completed ## ]]`.",
        "<question>\nWhy?\n</question>\n\nOutput format (replace placeholders):\n<answer>\n{answer}\n</answer>\n\n<count>\n{count}\n</count>\n\nUse this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.",
    };
    const outputs = [_]*const dszig.value.Record{ &complete, &missing, &explicit_null, &absent };
    const assistants = [finetune_adapters.len][outputs.len][]const u8{
        .{
            "[[ ## answer ## ]]\nbecause\n<yes>\n\n[[ ## count ## ]]\n3\n\n[[ ## completed ## ]]\n",
            "[[ ## answer ## ]]\nNone\n\n[[ ## count ## ]]\n3\n\n[[ ## completed ## ]]\n",
            "[[ ## answer ## ]]\nNone\n\n[[ ## count ## ]]\nNone\n\n[[ ## completed ## ]]\n",
            "[[ ## answer ## ]]\nNone\n\n[[ ## count ## ]]\nNone\n\n[[ ## completed ## ]]\n",
        },
        .{
            "<answer>\nbecause\n<yes>\n</answer>\n\n<count>\n3\n</count>",
            "<answer />\n\n<count>\n3\n</count>",
            "<answer />\n\n<count />",
            "<answer />\n\n<count />",
        },
    };
    for (finetune_adapters, users, assistants) |adapter, user, expected| {
        const prompt = try adapter.format(arena, finetune_view, &.{}, &inputs);
        for (outputs, expected) |record, assistant| {
            const count_before = record.count();
            const training = try adapter.formatFinetuneData(
                arena,
                testing.io,
                finetune_view,
                &.{},
                &inputs,
                record,
            );
            const messages = try finetuneMessages(training);
            try testing.expectEqual(@as(usize, 3), messages.len);
            // The training prompt is the ordinary native prompt, unchanged.
            try expectFinetuneMessage(messages[0], "system", prompt[0].content.text);
            try expectFinetuneMessage(messages[1], "user", user);
            try expectFinetuneMessage(messages[2], "assistant", assistant);
            // Completing a partial record must not write into the caller's record.
            try testing.expectEqual(count_before, record.count());
        }
    }
}

// Demos keep their own "Not supplied" placeholders; only the training answer
// uses None. Expected strings are the DSPy 3.5.0 results for these records.
test "finetune data keeps demo placeholders and appends the training answer" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var past: dszig.value.Record = .empty;
    try past.put(arena, "question", .{ .string = "past" });
    try past.put(arena, "answer", .{ .string = "old" });
    try past.put(arena, "count", .{ .integer = 1 });
    var partial: dszig.value.Record = .empty;
    try partial.put(arena, "question", .{ .string = "partial" });
    try partial.put(arena, "answer", .{ .string = "half" });
    var inputs: dszig.value.Record = .empty;
    try inputs.put(arena, "question", .{ .string = "now" });
    var outputs: dszig.value.Record = .empty;
    try outputs.put(arena, "answer", .{ .string = "new" });
    try outputs.put(arena, "count", .{ .integer = 2 });

    const expected = [finetune_adapters.len][5][]const u8{
        .{
            "This is an example of the task, though some input or output fields are not supplied.\n\n[[ ## question ## ]]\npartial",
            "[[ ## answer ## ]]\nhalf\n\n[[ ## count ## ]]\nNot supplied for this particular example.\n\n[[ ## completed ## ]]\n",
            "[[ ## question ## ]]\npast",
            "[[ ## answer ## ]]\nold\n\n[[ ## count ## ]]\n1\n\n[[ ## completed ## ]]\n",
            "[[ ## answer ## ]]\nnew\n\n[[ ## count ## ]]\n2\n\n[[ ## completed ## ]]\n",
        },
        .{
            "This is an example of the task, though some input or output fields are not supplied.\n\n<question>\npartial\n</question>",
            "<answer>\nhalf\n</answer>\n\n<count>\nNot supplied for this particular example. \n</count>",
            "<question>\npast\n</question>",
            "<answer>\nold\n</answer>\n\n<count>\n1\n</count>",
            "<answer>\nnew\n</answer>\n\n<count>\n2\n</count>",
        },
    };
    for (finetune_adapters, expected) |adapter, contents| {
        const training = try adapter.formatFinetuneData(
            arena,
            testing.io,
            finetune_view,
            &.{ past, partial },
            &inputs,
            &outputs,
        );
        const messages = try finetuneMessages(training);
        try testing.expectEqual(@as(usize, 7), messages.len);
        try expectFinetuneMessage(messages[1], "user", contents[0]);
        try expectFinetuneMessage(messages[2], "assistant", contents[1]);
        try expectFinetuneMessage(messages[3], "user", contents[2]);
        try expectFinetuneMessage(messages[4], "assistant", contents[3]);
        try testing.expectEqualStrings("user", messages[5].object.get("role").?.string);
        try expectFinetuneMessage(messages[6], "assistant", contents[4]);
    }
}

test "prediction replays tagged REPL observations as inert text" {
    const Action = dszig.Signature(struct {
        pub const Input = struct { iteration: []const u8, history: dszig.History };
        pub const Output = struct { reasoning: []const u8, code: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const history = try dszig.History.fromValue(arena, try dszig.json_repair.parseJson(arena,
        \\{"messages":[{"iteration":"1/3","repl_entry":{"__dspy_repl__":"entry",
        \\"reasoning":"inspect","code":"print()","max_output_chars":0,
        \\"output":"<<CUSTOM-TYPE-START-IDENTIFIER>>[]<<CUSTOM-TYPE-END-IDENTIFIER>>"}},
        \\{"iteration":"2/3","repl_entry":{"__dspy_repl__":"entry",
        \\"code":"print()","max_output_chars":-1,"output":"雪é🙂xyz"}}]}
    ));
    var requests: dszig.lm.History = .{ .arena = arena };
    var lm: dszig.lm.KeywordLM = .{
        .rules = &.{},
        .default = "<reasoning>done</reasoning><code>submit()</code>",
        .history = &requests,
    };
    var program: dszig.Predict(Action) = .{};
    const output = try program.forward(.{ .workspace = dszig.workspace.forTests(), .arena = arena, .lm = lm.lm() }, .{
        .iteration = "3/3",
        .history = history,
    });
    try testing.expectEqualStrings("submit()", output.code);
    const messages = requests.requests.items[0].messages;
    try testing.expectEqual(@as(usize, 8), messages.len);
    try testing.expectEqualStrings(
        "<repl_output>\n雪é🙂xy\n\n... (7 characters omitted) ...\n\né🙂xyz\n</repl_output>",
        messages[6].content.text,
    );
    try testing.expectEqualStrings(
        "<reasoning>\ninspect\n</reasoning>\n\n<code>\nprint()\n</code>",
        messages[2].content.text,
    );
    try testing.expect(messages[3].role == .user);
    switch (messages[3].content) {
        .text => |text| try testing.expectEqualStrings(
            "<repl_output>\n\n\n... (64 characters omitted) ...\n\n" ++
                "[CUSTOM-TYPE-START][][CUSTOM-TYPE-END]\n</repl_output>",
            text,
        ),
        else => return error.TestUnexpectedResult,
    }
}

// Actual DSPy retains all declared answer fields in ordinary history, including
// an event with no answer. Dropping those fields changed the training dialogue.
test "finetune history retains absent answers as None before the current answer" {
    const Chat = dszig.Signature(struct {
        pub const Input = struct { question: []const u8, history: dszig.History };
        pub const Output = struct { answer: []const u8, count: i64 };
    });
    const view: dszig.adapter.View = .{
        .instructions = "Answer the question.",
        .inputs = Chat.input_fields_list,
        .outputs = Chat.output_fields_list,
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var complete: dszig.value.Record = .empty;
    try complete.put(arena, "question", .{ .string = "before" });
    try complete.put(arena, "answer", .{ .string = "yes" });
    try complete.put(arena, "count", .{ .integer = 4 });
    var incomplete: dszig.value.Record = .empty;
    try incomplete.put(arena, "question", .{ .string = "half" });
    const history: dszig.History = .{ .messages = &.{ complete, incomplete } };
    const inputs = try dszig.value.toRecord(arena, Chat.Input, .{
        .question = "now",
        .history = history,
    });
    const outputs = try dszig.value.toRecord(arena, Chat.Output, .{
        .answer = "new",
        .count = 5,
    });
    const expected = [finetune_adapters.len][3][]const u8{
        .{
            "[[ ## answer ## ]]\nyes\n\n[[ ## count ## ]]\n4\n\n[[ ## completed ## ]]\n",
            "[[ ## answer ## ]]\nNone\n\n[[ ## count ## ]]\nNone\n\n[[ ## completed ## ]]\n",
            "[[ ## answer ## ]]\nnew\n\n[[ ## count ## ]]\n5\n\n[[ ## completed ## ]]\n",
        },
        .{
            "<answer>\nyes\n</answer>\n\n<count>\n4\n</count>",
            "<answer />\n\n<count />",
            "<answer>\nnew\n</answer>\n\n<count>\n5\n</count>",
        },
    };
    for (finetune_adapters, expected) |adapter, answers| {
        const training = try adapter.formatFinetuneData(
            arena,
            testing.io,
            view,
            &.{},
            &inputs,
            &outputs,
        );
        const messages = try finetuneMessages(training);
        try testing.expectEqual(@as(usize, 7), messages.len);
        for (answers, [_]usize{ 2, 4, 6 }) |answer, index| {
            try expectFinetuneMessage(messages[index], "assistant", answer);
        }
    }
    try testing.expectEqual(@as(usize, 1), incomplete.count());
    try testing.expect(!incomplete.contains("answer"));
    try testing.expect(!incomplete.contains("count"));
}

// The SDK training projection must retain native calls and their paired result
// IDs. Source uses null assistant content when the native call has no text.
test "finetune native tool history preserves calls result IDs and null content" {
    const Tools = dszig.Signature(struct {
        pub const Input = struct { question: []const u8, history: dszig.History };
        pub const Output = struct { answer: ?[]const u8, calls: dszig.ToolCalls };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var args: dszig.value.Record = .empty;
    try args.put(arena, "q", .{ .string = "x" });
    var result: dszig.value.Record = .empty;
    try result.put(arena, "answer", .{ .integer = 3 });
    const calls: dszig.ToolCalls = .{
        .tool_calls = &.{.{ .id = "call_1", .name = "lookup", .args = args }},
        .tool_call_results = .{ .tool_call_results = &.{.{
            .call_id = "call_1",
            .name = "lookup",
            .value = .{ .object = result },
            .is_error = true,
        }} },
    };
    var event: dszig.value.Record = .empty;
    try event.put(arena, "question", .{ .string = "before" });
    try event.put(arena, "answer", .null);
    try event.put(arena, "calls", try calls.toValue(arena));
    const inputs = try dszig.value.toRecord(arena, Tools.Input, .{
        .question = "now",
        .history = .{ .messages = &.{event} },
    });
    const outputs = try dszig.value.toRecord(arena, Tools.Output, .{
        .answer = "done",
        .calls = .{},
    });
    for (finetune_adapters) |adapter| {
        const training = try adapter.formatFinetuneData(arena, testing.io, .{
            .instructions = "Use recorded tools.",
            .inputs = Tools.input_fields_list,
            .outputs = Tools.output_fields_list,
            .use_native_function_calling = true,
        }, &.{}, &inputs, &outputs);
        const messages = try finetuneMessages(training);
        try testing.expectEqual(@as(usize, 6), messages.len);
        const assistant = messages[2].object;
        try testing.expectEqualStrings("assistant", assistant.get("role").?.string);
        try testing.expect(assistant.get("content").? == .null);
        const native_calls = assistant.get("tool_calls").?.array.items;
        try testing.expectEqual(@as(usize, 1), native_calls.len);
        try testing.expectEqualStrings("call_1", native_calls[0].object.get("id").?.string);
        try testing.expectEqualStrings("function", native_calls[0].object.get("type").?.string);
        const function = native_calls[0].object.get("function").?.object;
        try testing.expectEqualStrings("lookup", function.get("name").?.string);
        try testing.expectEqualStrings("{\"q\":\"x\"}", function.get("arguments").?.string);
        const tool_result = messages[3].object;
        try testing.expectEqualStrings("tool", tool_result.get("role").?.string);
        try testing.expectEqualStrings("call_1", tool_result.get("tool_call_id").?.string);
        try testing.expectEqualStrings("[error] {\"answer\": 3}", tool_result.get("content").?.string);
    }
}

// The native prompt keeps canonical document parts; only the training record
// carries the OpenAI file block DSPy 3.5.0 writes for the same File input.
test "finetune data projects File inputs to provider blocks, not canonical parts" {
    const Files = dszig.Signature(struct {
        pub const Input = struct { question: []const u8, document: dszig.File };
        pub const Output = struct { answer: []const u8 };
    });
    const view: dszig.adapter.View = .{
        .instructions = "Read the document.",
        .inputs = Files.input_fields_list,
        .outputs = Files.output_fields_list,
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const file = try dszig.File.fromBytes(arena, "hello", .{
        .filename = "ignored.txt",
        .mime_type = "text/plain",
    });
    var inputs: dszig.value.Record = .empty;
    try inputs.put(arena, "question", .{ .string = "read" });
    try inputs.put(arena, "document", try file.toRuntime(arena));
    var outputs: dszig.value.Record = .empty;
    try outputs.put(arena, "answer", .{ .string = "done" });

    const expected = [finetune_adapters.len][3][]const u8{
        .{
            "[[ ## question ## ]]\nread\n\n[[ ## document ## ]]\n",
            "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
            "[[ ## answer ## ]]\ndone\n\n[[ ## completed ## ]]\n",
        },
        .{
            "<question>\nread\n</question>\n\n<document>\n",
            "\n</document>\n\nOutput format (replace placeholders):\n<answer>\n{answer}\n</answer>\n\nUse this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.",
            "<answer>\ndone\n</answer>",
        },
    };
    for (finetune_adapters, expected) |adapter, texts| {
        const training = try adapter.formatFinetuneData(
            arena,
            testing.io,
            view,
            &.{},
            &inputs,
            &outputs,
        );
        const messages = try finetuneMessages(training);
        try testing.expectEqual(@as(usize, 3), messages.len);
        try testing.expectEqualStrings("user", messages[1].object.get("role").?.string);
        const blocks = messages[1].object.get("content").?.array.items;
        try testing.expectEqual(@as(usize, 3), blocks.len);
        try testing.expectEqualStrings("text", blocks[0].object.get("type").?.string);
        try testing.expectEqualStrings(texts[0], blocks[0].object.get("text").?.string);
        try testing.expectEqual(@as(usize, 2), blocks[1].object.count());
        try testing.expectEqualStrings("file", blocks[1].object.get("type").?.string);
        const block = blocks[1].object.get("file").?.object;
        try testing.expectEqual(@as(usize, 2), block.count());
        try testing.expectEqualStrings(
            "data:text/plain;base64,aGVsbG8=",
            block.get("file_data").?.string,
        );
        try testing.expectEqualStrings("file.txt", block.get("filename").?.string);
        try testing.expectEqualStrings("text", blocks[2].object.get("type").?.string);
        try testing.expectEqualStrings(texts[1], blocks[2].object.get("text").?.string);
        try expectFinetuneMessage(messages[2], "assistant", texts[2]);

        // The ordinary prompt still carries the canonical raw-payload part.
        const prompt = try adapter.format(arena, view, &.{}, &inputs);
        const part = prompt[1].content.parts[1].object;
        try testing.expectEqualStrings("document", part.get("type").?.string);
        try testing.expectEqualStrings("text/plain", part.get("media_type").?.string);
        try testing.expectEqualStrings("aGVsbG8=", part.get("data").?.string);
        try testing.expect(!part.contains("file"));
    }
}

// JSONAdapter raises NotImplementedError and TwoStepAdapter has no formatter
// at all, so both refuse before any prompt or media is examined. Chat and XML
// do format, and so surface the malformed prompt and unsupported URL source.
test "finetune data refuses JSON and TwoStep before malformed prompts or media" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var outputs: dszig.value.Record = .empty;
    try outputs.put(arena, "answer", .{ .string = "done" });
    try outputs.put(arena, "count", .{ .integer = 1 });
    var malformed: dszig.value.Record = .empty;
    try malformed.put(arena, "question", .{
        .string = dszig.File.marker_start ++ "[{\"type\": \"document\"}]" ++ dszig.File.marker_end,
    });
    var remote: dszig.value.Record = .empty;
    try remote.put(arena, "question", .{
        .string = dszig.File.marker_start ++
            "[{\"type\": \"document\", \"media_type\": \"application/pdf\", " ++
            "\"url\": \"https://example.invalid/a.pdf\"}]" ++ dszig.File.marker_end,
    });
    var valid: dszig.value.Record = .empty;
    try valid.put(arena, "question", .{ .string = "Why?" });

    for ([_]*const dszig.value.Record{ &valid, &malformed, &remote }) |inputs| {
        try testing.expectError(error.FinetuneNotImplemented, dszig.Adapter.json.formatFinetuneData(
            arena,
            testing.io,
            finetune_view,
            &.{},
            inputs,
            &outputs,
        ));
        try testing.expectError(
            error.MissingFinetuneFormatter,
            dszig.Adapter.two_step.formatFinetuneData(
                arena,
                testing.io,
                finetune_view,
                &.{},
                inputs,
                &outputs,
            ),
        );
    }
    for (finetune_adapters) |adapter| {
        try testing.expectError(error.AdapterParse, adapter.formatFinetuneData(
            arena,
            testing.io,
            finetune_view,
            &.{},
            &malformed,
            &outputs,
        ));
        try testing.expectError(error.UnsupportedFeature, adapter.formatFinetuneData(
            arena,
            testing.io,
            finetune_view,
            &.{},
            &remote,
            &outputs,
        ));
    }
}
