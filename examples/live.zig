//! Runs examples/live_suite.zig against a real OpenAI-compatible model and
//! writes the transcript of every request and real completion, so
//! tools/dspy_reference/check_live.py can replay it through DSPy 3.5.0.
//!
//!     OPENAI_API_KEY=... zig build live -- transcript.json
//!
//! For a Responses endpoint, set OPENAI_BASE_URL, DSZIG_LIVE_MODEL and
//! DSZIG_LIVE_API=responses. DSZIG_LIVE_REASONING=1 explicitly enables native
//! reasoning for a capable model. DSZIG_LIVE_RESPONSE_SCHEMA=1 and
//! DSZIG_LIVE_FUNCTION_CALLING=1 advertise the corresponding native capabilities.
//! CODEX_LB_API_KEY is accepted only with an explicit OPENAI_BASE_URL, so the
//! homelab credential can never fall through to the public OpenAI default.
//! `zig build live -- --smoke [transcript.json]` runs one real CoT request,
//! checks the answer to 2 + 2, and records its actual native metadata.
//! `--reasoning-smoke` requires a nonempty Responses summary in the typed result.
//! `--document-smoke` instead verifies typed inline File input on Responses.
//! `--embedding-smoke` verifies hosted embeddings, KNN, and a query cache hit.
//! `--optimization-smoke` bootstraps a demo and runs reloaded CoT inference.
//! `--tool-smoke` verifies a native function call, local execution, and submit.
//! `--stream-smoke [receipt.json]` verifies typed XML field routing over SSE.
//! `--schema-smoke [receipt.json]` verifies a strict provider JSON schema.
//! Managed via `devenv shell -- dz-live-stream-smoke [receipt.json]`.
//! `--python-rlm-smoke [receipt.json]` executes real model-written Python
//! in the bounded guest and validates the same code against a second input.
//! Defaults remain Chat Completions, gpt-4o-mini, and the public OpenAI URL.

const std = @import("std");
const dszig = @import("dszig");
const suite = @import("live_suite");

const Smoke = dszig.Signature(struct {
    pub const instructions = "Answer the arithmetic question. Give the final answer as a single digit.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});
const smoke_question = "What is 2 + 2?";
const reasoning_question = "What is the units digit of 7 raised to the 123rd power?";

const PythonSmoke = dszig.Signature(struct {
    pub const Input = struct { n: i64 };
    pub const Output = struct { answer: i64 };
});
const python_question = "Write Python that sums the integers from 1 through the provided variable n using a while loop. Use n, not a hardcoded value. Set answer and finish the Python code by calling SUBMIT(answer=answer).";

const StreamSmoke = dszig.Signature(struct {
    pub const instructions = "Answer the arithmetic question. Return the decimal integer in answer and the uppercase word EVEN or ODD in parity. Do not add explanations.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: i64, parity: []const u8 };
});

const StreamCapture = struct {
    const Field = struct {
        bytes: [32]u8 = undefined,
        len: usize = 0,
        terminal: bool = false,

        fn append(self: *Field, fragment: dszig.stream.Fragment) error{ StreamFragmentAfterTerminal, StreamCaptureOverflow }!void {
            if (self.terminal) return error.StreamFragmentAfterTerminal;
            if (fragment.text.len > self.bytes.len - self.len) return error.StreamCaptureOverflow;
            @memcpy(self.bytes[self.len..][0..fragment.text.len], fragment.text);
            self.len += fragment.text.len;
            self.terminal = fragment.is_last_chunk;
        }

        fn check(self: *const Field, expected: []const u8) !void {
            if (!self.terminal) return error.MissingStreamTerminal;
            if (!std.mem.eql(u8, self.bytes[0..self.len], expected)) return error.UnexpectedStreamFieldBytes;
        }
    };

    answer: Field = .{},
    parity: Field = .{},
    failure: ?error{ UnexpectedStreamField, StreamFragmentAfterTerminal, StreamCaptureOverflow } = null,

    fn onFragment(ptr: *anyopaque, fragment: dszig.stream.Fragment) dszig.lm.OpenAI.StreamSink.Action {
        const self: *StreamCapture = @ptrCast(@alignCast(ptr));
        const field = if (std.mem.eql(u8, fragment.field_name, "answer"))
            &self.answer
        else if (std.mem.eql(u8, fragment.field_name, "parity"))
            &self.parity
        else {
            self.failure = error.UnexpectedStreamField;
            return .stop;
        };
        // Borrowed text may be overwritten on the very next fragment.
        field.append(fragment) catch |err| {
            self.failure = err;
            return .stop;
        };
        return .proceed;
    }
};

const SchemaSmoke = dszig.Signature(struct {
    pub const instructions = "Answer the arithmetic question. Return the sum and whether it is even.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { sum: i64, @"résultat_pair": bool };
});

const DocumentSmoke = dszig.Signature(struct {
    pub const instructions = "Read the attached document and return only the requested code, with no surrounding punctuation.";
    pub const Input = struct { question: []const u8, document: dszig.File };
    pub const Output = struct { answer: []const u8 };
});
const document_question = "What is the shipment verification code in the attached document?";
const document_answer = "K7Q4-M2VX";

const ToolSmoke = dszig.Signature(struct {
    pub const instructions = "On the first turn, call only lookup_shipment. After receiving its result, call submit with only the code itself, not a sentence.";
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
    pub const descriptions = .{ .answer = "The bare verification code from the lookup, with no spaces, quotes, or surrounding text." };
});
const tool_question = "Look up the verification code for shipment SHP-314. Submit only the bare code.";
const tool_answer = "V8D2-X5JW";
const ToolAgent = dszig.react.ReActIters(ToolSmoke, &.{.{
    .name = "lookup_shipment",
    .desc = "Return the shipment verification code for the given shipment ID.",
    .args = &.{.{ .name = "shipment_id", .schema_type = "string" }},
    .func = lookupShipment,
}}, 2);

fn lookupShipment(_: dszig.Context, args: *const dszig.value.Record) dszig.react.ToolError!dszig.Value {
    const id = args.get("shipment_id") orelse return error.InvalidArguments;
    if (id != .string or !std.mem.eql(u8, id.string, "SHP-314")) return error.InvalidArguments;
    return .{ .string = tool_answer };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const smoke = args.len > 1 and std.mem.eql(u8, args[1], "--smoke");
    const reasoning_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--reasoning-smoke");
    const document_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--document-smoke");
    const embedding_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--embedding-smoke");
    const optimization_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--optimization-smoke");
    const tool_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--tool-smoke");
    const stream_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--stream-smoke");
    const python_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--python-rlm-smoke");
    const schema_smoke = args.len > 1 and std.mem.eql(u8, args[1], "--schema-smoke");
    const smoke_mode = smoke or reasoning_smoke or document_smoke or embedding_smoke or optimization_smoke or
        tool_smoke or stream_smoke or python_smoke or schema_smoke;
    if (args.len > 1 and (args.len > (if (smoke_mode) @as(usize, 3) else 2) or
        (!smoke_mode and std.mem.startsWith(u8, args[1], "-"))))
    {
        std.log.err("usage: live [[--smoke|--reasoning-smoke|--document-smoke|--embedding-smoke|" ++
            "--optimization-smoke|--tool-smoke|--stream-smoke|--schema-smoke|" ++
            "--python-rlm-smoke] output.json]", .{});
        std.process.exit(2);
    }
    const explicit_base_url = init.environ_map.get("OPENAI_BASE_URL");
    const key = init.environ_map.get("OPENAI_API_KEY") orelse homelab: {
        const homelab_key = init.environ_map.get("CODEX_LB_API_KEY") orelse {
            std.log.err("OPENAI_API_KEY or CODEX_LB_API_KEY must be set", .{});
            std.process.exit(2);
        };
        if (explicit_base_url == null) {
            std.log.err("OPENAI_BASE_URL must be set when using CODEX_LB_API_KEY", .{});
            std.process.exit(2);
        }
        break :homelab homelab_key;
    };
    const model = init.environ_map.get("DSZIG_LIVE_MODEL") orelse
        (if (reasoning_smoke) "gpt-5-mini" else "gpt-4o-mini");
    const base_url = explicit_base_url orelse "https://api.openai.com/v1";
    const api_name = init.environ_map.get("DSZIG_LIVE_API") orelse
        (if (reasoning_smoke) "responses" else "chat_completions");
    const api = std.meta.stringToEnum(dszig.lm.OpenAI.Api, api_name) orelse {
        std.log.err("DSZIG_LIVE_API must be chat_completions or responses", .{});
        std.process.exit(2);
    };
    const reasoning = enabled(init.environ_map.get("DSZIG_LIVE_REASONING"), "DSZIG_LIVE_REASONING");
    const response_schema = enabled(init.environ_map.get("DSZIG_LIVE_RESPONSE_SCHEMA"), "DSZIG_LIVE_RESPONSE_SCHEMA");
    const function_calling = enabled(init.environ_map.get("DSZIG_LIVE_FUNCTION_CALLING"), "DSZIG_LIVE_FUNCTION_CALLING");
    if (document_smoke and api != .responses) return error.DocumentSmokeRequiresResponses;
    if (reasoning_smoke and api != .responses) return error.ReasoningSmokeRequiresResponses;
    const smoke_output: ?[]const u8 = if (python_smoke)
        "live_python_rlm_smoke.json"
    else if (stream_smoke)
        "live_stream_smoke.json"
    else if (schema_smoke)
        "live_schema_smoke.json"
    else if (tool_smoke)
        "live_tool_smoke.json"
    else if (optimization_smoke)
        "live_optimization_smoke.json"
    else if (document_smoke)
        "live_document_smoke.json"
    else if (embedding_smoke)
        "live_embedding_smoke.json"
    else if (reasoning_smoke)
        "live_reasoning_smoke.json"
    else if (smoke)
        "live_smoke.json"
    else
        null;
    const out_path = if (smoke_output) |default|
        if (args.len > 2) args[2] else default
    else if (args.len > 1)
        args[1]
    else
        "live_dszig.json";

    var client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer client.deinit();
    var openai: dszig.lm.OpenAI = .{
        .client = &client,
        .base_url = base_url,
        .model = model,
        .api = api,
        .api_key = key,
        .supports_reasoning = reasoning or reasoning_smoke,
        .supports_response_schema = response_schema or schema_smoke,
        .supports_function_calling = function_calling or tool_smoke,
    };
    var buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    const w = &stdout.interface;
    if (embedding_smoke) {
        const json = try runEmbeddingSmoke(arena, init.gpa, &client, base_url, key);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        try w.print("embedding smoke: expected nearest example and repeat cache hit verified; wrote {s}\n", .{out_path});
        try w.flush();
        return;
    }
    if (stream_smoke) {
        openai.max_retries = 0;
        openai.max_response_bytes = 64 * 1024;
        const json = try runStreamSmoke(arena, &openai);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        try w.print("stream smoke: typed answer 4 and parity EVEN; both XML fields verified over SSE; wrote {s}\n", .{out_path});
        try w.flush();
        return;
    }
    if (python_smoke) {
        openai.max_retries = 0;
        openai.max_response_bytes = 64 * 1024;
    }
    var recorder: dszig.lm.RecordingLM = .{ .inner = openai.lm(), .arena = arena };
    if (schema_smoke) {
        const answer = try runSchemaSmoke(arena, &recorder);
        const json = try smokeTranscript(
            arena,
            model,
            api_name,
            recorder.lm(),
            recorder.calls.items,
            answer,
            "structured_output_smoke",
            smoke_question,
        );
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        try w.print("model: {s}; api: {s}; strict JSON schema: {s}; wrote {s}\n", .{
            model, api_name, answer, out_path,
        });
        try w.flush();
        return;
    }
    if (python_smoke) {
        const verified = runPythonSmoke(arena, &recorder) catch |err| {
            if (recorder.calls.items.len > 0) {
                const json = try smokeTranscript(arena, model, api_name, recorder.lm(), recorder.calls.items, @errorName(err), "bounded_python_rlm_refused", python_question);
                try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
            }
            return err;
        };
        const json = try smokeTranscript(arena, model, api_name, recorder.lm(), recorder.calls.items, verified, "bounded_python_rlm_smoke", python_question);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        try w.print("model: {s}; api: {s}; bounded Python outputs: {s}; {d} action calls; wrote {s}\n", .{
            model, api_name, verified, recorder.calls.items.len, out_path,
        });
        try w.flush();
        return;
    }
    if (tool_smoke) {
        const agent = try ToolAgent.init(arena);
        var trace: dszig.Trace = .{ .arena = arena };
        const result = try agent.forward(.{
            .arena = arena,
            .lm = recorder.lm(),
            .json_fallback = false,
            .trace = &trace,
            .max_trace_size = 1,
            .lm_options = .{ .parallel_tool_calls = false },
        }, .{ .question = tool_question });
        const calls = recorder.calls.items;
        // Preserve completed requests and responses even if the model misses
        // the expected trajectory. Only a successful exit qualifies the smoke.
        const json = try smokeTranscript(arena, model, api_name, recorder.lm(), calls, result.answer, "react_tool_smoke", tool_question);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        if (calls.len != 2) return error.UnexpectedSmokeCallCount;
        if (result.termination_reason != .submit or
            !singleNativeTool(&calls[0], "lookup_shipment") or
            !singleNativeTool(&calls[1], "submit")) return error.UnexpectedSmokeToolCalls;
        if (!std.mem.eql(u8, result.answer, tool_answer)) return error.UnexpectedSmokeAnswer;
        if (trace.steps.items.len != 1 or trace.evicted_steps != 1) return error.UnexpectedSmokeTrace;
        const traced = trace.steps.items[0].inputs.get("history") orelse return error.UnexpectedSmokeTrace;
        const traced_history = try dszig.History.fromValue(arena, traced);
        if (traced_history.messages.len != 2) return error.UnexpectedSmokeTrace;
        const lookup = try dszig.ToolCalls.fromValue(arena, traced_history.messages[0].get("tool_calls") orelse
            return error.UnexpectedSmokeTrace);
        const observations = (lookup.tool_call_results orelse return error.UnexpectedSmokeTrace).tool_call_results;
        if (lookup.tool_calls.len != 1 or observations.len != 1) return error.UnexpectedSmokeTrace;
        const observed = observations[0];
        if (observed.is_error or observed.value != .string or
            !std.mem.eql(u8, observed.value.string, tool_answer) or
            !std.mem.eql(u8, observed.name, "lookup_shipment")) return error.UnexpectedSmokeTrace;
        const submitted = traced_history.messages[1].get("answer") orelse return error.UnexpectedSmokeTrace;
        if (submitted != .string or !std.mem.eql(u8, submitted.string, tool_answer))
            return error.UnexpectedSmokeTrace;
        try w.print("model: {s}; api: {s}; tool answer: {s}; {d} calls; final trace history verified; wrote {s}\n", .{ model, api_name, result.answer, calls.len, out_path });
        try w.flush();
        return;
    }
    if (optimization_smoke) {
        const result = try optimizationSmoke(arena, init.gpa, &recorder);
        const json = try smokeTranscript(arena, model, api_name, recorder.lm(), recorder.calls.items, result, "bootstrap_few_shot_smoke", optimization_question);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        try w.print("model: {s}; api: {s}; bootstrapped and reloaded one demo; answer: {s}; {d} calls; wrote {s}\n", .{ model, api_name, result, recorder.calls.items.len, out_path });
        try w.flush();
        return;
    }
    if (document_smoke) {
        const document = try dszig.File.fromBytes(arena, "Shipment verification code: " ++ document_answer ++ "\n", .{
            .filename = "shipment.txt",
            .mime_type = "text/plain",
        });
        const program: dszig.Predict(DocumentSmoke) = .{};
        const result = try program.forward(.{
            .arena = arena,
            .lm = recorder.lm(),
            .json_fallback = false,
        }, .{ .question = document_question, .document = document });
        if (!std.mem.eql(u8, result.answer, document_answer)) return error.UnexpectedSmokeAnswer;
        if (recorder.calls.items.len != 1) return error.UnexpectedSmokeCallCount;
        const json = try smokeTranscript(arena, model, api_name, recorder.lm(), recorder.calls.items, result.answer, "document_smoke", document_question);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        try w.print("model: {s}; api: {s}; document code: {s}; wrote {s}\n", .{ model, api_name, result.answer, out_path });
        try w.flush();
        return;
    }
    if (smoke or reasoning_smoke) {
        const program: dszig.ChainOfThought(Smoke) = .{};
        const question = if (reasoning_smoke) reasoning_question else smoke_question;
        const expected = if (reasoning_smoke) "3" else "4";
        const result = try program.forward(.{
            .arena = arena,
            .lm = recorder.lm(),
            // A malformed model output must fail, not silently turn this
            // one-request smoke into an adapter-fallback conversation.
            .json_fallback = false,
        }, .{ .question = question });
        if (!std.mem.eql(u8, result.answer, expected)) return error.UnexpectedSmokeAnswer;
        if (recorder.calls.items.len != 1) return error.UnexpectedSmokeCallCount;
        const call = &recorder.calls.items[0];
        const scenario = if (reasoning_smoke) "native_reasoning_smoke" else "cot_smoke";
        const json = try smokeTranscript(arena, model, api_name, recorder.lm(), recorder.calls.items, result.answer, scenario, question);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
        const summary = if (call.reasonings.len == 1) call.reasonings[0] else null;
        const native_summary = if (summary) |text| text.len != 0 else false;
        if (reasoning_smoke) {
            const typed = result.reasoning orelse return error.MissingNativeReasoningSummary;
            if (typed.content.len == 0 or !std.mem.eql(u8, typed.content, summary orelse ""))
                return error.UnexpectedNativeReasoningSummary;
        }
        try w.print("model: {s}; api: {s}; answer: {s}; native summary: {s}; wrote {s}\n", .{
            model,
            api_name,
            result.answer,
            if (native_summary) "yes" else "no",
            out_path,
        });
        try w.flush();
        return;
    }
    const settings: suite.Settings = .{
        .temperature = try float(init.environ_map.get("DSZIG_LIVE_TEMPERATURE"), "DSZIG_LIVE_TEMPERATURE", 0.0),
        .copro_init_temperature = try float(init.environ_map.get("DSZIG_LIVE_COPRO_TEMPERATURE"), "DSZIG_LIVE_COPRO_TEMPERATURE", 1.4),
    };
    const workspace_buffer = try init.gpa.alloc(u8, suite.workspace_bytes_max);
    defer init.gpa.free(workspace_buffer);
    var workspace: dszig.Workspace = .init(workspace_buffer);
    const result = try suite.run(arena, init.gpa, &workspace, recorder.lm(), settings);
    const json = try suite.writeTranscript(arena, "dszig", model, recorder.lm(), recorder.calls.items, result, settings);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = json });
    for (result.cases) |c| {
        try w.print("{s}: {s}\n", .{ c.name, if (c.err) |e| @errorName(e) else "ok" });
    }
    try w.print("rag: baseline {d:.1}%  optimized {d:.1}%\n", .{ result.rag.baseline, result.rag.optimized });
    try w.print("dates: baseline {d:.1}%  optimized {d:.1}%\n", .{ result.dates.baseline, result.dates.optimized });
    try w.print("dates_search: baseline {d:.1}%  optimized {d:.1}%  candidates", .{ result.dates_search.baseline, result.dates_search.optimized });
    for (result.dates_search.candidates) |c| try w.print(" {d}:{d:.1}", .{ c.seed, c.score });
    try w.writeByte('\n');
    try w.print("dates_mipro: baseline {d:.1}%  optimized {d:.1}%  {d} trials\n", .{ result.dates_mipro.baseline, result.dates_mipro.optimized, result.dates_mipro.trials.len });
    try w.print("{d} real calls to {s}; wrote {s}\n", .{ recorder.calls.items.len, model, out_path });
    try w.flush();
}

fn runSchemaSmoke(arena: std.mem.Allocator, recorder: *dszig.lm.RecordingLM) ![]const u8 {
    const program: dszig.Predict(SchemaSmoke) = .{};
    const result = try program.forward(.{
        .arena = arena,
        .lm = recorder.lm(),
        .adapter = .json,
        .json_fallback = false,
    }, .{ .question = smoke_question });
    if (result.sum != 4 or !result.@"résultat_pair") return error.UnexpectedSmokeAnswer;
    if (recorder.calls.items.len != 1) return error.UnexpectedSmokeCallCount;
    const options = recorder.calls.items[0].options orelse return error.MissingSmokeSchema;
    const format = options.response_format orelse return error.MissingSmokeSchema;
    if (format != .json_schema or !format.json_schema.strict) return error.MissingSmokeSchema;
    const properties = format.json_schema.schema.object.get("properties") orelse return error.MissingSmokeSchema;
    if (properties != .object) return error.MissingSmokeSchema;
    const parity_field = properties.object.get("résultat_pair") orelse return error.MissingSmokeSchema;
    if (parity_field != .object) return error.MissingSmokeSchema;
    const title = parity_field.object.get("title") orelse return error.MissingSmokeSchema;
    if (title != .string or !std.mem.eql(u8, title.string, "Résultat Pair")) return error.UnexpectedSmokeSchema;
    return std.fmt.allocPrint(arena, "{d} (even: {})", .{ result.sum, result.@"résultat_pair" });
}

fn runEmbeddingSmoke(arena: std.mem.Allocator, allocator: std.mem.Allocator, client: *std.http.Client, base_url: []const u8, key: []const u8) ![]const u8 {
    const model = "text-embedding-3-small";
    const expected_topic = "orchard";
    var embedder = try dszig.Embedder.initHosted(allocator, .{
        .client = client,
        .model = "openai/" ++ model,
        .environment = &.{
            .{ .name = "OPENAI_BASE_URL", .value = base_url },
            .{ .name = "OPENAI_API_KEY", .value = key },
        },
    }, .{ .cache_capacity = 2 });
    defer embedder.deinit();

    const trainset = [_]dszig.Example{
        try dszig.Example.fromJson(arena, "{\"text\":\"A guide to pruning apple trees in an orchard.\",\"topic\":\"orchard\"}", &.{"text"}),
        try dszig.Example.fromJson(arena, "{\"text\":\"A guide to maintaining a bicycle drivetrain.\",\"topic\":\"bicycle\"}", &.{"text"}),
    };
    const query = try dszig.Example.fromJson(arena, "{\"text\":\"When should I prune fruit trees to improve my apple harvest?\"}", &.{});
    const knn = try dszig.KNN.init(arena, 1, &trainset, &embedder);
    const first = try knn.call(arena, query.record);
    if (embedder.hits != 0 or embedder.misses != 2) return error.UnexpectedEmbeddingSmokeCache;
    const second = try knn.call(arena, query.record);
    if (embedder.hits != 1 or embedder.misses != 2) return error.UnexpectedEmbeddingSmokeCache;
    for ([_][]const dszig.Example{ first, second }) |result| {
        if (result.len != 1) return error.UnexpectedEmbeddingSmokeResult;
        const topic = result[0].record.get("topic") orelse return error.UnexpectedEmbeddingSmokeResult;
        if (topic != .string or !std.mem.eql(u8, topic.string, expected_topic)) return error.UnexpectedEmbeddingSmokeResult;
    }

    return std.json.Stringify.valueAlloc(arena, .{
        .workload = "embedding_knn_smoke",
        .model = model,
        .nearest_topic = expected_topic,
        .cache_hits = embedder.hits,
        .cache_misses = embedder.misses,
    }, .{ .whitespace = .indent_2 });
}

fn runPythonSmoke(arena: std.mem.Allocator, recorder: *dszig.lm.RecordingLM) ![]const u8 {
    const guest = dszig.bounded_python;
    const options: guest.Options = .{ .fuel = 4096 };
    const slot = try arena.create(dszig.bounded_python_rlm.Session);
    slot.init(options, false);
    const program = try dszig.bounded_python_rlm.initProgram(PythonSmoke, arena, slot, .{
        .action_instructions = python_question,
        .max_iters = 2,
        .max_llm_calls = 0,
    });
    const first = try program.forward(.{
        .arena = arena,
        .lm = recorder.lm(),
        .json_fallback = false,
    }, .{ .n = 7 });
    if (first.outputs.answer != 28 or first.trajectory.len != recorder.calls.items.len)
        return error.UnexpectedPythonSmokeResult;

    // Re-run the very same model-written source on an input the model never
    // saw. A hard-coded answer cannot satisfy both checks.
    const machine = try arena.create(guest.Machine);
    machine.init();
    const inputs: []const guest.Binding = &.{.{ .name = "n", .value = .{ .integer = 11 } }};
    var alternate: guest.Result = .{ .observation = "" };
    for (first.trajectory) |entry| {
        alternate = try machine.run(entry.code, inputs, options, null);
    }
    if (alternate != .submit or alternate.submit.len != 1) return error.UnexpectedPythonSmokeResult;
    const field = alternate.submit[0];
    if (!std.mem.eql(u8, field.name, "answer") or field.value != .integer or field.value.integer != 66)
        return error.UnexpectedPythonSmokeResult;
    return "28 (n=7), 66 (n=11)";
}

fn runStreamSmoke(arena: std.mem.Allocator, openai: *dszig.lm.OpenAI) ![]const u8 {
    var program: dszig.Predict(StreamSmoke) = .{};
    var capture: StreamCapture = .{};
    var pending: [2][4096]u8 = undefined;
    var output: [2][4096]u8 = undefined;
    var xml_fields: [2][StreamSmoke.output_fields_list.len]dszig.lm.XmlStreamListener.Field = undefined;
    var bindings: [2]dszig.stream.Binding = undefined;
    var session = try dszig.stream.Session.init(arena, &program, &.{
        .{ .field_name = "answer", .pending_storage = &pending[0], .output_storage = &output[0], .xml_fields = &xml_fields[0] },
        .{ .field_name = "parity", .pending_storage = &pending[1], .output_storage = &output[1], .xml_fields = &xml_fields[1] },
    }, &bindings, .{ .context = &capture, .on_fragment = StreamCapture.onFragment });
    // Use the direct OpenAI LM, not a cache, replay, or transcript recorder.
    const result = dszig.callStream(&program, .{
        .arena = arena,
        .lm = openai.lm(),
        .adapter = .xml,
        .json_fallback = false,
    }, .{ .question = smoke_question }, &session, true) catch |err| {
        if (capture.failure) |failure| return failure;
        return err;
    };
    if (capture.failure) |err| return err;
    const prediction = result.prediction.fields();
    if (prediction.answer != 4 or !std.mem.eql(u8, prediction.parity, "EVEN"))
        return error.UnexpectedStreamPrediction;
    try capture.answer.check("4");
    try capture.parity.check("EVEN");
    if (!result.emit_prediction) return error.MissingStreamPrediction;

    // Serialize only the checked prediction and bounded observations, not
    // provider fragments, requests, headers, or raw completions.
    return std.json.Stringify.valueAlloc(arena, .{
        .case = "typed_xml_stream_smoke",
        .model = openai.model,
        .api = @tagName(openai.api),
        .prediction = prediction.*,
        .fields = .{
            .answer = .{ .bytes = capture.answer.len, .terminal = capture.answer.terminal },
            .parity = .{ .bytes = capture.parity.len, .terminal = capture.parity.terminal },
        },
        .emit_prediction = result.emit_prediction,
    }, .{});
}

fn singleNativeTool(call: *const dszig.lm.ReplayLM.Call, name: []const u8) bool {
    return call.tool_calls.len == 1 and call.tool_calls[0].len == 1 and
        std.mem.eql(u8, call.tool_calls[0][0].name, name);
}

const optimization_question = "What is 3 + 4?";

fn optimizationSmoke(arena: std.mem.Allocator, gpa: std.mem.Allocator, recorder: *dszig.lm.RecordingLM) ![]const u8 {
    const Program = dszig.ChainOfThought(Smoke);
    const ctx: dszig.Context = .{ .arena = arena, .lm = recorder.lm(), .json_fallback = false };
    var record: dszig.value.Record = .empty;
    try record.put(arena, "question", .{ .string = smoke_question });
    try record.put(arena, "answer", .{ .string = "4" });
    var optimizer: dszig.BootstrapFewShot = .{
        .metric = suite.exactMetric,
        .max_bootstrapped_demos = 1,
        .max_labeled_demos = 0,
        .max_rounds = 1,
        .max_errors = 1,
    };
    const compiled = try optimizer.compile(Program, arena, gpa, ctx, .{}, null, &.{
        .{ .record = record, .input_keys = &.{"question"} },
    });
    if (compiled.report.bootstrapped != 1 or compiled.report.errors != 0 or
        compiled.program.predict.state.demos.len != 1) return error.UnexpectedBootstrapResult;
    if (recorder.calls.items.len != 1) return error.UnexpectedSmokeCallCount;

    const artifact = try dszig.state.saveAlloc(arena, &compiled.program);
    var reloaded: Program = .{};
    try dszig.state.load(arena, &reloaded, artifact, .{});
    const demos = reloaded.predict.state.demos;
    if (demos.len != 1) return error.UnexpectedReloadedDemo;
    const demo_answer = demos[0].get("answer") orelse return error.UnexpectedReloadedDemo;
    if (demo_answer != .string or !std.mem.eql(u8, demo_answer.string, "4"))
        return error.UnexpectedReloadedDemo;

    const prediction = try reloaded.forward(ctx, .{ .question = optimization_question });
    if (!std.mem.eql(u8, prediction.answer, "7")) return error.UnexpectedSmokeAnswer;
    if (recorder.calls.items.len != 2) return error.UnexpectedSmokeCallCount;

    const messages = recorder.calls.items[1].messages;
    if (messages.len != 4 or messages[0].role != .system or
        !promptContains(messages[1], .user, "<question>\n" ++ smoke_question ++ "\n</question>") or
        !promptContains(messages[2], .assistant, "<answer>\n4\n</answer>") or
        !promptContains(messages[3], .user, "<question>\n" ++ optimization_question ++ "\n</question>"))
        return error.UnexpectedReloadedDemoPrompt;
    return prediction.answer;
}

fn promptContains(message: dszig.adapter.Message, role: dszig.adapter.Role, text: []const u8) bool {
    return message.role == role and message.content == .text and
        std.mem.indexOf(u8, message.content.text, text) != null;
}

/// A smoke transcript is deliberately not a full-suite result. It retains the
/// real request, completion and raw provider evidence without fake case data.
fn smokeTranscript(
    arena: std.mem.Allocator,
    model: []const u8,
    api: []const u8,
    lm: dszig.LM,
    recorded: []const dszig.lm.ReplayLM.Call,
    answer: []const u8,
    scenario: []const u8,
    question: []const u8,
) ![]const u8 {
    var calls: dszig.Array = .init(arena);
    for (recorded) |*call| try calls.append(try smokeCall(arena, call));

    var root: dszig.ObjectMap = .empty;
    try root.put(arena, "runtime", .{ .string = "dszig" });
    try root.put(arena, "target_dspy", .{ .string = suite.target_dspy });
    try root.put(arena, "scenario", .{ .string = scenario });
    try root.put(arena, "model", .{ .string = model });
    try root.put(arena, "api", .{ .string = api });
    try root.put(arena, "supports_reasoning", .{ .bool = lm.supports_reasoning });
    try root.put(arena, "supports_response_format", .{ .bool = lm.supports_response_format });
    try root.put(arena, "supports_response_schema", .{ .bool = lm.supports_response_schema });
    try root.put(arena, "supports_function_calling", .{ .bool = lm.supports_function_calling });
    if (lm.supports_decision_requests) try root.put(arena, "supports_decision_requests", .{ .bool = true });
    try root.put(arena, "question", .{ .string = question });
    try root.put(arena, "answer", .{ .string = answer });
    try root.put(arena, "calls", .{ .array = calls });
    var writer: std.Io.Writer.Allocating = .init(arena);
    try dszig.pyjson.writeJson(&writer.writer, .{ .object = root }, .indent2);
    return writer.written();
}

fn smokeCall(arena: std.mem.Allocator, call: *const dszig.lm.ReplayLM.Call) !dszig.Value {
    if (call.all().len != 1) return error.UnexpectedSmokeCallCount;
    var row = (try dszig.lm.requestValue(arena, .{
        .messages = call.messages,
        .options = call.options orelse return error.InvalidSmokeTranscript,
        .tools = call.tools,
        .decision = call.decision,
    })).object;
    try row.put(arena, "completion", if (call.completion) |text| .{ .string = text } else .null);
    var reasonings: dszig.Array = .init(arena);
    for (call.reasonings) |reasoning| try reasonings.append(if (reasoning) |text| .{ .string = text } else .null);
    try row.put(arena, "reasonings", .{ .array = reasonings });
    var tool_calls: dszig.Array = .init(arena);
    for (call.tool_calls) |native_calls| try tool_calls.append(try dszig.inference.toolCallsValue(arena, native_calls));
    try row.put(arena, "tool_calls", .{ .array = tool_calls });
    if (call.citations.len != 0) {
        var citations: dszig.Array = .init(arena);
        for (call.citations) |parts| try citations.append(try dszig.inference.citationsValue(arena, parts));
        try row.put(arena, "citations", .{ .array = citations });
    }
    var continuation: dszig.Array = .init(arena);
    for (call.continuation) |state| try continuation.append(try dszig.inference.continuationsValue(arena, state));
    try row.put(arena, "continuation", .{ .array = continuation });
    var provider_data: dszig.Array = .init(arena);
    for (call.provider_data) |data| try provider_data.append(data orelse .null);
    try row.put(arena, "provider_data", .{ .array = provider_data });
    if (call.metadata.len != 0) {
        var metadata: dszig.Array = .init(arena);
        for (call.metadata) |entry| try metadata.append(try dszig.lm.responseMetadataValue(arena, entry));
        try row.put(arena, "metadata", .{ .array = metadata });
    }
    if (call.generated_media.len != 0) {
        var flags: dszig.Array = .init(arena);
        for (call.generated_media) |flag| try flags.append(.{ .bool = flag });
        try row.put(arena, "generated_media", .{ .array = flags });
    }
    if (call.logprobs.len != 0) {
        var scores: dszig.Array = .init(arena);
        for (call.logprobs) |parts| try scores.append(if (parts) |entries| try dszig.lm.logprobs.toNativeValue(arena, entries) else .null);
        try row.put(arena, "logprobs", .{ .array = scores });
    }
    if (call.logprobs_complete.len != 0) {
        var flags: dszig.Array = .init(arena);
        for (call.logprobs_complete) |flag| try flags.append(.{ .bool = flag });
        try row.put(arena, "logprobs_complete", .{ .array = flags });
    }
    if (call.response_messages.len != 0) {
        var messages: dszig.Array = .init(arena);
        for (call.response_messages) |message|
            try messages.append(if (message) |entry| try entry.toValue(arena) else .null);
        try row.put(arena, "response_messages", .{ .array = messages });
    }
    if (call.adaptations.len != 0) {
        var adaptations: dszig.Array = .init(arena);
        for (call.adaptations) |entries|
            try adaptations.append(try dszig.sdk_response.adaptationsValue(arena, entries));
        try row.put(arena, "adaptations", .{ .array = adaptations });
    }
    return .{ .object = row };
}

fn float(raw: ?[]const u8, name: []const u8, default: f64) !f64 {
    const text = raw orelse return default;
    return std.fmt.parseFloat(f64, text) catch {
        std.log.err("{s} must be a number", .{name});
        std.process.exit(2);
    };
}

fn enabled(raw: ?[]const u8, name: []const u8) bool {
    const flag = raw orelse return false;
    if (std.mem.eql(u8, flag, "0")) return false;
    if (std.mem.eql(u8, flag, "1")) return true;
    std.log.err("{s} must be 0 or 1", .{name});
    std.process.exit(2);
}
