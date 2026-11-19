//! Gemini Developer API and Vertex AI generateContent transport.
const std = @import("std");
const lm_mod = @import("lm.zig");
const http = @import("http.zig");
const runtime = @import("runtime_value.zig");
const content = @import("openai_content.zig");
const sdk = lm_mod.sdk;
const Allocator = std.mem.Allocator;
const Error = lm_mod.Error;
const Value = runtime.Value;
const Object = runtime.ObjectMap;

pub const Gemini = struct {
    client: *std.http.Client,
    model: []const u8,
    api_key: ?[]const u8 = null,
    base_url: []const u8 = "https://generativelanguage.googleapis.com/v1beta",
    max_tokens: ?u32 = null,
    max_response_bytes: usize = 4 * 1024 * 1024,
    max_retries: u8 = 3,
    retry_base_ms: u32 = 500,
    timeout_ns: ?u64 = null,
    connect_timeout_ns: ?u64 = null,
    header_provider: ?http.HeaderProvider = null,
    adaptations: enum { note, silent, refuse } = .note,
    supports_reasoning: bool = false,
    door: enum { developer, vertex, vertex_express } = .developer,

    pub const LiveSession = GeminiLiveSession;

    pub const VertexOptions = struct {
        project: []const u8,
        location: []const u8 = "global",
        model: []const u8,
        credential: union(enum) {
            api_key: []const u8,
            oauth: http.HeaderProvider,
        },
        /// Trusted endpoint root, optionally with a gateway prefix.
        endpoint: ?[]const u8 = null,
    };

    /// `arena` owns the base URL. The model and credential state are borrowed.
    /// OAuth callers provide an Authorization header, refreshed per attempt.
    pub fn initVertex(arena: Allocator, client: *std.http.Client, options: VertexOptions) Error!Gemini {
        if (options.project.len == 0 or options.project.len > 256 or
            std.mem.eql(u8, options.project, ".") or std.mem.eql(u8, options.project, ".."))
            return error.LMFailure;
        const location = options.location;
        if (location.len == 0 or location.len > 63) return error.LMFailure;
        for (location) |byte| {
            if (!std.ascii.isAlphanumeric(byte) and byte != '-') return error.LMFailure;
        }
        var base: std.Io.Writer.Allocating = .init(arena);
        errdefer base.deinit();
        const out = &base.writer;
        if (options.endpoint) |root| {
            try writeVertexRoot(out, root);
        } else if (std.mem.eql(u8, location, "global")) {
            try vertexWrite(out, "https://aiplatform.googleapis.com");
        } else if (contains(&.{ "us", "eu" }, location)) {
            out.print("https://aiplatform.{s}.rep.googleapis.com", .{location}) catch
                return error.OutOfMemory;
        } else {
            out.print("https://{s}-aiplatform.googleapis.com", .{location}) catch
                return error.OutOfMemory;
        }
        try vertexWrite(out, "/v1/projects/");
        writeEscaped(out, options.project, .segment) catch return error.OutOfMemory;
        out.print("/locations/{s}/publishers/google", .{location}) catch return error.OutOfMemory;
        return .{
            .client = client,
            .model = options.model,
            .base_url = try base.toOwnedSlice(),
            .door = .vertex,
            .api_key = switch (options.credential) {
                .api_key => |key| key,
                .oauth => null,
            },
            .header_provider = switch (options.credential) {
                .api_key => null,
                .oauth => |provider| provider,
            },
        };
    }

    pub const VertexExpressOptions = struct {
        model: []const u8,
        api_key: []const u8,
        endpoint: ?[]const u8 = null,
    };

    /// Express requires no project or location. The key is sent in the query.
    pub fn initVertexExpress(
        arena: Allocator,
        client: *std.http.Client,
        options: VertexExpressOptions,
    ) Error!Gemini {
        var base: std.Io.Writer.Allocating = .init(arena);
        errdefer base.deinit();
        try writeVertexRoot(&base.writer, options.endpoint orelse "https://aiplatform.googleapis.com");
        try vertexWrite(&base.writer, "/v1/publishers/google");
        return .{
            .client = client,
            .model = options.model,
            .base_url = try base.toOwnedSlice(),
            .door = .vertex_express,
            .api_key = options.api_key,
        };
    }

    const State = lm_mod.ProviderState(Gemini, "dszig.lm.Gemini", .{
        "model",      "max_tokens",         "max_response_bytes", "max_retries",        "retry_base_ms",
        "timeout_ns", "connect_timeout_ns", "adaptations",        "supports_reasoning",
    });

    /// The caller chooses the endpoint and authentication protocol together.
    /// For Vertex, supply the same routing fields used by initVertex.
    pub const StateLoader = struct {
        client: *std.http.Client,
        base_url: []const u8,
        door: @FieldType(Gemini, "door") = .developer,
        api_key: ?[]const u8 = null,
        header_provider: ?http.HeaderProvider = null,

        pub fn loader(self: *StateLoader) lm_mod.LM.StateLoader {
            return .{ .ptr = self, .restore_engine = restore };
        }

        fn restore(ptr: *anyopaque, arena: Allocator, input: @import("runtime_value.zig").Value) lm_mod.StateError!lm_mod.LM {
            const self: *const StateLoader = @ptrCast(@alignCast(ptr));
            var model: Gemini = .{
                .client = self.client,
                .base_url = self.base_url,
                .door = self.door,
                .model = "",
                .api_key = self.api_key,
                .header_provider = self.header_provider,
            };
            try State.decode(arena, input, &model);
            const target = try arena.create(Gemini);
            target.* = model;
            return target.lm();
        }
    };

    pub fn lm(self: *Gemini) lm_mod.LM {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = complete,
                .complete_stream = completeStream,
                .complete_canonical = canonical,
                .dump_state = State.dump,
                .clone = clone,
                .model_name = modelName,
            },
            .supports_response_format = true,
            .supports_response_schema = true,
            .supports_function_calling = true,
            .supports_reasoning = self.supports_reasoning,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    fn clone(ptr: *anyopaque, arena: Allocator) anyerror!lm_mod.LM {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const model = try arena.dupe(u8, self.model);
        errdefer arena.free(model);
        const base = try arena.dupe(u8, self.base_url);
        errdefer arena.free(base);
        const copy = try arena.create(Gemini);
        copy.* = self.*;
        copy.model = model;
        copy.base_url = base;
        return copy.lm();
    }

    fn predictionRequest(self: *const Gemini, arena: Allocator, request: lm_mod.Request) Error!*const sdk.Request {
        for (request.tools) |tool| if (tool.strict != null) return error.LMFailure;
        return lm_mod.OpenAI.canonicalPredictionRequest(arena, request, self.model, self.max_tokens, null);
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) Error![]const lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const native = try self.predictionRequest(arena, request);
        defer native.release();
        const result = try arena.alloc(lm_mod.Completion, 1);
        result[0] = try self.completeFields(arena, native, .prediction);
        return result;
    }

    fn completeStream(
        ptr: *anyopaque,
        arena: Allocator,
        request: lm_mod.Request,
        sink: StreamSink,
    ) Error!lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        const native = try self.predictionRequest(arena, request);
        defer native.release();
        const model = wireModel(native.fields.model);
        var assembler: lm_mod.OpenAI.OpenAIStream = .{
            .arena = arena,
            .model = model,
            .stop = &.{},
        };
        const outcome = try self.runStream(arena, &native.fields, model, .prediction, sink, &assembler);
        if (outcome == .stopped) return error.StreamStopped;
        var completion = assembler.result orelse return error.LMFailure;
        try lm_mod.OpenAI.foldJudgments(arena, &completion, native);
        try lm_mod.publishResponse(arena, &completion, model, .borrowed);
        return completion;
    }

    /// Runs a buffered GenerateContent request. In function results, data and
    /// citations share the response text with ordinary text parts; image and
    /// document attachments are sent separately, in their original order.
    /// Replayed assistant citations carry only their cited text, not their
    /// grounding metadata.
    pub fn completeRequest(self: *Gemini, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        return (try canonical(self, arena, request)).response.?;
    }

    fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) Error!lm_mod.Completion {
        const self: *Gemini = @ptrCast(@alignCast(ptr));
        return self.completeFields(arena, request, .canonical);
    }

    fn completeFields(
        self: *const Gemini,
        arena: Allocator,
        request: *const sdk.Request,
        mode: RequestMode,
    ) Error!lm_mod.Completion {
        const fields = &request.fields;
        const model = wireModel(fields.model);
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try requestBody(arena, self.client.io, fields, model, mode, self.adaptations, &records);
        const reply = try self.send(arena, model, body, null);
        defer self.client.allocator.free(reply.body);
        const raw = @import("json_repair.zig").parseJson(arena, reply.body) catch |err|
            return codecError(err);
        var result = try parseResponse(arena, model, raw, records.items);
        try lm_mod.OpenAI.foldJudgments(arena, &result, request);
        try lm_mod.publishResponse(arena, &result, model, .borrowed);
        return result;
    }

    pub const StreamEvent = lm_mod.OpenAI.StreamEvent;
    pub const StreamSink = lm_mod.OpenAI.StreamSink;
    pub const StreamOutcome = lm_mod.OpenAI.StreamOutcome;

    /// Event payloads borrow frame storage during the callback. `.stop` closes the stream.
    /// GenerateContent SSE emits one end on exhaustion, including any late usage.
    pub fn streamRequest(
        self: *Gemini,
        backing: Allocator,
        request: *const sdk.Request,
        sink: StreamSink,
    ) Error!StreamOutcome {
        var scratch: std.heap.ArenaAllocator = .init(backing);
        defer scratch.deinit();
        const model = wireModel(request.fields.model);
        return self.runStream(scratch.allocator(), &request.fields, model, .canonical, sink, null);
    }

    fn runStream(
        self: *Gemini,
        arena: Allocator,
        fields: *const sdk.RequestFields,
        model: []const u8,
        mode: RequestMode,
        sink: StreamSink,
        assembler: ?*lm_mod.OpenAI.OpenAIStream,
    ) Error!StreamOutcome {
        const extensions = if (fields.config.extensions) |value|
            value.toValue(arena) catch |err| return codecError(err)
        else
            Value.null;
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        var state: GenerateStream = .{
            .model = model,
            .adaptations = &.{},
            .sink = sink,
            .assembler = assembler,
        };
        if (usesLive(model, extensions)) {
            self.runLive(arena, fields, extensions, &state) catch |err| return switch (err) {
                error.ContextWindowExceeded => error.ContextWindowExceeded,
                else => codecError(err),
            };
            return state.outcome;
        }
        const body = try requestBody(arena, self.client.io, fields, model, mode, self.adaptations, &records);
        state.adaptations = records.items;
        var consumer: http.ResponseConsumer = .{
            .context = &state,
            .consume = GenerateStream.consume,
            .on_head = GenerateStream.readHead,
        };
        const reply = try self.send(arena, model, body, &consumer);
        defer self.client.allocator.free(reply.body);
        return state.outcome;
    }

    /// Opens a persistent connection. Duplicate tool names return
    /// `error.InvalidResponse` before connecting. Do not move `session` until closed.
    pub fn live(self: *const Gemini, session: *LiveSession, options: LiveSession.Options) !void {
        if (self.door != .developer or self.header_provider != null) return error.LMFailure;
        // LiveConfig checks the complete tool list before the provider filters
        // out built-ins. This path does not go through sdk.Request validation.
        for (options.tools, 0..) |tool, index| {
            const name = switch (tool) {
                inline else => |item| item.name,
            };
            for (options.tools[0..index]) |previous| {
                const previous_name = switch (previous) {
                    inline else => |item| item.name,
                };
                if (std.mem.eql(u8, name, previous_name)) return error.InvalidResponse;
            }
        }
        const allocator = self.client.allocator;
        const model = wireModel(self.model);
        session.* = .{
            .timed = self.liveTimeouts(),
            .connection_arena = .init(allocator),
            .scratch = .init(allocator),
            .max_bytes = self.max_response_bytes,
            .audio_native = liveAudioNative(model),
        };
        errdefer session.connection_arena.deinit();
        errdefer session.scratch.deinit();
        errdefer session.buffer.deinit(allocator);
        const uri = try liveEndpoint(session.connection_arena.allocator(), self.base_url, self.api_key);
        session.client = try http.isolatedClient(self.client, session.timed.io());
        errdefer session.client.deinit();
        session.socket = try @import("websocket.zig").open(&session.client, uri);
        errdefer session.socket.deinit();
        // Close on every setup failure after upgrade, without masking the
        // original error. finish is a no-op if transport already disconnected.
        errdefer session.socket.finish() catch {};
        const arena = session.scratch.allocator();
        try session.socket.sendText(try session.setupFrame(arena, model, &options));
        var remaining = session.max_bytes;
        if (try waitLiveSetup(&session.socket, &session.scratch, &session.buffer, &remaining) == .rejected)
            return error.LMFailure;
        session.open = true;
    }

    fn liveTimeouts(self: *const Gemini) @import("timed_io.zig") {
        return .{
            .inner = self.client.io,
            .timeout_ns = self.timeout_ns orelse 600 * std.time.ns_per_s,
            .connect_timeout_ns = self.connect_timeout_ns orelse self.timeout_ns orelse
                5 * std.time.ns_per_s,
        };
    }

    fn runLive(
        self: *const Gemini,
        arena: Allocator,
        fields: *const sdk.RequestFields,
        extensions: Value,
        state: *GenerateStream,
    ) !void {
        if (self.door != .developer or self.header_provider != null) return error.LMFailure;
        const uri = try liveEndpoint(arena, self.base_url, self.api_key);
        const audio_native = liveAudioNative(state.model);
        const setup = try liveRequestSetup(arena, fields, state.model, extensions, audio_native);
        var timed = self.liveTimeouts();
        var client = try http.isolatedClient(self.client, timed.io());
        defer client.deinit();
        var socket = try @import("websocket.zig").open(&client, uri);
        defer socket.deinit();
        // Like DSPy's finally, close even if setup or turn preparation fails.
        // finish does not wait for the peer or send a second Close.
        defer socket.finish() catch {};
        try socket.sendText(setup);

        var buffer: std.ArrayList(u8) = .empty;
        defer buffer.deinit(client.allocator);
        var scratch: std.heap.ArenaAllocator = .init(client.allocator);
        defer scratch.deinit();
        var remaining = self.max_response_bytes;
        if (try waitLiveSetup(&socket, &scratch, &buffer, &remaining) == .rejected)
            return error.LMFailure;

        // Match DSPy: a rejected setup never opens local files for the turn.
        const input = if (audio_native)
            try liveRealtimeFrames(arena, fields.messages, self.client.io)
        else
            try liveTurnFrames(arena, fields.messages, self.client.io);
        for (input) |payload| try socket.sendText(payload);
        try state.start();
        const usage_allocator = if (state.assembler == null) client.allocator else arena;
        var usage: sdk.Usage = .{};
        defer if (state.assembler == null) usage.free(usage_allocator);
        var saw_tool = false;
        for (0..8192) |_| {
            if (state.outcome == .stopped) return;
            const value = try receiveLive(&socket, &scratch, &buffer, &remaining);
            if (value != .object) continue;
            if (value.object.get("error")) |detail|
                return state.emit(.{ .provider_error = try streamError(scratch.allocator(), detail) });
            if (try dispatchLive(scratch.allocator(), usage_allocator, state, value, &usage, &saw_tool)) return;
        }
        return error.LMFailure;
    }

    fn send(
        self: *const Gemini,
        arena: Allocator,
        model: []const u8,
        body: []const u8,
        consumer: ?*http.ResponseConsumer,
    ) Error!http.Response {
        var headers: [1]std.http.Header = undefined;
        var headers_len: usize = 0;
        const express_key = if (self.door == .vertex_express)
            self.api_key orelse return error.LMFailure
        else
            null;
        if (self.api_key) |key| {
            if (key.len == 0 or std.mem.indexOfAny(u8, key, "\r\n") != null) return error.LMFailure;
            if (express_key == null) {
                headers[0] = .{ .name = "x-goog-api-key", .value = key };
                headers_len = 1;
            }
        }
        const url = try endpoint(arena, self.base_url, model, consumer != null, express_key);
        const transport: http.Transport = .{
            .client = self.client,
            .max_response_bytes = self.max_response_bytes,
            .max_retries = self.max_retries,
            .retry_base_ms = self.retry_base_ms,
            .timeout_ns = self.timeout_ns,
            .connect_timeout_ns = self.connect_timeout_ns,
            .header_provider = self.header_provider,
            // DSPy leaves sub-400 replies to the Gemini response codec.
            .success = .below_400,
            .failure_error = error.LMFailure,
            .response_limit_error = error.LMFailure,
        };
        return transport.send(arena, .{
            .url = url,
            .method = .POST,
            .body = body,
            .content_type = "application/json",
            .headers = headers[0..headers_len],
            .response_consumer = consumer,
            .classify_failure = classifyGenerationFailure,
        }) catch |err| return switch (err) {
            error.ContextWindowExceeded => error.ContextWindowExceeded,
            else => codecError(err),
        };
    }
};

fn vertexWrite(out: *std.Io.Writer, value: []const u8) Error!void {
    out.writeAll(value) catch return error.OutOfMemory;
}

fn writeVertexRoot(out: *std.Io.Writer, root: []const u8) Error!void {
    const uri = std.Uri.parse(root) catch return error.LMFailure;
    if (!contains(&.{ "http", "https" }, uri.scheme)) return error.LMFailure;
    if (uri.host == null or uri.user != null or uri.password != null or
        uri.query != null or uri.fragment != null) return error.LMFailure;
    try vertexWrite(out, std.mem.trimEnd(u8, root, "/"));
}

const Escape = enum { segment, form, model };

fn isUnescaped(byte: u8, mode: Escape) bool {
    return std.ascii.isAlphanumeric(byte) or std.mem.indexOfScalar(u8, "-._~", byte) != null or
        (mode == .model and std.mem.indexOfScalar(u8, "/:@", byte) != null);
}

fn writeEscaped(out: *std.Io.Writer, bytes: []const u8, mode: Escape) std.Io.Writer.Error!void {
    const hex = "0123456789ABCDEF";
    for (bytes) |byte| {
        if (isUnescaped(byte, mode)) {
            try out.writeByte(byte);
        } else if (mode == .form and byte == ' ') {
            try out.writeByte('+');
        } else {
            try out.writeAll(&.{ '%', hex[byte >> 4], hex[byte & 15] });
        }
    }
}

fn wireModel(model: []const u8) []const u8 {
    const prefix = "gemini:";
    // Leave foreign bindings, empty suffixes and subsequent colons untouched.
    if (model.len > prefix.len and std.mem.startsWith(u8, model, prefix)) {
        return model[prefix.len..];
    }
    return model;
}

fn usesLive(model: []const u8, extensions: Value) bool {
    const transport = text(get(extensions, "transport"));
    for ([_][]const u8{ "live", "websocket", "ws" }) |name| {
        if (std.ascii.eqlIgnoreCase(transport, name)) return true;
    }
    return hasPhrase(model, "-live") or
        (model.len >= 4 and std.ascii.eqlIgnoreCase(model[model.len - 4 ..], "live"));
}

fn liveEndpoint(arena: Allocator, base_url: []const u8, key: ?[]const u8) !std.Uri {
    var uri = try std.Uri.parse(base_url);
    if (!contains(&.{ "http", "https" }, uri.scheme)) return error.LMFailure;
    if (uri.user != null or uri.password != null) return error.LMFailure;
    uri.fragment = null;
    uri.path = .{ .percent_encoded = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent" };
    uri.query = null;
    if (key) |bytes| {
        if (bytes.len == 0) return error.LMFailure;
        var query: std.Io.Writer.Allocating = .init(arena);
        errdefer query.deinit();
        try query.writer.writeAll("key=");
        try writeEscaped(&query.writer, bytes, .form);
        uri.query = .{ .percent_encoded = try query.toOwnedSlice() };
    }
    return uri;
}

const GeminiLiveSession = struct {
    timed: @import("timed_io.zig"),
    client: std.http.Client = undefined,
    socket: @import("websocket.zig") = undefined,
    connection_arena: std.heap.ArenaAllocator,
    scratch: std.heap.ArenaAllocator,
    buffer: std.ArrayList(u8) = .empty,
    pending: []const Event = &.{},
    max_bytes: usize,
    audio_native: bool,
    open: bool = false,

    pub const Options = struct {
        system: ?sdk.SystemContent = null,
        tools: []const sdk.RequestTool = &.{},
        voice: ?[]const u8 = null,
        audio_output: bool = false,
        extensions: ?sdk.JsonValue = null,
    };

    pub const Input = union(enum) {
        text: []const u8,
        turn: struct { parts: []const sdk.Part, complete: bool = true },
        audio: struct { data: []const u8, media_type: []const u8 = "audio/pcm;rate=16000" },
        image: struct { data: []const u8, media_type: []const u8 = "image/jpeg" },
        tool_result: struct { id: []const u8, parts: []const sdk.Part },
        interrupt,
        end_audio,
    };

    pub const Event = union(enum) {
        text: []const u8,
        audio: struct { data: []const u8, media_type: []const u8 },
        tool_call: struct { id: []const u8, name: []const u8, input: Value },
        usage: sdk.Usage,
        interrupted,
        turn_end: ?sdk.Usage,
        provider_error: GenerateStream.stream.ErrorDetail,
    };

    fn setupFrame(self: *const GeminiLiveSession, arena: Allocator, model: []const u8, options: *const Options) ![]const u8 {
        var setup = try liveSetupFields(arena, model, options.system, options.tools);
        var generation: Object = .empty;
        if (self.audio_native or options.audio_output)
            try generation.put(arena, "responseModalities", try singleton(arena, str("AUDIO")));
        if (options.voice) |voice| if (voice.len != 0) {
            const name = try wrapped(arena, "voiceName", str(voice));
            const prebuilt = try wrapped(arena, "prebuiltVoiceConfig", name);
            try generation.put(arena, "speechConfig", try wrapped(arena, "voiceConfig", prebuilt));
        };
        if (generation.count() != 0)
            try setup.put(arena, "generationConfig", .{ .object = generation });
        if (options.extensions) |extensions| {
            const value = try extensions.toValue(arena);
            if (value != .object) return error.LMFailure;
            for (value.object.keys(), value.object.values()) |key, item| try setup.put(arena, key, item);
        }
        if (self.audio_native) try setup.put(arena, "outputAudioTranscription", .{ .object = .empty });
        return liveJson(arena, try wrapped(arena, "setup", .{ .object = setup }));
    }

    pub fn close(self: *GeminiLiveSession) void {
        if (!self.open) return;
        self.open = false;
        self.socket.finish() catch {};
        self.socket.deinit();
        self.buffer.deinit(self.client.allocator);
        self.scratch.deinit();
        self.client.deinit();
        self.connection_arena.deinit();
    }

    /// A WebSocket write failure terminates the connection. Call close to
    /// release storage. A disconnected session rejects sends before preparing
    /// input. Local input preparation errors do not close the connection.
    pub fn send(self: *GeminiLiveSession, input: Input) !void {
        if (!self.open) return error.SessionClosed;
        if (!self.socket.isOpen()) return error.PeerClosed;
        var scratch: std.heap.ArenaAllocator = .init(self.client.allocator);
        defer scratch.deinit();
        const payload = try self.inputFrame(scratch.allocator(), input);
        self.socket.sendText(payload) catch |err| {
            self.pending = &.{};
            return err;
        };
    }

    fn inputFrame(self: *const GeminiLiveSession, arena: Allocator, input: Input) ![]const u8 {
        const names: std.StringHashMapUnmanaged([]const u8) = .empty;
        switch (input) {
            .text => |bytes| {
                if (self.audio_native) return liveFrame(arena, "realtimeInput", "text", str(bytes));
                return sessionTurn(arena, try singleton(arena, try textPart(arena, bytes)), true);
            },
            .turn => |turn| {
                var parts: runtime.Array = .init(arena);
                try parts.ensureTotalCapacity(turn.parts.len);
                for (turn.parts) |part| parts.appendAssumeCapacity(try encodePart(arena, part, &names, self.timed.inner));
                return sessionTurn(arena, .{ .array = parts }, turn.complete);
            },
            .audio => |item| return sessionMedia(arena, "audio", item.media_type, item.data),
            .image => |item| return sessionMedia(arena, "video", item.media_type, item.data),
            .tool_result => |item| {
                const bytes = try content.partsText(arena, item.parts, null);
                const output = try singleton(arena, try textPart(arena, bytes));
                var result: Object = .empty;
                try result.put(arena, "id", str(item.id));
                try result.put(arena, "response", try wrapped(arena, "output", output));
                const results = try singleton(arena, .{ .object = result });
                return liveFrame(arena, "toolResponse", "functionResponses", results);
            },
            .interrupt => return liveFrame(arena, "clientContent", "turnComplete", .{ .bool = true }),
            .end_audio => return liveFrame(arena, "realtimeInput", "audioStreamEnd", .{ .bool = true }),
        }
    }

    /// An event borrows session storage until the next recv or close.
    /// A WebSocket receive failure terminates the connection, including a
    /// transport read timeout. Call close to release storage.
    /// The caller serializes all operations on the session.
    pub fn recv(self: *GeminiLiveSession) !Event {
        if (!self.open) return error.SessionClosed;
        var remaining = self.max_bytes;
        if (self.pending.len == 0) {
            for (0..8192) |_| {
                const value = try receiveLive(&self.socket, &self.scratch, &self.buffer, &remaining);
                if (value != .object) continue;
                self.pending = try sessionEvents(self.scratch.allocator(), value);
                if (self.pending.len != 0) break;
            } else return error.LMFailure;
        }
        const event = self.pending[0];
        self.pending = self.pending[1..];
        return event;
    }
};

fn sessionMedia(arena: Allocator, key: []const u8, mime: []const u8, bytes: []const u8) ![]const u8 {
    var row: Object = .empty;
    try row.put(arena, "mimeType", str(mime));
    try row.put(arena, "data", str(bytes));
    return liveFrame(arena, "realtimeInput", key, .{ .object = row });
}

fn sessionTurn(arena: Allocator, parts: Value, complete: bool) ![]const u8 {
    var row: Object = .empty;
    try row.put(arena, "role", str("user"));
    try row.put(arena, "parts", parts);
    const turns = try singleton(arena, .{ .object = row });
    return liveContentFrame(arena, turns.array, complete);
}

fn sessionEvents(arena: Allocator, value: Value) ![]const GeminiLiveSession.Event {
    const Event = GeminiLiveSession.Event;
    var events: std.ArrayList(Event) = .empty;
    if (value.object.get("error")) |detail| {
        try events.append(arena, .{ .provider_error = try streamError(arena, detail) });
        return events.items;
    }
    for (try liveFunctionCalls(value)) |call| {
        if (call == .object) try events.append(arena, try sessionTool(arena, call));
    }
    const server = get(value, "serverContent");
    if (server != .object) return events.items;
    for (try liveModelParts(server)) |part| {
        if (part != .object) continue;
        if (part.object.get("text")) |raw| {
            try events.append(arena, .{ .text = (try truthyText(arena, raw)) orelse "" });
        } else if (get(part, "inlineData") == .object) {
            const inline_data = get(part, "inlineData");
            const mime = text(get(inline_data, "mimeType"));
            if (std.mem.startsWith(u8, mime, "audio/")) try events.append(arena, .{ .audio = .{
                .data = (try truthyText(arena, get(inline_data, "data"))) orelse "",
                .media_type = mime,
            } });
        } else if (get(part, "functionCall") == .object) {
            try events.append(arena, try sessionTool(arena, get(part, "functionCall")));
        }
    }
    if (try truthyText(arena, get(get(server, "outputTranscription"), "text"))) |bytes|
        try events.append(arena, .{ .text = bytes });
    const complete = try lm_mod.OpenAI.responseTruthy(get(server, "turnComplete"));
    const usage = try liveUsage(arena, value, server);
    if (!complete and usage != null) try events.append(arena, .{ .usage = usage.? });
    if (try lm_mod.OpenAI.responseTruthy(get(server, "interrupted")))
        try events.append(arena, .interrupted);
    if (complete) try events.append(arena, .{ .turn_end = if (usage) |counts|
        if (counts.isKnown()) counts else null
    else
        null });
    return events.items;
}

fn sessionTool(arena: Allocator, call: Value) !GeminiLiveSession.Event {
    const args = get(call, "args");
    return .{ .tool_call = .{
        .id = (try truthyText(arena, get(call, "id"))) orelse "fc_0",
        .name = (try truthyText(arena, get(call, "name"))) orelse "tool",
        .input = if (args == .object) args else .{ .object = .empty },
    } };
}

fn liveAudioNative(model: []const u8) bool {
    return hasPhrase(model, "live-preview") or hasPhrase(model, "native-audio");
}

fn liveRequestSetup(
    arena: Allocator,
    fields: *const sdk.RequestFields,
    model: []const u8,
    extensions: Value,
    audio_native: bool,
) ![]const u8 {
    if (extensions != .object and extensions != .null) return error.LMFailure;
    const output = text(get(extensions, "output"));
    const audio = audio_native or std.mem.eql(u8, output, "audio");
    const image = std.mem.eql(u8, output, "image");
    // Audio output checks the part type, regardless of the input frame format.
    var has_media = false;
    if (audio) {
        for (fields.messages) |message| for (message.parts) |part| {
            if (part == .audio or part == .video) has_media = true;
        };
    }

    var setup = try liveSetupFields(arena, model, fields.system, fields.tools);
    if (extensions == .object) {
        var entries = extensions.object.iterator();
        while (entries.next()) |entry| {
            if (contains(&.{ "transport", "prompt_caching", "output" }, entry.key_ptr.*)) continue;
            try setup.put(arena, entry.key_ptr.*, entry.value_ptr.*);
        }
    }
    try liveSetupModalities(arena, &setup, audio, image, has_media);
    if (audio and !std.mem.eql(u8, output, "audio")) {
        try setup.put(arena, "outputAudioTranscription", .{ .object = .empty });
    }
    return liveJson(arena, try wrapped(arena, "setup", .{ .object = setup }));
}

fn liveModelPath(arena: Allocator, model: []const u8) error{OutOfMemory}![]const u8 {
    // Gemini expects the same quoted resource name in setup JSON as in the HTTP path.
    const prefix = if (std.mem.startsWith(u8, model, "models/")) "" else "models/";
    var escaped_len = model.len;
    for (model) |byte| {
        if (!isUnescaped(byte, .model)) escaped_len += 2;
    }
    if (escaped_len == model.len) {
        if (prefix.len == 0) return model;
        return std.mem.concat(arena, u8, &.{ prefix, model });
    }
    const bytes = try arena.alloc(u8, prefix.len + escaped_len);
    @memcpy(bytes[0..prefix.len], prefix);
    var out: std.Io.Writer = .fixed(bytes[prefix.len..]);
    writeEscaped(&out, model, .model) catch unreachable;
    return bytes;
}

fn liveSetupFields(
    arena: Allocator,
    model: []const u8,
    system_content: ?sdk.SystemContent,
    tools: []const sdk.RequestTool,
) !Object {
    var setup: Object = .empty;
    try setup.put(arena, "model", str(try liveModelPath(arena, model)));
    if (system_content) |system| {
        // Test presence before flattening: a present sequence can render empty.
        const bytes: ?[]const u8 = switch (system) {
            .text => |value| if (value.len == 0) null else value,
            .parts => |parts| if (parts.len == 0) null else try content.partsText(arena, parts, null),
        };
        if (bytes) |value| {
            const parts = try singleton(arena, try textPart(arena, value));
            try setup.put(arena, "systemInstruction", try wrapped(arena, "parts", parts));
        }
    }
    const declarations = try functionDeclarations(arena, tools);
    if (declarations.items.len != 0) {
        const group = try wrapped(arena, "functionDeclarations", .{ .array = declarations });
        try setup.put(arena, "tools", try singleton(arena, group));
    }
    return setup;
}

fn liveSetupModalities(
    arena: Allocator,
    setup: *Object,
    audio: bool,
    image: bool,
    activity: bool,
) !void {
    // Extensions are already materialized in this request arena.
    var generation: Object = if (setup.get("generationConfig")) |value| switch (value) {
        .object => value.object,
        else => return error.LMFailure,
    } else .empty;
    if (audio or image or !generation.contains("responseModalities")) {
        const response_modality = if (audio) "AUDIO" else if (image) "IMAGE" else "TEXT";
        try generation.put(arena, "responseModalities", try singleton(arena, str(response_modality)));
        try setup.put(arena, "generationConfig", .{ .object = generation });
    }
    if (!activity) return;
    var realtime = try liveObjectCopy(arena, setup.get("realtimeInputConfig"));
    var detection = try liveObjectCopy(arena, realtime.get("automaticActivityDetection"));
    try detection.put(arena, "disabled", .{ .bool = true });
    try realtime.put(arena, "automaticActivityDetection", .{ .object = detection });
    try setup.put(arena, "realtimeInputConfig", .{ .object = realtime });
}

fn liveObjectCopy(arena: Allocator, item: ?Value) !Object {
    const value = item orelse return .empty;
    if (value != .object) return error.LMFailure;
    return value.object.clone(arena);
}

fn liveTurnFrames(arena: Allocator, messages: []const sdk.Message, io: std.Io) ![]const []const u8 {
    const payloads = try arena.alloc([]const u8, 1);
    if (messages.len == 1 and messages[0].role == .user) text_only: {
        for (messages[0].parts) |part| if (part != .text) break :text_only;
        const bytes = try content.partsText(arena, messages[0].parts, null);
        payloads[0] = try liveFrame(arena, "realtimeInput", "text", str(bytes));
        return payloads;
    }
    var turns: runtime.Array = .init(arena);
    var names: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer names.deinit(arena);
    for (messages) |message| for (message.parts) |part| {
        if (part == .tool_call) try names.put(arena, part.tool_call.id, part.tool_call.name);
    };
    try turns.ensureTotalCapacity(messages.len);
    for (messages) |message| turns.appendAssumeCapacity(try encodeMessage(arena, io, message, &names));
    payloads[0] = try liveContentFrame(arena, turns, true);
    return payloads;
}

fn liveContentFrame(arena: Allocator, turns: runtime.Array, complete: bool) ![]const u8 {
    var object: Object = .empty;
    try object.put(arena, "turns", .{ .array = turns });
    try object.put(arena, "turnComplete", .{ .bool = complete });
    return liveJson(arena, try wrapped(arena, "clientContent", .{ .object = object }));
}

fn liveFrame(arena: Allocator, outer: []const u8, inner: []const u8, value: Value) ![]const u8 {
    return liveJson(arena, try wrapped(arena, outer, try wrapped(arena, inner, value)));
}

fn liveRealtimeFrames(
    arena: Allocator,
    messages: []const sdk.Message,
    io: std.Io,
) ![]const []const u8 {
    var parts: runtime.Array = .init(arena);
    var texts: std.ArrayList([]const u8) = .empty;
    var media: std.ArrayList([]const u8) = .empty;
    const names: std.StringHashMapUnmanaged([]const u8) = .empty;
    for (messages) |message| for (message.parts) |part| switch (part) {
        .text => |item| {
            if (item.text.len != 0) try texts.append(
                arena,
                try liveFrame(arena, "realtimeInput", "text", str(item.text)),
            );
        },
        .audio, .video => |item| {
            if (item.data != null or item.path != null) try media.append(
                arena,
                try liveMediaFrame(arena, if (part == .audio) "audio" else "video", &item, io),
            );
        },
        .image, .document, .binary => try parts.append(try encodePart(arena, part, &names, io)),
        // The audio-native encoder projects media and text, not tool history.
        else => {},
    };
    var frames: std.ArrayList([]const u8) = .empty;
    if (media.items.len != 0) try frames.append(
        arena,
        try liveFrame(arena, "realtimeInput", "activityStart", .{ .object = .empty }),
    );
    if (parts.items.len != 0) {
        var turn: Object = .empty;
        try turn.put(arena, "role", str("user"));
        try turn.put(arena, "parts", .{ .array = parts });
        const turns = try singleton(arena, .{ .object = turn });
        try frames.append(arena, try liveContentFrame(arena, turns.array, false));
    }
    try frames.appendSlice(arena, texts.items);
    try frames.appendSlice(arena, media.items);
    if (media.items.len != 0) try frames.append(
        arena,
        try liveFrame(arena, "realtimeInput", "activityEnd", .{ .object = .empty }),
    );
    if (frames.items.len == 0) try frames.append(
        arena,
        try liveFrame(arena, "realtimeInput", "text", str("")),
    );
    return frames.toOwnedSlice(arena);
}

fn liveMediaFrame(arena: Allocator, name: []const u8, media: *const sdk.MediaPart, io: std.Io) ![]const u8 {
    const file = if (media.path) |path| try readMediaFile(arena, io, path) else null;
    defer if (file) |raw| arena.free(raw);
    var bytes: []const u8 = undefined;
    var mime = media.media_type;
    const audio = std.mem.eql(u8, name, "audio");
    if (mime.len == 0) mime = if (audio) "audio/pcm" else "video/mp4";
    if (audio and (std.mem.find(u8, mime, "wav") != null or
        std.mem.find(u8, mime, "wave") != null))
    {
        const raw = file orelse try decodeLiveWave(arena, media.data orelse return error.LMFailure);
        defer if (file == null) arena.free(raw);
        const pcm = liveWave(raw);
        bytes = try base64Media(arena, pcm.bytes);
        mime = try std.fmt.allocPrint(arena, "audio/pcm;rate={d}", .{pcm.rate});
    } else {
        bytes = if (file) |raw| try base64Media(arena, raw) else media.data orelse return error.LMFailure;
    }
    var media_value: Object = .empty;
    try media_value.put(arena, "mimeType", str(mime));
    try media_value.put(arena, "data", str(bytes));
    return liveFrame(arena, "realtimeInput", name, .{ .object = media_value });
}

fn decodeLiveWave(arena: Allocator, data: []const u8) Error![]const u8 {
    const delimiter = ";base64,";
    const prefix = if (std.mem.startsWith(u8, data, "data:")) std.mem.indexOf(u8, data, delimiter) else null;
    const payload = if (prefix) |i| data[i + delimiter.len ..] else data;
    // Match Python's strip/split trigger and measure the payload without
    // allocating a temporary copy stripped of whitespace.
    var encoded_len: usize = 0;
    var pads: usize = 0;
    var saw_space = false;
    var split = false;
    var i: usize = 0;
    while (i < payload.len) {
        const byte = payload[i];
        switch (byte) {
            'A'...'Z', 'a'...'z', '0'...'9', '+', '/' => {
                if (pads != 0) return error.LMFailure;
                encoded_len += 1;
                i += 1;
            },
            '=' => {
                if (pads == 2) return error.LMFailure;
                pads += 1;
                encoded_len += 1;
                i += 1;
            },
            else => {
                const len = std.unicode.utf8ByteSequenceLength(byte) catch return error.LMFailure;
                if (len > payload.len - i) return error.LMFailure;
                const cp = std.unicode.utf8Decode(payload[i..][0..len]) catch return error.LMFailure;
                if (!@import("unicode.zig").isSpace(cp)) return error.LMFailure;
                saw_space = true;
                if (i == 0 or i + len == payload.len or byte == ' ' or (byte >= '\t' and byte <= '\r')) split = true;
                i += len;
            },
        }
    }
    if ((saw_space and !split) or encoded_len % 4 != 0) return error.LMFailure;
    const decoded = try arena.alloc(u8, encoded_len / 4 * 3 - pads);
    errdefer arena.free(decoded);
    // Python accepts nonzero unused pad bits; Zig's standard decoder does not.
    const size = @import("python_base64.zig").decodeIgnoringNonAlphabetBytes(payload, .{ .bytes = decoded }, 0) catch return error.LMFailure;
    if (size != decoded.len) return error.LMFailure;
    return decoded;
}

fn liveWave(raw: []const u8) struct { bytes: []const u8, rate: u32 } {
    if (raw.len < 44 or !std.mem.eql(u8, raw[0..4], "RIFF") or
        !std.mem.eql(u8, raw[8..12], "WAVE"))
    {
        return .{ .bytes = raw, .rate = 16000 };
    }
    const rate = std.mem.readInt(u32, raw[24..28], .little);
    var offset: usize = 12;
    while (raw.len - offset >= 8) {
        const id = raw[offset..][0..4];
        const size = std.mem.readInt(u32, raw[offset + 4 ..][0..4], .little);
        offset += 8;
        if (std.mem.eql(u8, id, "data")) {
            return .{ .bytes = raw[offset..][0..@min(size, raw.len - offset)], .rate = rate };
        }
        if (size > raw.len - offset) break;
        offset += size;
    }
    return .{ .bytes = raw[44..], .rate = rate };
}

fn liveJson(arena: Allocator, value: Value) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(arena);
    errdefer output.deinit();
    // DSPy uses ensure_ascii=True. Escape lone surrogates instead of
    // writing their invalid UTF-8 representation into a WebSocket text frame.
    try @import("pyjson.zig").writeJsonDepth(&output.writer, value, .minified, 0, true);
    return output.toOwnedSlice();
}

fn waitLiveSetup(
    socket: *@import("websocket.zig"),
    scratch: *std.heap.ArenaAllocator,
    buffer: *std.ArrayList(u8),
    remaining: *usize,
) !enum { accepted, rejected } {
    for (0..8192) |_| {
        const value = try receiveLive(socket, scratch, buffer, remaining);
        if (value != .object) continue;
        if (value.object.contains("setupComplete")) return .accepted;
        if (value.object.contains("error")) return .rejected;
    }
    return error.LMFailure;
}

fn receiveLive(
    socket: *@import("websocket.zig"),
    scratch: *std.heap.ArenaAllocator,
    buffer: *std.ArrayList(u8),
    remaining: *usize,
) !Value {
    _ = scratch.reset(.retain_capacity);
    const message = try socket.receive(scratch.child_allocator, buffer, remaining.*);
    remaining.* -= message.bytes.len;
    // Text messages were validated by the WebSocket layer. DSPy also
    // decodes binary messages as strict UTF-8 before inspecting JSON.
    if (message.binary and !std.unicode.utf8ValidateSlice(message.bytes)) return .null;
    return @import("json_repair.zig").parseJson(scratch.allocator(), message.bytes) catch |err| {
        if (err == error.OutOfMemory) return err;
        return .null;
    };
}

fn dispatchLive(
    scratch: Allocator,
    usage_allocator: Allocator,
    state: *GenerateStream,
    value: Value,
    usage: *sdk.Usage,
    saw_tool: *bool,
) !bool {
    const server = get(value, "serverContent");
    if (try liveUsage(scratch, value, server)) |parsed| {
        if (parsed.isKnown()) {
            const replacement = try parsed.clone(usage_allocator);
            usage.free(usage_allocator);
            usage.* = replacement;
        }
    }
    // Validate the frame before exposing any tool calls to the sink.
    const parts = try liveModelParts(server);
    for (try liveFunctionCalls(value), 0..) |call, index| {
        if (call != .object) continue;
        const fallback = if (try truthyText(scratch, get(call, "id"))) |id|
            id
        else
            try std.fmt.allocPrint(scratch, "fc_{d}", .{index});
        try emitLiveTool(scratch, usage_allocator, state, call, index, fallback);
        saw_tool.* = true;
        if (state.outcome == .stopped) return true;
    }
    for (parts, 0..) |part, index| {
        const call = get(part, "functionCall");
        if (call == .object and !part.object.contains("text")) {
            const id = (try truthyText(scratch, get(call, "id"))) orelse "fc_0";
            try emitLiveTool(scratch, usage_allocator, state, call, index, id);
            saw_tool.* = true;
            if (state.outcome == .stopped) return true;
            continue;
        }
        const data = (try decodeLivePart(scratch, usage_allocator, state, part)) orelse continue;
        var key_buffer: [24]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buffer, "{d}", .{index}) catch unreachable;
        try state.emit(.{ .delta = .{ .part_index = key, .data = data } });
        if (state.outcome == .stopped) return true;
    }
    if (try truthyText(scratch, get(get(server, "outputTranscription"), "text"))) |bytes| {
        const retained = if (state.assembler != null) try usage_allocator.dupe(u8, bytes) else bytes;
        try state.emit(.{ .delta = .{ .part_index = "0", .data = .{ .text = .{ .text = retained } } } });
        if (state.outcome == .stopped) return true;
    }
    const complete = get(server, "turnComplete");
    if (!(lm_mod.OpenAI.responseTruthy(complete) catch |err| return codecError(err))) return false;
    try state.emit(.{ .end = .{
        .finish_reason = if (saw_tool.*) .tool_call else .stop,
        .usage = if (usage.isKnown()) usage.* else null,
        .provider_data = null,
    } });
    return true;
}

fn liveFunctionCalls(frame: Value) Error![]const Value {
    return responseElements(get(get(frame, "toolCall"), "functionCalls"));
}

fn liveModelParts(server: Value) Error![]const Value {
    const part_keys = [_][]const u8{ "text", "functionCall", "inlineData" };
    const raw_parts = get(get(server, "modelTurn"), "parts");
    if (raw_parts == .object) {
        // Python iterates object keys as strings, then tries to read any
        // key containing a recognized part field as if it were a mapping.
        for (raw_parts.object.keys()) |key| {
            for (part_keys) |field|
                if (std.mem.find(u8, key, field) != null) return error.LMFailure;
        }
    }
    const parts = try responseElements(raw_parts);
    for (parts) |part| switch (part) {
        // DSPy tests membership on each entry, then accesses a mapping
        // if a known part key matches. Unrecognized iterables are skipped.
        .null, .bool, .integer, .float, .number_string => return error.LMFailure,
        .string => |bytes| {
            for (part_keys) |key|
                if (std.mem.find(u8, bytes, key) != null) return error.LMFailure;
        },
        .array => |items| {
            for (items.items) |item| {
                if (item == .string and contains(&part_keys, item.string))
                    return error.LMFailure;
            }
        },
        else => {},
    };
    return parts;
}

fn liveUsage(arena: Allocator, value: Value, server: Value) !?sdk.Usage {
    const top_usage = get(value, "usageMetadata");
    const counts = if (top_usage == .object) top_usage else get(server, "usageMetadata");
    if (counts != .object) return null;
    if (counts.object.get("responseTokenCount")) |output| {
        var alias = try counts.object.clone(arena);
        try alias.put(arena, "candidatesTokenCount", output);
        return try readUsage(arena, .{ .object = alias });
    }
    return try readUsage(arena, counts);
}

fn emitLiveTool(
    scratch: Allocator,
    retained: Allocator,
    state: *GenerateStream,
    call: Value,
    index: usize,
    id: []const u8,
) !void {
    const payload_arena = if (state.assembler != null) retained else scratch;
    const args = call.object.get("args") orelse Value{ .object = .empty };
    var output: std.Io.Writer.Allocating = .init(payload_arena);
    errdefer output.deinit();
    try @import("pyjson.zig").writeJsonAscii(&output.writer, args);
    const input = try output.toOwnedSlice();
    const name = (try truthyText(scratch, get(call, "name"))) orelse "tool";
    var key_buffer: [24]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buffer, "{d}", .{index}) catch unreachable;
    try state.emit(.{ .delta = .{
        .part_index = key,
        .data = .{ .tool_call = .{
            .id = if (state.assembler != null) try retained.dupe(u8, id) else id,
            .name = if (state.assembler != null) try retained.dupe(u8, name) else name,
            .input = input,
        } },
    } });
}

fn decodeLivePart(
    scratch: Allocator,
    retained: Allocator,
    state: *const GenerateStream,
    part: Value,
) !?GenerateStream.stream.DeltaData {
    if (part != .object) return null;
    const retain = state.assembler != null;
    if (part.object.get("text")) |value| {
        const bytes = (try truthyText(scratch, value)) orelse "";
        return .{ .text = .{ .text = if (retain) try retained.dupe(u8, bytes) else bytes } };
    }
    const inline_data = get(part, "inlineData");
    const mime = text(get(inline_data, "mimeType"));
    const audio = std.mem.startsWith(u8, mime, "audio/");
    if (!audio and !std.mem.startsWith(u8, mime, "image/")) return null;
    const bytes = (try truthyText(scratch, get(inline_data, "data"))) orelse "";
    const media: GenerateStream.stream.Media = .{
        .data = if (retain) try retained.dupe(u8, bytes) else bytes,
        .media_type = if (retain) try retained.dupe(u8, mime) else mime,
    };
    return if (audio) .{ .audio = media } else .{ .image = media };
}

fn endpoint(
    arena: Allocator,
    base_url: []const u8,
    model: []const u8,
    stream: bool,
    query_key: ?[]const u8,
) Error![]const u8 {
    // Match urllib.quote(safe="/:@") on the model, not on the configured origin.
    var writer: std.Io.Writer.Allocating = .init(arena);
    const out = &writer.writer;
    out.writeAll(std.mem.trimEnd(u8, base_url, "/")) catch return error.OutOfMemory;
    out.writeByte('/') catch return error.OutOfMemory;
    if (!std.mem.startsWith(u8, model, "models/")) out.writeAll("models/") catch return error.OutOfMemory;
    writeEscaped(out, model, .model) catch return error.OutOfMemory;
    try vertexWrite(out, if (stream) ":streamGenerateContent?alt=sse" else ":generateContent");
    if (query_key) |key| {
        try vertexWrite(out, if (stream) "&key=" else "?key=");
        writeEscaped(out, key, .form) catch return error.OutOfMemory;
    }
    return writer.written();
}

test "Vertex chooses the residency host and isolates Express query credentials" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = std.testing.io };
    defer client.deinit();
    const options: Gemini.VertexOptions = .{
        .project = "billing:project",
        .location = "eu",
        .model = "gemini-2.5-flash",
        .credential = .{ .api_key = "fixture" },
    };
    const provider = try Gemini.initVertex(arena, &client, options);
    try std.testing.expectEqualStrings(
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/billing%3Aproject/locations/eu/publishers/google",
        provider.base_url,
    );
    var invalid = options;
    invalid.location = "eu/path";
    try std.testing.expectError(error.LMFailure, Gemini.initVertex(arena, &client, invalid));
    invalid = options;
    invalid.endpoint = "https://example.invalid?credential=";
    try std.testing.expectError(error.LMFailure, Gemini.initVertex(arena, &client, invalid));

    const express = try Gemini.initVertexExpress(arena, &client, .{
        .model = "models/gemini-2.5-flash",
        .api_key = "fixture +/&",
    });
    const url = try endpoint(arena, express.base_url, express.model, true, express.api_key);
    try std.testing.expectEqualStrings(
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=fixture+%2B%2F%26",
        url,
    );
}

const GenerateStream = struct {
    model: []const u8,
    adaptations: []const sdk.Adaptation,
    sink: Gemini.StreamSink,
    assembler: ?*lm_mod.OpenAI.OpenAIStream = null,
    outcome: Gemini.StreamOutcome = .exhausted,
    started: bool = false,
    previous: ?Slot = null,
    next_index: usize = 0,
    request_id: ?[]const u8 = null,
    retry_after: ?f64 = null,

    const Kind = enum { text, thinking, tool_call, media };
    const Slot = struct { index: usize, kind: Kind };
    const stream = @import("openai_stream.zig");
    const End = @FieldType(Gemini.StreamEvent, "end");
    const FrameResult = union(enum) { next, abort, terminal: End };

    fn readHead(context: *anyopaque, arena: Allocator, head: *const std.http.Client.Response.Head) Allocator.Error!void {
        const self: *GenerateStream = @ptrCast(@alignCast(context));
        // Match the locked provider's priority, not HTTP header order.
        const id_names = [_][]const u8{
            "x-request-id",    "request-id",      "x-amzn-requestid",      "x-amz-request-id",
            "x-ms-request-id", "apim-request-id", "x-typesafe-request-id",
        };
        const names = id_names ++ [_][]const u8{ "retry-after", "retry-after-ms", "x-ms-retry-after-ms" };
        var values: [names.len]?[]const u8 = @splat(null);
        var headers = head.iterateHeaders();
        while (headers.next()) |header| {
            for (names, &values) |name, *first| {
                if (!std.ascii.eqlIgnoreCase(name, header.name)) continue;
                if (first.* == null) first.* = header.value;
                break;
            }
        }
        self.retry_after = retryHint(values[id_names.len], false) orelse
            retryHint(values[id_names.len + 1], true) orelse retryHint(values[id_names.len + 2], true);
        for (values[0..id_names.len]) |entry| {
            const value = entry orelse continue;
            if (value.len == 0) continue;
            self.request_id = try arena.dupe(u8, value);
            break;
        }
    }

    fn retryHint(header: ?[]const u8, milliseconds: bool) ?f64 {
        const raw = header orelse return null;
        if (milliseconds and raw.len > 256) return null;
        const value = std.mem.trim(u8, raw, " \t");
        if (value.len == 0 or (milliseconds and value[0] == '-')) return null;
        // Accept decimal syntax only: Zig also parses hex floats, which are
        // not retry hints. The reference's millisecond headers forbid separators.
        for (value, 0..) |byte, i| switch (byte) {
            '0'...'9', '+', '-', '.', 'e', 'E' => {},
            '_' => {
                if (milliseconds or i == 0 or i + 1 == value.len or
                    !std.ascii.isDigit(value[i - 1]) or !std.ascii.isDigit(value[i + 1])) return null;
            },
            else => return null,
        };
        const number = std.fmt.parseFloat(f64, value) catch return null;
        if (!std.math.isFinite(number) or number < 0) return null;
        return if (milliseconds) number / 1000 else number;
    }

    fn consume(context: *anyopaque, arena: Allocator, reader: *std.Io.Reader, limit: usize) !void {
        const self: *GenerateStream = @ptrCast(@alignCast(context));
        var events = @import("sse.zig").Iterator.init(arena, reader, limit);
        defer events.deinit();
        // Raw delivery needs only the current frame and the latest terminal
        // frame. Typed assembly retains parsed leaves in the result arena.
        var frames: [2]std.heap.ArenaAllocator = .{ .init(arena), .init(arena) };
        defer for (&frames) |*frame| frame.deinit();
        var active: u1 = 0;
        var terminal: ?Gemini.StreamEvent = null;
        while (try events.next()) |bytes| {
            if (bytes.len == 0) continue;
            const temp = if (self.assembler != null) arena else frames[active].allocator();
            const value = try @import("json_repair.zig").parseJson(temp, bytes);
            if (value == .object) switch (try self.dispatch(temp, value)) {
                .abort => return,
                .next => {},
                .terminal => |end| {
                    // The latest terminal wins all fields.
                    terminal = .{ .end = end };
                    if (self.assembler == null) active ^= 1;
                },
            };
            _ = frames[active].reset(.retain_capacity);
        }
        if (terminal) |end| {
            try self.start();
            try self.emit(end);
        }
    }

    fn emit(self: *GenerateStream, event: Gemini.StreamEvent) Error!void {
        if (self.outcome == .stopped) return;
        if (self.sink.on_event(self.sink.context, event) == .stop) {
            self.outcome = .stopped;
            return;
        }
        if (self.assembler) |assembler| {
            // The raw sink receives the provider detail; collecting the stream
            // must also preserve the context failure as a recoverable LM error.
            if (event == .provider_error and event.provider_error.code == .context_length)
                return error.ContextWindowExceeded;
            try assembler.acceptRetainedEvent(event);
        }
    }

    fn start(self: *GenerateStream) Error!void {
        if (self.started) return;
        self.started = true;
        try self.emit(.{ .start = .{ .id = null, .model = self.model, .adaptations = self.adaptations } });
    }

    fn slot(self: *GenerateStream, index: usize, kind: Kind) usize {
        if (index == 0 and (kind == .text or kind == .thinking)) {
            if (self.previous) |previous| if (previous.kind == kind) return previous.index;
        }
        const result = self.next_index;
        self.next_index += 1;
        self.previous = .{ .index = result, .kind = kind };
        return result;
    }

    fn dispatch(self: *GenerateStream, arena: Allocator, value: Value) Error!FrameResult {
        const error_detail = value.object.get("error");
        // Report explicit errors and prompt blocks before rejecting candidates.
        const candidate_result: Error!Value = if (error_detail == null) firstCandidate(value) else .null;
        const failure = if (error_detail) |detail|
            try streamError(arena, detail)
        else
            try blockedStreamError(arena, value, candidate_result catch .null);
        if (failure) |detail| {
            var diagnostic = detail;
            diagnostic.request_id = self.request_id;
            diagnostic.retry_after = self.retry_after;
            try self.emit(.{ .provider_error = diagnostic });
            return .abort;
        }
        const candidate = try candidate_result;
        const finish = get(candidate, "finishReason");
        var saw_tool = false;
        var saw_delta = false;
        var scores = try readLogprobs(arena, get(candidate, "logprobsResult"));
        // DSPy validates terminal metadata before publishing deltas from
        // that frame. Keep earlier frames incremental.
        const has_finish = lm_mod.OpenAI.responseTruthy(finish) catch |err| return codecError(err);
        var terminal = if (has_finish) try decodeEnd(arena, value) else null;
        for (try candidateParts(candidate), 0..) |part, index| {
            const decoded = (try decodePart(arena, part)) orelse continue;
            const position = self.slot(index, decoded.kind);
            // All inline media separates slots, even when it produces no delta.
            var data = decoded.data orelse continue;
            if (data == .text) {
                data.text.scores = scores;
                scores = &.{};
            }
            try self.start();
            var buffer: [24]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{d}", .{position}) catch unreachable;
            try self.emit(.{ .delta = .{ .part_index = key, .data = data } });
            if (self.outcome == .stopped) return .abort;
            const continuation = try parseSignature(arena, decoded.signature);
            if (continuation.len != 0) {
                try self.emit(.{ .delta = .{
                    .part_index = key,
                    .data = .{ .continuation = continuation[0] },
                } });
                if (self.outcome == .stopped) return .abort;
            }
            saw_delta = true;
            saw_tool = saw_tool or decoded.kind == .tool_call;
        }
        if (terminal) |*end| {
            end.finish_reason = finishReason(text(finish), saw_tool);
            return .{ .terminal = end.* };
        }
        if (!saw_delta and value.object.contains("usageMetadata"))
            return .{ .terminal = try decodeEnd(arena, value) };
        return .next;
    }

    fn decodeEnd(arena: Allocator, value: Value) Error!End {
        return .{
            .finish_reason = .stop,
            .usage = try readUsage(arena, get(value, "usageMetadata")),
            .provider_data = sdk.PayloadValue.fromBorrowedValue(arena, value) catch |err|
                return codecError(err),
        };
    }

    const Decoded = struct { kind: Kind, data: ?stream.DeltaData = null, signature: Value = .null };

    fn decodePart(arena: Allocator, part: Value) Error!?Decoded {
        const signature = get(part, "thoughtSignature");
        const text_field = if (part == .object) part.object.get("text") else null;
        if (text_field) |value| {
            // A present null still identifies a part and can carry replay state.
            const bytes = if (value == .string) value.string else (try truthyText(arena, value)) orelse "";
            if (lm_mod.OpenAI.responseTruthy(get(part, "thought")) catch |err| return codecError(err))
                return .{ .kind = .thinking, .data = .{ .thinking = bytes }, .signature = signature };
            return .{ .kind = .text, .data = .{ .text = .{ .text = bytes } }, .signature = signature };
        }
        const call = get(part, "functionCall");
        if (call == .object) {
            const args = call.object.get("args") orelse Value{ .object = .empty };
            var writer: std.Io.Writer.Allocating = .init(arena);
            // ToolCallDelta.input uses Python json.dumps with compact separators.
            @import("pyjson.zig").writeJsonDepth(&writer.writer, args, .minified, 0, true) catch |err|
                return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
            return .{
                .kind = .tool_call,
                .data = .{ .tool_call = .{
                    .input = writer.written(),
                    .id = try truthyText(arena, get(call, "id")),
                    .name = try truthyText(arena, get(call, "name")),
                } },
                .signature = try functionCallSignature(signature, call),
            };
        }
        const inline_data = get(part, "inlineData");
        if (inline_data != .object) return null;
        const mime = text(get(inline_data, "mimeType"));
        const image = std.mem.startsWith(u8, mime, "image/");
        if (!image and !std.mem.startsWith(u8, mime, "audio/")) return .{ .kind = .media };
        const media: stream.Media = .{
            .media_type = mime,
            .data = (try truthyText(arena, get(inline_data, "data"))) orelse "",
        };
        return .{ .kind = .media, .data = if (image) .{ .image = media } else .{ .audio = media } };
    }
};

const StreamError = @import("openai_stream.zig").ErrorDetail;

fn truthyText(arena: Allocator, value: Value) Error!?[]const u8 {
    const truthy = lm_mod.OpenAI.responseTruthy(value) catch |err| return codecError(err);
    if (!truthy) return null;
    return try pythonText(arena, value);
}

fn functionCallSignature(outer: Value, call: Value) Error!Value {
    // Python's `outer or inner` checks only the outer value for truthiness.
    return if (lm_mod.OpenAI.responseTruthy(outer) catch |err| return codecError(err))
        outer
    else
        get(call, "thoughtSignature");
}

fn pythonText(arena: Allocator, value: Value) Error![]const u8 {
    if (value == .string) return value.string;
    var output: std.Io.Writer.Allocating = .init(arena);
    errdefer output.deinit();
    @import("pyjson.zig").writeStr(&output.writer, value) catch |err| return codecError(err);
    return try output.toOwnedSlice();
}

fn streamError(arena: Allocator, detail: Value) Error!StreamError {
    const status = try truthyText(arena, get(detail, "status")) orelse
        try truthyText(arena, get(detail, "code")) orelse "provider";
    const message = try truthyText(arena, get(detail, "message"));
    return .{
        // The status is a display fallback, not evidence for message heuristics.
        .code = providerError(status, message orelse ""),
        .provider_code = status,
        .message = message orelse status,
    };
}

fn isPromptBlock(reason: Value) Error!bool {
    if (reason == .string) return reason.string.len != 0 and
        !std.mem.eql(u8, reason.string, "BLOCK_REASON_UNSPECIFIED");
    return lm_mod.OpenAI.responseTruthy(reason) catch |err| return codecError(err);
}

fn blockedStreamError(arena: Allocator, frame: Value, candidate: Value) Error!?StreamError {
    const block = get(get(frame, "promptFeedback"), "blockReason");
    const message = message: {
        if (try isPromptBlock(block)) {
            break :message try std.fmt.allocPrint(
                arena,
                "Prompt blocked: {s} (gemini)",
                .{try pythonText(arena, block)},
            );
        }
        const finish = text(get(candidate, "finishReason"));
        if (!invalidFinish(finish)) return null;
        if (try truthyText(arena, get(candidate, "finishMessage"))) |detail| {
            break :message try std.fmt.allocPrint(arena, "{s} (gemini)", .{detail});
        }
        break :message try std.fmt.allocPrint(arena, "Candidate blocked: {s} (gemini)", .{finish});
    };
    return .{
        .code = .invalid_request,
        .provider_code = "inband_finish_reason",
        .message = message,
    };
}

fn hasPhrase(haystack: []const u8, comptime needle: []const u8) bool {
    comptime std.debug.assert(needle.len != 0);
    if (haystack.len < needle.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

fn isContextLengthMessage(message: []const u8) bool {
    return (hasPhrase(message, "token") and (hasPhrase(message, "limit") or hasPhrase(message, "exceed"))) or
        hasPhrase(message, "too long") or hasPhrase(message, "context length");
}

fn classifyGenerationFailure(allocator: Allocator, status: std.http.Status, body: []const u8) ?http.FailureDecision {
    // DSPy maps a bare HTTP 408 to a retryable timeout. A recognized envelope
    // can override the status in either direction.
    const fallback: ?http.FailureDecision = if (status == .request_timeout) .retry else null;
    var scratch: std.heap.ArenaAllocator = .init(allocator);
    defer scratch.deinit();
    const envelope = @import("json_repair.zig").parseJson(scratch.allocator(), body) catch |err|
        return if (err == error.OutOfMemory) .{ .terminal = err } else fallback;
    const detail = get(envelope, "error");
    // DSPy stringifies non-object error values before checking the message.
    // Only strings and arrays can contain one of the context-limit phrases.
    if (detail != .object) {
        const message = switch (detail) {
            .string => detail.string,
            .array => msg: {
                var output: std.Io.Writer.Allocating = .init(scratch.allocator());
                @import("pyjson.zig").writeStr(&output.writer, detail) catch |err|
                    return if (err == error.OutOfMemory) .{ .terminal = err } else fallback;
                break :msg output.written();
            },
            else => return fallback,
        };
        return if (isContextLengthMessage(message)) .{ .terminal = error.ContextWindowExceeded } else fallback;
    }
    // DSPy gives Google's explicit invalid-key reason precedence over message heuristics.
    const code = text(get(detail, "status"));
    if (std.mem.eql(u8, code, "INVALID_ARGUMENT")) {
        for (elems(get(detail, "details"))) |info| {
            if (std.mem.endsWith(u8, text(get(info, "@type")), "google.rpc.ErrorInfo") and
                std.mem.eql(u8, text(get(info, "reason")), "API_KEY_INVALID")) return .{ .terminal = error.LMFailure };
        }
    }
    // An absent message defaults to ""; a present non-string message makes
    // DSPy fall back to HTTP status rather than honoring the RPC status.
    const message = detail.object.get("message") orelse Value{ .string = "" };
    if (message != .string) return fallback;
    return switch (providerError(code, message.string)) {
        .context_length => .{ .terminal = error.ContextWindowExceeded },
        .invalid_request, .auth, .billing, .unsupported_model => .{ .terminal = error.LMFailure },
        .rate_limit, .server, .timeout => .retry,
        else => fallback,
    };
}

fn providerError(code: []const u8, message: []const u8) @FieldType(StreamError, "code") {
    if (isContextLengthMessage(message)) return .context_length;
    if (std.mem.eql(u8, code, "NOT_FOUND") and hasPhrase(message, "model")) {
        inline for (.{
            "not found",   "not exist",     "not supported",
            "unsupported", "not available", "unknown",
        }) |marker| {
            if (hasPhrase(message, marker)) return .unsupported_model;
        }
    }
    if (contains(&.{ "INVALID_ARGUMENT", "NOT_FOUND" }, code)) return .invalid_request;
    if (contains(&.{ "PERMISSION_DENIED", "UNAUTHENTICATED" }, code)) return .auth;
    if (contains(&.{ "INTERNAL", "UNAVAILABLE" }, code)) return .server;
    if (std.mem.eql(u8, code, "FAILED_PRECONDITION")) return .billing;
    if (std.mem.eql(u8, code, "RESOURCE_EXHAUSTED")) return .rate_limit;
    if (std.mem.eql(u8, code, "DEADLINE_EXCEEDED")) return .timeout;
    return .provider;
}

fn nonempty(value: Value) ?[]const u8 {
    const bytes = text(value);
    return if (bytes.len == 0) null else bytes;
}

fn finishReason(reason: []const u8, saw_tool: bool) sdk.FinishReason {
    if (saw_tool) return .tool_call;
    if (std.ascii.eqlIgnoreCase(reason, "MAX_TOKENS")) return .length;
    // DSPy normalizes the finish reason after case-sensitive in-band error admission.
    for ([_][]const u8{ "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII" }) |filtered| {
        if (std.ascii.eqlIgnoreCase(reason, filtered)) return .content_filter;
    }
    return .stop;
}

fn invalidFinish(reason: []const u8) bool {
    return contains(&.{
        "SAFETY",             "RECITATION",              "LANGUAGE",             "BLOCKLIST",                "PROHIBITED_CONTENT",
        "SPII",               "MALFORMED_FUNCTION_CALL", "IMAGE_SAFETY",         "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER",
        "NO_IMAGE",           "IMAGE_RECITATION",        "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS",      "MISSING_THOUGHT_SIGNATURE",
        "MALFORMED_RESPONSE",
    }, reason);
}

const RequestMode = enum { canonical, prediction };

fn requestBody(
    arena: Allocator,
    io: std.Io,
    fields: *const sdk.RequestFields,
    model: []const u8,
    mode: RequestMode,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error![]const u8 {
    const config = &fields.config;
    const extensions = if (config.extensions) |value|
        value.toValue(arena) catch |err| return codecError(err)
    else
        Value{ .object = .empty };
    if (extensions != .object) return error.LMFailure;
    var names: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer names.deinit(arena);
    for (fields.messages) |message| for (message.parts) |part| {
        if (part == .tool_call) try names.put(arena, part.tool_call.id, part.tool_call.name);
    };
    const cache = try cachePrefix(arena, fields, policy, records);
    const messages = fields.messages[cache.suffix..];
    var rows: runtime.Array = .init(arena);
    try rows.ensureTotalCapacity(messages.len);
    for (messages) |message| rows.appendAssumeCapacity(try encodeMessage(arena, io, message, &names));
    var body: Object = .empty;
    try body.put(arena, "contents", .{ .array = rows });
    if (cache.resource) |resource| {
        try body.put(arena, "cachedContent", str(resource));
    } else if (fields.system) |system| {
        var row: Object = .empty;
        const parts = switch (system) {
            .text => |bytes| try singleton(arena, try textPart(arena, bytes)),
            .parts => |parts| try joinedParts(arena, null, parts),
        };
        try row.put(arena, "parts", parts);
        try body.put(arena, "systemInstruction", .{ .object = row });
    }
    var generation: Object = .empty;
    inline for (.{ .{ "temperature", "temperature" }, .{ "top_p", "topP" }, .{ "frequency_penalty", "frequencyPenalty" }, .{ "presence_penalty", "presencePenalty" } }) |pair| {
        if (@field(config, pair[0])) |num| {
            const normalized: Value = if (num == @trunc(num) and num >= -0x1p63 and num < 0x1p63)
                .{ .integer = @intFromFloat(num) }
            else
                .{ .float = num };
            try generation.put(arena, pair[1], normalized);
        }
    }
    inline for (.{ .{ "max_tokens", "maxOutputTokens" }, .{ "top_k", "topK" }, .{ "seed", "seed" } }) |pair| {
        const value = @field(config, pair[0]);
        if (value != .null) try generation.put(arena, pair[1], value);
    }
    if (config.stop.len != 0) {
        var values: runtime.Array = .init(arena);
        try values.ensureTotalCapacity(config.stop.len);
        for (config.stop) |stop| values.appendAssumeCapacity(str(stop));
        try generation.put(arena, "stopSequences", .{ .array = values });
    }
    if (config.logprobs != .null) {
        try generation.put(arena, "responseLogprobs", .{ .bool = true });
        const order = @import("value.zig").numericOrder(config.logprobs, .{ .integer = 0 }) catch
            return error.LMFailure;
        if (order == .gt) try generation.put(arena, "logprobs", config.logprobs);
    }
    if (config.response_format) |format| {
        const value = format.toValue(arena) catch |err| return codecError(err);
        try generation.put(arena, "responseMimeType", str("application/json"));
        const kind = text(get(value, "type"));
        if (std.mem.eql(u8, kind, "json_schema")) {
            var schema = get(value, "schema");
            const found = try @import("typesafe.zig").responseJudgments(arena, schema);
            try noteJudgmentProbabilities(arena, config, policy, records, found);
            try encodeJudgmentSchema(arena, &schema, found);
            const field = if (try openApi(arena, schema)) "responseSchema" else "responseJsonSchema";
            try generation.put(arena, field, schema);
        } else if (!std.mem.eql(u8, kind, "json_object")) return error.LMFailure;
    }
    if (config.reasoning) |reasoning| {
        var effective = reasoning;
        // Typed DSPy prediction consumes the provider's visible thought summary.
        if (mode == .prediction and reasoning.effort != .off and reasoning.summary == null)
            effective.summary = .auto;
        const thinking = try thinkingConfig(arena, model, &effective, policy, records);
        try generation.put(arena, "thinkingConfig", thinking);
    }
    const output = text(get(extensions, "output"));
    const requested_modality: ?[]const u8 = if (std.mem.eql(u8, output, "image"))
        "IMAGE"
    else if (std.mem.eql(u8, output, "audio"))
        "AUDIO"
    else
        null;
    if (requested_modality) |name| {
        try generation.put(arena, "responseModalities", try singleton(arena, str(name)));
    }
    if (generation.count() != 0) try body.put(arena, "generationConfig", .{ .object = generation });
    if (cache.resource == null) try addTools(arena, &body, fields, policy, records);
    try applyExtensions(arena, &body, config, &extensions.object, policy, records);
    var writer: std.Io.Writer.Allocating = .init(arena);
    @import("pyjson.zig").writeJson(&writer.writer, .{ .object = body }, .minified) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
    return writer.written();
}

fn applyExtensions(
    arena: Allocator,
    body: *Object,
    config: *const sdk.Config,
    extensions: *const Object,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error!void {
    if (config.store) |store| try body.put(arena, "store", .{ .bool = store });
    if (config.service_tier) |tier| try body.put(arena, "serviceTier", str(tier));
    if (config.user_id) |user| try adaptString(
        arena,
        policy,
        records,
        "config.user_id",
        .dropped,
        "GenerateContent has no end-user attribution field (OpenAI and Anthropic carry it)",
        user,
        null,
    );
    var entries = extensions.iterator();
    while (entries.next()) |entry| {
        const key = entry.key_ptr.*;
        if (contains(&.{ "prompt_caching", "output" }, key)) continue;
        try body.put(arena, key, entry.value_ptr.*);
    }
}

const CachePrefix = struct {
    suffix: usize = 0,
    resource: ?[]const u8 = null,
};

fn cachePrefix(
    arena: Allocator,
    fields: *const sdk.RequestFields,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error!CachePrefix {
    const config = fields.config.cache orelse return .{};
    if (config.mode == .off) return .{};
    if (config.key) |key| {
        try adaptString(arena, policy, records, "config.cache.key", .dropped, "GenerateContent has no cache affinity key; implicit caching applies, and a " ++
            "stored cache (lm.cache(prefix), cache.resource) is the explicit tier", key, null);
    }
    if (config.retention == .long) {
        try adaptString(arena, policy, records, "config.cache.retention", .dropped, "GenerateContent takes no lifetime in-request; it belongs to the stored cache " ++
            "(cache_create(..., ttl_seconds=...) / cache_update)", "long", null);
    }
    const resource = config.resource orelse return .{};
    if (fields.messages.len == 0) return error.LMFailure;
    var result: CachePrefix = .{
        .resource = if (std.mem.startsWith(u8, resource, "cachedContents/")) resource else try std.fmt.allocPrint(arena, "cachedContents/{s}", .{resource}),
    };
    if (config.prefix_until_index != .null) {
        const last = fields.messages.len - 1;
        const order = @import("value.zig").numericOrder(
            config.prefix_until_index,
            .{ .integer = @intCast(last) },
        ) catch return error.LMFailure;
        if (order != .lt) return error.LMFailure;
        if (config.prefix_until_index != .integer) return error.LMFailure;
        result.suffix = std.math.cast(usize, config.prefix_until_index.integer) orelse
            return error.LMFailure;
        result.suffix += 1;
    }
    return result;
}

fn thinkingConfig(
    arena: Allocator,
    model: []const u8,
    reasoning: *const sdk.Reasoning,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
) Error!Value {
    const level = startsIgnoreCase(model, "gemini-3") or startsIgnoreCase(model, "models/gemini-3");
    var effort = reasoning.effort;
    if (effort == .off) {
        if (!level) return wrapped(arena, "thinkingBudget", .{ .integer = 0 });
        const reason = try std.fmt.allocPrint(arena, "{s} cannot disable thinking " ++
            "(the Gemini 3 class honours no off switch); the lowest level was sent and " ++
            "the thinking spend is visible in usage", .{model});
        try adaptString(arena, policy, records, "config.reasoning.effort", .substituted, reason, "off", "minimal");
        effort = .minimal;
    }

    var thinking: Object = .empty;
    if (reasoning.summary) |summary| {
        if (summary != .auto) {
            try adaptString(arena, policy, records, "config.reasoning.summary", .substituted, "GenerateContent has includeThoughts only, no detail levels; 'auto' shows the thoughts", @tagName(summary), "auto");
        }
        try thinking.put(arena, "includeThoughts", .{ .bool = true });
    }
    if (reasoning.thinking_budget != .null) {
        try thinking.put(arena, "thinkingBudget", reasoning.thinking_budget);
    } else {
        var selected = effort orelse return error.LMFailure;
        if (level) {
            if (selected == .xhigh or selected == .max) {
                try adaptString(arena, policy, records, "config.reasoning.effort", .clamped, "the Gemini 3 class has thinkingLevel minimal|low|medium|high; 'high' is the ceiling", @tagName(selected), "high");
                selected = .high;
            }
            try thinking.put(arena, "thinkingLevel", str(@tagName(selected)));
        } else {
            const budget: i64 = switch (selected) {
                .off => unreachable,
                .minimal => 1024,
                .low => 2048,
                .medium => 8192,
                .high => 16384,
                .xhigh => 24576,
                .max => 32768,
            };
            try thinking.put(arena, "thinkingBudget", .{ .integer = budget });
        }
    }
    return .{ .object = thinking };
}

const Judgment = @import("typesafe.zig").Judgment;

fn noteJudgmentProbabilities(
    arena: Allocator,
    config: *const sdk.Config,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
    found: []const Judgment,
) Error!void {
    for (found) |judgment| {
        if (judgment.kind == .box) continue;
        return switch (config.probabilities orelse .off) {
            .off => {},
            .required => error.LMFailure,
            .if_available => adaptString(
                arena,
                policy,
                records,
                "config.probabilities",
                .dropped,
                "this wire cannot measure a distribution over the declared keys; the answer carries the pick only",
                "if_available",
                null,
            ),
        };
    }
}

fn encodeJudgmentSchema(arena: Allocator, schema: *Value, found: []const Judgment) Error!void {
    if (found.len == 0) return;
    const properties = &schema.object.getPtr("properties").?.object;
    for (found) |judgment| {
        if (judgment.kind == .boolean or judgment.kind == .box) continue;
        const property = &properties.getPtr(judgment.name).?.object;
        const value = property.get("anyOf") orelse continue;
        const old_enum = property.get("enum") orelse .null;
        // A recognized explicit enum can coexist with a non-const anyOf.
        // Only const-branch judgments carry per-key labels.
        const branches: []const Value = if (old_enum == .array) &.{} else value.array.items;
        const values = if (old_enum == .array) old_enum.array else from_branches: {
            std.debug.assert(branches.len == judgment.keys.len);
            var items: runtime.Array = .init(arena);
            try items.ensureTotalCapacity(branches.len);
            for (branches) |branch| items.appendAssumeCapacity(get(branch, "const"));
            break :from_branches items;
        };
        _ = property.orderedRemove("anyOf");
        const ordered = judgment.kind == .ordered;
        try property.put(arena, "type", str(if (ordered) "integer" else "string"));
        try property.put(arena, "enum", .{ .array = values });

        if (try judgmentDescription(arena, property.get("description") orelse .null, judgment, branches)) |description|
            try property.put(arena, "description", str(description));
    }
}

fn judgmentDescription(
    arena: Allocator,
    existing: Value,
    judgment: Judgment,
    branches: []const Value,
) Error!?[]const u8 {
    for (branches) |branch| {
        if (text(get(branch, "title")).len != 0) break;
        if (text(get(branch, "description")).len != 0) break;
    } else return null;

    var out: std.Io.Writer.Allocating = .init(arena);
    const head = text(existing);
    if (head.len != 0) {
        out.writer.writeAll(head) catch return error.OutOfMemory;
        out.writer.writeByte(' ') catch return error.OutOfMemory;
    }
    const heading = if (judgment.kind == .ordered) "Levels: " else "Options: ";
    out.writer.writeAll(heading) catch return error.OutOfMemory;
    for (branches, judgment.keys, 0..) |branch, key, index| {
        if (index != 0) out.writer.writeAll("; ") catch return error.OutOfMemory;
        out.writer.writeAll(key) catch return error.OutOfMemory;
        const title = get(branch, "title");
        const description = get(branch, "description");
        if (title != .string and description != .string) continue;
        const label = text(title);
        const details = text(description);
        out.writer.writeAll(" = ") catch return error.OutOfMemory;
        out.writer.writeAll(label) catch return error.OutOfMemory;
        if (label.len != 0 and details.len != 0)
            out.writer.writeAll(": ") catch return error.OutOfMemory;
        out.writer.writeAll(details) catch return error.OutOfMemory;
    }
    return @import("value.zig").pyStrip(out.written());
}

fn startsIgnoreCase(value: []const u8, prefix: []const u8) bool {
    return value.len >= prefix.len and std.ascii.eqlIgnoreCase(value[0..prefix.len], prefix);
}

fn adaptString(
    arena: Allocator,
    policy: @FieldType(Gemini, "adaptations"),
    records: *std.ArrayList(sdk.Adaptation),
    field: []const u8,
    action: sdk.AdaptationAction,
    reason: []const u8,
    asked: []const u8,
    applied: ?[]const u8,
) Error!void {
    switch (policy) {
        .silent => {},
        .refuse => return error.LMFailure,
        .note => try records.append(arena, .{
            .field = field,
            .action = action,
            .reason = reason,
            .asked = sdk.PayloadValue.fromBorrowedValue(arena, str(asked)) catch |err|
                return codecError(err),
            .applied = if (applied) |bytes|
                sdk.PayloadValue.fromBorrowedValue(arena, str(bytes)) catch |err| return codecError(err)
            else
                .{},
        }),
    }
}

fn encodeSignature(
    arena: Allocator,
    row: *Value,
    states: []const sdk.Continuation,
    thinking: bool,
) Error!void {
    for (states) |state| {
        if (!std.mem.eql(u8, state.provider, "gemini")) continue;
        if (!std.mem.eql(u8, state.kind, "thought_signature")) continue;
        const data = state.data.toValue(arena) catch |err| return codecError(err);
        const signature = get(data, "value");
        if (lm_mod.OpenAI.responseTruthy(signature) catch |err| return codecError(err)) {
            if (thinking) try row.object.put(arena, "thought", .{ .bool = true });
            try row.object.put(arena, "thoughtSignature", signature);
        }
        return;
    }
}

fn parseSignature(arena: Allocator, value: Value) Error![]const sdk.Continuation {
    if (value == .null) return &.{};
    // The reference stringifies every non-null signature, including false and zero.
    const data = try wrapped(arena, "value", str(try pythonText(arena, value)));
    const states = try arena.alloc(sdk.Continuation, 1);
    states[0] = .{
        .provider = "gemini",
        .kind = "thought_signature",
        .data = sdk.PayloadValue.fromBorrowedValue(arena, data) catch |err| return codecError(err),
    };
    return states;
}

fn encodeMessage(
    arena: Allocator,
    io: std.Io,
    message: sdk.Message,
    names: *const std.StringHashMapUnmanaged([]const u8),
) Error!Value {
    var row: Object = .empty;
    try row.put(arena, "role", str(if (message.role == .assistant) "model" else "user"));
    const parts = if (message.role == .developer)
        try joinedParts(arena, "[developer]", message.parts)
    else blk: {
        var values: runtime.Array = .init(arena);
        try values.ensureTotalCapacity(message.parts.len);
        for (message.parts) |part| values.appendAssumeCapacity(try encodePart(arena, part, names, io));
        break :blk Value{ .array = values };
    };
    try row.put(arena, "parts", parts);
    return .{ .object = row };
}

fn encodePart(
    arena: Allocator,
    part: sdk.Part,
    names: *const std.StringHashMapUnmanaged([]const u8),
    file_io: ?std.Io,
) Error!Value {
    switch (part) {
        .data => |value| return textPart(arena, try content.dataText(arena, value.value)),
        // Grounding metadata is response-only. Replay the cited text, not its labels.
        .citation => |value| return textPart(arena, value.text orelse ""),
        // Gemini replays refusals as plain text, not signed model parts.
        .refusal => |value| return textPart(arena, value.text),
        .text => |value| {
            var row = try textPart(arena, value.text);
            try encodeSignature(arena, &row, value.continuation, false);
            return row;
        },
        .thinking => |value| {
            var row = try textPart(arena, value.text);
            try encodeSignature(arena, &row, value.continuation, true);
            return row;
        },
        .tool_call => |call| {
            var row: Object = .empty;
            try row.put(arena, "name", str(call.name));
            try row.put(arena, "args", call.input.toValue(arena) catch |err| return codecError(err));
            try row.put(arena, "id", str(call.id));
            var encoded = try wrapped(arena, "functionCall", .{ .object = row });
            try encodeSignature(arena, &encoded, call.continuation, false);
            return encoded;
        },
        .image, .audio, .video, .document, .binary => |media| return encodeMedia(arena, &media, file_io),
        .tool_result => |result| return encodeToolResult(arena, &result, names, file_io),
    }
}

fn encodeMedia(arena: Allocator, media: *const sdk.MediaPart, file_io: ?std.Io) Error!Value {
    // Gemini's inlineData and fileData do not carry SDK continuation state.
    const mime = if (media.media_type.len != 0) media.media_type else "application/octet-stream";
    var row: Object = .empty;
    try row.put(arena, "mimeType", str(mime));
    if (media.url orelse media.file_id) |uri| {
        try row.put(arena, "fileUri", str(uri));
        return wrapped(arena, "fileData", .{ .object = row });
    }
    const data = media.data orelse try encodeMediaFile(
        arena,
        file_io orelse return error.LMFailure,
        media.path orelse return error.LMFailure,
    );
    try row.put(arena, "data", str(data));
    return wrapped(arena, "inlineData", .{ .object = row });
}

fn encodeMediaFile(arena: Allocator, io: std.Io, path: []const u8) Error![]const u8 {
    const bytes = try readMediaFile(arena, io, path);
    defer arena.free(bytes);
    return base64Media(arena, bytes);
}

fn readMediaFile(arena: Allocator, io: std.Io, path: []const u8) Error![]u8 {
    const max_file_bytes = 16 * 1024 * 1024;
    // The reader's limit is exclusive at EOF; allow exactly max_file_bytes.
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(max_file_bytes + 1)) catch |err|
        return codecError(err);
    errdefer arena.free(bytes);
    if (bytes.len > max_file_bytes) return error.LMFailure;
    return bytes;
}

fn base64Media(arena: Allocator, bytes: []const u8) Error![]const u8 {
    const encoder = std.base64.standard.Encoder;
    return encoder.encode(try arena.alloc(u8, encoder.calcSize(bytes.len)), bytes);
}

fn encodeToolResult(
    arena: Allocator,
    result: *const sdk.ToolResultPart,
    names: *const std.StringHashMapUnmanaged([]const u8),
    file_io: ?std.Io,
) Error!Value {
    // Gemini replays thought signatures on model parts, not on function responses.
    const name = result.name orelse names.get(result.id) orelse return error.LMFailure;
    var attachments: runtime.Array = .init(arena);
    var body: std.Io.Writer.Allocating = .init(arena);
    const out = &body.writer;
    var has_non_media = false;
    var newline: []const u8 = "";
    for (result.content) |part| {
        switch (part) {
            .image, .document => |media| {
                try attachments.append(try encodeMedia(arena, &media, file_io));
                continue;
            },
            .text => |value| {
                // functionResponse renders text content, not its continuation state.
                out.writeAll(newline) catch return error.OutOfMemory;
                out.writeAll(value.text) catch return error.OutOfMemory;
            },
            .data => |value| {
                out.writeAll(newline) catch return error.OutOfMemory;
                value.value.writeJsonMinified(out) catch |err|
                    return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
            },
            .citation => |value| {
                // Only the citation's presentational fields belong in response text.
                has_non_media = true;
                var separator = newline;
                inline for (.{ "title", "url", "text" }) |field| {
                    if (@field(value, field)) |bytes| {
                        if (bytes.len != 0) {
                            out.writeAll(separator) catch return error.OutOfMemory;
                            out.writeAll(bytes) catch return error.OutOfMemory;
                            separator = " — ";
                            newline = "\n";
                        }
                    }
                }
                continue;
            },
            .audio, .video, .binary => return error.LMFailure,
        }
        has_non_media = true;
        newline = "\n";
    }
    const response = if (result.is_error)
        try wrapped(arena, "error", str(body.written()))
    else if (has_non_media or attachments.items.len == 0)
        try wrapped(arena, "result", str(body.written()))
    else
        Value{ .object = .empty };
    var row: Object = .empty;
    try row.put(arena, "name", str(name));
    try row.put(arena, "response", response);
    if (result.id.len != 0) try row.put(arena, "id", str(result.id));
    if (attachments.items.len != 0) try row.put(arena, "parts", .{ .array = attachments });
    return wrapped(arena, "functionResponse", .{ .object = row });
}

fn addTools(arena: Allocator, body: *Object, fields: *const sdk.RequestFields, policy: @FieldType(Gemini, "adaptations"), records: *std.ArrayList(sdk.Adaptation)) Error!void {
    const declarations = try functionDeclarations(arena, fields.tools);
    var tools: runtime.Array = .init(arena);
    if (declarations.items.len != 0) try tools.append(try wrapped(arena, "functionDeclarations", .{ .array = declarations }));
    for (fields.tools) |tool| if (tool == .builtin) {
        const builtin = tool.builtin;
        const key = if (std.mem.eql(u8, builtin.name, "web_search")) "googleSearch" else if (std.mem.eql(u8, builtin.name, "code_execution")) "codeExecution" else builtin.name;
        const config: Value = if (builtin.config) |value|
            value.toValue(arena) catch |err| return codecError(err)
        else
            .{ .object = .empty };
        try tools.append(try wrapped(arena, key, config));
    };
    if (tools.items.len != 0) try body.put(arena, "tools", .{ .array = tools });
    const choice = fields.config.tool_choice orelse return;
    if (choice.parallel == false) switch (policy) {
        .refuse => return error.LMFailure,
        .silent => {},
        .note => try records.append(arena, .{
            .field = "config.tool_choice.parallel",
            .action = .dropped,
            .reason = "GenerateContent has no parallel-tool-calls knob and may return several calls (OpenAI and Anthropic carry it)",
            .asked = sdk.PayloadValue.fromBorrowedValue(arena, .{ .bool = false }) catch |err| return codecError(err),
        }),
    };
    var row: Object = .empty;
    try row.put(arena, "mode", str(switch (choice.mode) {
        .none => "NONE",
        .required => "ANY",
        .auto => if (choice.allowed.len == 0) "AUTO" else "VALIDATED",
    }));
    if (choice.allowed.len != 0) {
        for (fields.tools) |tool|
            if (tool == .builtin and contains(choice.allowed, tool.builtin.name)) return error.LMFailure;
        var allowed: runtime.Array = .init(arena);
        try allowed.ensureTotalCapacity(choice.allowed.len);
        for (choice.allowed) |name| allowed.appendAssumeCapacity(str(name));
        try row.put(arena, "allowedFunctionNames", .{ .array = allowed });
    }
    try body.put(arena, "toolConfig", try wrapped(arena, "functionCallingConfig", .{ .object = row }));
}

fn functionDeclarations(arena: Allocator, tools: []const sdk.RequestTool) Error!runtime.Array {
    var declarations: runtime.Array = .init(arena);
    for (tools) |tool| {
        if (tool != .function) continue;
        const function = tool.function;
        var row: Object = .empty;
        try row.put(arena, "name", str(function.name));
        if (function.description) |description| if (description.len != 0)
            try row.put(arena, "description", str(description));
        if (function.parameters) |params| {
            const schema = params.toValue(arena) catch |err| return codecError(err);
            const schema_key = if (try openApi(arena, schema)) "parameters" else "parametersJsonSchema";
            try row.put(arena, schema_key, schema);
        } else {
            // Persistent Live tools bypass Request's parameter defaulting.
            var schema: Object = .empty;
            try schema.put(arena, "type", str("object"));
            try schema.put(arena, "properties", .{ .object = .empty });
            try row.put(arena, "parameters", .{ .object = schema });
        }
        try declarations.append(.{ .object = row });
    }
    return declarations;
}

fn wrapped(arena: Allocator, key: []const u8, value: Value) Error!Value {
    var object: Object = .empty;
    try object.put(arena, key, value);
    return .{ .object = object };
}

fn openApi(arena: Allocator, root: Value) Error!bool {
    var pending: std.ArrayList(Value) = .empty;
    defer pending.deinit(arena);
    try pending.append(arena, root);
    while (pending.pop()) |node| {
        if (node == .bool) return false;
        if (node != .object) continue;
        for (node.object.keys(), node.object.values()) |key, value| {
            if (!contains(&schema_fields, key)) return false;
            if (std.mem.eql(u8, key, "type") and value == .array) return false;
            if (std.mem.eql(u8, key, "enum") and value == .array) {
                for (value.array.items) |item| if (item != .string) return false;
            } else if (std.mem.eql(u8, key, "properties") and value == .object) {
                try pending.appendSlice(arena, value.object.values());
            } else if (std.mem.eql(u8, key, "anyOf") and value == .array) {
                try pending.appendSlice(arena, value.array.items);
            } else if (contains(&.{ "items", "anyOf" }, key)) try pending.append(arena, value);
        }
    }
    return true;
}

const schema_fields = [_][]const u8{
    "type",     "format",        "title",         "description", "nullable",  "enum",    "maxItems", "minItems", "properties",
    "required", "minProperties", "maxProperties", "minLength",   "maxLength", "pattern", "example",  "anyOf",    "propertyOrdering",
    "default",  "items",         "minimum",       "maximum",
};

fn firstCandidate(frame: Value) Error!Value {
    const candidates = get(frame, "candidates");
    if (candidates == .array) {
        const items = candidates.array.items;
        return if (items.len == 0) .null else items[0];
    }
    // DSPy indexes the first candidate after a falsy-value fallback.
    // Strings are indexable, but their first character is not a candidate.
    if (candidates == .string) return .null;
    if (lm_mod.OpenAI.responseTruthy(candidates) catch |err| return codecError(err))
        return error.LMFailure;
    return .null;
}

fn candidateParts(candidate: Value) Error![]const Value {
    return responseElements(get(get(candidate, "content"), "parts"));
}

fn responseElements(value: Value) Error![]const Value {
    switch (value) {
        .bool, .integer, .float, .number_string => {
            // DSPy iterates `value or []`: falsy scalars mean no elements, but
            // a truthy non-iterable is an error, not a successful empty turn.
            if (lm_mod.OpenAI.responseTruthy(value) catch |err| return codecError(err))
                return error.LMFailure;
            return &.{};
        },
        else => return elems(value),
    }
}

fn parseResponse(arena: Allocator, model: []const u8, raw: Value, records: []const sdk.Adaptation) Error!lm_mod.Completion {
    if (raw != .object or get(raw, "error") != .null) return error.LMFailure;
    if (try isPromptBlock(get(get(raw, "promptFeedback"), "blockReason"))) return error.LMFailure;
    const candidate = try firstCandidate(raw);
    const reason = text(get(candidate, "finishReason"));
    if (invalidFinish(reason)) return error.LMFailure;
    const grounding = get(candidate, "groundingMetadata");
    const wire_parts = try candidateParts(candidate);
    const parts = try arena.alloc(sdk.Part, @max(wire_parts.len, 1));
    var visible: std.Io.Writer.Allocating = .init(arena);
    var reasoning: std.Io.Writer.Allocating = .init(arena);
    var calls: std.ArrayList(lm_mod.ToolCall) = .empty;
    var unmapped: runtime.Array = .init(arena);
    var parts_len: usize = 0;
    for (wire_parts, 0..) |part, index| {
        const dest = &parts[parts_len];
        const signature = get(part, "thoughtSignature");
        const text_field = if (part == .object) part.object.get("text") else null;
        if (text_field) |value| {
            // A present null text field still carries part identity and replay state.
            const bytes = if (value == .string) value.string else (try truthyText(arena, value)) orelse "";
            const states = try parseSignature(arena, signature);
            if (lm_mod.OpenAI.responseTruthy(get(part, "thought")) catch |err| return codecError(err)) {
                dest.* = .{ .thinking = .{ .text = bytes, .continuation = states } };
                reasoning.writer.writeAll(bytes) catch return error.OutOfMemory;
            } else {
                dest.* = .{ .text = .{ .text = bytes, .continuation = states } };
                visible.writer.writeAll(bytes) catch return error.OutOfMemory;
            }
            parts_len += 1;
            continue;
        }
        const call = get(part, "functionCall");
        if (call != .object) {
            const inline_data = get(part, "inlineData");
            const file_data = get(part, "fileData");
            if (inline_data == .object or file_data == .object) {
                const media = if (inline_data == .object)
                    try parseMedia(arena, inline_data, .inline_bytes)
                else
                    try parseMedia(arena, file_data, .uri);
                if (media) |value_part| {
                    dest.* = value_part;
                    parts_len += 1;
                }
                continue;
            }
            if (part == .object and (part.object.contains("executableCode") or
                part.object.contains("codeExecutionResult"))) continue;
            var record: Object = .empty;
            try record.put(arena, "path", str(try std.fmt.allocPrint(arena, "candidates[0].content.parts[{d}]", .{index})));
            try record.put(arena, "type", str(try unmappedPartType(arena, part)));
            try unmapped.append(.{ .object = record });
            continue;
        }
        const name = (try truthyText(arena, get(call, "name"))) orelse return error.LMFailure;
        const states = try parseSignature(arena, try functionCallSignature(signature, call));
        const call_id = (try truthyText(arena, get(call, "id"))) orelse
            try std.fmt.allocPrint(arena, "tool_call_{d}", .{parts_len});
        const args = get(call, "args");
        const input: Value = if (args == .object) args else .{ .object = .empty };
        const parsed = sdk.JsonValue.fromBorrowedValue(arena, input) catch |err| return codecError(err);
        var json: std.Io.Writer.Allocating = .init(arena);
        @import("pyjson.zig").writeJson(&json.writer, input, .minified) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
        dest.* = .{ .tool_call = .{
            .name = name,
            .id = call_id,
            .input = parsed,
            .continuation = states,
        } };
        try calls.append(arena, .{
            .name = name,
            .id = call_id,
            .arguments = json.written(),
            .continuation = states,
        });
        parts_len += 1;
    }
    const citations = try readCitations(arena, grounding, visible.written());
    const final_parts = if (citations.len != 0) with_citations: {
        const combined = try arena.realloc(parts, parts_len + citations.len);
        for (combined[parts_len..], citations) |*slot, citation| slot.* = .{ .citation = citation };
        break :with_citations combined;
    } else without_citations: {
        if (parts_len == 0) {
            parts[0] = .{ .text = .{} };
            parts_len = 1;
        }
        break :without_citations parts[0..parts_len];
    };
    const id = try truthyText(arena, get(raw, "responseId"));
    const scores = try readLogprobs(arena, get(candidate, "logprobsResult"));
    const provider_data = if (unmapped.items.len == 0) raw else data: {
        var object = try raw.object.clone(arena);
        try object.put(arena, "_lm15_unmapped", .{ .array = unmapped });
        break :data Value{ .object = object };
    };
    return .{
        .logprobs = if (scores.len == 0) null else scores,
        .text = if (visible.written().len == 0 and calls.items.len != 0) null else visible.written(),
        .reasoning = if (reasoning.written().len != 0) reasoning.written() else null,
        .tool_calls = calls.items,
        .adaptations = records,
        .citations = citations,
        .message = .{ .role = .assistant, .parts = final_parts },
        .provider_data = sdk.PayloadValue.fromBorrowedValue(arena, provider_data) catch |err| return codecError(err),
        .metadata = .{
            .id = id,
            .model = model,
            .finish_reason = finishReason(reason, calls.items.len != 0),
            .usage = try readUsage(arena, get(raw, "usageMetadata")),
        },
    };
}

fn unmappedPartType(arena: Allocator, part: Value) Allocator.Error![]const u8 {
    return switch (part) {
        .object => |object| kind: {
            const keys = object.keys();
            if (keys.len <= 1) break :kind if (keys.len == 1 and keys[0].len != 0) keys[0] else "<empty>";
            const sorted = try arena.dupe([]const u8, keys);
            std.mem.sort([]const u8, sorted, {}, struct {
                fn less(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.less);
            break :kind try std.mem.join(arena, "+", sorted);
        },
        .null => "NoneType",
        .bool => "bool",
        .integer => "int",
        .float => "float",
        .number_string => |number| if (std.json.isNumberFormattedLikeAnInteger(number)) "int" else "float",
        .string => "str",
        .array => "list",
        .custom => unreachable, // Not a wire JSON value.
    };
}

const CitationKey = [3][]const u8;

const CitationContext = struct {
    pub fn hash(_: CitationContext, key: CitationKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (key) |item| {
            const len = item.len;
            hasher.update(std.mem.asBytes(&len));
            hasher.update(item);
        }
        return hasher.final();
    }

    pub fn eql(_: CitationContext, lhs: CitationKey, rhs: CitationKey) bool {
        for (lhs, rhs) |a, b| {
            if (!std.mem.eql(u8, a, b)) return false;
        }
        return true;
    }
};

fn readCitations(arena: Allocator, grounding: Value, answer: []const u8) Error![]const sdk.CitationPart {
    const chunks = elems(get(grounding, "groundingChunks"));
    const supports = elems(get(grounding, "groundingSupports"));
    var seen: std.HashMapUnmanaged(CitationKey, void, CitationContext, 80) = .empty;
    defer seen.deinit(arena);
    var citations: std.ArrayList(sdk.CitationPart) = .empty;
    for (supports) |support| {
        const segment = get(support, "segment");
        const cited = nonempty(get(segment, "text")) orelse try segmentText(segment, answer);
        for (elems(get(support, "groundingChunkIndices"))) |raw_index| {
            var source: Value = .null;
            if (try lm_mod.OpenAI.annotationIndex(raw_index, chunks.len)) |index| {
                if (index < chunks.len) {
                    inline for (.{ "web", "retrievedContext", "googleSearch" }) |name| {
                        const entry = get(chunks[index], name);
                        if (source == .null and try lm_mod.OpenAI.responseTruthy(entry)) source = entry;
                    }
                }
            }
            const url = try citationLabel(arena, source, "uri", "url");
            const title = try citationLabel(arena, source, "title", "name");
            if (url == null and title == null and cited == null) continue;
            const entry = try seen.getOrPut(arena, .{ url orelse "", title orelse "", cited orelse "" });
            if (entry.found_existing) continue;
            try citations.append(arena, .{ .text = cited, .title = title, .url = url });
        }
    }
    return citations.items;
}

fn citationLabel(arena: Allocator, source: Value, primary: []const u8, fallback: []const u8) Error!?[]const u8 {
    const field = get(source, primary);
    const value = if (try lm_mod.OpenAI.responseTruthy(field)) field else get(source, fallback);
    if (!try lm_mod.OpenAI.responseTruthy(value)) return null;
    if (value == .string) return value.string;
    var writer: std.Io.Writer.Allocating = .init(arena);
    @import("pyjson.zig").writeStr(&writer.writer, value) catch |err|
        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
    return writer.written();
}

fn segmentText(segment: Value, answer: []const u8) Error!?[]const u8 {
    const common = lm_mod.OpenAI;
    const start = try common.annotationIndex(get(segment, "startIndex"), answer.len) orelse return null;
    const end = try common.annotationIndex(get(segment, "endIndex"), answer.len) orelse return null;
    if (start >= end) return null;
    const first = common.codepointOffset(answer, start) orelse return null;
    const last = common.codepointOffset(answer[first..], end - start) orelse return null;
    return answer[first..][0..last];
}

fn readLogprobs(arena: Allocator, value: Value) Error![]const lm_mod.TokenLogprob {
    const chosen = try responseElements(get(value, "chosenCandidates"));
    const top_steps = get(value, "topCandidates");
    var scores: std.ArrayList(lm_mod.TokenLogprob) = .empty;
    for (chosen, 0..) |item, index| {
        if (item != .object) continue;
        // Alternate steps are accessed by numeric position only when there
        // is a chosen token. A JSON object cannot supply an integer key.
        if (top_steps == .object and index < top_steps.object.count()) return error.LMFailure;
        const alternatives = try responseElements(top_steps);
        var top: std.ArrayList(lm_mod.TopLogprob) = .empty;
        if (index < alternatives.len) {
            for (try responseElements(get(alternatives[index], "candidates"))) |alt|
                if (alt == .object) try top.append(arena, try readScore(arena, alt));
        }
        const token = try readScore(arena, item);
        try scores.append(arena, .{
            .token = token.token,
            .token_id = token.token_id,
            .logprob = token.logprob,
            .top = top.items,
        });
    }
    return scores.items;
}

fn readScore(arena: Allocator, item: Value) Error!lm_mod.TopLogprob {
    const numbers = @import("value.zig");
    const raw = get(item, "logProbability");
    const truthy = lm_mod.OpenAI.responseTruthy;
    const score: f64 = if (!(truthy(raw) catch return error.LMFailure))
        0
    else if (raw == .string)
        numbers.floatFromString(arena, raw.string) catch |err| return codecError(err)
    else
        numbers.numericFloat(raw) catch return error.LMFailure;
    const token = get(item, "token");
    const bytes = if (!(truthy(token) catch return error.LMFailure)) "" else if (token == .string)
        token.string
    else formatted: {
        var writer: std.Io.Writer.Allocating = .init(arena);
        @import("pyjson.zig").writeStr(&writer.writer, token) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
        break :formatted writer.written();
    };
    return .{
        .token = bytes,
        .token_id = @import("logprobs.zig").normalizeBorrowedId(arena, get(item, "tokenId")) catch |err|
            return codecError(err),
        .logprob = score,
    };
}

fn parseMedia(arena: Allocator, value: Value, mode: enum { inline_bytes, uri }) Error!?sdk.Part {
    const address = (try truthyText(arena, get(value, if (mode == .inline_bytes) "data" else "fileUri"))) orelse return null;
    const mime = (try truthyText(arena, get(value, "mimeType"))) orelse "application/octet-stream";
    const media: sdk.MediaPart = .{
        .media_type = mime,
        .data = if (mode == .inline_bytes) address else null,
        .url = if (mode == .uri) address else null,
    };
    if (std.mem.startsWith(u8, mime, "image/")) return .{ .image = media };
    if (std.mem.startsWith(u8, mime, "audio/")) return .{ .audio = media };
    return .{ .document = media };
}

fn readUsage(arena: Allocator, usage: Value) Error!sdk.Usage {
    if (usage != .object or usage.object.count() == 0) return .{};
    const common = lm_mod.OpenAI;
    const answer_key = if (usage.object.contains("candidatesTokenCount")) "candidatesTokenCount" else "responseTokenCount";
    const thoughts = try count(arena, usage, "thoughtsTokenCount");
    const prompt = try modality(arena, get(usage, "promptTokensDetails"));
    const tools = try modality(arena, get(usage, "toolUsePromptTokensDetails"));
    const candidate_details = get(usage, "candidatesTokensDetails");
    // Select the wire field before checking whether it holds a breakdown array.
    const output_details = if (try common.responseTruthy(candidate_details))
        candidate_details
    else
        get(usage, "responseTokensDetails");
    return common.settleUsage(arena, .{
        .input_tokens = try common.countSum(arena, try count(arena, usage, "promptTokenCount"), try count(arena, usage, "toolUsePromptTokenCount")),
        .output_tokens = try common.countSum(arena, try count(arena, usage, answer_key), thoughts),
        .total_tokens = try common.wireCount(arena, get(usage, "totalTokenCount")),
        .cache_read_tokens = try count(arena, usage, "cachedContentTokenCount"),
        .reasoning_tokens = thoughts,
        .input_audio_tokens = if (prompt == .null) tools else if (tools == .null) prompt else try common.countSum(arena, prompt, tools),
        .output_audio_tokens = try modality(arena, output_details),
    });
}

fn count(arena: Allocator, object: Value, key: []const u8) Error!Value {
    const value = try lm_mod.OpenAI.wireCount(arena, get(object, key));
    return if (value == .null) .{ .integer = 0 } else value;
}

fn modality(arena: Allocator, details: Value) Error!Value {
    var total: Value = .null;
    for (elems(details)) |row| {
        if (!std.mem.eql(u8, text(get(row, "modality")), "AUDIO")) continue;
        // DSPy sums the raw rows before validating the aggregate as SDK Usage.
        // Starting at zero also promotes a JSON boolean to a numeric count.
        const raw = row.object.get("tokenCount") orelse Value{ .integer = 0 };
        switch (raw) {
            .bool, .integer, .number_string, .float => {},
            else => return error.LMFailure,
        }
        const subtotal = if (total == .null) Value{ .integer = 0 } else total;
        total = @import("value.zig").numericSum(arena, subtotal, raw) catch |err|
            return codecError(err);
    }
    return total;
}

fn joinedParts(arena: Allocator, leading_line: ?[]const u8, parts: anytype) Error!Value {
    const rendered = try content.partsText(arena, parts, leading_line);
    return singleton(arena, try textPart(arena, rendered));
}

fn singleton(arena: Allocator, part: Value) Error!Value {
    var list: runtime.Array = .init(arena);
    try list.append(part);
    return .{ .array = list };
}

fn textPart(arena: Allocator, bytes: []const u8) Error!Value {
    var part: Object = .empty;
    try part.put(arena, "text", str(bytes));
    return .{ .object = part };
}

fn str(bytes: []const u8) Value {
    return .{ .string = bytes };
}
fn text(value: Value) []const u8 {
    return if (value == .string) value.string else "";
}
fn get(value: Value, key: []const u8) Value {
    return if (value == .object) value.object.get(key) orelse .null else .null;
}
fn elems(value: Value) []const Value {
    return if (value == .array) value.array.items else &.{};
}
fn contains(values: []const []const u8, needle: []const u8) bool {
    for (values) |value| if (std.mem.eql(u8, value, needle)) return true;
    return false;
}
fn codecError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
}

test "Gemini separates structured schema dialects and rejects blocked candidates" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const schema = try parse(arena,
        \\{"type":"object","properties":{"answer":{"anyOf":[{"type":"string"},false]}}}
    );
    try std.testing.expect(!try openApi(arena, schema));
    const ordinary = try parse(arena,
        \\{"type":"object","properties":{"answer":{"type":"string","default":{"x":false}}}}
    );
    try std.testing.expect(try openApi(arena, ordinary));
    const blocked = try parse(arena,
        \\{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}
    );
    try std.testing.expectError(error.LMFailure, parseResponse(arena, "fixture", blocked, &.{}));
}

test "Gemini judgment enum takes precedence over non-const branches" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var schema = try @import("json_repair.zig").parseJson(arena,
        \\{"type":"object","properties":{"grade":{
        \\"enum":["ok","no"],"anyOf":[{"type":"string"}]}}}
    );
    const found = try @import("typesafe.zig").responseJudgments(arena, schema);
    try encodeJudgmentSchema(arena, &schema, found);
    const grade = get(get(schema, "properties"), "grade");
    try std.testing.expectEqualStrings("string", text(get(grade, "type")));
    try std.testing.expect(get(grade, "anyOf") == .null);
    try std.testing.expectEqualStrings("ok", text(elems(get(grade, "enum"))[0]));
}

test "Gemini extensions override top-level wire fields without leaking control keys" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena,
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"Answer"}]}],
        \\"config":{"user_id":"actor","max_tokens":24,"extensions":{"output":"audio",
        \\"prompt_caching":true,"generationConfig":{"responseModalities":["TEXT"]}}}}
    ));
    defer request.release();

    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try parse(arena, try requestBody(arena, std.testing.io, &request.fields, request.fields.model, .canonical, .note, &records));
    const generation = get(body, "generationConfig");
    try std.testing.expectEqualStrings("TEXT", text(elems(get(generation, "responseModalities"))[0]));
    try std.testing.expectEqual(@as(usize, 1), generation.object.count());
    try std.testing.expect(!body.object.contains("output"));
    try std.testing.expect(!body.object.contains("prompt_caching"));
    try std.testing.expectEqualStrings("config.user_id", records.items[0].field);
    try std.testing.expectError(
        error.LMFailure,
        requestBody(arena, std.testing.io, &request.fields, request.fields.model, .canonical, .refuse, &records),
    );
}

test "Gemini projects canonical structured tool results and ordered attachments" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parse = @import("json_repair.zig").parseJson;
    // Expected functionResponse fields captured from DSPy 3.5.0 GeminiLM.complete.
    const image = "{\"type\":\"image\",\"media_type\":\"image/png\",\"data\":\"aW1hZ2U=\"}";
    const document = "{\"type\":\"document\",\"media_type\":\"application/pdf\",\"url\":\"https://example.test/report.pdf\"}";
    const cases = [_]struct {
        parts: []const u8,
        is_error: bool = false,
        response_key: ?[]const u8 = "result",
        response_text: []const u8 = "",
        documents: usize = 1,
    }{
        .{
            .parts = "{\"type\":\"text\",\"text\":\"Result:\"}," ++ image ++
                ",{\"type\":\"data\",\"value\":{\"city\":\"Zürich\",\"count\":9007199254740993}}," ++
                "{\"type\":\"text\",\"text\":\"\"}," ++ document ++
                ",{\"type\":\"citation\",\"title\":\"Weather\",\"url\":\"https://example.test/weather\",\"text\":\"Station reading\"}",
            .response_text = "Result:\n{\"city\":\"Zürich\",\"count\":9007199254740993}\n\n" ++
                "Weather — https://example.test/weather — Station reading",
        },
        .{ .parts = image ++ "," ++ document, .response_key = null },
        .{ .parts = image ++ "," ++ document, .is_error = true, .response_key = "error" },
        .{
            .parts = "{\"type\":\"data\",\"value\":{\"message\":\"indisponível\"}}," ++ image,
            .is_error = true,
            .response_key = "error",
            .response_text = "{\"message\":\"indisponível\"}",
            .documents = 0,
        },
    };
    for (cases) |case| {
        const request_json = try std.mem.concat(arena, u8, &.{
            \\{"model":"gemini-3-flash","messages":[
            \\{"role":"user","parts":[{"type":"text","text":"Look up the weather"}]},
            \\{"role":"assistant","parts":[{"type":"tool_call","id":"call-1","name":"lookup","input":{}}]},
            \\{"role":"tool","parts":[{"type":"tool_result","id":"call-1","is_error":
            ,
            if (case.is_error) "true" else "false",
            ",\"content\":[",
            case.parts,
            "]}]}]}",
        });
        const request = try sdk.Request.fromValue(arena, try parse(arena, request_json));
        defer request.release();

        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try parse(arena, try requestBody(arena, std.testing.io, &request.fields, request.fields.model, .canonical, .note, &records));
        const rows = elems(get(body, "contents"));
        try std.testing.expectEqual(@as(usize, 3), rows.len);
        try std.testing.expectEqualStrings("user", text(get(rows[2], "role")));
        const parts = elems(get(rows[2], "parts"));
        try std.testing.expectEqual(@as(usize, 1), parts.len);
        const result = get(parts[0], "functionResponse");
        try std.testing.expectEqualStrings("lookup", text(get(result, "name")));
        try std.testing.expectEqualStrings("call-1", text(get(result, "id")));

        const response = get(result, "response");
        try std.testing.expect(response == .object);
        if (case.response_key) |key| {
            try std.testing.expectEqual(@as(usize, 1), response.object.count());
            const field = get(response, key);
            try std.testing.expect(field == .string);
            try std.testing.expectEqualStrings(case.response_text, field.string);
        } else {
            try std.testing.expectEqual(@as(usize, 0), response.object.count());
        }
        const attachments = elems(get(result, "parts"));
        try std.testing.expectEqual(1 + case.documents, attachments.len);
        const inline_data = get(attachments[0], "inlineData");
        try std.testing.expectEqualStrings("image/png", text(get(inline_data, "mimeType")));
        try std.testing.expectEqualStrings("aW1hZ2U=", text(get(inline_data, "data")));
        if (case.documents != 0) {
            const file_data = get(attachments[1], "fileData");
            try std.testing.expectEqualStrings("application/pdf", text(get(file_data, "mimeType")));
            try std.testing.expectEqualStrings("https://example.test/report.pdf", text(get(file_data, "fileUri")));
        }
        try std.testing.expectEqual(@as(usize, 0), records.items.len);
    }
}

test "Gemini completeRequest accepts structured tool results but refuses audio before credentials" {
    const Probe = struct {
        fn prepare(ptr: *anyopaque, _: Allocator, _: []const std.http.Header) anyerror![]const std.http.Header {
            const reached: *bool = @ptrCast(@alignCast(ptr));
            reached.* = true;
            return error.StopBeforeNetwork;
        }
    };
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = std.testing.io };
    defer client.deinit();
    var reached = false;
    var provider: Gemini = .{
        .client = &client,
        .model = "gemini-3-flash",
        .base_url = "http://invalid.invalid",
        .header_provider = .{ .context = &reached, .prepare = Probe.prepare },
    };
    const data = try sdk.JsonValue.fromValue(arena, .{ .integer = 1 });
    defer data.free();
    const cases: []const sdk.ContentPart = &.{
        .{ .audio = .{ .media_type = "audio/wav", .data = "YXVkaW8=" } },
        .{ .data = .{ .value = data } },
    };
    for (cases) |part| {
        const request = try sdk.Request.fromFields(arena, &.{
            .model = provider.model,
            .messages = &.{.{ .role = .tool, .parts = &.{.{ .tool_result = .{
                .id = "call-1",
                .name = "lookup",
                .content = &.{part},
            } }} }},
        });
        defer request.release();

        reached = false;
        try std.testing.expectError(error.LMFailure, provider.completeRequest(arena, request));
        try std.testing.expectEqual(part == .data, reached);
    }
}

test "Gemini reads local media anew for each GenerateContent request" {
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, sse: bool) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var req = try server.receiveHead();
            try std.testing.expectEqual(std.http.Method.POST, req.head.method);
            try std.testing.expectEqualStrings(if (sse)
                "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse"
            else
                "/v1beta/models/gemini-3-flash:generateContent", req.head.target);
            // Both complete bodies were captured from the locked Python GeminiLM.
            const prefix = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Find it\"},{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"";
            const suffix = "\"}}]},{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{},\"id\":\"c1\"}}]}," ++
                "{\"role\":\"user\",\"parts\":[{\"functionResponse\":{\"name\":\"lookup\",\"response\":{},\"id\":\"c1\",\"parts\":[" ++
                "{\"inlineData\":{\"mimeType\":\"application/pdf\",\"data\":\"UERGAA==\"}}]}}]}]}";
            const expected = if (sse) prefix ++ "SU1BR0U=" ++ suffix else prefix ++ "aW1hZ2U=" ++ suffix;
            try std.testing.expectEqual(expected.len, req.head.content_length orelse return error.TestUnexpectedResult);
            var body: [prefix.len + 8 + suffix.len]u8 = undefined;
            var body_buffer: [256]u8 = undefined;
            try (try req.readerExpectContinue(&body_buffer)).readSliceAll(&body);
            try std.testing.expectEqualStrings(expected, &body);
            const reply = "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Found it\"}]},\"finishReason\":\"STOP\"}]}";
            try req.respond(if (sse) "data: " ++ reply ++ "\n\n" else reply, .{
                .keep_alive = false,
                .extra_headers = &.{.{ .name = "content-type", .value = if (sse) "text/event-stream" else "application/json" }},
            });
        }
    };
    const Capture = struct {
        events: [3]u8 = undefined,
        len: usize = 0,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (self.len == self.events.len) return .stop;
            self.events[self.len] = switch (event) {
                .start => 's',
                .delta => if (event.delta.data == .text and
                    std.mem.eql(u8, event.delta.data.text.text, "Found it")) 'd' else '?',
                .end => if (event.end.finish_reason == .stop) 'e' else '?',
                else => '?',
            };
            self.len += 1;
            return .proceed;
        }
    };
    const io = std.testing.io;
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var directory = std.testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(io, .{ .sub_path = "local.png", .data = "image" });
    try directory.dir.writeFile(io, .{ .sub_path = "local.pdf", .data = "PDF\x00" });
    const image = try directory.dir.realPathFileAlloc(io, "local.png", arena);
    const pdf = try directory.dir.realPathFileAlloc(io, "local.pdf", arena);
    const tool_input = try sdk.JsonValue.fromValue(arena, .{ .object = .empty });
    defer tool_input.free();
    const request = try sdk.Request.fromFields(arena, &.{
        .model = "gemini-3-flash",
        .messages = &.{
            .{ .role = .user, .parts = &.{
                .{ .text = .{ .text = "Find it" } },
                .{ .image = .{ .media_type = "image/png", .path = image } },
            } },
            .{ .role = .assistant, .parts = &.{.{ .tool_call = .{ .id = "c1", .name = "lookup", .input = tool_input } }} },
            .{ .role = .tool, .parts = &.{.{ .tool_result = .{
                .id = "c1",
                .content = &.{.{ .document = .{ .media_type = "application/pdf", .path = pdf } }},
            } }} },
        },
    });
    defer request.release();
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local-fixture",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    for ([_]bool{ false, true }) |sse| {
        if (sse) try directory.dir.writeFile(io, .{ .sub_path = "local.png", .data = "IMAGE" });
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, sse });
        defer serving.cancel(io) catch {};
        if (sse) {
            var capture: Capture = .{};
            try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
                .context = &capture,
                .on_event = Capture.onEvent,
            }));
            try std.testing.expectEqualStrings("sde", capture.events[0..capture.len]);
        } else {
            const response = try gemini.completeRequest(arena, request);
            defer response.release();
            try std.testing.expectEqualStrings("Found it", response.fields.message.parts[0].text.text);
        }
        try serving.await(io);
    }
    try directory.dir.deleteFile(io, "local.pdf");
    const Probe = struct {
        fn prepare(context: *anyopaque, _: Allocator, _: []const std.http.Header) anyerror![]const std.http.Header {
            const reached: *bool = @ptrCast(@alignCast(context));
            reached.* = true;
            return error.TestUnexpectedResult;
        }
    };
    var reached = false;
    gemini.header_provider = .{ .context = &reached, .prepare = Probe.prepare };
    try std.testing.expectError(error.LMFailure, gemini.completeRequest(arena, request));
    var capture: Capture = .{};
    try std.testing.expectError(error.LMFailure, gemini.streamRequest(arena, request, .{
        .context = &capture,
        .on_event = Capture.onEvent,
    }));
    try std.testing.expect(!reached);
    try std.testing.expectEqual(@as(usize, 0), capture.len);
}

test "Gemini replays structured tool results and media with continuation over HTTP" {
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, streaming: bool) !void {
            // Both DSPy 3.5.0 generation modes project refusal history as plain text.
            const expected = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Find the count\"}," ++
                "{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"aW1hZ2U=\"}}]}," ++
                "{\"role\":\"model\",\"parts\":[{\"text\":\"\",\"thought\":true,\"thoughtSignature\":9007199254740993}," ++
                "{\"text\":\"I can’t disclose private records.\"}," ++
                "{\"text\":\"Checking\"},{\"functionCall\":{\"name\":\"count\",\"args\":{},\"id\":\"call-2\"}," ++
                "\"thoughtSignature\":{\"opaque\":\"東京\"}}]}," ++
                "{\"role\":\"user\",\"parts\":[{\"functionResponse\":{\"name\":\"count\",\"response\":" ++
                "{\"result\":\"{\\\"observed\\\":[9007199254740993,\\\"東京\\\"]}" ++
                "\\nRegistry\\nobserved\\n\\nverified\"},\"id\":\"call-2\",\"parts\":[" ++
                "{\"fileData\":{\"mimeType\":\"application/pdf\",\"fileUri\":\"files/registry\"}}]}}]}]}";
            const stream = try listener.accept(io);
            defer stream.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = stream.reader(io, &read_buffer);
            var output = stream.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try std.testing.expectEqual(std.http.Method.POST, request.head.method);
            try std.testing.expectEqualStrings(if (streaming)
                "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse"
            else
                "/v1beta/models/gemini-3-flash:generateContent", request.head.target);
            try std.testing.expectEqualStrings("application/json", request.head.content_type orelse "");
            try std.testing.expectEqual(expected.len, request.head.content_length orelse return error.TestUnexpectedResult);
            var body_buffer: [256]u8 = undefined;
            var actual: [expected.len]u8 = undefined;
            const reader = try request.readerExpectContinue(&body_buffer);
            try reader.readSliceAll(&actual);
            try std.testing.expectEqualStrings(expected, &actual);
            const reply =
                \\{"responseId":9007199254740993,"candidates":[{"content":{"parts":[{"text":"Found it"}]},"finishReason":"STOP"}]}
            ;
            try request.respond(if (streaming) "data: " ++ reply ++ "\n\n" else reply, .{
                .keep_alive = false,
                .extra_headers = &.{.{ .name = "content-type", .value = if (streaming) "text/event-stream" else "application/json" }},
            });
        }
    };
    const Capture = struct {
        events: [3]u8 = undefined,
        len: usize = 0,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (self.len == self.events.len) return .stop;
            self.events[self.len] = switch (event) {
                .start => 's',
                .delta => if (event.delta.data == .text and
                    std.mem.eql(u8, event.delta.data.text.text, "Found it")) 'd' else '?',
                .end => if (event.end.finish_reason == .stop) 'e' else '?',
                else => '?',
            };
            self.len += 1;
            return .proceed;
        }
    };

    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local-fixture",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
        .connect_timeout_ns = 5 * std.time.ns_per_s,
    };
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena,
        \\{"model":"gemini-3-flash","messages":[
        \\{"role":"user","parts":[{"type":"text","text":"Find the count"},
        \\{"type":"image","media_type":"image/png","data":"aW1hZ2U=",
        \\"continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":"image-opaque"}}]}]},
        \\{"role":"assistant","parts":[
        \\{"type":"thinking","text":"","continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":9007199254740993}}]},
        \\{"type":"refusal","text":"I can’t disclose private records.","continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":"not-for-wire"}}]},
        \\{"type":"text","text":"Checking","continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":false}}]},
        \\{"type":"tool_call","id":"call-2","name":"count","input":{},"continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":{"opaque":"東京"}}}]}]},
        \\{"role":"tool","parts":[{"type":"tool_result","id":"call-2",
        \\"continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":"tool-result-opaque"}}],"content":[
        \\{"type":"data","value":{"observed":[9007199254740993,"東京"]}},
        \\{"type":"citation","title":"Registry"},
        \\{"type":"text","text":"observed","continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":"tool-opaque"}}]},
        \\{"type":"text","text":""},{"type":"text","text":"verified"},
        \\{"type":"document","media_type":"application/pdf","file_id":"files/registry",
        \\"continuation":[{"provider":"gemini","kind":"thought_signature","data":{"value":"document-opaque"}}]}]}]}]}
    ));
    defer request.release();

    for ([_]bool{ false, true }) |streaming| {
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, streaming });
        defer serving.cancel(io) catch {};
        if (streaming) {
            var capture: Capture = .{};
            try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
                .context = &capture,
                .on_event = Capture.onEvent,
            }));
            try std.testing.expectEqualStrings("sde", capture.events[0..capture.len]);
        } else {
            const reply = try gemini.completeRequest(arena, request);
            defer reply.release();
            // DSPy exposes truthy wire IDs as strings, without rounding integers.
            try std.testing.expectEqualStrings("9007199254740993", reply.fields.id orelse "");
            try std.testing.expectEqual(sdk.FinishReason.stop, reply.fields.finish_reason);
            try std.testing.expectEqual(@as(usize, 1), reply.fields.message.parts.len);
            const part = reply.fields.message.parts[0];
            try std.testing.expect(part == .text);
            try std.testing.expectEqualStrings("Found it", part.text.text);
        }
        try serving.await(io);
    }
}

test "Gemini generation removes only its first model binding prefix" {
    const Fixture = struct {
        const body =
            "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}";

        fn serve(io: std.Io, listener: *std.Io.net.Server, model_path: []const u8, streaming: bool) !void {
            const connection = try listener.accept(io);
            defer connection.close(io);
            var read_buffer: [2048]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var reader = connection.reader(io, &read_buffer);
            var writer = connection.writer(io, &write_buffer);
            var server = std.http.Server.init(&reader.interface, &writer.interface);
            var request = try server.receiveHead();
            var target: [128]u8 = undefined;
            const expected = try std.fmt.bufPrint(&target, "/v1beta/{s}:{s}", .{
                model_path,
                if (streaming) "streamGenerateContent?alt=sse" else "generateContent",
            });
            // Save mismatches before responding: the client's decoded result can
            // otherwise succeed even when the gateway received a different model.
            var actual: [128]u8 = undefined;
            const received = try std.fmt.bufPrint(&actual, "{s}", .{request.head.target});
            var transfer_buffer: [256]u8 = undefined;
            const input = try request.readerExpectContinue(&transfer_buffer);
            const prompt = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Capital?\"}]}]}";
            var posted: [prompt.len]u8 = undefined;
            try std.testing.expectEqual(@as(?u64, prompt.len), request.head.content_length);
            try input.readSliceAll(&posted);
            const response = if (streaming) "data: " ++ body ++ "\n\n" else body;
            try request.respond(response, .{
                .keep_alive = false,
                .extra_headers = &.{.{
                    .name = "content-type",
                    .value = if (streaming) "text/event-stream" else "application/json",
                }},
            });
            try std.testing.expectEqualStrings(prompt, &posted);
            try std.testing.expectEqualStrings(expected, received);
        }
    };
    const Capture = struct {
        model: []const u8,
        events: usize = 0,
        matches: bool = true,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.matches = self.matches and switch (self.events) {
                0 => event == .start and std.mem.eql(u8, self.model, event.start.model),
                1 => event == .delta and event.delta.data == .text and
                    std.mem.eql(u8, "ok", event.delta.data.text.text),
                2 => event == .end and event.end.finish_reason == .stop,
                else => false,
            };
            self.events += 1;
            return .proceed;
        }
    };
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local-fixture",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const cases = [_][3][]const u8{
        .{ "gemini:models/gemini-3-flash", "models/gemini-3-flash", "models/gemini-3-flash" },
        .{ "gemini:gemini:custom", "models/gemini:custom", "gemini:custom" },
        .{ "proxy:custom", "models/proxy:custom", "proxy:custom" },
        .{ "gemini:", "models/gemini:", "gemini:" },
    };
    for (cases) |case| {
        const request = try sdk.Request.fromFields(arena, &.{
            .model = case[0],
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Capital?" } }} }},
        });
        defer request.release();
        for ([_]bool{ false, true }) |streaming| {
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, case[1], streaming });
            defer serving.cancel(io) catch {};
            if (streaming) {
                var capture: Capture = .{ .model = case[2] };
                try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
                    .context = &capture,
                    .on_event = Capture.onEvent,
                }));
                try serving.await(io);
                try std.testing.expect(capture.matches);
                try std.testing.expectEqual(@as(usize, 3), capture.events);
            } else {
                const response = try gemini.completeRequest(arena, request);
                defer response.release();
                try serving.await(io);
                try std.testing.expectEqualStrings(case[2], response.fields.model);
                try std.testing.expectEqualStrings("ok", response.fields.message.parts[0].text.text);
            }
        }
        try std.testing.expectEqualStrings(case[0], request.fields.model);
    }
}

test "Gemini completeRequest replays signed thinking with null text" {
    try expectSignatureReplay(.buffered, .string);
}

test "Gemini streamed completion replays signed thinking with null text" {
    try expectSignatureReplay(.streamed, .string);
}

test "Gemini replays non-string thought signatures without losing precision" {
    try expectSignatureReplay(.buffered, .scalar);
    try expectSignatureReplay(.streamed, .scalar);
}

test "Gemini replays function-call correlators and thought signatures" {
    try expectSignatureReplay(.buffered, .tool);
    try expectSignatureReplay(.streamed, .tool);
}

const SignatureReplayMode = enum { buffered, streamed };
const SignatureReplayKind = enum { string, scalar, tool };

fn expectSignatureReplay(mode: SignatureReplayMode, kind: SignatureReplayKind) !void {
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, streamed: bool, signature_kind: SignatureReplayKind) !void {
            // Exact request bodies from the pinned DSPy GeminiLM loopbacks.
            const scalar = signature_kind == .scalar;
            const tool = signature_kind == .tool;
            const turns = [_]struct { body: []const u8, reply: []const u8 }{
                .{
                    .body = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Capital?\"}]}]}",
                    .reply = if (tool)
                        "{\"candidates\":[{\"content\":{\"parts\":[" ++
                            "{\"thoughtSignature\":9007199254740993,\"functionCall\":{\"id\":9007199254740993,\"name\":true,\"args\":{},\"thoughtSignature\":\"stale\"}}," ++
                            "{\"thoughtSignature\":false,\"functionCall\":{\"id\":false,\"name\":\"lookup\",\"thoughtSignature\":0}}]},\"finishReason\":\"STOP\"}]}"
                    else if (scalar)
                        "{\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":null,\"thoughtSignature\":9007199254740993},{\"text\":\"Paris\",\"thoughtSignature\":false}]},\"finishReason\":\"STOP\"}]}"
                    else
                        "{\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":null,\"thoughtSignature\":\"opaque-null\"},{\"text\":\"Paris\"}]},\"finishReason\":\"STOP\"}]}",
                },
                .{
                    .body = if (tool)
                        "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Capital?\"}]}," ++
                            "{\"role\":\"model\",\"parts\":[" ++
                            "{\"functionCall\":{\"name\":\"True\",\"args\":{},\"id\":\"9007199254740993\"},\"thoughtSignature\":\"9007199254740993\"}," ++
                            "{\"functionCall\":{\"name\":\"lookup\",\"args\":{},\"id\":\"tool_call_1\"},\"thoughtSignature\":\"0\"}]}," ++
                            "{\"role\":\"user\",\"parts\":[{\"text\":\"Country?\"}]}]}"
                    else if (scalar)
                        "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Capital?\"}]}," ++
                            "{\"role\":\"model\",\"parts\":[{\"text\":\"\",\"thought\":true,\"thoughtSignature\":\"9007199254740993\"},{\"text\":\"Paris\",\"thoughtSignature\":\"False\"}]}," ++
                            "{\"role\":\"user\",\"parts\":[{\"text\":\"Country?\"}]}]}"
                    else
                        "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Capital?\"}]}," ++
                            "{\"role\":\"model\",\"parts\":[{\"text\":\"\",\"thought\":true,\"thoughtSignature\":\"opaque-null\"},{\"text\":\"Paris\"}]}," ++
                            "{\"role\":\"user\",\"parts\":[{\"text\":\"Country?\"}]}]}",
                    .reply = "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"France\"}]},\"finishReason\":\"STOP\"}]}",
                },
            };
            for (turns, 0..) |turn, index| {
                const stream = try listener.accept(io);
                defer stream.close(io);
                var read_buffer: [4096]u8 = undefined;
                var write_buffer: [1024]u8 = undefined;
                var input = stream.reader(io, &read_buffer);
                var output = stream.writer(io, &write_buffer);
                var server = std.http.Server.init(&input.interface, &output.interface);
                var request = try server.receiveHead();
                const sse = streamed and index == 0;
                try std.testing.expectEqualStrings(if (sse)
                    "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse"
                else
                    "/v1beta/models/gemini-3-flash:generateContent", request.head.target);
                var body_buffer: [256]u8 = undefined;
                var body: [512]u8 = undefined;
                const len = request.head.content_length orelse return error.TestUnexpectedResult;
                if (len > body.len) return error.TestUnexpectedResult;
                try (try request.readerExpectContinue(&body_buffer)).readSliceAll(body[0..len]);
                if (sse) {
                    var frame: [512]u8 = undefined;
                    const data = try std.fmt.bufPrint(&frame, "data: {s}\n\n", .{turn.reply});
                    try request.respond(data, .{
                        .keep_alive = false,
                        .extra_headers = &.{.{ .name = "content-type", .value = "text/event-stream" }},
                    });
                } else {
                    try request.respond(turn.reply, .{ .keep_alive = false });
                }
                try std.testing.expectEqualStrings(turn.body, body[0..len]);
            }
        }
    };

    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local-fixture",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
        .connect_timeout_ns = 5 * std.time.ns_per_s,
    };

    var serving = try io.concurrent(Fixture.serve, .{ io, &listener, mode == .streamed, kind });
    defer serving.cancel(io) catch {};
    const first = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Capital?" } }} }},
    });
    defer first.release();
    const reply = if (mode == .streamed) streamed: {
        const Observer = struct {
            fn onEvent(_: *anyopaque, _: Gemini.StreamEvent) Gemini.StreamSink.Action {
                return .proceed;
            }
        };
        const completions = try gemini.lm().completeResponses(arena, .{
            .messages = &.{.{ .role = .user, .content = .{ .text = "Capital?" } }},
            .stream_sink = .{ .context = &gemini, .on_event = Observer.onEvent },
        });
        break :streamed completions[0].response orelse return error.TestUnexpectedResult;
    } else try gemini.completeRequest(arena, first);
    defer reply.release();
    const second = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{ first.fields.messages[0], reply.fields.message, .{
            .role = .user,
            .parts = &.{.{ .text = .{ .text = "Country?" } }},
        } },
    });
    defer second.release();
    const final = try gemini.completeRequest(arena, second);
    defer final.release();
    try serving.await(io);
    try std.testing.expectEqualStrings("France", final.fields.message.parts[0].text.text);
    const parts = reply.fields.message.parts;
    try std.testing.expectEqual(@as(usize, 2), parts.len);
    if (kind == .tool) {
        try std.testing.expectEqual(sdk.FinishReason.tool_call, reply.fields.finish_reason);
        try std.testing.expect(parts[0] == .tool_call and parts[1] == .tool_call);
        try std.testing.expectEqualStrings("9007199254740993", parts[0].tool_call.id);
        try std.testing.expectEqualStrings("tool_call_1", parts[1].tool_call.id);
    } else {
        try std.testing.expect(parts[0] == .thinking and parts[1] == .text);
        try std.testing.expectEqualStrings("", parts[0].thinking.text);
        try std.testing.expectEqualStrings("Paris", parts[1].text.text);
    }
}

test "Gemini replays grounded assistant citations as plain text parts" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const raw = try parse(arena,
        \\{"candidates":[{"content":{"parts":[{"text":"First answer"}]},"groundingMetadata":{
        \\"groundingChunks":[{"web":{"title":"Registry","uri":"https://example.test/registry"}}],
        \\"groundingSupports":[{"segment":{"text":"Quoted evidence"},"groundingChunkIndices":[0]},
        \\{"groundingChunkIndices":[0]}]},"finishReason":"STOP"}]}
    );
    const model = "gemini-3-flash";
    const first = try parseResponse(arena, model, raw, &.{});
    const assistant = first.message orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 3), assistant.parts.len);
    try std.testing.expect(assistant.parts[1] == .citation and assistant.parts[2] == .citation);
    try std.testing.expectEqualStrings("Registry", assistant.parts[2].citation.title orelse "");
    try std.testing.expectEqualStrings("https://example.test/registry", assistant.parts[2].citation.url orelse "");
    try std.testing.expectEqual(null, assistant.parts[2].citation.text);

    const request = try sdk.Request.fromFields(arena, &.{
        .model = model,
        .messages = &.{
            .{ .role = .user, .parts = &.{.{ .text = .{ .text = "Read" } }} },
            assistant,
            .{ .role = .user, .parts = &.{.{ .text = .{ .text = "Explain" } }} },
        },
    });
    defer request.release();
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try requestBody(arena, std.testing.io, &request.fields, request.fields.model, .canonical, .note, &records);
    // Captured from the second GeminiLM.complete call using the same first response.
    try std.testing.expectEqualStrings(
        \\{"contents":[{"role":"user","parts":[{"text":"Read"}]},{"role":"model","parts":[{"text":"First answer"},{"text":"Quoted evidence"},{"text":""}]},{"role":"user","parts":[{"text":"Explain"}]}]}
    , body);
}

test "Gemini retains tool call position in mixed responses" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[{"text":"Looking up"},{"functionCall":{"name":"weather","args":{"city":"Paris"}}}]},"finishReason":"MAX_TOKENS"}]}
    );
    const result = try parseResponse(arena, "fixture", raw, &.{});
    try std.testing.expectEqual(sdk.FinishReason.tool_call, result.metadata.finish_reason.?);
    try std.testing.expectEqualStrings("tool_call_1", result.message.?.parts[1].tool_call.id);
    try std.testing.expectEqualStrings("tool_call_1", result.tool_calls[0].id);
}

test "Gemini separates answer text from signed and unsigned thought summaries" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[
        \\ {"thought":true,"text":"","thoughtSignature":"opaque"},
        \\ {"thought":true,"text":"Check "},{"thought":true,"text":"the answer"},
        \\ {"text":"Paris","thoughtSignature":"answer-state"}
        \\]}}]}
    );
    const result = try parseResponse(arena, "gemini-3-flash", raw, &.{});
    try std.testing.expectEqualStrings("Check the answer", result.reasoning.?);
    try std.testing.expectEqualStrings("Paris", result.text.?);

    const parts = result.message.?.parts;
    const names: std.StringHashMapUnmanaged([]const u8) = .empty;
    const signed = try encodePart(arena, parts[0], &names, null);
    const unsigned = try encodePart(arena, parts[1], &names, null);
    const answer = try encodePart(arena, parts[3], &names, null);
    try std.testing.expect(get(signed, "thought").bool);
    try std.testing.expectEqualStrings("opaque", text(get(signed, "thoughtSignature")));
    try std.testing.expect(get(unsigned, "thought") == .null);
    try std.testing.expectEqualStrings("answer-state", text(get(answer, "thoughtSignature")));
}

test "Gemini buffered response retains usable parts and records unmapped parts in wire order" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"responseId":"fixture-response","_lm15_unmapped":"stale","candidates":[{"content":{"parts":[
        \\ {"zFuture":true,"aFuture":42},
        \\ {"thought":true,"text":"Check","thoughtSignature":"replay-state"},
        \\ null,
        \\ {"text":"Pa"},
        \\ {"functionCall":{"id":"wire-id","name":"lookup","args":{"city":"Paris"}},"inlineData":{"mimeType":"image/png","data":""}},
        \\ {"executableCode":null},
        \\ {"inlineData":{"mimeType":"image/png","data":""}},
        \\ {},
        \\ {"functionCall":{"name":"confirm","args":[]}},
        \\ {"text":"ris"}
        \\]},"finishReason":"STOP"}]}
    );
    var result = try parseResponse(arena, "fixture", raw, &.{});
    try lm_mod.publishResponse(arena, &result, "fixture", .borrowed);
    const response = result.response.?;
    defer response.release();
    try std.testing.expectEqualStrings("Paris", result.text.?);
    try std.testing.expectEqualStrings("Check", result.reasoning.?);
    try std.testing.expectEqualStrings("fixture-response", response.fields.id.?);
    try std.testing.expectEqual(sdk.FinishReason.tool_call, response.fields.finish_reason);
    const parts = response.fields.message.parts;
    try std.testing.expectEqual(@as(usize, 5), parts.len);
    try std.testing.expect(parts[0] == .thinking and parts[1] == .text and parts[2] == .tool_call and
        parts[3] == .tool_call and parts[4] == .text);
    try std.testing.expectEqualStrings("Pa", parts[1].text.text);
    try std.testing.expectEqualStrings("ris", parts[4].text.text);
    try std.testing.expectEqualStrings("wire-id", parts[2].tool_call.id);
    try std.testing.expectEqualStrings("tool_call_3", parts[3].tool_call.id);
    try std.testing.expectEqual(@as(usize, 2), result.tool_calls.len);
    try std.testing.expectEqualStrings("wire-id", result.tool_calls[0].id);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\"}", result.tool_calls[0].arguments);
    try std.testing.expectEqualStrings("tool_call_3", result.tool_calls[1].id);
    try std.testing.expectEqualStrings("{}", result.tool_calls[1].arguments);

    const data = response.fields.provider_data.?.root.object;
    const records = data.get("_lm15_unmapped").?.array.items;
    const paths = [_][]const u8{ "candidates[0].content.parts[0]", "candidates[0].content.parts[2]", "candidates[0].content.parts[7]" };
    const kinds = [_][]const u8{ "aFuture+zFuture", "NoneType", "<empty>" };
    try std.testing.expectEqual(paths.len, records.len);
    for (records, paths, kinds) |record, path, kind| {
        try std.testing.expectEqualStrings(path, record.object.get("path").?.string);
        try std.testing.expectEqualStrings(kind, record.object.get("type").?.string);
    }
    try std.testing.expectEqualStrings("fixture-response", data.get("responseId").?.string);
    const candidate = data.get("candidates").?.array.items[0].object;
    const wire_parts = candidate.get("content").?.object.get("parts").?.array.items;
    try std.testing.expectEqual(@as(usize, 10), wire_parts.len);
    try std.testing.expectEqual(@as(i64, 42), wire_parts[0].object.get("aFuture").?.integer);
    try std.testing.expectEqualStrings("stale", text(get(raw, "_lm15_unmapped")));
}

test "Gemini buffered response adds unmapped diagnostics even with no mapped parts" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[
        \\ {"codeExecutionResult":{"output":"42"}},["future"],{"functionCall":"future"}
        \\]}}]}
    );
    const result = try parseResponse(arena, "fixture", raw, &.{});
    try std.testing.expectEqualStrings("", result.text.?);
    const parts = result.message.?.parts;
    try std.testing.expectEqual(@as(usize, 1), parts.len);
    try std.testing.expect(parts[0] == .text);
    try std.testing.expectEqualStrings("", parts[0].text.text);

    const records = result.provider_data.?.root.object.get("_lm15_unmapped").?.array.items;
    const paths = [_][]const u8{ "candidates[0].content.parts[1]", "candidates[0].content.parts[2]" };
    const kinds = [_][]const u8{ "list", "functionCall" };
    try std.testing.expectEqual(paths.len, records.len);
    for (records, paths, kinds) |record, path, kind| {
        try std.testing.expectEqualStrings(path, record.object.get("path").?.string);
        try std.testing.expectEqualStrings(kind, record.object.get("type").?.string);
    }
}

test "Gemini buffered response still rejects an unnamed function call after an unmapped part" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[{"future":true},{"text":"Paris"},{"functionCall":{"args":{}}}]}}]}
    );
    try std.testing.expectError(error.LMFailure, parseResponse(arena, "fixture", raw, &.{}));
}

test "Gemini ignores empty media before assigning a synthetic tool ID" {
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"candidates":[{"content":{"parts":[
        \\ {"inlineData":{"mimeType":"image/png","data":""}},
        \\ {"fileData":{"mimeType":"audio/wav","fileUri":"gs://bucket/audio"}},
        \\ {"functionCall":{"name":"lookup","args":{}}}
        \\]}}]}
    );
    const result = try parseResponse(arena, "fixture", raw, &.{});
    const parts = result.message.?.parts;
    try std.testing.expectEqualStrings("gs://bucket/audio", parts[0].audio.url.?);
    try std.testing.expectEqualStrings("tool_call_1", parts[1].tool_call.id);
}

const live_test = struct {
    const Fixture = struct {
        fn sendFrame(output: *std.Io.Writer, payload: []const u8) !void {
            return sendDataFrame(output, 0x81, payload);
        }

        fn sendDataFrame(output: *std.Io.Writer, header: u8, payload: []const u8) !void {
            if (payload.len < 126) {
                try output.writeAll(&.{ header, @intCast(payload.len) });
            } else {
                try output.writeAll(&.{ header, 126 });
                try output.writeInt(u16, @intCast(payload.len), .big);
            }
            try output.writeAll(payload);
            try output.flush();
        }

        fn readFrame(input: *std.Io.Reader, buffer: []u8) ![]const u8 {
            return readMaskedFrame(input, buffer, 0x81);
        }

        fn readMaskedFrame(input: *std.Io.Reader, buffer: []u8, header: u8) ![]const u8 {
            try std.testing.expectEqual(header, try input.takeByte());
            const flags = try input.takeByte();
            try std.testing.expect(flags & 0x80 != 0);
            const len = switch (flags & 0x7f) {
                126 => try input.takeInt(u16, .big),
                127 => return error.TestUnexpectedResult,
                else => |n| n,
            };
            if (len > buffer.len) return error.TestUnexpectedResult;
            var mask: [4]u8 = undefined;
            try input.readSliceAll(&mask);
            try input.readSliceAll(buffer[0..len]);
            for (buffer[0..len], 0..) |*byte, index| byte.* ^= mask[index % mask.len];
            return buffer[0..len];
        }

        fn serve(io: std.Io, listener: *std.Io.net.Server, expected: []const []const u8, stream: bool, expected_setup: ?[]const u8) !void {
            return serveScenario(io, listener, expected, stream, expected_setup, null, &.{}, null, null);
        }

        fn serveScenario(
            io: std.Io,
            listener: *std.Io.net.Server,
            expected: []const []const u8,
            stream: bool,
            expected_setup: ?[]const u8,
            refused: ?*std.atomic.Value(bool),
            before_reply: []const []const u8,
            terminal_reply: ?[]const u8,
            tool_result: ?[]const u8,
        ) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try upgrade(&input.interface, &output.interface);
            var frame: [2048]u8 = undefined;
            const setup = try readFrame(&input.interface, &frame);
            try std.testing.expect(std.mem.startsWith(u8, setup, "{\"setup\":"));
            if (expected_setup) |frame_json| {
                try std.testing.expectEqualStrings(frame_json, setup);
            } else if (stream) {
                try std.testing.expect(std.mem.find(
                    u8,
                    setup,
                    "\"realtimeInputConfig\":{\"automaticActivityDetection\":{\"disabled\":true}}",
                ) != null);
            }
            if (refused) |sent| {
                // An unrelated frame before the refusal is not an acknowledgement.
                try sendFrame(&output.interface, "{\"unrelated\":{}}");
                sent.store(true, .release);
                try sendFrame(&output.interface, "{\"error\":{\"status\":\"INVALID_ARGUMENT\",\"message\":\"bad setup\"}}");
                try std.testing.expectEqualStrings(
                    "\x03\xe8",
                    try readMaskedFrame(&input.interface, &frame, 0x88),
                );
                try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
                return;
            }
            try sendFrame(&output.interface, "{\"setupComplete\":{}}");
            for (expected) |payload| {
                try std.testing.expectEqualStrings(payload, try readFrame(&input.interface, &frame));
            }
            if (stream) {
                for (before_reply) |payload| try sendFrame(&output.interface, payload);
                if (tool_result) |payload|
                    try std.testing.expectEqualStrings(payload, try readFrame(&input.interface, &frame));
                try sendFrame(&output.interface, terminal_reply orelse
                    \\{"serverContent":{"modelTurn":{"parts":[{"text":"received"}]},"turnComplete":true}}
                );
                // Hold the socket open: the client must Close exactly once
                // and disconnect without waiting for a reply.
                try std.testing.expectEqualStrings("\x03\xe8", try readMaskedFrame(&input.interface, &frame, 0x88));
                try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
            }
        }

        fn upgrade(input: *std.Io.Reader, output: *std.Io.Writer) !void {
            try replyUpgrade(input, output, "", "HTTP/1.1", "Upgrade: websocket\r\n");
        }

        fn replyUpgrade(input: *std.Io.Reader, output: *std.Io.Writer, interim: []const u8, version: []const u8, upgrade_headers: []const u8) !void {
            var server = std.http.Server.init(input, output);
            const req = try server.receiveHead();
            try std.testing.expectEqualStrings(
                "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=local",
                req.head.target,
            );
            var key: ?[]const u8 = null;
            var headers = req.iterateHeaders();
            while (headers.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Sec-WebSocket-Key")) key = header.value;
            }
            var sha = std.crypto.hash.Sha1.init(.{});
            sha.update(key orelse return error.TestUnexpectedResult);
            sha.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
            var accept: [28]u8 = undefined;
            try output.writeAll(interim);
            try output.print(
                "{s} 101 Switching Protocols\r\n{s}Connection: Upgrade\r\nSec-WebSocket-Accept: {s}\r\n\r\n",
                .{ version, upgrade_headers, std.base64.standard.Encoder.encode(&accept, &sha.finalResult()) },
            );
            try output.flush();
        }

        fn serveInvalidUpgrade(io: std.Io, listener: *std.Io.net.Server, interim: []const u8, version: []const u8, headers: []const u8) !?u8 {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [2048]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try replyUpgrade(&input.interface, &output.interface, interim, version, headers);
            return input.interface.takeByte() catch |err| switch (err) {
                error.EndOfStream => null,
                else => return err,
            };
        }

        fn serveInterimUpgrade(io: std.Io, listener: *std.Io.net.Server, streaming: bool) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try replyUpgrade(
                &input.interface,
                &output.interface,
                "HTTP/1.1 103 Early Hints\r\nLink: </example>; rel=preload\r\n\r\n" ++
                    "HTTP/1.1 100 Continue\r\n\r\n",
                "HTTP/1.1",
                "Upgrade: websocket\r\n",
            );
            var frame: [2048]u8 = undefined;
            const setup = try readFrame(&input.interface, &frame);
            try std.testing.expect(std.mem.startsWith(u8, setup, "{\"setup\":"));
            try sendFrame(&output.interface, "{\"setupComplete\":{}}");
            try std.testing.expectEqualStrings(
                "{\"realtimeInput\":{\"text\":\"inspect\"}}",
                try readFrame(&input.interface, &frame),
            );
            try sendFrame(&output.interface, if (streaming)
                "{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"text\":\"received\"}]},\"turnComplete\":true}}"
            else
                "{\"serverContent\":{\"outputTranscription\":{\"text\":\"received\"}}}");
            try std.testing.expectEqualStrings("\x03\xe8", try readMaskedFrame(&input.interface, &frame, 0x88));
            try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
        }

        fn serveSinkStop(io: std.Io, listener: *std.Io.net.Server, send_delta: bool) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try upgrade(&input.interface, &output.interface);
            var frame: [2048]u8 = undefined;
            try std.testing.expectEqualStrings(
                \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
            , try readFrame(&input.interface, &frame));
            try sendFrame(&output.interface, "{\"setupComplete\":{}}");
            try std.testing.expectEqualStrings(
                "{\"realtimeInput\":{\"text\":\"inspect\"}}",
                try readFrame(&input.interface, &frame),
            );
            if (send_delta)
                try sendFrame(&output.interface, "{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"text\":\"partial\"}]}}}");
            // Do not send turnComplete or a Close reply: neither should be needed.
            try std.testing.expectEqualStrings("\x03\xe8", try readMaskedFrame(&input.interface, &frame, 0x88));
            try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
        }

        fn serveFailedSetup(io: std.Io, listener: *std.Io.net.Server) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [2048]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var timed: @import("timed_io.zig") = .{
                .inner = io,
                .timeout_ns = 5 * std.time.ns_per_s,
                .connect_timeout_ns = null,
            };
            var input = socket.reader(timed.io(), &read_buffer);
            var output = socket.writer(timed.io(), &write_buffer);
            try upgrade(&input.interface, &output.interface);
            // No setup payload and no Close reply from the server.
            var frame: [125]u8 = undefined;
            try std.testing.expectEqualStrings("\x03\xe8", try readMaskedFrame(&input.interface, &frame, 0x88));
            try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
        }

        fn serveFailedTurn(io: std.Io, listener: *std.Io.net.Server) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try upgrade(&input.interface, &output.interface);
            var frame: [2048]u8 = undefined;
            try std.testing.expectEqualStrings(
                \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
            , try readFrame(&input.interface, &frame));
            try sendFrame(&output.interface, "{\"setupComplete\":{}}");
            // No partial turn and no Close reply: closing must not await the server.
            try std.testing.expectEqualStrings("\x03\xe8", try readMaskedFrame(&input.interface, &frame, 0x88));
            try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
        }

        fn serveTerminalFrame(io: std.Io, listener: *std.Io.net.Server, peer_frame: []const u8, reply: ?[]const u8, stall: bool, turn_complete: bool) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try upgrade(&input.interface, &output.interface);
            var frame: [2048]u8 = undefined;
            const setup = try readFrame(&input.interface, &frame);
            try std.testing.expectEqualStrings(
                \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
            , setup);
            try sendFrame(&output.interface, "{\"setupComplete\":{}}");
            try std.testing.expectEqualStrings(
                "{\"realtimeInput\":{\"text\":\"inspect\"}}",
                try readFrame(&input.interface, &frame),
            );
            try sendFrame(&output.interface, if (turn_complete)
                "{\"serverContent\":{\"outputTranscription\":{\"text\":\"partial\"},\"turnComplete\":true}}"
            else
                "{\"serverContent\":{\"outputTranscription\":{\"text\":\"partial\"}}}");
            try output.interface.writeAll(peer_frame);
            try output.interface.flush();
            if (reply) |payload| {
                try std.testing.expectEqualStrings(payload, try readMaskedFrame(&input.interface, &frame, 0x88));
            } else if (!stall) {
                // Abrupt EOF: keep the read half open to detect writes after EOF.
                try socket.shutdown(io, .send);
            }
            // Keep TCP open so a second Close cannot pass unnoticed, but
            // bound the wait so a missing disconnect fails rather than hangs.
            var timed: @import("timed_io.zig") = .{
                .inner = io,
                .timeout_ns = 5 * std.time.ns_per_s,
                .connect_timeout_ns = null,
            };
            input.io = timed.io();
            try std.testing.expectError(error.EndOfStream, input.interface.takeByte());
        }

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const found: *bool = @ptrCast(@alignCast(context));
            if (event == .delta and event.delta.data == .text) {
                found.* = std.mem.eql(u8, event.delta.data.text.text, "received");
            }
            return .proceed;
        }
    };
};

test "Gemini Live disconnects on terminal WebSocket frames" {
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    const base = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()});
    defer std.testing.allocator.free(base);
    const gemini: Gemini = .{
        .client = &client,
        .base_url = base,
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .timeout_ns = 5 * std.time.ns_per_s,
        .max_response_bytes = 96,
    };
    for ([_]struct { frame: []const u8, reply: ?[]const u8, err: anyerror, stall: bool = false }{
        // The peer sends only the oversized frame's header. The client must
        // close without waiting for the payload, including on a continuation.
        .{ .frame = "\x81\x61", .reply = "\x03\xf1", .err = error.MessageTooLarge },
        .{ .frame = "\x01\x30" ++ @as([48]u8, @splat(' ')) ++ "\x80\x31", .reply = "\x03\xf1", .err = error.MessageTooLarge },
        .{ .frame = "\x81\x01\xff", .reply = "\x03\xef", .err = error.InvalidFrame },
        .{ .frame = "\xc1\x00", .reply = "\x03\xea", .err = error.InvalidFrame },
        .{ .frame = "\x80\x00", .reply = "\x03\xea", .err = error.InvalidFrame },
        .{ .frame = "\x88\x03\x03\xe9\xff", .reply = "\x03\xef", .err = error.InvalidFrame },
        .{ .frame = "\x88\x02\x03\xed", .reply = "\x03\xea", .err = error.InvalidFrame },
        .{ .frame = "\x88\x01\x03", .reply = "\x03\xea", .err = error.InvalidFrame },
        .{ .frame = "\x88\x0c\x03\xe9going away", .reply = "\x03\xe9going away", .err = error.PeerClosed },
        .{ .frame = "\x88\x00", .reply = "", .err = error.PeerClosed },
        .{ .frame = "", .reply = null, .err = error.EndOfStream },
        .{ .frame = "\x81\x05ab", .reply = null, .err = error.EndOfStream },
        // Withhold the remaining payload without shutting down the write half.
        .{ .frame = "\x81\x05ab", .reply = null, .err = error.ReadFailed, .stall = true },
    }) |close_case| {
        var serving = try io.concurrent(live_test.Fixture.serveTerminalFrame, .{ io, &listener, close_case.frame, close_case.reply, close_case.stall, false });
        defer serving.cancel(io) catch {};
        var config = gemini;
        if (close_case.stall) config.timeout_ns = std.time.ns_per_s;
        var session: Gemini.LiveSession = undefined;
        try config.live(&session, .{});
        defer session.close();
        const invalid_turn: Gemini.LiveSession.Input = .{ .turn = .{
            .parts = &.{.{ .image = .{ .media_type = "image/png" } }},
        } };
        try std.testing.expectError(error.LMFailure, session.send(invalid_turn));
        try session.send(.{ .text = "inspect" });
        try std.testing.expectEqualDeep(Gemini.LiveSession.Event{ .text = "partial" }, try session.recv());
        try std.testing.expectError(close_case.err, session.recv());
        try std.testing.expectError(error.PeerClosed, session.send(.{ .text = "do not send" }));
        try std.testing.expectError(error.PeerClosed, session.send(invalid_turn));
        try std.testing.expectError(error.PeerClosed, session.recv());
        // Observe TCP EOF while the application still owns the session.
        try serving.await(io);
    }
}

test "Gemini Live disconnects after a WebSocket write failure" {
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    const base = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()});
    defer std.testing.allocator.free(base);
    const gemini: Gemini = .{
        .client = &client,
        .base_url = base,
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    var serving = try io.concurrent(live_test.Fixture.serveTerminalFrame, .{ io, &listener, "", null, true, true });
    defer serving.cancel(io) catch {};
    var session: Gemini.LiveSession = undefined;
    try gemini.live(&session, .{});
    defer session.close();
    try session.send(.{ .text = "inspect" });
    try std.testing.expectEqualDeep(Gemini.LiveSession.Event{ .text = "partial" }, try session.recv());

    // Expire only the failing write; keep the peer open and let any
    // subsequent I/O through so the test detects use of the abandoned writer.
    session.timed.timeout_ns = 0;
    try std.testing.expectError(error.WriteFailed, session.send(.{ .text = "deadline" }));
    session.timed.timeout_ns = gemini.timeout_ns;
    try std.testing.expectError(error.PeerClosed, session.send(.{ .text = "do not send" }));
    try std.testing.expectError(error.PeerClosed, session.recv()); // Even with turn_end queued.
    try serving.await(io); // No application or Close frame before TCP EOF.
}

test "Gemini Live closes when local turn preparation fails" {
    const io = std.testing.io;
    var directory = std.testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(io, .{ .sub_path = "image.png", .data = "image" });
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const path = try directory.dir.realPathFileAlloc(io, "image.png", arena);
    try directory.dir.deleteFile(io, "image.png");

    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .image = .{ .media_type = "image/png", .path = path } }} }},
    });
    defer request.release();

    const Counter = struct {
        fn event(context: *anyopaque, _: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const events_seen: *usize = @ptrCast(@alignCast(context));
            events_seen.* += 1;
            return .proceed;
        }
    };
    var serving = try io.concurrent(live_test.Fixture.serveFailedTurn, .{ io, &listener });
    defer serving.cancel(io) catch {};
    var events: usize = 0;
    const result = gemini.streamRequest(arena, request, .{ .context = &events, .on_event = Counter.event });
    try serving.await(io);
    try std.testing.expectError(error.LMFailure, result);
    try std.testing.expectEqual(0, events);
}

test "Gemini Live sends Unicode strings in valid WebSocket text frames" {
    const Fixture = live_test.Fixture;
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    const model = "gemini-live-native-audio";
    var gemini: Gemini = .{
        .client = &client,
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}", .{listener.socket.address.getPort()}),
        .api_key = "local",
        .model = model,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    // DSPy 3.5.0 emits JSON escapes in both the setup and the turn.
    // Test ordinary Unicode first, then Python's WTF-8 lone-surrogate form.
    inline for (.{ false, true }) |surrogate| {
        const text_input = "雪 😀" ++ (if (surrogate) " \xed\xa0\x80" else "");
        const escaped = "\\u96ea \\ud83d\\ude00" ++ (if (surrogate) " \\ud800" else "");
        const setup = "{\"setup\":{\"model\":\"models/" ++ model ++
            "\",\"systemInstruction\":{\"parts\":[{\"text\":\"" ++ escaped ++
            "\"}]},\"generationConfig\":{\"responseModalities\":[\"AUDIO\"]},\"outputAudioTranscription\":{}}}";
        const turn = "{\"realtimeInput\":{\"text\":\"" ++ escaped ++ "\"}}";
        {
            const request = try sdk.Request.fromFields(arena, &.{
                .model = model,
                .system = .{ .text = text_input },
                .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = text_input } }} }},
            });
            defer request.release();
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, &.{turn}, true, setup });
            defer serving.cancel(io) catch {};
            var found = false;
            const outcome = gemini.streamRequest(arena, request, .{ .context = &found, .on_event = Fixture.onEvent });
            try serving.await(io);
            try std.testing.expectEqual(.exhausted, try outcome);
            try std.testing.expect(found);
        }
        {
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, &.{turn}, true, setup });
            defer serving.cancel(io) catch {};
            var session: Gemini.LiveSession = undefined;
            const result = gemini.live(&session, .{ .system = .{ .text = text_input } });
            if (result) |_| {
                defer session.close();
                try session.send(.{ .text = text_input });
                try std.testing.expectEqualDeep(@as(Gemini.LiveSession.Event, .{ .text = "received" }), try session.recv());
                try std.testing.expectEqualDeep(@as(Gemini.LiveSession.Event, .{ .turn_end = null }), try session.recv());
            } else |_| {}
            try serving.await(io);
            try result;
        }
    }
}

test "Gemini Live sends local media over the WebSocket" {
    const Fixture = live_test.Fixture;
    const io = std.testing.io;
    var directory = std.testing.tmpDir(.{});
    defer directory.cleanup();
    const wav = "RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00" ++
        "data\x02\x00\x00\x00\x01\x02";
    try directory.dir.writeFile(io, .{ .sub_path = "audio.wav", .data = wav });
    try directory.dir.writeFile(io, .{ .sub_path = "image.png", .data = "image" });
    try directory.dir.writeFile(io, .{ .sub_path = "video.mp4", .data = "video" });
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const image: sdk.Part = .{ .image = .{
        .media_type = "image/png",
        .path = try directory.dir.realPathFileAlloc(io, "image.png", arena),
    } };
    const parts: []const sdk.Part = &.{
        image,
        .{ .text = .{ .text = "inspect" } },
        .{ .audio = .{ .media_type = "audio/wav", .path = try directory.dir.realPathFileAlloc(io, "audio.wav", arena) } },
        .{ .video = .{ .media_type = "video/mp4", .path = try directory.dir.realPathFileAlloc(io, "video.mp4", arena) } },
    };
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    const base = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()});
    var gemini: Gemini = .{
        .client = &client,
        .base_url = base,
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    {
        const data = try @import("json_repair.zig").parseJson(arena, "{\"limit\":3}");
        const developer = try sdk.Request.fromFields(arena, &.{
            .model = "fixture-live",
            .config = .{ .extensions = try sdk.PayloadValue.fromBorrowedValue(arena, try @import("json_repair.zig").parseJson(arena,
                \\{"generationConfig":{"responseModalities":["TEXT"]}}
            )) },
            .messages = &.{
                .{ .role = .developer, .parts = &.{
                    .{ .text = .{ .text = "Use this policy" } },
                    .{ .data = .{ .value = try sdk.JsonValue.fromBorrowedValue(arena, data) } },
                } },
                .{ .role = .user, .parts = &.{.{ .text = .{ .text = "Proceed" } }} },
            },
        });
        defer developer.release();
        // Setup and transcript captured from the locked DSPy GeminiLM.stream.
        const setup = "{\"setup\":{\"model\":\"models/fixture-live\",\"generationConfig\":{\"responseModalities\":[\"TEXT\"]}}}";
        const turn =
            \\{"clientContent":{"turns":[{"role":"user","parts":[{"text":"[developer]\nUse this policy\n{\"limit\":3}"}]},{"role":"user","parts":[{"text":"Proceed"}]}],"turnComplete":true}}
        ;
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, &.{turn}, true, setup });
        defer serving.cancel(io) catch {};
        var found = false;
        const outcome = gemini.streamRequest(arena, developer, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        });
        try serving.await(io);
        try std.testing.expectEqual(.exhausted, try outcome);
        try std.testing.expect(found);
    }
    const request = try sdk.Request.fromFields(arena, &.{
        .model = "gemini:gemini-live-native-audio",
        .messages = &.{.{ .role = .user, .parts = parts }},
    });
    defer request.release();
    // Compact JSON from locked DSPy 3.5.0 with the same file-backed parts.
    const expected: []const []const u8 = &.{
        "{\"realtimeInput\":{\"activityStart\":{}}}",
        "{\"clientContent\":{\"turns\":[{\"role\":\"user\",\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"aW1hZ2U=\"}}]}],\"turnComplete\":false}}",
        "{\"realtimeInput\":{\"text\":\"inspect\"}}",
        "{\"realtimeInput\":{\"audio\":{\"mimeType\":\"audio/pcm;rate=22050\",\"data\":\"AQI=\"}}}",
        "{\"realtimeInput\":{\"video\":{\"mimeType\":\"video/mp4\",\"data\":\"dmlkZW8=\"}}}",
        "{\"realtimeInput\":{\"activityEnd\":{}}}",
    };
    {
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, expected, true, null });
        defer serving.cancel(io) catch {};
        var found = false;
        try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        }));
        try serving.await(io);
        try std.testing.expect(found);
    }
    {
        const Transcript = struct {
            next: usize = 0,
            matches: bool = true,

            fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
                const self: *@This() = @ptrCast(@alignCast(context));
                if (event != .delta) return .proceed;
                const texts = [_][]const u8{ "other", "spoken ", "received" };
                const indexes = [_][]const u8{ "1", "0", "0" };
                const delta = event.delta;
                self.matches = self.matches and self.next < texts.len and
                    delta.data == .text and
                    std.mem.eql(u8, delta.data.text.text, texts[self.next]) and
                    std.mem.eql(u8, delta.part_index orelse "", indexes[self.next]);
                self.next += 1;
                return .proceed;
            }
        };
        const reply: []const []const u8 = &.{
            \\{"serverContent":{"modelTurn":{"parts":[{},{"text":"other"}]},"outputTranscription":{"text":"spoken "}}}
        };
        var serving = try io.concurrent(Fixture.serveScenario, .{ io, &listener, expected, true, null, null, reply, null, null });
        defer serving.cancel(io) catch {};
        var capture: Transcript = .{};
        try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
            .context = &capture,
            .on_event = Transcript.onEvent,
        }));
        try serving.await(io);
        try std.testing.expectEqual(@as(usize, 3), capture.next);
        try std.testing.expect(capture.matches);
    }
    {
        const Mixed = struct {
            next: usize = 0,
            matches: bool = true,

            fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
                const self: *@This() = @ptrCast(@alignCast(context));
                if (event != .delta) return .proceed;
                const expected_deltas = [_]struct { index: []const u8, payload: []const u8, audio: bool }{
                    .{ .index = "0", .payload = "AQI=", .audio = true },
                    .{ .index = "1", .payload = "other", .audio = false },
                    .{ .index = "0", .payload = "spoken ", .audio = false },
                    .{ .index = "0", .payload = "12345678", .audio = true },
                    .{ .index = "0", .payload = "received", .audio = false },
                };
                defer self.next += 1;
                if (self.next >= expected_deltas.len) {
                    self.matches = false;
                    return .proceed;
                }
                const want = expected_deltas[self.next];
                const delta = event.delta;
                self.matches = self.matches and std.mem.eql(u8, delta.part_index orelse "", want.index) and
                    if (want.audio)
                        delta.data == .audio and
                            std.mem.eql(u8, delta.data.audio.media_type orelse "", "audio/pcm") and
                            std.mem.eql(u8, delta.data.audio.data orelse "", want.payload)
                    else
                        delta.data == .text and std.mem.eql(u8, delta.data.text.text, want.payload);
                return .proceed;
            }
        };
        // The last nonempty usage snapshot replaces, rather than adds to or
        // merges with, the earlier snapshot. The terminal frame has no usage.
        const reply: []const []const u8 = &.{
            \\{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm","data":"AQI="}},{"text":"other"}]}}}
            ,
            \\{"serverContent":{"outputTranscription":{"text":"spoken "},"usageMetadata":{"promptTokenCount":3,"responseTokenCount":1,"promptTokensDetails":[{"modality":"AUDIO","tokenCount":2}]}}}
            ,
            \\{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm","data":12345678}}]},"usageMetadata":{"responseTokenCount":1}},"usageMetadata":{"promptTokenCount":5,"cachedContentTokenCount":2,"responseTokenCount":4,"candidatesTokenCount":1,"thoughtsTokenCount":2,"totalTokenCount":11,"responseTokensDetails":[{"modality":"AUDIO","tokenCount":3}]}}
            ,
            \\{"usageMetadata":{}}
            ,
        };
        const setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
        ;
        const input: []const []const u8 = &.{"{\"realtimeInput\":{\"text\":\"inspect\"}}"};
        var serving = try io.concurrent(Fixture.serveScenario, .{ io, &listener, input, true, setup, null, reply, null, null });
        defer serving.cancel(io) catch {};
        var capture: Mixed = .{};
        const completions = try gemini.lm().completeResponses(arena, .{
            .messages = &.{.{ .role = .user, .content = .{ .text = "inspect" } }},
            .stream_sink = .{ .context = &capture, .on_event = Mixed.onEvent },
        });
        const response = completions[0].response orelse return error.TestUnexpectedResult;
        defer response.release();
        try serving.await(io);
        try std.testing.expectEqual(@as(usize, 5), capture.next);
        try std.testing.expect(capture.matches);
        try std.testing.expectEqual(sdk.FinishReason.stop, response.fields.finish_reason);
        const assembled = response.fields.message.parts;
        try std.testing.expectEqual(@as(usize, 3), assembled.len);
        try std.testing.expect(assembled[0] == .text and assembled[1] == .audio and assembled[2] == .text);
        try std.testing.expectEqualStrings("spoken received", assembled[0].text.text);
        try std.testing.expectEqualStrings("audio/wav", assembled[1].audio.media_type);
        try std.testing.expectEqualStrings("UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAABAtdt+Oeu/A==", assembled[1].audio.data orelse "");
        try std.testing.expectEqualStrings("other", assembled[2].text.text);
        try std.testing.expectEqualDeep(sdk.Usage{
            .input_tokens = .{ .integer = 5 },
            .output_tokens = .{ .integer = 6 },
            .total_tokens = .{ .integer = 11 },
            .cache_read_tokens = .{ .integer = 2 },
            .reasoning_tokens = .{ .integer = 2 },
            .output_audio_tokens = .{ .integer = 3 },
        }, response.fields.usage);
    }
    // A provider refusal after partial output is not a completed turn.
    const failures = [_]struct { terminal: []const u8, expected: StreamError }{
        .{
            .terminal = "{\"error\":{\"status\":\"INVALID_ARGUMENT\",\"message\":\"Bad request\"}}",
            .expected = .{
                .code = .invalid_request,
                .provider_code = "INVALID_ARGUMENT",
                .message = "Bad request",
            },
        },
        .{
            .terminal = "{\"error\":{\"status\":\"INVALID_ARGUMENT\",\"message\":\"Maximum context length exceeded\"}}",
            .expected = .{
                .code = .context_length,
                .provider_code = "INVALID_ARGUMENT",
                .message = "Maximum context length exceeded",
            },
        },
        .{
            // DSPy displays the provider code when the message is absent,
            // but does not use it as evidence of context overflow.
            .terminal = "{\"error\":{\"code\":\"Token limit exceeded\"}}",
            .expected = .{
                .code = .provider,
                .provider_code = "Token limit exceeded",
                .message = "Token limit exceeded",
            },
        },
    };
    for (failures) |failure| {
        const Capture = struct {
            expected: StreamError,
            step: usize = 0,
            matches: bool = true,

            fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
                const self: *@This() = @ptrCast(@alignCast(context));
                self.matches = self.matches and switch (self.step) {
                    0 => event == .start,
                    1 => event == .delta and event.delta.data == .text and
                        std.mem.eql(u8, event.delta.data.text.text, "partial"),
                    2 => event == .provider_error and event.provider_error.code == self.expected.code and
                        std.mem.eql(u8, event.provider_error.provider_code, self.expected.provider_code) and
                        std.mem.eql(u8, event.provider_error.message, self.expected.message),
                    else => false,
                };
                self.step += 1;
                return .proceed;
            }
        };
        const setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
        ;
        const partial =
            \\{"serverContent":{"modelTurn":{"parts":[{"text":"partial"}]}}}
        ;
        for ([_]bool{ false, true }) |collect| {
            var serving = try io.concurrent(Fixture.serveScenario, .{
                io, &listener, &.{"{\"realtimeInput\":{\"text\":\"inspect\"}}"}, true, setup, null, &.{partial}, failure.terminal, null,
            });
            defer serving.cancel(io) catch {};
            var capture: Capture = .{ .expected = failure.expected };
            const sink: Gemini.StreamSink = .{ .context = &capture, .on_event = Capture.onEvent };
            if (collect) {
                const result = gemini.lm().completeResponses(arena, .{
                    .messages = &.{.{ .role = .user, .content = .{ .text = "inspect" } }},
                    .stream_sink = sink,
                });
                const expected_error = if (failure.expected.code == .context_length)
                    error.ContextWindowExceeded
                else
                    error.LMFailure;
                try std.testing.expectError(expected_error, result);
                try serving.await(io);
            } else {
                const live_request = try sdk.Request.fromFields(arena, &.{
                    .model = gemini.model,
                    .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "inspect" } }} }},
                });
                defer live_request.release();
                try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, live_request, sink));
                try serving.await(io);
            }
            try std.testing.expectEqual(@as(usize, 3), capture.step);
            try std.testing.expect(capture.matches);
        }
    }
    // Decode the entire terminal frame before publishing its deltas. A
    // malformed usage snapshot must not expose its tool call to the consumer.
    for ([_]bool{ true, false }) |valid_usage| {
        const Capture = struct {
            tools: usize = 0,
            end_usage: ?sdk.Usage = null,

            fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
                const self: *@This() = @ptrCast(@alignCast(context));
                switch (event) {
                    .delta => |delta| {
                        if (delta.data == .tool_call) self.tools += 1;
                    },
                    .end => |end| self.end_usage = end.usage,
                    else => {},
                }
                return .proceed;
            }
        };
        const setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
        ;
        const initial = "{\"usageMetadata\":{\"promptTokenCount\":3,\"responseTokenCount\":1}}";
        const terminal = "{\"toolCall\":{\"functionCalls\":[{\"id\":\"call\",\"name\":\"lookup\",\"args\":{}}]}," ++
            "\"serverContent\":{\"turnComplete\":true,\"usageMetadata\":{\"promptTokenCount\":9}},\"usageMetadata\":{\"promptTokenCount\":";
        const reply = if (valid_usage)
            terminal ++ "5,\"responseTokenCount\":2}}"
        else
            terminal ++ "\"invalid\"}}";
        var serving = try io.concurrent(Fixture.serveScenario, .{
            io, &listener, &.{"{\"realtimeInput\":{\"text\":\"inspect\"}}"}, true, setup, null, &.{initial}, reply, null,
        });
        defer serving.cancel(io) catch {};
        var capture: Capture = .{};
        const result = gemini.lm().completeResponses(arena, .{
            .messages = &.{.{ .role = .user, .content = .{ .text = "inspect" } }},
            .stream_sink = .{ .context = &capture, .on_event = Capture.onEvent },
        });
        if (valid_usage) {
            try serving.await(io);
            const response = (try result)[0].response orelse return error.TestUnexpectedResult;
            defer response.release();
            const usage: sdk.Usage = .{
                .input_tokens = .{ .integer = 5 },
                .output_tokens = .{ .integer = 2 },
                .total_tokens = .{ .integer = 7 },
                .cache_read_tokens = .{ .integer = 0 },
                .reasoning_tokens = .{ .integer = 0 },
            };
            try std.testing.expectEqualDeep(usage, response.fields.usage);
            try std.testing.expectEqualDeep(@as(?sdk.Usage, usage), capture.end_usage);
            try std.testing.expectEqual(sdk.FinishReason.tool_call, response.fields.finish_reason);
            try std.testing.expectEqual(@as(usize, 1), capture.tools);
        } else {
            try std.testing.expectError(error.LMFailure, result);
            try std.testing.expectEqual(@as(usize, 0), capture.tools);
            try std.testing.expectEqual(null, capture.end_usage);
            try serving.await(io);
        }
    }
    {
        const Capture = struct {
            started: bool = false,
            answer: bool = false,
            tool: bool = false,
            finish: ?sdk.FinishReason = null,

            fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
                const self: *@This() = @ptrCast(@alignCast(context));
                switch (event) {
                    .start => self.started = true,
                    .delta => switch (event.delta.data) {
                        .text => |answer| self.answer = std.mem.eql(u8, "ok", answer.text),
                        .tool_call => |call| self.tool = std.mem.eql(u8, "call-42", call.id orelse "") and
                            std.mem.eql(u8, "lookup", call.name orelse "") and
                            std.mem.eql(u8, "{\"key\": 42}", call.input),
                        else => {},
                    },
                    .end => self.finish = event.end.finish_reason,
                    else => {},
                }
                return .proceed;
            }
        };
        const setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
        ;
        const turn = "{\"realtimeInput\":{\"text\":\"inspect\"}}";
        const parts_request = try sdk.Request.fromFields(arena, &.{
            .model = gemini.model,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "inspect" } }} }},
        });
        defer parts_request.release();
        const cases = [_]struct {
            json: []const u8,
            malformed: bool = false,
            model_parts_only: bool = false,
            answer: bool = false,
            collect: bool = false,
        }{
            .{ .json = "true", .malformed = true },
            .{ .json = "1", .malformed = true },
            .{ .json = "0" },
            .{ .json = "false" },
            .{ .json = "{\"text\":\"lost\"}", .malformed = true, .model_parts_only = true, .collect = true },
            .{ .json = "{\"metadata\":\"ignored\"}", .model_parts_only = true },
            .{ .json = "null" },
            .{ .json = "[{\"text\":\"ok\"},null]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},false]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},0]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},1.25]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},\"some text\"]", .malformed = true, .model_parts_only = true, .collect = true },
            .{ .json = "[{\"text\":\"ok\"},[\"text\"]]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},[\"functionCall\"]]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},\"inlineData\"]", .malformed = true, .model_parts_only = true },
            .{ .json = "[{\"text\":\"ok\"},\"metadata\",[\"metadata\"],[\"some text\"]]", .model_parts_only = true, .answer = true },
            .{ .json = "" }, // A valid text part or tool call, depending on the envelope.
        };
        for ([_]bool{ false, true }) |tool_envelope| for (cases) |case| {
            if (tool_envelope and case.model_parts_only) continue;
            const has_payload = case.json.len == 0;
            const has_text = (has_payload and !tool_envelope) or case.answer;
            const has_tool = has_payload and tool_envelope;
            const reply = if (tool_envelope)
                try std.fmt.allocPrint(arena, "{{\"toolCall\":{{\"functionCalls\":{s}}},\"serverContent\":{{\"turnComplete\":true}}}}", .{
                    if (has_tool) "[{\"id\":\"call-42\",\"name\":\"lookup\",\"args\":{\"key\":42}}]" else case.json,
                })
            else
                try std.fmt.allocPrint(arena, "{{\"serverContent\":{{\"modelTurn\":{{\"parts\":{s}}},\"turnComplete\":true}}}}", .{
                    if (has_payload) "[{\"text\":\"ok\"},{}]" else case.json,
                });
            {
                var serving = try io.concurrent(Fixture.serveScenario, .{
                    io, &listener, &.{turn}, true, setup, null, &.{}, reply, null,
                });
                defer serving.cancel(io) catch {};
                var capture: Capture = .{};
                const streamed = gemini.streamRequest(arena, parts_request, .{
                    .context = &capture,
                    .on_event = Capture.onEvent,
                });
                if (case.malformed)
                    try std.testing.expectError(error.LMFailure, streamed)
                else
                    _ = try streamed;
                try serving.await(io);
                try std.testing.expectEqualDeep(Capture{
                    .started = true,
                    .answer = has_text,
                    .tool = has_tool,
                    .finish = if (case.malformed) null else if (has_tool) .tool_call else .stop,
                }, capture);
            }
            if (case.collect) {
                var serving = try io.concurrent(Fixture.serveScenario, .{
                    io, &listener, &.{turn}, true, setup, null, &.{}, reply, null,
                });
                defer serving.cancel(io) catch {};
                var capture: Capture = .{};
                try std.testing.expectError(error.LMFailure, gemini.lm().completeResponses(arena, .{
                    .messages = &.{.{ .role = .user, .content = .{ .text = "inspect" } }},
                    .stream_sink = .{ .context = &capture, .on_event = Capture.onEvent },
                }));
                try std.testing.expectEqualDeep(Capture{ .started = true }, capture);
                try serving.await(io);
            }
            {
                var serving = try io.concurrent(Fixture.serveScenario, .{
                    io, &listener, &.{turn}, true, setup, null, &.{}, reply, null,
                });
                defer serving.cancel(io) catch {};
                var session: Gemini.LiveSession = undefined;
                try gemini.live(&session, .{});
                defer session.close();
                try session.send(.{ .text = "inspect" });
                if (case.malformed) {
                    try std.testing.expectError(error.LMFailure, session.recv());
                } else {
                    const Event = Gemini.LiveSession.Event;
                    if (has_text)
                        try std.testing.expectEqualDeep(@as(Event, .{ .text = "ok" }), try session.recv());
                    if (has_tool) {
                        const event = try session.recv();
                        try std.testing.expect(event == .tool_call);
                        try std.testing.expectEqualStrings("call-42", event.tool_call.id);
                        try std.testing.expectEqualStrings("lookup", event.tool_call.name);
                        try std.testing.expectEqualDeep(Value{ .integer = 42 }, get(event.tool_call.input, "key"));
                    }
                    try std.testing.expectEqualDeep(@as(Event, .{ .turn_end = null }), try session.recv());
                }
                session.close();
                try serving.await(io);
            }
        };
    }
    try directory.dir.writeFile(io, .{ .sub_path = "image.png", .data = "IMAGE" });
    // Source inspects even URL-only media for setup but does not send it, or
    // earlier tool activity, as audio-native input. No input yields empty text.
    const audio_url: sdk.Part = .{ .audio = .{ .media_type = "audio/wav", .url = "https://example.invalid/audio" } };
    const video_url: sdk.Part = .{ .video = .{ .media_type = "video/mp4", .url = "https://example.invalid/video" } };
    const history: []const sdk.Message = &.{
        .{ .role = .assistant, .parts = &.{.{ .tool_call = .{
            .id = "call",
            .name = "lookup",
            .input = try sdk.JsonValue.fromBorrowedValue(arena, .{ .object = .empty }),
        } }} },
        .{ .role = .tool, .parts = &.{.{ .tool_result = .{
            .id = "call",
            .content = &.{.{ .text = .{ .text = "private result" } }},
        } }} },
        .{ .role = .user, .parts = &.{ audio_url, video_url, .{ .text = .{ .text = "continue" } } } },
    };
    const remote_only: []const sdk.Message = &.{.{ .role = .user, .parts = &.{ audio_url, video_url } }};
    for ([_][]const sdk.Message{ history, remote_only }, [_][]const u8{
        "{\"realtimeInput\":{\"text\":\"continue\"}}",
        "{\"realtimeInput\":{\"text\":\"\"}}",
    }) |messages, payload| {
        const filtered = try sdk.Request.fromFields(arena, &.{ .model = gemini.model, .messages = messages });
        defer filtered.release();
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, &.{payload}, true, null });
        defer serving.cancel(io) catch {};
        var found = false;
        try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, filtered, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        }));
        try serving.await(io);
        try std.testing.expect(found);
    }
    // DSPy strips the data URI and splits whitespace after trimming the
    // payload. This fixture mixes CRLF, Unicode space, and trailing tabs.
    {
        const inline_wav = try sdk.Request.fromFields(arena, &.{
            .model = gemini.model,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .audio = .{
                .media_type = "audio/wav",
                .data = "data:audio/wav;base64,UklGRiYA\r\n \u{2003}AABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQIAAAABAh==\t",
            } }} }},
        });
        defer inline_wav.release();
        const audio_frames: []const []const u8 = &.{ expected[0], expected[3], expected[5] };
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, audio_frames, true, null });
        defer serving.cancel(io) catch {};
        var found = false;
        try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, inline_wav, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        }));
        try serving.await(io);
        try std.testing.expect(found);
    }
    {
        // Explicit audio output disables activity detection but does not
        // switch a non-audio-native model to realtime input frames.
        const extensions = try @import("json_repair.zig").parseJson(arena,
            \\{"transport":"live","output":"audio","realtimeInputConfig":{"automaticActivityDetection":{"prefixPaddingMs":100}}}
        );
        const audio_request = try sdk.Request.fromFields(arena, &.{
            .model = "gemini-2.5-flash",
            .messages = &.{.{ .role = .user, .parts = &.{.{ .audio = .{
                .media_type = "audio/pcm",
                .data = "AQI=",
            } }} }},
            .config = .{ .extensions = try sdk.JsonValue.fromBorrowedValue(arena, extensions) },
        });
        defer audio_request.release();
        const setup =
            \\{"setup":{"model":"models/gemini-2.5-flash","realtimeInputConfig":{"automaticActivityDetection":{"prefixPaddingMs":100,"disabled":true}},"generationConfig":{"responseModalities":["AUDIO"]}}}
        ;
        const frames: []const []const u8 = &.{
            \\{"clientContent":{"turns":[{"role":"user","parts":[{"inlineData":{"mimeType":"audio/pcm","data":"AQI="}}]}],"turnComplete":true}}
        };
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, frames, true, setup });
        defer serving.cancel(io) catch {};
        var found = false;
        const outcome = gemini.streamRequest(arena, audio_request, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        });
        try serving.await(io);
        try std.testing.expectEqual(.exhausted, try outcome);
        try std.testing.expect(found);
    }
    // Locked DSPy 3.5.0 stream() defaults a missing modality to TEXT,
    // but preserves a present key (even null) unless audio/image is selected.
    const modality_cases = [_]struct { extensions: []const u8, setup: []const u8 }{
        .{
            .extensions = "{\"transport\":\"live\"}",
            .setup = "{\"setup\":{\"model\":\"models/gemini-2.5-flash\",\"generationConfig\":{\"responseModalities\":[\"TEXT\"]}}}",
        },
        .{
            .extensions = "{\"transport\":\"live\",\"generationConfig\":{\"temperature\":0.25}}",
            .setup = "{\"setup\":{\"model\":\"models/gemini-2.5-flash\",\"generationConfig\":{\"temperature\":0.25,\"responseModalities\":[\"TEXT\"]}}}",
        },
        .{
            .extensions = "{\"transport\":\"live\",\"generationConfig\":{\"responseModalities\":null}}",
            .setup = "{\"setup\":{\"model\":\"models/gemini-2.5-flash\",\"generationConfig\":{\"responseModalities\":null}}}",
        },
        .{
            .extensions = "{\"transport\":\"live\",\"output\":\"image\",\"generationConfig\":{\"responseModalities\":[\"TEXT\"],\"temperature\":0.25}}",
            .setup = "{\"setup\":{\"model\":\"models/gemini-2.5-flash\",\"generationConfig\":{\"responseModalities\":[\"IMAGE\"],\"temperature\":0.25}}}",
        },
    };
    for (modality_cases) |case| {
        const extensions = try @import("json_repair.zig").parseJson(arena, case.extensions);
        const prompt = try sdk.Request.fromFields(arena, &.{
            .model = "gemini-2.5-flash",
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "continue" } }} }},
            .config = .{ .extensions = try sdk.JsonValue.fromBorrowedValue(arena, extensions) },
        });
        defer prompt.release();
        const frames: []const []const u8 = &.{"{\"realtimeInput\":{\"text\":\"continue\"}}"};
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, frames, true, case.setup });
        defer serving.cancel(io) catch {};
        var found = false;
        const outcome = gemini.streamRequest(arena, prompt, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        });
        try serving.await(io);
        try std.testing.expectEqual(.exhausted, try outcome);
        try std.testing.expect(found);
    }
    // Locked DSPy 3.5.0 distinguishes an omitted system prompt from a
    // nonempty part sequence whose text projection is empty.
    const empty_system: sdk.SystemContent = .{ .parts = &.{.{ .text = .{ .text = "" } }} };
    const empty_setup =
        \\{"setup":{"model":"models/gemini-live-native-audio","systemInstruction":{"parts":[{"text":""}]},"generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
    ;
    const cases = [_]struct { system: ?sdk.SystemContent, setup: []const u8 }{
        .{ .system = empty_system, .setup = empty_setup },
        .{
            .system = null,
            .setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
            ,
        },
        .{
            .system = .{ .parts = &.{
                .{ .text = .{ .text = "first" } },
                .{ .text = .{ .text = "" } },
                .{ .text = .{ .text = "last" } },
            } },
            .setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","systemInstruction":{"parts":[{"text":"first\n\nlast"}]},"generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
            ,
        },
    };
    for (cases) |case| {
        const prompt = try sdk.Request.fromFields(arena, &.{
            .model = gemini.model,
            .system = case.system,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "continue" } }} }},
        });
        defer prompt.release();
        const frames: []const []const u8 = &.{"{\"realtimeInput\":{\"text\":\"continue\"}}"};
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, frames, true, case.setup });
        defer serving.cancel(io) catch {};
        var found = false;
        const outcome = gemini.streamRequest(arena, prompt, .{
            .context = &found,
            .on_event = Fixture.onEvent,
        });
        try serving.await(io);
        try std.testing.expectEqual(.exhausted, try outcome);
        try std.testing.expect(found);
    }
    // Both paths quote model names, but only stream() defaults to TEXT.
    const model_cases = [_]struct { model: []const u8, setup: []const u8 }{
        .{
            .model = "models/team/gemini flash:foo@bar?x#é%2F",
            .setup = "{\"setup\":{\"model\":\"models/team/gemini%20flash:foo@bar%3Fx%23%C3%A9%252F\"}}",
        },
        .{
            .model = "team/gemini-flash:foo@bar+β",
            .setup = "{\"setup\":{\"model\":\"models/team/gemini-flash:foo@bar%2B%CE%B2\"}}",
        },
        .{
            .model = "gemini:models/gemini-2.5-flash",
            .setup = "{\"setup\":{\"model\":\"models/gemini-2.5-flash\"}}",
        },
        .{
            .model = "models/gemini-2.5-flash",
            .setup = "{\"setup\":{\"model\":\"models/gemini-2.5-flash\"}}",
        },
    };
    for (model_cases) |case| {
        const frames: []const []const u8 = &.{"{\"realtimeInput\":{\"text\":\"continue\"}}"};
        const extensions = try @import("json_repair.zig").parseJson(arena, "{\"transport\":\"live\"}");
        const prompt = try sdk.Request.fromFields(arena, &.{
            .model = case.model,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "continue" } }} }},
            .config = .{ .extensions = try sdk.JsonValue.fromBorrowedValue(arena, extensions) },
        });
        defer prompt.release();
        {
            const stream_setup = try std.mem.concat(arena, u8, &.{
                case.setup[0 .. case.setup.len - 2],
                ",\"generationConfig\":{\"responseModalities\":[\"TEXT\"]}}}",
            });
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, frames, true, stream_setup });
            defer serving.cancel(io) catch {};
            var found = false;
            const outcome = gemini.streamRequest(arena, prompt, .{ .context = &found, .on_event = Fixture.onEvent });
            try serving.await(io);
            try std.testing.expectEqual(.exhausted, try outcome);
            try std.testing.expect(found);
        }
        {
            const session_frames: []const []const u8 = &.{
                "{\"clientContent\":{\"turns\":[{\"role\":\"user\",\"parts\":[{\"text\":\"continue\"}]}],\"turnComplete\":true}}",
            };
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, session_frames, false, case.setup });
            defer serving.cancel(io) catch {};
            var configured = gemini;
            configured.model = case.model;
            var session: Gemini.LiveSession = undefined;
            const outcome = configured.live(&session, .{});
            if (outcome) |_| {
                defer session.close();
                try session.send(.{ .text = "continue" });
                try serving.await(io);
            } else |err| {
                try serving.await(io);
                return err;
            }
        }
    }
    {
        const Sink = struct {
            stop_after: usize,
            events: usize = 0,
            valid: bool = true,

            fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
                const self: *@This() = @ptrCast(@alignCast(context));
                self.valid = self.valid and switch (self.events) {
                    0 => event == .start,
                    1 => event == .delta and event.delta.data == .text and
                        std.mem.eql(u8, event.delta.data.text.text, "partial"),
                    else => false,
                };
                self.events += 1;
                return if (self.events >= self.stop_after) .stop else .proceed;
            }
        };
        const prompt = try sdk.Request.fromFields(arena, &.{
            .model = gemini.model,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "inspect" } }} }},
        });
        defer prompt.release();
        for ([_]usize{ 1, 2 }) |stop_after| {
            var serving = try io.concurrent(Fixture.serveSinkStop, .{ io, &listener, stop_after == 2 });
            defer serving.cancel(io) catch {};
            var sink: Sink = .{ .stop_after = stop_after };
            const outcome = gemini.streamRequest(arena, prompt, .{ .context = &sink, .on_event = Sink.onEvent });
            try serving.await(io);
            try std.testing.expectEqual(.stopped, try outcome);
            try std.testing.expectEqual(stop_after, sink.events);
            try std.testing.expect(sink.valid);
        }
    }
    // A setup rejection must reach the client before it tries to open a turn's
    // local media, and neither public path may send a turn after the rejection.
    const missing_path = try std.fs.path.join(arena, &.{ std.fs.path.dirname(image.image.path.?).?, "missing.png" });
    const missing_media = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .image = .{
            .media_type = "image/png",
            .path = missing_path,
        } }} }},
    });
    defer missing_media.release();
    const rejected_setup =
        \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
    ;
    for ([_]bool{ true, false }) |streaming| {
        var refused: std.atomic.Value(bool) = .init(false);
        var serving = try io.concurrent(Fixture.serveScenario, .{ io, &listener, &.{}, streaming, rejected_setup, &refused, &.{}, null, null });
        defer serving.cancel(io) catch {};
        if (streaming) {
            var event_count: usize = 0;
            const Count = struct {
                fn event(context: *anyopaque, _: Gemini.StreamEvent) Gemini.StreamSink.Action {
                    const events_seen: *usize = @ptrCast(@alignCast(context));
                    events_seen.* += 1;
                    return .proceed;
                }
            };
            try std.testing.expectError(error.LMFailure, gemini.streamRequest(arena, missing_media, .{
                .context = &event_count,
                .on_event = Count.event,
            }));
            try std.testing.expectEqual(@as(usize, 0), event_count);
        } else {
            var session: Gemini.LiveSession = undefined;
            const result = gemini.live(&session, .{});
            if (result) |_| session.close() else |_| {}
            try std.testing.expectError(error.LMFailure, result);
        }
        // Before the fix, the missing file can fail before even connecting.
        try std.testing.expect(refused.load(.acquire));
        try serving.await(io);
    }
    const turn: []const []const u8 = &.{
        "{\"clientContent\":{\"turns\":[{\"role\":\"user\",\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"SU1BR0U=\"}}]}],\"turnComplete\":true}}",
    };
    {
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, turn, false, empty_setup });
        defer serving.cancel(io) catch {};
        var session: Gemini.LiveSession = undefined;
        try gemini.live(&session, .{ .system = empty_system });
        defer session.close();
        try session.send(.{ .turn = .{ .parts = &.{image} } });
        try serving.await(io);
        try directory.dir.deleteFile(io, "image.png");
        try std.testing.expectError(error.LMFailure, session.send(.{ .turn = .{ .parts = &.{image} } }));
    }
    {
        // DSPy publishes a canceled response's usage before interruption.
        // A frame that also completes the turn carries usage only on turn_end.
        const interrupted =
            \\{"usageMetadata":{"promptTokenCount":5,"responseTokenCount":2},"serverContent":{"interrupted":true,"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm","data":1234}}]},"outputTranscription":{"text":"partial"},"usageMetadata":{"promptTokenCount":99}}}
        ;
        const completed =
            \\{"serverContent":{"interrupted":true,"turnComplete":true,"usageMetadata":{"promptTokenCount":3,"responseTokenCount":1}}}
        ;
        const setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
        ;
        var serving = try io.concurrent(Fixture.serveScenario, .{
            io,    &listener, &.{"{\"clientContent\":{\"turnComplete\":true}}"}, true,
            setup, null,      &.{interrupted},                                   completed,
            null,
        });
        defer serving.cancel(io) catch {};
        var session: Gemini.LiveSession = undefined;
        try gemini.live(&session, .{});
        defer session.close();
        const Event = Gemini.LiveSession.Event;
        try session.send(.interrupt);
        try std.testing.expectEqualDeep(@as(Event, .{ .audio = .{ .data = "1234", .media_type = "audio/pcm" } }), try session.recv());
        try std.testing.expectEqualDeep(@as(Event, .{ .text = "partial" }), try session.recv());
        try std.testing.expectEqualDeep(@as(Event, .{ .usage = .{
            .input_tokens = .{ .integer = 5 },
            .output_tokens = .{ .integer = 2 },
            .total_tokens = .{ .integer = 7 },
            .cache_read_tokens = .{ .integer = 0 },
            .reasoning_tokens = .{ .integer = 0 },
        } }), try session.recv());
        try std.testing.expectEqual(Event.interrupted, try session.recv());
        try std.testing.expectEqual(Event.interrupted, try session.recv());
        try std.testing.expectEqualDeep(@as(Event, .{ .turn_end = .{
            .input_tokens = .{ .integer = 3 },
            .output_tokens = .{ .integer = 1 },
            .total_tokens = .{ .integer = 4 },
            .cache_read_tokens = .{ .integer = 0 },
            .reasoning_tokens = .{ .integer = 0 },
        } }), try session.recv());
        session.close();
        try serving.await(io);
    }
    {
        // Locked DSPy distinguishes omitted parameters from an explicit {}.
        const setup =
            \\{"setup":{"model":"models/gemini-live-native-audio","tools":[{"functionDeclarations":[{"name":"status","parameters":{"type":"object","properties":{}}},{"name":"verbatim","parameters":{}}]}],"generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
        ;
        const turn_text =
            \\{"realtimeInput":{"text":"find 42"}}
        ;
        const tool_call =
            \\{"toolCall":{"functionCalls":[{"id":"call-42","name":"status","args":{}}]},"serverContent":{},"usageMetadata":{"promptTokenCount":7,"responseTokenCount":2}}
        ;
        const tool_result =
            \\{"toolResponse":{"functionResponses":[{"id":"call-42","response":{"output":[{"text":"found"}]}}]}}
        ;
        const completed =
            \\{"serverContent":{"modelTurn":{"parts":[{"text":"done"}]},"turnComplete":true,"usageMetadata":{"promptTokenCount":11,"responseTokenCount":4}}}
        ;
        var serving = try io.concurrent(Fixture.serveScenario, .{
            io, &listener, &.{turn_text}, true, setup, null, &.{tool_call}, completed, tool_result,
        });
        defer serving.cancel(io) catch {};
        var session: Gemini.LiveSession = undefined;
        // The peer expects the valid setup on its very first connection.
        // A duplicate declaration must not consume that connection.
        const duplicate = gemini.live(&session, .{ .tools = &.{
            .{ .function = .{ .name = "status" } },
            .{ .function = .{ .name = "status", .description = "second registration" } },
        } });
        if (duplicate) |_| session.close() else |_| {}
        try std.testing.expectError(error.InvalidResponse, duplicate);
        try gemini.live(&session, .{ .tools = &.{
            .{ .function = .{ .name = "status" } },
            .{ .function = .{
                .name = "verbatim",
                .parameters = try sdk.JsonValue.fromBorrowedValue(arena, .{ .object = .empty }),
            } },
        } });
        defer session.close();
        const Event = Gemini.LiveSession.Event;
        try session.send(.{ .text = "find 42" });
        const call = try session.recv();
        try std.testing.expect(call == .tool_call);
        try std.testing.expectEqualStrings("status", call.tool_call.name);
        try std.testing.expect(call.tool_call.input == .object);
        try std.testing.expectEqual(@as(usize, 0), call.tool_call.input.object.count());
        // Reply before draining the queued usage, as a tool dispatcher may.
        try session.send(.{ .tool_result = .{ .id = call.tool_call.id, .parts = &.{.{ .text = .{ .text = "found" } }} } });
        try std.testing.expectEqualDeep(@as(Event, .{ .usage = .{
            .input_tokens = .{ .integer = 7 },
            .output_tokens = .{ .integer = 2 },
            .total_tokens = .{ .integer = 9 },
            .cache_read_tokens = .{ .integer = 0 },
            .reasoning_tokens = .{ .integer = 0 },
        } }), try session.recv());
        try std.testing.expectEqualDeep(@as(Event, .{ .text = "done" }), try session.recv());
        try std.testing.expectEqualDeep(@as(Event, .{ .turn_end = .{
            .input_tokens = .{ .integer = 11 },
            .output_tokens = .{ .integer = 4 },
            .total_tokens = .{ .integer = 15 },
            .cache_read_tokens = .{ .integer = 0 },
            .reasoning_tokens = .{ .integer = 0 },
        } }), try session.recv());
        session.close();
        try serving.await(io);
    }
}

test "Gemini Live streamed completion folds judgment answers into data parts" {
    const Fixture = live_test.Fixture;
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const setup =
        \\{"setup":{"model":"models/gemini-live-native-audio","generationConfig":{"responseModalities":["AUDIO"]},"outputAudioTranscription":{}}}
    ;
    const outer_first =
        \\{"serverContent":{"modelTurn":{"parts":[{"text":"\u00a0{\"pick\":"}]}}}
    ;
    const outer_last =
        \\{"serverContent":{"modelTurn":{"parts":[{"text":"true}\u3000"}]},"turnComplete":true}}
    ;
    const inner_first =
        \\{"serverContent":{"modelTurn":{"parts":[{"text":"{\u00a0\"pick\":"}]}}}
    ;
    const inner_last =
        \\{"serverContent":{"modelTurn":{"parts":[{"text":"true}"}]},"turnComplete":true}}
    ;
    const Capture = struct {
        expected: [2][]const u8,
        count: usize = 0,
        unchanged: bool = true,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (event == .delta and event.delta.data == .text) {
                self.unchanged = self.unchanged and self.count < self.expected.len and
                    std.mem.eql(u8, self.expected[self.count], event.delta.data.text.text);
                self.count += 1;
            }
            return .proceed;
        }
    };
    for ([_]enum { judgment, no_judgment, interior_space }{ .judgment, .no_judgment, .interior_space }) |case| {
        const interior = case == .interior_space;
        const schema = try @import("json_repair.zig").parseJson(arena, if (case != .no_judgment)
            \\{"properties":{"pick":{"type":"boolean"}}}
        else
            \\{"properties":{"pick":{"type":"string"}}}
        );
        var serving = try io.concurrent(Fixture.serveScenario, .{
            io,                                             &listener,                                &.{"{\"realtimeInput\":{\"text\":\"inspect\"}}"}, true, setup, null,
            &.{if (interior) inner_first else outer_first}, if (interior) inner_last else outer_last, null,
        });
        defer serving.cancel(io) catch {};
        var capture: Capture = .{ .expected = if (interior)
            .{ "{\u{a0}\"pick\":", "true}" }
        else
            .{ "\u{a0}{\"pick\":", "true}\u{3000}" } };
        const result = try gemini.lm().completeResponses(arena, .{
            .messages = &.{.{ .role = .user, .content = .{ .text = "inspect" } }},
            .options = .{ .response_format = .{ .json_schema = .{ .name = "fixture", .schema = schema } } },
            .stream_sink = .{ .context = &capture, .on_event = Capture.onEvent },
        });
        const response = result[0].response orelse return error.TestUnexpectedResult;
        defer response.release();
        try serving.await(io);
        try std.testing.expectEqual(@as(usize, 2), capture.count);
        try std.testing.expect(capture.unchanged);
        const parts = response.fields.message.parts;
        try std.testing.expectEqual(@as(usize, 1), parts.len);
        if (case == .judgment) {
            try std.testing.expect(std.meta.activeTag(parts[0]) == .data);
            const data = try parts[0].data.value.toValue(arena);
            try std.testing.expectEqual(Value{ .bool = true }, get(data, "pick"));
        } else {
            try std.testing.expect(std.meta.activeTag(parts[0]) == .text);
            const expected = if (interior) "{\u{a0}\"pick\":true}" else "\u{a0}{\"pick\":true}\u{3000}";
            try std.testing.expectEqualStrings(expected, parts[0].text.text);
        }
    }
}

test "Gemini Live closes an upgraded connection on local setup failure" {
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    const gemini: Gemini = .{
        .client = &client,
        .api_key = "local",
        .model = "gemini-live",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const extensions = try sdk.JsonValue.fromBorrowedValue(arena, str("not an object"));
    var serving = try io.concurrent(live_test.Fixture.serveFailedSetup, .{ io, &listener });
    defer serving.cancel(io) catch {};
    var session: Gemini.LiveSession = undefined;
    try std.testing.expectError(error.LMFailure, gemini.live(&session, .{ .extensions = extensions }));
    try serving.await(io);
}

test "Gemini Live refuses invalid WebSocket upgrades before sending setup" {
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local",
        .model = "gemini-live",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "hello" } }} }},
    });
    defer request.release();
    const Capture = struct {
        fn onEvent(context: *anyopaque, _: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const event_count: *usize = @ptrCast(@alignCast(context));
            event_count.* += 1;
            return .proceed;
        }
    };
    for ([_]struct {
        version: []const u8 = "HTTP/1.1",
        headers: []const u8 = "Upgrade: websocket\r\n",
        interim: []const u8 = "",
    }{
        .{ .headers = "Upgrade: h2c, websocket\r\n" },
        .{ .headers = "Upgrade: websocket\r\nuPgRaDe: websocket\r\n" },
        .{ .version = "HTTP/1.0" },
        .{ .headers = "Upgrade: websocket\r\nContent-Length: 0\r\n" },
        .{ .headers = "Upgrade: websocket\r\nTransfer-Encoding: chunked\r\n" },
        .{ .interim = "HTTP/1.1 103 Early Hints\r\nContent-Length: 0\r\n\r\n" },
        .{ .interim = "HTTP/1.1 100 Continue\r\nTransfer-Encoding: chunked\r\n\r\n" },
    }) |invalid| {
        for ([_]bool{ false, true }) |streaming| {
            var serving = try io.concurrent(live_test.Fixture.serveInvalidUpgrade, .{ io, &listener, invalid.interim, invalid.version, invalid.headers });
            defer _ = serving.cancel(io) catch null;
            if (streaming) {
                var event_count: usize = 0;
                const result = gemini.streamRequest(arena, request, .{
                    .context = &event_count,
                    .on_event = Capture.onEvent,
                });
                try std.testing.expectEqual(@as(?u8, null), try serving.await(io));
                try std.testing.expectError(error.LMFailure, result);
                try std.testing.expectEqual(@as(usize, 0), event_count);
            } else {
                var session: Gemini.LiveSession = undefined;
                const result = gemini.live(&session, .{});
                if (result) |_| session.close() else |_| {}
                try std.testing.expectEqual(@as(?u8, null), try serving.await(io));
                try std.testing.expectError(error.InvalidUpgrade, result);
            }
        }
    }
}

test "Gemini Live completes turns after interim HTTP responses" {
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "inspect" } }} }},
    });
    defer request.release();

    for ([_]bool{ false, true }) |streaming| {
        var serving = try io.concurrent(live_test.Fixture.serveInterimUpgrade, .{ io, &listener, streaming });
        defer serving.cancel(io) catch {};
        if (streaming) {
            var received = false;
            const outcome = try gemini.streamRequest(arena, request, .{
                .context = &received,
                .on_event = live_test.Fixture.onEvent,
            });
            try std.testing.expectEqual(.exhausted, outcome);
            try std.testing.expect(received);
        } else {
            var session: Gemini.LiveSession = undefined;
            try gemini.live(&session, .{});
            defer session.close();
            try session.send(.{ .text = "inspect" });
            try std.testing.expectEqualDeep(Gemini.LiveSession.Event{ .text = "received" }, try session.recv());
        }
        try serving.await(io);
    }
}

test "Gemini Live skips binary JSON that is not UTF-8" {
    const Fixture = live_test.Fixture;
    const Peer = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [2048]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            try Fixture.upgrade(&input.interface, &output.interface);
            var frame: [1024]u8 = undefined;
            try std.testing.expect(std.mem.startsWith(u8, try Fixture.readFrame(&input.interface, &frame), "{\"setup\":"));
            try Fixture.sendDataFrame(&output.interface, 0x82, "{\"error\":{\"message\":\"\xed\xa0\x80\"}}");
            try Fixture.sendFrame(&output.interface, "{\"setupComplete\":{}}");
            try std.testing.expectEqualStrings("{\"realtimeInput\":{\"text\":\"inspect\"}}", try Fixture.readFrame(&input.interface, &frame));
            try Fixture.sendDataFrame(&output.interface, 0x82, "{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"text\":\"\xed\xa0\x80\"}]}}}");
            try Fixture.sendDataFrame(&output.interface, 0x02, "{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"text\":\"received \xc3");
            try Fixture.sendDataFrame(&output.interface, 0x80, "\xa9\"}]},\"turnComplete\":true}}");
            try std.testing.expectEqualStrings("\x03\xe8", try Fixture.readMaskedFrame(&input.interface, &frame, 0x88));
        }
    };
    const Capture = struct {
        count: usize = 0,
        matches: bool = false,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (event == .delta and event.delta.data == .text) {
                self.count += 1;
                self.matches = std.mem.eql(u8, event.delta.data.text.text, "received é");
            }
            return .proceed;
        }
    };
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "local",
        .model = "gemini-live-native-audio",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "inspect" } }} }},
    });
    defer request.release();
    for ([_]bool{ false, true }) |streaming| {
        var serving = try io.concurrent(Peer.serve, .{ io, &listener });
        defer serving.cancel(io) catch {};
        if (streaming) {
            var capture: Capture = .{};
            try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
                .context = &capture,
                .on_event = Capture.onEvent,
            }));
            try std.testing.expectEqual(@as(usize, 1), capture.count);
            try std.testing.expect(capture.matches);
        } else {
            var session: Gemini.LiveSession = undefined;
            try gemini.live(&session, .{});
            defer session.close();
            try session.send(.{ .text = "inspect" });
            try std.testing.expectEqualDeep(Gemini.LiveSession.Event{ .text = "received é" }, try session.recv());
            try std.testing.expectEqualDeep(Gemini.LiveSession.Event{ .turn_end = null }, try session.recv());
        }
        try serving.await(io);
    }
}

test "Gemini Live clips a truncated WAV data chunk to the available samples" {
    const header = "RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00" ++
        "\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00";
    const raw = header ++ "data\xff\xff\xff\xff\x01\x02";
    const pcm = liveWave(raw);
    try std.testing.expectEqual(22050, pcm.rate);
    try std.testing.expectEqualStrings("\x01\x02", pcm.bytes);
}

test "Gemini SSE preserves diagnostic precedence and emits no successful terminal event" {
    const Capture = struct {
        detail: ?StreamError = null,
        event_count: usize = 0,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.event_count += 1;
            if (event == .provider_error) self.detail = event.provider_error;
            return .proceed;
        }
    };
    const cases = [_]struct { frame: []const u8, expected: StreamError }{
        .{
            .frame = "{\"error\":{\"status\":\"RESOURCE_EXHAUSTED\",\"message\":\"Token LIMIT\"}}",
            .expected = .{
                .code = .context_length,
                .provider_code = "RESOURCE_EXHAUSTED",
                .message = "Token LIMIT",
            },
        },
        .{
            .frame = "{\"error\":null,\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}",
            .expected = .{
                .code = .provider,
                .provider_code = "provider",
                .message = "provider",
            },
        },
        .{
            .frame = "{\"candidates\":[{\"finishReason\":\"SAFETY\",\"finishMessage\":\"Blocked\"}]}",
            .expected = .{
                .code = .invalid_request,
                .provider_code = "inband_finish_reason",
                .message = "Blocked (gemini)",
            },
        },
    };
    for (cases) |case| {
        var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer scratch.deinit();
        const arena = scratch.allocator();
        var capture: Capture = .{};
        var state: GenerateStream = .{
            .model = "fixture",
            .adaptations = &.{},
            .sink = .{ .context = &capture, .on_event = Capture.onEvent },
        };
        const frame = try @import("json_repair.zig").parseJson(arena, case.frame);
        try std.testing.expect(try state.dispatch(arena, frame) == .abort);
        try std.testing.expectEqual(1, capture.event_count);
        try std.testing.expectEqualDeep(case.expected, capture.detail.?);
    }
}

test "Gemini SSE joins split text without merging parallel tool calls" {
    const Capture = struct {
        indexes: [4]u8 = undefined,
        used: usize = 0,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (event == .delta) {
                const index = event.delta.part_index.?;
                if (self.used == self.indexes.len or index.len != 1) return .stop;
                self.indexes[self.used] = index[0];
                self.used += 1;
            }
            return .proceed;
        }
    };
    var scratch = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer scratch.deinit();
    var capture: Capture = .{};
    var state: GenerateStream = .{
        .model = "gemini-3-flash",
        .adaptations = &.{},
        .sink = .{ .context = &capture, .on_event = Capture.onEvent },
    };
    const frames =
        \\data: {"candidates":[{"content":{"parts":[{"text":"Pa"}]}}]}
        \\
        \\data: {"candidates":[{"content":{"parts":[{"text":"ris"}]}}]}
        \\
        \\data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"a"}}]}}]}
        \\
        \\data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"b"}}]}}]}
        \\
        \\
    ;
    var reader = std.Io.Reader.fixed(frames);
    try GenerateStream.consume(&state, scratch.allocator(), &reader, frames.len);
    try std.testing.expectEqualStrings("0012", capture.indexes[0..capture.used]);
    try std.testing.expectEqual(Gemini.StreamOutcome.exhausted, state.outcome);
}

test "Gemini generation honors normalized HTTP and RPC retry decisions" {
    const Case = enum {
        created,
        bad_request,
        context_overflow,
        context_overflow_permission,
        context_overflow_string,
        context_overflow_array,
        context_overflow_in_stream,
        invalid_input_in_stream,
        rate_limit_in_stream,
        filter_lowercase,
        filter_uppercase,
        prompt_block_numeric,
        prompt_block_zero,
        prompt_block_unspecified,
        malformed_candidates,
        empty_candidate_object,
        numeric_parts,
        true_parts,
        zero_parts,
        false_parts,
        null_parts,
        invalid_terminal_usage,
        invalid_terminal_usage_after_delta,
        valid_terminal_usage,
        null_rpc_message,
        invalid_key,
        permission_denied,
        unauthenticated,
        failed_precondition,
        invalid_argument,
        not_found,
        http_timeout,
        rate_limit,
        internal,
        transient,
        deadline,
    };
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, streaming: bool, case: Case, status: std.http.Status, retry: bool) !void {
            try reply(io, listener, streaming, case, status, null);
            if (retry) try reply(io, listener, streaming, .created, .created, null);
        }

        fn promptBlock(comptime reason: []const u8, streaming: bool) []const u8 {
            const payload = "{\"promptFeedback\":{\"blockReason\":" ++ reason ++
                "},\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]},\"finishReason\":\"STOP\"}]}";
            return if (streaming) "data: " ++ payload ++ "\n\n" else payload;
        }

        fn candidateEnvelope(comptime candidates: []const u8, streaming: bool) []const u8 {
            const payload = "{\"candidates\":" ++ candidates ++ ",\"usageMetadata\":{\"totalTokenCount\":4}}";
            return if (streaming) "data: " ++ payload ++ "\n\n" else payload;
        }

        fn contentParts(comptime parts: []const u8, streaming: bool) []const u8 {
            return candidateEnvelope(
                "[{\"content\":{\"parts\":" ++ parts ++ "},\"finishReason\":\"STOP\"}]",
                streaming,
            );
        }

        fn reply(io: std.Io, listener: *std.Io.net.Server, streaming: bool, case: Case, status: std.http.Status, headers: ?[]const std.http.Header) !void {
            const stream = try listener.accept(io);
            defer stream.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = stream.reader(io, &read_buffer);
            var output = stream.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try std.testing.expectEqual(.POST, request.head.method);
            try std.testing.expectEqualStrings(if (streaming)
                "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse"
            else
                "/v1beta/models/gemini-3-flash:generateContent", request.head.target);
            const expected = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Capital?\"}]}]}";
            try std.testing.expectEqual(expected.len, request.head.content_length.?);
            var body: [expected.len]u8 = undefined;
            var body_buffer: [256]u8 = undefined;
            try (try request.readerExpectContinue(&body_buffer)).readSliceAll(&body);
            try std.testing.expectEqualStrings(expected, &body);

            const success =
                \\{"candidates":[{"content":{"parts":[{"text":"Paris"}]},"finishReason":"STOP"}]}
            ;
            const response_body = switch (case) {
                // A decodable 400 body must not publish a response or events.
                .created, .bad_request => if (streaming) "data: " ++ success ++ "\n\n" else success,
                .context_overflow =>
                \\{"error":{"status":"INVALID_ARGUMENT","message":"The input token count exceeds the maximum number of tokens allowed."}}
                ,
                .invalid_key =>
                \\{"error":{"status":"INVALID_ARGUMENT","message":"Token limit","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}
                ,
                .context_overflow_permission => "{\"error\":{\"status\":\"PERMISSION_DENIED\",\"message\":\"Token limit\"}}",
                .context_overflow_string => "{\"error\":\"Token limit exceeded\"}",
                .context_overflow_array => "{\"error\":[\"Token limit exceeded\"]}",
                .context_overflow_in_stream, .invalid_input_in_stream => blk: {
                    const partial = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n";
                    break :blk if (case == .context_overflow_in_stream)
                        partial ++ "data: {\"error\":{\"status\":\"INVALID_ARGUMENT\",\"message\":\"Token limit exceeded\"}}\n\n"
                    else
                        partial ++ "data: {\"error\":{\"status\":\"INVALID_ARGUMENT\",\"message\":\"Invalid input\"}}\n\n";
                },
                .rate_limit_in_stream => "data: {\"error\":{\"status\":\"RESOURCE_EXHAUSTED\",\"message\":\"Try again later\"}}\n\n",
                .filter_lowercase, .filter_uppercase => filter: {
                    const response = if (case == .filter_lowercase)
                        "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]},\"finishReason\":\"safety\"}]}"
                    else
                        "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]},\"finishReason\":\"SAFETY\"}]}";
                    break :filter if (streaming) "data: " ++ response ++ "\n\n" else response;
                },
                .prompt_block_numeric => promptBlock("1", streaming),
                .prompt_block_zero => promptBlock("0", streaming),
                .prompt_block_unspecified => promptBlock("\"BLOCK_REASON_UNSPECIFIED\"", streaming),
                .malformed_candidates => candidateEnvelope("{\"content\":{\"parts\":[{\"text\":\"Paris\"}]}}", streaming),
                .empty_candidate_object => candidateEnvelope("{}", streaming),
                .numeric_parts => contentParts("1", streaming),
                .true_parts => contentParts("true", streaming),
                .zero_parts => contentParts("0", streaming),
                .false_parts => contentParts("false", streaming),
                .null_parts => contentParts("null", streaming),
                .invalid_terminal_usage, .invalid_terminal_usage_after_delta, .valid_terminal_usage => usage: {
                    const before = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Before\"}]}}]}\n\n";
                    const start = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Paris\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":";
                    const invalid = start ++ "\"invalid\"}}\n\n";
                    break :usage switch (case) {
                        .invalid_terminal_usage => invalid,
                        .invalid_terminal_usage_after_delta => before ++ invalid,
                        .valid_terminal_usage => start ++ "3}}\n\n",
                        else => unreachable,
                    };
                },
                .null_rpc_message => "{\"error\":{\"status\":\"UNAVAILABLE\",\"message\":null}}",
                .permission_denied => "{\"error\":{\"status\":\"PERMISSION_DENIED\",\"message\":\"Access denied\"}}",
                .unauthenticated => "{\"error\":{\"status\":\"UNAUTHENTICATED\",\"message\":\"Authentication required\"}}",
                .failed_precondition => "{\"error\":{\"status\":\"FAILED_PRECONDITION\",\"message\":\"Billing disabled\"}}",
                .invalid_argument => "{\"error\":{\"status\":\"INVALID_ARGUMENT\",\"message\":\"Invalid input\"}}",
                .not_found => "{\"error\":{\"status\":\"NOT_FOUND\",\"message\":\"Model not found\"}}",
                .http_timeout => "{\"error\":{\"message\":\"Try again later\"}}",
                .rate_limit => "{\"error\":{\"status\":\"RESOURCE_EXHAUSTED\",\"message\":\"Try again later\"}}",
                .internal => "{\"error\":{\"status\":\"INTERNAL\",\"message\":\"Try again later\"}}",
                .transient => "{\"error\":{\"status\":\"UNAVAILABLE\",\"message\":\"Try again later\"}}",
                .deadline => "{\"error\":{\"status\":\"DEADLINE_EXCEEDED\",\"message\":\"Try again later\"}}",
            };
            try request.respond(response_body, .{
                .status = status,
                .keep_alive = false,
                .extra_headers = headers orelse if (case == .context_overflow_in_stream) &.{
                    .{ .name = "content-type", .value = "text/event-stream" },
                    .{ .name = "request-id", .value = "gateway-lower-priority" },
                    .{ .name = "X-Request-Id", .value = "gemini-first" },
                    .{ .name = "x-request-id", .value = "gemini-second" },
                } else &.{.{ .name = "content-type", .value = if (streaming and
                    (case == .created or case == .bad_request or case == .invalid_input_in_stream or
                        case == .rate_limit_in_stream or case == .filter_lowercase or case == .filter_uppercase or
                        case == .prompt_block_numeric or case == .prompt_block_zero or case == .prompt_block_unspecified or
                        case == .malformed_candidates or case == .empty_candidate_object or
                        case == .numeric_parts or case == .true_parts or case == .zero_parts or
                        case == .false_parts or case == .null_parts or case == .invalid_terminal_usage or
                        case == .invalid_terminal_usage_after_delta or case == .valid_terminal_usage))
                    "text/event-stream"
                else
                    "application/json" }},
            });
        }
    };
    const Attempts = struct {
        count: usize = 0,
        limit: usize = 0,

        fn prepare(context: *anyopaque, _: Allocator, headers: []const std.http.Header) ![]const std.http.Header {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.count += 1;
            // An unexpected retry fails promptly instead of opening an unserved connection.
            if (self.count > self.limit) return error.UnexpectedRetry;
            return headers;
        }
    };
    const Capture = struct {
        events: [3]u8 = undefined,
        len: usize = 0,
        request_id_matches: bool = false,
        finish_reason: ?sdk.FinishReason = null,
        input_tokens: Value = .null,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (self.len == self.events.len) return .stop;
            self.events[self.len] = switch (event) {
                .start => 's',
                .delta => if (event.delta.data == .text) blk: {
                    const bytes = event.delta.data.text.text;
                    if (std.mem.eql(u8, "Paris", bytes)) break :blk 'd';
                    if (std.mem.eql(u8, "partial", bytes)) break :blk 'p';
                    if (std.mem.eql(u8, "Before", bytes)) break :blk 'b';
                    break :blk '?';
                } else '?',
                .end => end: {
                    self.finish_reason = event.end.finish_reason;
                    if (event.end.usage) |usage| self.input_tokens = usage.input_tokens;
                    break :end 'e';
                },
                .provider_error => |detail| blk: {
                    self.request_id_matches = switch (detail.code) {
                        .context_length => std.mem.eql(u8, detail.request_id orelse "", "gemini-first"),
                        .invalid_request => detail.request_id == null,
                        else => false,
                    };
                    break :blk switch (detail.code) {
                        .context_length => 'c',
                        .invalid_request => 'i',
                        else => '?',
                    };
                },
            };
            self.len += 1;
            return .proceed;
        }
    };
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var attempts: Attempts = .{};
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "loopback",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .header_provider = .{ .context = &attempts, .prepare = Attempts.prepare },
        .max_retries = 1,
        .retry_base_ms = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Capital?" } }} }},
    });
    defer request.release();
    for ([_]struct { Case, std.http.Status }{
        .{ .created, .created },
        .{ .bad_request, .bad_request },
        .{ .malformed_candidates, .ok },
        .{ .empty_candidate_object, .ok },
        .{ .numeric_parts, .ok },
        .{ .true_parts, .ok },
        .{ .zero_parts, .ok },
        .{ .false_parts, .ok },
        .{ .null_parts, .ok },
        .{ .invalid_key, .bad_request },
        .{ .context_overflow, .bad_request },
        .{ .context_overflow, .payload_too_large },
        .{ .context_overflow_string, .payload_too_large },
        .{ .context_overflow_array, .payload_too_large },
        .{ .null_rpc_message, .bad_request },
        .{ .context_overflow, .too_many_requests },
        .{ .context_overflow, .service_unavailable },
        .{ .invalid_key, .service_unavailable },
        .{ .permission_denied, .service_unavailable },
        .{ .unauthenticated, .service_unavailable },
        .{ .failed_precondition, .service_unavailable },
        .{ .invalid_argument, .service_unavailable },
        .{ .not_found, .service_unavailable },
        .{ .context_overflow_permission, .service_unavailable },
        .{ .transient, .service_unavailable },
        .{ .http_timeout, .request_timeout },
        .{ .rate_limit, .bad_request },
        .{ .internal, .bad_request },
        .{ .transient, .bad_request },
        .{ .deadline, .bad_request },
        .{ .permission_denied, .request_timeout },
    }) |row| {
        const case, const status = row;
        const retry = switch (case) {
            .http_timeout, .rate_limit, .internal, .transient, .deadline => true,
            else => false,
        };
        const empty = switch (case) {
            .empty_candidate_object, .zero_parts, .false_parts, .null_parts => true,
            else => false,
        };
        const succeeded = case == .created or empty or retry;
        const expected_error = switch (case) {
            .context_overflow, .context_overflow_permission, .context_overflow_string, .context_overflow_array => error.ContextWindowExceeded,
            else => error.LMFailure,
        };
        for ([_]bool{ false, true }) |streaming| {
            attempts = .{ .limit = if (retry) 2 else 1 };
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, streaming, case, status, retry });
            defer serving.cancel(io) catch {};
            if (streaming) {
                var capture: Capture = .{};
                const outcome = gemini.streamRequest(arena, request, .{ .context = &capture, .on_event = Capture.onEvent });
                if (succeeded) {
                    try std.testing.expectEqual(.exhausted, try outcome);
                    try std.testing.expectEqualStrings(if (empty) "se" else "sde", capture.events[0..capture.len]);
                    try std.testing.expectEqual(.stop, capture.finish_reason);
                } else {
                    try std.testing.expectError(expected_error, outcome);
                    try std.testing.expectEqual(0, capture.len);
                }
            } else {
                const response = gemini.completeRequest(arena, request);
                defer if (response) |value| value.release() else |_| {};
                if (succeeded) {
                    const fields = (try response).fields;
                    try std.testing.expectEqual(.stop, fields.finish_reason);
                    try std.testing.expectEqualStrings(if (empty) "" else "Paris", fields.message.parts[0].text.text);
                } else {
                    try std.testing.expectError(expected_error, response);
                }
            }
            try serving.await(io);
            try std.testing.expectEqual(attempts.limit, attempts.count);
        }
    }
    for ([_]struct { Case, []const u8 }{
        .{ .invalid_terminal_usage, "" },
        .{ .invalid_terminal_usage_after_delta, "sb" },
        .{ .valid_terminal_usage, "sde" },
    }) |row| {
        const case, const expected = row;
        attempts = .{ .limit = 1 };
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, true, case, .ok, false });
        defer serving.cancel(io) catch {};
        var capture: Capture = .{};
        const outcome = gemini.streamRequest(arena, request, .{ .context = &capture, .on_event = Capture.onEvent });
        try serving.await(io);
        try std.testing.expectEqual(1, attempts.count);
        if (case == .valid_terminal_usage) {
            try std.testing.expectEqual(.exhausted, try outcome);
            try std.testing.expectEqual(.stop, capture.finish_reason);
            try std.testing.expectEqual(Value{ .integer = 3 }, capture.input_tokens);
        } else {
            try std.testing.expectError(error.LMFailure, outcome);
        }
        try std.testing.expectEqualStrings(expected, capture.events[0..capture.len]);
    }

    // Check prompt blocking as well as case-sensitive finish-error admission.
    for ([_]Case{ .filter_lowercase, .filter_uppercase, .prompt_block_numeric, .prompt_block_zero, .prompt_block_unspecified }) |case| {
        const blocked = case == .filter_uppercase or case == .prompt_block_numeric;
        const finish: sdk.FinishReason = if (case == .filter_lowercase) .content_filter else .stop;
        attempts = .{ .limit = 2 };
        var serving = try io.concurrent(struct {
            fn run(fixture_io: std.Io, fixture_listener: *std.Io.net.Server, selected: Case) !void {
                try Fixture.reply(fixture_io, fixture_listener, false, selected, .ok, null);
                try Fixture.reply(fixture_io, fixture_listener, true, selected, .ok, null);
            }
        }.run, .{ io, &listener, case });
        defer serving.cancel(io) catch {};
        const response = gemini.completeRequest(arena, request);
        defer if (response) |value| value.release() else |_| {};
        if (blocked) {
            try std.testing.expectError(error.LMFailure, response);
        } else {
            const fields = (try response).fields;
            try std.testing.expectEqual(finish, fields.finish_reason);
            try std.testing.expectEqualStrings("partial", fields.message.parts[0].text.text);
        }

        var capture: Capture = .{};
        try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
            .context = &capture,
            .on_event = Capture.onEvent,
        }));
        try std.testing.expectEqualStrings(if (blocked) "i" else "spe", capture.events[0..capture.len]);
        try std.testing.expectEqual(if (blocked) null else finish, capture.finish_reason);
        try serving.await(io);
        try std.testing.expectEqual(2, attempts.count);
    }
    const RetryHint = struct {
        count: usize = 0,
        rate_limited: bool = false,
        seconds: ?f64 = null,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.count += 1;
            if (event == .provider_error) {
                self.rate_limited = event.provider_error.code == .rate_limit;
                self.seconds = event.provider_error.retry_after;
            }
            return .proceed;
        }
    };
    for ([_]struct { headers: []const std.http.Header, seconds: ?f64 }{
        .{ .headers = &.{
            .{ .name = "ReTrY-AfTeR", .value = "2.5" },
            .{ .name = "retry-after", .value = "9" },
            .{ .name = "retry-after-ms", .value = "4000" },
        }, .seconds = 2.5 },
        .{ .headers = &.{
            .{ .name = "Retry-After", .value = "NaN" },
            .{ .name = "Retry-After-Ms", .value = "125.5" },
            .{ .name = "X-MS-Retry-After-MS", .value = "800" },
        }, .seconds = 0.1255 },
        .{ .headers = &.{
            .{ .name = "retry-after", .value = "-1" },
            .{ .name = "retry-after-ms", .value = "0x10" },
            .{ .name = "x-ms-retry-after-ms", .value = "0" },
        }, .seconds = 0 },
        .{ .headers = &.{
            .{ .name = "retry-after", .value = "inf" },
            .{ .name = "retry-after-ms", .value = "-0" },
            .{ .name = "x-ms-retry-after-ms", .value = "1e999" },
        }, .seconds = null },
        .{ .headers = &.{}, .seconds = null },
    }) |row| {
        attempts = .{ .limit = 1 };
        var serving = try io.concurrent(Fixture.reply, .{ io, &listener, true, .rate_limit_in_stream, .ok, row.headers });
        defer serving.cancel(io) catch {};
        var hint: RetryHint = .{};
        try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
            .context = &hint,
            .on_event = RetryHint.onEvent,
        }));
        try serving.await(io);
        try std.testing.expectEqual(1, attempts.count);
        try std.testing.expectEqual(1, hint.count);
        try std.testing.expect(hint.rate_limited);
        try std.testing.expectEqual(row.seconds, hint.seconds);
    }
    for ([_]Case{ .context_overflow_in_stream, .invalid_input_in_stream }) |case| {
        const overflow = case == .context_overflow_in_stream;
        for ([_]bool{ false, true }) |collect| {
            attempts = .{ .limit = 1 };
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, true, case, .ok, false });
            defer serving.cancel(io) catch {};
            var capture: Capture = .{};
            const sink: Gemini.StreamSink = .{ .context = &capture, .on_event = Capture.onEvent };
            if (collect) {
                const result = gemini.lm().completeResponses(arena, .{
                    .messages = &.{.{ .role = .user, .content = .{ .text = "Capital?" } }},
                    .stream_sink = sink,
                });
                try std.testing.expectError(if (overflow) error.ContextWindowExceeded else error.LMFailure, result);
            } else {
                try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, sink));
            }
            try serving.await(io);
            try std.testing.expectEqual(1, attempts.count);
            try std.testing.expectEqualStrings(if (overflow) "spc" else "spi", capture.events[0..capture.len]);
            try std.testing.expect(capture.request_id_matches);
        }
    }
}

test "Gemini preserves generated media with gateway scalar fields" {
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, streaming: bool) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            const expected = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Show media\"}]}]}";
            try std.testing.expectEqual(expected.len, request.head.content_length.?);
            var body: [expected.len]u8 = undefined;
            var body_buffer: [256]u8 = undefined;
            try (try request.readerExpectContinue(&body_buffer)).readSliceAll(&body);
            const payload = "{\"candidates\":[{\"content\":{\"parts\":[" ++
                "{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":1234}}," ++
                "{\"inlineData\":{\"mimeType\":false,\"data\":\"cGRm\"}}," ++
                "{\"inlineData\":{\"mimeType\":\"audio/wav\",\"data\":\"YWJj\"}}," ++
                "{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":false}}" ++
                "]},\"finishReason\":\"STOP\"}]}";
            try request.respond(if (streaming) "data: " ++ payload ++ "\n\n" else payload, .{
                .keep_alive = false,
                .extra_headers = &.{.{ .name = "content-type", .value = if (streaming) "text/event-stream" else "application/json" }},
            });
            try std.testing.expectEqualStrings(expected, &body);
        }
    };
    const Capture = struct {
        next: usize = 0,
        valid: bool = true,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (event != .delta) return .proceed;
            const key = event.delta.part_index orelse "";
            const data = event.delta.data;
            self.valid = self.valid and switch (self.next) {
                0 => std.mem.eql(u8, key, "0") and data == .image and std.mem.eql(u8, data.image.data orelse "", "1234"),
                1 => std.mem.eql(u8, key, "2") and data == .audio and std.mem.eql(u8, data.audio.data orelse "", "YWJj"),
                2 => std.mem.eql(u8, key, "3") and data == .image and if (data.image.data) |bytes| bytes.len == 0 else false,
                else => false,
            };
            self.next += 1;
            return .proceed;
        }
    };
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "loopback",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Show media" } }} }},
    });
    defer request.release();

    for ([_]bool{ false, true }) |streaming| {
        var serving = try io.concurrent(Fixture.serve, .{ io, &listener, streaming });
        defer serving.cancel(io) catch {};
        if (streaming) {
            var capture: Capture = .{};
            try std.testing.expectEqual(.exhausted, try gemini.streamRequest(arena, request, .{
                .context = &capture,
                .on_event = Capture.onEvent,
            }));
            try serving.await(io);
            try std.testing.expectEqual(@as(usize, 3), capture.next);
            try std.testing.expect(capture.valid);
        } else {
            const response = try gemini.completeRequest(arena, request);
            defer response.release();
            try serving.await(io);
            const parts = response.fields.message.parts;
            try std.testing.expectEqual(@as(usize, 3), parts.len);
            try std.testing.expect(parts[0] == .image and parts[1] == .document and parts[2] == .audio);
            try std.testing.expectEqualStrings("1234", parts[0].image.data.?);
            try std.testing.expectEqualStrings("application/octet-stream", parts[1].document.media_type);
            try std.testing.expectEqualStrings("cGRm", parts[1].document.data.?);
            try std.testing.expectEqualStrings("YWJj", parts[2].audio.data.?);
        }
    }
}

test "Gemini generation validates usage and token score telemetry" {
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server, streaming: bool, payload: []const u8) !void {
            const socket = try listener.accept(io);
            defer socket.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = socket.reader(io, &read_buffer);
            var output = socket.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try std.testing.expectEqual(.POST, request.head.method);
            try std.testing.expectEqualStrings(if (streaming)
                "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse"
            else
                "/v1beta/models/gemini-3-flash:generateContent", request.head.target);
            const expected = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Say hello\"}]}]}";
            try std.testing.expectEqual(expected.len, request.head.content_length.?);
            var body: [expected.len]u8 = undefined;
            var body_buffer: [256]u8 = undefined;
            try (try request.readerExpectContinue(&body_buffer)).readSliceAll(&body);
            try std.testing.expectEqualStrings(expected, &body);
            var frame: [1024]u8 = undefined;
            try request.respond(if (streaming) try std.fmt.bufPrint(&frame, "data: {s}\n\n", .{payload}) else payload, .{
                .keep_alive = false,
                .extra_headers = &.{.{ .name = "content-type", .value = if (streaming) "text/event-stream" else "application/json" }},
            });
        }
    };
    const Capture = struct {
        usage: ?sdk.Usage = null,
        event_count: usize = 0,
        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.event_count += 1;
            if (event == .end) self.usage = event.end.usage;
            return .proceed;
        }
    };
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "loopback",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}/v1beta", .{listener.socket.address.getPort()}),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = gemini.model,
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Say hello" } }} }},
    });
    defer request.release();
    for ([_]struct { []const u8, Error!Value }{
        .{ "{\"bad\":true}", .null },
        .{ "[]", .{ .integer = 2 } },
        .{ "[{\"modality\":\"TEXT\",\"tokenCount\":null}]", .null },
        .{ "[{\"modality\":\"AUDIO\"}]", .{ .integer = 0 } },
        .{ "[{\"modality\":\"AUDIO\",\"tokenCount\":1},{\"modality\":\"AUDIO\",\"tokenCount\":2}]", .{ .integer = 3 } },
        .{ "[{\"modality\":\"AUDIO\",\"tokenCount\":0.5},{\"modality\":\"AUDIO\",\"tokenCount\":1.5}]", .{ .integer = 2 } },
        .{ "[{\"modality\":\"AUDIO\",\"tokenCount\":true}]", .{ .integer = 1 } },
        .{ "[{\"modality\":\"AUDIO\",\"tokenCount\":0.5}]", error.LMFailure },
        .{ "[{\"modality\":\"AUDIO\",\"tokenCount\":null}]", error.LMFailure },
        .{ "[{\"modality\":\"AUDIO\",\"tokenCount\":\"3\"}]", error.LMFailure },
    }) |row| {
        const primary, const expected_audio = row;
        const payload = try std.fmt.allocPrint(arena, "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"Hello\"}}]}},\"finishReason\":\"STOP\"}}]," ++
            "\"usageMetadata\":{{\"candidatesTokenCount\":3,\"candidatesTokensDetails\":{s}," ++
            "\"responseTokensDetails\":[{{\"modality\":\"AUDIO\",\"tokenCount\":2}}]," ++
            "\"promptTokenCount\":3,\"promptTokensDetails\":[{{\"modality\":\"AUDIO\",\"tokenCount\":0.5}}]," ++
            "\"toolUsePromptTokenCount\":2,\"toolUsePromptTokensDetails\":[{{\"modality\":\"AUDIO\",\"tokenCount\":1.5}}]}}}}", .{primary});
        for ([_]bool{ false, true }) |streaming| {
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, streaming, payload });
            defer serving.cancel(io) catch {};
            var capture: Capture = .{};
            const usage: Error!sdk.Usage = if (streaming) streamed: {
                const result = gemini.streamRequest(arena, request, .{
                    .context = &capture,
                    .on_event = Capture.onEvent,
                });
                if (result) |outcome| {
                    try std.testing.expectEqual(.exhausted, outcome);
                    break :streamed capture.usage orelse return error.TestUnexpectedResult;
                } else |err| break :streamed err;
            } else buffered: {
                if (gemini.completeRequest(arena, request)) |response| {
                    defer response.release();
                    break :buffered response.fields.usage;
                } else |err| break :buffered err;
            };
            try serving.await(io);
            if (expected_audio) |audio| {
                // All counts here are scalar integers or null, safe after the callback/response.
                const counts = try usage;
                try std.testing.expectEqual(Value{ .integer = 3 }, counts.output_tokens);
                try std.testing.expectEqual(audio, counts.output_audio_tokens);
                try std.testing.expectEqual(Value{ .integer = 5 }, counts.input_tokens);
                try std.testing.expectEqual(Value{ .integer = 2 }, counts.input_audio_tokens);
            } else |err| {
                try std.testing.expectError(err, usage);
                try std.testing.expectEqual(@as(usize, 0), capture.event_count);
            }
        }
    }
    const ScoreCapture = struct {
        events: usize = 0,
        score: ?f64 = null,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.events += 1;
            if (event == .delta and event.delta.data == .text) {
                const scores = event.delta.data.text.scores;
                if (scores.len != 0) self.score = scores[0].logprob;
            }
            return .proceed;
        }
    };
    for ([_]struct { wire: []const u8, expected: error{LMFailure}!?f64 }{
        .{ .wire = "{\"chosenCandidates\":true}", .expected = error.LMFailure },
        .{ .wire = "{\"chosenCandidates\":[{\"token\":\"Hello\",\"logProbability\":-0.5}],\"topCandidates\":1}", .expected = error.LMFailure },
        .{ .wire = "{\"chosenCandidates\":[{\"token\":\"Hello\",\"logProbability\":-0.5}],\"topCandidates\":[{\"candidates\":true}]}", .expected = error.LMFailure },
        .{ .wire = "{\"chosenCandidates\":false,\"topCandidates\":1}", .expected = null },
        .{ .wire = "{\"chosenCandidates\":[{\"token\":\"Hello\",\"logProbability\":-0.5}],\"topCandidates\":[{\"candidates\":0}]}", .expected = -0.5 },
        .{ .wire = "{\"chosenCandidates\":[{\"token\":\"Hello\",\"logProbability\":-0.5}],\"topCandidates\":{\"0\":{}}}", .expected = error.LMFailure },
    }) |row| {
        const payload = try std.fmt.allocPrint(
            arena,
            "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"Hello\"}}]}},\"finishReason\":\"STOP\",\"logprobsResult\":{s}}}]}}",
            .{row.wire},
        );
        for ([_]bool{ false, true }) |streaming| {
            var serving = try io.concurrent(Fixture.serve, .{ io, &listener, streaming, payload });
            defer serving.cancel(io) catch {};
            var capture: ScoreCapture = .{};
            const result: Error!?f64 = if (streaming) streamed: {
                const outcome = gemini.streamRequest(arena, request, .{ .context = &capture, .on_event = ScoreCapture.onEvent }) catch |err| break :streamed err;
                try std.testing.expectEqual(.exhausted, outcome);
                break :streamed capture.score;
            } else buffered: {
                const response = gemini.completeRequest(arena, request) catch |err| break :buffered err;
                defer response.release();
                try std.testing.expectEqualStrings("Hello", response.fields.message.parts[0].text.text);
                const scores = response.fields.logprobs orelse &.{};
                break :buffered if (scores.len == 0) null else scores[0].logprob;
            };
            try serving.await(io);
            if (row.expected) |score| {
                try std.testing.expectEqual(score, try result);
                if (streaming) try std.testing.expectEqual(@as(usize, 3), capture.events);
            } else |err| {
                try std.testing.expectError(err, result);
                try std.testing.expectEqual(@as(usize, 0), capture.events);
            }
        }
    }
}

test "Gemini SSE preserves tool correlators after unstreamed inline media" {
    const Fixture = struct {
        fn serve(io: std.Io, listener: *std.Io.net.Server) !void {
            const stream = try listener.accept(io);
            defer stream.close(io);
            var read_buffer: [4096]u8 = undefined;
            var write_buffer: [1024]u8 = undefined;
            var input = stream.reader(io, &read_buffer);
            var output = stream.writer(io, &write_buffer);
            var server = std.http.Server.init(&input.interface, &output.interface);
            var request = try server.receiveHead();
            try std.testing.expectEqualStrings(
                "/v1beta/models/gemini-3-flash:streamGenerateContent?alt=sse",
                request.head.target,
            );
            const expected = "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Find it\"}]}]}";
            const len = request.head.content_length orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(expected.len, len);
            var body: [expected.len]u8 = undefined;
            var body_buffer: [256]u8 = undefined;
            try (try request.readerExpectContinue(&body_buffer)).readSliceAll(&body);
            try request.respond(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Before\"}," ++
                    "{\"inlineData\":{\"mimeType\":\"application/pdf\",\"data\":\"cGRm\"}}]}}]}\n\n" ++
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"After\"}," ++
                    "{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"都市\":\"Zürich 🗺\",\"exact\":9007199254740993}}}]},\"finishReason\":\"STOP\"}]}\n\n",
                .{
                    .keep_alive = false,
                    .extra_headers = &.{.{ .name = "content-type", .value = "text/event-stream" }},
                },
            );
            try std.testing.expectEqualStrings(expected, &body);
        }
    };
    const Capture = struct {
        indexes: [3]u8 = undefined,
        used: usize = 0,
        args_match: bool = false,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (event == .delta) {
                const index = event.delta.part_index orelse return .stop;
                if (self.used == self.indexes.len or index.len != 1) return .stop;
                self.indexes[self.used] = index[0];
                self.used += 1;
                if (event.delta.data == .tool_call) {
                    // Captured from the locked Python GeminiLM.stream.
                    const expected = "{\"\\u90fd\\u5e02\":\"Z\\u00fcrich \\ud83d\\uddfa\",\"exact\":9007199254740993}";
                    self.args_match = std.mem.eql(u8, event.delta.data.tool_call.input, expected);
                }
            }
            return .proceed;
        }
    };
    const io = std.testing.io;
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", 0);
    var listener = try address.listen(io, .{});
    defer listener.deinit(io);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var gemini: Gemini = .{
        .client = &client,
        .api_key = "loopback",
        .model = "gemini-3-flash",
        .base_url = try std.fmt.allocPrint(
            arena,
            "http://127.0.0.1:{d}/v1beta",
            .{listener.socket.address.getPort()},
        ),
        .max_retries = 0,
        .timeout_ns = 5 * std.time.ns_per_s,
    };

    var serving = try io.concurrent(Fixture.serve, .{ io, &listener });
    defer serving.cancel(io) catch {};
    var capture: Capture = .{};
    const results = try gemini.lm().completeResponses(arena, .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "Find it" } }},
        .stream_sink = .{ .context = &capture, .on_event = Capture.onEvent },
    });
    const response = results[0].response orelse return error.TestUnexpectedResult;
    defer response.release();
    try serving.await(io);

    // Pinned DSPy reserves slot 1 for inline media but emits no delta for it.
    try std.testing.expectEqualStrings("023", capture.indexes[0..capture.used]);
    try std.testing.expectEqual(sdk.FinishReason.tool_call, response.fields.finish_reason);
    const parts = response.fields.message.parts;
    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expect(parts[0] == .text and parts[1] == .text and parts[2] == .tool_call);
    try std.testing.expectEqualStrings("Before", parts[0].text.text);
    try std.testing.expectEqualStrings("After", parts[1].text.text);
    try std.testing.expectEqualStrings("lookup", parts[2].tool_call.name);
    try std.testing.expectEqualStrings("tool_call_3", parts[2].tool_call.id);
    try std.testing.expect(capture.args_match);
    const args = parts[2].tool_call.input.root.object;
    try std.testing.expectEqualStrings("Zürich 🗺", args.get("都市").?.string);
    try std.testing.expectEqual(@as(i64, 9007199254740993), args.get("exact").?.integer);
}

test "Gemini SSE publishes only the final terminal event and retains late usage" {
    const Capture = struct {
        events: [6]u8 = undefined,
        count: usize = 0,
        finish: ?sdk.FinishReason = null,
        total: Value = .null,
        saw_usage: bool = false,
        raw_usage: bool = false,
        stop_on_delta: bool,

        fn onEvent(context: *anyopaque, event: Gemini.StreamEvent) Gemini.StreamSink.Action {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.events[self.count] = switch (event) {
                .start => 's',
                .delta => 'd',
                .provider_error => 'x',
                .end => |end| blk: {
                    self.finish = end.finish_reason;
                    self.saw_usage = end.usage != null;
                    self.total = if (end.usage) |usage| usage.total_tokens else .null;
                    self.raw_usage = end.provider_data.?.root.object.contains("usageMetadata");
                    break :blk 'e';
                },
            };
            self.count += 1;
            return if (self.stop_on_delta and event == .delta) .stop else .proceed;
        }
    };
    const delta = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n";
    const finish = "data: {\"candidates\":[{\"finishReason\":\"MAX_TOKENS\"}]}\n\n";
    const usage = "data: {\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1,\"totalTokenCount\":4}}\n\n";
    for ([_]struct {
        frames: []const u8,
        sequence: []const u8,
        reason: ?sdk.FinishReason = null,
        raw_usage: bool = false,
        total: ?i64 = null,
        stop: bool = false,
    }{
        .{ .frames = delta ++ finish ++ usage, .sequence = "sde", .reason = .stop, .raw_usage = true, .total = 4 },
        .{ .frames = finish ++ delta, .sequence = "sde", .reason = .length },
        .{ .frames = "data: {\"usageMetadata\":null}\n\n", .sequence = "se", .reason = .stop, .raw_usage = true },
        .{ .frames = delta, .sequence = "sd" },
        .{ .frames = finish ++ delta ++ usage, .sequence = "sd", .stop = true },
        .{ .frames = delta ++ "data: {\"candidates\":[{\"finishReason\":42}]}\n\n", .sequence = "sde", .reason = .stop },
        .{ .frames = delta ++ "data: {\"candidates\":[{\"finishReason\":0}]}\n\n", .sequence = "sd" },
        .{ .frames = finish ++ "data: {\"error\":{\"message\":\"failed\"}}\n\n", .sequence = "x" },
    }) |case| {
        var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer memory.deinit();
        var capture: Capture = .{ .stop_on_delta = case.stop };
        var state: GenerateStream = .{
            .model = "fixture",
            .adaptations = &.{},
            .sink = .{ .context = &capture, .on_event = Capture.onEvent },
        };
        var reader = std.Io.Reader.fixed(case.frames);
        try GenerateStream.consume(&state, memory.allocator(), &reader, case.frames.len);
        try std.testing.expectEqualStrings(case.sequence, capture.events[0..capture.count]);
        try std.testing.expectEqual(case.reason, capture.finish);
        try std.testing.expectEqual(case.reason != null, capture.saw_usage);
        try std.testing.expectEqual(case.raw_usage, capture.raw_usage);
        if (case.total) |total| {
            try std.testing.expectEqual(.integer, std.meta.activeTag(capture.total));
            try std.testing.expectEqual(total, capture.total.integer);
        } else try std.testing.expect(capture.total == .null);
        try std.testing.expectEqual(case.stop, state.outcome == .stopped);
    }
}

test "Gemini SSE assembly includes deltas after the first finish frame" {
    const Continue = struct {
        fn onEvent(_: *anyopaque, _: Gemini.StreamEvent) Gemini.StreamSink.Action {
            return .proceed;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var assembler: lm_mod.OpenAI.OpenAIStream = .{ .arena = arena, .model = "fixture", .stop = &.{} };
    var state: GenerateStream = .{
        .model = "fixture",
        .adaptations = &.{},
        .sink = .{ .context = &assembler, .on_event = Continue.onEvent },
        .assembler = &assembler,
    };
    const frames =
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]},\"finishReason\":\"STOP\"}]}\n\n" ++
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\"}]}}]}\n\n" ++
        "data: {\"usageMetadata\":{\"totalTokenCount\":4}}\n\n";
    var reader = std.Io.Reader.fixed(frames);
    try GenerateStream.consume(&state, arena, &reader, frames.len);
    const result = assembler.result orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("ab", result.text.?);
    try std.testing.expectEqual(@as(i64, 4), result.metadata.usage.total_tokens.integer);
    try std.testing.expect(result.provider_data.?.root.object.contains("usageMetadata"));
}
