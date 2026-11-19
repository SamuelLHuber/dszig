//! Native Anthropic Messages generation.
const std = @import("std");
const lm_mod = @import("lm.zig");
const http = @import("http.zig");
const runtime = @import("runtime_value.zig");
const content_codec = @import("openai_content.zig");
const sdk = lm_mod.sdk;
const Allocator = std.mem.Allocator;
const Error = lm_mod.Error;
const Value = runtime.Value;
const Object = runtime.ObjectMap;
const ThinkingReplay = enum { signed, unsigned };
const ToolResultMedia = enum { native, images, reject };
fn str(bytes: []const u8) Value {
    return .{ .string = bytes };
}

pub const Anthropic = struct {
    client: *std.http.Client,
    base_url: []const u8 = "https://api.anthropic.com/v1",
    model: []const u8,
    api_key: ?[]const u8 = null,
    api_version: []const u8 = "2023-06-01",
    max_tokens: ?u32 = null,
    max_response_bytes: usize = 4 * 1024 * 1024,
    max_retries: u8 = 3,
    retry_base_ms: u32 = 500,
    timeout_ns: ?u64 = null,
    connect_timeout_ns: ?u64 = null,
    header_provider: ?http.HeaderProvider = null,
    /// Borrowed application instructions sent before the caller's system blocks.
    system_prefix: []const u8 = "",
    /// Borrowed allowed model prefixes for compatible endpoints. Null allows all.
    model_prefixes: ?[]const []const u8 = null,
    /// Set to reject on endpoints whose model sampling is fixed.
    sampling_params: enum { send, reject } = .send,
    /// Set to reject on endpoints that ignore disable_parallel_tool_use.
    parallel_tool_calls: enum { send, reject } = .send,
    /// Set to reject on endpoints that accept but ignore response schemas.
    structured_output: enum { send, reject } = .send,
    /// Endpoints with implicit caching can opt out of explicit cache markers.
    cache_control: enum { anthropic, none } = .anthropic,
    thinking_format: enum { anthropic, deepseek, adaptive, effort } = .anthropic,
    thinking_replay: ThinkingReplay = .signed,
    tool_result_media: ToolResultMedia = .native,
    /// Borrowed endpoint effort levels, in their declared order for diagnostics.
    reasoning_efforts: ?[]const sdk.ReasoningEffort = null,
    supports_reasoning: bool = false,
    supports_response_schema: bool = false,
    adaptations: enum { note, silent, refuse } = .note,

    pub fn lm(self: *Anthropic) lm_mod.LM {
        const supports_schema = self.supports_response_schema and self.structured_output == .send;
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = complete,
                .complete_stream = completeStream,
                .complete_canonical = canonical,
                .model_name = modelName,
                .clone = clone,
            },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = supports_schema,
            .supports_response_schema = supports_schema,
            .supports_function_calling = true,
        };
    }

    fn modelName(ptr: *anyopaque) []const u8 {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        return self.model;
    }

    fn clone(ptr: *anyopaque, arena: Allocator) anyerror!lm_mod.LM {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        const model = try arena.dupe(u8, self.model);
        errdefer arena.free(model);
        const base_url = try arena.dupe(u8, self.base_url);
        errdefer arena.free(base_url);
        const system_prefix = try arena.dupe(u8, self.system_prefix);
        errdefer arena.free(system_prefix);
        const model_prefixes = if (self.model_prefixes) |prefixes| try clonePrefixes(arena, prefixes) else null;
        errdefer if (model_prefixes) |prefixes| {
            for (prefixes) |prefix| arena.free(prefix);
            arena.free(prefixes);
        };
        const reasoning_efforts = if (self.reasoning_efforts) |levels|
            try arena.dupe(sdk.ReasoningEffort, levels)
        else
            null;
        errdefer if (reasoning_efforts) |levels| arena.free(levels);
        const copy = try arena.create(Anthropic);
        copy.* = self.*;
        copy.model = model;
        copy.base_url = base_url;
        copy.system_prefix = system_prefix;
        copy.model_prefixes = model_prefixes;
        copy.reasoning_efforts = reasoning_efforts;
        return copy.lm();
    }

    fn clonePrefixes(arena: Allocator, prefixes: []const []const u8) Allocator.Error![]const []const u8 {
        const result = try arena.alloc([]const u8, prefixes.len);
        errdefer arena.free(result);
        var copied: usize = 0;
        errdefer for (result[0..copied]) |prefix| arena.free(prefix);
        for (result, prefixes) |*entry, prefix| {
            entry.* = try arena.dupe(u8, prefix);
            copied += 1;
        }
        return result;
    }

    fn predictionRequest(
        self: *const Anthropic,
        arena: Allocator,
        request: lm_mod.Request,
    ) Error!*const sdk.Request {
        for (request.tools) |tool| if (tool.strict != null) return error.LMFailure;
        return lm_mod.OpenAI.canonicalPredictionRequest(
            arena,
            request,
            self.model,
            self.max_tokens,
            null,
        );
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) Error![]const lm_mod.Completion {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        const native = try self.predictionRequest(arena, request);
        defer native.release();
        const results = try arena.alloc(lm_mod.Completion, 1);
        results[0] = try canonical(ptr, arena, native);
        return results;
    }

    fn completeStream(
        ptr: *anyopaque,
        arena: Allocator,
        request: lm_mod.Request,
        sink: StreamSink,
    ) Error!lm_mod.Completion {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        const native = try self.predictionRequest(arena, request);
        defer native.release();
        var assembler: lm_mod.OpenAI.OpenAIStream = .{
            .arena = arena,
            .model = native.fields.model,
            .stop = &.{},
        };
        const outcome = try self.runStream(arena, arena, native, sink, &assembler);
        if (outcome == .stopped) return error.StreamStopped;
        var completion = assembler.result orelse return error.LMFailure;
        try lm_mod.publishResponse(arena, &completion, native.fields.model, .borrowed);
        return completion;
    }

    pub fn completeRequest(self: *Anthropic, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        return (try canonical(self, arena, request)).response.?;
    }

    pub const StreamEvent = lm_mod.OpenAI.StreamEvent;
    pub const StreamSink = lm_mod.OpenAI.StreamSink;
    pub const StreamOutcome = lm_mod.OpenAI.StreamOutcome;

    /// Events borrow their frame for the duration of the callback. Returning
    /// `.stop` closes the connection without waiting for another event.
    pub fn streamRequest(
        self: *Anthropic,
        backing: Allocator,
        request: *const sdk.Request,
        sink: StreamSink,
    ) Error!StreamOutcome {
        var scratch: std.heap.ArenaAllocator = .init(backing);
        defer scratch.deinit();
        return self.runStream(scratch.allocator(), backing, request, sink, null);
    }

    fn runStream(
        self: *Anthropic,
        arena: Allocator,
        backing: Allocator,
        request: *const sdk.Request,
        sink: StreamSink,
        assembler: ?*lm_mod.OpenAI.OpenAIStream,
    ) Error!StreamOutcome {
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try self.requestBody(arena, request, &records, true);
        var stream: MessageStream = .{
            .backing = backing,
            .model = request.fields.model,
            .adaptations = records.items,
            .sink = sink,
            .assembler = assembler,
        };
        var consumer: http.ResponseConsumer = .{
            .context = &stream,
            .consume = MessageStream.consume,
        };
        const response = try self.send(arena, request, body, &consumer);
        defer self.client.allocator.free(response.body);
        return stream.outcome;
    }

    fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) Error!lm_mod.Completion {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try self.requestBody(arena, request, &records, false);
        const response = try self.send(arena, request, body, null);
        defer self.client.allocator.free(response.body);
        const input = @import("json_repair.zig").parseJson(arena, response.body) catch |err|
            return codecError(err);
        const model = request.fields.model;
        var result = try parseResponse(arena, model, input, records.items);
        try lm_mod.OpenAI.foldJudgments(arena, &result, request);
        try lm_mod.publishResponse(arena, &result, model, .borrowed);
        return result;
    }

    fn send(
        self: *const Anthropic,
        arena: Allocator,
        request: *const sdk.Request,
        body: []const u8,
        consumer: ?*http.ResponseConsumer,
    ) Error!http.Response {
        const url = try std.fmt.allocPrint(arena, "{s}/messages", .{std.mem.trimEnd(u8, self.base_url, "/")});
        if (self.api_version.len == 0 or std.mem.indexOfAny(u8, self.api_version, "\r\n") != null)
            return error.LMFailure;
        var headers: [3]std.http.Header = undefined;
        headers[0] = .{ .name = "anthropic-version", .value = self.api_version };
        var headers_len: usize = 1;
        if (self.api_key) |key| {
            if (key.len == 0 or std.mem.indexOfAny(u8, key, "\r\n") != null) return error.LMFailure;
            headers[1] = .{ .name = "x-api-key", .value = key };
            headers_len = 2;
        }
        for (request.fields.tools) |tool| {
            if (tool != .builtin or !is(tool.builtin.name, "code_execution")) continue;
            headers[headers_len] = .{
                .name = "anthropic-beta",
                .value = "code-execution-2025-05-22",
            };
            headers_len += 1;
            break;
        }
        const transport: http.Transport = .{
            .client = self.client,
            .max_response_bytes = self.max_response_bytes,
            .max_retries = self.max_retries,
            .retry_base_ms = self.retry_base_ms,
            .timeout_ns = self.timeout_ns,
            .connect_timeout_ns = self.connect_timeout_ns,
            .header_provider = self.header_provider,
            .success = .ok,
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
        }) catch |err| return codecError(err);
    }

    fn requestBody(
        self: *const Anthropic,
        arena: Allocator,
        request: *const sdk.Request,
        records: *std.ArrayList(sdk.Adaptation),
        stream: bool,
    ) Error![]const u8 {
        const fields = &request.fields;
        if (self.model_prefixes) |prefixes| {
            for (prefixes) |prefix| {
                if (std.mem.startsWith(u8, fields.model, prefix)) break;
            } else return error.LMFailure;
        }
        const config = &fields.config;
        var body: Object = .empty;
        var messages = try encodeMessages(arena, fields.messages, .{
            .replay = self.thinking_replay,
            .tool_result_media = self.tool_result_media,
        });
        const system_text: ?[]const u8 = if (fields.system) |system| switch (system) {
            .text => |value| if (value.len != 0) value else null,
            .parts => |parts| if (parts.len != 0)
                try content_codec.partsText(arena, parts, null)
            else
                null,
        } else null;
        const marker = try self.cacheMarker(
            arena,
            config.cache,
            &messages.array,
            system_text != null,
            records,
        );
        const budget = try self.addReasoning(arena, &body, fields, records);
        try body.put(arena, "model", str(fields.model));
        try body.put(arena, "messages", messages);
        try body.put(arena, "stream", .{ .bool = stream });
        var max_tokens = config.max_tokens;
        if (max_tokens == .null) {
            const ceiling = outputCeiling(fields.model);
            max_tokens = .{ .integer = ceiling orelse 16384 };
            if (ceiling != null and budget != .null) {
                const math = @import("value.zig");
                const order = math.numericOrder(budget, max_tokens) catch return error.LMFailure;
                max_tokens = if (order == .lt)
                    math.numericDifference(arena, max_tokens, budget) catch |err|
                        return codecError(err)
                else
                    .{ .integer = 16384 };
            }
            try self.adapt(arena, records, "config.max_tokens", .defaulted, if (ceiling != null)
                "the Messages API requires max_tokens and none was set; the model's output ceiling was used"
            else
                "the Messages API requires max_tokens and none was set; 16384 was used (this server's ceiling is its own)", .null, max_tokens);
        }
        try body.put(arena, "max_tokens", if (budget == .null) max_tokens else try lm_mod.OpenAI.countSum(arena, budget, max_tokens));
        if (system_text) |rendered| {
            if (marker) |cache_control| {
                var block = try textBlock(arena, rendered);
                try block.object.put(arena, "cache_control", cache_control);
                var blocks: runtime.Array = .init(arena);
                try blocks.append(block);
                try body.put(arena, "system", .{ .array = blocks });
            } else {
                try body.put(arena, "system", str(rendered));
            }
        }
        try self.addSampling(arena, &body, config, records);
        try self.addTools(arena, &body, fields, records);
        if (config.response_format) |format| {
            try self.addFormat(arena, &body, config, format, records);
        }
        try self.addOutputOptions(arena, &body, config, records);
        // Extensions can replace system; the application prefix still comes first.
        if (self.system_prefix.len != 0) {
            try prependSystem(arena, &body, self.system_prefix);
        }
        var writer: std.Io.Writer.Allocating = .init(arena);
        @import("pyjson.zig").writeJson(&writer.writer, .{ .object = body }, .minified) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
        if (!std.unicode.utf8ValidateSlice(writer.written())) return error.LMFailure;
        return writer.written();
    }

    fn prependSystem(arena: Allocator, body: *Object, prefix: []const u8) Error!void {
        const existing = body.get("system") orelse .null;
        var blocks: runtime.Array = if (existing == .array) existing.array else .init(arena);
        try blocks.insert(0, try textBlock(arena, prefix));
        switch (existing) {
            .null, .array => {},
            .string => |rendered| try blocks.append(try textBlock(arena, rendered)),
            else => {
                var writer: std.Io.Writer.Allocating = .init(arena);
                @import("pyjson.zig").writeStr(&writer.writer, existing) catch |err|
                    return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
                try blocks.append(try textBlock(arena, writer.written()));
            },
        }
        try body.put(arena, "system", .{ .array = blocks });
    }

    fn cacheMarker(
        self: *const Anthropic,
        arena: Allocator,
        configured: ?sdk.CacheConfig,
        messages: *runtime.Array,
        has_system: bool,
        records: *std.ArrayList(sdk.Adaptation),
    ) Error!?Value {
        const cache = configured orelse return null;
        if (cache.key) |key| {
            try self.adapt(arena, records, "config.cache.key", .dropped, "the Messages API has no cache affinity key (OpenAI's prompt_cache_key); " ++
                "marks on blocks are its mechanism", str(key), .null);
        }
        if (cache.retention == .long and self.cache_control == .none) {
            try self.adapt(
                arena,
                records,
                "config.cache.retention",
                .dropped,
                "this server caches implicitly and has no cache-control TTL",
                str("long"),
                .null,
            );
        }
        if (cache.resource != null) return error.LMFailure;
        if (cache.mode == .off or self.cache_control == .none) return null;
        var last_block: ?*Object = null;
        if (messages.items.len != 0 and
            (cache.prefix == .history or cache.prefix_until_index != .null))
        {
            const last = messages.items.len - 1;
            const index = switch (cache.prefix_until_index) {
                .integer => |requested| @min(last, std.math.cast(usize, requested) orelse last),
                .null, .number_string => last,
                else => unreachable,
            };
            const blocks = &messages.items[index].object.getPtr("content").?.array;
            if (blocks.items.len != 0) {
                last_block = &blocks.items[blocks.items.len - 1].object;
            }
        }
        if (!has_system and last_block == null) return null;
        var object: Object = .empty;
        try object.put(arena, "type", str("ephemeral"));
        if (cache.retention == .long) try object.put(arena, "ttl", str("1h"));
        const marker: Value = .{ .object = object };
        if (last_block) |block| try block.put(arena, "cache_control", marker);
        return marker;
    }

    fn addReasoning(
        self: *const Anthropic,
        arena: Allocator,
        body: *Object,
        fields: *const sdk.RequestFields,
        records: *std.ArrayList(sdk.Adaptation),
    ) Error!Value {
        const reasoning = fields.config.reasoning orelse return .null;
        const asked = reasoning.effort orelse return error.LMFailure;
        if (asked == .off) {
            if (self.thinking_format != .anthropic) {
                try putThinking(arena, body, "disabled", false, .null);
            }
            return .null;
        }
        const effort = try self.resolveEffort(arena, records, asked);
        if (reasoning.summary) |summary| if (summary != .auto) {
            try self.adapt(arena, records, "config.reasoning.summary", .substituted, "the Messages API has no summary detail levels; it returns one summary of the " ++
                "thinking, which is 'auto'", str(@tagName(summary)), str("auto"));
        };
        const adaptive = self.thinking_format != .anthropic or adaptiveModel(fields.model);
        const display = self.thinking_format == .anthropic and reasoning.summary != null;
        var budget = reasoning.thinking_budget;
        if (adaptive) {
            if (budget != .null) {
                const reason = switch (self.thinking_format) {
                    .deepseek => "this server ignores budget_tokens; effort is the dial",
                    .adaptive => "this server accepts budget_tokens without translating it; " ++
                        "effort is the dial (protocols--messages.md)",
                    .anthropic, .effort => try std.fmt.allocPrint(
                        arena,
                        "{s} takes thinking.type 'adaptive' with output_config.effort; " ++
                            "budget_tokens is rejected by the API (live 2026-09-02). " ++
                            "Thinking is bounded only by max_tokens, which covers thinking and " ++
                            "answer together: lower the effort or raise max_tokens",
                        .{fields.model},
                    ),
                };
                try self.adapt(arena, records, "config.reasoning.thinking_budget", .dropped, reason, budget, .null);
                budget = .null;
            }
            const clamp_minimal = effort == .minimal and self.thinking_format == .anthropic;
            if (clamp_minimal) {
                try self.adapt(arena, records, "config.reasoning.effort", .clamped, "this model class has no 'minimal' level (output_config.effort is " ++
                    "low|medium|high|xhigh|max); 'low' is the floor", str("minimal"), str("low"));
            }
            var output: Object = .empty;
            try output.put(arena, "effort", str(if (clamp_minimal) "low" else @tagName(effort)));
            try body.put(arena, "output_config", .{ .object = output });
            if (self.thinking_format != .effort) {
                const kind = if (self.thinking_format == .deepseek) "enabled" else "adaptive";
                try putThinking(arena, body, kind, display, .null);
            }
        } else {
            if (budget == .null) budget = .{ .integer = switch (effort) {
                .off => unreachable,
                .minimal => 1024,
                .low => 2048,
                .medium => 8192,
                .high => 16384,
                .xhigh => 24576,
                .max => 32768,
            } };
            try putThinking(arena, body, "enabled", display, budget);
        }
        return budget;
    }

    fn resolveEffort(
        self: *const Anthropic,
        arena: Allocator,
        records: *std.ArrayList(sdk.Adaptation),
        asked: sdk.ReasoningEffort,
    ) Error!sdk.ReasoningEffort {
        const levels = self.reasoning_efforts orelse return asked;
        const nearest = try @import("openai_request.zig").nearestEffort(asked, levels);
        if (nearest == asked) return asked;
        var reason: std.Io.Writer.Allocating = .init(arena);
        const writer = &reason.writer;
        writer.print("this server has no '{s}' level (it accepts ", .{@tagName(asked)}) catch
            return error.OutOfMemory;
        for (levels, 0..) |level, index| {
            if (index != 0) writer.writeAll(", ") catch return error.OutOfMemory;
            writer.writeAll(@tagName(level)) catch return error.OutOfMemory;
        }
        writer.writeAll(") and would have accepted the word silently") catch
            return error.OutOfMemory;
        try self.adapt(
            arena,
            records,
            "config.reasoning.effort",
            .clamped,
            reason.written(),
            str(@tagName(asked)),
            str(@tagName(nearest)),
        );
        return nearest;
    }

    fn putThinking(arena: Allocator, body: *Object, kind: []const u8, display: bool, budget: Value) Error!void {
        var thinking: Object = .empty;
        try thinking.put(arena, "type", str(kind));
        if (display) try thinking.put(arena, "display", str("summarized"));
        if (budget != .null) try thinking.put(arena, "budget_tokens", budget);
        try body.put(arena, "thinking", .{ .object = thinking });
    }

    fn addSampling(self: *const Anthropic, arena: Allocator, body: *Object, config: *const sdk.Config, records: *std.ArrayList(sdk.Adaptation)) Error!void {
        if (self.sampling_params == .reject) {
            inline for (.{ "temperature", "top_p", "top_k" }) |name| {
                const asked: Value = if (comptime std.mem.eql(u8, name, "top_k"))
                    config.top_k
                else if (@field(config, name)) |number|
                    .{ .float = number }
                else
                    .null;
                if (asked != .null) {
                    try self.adapt(
                        arena,
                        records,
                        "config." ++ name,
                        .dropped,
                        "this server ignores sampling parameters (the model's sampling is fixed)",
                        asked,
                        .null,
                    );
                }
            }
        }
        inline for (.{ "seed", "frequency_penalty", "presence_penalty" }) |name| {
            const asked: Value = if (comptime std.mem.eql(u8, name, "seed")) config.seed else if (@field(config, name)) |num| .{ .float = num } else .null;
            if (asked != .null) try self.adapt(arena, records, "config." ++ name, .dropped, "the Messages API has no " ++ name ++ " field", asked, .null);
        }
        if (self.sampling_params == .send) {
            if (config.temperature) |temp| {
                const applied: Value = .{ .float = @min(temp, 1) };
                if (temp > 1) try self.adapt(arena, records, "config.temperature", .clamped, "the Messages API accepts temperature in [0, 1]; the canonical range is [0, 2]", .{ .float = temp }, applied);
                try body.put(arena, "temperature", applied);
            }
            if (config.top_p) |top_p| try body.put(arena, "top_p", .{ .float = top_p });
            if (config.top_k != .null) try body.put(arena, "top_k", config.top_k);
        }
        if (config.stop.len != 0) try body.put(arena, "stop_sequences", try strings(arena, config.stop));
    }

    fn addTools(self: *const Anthropic, arena: Allocator, body: *Object, fields: *const sdk.RequestFields, records: *std.ArrayList(sdk.Adaptation)) Error!void {
        if (fields.tools.len != 0) {
            var tools: runtime.Array = .init(arena);
            for (fields.tools) |tool| {
                const name = switch (tool) {
                    .function => |function| function.name,
                    .builtin => |builtin| builtin.name,
                };
                if (fields.config.tool_choice) |choice| {
                    const force = choice.mode == .required and choice.allowed.len == 1;
                    if (!force and choice.mode != .none and choice.allowed.len != 0 and
                        !contains(choice.allowed, name)) continue;
                }
                const item = try encodeTool(arena, tool, name);
                try tools.append(.{ .object = item });
            }
            if (tools.items.len != fields.tools.len) {
                var names = try runtime.Array.initCapacity(arena, tools.items.len);
                for (tools.items) |item| names.appendAssumeCapacity(item.object.get("name").?);
                try self.adapt(
                    arena,
                    records,
                    "config.tool_choice.allowed",
                    .client_side,
                    "the Messages API cannot restrict to a subset of the declared tools; only the allowed tools were sent, which is what the allowlist means",
                    try strings(arena, fields.config.tool_choice.?.allowed),
                    .{ .array = names },
                );
            }
            try body.put(arena, "tools", .{ .array = tools });
        }
        if (fields.config.tool_choice) |choice| {
            const force = choice.mode == .required and choice.allowed.len == 1;
            var row: Object = .empty;
            try row.put(arena, "type", str(switch (choice.mode) {
                .auto => "auto",
                .required => if (force) "tool" else "any",
                .none => "none",
            }));
            if (force) try row.put(arena, "name", str(choice.allowed[0]));
            if (choice.parallel) |parallel| {
                if (self.parallel_tool_calls == .reject) {
                    try self.adapt(
                        arena,
                        records,
                        "config.tool_choice.parallel",
                        .dropped,
                        "this server accepts disable_parallel_tool_use and does not apply it " ++
                            "(guide--anthropic-api.md); the model may return several calls",
                        .{ .bool = parallel },
                        .null,
                    );
                } else if (!parallel and choice.mode != .none) {
                    try row.put(arena, "disable_parallel_tool_use", .{ .bool = true });
                }
            }
            try body.put(arena, "tool_choice", .{ .object = row });
        }
    }

    fn addFormat(
        self: *const Anthropic,
        arena: Allocator,
        body: *Object,
        config: *const sdk.Config,
        format: sdk.JsonValue,
        records: *std.ArrayList(sdk.Adaptation),
    ) Error!void {
        const value = try project(arena, format);
        if (self.structured_output == .reject) {
            return self.adapt(
                arena,
                records,
                "config.response_format",
                .dropped,
                "this server accepts output_config.format and does not apply it; " ++
                    "describe the shape in the prompt",
                value,
                .null,
            );
        }
        if (!std.mem.eql(u8, text(get(value, "type")), "json_schema")) return error.LMFailure;
        var schema = get(value, "schema");
        if (schema != .object) return error.LMFailure;
        const found = try @import("typesafe.zig").responseJudgments(arena, schema);
        for (found) |judgment| {
            if (judgment.kind == .box) continue;
            switch (config.probabilities orelse .off) {
                .off => {},
                .required => return error.LMFailure,
                .if_available => try self.adapt(
                    arena,
                    records,
                    "config.probabilities",
                    .dropped,
                    "this wire cannot measure a distribution over the declared keys; the answer carries the pick only",
                    str("if_available"),
                    .null,
                ),
            }
            break;
        }
        try encodeJudgmentSchema(arena, &schema, found);
        var wire_format: Object = .empty;
        try wire_format.put(arena, "type", str("json_schema"));
        try wire_format.put(arena, "schema", schema);
        const existing = body.get("output_config");
        var output: Object = if (existing) |item| item.object else .empty;
        try output.put(arena, "format", .{ .object = wire_format });
        try body.put(arena, "output_config", .{ .object = output });
    }

    fn addOutputOptions(
        self: *const Anthropic,
        arena: Allocator,
        body: *Object,
        config: *const sdk.Config,
        records: *std.ArrayList(sdk.Adaptation),
    ) Error!void {
        if (config.service_tier) |tier| try body.put(arena, "service_tier", str(tier));
        if (config.user_id) |user| {
            var metadata: Object = .empty;
            try metadata.put(arena, "user_id", str(user));
            try body.put(arena, "metadata", .{ .object = metadata });
        }
        if (config.store) |store| try self.adapt(arena, records, "config.store", if (store) .dropped else .satisfied, if (store)
            "the Messages API has no stored-response object to opt into (OpenAI and Gemini carry `store`)"
        else
            "the Messages API has no stored-response object to opt out of; nothing retrievable is kept", .{ .bool = store }, .null);
        if (config.logprobs != .null) try self.adapt(arena, records, "config.logprobs", .dropped, "the Messages API does not expose token log probabilities (OpenAI and Gemini carry them); Response.logprobs will be absent", config.logprobs, .null);
        if (config.extensions) |extensions| {
            const extra = try project(arena, extensions);
            if (extra != .object) return error.LMFailure;
            var iterator = extra.object.iterator();
            while (iterator.next()) |entry| {
                const key = entry.key_ptr.*;
                if (std.mem.eql(u8, key, "prompt_caching")) continue;
                // The HTTP consumer and the requested framing must agree.
                if (std.mem.eql(u8, key, "stream")) {
                    if (entry.value_ptr.* != .bool or
                        entry.value_ptr.bool != body.get("stream").?.bool) return error.LMFailure;
                }
                try body.put(arena, key, entry.value_ptr.*);
            }
        }
    }

    fn adapt(self: *const Anthropic, arena: Allocator, records: *std.ArrayList(sdk.Adaptation), field: []const u8, action: sdk.AdaptationAction, reason: []const u8, asked: Value, applied: Value) Error!void {
        if (self.adaptations == .silent) return;
        if (self.adaptations == .refuse and action != .defaulted and action != .satisfied)
            return error.LMFailure;
        try records.append(arena, .{
            .field = field,
            .action = action,
            .reason = reason,
            .asked = try payload(arena, asked),
            .applied = try payload(arena, applied),
        });
    }
};

const MessageStream = struct {
    backing: Allocator,
    model: []const u8,
    adaptations: []const sdk.Adaptation,
    sink: Anthropic.StreamSink,
    assembler: ?*lm_mod.OpenAI.OpenAIStream = null,
    outcome: Anthropic.StreamOutcome = .exhausted,
    started: bool = false,
    saw_end: bool = false,
    finish: ?sdk.FinishReason = null,
    usage_frame: bool = false,

    fn consume(context: *anyopaque, arena: Allocator, reader: *std.Io.Reader, limit: usize) !void {
        const self: *MessageStream = @ptrCast(@alignCast(context));
        var events = @import("sse.zig").Iterator.init(arena, reader, limit);
        defer events.deinit();
        var frame: std.heap.ArenaAllocator = .init(self.backing);
        defer frame.deinit();
        // Typed assembly retains parsed leaves; raw observation reuses frame storage.
        const temp = if (self.assembler != null) self.backing else frame.allocator();
        var terminal: std.ArrayList(u8) = .empty;
        defer terminal.deinit(self.backing);
        var retained_terminal: Value = .null;
        const parse = @import("json_repair.zig").parseJson;
        while (try events.next()) |data| {
            if (data.len == 0) continue;
            const value = try parse(temp, data);
            if (value != .object) return error.LMFailure;
            const kind = text(get(value, "type"));
            if (is(kind, "message_delta")) {
                const stop = nonempty(get(get(value, "delta"), "stop_reason"));
                const usage = get(value, "usage");
                const has_usage = usage == .object and usage.object.count() != 0;
                if (stop != null or has_usage) self.saw_end = true;
                if (stop) |reason| self.finish = finishReason(reason, false);
                if (has_usage or (stop != null and !self.usage_frame)) {
                    if (self.assembler != null) {
                        retained_terminal = value;
                    } else {
                        terminal.clearRetainingCapacity();
                        try terminal.appendSlice(self.backing, data);
                    }
                    self.usage_frame = has_usage;
                }
            } else if (is(kind, "message_stop")) {
                self.saw_end = true;
            } else {
                try self.dispatch(temp, value);
            }
            if (self.outcome == .stopped) return;
            _ = frame.reset(.retain_capacity);
        }
        if (!self.saw_end) return;
        const final = if (terminal.items.len == 0) retained_terminal else try parse(temp, terminal.items);
        try self.start(null, self.model);
        try self.emit(.{ .end = .{
            .finish_reason = self.finish,
            .usage = if (self.usage_frame) try readUsage(temp, get(final, "usage")) else null,
            .provider_data = if (final != .null) try payload(temp, final) else null,
        } });
    }

    fn emit(self: *MessageStream, event: Anthropic.StreamEvent) Error!void {
        if (self.outcome == .stopped) return;
        if (self.sink.on_event(self.sink.context, event) == .stop) {
            self.outcome = .stopped;
            return;
        }
        if (self.assembler) |assembler| try assembler.acceptRetainedEvent(event);
    }

    fn start(self: *MessageStream, id: ?[]const u8, model: []const u8) Error!void {
        if (self.started) return;
        self.started = true;
        try self.emit(.{ .start = .{ .id = id, .model = model, .adaptations = self.adaptations } });
    }

    fn dispatch(self: *MessageStream, arena: Allocator, value: Value) Error!void {
        const kind = text(get(value, "type"));
        if (is(kind, "message_start")) {
            const message = get(value, "message");
            try self.start(nonempty(get(message, "id")), nonempty(get(message, "model")) orelse self.model);
        } else if (is(kind, "content_block_start") or is(kind, "content_block_delta")) {
            try self.content(arena, value, is(kind, "content_block_start"));
        } else if (is(kind, "error")) {
            const detail = get(value, "error");
            const code = nonempty(get(detail, "type")) orelse nonempty(get(detail, "code")) orelse
                nonempty(get(value, "code")) orelse nonempty(get(value, "error_type")) orelse "provider";
            try self.emit(.{ .provider_error = .{
                .code = providerError(code),
                .provider_code = code,
                .message = nonempty(get(detail, "message")) orelse nonempty(get(value, "message")) orelse code,
            } });
        }
    }

    fn content(self: *MessageStream, arena: Allocator, value: Value, block_start: bool) Error!void {
        const item = get(value, if (block_start) "content_block" else "delta");
        const kind = text(get(item, "type"));
        const index_value = get(value, "index");
        var index_buffer: [24]u8 = undefined;
        const index = switch (index_value) {
            .null => "0",
            .integer => |n| if (n >= 0)
                std.fmt.bufPrint(&index_buffer, "{d}", .{n}) catch unreachable
            else
                return error.LMFailure,
            else => return error.LMFailure,
        };
        const data: @import("openai_stream.zig").DeltaData = if (block_start) start_data: {
            if (is(kind, "tool_use")) {
                const input = get(item, "input");
                var json: std.Io.Writer.Allocating = .init(arena);
                if (input == .object and input.object.count() != 0)
                    @import("pyjson.zig").writeJson(&json.writer, input, .minified) catch |err|
                        return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
                break :start_data .{ .tool_call = .{
                    .input = nonempty(input) orelse json.written(),
                    .id = nonempty(get(item, "id")),
                    .name = nonempty(get(item, "name")),
                } };
            }
            if (!is(kind, "redacted_thinking") or get(item, "data") == .null) return;
            const thinking = try parseThinking(arena, item, true);
            try self.start(null, self.model);
            try self.emit(.{ .delta = .{ .part_index = index, .data = .{ .thinking = "" } } });
            break :start_data .{ .continuation = thinking.continuation[0] };
        } else if (is(kind, "text_delta"))
            .{ .text = .{ .text = text(get(item, "text")) } }
        else if (is(kind, "thinking_delta"))
            .{ .thinking = text(get(item, "thinking")) }
        else if (is(kind, "input_json_delta"))
            .{ .tool_call = .{ .input = text(get(item, "partial_json")), .id = null, .name = null } }
        else if (is(kind, "signature_delta") and nonempty(get(item, "signature")) != null)
            .{ .continuation = (try parseThinking(arena, item, false)).continuation[0] }
        else if (is(kind, "citation_delta") or is(kind, "citations_delta")) citation: {
            const nested = get(item, "citation");
            const source = if (nested == .object) nested else item;
            break :citation .{ .citation = .{
                .text = nonempty(get(source, "cited_text")) orelse nonempty(get(source, "text")),
                .title = nonempty(get(source, "title")),
                .url = nonempty(get(source, "url")),
            } };
        } else return;
        try self.start(null, self.model);
        try self.emit(.{ .delta = .{ .part_index = index, .data = data } });
    }
};

fn providerError(code: []const u8) @FieldType(@import("openai_stream.zig").ErrorDetail, "code") {
    if (is(code, "overloaded_error") or is(code, "api_error")) return .server;
    if (is(code, "rate_limit_error")) return .rate_limit;
    if (is(code, "invalid_request_error") or is(code, "request_too_large") or
        is(code, "not_found_error") or is(code, "resource_not_found_error")) return .invalid_request;
    if (is(code, "authentication_error") or is(code, "permission_error") or
        is(code, "invalid_authentication_error")) return .auth;
    if (is(code, "DeploymentNotFound")) return .unsupported_model;
    if (is(code, "timeout_error")) return .timeout;
    if (is(code, "billing_error")) return .billing;
    return .provider;
}

fn is(value: []const u8, expected: []const u8) bool {
    return std.mem.eql(u8, value, expected);
}

fn nonempty(value: Value) ?[]const u8 {
    const bytes = text(value);
    return if (bytes.len == 0) null else bytes;
}

fn outputCeiling(model: []const u8) ?i64 {
    const rules = .{
        .{ "claude-3-haiku", 4096 },     .{ "claude-3-opus", 4096 },    .{ "claude-3-sonnet", 4096 },
        .{ "claude-3-5-", 8192 },        .{ "claude-3.5-", 8192 },      .{ "claude-haiku-4-5", 64000 },
        .{ "claude-sonnet-4-5", 64000 }, .{ "claude-opus-4-5", 64000 }, .{ "claude", 128000 },
    };
    inline for (rules) |rule| {
        if (std.ascii.findIgnoreCase(model, rule[0]) != null) return rule[1];
    }
    return null;
}

fn adaptiveModel(model: []const u8) bool {
    for ([_][]const u8{
        "sonnet-5", "opus-5", "sonnet-4-6", "opus-4-6", "opus-4-7",
        "opus-4-8", "fable",  "mythos",     "haiku-5",
    }) |marker| {
        if (std.ascii.findIgnoreCase(model, marker) != null) return true;
    }
    return false;
}

const MessagePolicy = struct {
    replay: ThinkingReplay = .signed,
    tool_result_media: ToolResultMedia = .native,
};

fn encodeMessages(arena: Allocator, messages: []const sdk.Message, policy: MessagePolicy) Error!Value {
    var rows: runtime.Array = .init(arena);
    for (messages) |message| {
        var blocks: runtime.Array = .init(arena);
        if (message.role == .developer) {
            const rendered = try content_codec.partsText(arena, message.parts, "[developer]");
            try blocks.append(try textBlock(arena, rendered));
        } else {
            for (message.parts) |part| try blocks.append(try encodeBlock(arena, part, policy));
        }
        var row: Object = .empty;
        try row.put(arena, "role", str(if (message.role == .assistant) "assistant" else "user"));
        try row.put(arena, "content", .{ .array = blocks });
        try rows.append(.{ .object = row });
    }
    return .{ .array = rows };
}

fn encodeBlock(arena: Allocator, part: sdk.Part, policy: MessagePolicy) Error!Value {
    var block: Object = .empty;
    switch (part) {
        .text => |item| return textBlock(arena, item.text),
        .data => |item| return textBlock(arena, try content_codec.dataText(arena, item.value)),
        .image => |item| return encodeMedia(arena, "image", &item),
        .document => |item| return encodeMedia(arena, "document", &item),
        .thinking => |item| return encodeThinking(arena, &item, policy.replay),
        .tool_call => |item| {
            try block.put(arena, "type", str("tool_use"));
            try block.put(arena, "id", str(item.id));
            try block.put(arena, "name", str(item.name));
            try block.put(arena, "input", try project(arena, item.input));
        },
        .tool_result => |item| {
            try block.put(arena, "type", str("tool_result"));
            try block.put(arena, "tool_use_id", str(item.id));
            if (item.content.len == 1 and item.content[0] == .text) {
                try block.put(arena, "content", str(item.content[0].text.text));
            } else if (item.content.len == 1 and item.content[0] == .data) {
                const rendered = try content_codec.dataText(arena, item.content[0].data.value);
                try block.put(arena, "content", str(rendered));
            } else if (item.content.len != 0) {
                var blocks: runtime.Array = .init(arena);
                try blocks.ensureTotalCapacity(item.content.len);
                for (item.content) |child| {
                    const content = switch (child) {
                        .text => |text_part| try textBlock(arena, text_part.text),
                        .data => |data| try textBlock(arena, try content_codec.dataText(arena, data.value)),
                        .image => |media| if (policy.tool_result_media != .reject)
                            try encodeMedia(arena, "image", &media)
                        else
                            return error.LMFailure,
                        .document => |media| if (policy.tool_result_media == .native)
                            try encodeMedia(arena, "document", &media)
                        else
                            return error.LMFailure,
                        else => return error.LMFailure,
                    };
                    blocks.appendAssumeCapacity(content);
                }
                try block.put(arena, "content", .{ .array = blocks });
            }
            if (item.is_error) try block.put(arena, "is_error", .{ .bool = true });
        },
        else => return error.LMFailure,
    }
    return .{ .object = block };
}

fn encodeMedia(arena: Allocator, kind: []const u8, media: *const sdk.MediaPart) Error!Value {
    var source: Object = .empty;
    if (media.url) |url| {
        try source.put(arena, "type", str("url"));
        try source.put(arena, "url", str(url));
    } else if (media.file_id) |file_id| {
        try source.put(arena, "type", str("file"));
        try source.put(arena, "file_id", str(file_id));
    } else if (media.data) |data| {
        try source.put(arena, "type", str("base64"));
        try source.put(arena, "media_type", str(media.media_type));
        try source.put(arena, "data", str(data));
    } else {
        // Reading a local path must remain an explicit caller operation.
        return error.LMFailure;
    }
    var block: Object = .empty;
    try block.put(arena, "type", str(kind));
    try block.put(arena, "source", .{ .object = source });
    return .{ .object = block };
}

fn encodeThinking(arena: Allocator, item: *const sdk.ThinkingPart, replay: ThinkingReplay) Error!Value {
    var block: Object = .empty;
    for (item.continuation) |state| {
        if (!std.mem.eql(u8, state.provider, "anthropic")) continue;
        if (!std.mem.eql(u8, state.kind, "redacted_thinking")) continue;
        const data = try project(arena, state.data);
        if (data != .object) return error.LMFailure;
        block = data.object;
        try block.put(arena, "type", str("redacted_thinking"));
        return .{ .object = block };
    }
    for (item.continuation) |state| {
        if (!std.mem.eql(u8, state.provider, "anthropic")) continue;
        if (!std.mem.eql(u8, state.kind, "thinking_signature")) continue;
        const signature = text(get(try project(arena, state.data), "signature"));
        if (signature.len == 0) break;
        try block.put(arena, "type", str("thinking"));
        try block.put(arena, "thinking", str(item.text));
        try block.put(arena, "signature", str(signature));
        return .{ .object = block };
    }
    if (replay == .unsigned and item.text.len != 0) {
        try block.put(arena, "type", str("thinking"));
        try block.put(arena, "thinking", str(item.text));
        return .{ .object = block };
    }
    return textBlock(arena, item.text);
}

fn parseThinking(arena: Allocator, block: Value, redacted: bool) Error!sdk.ThinkingPart {
    const key = if (redacted) "data" else "signature";
    const value = get(block, key);
    var thinking = text(get(block, "thinking"));
    if (thinking.len == 0) thinking = text(get(block, "text"));
    var result: sdk.ThinkingPart = .{ .text = if (redacted) "" else thinking };
    if (if (redacted) value != .null else text(value).len != 0) {
        var data: Object = .empty;
        try data.put(arena, key, value);
        const state = try arena.alloc(sdk.Continuation, 1);
        state[0] = .{
            .provider = "anthropic",
            .kind = if (redacted) "redacted_thinking" else "thinking_signature",
            .data = try payload(arena, .{ .object = data }),
        };
        result.continuation = state;
    }
    return result;
}

fn encodeTool(arena: Allocator, tool: sdk.RequestTool, name: []const u8) Error!Object {
    var item: Object = .empty;
    switch (tool) {
        .function => |function| {
            try item.put(arena, "name", str(name));
            if (function.description) |description| if (description.len != 0) {
                try item.put(arena, "description", str(description));
            };
            try item.put(arena, "input_schema", try project(arena, function.parameters.?));
        },
        .builtin => |builtin| {
            const kind = if (is(name, "web_search")) "web_search_20250305" else if (is(name, "code_execution")) "code_execution_20250522" else name;
            try item.put(arena, "type", str(kind));
            try item.put(arena, "name", str(name));
            if (builtin.config) |config| {
                const value = try project(arena, config);
                if (value != .object) return error.LMFailure;
                for (value.object.keys(), value.object.values()) |key, field| {
                    try item.put(arena, key, field);
                }
            }
        },
    }
    return item;
}

fn citationField(arena: Allocator, item: Value, comptime keys: []const []const u8) Error!?[]const u8 {
    inline for (keys) |key| {
        const value = try lm_mod.OpenAI.responseText(arena, get(item, key));
        if (value.len != 0) return value;
    }
    return null;
}

fn encodeJudgmentSchema(
    arena: Allocator,
    schema: *Value,
    found: []const @import("typesafe.zig").Judgment,
) Error!void {
    if (found.len == 0) return;
    const properties = &schema.object.getPtr("properties").?.object;
    for (found) |judgment| {
        const property = &properties.getPtr(judgment.name).?.object;
        const kind = property.get("type") orelse continue;
        const branches = property.getPtr("anyOf") orelse continue;
        if (branches.* != .array) continue;
        for (branches.array.items) |*branch| {
            if (branch.* != .object) return error.LMFailure;
            if (!branch.object.contains("type")) try branch.object.put(arena, "type", kind);
        }
        // The wire rejects sibling type + anyOf on judgment properties.
        std.debug.assert(property.contains("type"));
        _ = property.orderedRemove("type");
    }
}

fn appendCitations(
    arena: Allocator,
    parts: *std.ArrayList(sdk.Part),
    citations: *std.ArrayList(sdk.CitationPart),
    items: Value,
) Error!void {
    if (items != .array) return;
    for (items.array.items) |item| {
        if (item != .object) continue;
        const citation: sdk.CitationPart = .{
            .url = try citationField(arena, item, &.{ "url", "uri" }),
            .title = try citationField(arena, item, &.{ "title", "document_title", "source_title" }),
            .text = try citationField(arena, item, &.{ "cited_text", "text", "quote" }),
        };
        if (citation.url == null and citation.title == null and citation.text == null) continue;
        try parts.append(arena, .{ .citation = citation });
        try citations.append(arena, citation);
    }
}

fn parseResponse(arena: Allocator, model: []const u8, input: Value, records: []const sdk.Adaptation) Error!lm_mod.Completion {
    if (input != .object) return error.LMFailure;
    var parts: std.ArrayList(sdk.Part) = .empty;
    var calls: std.ArrayList(lm_mod.ToolCall) = .empty;
    var citations: std.ArrayList(sdk.CitationPart) = .empty;
    var visible: std.Io.Writer.Allocating = .init(arena);
    var reasoning: std.Io.Writer.Allocating = .init(arena);
    const content = get(input, "content");
    if (content != .null and content != .array) return error.LMFailure;
    for (if (content == .array) content.array.items else &.{}) |block| {
        const kind = text(get(block, "type"));
        if (std.mem.eql(u8, kind, "text")) {
            const chunk = try lm_mod.OpenAI.responseText(arena, get(block, "text"));
            visible.writer.writeAll(chunk) catch return error.OutOfMemory;
            try parts.append(arena, .{ .text = .{ .text = chunk } });
            try appendCitations(arena, &parts, &citations, get(block, "citations"));
        } else if (std.mem.eql(u8, kind, "tool_use")) {
            const name = text(get(block, "name"));
            if (name.len == 0) return error.LMFailure;
            var id = text(get(block, "id"));
            if (id.len == 0) id = try std.fmt.allocPrint(arena, "tool_{d}", .{parts.items.len});
            const args = get(block, "input");
            const object: Value = if (args == .object) args else .{ .object = .empty };
            const encoded = try payload(arena, object);
            var json: std.Io.Writer.Allocating = .init(arena);
            @import("pyjson.zig").writeJson(&json.writer, object, .minified) catch |err|
                return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
            try parts.append(arena, .{ .tool_call = .{ .id = id, .name = name, .input = encoded } });
            try calls.append(arena, .{ .id = id, .name = name, .arguments = json.written() });
        } else if (std.mem.eql(u8, kind, "thinking") or
            std.mem.eql(u8, kind, "redacted_thinking"))
        {
            const item = try parseThinking(arena, block, std.mem.eql(u8, kind, "redacted_thinking"));
            reasoning.writer.writeAll(item.text) catch return error.OutOfMemory;
            try parts.append(arena, .{ .thinking = item });
        } else if (!is(kind, "server_tool_use") and !is(kind, "web_search_tool_result") and
            !is(kind, "code_execution_tool_result"))
        {
            return error.LMFailure;
        }
    }
    if (parts.items.len == 0) try parts.append(arena, .{ .text = .{} });
    const id = text(get(input, "id"));
    const model_id = text(get(input, "model"));
    return .{
        .text = if (visible.written().len == 0 and calls.items.len != 0) null else visible.written(),
        .reasoning = if (reasoning.written().len == 0) null else reasoning.written(),
        .tool_calls = calls.items,
        .citations = citations.items,
        .message = .{ .role = .assistant, .parts = parts.items },
        .provider_data = try payload(arena, input),
        .adaptations = records,
        .metadata = .{
            .id = if (id.len != 0) id else null,
            .model = if (model_id.len != 0) model_id else model,
            .usage = try readUsage(arena, get(input, "usage")),
            .finish_reason = finishReason(text(get(input, "stop_reason")), calls.items.len != 0),
        },
    };
}

fn finishReason(reason: []const u8, tool: bool) sdk.FinishReason {
    if (tool) return .tool_call;
    for ([_][]const u8{ "max_tokens", "model_context_window_exceeded" }) |candidate|
        if (std.ascii.eqlIgnoreCase(reason, candidate)) return .length;
    for ([_][]const u8{ "tool_use", "pause_turn" }) |candidate|
        if (std.ascii.eqlIgnoreCase(reason, candidate)) return .tool_call;
    for ([_][]const u8{ "refusal", "safety", "content_filter" }) |candidate|
        if (std.ascii.eqlIgnoreCase(reason, candidate)) return .content_filter;
    return .stop;
}

fn readUsage(arena: Allocator, usage: Value) Error!sdk.Usage {
    const common = lm_mod.OpenAI;
    const input = try common.wireCount(arena, get(usage, "input_tokens"));
    const read = try common.wireCount(arena, get(usage, "cache_read_input_tokens"));
    const write = try common.wireCount(arena, get(usage, "cache_creation_input_tokens"));
    const whole = if (input == .null or read == .null or write == .null) .null else try common.countSum(arena, try common.countSum(arena, input, read), write);
    const details = get(usage, "output_tokens_details");
    return common.settleUsage(arena, .{
        .input_tokens = whole,
        .cache_read_tokens = read,
        .cache_write_tokens = write,
        .output_tokens = try common.wireCount(arena, get(usage, "output_tokens")),
        .total_tokens = try common.wireCount(arena, get(usage, "total_tokens")),
        .reasoning_tokens = try common.wireCount(arena, get(details, "thinking_tokens")),
    });
}

fn strings(arena: Allocator, values: []const []const u8) Error!Value {
    var list: runtime.Array = .init(arena);
    try list.ensureTotalCapacity(values.len);
    for (values) |value| try list.append(str(value));
    return .{ .array = list };
}

fn textBlock(arena: Allocator, value: []const u8) Error!Value {
    var block: Object = .empty;
    try block.put(arena, "type", str("text"));
    try block.put(arena, "text", str(value));
    return .{ .object = block };
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}

fn get(value: Value, key: []const u8) Value {
    return if (value == .object) value.object.get(key) orelse .null else .null;
}

fn text(value: Value) []const u8 {
    return if (value == .string) value.string else "";
}

fn project(arena: Allocator, value: sdk.JsonValue) Error!Value {
    return value.toValue(arena) catch |err| return codecError(err);
}

fn payload(arena: Allocator, value: Value) Error!sdk.PayloadValue {
    return sdk.PayloadValue.fromBorrowedValue(arena, value) catch |err| return codecError(err);
}

fn codecError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
}

test "Anthropic typed stream assembles distinct indexes and requires a terminal event" {
    const deltas =
        \\data: {"type":"content_block_delta","index":10,"delta":{"type":"text_delta","text":"second"}}
        \\
        \\data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"first"}}
        \\
        \\data: {"type":"content_block_delta","index":10,"delta":{"type":"text_delta","text":"!"}}
        \\
        \\
    ;
    const sink = struct {
        fn onEvent(_: *anyopaque, _: Anthropic.StreamEvent) Anthropic.StreamSink.Action {
            return .proceed;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    for ([_]bool{ false, true }) |terminal| {
        const wire = if (terminal) deltas ++ "data: {\"type\":\"message_stop\"}\n\n" else deltas;
        var assembler: lm_mod.OpenAI.OpenAIStream = .{
            .arena = arena,
            .model = "fixture",
            .stop = &.{},
        };
        var stream: MessageStream = .{
            .backing = arena,
            .model = "fixture",
            .adaptations = &.{},
            .sink = .{ .context = &assembler, .on_event = sink.onEvent },
            .assembler = &assembler,
        };
        var reader: std.Io.Reader = .fixed(wire);
        try MessageStream.consume(&stream, arena, &reader, wire.len);
        if (!terminal) {
            try std.testing.expect(assembler.result == null);
            continue;
        }
        const result = assembler.result.?;
        const parts = result.message.?.parts;
        try std.testing.expectEqual(@as(usize, 2), parts.len);
        try std.testing.expectEqualStrings("first", parts[0].text.text);
        try std.testing.expectEqualStrings("second!", parts[1].text.text);
    }
}

test "Anthropic reasoning replays signed and redacted blocks without showing either as the answer" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const input = try parse(arena,
        \\{"content":[{"type":"thinking","thinking":"private summary","signature":"opaque"},
        \\{"type":"redacted_thinking","data":"hidden"},{"type":"text","text":"Paris"}]}
    );
    const result = try parseResponse(arena, "claude-opus-4-7", input, &.{});
    try std.testing.expectEqualStrings("Paris", result.text.?);
    try std.testing.expectEqualStrings("private summary", result.reasoning.?);
    const message = result.message.?;
    const signed = try encodeBlock(arena, message.parts[0], .{});
    try std.testing.expectEqualStrings("thinking", text(get(signed, "type")));
    try std.testing.expectEqualStrings("opaque", text(get(signed, "signature")));
    const hidden = try encodeBlock(arena, message.parts[1], .{});
    try std.testing.expectEqualStrings("redacted_thinking", text(get(hidden, "type")));
    try std.testing.expectEqualStrings("hidden", text(get(hidden, "data")));
}

test "Anthropic hosted blocks do not shift client calls, but citations do" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const raw =
        \\{"content":[{"type":"server_tool_use","id":"server","name":"web_search"},
        \\{"type":"text","text":"Paris","citations":[{"uri":"https://example.org","quote":"Paris"}]},
        \\{"type":"web_search_tool_result","tool_use_id":"server"},
        \\{"type":"tool_use","name":"city","input":{}}]}
    ;
    const result = try parseResponse(
        arena,
        "fixture",
        try @import("json_repair.zig").parseJson(arena, raw),
        &.{},
    );
    const parts = result.message.?.parts;
    try std.testing.expectEqualStrings("Paris", parts[0].text.text);
    try std.testing.expectEqualStrings("https://example.org", parts[1].citation.url.?);
    try std.testing.expectEqualStrings("tool_2", parts[2].tool_call.id);
}

test "Anthropic alternate thinking distinguishes an absent request from off" {
    var backing: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const provider: Anthropic = .{
        .client = undefined,
        .model = "deepseek-fixture",
        .thinking_format = .deepseek,
    };
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    for ([_]?sdk.Reasoning{ null, .{ .effort = .off } }) |reasoning| {
        const request = try sdk.Request.fromFields(arena, &.{
            .model = provider.model,
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Next" } }} }},
            .config = .{ .reasoning = reasoning, .max_tokens = .{ .integer = 16 } },
        });
        defer request.release();
        const body = try parse(arena, try provider.requestBody(arena, request, &records, false));
        const thinking = get(body, "thinking");
        if (reasoning == null) {
            try std.testing.expect(thinking == .null);
        } else {
            try std.testing.expectEqualStrings("disabled", text(get(thinking, "type")));
        }
    }
}

test "Anthropic implicit cache policy cannot drop a referenced prompt prefix" {
    var backing: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const request = try sdk.Request.fromFields(arena, &.{
        .model = "fixture",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Next" } }} }},
        .config = .{ .cache = .{ .resource = "stored-prefix" } },
    });
    defer request.release();
    const provider: Anthropic = .{
        .client = undefined,
        .model = "fixture",
        .cache_control = .none,
        .adaptations = .silent,
    };
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    try std.testing.expectError(error.LMFailure, provider.requestBody(arena, request, &records, false));
}

test "Anthropic format rejection retains adaptive effort without advertising schemas" {
    var backing: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena,
        \\{"model":"claude-opus-4-6","messages":[{"role":"user","parts":[{"type":"text","text":"Next"}]}],
        \\"config":{"max_tokens":16,"reasoning":{"effort":"low"},
        \\"response_format":{"type":"json_schema","schema":{"type":"object"}}}}
    ));
    defer request.release();
    var provider: Anthropic = .{
        .client = undefined,
        .model = "fixture",
        .supports_response_schema = true,
        .structured_output = .reject,
    };
    try std.testing.expect(!provider.lm().supports_response_schema);
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try parse(arena, try provider.requestBody(arena, request, &records, false));
    const output = get(body, "output_config");
    try std.testing.expectEqualStrings("low", text(get(output, "effort")));
    try std.testing.expect(get(output, "format") == .null);
    provider.adaptations = .refuse;
    try std.testing.expectError(error.LMFailure, provider.requestBody(arena, request, &records, false));
}

test "Anthropic parallel-tool rejection preserves forced tool selection" {
    var backing: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena,
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"Next"}]}],
        \\"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
        \\"config":{"max_tokens":16,"tool_choice":{"mode":"required","allowed":["lookup"],"parallel":false}}}
    ));
    defer request.release();
    var provider: Anthropic = .{
        .client = undefined,
        .model = "fixture",
        .parallel_tool_calls = .reject,
    };
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try parse(arena, try provider.requestBody(arena, request, &records, true));
    const choice = get(body, "tool_choice");
    try std.testing.expectEqualStrings("tool", text(get(choice, "type")));
    try std.testing.expectEqualStrings("lookup", text(get(choice, "name")));
    try std.testing.expect(get(choice, "disable_parallel_tool_use") == .null);
    provider.adaptations = .refuse;
    try std.testing.expectError(error.LMFailure, provider.requestBody(arena, request, &records, true));
}

test "Anthropic fixed sampling omits explicit zero and refuses under strict policy" {
    var backing: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    const request = try sdk.Request.fromFields(arena, &.{
        .model = "fixture",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Next" } }} }},
        .config = .{ .temperature = 0, .max_tokens = .{ .integer = 16 } },
    });
    defer request.release();
    var provider: Anthropic = .{ .client = undefined, .model = "fixture", .sampling_params = .reject };
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const payload_bytes = try provider.requestBody(arena, request, &records, false);
    const wire = try @import("json_repair.zig").parseJson(arena, payload_bytes);
    try std.testing.expect(get(wire, "temperature") == .null);
    try std.testing.expectEqualStrings("config.temperature", records.items[0].field);
    provider.adaptations = .refuse;
    try std.testing.expectError(error.LMFailure, provider.requestBody(arena, request, &records, false));
}

test "Anthropic rejects foreign canonical models before credentials" {
    const Probe = struct {
        fn prepare(ptr: *anyopaque, _: Allocator, _: []const std.http.Header) anyerror![]const std.http.Header {
            const called: *bool = @ptrCast(@alignCast(ptr));
            called.* = true;
            return error.UnexpectedCredentialRequest;
        }

        fn onEvent(_: *anyopaque, _: Anthropic.StreamEvent) Anthropic.StreamSink.Action {
            return .proceed;
        }
    };
    var backing: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer backing.deinit();
    const arena = backing.allocator();
    var client: std.http.Client = .{ .allocator = arena, .io = std.testing.io };
    defer client.deinit();
    var called = false;
    var provider: Anthropic = .{
        .client = &client,
        .base_url = "http://invalid.invalid",
        .model = "served-local",
        .model_prefixes = &.{"served-"},
        .header_provider = .{ .context = &called, .prepare = Probe.prepare },
    };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = "foreign",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "Next" } }} }},
    });
    defer request.release();
    try std.testing.expectError(error.LMFailure, provider.completeRequest(arena, request));
    try std.testing.expectError(error.LMFailure, provider.streamRequest(arena, request, .{
        .context = &called,
        .on_event = Probe.onEvent,
    }));
    try std.testing.expect(!called);
}

test "Anthropic system prefix leaves the cache marker on the caller's block" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena,
        \\{"model":"fixture","system":"Caller instructions",
        \\"messages":[{"role":"user","parts":[{"type":"text","text":"Pick"}]}],
        \\"config":{"max_tokens":16,"cache":{"prefix":"stable","retention":"long"}}}
    ));
    defer request.release();
    const provider: Anthropic = .{
        .client = undefined,
        .model = "fixture",
        .system_prefix = "Application policy",
    };
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try parse(arena, try provider.requestBody(arena, request, &records, true));
    const blocks = get(body, "system").array.items;
    try std.testing.expectEqual(@as(usize, 2), blocks.len);
    try std.testing.expectEqualStrings("Application policy", text(get(blocks[0], "text")));
    try std.testing.expect(get(blocks[0], "cache_control") == .null);
    try std.testing.expectEqualStrings("Caller instructions", text(get(blocks[1], "text")));
    try std.testing.expectEqualStrings("1h", text(get(get(blocks[1], "cache_control"), "ttl")));
}

test "Anthropic judgment encoding does not change the caller schema" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const raw =
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"Pick"}]}],
        \\"config":{"max_tokens":16,"probabilities":"if_available","response_format":{
        \\"type":"json_schema","schema":{"type":"object","properties":{"pick":{
        \\"type":"string","anyOf":[{"const":"a"},{"const":"b"}]}}}}}}
    ;
    const parse = @import("json_repair.zig").parseJson;
    const request = try sdk.Request.fromValue(arena, try parse(arena, raw));
    defer request.release();
    const provider: Anthropic = .{ .client = undefined, .model = "fixture" };
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try parse(arena, try provider.requestBody(arena, request, &records, false));
    const wire = get(get(get(body, "output_config"), "format"), "schema");
    const own_format = try project(arena, request.fields.config.response_format.?);
    const wire_pick = get(get(wire, "properties"), "pick");
    const own_pick = get(get(get(own_format, "schema"), "properties"), "pick");
    try std.testing.expect(get(wire_pick, "type") == .null);
    try std.testing.expectEqualStrings("string", text(get(get(wire_pick, "anyOf").array.items[0], "type")));
    try std.testing.expectEqualStrings("string", text(get(own_pick, "type")));
    try std.testing.expect(get(get(own_pick, "anyOf").array.items[0], "type") == .null);
}

test "Anthropic sends a single data tool result as compact JSON text" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const raw =
        \\{"type":"tool_result","id":"lookup-1","content":[
        \\{"type":"data","value":{"city":"Zürich","count":9007199254740993}}]}
    ;
    const parsed = try @import("json_repair.zig").parseJson(arena, raw);
    const part = try sdk.Part.fromValue(arena, parsed);
    defer part.free(arena);
    const wire = try encodeBlock(arena, part, .{});
    try std.testing.expectEqualStrings(
        "{\"city\":\"Zürich\",\"count\":9007199254740993}",
        text(get(wire, "content")),
    );
}

test "Anthropic forcing one function preserves the other declarations" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var client: std.http.Client = .{ .allocator = arena, .io = std.testing.io };
    const provider: Anthropic = .{ .client = &client, .model = "fixture" };
    const request = try sdk.Request.fromFields(arena, &.{
        .model = "fixture",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "query" } }} }},
        .tools = &.{ .{ .function = .{ .name = "first" } }, .{ .function = .{ .name = "second" } } },
        .config = .{
            .max_tokens = .{ .integer = 32 },
            .tool_choice = .{ .mode = .required, .allowed = &.{"second"} },
        },
    });
    defer request.release();
    var records: std.ArrayList(sdk.Adaptation) = .empty;
    const body = try provider.requestBody(arena, request, &records, false);
    const value = try @import("json_repair.zig").parseJson(arena, body);
    const tools = get(value, "tools");
    try std.testing.expectEqual(@as(usize, 2), tools.array.items.len);
    try std.testing.expectEqualStrings("first", text(get(tools.array.items[0], "name")));
    try std.testing.expectEqualStrings("second", text(get(get(value, "tool_choice"), "name")));
    try std.testing.expectEqual(@as(usize, 0), records.items.len);
}
