//! Native nonstreaming Anthropic Messages generation.
const std = @import("std");
const lm_mod = @import("lm.zig");
const http = @import("http.zig");
const runtime = @import("runtime_value.zig");
const sdk = lm_mod.sdk;
const Allocator = std.mem.Allocator;
const Error = lm_mod.Error;
const Value = runtime.Value;
const Object = runtime.ObjectMap;
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
    supports_reasoning: bool = false,
    supports_response_schema: bool = false,
    adaptations: enum { note, silent, refuse } = .note,

    pub fn lm(self: *Anthropic) lm_mod.LM {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = complete,
                .complete_canonical = canonical,
                .model_name = modelName,
                .clone = clone,
            },
            .supports_reasoning = self.supports_reasoning,
            .supports_response_format = self.supports_response_schema,
            .supports_response_schema = self.supports_response_schema,
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
        const copy = try arena.create(Anthropic);
        copy.* = self.*;
        copy.model = model;
        copy.base_url = base_url;
        return copy.lm();
    }

    fn complete(ptr: *anyopaque, arena: Allocator, request: lm_mod.Request) Error![]const lm_mod.Completion {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        for (request.tools) |tool| if (tool.strict != null) return error.LMFailure;
        const native = try lm_mod.OpenAI.canonicalPredictionRequest(
            arena,
            request,
            self.model,
            self.max_tokens,
            null,
        );
        defer native.release();
        const results = try arena.alloc(lm_mod.Completion, 1);
        results[0] = try canonical(ptr, arena, native);
        return results;
    }

    pub fn completeRequest(self: *Anthropic, arena: Allocator, request: *const sdk.Request) Error!*const sdk.Response {
        return (try canonical(self, arena, request)).response.?;
    }

    fn canonical(ptr: *anyopaque, arena: Allocator, request: *const sdk.Request) Error!lm_mod.Completion {
        const self: *Anthropic = @ptrCast(@alignCast(ptr));
        var records: std.ArrayList(sdk.Adaptation) = .empty;
        const body = try self.requestBody(arena, request, &records);
        const url = try std.fmt.allocPrint(arena, "{s}/messages", .{std.mem.trimEnd(u8, self.base_url, "/")});
        if (self.api_version.len == 0 or std.mem.indexOfAny(u8, self.api_version, "\r\n") != null)
            return error.LMFailure;
        var headers: [2]std.http.Header = undefined;
        headers[0] = .{ .name = "anthropic-version", .value = self.api_version };
        var headers_len: usize = 1;
        if (self.api_key) |key| {
            if (key.len == 0 or std.mem.indexOfAny(u8, key, "\r\n") != null) return error.LMFailure;
            headers[1] = .{ .name = "x-api-key", .value = key };
            headers_len = 2;
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
        const response = transport.send(arena, .{
            .url = url,
            .method = .POST,
            .body = body,
            .content_type = "application/json",
            .headers = headers[0..headers_len],
        }) catch |err| return codecError(err);
        defer self.client.allocator.free(response.body);
        const input = @import("json_repair.zig").parseJson(arena, response.body) catch |err|
            return codecError(err);
        return parseResponse(arena, request.fields.model, input, records.items);
    }

    fn requestBody(
        self: *const Anthropic,
        arena: Allocator,
        request: *const sdk.Request,
        records: *std.ArrayList(sdk.Adaptation),
    ) Error![]const u8 {
        const fields = &request.fields;
        const config = &fields.config;
        if (config.cache != null or config.probabilities != null) return error.LMFailure;
        var body: Object = .empty;
        const budget = try self.addReasoning(arena, &body, fields, records);
        try body.put(arena, "model", str(fields.model));
        try body.put(arena, "messages", try encodeMessages(arena, fields.messages));
        try body.put(arena, "stream", .{ .bool = false });
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
        if (fields.system) |system| {
            if (system != .text) return error.LMFailure;
            if (system.text.len != 0) try body.put(arena, "system", str(system.text));
        }
        try self.addSampling(arena, &body, config, records);
        try self.addTools(arena, &body, fields, records);
        try self.addOutput(arena, &body, config, records);
        var writer: std.Io.Writer.Allocating = .init(arena);
        @import("pyjson.zig").writeJson(&writer.writer, .{ .object = body }, .minified) catch |err|
            return if (err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
        if (!std.unicode.utf8ValidateSlice(writer.written())) return error.LMFailure;
        return writer.written();
    }

    fn addReasoning(
        self: *const Anthropic,
        arena: Allocator,
        body: *Object,
        fields: *const sdk.RequestFields,
        records: *std.ArrayList(sdk.Adaptation),
    ) Error!Value {
        const reasoning = fields.config.reasoning orelse return .null;
        const effort = reasoning.effort orelse return error.LMFailure;
        if (effort == .off) return .null;
        if (reasoning.summary) |summary| if (summary != .auto) {
            try self.adapt(arena, records, "config.reasoning.summary", .substituted, "the Messages API has no summary detail levels; it returns one summary of the " ++
                "thinking, which is 'auto'", str(@tagName(summary)), str("auto"));
        };
        var thinking: Object = .empty;
        const adaptive = adaptiveModel(fields.model);
        try thinking.put(arena, "type", str(if (adaptive) "adaptive" else "enabled"));
        if (reasoning.summary != null) try thinking.put(arena, "display", str("summarized"));
        var budget = reasoning.thinking_budget;
        if (adaptive) {
            if (budget != .null) {
                const reason = try std.fmt.allocPrint(arena, "{s} takes thinking.type 'adaptive' with output_config.effort; " ++
                    "budget_tokens is rejected by the API (live 2026-09-02). Thinking is bounded only " ++
                    "by max_tokens, which covers thinking and answer together: lower the effort or " ++
                    "raise max_tokens", .{fields.model});
                try self.adapt(arena, records, "config.reasoning.thinking_budget", .dropped, reason, budget, .null);
                budget = .null;
            }
            if (effort == .minimal) {
                try self.adapt(arena, records, "config.reasoning.effort", .clamped, "this model class has no 'minimal' level (output_config.effort is " ++
                    "low|medium|high|xhigh|max); 'low' is the floor", str("minimal"), str("low"));
            }
            var output: Object = .empty;
            try output.put(arena, "effort", str(if (effort == .minimal) "low" else @tagName(effort)));
            try body.put(arena, "output_config", .{ .object = output });
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
            try thinking.put(arena, "budget_tokens", budget);
        }
        try body.put(arena, "thinking", .{ .object = thinking });
        return budget;
    }

    fn addSampling(self: *const Anthropic, arena: Allocator, body: *Object, config: *const sdk.Config, records: *std.ArrayList(sdk.Adaptation)) Error!void {
        inline for (.{ "seed", "frequency_penalty", "presence_penalty" }) |name| {
            const asked: Value = if (comptime std.mem.eql(u8, name, "seed")) config.seed else if (@field(config, name)) |num| .{ .float = num } else .null;
            if (asked != .null) try self.adapt(arena, records, "config." ++ name, .dropped, "the Messages API has no " ++ name ++ " field", asked, .null);
        }
        if (config.temperature) |temp| {
            const applied: Value = .{ .float = @min(temp, 1) };
            if (temp > 1) try self.adapt(arena, records, "config.temperature", .clamped, "the Messages API accepts temperature in [0, 1]; the canonical range is [0, 2]", .{ .float = temp }, applied);
            try body.put(arena, "temperature", applied);
        }
        if (config.top_p) |top_p| try body.put(arena, "top_p", .{ .float = top_p });
        if (config.top_k != .null) try body.put(arena, "top_k", config.top_k);
        if (config.stop.len != 0) try body.put(arena, "stop_sequences", try strings(arena, config.stop));
    }

    fn addTools(self: *const Anthropic, arena: Allocator, body: *Object, fields: *const sdk.RequestFields, records: *std.ArrayList(sdk.Adaptation)) Error!void {
        if (fields.tools.len != 0) {
            var tools: runtime.Array = .init(arena);
            var names: runtime.Array = .init(arena);
            for (fields.tools) |tool| {
                if (tool != .function) return error.LMFailure;
                const function = tool.function;
                if (fields.config.tool_choice) |choice| {
                    const force = choice.mode == .required and choice.allowed.len == 1;
                    if (!force and choice.mode != .none and choice.allowed.len != 0 and
                        !contains(choice.allowed, function.name)) continue;
                }
                var item: Object = .empty;
                try item.put(arena, "name", str(function.name));
                if (function.description) |description| if (description.len != 0) {
                    try item.put(arena, "description", str(description));
                };
                try item.put(arena, "input_schema", try project(arena, function.parameters.?));
                try tools.append(.{ .object = item });
                try names.append(str(function.name));
            }
            if (tools.items.len != fields.tools.len) try self.adapt(
                arena,
                records,
                "config.tool_choice.allowed",
                .client_side,
                "the Messages API cannot restrict to a subset of the declared tools; only the allowed tools were sent, which is what the allowlist means",
                try strings(arena, fields.config.tool_choice.?.allowed),
                .{ .array = names },
            );
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
            if (choice.parallel == false and choice.mode != .none)
                try row.put(arena, "disable_parallel_tool_use", .{ .bool = true });
            try body.put(arena, "tool_choice", .{ .object = row });
        }
    }

    fn addOutput(self: *const Anthropic, arena: Allocator, body: *Object, config: *const sdk.Config, records: *std.ArrayList(sdk.Adaptation)) Error!void {
        if (config.response_format) |format| {
            const value = try project(arena, format);
            if (!std.mem.eql(u8, text(get(value, "type")), "json_schema")) return error.LMFailure;
            const schema = get(value, "schema");
            if (schema != .object) return error.LMFailure;
            var wire_format: Object = .empty;
            try wire_format.put(arena, "type", str("json_schema"));
            try wire_format.put(arena, "schema", schema);
            const existing = body.get("output_config");
            var output: Object = if (existing) |item| item.object else .empty;
            try output.put(arena, "format", .{ .object = wire_format });
            try body.put(arena, "output_config", .{ .object = output });
        }
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
                // Do not let a raw extension turn a bounded JSON read into SSE.
                if (std.mem.eql(u8, key, "stream")) {
                    if (entry.value_ptr.* != .bool or entry.value_ptr.bool) return error.LMFailure;
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

fn encodeMessages(arena: Allocator, messages: []const sdk.Message) Error!Value {
    var rows: runtime.Array = .init(arena);
    for (messages) |message| {
        var blocks: runtime.Array = .init(arena);
        if (message.role == .developer) {
            var joined: std.Io.Writer.Allocating = .init(arena);
            const writer = &joined.writer;
            writer.writeAll("[developer]\n") catch return error.OutOfMemory;
            for (message.parts, 0..) |part, index| {
                if (part != .text) return error.LMFailure;
                if (index != 0) writer.writeByte('\n') catch return error.OutOfMemory;
                writer.writeAll(part.text.text) catch return error.OutOfMemory;
            }
            try blocks.append(try textBlock(arena, joined.written()));
        } else {
            for (message.parts) |part| try blocks.append(try encodeBlock(arena, part));
        }
        var row: Object = .empty;
        try row.put(arena, "role", str(if (message.role == .assistant) "assistant" else "user"));
        try row.put(arena, "content", .{ .array = blocks });
        try rows.append(.{ .object = row });
    }
    return .{ .array = rows };
}

fn encodeBlock(arena: Allocator, part: sdk.Part) Error!Value {
    var block: Object = .empty;
    switch (part) {
        .text => |item| return textBlock(arena, item.text),
        .thinking => |item| return encodeThinking(arena, &item),
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
            } else if (item.content.len != 0) {
                var blocks: runtime.Array = .init(arena);
                for (item.content) |child| {
                    if (child != .text) return error.LMFailure;
                    try blocks.append(try textBlock(arena, child.text.text));
                }
                try block.put(arena, "content", .{ .array = blocks });
            }
            if (item.is_error) try block.put(arena, "is_error", .{ .bool = true });
        },
        else => return error.LMFailure,
    }
    return .{ .object = block };
}

fn encodeThinking(arena: Allocator, item: *const sdk.ThinkingPart) Error!Value {
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

fn parseResponse(arena: Allocator, model: []const u8, input: Value, records: []const sdk.Adaptation) Error!lm_mod.Completion {
    if (input != .object) return error.LMFailure;
    var parts: std.ArrayList(sdk.Part) = .empty;
    var calls: std.ArrayList(lm_mod.ToolCall) = .empty;
    var visible: std.Io.Writer.Allocating = .init(arena);
    var reasoning: std.Io.Writer.Allocating = .init(arena);
    const content = get(input, "content");
    if (content != .null and content != .array) return error.LMFailure;
    for (if (content == .array) content.array.items else &.{}) |block| {
        const kind = text(get(block, "type"));
        if (std.mem.eql(u8, kind, "text")) {
            const chunk = text(get(block, "text"));
            const citations = get(block, "citations");
            if (citations != .null and (citations != .array or citations.array.items.len != 0)) return error.LMFailure;
            visible.writer.writeAll(chunk) catch return error.OutOfMemory;
            try parts.append(arena, .{ .text = .{ .text = chunk } });
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
        } else return error.LMFailure;
    }
    if (parts.items.len == 0) try parts.append(arena, .{ .text = .{} });
    const id = text(get(input, "id"));
    const model_id = text(get(input, "model"));
    var result: lm_mod.Completion = .{
        .text = if (visible.written().len == 0 and calls.items.len != 0) null else visible.written(),
        .reasoning = if (reasoning.written().len == 0) null else reasoning.written(),
        .tool_calls = calls.items,
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
    try lm_mod.publishResponse(arena, &result, model, .borrowed);
    return result;
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
    defer result.response.?.release();
    try std.testing.expectEqualStrings("Paris", result.text.?);
    try std.testing.expectEqualStrings("private summary", result.reasoning.?);
    const message = result.response.?.fields.message;
    const signed = try encodeBlock(arena, message.parts[0]);
    try std.testing.expectEqualStrings("thinking", text(get(signed, "type")));
    try std.testing.expectEqualStrings("opaque", text(get(signed, "signature")));
    const hidden = try encodeBlock(arena, message.parts[1]);
    try std.testing.expectEqualStrings("redacted_thinking", text(get(hidden, "type")));
    try std.testing.expectEqualStrings("hidden", text(get(hidden, "data")));
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
    const body = try provider.requestBody(arena, request, &records);
    const value = try @import("json_repair.zig").parseJson(arena, body);
    const tools = get(value, "tools");
    try std.testing.expectEqual(@as(usize, 2), tools.array.items.len);
    try std.testing.expectEqualStrings("first", text(get(tools.array.items[0], "name")));
    try std.testing.expectEqualStrings("second", text(get(get(value, "tool_choice"), "name")));
    try std.testing.expectEqual(@as(usize, 0), records.items.len);
}
