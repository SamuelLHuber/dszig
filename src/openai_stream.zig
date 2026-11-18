//! OpenAI SSE delivery and Responses materialization with an ordered stop cutter.
//! Provider codecs are compile-time functions shared with the buffered path.
const std = @import("std");
const sdk = @import("sdk_response.zig");
const lm = @import("lm.zig");
const runtime = @import("runtime_value.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Scores = []const lm.TokenLogprob;

pub const Event = union(enum) {
    start: struct { id: ?[]const u8, model: []const u8, adaptations: []const sdk.Adaptation },
    delta: struct { part_index: ?[]const u8, data: DeltaData },
    end: struct { finish_reason: ?lm.FinishReason, usage: ?lm.Usage, provider_data: ?sdk.JsonValue },
    provider_error: ErrorDetail,
};

pub const DeltaData = union(enum) {
    text: Text,
    thinking: []const u8,
    tool_call: Tool,
    image: Media,
    audio: Media,
    citation: sdk.CitationPart,
    continuation: sdk.Continuation,
};

pub const Media = struct {
    data: ?[]const u8 = null,
    url: ?[]const u8 = null,
    file_id: ?[]const u8 = null,
    media_type: ?[]const u8 = null,
};

pub const Tool = struct { input: []const u8, id: ?[]const u8, name: ?[]const u8 };
pub const Text = struct { text: []const u8, scores: Scores = &.{}, logprobs_complete: bool = true };

pub const ErrorDetail = struct {
    code: enum { provider, server, rate_limit, invalid_request, timeout, unsupported_model, context_length, auth, billing },
    provider_code: []const u8,
    message: []const u8,
};

pub const Sink = struct {
    context: *anyopaque,
    on_event: *const fn (*anyopaque, Event) Action,

    pub const Action = enum { proceed, stop };
};

pub const Outcome = enum { exhausted, stopped };

pub const ReplayError = std.mem.Allocator.Error ||
    error{ UnsupportedPart, IncompleteLogprobs, InvalidToolInput };

/// Visit a completed SDK response without transport or response assembly.
/// Like provider streaming, an early sink stop suppresses later events/errors.
/// The response is retained during delivery; event data is callback-borrowed.
pub fn responseToEvents(
    backing: Allocator,
    response: *const sdk.Response,
    sink: Sink,
) ReplayError!Outcome {
    const retained = response.retain();
    defer retained.release();
    const fields = &retained.fields;
    if (sink.on_event(sink.context, .{ .start = .{
        .id = fields.id,
        .model = fields.model,
        .adaptations = fields.adaptations,
    } }) == .stop) return .stopped;

    const parts = fields.message.parts;
    if (!fields.logprobs_complete) {
        for (parts) |part| {
            if (part == .text) break;
        } else return error.IncompleteLogprobs;
    }
    var scores: Text = .{
        .text = "",
        .scores = fields.logprobs orelse &.{},
        .logprobs_complete = fields.logprobs_complete,
    };
    for (parts, 0..) |*part, index| {
        var digits: [20]u8 = undefined;
        const key = std.fmt.bufPrint(&digits, "{d}", .{index}) catch unreachable;
        if (try replayPart(backing, part, key, &scores, sink) == .stop) return .stopped;
    }
    for (fields.message.continuation) |state| {
        if (replayDelta(sink, null, .{ .continuation = state }) == .stop) return .stopped;
    }
    return if (sink.on_event(sink.context, .{ .end = .{
        .finish_reason = fields.finish_reason,
        .usage = fields.usage,
        .provider_data = fields.provider_data,
    } }) == .stop) .stopped else .exhausted;
}

fn replayPart(
    backing: Allocator,
    part: *const sdk.Part,
    key: []const u8,
    scores: *Text,
    sink: Sink,
) ReplayError!Sink.Action {
    var arguments: std.Io.Writer.Allocating = .init(backing);
    defer arguments.deinit();
    const delta: DeltaData = switch (part.*) {
        .text => |text| blk: {
            const fragment: Text = .{
                .text = text.text,
                .scores = scores.scores,
                .logprobs_complete = scores.logprobs_complete,
            };
            scores.* = .{ .text = "" };
            break :blk .{ .text = fragment };
        },
        .thinking => |thinking| .{ .thinking = thinking.text },
        .tool_call => |tool| blk: {
            tool.input.writeJsonAscii(&arguments.writer) catch |err| return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                else => error.InvalidToolInput,
            };
            break :blk .{ .tool_call = .{
                .input = arguments.written(),
                .id = tool.id,
                .name = tool.name,
            } };
        },
        .image => |image| .{ .image = .{
            .data = image.data,
            .url = image.url,
            .file_id = image.file_id,
            .media_type = image.media_type,
        } },
        .audio => |audio| .{ .audio = .{
            .data = audio.data orelse return error.UnsupportedPart,
            .media_type = audio.media_type,
        } },
        .citation => |cited| .{ .citation = .{
            .text = cited.text,
            .url = cited.url,
            .title = cited.title,
        } },
        else => return error.UnsupportedPart,
    };
    if (replayDelta(sink, key, delta) == .stop) return .stop;
    const continuation = switch (part.*) {
        inline else => |record| record.continuation,
    };
    for (continuation) |state| {
        if (replayDelta(sink, key, .{ .continuation = state }) == .stop) return .stop;
    }
    return .proceed;
}

fn replayDelta(sink: Sink, index: ?[]const u8, data: DeltaData) Sink.Action {
    std.debug.assert(index != null or data == .continuation);
    return sink.on_event(sink.context, .{ .delta = .{ .part_index = index, .data = data } });
}

fn errorCode(code: []const u8, message: []const u8) @FieldType(ErrorDetail, "code") {
    const groups = .{
        .server = .{"server_error"},
        .rate_limit = .{ "rate_limit_exceeded", "no_capacity", "too_many_requests", "rate_limit_error" },
        .invalid_request = .{
            "invalid_prompt",       "invalid_image",                "invalid_image_format", "invalid_base64_image",           "invalid_image_url",
            "image_too_large",      "image_too_small",              "image_parse_error",    "image_content_policy_violation", "invalid_image_mode",
            "image_file_too_large", "unsupported_image_media_type", "empty_image_file",     "failed_to_download_image",       "image_file_not_found",
        },
        .timeout = .{"vector_store_timeout"},
        .unsupported_model = .{ "model_not_found", "model_not_available", "unsupported_model", "DeploymentNotFound" },
        .context_length = .{"context_length_exceeded"},
        .auth = .{ "invalid_api_key", "authentication_error" },
        .billing = .{ "insufficient_quota", "1113", "exceeded_current_quota_error" },
    };
    const forms = [_]struct { code: []const u8, prefix: []const u8 = "", suffix: []const u8 = "", contains: []const u8 = "" }{
        .{ .code = "not_found_error", .prefix = "model: " },
        .{ .code = "invalid_request_error", .contains = "The supported API model names are " },
        .{ .code = "1211" },
        .{ .code = "1214", .prefix = "modelCode: " },
        .{ .code = "400", .suffix = " is not a valid model ID" },
        .{ .code = "invalid-argument", .prefix = "Model not found: " },
        .{ .code = "validation_error", .contains = "The provided model identifier is invalid" },
        .{ .code = "invalid_request_error", .prefix = "Deployment ", .suffix = " doesn't exist or isn't accessible." },
    };
    for (forms) |form| {
        if (std.mem.eql(u8, code, form.code) and
            std.mem.startsWith(u8, message, form.prefix) and std.mem.endsWith(u8, message, form.suffix) and
            std.mem.indexOf(u8, message, form.contains) != null) return .unsupported_model;
    }
    inline for (@typeInfo(@TypeOf(groups)).@"struct".field_names) |name| {
        inline for (@field(groups, name)) |member| {
            if (std.mem.eql(u8, code, member)) return @field(@FieldType(ErrorDetail, "code"), name);
        }
    }
    return .provider;
}

pub fn Reader(comptime codec: anytype) type {
    return struct {
        arena: Allocator,
        model: []const u8,
        stop: []const []const u8,
        chat_reasoning_usage: ?@FieldType(@import("openai_request.zig").Compat, "reasoning_usage") = null,
        sink: ?Sink = null,
        adaptations: []const sdk.Adaptation = &.{},
        outcome: Outcome = .exhausted,
        terminal_usage: ?lm.Usage = null,
        hold_codepoints: usize = 0,
        slots: std.StringArrayHashMapUnmanaged(Slot) = .empty,
        pending: std.ArrayList(Delta) = .empty,
        pending_head: usize = 0,
        pending_text: std.ArrayList(u8) = .empty,
        scores: std.ArrayList(lm.TokenLogprob) = .empty,
        metadata: lm.ResponseMetadata = .{},
        provider_data: ?sdk.JsonValue = null,
        started: bool = false,
        ended: bool = false,
        cut: bool = false,
        scores_complete: bool = true,
        result: ?lm.Completion = null,
        max_bytes: usize = 0,
        synthesized_bytes: usize = 0,

        const Self = @This();
        const Delta = struct { index: usize, data: DeltaData };
        const Slot = struct {
            index: []const u8,
            text: ?std.ArrayList(u8) = null,
            thinking: ?std.ArrayList(u8) = null,
            arguments: ?std.ArrayList(u8) = null,
            id: ?[]const u8 = null,
            name: ?[]const u8 = null,
            image: ?sdk.MediaPart = null,
            audio: ?std.ArrayList(u8) = null,
            citations: std.ArrayList(sdk.CitationPart) = .empty,
            continuation: std.ArrayList(sdk.Continuation) = .empty,
            seen_text: std.ArrayList(u8) = .empty,
            seen_summary: std.ArrayList(u8) = .empty,
            seen_arguments: std.ArrayList(u8) = .empty,
            summary_index: usize = 0,
            named: bool = false,
        };

        pub fn consume(context: *anyopaque, arena: Allocator, reader: *std.Io.Reader, limit: usize) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            std.debug.assert(self.result == null);
            std.debug.assert(arena.ptr == self.arena.ptr);
            self.max_bytes = limit;
            for (self.stop) |sequence| {
                const count = std.unicode.utf8CountCodepoints(sequence) catch return error.LMFailure;
                self.hold_codepoints = @max(self.hold_codepoints, count -| 1);
            }
            var events = @import("sse.zig").Iterator.init(arena, reader, limit);
            defer events.deinit();
            while (events.next() catch |err| return if (err == error.WriteFailed) error.OutOfMemory else err) |data| {
                self.feed(data) catch |err| return if (err == error.WriteFailed) error.OutOfMemory else err;
                if (self.cut or self.outcome == .stopped) break;
            }
            if (self.sink != null) return self.finishEvents();
            if (!self.ended) return error.LMFailure;
            if (!self.cut) try self.take(self.pending_text.items.len, false);
            self.result = self.assemble() catch |err| return if (err == error.WriteFailed) error.OutOfMemory else err;
        }

        fn emit(self: *Self, event: Event) void {
            if (self.outcome == .stopped) return;
            const sink = self.sink.?;
            if (sink.on_event(sink.context, event) == .stop) self.outcome = .stopped;
        }

        fn start(self: *Self) void {
            if (self.started) return;
            self.started = true;
            if (self.sink != null) self.emit(.{ .start = .{
                .id = self.metadata.id,
                .model = self.metadata.model orelse self.model,
                .adaptations = self.adaptations,
            } });
        }

        fn finishEvents(self: *Self) !void {
            if (!self.ended or self.outcome == .stopped) return;
            self.start();
            if (!self.cut) try self.take(self.pending_text.items.len, false);
            self.emit(.{ .end = .{
                .finish_reason = self.metadata.finish_reason,
                .usage = self.terminal_usage,
                .provider_data = self.provider_data,
            } });
        }

        fn slot(self: *Self, index: usize) !*Slot {
            std.debug.assert(index < self.slots.count());
            return &self.slots.values()[index];
        }

        fn partIndex(self: *Self, value: Value) !usize {
            var buffer: [4301]u8 = undefined;
            const key = try indexText(value, &buffer);
            if (self.slots.getIndex(key)) |index| return index;
            const owned = try self.arena.dupe(u8, key);
            try self.slots.put(self.arena, owned, .{ .index = owned });
            return self.slots.count() - 1;
        }

        fn feed(self: *Self, data: []const u8) !void {
            if (std.mem.eql(u8, data, "[DONE]")) {
                self.ended = true;
                return;
            }
            if (data.len == 0) return;
            const frame = try codec.parse(self.arena, data);
            if (self.chat_reasoning_usage) |reasoning_usage| return self.chatFrame(frame, reasoning_usage);
            if (frame != .object) return error.LMFailure;
            const kind = try codec.text(self.arena, frame.object.get("type"));
            if (std.mem.eql(u8, kind, "response.created")) {
                if (self.started) return;
                const response = frame.object.get("response") orelse .null;
                if (response == .object) {
                    self.metadata.id = try codec.identity(self.arena, response.object.get("id"));
                    self.metadata.model = try codec.identity(self.arena, response.object.get("model"));
                }
                self.start();
                return;
            }
            if (std.mem.eql(u8, kind, "response.completed") or std.mem.eql(u8, kind, "response.incomplete")) {
                const raw = frame.object.get("response") orelse .null;
                const final: Value = if (raw == .object) raw else .{ .object = .empty };
                self.ended = true;
                self.metadata.usage = try codec.usage(self.arena, final);
                self.terminal_usage = self.metadata.usage;
                self.metadata.finish_reason = try codec.finish(self.arena, final, hasTools(final));
                self.provider_data = try codec.payload(self.arena, final);
                return;
            }
            if (std.mem.eql(u8, kind, "response.error") or std.mem.eql(u8, kind, "error")) {
                if (self.sink == null) return error.LMFailure;
                try self.take(self.pending_text.items.len, false);
                if (self.outcome != .stopped) self.emit(.{ .provider_error = try self.errorDetail(frame) });
                return;
            }
            try self.deltaFrame(frame, kind);
        }

        fn chatFrame(self: *Self, frame: Value, reasoning_usage: @FieldType(@import("openai_request.zig").Compat, "reasoning_usage")) !void {
            std.debug.assert(self.sink != null and self.stop.len == 0);
            if (frame != .object) return;
            const err = field(frame, "error");
            if (err == .object) {
                const code = try codec.identity(self.arena, err.object.get("code")) orelse
                    try codec.identity(self.arena, err.object.get("type")) orelse "provider";
                self.emit(.{ .provider_error = try self.makeError(code, err.object.get("message")) });
                return;
            }
            const choices = field(frame, "choices");
            const choice = if (choices == .array and choices.array.items.len != 0) choices.array.items[0] else .null;
            const delta = field(choice, "delta");
            const primary = field(delta, "reasoning_content");
            const thinking = if (try codec.truthy(primary)) primary else field(delta, "reasoning");
            if (try codec.truthy(thinking)) {
                self.chatDelta("0", .{ .thinking = try codec.string(self.arena, thinking) });
                if (self.outcome == .stopped) return;
            }
            const text = field(delta, "content");
            if (text == .string and text.string.len != 0) {
                const scores = field(field(choice, "logprobs"), "content");
                self.chatDelta("0", .{ .text = .{ .text = text.string, .scores = try codec.scores(self.arena, scores) } });
                if (self.outcome == .stopped) return;
            }
            try self.chatTools(field(delta, "tool_calls"));
            if (self.outcome == .stopped) return;
            try self.chatTerminal(frame, field(choice, "finish_reason"), reasoning_usage);
        }

        fn chatDelta(self: *Self, part_index: []const u8, data: DeltaData) void {
            self.start();
            self.emit(.{ .delta = .{ .part_index = part_index, .data = data } });
        }

        fn chatTools(self: *Self, calls: Value) !void {
            if (calls != .array) return;
            for (calls.array.items) |call| {
                if (call != .object) continue;
                const function = field(call, "function");
                const raw = field(call, "index");
                var digits: [4301]u8 = undefined;
                const index = try indexText(if (try codec.truthy(raw)) raw else .{ .integer = 0 }, &digits);
                if (index[0] == '-') return error.LMFailure;
                self.chatDelta(index, .{ .tool_call = .{
                    .input = try codec.text(self.arena, field(function, "arguments")),
                    .id = try codec.identity(self.arena, field(call, "id")),
                    .name = try codec.identity(self.arena, field(function, "name")),
                } });
                if (self.outcome == .stopped) return;
            }
        }

        fn field(value: Value, key: []const u8) Value {
            if (value != .object) return .null;
            return value.object.get(key) orelse .null;
        }

        fn chatTerminal(self: *Self, frame: Value, finish: Value, reasoning_usage: @FieldType(@import("openai_request.zig").Compat, "reasoning_usage")) !void {
            const has_finish = try codec.truthy(finish);
            const has_usage = field(frame, "usage") == .object;
            if (!has_finish and !has_usage) return;
            self.ended = true;
            if (has_finish) self.metadata.finish_reason = codec.chat_finish(.{ .string = try codec.string(self.arena, finish) }, false);
            // MAP-3: usage-bearing data outranks finish-only data. Known-empty
            // usage is still present, whereas [DONE] replaces neither field.
            if (has_usage or self.terminal_usage == null) self.provider_data = try codec.payload(self.arena, frame);
            if (has_usage) self.terminal_usage = try codec.chat_usage(self.arena, frame, reasoning_usage);
        }

        fn makeError(self: *Self, code: []const u8, raw_message: ?Value) !ErrorDetail {
            const message = try codec.identity(self.arena, raw_message) orelse code;
            return .{ .code = errorCode(code, message), .provider_code = code, .message = message };
        }

        fn hasTools(final: Value) bool {
            const output = final.object.get("output") orelse return false;
            if (output != .array) return false;
            for (output.array.items) |item| {
                if (item == .object and codec.equals(item.object.get("type") orelse .null, "function_call")) return true;
            }
            return false;
        }

        fn errorDetail(self: *Self, frame: Value) !ErrorDetail {
            const raw = frame.object.get("error") orelse .null;
            const nested_code = if (raw == .object)
                try codec.identity(self.arena, raw.object.get("code")) orelse try codec.identity(self.arena, raw.object.get("type"))
            else
                null;
            const code = nested_code orelse try codec.identity(self.arena, frame.object.get("code")) orelse
                (if (raw != .object) try codec.identity(self.arena, frame.object.get("error_type")) else null) orelse "provider";
            const nested_message = if (raw == .object) try codec.identity(self.arena, raw.object.get("message")) else null;
            const message = nested_message orelse try codec.identity(self.arena, frame.object.get("message")) orelse code;
            return .{ .code = errorCode(code, message), .provider_code = code, .message = message };
        }

        fn deltaFrame(self: *Self, frame: Value, kind: []const u8) !void {
            const tags = [_][]const u8{
                "response.output_item.done",              "response.output_item.added",            "response.output_text.delta",
                "response.refusal.delta",                 "response.reasoning_summary_text.delta", "response.reasoning_text.delta",
                "response.function_call_arguments.delta", "response.output_text.annotation.added", "response.output_image.delta",
                "response.image.delta",                   "response.output_audio.delta",
            };
            for (tags) |tag| {
                if (std.mem.eql(u8, kind, tag)) break;
            } else return;
            const raw_index = frame.object.get("output_index") orelse .null;
            const index = try self.partIndex(if (try codec.truthy(raw_index)) raw_index else .{ .integer = 0 });
            if (std.mem.eql(u8, kind, "response.output_item.done") or std.mem.eql(u8, kind, "response.output_item.added")) {
                const item = frame.object.get("item") orelse .null;
                if (item == .object) try self.itemFrame(item, index, std.mem.endsWith(u8, kind, ".done"));
                return;
            }
            if (std.mem.eql(u8, kind, "response.output_text.delta") or std.mem.eql(u8, kind, "response.refusal.delta")) {
                const text = try codec.text(self.arena, frame.object.get("delta"));
                try (try self.slot(index)).seen_text.appendSlice(self.arena, text);
                return self.push(.{ .index = index, .data = .{ .text = .{ .text = text, .scores = try codec.scores(self.arena, frame.object.get("logprobs") orelse .null) } } });
            }
            if (std.mem.eql(u8, kind, "response.reasoning_summary_text.delta") or std.mem.eql(u8, kind, "response.reasoning_text.delta")) {
                var text = try codec.text(self.arena, frame.object.get("delta"));
                if (std.mem.eql(u8, kind, "response.reasoning_summary_text.delta")) {
                    const entry = try self.slot(index);
                    var buffer: [4301]u8 = undefined;
                    const raw = frame.object.get("summary_index") orelse .null;
                    const digits = try indexText(if (try codec.truthy(raw)) raw else .{ .integer = 0 }, &buffer);
                    const part = if (digits[0] == '-') 0 else std.fmt.parseInt(usize, digits, 10) catch return error.LMFailure;
                    if (part > entry.summary_index) {
                        // Every prefix byte must fit inside the admitted wire.
                        const prefix_size = part - entry.summary_index;
                        if (prefix_size > self.max_bytes - self.synthesized_bytes) return error.LMFailure;
                        self.synthesized_bytes += prefix_size;
                        const joined = try self.arena.alloc(u8, prefix_size + text.len);
                        @memset(joined[0 .. part - entry.summary_index], '\n');
                        @memcpy(joined[part - entry.summary_index ..], text);
                        text = joined;
                        entry.summary_index = part;
                    }
                    try entry.seen_summary.appendSlice(self.arena, text);
                }
                return self.push(.{ .index = index, .data = .{ .thinking = text } });
            }
            if (std.mem.eql(u8, kind, "response.function_call_arguments.delta")) return self.toolDelta(frame, index, try codec.text(self.arena, frame.object.get("delta")));
            if (std.mem.eql(u8, kind, "response.output_text.annotation.added")) {
                const annotation = frame.object.get("annotation") orelse .null;
                if (annotation == .object) {
                    if (try codec.citation(self.arena, annotation, "")) |citation| try self.push(.{ .index = index, .data = .{ .citation = citation } });
                }
                return;
            }
            const image = std.mem.eql(u8, kind, "response.output_image.delta") or std.mem.eql(u8, kind, "response.image.delta");
            if (image or std.mem.eql(u8, kind, "response.output_audio.delta")) {
                const encoded = try codec.text(self.arena, frame.object.get("delta"));
                return self.push(.{
                    .index = index,
                    .data = if (image)
                        .{ .image = .{ .data = encoded, .media_type = "image/png" } }
                    else
                        .{ .audio = .{ .data = encoded, .media_type = "audio/wav" } },
                });
            }
        }

        fn toolDelta(self: *Self, frame: Value, index: usize, input: []const u8) !void {
            const entry = try self.slot(index);
            try entry.seen_arguments.appendSlice(self.arena, input);
            const name = try codec.identity(self.arena, frame.object.get("name"));
            if (name != null) entry.named = true;
            const id = try codec.identity(self.arena, frame.object.get("call_id")) orelse try codec.identity(self.arena, frame.object.get("id"));
            try self.push(.{ .index = index, .data = .{ .tool_call = .{ .input = input, .id = id, .name = name } } });
        }

        fn itemFrame(self: *Self, item: Value, index: usize, done: bool) !void {
            const kind = item.object.get("type") orelse .null;
            if (codec.equals(kind, "function_call")) {
                if (!done) return self.toolDelta(item, index, try codec.text(self.arena, item.object.get("arguments")));
                const raw = item.object.get("arguments") orelse .null;
                const final = if (raw == .string) raw.string else if (raw == .null) "" else try codec.json(self.arena, raw);
                const entry = try self.slot(index);
                const rest = try self.remainder(&entry.seen_arguments, final, true);
                if (rest.len != 0 or !entry.named) {
                    const name = try codec.identity(self.arena, item.object.get("name"));
                    if (name != null) entry.named = true;
                    const id = try codec.identity(self.arena, item.object.get("call_id")) orelse try codec.identity(self.arena, item.object.get("id"));
                    try self.push(.{ .index = index, .data = .{ .tool_call = .{ .input = rest, .id = id, .name = name } } });
                }
            } else if (codec.equals(kind, "message") and done) {
                var final: std.Io.Writer.Allocating = .init(self.arena);
                defer final.deinit();
                const content = item.object.get("content") orelse .null;
                if (content == .array) for (content.array.items) |part| {
                    if (part != .object) continue;
                    const tag = part.object.get("type") orelse .null;
                    if (codec.equals(tag, "output_text") or codec.equals(tag, "text")) {
                        try final.writer.writeAll(try codec.text(self.arena, part.object.get("text")));
                    } else if (codec.equals(tag, "refusal")) {
                        const refused = try codec.text(self.arena, part.object.get("refusal"));
                        try final.writer.writeAll(if (refused.len != 0) refused else try codec.text(self.arena, part.object.get("text")));
                    }
                };
                const entry = try self.slot(index);
                const rest = try self.remainder(&entry.seen_text, final.written(), false);
                if (rest.len != 0) try self.push(.{ .index = index, .data = .{ .text = .{ .text = rest } } });
            } else if (codec.equals(kind, "reasoning")) {
                if (!done) return self.push(.{ .index = index, .data = .{ .thinking = "" } });
                try self.reasoningDone(item, index);
            }
        }

        fn reasoningDone(self: *Self, item: Value, index: usize) !void {
            const summary = item.object.get("summary") orelse .null;
            if (summary == .array) {
                var final: std.Io.Writer.Allocating = .init(self.arena);
                defer final.deinit();
                for (summary.array.items, 0..) |part, i| {
                    if (i != 0) try final.writer.writeByte('\n');
                    const value = if (part == .object) part.object.get("text") orelse .null else part;
                    try final.writer.writeAll(try codec.string(self.arena, value));
                }
                const entry = try self.slot(index);
                const rest = try self.remainder(&entry.seen_summary, final.written(), false);
                if (rest.len != 0) try self.push(.{ .index = index, .data = .{ .thinking = rest } });
            }
            var state: runtime.ObjectMap = .empty;
            for ([_][]const u8{ "id", "encrypted_content" }) |key| {
                if (item.object.get(key)) |value| if (try codec.truthy(value)) try state.put(self.arena, key, value);
            }
            if (state.count() != 0) try self.push(.{ .index = index, .data = .{ .continuation = .{ .provider = "openai", .kind = "reasoning_item", .data = try codec.payload(self.arena, .{ .object = state }) } } });
        }

        fn remainder(self: *Self, seen: *std.ArrayList(u8), final: []const u8, json: bool) ![]const u8 {
            if (std.mem.startsWith(u8, final, seen.items)) {
                // A final writer's storage is reused/freed by its caller.
                const rest = try self.arena.dupe(u8, final[seen.items.len..]);
                seen.clearRetainingCapacity();
                try seen.appendSlice(self.arena, final);
                return rest;
            }
            if (json and try sameJson(self.arena, seen.items, final)) return "";
            return error.LMFailure;
        }

        fn push(self: *Self, delta: Delta) !void {
            if (self.outcome == .stopped) return;
            if ((try self.slot(delta.index)).index[0] == '-') return error.LMFailure;
            self.start();
            if (self.outcome == .stopped) return;
            if (self.stop.len == 0) return self.deliver(delta);
            if (delta.data != .text) {
                if (self.pending_head == self.pending.items.len) return self.deliver(delta);
                try self.pending.append(self.arena, delta);
                return;
            }
            try self.pending.append(self.arena, delta);
            try self.pending_text.appendSlice(self.arena, delta.data.text.text);
            var first: ?usize = null;
            for (self.stop) |sequence| {
                if (sequence.len == 0) continue;
                if (std.mem.indexOf(u8, self.pending_text.items, sequence)) |at| {
                    if (first == null or at < first.?) first = at;
                }
            }
            if (first) |at| {
                try self.take(at, true);
                self.cut = true;
                self.ended = true;
                self.metadata.finish_reason = .stop;
                // Terminal fields seen before the cut are coalescer-private.
                self.metadata.usage = .{};
                self.terminal_usage = null;
                self.provider_data = null;
                return;
            }
            var end = self.pending_text.items.len;
            for (0..self.hold_codepoints) |_| {
                if (end == 0) break;
                end -= 1;
                while (end > 0 and self.pending_text.items[end] & 0xc0 == 0x80) end -= 1;
            }
            try self.take(end, false);
        }

        fn take(self: *Self, count: usize, cutting: bool) !void {
            var remaining = count;
            var consumed: usize = 0;
            while (self.pending_head < self.pending.items.len and self.outcome != .stopped) {
                const delta = self.pending.items[self.pending_head];
                if (delta.data == .text) {
                    const text = delta.data.text;
                    if (cutting and remaining == 0) break;
                    if (text.text.len > remaining) {
                        if (!cutting) break;
                        const kept = try scoresBefore(text.scores, text.text, remaining);
                        try self.deliver(.{ .index = delta.index, .data = .{ .text = .{
                            .text = text.text[0..remaining],
                            .scores = kept.scores,
                            .logprobs_complete = text.logprobs_complete and !kept.incomplete,
                        } } });
                        break;
                    }
                    remaining -= text.text.len;
                    consumed += text.text.len;
                }
                try self.deliver(delta);
                self.pending_head += 1;
            }
            if (!cutting and consumed != 0) {
                std.mem.copyForwards(u8, self.pending_text.items, self.pending_text.items[consumed..]);
                self.pending_text.items.len -= consumed;
            }
            if (self.pending_head == self.pending.items.len) {
                self.pending.clearRetainingCapacity();
                self.pending_head = 0;
            }
        }

        fn append(self: *Self, target: *?std.ArrayList(u8), bytes: []const u8) !void {
            if (target.* == null) target.* = .empty;
            try target.*.?.appendSlice(self.arena, bytes);
        }

        fn deliver(self: *Self, delta: Delta) !void {
            const entry = try self.slot(delta.index);
            if (self.sink != null) {
                self.emit(.{ .delta = .{ .part_index = entry.index, .data = delta.data } });
                return;
            }
            switch (delta.data) {
                .text => |text| {
                    try self.append(&entry.text, text.text);
                    try self.scores.appendSlice(self.arena, text.scores);
                    self.scores_complete = self.scores_complete and text.logprobs_complete;
                },
                .thinking => |text| try self.append(&entry.thinking, text),
                .tool_call => |tool| {
                    try self.append(&entry.arguments, tool.input);
                    if (tool.id) |id| entry.id = id;
                    if (tool.name) |name| entry.name = name;
                },
                .continuation => |state| try entry.continuation.append(self.arena, state),
                .citation => |citation| try entry.citations.append(self.arena, citation),
                .image => |media| {
                    const data = media.data.?;
                    if (data.len == 0) return error.LMFailure;
                    @import("file.zig").validateBase64(data) catch return error.LMFailure;
                    entry.image = .{ .media_type = "image/png", .data = data };
                },
                .audio => |media| {
                    const data = media.data.?;
                    if (entry.audio == null) entry.audio = .empty;
                    const decoder = @import("python_base64.zig");
                    const padding: u2 = @intCast((4 - data.len % 4) % 4);
                    // Source retries with padding, then omits chunks which
                    // still cannot decode. An all-empty result refuses below.
                    const size = decoder.decode(data, .count, padding) catch return;
                    const raw = try entry.audio.?.addManyAsSlice(self.arena, size);
                    _ = try decoder.decode(data, .{ .bytes = raw }, padding);
                },
            }
        }

        fn assemble(self: *Self) !lm.Completion {
            const indexes = try self.arena.alloc(usize, self.slots.count());
            for (indexes, 0..) |*index, i| index.* = i;
            std.mem.sort(usize, indexes, self, struct {
                fn less(reader: *Self, a: usize, b: usize) bool {
                    const left = reader.slots.values()[a].index;
                    const right = reader.slots.values()[b].index;
                    return if (left.len != right.len) left.len < right.len else std.mem.lessThan(u8, left, right);
                }
            }.less);
            var parts: std.ArrayList(sdk.Part) = .empty;
            var generated = false;
            var calls: std.ArrayList(lm.ToolCall) = .empty;
            var citations: std.ArrayList(sdk.CitationPart) = .empty;
            var continuation: std.ArrayList(sdk.Continuation) = .empty;
            for (indexes) |index| {
                const entry = &self.slots.values()[index];
                const states = entry.continuation.items;
                try continuation.appendSlice(self.arena, states);
                if (entry.thinking) |value| {
                    try parts.append(self.arena, .{ .thinking = .{ .text = value.items, .continuation = states } });
                }
                if (entry.text) |value| {
                    try parts.append(self.arena, .{ .text = .{ .text = value.items, .continuation = states } });
                }
                if (entry.image) |media| {
                    var image = media;
                    image.continuation = states;
                    try parts.append(self.arena, .{ .image = image });
                    generated = true;
                }
                if (entry.audio) |audio| {
                    if (audio.items.len == 0) return error.LMFailure;
                    const encoded = try self.arena.alloc(u8, std.base64.standard.Encoder.calcSize(audio.items.len));
                    _ = std.base64.standard.Encoder.encode(encoded, audio.items);
                    try parts.append(self.arena, .{ .audio = .{ .media_type = "audio/wav", .data = encoded, .continuation = states } });
                    generated = true;
                }
                for (entry.citations.items) |cited| {
                    var citation = cited;
                    citation.continuation = states;
                    try parts.append(self.arena, .{ .citation = citation });
                    try citations.append(self.arena, citation);
                }
                if (entry.arguments) |arguments| {
                    const name = entry.name orelse return error.LMFailure;
                    if (name.len == 0) return error.LMFailure;
                    const id = entry.id orelse try std.fmt.allocPrint(self.arena, "tool_call_{s}", .{entry.index});
                    const input = try self.toolInput(arguments.items);
                    try parts.append(self.arena, .{ .tool_call = .{ .id = id, .name = name, .input = try codec.payload(self.arena, input), .continuation = states } });
                    try calls.append(self.arena, .{ .id = id, .name = name, .arguments = try codec.arguments(self.arena, input) });
                } else if (states.len != 0 and entry.thinking == null and entry.text == null and entry.image == null and entry.audio == null and entry.citations.items.len == 0) {
                    try parts.append(self.arena, .{ .text = .{ .text = "", .continuation = states } });
                }
            }
            if (parts.items.len == 0) try parts.append(self.arena, .{ .text = .{ .text = "" } });
            if (self.metadata.finish_reason == null or self.metadata.finish_reason == .stop) self.metadata.finish_reason = if (calls.items.len != 0) .tool_call else .stop;
            if (self.metadata.model == null) self.metadata.model = self.model;
            var completion: lm.Completion = .{ .text = null, .message = .{ .role = .assistant, .parts = parts.items }, .tool_calls = calls.items, .citations = citations.items, .continuation = continuation.items, .metadata = self.metadata, .provider_data = self.provider_data, .generated_media = generated, .logprobs = if (self.scores.items.len != 0) self.scores.items else null, .logprobs_complete = self.scores_complete };
            try codec.project(self.arena, &completion);
            return completion;
        }

        fn toolInput(self: *Self, raw: []const u8) !Value {
            if (raw.len == 0) return .{ .object = .empty };
            const parsed = codec.parse(self.arena, raw) catch |err| {
                if (err == error.OutOfMemory) return err;
                var partial: runtime.ObjectMap = .empty;
                try partial.put(self.arena, "partial_json", .{ .string = raw });
                return .{ .object = partial };
            };
            if (!codec.finite(parsed)) {
                // parse_constant refuses literal NaN/Infinity before SDK
                // validation. Exponent overflow instead parses successfully
                // and is refused by ToolCallPart's finite-JSON admission.
                const tokens = try @import("json_repair.zig").parseJsonTokens(self.arena, raw);
                for (tokens) |token| {
                    if (token == .scalar and token.scalar == .float) {
                        var partial: runtime.ObjectMap = .empty;
                        try partial.put(self.arena, "partial_json", .{ .string = raw });
                        return .{ .object = partial };
                    }
                }
                return error.LMFailure;
            }
            if (parsed == .object) return parsed;
            var wrapped: runtime.ObjectMap = .empty;
            try wrapped.put(self.arena, "value", parsed);
            return .{ .object = wrapped };
        }
    };
}

/// Part indexes are correlators, not array offsets. Preserve arbitrary-size
/// Python integers rather than allocating up to the largest index. Negative
/// correlators are rejected only when constructing an actual canonical delta.
fn indexText(value: Value, buffer: *[4301]u8) ![]const u8 {
    switch (value) {
        .integer => |n| return try std.fmt.bufPrint(buffer, "{d}", .{n}),
        .bool => |b| return if (b) "1" else "0",
        .number_string => |digits| {
            if (!std.json.isNumberFormattedLikeAnInteger(digits)) return error.LMFailure;
            return digits;
        },
        .float => |n| {
            if (!std.math.isFinite(n)) return error.LMFailure;
            var temporary: [310]u8 = undefined;
            const digits = @import("value.zig").integralFloatText(@trunc(n), &temporary) orelse return error.LMFailure;
            @memcpy(buffer[0..digits.len], digits);
            return buffer[0..digits.len];
        },
        .string => |raw| {
            const text = @import("unicode.zig").strip(raw, false);
            if (text.len == 0) return error.LMFailure;
            const negative = text[0] == '-';
            var at: usize = @intFromBool(negative or text[0] == '+');
            var count: usize = 0;
            var previous_digit = false;
            while (at < text.len) {
                if (text[at] == '_') {
                    if (!previous_digit) return error.LMFailure;
                    previous_digit = false;
                    at += 1;
                    continue;
                }
                const size = std.unicode.utf8ByteSequenceLength(text[at]) catch return error.LMFailure;
                if (size > text.len - at) return error.LMFailure;
                const cp = std.unicode.utf8Decode(text[at..][0..size]) catch return error.LMFailure;
                const digit = @import("unicode.zig").decimalDigit(cp) orelse return error.LMFailure;
                if (count == 4300) return error.LMFailure;
                buffer[count] = '0' + digit;
                count += 1;
                previous_digit = true;
                at += size;
            }
            if (!previous_digit) return error.LMFailure;
            var first: usize = 0;
            while (first < count and buffer[first] == '0') first += 1;
            if (first == count) return "0";
            if (negative) {
                if (first != 0) {
                    buffer[first - 1] = '-';
                    return buffer[first - 1 .. count];
                }
                std.mem.copyBackwards(u8, buffer[1..][0..count], buffer[0..count]);
                buffer[0] = '-';
                return buffer[0 .. count + 1];
            }
            return buffer[first..count];
        },
        else => return error.LMFailure,
    }
}

fn scoresBefore(scores: Scores, text: []const u8, boundary: usize) !struct { scores: Scores, incomplete: bool } {
    std.debug.assert(boundary <= text.len);
    if (scores.len == 0 or boundary == 0) return .{ .scores = &.{}, .incomplete = false };
    if (boundary == text.len) return .{ .scores = scores, .incomplete = false };
    var end: usize = 0;
    var kept: usize = 0;
    var crossing = false;
    for (scores, 0..) |score, i| {
        const length = if (score.bytes) |bytes| bytes.len else score.token.len;
        if (length > text.len - end) return .{ .scores = &.{}, .incomplete = true };
        if (score.bytes) |bytes| {
            for (bytes, 0..) |byte, offset| {
                if (byte != .integer or byte.integer < 0 or byte.integer > 255 or byte.integer != text[end + offset]) return .{ .scores = &.{}, .incomplete = true };
            }
        } else if (!std.mem.eql(u8, score.token, text[end..][0..length])) return .{ .scores = &.{}, .incomplete = true };
        end += length;
        if (end - length < boundary) {
            if (end <= boundary) kept = i + 1 else crossing = true;
        }
    }
    if (end != text.len) return .{ .scores = &.{}, .incomplete = true };
    return .{ .scores = scores[0..kept], .incomplete = crossing };
}

fn sameJson(arena: Allocator, a: []const u8, b: []const u8) !bool {
    const decoder = @import("json_repair.zig");
    const left = decoder.parseJsonTokens(arena, a) catch |err| return if (err == error.OutOfMemory) err else false;
    const right = decoder.parseJsonTokens(arena, b) catch |err| return if (err == error.OutOfMemory) err else false;
    if (left.len != right.len) return false;
    for (left, right) |x, y| {
        if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
        switch (x) {
            .number => |number| {
                const first = @import("decimal.zig").parse(arena, number) catch |err| return if (err == error.OutOfMemory) err else false;
                const second = @import("decimal.zig").parse(arena, y.number) catch |err| return if (err == error.OutOfMemory) err else false;
                if (!(first.equal(second) catch false)) return false;
            },
            .scalar => |scalar| {
                if (std.meta.activeTag(scalar) != std.meta.activeTag(y.scalar)) return false;
                const equal = switch (scalar) {
                    .null => true,
                    .bool => scalar.bool == y.scalar.bool,
                    .string => std.mem.eql(u8, scalar.string, y.scalar.string),
                    // json.loads uses the same NaN literal singleton.
                    .float => (std.math.isNan(scalar.float) and std.math.isNan(y.scalar.float)) or scalar.float == y.scalar.float,
                    else => unreachable,
                };
                if (!equal) return false;
            },
            else => {},
        }
    }
    return true;
}
