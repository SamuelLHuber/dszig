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

/// Successful and partial responses are independently owned. Diagnostic strings
/// (provider error or unnamed-tool index) borrow the original event input.
pub const Materialized = union(enum) {
    complete: *const sdk.Response,
    missing_end: ?*const sdk.Response,
    after_end: *const sdk.Response,
    unnamed_tool: struct { part_index: []const u8, partial: *const sdk.Response },
    provider_error: ErrorDetail,

    pub fn release(self: Materialized) void {
        const response: ?*const sdk.Response = switch (self) {
            .complete, .after_end => |r| r,
            .missing_end => |r| r,
            .unnamed_tool => |failure| failure.partial,
            .provider_error => null,
        };
        if (response) |r| r.release();
    }
};

/// A synchronous event source. Event payloads remain valid until close
/// returns; close may release all source storage, even on error.
pub const Source = struct {
    context: *anyopaque,
    next: *const fn (*anyopaque) anyerror!?Event,
    close: *const fn (*anyopaque) anyerror!void,
};

/// Preserve the primary outcome and any secondary failure separately.
pub const SourceResult = struct {
    outcome: anyerror!Materialized,
    stopped: bool = false,
    read_after_end: ?anyerror = null,
    close_error: ?anyerror = null,
    diagnostic_bytes: ?[]u8 = null,

    /// Use the allocator originally supplied to materializeStreamSource.
    pub fn release(self: SourceResult, backing: Allocator) void {
        if (self.outcome) |result| result.release() else |_| {}
        if (self.diagnostic_bytes) |bytes| backing.free(bytes);
    }

    fn ownDiagnostic(self: *SourceResult, backing: Allocator, result: Materialized) !Materialized {
        var owned = result;
        switch (owned) {
            .unnamed_tool => |*failure| {
                const bytes = try backing.dupe(u8, failure.part_index);
                self.diagnostic_bytes = bytes;
                failure.part_index = bytes;
            },
            .provider_error => |*detail| {
                const size = std.math.add(usize, detail.provider_code.len, detail.message.len) catch return error.OutOfMemory;
                const bytes = try backing.alloc(u8, size);
                const split = detail.provider_code.len;
                @memcpy(bytes[0..split], detail.provider_code);
                @memcpy(bytes[split..], detail.message);
                self.diagnostic_bytes = bytes;
                detail.provider_code = bytes[0..split];
                detail.message = bytes[split..];
            },
            else => {},
        }
        return owned;
    }
};

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
        assemble_events: bool = false,
        provider_failed: bool = false,
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
        message_continuation: std.ArrayList(sdk.Continuation) = .empty,
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
            audio_media_type: ?[]const u8 = null,
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
                if (self.cut or self.outcome == .stopped or
                    (self.assemble_events and self.provider_failed)) break;
            }
            if (self.sink != null) {
                try self.finishEvents();
                if (!self.assemble_events) return;
                if (self.outcome == .stopped or self.provider_failed) return;
                self.metadata.usage = self.terminal_usage orelse .{};
            } else if (!self.cut) {
                try self.take(self.pending_text.items.len, false);
            }
            if (!self.ended) return error.LMFailure;
            self.result = self.assemble(false) catch |err| return if (err == error.WriteFailed) error.OutOfMemory else err;
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
            if (!self.ended or self.outcome == .stopped or
                (self.assemble_events and self.provider_failed)) return;
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
            return self.internIndex(key, true);
        }

        fn internIndex(self: *Self, key: []const u8, comptime copy_key: bool) !usize {
            if (self.slots.getIndex(key)) |index| return index;
            const owned = if (copy_key) try self.arena.dupe(u8, key) else key;
            try self.slots.put(self.arena, owned, .{ .index = owned });
            return self.slots.count() - 1;
        }

        pub fn materialize(
            backing: Allocator,
            request: *const sdk.Request,
            events: []const Event,
        ) lm.Error!Materialized {
            if (events.len > 65_536) return error.LMFailure;
            var scratch: std.heap.ArenaAllocator = .init(backing);
            defer scratch.deinit();
            const arena = scratch.allocator();
            var self: Self = .{ .arena = arena, .model = request.fields.model, .stop = &.{} };
            for (events, 0..) |event, i| {
                if (try self.foldEvent(backing, request, event)) |result| {
                    return if (result == .complete and i + 1 != events.len)
                        .{ .after_end = result.complete }
                    else
                        result;
                }
            }
            return self.missingEnd(backing, request);
        }

        pub fn materializeSource(
            backing: Allocator,
            request: *const sdk.Request,
            source: Source,
            observer: ?Sink,
        ) SourceResult {
            var scratch: std.heap.ArenaAllocator = .init(backing);
            defer scratch.deinit();
            var self: Self = .{
                .arena = scratch.allocator(),
                .model = request.fields.model,
                .stop = &.{},
            };
            var report: SourceResult = .{ .outcome = error.LMFailure };
            report.outcome = self.driveSource(backing, request, source, observer, &report);
            if (report.outcome) |result| {
                report.outcome = report.ownDiagnostic(backing, result) catch |err| blk: {
                    result.release();
                    break :blk err;
                };
            } else |_| {}
            source.close(source.context) catch |err| {
                report.close_error = err;
            };
            return report;
        }

        fn driveSource(
            self: *Self,
            backing: Allocator,
            request: *const sdk.Request,
            source: Source,
            observer: ?Sink,
            report: *SourceResult,
        ) anyerror!Materialized {
            for (0..65_536) |_| {
                const event = try source.next(source.context) orelse
                    return self.missingEnd(backing, request);
                const folded = try self.foldEvent(backing, request, event);
                const stop = if (observer) |sink|
                    sink.on_event(sink.context, event) == .stop
                else
                    false;
                if (folded) |result| {
                    if (result != .complete) return result;
                    // The end is final. Check one more pull even if the observer
                    // asked to stop on the end notification.
                    const trailing = source.next(source.context) catch |err| {
                        report.read_after_end = err;
                        return result;
                    };
                    return if (trailing == null) result else .{ .after_end = result.complete };
                }
                if (stop) {
                    report.stopped = true;
                    return self.missingEnd(backing, request);
                }
            }
            if (try source.next(source.context) == null) return self.missingEnd(backing, request);
            return error.LMFailure;
        }

        fn foldEvent(
            self: *Self,
            backing: Allocator,
            request: *const sdk.Request,
            event: Event,
        ) lm.Error!?Materialized {
            switch (event) {
                .start => |start_event| {
                    if (start_event.id) |id| {
                        if (id.len != 0) self.metadata.id = id;
                    }
                    if (start_event.model.len != 0) self.metadata.model = start_event.model;
                    if (start_event.adaptations.len != 0) self.adaptations = start_event.adaptations;
                },
                .provider_error => |detail| return .{ .provider_error = detail },
                .delta => |delta| self.accumulateDelta(delta.part_index, delta.data) catch |err|
                    return mapAssemblyError(err),
                .end => |end_event| {
                    self.metadata.finish_reason = end_event.finish_reason;
                    self.metadata.usage = end_event.usage orelse .{};
                    self.provider_data = end_event.provider_data;
                    const unnamed = self.firstUnnamedTool();
                    const response = try self.materializedResponse(backing, request);
                    if (unnamed) |part_index| return .{ .unnamed_tool = .{
                        .part_index = part_index,
                        .partial = response,
                    } };
                    return .{ .complete = response };
                },
            }
            return null;
        }

        fn missingEnd(
            self: *Self,
            backing: Allocator,
            request: *const sdk.Request,
        ) lm.Error!Materialized {
            const partial = self.materializedResponse(backing, request) catch |err| switch (err) {
                error.OutOfMemory => return err,
                else => null,
            };
            return .{ .missing_end = partial };
        }

        fn accumulateDelta(self: *Self, index: ?[]const u8, data: DeltaData) !void {
            const key = index orelse {
                if (data != .continuation) return error.LMFailure;
                return self.message_continuation.append(self.arena, data.continuation);
            };
            if (key.len == 0 or key.len > 4300) return error.LMFailure;
            if (key.len != 1 and key[0] == '0') return error.LMFailure;
            for (key) |digit| {
                if (digit < '0' or digit > '9') return error.LMFailure;
            }
            if (data == .image and data.image.data == null and data.image.url == null and
                data.image.file_id == null) return;
            try self.deliver(.{ .index = try self.internIndex(key, false), .data = data });
        }

        fn firstUnnamedTool(self: *const Self) ?[]const u8 {
            var first: ?[]const u8 = null;
            for (self.slots.values()) |entry| {
                const name: []const u8 = entry.name orelse "";
                if (entry.arguments == null or name.len != 0) continue;
                if (first == null or indexLess(entry.index, first.?)) first = entry.index;
            }
            return first;
        }

        fn materializedResponse(
            self: *Self,
            backing: Allocator,
            request: *const sdk.Request,
        ) lm.Error!*const sdk.Response {
            var completion = self.assemble(true) catch |err| return mapAssemblyError(err);
            try codec.fold(self.arena, &completion, request);
            return sdk.Response.fromFields(backing, &.{
                .id = completion.metadata.id,
                .model = completion.metadata.model.?,
                .message = completion.message.?,
                .finish_reason = completion.metadata.finish_reason.?,
                .usage = completion.metadata.usage,
                .logprobs = completion.logprobs,
                .logprobs_complete = completion.logprobs_complete,
                .provider_data = completion.provider_data,
                .adaptations = self.adaptations,
            }) catch |err| return mapAssemblyError(err);
        }

        fn mapAssemblyError(err: anyerror) lm.Error {
            return if (err == error.OutOfMemory) error.OutOfMemory else error.LMFailure;
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
                self.provider_failed = true;
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
                self.provider_failed = true;
                self.emit(.{ .provider_error = try self.makeError(code, err.object.get("message")) });
                return;
            }
            const choices = field(frame, "choices");
            const choice = if (choices == .array and choices.array.items.len != 0) choices.array.items[0] else .null;
            const delta = field(choice, "delta");
            const primary = field(delta, "reasoning_content");
            const thinking = if (try codec.truthy(primary)) primary else field(delta, "reasoning");
            if (try codec.truthy(thinking)) {
                try self.chatDelta("0", .{ .thinking = try codec.string(self.arena, thinking) });
                if (self.outcome == .stopped) return;
            }
            const text = field(delta, "content");
            if (text == .string and text.string.len != 0) {
                const scores = field(field(choice, "logprobs"), "content");
                try self.chatDelta("0", .{ .text = .{
                    .text = text.string,
                    .scores = try codec.scores(self.arena, scores),
                } });
                if (self.outcome == .stopped) return;
            }
            try self.chatTools(field(delta, "tool_calls"));
            if (self.outcome == .stopped) return;
            try self.chatTerminal(frame, field(choice, "finish_reason"), reasoning_usage);
        }

        fn chatDelta(self: *Self, part_index: []const u8, data: DeltaData) !void {
            self.start();
            if (self.outcome == .stopped) return;
            if (self.assemble_events) {
                const index = try self.internIndex(part_index, true);
                try self.deliver(.{ .index = index, .data = data });
            } else {
                self.emit(.{ .delta = .{ .part_index = part_index, .data = data } });
            }
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
                try self.chatDelta(index, .{ .tool_call = .{
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
                if (!self.assemble_events or self.outcome == .stopped) return;
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
                    if (media.data) |data| {
                        if (data.len == 0) return error.LMFailure;
                        @import("file.zig").validateBase64(data) catch return error.LMFailure;
                    }
                    entry.image = .{
                        .media_type = media.media_type orelse "image/png",
                        .data = media.data,
                        .url = media.url,
                        .file_id = media.file_id,
                    };
                },
                .audio => |media| {
                    if (entry.audio == null) {
                        entry.audio = .empty;
                        entry.audio_media_type = media.media_type;
                    }
                    const data = media.data orelse "";
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

        fn assemble(self: *Self, comptime canonical_only: bool) !lm.Completion {
            const indexes = try self.arena.alloc(usize, self.slots.count());
            for (indexes, 0..) |*index, i| index.* = i;
            std.mem.sort(usize, indexes, self, struct {
                fn less(reader: *Self, a: usize, b: usize) bool {
                    const left = reader.slots.values()[a].index;
                    const right = reader.slots.values()[b].index;
                    return indexLess(left, right);
                }
            }.less);
            var parts: std.ArrayList(sdk.Part) = .empty;
            var generated = false;
            var has_tool = false;
            var calls: std.ArrayList(lm.ToolCall) = .empty;
            var citations: std.ArrayList(sdk.CitationPart) = .empty;
            var continuation: std.ArrayList(sdk.Continuation) = .empty;
            for (indexes) |index| {
                const entry = &self.slots.values()[index];
                const states = entry.continuation.items;
                if (!canonical_only) try continuation.appendSlice(self.arena, states);
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
                    try parts.append(self.arena, .{ .audio = try audioPart(
                        self.arena,
                        audio.items,
                        entry.audio_media_type,
                        states,
                    ) });
                    generated = true;
                }
                for (entry.citations.items) |cited| {
                    var citation = cited;
                    citation.continuation = states;
                    try parts.append(self.arena, .{ .citation = citation });
                    if (!canonical_only) try citations.append(self.arena, citation);
                }
                if (entry.arguments) |arguments| {
                    const name: []const u8 = entry.name orelse "";
                    if (name.len == 0) {
                        if (canonical_only) continue;
                        return error.LMFailure;
                    }
                    const raw_id: []const u8 = entry.id orelse "";
                    const id = if (raw_id.len != 0) raw_id else try std.fmt.allocPrint(self.arena, "tool_call_{s}", .{entry.index});
                    const input = try self.toolInput(arguments.items);
                    try parts.append(self.arena, .{ .tool_call = .{ .id = id, .name = name, .input = try codec.payload(self.arena, input), .continuation = states } });
                    has_tool = true;
                    if (!canonical_only) try calls.append(self.arena, .{
                        .id = id,
                        .name = name,
                        .arguments = try codec.arguments(self.arena, input),
                    });
                } else if (states.len != 0 and entry.thinking == null and entry.text == null and entry.image == null and entry.audio == null and entry.citations.items.len == 0) {
                    try parts.append(self.arena, .{ .text = .{ .text = "", .continuation = states } });
                }
            }
            if (parts.items.len == 0) try parts.append(self.arena, .{ .text = .{ .text = "" } });
            if (self.metadata.finish_reason == null or self.metadata.finish_reason == .stop) {
                self.metadata.finish_reason = if (has_tool) .tool_call else .stop;
            }
            if (self.metadata.model == null) self.metadata.model = self.model;
            var completion: lm.Completion = .{ .text = null, .message = .{ .role = .assistant, .parts = parts.items }, .tool_calls = calls.items, .citations = citations.items, .continuation = continuation.items, .metadata = self.metadata, .provider_data = self.provider_data, .generated_media = generated, .logprobs = if (self.scores.items.len != 0) self.scores.items else null, .logprobs_complete = self.scores_complete };
            completion.message.?.continuation = self.message_continuation.items;
            if (!canonical_only) try codec.project(self.arena, &completion);
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

fn indexLess(left: []const u8, right: []const u8) bool {
    return if (left.len != right.len) left.len < right.len else std.mem.lessThan(u8, left, right);
}

fn audioPart(
    arena: Allocator,
    bytes: []const u8,
    media_type: ?[]const u8,
    continuation: []const sdk.Continuation,
) !sdk.MediaPart {
    const kind = media_type orelse "audio/pcm";
    const pcm = std.mem.eql(u8, kind, "audio/pcm") or std.mem.eql(u8, kind, "audio/pcm16");
    const raw = if (pcm) try pcmWave(arena, bytes) else bytes;
    if (raw.len == 0) return error.LMFailure;
    const encoder = std.base64.standard.Encoder;
    const encoded = try arena.alloc(u8, encoder.calcSize(raw.len));
    _ = encoder.encode(encoded, raw);
    return .{
        .media_type = if (pcm) "audio/wav" else kind,
        .data = encoded,
        .continuation = continuation,
    };
}

fn pcmWave(arena: Allocator, bytes: []const u8) ![]const u8 {
    if (bytes.len > std.math.maxInt(u32) - 36) return error.LMFailure;
    const raw = try arena.alloc(u8, 44 + bytes.len);
    // PCM, 24 kHz, mono, 16-bit, with the source's unpadded data chunk.
    const header = "RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" ++
        "\x01\x00\x01\x00\xc0\x5d\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00" ++
        "data\x00\x00\x00\x00";
    comptime std.debug.assert(header.len == 44);
    @memcpy(raw[0..44], header);
    std.mem.writeInt(u32, raw[4..8], @intCast(36 + bytes.len), .little);
    std.mem.writeInt(u32, raw[40..44], @intCast(bytes.len), .little);
    @memcpy(raw[44..], bytes);
    return raw;
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
