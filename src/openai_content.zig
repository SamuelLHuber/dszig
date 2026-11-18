//! Canonical SDK message parts on the two OpenAI generation wires.
//! Configuration/adaptation decisions belong to the request builder; this
//! codec borrows the request and writes only execution-arena containers.
const std = @import("std");
const sdk = @import("sdk_response.zig");
const runtime = @import("runtime_value.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Array = runtime.Array;
const Object = runtime.ObjectMap;
pub const Error = Allocator.Error || error{LMFailure};
pub const Dialect = enum { chat, responses };
pub const Policy = struct {
    instruction_role: enum { system, developer } = .system,
    thinking_replay: enum { as_text, native, omit } = .as_text,
    assistant_reasoning_content: enum { omit, include_empty } = .omit,
    tool_result_name: enum { omit, include } = .omit,
    tool_result_media: enum { native, images, reject } = .reject,
    commentary_phase: enum { omit, tag } = .omit,
};
pub const Options = struct {
    dialect: Dialect,
    policy: Policy,
    max_media_bytes: usize,
    breakpoint_index: ?usize = null,
    stable_prefix: bool = false,
};

pub fn messages(
    arena: Allocator,
    io: std.Io,
    request: *const sdk.Request,
    options: Options,
) Error!Array {
    // Admission precedes all rendering, including local-media reads.
    for (request.fields.messages) |message| {
        if (message.role == .assistant) {
            for (message.parts) |part| if (isMedia(part)) return error.LMFailure;
        }
    }
    var codec: Codec = .{ .arena = arena, .io = io, .options = options };
    var rows: Array = .init(arena);
    if (options.dialect == .chat or options.stable_prefix) {
        if (request.fields.system) |system| {
            if (systemPresent(system)) {
                const text = try systemText(arena, system);
                const content = if (options.stable_prefix)
                    try markedText(arena, if (options.dialect == .chat) "text" else "input_text", text)
                else
                    Value{ .string = text };
                try rows.append(try row(arena, @tagName(options.policy.instruction_role), content));
            }
        }
    }
    for (request.fields.messages, 0..) |*message, index| {
        if (message.role == .tool) {
            try codec.toolRows(&rows, message);
        } else if (options.dialect == .chat) {
            try codec.chatRow(&rows, message, index);
        } else {
            try codec.responseRows(&rows, message, index);
        }
    }
    return rows;
}

pub fn systemPresent(system: sdk.SystemContent) bool {
    return switch (system) {
        .text => |text| text.len != 0,
        .parts => |parts| parts.len != 0,
    };
}

pub fn systemText(arena: Allocator, system: sdk.SystemContent) Error![]const u8 {
    return switch (system) {
        .text => |text| text,
        .parts => |parts| partsText(arena, parts),
    };
}

pub fn markedText(arena: Allocator, kind: []const u8, text: []const u8) Error!Value {
    var block = try textBlock(arena, kind, text);
    try mark(arena, &block);
    var blocks: Array = .init(arena);
    try blocks.append(block);
    return .{ .array = blocks };
}

const Codec = struct {
    arena: Allocator,
    io: std.Io,
    options: Options,

    fn chatRow(self: *const Codec, rows: *Array, message: *const sdk.Message, index: usize) Error!void {
        if (message.role == .assistant) {
            var bits: std.ArrayList([]const u8) = .empty;
            var thinking: std.ArrayList([]const u8) = .empty;
            var calls: Array = .init(self.arena);
            for (message.parts) |*part| switch (part.*) {
                .text => |text| try bits.append(self.arena, text.text),
                .refusal => |text| if (text.text.len != 0) {
                    try bits.append(self.arena, text.text);
                },
                .thinking => |text| if (text.text.len != 0) {
                    if (self.options.policy.thinking_replay == .as_text)
                        try bits.append(self.arena, text.text);
                    try thinking.append(self.arena, text.text);
                },
                .tool_call => |*tool| try calls.append(try self.call(tool)),
                else => {},
            };
            const content: Value = if (bits.items.len == 0)
                .null
            else
                .{ .string = try std.mem.join(self.arena, "\n", bits.items) };
            var item = try row(self.arena, "assistant", content);
            if (self.options.policy.thinking_replay == .native) {
                if (thinking.items.len != 0 or
                    self.options.policy.assistant_reasoning_content == .include_empty)
                {
                    try item.object.put(self.arena, "reasoning_content", .{
                        .string = try std.mem.join(self.arena, "\n", thinking.items),
                    });
                }
            }
            if (calls.items.len != 0)
                try item.object.put(self.arena, "tool_calls", .{ .array = calls });
            try rows.append(item);
            return;
        }
        const at_breakpoint = self.options.breakpoint_index == index;
        const content = try self.chatContent(message.parts, at_breakpoint);
        if (content == .array and content.array.items.len == 0) return;
        const role = if (message.role == .developer)
            @tagName(self.options.policy.instruction_role)
        else
            @tagName(message.role);
        try rows.append(try row(self.arena, role, content));
    }

    fn chatContent(self: *const Codec, parts: []const sdk.Part, at_breakpoint: bool) Error!Value {
        if (parts.len == 1 and !at_breakpoint) {
            switch (parts[0]) {
                .text => |text| return .{ .string = text.text },
                .data => |data| return .{ .string = try dataText(self.arena, data.value) },
                else => {},
            }
        }
        var blocks: Array = .init(self.arena);
        for (parts) |*part| switch (part.*) {
            .tool_call, .tool_result, .thinking => {},
            .image => |*image| try blocks.append(try self.chatImage(image)),
            .audio, .video, .document, .binary => return error.LMFailure,
            else => try blocks.append(try textBlock(self.arena, "text", try partText(self.arena, part))),
        };
        if (at_breakpoint) try markLast(self.arena, &blocks, "text");
        return .{ .array = blocks };
    }

    fn responseRows(self: *const Codec, rows: *Array, message: *const sdk.Message, index: usize) Error!void {
        var blocks: Array = .init(self.arena);
        if (message.role == .assistant) {
            for (message.parts) |*part| switch (part.*) {
                .text => |text| try blocks.append(try textBlock(self.arena, "output_text", text.text)),
                .refusal => |text| blk: {
                    var block: Object = .empty;
                    try block.put(self.arena, "type", .{ .string = "refusal" });
                    try block.put(self.arena, "refusal", .{ .string = text.text });
                    try blocks.append(.{ .object = block });
                    break :blk;
                },
                .thinking => |*thinking| {
                    if (try self.reasoning(thinking)) |item| {
                        try rows.append(item);
                    } else if (thinking.text.len != 0) {
                        try blocks.append(try textBlock(self.arena, "output_text", thinking.text));
                    }
                },
                else => {},
            };
        } else {
            for (message.parts) |*part| switch (part.*) {
                .tool_call, .tool_result => {},
                else => try blocks.append(try self.inputPart(part)),
            };
        }
        if (self.options.breakpoint_index == index)
            try markLast(self.arena, &blocks, "input_text");
        if (blocks.items.len != 0) {
            const role = if (message.role == .developer)
                @tagName(self.options.policy.instruction_role)
            else
                @tagName(message.role);
            var item = try row(self.arena, role, .{ .array = blocks });
            if (self.options.policy.commentary_phase == .tag and message.role == .assistant) {
                for (message.parts) |part| {
                    if (part == .tool_call) {
                        try item.object.put(self.arena, "phase", .{ .string = "commentary" });
                        break;
                    }
                }
            }
            try rows.append(item);
        }
        for (message.parts) |*part| {
            if (part.* == .tool_call) try rows.append(try self.call(&part.tool_call));
        }
    }

    fn reasoning(self: *const Codec, thinking: *const sdk.ThinkingPart) Error!?Value {
        for (thinking.continuation) |state| {
            if (!std.mem.eql(u8, state.provider, "openai") or
                !std.mem.eql(u8, state.kind, "reasoning_item")) continue;
            const data = state.data.toValue(self.arena) catch |err| return mapError(err);
            if (data != .object or data.object.count() == 0) return null;
            var item: Object = .empty;
            try item.put(self.arena, "type", .{ .string = "reasoning" });
            var fields = data.object.iterator();
            while (fields.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, "id") or
                    std.mem.eql(u8, entry.key_ptr.*, "encrypted_content"))
                {
                    try item.put(self.arena, entry.key_ptr.*, entry.value_ptr.*);
                }
            }
            var summary: Array = .init(self.arena);
            if (thinking.text.len != 0)
                try summary.append(try textBlock(self.arena, "summary_text", thinking.text));
            try item.put(self.arena, "summary", .{ .array = summary });
            return .{ .object = item };
        }
        return null;
    }

    fn call(self: *const Codec, tool: *const sdk.ToolCallPart) Error!Value {
        var function: Object = .empty;
        if (self.options.dialect == .responses) {
            try function.put(self.arena, "type", .{ .string = "function_call" });
            try function.put(self.arena, "call_id", .{ .string = tool.id });
        }
        try function.put(self.arena, "name", .{ .string = tool.name });
        const input = tool.input.toValue(self.arena) catch |err| return mapError(err);
        var encoded: std.Io.Writer.Allocating = .init(self.arena);
        pyjson.writeJsonDepth(&encoded.writer, input, .minified, 0, true) catch |err|
            return mapError(err);
        try function.put(self.arena, "arguments", .{ .string = encoded.written() });
        if (self.options.dialect == .responses) return .{ .object = function };
        var item: Object = .empty;
        try item.put(self.arena, "id", .{ .string = tool.id });
        try item.put(self.arena, "type", .{ .string = "function" });
        try item.put(self.arena, "function", .{ .object = function });
        return .{ .object = item };
    }

    fn toolRows(self: *const Codec, rows: *Array, message: *const sdk.Message) Error!void {
        for (message.parts) |*part| {
            if (part.* != .tool_result) continue;
            const tool = &part.tool_result;
            const content = try self.toolContent(tool);
            var item: Object = .empty;
            if (self.options.dialect == .responses) {
                try item.put(self.arena, "type", .{ .string = "function_call_output" });
                try item.put(self.arena, "call_id", .{ .string = tool.id });
                try item.put(self.arena, "output", content);
            } else {
                try item.put(self.arena, "role", .{ .string = "tool" });
                try item.put(self.arena, "tool_call_id", .{ .string = tool.id });
                try item.put(self.arena, "content", content);
            }
            if (self.options.policy.tool_result_name == .include) {
                if (tool.name) |name| {
                    if (name.len != 0) try item.put(self.arena, "name", .{ .string = name });
                }
            }
            try rows.append(.{ .object = item });
        }
    }

    fn toolContent(self: *const Codec, tool: *const sdk.ToolResultPart) Error!Value {
        var has_media = false;
        for (tool.content) |part| switch (part) {
            .image => {
                if (self.options.policy.tool_result_media == .reject) return error.LMFailure;
                has_media = true;
            },
            .document => {
                if (self.options.policy.tool_result_media != .native) return error.LMFailure;
                has_media = true;
            },
            .audio, .video, .binary => return error.LMFailure,
            else => {},
        };
        if (!has_media) {
            const text = try partsText(self.arena, tool.content);
            return .{ .string = if (tool.is_error)
                try std.fmt.allocPrint(self.arena, "[error] {s}", .{text})
            else
                text };
        }
        var blocks: Array = .init(self.arena);
        for (tool.content) |content| {
            var part: sdk.Part = switch (content) {
                inline else => |value, tag| @unionInit(sdk.Part, @tagName(tag), value),
            };
            try blocks.append(if (self.options.dialect == .responses)
                try self.inputPart(&part)
            else if (part == .image)
                try self.chatImage(&part.image)
            else
                try textBlock(self.arena, "text", try partText(self.arena, &part)));
        }
        if (tool.is_error) try errorPrefix(self.arena, &blocks, self.options.dialect);
        return .{ .array = blocks };
    }

    fn chatImage(self: *const Codec, image: *const sdk.MediaPart) Error!Value {
        if (image.file_id != null) return error.LMFailure;
        var spec: Object = .empty;
        try spec.put(self.arena, "url", .{
            .string = image.url orelse try self.mediaData(image, true),
        });
        if (image.detail) |detail|
            try spec.put(self.arena, "detail", .{ .string = @tagName(detail) });
        var block: Object = .empty;
        try block.put(self.arena, "type", .{ .string = "image_url" });
        try block.put(self.arena, "image_url", .{ .object = spec });
        return .{ .object = block };
    }

    fn inputPart(self: *const Codec, part: *const sdk.Part) Error!Value {
        switch (part.*) {
            .text => |text| return textBlock(self.arena, "input_text", text.text),
            .data => |data| return textBlock(self.arena, "input_text", try dataText(self.arena, data.value)),
            .thinking, .citation => return textBlock(self.arena, "input_text", try partText(self.arena, part)),
            .image, .audio, .video, .document, .binary => return self.inputMedia(part),
            else => return error.LMFailure,
        }
    }

    fn inputMedia(self: *const Codec, part: *const sdk.Part) Error!Value {
        const media = switch (part.*) {
            .image => &part.image,
            .audio => &part.audio,
            .video => &part.video,
            .document => &part.document,
            .binary => &part.binary,
            else => unreachable,
        };
        const kind = switch (part.*) {
            .image => "input_image",
            .audio => "input_audio",
            .video => "input_video",
            .document, .binary => "input_file",
            else => unreachable,
        };
        var block: Object = .empty;
        try block.put(self.arena, "type", .{ .string = kind });
        // Images prefer file_id; the other media kinds prefer URL.
        if (part.* == .image and media.file_id != null) {
            try block.put(self.arena, "file_id", .{ .string = media.file_id.? });
        } else if (media.url) |url| {
            const field = switch (part.*) {
                .image => "image_url",
                .audio => "audio_url",
                .video => "video_url",
                else => "file_url",
            };
            try block.put(self.arena, field, .{ .string = url });
        } else if (media.file_id) |id| {
            try block.put(self.arena, "file_id", .{ .string = id });
        } else if (part.* == .audio) {
            try block.put(self.arena, "audio", .{ .string = try self.mediaData(media, false) });
            const subtype = mediaSubtype(media.media_type, "audio/wav");
            const format = if (std.mem.eql(u8, subtype, "mpeg") or
                std.mem.eql(u8, subtype, "mp3")) "mp3" else subtype;
            try block.put(self.arena, "format", .{ .string = format });
        } else {
            if (part.* == .document or part.* == .binary) {
                const subtype = mediaSubtype(media.media_type, "application/octet-stream");
                const extension = subtype[0 .. std.mem.indexOfScalar(u8, subtype, '+') orelse subtype.len];
                try block.put(self.arena, "filename", .{ .string = try std.fmt.allocPrint(
                    self.arena,
                    "file.{s}",
                    .{if (extension.len == 0) "bin" else extension},
                ) });
            }
            try block.put(self.arena, if (part.* == .image) "image_url" else if (part.* == .video)
                "video_data"
            else
                "file_data", .{ .string = try self.mediaData(media, true) });
        }
        if (part.* == .image and media.file_id == null) {
            if (media.detail) |detail|
                try block.put(self.arena, "detail", .{ .string = @tagName(detail) });
        }
        return .{ .object = block };
    }

    fn mediaData(self: *const Codec, media: *const sdk.MediaPart, with_prefix: bool) Error![]const u8 {
        if (media.data) |data| return if (with_prefix)
            std.fmt.allocPrint(self.arena, "data:{s};base64,{s}", .{ media.media_type, data })
        else
            data;
        const path = media.path orelse return error.LMFailure;
        const bytes = std.Io.Dir.cwd().readFileAlloc(
            self.io,
            path,
            self.arena,
            .limited(self.options.max_media_bytes),
        ) catch |err| return mapError(err);
        defer self.arena.free(bytes);
        const prefix_size = if (with_prefix)
            std.math.add(usize, "data:;base64,".len, media.media_type.len) catch return error.OutOfMemory
        else
            0;
        const size = std.base64.standard.Encoder.calcSize(bytes.len);
        const total = std.math.add(usize, prefix_size, size) catch return error.OutOfMemory;
        const encoded = try self.arena.alloc(u8, total);
        if (with_prefix) {
            const prefix = std.fmt.bufPrint(encoded[0..prefix_size], "data:{s};base64,", .{media.media_type}) catch unreachable;
            std.debug.assert(prefix.len == prefix_size);
        }
        _ = std.base64.standard.Encoder.encode(encoded[prefix_size..], bytes);
        return encoded;
    }
};

fn partsText(arena: Allocator, parts: anytype) Error![]const u8 {
    var bits: std.ArrayList([]const u8) = .empty;
    for (parts) |part| {
        var canonical: sdk.Part = switch (part) {
            inline else => |value, tag| @unionInit(sdk.Part, @tagName(tag), value),
        };
        const text = try partText(arena, &canonical);
        switch (canonical) {
            .text, .data => try bits.append(arena, text),
            else => if (text.len != 0) try bits.append(arena, text),
        }
    }
    return std.mem.join(arena, "\n", bits.items);
}

fn partText(arena: Allocator, part: *const sdk.Part) Error![]const u8 {
    return switch (part.*) {
        .text => |text| text.text,
        .data => |data| dataText(arena, data.value),
        .thinking => |text| text.text,
        .citation => |citation| blk: {
            var bits: std.ArrayList([]const u8) = .empty;
            inline for (.{ "title", "url", "text" }) |name| {
                if (@field(citation, name)) |text| {
                    if (text.len != 0) try bits.append(arena, text);
                }
            }
            break :blk try std.mem.join(arena, " — ", bits.items);
        },
        .image, .audio, .video, .document, .binary => error.LMFailure,
        else => "",
    };
}

fn dataText(arena: Allocator, data: sdk.JsonValue) Error![]const u8 {
    const value = data.toValue(arena) catch |err| return mapError(err);
    var encoded: std.Io.Writer.Allocating = .init(arena);
    pyjson.writeJson(&encoded.writer, value, .minified) catch |err| return mapError(err);
    return encoded.written();
}

fn row(arena: Allocator, role: []const u8, content: Value) Error!Value {
    var item: Object = .empty;
    try item.put(arena, "role", .{ .string = role });
    try item.put(arena, "content", content);
    return .{ .object = item };
}

fn textBlock(arena: Allocator, kind: []const u8, text: []const u8) Error!Value {
    var block: Object = .empty;
    try block.put(arena, "type", .{ .string = kind });
    try block.put(arena, "text", .{ .string = text });
    return .{ .object = block };
}

fn mark(arena: Allocator, block: *Value) Error!void {
    std.debug.assert(block.* == .object);
    var value: Object = .empty;
    try value.put(arena, "mode", .{ .string = "explicit" });
    try block.object.put(arena, "prompt_cache_breakpoint", .{ .object = value });
}

fn markLast(arena: Allocator, blocks: *Array, kind: []const u8) Error!void {
    if (blocks.items.len == 0) return error.LMFailure;
    const last = &blocks.items[blocks.items.len - 1];
    const actual = last.object.get("type") orelse return error.LMFailure;
    if (actual != .string or !std.mem.eql(u8, actual.string, kind)) return error.LMFailure;
    try mark(arena, last);
}

fn errorPrefix(arena: Allocator, blocks: *Array, dialect: Dialect) Error!void {
    const kind = if (dialect == .chat) "text" else "input_text";
    for (blocks.items) |*block| {
        const actual = block.object.get("type").?;
        if (actual == .string and std.mem.eql(u8, actual.string, kind)) {
            const text = block.object.get("text").?.string;
            try block.object.put(arena, "text", .{
                .string = try std.fmt.allocPrint(arena, "[error] {s}", .{text}),
            });
            return;
        }
    }
    try blocks.insert(0, try textBlock(arena, kind, "[error]"));
}

fn mediaSubtype(media_type: []const u8, fallback: []const u8) []const u8 {
    const media = if (media_type.len == 0) fallback else media_type;
    return if (std.mem.indexOfScalar(u8, media, '/')) |slash| media[slash + 1 ..] else media;
}

fn isMedia(part: sdk.Part) bool {
    return switch (part) {
        .image, .audio, .video, .document, .binary => true,
        else => false,
    };
}

fn mapError(err: anyerror) Error {
    return if (err == error.OutOfMemory or err == error.WriteFailed) error.OutOfMemory else error.LMFailure;
}
