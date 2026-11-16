//! Text or canonical lm15 content parts. Custom-type markers are expanded once,
//! at the adapter boundary; they never masquerade as text on the provider wire.
const std = @import("std");
const inference = @import("inference.zig");
const pyjson = @import("pyjson.zig");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;

pub const start_marker = "<<CUSTOM-TYPE-START-IDENTIFIER>>";
pub const end_marker = "<<CUSTOM-TYPE-END-IDENTIFIER>>";
pub const Error = Allocator.Error || error{ InvalidContent, JsonTooDeep };

pub const Content = union(enum) {
    text: []const u8,
    /// Canonical part dictionaries, not provider-specific content blocks.
    parts: []const Value,

    pub fn clone(self: Content, arena: Allocator) Allocator.Error!Content {
        return switch (self) {
            .text => |text| .{ .text = try arena.dupe(u8, text) },
            .parts => |parts| blk: {
                const copies = try arena.alloc(Value, parts.len);
                for (parts, copies) |part, *copy| copy.* = try inference.cloneJsonValue(arena, part);
                break :blk .{ .parts = copies };
            },
        };
    }

    /// Borrowed JSON view; cloning is explicit at an ownership boundary.
    pub fn toValue(self: Content, arena: Allocator) Value {
        return switch (self) {
            .text => |text| .{ .string = text },
            .parts => |parts| blk: {
                var array: @import("runtime_value.zig").Array = .init(arena);
                array.items = @constCast(parts);
                array.capacity = parts.len;
                break :blk .{ .array = array };
            },
        };
    }

    pub fn fromValue(arena: Allocator, input: Value) Error!Content {
        if (input == .string) return .{ .text = try arena.dupe(u8, input.string) };
        if (input != .array) return error.InvalidContent;
        var builder: Builder = .{ .arena = arena };
        for (input.array.items) |part| try builder.append(try inference.cloneJsonValue(arena, part));
        return builder.finish();
    }

    pub fn fromRendered(arena: Allocator, rendered: []const u8) Error!Content {
        const first = std.mem.indexOf(u8, rendered, start_marker) orelse return .{ .text = rendered };
        if (std.mem.indexOfPos(u8, rendered, first + start_marker.len, end_marker) == null)
            return .{ .text = rendered };
        var builder: Builder = .{ .arena = arena };
        var offset: usize = 0;
        while (std.mem.indexOfPos(u8, rendered, offset, start_marker)) |begin| {
            const payload_start = begin + start_marker.len;
            const end = std.mem.indexOfPos(u8, rendered, payload_start, end_marker) orelse break;
            if (begin != offset) try builder.appendText(rendered[offset..begin]);
            const payload = stripPythonWhitespace(rendered[payload_start..end]);
            if (try payloadValues(arena, payload)) |values| {
                for (values) |v| {
                    if (v == .object) {
                        try builder.append(v);
                    } else {
                        var text: std.Io.Writer.Allocating = .init(arena);
                        pyjson.writeStr(&text.writer, v) catch |err| return switch (err) {
                            error.WriteFailed => error.OutOfMemory,
                            error.JsonTooDeep => error.JsonTooDeep,
                            error.ParseValue => error.InvalidContent,
                        };
                        try builder.appendText(text.written());
                    }
                }
            } else {
                // DSPy deliberately keeps the ORIGINAL payload for non-list JSON.
                try builder.appendText(payload);
            }
            offset = end + end_marker.len;
        }
        if (offset != rendered.len) try builder.appendText(rendered[offset..]);
        return builder.finish();
    }

    pub fn isEmpty(self: Content) bool {
        return switch (self) {
            .text => |text| text.len == 0,
            .parts => |parts| parts.len == 0,
        };
    }

    pub fn contains(self: Content, needle: []const u8) bool {
        return switch (self) {
            .text => |text| std.mem.indexOf(u8, text, needle) != null,
            .parts => |parts| blk: {
                for (parts) |part| {
                    const text = textOf(part) orelse continue;
                    if (std.mem.indexOf(u8, text, needle) != null) break :blk true;
                }
                break :blk false;
            },
        };
    }

    pub fn eql(a: Content, b: Content) @import("runtime_value.zig").CustomError!bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .text => |text| std.mem.eql(u8, text, b.text),
            .parts => |parts| blk: {
                if (parts.len != b.parts.len) break :blk false;
                for (parts, b.parts) |x, y| if (!(try inference.jsonEqual(x, y))) break :blk false;
                break :blk true;
            },
        };
    }
};

fn payloadValues(arena: Allocator, payload: []const u8) Error!?[]const Value {
    if (try parse(arena, payload)) |v| return if (v == .array) v.array.items else null;
    // Match json.loads(json.loads(f'"{payload}"')) for JSON-escaped markers.
    const quoted = try std.fmt.allocPrint(arena, "\"{s}\"", .{payload});
    const decoded = (try parse(arena, quoted)) orelse return null;
    if (decoded != .string) return null;
    const value = (try parse(arena, decoded.string)) orelse return null;
    return if (value == .array) value.array.items else null;
}

fn parse(arena: Allocator, input: []const u8) Allocator.Error!?Value {
    return std.json.parseFromSliceLeaky(Value, arena, input, .{ .allocate = .alloc_always }) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

fn textOf(part: Value) ?[]const u8 {
    if (part != .object) return null;
    const kind = part.object.get("type") orelse return null;
    if (kind != .string or !std.mem.eql(u8, kind.string, "text")) return null;
    const continuation = part.object.get("continuation") orelse .null;
    if (continuation != .null and (continuation != .array or continuation.array.items.len != 0)) return null;
    const text = part.object.get("text") orelse return "";
    return if (text == .string) text.string else null;
}

fn stripPythonWhitespace(input: []const u8) []const u8 {
    var begin: usize = 0;
    var end = input.len;
    while (begin < end) {
        const size = std.unicode.utf8ByteSequenceLength(input[begin]) catch break;
        if (size > end - begin) break;
        const codepoint = std.unicode.utf8Decode(input[begin..][0..size]) catch break;
        if (!@import("unicode.zig").isSpace(codepoint)) break;
        begin += size;
    }
    while (end > begin) {
        var start = end - 1;
        while (start > begin and input[start] & 0xc0 == 0x80) start -= 1;
        const codepoint = std.unicode.utf8Decode(input[start..end]) catch break;
        if (!@import("unicode.zig").isSpace(codepoint)) break;
        end = start;
    }
    return input[begin..end];
}

fn canonicalContinuation(arena: Allocator, input: Value) Error!?Value {
    if (input == .null) return null;
    if (input != .array) return error.InvalidContent;
    if (input.array.items.len == 0) return null;
    var array: @import("runtime_value.zig").Array = .init(arena);
    try array.ensureTotalCapacity(input.array.items.len);
    for (input.array.items) |state| {
        if (state != .object) return error.InvalidContent;
        const provider = state.object.get("provider") orelse return error.InvalidContent;
        const kind = state.object.get("kind") orelse return error.InvalidContent;
        const data = state.object.get("data") orelse Value{ .object = .empty };
        if (provider != .string or provider.string.len == 0 or
            kind != .string or kind.string.len == 0 or data != .object) return error.InvalidContent;
        var normalized: @import("runtime_value.zig").ObjectMap = .empty;
        try normalized.ensureTotalCapacity(arena, 3);
        normalized.putAssumeCapacity("provider", provider);
        normalized.putAssumeCapacity("kind", kind);
        normalized.putAssumeCapacity("data", data);
        array.appendAssumeCapacity(.{ .object = normalized });
    }
    return .{ .array = array };
}

const Builder = struct {
    arena: Allocator,
    parts: std.ArrayList(Value) = .empty,
    pending_text: ?[]const u8 = null,
    text_buffer: ?std.Io.Writer.Allocating = null,

    fn appendText(self: *Builder, text: []const u8) Error!void {
        const before = self.pending_text orelse {
            self.pending_text = text;
            return;
        };
        if (self.text_buffer == null) {
            self.text_buffer = .init(self.arena);
            self.text_buffer.?.writer.writeAll(before) catch return error.OutOfMemory;
        }
        self.text_buffer.?.writer.writeAll(text) catch return error.OutOfMemory;
    }

    fn pending(self: *Builder) ?[]const u8 {
        return if (self.text_buffer) |*buffer| buffer.written() else self.pending_text;
    }

    fn flushText(self: *Builder) Error!void {
        const text = self.pending() orelse return;
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.put(self.arena, "type", .{ .string = "text" });
        try object.put(self.arena, "text", .{ .string = text });
        try self.parts.append(self.arena, .{ .object = object });
        self.pending_text = null;
        self.text_buffer = null;
    }

    fn append(self: *Builder, part: Value) Error!void {
        if (part != .object) return error.InvalidContent;
        const kind = part.object.get("type") orelse return error.InvalidContent;
        if (kind != .string) return error.InvalidContent;
        const continuation = try canonicalContinuation(self.arena, part.object.get("continuation") orelse .null);
        var canonical: @import("runtime_value.zig").ObjectMap = .empty;
        try canonical.put(self.arena, "type", kind);
        if (std.mem.eql(u8, kind.string, "text")) {
            const text = part.object.get("text") orelse Value{ .string = "" };
            if (text != .string) return error.InvalidContent;
            if (continuation == null) return self.appendText(text.string);
            try canonical.put(self.arena, "text", text);
        } else if (std.mem.eql(u8, kind.string, "document")) {
            const media_type = part.object.get("media_type") orelse return error.InvalidContent;
            if (media_type != .string or media_type.string.len == 0) return error.InvalidContent;
            try canonical.put(self.arena, "media_type", media_type);
            var count: usize = 0;
            for ([_][]const u8{ "data", "url", "file_id", "path" }) |key| {
                const source = part.object.get(key) orelse continue;
                if (source == .null) continue;
                if (source != .string or source.string.len == 0) return error.InvalidContent;
                if (std.mem.eql(u8, key, "data"))
                    @import("file.zig").validateBase64(source.string) catch return error.InvalidContent;
                try canonical.put(self.arena, key, source);
                count += 1;
            }
            if (count != 1) return error.InvalidContent;
        } else return error.InvalidContent;
        if (continuation) |states| try canonical.put(self.arena, "continuation", states);
        try self.flushText();
        try self.parts.append(self.arena, .{ .object = canonical });
    }

    fn finish(self: *Builder) Error!Content {
        if (self.parts.items.len == 0) return .{ .text = self.pending() orelse "" };
        try self.flushText();
        if (self.parts.items.len == 1) {
            if (textOf(self.parts.items[0])) |text| return .{ .text = text };
        }
        return .{ .parts = try self.parts.toOwnedSlice(self.arena) };
    }
};
