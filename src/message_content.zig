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

    /// SDK training/display projection, not canonical serialization or LM HTTP policy.
    /// Leaves borrow Content; containers, data URIs and filenames belong to arena.
    /// Local documents are read only through the explicitly supplied I/O, so the
    /// caller's arena capacity bounds all memory. URL documents are refused.
    pub fn toOpenAIChatValue(self: Content, arena: Allocator, io: std.Io) !Value {
        const parts = switch (self) {
            .text => |text| return .{ .string = text },
            .parts => |parts| parts,
        };
        // The source writer ignores text/document continuation state, and a lone
        // text part is a bare string on the Chat wire even when it carries one.
        if (parts.len == 1) {
            if (try openAIChatText(parts[0])) |text| return .{ .string = text };
        }
        var array: @import("runtime_value.zig").Array = .init(arena);
        try array.ensureTotalCapacity(parts.len);
        for (parts) |part| {
            var block: @import("runtime_value.zig").ObjectMap = .empty;
            try block.ensureTotalCapacity(arena, 2);
            if (try openAIChatText(part)) |text| {
                block.putAssumeCapacity("type", .{ .string = "text" });
                block.putAssumeCapacity("text", .{ .string = text });
            } else {
                block.putAssumeCapacity("type", .{ .string = "file" });
                block.putAssumeCapacity("file", try openAIChatFile(arena, io, part));
            }
            array.appendAssumeCapacity(.{ .object = block });
        }
        std.debug.assert(array.items.len == parts.len);
        return .{ .array = array };
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

/// Text of a canonical text part, null for a document, error for anything else.
fn openAIChatText(part: Value) error{InvalidContent}!?[]const u8 {
    if (part != .object) return error.InvalidContent;
    const kind = part.object.get("type") orelse return error.InvalidContent;
    if (kind != .string) return error.InvalidContent;
    if (std.mem.eql(u8, kind.string, "document")) return null;
    if (!std.mem.eql(u8, kind.string, "text")) return error.InvalidContent;
    const text = part.object.get("text") orelse return "";
    if (text != .string) return error.InvalidContent;
    return text.string;
}

/// Source order: URL refusal, then file_id, then media bytes, then the filename.
fn openAIChatFile(arena: Allocator, io: std.Io, part: Value) !Value {
    std.debug.assert(part == .object);
    const media_type = part.object.get("media_type") orelse return error.InvalidContent;
    if (media_type != .string) return error.InvalidContent;
    if (part.object.get("url")) |url| {
        if (url != .null) return error.UnsupportedFeature;
    }
    var file: @import("runtime_value.zig").ObjectMap = .empty;
    if (part.object.get("file_id")) |id| {
        if (id != .null) {
            if (id != .string) return error.InvalidContent;
            try file.put(arena, "file_id", id);
            return .{ .object = file };
        }
    }
    const data = part.object.get("data") orelse Value.null;
    const uri = if (data == .null) local_file: {
        const path = part.object.get("path") orelse return error.InvalidContent;
        if (path != .string) return error.InvalidContent;
        // An empty filename avoids copying a basename the provider name ignores.
        const loaded = try @import("file.zig").File.fromPath(arena, io, path.string, .{
            .mime_type = media_type.string,
            .filename = "",
        });
        break :local_file loaded.file_data.?;
    } else inline_data: {
        if (data != .string) return error.InvalidContent;
        // Do not decode and re-encode: the SDK keeps the original base64 spelling.
        break :inline_data try std.fmt.allocPrint(arena, "data:{s};base64,{s}", .{
            media_type.string, data.string,
        });
    };
    std.debug.assert(std.mem.startsWith(u8, uri, "data:"));
    // The provider filename comes from the MIME type alone, never the local path.
    const extension = @import("mime_types.zig").guessExtension(media_type.string, true);
    try file.ensureTotalCapacity(arena, 2);
    file.putAssumeCapacity("file_data", .{ .string = uri });
    file.putAssumeCapacity("filename", .{
        .string = try std.fmt.allocPrint(arena, "file{s}", .{extension orelse ".bin"}),
    });
    return .{ .object = file };
}

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

fn testDocument(
    arena: Allocator,
    media_type: []const u8,
    key: []const u8,
    source: []const u8,
) !Value {
    var object: @import("runtime_value.zig").ObjectMap = .empty;
    try object.put(arena, "type", .{ .string = "document" });
    try object.put(arena, "media_type", .{ .string = media_type });
    try object.put(arena, key, .{ .string = source });
    return .{ .object = object };
}

fn expectOpenAIChat(arena: Allocator, expected_json: []const u8, content: Content) !void {
    const expected = (try parse(arena, expected_json)).?;
    const actual = try content.toOpenAIChatValue(arena, std.testing.io);
    try std.testing.expect(try inference.jsonEqual(expected, actual));
}

// Goal: the SDK training projection emits provider blocks, not canonical parts.
// Method: canonical Content goes in, the exact OpenAI Chat JSON shape comes out.
test "toOpenAIChatValue projects text, inline documents and file ids" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();

    const text: Content = .{ .text = "plain" };
    const projected = try text.toOpenAIChatValue(arena, std.testing.io);
    try std.testing.expectEqual(text.text.ptr, projected.string.ptr);

    // Non-canonical base64 spelling survives untouched; the MIME is lowercased
    // only for the extension lookup, never inside the data URI.
    const mixed = try Content.fromValue(arena, (try parse(arena,
        \\[{"type":"text","text":"read "},
        \\ {"type":"document","media_type":"Application/PDF","data":"aGVs\nbG8="},
        \\ {"type":"document","media_type":"text/plain","file_id":"file-1"},
        \\ {"type":"text","text":"now"}]
    )).?);
    try expectOpenAIChat(arena,
        \\[{"type":"text","text":"read "},
        \\ {"type":"file","file":{"file_data":"data:Application/PDF;base64,aGVs\nbG8=",
        \\   "filename":"file.pdf"}},
        \\ {"type":"file","file":{"file_id":"file-1"}},
        \\ {"type":"text","text":"now"}]
    , mixed);
    const actual = try mixed.toOpenAIChatValue(arena, std.testing.io);
    const projected_text = actual.array.items[0].object.get("text").?.string;
    try std.testing.expectEqual(mixed.parts[0].object.get("text").?.string.ptr, projected_text.ptr);

    // Continuation state is dropped, and a lone text part is a bare string.
    const continued = try Content.fromValue(arena, (try parse(arena,
        \\[{"type":"text","text":"kept","continuation":[{"provider":"p","kind":"k"}]}]
    )).?);
    try std.testing.expect(continued == .parts);
    try expectOpenAIChat(arena, "\"kept\"", continued);
    // The canonical view is unchanged by the provider projection.
    const canonical = continued.toValue(arena).array.items[0].object;
    try std.testing.expect(canonical.get("continuation") != null);
}

// Goal: filenames follow pinned CPython mimetypes.guess_extension(strict=True).
// Method: receipt observations, including aliases, case and absent strict entries.
test "toOpenAIChatValue filenames use the pinned inverse MIME database" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const guess = @import("mime_types.zig").guessExtension;

    for ([_][2][]const u8{
        .{ "application/pdf", "file.pdf" },
        .{ "TEXT/Plain", "file.txt" },
        .{ "application/msword", "file.doc" },
        .{ "application/octet-stream", "file.bin" },
        .{ "audio/midi", "file.mid" },
        // Only the non-strict database knows image/jpg.
        .{ "image/jpg", "file.bin" },
        // This key keeps uppercase in CPython's inverse map, so lower() misses it.
        .{ "application/vnd.HandHeld-Entertainment+xml", "file.bin" },
        .{ "application/x-dszig-unknown", "file.bin" },
        // U+212A KELVIN SIGN lowercases to ASCII k in str.lower().
        .{ "application/p\xe2\x84\xaacs10", "file.p10" },
    }) |case| {
        const content: Content = .{ .parts = &.{
            try testDocument(arena, case[0], "data", "QQ=="),
            try testDocument(arena, case[0], "file_id", "f"),
        } };
        const actual = try content.toOpenAIChatValue(arena, std.testing.io);
        const file = actual.array.items[0].object.get("file").?.object;
        try std.testing.expectEqualStrings(case[1], file.get("filename").?.string);
        const reference = actual.array.items[1].object.get("file").?.object;
        try std.testing.expectEqual(@as(usize, 1), reference.count());
        try std.testing.expectEqualStrings("f", reference.get("file_id").?.string);
    }
    try std.testing.expectEqualStrings(".jpg", guess("image/jpg", false).?);
    try std.testing.expectEqualStrings(".mid", guess("audio/midi", false).?);
    try std.testing.expectEqual(@as(?[]const u8, null), guess("image/jpg", true));
    try std.testing.expectEqual(@as(?[]const u8, null), guess("", true));
    const oversized: [200]u8 = @splat('x');
    try std.testing.expectEqual(@as(?[]const u8, null), guess(&oversized, false));
}

// Goal: local paths are really read through the injected I/O, and every
// refusal is explicit. Method: a temporary file, then each invalid neighbour.
test "toOpenAIChatValue reads local paths and refuses explicitly" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const io = std.testing.io;
    const expectError = std.testing.expectError;
    const pdf = "application/pdf";
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();

    try temporary.dir.writeFile(io, .{ .sub_path = "notes.csv", .data = "hello" });
    const directory = try temporary.dir.realPathFileAlloc(io, ".", arena);
    const path = try std.fmt.allocPrint(arena, "{s}/notes.csv", .{directory});
    const local: Content = .{ .parts = &.{try testDocument(arena, pdf, "path", path)} };
    // The filename is derived from the MIME type, NOT from the .csv basename.
    try expectOpenAIChat(arena,
        \\[{"type":"file","file":{"file_data":"data:application/pdf;base64,aGVsbG8=",
        \\   "filename":"file.pdf"}}]
    , local);

    const missing = try std.fmt.allocPrint(arena, "{s}/absent.pdf", .{directory});
    const absent: Content = .{ .parts = &.{try testDocument(arena, pdf, "path", missing)} };
    try expectError(error.FileNotFound, absent.toOpenAIChatValue(arena, io));
    const folder: Content = .{ .parts = &.{try testDocument(arena, pdf, "path", directory)} };
    try expectError(error.FileNotFound, folder.toOpenAIChatValue(arena, io));

    // A URL is refused before any later part is read or encoded.
    const remote: Content = .{ .parts = &.{
        try testDocument(arena, pdf, "url", "https://example.invalid/a.pdf"),
        try testDocument(arena, pdf, "path", missing),
    } };
    try expectError(error.UnsupportedFeature, remote.toOpenAIChatValue(arena, io));

    const sourceless: Content = .{ .parts = &.{(try parse(arena,
        \\{"type":"document","media_type":"application/pdf"}
    )).?} };
    try expectError(error.InvalidContent, sourceless.toOpenAIChatValue(arena, io));
    const image: Content = .{ .parts = &.{ (try parse(arena, "{\"type\":\"image\"}")).?, .null } };
    try expectError(error.InvalidContent, image.toOpenAIChatValue(arena, io));

    // The caller's arena capacity is the memory limit: no hidden allocator.
    var small: [96]u8 = undefined;
    var bounded: std.heap.FixedBufferAllocator = .init(&small);
    try expectError(error.OutOfMemory, local.toOpenAIChatValue(bounded.allocator(), io));
    const larger_than_arena: [4096]u8 = @splat('x');
    const borrowed: Content = .{ .text = &larger_than_arena };
    const borrowed_value = try borrowed.toOpenAIChatValue(bounded.allocator(), io);
    try std.testing.expectEqual(borrowed.text.ptr, borrowed_value.string.ptr);
}
