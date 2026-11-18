//! Incremental SSE data framing. Storage is reused between events; JSON
//! consumers must copy any retained leaves before advancing the iterator.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Iterator = struct {
    reader: *std.Io.Reader,
    arena: Allocator,
    remaining: usize,
    line: std.Io.Writer.Allocating,
    data: std.Io.Writer.Allocating,
    ended: bool = false,

    pub fn init(arena: Allocator, reader: *std.Io.Reader, limit: usize) Iterator {
        return .{ .reader = reader, .arena = arena, .remaining = limit, .line = .init(arena), .data = .init(arena) };
    }

    fn leftStrip(text: []const u8) []const u8 {
        var start: usize = 0;
        while (start < text.len) {
            const size = std.unicode.utf8ByteSequenceLength(text[start]) catch break;
            if (size > text.len - start) break;
            const cp = std.unicode.utf8Decode(text[start..][0..size]) catch break;
            if (!@import("unicode.zig").isSpace(cp)) break;
            start += size;
        }
        return text[start..];
    }

    pub fn deinit(self: *Iterator) void {
        self.line.deinit();
        self.data.deinit();
    }

    pub fn next(self: *Iterator) !?[]const u8 {
        self.data.clearRetainingCapacity();
        var has_data = false;
        while (!self.ended) {
            self.line.clearRetainingCapacity();
            const count = self.reader.streamDelimiterLimit(&self.line.writer, '\n', .limited(self.remaining +| 1)) catch |err| return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                else => err,
            };
            if (count > self.remaining) return error.StreamTooLong;
            self.remaining -= count;
            const delimiter = self.reader.takeByte() catch |err| switch (err) {
                error.EndOfStream => end: {
                    self.ended = true;
                    break :end null;
                },
                else => return err,
            };
            if (delimiter != null) {
                std.debug.assert(delimiter.? == '\n');
                if (self.remaining == 0) return error.StreamTooLong;
                self.remaining -= 1;
            }
            const raw = std.mem.trimEnd(u8, self.line.written(), "\r");
            const line = if (std.unicode.utf8ValidateSlice(raw)) raw else try replaceInvalidUtf8(self.arena, raw);
            if (line.len == 0) {
                if (has_data) return self.data.written();
                continue;
            }
            if (!std.mem.startsWith(u8, line, "data:")) continue;
            if (has_data) self.data.writer.writeByte('\n') catch return error.OutOfMemory;
            self.data.writer.writeAll(leftStrip(line[5..])) catch return error.OutOfMemory;
            has_data = true;
        }
        return if (has_data) self.data.written() else null;
    }
};

/// Match the byte-line decoder's errors="replace", including one replacement
/// for a truncated valid prefix and separate replacements for surrogate bytes.
fn replaceInvalidUtf8(arena: Allocator, text: []const u8) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    var at: usize = 0;
    while (at < text.len) {
        const first = text[at];
        if (first < 0x80) {
            out.writer.writeByte(first) catch return error.OutOfMemory;
            at += 1;
            continue;
        }
        const width: usize = if (first >= 0xc2 and first <= 0xdf) 2 else if (first >= 0xe0 and first <= 0xef) 3 else if (first >= 0xf0 and first <= 0xf4) 4 else 0;
        var valid: usize = 1;
        while (valid < width and valid < text.len - at) {
            const byte = text[at + valid];
            if (byte < 0x80 or byte > 0xbf) break;
            if (valid == 1 and ((first == 0xe0 and byte < 0xa0) or
                (first == 0xed and byte > 0x9f) or
                (first == 0xf0 and byte < 0x90) or
                (first == 0xf4 and byte > 0x8f))) break;
            valid += 1;
        }
        out.writer.writeAll(if (valid == width) text[at..][0..width] else "\xef\xbf\xbd") catch return error.OutOfMemory;
        at += valid;
    }
    return out.written();
}

test "SSE preserves data lines and replaces malformed UTF8 within the byte budget" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const bytes = ": comment\r\ndata: \xc2\xa0first\xe2\x82\r\ndata: \xed\xa0\x80\r\n\r\ndata: last";
    var input: std.Io.Reader = .fixed(bytes);
    var events = Iterator.init(arena.allocator(), &input, bytes.len);
    defer events.deinit();
    try std.testing.expectEqualStrings("first\xef\xbf\xbd\n\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd", (try events.next()).?);
    try std.testing.expectEqualStrings("last", (try events.next()).?);
    try std.testing.expectEqual(@as(?[]const u8, null), try events.next());
    var limited_input: std.Io.Reader = .fixed(bytes);
    var limited = Iterator.init(arena.allocator(), &limited_input, 8);
    defer limited.deinit();
    try std.testing.expectError(error.StreamTooLong, limited.next());
}
