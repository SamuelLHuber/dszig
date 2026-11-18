//! DSPy Image stores a URL reference or a data URI. Construction never fetches
//! a URL or opens a path; inline media validation is deferred to formatting.
const std = @import("std");
const file_module = @import("file.zig");
const File = file_module.File;
const mime_types = @import("mime_types.zig");
const pyjson = @import("pyjson.zig");
const runtime = @import("runtime_value.zig");
const unicode = @import("unicode.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Writer = std.Io.Writer;

pub const Image = struct {
    url: []const u8,

    pub const model_name = "Image";
    pub const Error = error{ InvalidImage, InvalidImageDataUri, InvalidImageBase64 };
    pub const marker_start = File.marker_start;
    pub const marker_end = File.marker_end;
    const max_legacy_wrappers = 64;

    /// Borrow a string, legacy URL dictionary's value, or genuine Image field.
    /// Legacy dictionaries can wrap another source, up to 64 levels.
    pub fn fromValue(arena: Allocator, input: Value) (Error || Allocator.Error)!Image {
        var source = input;
        for (0..max_legacy_wrappers + 1) |depth| {
            switch (source) {
                .string => |text| return fromString(arena, text),
                .object => |object| {
                    if (depth == max_legacy_wrappers) return error.InvalidImage;
                    source = object.get("url") orelse return error.InvalidImage;
                },
                .custom => |box| return if (box.get(Image)) |model| model.* else error.InvalidImage,
                else => return error.InvalidImage,
            }
        }
        unreachable;
    }

    /// Validate a positional URL/URI before stripping its model-field whitespace.
    /// Lowercase `data:` is recognized without inspecting the inline payload.
    pub fn fromString(arena: Allocator, text: []const u8) (Error || Allocator.Error)!Image {
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidImage;
        if (!std.mem.startsWith(u8, text, "data:")) {
            const authority = remoteAuthority(text) orelse return error.InvalidImage;
            // MIME guessing shares urlparse's bracket and Unicode authority checks.
            // Ordinary ASCII authorities need no MIME lookup during construction.
            for (authority) |byte| {
                if (byte == '[' or byte == ']' or byte >= 0x80) {
                    _ = try guessMime(arena, text);
                    break;
                }
            }
        }
        return .{ .url = unicode.strip(text, true) };
    }

    /// Encode already-encoded media file bytes, without decoding or inspecting them.
    /// The caller supplies the MIME type and inclusive input byte limit. The URI
    /// belongs to arena; empty bytes construct a URI that fails at formatting.
    pub fn fromEncodedBytes(
        arena: Allocator,
        bytes: []const u8,
        media_type: []const u8,
        max_bytes: usize,
    ) (Error || Allocator.Error || error{StreamTooLong})!Image {
        if (bytes.len > max_bytes) return error.StreamTooLong;
        if (media_type.len == 0 or std.mem.indexOfScalar(u8, media_type, ',') != null)
            return error.InvalidImage;
        const document = File.fromBytes(arena, bytes, .{ .mime_type = media_type }) catch |err|
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidImage,
            };
        return .{ .url = document.file_data.? };
    }

    /// Explicitly read a caller-trusted regular file (symlinks are followed).
    /// `max_bytes` is inclusive and checked again while reading in case of growth.
    /// Use the MIME table for the pathname, not image decoding or content sniffing.
    pub fn fromPath(arena: Allocator, io: std.Io, path: []const u8, max_bytes: usize) !Image {
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidImagePath;
        const dir = std.Io.Dir.cwd();
        const stat = try dir.statFile(io, path, .{});
        if (stat.kind != .file) return error.InvalidImagePath;
        if (stat.size > max_bytes) return error.StreamTooLong;
        const file = try dir.openFile(io, path, .{});
        defer file.close(io);
        const opened_stat = try file.stat(io);
        if (opened_stat.kind != .file) return error.InvalidImagePath;
        if (opened_stat.size > max_bytes) return error.StreamTooLong;
        // A local path that looks like a data URI can produce allocated MIME
        // scratch; keep that storage separate from the final URI.
        var scratch: std.heap.ArenaAllocator = .init(arena);
        defer scratch.deinit();
        const guessed = try guessMime(scratch.allocator(), path) orelse
            return error.UnknownImageMimeType;
        if (!std.unicode.utf8ValidateSlice(guessed)) return error.InvalidImage;
        return encodeFile(arena, io, file, guessed, @intCast(opened_stat.size), max_bytes);
    }

    fn encodeFile(
        arena: Allocator,
        io: std.Io,
        file: std.Io.File,
        media_type: []const u8,
        size_hint: usize,
        max_bytes: usize,
    ) !Image {
        const groups = size_hint / 3 + @intFromBool(size_hint % 3 != 0);
        const encoded_len = std.math.mul(usize, groups, 4) catch return error.OutOfMemory;
        const prefix_len = std.math.add(usize, media_type.len, 5 + base64_suffix.len + 1) catch
            return error.OutOfMemory;
        const capacity = std.math.add(usize, prefix_len, encoded_len) catch
            return error.OutOfMemory;
        var output: Writer.Allocating = try .initCapacity(arena, capacity);
        errdefer output.deinit();
        output.writer.writeAll("data:") catch return error.OutOfMemory;
        output.writer.writeAll(media_type) catch return error.OutOfMemory;
        output.writer.writeAll(base64_suffix ++ ",") catch return error.OutOfMemory;
        var reader = file.reader(io, &.{});
        // Complete chunks divide evenly into base64 groups; a short chunk is EOF.
        var input: [3 * 1024]u8 = undefined;
        var remaining = max_bytes;
        while (remaining != 0) {
            const requested = @min(remaining, input.len);
            const count = reader.interface.readSliceShort(input[0..requested]) catch
                return reader.err.?;
            if (count != 0) {
                const encoder = std.base64.standard.Encoder;
                const dest = output.writer.writableSlice(encoder.calcSize(count)) catch
                    return error.OutOfMemory;
                _ = encoder.encode(dest, input[0..count]);
            }
            if (count < requested) return .{ .url = try output.toOwnedSlice() };
            remaining -= count;
        }
        // Accept an exactly-full file, but reject a file that grew past the cap.
        if ((reader.interface.readSliceShort(input[0..1]) catch return reader.err.?) != 0)
            return error.StreamTooLong;
        return .{ .url = try output.toOwnedSlice() };
    }

    /// Round-trip model projection; the map is allocated but its URL is borrowed.
    pub fn toValue(self: Image, arena: Allocator) Allocator.Error!Value {
        var object: runtime.ObjectMap = .empty;
        errdefer object.deinit(arena);
        try object.put(arena, "url", .{ .string = self.url });
        return .{ .object = object };
    }

    /// Allocate a model box which borrows the URL; a runtime clone owns its copy.
    pub fn toRuntime(self: Image, arena: Allocator) Allocator.Error!Value {
        const box = try arena.create(Image);
        box.* = self;
        return .{ .custom = .{ .ptr = box, .vtable = &runtime_vtable } };
    }

    /// Borrow the typed slot as well as its URL. Deinit of the handle is a no-op.
    pub fn borrowedRuntime(self: *const Image) Value {
        return .{ .custom = .{ .ptr = self, .vtable = &runtime_vtable, .owns_box = false } };
    }

    /// Return the canonical one-element lm15 image part list, borrowing the URL
    /// or inline payload from this model and allocating the containers in arena.
    pub fn format(self: Image, arena: Allocator) (Error || Allocator.Error)!Value {
        const image_part = try self.part(arena);
        var object: runtime.ObjectMap = .empty;
        errdefer object.deinit(arena);
        try object.ensureTotalCapacity(arena, 3);
        object.putAssumeCapacity("type", .{ .string = "image" });
        object.putAssumeCapacity("media_type", .{ .string = image_part.media_type });
        object.putAssumeCapacity(@tagName(image_part.address_key), .{ .string = image_part.address });
        var array: runtime.Array = .init(arena);
        try array.ensureTotalCapacityPrecise(1);
        array.appendAssumeCapacity(.{ .object = object });
        return .{ .array = array };
    }

    /// Non-round-trip dump, with exactly the text used in a DSPy prompt.
    pub fn serialize(self: Image, arena: Allocator) (Error || Allocator.Error)![]const u8 {
        const image_part = try self.part(arena);
        var output: Writer.Allocating = .init(arena);
        errdefer output.deinit();
        image_part.write(&output.writer) catch return error.OutOfMemory;
        return output.toOwnedSlice();
    }

    /// Render directly to a writer. Only remote URLs containing tab/CR/LF may
    /// require temporary scratch space during MIME lookup.
    pub fn writeSerialized(self: Image, writer: *Writer) (Error || Writer.Error)!void {
        try (try self.partForWriter()).write(writer);
    }

    fn partForWriter(self: Image) (Error || Writer.Error)!Part {
        return self.part(std.heap.page_allocator) catch |err| return switch (err) {
            error.OutOfMemory => error.WriteFailed,
            else => |failure| failure,
        };
    }

    const default_media_type = "image/png";
    const base64_suffix = ";base64";
    const Part = struct {
        media_type: []const u8,
        address_key: enum { url, data },
        address: []const u8,

        fn write(self: Part, writer: *Writer) Writer.Error!void {
            try writer.writeAll(marker_start ++ "[{\"type\": \"image\", \"media_type\": ");
            try pyjson.writeJsonString(writer, self.media_type);
            try writer.writeAll(if (self.address_key == .data) ", \"data\": " else ", \"url\": ");
            try pyjson.writeJsonString(writer, self.address);
            try writer.writeAll("}]" ++ marker_end);
        }
    };

    fn part(self: Image, arena: Allocator) (Error || Allocator.Error)!Part {
        if (!std.unicode.utf8ValidateSlice(self.url)) return error.InvalidImage;
        if (std.mem.startsWith(u8, self.url, "data:")) {
            const body = self.url[5..];
            const comma = std.mem.indexOfScalar(u8, body, ',') orelse return error.InvalidImageDataUri;
            const head = body[0..comma];
            const payload = body[comma + 1 ..];
            if (!std.mem.endsWith(u8, head, base64_suffix) or payload.len == 0)
                return error.InvalidImageDataUri;
            file_module.validateBase64(payload) catch return error.InvalidImageBase64;
            const media_type = head[0 .. head.len - base64_suffix.len];
            return .{
                .media_type = if (media_type.len == 0) default_media_type else media_type,
                .address_key = .data,
                .address = payload,
            };
        }
        _ = remoteAuthority(self.url) orelse return error.InvalidImage;
        const mime = (try guessMime(arena, self.url)) orelse default_media_type;
        return .{
            .media_type = if (std.mem.startsWith(u8, mime, "image/")) mime else default_media_type,
            .address_key = .url,
            .address = self.url,
        };
    }

    fn guessMime(arena: Allocator, text: []const u8) (Error || Allocator.Error)!?[]const u8 {
        const guess = mime_types.guessType(arena, text, true) catch |err| return switch (err) {
            error.InvalidMimePath => error.InvalidImage,
            error.OutOfMemory => error.OutOfMemory,
        };
        return guess.mime_type;
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(Image),
        .type_identity = runtime.typeIdentity(Image),
        .model_name = model_name,
        .is_type = true,
        .hashable = true,
        .raw_value = rawRuntimeValue,
        .model_dump = modelDumpRuntimeValue,
        .serialized = serializedRuntimeValue,
        .write_str = writeRuntimeStr,
        .write_repr = writeRuntimeRepr,
        .write_json = writeRuntimeJson,
        .write_serialized_json = writeRuntimeSerializedJson,
        .clone = cloneRuntime,
        .deinit = deinitRuntime,
        .eql = equalRuntime,
    };

    fn runtimeImage(ptr: *const anyopaque) *const Image {
        return @ptrCast(@alignCast(ptr));
    }

    fn rawRuntimeValue(arena: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return runtimeImage(ptr).toValue(arena);
    }

    fn modelDumpRuntimeValue(arena: Allocator, ptr: *const anyopaque) runtime.CustomError!Value {
        return serializedRuntimeValue(arena, ptr, 0);
    }

    fn serializedRuntimeValue(arena: Allocator, ptr: *const anyopaque, _: u32) runtime.CustomError!Value {
        return .{ .string = runtimeImage(ptr).serialize(arena) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        } };
    }

    fn runtimePart(ptr: *const anyopaque) runtime.CustomWriteError!Part {
        return runtimeImage(ptr).partForWriter() catch |err| switch (err) {
            error.WriteFailed => error.WriteFailed,
            else => error.ParseValue,
        };
    }

    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        try (try runtimePart(ptr)).write(writer);
    }

    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, _: u32) runtime.CustomWriteError!void {
        const url = runtimeImage(ptr).url;
        if (std.mem.indexOf(u8, url, "base64") != null) {
            const separator = "base64,";
            const start = std.mem.indexOf(u8, url, separator) orelse return error.ParseValue;
            const rest = url[start + separator.len ..];
            const payload = rest[0 .. std.mem.indexOf(u8, rest, separator) orelse rest.len];
            const head = url[0 .. std.mem.indexOfScalar(u8, url, ';') orelse url.len];
            const subtype = head[if (std.mem.lastIndexOfScalar(u8, head, '/')) |i| i + 1 else 0..];
            var count: usize = 0;
            for (payload) |byte| if (byte & 0xc0 != 0x80) {
                count += 1;
            };
            try writer.print("Image(url=data:image/{s};base64,<IMAGE_BASE_64_ENCODED({d})>)", .{
                subtype, count,
            });
        } else {
            try writer.print("Image(url='{s}')", .{url});
        }
    }

    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(runtimeImage(ptr).*);
    }

    fn writeRuntimeSerializedJson(
        writer: *Writer,
        ptr: *const anyopaque,
        _: runtime.JsonStyle,
        _: usize,
    ) runtime.CustomWriteError!void {
        const image_part = try runtimePart(ptr);
        try writer.writeByte('"');
        var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
        try image_part.write(&escaped.writer);
        try writer.writeByte('"');
    }

    fn cloneRuntime(
        context: *runtime.CloneContext,
        ptr: *const anyopaque,
    ) Allocator.Error!runtime.Custom {
        const arena = context.allocator;
        const box = try arena.create(Image);
        errdefer arena.destroy(box);
        box.* = .{ .url = try arena.dupe(u8, runtimeImage(ptr).url) };
        return .{ .ptr = box, .vtable = &runtime_vtable, .owns_data = true };
    }

    fn deinitRuntime(arena: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        const box = runtimeImage(ptr);
        if (owns_data) arena.free(box.url);
        arena.destroy(box);
    }

    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
        return std.mem.eql(u8, runtimeImage(left).url, runtimeImage(right).url);
    }
};

/// Check the scheme and nonempty authority as urlparse sees them, preserving
/// the original field for output. Do not require valid DNS or a numeric port.
fn remoteAuthority(text: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < text.len and text[i] <= 0x20) : (i += 1) {}
    var scheme: [5]u8 = undefined;
    var len: usize = 0;
    while (i < text.len and text[i] != ':') : (i += 1) {
        const byte = text[i];
        if (ignoredUrlByte(byte)) continue;
        if (len == scheme.len) return null;
        scheme[len] = std.ascii.toLower(byte);
        len += 1;
    }
    if (i == text.len) return null;
    const name = scheme[0..len];
    if (!std.mem.eql(u8, name, "http") and !std.mem.eql(u8, name, "https") and
        !std.mem.eql(u8, name, "gs")) return null;
    i += 1;
    for (0..2) |_| {
        while (i < text.len and ignoredUrlByte(text[i])) : (i += 1) {}
        if (i == text.len or text[i] != '/') return null;
        i += 1;
    }
    const tail = text[i..];
    const authority = tail[0 .. std.mem.indexOfAny(u8, tail, "/?#") orelse tail.len];
    for (authority) |byte| if (!ignoredUrlByte(byte)) return authority;
    return null;
}

fn ignoredUrlByte(byte: u8) bool {
    return byte == '\t' or byte == '\r' or byte == '\n';
}

test "Image matches the fourteen frozen source rows" {
    const testing = std.testing;
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const legacy_url = "https://example.invalid/z.webp";
    var legacy: runtime.ObjectMap = .empty;
    try legacy.put(arena, "url", .{ .string = legacy_url });
    const valid = [_]struct {
        url: []const u8,
        mime: []const u8,
        data: ?[]const u8 = null,
    }{
        .{ .url = "https://example.invalid/map.PNG", .mime = "image/png" },
        .{ .url = "https://example.invalid/readme.txt", .mime = "image/png" },
        .{ .url = "gs://bucket/view.jpg", .mime = "image/jpeg" },
        .{ .url = "HTTP://example.invalid/image.gif", .mime = "image/gif" },
        .{ .url = "https://example.invalid/map.png?width=3", .mime = "image/png" },
        .{ .url = legacy_url, .mime = "image/webp" },
        .{ .url = "data:image/png;base64,YWJj", .mime = "image/png", .data = "YWJj" },
        .{ .url = "data:;base64,YWJj", .mime = "image/png", .data = "YWJj" },
        .{ .url = "data:text/plain;base64,YWJj", .mime = "text/plain", .data = "YWJj" },
    };
    for (valid) |row| {
        const input: Value = if (std.mem.eql(u8, row.url, legacy_url))
            .{ .object = legacy }
        else
            .{ .string = row.url };
        const image = try Image.fromValue(arena, input);
        try expectRawImageUrl(arena, image, row.url);
        const parts = try image.format(arena);
        try testing.expectEqual(@as(usize, 1), parts.array.items.len);
        const object = parts.array.items[0].object;
        const address_key = if (row.data != null) "data" else "url";
        const address = row.data orelse row.url;
        const keys = [_][]const u8{ "type", "media_type", address_key };
        try testing.expectEqual(keys.len, object.count());
        for (keys, object.keys()) |expected, actual| try testing.expectEqualStrings(expected, actual);
        try testing.expectEqualStrings("image", object.get("type").?.string);
        try testing.expectEqualStrings(row.mime, object.get("media_type").?.string);
        try testing.expectEqualStrings(address, object.get(address_key).?.string);
        const rendered = try std.mem.concat(arena, u8, &.{
            "<<CUSTOM-TYPE-START-IDENTIFIER>>[{\"type\": \"image\", \"media_type\": \"",
            row.mime,
            "\", \"",
            address_key,
            "\": \"",
            address,
            "\"}]<<CUSTOM-TYPE-END-IDENTIFIER>>",
        });
        try testing.expectEqualStrings(rendered, try image.serialize(arena));
    }
    const invalid_inline = [_]struct { url: []const u8, err: Image.Error }{
        .{ .url = "data:image/png;base64,@!", .err = error.InvalidImageBase64 },
        .{ .url = "data:image/png;base64,", .err = error.InvalidImageDataUri },
        .{ .url = "data:image/png,YWJj", .err = error.InvalidImageDataUri },
    };
    for (invalid_inline) |row| {
        const image = try Image.fromValue(arena, .{ .string = row.url });
        try expectRawImageUrl(arena, image, row.url);
        try testing.expectError(row.err, image.format(arena));
        try testing.expectError(row.err, image.serialize(arena));
    }
    for ([_][]const u8{ "map.png", "https:///map.png" }) |url| {
        try testing.expectError(error.InvalidImage, Image.fromValue(arena, .{ .string = url }));
    }
}

fn expectRawImageUrl(arena: Allocator, image: Image, url: []const u8) !void {
    const raw = (try image.toValue(arena)).object;
    try std.testing.expectEqual(@as(usize, 1), raw.count());
    try std.testing.expectEqualStrings(url, raw.get("url").?.string);
}

test "Image runtime preserves model identity, raw and prompt projections" {
    const testing = std.testing;
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const image = try Image.fromString(arena, "  https://example.invalid/z.webp?\"雪\\=1  ");
    const actual = (try image.toRuntime(arena)).custom;
    const borrowed = image.borrowedRuntime().custom;
    borrowed.deinit(testing.allocator);
    try testing.expectEqual(&image, borrowed.get(Image).?);
    try testing.expect(try actual.eql(borrowed));
    try testing.expectEqualStrings(image.url, (try Image.fromValue(arena, .{ .custom = actual })).url);
    const raw = try actual.rawValue(arena);
    try testing.expectEqualStrings(image.url, raw.object.get("url").?.string);
    const raw_json = "{\"url\":\"https://example.invalid/z.webp?\\\"雪\\\\=1\"}";
    try testing.expectEqualStrings(raw_json, try std.json.Stringify.valueAlloc(
        arena,
        Value{ .custom = actual },
        .{},
    ));
    const rendered = Image.marker_start ++ "[{\"type\": \"image\", \"media_type\": \"image/webp\"," ++
        " \"url\": \"https://example.invalid/z.webp?\\\"雪\\\\=1\"}]" ++ Image.marker_end;
    try testing.expectEqualStrings(rendered, (try actual.serialize(arena)).string);
    try testing.expectEqualStrings(rendered, (try actual.modelDump(arena)).string);
    var stream: Writer.Allocating = .init(arena);
    try image.writeSerialized(&stream.writer);
    try testing.expectEqualStrings(rendered, stream.written());
    stream.clearRetainingCapacity();
    try actual.writeSerializedJson(&stream.writer, .compact, 0);
    const parsed = try std.json.parseFromSliceLeaky(Value, arena, stream.written(), .{});
    try testing.expectEqualStrings(rendered, parsed.string);
    stream.clearRetainingCapacity();
    const inline_image = try Image.fromString(arena, "data:image/png;base64,YWJj");
    try inline_image.borrowedRuntime().custom.writeRepr(&stream.writer);
    try testing.expectEqualStrings(
        "Image(url=data:image/png;base64,<IMAGE_BASE_64_ENCODED(4)>)",
        stream.written(),
    );
}

test "Image typed signature distinguishes a model from a raw URL dictionary" {
    const testing = std.testing;
    const types = @import("types.zig");
    const value = @import("value.zig");
    const adapter = @import("adapter.zig");
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { photo: Image };
        pub const Output = struct { answer: []const u8 };
    });
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();

    const schema = (try std.json.parseFromSliceLeaky(Value, arena, types.schema(Image), .{})).object;
    try testing.expectEqualStrings("Image", schema.get("title").?.string);
    try testing.expect(!schema.get("additionalProperties").?.bool);
    const properties = schema.get("properties").?.object;
    try testing.expectEqual(@as(usize, 1), properties.count());
    try testing.expectEqualStrings("string", properties.get("url").?.object.get("type").?.string);
    const required = schema.get("required").?.array.items;
    try testing.expectEqual(@as(usize, 1), required.len);
    try testing.expectEqualStrings("url", required[0].string);

    const image = try Image.fromString(arena, "gs://bucket/view.jpg");
    const raw = try image.toValue(arena);
    const boxed = try types.validate(Image, arena, raw);
    try testing.expect(boxed.custom.get(Image) != null);
    try testing.expectEqual(boxed.custom.ptr, (try types.validate(Image, arena, boxed)).custom.ptr);
    const projected = try value.fromValue(arena, Image, boxed);
    try testing.expectEqual(image.url.ptr, projected.url.ptr);
    const field = Sig.input_fields_list[0];
    const formatted = try adapter.formattedValue(arena, field, boxed);
    try testing.expectEqualStrings(try image.serialize(arena), formatted.string);
    try testing.expect((try adapter.formattedValue(arena, field, raw)) == .object);
}

test "Image bounds nested legacy sources without accepting foreign model boxes" {
    const testing = std.testing;
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const url = "https://example.invalid/p.png";
    var source: Value = .{ .string = url };
    for (0..64) |_| {
        var wrapper: runtime.ObjectMap = .empty;
        try wrapper.put(arena, "url", source);
        source = .{ .object = wrapper };
    }
    try testing.expectEqualStrings(url, (try Image.fromValue(arena, source)).url);
    var over_limit: runtime.ObjectMap = .empty;
    try over_limit.put(arena, "url", source);
    try testing.expectError(error.InvalidImage, Image.fromValue(arena, .{ .object = over_limit }));

    const foreign: File = .{ .file_id = url };
    try testing.expectError(error.InvalidImage, Image.fromValue(arena, foreign.borrowedRuntime()));
}

test "Image URL admission and late inline validation remain separate" {
    const testing = std.testing;
    const arena = testing.allocator;
    for ([_][]const u8{
        "data:;base64,YWJj",
        "http://a:invalid/path",
        "http://[::1]/a.png",
        "http://雪.invalid/a.png",
    }) |url| {
        const image = try Image.fromString(arena, url);
        try testing.expectEqualStrings(url, image.url);
    }
    for ([_][]const u8{
        "ftp://example.invalid/file.jpg",
        "http://[bad]/a.png",
        "http://a／b/a.png",
        "http://\n\t/path",
        " data:image/png;base64,YWJj",
        "DATA:image/png;base64,YWJj",
        "\xff",
    }) |url| {
        try testing.expectError(error.InvalidImage, Image.fromString(arena, url));
    }
    try testing.expectError(error.InvalidImage, Image.fromValue(arena, .{ .object = .empty }));
    try testing.expectError(error.InvalidImage, Image.fromValue(arena, .{ .integer = 7 }));
    // Model construction strips only outer whitespace, retaining raw payload
    // spacing and URL tabs even though each is normalized for validation.
    const url = try Image.fromString(arena, "\r HTTPS://ex\tample.invalid/a.JPEG?x=1 \n");
    try testing.expectEqualStrings("HTTPS://ex\tample.invalid/a.JPEG?x=1", url.url);
    const inline_image = try Image.fromString(arena, "data:image/png;base64,YW Jj");
    var stream: Writer.Allocating = .init(arena);
    defer stream.deinit();
    try url.writeSerialized(&stream.writer);
    try testing.expect(std.mem.indexOf(u8, stream.written(), "\"media_type\": \"image/jpeg\"") != null);
    try testing.expect(std.mem.indexOf(u8, stream.written(), "ex\\tample") != null);
    stream.clearRetainingCapacity();
    try inline_image.writeSerialized(&stream.writer);
    try testing.expect(std.mem.indexOf(u8, stream.written(), "\"data\": \"YW Jj\"") != null);
    const bad = try Image.fromString(arena, "data:image/png;base64,@!");
    stream.clearRetainingCapacity();
    try testing.expectError(
        error.ParseValue,
        bad.borrowedRuntime().custom.writeSerializedJson(&stream.writer, .compact, 0),
    );
    try testing.expectEqual(@as(usize, 0), stream.written().len);
}

test "Image explicit file and encoded bytes use inclusive bounded reads" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const encoded = try Image.fromEncodedBytes(allocator, "abc", "image/jpeg", 3);
    defer allocator.free(encoded.url);
    try testing.expectEqualStrings("data:image/jpeg;base64,YWJj", encoded.url);
    try testing.expectError(
        error.StreamTooLong,
        Image.fromEncodedBytes(allocator, "abc", "image/jpeg", 2),
    );
    try testing.expectError(
        error.InvalidImage,
        Image.fromEncodedBytes(allocator, "abc", "image/,jpeg", 3),
    );

    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "photo.JPG", .data = "abc" });
    const path = try temporary.dir.realPathFileAlloc(testing.io, "photo.JPG", allocator);
    defer allocator.free(path);
    try testing.expectError(error.InvalidImage, Image.fromString(allocator, path));
    const from_disk = try Image.fromPath(allocator, testing.io, path, 3);
    defer allocator.free(from_disk.url);
    try testing.expectEqualStrings(encoded.url, from_disk.url);
    try testing.expectError(error.StreamTooLong, Image.fromPath(allocator, testing.io, path, 2));
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "photo.unknown-extension", .data = "" });
    const unknown = try temporary.dir.realPathFileAlloc(testing.io, "photo.unknown-extension", allocator);
    defer allocator.free(unknown);
    try testing.expectError(
        error.UnknownImageMimeType,
        Image.fromPath(allocator, testing.io, unknown, 0),
    );
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "photo.JPG", .data = "" });
    const empty = try Image.fromPath(allocator, testing.io, path, 0);
    defer allocator.free(empty.url);
    try testing.expectError(error.InvalidImageDataUri, empty.serialize(allocator));

    const multiple_chunks: [3 * 1024 + 1]u8 = @splat('x');
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "photo.JPG", .data = &multiple_chunks });
    const streamed = try Image.fromPath(allocator, testing.io, path, multiple_chunks.len);
    defer allocator.free(streamed.url);
    const expected = try Image.fromEncodedBytes(
        allocator,
        &multiple_chunks,
        "image/jpeg",
        multiple_chunks.len,
    );
    defer allocator.free(expected.url);
    try testing.expectEqualStrings(expected.url, streamed.url);
    const open_file = try temporary.dir.openFile(testing.io, "photo.JPG", .{});
    defer open_file.close(testing.io);
    try testing.expectError(error.StreamTooLong, Image.encodeFile(
        allocator,
        testing.io,
        open_file,
        "image/jpeg",
        multiple_chunks.len - 1,
        multiple_chunks.len - 1,
    ));
}

test "Image explicit path rejects directories and releases failed output allocations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    try testing.expectError(error.InvalidImagePath, Image.fromPath(allocator, testing.io, ".", 7));
    try testing.expectError(error.InvalidImagePath, Image.fromPath(allocator, testing.io, "\x00", 7));

    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "photo.png", .data = "abc" });
    const path = try temporary.dir.realPathFileAlloc(testing.io, "photo.png", allocator);
    defer allocator.free(path);
    try testing.checkAllAllocationFailures(allocator, struct {
        fn exercise(failing: Allocator, path_to_read: []const u8) !void {
            const image = try Image.fromPath(failing, testing.io, path_to_read, 3);
            defer failing.free(image.url);
            try testing.expectEqualStrings("data:image/png;base64,YWJj", image.url);
        }
    }.exercise, .{path});
}

test "Image runtime clone owns one independent payload across aliases" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const alias_pair = blk: {
        var source: std.heap.ArenaAllocator = .init(allocator);
        defer source.deinit();
        const image = try Image.fromEncodedBytes(source.allocator(), "abc", "image/png", 3);
        const original = (try image.toRuntime(source.allocator())).custom;
        const copies = try cloneImageAliases(allocator, original);
        errdefer for (copies) |copy| copy.deinit(allocator);
        try testing.expect(copies[0].ptr != original.ptr);
        try testing.expect(copies[0].get(Image).?.url.ptr != image.url.ptr);
        break :blk copies;
    };
    defer alias_pair[1].deinit(allocator);
    alias_pair[0].deinit(allocator);
    const retained = alias_pair[1].get(Image).?;
    try testing.expectEqualStrings("data:image/png;base64,YWJj", retained.url);
    const independent = try alias_pair[1].clone(allocator);
    defer independent.deinit(allocator);
    try testing.expect(independent.ptr != alias_pair[1].ptr);
    try testing.expect(try independent.eql(alias_pair[1]));
    try testing.checkAllAllocationFailures(allocator, exerciseImageAllocations, .{});
}

fn cloneImageAliases(allocator: Allocator, source: runtime.Custom) ![2]runtime.Custom {
    var context = runtime.CloneContext.init(allocator);
    defer context.deinit();
    const first = try context.cloneCustom(source);
    errdefer first.deinit(allocator);
    const second = try context.cloneCustom(source);
    errdefer second.deinit(allocator);
    try std.testing.expectEqual(first.ptr, second.ptr);
    return .{ first, second };
}

fn exerciseImageAllocations(allocator: Allocator) !void {
    const image = try Image.fromEncodedBytes(allocator, "abc", "image/png", 3);
    defer allocator.free(image.url);
    const aliases = try cloneImageAliases(allocator, image.borrowedRuntime().custom);
    defer aliases[0].deinit(allocator);
    defer aliases[1].deinit(allocator);
    const rendered = try image.serialize(allocator);
    defer allocator.free(rendered);
    var parts = (try image.format(allocator)).array;
    defer parts.deinit();
    defer parts.items[0].object.deinit(allocator);
    var raw = (try image.toValue(allocator)).object;
    defer raw.deinit(allocator);
}
