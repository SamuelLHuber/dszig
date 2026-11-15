//! DSPy 3.5 File construction and canonical lm15 document serialization.
//! Constructors validate model fields, not media: malformed inline data is
//! deliberately rejected only by format/serialize, just as it is upstream.
const std = @import("std");
const pyjson = @import("pyjson.zig");
const unicode = @import("unicode.zig");
const mime_types = @import("mime_types.zig");
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");
const Value = runtime.Value;
const Writer = std.Io.Writer;

pub const File = struct {
    file_data: ?[]const u8 = null,
    file_id: ?[]const u8 = null,
    filename: ?[]const u8 = null,

    pub const Error = error{ InvalidFile, InvalidFileDataUri, InvalidFileBase64 };
    pub const Options = struct {
        filename: ?[]const u8 = null,
        mime_type: []const u8 = "application/octet-stream",
    };
    pub const PathOptions = struct {
        filename: ?[]const u8 = null,
        mime_type: ?[]const u8 = null,
    };

    pub const schema_description = "A file input type for DSPy.\nSee https://platform.openai.com/docs/api-reference/chat/create#chat_create-messages-user_message-content-array_of_content_parts-file_content_part-file for specification.\n\nThe file_data field should be a data URI with the format:\n    data:<mime_type>;base64,<base64_encoded_data>\n\nConstruct from in-memory values only: raw ``bytes``, a data URI string, a\n``{\"file_data\"|\"file_id\"|\"filename\"}`` dict, or another ``File``. Construction and adapter\nparsing never access the filesystem; use :meth:`from_path` to read a local file,\n:meth:`from_bytes` for raw bytes, or :meth:`from_file_id` to reference a preuploaded file.\n\nExamples:\n    ```python\n    import dspy\n\n    class QA(dspy.Signature):\n        file: dspy.File = dspy.InputField()\n        summary = dspy.OutputField()\n    program = dspy.Predict(QA)\n    result = program(file=dspy.File.from_path(\"./research.pdf\"))\n    print(result.summary)\n    ```";

    pub const marker_start = "<<CUSTOM-TYPE-START-IDENTIFIER>>";
    pub const marker_end = "<<CUSTOM-TYPE-END-IDENTIFIER>>";

    /// Construct a fresh File from an optional positional source and keyword
    /// fields. Unlike validation, copying a File omits its null fields before
    /// merging keywords, and an entirely empty copy is invalid. No storage is
    /// allocated: the validated strings borrow their input storage.
    pub fn init(source: ?Value, keywords: Value) Error!File {
        if (keywords != .object) return error.InvalidFile;
        var values: [3]Value = @splat(.null);
        var present: [3]bool = @splat(false);
        if (source) |input| switch (input) {
            .custom => |custom| {
                const file = custom.get(File) orelse return error.InvalidFile;
                inline for (field_names, 0..) |name, index| {
                    if (@field(file, name)) |text| {
                        values[index] = .{ .string = text };
                        present[index] = true;
                    }
                }
            },
            .string => |text| {
                if (!std.mem.startsWith(u8, text, "data:")) return error.InvalidFile;
                values[0] = input;
                present[0] = true;
            },
            .object => |object| {
                if (object.count() == 0) return error.InvalidFile;
                for (object.keys(), object.values()) |key, value| {
                    const index = fieldIndex(key) orelse return error.InvalidFile;
                    values[index] = value;
                    present[index] = true;
                }
            },
            else => return error.InvalidFile,
        };
        for (keywords.object.keys(), keywords.object.values()) |key, value| {
            const index = fieldIndex(key) orelse return error.InvalidFile;
            if (present[index]) return error.InvalidFile;
            values[index] = value;
            present[index] = true;
        }
        if (!present[0] and !present[1] and !present[2]) return error.InvalidFile;
        return .{
            .file_data = try field(values[0]),
            .file_id = try field(values[1]),
            .filename = try field(values[2]),
        };
    }

    const field_names = .{ "file_data", "file_id", "filename" };

    fn fieldIndex(key: []const u8) ?usize {
        inline for (field_names, 0..) |name, index| {
            if (std.mem.eql(u8, key, name)) return index;
        }
        return null;
    }

    /// Borrow the input's UTF-8 strings, after Pydantic-compatible stripping.
    /// A positional string must start with data: BEFORE field stripping. An
    /// object needs a recognized key, but its nullable values need not describe
    /// usable media. Neither form touches the filesystem.
    pub fn fromValue(input: Value) Error!File {
        if (input == .custom) return (input.custom.get(File) orelse return error.InvalidFile).*;
        if (input == .string) {
            if (!std.mem.startsWith(u8, input.string, "data:")) return error.InvalidFile;
            return .{ .file_data = try field(.{ .string = input.string }) };
        }
        if (input != .object) return error.InvalidFile;
        var found = false;
        for (input.object.keys()) |key| {
            if (!std.mem.eql(u8, key, "file_data") and
                !std.mem.eql(u8, key, "file_id") and
                !std.mem.eql(u8, key, "filename")) return error.InvalidFile;
            found = true;
        }
        if (!found) return error.InvalidFile;
        return .{
            .file_data = try field(input.object.get("file_data") orelse .null),
            .file_id = try field(input.object.get("file_id") orelse .null),
            .filename = try field(input.object.get("filename") orelse .null),
        };
    }

    /// Borrow the supplied UTF-8 strings, after field validation/stripping.
    /// Empty IDs remain empty and are accepted here, but do not take priority
    /// over inline data when formatting.
    pub fn fromFileId(id: ?[]const u8, filename: ?[]const u8) Error!File {
        return .{
            .file_id = try field(optionalString(id)),
            .filename = try field(optionalString(filename)),
        };
    }

    /// Own the encoded data URI and optional stripped filename in arena.
    /// Encoding empty bytes succeeds; formatting that empty payload fails late.
    pub fn fromBytes(arena: Allocator, bytes: []const u8, options: Options) (Error || Allocator.Error)!File {
        if (!std.unicode.utf8ValidateSlice(options.mime_type)) return error.InvalidFile;
        const filename = try ownedFilename(arena, options.filename);
        errdefer if (filename) |name| arena.free(name);
        const size = try uriSize(options.mime_type.len, bytes.len);
        const uri = try arena.alloc(u8, size);
        @memcpy(uri[0..5], "data:");
        @memcpy(uri[5..][0..options.mime_type.len], options.mime_type);
        const suffix = 5 + options.mime_type.len;
        @memcpy(uri[suffix..][0..8], ";base64,");
        _ = std.base64.standard.Encoder.encode(uri[suffix + 8 ..], bytes);
        return .{ .file_data = uri, .filename = filename };
    }

    /// Read a real regular file (following symlinks), with pinned POSIX basename
    /// and CPython MIME defaults. Both the URI and filename belong to arena.
    /// I/O errors after the isfile-equivalent check retain their native errors;
    /// failed isfile checks report FileNotFound rather than constructing media.
    pub fn fromPath(arena: Allocator, io: std.Io, path: []const u8, options: PathOptions) !File {
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.FileNotFound;
        const directory = std.Io.Dir.cwd();
        const stat = directory.statFile(io, path, .{}) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => return error.FileNotFound,
        };
        if (stat.kind != .file) return error.FileNotFound;
        const file = try directory.openFile(io, path, .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        var input: [3 * 1024]u8 = undefined;
        // Resolve a pure MIME result before streaming so the URI can be written
        // directly, but preserve source ordering: a read failure precedes any
        // MIME-guess failure. No raw-file-sized allocation survives in arena.
        const guessed_mime: mime_types.Error![]const u8 = if (options.mime_type) |explicit|
            explicit
        else result: {
            const guessed = mime_types.guessType(arena, path, true) catch |err| break :result err;
            break :result guessed.mime_type orelse "application/octet-stream";
        };
        const media_type = guessed_mime catch |err| {
            while ((reader.interface.readSliceShort(&input) catch return reader.err.?) != 0) {}
            return err;
        };
        const file_size = std.math.cast(usize, stat.size) orelse return error.OutOfMemory;
        var output: Writer.Allocating = try .initCapacity(arena, try uriSize(media_type.len, file_size));
        errdefer output.deinit();
        output.writer.writeAll("data:") catch return error.OutOfMemory;
        output.writer.writeAll(media_type) catch return error.OutOfMemory;
        output.writer.writeAll(";base64,") catch return error.OutOfMemory;
        while (true) {
            const count = reader.interface.readSliceShort(&input) catch return reader.err.?;
            if (count == 0) break;
            const destination = output.writer.writableSlice(std.base64.standard.Encoder.calcSize(count)) catch return error.OutOfMemory;
            _ = std.base64.standard.Encoder.encode(destination, input[0..count]);
            // readSliceShort only returns short at EOF; full chunks are always
            // divisible by three, so padding appears only on the final chunk.
            if (count != input.len) break;
        }
        if (!std.unicode.utf8ValidateSlice(media_type)) return error.InvalidFile;
        const basename_start = if (std.mem.lastIndexOfScalar(u8, path, '/')) |index| index + 1 else 0;
        const filename = try ownedFilename(arena, options.filename orelse path[basename_start..]);
        return .{ .file_data = output.written(), .filename = filename };
    }

    /// Retain File class identity in erased records and tool results. Fields
    /// borrow their existing storage; cloning the runtime value owns a copy.
    pub fn toRuntime(self: File, arena: Allocator) Allocator.Error!Value {
        const boxed = try arena.create(File);
        boxed.* = self;
        return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(File),
        .model_name = "File",
        .is_type = true,
        .hashable = true,
        .raw_value = rawRuntimeValue,
        .serialized = serializedRuntimeValue,
        .write_str = writeRuntimeStr,
        .write_repr = writeRuntimeRepr,
        .write_json = writeRuntimeJson,
        .write_serialized_json = writeRuntimeSerializedJson,
        .clone = cloneRuntime,
        .deinit = deinitRuntime,
        .eql = equalRuntime,
    };

    fn runtimeFile(ptr: *const anyopaque) *const File {
        return @ptrCast(@alignCast(ptr));
    }

    fn rawRuntimeValue(arena: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return runtimeFile(ptr).toValue(arena);
    }

    fn serializedRuntimeValue(arena: Allocator, ptr: *const anyopaque) runtime.CustomError!Value {
        return .{ .string = runtimeFile(ptr).serialize(arena) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        } };
    }

    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        runtimeFile(ptr).writeSerialized(writer) catch |err| return switch (err) {
            error.WriteFailed => error.WriteFailed,
            else => error.ParseValue,
        };
    }

    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        try runtimeFile(ptr).writeRepr(writer);
    }

    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(runtimeFile(ptr).*);
    }

    fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, _: usize) runtime.CustomWriteError!void {
        try writer.writeByte('"');
        var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
        try writeRuntimeStr(&escaped.writer, ptr);
        try writer.writeByte('"');
    }

    fn cloneRuntime(arena: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const source = runtimeFile(ptr);
        const copy = try arena.create(File);
        copy.* = .{};
        errdefer {
            inline for (.{ "file_data", "file_id", "filename" }) |name| {
                if (@field(copy.*, name)) |text| arena.free(text);
            }
            arena.destroy(copy);
        }
        inline for (.{ "file_data", "file_id", "filename" }) |name| {
            @field(copy.*, name) = if (@field(source.*, name)) |text| try arena.dupe(u8, text) else null;
        }
        return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
    }

    fn deinitRuntime(arena: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        const file = runtimeFile(ptr);
        if (owns_data) {
            inline for (.{ "file_data", "file_id", "filename" }) |name| {
                if (@field(file.*, name)) |text| arena.free(text);
            }
        }
        arena.destroy(file);
    }

    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) bool {
        const a = runtimeFile(left);
        const b = runtimeFile(right);
        inline for (.{ "file_data", "file_id", "filename" }) |name| {
            const av = @field(a.*, name);
            const bv = @field(b.*, name);
            if (av == null or bv == null) {
                if (av != null or bv != null) return false;
            } else if (!std.mem.eql(u8, av.?, bv.?)) return false;
        }
        return true;
    }

    /// Round-trip model dump: all three RAW nullable fields, never a marker.
    /// Container storage is allocated in arena; field strings remain borrowed.
    pub fn toValue(self: File, arena: Allocator) Allocator.Error!Value {
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.ensureTotalCapacity(arena, 3);
        object.putAssumeCapacity("file_data", optionalString(self.file_data));
        object.putAssumeCapacity("file_id", optionalString(self.file_id));
        object.putAssumeCapacity("filename", optionalString(self.filename));
        return .{ .object = object };
    }

    /// The exact ordered part_to_dict(document(...)) representation. Filename
    /// is display-only. Validation may strip whitespace or a nested data URI
    /// while checking base64, but the canonical part retains the RAW payload.
    pub fn format(self: File, arena: Allocator) (Error || Allocator.Error)!Value {
        const part = try self.documentPart();
        var object: @import("runtime_value.zig").ObjectMap = .empty;
        try object.ensureTotalCapacity(arena, 3);
        object.putAssumeCapacity("type", .{ .string = "document" });
        object.putAssumeCapacity("media_type", .{ .string = part.media_type });
        object.putAssumeCapacity(if (part.file_id != null) "file_id" else "data", .{
            .string = part.file_id orelse part.data.?,
        });
        var array: @import("runtime_value.zig").Array = .init(arena);
        try array.ensureTotalCapacity(1);
        array.appendAssumeCapacity(.{ .object = object });
        return .{ .array = array };
    }

    /// Python str(File) / non-round-trip model dump. Write directly rather
    /// than allocating a temporary canonical JSON tree merely to render it.
    pub fn serialize(self: File, arena: Allocator) (Error || Allocator.Error)![]const u8 {
        var output: Writer.Allocating = .init(arena);
        errdefer output.deinit();
        self.writeSerialized(&output.writer) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            else => |failure| failure,
        };
        return output.written();
    }

    /// Write the marker without allocating an intermediate serialized string.
    pub fn writeSerialized(self: File, writer: *Writer) (Error || Writer.Error)!void {
        const part = try self.documentPart();
        try writer.writeAll(marker_start ++ "[{\"type\": \"document\", \"media_type\": ");
        try pyjson.writeJsonString(writer, part.media_type);
        try writer.writeAll(if (part.file_id != null) ", \"file_id\": " else ", \"data\": ");
        try pyjson.writeJsonString(writer, part.file_id orelse part.data.?);
        try writer.writeAll("}]" ++ marker_end);
    }

    /// File.__repr__ uses literal single quotes, NOT Python repr escaping.
    /// Its lengths count Unicode code points, not UTF-8 bytes. Even malformed
    /// data URIs retain the source's simple split-based display behavior.
    pub fn writeRepr(self: File, writer: *Writer) Writer.Error!void {
        try writer.writeAll("File(");
        var separator: []const u8 = "";
        if (self.file_data) |data| {
            if (std.mem.startsWith(u8, data, "data:")) {
                const after_colon = data[5..];
                const mime_end = std.mem.indexOfAny(u8, after_colon, ";:") orelse after_colon.len;
                const length_data = if (std.mem.indexOf(u8, data, "base64,")) |start| blk: {
                    const rest = data[start + 7 ..];
                    break :blk rest[0 .. std.mem.indexOf(u8, rest, "base64,") orelse rest.len];
                } else data;
                try writer.print("file_data=<DATA_URI({s}, {d} chars)>", .{ after_colon[0..mime_end], codepointCount(length_data) });
            } else {
                try writer.print("file_data=<DATA({d} chars)>", .{codepointCount(data)});
            }
            separator = ", ";
        }
        if (self.file_id) |id| {
            try writer.print("{s}file_id='{s}'", .{ separator, id });
            separator = ", ";
        }
        if (self.filename) |filename| try writer.print("{s}filename='{s}'", .{ separator, filename });
        try writer.writeByte(')');
    }

    const Document = struct {
        media_type: []const u8,
        data: ?[]const u8 = null,
        file_id: ?[]const u8 = null,
    };

    fn documentPart(self: File) Error!Document {
        if (self.file_id) |id| {
            if (id.len != 0) return .{ .media_type = "application/pdf", .file_id = id };
        }
        const uri = self.file_data orelse return error.InvalidFileDataUri;
        if (!std.mem.startsWith(u8, uri, "data:")) return error.InvalidFileDataUri;
        const comma = std.mem.indexOfScalar(u8, uri, ',') orelse return error.InvalidFileDataUri;
        const head = uri[5..comma];
        const payload = uri[comma + 1 ..];
        if (!std.mem.endsWith(u8, head, ";base64") or payload.len == 0) return error.InvalidFileDataUri;
        try validateBase64(payload);
        const media_type = head[0 .. head.len - 7];
        return .{ .media_type = if (media_type.len == 0) "application/pdf" else media_type, .data = payload };
    }
};

fn optionalString(text: ?[]const u8) Value {
    return if (text) |string| .{ .string = string } else .null;
}

fn field(input: Value) File.Error!?[]const u8 {
    return switch (input) {
        .null => null,
        .string => |text| blk: {
            if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidFile;
            break :blk strip(text, true);
        },
        else => error.InvalidFile,
    };
}

fn ownedFilename(arena: Allocator, filename: ?[]const u8) (File.Error || Allocator.Error)!?[]const u8 {
    return if (try field(optionalString(filename))) |name| try arena.dupe(u8, name) else null;
}

fn uriSize(mime_len: usize, bytes_len: usize) Allocator.Error!usize {
    const groups = (std.math.add(usize, bytes_len, 2) catch return error.OutOfMemory) / 3;
    const encoded_len = std.math.mul(usize, groups, 4) catch return error.OutOfMemory;
    const prefix_len = std.math.add(usize, mime_len, 13) catch return error.OutOfMemory;
    return std.math.add(usize, prefix_len, encoded_len) catch error.OutOfMemory;
}

/// Rust/Pydantic White_Space excludes U+001C..U+001F; CPython str.strip and
/// str.split include them. Both paths share the existing Unicode table.
fn isSpace(cp: u21, pydantic: bool) bool {
    return unicode.isSpace(cp) and !(pydantic and cp >= 0x1c and cp <= 0x1f);
}

fn strip(text: []const u8, pydantic: bool) []const u8 {
    var start: usize = 0;
    var end = text.len;
    while (start < end) {
        const len = std.unicode.utf8ByteSequenceLength(text[start]) catch break;
        if (len > end - start) break;
        const cp = std.unicode.utf8Decode(text[start..][0..len]) catch break;
        if (!isSpace(cp, pydantic)) break;
        start += len;
    }
    while (end > start) {
        var first = end - 1;
        while (first > start and text[first] & 0xc0 == 0x80) first -= 1;
        const cp = std.unicode.utf8Decode(text[first..end]) catch break;
        if (!isSpace(cp, pydantic)) break;
        end = first;
    }
    return text[start..end];
}

fn codepointCount(text: []const u8) usize {
    var count: usize = 0;
    for (text) |byte| {
        if (byte & 0xc0 != 0x80) count += 1;
    }
    return count;
}

/// lm15 _base64_payload and _validate_base64_data validate shape, not decoded
/// bytes or unused padding bits. Whitespace normalization is conditional: an
/// interior Unicode space alone does NOT trigger it; boundary whitespace or
/// any ASCII split whitespace does. Empty normalized payloads are accepted.
pub fn validateBase64(raw: []const u8) File.Error!void {
    var payload = raw;
    if (std.mem.startsWith(u8, payload, "data:")) {
        if (std.mem.indexOf(u8, payload, ";base64,")) |index| payload = payload[index + 8 ..];
    }
    const stripped = strip(payload, false);
    const normalize = stripped.len != payload.len or std.mem.indexOfAny(u8, stripped, " \n\r\t\x0b\x0c") != null;
    var count: usize = 0;
    var padding: usize = 0;
    var i: usize = 0;
    while (i < payload.len) {
        const len = std.unicode.utf8ByteSequenceLength(payload[i]) catch return error.InvalidFileBase64;
        if (len > payload.len - i) return error.InvalidFileBase64;
        const cp = std.unicode.utf8Decode(payload[i..][0..len]) catch return error.InvalidFileBase64;
        i += len;
        if (normalize and unicode.isSpace(cp)) continue;
        count += 1;
        if (cp == '=') {
            padding += 1;
            if (padding > 2) return error.InvalidFileBase64;
        } else if (padding != 0 or !((cp >= 'A' and cp <= 'Z') or
            (cp >= 'a' and cp <= 'z') or (cp >= '0' and cp <= '9') or cp == '+' or cp == '/'))
        {
            return error.InvalidFileBase64;
        }
    }
    if (count % 4 != 0) return error.InvalidFileBase64;
}

test "File runtime identity distinguishes raw dictionaries through nested serialization" {
    const value = @import("value.zig");
    const adapter = @import("adapter.zig");
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { attachment: File };
        pub const Output = struct { answer: []const u8 };
    });
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const file = try File.fromBytes(arena, "Hello", .{ .mime_type = "text/plain", .filename = "user filename.txt" });
    const actual = try value.toValue(arena, File, file);
    const dictionary = try file.toValue(arena);
    const raw_json = "{\"file_data\":\"data:text/plain;base64,SGVsbG8=\",\"file_id\":null,\"filename\":\"user filename.txt\"}";
    const marker = File.marker_start ++ "[{\"type\": \"document\", \"media_type\": \"text/plain\", \"data\": \"SGVsbG8=\"}]" ++ File.marker_end;

    try std.testing.expect(actual == .custom);
    try std.testing.expect(actual.custom.get(File) != null);
    try std.testing.expect(dictionary == .object);
    try std.testing.expect(!value.pyEqual(actual, dictionary));
    try std.testing.expect(!value.pyEqual(dictionary, actual));
    try std.testing.expectEqualStrings(raw_json, try std.json.Stringify.valueAlloc(arena, actual, .{}));
    try std.testing.expectEqualStrings(raw_json, try std.json.Stringify.valueAlloc(arena, dictionary, .{}));
    const formatted = try adapter.formattedValue(arena, Sig.input_fields_list[0], actual);
    try std.testing.expect(formatted == .string);
    try std.testing.expectEqualStrings(marker, formatted.string);
    const formatted_dictionary = try adapter.formattedValue(arena, Sig.input_fields_list[0], dictionary);
    try std.testing.expect(formatted_dictionary == .object);
    try std.testing.expect(value.pyEqual(dictionary, formatted_dictionary));

    var files: runtime.Array = .init(arena);
    try files.append(actual);
    try files.append(dictionary);
    try files.append(actual);
    var nested: runtime.ObjectMap = .empty;
    try nested.put(arena, "files", .{ .array = files });
    const input: Value = .{ .object = nested };
    try std.testing.expectEqualStrings(
        "{\"files\":[" ++ raw_json ++ "," ++ raw_json ++ "," ++ raw_json ++ "]}",
        try std.json.Stringify.valueAlloc(arena, input, .{}),
    );
    const serialized = try value.serializedValue(arena, input);
    const serialized_files = serialized.object.get("files").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), serialized_files.len);
    try std.testing.expectEqualStrings(marker, serialized_files[0].string);
    try std.testing.expect(serialized_files[1] == .object);
    try std.testing.expect(value.pyEqual(dictionary, serialized_files[1]));
    try std.testing.expectEqualStrings(marker, serialized_files[2].string);
    // Serialization must not replace the borrowed original graph in place.
    try std.testing.expectEqual(actual.custom.ptr, files.items[0].custom.ptr);
    try std.testing.expectEqual(actual.custom.ptr, files.items[2].custom.ptr);
    var streamed: Writer.Allocating = .init(arena);
    try pyjson.writeJson(&streamed.writer, input, .minified);
    try std.testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, serialized, .{}),
        streamed.written(),
    );

    var plain_files: runtime.Array = .init(arena);
    try plain_files.append(dictionary);
    var plain_nested: runtime.ObjectMap = .empty;
    try plain_nested.put(arena, "files", .{ .array = plain_files });
    const plain_input: Value = .{ .object = plain_nested };
    const plain_serialized = try value.serializedValue(arena, plain_input);
    try std.testing.expect(value.pyEqual(plain_input, plain_serialized));
    try std.testing.expectEqual(plain_files.items.ptr, plain_serialized.object.get("files").?.array.items.ptr);
    var plain_streamed: Writer.Allocating = .init(arena);
    try pyjson.writeJson(&plain_streamed.writer, plain_input, .minified);
    try std.testing.expectEqualStrings("{\"files\":[" ++ raw_json ++ "]}", plain_streamed.written());

    const escaped = try File.fromFileId("file-\"\\\n雪\x00-id", "q\".txt");
    const escaped_runtime = try escaped.toRuntime(arena);
    var escaped_streamed: Writer.Allocating = .init(arena);
    try pyjson.writeJson(&escaped_streamed.writer, escaped_runtime, .compact);
    const escaped_marker = try escaped.serialize(arena);
    try std.testing.expectEqualStrings(
        try std.json.Stringify.valueAlloc(arena, escaped_marker, .{}),
        escaped_streamed.written(),
    );
    const decoded = try std.json.parseFromSliceLeaky(Value, arena, escaped_streamed.written(), .{});
    try std.testing.expectEqualStrings(escaped_marker, decoded.string);
}

test "File construction distinguishes nullable instance copies from validation and dictionaries" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const empty: Value = .{ .object = .empty };
    const nullable = try (File{}).toRuntime(arena);
    try std.testing.expectError(error.InvalidFile, File.init(nullable, empty));
    const nullable_fields = try (File{}).toValue(arena);
    const from_fields = try File.init(null, nullable_fields);
    try std.testing.expectEqual(@as(?[]const u8, null), from_fields.file_data);
    try std.testing.expectEqual(@as(?[]const u8, null), from_fields.file_id);
    try std.testing.expectEqual(@as(?[]const u8, null), from_fields.filename);

    const original = try File.fromFileId(" file-copy ", null);
    const instance = try original.toRuntime(arena);
    var keywords: runtime.ObjectMap = .empty;
    try keywords.put(arena, "filename", .{ .string = " additional.txt " });
    const named = try File.init(instance, .{ .object = keywords });
    try std.testing.expectEqualStrings("file-copy", named.file_id.?);
    try std.testing.expectEqualStrings("additional.txt", named.filename.?);
    try std.testing.expectEqual(@as(?[]const u8, null), original.filename);
    const fresh = try named.toRuntime(arena);
    try std.testing.expect(fresh.custom.ptr != instance.custom.ptr);

    // A dictionary retains explicit null keys, so its filename collides;
    // an actual instance omits that null field before merging keywords.
    const dictionary = try original.toValue(arena);
    try std.testing.expectError(error.InvalidFile, File.init(dictionary, .{ .object = keywords }));
    try keywords.put(arena, "file_id", .{ .string = "replacement" });
    try std.testing.expectError(error.InvalidFile, File.init(instance, .{ .object = keywords }));
    try std.testing.expectError(error.InvalidFile, File.init(null, empty));
}

test "File validation preserves actual instances and lookalikes do not acquire identity" {
    const value = @import("value.zig");
    const types = @import("types.zig");
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const file = try File.fromBytes(arena, "Hello", .{ .mime_type = "text/plain", .filename = "user filename.txt" });
    const actual = try file.toRuntime(arena);
    const validated = try types.validate(File, arena, actual);
    try std.testing.expectEqual(actual.custom.ptr, validated.custom.ptr);
    try std.testing.expectEqual(actual.custom.vtable, validated.custom.vtable);

    const dictionary = try file.toValue(arena);
    const coerced = try types.validate(File, arena, dictionary);
    try std.testing.expect(coerced == .custom);
    try std.testing.expect(coerced.custom.ptr != actual.custom.ptr);
    try std.testing.expect(value.pyEqual(actual, coerced));
    // Validation returns a model; it must not mutate the caller's dictionary.
    try std.testing.expect(dictionary == .object);
    try std.testing.expect(!value.pyEqual(dictionary, actual));
    const converted = try value.fromValue(arena, File, actual);
    try std.testing.expectEqualStrings(file.file_data.?, converted.file_data.?);
    try std.testing.expectEqual(file.file_data.?.ptr, converted.file_data.?.ptr);
}

test "File runtime clone owns fields after source arena release" {
    const value = @import("value.zig");
    const inference = @import("inference.zig");
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);
    const allocator = debug_allocator.allocator();
    var source: std.heap.ArenaAllocator = .init(std.testing.allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const temporary = try File.fromBytes(source.allocator(), "Hello", .{ .mime_type = "text/plain", .filename = "user filename.txt" });
    var files: runtime.Array = .init(source.allocator());
    try files.append(try temporary.toRuntime(source.allocator()));
    try files.append(try temporary.toValue(source.allocator()));
    var nested: runtime.ObjectMap = .empty;
    try nested.put(source.allocator(), "files", .{ .array = files });
    const borrowed: Value = .{ .object = nested };
    const owned = try inference.cloneJsonValue(allocator, borrowed);
    defer inference.freeJsonValue(allocator, owned);
    const owned_files = owned.object.get("files").?.array.items;
    try std.testing.expect(owned_files[0].custom.ptr != files.items[0].custom.ptr);
    try std.testing.expect(owned_files[0].custom.owns_data);
    try std.testing.expect(!files.items[0].custom.owns_data);
    source.deinit();
    source_live = false;

    var check: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer check.deinit();
    const arena = check.allocator();
    const expected = try File.fromBytes(arena, "Hello", .{ .mime_type = "text/plain", .filename = "user filename.txt" });
    const owned_file = owned_files[0].custom.get(File).?.*;
    try std.testing.expectEqualStrings(expected.file_data.?, owned_file.file_data.?);
    try std.testing.expectEqual(@as(?[]const u8, null), owned_file.file_id);
    try std.testing.expectEqualStrings(expected.filename.?, owned_file.filename.?);
    try std.testing.expect(value.pyEqual(try expected.toRuntime(arena), owned_files[0]));
    try std.testing.expect(owned_files[1] == .object);
    try std.testing.expect(value.pyEqual(try expected.toValue(arena), owned_files[1]));
    try std.testing.expect(!value.pyEqual(owned_files[0], owned_files[1]));
}

test "all-null File constructs but rejects when formatted" {
    const value = @import("value.zig");
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const parsed = try std.json.parseFromSliceLeaky(Value, arena, "{\"file_data\":null,\"file_id\":null,\"filename\":null}", .{});
    const invalid = try File.fromValue(parsed);
    try std.testing.expect(invalid.file_data == null and invalid.file_id == null and invalid.filename == null);
    const runtime_value = try invalid.toRuntime(arena);
    try std.testing.expectEqualStrings(
        "{\"file_data\":null,\"file_id\":null,\"filename\":null}",
        try std.json.Stringify.valueAlloc(arena, runtime_value, .{}),
    );
    try std.testing.expectError(error.InvalidFileDataUri, invalid.format(arena));
    try std.testing.expectError(error.InvalidFileDataUri, invalid.serialize(arena));
    try std.testing.expectError(error.ParseValue, value.serializedValue(arena, runtime_value));
    var streamed: Writer.Allocating = .init(arena);
    try std.testing.expectError(error.ParseValue, pyjson.writeJson(&streamed.writer, runtime_value, .compact));
}
