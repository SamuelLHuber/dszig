//! DSPy 3.5 Audio for already encoded audio. Only explicit I/O constructors
//! access files or URLs. No constructor decodes, sniffs, or transcodes audio.
const std = @import("std");
const file_model = @import("file.zig");
const media_download = @import("media_download.zig");
const mime_types = @import("mime_types.zig");
const pyjson = @import("pyjson.zig");
const python = @import("python_value.zig");
const runtime = @import("runtime_value.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Writer = std.Io.Writer;

/// Immutable UTF-8 fields, borrowing validated strings where possible.
/// Ingress allocations belong to the supplied arena. Keep borrowed storage
/// alive and unchanged, or use clone for a fully independent Audio.
pub const Audio = struct {
    data: []const u8,
    audio_format: []const u8,

    pub const Error = error{ InvalidAudio, InvalidAudioDataUri, InvalidAudioBase64 };
    pub const Options = struct { audio_format: []const u8 = "wav" };
    pub const PathOptions = struct { max_bytes: usize };
    pub const UrlOptions = media_download.Options;
    pub const WavOptions = struct {
        sample_rate_hz: u32,
        channels: u16 = 1,
        /// Inclusive limit on the complete WAV before base64.
        max_bytes: usize,
        audio_format: []const u8 = "wav",
    };

    /// Samples are interleaved by frame; all four types produce PCM16.
    pub const Samples = union(enum) {
        s16: []const i16,
        s32: []const i32,
        f32: []const f32,
        f64: []const f64,

        fn len(self: Samples) usize {
            return switch (self) {
                inline else => |data| data.len,
            };
        }

        fn pcm16(self: Samples, index: usize) i16 {
            return switch (self) {
                .s16 => |data| data[index],
                .s32 => |data| @intCast(data[index] >> 16),
                .f32 => |data| floatPcm16(data[index]),
                .f64 => |data| floatPcm16(data[index]),
            };
        }
    };
    pub const Fields = struct { data: []const u8, audio_format: []const u8 };

    pub const schema_description = "An audio input type for DSPy.\n\nConstruct from in-memory values only: a data URI string, raw ``bytes`` (with\n``audio_format=``), a numpy-like array, a ``{\"data\", \"audio_format\"}`` dict, or another\n``Audio``. Raw base64 is passed as ``Audio(data=..., audio_format=...)``. Construction and\nadapter parsing never access the filesystem or network; use :meth:`from_path` to read a local\nfile, :meth:`from_url` to download a remote resource, or :meth:`from_array` for array data.";
    pub const marker_start = file_model.File.marker_start;
    pub const marker_end = file_model.File.marker_end;

    /// Construct with the Source's single positional slot and keyword record.
    /// sampling_rate cannot be applied to any of the supported inputs.
    pub fn init(allocator: Allocator, source: ?Value, keywords: Value) (Error || Allocator.Error)!Audio {
        if (keywords != .object) return error.InvalidAudio;
        const input = source orelse return fromValue(allocator, keywords);
        for (keywords.object.keys()) |key| {
            if (!std.mem.eql(u8, key, "audio_format") and !std.mem.eql(u8, key, "sampling_rate"))
                return error.InvalidAudio;
        }
        if ((keywords.object.get("sampling_rate") orelse .null) != .null) return error.InvalidAudio;
        const format_value = keywords.object.get("audio_format") orelse .null;
        if (format_value == .null) return fromValue(allocator, input);
        const builtin = python.view(input) orelse return error.InvalidAudio;
        if (builtin != .bytes) return error.InvalidAudio;
        const field: StringField = if (isTruthy(format_value) catch return error.InvalidAudio)
            try StringField.read(format_value)
        else
            .{ .text = "wav" };
        const audio_format = try field.snapshot(allocator);
        errdefer if (field.mutable) allocator.free(audio_format);
        return fromEncodedBytes(allocator, builtin.bytes, audio_format);
    }

    /// Accept a positional data URI, a strict two-field dict, genuine bytes
    /// (default format wav), or a genuine native Audio. Strings and genuine
    /// Audio fields borrow; bytearray fields snapshot into allocator. No input
    /// is opened as a path or URL, and lookalike models do not count as Audio.
    pub fn fromValue(allocator: Allocator, input: Value) (Error || Allocator.Error)!Audio {
        switch (input) {
            .string => |uri| return fromDataUri(uri),
            .object => |object| {
                if (object.count() != 2) return error.InvalidAudio;
                return fromDictionaryFields(allocator, object.get("data") orelse return error.InvalidAudio, object.get("audio_format") orelse return error.InvalidAudio);
            },
            .custom => |custom| {
                if (custom.get(Audio)) |model| return model.*;
                const builtin = python.view(input) orelse return error.InvalidAudio;
                switch (builtin) {
                    .bytes => |bytes| return fromBytes(allocator, bytes, .{}),
                    .dictionary => |entries| {
                        if (entries.len != 2) return error.InvalidAudio;
                        var data: ?Value = null;
                        var audio_format: ?Value = null;
                        for (entries) |entry| {
                            if (entry.key != .string) return error.InvalidAudio;
                            const key = entry.key.string;
                            if (std.mem.eql(u8, key, "data")) data = entry.value;
                            if (std.mem.eql(u8, key, "audio_format")) audio_format = entry.value;
                        }
                        return fromDictionaryFields(allocator, data orelse return error.InvalidAudio, audio_format orelse return error.InvalidAudio);
                    },
                    else => return error.InvalidAudio,
                }
            },
            else => return error.InvalidAudio,
        }
    }

    /// Borrow raw model fields, preserving format spelling and base64 text.
    /// Empty and malformed payloads are accepted until format/serialize.
    pub fn fromFields(fields: Fields) Error!Audio {
        if (!std.unicode.utf8ValidateSlice(fields.data) or !std.unicode.utf8ValidateSlice(fields.audio_format))
            return error.InvalidAudio;
        return .{ .data = fields.data, .audio_format = fields.audio_format };
    }

    /// Base64-encode bytes of existing encoded media into allocator. Borrow the
    /// format option; as in Source, an empty option defaults to wav.
    pub fn fromBytes(allocator: Allocator, bytes: []const u8, options: Options) (Error || Allocator.Error)!Audio {
        if (!std.unicode.utf8ValidateSlice(options.audio_format)) return error.InvalidAudio;
        return fromEncodedBytes(allocator, bytes, if (options.audio_format.len == 0) "wav" else options.audio_format);
    }

    /// Encode interleaved samples as a conventional RIFF PCM16 WAV.
    /// Base64 data is owned; the caller's WAV format spelling is borrowed.
    pub fn fromWavSamples(allocator: Allocator, samples: Samples, options: WavOptions) !Audio {
        if (!std.ascii.eqlIgnoreCase(options.audio_format, "wav")) return error.InvalidAudio;
        if (options.channels == 0 or options.sample_rate_hz == 0)
            return error.InvalidAudioSampleLayout;
        if (samples.len() % options.channels != 0) return error.InvalidAudioSampleLayout;
        const frame_bytes = std.math.mul(u16, options.channels, 2) catch
            return error.InvalidAudioSampleLayout;
        const byte_rate = std.math.mul(u32, options.sample_rate_hz, frame_bytes) catch
            return error.InvalidAudioSampleLayout;
        const data_bytes = std.math.mul(usize, samples.len(), 2) catch return error.AudioTooLarge;
        if (data_bytes > std.math.maxInt(u32) - 36) return error.AudioTooLarge;
        const header: [44]u8 = wavHeader(@intCast(data_bytes), options, frame_bytes, byte_rate);
        const wav_bytes = std.math.add(usize, header.len, data_bytes) catch return error.AudioTooLarge;
        if (wav_bytes > options.max_bytes) return error.AudioTooLarge;
        const data = try allocator.alloc(u8, try encodedSize(wav_bytes));
        const encoder = std.base64.standard.Encoder;
        // Keep intermediate chunks on sample and base64 group boundaries.
        var chunk: [3 * 1024]u8 = undefined;
        @memcpy(chunk[0..header.len], &header);
        var used: usize = header.len;
        var encoded: usize = 0;
        for (0..samples.len()) |i| {
            std.mem.writeInt(i16, chunk[used..][0..2], samples.pcm16(i), .little);
            used += 2;
            if (used == chunk.len) {
                encoded += encoder.encode(data[encoded..], &chunk).len;
                used = 0;
            }
        }
        encoded += encoder.encode(data[encoded..], chunk[0..used]).len;
        std.debug.assert(encoded == data.len);
        return .{ .data = data, .audio_format = options.audio_format };
    }

    /// Borrow fields from a data URI. Source partitions at the first comma
    /// and does not require a ;base64 parameter or validate the payload here.
    pub fn fromDataUri(uri: []const u8) Error!Audio {
        const prefix = "data:audio/";
        if (!std.mem.startsWith(u8, uri, prefix)) return error.InvalidAudio;
        if (!std.unicode.utf8ValidateSlice(uri)) return error.InvalidAudioDataUri;
        const rest = uri[prefix.len..];
        const comma = std.mem.indexOfScalar(u8, rest, ',') orelse return error.InvalidAudioDataUri;
        const header = rest[0..comma];
        const semicolon = std.mem.indexOfScalar(u8, header, ';') orelse header.len;
        if (semicolon == 0) return error.InvalidAudioDataUri;
        return .{ .data = rest[comma + 1 ..], .audio_format = normalizeFormat(header[0..semicolon]) };
    }

    /// Download caller-trusted HTTP(S) media and require an audio MIME if
    /// provided. Redirect targets must also be trusted. Both fields are owned.
    pub fn fromUrl(
        allocator: Allocator,
        client: *std.http.Client,
        url: []const u8,
        options: UrlOptions,
    ) !Audio {
        var scratch: std.heap.ArenaAllocator = .init(client.allocator);
        defer scratch.deinit();
        const response = try media_download.download(scratch.allocator(), client, url, options);
        defer client.allocator.free(response.body);
        const format_name = try formatFromMime(response.content_type orelse "audio/wav");
        const owned_format = try allocator.dupe(u8, format_name);
        errdefer allocator.free(owned_format);
        return fromEncodedBytes(allocator, response.body, owned_format);
    }

    /// Explicit, caller-initiated read of a local regular file, following
    /// symlinks like Source. Both fields are allocated in allocator.
    /// Limit is inclusive and counts bytes before base64.
    /// The read also enforces the limit if the file grows after stat. This is
    /// a byte bound, not a deadline or a sandbox for untrusted paths.
    pub fn fromPath(allocator: Allocator, io: std.Io, path: []const u8, options: PathOptions) !Audio {
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.FileNotFound;
        const directory = std.Io.Dir.cwd();
        const initial = directory.statFile(io, path, .{}) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => return error.FileNotFound,
        };
        if (initial.kind != .file) return error.FileNotFound;
        var scratch = std.heap.ArenaAllocator.init(allocator);
        defer scratch.deinit();
        const guessed = try mime_types.guessType(scratch.allocator(), path, true);
        const media_type = guessed.mime_type orelse return error.UnsupportedAudioMimeType;
        const audio_format = try formatFromMime(media_type);
        if (initial.size > options.max_bytes) return error.AudioTooLarge;
        const file = try directory.openFile(io, path, .{ .allow_directory = false });
        defer file.close(io);
        const opened = try file.stat(io);
        if (opened.kind != .file) return error.FileNotFound;
        if (opened.size > options.max_bytes) return error.AudioTooLarge;
        var reader = file.reader(io, &.{});
        var output: Writer.Allocating = try .initCapacity(allocator, try encodedSize(@intCast(opened.size)));
        errdefer output.deinit();
        var remaining = options.max_bytes;
        // Full chunks are divisible by three. Only the last can add padding.
        var chunk: [3 * 1024]u8 = undefined;
        while (remaining != 0) {
            const room = @min(remaining, chunk.len);
            const count = reader.interface.readSliceShort(chunk[0..room]) catch return reader.err.?;
            if (count == 0) break;
            const encoded = output.writer.writableSlice(try encodedSize(count)) catch return error.OutOfMemory;
            _ = std.base64.standard.Encoder.encode(encoded, chunk[0..count]);
            remaining -= count;
            if (count < room) break;
        }
        if (remaining == 0 and (reader.interface.readSliceShort(chunk[0..1]) catch return reader.err.?) != 0)
            return error.AudioTooLarge;
        const data = try output.toOwnedSlice();
        errdefer allocator.free(data);
        return .{ .data = data, .audio_format = try allocator.dupe(u8, audio_format) };
    }

    /// Own an independent copy of both fields.
    pub fn clone(self: Audio, allocator: Allocator) Allocator.Error!Audio {
        const data = try allocator.dupe(u8, self.data);
        errdefer allocator.free(data);
        return .{ .data = data, .audio_format = try allocator.dupe(u8, self.audio_format) };
    }

    /// Free both fields of an owned Audio returned by clone, fromPath or fromUrl.
    /// Do not call this on views returned by other constructors.
    pub fn deinitOwned(self: Audio, allocator: Allocator) void {
        allocator.free(self.audio_format);
        allocator.free(self.data);
    }

    /// Allocate only the model box; its immutable fields borrow from self.
    /// Cloning the runtime value produces an independent owning box.
    pub fn toRuntime(self: Audio, allocator: Allocator) Allocator.Error!Value {
        const boxed = try allocator.create(Audio);
        boxed.* = self;
        return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
    }

    /// The typed slot and its fields must outlive the borrowed runtime handle.
    pub fn borrowedRuntime(self: *const Audio) Value {
        return .{ .custom = .{ .ptr = self, .vtable = &runtime_vtable, .owns_box = false } };
    }

    /// Round-trip model dump: map storage is allocated; strings borrow self.
    pub fn toValue(self: Audio, allocator: Allocator) Allocator.Error!Value {
        var object: runtime.ObjectMap = .empty;
        try object.ensureTotalCapacity(allocator, 2);
        object.putAssumeCapacity("data", .{ .string = self.data });
        object.putAssumeCapacity("audio_format", .{ .string = self.audio_format });
        return .{ .object = object };
    }

    /// Ordered lm15 part_to_dict list. Use an arena for the projection; the
    /// payload and constant keys are borrowed, and data is never normalized.
    pub fn format(self: Audio, arena: Allocator) (Error || Allocator.Error)!Value {
        const subtype = try self.validatedMediaSubtype();
        const media_type = try std.mem.concat(arena, u8, &.{ "audio/", subtype });
        errdefer arena.free(media_type);
        var object: runtime.ObjectMap = .empty;
        try object.ensureTotalCapacity(arena, 3);
        errdefer object.deinit(arena);
        object.putAssumeCapacity("type", .{ .string = "audio" });
        object.putAssumeCapacity("media_type", .{ .string = media_type });
        object.putAssumeCapacity("data", .{ .string = self.data });
        var list: runtime.Array = .init(arena);
        try list.ensureTotalCapacity(1);
        list.appendAssumeCapacity(.{ .object = object });
        return .{ .array = list };
    }

    /// Default model dump and Python str(Audio) yield the marker string.
    pub fn serialize(self: Audio, allocator: Allocator) (Error || Allocator.Error)![]const u8 {
        var output: Writer.Allocating = .init(allocator);
        errdefer output.deinit();
        self.writeSerialized(&output.writer) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            else => |failure| failure,
        };
        return output.toOwnedSlice();
    }

    pub fn writeSerialized(self: Audio, writer: *Writer) (Error || Writer.Error)!void {
        const subtype = try self.validatedMediaSubtype();
        try writer.writeAll(marker_start ++ "[{\"type\": \"audio\", \"media_type\": \"audio/");
        var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
        try escaped.writer.writeAll(subtype);
        try writer.writeAll("\", \"data\": ");
        try pyjson.writeJsonString(writer, self.data);
        try writer.writeAll("}]" ++ marker_end);
    }

    /// Source repr redacts data and measures Unicode code points, not bytes.
    pub fn writeRepr(self: Audio, writer: *Writer) Writer.Error!void {
        var count: usize = 0;
        for (self.data) |byte| {
            if (byte & 0xc0 != 0x80) count += 1;
        }
        try writer.print("Audio(data=<AUDIO_BASE_64_ENCODED({d})>, audio_format='{s}')", .{ count, self.audio_format });
    }

    fn validatedMediaSubtype(self: Audio) Error![]const u8 {
        if (self.data.len == 0) return error.InvalidAudioBase64;
        file_model.validateBase64(self.data) catch return error.InvalidAudioBase64;
        if (std.mem.eql(u8, self.audio_format, "mp3")) return "mpeg";
        return self.audio_format;
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(Audio),
        .type_identity = runtime.typeIdentity(Audio),
        .model_name = "Audio",
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

    fn runtimeAudio(ptr: *const anyopaque) *const Audio {
        return @ptrCast(@alignCast(ptr));
    }

    fn rawRuntimeValue(allocator: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return runtimeAudio(ptr).toValue(allocator);
    }

    fn modelDumpRuntimeValue(allocator: Allocator, ptr: *const anyopaque) runtime.CustomError!Value {
        return serializedRuntimeValue(allocator, ptr, 0);
    }

    fn serializedRuntimeValue(allocator: Allocator, ptr: *const anyopaque, _: u32) runtime.CustomError!Value {
        return .{ .string = runtimeAudio(ptr).serialize(allocator) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        } };
    }

    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        runtimeAudio(ptr).writeSerialized(writer) catch |err| return switch (err) {
            error.WriteFailed => error.WriteFailed,
            else => error.ParseValue,
        };
    }

    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, _: u32) runtime.CustomWriteError!void {
        try runtimeAudio(ptr).writeRepr(writer);
    }

    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(runtimeAudio(ptr).*);
    }

    fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, _: usize) runtime.CustomWriteError!void {
        try writer.writeByte('"');
        var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
        try writeRuntimeStr(&escaped.writer, ptr);
        try writer.writeByte('"');
    }

    fn cloneRuntime(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const allocator = context.allocator;
        const copy = try runtimeAudio(ptr).clone(allocator);
        errdefer copy.deinitOwned(allocator);
        const boxed = try allocator.create(Audio);
        boxed.* = copy;
        return .{ .ptr = boxed, .vtable = &runtime_vtable, .owns_data = true };
    }

    fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        const boxed = runtimeAudio(ptr);
        const model = boxed.*;
        allocator.destroy(boxed);
        if (owns_data) model.deinitOwned(allocator);
    }

    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
        const a = runtimeAudio(left);
        const b = runtimeAudio(right);
        return std.mem.eql(u8, a.data, b.data) and std.mem.eql(u8, a.audio_format, b.audio_format);
    }
};

fn fromDictionaryFields(allocator: Allocator, data: Value, audio_format: Value) (Audio.Error || Allocator.Error)!Audio {
    const data_field = try StringField.read(data);
    const format_field = try StringField.read(audio_format);
    const text = try data_field.snapshot(allocator);
    errdefer if (data_field.mutable) allocator.free(text);
    return .{ .data = text, .audio_format = try format_field.snapshot(allocator) };
}

const StringField = struct {
    text: []const u8,
    mutable: bool = false,

    fn read(value: Value) Audio.Error!StringField {
        const field: StringField = if (value == .string) .{ .text = value.string } else field: {
            const builtin = python.view(value) orelse return error.InvalidAudio;
            break :field switch (builtin) {
                .bytes => |text| .{ .text = text },
                .bytearray => |text| .{ .text = text, .mutable = true },
                else => return error.InvalidAudio,
            };
        };
        if (!std.unicode.utf8ValidateSlice(field.text)) return error.InvalidAudio;
        return field;
    }

    fn snapshot(self: StringField, allocator: Allocator) Allocator.Error![]const u8 {
        return if (self.mutable) try allocator.dupe(u8, self.text) else self.text;
    }
};

fn isTruthy(value: Value) error{ParseValue}!bool {
    return switch (value) {
        .null => false,
        .bool => |flag| flag,
        .integer => |number| number != 0,
        .float => |number| number != 0,
        .number_string => |number| (std.fmt.parseFloat(f64, number) catch return error.ParseValue) != 0,
        .string => |text| text.len != 0,
        .array => |array| array.items.len != 0,
        .object => |object| object.count() != 0,
        .custom => |custom| try custom.truthy(),
    };
}

fn floatPcm16(sample: f64) i16 {
    if (std.math.isNan(sample)) return 0;
    if (sample >= 1.0) return std.math.maxInt(i16);
    if (sample <= -1.0) return std.math.minInt(i16);
    // libsndfile rounds at 32-bit precision before taking the high 16 bits.
    const scaled = sample * 0x1p31;
    const floor = @floor(scaled);
    const lower: i64 = @intFromFloat(floor);
    const remainder = scaled - floor;
    const increment = remainder > 0.5 or (remainder == 0.5 and lower & 1 != 0);
    const rounded = @min(lower + @intFromBool(increment), std.math.maxInt(i32));
    return @intCast(rounded >> 16);
}

fn wavHeader(data_bytes: u32, options: Audio.WavOptions, frame_bytes: u16, byte_rate: u32) [44]u8 {
    var bytes = ("RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" ++
        "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00" ++
        "data\x00\x00\x00\x00").*;
    std.mem.writeInt(u32, bytes[4..8], data_bytes + 36, .little);
    std.mem.writeInt(u16, bytes[22..24], options.channels, .little);
    std.mem.writeInt(u32, bytes[24..28], options.sample_rate_hz, .little);
    std.mem.writeInt(u32, bytes[28..32], byte_rate, .little);
    std.mem.writeInt(u16, bytes[32..34], frame_bytes, .little);
    std.mem.writeInt(u32, bytes[40..44], data_bytes, .little);
    return bytes;
}

fn formatFromMime(media_type: []const u8) (Audio.Error || error{UnsupportedAudioMimeType})![]const u8 {
    const prefix = "audio/";
    if (!std.mem.startsWith(u8, media_type, prefix)) return error.UnsupportedAudioMimeType;
    // Source takes split('/')[1], not the entire suffix after audio/.
    const subtype = media_type[prefix.len..];
    const end = std.mem.indexOfScalar(u8, subtype, '/') orelse subtype.len;
    const format_name = normalizeFormat(subtype[0..end]);
    if (!std.unicode.utf8ValidateSlice(format_name)) return error.InvalidAudio;
    return format_name;
}

fn normalizeFormat(audio_format: []const u8) []const u8 {
    return if (std.mem.startsWith(u8, audio_format, "x-")) audio_format[2..] else audio_format;
}

fn encodedSize(bytes: usize) Allocator.Error!usize {
    const groups = bytes / 3 + @intFromBool(bytes % 3 != 0);
    return std.math.mul(usize, groups, 4) catch error.OutOfMemory;
}

fn fromEncodedBytes(allocator: Allocator, bytes: []const u8, audio_format: []const u8) Allocator.Error!Audio {
    const data = try allocator.alloc(u8, try encodedSize(bytes.len));
    _ = std.base64.standard.Encoder.encode(data, bytes);
    return .{ .data = data, .audio_format = audio_format };
}

test "Audio preserves raw fields and emits the Source lm15 part" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const audio = try Audio.fromDataUri("data:audio/x-mpeg;charset=ignored,Zg==");
    const raw = try audio.toValue(arena);
    try std.testing.expectEqualStrings("mpeg", raw.object.get("audio_format").?.string);
    try std.testing.expectEqualStrings("Zg==", raw.object.get("data").?.string);
    const formatted = try audio.format(arena);
    try std.testing.expectEqual(@as(usize, 1), formatted.array.items.len);
    const part = formatted.array.items[0].object;
    try std.testing.expectEqualStrings("audio", part.get("type").?.string);
    try std.testing.expectEqualStrings("audio/mpeg", part.get("media_type").?.string);
    try std.testing.expectEqualStrings("Zg==", part.get("data").?.string);
    const boxed = try audio.toRuntime(arena);
    const dump = try boxed.custom.modelDump(arena);
    try std.testing.expectEqualStrings(
        Audio.marker_start ++ "[{\"type\": \"audio\", \"media_type\": \"audio/mpeg\", \"data\": \"Zg==\"}]" ++ Audio.marker_end,
        dump.string,
    );

    const empty = try Audio.fromBytes(arena, "", .{});
    try std.testing.expectEqualStrings("", empty.data);
    try std.testing.expectError(error.InvalidAudioBase64, empty.format(arena));
    const invalid = try Audio.fromFields(.{ .data = "Zg=", .audio_format = "wav" });
    try std.testing.expectError(error.InvalidAudioBase64, invalid.serialize(arena));
}

test "WAV samples preserve the PCM16 rounding boundary and inclusive byte limit" {
    const allocator = std.testing.allocator;
    const samples: Audio.Samples = .{ .f64 = &.{ -0x1p-32, -0x1.8p-31, -0x1p-16, 0x1p-16 } };
    var options: Audio.WavOptions = .{ .sample_rate_hz = 16000, .max_bytes = 51 };
    try std.testing.expectError(error.AudioTooLarge, Audio.fromWavSamples(allocator, samples, options));
    options.max_bytes += 1;
    const audio = try Audio.fromWavSamples(allocator, samples, options);
    defer allocator.free(audio.data);
    var bytes: [52]u8 = undefined;
    try std.base64.standard.Decoder.decode(&bytes, audio.data);
    try std.testing.expectEqualStrings("RIFF", bytes[0..4]);
    try std.testing.expectEqual(@as(u32, 8), std.mem.readInt(u32, bytes[40..44], .little));
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 255, 255, 255, 255, 0, 0 }, bytes[44..]);
}

test "Audio snapshots bytearray fields and runtime clone outlives the source arena" {
    const allocator = std.testing.allocator;
    var source: std.heap.ArenaAllocator = .init(allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const arena = source.allocator();
    const data = try python.bytearray(arena, "Zg==");
    const subtype = try python.bytearray(arena, "mp3");
    var fields: runtime.ObjectMap = .empty;
    try fields.put(arena, "data", data);
    try fields.put(arena, "audio_format", subtype);
    const audio = try Audio.fromValue(arena, .{ .object = fields });
    try python.bytearraySet(data, 0, 'A');
    try python.bytearraySet(subtype, 0, 'x');
    try std.testing.expectEqualStrings("Zg==", audio.data);
    try std.testing.expectEqualStrings("mp3", audio.audio_format);

    const borrowed = audio.borrowedRuntime().custom;
    const owned = try borrowed.clone(allocator);
    defer owned.deinit(allocator);
    try std.testing.expect(owned.owns_data);
    source.deinit();
    source_live = false;

    const copy = owned.get(Audio).?;
    try std.testing.expectEqualStrings("Zg==", copy.data);
    try std.testing.expectEqualStrings("mp3", copy.audio_format);
}

test "Audio path byte limit includes exact-size and empty files" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var temporary = testing.tmpDir(.{});
    defer temporary.cleanup();

    // Cross the loader's internal chunk boundary, leaving a padded last chunk.
    const input: [3073]u8 = @splat('a');
    try temporary.dir.writeFile(testing.io, .{ .sub_path = "sample.mp3", .data = &input });
    const path = try temporary.dir.realPathFileAlloc(testing.io, "sample.mp3", allocator);
    defer allocator.free(path);

    try testing.expectError(error.AudioTooLarge, Audio.fromPath(allocator, testing.io, path, .{ .max_bytes = input.len - 1 }));
    const exact = try Audio.fromPath(allocator, testing.io, path, .{ .max_bytes = input.len });
    defer exact.deinitOwned(allocator);
    const encoded = try Audio.fromBytes(allocator, &input, .{});
    defer allocator.free(encoded.data);
    try testing.expectEqualStrings("mpeg", exact.audio_format);
    try testing.expectEqualStrings(encoded.data, exact.data);

    try temporary.dir.writeFile(testing.io, .{ .sub_path = "sample.mp3", .data = "" });
    const empty = try Audio.fromPath(allocator, testing.io, path, .{ .max_bytes = 0 });
    defer empty.deinitOwned(allocator);
    try testing.expectEqualStrings("", empty.data);
    try testing.expectError(error.InvalidAudioBase64, empty.format(allocator));
}
