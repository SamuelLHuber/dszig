//! Experimental DSPy Document. The adapter consumes its serialized marker;
//! providers see only the existing canonical text and document content parts.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const pyjson = @import("pyjson.zig");
const message_content = @import("message_content.zig");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Value = runtime.Value;

pub const Document = struct {
    data: []const u8,
    title: ?[]const u8 = null,
    media_type: MediaType = .@"text/plain",
    context: ?[]const u8 = null,

    pub const MediaType = enum { @"text/plain", @"application/pdf" };
    pub const Error = error{ InvalidDocument, InvalidDocumentBase64 };
    pub const model_name = "Document";
    pub const model_module = "dspy.adapters.types.document";
    pub const description = "A document containing text content that can be referenced and cited. Include the full text content and optionally a title for proper referencing.";
    pub const schema_description =
        \\Experimental: This class may change or be removed in a future release without warning (introduced in v3.0.4).
        \\
        \\A document type for providing content that can be cited by language models.
        \\
        \\    This type represents documents that can be passed to language models for citation-enabled
        \\    responses, particularly useful with Anthropic's Citations API. Documents include the content
        \\    and metadata that helps the LM understand and reference the source material.
        \\
        \\    Attributes:
        \\        data: The text content of the document
        \\        title: Optional title for the document (used in citations)
        \\        media_type: MIME type of the document content (defaults to "text/plain")
        \\        context: Optional context information about the document
        \\
        \\    Examples:
        \\        ```python
        \\        import dspy
        \\        from dspy.signatures import Signature
        \\        from dspy.experimental import Document, Citations
        \\
        \\        class AnswerWithSources(Signature):
        \\            '''Answer questions using provided documents with citations.'''
        \\            documents: list[Document] = dspy.InputField()
        \\            question: str = dspy.InputField()
        \\            answer: str = dspy.OutputField()
        \\            citations: Citations = dspy.OutputField()
        \\
        \\        # Create documents
        \\        docs = [
        \\            Document(
        \\                data="The Earth orbits the Sun in an elliptical path.",
        \\                title="Basic Astronomy Facts"
        \\            ),
        \\            Document(
        \\                data="Water boils at 100°C at standard atmospheric pressure.",
        \\                title="Physics Fundamentals",
        \\            )
        \\        ]
        \\
        \\        # Use with a citation-supporting model
        \\        lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
        \\        predictor = dspy.Predict(AnswerWithSources)
        \\        result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)
        \\        print(result.citations)
        \\        ```
        \\
    ++ "    ";

    /// Accept an actual Document, a bare text string, or model fields. Unknown
    /// object keys are ignored. Strings borrow storage; bytearrays are copied.
    /// PDF data is checked only when it is formatted, not at construction.
    pub fn fromValue(arena: Allocator, input: Value) (Error || Allocator.Error)!Document {
        switch (input) {
            .custom => |c| return if (c.get(Document)) |document| document.* else error.InvalidDocument,
            .string => |text| return .{ .data = try validString(text) },
            .object => |object| {
                const media_type: MediaType = if (object.get("media_type")) |media| switch (media) {
                    .string => |name| std.meta.stringToEnum(MediaType, name) orelse return error.InvalidDocument,
                    else => return error.InvalidDocument,
                } else .@"text/plain";
                const data = try StringField.parse(object.get("data") orelse .null) orelse return error.InvalidDocument;
                const title = try StringField.parse(object.get("title") orelse .null);
                const context = try StringField.parse(object.get("context") orelse .null);
                const data_text = try data.materialize(arena);
                errdefer if (data.mutable) arena.free(data_text);
                const title_text = if (title) |field| try field.materialize(arena) else null;
                errdefer if (title) |field| {
                    if (field.mutable) arena.free(title_text.?);
                };
                return .{
                    .data = data_text,
                    .title = title_text,
                    .media_type = media_type,
                    .context = if (context) |field| try field.materialize(arena) else null,
                };
            },
            else => return error.InvalidDocument,
        }
    }

    /// Round-trip view, with borrowed string fields and an arena-owned map.
    pub fn toValue(self: Document, arena: Allocator) Allocator.Error!Value {
        var object: runtime.ObjectMap = .empty;
        try object.ensureTotalCapacity(arena, 4);
        object.putAssumeCapacity("data", .{ .string = self.data });
        object.putAssumeCapacity("title", optionalString(self.title));
        object.putAssumeCapacity("media_type", .{ .string = @tagName(self.media_type) });
        object.putAssumeCapacity("context", optionalString(self.context));
        return .{ .object = object };
    }

    /// The default Python-mode and adapter model dumps are the same marker.
    pub fn serialize(self: Document, arena: Allocator) (Error || Allocator.Error)![]const u8 {
        var output: Writer.Allocating = .init(arena);
        errdefer output.deinit();
        self.writeSerialized(&output.writer) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            else => |failure| failure,
        };
        return output.toOwnedSlice();
    }

    pub fn writeSerialized(self: Document, writer: *Writer) (Error || Writer.Error)!void {
        _ = try validString(self.data);
        const title = try nonempty(self.title);
        const context = try nonempty(self.context);
        if (self.media_type == .@"application/pdf") {
            // lm15 canonical semantics validate base64 shape, not PDF bytes;
            // retain the original string, even when validation normalizes it.
            if (self.data.len == 0) return error.InvalidDocumentBase64;
            @import("file.zig").validateBase64(self.data) catch return error.InvalidDocumentBase64;
        }
        try writer.writeAll(message_content.start_marker ++ "[");
        if (title != null or context != null) {
            try writer.writeAll("{\"type\": \"text\", \"text\": \"");
            var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
            if (title) |text| try escaped.writer.print("Title: {s}\n", .{text});
            if (context) |text| try escaped.writer.print("Context: {s}\n", .{text});
            try writer.writeAll("\"}, ");
        }
        try writer.writeAll(switch (self.media_type) {
            .@"text/plain" => "{\"type\": \"text\", \"text\": ",
            .@"application/pdf" => "{\"type\": \"document\", \"media_type\": \"application/pdf\", \"data\": ",
        });
        try pyjson.writeJsonString(writer, self.data);
        try writer.writeAll("}]" ++ message_content.end_marker);
    }

    pub fn toRuntime(self: Document, arena: Allocator) Allocator.Error!Value {
        const boxed = try arena.create(Document);
        boxed.* = self;
        return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
    }

    pub fn borrowedRuntime(self: *const Document) Value {
        return .{ .custom = .{ .ptr = self, .vtable = &runtime_vtable, .owns_box = false } };
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(Document),
        .type_identity = runtime.typeIdentity(Document),
        .model_name = model_name,
        .is_type = true,
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

    fn runtimeDocument(ptr: *const anyopaque) *const Document {
        return @ptrCast(@alignCast(ptr));
    }

    fn rawRuntimeValue(arena: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return runtimeDocument(ptr).toValue(arena);
    }

    fn modelDumpRuntimeValue(arena: Allocator, ptr: *const anyopaque) runtime.CustomError!Value {
        return .{ .string = runtimeDocument(ptr).serialize(arena) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        } };
    }

    fn serializedRuntimeValue(arena: Allocator, ptr: *const anyopaque, _: u32) runtime.SerializationError!Value {
        // DSPy's serialize_for_json catches model serialization failures and
        // uses Python str() on the whole input, not only the bad nested model.
        return modelDumpRuntimeValue(arena, ptr) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.ParseValue => error.PythonSerializeFallback,
        };
    }

    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        const self = runtimeDocument(ptr);
        try writer.writeAll("Document(");
        if (self.title) |title| if (title.len != 0) try writer.print("'{s}': ", .{title});
        var count: usize = 0;
        for (self.data) |byte| {
            if (byte & 0xc0 != 0x80) count += 1;
        }
        try writer.print("{d} chars)", .{count});
    }

    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
        const self = runtimeDocument(ptr);
        try writer.writeAll("Document(data=");
        try pyjson.writeRepr(writer, .{ .string = self.data }, depth);
        try writer.writeAll(", title=");
        try pyjson.writeRepr(writer, optionalString(self.title), depth);
        try writer.writeAll(", media_type=");
        try pyjson.writeRepr(writer, .{ .string = @tagName(self.media_type) }, depth);
        try writer.writeAll(", context=");
        try pyjson.writeRepr(writer, optionalString(self.context), depth);
        try writer.writeByte(')');
    }

    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(runtimeDocument(ptr).*);
    }

    fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, _: usize) runtime.CustomWriteError!void {
        try writer.writeByte('"');
        var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
        runtimeDocument(ptr).writeSerialized(&escaped.writer) catch |err| return switch (err) {
            error.WriteFailed => error.WriteFailed,
            else => error.ParseValue,
        };
        try writer.writeByte('"');
    }

    fn cloneRuntime(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const arena = context.allocator;
        const source = runtimeDocument(ptr);
        const copy = try arena.create(Document);
        errdefer arena.destroy(copy);
        copy.* = .{ .data = try arena.dupe(u8, source.data), .media_type = source.media_type };
        errdefer arena.free(copy.data);
        copy.title = if (source.title) |title| try arena.dupe(u8, title) else null;
        errdefer if (copy.title) |title| arena.free(title);
        copy.context = if (source.context) |text| try arena.dupe(u8, text) else null;
        return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
    }

    fn deinitRuntime(arena: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        const document = runtimeDocument(ptr);
        if (owns_data) {
            arena.free(document.data);
            if (document.title) |title| arena.free(title);
            if (document.context) |context| arena.free(context);
        }
        arena.destroy(document);
    }

    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
        const a = runtimeDocument(left);
        const b = runtimeDocument(right);
        return a.media_type == b.media_type and std.mem.eql(u8, a.data, b.data) and
            optionalEqual(a.title, b.title) and optionalEqual(a.context, b.context);
    }
};

const StringField = struct {
    text: []const u8,
    mutable: bool = false,

    fn parse(input: Value) Document.Error!?StringField {
        switch (input) {
            .null => return null,
            .string => |text| return .{ .text = try validString(text) },
            else => {},
        }
        if (@import("python_value.zig").view(input)) |builtin| {
            return switch (builtin) {
                .bytes => |bytes| .{ .text = try validString(bytes) },
                .bytearray => |bytes| .{ .text = try validString(bytes), .mutable = true },
                else => error.InvalidDocument,
            };
        }
        return error.InvalidDocument;
    }

    fn materialize(self: StringField, arena: Allocator) Allocator.Error![]const u8 {
        return if (self.mutable) arena.dupe(u8, self.text) else self.text;
    }
};

fn validString(text: []const u8) Document.Error![]const u8 {
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidDocument;
    return text;
}

fn nonempty(text: ?[]const u8) Document.Error!?[]const u8 {
    const bytes = text orelse return null;
    return if (bytes.len == 0) null else try validString(bytes);
}

fn optionalString(text: ?[]const u8) Value {
    return if (text) |bytes| .{ .string = bytes } else .null;
}

fn optionalEqual(a: ?[]const u8, b: ?[]const u8) bool {
    return if (a) |x| if (b) |y| std.mem.eql(u8, x, y) else false else b == null;
}

test "Document typed adapter input expands framed text and canonical PDF" {
    const adapter = @import("adapter.zig");
    const value = @import("value.zig");
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { attachment: Document };
        pub const Output = struct { answer: []const u8 };
    });
    const view: adapter.View = .{
        .instructions = Sig.instructions,
        .inputs = Sig.input_fields_list,
        .outputs = Sig.output_fields_list,
    };
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const plain: Sig.Input = .{ .attachment = .{ .data = "body", .title = "T", .context = "C" } };
    const pdf: Sig.Input = .{ .attachment = .{ .data = "JVBERi0=", .title = "T", .context = "C", .media_type = .@"application/pdf" } };
    var inputs: value.Record = .empty;
    try inputs.ensureTotalCapacity(arena, 1);
    const chat_end = "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.";
    const xml_end = "\n</attachment>\n\nOutput format (replace placeholders):\n<answer>\n{answer}\n</answer>\n\nUse this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.";
    const cases = [_]struct { mode: adapter.Adapter, prefix: []const u8, suffix: []const u8 }{
        .{ .mode = .chat, .prefix = "[[ ## attachment ## ]]\n", .suffix = chat_end },
        .{ .mode = .json, .prefix = "[[ ## attachment ## ]]\n", .suffix = "\n\nRespond with a JSON object in the following order of fields: `answer`." },
        .{ .mode = .xml, .prefix = "<attachment>\n", .suffix = xml_end },
    };
    for (cases) |case| {
        try value.toRecordInto(arena, Sig.Input, &plain, &inputs);
        const messages = try case.mode.format(arena, view, &.{}, &inputs);
        try std.testing.expectEqual(@as(usize, 2), messages.len);
        try std.testing.expect(messages[0].content.contains("Type description of Document: " ++ Document.description));
        try std.testing.expectEqual(.user, messages[1].role);
        try std.testing.expectEqualStrings(
            try std.fmt.allocPrint(arena, "{s}Title: T\nContext: C\nbody{s}", .{ case.prefix, case.suffix }),
            messages[1].content.text,
        );

        const pdf_record = try value.toRecord(arena, Sig.Input, pdf);
        const pdf_messages = try case.mode.format(arena, view, &.{}, &pdf_record);
        const parts = pdf_messages[1].content.parts;
        try std.testing.expectEqual(@as(usize, 3), parts.len);
        try std.testing.expectEqualStrings(
            try std.fmt.allocPrint(arena, "{s}Title: T\nContext: C\n", .{case.prefix}),
            parts[0].object.get("text").?.string,
        );
        try std.testing.expectEqualStrings(
            "{\"type\":\"document\",\"media_type\":\"application/pdf\",\"data\":\"JVBERi0=\"}",
            try std.json.Stringify.valueAlloc(arena, parts[1], .{}),
        );
        try std.testing.expectEqualStrings(case.suffix, parts[2].object.get("text").?.string);
    }
    try std.testing.expect(Sig.input_fields_list[0].xml_literal);

    const bad: Sig.Input = .{ .attachment = .{ .data = "abc", .media_type = .@"application/pdf" } };
    try value.toRecordInto(arena, Sig.Input, &bad, &inputs);
    const bad_messages = try adapter.Adapter.chat.format(arena, view, &.{}, &inputs);
    try std.testing.expectEqualStrings("[[ ## attachment ## ]]\nDocument(3 chars)" ++ chat_end, bad_messages[1].content.text);
}

test "Document model dump and validation keep raw fields distinct from the adapter marker" {
    const value = @import("value.zig");
    const types = @import("types.zig");
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw_input = try std.json.parseFromSliceLeaky(Value, arena, "{\"data\":\" 雪  \",\"title\":\"é\\\"\\n\",\"context\":\"\",\"ignored\":1}", .{});
    const model = try types.validate(Document, arena, raw_input);
    const document = try value.fromValue(arena, Document, model);
    try std.testing.expectEqual(.@"text/plain", document.media_type);
    try std.testing.expectEqualStrings(" 雪  ", document.data);
    try std.testing.expectEqual(model.custom.ptr, (try types.validate(Document, arena, model)).custom.ptr);
    try std.testing.expectEqualStrings(
        "{\"data\":\" 雪  \",\"title\":\"é\\\"\\n\",\"media_type\":\"text/plain\",\"context\":\"\"}",
        try std.json.Stringify.valueAlloc(arena, model, .{}),
    );
    const marker = message_content.start_marker ++ "[{\"type\": \"text\", \"text\": \"Title: é\\\"\\n\\n\"}, {\"type\": \"text\", \"text\": \" 雪  \"}]" ++ message_content.end_marker;
    try std.testing.expectEqualStrings(marker, (try model.custom.modelDump(arena)).string);
    try std.testing.expectEqualStrings(marker, (try value.serializedValue(arena, model)).string);
    var streamed: Writer.Allocating = .init(arena);
    try pyjson.writeJson(&streamed.writer, model, .compact);
    const streamed_string = try std.json.parseFromSliceLeaky(Value, arena, streamed.written(), .{});
    try std.testing.expectEqualStrings(marker, streamed_string.string);

    const plain = try types.validate(Document, arena, .{ .string = "" });
    const empty = try message_content.Content.fromRendered(arena, (try plain.custom.modelDump(arena)).string);
    try std.testing.expectEqualStrings("", empty.text);
    try std.testing.expectError(error.ParseValue, types.validate(Document, arena, .{ .integer = 1 }));
    try std.testing.expectError(error.ParseValue, types.validate(Document, arena, .{ .object = .empty }));
    const bad_media = try std.json.parseFromSliceLeaky(Value, arena, "{\"data\":\"x\",\"media_type\":\"image/png\"}", .{});
    try std.testing.expectError(error.ParseValue, types.validate(Document, arena, bad_media));
    const malformed = try std.json.parseFromSliceLeaky(Value, arena, "{\"data\":\"abc\",\"media_type\":\"application/pdf\"}", .{});
    const bad_model = try types.validate(Document, arena, malformed);
    try std.testing.expectError(error.ParseValue, bad_model.custom.modelDump(arena));
    try std.testing.expectError(error.InvalidDocumentBase64, (Document{ .data = "", .media_type = .@"application/pdf" }).serialize(arena));
    const raw_pdf = "data:application/pdf;base64, JVBERi0=\n";
    const wrapped = try (Document{ .data = raw_pdf, .media_type = .@"application/pdf" }).serialize(arena);
    const content = try message_content.Content.fromRendered(arena, wrapped);
    try std.testing.expectEqualStrings(raw_pdf, content.parts[0].object.get("data").?.string);

    const schema = try std.json.parseFromSliceLeaky(Value, arena, types.schema(Document), .{});
    try std.testing.expect(schema.object.get("additionalProperties") == null);
    try std.testing.expectEqualStrings(Document.schema_description, schema.object.get("description").?.string);
    try std.testing.expectEqual(@as(usize, 1), schema.object.get("required").?.array.items.len);
    try std.testing.expectEqualStrings("data", schema.object.get("required").?.array.items[0].string);
    const media = schema.object.get("properties").?.object.get("media_type").?.object;
    try std.testing.expectEqualStrings("text/plain", media.get("default").?.string);
    try std.testing.expectEqualStrings("text/plain", media.get("enum").?.array.items[0].string);
    try std.testing.expectEqualStrings("application/pdf", media.get("enum").?.array.items[1].string);
}

test "Document list preserves framing and falls back on the whole list" {
    const adapter = @import("adapter.zig");
    const value = @import("value.zig");
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { documents: []const Document };
        pub const Output = struct { answer: []const u8 };
    });
    const view: adapter.View = .{
        .instructions = Sig.instructions,
        .inputs = Sig.input_fields_list,
        .outputs = Sig.output_fields_list,
    };
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const plain: Document = .{ .data = "雪", .title = "Plain" };
    const pdf: Document = .{ .data = "JVBERi0=", .media_type = .@"application/pdf" };
    const good = try value.toRecord(arena, Sig.Input, .{ .documents = &.{ plain, pdf } });
    const malformed = try value.toRecord(arena, Sig.Input, .{ .documents = &.{ plain, .{ .data = "abc", .media_type = .@"application/pdf" } } });
    const prompt = try adapter.Adapter.chat.format(arena, view, &.{}, &good);
    try std.testing.expect(prompt[0].content.contains("`documents` (list[Document])"));
    try std.testing.expect(prompt[0].content.contains("Type description of Document: " ++ Document.description));
    const parts = prompt[1].content.parts;
    const end = "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.";
    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expectEqualStrings("[[ ## documents ## ]]\n[\"Title: Plain\n雪\", \"", parts[0].object.get("text").?.string);
    try std.testing.expectEqualStrings("JVBERi0=", parts[1].object.get("data").?.string);
    try std.testing.expectEqualStrings("\"]" ++ end, parts[2].object.get("text").?.string);

    const fallback = try adapter.Adapter.chat.format(arena, view, &.{}, &malformed);
    try std.testing.expectEqualStrings(
        "[[ ## documents ## ]]\n[Document(data='雪', title='Plain', media_type='text/plain', context=None), Document(data='abc', title=None, media_type='application/pdf', context=None)]" ++ end,
        fallback[1].content.text,
    );
}

test "Document runtime clone owns strings past source lifetime" {
    const allocator = std.testing.allocator;
    var scratch: std.heap.ArenaAllocator = .init(allocator);
    const copy = blk: {
        defer scratch.deinit();
        const arena = scratch.allocator();
        const draft: Document = .{
            .data = try arena.dupe(u8, "body"),
            .title = try arena.dupe(u8, "source"),
            .context = try arena.dupe(u8, "context"),
        };
        break :blk try draft.borrowedRuntime().custom.clone(allocator);
    };
    defer copy.deinit(allocator);
    const document = copy.get(Document).?;
    try std.testing.expectEqualStrings("body", document.data);
    try std.testing.expectEqualStrings("source", document.title.?);
    try std.testing.expectEqualStrings("context", document.context.?);
}
