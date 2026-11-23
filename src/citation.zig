//! DSPy 3.5 citation models. Typed views borrow fields; owning clones copy
//! every nested field and keep genuine model identity in erased runtime values.
//!
//! Upstream (DSPy 3.5.0): dspy/adapters/types/citation.py
const std = @import("std");
const runtime = @import("runtime_value.zig");
const values = @import("value.zig");
const types = @import("types.zig");
const pyjson = @import("pyjson.zig");
const python = @import("python_value.zig");
const inference = @import("inference.zig");
const Integer = @import("integer.zig").Integer;
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Writer = std.Io.Writer;

pub fn isCitation(comptime T: type) bool {
    return T == Citations or T == Citations.Citation;
}

/// Custom-type discovery follows annotation arguments, not model fields.
pub fn annotationType(comptime T: type) ?type {
    if (isCitation(T)) return T;
    if (comptime types.isDict(T)) return annotationType(T.dszig_dict_value);
    return switch (@typeInfo(T)) {
        .optional => |o| annotationType(o.child),
        .array => |a| if (a.len == 0) null else annotationType(a.child),
        .pointer => |p| if (p.size == .slice and T != []const u8 and T != []u8) annotationType(p.child) else null,
        else => null,
    };
}

pub const Citations = struct {
    citations: []const Citation,
    runtime_origin: ?runtime.Custom = null,

    pub const Error = runtime.CustomError;
    pub const model_name = "Citations";
    pub const description = "Citations with quoted text and source references. " ++
        "Include the exact text being cited and information about its source.";
    // @experimental prepends a notice without dedenting the original docstring.
    // Pydantic retains its body indentation and final newline plus four spaces.
    pub const schema_description =
        \\Experimental: This class may change or be removed in a future release without warning (introduced in v3.0.4).
        \\
        \\Citations extracted from an LM response with source references.
        \\
        \\    This type represents citations returned by language models that support
        \\    citation extraction, particularly Anthropic's Citations API.
        \\    Citations include the quoted text and source information.
        \\
        \\    Examples:
        \\        ```python
        \\        import os
        \\        import dspy
        \\        from dspy.signatures import Signature
        \\        from dspy.experimental import Citations, Document
        \\        os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
        \\
        \\        class AnswerWithSources(Signature):
        \\            '''Answer questions using provided documents with citations.'''
        \\            documents: list[Document] = dspy.InputField()
        \\            question: str = dspy.InputField()
        \\            answer: str = dspy.OutputField()
        \\            citations: Citations = dspy.OutputField()
        \\
        \\        # Create documents to provide as sources
        \\        docs = [
        \\            Document(
        \\                data="The Earth orbits the Sun in an elliptical path.",
        \\                title="Basic Astronomy Facts"
        \\            ),
        \\            Document(
        \\                data="Water boils at 100°C at standard atmospheric pressure.",
        \\                title="Physics Fundamentals",
        \\                context="Author: Dr. Smith; year: 2023"
        \\            )
        \\        ]
        \\
        \\        # Use with a model that supports citations like Claude
        \\        lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
        \\        predictor = dspy.Predict(AnswerWithSources)
        \\        result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)
        \\
        \\        for citation in result.citations.citations:
        \\            print(citation.format())
        \\        ```
    ++ "\n    ";

    pub const Citation = struct {
        type: []const u8 = "char_location",
        cited_text: []const u8,
        document_index: ?Integer = null,
        document_title: ?[]const u8 = null,
        url: ?[]const u8 = null,
        start_char_index: ?Integer = null,
        end_char_index: ?Integer = null,
        supported_text: ?[]const u8 = null,
        runtime_origin: ?runtime.Custom = null,

        pub const model_name = "Citation";
        pub const description = "";
        pub const field_names = .{ "type", "cited_text", "document_index", "document_title", "url", "start_char_index", "end_char_index", "supported_text" };
        pub const schema_description =
            "One citation: the quoted text plus whatever the source reported.\n\n" ++
            "Native citations arrive as lm15 `CitationPart`s carrying text, title\n" ++
            "and URL. Character offsets and document indexes are optional because\n" ++
            "not every provider reports them.";

        /// Direct TypeAdapter validation accepts dictionaries and existing
        /// Citation instances. Extra fields are ignored, not retained.
        pub fn fromValue(arena: Allocator, input: Value) Error!Citation {
            if (input == .custom) {
                if (input.custom.get(Citation)) |existing| {
                    var result = existing.*;
                    result.runtime_origin = input.custom;
                    return result;
                }
            }
            if (!isDictionary(input)) return error.ParseValue;
            var result: Citation = .{ .cited_text = undefined };
            inline for (field_names) |name| {
                if (dictionaryGet(input, name)) |field| {
                    const T = @TypeOf(@field(result, name));
                    if (T == []const u8) {
                        @field(result, name) = (try types.validate([]const u8, arena, field)).string;
                    } else if (T == ?[]const u8) {
                        @field(result, name) = if (field == .null) null else (try types.validate([]const u8, arena, field)).string;
                    } else {
                        const checked = if (field == .null) field else try values.coerceInteger(arena, field);
                        @field(result, name) = if (checked == .null) null else Integer.fromValue(checked);
                    }
                } else if (comptime std.mem.eql(u8, name, "cited_text")) return error.ParseValue;
            }
            return result;
        }

        /// Keyword construction requires a mapping with actual string keys.
        pub fn init(arena: Allocator, keywords: Value) Error!Citation {
            return Citation.fromKeywords(arena, keywords);
        }

        pub fn fromJson(arena: Allocator, text: []const u8) Error!Citation {
            return Citation.fromValue(arena, try parseModelJson(arena, text));
        }

        fn fromKeywords(arena: Allocator, input: Value) Error!Citation {
            try requireKeywords(input);
            return Citation.fromValue(arena, input);
        }

        pub fn validate(arena: Allocator, input: Value) Error!Value {
            return (try Citation.fromValue(arena, input)).toRuntime(arena);
        }

        /// Canonical parts have already enforced their optional-text invariant.
        pub fn fromPart(part: inference.CitationPart) Citation {
            return .{
                .cited_text = nonempty(part.text) orelse nonempty(part.title) orelse nonempty(part.url) orelse "",
                .document_title = part.title,
                .url = part.url,
            };
        }

        fn fieldValue(self: Citation, comptime name: []const u8) Value {
            const field = @field(self, name);
            const T = @TypeOf(field);
            if (T == []const u8) return .{ .string = field };
            if (T == ?[]const u8) return if (field) |text| .{ .string = text } else .null;
            return if (field) |integer| integer.value else .null;
        }

        /// format() is the raw dictionary with None-valued fields omitted.
        pub fn format(self: Citation, arena: Allocator) Allocator.Error!Value {
            return self.dump(arena, true);
        }

        pub fn dumpJson(self: Citation, arena: Allocator, round_trip: bool) Error![]const u8 {
            return strictJsonText(arena, if (round_trip) try self.toValue(arena) else try self.format(arena));
        }

        /// round_trip=True retains every source field, including null defaults.
        pub fn toValue(self: Citation, arena: Allocator) Allocator.Error!Value {
            return self.dump(arena, false);
        }

        fn dump(self: Citation, arena: Allocator, omit_null: bool) Allocator.Error!Value {
            var object: runtime.ObjectMap = .empty;
            try object.ensureTotalCapacity(arena, field_names.len);
            inline for (field_names) |name| {
                const field = self.fieldValue(name);
                if (!omit_null or field != .null) object.putAssumeCapacity(name, field);
            }
            return .{ .object = object };
        }

        pub fn toRuntime(self: Citation, arena: Allocator) Allocator.Error!Value {
            if (self.runtime_origin) |origin| return .{ .custom = origin };
            const boxed = try arena.create(Citation);
            boxed.* = self;
            return .{ .custom = .{ .ptr = boxed, .vtable = &Citation.runtime_vtable } };
        }

        /// A child stored inside Citations already has stable storage. Borrow
        /// that storage without allocating a second box; its deinit is a no-op.
        pub fn borrowedRuntime(self: *const Citation) Value {
            if (self.runtime_origin) |origin| return .{ .custom = origin.borrowed() };
            return .{ .custom = .{
                .ptr = self,
                .vtable = &Citation.runtime_vtable,
                .owns_box = false,
            } };
        }

        pub const Field = struct { name: []const u8, value: Value };
        pub const Iterator = struct {
            model: *const Citation,
            index: usize = 0,

            pub fn next(self: *Citation.Iterator) ?Citation.Field {
                inline for (field_names, 0..) |name, index| {
                    if (self.index == index) {
                        self.index += 1;
                        return .{ .name = name, .value = self.model.fieldValue(name) };
                    }
                }
                return null;
            }
        };

        /// Field rows borrow this model, as do its runtime iterator's tuples.
        pub fn iterator(self: *const Citation) Citation.Iterator {
            return .{ .model = self };
        }

        pub fn truthy(_: Citation) bool {
            return true;
        }

        pub fn eql(self: Citation, other: Citation) runtime.CustomError!bool {
            inline for (field_names) |name| {
                if (!try values.pyEqual(self.fieldValue(name), other.fieldValue(name))) return false;
            }
            return true;
        }

        pub fn writeStr(self: Citation, writer: *Writer) runtime.CustomWriteError!void {
            try self.writeDisplay(writer, false, 0);
        }

        pub fn writeRepr(self: Citation, writer: *Writer) runtime.CustomWriteError!void {
            try self.writeDisplay(writer, true, 0);
        }

        fn writeDisplay(self: Citation, writer: *Writer, repr: bool, depth: u32) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            if (repr) try writer.writeAll("Citation(");
            inline for (field_names, 0..) |name, index| {
                if (index != 0) try writer.writeAll(if (repr) ", " else " ");
                try writer.writeAll(name ++ "=");
                try pyjson.writeRepr(writer, self.fieldValue(name), depth + 1);
            }
            if (repr) try writer.writeByte(')');
        }

        fn writeFields(self: Citation, writer: *Writer, style: runtime.JsonStyle, depth: u32, omit_null: bool, projection: bool) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            try writer.writeByte('{');
            var first = true;
            inline for (field_names) |name| {
                const field = self.fieldValue(name);
                if (!omit_null or field != .null) {
                    try separator(writer, style, depth + 1, first);
                    first = false;
                    try writer.writeAll("\"" ++ name);
                    try writer.writeAll(if (style == .minified) "\":" else "\": ");
                    if (projection) {
                        try pyjson.writeJsonProjection(writer, field, style, depth + 1);
                    } else {
                        try writePlainJson(writer, field, style, depth + 1);
                    }
                }
            }
            try closeContainer(writer, style, depth, first, '}');
        }

        pub fn jsonStringify(self: Citation, writer: *std.json.Stringify) Writer.Error!void {
            try writer.beginObject();
            inline for (field_names) |name| {
                try writer.objectField(name);
                try writer.write(self.fieldValue(name));
            }
            try writer.endObject();
        }

        /// This result owns all fields and can be released with deinit().
        pub fn clone(self: Citation, allocator: Allocator) Allocator.Error!Citation {
            var context: runtime.CloneContext = .init(allocator);
            defer context.deinit();
            return self.cloneWithContext(&context);
        }

        /// Integer fields can retain arbitrary runtime values, so one erased
        /// clone must keep sharing the caller's box memo across all fields.
        fn cloneWithContext(self: Citation, context: *runtime.CloneContext) Allocator.Error!Citation {
            const allocator = context.allocator;
            var copy: Citation = .{ .type = "", .cited_text = "" };
            errdefer copy.deinit(allocator);
            inline for (field_names) |name| {
                const T = @TypeOf(@field(self, name));
                if (T == []const u8) {
                    @field(copy, name) = try allocator.dupe(u8, @field(self, name));
                } else if (T == ?[]const u8) {
                    @field(copy, name) = if (@field(self, name)) |text| try allocator.dupe(u8, text) else null;
                } else {
                    if (@field(self, name)) |integer| {
                        // Clone first: a failed in-place clone would leave a non-null optional with an undefined payload.
                        const value = try cloneField(context, integer.value);
                        @field(copy, name) = .{ .value = value };
                    }
                }
            }
            return copy;
        }

        /// Only owning clones adopt field storage. Constructor/decoder views
        /// borrow their strings and must not call this field destructor.
        pub fn deinit(self: Citation, allocator: Allocator) void {
            inline for (field_names) |name| {
                const T = @TypeOf(@field(self, name));
                if (T == []const u8) {
                    allocator.free(@field(self, name));
                } else if (T == ?[]const u8) {
                    if (@field(self, name)) |text| allocator.free(text);
                } else {
                    if (@field(self, name)) |integer| inference.freeJsonValue(allocator, integer.value);
                }
            }
        }

        const runtime_vtable: runtime.CustomVTable = .{
            .type_name = @typeName(Citation),
            .type_identity = runtime.typeIdentity(Citation),
            .model_name = Citation.model_name,
            .is_type = true,
            .iterate = Citation.iterateRuntime,
            .raw_value = Citation.rawRuntimeValue,
            .model_dump = Citation.modelDumpRuntimeValue,
            .serialized = Citation.serializedRuntimeValue,
            .write_str = Citation.writeRuntimeStr,
            .write_repr = Citation.writeRuntimeRepr,
            .write_json = Citation.writeRuntimeJson,
            .write_serialized_json = Citation.writeRuntimeSerializedJson,
            .clone = Citation.cloneRuntime,
            .deinit = Citation.deinitRuntime,
            .eql = Citation.equalRuntime,
        };

        fn actual(ptr: *const anyopaque) *const Citation {
            return @ptrCast(@alignCast(ptr));
        }
        fn iterateRuntime(arena: Allocator, ptr: *const anyopaque) Error![]Value {
            const result = try arena.alloc(Value, field_names.len);
            var initialized: usize = 0;
            errdefer {
                for (result[0..initialized]) |item| item.custom.deinit(arena);
                arena.free(result);
            }
            var fields_iterator = Citation.actual(ptr).iterator();
            while (fields_iterator.next()) |field| {
                const pair = try arena.dupe(Value, &.{ .{ .string = field.name }, field.value });
                errdefer arena.free(pair);
                result[initialized] = try python.tupleFromOwnedSlice(arena, pair);
                initialized += 1;
            }
            return result;
        }
        fn rawRuntimeValue(arena: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return Citation.actual(ptr).toValue(arena);
        }
        fn modelDumpRuntimeValue(arena: Allocator, ptr: *const anyopaque) Error!Value {
            return Citation.actual(ptr).format(arena);
        }
        fn serializedRuntimeValue(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
            return values.serializedValueDepth(arena, try Citation.actual(ptr).format(arena), depth + 1);
        }
        fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            try Citation.actual(ptr).writeStr(writer);
        }
        fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
            try Citation.actual(ptr).writeDisplay(writer, true, depth);
        }
        fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
            try writer.write(Citation.actual(ptr).*);
        }
        fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            try Citation.actual(ptr).writeFields(writer, style, @intCast(depth), true, true);
        }
        fn cloneRuntime(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
            const allocator = context.allocator;
            const copy = try allocator.create(Citation);
            errdefer allocator.destroy(copy);
            copy.* = try Citation.actual(ptr).cloneWithContext(context);
            return .{ .ptr = copy, .vtable = &Citation.runtime_vtable, .owns_data = true };
        }
        fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
            if (owns_data) Citation.actual(ptr).deinit(allocator);
            allocator.destroy(Citation.actual(ptr));
        }
        fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
            return left == right or try Citation.actual(left).eql(Citation.actual(right).*);
        }
    };

    /// The before-model validator deliberately accepts only a list of citation
    /// dictionaries, a dictionary containing a list, or one citation dictionary.
    pub fn fromValue(arena: Allocator, input: Value) Error!Citations {
        if (input == .custom) {
            if (input.custom.get(Citations)) |existing| {
                var result = existing.*;
                result.runtime_origin = input.custom;
                return result;
            }
        }
        if (input == .array) {
            for (input.array.items) |item| {
                if (!isDictionary(item) or dictionaryGet(item, "cited_text") == null) return error.ParseValue;
            }
            return fromDictList(arena, input);
        }
        if (!isDictionary(input)) return error.ParseValue;
        if (dictionaryGet(input, "citations")) |items| {
            if (items != .array) return error.ParseValue;
            const result = try arena.alloc(Citation, items.array.items.len);
            for (items.array.items, result) |item, *slot| {
                slot.* = if (isDictionary(item)) try Citation.fromKeywords(arena, item) else try Citation.fromValue(arena, item);
            }
            return .{ .citations = result };
        }
        if (dictionaryGet(input, "cited_text") != null) {
            const result = try arena.alloc(Citation, 1);
            result[0] = try Citation.fromKeywords(arena, input);
            return .{ .citations = result };
        }
        return error.ParseValue;
    }

    pub fn init(arena: Allocator, keywords: Value) Error!Citations {
        try requireKeywords(keywords);
        return fromValue(arena, keywords);
    }

    pub fn fromJson(arena: Allocator, text: []const u8) Error!Citations {
        return fromValue(arena, try parseModelJson(arena, text));
    }

    /// Unlike the annotation validator, the source classmethod accepts any
    /// iterable, then applies Citation(**item) to every element.
    pub fn fromDictList(arena: Allocator, input: Value) Error!Citations {
        const items = try iterableItems(arena, input);
        const result = try arena.alloc(Citation, items.len);
        for (items, result) |item, *slot| slot.* = try Citation.fromKeywords(arena, item);
        return .{ .citations = result };
    }

    pub fn validate(arena: Allocator, input: Value) Error!Value {
        return (try fromValue(arena, input)).toRuntime(arena);
    }

    pub fn fromParts(arena: Allocator, parts: []const inference.CitationPart) Allocator.Error!Citations {
        const result = try arena.alloc(Citation, parts.len);
        for (parts, result) |part, *slot| slot.* = Citation.fromPart(part);
        return .{ .citations = result };
    }

    pub fn parseLmResponse(arena: Allocator, response: @import("lm.zig").Completion) Allocator.Error!?Citations {
        if (response.citations.len == 0) return null;
        return try fromParts(arena, response.citations);
    }

    pub fn isStreamable() bool {
        return true;
    }

    pub fn len(self: Citations) usize {
        return self.citations.len;
    }

    pub fn truthy(self: Citations) bool {
        return self.len() != 0;
    }

    pub const Iterator = struct {
        items: []const Citation,
        index: usize = 0,

        pub fn next(self: *Iterator) ?*const Citation {
            if (self.index == self.items.len) return null;
            const result = &self.items[self.index];
            self.index += 1;
            return result;
        }
    };

    pub fn iterator(self: Citations) Iterator {
        return .{ .items = self.citations };
    }

    pub fn get(self: Citations, index: isize) error{ParseValue}!*const Citation {
        const position = if (index < 0) self.len() -| @abs(index) else @as(usize, @intCast(index));
        if ((index < 0 and @abs(index) > self.len()) or position >= self.len()) return error.ParseValue;
        return &self.citations[position];
    }

    /// Python list indexing accepts bool and actual integers, not strings or
    /// integral floats. Arbitrary-width indexes fail without narrowing first.
    pub fn indexValue(self: Citations, index: Value) error{ParseValue}!Value {
        const position: isize = switch (index) {
            .bool => |b| @intFromBool(b),
            .integer => |i| std.math.cast(isize, i) orelse return error.ParseValue,
            .number_string => |text| std.fmt.parseInt(isize, text, 10) catch return error.ParseValue,
            else => return error.ParseValue,
        };
        return self.getRuntime(position);
    }

    pub const Slice = struct {
        start: ?isize = null,
        stop: ?isize = null,
        step: isize = 1,
    };

    /// A slice is a new list, not another Citations model. Its actual Citation
    /// elements retain their original boxes and borrow the original storage.
    pub fn slice(self: Citations, arena: Allocator, range: Slice) Error![]const Citation {
        if (range.step == 0) return error.ParseValue;
        const length = std.math.cast(isize, self.len()) orelse return error.ParseValue;
        const reverse = range.step < 0;
        const start = sliceBound(range.start, length, reverse, if (reverse) length - 1 else 0);
        const stop = sliceBound(range.stop, length, reverse, if (reverse) -1 else length);
        const distance: usize = if (reverse)
            if (start > stop) @intCast(start - stop) else 0
        else if (stop > start) @intCast(stop - start) else 0;
        const stride = @abs(range.step);
        const count = if (distance == 0) 0 else 1 + (distance - 1) / stride;
        const result = try arena.alloc(Citation, count);
        var position = start;
        for (result, 0..) |*slot, index| {
            const original = &self.citations[@intCast(position)];
            slot.* = original.*;
            slot.runtime_origin = original.borrowedRuntime().custom;
            if (index + 1 != count) position += range.step;
        }
        return result;
    }

    pub fn contains(self: Citations, item: Value) runtime.CustomError!bool {
        for (self.citations) |*element| {
            if (try values.pyEqual(element.borrowedRuntime(), item)) return true;
        }
        return false;
    }

    pub fn getRuntime(self: Citations, index: isize) error{ParseValue}!Value {
        return (try self.get(index)).borrowedRuntime();
    }

    pub fn eql(self: Citations, other: Citations) runtime.CustomError!bool {
        if (self.len() != other.len()) return false;
        for (self.citations, other.citations) |a, b| if (!try a.eql(b)) return false;
        return true;
    }

    pub fn toRuntime(self: Citations, arena: Allocator) Allocator.Error!Value {
        if (self.runtime_origin) |origin| return .{ .custom = origin };
        const boxed = try arena.create(Citations);
        boxed.* = self;
        return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
    }

    /// Borrow stable collection storage without creating an outer model box.
    pub fn borrowedRuntime(self: *const Citations) Value {
        if (self.runtime_origin) |origin| return .{ .custom = origin.borrowed() };
        return .{ .custom = .{
            .ptr = self,
            .vtable = &runtime_vtable,
            .owns_box = false,
        } };
    }

    /// Raw round-trip fields, not the string produced by format().
    pub fn toValue(self: Citations, arena: Allocator) Allocator.Error!Value {
        var array: runtime.Array = .init(arena);
        try array.ensureTotalCapacity(self.len());
        for (self.citations) |item| array.appendAssumeCapacity(try item.toValue(arena));
        var object: runtime.ObjectMap = .empty;
        try object.put(arena, "citations", .{ .array = array });
        return .{ .object = object };
    }

    pub fn dumpJson(self: Citations, arena: Allocator, round_trip: bool) Error![]const u8 {
        return strictJsonText(arena, if (round_trip) try self.toValue(arena) else .{ .string = try self.format(arena) });
    }

    /// The source string is a JSON list, not {"citations": [...]}.
    pub fn format(self: Citations, arena: Allocator) Error![]const u8 {
        var output: Writer.Allocating = .init(arena);
        errdefer output.deinit();
        self.writeFormat(&output.writer, 0) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.ParseValue, error.JsonTooDeep => error.ParseValue,
        };
        return output.toOwnedSlice();
    }

    /// Format one canonical part without allocating intermediate model storage.
    pub fn writePart(part: inference.CitationPart, writer: *Writer) runtime.CustomWriteError!void {
        const citations: Citations = .{ .citations = &.{Citation.fromPart(part)} };
        try citations.writeFormat(writer, 0);
    }

    fn writeFormat(self: Citations, writer: *Writer, depth: u32) runtime.CustomWriteError!void {
        if (depth >= pyjson.max_depth) return error.JsonTooDeep;
        try writer.writeByte('[');
        for (self.citations, 0..) |item, index| {
            if (index != 0) try writer.writeAll(", ");
            try item.writeFields(writer, .compact, depth + 1, true, false);
        }
        try writer.writeByte(']');
    }

    pub fn writeStr(self: Citations, writer: *Writer) runtime.CustomWriteError!void {
        try self.writeDisplay(writer, false, 0);
    }

    pub fn writeRepr(self: Citations, writer: *Writer) runtime.CustomWriteError!void {
        try self.writeDisplay(writer, true, 0);
    }

    fn writeDisplay(self: Citations, writer: *Writer, repr: bool, depth: u32) runtime.CustomWriteError!void {
        if (depth >= pyjson.max_depth - 1) return error.JsonTooDeep;
        try writer.writeAll(if (repr) "Citations(citations=[" else "citations=[");
        for (self.citations, 0..) |item, index| {
            if (index != 0) try writer.writeAll(", ");
            try item.writeDisplay(writer, true, depth + 1);
        }
        try writer.writeAll(if (repr) "])" else "]");
    }

    pub fn jsonStringify(self: Citations, writer: *std.json.Stringify) Writer.Error!void {
        try writer.beginObject();
        try writer.objectField("citations");
        try writer.write(self.citations);
        try writer.endObject();
    }

    pub fn clone(self: Citations, allocator: Allocator) Allocator.Error!Citations {
        var context: runtime.CloneContext = .init(allocator);
        defer context.deinit();
        return self.cloneWithContext(&context);
    }

    fn cloneWithContext(self: Citations, context: *runtime.CloneContext) Allocator.Error!Citations {
        const allocator = context.allocator;
        const result = try allocator.alloc(Citation, self.len());
        var initialized: usize = 0;
        errdefer {
            for (result[0..initialized]) |item| item.deinit(allocator);
            allocator.free(result);
        }
        for (self.citations, result) |item, *slot| {
            slot.* = try item.cloneWithContext(context);
            initialized += 1;
        }
        return .{ .citations = result };
    }

    pub fn deinit(self: Citations, allocator: Allocator) void {
        for (self.citations) |item| item.deinit(allocator);
        allocator.free(self.citations);
    }

    const runtime_vtable: runtime.CustomVTable = .{
        .type_name = @typeName(Citations),
        .type_identity = runtime.typeIdentity(Citations),
        .model_name = model_name,
        .is_type = true,
        .bool_value = runtimeTruthy,
        .iterate = iterateRuntime,
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

    fn actual(ptr: *const anyopaque) *const Citations {
        return @ptrCast(@alignCast(ptr));
    }
    fn runtimeTruthy(ptr: *const anyopaque) error{ParseValue}!bool {
        return actual(ptr).truthy();
    }
    fn iterateRuntime(arena: Allocator, ptr: *const anyopaque) Error![]Value {
        const result = try arena.alloc(Value, actual(ptr).len());
        for (actual(ptr).citations, result) |*item, *slot| slot.* = item.borrowedRuntime();
        return result;
    }
    fn rawRuntimeValue(arena: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return actual(ptr).toValue(arena);
    }
    fn modelDumpRuntimeValue(arena: Allocator, ptr: *const anyopaque) Error!Value {
        return .{ .string = try actual(ptr).format(arena) };
    }
    fn serializedRuntimeValue(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
        if (depth >= pyjson.max_depth) return error.ParseValue;
        return .{ .string = try actual(ptr).format(arena) };
    }
    fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        try actual(ptr).writeStr(writer);
    }
    fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
        try actual(ptr).writeDisplay(writer, true, depth);
    }
    fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
        try writer.write(actual(ptr).*);
    }
    fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
        if (depth >= pyjson.max_depth) return error.JsonTooDeep;
        try writer.writeByte('"');
        var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
        try actual(ptr).writeFormat(&escaped.writer, @intCast(depth));
        try writer.writeByte('"');
    }
    fn cloneRuntime(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const allocator = context.allocator;
        const copy = try allocator.create(Citations);
        errdefer allocator.destroy(copy);
        copy.* = try actual(ptr).cloneWithContext(context);
        return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
    }
    fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        if (owns_data) actual(ptr).deinit(allocator);
        allocator.destroy(actual(ptr));
    }
    fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
        return left == right or try actual(left).eql(actual(right).*);
    }

    /// Explicit listener object identities: source EngineChunk/AttributeDict
    /// are dictionary subclasses with attribute access; SimpleNamespace has
    /// attributes but no dictionary .get(). Plain dictionaries remain plain.
    pub const StreamObject = struct {
        pub const Kind = enum { namespace, attribute_dict, engine_chunk };
        attributes: runtime.ObjectMap,
        kind: Kind = .namespace,

        pub fn toRuntime(self: StreamObject, arena: Allocator) Allocator.Error!Value {
            const boxed = try arena.create(StreamObject);
            boxed.* = self;
            return .{ .custom = .{ .ptr = boxed, .vtable = table(self.kind) } };
        }

        const vtable: runtime.CustomVTable = .{
            .type_name = @typeName(StreamObject),
            .type_identity = runtime.typeIdentity(StreamObject),
            .model_name = "SimpleNamespace",
            .bool_value = boolValue,
            .iterate = iterate,
            .eql_value = equalValue,
            .raw_value = rawValue,
            .serialized = serialize,
            .write_str = StreamObject.writeStr,
            .write_repr = StreamObject.writeRepr,
            .write_json = writeJson,
            .write_serialized_json = writeSerialized,
            .clone = cloneObject,
            .deinit = deinitObject,
            .eql = equal,
        };
        const attribute_dict_vtable: runtime.CustomVTable = blk: {
            var result = vtable;
            result.model_name = "AttributeDict";
            break :blk result;
        };
        const engine_chunk_vtable: runtime.CustomVTable = blk: {
            var result = vtable;
            result.model_name = "EngineChunk";
            break :blk result;
        };
        fn table(kind: Kind) *const runtime.CustomVTable {
            return switch (kind) {
                .namespace => &vtable,
                .attribute_dict => &attribute_dict_vtable,
                .engine_chunk => &engine_chunk_vtable,
            };
        }
        fn boolValue(ptr: *const anyopaque) error{ParseValue}!bool {
            return object(ptr).kind == .namespace or object(ptr).attributes.count() != 0;
        }
        fn iterate(arena: Allocator, ptr: *const anyopaque) Error![]Value {
            if (object(ptr).kind == .namespace) return error.ParseValue;
            const result = try arena.alloc(Value, object(ptr).attributes.count());
            for (object(ptr).attributes.keys(), result) |key, *slot| slot.* = .{ .string = key };
            return result;
        }
        fn object(ptr: *const anyopaque) *const StreamObject {
            return @ptrCast(@alignCast(ptr));
        }
        fn rawValue(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return .{ .object = object(ptr).attributes };
        }
        fn serialize(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
            if (object(ptr).kind == .namespace) return error.PythonSerializeFallback;
            return values.serializedValueDepth(arena, .{ .object = object(ptr).attributes }, depth + 1);
        }
        fn writeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            try StreamObject.writeRepr(writer, ptr, 0);
        }
        fn writeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            if (object(ptr).kind != .namespace) return pyjson.writeRepr(writer, .{ .object = object(ptr).attributes }, depth);
            try writer.writeAll("namespace(");
            for (object(ptr).attributes.keys(), object(ptr).attributes.values(), 0..) |key, item, index| {
                if (index != 0) try writer.writeAll(", ");
                try writer.writeAll(key);
                try writer.writeByte('=');
                try pyjson.writeRepr(writer, item, depth + 1);
            }
            try writer.writeByte(')');
        }
        fn writeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
            try writer.write(Value{ .object = object(ptr).attributes });
        }
        fn writeSerialized(writer: *Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
            if (object(ptr).kind == .namespace) return error.ParseValue;
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            try pyjson.writeJsonProjection(writer, .{ .object = object(ptr).attributes }, style, @intCast(depth));
        }
        fn cloneObject(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
            const allocator = context.allocator;
            const copy = try allocator.create(StreamObject);
            errdefer allocator.destroy(copy);
            copy.* = .{ .attributes = (try cloneField(context, .{ .object = object(ptr).attributes })).object, .kind = object(ptr).kind };
            return .{ .ptr = copy, .vtable = table(copy.kind), .owns_data = true };
        }
        fn deinitObject(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
            if (owns_data) inference.freeJsonValue(allocator, .{ .object = object(ptr).attributes });
            allocator.destroy(object(ptr));
        }
        fn equal(a: *const anyopaque, b: *const anyopaque) runtime.CustomError!bool {
            if ((object(a).kind == .namespace) != (object(b).kind == .namespace)) return false;
            return a == b or try values.recordEqual(&object(a).attributes, &object(b).attributes);
        }
        fn equalValue(ptr: *const anyopaque, other: Value) runtime.CustomError!bool {
            return object(ptr).kind != .namespace and other == .object and try values.recordEqual(&object(ptr).attributes, &other.object);
        }
    };

    pub fn parseStreamChunk(arena: Allocator, chunk: Value) runtime.CustomError!?Citations {
        const choices = attribute(chunk, "choices") orelse return null;
        if (!try truth(choices)) return null;
        const first = try indexZero(choices) orelse return null;
        const delta = attribute(first, "delta") orelse return null;
        const fields = attribute(delta, "provider_specific_fields") orelse return null;
        if (!try truth(fields)) return null;
        const data = dictionaryGet(fields, "citation") orelse return null;
        if (!try truth(data) or !isDictionary(data)) return null;
        var part: inference.CitationPart = .{};
        inline for (.{ "text", "title", "url" }) |name| {
            if (dictionaryGet(data, name)) |item| {
                if (item != .null) {
                    if (item != .string or item.string.len == 0) return null;
                    @field(part, name) = item.string;
                }
            }
        }
        if (part.text == null and part.title == null and part.url == null) return null;
        return try fromParts(arena, &.{part});
    }
};

fn parseModelJson(arena: Allocator, text: []const u8) runtime.CustomError!Value {
    const parsed = @import("json_repair.zig").parseJson(arena, text) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.ParseValue,
    };
    if (!validJsonStrings(parsed, 0)) return error.ParseValue;
    return parsed;
}

fn strictJsonText(arena: Allocator, input: Value) runtime.CustomError![]const u8 {
    const projected = try values.serializedValue(arena, input);
    if (!validJsonStrings(projected, 0)) return error.ParseValue;
    var output: Writer.Allocating = .init(arena);
    errdefer output.deinit();
    pyjson.writeJson(&output.writer, projected, .minified) catch |err| return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        error.ParseValue, error.JsonTooDeep => error.ParseValue,
    };
    return output.toOwnedSlice();
}

fn validJsonStrings(input: Value, depth: u32) bool {
    if (depth >= pyjson.max_depth) return false;
    return switch (input) {
        .string => |text| std.unicode.utf8ValidateSlice(text),
        .array => |array| blk: {
            for (array.items) |item| if (!validJsonStrings(item, depth + 1)) break :blk false;
            break :blk true;
        },
        .object => |object| blk: {
            for (object.keys(), object.values()) |key, item| {
                if (!std.unicode.utf8ValidateSlice(key) or !validJsonStrings(item, depth + 1)) break :blk false;
            }
            break :blk true;
        },
        .custom => false,
        else => true,
    };
}

fn sliceBound(bound: ?isize, length: isize, reverse: bool, fallback: isize) isize {
    var index = bound orelse return fallback;
    if (index < 0) index = if (index < -length) (if (reverse) -1 else 0) else index + length;
    return std.math.clamp(index, @as(isize, if (reverse) -1 else 0), if (reverse) length - 1 else length);
}

fn nonempty(text: ?[]const u8) ?[]const u8 {
    return if (text) |s| if (s.len != 0) s else null else null;
}

fn isDictionary(input: Value) bool {
    if (input == .object) return true;
    if (input == .custom) {
        if (input.custom.get(Citations.StreamObject)) |object| return object.kind != .namespace;
    }
    const payload = python.view(input) orelse return false;
    return payload == .dictionary;
}

fn dictionaryGet(input: Value, name: []const u8) ?Value {
    if (input == .object) return input.object.get(name);
    if (input == .custom) {
        if (input.custom.get(Citations.StreamObject)) |object| {
            return if (object.kind != .namespace) object.attributes.get(name) else null;
        }
    }
    const payload = python.view(input) orelse return null;
    if (payload != .dictionary) return null;
    for (payload.dictionary) |entry| {
        if (entry.key == .string and std.mem.eql(u8, entry.key.string, name)) return entry.value;
    }
    return null;
}

fn requireKeywords(input: Value) error{ParseValue}!void {
    if (input == .object) return;
    if (input == .custom) {
        if (input.custom.get(Citations.StreamObject)) |object| {
            return if (object.kind != .namespace) {} else error.ParseValue;
        }
    }
    const payload = python.view(input) orelse return error.ParseValue;
    if (payload != .dictionary) return error.ParseValue;
    for (payload.dictionary) |entry| if (entry.key != .string) return error.ParseValue;
}

fn iterableItems(arena: Allocator, input: Value) runtime.CustomError![]const Value {
    return switch (input) {
        .array => |array| array.items,
        .string => |text| if (text.len == 0) &.{} else error.ParseValue,
        .object => |object| if (object.count() == 0) &.{} else error.ParseValue,
        .custom => |custom| blk: {
            if (custom.vtable.iterate) |iterate| break :blk try iterate(arena, custom.ptr);
            const payload = python.view(input) orelse break :blk error.ParseValue;
            break :blk switch (payload) {
                .tuple, .set, .frozen_set => |items| items,
                .bytes => |data| if (data.len == 0) &.{} else error.ParseValue,
                .dictionary => |entries| if (entries.len == 0) &.{} else error.ParseValue,
                else => error.ParseValue,
            };
        },
        else => error.ParseValue,
    };
}

fn attribute(input: Value, name: []const u8) ?Value {
    if (input != .custom) return null;
    const object = input.custom.get(Citations.StreamObject) orelse return null;
    return object.attributes.get(name);
}

fn indexZero(input: Value) runtime.CustomError!?Value {
    if (input == .array) return if (input.array.items.len != 0) input.array.items[0] else null;
    const payload = python.view(input) orelse return null;
    return switch (payload) {
        .tuple => |items| if (items.len != 0) items[0] else null,
        .dictionary => |entries| blk: {
            for (entries) |entry| if (try values.pyEqual(entry.key, .{ .integer = 0 })) break :blk entry.value;
            break :blk null;
        },
        else => null,
    };
}

fn truth(input: Value) error{ParseValue}!bool {
    return switch (input) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .number_string => |s| !std.mem.eql(u8, s, "0") and !std.mem.eql(u8, s, "-0"),
        .string => |s| s.len != 0,
        .array => |a| a.items.len != 0,
        .object => |o| o.count() != 0,
        .custom => |custom| try custom.truthy(),
    };
}

fn separator(writer: *Writer, style: runtime.JsonStyle, depth: u32, first: bool) Writer.Error!void {
    switch (style) {
        .compact => if (!first) try writer.writeAll(", "),
        .minified => if (!first) try writer.writeByte(','),
        .indent2, .orjson_indent2 => {
            if (!first) try writer.writeByte(',');
            try writer.writeByte('\n');
            try writer.splatByteAll(' ', depth * 2);
        },
    }
}

fn closeContainer(writer: *Writer, style: runtime.JsonStyle, depth: u32, empty: bool, closing: u8) Writer.Error!void {
    if (!empty and (style == .indent2 or style == .orjson_indent2)) {
        try writer.writeByte('\n');
        try writer.splatByteAll(' ', depth * 2);
    }
    try writer.writeByte(closing);
}

/// Citation.format feeds Python json.dumps, not DSPy's custom serializer.
/// Validated Integer normally holds an integer, but the actual Enum bypass
/// can retain a container or an otherwise non-JSON value.
fn writePlainJson(writer: *Writer, input: Value, style: runtime.JsonStyle, depth: u32) runtime.CustomWriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    switch (input) {
        .array => |items| {
            try writer.writeByte('[');
            for (items.items, 0..) |item, index| {
                try separator(writer, style, depth + 1, index == 0);
                try writePlainJson(writer, item, style, depth + 1);
            }
            try closeContainer(writer, style, depth, items.items.len == 0, ']');
        },
        .object => |object| {
            try writer.writeByte('{');
            for (object.keys(), object.values(), 0..) |key, item, index| {
                try separator(writer, style, depth + 1, index == 0);
                try pyjson.writeJsonString(writer, key);
                try writer.writeAll(if (style == .minified) ":" else ": ");
                try writePlainJson(writer, item, style, depth + 1);
            }
            try closeContainer(writer, style, depth, object.count() == 0, '}');
        },
        .custom => {
            const payload = python.view(input) orelse return error.ParseValue;
            switch (payload) {
                .tuple => |items| {
                    try writer.writeByte('[');
                    for (items, 0..) |item, index| {
                        try separator(writer, style, depth + 1, index == 0);
                        try writePlainJson(writer, item, style, depth + 1);
                    }
                    try closeContainer(writer, style, depth, items.len == 0, ']');
                },
                .dictionary => |entries| {
                    try writer.writeByte('{');
                    for (entries, 0..) |entry, index| {
                        try separator(writer, style, depth + 1, index == 0);
                        if (entry.key == .string) {
                            try pyjson.writeJsonString(writer, entry.key.string);
                        } else switch (entry.key) {
                            .null, .bool, .integer, .float, .number_string => {
                                try writer.writeByte('"');
                                try pyjson.writeJsonDepth(writer, entry.key, .compact, depth + 1, false);
                                try writer.writeByte('"');
                            },
                            else => return error.ParseValue,
                        }
                        try writer.writeAll(if (style == .minified) ":" else ": ");
                        try writePlainJson(writer, entry.value, style, depth + 1);
                    }
                    try closeContainer(writer, style, depth, entries.len == 0, '}');
                },
                .float => try pyjson.writeJsonDepth(writer, input, style, depth, false),
                else => return error.ParseValue,
            }
        },
        else => try pyjson.writeJsonDepth(writer, input, style, depth, false),
    }
}

/// The generic arena clone is intentionally leaky on allocation failure.
/// Owning model clones instead unwind every initialized nested allocation.
fn cloneField(context: *runtime.CloneContext, input: Value) Allocator.Error!Value {
    const allocator = context.allocator;
    switch (input) {
        .array => |array| {
            var copy: runtime.Array = .init(allocator);
            errdefer {
                for (copy.items) |item| inference.freeJsonValue(allocator, item);
                copy.deinit();
            }
            try copy.ensureTotalCapacity(array.items.len);
            for (array.items) |item| copy.appendAssumeCapacity(try cloneField(context, item));
            return .{ .array = copy };
        },
        .object => |object| {
            var copy: runtime.ObjectMap = .empty;
            errdefer inference.freeJsonValue(allocator, .{ .object = copy });
            try copy.ensureTotalCapacity(allocator, object.count());
            for (object.keys(), object.values()) |key, item| {
                const owned_key = try allocator.dupe(u8, key);
                errdefer allocator.free(owned_key);
                copy.putAssumeCapacity(owned_key, try cloneField(context, item));
            }
            return .{ .object = copy };
        },
        else => return context.cloneValue(input),
    }
}

test "Citation lax fields retain arbitrary integers and genuine runtime identity" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const input = try @import("json_repair.zig").parseJson(arena,
        \\{"cited_text":"A","document_index":"1_000","start_char_index":18446744073709551616,"end_char_index":1.0,"extra":"ignored"}
    );
    const boxed = try types.validate(Citations.Citation, arena, input);
    const typed = boxed.custom.get(Citations.Citation).?;
    try std.testing.expectEqualStrings("char_location", typed.type);
    try std.testing.expectEqual(@as(i64, 1000), typed.document_index.?.value.integer);
    try std.testing.expectEqualStrings("18446744073709551616", typed.start_char_index.?.value.number_string);
    try std.testing.expectEqual(@as(i64, 1), typed.end_char_index.?.value.integer);
    try std.testing.expect(!try values.pyEqual(boxed, input));
    try std.testing.expectEqual(boxed.custom.ptr, (try types.validate(Citations.Citation, arena, boxed)).custom.ptr);
    const view = try values.fromValue(arena, Citations.Citation, boxed);
    try std.testing.expectEqual(boxed.custom.ptr, (try values.toValue(arena, Citations.Citation, view)).custom.ptr);
    const formatted = try typed.format(arena);
    try std.testing.expectEqual(@as(usize, 5), formatted.object.count());
    try std.testing.expect(formatted.object.get("url") == null);
    try std.testing.expectEqual(@as(usize, 8), (try typed.toValue(arena)).object.count());

    const binary = try python.bytes(arena, "bytes");
    var fields: runtime.ObjectMap = .empty;
    try fields.put(arena, "cited_text", binary);
    try fields.put(arena, "document_index", try python.bytes(arena, " 2 "));
    const bytes_model = try Citations.Citation.fromValue(arena, .{ .object = fields });
    try std.testing.expectEqualStrings("bytes", bytes_model.cited_text);
    try std.testing.expectEqual(@as(i64, 2), bytes_model.document_index.?.value.integer);
    for ([_][]const u8{
        "null",                                              "[]", "{}", "{\"cited_text\":0}", "{\"cited_text\":null}",
        "{\"cited_text\":\"A\",\"document_index\":\"1e3\"}",
        "{\"cited_text\":\"A\",\"document_index\":\"１\"}",
        "{\"cited_text\":\"A\",\"document_index\":1.5}",
    }) |text| {
        const bad = try @import("json_repair.zig").parseJson(arena, text);
        try std.testing.expectError(error.ParseValue, Citations.Citation.fromValue(arena, bad));
    }
    var wide: runtime.ObjectMap = .empty;
    try wide.put(arena, "cited_text", .{ .string = "wide float" });
    try wide.put(arena, "document_index", .{ .float = 1e30 });
    try std.testing.expectError(error.ParseValue, Citations.Citation.fromValue(arena, .{ .object = wide }));
    try wide.put(arena, "document_index", .{ .float = -0x1p63 });
    try std.testing.expectError(error.ParseValue, Citations.Citation.fromValue(arena, .{ .object = wide }));
}

test "Citations source validator differs from from_dict_list and preserves nested identity" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const child = try (Citations.Citation{ .cited_text = "A" }).toRuntime(arena);
    var items: runtime.Array = .init(arena);
    try items.append(child);
    try std.testing.expectError(error.ParseValue, Citations.fromValue(arena, .{ .array = items }));
    var wrapped: runtime.ObjectMap = .empty;
    try wrapped.put(arena, "citations", .{ .array = items });
    const parent = try types.validate(Citations, arena, .{ .object = wrapped });
    const collection = parent.custom.get(Citations).?;
    try std.testing.expectEqual(child.custom.ptr, (try collection.getRuntime(0)).custom.ptr);
    try std.testing.expectEqual(child.custom.ptr, (try collection.getRuntime(-1)).custom.ptr);
    try std.testing.expect(try collection.contains(child));
    try std.testing.expect(try parent.custom.truthy());
    try std.testing.expectEqual(parent.custom.ptr, (try types.validate(Citations, arena, parent)).custom.ptr);
    const iterator = try parent.custom.vtable.iterate.?(arena, parent.custom.ptr);
    try std.testing.expectEqual(child.custom.ptr, iterator[0].custom.ptr);
    try std.testing.expectError(error.ParseValue, collection.indexValue(.{ .float = 0 }));
    try std.testing.expectError(error.ParseValue, collection.indexValue(.{ .number_string = "18446744073709551616" }));
    try std.testing.expectEqual(child.custom.ptr, (try collection.indexValue(.{ .bool = false })).custom.ptr);
    const reverse = try collection.slice(arena, .{ .step = -1 });
    try std.testing.expectEqual(child.custom.ptr, (try reverse[0].toRuntime(arena)).custom.ptr);
    try std.testing.expectError(error.ParseValue, collection.slice(arena, .{ .step = 0 }));

    const dictionary = try @import("json_repair.zig").parseJson(arena, "{\"cited_text\":\"B\"}");
    const tuple = try python.tuple(arena, &.{dictionary});
    const from_tuple = try Citations.fromDictList(arena, tuple);
    try std.testing.expectEqualStrings("B", from_tuple.citations[0].cited_text);
    try std.testing.expectError(error.ParseValue, Citations.fromValue(arena, tuple));
    try std.testing.expect(!(try Citations.fromDictList(arena, .{ .string = "" })).truthy());
    try std.testing.expect(!(try Citations.fromValue(arena, .{ .array = .init(arena) })).truthy());
}

test "Citations own cloned fields and borrowed child classes survive their source arena" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    const arena = scratch.allocator();
    const input = try @import("json_repair.zig").parseJson(arena,
        \\[{"cited_text":"雪","document_title":"Title","url":"https://example.invalid","document_index":18446744073709551616}]
    );
    const original = try Citations.validate(arena, input);
    const cloned = try original.custom.clone(std.testing.allocator);
    defer cloned.deinit(std.testing.allocator);
    scratch.deinit();
    const owned = cloned.get(Citations).?;
    try std.testing.expectEqualStrings("雪", owned.citations[0].cited_text);
    try std.testing.expectEqualStrings("Title", owned.citations[0].document_title.?);
    try std.testing.expectEqualStrings("18446744073709551616", owned.citations[0].document_index.?.value.number_string);
    const interior = try owned.getRuntime(0);
    interior.custom.deinit(std.testing.allocator);
    const cloned_child = try interior.custom.clone(std.testing.allocator);
    defer cloned_child.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("雪", cloned_child.get(Citations.Citation).?.cited_text);
    try std.testing.expect(try values.pyEqual(interior, .{ .custom = cloned_child }));
    var buffer: [1024]u8 = undefined;
    var writer: Writer = .fixed(&buffer);
    try pyjson.writeJson(&writer, .{ .custom = cloned }, .compact);
    try std.testing.expectEqualStrings(
        "\"[{\\\"type\\\": \\\"char_location\\\", \\\"cited_text\\\": \\\"雪\\\", \\\"document_index\\\": 18446744073709551616, \\\"document_title\\\": \\\"Title\\\", \\\"url\\\": \\\"https://example.invalid\\\"}]\"",
        writer.buffered(),
    );
}

test "citation schema and direct-only native metadata retain text and XML conventions" {
    const signature = @import("signature.zig");
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct {
            direct: Citations,
            optional: ?Citations,
            citation: Citations.Citation,
            citations: []const Citations.Citation,
            nested: []const []const Citations.Citation,
        };
    });
    const fields = Sig.output_fields_list;
    try std.testing.expect(fields[0].native_citations);
    for (fields[1..]) |field| try std.testing.expect(!field.native_citations);
    try std.testing.expectEqualStrings(Citations.description, fields[0].type_description);
    try std.testing.expectEqualStrings(Citations.description, fields[1].type_description);
    try std.testing.expect(fields[0].xml_literal and fields[1].xml_literal and fields[2].xml_literal and fields[3].xml_literal);
    try std.testing.expect(!fields[4].xml_literal);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const schema = try @import("json_repair.zig").parseJson(arena, types.schemaXml(Citations));
    const properties = schema.object.get("properties").?.object;
    try std.testing.expectEqual(@as(usize, 1), properties.count());
    const child = schema.object.get("$defs").?.object.get("Citation").?.object;
    try std.testing.expectEqualStrings(Citations.Citation.schema_description, child.get("description").?.string);
    const child_properties = child.get("properties").?.object;
    try std.testing.expectEqual(@as(usize, 8), child_properties.count());
    try std.testing.expectEqualStrings("char_location", child_properties.get("type").?.object.get("default").?.string);
    try std.testing.expectEqualStrings("Document Index", child_properties.get("document_index").?.object.get("title").?.string);
    try std.testing.expectEqual(@as(usize, 1), child.get("required").?.array.items.len);
}

test "citation stream chunks require genuine attribute objects and validate canonical optional text" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const raw = try @import("json_repair.zig").parseJson(arena,
        \\{"choices":[{"delta":{"provider_specific_fields":{"citation":{"text":"T","title":"Title"}}}}]}
    );
    try std.testing.expect((try Citations.parseStreamChunk(arena, raw)) == null);
    const raw_choice = raw.object.get("choices").?.array.items[0];
    const delta = try (Citations.StreamObject{ .attributes = raw_choice.object.get("delta").?.object, .kind = .attribute_dict }).toRuntime(arena);
    var choice_fields: runtime.ObjectMap = .empty;
    try choice_fields.put(arena, "delta", delta);
    const choice = try (Citations.StreamObject{ .attributes = choice_fields }).toRuntime(arena);
    var chunk_fields: runtime.ObjectMap = .empty;
    try chunk_fields.put(arena, "choices", try python.tuple(arena, &.{choice}));
    const chunk = try (Citations.StreamObject{ .attributes = chunk_fields, .kind = .engine_chunk }).toRuntime(arena);
    const parsed = (try Citations.parseStreamChunk(arena, chunk)).?;
    try std.testing.expectEqualStrings("T", parsed.citations[0].cited_text);
    try std.testing.expectEqualStrings("Title", parsed.citations[0].document_title.?);
    const fallback = try Citations.fromParts(arena, &.{.{ .url = "url", .title = "title" }});
    try std.testing.expectEqualStrings("title", fallback.citations[0].cited_text);
    try std.testing.expect((try Citations.parseLmResponse(arena, .{ .text = null })) == null);
    try std.testing.expectEqualStrings("url", (try Citations.parseLmResponse(arena, .{ .text = null, .citations = &.{.{ .url = "url" }} })).?.citations[0].cited_text);
}

test "citation owning clone and tuple iteration unwind all allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(allocator: Allocator) !void {
            const original = try (Citations{ .citations = &.{
                .{ .cited_text = "borrowed 雪", .document_title = "Title", .document_index = .{ .value = .{ .number_string = "18446744073709551616" } } },
                .{ .cited_text = "second", .url = "https://example.invalid" },
            } }).toRuntime(std.testing.allocator);
            defer original.custom.deinit(std.testing.allocator);
            const copy = try original.custom.clone(allocator);
            defer copy.deinit(allocator);
            try std.testing.expectEqualStrings("borrowed 雪", copy.get(Citations).?.citations[0].cited_text);
        }
    }.run, .{});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(allocator: Allocator) !void {
            const original = try (Citations.Citation{ .cited_text = "borrowed 雪" }).toRuntime(std.testing.allocator);
            defer original.custom.deinit(std.testing.allocator);
            const items = try original.custom.vtable.iterate.?(allocator, original.custom.ptr);
            defer allocator.free(items);
            defer for (items) |item| item.custom.deinit(allocator);
            try std.testing.expectEqual(@as(usize, 8), items.len);
            const field = python.view(items[1]).?.tuple;
            try std.testing.expectEqualStrings("cited_text", field[0].string);
            try std.testing.expectEqualStrings("borrowed 雪", field[1].string);
        }
    }.run, .{});
}
