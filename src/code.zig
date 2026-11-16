//! DSPy 3.5 Code retains its concrete class while serializing as a raw string.
//! Typed values and runtime boxes borrow content; owning runtime clones copy it.
//! A borrowed base projection also requires the original runtime box to stay alive.

const std = @import("std");
const runtime = @import("runtime_value.zig");
const Value = runtime.Value;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const pyjson = @import("pyjson.zig");
const unicode = @import("unicode.zig");
const strip = @import("xml_unicode.zig").strip;

pub const Error = error{ParseValue};

/// The unparameterized DSPy Code class, whose language is Python.
pub const Code = CodeType("python", true, void);

/// A language-specialized Code type. Different languages are distinct types;
/// repeated instantiations with the same language use Zig's canonical type.
pub fn CodeWithLanguage(comptime language: []const u8) type {
    return CodeType(language, false, void);
}

/// Give separate class declarations distinct identities even when their
/// language and model name match. Reuse the same tag to refer to that class.
pub fn CodeWithLanguageIdentity(comptime language: []const u8, comptime Identity: type) type {
    return CodeType(language, false, Identity);
}

pub fn isCode(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "dszig_code_type");
}

/// DSPy discovers custom types through annotation arguments, not model fields.
pub fn annotationType(comptime T: type) ?type {
    if (isCode(T)) return T;
    if (@typeInfo(T) == .@"struct" and @hasDecl(T, "dszig_dict_value")) return annotationType(T.dszig_dict_value);
    return switch (@typeInfo(T)) {
        .optional => |o| annotationType(o.child),
        .array => |a| if (a.len == 0) null else annotationType(a.child),
        .pointer => |p| if (p.size == .slice and T != []const u8 and T != []u8) annotationType(p.child) else null,
        else => null,
    };
}

fn CodeType(comptime language_name: []const u8, comptime is_default: bool, comptime Identity: type) type {
    if (!std.unicode.utf8ValidateSlice(language_name)) @compileError("dszig: Code language must be valid UTF-8");
    return struct {
        const Self = @This();
        pub const dszig_code_type = true;
        pub const class_identity = Identity;
        pub const language = language_name;
        pub const model_name = if (is_default) "Code" else "Code_" ++ language_name;
        pub const schema_name = definitionName(model_name);
        pub const description = "Code represented in a string, specified in the `code` field. If this is an output field, the code " ++
            "field should follow the markdown code block format, e.g. \n```" ++ unicode.lowerComptime(language_name) ++ "\n{code}\n```" ++
            "\nProgramming language: " ++ language_name;
        pub const schema_description_json = if (is_default) default_schema_description_json else "";

        code: []const u8,
        /// A base-typed view can borrow a specialized instance without losing
        /// its concrete class. Only toRuntime carries this identity onward;
        /// toValue remains the public plain-string dump.
        runtime_origin: ?runtime.Custom = null,

        /// Validate text, borrowing either the entire original or its code body.
        pub fn init(text: []const u8) Self {
            return .{ .code = filterCode(text) };
        }

        /// Existing instances are already validated and retain their exact slice.
        /// Base Code also preserves the concrete type of language-specialized
        /// instances. Strings and JSON objects produce this annotation's type.
        pub fn validateInput(input: anytype) Error!(if (is_default and isCode(@TypeOf(input))) @TypeOf(input) else Self) {
            const T = @TypeOf(input);
            if (comptime T == Self or (is_default and isCode(T))) return input;
            if (comptime isCode(T)) {
                if (input.runtime_origin) |origin| return fromValue(.{ .custom = origin });
                return error.ParseValue;
            }
            if (T == Value) return fromValue(input);
            if (T == []const u8 or T == []u8) return init(input);
            if (@typeInfo(T) == .pointer) {
                const p = @typeInfo(T).pointer;
                if (p.size == .one and @typeInfo(p.child) == .array and @typeInfo(p.child).array.child == u8) return init(input);
            }
            return error.ParseValue;
        }

        pub fn fromValue(input: Value) Error!Self {
            if (input == .custom) {
                if (input.custom.get(Self)) |actual| return actual.*;
                if (comptime is_default) {
                    const metadata = input.custom.vtable.code orelse return error.ParseValue;
                    return .{ .code = metadata.content(input.custom.ptr), .runtime_origin = input.custom };
                }
                return error.ParseValue;
            }
            const content = if (input == .object) input.object.get("code") orelse return error.ParseValue else input;
            if (content != .string) return error.ParseValue;
            return init(content.string);
        }

        /// Existing instances bypass filtering and retain their exact box.
        pub fn validate(arena: Allocator, input: Value) runtime.CustomError!Value {
            const checked = try fromValue(input);
            return if (input == .custom) input else checked.toRuntime(arena);
        }

        pub fn format(self: Self) []const u8 {
            return self.code;
        }

        /// No wrapper object or custom-type markers, including nested records.
        pub fn toValue(self: Self) Value {
            return .{ .string = self.code };
        }

        /// Unlike other Type subclasses, Code overrides the serializer itself:
        /// ordinary and round-trip dumps are both strings, never {code: ...}.
        pub fn jsonStringify(self: Self, writer: *std.json.Stringify) Writer.Error!void {
            try writer.write(self.code);
        }

        pub fn instanceLanguage(self: Self) []const u8 {
            return if (self.runtime_origin) |origin| origin.vtable.code.?.language else language;
        }

        /// Box the actual class without owning its borrowed content. A base
        /// projection returns its original specialized box, not a new Code.
        pub fn toRuntime(self: Self, arena: Allocator) Allocator.Error!Value {
            if (self.runtime_origin) |origin| return .{ .custom = origin };
            const boxed = try arena.create(Self);
            boxed.* = self;
            return .{ .custom = .{ .ptr = boxed, .vtable = &runtime_vtable } };
        }

        /// Borrow a stable typed slot without erasing a projected specialization.
        pub fn borrowedRuntime(self: *const Self) Value {
            if (self.runtime_origin) |origin| return .{ .custom = origin.borrowed() };
            return .{ .custom = .{
                .ptr = self,
                .vtable = &runtime_vtable,
                .owns_box = false,
            } };
        }

        pub fn writeStr(self: Self, writer: *Writer) runtime.CustomWriteError!void {
            if (self.runtime_origin) |origin| return origin.writeStr(writer);
            try writer.writeAll("code=");
            try pyjson.writeReprString(writer, self.code);
        }

        pub fn writeRepr(self: Self, writer: *Writer) runtime.CustomWriteError!void {
            if (self.runtime_origin) |origin| return origin.writeRepr(writer);
            try writer.writeAll(model_name ++ "(");
            try self.writeStr(writer);
            try writer.writeByte(')');
        }

        const runtime_vtable: runtime.CustomVTable = .{
            .type_name = @typeName(Self),
            .type_identity = runtime.typeIdentity(Self),
            .model_name = model_name,
            .is_type = true,
            .hashable = false,
            .code = .{ .language = language, .content = runtimeContent },
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

        fn runtimeCode(ptr: *const anyopaque) *const Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn runtimeContent(ptr: *const anyopaque) []const u8 {
            return runtimeCode(ptr).code;
        }

        fn rawRuntimeValue(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return runtimeCode(ptr).toValue();
        }

        fn serializedRuntimeValue(_: Allocator, ptr: *const anyopaque, _: u32) runtime.CustomError!Value {
            return runtimeCode(ptr).toValue();
        }

        fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            try runtimeCode(ptr).writeStr(writer);
        }

        fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, _: u32) runtime.CustomWriteError!void {
            try runtimeCode(ptr).writeRepr(writer);
        }

        fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
            try writer.write(runtimeCode(ptr).code);
        }

        fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, _: usize) runtime.CustomWriteError!void {
            try pyjson.writeJsonString(writer, runtimeCode(ptr).code);
        }

        fn cloneRuntime(allocator: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
            const copy = try allocator.create(Self);
            errdefer allocator.destroy(copy);
            // Runtime projections are never boxed as a different class. Clone
            // the concrete instance and reset any borrowed projection origin.
            copy.* = .{ .code = try allocator.dupe(u8, runtimeCode(ptr).code) };
            return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
        }

        fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
            const code = runtimeCode(ptr);
            if (owns_data) allocator.free(code.code);
            allocator.destroy(code);
        }

        fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
            // Custom.eql checks concrete class identity before field equality.
            return std.mem.eql(u8, runtimeCode(left).code, runtimeCode(right).code);
        }
    };
}

/// The multiline search takes precedence over every inline fence. Its greedy
/// header may itself include earlier closing/opening fences before the LF.
fn filterCode(text: []const u8) []const u8 {
    const opening = std.mem.indexOf(u8, text, "```") orelse return text;
    const body = opening + 3;
    if (std.mem.indexOfScalarPos(u8, text, body, '\n')) |newline| {
        if (std.mem.indexOfPos(u8, text, newline + 1, "```")) |closing| return strip(text[newline + 1 .. closing]);
    }
    if (std.mem.indexOfPos(u8, text, body, "```")) |closing| return strip(text[body..closing]);
    return text;
}

/// Pydantic's definition-key normalization, applied to Unicode code points.
fn definitionName(comptime text: []const u8) []const u8 {
    comptime {
        var out: []const u8 = "";
        var iterator = (std.unicode.Utf8View.init(text) catch unreachable).iterator();
        while (iterator.nextCodepoint()) |cp| {
            if (cp == '.') {
                out = out ++ "__";
            } else if (cp <= 0x7f and (std.ascii.isAlphanumeric(@intCast(cp)) or cp == '_' or cp == '-')) {
                out = out ++ &[_]u8{@intCast(cp)};
            } else out = out ++ "_";
        }
        return out;
    }
}

// DSPy Code's class docstring is present only on the default model schema;
// create_model-generated language subclasses have no own schema description.
const default_schema_description_json = "\"Code type in DSPy.\\n\\nThis type is useful for code generation and code analysis.\\n\\nExample 1: dspy.Code as output type in code generation:\\n\\n```python\\nimport dspy\\n\\ndspy.configure(lm=dspy.LM(\\\"openai/gpt-4o-mini\\\"))\\n\\n\\nclass CodeGeneration(dspy.Signature):\\n    '''Generate python code to answer the question.'''\\n\\n    question: str = dspy.InputField(description=\\\"The question to answer\\\")\\n    code: dspy.Code[\\\"java\\\"] = dspy.OutputField(description=\\\"The code to execute\\\")\\n\\n\\npredict = dspy.Predict(CodeGeneration)\\n\\nresult = predict(question=\\\"Given an array, find if any of the two numbers sum up to 10\\\")\\nprint(result.code)\\n```\\n\\nExample 2: dspy.Code as input type in code analysis:\\n\\n```python\\nimport dspy\\nimport inspect\\n\\ndspy.configure(lm=dspy.LM(\\\"openai/gpt-4o-mini\\\"))\\n\\nclass CodeAnalysis(dspy.Signature):\\n    '''Analyze the time complexity of the function.'''\\n\\n    code: dspy.Code[\\\"python\\\"] = dspy.InputField(description=\\\"The function to analyze\\\")\\n    result: str = dspy.OutputField(description=\\\"The time complexity of the function\\\")\\n\\n\\npredict = dspy.Predict(CodeAnalysis)\\n\\n\\ndef sleepsort(x):\\n    import time\\n\\n    for i in x:\\n        time.sleep(i)\\n        print(i)\\n\\nresult = predict(code=inspect.getsource(sleepsort))\\nprint(result.result)\\n```\"";

const testing = std.testing;

/// Adapter results and traces retain the actual class, not a string lookalike.
fn expectCode(comptime T: type, expected: []const u8, actual: Value) !void {
    try testing.expect(actual == .custom);
    try testing.expectEqualStrings(expected, (actual.custom.get(T) orelse return error.TestUnexpectedResult).code);
}

test "Code typed nested values validate and round trip as raw strings" {
    const types = @import("types.zig");
    const value = @import("value.zig");
    const Java = CodeWithLanguage("JAVA");
    const Bundle = struct {
        pub const model_name = "Bundle";
        primary: Java,
        optional: ?Code,
        snippets: []const Code,
        files: types.Dict(Java),
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const input = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"primary":"```java\nrun();\n```","optional":null,"snippets":["```x```",{"code":" ```y``` "}],"files":{"main":{"code":"```java\nmain();\n```","extra":true}}}
    , .{});
    const checked = try types.validate(Bundle, arena, input);
    const decoded = try value.fromValue(arena, Bundle, checked);
    try testing.expectEqualStrings("run();", decoded.primary.code);
    try testing.expect(decoded.optional == null);
    try testing.expectEqualStrings("x", decoded.snippets[0].code);
    try testing.expectEqualStrings("y", decoded.snippets[1].code);
    try testing.expectEqualStrings("main();", decoded.files.get("main").?.code);
    const encoded = try value.toValue(arena, Bundle, decoded);
    const expected = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"primary":"run();","optional":null,"snippets":["x","y"],"files":{"main":"main();"}}
    , .{});
    try testing.expect(try value.pyEqual(expected, try value.serializedValue(arena, encoded)));
    const cloned = try value.cloneValue(arena, encoded);
    try testing.expectEqualStrings("main();", (try value.fromValue(arena, Bundle, cloned)).files.get("main").?.code);
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { bundle: Bundle };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const output = try value.toRecord(arena, Sig.Output, .{ .bundle = decoded });
    inline for (.{ adapter.Adapter.chat, adapter.Adapter.json, adapter.Adapter.xml }) |which| {
        const text = try adapter.formatAssistant(which, arena, view, &output);
        const parsed = try which.parse(arena, view, text);
        const restored = try value.fromRecord(arena, Sig.Output, &parsed);
        try testing.expectEqualStrings("run();", restored.bundle.primary.code);
        try testing.expect(restored.bundle.optional == null);
        try testing.expectEqualStrings("y", restored.bundle.snippets[1].code);
        try testing.expectEqualStrings("main();", restored.bundle.files.get("main").?.code);
    }
    const provider = try adapter.responseFormat(arena, view, true);
    const model = provider.json_schema.schema.object.get("$defs").?.object.get("Bundle").?.object;
    try testing.expect(!model.get("properties").?.object.get("optional").?.object.contains("title"));
    try testing.expectEqualStrings("#/$defs/Code_JAVA", model.get("properties").?.object.get("primary").?.object.get("$ref").?.string);
    for ([_][]const u8{
        \\{"primary":"ok","optional":7,"snippets":[],"files":{}}
        ,
        \\{"primary":"ok","optional":null,"snippets":[{"code":7}],"files":{}}
        ,
        \\{"primary":"ok","optional":null,"snippets":[],"files":{"bad":{}}}
        ,
    }) |text| {
        const invalid = try std.json.parseFromSliceLeaky(Value, arena, text, .{});
        try testing.expectError(error.ParseValue, types.validate(Bundle, arena, invalid));
        try testing.expectError(error.TypeMismatch, value.fromValue(arena, Bundle, invalid));
    }
}

test "Code JSON chat and XML adapters validate actual custom values and format raw strings" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const value = @import("value.zig");
    const Java = CodeWithLanguage("JAVA");
    const Sig = signature.Signature(struct {
        pub const Input = struct { source: Code };
        pub const Output = struct { result: Java };
    });
    const view: adapter.View = .{ .instructions = Sig.instructions, .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const inputs = try value.toRecord(arena, Sig.Input, .{ .source = Code.init("```python\nprint(1)\n```") });
    const outputs = try value.toRecord(arena, Sig.Output, .{ .result = Java.init("run();") });
    const cases = .{
        .{ adapter.Adapter.chat, "[[ ## result ## ]]\n```java\nrun();\n```\n[[ ## completed ## ]]", "[[ ## source ## ]]\nprint(1)", "[[ ## result ## ]]\nrun();\n\n[[ ## completed ## ]]\n" },
        .{ adapter.Adapter.json, "{\"result\":{\"code\":\"```java\\nrun();\\n```\",\"extra\":4}}", "[[ ## source ## ]]\nprint(1)", "{\n  \"result\": \"run();\"\n}" },
        .{ adapter.Adapter.xml, "<result><code>```java\nrun();\n```</code></result>", "<source>\nprint(1)\n</source>", "<result>\nrun();\n</result>" },
    };
    inline for (cases) |case| {
        const parsed = try case[0].parse(arena, view, case[1]);
        try expectCode(Java, "run();", parsed.get("result").?);
        try testing.expectEqualStrings("run();", (try value.fromRecord(arena, Sig.Output, &parsed)).result.code);
        try testing.expectEqualStrings(case[2], try adapter.formatUser(case[0], arena, view, &inputs, false));
        try testing.expectEqualStrings(case[3], try adapter.formatAssistant(case[0], arena, view, &outputs));
    }
    for ([_][]const u8{
        "{\"result\":{\"code\":7}}",
        "{\"result\":7}",
        "{\"result\":null}",
        "{\"result\":{}}",
        "{}",
    }) |completion| try testing.expectError(error.AdapterParse, adapter.Adapter.json.parse(arena, view, completion));
    for ([_][]const u8{
        "<result />",
        "<result></result>",
        "<result><wrong>x</wrong></result>",
        "<result><code>run();</code><extra>4</extra></result>",
    }) |completion| try testing.expectError(error.AdapterParse, adapter.Adapter.xml.parse(arena, view, completion));
    const literal = try adapter.Adapter.xml.parse(arena, view, "<result>if (a < b) x = '<tag>';</result>");
    try expectCode(Java, "if (a < b) x = '<tag>';", literal.get("result").?);
    // A direct custom Type retries the original text after candidate validation;
    // an object provided by JSONAdapter has no such string fallback.
    const fallback = try adapter.Adapter.chat.parse(arena, view, "[[ ## result ## ]]\n{\"code\":7}");
    try expectCode(Java, "{\"code\":7}", fallback.get("result").?);
    const parsed_object_text = try adapter.Adapter.chat.parse(arena, view, "[[ ## result ## ]]\n{\"code\":\"inside\"}");
    try expectCode(Java, "inside", parsed_object_text.get("result").?);
    const fenced_json = try adapter.Adapter.json.parse(arena, view, "{\"result\":\"```json\\n{\\\"code\\\":\\\"inside\\\"}\\n```\"}");
    try expectCode(Java, "inside", fenced_json.get("result").?);
}

test "Code optional and collection adapters keep null empty and invalid values distinct" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const value = @import("value.zig");
    const Optional = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { result: ?Code };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Optional.input_fields_list, .outputs = Optional.output_fields_list };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    inline for (.{
        .{ adapter.Adapter.chat, "[[ ## result ## ]]\nnull", "[[ ## result ## ]]\n", "" },
        .{ adapter.Adapter.json, "{\"result\":null}", "{\"result\":\"\"}", "{}" },
        .{ adapter.Adapter.xml, "<result />", "<result></result>", "" },
    }) |case| {
        const nulls = try case[0].parse(arena, view, case[1]);
        try testing.expect(nulls.get("result").? == .null);
        const empty = try case[0].parse(arena, view, case[2]);
        try testing.expectEqualStrings("", (try value.fromRecord(arena, Optional.Output, &empty)).result.?.code);
        const omitted = try case[0].parse(arena, view, case[3]);
        try testing.expect(omitted.get("result").? == .null);
    }
    try testing.expectError(error.AdapterParse, adapter.Adapter.json.parse(arena, view, "{\"result\":7}"));
    try testing.expectError(error.AdapterParse, adapter.Adapter.chat.parse(arena, view, "[[ ## result ## ]]\n{\"code\":7}"));
    const Collection = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { snippets: []const Code };
    });
    const list_view: adapter.View = .{ .instructions = "", .inputs = Collection.input_fields_list, .outputs = Collection.output_fields_list };
    const collection = try adapter.Adapter.xml.parse(arena, list_view, "<snippets>[\"```x```\", {\"code\":\"```y```\"}]</snippets>");
    const typed = try value.fromRecord(arena, Collection.Output, &collection);
    try testing.expectEqualStrings("x", typed.snippets[0].code);
    try testing.expectEqualStrings("y", typed.snippets[1].code);
    try testing.expectEqualStrings("<snippets>\n[\"x\", \"y\"]\n</snippets>", try adapter.formatAssistant(.xml, arena, list_view, &collection));
    try testing.expectError(error.AdapterParse, adapter.Adapter.xml.parse(arena, list_view, "<snippets>[{\"code\":7}]</snippets>"));
}

test "Code XML literal selection stops at the immediate list item" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const types = @import("types.zig");
    const value = @import("value.zig");
    const Snippet = CodeWithLanguage("JAVA");
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    inline for (.{
        .{ Snippet, "\"run();\"", "<result>\nrun();\n</result>" },
        .{ ?Snippet, "\"run();\"", "<result>\nrun();\n</result>" },
        .{ []const Snippet, "[\"run();\"]", "<result>\n[\"run();\"]\n</result>" },
        .{ ?[]const Snippet, "[\"run();\"]", "<result>\n[\"run();\"]\n</result>" },
        .{ types.Dict(Snippet), "{\"main\":\"run();\",\"a b\":\"run();\"}", "<result>\n<main>\nrun();\n</main>\n<entry key=\"a b\">\nrun();\n</entry>\n</result>" },
        .{ []const []const Snippet, "[[\"run();\"]]", "<result>\n<item>\n<item>\nrun();\n</item>\n</item>\n</result>" },
        .{ []const ?Snippet, "[\"run();\",null]", "<result>\n<item>\nrun();\n</item>\n<item />\n</result>" },
        .{ []const types.Dict(Snippet), "[{\"main\":\"run();\"}]", "<result>\n<item>\n<main>\nrun();\n</main>\n</item>\n</result>" },
        .{ ?types.Dict(Snippet), "{\"main\":\"run();\"}", "<result>\n<main>\nrun();\n</main>\n</result>" },
        .{ []const Snippet, "[]", "<result>\n[]\n</result>" },
        .{ ?[]const Snippet, "null", "<result />" },
        .{ types.Dict(Snippet), "{}", "<result>\n\n</result>" },
    }) |case| {
        const Sig = signature.Signature(struct {
            pub const Input = struct {};
            pub const Output = struct { result: case[0] };
        });
        const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
        const expected = try std.json.parseFromSliceLeaky(Value, arena, case[1], .{});
        const typed = try value.fromValue(arena, case[0], expected);
        const outputs = try value.toRecord(arena, Sig.Output, .{ .result = typed });
        const text = try adapter.formatAssistant(.xml, arena, view, &outputs);
        try testing.expectEqualStrings(case[2], text);
        const parsed = try adapter.Adapter.xml.parse(arena, view, text);
        try testing.expect(try value.pyEqual(expected, try value.serializedValue(arena, parsed.get("result").?)));
        const restored = try value.fromRecord(arena, Sig.Output, &parsed);
        try testing.expect(try value.pyEqual(expected, try value.serializedValue(arena, try value.toValue(arena, case[0], restored.result))));
    }
    inline for (.{
        .{
            types.Dict(Snippet),
            "{\"main\":\"run();\",\"a b\":\"run();\"}",
            "<source>\n{\"main\": \"run();\", \"a b\": \"run();\"}\n</source>",
            "<result>\n<entry key=\"example key\">\n<code>\n...\n</code>\n</entry>\n</result>",
        },
        .{
            []const []const Snippet,
            "[[\"run();\"]]",
            "<source>\n[[\"run();\"]]\n</source>",
            "<result>\n<item>\n<item>\n<code>\n...\n</code>\n</item>\n</item>\n</result>",
        },
        .{
            []const ?Snippet,
            "[\"run();\",null]",
            "<source>\n[\"run();\", null]\n</source>",
            "<result>\n<item>\n<code>\n...\n</code>\n</item>\n</result>",
        },
    }) |case| {
        const Sig = signature.Signature(struct {
            pub const Input = struct { source: case[0] };
            pub const Output = struct { result: case[0] };
        });
        const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
        const input = try std.json.parseFromSliceLeaky(Value, arena, case[1], .{});
        const typed = try value.fromValue(arena, case[0], input);
        const inputs = try value.toRecord(arena, Sig.Input, .{ .source = typed });
        try testing.expectEqualStrings(case[2], try adapter.formatUser(.xml, arena, view, &inputs, false));
        try testing.expectEqualStrings(
            "Output format (replace placeholders):\n" ++ case[3],
            try adapter.XmlAdapter.fieldStructure(arena, Sig.output_fields_list),
        );
    }
}

test "Code provider definition collisions fail closed and equivalent models share" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const Cpp = CodeWithLanguage("C++");
    const Other = struct {
        pub const model_name = "Code_C__";
        count: i64,
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    inline for (.{
        .{ Cpp, Other },
        .{ Other, Cpp },
        .{ []const Cpp, []const Other },
        .{ []const Other, []const Cpp },
        .{ []const Cpp, Other },
        .{ Other, []const Cpp },
        .{ Cpp, []const Other },
        .{ []const Other, Cpp },
        .{ Cpp, CodeWithLanguage("C##") },
        .{ CodeWithLanguage("C##"), Cpp },
        .{ []const Cpp, []const CodeWithLanguage("C##") },
        .{ []const CodeWithLanguage("C##"), []const Cpp },
    }) |case| {
        const Sig = signature.Signature(struct {
            pub const Input = struct {};
            pub const Output = struct { code: case[0], model: case[1] };
        });
        const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
        try testing.expectError(error.AdapterParse, adapter.responseFormat(arena, view, true));
        try testing.expect((try adapter.responseFormat(arena, view, false)) == .json_object);
    }
    const Shared = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { first: Cpp, second: Cpp, snippets: []const Cpp, optional: ?Cpp };
    });
    const response = try adapter.responseFormat(arena, .{ .instructions = "", .inputs = Shared.input_fields_list, .outputs = Shared.output_fields_list }, true);
    const root = response.json_schema.schema.object;
    try testing.expectEqual(@as(usize, 1), root.get("$defs").?.object.count());
    const properties = root.get("properties").?.object;
    try testing.expectEqualStrings("#/$defs/Code_C__", properties.get("first").?.object.get("$ref").?.string);
    try testing.expectEqualStrings("#/$defs/Code_C__", properties.get("second").?.object.get("$ref").?.string);
    try testing.expectEqualStrings("#/$defs/Code_C__", properties.get("snippets").?.object.get("items").?.object.get("$ref").?.string);
    try testing.expectEqualStrings("#/$defs/Code_C__", properties.get("optional").?.object.get("anyOf").?.array.items[0].object.get("$ref").?.string);
}

test "Code textual prompt schemas preserve Unicode and required JSON escaping" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    inline for (.{
        .{ "Café", "Code_Caf_", "Code_Café" },
        .{ "Cafe\u{0301}", "Code_Cafe_", "Code_Cafe\u{0301}" },
        .{ "Café\x08\x0c\n\r\t\x1f", "Code_Caf_______", "Code_Café\\b\\f\\n\\r\\t\\u001f" },
        .{ "q\"uote\\\n\x01", "Code_q_uote___", "Code_q\\\"uote\\\\\\n\\u0001" },
        .{ "é😀\x7f\u{2028}", "Code_____", "Code_é😀\x7f\u{2028}" },
    }) |case| {
        const Sig = signature.Signature(struct {
            pub const Input = struct {};
            pub const Output = struct { result: []const CodeWithLanguage(case[0]) };
        });
        var text: std.Io.Writer.Allocating = .init(arena);
        try adapter.translateFieldType(&text.writer, Sig.output_fields_list[0], false);
        try testing.expectEqualStrings(
            "{result}        # note: the value you produce must adhere to the JSON schema: " ++
                "{\"type\": \"array\", \"$defs\": {\"" ++ case[1] ++ "\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"title\": \"Code\"}}, \"required\": [\"code\"], \"title\": \"" ++ case[2] ++ "\"}}, \"items\": {\"$ref\": \"#/$defs/" ++ case[1] ++ "\"}}",
            text.written(),
        );
    }
}

test "Code provider schemas preserve model validation despite raw-string serialization" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const types = @import("types.zig");
    const Java = CodeWithLanguage("JAVA");
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { result: Java, optional: ?Java, snippets: []const Code };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const response = try adapter.responseFormat(arena, view, true);
    try testing.expect(response == .json_schema);
    const root = response.json_schema.schema.object;
    const properties = root.get("properties").?.object;
    try testing.expectEqualStrings("#/$defs/Code_JAVA", properties.get("result").?.object.get("$ref").?.string);
    try testing.expectEqualStrings("#/$defs/Code_JAVA", properties.get("optional").?.object.get("anyOf").?.array.items[0].object.get("$ref").?.string);
    try testing.expect(!properties.get("optional").?.object.contains("title"));
    try testing.expectEqualStrings("#/$defs/Code", properties.get("snippets").?.object.get("items").?.object.get("$ref").?.string);
    const definitions = root.get("$defs").?.object;
    const java = definitions.get("Code_JAVA").?.object;
    try testing.expectEqualStrings("object", java.get("type").?.string);
    try testing.expectEqualStrings("string", java.get("properties").?.object.get("code").?.object.get("type").?.string);
    try testing.expectEqualStrings("code", java.get("required").?.array.items[0].string);
    try testing.expect(!java.get("additionalProperties").?.bool);
    try testing.expect(!java.contains("description"));
    try testing.expect(definitions.get("Code").?.object.contains("description"));
    const Cpp = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { result: CodeWithLanguage("C++") };
    });
    const cpp = try adapter.responseFormat(arena, .{ .instructions = "", .inputs = Cpp.input_fields_list, .outputs = Cpp.output_fields_list }, true);
    try testing.expectEqualStrings("#/$defs/Code_C__", cpp.json_schema.schema.object.get("properties").?.object.get("result").?.object.get("$ref").?.string);
    try testing.expectEqualStrings("Code_C++", cpp.json_schema.schema.object.get("$defs").?.object.get("Code_C__").?.object.get("title").?.string);
    const quoted = try std.json.parseFromSliceLeaky(Value, arena, types.schema(?CodeWithLanguage("q\"uote")), .{});
    try testing.expectEqualStrings("Code_q\"uote", quoted.object.get("$defs").?.object.get("Code_q_uote").?.object.get("title").?.string);
    const unicode_schema = try std.json.parseFromSliceLeaky(Value, arena, types.schema(?CodeWithLanguage("İΣ")), .{});
    try testing.expectEqualStrings("Code_İΣ", unicode_schema.object.get("$defs").?.object.get("Code___").?.object.get("title").?.string);
    const Described = signature.Signature(struct {
        pub const Input = struct { sources: []const types.Dict(Java) };
        pub const Output = struct { result: Java };
    });
    const description = try adapter.fieldDescriptionString(arena, Described.input_fields_list);
    try testing.expectEqualStrings(
        "1. `sources` (list[dict[str, Code_JAVA]]): \n    Type description of Code_JAVA: Code represented in a string, specified in the `code` field. If this is an output field, the code field should follow the markdown code block format, e.g. \n```java\n{code}\n```\nProgramming language: JAVA",
        description,
    );
}

test "Code typed Predict traces and saved demos remain consumer-visible raw strings" {
    const adapter = @import("adapter.zig");
    const signature = @import("signature.zig");
    const predict = @import("predict.zig");
    const lm = @import("lm.zig");
    const state = @import("state.zig");
    const value = @import("value.zig");
    const Java = CodeWithLanguage("JAVA");
    const Sig = signature.Signature(struct {
        pub const Input = struct { source: Code };
        pub const Output = struct { result: Java };
    });
    const ConsumerLM = struct {
        response: []const u8,

        fn complete(ptr: *anyopaque, arena: std.mem.Allocator, request: lm.Request) lm.Error![]const lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const prompt = request.messages[request.messages.len - 1].content.text;
            if (std.mem.indexOf(u8, prompt, "print(1)") == null or
                std.mem.indexOf(u8, prompt, "\"code\":") != null or
                std.mem.indexOf(u8, prompt, "<<CUSTOM-TYPE") != null) return error.LMFailure;
            const out = try arena.alloc(lm.Completion, 1);
            out[0] = .{ .text = self.response };
            return out;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    inline for (.{
        .{ adapter.Adapter.chat, "[[ ## result ## ]]\n```java\nrun();\n```" },
        .{ adapter.Adapter.json, "{\"result\":{\"code\":\"```java\\nrun();\\n```\"}}" },
        .{ adapter.Adapter.xml, "<result><code>```java\nrun();\n```</code></result>" },
    }) |case| {
        var model: ConsumerLM = .{ .response = case[1] };
        const program: predict.Predict(Sig) = .{};
        var trace: predict.Trace = .{ .arena = arena };
        const result = try program.forward(.{
            .arena = arena,
            .lm = .{ .ptr = &model, .vtable = &.{ .complete = ConsumerLM.complete } },
            .adapter = case[0],
            .json_fallback = false,
            .trace = &trace,
        }, .{ .source = Code.init("```python\nprint(1)\n```") });
        try testing.expectEqualStrings("run();", result.result.code);
        try expectCode(Code, "print(1)", trace.steps.items[0].inputs.get("source").?);
        try expectCode(Java, "run();", trace.steps.items[0].outputs.get("result").?);
    }
    var program: predict.Predict(Sig) = .{};
    const Demo = struct { source: Code, result: Java };
    const demo = try value.toRecord(arena, Demo, .{ .source = Code.init("print(1)"), .result = Java.init("run();") });
    program.state.demos = &.{demo};
    const json = try state.saveAlloc(arena, &program);
    var restored: predict.Predict(Sig) = .{};
    try state.load(arena, &restored, json);
    const round_trip = try value.fromRecord(arena, Demo, &restored.state.demos[0]);
    try testing.expectEqualStrings("print(1)", round_trip.source.code);
    try testing.expectEqualStrings("run();", round_trip.result.code);
    const restored_view = try restored.state.view(arena);
    try testing.expectError(error.AdapterParse, adapter.Adapter.json.parse(arena, restored_view, "{\"result\":{\"code\":7}}"));
}

test "Code extracts the first multiline regex match before any inline fence" {
    const cases = [_][2][]const u8{
        .{ "prefix ```python\n  print(1)\n``` suffix", "print(1)" },
        .{ "prefix ``` print(1) ``` suffix", "print(1)" },
        .{ "```first``` gap ```second```", "first" },
        .{ "```inline```\nprose\n```python\nsecond\n```", "prose" },
        .{ "```inline``` gap ```python\nsecond\n```", "second" },
        .{ "```first```\nsecond```", "second" },
        .{ "````python\nbody\n````", "body" },
        .{ "``````", "" },
        .{ "```python\n```", "" },
        .{ "```\n  body\n```", "body" },
        .{ "```python\r body\r```", "python\r body" },
        .{ "before ```Py\r\n  first\r\n  second\r\n``` after", "first\r\n  second" },
        .{ "```language\u{2028} body```", "language\u{2028} body" },
        .{ "```python\n\u{85}\u{a0}\u{1680}\u{2000}\u{2003}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}body\u{3000}\x1c\n```", "body" },
        .{ "```\n\u{200b}body\u{feff}\n```", "\u{200b}body\u{feff}" },
    };
    for (cases) |case| try testing.expectEqualStrings(case[1], Code.init(case[0]).code);
}

test "Code preserves unfenced and unclosed source and valid instance identity" {
    const inputs = [_][]const u8{ " \tprint(1)\r\n\u{2003}", " \t```python\n body\n\u{2003}", "before ```", "", "{\"code\":\"inside\"}" };
    for (inputs) |input| {
        const value = Code.init(input);
        try testing.expectEqualStrings(input, value.code);
        try testing.expectEqual(input.ptr, value.code.ptr);
    }
    const constructed: Code = .{ .code = "```valid instance remains unchanged```" };
    const valid = try Code.validateInput(constructed);
    try testing.expectEqual(constructed.code.ptr, valid.code.ptr);
    try testing.expectEqualStrings(constructed.code, valid.toValue().string);
    try testing.expectEqualStrings("body", (try Code.validateInput("```body```")).code);
    const java = CodeWithLanguage("java").init("run();");
    const as_base = try Code.validateInput(java);
    try testing.expectEqual(java.code.ptr, as_base.code.ptr);
    try testing.expectEqualStrings("run();", as_base.code);
    try testing.expectEqualStrings("java", @TypeOf(as_base).language);
    const as_same = try CodeWithLanguage("java").validateInput(java);
    try testing.expectEqual(java.code.ptr, as_same.code.ptr);
    try testing.expectError(error.ParseValue, CodeWithLanguage("python").validateInput(java));
    try testing.expectError(error.ParseValue, CodeWithLanguage("java").validateInput(Code.init("run();")));
    try testing.expectError(error.ParseValue, Code.validateInput(@as(u8, 7)));
}

test "Code validates object payloads and serializes raw strings without allocating" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const input = try std.json.parseFromSliceLeaky(Value, arena, "{\"code\":\"```java\\nrun();\\n```\",\"extra\":7}", .{});
    const java = try CodeWithLanguage("java").fromValue(input);
    try testing.expectEqualStrings("run();", java.format());
    try testing.expectEqualStrings("run();", java.toValue().string);
    for ([_][]const u8{ "{}", "{\"other\":\"x\"}", "{\"code\":7}", "{\"code\":null}", "{\"code\":[\"x\"]}", "null", "true", "7", "[\"x\"]" }) |text| {
        const invalid = try std.json.parseFromSliceLeaky(Value, arena, text, .{});
        try testing.expectError(error.ParseValue, Code.fromValue(invalid));
    }
}

test "separate same-language Code class identities reject sibling instances" {
    const types = @import("types.zig");
    const value = @import("value.zig");
    const JavaA = CodeWithLanguageIdentity("JAVA", struct {});
    const JavaB = CodeWithLanguageIdentity("JAVA", struct {});
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const a = try JavaA.init("x = 1").toRuntime(arena);
    const b = try JavaB.init("x = 1").toRuntime(arena);
    try testing.expect(!try value.pyEqual(a, b));
    try testing.expect(!try value.pyEqual(b, a));
    try testing.expectError(error.ParseValue, types.validate(JavaA, arena, b));
    try testing.expectError(error.ParseValue, types.validate(JavaB, arena, a));
    const as_base_a = try types.validate(Code, arena, a);
    const as_base_b = try types.validate(Code, arena, b);
    try testing.expectEqual(a.custom.ptr, as_base_a.custom.ptr);
    try testing.expectEqual(b.custom.ptr, as_base_b.custom.ptr);
    const projection = try Code.fromValue(b);
    try testing.expectEqualStrings("JAVA", projection.instanceLanguage());
    try testing.expectEqual(b.custom.ptr, (try projection.toRuntime(arena)).custom.ptr);
}

test "Code runtime boxes keep class identity apart from strings and dictionaries" {
    const value = @import("value.zig");
    const Java = CodeWithLanguage("JAVA");
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const content = "x = 'a\"b'\n";
    const base = try (Code{ .code = content }).toRuntime(arena);
    const java = try (Java{ .code = content }).toRuntime(arena);
    const plain: Value = .{ .string = content };
    const dictionary = try std.json.parseFromSliceLeaky(Value, arena, "{\"code\":\"x = 'a\\\"b'\\n\"}", .{});
    for ([_]Value{ base, java }) |actual| {
        try testing.expect(actual.custom.vtable.is_type);
        try testing.expect(!actual.custom.vtable.hashable);
        try testing.expect(actual.custom.vtable.code != null);
        try testing.expect(!try value.pyEqual(actual, plain));
        try testing.expect(!try value.pyEqual(plain, actual));
        try testing.expect(!try value.pyEqual(actual, dictionary));
        try testing.expect(!try value.pyEqual(dictionary, actual));
        // Both the ordinary and the round-trip dumps are the bare string.
        const raw = try actual.custom.rawValue(arena);
        const serialized = try actual.custom.serialize(arena);
        try testing.expectEqualStrings(content, raw.string);
        try testing.expectEqualStrings(content, serialized.string);
        try testing.expectEqualStrings("\"x = 'a\\\"b'\\n\"", try std.json.Stringify.valueAlloc(arena, actual, .{}));
        var json: Writer.Allocating = .init(arena);
        try pyjson.writeJson(&json.writer, actual, .compact);
        try testing.expectEqualStrings("\"x = 'a\\\"b'\\n\"", json.written());
        var text: Writer.Allocating = .init(arena);
        try pyjson.writeStr(&text.writer, actual);
        try testing.expectEqualStrings("code='x = \\'a\"b\\'\\n'", text.written());
    }
    var repr: Writer.Allocating = .init(arena);
    try pyjson.writeRepr(&repr.writer, base, 0);
    try testing.expectEqualStrings("Code(code='x = \\'a\"b\\'\\n')", repr.written());
    var java_repr: Writer.Allocating = .init(arena);
    try pyjson.writeRepr(&java_repr.writer, java, 0);
    try testing.expectEqualStrings("Code_JAVA(code='x = \\'a\"b\\'\\n')", java_repr.written());
    try testing.expectEqualStrings("JAVA", java.custom.vtable.code.?.language);
    try testing.expectEqualStrings("python", base.custom.vtable.code.?.language);
    try testing.expectEqualStrings(content, java.custom.vtable.code.?.content(java.custom.ptr));
    // Equal fields compare equal only within one concrete class.
    try testing.expect(try value.pyEqual(base, try (Code{ .code = content }).toRuntime(arena)));
    try testing.expect(try value.pyEqual(java, try (Java{ .code = content }).toRuntime(arena)));
    try testing.expect(!try value.pyEqual(base, java));
    try testing.expect(!try value.pyEqual(java, base));
    try testing.expect(!try value.pyEqual(java, try (CodeWithLanguage("python"){ .code = content }).toRuntime(arena)));
    try testing.expect(!try value.pyEqual(java, try (Java{ .code = "other" }).toRuntime(arena)));
    const empty = try (Code{ .code = "" }).toRuntime(arena);
    try testing.expect(empty.custom.truthy());
    try testing.expectEqualStrings("\"\"", try std.json.Stringify.valueAlloc(arena, empty, .{}));
    // The typed public dump stays a string even with std.json reflection.
    try testing.expectEqualStrings("\"run();\"", try std.json.Stringify.valueAlloc(arena, Java.init("run();"), .{}));
    try testing.expect(Java.init("run();").toValue() == .string);
}

test "Code validation reuses actual boxes and enforces specialization inheritance" {
    const types = @import("types.zig");
    const Java = CodeWithLanguage("JAVA");
    const Python = CodeWithLanguage("python");
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    // Existing instances are not filtered again, unlike strings and objects.
    const fenced = "```java\nrun();\n```";
    const base = try (Code{ .code = fenced }).toRuntime(arena);
    const java = try (Java{ .code = fenced }).toRuntime(arena);
    const python = try (Python{ .code = fenced }).toRuntime(arena);
    for ([_]Value{ base, java, python }) |actual| {
        const checked = try types.validate(Code, arena, actual);
        try testing.expectEqual(actual.custom.ptr, checked.custom.ptr);
        try testing.expectEqual(actual.custom.vtable, checked.custom.vtable);
        try testing.expectEqualStrings(fenced, (try Code.fromValue(actual)).code);
    }
    try testing.expectEqual(java.custom.ptr, (try types.validate(Java, arena, java)).custom.ptr);
    try testing.expectError(error.ParseValue, types.validate(Java, arena, base));
    try testing.expectError(error.ParseValue, types.validate(Java, arena, python));
    try testing.expectError(error.ParseValue, types.validate(Python, arena, base));
    try testing.expectError(error.ParseValue, types.validate(Python, arena, java));
    const other = try @import("reasoning.zig").Reasoning.toRuntime(.{ .content = "run();" }, arena);
    try testing.expectError(error.ParseValue, types.validate(Code, arena, other));
    // Strings and objects construct this annotation's class after filtering.
    const from_text = try types.validate(Java, arena, .{ .string = fenced });
    try expectCode(Java, "run();", from_text);
    const object = try std.json.parseFromSliceLeaky(Value, arena, "{\"code\":\"```x```\",\"extra\":1}", .{});
    try expectCode(Code, "x", try types.validate(Code, arena, object));
    try testing.expectEqual(@as(usize, 2), object.object.count());

    // A base-typed projection borrows the specialized box and restores it.
    const projected = try Code.fromValue(java);
    try testing.expectEqualStrings("JAVA", projected.instanceLanguage());
    try testing.expectEqualStrings("python", Code.init("x").instanceLanguage());
    try testing.expectEqual(java.custom.ptr, (try projected.toRuntime(arena)).custom.ptr);
    try testing.expect(projected.toValue() == .string);
    try testing.expectEqualStrings(fenced, projected.toValue().string);
    var repr: Writer.Allocating = .init(arena);
    try projected.writeRepr(&repr.writer);
    try testing.expectEqualStrings("Code_JAVA(code='```java\\nrun();\\n```')", repr.written());
    const as_java = try Java.validateInput(projected);
    try testing.expectEqual(projected.code.ptr, as_java.code.ptr);
    try testing.expectError(error.ParseValue, Python.validateInput(projected));
    try testing.expectError(error.ParseValue, Java.validateInput(Code.init("run();")));
    try testing.expectEqual(java.custom.ptr, (try Code.validateInput(projected)).runtime_origin.?.ptr);
}

test "Code owning clones copy content without a borrowed projection origin" {
    const inference = @import("inference.zig");
    const types = @import("types.zig");
    const value = @import("value.zig");
    const Java = CodeWithLanguage("JAVA");
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);
    const allocator = debug_allocator.allocator();
    var source: std.heap.ArenaAllocator = .init(testing.allocator);
    var source_live = true;
    defer if (source_live) source.deinit();
    const expected = "clone 雪\n'\"\\\x00";
    const content = try source.allocator().dupe(u8, expected);
    const java = try (Java{ .code = content }).toRuntime(source.allocator());
    const projected = try Code.fromValue(java);
    var items: runtime.Array = .init(source.allocator());
    try items.append(try projected.toRuntime(source.allocator()));
    try items.append(.{ .string = content });
    const owned = try inference.cloneJsonValue(allocator, .{ .array = items });
    defer inference.freeJsonValue(allocator, owned);
    const copy = owned.array.items[0];
    try testing.expect(copy.custom.owns_data);
    try testing.expect(copy.custom.ptr != java.custom.ptr);
    const typed = copy.custom.get(Java).?;
    try testing.expect(typed.runtime_origin == null);
    try testing.expect(typed.code.ptr != content.ptr);
    @memset(content, 'x');
    source.deinit();
    source_live = false;

    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    try testing.expectEqualStrings(expected, typed.code);
    try testing.expectEqual(copy.custom.ptr, (try types.validate(Code, arena, copy)).custom.ptr);
    const reprojected = try Code.fromValue(copy);
    try testing.expectEqualStrings("JAVA", reprojected.instanceLanguage());
    try testing.expectEqualStrings(expected, (try value.serializedValue(arena, owned)).array.items[0].string);
    try testing.expect(!try value.pyEqual(copy, owned.array.items[1]));
    const second = try inference.cloneJsonValue(allocator, copy);
    defer inference.freeJsonValue(allocator, second);
    try testing.expect(try value.pyEqual(second, copy));
}

test "Code clone allocation failures preserve the borrowed source" {
    try testing.checkAllAllocationFailures(testing.allocator, struct {
        fn clone(allocator: Allocator) !void {
            const inference = @import("inference.zig");
            const Java = CodeWithLanguage("JAVA");
            const original = try (Java{ .code = "borrowed 雪" }).toRuntime(testing.allocator);
            defer original.custom.deinit(testing.allocator);
            const copy = try inference.cloneJsonValue(allocator, original);
            defer inference.freeJsonValue(allocator, copy);
            try testing.expectEqualStrings("borrowed 雪", copy.custom.get(Java).?.code);
            try testing.expectEqualStrings("borrowed 雪", original.custom.get(Java).?.code);
        }
    }.clone, .{});
}

test "Code language descriptions lowercase Unicode only in the markdown fence" {
    const java = CodeWithLanguage("JAVA").description;
    try testing.expect(std.mem.indexOf(u8, java, "```java\n{code}\n```") != null);
    try testing.expect(std.mem.endsWith(u8, java, "Programming language: JAVA"));
    const greek = CodeWithLanguage("İΣ").description;
    try testing.expect(std.mem.indexOf(u8, greek, "```i\u{307}ς\n{code}\n```") != null);
    try testing.expect(std.mem.endsWith(u8, greek, "Programming language: İΣ"));
    try testing.expectEqualStrings("Code_C__", CodeWithLanguage("C++").schema_name);
    try testing.expectEqualStrings("Code___", CodeWithLanguage("İΣ").schema_name);
    try testing.expect(Code != CodeWithLanguage("python"));
    try testing.expect(CodeWithLanguage("java") != CodeWithLanguage("python"));
}
