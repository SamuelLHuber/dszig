//! Comptime signatures: the Zig counterpart of `dspy.Signature`.
//!
//! A signature spec is a plain type with `Input` and `Output` struct decls
//! and optional `instructions` / `descriptions` decls:
//!
//! ```zig
//! const QA = dszig.Signature(struct {
//!     pub const instructions = "Answer questions with short factoid answers.";
//!     pub const Input = struct { context: []const []const u8, question: []const u8 };
//!     pub const Output = struct { answer: []const u8 };
//!     pub const descriptions = .{ .answer = "often between 1 and 5 words" };
//! });
//! ```
//!
//! Every structural mistake DSPy reports at class-creation time or at the
//! first LM call is a `@compileError` here.

const std = @import("std");
const nested = @import("types.zig");
const History = @import("history.zig").History;
const ToolCalls = @import("tool.zig").ToolCalls;
const assert = std.debug.assert;

/// Field value kinds supported in v0.1, with their DSPy annotation.
pub const Kind = enum {
    /// `[]const u8` <-> `str`
    str,
    /// `[]const []const u8` <-> `list[str]`
    str_list,
    /// `bool` <-> `bool`
    boolean,
    /// any integer type <-> `int`
    int,
    /// `f32` / `f64` <-> `float`
    float,
    /// exhaustive enum <-> `Literal['a', 'b']`
    literal,
    /// `std.json.ObjectMap` <-> `dict[str, Any]`
    json_object,
    /// `StrDict` <-> `dict[str, str]`
    str_dict,
    /// Structs (Pydantic models), other slices (`list[T]`) and optionals,
    /// typed by a Pydantic JSON schema (see types.zig).
    json,
};

/// A `dict[str, str]` field value: every value is a `.string`.
pub const StrDict = struct { map: std.json.ObjectMap };

/// DSPy native reasoning. Its custom serializer emits the content string.
pub const Reasoning = struct { content: []const u8 };

/// `dict[str, V]` for any supported nested `V`.
pub const Dict = nested.Dict;

pub const Field = struct {
    name: []const u8,
    kind: Kind,
    /// Python annotation as DSPy prints it, e.g. `list[str]`.
    type_name: []const u8,
    /// Documentation supplied by custom DSPy types, independent of user desc.
    type_description: []const u8 = "",
    /// `desc` as stored by DSPy: user text, or `${name}` when none given.
    desc: []const u8,
    /// `prefix` as stored by DSPy: `infer_prefix(name) + ":"`.
    prefix: []const u8,
    /// Allowed values for `.literal`, in declaration order.
    literals: []const []const u8 = &.{},
    /// `.json`: the Pydantic JSON schema, its validator, and whether the
    /// type is `Optional[str]` (whose text DSPy keeps unparsed).
    schema: []const u8 = "",
    /// TypeAdapter schemas preserve model field order for XML/native output.
    xml_schema: []const u8 = "",
    validate: ?*const fn (std.mem.Allocator, std.json.Value) nested.Error!std.json.Value = null,
    optional_str: bool = false,
    /// Native Reasoning fields are omitted from prompts when the LM supports it.
    native_reasoning: bool = false,
    /// Default CoT uses this wire name only when native reasoning is unavailable.
    reasoning_fallback: ?[]const u8 = null,
    /// Custom DSPy content types keep their JSON conventions inside XML leaves.
    xml_literal: bool = false,
    native_tools: bool = false,
    native_tool_calls: bool = false,
    nullable: bool = false,
    /// Lazily encode a declared Zig field default into the per-call arena.
    default_value: ?*const fn (std.mem.Allocator) std.mem.Allocator.Error!std.json.Value = null,
};

/// An absent field differs from an explicit null value.
pub fn outputDefault(arena: std.mem.Allocator, field: Field) std.mem.Allocator.Error!?std.json.Value {
    if (field.default_value) |get| return try get(arena);
    return if (field.nullable) @as(std.json.Value, .null) else null;
}

fn defaultProvider(comptime T: type, comptime raw: *const anyopaque) *const fn (std.mem.Allocator) std.mem.Allocator.Error!std.json.Value {
    return struct {
        fn get(arena: std.mem.Allocator) std.mem.Allocator.Error!std.json.Value {
            const v: T = @as(*const T, @ptrCast(@alignCast(raw))).*;
            return @import("value.zig").toValue(arena, T, v);
        }
    }.get;
}

pub const Description = struct { name: []const u8, text: []const u8 };

pub const reserved_names = [_][]const u8{"completed"};

/// Build a signature type from a spec type (see module docs).
pub fn Signature(comptime Spec: type) type {
    comptime {
        if (!@hasDecl(Spec, "Input")) {
            @compileError("dszig: signature spec " ++ @typeName(Spec) ++
                " must declare `pub const Input = struct { ... }`");
        }
        if (!@hasDecl(Spec, "Output")) {
            @compileError("dszig: signature spec " ++ @typeName(Spec) ++
                " must declare `pub const Output = struct { ... }`");
        }
        const instructions: ?[]const u8 = if (@hasDecl(Spec, "instructions")) Spec.instructions else null;
        const descriptions = if (@hasDecl(Spec, "descriptions")) descriptionList(Spec.descriptions) else &[_]Description{};
        return Make(Spec.Input, Spec.Output, instructions, descriptions);
    }
}

fn descriptionList(comptime descriptions: anytype) []const Description {
    comptime {
        const D = @TypeOf(descriptions);
        if (@typeInfo(D) != .@"struct") {
            @compileError("dszig: `descriptions` must be a struct literal like .{ .field = \"text\" }");
        }
        const names = @typeInfo(D).@"struct".field_names;
        var list: [names.len]Description = undefined;
        for (names, 0..) |name, i| {
            const text: []const u8 = @field(descriptions, name);
            list[i] = .{ .name = name, .text = text };
        }
        const final = list;
        return &final;
    }
}

/// Lower-level constructor used by `Signature` and by signature transforms
/// such as `ChainOfThought`'s `reasoning` prepend.
pub fn Make(
    comptime InputT: type,
    comptime OutputT: type,
    comptime instructions_opt: ?[]const u8,
    comptime descriptions: []const Description,
) type {
    comptime {
        const input_fields = fieldsOf(InputT, "Input", descriptions);
        const output_fields = fieldsOf(OutputT, "Output", descriptions);
        validate(&input_fields, &output_fields, descriptions);
        const all = input_fields ++ output_fields;
        const instructions_final = instructions_opt orelse defaultInstructions(&input_fields, &output_fields);
        return struct {
            pub const Input = InputT;
            pub const Output = OutputT;
            pub const input_fields_list: []const Field = &input_fields;
            pub const output_fields_list: []const Field = &output_fields;
            pub const fields: []const Field = &all;
            pub const instructions: []const u8 = instructions_final;
            pub const description_list: []const Description = descriptions;
            pub const is_dszig_signature = true;

            /// `signature.prepend(name, OutputField(), type_)` from DSPy.
            pub fn PrependOutput(comptime name: [:0]const u8, comptime T: type) type {
                const info = @typeInfo(OutputT).@"struct";
                const names = [_][:0]const u8{name} ++ info.field_names[0..info.field_names.len].*;
                const types = [_]type{T} ++ info.field_types[0..info.field_types.len].*;
                const attrs = [_]@TypeOf(info.field_attrs[0]){.{}} ++ info.field_attrs[0..info.field_attrs.len].*;
                const NewOutput = @Struct(.auto, null, &names, &types, &attrs);
                return Make(InputT, NewOutput, instructions_final, descriptions);
            }
        };
    }
}

pub fn isSignature(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "is_dszig_signature");
}

fn fieldsOf(comptime T: type, comptime role: []const u8, comptime descriptions: []const Description) [fieldCount(T, role)]Field {
    comptime {
        const info = @typeInfo(T).@"struct";
        var out: [info.field_names.len]Field = undefined;
        for (info.field_names, info.field_types, 0..) |name, FT, i| {
            const kind = kindOf(FT, name);
            out[i] = .{
                .name = name,
                .kind = kind,
                .type_name = typeName(FT, kind),
                .desc = descriptionFor(name, descriptions) orelse "${" ++ name ++ "}",
                .prefix = inferPrefix(name) ++ ":",
                .literals = if (kind == .literal) std.meta.fieldNames(FT) else &.{},
                .schema = if (FT == History) History.schema else if (FT == ToolCalls) ToolCalls.schema else if (kind == .json) nested.schema(FT) else "",
                .xml_schema = if (FT == History) History.schema else if (FT == ToolCalls) ToolCalls.schema else if (kind == .json) nested.schemaXml(FT) else "",
                .validate = if (FT == History) History.validate else if (FT == ToolCalls) ToolCalls.validate else if (kind == .json) nested.Validator(FT) else null,
                .optional_str = nested.isOptionalStr(FT),
                .native_reasoning = FT == Reasoning or FT == ?Reasoning,
                .xml_literal = FT == History or FT == ToolCalls,
                .native_tool_calls = FT == ToolCalls,
                .type_description = if (FT == ToolCalls) ToolCalls.description else "",
                .nullable = @typeInfo(FT) == .optional,
                .default_value = if (info.field_attrs[i].default_value_ptr) |raw| defaultProvider(FT, raw) else null,
            };
        }
        return out;
    }
}

fn fieldCount(comptime T: type, comptime role: []const u8) usize {
    if (@typeInfo(T) != .@"struct") {
        @compileError("dszig: signature `" ++ role ++ "` must be a struct, found " ++ @typeName(T));
    }
    const info = @typeInfo(T).@"struct";
    if (info.is_tuple) @compileError("dszig: signature `" ++ role ++ "` must not be a tuple");
    return info.field_names.len;
}

fn descriptionFor(comptime name: []const u8, comptime descriptions: []const Description) ?[]const u8 {
    for (descriptions) |d| {
        if (std.mem.eql(u8, d.name, name)) return d.text;
    }
    return null;
}

/// Map a Zig field type onto a supported DSPy annotation.
pub fn kindOf(comptime T: type, comptime name: []const u8) Kind {
    if (T == []const u8 or T == Reasoning or T == ?Reasoning) return .str;
    if (T == []const []const u8) return .str_list;
    if (T == bool) return .boolean;
    if (T == std.json.ObjectMap) return .json_object;
    if (T == StrDict) return .str_dict;
    if (nested.isNested(T)) return .json;
    switch (@typeInfo(T)) {
        .int => return .int,
        .float => |f| {
            if (f.bits != 32 and f.bits != 64) {
                @compileError("dszig: field `" ++ name ++ "` float must be f32 or f64");
            }
            return .float;
        },
        .@"enum" => |e| {
            if (e.mode != .exhaustive) {
                @compileError("dszig: field `" ++ name ++ "` enum must be exhaustive to map to Literal[...]");
            }
            if (e.field_names.len == 0) @compileError("dszig: field `" ++ name ++ "` enum has no values");
            return .literal;
        },
        else => @compileError("dszig: field `" ++ name ++ "` has unsupported type " ++ @typeName(T) ++
            "; supported: []const u8, []const []const u8, bool, integers, f32/f64, exhaustive enums, std.json.ObjectMap, StrDict"),
    }
}

fn typeName(comptime T: type, comptime kind: Kind) []const u8 {
    return switch (kind) {
        .str => "str",
        .str_list => "list[str]",
        .boolean => "bool",
        .int => "int",
        .float => "float",
        .json_object => "dict[str, Any]",
        .str_dict => "dict[str, str]",
        .json => if (T == History) "History" else if (T == ToolCalls) "ToolCalls" else nested.pyName(T),
        .literal => blk: {
            var s: []const u8 = "Literal[";
            for (std.meta.fieldNames(T), 0..) |tag, i| {
                if (i > 0) s = s ++ ", ";
                s = s ++ "'" ++ tag ++ "'";
            }
            break :blk s ++ "]";
        },
    };
}

fn validate(comptime inputs: []const Field, comptime outputs: []const Field, comptime descriptions: []const Description) void {
    if (outputs.len == 0) @compileError("dszig: signature must declare at least one output field");
    for (inputs ++ outputs, 0..) |f, i| {
        if (!isIdentifier(f.name)) {
            @compileError("dszig: field name `" ++ f.name ++ "` must match \\w+ (ASCII letters, digits, '_')");
        }
        for (reserved_names) |r| {
            if (std.mem.eql(u8, f.name, r)) @compileError("dszig: field name `" ++ f.name ++ "` is reserved by the chat protocol");
        }
        for ((inputs ++ outputs)[0..i]) |prev| {
            if (std.mem.eql(u8, prev.name, f.name)) {
                @compileError("dszig: field `" ++ f.name ++ "` is declared as both input and output");
            }
        }
        if (f.kind == .literal) {
            for (f.literals) |tag| {
                if (std.mem.indexOfScalar(u8, tag, '\'') != null) {
                    @compileError("dszig: literal value `" ++ tag ++ "` of field `" ++ f.name ++ "` must not contain a quote");
                }
            }
        }
    }
    for (descriptions) |d| {
        var found = false;
        for (inputs ++ outputs) |f| found = found or std.mem.eql(u8, f.name, d.name);
        if (!found) @compileError("dszig: description for unknown field `" ++ d.name ++ "`");
    }
}

fn isIdentifier(comptime name: []const u8) bool {
    if (name.len == 0) return false;
    for (name) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    }
    return true;
}

/// DSPy `_default_instructions`.
fn defaultInstructions(comptime inputs: []const Field, comptime outputs: []const Field) []const u8 {
    return "Given the fields " ++ backtickList(inputs) ++ ", produce the fields " ++ backtickList(outputs) ++ ".";
}

fn backtickList(comptime fields: []const Field) []const u8 {
    var s: []const u8 = "";
    for (fields, 0..) |f, i| {
        if (i > 0) s = s ++ ", ";
        s = s ++ "`" ++ f.name ++ "`";
    }
    return s;
}

/// Port of DSPy `infer_prefix`: snake/camel case to Title Case words.
pub fn inferPrefix(comptime name: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(100_000);
        const s1 = splitCamelWords(name);
        const s2 = splitPairs(s1, isLowerOrDigit, isUpper);
        const s3 = splitPairs(s2, std.ascii.isAlphabetic, std.ascii.isDigit);
        const s4 = splitPairs(s3, std.ascii.isDigit, std.ascii.isAlphabetic);
        var out: []const u8 = "";
        var it = std.mem.splitScalar(u8, s4, '_');
        var first = true;
        while (it.next()) |word| {
            if (!first) out = out ++ " ";
            first = false;
            out = out ++ (if (isUpperWord(word)) word else capitalize(word));
        }
        return out;
    }
}

fn isUpper(c: u8) bool {
    return std.ascii.isUpper(c);
}

fn isLowerOrDigit(c: u8) bool {
    return std.ascii.isLower(c) or std.ascii.isDigit(c);
}

/// `re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)`.
fn splitCamelWords(comptime s: []const u8) []const u8 {
    var out: []const u8 = "";
    var i: usize = 0;
    while (i < s.len) {
        if (i + 2 < s.len and s[i] != '\n' and isUpper(s[i + 1]) and std.ascii.isLower(s[i + 2])) {
            var end = i + 3;
            while (end < s.len and std.ascii.isLower(s[end])) end += 1;
            out = out ++ s[i .. i + 1] ++ "_" ++ s[i + 1 .. end];
            i = end;
        } else {
            out = out ++ s[i .. i + 1];
            i += 1;
        }
    }
    return out;
}

/// `re.sub("(A)(B)", r"\1_\2", s)` for single-character classes A and B.
fn splitPairs(comptime s: []const u8, comptime a: fn (u8) bool, comptime b: fn (u8) bool) []const u8 {
    var out: []const u8 = "";
    var i: usize = 0;
    while (i < s.len) {
        if (i + 1 < s.len and a(s[i]) and b(s[i + 1])) {
            out = out ++ s[i .. i + 1] ++ "_" ++ s[i + 1 .. i + 2];
            i += 2;
        } else {
            out = out ++ s[i .. i + 1];
            i += 1;
        }
    }
    return out;
}

/// Python `str.isupper()` for ASCII.
fn isUpperWord(comptime w: []const u8) bool {
    var cased = false;
    for (w) |c| {
        if (std.ascii.isLower(c)) return false;
        if (std.ascii.isUpper(c)) cased = true;
    }
    return cased;
}

/// Python `str.capitalize()` for ASCII.
fn capitalize(comptime w: []const u8) []const u8 {
    if (w.len == 0) return w;
    var buf: [w.len]u8 = undefined;
    buf[0] = std.ascii.toUpper(w[0]);
    for (w[1..], 1..) |c, i| buf[i] = std.ascii.toLower(c);
    const final = buf;
    return &final;
}

const testing = std.testing;

test "inferPrefix matches DSPy" {
    try testing.expectEqualStrings("Answer", comptime inferPrefix("answer"));
    try testing.expectEqualStrings("Search Query", comptime inferPrefix("search_query"));
    try testing.expectEqualStrings("Camel Case Name", comptime inferPrefix("camelCaseName"));
    try testing.expectEqualStrings("HTML Parser", comptime inferPrefix("HTMLParser"));
    try testing.expectEqualStrings("Text 2 Number", comptime inferPrefix("text2number"));
    try testing.expectEqualStrings("Top K", comptime inferPrefix("top_k"));
}

test "signature structure" {
    const QA = Signature(struct {
        pub const instructions = "Answer questions with short factoid answers.";
        pub const Input = struct { context: []const []const u8, question: []const u8 };
        pub const Output = struct { answer: []const u8 };
        pub const descriptions = .{ .context = "may contain relevant facts", .answer = "often between 1 and 5 words" };
    });
    comptime assert(isSignature(QA));
    try testing.expectEqual(@as(usize, 3), QA.fields.len);
    try testing.expectEqualStrings("list[str]", QA.fields[0].type_name);
    try testing.expectEqualStrings("${question}", QA.fields[1].desc);
    try testing.expectEqualStrings("Question:", QA.fields[1].prefix);
    const CoT = QA.PrependOutput("reasoning", []const u8);
    try testing.expectEqualStrings("reasoning", CoT.output_fields_list[0].name);
    try testing.expectEqualStrings("answer", CoT.output_fields_list[1].name);
    try testing.expectEqualStrings(QA.instructions, CoT.instructions);
    try testing.expectEqualStrings("often between 1 and 5 words", CoT.output_fields_list[1].desc);

    const Default = Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { sentiment: enum { positive, negative }, score: f64 };
    });
    try testing.expectEqualStrings("Given the fields `question`, produce the fields `sentiment`, `score`.", Default.instructions);
    try testing.expectEqualStrings("Literal['positive', 'negative']", Default.fields[1].type_name);
}
