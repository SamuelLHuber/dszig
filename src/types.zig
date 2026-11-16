//! Nested field types, as DSPy types them with Pydantic: a Zig struct is a
//! `BaseModel` (named by its `model_name` decl or its type name), a slice is
//! `list[T]`, a fixed array is a homogeneous fixed tuple, and an optional is
//! `Optional[T]`. This file gives their Python
//! annotation names, their Pydantic JSON schemas with keys in DSPy's
//! `move_type_to_front` order, and Pydantic's lax validation, which also
//! normalizes a value to declaration order with defaults filled in.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const code = @import("code.zig");
const BoundingBox = @import("bounding_box.zig").BoundingBox;
const File = @import("file.zig").File;
const citation = @import("citation.zig");
const decision = @import("decision.zig");
const Integer = @import("integer.zig").Integer;
const decimal_type = @import("decimal_type.zig");

pub const Error = error{ ParseValue, OutOfMemory };

/// `dict[str, V]`: keys and values in insertion order.
pub fn Dict(comptime V: type) type {
    return struct {
        const Self = @This();
        pub const dszig_dict_value = V;

        keys: []const []const u8 = &.{},
        values: []const V = &.{},

        pub fn get(self: Self, key: []const u8) ?V {
            for (self.keys, self.values) |k, v| {
                if (std.mem.eql(u8, k, key)) return v;
            }
            return null;
        }
    };
}

pub fn isDict(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "dszig_dict_value");
}

/// A struct typed as a Pydantic model (not a dict).
fn isModel(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and T != Integer and !decimal_type.isDecimal(T) and !@import("literal.zig").isLiteral(T) and T != @import("runtime_value.zig").ObjectMap and !isDict(T) and !@hasField(T, "map");
}

/// Types that need the JSON-typed field kind.
pub fn isNested(comptime T: type) bool {
    if (T == Value) return true;
    if (T == Integer) return false;
    if (T == []const u8 or T == []const []const u8 or T == @import("runtime_value.zig").ObjectMap) return false;
    return switch (@typeInfo(T)) {
        .optional, .array => true,
        .@"struct" => !@hasField(T, "map"),
        .pointer => |p| p.size == .slice,
        else => false,
    };
}

/// `Optional[str]`: DSPy returns its text unparsed.
pub fn isOptionalStr(comptime T: type) bool {
    return T == ?[]const u8;
}

pub fn modelName(comptime T: type) []const u8 {
    if (@hasDecl(T, "model_name")) return T.model_name;
    const full = @typeName(T);
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return full;
    return full[dot + 1 ..];
}

/// `get_annotation_name(annotation)`.
pub fn pyName(comptime T: type) []const u8 {
    comptime {
        if (T == []const u8 or T == @import("signature.zig").Reasoning) return "str";
        if (T == bool) return "bool";
        if (T == Integer) return "int";
        if (decimal_type.isDecimal(T)) return "Decimal";
        if (@import("literal.zig").isLiteral(T)) return T.annotation_name;
        if (T == Value) return "Any";
        if (T == @import("runtime_value.zig").ObjectMap) return "dict[str, Any]";
        if (isDict(T)) return "dict[str, " ++ pyName(T.dszig_dict_value) ++ "]";
        return switch (@typeInfo(T)) {
            .int => "int",
            .float => "float",
            .@"enum" => |e| blk: {
                var s: []const u8 = "Literal[";
                for (e.field_names, 0..) |tag, i| s = s ++ (if (i > 0) ", " else "") ++ "'" ++ tag ++ "'";
                break :blk s ++ "]";
            },
            .optional => |o| "Union[" ++ pyName(o.child) ++ ", NoneType]",
            .array => |a| blk: {
                var name: []const u8 = "tuple[";
                for (0..a.len) |i| name = name ++
                    (if (i == 0) "" else ", ") ++ pyName(a.child);
                break :blk name ++ "]";
            },
            .pointer => |p| "list[" ++ pyName(p.child) ++ "]",
            .@"struct" => modelName(T),
            else => @compileError("dszig: unsupported nested type " ++ @typeName(T)),
        };
    }
}

const Entry = struct { key: []const u8, val: []const u8 };

fn lessKey(a: []const u8, b: []const u8) bool {
    const at = std.mem.eql(u8, a, "type");
    const bt = std.mem.eql(u8, b, "type");
    if (at != bt) return at;
    return std.mem.lessThan(u8, a, b);
}

fn render(comptime entries: []const Entry) []const u8 {
    comptime {
        var sorted: [entries.len]Entry = entries[0..entries.len].*;
        var i: usize = 1;
        while (i < sorted.len) : (i += 1) {
            var j = i;
            while (j > 0 and lessKey(sorted[j].key, sorted[j - 1].key)) : (j -= 1) {
                const t = sorted[j];
                sorted[j] = sorted[j - 1];
                sorted[j - 1] = t;
            }
        }
        return renderOrdered(&sorted);
    }
}

fn renderOrdered(comptime entries: []const Entry) []const u8 {
    comptime {
        var s: []const u8 = "{";
        for (entries, 0..) |e, i| s = s ++ (if (i > 0) ", " else "") ++ "\"" ++ e.key ++ "\": " ++ e.val;
        return s ++ "}";
    }
}

fn quote(comptime s: []const u8) []const u8 {
    comptime {
        if (!std.unicode.utf8ValidateSlice(s)) @compileError("dszig: schema text must be valid UTF-8");
        var out: []const u8 = "\"";
        for (s) |byte| {
            out = out ++ switch (byte) {
                '"' => "\\\"",
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                0x08 => "\\b",
                0x0c => "\\f",
                else => if (byte < 0x20) std.fmt.comptimePrint("\\u{x:0>4}", .{byte}) else &[_]u8{byte},
            };
        }
        return out ++ "\"";
    }
}

fn definitionName(comptime T: type) []const u8 {
    return if (code.isCode(T) or decision.isDecision(T)) T.schema_name else modelName(T);
}

/// Pydantic's `name.title().replace("_", " ").strip()`.
pub fn title(comptime name: []const u8) []const u8 {
    return @import("unicode.zig").schemaTitleComptime(name);
}

/// Schema of `T` where it is used (a struct is a `$ref`).
fn entriesOf(comptime T: type) []const Entry {
    comptime {
        if (T == []const u8) return &.{.{ .key = "type", .val = quote("string") }};
        if (T == bool) return &.{.{ .key = "type", .val = quote("boolean") }};
        if (T == Integer) return &.{.{ .key = "type", .val = quote("integer") }};
        if (decimal_type.isDecimal(T)) return &.{.{ .key = "anyOf", .val = "[{\"type\": \"number\"}, {\"type\": \"string\"}]" }};
        if (@import("literal.zig").isLiteral(T)) return literalEntries(T);
        if (T == Value) return &.{};
        if (T == @import("runtime_value.zig").ObjectMap) return &.{ .{ .key = "type", .val = quote("object") }, .{ .key = "additionalProperties", .val = "true" } };
        if (isDict(T)) return &.{ .{ .key = "type", .val = quote("object") }, .{ .key = "additionalProperties", .val = if (T.dszig_dict_value == Value) "true" else render(entriesOf(T.dszig_dict_value)) } };
        return switch (@typeInfo(T)) {
            .int => &.{.{ .key = "type", .val = quote("integer") }},
            .float => &.{.{ .key = "type", .val = quote("number") }},
            .@"enum" => |e| blk: {
                var s: []const u8 = "[";
                for (e.field_names, 0..) |tag, i| s = s ++ (if (i > 0) ", " else "") ++ quote(tag);
                break :blk &.{ .{ .key = "type", .val = quote("string") }, .{ .key = "enum", .val = s ++ "]" } };
            },
            .optional => |o| &.{.{ .key = "anyOf", .val = "[" ++ render(entriesOf(o.child)) ++ ", {\"type\": \"null\"}]" }},
            .array => |a| blk: {
                const length = std.fmt.comptimePrint("{d}", .{a.len});
                const bounds: []const Entry = &.{
                    .{ .key = "type", .val = quote("array") },
                    .{ .key = "maxItems", .val = length },
                    .{ .key = "minItems", .val = length },
                };
                if (a.len == 0) break :blk bounds;
                var prefix: []const u8 = "[";
                for (0..a.len) |i| prefix = prefix ++
                    (if (i == 0) "" else ", ") ++ render(entriesOf(a.child));
                break :blk bounds ++ &[_]Entry{
                    .{ .key = "prefixItems", .val = prefix ++ "]" },
                };
            },
            .pointer => |p| &.{ .{ .key = "type", .val = quote("array") }, .{ .key = "items", .val = render(entriesOf(p.child)) } },
            .@"struct" => &.{.{ .key = "$ref", .val = quote("#/$defs/" ++ definitionName(T)) }},
            else => @compileError("dszig: unsupported nested type " ++ @typeName(T)),
        };
    }
}

fn literalEntries(comptime T: type) []const Entry {
    comptime {
        const values = T.dszig_literal_values;
        var encoded: []const u8 = if (values.len == 1) enumSchemaValue(values[0]) else "[";
        if (values.len > 1) {
            for (values, 0..) |v, i| encoded = encoded ++ (if (i == 0) "" else ", ") ++ enumSchemaValue(v);
            encoded = encoded ++ "]";
        }
        const fields: []const Entry = &.{.{ .key = if (values.len == 1) "const" else "enum", .val = encoded }};
        const first = literalJsonType(values[0]) orelse return fields;
        for (values[1..]) |v| {
            const kind = literalJsonType(v) orelse return fields;
            if (!std.mem.eql(u8, first, kind)) return fields;
        }
        return &[_]Entry{.{ .key = "type", .val = quote(first) }} ++ fields;
    }
}

fn literalJsonType(comptime v: Value) ?[]const u8 {
    // Pydantic selects the type after recursively projecting Enum values.
    return switch (v) {
        .null => "null",
        .bool => "boolean",
        .integer, .number_string => "integer",
        .float => "number",
        .string => "string",
        .array => "array",
        .custom => |c| if (c.vtable.enumeration) |inspect|
            literalJsonType(inspect(c.ptr).value)
        else if (@import("python_value.zig").view(v)) |builtin|
            if (builtin == .bytes) "string" else null
        else
            null,
        else => null,
    };
}

fn enumEntries(comptime T: type) []const Entry {
    comptime {
        if (T.dszig_enum_values.len == 0)
            @compileError("dszig: an empty Enum has no JSON schema");
        var encoded: []const u8 = "[";
        for (T.dszig_enum_values, 0..) |v, i| {
            encoded = encoded ++ (if (i == 0) "" else ", ") ++ enumSchemaValue(v);
        }
        const kind = T.dszig_enum_json_type;
        const fields: []const Entry = &.{
            .{ .key = "enum", .val = encoded ++ "]" },
            .{ .key = "title", .val = quote(T.model_name) },
        };
        return if (kind) |name| &[_]Entry{.{ .key = "type", .val = quote(name) }} ++ fields else fields;
    }
}

fn enumSchemaValue(comptime v: Value) []const u8 {
    comptime {
        return switch (v) {
            .null => "null",
            .bool => |b| if (b) "true" else "false",
            .integer => |i| std.fmt.comptimePrint("{d}", .{i}),
            .number_string => |s| s,
            .string => |s| quote(s),
            .float => |f| blk: {
                if (std.math.isNan(f)) break :blk "NaN";
                if (std.math.isInf(f)) break :blk if (f < 0) "-Infinity" else "Infinity";
                var buffer: [64]u8 = undefined;
                var writer: std.Io.Writer = .fixed(&buffer);
                @import("pyjson.zig").writeStr(&writer, v) catch @compileError("dszig: invalid enum float");
                const result = buffer[0..writer.buffered().len].*;
                break :blk &result;
            },
            .array => |a| blk: {
                var out: []const u8 = "[";
                for (a.items, 0..) |item, i| out = out ++ (if (i == 0) "" else ", ") ++ enumSchemaValue(item);
                break :blk out ++ "]";
            },
            .object => |o| blk: {
                var out: []const u8 = "{";
                for (o.keys(), o.values(), 0..) |key, item, i|
                    out = out ++ (if (i == 0) "" else ", ") ++ quote(key) ++ ": " ++ enumSchemaValue(item);
                break :blk out ++ "}";
            },
            .custom => |c| if (c.vtable.enumeration) |inspect|
                enumSchemaValue(inspect(c.ptr).value)
            else blk: {
                if (@import("python_value.zig").view(v)) |builtin| {
                    if (builtin == .bytes) {
                        if (!std.unicode.utf8ValidateSlice(builtin.bytes))
                            @compileError("dszig: non-UTF-8 bytes cannot produce a JSON schema");
                        break :blk quote(builtin.bytes);
                    }
                }
                @compileError("dszig: enum JSON schema requires serializable native values");
            },
        };
    }
}

fn hasDefault(comptime S: type, comptime i: usize) bool {
    return @typeInfo(S).@"struct".field_attrs[i].default_value_ptr != null;
}

fn modelEntries(comptime S: type, comptime ordered_properties: bool) []const Entry {
    comptime {
        if (@import("enum.zig").isEnum(S)) return enumEntries(S);
        if (decision.isDecision(S)) return decisionEntries(S, ordered_properties);
        const info = @typeInfo(S).@"struct";
        var props: [if (code.isCode(S) or S == citation.Citations) 1 else if (S == citation.Citations.Citation) 8 else info.field_names.len]Entry = undefined;
        var n_props = 0;
        var required: []const u8 = "[";
        var n_required = 0;
        for (info.field_names, info.field_types, 0..) |name, FT, i| {
            if (code.isCode(S) and !std.mem.eql(u8, name, "code")) continue;
            if (citation.isCitation(S) and std.mem.eql(u8, name, "runtime_origin")) continue;
            var e: []const Entry = entriesOf(FT) ++ &[_]Entry{.{ .key = "title", .val = quote(title(name)) }};
            if (isModel(FT) or (@typeInfo(FT) == .optional and isModel(@typeInfo(FT).optional.child))) e = entriesOf(FT);
            if (S == BoundingBox) e = e ++ &[_]Entry{.{ .key = "description", .val = quote(BoundingBox.edge_descriptions[i]) }};
            if (hasDefault(S, i)) {
                if (S == citation.Citations.Citation and std.mem.eql(u8, name, "type")) {
                    e = e ++ &[_]Entry{.{ .key = "default", .val = quote("char_location") }};
                } else {
                    if (@typeInfo(FT) != .optional or @as(*const FT, @ptrCast(@alignCast(info.field_attrs[i].default_value_ptr.?))).* != null) {
                        @compileError("dszig: model field `" ++ name ++ "` may only default to null");
                    }
                    e = e ++ &[_]Entry{.{ .key = "default", .val = "null" }};
                }
            } else {
                required = required ++ (if (n_required > 0) ", " else "") ++ quote(name);
                n_required += 1;
            }
            props[n_props] = .{ .key = name, .val = render(e) };
            n_props += 1;
        }
        const final = props;
        const description: []const Entry = if (code.isCode(S) and S.schema_description_json.len > 0)
            &.{.{ .key = "description", .val = S.schema_description_json }}
        else if (@hasDecl(S, "schema_description"))
            &.{.{ .key = "description", .val = quote(S.schema_description) }}
        else if (S == @import("signature.zig").Reasoning)
            &.{.{ .key = "description", .val = "\"Reasoning type in DSPy.\\n\\nThis type is useful when you want the DSPy output to include the reasoning of the LM. We build this type so that\\nDSPy can support the reasoning model and non-reasoning model with the same code.\\n\\nThis is a str-like type, you can convert a string directly to a Reasoning object, and from DSPy adapters'\\nperspective, `Reasoning` is treated as a string.\"" }}
        else
            &.{};
        return &[_]Entry{
            .{ .key = "type", .val = quote("object") },
            .{ .key = "properties", .val = if (ordered_properties) renderOrdered(&final) else render(&final) },
        } ++ (if (n_required != 0) &[_]Entry{.{ .key = "required", .val = required ++ "]" }} else &[_]Entry{}) ++ &[_]Entry{
            .{ .key = "title", .val = quote(modelName(S)) },
        } ++ description ++ (if (S == BoundingBox or S == File) &[_]Entry{.{ .key = "additionalProperties", .val = "false" }} else &[_]Entry{});
    }
}

fn decisionEntries(comptime S: type, comptime ordered_properties: bool) []const Entry {
    comptime {
        const confidence = render(&.{
            .{ .key = "type", .val = "\"number\"" },
            .{ .key = "description", .val = "\"Confidence in the value, from 0 to 1.\"" },
            .{ .key = "maximum", .val = "1" },
            .{ .key = "minimum", .val = "0" },
            .{ .key = "title", .val = "\"Confidence\"" },
        });
        const member: []const Entry = switch (S.kind) {
            .noul => entriesOf(bool),
            .score => entriesOf(f64) ++ (if (S.configured) &[_]Entry{
                .{ .key = "maximum", .val = std.fmt.comptimePrint("{d}", .{S.members.len - 1}) },
                .{ .key = "minimum", .val = "0" },
            } else &[_]Entry{}),
            .choice => if (S.configured) literalEntries(S.LiteralType) else &.{
                .{ .key = "anyOf", .val = "[{\"type\": \"string\"}, {\"type\": \"integer\"}, {\"type\": \"boolean\"}, {\"type\": \"null\"}]" },
            },
        };
        const properties: []const Entry = &.{
            .{ .key = "confidence", .val = confidence },
            .{ .key = "value", .val = render(member ++ &[_]Entry{.{ .key = "title", .val = "\"Value\"" }}) },
        };
        return &[_]Entry{
            .{ .key = "type", .val = "\"object\"" },
            .{ .key = "additionalProperties", .val = "false" },
            .{ .key = "properties", .val = if (ordered_properties) renderOrdered(properties) else render(properties) },
            .{ .key = "required", .val = "[\"confidence\", \"value\"]" },
            .{ .key = "title", .val = quote(S.model_name) },
        } ++ (if (S.schema_description_json.len != 0) &[_]Entry{
            .{ .key = "description", .val = S.schema_description_json },
        } else &[_]Entry{});
    }
}

fn collectDefs(comptime T: type, comptime seen: []const type) []const type {
    comptime {
        var out = seen;
        if (T == Value or T == Integer or decimal_type.isDecimal(T) or @import("literal.zig").isLiteral(T) or T == @import("runtime_value.zig").ObjectMap) return out;
        if (isDict(T)) return collectDefs(T.dszig_dict_value, out);
        switch (@typeInfo(T)) {
            .optional => |o| out = collectDefs(o.child, out),
            .array => |a| if (a.len != 0) {
                out = collectDefs(a.child, out);
            },
            .pointer => |p| if (T != []const u8) {
                out = collectDefs(p.child, out);
            },
            .@"struct" => |s| {
                for (out) |t| if (t == T) return out;
                out = out ++ &[_]type{T};
                if (code.isCode(T) or decision.isDecision(T) or @import("enum.zig").isEnum(T)) return out;
                for (s.field_names, s.field_types) |name, FT| {
                    if (citation.isCitation(T) and std.mem.eql(u8, name, "runtime_origin")) continue;
                    out = collectDefs(FT, out);
                }
            },
            else => {},
        }
        return out;
    }
}

/// `json.dumps(_get_json_schema(T))`.
pub fn schema(comptime T: type) []const u8 {
    return comptime schemaImpl(T, false);
}

/// XML and native schemas preserve the model's field declaration order.
pub fn schemaXml(comptime T: type) []const u8 {
    return comptime schemaImpl(T, true);
}

fn schemaImpl(comptime T: type, comptime ordered_properties: bool) []const u8 {
    comptime {
        @setEvalBranchQuota(1_000_000);
        const is_model = isModel(T);
        const all = collectDefs(T, &.{});
        const defs = if (is_model) all[1..] else all;
        var top: []const Entry = if (is_model) modelEntries(T, ordered_properties) else entriesOf(T);
        if (defs.len > 0) {
            var d: [defs.len]Entry = undefined;
            for (defs, 0..) |D, i| {
                for (defs[0..i]) |E| if (std.mem.eql(u8, definitionName(D), definitionName(E))) @compileError("dszig: two models share schema definition name " ++ definitionName(D));
                d[i] = .{ .key = definitionName(D), .val = render(modelEntries(D, ordered_properties)) };
            }
            const final = d;
            top = top ++ &[_]Entry{.{ .key = "$defs", .val = render(&final) }};
        }
        return render(top);
    }
}

/// `TypeAdapter(T).validate_python(v)`, including configured model strictness.
pub fn validate(comptime T: type, arena: Allocator, v: Value) Error!Value {
    return validateMode(T, arena, v, false);
}

fn validateMode(comptime T: type, arena: Allocator, input: Value, comptime xml_lax: bool) Error!Value {
    var v = input;
    if (T == Value) return input;
    if (comptime code.isCode(T) or @import("enum.zig").isEnum(T) or @import("literal.zig").isLiteral(T)) return T.validate(arena, v);
    if (T == Integer) return @import("value.zig").coerceInteger(arena, v);
    if (comptime decimal_type.isDecimal(T)) return T.validate(arena, v);
    if (comptime citation.isCitation(T)) return T.validate(arena, v);
    if (comptime decision.isDecision(T)) return if (xml_lax) T.validateLax(arena, v) else T.validate(arena, v);
    if (T == File) {
        const file = File.fromValue(arena, v) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        };
        return if (v == .custom and v.custom.get(File) != null) v else file.toRuntime(arena);
    }
    if (T == BoundingBox) {
        const box = (if (xml_lax) BoundingBox.fromXmlValue(v) else BoundingBox.fromValue(v)) catch return error.ParseValue;
        return if (v == .custom) v else box.toRuntime(arena);
    }
    if (T == @import("signature.zig").Reasoning) {
        const reasoning = @import("signature.zig").Reasoning.fromValue(v) catch return error.ParseValue;
        return if (v == .custom) v else reasoning.toRuntime(arena);
    }
    if (v == .custom and (T == bool or @typeInfo(T) == .float)) {
        if (v.custom.integerValue()) |integer| v = integer;
    }
    if (T == []const u8 and v == .custom) {
        if (v.custom.vtable.enumeration) |inspect| {
            const member_value = inspect(v.custom.ptr).value;
            if (member_value == .string) return member_value;
            var out: std.Io.Writer.Allocating = .init(arena);
            @import("pyjson.zig").writeStr(&out.writer, member_value) catch |err| return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                error.JsonTooDeep, error.ParseValue => error.ParseValue,
            };
            return .{ .string = out.written() };
        }
    }
    if (v == .custom and comptime @typeInfo(T) == .float) {
        if (v.custom.vtable.float_value) |convert| return .{ .float = try convert(v.custom.ptr) };
    }
    if (v == .custom and (T == []const u8 or T == bool or @typeInfo(T) == .float)) {
        const inspect = v.custom.vtable.python orelse return error.ParseValue;
        v = switch (inspect(v.custom.ptr)) {
            .bytes => |data| blk: {
                if (T == []const u8 and !std.unicode.utf8ValidateSlice(data)) return error.ParseValue;
                break :blk .{ .string = data };
            },
            .bytearray => |data| blk: {
                if (T != []const u8 or !std.unicode.utf8ValidateSlice(data)) return error.ParseValue;
                break :blk .{ .string = try arena.dupe(u8, data) };
            },
            // Pydantic's lax bool validator uses Decimal's binary64 conversion,
            // including rounding to one and underflow to signed zero.
            .decimal => |data| if (T == bool) .{ .float = try data.toFloat() } else return error.ParseValue,
            else => return error.ParseValue,
        };
    }
    if (T == []const u8) return if (v == .string) v else error.ParseValue;
    if (T == @import("runtime_value.zig").ObjectMap) return stringKeyedObject(arena, v);
    if (comptime isDict(T)) {
        var out: @import("runtime_value.zig").ObjectMap = .empty;
        if (v == .object) {
            try out.ensureTotalCapacity(arena, v.object.count());
            var it = v.object.iterator();
            while (it.next()) |e| out.putAssumeCapacity(e.key_ptr.*, try validateMode(T.dszig_dict_value, arena, e.value_ptr.*, xml_lax));
        } else {
            if (v != .custom) return error.ParseValue;
            const inspect = v.custom.vtable.python orelse return error.ParseValue;
            const entries = switch (inspect(v.custom.ptr)) {
                .dictionary => |entries| entries,
                else => return error.ParseValue,
            };
            try out.ensureTotalCapacity(arena, entries.len);
            for (entries) |entry| {
                const key = try validateMode([]const u8, arena, entry.key, xml_lax);
                const item = try validateMode(T.dszig_dict_value, arena, entry.value, xml_lax);
                try out.put(arena, key.string, item);
            }
        }
        return .{ .object = out };
    }
    if (T == bool) return switch (v) {
        .bool => v,
        .integer => |i| if (i == 0 or i == 1) Value{ .bool = i == 1 } else error.ParseValue,
        .float => |f| if (f == 0 or f == 1) Value{ .bool = f == 1 } else error.ParseValue,
        .number_string => |s| blk: {
            const parsed = Value.parseFromNumberSlice(s);
            if (parsed == .number_string) break :blk error.ParseValue;
            break :blk validateMode(bool, arena, parsed, xml_lax);
        },
        .string => |s| boolText(s),
        else => error.ParseValue,
    };
    switch (@typeInfo(T)) {
        .int => {
            const value = @import("value.zig");
            const parsed = try value.coerceInteger(arena, v);
            _ = value.fromValue(arena, T, parsed) catch return error.ParseValue;
            return parsed;
        },
        .float => return switch (v) {
            .float => v,
            .bool => |b| .{ .float = @floatFromInt(@intFromBool(b)) },
            .integer => |i| .{ .float = @floatFromInt(i) },
            .number_string => |s| blk: {
                const number = std.fmt.parseFloat(f64, s) catch return error.ParseValue;
                if (!std.math.isFinite(number)) return error.ParseValue;
                break :blk .{ .float = number };
            },
            .string => |s| blk: {
                const token = @import("unicode.zig").strip(s, true);
                const unsigned = if (token.len != 0 and (token[0] == '-' or token[0] == '+')) token[1..] else token;
                if (std.ascii.startsWithIgnoreCase(unsigned, "0x")) return error.ParseValue;
                var number = std.fmt.parseFloat(f64, token) catch return error.ParseValue;
                if (token.len != 0 and token[0] == '-' and std.math.isNan(number)) number = -number;
                break :blk .{ .float = number };
            },
            else => error.ParseValue,
        },
        .@"enum" => {
            if (v != .string) return error.ParseValue;
            if (std.meta.stringToEnum(T, v.string) == null) return error.ParseValue;
            return v;
        },
        .optional => |o| return if (v == .null) v else validateMode(o.child, arena, v, xml_lax),
        .array => |a| {
            const python = @import("python_value.zig");
            const items = if (v == .array) v.array.items else blk: {
                const builtin = python.view(v) orelse return error.ParseValue;
                break :blk switch (builtin) {
                    .tuple, .set, .frozen_set => |items| items,
                    else => return error.ParseValue,
                };
            };
            if (items.len != a.len) return error.ParseValue;
            const normalized = try arena.alloc(Value, a.len);
            errdefer arena.free(normalized);
            for (items, normalized) |item, *target|
                target.* = try validateMode(a.child, arena, item, xml_lax);
            return python.tupleFromOwnedSlice(arena, normalized);
        },
        .pointer => |p| {
            const items = if (v == .array) v.array.items else blk: {
                if (v != .custom) return error.ParseValue;
                if (v.custom.vtable.iterate) |iterate| {
                    const items = try iterate(arena, v.custom.ptr);
                    for (items) |*item| item.* = try validateMode(p.child, arena, item.*, xml_lax);
                    return .{ .array = .fromOwnedSlice(arena, items) };
                }
                const inspect = v.custom.vtable.python orelse return error.ParseValue;
                break :blk switch (inspect(v.custom.ptr)) {
                    .tuple, .set, .frozen_set => |items| items,
                    else => return error.ParseValue,
                };
            };
            var out: @import("runtime_value.zig").Array = .init(arena);
            try out.ensureTotalCapacity(items.len);
            for (items) |item| out.appendAssumeCapacity(try validateMode(p.child, arena, item, xml_lax));
            return .{ .array = out };
        },
        .@"struct" => |s| {
            if (v != .object) return error.ParseValue;
            var out: @import("runtime_value.zig").ObjectMap = .empty;
            inline for (s.field_names, s.field_types, 0..) |name, FT, i| {
                if (v.object.get(name)) |fv| {
                    try out.put(arena, name, try validateMode(FT, arena, fv, xml_lax));
                } else if (comptime hasDefault(T, i)) {
                    try out.put(arena, name, .null);
                } else return error.ParseValue;
            }
            return .{ .object = out };
        },
        else => @compileError("dszig: unsupported nested type " ++ @typeName(T)),
    }
}

fn stringKeyedObject(arena: Allocator, input: Value) Error!Value {
    if (input == .object) return input;
    if (input != .custom) return error.ParseValue;
    const inspect = input.custom.vtable.python orelse return error.ParseValue;
    const entries = switch (inspect(input.custom.ptr)) {
        .dictionary => |entries| entries,
        else => return error.ParseValue,
    };
    var result: @import("runtime_value.zig").ObjectMap = .empty;
    try result.ensureTotalCapacity(arena, entries.len);
    for (entries) |entry| {
        const key = try validateMode([]const u8, arena, entry.key, false);
        try result.put(arena, key.string, entry.value);
    }
    return .{ .object = result };
}

fn boolText(s: []const u8) Error!Value {
    const yes = [_][]const u8{ "1", "on", "t", "true", "y", "yes" };
    const no = [_][]const u8{ "0", "off", "f", "false", "n", "no" };
    for (yes) |w| if (std.ascii.eqlIgnoreCase(s, w)) return .{ .bool = true };
    for (no) |w| if (std.ascii.eqlIgnoreCase(s, w)) return .{ .bool = false };
    return error.ParseValue;
}

/// XML's decoded object/list path passes strict=False to TypeAdapter.
pub fn XmlValidator(comptime T: type) *const fn (Allocator, Value) Error!Value {
    return struct {
        fn run(arena: Allocator, v: Value) Error!Value {
            return validateMode(T, arena, v, true);
        }
    }.run;
}

pub fn Validator(comptime T: type) *const fn (Allocator, Value) Error!Value {
    return struct {
        fn run(arena: Allocator, v: Value) Error!Value {
            return validate(T, arena, v);
        }
    }.run;
}
