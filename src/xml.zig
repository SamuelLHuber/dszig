//! DSPy 3.5.0 XMLAdapter. Tags delimit typed fields; scalar bodies are
//! literal text, deliberately not XML entity-decoded or parsed as markup.
const std = @import("std");
const adapter = @import("adapter.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const pyjson = @import("pyjson.zig");
const unicode = @import("xml_unicode.zig");
const fieldSchema = adapter.fieldSchema;
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Field = signature.Field;
const Record = value.Record;
const Error = adapter.Error;
const Writer = std.Io.Writer;
const max_depth = pyjson.max_depth;
const empty_schema: Value = .{ .object = .empty };
const reasoning_schema = @import("types.zig").schemaXml(signature.Reasoning);

fn get(node: Value, key: []const u8) Value {
    return if (node == .object) node.object.get(key) orelse .null else .null;
}
fn string(node: Value) []const u8 {
    return if (node == .string) node.string else "";
}
fn isType(node: Value, name: []const u8) bool {
    return std.mem.eql(u8, string(get(node, "type")), name);
}
fn choices(node: Value) Value {
    const any = get(node, "anyOf");
    return if (any == .array) any else get(node, "oneOf");
}
fn resolve(node: Value, definitions: Value) Value {
    const ref = string(get(node, "$ref"));
    if (ref.len == 0) return node;
    const slash = std.mem.lastIndexOfScalar(u8, ref, '/') orelse return node;
    const found = get(definitions, ref[slash + 1 ..]);
    return if (found == .object) found else node;
}
fn writeError(err: pyjson.Error) Error {
    return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
        error.ParseValue => error.AdapterParse,
    };
}

fn usesNested(field: Field, schema: Value) bool {
    if (field.xml_literal) return false;
    switch (field.kind) {
        .str_list, .json_object, .str_dict => return true,
        .json => {},
        else => return false,
    }
    var node = schema;
    const branches = choices(node);
    if (branches == .array) {
        var nonnull: ?Value = null;
        for (branches.array.items) |branch| {
            if (isType(branch, "null")) continue;
            if (nonnull != null) return false;
            nonnull = branch;
        }
        node = nonnull orelse return false;
    }
    node = resolve(node, get(schema, "$defs"));
    return isType(node, "array") or isType(node, "object");
}

const validName = unicode.identifier;

fn writeAttribute(w: *Writer, key: []const u8) Writer.Error!void {
    const quote: u8 = if (std.mem.indexOfScalar(u8, key, '"') != null and std.mem.indexOfScalar(u8, key, '\'') == null) '\'' else '"';
    try w.writeByte(quote);
    for (key) |c| switch (c) {
        '&' => try w.writeAll("&amp;"),
        '<' => try w.writeAll("&lt;"),
        '>' => try w.writeAll("&gt;"),
        '\n' => try w.writeAll("&#10;"),
        '\r' => try w.writeAll("&#13;"),
        '\t' => try w.writeAll("&#9;"),
        '"' => if (quote == '"') try w.writeAll("&quot;") else try w.writeByte(c),
        else => try w.writeByte(c),
    };
    try w.writeByte(quote);
}

fn writeXml(w: *Writer, v: Value, tag: []const u8, key: ?[]const u8, depth: usize) pyjson.Error!void {
    if (depth >= max_depth) return error.JsonTooDeep;
    try w.print("<{s}", .{tag});
    if (key) |k| {
        try w.writeAll(" key=");
        try writeAttribute(w, k);
    }
    if (v == .null) return w.writeAll(" />");
    try w.writeAll(">\n");
    switch (v) {
        .array => |items| for (items.items, 0..) |item, i| {
            if (i != 0) try w.writeByte('\n');
            try writeXml(w, item, "item", null, depth + 1);
        },
        .object => |object| {
            var it = object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try w.writeByte('\n');
                first = false;
                const name = entry.key_ptr.*;
                try writeXml(w, entry.value_ptr.*, if (validName(name)) name else "entry", if (validName(name)) null else name, depth + 1);
            }
        },
        else => try pyjson.writeStr(w, v),
    }
    try w.print("\n</{s}>", .{tag});
}

fn example(arena: Allocator, node: Value, definitions: Value, seen: []const []const u8, depth: usize) Error!Value {
    if (depth >= max_depth) return error.JsonTooDeep;
    const ref = string(get(node, "$ref"));
    if (ref.len > 0) {
        for (seen) |prior| if (std.mem.eql(u8, ref, prior)) return .{ .string = "..." };
        var stack: [max_depth][]const u8 = undefined;
        @memcpy(stack[0..seen.len], seen);
        stack[seen.len] = ref;
        return example(arena, resolve(node, definitions), definitions, stack[0 .. seen.len + 1], depth + 1);
    }
    const branches = choices(node);
    if (branches == .array) {
        for (branches.array.items) |branch| if (!isType(branch, "null")) return example(arena, branch, definitions, seen, depth + 1);
        return .{ .string = "..." };
    }
    if (node == .object and node.object.contains("const")) return get(node, "const");
    const members = get(node, "enum");
    if (members == .array) {
        var out: Writer.Allocating = .init(arena);
        out.writer.writeAll("one of: ") catch return error.OutOfMemory;
        for (members.array.items, 0..) |member, i| {
            if (i > 0) out.writer.writeAll("; ") catch return error.OutOfMemory;
            pyjson.writeStr(&out.writer, member) catch |err| return writeError(err);
        }
        return .{ .string = out.written() };
    }
    if (isType(node, "array")) {
        var items: @import("runtime_value.zig").Array = .init(arena);
        const prefix = get(node, "prefixItems");
        if (prefix == .array) {
            for (prefix.array.items) |child| try items.append(try example(arena, child, definitions, seen, depth + 1));
        } else try items.append(try example(arena, get(node, "items"), definitions, seen, depth + 1));
        return .{ .array = items };
    }
    const properties = get(node, "properties");
    const additional = get(node, "additionalProperties");
    var object: Record = .empty;
    if ((properties != .object or properties.object.count() == 0) and additional == .object) {
        try object.put(arena, "example key", try example(arena, additional, definitions, seen, depth + 1));
    } else if (properties == .object) {
        var it = properties.object.iterator();
        while (it.next()) |entry| try object.put(arena, entry.key_ptr.*, try example(arena, entry.value_ptr.*, definitions, seen, depth + 1));
    }
    return if (object.count() > 0) .{ .object = object } else .{ .string = "..." };
}

fn primitiveSchema(node: Value) bool {
    if (get(node, "enum") == .array or (node == .object and node.object.contains("const"))) return false;
    if (choices(node) == .array) {
        for (choices(node).array.items) |branch| if (!primitiveSchema(branch)) return false;
        return true;
    }
    return isType(node, "string") or isType(node, "integer") or isType(node, "number") or isType(node, "boolean") or isType(node, "null");
}

fn scanRequirements(node: Value, nullable: *bool, mapping: *bool, depth: usize) Error!void {
    if (depth >= max_depth) return error.JsonTooDeep;
    switch (node) {
        .object => |object| {
            nullable.* = nullable.* or isType(node, "null");
            const additional = get(node, "additionalProperties");
            mapping.* = mapping.* or additional == .object or (additional == .bool and additional.bool);
            for (object.values()) |child| try scanRequirements(child, nullable, mapping, depth + 1);
        },
        .array => |items| for (items.items) |child| try scanRequirements(child, nullable, mapping, depth + 1),
        else => {},
    }
}

pub const XmlAdapter = struct {
    pub fn fieldStructure(arena: Allocator, definitions: []const Field) Error![]const u8 {
        if (definitions.len == 0) return "";
        var out: Writer.Allocating = .init(arena);
        const w = &out.writer;
        w.writeAll("Output format (replace placeholders):\n") catch return error.OutOfMemory;
        for (definitions, 0..) |field, i| {
            if (i > 0) w.writeAll("\n\n") catch return error.OutOfMemory;
            const schema = try fieldSchema(arena, field);
            if (usesNested(field, schema)) {
                const v = try example(arena, schema, get(schema, "$defs"), &.{}, 0);
                writeXml(w, v, field.name, null, 0) catch |err| return writeError(err);
            } else {
                w.print("<{s}>\n", .{field.name}) catch return error.OutOfMemory;
                if (primitiveSchema(schema) and field.kind != .literal) {
                    w.print("{{{s}}}", .{field.name}) catch return error.OutOfMemory;
                } else adapter.translateFieldType(w, field, false) catch return error.OutOfMemory;
                w.print("\n</{s}>", .{field.name}) catch return error.OutOfMemory;
            }
        }
        return out.written();
    }

    pub fn requirements(arena: Allocator, definitions: []const Field) Error![]const u8 {
        if (definitions.len == 0) return "";
        var out: Writer.Allocating = .init(arena);
        const w = &out.writer;
        w.writeAll(try fieldStructure(arena, definitions)) catch return error.OutOfMemory;
        w.writeAll("\n\nUse this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.") catch return error.OutOfMemory;
        var nullable = false;
        var mapping = false;
        for (definitions) |field| {
            const schema = try fieldSchema(arena, field);
            if (usesNested(field, schema)) {
                try scanRequirements(schema, &nullable, &mapping, 0);
            } else {
                nullable = nullable or allowsNull(schema);
            }
        }
        if (nullable) w.writeAll(" For null, self-close the field's own tag. For an empty string, use an opening and closing tag with no content.") catch return error.OutOfMemory;
        if (mapping) w.writeAll(" Dictionary keys: <entry key=\"...\">; escape key attributes only.") catch return error.OutOfMemory;
        return out.written();
    }

    pub fn fields(arena: Allocator, definitions: []const Field, record: *const Record, output: bool, missing: ?[]const u8) Error![]const u8 {
        var out: Writer.Allocating = .init(arena);
        const w = &out.writer;
        var first = true;
        for (definitions) |field| {
            const v = record.get(field.name) orelse if (missing) |text| Value{ .string = text } else continue;
            if (!first) w.writeAll("\n\n") catch return error.OutOfMemory;
            first = false;
            if (output and (v == .null or ((v == .object or v == .array) and usesNested(field, try fieldSchema(arena, field))))) {
                const serialized = try adapter.formattedValue(arena, field, v);
                writeXml(w, serialized, field.name, null, 0) catch |err| return writeError(err);
            } else {
                w.print("<{s}>\n", .{field.name}) catch return error.OutOfMemory;
                try adapter.writeFieldValue(arena, w, field, v);
                w.print("\n</{s}>", .{field.name}) catch return error.OutOfMemory;
            }
        }
        return unicode.strip(out.written());
    }

    pub fn user(arena: Allocator, view: adapter.View, inputs: *const Record, prefix: []const u8, main_request: bool) Error![]const u8 {
        const body = try fields(arena, view.inputs, inputs, false, null);
        return unicode.strip(try std.mem.join(arena, "\n\n", &.{ prefix, body, if (main_request) try requirements(arena, view.outputs) else "" }));
    }

    pub fn parse(arena: Allocator, view: adapter.View, completion: []const u8) Error!Record {
        var schemas: Record = .empty;
        for (view.outputs) |field| {
            // Unlike JSONAdapter's provider schema, XML parsing uses the
            // actual Reasoning model schema, including its required content.
            const schema = if (field.reasoning_type and field.kind == .str)
                std.json.parseFromSliceLeaky(Value, arena, reasoning_schema, .{}) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return error.AdapterParse,
                }
            else
                try fieldSchema(arena, field);
            try schemas.put(arena, field.name, schema);
        }
        var root_schema: Record = .empty;
        try root_schema.put(arena, "properties", .{ .object = schemas });
        const text = try stripPrefix(view, completion);
        var parser: Parser = .{ .arena = arena, .text = text };
        const scope: Scope = .{ .schema = .{ .object = root_schema }, .definitions = empty_schema };
        const nodes = try parser.children(&scope, 0);
        for (nodes) |node| if (view.findOutput(node.name) == null) return error.AdapterParse;
        if (std.mem.indexOfScalar(u8, text[parser.pos..], '<') != null) return error.AdapterParse;
        var result: Record = .empty;
        for (view.outputs) |field| {
            const schema = schemas.get(field.name).?;
            const group = try groupNamed(arena, nodes, field.name);
            if (group.len == 0) {
                const fallback = (try signature.outputDefault(arena, field)) orelse return error.AdapterParse;
                try result.put(arena, field.name, fallback);
                continue;
            }
            const decoded = try elementsValue(arena, group, schema, get(schema, "$defs"), 0);
            var parsing_field = field;
            if ((decoded == .object or decoded == .array) and group[0].children.len > 0) {
                if (field.xml_validate) |validate| parsing_field.validate = validate;
            }
            const parsed = try adapter.parseFieldValue(arena, parsing_field, decoded);
            try result.put(arena, field.name, parsed);
        }
        return result;
    }
};

fn allowsNull(schema: Value) bool {
    if (isType(schema, "null")) return true;
    const branches = get(schema, "anyOf");
    if (branches == .array) for (branches.array.items) |branch| if (isType(branch, "null")) return true;
    return false;
}

const Node = struct {
    name: []const u8,
    key: []const u8,
    text: ?[]const u8 = null,
    children: []const Node = &.{},
};

fn groupNamed(arena: Allocator, nodes: []const Node, name: []const u8) Error![]const Node {
    var count: usize = 0;
    for (nodes) |node| if (std.mem.eql(u8, node.key, name)) {
        count += 1;
    };
    if (count == 0) return &.{};
    if (count == 1) for (nodes, 0..) |node, i| if (std.mem.eql(u8, node.key, name)) return nodes[i .. i + 1];
    const result = try arena.alloc(Node, count);
    var i: usize = 0;
    for (nodes) |node| if (std.mem.eql(u8, node.key, name)) {
        result[i] = node;
        i += 1;
    };
    return result;
}

fn literalValue(arena: Allocator, text: []const u8, schema: Value) Error!Value {
    const members = get(schema, "enum");
    const constant = get(schema, "const");
    const cleaned = unicode.strip(text);
    var unwrapped = cleaned;
    if ((std.mem.startsWith(u8, unwrapped, "Literal[") or std.mem.startsWith(u8, unwrapped, "str[")) and std.mem.endsWith(u8, unwrapped, "]")) {
        unwrapped = unwrapped[(std.mem.indexOfScalar(u8, unwrapped, '[').? + 1) .. unwrapped.len - 1];
    }
    if (unwrapped.len > 1 and (unwrapped[0] == '\'' or unwrapped[0] == '"') and unwrapped[unwrapped.len - 1] == unwrapped[0]) unwrapped = unwrapped[1 .. unwrapped.len - 1];
    const candidates: []const Value = if (members == .array) members.array.items else &.{constant};
    for (candidates) |member| if (member == .string and std.mem.eql(u8, member.string, unwrapped)) return member;
    const decoded = std.json.parseFromSliceLeaky(Value, arena, unwrapped, .{}) catch return .{ .string = text };
    for (candidates) |member| {
        if (member == .integer and decoded == .integer and member.integer == decoded.integer) return member;
        if (member == .bool and decoded == .bool and member.bool == decoded.bool) return member;
        if (member == .null and decoded == .null) return member;
    }
    return .{ .string = text };
}

fn elementsValue(arena: Allocator, nodes: []const Node, input_schema: Value, definitions: Value, depth: usize) Error!Value {
    if (depth >= max_depth) return error.JsonTooDeep;
    var schema = resolve(input_schema, definitions);
    if (nodes.len == 1 and allowsNull(schema) and nodes[0].children.len == 0 and nodes[0].text == null) return .null;
    var coerce_literal = true;
    const branches = choices(schema);
    if (branches == .array) {
        const discriminator = get(schema, "discriminator");
        const name = string(get(discriminator, "propertyName"));
        const tags = try groupNamed(arena, nodes[0].children, name);
        if (tags.len == 1) {
            const ref = get(get(discriminator, "mapping"), unicode.strip(tags[0].text orelse ""));
            if (ref == .string) {
                var reference: Record = .empty;
                try reference.put(arena, "$ref", ref);
                return elementsValue(arena, nodes, .{ .object = reference }, definitions, depth + 1);
            }
        }
        var count: usize = 0;
        var selected: ?Value = null;
        for (branches.array.items) |branch| {
            if (isType(branch, "null")) continue;
            count += 1;
            if (selected == null) selected = branch;
        }
        coerce_literal = count == 1;
        schema = selected orelse return error.AdapterParse;
        if (nodes[0].children.len > 0) for (branches.array.items) |branch| {
            if (!isType(branch, "null") and !isType(branch, "string")) {
                schema = branch;
                break;
            }
        };
    }
    if (nodes.len > 1 and !isType(schema, "array") and get(schema, "type") != .null) return error.AdapterParse;
    if (isType(schema, "array")) {
        var items = nodes;
        if (nodes.len == 1 and nodes[0].children.len == 0) {
            const text = unicode.strip(nodes[0].text orelse "");
            if (text.len == 0) return .{ .array = .init(arena) };
            if (text[0] == '[') return .{ .string = text };
        }
        if (nodes.len == 1) {
            const group = try groupNamed(arena, nodes[0].children, "item");
            if (group.len > 0) items = group;
        }
        const prefix = get(schema, "prefixItems");
        var out: @import("runtime_value.zig").Array = .init(arena);
        for (items, 0..) |item, i| {
            const child = if (prefix == .array and i < prefix.array.items.len) prefix.array.items[i] else get(schema, "items");
            try out.append(try elementsValue(arena, &.{item}, child, definitions, depth + 1));
        }
        return .{ .array = out };
    }
    const first = nodes[0];
    if (get(schema, "type") != .null and !isType(schema, "object") and first.children.len > 0) return error.AdapterParse;
    if (first.children.len == 0) {
        if (isType(schema, "object") and unicode.strip(first.text orelse "").len == 0) return .{ .object = .empty };
        if (nodes.len == 1) {
            const text = first.text orelse "";
            if (coerce_literal and (get(schema, "enum") == .array or (schema == .object and schema.object.contains("const")))) return literalValue(arena, text, schema);
            return .{ .string = text };
        }
        var values: @import("runtime_value.zig").Array = .init(arena);
        for (nodes) |node| try values.append(.{ .string = node.text orelse "" });
        return .{ .array = values };
    }
    var result: Record = .empty;
    const properties = get(schema, "properties");
    const additional = get(schema, "additionalProperties");
    for (first.children) |node| {
        if (result.contains(node.key)) continue;
        const group = try groupNamed(arena, first.children, node.key);
        const explicit = get(properties, node.key);
        const child = if (explicit == .object) explicit else if (additional == .object) additional else empty_schema;
        try result.put(arena, node.key, try elementsValue(arena, group, child, definitions, depth + 1));
    }
    return .{ .object = result };
}

/// Flatten schema alternatives without heap allocation. The same depth bound as
/// JSON rendering bounds recursive schemas and adversarial union expansions.
const SchemaIterator = struct {
    stack: [max_depth]Value = undefined,
    len: usize = 0,
    definitions: Value,
    fn init(schema: Value, definitions: Value) SchemaIterator {
        var it: SchemaIterator = .{ .definitions = definitions };
        it.stack[0] = schema;
        it.len = 1;
        return it;
    }
    fn next(it: *SchemaIterator) Error!?Value {
        var depth: usize = 0;
        while (it.len > 0) {
            if (depth >= max_depth) return error.JsonTooDeep;
            depth += 1;
            it.len -= 1;
            const node = resolve(it.stack[it.len], it.definitions);
            const branches = choices(node);
            if (branches == .array) {
                if (it.len + branches.array.items.len > max_depth) return error.JsonTooDeep;
                var i = branches.array.items.len;
                while (i > 0) {
                    i -= 1;
                    it.stack[it.len] = branches.array.items[i];
                    it.len += 1;
                }
            } else return if (node == .object) node else empty_schema;
        }
        return null;
    }
};

const SchemaSet = struct {
    items: [max_depth]Value = undefined,
    len: usize = 0,

    fn append(set: *SchemaSet, schema: Value) Error!void {
        if (set.len == max_depth) return error.JsonTooDeep;
        set.items[set.len] = schema;
        set.len += 1;
    }
};

fn childSchemas(schema: Value, definitions: Value, name: []const u8, index: usize, result: *SchemaSet, depth: usize) Error!void {
    if (depth >= max_depth) return error.JsonTooDeep;
    var it = SchemaIterator.init(schema, definitions);
    while (try it.next()) |node| {
        if (isType(node, "array")) {
            const prefix = get(node, "prefixItems");
            const item = if (std.mem.eql(u8, name, "item") and prefix == .array and index < prefix.array.items.len) prefix.array.items[index] else get(node, "items");
            if (std.mem.eql(u8, name, "item")) {
                try result.append(if (item == .object) item else empty_schema);
            } else try childSchemas(item, definitions, name, index, result, depth + 1);
            continue;
        }
        if (get(node, "type") != .null and !isType(node, "object")) continue;
        const properties = get(node, "properties");
        const explicit = get(properties, name);
        if (explicit == .object) {
            try result.append(explicit);
            continue;
        }
        const additional = get(node, "additionalProperties");
        if (additional == .bool and !additional.bool) continue;
        if (additional == .null and properties == .object) continue;
        try result.append(if (additional == .object) additional else empty_schema);
    }
}

const Scope = struct {
    name: ?[]const u8 = null,
    schema: Value,
    definitions: Value,
    parent: ?*const Scope = null,
};
const Opening = struct { name: []const u8, key: []const u8, end: usize, self_closing: bool };

const whitespace = unicode.whitespaceEnd;
fn newlineEnd(text: []const u8, start: usize) ?usize {
    var pos = start;
    while (pos < text.len and (text[pos] == ' ' or text[pos] == '\t')) pos += 1;
    if (pos < text.len and text[pos] == '\r') pos += 1;
    return if (pos < text.len and text[pos] == '\n') pos + 1 else null;
}
fn closeEnd(text: []const u8, start: usize, name: []const u8, multiline: bool) ?usize {
    if (!std.mem.startsWith(u8, text[start..], "</")) return null;
    var pos = start + 2;
    if (!std.mem.startsWith(u8, text[pos..], name)) return null;
    pos += name.len;
    if (multiline) {
        while (pos < text.len and (text[pos] == ' ' or text[pos] == '\t')) pos += 1;
    } else pos = whitespace(text, pos);
    if (pos >= text.len or text[pos] != '>') return null;
    pos += 1;
    if (!multiline) return pos;
    while (pos < text.len and (text[pos] == ' ' or text[pos] == '\t')) pos += 1;
    if (pos < text.len and text[pos] == '\r') pos += 1;
    if (pos == text.len) return pos;
    return if (text[pos] == '\n') pos + 1 else null;
}

fn decodeAttribute(arena: Allocator, raw: []const u8) Error![]const u8 {
    const utf8 = std.unicode.Utf8View.init(raw) catch return error.AdapterParse;
    var codepoints = utf8.iterator();
    while (codepoints.nextCodepoint()) |cp| {
        if (!(cp == 9 or cp == 10 or cp == 13 or (cp >= 0x20 and cp <= 0xd7ff) or
            (cp >= 0xe000 and cp <= 0xfffd) or (cp >= 0x10000 and cp <= 0x10ffff))) return error.AdapterParse;
    }
    if (std.mem.indexOfAny(u8, raw, "&<\r\n\t") == null) return raw;
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (pos < raw.len) : (pos += 1) {
        switch (raw[pos]) {
            '<' => return error.AdapterParse,
            '\r', '\n', '\t' => {
                try out.append(arena, ' ');
                if (raw[pos] == '\r' and pos + 1 < raw.len and raw[pos + 1] == '\n') pos += 1;
            },
            '&' => {
                const end = std.mem.indexOfScalarPos(u8, raw, pos, ';') orelse return error.AdapterParse;
                const entity = raw[pos + 1 .. end];
                const names = [_][]const u8{ "amp", "lt", "gt", "quot", "apos" };
                const chars = "&<>\"'";
                var found = false;
                for (names, 0..) |name, i| if (std.mem.eql(u8, entity, name)) {
                    try out.append(arena, chars[i]);
                    found = true;
                    break;
                };
                if (!found) {
                    if (entity.len < 2 or entity[0] != '#') return error.AdapterParse;
                    const hex = entity.len > 2 and entity[1] == 'x';
                    const codepoint = std.fmt.parseInt(u21, entity[if (hex) 2 else 1..], if (hex) 16 else 10) catch return error.AdapterParse;
                    if (!(codepoint == 9 or codepoint == 10 or codepoint == 13 or (codepoint >= 0x20 and codepoint <= 0xd7ff) or (codepoint >= 0xe000 and codepoint <= 0xfffd) or codepoint >= 0x10000)) return error.AdapterParse;
                    var buffer: [4]u8 = undefined;
                    const count = std.unicode.utf8Encode(codepoint, &buffer) catch return error.AdapterParse;
                    try out.appendSlice(arena, buffer[0..count]);
                }
                pos = end;
            },
            else => try out.append(arena, raw[pos]),
        }
    }
    return out.items;
}

const Parser = struct {
    arena: Allocator,
    text: []const u8,
    pos: usize = 0,

    fn opening(parser: *const Parser, start: usize, decode_key: bool) Error!?Opening {
        const text = parser.text;
        if (start >= text.len or text[start] != '<') return null;
        const name_start = start + 1;
        var pos = unicode.tagNameEnd(text, name_start);
        if (pos == name_start) return null;
        const name = text[name_start..pos];
        if (decode_key and !unicode.xmlName(name)) return null;
        const after_name = pos;
        pos = whitespace(text, pos);
        var key = name;
        if (pos > after_name and std.mem.startsWith(u8, text[pos..], "key=")) {
            pos += 4;
            if (pos >= text.len or (text[pos] != '\'' and text[pos] != '"')) return null;
            const quote = text[pos];
            pos += 1;
            const end = std.mem.indexOfScalarPos(u8, text, pos, quote) orelse return null;
            const decoded = if (decode_key) try decodeAttribute(parser.arena, text[pos..end]) else text[pos..end];
            if (std.mem.eql(u8, name, "entry")) key = decoded;
            pos = whitespace(text, end + 1);
        }
        var self_closing = false;
        if (pos < text.len and text[pos] == '/') {
            self_closing = true;
            pos += 1;
        }
        if (pos >= text.len or text[pos] != '>') return null;
        return .{ .name = name, .key = key, .end = pos + 1, .self_closing = self_closing };
    }

    fn children(parser: *Parser, scope: *const Scope, depth: usize) Error![]const Node {
        if (depth >= max_depth) return error.JsonTooDeep;
        var nodes: std.ArrayList(Node) = .empty;
        while (true) {
            parser.pos = whitespace(parser.text, parser.pos);
            if (parser.pos == parser.text.len or std.mem.startsWith(u8, parser.text[parser.pos..], "</")) return nodes.items;
            if (scope.parent == null and parser.text[parser.pos] != '<') return nodes.items;
            const opening_tag = (try parser.opening(parser.pos, true)) orelse return error.AdapterParse;
            var schemas: SchemaSet = .{};
            try childSchemas(scope.schema, scope.definitions, opening_tag.key, nodes.items.len, &schemas, 0);
            if (schemas.len == 0) return error.AdapterParse;
            var child_schema = schemas.items[0];
            if (schemas.len > 1) {
                var object: Record = .empty;
                var branches: @import("runtime_value.zig").Array = .init(parser.arena);
                try branches.appendSlice(schemas.items[0..schemas.len]);
                try object.put(parser.arena, "anyOf", .{ .array = branches });
                child_schema = .{ .object = object };
            }
            var definitions = scope.definitions;
            const own_definitions = get(child_schema, "$defs");
            if (own_definitions == .object) {
                var merged: Record = if (definitions == .object) try definitions.object.clone(parser.arena) else .empty;
                var it = own_definitions.object.iterator();
                while (it.next()) |entry| try merged.put(parser.arena, entry.key_ptr.*, entry.value_ptr.*);
                definitions = .{ .object = merged };
            }
            var node: Node = .{ .name = opening_tag.name, .key = opening_tag.key };
            parser.pos = opening_tag.end;
            if (!opening_tag.self_closing) {
                var structured = false;
                var has_string = false;
                var it = SchemaIterator.init(child_schema, definitions);
                while (try it.next()) |choice| {
                    structured = structured or isType(choice, "object") or isType(choice, "array") or (choice == .object and choice.object.count() == 0);
                    has_string = has_string or isType(choice, "string");
                }
                const nested_start = whitespace(parser.text, parser.pos);
                var nested = structured and nested_start < parser.text.len and parser.text[nested_start] == '<';
                if (structured and has_string and newlineEnd(parser.text, parser.pos) != null) {
                    nested = false;
                    if (nested_start < parser.text.len and parser.text[nested_start] == '<') {
                        const tag_end = std.mem.indexOfScalarPos(u8, parser.text, nested_start, '>');
                        if (tag_end) |end| {
                            if (std.mem.indexOfScalar(u8, parser.text[nested_start..end], '\n') == null) {
                                var after = end + 1;
                                while (after < parser.text.len and (parser.text[after] == ' ' or parser.text[after] == '\t')) after += 1;
                                nested = after == parser.text.len or newlineEnd(parser.text, after) != null;
                            }
                        }
                    }
                }
                if (nested) {
                    parser.pos = nested_start;
                    const child_scope: Scope = .{ .name = node.name, .schema = child_schema, .definitions = definitions, .parent = scope };
                    node.children = try parser.children(&child_scope, depth + 1);
                    node.text = "";
                    parser.pos = closeEnd(parser.text, parser.pos, node.name, false) orelse return error.AdapterParse;
                } else node.text = try parser.scalar(node.name, scope);
            }
            try nodes.append(parser.arena, node);
        }
    }

    fn siblingAllowed(parser: *const Parser, scope: *const Scope, name: []const u8) Error!bool {
        var it = SchemaIterator.init(scope.schema, scope.definitions);
        while (try it.next()) |node| {
            if (isType(node, "array")) {
                if (std.mem.eql(u8, name, "item")) return true;
                const prefix = get(node, "prefixItems");
                if (prefix == .array) {
                    for (prefix.array.items) |item| if (try parser.objectSiblingAllowed(item, scope.definitions, name)) return true;
                } else if (try parser.objectSiblingAllowed(get(node, "items"), scope.definitions, name)) return true;
            } else if (try parser.objectSiblingAllowed(node, scope.definitions, name)) return true;
        }
        return false;
    }

    fn objectSiblingAllowed(_: *const Parser, schema: Value, definitions: Value, name: []const u8) Error!bool {
        var it = SchemaIterator.init(schema, definitions);
        while (try it.next()) |node| {
            const properties = get(node, "properties");
            if (get(properties, name) != .null) return true;
            if (std.mem.eql(u8, name, "entry") and properties == .object) for (properties.object.keys()) |key| if (!validName(key)) return true;
            if (node.object.count() == 0) return true;
            if (isType(node, "object")) {
                const additional = get(node, "additionalProperties");
                if (additional == .object or (additional == .bool and additional.bool) or (additional == .null and properties != .object)) return true;
            }
        }
        return false;
    }

    fn boundary(parser: *const Parser, scope: *const Scope, start: usize, depth: usize) Error!bool {
        if (depth >= max_depth) return error.JsonTooDeep;
        const pos = whitespace(parser.text, start);
        if (try parser.opening(pos, false)) |tag| if (try parser.siblingAllowed(scope, tag.name)) return true;
        if (scope.name) |name| {
            if (closeEnd(parser.text, pos, name, false)) |end| return parser.boundary(scope.parent.?, end, depth + 1);
            return false;
        }
        return std.mem.indexOfScalar(u8, parser.text[pos..], '<') == null;
    }

    fn scalar(parser: *Parser, name: []const u8, scope: *const Scope) Error![]const u8 {
        const multiline_start = newlineEnd(parser.text, parser.pos);
        const multiline = multiline_start != null;
        const start = multiline_start orelse parser.pos;
        var scan = start;
        while (std.mem.indexOfScalarPos(u8, parser.text, scan, '<')) |candidate| {
            scan = candidate + 1;
            if (scope.name) |enclosing| {
                if (closeEnd(parser.text, candidate, enclosing, false)) |end| {
                    if (try parser.boundary(scope.parent.?, end, 0)) return error.AdapterParse;
                }
            }
            const end = closeEnd(parser.text, candidate, name, multiline) orelse continue;
            if (!try parser.boundary(scope, end, 0)) continue;
            var value_end = candidate;
            if (multiline) {
                const line_start = if (std.mem.lastIndexOfScalar(u8, parser.text[start..candidate], '\n')) |last| start + last + 1 else start;
                if (std.mem.trim(u8, parser.text[line_start..candidate], " \t").len == 0) value_end = line_start;
                if (value_end > start and parser.text[value_end - 1] == '\n') value_end -= 1;
                if (value_end > start and parser.text[value_end - 1] == '\r') value_end -= 1;
            }
            parser.pos = end;
            return if (multiline) parser.text[start..value_end] else unicode.strip(parser.text[start..value_end]);
        }
        return error.AdapterParse;
    }
};

fn stripPrefix(view: adapter.View, completion: []const u8) Error![]const u8 {
    var text = unicode.strip(completion);
    var fenced = false;
    while (true) {
        text = unicode.trimStart(text);
        var thought: ?[]const u8 = null;
        for ([_][]const u8{ "think", "thinking" }) |name| {
            if (view.findOutput(name) != null) continue;
            const opening = if (std.mem.eql(u8, name, "think")) "<think>" else "<thinking>";
            if (std.mem.startsWith(u8, text, opening)) thought = name;
        }
        if (thought) |name| {
            const closing = if (std.mem.eql(u8, name, "think")) "</think>" else "</thinking>";
            const end = std.mem.indexOf(u8, text, closing) orelse return error.AdapterParse;
            text = text[end + closing.len ..];
            continue;
        }
        if (!fenced and std.mem.startsWith(u8, text, "```")) {
            var pos: usize = 3;
            if (std.mem.startsWith(u8, text[pos..], "xml")) pos += 3;
            if (newlineEnd(text, pos)) |end| {
                text = text[end..];
                fenced = true;
                continue;
            }
        }
        break;
    }
    if (fenced) {
        const fence = std.mem.lastIndexOf(u8, text, "\n```") orelse return error.AdapterParse;
        if (std.mem.indexOfScalar(u8, text[fence + 4 ..], '<') != null) return error.AdapterParse;
        text = unicode.strip(text[0..fence]);
    }
    if (text.len > 0 and text[0] != '<') {
        if (std.mem.indexOfScalar(u8, text, '<')) |first| text = text[first..];
    }
    return text;
}

const testing = std.testing;

test "XML primitive prompts use literal tags and exact output reminder" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "Answer the question.", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    var inputs: Record = .empty;
    try inputs.put(arena, "question", .{ .string = "A < B & C?" });
    const messages = try adapter.Adapter.xml.format(arena, view, &.{}, &inputs);
    try testing.expectEqual(@as(usize, 2), messages.len);
    try testing.expectEqualStrings(
        "Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\n" ++
            "Output format (replace placeholders):\n<answer>\n{answer}\n</answer>\n" ++
            "In adhering to this structure, your objective is: \n        Answer the question.",
        messages[0].content.text,
    );
    try testing.expectEqualStrings(
        "<question>\nA < B & C?\n</question>\n\nOutput format (replace placeholders):\n<answer>\n{answer}\n</answer>\n\n" ++
            "Use this format for text outputs. Put field tags on separate lines. Write string values literally, without XML or JSON escaping.",
        messages[1].content.text,
    );
    try testing.expectEqualStrings("XMLAdapter", adapter.Adapter.xml.className());
}

test "XML nested schema examples and dictionary attribute quoting match upstream" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Item = struct {
        pub const model_name = "Item";
        name: []const u8,
        labels: []const []const u8,
    };
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { item: Item, names: signature.StrDict };
    });
    try testing.expectEqualStrings(
        "Output format (replace placeholders):\n<item>\n<name>\n...\n</name>\n<labels>\n<item>\n...\n</item>\n</labels>\n</item>\n\n" ++
            "<names>\n<entry key=\"example key\">\n...\n</entry>\n</names>",
        try XmlAdapter.fieldStructure(arena, Sig.output_fields_list),
    );
    var object: Record = .empty;
    try object.put(arena, "a \"quoted\" & <key>\n", .{ .string = "raw <value> & text" });
    var outputs: Record = .empty;
    try outputs.put(arena, "names", .{ .object = object });
    try testing.expectEqualStrings(
        "<names>\n<entry key='a \"quoted\" &amp; &lt;key&gt;&#10;'>\nraw <value> & text\n</entry>\n</names>",
        try XmlAdapter.fields(arena, Sig.output_fields_list[1..], &outputs, true, null),
    );
}

test "XML dictionary names use Python Unicode identifiers not arbitrary UTF8" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { names: signature.StrDict };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    var names: Record = .empty;
    try names.put(arena, "é", .{ .string = "accent" });
    try names.put(arena, "e\u{301}", .{ .string = "combining" });
    try names.put(arena, "😀", .{ .string = "emoji" });
    try names.put(arena, "a-b.c", .{ .string = "punctuation" });
    var outputs: Record = .empty;
    try outputs.put(arena, "names", .{ .object = names });
    const rendered = try XmlAdapter.fields(arena, view.outputs, &outputs, true, null);
    try testing.expectEqualStrings(
        "<names>\n<é>\naccent\n</é>\n<e\u{301}>\ncombining\n</e\u{301}>\n<entry key=\"😀\">\nemoji\n</entry>\n<a-b.c>\npunctuation\n</a-b.c>\n</names>",
        rendered,
    );
    const parsed = try XmlAdapter.parse(arena, view, rendered);
    try testing.expectEqualStrings("emoji", parsed.get("names").?.object.get("😀").?.string);
    try testing.expectEqualStrings("combining", parsed.get("names").?.object.get("e\u{301}").?.string);
    try testing.expectError(error.AdapterParse, XmlAdapter.parse(arena, view, "<names><entry key=\"bad\x00key\">x</entry></names>"));
}

test "XML literal scalar leaves retain markup entities and inner closing tags" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const completion = "Some prose\n<answer>\n  <b>A &amp; B</b>\n</answer>\nthat close was literal\nlast\n  </answer>\nSign-off";
    const parsed = try XmlAdapter.parse(arena, view, completion);
    try testing.expectEqualStrings("  <b>A &amp; B</b>\n</answer>\nthat close was literal\nlast", parsed.get("answer").?.string);
    const compact = try XmlAdapter.parse(arena, view, "<answer>  compact <b>value</b>  </answer>");
    try testing.expectEqualStrings("compact <b>value</b>", compact.get("answer").?.string);
    try testing.expectError(error.AdapterParse, XmlAdapter.parse(arena, view, "<answer>\nvalue</answer> prose"));
}

test "runtime schema union validates XML chat and JSON without a native callback" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const fields = [_]Field{.{
        .name = "result",
        .kind = .json,
        .type_name = "Union[int, str]",
        .desc = "${result}",
        .prefix = "Result:",
        .schema = "{\"anyOf\":[{\"type\":\"integer\"},{\"type\":\"string\"}]}",
    }};
    const view: adapter.View = .{ .instructions = "", .inputs = &.{}, .outputs = &fields };
    const adapters = [_]adapter.Adapter{ .xml, .chat, .json };
    const numbers = [_][]const u8{ "<result>42</result>", "[[ ## result ## ]]\n42", "{\"result\":42}" };
    const strings = [_][]const u8{ "<result>hello</result>", "[[ ## result ## ]]\nhello", "{\"result\":\"hello\"}" };
    const invalid = [_][]const u8{ "<result>{}</result>", "[[ ## result ## ]]\n{}", "{\"result\":{}}" };
    for (adapters, numbers, strings, invalid) |which, number_text, string_text, invalid_text| {
        const number = try which.parse(arena, view, number_text);
        try testing.expectEqual(@as(i64, 42), number.get("result").?.integer);
        const literal = try which.parse(arena, view, string_text);
        try testing.expectEqualStrings("hello", literal.get("result").?.string);
        try testing.expectError(error.AdapterParse, which.parse(arena, view, invalid_text));
    }
}

test "runtime number unions preserve large integers and dictionary key schemas reject forbidden names" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const fields = [_]Field{.{
        .name = "result",
        .kind = .json,
        .type_name = "Union[float, int]",
        .desc = "${result}",
        .prefix = "Result:",
        .schema = "{\"anyOf\":[{\"type\":\"number\"},{\"type\":\"integer\"}]}",
    }};
    const view: adapter.View = .{ .instructions = "", .inputs = &.{}, .outputs = &fields };
    const adapters = [_]adapter.Adapter{ .xml, .chat, .json };
    const numbers = [_][]const u8{
        "<result>9007199254740993</result>",
        "[[ ## result ## ]]\n9007199254740993",
        "{\"result\":9007199254740993}",
    };
    for (adapters, numbers) |which, completion| {
        const parsed = try which.parse(arena, view, completion);
        try testing.expectEqual(@as(i64, 9007199254740993), parsed.get("result").?.integer);
    }
    var dictionary_fields = fields;
    dictionary_fields[0].type_name = "dict[Literal['allowed'], int]";
    dictionary_fields[0].schema = "{\"type\":\"object\",\"propertyNames\":{\"enum\":[\"allowed\"]},\"additionalProperties\":{\"type\":\"integer\"}}";
    const dictionary_view: adapter.View = .{ .instructions = "", .inputs = &.{}, .outputs = &dictionary_fields };
    const valid = [_][]const u8{
        "<result><allowed>7</allowed></result>",
        "[[ ## result ## ]]\n{\"allowed\":7}",
        "{\"result\":{\"allowed\":7}}",
    };
    const invalid = [_][]const u8{
        "<result><forbidden>7</forbidden></result>",
        "[[ ## result ## ]]\n{\"forbidden\":7}",
        "{\"result\":{\"forbidden\":7}}",
    };
    for (adapters, valid, invalid) |which, valid_text, invalid_text| {
        const parsed = try which.parse(arena, dictionary_view, valid_text);
        try testing.expectEqual(@as(i64, 7), parsed.get("result").?.object.get("allowed").?.integer);
        try testing.expectError(error.AdapterParse, which.parse(arena, dictionary_view, invalid_text));
    }
}

test "runtime integer schemas preserve Python sized integers including float first unions" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const schemas = [_][]const u8{
        "{\"type\":\"integer\"}",
        "{\"anyOf\":[{\"type\":\"number\"},{\"type\":\"integer\"}]}",
    };
    const integers = [_][]const u8{
        "18446744073709551615",
        "-18446744073709551616",
        "340282366920938463463374607431768211455",
    };
    const adapters = [_]adapter.Adapter{ .xml, .chat, .json };
    for (schemas) |schema| {
        const fields = [_]Field{.{
            .name = "result",
            .kind = .json,
            .type_name = "int",
            .desc = "${result}",
            .prefix = "Result:",
            .schema = schema,
        }};
        const view: adapter.View = .{ .instructions = "", .inputs = &.{}, .outputs = &fields };
        for (integers) |integer| {
            const completions = [_][]const u8{
                try std.fmt.allocPrint(arena, "<result>{s}</result>", .{integer}),
                try std.fmt.allocPrint(arena, "[[ ## result ## ]]\n{s}", .{integer}),
                try std.fmt.allocPrint(arena, "{{\"result\":{s}}}", .{integer}),
            };
            for (adapters, completions) |which, completion| {
                const parsed = try which.parse(arena, view, completion);
                try testing.expect(parsed.get("result").? == .number_string);
                try testing.expectEqualStrings(integer, parsed.get("result").?.number_string);
            }
        }
    }
}

test "XML Unicode framing whitespace is trimmed without stripping multiline content" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const compact = try XmlAdapter.parse(arena, view, "\u{a0}<answer>\u{2003}A\u{2003}</answer>\u{a0}");
    try testing.expectEqualStrings("A", compact.get("answer").?.string);
    const literal = try XmlAdapter.parse(arena, view, "<answer>\n\u{2003}A\u{2003}\n</answer>");
    try testing.expectEqualStrings("\u{2003}A\u{2003}", literal.get("answer").?.string);
}

test "XML nullable self closing empty string and omitted optional are distinct" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: ?[]const u8, count: ?i64 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const nulls = try XmlAdapter.parse(arena, view, "<answer /><count/>");
    try testing.expect(nulls.get("answer").? == .null);
    try testing.expect(nulls.get("count").? == .null);
    const empty = try XmlAdapter.parse(arena, view, "<answer></answer>");
    try testing.expectEqualStrings("", empty.get("answer").?.string);
    try testing.expect(empty.get("count").? == .null);
    const literal_null = try XmlAdapter.parse(arena, view, "<answer>null</answer><count>7</count>");
    try testing.expectEqualStrings("null", literal_null.get("answer").?.string);
    try testing.expectEqual(@as(i64, 7), literal_null.get("count").?.integer);
    const reminder = try XmlAdapter.requirements(arena, view.outputs);
    try testing.expect(std.mem.endsWith(u8, reminder, " For null, self-close the field's own tag. For an empty string, use an opening and closing tag with no content."));
}

test "XML custom nullable Reasoning distinguishes null empty content and omitted output" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { analysis: ?signature.Reasoning, answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    for ([_][]const u8{
        "<analysis/><answer>yes</answer>",
        "<analysis /><answer>yes</answer>",
        "<answer>yes</answer>",
    }) |completion| {
        const parsed = try XmlAdapter.parse(arena, view, completion);
        try testing.expect(parsed.get("analysis").? == .null);
        try testing.expectEqualStrings("yes", parsed.get("answer").?.string);
    }
    for ([_][]const u8{
        "<analysis></analysis><answer>yes</answer>",
        "<analysis><content></content></analysis><answer>yes</answer>",
    }) |completion| {
        const parsed = try XmlAdapter.parse(arena, view, completion);
        try testing.expectEqualStrings("", parsed.get("analysis").?.string);
    }
    for ([_][]const u8{
        "<analysis>step</analysis><answer>yes</answer>",
        "<analysis><content>step</content></analysis><answer>yes</answer>",
    }) |completion| {
        const parsed = try XmlAdapter.parse(arena, view, completion);
        try testing.expectEqualStrings("step", parsed.get("analysis").?.string);
    }
    const structure = try XmlAdapter.fieldStructure(arena, view.outputs);
    try testing.expect(std.mem.indexOf(u8, structure, "<content>") == null);
    const requirements = try XmlAdapter.requirements(arena, view.outputs);
    try testing.expect(std.mem.indexOf(u8, requirements, "For null, self-close the field's own tag.") != null);

    const Required = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { plan: signature.Reasoning, answer: []const u8 };
    });
    const required_view: adapter.View = .{ .instructions = "", .inputs = Required.input_fields_list, .outputs = Required.output_fields_list };
    try testing.expectError(error.AdapterParse, XmlAdapter.parse(arena, required_view, "<answer>yes</answer>"));
    try testing.expectError(error.AdapterParse, adapter.Adapter.json.parse(arena, required_view, "{\"plan\":null,\"answer\":\"yes\"}"));
}

test "required Reasoning and default CoT distinguish null forms from explicit empty content" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Required = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { plan: signature.Reasoning, answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Required.input_fields_list, .outputs = Required.output_fields_list };
    for ([_][]const u8{
        "<plan/><answer>yes</answer>",
        "<plan></plan><answer>yes</answer>",
        "<plan>\n\n</plan><answer>yes</answer>",
    }) |completion| try testing.expectError(error.AdapterParse, XmlAdapter.parse(arena, view, completion));
    for ([_][]const u8{
        "<plan>step</plan><answer>yes</answer>",
        "<plan><content>step</content></plan><answer>yes</answer>",
    }) |completion| {
        const parsed = try XmlAdapter.parse(arena, view, completion);
        try testing.expectEqualStrings("step", parsed.get("plan").?.string);
    }
    const empty_content = try XmlAdapter.parse(arena, view, "<plan><content></content></plan><answer>yes</answer>");
    try testing.expectEqualStrings("", empty_content.get("plan").?.string);

    const Task = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 };
    });
    const cot: @import("predict.zig").ChainOfThought(Task) = .{};
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{
        "<answer>yes</answer>",
        "{\"answer\":\"yes\"}",
        "{\"plan\":{\"content\":\"\"},\"answer\":\"yes\"}",
    } };
    const context: @import("predict.zig").Context = .{ .arena = arena, .lm = script.lm(), .json_fallback = false };
    try testing.expectError(error.AdapterParse, cot.forward(context, .{}));
    const json_context: @import("predict.zig").Context = .{ .arena = arena, .lm = script.lm(), .adapter = .json, .json_fallback = false };
    try testing.expectError(error.AdapterParse, cot.forward(json_context, .{}));
    const json_content = try cot.forward(json_context, .{});
    try testing.expectEqualStrings("", json_content.reasoning.?.content);
}

test "XML typed nested objects list items literals nulls and mapping keys" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Item = struct {
        pub const model_name = "Item";
        name: []const u8,
        count: i64,
        active: bool,
        kind: enum { small, large },
        note: ?[]const u8,
    };
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { items: []const Item, scores: signature.Dict(i64) };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const parsed = try XmlAdapter.parse(
        arena,
        view,
        "<items>\n<item>\n<name>A & B</name>\n<count>3</count>\n<active>True</active>\n<kind>'small'</kind>\n<note />\n</item>\n</items>\n" ++
            "<scores><entry key=\"a &amp; b\">7</entry><normal>2</normal></scores>",
    );
    const item = parsed.get("items").?.array.items[0].object;
    try testing.expectEqualStrings("A & B", item.get("name").?.string);
    try testing.expectEqual(@as(i64, 3), item.get("count").?.integer);
    try testing.expect(item.get("active").?.bool);
    try testing.expectEqualStrings("small", item.get("kind").?.string);
    try testing.expect(item.get("note").? == .null);
    try testing.expectEqual(@as(i64, 7), parsed.get("scores").?.object.get("a & b").?.integer);
    try testing.expectError(error.AdapterParse, XmlAdapter.parse(
        arena,
        view,
        "<items><item><name>A</name><count>3</count><active>True</active><kind>small</kind><note/><extra>x</extra></item></items><scores/>",
    ));
    try testing.expectError(error.AdapterParse, XmlAdapter.parse(
        arena,
        view,
        "<items><item><name>A</name><count>3</count><active>True</active><kind>small</kind><note/></item></items><scores><entry key=\"bad & key\">2</entry></scores>",
    ));
}

test "XML repeated arrays work while repeated scalars unknown and missing fields fail" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { names: []const []const u8, answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const repeated = try XmlAdapter.parse(arena, view, "<names>A</names><names>B</names><answer>yes</answer>");
    try testing.expectEqual(@as(usize, 2), repeated.get("names").?.array.items.len);
    try testing.expectEqualStrings("A", repeated.get("names").?.array.items[0].string);
    const wrapped = try XmlAdapter.parse(arena, view, "<names><item>A</item><item>B</item></names><answer>yes</answer>");
    try testing.expectEqual(@as(usize, 2), wrapped.get("names").?.array.items.len);
    const empty = try XmlAdapter.parse(arena, view, "<names/><answer/>");
    try testing.expectEqual(@as(usize, 0), empty.get("names").?.array.items.len);
    try testing.expectEqualStrings("", empty.get("answer").?.string);
    for ([_][]const u8{
        "<names/><answer>A</answer><answer>B</answer>",
        "<names/><answer>A</answer><unknown/>",
        "<names/>",
        "<names/><entry key=\"answer\">A</entry>",
        "<names/><answer>A</answer></wrong>",
        "<names><item>A</names><answer>A</answer>",
    }) |completion| try testing.expectError(error.AdapterParse, XmlAdapter.parse(arena, view, completion));
}

test "XML leading thoughts fences and output fields named think" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { answer: []const u8 };
    });
    const view: adapter.View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    for ([_][]const u8{
        "<think>discard <arbitrary>markup</think>\n```xml\n<answer>\nA\n</answer>\n```",
        "```\n<thinking>discard</thinking>\n<answer>A</answer>\n```",
        "<thinking>x</thinking><think>y</think>Some prose <answer>A</answer>",
    }) |completion| {
        const parsed = try XmlAdapter.parse(arena, view, completion);
        try testing.expectEqualStrings("A", parsed.get("answer").?.string);
    }
    for ([_][]const u8{
        "<think>unfinished<answer>A</answer>",
        "```xml\n<answer>A</answer>",
        "```xml\n<answer>A</answer>\n```<unknown>",
    }) |completion| try testing.expectError(error.AdapterParse, XmlAdapter.parse(arena, view, completion));
    const Think = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { think: []const u8 };
    });
    const thinking = try XmlAdapter.parse(arena, .{ .instructions = "", .inputs = Think.input_fields_list, .outputs = Think.output_fields_list }, "<think>retained</think>");
    try testing.expectEqualStrings("retained", thinking.get("think").?.string);
}
