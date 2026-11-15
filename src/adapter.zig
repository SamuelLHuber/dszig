//! Prompt adapters: DSPy 3.5.0 XMLAdapter and explicit ChatAdapter,
//! JSONAdapter and TwoStepAdapter. They turn (signature view, demos, inputs) into chat
//! messages and parse an LM completion back into an output record.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Writer = std.Io.Writer;
const inference = @import("inference.zig");
const tool = @import("tool.zig");

const pyjson = @import("pyjson.zig");
pub const XmlAdapter = @import("xml.zig").XmlAdapter;
const History = @import("history.zig").History;
const signature = @import("signature.zig");
const value = @import("value.zig");
const Field = signature.Field;
const Record = value.Record;

pub const Role = enum {
    system,
    user,
    assistant,
    tool,

    pub fn name(role: Role) []const u8 {
        return @tagName(role);
    }
};

pub const Content = @import("message_content.zig").Content;

pub const Message = struct {
    role: Role,
    content: Content,
    reasoning: ?[]const u8 = null,
    tool_calls: []const inference.ToolCall = &.{},
    tool_call_id: ?[]const u8 = null,
    continuation: []const inference.Continuation = &.{},
};

/// Runtime view of a signature, after instruction/description overrides
/// loaded from state or set by an optimizer.
pub const View = struct {
    instructions: []const u8,
    inputs: []const Field,
    outputs: []const Field,
    use_native_function_calling: bool = false,

    pub fn hasField(view: View, field_name: []const u8) bool {
        return view.findOutput(field_name) != null or view.findInput(field_name) != null;
    }

    pub fn findOutput(view: View, field_name: []const u8) ?Field {
        for (view.outputs) |f| if (std.mem.eql(u8, f.name, field_name)) return f;
        return null;
    }

    pub fn findInput(view: View, field_name: []const u8) ?Field {
        for (view.inputs) |f| if (std.mem.eql(u8, f.name, field_name)) return f;
        return null;
    }
};

pub const Error = error{ AdapterParse, JsonTooDeep } || value.ParseError;

/// Expand media where DSPy constructs user/tool parts; assistant/system stay text.
pub fn renderedContent(arena: Allocator, rendered: []const u8) Error!Content {
    return Content.fromRendered(arena, rendered) catch |err| return switch (err) {
        error.InvalidContent => error.AdapterParse,
        error.JsonTooDeep => error.JsonTooDeep,
        error.OutOfMemory => error.OutOfMemory,
    };
}

pub const Adapter = enum {
    chat,
    json,
    xml,
    two_step,

    pub fn format(adapter: Adapter, arena: Allocator, view: View, demos: []const Record, inputs: *const Record) Error![]Message {
        var messages: std.ArrayList(Message) = .empty;
        try messages.append(arena, .{ .role = .system, .content = .{ .text = try systemMessage(adapter, arena, view) } });
        try formatDemos(adapter, arena, view, demos, &messages);
        if (adapter == .two_step) {
            try messages.append(arena, .{ .role = .user, .content = try renderedContent(arena, try twoStepFields(arena, view.inputs, inputs, "")) });
            return messages.items;
        }
        var current_view = view;
        for (view.inputs) |field| {
            if (!std.mem.eql(u8, field.type_name, "History")) continue;
            var ordinary: std.ArrayList(Field) = .empty;
            for (view.inputs) |input_field| {
                if (!std.mem.eql(u8, input_field.type_name, "History")) try ordinary.append(arena, input_field);
            }
            current_view.inputs = ordinary.items;
            break;
        }
        for (view.inputs) |field| {
            if (!std.mem.eql(u8, field.type_name, "History")) continue;
            const history_value = inputs.get(field.name) orelse continue;
            const history = try History.fromValue(arena, history_value);
            try messages.appendSlice(arena, try history.format(adapter, arena, current_view));
        }
        const content = try userMessage(adapter, arena, current_view, inputs, "", true);
        if (content.len > 0) try messages.append(arena, .{ .role = .user, .content = try renderedContent(arena, content) });
        return messages.items;
    }

    pub fn parse(adapter: Adapter, arena: Allocator, view: View, completion: []const u8) Error!Record {
        return switch (adapter) {
            .chat => parseChat(arena, view, completion),
            .json => parseJson(arena, view, completion),
            .xml => XmlAdapter.parse(arena, view, completion),
            // Extraction needs its separate LM; use TwoStepAdapter.parse.
            .two_step => error.AdapterParse,
        };
    }

    /// Name used by DSPy for the adapter class.
    pub fn className(adapter: Adapter) []const u8 {
        return switch (adapter) {
            .chat => "ChatAdapter",
            .json => "JSONAdapter",
            .xml => "XMLAdapter",
            .two_step => "TwoStepAdapter",
        };
    }
};

fn twoStepSystem(arena: Allocator, view: View) Error![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    w.writeAll("You are a helpful assistant that can solve tasks based on user input.\nAs input, you will be provided with:\n") catch return error.OutOfMemory;
    try writeStripped(arena, w, fieldDescriptions, view.inputs);
    w.writeAll("\nYour outputs must contain:\n") catch return error.OutOfMemory;
    try writeStripped(arena, w, fieldDescriptions, view.outputs);
    w.writeAll("\nYou should lay out your outputs in detail so that your answer can be understood by another agent") catch return error.OutOfMemory;
    if (view.instructions.len > 0) {
        w.writeAll("\nSpecific instructions: ") catch return error.OutOfMemory;
        w.writeAll(view.instructions) catch return error.OutOfMemory;
    }
    return aw.written();
}

/// Unlike the structured adapters, this is Python str(value), including
/// typed model representations rather than their JSON wire serialization.
fn twoStepFields(arena: Allocator, fields: []const Field, record: *const Record, prefix: []const u8) Error![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    w.writeAll(prefix) catch return error.OutOfMemory;
    for (fields) |field| {
        const input = record.get(field.name) orelse continue;
        w.print("\n\n{s}: ", .{field.name}) catch return error.OutOfMemory;
        if (field.python_str) |render| {
            w.writeAll(try render(arena, input)) catch return error.OutOfMemory;
        } else {
            pyjson.writeStr(w, input) catch |err| return mapWriteError(err);
        }
    }
    return value.pyStrip(aw.written());
}

const incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied.";
const missing_incomplete = "Not supplied for this particular example. ";
const missing_complete = "Not supplied for this conversation history message. ";

fn systemMessage(adapter: Adapter, arena: Allocator, view: View) Error![]const u8 {
    if (adapter == .two_step) return twoStepSystem(arena, view);
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    writeAll(w, "Your input fields are:\n") catch return error.OutOfMemory;
    try writeStripped(arena, w, fieldDescriptions, view.inputs);
    writeAll(w, "\nYour output fields are:\n") catch return error.OutOfMemory;
    try writeStripped(arena, w, fieldDescriptions, view.outputs);
    writeAll(w, "\n") catch return error.OutOfMemory;
    try writeStrippedStructure(adapter, arena, w, view);
    writeAll(w, "\n") catch return error.OutOfMemory;
    try taskDescription(arena, w, view.instructions);
    return aw.written();
}

fn writeAll(w: *Writer, s: []const u8) Writer.Error!void {
    try w.writeAll(s);
}

fn writeStripped(
    arena: Allocator,
    w: *Writer,
    comptime render: fn (*Writer, []const Field) pyjson.Error!void,
    fields: []const Field,
) Error!void {
    var aw: Writer.Allocating = .init(arena);
    render(&aw.writer, fields) catch |err| return mapWriteError(err);
    w.writeAll(value.pyStrip(aw.written())) catch return error.OutOfMemory;
}

pub fn mapWriteError(err: pyjson.Error) Error {
    return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
        error.ParseValue => error.AdapterParse,
    };
}

/// `get_field_description_string(fields)`.
pub fn fieldDescriptionString(arena: Allocator, fields: []const Field) Error![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    try writeStripped(arena, &aw.writer, fieldDescriptions, fields);
    return aw.written();
}

/// `get_field_description_string`.
fn fieldDescriptions(w: *Writer, fields: []const Field) pyjson.Error!void {
    for (fields, 0..) |f, i| {
        if (i > 0) try w.writeByte('\n');
        try w.print("{d}. `{s}` ({s}): ", .{ i + 1, f.name, f.type_name });
        if (!isDefaultDesc(f)) try w.writeAll(f.desc);
        if (f.type_description.len > 0) try w.print("\n    Type description of {s}: {s}", .{ if (f.type_description_name.len > 0) f.type_description_name else f.type_name, f.type_description });
    }
}

fn isDefaultDesc(f: Field) bool {
    return f.desc.len == f.name.len + 3 and std.mem.startsWith(u8, f.desc, "${") and
        std.mem.eql(u8, f.desc[2 .. 2 + f.name.len], f.name) and f.desc[f.desc.len - 1] == '}';
}

/// TypeAdapter(field.annotation).json_schema(), shared by typed XML and
/// provider response schemas. The XML schema variant retains model field order.
pub fn fieldSchema(arena: Allocator, field: Field) Error!Value {
    if (field.kind == .json or field.code_type or (field.kind == .literal and field.schema.len > 0)) {
        const source = if (field.xml_schema.len > 0) field.xml_schema else field.schema;
        return @import("json_repair.zig").parseJson(arena, source) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.AdapterParse,
        };
    }
    const source: []const u8 = switch (field.kind) {
        .str => "{\"type\":\"string\"}",
        .boolean => "{\"type\":\"boolean\"}",
        .int => "{\"type\":\"integer\"}",
        .float => "{\"type\":\"number\"}",
        .str_list => "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}",
        .json_object => "{\"type\":\"object\",\"additionalProperties\":true}",
        .str_dict => "{\"type\":\"object\",\"additionalProperties\":{\"type\":\"string\"}}",
        .literal => {
            var out: Record = .empty;
            if (field.literals.len == 1) {
                try out.put(arena, "const", .{ .string = field.literals[0] });
            } else {
                var members: @import("runtime_value.zig").Array = .init(arena);
                for (field.literals) |member| try members.append(.{ .string = member });
                try out.put(arena, "enum", .{ .array = members });
            }
            try out.put(arena, "type", .{ .string = "string" });
            return .{ .object = out };
        },
        .json => unreachable,
    };
    return std.json.parseFromSliceLeaky(Value, arena, source, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.AdapterParse,
    };
}

/// JSONAdapter chooses a provider format before execution, never by retrying an
/// LM call. Match DSPy 3.5's root-only dictionary guard, including its upstream
/// limitation: nested dictionaries reach enforceRequired and lose their keys.
/// Optional dictionary anyOf branches are not traversed and remain open.
pub fn responseFormat(arena: Allocator, view: View, supports_schema: bool) Error!inference.ResponseFormat {
    if (!supports_schema) return .json_object;
    for (view.outputs) |field| {
        if (field.kind == .json_object or field.kind == .str_dict) return .json_object;
        if (field.native_tool_calls and !view.use_native_function_calling) return .json_object;
    }
    var properties: Record = .empty;
    var definitions: Record = .empty;
    for (view.outputs) |field| {
        if (field.native_tool_calls and view.use_native_function_calling) continue;
        var schema = fieldSchema(arena, field) catch |err| switch (err) {
            error.AdapterParse => return .json_object,
            else => return err,
        };
        const additional = schema.object.get("additionalProperties");
        if (additional) |mapping| {
            if (mapping == .object or (mapping == .bool and mapping.bool)) return .json_object;
        }
        if (schema.object.get("$defs")) |defs| {
            if (defs == .object) {
                var it = defs.object.iterator();
                while (it.next()) |entry| try putDefinition(arena, &definitions, entry.key_ptr.*, entry.value_ptr.*);
            }
            _ = schema.object.orderedRemove("$defs");
        }
        const schema_type = schema.object.get("type");
        const model_title = schema.object.get("title");
        if (schema_type != null and schema_type.? == .string and std.mem.eql(u8, schema_type.?.string, "object") and
            model_title != null and model_title.? == .string)
        {
            const name = if (field.schema_definition_name.len > 0) field.schema_definition_name else model_title.?.string;
            try putDefinition(arena, &definitions, name, schema);
            var reference: Record = .empty;
            try reference.put(arena, "$ref", .{ .string = try std.fmt.allocPrint(arena, "#/$defs/{s}", .{name}) });
            schema = .{ .object = reference };
        } else if (!schema.object.contains("title") and needsFieldTitle(schema)) {
            try schema.object.put(arena, "title", .{ .string = try schemaTitle(arena, field.name) });
        }
        if (field.default_value) |default_value| try schema.object.put(arena, "default", try default_value(arena));
        try properties.put(arena, field.name, schema);
    }
    var root: Record = .empty;
    if (definitions.count() > 0) try root.put(arena, "$defs", .{ .object = definitions });
    try root.put(arena, "additionalProperties", .{ .bool = false });
    try root.put(arena, "properties", .{ .object = properties });
    try root.put(arena, "title", .{ .string = "DSPyProgramOutputs" });
    try root.put(arena, "type", .{ .string = "object" });
    var schema: Value = .{ .object = root };
    try enforceRequired(arena, &schema, 0);
    return .{ .json_schema = .{ .name = "DSPyProgramOutputs", .schema = schema } };
}

/// Python class/module identities are not available for Pydantic's collision
/// renaming. Never make distinct models alias the same provider definition.
fn putDefinition(arena: Allocator, definitions: *Record, name: []const u8, schema: Value) Error!void {
    const entry = try definitions.getOrPut(arena, name);
    if (entry.found_existing) {
        if (!(try value.pyEqual(entry.value_ptr.*, schema))) return error.AdapterParse;
    } else {
        entry.value_ptr.* = schema;
    }
}

fn needsFieldTitle(schema: Value) bool {
    if (schema.object.contains("$ref")) return false;
    if (schema.object.get("anyOf")) |branches| {
        if (branches == .array) for (branches.array.items) |branch| {
            if (branch == .object and branch.object.contains("$ref")) return false;
        };
    }
    return true;
}

fn schemaTitle(arena: Allocator, name: []const u8) Allocator.Error![]const u8 {
    const title = try arena.dupe(u8, name);
    var previous_cased = false;
    for (title) |*byte| {
        if (byte.* == '_') byte.* = ' ';
        const cased = std.ascii.isAlphabetic(byte.*);
        if (cased) byte.* = if (previous_cased) std.ascii.toLower(byte.*) else std.ascii.toUpper(byte.*);
        previous_cased = cased;
    }
    return std.mem.trim(u8, title, " ");
}

/// Match DSPy's strict-schema traversal: direct object properties, array items,
/// and definitions. Union schemas retain their upstream representation.
fn enforceRequired(arena: Allocator, schema: *Value, depth: usize) Error!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (schema.* != .object) return;
    const schema_type = schema.object.get("type") orelse .null;
    if (schema_type == .string and std.mem.eql(u8, schema_type.string, "object")) {
        if (!schema.object.contains("properties")) try schema.object.put(arena, "properties", .{ .object = .empty });
        const props = schema.object.get("properties").?;
        if (props == .object) {
            var required: @import("runtime_value.zig").Array = .init(arena);
            for (props.object.keys()) |name| try required.append(.{ .string = name });
            try schema.object.put(arena, "required", .{ .array = required });
            try schema.object.put(arena, "additionalProperties", .{ .bool = false });
            const children = schema.object.getPtr("properties").?;
            for (children.object.values()) |*child| try enforceRequired(arena, child, depth + 1);
        }
    }
    if (schema_type == .string and std.mem.eql(u8, schema_type.string, "array")) {
        if (schema.object.getPtr("items")) |item| try enforceRequired(arena, item, depth + 1);
    }
    for ([_][]const u8{ "$defs", "definitions" }) |key| {
        if (schema.object.getPtr(key)) |defs| {
            if (defs.* == .object) for (defs.object.values()) |*child| try enforceRequired(arena, child, depth + 1);
        }
    }
}

/// Serialize custom values without discarding their round-trip model fields.
pub fn formattedValue(arena: Allocator, field: Field, input: Value) Error!Value {
    if (field.native_tools and input == .array) return tool.formatCatalogValue(arena, input);
    if (field.native_tool_calls and input == .object) {
        const calls = tool.ToolCalls.fromValue(arena, input) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.ParseValue => error.AdapterParse,
        };
        return value.serializedValue(arena, try calls.formatValue(arena)) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.ParseValue => error.AdapterParse,
        };
    }
    return value.serializedValue(arena, input) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.ParseValue => error.AdapterParse,
    };
}

pub fn writeFieldValue(arena: Allocator, w: *Writer, field: Field, input: Value) Error!void {
    value.formatFieldValue(w, field, try formattedValue(arena, field, input)) catch |err| return mapWriteError(err);
}

/// `translate_field_type`.
pub fn translateFieldType(w: *Writer, f: Field, is_input: bool) Writer.Error!void {
    try w.print("{{{s}}}", .{f.name});
    if (is_input or f.kind == .str) return;
    try w.writeAll("        # note: the value you produce ");
    switch (f.kind) {
        .str => unreachable,
        .boolean => try w.writeAll("must be True or False"),
        .int => try w.writeAll("must be a single int value"),
        .float => try w.writeAll("must be a single float value"),
        .literal => {
            try w.writeAll("must exactly match (no extra characters) one of: ");
            for (f.literals, 0..) |lit, i| {
                if (i > 0) try w.writeAll("; ");
                try w.writeAll(lit);
            }
        },
        .str_list => try w.writeAll(
            \\must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}
        ),
        .json_object => try w.writeAll(
            \\must adhere to the JSON schema: {"type": "object", "additionalProperties": true}
        ),
        .str_dict => try w.writeAll(
            \\must adhere to the JSON schema: {"type": "object", "additionalProperties": {"type": "string"}}
        ),
        .json => try w.print("must adhere to the JSON schema: {s}", .{f.schema}),
    }
}

/// `format_field_with_value` (user role) over the translated field types.
fn structureFields(arena: Allocator, w: *Writer, fields: []const Field, is_input: bool) Error!void {
    var aw: Writer.Allocating = .init(arena);
    const iw = &aw.writer;
    for (fields, 0..) |f, i| {
        if (i > 0) iw.writeAll("\n\n") catch return error.OutOfMemory;
        iw.print("[[ ## {s} ## ]]\n", .{f.name}) catch return error.OutOfMemory;
        translateFieldType(iw, f, is_input) catch return error.OutOfMemory;
    }
    w.writeAll(value.pyStrip(aw.written())) catch return error.OutOfMemory;
}

fn writeStrippedStructure(adapter: Adapter, arena: Allocator, w: *Writer, view: View) Error!void {
    if (adapter == .xml) {
        w.writeAll(try XmlAdapter.fieldStructure(arena, view.outputs)) catch return error.OutOfMemory;
        return;
    }
    var aw: Writer.Allocating = .init(arena);
    const sw = &aw.writer;
    sw.writeAll("All interactions will be structured in the following way, with the appropriate values filled in.\n\n") catch
        return error.OutOfMemory;
    switch (adapter) {
        .xml, .two_step => unreachable,
        .chat => {
            try structureFields(arena, sw, view.inputs, true);
            sw.writeAll("\n\n") catch return error.OutOfMemory;
            try structureFields(arena, sw, view.outputs, false);
            sw.writeAll("\n\n[[ ## completed ## ]]\n") catch return error.OutOfMemory;
        },
        .json => {
            sw.writeAll("Inputs will have the following structure:\n\n") catch return error.OutOfMemory;
            try structureFields(arena, sw, view.inputs, true);
            sw.writeAll("\n\nOutputs will be a JSON object with the following fields.\n\n") catch return error.OutOfMemory;
            var record: Record = .empty;
            for (view.outputs) |f| {
                var fw: Writer.Allocating = .init(arena);
                translateFieldType(&fw.writer, f, false) catch return error.OutOfMemory;
                try record.put(arena, f.name, .{ .string = fw.written() });
            }
            pyjson.writeJson(sw, .{ .object = record }, .indent2) catch |err| return mapWriteError(err);
        },
    }
    w.writeAll(value.pyStrip(aw.written())) catch return error.OutOfMemory;
}

/// `format_task_description`: dedent, then indent every line by 8 spaces.
fn taskDescription(arena: Allocator, w: *Writer, instructions: []const u8) Error!void {
    w.writeAll("In adhering to this structure, your objective is: ") catch return error.OutOfMemory;
    const dedented = try dedent(arena, instructions);
    var lines = LineIterator{ .text = dedented };
    while (lines.next()) |line| {
        w.writeAll("\n        ") catch return error.OutOfMemory;
        w.writeAll(line) catch return error.OutOfMemory;
    }
}

/// Python `textwrap.dedent`.
pub fn dedent(arena: Allocator, text: []const u8) Allocator.Error![]const u8 {
    var margin: ?[]const u8 = null;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const indent_len = std.mem.indexOfNone(u8, line, " \t") orelse continue;
        const indent = line[0..indent_len];
        if (margin) |m| {
            var common: usize = 0;
            while (common < m.len and common < indent.len and m[common] == indent[common]) common += 1;
            margin = m[0..common];
        } else margin = indent;
    }
    var out: std.ArrayList(u8) = .empty;
    it = std.mem.splitScalar(u8, text, '\n');
    var first = true;
    while (it.next()) |line| {
        if (!first) try out.append(arena, '\n');
        first = false;
        if (std.mem.indexOfNone(u8, line, " \t") == null) continue;
        try out.appendSlice(arena, line[if (margin) |m| m.len else 0..]);
    }
    return out.items;
}

/// Python `str.splitlines()` (no keepends).
pub const LineIterator = struct {
    text: []const u8,
    index: usize = 0,

    pub fn next(it: *LineIterator) ?[]const u8 {
        if (it.index >= it.text.len) return null;
        const start = it.index;
        var i = start;
        while (i < it.text.len) {
            const len = terminatorLen(it.text[i..]);
            if (len > 0) {
                it.index = i + len;
                return it.text[start..i];
            }
            i += 1;
        }
        it.index = it.text.len;
        return it.text[start..];
    }

    fn terminatorLen(s: []const u8) usize {
        switch (s[0]) {
            '\r' => return if (s.len > 1 and s[1] == '\n') 2 else 1,
            '\n', 0x0b, 0x0c, 0x1c, 0x1d, 0x1e => return 1,
            0xc2 => return if (s.len > 1 and s[1] == 0x85) 2 else 0,
            0xe2 => return if (s.len > 2 and s[1] == 0x80 and (s[2] == 0xa8 or s[2] == 0xa9)) 3 else 0,
            else => return 0,
        }
    }
};

fn formatDemos(adapter: Adapter, arena: Allocator, view: View, demos: []const Record, messages: *std.ArrayList(Message)) Error!void {
    for ([_]bool{ false, true }) |want_complete| {
        for (demos) |*demo| {
            const complete = isComplete(view, demo);
            if (complete != want_complete) continue;
            if (!complete and !(hasAny(view.inputs, demo) and hasAny(view.outputs, demo))) continue;
            const prefix = if (complete) "" else incomplete_demo_prefix;
            const missing = if (complete) missing_complete else missing_incomplete;
            try messages.append(arena, .{ .role = .user, .content = try renderedContent(arena, try userMessage(adapter, arena, view, demo, prefix, false)) });
            try messages.append(arena, .{ .role = .assistant, .content = .{ .text = try assistantMessage(adapter, arena, view, demo, missing) } });
        }
    }
}

fn isComplete(view: View, demo: *const Record) bool {
    for ([_][]const Field{ view.inputs, view.outputs }) |fields| {
        for (fields) |f| {
            const v = demo.get(f.name) orelse return false;
            if (v == .null) return false;
        }
    }
    return true;
}

fn hasAny(fields: []const Field, demo: *const Record) bool {
    for (fields) |f| if (demo.contains(f.name)) return true;
    return false;
}

/// `format_user_message_content` (shared by both adapters).
/// `ChatAdapter.format_user_message_content(signature, inputs)` over input
/// fields only (how `ReAct` renders its trajectory).
pub fn formatInputs(arena: Allocator, fields: []const Field, inputs: *const Record) Error![]const u8 {
    return userMessage(.chat, arena, .{ .instructions = "", .inputs = fields, .outputs = &.{} }, inputs, "", false);
}

/// Render a history user turn using the same formatter as the current request.
pub fn formatUser(adapter: Adapter, arena: Allocator, view: View, inputs: *const Record, main_request: bool) Error![]const u8 {
    return userMessage(adapter, arena, view, inputs, "", main_request);
}

/// History assistant turns contain only fields actually supplied by that turn.
pub fn formatAssistantPartial(adapter: Adapter, arena: Allocator, view: View, outputs: *const Record) Error![]const u8 {
    var present: std.ArrayList(Field) = .empty;
    for (view.outputs) |field| if (outputs.contains(field.name)) try present.append(arena, field);
    var partial_view = view;
    partial_view.outputs = present.items;
    return assistantMessage(adapter, arena, partial_view, outputs, "");
}

fn userMessage(adapter: Adapter, arena: Allocator, view: View, inputs: *const Record, prefix: []const u8, main_request: bool) Error![]const u8 {
    if (adapter == .two_step) return twoStepFields(arena, view.inputs, inputs, prefix);
    if (adapter == .xml) return XmlAdapter.user(arena, view, inputs, prefix, main_request);
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    w.writeAll(prefix) catch return error.OutOfMemory;
    for (view.inputs) |f| {
        const v = inputs.get(f.name) orelse continue;
        w.print("\n\n[[ ## {s} ## ]]\n", .{f.name}) catch return error.OutOfMemory;
        try writeFieldValue(arena, w, f, v);
    }
    if (main_request) {
        w.writeAll("\n\n") catch return error.OutOfMemory;
        outputRequirements(adapter, w, view) catch return error.OutOfMemory;
    }
    w.writeAll("\n\n") catch return error.OutOfMemory;
    return value.pyStrip(aw.written());
}

fn outputRequirements(adapter: Adapter, w: *Writer, view: View) Writer.Error!void {
    switch (adapter) {
        .xml, .two_step => unreachable,
        .chat => try w.writeAll("Respond with the corresponding output fields, starting with the field "),
        .json => try w.writeAll("Respond with a JSON object in the following order of fields: "),
    }
    for (view.outputs, 0..) |f, i| {
        if (i > 0) try w.writeAll(", then ");
        switch (adapter) {
            .xml, .two_step => unreachable,
            .chat => try w.print("`[[ ## {s} ## ]]`", .{f.name}),
            .json => try w.print("`{s}`", .{f.name}),
        }
        if (f.native_tool_calls) {
            try w.writeAll(" (must be a JSON object like {\"tool_calls\": [{\"name\": \"...\", \"args\": {...}}]})");
        } else if (f.kind != .str or f.native_reasoning or f.code_type) {
            try w.print(" (must be formatted as a valid Python {s})", .{f.type_name});
        }
    }
    switch (adapter) {
        .xml, .two_step => unreachable,
        .chat => try w.writeAll(", and then ending with the marker for `[[ ## completed ## ]]`."),
        .json => try w.writeAll("."),
    }
}

/// `format_assistant_message_content`.
fn assistantMessage(adapter: Adapter, arena: Allocator, view: View, outputs: *const Record, missing: []const u8) Error![]const u8 {
    if (adapter == .two_step) return twoStepFields(arena, view.outputs, outputs, "");
    if (adapter == .xml) return XmlAdapter.fields(arena, view.outputs, outputs, true, missing);
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    switch (adapter) {
        .xml, .two_step => unreachable,
        .chat => {
            var fw: Writer.Allocating = .init(arena);
            for (view.outputs, 0..) |f, i| {
                if (i > 0) fw.writer.writeAll("\n\n") catch return error.OutOfMemory;
                fw.writer.print("[[ ## {s} ## ]]\n", .{f.name}) catch return error.OutOfMemory;
                const v = outputs.get(f.name) orelse Value{ .string = missing };
                try writeFieldValue(arena, &fw.writer, f, v);
            }
            w.writeAll(value.pyStrip(fw.written())) catch return error.OutOfMemory;
            w.writeAll("\n\n[[ ## completed ## ]]\n") catch return error.OutOfMemory;
        },
        .json => {
            var record: Record = .empty;
            for (view.outputs) |f| {
                try record.put(arena, f.name, try formattedValue(arena, f, outputs.get(f.name) orelse Value{ .string = missing }));
            }
            pyjson.writeJson(w, .{ .object = record }, .indent2) catch |err| return mapWriteError(err);
        },
    }
    return aw.written();
}

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c >= 0x80;
}

/// Match `\[\[ ## (\w+) ## \]\]` at the start of `s`; returns name and end.
fn matchHeader(s: []const u8) ?struct { name: []const u8, end: usize } {
    const open = "[[ ## ";
    const close = " ## ]]";
    if (!std.mem.startsWith(u8, s, open)) return null;
    var i: usize = open.len;
    while (i < s.len and isWordByte(s[i])) i += 1;
    if (i == open.len) return null;
    if (!std.mem.startsWith(u8, s[i..], close)) return null;
    return .{ .name = s[open.len..i], .end = i + close.len };
}

const Section = struct { header: ?[]const u8, lines: std.ArrayList([]const u8) };

/// Native signatures retain their generated validator. Runtime schema-backed
/// Fields use the same supported type vocabulary instead of dereferencing null.
pub fn parseFieldValue(arena: Allocator, field: Field, input: Value) Error!Value {
    const parsed = if (field.kind == .json and field.validate == null)
        @import("xml_schema.zig").parse(arena, try fieldSchema(arena, field), input)
    else
        value.parseJsonValue(arena, field, input);
    return parsed catch |err| switch (err) {
        error.ParseValue => error.AdapterParse,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn parseChat(arena: Allocator, view: View, completion: []const u8) Error!Record {
    var sections: std.ArrayList(Section) = .empty;
    try sections.append(arena, .{ .header = null, .lines = .empty });
    var lines = LineIterator{ .text = completion };
    while (lines.next()) |line| {
        if (matchHeader(value.pyStrip(line))) |m| {
            // DSPy slices the *unstripped* line at the stripped match end.
            const remaining = value.pyStrip(if (m.end <= line.len) line[m.end..] else "");
            var section: Section = .{ .header = m.name, .lines = .empty };
            if (remaining.len > 0) try section.lines.append(arena, remaining);
            try sections.append(arena, section);
        } else {
            try sections.items[sections.items.len - 1].lines.append(arena, line);
        }
    }
    var fields: Record = .empty;
    for (sections.items) |section| {
        const header = section.header orelse continue;
        if (fields.contains(header)) continue;
        const field = view.findOutput(header) orelse continue;
        const joined = try std.mem.join(arena, "\n", section.lines.items);
        const parsed = try parseFieldValue(arena, field, .{ .string = value.pyStrip(joined) });
        try fields.put(arena, field.name, parsed);
    }
    try requireAllOutputs(arena, view, &fields);
    return fields;
}

fn requireAllOutputs(arena: Allocator, view: View, fields: *Record) Error!void {
    var completed: Record = .empty;
    for (view.outputs) |field| {
        const supplied = fields.get(field.name);
        const parsed = supplied orelse (try signature.outputDefault(arena, field)) orelse return error.AdapterParse;
        try completed.put(arena, field.name, parsed);
    }
    fields.* = completed;
}

fn parseJson(arena: Allocator, view: View, completion: []const u8) Error!Record {
    const object = (try parseObject(arena, completion)) orelse blk: {
        const span = firstBalancedBraces(completion) orelse return error.AdapterParse;
        break :blk (try parseObject(arena, span)) orelse return error.AdapterParse;
    };
    var fields: Record = .empty;
    var it = object.iterator();
    while (it.next()) |entry| {
        const field = view.findOutput(entry.key_ptr.*) orelse continue;
        const parsed = try parseFieldValue(arena, field, entry.value_ptr.*);
        try fields.put(arena, field.name, parsed);
    }
    try requireAllOutputs(arena, view, &fields);
    return fields;
}

fn parseObject(arena: Allocator, text: []const u8) Error!?Record {
    const v = @import("json_repair.zig").loads(arena, text) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
        error.ParseValue, error.IntegerTooLong => error.AdapterParse,
    };
    return if (v == .object) v.object else null;
}

/// Leftmost match of the recursive regex `\{(?:[^{}]|(?R))*\}`.
fn firstBalancedBraces(s: []const u8) ?[]const u8 {
    var start: usize = 0;
    while (std.mem.indexOfScalarPos(u8, s, start, '{')) |open| {
        var depth: usize = 0;
        var i = open;
        while (i < s.len) : (i += 1) {
            if (s[i] == '{') depth += 1;
            if (s[i] == '}') {
                depth -= 1;
                if (depth == 0) return s[open .. i + 1];
            }
        }
        start = open + 1;
    }
    return null;
}

const testing = std.testing;

test "splitlines and dedent match Python" {
    var it = LineIterator{ .text = "a\r\nb\n\nc\n" };
    try testing.expectEqualStrings("a", it.next().?);
    try testing.expectEqualStrings("b", it.next().?);
    try testing.expectEqualStrings("", it.next().?);
    try testing.expectEqualStrings("c", it.next().?);
    try testing.expect(it.next() == null);
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqualStrings("a\n  b\n\nc", try dedent(arena.allocator(), "    a\n      b\n  \n    c"));
}

test "chat parse takes first section and requires all outputs" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { reasoning: []const u8, answer: []const u8 };
    });
    const view: View = .{ .instructions = Sig.instructions, .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const r = try Adapter.chat.parse(arena, view,
        \\[[ ## reasoning ## ]] think
        \\more
        \\[[ ## answer ## ]]
        \\Paris
        \\[[ ## answer ## ]]
        \\Lyon
        \\[[ ## completed ## ]]
    );
    try testing.expectEqualStrings("think\nmore", r.get("reasoning").?.string);
    try testing.expectEqualStrings("Paris", r.get("answer").?.string);
    try testing.expectError(error.AdapterParse, Adapter.chat.parse(arena, view, "[[ ## answer ## ]]\nx"));
    const j = try Adapter.json.parse(arena, view, "Sure:\n```json\n{\"answer\": \"Paris\", \"reasoning\": \"r\", \"x\": 1}\n```");
    // DSPy 3.5 apply_output_field_defaults emits signature order.
    try testing.expectEqualStrings("reasoning", j.keys()[0]);
}

test "JSON response format closes model objects and falls back for direct mappings" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Item = struct {
        pub const model_name = "Item";
        count: i64,
        note: ?[]const u8 = null,
    };
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { item: Item, label: []const u8 };
    });
    const view: View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const chosen = try responseFormat(arena, view, true);
    try testing.expect(chosen == .json_schema);
    try testing.expectEqualStrings("DSPyProgramOutputs", chosen.json_schema.name);
    const root = chosen.json_schema.schema.object;
    try testing.expect(!root.get("additionalProperties").?.bool);
    try testing.expectEqual(@as(usize, 2), root.get("required").?.array.items.len);
    const item = root.get("$defs").?.object.get("Item").?.object;
    try testing.expect(!item.get("additionalProperties").?.bool);
    try testing.expectEqual(@as(usize, 2), item.get("required").?.array.items.len);
    try testing.expectEqualStrings("#/$defs/Item", root.get("properties").?.object.get("item").?.object.get("$ref").?.string);
    try testing.expect((try responseFormat(arena, view, false)) == .json_object);
    const Mapping = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { names: signature.StrDict };
    });
    try testing.expect((try responseFormat(arena, .{ .instructions = "", .inputs = Mapping.input_fields_list, .outputs = Mapping.output_fields_list }, true)) == .json_object);
}

test "JSON response schemas retain upstream nested dictionary limitations without rejecting parsed keys" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const ModelWithMapping = struct {
        pub const model_name = "ModelWithMapping";
        labels: signature.Dict(i64),
    };
    const ReferencedMapping = struct {
        pub const model_name = "ReferencedMapping";
        item: ModelWithMapping,
    };
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct {
            model: ModelWithMapping,
            list: []const signature.Dict(i64),
            definitions: ReferencedMapping,
            optional: ?signature.Dict(i64),
            optional_model: ?ModelWithMapping,
        };
    });
    const view: View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const chosen = try responseFormat(arena, view, true);
    try testing.expect(chosen == .json_schema);
    const root = chosen.json_schema.schema.object;
    const properties = root.get("properties").?.object;
    const definitions = root.get("$defs").?.object;
    const model = definitions.get("ModelWithMapping").?.object;
    const mapping = model.get("properties").?.object.get("labels").?.object;
    try testing.expect(!mapping.get("additionalProperties").?.bool);
    try testing.expectEqual(@as(usize, 0), mapping.get("properties").?.object.count());
    try testing.expectEqual(@as(usize, 0), mapping.get("required").?.array.items.len);
    const list_mapping = properties.get("list").?.object.get("items").?.object;
    try testing.expect(!list_mapping.get("additionalProperties").?.bool);
    try testing.expectEqual(@as(usize, 0), list_mapping.get("properties").?.object.count());
    try testing.expectEqual(@as(usize, 0), list_mapping.get("required").?.array.items.len);
    const referenced = definitions.get("ReferencedMapping").?.object;
    try testing.expect(!referenced.get("additionalProperties").?.bool);
    try testing.expectEqualStrings(
        "#/$defs/ModelWithMapping",
        referenced.get("properties").?.object.get("item").?.object.get("$ref").?.string,
    );
    const optional = properties.get("optional").?.object.get("anyOf").?.array.items;
    try testing.expectEqualStrings("integer", optional[0].object.get("additionalProperties").?.object.get("type").?.string);
    try testing.expect(!optional[0].object.contains("properties"));
    try testing.expect(!optional[0].object.contains("required"));
    try testing.expectEqualStrings("null", optional[1].object.get("type").?.string);
    const optional_model = properties.get("optional_model").?.object.get("anyOf").?.array.items;
    try testing.expectEqualStrings("#/$defs/ModelWithMapping", optional_model[0].object.get("$ref").?.string);
    try testing.expectEqualStrings("null", optional_model[1].object.get("type").?.string);
    try testing.expect(!root.get("additionalProperties").?.bool);
    try testing.expectEqual(@as(usize, 5), root.get("required").?.array.items.len);

    // Upstream's generated provider schema is over-restrictive, but parsing
    // still validates against the annotation and accepts nonempty dictionaries.
    const parsed = try Adapter.json.parse(arena, view,
        \\{"model":{"labels":{"allowed":7}},"list":[{"allowed":8}],"definitions":{"item":{"labels":{"allowed":9}}},"optional":{"allowed":10},"optional_model":{"labels":{"allowed":11}}}
    );
    try testing.expectEqual(@as(i64, 7), parsed.get("model").?.object.get("labels").?.object.get("allowed").?.integer);
    try testing.expectEqual(@as(i64, 8), parsed.get("list").?.array.items[0].object.get("allowed").?.integer);
    try testing.expectEqual(@as(i64, 9), parsed.get("definitions").?.object.get("item").?.object.get("labels").?.object.get("allowed").?.integer);
    try testing.expectEqual(@as(i64, 10), parsed.get("optional").?.object.get("allowed").?.integer);
    try testing.expectEqual(@as(i64, 11), parsed.get("optional_model").?.object.get("labels").?.object.get("allowed").?.integer);
}

test "JSON nullable custom Reasoning retains its model schema and accepts explicit null" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct { analysis: ?signature.Reasoning, answer: []const u8 };
    });
    const view: View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const chosen = try responseFormat(arena, view, true);
    try testing.expect(chosen == .json_schema);
    const root = chosen.json_schema.schema.object;
    const analysis = root.get("properties").?.object.get("analysis").?.object;
    const alternatives = analysis.get("anyOf").?.array.items;
    try testing.expectEqualStrings("#/$defs/Reasoning", alternatives[0].object.get("$ref").?.string);
    try testing.expectEqualStrings("null", alternatives[1].object.get("type").?.string);
    const reasoning = root.get("$defs").?.object.get("Reasoning").?.object;
    try testing.expect(!reasoning.get("additionalProperties").?.bool);
    try testing.expectEqualStrings("string", reasoning.get("properties").?.object.get("content").?.object.get("type").?.string);
    try testing.expectEqualStrings("content", reasoning.get("required").?.array.items[0].string);
    const nulls = try Adapter.json.parse(arena, view, "{\"analysis\":null,\"answer\":\"yes\"}");
    try testing.expect(nulls.get("analysis").? == .null);
    const omitted = try Adapter.json.parse(arena, view, "{\"answer\":\"yes\"}");
    try testing.expect(omitted.get("analysis").? == .null);
    for ([_][]const u8{
        "{\"analysis\":\"\",\"answer\":\"yes\"}",
        "{\"analysis\":{\"content\":\"\"},\"answer\":\"yes\"}",
    }) |completion| {
        const empty = try Adapter.json.parse(arena, view, completion);
        try testing.expectEqualStrings("", (try signature.Reasoning.fromValue(empty.get("analysis").?)).content);
    }
    try testing.expectError(error.AdapterParse, Adapter.json.parse(arena, view, "{\"analysis\":{\"content\":7},\"answer\":\"yes\"}"));
}

test "adapter defaults follow signature order and preserve explicit null" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct {
            first: ?[]const u8 = "fallback",
            second: ?i64,
            required: []const u8,
        };
    });
    const view: View = .{ .instructions = "", .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    const adapters = [_]Adapter{ .chat, .json, .xml };
    const absent = [_][]const u8{
        "[[ ## required ## ]]\nprovided",
        "{\"required\":\"provided\"}",
        "<required>provided</required>",
    };
    const explicit = [_][]const u8{
        "[[ ## first ## ]]\nnull\n[[ ## required ## ]]\nprovided",
        "{\"first\":null,\"required\":\"provided\"}",
        "<first/><required>provided</required>",
    };
    for (adapters, absent, explicit) |which, absent_text, explicit_text| {
        const completed = try which.parse(arena, view, absent_text);
        try testing.expectEqualStrings("first", completed.keys()[0]);
        try testing.expectEqualStrings("fallback", completed.get("first").?.string);
        try testing.expect(completed.get("second").? == .null);
        try testing.expectEqualStrings("provided", completed.get("required").?.string);
        const supplied = try which.parse(arena, view, explicit_text);
        // Optional[str] intentionally retains literal \"null\" in a text leaf.
        if (which == .chat) {
            try testing.expectEqualStrings("null", supplied.get("first").?.string);
        } else try testing.expect(supplied.get("first").? == .null);
        try testing.expectEqualStrings("provided", supplied.get("required").?.string);
    }
}
