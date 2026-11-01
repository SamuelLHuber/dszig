//! Prompt adapters: byte-compatible ports of DSPy 3.4.1 `ChatAdapter` and
//! `JSONAdapter`. They turn (signature view, demos, inputs) into chat
//! messages and parse an LM completion back into an output record.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Writer = std.Io.Writer;

const pyjson = @import("pyjson.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const Field = signature.Field;
const Record = value.Record;

pub const Role = enum {
    system,
    user,
    assistant,

    pub fn name(role: Role) []const u8 {
        return @tagName(role);
    }
};

pub const Message = struct { role: Role, content: []const u8 };

/// Runtime view of a signature, after instruction/description overrides
/// loaded from state or set by an optimizer.
pub const View = struct {
    instructions: []const u8,
    inputs: []const Field,
    outputs: []const Field,

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

pub const Error = error{ AdapterParse, JsonTooDeep } || Allocator.Error;

pub const Adapter = enum {
    chat,
    json,

    pub fn format(adapter: Adapter, arena: Allocator, view: View, demos: []const Record, inputs: *const Record) Error![]Message {
        var messages: std.ArrayList(Message) = .empty;
        try messages.append(arena, .{ .role = .system, .content = try systemMessage(adapter, arena, view) });
        try formatDemos(adapter, arena, view, demos, &messages);
        const content = try userMessage(adapter, arena, view, inputs, "", true);
        if (content.len > 0) try messages.append(arena, .{ .role = .user, .content = content });
        return messages.items;
    }

    pub fn parse(adapter: Adapter, arena: Allocator, view: View, completion: []const u8) Error!Record {
        return switch (adapter) {
            .chat => parseChat(arena, view, completion),
            .json => parseJson(arena, view, completion),
        };
    }

    /// Name used by DSPy for the adapter class.
    pub fn className(adapter: Adapter) []const u8 {
        return switch (adapter) {
            .chat => "ChatAdapter",
            .json => "JSONAdapter",
        };
    }
};

const incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied.";
const missing_incomplete = "Not supplied for this particular example. ";
const missing_complete = "Not supplied for this conversation history message. ";

fn systemMessage(adapter: Adapter, arena: Allocator, view: View) Error![]const u8 {
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

fn mapWriteError(err: pyjson.Error) Error {
    return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        error.JsonTooDeep => error.JsonTooDeep,
    };
}

/// `get_field_description_string`.
fn fieldDescriptions(w: *Writer, fields: []const Field) pyjson.Error!void {
    for (fields, 0..) |f, i| {
        if (i > 0) try w.writeByte('\n');
        try w.print("{d}. `{s}` ({s}): ", .{ i + 1, f.name, f.type_name });
        if (!isDefaultDesc(f)) try w.writeAll(f.desc);
    }
}

fn isDefaultDesc(f: Field) bool {
    return f.desc.len == f.name.len + 3 and std.mem.startsWith(u8, f.desc, "${") and
        std.mem.eql(u8, f.desc[2 .. 2 + f.name.len], f.name) and f.desc[f.desc.len - 1] == '}';
}

/// `translate_field_type`.
fn translateFieldType(w: *Writer, f: Field, is_input: bool) Writer.Error!void {
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
    var aw: Writer.Allocating = .init(arena);
    const sw = &aw.writer;
    sw.writeAll("All interactions will be structured in the following way, with the appropriate values filled in.\n\n") catch
        return error.OutOfMemory;
    switch (adapter) {
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
        try out.appendSlice(arena, line[if (margin) |m| m.len else 0 ..]);
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
            try messages.append(arena, .{ .role = .user, .content = try userMessage(adapter, arena, view, demo, prefix, false) });
            try messages.append(arena, .{ .role = .assistant, .content = try assistantMessage(adapter, arena, view, demo, missing) });
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
fn userMessage(adapter: Adapter, arena: Allocator, view: View, inputs: *const Record, prefix: []const u8, main_request: bool) Error![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    w.writeAll(prefix) catch return error.OutOfMemory;
    for (view.inputs) |f| {
        const v = inputs.get(f.name) orelse continue;
        w.print("\n\n[[ ## {s} ## ]]\n", .{f.name}) catch return error.OutOfMemory;
        value.formatFieldValue(w, f, v) catch |err| return mapWriteError(err);
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
        .chat => try w.writeAll("Respond with the corresponding output fields, starting with the field "),
        .json => try w.writeAll("Respond with a JSON object in the following order of fields: "),
    }
    for (view.outputs, 0..) |f, i| {
        if (i > 0) try w.writeAll(", then ");
        switch (adapter) {
            .chat => try w.print("`[[ ## {s} ## ]]`", .{f.name}),
            .json => try w.print("`{s}`", .{f.name}),
        }
        if (f.kind != .str) try w.print(" (must be formatted as a valid Python {s})", .{f.type_name});
    }
    switch (adapter) {
        .chat => try w.writeAll(", and then ending with the marker for `[[ ## completed ## ]]`."),
        .json => try w.writeAll("."),
    }
}

/// `format_assistant_message_content`.
fn assistantMessage(adapter: Adapter, arena: Allocator, view: View, outputs: *const Record, missing: []const u8) Error![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    const w = &aw.writer;
    switch (adapter) {
        .chat => {
            var fw: Writer.Allocating = .init(arena);
            for (view.outputs, 0..) |f, i| {
                if (i > 0) fw.writer.writeAll("\n\n") catch return error.OutOfMemory;
                fw.writer.print("[[ ## {s} ## ]]\n", .{f.name}) catch return error.OutOfMemory;
                const v = outputs.get(f.name) orelse Value{ .string = missing };
                value.formatFieldValue(&fw.writer, f, v) catch |err| return mapWriteError(err);
            }
            w.writeAll(value.pyStrip(fw.written())) catch return error.OutOfMemory;
            w.writeAll("\n\n[[ ## completed ## ]]\n") catch return error.OutOfMemory;
        },
        .json => {
            var record: Record = .empty;
            for (view.outputs) |f| {
                try record.put(arena, f.name, outputs.get(f.name) orelse Value{ .string = missing });
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
        const parsed = value.parseText(arena, field, value.pyStrip(joined)) catch |err| switch (err) {
            error.ParseValue => return error.AdapterParse,
            error.OutOfMemory => return error.OutOfMemory,
        };
        try fields.put(arena, field.name, parsed);
    }
    try requireAllOutputs(view, &fields);
    return fields;
}

fn requireAllOutputs(view: View, fields: *const Record) Error!void {
    if (fields.count() != view.outputs.len) return error.AdapterParse;
    for (view.outputs) |f| if (!fields.contains(f.name)) return error.AdapterParse;
}

fn parseJson(arena: Allocator, view: View, completion: []const u8) Error!Record {
    const object = parseObject(arena, completion) orelse blk: {
        const span = firstBalancedBraces(completion) orelse return error.AdapterParse;
        break :blk parseObject(arena, span) orelse return error.AdapterParse;
    };
    var fields: Record = .empty;
    var it = object.iterator();
    while (it.next()) |entry| {
        const field = view.findOutput(entry.key_ptr.*) orelse continue;
        const parsed = value.parseJsonValue(arena, field, entry.value_ptr.*) catch |err| switch (err) {
            error.ParseValue => return error.AdapterParse,
            error.OutOfMemory => return error.OutOfMemory,
        };
        try fields.put(arena, field.name, parsed);
    }
    try requireAllOutputs(view, &fields);
    return fields;
}

fn parseObject(arena: Allocator, text: []const u8) ?Record {
    const v = std.json.parseFromSliceLeaky(Value, arena, value.pyStrip(text), .{
        .duplicate_field_behavior = .use_last,
    }) catch return null;
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
    try testing.expectEqualStrings("answer", j.keys()[0]);
}
