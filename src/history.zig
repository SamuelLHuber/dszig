//! Reusable DSPy 3.5 structured conversation History. Events are ordered
//! signature-field records, not flattened prompts. Values borrow their source
//! storage; clone into a longer-lived allocator before releasing a call arena.
//! Text/native function-tool replay includes per-call continuation. Tagged
//! DSPy REPL entries can be replayed as text without executing their code.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const adapter = @import("adapter.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const types = @import("types.zig");
const tool = @import("tool.zig");
const pyjson = @import("pyjson.zig");
const message_content = @import("message_content.zig");
const Record = value.Record;

pub const History = struct {
    messages: []const Record = &.{},

    pub const schema =
        \\{"type": "object", "additionalProperties": false, "properties": {"messages": {"type": "array", "items": {"type": "object", "additionalProperties": true}, "title": "Messages"}}, "required": ["messages"], "title": "History"}
    ;

    pub fn validate(arena: Allocator, input: Value) types.Error!Value {
        const history = fromValue(arena, input) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        };
        return history.toValue(arena);
    }

    pub fn fromValue(arena: Allocator, input: Value) adapter.Error!History {
        if (input != .object or input.object.count() != 1) return error.AdapterParse;
        const items = input.object.get("messages") orelse return error.AdapterParse;
        if (items != .array) return error.AdapterParse;
        const messages = try arena.alloc(Record, items.array.items.len);
        for (items.array.items, messages) |item, *message| {
            if (item != .object) return error.AdapterParse;
            message.* = item.object;
        }
        return .{ .messages = messages };
    }

    pub fn toValue(self: History, arena: Allocator) Allocator.Error!Value {
        var messages: @import("runtime_value.zig").Array = .init(arena);
        try messages.ensureTotalCapacity(self.messages.len);
        for (self.messages) |message| messages.appendAssumeCapacity(.{ .object = message });
        var out: Record = .empty;
        try out.put(arena, "messages", .{ .array = messages });
        return .{ .object = out };
    }

    pub fn clone(self: History, arena: Allocator) Allocator.Error!History {
        const messages = try arena.alloc(Record, self.messages.len);
        for (self.messages, messages) |*message, *out| out.* = try value.cloneRecord(arena, message);
        return .{ .messages = messages };
    }

    /// History owns message ordering; the selected adapter owns field syntax.
    /// The view excludes the History field itself, as in DSPy History.format.
    pub fn format(self: History, selected_adapter: adapter.Adapter, arena: Allocator, view: adapter.View) adapter.Error![]adapter.Message {
        var messages: std.ArrayList(adapter.Message) = .empty;
        for (self.messages) |*event| {
            if (try appendReplEvent(arena, selected_adapter, view, event, &messages)) continue;
            var main_request = false;
            for (view.inputs) |field| {
                if (event.contains(field.name)) {
                    main_request = true;
                    break;
                }
            }
            const user_content = try adapter.formatUser(selected_adapter, arena, view, event, main_request);
            if (user_content.len > 0) try messages.append(arena, .{ .role = .user, .content = try adapter.renderedContent(arena, user_content) });

            var calls_field: ?[]const u8 = null;
            var recorded_calls: ?tool.ToolCalls = null;
            for (event.keys(), event.values()) |name, field_value| {
                if (field_value != .object or !field_value.object.contains("tool_calls")) continue;
                recorded_calls = tool.ToolCalls.fromValue(arena, field_value) catch |err| return mapParseError(err);
                calls_field = name;
                break;
            }
            if (view.use_native_function_calling) {
                if (recorded_calls) |calls| {
                    try appendNativeEvent(arena, view, selected_adapter, event, calls, &messages);
                    continue;
                }
            }
            var assistant_values = event.*;
            var results: ?tool.ToolCallResults = null;
            if (recorded_calls) |recorded| {
                var calls = recorded;
                results = calls.tool_call_results;
                calls.tool_call_results = null;
                assistant_values = try event.clone(arena);
                try assistant_values.put(arena, calls_field.?, try calls.formatValue(arena));
            }
            const assistant_content = try adapter.formatAssistant(selected_adapter, arena, view, &assistant_values);
            if (assistant_content.len > 0) try messages.append(arena, .{ .role = .assistant, .content = .{ .text = assistant_content } });

            if (results) |tool_results| {
                var fields: Record = .empty;
                try fields.put(arena, "tool_call_results", try tool_results.toValue(arena));
                const result_field: signature.Field = .{
                    .name = "tool_call_results",
                    .kind = .json_object,
                    .type_name = "ToolCallResults",
                    .desc = "${tool_call_results}",
                    .prefix = "Tool Call Results:",
                    .xml_literal = true,
                };
                const content = try adapter.formatUser(selected_adapter, arena, .{
                    .instructions = "",
                    .inputs = &.{result_field},
                    .outputs = &.{},
                }, &fields, false);
                if (content.len > 0) try messages.append(arena, .{ .role = .user, .content = try adapter.renderedContent(arena, content) });
            }
        }
        return messages.items;
    }
};

fn appendReplEvent(
    arena: Allocator,
    selected: adapter.Adapter,
    view: adapter.View,
    event: *const Record,
    messages: *std.ArrayList(adapter.Message),
) adapter.Error!bool {
    const entry_index = event.getIndex("repl_entry") orelse return false;
    const entry = event.values()[entry_index];
    if (entry != .object) return false;
    const tag = entry.object.get("__dspy_repl__") orelse return false;
    if (tag != .string) return false;
    if (std.mem.eql(u8, tag.string, "extract_fallback")) {
        try appendReplFallback(arena, selected, event, entry_index + 1, messages);
        return true;
    }
    if (!std.mem.eql(u8, tag.string, "entry")) return false;

    const code = try replString(&entry.object, "code", null);
    const output = try replString(&entry.object, "output", null);
    const reasoning = try replString(&entry.object, "reasoning", "");
    const limit = try replOutputLimit(&entry.object);

    var inputs: Record = .empty;
    var main_request = false;
    for (view.inputs) |field| {
        const index = event.getIndex(field.name) orelse continue;
        if (index >= entry_index) continue;
        try inputs.put(arena, field.name, event.values()[index]);
        main_request = true;
    }
    const user = try adapter.formatUser(selected, arena, view, &inputs, main_request);
    if (user.len != 0) try messages.append(arena, .{
        .role = .user,
        .content = try adapter.renderedContent(arena, user),
    });

    var fields: [2]signature.Field = undefined;
    var action: Record = .empty;
    var count: usize = 0;
    if (replReasoningName(view.outputs)) |name| {
        fields[count] = replField(name);
        count += 1;
        try action.put(arena, name, .{ .string = reasoning });
    }
    fields[count] = replField("code");
    try action.put(arena, "code", .{ .string = code });
    const text = try adapter.formatAssistant(selected, arena, .{
        .instructions = "",
        .inputs = &.{},
        .outputs = fields[0 .. count + 1],
    }, &action);
    try appendReplText(arena, messages, .assistant, text, false);

    var observation: Record = .empty;
    try observation.put(arena, "repl_output", .{
        .string = try truncateReplOutput(arena, output, limit),
    });
    const rendered = try adapter.formatUser(selected, arena, .{
        .instructions = "",
        .inputs = &.{replField("repl_output")},
        .outputs = &.{},
    }, &observation, false);
    try appendReplText(arena, messages, .user, rendered, false);
    return true;
}

/// Compact a tagged entry for a reflection record. Keep its original rendering
/// limit: DSPy changes the stored observation here, not the replay setting.
pub fn compactReplEntry(
    arena: Allocator,
    entry: *const Record,
    max_chars: i64,
) value.ParseError!?Value {
    const tag = entry.get("__dspy_repl__").?;
    std.debug.assert(tag == .string);
    std.debug.assert(std.mem.eql(u8, tag.string, "entry"));
    const output = replString(entry, "output", null) catch return error.ParseValue;
    const limit = replOutputLimit(entry) catch return error.ParseValue;
    const compacted = truncateReplOutput(arena, output, @min(limit, max_chars)) catch |err|
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ParseValue,
        };
    if (compacted.len == output.len and compacted.ptr == output.ptr) return null;
    var record = try entry.clone(arena);
    try record.put(arena, "output", .{ .string = compacted });
    return .{ .object = record };
}

fn replOutputLimit(entry: *const Record) adapter.Error!i64 {
    const limit = entry.get("max_output_chars") orelse return 10_000;
    if (limit != .integer) return error.AdapterParse;
    return limit.integer;
}

fn replString(record: *const Record, name: []const u8, default: ?[]const u8) adapter.Error![]const u8 {
    const item = record.get(name) orelse return default orelse error.AdapterParse;
    if (item == .null) return default orelse error.AdapterParse;
    if (item != .string) return error.AdapterParse;
    return item.string;
}

fn replField(name: []const u8) signature.Field {
    return .{ .name = name, .kind = .str, .type_name = "str", .desc = "", .prefix = "" };
}

fn replReasoningName(fields: []const signature.Field) ?[]const u8 {
    for (fields) |field| {
        if (field.reasoning_fallback != null) return field.name;
    }
    for (fields) |field| if (std.mem.eql(u8, field.name, "reasoning")) return "reasoning";
    for (fields) |field| if (std.mem.eql(u8, field.name, "code")) return null;
    return "reasoning";
}

fn appendReplFallback(
    arena: Allocator,
    selected: adapter.Adapter,
    event: *const Record,
    first: usize,
    messages: *std.ArrayList(adapter.Message),
) adapter.Error!void {
    const names = event.keys()[first..];
    if (names.len == 0) return;
    const fields = try arena.alloc(signature.Field, names.len);
    for (names, fields) |name, *field| field.* = replField(name);
    const rendered = try adapter.formatAssistant(selected, arena, .{
        .instructions = "",
        .inputs = &.{},
        .outputs = fields,
    }, event);
    try appendReplText(arena, messages, .assistant, rendered, true);
}

fn appendReplText(
    arena: Allocator,
    messages: *std.ArrayList(adapter.Message),
    role: @FieldType(adapter.Message, "role"),
    text: []const u8,
    omit_media: bool,
) adapter.Error!void {
    // These turns are REPL text even if generated code prints a valid marker.
    const escaped = try escapeReplText(arena, text, omit_media);
    try messages.append(arena, .{ .role = role, .content = .{ .text = escaped } });
}

pub fn escapeReplText(arena: Allocator, text: []const u8, omit_media: bool) Allocator.Error![]const u8 {
    const start = message_content.start_marker;
    const end = message_content.end_marker;
    var out: std.Io.Writer.Allocating = .init(arena);
    var cursor: usize = 0;
    var search_media_end = omit_media;
    for (text, 0..) |byte, index| {
        if (index < cursor or byte != '<') continue;
        const rest = text[index..];
        if (std.mem.startsWith(u8, rest, start)) {
            out.writer.writeAll(text[cursor..index]) catch return error.OutOfMemory;
            cursor = index + start.len;
            if (search_media_end) {
                if (std.mem.indexOf(u8, text[cursor..], end)) |offset| {
                    out.writer.writeAll("<media omitted from replay>") catch
                        return error.OutOfMemory;
                    cursor += offset + end.len;
                    continue;
                }
                // No later start can have a closing delimiter either.
                search_media_end = false;
            }
            out.writer.writeAll("[CUSTOM-TYPE-START]") catch return error.OutOfMemory;
        } else if (std.mem.startsWith(u8, rest, end)) {
            out.writer.writeAll(text[cursor..index]) catch return error.OutOfMemory;
            out.writer.writeAll("[CUSTOM-TYPE-END]") catch return error.OutOfMemory;
            cursor = index + end.len;
        }
    }
    if (cursor == 0) return text;
    out.writer.writeAll(text[cursor..]) catch return error.OutOfMemory;
    return out.written();
}

pub fn truncateReplOutput(arena: Allocator, text: []const u8, limit: i64) adapter.Error![]const u8 {
    if (limit >= 0 and text.len <= @as(u64, @intCast(limit))) return text;
    const utf8 = std.unicode.Utf8View.init(text) catch return error.AdapterParse;
    const count = std.unicode.utf8CountCodepoints(text) catch return error.AdapterParse;
    const length: i128 = count;
    if (length <= limit) return text;

    const half: i128 = @divFloor(limit, 2);
    const head_end: usize = @intCast(if (half < 0) @max(0, length + half) else half);
    // Negative limits use a positive slice start; -0 starts at zero.
    const tail_start: usize = @intCast(if (limit < 0)
        @min(length, -half)
    else if (limit == 0)
        0
    else
        length - (limit - half));

    var iter = utf8.iterator();
    for (0..@min(head_end, tail_start)) |_| _ = iter.nextCodepointSlice().?;
    const first = iter.i;
    for (@min(head_end, tail_start)..@max(head_end, tail_start)) |_| {
        _ = iter.nextCodepointSlice().?;
    }
    const head_byte = if (head_end <= tail_start) first else iter.i;
    const tail_byte = if (tail_start <= head_end) first else iter.i;

    var grouped_buffer: [51]u8 = undefined;
    const omitted = pyjson.groupedDecimal(&grouped_buffer, @intCast(length - limit));
    return std.mem.concat(arena, u8, &.{
        text[0..head_byte],
        "\n\n... (",
        omitted,
        " characters omitted) ...\n\n",
        text[tail_byte..],
    });
}

fn appendNativeEvent(
    arena: Allocator,
    view: adapter.View,
    selected_adapter: adapter.Adapter,
    event: *const Record,
    calls: tool.ToolCalls,
    messages: *std.ArrayList(adapter.Message),
) adapter.Error!void {
    const fields = try arena.alloc(signature.Field, view.outputs.len);
    var count: usize = 0;
    for (view.outputs) |field| {
        if (field.native_tool_calls) continue;
        const item = event.get(field.name) orelse continue;
        if (item == .null) continue;
        fields[count] = field;
        count += 1;
    }
    var content_view = view;
    content_view.outputs = fields[0..count];
    const content = if (count == 0) "" else try adapter.formatAssistant(selected_adapter, arena, content_view, event);
    var valid_results = false;
    if (calls.tool_call_results) |results| {
        valid_results = calls.tool_calls.len == results.tool_call_results.len;
        if (valid_results) {
            for (calls.tool_calls, results.tool_call_results) |call, result| {
                const call_id = call.id orelse {
                    valid_results = false;
                    break;
                };
                const result_id = result.call_id orelse {
                    valid_results = false;
                    break;
                };
                if (call_id.len == 0 or !std.mem.eql(u8, call_id, result_id)) {
                    valid_results = false;
                    break;
                }
            }
        }
    }
    if (content.len > 0 or valid_results) try messages.append(arena, .{
        .role = .assistant,
        .content = .{ .text = content },
        .tool_calls = if (valid_results) tool.callsToNative(arena, calls.tool_calls) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.ParseValue => error.AdapterParse,
            error.JsonTooDeep => error.JsonTooDeep,
        } else &.{},
    });
    if (valid_results) {
        for (calls.tool_call_results.?.tool_call_results) |result| {
            const serialized = value.serializedValue(arena, result.value) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.ParseValue => error.AdapterParse,
            };
            const bare_text = result.value == .string or
                (result.value == .custom and result.value.custom.vtable.is_type and serialized == .string);
            const text = if (bare_text) serialized.string else blk: {
                var out: std.Io.Writer.Allocating = .init(arena);
                pyjson.writeJson(&out.writer, serialized, .compact) catch |err| return switch (err) {
                    error.WriteFailed => error.OutOfMemory,
                    error.JsonTooDeep => error.JsonTooDeep,
                    error.ParseValue => error.AdapterParse,
                };
                break :blk out.written();
            };
            // The supported OpenAI tool-row formats have no error flag;
            // lm15 carries the recorded status as this exact text prefix.
            const result_content = if (result.is_error) try std.fmt.allocPrint(arena, "[error] {s}", .{text}) else text;
            try messages.append(arena, .{ .role = .tool, .content = try adapter.renderedContent(arena, result_content), .tool_call_id = result.call_id });
        }
    }
}

fn mapParseError(err: types.Error) adapter.Error {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.ParseValue => error.AdapterParse,
    };
}
