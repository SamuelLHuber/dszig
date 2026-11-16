//! Reusable DSPy 3.5 structured conversation History. Events are ordered
//! signature-field records, not flattened prompts. Values borrow their source
//! storage; clone into a longer-lived allocator before releasing a call arena.
//! Text/native function-tool replay includes per-call continuation. REPL events
//! and media parts are outside this native JSON-value representation.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const adapter = @import("adapter.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const types = @import("types.zig");
const tool = @import("tool.zig");
const pyjson = @import("pyjson.zig");
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
