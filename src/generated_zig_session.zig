//! Pure host-side protocol validation, not an OS sandbox or process supervisor.
//! Construct in place before starting the worker. All storage is fixed; no method allocates.
const std = @import("std");
const protocol = @import("generated_zig_protocol.zig");
const assert = std.debug.assert;

pub const tools_max: u16 = 256;
pub const json_depth_max: u8 = 64;

pub const Options = struct {
    tools: []const []const u8 = &.{},
    max_calls: u32 = 64,
    max_frame_bytes: u64 = 4 * 1024 * 1024,
    /// JSON bytes are charged twice (UTF-8 and syntax scans), and name comparisons
    /// charge their full candidate length. This bounds validation/dispatch work.
    max_work_bytes: u64 = 8 * 1024 * 1024,
    max_json_depth: u8 = json_depth_max,
};

pub const Error = protocol.Error || error{
    InvalidOptions,
    DuplicateToolName,
    InvalidState,
    SessionFailed,
    UnexpectedFrame,
    GuestFailure,
    UnknownTool,
    CallLimitExceeded,
    FrameBudgetExceeded,
    WorkBudgetExceeded,
    InvalidJson,
    JsonDepthExceeded,
    ChildFailed,
    MissingFinish,
};

pub const Event = union(enum) {
    /// Argument borrows the frame until the transport reuses it. Dispatch synchronously.
    tool_call: struct { index: u16, argument: []const u8 },
    /// Validated but NOT published. Borrows Session storage, not transport storage.
    finish_pending: []const u8,
};

pub const State = enum { ready, running, waiting_reply, finish_pending, published, failed };

pub const Session = struct {
    state: State = .failed,
    names: [tools_max][protocol.tool_name_bytes_max]u8 = undefined,
    name_lengths: [tools_max]u16 = undefined,
    tool_count: u16 = 0,
    calls: u32 = 0,
    max_calls: u32 = 0,
    frame_bytes_left: u64 = 0,
    work_bytes_left: u64 = 0,
    depth_limit: u8 = 0,
    finish_buffer: [protocol.payload_bytes_max]u8 = undefined,
    finish_bytes: u32 = 0,

    /// Copies names once: the caller may reuse all Options backing after initialization.
    /// An initialization error leaves the session failed, including on reinitialization.
    pub fn init(session: *Session, options: *const Options) Error!void {
        session.* = .{};
        if (options.tools.len > tools_max) return error.InvalidOptions;
        if (options.max_json_depth == 0) return error.InvalidOptions;
        if (options.max_json_depth > json_depth_max) return error.InvalidOptions;
        if (options.max_frame_bytes < protocol.header_bytes) return error.InvalidOptions;
        if (options.max_work_bytes == 0) return error.InvalidOptions;
        for (options.tools, 0..) |name, index| {
            if (name.len == 0) return error.InvalidToolName;
            if (name.len > protocol.tool_name_bytes_max) return error.InvalidToolName;
            // Names are opaque exact bytes, matching protocol.ToolCall's name contract.
            for (0..index) |previous| {
                if (std.mem.eql(u8, name, session.toolName(previous))) {
                    return error.DuplicateToolName;
                }
            }
            @memcpy(session.names[index][0..name.len], name);
            session.name_lengths[index] = @intCast(name.len);
        }
        session.tool_count = @intCast(options.tools.len);
        session.max_calls = options.max_calls;
        session.frame_bytes_left = options.max_frame_bytes;
        session.work_bytes_left = options.max_work_bytes;
        session.depth_limit = options.max_json_depth;
        session.state = .ready;
        assert(session.finish_bytes == 0);
        assert(session.calls == 0);
    }

    /// Validate before sending an input frame. Transport send failures MUST call abort.
    pub fn begin(session: *Session, input_json: []const u8) Error!void {
        errdefer session.abort();
        try session.expect(.ready);
        try session.chargeFrame(input_json);
        try session.checkJson(input_json);
        session.state = .running;
    }

    /// The transport must decode/validate the complete header before this call, and
    /// MUST abort on malformed headers, partial frames, EOF, deadlines or I/O failure.
    pub fn receive(session: *Session, kind: protocol.Kind, payload: []const u8) Error!Event {
        errdefer session.abort();
        try session.expect(.running);
        try session.chargeFrame(payload);
        switch (kind) {
            .tool_call => {
                if (session.calls == session.max_calls) return error.CallLimitExceeded;
                const call = try protocol.ToolCall.decode(payload);
                const index = try session.lookup(call.name);
                try session.checkJson(call.argument);
                session.calls += 1;
                session.state = .waiting_reply;
                return .{ .tool_call = .{ .index = index, .argument = call.argument } };
            },
            .finish => {
                try session.checkJson(payload);
                @memcpy(session.finish_buffer[0..payload.len], payload);
                session.finish_bytes = @intCast(payload.len);
                session.state = .finish_pending;
                return .{ .finish_pending = session.finish_buffer[0..session.finish_bytes] };
            },
            .failure => return error.GuestFailure,
            .input, .tool_reply => return error.UnexpectedFrame,
        }
    }

    /// Validate before sending a reply frame. Callback or send failure MUST abort.
    pub fn reply(session: *Session, reply_json: []const u8) Error!void {
        errdefer session.abort();
        try session.expect(.waiting_reply);
        try session.chargeFrame(reply_json);
        try session.checkJson(reply_json);
        session.state = .running;
    }

    pub fn reject(session: *Session) void {
        session.abort();
    }

    /// Revokes pending output. The supervisor remains responsible for killing and
    /// joining the entire process tree; this pure module cannot do either.
    pub fn abort(session: *Session) void {
        session.state = .failed;
        session.finish_bytes = 0;
    }

    /// TRUST PRECONDITION: caller has joined the WHOLE process tree, drained the
    /// protocol stream through EOF (including any trailing/partial frame), and
    /// observed no transport/policy/deadline failure. Merely waitpid'ing the root
    /// is insufficient. Call abort on any such failure, never call joined instead.
    /// This explicit notification is the sole publication point. Returned output
    /// remains in Session storage until reinit; do not move/reuse Session meanwhile.
    pub fn joined(session: *Session, exit_success: bool) Error![]const u8 {
        errdefer session.abort();
        if (session.state == .failed) return error.SessionFailed;
        if (!exit_success) return error.ChildFailed;
        if (session.state != .finish_pending) return error.MissingFinish;
        assert(session.finish_bytes > 0);
        assert(session.finish_bytes <= protocol.payload_bytes_max);
        session.state = .published;
        return session.finish_buffer[0..session.finish_bytes];
    }

    fn expect(session: *const Session, expected: State) Error!void {
        if (session.state == .failed) return error.SessionFailed;
        if (session.state != expected) return error.InvalidState;
    }

    fn toolName(session: *const Session, index: usize) []const u8 {
        assert(index < tools_max);
        assert(session.name_lengths[index] <= protocol.tool_name_bytes_max);
        return session.names[index][0..session.name_lengths[index]];
    }

    fn lookup(session: *Session, candidate: []const u8) Error!u16 {
        for (0..session.tool_count) |index| {
            const registered = session.toolName(index);
            try session.chargeWork(registered.len);
            if (std.mem.eql(u8, candidate, registered)) return @intCast(index);
        }
        return error.UnknownTool;
    }

    fn chargeFrame(session: *Session, payload: []const u8) Error!void {
        if (payload.len > protocol.payload_bytes_max) return error.FrameCapacityExceeded;
        const bytes = protocol.header_bytes + @as(u64, @intCast(payload.len));
        if (bytes > session.frame_bytes_left) return error.FrameBudgetExceeded;
        session.frame_bytes_left -= bytes;
    }

    fn chargeWork(session: *Session, bytes: u64) Error!void {
        if (bytes > session.work_bytes_left) return error.WorkBudgetExceeded;
        session.work_bytes_left -= bytes;
    }

    fn checkJson(session: *Session, bytes: []const u8) Error!void {
        assert(bytes.len <= protocol.payload_bytes_max);
        try session.chargeWork(@as(u64, @intCast(bytes.len)) * 2);
        try validateJson(bytes, session.depth_limit);
    }
};

const JsonState = enum {
    root_value,
    root_end,
    array_first,
    array_value,
    array_after,
    object_first,
    object_key,
    object_colon,
    object_value,
    object_after,
};

/// Each container consumes one fixed stack slot. The root scalar consumes none.
/// UTF-8 and syntax each scan at most the payload length; there is no recursion.
fn validateJson(bytes: []const u8, depth_limit: u8) Error!void {
    if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidJson;
    var states: [json_depth_max + 1]JsonState = undefined;
    states[0] = .root_value;
    var depth: u8 = 0;
    var position: u32 = 0;
    while (true) {
        skipWhitespace(bytes, &position);
        const state = states[depth];
        if (state == .root_end) {
            if (position != bytes.len) return error.InvalidJson;
            return;
        }
        if (position == bytes.len) return error.InvalidJson;
        const byte = bytes[position];
        switch (state) {
            .root_value, .array_value, .object_value => {
                states[depth] = switch (state) {
                    .root_value => .root_end,
                    .array_value => .array_after,
                    .object_value => .object_after,
                    else => unreachable,
                };
                try jsonValue(bytes, &position, &states, &depth, depth_limit);
            },
            .array_first => {
                if (byte == ']') {
                    depth -= 1;
                    position += 1;
                } else states[depth] = .array_value;
            },
            .object_first, .object_key => {
                if (state == .object_first and byte == '}') {
                    depth -= 1;
                    position += 1;
                } else {
                    if (byte != '"') return error.InvalidJson;
                    try jsonString(bytes, &position);
                    states[depth] = .object_colon;
                }
            },
            .object_colon => {
                if (byte != ':') return error.InvalidJson;
                position += 1;
                states[depth] = .object_value;
            },
            .array_after, .object_after => {
                const array = state == .array_after;
                if (byte == (if (array) @as(u8, ']') else @as(u8, '}'))) {
                    depth -= 1;
                } else if (byte == ',') {
                    states[depth] = if (array) .array_value else .object_key;
                } else return error.InvalidJson;
                position += 1;
            },
            .root_end => unreachable,
        }
    }
}

fn skipWhitespace(bytes: []const u8, position: *u32) void {
    while (position.* < bytes.len) : (position.* += 1) {
        switch (bytes[position.*]) {
            ' ', '\t', '\n', '\r' => {},
            else => return,
        }
    }
}

fn jsonValue(
    bytes: []const u8,
    position: *u32,
    states: *[json_depth_max + 1]JsonState,
    depth: *u8,
    depth_limit: u8,
) Error!void {
    assert(position.* < bytes.len);
    switch (bytes[position.*]) {
        '[', '{' => |opening| {
            if (depth.* == depth_limit) return error.JsonDepthExceeded;
            depth.* += 1;
            states[depth.*] = if (opening == '[') .array_first else .object_first;
            position.* += 1;
        },
        '"' => try jsonString(bytes, position),
        't' => try jsonLiteral(bytes, position, "true"),
        'f' => try jsonLiteral(bytes, position, "false"),
        'n' => try jsonLiteral(bytes, position, "null"),
        '-', '0'...'9' => try jsonNumber(bytes, position),
        else => return error.InvalidJson,
    }
}

fn jsonLiteral(bytes: []const u8, position: *u32, literal: []const u8) Error!void {
    if (literal.len > bytes.len - position.*) return error.InvalidJson;
    if (!std.mem.eql(u8, bytes[position.*..][0..literal.len], literal)) {
        return error.InvalidJson;
    }
    position.* += @intCast(literal.len);
}

fn jsonString(bytes: []const u8, position: *u32) Error!void {
    assert(bytes[position.*] == '"');
    position.* += 1;
    while (position.* < bytes.len) {
        const byte = bytes[position.*];
        position.* += 1;
        if (byte == '"') return;
        if (byte < 0x20) return error.InvalidJson;
        if (byte != '\\') continue;
        if (position.* == bytes.len) return error.InvalidJson;
        const escape = bytes[position.*];
        position.* += 1;
        switch (escape) {
            '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {},
            'u' => {
                const code = try jsonHex(bytes, position);
                if (code >= 0xD800 and code <= 0xDBFF) {
                    if (bytes.len - position.* < 6) return error.InvalidJson;
                    if (bytes[position.*] != '\\') return error.InvalidJson;
                    if (bytes[position.* + 1] != 'u') return error.InvalidJson;
                    position.* += 2;
                    const low = try jsonHex(bytes, position);
                    if (low < 0xDC00 or low > 0xDFFF) return error.InvalidJson;
                } else if (code >= 0xDC00 and code <= 0xDFFF) return error.InvalidJson;
            },
            else => return error.InvalidJson,
        }
    }
    return error.InvalidJson;
}

fn jsonHex(bytes: []const u8, position: *u32) Error!u16 {
    if (bytes.len - position.* < 4) return error.InvalidJson;
    var code: u16 = 0;
    for (0..4) |_| {
        const digit: u16 = switch (bytes[position.*]) {
            '0'...'9' => |byte| byte - '0',
            'a'...'f' => |byte| byte - 'a' + 10,
            'A'...'F' => |byte| byte - 'A' + 10,
            else => return error.InvalidJson,
        };
        code = (code << 4) | digit;
        position.* += 1;
    }
    return code;
}

fn jsonNumber(bytes: []const u8, position: *u32) Error!void {
    if (bytes[position.*] == '-') position.* += 1;
    if (position.* == bytes.len) return error.InvalidJson;
    if (bytes[position.*] == '0') {
        position.* += 1;
    } else {
        if (bytes[position.*] < '1' or bytes[position.*] > '9') return error.InvalidJson;
        jsonDigits(bytes, position);
    }
    if (position.* < bytes.len and bytes[position.*] == '.') {
        position.* += 1;
        const start = position.*;
        jsonDigits(bytes, position);
        if (position.* == start) return error.InvalidJson;
    }
    if (position.* < bytes.len and
        (bytes[position.*] == 'e' or bytes[position.*] == 'E'))
    {
        position.* += 1;
        if (position.* < bytes.len and
            (bytes[position.*] == '+' or bytes[position.*] == '-'))
        {
            position.* += 1;
        }
        const start = position.*;
        jsonDigits(bytes, position);
        if (position.* == start) return error.InvalidJson;
    }
}

fn jsonDigits(bytes: []const u8, position: *u32) void {
    while (position.* < bytes.len) : (position.* += 1) {
        if (bytes[position.*] < '0' or bytes[position.*] > '9') return;
    }
}

test "finish is retained but published only after explicit successful tree join" {
    var session: Session = undefined;
    try session.init(&.{ .max_calls = 0 });
    try session.begin("{}");
    var frame = [_]u8{ '[', '1', ']' };
    const event = try session.receive(.finish, &frame);
    try std.testing.expectEqual(State.finish_pending, session.state);
    @memset(&frame, 'x');
    try std.testing.expectEqualStrings("[1]", event.finish_pending);
    try std.testing.expectEqualStrings("[1]", try session.joined(true));
    try std.testing.expectEqual(State.published, session.state);
}

test "tool calls use exact startup indices and require validated replies" {
    var session: Session = undefined;
    var mutable_name = [_]u8{'a'};
    try session.init(&.{ .tools = &.{ &mutable_name, "B" }, .max_calls = 1 });
    mutable_name[0] = 'x';
    try session.begin("null");
    var frame: [128]u8 = undefined;
    const payload = try (protocol.ToolCall{ .name = "a", .argument = "[true]" })
        .encode(&frame);
    const event = try session.receive(.tool_call, payload);
    try std.testing.expectEqual(@as(u16, 0), event.tool_call.index);
    try std.testing.expectEqualStrings("[true]", event.tool_call.argument);
    try session.reply("{\"ok\":true}");
    try std.testing.expectError(error.CallLimitExceeded, session.receive(.tool_call, payload));
    try std.testing.expectError(error.SessionFailed, session.receive(.finish, "0"));
}

test "invalid transitions and unsuccessful exits revoke pending finish" {
    var session: Session = undefined;
    const invalid_kinds = [_]protocol.Kind{ .input, .tool_reply, .failure };
    for (invalid_kinds) |kind| {
        try session.init(&.{});
        try session.begin("0");
        if (session.receive(kind, "0")) |_| return error.TestUnexpectedResult else |_| {}
        try std.testing.expectEqual(State.failed, session.state);
    }
    try session.init(&.{});
    try std.testing.expectError(error.MissingFinish, session.joined(true));
    try session.init(&.{});
    try session.begin("0");
    _ = try session.receive(.finish, "0");
    try std.testing.expectError(error.ChildFailed, session.joined(false));
    try std.testing.expectError(error.SessionFailed, session.joined(true));
    try session.init(&.{});
    try session.begin("0");
    _ = try session.receive(.finish, "0");
    try std.testing.expectError(error.InvalidState, session.receive(.finish, "1"));
    try std.testing.expectError(error.SessionFailed, session.joined(true));
    try session.init(&.{});
    try session.begin("0");
    _ = try session.receive(.finish, "0");
    session.abort();
    try std.testing.expectError(error.SessionFailed, session.joined(true));
}

test "waiting for reply rejects another call or finish and malformed reply" {
    var session: Session = undefined;
    var frame: [32]u8 = undefined;
    const call = try (protocol.ToolCall{ .name = "t", .argument = "0" }).encode(&frame);
    for ([_]protocol.Kind{ .tool_call, .finish }) |kind| {
        try session.init(&.{ .tools = &.{"t"} });
        try session.begin("0");
        _ = try session.receive(.tool_call, call);
        try std.testing.expectError(error.InvalidState, session.receive(kind, call));
        try std.testing.expectEqual(State.failed, session.state);
    }
    try session.init(&.{ .tools = &.{"t"} });
    try session.begin("0");
    _ = try session.receive(.tool_call, call);
    try std.testing.expectError(error.InvalidJson, session.reply("true false"));
    try std.testing.expectEqual(State.failed, session.state);
}

test "JSON grammar unicode and depth boundaries" {
    const valid = [_][]const u8{
        "null",
        " true \r\n",
        "-0.125e+9",
        "[[],{},false]",
        "{\"a\": [1,2],\"a\":0}",
        "\"é😀\\uD83D\\uDE00\\n\"",
    };
    for (valid) |json| try validateJson(json, json_depth_max);
    const invalid = [_][]const u8{
        "",
        " ",
        "nullx",
        "true false",
        "01",
        "-",
        "1.",
        "1e",
        "+1",
        "NaN",
        "[1,]",
        "{\"a\":1,}",
        "{a:1}",
        "[}",
        "[1 2]",
        "{\"a\" 1}",
        "\"\\uD800\"",
        "\"\\uDC00\"",
        "\"\\uD800\\u0000\"",
        "\"\\x\"",
        "\"\n\"",
        "\"\xc0\x80\"",
        "\"\xed\xa0\x80\"",
        "\"unterminated",
    };
    for (invalid) |json| try std.testing.expectError(error.InvalidJson, validateJson(json, 64));
    var nested: [json_depth_max * 2]u8 = undefined;
    @memset(nested[0..json_depth_max], '[');
    @memset(nested[json_depth_max..], ']');
    try validateJson(&nested, json_depth_max);
    try std.testing.expectError(error.JsonDepthExceeded, validateJson(&nested, 63));
    try validateJson("0", 1);
}

test "startup validation exact names and malformed tool arguments" {
    var session: Session = undefined;
    try std.testing.expectError(error.InvalidToolName, session.init(&.{ .tools = &.{""} }));
    const duplicates = Options{ .tools = &.{ "a", "a" } };
    try std.testing.expectError(error.DuplicateToolName, session.init(&duplicates));
    try std.testing.expectEqual(State.failed, session.state);
    try std.testing.expectError(error.InvalidOptions, session.init(&.{ .max_json_depth = 65 }));
    var frame: [64]u8 = undefined;
    try session.init(&.{ .tools = &.{"a"} });
    try session.begin("0");
    const unknown = try (protocol.ToolCall{ .name = "A", .argument = "0" }).encode(&frame);
    try std.testing.expectError(error.UnknownTool, session.receive(.tool_call, unknown));
    try session.init(&.{ .tools = &.{"a"} });
    try session.begin("0");
    const malformed = try (protocol.ToolCall{ .name = "a", .argument = "[1,]" }).encode(&frame);
    try std.testing.expectError(error.InvalidJson, session.receive(.tool_call, malformed));
}

test "wire and work budgets include input reply and complete frame overhead" {
    var session: Session = undefined;
    try session.init(&.{ .max_frame_bytes = 18 });
    try session.begin("0");
    _ = try session.receive(.finish, "0");
    _ = try session.joined(true);
    try session.init(&.{ .max_frame_bytes = 17 });
    try session.begin("0");
    try std.testing.expectError(error.FrameBudgetExceeded, session.receive(.finish, "0"));
    try session.init(&.{ .max_work_bytes = 4 });
    try session.begin("0");
    _ = try session.receive(.finish, "0");
    try session.init(&.{ .max_work_bytes = 3 });
    try session.begin("0");
    try std.testing.expectError(error.WorkBudgetExceeded, session.receive(.finish, "0"));
    try session.init(&.{ .tools = &.{"a"}, .max_frame_bytes = 29 });
    try session.begin("0");
    var frame: [32]u8 = undefined;
    const call = try (protocol.ToolCall{ .name = "a", .argument = "0" }).encode(&frame);
    _ = try session.receive(.tool_call, call);
    try std.testing.expectError(error.FrameBudgetExceeded, session.reply("0"));
    try session.init(&.{});
    var large: [protocol.payload_bytes_max + 1]u8 = @splat(' ');
    try std.testing.expectError(error.FrameCapacityExceeded, session.begin(&large));
}
