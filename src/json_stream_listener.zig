//! Incremental JSON wire-field projection. Emits raw JSON text of the wire
//! field's value (escapes untouched) and stops at the object close or at the
//! first value of a non-wire key. Final JSON decoding stays with the adapter.
const std = @import("std");
const unicode = @import("unicode.zig");
const assert = std.debug.assert;
const chat = @import("chat_stream_listener.zig");

const lookbehind_max = 10;
const depth_max = 64;
const key_max = 256;

pub const JsonStreamListener = struct {
    pub const Fragment = chat.ChatStreamListener.Fragment;
    pub const Error = error{
        BufferTooSmall,
        InvalidJson,
        TooDeep,
        KeyTooLong,
        UnsupportedKey,
    };

    pending: []u8,
    output: []u8,
    wire_field: []const u8,
    allow_reuse: bool,
    state: State = .{},

    const State = struct {
        failed: bool = false,
        ended: bool = false,
        started: bool = false,
        pending_len: usize = 0,
        lengths: [lookbehind_max + 1]usize = undefined,
        count: usize = 0,
        phase: Phase = .value,
        depth: usize = 1,
        stack: u64 = 1,
        escape: Escape = .none,
        unicode_left: u8 = 0,
        in_key: bool = false,
        other_key: bool = false,
        other_value_started: bool = false,
        wire: []const u8 = "",
        key_len: usize = 0,
        key: [key_max]u8 = undefined,
        num: Number = .minus,
        literal: []const u8 = "",
        literal_index: usize = 0,
    };

    const Phase = enum { value, value_or_close, key_or_close, key, colon, after_value, string, number, literal, done };
    const Escape = enum { none, backslash, unicode };
    const Number = enum { minus, zero, int, dot, frac, exp_start, exp_sign, exp };
    const Step = enum { consumed, reprocess, complete, other_value };

    /// Borrows disjoint caller buffers. Output must be at least as large as
    /// pending because every fragment is copied out of the pending window.
    /// Returned text lasts until the next receive, finalize, or reset.
    pub fn init(options: struct {
        wire_field: []const u8,
        pending_storage: []u8,
        output_storage: []u8,
        allow_reuse: bool = false,
    }) Error!JsonStreamListener {
        if (options.output_storage.len < options.pending_storage.len) return error.BufferTooSmall;
        return .{
            .pending = options.pending_storage,
            .output = options.output_storage,
            .wire_field = options.wire_field,
            .allow_reuse = options.allow_reuse,
        };
    }

    pub fn reset(self: *JsonStreamListener) void {
        self.state = .{};
    }

    /// JSON streams never report cache hits; kept for parity with ChatStreamListener.
    pub fn cacheHit(self: *const JsonStreamListener) bool {
        _ = self;
        return false;
    }

    /// Pass null for a chunk with no text. Any error poisons the listener:
    /// further receives return error.InvalidJson until reset.
    pub fn receive(self: *JsonStreamListener, content: ?[]const u8) Error!?Fragment {
        if (self.state.failed) return error.InvalidJson;
        errdefer self.state.failed = true;
        if (self.state.ended) {
            if (!self.allow_reuse) return null;
            self.reset();
        }
        const text = content orelse return null;
        if (!self.state.started) return self.receiveOpening(text);
        if (text.len == 0) return null;
        try self.append(text);
        return self.process(text.len);
    }

    /// Flushes the queue on source exhaustion.
    pub fn finalize(self: *JsonStreamListener) Error!?Fragment {
        const state = &self.state;
        if (state.failed or state.ended or !state.started or state.pending_len == 0) return null;
        state.ended = true;
        return .{ .text = self.copyOut(self.pending[0..state.pending_len]), .is_last_chunk = true };
    }

    fn receiveOpening(self: *JsonStreamListener, text: []const u8) Error!?Fragment {
        const state = &self.state;
        if (state.pending_len == 0) {
            if (std.mem.indexOfScalar(u8, text, '"') == null) return null;
            try self.append(text);
            return null;
        }
        try self.append(text);
        const concat = self.pending[0..state.pending_len];
        if (findWire(concat, self.wire_field)) |value_start| {
            state.started = true;
            state.wire = self.wire_field;
            const value = lstrip(concat[value_start..]);
            @memmove(self.pending[0..value.len], value);
            state.pending_len = value.len;
            if (value.len == 0) return null;
            return self.process(value.len);
        }
        if (!endsWithMarkerPrefix(unicode.strip(concat, false), self.wire_field)) state.pending_len = 0;
        return null;
    }

    fn append(self: *JsonStreamListener, text: []const u8) Error!void {
        const state = &self.state;
        assert(state.pending_len <= self.pending.len);
        if (text.len > self.pending.len - state.pending_len) return error.BufferTooSmall;
        @memcpy(self.pending[state.pending_len..][0..text.len], text);
        state.pending_len += text.len;
    }

    /// The newest `new_len` bytes of pending are the chunk just received.
    fn process(self: *JsonStreamListener, new_len: usize) Error!?Fragment {
        const state = &self.state;
        const len = state.pending_len;
        const event = try feed(state, self.pending[len - new_len .. len]);
        state.lengths[state.count] = new_len;
        state.count += 1;

        const flush_all = !couldForm(unicode.strip(self.pending[0..len], false));
        const token_len: usize = if (flush_all)
            len
        else if (state.count > lookbehind_max)
            state.lengths[0]
        else
            0;

        if (event) |kind| {
            const last = self.pending[token_len..len];
            const cut = switch (kind) {
                .complete => cutAt(last, std.mem.lastIndexOfScalar(u8, last, '}')),
                .other_value => cutAt(last, std.mem.indexOf(u8, last, state.key[0..state.key_len])),
                .consumed, .reprocess => unreachable,
            };
            state.ended = true;
            return .{ .text = self.copyOut(self.pending[0 .. token_len + cut]), .is_last_chunk = true };
        }

        if (token_len == 0) return null;
        const text = self.copyOut(self.pending[0..token_len]);
        if (flush_all) {
            state.pending_len = 0;
            state.count = 0;
        } else {
            const rest = len - token_len;
            @memmove(self.pending[0..rest], self.pending[token_len..len]);
            state.pending_len = rest;
            state.count -= 1;
            std.mem.copyForwards(usize, state.lengths[0..state.count], state.lengths[1..][0..state.count]);
        }
        return .{ .text = text, .is_last_chunk = false };
    }

    fn copyOut(self: *JsonStreamListener, bytes: []const u8) []const u8 {
        assert(bytes.len <= self.output.len);
        @memcpy(self.output[0..bytes.len], bytes);
        return self.output[0..bytes.len];
    }

    /// Feeds bytes through the incremental validator. Returns a terminal step
    /// when the top-level object closes or a non-wire key's value begins.
    fn feed(state: *State, bytes: []const u8) Error!?Step {
        var index: usize = 0;
        while (index < bytes.len) {
            switch (try step(state, bytes[index])) {
                .consumed => index += 1,
                .reprocess => {},
                .complete => {
                    for (bytes[index + 1 ..]) |c| if (!isWhitespace(c)) return error.InvalidJson;
                    return .complete;
                },
                .other_value => unreachable,
            }
        }
        if (state.other_value_started) {
            switch (state.phase) {
                .number => switch (state.num) {
                    .zero, .int, .frac, .exp => return .other_value,
                    else => {},
                },
                .string => if (!state.in_key) return .other_value,
                .after_value, .key_or_close, .value_or_close, .key => return .other_value,
                else => {},
            }
        }
        return null;
    }

    fn step(state: *State, c: u8) Error!Step {
        switch (state.phase) {
            .value => return if (isWhitespace(c)) .consumed else valueStart(state, c),
            .value_or_close => {
                if (isWhitespace(c)) return .consumed;
                if (c == ']') return close(state, false);
                return valueStart(state, c);
            },
            .key_or_close => {
                if (isWhitespace(c)) return .consumed;
                if (c == '}') return close(state, true);
                return keyStart(state, c);
            },
            .key => return if (isWhitespace(c)) .consumed else keyStart(state, c),
            .colon => {
                if (isWhitespace(c)) return .consumed;
                if (c != ':') return error.InvalidJson;
                state.phase = .value;
                return .consumed;
            },
            .after_value => {
                if (isWhitespace(c)) return .consumed;
                switch (c) {
                    ',' => {
                        state.phase = if (isObject(state)) .key else .value;
                        return .consumed;
                    },
                    '}' => return close(state, true),
                    ']' => return close(state, false),
                    else => return error.InvalidJson,
                }
            },
            .string => return stringByte(state, c),
            .number => return numberByte(state, c),
            .literal => {
                if (c != state.literal[state.literal_index]) return error.InvalidJson;
                state.literal_index += 1;
                if (state.literal_index == state.literal.len) state.phase = .after_value;
                return .consumed;
            },
            .done => return error.InvalidJson,
        }
    }

    fn valueStart(state: *State, c: u8) Error!Step {
        if (state.depth == 1 and state.other_key) state.other_value_started = true;
        switch (c) {
            '{' => {
                try push(state, true);
                state.phase = .key_or_close;
            },
            '[' => {
                try push(state, false);
                state.phase = .value_or_close;
            },
            '"' => beginString(state, false),
            '-' => beginNumber(state, .minus),
            '0' => beginNumber(state, .zero),
            '1'...'9' => beginNumber(state, .int),
            't' => beginLiteral(state, "true"),
            'f' => beginLiteral(state, "false"),
            'n' => beginLiteral(state, "null"),
            else => return error.InvalidJson,
        }
        return .consumed;
    }

    fn keyStart(state: *State, c: u8) Error!Step {
        if (c != '"') return error.InvalidJson;
        beginString(state, true);
        return .consumed;
    }

    fn close(state: *State, object: bool) Error!Step {
        if (isObject(state) != object) return error.InvalidJson;
        state.depth -= 1;
        if (state.depth == 0) {
            state.phase = .done;
            return .complete;
        }
        state.phase = .after_value;
        return .consumed;
    }

    fn push(state: *State, object: bool) Error!void {
        if (state.depth == depth_max) return error.TooDeep;
        state.depth += 1;
        const bit = @as(u64, 1) << @intCast(state.depth - 1);
        if (object) state.stack |= bit else state.stack &= ~bit;
    }

    fn isObject(state: *const State) bool {
        return (state.stack >> @intCast(state.depth - 1)) & 1 == 1;
    }

    fn beginString(state: *State, key: bool) void {
        state.phase = .string;
        state.in_key = key;
        state.escape = .none;
        if (key and state.depth == 1 and !state.other_key) state.key_len = 0;
    }

    fn beginNumber(state: *State, num: Number) void {
        state.phase = .number;
        state.num = num;
    }

    fn beginLiteral(state: *State, literal: []const u8) void {
        state.phase = .literal;
        state.literal = literal;
        state.literal_index = 1;
    }

    fn stringByte(state: *State, c: u8) Error!Step {
        switch (state.escape) {
            .none => {
                if (c == '"') {
                    if (state.in_key) {
                        if (state.depth == 1 and !state.other_key and
                            !std.mem.eql(u8, state.key[0..state.key_len], state.wire))
                        {
                            state.other_key = true;
                        }
                        state.phase = .colon;
                    } else {
                        state.phase = .after_value;
                    }
                    return .consumed;
                }
                if (c == '\\') {
                    if (state.in_key and state.depth == 1) return error.UnsupportedKey;
                    state.escape = .backslash;
                    return .consumed;
                }
                if (c < 0x20) return error.InvalidJson;
                if (state.in_key and state.depth == 1 and !state.other_key) try captureKey(state, c);
                return .consumed;
            },
            .backslash => {
                switch (c) {
                    '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => state.escape = .none,
                    'u' => {
                        state.escape = .unicode;
                        state.unicode_left = 4;
                    },
                    else => return error.InvalidJson,
                }
                return .consumed;
            },
            .unicode => {
                if (!std.ascii.isHex(c)) return error.InvalidJson;
                state.unicode_left -= 1;
                if (state.unicode_left == 0) state.escape = .none;
                return .consumed;
            },
        }
    }

    fn captureKey(state: *State, c: u8) Error!void {
        if (state.key_len == key_max) return error.KeyTooLong;
        state.key[state.key_len] = c;
        state.key_len += 1;
    }

    fn numberByte(state: *State, c: u8) Error!Step {
        const digit = std.ascii.isDigit(c);
        const exp = c == 'e' or c == 'E';
        switch (state.num) {
            .minus => state.num = if (c == '0') .zero else if (digit) .int else return error.InvalidJson,
            .zero => {
                if (c == '.') state.num = .dot else if (exp) state.num = .exp_start else return endNumber(state);
            },
            .int => {
                if (c == '.') state.num = .dot else if (exp) state.num = .exp_start else if (!digit) return endNumber(state);
            },
            .dot => state.num = if (digit) .frac else return error.InvalidJson,
            .frac => {
                if (exp) state.num = .exp_start else if (!digit) return endNumber(state);
            },
            .exp_start => state.num = if (c == '+' or c == '-') .exp_sign else if (digit) .exp else return error.InvalidJson,
            .exp_sign => state.num = if (digit) .exp else return error.InvalidJson,
            .exp => {
                if (!digit) return endNumber(state);
            },
        }
        return .consumed;
    }

    fn endNumber(state: *State) Step {
        state.phase = .after_value;
        return .reprocess;
    }
};

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

/// A flush must wait while the text may still grow into a closing quote,
/// a separator after a string, or a closing brace.
fn couldForm(text: []const u8) bool {
    return std.mem.endsWith(u8, text, "\"") or std.mem.endsWith(u8, text, "\",") or
        std.mem.indexOfScalar(u8, text, '}') != null;
}

/// Index into `last` where the emitted fragment ends; null drops one codepoint.
fn cutAt(last: []const u8, index: ?usize) usize {
    if (index) |i| return i;
    if (last.len == 0) return 0;
    var start = last.len - 1;
    while (start > 0 and last[start] & 0xc0 == 0x80) start -= 1;
    return start;
}

fn lstrip(text: []const u8) []const u8 {
    const stripped = unicode.strip(text, false);
    if (stripped.len == 0) return text[text.len..];
    const offset = @intFromPtr(stripped.ptr) - @intFromPtr(text.ptr);
    return text[offset..];
}

fn markerByte(field: []const u8, index: usize) u8 {
    if (index == 0) return '"';
    if (index <= field.len) return field[index - 1];
    return if (index == field.len + 1) '"' else ':';
}

fn markerAt(text: []const u8, start: usize, field: []const u8) bool {
    const len = field.len + 3;
    if (text.len - start < len) return false;
    for (0..len) |j| {
        if (text[start + j] != markerByte(field, j)) return false;
    }
    return true;
}

/// Returns the index just past the `"field":` marker in text.
fn findWire(text: []const u8, field: []const u8) ?usize {
    var index: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, index, '"')) |start| {
        if (markerAt(text, start, field)) return start + field.len + 3;
        index = start + 1;
    }
    return null;
}

/// True when a proper nonempty prefix of the `"field":` marker ends text.
fn endsWithMarkerPrefix(text: []const u8, field: []const u8) bool {
    const marker_len = field.len + 3;
    var k: usize = 1;
    while (k < marker_len and k <= text.len) : (k += 1) {
        const base = text.len - k;
        var j: usize = 0;
        while (j < k and text[base + j] == markerByte(field, j)) : (j += 1) {}
        if (j == k) return true;
    }
    return false;
}

test "JSON listener refuses a full pending window and requires reset" {
    var pending: [12]u8 = undefined;
    var output: [12]u8 = undefined;
    var listener = try JsonStreamListener.init(.{
        .wire_field = "answer",
        .pending_storage = &pending,
        .output_storage = &output,
    });
    try std.testing.expectEqual(null, try listener.receive("{\"answer\":"));
    try std.testing.expectError(error.BufferTooSmall, listener.receive("123}"));
    try std.testing.expectError(error.InvalidJson, listener.receive(null));
    listener.reset();
    try std.testing.expectEqual(null, try listener.receive("{\"answer\":"));
    const last = (try listener.receive("1}")).?;
    try std.testing.expectEqualStrings("1", last.text);
    try std.testing.expect(last.is_last_chunk);
}
