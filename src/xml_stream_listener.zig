//! XMLAdapter field streaming, not XML validation or prediction parsing.
//! Literal fields preserve tag-like text until a closing boundary is known.
//! Nested targets use the source's legacy raw-text route (including its
//! chunk-sensitive completion/cache heuristic); this does not validate or
//! stream individual children. A non-self-closing structured predecessor is
//! unsupported on the literal route: skipping it requires its complete schema.
const std = @import("std");
const unicode = @import("unicode.zig");

const legacy_lookbehind = 10;

pub const XmlStreamListener = struct {
    pending: []u8,
    output: []u8,
    fields: []const Field,
    target: Field,
    allow_reuse: bool,
    state: State = .{},

    pub const Field = struct { name: []const u8, nested: bool = false };
    pub const Fragment = @import("chat_stream_listener.zig").ChatStreamListener.Fragment;
    pub const Error = error{ BufferTooSmall, UnknownField, UnsupportedNestedPredecessor };

    const State = struct {
        offset: usize = 0,
        len: usize = 0,
        ended: bool = false,
        complete: bool = false,
        cache_hit: bool = false,
        prefix_done: bool = false,
        fenced: bool = false,
        thought_close: ?[]const u8 = null,
        preceding: ?Field = null,
        scalar: Scalar = .{},
        started: bool = false,
        legacy_open: bool = false,
        legacy_count: usize = 0,
        legacy_lengths: [legacy_lookbehind + 1]usize = undefined,
        held: []const u8 = "",
    };

    /// Borrows field names and disjoint buffers. Input must not overlap either
    /// buffer, and chunks are UTF-8 text (not parts of a UTF-8 code point).
    /// Fragment text is valid until the next receive, finish, finalize, or reset.
    pub fn init(options: struct {
        wire_field: []const u8,
        output_fields: []const Field,
        pending_storage: []u8,
        output_storage: []u8,
        allow_reuse: bool = false,
    }) Error!XmlStreamListener {
        const target = fieldNamed(options.output_fields, options.wire_field) orelse
            return error.UnknownField;
        return .{
            .pending = options.pending_storage,
            .output = options.output_storage,
            .fields = options.output_fields,
            .target = target,
            .allow_reuse = options.allow_reuse,
        };
    }

    pub fn reset(self: *XmlStreamListener) void {
        self.state = .{};
    }

    pub fn cacheHit(self: *const XmlStreamListener) bool {
        return self.state.cache_hit;
    }

    /// Null is a chunk without text, not a provider finish signal. On an error
    /// the response state is discarded; stop feeding that response.
    pub fn receive(self: *XmlStreamListener, content: ?[]const u8) Error!?Fragment {
        return self.receiveChunk(content, false);
    }

    /// Feed a provider finish event, including any text carried by the event.
    pub fn finishChunk(self: *XmlStreamListener, content: ?[]const u8) Error!?Fragment {
        return self.receiveChunk(content, true);
    }

    pub fn finish(self: *XmlStreamListener) Error!?Fragment {
        return self.finishChunk(null);
    }

    /// Source exhaustion, distinct from receiving a provider finish event.
    /// A missing closing tag need not emit a terminal fragment.
    pub fn finalize(self: *XmlStreamListener) Error!?Fragment {
        errdefer self.reset();
        self.state.complete = true;
        if (self.state.ended) return null;
        if (!self.target.nested) {
            if (!self.state.started and self.state.preceding == null) return null;
            return self.literal("", true);
        }
        if (!self.state.started) return null;
        self.state.ended = true;
        const bytes = self.buffer();
        const value = bytes[0 .. exactTag(bytes, self.target.name, true) orelse bytes.len];
        if (value.len == 0) return null;
        const held = self.state.held;
        if (value.len > self.output.len - held.len) return error.BufferTooSmall;
        @memmove(self.output[0..held.len], held);
        @memcpy(self.output[held.len..][0..value.len], value);
        return .{ .text = self.output[0 .. held.len + value.len], .is_last_chunk = true };
    }

    fn receiveChunk(
        self: *XmlStreamListener,
        content: ?[]const u8,
        final: bool,
    ) Error!?Fragment {
        errdefer self.reset();
        if (self.state.ended) {
            if (!self.allow_reuse) return null;
            if (!self.state.complete) {
                // The target can end while its response still has trailing fields or framing.
                self.state.complete = final;
                return null;
            }
            self.reset();
        }
        if (self.target.nested) return self.legacy(content orelse return null);
        self.state.complete = final;
        return self.literal(content orelse "", final);
    }

    fn buffer(self: *const XmlStreamListener) []const u8 {
        return self.pending[self.state.offset..][0..self.state.len];
    }

    fn append(self: *XmlStreamListener, text: []const u8) Error!void {
        const state = &self.state;
        std.debug.assert(state.offset + state.len <= self.pending.len);
        if (text.len > self.pending.len - state.len) return error.BufferTooSmall;
        if (text.len > self.pending.len - state.offset - state.len) {
            @memmove(self.pending[0..state.len], self.buffer());
            state.offset = 0;
        }
        @memcpy(self.pending[state.offset + state.len ..][0..text.len], text);
        state.len += text.len;
    }

    fn discard(self: *XmlStreamListener, count: usize) void {
        std.debug.assert(count <= self.state.len);
        self.state.len -= count;
        self.state.offset = if (self.state.len == 0) 0 else self.state.offset + count;
    }

    fn literal(self: *XmlStreamListener, text: []const u8, final: bool) Error!?Fragment {
        const state = &self.state;
        try self.append(text);
        // A plain suffix cannot resolve a provisional close. Do not rescan it on each delta.
        if (state.scalar.waiting_for_tag and !final and indexByte(text, '<') == null) {
            return null;
        }
        if (!state.prefix_done and !self.stripPrefix(final)) return null;
        while (true) {
            if (!state.started and state.preceding == null) {
                const opening = self.seekOpening() orelse return null;
                state.started = equal(opening.field.name, self.target.name);
                if (opening.empty) {
                    if (!state.started) continue;
                    state.ended = true;
                    return .{ .text = "", .is_last_chunk = true };
                }
                if (!state.started) {
                    if (opening.field.nested) return error.UnsupportedNestedPredecessor;
                    state.preceding = opening.field;
                }
            }
            const active = state.preceding orelse self.target;
            const read = state.scalar.read(self.buffer(), active.name, self.fields, final);
            if (state.started) {
                state.ended = read.done;
                if (read.value.len > self.output.len) return error.BufferTooSmall;
                @memcpy(self.output[0..read.value.len], read.value);
                self.discard(read.consumed);
                if (read.value.len == 0 and !read.done) return null;
                return .{ .text = self.output[0..read.value.len], .is_last_chunk = read.done };
            }
            self.discard(read.consumed);
            if (!read.done) {
                if (final) state.ended = true; // Source suppresses the predecessor's parse error.
                return null;
            }
            state.preceding = null;
            state.scalar = .{};
        }
    }

    fn stripPrefix(self: *XmlStreamListener, final: bool) bool {
        const state = &self.state;
        while (true) {
            if (state.thought_close) |closing| {
                const end = std.mem.indexOf(u8, self.buffer(), closing) orelse {
                    self.discard(self.buffer().len -| (closing.len - 1));
                    return false;
                };
                self.discard(end + closing.len);
                state.thought_close = null;
            }
            const text = self.buffer();
            self.discard(trimStartOffset(text));
            const prefix = self.buffer();
            const thoughts = [_]struct {
                name: []const u8,
                opening: []const u8,
                closing: []const u8,
            }{
                .{ .name = "think", .opening = "<think>", .closing = "</think>" },
                .{ .name = "thinking", .opening = "<thinking>", .closing = "</thinking>" },
            };
            for (thoughts) |thought| {
                if (starts(prefix, thought.opening)) {
                    if (fieldNamed(self.fields, thought.name) != null) break;
                    self.discard(thought.opening.len);
                    state.thought_close = thought.closing;
                    break;
                }
                if (!final and starts(thought.opening, prefix)) return false;
            }
            if (state.thought_close != null) continue;
            switch (fence(prefix)) {
                .full => |end| if (!state.fenced) {
                    self.discard(end);
                    state.fenced = true;
                    continue;
                },
                .partial => if (!final) return false,
                .none => {},
            }
            state.prefix_done = true;
            return true;
        }
    }

    const Opening = struct { field: Field, empty: bool };

    fn seekOpening(self: *XmlStreamListener) ?Opening {
        const text = self.buffer();
        var scan: usize = 0;
        var last = text.len;
        while (std.mem.indexOfScalarPos(u8, text, scan, '<')) |pos| {
            last = pos;
            for (self.fields) |field| {
                const match = openingTag(text[pos..], field.name, false);
                if (match == .full) {
                    const end = pos + match.full;
                    const empty = text[end - 2] == '/';
                    self.discard(end);
                    return .{ .field = field, .empty = empty };
                }
            }
            scan = pos + 1;
        }
        self.discard(last);
        return null;
    }

    fn legacy(self: *XmlStreamListener, chunk: []const u8) Error!?Fragment {
        const state = &self.state;
        var text = chunk;
        if (exactTag(chunk, self.target.name, false)) |start| {
            const rest = chunk[start + self.target.name.len + 2 ..];
            if (exactTag(rest, self.target.name, true) != null) {
                state.cache_hit = true;
                state.started = true;
                state.ended = true;
                return null;
            }
        }
        if (!state.started) {
            if (!state.legacy_open) {
                if (indexByte(text, '<') == null) return null;
                try self.append(text);
                state.legacy_open = true;
                return null;
            }
            try self.append(text);
            const accumulated = self.buffer();
            const start = exactTag(accumulated, self.target.name, false) orelse {
                if (!splitExactOpening(unicode.strip(accumulated, false), self.target.name)) {
                    self.discard(accumulated.len);
                    state.legacy_open = false;
                }
                return null;
            };
            state.started = true;
            const body = accumulated[start + self.target.name.len + 2 ..];
            text = body[trimStartOffset(body)..];
            // Remove the opening and leading framing before queuing the first body chunk.
            self.discard(accumulated.len - text.len);
            text = self.buffer();
        } else {
            if (text.len == 0) return null;
            try self.append(text);
        }
        if (text.len == 0) return null;
        state.legacy_lengths[state.legacy_count] = text.len;
        state.legacy_count += 1;
        return self.legacyDrain();
    }

    fn legacyDrain(self: *XmlStreamListener) Error!?Fragment {
        const state = &self.state;
        const bytes = self.buffer();
        const stripped = unicode.strip(bytes, false);
        const hold = std.mem.endsWith(u8, stripped, "<") or
            std.mem.indexOf(u8, stripped, "</") != null;
        const popped = if (hold and state.legacy_count > legacy_lookbehind)
            state.legacy_lengths[0]
        else
            0;
        const rest = bytes[popped..];
        const closing = exactTag(rest, self.target.name, true);
        const drain = if (!hold or closing != null) bytes.len else popped;
        if (drain == 0) return null;
        var value = bytes[0..if (closing) |pos| popped + pos else drain];
        if (!state.scalar.value_started) value = value[trimStartOffset(value)..];
        const visible = rstrip(value);
        const held = state.held.len;
        if (value.len > self.output.len - held) return error.BufferTooSmall;
        @memmove(self.output[0..held], state.held);
        @memcpy(self.output[held..][0..value.len], value);
        state.held = self.output[held + visible.len .. held + value.len];
        if (visible.len != 0) state.scalar.value_started = true;
        state.ended = closing != null;
        self.discard(drain);
        if (closing != null or !hold) {
            state.legacy_count = 0;
        } else {
            state.legacy_count -= 1;
            @memmove(
                state.legacy_lengths[0..state.legacy_count],
                state.legacy_lengths[1..][0..state.legacy_count],
            );
        }
        if (visible.len == 0 and !state.ended) return null;
        return .{
            .text = if (visible.len == 0) "" else self.output[0 .. held + visible.len],
            .is_last_chunk = state.ended,
        };
    }
};

const Scalar = struct {
    multiline: ?bool = null,
    at_line_start: bool = true,
    value_started: bool = false,
    waiting_for_tag: bool = false,

    const Read = struct { value: []const u8 = "", consumed: usize = 0, done: bool = false };

    fn read(
        self: *Scalar,
        text: []const u8,
        name: []const u8,
        fields: []const XmlStreamListener.Field,
        final: bool,
    ) Read {
        self.waiting_for_tag = false;
        var start: usize = 0;
        if (self.multiline == null) {
            switch (newline(text)) {
                .full => |end| start = end,
                .partial => if (!final) return .{},
                .none => {},
            }
            self.multiline = start != 0;
        }
        const multiline = self.multiline.?;
        var scan = start;
        var end = text.len;
        var close_end: ?usize = null;
        while (std.mem.indexOfScalarPos(u8, text, scan, '<')) |candidate| {
            scan = candidate + 1;
            switch (closingTag(text[candidate..], name, multiline, final)) {
                .none => continue,
                .partial => if (final) continue,
                .full => |len| {
                    const after = candidate + len;
                    switch (continuation(text[after..], fields, final)) {
                        .invalid => continue,
                        .sibling => close_end = after,
                        .prose => if (final) {
                            close_end = after;
                        } else {
                            self.waiting_for_tag = true;
                        },
                        .partial => {},
                    }
                },
            }
            end = candidate;
            break;
        }
        var value = text[start..end];
        if (multiline) {
            const line = if (std.mem.lastIndexOfScalar(u8, value, '\n')) |last| last + 1 else 0;
            if ((line > 0 or self.at_line_start) and spaces(value[line..]) == value.len - line) {
                value = value[0..line];
            }
            value = dropSuffix(value, '\n');
            value = dropSuffix(value, '\r');
        } else value = rstrip(value);
        const consumed = close_end orelse start + value.len;
        if (value.len > 0) self.at_line_start = value[value.len - 1] == '\n';
        if (!multiline and !self.value_started) value = value[trimStartOffset(value)..];
        self.value_started = self.value_started or value.len != 0;
        return .{ .value = value, .consumed = consumed, .done = close_end != null };
    }
};

const Match = union(enum) { none, partial, full: usize };

fn openingTag(text: []const u8, name: []const u8, allow_key: bool) Match {
    std.debug.assert(text.len > 0 and text[0] == '<');
    if (!starts(text[1..], name)) return if (starts(name, text[1..])) .partial else .none;
    var pos = 1 + name.len;
    const before_space = pos;
    pos += spaceOffset(text[pos..], allow_key);
    if (allow_key and pos > before_space and pos < text.len and text[pos] == 'k') {
        if (!starts(text[pos..], "key=")) {
            return if (starts("key=", text[pos..])) .partial else .none;
        }
        pos += 4;
        if (pos == text.len) return .partial;
        const quote = text[pos];
        if (quote != '\'' and quote != '"') return .none;
        pos = (std.mem.indexOfScalarPos(u8, text, pos + 1, quote) orelse return .partial) + 1;
        pos += spaceOffset(text[pos..], true);
    }
    if (pos < text.len and text[pos] == '/') pos += 1;
    if (pos == text.len) return .partial;
    return if (text[pos] == '>') .{ .full = pos + 1 } else .none;
}

const Continuation = enum { sibling, prose, partial, invalid };

fn continuation(
    text: []const u8,
    fields: []const XmlStreamListener.Field,
    final: bool,
) Continuation {
    const rest = text[spaceOffset(text, true)..];
    const first = indexByte(rest, '<') orelse return .prose;
    if (first != 0) return .invalid;
    var partial = false;
    for (fields) |field| {
        switch (openingTag(rest, field.name, true)) {
            .full => return .sibling,
            .partial => partial = true,
            .none => {},
        }
    }
    return if (partial and !final) .partial else .invalid;
}

fn closingTag(text: []const u8, name: []const u8, multiline: bool, final: bool) Match {
    if (!starts(text, "</")) return if (text.len == 1) .partial else .none;
    if (!starts(text[2..], name)) return if (starts(name, text[2..])) .partial else .none;
    var pos = 2 + name.len;
    pos += if (multiline) spaces(text[pos..]) else spaceOffset(text[pos..], true);
    if (pos == text.len) return .partial;
    if (text[pos] != '>') return .none;
    pos += 1;
    if (!multiline) return .{ .full = pos };
    return switch (newline(text[pos..])) {
        .full => |len| .{ .full = pos + len },
        .partial => if (final) .{ .full = text.len } else .partial,
        .none => .none,
    };
}

fn newline(text: []const u8) Match {
    var pos = spaces(text);
    if (pos < text.len and text[pos] == '\r') pos += 1;
    if (pos == text.len) return .partial;
    return if (text[pos] == '\n') .{ .full = pos + 1 } else .none;
}

fn fence(text: []const u8) Match {
    if (!starts(text, "```")) return if (starts("```", text)) .partial else .none;
    var pos: usize = 3;
    if (starts(text[pos..], "xml")) {
        pos += 3;
    } else if (starts("xml", text[pos..])) return .partial;
    return switch (newline(text[pos..])) {
        .full => |len| .{ .full = pos + len },
        else => |result| result,
    };
}

fn exactTag(text: []const u8, name: []const u8, close: bool) ?usize {
    const prefix: []const u8 = if (close) "</" else "<";
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, text, scan, prefix)) |pos| {
        const rest = text[pos + prefix.len ..];
        if (starts(rest, name) and rest.len > name.len and rest[name.len] == '>') return pos;
        scan = pos + 1;
    }
    return null;
}

fn splitExactOpening(text: []const u8, name: []const u8) bool {
    const last = std.mem.lastIndexOfScalar(u8, text, '<') orelse return false;
    return starts(name, text[last + 1 ..]);
}

fn fieldNamed(fields: []const XmlStreamListener.Field, name: []const u8) ?XmlStreamListener.Field {
    for (fields) |field| if (equal(field.name, name)) return field;
    return null;
}

fn spaces(text: []const u8) usize {
    for (text, 0..) |byte, i| if (byte != ' ' and byte != '\t') return i;
    return text.len;
}

fn trimStartOffset(text: []const u8) usize {
    return spaceOffset(text, false);
}

// Python str/re whitespace includes the four C0 separators; regex's \s does not.
fn spaceOffset(text: []const u8, regex_whitespace: bool) usize {
    var pos: usize = 0;
    while (pos < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[pos]) catch break;
        if (len > text.len - pos) break;
        const cp = std.unicode.utf8Decode(text[pos..][0..len]) catch break;
        if (!unicode.isSpace(cp) or (regex_whitespace and cp >= 0x1c and cp <= 0x1f)) break;
        pos += len;
    }
    return pos;
}

fn rstrip(text: []const u8) []const u8 {
    const stripped = unicode.strip(text, false);
    if (stripped.len == 0) return text[0..0];
    return text[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(text.ptr) + stripped.len];
}

fn dropSuffix(text: []const u8, byte: u8) []const u8 {
    if (text.len != 0 and text[text.len - 1] == byte) return text[0 .. text.len - 1];
    return text;
}

fn indexByte(text: []const u8, byte: u8) ?usize {
    return std.mem.indexOfScalar(u8, text, byte);
}

fn starts(text: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, text, prefix);
}

fn equal(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test "literal XML boundary waits for a complete sibling opening" {
    var pending: [128]u8 = undefined;
    var output: [128]u8 = undefined;
    var listener = try XmlStreamListener.init(.{
        .wire_field = "answer",
        .output_fields = &.{ .{ .name = "answer" }, .{ .name = "note" } },
        .pending_storage = &pending,
        .output_storage = &output,
    });
    const first = (try listener.receive("<answer>  yes</answer><no")).?;
    try std.testing.expectEqualStrings("yes", first.text);
    try std.testing.expect(!first.is_last_chunk);
    try std.testing.expect((try listener.receive("te key='a>b'")) == null);
    const last = (try listener.receive(">")).?;
    try std.testing.expectEqualStrings("", last.text);
    try std.testing.expect(last.is_last_chunk);
    try std.testing.expect((try listener.finalize()) == null);
    try std.testing.expect(!listener.cacheHit());
}

test "literal XML refuses a structured predecessor and resets on capacity errors" {
    var pending: [32]u8 = undefined;
    var output: [4]u8 = undefined;
    var listener = try XmlStreamListener.init(.{
        .wire_field = "answer",
        .output_fields = &.{ .{ .name = "items", .nested = true }, .{ .name = "answer" } },
        .pending_storage = &pending,
        .output_storage = &output,
    });
    try std.testing.expectError(
        error.UnsupportedNestedPredecessor,
        listener.receive("<items><answer>fake</answer>"),
    );
    try std.testing.expectError(
        error.BufferTooSmall,
        listener.receive("<answer>longer than output"),
    );
    const first = (try listener.receive("<items /><answer>yes</answer>")).?;
    try std.testing.expectEqualStrings("yes", first.text);
    try std.testing.expect(!first.is_last_chunk);
    try std.testing.expect((try listener.finish()).?.is_last_chunk);
}

test "literal XML provider finish can carry text and enables reuse" {
    var pending: [64]u8 = undefined;
    var output: [64]u8 = undefined;
    var listener = try XmlStreamListener.init(.{
        .wire_field = "answer",
        .output_fields = &.{.{ .name = "answer" }},
        .pending_storage = &pending,
        .output_storage = &output,
        .allow_reuse = true,
    });
    const first = (try listener.receive("<answer>first ")).?;
    try std.testing.expectEqualStrings("first", first.text);
    try std.testing.expect(!first.is_last_chunk);
    const last = (try listener.finishChunk("response</answer>sign off")).?;
    try std.testing.expectEqualStrings(" response", last.text);
    try std.testing.expect(last.is_last_chunk);
    try std.testing.expect((try listener.finalize()) == null);
    const reused = (try listener.finishChunk("<answer>next</answer>")).?;
    try std.testing.expectEqualStrings("next", reused.text);
    try std.testing.expect(reused.is_last_chunk);
    try std.testing.expect(!listener.cacheHit());
}
