//! Incremental ChatAdapter text field projection. Final prediction parsing
//! remains the adapter's responsibility.
const std = @import("std");
const unicode = @import("unicode.zig");
const assert = std.debug.assert;
const prefix = "[[ ## ";
const suffix = " ## ]]";
const lookbehind_max = 10;

pub const ChatStreamListener = struct {
    pending: []u8,
    output: []u8,
    marker: []const u8,
    allow_reuse: bool,
    state: State = .{},

    const State = struct {
        pending_len: usize = 0,
        lengths: [lookbehind_max + 1]usize = undefined,
        count: usize = 0,
        whitespace: []const u8 = "",
        started: bool = false,
        ended: bool = false,
        value_started: bool = false,
        cache_hit: bool = false,
    };

    pub const Fragment = struct { text: []const u8, is_last_chunk: bool };
    pub const Error = error{BufferTooSmall};

    /// Borrows disjoint caller buffers, which must not overlap input text.
    /// Returned text lasts until the next receive, finalize, or reset.
    pub fn init(options: struct {
        wire_field: []const u8,
        marker_storage: []u8,
        pending_storage: []u8,
        output_storage: []u8,
        allow_reuse: bool = false,
    }) Error!ChatStreamListener {
        const marker = std.fmt.bufPrint(
            options.marker_storage,
            prefix ++ "{s}" ++ suffix,
            .{options.wire_field},
        ) catch return error.BufferTooSmall;
        return .{
            .marker = marker,
            .pending = options.pending_storage,
            .output = options.output_storage,
            .allow_reuse = options.allow_reuse,
        };
    }

    pub fn reset(self: *ChatStreamListener) void {
        self.state = .{};
    }

    /// DSPy flags a chunk containing both markers as a cache hit, regardless
    /// of whether it actually came from a cache.
    pub fn cacheHit(self: *const ChatStreamListener) bool {
        return self.state.cache_hit;
    }

    /// Pass null for a chunk with no text. Capacity errors clear the state;
    /// stop feeding the failed response before reusing the listener.
    pub fn receive(self: *ChatStreamListener, content: ?[]const u8) Error!?Fragment {
        errdefer self.reset();
        const state = &self.state;
        if (state.ended) {
            if (!self.allow_reuse) return null;
            self.reset();
        }
        const text = content orelse return null;
        if (std.mem.indexOf(u8, text, self.marker)) |start| {
            if (hasEnd(text[start + self.marker.len ..])) {
                state.cache_hit = true;
                state.started = true;
                state.ended = true;
                return null;
            }
        }
        if (!state.started) return self.receiveOpening(text);
        if (text.len == 0) return null;
        _ = try self.append(text);
        return self.receivePending(text.len);
    }

    fn receiveOpening(self: *ChatStreamListener, text: []const u8) Error!?Fragment {
        const state = &self.state;
        assert(!state.started);
        assert(state.count == 0);
        if (state.pending_len == 0) {
            if (std.mem.indexOfScalar(u8, text, '[') != null) _ = try self.append(text);
            return null;
        }
        const buffered = try self.append(text);
        if (std.mem.indexOf(u8, buffered, self.marker)) |start| {
            state.started = true;
            const value = lstrip(buffered[start + self.marker.len ..]);
            @memmove(self.pending[0..value.len], value);
            state.pending_len = value.len;
            return if (value.len == 0) null else self.receivePending(value.len);
        }
        const trimmed = unicode.strip(buffered, false);
        var len = @min(trimmed.len, self.marker.len - 1);
        while (len != 0) : (len -= 1) {
            if (std.mem.endsWith(u8, trimmed, self.marker[0..len])) return null;
        }
        state.pending_len = 0;
        return null;
    }

    fn append(self: *ChatStreamListener, text: []const u8) Error![]const u8 {
        const len = self.state.pending_len;
        assert(len <= self.pending.len);
        if (text.len > self.pending.len - len) return error.BufferTooSmall;
        @memcpy(self.pending[len..][0..text.len], text);
        self.state.pending_len += text.len;
        return self.pending[0..self.state.pending_len];
    }

    fn receivePending(self: *ChatStreamListener, chunk_len: usize) Error!?Fragment {
        const state = &self.state;
        assert(state.started);
        assert(state.count <= lookbehind_max);
        state.lengths[state.count] = chunk_len;
        state.count += 1;
        const bytes = self.pending[0..state.pending_len];
        const trimmed = unicode.strip(bytes, false);
        const maybe_end = std.mem.indexOf(u8, trimmed, "[[ ##") != null or
            std.mem.endsWith(u8, trimmed, "[") or std.mem.endsWith(u8, trimmed, "[[ #");
        if (!maybe_end) {
            state.count = 0;
            state.pending_len = 0;
            return self.emit(beforeMarker(bytes), false, false);
        }
        const drop = if (state.count > lookbehind_max) state.lengths[0] else 0;
        const rest = bytes[drop..];
        if (hasEnd(rest)) {
            state.ended = true;
            const cutoff = drop + beforeMarker(rest).len;
            return self.emit(rstrip(bytes[0..cutoff]), true, false);
        }
        if (drop == 0) return null;
        const fragment = try self.emit(bytes[0..drop], false, false);
        @memmove(self.pending[0..rest.len], rest);
        state.pending_len = rest.len;
        state.count -= 1;
        @memmove(state.lengths[0..state.count], state.lengths[1..][0..state.count]);
        return fragment;
    }

    /// Flush on source exhaustion. Exhaustion need not emit a fragment.
    pub fn finalize(self: *ChatStreamListener) Error!?Fragment {
        errdefer self.reset();
        if (self.state.ended or !self.state.started) return null;
        self.state.ended = true;
        const text = beforeMarker(self.pending[0..self.state.pending_len]);
        return if (text.len == 0) null else self.emit(text, true, true);
    }

    fn emit(self: *ChatStreamListener, bytes: []const u8, last: bool, raw: bool) Error!?Fragment {
        const state = &self.state;
        const text = if (!raw and !state.value_started) lstrip(bytes) else bytes;
        const held = state.whitespace.len;
        assert(held <= self.output.len);
        if (text.len > self.output.len - held) return error.BufferTooSmall;
        const visible = if (raw) text else rstrip(text);
        @memmove(self.output[0..held], state.whitespace);
        @memcpy(self.output[held..][0..text.len], text);
        const all = self.output[0 .. held + text.len];
        if (visible.len == 0) {
            state.whitespace = all;
            return if (last) .{ .text = "", .is_last_chunk = true } else null;
        }
        state.value_started = true;
        state.whitespace = all[held + visible.len ..];
        return .{ .text = all[0 .. held + visible.len], .is_last_chunk = last };
    }
};

fn lstrip(text: []const u8) []const u8 {
    const stripped = unicode.strip(text, false);
    if (stripped.len == 0) return text[text.len..];
    return text[@intFromPtr(stripped.ptr) - @intFromPtr(text.ptr) ..];
}

fn rstrip(text: []const u8) []const u8 {
    const stripped = unicode.strip(text, false);
    if (stripped.len == 0) return text[0..0];
    return text[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(text.ptr) + stripped.len];
}

fn beforeMarker(text: []const u8) []const u8 {
    return text[0 .. std.mem.indexOf(u8, text, "[[") orelse text.len];
}

fn hasEnd(text: []const u8) bool {
    var offset: usize = 0;
    while (std.mem.indexOfPos(u8, text, offset, prefix)) |start| {
        const name = start + prefix.len;
        var end = name;
        while (end < text.len) {
            const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
            if (len > text.len - end) break;
            const cp = std.unicode.utf8Decode(text[end..][0..len]) catch break;
            if (!unicode.isWord(cp)) break;
            end += len;
        }
        if (end > name and std.mem.startsWith(u8, text[end..], suffix)) return true;
        offset = end;
    }
    return false;
}
