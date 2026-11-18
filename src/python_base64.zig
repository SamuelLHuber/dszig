//! CPython's default non-validating b64decode. Nonalphabet ASCII and leading
//! or excess padding are ignored; incomplete final quartets still refuse.
const std = @import("std");
const Value = @import("runtime_value.zig").Value;

pub const Output = union(enum) { count, bytes: []u8, numbers: []Value };

/// Additional trailing padding is virtual, so streaming fallback never copies
/// the encoded chunk. The embeddings path passes zero and retains its original
/// strict quartet boundary and direct little-endian float output.
pub fn decode(text: []const u8, destination: Output, padding: u2) error{InvalidBase64}!usize {
    // CPython validates the whole string as ASCII before decoding, even when
    // an earlier padded quartet would otherwise terminate processing.
    for (text) |c| if (c > 127) return error.InvalidBase64;
    var chunk: [4]u8 = undefined;
    var position: u3 = 0;
    var previous: u8 = 0;
    var pads: u3 = 0;
    var length: usize = 0;
    for (text) |c| {
        if (c == '=') {
            if (position >= 2) {
                pads += 1;
                if (@as(u4, position) + pads >= 4) return length;
            }
            continue;
        }
        const digit: u8 = switch (c) {
            'A'...'Z' => c - 'A',
            'a'...'z' => c - 'a' + 26,
            '0'...'9' => c - '0' + 52,
            '+' => 62,
            '/' => 63,
            else => continue,
        };
        pads = 0;
        const byte: ?u8 = switch (position) {
            0 => null,
            1 => (previous << 2) | (digit >> 4),
            2 => (previous << 4) | (digit >> 2),
            3 => (previous << 6) | digit,
            else => unreachable,
        };
        if (byte) |b| {
            switch (destination) {
                .count => {},
                .bytes => |output| {
                    if (length >= output.len) return error.InvalidBase64;
                    output[length] = b;
                },
                .numbers => |output| {
                    chunk[length % 4] = b;
                    if (length % 4 == 3) {
                        if (length / 4 >= output.len) return error.InvalidBase64;
                        const number: f32 = @bitCast(std.mem.readInt(u32, &chunk, .little));
                        output[length / 4] = .{ .float = number };
                    }
                },
            }
            length += 1;
        }
        previous = digit;
        position = if (position == 3) 0 else position + 1;
    }
    if (position == 0) return length;
    if (position >= 2 and @as(u4, position) + pads + padding >= 4) return length;
    return error.InvalidBase64;
}
