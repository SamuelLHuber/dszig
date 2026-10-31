//! Bit-exact port of CPython 3.x `random.Random` (MT19937) for the subset
//! DSPy optimizers use: `Random(seed)`, `shuffle`, and `sample`.
//!
//! Matching Python's stream is what makes a Zig-compiled program select the
//! same few-shot demos as DSPy given the same seed and trainset.

const std = @import("std");
const assert = std.debug.assert;

const state_len = 624;
const shift_len = 397;
const matrix_a: u32 = 0x9908b0df;
const upper_mask: u32 = 0x80000000;
const lower_mask: u32 = 0x7fffffff;

pub const Random = struct {
    mt: [state_len]u32,
    index: u32,

    /// Equivalent to `random.Random(seed)` for a non-negative integer seed
    /// below 2^32 (CPython seeds with the 32-bit words of `abs(seed)`).
    pub fn init(seed: u32) Random {
        var r: Random = .{ .mt = undefined, .index = state_len };
        r.initByArray(&.{seed});
        return r;
    }

    fn initGenrand(r: *Random, s: u32) void {
        r.mt[0] = s;
        var i: u32 = 1;
        while (i < state_len) : (i += 1) {
            const prev = r.mt[i - 1];
            r.mt[i] = (1812433253 *% (prev ^ (prev >> 30))) +% i;
        }
        r.index = state_len;
    }

    fn initByArray(r: *Random, key: []const u32) void {
        assert(key.len > 0);
        r.initGenrand(19650218);
        var i: u32 = 1;
        var j: u32 = 0;
        var k: usize = @max(state_len, key.len);
        while (k > 0) : (k -= 1) {
            const prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ ((prev ^ (prev >> 30)) *% 1664525)) +% key[j] +% j;
            i += 1;
            j += 1;
            if (i >= state_len) {
                r.mt[0] = r.mt[state_len - 1];
                i = 1;
            }
            if (j >= key.len) j = 0;
        }
        k = state_len - 1;
        while (k > 0) : (k -= 1) {
            const prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ ((prev ^ (prev >> 30)) *% 1566083941)) -% i;
            i += 1;
            if (i >= state_len) {
                r.mt[0] = r.mt[state_len - 1];
                i = 1;
            }
        }
        r.mt[0] = 0x80000000;
    }

    fn twist(r: *Random) void {
        var kk: usize = 0;
        while (kk < state_len) : (kk += 1) {
            const y = (r.mt[kk] & upper_mask) | (r.mt[(kk + 1) % state_len] & lower_mask);
            const mag: u32 = if (y & 1 == 1) matrix_a else 0;
            r.mt[kk] = r.mt[(kk + shift_len) % state_len] ^ (y >> 1) ^ mag;
        }
        r.index = 0;
    }

    pub fn nextU32(r: *Random) u32 {
        if (r.index >= state_len) r.twist();
        assert(r.index < state_len);
        var y = r.mt[r.index];
        r.index += 1;
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= (y >> 18);
        return y;
    }

    /// `getrandbits(k)` for 1 <= k <= 32.
    pub fn getrandbits(r: *Random, k: u6) u32 {
        assert(k >= 1);
        assert(k <= 32);
        const shift: u5 = @intCast(32 - @as(u32, k));
        return r.nextU32() >> shift;
    }

    /// `_randbelow_with_getrandbits(n)`: rejection sampling, n in [1, 2^32).
    pub fn randbelow(r: *Random, n: usize) usize {
        assert(n > 0);
        assert(n <= std.math.maxInt(u32));
        const k: u6 = @intCast(std.math.log2_int(usize, n) + 1);
        var x: usize = r.getrandbits(k);
        while (x >= n) x = r.getrandbits(k);
        return x;
    }

    /// `random.shuffle(items)`.
    pub fn shuffle(r: *Random, comptime T: type, items: []T) void {
        if (items.len < 2) return;
        var i: usize = items.len - 1;
        while (i > 0) : (i -= 1) {
            const j = r.randbelow(i + 1);
            std.mem.swap(T, &items[i], &items[j]);
        }
    }

    /// `random.sample(range(n), k)`: writes the chosen population indices
    /// into `out` (length k). `scratch` must hold at least `n` entries; it is
    /// used only by the pool strategy, exactly as in CPython.
    pub fn sampleIndices(r: *Random, n: usize, out: []usize, scratch: []usize) void {
        const k = out.len;
        assert(k <= n);
        var setsize: usize = 21;
        if (k > 5) setsize += ceilPow4(k * 3);
        if (n <= setsize) {
            assert(scratch.len >= n);
            const pool = scratch[0..n];
            for (pool, 0..) |*p, idx| p.* = idx;
            for (out, 0..) |*o, i| {
                const j = r.randbelow(n - i);
                o.* = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            for (out, 0..) |*o, i| {
                var j = r.randbelow(n);
                while (std.mem.indexOfScalar(usize, out[0..i], j) != null) j = r.randbelow(n);
                o.* = j;
            }
        }
    }
};

/// `4 ** ceil(log(x, 4))` as computed by CPython for x >= 1.
fn ceilPow4(x: usize) usize {
    assert(x >= 1);
    var p: usize = 1;
    while (p < x) p *= 4;
    return p;
}

const testing = std.testing;

test "Random(0).getrandbits(32) matches CPython" {
    var r = Random.init(0);
    try testing.expectEqual(@as(u32, 3626764237), r.getrandbits(32));
    try testing.expectEqual(@as(u32, 1654615998), r.getrandbits(32));
    try testing.expectEqual(@as(u32, 3255389356), r.getrandbits(32));
}

test "sample and shuffle stream matches CPython" {
    var r = Random.init(0);
    var scratch: [128]usize = undefined;
    var a: [4]usize = undefined;
    r.sampleIndices(10, &a, &scratch);
    try testing.expectEqualSlices(usize, &.{ 6, 9, 0, 2 }, &a);
    var b: [7]usize = undefined;
    r.sampleIndices(30, &b, &scratch);
    try testing.expectEqualSlices(usize, &.{ 16, 15, 12, 25, 9, 28, 11 }, &b);
    var c: [3]usize = undefined;
    r.sampleIndices(100, &c, &scratch);
    try testing.expectEqualSlices(usize, &.{ 74, 27, 64 }, &c);
    var l = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };
    r.shuffle(usize, &l);
    try testing.expectEqualSlices(usize, &.{ 5, 4, 3, 6, 0, 1, 7, 2 }, &l);
}
