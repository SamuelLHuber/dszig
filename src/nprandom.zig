//! The parts of NumPy 2.x `np.random.default_rng(seed)` that DSPy's SIMBA
//! draws from: `SeedSequence` seeding of PCG64 (XSL-RR), `random()` and
//! `poisson(lam)`, plus `np.percentile` with linear interpolation.

const std = @import("std");
const assert = std.debug.assert;

const pool_size = 4;
const init_a: u32 = 0x43b0d7e5;
const mult_a: u32 = 0x931e8875;
const init_b: u32 = 0x8b51f9dd;
const mult_b: u32 = 0x58f38ded;
const mix_mult_l: u32 = 0xca01f9dd;
const mix_mult_r: u32 = 0x4973f715;
const xshift = 16;

const pcg_mult: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

pub const Generator = struct {
    state: u128,
    inc: u128,

    /// `np.random.default_rng(seed)` for a non-negative integer seed.
    pub fn init(seed: u64) Generator {
        var entropy: [2]u32 = .{ @truncate(seed), @truncate(seed >> 32) };
        const words: []const u32 = if (seed >> 32 == 0) entropy[0..1] else entropy[0..2];
        const pool = mixEntropy(words);
        var state32: [8]u32 = undefined;
        var hash_const = init_b;
        for (&state32, 0..) |*s, i| {
            var v = pool[i % pool_size];
            v ^= hash_const;
            hash_const *%= mult_b;
            v *%= hash_const;
            v ^= v >> xshift;
            s.* = v;
        }
        var words64: [4]u64 = undefined;
        for (&words64, 0..) |*w, i| w.* = @as(u64, state32[2 * i]) | (@as(u64, state32[2 * i + 1]) << 32);
        const initstate = (@as(u128, words64[0]) << 64) | words64[1];
        const initseq = (@as(u128, words64[2]) << 64) | words64[3];
        var g: Generator = .{ .state = 0, .inc = (initseq << 1) | 1 };
        g.step();
        g.state +%= initstate;
        g.step();
        return g;
    }

    fn step(g: *Generator) void {
        g.state = g.state *% pcg_mult +% g.inc;
    }

    pub fn next64(g: *Generator) u64 {
        g.step();
        const hi: u64 = @truncate(g.state >> 64);
        const lo: u64 = @truncate(g.state);
        const rot: u6 = @truncate(g.state >> 122);
        return std.math.rotr(u64, hi ^ lo, rot);
    }

    /// `rng.random()`.
    pub fn random(g: *Generator) f64 {
        return @as(f64, @floatFromInt(g.next64() >> 11)) * (1.0 / 9007199254740992.0);
    }

    /// `rng.poisson(lam)` (NumPy's `random_poisson`).
    pub fn poisson(g: *Generator, lam: f64) u64 {
        assert(lam >= 0);
        if (lam >= 10) return g.poissonPtrs(lam);
        if (lam == 0) return 0;
        const enlam = @exp(-lam);
        var x: u64 = 0;
        var prod: f64 = 1.0;
        while (true) {
            prod *= g.random();
            if (prod > enlam) x += 1 else return x;
        }
    }

    fn poissonPtrs(g: *Generator, lam: f64) u64 {
        const slam = @sqrt(lam);
        const loglam = @log(lam);
        const b = 0.931 + 2.53 * slam;
        const a = -0.059 + 0.02483 * b;
        const invalpha = 1.1239 + 1.1328 / (b - 3.4);
        const vr = 0.9277 - 3.6224 / (b - 2);
        while (true) {
            const u = g.random() - 0.5;
            const v = g.random();
            const us = 0.5 - @abs(u);
            const k = @floor((2 * a / us + b) * u + lam + 0.43);
            if (us >= 0.07 and v <= vr) return @intFromFloat(k);
            if (k < 0 or (us < 0.013 and v > us)) continue;
            if (@log(v) + @log(invalpha) - @log(a / (us * us) + b) <= -lam + k * loglam - loggam(k + 1)) {
                return @intFromFloat(k);
            }
        }
    }
};

fn hashmix(value: u32, hash_const: *u32) u32 {
    var v = value ^ hash_const.*;
    hash_const.* *%= mult_a;
    v *%= hash_const.*;
    v ^= v >> xshift;
    return v;
}

fn mix(x: u32, y: u32) u32 {
    var r = mix_mult_l *% x -% mix_mult_r *% y;
    r ^= r >> xshift;
    return r;
}

fn mixEntropy(entropy: []const u32) [pool_size]u32 {
    var mixer: [pool_size]u32 = undefined;
    var hash_const = init_a;
    for (&mixer, 0..) |*m, i| m.* = hashmix(if (i < entropy.len) entropy[i] else 0, &hash_const);
    for (0..pool_size) |src| {
        for (0..pool_size) |dst| {
            if (src != dst) mixer[dst] = mix(mixer[dst], hashmix(mixer[src], &hash_const));
        }
    }
    if (entropy.len > pool_size) {
        for (entropy[pool_size..]) |e| {
            for (&mixer) |*m| m.* = mix(m.*, hashmix(e, &hash_const));
        }
    }
    return mixer;
}

/// NumPy's `random_loggam`.
fn loggam(x: f64) f64 {
    const coef = [10]f64{
        8.333333333333333e-02,  -2.777777777777778e-03, 7.936507936507937e-04,
        -5.952380952380952e-04, 8.417508417508418e-04,  -1.917526917526918e-03,
        6.410256410256410e-03,  -2.955065359477124e-02, 1.796443723688307e-01,
        -1.39243221690590e+00,
    };
    if (x == 1.0 or x == 2.0) return 0.0;
    const n: f64 = if (x < 7.0) @floor(7 - x) else 0;
    var x0 = x + n;
    const x2 = (1.0 / x0) * (1.0 / x0);
    var gl0 = coef[9];
    var k: usize = 9;
    while (k > 0) {
        k -= 1;
        gl0 *= x2;
        gl0 += coef[k];
    }
    var gl = gl0 / x0 + 0.5 * 1.8378770664093453e+00 + (x0 - 0.5) * @log(x0) - x0;
    if (x < 7.0) {
        var i: f64 = 1;
        while (i <= n) : (i += 1) {
            gl -= @log(x0 - 1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

/// `np.percentile(values, q)` (method `linear`); sorts `values` in place.
pub fn percentile(values: []f64, q: f64) f64 {
    assert(values.len > 0);
    assert(q >= 0 and q <= 100);
    std.sort.pdq(f64, values, {}, std.sort.asc(f64));
    const virtual = (q / 100) * @as(f64, @floatFromInt(values.len - 1));
    const prev_f = @floor(virtual);
    const prev: usize = @intFromFloat(prev_f);
    const next = @min(prev + 1, values.len - 1);
    const t = virtual - prev_f;
    const a = values[prev];
    const b = values[next];
    const diff = b - a;
    return if (t >= 0.5) b - diff * (1 - t) else a + diff * t;
}

const testing = std.testing;

test "default_rng(0) seeds PCG64 like NumPy" {
    const g: Generator = .init(0);
    try testing.expectEqual(@as(u128, 35399562948360463058890781895381311971), g.state);
    try testing.expectEqual(@as(u128, 87136372517582989555478159403783844777), g.inc);
}

test "random and poisson match NumPy 2.4.6" {
    var g: Generator = .init(5);
    try testing.expectEqual(@as(f64, 0.8050029237453802), g.random());
    try testing.expectEqual(@as(f64, 0.8079407897364937), g.random());
    try testing.expectEqual(@as(f64, 0.515325561042142), g.random());
    var p: Generator = .init(0);
    const lams = [_]f64{ 0.5, 1.0, 0.25, 2.0, 0.0, 12.0, 35.5, 3.0 };
    const want = [_]u64{ 1, 0, 0, 7, 0, 16, 40, 4 };
    for (lams, want) |lam, w| try testing.expectEqual(w, p.poisson(lam));
}

test "percentile interpolates linearly" {
    var v = [_]f64{ 0, 1, 1, 0, 1, 0.5, 0.25 };
    try testing.expectEqual(@as(f64, 0.0), percentile(&v, 10));
    try testing.expectEqual(@as(f64, 1.0), percentile(&v, 90));
    var w = [_]f64{ 3, 1, 2, 4 };
    try testing.expectEqual(@as(f64, 1.3), percentile(&w, 10));
}
