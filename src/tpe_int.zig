//! Optuna 4.5.0's default, independent integer TPE for `suggest_int(name,
//! low, high)` (step=1, log=false), as used by BootstrapFewShotWithOptuna.
//! This is NOT a categorical or multivariate approximation. Failed trials are
//! excluded, complete trials are split stably for a maximizing study, and
//! each parameter consumes its own 24 mixture/24 quantile draws in call order.
//!
//! `init(seed)` reproduces NumPy RandomState's two separately seeded streams.
//! `initUnseeded(io)` lazily obtains independent entropy for those streams.
//! The sampler owns no allocations. `sample` uses caller-owned scratch; pass
//! an arena and reset it between calls. Missing historical parameters are null.
//! Integer endpoints must have exact binary64 half-step boundaries, which
//! includes every physically materializable predictor demo index. This module
//! does not expose Optuna's other distributions, pruning, or custom samplers.
//!
//! Sources: optuna/samplers/_tpe/{sampler,parzen_estimator,
//! probability_distributions,_truncnorm}.py and optuna/_transform.py, v4.5.0.
//! NumPy scalar quicksort tie order matches the locked Darwin/arm64 oracle.
//! Numerical bit parity is NOT complete: a high-precision logarithm can differ
//! from Darwin/NumPy at rounding boundaries. The independent oracle diagnostic
//! retains six differences across 10009 log1p inputs; those differences can
//! change an acquisition winner near a mathematical tie. Native consumer tests
//! keep source expectations, without requiring every private helper to share
//! Darwin's rounding errors. No platform libc or runtime Python is used.

// Optuna-derived portions: MIT License.
// Copyright (c) 2018 Preferred Networks, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const pyrandom = @import("pyrandom.zig");

pub const n_startup_trials = 10;
pub const n_ei_candidates = 24;
pub const max_exact_bound: i64 = (1 << 52) - 1;

pub const Param = struct {
    name: []const u8,
    low: i64,
    high: i64,
    /// An enqueued fixed parameter bypasses sampling, even outside the bounds,
    /// just as Trial._is_fixed_param does (upstream additionally warns).
    fixed: ?i64 = null,
};

pub const State = enum { complete, fail };
pub const Trial = struct {
    /// Ordered like `params` passed to sample; null means never suggested.
    params: []const ?i64,
    value: f64,
    state: State = .complete,
};

pub const Error = Allocator.Error || std.Io.RandomSecureError || error{
    InvalidBounds,
    UnsupportedIntegerBounds,
    HistoryShapeMismatch,
};

pub const Sampler = struct {
    rng: ?pyrandom.Random,
    startup_rng: ?pyrandom.Random,
    io: ?std.Io = null,

    pub fn init(seed: u32) Sampler {
        return .{ .rng = pyrandom.Random.initNumpy(seed), .startup_rng = pyrandom.Random.initNumpy(seed) };
    }

    /// No deterministic default seed. Fixed/singleton parameters do not even
    /// initialize a stream; its entropy is requested on the first actual draw.
    pub fn initUnseeded(io: std.Io) Sampler {
        return .{ .rng = null, .startup_rng = null, .io = io };
    }

    fn numpy(self: *Sampler, tpe: bool) std.Io.RandomSecureError!*pyrandom.Random {
        const stream = if (tpe) &self.rng else &self.startup_rng;
        if (stream.* == null) {
            var bytes: [4]u8 = undefined;
            try self.io.?.randomSecure(&bytes);
            stream.* = pyrandom.Random.initNumpy(std.mem.readInt(u32, &bytes, .little));
        }
        return &stream.*.?;
    }

    /// History is in trial-number order, as in an Optuna study. The output is
    /// in suggestion order, NOT alphabetically sorted (multivariate=false).
    /// Duplicate names return the first suggestion without consuming RNG.
    pub fn sample(self: *Sampler, scratch: Allocator, params: []const Param, trials: []const Trial, out: []i64) Error!void {
        if (out.len != params.len) return error.HistoryShapeMismatch;
        var n: usize = 0;
        for (trials) |trial| {
            if (trial.params.len != params.len) return error.HistoryShapeMismatch;
            if (isComplete(trial)) n += 1;
        }
        var below: ?[]bool = null;
        for (params, out, 0..) |param, *chosen, pi| {
            if (param.low > param.high) return error.InvalidBounds;
            if (param.low < -max_exact_bound or param.high > max_exact_bound) return error.UnsupportedIntegerBounds;
            const previous: ?usize = for (params[0..pi], 0..) |prior, j| {
                if (std.mem.eql(u8, prior.name, param.name)) break j;
            } else null;
            if (previous) |j| {
                chosen.* = out[j];
            } else if (param.fixed) |fixed| {
                chosen.* = fixed;
            } else if (param.low == param.high) {
                chosen.* = param.low;
            } else if (n < n_startup_trials) {
                const rng = try self.numpy(false);
                const low = asFloat(param.low) - 0.5;
                const high = asFloat(param.high) + 0.5;
                const sampled = low + (high - low) * rng.random();
                chosen.* = discretize(param, sampled);
            } else {
                if (below == null) below = try splitTrials(scratch, trials, n);
                chosen.* = try self.sampleTpe(scratch, param, pi, trials, below.?);
            }
        }
    }

    fn sampleTpe(self: *Sampler, scratch: Allocator, param: Param, pi: usize, trials: []const Trial, below: []const bool) Error!i64 {
        var lower: std.ArrayList(f64) = .empty;
        var upper: std.ArrayList(f64) = .empty;
        for (trials, below) |trial, is_below| {
            if (!isComplete(trial)) continue;
            if (trial.params[pi]) |x| try (if (is_below) &lower else &upper).append(scratch, asFloat(x));
        }
        const mpe_below = try Estimator.init(scratch, param, lower.items);
        const mpe_above = try Estimator.init(scratch, param, upper.items);
        const samples = try mpe_below.sample(scratch, try self.numpy(true));
        const terms = try scratch.alloc(f64, @max(mpe_below.mus.len, mpe_above.mus.len));
        var best: usize = 0;
        var best_acq: f64 = -std.math.inf(f64);
        for (samples, 0..) |x, i| {
            // Source evaluates unique integer bins before expanding them back
            // to candidate order. Equal bins cannot beat their first instance.
            if (std.mem.indexOfScalar(i64, samples[0..i], x) != null) continue;
            const acq = mpe_below.logPdf(x, terms[0..mpe_below.mus.len]) - mpe_above.logPdf(x, terms[0..mpe_above.mus.len]);
            // np.argmax selects the first maximum, including the first NaN.
            if (i == 0 or (!std.math.isNan(best_acq) and (std.math.isNan(acq) or acq > best_acq))) {
                best = i;
                best_acq = acq;
            }
        }
        return samples[best];
    }
};

fn isComplete(trial: Trial) bool {
    return trial.state == .complete and !std.math.isNan(trial.value);
}

fn asFloat(x: i64) f64 {
    return @floatFromInt(x);
}

fn splitTrials(arena: Allocator, trials: []const Trial, n: usize) Allocator.Error![]bool {
    const ranked = try arena.alloc(usize, n);
    var count: usize = 0;
    for (trials, 0..) |trial, i| if (isComplete(trial)) {
        ranked[count] = i;
        count += 1;
    };
    assert(count == n);
    // Explicit trial-number tie break gives Python's stable reverse sort,
    // without quadratic sorting for a long study.
    std.sort.pdq(usize, ranked, trials, struct {
        fn less(history: []const Trial, a: usize, b: usize) bool {
            return history[a].value > history[b].value or (history[a].value == history[b].value and a < b);
        }
    }.less);
    const n_below = @min(@as(usize, @intFromFloat(@ceil(0.1 * @as(f64, @floatFromInt(n))))), 25);
    const below = try arena.alloc(bool, trials.len);
    @memset(below, false);
    for (ranked[0..@min(n_below, n)]) |i| below[i] = true;
    return below;
}

fn roundEven(x: f64) f64 {
    const low = @floor(x);
    const fraction = x - low;
    if (fraction < 0.5) return low;
    if (fraction > 0.5) return low + 1.0;
    return if (@mod(low, 2.0) == 0) low else low + 1.0;
}

fn discretize(param: Param, x: f64) i64 {
    const low = asFloat(param.low);
    const high = asFloat(param.high);
    // NumPy rounds relative to low, not to the origin.
    return @intFromFloat(@min(high, @max(low, low + roundEven(x - low))));
}

const Estimator = struct {
    param: Param,
    weights: []f64,
    log_weights: []f64,
    mus: []f64,
    sigmas: []f64,
    log_normalizers: []f64,

    fn init(arena: Allocator, param: Param, observations: []const f64) Allocator.Error!Estimator {
        const m = observations.len;
        const n = m + 1;
        const weights = try arena.alloc(f64, n);
        defaultWeights(weights[0..m]);
        weights[m] = 1.0;
        const total = pairwiseSum(weights);
        const log_weights = try arena.alloc(f64, n);
        for (weights, log_weights) |*w, *lw| {
            w.* /= total;
            lw.* = logarithm(w.*);
        }
        const mus = try arena.alloc(f64, n);
        @memcpy(mus[0..m], observations);
        const low = asFloat(param.low);
        const high = asFloat(param.high);
        const width = high - low + 1.0;
        mus[m] = 0.5 * (low + high);
        const indices = try arena.alloc(usize, n);
        argsort(mus, indices);
        const sigmas = try arena.alloc(f64, n);
        const min_sigma = width / @min(100.0, 1.0 + @as(f64, @floatFromInt(n)));
        for (indices, 0..) |index, i| {
            const mu = mus[index];
            const previous = if (i == 0) low - 0.5 else mus[indices[i - 1]];
            const next = if (i + 1 == n) high + 0.5 else mus[indices[i + 1]];
            var sigma = @max(mu - previous, next - mu);
            // consider_endpoints=False. Both operations run for n>=2.
            if (n >= 2 and i == 0) sigma = next - mu;
            if (n >= 2 and i + 1 == n) sigma = mu - previous;
            sigmas[index] = @min(width, @max(min_sigma, sigma));
        }
        sigmas[m] = width;
        const log_normalizers = try arena.alloc(f64, n);
        for (mus, sigmas, log_normalizers) |mu, sigma, *normalizer| {
            normalizer.* = logMass((low - 0.5 - mu) / sigma, (high + 0.5 - mu) / sigma);
        }
        return .{ .param = param, .weights = weights, .log_weights = log_weights, .mus = mus, .sigmas = sigmas, .log_normalizers = log_normalizers };
    }

    fn sample(self: Estimator, arena: Allocator, rng: *pyrandom.Random) Allocator.Error![n_ei_candidates]i64 {
        // RandomState.choice(p=weights) uses sequential cumsum (normalizing
        // by its final element), not the pairwise sum used above.
        const cdf = try arena.alloc(f64, self.weights.len);
        var cumulative: f64 = 0;
        for (self.weights, cdf) |weight, *c| {
            cumulative += weight;
            c.* = cumulative;
        }
        for (cdf) |*c| c.* /= cumulative;
        var active: [n_ei_candidates]usize = undefined;
        for (&active) |*index| {
            const q = rng.random();
            var low: usize = 0;
            var high = cdf.len;
            while (low < high) {
                const middle = low + (high - low) / 2;
                if (cdf[middle] <= q) low = middle + 1 else high = middle;
            }
            index.* = low;
        }
        var qs: [n_ei_candidates]f64 = undefined;
        var aa: [n_ei_candidates]f64 = undefined;
        var bb: [n_ei_candidates]f64 = undefined;
        var ys: [n_ei_candidates]f64 = undefined;
        var left: [n_ei_candidates]bool = undefined;
        for (active, &qs, &aa, &bb, &ys, &left) |index, *q, *a, *b, *y, *l| {
            q.* = rng.random();
            a.* = (asFloat(self.param.low) - 0.5 - self.mus[index]) / self.sigmas[index];
            b.* = (asFloat(self.param.high) + 0.5 - self.mus[index]) / self.sigmas[index];
            l.* = a.* < 0;
            y.* = if (l.*)
                logSum(logCdf(a.*), logarithm(q.*) + self.log_normalizers[index])
            else
                logSum(logCdf(-b.*), logOnePlus(-q.*) + self.log_normalizers[index]);
        }
        var xs: [n_ei_candidates]f64 = undefined;
        // _ndtri_exp terminates collectively for each ppf branch; solving
        // candidates independently changes its last Newton update.
        inverseLogCdfBatch(&ys, &left, true, &xs);
        inverseLogCdfBatch(&ys, &left, false, &xs);
        var out: [n_ei_candidates]i64 = undefined;
        for (active, qs, aa, bb, xs, left, &out) |index, q, a, b, x, l, *chosen| {
            const z = if (q == 0) a else if (q == 1) b else if (l) x else -x;
            chosen.* = discretize(self.param, z * self.sigmas[index] + self.mus[index]);
        }
        return out;
    }

    fn logPdf(self: Estimator, x: i64, terms: []f64) f64 {
        assert(terms.len == self.weights.len);
        var maximum: f64 = -std.math.inf(f64);
        for (self.mus, self.sigmas, self.log_normalizers, self.log_weights, terms) |mu, sigma, normalizer, lw, *term| {
            term.* = logMass((asFloat(x) - 0.5 - mu) / sigma, (asFloat(x) + 0.5 - mu) / sigma);
            term.* -= normalizer;
            term.* += lw;
            maximum = @max(maximum, term.*);
        }
        if (std.math.isNegativeInf(maximum)) maximum = 0;
        for (terms) |*term| term.* = @exp(term.* - maximum);
        return logarithm(pairwiseSum(terms)) + maximum;
    }
};

fn defaultWeights(out: []f64) void {
    const n = out.len;
    if (n < 25) {
        @memset(out, 1.0);
        return;
    }
    const ramp = n - 25;
    const start = 1.0 / @as(f64, @floatFromInt(n));
    if (ramp > 0) {
        const step = (1.0 - start) / @as(f64, @floatFromInt(@max(ramp - 1, 1)));
        for (out[0..ramp], 0..) |*weight, i| weight.* = @as(f64, @floatFromInt(i)) * step + start;
        if (ramp > 1) out[ramp - 1] = 1.0;
    }
    @memset(out[ramp..], 1.0);
}

// NumPy's contiguous binary64 sum uses eight accumulators and 128-element
// recursive blocks. It is distinct from the sequential choice CDF above.
fn pairwiseSum(xs: []const f64) f64 {
    if (xs.len < 8) {
        var result: f64 = 0;
        for (xs) |x| result += x;
        return result;
    }
    if (xs.len <= 128) {
        var r: [8]f64 = xs[0..8].*;
        var i: usize = 8;
        while (i < xs.len - xs.len % 8) : (i += 8) {
            for (&r, xs[i..][0..8]) |*acc, x| acc.* += x;
        }
        var result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < xs.len) : (i += 1) result += xs[i];
        return result;
    }
    const middle = (xs.len / 2) - (xs.len / 2) % 8;
    return pairwiseSum(xs[0..middle]) + pairwiseSum(xs[middle..]);
}

// The truncated-normal equations below are translated from Optuna 4.5's
// SciPy-derived _truncnorm.py.
// Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Exact binary64 constants observed in the locked source module. In particular,
// do not recompute sqrt(3)/pi at comptime with a higher-precision pi.
const normal_log_c: f64 = 0x1.d67f1c864beb4p-1;
const inverse_cdf_approx_c: f64 = 0x1.1a47c7ee5a514p-1;
const sqrt_two: f64 = 1.4142135623730951;

/// Evaluate with the pinned compiler's binary128 kernel, then round to binary64.
/// Its binary64 kernel misses an actual source log-sum-exp boundary by one ulp;
/// cancellation against the maximum amplifies that error into a different TPE
/// acquisition winner. Extended precision avoids that known miss without
/// input-specific corrections, native C linkage, or runtime dependencies.
fn logarithm(x: f64) f64 {
    @setFloatMode(.strict);
    return @floatCast(@log(@as(f128, x)));
}

/// Compensate the rounding of 1+x before taking its logarithm. The correction
/// recovers the lost low part; simply using log(1+x) loses tiny quantiles.
/// std.math.log1p's rational kernel is one ulp off the pinned NumPy result at
/// a central prior boundary. This correction fixes that boundary, but using a
/// more accurate real logarithm does not establish Darwin libm bit parity.
fn logOnePlus(x: f64) f64 {
    @setFloatMode(.strict);
    if (x == 0 or std.math.isNan(x) or x == std.math.inf(f64)) return x;
    if (x == -1) return -std.math.inf(f64);
    if (x < -1) return std.math.nan(f64);
    const u = 1.0 + x;
    if (u == 1) return x;
    return logarithm(u) - ((u - 1.0) - x) / u;
}

fn logSum(p: f64, q: f64) f64 {
    if (p == q) return p + 0.6931471805599453;
    return @max(p, q) + logOnePlus(@exp(-@abs(p - q)));
}

fn logDiff(p: f64, q: f64) f64 {
    return p + logOnePlus(-@exp(q - p));
}

fn normalCdf(x: f64) f64 {
    const z = x / sqrt_two;
    if (z < -1.0 / sqrt_two) return 0.5 * erfcPositive(-z);
    if (z < 1.0 / sqrt_two) return 0.5 + 0.5 * erf(z);
    return 1.0 - 0.5 * erfcPositive(z);
}

fn logCdf(x: f64) f64 {
    if (x > 6) return -normalCdf(-x);
    if (x > -20) return logarithm(normalCdf(x));
    const log_lhs = -0.5 * (x * x) - logarithm(-x) - normal_log_c;
    var previous: f64 = 0;
    var rhs: f64 = 1;
    var numerator: f64 = 1;
    var denominator: f64 = 1;
    const denominator_factor = 1.0 / (x * x);
    var sign: f64 = 1;
    var i: usize = 0;
    while (@abs(previous - rhs) > std.math.floatEps(f64)) {
        i += 1;
        previous = rhs;
        sign = -sign;
        denominator *= denominator_factor;
        numerator *= @as(f64, @floatFromInt(2 * i - 1));
        rhs += sign * numerator * denominator;
    }
    return log_lhs + logarithm(rhs);
}

fn logMass(a: f64, b: f64) f64 {
    if (b <= 0) return logDiff(logCdf(b), logCdf(a));
    if (a > 0) return logDiff(logCdf(-a), logCdf(-b));
    // _ndtr intentionally uses erf here, unlike _ndtr_single above.
    const cdf_a = 0.5 + 0.5 * erf(a / sqrt_two);
    const cdf_minus_b = 0.5 + 0.5 * erf(-b / sqrt_two);
    return logOnePlus(-cdf_a - cdf_minus_b);
}

fn inverseLogCdfBatch(ys: []const f64, left: []const bool, selected: bool, xs: []f64) void {
    assert(ys.len == left.len and xs.len == ys.len);
    var found = false;
    for (ys, left, xs) |y, is_left, *x| {
        if (is_left != selected) continue;
        found = true;
        const z = if (y > -1e-2) logarithm(-std.math.expm1(y)) else y;
        x.* = if (z < -5) -@sqrt(-2.0 * (z + normal_log_c)) else -inverse_cdf_approx_c * logarithm(std.math.expm1(-z));
    }
    if (!found) return;
    for (0..100) |_| {
        var converged = true;
        for (ys, left, xs) |y, is_left, *x| {
            if (is_left != selected) continue;
            const z = if (y > -1e-2) logarithm(-std.math.expm1(y)) else y;
            const cdf = logCdf(x.*);
            const pdf = -0.5 * (x.* * x.*) - normal_log_c;
            const dx = (cdf - z) * @exp(cdf - pdf);
            x.* -= dx;
            if (!(@abs(dx) < 1e-8 * @abs(x.*))) converged = false;
        }
        if (converged) break;
    }
    for (ys, left, xs) |y, is_left, *x| {
        if (is_left == selected and y > -1e-2) x.* *= -1;
    }
}

// erf/erfc rational coefficients and split-exponential evaluation derive from
// FreeBSD msun/s_erf.c (also the source of Optuna's _erf.py).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
const erx = 8.45062911510467529297e-01;
const pp = [_]f64{ 1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02, -5.77027029648944159157e-03, -2.37630166566501626084e-05 };
const qq = [_]f64{ 1, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03, 1.32494738004321644526e-04, -3.96022827877536812320e-06 };
const pa = [_]f64{ -2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01, 3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02, -2.16637559486879084300e-03 };
const qa = [_]f64{ 1, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02, 1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02 };
const ra = [_]f64{ -9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e01, -6.23753324503260060396e01, -1.62396669462573470355e02, -1.84605092906711035994e02, -8.12874355063065934246e01, -9.81432934416914548592e00 };
const sa = [_]f64{ 1, 1.96512716674392571292e01, 1.37657754143519042600e02, 4.34565877475229228821e02, 6.45387271733267880336e02, 4.29008140027567833386e02, 1.08635005541779435134e02, 6.57024977031928170135e00, -6.04244152148580987438e-02 };
const rb = [_]f64{ -9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e01, -1.60636384855821916062e02, -6.37566443368389627722e02, -1.02509513161107724954e03, -4.83519191608651397019e02 };
const sb = [_]f64{ 1, 3.03380607434824582924e01, 3.25792512996573918826e02, 1.53672958608443695994e03, 3.19985821950859553908e03, 2.55305040643316442583e03, 4.74528541206955367215e02, -2.24409524465858183362e01 };

fn polynomial(coefficients: []const f64, x: f64) f64 {
    var result = coefficients[coefficients.len - 1];
    var i = coefficients.len - 1;
    while (i > 0) {
        i -= 1;
        result = result * x + coefficients[i];
    }
    return result;
}

fn erf(x: f64) f64 {
    const a = @abs(x);
    var result: f64 = undefined;
    if (a < 0x1p-28) {
        result = x + 1.28379167095512586316e-01 * x;
        return result;
    } else if (a < 0.84375) {
        const z = x * x;
        // Preserve Optuna _erf.py's rational factorization. Regrouping as
        // x + x*R changes a CDF bin boundary by one ulp, which can turn a
        // source-asymmetric TPE acquisition into an exact tie.
        return x * (1.0 + polynomial(&pp, z) / polynomial(&qq, z));
    } else if (a < 1.25) {
        result = erx + polynomial(&pa, a - 1.0) / polynomial(&qa, a - 1.0);
    } else if (a >= 6.0) {
        result = 1.0;
    } else {
        result = 1.0 - erfcPositive(a);
    }
    return if (x < 0) -result else result;
}

fn erfcPositive(x: f64) f64 {
    assert(x >= 0);
    if (x < 0x1p-56) return 1.0 - x;
    if (x < 0.84375) {
        const z = x * x;
        const r = polynomial(&pp, z) / polynomial(&qq, z);
        return if (x < 0.25) 1.0 - (x + x * r) else 0.5 - ((x - 0.5) + x * r);
    }
    if (x < 1.25) return (1.0 - erx) - polynomial(&pa, x - 1.0) / polynomial(&qa, x - 1.0);
    if (x >= 28.0) return 0;
    const s = 1.0 / (x * x);
    const ratio = if (x < 1.0 / 0.35) polynomial(&ra, s) / polynomial(&sa, s) else polynomial(&rb, s) / polynomial(&sb, s);
    const bits: u64 = @bitCast(x);
    const z: f64 = @bitCast(bits & 0xffffffff00000000);
    return (@exp(-z * z - 0.5625) * @exp((z - x) * (z + x) + ratio)) / x;
}

// NumPy scalar aquicksort_/aheapsort_ translations, matching the native KNN
// convention but operating on binary64 Parzen means.
// Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the NumPy Developers nor the names of any contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
fn argsort(scores: []const f64, indices: []usize) void {
    assert(scores.len == indices.len);
    for (indices, 0..) |*index, i| index.* = i;
    if (indices.len < 2) return;
    const Partition = struct { left: usize, right: usize, depth: i32 };
    var stack: [@bitSizeOf(usize)]Partition = undefined;
    var stack_len: usize = 0;
    var part: Partition = .{ .left = 0, .right = indices.len - 1, .depth = 2 * @as(i32, @intCast(std.math.log2_int(usize, indices.len))) };
    while (true) {
        if (part.depth < 0) {
            heapArgsort(scores, indices[part.left .. part.right + 1]);
        } else {
            while (part.right - part.left > 15) {
                const pivot = partition(scores, indices, part.left, part.right);
                part.depth -= 1;
                assert(stack_len < stack.len);
                if (pivot - part.left < part.right - pivot) {
                    stack[stack_len] = .{ .left = pivot + 1, .right = part.right, .depth = part.depth };
                    part.right = pivot - 1;
                } else {
                    stack[stack_len] = .{ .left = part.left, .right = pivot - 1, .depth = part.depth };
                    part.left = pivot + 1;
                }
                stack_len += 1;
            }
            insertionArgsort(scores, indices, part.left, part.right);
        }
        if (stack_len == 0) break;
        stack_len -= 1;
        part = stack[stack_len];
    }
}

fn partition(scores: []const f64, indices: []usize, left: usize, right: usize) usize {
    const middle = left + (right - left) / 2;
    if (scores[indices[middle]] < scores[indices[left]]) std.mem.swap(usize, &indices[middle], &indices[left]);
    if (scores[indices[right]] < scores[indices[middle]]) std.mem.swap(usize, &indices[right], &indices[middle]);
    if (scores[indices[middle]] < scores[indices[left]]) std.mem.swap(usize, &indices[middle], &indices[left]);
    const pivot = scores[indices[middle]];
    var i = left;
    var j = right - 1;
    std.mem.swap(usize, &indices[middle], &indices[j]);
    while (true) {
        i += 1;
        while (scores[indices[i]] < pivot) i += 1;
        j -= 1;
        while (pivot < scores[indices[j]]) j -= 1;
        if (i >= j) break;
        std.mem.swap(usize, &indices[i], &indices[j]);
    }
    std.mem.swap(usize, &indices[i], &indices[right - 1]);
    return i;
}

fn insertionArgsort(scores: []const f64, indices: []usize, left: usize, right: usize) void {
    var i = left + 1;
    while (i <= right) : (i += 1) {
        const index = indices[i];
        var j = i;
        while (j > left and scores[index] < scores[indices[j - 1]]) : (j -= 1) indices[j] = indices[j - 1];
        indices[j] = index;
    }
}

fn heapArgsort(scores: []const f64, indices: []usize) void {
    var n = indices.len;
    var root = n / 2;
    while (root > 0) : (root -= 1) heapSift(scores, indices, root, n, indices[root - 1]);
    while (n > 1) {
        const temporary = indices[n - 1];
        indices[n - 1] = indices[0];
        n -= 1;
        heapSift(scores, indices, 1, n, temporary);
    }
}

fn heapSift(scores: []const f64, indices: []usize, root: usize, count: usize, temporary: usize) void {
    var parent = root;
    while (parent <= count / 2) {
        var child = parent * 2;
        if (child < count and scores[indices[child - 1]] < scores[indices[child]]) child += 1;
        if (!(scores[temporary] < scores[indices[child - 1]])) break;
        indices[parent - 1] = indices[child - 1];
        parent = child;
    }
    indices[parent - 1] = temporary;
}

const testing = std.testing;

// Every expected suggestion below is recorded by the actual Optuna 4.5.0
// oracle in tools/dspy_reference/bootstrap_optuna_oracle.py. Objectives are
// independent of the implementation: a wrong native choice cannot repair an
// expected history by changing a fixture.
test "integer TPE source sequences include startup and independent post-startup sampling" {
    const expected = [_][3]i64{
        .{ 3, 8, 4 },  .{ 3, 4, 4 },  .{ 2, 6, 4 },  .{ 2, 11, 4 },
        .{ 5, 1, 4 },  .{ 4, 4, 4 },  .{ 3, 12, 4 }, .{ 0, -6, 4 },
        .{ 0, 10, 4 }, .{ 4, 11, 4 }, .{ 1, -2, 4 }, .{ 2, 8, 4 },
        .{ 1, 13, 4 }, .{ 3, 8, 4 },  .{ 2, 8, 4 },  .{ 4, 10, 4 },
        .{ 1, 1, 4 },  .{ 3, 6, 4 },  .{ 5, 13, 4 }, .{ 2, 6, 4 },
        .{ 4, 9, 4 },  .{ 3, 7, 4 },  .{ 3, 10, 4 }, .{ 2, 1, 4 },
        .{ 1, 4, 4 },  .{ 3, 11, 4 }, .{ 4, 8, 4 },  .{ 5, -7, 4 },
        .{ 4, 12, 4 }, .{ 4, 3, 4 },  .{ 5, -2, 4 }, .{ 4, 4, 4 },
        .{ 4, 5, 4 },  .{ 3, 0, 4 },  .{ 4, 2, 4 },  .{ 5, 2, 4 },
    };
    try expectSequence(0, .{
        .{ .name = "z", .low = 0, .high = 5 },
        .{ .name = "a", .low = -7, .high = 13 },
        .{ .name = "fixed", .low = 4, .high = 4 },
    }, &expected, false);
    const queued = [_][3]i64{
        .{ 8, 11, 7 },  .{ 5, 13, 7 },  .{ -4, 14, 7 }, .{ -4, 19, 7 },
        .{ 6, 23, 7 },  .{ 3, 13, 7 },  .{ 2, 17, 7 },  .{ -2, 11, 7 },
        .{ 1, 12, 7 },  .{ 2, 15, 7 },  .{ -2, 10, 7 }, .{ 8, 19, 7 },
        .{ -5, 20, 7 }, .{ -1, 22, 7 }, .{ -5, 20, 7 }, .{ -2, 17, 7 },
        .{ -1, 21, 7 }, .{ -5, 16, 7 }, .{ -3, 18, 7 }, .{ 0, 15, 7 },
    };
    try expectSequence(19, .{
        .{ .name = "z", .low = -5, .high = 8 },
        .{ .name = "a", .low = 10, .high = 23 },
        .{ .name = "fixed", .low = 7, .high = 7 },
    }, &queued, true);
}

fn expectSequence(seed: u32, initial: [3]Param, expected: []const [3]i64, queued: bool) !void {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const history = try arena_state.allocator().alloc(Trial, expected.len);
    const stored = try arena_state.allocator().alloc([3]?i64, expected.len);
    var sampler: Sampler = .init(seed);
    for (expected, 0..) |want, i| {
        _ = scratch.reset(.retain_capacity);
        var params = initial;
        if (queued and (i == 0 or i == 11)) params[0].fixed = params[0].high;
        var got: [3]i64 = undefined;
        try sampler.sample(scratch.allocator(), &params, history[0..i], &got);
        try testing.expectEqualSlices(i64, &want, &got);
        for (got, &stored[i]) |x, *saved| saved.* = x;
        history[i] = .{ .params = &stored[i], .value = @floatFromInt(@mod(3 * got[0] + 4 * got[1] + 5 * got[2] + 7 * @as(i64, @intCast(i)), 31)) };
    }
}

test "integer TPE retains stable objective ties and dynamic integer bounds" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const tied = [_]i64{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    const dynamic = [_]i64{ -1, -2, 6, -3, -1, 4, -1, 6, 10, 3, 7, 2, 1, 6, 8, 5, 5, 8, 10, 4, 5, 6, 7, 8, 11, 5, 7, 8, 9, 3, 1, 7, 7, 6, 9, 6 };
    var choices: [36][2]?i64 = undefined;
    var history: [36]Trial = undefined;
    var sampler: Sampler = .init(9);
    const params = [_]Param{
        .{ .name = "x", .low = 0, .high = 1 },
        .{ .name = "fixed", .low = -4, .high = -4 },
    };
    for (tied, 0..) |want, i| {
        _ = scratch.reset(.retain_capacity);
        var got: [2]i64 = undefined;
        try sampler.sample(scratch.allocator(), &params, history[0..i], &got);
        try testing.expectEqual(want, got[0]);
        try testing.expectEqual(@as(i64, -4), got[1]);
        choices[i] = .{ got[0], got[1] };
        history[i] = .{ .params = &choices[i], .value = 1 };
    }
    sampler = .init(41);
    for (dynamic, 0..) |want, i| {
        _ = scratch.reset(.retain_capacity);
        const p = [_]Param{.{ .name = "x", .low = -3 + @as(i64, @intCast(i % 3)), .high = 7 + @as(i64, @intCast(i % 5)) }};
        var got: [1]i64 = undefined;
        try sampler.sample(scratch.allocator(), &p, history[0..i], &got);
        try testing.expectEqual(want, got[0]);
        choices[i][0] = got[0];
        history[i] = .{ .params = choices[i][0..1], .value = @floatFromInt(@mod(got[0] * 13 + @as(i64, @intCast(i)) * 7, 19)) };
    }
}

test "integer TPE uses all 300 observations and omits absent and failed parameters" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const params = [_]Param{
        .{ .name = "z", .low = -5, .high = 8 },
        .{ .name = "a", .low = 10, .high = 23 },
        .{ .name = "fixed", .low = 7, .high = 7 },
    };
    var choices: [309][3]?i64 = undefined;
    var history: [309]Trial = undefined;
    for (0..300) |i| {
        choices[i] = .{
            -5 + @as(i64, @intCast((i * 3) % 14)),
            if (i % 7 == 0) null else 10 + @as(i64, @intCast((i * 4) % 14)),
            7,
        };
        history[i] = .{ .params = &choices[i], .value = @floatFromInt((i * 17) % 101) };
    }
    for (300..303) |i| {
        choices[i] = .{ null, null, null };
        history[i] = .{ .params = &choices[i], .value = 0, .state = .fail };
    }
    const expected = [_][3]i64{ .{ 4, 23, 7 }, .{ -4, 13, 7 }, .{ 0, 15, 7 }, .{ -4, 11, 7 }, .{ -5, 19, 7 }, .{ 0, 21, 7 } };
    var sampler: Sampler = .init(19);
    for (expected, 303..) |want, i| {
        _ = scratch.reset(.retain_capacity);
        var got: [3]i64 = undefined;
        try sampler.sample(scratch.allocator(), &params, history[0..i], &got);
        try testing.expectEqualSlices(i64, &want, &got);
        for (got, &choices[i]) |x, *saved| saved.* = x;
        const source_index: i64 = @intCast(i - 3);
        history[i] = .{ .params = &choices[i], .value = @floatFromInt(@mod(3 * got[0] + 4 * got[1] + 5 * got[2] + source_index * 7, 31)) };
    }
}

test "logarithm preserves actual source acquisition cancellation boundary" {
    // Source trial18 above-distribution exp sum. Its scalar math.log result,
    // independently captured from the locked oracle, is one binary64 ulp above
    // the pinned compiler's direct binary64 logarithm.
    try testing.expectEqual(@as(u64, 0x40011edcca43b22d), @as(u64, @bitCast(logarithm(0x1.10003c910e667p+3))));
    try testing.expectEqual(@as(f64, 0), logarithm(1));
    try testing.expect(std.math.isNegativeInf(logarithm(0)));
    try testing.expectEqual(std.math.inf(f64), logarithm(std.math.inf(f64)));
    try testing.expect(std.math.isNan(logarithm(-1)));
}

test "log1p preserves quantile endpoints and tiny-tail IEEE behavior" {
    // Quantile zero retains its sign; quantile one gives log(0). Tiny Gaussian
    // tail probabilities must survive the rounding of 1+x to exactly one.
    try testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(logOnePlus(0.0))));
    for ([_]f64{ 1e-20, -1e-20, 1e-100, -1e-100 }) |x| {
        try testing.expectEqual(x, logOnePlus(x));
    }
    try testing.expectEqual(@as(u64, 0x8000000000000000), @as(u64, @bitCast(logOnePlus(-0.0))));
    try testing.expect(std.math.isNegativeInf(logOnePlus(-1)));
    try testing.expect(std.math.isNan(logOnePlus(-1.5)));
    try testing.expectEqual(std.math.inf(f64), logOnePlus(std.math.inf(f64)));
    try testing.expect(std.math.isNan(logOnePlus(std.math.nan(f64))));
}

test "integer Parzen sigmas density and truncated draws match the source" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Exact scalar math.erf and bounded-rational source outputs. These prior
    // boundaries distinguish the source's tiny acquisition differences from a
    // mathematically symmetric density; the sequence fixture remains unchanged.
    const erf_inputs = [_]f64{ 0x1.6a09e667f3bccp-3, 0x1.6a09e667f3bccp-2, 0x1.6a09e667f3bccp-1 };
    const erf_bits = [_]u64{ 0x3fc944d158b76b61, 0x3fd881d788cab1da, 0x3fe5d897a241a6fa };
    for (erf_inputs, erf_bits) |x, expected| try testing.expectEqual(expected, @as(u64, @bitCast(erf(x))));
    try testing.expectEqual(@as(u64, 0x3fd3bf143b9aa713), @as(u64, @bitCast(normalCdf(-0.5))));
    try testing.expectEqual(@as(u64, 0xbff2d088d9f06be4), @as(u64, @bitCast(logCdf(-0.5))));
    try testing.expectEqual(@as(u64, 0xbffa72f2ba3f0410), @as(u64, @bitCast(logMass(-0.5, 0))));
    try testing.expectEqual(@as(u64, 0xbffa72f2ba3f0413), @as(u64, @bitCast(logMass(0, 0.5))));
    try testing.expectEqual(@as(u64, 0xbfeeb7a27583ce35), @as(u64, @bitCast(logMass(-0.5, 0.5))));
    const tied_estimator = try Estimator.init(arena, .{ .name = "x", .low = 0, .high = 1 }, &.{ 0, 1 });
    var tied_terms: [3]f64 = undefined;
    try testing.expectEqual(@as(u64, 0xbfe62e42fefa39ee), @as(u64, @bitCast(tied_estimator.logPdf(0, &tied_terms))));
    try testing.expectEqual(@as(u64, 0xbfe62e42fefa39f1), @as(u64, @bitCast(tied_estimator.logPdf(1, &tied_terms))));
    const estimator = try Estimator.init(arena, .{ .name = "x", .low = 0, .high = 5 }, &.{ 0, 1, 1, 4, 5 });
    try testing.expectEqualSlices(f64, &.{ 0, 1, 1, 4, 5, 2.5 }, estimator.mus);
    try testing.expectEqualSlices(f64, &.{ 1, 1, 1.5, 1.5, 1, 6 }, estimator.sigmas);
    const expected_pdf = [_]f64{ 0.2054826298256471, 0.2140737177586488, 0.15184589517585662, 0.11949576323252963, 0.14675121318920925, 0.16235078081810864 };
    var terms: [6]f64 = undefined;
    for (expected_pdf, 0..) |expected, x| try testing.expectApproxEqAbs(expected, @exp(estimator.logPdf(@intCast(x), &terms)), 2e-13);
    var rng: pyrandom.Random = .initNumpy(19);
    const samples = try estimator.sample(arena, &rng);
    try testing.expectEqualSlices(i64, &.{ 1, 5, 1, 1, 1, 1, 3, 5, 3, 4, 1, 2, 4, 0, 1, 2, 0, 2, 0, 5, 2, 1, 4, 5 }, &samples);

    const aa = [_]f64{ -40, -20, -9, -3, -1, 0, 1, 8 };
    const bb = [_]f64{ -39.9, -19.5, -8, -2.7, 1, 1, 3, 8.1 };
    const qqs = [_]f64{ 0, 1e-12, 0.01, 0.25, 0.5, 0.75, 1 - 1e-12, 1 };
    const masses = [_]f64{ -800.6294741601215, -194.01701594392597, -35.013618593437144, -6.1577194955896335, -0.38171514630212616, -1.0748623268620714, -1.8495664205476083, -35.59634237383949 };
    const ppfs = [_]f64{ -40, -19.99999999900582, -8.546703681127418, -2.8977639038444534, -0.0, 0.693520634699274, 2.999999999964506, 8.1 };
    var ys: [8]f64 = undefined;
    var left: [8]bool = undefined;
    var xs: [8]f64 = undefined;
    for (aa, bb, qqs, masses, &ys, &left) |a, b, q, expected, *y, *l| {
        const mass = logMass(a, b);
        try testing.expectApproxEqAbs(expected, mass, 2e-11);
        l.* = a < 0;
        y.* = if (l.*) logSum(logCdf(a), logarithm(q) + mass) else logSum(logCdf(-b), logOnePlus(-q) + mass);
    }
    inverseLogCdfBatch(&ys, &left, true, &xs);
    inverseLogCdfBatch(&ys, &left, false, &xs);
    for (aa, bb, qqs, ppfs, left, xs) |a, b, q, expected, l, x| {
        const result = if (q == 0) a else if (q == 1) b else if (l) x else -x;
        try testing.expectApproxEqAbs(expected, result, 2e-9);
    }
}

test "unseeded streams request entropy lazily and fixed parameters never draw" {
    const Entropy = struct {
        calls: u32 = 0,
        fn fill(ptr: ?*anyopaque, bytes: []u8) std.Io.RandomSecureError!void {
            const self: *@This() = @ptrCast(@alignCast(ptr.?));
            self.calls += 1;
            assert(bytes.len == 4);
            std.mem.writeInt(u32, bytes[0..4], self.calls * 19, .little);
        }
    };
    var entropy: Entropy = .{};
    var vtable = testing.io.vtable.*;
    vtable.randomSecure = Entropy.fill;
    const io: std.Io = .{ .userdata = &entropy, .vtable = &vtable };
    var sampler: Sampler = .initUnseeded(io);
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    var out: [1]i64 = undefined;
    try sampler.sample(scratch.allocator(), &.{.{ .name = "x", .low = 3, .high = 3 }}, &.{}, &out);
    try testing.expectEqual(@as(u32, 0), entropy.calls);
    // Optuna accepts out-of-range fixed values (with a warning).
    try sampler.sample(scratch.allocator(), &.{.{ .name = "x", .low = 0, .high = 5, .fixed = 99 }}, &.{}, &out);
    try testing.expectEqual(@as(i64, 99), out[0]);
    try testing.expectEqual(@as(u32, 0), entropy.calls);
    const params = [_]Param{.{ .name = "x", .low = 0, .high = 5 }};
    const failed: [3]Trial = @splat(.{ .params = &.{null}, .value = 0, .state = .fail });
    try sampler.sample(scratch.allocator(), &params, &failed, &out);
    var seeded: Sampler = .init(19);
    var expected: [1]i64 = undefined;
    try seeded.sample(scratch.allocator(), &params, &failed, &expected);
    try testing.expectEqualSlices(i64, &expected, &out);
    try testing.expectEqual(@as(u32, 1), entropy.calls);
    const complete: [10]Trial = @splat(.{ .params = &.{2}, .value = 1 });
    try sampler.sample(scratch.allocator(), &params, &complete, &out);
    try testing.expectEqual(@as(u32, 2), entropy.calls);
    try testing.expectError(error.InvalidBounds, sampler.sample(scratch.allocator(), &.{.{ .name = "x", .low = 0, .high = -1 }}, &.{}, &out));
}

test "integer suggestions preserve fixed out-of-bounds values and repeated parameter names" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const params = [_]Param{
        .{ .name = "x", .low = 0, .high = 5, .fixed = 99 },
        .{ .name = "x", .low = 1, .high = 3 },
        .{ .name = "y", .low = 0, .high = 3 },
    };
    var sampler: Sampler = .init(5);
    var choices: [3]i64 = undefined;
    try sampler.sample(scratch.allocator(), &params, &.{}, &choices);
    try testing.expectEqualSlices(i64, &.{ 99, 99, 0 }, &choices);
}
