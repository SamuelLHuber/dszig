//! Port of Optuna 4.5.0 `TPESampler(seed=seed, multivariate=True)` for the
//! categorical search spaces DSPy's MIPROv2 builds (one instruction index and
//! one demo-set index per predictor). The first `n_startup_trials` trials use
//! Optuna's `RandomSampler`; later ones fit the Parzen estimators. NumPy's
//! `RandomState` stream, cumulative sums and pairwise sums are reproduced
//! exactly; `log`/`exp` come from Zig's libm, so only an exact tie broken by a
//! last-ulp difference could diverge from NumPy.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const pyrandom = @import("pyrandom.zig");

pub const n_startup_trials = 10;
pub const n_ei_candidates = 24;
const prior_weight = 1.0;

/// `trial.suggest_categorical(name, range(choices))`.
pub const Param = struct { name: []const u8, choices: usize };

/// A completed trial: one choice index per `Param`, in the same order.
pub const Trial = struct { params: []const usize, value: f64 };

pub const Sampler = struct {
    /// `TPESampler._rng`, used once the startup trials are done.
    rng: pyrandom.Random,
    /// `TPESampler._random_sampler._rng`: a separate stream, same seed.
    startup_rng: pyrandom.Random,

    pub fn init(seed: u32) Sampler {
        return .{ .rng = .initNumpy(seed), .startup_rng = .initNumpy(seed) };
    }

    /// Writes the next trial's choices into `out` (ordered like `params`),
    /// given the completed `trials` (a study maximizing `value`).
    pub fn sample(s: *Sampler, scratch: Allocator, params: []const Param, trials: []const Trial, out: []usize) Allocator.Error!void {
        assert(out.len == params.len);
        for (params) |p| assert(p.choices > 0);
        for (trials) |t| assert(t.params.len == params.len);
        if (trials.len < n_startup_trials) {
            for (params, out) |p, *o| o.* = s.randomChoice(p.choices);
            return;
        }
        try s.sampleTpe(scratch, params, trials, out);
    }

    /// `RandomSampler`: a uniform draw per one-hot slot, then the argmax.
    /// Single-choice distributions are fixed and draw nothing.
    fn randomChoice(s: *Sampler, choices: usize) usize {
        if (choices == 1) return 0;
        var best: usize = 0;
        var best_value: f64 = -1;
        for (0..choices) |i| {
            const u = s.startup_rng.random();
            if (u > best_value) {
                best = i;
                best_value = u;
            }
        }
        return best;
    }

    fn sampleTpe(s: *Sampler, scratch: Allocator, params: []const Param, trials: []const Trial, out: []usize) Allocator.Error!void {
        // The relative search space is sorted by name and skips fixed params.
        var order: std.ArrayList(usize) = .empty;
        for (params, 0..) |p, i| if (p.choices > 1) try order.append(scratch, i);
        std.sort.insertion(usize, order.items, params, nameLess);
        @memset(out, 0);
        if (order.items.len == 0) return;

        const n = trials.len;
        const n_below: usize = @min(@as(usize, @intFromFloat(@ceil(0.1 * @as(f64, @floatFromInt(n))))), 25);
        const ranked = try scratch.alloc(usize, n);
        for (ranked, 0..) |*r, i| r.* = i;
        std.sort.insertion(usize, ranked, trials, valueGreater);
        const is_below = try scratch.alloc(bool, n);
        @memset(is_below, false);
        for (ranked[0..@min(n_below, n)]) |i| is_below[i] = true;
        var below: std.ArrayList(usize) = .empty;
        var above: std.ArrayList(usize) = .empty;
        for (is_below, 0..) |b, i| try (if (b) &below else &above).append(scratch, i);

        const mpe_below = try Estimator.init(scratch, params, order.items, trials, below.items);
        const mpe_above = try Estimator.init(scratch, params, order.items, trials, above.items);
        const samples = try mpe_below.sample(scratch, &s.rng, params, order.items);
        // Reuse one workspace for both mixtures and every candidate.
        const terms = try scratch.alloc(f64, @max(mpe_below.weights.len, mpe_above.weights.len));

        var best: usize = 0;
        var best_acq: f64 = -std.math.inf(f64);
        for (0..n_ei_candidates) |c| {
            const x = samples[c * order.items.len ..][0..order.items.len];
            const acq = mpe_below.logPdf(params, order.items, x, terms[0..mpe_below.weights.len]) - mpe_above.logPdf(params, order.items, x, terms[0..mpe_above.weights.len]);
            if (c == 0 or acq > best_acq) {
                best = c;
                best_acq = acq;
            }
        }
        for (order.items, samples[best * order.items.len ..][0..order.items.len]) |i, v| out[i] = v;
    }

    fn nameLess(params: []const Param, a: usize, b: usize) bool {
        return std.mem.lessThan(u8, params[a].name, params[b].name);
    }

    fn valueGreater(trials: []const Trial, a: usize, b: usize) bool {
        return trials[a].value > trials[b].value;
    }
};

/// `_ParzenEstimator` over categorical params: a mixture with one kernel per
/// observation plus the prior.
const Estimator = struct {
    weights: []f64,
    /// Per relative param: `weights.len` rows of `choices` probabilities.
    kernels: []const []f64,

    fn init(arena: Allocator, params: []const Param, order: []const usize, trials: []const Trial, members: []const usize) Allocator.Error!Estimator {
        const m = members.len;
        const n_kernels = m + 1;
        const weights = try arena.alloc(f64, if (m == 0) 1 else n_kernels);
        if (m == 0) {
            weights[0] = 1.0;
        } else {
            defaultWeights(weights[0..m]);
            weights[m] = prior_weight;
            const total = pairwiseSum(weights);
            for (weights) |*w| w.* /= total;
        }
        const kernels = try arena.alloc([]f64, order.len);
        for (kernels, order) |*rows, pi| {
            const k = params[pi].choices;
            if (m == 0) {
                rows.* = try arena.alloc(f64, k);
                @memset(rows.*, 1.0 / @as(f64, @floatFromInt(k)));
                continue;
            }
            rows.* = try arena.alloc(f64, n_kernels * k);
            @memset(rows.*, prior_weight / @as(f64, @floatFromInt(n_kernels)));
            for (members, 0..) |t, j| rows.*[j * k + trials[t].params[pi]] += 1;
            for (0..n_kernels) |j| {
                const row = rows.*[j * k ..][0..k];
                const sum = pairwiseSum(row);
                const div = if (sum == 0) 1 else sum;
                for (row) |*v| v.* /= div;
            }
        }
        return .{ .weights = weights, .kernels = kernels };
    }

    /// `default_weights(n)`: flat below 25 observations, else a linear ramp
    /// (`np.linspace(1/n, 1, n - 25)`) followed by 25 ones.
    fn defaultWeights(out: []f64) void {
        const n = out.len;
        if (n < 25) {
            @memset(out, 1.0);
            return;
        }
        const num = n - 25;
        const start = 1.0 / @as(f64, @floatFromInt(n));
        if (num > 0) {
            const step = (1.0 - start) / @as(f64, @floatFromInt(@max(num - 1, 1)));
            for (out[0..num], 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) * step + start;
            if (num > 1) out[num - 1] = 1.0;
        }
        @memset(out[num..], 1.0);
    }

    /// `_MixtureOfProductDistribution.sample(rng, n_ei_candidates)`: returns
    /// `n_ei_candidates` rows of `order.len` choice indices.
    fn sample(e: Estimator, arena: Allocator, rng: *pyrandom.Random, params: []const Param, order: []const usize) Allocator.Error![]usize {
        const cdf = try arena.alloc(f64, e.weights.len);
        cumsum(e.weights, cdf);
        const last = cdf[cdf.len - 1];
        for (cdf) |*c| c.* /= last;
        var active: [n_ei_candidates]usize = undefined;
        for (&active) |*a| {
            const u = rng.random();
            var idx: usize = 0;
            while (idx < cdf.len and cdf[idx] <= u) idx += 1;
            a.* = @min(idx, e.weights.len - 1);
        }
        const out = try arena.alloc(usize, n_ei_candidates * order.len);
        for (order, e.kernels, 0..) |pi, rows, r| {
            const k = params[pi].choices;
            const cum = try arena.alloc(f64, k);
            for (active, 0..) |a, c| {
                const rnd = rng.random();
                const row = if (rows.len == k) rows else rows[a * k ..][0..k];
                cumsum(row, cum);
                cum[k - 1] = 1;
                var v: usize = 0;
                for (cum) |p| {
                    if (p < rnd) v += 1;
                }
                out[c * order.len + r] = v;
            }
        }
        return out;
    }

    fn logPdf(e: Estimator, params: []const Param, order: []const usize, x: []const usize, terms: []f64) f64 {
        assert(terms.len == e.weights.len);
        var max: f64 = -std.math.inf(f64);
        for (terms, e.weights, 0..) |*t, w, j| {
            var v: f64 = 0;
            for (order, e.kernels, x) |pi, rows, xi| {
                const k = params[pi].choices;
                const row = if (rows.len == k) rows else rows[j * k ..][0..k];
                v += @log(row[xi]);
            }
            v += @log(w);
            t.* = v;
            max = @max(max, v);
        }
        if (std.math.isNegativeInf(max)) max = 0;
        for (terms) |*t| t.* = @exp(t.* - max);
        return @log(pairwiseSum(terms)) + max;
    }
};

fn cumsum(xs: []const f64, out: []f64) void {
    assert(out.len == xs.len);
    var acc: f64 = 0;
    for (xs, out) |x, *o| {
        acc += x;
        o.* = acc;
    }
}

/// NumPy's pairwise summation (`np.sum` along a contiguous axis).
pub fn pairwiseSum(xs: []const f64) f64 {
    const n = xs.len;
    if (n < 8) {
        var res: f64 = 0;
        for (xs) |x| res += x;
        return res;
    }
    if (n <= 128) {
        var r: [8]f64 = xs[0..8].*;
        var i: usize = 8;
        while (i < n - n % 8) : (i += 8) {
            for (&r, xs[i..][0..8]) |*acc, x| acc.* += x;
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < n) : (i += 1) res += xs[i];
        return res;
    }
    var n2 = n / 2;
    n2 -= n2 % 8;
    return pairwiseSum(xs[0..n2]) + pairwiseSum(xs[n2..]);
}

test "TPE retains all kernels beyond 300 completed observations" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const params = [_]Param{.{ .name = "choice", .choices = 2 }};
    var choices: [300][1]usize = @splat(.{0});
    choices[299][0] = 1;
    var trials: [300]Trial = undefined;
    var members: [300]usize = undefined;
    for (&trials, &members, &choices, 0..) |*trial, *member, *choice, i| {
        trial.* = .{ .params = choice, .value = 0 };
        member.* = i;
    }
    const estimator = try Estimator.init(arena, &params, &.{0}, &trials, &members);
    try std.testing.expectEqual(@as(usize, 301), estimator.weights.len);
    const terms = try arena.alloc(f64, estimator.weights.len);
    // Optuna 4.5.0's _ParzenEstimator: the final observation is the only
    // vote for choice 1, so truncating the long history changes its density.
    try std.testing.expectApproxEqAbs(@as(f64, 0.9876113913805615), @exp(estimator.logPdf(&params, &.{0}, &.{0}, terms)), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01238860861943882), @exp(estimator.logPdf(&params, &.{0}, &.{1}, terms)), 1e-12);
}

test "TPE suggestions match Optuna after 300 completed trials" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const params = [_]Param{
        .{ .name = "z_instruction", .choices = 3 },
        .{ .name = "a_demos", .choices = 4 },
        .{ .name = "fixed", .choices = 1 },
    };
    var choices: [304][3]usize = undefined;
    var trials: [304]Trial = undefined;
    for (0..300) |i| {
        choices[i] = .{ i % 3, (i / 3) % 4, 0 };
        trials[i] = .{ .params = &choices[i], .value = @floatFromInt((i * 17) % 101) };
    }
    // Independently probed with Optuna 4.5.0 TPESampler(seed=19,
    // multivariate=True), maximizing the same completed-trial history.
    const expected = [_][3]usize{ .{ 0, 1, 0 }, .{ 0, 3, 0 }, .{ 2, 3, 0 }, .{ 0, 1, 0 } };
    var sampler: Sampler = .init(19);
    for (expected, 300..) |want, i| {
        _ = scratch.reset(.retain_capacity);
        try sampler.sample(scratch.allocator(), &params, trials[0..i], &choices[i]);
        try std.testing.expectEqualSlices(usize, &want, &choices[i]);
        trials[i] = .{ .params = &choices[i], .value = @floatFromInt((i * 17) % 101) };
    }
}
