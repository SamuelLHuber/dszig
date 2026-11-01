//! DSZig's TPE sampler must suggest exactly what Optuna 4.5.0's
//! TPESampler(seed, multivariate=True) suggested in
//! tools/dspy_reference/gen_optuna.py, trial by trial, including the
//! RandomSampler startup phase and the Parzen-estimator phase.

const std = @import("std");
const dszig = @import("dszig");
const testing = std.testing;
const tpe = dszig.tpe;

const fixture = @embedFile("fixtures/optuna_tpe.json");

const Scenario = struct {
    seed: u32,
    params: []const struct { []const u8, usize },
    default: struct { params: []const usize, value: f64 },
    trials: []const struct { params: []const usize, value: f64 },
};

test "TPE suggestions match Optuna 4.5.0 trial by trial" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const data = try std.json.parseFromSliceLeaky(struct { optuna: []const u8, scenarios: []const Scenario }, arena, fixture, .{});
    try testing.expectEqualStrings("4.5.0", data.optuna);
    var checked: usize = 0;
    for (data.scenarios) |scenario| {
        const params = try arena.alloc(tpe.Param, scenario.params.len);
        for (params, scenario.params) |*p, sp| p.* = .{ .name = sp[0], .choices = sp[1] };
        var history: std.ArrayList(tpe.Trial) = .empty;
        try history.append(arena, .{ .params = scenario.default.params, .value = scenario.default.value });
        var sampler: tpe.Sampler = .init(scenario.seed);
        for (scenario.trials) |want| {
            const got = try arena.alloc(usize, params.len);
            try sampler.sample(arena, params, history.items, got);
            if (!std.mem.eql(usize, want.params, got)) {
                std.debug.print("seed {d} trial {d}: want {any} got {any}\n", .{ scenario.seed, history.items.len - 1, want.params, got });
            }
            try testing.expectEqualSlices(usize, want.params, got);
            try history.append(arena, .{ .params = got, .value = want.value });
            checked += 1;
        }
    }
    try testing.expectEqual(@as(usize, 142), checked);
}
