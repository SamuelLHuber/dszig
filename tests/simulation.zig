//! Seeded campaigns execute real optimize.evaluate through the shared native harness.
//! The oracle checks outcomes, call counts, rounded scores, input roles and scratch reuse.

const std = @import("std");
const simulation = @import("evaluation_simulation");
const testing = std.testing;

fn campaign(seed: u64) !void {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = undefined;
    var case_index: u32 = 0;
    while (case_index < 512) : (case_index += 1) {
        simulation.generate(&scenario, seed, case_index);
        try testing.expect(simulation.valid_case(&scenario));
        if (harness.check(&scenario, .correct)) |violation| {
            std.debug.print("simulation seed={d} index={d} violation={s}\n", .{
                seed, case_index, @tagName(violation),
            });
            return error.SimulationViolation;
        }
    }
}

test "native evaluation campaign seed zero: 512 cases" {
    try campaign(0);
}

test "native evaluation campaign seed one: 512 cases" {
    try campaign(1);
}

test "native evaluation campaign mixed seed: 512 cases" {
    try campaign(0x9e37_79b9_7f4a_7c15);
}

test "native evaluation campaign maximum seed: 512 cases" {
    try campaign(std.math.maxInt(u64));
}

test "native evaluation rounds all 32-row success fractions including both half-cent ties" {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = .{ .len = 32, .max_errors = 33 };
    var successes: u8 = 0;
    while (successes <= 32) : (successes += 1) {
        for (scenario.rows[0..scenario.len], 0..) |*row, index| {
            row.* = .{ .value = @as(i64, @intCast(index)) - 16, .score = index < successes };
        }
        // 1/32 is 3.125% -> 3.12; 3/32 is 9.375% -> 9.38 (ties to even).
        try testing.expect(simulation.valid_case(&scenario));
        try testing.expectEqual(
            @as(?simulation.Violation, null),
            harness.check(&scenario, .correct),
        );
    }
}

test "native scenario validation accepts execution limits and rejects adjacent invalid inputs" {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = .{ .len = 1, .max_errors = 0 };
    scenario.rows[0].value = -1000;
    try testing.expect(simulation.valid_case(&scenario));
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));

    scenario.len = 32;
    scenario.max_errors = 33;
    scenario.rows[31].value = 1000;
    try testing.expect(simulation.valid_case(&scenario));
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));

    scenario.len = 0;
    try testing.expect(!simulation.valid_case(&scenario));
    scenario.len = 33;
    try testing.expect(!simulation.valid_case(&scenario));
    scenario.len = 32;
    scenario.max_errors = 34;
    try testing.expect(!simulation.valid_case(&scenario));
    scenario.max_errors = 33;
    scenario.rows[0].value = -1001;
    try testing.expect(!simulation.valid_case(&scenario));
    scenario.rows[0].value = -1000;
    scenario.rows[31].value = 1001;
    try testing.expect(!simulation.valid_case(&scenario));
}

test "native evaluation program failure bypasses a fatal metric on that row" {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = .{ .len = 3, .max_errors = 2 };
    scenario.rows[0] = .{
        .value = -1000,
        .program_fault = .failure,
        .metric_fault = .out_of_memory,
    };
    scenario.rows[1] = .{ .value = 1000 };
    scenario.rows[2] = .{ .value = 17, .score = false };
    try testing.expect(simulation.valid_case(&scenario));
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));
}

test "native evaluation fatal program and metric OOM ignore remaining recoverable budget" {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = .{ .len = 4, .max_errors = 33 };
    scenario.rows[0] = .{ .value = -3, .metric_fault = .failure };
    scenario.rows[1] = .{ .value = 7, .program_fault = .out_of_memory };
    scenario.rows[2] = .{ .value = 11, .metric_fault = .out_of_memory };
    try testing.expect(simulation.valid_case(&scenario));
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));

    // The second run reaches the metric OOM instead; neither run may execute later rows.
    scenario.rows[1].program_fault = .none;
    scenario.rows[1].metric_fault = .out_of_memory;
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));
}

test "native evaluation recoverable stop boundary includes zero and budgets beyond row count" {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = .{ .len = 4 };
    scenario.rows[0] = .{ .value = -1, .program_fault = .failure };
    scenario.rows[1] = .{ .value = 2 };
    scenario.rows[2] = .{ .value = -3, .metric_fault = .failure };
    scenario.rows[3] = .{ .value = 4, .program_fault = .out_of_memory };
    for ([_]u8{ 0, 1, 2, 3, 32, 33 }) |budget| {
        scenario.max_errors = budget;
        try testing.expect(simulation.valid_case(&scenario));
        try testing.expectEqual(
            @as(?simulation.Violation, null),
            harness.check(&scenario, .correct),
        );
    }

    // Reuse after both early-stop and fatal paths must also reclaim scratch fully.
    scenario = .{ .len = 32, .max_errors = 33 };
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));
}

test "native wrong-answer negative control shrinks to a one-row score counterexample" {
    const harness = try testing.allocator.create(simulation.Harness);
    defer testing.allocator.destroy(harness);
    try harness.init();

    var scenario: simulation.Case = .{ .len = 32, .max_errors = 33 };
    for (&scenario.rows, 0..) |*row, index| {
        // Removing the only scoring row must be rejected and restored even
        // while a nonempty suffix is shifted; surrounding rows may disappear.
        row.* = .{
            .value = @as(i64, @intCast(index)) * 31 - 480,
            .score = index == 16,
        };
    }
    try testing.expect(simulation.valid_case(&scenario));
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));
    try testing.expectEqual(
        @as(?simulation.Violation, .score),
        harness.check(&scenario, .wrong_answer),
    );

    const attempts = harness.shrink(&scenario, .score, .wrong_answer);
    try testing.expect(attempts <= 4096);
    try testing.expect(simulation.valid_case(&scenario));
    try testing.expectEqual(@as(u8, 1), scenario.len);
    try testing.expectEqual(
        @as(?simulation.Violation, .score),
        harness.check(&scenario, .wrong_answer),
    );
    try testing.expectEqual(@as(?simulation.Violation, null), harness.check(&scenario, .correct));
}
