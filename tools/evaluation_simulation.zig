//! Seeded sequential failure simulation against the real native evaluator.
//! The harness provisions its backing at startup. The current evaluator still
//! performs allocator operations within that backing: this is not a claim of
//! allocator-free execution or a concurrent scheduler simulation.
const std = @import("std");
const dszig = @import("dszig");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Record = dszig.value.Record;

pub const rows_max = 32;
pub const replay_version: u32 = 1;
pub const Fault = enum { none, failure, out_of_memory };
pub const Row = struct {
    value: i64 = 0,
    program_fault: Fault = .none,
    metric_fault: Fault = .none,
    score: bool = true,
};
pub const Case = struct {
    len: u8 = 1,
    max_errors: u8 = 1,
    rows: [rows_max]Row = @splat(.{}),
};
pub const Replay = struct {
    version: u32 = replay_version,
    seed: u64 = 0,
    case_index: u32 = 0,
    scenario: Case,
};
pub const Mode = enum { correct, wrong_answer };
pub const Violation = enum {
    outcome,
    program_calls,
    score,
    dataset_mutation,
    scratch_retention,
    unexpected_lm_call,
};

pub fn valid_case(scenario: *const Case) bool {
    if (scenario.len == 0 or scenario.len > rows_max) return false;
    if (scenario.max_errors > rows_max + 1) return false;
    for (scenario.rows[0..scenario.len]) |row| {
        if (row.value < -1000 or row.value > 1000) return false;
    }
    return true;
}

// Version 1 uses SplitMix64 with an independent stream for each case index.
// Random access avoids replay work proportional to the original case index.
const Random = struct {
    state: u64,

    fn next(self: *Random) u64 {
        self.state +%= 0x9e3779b97f4a7c15;
        var value = self.state;
        value = (value ^ (value >> 30)) *% 0xbf58476d1ce4e5b9;
        value = (value ^ (value >> 27)) *% 0x94d049bb133111eb;
        return value ^ (value >> 31);
    }
};

pub fn generate(scenario: *Case, seed: u64, case_index: u32) void {
    var random: Random = .{ .state = seed ^ (@as(u64, case_index) *% 0xd1342543de82ef95) };
    scenario.* = .{};
    scenario.len = @intCast(1 + random.next() % rows_max);
    scenario.max_errors = @intCast(random.next() % (rows_max + 2));
    // Alternate clean, recoverable-only, and fatal-capable cases. Uniform
    // faults alone would almost always stop before exercising a long prefix.
    const fault_kinds: u64 = 1 + random.next() % 3;
    for (scenario.rows[0..scenario.len]) |*row| {
        row.* = .{
            .value = @as(i64, @intCast(random.next() % 2001)) - 1000,
            .program_fault = @fromBackingInt(@intCast(random.next() % fault_kinds)),
            .metric_fault = @fromBackingInt(@intCast(random.next() % fault_kinds)),
            .score = random.next() & 1 != 0,
        };
    }
    assert(valid_case(scenario));
    assert(scenario.len <= rows_max);
}

const Program = struct {
    calls: *u32,
    mode: Mode,
    pub const Input = struct { value: i64, program_fault: u8 };
    pub const Output = struct { answer: i64 };

    pub fn forward(self: *const Program, _: dszig.Context, input: Input) !Output {
        assert(input.program_fault <= @backingInt(Fault.out_of_memory));
        assert(self.calls.* < rows_max);
        self.calls.* += 1;
        switch (@as(Fault, @fromBackingInt(@intCast(input.program_fault)))) {
            .none => {},
            .failure => return error.LMFailure,
            .out_of_memory => return error.OutOfMemory,
        }
        const answer = 2 * input.value + @as(i64, @intFromBool(self.mode == .wrong_answer));
        return .{ .answer = answer };
    }

    fn metric(
        _: Allocator,
        example: *const dszig.Example,
        prediction: ?*const dszig.optimize.RecordPrediction,
        trace: ?*const dszig.Trace,
    ) dszig.value.ParseError!f64 {
        assert(trace == null);
        const output = prediction orelse return error.ParseValue;
        const fault = example.record.get("metric_fault").?.integer;
        assert(fault >= 0);
        assert(fault <= @backingInt(Fault.out_of_memory));
        // Match the validated wire integer without narrowing it to an enum tag.
        switch (fault) {
            @backingInt(Fault.none) => {},
            @backingInt(Fault.failure) => return error.ParseValue,
            @backingInt(Fault.out_of_memory) => return error.OutOfMemory,
            else => unreachable,
        }
        const answer = output.get("answer") orelse return error.ParseValue;
        if (answer != .integer) return error.ParseValue;
        const expected = 2 * example.record.get("value").?.integer;
        return @floatFromInt(@intFromBool(
            example.record.get("score").?.bool and answer.integer == expected,
        ));
    }
};

const Expected = struct {
    failure: ?anyerror = null,
    calls: u32,
    successes: u32 = 0,
    errors: u32 = 0,
    cents: u32 = 0,
};

fn classify(row: *const Row) ?anyerror {
    // The program must finish successfully before the metric can run.
    return switch (row.program_fault) {
        .out_of_memory => error.OutOfMemory,
        .failure => error.LMFailure,
        .none => switch (row.metric_fault) {
            .none => null,
            .failure => error.ParseValue,
            .out_of_memory => error.OutOfMemory,
        },
    };
}

fn expected_result(scenario: *const Case) Expected {
    assert(valid_case(scenario));
    var recoverable: [rows_max]u8 = undefined;
    var recoverable_count: u8 = 0;
    var fatal_index: u8 = scenario.len;
    var expected: Expected = .{ .calls = scenario.len };
    // Derive stop positions first, independently of evaluate's running budget.
    for (scenario.rows[0..scenario.len], 0..) |*row, index| {
        if (classify(row)) |failure| {
            if (failure == error.OutOfMemory) {
                fatal_index = @min(fatal_index, @as(u8, @intCast(index)));
            } else {
                recoverable[recoverable_count] = @intCast(index);
                recoverable_count += 1;
            }
        } else {
            expected.successes += @intFromBool(row.score);
        }
    }
    const quota = @max(@as(u8, 1), scenario.max_errors);
    const budget_index = if (quota <= recoverable_count)
        recoverable[quota - 1]
    else
        scenario.len;
    const stop_index = @min(fatal_index, budget_index);
    if (stop_index < scenario.len) {
        expected.failure = classify(&scenario.rows[stop_index]).?;
        expected.calls = @as(u32, stop_index) + 1;
        return expected;
    }
    expected.errors = recoverable_count;
    // All scores are 0 or 1, so integer rational cents are independent of
    // production binary64 rounding. For n<=32, half-cent ties occur at n=32
    // and are exact binary fractions (e.g. 100/32 = 3.125 -> 3.12).
    const numerator = expected.successes * 10000;
    expected.cents = numerator / scenario.len;
    const remainder_twice = 2 * (numerator % scenario.len);
    if (remainder_twice > scenario.len or
        (remainder_twice == scenario.len and expected.cents & 1 != 0))
    {
        expected.cents += 1;
    }
    assert(expected.cents <= 10000);
    return expected;
}

const keys = [_][]const u8{ "value", "program_fault", "metric_fault", "score" };
const input_keys = [_][]const u8{ "value", "program_fault" };

/// Initialize in place and never move: the allocators borrow these buffers.
pub const Harness = struct {
    setup_buffer: [64 * 1024]u8 align(16),
    execution_buffer: [64 * 1024]u8 align(16),
    setup: std.heap.FixedBufferAllocator,
    execution: std.heap.FixedBufferAllocator,
    examples: [rows_max]dszig.Example,
    original_records: [rows_max]Record,
    setup_end: usize,
    program_calls: u32,
    lm_calls: u32,

    pub fn init(self: *Harness) !void {
        self.setup = .init(&self.setup_buffer);
        self.execution = .init(&self.execution_buffer);
        self.program_calls = 0;
        self.lm_calls = 0;
        for (&self.examples, &self.original_records) |*example, *original| {
            example.* = .{ .record = .empty, .input_keys = &input_keys };
            for (keys) |key| try example.record.put(self.setup.allocator(), key, .null);
            original.* = example.record;
        }
        self.setup_end = self.setup.end_index;
        assert(self.setup_end < self.setup_buffer.len);
        assert(self.execution.end_index == 0);
    }

    pub fn check(self: *Harness, scenario: *const Case, mode: Mode) ?Violation {
        assert(valid_case(scenario));
        assert(self.setup.end_index == self.setup_end);
        // Do not reset leaked scratch before observing it: a previous failed
        // check remains a failure rather than being hidden by the next run.
        if (self.execution.end_index != 0) return .scratch_retention;
        // Reclaim only after checking the result below. Each shrink attempt
        // then reproduces the property from a fresh execution frame.
        defer self.execution.reset();
        self.configure(scenario);
        const expected = expected_result(scenario);
        const program: Program = .{ .calls = &self.program_calls, .mode = mode };
        const context: dszig.Context = .{
            .arena = self.execution.allocator(),
            .lm = .{ .ptr = &self.lm_calls, .vtable = &.{ .complete = unexpected_complete } },
        };
        const actual = dszig.optimize.evaluate(
            Program,
            self.execution.allocator(),
            context,
            &program,
            self.examples[0..scenario.len],
            Program.metric,
            scenario.max_errors,
        );
        // Consume the result even when a storage invariant fails.
        var evaluation: ?dszig.optimize.Evaluation = null;
        var failure: ?anyerror = null;
        if (actual) |result| evaluation = result else |err| failure = err;
        if (self.execution.end_index != 0) return .scratch_retention;
        if (!self.dataset_unchanged(scenario)) return .dataset_mutation;
        if (self.lm_calls != 0) return .unexpected_lm_call;
        if (self.program_calls != expected.calls) return .program_calls;
        if (failure != expected.failure) return .outcome;
        if (evaluation) |result| {
            const score = @as(f64, @floatFromInt(expected.cents)) / 100.0;
            if (result.total != scenario.len or result.errors != expected.errors) return .outcome;
            if (result.sum != @as(f64, @floatFromInt(expected.successes))) return .score;
            if (@as(u64, @bitCast(result.score)) != @as(u64, @bitCast(score))) return .score;
        }
        return null;
    }

    fn configure(self: *Harness, scenario: *const Case) void {
        assert(valid_case(scenario));
        assert(self.execution.end_index == 0);
        self.program_calls = 0;
        self.lm_calls = 0;
        for (scenario.rows[0..scenario.len], 0..) |row, index| {
            const example = &self.examples[index];
            // Restore caller metadata as well as values after an observed
            // mutation, so shrinking cannot mistake contamination for replay.
            example.* = .{ .record = self.original_records[index], .input_keys = &input_keys };
            example.record.clearRetainingCapacity();
            example.record.putAssumeCapacity("value", .{ .integer = row.value });
            example.record.putAssumeCapacity(
                "program_fault",
                .{ .integer = @backingInt(row.program_fault) },
            );
            example.record.putAssumeCapacity(
                "metric_fault",
                .{ .integer = @backingInt(row.metric_fault) },
            );
            example.record.putAssumeCapacity("score", .{ .bool = row.score });
        }
        assert(self.setup.end_index == self.setup_end);
    }

    fn dataset_unchanged(self: *const Harness, scenario: *const Case) bool {
        assert(valid_case(scenario));
        assert(self.setup.end_index == self.setup_end);
        for (self.examples[0..scenario.len], scenario.rows[0..scenario.len]) |*example, row| {
            if (example.record.count() != keys.len) return false;
            for (example.record.keys(), keys) |actual, expected| {
                if (!std.mem.eql(u8, actual, expected)) return false;
            }
            if (example.input_keys.len != input_keys.len) return false;
            for (example.input_keys, input_keys) |actual, expected| {
                if (!std.mem.eql(u8, actual, expected)) return false;
            }
            const expected_values = [_]dszig.Value{
                .{ .integer = row.value },
                .{ .integer = @backingInt(row.program_fault) },
                .{ .integer = @backingInt(row.metric_fault) },
                .{ .bool = row.score },
            };
            for (example.record.values(), expected_values) |actual, expected| {
                if (std.meta.activeTag(actual) != std.meta.activeTag(expected)) return false;
                switch (expected) {
                    .integer => if (actual.integer != expected.integer) return false,
                    .bool => if (actual.bool != expected.bool) return false,
                    else => unreachable,
                }
            }
        }
        return true;
    }

    /// Greedy, deterministic, bounded shrinking; every accepted change retains
    /// the SAME violated invariant, not just an arbitrary failure.
    pub fn shrink(self: *Harness, scenario: *Case, violation: Violation, mode: Mode) u32 {
        assert(valid_case(scenario));
        assert(self.check(scenario, mode) == violation);
        var attempts: u32 = 0;
        var changed = true;
        while (changed and attempts < shrink_attempts_max) {
            changed = self.shrink_rows(scenario, violation, mode, &attempts);
            if (attempts == shrink_attempts_max) break;
            changed = self.shrink_fields(scenario, violation, mode, &attempts) or changed;
        }
        assert(attempts <= shrink_attempts_max);
        assert(self.check(scenario, mode) == violation);
        return attempts;
    }

    fn shrink_rows(
        self: *Harness,
        scenario: *Case,
        violation: Violation,
        mode: Mode,
        attempts: *u32,
    ) bool {
        assert(valid_case(scenario));
        var changed = false;
        var index: u8 = 0;
        while (scenario.len > 1 and index < scenario.len and attempts.* < shrink_attempts_max) {
            const removed = scenario.rows[index];
            const tail = scenario.len - index - 1;
            std.mem.copyForwards(
                Row,
                scenario.rows[index..][0..tail],
                scenario.rows[index + 1 ..][0..tail],
            );
            scenario.len -= 1;
            scenario.rows[scenario.len] = .{};
            if (self.retain(scenario, violation, mode, attempts)) {
                changed = true;
            } else {
                // Restore only the removed row and shifted suffix, rather
                // than copying the whole fixed-capacity case per attempt.
                std.mem.copyBackwards(
                    Row,
                    scenario.rows[index + 1 ..][0..tail],
                    scenario.rows[index..][0..tail],
                );
                scenario.rows[index] = removed;
                scenario.len += 1;
                index += 1;
            }
        }
        assert(scenario.len > 0);
        return changed;
    }

    fn shrink_fields(
        self: *Harness,
        scenario: *Case,
        violation: Violation,
        mode: Mode,
        attempts: *u32,
    ) bool {
        assert(valid_case(scenario));
        var changed = false;
        // Every accepted step strictly decreases this finite lexicographic
        // complexity: length, budget, then each field's magnitude/tag/score.
        if (scenario.max_errors != 0 and attempts.* < shrink_attempts_max) {
            scenario.max_errors -= 1;
            if (self.retain(scenario, violation, mode, attempts)) {
                changed = true;
            } else {
                scenario.max_errors += 1;
            }
        }
        for (0..scenario.len) |index| {
            inline for (.{ "value", "program_fault", "metric_fault", "score" }) |field| {
                if (attempts.* == shrink_attempts_max) return changed;
                const original = @field(scenario.rows[index], field);
                const Value = @TypeOf(original);
                const simple: Value = switch (@typeInfo(Value)) {
                    .bool => false,
                    .int => 0,
                    .@"enum" => .none,
                    else => unreachable,
                };
                if (original != simple) {
                    @field(scenario.rows[index], field) = simple;
                    if (self.retain(scenario, violation, mode, attempts)) {
                        changed = true;
                    } else {
                        @field(scenario.rows[index], field) = original;
                    }
                }
            }
        }
        assert(valid_case(scenario));
        return changed;
    }

    fn retain(
        self: *Harness,
        scenario: *const Case,
        violation: Violation,
        mode: Mode,
        attempts: *u32,
    ) bool {
        assert(valid_case(scenario));
        assert(attempts.* < shrink_attempts_max);
        attempts.* += 1;
        return self.check(scenario, mode) == violation;
    }
};

const shrink_attempts_max: u32 = 4096;

fn unexpected_complete(
    pointer: *anyopaque,
    _: Allocator,
    _: dszig.lm.Request,
) dszig.lm.Error![]const dszig.lm.Completion {
    const calls: *u32 = @ptrCast(@alignCast(pointer));
    assert(calls.* < rows_max);
    calls.* += 1;
    return error.LMFailure;
}
