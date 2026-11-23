//! Deterministic simulation of bounded guest globals, arithmetic, and host effects.
const std = @import("std");
const guest = @import("dszig").bounded_python;

const generated_cases = 256;
const replays = 3;
const run_options: guest.Options = .{ .fuel = 4096 };

const setup_source = "stale_marker = 91\n";
const marker_source = "print(stale_marker)\n";
const arithmetic_source =
    \\total = start + numerator - divisor * scale
    \\SUBMIT(total=total, quotient=numerator // divisor, remainder=numerator % divisor)
;
const loop_source =
    \\i = 0
    \\while i < limit:
    \\    total = total + i * stride - offset
    \\    i = i + 1
    \\print(total)
    \\print(i)
;
const host_print_source =
    \\reply = llm_query("lookup")
    \\print(reply)
    \\print(llm_query("describe"))
;
const host_submit_source =
    \\SUBMIT(zeta=llm_query(reply), alpha=llm_query("finish"), total=total)
;

const Options = struct {
    seed: u64 = 0,
    inject_wrong_result: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = parse_options(args[1..]) catch |err| {
        std.log.err("usage: bounded-python-sim [--seed N] [--inject-wrong-result]", .{});
        return err;
    };
    var machine: guest.Machine = undefined;
    machine.init();

    // -7 // 3 = -3 and -7 % 3 = 2. The total is 10 - 7 - 3*4 = -9;
    // the three loop terms are -1, 1, 3, leaving -6.
    const control: Scenario = .{
        .start = 10,
        .divisor = 3,
        .quotient = -3,
        .remainder = 2,
        .scale = 4,
        .limit = 3,
        .stride = 2,
        .offset = 1,
    };
    const control_expected: Expected = .{ .arithmetic_total = -9, .loop_total = -6 };
    try run_scenario(&machine, options.seed, 0, &control, control_expected, false);
    if (options.inject_wrong_result) {
        // Require the unmodified control to pass before attributing a failure to the fault.
        try run_scenario(&machine, options.seed, 0, &control, control_expected, true);
        std.log.err("seed={d} case=0: planted wrong result was not detected", .{options.seed});
        return error.ControlNotDetected;
    }

    var random = std.Random.DefaultPrng.init(options.seed);
    for (0..generated_cases) |index| {
        const scenario = Scenario.generate(random.random(), index);
        try run_scenario(&machine, options.seed, index + 1, &scenario, scenario.expected(), false);
    }
    var buffer: [256]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    try stdout.interface.print("bounded-python simulation passed: seed={d} cases={d} replays={d}\n", .{
        options.seed, generated_cases + 1, replays,
    });
    try stdout.interface.flush();
}

fn parse_options(args: []const []const u8) !Options {
    if (args.len > 3) return error.InvalidArguments;
    var options: Options = .{};
    var seed_seen = false;
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--seed")) {
            if (seed_seen or index + 1 == args.len) return error.InvalidArguments;
            index += 1;
            options.seed = try std.fmt.parseInt(u64, args[index], 0);
            seed_seen = true;
        } else if (std.mem.eql(u8, arg, "--inject-wrong-result")) {
            if (options.inject_wrong_result) return error.InvalidArguments;
            options.inject_wrong_result = true;
        } else return error.InvalidArguments;
    }
    return options;
}

const Expected = struct {
    arithmetic_total: i64,
    loop_total: i64,
};

const Scenario = struct {
    start: i64,
    divisor: i64,
    quotient: i64,
    remainder: i64,
    scale: i64,
    limit: i64,
    stride: i64,
    offset: i64,

    fn generate(random: std.Random, index: usize) Scenario {
        const magnitude = random.intRangeAtMost(i64, 1, 11);
        const sign: i64 = if (random.boolean()) 1 else -1;
        return .{
            .start = random.intRangeAtMost(i64, -100, 100),
            .divisor = sign * magnitude,
            .quotient = random.intRangeAtMost(i64, -11, 11),
            .remainder = sign * random.intRangeAtMost(i64, 0, magnitude - 1),
            .scale = random.intRangeAtMost(i64, 1, 11),
            .limit = switch (index) {
                0 => 0,
                1 => 1,
                else => random.intRangeAtMost(i64, 2, 16),
            },
            .stride = random.intRangeAtMost(i64, -9, 9),
            .offset = random.intRangeAtMost(i64, -9, 9),
        };
    }

    fn numerator(self: *const Scenario) i64 {
        // Construct a dividend from its floor quotient and divisor-signed remainder.
        return self.divisor * self.quotient + self.remainder;
    }

    fn expected(self: *const Scenario) Expected {
        // Use the constructed quotient and remainder instead of guest division or modulo.
        const total = self.start + self.divisor * (self.quotient - self.scale) + self.remainder;
        const index_sum = @divExact(self.limit * (self.limit - 1), 2);
        return .{
            .arithmetic_total = total,
            .loop_total = total + self.stride * index_sum - self.offset * self.limit,
        };
    }
};

fn run_scenario(
    machine: *guest.Machine,
    seed: u64,
    index: usize,
    scenario: *const Scenario,
    expected: Expected,
    inject_wrong_result: bool,
) !void {
    // A fault changes only an input operand. Both modes use the same guest and checker.
    const inputs = [_]guest.Binding{
        integer("start", scenario.start),
        integer("numerator", scenario.numerator()),
        integer("divisor", scenario.divisor),
        integer("scale", scenario.scale + @as(i64, @intFromBool(inject_wrong_result))),
        integer("limit", scenario.limit),
        integer("stride", scenario.stride),
        integer("offset", scenario.offset),
    };
    for (0..replays) |replay| {
        machine.reset();
        run_replay(machine, scenario, &inputs, expected) catch |err| {
            std.log.err("seed={d} case={d} replay={d}: {s}", .{ seed, index, replay, @errorName(err) });
            return err;
        };
    }
}

fn run_replay(
    machine: *guest.Machine,
    scenario: *const Scenario,
    inputs: []const guest.Binding,
    expected: Expected,
) !void {
    try expect_observation("setup", machine.run(setup_source, inputs, run_options, null), "");
    try expect_submit("arithmetic", machine.run(arithmetic_source, &.{}, run_options, null), &.{
        integer("total", expected.arithmetic_total),
        integer("quotient", scenario.quotient),
        integer("remainder", scenario.remainder),
    });
    var output_buffer: [64]u8 = undefined;
    const loop_output = try std.fmt.bufPrint(&output_buffer, "{d}\n{d}\n", .{
        expected.loop_total, scenario.limit,
    });
    try expect_observation("loop", machine.run(loop_source, &.{}, run_options, null), loop_output);
    try run_host_effects(machine, expected.loop_total);

    // Prove the marker is present before clearing it. Probe separately so this empty
    // input slice cannot consume the next replay's one-time input seeding.
    try expect_observation("dirty marker", machine.run(marker_source, &.{}, run_options, null), "91\n");
    machine.reset();
    try expect_error(
        "reset marker",
        machine.run(marker_source, &.{}, run_options, null),
        error.NameError,
    );
}

fn run_host_effects(machine: *guest.Machine, loop_total: i64) !void {
    var script: ScriptedHost = .{};
    const host: guest.Host = .{ .ptr = &script, .query = ScriptedHost.query };
    try expect_observation(
        "host print",
        machine.run(host_print_source, &.{}, run_options, host),
        "ticket\ndetails\n",
    );
    try script.expect_cursor("host print", 2);
    try expect_submit("host submit", machine.run(host_submit_source, &.{}, run_options, host), &.{
        .{ .name = "zeta", .value = .{ .string = "combined" } },
        .{ .name = "alpha", .value = .{ .string = "complete" } },
        integer("total", loop_total),
    });
    try script.expect_cursor("host submit", 4);
    try expect_error(
        "host denial",
        machine.run("llm_query(\"deny\")\nllm_query(\"unreachable\")\n", &.{}, run_options, host),
        error.ScriptedDenial,
    );
    try script.expect_cursor("host denial", ScriptedHost.steps.len);
}

const ScriptedHost = struct {
    cursor: usize = 0,
    // Count attempted calls as well as matched calls, including an unexpected call
    // swallowed by the guest after the scripted denial.
    attempts: usize = 0,

    const steps = [_]struct { request: []const u8, reply: ?[]const u8 }{
        .{ .request = "lookup", .reply = "ticket" },
        .{ .request = "describe", .reply = "details" },
        .{ .request = "ticket", .reply = "combined" },
        .{ .request = "finish", .reply = "complete" },
        .{ .request = "deny", .reply = null },
    };

    fn query(ptr: *anyopaque, request: []const u8) anyerror![]const u8 {
        const self: *ScriptedHost = @ptrCast(@alignCast(ptr));
        self.attempts += 1;
        if (self.cursor == steps.len) {
            std.log.err("host order: unexpected extra request '{s}'", .{request});
            return error.HostOrderMismatch;
        }
        const step = steps[self.cursor];
        if (!std.mem.eql(u8, step.request, request)) {
            std.log.err("host order: request[{d}] expected '{s}', got '{s}'", .{
                self.cursor, step.request, request,
            });
            return error.HostOrderMismatch;
        }
        self.cursor += 1;
        return step.reply orelse return error.ScriptedDenial;
    }

    fn expect_cursor(self: *const ScriptedHost, check: []const u8, expected: usize) !void {
        if (self.cursor != expected or self.attempts != expected) {
            std.log.err("{s}: expected {d} host calls, matched {d} of {d} attempts", .{
                check, expected, self.cursor, self.attempts,
            });
            return error.HostOrderMismatch;
        }
    }
};

fn integer(name: []const u8, value: i64) guest.Binding {
    return .{ .name = name, .value = .{ .integer = value } };
}

fn expect_observation(check: []const u8, outcome: anyerror!guest.Result, expected: []const u8) !void {
    const result = try expect_success(check, outcome);
    switch (result) {
        .observation => |actual| {
            if (std.mem.eql(u8, expected, actual)) return;
            std.log.err("{s}: expected output {any}, got {any}", .{ check, expected, actual });
            return error.ObservationMismatch;
        },
        .submit => {
            std.log.err("{s}: expected observation, got submit", .{check});
            return error.ResultKindMismatch;
        },
    }
}

fn expect_submit(check: []const u8, outcome: anyerror!guest.Result, expected: []const guest.Binding) !void {
    const result = try expect_success(check, outcome);
    const actual = switch (result) {
        .observation => {
            std.log.err("{s}: expected submit, got observation", .{check});
            return error.ResultKindMismatch;
        },
        .submit => |bindings| bindings,
    };
    if (expected.len != actual.len) {
        std.log.err("{s}: expected {d} submit bindings, got {d}", .{ check, expected.len, actual.len });
        return error.SubmitMismatch;
    }
    for (expected, actual, 0..) |want, got, index| {
        if (!std.mem.eql(u8, want.name, got.name) or !value_equal(want.value, got.value)) {
            std.log.err("{s}: submit[{d}] expected {s}={any}, got {s}={any}", .{
                check, index, want.name, want.value, got.name, got.value,
            });
            return error.SubmitMismatch;
        }
    }
}

fn value_equal(expected: guest.Value, actual: guest.Value) bool {
    if (std.meta.activeTag(expected) != std.meta.activeTag(actual)) return false;
    return switch (expected) {
        .none => true,
        .boolean => |value| value == actual.boolean,
        .integer => |value| value == actual.integer,
        .string => |value| std.mem.eql(u8, value, actual.string),
    };
}

fn expect_success(check: []const u8, outcome: anyerror!guest.Result) !guest.Result {
    return outcome catch |err| {
        std.log.err("{s}: unexpected guest error {s}", .{ check, @errorName(err) });
        return err;
    };
}

fn expect_error(check: []const u8, outcome: anyerror!guest.Result, expected: anyerror) !void {
    if (outcome) |_| {
        std.log.err("{s}: expected {s}, but guest succeeded", .{ check, @errorName(expected) });
    } else |actual| {
        if (actual == expected) return;
        std.log.err("{s}: expected {s}, got {s}", .{ check, @errorName(expected), @errorName(actual) });
    }
    return error.GuestErrorMismatch;
}
