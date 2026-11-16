//! Bounded sequential evaluator simulation, concrete replay and shrinking.
const std = @import("std");
const simulation = @import("evaluation_simulation");

const Options = struct {
    seed: u64 = 0,
    cases: u32 = 10000,
    replay_path: ?[]const u8 = null,
    mode: simulation.Mode = .correct,
};
const cases_max: u32 = 1_000_000;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const options = parse_options(args[1..]) catch |err| {
        std.log.err("usage: simulate [--seed N] [--cases 1..1000000] [--wrong-answer] " ++
            "[--replay concrete-case.json]", .{});
        return err;
    };
    var harness: simulation.Harness = undefined;
    try harness.init();
    var output_buffer: [16 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &output_buffer);
    if (options.replay_path) |path| {
        // Loading/parsing the bounded replay belongs to startup, not execution.
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, arena, .limited(16 * 1024));
        var replay = try std.json.parseFromSliceLeaky(simulation.Replay, arena, bytes, .{});
        if (replay.version != simulation.replay_version) return error.InvalidReplayVersion;
        if (!simulation.valid_case(&replay.scenario)) return error.InvalidReplayCase;
        try exercise(&harness, &replay, options.mode, &stdout.interface);
        try stdout.interface.print("simulation replay passed: version={d} seed={d} case={d}\n", .{
            replay.version, replay.seed, replay.case_index,
        });
    } else {
        for (0..options.cases) |index| {
            var replay: simulation.Replay = .{
                .seed = options.seed,
                .case_index = @intCast(index),
                .scenario = undefined,
            };
            simulation.generate(&replay.scenario, options.seed, @intCast(index));
            try exercise(&harness, &replay, options.mode, &stdout.interface);
        }
        try stdout.interface.print("simulation passed: version={d} seed={d} cases={d}\n", .{
            simulation.replay_version, options.seed, options.cases,
        });
    }
    try stdout.interface.flush();
}

fn parse_options(args: []const []const u8) !Options {
    var options: Options = .{};
    var campaign_options = false;
    var index: usize = 0;
    if (args.len > 7) return error.InvalidArguments;
    while (index < args.len) : (index += 1) {
        const argument = args[index];
        if (std.mem.eql(u8, argument, "--wrong-answer")) {
            if (options.mode == .wrong_answer) return error.InvalidArguments;
            options.mode = .wrong_answer;
            continue;
        }
        if (index + 1 == args.len) return error.InvalidArguments;
        index += 1;
        const value = args[index];
        if (std.mem.eql(u8, argument, "--seed")) {
            options.seed = try std.fmt.parseInt(u64, value, 0);
            campaign_options = true;
        } else if (std.mem.eql(u8, argument, "--cases")) {
            options.cases = try std.fmt.parseInt(u32, value, 10);
            campaign_options = true;
        } else if (std.mem.eql(u8, argument, "--replay")) {
            if (options.replay_path != null) return error.InvalidArguments;
            options.replay_path = value;
        } else return error.InvalidArguments;
    }
    if (options.cases == 0 or options.cases > cases_max) return error.InvalidArguments;
    if (options.replay_path != null and campaign_options) return error.InvalidArguments;
    return options;
}

fn exercise(
    harness: *simulation.Harness,
    replay: *simulation.Replay,
    mode: simulation.Mode,
    writer: *std.Io.Writer,
) !void {
    std.debug.assert(simulation.valid_case(&replay.scenario));
    std.debug.assert(replay.version == simulation.replay_version);
    if (harness.check(&replay.scenario, mode)) |violation| {
        const attempts = harness.shrink(&replay.scenario, violation, mode);
        std.log.err("simulation violation={s} mode={s} version={d} seed={d} case={d}; " ++
            "shrink_attempts={d}; concrete replay JSON follows on stdout", .{
            @tagName(violation), @tagName(mode),    replay.version,
            replay.seed,         replay.case_index, attempts,
        });
        try std.json.Stringify.value(replay, .{}, writer);
        try writer.writeByte('\n');
        try writer.flush();
        return error.SimulationViolation;
    }
}
