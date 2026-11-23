const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // Zig 0.17.0's self-hosted x86_64 backend (the Debug default) miscompiles
    // `?anyerror` equality and fails to encode src/decimal.zig, so Debug
    // x86_64 builds use LLVM unless -Dllvm says otherwise.
    const use_llvm: ?bool = b.option(bool, "llvm", "Force (true) or forbid (false) the LLVM backend") orelse
        if (optimize == .debug and target.result.cpu.arch == .x86_64) true else null;

    const dszig = b.addModule("dszig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{ .root_module = dszig, .use_llvm = use_llvm });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const rag_program = b.createModule(.{
        .root_source_file = b.path("examples/rag_program.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "dszig", .module = dszig }},
    });
    const live_suite = b.createModule(.{
        .root_source_file = b.path("examples/live_suite.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "dszig", .module = dszig },
            .{ .name = "rag_program", .module = rag_program },
        },
    });
    const evaluation_simulation = b.createModule(.{
        .root_source_file = b.path("tools/evaluation_simulation.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "dszig", .module = dszig }},
    });
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "dszig", .module = dszig },
        .{ .name = "rag_program", .module = rag_program },
        .{ .name = "live_suite", .module = live_suite },
        .{ .name = "evaluation_simulation", .module = evaluation_simulation },
    };

    const test_step = b.step("test", "Run unit, DSPy golden, behaviour and compile-fail tests");
    test_step.dependOn(&run_unit_tests.step);
    for ([_][]const u8{ "tests/golden.zig", "tests/behavior.zig", "tests/live_replay.zig", "tests/optuna.zig", "tests/trace_selection.zig", "tests/simulation.zig", "tests/parallel.zig" }) |path| {
        const t = b.addTest(.{ .use_llvm = use_llvm, .root_module = b.createModule(.{
            .root_source_file = b.path(path),
            .target = target,
            .optimize = optimize,
            .imports = imports,
        }) });
        const run_test = b.addRunArtifact(t);
        test_step.dependOn(&run_test.step);
        if (std.mem.eql(u8, path, "tests/simulation.zig")) {
            b.step("simulation-test", "Run native evaluator simulation properties").dependOn(
                &run_test.step,
            );
        }
    }
    addCompileFailTests(b, test_step);

    const examples_step = b.step("examples", "Build and run the examples");
    for ([_][]const u8{ "rag", "bootstrap", "parallel" }) |name| {
        const exe = b.addExecutable(.{ .name = name, .use_llvm = use_llvm, .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("examples/{s}.zig", .{name})),
            .target = target,
            .optimize = optimize,
            .imports = imports,
        }) });
        b.installArtifact(exe);
        const run = b.addRunArtifact(exe);
        run.addPassthruArgs();
        b.step(name, b.fmt("Run examples/{s}.zig", .{name})).dependOn(&run.step);
        examples_step.dependOn(&b.addRunArtifact(exe).step);
    }

    // Needs OPENAI_API_KEY and network, so it is not part of `examples`.
    const live = b.addExecutable(.{ .name = "live", .use_llvm = use_llvm, .root_module = b.createModule(.{
        .root_source_file = b.path("examples/live.zig"),
        .target = target,
        .optimize = optimize,
        .imports = imports,
    }) });
    b.installArtifact(live);
    const run_live = b.addRunArtifact(live);
    run_live.addPassthruArgs();
    b.step("live", "Run examples/live.zig against a real OpenAI-compatible model").dependOn(&run_live.step);
    examples_step.dependOn(&live.step);

    const replay = b.addExecutable(.{ .name = "replay", .use_llvm = use_llvm, .root_module = b.createModule(.{
        .root_source_file = b.path("examples/replay.zig"),
        .target = target,
        .optimize = optimize,
        .imports = imports,
    }) });
    b.installArtifact(replay);
    const run_replay = b.addRunArtifact(replay);
    run_replay.addPassthruArgs();
    b.step("replay", "Verify a DSPy 3.5 transcript through the native Zig suite").dependOn(&run_replay.step);
    examples_step.dependOn(&replay.step);

    const bench = b.addExecutable(.{ .name = "bench", .use_llvm = use_llvm, .root_module = b.createModule(.{
        .root_source_file = b.path("examples/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = imports,
    }) });
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    run_bench.addPassthruArgs();
    b.step("bench", "Time the native suite replaying a transcript (compare with tools/dspy_reference/bench.py)").dependOn(&run_bench.step);
    examples_step.dependOn(&bench.step);

    const simulate = b.addExecutable(.{ .name = "simulate", .use_llvm = use_llvm, .root_module = b.createModule(.{
        .root_source_file = b.path("tools/simulate.zig"),
        .target = target,
        .optimize = optimize,
        .imports = imports,
    }) });
    b.installArtifact(simulate);
    const run_simulate = b.addRunArtifact(simulate);
    run_simulate.addPassthruArgs();
    b.step("simulate", "Run seeded native evaluator simulation or a concrete replay").dependOn(
        &run_simulate.step,
    );

    const python_sim = b.addExecutable(.{ .name = "bounded-python-sim", .use_llvm = use_llvm, .root_module = b.createModule(.{
        .root_source_file = b.path("tools/bounded_python_sim.zig"),
        .target = target,
        .optimize = optimize,
        .imports = imports,
    }) });
    const run_python_sim = b.addRunArtifact(python_sim);
    run_python_sim.addPassthruArgs();
    b.step("bounded-python-sim", "Run deterministic bounded Python guest simulation").dependOn(&run_python_sim.step);
    const run_python_sim_tests = b.addRunArtifact(python_sim);
    run_python_sim_tests.addArgs(&.{ "--seed", "42" });
    test_step.dependOn(&run_python_sim_tests.step);
}

/// Each tests/compile_fail/<name>.zig must fail to compile with `message`.
fn addCompileFailTests(b: *std.Build, step: *std.Build.Step) void {
    const cases = [_][2][]const u8{
        .{ "duplicate_field", "is declared as both input and output" },
        .{ "invalid_name", "must match \\\\w+" },
        .{ "missing_output_decl", "must declare `pub const Output" },
        .{ "no_outputs", "signature must declare at least one output field" },
        .{ "non_exhaustive_enum", "enum must be exhaustive" },
        .{ "reserved_completed", "is reserved by the chat protocol" },
        .{ "unknown_description", "description for unknown field `answr`" },
        .{ "unsupported_type", "has unsupported type" },
        .{ "intenum_unicode_bytes", "invalid IntEnum integer bytes" },
        .{ "intenum_over_limit_bytes", "exceeds CPython's default 4300-digit limit" },
        .{ "annotated_choice_type", "Choice criteria must match the Literal members, including their Python types." },
        .{ "annotated_float_score", "Use Score[...] directly as the field type, not Annotated[float, Score[...]]." },
    };
    for (cases) |case| {
        const run = b.addSystemCommand(&.{ b.graph.zig_exe, "test", "--dep", "dszig" });
        run.addPrefixedFileArg("-Mroot=", b.path(b.fmt("tests/compile_fail/{s}.zig", .{case[0]})));
        run.addPrefixedFileArg("-Mdszig=", b.path("src/root.zig"));
        run.setName(b.fmt("compile-fail {s}", .{case[0]}));
        run.expectExitCode(1);
        run.addCheck(.{ .expect_stderr_match = case[1] });
        step.dependOn(&run.step);
    }
}
