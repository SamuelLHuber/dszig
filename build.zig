const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dszig = b.addModule("dszig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{ .root_module = dszig });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const golden = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/golden.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "dszig", .module = dszig }},
    }) });
    const run_golden = b.addRunArtifact(golden);

    const test_step = b.step("test", "Run unit and DSPy golden tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_golden.step);
}
