//! Actual typed native program calls on startup-created workers, with zero-byte
//! per-attempt scratch. Every parallel output is compared to a serial execution.
const std = @import("std");
const dszig = @import("dszig");
const count = 64;
const rounds = 128;
const Compute = struct {
    compiled: bool = false,
    pub const Input = struct { value: u64 };
    pub const Output = u64;

    pub fn forward(_: *const Compute, _: dszig.Context, input: Input) !Output {
        var value = input.value;
        for (0..4096) |_| value = std.math.rotl(u64, value *% 0x9e3779b97f4a7c15 +% 17, 13);
        return value;
    }
};
const Pool = dszig.Parallel(dszig.parallel.ProgramResult(Compute), .{
    .workers = 4,
    .tasks = count,
    .attempts = count,
    .failures = count,
});

pub fn main(init: std.process.Init) !void {
    var pool: Pool = undefined;
    try pool.init(init.io, .{});
    defer pool.deinit();
    var scratch: [count]std.heap.FixedBufferAllocator = undefined;
    var buffers: [count][0]u8 = undefined;
    var models: [count]dszig.lm.ScriptedLM = undefined;
    var contexts: [count]dszig.Context = undefined;
    var calls: [count]Pool.Program(Compute) = undefined;
    var tasks: [count]Pool.Task = undefined;
    const program: Compute = .{};
    for (0..count) |index| {
        scratch[index] = .init(&buffers[index]);
        models[index] = .{ .responses = &.{} };
        contexts[index] = .{ .arena = scratch[index].allocator(), .lm = models[index].lm() };
        calls[index] = .{
            .program = &program,
            .arguments = .{ .input = .{ .value = index } },
            .contexts = &contexts,
        };
        tasks[index] = calls[index].task();
    }
    var expected: [count]u64 = undefined;
    var checksum: u64 = 0;
    var parallel_ns: i96 = 0;
    var serial_ns: i96 = 0;
    for (0..rounds) |round| {
        for (&calls, 0..) |*call, index| call.arguments.input.value = round * count + index;
        const serial_start = std.Io.Clock.awake.now(init.io);
        for (&calls, 0..) |*call, index| {
            expected[index] = try program.forward(contexts[index], call.arguments.input);
        }
        serial_ns += serial_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
        const parallel_start = std.Io.Clock.awake.now(init.io);
        const report = try pool.forward(&tasks, .{ .straggler_timeout_ns = 0 });
        parallel_ns += parallel_start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
        for (report.results, expected) |slot, wanted| {
            if (slot != .value or slot.value.* != wanted) return error.NativeParallelMismatch;
            checksum +%= slot.value.*;
        }
    }
    var scratch_bytes: usize = 0;
    var lm_calls: usize = 0;
    for (scratch, models) |allocator, model| {
        scratch_bytes += allocator.end_index;
        lm_calls += model.index;
    }
    if (scratch_bytes != 0 or lm_calls != 0) return error.UnexpectedExecutionResourceUse;
    var buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    try stdout.interface.print(
        "native parallel: runs={d} checksum={d} parallel_ns={d} serial_ns={d} scratch_bytes={d} lm_calls={d}\n",
        .{ count * rounds, checksum, parallel_ns, serial_ns, scratch_bytes, lm_calls },
    );
    try stdout.interface.flush();
}
