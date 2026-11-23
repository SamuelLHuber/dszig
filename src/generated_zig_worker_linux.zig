//! Trusted PID1 supervisor of one generated-Zig run. It executes only inside the
//! fresh namespace child created by generated_zig_namespace_linux.zig, after
//! identity mapping and cgroup attachment. Ordering is the safety argument:
//! descriptors, private filesystem, rlimits, in-child source validation, trusted
//! file materialization, policy-confined compiler, complete descendant kill and
//! reap, then the policy-entering runtime. The host owns the absolute deadline
//! and kills the lane when any blocking step here outlives it.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const assert = std.debug.assert;
const namespace = @import("generated_zig_namespace_linux.zig");
const policy = @import("generated_zig_policy_linux.zig");
const source_policy = @import("generated_zig_source.zig");

pub const Config = struct {
    workspace: [:0]const u8,
    compiler: [:0]const u8,
    read_only_roots: []const [:0]const u8,
    artifact_bytes: u32 = 128 * 1024 * 1024,
};

/// Everything is inherited copy-on-write by the namespace child and must stay
/// stable until spawn returns. The parent closes its child endpoints afterwards.
pub const Context = struct {
    config: Config,
    source: [:0]const u8,
    scratch: *[2097152]u8,
    socket_child: i32,
    socket_parent: i32,
    diagnostics_read: i32,
    diagnostics_write: i32,
};

/// Supervisor exit codes. Zero is published only after the runtime exited zero.
pub const exit_unsupported: u8 = 100;
pub const exit_descriptors: u8 = 101;
pub const exit_configuration: u8 = 102;
pub const exit_filesystem: u8 = 103;
pub const exit_limits: u8 = 104;
pub const exit_source: u8 = 105;
pub const exit_materialize: u8 = 106;
pub const exit_compiler_spawn: u8 = 107;
pub const exit_compiler_failed: u8 = 108;
pub const exit_reap: u8 = 109;
pub const exit_runtime_spawn: u8 = 110;
pub const exit_runtime_failed: u8 = 111;
/// Used only by the forked compiler/runtime children before their execve.
const exit_child_stdio: u8 = 120;
const exit_child_policy: u8 = 121;
const exit_child_exec: u8 = 122;

const interruptions_max: u8 = 64;
/// The lane's pids.max is at most 64, and unreaped zombies stay charged.
const tasks_max: u8 = 64;
const write_attempts_max: u16 = 1024;
const diagnostic_bytes_max: u16 = 256;
const signal_kill: usize = 9;
const signal_child: usize = 17;
const every_process: usize = std.math.maxInt(usize);

const target: ?[*:0]const u8 = switch (builtin.cpu.arch) {
    .aarch64 => "aarch64-linux-none",
    .x86_64 => "x86_64-linux-none",
    else => null,
};

const guest_source = @embedFile("generated_zig_guest.zig");
const protocol_source = @embedFile("generated_zig_protocol.zig");
const policy_source = @embedFile("generated_zig_policy_linux.zig");

/// The generated module runs only after enterGuest succeeded. Every failure
/// exits nonzero without std's stack dumping, which the guest policy forbids.
const wrapper_source =
    \\const std = @import("std");
    \\const guest = @import("guest");
    \\const policy = @import("policy");
    \\const generated = @import("generated");
    \\const linux = std.os.linux;
    \\
    \\pub const std_options: std.Options = .{ .enable_segfault_handler = false };
    \\pub const panic = std.debug.FullPanic(halt);
    \\
    \\var context: guest.Context = .{};
    \\
    \\fn report(message: []const u8) void {
    \\    const bytes = @min(message.len, 256);
    \\    _ = linux.syscall3(.write, 2, @intFromPtr(message.ptr), bytes);
    \\    _ = linux.syscall3(.write, 2, @intFromPtr("\n"), 1);
    \\}
    \\
    \\fn halt(message: []const u8, _: ?usize) noreturn {
    \\    report(message);
    \\    linux.exit_group(70);
    \\}
    \\
    \\pub fn main() noreturn {
    \\    policy.enterGuest() catch |err| {
    \\        report(@errorName(err));
    \\        linux.exit_group(71);
    \\    };
    \\    context.start() catch |err| {
    \\        report(@errorName(err));
    \\        linux.exit_group(72);
    \\    };
    \\    generated.run(&context) catch |err| {
    \\        report(@errorName(err));
    \\        linux.exit_group(73);
    \\    };
    \\    if (!context.finished) {
    \\        report("GeneratedRunDidNotFinish");
    \\        linux.exit_group(74);
    \\    }
    \\    linux.exit_group(0);
    \\}
    \\
;

const Status = struct {
    code: u8,
    signal: u8,

    fn decode(status: i32) Status {
        const bits: u32 = @bitCast(status);
        return .{ .code = @intCast((bits >> 8) & 255), .signal = @intCast(bits & 127) };
    }

    fn success(status: Status) bool {
        return status.code == 0 and status.signal == 0;
    }
};

/// Namespace PID1 entry. The returned code becomes the direct child's exit code;
/// PID1 exit makes the kernel kill and reap any descendant still alive.
pub fn entry(raw: *anyopaque) u8 {
    if (comptime builtin.os.tag != .linux) return exit_unsupported;
    if (comptime target == null) return exit_unsupported;
    const context: *const Context = @ptrCast(@alignCast(raw));
    if (!installDescriptors(context)) return exit_descriptors;
    // From here FD0/FD1 are the protocol socket and FD2 is the diagnostics pipe.
    const config = &context.config;
    if (!std.mem.startsWith(u8, config.compiler, "/nix/store/") or
        config.compiler.len > 4095 or config.read_only_roots.len == 0 or
        config.read_only_roots.len > policy.roots_max)
    {
        return fail(exit_configuration, "invalid configuration", null);
    }
    namespace.privateFilesystem(config.workspace, config.artifact_bytes) catch |err|
        return fail(exit_filesystem, "private filesystem", @errorName(err));
    if (!installLimits()) return fail(exit_limits, "resource limits", null);

    var scratch = std.heap.FixedBufferAllocator.init(context.scratch);
    source_policy.validate(scratch.allocator(), context.source) catch |err|
        return fail(exit_source, "source rejected", @errorName(err));

    if (!writeFile("generated_zig_guest.zig", guest_source) or
        !writeFile("generated_zig_protocol.zig", protocol_source) or
        !writeFile("generated_zig_policy_linux.zig", policy_source) or
        !writeFile("main.zig", wrapper_source) or
        !writeFile("generated.zig", context.source))
    {
        return fail(exit_materialize, "source materialization", null);
    }

    const compiler = fork() orelse return fail(exit_compiler_spawn, "compiler fork", null);
    if (compiler == 0) compilerChild(config);
    const compiled = wait(compiler);
    // Unconditional: a successful compiler may still have left descendants.
    if (!killDescendants()) return fail(exit_reap, "compiler descendants not reaped", null);
    const compile_status = compiled orelse return fail(exit_reap, "compiler wait", null);
    if (!compile_status.success())
        return failStatus(exit_compiler_failed, "compiler", compile_status);

    const runtime = fork() orelse return fail(exit_runtime_spawn, "runtime fork", null);
    if (runtime == 0) runtimeChild();
    const runtime_status = wait(runtime) orelse return fail(exit_reap, "runtime wait", null);
    if (!runtime_status.success())
        return failStatus(exit_runtime_failed, "runtime", runtime_status);
    return 0;
}

/// Parent endpoints close first, so the host observes EOF once this namespace
/// dies. O_NONBLOCK is cleared on the child's own open file descriptions only.
fn installDescriptors(context: *const Context) bool {
    const all = [4]i32{ context.socket_child, context.socket_parent, context.diagnostics_read, context.diagnostics_write };
    for (all, 0..) |descriptor, index| {
        if (descriptor < 3) return false;
        for (all[0..index]) |earlier| if (earlier == descriptor) return false;
    }
    if (linux.close(context.socket_parent) != 0) return false;
    if (linux.close(context.diagnostics_read) != 0) return false;
    if (!clearNonblocking(context.socket_child)) return false;
    if (!clearNonblocking(context.diagnostics_write)) return false;
    if (linux.dup3(context.socket_child, 0, 0) != 0) return false;
    if (linux.dup3(context.socket_child, 1, 0) != 1) return false;
    if (linux.dup3(context.diagnostics_write, 2, 0) != 2) return false;
    if (linux.close(context.socket_child) != 0) return false;
    if (linux.close(context.diagnostics_write) != 0) return false;
    // The host's signal mask must not outlive into compiler or guest.
    const empty_mask: u64 = 0;
    return linux.syscall4(.rt_sigprocmask, 2, @intFromPtr(&empty_mask), 0, 8) == 0;
}

fn clearNonblocking(descriptor: i32) bool {
    const nonblocking: usize = @as(u32, @bitCast(linux.O{ .NONBLOCK = true }));
    const flags = linux.fcntl(descriptor, linux.F.GETFL, 0);
    if (failed(flags)) return false;
    if (linux.fcntl(descriptor, linux.F.SETFL, flags & ~nonblocking) != 0) return false;
    const readback = linux.fcntl(descriptor, linux.F.GETFL, 0);
    if (failed(readback)) return false;
    return readback & nonblocking == 0;
}

/// Inherited by the compiler and the runtime. Aggregate memory, tasks and
/// artifact bytes are bounded by the cgroup lane and the workspace tmpfs.
fn installLimits() bool {
    return limit(.CORE, 0) and limit(.NOFILE, 64) and limit(.NPROC, 128) and
        limit(.FSIZE, 32 * 1024 * 1024) and limit(.CPU, 30) and
        limit(.AS, 4 * 1024 * 1024 * 1024) and limit(.STACK, 8 * 1024 * 1024);
}

fn limit(resource: linux.rlimit_resource, value: u64) bool {
    const wanted: linux.rlimit = .{ .cur = value, .max = value };
    if (linux.setrlimit(resource, &wanted) != 0) return false;
    var actual: linux.rlimit = undefined;
    if (linux.prlimit(0, resource, null, &actual) != 0) return false;
    return actual.cur == value and actual.max == value;
}

/// Exclusive creation in the fresh private workspace: nothing may pre-exist.
fn writeFile(name: [*:0]const u8, contents: []const u8) bool {
    const opened = linux.openat(linux.AT.FDCWD, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, 0o400);
    if (failed(opened)) return false;
    const descriptor: i32 = @intCast(opened);
    var offset: usize = 0;
    var complete = false;
    for (0..write_attempts_max) |_| {
        if (offset == contents.len) {
            complete = true;
            break;
        }
        const result = linux.write(descriptor, contents[offset..].ptr, contents.len - offset);
        if (is(result, .INTR)) continue;
        if (result == 0 or result > contents.len - offset) break;
        offset += result;
    }
    if (linux.close(descriptor) != 0) return false;
    return complete;
}

/// Plain fork semantics through clone: aarch64 has no fork system call.
fn fork() ?usize {
    const result = linux.syscall5(.clone, signal_child, 0, 0, 0, 0);
    if (failed(result)) return null;
    return result;
}

/// Compiler output can never reach the protocol stream: stdout is rebound to
/// the diagnostics pipe before the policy and before execve.
fn compilerChild(config: *const Config) noreturn {
    const compile_target = comptime target.?;
    const null_device = linux.openat(linux.AT.FDCWD, "/dev/null", .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, 0);
    if (failed(null_device)) childExit(exit_child_stdio, "compiler stdin", null);
    if (linux.dup3(@intCast(null_device), 0, 0) != 0)
        childExit(exit_child_stdio, "compiler stdin", null);
    if (linux.dup3(2, 1, 0) != 1) childExit(exit_child_stdio, "compiler stdout", null);
    policy.enterCompiler(config.read_only_roots, config.workspace) catch |err|
        childExit(exit_child_policy, "compiler policy", @errorName(err));
    const arguments = [_:null]?[*:0]const u8{
        config.compiler.ptr,  "build-exe",
        "-target",            compile_target,
        "-mcpu",              "baseline",
        "-OReleaseSafe",      "--dep",
        "guest",              "--dep",
        "policy",             "--dep",
        "generated",          "-Mroot=main.zig",
        "-target",            compile_target,
        "-mcpu",              "baseline",
        "-OReleaseSafe",      "-Mguest=generated_zig_guest.zig",
        "-target",            compile_target,
        "-mcpu",              "baseline",
        "-OReleaseSafe",      "-Mpolicy=generated_zig_policy_linux.zig",
        "-target",            compile_target,
        "-mcpu",              "baseline",
        "-OReleaseSafe",      "--dep",
        "guest",              "-Mgenerated=generated.zig",
        "--cache-dir",        "cache",
        "--global-cache-dir", "global-cache",
        "-femit-bin=program", "--color",
        "off",
    };
    _ = linux.execve(config.compiler.ptr, &arguments, &environment);
    childExit(exit_child_exec, "compiler exec", null);
}

/// FD0/FD1 stay the protocol socket and FD2 the diagnostics pipe. The trusted
/// wrapper enters the guest policy before any generated code runs.
fn runtimeChild() noreturn {
    if (linux.syscall3(.close_range, 3, std.math.maxInt(u32), 0) != 0)
        childExit(exit_child_stdio, "runtime descriptors", null);
    const arguments = [_:null]?[*:0]const u8{"program"};
    _ = linux.execve("./program", &arguments, &environment);
    childExit(exit_child_exec, "runtime exec", null);
}

const environment = [_:null]?[*:0]const u8{ "LANG=C", "HOME=/nonexistent" };

fn childExit(code: u8, stage: []const u8, detail: ?[]const u8) noreturn {
    linux.exit_group(fail(code, stage, detail));
}

/// Waits exactly this direct child. Orphans reparented to PID1 stay zombies
/// until killDescendants; the host deadline bounds the blocking wait itself.
fn wait(process: usize) ?Status {
    var status: i32 = 0;
    for (0..@as(u16, interruptions_max) + 1) |_| {
        const result = linux.syscall4(.wait4, process, @intFromPtr(&status), 0, 0);
        if (is(result, .INTR)) continue;
        if (result != process) return null;
        return Status.decode(status);
    }
    return null;
}

/// As namespace PID1, kill(-1) reaches every other process of the namespace
/// and every orphan is our child, so ECHILD proves the namespace holds only
/// this supervisor. A task with a pending fatal signal cannot complete a fork.
fn killDescendants() bool {
    const killed = linux.syscall2(.kill, every_process, signal_kill);
    if (killed != 0 and !is(killed, .SRCH)) return false;
    var status: i32 = 0;
    var interruptions: u8 = 0;
    var reaped: u8 = 0;
    var childless = false;
    for (0..@as(u16, tasks_max) + interruptions_max + 1) |_| {
        const result = linux.syscall4(.wait4, every_process, @intFromPtr(&status), 0, 0);
        if (is(result, .CHILD)) {
            childless = true;
            break;
        }
        if (is(result, .INTR)) {
            if (interruptions == interruptions_max) return false;
            interruptions += 1;
            continue;
        }
        if (failed(result)) return false;
        if (reaped == tasks_max) return false;
        reaped += 1;
    }
    if (!childless) return false;
    return is(linux.syscall2(.kill, every_process, 0), .SRCH);
}

fn failStatus(code: u8, stage: []const u8, status: Status) u8 {
    var buffer: [48]u8 = undefined;
    const detail = std.fmt.bufPrint(&buffer, "exit code {d} signal {d}", .{ status.code, status.signal }) catch unreachable;
    return fail(code, stage, detail);
}

/// Best-effort bounded diagnostic on FD2; the exit code is the authority.
fn fail(code: u8, stage: []const u8, detail: ?[]const u8) u8 {
    assert(code != 0);
    var buffer: [diagnostic_bytes_max]u8 = undefined;
    const prefix = "dszig-worker: ";
    var length: usize = 0;
    for ([_][]const u8{ prefix, stage, if (detail != null) ": " else "", detail orelse "", "\n" }) |part| {
        const bytes = @min(part.len, buffer.len - length);
        @memcpy(buffer[length..][0..bytes], part[0..bytes]);
        length += bytes;
    }
    _ = linux.write(2, &buffer, length);
    return code;
}

fn failed(result: usize) bool {
    return result > std.math.maxInt(usize) - 4096;
}

fn is(result: usize, errno: linux.E) bool {
    return result == 0 -% @as(usize, @backingInt(errno));
}

comptime {
    assert(source_policy.scratch_bytes_recommended == 2097152);
    assert(guest_source.len <= 1024 * 1024);
    assert(protocol_source.len <= 1024 * 1024);
    assert(policy_source.len <= 1024 * 1024);
}
