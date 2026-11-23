//! Irreversible Linux confinement for disposable compiler/guest children only.
//! Namespace and aggregate cgroup setup precede these policies. Neither policy
//! provides deadlines or joins descendants; the external supervisor owns those.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const assert = std.debug.assert;

pub const roots_max: u16 = 256;
pub const Error = error{
    UnsupportedPlatform,
    UnsafeBuildMode,
    InvalidPolicyPath,
    PolicyCapacityExceeded,
    LandlockUnavailable,
    LandlockRuleFailed,
    DescriptorCleanupFailed,
    NoNewPrivilegesFailed,
    SeccompUnavailable,
};

const Instruction = extern struct { code: u16, yes: u8 = 0, no: u8 = 0, value: u32 };
const Program = extern struct { length: u16, instructions: [*]const Instruction };
const Ruleset = extern struct { filesystem: u64, network: u64, scopes: u64 };
const PathRule = extern struct { access: u64, descriptor: i32 };
const filesystem_all: u64 = (1 << 16) - 1;
const read_access: u64 = (1 << 0) | (1 << 2) | (1 << 3);
const file_access: u64 = (1 << 0) | (1 << 1) | (1 << 2);
const instruction_max = 512;

/// Call only in a disposable child, with fresh private workspace and the exact
/// immutable Nix toolchain closure. No host directory, home or whole-store grant.
/// The compiler may fork, but cannot create new namespaces or contact sockets.
pub fn enterCompiler(read_only_roots: []const [:0]const u8, workspace: [:0]const u8) Error!void {
    try supported();
    if (read_only_roots.len == 0) return error.InvalidPolicyPath;
    if (read_only_roots.len > roots_max) return error.PolicyCapacityExceeded;
    try validatePath(workspace, "/tmp/dszig-");
    for (read_only_roots) |path| try validatePath(path, "/nix/store/");
    try closeInheritedDescriptors();
    try noNewPrivileges();
    try compilerFilesystem(read_only_roots, workspace);
    var filter: Filter = .{};
    filter.begin();
    filter.compilerCalls();
    filter.deny(linux.SECCOMP.RET.ERRNO | @as(u32, @backingInt(linux.E.PERM)));
    try filter.install();
}

/// Apply after trusted static-ELF startup and before generated.run. Read only
/// stdin; write only stdout/stderr. No files, sockets, heap mappings or children.
/// Even raw syscalls obtained through guest.std must pass this kernel filter.
pub fn enterGuest() Error!void {
    try supported();
    try closeInheritedDescriptors();
    try noNewPrivileges();
    var filter: Filter = .{};
    filter.begin();
    filter.descriptorCall(.read, &.{0});
    filter.descriptorCall(.write, &.{ 1, 2 });
    filter.allow(.exit);
    filter.allow(.exit_group);
    filter.allow(.rt_sigreturn);
    filter.allow(.rt_sigprocmask);
    filter.deny(linux.SECCOMP.RET.KILL_PROCESS);
    try filter.install();
}

fn supported() Error!void {
    if (builtin.os.tag != .linux) return error.UnsupportedPlatform;
    switch (builtin.cpu.arch) {
        .x86_64, .aarch64 => {},
        else => return error.UnsupportedPlatform,
    }
    switch (builtin.mode) {
        .Debug, .ReleaseSafe => {},
        .ReleaseFast, .ReleaseSmall => return error.UnsafeBuildMode,
    }
}

fn closeInheritedDescriptors() Error!void {
    const result = linux.syscall3(.close_range, 3, std.math.maxInt(u32), 0);
    if (result != 0) return error.DescriptorCleanupFailed;
}

fn noNewPrivileges() Error!void {
    if (linux.prctl(@backingInt(linux.PR.SET_NO_NEW_PRIVS), 1, 0, 0, 0) != 0)
        return error.NoNewPrivilegesFailed;
}

fn validatePath(path: []const u8, prefix: []const u8) Error!void {
    if (path.len > 4095) return error.InvalidPolicyPath;
    if (!std.mem.startsWith(u8, path, prefix)) return error.InvalidPolicyPath;
    if (path.len == prefix.len) return error.InvalidPolicyPath;
    var components = std.mem.splitScalar(u8, path, '/');
    while (components.next()) |part| {
        if (std.mem.eql(u8, part, ".")) return error.InvalidPolicyPath;
        if (std.mem.eql(u8, part, "..")) return error.InvalidPolicyPath;
    }
}

fn compilerFilesystem(roots: []const [:0]const u8, workspace: [:0]const u8) Error!void {
    const abi = linux.syscall3(.landlock_create_ruleset, 0, 0, 1);
    if (abi < 7 or abi > 1024) return error.LandlockUnavailable;
    const attributes: Ruleset = .{ .filesystem = filesystem_all, .network = 3, .scopes = 3 };
    const descriptor = linux.syscall3(.landlock_create_ruleset, @intFromPtr(&attributes), @sizeOf(Ruleset), 0);
    if (descriptor > std.math.maxInt(i32)) return error.LandlockUnavailable;
    defer _ = linux.syscall1(.close, descriptor);
    for (roots) |path| try addPath(descriptor, path, read_access);
    try addPath(descriptor, workspace, filesystem_all);
    try addPath(descriptor, "/dev/null", (1 << 1) | (1 << 2));
    if (linux.syscall2(.landlock_restrict_self, descriptor, 0) != 0)
        return error.LandlockUnavailable;
}

fn addPath(ruleset: usize, path: [:0]const u8, access: u64) Error!void {
    const opened = linux.openat(linux.AT.FDCWD, path, .{ .PATH = true, .CLOEXEC = true }, 0);
    if (opened > std.math.maxInt(i32)) return error.LandlockRuleFailed;
    defer _ = linux.syscall1(.close, opened);
    var metadata: linux.Statx = undefined;
    if (linux.statx(@intCast(opened), "", linux.AT.EMPTY_PATH, .{ .TYPE = true }, &metadata) != 0)
        return error.LandlockRuleFailed;
    if (!metadata.mask.TYPE) return error.LandlockRuleFailed;
    const directory = metadata.mode & linux.S.IFMT == linux.S.IFDIR;
    const rule: PathRule = .{ .access = if (directory) access else access & file_access, .descriptor = @intCast(opened) };
    if (linux.syscall4(.landlock_add_rule, ruleset, 1, @intFromPtr(&rule), 0) != 0)
        return error.LandlockRuleFailed;
}

const Filter = struct {
    instructions: [instruction_max]Instruction = undefined,
    length: u16 = 0,

    fn append(filter: *Filter, instruction: Instruction) void {
        assert(filter.length < instruction_max);
        filter.instructions[filter.length] = instruction;
        filter.length += 1;
        assert(filter.length <= instruction_max);
    }

    fn begin(filter: *Filter) void {
        assert(filter.length == 0);
        const architecture: u32 = switch (builtin.cpu.arch) {
            .x86_64 => 0xc000003e,
            .aarch64 => 0xc00000b7,
            else => unreachable,
        };
        filter.append(.{ .code = 0x20, .value = @offsetOf(linux.SECCOMP.data, "arch") });
        filter.append(.{ .code = 0x15, .yes = 1, .value = architecture });
        filter.deny(linux.SECCOMP.RET.KILL_PROCESS);
        filter.append(.{ .code = 0x20, .value = @offsetOf(linux.SECCOMP.data, "nr") });
    }

    fn allow(filter: *Filter, syscall: linux.SYS) void {
        filter.append(.{ .code = 0x15, .no = 1, .value = @intCast(@backingInt(syscall)) });
        filter.deny(linux.SECCOMP.RET.ALLOW);
    }

    fn deny(filter: *Filter, action: u32) void {
        filter.append(.{ .code = 0x06, .value = action });
    }

    fn descriptorCall(filter: *Filter, syscall: linux.SYS, comptime descriptors: []const u32) void {
        assert(descriptors.len > 0);
        assert(descriptors.len <= 2);
        const body_length = 5 + descriptors.len * 2;
        filter.append(.{ .code = 0x15, .no = @intCast(body_length), .value = @intCast(@backingInt(syscall)) });
        filter.append(.{ .code = 0x20, .value = @offsetOf(linux.SECCOMP.data, "arg0") + 4 });
        filter.append(.{ .code = 0x15, .yes = 1, .value = 0 });
        filter.deny(linux.SECCOMP.RET.KILL_PROCESS);
        filter.append(.{ .code = 0x20, .value = @offsetOf(linux.SECCOMP.data, "arg0") });
        for (descriptors) |descriptor| {
            filter.append(.{ .code = 0x15, .no = 1, .value = descriptor });
            filter.deny(linux.SECCOMP.RET.ALLOW);
        }
        filter.deny(linux.SECCOMP.RET.KILL_PROCESS);
    }

    fn compilerCalls(filter: *Filter) void {
        const names = .{
            "read",          "write",           "readv",          "writev",       "pread64",           "pwrite64",     "preadv",          "pwritev",
            "lseek",         "close",           "flock",          "fchmod",       "openat",            "newfstatat",   "fstat",           "statx",
            "getdents64",    "readlink",        "readlinkat",     "access",       "faccessat",         "faccessat2",   "mkdir",           "mkdirat",
            "unlink",        "unlinkat",        "rename",         "renameat",     "renameat2",         "ftruncate",    "fallocate",       "fsync",
            "fdatasync",     "fcntl",           "dup",            "dup2",         "dup3",              "pipe",         "pipe2",           "poll",
            "ppoll",         "pselect6",        "epoll_create1",  "epoll_ctl",    "epoll_pwait",       "epoll_pwait2", "sendfile",        "copy_file_range",
            "clock_gettime", "clock_nanosleep", "nanosleep",      "futex",        "futex_waitv",       "mmap",         "mprotect",        "munmap",
            "mremap",        "madvise",         "brk",            "getrandom",    "sched_getaffinity", "sched_yield",  "prlimit64",       "getrlimit",
            "setrlimit",     "rt_sigaction",    "rt_sigprocmask", "rt_sigreturn", "sigaltstack",       "getpid",       "gettid",          "getppid",
            "getuid",        "geteuid",         "getgid",         "getegid",      "getresuid",         "getresgid",    "set_tid_address", "set_robust_list",
            "rseq",          "uname",           "arch_prctl",     "execve",       "execveat",          "wait4",        "waitid",          "exit",
            "exit_group",    "getcwd",          "chdir",          "fchdir",       "umask",             "getrusage",    "times",           "sysinfo",
        };
        inline for (names) |name| {
            if (@hasField(linux.SYS, name)) filter.allow(@field(linux.SYS, name));
        }
        filter.cloneCall();
        if (@hasField(linux.SYS, "clone3")) {
            filter.append(.{ .code = 0x15, .no = 1, .value = @backingInt(linux.SYS.clone3) });
            filter.deny(linux.SECCOMP.RET.ERRNO | @as(u32, @backingInt(linux.E.NOSYS)));
        }
    }

    fn cloneCall(filter: *Filter) void {
        filter.append(.{ .code = 0x15, .no = 7, .value = @backingInt(linux.SYS.clone) });
        filter.append(.{ .code = 0x20, .value = @offsetOf(linux.SECCOMP.data, "arg0") + 4 });
        filter.append(.{ .code = 0x15, .yes = 1, .value = 0 });
        filter.deny(linux.SECCOMP.RET.ERRNO | @as(u32, @backingInt(linux.E.PERM)));
        filter.append(.{ .code = 0x20, .value = @offsetOf(linux.SECCOMP.data, "arg0") });
        // All CLONE_NEW* flags plus CLONE_PARENT; descendants must stay in lane.
        filter.append(.{ .code = 0x45, .no = 1, .value = 0x7e028080 });
        filter.deny(linux.SECCOMP.RET.ERRNO | @as(u32, @backingInt(linux.E.PERM)));
        filter.deny(linux.SECCOMP.RET.ALLOW);
    }

    fn install(filter: *const Filter) Error!void {
        assert(filter.length > 4);
        assert(filter.length <= instruction_max);
        const program: Program = .{ .length = filter.length, .instructions = &filter.instructions };
        if (linux.seccomp(linux.SECCOMP.SET_MODE_FILTER, linux.SECCOMP.FILTER_FLAG.TSYNC, &program) != 0) return error.SeccompUnavailable;
    }
};

comptime {
    assert(@sizeOf(Instruction) == 8);
    assert(@offsetOf(PathRule, "descriptor") == 8);
    assert(instruction_max < linux.BPF.MAXINSNS);
}
