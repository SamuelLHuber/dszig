//! The direct child is PID1 of a fresh user/PID/network/mount namespace.
//! Parent mapping and cgroup attachment complete BEFORE the startup gate opens.
//! Waiting that direct PID1 is essential: Linux tears down and reaps the PID
//! namespace's remaining descendants before its init completes exit.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const cgroup = @import("generated_zig_cgroup_linux.zig");

pub const Error = cgroup.Error || error{
    InvalidChildState,
    StartupPipeFailed,
    NamespaceCloneFailed,
    NamespaceMappingFailed,
    StartupGateFailed,
    ChildWaitFailed,
    ChildKillFailed,
    FilesystemSetupFailed,
    InvalidWorkspace,
};

pub const Exit = struct {
    code: u8,
    signal: u8,

    pub fn success(exit: Exit) bool {
        return exit.code == 0 and exit.signal == 0;
    }
};

pub const Child = struct {
    process: ?i32 = null,
    exit: ?Exit = null,

    /// Call only with a fresh empty lane and stable inherited context. On ANY
    /// error, process may remain live: caller MUST kill/join before releasing
    /// this child/lane. Keeping the PID in the out-parameter makes that explicit.
    pub fn spawn(child: *Child, lane: *const cgroup.Lane, context: *anyopaque, entry: *const fn (*anyopaque) u8) Error!void {
        if (builtin.os.tag != .linux) return error.UnsupportedPlatform;
        if (child.process != null or child.exit != null) return error.InvalidChildState;
        try lane.verify();
        if (!try lane.empty()) return error.CgroupBusy;
        var gate: [2]i32 = undefined;
        if (linux.pipe2(&gate, .{ .CLOEXEC = true }) != 0) return error.StartupPipeFailed;
        defer _ = linux.syscall1(.close, @intCast(gate[0]));
        defer _ = linux.syscall1(.close, @intCast(gate[1]));
        const uid = linux.syscall0(.getuid);
        const gid = linux.syscall0(.getgid);
        // SIGCHLD | NEWUSER | NEWPID | NEWNS | NEWNET. No shared VM or descriptors.
        const result = linux.syscall5(.clone, 17 | 0x10000000 | 0x20000000 |
            0x00020000 | 0x40000000, 0, 0, 0, 0);
        if (result == 0) childMain(gate, context, entry);
        if (result > std.math.maxInt(i32)) return error.NamespaceCloneFailed;
        child.process = @intCast(result);
        try mapIdentity(@intCast(result), uid, gid);
        try lane.attach(@intCast(result));
        const acknowledgment = [_]u8{0xa5};
        if (linux.syscall3(.write, @intCast(gate[1]), @intFromPtr(&acknowledgment), 1) != 1)
            return error.StartupGateFailed;
    }

    pub fn tryJoin(child: *Child) Error!?Exit {
        if (child.exit) |exit| return exit;
        const process = child.process orelse return error.InvalidChildState;
        var status: i32 = 0;
        const result = linux.wait4(process, &status, linux.W.NOHANG, null);
        if (result == 0) return null;
        if (result == 0 -% @as(usize, @backingInt(linux.E.INTR))) return null;
        if (result != @as(usize, @intCast(process))) return error.ChildWaitFailed;
        return child.reaped(status);
    }

    pub fn join(child: *Child) Error!Exit {
        if (child.exit) |exit| return exit;
        const process = child.process orelse return error.InvalidChildState;
        var status: i32 = 0;
        var interruptions: u8 = 0;
        while (interruptions <= 64) : (interruptions += 1) {
            const result = linux.wait4(process, &status, 0, null);
            if (result == 0 -% @as(usize, @backingInt(linux.E.INTR))) continue;
            if (result != @as(usize, @intCast(process))) return error.ChildWaitFailed;
            return child.reaped(status);
        }
        return error.ChildWaitFailed;
    }

    /// Also handles failure before attachment: the startup-gated child can be
    /// outside the lane until attach succeeds. Never discard its PID first.
    pub fn kill(child: *const Child, lane: *const cgroup.Lane) Error!void {
        const lane_result = lane.kill();
        if (child.process) |process| {
            const result = linux.kill(process, .KILL);
            if (result != 0 and result != 0 -% @as(usize, @backingInt(linux.E.SRCH)))
                return error.ChildKillFailed;
        }
        try lane_result;
    }

    fn reaped(child: *Child, status: i32) Exit {
        const bits: u32 = @bitCast(status);
        const exit: Exit = .{ .code = @intCast((bits >> 8) & 255), .signal = @intCast(bits & 127) };
        child.exit = exit;
        child.process = null;
        return exit;
    }
};

/// Must run in the namespace PID1, before any compiler or generated work.
/// A fresh tmpfs bounds aggregate compiler artifacts, not just individual files.
pub fn privateFilesystem(workspace: [:0]const u8, bytes: u32) Error!void {
    if (!std.mem.startsWith(u8, workspace, "/tmp/dszig-")) return error.InvalidWorkspace;
    if (workspace.len > 4095 or std.mem.indexOf(u8, workspace, "..") != null)
        return error.InvalidWorkspace;
    if (bytes < 1024 * 1024 or bytes > 256 * 1024 * 1024) return error.InvalidWorkspace;
    if (linux.mount(null, "/", null, linux.MS.PRIVATE | linux.MS.REC, 0) != 0)
        return error.FilesystemSetupFailed;
    var buffer: [64]u8 = undefined;
    const options = std.mem.printSentinel(&buffer, "size={d},nr_inodes=8192,mode=0700", .{bytes}, 0) catch unreachable;
    if (linux.mount("tmpfs", workspace, "tmpfs", linux.MS.NOSUID | linux.MS.NODEV, @intFromPtr(options.ptr)) != 0) return error.FilesystemSetupFailed;
    if (linux.mount("proc", "/proc", "proc", linux.MS.NOSUID | linux.MS.NODEV |
        linux.MS.NOEXEC, 0) != 0) return error.FilesystemSetupFailed;
    if (linux.chdir(workspace) != 0) return error.FilesystemSetupFailed;
}

fn childMain(gate: [2]i32, context: *anyopaque, entry: *const fn (*anyopaque) u8) noreturn {
    _ = linux.syscall1(.close, @intCast(gate[1]));
    var acknowledgment: [1]u8 = undefined;
    const count = linux.syscall3(.read, @intCast(gate[0]), @intFromPtr(&acknowledgment), 1);
    if (count != 1 or acknowledgment[0] != 0xa5) linux.exit_group(125);
    _ = linux.syscall1(.close, @intCast(gate[0]));
    linux.exit_group(entry(context));
}

fn mapIdentity(process: i32, uid: usize, gid: usize) Error!void {
    var path_buffer: [64]u8 = undefined;
    const setgroups = std.mem.printSentinel(&path_buffer, "/proc/{d}/setgroups", .{process}, 0) catch unreachable;
    try writeMapping(setgroups, "deny");
    var map_buffer: [64]u8 = undefined;
    const uid_text = std.fmt.bufPrint(&map_buffer, "0 {d} 1\n", .{uid}) catch unreachable;
    const uid_path = std.mem.printSentinel(&path_buffer, "/proc/{d}/uid_map", .{process}, 0) catch unreachable;
    try writeMapping(uid_path, uid_text);
    const gid_text = std.fmt.bufPrint(&map_buffer, "0 {d} 1\n", .{gid}) catch unreachable;
    const gid_path = std.mem.printSentinel(&path_buffer, "/proc/{d}/gid_map", .{process}, 0) catch unreachable;
    try writeMapping(gid_path, gid_text);
}

fn writeMapping(path: [:0]const u8, contents: []const u8) Error!void {
    const descriptor = linux.openat(linux.AT.FDCWD, path, .{ .ACCMODE = .WRONLY, .CLOEXEC = true, .NOFOLLOW = true }, 0);
    if (descriptor > std.math.maxInt(i32)) return error.NamespaceMappingFailed;
    defer _ = linux.syscall1(.close, descriptor);
    if (linux.syscall3(.write, descriptor, @intFromPtr(contents.ptr), contents.len) != contents.len)
        return error.NamespaceMappingFailed;
}
