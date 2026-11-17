//! Startup-owned, empty cgroup-v2 lanes. The executor stays outside the lane.
//! Empty membership is NOT wait/reap: the supervisor must also join its direct
//! PID-namespace init child, which reaps that namespace's descendants on exit.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

pub const Error = error{
    UnsupportedPlatform,
    InvalidLimits,
    InvalidCgroup,
    CgroupBusy,
    CgroupReadFailed,
    CgroupWriteFailed,
    CgroupControlTooLarge,
    CgroupLimitsUnavailable,
    InvalidProcess,
};

pub const Limits = struct {
    memory_bytes: u64 = 1024 * 1024 * 1024,
    processes: u16 = 32,
};

pub const Lane = struct {
    descriptor: i32,
    limits: Limits,

    /// The descriptor is borrowed for the lane's complete lifetime. The caller
    /// creates and owns this EMPTY delegated domain; no existing workload may
    /// be passed here. Readback and filesystem identity fail closed.
    pub fn configure(descriptor: i32, limits: Limits) Error!Lane {
        if (builtin.os.tag != .linux) return error.UnsupportedPlatform;
        switch (builtin.cpu.arch) {
            .aarch64, .x86_64 => {},
            else => return error.UnsupportedPlatform,
        }
        if (descriptor < 3) return error.InvalidCgroup;
        if (limits.memory_bytes < 16 * 1024 * 1024) return error.InvalidLimits;
        if (limits.memory_bytes > 4 * 1024 * 1024 * 1024) return error.InvalidLimits;
        if (limits.processes < 2 or limits.processes > 64) return error.InvalidLimits;
        var filesystem: [32]usize = @splat(0);
        if (linux.syscall2(.fstatfs, @intCast(descriptor), @intFromPtr(&filesystem)) != 0)
            return error.InvalidCgroup;
        if (filesystem[0] != 0x63677270) return error.InvalidCgroup;
        var lane: Lane = .{ .descriptor = descriptor, .limits = limits };
        var buffer: [256]u8 = undefined;
        const kind = try lane.readControl("cgroup.type", &buffer);
        if (!std.mem.eql(u8, kind, "domain\n")) return error.InvalidCgroup;
        if (!try lane.empty()) return error.CgroupBusy;
        try lane.setNumber("memory.max", limits.memory_bytes);
        // The kernel rounds to its page size (which need not be 4 KiB). Retain
        // the effective stricter bound, never silently accept a larger budget.
        const effective_memory = try lane.number("memory.max");
        if (effective_memory > limits.memory_bytes or effective_memory < 16 * 1024 * 1024)
            return error.CgroupLimitsUnavailable;
        lane.limits.memory_bytes = effective_memory;
        try lane.setNumber("memory.swap.max", 0);
        try lane.setNumber("memory.oom.group", 1);
        try lane.setNumber("pids.max", limits.processes);
        try lane.verify();
        return lane;
    }

    /// Reverify immediately before launch; never silently inherit relaxed limits.
    pub fn verify(lane: *const Lane) Error!void {
        if (try lane.number("memory.max") != lane.limits.memory_bytes)
            return error.CgroupLimitsUnavailable;
        if (try lane.number("memory.swap.max") != 0) return error.CgroupLimitsUnavailable;
        if (try lane.number("memory.oom.group") != 1) return error.CgroupLimitsUnavailable;
        if (try lane.number("pids.max") != lane.limits.processes)
            return error.CgroupLimitsUnavailable;
    }

    /// Attach the blocked child BEFORE acknowledging its startup pipe. The child
    /// must not execute generated work until mapping and membership both succeed.
    pub fn attach(lane: *const Lane, process: i32) Error!void {
        if (process <= 0) return error.InvalidProcess;
        try lane.setNumber("cgroup.procs", @intCast(process));
    }

    /// Atomic recursive kernel kill. The caller still waits/reaps PID1 and then
    /// observes empty(); returning from kill() alone never releases the lane.
    pub fn kill(lane: *const Lane) Error!void {
        try lane.writeControl("cgroup.kill", "1");
    }

    pub fn empty(lane: *const Lane) Error!bool {
        var buffer: [256]u8 = undefined;
        const contents = try lane.readControl("cgroup.events", &buffer);
        var lines = std.mem.splitScalar(u8, contents, '\n');
        var populated: ?bool = null;
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "populated ")) continue;
            if (populated != null) return error.InvalidCgroup;
            const value = line[10..];
            if (std.mem.eql(u8, value, "0")) {
                populated = false;
            } else if (std.mem.eql(u8, value, "1")) {
                populated = true;
            } else return error.InvalidCgroup;
        }
        return !(populated orelse return error.InvalidCgroup);
    }

    fn number(lane: *const Lane, name: [:0]const u8) Error!u64 {
        var buffer: [256]u8 = undefined;
        const contents = try lane.readControl(name, &buffer);
        const digits = std.mem.trim(u8, contents, "\n");
        if (digits.len == 0 or digits.len > 20) return error.InvalidCgroup;
        return std.fmt.parseInt(u64, digits, 10) catch error.InvalidCgroup;
    }

    fn setNumber(lane: *const Lane, name: [:0]const u8, value: u64) Error!void {
        var buffer: [32]u8 = undefined;
        const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable;
        std.debug.assert(text.len > 0);
        std.debug.assert(text.len <= 20);
        try lane.writeControl(name, text);
    }

    fn readControl(lane: *const Lane, name: [:0]const u8, buffer: *[256]u8) Error![]const u8 {
        const descriptor = linux.openat(lane.descriptor, name, .{ .CLOEXEC = true, .NOFOLLOW = true }, 0);
        if (descriptor > std.math.maxInt(i32)) return error.CgroupReadFailed;
        defer _ = linux.syscall1(.close, descriptor);
        var length: u16 = 0;
        var interruptions: u8 = 0;
        while (length < buffer.len) {
            const count = linux.syscall3(.read, descriptor, @intFromPtr(buffer[length..].ptr), buffer.len - length);
            if (count == 0 -% @as(usize, @backingInt(linux.E.INTR))) {
                if (interruptions == 64) return error.CgroupReadFailed;
                interruptions += 1;
                continue;
            }
            if (count > buffer.len - length) return error.CgroupReadFailed;
            if (count == 0) return buffer[0..length];
            length += @intCast(count);
        }
        return error.CgroupControlTooLarge;
    }

    fn writeControl(lane: *const Lane, name: [:0]const u8, contents: []const u8) Error!void {
        std.debug.assert(contents.len > 0);
        std.debug.assert(contents.len <= 32);
        const descriptor = linux.openat(lane.descriptor, name, .{ .ACCMODE = .WRONLY, .CLOEXEC = true, .NOFOLLOW = true }, 0);
        if (descriptor > std.math.maxInt(i32)) return error.CgroupWriteFailed;
        defer _ = linux.syscall1(.close, descriptor);
        var interruptions: u8 = 0;
        while (interruptions <= 64) : (interruptions += 1) {
            const count = linux.syscall3(.write, descriptor, @intFromPtr(contents.ptr), contents.len);
            if (count == 0 -% @as(usize, @backingInt(linux.E.INTR))) continue;
            if (count != contents.len) return error.CgroupWriteFailed;
            return;
        }
        return error.CgroupWriteFailed;
    }
};
