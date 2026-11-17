//! Host-side supervisor for one isolated generated-Zig run at a time.
//! Construct in place at startup; every buffer is fixed and no method allocates.
//! The executor stays OUTSIDE the cgroup lane. Output is published only after
//! the direct namespace PID1 is reaped, the lane is empty, and both the protocol
//! and diagnostic streams reached EOF. Any other outcome aborts, kills and joins.
//! If that join cannot be proved the executor quarantines itself, keeps the Child
//! handle, and refuses every later run.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const assert = std.debug.assert;
const cgroup = @import("generated_zig_cgroup_linux.zig");
const namespace = @import("generated_zig_namespace_linux.zig");
const protocol = @import("generated_zig_protocol.zig");
const session = @import("generated_zig_session.zig");
const source = @import("generated_zig_source.zig");
const worker = @import("generated_zig_worker_linux.zig");

pub const timeout_ms_max: u32 = 120_000;
pub const calls_max: u32 = 256;
pub const frame_bytes_max: u64 = 8 * 1024 * 1024;
pub const work_bytes_max: u64 = 16 * 1024 * 1024;
pub const roots_max: u16 = 256;
pub const diagnostic_bytes_max: u32 = 16 * 1024;

const path_bytes_max: u16 = 4095;
/// Post-kill proof budget: SIGKILL plus cgroup.kill cannot be refused, so a child
/// that is still unreaped after this many 1ms pauses means the kernel state is
/// unknown. That is quarantine, never a silent release.
const cleanup_attempts_max: u32 = 5000;
const settle_attempts_max: u8 = 64;
const interruptions_max: u8 = 64;
const pause_ns: u64 = 1_000_000;

pub const CapabilityError = error{ToolFailed};

pub const Capability = struct {
    name: []const u8,
    context: *anyopaque,
    /// Trusted, bounded host code. Writes one JSON value into the fixed buffer and
    /// returns its length. The u64 is an ABSOLUTE CLOCK_MONOTONIC deadline in
    /// nanoseconds which the callback MUST honor; the kernel guard protects the
    /// host from generated code, not from a registered callback.
    call: *const fn (*anyopaque, []const u8, []u8, u64) CapabilityError!u32,
};

pub const Options = struct {
    lane: *const cgroup.Lane,
    worker: worker.Config,
    tools: []const Capability = &.{},
    timeout_ms: u32 = 60_000,
    max_calls: u32 = 64,
    max_frame_bytes: u64 = 4 * 1024 * 1024,
    max_work_bytes: u64 = 8 * 1024 * 1024,
};

pub const Error = session.Error || namespace.Error || CapabilityError || error{
    UnsafeBuildMode,
    InvalidExecutorOptions,
    InvalidExecutorPath,
    ExecutorBusy,
    ExecutorQuarantined,
    SourceTooLarge,
    InvalidSourceByte,
    ClockFailed,
    TransportSetupFailed,
    PollFailed,
    ReceiveFailed,
    SendFailed,
    PartialFrame,
    DiagnosticsOverflow,
    DeadlineExceeded,
    WorkLoopExceeded,
    ToolReplyTooLarge,
};

const State = enum { idle, running, quarantined };
const Timespec = extern struct { seconds: isize, nanoseconds: isize };
const PollDescriptor = extern struct { descriptor: i32, events: i16, observed: i16 };
const poll_in: i16 = 0x001;
const poll_out: i16 = 0x004;

pub const Executor = struct {
    options: Options,
    state: State,
    child: namespace.Child,
    context: worker.Context,
    socket: i32,
    diagnostic_descriptor: i32,
    diagnostic_bytes: u32,
    names: [session.tools_max][]const u8,
    session: session.Session,
    source_buffer: [source.source_bytes_max + 1]u8,
    scratch: [source.scratch_bytes_recommended]u8,
    receive_buffer: [protocol.payload_bytes_max]u8,
    reply_buffer: [protocol.payload_bytes_max]u8,
    diagnostic_buffer: [diagnostic_bytes_max]u8,

    /// Call exactly once on fresh storage. Options, tool slices and tool contexts
    /// are borrowed for the executor's whole lifetime. Reinitializing a running or
    /// quarantined executor would discard a live Child handle: never do that.
    pub fn init(executor: *Executor, options: *const Options) Error!void {
        // Fields are assigned individually: a whole-struct literal would put the
        // megabyte-scale fixed buffers on the caller's stack.
        executor.state = .quarantined;
        executor.child = .{};
        executor.socket = -1;
        executor.diagnostic_descriptor = -1;
        executor.diagnostic_bytes = 0;
        executor.session.state = .failed;
        executor.session.finish_bytes = 0;
        try supported();
        if (options.timeout_ms == 0) return error.InvalidExecutorOptions;
        if (options.timeout_ms > timeout_ms_max) return error.InvalidExecutorOptions;
        if (options.max_calls > calls_max) return error.InvalidExecutorOptions;
        if (options.max_frame_bytes < protocol.header_bytes) return error.InvalidExecutorOptions;
        if (options.max_frame_bytes > frame_bytes_max) return error.InvalidExecutorOptions;
        if (options.max_work_bytes == 0) return error.InvalidExecutorOptions;
        if (options.max_work_bytes > work_bytes_max) return error.InvalidExecutorOptions;
        if (options.tools.len > session.tools_max) return error.InvalidExecutorOptions;
        if (options.worker.read_only_roots.len == 0) return error.InvalidExecutorOptions;
        if (options.worker.read_only_roots.len > roots_max) return error.InvalidExecutorOptions;
        if (options.worker.artifact_bytes < 1024 * 1024) return error.InvalidExecutorOptions;
        if (options.worker.artifact_bytes > 256 * 1024 * 1024) return error.InvalidExecutorOptions;
        try validateWorkspace(options.worker.workspace);
        try verifyWorkspace(options.worker.workspace);
        try validateStorePath(options.worker.compiler);
        for (options.worker.read_only_roots) |root| try validateStorePath(root);
        executor.options = options.*;
        for (options.tools, 0..) |tool, index| executor.names[index] = tool.name;
        // Rejects empty, oversize and duplicate names now instead of at first run.
        try executor.session.init(&executor.sessionOptions());
        executor.session.abort();
        executor.state = .idle;
    }

    /// Returned JSON borrows Session storage until the next run. `input_json` must
    /// stay stable for the whole call: it is transmitted without copying.
    pub fn run(executor: *Executor, source_text: []const u8, input_json: []const u8) Error![]const u8 {
        switch (executor.state) {
            .idle => {},
            .running => return error.ExecutorBusy,
            .quarantined => return error.ExecutorQuarantined,
        }
        executor.state = .running;
        executor.diagnostic_bytes = 0;
        // Revoke the previous run's output before any buffer is reused.
        executor.session.abort();
        const exit = executor.supervise(source_text, input_json) catch |failure| {
            executor.cleanup();
            if (executor.state == .quarantined) return error.ExecutorQuarantined;
            return failure;
        };
        // supervise() returns only with PID1 reaped, the lane empty, both streams
        // at EOF and both parent descriptors closed.
        assert(executor.child.process == null);
        assert(executor.socket == -1);
        assert(executor.diagnostic_descriptor == -1);
        executor.child = .{};
        executor.state = .idle;
        return executor.session.joined(exit.success());
    }

    /// Child stderr observed during the latest run, capped at diagnostic_bytes_max.
    pub fn diagnostics(executor: *const Executor) []const u8 {
        return executor.diagnostic_buffer[0..executor.diagnostic_bytes];
    }

    fn sessionOptions(executor: *const Executor) session.Options {
        return .{
            .tools = executor.names[0..executor.options.tools.len],
            .max_calls = executor.options.max_calls,
            .max_frame_bytes = executor.options.max_frame_bytes,
            .max_work_bytes = executor.options.max_work_bytes,
        };
    }

    fn supervise(executor: *Executor, source_text: []const u8, input_json: []const u8) Error!namespace.Exit {
        assert(executor.child.process == null);
        assert(executor.child.exit == null);
        const started = try monotonic();
        const deadline = started + @as(u64, executor.options.timeout_ms) * 1_000_000;
        if (source_text.len > source.source_bytes_max) return error.SourceTooLarge;
        if (std.mem.indexOfScalar(u8, source_text, 0) != null) return error.InvalidSourceByte;
        const lane = executor.options.lane;
        try lane.verify();
        if (!try lane.empty()) return error.CgroupBusy;
        try verifyWorkspace(executor.options.worker.workspace);
        @memcpy(executor.source_buffer[0..source_text.len], source_text);
        executor.source_buffer[source_text.len] = 0;
        try executor.session.init(&executor.sessionOptions());
        try executor.session.begin(input_json);
        if (try monotonic() >= deadline) return error.DeadlineExceeded;

        var sockets: [2]i32 = undefined;
        // AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC.
        if (linux.syscall4(.socketpair, 1, 1 | 0o4000 | 0o2000000, 0, @intFromPtr(&sockets)) != 0)
            return error.TransportSetupFailed;
        var pipe: [2]i32 = undefined;
        if (linux.pipe2(&pipe, .{ .CLOEXEC = true, .NONBLOCK = true }) != 0) {
            close(sockets[0]);
            close(sockets[1]);
            return error.TransportSetupFailed;
        }
        executor.socket = sockets[0];
        executor.diagnostic_descriptor = pipe[0];
        executor.context = .{
            .config = executor.options.worker,
            .source = executor.source_buffer[0..source_text.len :0],
            .scratch = &executor.scratch,
            .socket_child = sockets[1],
            .socket_parent = sockets[0],
            .diagnostics_read = pipe[0],
            .diagnostics_write = pipe[1],
        };
        const spawned = executor.child.spawn(lane, @ptrCast(&executor.context), worker.entry);
        // Child endpoints close on success AND error; otherwise EOF never arrives.
        close(sockets[1]);
        close(pipe[1]);
        try spawned;

        const peer_reset = try executor.transport(input_json, deadline);

        // Streams are at EOF, yet PID1 may still be exiting. Bounded by the deadline.
        const exit = while (true) {
            if (try executor.child.tryJoin()) |reaped| break reaped;
            const now = try monotonic();
            if (now >= deadline) return error.DeadlineExceeded;
            pause(@min(pause_ns, deadline - now));
        };
        var attempts: u8 = 0;
        while (!try lane.empty()) : (attempts += 1) {
            if (attempts == settle_attempts_max) return error.CgroupBusy;
            pause(pause_ns);
        }
        executor.closeDescriptors();
        // Closing an unread socket can reset the peer. Preserve an authoritative
        // failed worker exit, but never publish a zero exit with an unclean stream.
        if (peer_reset and exit.success()) return error.ReceiveFailed;
        return exit;
    }

    /// Runs until BOTH streams reach EOF. A finish frame stays pending inside the
    /// session; nothing here publishes it.
    fn transport(executor: *Executor, input_json: []const u8, deadline: u64) Error!bool {
        var header: [protocol.header_bytes]u8 = undefined;
        var header_have: u32 = 0;
        var frame: ?protocol.Header = null;
        var payload_have: u32 = 0;
        var finished = false;
        var protocol_open = true;
        var diagnostics_open = true;
        var peer_reset = false;

        // Header plus borrowed payload: neither input nor reply is copied to send.
        var send_header = try (protocol.Header{ .kind = .input, .payload_bytes = @intCast(input_json.len) }).encode();
        var send_payload: []const u8 = input_json;
        var send_done: u32 = 0;
        var sending = true;

        // Every productive wake moves at least one byte, so this bounds the loop
        // even against byte-at-a-time peers; the deadline bounds everything else.
        const wakes_max: u64 = executor.options.max_frame_bytes * 2 + diagnostic_bytes_max + 4096;
        var wakes: u64 = 0;
        while (protocol_open or diagnostics_open) : (wakes += 1) {
            if (wakes == wakes_max) return error.WorkLoopExceeded;
            const now = try monotonic();
            if (now >= deadline) return error.DeadlineExceeded;
            var polled = [2]PollDescriptor{
                .{ .descriptor = -1, .events = 0, .observed = 0 },
                .{ .descriptor = -1, .events = 0, .observed = 0 },
            };
            if (sending) {
                polled[0] = .{ .descriptor = executor.socket, .events = poll_out, .observed = 0 };
            } else if (protocol_open) {
                polled[0] = .{ .descriptor = executor.socket, .events = poll_in, .observed = 0 };
            }
            if (diagnostics_open) {
                polled[1] = .{ .descriptor = executor.diagnostic_descriptor, .events = poll_in, .observed = 0 };
            }
            const left = deadline - now;
            var wait: Timespec = .{ .seconds = @intCast(left / 1_000_000_000), .nanoseconds = @intCast(left % 1_000_000_000) };
            // ppoll exists on both supported architectures; aarch64 has no poll.
            const ready = linux.syscall5(.ppoll, @intFromPtr(&polled), polled.len, @intFromPtr(&wait), 0, 8);
            if (failed(ready)) {
                if (errno(ready) == .INTR) continue;
                return error.PollFailed;
            }
            if (ready == 0) continue;

            if (polled[1].observed != 0) {
                if (try executor.receiveDiagnostics()) diagnostics_open = false;
            }
            if (polled[0].observed == 0) continue;

            if (sending) {
                // Receiving is paused: a later frame could otherwise reuse the
                // reply buffer while it is still queued.
                const total: u32 = protocol.header_bytes + @as(u32, @intCast(send_payload.len));
                while (send_done < total) {
                    if (try monotonic() >= deadline) return error.DeadlineExceeded;
                    const rest: []const u8 = if (send_done < protocol.header_bytes)
                        send_header[send_done..]
                    else
                        send_payload[send_done - protocol.header_bytes ..];
                    // MSG_NOSIGNAL | MSG_DONTWAIT: a dead peer is EPIPE, never SIGPIPE.
                    const sent = linux.syscall6(.sendto, @intCast(executor.socket), @intFromPtr(rest.ptr), rest.len, 0x4000 | 0x40, 0, 0);
                    if (failed(sent)) {
                        switch (errno(sent)) {
                            .AGAIN, .INTR => break,
                            .PIPE, .CONNRESET => {
                                peer_reset = true;
                                protocol_open = false;
                                sending = false;
                                break;
                            },
                            else => return error.SendFailed,
                        }
                    }
                    if (sent == 0 or sent > rest.len) return error.SendFailed;
                    send_done += @intCast(sent);
                }
                if (send_done == total) sending = false;
                continue;
            }

            // One read per wake; the socket is level-triggered.
            if (frame == null) {
                const count = (try receive(executor.socket, header[header_have..], &peer_reset)) orelse continue;
                if (count == 0) {
                    if (header_have != 0) return error.PartialFrame;
                    protocol_open = false;
                    continue;
                }
                // Any byte after the finish frame revokes it.
                if (finished) return error.UnexpectedFrame;
                header_have += @intCast(count);
                if (header_have < protocol.header_bytes) continue;
                frame = try protocol.Header.decode(&header);
                header_have = 0;
                payload_have = 0;
            } else {
                const wanted = frame.?.payload_bytes;
                const count = (try receive(executor.socket, executor.receive_buffer[payload_have..wanted], &peer_reset)) orelse continue;
                if (count == 0) return error.PartialFrame;
                payload_have += @intCast(count);
            }
            const complete = frame.?;
            if (payload_have < complete.payload_bytes) continue;
            frame = null;
            const payload = executor.receive_buffer[0..complete.payload_bytes];
            switch (try executor.session.receive(complete.kind, payload)) {
                .finish_pending => finished = true,
                .tool_call => |call| {
                    // The session approved this index against names copied from
                    // exactly these registered capabilities.
                    assert(call.index < executor.options.tools.len);
                    const capability = executor.options.tools[call.index];
                    if (try monotonic() >= deadline) return error.DeadlineExceeded;
                    const written = try capability.call(capability.context, call.argument, &executor.reply_buffer, deadline);
                    if (try monotonic() >= deadline) return error.DeadlineExceeded;
                    if (written > executor.reply_buffer.len) return error.ToolReplyTooLarge;
                    const reply = executor.reply_buffer[0..written];
                    try executor.session.reply(reply);
                    send_header = try (protocol.Header{ .kind = .tool_reply, .payload_bytes = written }).encode();
                    send_payload = reply;
                    send_done = 0;
                    sending = true;
                },
            }
        }
        assert(!sending);
        assert(frame == null);
        assert(header_have == 0);
        return peer_reset;
    }

    /// Returns true at EOF. Bytes beyond the fixed capacity fail the run.
    fn receiveDiagnostics(executor: *Executor) Error!bool {
        if (executor.diagnostic_bytes == diagnostic_bytes_max) {
            var probe: [1]u8 = undefined;
            const count = (try receive(executor.diagnostic_descriptor, &probe, null)) orelse return false;
            if (count == 0) return true;
            return error.DiagnosticsOverflow;
        }
        const count = (try receive(executor.diagnostic_descriptor, executor.diagnostic_buffer[executor.diagnostic_bytes..], null)) orelse return false;
        if (count == 0) return true;
        executor.diagnostic_bytes += @intCast(count);
        return false;
    }

    /// Failure path. Never returns the executor to idle unless a launched PID1 is
    /// reaped AND the lane is empty.
    fn cleanup(executor: *Executor) void {
        executor.session.abort();
        const launched = executor.child.process != null or executor.child.exit != null;
        // Nothing of ours can be live when clone never happened; the next run
        // re-verifies the lane itself. Never kill a lane this run did not populate.
        const proven = if (launched) executor.killAndJoin() else true;
        // Nonblocking and capped by the fixed buffer: keeps the last diagnostics.
        var reads: u32 = 0;
        while (executor.diagnostic_descriptor >= 0 and reads < diagnostic_bytes_max) : (reads += 1) {
            if (executor.diagnostic_bytes == diagnostic_bytes_max) break;
            const count = receive(executor.diagnostic_descriptor, executor.diagnostic_buffer[executor.diagnostic_bytes..], null) catch break;
            const bytes = count orelse break;
            if (bytes == 0) break;
            executor.diagnostic_bytes += @intCast(bytes);
        }
        executor.closeDescriptors();
        if (proven) {
            executor.child = .{};
            executor.state = .idle;
        } else {
            // Child keeps its PID: the caller cannot reset away a live process.
            executor.state = .quarantined;
        }
    }

    fn killAndJoin(executor: *Executor) bool {
        const lane = executor.options.lane;
        var attempts: u32 = 0;
        while (attempts < cleanup_attempts_max) : (attempts += 1) {
            // Repeated: a process forked during the first kill is caught by the next.
            // A kill error is not proof either way; only reap plus empty is.
            if (attempts % 64 == 0) executor.child.kill(lane) catch {};
            if (executor.child.process != null) {
                _ = executor.child.tryJoin() catch return false;
            }
            if (executor.child.process == null) {
                const empty = lane.empty() catch false;
                if (empty) return true;
            }
            pause(pause_ns);
        }
        return false;
    }

    fn closeDescriptors(executor: *Executor) void {
        if (executor.socket >= 0) close(executor.socket);
        if (executor.diagnostic_descriptor >= 0) close(executor.diagnostic_descriptor);
        executor.socket = -1;
        executor.diagnostic_descriptor = -1;
    }
};

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

/// Exactly one private component directly under /tmp.
fn validateWorkspace(path: []const u8) Error!void {
    const prefix = "/tmp/dszig-";
    if (path.len > path_bytes_max) return error.InvalidExecutorPath;
    if (!std.mem.startsWith(u8, path, prefix)) return error.InvalidExecutorPath;
    const name = path[prefix.len..];
    if (name.len == 0) return error.InvalidExecutorPath;
    if (std.mem.indexOfScalar(u8, name, '/') != null) return error.InvalidExecutorPath;
    if (std.mem.indexOf(u8, name, "..") != null) return error.InvalidExecutorPath;
}

/// A single-component path alone is insufficient: reject symlink substitution
/// and directories accessible to other users before creating the mount namespace.
fn verifyWorkspace(path: [:0]const u8) Error!void {
    const descriptor = linux.openat(linux.AT.FDCWD, path, .{ .PATH = true, .DIRECTORY = true, .CLOEXEC = true, .NOFOLLOW = true }, 0);
    if (failed(descriptor)) return error.InvalidExecutorPath;
    defer close(@intCast(descriptor));
    var metadata: linux.Statx = undefined;
    if (linux.statx(@intCast(descriptor), "", linux.AT.EMPTY_PATH, .{ .TYPE = true, .MODE = true, .UID = true }, &metadata) != 0)
        return error.InvalidExecutorPath;
    if (!metadata.mask.TYPE or !metadata.mask.MODE or !metadata.mask.UID)
        return error.InvalidExecutorPath;
    if (metadata.mode & linux.S.IFMT != linux.S.IFDIR) return error.InvalidExecutorPath;
    if (metadata.mode & 0o077 != 0) return error.InvalidExecutorPath;
    if (metadata.uid != linux.syscall0(.getuid)) return error.InvalidExecutorPath;
}

fn validateStorePath(path: []const u8) Error!void {
    const prefix = "/nix/store/";
    if (path.len > path_bytes_max) return error.InvalidExecutorPath;
    if (!std.mem.startsWith(u8, path, prefix)) return error.InvalidExecutorPath;
    if (path.len == prefix.len) return error.InvalidExecutorPath;
    var components = std.mem.splitScalar(u8, path[prefix.len..], '/');
    while (components.next()) |part| {
        if (part.len == 0) return error.InvalidExecutorPath;
        if (std.mem.eql(u8, part, ".")) return error.InvalidExecutorPath;
        if (std.mem.eql(u8, part, "..")) return error.InvalidExecutorPath;
    }
}

fn failed(result: usize) bool {
    return result > std.math.maxInt(usize) - 4096;
}

fn errno(result: usize) linux.E {
    assert(failed(result));
    return @fromBackingInt(@intCast(@as(u16, @intCast(0 -% result))));
}

fn close(descriptor: i32) void {
    assert(descriptor >= 0);
    _ = linux.syscall1(.close, @intCast(descriptor));
}

fn monotonic() Error!u64 {
    var now: Timespec = undefined;
    // CLOCK_MONOTONIC.
    if (linux.syscall2(.clock_gettime, 1, @intFromPtr(&now)) != 0) return error.ClockFailed;
    if (now.seconds < 0 or now.nanoseconds < 0) return error.ClockFailed;
    return @as(u64, @intCast(now.seconds)) * 1_000_000_000 + @as(u64, @intCast(now.nanoseconds));
}

/// Best-effort bounded sleep. An early return only spends a caller's attempt.
fn pause(nanoseconds: u64) void {
    assert(nanoseconds <= 1_000_000_000);
    var wait: Timespec = .{ .seconds = 0, .nanoseconds = @intCast(nanoseconds) };
    if (nanoseconds == 1_000_000_000) wait = .{ .seconds = 1, .nanoseconds = 0 };
    _ = linux.syscall5(.ppoll, 0, 0, @intFromPtr(&wait), 0, 8);
}

/// null: nothing available now. 0: EOF. Descriptors are nonblocking.
fn receive(descriptor: i32, buffer: []u8, peer_reset: ?*bool) Error!?usize {
    assert(buffer.len > 0);
    var interruptions: u8 = 0;
    while (interruptions <= interruptions_max) : (interruptions += 1) {
        const count = linux.syscall3(.read, @intCast(descriptor), @intFromPtr(buffer.ptr), buffer.len);
        if (failed(count)) {
            switch (errno(count)) {
                .INTR => continue,
                .AGAIN => return null,
                .CONNRESET => {
                    const reset = peer_reset orelse return error.ReceiveFailed;
                    reset.* = true;
                    return 0;
                },
                else => return error.ReceiveFailed,
            }
        }
        if (count > buffer.len) return error.ReceiveFailed;
        return count;
    }
    return null;
}
