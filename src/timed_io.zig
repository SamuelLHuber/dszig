//! Request-local deadlines for blocking I/O backends. Each network operation
//! races its own inactivity timer; cancellation always joins the losing task.
const std = @import("std");
const Io = std.Io;
const net = Io.net;
const TimedIo = @This();

inner: Io,
timeout_ns: ?u64,
connect_timeout_ns: ?u64,
failure_code: std.atomic.Value(u16) = .init(0),

pub fn io(self: *TimedIo) Io {
    return .{ .userdata = self, .vtable = &vtable };
}

pub fn resetFailure(self: *TimedIo) void {
    self.failure_code.store(0, .monotonic);
}

pub fn failure(self: *const TimedIo) ?anyerror {
    const code = self.failure_code.load(.monotonic);
    return if (code == 0) null else @errorFromInt(code);
}

fn recordFailure(self: *TimedIo, err: anyerror) void {
    _ = self.failure_code.cmpxchgStrong(0, @intFromError(err), .monotonic, .monotonic);
}

fn state(userdata: ?*anyopaque) *TimedIo {
    return @ptrCast(@alignCast(userdata.?));
}

const ConnectResult = net.IpAddress.ConnectError!net.Socket;

fn discard(self: *TimedIo, comptime R: type, result: R) void {
    if (R == ConnectResult) {
        const socket = result catch return;
        socket.close(self.inner);
    }
}

fn callTimed(self: *TimedIo, ns: u64, comptime function: anytype, args: std.meta.ArgsTuple(@TypeOf(function))) @typeInfo(@TypeOf(function)).@"fn".return_type.? {
    const R = @typeInfo(@TypeOf(function)).@"fn".return_type.?;
    if (ns == 0) {
        self.recordFailure(error.RequestTimedOut);
        return error.Canceled;
    }
    const Result = union(enum) { operation: R, timer: Io.Cancelable!void };
    var buffer: [2]Result = undefined;
    var select = Io.Select(Result).init(self.inner, &buffer);
    defer while (select.cancel()) |result| {
        switch (result) {
            .operation => |value| self.discard(R, value),
            .timer => {},
        }
    };
    select.concurrent(.operation, function, args) catch |err| {
        self.recordFailure(err);
        return error.Canceled;
    };
    const timeout: Io.Timeout = .{ .duration = .{
        .raw = .fromNanoseconds(ns),
        .clock = .awake,
    } };
    // async may execute inline when its worker budget is exhausted. A timer
    // must run concurrently, even when the operation already completed.
    select.concurrent(.timer, Io.Timeout.sleep, .{ timeout, self.inner }) catch |err| {
        self.recordFailure(err);
        return error.Canceled;
    };
    const completed = select.await() catch |err| {
        // External cancellation wins over a deadline recorded by another
        // address attempt sharing this request's adapter.
        self.failure_code.store(@intFromError(err), .monotonic);
        return err;
    };
    switch (completed) {
        .operation => |result| return result,
        .timer => |result| {
            try result;
            self.recordFailure(error.RequestTimedOut);
            return error.Canceled;
        },
    }
}

fn operate(userdata: ?*anyopaque, operation: Io.Operation) Io.Cancelable!Io.Operation.Result {
    const self = state(userdata);
    switch (operation) {
        .net_read, .net_write, .net_receive, .net_send => {
            if (self.timeout_ns) |ns| return self.callTimed(ns, Io.operate, .{ self.inner, operation });
        },
        else => {},
    }
    return self.inner.operate(operation);
}

fn connect(userdata: ?*anyopaque, address: *const net.IpAddress, options: net.IpAddress.ConnectOptions) ConnectResult {
    const self = state(userdata);
    if (self.connect_timeout_ns) |ns| {
        // The underlying Threaded backend does not implement ConnectOptions.timeout.
        return self.callTimed(ns, connectInner, .{ self.inner, address.*, options });
    }
    return self.inner.vtable.netConnectIp(self.inner.userdata, address, options);
}

fn connectInner(inner: Io, address: net.IpAddress, options: net.IpAddress.ConnectOptions) ConnectResult {
    var untimed = options;
    untimed.timeout = .none;
    return inner.vtable.netConnectIp(inner.userdata, &address, untimed);
}

fn lookup(userdata: ?*anyopaque, name: net.HostName, queue: *Io.Queue(net.HostName.LookupResult), options: net.HostName.LookupOptions) net.HostName.LookupError!void {
    const self = state(userdata);
    // Lookup owns queue closure even when an already-expired deadline prevents
    // the backend from starting. Otherwise the HTTP address consumer waits forever.
    defer queue.close(self.inner);
    if (self.connect_timeout_ns) |ns| return self.callTimed(ns, lookupInner, .{ self.inner, name, queue, options });
    return lookupInner(self.inner, name, queue, options);
}

fn lookupInner(inner: Io, name: net.HostName, queue: *Io.Queue(net.HostName.LookupResult), options: net.HostName.LookupOptions) net.HostName.LookupError!void {
    return inner.vtable.netLookup(inner.userdata, name, queue, options);
}

// All other capabilities retain the original backend and its userdata. A
// request-local adapter must never change a shared Client or connection's Io.
const vtable: Io.VTable = table: {
    var result: Io.VTable = undefined;
    for (@typeInfo(Io.VTable).@"struct".field_names) |name| {
        @field(result, name) = forward(name);
    }
    result.operate = operate;
    result.netConnectIp = connect;
    result.netLookup = lookup;
    break :table result;
};

fn forward(comptime name: []const u8) @FieldType(Io.VTable, name) {
    const Function = @typeInfo(@FieldType(Io.VTable, name)).pointer.child;
    const info = @typeInfo(Function).@"fn";
    const P = info.param_types;
    const R = info.return_type.?;
    return switch (P.len) {
        1 => struct {
            fn run(a: P[0].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{self.inner.userdata});
            }
        }.run,
        2 => struct {
            fn run(a: P[0].?, b: P[1].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b });
            }
        }.run,
        3 => struct {
            fn run(a: P[0].?, b: P[1].?, c: P[2].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b, c });
            }
        }.run,
        4 => struct {
            fn run(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b, c, d });
            }
        }.run,
        5 => struct {
            fn run(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b, c, d, e });
            }
        }.run,
        6 => struct {
            fn run(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b, c, d, e, f });
            }
        }.run,
        7 => struct {
            fn run(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b, c, d, e, f, g });
            }
        }.run,
        8 => struct {
            fn run(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?) R {
                const self = state(a);
                return @call(.auto, @field(self.inner.vtable, name), .{ self.inner.userdata, b, c, d, e, f, g, h });
            }
        }.run,
        else => @compileError("unsupported Io.VTable callback arity"),
    };
}

test "expired DNS deadline closes the result stream without waiting" {
    var timed: TimedIo = .{
        .inner = std.testing.io,
        .timeout_ns = 0,
        .connect_timeout_ns = 0,
    };
    var storage: [1]net.HostName.LookupResult = undefined;
    var queue = Io.Queue(net.HostName.LookupResult).init(&storage);
    try std.testing.expectError(error.Canceled, lookup(&timed, .{ .bytes = "unused.invalid" }, &queue, .{ .port = 443 }));
    try std.testing.expectEqual(error.RequestTimedOut, timed.failure().?);
    var received: [1]net.HostName.LookupResult = undefined;
    // min=0 makes this a bounded consumer read even if queue closure regresses.
    try std.testing.expectError(error.Closed, queue.get(std.testing.io, &received, 0));
}
