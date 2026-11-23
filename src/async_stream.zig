//! One producer task and one consumer. Acknowledgement keeps borrowed events
//! alive without copying provider, parser, status, or prediction storage.
const std = @import("std");
const predict = @import("predict.zig");
const lm = @import("lm.zig");
const route = @import("predict_stream.zig");
const status = @import("stream_status.zig");
const Io = std.Io;
const assert = std.debug.assert;

pub const StatusProvider = struct {
    context: *anyopaque,
    provide: *const fn (*anyopaque, status.Event) ?status.Message,
};

pub const Options = struct {
    listeners: []const route.Listener = &.{},
    bindings: []route.Binding = &.{},
    include_prediction: bool = true,
    status_provider: ?StatusProvider = null,
};

pub fn AsyncProgram(comptime Program: type) type {
    const Input = Program.Input;
    const Result = @typeInfo(@TypeOf(predict.callProgram(
        @as(*const Program, undefined),
        @as(predict.Context, undefined),
        @as(Input, undefined),
    ))).error_union.payload;
    return struct {
        const Self = @This();
        pub const Event = union(enum) {
            raw: lm.OpenAI.StreamEvent,
            field: route.Fragment,
            status_message: status.Message,
            prediction: *const Result,
        };
        const Phase = enum { idle, receiving, borrowed, exhausted };

        io: Io = undefined,
        events: Io.Queue(Event) = undefined,
        event_slot: [1]Event = undefined,
        acknowledgement: Io.Queue(u8) = undefined,
        acknowledgement_slot: [1]u8 = undefined,
        phase: Phase = .idle,
        future: Io.Future(anyerror!Result) = undefined,
        session: route.Session = undefined,
        status_provider: ?StatusProvider = null,
        delivery: Io.Mutex = .init,
        producer_stopped: std.atomic.Value(bool) = .init(false),

        /// Keep this object, program, input backing, context arena, and
        /// listener buffers alive and immovable until exhaustion or cancel.
        /// Caller and worker must not mutate the same storage concurrently.
        pub fn start(
            self: *Self,
            io: Io,
            program: *Program,
            ctx: predict.Context,
            input: Input,
            options: Options,
        ) !void {
            if (self.phase != .idle) return error.StreamSessionBusy;
            if (ctx.stream_sink != null or ctx.stream_session != null or ctx.status_sink != null)
                return error.ConflictingStreamSinks;
            self.io = io;
            self.events = .init(&self.event_slot);
            self.acknowledgement = .init(&self.acknowledgement_slot);
            self.status_provider = options.status_provider;
            self.delivery = .init;
            self.producer_stopped.store(false, .monotonic);
            self.session = try route.Session.init(ctx.arena, program, options.listeners, options.bindings, .{
                .context = self,
                .on_fragment = onFragment,
            });
            self.future = try io.concurrent(run, .{ self, program, ctx, input, options.include_prediction });
            self.phase = .receiving;
        }

        /// The borrowed event is invalidated by the next `next` or `cancel`.
        /// Termination joins the producer and surfaces its actual program error.
        pub fn next(self: *Self) anyerror!?Event {
            switch (self.phase) {
                .idle => return error.StreamNotStarted,
                .exhausted => return null,
                .borrowed => {
                    const count = self.acknowledgement.putUncancelable(self.io, &.{0}, 1) catch unreachable;
                    assert(count == 1);
                    self.phase = .receiving;
                },
                .receiving => {},
            }
            const event = self.events.getOne(self.io) catch |err| switch (err) {
                error.Closed => {
                    self.phase = .exhausted;
                    _ = try self.future.await(self.io);
                    return null;
                },
                error.Canceled => {
                    self.cancel();
                    return err;
                },
            };
            self.phase = .borrowed;
            return event;
        }

        /// Stop and join even when suspended in delivery or awaiting I/O.
        /// User callbacks doing noncancelable work must return before this does.
        pub fn cancel(self: *Self) void {
            switch (self.phase) {
                .idle, .exhausted => return,
                .receiving, .borrowed => {},
            }
            self.acknowledgement.close(self.io);
            self.events.close(self.io);
            _ = self.future.cancel(self.io) catch {};
            self.phase = .exhausted;
        }

        /// Retrieve the final result after exhaustion, even when its event was
        /// suppressed. This does not consume or skip pending events.
        pub fn result(self: *Self) anyerror!Result {
            if (self.phase != .exhausted) return error.StreamNotExhausted;
            return self.future.await(self.io);
        }

        fn run(self: *Self, program: *Program, ctx: predict.Context, input: Input, include: bool) anyerror!Result {
            defer self.events.close(self.io);
            var scoped = ctx;
            if (self.session.bindings.len == 0) scoped.stream_sink = .{ .context = self, .on_event = onRaw };
            scoped.status_sink = .{
                .context = self,
                .on_message = onStatus,
                .provide = if (self.status_provider != null) provideStatus else null,
            };
            const output = try predict.callStream(program, scoped, input, &self.session, include);
            if (self.producer_stopped.load(.monotonic)) return error.StreamStopped;
            if (output.emit_prediction and self.send(.{ .prediction = &output.prediction }) == .stop)
                return error.StreamStopped;
            return output.prediction;
        }

        fn send(self: *Self, event: Event) lm.OpenAI.StreamSink.Action {
            self.delivery.lockUncancelable(self.io);
            defer self.delivery.unlock(self.io);
            if (self.producer_stopped.load(.monotonic)) return .stop;
            self.events.putOne(self.io, event) catch {
                self.producer_stopped.store(true, .monotonic);
                return .stop;
            };
            var ack: [1]u8 = undefined;
            const count = self.acknowledgement.getUncancelable(self.io, &ack, 1) catch {
                self.producer_stopped.store(true, .monotonic);
                return .stop;
            };
            assert(count == 1);
            return .proceed;
        }

        fn onRaw(ptr: *anyopaque, event: lm.OpenAI.StreamEvent) lm.OpenAI.StreamSink.Action {
            const self: *Self = @ptrCast(@alignCast(ptr));
            return self.send(.{ .raw = event });
        }

        fn onFragment(ptr: *anyopaque, fragment: route.Fragment) lm.OpenAI.StreamSink.Action {
            const self: *Self = @ptrCast(@alignCast(ptr));
            return self.send(.{ .field = fragment });
        }

        fn onStatus(ptr: *anyopaque, message: status.Message) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            _ = self.send(.{ .status_message = message });
        }

        fn provideStatus(ptr: *anyopaque, event: status.Event) ?status.Message {
            const self: *Self = @ptrCast(@alignCast(ptr));
            if (self.producer_stopped.load(.monotonic)) return null;
            const provider = self.status_provider.?;
            return provider.provide(provider.context, event);
        }
    };
}

test "asynchronous stream relays the program error and joins before returning" {
    const Probe = struct {
        pub const Input = void;
        cleaned: *bool,

        pub fn forward(self: *const @This(), _: predict.Context, _: Input) error{Intentional}!u8 {
            defer self.cleaned.* = true;
            return error.Intentional;
        }
    };

    var cleaned = false;
    var probe: Probe = .{ .cleaned = &cleaned };
    var source: AsyncProgram(Probe) = .{};
    defer source.cancel();
    try source.start(std.testing.io, &probe, .{ .arena = std.testing.allocator }, {}, .{});
    try std.testing.expectError(error.StreamNotExhausted, source.result());
    try std.testing.expectError(error.Intentional, source.next());
    try std.testing.expect(cleaned);
    try std.testing.expectError(error.Intentional, source.result());
    try std.testing.expectEqual(null, try source.next());
}
