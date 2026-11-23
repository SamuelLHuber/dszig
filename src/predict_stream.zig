//! Caller-owned field listener routing for a synchronous program call.
const std = @import("std");
const predict = @import("predict.zig");
const lm = @import("lm.zig");
const adapter = @import("adapter.zig");
const xml = @import("xml.zig");
const Field = @import("signature.zig").Field;

pub const Error = lm.ChatStreamListener.Error || lm.JsonStreamListener.Error ||
    lm.XmlStreamListener.Error || error{
    UnknownStreamField,
    AmbiguousStreamField,
    StreamSessionBusy,
    UnsupportedStreamField,
    ConflictingStreamSinks,
    StreamStopped,
};

pub const Fragment = struct {
    predictor_name: ?[]const u8,
    field_name: []const u8,
    text: []const u8,
    is_last_chunk: bool,
};

pub const Sink = struct {
    context: *anyopaque,
    /// The fragment bytes are borrowed only during this callback.
    on_fragment: *const fn (*anyopaque, Fragment) lm.OpenAI.StreamSink.Action,
};

pub const Listener = struct {
    field_name: []const u8,
    /// Null finds the unique predictor declaring this output field.
    predictor: ?*const predict.PredictState = null,
    predictor_name: ?[]const u8 = null,
    allow_reuse: bool = false,
    marker_storage: []u8 = &.{},
    pending_storage: []u8,
    output_storage: []u8,
    xml_fields: []lm.XmlStreamListener.Field = &.{},
};

pub const Binding = struct {
    listener: Listener,
    parser: Parser = .none,
    citations: bool = false,

    const Parser = union(enum) {
        none,
        chat: lm.ChatStreamListener,
        json: lm.JsonStreamListener,
        xml: lm.XmlStreamListener,
        reasoning,

        fn started(self: *const Parser) bool {
            return switch (self.*) {
                .none, .reasoning => false,
                inline else => |*parser| parser.started(),
            };
        }

        fn cacheHit(self: *const Parser) bool {
            return switch (self.*) {
                .none, .reasoning => false,
                inline else => |*parser| parser.cacheHit(),
            };
        }
    };

    fn finalize(self: *Binding, sink: Sink) Error!void {
        const result = switch (self.parser) {
            .none, .reasoning => null,
            inline else => |*parser| try parser.finalize(),
        };
        if (result) |fragment| try self.emit(sink, fragment);
    }

    fn emit(self: *const Binding, sink: Sink, fragment: lm.ChatStreamListener.Fragment) Error!void {
        const action = sink.on_fragment(sink.context, .{
            .predictor_name = self.listener.predictor_name,
            .field_name = self.listener.field_name,
            .text = fragment.text,
            .is_last_chunk = fragment.is_last_chunk,
        });
        if (action == .stop) return error.StreamStopped;
    }

    fn configure(
        self: *Binding,
        arena: std.mem.Allocator,
        kind: adapter.Adapter,
        fields: []const Field,
        wire_name: []const u8,
    ) (Error || adapter.Error)!void {
        const listener = &self.listener;
        self.parser = switch (kind) {
            .chat => .{ .chat = try lm.ChatStreamListener.init(.{
                .wire_field = wire_name,
                .marker_storage = listener.marker_storage,
                .pending_storage = listener.pending_storage,
                .output_storage = listener.output_storage,
            }) },
            .json => .{ .json = try lm.JsonStreamListener.init(.{
                .wire_field = wire_name,
                .pending_storage = listener.pending_storage,
                .output_storage = listener.output_storage,
            }) },
            .xml => blk: {
                if (fields.len > listener.xml_fields.len) return error.BufferTooSmall;
                const wire_fields = listener.xml_fields[0..fields.len];
                for (fields, wire_fields) |field, *wire| wire.* = .{
                    .name = field.name,
                    .nested = try xml.XmlAdapter.fieldUsesNested(arena, field),
                };
                break :blk .{ .xml = try lm.XmlStreamListener.init(.{
                    .wire_field = wire_name,
                    .output_fields = wire_fields,
                    .pending_storage = listener.pending_storage,
                    .output_storage = listener.output_storage,
                }) };
            },
            .two_step => return error.UnsupportedStreamField,
        };
    }

    fn receive(self: *Binding, sink: Sink, event: lm.OpenAI.StreamEvent) Error!void {
        switch (event) {
            .delta => |delta| switch (delta.data) {
                .text => |part| {
                    const fragment = switch (self.parser) {
                        .none, .reasoning => null,
                        inline else => |*parser| try parser.receive(part.text),
                    };
                    if (fragment) |piece| try self.emit(sink, piece);
                },
                .thinking => |text| if (self.parser == .reasoning and text.len != 0) {
                    try self.emit(sink, .{ .text = text, .is_last_chunk = false });
                },
                .citation => |part| if (self.citations) {
                    try self.receiveCitation(sink, part);
                },
                else => {},
            },
            .end => |end| if (self.parser == .xml and end.finish_reason != null) {
                if (try self.parser.xml.finish()) |piece| try self.emit(sink, piece);
            },
            else => {},
        }
    }

    fn receiveCitation(self: *Binding, sink: Sink, part: lm.sdk.CitationPart) Error!void {
        const ended = switch (self.parser) {
            .none, .reasoning => true,
            inline else => |*parser| parser.state.ended,
        };
        if (ended) return;
        if (part.text == null and part.title == null and part.url == null) return;
        var writer = std.Io.Writer.fixed(self.listener.output_storage);
        @import("citation.zig").Citations.writePart(part, &writer) catch |err| return switch (err) {
            error.WriteFailed => error.BufferTooSmall,
            error.ParseValue, error.JsonTooDeep => error.UnsupportedStreamField,
        };
        try self.emit(sink, .{ .text = writer.buffered(), .is_last_chunk = false });
    }
};

pub const Session = struct {
    bindings: []Binding,
    sink: Sink,
    active: std.atomic.Value(bool) = .init(false),
    in_flight: std.atomic.Value(bool) = .init(false),

    /// Binds listeners before any model call. Descriptors and buffers must
    /// remain valid for the session; all mutable buffers must be disjoint.
    pub fn init(
        arena: std.mem.Allocator,
        program: anytype,
        listeners: []const Listener,
        storage: []Binding,
        sink: Sink,
    ) (Error || std.mem.Allocator.Error)!Session {
        if (listeners.len > storage.len) return error.BufferTooSmall;
        const needs_discovery = for (listeners) |listener| {
            if (listener.predictor == null) break true;
        } else false;
        const named = if (needs_discovery) try predict.namedPredictors(arena, program) else &.{};
        const bindings = storage[0..listeners.len];
        for (listeners, bindings) |listener, *bound| {
            bound.* = .{ .listener = listener };
            if (listener.predictor != null) continue;
            for (named) |entry| {
                const target = entry.param.predict;
                if (!hasField(target.fields[target.input_count..], listener.field_name)) continue;
                if (bound.listener.predictor != null) return error.AmbiguousStreamField;
                bound.listener.predictor = target;
                bound.listener.predictor_name = entry.name;
            }
            if (bound.listener.predictor == null) return error.UnknownStreamField;
        }
        return .{ .bindings = bindings, .sink = sink };
    }

    pub fn finish(self: *Session, include_prediction: bool) Error!bool {
        var started = false;
        var cache_hit = false;
        for (self.bindings) |*binding| {
            try binding.finalize(self.sink);
            started = started or binding.parser.started();
            cache_hit = cache_hit or binding.parser.cacheHit();
        }
        return include_prediction or self.bindings.len == 0 or cache_hit or !started;
    }

    pub fn selects(self: *const Session, target: *const predict.PredictState) bool {
        for (self.bindings) |*binding| {
            if (binding.listener.predictor == target) return true;
        }
        return false;
    }

    /// Discard borrowed attempt metadata on return to the program caller.
    pub fn clear(self: *Session) void {
        for (self.bindings) |*binding| binding.parser = .none;
    }
};

pub const NativeFields = struct { reasoning: bool, tools: bool };

pub const Attempt = struct {
    session: *Session,
    target: *const predict.PredictState,
    failure: ?Error = null,
    published: ?*bool = null,

    pub fn prepare(
        self: *Attempt,
        arena: std.mem.Allocator,
        kind: adapter.Adapter,
        original: []const Field,
        wire_fields: []const Field,
        native: NativeFields,
    ) (Error || adapter.Error)!void {
        for (self.session.bindings) |*binding| {
            if (binding.listener.predictor != self.target) continue;
            if (binding.listener.allow_reuse) {
                try binding.finalize(self.session.sink);
                binding.parser = .none;
            }
            if (binding.parser != .none) continue;
            const target = try findWireField(original, wire_fields, binding.listener.field_name, native);
            switch (target) {
                .reasoning => {
                    binding.citations = false;
                    binding.parser = .reasoning;
                },
                .text => |field| {
                    binding.citations = field.native_citations;
                    try binding.configure(arena, kind, wire_fields, field.name);
                },
            }
        }
    }

    pub fn sink(self: *Attempt) lm.OpenAI.StreamSink {
        return .{ .context = self, .on_event = receive };
    }

    fn publish(context: *anyopaque, fragment: Fragment) lm.OpenAI.StreamSink.Action {
        const self: *Attempt = @ptrCast(@alignCast(context));
        self.published.?.* = true;
        const target = self.session.sink;
        return target.on_fragment(target.context, fragment);
    }

    fn receive(context: *anyopaque, event: lm.OpenAI.StreamEvent) lm.OpenAI.StreamSink.Action {
        const self: *Attempt = @ptrCast(@alignCast(context));
        const routed: Sink = if (self.published != null)
            .{ .context = self, .on_fragment = publish }
        else
            self.session.sink;
        for (self.session.bindings) |*binding| {
            if (binding.listener.predictor != self.target) continue;
            binding.receive(routed, event) catch |err| {
                self.failure = err;
                return .stop;
            };
        }
        return .proceed;
    }
};

const WireField = union(enum) { reasoning, text: Field };

/// Native adaptation preserves output order while removing native-only
/// carriers and renaming reasoning fallbacks. Read the actual wire name.
fn findWireField(
    original: []const Field,
    wire: []const Field,
    name: []const u8,
    native: NativeFields,
) Error!WireField {
    std.debug.assert(wire.len <= original.len);
    var index: usize = 0;
    var remove_tool_output = native.tools;
    for (original) |field| {
        const selected = std.mem.eql(u8, field.name, name);
        if (field.native_tool_calls and remove_tool_output) {
            remove_tool_output = false;
            if (selected) return error.UnsupportedStreamField;
            continue;
        }
        if (field.native_reasoning and native.reasoning) {
            if (selected) return .reasoning;
            continue;
        }
        std.debug.assert(index < wire.len);
        if (selected) return .{ .text = wire[index] };
        index += 1;
    }
    std.debug.assert(index == wire.len);
    return error.UnknownStreamField;
}

test "only the negotiated tool output has no wire field" {
    const fields = [_]Field{
        .{ .name = "calls", .kind = .json, .type_name = "ToolCalls", .desc = "", .prefix = "", .native_tool_calls = true },
        .{ .name = "spare", .kind = .json, .type_name = "ToolCalls", .desc = "", .prefix = "", .native_tool_calls = true },
        .{ .name = "answer", .kind = .str, .type_name = "str", .desc = "", .prefix = "" },
    };
    const native: NativeFields = .{ .tools = true, .reasoning = false };
    try std.testing.expectError(error.UnsupportedStreamField, findWireField(&fields, fields[1..], "calls", native));
    for (fields[1..]) |f| {
        const wire = try findWireField(&fields, fields[1..], f.name, native);
        try std.testing.expectEqualStrings(f.name, wire.text.name);
    }
}

fn hasField(fields: []const Field, name: []const u8) bool {
    for (fields) |field| {
        if (std.mem.eql(u8, name, field.name)) return true;
    }
    return false;
}
