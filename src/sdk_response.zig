//! Ordered lm15 request and response artifacts, independent of LM execution.
//! Constructors and clones own their storage; value projections borrow leaves.
//! Tool-result content has a separate union: nested protocol parts cannot exist.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const inference = @import("inference.zig");
const scores = @import("logprobs.zig");
const pyjson = @import("pyjson.zig");
const json = @import("sdk_json.zig");
const graph = @import("json_graph.zig");
const Payload = @import("request_cache.zig").Payload;
pub const JsonValue = json.Value;
/// The same graph and payload owner, without the JSON-field restriction.
pub const PayloadValue = json.Value;
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
pub const Usage = @import("lm_usage.zig").Usage;
pub const Error = Allocator.Error || error{InvalidResponse};
pub const Role = enum { user, assistant, tool, developer };
pub const FinishReason = enum { stop, length, tool_call, content_filter, @"error" };
pub const JudgmentMethod = enum { provider_classification, candidate_sequence_likelihood };
pub const ImageDetail = enum { low, high, auto };
pub const AdaptationAction = enum { dropped, clamped, substituted, client_side, satisfied, defaulted };
pub const Continuation = inference.Continuation;
pub const CitationPart = inference.CitationPart;

pub const TextPart = struct { text: []const u8 = "", continuation: []const Continuation = &.{} };
pub const ThinkingPart = struct { text: []const u8 = "", continuation: []const Continuation = &.{} };
pub const RefusalPart = struct { text: []const u8 = "", continuation: []const Continuation = &.{} };
pub const MediaPart = struct {
    media_type: []const u8 = "",
    data: ?[]const u8 = null,
    url: ?[]const u8 = null,
    file_id: ?[]const u8 = null,
    path: ?[]const u8 = null,
    detail: ?ImageDetail = null,
    continuation: []const Continuation = &.{},
};
pub const ToolCallPart = struct {
    id: []const u8,
    name: []const u8,
    input: JsonValue,
    continuation: []const Continuation = &.{},
};
pub const DataPart = struct {
    value: JsonValue,
    probabilities: ?JsonValue = null,
    method: ?JudgmentMethod = null,
    continuation: []const Continuation = &.{},
};
pub const ContentPart = union(enum) {
    text: TextPart,
    image: MediaPart,
    audio: MediaPart,
    video: MediaPart,
    document: MediaPart,
    binary: MediaPart,
    citation: CitationPart,
    data: DataPart,

    pub fn jsonStringify(self: ContentPart, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        return writePart(ContentPart, self, stream);
    }
};
pub const ToolResultPart = struct {
    id: []const u8,
    content: []const ContentPart = &.{},
    name: ?[]const u8 = null,
    is_error: bool = false,
    continuation: []const Continuation = &.{},
};
pub const Part = union(enum) {
    text: TextPart,
    image: MediaPart,
    audio: MediaPart,
    video: MediaPart,
    document: MediaPart,
    binary: MediaPart,
    tool_call: ToolCallPart,
    tool_result: ToolResultPart,
    thinking: ThinkingPart,
    refusal: RefusalPart,
    citation: CitationPart,
    data: DataPart,

    pub fn jsonStringify(self: Part, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        return writePart(Part, self, stream);
    }
    pub fn fromValue(allocator: Allocator, input: Value) Error!Part {
        return read(Part, allocator, input, false);
    }
    pub fn clone(self: *const Part, allocator: Allocator) Allocator.Error!Part {
        return copy(Part, allocator, self.*);
    }
    pub fn free(self: *const Part, allocator: Allocator) void {
        release(Part, allocator, self.*);
    }
    pub fn toValue(self: *const Part, arena: Allocator) Error!Value {
        return project(Part, arena, self.*, false);
    }
};
pub const Message = struct {
    role: Role,
    parts: []const Part,
    continuation: []const Continuation = &.{},

    pub fn jsonStringify(self: Message, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(Message, self, stream, false);
        try stream.endObject();
    }
    pub fn fromValue(allocator: Allocator, input: Value) Error!Message {
        return read(Message, allocator, input, false);
    }
    pub fn clone(self: *const Message, allocator: Allocator) Allocator.Error!Message {
        return copy(Message, allocator, self.*);
    }
    pub fn free(self: *const Message, allocator: Allocator) void {
        release(Message, allocator, self.*);
    }
    pub fn toValue(self: *const Message, arena: Allocator) Error!Value {
        return project(Message, arena, self.*, false);
    }
};
pub const Adaptation = struct {
    field: []const u8,
    action: AdaptationAction,
    reason: []const u8,
    asked: PayloadValue = .{},
    applied: PayloadValue = .{},

    pub fn jsonStringify(self: Adaptation, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(Adaptation, self, stream, false);
        try stream.endObject();
    }
};
pub const ReasoningEffort = enum { off, minimal, low, medium, high, xhigh, max };
pub const ReasoningSummary = enum { auto, concise, detailed };
pub const ProbabilityPolicy = enum { off, if_available, required };
pub const Reasoning = struct {
    effort: ?ReasoningEffort = null,
    thinking_budget: Value = .null,
    summary: ?ReasoningSummary = null,

    pub fn jsonStringify(self: Reasoning, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(Reasoning, self, stream, false);
        try stream.endObject();
    }
};
pub const CacheConfig = struct {
    mode: enum { auto, off } = .auto,
    retention: ?enum { short, long } = null,
    key: ?[]const u8 = null,
    prefix_until_index: Value = .null,
    prefix: ?enum { stable, history } = null,
    resource: ?[]const u8 = null,

    pub fn jsonStringify(self: CacheConfig, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(CacheConfig, self, stream, false);
        try stream.endObject();
    }
};
pub const ToolChoice = struct {
    mode: enum { auto, required, none } = .auto,
    allowed: []const []const u8 = &.{},
    parallel: ?bool = null,

    pub fn jsonStringify(self: ToolChoice, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(ToolChoice, self, stream, false);
        try stream.endObject();
    }
};
pub const Config = struct {
    max_tokens: Value = .null,
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: Value = .null,
    stop: []const []const u8 = &.{},
    response_format: ?JsonValue = null,
    tool_choice: ?ToolChoice = null,
    reasoning: ?Reasoning = null,
    cache: ?CacheConfig = null,
    seed: Value = .null,
    frequency_penalty: ?f64 = null,
    presence_penalty: ?f64 = null,
    service_tier: ?[]const u8 = null,
    user_id: ?[]const u8 = null,
    store: ?bool = null,
    logprobs: Value = .null,
    probabilities: ?ProbabilityPolicy = null,
    extensions: ?JsonValue = null,

    pub fn toValue(self: *const Config, arena: Allocator) Error!Value {
        return project(Config, arena, self.*, false);
    }

    pub fn jsonStringify(self: Config, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(Config, self, stream, false);
        try stream.endObject();
    }
};
pub const RequestTool = union(enum) {
    function: struct {
        name: []const u8,
        description: ?[]const u8 = null,
        /// Null is constructor input for the SDK's fresh empty object schema.
        parameters: ?JsonValue = null,
    },
    builtin: struct {
        name: []const u8,
        config: ?JsonValue = null,
    },

    pub fn jsonStringify(self: RequestTool, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        return writePart(RequestTool, self, stream);
    }
};
pub const SystemContent = union(enum) {
    text: []const u8,
    parts: []const ContentPart,

    pub fn jsonStringify(self: SystemContent, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        switch (self) {
            .text => |text| try stream.write(text),
            .parts => |parts| try stream.write(parts),
        }
    }
};
pub const RequestFields = struct {
    model: []const u8,
    messages: []const Message,
    system: ?SystemContent = null,
    tools: []const RequestTool = &.{},
    config: Config = .{},

    pub fn jsonStringify(self: RequestFields, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(RequestFields, self, stream, false);
        try stream.endObject();
    }
    pub fn toValue(self: *const RequestFields, arena: Allocator) Error!Value {
        return project(RequestFields, arena, self.*, false);
    }
    pub fn free(self: *const RequestFields, allocator: Allocator) void {
        release(RequestFields, allocator, self.*);
    }
};

/// A complete, stable SDK request, distinct from the execution adapter request.
/// The object owns immutable headers and retains owned mutable JSON payloads.
pub const Request = struct {
    fields: RequestFields,
    allocator: Allocator,
    references: std.atomic.Value(u32) = .init(1),
    /// Seed variants own only the changed seed; the root owns everything else.
    seed_parent: ?*const Request = null,

    pub fn fromFields(allocator: Allocator, input: *const RequestFields) Error!*const Request {
        var context: CopyContext = .{
            .allocator = allocator,
            .runtime_cloner = .init(allocator),
            .json_cloner = .{ .allocator = allocator },
            .share_owned_json = true,
        };
        defer context.deinit();
        var fields = try copyWithContext(RequestFields, &context, input.*);
        errdefer fields.free(allocator);
        try validateRequest(allocator, &fields);
        return adopt(allocator, fields);
    }

    pub fn fromValue(allocator: Allocator, input: Value) Error!*const Request {
        const fields = try read(RequestFields, allocator, input, false);
        errdefer fields.free(allocator);
        return adopt(allocator, fields);
    }

    /// Own typed headers while retaining opaque graph payloads from an owning
    /// canonical record. Explicitly borrowed input is snapshotted once first.
    pub fn fromPayload(allocator: Allocator, input: PayloadValue) Error!*const Request {
        const fields = try readPayload(RequestFields, allocator, input);
        errdefer fields.free(allocator);
        return adopt(allocator, fields);
    }

    fn adopt(allocator: Allocator, fields: RequestFields) Allocator.Error!*const Request {
        const object = try allocator.create(Request);
        object.* = .{ .fields = fields, .allocator = allocator };
        std.debug.assert(object.references.load(.monotonic) == 1);
        return object;
    }

    pub fn retain(self: *const Request) *const Request {
        const previous = @constCast(self).references.fetchAdd(1, .monotonic);
        std.debug.assert(previous > 0);
        std.debug.assert(previous < std.math.maxInt(u32));
        return self;
    }

    pub fn release(self: *const Request) void {
        const previous = @constCast(self).references.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous != 1) return;
        const allocator = self.allocator;
        if (self.seed_parent) |parent| {
            std.debug.assert(parent.seed_parent == null);
            runtime.freeValue(allocator, self.fields.config.seed);
            parent.release();
        } else {
            self.fields.free(allocator);
        }
        allocator.destroy(self);
    }

    /// Keep immutable message headers and mutable payload identity while
    /// replacing only the seed. Retains the root directly, never a chain.
    pub fn withSeed(self: *const Request, allocator: Allocator, input: Value) Error!*const Request {
        const parent = self.seed_parent orelse self;
        std.debug.assert(parent.seed_parent == null);
        var seed = try copy(Value, allocator, input);
        errdefer runtime.freeValue(allocator, seed);
        try configInteger(allocator, &seed, null);
        const request = try allocator.create(Request);
        request.* = .{
            .fields = parent.fields,
            .allocator = allocator,
            .seed_parent = parent.retain(),
        };
        request.fields.config.seed = seed;
        return request;
    }

    pub fn clone(self: *const Request, allocator: Allocator) Allocator.Error!*const Request {
        const fields = try copy(RequestFields, allocator, self.fields);
        errdefer fields.free(allocator);
        return adopt(allocator, fields);
    }

    /// Freeze local media before cache lookup and engine execution. The caller
    /// owns one returned reference, including when no replacement is necessary.
    /// Unlike Source's unbounded read_bytes, native reads have an explicit cap.
    pub fn snapshot(
        self: *const Request,
        allocator: Allocator,
        io: std.Io,
        maximum_file_bytes: usize,
    ) !*const Request {
        var has_paths = false;
        for (self.fields.messages) |message|
            for (message.parts) |part| {
                if (partHasPath(part)) has_paths = true;
            };
        if (self.fields.system) |system| if (system == .parts) {
            for (system.parts) |part| {
                if (partHasPath(part)) has_paths = true;
            }
        };
        if (!has_paths) return self.retain();

        var context: CopyContext = .{
            .allocator = allocator,
            .runtime_cloner = .init(allocator),
            .json_cloner = .{ .allocator = allocator },
            .share_owned_json = true,
        };
        defer context.deinit();
        var fields = try copyWithContext(RequestFields, &context, self.fields);
        errdefer fields.free(allocator);
        // These immutable slice views now refer exclusively to our own headers.
        for (fields.messages) |*message|
            for (@constCast(message.parts)) |*part|
                try snapshotPart(@TypeOf(part.*), part, allocator, io, maximum_file_bytes);
        if (fields.system) |system| if (system == .parts) {
            for (@constCast(system.parts)) |*part|
                try snapshotPart(@TypeOf(part.*), part, allocator, io, maximum_file_bytes);
        };
        return adopt(allocator, fields);
    }

    pub fn jsonStringify(self: *const Request, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        return self.fields.jsonStringify(stream);
    }

    pub fn toValue(self: *const Request, arena: Allocator) Error!Value {
        return self.fields.toValue(arena);
    }

    /// Owning canonical serde headers with the original opaque payload roots.
    pub fn toPayload(self: *const Request, allocator: Allocator) Error!PayloadValue {
        return recordPayload(RequestFields, allocator, self.fields);
    }

    /// Display messages without resolving file paths. The returned record owns
    /// its headers and retains the original mutable payloads of structured parts.
    pub fn historyMessages(self: *const Request, allocator: Allocator) Error!PayloadValue {
        const storage = try recordStorage(allocator, .{
            .system = self.fields.system,
            .messages = self.fields.messages,
        });
        errdefer storage.release();
        const arena = storage.allocator();
        const messages = try arena.create(graph.Array);
        messages.* = .init(arena);
        const count = self.fields.messages.len + @as(usize, @intFromBool(self.fields.system != null));
        try messages.ensureTotalCapacity(count);
        if (self.fields.system) |system| {
            const content = try projectAs(.graph, SystemContent, arena, system, false);
            messages.appendAssumeCapacity(try displayMessage(arena, "system", content));
        }
        for (self.fields.messages) |message| {
            const content = try displayContent(arena, message.parts);
            messages.appendAssumeCapacity(try displayMessage(arena, @tagName(message.role), content));
        }
        std.debug.assert(messages.items.len == count);
        storage.root = .{ .array = messages };
        return .{ .root = storage.root, .storage = storage };
    }
};

pub const ResponseFields = struct {
    id: ?[]const u8 = null,
    model: []const u8,
    message: Message,
    finish_reason: FinishReason,
    usage: Usage = .{},
    logprobs: ?[]const scores.TokenLogprob = null,
    logprobs_complete: bool = true,
    provider_data: ?JsonValue = null,
    adaptations: []const Adaptation = &.{},

    /// Durable native serialization retains empty scores and IEEE-754 tags.
    /// Use toValue() for the canonical SDK projection.
    pub fn jsonStringify(self: ResponseFields, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeRecord(ResponseFields, self, stream, false);
        try stream.endObject();
    }
    pub fn fromValue(allocator: Allocator, input: Value) Error!ResponseFields {
        return read(ResponseFields, allocator, input, false);
    }
    pub fn fromNativeValue(allocator: Allocator, input: Value) Error!ResponseFields {
        return read(ResponseFields, allocator, input, true);
    }
    pub fn clone(self: *const ResponseFields, allocator: Allocator) Allocator.Error!ResponseFields {
        return copy(ResponseFields, allocator, self.*);
    }
    pub fn free(self: *const ResponseFields, allocator: Allocator) void {
        release(ResponseFields, allocator, self.*);
    }
    /// Canonical SDK projection, including provider data. Empty scores are
    /// omitted here, but not in native persistence or the list-call view.
    pub fn toValue(self: *const ResponseFields, arena: Allocator) Error!Value {
        return project(ResponseFields, arena, self.*, false);
    }
    pub fn toNativeValue(self: *const ResponseFields, arena: Allocator) Error!Value {
        return project(ResponseFields, arena, self.*, true);
    }
};

/// A stable SDK object. Retaining preserves object and mutable-payload identity;
/// clone makes an independent deep copy. The allocator outlives every reference.
pub const Response = struct {
    fields: ResponseFields,
    allocator: Allocator,
    references: std.atomic.Value(u32) = .init(1),

    pub fn fromFields(allocator: Allocator, input: *const ResponseFields) Error!*const Response {
        var context: CopyContext = .{
            .allocator = allocator,
            .runtime_cloner = .init(allocator),
            .json_cloner = .{ .allocator = allocator },
            .share_owned_json = true,
        };
        defer context.deinit();
        var owned = try copyWithContext(ResponseFields, &context, input.*);
        errdefer owned.free(allocator);
        try validateRecord(ResponseFields, allocator, &owned);
        // Native headers are values, not already validated SDK class instances.
        try validateMessage(&owned.message);
        try validateContinuationList(allocator, owned.message.continuation);
        for (owned.message.parts) |part| switch (part) {
            inline else => |payload| try validatePayload(allocator, payload),
        };
        for (owned.adaptations) |adaptation| try validatePayload(allocator, adaptation);
        return adopt(allocator, owned);
    }

    pub fn fromValue(allocator: Allocator, input: Value) Error!*const Response {
        const fields = try ResponseFields.fromValue(allocator, input);
        errdefer fields.free(allocator);
        return adopt(allocator, fields);
    }

    /// Own typed headers while retaining opaque graph payloads from an owning
    /// canonical record. Explicitly borrowed input is snapshotted once first.
    pub fn fromPayload(allocator: Allocator, input: PayloadValue) Error!*const Response {
        const fields = try readPayload(ResponseFields, allocator, input);
        errdefer fields.free(allocator);
        return adopt(allocator, fields);
    }

    pub fn fromNativeValue(allocator: Allocator, input: Value) Error!*const Response {
        const fields = try ResponseFields.fromNativeValue(allocator, input);
        errdefer fields.free(allocator);
        return adopt(allocator, fields);
    }

    fn adopt(allocator: Allocator, fields: ResponseFields) Allocator.Error!*const Response {
        const object = try allocator.create(Response);
        object.* = .{ .fields = fields, .allocator = allocator };
        std.debug.assert(object.references.load(.monotonic) == 1);
        return object;
    }

    pub fn retain(self: *const Response) *const Response {
        const previous = @constCast(self).references.fetchAdd(1, .monotonic);
        std.debug.assert(previous > 0);
        std.debug.assert(previous < std.math.maxInt(u32));
        return self;
    }

    pub fn release(self: *const Response) void {
        const previous = @constCast(self).references.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous != 1) return;
        const allocator = self.allocator;
        self.fields.free(allocator);
        allocator.destroy(self);
    }

    /// The context borrows each returned object until deinit. Keep successful
    /// results alive so repeated references and cross-response graphs can share.
    pub const CloneContext = struct {
        fields: CopyContext,
        objects: std.AutoHashMapUnmanaged(*const Response, *const Response) = .empty,

        pub fn init(allocator: Allocator) CloneContext {
            return .{ .fields = .{
                .allocator = allocator,
                .runtime_cloner = .init(allocator),
                .json_cloner = .{ .allocator = allocator },
            } };
        }

        pub fn deinit(self: *CloneContext) void {
            self.objects.deinit(self.fields.allocator);
            self.fields.deinit();
        }

        /// Related public payloads share the response fields' identity memo.
        /// Keep returned values alive until this context is deinitialized.
        pub fn clonePayload(self: *CloneContext, input: PayloadValue) Allocator.Error!PayloadValue {
            return self.fields.json_cloner.clone(input);
        }

        pub fn clone(self: *CloneContext, input: *const Response) Allocator.Error!*const Response {
            if (self.objects.get(input)) |existing| return existing.retain();
            const fields = try copyWithContext(ResponseFields, &self.fields, input.fields);
            errdefer fields.free(self.fields.allocator);
            const object = try Response.adopt(self.fields.allocator, fields);
            self.objects.put(self.fields.allocator, input, object) catch |err| {
                // fields has its own unwind above; only release the header here.
                self.fields.allocator.destroy(object);
                return err;
            };
            return object;
        }
    };

    pub fn clone(self: *const Response, allocator: Allocator) Allocator.Error!*const Response {
        var context: CloneContext = .init(allocator);
        defer context.deinit();
        return context.clone(self);
    }

    pub fn jsonStringify(self: *const Response, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        return self.fields.jsonStringify(stream);
    }

    pub fn toValue(self: *const Response, arena: Allocator) Error!Value {
        return self.fields.toValue(arena);
    }

    /// Canonical serde record with owning immutable headers. Opaque payload
    /// roots retain their exact graph identity and their existing owners.
    pub fn toPayload(self: *const Response, allocator: Allocator) Error!PayloadValue {
        return recordPayload(ResponseFields, allocator, self.fields);
    }

    pub fn toNativeValue(self: *const Response, arena: Allocator) Error!Value {
        return self.fields.toNativeValue(arena);
    }
};

fn displayContent(arena: Allocator, parts: []const Part) Error!graph.Value {
    var byte_count = parts.len -| 1;
    for (parts) |part| {
        if (part != .text) return projectAs(.graph, []const Part, arena, parts, false);
        byte_count = std.math.add(usize, byte_count, part.text.text.len) catch
            return error.OutOfMemory;
    }
    const joined = try arena.alloc(u8, byte_count);
    var next: usize = 0;
    for (parts, 0..) |part, index| {
        if (index != 0) {
            joined[next] = '\n';
            next += 1;
        }
        const text = part.text.text;
        @memcpy(joined[next..][0..text.len], text);
        next += text.len;
    }
    std.debug.assert(next == joined.len);
    return .{ .string = joined };
}

fn displayMessage(
    arena: Allocator,
    role: []const u8,
    content: graph.Value,
) Allocator.Error!graph.Value {
    var object: graph.Record = .empty;
    try object.put(arena, "role", .{ .string = role });
    try object.put(arena, "content", content);
    return projectedObject(.graph, arena, object);
}

fn recordPayload(comptime T: type, allocator: Allocator, fields: T) Error!PayloadValue {
    const storage = try recordStorage(allocator, fields);
    errdefer storage.release();
    storage.root = try projectAs(.graph, T, storage.allocator(), fields, false);
    return .{ .root = storage.root, .storage = storage };
}

fn recordStorage(allocator: Allocator, fields: anytype) Allocator.Error!*Payload {
    var parents: std.ArrayList(*Payload) = .empty;
    defer parents.deinit(allocator);
    try collectPayloadParents(@TypeOf(fields), allocator, fields, &parents);
    const owned_parents = try parents.toOwnedSlice(allocator);
    errdefer allocator.free(owned_parents);
    return Payload.createDependent(allocator, owned_parents);
}

/// The two SDK artifacts must be cloned together: Data and provider metadata
/// can refer to the same mutable graph even though their typed fields differ.
pub const Artifacts = struct {
    message: ?Message = null,
    provider_data: ?JsonValue = null,

    pub fn clone(self: Artifacts, allocator: Allocator) Allocator.Error!Artifacts {
        return copy(Artifacts, allocator, self);
    }

    pub fn free(self: Artifacts, allocator: Allocator) void {
        release(Artifacts, allocator, self);
    }
};

pub fn cloneAdaptations(allocator: Allocator, input: []const Adaptation) Allocator.Error![]const Adaptation {
    return copy([]const Adaptation, allocator, input);
}
pub fn freeAdaptations(allocator: Allocator, input: []const Adaptation) void {
    release([]const Adaptation, allocator, input);
}
pub fn adaptationsValue(arena: Allocator, input: []const Adaptation) Error!Value {
    return project([]const Adaptation, arena, input, false);
}
pub fn adaptationsFromValue(allocator: Allocator, input: Value) Error![]const Adaptation {
    return read([]const Adaptation, allocator, input, false);
}

fn writePart(comptime T: type, part: T, stream: *std.json.Stringify) std.Io.Writer.Error!void {
    try stream.beginObject();
    switch (part) {
        inline else => |payload, tag| {
            try stream.objectField("type");
            try stream.write(@tagName(tag));
            try writeRecord(@TypeOf(payload), payload, stream, @TypeOf(payload) == MediaPart and tag != .image);
        },
    }
    try stream.endObject();
}

fn writeRecord(
    comptime T: type,
    record: T,
    stream: *std.json.Stringify,
    comptime ignore_detail: bool,
) std.Io.Writer.Error!void {
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types) |name, FT| {
        const field = @field(record, name);
        if (fieldPresent(T, FT, name, field, true, ignore_detail)) {
            try stream.objectField(name);
            if (FT == Usage) {
                try stream.beginObject();
                try writeRecord(Usage, field, stream, false);
                try stream.endObject();
            } else try stream.write(field);
        }
    }
}

fn fieldPresent(
    comptime T: type,
    comptime FT: type,
    comptime name: []const u8,
    field: FT,
    native: bool,
    comptime ignore_detail: bool,
) bool {
    if (comptime T == MediaPart and std.mem.eql(u8, name, "detail") and ignore_detail) return false;
    if (comptime T == ResponseFields and FT == Usage) {
        if (!native) {
            inline for (@typeInfo(Usage).@"struct".field_names) |counter|
                if (@field(field, counter) != .null) return true;
            return false;
        }
    }
    if (@typeInfo(FT) == .optional) {
        if (field == null) return false;
    }
    if (FT == JsonValue and field.root == .null)
        return T == DataPart and std.mem.eql(u8, name, "value");
    if (FT == Value and field == .null) {
        return T == DataPart and std.mem.eql(u8, name, "value");
    }
    if (comptime std.mem.eql(u8, name, "continuation") or std.mem.eql(u8, name, "adaptations"))
        return field.len != 0;
    if (comptime T == ResponseFields and std.mem.eql(u8, name, "logprobs"))
        return native or field.?.len != 0;
    if (comptime T == ResponseFields and std.mem.eql(u8, name, "logprobs_complete")) return !field;
    if (comptime T == ToolResultPart and std.mem.eql(u8, name, "is_error")) return field;
    if (comptime T == RequestFields and std.mem.eql(u8, name, "config")) {
        inline for (@typeInfo(Config).@"struct".field_names, @typeInfo(Config).@"struct".field_types) |key, Child| {
            if (fieldPresent(Config, Child, key, @field(field, key), native, false)) return true;
        }
        return false;
    }
    if (comptime T == RequestFields and std.mem.eql(u8, name, "system")) {
        if (field.? == .parts) return field.?.parts.len != 0;
    }
    if (comptime T == Config and std.mem.eql(u8, name, "extensions")) {
        return field.?.root != .object or field.?.root.object.count() != 0;
    }
    if (comptime T == @FieldType(RequestTool, "builtin") and std.mem.eql(u8, name, "config")) {
        return field.?.root != .object or field.?.root.object.count() != 0;
    }
    if (comptime T == @FieldType(RequestTool, "function") and std.mem.eql(u8, name, "description"))
        return field.?.len != 0;
    if (comptime T == RequestFields and std.mem.eql(u8, name, "tools")) return field.len != 0;
    if (comptime T == Config and std.mem.eql(u8, name, "stop")) return field.len != 0;
    if (comptime T == ToolChoice and std.mem.eql(u8, name, "allowed")) return field.len != 0;
    return true;
}

// The records above contain only immutable leaves, optional fields and slices.
// These private codecs specialize at comptime; each field is allocated once,
// and a failed constructor releases precisely the initialized prefix.
fn readPayload(comptime T: type, allocator: Allocator, input: PayloadValue) Error!T {
    var context: json.CloneContext = .{ .allocator = allocator };
    defer context.deinit();
    const owned = if (input.storage != null) input.retain() else try context.clone(input);
    defer owned.free();
    return readAs(.graph, T, allocator, owned.root, false, owned.storage);
}

fn read(comptime T: type, allocator: Allocator, input: Value, native: bool) Error!T {
    return readAs(.runtime, T, allocator, input, native, null);
}

fn readAs(comptime projection: Projection, comptime T: type, allocator: Allocator, input: Projected(projection), native: bool, owner: ?*Payload) Error!T {
    if (T == Reasoning and !native) return readReasoning(projection, allocator, input, owner);
    if (T == SystemContent) return switch (input) {
        .string => .{ .text = try allocator.dupe(u8, input.string) },
        .array => .{ .parts = try readAs(projection, []const ContentPart, allocator, input, native, owner) },
        else => error.InvalidResponse,
    };
    if (T == JsonValue) {
        if (projection == .runtime) return JsonValue.fromValue(allocator, input);
        try json.validate(input, 0);
        return (PayloadValue{ .root = input, .storage = owner }).retain();
    }
    if (T == Value) {
        if (projection == .runtime) return inference.cloneJsonValue(allocator, input);
        var scratch: std.heap.ArenaAllocator = .init(allocator);
        defer scratch.deinit();
        const value = graph.toValue(scratch.allocator(), input) catch |err| return graphError(err);
        return inference.cloneJsonValue(allocator, value);
    }
    if (T == Usage) {
        if (input != .object) return .{};
        var scratch: std.heap.ArenaAllocator = .init(allocator);
        defer scratch.deinit();
        const value = if (projection == .runtime) input else graph.toValue(scratch.allocator(), input) catch |err| return graphError(err);
        const usage = Usage.fromValue(scratch.allocator(), value) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidResponse,
        };
        return usage.clone(allocator);
    }
    if (T == []const scores.TokenLogprob) {
        var scratch: std.heap.ArenaAllocator = .init(allocator);
        defer scratch.deinit();
        const value = if (projection == .runtime) input else graph.toValue(scratch.allocator(), input) catch |err| return graphError(err);
        return (if (native) scores.fromNativeValue(allocator, value) else scores.fromValue(allocator, value)) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidResponse,
        };
    }
    if (T == []const Continuation and input == .null) return &.{};
    if (T == []const Adaptation and input == .null) return &.{};
    return switch (@typeInfo(T)) {
        .bool => if (input == .bool) input.bool else error.InvalidResponse,
        .float => switch (input) {
            .float => input.float,
            .integer => @floatFromInt(input.integer),
            else => error.InvalidResponse,
        },
        .@"enum" => if (input == .string)
            std.meta.stringToEnum(T, input.string) orelse error.InvalidResponse
        else
            error.InvalidResponse,
        .optional => |info| if (input == .null) null else try readAs(projection, info.child, allocator, input, native, owner),
        .pointer => |info| if (info.child == u8)
            if (input == .string) try allocator.dupe(u8, input.string) else error.InvalidResponse
        else
            readSlice(projection, T, allocator, input, native, owner),
        .@"struct" => readRecord(projection, T, allocator, input, native, false, owner),
        .@"union" => readPart(projection, T, allocator, input, native, owner),
        else => @compileError("unsupported SDK field: " ++ @typeName(T)),
    };
}

// Record import deliberately differs from the genuine typed constructor:
// legacy off discards budgets/summary; a missing effort defaults to medium.
fn readReasoning(comptime projection: Projection, allocator: Allocator, input: Projected(projection), owner: ?*Payload) Error!Reasoning {
    if (input != .object) return error.InvalidResponse;
    const enabled = input.object.get("enabled") orelse Projected(projection).null;
    var effort = input.object.get("effort") orelse Projected(projection){
        .string = if (enabled == .bool and !enabled.bool) "off" else "medium",
    };
    if (effort == .string) {
        if (std.mem.eql(u8, effort.string, "off")) return .{ .effort = .off };
        if (std.mem.eql(u8, effort.string, "adaptive")) effort = .{ .string = "medium" };
    }
    var result: Reasoning = .{ .effort = try readAs(projection, ?ReasoningEffort, allocator, effort, false, owner) };
    errdefer release(Reasoning, allocator, result);
    const budget = input.object.get("thinking_budget") orelse input.object.get("budget") orelse Projected(projection).null;
    result.thinking_budget = try readAs(projection, Value, allocator, budget, false, owner);
    const summary = input.object.get("summary") orelse Projected(projection).null;
    result.summary = try readAs(projection, ?ReasoningSummary, allocator, summary, false, owner);
    try validateReasoning(allocator, &result);
    std.debug.assert(result.effort != null);
    std.debug.assert(result.effort != .off);
    return result;
}

fn readSlice(comptime projection: Projection, comptime T: type, allocator: Allocator, input: Projected(projection), native: bool, owner: ?*Payload) Error!T {
    if (T == []const []const u8 and input == .string) {
        const out = try allocator.alloc([]const u8, 1);
        errdefer allocator.free(out);
        out[0] = try allocator.dupe(u8, input.string);
        return out;
    }
    if (input != .array) return error.InvalidResponse;
    const Child = @typeInfo(T).pointer.child;
    const out = try allocator.alloc(Child, input.array.items.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |item| release(Child, allocator, item);
        allocator.free(out);
    }
    for (input.array.items, out) |item, *target| {
        target.* = if (Child == Part and item != .object) block: {
            if (projection == .runtime) break :block try messagePartText(allocator, item);
            var scratch: std.heap.ArenaAllocator = .init(allocator);
            defer scratch.deinit();
            const value = graph.toValue(scratch.allocator(), item) catch |err| return graphError(err);
            break :block try messagePartText(allocator, value);
        } else try readAs(projection, Child, allocator, item, native, owner);
        initialized += 1;
    }
    return out;
}

fn messagePartText(allocator: Allocator, input: Value) Error!Part {
    var text: std.Io.Writer.Allocating = .init(allocator);
    errdefer text.deinit();
    pyjson.writeStr(&text.writer, input) catch |err| return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        else => error.InvalidResponse,
    };
    return .{ .text = .{ .text = try text.toOwnedSlice() } };
}

fn readRecord(comptime projection: Projection, comptime T: type, allocator: Allocator, input: Projected(projection), native: bool, comptime ignore_detail: bool, owner: ?*Payload) Error!T {
    if (input != .object) return error.InvalidResponse;
    const info = @typeInfo(T).@"struct";
    var out: T = undefined;
    var initialized: usize = 0;
    errdefer inline for (info.field_names, info.field_types, 0..) |name, FT, index| {
        if (index < initialized) release(FT, allocator, @field(out, name));
    };
    inline for (info.field_names, info.field_types, 0..) |name, FT, index| {
        if (comptime T == MediaPart and std.mem.eql(u8, name, "detail")) {
            if (comptime ignore_detail) {
                @field(out, name) = null;
                initialized += 1;
                continue;
            }
        }
        if (input.object.get(name)) |field| {
            if (comptime T == @FieldType(RequestTool, "function") and std.mem.eql(u8, name, "parameters"))
                if (field == .null) return error.InvalidResponse;
            if (comptime T == Config and std.mem.eql(u8, name, "stop")) {
                @field(out, name) = if (field == .null) &.{} else try readAs(projection, FT, allocator, field, native, owner);
            } else if (comptime T == Adaptation and
                (std.mem.eql(u8, name, "asked") or std.mem.eql(u8, name, "applied")))
            {
                @field(out, name) = if (projection == .runtime)
                    try PayloadValue.fromPythonValue(allocator, field)
                else
                    (PayloadValue{ .root = field, .storage = owner }).retain();
            } else @field(out, name) = try readAs(projection, FT, allocator, field, native, owner);
        } else if (comptime T == Continuation and std.mem.eql(u8, name, "data")) {
            @field(out, name) = try JsonValue.fromValue(allocator, .{ .object = .empty });
        } else if (comptime info.field_attrs[index].default_value_ptr != null) {
            const default: *const FT = @ptrCast(@alignCast(info.field_attrs[index].default_value_ptr.?));
            @field(out, name) = try copy(FT, allocator, default.*);
        } else return error.InvalidResponse;
        initialized += 1;
    }
    try validateRecord(T, allocator, &out);
    return out;
}

fn readPart(comptime projection: Projection, comptime T: type, allocator: Allocator, input: Projected(projection), native: bool, owner: ?*Payload) Error!T {
    if (input != .object) return error.InvalidResponse;
    const kind = input.object.get("type") orelse return error.InvalidResponse;
    if (kind != .string) return error.InvalidResponse;
    const Tag = std.meta.Tag(T);
    const tag = std.meta.stringToEnum(Tag, kind.string) orelse return error.InvalidResponse;
    switch (tag) {
        inline else => |known| {
            const name = @tagName(known);
            const Fields = @FieldType(T, name);
            const ignore_detail = Fields == MediaPart and known != .image;
            const record = try readRecord(projection, Fields, allocator, input, native, ignore_detail, owner);
            return @unionInit(T, name, record);
        },
    }
}

const CopyContext = struct {
    allocator: Allocator,
    runtime_cloner: runtime.CloneContext,
    json_cloner: json.CloneContext,
    share_owned_json: bool = false,

    fn deinit(self: *CopyContext) void {
        self.json_cloner.deinit();
        self.runtime_cloner.deinit();
    }
};

fn copy(comptime T: type, allocator: Allocator, input: T) Allocator.Error!T {
    var context: CopyContext = .{
        .allocator = allocator,
        .runtime_cloner = .init(allocator),
        .json_cloner = .{ .allocator = allocator },
    };
    defer context.deinit();
    return copyWithContext(T, &context, input);
}

fn copyWithContext(comptime T: type, context: *CopyContext, input: T) Allocator.Error!T {
    const allocator = context.allocator;
    if (T == JsonValue) {
        if (context.share_owned_json) if (input.storage) |storage| {
            storage.retain();
            return input;
        };
        return context.json_cloner.clone(input);
    }
    if (T == Value) return context.runtime_cloner.cloneValue(input);
    if (T == Usage) return input.clone(allocator);
    if (T == []const scores.TokenLogprob) return scores.clone(allocator, input);
    switch (@typeInfo(T)) {
        .bool, .float, .@"enum" => return input,
        .optional => |info| return if (input) |item| try copyWithContext(info.child, context, item) else null,
        .pointer => |info| {
            if (info.child == u8) return allocator.dupe(u8, input);
            const out = try allocator.alloc(info.child, input.len);
            var initialized: usize = 0;
            errdefer {
                for (out[0..initialized]) |item| release(info.child, allocator, item);
                allocator.free(out);
            }
            for (input, out) |item, *target| {
                target.* = try copyWithContext(info.child, context, item);
                initialized += 1;
            }
            return out;
        },
        .@"struct" => |info| {
            var out: T = undefined;
            var initialized: usize = 0;
            errdefer inline for (info.field_names, info.field_types, 0..) |name, FT, index| {
                if (index < initialized) release(FT, allocator, @field(out, name));
            };
            inline for (info.field_names, info.field_types) |name, FT| {
                @field(out, name) = try copyWithContext(FT, context, @field(input, name));
                initialized += 1;
            }
            return out;
        },
        .@"union" => switch (input) {
            inline else => |payload, tag| return @unionInit(T, @tagName(tag), try copyWithContext(@TypeOf(payload), context, payload)),
        },
        else => @compileError("unsupported SDK field: " ++ @typeName(T)),
    }
}

fn release(comptime T: type, allocator: Allocator, input: T) void {
    if (T == JsonValue) return input.free();
    if (T == Value) return runtime.freeValue(allocator, input);
    if (T == Usage) return input.free(allocator);
    if (T == []const scores.TokenLogprob) return scores.free(allocator, input);
    switch (@typeInfo(T)) {
        .bool, .float, .@"enum" => {},
        .optional => |info| if (input) |item| release(info.child, allocator, item),
        .pointer => |info| {
            if (info.child != u8) for (input) |item| release(info.child, allocator, item);
            allocator.free(input);
        },
        .@"struct" => |info| inline for (info.field_names, info.field_types) |name, FT| {
            release(FT, allocator, @field(input, name));
        },
        .@"union" => switch (input) {
            inline else => |payload| release(@TypeOf(payload), allocator, payload),
        },
        else => @compileError("unsupported SDK field: " ++ @typeName(T)),
    }
}

const Projection = enum { runtime, graph };
fn Projected(comptime kind: Projection) type {
    return if (kind == .runtime) Value else graph.Value;
}
fn ProjectedObject(comptime kind: Projection) type {
    return if (kind == .runtime) runtime.ObjectMap else graph.Record;
}

fn project(comptime T: type, arena: Allocator, input: T, native: bool) Error!Value {
    return projectAs(.runtime, T, arena, input, native);
}

fn projectAs(comptime kind: Projection, comptime T: type, arena: Allocator, input: T, native: bool) Error!Projected(kind) {
    if (T == SystemContent) return switch (input) {
        .text => |text| projectAs(kind, []const u8, arena, text, native),
        .parts => |parts| projectAs(kind, []const ContentPart, arena, parts, native),
    };
    if (T == JsonValue) {
        if (kind == .runtime) return input.toValue(arena);
        return if (input.storage != null) input.root else try graph.clone(arena, input.root);
    }
    if (T == Value) {
        if (kind == .runtime) return input;
        var cloner: runtime.CloneContext = .init(arena);
        defer cloner.deinit();
        return graph.fromValue(arena, try cloner.cloneValue(input)) catch |err| return graphError(err);
    }
    if (T == Usage) {
        const value = try input.toValue(arena);
        return if (kind == .runtime) value else graph.fromValue(arena, value) catch |err| return graphError(err);
    }
    if (T == []const scores.TokenLogprob) {
        const value = try (if (native) scores.toNativeValue(arena, input) else scores.toValue(arena, input));
        if (kind == .runtime) return value;
        // Score codecs borrow token text. The graph record must outlive SDK
        // headers, while its numeric/custom counter leaves are immutable.
        var cloner: runtime.CloneContext = .init(arena);
        defer cloner.deinit();
        return graph.fromValue(arena, try cloner.cloneValue(value)) catch |err| return graphError(err);
    }
    switch (@typeInfo(T)) {
        .bool => return .{ .bool = input },
        .float => return .{ .float = input },
        .@"enum" => return .{ .string = @tagName(input) },
        .optional => |info| return if (input) |item| try projectAs(kind, info.child, arena, item, native) else .null,
        .pointer => |info| {
            if (info.child == u8) return .{ .string = if (kind == .runtime) input else try arena.dupe(u8, input) };
            var array: if (kind == .runtime) runtime.Array else graph.Array = .init(arena);
            try array.ensureTotalCapacity(input.len);
            for (input) |item| array.appendAssumeCapacity(try projectAs(kind, info.child, arena, item, native));
            if (kind == .runtime) return .{ .array = array };
            const header = try arena.create(graph.Array);
            header.* = array;
            return .{ .array = header };
        },
        .@"struct" => {
            var object: ProjectedObject(kind) = .empty;
            try projectRecord(kind, T, arena, input, native, &object, false);
            return projectedObject(kind, arena, object);
        },
        .@"union" => switch (input) {
            inline else => |payload, tag| {
                var object: ProjectedObject(kind) = .empty;
                try object.put(arena, "type", .{ .string = @tagName(tag) });
                const ignore_detail = @TypeOf(payload) == MediaPart and tag != .image;
                try projectRecord(kind, @TypeOf(payload), arena, payload, native, &object, ignore_detail);
                return projectedObject(kind, arena, object);
            },
        },
        else => @compileError("unsupported SDK field: " ++ @typeName(T)),
    }
}

fn projectedObject(comptime kind: Projection, arena: Allocator, object: ProjectedObject(kind)) Allocator.Error!Projected(kind) {
    if (kind == .runtime) return .{ .object = object };
    const header = try arena.create(graph.Record);
    header.* = object;
    return .{ .object = header };
}

fn projectRecord(
    comptime kind: Projection,
    comptime T: type,
    arena: Allocator,
    input: T,
    native: bool,
    object: *ProjectedObject(kind),
    comptime ignore_detail: bool,
) Error!void {
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types) |name, FT| {
        if (comptime T == MediaPart and std.mem.eql(u8, name, "detail")) {
            if (comptime ignore_detail) continue;
        }
        try projectField(kind, T, FT, arena, name, @field(input, name), native, object);
    }
}

fn projectField(
    comptime kind: Projection,
    comptime T: type,
    comptime FT: type,
    arena: Allocator,
    comptime name: []const u8,
    field: FT,
    native: bool,
    object: *ProjectedObject(kind),
) Error!void {
    if (!fieldPresent(T, FT, name, field, native, false)) return;
    try object.put(arena, name, try projectAs(kind, FT, arena, field, native));
}

// The SDK record type graph is finite: tool-result content excludes protocol
// parts. Opaque JSON/Python graphs terminate this traversal at their owner.
fn collectPayloadParents(comptime T: type, allocator: Allocator, input: T, parents: *std.ArrayList(*Payload)) Allocator.Error!void {
    if (T == JsonValue) {
        if (input.storage) |storage| try parents.append(allocator, storage);
        return;
    }
    if (T == Value or T == Usage) return;
    switch (@typeInfo(T)) {
        .optional => |info| if (input) |item| try collectPayloadParents(info.child, allocator, item, parents),
        .pointer => |info| {
            if (info.child != u8) for (input) |item| try collectPayloadParents(info.child, allocator, item, parents);
        },
        .@"struct" => |info| inline for (info.field_names, info.field_types) |name, FT| {
            try collectPayloadParents(FT, allocator, @field(input, name), parents);
        },
        .@"union" => switch (input) {
            inline else => |item| try collectPayloadParents(@TypeOf(item), allocator, item, parents),
        },
        else => {},
    }
}

fn graphError(err: graph.ConvertError) Error {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidResponse,
    };
}

fn validateRequest(allocator: Allocator, fields: *RequestFields) Error!void {
    try nonempty(fields.model);
    if (fields.messages.len == 0) return error.InvalidResponse;
    for (fields.messages) |message| {
        try validateMessage(&message);
        for (message.parts) |part| switch (part) {
            inline else => |payload| {
                var checked = payload;
                try validateRecord(@TypeOf(payload), allocator, &checked);
            },
        };
    }
    if (fields.system) |system| switch (system) {
        .text => |text| try nonempty(text),
        .parts => |parts| {
            if (parts.len == 0) return error.InvalidResponse;
            for (parts) |part| {
                if (part == .citation) return error.InvalidResponse;
                if (part == .data and part.data.probabilities != null) return error.InvalidResponse;
                switch (part) {
                    inline else => |payload| {
                        var checked = payload;
                        try validateRecord(@TypeOf(payload), allocator, &checked);
                    },
                }
            }
        },
    };
    // The cloned headers are exclusively owned even when JSON leaves share.
    const tools = @constCast(fields.tools);
    for (tools, 0..) |*tool, index| {
        const name = requestToolName(tool.*);
        try nonempty(name);
        for (tools[0..index]) |previous|
            if (std.mem.eql(u8, requestToolName(previous), name)) return error.InvalidResponse;
        switch (tool.*) {
            .function => |*function| {
                if (function.parameters == null) function.parameters = try defaultToolParameters(allocator);
                try jsonObject(function.parameters.?);
            },
            .builtin => |builtin| if (builtin.config) |config| try jsonObject(config),
        }
    }
    try validateConfig(allocator, &fields.config);
    if (fields.config.tool_choice) |choice| for (choice.allowed) |allowed| {
        var found = false;
        for (tools) |tool| if (std.mem.eql(u8, requestToolName(tool), allowed)) {
            found = true;
            break;
        };
        if (!found) return error.InvalidResponse;
    };
}

fn requestToolName(tool: RequestTool) []const u8 {
    return switch (tool) {
        inline else => |payload| payload.name,
    };
}

fn partHasPath(part: anytype) bool {
    return switch (part) {
        inline else => |payload| if (@TypeOf(payload) == MediaPart)
            payload.path != null
        else if (@TypeOf(payload) == ToolResultPart) nested: {
            for (payload.content) |content| if (partHasPath(content)) break :nested true;
            break :nested false;
        } else false,
    };
}

fn snapshotPart(
    comptime T: type,
    part: *T,
    allocator: Allocator,
    io: std.Io,
    maximum_file_bytes: usize,
) !void {
    switch (part.*) {
        inline else => |*payload| if (@TypeOf(payload.*) == MediaPart) {
            const path = payload.path orelse return;
            const bytes = try std.Io.Dir.cwd().readFileAlloc(
                io,
                path,
                allocator,
                .limited(maximum_file_bytes +| 1),
            );
            defer allocator.free(bytes);
            // Reader's limit is exclusive at EOF. Probe one extra byte so an
            // exactly-full file succeeds while growth beyond the cap refuses.
            if (bytes.len > maximum_file_bytes) return error.StreamTooLong;
            const encoder = std.base64.standard.Encoder;
            const groups = bytes.len / 3 + @intFromBool(bytes.len % 3 != 0);
            const encoded_len = std.math.mul(usize, groups, 4) catch return error.OutOfMemory;
            const encoded = try allocator.alloc(u8, encoded_len);
            _ = encoder.encode(encoded, bytes);
            // Publish only after both the read and encoding have succeeded.
            if (payload.data) |previous| allocator.free(previous);
            allocator.free(path);
            payload.path = null;
            payload.data = encoded;
            // Source's dataclass replacement revalidates the new media address.
            try validateMedia(payload);
        } else if (@TypeOf(payload.*) == ToolResultPart) {
            for (@constCast(payload.content)) |*content|
                try snapshotPart(ContentPart, content, allocator, io, maximum_file_bytes);
        },
    }
}

fn defaultToolParameters(allocator: Allocator) Error!JsonValue {
    var object: runtime.ObjectMap = .empty;
    defer object.deinit(allocator);
    try object.put(allocator, "type", .{ .string = "object" });
    try object.put(allocator, "properties", .{ .object = .empty });
    return JsonValue.fromValue(allocator, .{ .object = object });
}

fn jsonObject(input: JsonValue) Error!void {
    if (input.root != .object) return error.InvalidResponse;
    try json.validate(input.root, 0);
}

/// Normalize only builtin integral floats; genuine integer subclasses retain
/// their identity. The slot owns its old value and any replacement.
fn configInteger(allocator: Allocator, slot: *Value, minimum: ?i64) Error!void {
    if (slot.* == .null) return;
    const normalized = try normalizeConfigInteger(allocator, slot.*);
    if (std.meta.activeTag(normalized) != std.meta.activeTag(slot.*)) {
        runtime.freeValue(allocator, slot.*);
        slot.* = normalized;
    }
    if (minimum) |lower| {
        const order = @import("value.zig").numericOrder(slot.*, .{ .integer = lower }) catch return error.InvalidResponse;
        if (order != .eq and order != .gt) return error.InvalidResponse;
    }
}

/// Borrow integer inputs unchanged; allocate only when an integral float
/// needs a decimal representation beyond i64.
pub fn normalizeConfigInteger(allocator: Allocator, input: Value) Error!Value {
    const numbers = @import("value.zig");
    const number: ?f64 = switch (input) {
        .float => input.float,
        .custom => if (@import("python_value.zig").view(input)) |builtin|
            if (builtin == .float) builtin.float else null
        else
            null,
        else => null,
    };
    if (number) |floating| {
        var buffer: [310]u8 = undefined;
        const text = numbers.integralFloatText(floating, &buffer) orelse return error.InvalidResponse;
        return if (std.fmt.parseInt(i64, text, 10)) |small|
            .{ .integer = small }
        else |_|
            .{ .number_string = try allocator.dupe(u8, text) };
    }
    switch (input) {
        .null => {},
        .integer, .number_string => {
            _ = numbers.parseInteger(input) catch return error.InvalidResponse;
        },
        .custom => |custom| {
            const base = custom.integerValue() orelse return error.InvalidResponse;
            _ = numbers.parseInteger(base) catch return error.InvalidResponse;
        },
        else => return error.InvalidResponse,
    }
    return input;
}

fn validateReasoning(allocator: Allocator, reasoning: *Reasoning) Error!void {
    try configInteger(allocator, &reasoning.thinking_budget, 1);
    if (reasoning.effort == null) {
        if (reasoning.thinking_budget == .null) return error.InvalidResponse;
        reasoning.effort = .minimal;
        const levels = [_]ReasoningEffort{ .minimal, .low, .medium, .high, .xhigh, .max };
        const budgets = [_]i64{ 1024, 2048, 8192, 16384, 24576, 32768 };
        for (levels, budgets) |level, budget| {
            const order = @import("value.zig").numericOrder(reasoning.thinking_budget, .{ .integer = budget }) catch return error.InvalidResponse;
            if (order == .eq or order == .gt) reasoning.effort = level;
        }
    }
    if (reasoning.effort == .off) {
        if (reasoning.thinking_budget != .null or reasoning.summary != null) return error.InvalidResponse;
    }
    std.debug.assert(reasoning.effort != null);
}

fn validateCacheConfig(allocator: Allocator, cache: *CacheConfig) Error!void {
    if (cache.key) |key| try nonempty(key);
    if (cache.resource) |resource| try nonempty(resource);
    if (cache.prefix != null and cache.prefix_until_index != .null) return error.InvalidResponse;
    if (cache.mode == .off) {
        if (cache.retention != null or cache.key != null or cache.prefix != null or
            cache.prefix_until_index != .null or cache.resource != null) return error.InvalidResponse;
    }
    try configInteger(allocator, &cache.prefix_until_index, 0);
}

fn validateToolChoice(choice: *const ToolChoice) Error!void {
    for (choice.allowed) |allowed| try nonempty(allowed);
    if (choice.mode == .none) {
        if (choice.allowed.len != 0 or choice.parallel != null) return error.InvalidResponse;
    }
}

fn validateConfig(allocator: Allocator, config: *Config) Error!void {
    try configInteger(allocator, &config.max_tokens, 1);
    try configInteger(allocator, &config.top_k, 1);
    try configInteger(allocator, &config.seed, null);
    try configInteger(allocator, &config.logprobs, 0);
    if (config.temperature) |number| if (!(number >= 0 and number <= 2)) return error.InvalidResponse;
    if (config.top_p) |number| if (!(number >= 0 and number <= 1)) return error.InvalidResponse;
    inline for (.{ "frequency_penalty", "presence_penalty" }) |name| {
        if (@field(config, name)) |number| if (!(number >= -2 and number <= 2)) return error.InvalidResponse;
    }
    for (config.stop) |stop| try nonempty(stop);
    if (config.service_tier) |tier| try nonempty(tier);
    if (config.user_id) |user| try nonempty(user);
    if (config.tool_choice) |*choice| try validateToolChoice(choice);
    if (config.reasoning) |*reasoning| try validateReasoning(allocator, reasoning);
    if (config.cache) |*cache| try validateCacheConfig(allocator, cache);
    if (config.extensions) |extensions| try jsonObject(extensions);
    if (config.response_format) |format| try validateResponseFormat(format);
}

fn validateResponseFormat(format: JsonValue) Error!void {
    try jsonObject(format);
    const fields = format.root.object;
    const kind = fields.get("type") orelse return error.InvalidResponse;
    if (kind != .string) return error.InvalidResponse;
    const schema = std.mem.eql(u8, kind.string, "json_schema");
    if (!schema and !std.mem.eql(u8, kind.string, "json_object")) return error.InvalidResponse;
    for (fields.keys()) |key| {
        if (std.mem.eql(u8, key, "type")) continue;
        if (!schema) return error.InvalidResponse;
        if (!std.mem.eql(u8, key, "schema") and !std.mem.eql(u8, key, "name") and
            !std.mem.eql(u8, key, "strict")) return error.InvalidResponse;
    }
    if (schema) {
        const value = fields.get("schema") orelse return error.InvalidResponse;
        if (value != .object) return error.InvalidResponse;
        if (fields.get("name")) |name| {
            if (name != .string) return error.InvalidResponse;
            try nonempty(name.string);
        }
        if (fields.get("strict")) |strict| if (strict != .bool) return error.InvalidResponse;
    }
}

fn validateRecord(comptime T: type, allocator: Allocator, record: *T) Error!void {
    if (T == RequestFields) {
        try validateRequest(allocator, record);
    } else if (T == Config) {
        try validateConfig(allocator, record);
    } else if (T == Reasoning) {
        try validateReasoning(allocator, record);
    } else if (T == CacheConfig) {
        try validateCacheConfig(allocator, record);
    } else if (T == ToolChoice) {
        try validateToolChoice(record);
    } else if (T == ResponseFields) {
        try nonempty(record.model);
        if (record.id) |id| try nonempty(id);
        if (record.message.role != .assistant) return error.InvalidResponse;
        if (record.provider_data) |data| try json.validate(data.root, 0);
    } else if (T == Message) {
        try validateMessage(record);
    } else if (T == ToolCallPart) {
        try nonempty(record.id);
        try nonempty(record.name);
        try jsonObject(record.input);
    } else if (T == ToolResultPart) {
        try nonempty(record.id);
        if (record.name) |name| try nonempty(name);
        if (record.content.len == 0) return error.InvalidResponse;
        for (record.content) |part| if (part == .data and part.data.probabilities != null)
            return error.InvalidResponse;
    } else if (T == DataPart) {
        try json.validate(record.value.root, 0);
        if ((record.probabilities == null) != (record.method == null)) return error.InvalidResponse;
        if (record.probabilities) |*probabilities| try normalizeProbabilities(&probabilities.root);
    } else if (T == CitationPart) {
        if (record.text == null and record.title == null and record.url == null)
            return error.InvalidResponse;
        inline for (.{ "text", "title", "url" }) |name| {
            if (@field(record, name)) |text| try nonempty(text);
        }
    } else if (T == RefusalPart) {
        try nonempty(record.text);
    } else if (T == MediaPart) {
        try validateMedia(record);
    } else if (T == Continuation) {
        try nonempty(record.provider);
        try nonempty(record.kind);
        try jsonObject(record.data);
    } else if (T == Adaptation) {
        try nonempty(record.field);
        try nonempty(record.reason);
    }
}

fn validatePayload(allocator: Allocator, input: anytype) Error!void {
    var record = input;
    try validateRecord(@TypeOf(input), allocator, &record);
    if (comptime @hasField(@TypeOf(input), "continuation"))
        try validateContinuationList(allocator, input.continuation);
}

fn validateContinuationList(allocator: Allocator, states: []const Continuation) Error!void {
    for (states) |state| {
        var record = state;
        try validateRecord(Continuation, allocator, &record);
    }
}

fn nonempty(text: []const u8) Error!void {
    if (text.len == 0) return error.InvalidResponse;
}

fn validateMessage(message: *const Message) Error!void {
    if (message.parts.len == 0) return error.InvalidResponse;
    for (message.parts) |part| switch (message.role) {
        .tool => if (part != .tool_result) return error.InvalidResponse,
        .assistant => if (part == .tool_result) return error.InvalidResponse,
        .user, .developer => switch (part) {
            .tool_call, .tool_result, .thinking, .refusal, .citation => return error.InvalidResponse,
            .data => |data| if (data.probabilities != null) return error.InvalidResponse,
            else => {},
        },
    };
}

fn normalizeProbabilities(input: *@import("json_graph.zig").Value) Error!void {
    if (input.* != .object or input.object.count() == 0) return error.InvalidResponse;
    for (input.object.keys(), input.object.values()) |name, *distribution| {
        try nonempty(name);
        if (distribution.* != .object or distribution.object.count() == 0) return error.InvalidResponse;
        for (distribution.object.keys(), distribution.object.values()) |key, *probability| {
            try nonempty(key);
            const number = switch (probability.*) {
                .integer => |value| @as(f64, @floatFromInt(value)),
                .float => |value| value,
                .number_string => |value| std.fmt.parseFloat(f64, value) catch return error.InvalidResponse,
                else => return error.InvalidResponse,
            };
            if (!std.math.isFinite(number) or number < 0 or number > 1) return error.InvalidResponse;
            // The payload arena owns the old immutable leaf until final release.
            probability.* = .{ .float = number };
        }
    }
}

fn validateMedia(media: *const MediaPart) Error!void {
    try nonempty(media.media_type);
    var addresses: u8 = 0;
    inline for (.{ "data", "url", "file_id", "path" }) |name| {
        if (@field(media, name)) |text| {
            try nonempty(text);
            addresses += 1;
        }
    }
    if (addresses != 1) return error.InvalidResponse;
    if (media.data) |data| try validateBase64(data);
}

fn validateBase64(raw: []const u8) Error!void {
    var data = raw;
    if (std.mem.startsWith(u8, data, "data:")) {
        if (std.mem.indexOf(u8, data, ";base64,")) |index| data = data[index + 8 ..];
    }
    var count: usize = 0;
    var padding: u8 = 0;
    var saw_space = false;
    var split = false;
    var index: usize = 0;
    while (index < data.len) {
        const length = std.unicode.utf8ByteSequenceLength(data[index]) catch return error.InvalidResponse;
        if (length > data.len - index) return error.InvalidResponse;
        const cp = std.unicode.utf8Decode(data[index..][0..length]) catch return error.InvalidResponse;
        index += length;
        if (@import("unicode.zig").isSpace(cp)) {
            saw_space = true;
            // DSPy splits on all Python whitespace only if strip() changed
            // the payload or an ASCII whitespace separator is present.
            if (count == 0 or index == data.len or cp == ' ' or (cp >= '\t' and cp <= '\r'))
                split = true;
            continue;
        }
        count += 1;
        if (cp == '=') {
            padding += 1;
            if (padding > 2) return error.InvalidResponse;
        } else {
            if (padding != 0) return error.InvalidResponse;
            if (!(cp >= 'A' and cp <= 'Z') and !(cp >= 'a' and cp <= 'z') and
                !(cp >= '0' and cp <= '9') and cp != '+' and cp != '/') return error.InvalidResponse;
        }
    }
    if ((saw_space and !split) or count % 4 != 0) return error.InvalidResponse;
}

test "SDK requests apply Python whitespace rules to inline media" {
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    const cases = [_]struct { data: []const u8, valid: bool }{
        .{ .data = "A\u{a0}QJ=", .valid = false },
        .{ .data = "data:audio/pcm;base64,A\u{a0}QJ=", .valid = false },
        .{ .data = "\u{a0}AQJ=", .valid = true },
        .{ .data = "AQJ=\u{a0}\u{2003}", .valid = true },
        .{ .data = "A\u{a0}QJ=\n", .valid = true },
        .{ .data = "A\u{a0} QJ=", .valid = true },
        .{ .data = "A\u{a0}QJ=\u{2003}", .valid = true },
    };
    for (cases) |case| {
        const request = Request.fromFields(arena, &.{
            .model = "gemini-live-native-audio",
            .messages = &.{.{ .role = .user, .parts = &.{.{ .audio = .{
                .media_type = "audio/pcm",
                .data = case.data,
            } }} }},
        });
        if (case.valid) {
            const accepted = try request;
            defer accepted.release();
            try std.testing.expectEqualStrings(case.data, accepted.fields.messages[0].parts[0].audio.data.?);
        } else {
            if (request) |accepted| {
                accepted.release();
                return error.TestUnexpectedResult;
            } else |err| try std.testing.expectEqual(error.InvalidResponse, err);
        }
    }
}

test "SDK request snapshot freezes real files and retains no-path request identity" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var directory = testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(testing.io, .{ .sub_path = "media", .data = "\x00\xffhello" });
    const path = try directory.dir.realPathFileAlloc(testing.io, "media", allocator);
    defer allocator.free(path);
    const payload = try JsonValue.fromValue(allocator, .{ .object = .empty });
    defer payload.free();
    const original = try Request.fromFields(allocator, &.{
        .model = "snapshot",
        .messages = &.{.{ .role = .user, .parts = &.{
            .{ .image = .{ .path = path, .media_type = "image/png" } },
            .{ .data = .{ .value = payload } },
        } }},
    });
    defer original.release();
    const frozen = try original.snapshot(allocator, testing.io, 7);
    defer frozen.release();
    try testing.expect(frozen != original);
    try testing.expectEqualStrings(path, original.fields.messages[0].parts[0].image.path.?);
    const image = frozen.fields.messages[0].parts[0].image;
    try testing.expect(image.path == null);
    try testing.expectEqualStrings("AP9oZWxsbw==", image.data.?);
    try testing.expect(frozen.fields.messages[0].parts[1].data.value.root.object == payload.root.object);
    try directory.dir.writeFile(testing.io, .{ .sub_path = "media", .data = "new" });
    const fresh = try original.snapshot(allocator, testing.io, 7);
    defer fresh.release();
    try testing.expectEqualStrings("bmV3", fresh.fields.messages[0].parts[0].image.data.?);
    try testing.expectEqualStrings("AP9oZWxsbw==", image.data.?);
    const retained = try frozen.snapshot(allocator, testing.io, 0);
    defer retained.release();
    try testing.expect(retained == frozen);
}

test "SDK request snapshot covers system and tool media with transactional bounded failures" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var directory = testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(testing.io, .{ .sub_path = "media", .data = "abc" });
    const path = try directory.dir.realPathFileAlloc(testing.io, "media", allocator);
    defer allocator.free(path);
    const media: MediaPart = .{ .path = path, .media_type = "application/octet-stream" };
    const request = try Request.fromFields(allocator, &.{
        .model = "snapshot",
        .messages = &.{.{ .role = .tool, .parts = &.{.{ .tool_result = .{
            .id = "call",
            .content = &.{
                .{ .audio = media }, .{ .video = media }, .{ .binary = media },
            },
        } }} }},
        .system = .{ .parts = &.{.{ .document = media }} },
    });
    defer request.release();
    try testing.expectError(error.StreamTooLong, request.snapshot(allocator, testing.io, 2));
    try testing.expectEqualStrings(path, request.fields.system.?.parts[0].document.path.?);
    const frozen = try request.snapshot(allocator, testing.io, 3);
    defer frozen.release();
    for (frozen.fields.messages[0].parts[0].tool_result.content) |part| switch (part) {
        inline else => |value| if (@TypeOf(value) == MediaPart) {
            try testing.expect(value.path == null);
            try testing.expectEqualStrings("YWJj", value.data.?);
        },
    };
    try testing.expect(frozen.fields.system.?.parts[0].document.path == null);
    try testing.expectEqualStrings("YWJj", frozen.fields.system.?.parts[0].document.data.?);
    try directory.dir.deleteFile(testing.io, "media");
    try testing.expectError(error.FileNotFound, request.snapshot(allocator, testing.io, 3));
    try directory.dir.writeFile(testing.io, .{ .sub_path = "media", .data = "" });
    try testing.expectError(error.InvalidResponse, request.snapshot(allocator, testing.io, 0));
    const retained = try frozen.snapshot(allocator, testing.io, 0);
    defer retained.release();
    try testing.expect(retained == frozen);
}

test "SDK request snapshot unwinds every allocation failure after partial media replacement" {
    const testing = std.testing;
    var directory = testing.tmpDir(.{});
    defer directory.cleanup();
    try directory.dir.writeFile(testing.io, .{ .sub_path = "media", .data = "abc" });
    const path = try directory.dir.realPathFileAlloc(testing.io, "media", testing.allocator);
    defer testing.allocator.free(path);
    const Check = struct {
        fn run(allocator: Allocator, local_path: []const u8) !void {
            const request = try Request.fromFields(allocator, &.{
                .model = "snapshot",
                .messages = &.{.{ .role = .user, .parts = &.{
                    .{ .image = .{ .path = local_path, .media_type = "image/png" } },
                    .{ .document = .{ .path = local_path, .media_type = "application/pdf" } },
                } }},
            });
            defer request.release();
            const frozen = try request.snapshot(allocator, testing.io, 3);
            defer frozen.release();
            try testing.expectEqualStrings("YWJj", frozen.fields.messages[0].parts[0].image.data.?);
            try testing.expectEqualStrings("YWJj", frozen.fields.messages[0].parts[1].document.data.?);
            try testing.expectEqualStrings(local_path, request.fields.messages[0].parts[0].image.path.?);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Check.run, .{path});
}

test "SDK tool and continuation aliases survive growth and independent owning clones" {
    const allocator = std.testing.allocator;
    const independent = block: {
        const payload = try JsonValue.fromValue(allocator, .{ .object = .empty });
        defer payload.free();
        const states = [_]Continuation{.{ .provider = "fixture", .kind = "state", .data = payload }};
        const response = try Response.fromFields(allocator, &.{
            .model = "tools",
            .finish_reason = .tool_call,
            .message = .{ .role = .assistant, .continuation = &states, .parts = &.{
                .{ .tool_call = .{ .id = "one", .name = "lookup", .input = payload, .continuation = &states } },
                .{ .tool_call = .{ .id = "two", .name = "lookup", .input = payload, .continuation = &states } },
                .{ .data = .{ .value = payload, .continuation = &states } },
            } },
            .provider_data = payload,
        });
        defer response.release();
        const clone_object = try response.clone(allocator);
        errdefer clone_object.release();
        const items = try payload.storage.?.allocator().create(@import("json_graph.zig").Array);
        items.* = .init(payload.storage.?.allocator());
        try payload.root.object.put(payload.storage.?.allocator(), "items", .{ .array = items });
        for (0..100) |index| try items.append(.{ .integer = @intCast(index) });
        const parts = response.fields.message.parts;
        try std.testing.expect(parts[0].tool_call.input.root.object == payload.root.object);
        try std.testing.expect(parts[1].tool_call.input.root.object == payload.root.object);
        try std.testing.expect(parts[2].data.value.root.object == payload.root.object);
        try std.testing.expect(response.fields.provider_data.?.root.object == payload.root.object);
        try std.testing.expectEqual(@as(i64, 99), parts[1].tool_call.input.root.object.get("items").?.array.items[99].integer);
        try std.testing.expectEqual(@as(i64, 99), parts[1].tool_call.continuation[0].data.root.object.get("items").?.array.items[99].integer);
        try std.testing.expectEqual(@as(i64, 99), response.fields.message.continuation[0].data.root.object.get("items").?.array.items[99].integer);
        try std.testing.expect(!clone_object.fields.message.parts[0].tool_call.input.root.object.contains("items"));
        break :block clone_object;
    };
    defer independent.release();
    const parts = independent.fields.message.parts;
    const first = parts[0].tool_call.input;
    try std.testing.expect(first.root.object == parts[1].tool_call.input.root.object);
    try std.testing.expect(first.root.object == parts[2].data.value.root.object);
    try std.testing.expect(first.root.object == independent.fields.provider_data.?.root.object);
    try first.root.object.put(first.storage.?.allocator(), "escaped", .{ .bool = true });
    try std.testing.expect(parts[1].tool_call.continuation[0].data.root.object.get("escaped").?.bool);
    try std.testing.expect(parts[2].data.continuation[0].data.root.object.get("escaped").?.bool);
    try std.testing.expect(independent.fields.message.continuation[0].data.root.object.get("escaped").?.bool);
    try std.testing.expect(parts[1].tool_call.input.root.object.get("escaped").?.bool);
}

test "SDK tool and continuation construction clone and public output unwind allocation failures" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            const payload = try JsonValue.fromValue(allocator, .{ .object = .empty });
            defer payload.free();
            const states = [_]Continuation{.{ .provider = "fixture", .kind = "state", .data = payload }};
            const response = try Response.fromFields(allocator, &.{
                .model = "tools",
                .finish_reason = .tool_call,
                .message = .{ .role = .assistant, .continuation = &states, .parts = &.{
                    .{ .tool_call = .{ .id = "one", .name = "lookup", .input = payload, .continuation = &states } },
                    .{ .tool_call = .{ .id = "two", .name = "lookup", .input = payload, .continuation = &states } },
                } },
            });
            defer response.release();
            const independent = try response.clone(allocator);
            defer independent.release();
            var result = try @import("call_result.zig").CallResult.native(allocator, independent, .{});
            defer result.free(allocator);
            try std.testing.expect(result.responses[0] == independent);
        }
    };
    var stable: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(stable.allocator(), Check.run, .{});
}

test "SDK request ownership preserves input aliases and independent exact config clones" {
    const allocator = std.testing.allocator;
    const payload = try JsonValue.fromValue(allocator, .{ .object = .empty });
    defer payload.free();
    const request = try Request.fromFields(allocator, &.{
        .model = "request-model",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .data = .{ .value = payload } }} }},
        .config = .{
            .extensions = payload,
            .seed = .{ .float = -0x1p100 },
            .reasoning = .{ .thinking_budget = .{ .float = 24576 } },
        },
    });
    const retained = request.retain();
    request.release();
    defer retained.release();
    const independent = try retained.clone(allocator);
    defer independent.release();
    try payload.root.object.put(payload.storage.?.allocator(), "changed", .{ .bool = true });
    try std.testing.expect(retained.fields.messages[0].parts[0].data.value.root.object.contains("changed"));
    try std.testing.expect(retained.fields.config.extensions.?.root.object == payload.root.object);
    try std.testing.expect(independent.fields.config.extensions.?.root.object != payload.root.object);
    try std.testing.expect(independent.fields.config.extensions.?.root.object ==
        independent.fields.messages[0].parts[0].data.value.root.object);
    try std.testing.expect(!independent.fields.config.extensions.?.root.object.contains("changed"));
    try std.testing.expectEqual(ReasoningEffort.xhigh, independent.fields.config.reasoning.?.effort.?);
    try std.testing.expectEqualStrings("-1267650600228229401496703205376", independent.fields.config.seed.number_string);
}

test "SDK request records retain opaque aliases after request and input release" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            const record = block: {
                var input: std.heap.ArenaAllocator = .init(allocator);
                defer input.deinit();
                const parsed = try std.json.parseFromSliceLeaky(Value, input.allocator(),
                    \\{"shared":{"type":"object","properties":{},"extra":{"items":[]}},"format":{"type":"json_schema"}}
                , .{});
                const payload = try JsonValue.fromValue(allocator, parsed);
                defer payload.free();
                const shared: JsonValue = .{
                    .root = payload.root.object.get("shared").?,
                    .storage = payload.storage,
                };
                const format: JsonValue = .{
                    .root = payload.root.object.get("format").?,
                    .storage = payload.storage,
                };
                try format.root.object.put(payload.storage.?.allocator(), "schema", shared.root);
                const request = try Request.fromFields(allocator, &.{
                    .model = "request-record",
                    .messages = &.{.{ .role = .user, .parts = &.{.{ .data = .{ .value = shared } }} }},
                    .tools = &.{
                        .{ .function = .{ .name = "f", .parameters = shared } },
                        .{ .builtin = .{ .name = "b", .config = shared } },
                    },
                    .config = .{ .response_format = format, .extensions = shared },
                });
                defer request.release();
                break :block try request.toPayload(allocator);
            };
            defer record.free();
            const root = record.root.object;
            try std.testing.expectEqualStrings("request-record", root.get("model").?.string);
            const data = root.get("messages").?.array.items[0].object.get("parts").?.array.items[0]
                .object.get("value").?;
            const items = data.object.get("extra").?.object.get("items").?.array;
            try items.append(.{ .integer = 7 });
            const aliases = [_]graph.Value{
                root.get("tools").?.array.items[0].object.get("parameters").?,
                root.get("tools").?.array.items[1].object.get("config").?,
                root.get("config").?.object.get("response_format").?.object.get("schema").?,
                root.get("config").?.object.get("extensions").?,
            };
            for (aliases) |alias| {
                try std.testing.expect(alias.object == data.object);
                try std.testing.expectEqual(@as(i64, 7), alias.object.get("extra").?
                    .object.get("items").?.array.items[0].integer);
            }
        }
    };
    var stable: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(stable.allocator(), Check.run, .{});
}

test "SDK request imports retain owning aliases and snapshot borrowed records" {
    const Check = struct {
        fn run(allocator: Allocator, owning: bool) !void {
            const requests = block: {
                var input: std.heap.ArenaAllocator = .init(allocator);
                defer input.deinit();
                const parsed = try std.json.parseFromSliceLeaky(Value, input.allocator(),
                    \\{"model":"request-import","messages":[{"role":"user","parts":[{"type":"data","value":{"type":"object","properties":{},"extra":{"items":[]}}}]}],"tools":[{"type":"function","name":"f"},{"type":"builtin","name":"b"}],"config":{"response_format":{"type":"json_schema"}}}
                , .{});
                const record = try JsonValue.fromValue(allocator, parsed);
                defer record.free();
                const root = record.root.object;
                const shared = root.get("messages").?.array.items[0].object.get("parts").?
                    .array.items[0].object.get("value").?;
                const arena = record.storage.?.allocator();
                try root.get("tools").?.array.items[0].object.put(arena, "parameters", shared);
                try root.get("tools").?.array.items[1].object.put(arena, "config", shared);
                const config = root.get("config").?.object;
                try config.get("response_format").?.object.put(arena, "schema", shared);
                try config.put(arena, "extensions", shared);
                const view: PayloadValue = .{
                    .root = record.root,
                    .storage = if (owning) record.storage else null,
                };
                const first = try Request.fromPayload(allocator, view);
                errdefer first.release();
                const second = try Request.fromPayload(allocator, view);
                errdefer second.release();
                const imported = first.fields.messages[0].parts[0].data.value.root;
                try std.testing.expect((imported.object == shared.object) == owning);
                break :block [2]*const Request{ first, second };
            };
            defer for (requests) |request| request.release();
            const first = requests[0];
            try std.testing.expectEqualStrings("request-import", first.fields.model);
            const shared = first.fields.messages[0].parts[0].data.value.root;
            try shared.object.get("extra").?.object.get("items").?.array.append(.{ .integer = 7 });
            const aliases = [_]graph.Value{
                first.fields.tools[0].function.parameters.?.root,
                first.fields.tools[1].builtin.config.?.root,
                first.fields.config.response_format.?.root.object.get("schema").?,
                first.fields.config.extensions.?.root,
            };
            for (aliases) |alias| {
                try std.testing.expect(alias.object == shared.object);
                try std.testing.expectEqual(@as(i64, 7), alias.object.get("extra").?
                    .object.get("items").?.array.items[0].integer);
            }
            const other = requests[1].fields.messages[0].parts[0].data.value.root;
            try std.testing.expect((other.object == shared.object) == owning);
            try std.testing.expectEqual(@as(usize, if (owning) 1 else 0), other.object.get("extra").?.object.get("items").?.array.items.len);
        }
    };
    var stable: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(stable.allocator(), Check.run, .{true});
    try std.testing.checkAllAllocationFailures(stable.allocator(), Check.run, .{false});
}

test "SDK request reasoning record imports preserve legacy defaults without weakening typed construction" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { []const u8, ReasoningEffort, ?i64 }{
        .{ "{}", .medium, null },
        .{ "{\"enabled\":false,\"thinking_budget\":\"ignored\",\"summary\":17}", .off, null },
        .{ "{\"effort\":\"adaptive\",\"budget\":7}", .medium, 7 },
        .{ "{\"budget\":2000}", .medium, 2000 },
        .{ "{\"effort\":null,\"thinking_budget\":2000}", .minimal, 2000 },
        .{ "{\"thinking_budget\":null,\"budget\":2000}", .medium, null },
    };
    for (cases) |case| {
        const encoded = try std.fmt.allocPrint(
            allocator,
            "{{\"model\":\"fixture\",\"messages\":[{{\"role\":\"user\",\"parts\":[{{\"type\":\"text\",\"text\":\"next\"}}]}}],\"config\":{{\"reasoning\":{s}}}}}",
            .{case[0]},
        );
        defer allocator.free(encoded);
        const parsed = try std.json.parseFromSlice(Value, allocator, encoded, .{});
        defer parsed.deinit();
        const record = try JsonValue.fromValue(allocator, parsed.value);
        defer record.free();
        const runtime_request = try Request.fromValue(allocator, parsed.value);
        defer runtime_request.release();
        const graph_request = try Request.fromPayload(allocator, record);
        defer graph_request.release();
        for ([_]*const Request{ runtime_request, graph_request }) |request| {
            const reasoning = request.fields.config.reasoning.?;
            try std.testing.expectEqual(case[1], reasoning.effort.?);
            try std.testing.expectEqual(@as(?ReasoningSummary, null), reasoning.summary);
            if (case[2]) |budget| {
                try std.testing.expectEqual(budget, reasoning.thinking_budget.integer);
            } else {
                try std.testing.expect(reasoning.thinking_budget == .null);
            }
        }
    }
    const typed = try Request.fromFields(allocator, &.{
        .model = "fixture",
        .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "next" } }} }},
        .config = .{ .reasoning = .{ .thinking_budget = .{ .integer = 2000 } } },
    });
    defer typed.release();
    try std.testing.expectEqual(ReasoningEffort.minimal, typed.fields.config.reasoning.?.effort.?);
    try std.testing.expectEqual(@as(i64, 2000), typed.fields.config.reasoning.?.thinking_budget.integer);
}

test "SDK request rejects empty system part lists across public inputs" {
    const allocator = std.testing.allocator;
    const messages: []const Message = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "question" } }} }};
    const typed = Request.fromFields(allocator, &.{
        .model = "model",
        .messages = messages,
        .system = .{ .parts = &.{} },
    });
    defer if (typed) |request| request.release() else |_| {};
    try std.testing.expectError(error.InvalidResponse, typed);

    var scratch: std.heap.ArenaAllocator = .init(allocator);
    defer scratch.deinit();
    const record = try std.json.parseFromSliceLeaky(Value, scratch.allocator(),
        \\{"model":"model","messages":[{"role":"user","parts":[{"type":"text","text":"question"}]}],"system":[]}
    , .{});
    const value = Request.fromValue(allocator, record);
    defer if (value) |request| request.release() else |_| {};
    try std.testing.expectError(error.InvalidResponse, value);
    const payload = try JsonValue.fromValue(allocator, record);
    defer payload.free();
    const imported = Request.fromPayload(allocator, .{ .root = payload.root, .storage = payload.storage });
    defer if (imported) |request| request.release() else |_| {};
    try std.testing.expectError(error.InvalidResponse, imported);

    // Presence is about the part list, not its rendered text.
    const accepted = try Request.fromFields(allocator, &.{
        .model = "model",
        .messages = messages,
        .system = .{ .parts = &.{.{ .text = .{ .text = "" } }} },
    });
    defer accepted.release();
    try std.testing.expectEqualStrings("", accepted.fields.system.?.parts[0].text.text);
}

test "SDK request rejects contradictory config and unavailable tool selections" {
    const allocator = std.testing.allocator;
    const messages: []const Message = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "question" } }} }};
    const configs = [_]Config{
        .{ .tool_choice = .{ .allowed = &.{"missing"} } },
        .{ .reasoning = .{ .effort = .off, .thinking_budget = .{ .integer = 1 } } },
        .{ .cache = .{ .mode = .off, .key = "routing" } },
        .{ .cache = .{ .prefix = .stable, .prefix_until_index = .{ .integer = 0 } } },
        .{ .tool_choice = .{ .mode = .none, .parallel = false } },
        .{ .max_tokens = .{ .bool = true } },
        .{ .temperature = std.math.nan(f64) },
    };
    for (configs) |config| {
        try std.testing.expectError(error.InvalidResponse, Request.fromFields(allocator, &.{
            .model = "model",
            .messages = messages,
            .config = config,
        }));
    }
    try std.testing.expectError(error.InvalidResponse, Request.fromFields(allocator, &.{
        .model = "model",
        .messages = messages,
        .tools = &.{ .{ .function = .{ .name = "same" } }, .{ .builtin = .{ .name = "same" } } },
    }));
}

test "SDK request construction and clone unwind every allocation failure" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            const request = try Request.fromFields(allocator, &.{
                .model = "model",
                .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "question" } }} }},
                .system = .{ .parts = &.{.{ .text = .{ .text = "instructions" } }} },
                .tools = &.{.{ .function = .{ .name = "lookup" } }},
                .config = .{
                    .seed = .{ .float = -0x1p100 },
                    .tool_choice = .{ .allowed = &.{"lookup"}, .parallel = false },
                    .stop = &.{"end"},
                },
            });
            defer request.release();
            const clone = try request.clone(allocator);
            defer clone.release();
            try std.testing.expectEqualStrings("lookup", clone.fields.config.tool_choice.?.allowed[0]);
            try std.testing.expect(clone.fields.tools[0].function.parameters.?.root.object.contains("properties"));
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Check.run, .{});
}

test "SDK adaptation Any payloads preserve custom kinds and independent mutable clones" {
    const testing = std.testing;
    const python = @import("python_value.zig");
    var source_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer source_memory.deinit();
    const source = source_memory.allocator();
    const raw = try python.tuple(source, &.{
        try python.bytes(source, "\xff"),
        try python.bytearray(source, "abc"),
        try python.decimal(source, "1.25"),
        .{ .float = std.math.nan(f64) },
    });
    const input = try PayloadValue.fromPythonValue(testing.allocator, raw);
    defer input.free();
    try testing.expectError(error.InvalidResponse, JsonValue.fromValue(testing.allocator, raw));
    try testing.expectError(error.InvalidResponse, Response.fromFields(testing.allocator, &.{
        .model = "strict-data",
        .finish_reason = .stop,
        .message = .{ .role = .assistant, .parts = &.{.{ .data = .{ .value = input } }} },
    }));
    try testing.expectError(error.InvalidResponse, Response.fromFields(testing.allocator, &.{
        .model = "strict-provider",
        .finish_reason = .stop,
        .provider_data = input,
        .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "answer" } }} },
    }));
    source_memory.deinit();
    source_memory = .init(testing.allocator);

    const original = try Response.fromFields(testing.allocator, &.{
        .model = "any",
        .finish_reason = .stop,
        .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "answer" } }} },
        .adaptations = &.{.{
            .field = "config.extensions",
            .action = .substituted,
            .reason = "native Any",
            .asked = input,
            .applied = input,
        }},
    });
    defer original.release();
    const copied = try original.clone(testing.allocator);
    defer copied.release();
    const asked = original.fields.adaptations[0].asked.root.sequence;
    const applied = original.fields.adaptations[0].applied.root.sequence;
    const clone = copied.fields.adaptations[0].asked;
    try testing.expect(asked == input.root.sequence);
    try testing.expect(asked == applied);
    try testing.expect(clone.root.sequence != asked);
    try testing.expect(clone.root.sequence == copied.fields.adaptations[0].applied.root.sequence);
    try testing.expectEqualStrings("\xff", python.view(.{ .custom = clone.root.sequence.items.items[0].custom }).?.bytes);
    const decimal = python.view(.{ .custom = clone.root.sequence.items.items[2].custom }).?.decimal;
    try testing.expectEqualStrings("125", decimal.digits);
    try testing.expectEqual(@as(i64, -2), decimal.exponent);
    try testing.expect(std.math.isNan(clone.root.sequence.items.items[3].float));
    const mutable: Value = .{ .custom = asked.items.items[1].custom };
    try python.bytearrayExtend(input.storage.?.allocator(), mutable, "changed");
    try testing.expectEqualStrings("abcchanged", python.view(.{ .custom = applied.items.items[1].custom }).?.bytearray);
    try testing.expectEqualStrings("abc", python.view(.{ .custom = clone.root.sequence.items.items[1].custom }).?.bytearray);
}

test "SDK adaptation Any reader owns Python values and unwinds allocation failures" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            var source_memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer source_memory.deinit();
            const source = source_memory.allocator();
            var record: runtime.ObjectMap = .empty;
            try record.put(source, "field", .{ .string = "config.extensions" });
            try record.put(source, "action", .{ .string = "substituted" });
            try record.put(source, "reason", .{ .string = "Any reader" });
            const python = @import("python_value.zig");
            const tuple = try python.tuple(source, &.{try python.bytearray(source, "owned")});
            const dictionary = try python.dictionary(source, &.{.{ .key = .{ .integer = 7 }, .value = tuple }});
            const set = try python.set(source, &.{ .{ .integer = 1 }, .{ .integer = 7 } });
            const frozen = try python.frozenSet(source, &.{ .{ .integer = 1 }, .{ .integer = 7 } });
            const equal_tuple = try python.tuple(source, &.{try python.bytearray(source, "owned")});
            try record.put(source, "asked", try python.tuple(source, &.{
                tuple, tuple, dictionary, dictionary, set, set, frozen, frozen, equal_tuple,
            }));
            var rows: runtime.Array = .init(source);
            try rows.append(.{ .object = record });
            const records = try adaptationsFromValue(allocator, .{ .array = rows });
            defer freeAdaptations(allocator, records);
            const copied = try cloneAdaptations(allocator, records);
            defer freeAdaptations(allocator, copied);
            source_memory.deinit();
            source_memory = .init(std.testing.allocator);
            try std.testing.expect(copied[0].applied.root == .null);
            for ([_]PayloadValue{ records[0].asked, copied[0].asked }) |payload| {
                const items = payload.root.sequence.items.items;
                try std.testing.expect(items[0].sequence == items[1].sequence);
                try std.testing.expect(items[2].dictionary == items[3].dictionary);
                try std.testing.expect(items[2].dictionary.entries.items[0].value.sequence == items[0].sequence);
                try std.testing.expect(items[4].sequence == items[5].sequence);
                try std.testing.expect(items[6].sequence == items[7].sequence);
                try std.testing.expect(items[8].sequence != items[0].sequence);
            }
            const original = records[0].asked.root.sequence.items.items;
            const cloned = copied[0].asked.root.sequence.items.items;
            try std.testing.expect(original[0].sequence != cloned[0].sequence);
            const original_raw: Value = .{ .custom = original[0].sequence.items.items[0].custom };
            const copied_raw: Value = .{ .custom = cloned[0].sequence.items.items[0].custom };
            try python.bytearrayExtend(records[0].asked.storage.?.allocator(), original_raw, "x");
            try python.bytearrayExtend(copied[0].asked.storage.?.allocator(), copied_raw, "y");
            try std.testing.expectEqualStrings("ownedx", python.view(.{ .custom = original[1].sequence.items.items[0].custom }).?.bytearray);
            try std.testing.expectEqualStrings("ownedy", python.view(.{ .custom = cloned[3].dictionary.entries.items[0].value.sequence.items.items[0].custom }).?.bytearray);
        }
    };
    var backing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(backing.allocator(), Check.run, .{});
}

test "SDK adaptation Any clones preserve cycles while wire projection refuses them" {
    var source_memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer source_memory.deinit();
    const source = source_memory.allocator();
    const root = try graph.parse(source, "{\"items\":[]}");
    try root.object.put(source, "self", root);
    const adaptations = try cloneAdaptations(std.testing.allocator, &.{.{
        .field = "config.extensions",
        .action = .substituted,
        .reason = "cyclic Any",
        .asked = .borrowed(root),
        .applied = .borrowed(root),
    }});
    defer freeAdaptations(std.testing.allocator, adaptations);
    source_memory.deinit();
    source_memory = .init(std.testing.allocator);
    const asked = adaptations[0].asked;
    try std.testing.expect(asked.root.object == adaptations[0].applied.root.object);
    try std.testing.expect(asked.root.object == asked.root.object.get("self").?.object);
    try asked.root.object.get("items").?.array.append(.{ .integer = 7 });
    try std.testing.expectEqual(@as(i64, 7), adaptations[0].applied.root.object.get("items").?.array.items[0].integer);
    var scratch: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer scratch.deinit();
    try std.testing.expectError(error.InvalidResponse, adaptationsValue(scratch.allocator(), adaptations));
}

test "canonical SDK usage omits unknown counters but preserves reported zero" {
    const testing = std.testing;
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var response: ResponseFields = .{
        .model = "fixture",
        .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "answer" } }} },
        .finish_reason = .stop,
    };
    const unknown = try response.toValue(arena);
    try testing.expect(!unknown.object.contains("usage"));
    // Native persistence may retain the typed empty Usage. Canonical serde
    // omits it without turning the actual SDK Usage into a nullable field.
    const native = try response.toNativeValue(arena);
    try testing.expectEqual(@as(usize, 0), native.object.get("usage").?.object.count());
    const restored = try ResponseFields.fromNativeValue(testing.allocator, native);
    defer restored.free(testing.allocator);
    try testing.expect(!(try restored.toValue(arena)).object.contains("usage"));
    response.usage = .{ .input_tokens = .{ .integer = 0 } };
    const clone = try response.clone(testing.allocator);
    defer clone.free(testing.allocator);
    const zero = (try clone.toValue(arena)).object.get("usage").?;
    try testing.expectEqual(@as(usize, 1), zero.object.count());
    try testing.expectEqual(@as(i64, 0), zero.object.get("input_tokens").?.integer);
}

test "public SDK payload aliases survive source release and independent cache clones" {
    const testing = std.testing;
    const lm = @import("lm.zig");
    const Engine = struct {
        completion: [1]lm.Completion,
        fn complete(ptr: *anyopaque, _: Allocator, _: lm.Request) lm.Error![]const lm.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            return &self.completion;
        }
        const vtable: lm.LM.VTable = .{ .complete = complete };
    };
    var source_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    var source_live = true;
    defer if (source_live) source_memory.deinit();
    const source = source_memory.allocator();
    const root = try graph.parse(source,
        \\{"data":{"items":[0]},"probabilities":{"pick":{"yes":0.75,"no":0.25}},"empty":[],"other":[]}
    );
    const parts = try source.alloc(Part, 1);
    parts[0] = .{ .data = .{
        .value = .borrowed(root.object.get("data").?),
        .probabilities = .borrowed(root.object.get("probabilities").?),
        .method = .provider_classification,
    } };
    const original = try Response.fromFields(testing.allocator, &.{
        .model = "graph",
        .message = .{ .role = .assistant, .parts = parts },
        .provider_data = .borrowed(root),
        .adaptations = &.{.{
            .field = "config.extensions",
            .action = .substituted,
            .reason = "shared data",
            .asked = .borrowed(root.object.get("data").?),
            .applied = .borrowed(root),
        }},
        .finish_reason = .stop,
    });
    defer original.release();
    var engine: Engine = .{ .completion = .{.{
        .text = null,
        .response = original,
        .provider_data = original.fields.provider_data,
        .metadata = .{ .model = original.fields.model, .finish_reason = original.fields.finish_reason },
    }} };
    const model: lm.LM = .{ .ptr = &engine, .vtable = &Engine.vtable };
    var execution_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer execution_memory.deinit();
    const request: lm.Request = .{
        .messages = &.{.{ .role = .user, .content = .{ .text = "shared payload" } }},
        .options = .{ .n = 2 },
    };
    var cache: @import("cache.zig").CachedLM = .{ .gpa = testing.allocator, .inner = model };
    var cache_live = true;
    defer if (cache_live) cache.deinit();
    const first = try cache.lm().callResult(execution_memory.allocator(), request);
    defer first.free(execution_memory.allocator());
    try testing.expect(first.responses[0] == original);
    try testing.expect(first.responses[1] == original);
    source_memory.deinit();
    source_live = false;
    const second = try cache.lm().callResult(execution_memory.allocator(), request);
    defer second.free(execution_memory.allocator());
    try testing.expect(second.responses[0] != second.responses[1]);
    try testing.expect(second.responses[0] != original);
    const escaped = try second.clone(testing.allocator);
    defer escaped.free(testing.allocator);
    cache.deinit();
    cache_live = false;
    const data = first.responses[0].fields.message.parts[0].data;
    const provider = first.responses[0].fields.provider_data.?.root;
    try testing.expect(data.value.root.object == provider.object.get("data").?.object);
    try testing.expect(first.responses[0].fields.adaptations[0].asked.root.object == data.value.root.object);
    try testing.expect(first.responses[0].fields.adaptations[0].applied.root.object == provider.object);
    try testing.expect(data.probabilities.?.root.object == provider.object.get("probabilities").?.object);
    const items = data.value.root.object.get("items").?.array;
    for (0..100) |index| try items.append(.{ .integer = @intCast(index) });
    try testing.expectEqual(@as(usize, 101), provider.object.get("data").?.object.get("items").?.array.items.len);
    try testing.expectEqualStrings("{\"items\": [0]}", first.outputs.root.array.items[0].string);
    const copied = escaped.responses[0].fields.message.parts[0].data;
    const copied_provider = escaped.responses[0].fields.provider_data.?.root;
    try testing.expect(copied.value.root.object != data.value.root.object);
    try testing.expect(copied.value.root.object == copied_provider.object.get("data").?.object);
    try testing.expect(escaped.responses[0].fields.adaptations[0].asked.root.object == copied.value.root.object);
    try testing.expect(escaped.responses[0].fields.adaptations[0].applied.root.object == copied_provider.object);
    try testing.expectEqual(@as(usize, 1), copied.value.root.object.get("items").?.array.items.len);
    const empty = copied_provider.object.get("empty").?.array;
    const other = copied_provider.object.get("other").?.array;
    try testing.expect(empty != other);
    try empty.append(.{ .integer = 7 });
    try testing.expectEqual(@as(usize, 0), other.items.len);
}

test "history display joins only text messages and keeps local paths unresolved" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const media: MediaPart = .{
        .media_type = "image/png",
        .path = "unresolved-history-fixture.png",
    };
    const display = capture: {
        const request = try Request.fromFields(allocator, &.{
            .model = "history",
            .system = .{ .parts = &.{.{ .text = .{ .text = "system part" } }} },
            .messages = &.{
                .{ .role = .developer, .parts = &.{
                    .{ .text = .{ .text = "first" } },
                    .{ .text = .{ .text = "" } },
                    .{ .text = .{ .text = "last" } },
                } },
                .{ .role = .user, .parts = &.{
                    .{ .text = .{ .text = "photo" } },
                    .{ .image = media },
                } },
            },
        });
        defer request.release();
        break :capture try request.historyMessages(allocator);
    };
    defer display.free();
    const entries = display.root.array.items;
    const system = entries[0].object;
    try testing.expectEqualStrings("system", system.get("role").?.string);
    try testing.expectEqualStrings(
        "system part",
        system.get("content").?.array.items[0].object.get("text").?.string,
    );
    const developer = entries[1].object;
    try testing.expectEqualStrings("developer", developer.get("role").?.string);
    try testing.expectEqualStrings("first\n\nlast", developer.get("content").?.string);
    const parts = entries[2].object.get("content").?.array.items;
    try testing.expectEqualStrings("photo", parts[0].object.get("text").?.string);
    try testing.expectEqualStrings(media.path.?, parts[1].object.get("path").?.string);
    try testing.expect(parts[1].object.get("data") == null);
}
