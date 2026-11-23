//! DSPy execution metadata around ordered, unmodified SDK responses.
//! Returned results and canonical dump records own their data.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const sdk = @import("sdk_response.zig");
const numeric = @import("value.zig");
const pyjson = @import("pyjson.zig");
const graph = @import("json_graph.zig");
const projection = @import("call_result_projection.zig");
const tracker = @import("usage_tracker.zig");
const Payload = @import("request_cache.zig").Payload;
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
pub const ModelType = enum { auto, chat, responses, text };
pub const cache_format = "dspy-lm15-result-v2";
pub const Error = Allocator.Error || error{ InvalidCallResult, InvalidCost };

pub const CallResult = struct {
    responses: []const *const sdk.Response = &.{},
    outputs: sdk.PayloadValue,
    usage: sdk.PayloadValue,
    cost: sdk.PayloadValue = .{},
    response_model: ?[]const u8 = null,
    cache_hit: bool = false,
    cost_details: sdk.PayloadValue,

    /// Every default container has its own identity, including empty results.
    pub fn init(allocator: Allocator) Allocator.Error!CallResult {
        const storage = try Payload.create(allocator);
        errdefer storage.release();
        const arena = storage.allocator();
        storage.root = try outputArray(arena);
        const usage = try outputObject(arena);
        const details = try outputObject(arena);
        const outputs: sdk.PayloadValue = .{ .root = storage.root, .storage = storage };
        return .{
            .outputs = outputs,
            .usage = .{ .root = usage, .storage = outputs.retain().storage },
            .cost_details = .{ .root = details, .storage = outputs.retain().storage },
        };
    }

    pub const NativeOptions = struct { model_type: ModelType = .chat, logprobs: bool = false };

    pub fn native(allocator: Allocator, response: *const sdk.Response, options: NativeOptions) Error!CallResult {
        var result = try CallResult.init(allocator);
        errdefer result.free(allocator);
        const responses = try allocator.alloc(*const sdk.Response, 1);
        responses[0] = response.retain();
        result.responses = responses;
        const arena = result.outputs.storage.?.allocator();
        try result.outputs.root.array.append(try convenienceOutput(arena, &response.fields, options));
        try nativeUsage(arena, result.usage.root.object, &response.fields.usage);
        result.response_model = try allocator.dupe(u8, response.fields.model);
        return result;
    }

    pub fn clone(self: *const CallResult, allocator: Allocator) Allocator.Error!CallResult {
        var sdk_context: sdk.Response.CloneContext = .init(allocator);
        defer sdk_context.deinit();
        const responses = try cloneResponses(allocator, &sdk_context, self.responses);
        errdefer freeResponses(allocator, responses);
        const outputs = try sdk_context.clonePayload(self.outputs);
        errdefer outputs.free();
        const usage = try sdk_context.clonePayload(self.usage);
        errdefer usage.free();
        const cost = try sdk_context.clonePayload(self.cost);
        errdefer cost.free();
        const details = try sdk_context.clonePayload(self.cost_details);
        errdefer details.free();
        return .{
            .responses = responses,
            .outputs = outputs,
            .usage = usage,
            .cost = cost,
            .cost_details = details,
            .cache_hit = self.cache_hit,
            .response_model = if (self.response_model) |model| try allocator.dupe(u8, model) else null,
        };
    }

    pub fn free(self: *const CallResult, allocator: Allocator) void {
        freeResponses(allocator, self.responses);
        self.outputs.free();
        self.usage.free();
        self.cost.free();
        self.cost_details.free();
        if (self.response_model) |model| allocator.free(model);
    }

    /// An owning canonical graph record. Opaque SDK payloads and cost retain
    /// their original owners; plain metadata containers are fresh projections.
    pub fn dump(self: *const CallResult, allocator: Allocator) Error!sdk.PayloadValue {
        const records = try allocator.alloc(sdk.PayloadValue, self.responses.len);
        var initialized: usize = 0;
        defer {
            for (records[0..initialized]) |record| record.free();
            allocator.free(records);
        }
        var cloner: @import("sdk_json.zig").CloneContext = .{ .allocator = allocator };
        defer cloner.deinit();
        var fields: [4]sdk.PayloadValue = undefined;
        var field_count: usize = 0;
        defer for (fields[0..field_count]) |field| field.free();
        for ([_]sdk.PayloadValue{ self.outputs, self.usage, self.cost, self.cost_details }, &fields) |input, *field| {
            field.* = if (input.storage != null) input.retain() else try cloner.clone(input);
            field_count += 1;
        }
        var parents: std.ArrayList(*Payload) = .empty;
        defer parents.deinit(allocator);
        for (self.responses, records) |response, *record| {
            record.* = response.toPayload(allocator) catch |err| return sdkError(err);
            initialized += 1;
            try parents.append(allocator, record.storage.?);
        }
        const storage = try Payload.create(allocator);
        errdefer storage.release();
        const arena = storage.allocator();
        storage.root = try outputObject(arena);
        const responses = try outputArray(arena);
        try responses.array.ensureTotalCapacity(records.len);
        for (records) |record| responses.array.appendAssumeCapacity(record.root);
        const object = storage.root.object;
        try object.put(arena, "_dspy_format", .{ .string = cache_format });
        try object.put(arena, "responses", responses);
        inline for (.{ .{ "outputs", 0 }, .{ "usage", 1 }, .{ "cost_details", 3 } }) |field| {
            const projected = projection.projectOwned(arena, fields[field[1]].root, .plain) catch |err| return metadataError(err);
            try object.put(arena, field[0], projected.value);
            if (projected.borrows_payload) if (fields[field[1]].storage) |owner| try parents.append(allocator, owner);
        }
        try object.put(arena, "cost", fields[2].root);
        switch (fields[2].root) {
            .null, .bool, .integer, .float => {},
            else => if (fields[2].storage) |owner| try parents.append(allocator, owner),
        }
        try object.put(arena, "response_model", if (self.response_model) |model| .{ .string = try arena.dupe(u8, model) } else .null);
        storage.attachDependencies(try parents.toOwnedSlice(allocator));
        return .{ .root = storage.root, .storage = storage };
    }

    /// A cache hit retains SDK usage and cost, but public usage is empty.
    /// Like Source, load does not validate the informational format marker.
    /// Owning records retain original metadata identity. Explicitly borrowed
    /// records are snapshotted first, matching the SDK ownership boundary.
    pub fn load(allocator: Allocator, input: sdk.PayloadValue) Error!CallResult {
        var cloner: @import("sdk_json.zig").CloneContext = .{ .allocator = allocator };
        defer cloner.deinit();
        const owned = if (input.storage != null) input.retain() else try cloner.clone(input);
        defer owned.free();
        const record = owned.root;
        if (record != .object) return error.InvalidCallResult;
        const raw_responses = try required(record, "responses");
        const raw_outputs = try required(record, "outputs");
        if (raw_responses != .array) return error.InvalidCallResult;
        const responses = try allocator.alloc(*const sdk.Response, raw_responses.array.items.len);
        var initialized: usize = 0;
        errdefer {
            for (responses[0..initialized]) |response| response.release();
            allocator.free(responses);
        }
        for (raw_responses.array.items, responses) |raw, *response| {
            response.* = sdk.Response.fromPayload(allocator, .{ .root = raw, .storage = owned.storage }) catch |err| return sdkError(err);
            initialized += 1;
        }
        var parents: std.ArrayList(*Payload) = .empty;
        defer parents.deinit(allocator);
        if (owned.storage) |owner| try parents.append(allocator, owner);
        const storage = try dependentStorage(allocator, &parents);
        errdefer storage.release();
        const arena = storage.allocator();
        storage.root = projection.project(arena, raw_outputs, .attributes) catch |err| return metadataError(err);
        const usage = try outputObject(arena);
        const cost = try required(record, "cost");
        const details = record.object.get("cost_details") orelse try outputObject(arena);
        const model = try required(record, "response_model");
        if (model != .null and model != .string) return error.InvalidCallResult;
        const owned_model = if (model == .string) try allocator.dupe(u8, model.string) else null;
        const outputs: sdk.PayloadValue = .{ .root = storage.root, .storage = storage };
        return .{
            .responses = responses,
            .outputs = outputs,
            .usage = .{ .root = usage, .storage = outputs.retain().storage },
            .cost = .{ .root = cost, .storage = outputs.retain().storage },
            .cost_details = .{ .root = details, .storage = outputs.retain().storage },
            .response_model = owned_model,
            .cache_hit = true,
        };
    }
};

pub fn combine(allocator: Allocator, results: []const CallResult, model_type: ModelType) Error!CallResult {
    // Source accepts this argument but deliberately does not branch on it.
    _ = model_type;
    var response_count: usize = 0;
    var output_count: usize = 0;
    var parent_count: usize = 0;
    var structured = false;
    var response_model: ?[]const u8 = null;
    var model_selected = false;
    for (results) |result| {
        response_count = std.math.add(usize, response_count, result.responses.len) catch return error.OutOfMemory;
        if (result.outputs.root != .array) return error.InvalidCallResult;
        inline for (.{ result.outputs, result.usage, result.cost, result.cost_details }) |payload| {
            if (payload.storage != null)
                parent_count = std.math.add(usize, parent_count, 1) catch return error.OutOfMemory;
        }
        output_count = std.math.add(usize, output_count, result.outputs.root.array.items.len) catch return error.OutOfMemory;
        for (result.outputs.root.array.items) |output| if (output == .object) {
            structured = true;
        };
        if (!model_selected and result.responses.len != 0) {
            response_model = result.response_model;
            model_selected = true;
        }
    }
    const responses = try allocator.alloc(*const sdk.Response, response_count);
    var written_responses: usize = 0;
    errdefer {
        for (responses[0..written_responses]) |response| response.release();
        allocator.free(responses);
    }
    for (results) |result| for (result.responses) |response| {
        responses[written_responses] = response.retain();
        written_responses += 1;
    };
    const storage = block: {
        const parents = try allocator.alloc(*Payload, parent_count);
        errdefer allocator.free(parents);
        var written: usize = 0;
        for (results) |result| {
            inline for (.{ result.outputs, result.usage, result.cost, result.cost_details }) |payload| {
                if (payload.storage) |owner| {
                    parents[written] = owner;
                    written += 1;
                }
            }
        }
        std.debug.assert(written == parents.len);
        break :block try Payload.createDependent(allocator, parents);
    };
    errdefer storage.release();
    const arena = storage.allocator();
    storage.root = try outputArray(arena);
    const list = storage.root.array;
    try list.ensureTotalCapacity(output_count);
    for (results) |result| for (result.outputs.root.array.items) |output| {
        if (structured and output != .object) {
            const wrapped = try outputObject(arena);
            try wrapped.object.put(arena, "text", output);
            list.appendAssumeCapacity(wrapped);
        } else list.appendAssumeCapacity(output);
    };
    const outputs: sdk.PayloadValue = .{ .root = storage.root, .storage = storage };
    const metadata = try combineMetadata(arena, results);
    const model = if (response_model) |name| try allocator.dupe(u8, name) else null;
    return .{
        .responses = responses,
        .outputs = outputs,
        .usage = .{ .root = metadata.usage, .storage = outputs.retain().storage },
        .cost = .{ .root = metadata.cost, .storage = outputs.retain().storage },
        .cost_details = .{ .root = metadata.details, .storage = outputs.retain().storage },
        .response_model = model,
    };
}

const Metadata = struct { usage: graph.Value, cost: graph.Value, details: graph.Value };
fn combineMetadata(arena: Allocator, results: []const CallResult) Error!Metadata {
    const entries = try arena.alloc(?graph.Value, results.len);
    for (results, entries) |result, *entry|
        entry.* = tracker.prepareEntry(arena, result.usage.root) catch |err| return metadataError(err);
    var usage: graph.Value = .null;
    for (entries) |entry| if (entry) |value| {
        usage = tracker.merge(arena, usage, value) catch |err| return metadataError(err);
    };
    if (usage == .null) usage = try outputObject(arena);
    var cost: Value = .{ .integer = 0 };
    var complete_costs = results.len != 0;
    for (results) |result| if (result.cost.root == .null) {
        complete_costs = false;
    };
    // SUM is evaluated only when EVERY cost is non-null.
    if (complete_costs) for (results) |result| {
        const value = result.cost.toValue(arena) catch |err| return sdkError(err);
        cost = numeric.numericSum(arena, cost, value) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidCost,
        };
    };
    const details = try outputObject(arena);
    var details_present = false;
    for (results) |result| {
        if (graph.truthy(result.cost_details.root) catch |err| return metadataError(err)) {
            details_present = true;
            break;
        }
    }
    if (details_present) {
        const candidates = try outputArray(arena);
        try candidates.array.ensureTotalCapacity(results.len);
        for (results) |result| candidates.array.appendAssumeCapacity(result.cost_details.root);
        try details.object.put(arena, "candidates", candidates);
    }
    return .{
        .usage = usage,
        .cost = if (complete_costs) graph.fromValue(arena, cost) catch |err| return metadataError(err) else .null,
        .details = details,
    };
}

fn nativeUsage(arena: Allocator, object: *graph.Record, usage: *const @import("lm_usage.zig").Usage) Allocator.Error!void {
    var context: runtime.CloneContext = .init(arena);
    defer context.deinit();
    inline for (.{
        .{ "prompt_tokens", "input_tokens" },
        .{ "completion_tokens", "output_tokens" },
        .{ "total_tokens", "total_tokens" },
    }) |names| {
        try usageCount(arena, &context, object, names[0], @field(usage, names[1]));
    }
    if (usage.isKnown()) {
        inline for (.{
            .{ "prompt_tokens_details", .{
                .{ "cached_tokens", "cache_read_tokens" },
                .{ "cache_creation_tokens", "cache_write_tokens" },
                .{ "audio_tokens", "input_audio_tokens" },
            } },
            .{ "completion_tokens_details", .{
                .{ "reasoning_tokens", "reasoning_tokens" },
                .{ "audio_tokens", "output_audio_tokens" },
            } },
        }) |group| {
            var known = false;
            inline for (group[1]) |names| {
                if (@field(usage, names[1]) != .null) known = true;
            }
            var details: graph.Value = .null;
            if (known) {
                details = try outputObject(arena);
                inline for (group[1]) |names|
                    try usageCount(arena, &context, details.object, names[0], @field(usage, names[1]));
            }
            try object.put(arena, group[0], details);
        }
    } else std.debug.assert(object.count() == 0);
}

fn usageCount(arena: Allocator, context: *runtime.CloneContext, object: *graph.Record, name: []const u8, count: Value) Allocator.Error!void {
    if (count == .null) return;
    const copied = try context.cloneValue(count);
    const value: graph.Value = switch (copied) {
        .integer => |number| .{ .integer = number },
        .number_string => |text| .{ .number_string = text },
        .custom => |custom| .{ .custom = custom },
        else => unreachable, // Usage validates canonical nonnegative integers.
    };
    try object.put(arena, name, value);
}

fn convenienceOutput(allocator: Allocator, response: *const sdk.ResponseFields, options: CallResult.NativeOptions) Error!graph.Value {
    var text: std.Io.Writer.Allocating = .init(allocator);
    defer text.deinit();
    var thinking: std.Io.Writer.Allocating = .init(allocator);
    defer thinking.deinit();
    var has_text = false;
    var has_tools = false;
    var has_citations = false;
    for (response.message.parts) |part| switch (part) {
        .text => |content| {
            has_text = true;
            text.writer.writeAll(content.text) catch return error.OutOfMemory;
        },
        .data => |content| {
            has_text = true;
            content.value.writeJson(&text.writer) catch |err| return metadataError(err);
        },
        .thinking => |content| thinking.writer.writeAll(content.text) catch return error.OutOfMemory,
        .tool_call => {
            has_tools = true;
        },
        .citation => {
            has_citations = true;
        },
        else => {},
    };
    const visible: graph.Value = if (has_text) .{ .string = try text.toOwnedSlice() } else .null;
    const structured = options.model_type == .responses or thinking.written().len != 0 or
        has_tools or has_citations or options.logprobs;
    if (!structured) return visible;
    const object = try outputObject(allocator);
    try object.object.put(allocator, "text", visible);
    if (thinking.written().len != 0)
        try object.object.put(allocator, "reasoning_content", .{ .string = try thinking.toOwnedSlice() });
    if (has_tools) try object.object.put(allocator, "tool_calls", try toolCalls(allocator, response));
    if (has_citations) try object.object.put(allocator, "citations", try citations(allocator, response));
    if (options.logprobs) try object.object.put(allocator, "logprobs", try publicLogprobs(allocator, response, options.model_type));
    return object;
}

fn toolCalls(arena: Allocator, response: *const sdk.ResponseFields) Error!graph.Value {
    const array = try outputArray(arena);
    for (response.message.parts) |part| if (part == .tool_call) {
        const call = part.tool_call;
        var arguments: std.Io.Writer.Allocating = .init(arena);
        defer arguments.deinit();
        call.input.writeJsonAscii(&arguments.writer) catch |err| return metadataError(err);
        const function = try outputObject(arena);
        try function.object.put(arena, "name", .{ .string = try arena.dupe(u8, call.name) });
        try function.object.put(arena, "arguments", .{ .string = try arguments.toOwnedSlice() });
        const object = try outputObject(arena);
        try object.object.put(arena, "id", .{ .string = try arena.dupe(u8, call.id) });
        try object.object.put(arena, "type", .{ .string = "function" });
        try object.object.put(arena, "function", function);
        try array.array.append(object);
    };
    return array;
}

fn citations(arena: Allocator, response: *const sdk.ResponseFields) Allocator.Error!graph.Value {
    const array = try outputArray(arena);
    for (response.message.parts) |part| if (part == .citation) {
        const object = try outputObject(arena);
        inline for (.{ "text", "title", "url" }) |name| {
            if (@field(part.citation, name)) |text|
                try object.object.put(arena, name, .{ .string = try arena.dupe(u8, text) });
        }
        try array.array.append(object);
    };
    return array;
}

fn publicLogprobs(arena: Allocator, response: *const sdk.ResponseFields, mode: ModelType) Error!graph.Value {
    if (mode == .text) if (response.provider_data) |payload| {
        const provider = payload.root;
        if (provider == .object) if (provider.object.get("choices")) |choices| {
            if (choices == .array and choices.array.items.len != 0) {
                const first = choices.array.items[0];
                if (first == .object) if (first.object.get("logprobs")) |raw|
                    return graph.clone(arena, raw);
            }
        };
    };
    const content = try outputArray(arena);
    for (response.logprobs orelse &.{}) |token| {
        const object = try publicToken(arena, token.token, token.logprob, token.bytes);
        const top = try outputArray(arena);
        for (token.top) |alternative|
            try top.array.append(try publicToken(arena, alternative.token, alternative.logprob, alternative.bytes));
        try object.object.put(arena, "top_logprobs", top);
        try content.array.append(object);
    }
    const object = try outputObject(arena);
    try object.object.put(arena, "content", content);
    return object;
}

fn publicToken(arena: Allocator, token: []const u8, logprob: f64, bytes: ?[]const Value) Error!graph.Value {
    const object = try outputObject(arena);
    try object.object.put(arena, "token", .{ .string = try arena.dupe(u8, token) });
    try object.object.put(arena, "logprob", .{ .float = logprob });
    var data: graph.Value = .null;
    if (bytes) |items| {
        data = try outputArray(arena);
        try data.array.ensureTotalCapacity(items.len);
        for (items) |item| data.array.appendAssumeCapacity(switch (item) {
            .integer => |number| .{ .integer = number },
            else => return error.InvalidCallResult,
        });
    }
    try object.object.put(arena, "bytes", data);
    return object;
}

fn outputArray(arena: Allocator) Allocator.Error!graph.Value {
    const array = try arena.create(graph.Array);
    array.* = .init(arena);
    return .{ .array = array };
}

fn outputObject(arena: Allocator) Allocator.Error!graph.Value {
    const object = try arena.create(graph.Record);
    object.* = .empty;
    return .{ .object = object };
}

fn cloneResponses(allocator: Allocator, context: *sdk.Response.CloneContext, input: []const *const sdk.Response) Allocator.Error![]const *const sdk.Response {
    const out = try allocator.alloc(*const sdk.Response, input.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |response| response.release();
        allocator.free(out);
    }
    for (input, out) |response, *target| {
        target.* = try context.clone(response);
        initialized += 1;
    }
    return out;
}
fn freeResponses(allocator: Allocator, input: []const *const sdk.Response) void {
    for (input) |response| response.release();
    allocator.free(input);
}
test "SDK response identity and cross-response payload aliases survive combine and owning clone" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const combined = block: {
        var memory: std.heap.ArenaAllocator = .init(allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        var items: runtime.Array = .init(arena);
        try items.append(.{ .integer = 0 });
        var data: runtime.ObjectMap = .empty;
        try data.put(arena, "items", .{ .array = items });
        const payload = try sdk.JsonValue.fromValue(allocator, .{ .object = data });
        defer payload.free();
        const fields: sdk.ResponseFields = .{
            .model = "shared-response",
            .message = .{ .role = .assistant, .parts = &.{.{ .data = .{ .value = payload } }} },
            .finish_reason = .stop,
            .provider_data = payload,
        };
        const one = try sdk.Response.fromFields(allocator, &fields);
        defer one.release();
        const two = try sdk.Response.fromFields(allocator, &fields);
        defer two.release();
        var first = try CallResult.native(allocator, one, .{});
        defer first.free(allocator);
        try first.outputs.root.array.append(one.fields.provider_data.?.root);
        first.usage.free();
        first.usage = one.fields.provider_data.?.retain();
        first.cost = one.fields.provider_data.?.retain();
        first.cost_details.free();
        first.cost_details = one.fields.provider_data.?.retain();
        const individual = try first.clone(allocator);
        defer individual.free(allocator);
        const copied_root = individual.responses[0].fields.provider_data.?.root.object;
        try testing.expect(copied_root == individual.outputs.root.array.items[1].object);
        try testing.expect(copied_root == individual.usage.root.object);
        try testing.expect(copied_root == individual.cost.root.object);
        try testing.expect(copied_root == individual.cost_details.root.object);
        const second = try CallResult.native(allocator, two, .{});
        defer second.free(allocator);
        const result = try combine(allocator, &.{ first, second, first }, .chat);
        errdefer result.free(allocator);
        try testing.expect(result.responses[0] == one);
        try testing.expect(result.responses[1] == two);
        try testing.expect(result.responses[2] == one);
        break :block result;
    };
    defer combined.free(allocator);
    const cloned = try combined.clone(allocator);
    defer cloned.free(allocator);
    try testing.expect(cloned.responses[0] == cloned.responses[2]);
    try testing.expect(cloned.responses[0] != cloned.responses[1]);
    try testing.expect(cloned.responses[0] != combined.responses[0]);
    const original = combined.responses[0].fields.provider_data.?.root;
    const copied = cloned.responses[0].fields.provider_data.?.root;
    try testing.expect(original.object != copied.object);
    try testing.expect(original.object == combined.responses[1].fields.provider_data.?.root.object);
    try testing.expect(copied.object == cloned.responses[1].fields.provider_data.?.root.object);
    try testing.expect(copied.object == cloned.responses[0].fields.message.parts[0].data.value.root.object);
    try testing.expect(copied.object == cloned.outputs.root.array.items[1].object);
    try testing.expect(copied.object == cloned.outputs.root.array.items[4].object);
    const candidates = cloned.cost_details.root.object.get("candidates").?.array.items;
    try testing.expect(candidates[0].object == copied.object);
    try testing.expect(candidates[2].object == copied.object);
    const original_items = original.object.get("items").?.array;
    for (0..100) |index| try original_items.append(.{ .integer = @intCast(index) });
    try testing.expectEqual(@as(usize, 101), combined.responses[1].fields.provider_data.?.root.object.get("items").?.array.items.len);
    try testing.expectEqual(@as(usize, 1), copied.object.get("items").?.array.items.len);
    try testing.expectEqualStrings("{\"items\": [0]}", combined.outputs.root.array.items[0].object.get("text").?.string);
    try copied.object.get("items").?.array.append(.{ .integer = 999 });
    try testing.expectEqual(@as(usize, 2), cloned.outputs.root.array.items[1].object.get("items").?.array.items.len);
    try testing.expectEqual(@as(usize, 101), original_items.items.len);
}

test "combined mutable output rows retain aliases and lifetime independently of their source lists" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const escaped = block: {
        const response = try sdk.Response.fromFields(allocator, &.{
            .model = "output-alias",
            .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "before" } }} },
            .finish_reason = .stop,
        });
        defer response.release();
        const original = try CallResult.native(allocator, response, .{ .model_type = .responses });
        defer original.free(allocator);
        const arena = original.outputs.storage.?.allocator();
        const row = original.outputs.root.array.items[0];
        const items = try outputArray(arena);
        try items.array.append(.{ .integer = 1 });
        try row.object.put(arena, "items", items);
        const combined = try combine(allocator, &.{ original, original }, .responses);
        defer combined.free(allocator);
        const independent = try combined.clone(allocator);
        defer independent.free(allocator);
        try row.object.put(arena, "text", .{ .string = "after" });
        for (0..100) |index| try items.array.append(.{ .integer = @intCast(index) });
        try original.outputs.root.array.append(.{ .string = "extra" });
        try testing.expectEqual(@as(usize, 2), combined.outputs.root.array.items.len);
        try testing.expect(combined.outputs.root.array != original.outputs.root.array);
        try testing.expect(combined.outputs.root.array.items[0].object == row.object);
        try testing.expect(combined.outputs.root.array.items[1].object == row.object);
        const copied = independent.outputs.root.array.items;
        try testing.expect(copied[0].object == copied[1].object);
        try testing.expect(copied[0].object != row.object);
        try testing.expectEqualStrings("before", copied[0].object.get("text").?.string);
        try testing.expectEqual(@as(usize, 1), copied[0].object.get("items").?.array.items.len);
        break :block combined.outputs.retain();
    };
    defer escaped.free();
    const rows = escaped.root.array.items;
    try testing.expect(rows[0].object == rows[1].object);
    try testing.expectEqualStrings("after", rows[1].object.get("text").?.string);
    try testing.expectEqual(@as(usize, 101), rows[1].object.get("items").?.array.items.len);
}

test "empty output lists are independent and cyclic owning copies remain cyclic" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const original = try CallResult.init(allocator);
    defer original.free(allocator);
    const empty = try CallResult.init(allocator);
    defer empty.free(allocator);
    try original.outputs.root.array.append(original.outputs.root);
    const copied = try original.clone(allocator);
    defer copied.free(allocator);
    try testing.expectEqual(@as(usize, 0), empty.outputs.root.array.items.len);
    try testing.expect(original.outputs.root.array != empty.outputs.root.array);
    try testing.expect(copied.outputs.root.array != original.outputs.root.array);
    try testing.expect(copied.outputs.root.array.items[0].array == copied.outputs.root.array);
    try testing.expectError(error.InvalidCallResult, copied.dump(allocator));
}

test "mutable metadata defaults and whole-result clones retain genuine shared objects" {
    const allocator = std.testing.allocator;
    const python = @import("python_value.zig");
    var original = try CallResult.init(allocator);
    defer original.free(allocator);
    const empty = try CallResult.init(allocator);
    defer empty.free(allocator);
    try std.testing.expect(original.usage.root.object != empty.usage.root.object);
    try std.testing.expect(original.cost_details.root.object != empty.cost_details.root.object);
    const arena = original.outputs.storage.?.allocator();
    const shared = original.cost_details.root;
    const items = try outputArray(arena);
    try items.array.append(.{ .integer = 1 });
    const bytes = try python.bytearray(arena, "ab");
    try shared.object.put(arena, "items", items);
    try shared.object.put(arena, "raw", .{ .custom = bytes.custom });
    try original.outputs.root.array.append(shared);
    try original.outputs.root.array.append(shared);
    original.usage.free();
    original.usage = .{ .root = shared, .storage = original.outputs.retain().storage };
    original.cost = .{ .root = .{ .custom = bytes.custom }, .storage = original.outputs.retain().storage };
    const record = try original.dump(allocator);
    defer record.free();
    const loaded = try CallResult.load(allocator, record);
    defer loaded.free(allocator);
    try std.testing.expect(loaded.cost.root.custom.ptr == loaded.cost_details.root.object.get("raw").?.custom.ptr);
    try std.testing.expect(loaded.cost.root.custom.ptr == loaded.outputs.root.array.items[0].object.get("raw").?.custom.ptr);
    try cloneSharedMetadata(allocator, &original);
    var backing = std.testing.FailingAllocator.init(allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(backing.allocator(), cloneSharedMetadata, .{&original});
}

test "canonical loading retains genuine container aliases across metadata fields" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            var source_memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer source_memory.deinit();
            const source = source_memory.allocator();
            const python = @import("python_value.zig");
            const tuple = try python.tuple(source, &.{try python.bytearray(source, "ab")});
            var outputs: runtime.Array = .init(source);
            try outputs.appendSlice(&.{ tuple, tuple });
            var record: runtime.ObjectMap = .empty;
            try record.put(source, "responses", .{ .array = .init(source) });
            try record.put(source, "outputs", .{ .array = outputs });
            try record.put(source, "cost", tuple);
            try record.put(source, "cost_details", try python.dictionary(source, &.{.{
                .key = .{ .integer = 7 },
                .value = tuple,
            }}));
            try record.put(source, "response_model", .null);
            const owned_record = try sdk.PayloadValue.fromPythonValue(allocator, .{ .object = record });
            defer owned_record.free();
            const loaded = try CallResult.load(allocator, owned_record);
            defer loaded.free(allocator);
            source_memory.deinit();
            source_memory = .init(std.testing.allocator);
            const shared = loaded.cost.root.sequence;
            try std.testing.expect(shared == loaded.outputs.root.array.items[0].sequence);
            try std.testing.expect(shared == loaded.outputs.root.array.items[1].sequence);
            try std.testing.expect(shared == loaded.cost_details.root.dictionary.entries.items[0].value.sequence);
            const raw: Value = .{ .custom = shared.items.items[0].custom };
            try python.bytearrayExtend(loaded.cost.storage.?.allocator(), raw, "c");
            const detail_raw: Value = .{ .custom = loaded.cost_details.root.dictionary.entries.items[0].value.sequence.items.items[0].custom };
            try std.testing.expectEqualStrings("abc", python.view(detail_raw).?.bytearray);
        }
    };
    var backing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(backing.allocator(), Check.run, .{});
}

test "canonical load copies output dictionaries per occurrence but retains original tuple and metadata roots" {
    const allocator = std.testing.allocator;
    const storage = try Payload.create(allocator);
    defer storage.release();
    const arena = storage.allocator();
    storage.root = try outputObject(arena);
    const shared = try outputObject(arena);
    const items = try outputArray(arena);
    try items.array.append(.{ .integer = 1 });
    try shared.object.put(arena, "items", items);
    const tuple = try arena.create(graph.Sequence);
    tuple.* = .{ .kind = .tuple, .items = .init(arena) };
    try tuple.items.append(shared);
    const outputs = try outputArray(arena);
    try outputs.array.appendSlice(&.{ shared, shared, .{ .sequence = tuple } });
    try storage.root.object.put(arena, "responses", try outputArray(arena));
    try storage.root.object.put(arena, "outputs", outputs);
    try storage.root.object.put(arena, "cost", .{ .sequence = tuple });
    try storage.root.object.put(arena, "cost_details", shared);
    try storage.root.object.put(arena, "response_model", .null);
    const loaded = try CallResult.load(allocator, .{ .root = storage.root, .storage = storage });
    defer loaded.free(allocator);
    try items.array.append(.{ .integer = 9 });
    const rows = loaded.outputs.root.array.items;
    try std.testing.expect(rows[0].object != rows[1].object);
    try std.testing.expect(rows[0].object != shared.object);
    try std.testing.expect(rows[0].object.get("items").?.array != rows[1].object.get("items").?.array);
    try std.testing.expectEqual(@as(usize, 1), rows[0].object.get("items").?.array.items.len);
    try std.testing.expect(rows[2].sequence == tuple);
    try std.testing.expect(loaded.cost.root.sequence == tuple);
    try std.testing.expect(loaded.cost_details.root.object == shared.object);
    const dumped = try loaded.dump(allocator);
    defer dumped.free();
    const dump_rows = dumped.root.object.get("outputs").?.array.items;
    try std.testing.expect(dump_rows[2] == .array);
    try std.testing.expectEqual(@as(i64, 9), dump_rows[2].array.items[0].object.get("items").?.array.items[1].integer);
    try std.testing.expect(dumped.root.object.get("cost").?.sequence == tuple);
}

test "owning canonical records retain SDK opaque aliases after every source owner is released" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            const record = block: {
                const owner = try Payload.create(allocator);
                defer owner.release();
                const arena = owner.allocator();
                owner.root = try outputObject(arena);
                try owner.root.object.put(arena, "items", try outputArray(arena));
                const input: sdk.PayloadValue = .{ .root = owner.root, .storage = owner };
                const raw = try sdk.PayloadValue.fromPythonValue(allocator, try @import("python_value.zig").bytearray(arena, "ab"));
                defer raw.free();
                const response = try sdk.Response.fromFields(allocator, &.{
                    .model = "escaped-sdk-model",
                    .message = .{ .role = .assistant, .parts = &.{.{ .tool_call = .{ .id = "call", .name = "tool", .input = input } }} },
                    .finish_reason = .tool_call,
                    .adaptations = &.{.{ .field = "config.extensions", .action = .substituted, .reason = "opaque", .asked = raw, .applied = raw }},
                });
                defer response.release();
                var original = try CallResult.native(allocator, response, .{});
                defer original.free(allocator);
                original.cost = input.retain();
                try original.cost_details.root.object.put(original.cost_details.storage.?.allocator(), "raw", raw.root);
                // The details' leaf belongs to the SDK response, not to the
                // four public metadata owners. The dump must retain it too.
                break :block try original.dump(allocator);
            };
            defer record.free();
            const loaded = try CallResult.load(allocator, record);
            defer loaded.free(allocator);
            const retained = loaded.responses[0];
            const input = retained.fields.message.parts[0].tool_call.input;
            try std.testing.expect(loaded.cost.root.object == input.root.object);
            try loaded.cost.root.object.get("items").?.array.append(.{ .integer = 9 });
            try std.testing.expectEqual(@as(i64, 9), input.root.object.get("items").?.array.items[0].integer);
            try std.testing.expectEqualStrings("escaped-sdk-model", retained.fields.model);
            const asked = retained.fields.adaptations[0].asked;
            const detail = loaded.cost_details.root.object.get("raw").?.custom;
            try std.testing.expect(asked.root.custom.ptr == detail.ptr);
            try std.testing.expect(detail.ptr == retained.fields.adaptations[0].applied.root.custom.ptr);
            try @import("python_value.zig").bytearrayExtend(asked.storage.?.allocator(), .{ .custom = detail }, "c");
            try std.testing.expectEqualStrings("abc", @import("python_value.zig").view(.{ .custom = detail }).?.bytearray);
        }
    };
    var backing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(backing.allocator(), Check.run, .{});
}

test "plain result projection uses model iteration rather than model serializers" {
    const allocator = std.testing.allocator;
    const result = try CallResult.init(allocator);
    defer result.free(allocator);
    const arena = result.outputs.storage.?.allocator();
    const empty = try (@import("reasoning.zig").Reasoning{ .content = "" }).toRuntime(arena);
    const code = try (@import("code.zig").Code{ .code = "x = 1" }).toRuntime(arena);
    try result.outputs.root.array.appendSlice(&.{ .{ .custom = empty.custom }, .{ .custom = code.custom } });
    const dumped = try result.dump(allocator);
    defer dumped.free();
    const rows = dumped.root.object.get("outputs").?.array.items;
    try std.testing.expectEqual(@as(usize, 0), rows[0].object.count());
    try std.testing.expectEqualStrings("x = 1", rows[1].object.get("code").?.string);
    const nonempty = try (@import("reasoning.zig").Reasoning{ .content = "plan" }).toRuntime(arena);
    result.outputs.root.array.items[0] = .{ .custom = nonempty.custom };
    try std.testing.expectError(error.InvalidCallResult, result.dump(allocator));
    result.outputs.root.array.items[0] = .null;
    try result.cost_details.root.object.put(arena, "self", result.cost_details.root);
    try std.testing.expectError(error.InvalidCallResult, result.dump(allocator));
}

fn cloneSharedMetadata(allocator: Allocator, original: *const CallResult) !void {
    const python = @import("python_value.zig");
    const copied = try original.clone(allocator);
    defer copied.free(allocator);
    const root = copied.usage.root.object;
    try std.testing.expect(root == copied.cost_details.root.object);
    try std.testing.expect(root == copied.outputs.root.array.items[0].object);
    try std.testing.expect(root == copied.outputs.root.array.items[1].object);
    try std.testing.expect(root != original.usage.root.object);
    try std.testing.expect(root.get("raw").?.custom.ptr == copied.cost.root.custom.ptr);
    try std.testing.expect(copied.cost.root.custom.ptr != original.cost.root.custom.ptr);
    const raw: Value = .{ .custom = copied.cost.root.custom };
    try python.bytearrayExtend(copied.cost.storage.?.allocator(), raw, "d");
    try std.testing.expectEqualStrings("abd", python.view(raw).?.bytearray);
    try std.testing.expectEqualStrings("ab", python.view(.{ .custom = original.cost.root.custom }).?.bytearray);
}

test "combined output owner acquisition unwinds every allocation failure" {
    const Check = struct {
        fn run(allocator: Allocator) !void {
            const response = try sdk.Response.fromFields(allocator, &.{
                .model = "output-fault",
                .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "answer" } }} },
                .finish_reason = .stop,
            });
            defer response.release();
            const candidate = try CallResult.native(allocator, response, .{ .model_type = .responses });
            defer candidate.free(allocator);
            const arena = candidate.cost_details.storage.?.allocator();
            try candidate.cost_details.root.object.put(arena, "marker", .{ .integer = 1 });
            const combined = try combine(allocator, &.{ candidate, candidate }, .responses);
            defer combined.free(allocator);
            const independent = try combined.clone(allocator);
            defer independent.free(allocator);
            const rows = independent.outputs.root.array.items;
            try std.testing.expect(rows[0].object == rows[1].object);
            try std.testing.expect(rows[0].object != candidate.outputs.root.array.items[0].object);
            try candidate.cost_details.root.object.put(arena, "marker", .{ .integer = 7 });
            const candidates = combined.cost_details.root.object.get("candidates").?.array.items;
            try std.testing.expect(candidates[0].object == candidate.cost_details.root.object);
            try std.testing.expect(candidates[0].object == candidates[1].object);
            try std.testing.expectEqual(@as(i64, 7), candidates[1].object.get("marker").?.integer);
            const copies = independent.cost_details.root.object.get("candidates").?.array.items;
            try std.testing.expect(copies[0].object == copies[1].object);
            try std.testing.expect(copies[0].object != candidates[0].object);
            try std.testing.expectEqual(@as(i64, 1), copies[0].object.get("marker").?.integer);
        }
    };
    var backing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(backing.allocator(), Check.run, .{});
}

test "SDK response clone unwinds shared objects at every allocation failure" {
    const allocator = std.testing.allocator;
    const payload = try sdk.JsonValue.fromValue(allocator, .{ .object = .empty });
    defer payload.free();
    const fields: sdk.ResponseFields = .{
        .model = "allocation-failure",
        .message = .{ .role = .assistant, .parts = &.{.{ .text = .{ .text = "answer" } }} },
        .finish_reason = .stop,
        .provider_data = payload,
    };
    const first = try sdk.Response.fromFields(allocator, &fields);
    defer first.release();
    const second = try sdk.Response.fromFields(allocator, &fields);
    defer second.release();
    const one = try CallResult.native(allocator, first, .{});
    defer one.free(allocator);
    try one.outputs.root.array.append(first.fields.provider_data.?.root);
    const two = try CallResult.native(allocator, second, .{});
    defer two.free(allocator);
    const combined = try combine(allocator, &.{ one, two, one }, .chat);
    defer combined.free(allocator);
    try std.testing.checkAllAllocationFailures(allocator, cloneSharedResponses, .{&combined});
}

fn cloneSharedResponses(allocator: Allocator, original: *const CallResult) !void {
    const result = try original.clone(allocator);
    defer result.free(allocator);
    try std.testing.expect(result.responses[0] == result.responses[2]);
    try std.testing.expect(result.responses[0] != result.responses[1]);
    try std.testing.expect(result.responses[0] != original.responses[0]);
    const root = result.responses[0].fields.provider_data.?.root.object;
    try std.testing.expect(root == result.responses[1].fields.provider_data.?.root.object);
    try std.testing.expect(root != original.responses[0].fields.provider_data.?.root.object);
    try std.testing.expect(root == result.outputs.root.array.items[1].object);
}

fn dependentStorage(allocator: Allocator, parents: *std.ArrayList(*Payload)) Allocator.Error!*Payload {
    const owned = try parents.toOwnedSlice(allocator);
    return Payload.createDependent(allocator, owned) catch |err| {
        allocator.free(owned);
        return err;
    };
}

fn required(record: graph.Value, name: []const u8) Error!graph.Value {
    return record.object.get(name) orelse error.InvalidCallResult;
}
fn metadataError(err: anyerror) Error {
    return if (err == error.OutOfMemory or err == error.WriteFailed) error.OutOfMemory else error.InvalidCallResult;
}
fn sdkError(err: sdk.Error) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidCallResult;
}
