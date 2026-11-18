//! MAP-14 candidate scoring: tokenize complete chat templates, then score
//! each distinct prefix once. Token IDs are exact decimal integers, not f64.
const std = @import("std");
const sdk = @import("sdk_response.zig");
const runtime = @import("runtime_value.zig");
const typesafe = @import("typesafe.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const Array = runtime.Array;
const Object = runtime.ObjectMap;
const Judgment = typesafe.Judgment;
pub const Error = Allocator.Error || error{LMFailure};
const method = "candidate_sequence_likelihood";

pub const Format = struct {
    schema: Value,
    measured: []const Judgment,

    pub fn fromRequest(arena: Allocator, request: *const sdk.Request) Error!?Format {
        const configured = request.fields.config.response_format orelse return null;
        const format = configured.toValue(arena) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.LMFailure,
        };
        const kind = format.object.get("type") orelse return null;
        if (kind != .string or !std.mem.eql(u8, kind.string, "json_schema")) return null;
        const schema = format.object.get("schema") orelse return null;
        const found = try typesafe.responseJudgments(arena, schema);
        var count: usize = 0;
        for (found) |judgment| {
            if (judgment.keys.len == 0) continue;
            found[count] = judgment;
            count += 1;
        }
        if (count == 0) return null;
        return .{ .schema = schema, .measured = found[0..count] };
    }

    pub fn mixed(self: Format) bool {
        return self.schema.object.get("properties").?.object.count() != self.measured.len;
    }

    fn measures(self: Format, name: []const u8) bool {
        for (self.measured) |j| if (std.mem.eql(u8, name, j.name)) return true;
        return false;
    }
};

pub fn validate(request: *const sdk.Request) Error!void {
    const cfg = request.fields.config;
    if (request.fields.tools.len != 0 or cfg.tool_choice != null or cfg.store != null or
        cfg.user_id != null or cfg.service_tier != null) return error.LMFailure;
    if (cfg.cache) |cache| {
        if (cache.resource != null or cache.mode == .off or cache.retention != null) return error.LMFailure;
    }
    if (cfg.extensions) |extensions| {
        var iterator = extensions.root.object.iterator();
        while (iterator.next()) |entry| {
            const key = entry.key_ptr.*;
            const value = entry.value_ptr.*;
            switch (value) {
                .integer, .number_string => {},
                .float => |number| if (!std.math.isFinite(number)) return error.LMFailure,
                else => return error.LMFailure,
            }
            if (std.mem.eql(u8, key, "n")) {
                const number: Value = switch (value) {
                    inline .integer, .float, .number_string => |scalar, tag| @unionInit(Value, @tagName(tag), scalar),
                    else => unreachable,
                };
                const order = @import("value.zig").numericOrder(number, .{ .integer = 1 }) catch return error.LMFailure;
                if (order != .eq) return error.LMFailure;
            } else {
                const Harmless = enum { temperature, top_p, top_k, seed, frequency_penalty, presence_penalty };
                if (std.meta.stringToEnum(Harmless, key) == null) return error.LMFailure;
            }
        }
    }
}

fn text(value: Value, key: []const u8) []const u8 {
    const field = value.object.get(key) orelse return "";
    return if (field == .string) field.string else "";
}

pub fn adapt(arena: Allocator, policy: @import("openai_request.zig").Adaptations, records: *std.ArrayList(sdk.Adaptation), field: []const u8, action: sdk.AdaptationAction, reason: []const u8, asked: Value) Error!void {
    if (policy == .refuse) return error.LMFailure;
    try records.append(arena, .{
        .field = field,
        .action = action,
        .reason = reason,
        .asked = try json(arena, asked),
    });
}

pub fn adaptations(arena: Allocator, policy: @import("openai_request.zig").Adaptations, records: *std.ArrayList(sdk.Adaptation), format: Format, cfg: sdk.Config) Error!void {
    if (format.mixed()) try adapt(arena, policy, records, "config.response_format", .client_side, "an additional structured-output call answers ordinary properties, which are never scored", .null);
    var settings = cfg;
    settings.response_format = null;
    const actual = settings.toValue(arena) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    };
    inline for (.{ "max_tokens", "temperature", "top_p", "top_k", "stop", "seed", "frequency_penalty", "presence_penalty", "reasoning", "logprobs", "cache", "extensions" }) |name| {
        if (actual.object.get(name)) |asked| try adapt(arena, policy, records, "config." ++ name, .dropped, "candidate likelihood measures unmodified next-token probabilities with max_tokens=1 per trie node; " ++
            "this generation setting has no measurement slot (any generated JSON call still uses its usual mapping)", asked);
    }
}

pub fn appendUnique(arena: Allocator, records: *std.ArrayList(sdk.Adaptation), additional: []const sdk.Adaptation) Error!void {
    const initial = records.items.len;
    for (additional) |record| {
        for (records.items[0..initial]) |prior| {
            if (prior.action != record.action or !std.mem.eql(u8, prior.field, record.field) or !std.mem.eql(u8, prior.reason, record.reason)) continue;
            const a = prior.asked.toValue(arena) catch return error.LMFailure;
            const b = record.asked.toValue(arena) catch return error.LMFailure;
            const x = prior.applied.toValue(arena) catch return error.LMFailure;
            const y = record.applied.toValue(arena) catch return error.LMFailure;
            const equal = @import("value.zig").pyEqual;
            if ((equal(a, b) catch return error.LMFailure) and (equal(x, y) catch return error.LMFailure)) break;
        } else try records.append(arena, record);
    }
}

pub fn question(arena: Allocator, schema: Value, judgment: Judgment) Error![]const u8 {
    const prop = schema.object.get("properties").?.object.get(judgment.name).?;
    const description = text(prop, "description");
    const branches = prop.object.get("anyOf") orelse .null;
    const enum_value = prop.object.get("enum") orelse .null;
    var out: std.Io.Writer.Allocating = .init(arena);
    out.writer.print("{s}\nOptions:\n", .{if (description.len == 0) judgment.name else description}) catch return error.OutOfMemory;
    for (judgment.keys, 0..) |key, i| {
        out.writer.print("- {s}", .{key}) catch return error.OutOfMemory;
        if (judgment.kind != .boolean and enum_value == .null and branches == .array) {
            const branch = branches.array.items[i];
            const title = text(branch, "title");
            const desc = text(branch, "description");
            if (title.len != 0 or desc.len != 0) out.writer.writeAll(": ") catch return error.OutOfMemory;
            if (title.len != 0) out.writer.writeAll(title) catch return error.OutOfMemory;
            if (title.len != 0 and desc.len != 0) out.writer.writeAll(" - ") catch return error.OutOfMemory;
            if (desc.len != 0) out.writer.writeAll(desc) catch return error.OutOfMemory;
        }
        out.writer.writeByte('\n') catch return error.OutOfMemory;
    }
    out.writer.writeAll("Answer with the option only, spelled exactly as listed.") catch return error.OutOfMemory;
    return out.written();
}

pub fn object(arena: Allocator, fields: []const struct { []const u8, Value }) Error!Value {
    var out: Object = .empty;
    try out.ensureTotalCapacity(arena, fields.len);
    for (fields) |field| try out.put(arena, field[0], field[1]);
    return .{ .object = out };
}

pub fn tokenize(arena: Allocator, model: []const u8, conversation: Array, ask: []const u8, answer: []const u8, open: bool) Error!Value {
    var rows: Array = .init(arena);
    try rows.ensureTotalCapacity(conversation.items.len + 2);
    try rows.appendSlice(conversation.items);
    for ([_][]const u8{ "user", "assistant" }, [_][]const u8{ ask, answer }) |role, content| {
        try rows.append(try object(arena, &.{ .{ "role", .{ .string = role } }, .{ "content", .{ .string = content } } }));
    }
    return object(arena, &.{
        .{ "model", .{ .string = model } },               .{ "messages", .{ .array = rows } },
        .{ "add_generation_prompt", .{ .bool = false } }, .{ "continue_final_message", .{ .bool = open } },
    });
}

fn digits(bytes: []const u8) bool {
    if (bytes.len == 0) return false;
    for (bytes) |byte| if (!std.ascii.isDigit(byte)) return false;
    return true;
}

const Node = struct {
    parent: ?usize,
    /// Root nodes share a tokenization prefill. Other nodes store one edge.
    prefix: []const usize,
    token: usize = undefined,
    depth: usize = 0,
    cost: usize,
    prompt: ?usize = null,
    score: f64 = 0,
};
const Edge = struct { parent: usize, token: usize };
pub const Replies = struct { open: []const usize, closed: []const usize };
pub const Batch = struct {
    arena: Allocator,
    tokens: std.StringArrayHashMapUnmanaged(void) = .empty,
    edges: std.AutoHashMapUnmanaged(Edge, usize) = .empty,
    nodes: std.ArrayList(Node) = .empty,
    prompts: std.ArrayList(usize) = .empty,
    prompt_cost: usize = 0,

    pub fn readTokens(self: *Batch, data: Value) Error![]const usize {
        if (data != .object) return error.LMFailure;
        const tokens = data.object.get("tokens") orelse return error.LMFailure;
        if (tokens != .array or tokens.array.items.len == 0) return error.LMFailure;
        const ids = try self.arena.alloc(usize, tokens.array.items.len);
        for (tokens.array.items, ids) |token, *id| {
            const decimal = switch (token) {
                .integer => |n| if (n >= 0) try std.fmt.allocPrint(self.arena, "{d}", .{n}) else return error.LMFailure,
                .number_string => |s| if (digits(s)) s else return error.LMFailure,
                else => return error.LMFailure,
            };
            const entry = try self.tokens.getOrPut(self.arena, decimal);
            id.* = entry.index;
        }
        return ids;
    }

    pub fn add(self: *Batch, prefix: []const usize, replies: []const Replies) Error![]const usize {
        const terminal = try self.arena.alloc(usize, replies.len);
        var cost: usize = 0;
        for (prefix) |token| cost += self.tokens.keys()[token].len + 1;
        const root = self.nodes.items.len;
        try self.nodes.append(self.arena, .{ .parent = null, .prefix = prefix, .cost = cost });
        for (replies, terminal) |reply, *end| {
            if (reply.open.len <= prefix.len or reply.closed.len <= reply.open.len or
                !std.mem.startsWith(usize, reply.open, prefix) or
                !std.mem.startsWith(usize, reply.closed, reply.open)) return error.LMFailure;
            var parent = root;
            for (reply.closed[prefix.len .. reply.open.len + 1]) |token| {
                const node = &self.nodes.items[parent];
                if (node.prompt == null) {
                    // The trie itself is linear in tokenized paths. Bound the
                    // potentially quadratic projection into explicit prompts.
                    const max_prompt_bytes = 16 * 1024 * 1024;
                    const added = node.cost + 2;
                    if (added > max_prompt_bytes - self.prompt_cost) return error.LMFailure;
                    self.prompt_cost += added;
                    node.prompt = self.prompts.items.len;
                    try self.prompts.append(self.arena, parent);
                }
                const link = try self.edges.getOrPut(self.arena, .{ .parent = parent, .token = token });
                if (!link.found_existing) {
                    link.value_ptr.* = self.nodes.items.len;
                    const child: Node = .{
                        .parent = parent,
                        .prefix = prefix,
                        .token = token,
                        .depth = node.depth + 1,
                        .cost = node.cost + self.tokens.keys()[token].len + 1,
                    };
                    try self.nodes.append(self.arena, child);
                }
                parent = link.value_ptr.*;
            }
            end.* = parent;
        }
        return terminal;
    }

    fn tokenValue(self: Batch, id: usize) Value {
        return .{ .number_string = self.tokens.keys()[id] };
    }

    fn decimalLess(_: void, a: []const u8, b: []const u8) bool {
        if (a.len != b.len) return a.len < b.len;
        return std.mem.order(u8, a, b) == .lt;
    }

    pub fn payload(self: Batch, model: []const u8) Error!Value {
        var prompts: Array = .init(self.arena);
        try prompts.ensureTotalCapacity(self.prompts.items.len);
        for (self.prompts.items) |index| {
            const node = self.nodes.items[index];
            var row: Array = .init(self.arena);
            try row.resize(node.prefix.len + node.depth);
            for (node.prefix, row.items[0..node.prefix.len]) |token, *slot| slot.* = self.tokenValue(token);
            var cursor = node;
            var end = row.items.len;
            while (cursor.parent) |parent| {
                end -= 1;
                row.items[end] = self.tokenValue(cursor.token);
                cursor = self.nodes.items[parent];
            }
            std.debug.assert(end == node.prefix.len);
            try prompts.append(.{ .array = row });
        }
        var union_ids: std.StringArrayHashMapUnmanaged(void) = .empty;
        var edges = self.edges.keyIterator();
        while (edges.next()) |edge| try union_ids.put(self.arena, self.tokens.keys()[edge.token], {});
        const sorted = try self.arena.dupe([]const u8, union_ids.keys());
        std.mem.sortUnstable([]const u8, sorted, {}, decimalLess);
        var ids: Array = .init(self.arena);
        try ids.ensureTotalCapacity(sorted.len);
        for (sorted) |id| try ids.append(.{ .number_string = id });
        return object(self.arena, &.{
            .{ "model", .{ .string = model } },          .{ "prompt", .{ .array = prompts } },
            .{ "max_tokens", .{ .integer = 1 } },        .{ "temperature", .{ .float = 1 } },
            .{ "logprobs", .{ .integer = 0 } },          .{ "return_tokens_as_token_ids", .{ .bool = true } },
            .{ "logprob_token_ids", .{ .array = ids } },
        });
    }

    /// Parse all rows first: a malformed later row is not a missing-score fallback.
    pub fn readScores(self: *Batch, reply: Value) Error!bool {
        if (reply != .object) return error.LMFailure;
        const choices = reply.object.get("choices") orelse return error.LMFailure;
        const count = self.prompts.items.len;
        if (choices != .array or choices.array.items.len != count) return error.LMFailure;
        const Table = std.AutoHashMapUnmanaged(usize, f64);
        const tables = try self.arena.alloc(?Table, count);
        @memset(tables, null);
        for (choices.array.items) |row| {
            if (row != .object) return error.LMFailure;
            const n = row.object.get("index") orelse return error.LMFailure;
            if (n != .integer or n.integer < 0 or n.integer >= count) return error.LMFailure;
            const index: usize = @intCast(n.integer);
            if (tables[index] != null) return error.LMFailure;
            tables[index] = .empty;
            const logs = row.object.get("logprobs") orelse .null;
            if (logs == .null) continue;
            if (logs != .object) return error.LMFailure;
            const list = logs.object.get("top_logprobs") orelse .null;
            if (list == .null) continue;
            if (list != .array or list.array.items.len > 1) return error.LMFailure;
            if (list.array.items.len == 0 or list.array.items[0] == .null) continue;
            const top = list.array.items[0];
            if (top != .object) return error.LMFailure;
            var pairs = top.object.iterator();
            while (pairs.next()) |pair| {
                const key = pair.key_ptr.*;
                const marker = "token_id:";
                if (!std.mem.startsWith(u8, key, marker) or !digits(key[marker.len..])) continue;
                const suffix = key[marker.len..];
                if (suffix.len > 4300) return error.LMFailure;
                const trimmed = std.mem.trimStart(u8, suffix, "0");
                const score = pair.value_ptr.*;
                switch (score) {
                    .float, .integer, .number_string => {},
                    else => return error.LMFailure,
                }
                const logprob = @import("value.zig").numericFloat(score) catch return error.LMFailure;
                if (!(logprob <= 0)) return error.LMFailure;
                const id = self.tokens.getIndex(if (trimmed.len == 0) "0" else trimmed) orelse continue;
                try tables[index].?.put(self.arena, id, logprob);
            }
        }
        // A parent is always inserted before its children. Summing in that
        // order preserves the wire's left-to-right floating point arithmetic.
        for (self.nodes.items) |*node| if (node.parent) |p| {
            const parent = self.nodes.items[p];
            const score = tables[parent.prompt.?].?.get(node.token) orelse return false;
            node.score = parent.score + score;
        };
        return true;
    }

    pub const Fold = struct { data: sdk.DataPart, provider: Value };

    pub fn fold(self: Batch, found: []const Judgment, terminals: []const []const usize, calls: usize) Error!Fold {
        std.debug.assert(found.len == terminals.len);
        const a = self.arena;
        var answers: Object = .empty;
        var probabilities: Object = .empty;
        var coverage: Object = .empty;
        for (found, terminals) |j, ends| {
            std.debug.assert(j.keys.len == ends.len and ends.len != 0);
            var best: usize = 0;
            var mass: f64 = 0;
            for (ends, 0..) |node, i| {
                const score = self.nodes.items[node].score;
                if (score > self.nodes.items[ends[best]].score) best = i;
                mass += @exp(score);
            }
            const max = self.nodes.items[ends[best]].score;
            if (!std.math.isFinite(max)) return error.LMFailure;
            var dist: Object = .empty;
            try dist.ensureTotalCapacity(a, ends.len);
            var sum: f64 = 0;
            for (j.keys, ends) |key, node| {
                const weight = @exp(self.nodes.items[node].score - max);
                sum += weight;
                try dist.put(a, key, .{ .float = weight });
            }
            // Python picks the first *normalized* tie, including rounding ties.
            best = 0;
            for (dist.values(), 0..) |*weight, i| {
                weight.float /= sum;
                if (weight.float > dist.values()[best].float) best = i;
            }
            const answer: Value = switch (j.kind) {
                .boolean => .{ .bool = best == 0 },
                .ordered => .{ .integer = @intCast(best) },
                .choice => .{ .string = j.keys[best] },
                .box => unreachable,
            };
            try answers.put(a, j.name, answer);
            try probabilities.put(a, j.name, .{ .object = dist });
            try coverage.put(a, j.name, .{ .float = mass });
        }
        return .{
            .data = .{ .value = try json(a, .{ .object = answers }), .probabilities = try json(a, .{ .object = probabilities }), .method = .candidate_sequence_likelihood },
            .provider = try object(a, &.{
                .{ "coverage", .{ .object = coverage } },
                .{ "judgments", try object(a, &.{
                    .{ "nodes", .{ .integer = @intCast(self.prompts.items.len) } },
                    .{ "tokenize_calls", .{ .integer = @intCast(calls) } },
                    .{ "method", .{ .string = method } },
                }) },
            }),
        };
    }
};

pub fn json(arena: Allocator, value: Value) Error!sdk.JsonValue {
    const graph = @import("json_graph.zig");
    return .borrowed(graph.fromValue(arena, value) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    });
}

pub fn generatedParts(arena: Allocator, format: Format, message: sdk.Message, fallback: bool) Error![]sdk.Part {
    const index = for (message.parts, 0..) |part, i| {
        if (part == .text) break i;
    } else return error.LMFailure;
    for (message.parts[index + 1 ..]) |part| if (part == .text) return error.LMFailure;
    const visible = message.parts[index].text;
    const parsed = @import("json_repair.zig").parseJson(arena, visible.text) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.LMFailure,
    };
    if (parsed != .object) return error.LMFailure;
    const required = format.schema.object.get("required") orelse .null;
    if (required != .null) {
        if (required != .array) return error.LMFailure;
        for (required.array.items) |key| {
            if (key != .string) return error.LMFailure;
            if ((fallback or !format.measures(key.string)) and !parsed.object.contains(key.string)) return error.LMFailure;
        }
    }
    const value = try json(arena, parsed);
    @import("sdk_json.zig").validate(value.root, 0) catch return error.LMFailure;
    const parts = try arena.dupe(sdk.Part, message.parts);
    parts[index] = .{ .data = .{ .value = value, .continuation = visible.continuation } };
    return parts;
}

test "candidate scoring shares prefixes and reads exact wide ids and zero-likelihood edges" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const parse = @import("json_repair.zig").parseJson;
    var batch: Batch = .{ .arena = a };
    const prefix = try batch.readTokens(try parse(a, "{\"tokens\":[5]}"));
    const open = try batch.readTokens(try parse(a, "{\"tokens\":[5,18446744073709551617]}"));
    const terminals = try batch.add(prefix, &.{
        .{ .open = open, .closed = try batch.readTokens(try parse(a, "{\"tokens\":[5,18446744073709551617,2,42]}")) },
        .{ .open = open, .closed = try batch.readTokens(try parse(a, "{\"tokens\":[5,18446744073709551617,3]}")) },
    });
    const body = try batch.payload("fixture");
    const prompts = body.object.get("prompt").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), prompts.len);
    try std.testing.expectEqualStrings("18446744073709551617", prompts[1].array.items[1].number_string);
    const union_ids = body.object.get("logprob_token_ids").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), union_ids.len);
    try std.testing.expectEqualStrings("2", union_ids[0].number_string);
    try std.testing.expectEqualStrings("3", union_ids[1].number_string);
    try std.testing.expectEqualStrings("18446744073709551617", union_ids[2].number_string);

    const reply = try parse(a,
        \\{"choices":[
        \\  {"index":1,"logprobs":{"top_logprobs":[{"token_id:2":-Infinity,"token_id:03":-0.5}]}},
        \\  {"index":0,"logprobs":{"top_logprobs":[{"token_id:18446744073709551617":-0.25}]}}
        \\]}
    );
    try std.testing.expect(try batch.readScores(reply));
    const folded = try batch.fold(&.{.{ .name = "pick", .kind = .boolean, .keys = &.{ "true", "false" } }}, &.{terminals}, 5);
    try std.testing.expectEqual(false, folded.data.value.root.object.get("pick").?.bool);
    const dist = folded.data.probabilities.?.root.object.get("pick").?.object;
    try std.testing.expectEqual(@as(f64, 0), dist.get("true").?.float);
    try std.testing.expectEqual(@as(f64, 1), dist.get("false").?.float);
    try std.testing.expectApproxEqAbs(@exp(@as(f64, -0.75)), folded.provider.object.get("coverage").?.object.get("pick").?.float, 1e-15);
}
