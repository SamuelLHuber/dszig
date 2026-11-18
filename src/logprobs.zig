//! Canonical lm15 token scores, independent of provider wire dialects.
//! Unknown byte sequences and vocabulary IDs are not invented. Byte entries
//! follow the SDK's integer protocol, which deliberately does not cap at 255.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const numbers = @import("value.zig");
const python = @import("python_value.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
pub const Error = Allocator.Error || error{InvalidLogprobs};

pub const TopLogprob = struct {
    token: []const u8,
    logprob: f64,
    bytes: ?[]const Value = null,
    token_id: Value = .null,

    pub fn jsonStringify(top: TopLogprob, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeNativeTop(top, stream);
        try stream.endObject();
    }
};

pub const TokenLogprob = struct {
    token: []const u8,
    logprob: f64,
    bytes: ?[]const Value = null,
    token_id: Value = .null,
    top: []const TopLogprob = &.{},

    pub fn jsonStringify(token: TokenLogprob, stream: *std.json.Stringify) std.Io.Writer.Error!void {
        try stream.beginObject();
        try writeNativeTop(.{ .token = token.token, .logprob = token.logprob, .bytes = token.bytes, .token_id = token.token_id }, stream);
        if (token.top.len != 0) {
            try stream.objectField("top");
            try stream.write(token.top);
        }
        try stream.endObject();
    }
};

/// Validated owning copy. Scores may be nonfinite; the SDK does not clamp them.
/// Caller-provided integer IDs accept integral builtin floats, unlike bytes.
pub fn checked(allocator: Allocator, source: []const TokenLogprob) Error![]const TokenLogprob {
    const out = try clone(allocator, source);
    errdefer free(allocator, out);
    for (out) |*token| {
        const token_id = try normalizeOwnedId(allocator, token.token_id);
        token.token_id = token_id;
        try validateBytes(token.bytes);
        // clone() allocated this top array; no caller-owned const storage is
        // made writable here.
        for (@constCast(token.top)) |*top| {
            const top_id = try normalizeOwnedId(allocator, top.token_id);
            top.token_id = top_id;
            try validateBytes(top.bytes);
        }
    }
    return out;
}

/// Deep copy borrowing nothing from its source, including boxed integers.
pub fn clone(allocator: Allocator, source: []const TokenLogprob) Allocator.Error![]TokenLogprob {
    var context: runtime.CloneContext = .init(allocator);
    defer context.deinit();
    const out = try allocator.alloc(TokenLogprob, source.len);
    var written: usize = 0;
    errdefer {
        for (out[0..written]) |token| freeToken(allocator, token);
        allocator.free(out);
    }
    for (source, out) |token, *target| {
        target.* = try cloneToken(&context, token);
        written += 1;
    }
    return out;
}

fn cloneToken(context: *runtime.CloneContext, source: TokenLogprob) Allocator.Error!TokenLogprob {
    const allocator = context.allocator;
    const token = try allocator.dupe(u8, source.token);
    errdefer allocator.free(token);
    const bytes = try cloneBytes(context, source.bytes);
    errdefer freeBytes(allocator, bytes);
    const token_id = try context.cloneValue(source.token_id);
    errdefer runtime.freeValue(allocator, token_id);
    const top = try allocator.alloc(TopLogprob, source.top.len);
    var written: usize = 0;
    errdefer {
        for (top[0..written]) |item| freeTop(allocator, item);
        allocator.free(top);
    }
    for (source.top, top) |item, *target| {
        target.* = try cloneTop(context, item);
        written += 1;
    }
    return .{ .token = token, .logprob = source.logprob, .bytes = bytes, .token_id = token_id, .top = top };
}

fn cloneTop(context: *runtime.CloneContext, source: TopLogprob) Allocator.Error!TopLogprob {
    const allocator = context.allocator;
    const token = try allocator.dupe(u8, source.token);
    errdefer allocator.free(token);
    const bytes = try cloneBytes(context, source.bytes);
    errdefer freeBytes(allocator, bytes);
    return .{ .token = token, .logprob = source.logprob, .bytes = bytes, .token_id = try context.cloneValue(source.token_id) };
}

fn cloneBytes(context: *runtime.CloneContext, source: ?[]const Value) Allocator.Error!?[]const Value {
    const items = source orelse return null;
    const out = try context.allocator.alloc(Value, items.len);
    var written: usize = 0;
    errdefer {
        for (out[0..written]) |item| runtime.freeValue(context.allocator, item);
        context.allocator.free(out);
    }
    for (items, out) |item, *target| {
        target.* = try context.cloneValue(item);
        written += 1;
    }
    return out;
}

pub fn free(allocator: Allocator, tokens: []const TokenLogprob) void {
    for (tokens) |token| freeToken(allocator, token);
    allocator.free(tokens);
}

fn freeToken(allocator: Allocator, token: TokenLogprob) void {
    for (token.top) |top| freeTop(allocator, top);
    allocator.free(token.top);
    freeTop(allocator, .{ .token = token.token, .logprob = token.logprob, .bytes = token.bytes, .token_id = token.token_id });
}

fn freeTop(allocator: Allocator, top: TopLogprob) void {
    allocator.free(top.token);
    freeBytes(allocator, top.bytes);
    runtime.freeValue(allocator, top.token_id);
}

fn freeBytes(allocator: Allocator, bytes: ?[]const Value) void {
    const items = bytes orelse return;
    for (items) |item| runtime.freeValue(allocator, item);
    allocator.free(items);
}

pub fn validateBytes(bytes: ?[]const Value) Error!void {
    for (bytes orelse return) |item| {
        const integer = try integerValue(item);
        const order = numbers.numericOrder(integer, .{ .integer = 0 }) catch return error.InvalidLogprobs;
        if (order == null or order.? == .lt) return error.InvalidLogprobs;
    }
}

fn integerValue(raw: Value) Error!Value {
    return switch (raw) {
        .integer => raw,
        .number_string => if (std.json.isNumberFormattedLikeAnInteger(raw.number_string))
            numbers.parseInteger(raw) catch error.InvalidLogprobs
        else
            error.InvalidLogprobs,
        .custom => raw.custom.integerValue() orelse error.InvalidLogprobs,
        else => error.InvalidLogprobs,
    };
}

/// Replaces an owned float ID only after its replacement is ready.
fn normalizeOwnedId(allocator: Allocator, raw: Value) Error!Value {
    if (raw == .null) return raw;
    const floating: ?f64 = if (raw == .float) raw.float else if (python.view(raw)) |builtin|
        if (builtin == .float) builtin.float else null
    else
        null;
    if (floating) |number| {
        var buffer: [310]u8 = undefined;
        const text = numbers.integralFloatText(number, &buffer) orelse return error.InvalidLogprobs;
        const parsed = Value.parseFromNumberSlice(text);
        const result = if (parsed == .number_string)
            Value{ .number_string = try allocator.dupe(u8, parsed.number_string) }
        else
            parsed;
        runtime.freeValue(allocator, raw);
        return result;
    }
    _ = try integerValue(raw);
    return raw;
}

fn scoreNumber(raw: Value) Error!f64 {
    switch (raw) {
        .float, .integer, .number_string => return numbers.numericFloat(raw) catch error.InvalidLogprobs,
        .custom => if (python.view(raw)) |builtin| {
            if (builtin == .float) return builtin.float;
        },
        else => {},
    }
    return error.InvalidLogprobs;
}

/// SDK serde accepts sequences, including an empty iterable, not just lists.
fn sequence(input: Value) Error![]const Value {
    switch (input) {
        .array => return input.array.items,
        .string => if (input.string.len == 0) return &.{},
        .object => if (input.object.count() == 0) return &.{},
        .custom => if (python.view(input)) |builtin| switch (builtin) {
            .tuple => |items| return items,
            .set => |items| return items,
            .frozen_set => |items| return items,
            else => {},
        },
        else => {},
    }
    return error.InvalidLogprobs;
}

/// Canonical nonnull SDK sequence. Unknown fields in individual records are
/// ignored, as token_logprob_from_dict does; missing required fields refuse.
pub fn fromValue(allocator: Allocator, input: Value) Error![]const TokenLogprob {
    return readScores(allocator, input, false);
}

fn readScores(allocator: Allocator, input: Value, native: bool) Error![]const TokenLogprob {
    const items = try sequence(input);
    const out = try allocator.alloc(TokenLogprob, items.len);
    var written: usize = 0;
    errdefer {
        for (out[0..written]) |token| {
            for (token.top) |top| freeTop(allocator, top);
            allocator.free(token.top);
            freeTop(allocator, .{ .token = token.token, .logprob = token.logprob, .bytes = token.bytes, .token_id = token.token_id });
        }
        allocator.free(out);
    }
    for (items, out) |item, *target| {
        target.* = try tokenFromValue(allocator, item, native);
        written += 1;
    }
    return out;
}

fn topFromValue(allocator: Allocator, input: Value, native: bool) Error!TopLogprob {
    if (input != .object) return error.InvalidLogprobs;
    const token = input.object.get("token") orelse return error.InvalidLogprobs;
    if (token != .string) return error.InvalidLogprobs;
    const raw_score = input.object.get("logprob") orelse return error.InvalidLogprobs;
    const score = try scoreNumber(if (native) try restoreNativeScore(raw_score) else raw_score);
    const raw_bytes = input.object.get("bytes") orelse .null;
    const bytes = if (raw_bytes == .null) null else try sequence(raw_bytes);
    try validateBytes(bytes);
    var context: runtime.CloneContext = .init(allocator);
    defer context.deinit();
    var out = try cloneTop(&context, .{ .token = token.string, .logprob = score, .bytes = bytes, .token_id = input.object.get("token_id") orelse .null });
    errdefer freeTop(allocator, out);
    const token_id = try normalizeOwnedId(allocator, out.token_id);
    out.token_id = token_id;
    return out;
}

fn tokenFromValue(allocator: Allocator, input: Value, native: bool) Error!TokenLogprob {
    const base = try topFromValue(allocator, input, native);
    errdefer freeTop(allocator, base);
    const raw_top = input.object.get("top");
    const items = if (raw_top) |present| try sequence(present) else &.{};
    const top = try allocator.alloc(TopLogprob, items.len);
    var written: usize = 0;
    errdefer {
        for (top[0..written]) |item| freeTop(allocator, item);
        allocator.free(top);
    }
    for (items, top) |item, *target| {
        target.* = try topFromValue(allocator, item, native);
        written += 1;
    }
    return .{ .token = base.token, .logprob = base.logprob, .bytes = base.bytes, .token_id = base.token_id, .top = top };
}

/// Borrowed leaves and allocated containers, like other native metadata views.
pub fn toValue(arena: Allocator, tokens: []const TokenLogprob) Allocator.Error!Value {
    return scoreValues(arena, tokens, false);
}

/// Native persistence is JSON, not Python's nonstandard NaN/Infinity JSON.
/// Only nonfinite scores use an explicit private IEEE-754 bit tag.
pub fn toNativeValue(arena: Allocator, tokens: []const TokenLogprob) Allocator.Error!Value {
    return scoreValues(arena, tokens, true);
}

fn scoreValues(arena: Allocator, tokens: []const TokenLogprob, native: bool) Allocator.Error!Value {
    var array: runtime.Array = .init(arena);
    try array.ensureTotalCapacity(tokens.len);
    for (tokens) |token| {
        var out = try topValue(arena, .{ .token = token.token, .logprob = token.logprob, .bytes = token.bytes, .token_id = token.token_id }, native);
        if (token.top.len != 0) {
            var top: runtime.Array = .init(arena);
            try top.ensureTotalCapacity(token.top.len);
            for (token.top) |item| top.appendAssumeCapacity(try topValue(arena, item, native));
            try out.object.put(arena, "top", .{ .array = top });
        }
        array.appendAssumeCapacity(out);
    }
    return .{ .array = array };
}

fn nativeScore(arena: Allocator, score: f64) Allocator.Error!Value {
    if (std.math.isFinite(score)) return .{ .float = score };
    var object: runtime.ObjectMap = .empty;
    const bits = try std.fmt.allocPrint(arena, "{x:0>16}", .{@as(u64, @bitCast(score))});
    try object.put(arena, "$dszig_f64", .{ .string = bits });
    return .{ .object = object };
}

fn writeNativeTop(top: TopLogprob, stream: *std.json.Stringify) std.Io.Writer.Error!void {
    try stream.objectField("token");
    try stream.write(top.token);
    try stream.objectField("logprob");
    if (std.math.isFinite(top.logprob)) {
        try stream.write(top.logprob);
    } else {
        var buffer: [16]u8 = undefined;
        const bits = std.fmt.bufPrint(&buffer, "{x:0>16}", .{@as(u64, @bitCast(top.logprob))}) catch unreachable;
        try stream.write(.{ .@"$dszig_f64" = bits });
    }
    if (top.bytes) |bytes| {
        try stream.objectField("bytes");
        try stream.write(bytes);
    }
    if (top.token_id != .null) {
        try stream.objectField("token_id");
        try stream.write(top.token_id);
    }
}

/// Owning native-persistence reader. Decode only scores' reserved bit tags,
/// then reuse the canonical SDK reader rather than relaxing its public rules.
pub fn fromNativeValue(allocator: Allocator, input: Value) Error![]const TokenLogprob {
    return readScores(allocator, input, true);
}

fn restoreNativeScore(input: Value) Error!Value {
    if (input != .object) return input;
    if (input.object.count() != 1) return error.InvalidLogprobs;
    const bits = input.object.get("$dszig_f64") orelse return error.InvalidLogprobs;
    if (bits != .string or bits.string.len != 16) return error.InvalidLogprobs;
    for (bits.string) |byte| if (!std.ascii.isHex(byte)) return error.InvalidLogprobs;
    const number = std.fmt.parseInt(u64, bits.string, 16) catch return error.InvalidLogprobs;
    return .{ .float = @bitCast(number) };
}

fn topValue(arena: Allocator, top: TopLogprob, native: bool) Allocator.Error!Value {
    var out: runtime.ObjectMap = .empty;
    try out.put(arena, "token", .{ .string = top.token });
    try out.put(arena, "logprob", if (native) try nativeScore(arena, top.logprob) else .{ .float = top.logprob });
    if (top.bytes) |items| {
        var bytes: runtime.Array = .init(arena);
        try bytes.appendSlice(items);
        try out.put(arena, "bytes", .{ .array = bytes });
    }
    if (top.token_id != .null) try out.put(arena, "token_id", top.token_id);
    return .{ .object = out };
}

test "owned score records outlive mutated token, byte and alternative storage" {
    const testing = std.testing;
    const kept = blk: {
        var memory: std.heap.ArenaAllocator = .init(testing.allocator);
        defer memory.deinit();
        const arena = memory.allocator();
        const token = try arena.dupe(u8, "chosen");
        const alternative = try arena.dupe(u8, "alternate");
        const wide = try arena.dupe(u8, "18446744073709551617");
        var bytes = [_]Value{ .{ .integer = 256 }, .{ .number_string = wide } };
        var top = [_]TopLogprob{.{ .token = alternative, .logprob = -std.math.inf(f64), .bytes = &bytes }};
        const source = [_]TokenLogprob{.{
            .token = token,
            .logprob = -0.0,
            .bytes = &bytes,
            .token_id = .{ .float = 2.0 },
            .top = &top,
        }};
        const owned = try checked(testing.allocator, &source);
        token[0] = 'X';
        alternative[0] = 'Y';
        wide[0] = '9';
        bytes[0] = .{ .integer = 0 };
        top[0].logprob = 0;
        break :blk owned;
    };
    defer free(testing.allocator, kept);
    try testing.expectEqualStrings("chosen", kept[0].token);
    try testing.expectEqual(@as(i64, 256), kept[0].bytes.?[0].integer);
    try testing.expectEqualStrings("18446744073709551617", kept[0].bytes.?[1].number_string);
    try testing.expectEqual(@as(i64, 2), kept[0].token_id.integer);
    try testing.expectEqualStrings("alternate", kept[0].top[0].token);
    try testing.expectEqual(-std.math.inf(f64), kept[0].top[0].logprob);
    try testing.expectEqual(@as(u64, 0x8000000000000000), @as(u64, @bitCast(kept[0].logprob)));
}

fn checkedThenFree(allocator: Allocator, source: []const TokenLogprob) !void {
    const owned = try checked(allocator, source);
    defer free(allocator, owned);
    try std.testing.expectEqualStrings("chosen", owned[0].token);
    try std.testing.expectEqual(@as(i64, 256), owned[0].top[0].bytes.?[0].integer);
}

test "score clones unwind every allocation including wide float ID normalization" {
    const bytes = [_]Value{ .{ .integer = 256 }, .{ .number_string = "18446744073709551617" } };
    const top = [_]TopLogprob{.{ .token = "alternate", .logprob = -2, .bytes = &bytes, .token_id = .{ .float = 1e100 } }};
    const source = [_]TokenLogprob{.{
        .token = "chosen",
        .logprob = -1,
        .bytes = &bytes,
        .token_id = .{ .float = -1e100 },
        .top = &top,
    }};
    try std.testing.checkAllAllocationFailures(std.testing.allocator, checkedThenFree, .{&source});
}

test "score bytes reject bool and float while signed integral vocabulary IDs remain valid" {
    const testing = std.testing;
    for ([_]Value{ .{ .bool = true }, .{ .float = 1.0 }, .{ .integer = -1 } }) |invalid| {
        const bytes = [_]Value{invalid};
        const source = [_]TokenLogprob{.{ .token = "x", .logprob = 0, .bytes = &bytes }};
        try testing.expectError(error.InvalidLogprobs, checked(testing.allocator, &source));
    }
    const source = [_]TokenLogprob{.{ .token = "x", .logprob = std.math.nan(f64), .token_id = .{ .float = -2.0 } }};
    const owned = try checked(testing.allocator, &source);
    defer free(testing.allocator, owned);
    try testing.expectEqual(@as(i64, -2), owned[0].token_id.integer);
    try testing.expect(std.math.isNan(owned[0].logprob));
}
