//! Advisory DSPy 3.5 billing. Counts stay exact until Source's float-rate multiplication.
//! Catalogue lookup owns routing; this module never inherits another entry's prices.
const std = @import("std");
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");
const value = @import("value.zig");
const unicode = @import("unicode.zig");
const model_prices = @import("model_prices.zig");
const Value = runtime.Value;
const ObjectMap = runtime.ObjectMap;
const Usage = @import("lm_usage.zig").Usage;

pub const Facts = struct {
    provider: ?[]const u8 = null,
    requested_model: []const u8,
    response_model: ?[]const u8 = null,
    usage: Usage = .{},
    provider_data: ?Value = null,
    namespaces: ?[]const []const u8 = null,
    generated_media: bool = false,
    has_hosted_tools: bool = false,
    service_tier: ?[]const u8 = null,
    long_cache_retention: bool = false,
};

pub const Estimate = struct { cost: Value, details: Value };

/// All returned fields own their storage in arena. Pricing cannot fail generation:
/// only allocation failure escapes; metadata/arithmetic failures become unknown.
pub fn estimate(arena: Allocator, catalogue: *const model_prices.Catalogue, facts: *const Facts) Allocator.Error!Estimate {
    return estimateInner(arena, catalogue, facts) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => unknown(arena, "pricing metadata could not be interpreted"),
    };
}

fn estimateInner(arena: Allocator, catalogue: *const model_prices.Catalogue, facts: *const Facts) !Estimate {
    const raw = try dictionaryOrEmpty(facts.provider_data orelse .null);
    // Only an ABSENT key defaults to {}; a present null has no .get and is uninterpretable.
    const hidden = raw.object.get("_hidden_params") orelse Value{ .object = .empty };
    const reported = try get(hidden, "response_cost");
    if (facts.namespaces == null and reported != .null) {
        return .{ .cost = try value.cloneValue(arena, reported), .details = try parse(arena, "{\"kind\":\"reported\",\"currency\":\"USD\",\"source\":\"litellm\",\"field\":\"_hidden_params.response_cost\"}") };
    }
    const provider = facts.provider orelse return unknown(arena, "custom engine has no pricing provider");
    if (facts.generated_media) return unknown(arena, "generated-media billing is not covered by text token rates");
    if (equal(provider, "claude-code") or equal(provider, "openai-codex"))
        return unknown(arena, "subscription usage is not per-token API billing");
    if (facts.has_hosted_tools) return unknown(arena, "hosted-tool fees are not included in token rates");
    if (facts.namespaces) |names| {
        if (names.len == 0) return unknown(arena, "declared provider names no metadata namespace for pricing");
    }
    var priced_model = facts.response_model orelse return error.InvalidMetadata;
    var lookup = try catalogue.lookup(provider, priced_model, facts.namespaces);
    if (!lookup.has_info) {
        priced_model = facts.requested_model;
        lookup = try catalogue.lookup(provider, priced_model, facts.namespaces);
    }
    const empty: ObjectMap = .empty;
    const info = lookup.entry orelse &empty;
    const route = info.get("litellm_provider") orelse .null;
    const unprefixed = if (std.mem.startsWith(u8, priced_model, "azure/")) priced_model[6..] else priced_model;
    if (route == .string and equal(route.string, "azure") and equal(unprefixed, "model-router"))
        return unknown(arena, "Azure model-router fees exclude the selected model's inference cost");
    const usage = &facts.usage;
    if (usage.input_tokens == .null or usage.output_tokens == .null)
        return unknown(arena, "input or output usage was not reported");
    const counts = try separatedCounts(arena, usage);
    if (try negative(counts[0]) or try negative(counts[1]))
        return unknown(arena, "usage dimensions cannot be separated safely");
    const suffix = (try tierSuffix(raw, facts.service_tier)) orelse
        return unknown(arena, "unrecognised service-tier pricing");
    const cost = try dimensionsCost(arena, info, &counts, usage.input_tokens, suffix, facts.long_cache_retention and anthropic(provider));
    if (cost == .null) return unknown(arena, "a billed token dimension has no known rate");
    if (!lookup.has_info) return unknown(arena, "no matching pricing entry");
    return .{ .cost = cost, .details = try estimateDetails(arena, catalogue, facts, provider, priced_model) };
}

fn separatedCounts(arena: Allocator, usage: *const Usage) ![7]Value {
    const cached = zeroIfNull(usage.cache_read_tokens);
    const written = zeroIfNull(usage.cache_write_tokens);
    const reasoning = zeroIfNull(usage.reasoning_tokens);
    const input_audio = zeroIfNull(usage.input_audio_tokens);
    const output_audio = zeroIfNull(usage.output_audio_tokens);
    const uncached = try value.numericDifference(arena, usage.input_tokens, cached);
    const unwritten = try value.numericDifference(arena, uncached, written);
    const nonreasoning = try value.numericDifference(arena, usage.output_tokens, reasoning);
    return .{
        try value.numericDifference(arena, unwritten, input_audio),
        try value.numericDifference(arena, nonreasoning, output_audio),
        reasoning,
        cached,
        written,
        input_audio,
        output_audio,
    };
}

fn dimensionsCost(arena: Allocator, info: *const ObjectMap, counts: *const [7]Value, total_input: Value, suffix: []const u8, long_write: bool) !Value {
    const keys = [_][]const u8{ "input_cost_per_token", "output_cost_per_token", "output_cost_per_reasoning_token", "cache_read_input_token_cost", "cache_creation_input_token_cost", "input_cost_per_audio_token", "output_cost_per_audio_token" };
    const fallbacks = [_]?[]const u8{ null, null, "output_cost_per_token", "input_cost_per_token", "input_cost_per_token", "input_cost_per_token", "output_cost_per_token" };
    var rates: [7]?f64 = undefined;
    // Source resolves EVERY dimension before inspecting zero counts or missing rates.
    for (keys, fallbacks, 0..) |key, fallback, index|
        rates[index] = try price(arena, info, key, fallback, total_input, suffix, index == 4 and long_write);
    for (counts, rates) |count, amount| {
        if (try truthy(count) and amount == null) return .null;
    }
    var sum: Value = .{ .integer = 0 };
    for (counts, rates) |count, amount| {
        if (try truthy(count)) {
            if (amount) |rate| {
                const product = try value.numericProduct(arena, count, .{ .float = rate });
                sum = try value.numericSum(arena, sum, product);
            }
        }
    }
    return sum;
}

fn price(arena: Allocator, info: *const ObjectMap, key: []const u8, fallback: ?[]const u8, total_input: Value, suffix: []const u8, long_write: bool) !?f64 {
    const duration = if (long_write) "_above_1hr" else "";
    if (try dimensionRate(arena, info, key, duration, total_input, suffix)) |rate| return rate;
    if (!long_write) {
        if (fallback) |name| return dimensionRate(arena, info, name, "", total_input, suffix);
    }
    return null;
}

fn dimensionRate(arena: Allocator, info: *const ObjectMap, name: []const u8, duration: []const u8, total_input: Value, suffix: []const u8) !?f64 {
    std.debug.assert(name.len <= 64);
    std.debug.assert(duration.len <= 16);
    std.debug.assert(suffix.len <= 16);
    var prefix_buffer: [96]u8 = undefined;
    var ending_buffer: [32]u8 = undefined;
    const prefix = std.fmt.bufPrint(&prefix_buffer, "{s}{s}_above_", .{ name, duration }) catch unreachable;
    const ending = std.fmt.bufPrint(&ending_buffer, "k_tokens{s}", .{suffix}) catch unreachable;
    var best: ?[]const u8 = null;
    var best_count: Value = .{ .integer = 0 };
    for (info.keys()) |field| {
        if (!std.mem.startsWith(u8, field, prefix) or !std.mem.endsWith(u8, field, ending)) continue;
        if (field.len <= prefix.len + ending.len) continue;
        const digits = (try normalizedDigits(arena, field[prefix.len .. field.len - ending.len])) orelse continue;
        const scaled = try thousands(arena, digits);
        if (!try greater(total_input, scaled)) continue;
        if (best == null or try greater(scaled, best_count)) {
            best = digits;
            best_count = scaled;
        }
    }
    // A zero threshold is falsy in Source and deliberately selects the base key.
    const has_context = best != null and try greater(best_count, .{ .integer = 0 });
    // CPython's default integer-text limit bounds the normalized threshold.
    // Fixed dimension/duration/tier names plus 4300 digits fit this stack key.
    var key_buffer: [4480]u8 = undefined;
    const rate_key = std.fmt.bufPrint(&key_buffer, "{s}{s}{s}{s}{s}{s}", .{
        name,                            duration,                            if (has_context) "_above_" else "",
        if (has_context) best.? else "", if (has_context) "k_tokens" else "", suffix,
    }) catch unreachable;
    return validRate(info.get(rate_key) orelse .null);
}

fn normalizedDigits(arena: Allocator, text: []const u8) !?[]const u8 {
    var offset: usize = 0;
    var length: usize = 0;
    var ascii = true;
    // Validate the entire regex capture before applying int()'s digit limit:
    // a later non-digit means no regex match, not an integer conversion error.
    while (offset < text.len) {
        const width = std.unicode.utf8ByteSequenceLength(text[offset]) catch return null;
        if (width > text.len - offset) return null;
        const cp = std.unicode.utf8Decode(text[offset..][0..width]) catch return null;
        _ = unicode.decimalDigit(cp) orelse return null;
        ascii = ascii and width == 1;
        length += 1;
        offset += width;
    }
    if (length == 0) return null;
    if (length > 4300) return error.InvalidMetadata;
    const digits: []const u8 = if (ascii) text else normalized: {
        const output = try arena.alloc(u8, length);
        offset = 0;
        for (output) |*digit| {
            const width = std.unicode.utf8ByteSequenceLength(text[offset]) catch unreachable;
            const cp = std.unicode.utf8Decode(text[offset..][0..width]) catch unreachable;
            digit.* = '0' + unicode.decimalDigit(cp).?;
            offset += width;
        }
        std.debug.assert(offset == text.len);
        break :normalized output;
    };
    var first: usize = 0;
    while (first + 1 < length and digits[first] == '0') : (first += 1) {}
    return digits[first..length];
}

/// Canonical integer for digits * 1000; digits carry no redundant leading zero.
fn thousands(arena: Allocator, digits: []const u8) !Value {
    if (std.fmt.parseInt(i64, digits, 10)) |number| {
        const scaled = @mulWithOverflow(number, @as(i64, 1000));
        if (scaled[1] == 0) return .{ .integer = scaled[0] };
    } else |_| {}
    const text = try std.fmt.allocPrint(arena, "{s}000", .{digits});
    if (std.fmt.parseInt(i64, text, 10)) |small| return .{ .integer = small } else |_| {}
    return .{ .number_string = text };
}

fn greater(a: Value, b: Value) !bool {
    const order = (try value.numericOrder(a, b)) orelse return false;
    return order == .gt;
}

fn validRate(input: Value) !?f64 {
    switch (input) {
        .integer, .float, .number_string => {},
        else => return null,
    }
    // Source math.isfinite converts integers too, and can raise on enormous ints.
    const rate = try value.numericFloat(input);
    if (!std.math.isFinite(rate) or rate < 0) return null;
    return rate;
}

fn tierSuffix(raw: Value, requested: ?[]const u8) !?[]const u8 {
    var tier = try get(raw, "service_tier");
    if (!try truthy(tier)) {
        const metadata = try dictionaryOrEmpty(try get(raw, "usageMetadata"));
        tier = try get(metadata, "serviceTier");
        if (!try truthy(tier)) tier = if (requested) |text| .{ .string = text } else .null;
    }
    if (tier == .null) return "";
    // Dict lookup on an unhashable tier raises; every other unknown tier is refused.
    if (tier == .array or tier == .object) return error.InvalidMetadata;
    if (tier != .string) return null;
    for ([_][]const u8{ "default", "standard", "auto", "ON_DEMAND" }) |name| {
        if (equal(tier.string, name)) return "";
    }
    if (equal(tier.string, "priority")) return "_priority";
    if (equal(tier.string, "flex")) return "_flex";
    return null;
}

fn dictionaryOrEmpty(input: Value) !Value {
    if (!try truthy(input)) return .{ .object = .empty };
    if (input != .object) return error.InvalidMetadata;
    return input;
}

fn get(input: Value, key: []const u8) !Value {
    if (input != .object) return error.InvalidMetadata;
    return input.object.get(key) orelse .null;
}

fn truthy(input: Value) !bool {
    return switch (input) {
        .null => false,
        .bool => |b| b,
        .integer => |n| n != 0,
        .float => |n| n != 0,
        .number_string => blk: {
            const order = try value.numericOrder(input, .{ .integer = 0 });
            break :blk if (order) |known| known != .eq else true;
        },
        .string => |s| s.len != 0,
        .array => |a| a.items.len != 0,
        .object => |o| o.count() != 0,
        .custom => |c| try c.truthy(),
    };
}

fn zeroIfNull(input: Value) Value {
    return if (input == .null) .{ .integer = 0 } else input;
}

fn negative(input: Value) !bool {
    return greater(.{ .integer = 0 }, input);
}

fn anthropic(provider: []const u8) bool {
    for ([_][]const u8{ "anthropic", "aws-anthropic", "bedrock-anthropic", "azure-anthropic", "vertex-anthropic", "deepseek-anthropic", "meta-anthropic", "moonshotai-anthropic" }) |name|
        if (equal(provider, name)) return true;
    return false;
}

fn equal(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn parse(arena: Allocator, json: []const u8) Allocator.Error!Value {
    return std.json.parseFromSliceLeaky(Value, arena, json, .{ .allocate = .alloc_always }) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => unreachable,
    };
}

fn put(arena: Allocator, object: *ObjectMap, key: []const u8, input: Value) Allocator.Error!void {
    try object.put(arena, try arena.dupe(u8, key), try value.cloneValue(arena, input));
}

fn unknown(arena: Allocator, reason: []const u8) Allocator.Error!Estimate {
    var details: ObjectMap = .empty;
    try put(arena, &details, "kind", .{ .string = "unknown" });
    try put(arena, &details, "reason", .{ .string = reason });
    return .{ .cost = .null, .details = .{ .object = details } };
}

fn estimateDetails(arena: Allocator, catalogue: *const model_prices.Catalogue, facts: *const Facts, provider: []const u8, model: []const u8) !Value {
    var metadata = try catalogue.sourceInfoValue(arena);
    if (facts.namespaces) |names| {
        var items: std.array_list.Managed(Value) = .init(arena);
        for (names) |name| try items.append(.{ .string = try arena.dupe(u8, name) });
        try metadata.object.put(arena, try arena.dupe(u8, "namespaces"), .{ .array = items });
    }
    var details: ObjectMap = .empty;
    try put(arena, &details, "kind", .{ .string = "estimate" });
    try put(arena, &details, "currency", .{ .string = "USD" });
    try put(arena, &details, "provider", .{ .string = provider });
    try put(arena, &details, "model", .{ .string = model });
    try details.put(arena, try arena.dupe(u8, "metadata"), metadata);
    return .{ .object = details };
}

const testing = std.testing;
const fixture =
    \\{"openai/m":{"litellm_provider":"openai","input_cost_per_token":0.5,
    \\"output_cost_per_token":0.25,"input_cost_per_token_above_200k_tokens":2.0,
    \\"input_cost_per_token_priority":4.0,"cache_read_input_token_cost":0.125},
    \\"openai/u":{"litellm_provider":"openai","input_cost_per_token":0.5,
    \\"output_cost_per_token":0.25,"input_cost_per_token_above_٢٠٠k_tokens":2.0},
    \\"openai/z":{"litellm_provider":"openai","input_cost_per_token":0.0,
    \\"output_cost_per_token":0.0},
    \\"openai/s":{"litellm_provider":"openai","input_cost_per_token":0.1,
    \\"output_cost_per_token":0.2,"cache_read_input_token_cost":0.3},
    \\"anthropic/c":{"litellm_provider":"anthropic","input_cost_per_token":0.5,
    \\"output_cost_per_token":0.25,"cache_creation_input_token_cost":1.0}}
;

fn priced(arena: Allocator, facts: Facts) !Estimate {
    const catalogue = try model_prices.Catalogue.init(arena, fixture, .{});
    return estimate(arena, &catalogue, &facts);
}

fn reasonOf(result: Estimate) []const u8 {
    return result.details.object.get("reason").?.string;
}

fn counted(input: i64, output: i64) Usage {
    return .{ .input_tokens = .{ .integer = input }, .output_tokens = .{ .integer = output } };
}

test "reported cost is authoritative only without declared namespaces" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const zero = try parse(arena, "{\"_hidden_params\":{\"response_cost\":0}}");
    const reported = try priced(arena, .{ .requested_model = "m", .provider_data = zero });
    try testing.expectEqual(@as(i64, 0), reported.cost.integer);
    try testing.expectEqualStrings("reported", reported.details.object.get("kind").?.string);

    const text = try parse(arena, "{\"_hidden_params\":{\"response_cost\":{\"usd\":\"1\"}}}");
    const object = try priced(arena, .{ .requested_model = "m", .provider_data = text });
    try testing.expectEqualStrings("1", object.cost.object.get("usd").?.string);

    const declared = try priced(arena, .{ .requested_model = "m", .provider_data = zero, .namespaces = &.{"openai"} });
    try testing.expect(declared.cost == .null);
    try testing.expectEqualStrings("custom engine has no pricing provider", reasonOf(declared));

    const hidden_null = try parse(arena, "{\"_hidden_params\":null}");
    const refused = try priced(arena, .{ .requested_model = "m", .provider_data = hidden_null });
    try testing.expectEqualStrings("pricing metadata could not be interpreted", reasonOf(refused));
}

test "threshold is strict and a Unicode digit key selects its ASCII rate key" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const at = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(200_000, 0) });
    try testing.expectEqual(@as(f64, 100_000.0), at.cost.float);
    const above = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(200_001, 0) });
    try testing.expectEqual(@as(f64, 400_002.0), above.cost.float);
    // The priority tier owns no threshold key, so its flat rate survives.
    const priority = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(200_001, 0), .service_tier = "priority" });
    try testing.expectEqual(@as(f64, 800_004.0), priority.cost.float);
    const unicode_key = try priced(arena, .{ .provider = "openai", .requested_model = "u", .response_model = "u", .usage = counted(200_001, 0) });
    try testing.expectEqualStrings("a billed token dimension has no known rate", reasonOf(unicode_key));
}

test "served tier overrides the requested tier and malformed tiers are refused" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const served = try parse(arena, "{\"service_tier\":\"default\"}");
    const standard = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(2, 0), .provider_data = served, .service_tier = "priority" });
    try testing.expectEqual(@as(f64, 1.0), standard.cost.float);
    const nested = try parse(arena, "{\"service_tier\":\"\",\"usageMetadata\":{\"serviceTier\":\"priority\"}}");
    const fallback = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(2, 0), .provider_data = nested });
    try testing.expectEqual(@as(f64, 8.0), fallback.cost.float);
    const list = try parse(arena, "{\"service_tier\":[1]}");
    const unhashable = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(2, 0), .provider_data = list });
    try testing.expectEqualStrings("pricing metadata could not be interpreted", reasonOf(unhashable));
    const empty = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = counted(2, 0), .service_tier = "" });
    try testing.expectEqualStrings("unrecognised service-tier pricing", reasonOf(empty));
}

test "counts subtract exactly, sum in Source order and refuse float overflow" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    var wide = counted(0, 0);
    wide.input_tokens = .{ .number_string = "1000000000000000000000000000003" };
    wide.cache_read_tokens = .{ .number_string = "1000000000000000000000000000000" };
    const exact = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = wide });
    // Three uncached tokens at 2.0 above the threshold, then 1e30 cached at 0.125.
    try testing.expectEqual(@as(f64, 6.0) + 1e30 * 0.125, exact.cost.float);

    var ordered = counted(2, 1);
    ordered.cache_read_tokens = .{ .integer = 1 };
    const sum = try priced(arena, .{ .provider = "openai", .requested_model = "s", .response_model = "s", .usage = ordered });
    try testing.expectEqual(@as(f64, 0.6000000000000001), sum.cost.float);

    const nothing = try priced(arena, .{ .provider = "openai", .requested_model = "z", .response_model = "z", .usage = counted(0, 0) });
    try testing.expectEqual(@as(i64, 0), nothing.cost.integer);

    const digits: [400]u8 = @splat('9');
    var huge = counted(0, 0);
    huge.input_tokens = .{ .number_string = &digits };
    const overflow = try priced(arena, .{ .provider = "openai", .requested_model = "z", .response_model = "z", .usage = huge });
    try testing.expect(overflow.cost == .null);
    try testing.expectEqualStrings("pricing metadata could not be interpreted", reasonOf(overflow));
}

test "long Anthropic cache writes have no ordinary-rate fallback" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    var usage = counted(3, 0);
    usage.cache_write_tokens = .{ .integer = 1 };
    const short = try priced(arena, .{ .provider = "anthropic", .requested_model = "c", .response_model = "c", .usage = usage });
    try testing.expectEqual(@as(f64, 2.0), short.cost.float);
    const long = try priced(arena, .{ .provider = "anthropic", .requested_model = "c", .response_model = "c", .usage = usage, .long_cache_retention = true });
    try testing.expectEqualStrings("a billed token dimension has no known rate", reasonOf(long));
    // Retention is an Anthropic-route fact only; other providers keep the ordinary key.
    const other = try priced(arena, .{ .provider = "openai", .requested_model = "m", .response_model = "m", .usage = usage, .long_cache_retention = true });
    try testing.expectEqual(@as(f64, 1.5), other.cost.float);
}

test "oversized matched threshold captures are advisory errors but nonmatches are ignored" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const arena = state.allocator();
    const digits: [4301]u8 = @splat('0');
    for ([_][]const u8{ "", "x" }) |trailing| {
        const json = try std.fmt.allocPrint(arena, "{{\"openai/m\":{{\"litellm_provider\":\"openai\",\"input_cost_per_token\":0.5," ++
            "\"input_cost_per_token_above_{s}{s}k_tokens\":2.0}}}}", .{ &digits, trailing });
        const catalogue = try model_prices.Catalogue.init(arena, json, .{});
        const facts: Facts = .{
            .provider = "openai",
            .requested_model = "m",
            .response_model = "m",
            .usage = counted(1, 0),
        };
        const result = try estimate(arena, &catalogue, &facts);
        if (trailing.len == 0) {
            try testing.expect(result.cost == .null);
            try testing.expectEqualStrings("pricing metadata could not be interpreted", reasonOf(result));
        } else {
            try testing.expectEqual(@as(f64, 0.5), result.cost.float);
        }
    }
}
