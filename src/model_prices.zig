//! Advisory model price catalogue: the bundled LiteLLM snapshot and the lookup
//! DSPy 3.5's `dspy.clients.model_metadata.model_info` performs on it.
//!
//! A catalogue is built once at startup and is immutable afterwards. Building
//! parses the JSON text into the caller's arena and indexes it; a lookup then
//! allocates nothing and returns a borrowed entry.
//!
//! Loading is local only. The source reads a remote map unless
//! `LITELLM_LOCAL_MODEL_COST_MAP=true`; bundled() is that forced-local mode and
//! nothing here opens a connection. A caller that obtains other catalogue text
//! passes it to init() with a SourceInfo describing where it came from. The
//! source's shrink check on a fetched map belongs to such a refresh, not here.
//!
//! Differences from the source, all refusals rather than different answers:
//!   - Capability rules name Python regular expressions. Only the pattern
//!     texts of the bundled snapshot are ported (model_prices_patterns.zig);
//!     init() refuses any other pattern text with UnsupportedPricingPattern,
//!     including one the source would skip as an invalid expression.
//!   - A rule's `fill_missing_for_providers` must be absent, null or a list of
//!     strings; other shapes are refused with UnsupportedPricingRule.
//!   - JSON the native parser refuses (NaN, Infinity, a byte order mark) fails
//!     init() although Python's `json.loads` accepts it.
//!
//! Metadata is advisory and never routes a request.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");
const patterns = @import("model_prices_patterns.zig");
const Value = runtime.Value;
const ObjectMap = runtime.ObjectMap;

/// The packaged `snapshot.json.gz` of DSPy 3.5, decompressed and unmodified.
pub const snapshot_json = @embedFile("model_prices_snapshot.json");
/// The licence notice that must accompany the snapshot wherever it is shipped.
pub const snapshot_license = @embedFile("model_prices_LICENSE");
/// LiteLLM release the snapshot was taken from, as its provenance records.
pub const snapshot_version = "1.108.3";
/// SHA-256 of snapshot_json, as the same provenance records.
pub const snapshot_sha256 = "88c7b1aca7842e08f2e38b29b90b735c753c82b159ef946b1788066fd596eb49";
/// The source refuses a fetched map above 32 MiB; init() applies it to any text.
pub const catalogue_bytes_max = 32 * 1024 * 1024;

const reserved_keys = [_][]const u8{ "sample_spec", "fallback_generalizations" };
const capability_prefix = "supports_";

/// What `source_info()` reports about the loaded map.
pub const SourceInfo = struct {
    source: []const u8 = "local",
    is_env_forced: bool = true,
    fallback_reason: ?[]const u8 = null,
    snapshot_version: []const u8 = snapshot_version,
};

/// The result of one lookup.
///
/// `entry` is the single matching catalogue entry, borrowed from the
/// catalogue, or null when no entry matched. Rates are read from it alone:
/// they never come from another entry or from a rule.
///
/// `has_info` is the truthiness of the dictionary the source's `model_info`
/// returns. It is true for a non-empty entry, and also when only capability
/// fields were inherited from the provider's bare entry or supplied by a rule,
/// so it can be true while `entry` is null or empty. The inherited capability
/// fields themselves are not exposed; a consumer reads `entry` as an empty
/// dictionary in that case.
pub const Lookup = struct {
    entry: ?*const ObjectMap,
    has_info: bool,
};

pub const LookupError = error{
    /// `fallback_generalizations` is present but not an object. The source
    /// raises on every lookup that reaches it.
    InvalidPricingRules,
    /// A matching rule's `model_info` is neither empty nor an object. The
    /// source raises when such a rule matches.
    InvalidPricingRule,
};

const Rule = struct {
    pattern: patterns.Pattern,
    /// `fill_missing_for_providers`, or null when the rule names none.
    providers: ?[]const []const u8,
    /// The rule's `model_info` has a capability field.
    supports: bool,
    /// The rule's `model_info` makes the source raise when the rule matches.
    fails: bool,
};

/// One string of an entry's `aliases`, in entry order and then list order.
const Alias = struct {
    entry: *const ObjectMap,
    provider: []const u8,
    name: []const u8,
};

pub const Catalogue = struct {
    data: ObjectMap,
    /// Indexes of the keys holding a slash, ordered by key bytes. A namespaced
    /// key is searched here without building its text.
    namespaced: []const u32,
    aliases: []const Alias,
    rules: []const Rule,
    rules_valid: bool,
    rules_can_fail: bool,
    info: SourceInfo,

    /// Parses and indexes catalogue text. Everything the catalogue keeps is
    /// copied into `arena`, so `json` and the strings of `info` may be released
    /// afterwards; the catalogue is valid while `arena` lives. Parsing leaves
    /// its scratch in `arena`, so use a startup arena.
    ///
    /// Refused: text above catalogue_bytes_max, a root that is not an object,
    /// an entry other than the two reserved ones that is not an object, and
    /// the unsupported rule shapes this file's header lists.
    pub fn init(arena: Allocator, json: []const u8, info: SourceInfo) !Catalogue {
        if (json.len > catalogue_bytes_max) return error.CatalogueTooLarge;
        // Python's json.loads keeps the last value of a repeated key.
        const root = try std.json.parseFromSliceLeaky(Value, arena, json, .{
            .allocate = .alloc_always,
            .duplicate_field_behavior = .use_last,
        });
        if (root != .object) return error.InvalidCatalogue;
        const data = root.object;
        for (data.keys(), data.values()) |key, entry| {
            if (isReserved(key)) continue;
            if (entry != .object) return error.InvalidCatalogue;
        }

        var catalogue: Catalogue = .{
            .data = data,
            .namespaced = try namespacedIndex(arena, &data),
            .aliases = try aliasIndex(arena, &data),
            .rules = &.{},
            .rules_valid = true,
            .rules_can_fail = false,
            .info = .{
                .source = try arena.dupe(u8, info.source),
                .is_env_forced = info.is_env_forced,
                .fallback_reason = if (info.fallback_reason) |reason|
                    try arena.dupe(u8, reason)
                else
                    null,
                .snapshot_version = try arena.dupe(u8, info.snapshot_version),
            },
        };
        if (data.getPtr("fallback_generalizations")) |fallback| {
            if (fallback.* == .object) {
                catalogue.rules = try compileRules(arena, &fallback.object);
            } else {
                catalogue.rules_valid = false;
            }
        }
        for (catalogue.rules) |rule| {
            if (rule.fails) catalogue.rules_can_fail = true;
        }
        return catalogue;
    }

    /// The bundled snapshot, loaded as the source loads it when the local map
    /// is forced: no fetch is attempted and none is reported as failed.
    pub fn bundled(arena: Allocator) !Catalogue {
        return init(arena, snapshot_json, .{
            .source = "local",
            .is_env_forced = true,
            .fallback_reason = null,
            .snapshot_version = snapshot_version,
        });
    }

    /// `source_info()` as a dictionary in the source's key order. The result
    /// owns its keys and strings in `arena` and borrows nothing.
    pub fn sourceInfoValue(self: *const Catalogue, arena: Allocator) Allocator.Error!Value {
        var object: ObjectMap = .empty;
        try object.put(arena, try arena.dupe(u8, "source"), .{
            .string = try arena.dupe(u8, self.info.source),
        });
        try object.put(arena, try arena.dupe(u8, "is_env_forced"), .{
            .bool = self.info.is_env_forced,
        });
        try object.put(
            arena,
            try arena.dupe(u8, "fallback_reason"),
            if (self.info.fallback_reason) |reason|
                .{ .string = try arena.dupe(u8, reason) }
            else
                .null,
        );
        try object.put(arena, try arena.dupe(u8, "snapshot_version"), .{
            .string = try arena.dupe(u8, self.info.snapshot_version),
        });
        assert(object.count() == 4);
        return .{ .object = object };
    }

    /// `model_info(provider, model, namespaces=namespaces)`.
    ///
    /// `namespaces` names the catalogue namespaces to read; null selects the
    /// provider's own, and an empty list reads nothing. The entry is the first
    /// of: `namespace/model` in namespace order, the bare `model` entry when
    /// its `litellm_provider` is one of the namespaces, then the first entry in
    /// catalogue order of such a provider that lists the model as an alias.
    ///
    /// Allocates nothing. The result borrows the catalogue only; `provider`,
    /// `model` and `namespaces` are not retained.
    pub fn lookup(
        self: *const Catalogue,
        provider: []const u8,
        model: []const u8,
        namespaces: ?[]const []const u8,
    ) LookupError!Lookup {
        const own = [1][]const u8{provider};
        const names: []const []const u8 = namespaces orelse (providerNames(provider) orelse &own);
        if (names.len == 0) return .{ .entry = null, .has_info = false };

        const bare = self.bareEntry(model, names);
        const found = self.findEntry(model, names, bare);
        if (!self.rules_valid) return error.InvalidPricingRules;

        var has_info = false;
        if (found) |entry| has_info = entry.count() > 0;
        if (!has_info) {
            // Capabilities come from the bare entry even when another entry,
            // or none, was found. A null capability still fills a key.
            if (bare) |entry| has_info = hasCapability(entry);
        }
        for (self.rules) |*rule| {
            // Once the answer is true only a failing rule can change it.
            if (has_info) {
                if (!self.rules_can_fail) break;
                if (!rule.fails) continue;
            }
            if (!rule.fails) {
                if (!rule.supports) continue;
            }
            if (found) |entry| {
                if (rule.providers) |listed| {
                    const entry_provider = stringField(entry, "litellm_provider") orelse continue;
                    if (!contains(listed, entry_provider)) continue;
                }
            }
            if (!patterns.search(rule.pattern, model)) continue;
            if (rule.fails) return error.InvalidPricingRule;
            assert(rule.supports);
            has_info = true;
        }
        return .{ .entry = found, .has_info = has_info };
    }

    /// The bare `model` entry when its provider is one of `names`. A reserved
    /// key qualifies here exactly as it does in the source.
    fn bareEntry(
        self: *const Catalogue,
        model: []const u8,
        names: []const []const u8,
    ) ?*const ObjectMap {
        const value = self.data.getPtr(model) orelse return null;
        if (value.* != .object) return null;
        const provider = stringField(&value.object, "litellm_provider") orelse return null;
        if (!contains(names, provider)) return null;
        return &value.object;
    }

    fn findEntry(
        self: *const Catalogue,
        model: []const u8,
        names: []const []const u8,
        bare: ?*const ObjectMap,
    ) ?*const ObjectMap {
        for (names) |name| {
            if (self.namespacedEntry(name, model)) |entry| return entry;
        }
        if (bare) |entry| {
            if (!isReserved(model)) return entry;
        }
        for (self.aliases) |*alias| {
            if (!contains(names, alias.provider)) continue;
            if (std.mem.eql(u8, alias.name, model)) return alias.entry;
            for (names) |name| {
                if (orderNamespaced(name, model, alias.name) == .eq) return alias.entry;
            }
        }
        return null;
    }

    /// The entry stored under `name/model`, found by binary search.
    fn namespacedEntry(
        self: *const Catalogue,
        name: []const u8,
        model: []const u8,
    ) ?*const ObjectMap {
        const keys = self.data.keys();
        var low: usize = 0;
        var high: usize = self.namespaced.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const index = self.namespaced[middle];
            switch (orderNamespaced(name, model, keys[index])) {
                .lt => high = middle,
                .gt => low = middle + 1,
                .eq => {
                    const entry = &self.data.values()[index];
                    // A key holding a slash is never reserved, so init()
                    // checked that this entry is an object.
                    assert(entry.* == .object);
                    return &entry.object;
                },
            }
        }
        return null;
    }
};

fn isReserved(key: []const u8) bool {
    return contains(&reserved_keys, key);
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |candidate| {
        if (std.mem.eql(u8, candidate, name)) return true;
    }
    return false;
}

fn stringField(entry: *const ObjectMap, key: []const u8) ?[]const u8 {
    const value = entry.getPtr(key) orelse return null;
    return if (value.* == .string) value.string else null;
}

fn hasCapability(entry: *const ObjectMap) bool {
    for (entry.keys()) |key| {
        if (std.mem.startsWith(u8, key, capability_prefix)) return true;
    }
    return false;
}

/// Orders the text `name ++ "/" ++ model` against `key` without building it.
fn orderNamespaced(name: []const u8, model: []const u8, key: []const u8) std.math.Order {
    const length = name.len + 1 + model.len;
    const shared = @min(length, key.len);
    var offset: usize = 0;
    for ([_][]const u8{ name, "/", model }) |part| {
        if (offset == shared) break;
        const take = @min(part.len, shared - offset);
        const order = std.mem.order(u8, part[0..take], key[offset..][0..take]);
        if (order != .eq) return order;
        offset += take;
    }
    assert(offset == shared);
    return std.math.order(length, key.len);
}

fn keyIndexLessThan(keys: []const []const u8, left: u32, right: u32) bool {
    return std.mem.order(u8, keys[left], keys[right]) == .lt;
}

fn namespacedIndex(arena: Allocator, data: *const ObjectMap) Allocator.Error![]const u32 {
    const keys: []const []const u8 = data.keys();
    // The text is at most catalogue_bytes_max, so the key count fits u32.
    assert(keys.len <= catalogue_bytes_max);
    var count: usize = 0;
    for (keys) |key| count += @intFromBool(std.mem.indexOfScalar(u8, key, '/') != null);

    const indexes = try arena.alloc(u32, count);
    var next: usize = 0;
    for (keys, 0..) |key, index| {
        if (std.mem.indexOfScalar(u8, key, '/') == null) continue;
        indexes[next] = @intCast(index);
        next += 1;
    }
    assert(next == count);
    std.mem.sort(u32, indexes, keys, keyIndexLessThan);
    return indexes;
}

fn aliasIndex(arena: Allocator, data: *const ObjectMap) Allocator.Error![]const Alias {
    var aliases: std.ArrayList(Alias) = .empty;
    for (data.keys(), data.values()) |key, *value| {
        if (isReserved(key)) continue;
        assert(value.* == .object);
        const entry = &value.object;
        // Only an entry whose provider is a string can be one of the names.
        const provider = stringField(entry, "litellm_provider") orelse continue;
        const listed = entry.getPtr("aliases") orelse continue;
        if (listed.* != .array) continue;
        for (listed.array.items) |alias| {
            if (alias != .string) continue;
            try aliases.append(arena, .{
                .entry = entry,
                .provider = provider,
                .name = alias.string,
            });
        }
    }
    return aliases.items;
}

/// Python truthiness of a parsed JSON value.
fn truthy(value: Value) bool {
    return switch (value) {
        .null => false,
        .bool => |inner| inner,
        .integer => |inner| inner != 0,
        .float => |inner| inner != 0,
        // Only integers beyond i64 and non-finite floats are kept as text.
        .number_string => true,
        .string => |inner| inner.len != 0,
        .array => |inner| inner.items.len != 0,
        .object => |inner| inner.count() != 0,
        .custom => true,
    };
}

fn compileRules(arena: Allocator, fallback: *const ObjectMap) ![]const Rule {
    const listed = fallback.getPtr("rules") orelse return &.{};
    if (listed.* != .array) return &.{};

    var rules: std.ArrayList(Rule) = .empty;
    for (listed.array.items) |*item| {
        // The source skips a rule that is not an object with a string pattern.
        if (item.* != .object) continue;
        const text = stringField(&item.object, "pattern") orelse continue;
        const pattern = patterns.identify(text) orelse return error.UnsupportedPricingPattern;

        var rule: Rule = .{ .pattern = pattern, .providers = null, .supports = false, .fails = false };
        if (item.object.getPtr("fill_missing_for_providers")) |providers| {
            if (providers.* != .null) rule.providers = try ruleProviders(arena, providers.*);
        }
        if (item.object.getPtr("model_info")) |model_info| {
            if (model_info.* == .object) {
                rule.supports = hasCapability(&model_info.object);
            } else {
                rule.fails = truthy(model_info.*);
            }
        }
        try rules.append(arena, rule);
    }
    return rules.items;
}

fn ruleProviders(arena: Allocator, listed: Value) ![]const []const u8 {
    if (listed != .array) return error.UnsupportedPricingRule;
    const providers = try arena.alloc([]const u8, listed.array.items.len);
    for (listed.array.items, providers) |item, *provider| {
        if (item != .string) return error.UnsupportedPricingRule;
        provider.* = item.string;
    }
    return providers;
}

const Route = struct { provider: []const u8, names: []const []const u8 };

/// Metadata namespaces, not inference routes: a cloud deployment's prices must
/// not fall back to a public provider's. Null means the provider's own name.
fn providerNames(provider: []const u8) ?[]const []const u8 {
    const routes = [_]Route{
        .{ .provider = "openai-chat", .names = &.{"openai"} },
        .{ .provider = "claude-code", .names = &.{"anthropic"} },
        .{ .provider = "openai-codex", .names = &.{"openai"} },
        .{ .provider = "vertex", .names = &.{ "vertex_ai", "gemini" } },
        .{ .provider = "vertex-express", .names = &.{ "vertex_ai", "gemini" } },
        .{ .provider = "vertex-anthropic", .names = &.{"vertex_ai"} },
        .{ .provider = "azure-chat", .names = &.{"azure"} },
        .{ .provider = "azure-anthropic", .names = &.{"azure_ai"} },
        .{ .provider = "bedrock-chat", .names = &.{"bedrock"} },
        .{ .provider = "bedrock-mantle-chat", .names = &.{"bedrock_mantle"} },
        .{ .provider = "bedrock-anthropic", .names = &.{"bedrock"} },
        .{ .provider = "aws-anthropic", .names = &.{"aws-anthropic"} },
        .{ .provider = "deepseek-anthropic", .names = &.{"deepseek"} },
        .{ .provider = "meta-chat", .names = &.{"meta"} },
        .{ .provider = "meta-anthropic", .names = &.{"meta"} },
        .{ .provider = "moonshotai", .names = &.{ "moonshot", "moonshotai" } },
        .{ .provider = "moonshotai-responses", .names = &.{ "moonshot", "moonshotai" } },
        .{ .provider = "moonshotai-anthropic", .names = &.{ "moonshot", "moonshotai" } },
        .{ .provider = "vllm", .names = &.{ "hosted_vllm", "vllm" } },
        .{ .provider = "fireworks", .names = &.{"fireworks_ai"} },
        .{ .provider = "together", .names = &.{"together_ai"} },
    };
    for (routes) |route| {
        if (std.mem.eql(u8, route.provider, provider)) return route.names;
    }
    return null;
}

const testing = std.testing;

fn expectLookup(
    catalogue: *const Catalogue,
    provider: []const u8,
    model: []const u8,
    namespaces: ?[]const []const u8,
    expected_provider: ?[]const u8,
    expected_has_info: bool,
) !?*const ObjectMap {
    const result = try catalogue.lookup(provider, model, namespaces);
    errdefer std.debug.print("provider {s} model {s}\n", .{ provider, model });
    try testing.expectEqual(expected_has_info, result.has_info);
    if (expected_provider) |name| {
        try testing.expectEqualStrings(name, stringField(result.entry.?, "litellm_provider").?);
    } else {
        try testing.expectEqual(@as(?*const ObjectMap, null), result.entry);
    }
    return result.entry;
}

test "bundled lookups keep provider namespaces apart and take rates from one entry" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const catalogue = try Catalogue.bundled(memory.allocator());

    // A public provider reads its bare entry; a cloud deployment of the same
    // model reads its own namespace and never the public price.
    const public = (try expectLookup(&catalogue, "openai", "gpt-4o", null, "openai", true)).?;
    const deployed = (try expectLookup(&catalogue, "azure-chat", "gpt-4o", null, "azure", true)).?;
    try testing.expect(public != deployed);
    try testing.expectEqual(@as(f64, 2.5e-06), public.get("input_cost_per_token").?.float);
    try testing.expectEqual(public, &catalogue.data.getPtr("gpt-4o").?.object);
    try testing.expectEqualStrings("azure", deployed.get("litellm_provider").?.string);
    _ = try expectLookup(&catalogue, "azure-chat", "gpt-3.5-turbo-0125", null, null, false);
    _ = try expectLookup(&catalogue, "openai", "gpt-3.5-turbo-0125", null, "openai", true);

    // The route alias names two namespaces in order; the bare entry belongs to
    // a third provider and is not read.
    _ = try expectLookup(&catalogue, "vertex", "gemini-2.5-pro", null, "gemini", true);
    _ = try expectLookup(&catalogue, "vertex", "gemini-3.8-live", null, "vertex_ai", true);
    _ = try expectLookup(&catalogue, "gemini", "gemini-3.8-live", null, "gemini", true);

    // Explicit namespaces replace the provider's own; an empty list reads nothing.
    _ = try expectLookup(&catalogue, "openai", "gpt-4o", &.{"azure"}, "azure", true);
    _ = try expectLookup(&catalogue, "azure-chat", "gpt-4o", &.{"openai"}, "openai", true);
    _ = try expectLookup(&catalogue, "openai", "gpt-4o", &.{}, null, false);
    _ = try expectLookup(&catalogue, "openai", "gpt-4o", &.{"nowhere"}, null, false);
    _ = try expectLookup(&catalogue, "openai", "gpt-5", &.{"nowhere"}, null, true);
}

test "bundled rules give capability truthiness without an entry or a price" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const catalogue = try Catalogue.bundled(memory.allocator());

    // Unmapped ids of a known family have information but no entry.
    _ = try expectLookup(&catalogue, "anthropic", "claude-newfamily-5", null, null, true);
    _ = try expectLookup(&catalogue, "openai", "gpt-9-zeta", null, null, true);
    _ = try expectLookup(&catalogue, "wandb", "wandb/new-model", null, null, true);
    _ = try expectLookup(&catalogue, "vertex", "gemini-7-pro", null, null, true);

    // Ids no rule describes have none, so a caller may fall back to another id.
    _ = try expectLookup(&catalogue, "bedrock-chat", "anthropic.claude-3-haiku-20240307-v1:0", null, null, false);
    _ = try expectLookup(&catalogue, "wandb", "new-model", null, null, false);
    _ = try expectLookup(&catalogue, "openai", "gpt-4-unmapped", null, null, false);
    _ = try expectLookup(&catalogue, "vertex", "gemini-7-pro-tts", null, null, false);
    _ = try expectLookup(&catalogue, "local", "", null, null, false);

    // An exact entry is found by its namespace even where no rule matches it.
    _ = try expectLookup(&catalogue, "wandb", "meta-llama/Llama-3.1-8B-Instruct", null, "wandb", true);
}

fn parse(arena: Allocator, json: []const u8) !Catalogue {
    return Catalogue.init(arena, json, .{});
}

test "namespace key, bare entry and alias are tried in that order" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const catalogue = try parse(memory.allocator(),
        \\{"z/first": {"litellm_provider": "a", "aliases": ["m", 7, "b/q", "solo"], "rank": 1},
        \\ "b/second": {"litellm_provider": "b", "aliases": ["m", "n"], "rank": 2},
        \\ "c/third": {"litellm_provider": "zz", "aliases": "m", "rank": 3},
        \\ "n": {"litellm_provider": "b", "rank": 4},
        \\ "b/n": {"litellm_provider": "elsewhere", "rank": 5},
        \\ "c/n": {"litellm_provider": "c", "rank": 6},
        \\ "solo": {"litellm_provider": "b", "rank": 7}}
    );
    const rank = struct {
        fn of(source: *const Catalogue, provider: []const u8, model: []const u8, names: ?[]const []const u8) !?i64 {
            const entry = (try source.lookup(provider, model, names)).entry orelse return null;
            return entry.get("rank").?.integer;
        }
    }.of;

    // The namespaced key wins over the bare entry, which wins over an alias.
    try testing.expectEqual(@as(?i64, 5), try rank(&catalogue, "b", "n", null));
    try testing.expectEqual(@as(?i64, 6), try rank(&catalogue, "b", "n", &.{ "c", "b" }));
    try testing.expectEqual(@as(?i64, 5), try rank(&catalogue, "x", "n", &.{ "missing", "b" }));
    try testing.expectEqual(@as(?i64, 7), try rank(&catalogue, "x", "solo", &.{ "a", "b" }));
    try testing.expectEqual(@as(?i64, 7), try rank(&catalogue, "b", "solo", null));
    try testing.expectEqual(@as(?i64, 1), try rank(&catalogue, "a", "solo", null));
    try testing.expectEqual(@as(?i64, null), try rank(&catalogue, "c", "solo", null));

    // An alias belongs to the entry's provider field, not to its key, and the
    // first entry in catalogue order wins whatever the namespace order is.
    try testing.expectEqual(@as(?i64, 2), try rank(&catalogue, "b", "m", null));
    try testing.expectEqual(@as(?i64, 1), try rank(&catalogue, "a", "m", null));
    try testing.expectEqual(@as(?i64, 1), try rank(&catalogue, "x", "m", &.{ "b", "a" }));
    try testing.expectEqual(@as(?i64, null), try rank(&catalogue, "z", "m", null));
    try testing.expectEqual(@as(?i64, null), try rank(&catalogue, "c", "m", null));
    // A string is not a list of aliases, and a number is not an alias.
    try testing.expectEqual(@as(?i64, null), try rank(&catalogue, "zz", "m", null));
    try testing.expectEqual(@as(?i64, null), try rank(&catalogue, "a", "7", null));
    // An alias may spell a namespaced key of any requested namespace.
    try testing.expectEqual(@as(?i64, 1), try rank(&catalogue, "x", "q", &.{ "a", "b" }));
    try testing.expectEqual(@as(?i64, null), try rank(&catalogue, "a", "q", null));
    try testing.expectEqual(@as(?i64, 1), try rank(&catalogue, "a", "b/q", null));
}

test "capabilities inherit from the bare entry while rates never do" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const catalogue = try parse(memory.allocator(),
        \\{"priced": {"litellm_provider": "p", "input_cost_per_token": 1, "supports_vision": true},
        \\ "p/priced": {},
        \\ "q/priced": {},
        \\ "nulled": {"litellm_provider": "p", "supports_vision": null},
        \\ "p/nulled": {},
        \\ "plain": {"litellm_provider": "p", "input_cost_per_token": 1, "support_vision": true},
        \\ "p/plain": {},
        \\ "sample_spec": {"litellm_provider": "p", "input_cost_per_token": 9, "supports_vision": true}}
    );

    // The empty namespaced entry is the match: information exists through the
    // inherited capability, and no rate is readable.
    const inherited = try catalogue.lookup("p", "priced", null);
    try testing.expect(inherited.has_info);
    try testing.expectEqual(@as(usize, 0), inherited.entry.?.count());
    try testing.expect(inherited.entry.?.get("input_cost_per_token") == null);

    // Another provider's empty entry inherits nothing and stays falsy.
    const isolated = try catalogue.lookup("q", "priced", null);
    try testing.expect(!isolated.has_info);
    try testing.expectEqual(@as(usize, 0), isolated.entry.?.count());

    // A null capability still fills a key; a non-capability field does not.
    try testing.expect((try catalogue.lookup("p", "nulled", null)).has_info);
    try testing.expect(!(try catalogue.lookup("p", "plain", null)).has_info);

    // A reserved key is never an entry, yet it is the bare capability source.
    const reserved = try catalogue.lookup("p", "sample_spec", null);
    try testing.expectEqual(@as(?*const ObjectMap, null), reserved.entry);
    try testing.expect(reserved.has_info);
    try testing.expect(!(try catalogue.lookup("q", "sample_spec", null)).has_info);
}

test "a rule fills only a missing entry or one of its listed providers" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const catalogue = try parse(memory.allocator(),
        \\{"p/wandb/empty": {},
        \\ "p/wandb/other": {"aliases": []},
        \\ "fallback_generalizations": {"rules": [
        \\   "skipped", {"pattern": 5, "model_info": 1}, {"model_info": {"supports_x": true}},
        \\   {"pattern": "^wandb/", "fill_missing_for_providers": ["openai"],
        \\    "model_info": {"supports_reasoning": true}},
        \\   {"pattern": "claude-(?:fable|mythos)-", "model_info": {"thinking_always_on": true}},
        \\   {"pattern": "^(?:[a-z-]+\\.)?anthropic\\.claude-", "model_info": 0},
        \\   {"pattern": "claude-[a-z]+-4[-._]6(?!\\d)", "fill_missing_for_providers": null,
        \\    "model_info": {"supports_legacy_thinking": false}}]}}
    );
    try testing.expectEqual(@as(usize, 4), catalogue.rules.len);
    try testing.expect(!catalogue.rules_can_fail);

    // No entry: the provider list does not apply and the rule decides.
    try testing.expect((try catalogue.lookup("p", "wandb/missing", null)).has_info);
    // An empty entry has no provider, so the rule is skipped and nothing fills.
    try testing.expect(!(try catalogue.lookup("p", "wandb/empty", null)).has_info);
    // A non-empty entry is information by itself.
    try testing.expect((try catalogue.lookup("p", "wandb/other", null)).has_info);
    // A matching rule without a capability field supplies nothing.
    try testing.expect(!(try catalogue.lookup("p", "claude-fable-5", null)).has_info);
    try testing.expect(!(try catalogue.lookup("p", "anthropic.claude-x", null)).has_info);
    // An explicit false capability is still a filled key.
    try testing.expect((try catalogue.lookup("p", "claude-opus-4-6", null)).has_info);
    try testing.expect(!(try catalogue.lookup("p", "claude-opus-4-7", null)).has_info);
}

test "unsupported rules are refused and malformed rules fail as the source does" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    try testing.expectError(error.UnsupportedPricingPattern, parse(arena,
        \\{"fallback_generalizations": {"rules": [{"pattern": "^custom-", "model_info": {}}]}}
    ));
    try testing.expectError(error.UnsupportedPricingPattern, parse(arena,
        \\{"fallback_generalizations": {"rules": [{"pattern": "("}]}}
    ));
    try testing.expectError(error.UnsupportedPricingRule, parse(arena,
        \\{"fallback_generalizations": {"rules": [
        \\  {"pattern": "^wandb/", "fill_missing_for_providers": "openai"}]}}
    ));
    try testing.expectError(error.UnsupportedPricingRule, parse(arena,
        \\{"fallback_generalizations": {"rules": [
        \\  {"pattern": "^wandb/", "fill_missing_for_providers": ["openai", null]}]}}
    ));

    // Rules that are not a list are no rules at all.
    const unlisted = try parse(arena,
        \\{"fallback_generalizations": {"rules": {"pattern": "^custom-"}}}
    );
    try testing.expect(!(try unlisted.lookup("p", "custom-x", null)).has_info);

    // A truthy model_info that is not an object fails only when its rule matches,
    // even after an entry already answered.
    const failing = try parse(arena,
        \\{"p/wandb/known": {"litellm_provider": "p", "input_cost_per_token": 1},
        \\ "fallback_generalizations": {"rules": [{"pattern": "^wandb/", "model_info": [1]}]}}
    );
    try testing.expect(failing.rules_can_fail);
    try testing.expectError(error.InvalidPricingRule, failing.lookup("p", "wandb/known", null));
    try testing.expectError(error.InvalidPricingRule, failing.lookup("p", "wandb/missing", null));
    try testing.expect(!(try failing.lookup("p", "other", null)).has_info);

    // A fallback section that is not an object fails every lookup that reads it.
    const broken = try parse(arena,
        \\{"m": {"litellm_provider": "p", "input_cost_per_token": 1},
        \\ "fallback_generalizations": null}
    );
    try testing.expectError(error.InvalidPricingRules, broken.lookup("p", "m", null));
    try testing.expect(!(try broken.lookup("p", "m", &.{})).has_info);
}

test "init refuses malformed catalogues and owns what it keeps" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();

    try testing.expectError(error.InvalidCatalogue, parse(arena, "[]"));
    try testing.expectError(error.InvalidCatalogue, parse(arena, "{\"m\": 1}"));
    try testing.expectError(error.InvalidCatalogue, parse(arena, "{\"m\": {}, \"n\": null}"));
    _ = try parse(arena, "{\"sample_spec\": 1, \"m\": {}}");

    const oversized = try testing.allocator.alloc(u8, catalogue_bytes_max + 1);
    defer testing.allocator.free(oversized);
    @memset(oversized, ' ');
    try testing.expectError(error.CatalogueTooLarge, parse(arena, oversized));

    // A repeated key keeps its last value, as Python's json.loads does.
    const repeated = try parse(arena,
        \\{"p/m": {"litellm_provider": "p", "rank": 1}, "p/m": {"litellm_provider": "p", "rank": 2}}
    );
    try testing.expectEqual(@as(i64, 2), (try repeated.lookup("p", "m", null)).entry.?.get("rank").?.integer);

    // The caller's text and source strings may be reused once init returns.
    var json = "{\"p/m\": {\"litellm_provider\": \"p\", \"aliases\": [\"other\"]}}".*;
    var source = "remote".*;
    var reason = "TimeoutError".*;
    var version = "9.9.9".*;
    const owned = try Catalogue.init(arena, &json, .{
        .source = &source,
        .is_env_forced = false,
        .fallback_reason = &reason,
        .snapshot_version = &version,
    });
    @memset(&json, ' ');
    @memset(&source, 'x');
    @memset(&reason, 'x');
    @memset(&version, 'x');
    _ = try expectLookup(&owned, "p", "m", null, "p", true);
    _ = try expectLookup(&owned, "p", "other", null, "p", true);
    const info = try owned.sourceInfoValue(arena);
    try testing.expectEqualStrings("remote", info.object.get("source").?.string);
    try testing.expect(!info.object.get("is_env_forced").?.bool);
    try testing.expectEqualStrings("TimeoutError", info.object.get("fallback_reason").?.string);
    try testing.expectEqualStrings("9.9.9", info.object.get("snapshot_version").?.string);
}
