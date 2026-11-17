//! SDK token usage: eight optional exact counters and their two projections.
//!
//! A counter is unknown (`.null`) or an exact nonnegative Python integer.
//! Unknown is never zero: relations are checked only between known counters,
//! and a total is inferred only when both of its addends are known.
//!
//! Usage tracking, pricing and provider-specific extra fields are not modeled.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const runtime = @import("runtime_value.zig");
const value = @import("value.zig");
const python = @import("python_value.zig");
const Value = runtime.Value;
const ObjectMap = runtime.ObjectMap;

pub const Error = Allocator.Error || error{InvalidUsage};

pub const Usage = struct {
    input_tokens: Value = .null,
    output_tokens: Value = .null,
    total_tokens: Value = .null,
    cache_read_tokens: Value = .null,
    cache_write_tokens: Value = .null,
    reasoning_tokens: Value = .null,
    input_audio_tokens: Value = .null,
    output_audio_tokens: Value = .null,

    const names = @typeInfo(Usage).@"struct".field_names;
    const Counter = struct { key: []const u8, count: Value };

    /// Validates raw counters as the SDK constructor does. Each raw counter is
    /// `.null`, an exact integer (`.integer`, integer `.number_string`, or a
    /// genuine int-subclass box such as an IntEnum member), or an exact builtin
    /// float (`.float` or its builtin box) holding a finite integral value,
    /// which becomes a plain integer. Everything else is refused, including
    /// bool, text, Decimal and ordinary Enum members.
    ///
    /// A validated counter is `.null`, `.integer`, canonical `.number_string`
    /// beyond i64, or the caller's own int-subclass box.
    ///
    /// The result borrows: number text and boxes stay the caller's, while
    /// converted wide floats and inferred wide totals live in `arena`. It is
    /// valid while both live and must not be passed to free(); clone() it to
    /// own it. A refusal leaves its scratch allocations in `arena`.
    pub fn init(arena: Allocator, raw: Usage) Error!Usage {
        var usage: Usage = .{};
        inline for (names) |name|
            @field(usage, name) = try count(arena, @field(raw, name));
        try usage.checkParts(arena);

        if (usage.input_tokens != .null) {
            if (usage.output_tokens != .null) {
                const whole = try sum(arena, usage.input_tokens, usage.output_tokens);
                if (usage.total_tokens == .null) {
                    usage.total_tokens = whole;
                } else {
                    if (order(usage.total_tokens, whole) != .eq) return error.InvalidUsage;
                }
            }
        }
        usage.assertCounts();
        return usage;
    }

    /// Reads an SDK usage dictionary: the eight known names are looked up, an
    /// absent or null name is unknown, and every other key is ignored. Only a
    /// native `.object` is a dictionary here. Lifetime is that of init(), with
    /// `source` as the borrowed input.
    pub fn fromValue(arena: Allocator, source: Value) Error!Usage {
        if (source != .object) return error.InvalidUsage;
        var raw: Usage = .{};
        inline for (names) |name| {
            if (source.object.get(name)) |counter| @field(raw, name) = counter;
        }
        return init(arena, raw);
    }

    /// Sparse SDK dictionary: known counters only, in declaration order. A
    /// reported zero is kept; an entirely unknown usage is an empty object.
    /// The result owns its keys and values, borrows nothing from this usage,
    /// and may be released with runtime.freeValue using the same allocator.
    pub fn toValue(usage: *const Usage, arena: Allocator) Allocator.Error!Value {
        usage.assertCounts();
        var context: runtime.CloneContext = .init(arena);
        defer context.deinit();

        var object: ObjectMap = .empty;
        errdefer runtime.freeValue(arena, .{ .object = object });
        inline for (names) |name|
            try putCount(&context, &object, name, @field(usage, name));
        return .{ .object = object };
    }

    /// DSPy's usage dictionary. Known prompt/completion/total counts come
    /// first; once any counter at all is known, both detail entries follow,
    /// each a sparse dictionary or null when it has no known member. An
    /// entirely unknown usage is an empty object. Ownership is that of
    /// toValue().
    pub fn dspyValue(usage: *const Usage, arena: Allocator) Allocator.Error!Value {
        usage.assertCounts();
        var context: runtime.CloneContext = .init(arena);
        defer context.deinit();

        var data: ObjectMap = .empty;
        errdefer runtime.freeValue(arena, .{ .object = data });
        try putCount(&context, &data, "prompt_tokens", usage.input_tokens);
        try putCount(&context, &data, "completion_tokens", usage.output_tokens);
        try putCount(&context, &data, "total_tokens", usage.total_tokens);
        if (usage.isKnown()) {
            try putDetails(&context, &data, "prompt_tokens_details", &.{
                .{ .key = "cached_tokens", .count = usage.cache_read_tokens },
                .{ .key = "cache_creation_tokens", .count = usage.cache_write_tokens },
                .{ .key = "audio_tokens", .count = usage.input_audio_tokens },
            });
            try putDetails(&context, &data, "completion_tokens_details", &.{
                .{ .key = "reasoning_tokens", .count = usage.reasoning_tokens },
                .{ .key = "audio_tokens", .count = usage.output_audio_tokens },
            });
        } else {
            assert(data.count() == 0);
        }
        return .{ .object = data };
    }

    /// Independent copy borrowing nothing from this usage, its input or its
    /// arena. A box repeated across counters is cloned once and stays shared.
    /// Release with free() using the same allocator. Ordinary small counts
    /// allocate nothing.
    pub fn clone(usage: *const Usage, allocator: Allocator) Allocator.Error!Usage {
        usage.assertCounts();
        var context: runtime.CloneContext = .init(allocator);
        defer context.deinit();

        // Counters not yet cloned are null, so a partial copy frees exactly
        // what it owns.
        var result: Usage = .{};
        errdefer result.free(allocator);
        inline for (names) |name|
            @field(result, name) = try context.cloneValue(@field(usage, name));
        return result;
    }

    /// Releases a clone() result only; init()/fromValue() results borrow.
    pub fn free(usage: *const Usage, allocator: Allocator) void {
        usage.assertCounts();
        inline for (names) |name| runtime.freeValue(allocator, @field(usage, name));
    }

    fn isKnown(usage: *const Usage) bool {
        inline for (names) |name| {
            if (@field(usage, name) != .null) return true;
        }
        return false;
    }

    /// Projections and copies take validated counters, not raw input.
    fn assertCounts(usage: *const Usage) void {
        inline for (names) |name| {
            switch (@field(usage, name)) {
                .null, .integer, .number_string => {},
                .custom => |custom| assert(custom.integerValue() != null),
                .bool, .float, .string, .array, .object => unreachable,
            }
        }
    }

    /// A part never exceeds its known whole. The two cache counters are also
    /// bounded together; an unknown one contributes nothing to that sum.
    fn checkParts(usage: *const Usage, arena: Allocator) Error!void {
        const input = usage.input_tokens;
        if (input != .null) {
            const read = usage.cache_read_tokens;
            const write = usage.cache_write_tokens;
            const cached: Value = if (read == .null)
                write
            else if (write == .null)
                read
            else
                try sum(arena, read, write);
            if (exceeds(cached, input)) return error.InvalidUsage;
            if (exceeds(usage.input_audio_tokens, input)) return error.InvalidUsage;
        }
        const output = usage.output_tokens;
        if (output != .null) {
            if (exceeds(usage.reasoning_tokens, output)) return error.InvalidUsage;
            if (exceeds(usage.output_audio_tokens, output)) return error.InvalidUsage;
        }
    }
};

comptime {
    assert(Usage.names.len == 8);
    for (@typeInfo(Usage).@"struct".field_types) |Field| assert(Field == Value);
}

/// One raw counter to a validated one. Only wide float conversions allocate.
fn count(arena: Allocator, raw: Value) Error!Value {
    switch (raw) {
        .null => return .null,
        .integer => |number| return if (number >= 0) raw else error.InvalidUsage,
        .number_string => |text| return integerCount(text),
        .float => |number| return floatCount(arena, number),
        .custom => |custom| {
            if (custom.integerValue()) |base| {
                // An int subclass keeps its identity; only its base is checked.
                _ = switch (base) {
                    .integer => |number| if (number >= 0) base else return error.InvalidUsage,
                    .number_string => |text| try integerCount(text),
                    else => unreachable,
                };
                return raw;
            }
            if (python.view(raw)) |builtin| {
                if (builtin == .float) return floatCount(arena, builtin.float);
            }
            return error.InvalidUsage;
        },
        .bool, .string, .array, .object => return error.InvalidUsage,
    }
}

/// Exact integer text to its canonical nonnegative form, borrowing `text`.
/// Float-shaped number text is an overflowed (infinite) float and is refused.
fn integerCount(text: []const u8) error{InvalidUsage}!Value {
    const negative = text.len > 0 and text[0] == '-';
    const digits = text[@intFromBool(negative)..];
    if (digits.len == 0) return error.InvalidUsage;
    for (digits) |byte| {
        if (!std.ascii.isDigit(byte)) return error.InvalidUsage;
    }
    const significant = std.mem.trimStart(u8, digits, "0");
    if (significant.len == 0) return .{ .integer = 0 };
    if (negative) return error.InvalidUsage;
    assert(significant[0] != '0');
    if (significant.len <= 19) {
        if (std.fmt.parseInt(i64, significant, 10)) |small| {
            return .{ .integer = small };
        } else |_| {}
    }
    return .{ .number_string = significant };
}

/// Python int(float) for a finite integral builtin float, exactly.
fn floatCount(arena: Allocator, number: f64) Error!Value {
    if (!std.math.isFinite(number)) return error.InvalidUsage;
    if (number != @trunc(number)) return error.InvalidUsage;
    // Negative zero is the integer zero, not a negative count.
    if (number < 0) return error.InvalidUsage;
    const i64_limit: f64 = 9223372036854775808.0;
    if (number < i64_limit) {
        const small: i64 = @intFromFloat(number);
        return .{ .integer = small };
    }
    var buffer: [310]u8 = undefined;
    const text = value.integralFloatText(number, &buffer).?;
    assert(text.len >= 19);
    assert(text[0] != '-');
    return .{ .number_string = try arena.dupe(u8, text) };
}

/// Exact sum of two validated counters; small sums allocate nothing.
fn sum(arena: Allocator, a: Value, b: Value) Allocator.Error!Value {
    assert(a != .null);
    assert(b != .null);
    return value.numericSum(arena, a, b) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        // Both operands are validated exact integers.
        error.InvalidNumber => unreachable,
    };
}

fn order(a: Value, b: Value) std.math.Order {
    assert(a != .null);
    assert(b != .null);
    // Validated exact integers are always ordered.
    return (value.numericOrder(a, b) catch unreachable).?;
}

/// An unknown part cannot break its relation.
fn exceeds(part: Value, whole: Value) bool {
    assert(whole != .null);
    if (part == .null) return false;
    return order(part, whole) == .gt;
}

fn putCount(
    context: *runtime.CloneContext,
    object: *ObjectMap,
    key: []const u8,
    counter: Value,
) Allocator.Error!void {
    if (counter == .null) return;
    const allocator = context.allocator;
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    const child = try context.cloneValue(counter);
    errdefer runtime.freeValue(allocator, child);
    try object.putNoClobber(allocator, owned_key, child);
}

fn putDetails(
    context: *runtime.CloneContext,
    data: *ObjectMap,
    key: []const u8,
    counters: []const Usage.Counter,
) Allocator.Error!void {
    const allocator = context.allocator;
    var details: ObjectMap = .empty;
    errdefer runtime.freeValue(allocator, .{ .object = details });
    for (counters) |counter| try putCount(context, &details, counter.key, counter.count);

    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    // An empty map holds no storage, so replacing it with null leaks nothing.
    const child: Value = if (details.count() == 0) .null else .{ .object = details };
    try data.putNoClobber(allocator, owned_key, child);
}

const testing = std.testing;

fn expectJson(arena: Allocator, expected: []const u8, actual: Value) !void {
    try testing.expectEqualStrings(expected, try std.json.Stringify.valueAlloc(arena, actual, .{}));
}

fn expectRefused(arena: Allocator, raw: Usage) !void {
    try testing.expectError(error.InvalidUsage, Usage.init(arena, raw));
}

test "counts keep exact integers, convert integral floats and refuse other kinds" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const usage = try Usage.init(arena, .{
        .input_tokens = .{ .number_string = "18446744073709551616" },
        .output_tokens = .{ .float = 1e30 },
        .cache_read_tokens = .{ .float = -0.0 },
        .cache_write_tokens = .{ .number_string = "0007" },
        .reasoning_tokens = .{ .float = 9223372036854774784.0 },
        .input_audio_tokens = .{ .integer = 0 },
        .output_audio_tokens = .{ .float = 9223372036854775808.0 },
    });
    try testing.expectEqualStrings("18446744073709551616", usage.input_tokens.number_string);
    try testing.expectEqualStrings("1000000000000000019884624838656", usage.output_tokens.number_string);
    try testing.expectEqualStrings("1000000000018446763958334390272", usage.total_tokens.number_string);
    try testing.expectEqual(@as(i64, 0), usage.cache_read_tokens.integer);
    try testing.expectEqual(@as(i64, 7), usage.cache_write_tokens.integer);
    try testing.expectEqual(@as(i64, 9223372036854774784), usage.reasoning_tokens.integer);
    try testing.expectEqual(@as(i64, 0), usage.input_audio_tokens.integer);
    try testing.expectEqualStrings("9223372036854775808", usage.output_audio_tokens.number_string);

    const widest = try Usage.init(arena, .{ .total_tokens = .{ .float = 1e308 } });
    try testing.expectEqual(@as(usize, 309), widest.total_tokens.number_string.len);
    try testing.expect(try value.pyEqual(widest.total_tokens, .{ .float = 1e308 }));

    const boxed = try python.float(arena, 3.0);
    const from_box = try Usage.init(arena, .{ .input_tokens = boxed });
    try testing.expectEqual(@as(i64, 3), from_box.input_tokens.integer);

    const refused = [_]Value{
        .{ .bool = true },
        .{ .bool = false },
        .{ .string = "5" },
        .{ .integer = -1 },
        .{ .float = -1.0 },
        .{ .float = 1.5 },
        .{ .float = std.math.nan(f64) },
        .{ .float = std.math.inf(f64) },
        .{ .number_string = "-18446744073709551616" },
        .{ .number_string = "1e400" },
        .{ .number_string = "" },
        .{ .array = .init(arena) },
        try python.float(arena, 2.5),
    };
    for (refused) |counter| {
        inline for (Usage.names) |name| {
            var raw: Usage = .{};
            @field(raw, name) = counter;
            try expectRefused(arena, raw);
        }
    }
}

test "integer subclasses keep their identity while ordinary enum members are refused" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const enums = @import("enum.zig");
    const entries = &[_]enums.Entry{
        .{ .name = "five", .value = .{ .integer = 5 } },
        .{ .name = "wide", .value = .{ .number_string = "18446744073709551616" } },
        .{ .name = "negative", .value = .{ .integer = -5 } },
    };
    const Count = enums.IntEnum("UsageCount", entries);
    const Plain = enums.Enum("UsagePlain", entries);
    const five = try Count.member("five");
    const wide = try Count.member("wide");

    const usage = try Usage.init(arena, .{ .input_tokens = wide, .output_tokens = five, .reasoning_tokens = five });
    try testing.expect(usage.input_tokens.custom.ptr == wide.custom.ptr);
    try testing.expect(usage.reasoning_tokens.custom.ptr == five.custom.ptr);
    // Python adds two IntEnum members into a plain int.
    try testing.expectEqualStrings("18446744073709551621", usage.total_tokens.number_string);

    try expectRefused(arena, .{ .input_tokens = try Count.member("negative") });
    try expectRefused(arena, .{ .input_tokens = try Plain.member("five") });
    try expectRefused(arena, .{ .output_tokens = five, .reasoning_tokens = wide });
    try expectRefused(arena, .{ .input_tokens = five, .output_tokens = five, .total_tokens = five });
}

test "parts are bounded by known wholes and unknown counters never count as zero" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const n = struct {
        fn n(number: i64) Value {
            return .{ .integer = number };
        }
    }.n;

    try expectRefused(arena, .{ .input_tokens = n(5), .cache_read_tokens = n(3), .cache_write_tokens = n(3) });
    try expectRefused(arena, .{ .input_tokens = n(5), .cache_read_tokens = n(6) });
    try expectRefused(arena, .{ .input_tokens = n(5), .cache_write_tokens = n(6) });
    try expectRefused(arena, .{ .input_tokens = n(5), .input_audio_tokens = n(6) });
    try expectRefused(arena, .{ .output_tokens = n(5), .reasoning_tokens = n(6) });
    try expectRefused(arena, .{ .output_tokens = n(5), .output_audio_tokens = n(6) });
    try expectRefused(arena, .{ .input_tokens = n(1), .output_tokens = n(2), .total_tokens = n(4) });
    try expectRefused(arena, .{ .input_tokens = n(0), .output_tokens = n(0), .total_tokens = n(1) });
    try expectRefused(arena, .{
        .input_tokens = .{ .number_string = "18446744073709551616" },
        .cache_read_tokens = .{ .number_string = "18446744073709551616" },
        .cache_write_tokens = n(1),
    });

    // Reasoning and output audio are bounded separately, not together.
    const boundary = try Usage.init(arena, .{
        .input_tokens = n(5),
        .output_tokens = n(4),
        .total_tokens = .{ .float = 9.0 },
        .cache_read_tokens = n(2),
        .cache_write_tokens = n(3),
        .reasoning_tokens = n(4),
        .input_audio_tokens = n(5),
        .output_audio_tokens = n(4),
    });
    try testing.expectEqual(@as(i64, 9), boundary.total_tokens.integer);

    // Without their whole, parts and an explicit total are unconstrained.
    const parts = try Usage.init(arena, .{
        .total_tokens = n(1),
        .cache_read_tokens = n(100),
        .cache_write_tokens = n(100),
        .reasoning_tokens = n(100),
        .input_audio_tokens = n(100),
        .output_audio_tokens = n(100),
    });
    try testing.expect(parts.input_tokens == .null);
    try testing.expect(parts.output_tokens == .null);

    const input_only = try Usage.init(arena, .{ .input_tokens = n(7) });
    try testing.expect(input_only.total_tokens == .null);
    const output_only = try Usage.init(arena, .{ .output_tokens = n(7), .total_tokens = n(3) });
    try testing.expectEqual(@as(i64, 3), output_only.total_tokens.integer);
    const zeros = try Usage.init(arena, .{ .input_tokens = n(0), .output_tokens = n(0) });
    try testing.expectEqual(@as(i64, 0), zeros.total_tokens.integer);
    const carried = try Usage.init(arena, .{ .input_tokens = n(std.math.maxInt(i64)), .output_tokens = n(1) });
    try testing.expectEqualStrings("9223372036854775808", carried.total_tokens.number_string);
}

test "SDK and DSPy projections stay sparse and separate zero from unknown" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const unknown = try Usage.fromValue(arena, .{ .object = .empty });
    try expectJson(arena, "{}", try unknown.toValue(arena));
    try expectJson(arena, "{}", try unknown.dspyValue(arena));

    const zero_detail = try Usage.init(arena, .{ .cache_read_tokens = .{ .integer = 0 } });
    try expectJson(arena, "{\"cache_read_tokens\":0}", try zero_detail.toValue(arena));
    try expectJson(
        arena,
        "{\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":null}",
        try zero_detail.dspyValue(arena),
    );

    const source = try std.json.parseFromSliceLeaky(Value, arena,
        \\{"output_audio_tokens":2,"provider_extra":"ignored","input_tokens":18446744073709551616,
        \\ "output_tokens":7.0,"cache_write_tokens":null,"reasoning_tokens":0,"total_tokens":18446744073709551623}
    , .{});
    const usage = try Usage.fromValue(arena, source);
    try expectJson(
        arena,
        "{\"input_tokens\":18446744073709551616,\"output_tokens\":7," ++
            "\"total_tokens\":18446744073709551623,\"reasoning_tokens\":0,\"output_audio_tokens\":2}",
        try usage.toValue(arena),
    );
    try expectJson(
        arena,
        "{\"prompt_tokens\":18446744073709551616,\"completion_tokens\":7," ++
            "\"total_tokens\":18446744073709551623,\"prompt_tokens_details\":null," ++
            "\"completion_tokens_details\":{\"reasoning_tokens\":0,\"audio_tokens\":2}}",
        try usage.dspyValue(arena),
    );

    const counts_only = try Usage.init(arena, .{ .input_tokens = .{ .integer = 0 } });
    try expectJson(
        arena,
        "{\"prompt_tokens\":0,\"prompt_tokens_details\":null,\"completion_tokens_details\":null}",
        try counts_only.dspyValue(arena),
    );

    try testing.expectError(error.InvalidUsage, Usage.fromValue(arena, .null));
    try testing.expectError(error.InvalidUsage, Usage.fromValue(arena, .{ .string = "{}" }));
}

fn exerciseOwnership(gpa: Allocator) !void {
    const copy: Usage = copy: {
        var source: std.heap.ArenaAllocator = .init(testing.allocator);
        defer source.deinit();
        const arena = source.allocator();
        const text = try arena.dupe(u8, "18446744073709551616");
        const usage = try Usage.init(arena, .{
            .input_tokens = .{ .number_string = text },
            .output_tokens = .{ .float = 1e30 },
            .cache_read_tokens = .{ .number_string = text },
            .reasoning_tokens = .{ .integer = 0 },
        });
        const owned = try usage.clone(gpa);
        // The clone must not observe its source being overwritten or released.
        @memset(text, '9');
        break :copy owned;
    };
    defer copy.free(gpa);
    try testing.expectEqualStrings("18446744073709551616", copy.input_tokens.number_string);
    try testing.expectEqualStrings("18446744073709551616", copy.cache_read_tokens.number_string);
    try testing.expectEqualStrings("1000000000018446763958334390272", copy.total_tokens.number_string);

    const sdk = try copy.toValue(gpa);
    defer runtime.freeValue(gpa, sdk);
    try testing.expectEqual(@as(usize, 5), sdk.object.count());
    try testing.expect(sdk.object.get("input_tokens").?.number_string.ptr != copy.input_tokens.number_string.ptr);

    const dspy = try copy.dspyValue(gpa);
    defer runtime.freeValue(gpa, dspy);
    try testing.expectEqual(@as(usize, 5), dspy.object.count());
    try testing.expectEqual(@as(i64, 0), dspy.object.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer);

    const again = try copy.clone(gpa);
    defer again.free(gpa);
    try testing.expect(again.total_tokens.number_string.ptr != copy.total_tokens.number_string.ptr);
}

test "clones and projections outlive their source and release every partial allocation" {
    var stable = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable.allocator(), exerciseOwnership, .{});
}

test "small counts validate and clone without allocating" {
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    const none = failing.allocator();
    const usage = try Usage.init(none, .{
        .input_tokens = .{ .integer = 13 },
        .output_tokens = .{ .float = 7.0 },
        .cache_read_tokens = .{ .integer = 4 },
        .cache_write_tokens = .{ .integer = 9 },
        .reasoning_tokens = .{ .integer = 7 },
    });
    try testing.expectEqual(@as(i64, 20), usage.total_tokens.integer);
    const copy = try usage.clone(none);
    defer copy.free(none);
    try testing.expectEqual(@as(i64, 20), copy.total_tokens.integer);
}
