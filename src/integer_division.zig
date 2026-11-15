//! Correctly rounded Python int/int true division without converting either
//! operand to floating point. Operands remain borrowed; the only allocation is
//! one reusable big-integer division workspace.
const std = @import("std");
const Allocator = std.mem.Allocator;
const big = std.math.big.int;
const Const = big.Const;
const Mutable = big.Mutable;
const Limb = std.math.big.Limb;
const Log2Limb = std.math.big.Log2Limb;
const limb_bits = @bitSizeOf(Limb);
const sign_mask: u64 = 0x8000000000000000;
const implicit_bit: u64 = 1 << 52;

pub const Error = Allocator.Error || error{InvalidNumber};

/// Returns numerator / denominator, rounded once to binary64, ties to even.
/// Division by zero and a result that rounds to infinity return InvalidNumber.
/// Inputs may share storage and are never mutated. Zero integers have no sign;
/// an underflowed nonzero quotient retains its sign.
pub fn divide(allocator: Allocator, numerator: Const, denominator: Const) Error!f64 {
    const n = magnitude(numerator);
    const d = magnitude(denominator);
    if (d.limbs.len == 1 and d.limbs[0] == 0) return error.InvalidNumber;
    const numerator_is_zero = n.limbs.len == 1 and n.limbs[0] == 0;
    const numerator_positive = numerator_is_zero or numerator.positive;
    const sign: u64 = if (numerator_positive != denominator.positive) sign_mask else 0;
    if (numerator_is_zero) return @bitCast(sign);

    const n_bits = n.bitCountAbs();
    const d_bits = d.bitCountAbs();
    var exponent: i32 = undefined;
    if (n_bits >= d_bits) {
        const difference = n_bits - d_bits;
        // floor(log2(n/d)) is difference or difference-1. Classify before
        // converting the difference to a signed type or materializing a shift.
        if (difference > 1024) return error.InvalidNumber;
        exponent = @intCast(difference);
        if (orderShifted(n, 0, d, difference) == .lt) exponent -= 1;
    } else {
        const difference = d_bits - n_bits;
        if (difference > 1075) return @bitCast(sign);
        exponent = -@as(i32, @intCast(difference));
        if (orderShifted(n, difference, d, 0) == .lt) exponent -= 1;
    }
    if (exponent > 1023) return error.InvalidNumber;
    if (exponent < -1075) return @bitCast(sign);

    // Normal results need 53 significant bits. Subnormals instead use the
    // fixed 2^-1074 quantum, including the tie at half the minimum subnormal.
    const scale: i32 = if (exponent >= -1022) 52 - exponent else 1074;
    const n_shift: usize = if (scale > 0) @intCast(scale) else 0;
    const d_shift: usize = if (scale < 0) @intCast(-scale) else 0;
    const n_len = shiftedLen(n, n_shift);
    const d_len = shiftedLen(d, d_shift);
    const work_len = @max(n_len, d_len);
    const operand_len = std.math.add(usize, n_len, d_len) catch return error.OutOfMemory;
    // The pinned stdlib's calcDivLimbsBufferLen is a_len + b_len + 4.
    const division_len = std.math.add(usize, operand_len, 4) catch return error.OutOfMemory;
    const remainder_end = std.math.add(usize, work_len, d_len) catch return error.OutOfMemory;
    const allocation_len = std.math.add(usize, remainder_end, division_len) catch return error.OutOfMemory;
    const storage = try allocator.alloc(Limb, allocation_len);
    defer allocator.free(storage);

    var quotient: Mutable = .{ .limbs = storage[0..work_len], .len = 1, .positive = true };
    var remainder: Mutable = .{ .limbs = storage[work_len..remainder_end], .len = 1, .positive = true };
    var scaled_n = n;
    var scaled_d = d;
    if (n_shift != 0) {
        quotient.shiftLeft(n, n_shift);
        scaled_n = quotient.toConst();
    } else if (d_shift != 0) {
        quotient.shiftLeft(d, d_shift);
        scaled_d = quotient.toConst();
    }
    // divTrunc copies both inputs into its scratch before writing quotient,
    // and explicitly permits quotient to alias either operand. Reusing the
    // scaled operand's storage avoids a separate shifted-number allocation.
    quotient.divTrunc(&remainder, scaled_n, scaled_d, storage[remainder_end..]);
    var significand = quotient.toInt(u64) catch unreachable;
    std.debug.assert(significand < (implicit_bit << 1));
    // Compare virtually shifted limbs: neither 2*remainder nor the original
    // scaled denominator needs another allocation (scaled_d may now be gone).
    const halfway = orderShifted(remainder.toConst(), 1, d, d_shift);
    if (halfway == .gt or (halfway == .eq and significand & 1 != 0)) significand += 1;

    if (exponent < -1022) {
        std.debug.assert(significand <= implicit_bit);
        // The carry to 2^52 is already the smallest normal's bit pattern.
        return @bitCast(sign | significand);
    }
    std.debug.assert(significand >= implicit_bit);
    if (significand == implicit_bit << 1) {
        significand >>= 1;
        exponent += 1;
        if (exponent > 1023) return error.InvalidNumber;
    }
    const biased_exponent: u64 = @intCast(exponent + 1023);
    return @bitCast(sign | (biased_exponent << 52) | (significand - implicit_bit));
}

fn magnitude(value: Const) Const {
    std.debug.assert(value.limbs.len != 0);
    var len = value.limbs.len;
    while (len > 1 and value.limbs[len - 1] == 0) : (len -= 1) {}
    return .{ .limbs = value.limbs[0..len], .positive = true };
}

fn shiftedLen(value: Const, shift: usize) usize {
    const partial = shift % limb_bits;
    const extra = partial != 0 and value.limbs[value.limbs.len - 1] >>
        @as(Log2Limb, @intCast(limb_bits - partial)) != 0;
    return value.limbs.len + shift / limb_bits + @intFromBool(extra);
}

/// Compares nonnegative values after virtual left shifts, including zero.
fn orderShifted(a: Const, a_shift: usize, b: Const, b_shift: usize) std.math.Order {
    const a_bits = a.bitCountAbs();
    const b_bits = b.bitCountAbs();
    if (a_bits == 0) return if (b_bits == 0) .eq else .lt;
    if (b_bits == 0) return .gt;
    const shifted_a_bits = a_bits + a_shift;
    const shifted_b_bits = b_bits + b_shift;
    if (shifted_a_bits < shifted_b_bits) return .lt;
    if (shifted_a_bits > shifted_b_bits) return .gt;
    var index = (shifted_a_bits - 1) / limb_bits + 1;
    while (index != 0) {
        index -= 1;
        const a_limb = shiftedLimb(a.limbs, a_shift, index);
        const b_limb = shiftedLimb(b.limbs, b_shift, index);
        if (a_limb < b_limb) return .lt;
        if (a_limb > b_limb) return .gt;
    }
    return .eq;
}

fn shiftedLimb(limbs: []const Limb, shift: usize, index: usize) Limb {
    const whole = shift / limb_bits;
    if (index < whole) return 0;
    const source = index - whole;
    const partial: Log2Limb = @intCast(shift % limb_bits);
    var result: Limb = if (source < limbs.len) limbs[source] << partial else 0;
    if (partial != 0 and source != 0 and source - 1 < limbs.len) {
        result |= limbs[source - 1] >> @as(Log2Limb, @intCast(limb_bits - @as(usize, partial)));
    }
    return result;
}

// Expected bits below were observed with the independent CPython 3.11.17 probe
// /tmp/dszig-integer-division-source.py, not any frozen DSPy oracle. The receipt
// also contains all sign variants, three scale sizes, and 256 random huge pairs.
fn expectRatio(numerator: Const, denominator: Const, bits: u64) !void {
    const result = try divide(std.testing.allocator, numerator, denominator);
    try std.testing.expectEqual(bits, @as(u64, @bitCast(result)));
}

fn expectSignedRatios(numerator: Const, denominator: Const, bits: u64) !void {
    try expectRatio(numerator, denominator, bits);
    try expectRatio(numerator.negate(), denominator, bits | sign_mask);
    try expectRatio(numerator, denominator.negate(), bits | sign_mask);
    try expectRatio(numerator.negate(), denominator.negate(), bits);
}

fn testInteger(coefficient: u64, shift: usize, adjustment: i64) !big.Managed {
    var result = try big.Managed.initSet(std.testing.allocator, coefficient);
    errdefer result.deinit();
    try result.shiftLeft(&result, shift);
    if (adjustment != 0) try result.addScalar(&result, adjustment);
    return result;
}

test "integer true division preserves huge equal and unequal ratios and borrowed operands" {
    const allocator = std.testing.allocator;
    var huge = try big.Managed.init(allocator);
    defer huge.deinit();
    const tail = "123456789012345678901234567890123456789";
    var digits: [401]u8 = @splat('0');
    digits[0] = '1';
    @memcpy(digits[digits.len - tail.len ..], tail);
    try huge.setString(10, &digits);
    try expectSignedRatios(huge.toConst(), huge.toConst(), 0x3ff0000000000000);
    var near = try huge.clone();
    defer near.deinit();
    try near.addScalar(&near, 1);
    try expectSignedRatios(near.toConst(), huge.toConst(), 0x3ff0000000000000);

    var twice = try huge.clone();
    defer twice.deinit();
    try twice.shiftLeft(&twice, 1);
    var thrice = try huge.clone();
    defer thrice.deinit();
    try thrice.add(&twice, &huge);
    const numerator_before = try allocator.dupe(Limb, twice.toConst().limbs);
    defer allocator.free(numerator_before);
    const denominator_before = try allocator.dupe(Limb, thrice.toConst().limbs);
    defer allocator.free(denominator_before);
    try expectSignedRatios(twice.toConst(), thrice.toConst(), 0x3fe5555555555555);
    try std.testing.expectEqualSlices(Limb, numerator_before, twice.toConst().limbs);
    try std.testing.expectEqualSlices(Limb, denominator_before, thrice.toConst().limbs);
    try std.testing.expect(twice.toConst().positive and thrice.toConst().positive);
}

test "integer true division classifies certain extremes and zero without allocating" {
    var enormous = try testInteger(1, 100000, 0);
    defer enormous.deinit();
    const three: Const = .{ .limbs = &.{3}, .positive = true };
    const zero: Const = .{ .limbs = &.{0}, .positive = true };
    var empty_storage: [0]u8 = .{};
    var empty = std.heap.FixedBufferAllocator.init(&empty_storage);
    const allocator = empty.allocator();
    try std.testing.expectError(error.InvalidNumber, divide(allocator, enormous.toConst(), three));
    try std.testing.expectError(error.InvalidNumber, divide(allocator, enormous.toConst().negate(), three));
    try std.testing.expectError(error.InvalidNumber, divide(allocator, enormous.toConst(), zero));
    try std.testing.expectError(error.InvalidNumber, divide(allocator, zero, zero));
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(try divide(allocator, three, enormous.toConst()))));
    try std.testing.expectEqual(sign_mask, @as(u64, @bitCast(try divide(allocator, three.negate(), enormous.toConst()))));
    try std.testing.expectEqual(sign_mask, @as(u64, @bitCast(try divide(allocator, zero, enormous.toConst().negate()))));
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(try divide(allocator, zero.negate(), three))));
}

test "integer true division rounds half minimum subnormal with signed zero" {
    for ([_]usize{ 1400, 10000 }) |scale| {
        var denominator = try testInteger(1, 1075 + scale, 0);
        defer denominator.deinit();
        const expected = [_]u64{ 0, 0, 1 };
        for ([_]i64{ -1, 0, 1 }, expected) |adjustment, bits| {
            var numerator = try testInteger(1, scale, adjustment);
            defer numerator.deinit();
            try expectSignedRatios(numerator.toConst(), denominator.toConst(), bits);
        }
    }
}

test "integer true division rounds subnormal midpoints and the normal transition" {
    const cases = [_]struct { lower: u64, expected: [3]u64 }{
        .{ .lower = 1, .expected = .{ 1, 2, 2 } },
        .{ .lower = 2, .expected = .{ 2, 2, 3 } },
        .{ .lower = 0x000fffffffffffff, .expected = .{ 0x000fffffffffffff, 0x0010000000000000, 0x0010000000000000 } },
    };
    for ([_]usize{ 0, 1400, 10000 }) |scale| {
        var denominator = try testInteger(1, 1076 + scale, 0);
        defer denominator.deinit();
        for (cases) |case| {
            for ([_]i64{ -1, 0, 1 }, case.expected) |adjustment, bits| {
                var numerator = try testInteger(2 * case.lower + 1, scale + 1, adjustment);
                defer numerator.deinit();
                try expectSignedRatios(numerator.toConst(), denominator.toConst(), bits);
            }
        }
    }
}

test "integer true division rounds normal midpoints once with significand carry" {
    const cases = [_]struct { lower: u64, expected: [3]u64 }{
        .{ .lower = 0x0010000000000000, .expected = .{ 0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000001 } },
        .{ .lower = 0x0010000000000001, .expected = .{ 0x3ff0000000000001, 0x3ff0000000000002, 0x3ff0000000000002 } },
        .{ .lower = 0x001fffffffffffff, .expected = .{ 0x3fffffffffffffff, 0x4000000000000000, 0x4000000000000000 } },
    };
    for ([_]usize{ 0, 1400, 10000 }) |scale| {
        var denominator = try testInteger(1, 53 + scale, 0);
        defer denominator.deinit();
        for (cases) |case| {
            for ([_]i64{ -1, 0, 1 }, case.expected) |adjustment, bits| {
                var numerator = try testInteger(2 * case.lower + 1, scale, adjustment);
                defer numerator.deinit();
                try expectSignedRatios(numerator.toConst(), denominator.toConst(), bits);
            }
        }
    }
}

test "integer true division distinguishes maximum finite from the overflow midpoint" {
    for ([_]usize{ 0, 1400, 10000 }) |scale| {
        var denominator = try testInteger(1, scale, 0);
        defer denominator.deinit();
        var maximum = try testInteger(0x001fffffffffffff, 971 + scale, 0);
        defer maximum.deinit();
        try expectSignedRatios(maximum.toConst(), denominator.toConst(), 0x7fefffffffffffff);
        for ([_]i64{ -1, 0, 1 }) |adjustment| {
            var numerator = try testInteger(0x003fffffffffffff, 970 + scale, adjustment);
            defer numerator.deinit();
            if (adjustment < 0) {
                try expectSignedRatios(numerator.toConst(), denominator.toConst(), 0x7fefffffffffffff);
            } else {
                try std.testing.expectError(error.InvalidNumber, divide(std.testing.allocator, numerator.toConst(), denominator.toConst()));
                try std.testing.expectError(error.InvalidNumber, divide(std.testing.allocator, numerator.toConst().negate(), denominator.toConst()));
                try std.testing.expectError(error.InvalidNumber, divide(std.testing.allocator, numerator.toConst(), denominator.toConst().negate()));
                try std.testing.expectError(error.InvalidNumber, divide(std.testing.allocator, numerator.toConst().negate(), denominator.toConst().negate()));
            }
        }
    }
}

test "integer true division accepts borrowed leading zero limbs without mutation" {
    const numerator_limbs = [_]Limb{ 3, 0, 0 };
    const denominator_limbs = [_]Limb{ 2, 0 };
    const numerator: Const = .{ .limbs = &numerator_limbs, .positive = true };
    const denominator: Const = .{ .limbs = &denominator_limbs, .positive = true };
    try expectSignedRatios(numerator, denominator, 0x3ff8000000000000);
    try std.testing.expectEqualSlices(Limb, &.{ 3, 0, 0 }, &numerator_limbs);
    try std.testing.expectEqualSlices(Limb, &.{ 2, 0 }, &denominator_limbs);
}

test "integer true division propagates allocation failure without leaking scratch" {
    var numerator = try testInteger(0x0020000000000001, 1400, 1);
    defer numerator.deinit();
    var denominator = try testInteger(1, 1453, 0);
    defer denominator.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(allocator: Allocator, n: Const, d: Const) !void {
            const result = try divide(allocator, n, d);
            try std.testing.expectEqual(@as(u64, 0x3ff0000000000001), @as(u64, @bitCast(result)));
        }
    }.run, .{ numerator.toConst(), denominator.toConst() });
}
