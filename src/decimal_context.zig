//! Contextual arithmetic over the exact coefficient/exponent in decimal.zig.
//! Inputs are borrowed and never changed. Every successful arithmetic result
//! owns its coefficient, including zero, Infinity and NaN; call result.deinit(a).
//! No current-context singleton is stored here. A caller owns its Context.
//! Operands here are already Decimal metadata; mixed float arithmetic must be
//! rejected by the runtime boundary. For float comparisons, that boundary sets
//! the FloatOperation flag WITHOUT honoring its trap for equality/inequality,
//! but calls signal(.float_operation) BEFORE order() for ordered comparisons.
//! This distinction preserves Source NaN/trap sequencing as well as exactness.
const std = @import("std");
const decimal = @import("decimal.zig");
const Allocator = std.mem.Allocator;
const Decimal = decimal.Decimal;

pub const Rounding = enum {
    ceiling,
    down,
    floor,
    half_down,
    half_even,
    half_up,
    up,
    zero_five_up,
};

pub const Signal = enum {
    invalid_operation,
    division_by_zero,
    overflow,
    underflow,
    subnormal,
    inexact,
    rounded,
    clamped,
    float_operation,
};

pub const Signals = std.EnumSet(Signal);

// CPython's signal precedence, independent of the set's representation. Flags
// for ALL conditions are committed before any enabled trap is returned.
const signal_priority = [_]Signal{
    .invalid_operation,
    .float_operation,
    .division_by_zero,
    .overflow,
    .underflow,
    .subnormal,
    .inexact,
    .rounded,
    .clamped,
};

pub const Context = struct {
    precision: usize = 28,
    rounding: Rounding = .half_even,
    emin: i64 = -999999,
    emax: i64 = 999999,
    capitals: bool = true,
    clamp: bool = false,
    flags: Signals = .empty,
    traps: Signals = .initMany(&.{ .invalid_operation, .division_by_zero, .overflow }),

    /// Trapped conditions retain their identity here; the runtime boundary may
    /// map them to its own error type. InvalidContext is an invalid configuration
    /// (precision outside 1..MAX_PREC, Emin outside -MAX_EMAX..0, or Emax outside
    /// 0..MAX_EMAX), NOT a Decimal signal. Allocation errors do not clear flags.
    pub const Error = Allocator.Error || error{
        InvalidContext,
        InvalidOperation,
        DivisionByZero,
        Overflow,
        Underflow,
        Subnormal,
        Inexact,
        Rounded,
        Clamped,
        FloatOperation,
    };

    pub fn clearFlags(self: *Context) void {
        self.flags = .empty;
    }

    /// Set a sticky flag, then honor its trap. Already-sticky flags do not cause
    /// a later operation to trap unless that operation raises them again.
    pub fn signal(self: *Context, condition: Signal) Error!void {
        try self.raise(.initOne(condition));
    }

    /// Quiet NaNs compare unequal without signaling. A signaling NaN raises
    /// InvalidOperation, or compares unequal when that trap is disabled.
    /// Comparison never allocates, rounds, or applies exponent limits.
    pub fn equal(self: *Context, left: Decimal, right: Decimal) Error!bool {
        if (left.kind == .signaling_nan or right.kind == .signaling_nan) {
            try self.signal(.invalid_operation);
            return false;
        }
        if (isNan(left) or isNan(right)) return false;
        return left.equal(right) catch unreachable;
    }

    /// null denotes unordered, NOT equality. Both kinds of NaN raise
    /// InvalidOperation for ordering, returning null when the trap is disabled.
    /// Comparison is exact and allocation-free even across enormous exponents.
    pub fn order(self: *Context, left: Decimal, right: Decimal) Error!?std.math.Order {
        if (isNan(left) or isNan(right)) {
            try self.signal(.invalid_operation);
            return null;
        }
        return left.order(right) catch unreachable;
    }

    pub fn add(self: *Context, allocator: Allocator, left: Decimal, right: Decimal) Error!Decimal {
        return self.addOrSubtract(allocator, left, right, false);
    }

    pub fn subtract(self: *Context, allocator: Allocator, left: Decimal, right: Decimal) Error!Decimal {
        return self.addOrSubtract(allocator, left, right, true);
    }

    pub fn multiply(self: *Context, allocator: Allocator, left: Decimal, right: Decimal) Error!Decimal {
        try self.validate();
        if (try self.propagateNan(allocator, left, right)) |nan| return nan;
        const negative = left.negative != right.negative;
        if (left.kind == .infinity or right.kind == .infinity) {
            if (left.isZero() or right.isZero()) return self.invalid(allocator);
            return owned(allocator, .infinity, negative, "0", 0);
        }
        const exponent = @as(i128, left.exponent) + right.exponent;
        if (left.isZero() or right.isZero())
            return self.finish(allocator, negative, "0", exponent, false);
        if (std.mem.eql(u8, left.digits, "1"))
            return self.finish(allocator, negative, right.digits, exponent, false);
        if (std.mem.eql(u8, right.digits, "1"))
            return self.finish(allocator, negative, left.digits, exponent, false);

        const size = try sizeSum(left.digits.len, right.digits.len);
        const product = try allocator.alloc(u8, size);
        defer allocator.free(product);
        @memset(product, 0);
        // Schoolbook base-ten multiplication. Each cell and carry is bounded
        // by 9, so no coefficient-size-dependent machine integer is needed.
        var i = left.digits.len;
        while (i != 0) {
            i -= 1;
            var carry: u16 = 0;
            var j = right.digits.len;
            while (j != 0) {
                j -= 1;
                const index = i + j + 1;
                const next = @as(u16, left.digits[i] - '0') * (right.digits[j] - '0') + product[index] + carry;
                product[index] = @intCast(next % 10);
                carry = next / 10;
            }
            product[i] = @intCast(carry);
        }
        for (product) |*digit| digit.* += '0';
        const start: usize = @intFromBool(product[0] == '0');
        return self.finish(allocator, negative, product[start..], exponent, false);
    }

    pub fn divide(self: *Context, allocator: Allocator, left: Decimal, right: Decimal) Error!Decimal {
        try self.validate();
        if (try self.propagateNan(allocator, left, right)) |nan| return nan;
        const negative = left.negative != right.negative;
        if (left.kind == .infinity) {
            if (right.kind == .infinity) return self.invalid(allocator);
            // Infinity / zero is Infinity, without DivisionByZero.
            return owned(allocator, .infinity, negative, "0", 0);
        }
        if (right.kind == .infinity) {
            try self.signal(.clamped);
            return owned(allocator, .finite, negative, "0", self.etiny());
        }
        if (right.isZero()) {
            if (left.isZero()) return self.invalid(allocator);
            try self.signal(.division_by_zero);
            return owned(allocator, .infinity, negative, "0", 0);
        }
        const ideal_exponent = @as(i128, left.exponent) - right.exponent;
        if (left.isZero()) return self.finish(allocator, negative, "0", ideal_exponent, false);

        // Long division works on coefficients only: exponent differences never
        // create powers of ten. One remainder of at most divisor.len+1 digits
        // and at most precision+2 significant quotient digits are retained.
        const storage = try allocator.alloc(u8, try sizeSum(right.digits.len, 1));
        defer allocator.free(storage);
        var remainder = Remainder.init(storage);
        var quotient: std.ArrayList(u8) = .empty;
        defer quotient.deinit(allocator);
        const limit = self.precision + 2;
        for (left.digits, 0..) |digit, index| {
            const next = remainder.step(digit, right.digits);
            if (next != '0' or quotient.items.len != 0) try quotient.append(allocator, next);
            if (quotient.items.len == limit) {
                const rest = left.digits[index + 1 ..];
                return self.finish(
                    allocator,
                    negative,
                    quotient.items,
                    ideal_exponent + rest.len,
                    !remainder.isZero() or hasNonzero(rest),
                );
            }
        }
        var fractional: i128 = 0;
        while (!remainder.isZero() and quotient.items.len < limit) {
            const next = remainder.step('0', right.digits);
            fractional += 1;
            if (next != '0' or quotient.items.len != 0) try quotient.append(allocator, next);
        }
        std.debug.assert(quotient.items.len != 0);
        // Stopping immediately on exact division preserves the preferred
        // exponent, without inventing extra trailing fractional zeroes.
        return self.finish(allocator, negative, quotient.items, ideal_exponent - fractional, !remainder.isZero());
    }

    /// Contextual abs(), not copy_abs(): finite coefficients are rounded and
    /// range-checked. Source NaN propagation precedes sign removal, so a NaN's
    /// sign is retained, even though finite/infinite results are nonnegative.
    pub fn absolute(self: *Context, allocator: Allocator, value: Decimal) Error!Decimal {
        try self.validate();
        if (try self.propagateNan(allocator, value, null)) |nan| return nan;
        if (value.kind == .infinity) return owned(allocator, .infinity, false, "0", 0);
        return self.finish(allocator, false, value.digits, value.exponent, false);
    }

    fn validate(self: *const Context) Error!void {
        if (self.precision == 0 or @as(i128, self.precision) > decimal.max_exponent or
            self.emin > 0 or self.emin < -decimal.max_exponent or
            self.emax < 0 or self.emax > decimal.max_exponent) return error.InvalidContext;
    }

    fn etiny(self: *const Context) i64 {
        return @intCast(@as(i128, self.emin) - self.precision + 1);
    }

    fn etop(self: *const Context) i64 {
        return @intCast(@as(i128, self.emax) - self.precision + 1);
    }

    fn raise(self: *Context, raised: Signals) Error!void {
        self.flags.setUnion(raised);
        for (signal_priority) |condition| {
            if (!raised.contains(condition) or !self.traps.contains(condition)) continue;
            return switch (condition) {
                .invalid_operation => error.InvalidOperation,
                .division_by_zero => error.DivisionByZero,
                .overflow => error.Overflow,
                .underflow => error.Underflow,
                .subnormal => error.Subnormal,
                .inexact => error.Inexact,
                .rounded => error.Rounded,
                .clamped => error.Clamped,
                .float_operation => error.FloatOperation,
            };
        }
    }

    fn invalid(self: *Context, allocator: Allocator) Error!Decimal {
        try self.signal(.invalid_operation);
        return owned(allocator, .quiet_nan, false, "", 0);
    }

    fn propagateNan(self: *Context, allocator: Allocator, left: Decimal, right: ?Decimal) Error!?Decimal {
        // A signaling NaN takes precedence over a quiet NaN, otherwise the
        // leftmost NaN wins. Subtraction must not negate a right-hand NaN.
        const selected: Decimal = if (left.kind == .signaling_nan)
            left
        else if (right != null and right.?.kind == .signaling_nan)
            right.?
        else if (left.kind == .quiet_nan)
            left
        else if (right != null and right.?.kind == .quiet_nan)
            right.?
        else
            return null;
        if (selected.kind == .signaling_nan) try self.signal(.invalid_operation);
        const max_payload = self.precision - @intFromBool(self.clamp);
        var payload = selected.digits[selected.digits.len - @min(selected.digits.len, max_payload) ..];
        while (payload.len != 0 and payload[0] == '0') payload = payload[1..];
        return try owned(allocator, .quiet_nan, selected.negative, payload, 0);
    }

    fn addOrSubtract(self: *Context, allocator: Allocator, left: Decimal, right: Decimal, subtract_right: bool) Error!Decimal {
        try self.validate();
        if (try self.propagateNan(allocator, left, right)) |nan| return nan;
        const right_negative = right.negative != subtract_right;
        if (left.kind == .infinity or right.kind == .infinity) {
            if (left.kind == .infinity and right.kind == .infinity and left.negative != right_negative)
                return self.invalid(allocator);
            return owned(allocator, .infinity, if (left.kind == .infinity) left.negative else right_negative, "0", 0);
        }
        const preferred = @min(left.exponent, right.exponent);
        const left_zero = left.isZero();
        const right_zero = right.isZero();
        const same_sign = left.negative == right_negative;
        var larger = left;
        var smaller = right;
        var negative = left.negative;
        if (!same_sign) {
            var left_magnitude = left;
            left_magnitude.negative = false;
            var right_magnitude = right;
            right_magnitude.negative = false;
            const comparison = left_magnitude.order(right_magnitude) catch unreachable;
            if (comparison == .eq)
                return self.finish(allocator, self.rounding == .floor, "0", preferred, false);
            if (comparison == .lt) {
                larger = right;
                smaller = left;
                negative = right_negative;
            }
        } else if (left_zero and right_zero) {
            return self.finish(allocator, negative, "0", preferred, false);
        }

        // A zero contributes its preferred quantum, but never a high digit.
        // Including a coarser zero's exponent here could allocate a gigantic
        // ring for a one-digit result when the context precision is large.
        const left_end: i128 = if (left_zero) preferred else @as(i128, left.exponent) + left.digits.len;
        const right_end: i128 = if (right_zero) preferred else @as(i128, right.exponent) + right.digits.len;
        const end = @max(left_end, right_end);
        const carry_room: i128 = @intFromBool(same_sign and !left_zero and !right_zero);
        const capacity: usize = @intCast(@min(@as(i128, self.precision) + 2, end - preferred + carry_room));
        const storage = try allocator.alloc(u8, capacity);
        defer allocator.free(storage);
        var leading: LeadingDigits = .{ .buffer = storage };
        var position: i128 = preferred;
        var carry_or_borrow: i16 = 0;
        while (position < end) {
            const left_active = activeAt(larger, position);
            const right_active = activeAt(smaller, position);
            if (!left_active and !right_active) {
                const next = @min(
                    if (position < larger.exponent) @as(i128, larger.exponent) else end,
                    if (position < smaller.exponent) @as(i128, smaller.exponent) else end,
                );
                const gap = next - position;
                std.debug.assert(gap > 0);
                if (same_sign and carry_or_borrow != 0) {
                    leading.emit('1', 1);
                    leading.emit('0', gap - 1);
                    carry_or_borrow = 0;
                } else {
                    leading.emit(if (carry_or_borrow == 0) '0' else '9', gap);
                }
                position = next;
                continue;
            }
            const a = alignedDigit(larger, position);
            const b = alignedDigit(smaller, position);
            if (same_sign) {
                const next = a + b + carry_or_borrow;
                leading.emit(@as(u8, @intCast(@mod(next, 10))) + '0', 1);
                carry_or_borrow = @divTrunc(next, 10);
            } else {
                var next = a - b - carry_or_borrow;
                carry_or_borrow = @intFromBool(next < 0);
                if (next < 0) next += 10;
                leading.emit(@as(u8, @intCast(next)) + '0', 1);
            }
            position += 1;
        }
        if (same_sign and carry_or_borrow != 0) leading.emit('1', 1);
        if (!same_sign) std.debug.assert(carry_or_borrow == 0);
        std.debug.assert(leading.len != 0);
        const exponent = @as(i128, preferred) + leading.width - leading.len;
        const sticky = leading.sticky;
        return self.finish(allocator, negative, leading.bigEndian(), exponent, sticky);
    }

    /// The borrowed coefficient is normalized and exact, except that sticky
    /// may represent a nonzero suffix strictly below its last digit. Producers
    /// retain precision+2 digits before discarding such a suffix. Precision and
    /// Etiny are applied together, once, to avoid double rounding at underflow.
    fn finish(self: *Context, allocator: Allocator, negative: bool, digits: []const u8, exponent: i128, sticky: bool) Error!Decimal {
        std.debug.assert(digits.len != 0);
        const tiny: i128 = self.etiny();
        const top: i128 = self.etop();
        var raised: Signals = .empty;
        if (std.mem.eql(u8, digits, "0")) {
            std.debug.assert(!sticky);
            const upper: i128 = if (self.clamp) top else self.emax;
            const result_exponent = @min(@max(exponent, tiny), upper);
            if (result_exponent != exponent) raised.insert(.clamped);
            try self.raise(raised);
            return owned(allocator, .finite, negative, "0", result_exponent);
        }
        const adjusted = exponent + digits.len - 1;
        if (adjusted > self.emax) return self.overflowResult(allocator, negative);
        const subnormal = adjusted < self.emin;
        if (subnormal) raised.insert(.subnormal);
        var result_exponent = @max(exponent, @max(exponent + digits.len - self.precision, tiny));
        const drop = result_exponent - exponent;
        const keep: usize = if (drop >= digits.len) 0 else digits.len - @as(usize, @intCast(drop));
        std.debug.assert(!sticky or drop != 0);
        var increment = false;
        if (drop != 0) {
            raised.insert(.rounded);
            const guard: u8 = if (drop > digits.len) '0' else digits[keep];
            const rest = if (drop > digits.len) true else sticky or hasNonzero(digits[keep + 1 ..]);
            const inexact = guard != '0' or rest;
            if (inexact) {
                raised.insert(.inexact);
                if (subnormal) raised.insert(.underflow);
            }
            const last: u8 = if (keep == 0) '0' else digits[keep - 1];
            increment = roundsUp(self.rounding, negative, last, guard, rest);
        }
        const carry = increment and allNines(digits[0..keep]);
        var result_length: usize = if (keep == 0) 1 else keep + @intFromBool(carry);
        if (result_length > self.precision) {
            result_length -= 1;
            result_exponent += 1;
        }
        if (keep != 0 and result_exponent + result_length - 1 > self.emax)
            return self.overflowResult(allocator, negative);
        const zero = keep == 0 and !increment;
        if (zero) raised.insert(.clamped);
        var padding: usize = 0;
        if (!zero and self.clamp and result_exponent > top) {
            padding = @intCast(result_exponent - top);
            result_exponent = top;
            raised.insert(.clamped);
        }
        // Allocate the result only after resolving traps. Caller scratch is
        // released by its defer on error; all co-occurring flags survive.
        try self.raise(raised);
        const output = try allocator.alloc(u8, try sizeSum(result_length, padding));
        if (keep == 0) {
            output[0] = if (increment) '1' else '0';
        } else if (carry) {
            output[0] = '1';
            @memset(output[1..result_length], '0');
        } else {
            @memcpy(output[0..keep], digits[0..keep]);
            if (increment) {
                var index = keep;
                while (index != 0) {
                    index -= 1;
                    if (output[index] != '9') {
                        output[index] += 1;
                        break;
                    }
                    output[index] = '0';
                }
            }
        }
        @memset(output[result_length..], '0');
        return .{ .negative = negative, .digits = output, .exponent = @intCast(result_exponent) };
    }

    fn overflowResult(self: *Context, allocator: Allocator, negative: bool) Error!Decimal {
        try self.raise(.initMany(&.{ .overflow, .inexact, .rounded }));
        const infinite = switch (self.rounding) {
            .half_down, .half_even, .half_up, .up => true,
            .ceiling => !negative,
            .floor => negative,
            .down, .zero_five_up => false,
        };
        if (infinite) return owned(allocator, .infinity, negative, "0", 0);
        const digits = try allocator.alloc(u8, self.precision);
        @memset(digits, '9');
        return .{ .negative = negative, .digits = digits, .exponent = self.etop() };
    }
};

fn owned(allocator: Allocator, kind: decimal.Kind, negative: bool, digits: []const u8, exponent: i128) Allocator.Error!Decimal {
    return .{ .kind = kind, .negative = negative, .digits = try allocator.dupe(u8, digits), .exponent = @intCast(exponent) };
}

fn isNan(value: Decimal) bool {
    return value.kind == .quiet_nan or value.kind == .signaling_nan;
}

fn sizeSum(a: usize, b: usize) Allocator.Error!usize {
    const sum, const overflow = @addWithOverflow(a, b);
    if (overflow != 0) return error.OutOfMemory;
    return sum;
}

fn hasNonzero(digits: []const u8) bool {
    for (digits) |digit| if (digit != '0') return true;
    return false;
}

fn allNines(digits: []const u8) bool {
    for (digits) |digit| if (digit != '9') return false;
    return true;
}

fn roundsUp(rounding: Rounding, negative: bool, last: u8, guard: u8, rest: bool) bool {
    const inexact = guard != '0' or rest;
    return switch (rounding) {
        .ceiling => inexact and !negative,
        .down => false,
        .floor => inexact and negative,
        .half_down => guard > '5' or (guard == '5' and rest),
        .half_even => guard > '5' or (guard == '5' and (rest or (last - '0') % 2 != 0)),
        .half_up => guard >= '5',
        .up => inexact,
        .zero_five_up => inexact and (last == '0' or last == '5'),
    };
}

fn activeAt(value: Decimal, position: i128) bool {
    return position >= value.exponent and position < @as(i128, value.exponent) + value.digits.len;
}

fn alignedDigit(value: Decimal, position: i128) i16 {
    if (!activeAt(value, position)) return 0;
    const index = value.digits.len - 1 - @as(usize, @intCast(position - value.exponent));
    return value.digits[index] - '0';
}

/// Low-to-high sparse result accumulator. Only the highest precision+2 digits
/// are retained in a ring, plus a sticky bit for dropped nonzero digits. High
/// zeroes remain pending until a higher nonzero arrives: cancellation can then
/// expose a tiny exact difference without having discarded it as a low suffix.
/// Arbitrarily long zero/nine runs cost at most one ring's worth of work.
const LeadingDigits = struct {
    buffer: []u8,
    start: usize = 0,
    len: usize = 0,
    width: i128 = 0,
    pending_zeroes: i128 = 0,
    sticky: bool = false,

    fn emit(self: *LeadingDigits, digit: u8, count: i128) void {
        std.debug.assert(count >= 0);
        if (count == 0) return;
        if (digit == '0') {
            self.pending_zeroes += count;
            return;
        }
        self.appendRun('0', self.pending_zeroes);
        self.pending_zeroes = 0;
        self.appendRun(digit, count);
    }

    fn appendRun(self: *LeadingDigits, digit: u8, count: i128) void {
        if (count == 0) return;
        self.width += count;
        if (count >= self.buffer.len) {
            for (0..self.len) |index| {
                if (self.buffer[(self.start + index) % self.buffer.len] != '0') self.sticky = true;
            }
            if (digit != '0' and count > self.buffer.len) self.sticky = true;
            @memset(self.buffer, digit);
            self.start = 0;
            self.len = self.buffer.len;
            return;
        }
        const n: usize = @intCast(count);
        for (0..n) |_| {
            if (self.len == self.buffer.len) {
                if (self.buffer[self.start] != '0') self.sticky = true;
                self.buffer[self.start] = digit;
                self.start = (self.start + 1) % self.buffer.len;
            } else {
                self.buffer[self.len] = digit;
                self.len += 1;
            }
        }
    }

    /// Consume the ring in place, without another coefficient allocation.
    fn bigEndian(self: *LeadingDigits) []const u8 {
        std.mem.reverse(u8, self.buffer[0..self.start]);
        std.mem.reverse(u8, self.buffer[self.start..self.len]);
        return self.buffer[0..self.len];
    }
};

/// Circular big-endian decimal remainder. Appending a low digit and removing
/// leading zeroes move indices, not the coefficient. Before each step r < d;
/// afterwards 10*r+digit < 10*d, so at most nine subtractions are required.
const Remainder = struct {
    buffer: []u8,
    start: usize = 0,
    len: usize = 1,

    fn init(buffer: []u8) Remainder {
        buffer[0] = '0';
        return .{ .buffer = buffer };
    }

    fn digitAt(self: *const Remainder, index: usize) u8 {
        return self.buffer[(self.start + index) % self.buffer.len];
    }

    fn isZero(self: *const Remainder) bool {
        return self.len == 1 and self.digitAt(0) == '0';
    }

    fn atLeast(self: *const Remainder, divisor: []const u8) bool {
        if (self.len != divisor.len) return self.len > divisor.len;
        for (divisor, 0..) |digit, index| {
            const current = self.digitAt(index);
            if (current != digit) return current > digit;
        }
        return true;
    }

    fn subtract(self: *Remainder, divisor: []const u8) void {
        var borrow: i16 = 0;
        var offset: usize = 0;
        while (offset < self.len) : (offset += 1) {
            const index = (self.start + self.len - 1 - offset) % self.buffer.len;
            const other: i16 = if (offset < divisor.len) divisor[divisor.len - 1 - offset] - '0' else 0;
            var digit = @as(i16, self.buffer[index] - '0') - other - borrow;
            borrow = @intFromBool(digit < 0);
            if (digit < 0) digit += 10;
            self.buffer[index] = @as(u8, @intCast(digit)) + '0';
        }
        std.debug.assert(borrow == 0);
        while (self.len > 1 and self.digitAt(0) == '0') {
            self.start = (self.start + 1) % self.buffer.len;
            self.len -= 1;
        }
    }

    fn step(self: *Remainder, digit: u8, divisor: []const u8) u8 {
        if (self.isZero()) {
            self.buffer[self.start] = digit;
        } else {
            std.debug.assert(self.len < self.buffer.len);
            self.buffer[(self.start + self.len) % self.buffer.len] = digit;
            self.len += 1;
        }
        var quotient: u8 = 0;
        while (self.atLeast(divisor)) {
            self.subtract(divisor);
            quotient += 1;
        }
        std.debug.assert(quotient <= 9);
        return quotient + '0';
    }
};

const TestOperation = enum { add, subtract, multiply, divide, absolute };

fn expectFlags(context: Context, expected: []const Signal) !void {
    const set: Signals = .initMany(expected);
    for (signal_priority) |condition| try std.testing.expectEqual(set.contains(condition), context.flags.contains(condition));
}

fn expectDecimal(expected: Decimal, actual: Decimal) !void {
    try std.testing.expectEqual(expected.kind, actual.kind);
    try std.testing.expectEqual(expected.negative, actual.negative);
    try std.testing.expectEqual(expected.exponent, actual.exponent);
    try std.testing.expectEqualStrings(expected.digits, actual.digits);
}

fn expectOperation(context: *Context, operation: TestOperation, left_text: []const u8, right_text: []const u8, expected_text: []const u8, flags: []const Signal) !void {
    const allocator = std.testing.allocator;
    const left = try decimal.parse(allocator, left_text);
    defer left.deinit(allocator);
    const right = try decimal.parse(allocator, right_text);
    defer right.deinit(allocator);
    const expected = try decimal.parse(allocator, expected_text);
    defer expected.deinit(allocator);
    context.clearFlags();
    const result = switch (operation) {
        .add => try context.add(allocator, left, right),
        .subtract => try context.subtract(allocator, left, right),
        .multiply => try context.multiply(allocator, left, right),
        .divide => try context.divide(allocator, left, right),
        .absolute => try context.absolute(allocator, left),
    };
    defer result.deinit(allocator);
    try expectDecimal(expected, result);
    try expectFlags(context.*, flags);
}

test "Decimal context exact arithmetic preserves subfloat differences and quantum" {
    var context: Context = .{};
    try expectOperation(&context, .subtract, "1.000000000000000000000000001", "1", "1E-27", &.{});
    try expectOperation(&context, .subtract, "9007199254740993", "9007199254740992", "1", &.{});
    try expectOperation(&context, .add, "1.20", "3.400", "4.600", &.{});
    try expectOperation(&context, .multiply, "123456789", "987654321", "121932631112635269", &.{});
    try expectOperation(&context, .multiply, "-1.20", "3.0", "-3.600", &.{});
    try expectOperation(&context, .divide, "1.00", "2", "0.50", &.{});
    try expectOperation(&context, .divide, "100", "2.00", "5E+1", &.{});
    try expectOperation(&context, .divide, "1", "8", "0.125", &.{});
    try expectOperation(&context, .divide, "1", "3", "0.3333333333333333333333333333", &.{ .inexact, .rounded });
}

test "Decimal context alignment skips huge gaps and retains cancellation" {
    var context: Context = .{ .precision = 7, .emax = 1000000000 };
    try expectOperation(&context, .add, "1E+1000000000", "1", "1.000000E+1000000000", &.{ .inexact, .rounded });
    try expectOperation(&context, .subtract, "1E+1000000000", "1", "1.000000E+1000000000", &.{ .inexact, .rounded });
    context.rounding = .down;
    try expectOperation(&context, .subtract, "1E+1000000000", "1", "9.999999E+999999999", &.{ .inexact, .rounded });
    try expectOperation(&context, .add, "1E+1000000000", "0", "1.000000E+1000000000", &.{.rounded});
    context.precision = 3;
    try expectOperation(&context, .subtract, "100000000000000000000000000000001", "1E+32", "1", &.{});
    try expectOperation(&context, .subtract, "1E+32", "100000000000000000000000000000001", "-1", &.{});
}

test "Decimal context all eight rounding modes handle both tie signs" {
    const Case = struct { mode: Rounding, positive: []const u8, negative: []const u8 };
    const cases = [_]Case{
        .{ .mode = .ceiling, .positive = "3", .negative = "-2" },
        .{ .mode = .down, .positive = "2", .negative = "-2" },
        .{ .mode = .floor, .positive = "2", .negative = "-3" },
        .{ .mode = .half_down, .positive = "2", .negative = "-2" },
        .{ .mode = .half_even, .positive = "2", .negative = "-2" },
        .{ .mode = .half_up, .positive = "3", .negative = "-3" },
        .{ .mode = .up, .positive = "3", .negative = "-3" },
        .{ .mode = .zero_five_up, .positive = "2", .negative = "-2" },
    };
    for (cases) |case| {
        var context: Context = .{ .precision = 1, .rounding = case.mode };
        try expectOperation(&context, .add, "2.5", "0", case.positive, &.{ .inexact, .rounded });
        try expectOperation(&context, .multiply, "-2.5", "1", case.negative, &.{ .inexact, .rounded });
    }
    var context: Context = .{ .precision = 2 };
    try expectOperation(&context, .absolute, "-9.95", "0", "10", &.{ .inexact, .rounded });
    try expectOperation(&context, .add, "1.251", "0", "1.3", &.{ .inexact, .rounded });
    context.rounding = .zero_five_up;
    try expectOperation(&context, .multiply, "1.51", "1", "1.6", &.{ .inexact, .rounded });
    try expectOperation(&context, .multiply, "1.01", "1", "1.1", &.{ .inexact, .rounded });
    try expectOperation(&context, .multiply, "1.00", "1", "1.0", &.{.rounded});
}

test "Decimal context rounds once at Etiny and distinguishes exact subnormals" {
    var context: Context = .{ .precision = 3, .emin = -2, .emax = 2, .traps = .empty };
    try expectOperation(&context, .multiply, "1E-4", "1", "0.0001", &.{.subnormal});
    try expectOperation(&context, .multiply, "1.00E-3", "1", "0.0010", &.{ .subnormal, .rounded });
    try expectOperation(&context, .multiply, "1.5E-4", "1", "0.0002", &.{ .underflow, .subnormal, .inexact, .rounded });
    // Rounding to precision first would turn 0.00005001 into the wrong tie.
    try expectOperation(&context, .absolute, "0.00005001", "0", "0.0001", &.{ .underflow, .subnormal, .inexact, .rounded });
    try expectOperation(&context, .divide, "1E-4", "2", "0.0000", &.{ .underflow, .subnormal, .inexact, .rounded, .clamped });
    try expectOperation(&context, .absolute, "0.00999", "0", "0.0100", &.{ .underflow, .subnormal, .inexact, .rounded });
    context.rounding = .floor;
    try expectOperation(&context, .multiply, "-1E-4", "0.1", "-0.0001", &.{ .underflow, .subnormal, .inexact, .rounded });
    context.rounding = .ceiling;
    try expectOperation(&context, .multiply, "-1E-4", "0.1", "-0.0000", &.{ .underflow, .subnormal, .inexact, .rounded, .clamped });
}

test "Decimal context overflow follows sign and rounding direction" {
    const Case = struct { mode: Rounding, positive: []const u8, negative: []const u8 };
    const cases = [_]Case{
        .{ .mode = .ceiling, .positive = "Infinity", .negative = "-999" },
        .{ .mode = .down, .positive = "999", .negative = "-999" },
        .{ .mode = .floor, .positive = "999", .negative = "-Infinity" },
        .{ .mode = .half_down, .positive = "Infinity", .negative = "-Infinity" },
        .{ .mode = .half_even, .positive = "Infinity", .negative = "-Infinity" },
        .{ .mode = .half_up, .positive = "Infinity", .negative = "-Infinity" },
        .{ .mode = .up, .positive = "Infinity", .negative = "-Infinity" },
        .{ .mode = .zero_five_up, .positive = "999", .negative = "-999" },
    };
    for (cases) |case| {
        var context: Context = .{ .precision = 3, .emin = -2, .emax = 2, .rounding = case.mode, .traps = .empty };
        try expectOperation(&context, .multiply, "1E+2", "10", case.positive, &.{ .overflow, .inexact, .rounded });
        try expectOperation(&context, .multiply, "-1E+2", "10", case.negative, &.{ .overflow, .inexact, .rounded });
    }
    var context: Context = .{ .precision = 3, .emin = -2, .emax = 2, .traps = .empty };
    try expectOperation(&context, .add, "999.5", "0", "Infinity", &.{ .overflow, .inexact, .rounded });
}

test "Decimal context clamp and signed zero preserve result exponent policy" {
    var context: Context = .{ .precision = 7, .emin = -2, .emax = 2, .clamp = true };
    try expectOperation(&context, .add, "1E+2", "0", "100.0000", &.{.clamped});
    try expectOperation(&context, .multiply, "0E-99", "1", "0E-8", &.{.clamped});
    try expectOperation(&context, .multiply, "-0E+99", "1", "-0E-4", &.{.clamped});
    context.clamp = false;
    try expectOperation(&context, .multiply, "-0E+99", "1", "-0E+2", &.{.clamped});
    try expectOperation(&context, .add, "0.00", "-0.000", "0.000", &.{});
    try expectOperation(&context, .subtract, "-0.00", "0.000", "-0.000", &.{});
    try expectOperation(&context, .absolute, "-0.00", "0", "0.00", &.{});
    context.rounding = .floor;
    try expectOperation(&context, .subtract, "1", "1", "-0", &.{});
    try expectOperation(&context, .add, "0.00", "-0.000", "-0.000", &.{});
}

test "Decimal context NaNs preserve precedence sign and suffix payload" {
    var context: Context = .{ .precision = 3, .traps = .empty };
    try expectOperation(&context, .subtract, "NaN12345", "-sNaN67890", "-NaN890", &.{.invalid_operation});
    try expectOperation(&context, .add, "-sNaN12345", "sNaN67890", "-NaN345", &.{.invalid_operation});
    try expectOperation(&context, .add, "-NaN12345", "NaN67890", "-NaN345", &.{});
    try expectOperation(&context, .absolute, "-NaN12345", "0", "-NaN345", &.{});
    context.clamp = true;
    try expectOperation(&context, .multiply, "NaN12001", "1", "NaN1", &.{});
    context.precision = 1;
    try expectOperation(&context, .add, "-sNaN42", "1", "-NaN", &.{.invalid_operation});
    try expectOperation(&context, .multiply, "Infinity", "-0", "NaN", &.{.invalid_operation});
    try expectOperation(&context, .divide, "Infinity", "-Infinity", "NaN", &.{.invalid_operation});
    try expectOperation(&context, .divide, "-1", "-0", "Infinity", &.{.division_by_zero});
    try expectOperation(&context, .divide, "Infinity", "0", "Infinity", &.{});
    try expectOperation(&context, .divide, "-1", "Infinity", "-0E-999999", &.{.clamped});
}

test "Decimal context comparisons are exact and NaN ordering is unordered" {
    const allocator = std.testing.allocator;
    var context: Context = .{};
    const left = try decimal.parse(allocator, "9007199254740993");
    defer left.deinit(allocator);
    const right = try decimal.parse(allocator, "9007199254740992");
    defer right.deinit(allocator);
    const quiet = try decimal.parse(allocator, "NaN42");
    defer quiet.deinit(allocator);
    const signaling = try decimal.parse(allocator, "sNaN42");
    defer signaling.deinit(allocator);
    try std.testing.expectEqual(std.math.Order.gt, (try context.order(left, right)).?);
    try std.testing.expect(!try context.equal(left, right));
    try std.testing.expect(!try context.equal(quiet, left));
    try expectFlags(context, &.{});
    try std.testing.expectError(error.InvalidOperation, context.order(quiet, left));
    try expectFlags(context, &.{.invalid_operation});
    context.clearFlags();
    try std.testing.expectError(error.InvalidOperation, context.equal(signaling, left));
    context.traps = .empty;
    context.clearFlags();
    try std.testing.expectEqual(@as(?std.math.Order, null), try context.order(quiet, left));
    try std.testing.expect(!try context.equal(signaling, left));
    try expectFlags(context, &.{.invalid_operation});
}

test "Decimal context traps commit all flags and do not retrap old flags" {
    const allocator = std.testing.allocator;
    const tiny = try decimal.parse(allocator, "1E-4");
    defer tiny.deinit(allocator);
    const ten = try decimal.parse(allocator, "10");
    defer ten.deinit(allocator);
    const third_divisor = try decimal.parse(allocator, "3");
    defer third_divisor.deinit(allocator);
    const one = try decimal.parse(allocator, "1");
    defer one.deinit(allocator);
    var context: Context = .{ .precision = 3, .emin = -2, .emax = 2, .traps = .full };
    try std.testing.expectError(error.Underflow, context.divide(allocator, tiny, ten));
    try expectFlags(context, &.{ .underflow, .subnormal, .inexact, .rounded, .clamped });
    context.clearFlags();
    context.traps = .initMany(&.{ .subnormal, .inexact, .rounded, .clamped });
    try std.testing.expectError(error.Subnormal, context.divide(allocator, tiny, ten));
    try expectFlags(context, &.{ .underflow, .subnormal, .inexact, .rounded, .clamped });
    context.clearFlags();
    context.traps = .initMany(&.{ .inexact, .rounded });
    try std.testing.expectError(error.Inexact, context.divide(allocator, one, third_divisor));
    try expectFlags(context, &.{ .inexact, .rounded });
    const exact = try context.multiply(allocator, one, one);
    defer exact.deinit(allocator);
    try std.testing.expectEqualStrings("1", exact.digits);
    try expectFlags(context, &.{ .inexact, .rounded });
    context.clearFlags();
    context.traps = .initOne(.float_operation);
    try std.testing.expectError(error.FloatOperation, context.signal(.float_operation));
    try expectFlags(context, &.{.float_operation});
}

test "Decimal context owns coefficients independently of operand storage" {
    const allocator = std.testing.allocator;
    var context: Context = .{};
    const left = try decimal.parse(allocator, "123.4500");
    const right = decimal.parse(allocator, "2") catch |err| {
        left.deinit(allocator);
        return err;
    };
    const result = context.multiply(allocator, left, right) catch |err| {
        left.deinit(allocator);
        right.deinit(allocator);
        return err;
    };
    left.deinit(allocator);
    right.deinit(allocator);
    defer result.deinit(allocator);
    try std.testing.expectEqualStrings("2469000", result.digits);
    try std.testing.expectEqual(@as(i64, -4), result.exponent);
}

test "Decimal context arithmetic releases scratch and results at every allocation failure" {
    // Quotient ArrayList growth must not make the fault-site count depend on
    // whether a backing allocation happens to have spare adjacent capacity.
    var stable = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(stable.allocator(), struct {
        fn run(allocator: Allocator) !void {
            var context: Context = .{};
            const left: Decimal = .{ .digits = "123456789" };
            const right: Decimal = .{ .digits = "987654321" };
            const sum = try context.add(allocator, left, right);
            defer sum.deinit(allocator);
            const difference = try context.subtract(allocator, left, right);
            defer difference.deinit(allocator);
            const product = try context.multiply(allocator, left, right);
            defer product.deinit(allocator);
            const quotient = try context.divide(allocator, left, right);
            defer quotient.deinit(allocator);
            const positive = try context.absolute(allocator, .{ .negative = true, .digits = "123456789" });
            defer positive.deinit(allocator);
            const nan = try context.add(allocator, .{ .kind = .quiet_nan, .digits = "12345678901234567890123456789" }, left);
            defer nan.deinit(allocator);
            const infinity = try context.absolute(allocator, .{ .kind = .infinity, .negative = true, .digits = "0" });
            defer infinity.deinit(allocator);
            const zero = try context.divide(allocator, left, infinity);
            defer zero.deinit(allocator);

            context.clearFlags();
            context.traps = .initOne(.inexact);
            const unexpected = context.divide(allocator, left, right) catch |err| {
                if (err != error.Inexact) return err;
                try expectFlags(context, &.{ .inexact, .rounded });
                return;
            };
            unexpected.deinit(allocator);
            return error.TestUnexpectedResult;
        }
    }.run, .{});
}

test "Decimal context result allocation failure retains newly raised and existing flags" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var context: Context = .{ .precision = 3 };
    try context.signal(.float_operation);
    try std.testing.expectError(error.OutOfMemory, context.absolute(failing.allocator(), .{ .digits = "12345", .exponent = -4 }));
    try expectFlags(context, &.{ .float_operation, .inexact, .rounded });
}

test "Decimal context rejects invalid settings before allocation without clearing flags" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const allocator = failing.allocator();
    const zero: Decimal = .{ .digits = "0" };
    const invalid = [_]Context{
        .{ .precision = 0 },
        .{ .emin = 1 },
        .{ .emax = -1 },
        .{ .emin = -decimal.max_exponent - 1 },
        .{ .emax = decimal.max_exponent + 1 },
    };
    for (invalid) |initial| {
        var context = initial;
        try context.signal(.float_operation);
        try std.testing.expectError(error.InvalidContext, context.add(allocator, zero, zero));
        try std.testing.expectError(error.InvalidContext, context.subtract(allocator, zero, zero));
        try std.testing.expectError(error.InvalidContext, context.multiply(allocator, zero, zero));
        try std.testing.expectError(error.InvalidContext, context.divide(allocator, zero, zero));
        try std.testing.expectError(error.InvalidContext, context.absolute(allocator, zero));
        try expectFlags(context, &.{.float_operation});
    }
    if (comptime @bitSizeOf(usize) >= 64) {
        var context: Context = .{ .precision = std.math.maxInt(usize) };
        try std.testing.expectError(error.InvalidContext, context.absolute(allocator, zero));
        try expectFlags(context, &.{});
    }
}

test "Decimal context handles full exponent ranges and maximum precision zero arithmetic" {
    var context: Context = .{
        .precision = 1,
        .emin = -decimal.max_exponent,
        .emax = decimal.max_exponent,
        .traps = .empty,
    };
    try expectOperation(&context, .multiply, "0E+999999999999999999", "1E+999999999999999999", "0E+999999999999999999", &.{.clamped});
    try expectOperation(&context, .divide, "0E-1999999999999999997", "1E+999999999999999999", "0E-999999999999999999", &.{.clamped});
    try expectOperation(&context, .divide, "1E-1999999999999999997", "1E+999999999999999999", "0E-999999999999999999", &.{ .underflow, .subnormal, .inexact, .rounded, .clamped });
    try expectOperation(&context, .multiply, "1E+999999999999999999", "10", "Infinity", &.{ .overflow, .inexact, .rounded });
    try expectOperation(&context, .add, "1E+999999999999999999", "1", "1E+999999999999999999", &.{ .inexact, .rounded });
    try expectOperation(&context, .subtract, "1E+999999999999999999", "1", "1E+999999999999999999", &.{ .inexact, .rounded });

    if (comptime @bitSizeOf(usize) >= 64) {
        context.precision = @intCast(decimal.max_exponent);
        try expectOperation(&context, .add, "1", "0E+999999999999999999", "1", &.{});
        try expectOperation(&context, .add, "0E+999999999999999999", "1", "1", &.{});
        try expectOperation(&context, .subtract, "1", "0E+999999999999999999", "1", &.{});
        try expectOperation(&context, .subtract, "0E+999999999999999999", "1", "-1", &.{});
        try expectOperation(&context, .multiply, "0E-999999999999999999", "0E-999999999999999999", "0E-1999999999999999997", &.{.clamped});
        context.clamp = true;
        try expectOperation(&context, .multiply, "0E+999999999999999999", "1", "0E+1", &.{.clamped});
    }
}
