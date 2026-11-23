//! Exact CPython Decimal coefficient/exponent representation. Parsing does not
//! round through binary floating point or apply an arithmetic context. A parsed
//! coefficient is owned by the supplied allocator; formatting borrows it.
const std = @import("std");
const unicode = @import("unicode.zig");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

pub const Error = Allocator.Error || error{ParseValue};
pub const Kind = enum { finite, infinity, quiet_nan, signaling_nan };
pub const max_exponent: i64 = 999999999999999999;
pub const min_exponent: i64 = -1999999999999999997;

pub const Decimal = struct {
    negative: bool = false,
    /// ASCII coefficient, without leading zeroes. Finite zero and Infinity
    /// have coefficient "0"; NaNs without a diagnostic have an empty one.
    digits: []const u8,
    exponent: i64 = 0,
    kind: Kind = .finite,

    pub fn deinit(self: Decimal, allocator: Allocator) void {
        allocator.free(self.digits);
    }

    pub fn isZero(self: Decimal) bool {
        return self.kind == .finite and std.mem.eql(u8, self.digits, "0");
    }

    pub fn truthy(self: Decimal) bool {
        return !self.isZero();
    }

    /// Quiet copy_abs: no context rounding or signaling, even for sNaN. This
    /// metadata view borrows the original coefficient; boxing it must own a copy.
    pub fn copyAbs(self: Decimal) Decimal {
        var result = self;
        result.negative = false;
        return result;
    }

    /// CPython's canonical str preserves quantum/trailing zeroes. Capitals is
    /// a formatting-context setting, not a change to the stored exponent.
    pub fn write(self: Decimal, writer: *Writer, capitals: bool) Writer.Error!void {
        if (self.negative) try writer.writeByte('-');
        switch (self.kind) {
            .infinity => return writer.writeAll("Infinity"),
            .quiet_nan, .signaling_nan => {
                try writer.writeAll(if (self.kind == .quiet_nan) "NaN" else "sNaN");
                return writer.writeAll(self.digits);
            },
            .finite => {},
        }
        const adjusted = @as(i128, self.exponent) + self.digits.len - 1;
        if (self.exponent <= 0 and adjusted >= -6) {
            const point = @as(i128, self.digits.len) + self.exponent;
            if (point <= 0) {
                try writer.writeAll("0.");
                try writer.splatByteAll('0', @intCast(-point));
                try writer.writeAll(self.digits);
            } else {
                const index: usize = @intCast(point);
                try writer.writeAll(self.digits[0..index]);
                if (index < self.digits.len) {
                    try writer.writeByte('.');
                    try writer.writeAll(self.digits[index..]);
                }
            }
        } else {
            try writer.writeByte(self.digits[0]);
            if (self.digits.len > 1) {
                try writer.writeByte('.');
                try writer.writeAll(self.digits[1..]);
            }
            try writer.print("{c}{s}{d}", .{ @as(u8, if (capitals) 'E' else 'e'), if (adjusted >= 0) "+" else "", adjusted });
        }
    }

    /// Numeric comparison is independent of coefficient quantum. Ordering a
    /// quiet or signaling NaN fails under the default Python Decimal context.
    pub fn order(self: Decimal, other: Decimal) error{ParseValue}!std.math.Order {
        if (self.kind == .quiet_nan or self.kind == .signaling_nan or
            other.kind == .quiet_nan or other.kind == .signaling_nan) return error.ParseValue;
        if (self.isZero() and other.isZero()) return .eq;
        if (self.negative != other.negative) return if (self.negative) .lt else .gt;
        const magnitude: std.math.Order = magnitude: {
            if (self.kind == .infinity)
                break :magnitude if (other.kind == .infinity) .eq else .gt;
            if (other.kind == .infinity) break :magnitude .lt;
            if (self.isZero()) break :magnitude .lt;
            if (other.isZero()) break :magnitude .gt;
            const left = @as(i128, self.exponent) + self.digits.len;
            const right = @as(i128, other.exponent) + other.digits.len;
            if (left != right) break :magnitude std.math.order(left, right);
            for (0..@max(self.digits.len, other.digits.len)) |index| {
                const a = if (index < self.digits.len) self.digits[index] else '0';
                const b = if (index < other.digits.len) other.digits[index] else '0';
                if (a != b) break :magnitude std.math.order(a, b);
            }
            break :magnitude .eq;
        };
        return if (self.negative) magnitude.invert() else magnitude;
    }

    /// Unlike ordering, a quiet NaN compares unequal rather than raising.
    /// Signaling NaN equality is fallible; it must not become silently false.
    pub fn equal(self: Decimal, other: Decimal) error{ParseValue}!bool {
        if (self.kind == .signaling_nan or other.kind == .signaling_nan) return error.ParseValue;
        if (self.kind == .quiet_nan or other.kind == .quiet_nan) return false;
        return try self.order(other) == .eq;
    }

    /// Every binary64 rounding boundary fits within 768 significant decimal
    /// digits (the pinned std parser's bound). Preserve that prefix and a
    /// sticky digit rather than allocating a temporary string for an arbitrary
    /// coefficient. The stored Decimal itself is never truncated or rounded.
    pub fn toFloat(self: Decimal) error{ParseValue}!f64 {
        if (self.kind == .signaling_nan) return error.ParseValue;
        const magnitude: f64 = switch (self.kind) {
            .quiet_nan => std.math.nan(f64),
            .infinity => std.math.inf(f64),
            .finite => finite: {
                var buffer: [800]u8 = undefined;
                var writer: Writer = .fixed(&buffer);
                const length = @min(self.digits.len, 768);
                writer.writeAll(self.digits[0..length]) catch unreachable;
                var exponent = @as(i128, self.exponent) + self.digits.len - length;
                if (length != self.digits.len) {
                    for (self.digits[length..]) |digit| {
                        if (digit != '0') {
                            writer.writeByte('1') catch unreachable;
                            exponent -= 1;
                            break;
                        }
                    }
                }
                writer.print("e{d}", .{exponent}) catch unreachable;
                break :finite std.fmt.parseFloat(f64, writer.buffered()) catch return error.ParseValue;
            },
            .signaling_nan => unreachable,
        };
        return if (self.negative) -magnitude else magnitude;
    }

    /// Python numeric hashing modulo 2**61-1, including negative exponents.
    /// NaN hashes depend on object identity; the owning runtime supplies that
    /// identity rather than assigning the same hash to every NaN payload.
    pub fn hash(self: Decimal, identity_hash: i64) error{ParseValue}!i64 {
        switch (self.kind) {
            .quiet_nan => return identity_hash,
            .signaling_nan => return error.ParseValue,
            .infinity => return if (self.negative) -314159 else 314159,
            .finite => {},
        }
        const modulus: u64 = (1 << 61) - 1;
        var coefficient: u64 = 0;
        for (self.digits) |digit| {
            const next = @as(u128, coefficient) * 10 + digit - '0';
            const folded: u64 = @intCast((next & modulus) + (next >> 61));
            coefficient = if (folded >= modulus) folded - modulus else folded;
        }
        // Fermat's theorem handles the inverse without a giant 10**exponent.
        var exponent: u64 = @intCast(@mod(self.exponent, @as(i64, modulus - 1)));
        var factor: u64 = 10;
        var power: u64 = 1;
        while (exponent != 0) : (exponent >>= 1) {
            if (exponent & 1 != 0) power = @intCast((@as(u128, power) * factor) % modulus);
            factor = @intCast((@as(u128, factor) * factor) % modulus);
        }
        const magnitude: i64 = @intCast((@as(u128, coefficient) * power) % modulus);
        const signed = if (self.negative) -magnitude else magnitude;
        return if (signed == -1) -2 else signed;
    }
};

/// Exact Decimal(float), not Decimal(str(float)). The output borrows buffer;
/// every finite binary64 coefficient fits in 768 decimal digits. No arithmetic
/// context or float conversion is applied while constructing this coefficient.
pub fn fromFloatExact(value: f64, buffer: *[768]u8) Decimal {
    const bits: u64 = @bitCast(value);
    const negative = bits >> 63 != 0;
    const encoded_exponent: i32 = @intCast((bits >> 52) & 0x7ff);
    var mantissa = bits & 0x000f_ffff_ffff_ffff;
    if (encoded_exponent == 0x7ff)
        return .{ .negative = negative and mantissa == 0, .digits = if (mantissa == 0) "0" else "", .kind = if (mantissa == 0) .infinity else .quiet_nan };
    if (encoded_exponent != 0) mantissa |= 1 << 52;
    var power: i32 = if (encoded_exponent == 0) -1074 else encoded_exponent - 1075;
    if (mantissa == 0) {
        buffer[0] = '0';
        return .{ .negative = negative, .digits = buffer[0..1] };
    }
    if (power < 0) {
        const removed = @min(@as(i32, @intCast(@ctz(mantissa))), -power);
        mantissa >>= @intCast(removed);
        power += removed;
    }
    var length: usize = 0;
    while (mantissa != 0) : (mantissa /= 10) {
        buffer[length] = @intCast(mantissa % 10);
        length += 1;
    }
    const multiplier: u8 = if (power < 0) 5 else 2;
    for (0..@abs(power)) |_| {
        var carry: u8 = 0;
        for (buffer[0..length]) |*digit| {
            const next = digit.* * multiplier + carry;
            digit.* = next % 10;
            carry = next / 10;
        }
        if (carry != 0) {
            std.debug.assert(length < buffer.len);
            buffer[length] = carry;
            length += 1;
        }
    }
    std.mem.reverse(u8, buffer[0..length]);
    for (buffer[0..length]) |*digit| digit.* += '0';
    return .{ .negative = negative, .digits = buffer[0..length], .exponent = @min(power, 0) };
}

test "Decimal float conversion preserves signed zero but discards floating NaN sign" {
    const a = std.testing.allocator;
    const nan = try parse(a, "-NaN123");
    defer nan.deinit(a);
    const floating_nan = try nan.toFloat();
    try std.testing.expect(std.math.signbit(floating_nan));
    var buffer: [768]u8 = undefined;
    const converted_nan = fromFloatExact(floating_nan, &buffer);
    try std.testing.expect(converted_nan.kind == .quiet_nan);
    try std.testing.expect(!converted_nan.negative);
    try std.testing.expectEqualStrings("", converted_nan.digits);
    const zero = fromFloatExact(-@as(f64, 0), &buffer);
    try std.testing.expect(zero.negative and zero.isZero());
}

/// Underscores are ignored throughout Decimal's string grammar, unlike Python
/// numeric literals. Unicode Nd digits are mapped without normalizing other
/// characters. Whitespace is trimmed BEFORE underscores are ignored.
const Cursor = struct {
    source: []const u8,
    index: usize = 0,
    current: ?u8 = null,

    fn advance(self: *Cursor) error{ParseValue}!void {
        self.current = null;
        while (self.index < self.source.len) {
            const width = std.unicode.utf8ByteSequenceLength(self.source[self.index]) catch return error.ParseValue;
            if (width > self.source.len - self.index) return error.ParseValue;
            const cp = std.unicode.utf8Decode(self.source[self.index..][0..width]) catch return error.ParseValue;
            self.index += width;
            if (cp == '_') continue;
            self.current = if (cp < 128) @intCast(cp) else if (unicode.decimalDigit(cp)) |digit| '0' + digit else return error.ParseValue;
            return;
        }
    }

    fn word(self: *Cursor, expected: []const u8) error{ParseValue}!void {
        for (expected) |byte| {
            const actual = self.current orelse return error.ParseValue;
            if (std.ascii.toLower(actual) != byte) return error.ParseValue;
            try self.advance();
        }
    }
};

pub fn parse(allocator: Allocator, text: []const u8) Error!Decimal {
    const source = unicode.strip(text, false);
    var cursor: Cursor = .{ .source = source };
    try cursor.advance();
    var result: Decimal = .{ .digits = "" };
    if (cursor.current == '-' or cursor.current == '+') {
        result.negative = cursor.current == '-';
        try cursor.advance();
    }
    const first = cursor.current orelse return error.ParseValue;
    switch (std.ascii.toLower(first)) {
        'i' => {
            try cursor.word("inf");
            if (cursor.current != null) try cursor.word("inity");
            if (cursor.current != null) return error.ParseValue;
            result.kind = .infinity;
            result.digits = try allocator.dupe(u8, "0");
            return result;
        },
        'n', 's' => {
            result.kind = if (std.ascii.toLower(first) == 's') .signaling_nan else .quiet_nan;
            try cursor.word(if (result.kind == .signaling_nan) "snan" else "nan");
        },
        else => {},
    }
    var digits: usize = 0;
    var leading_zeroes: usize = 0;
    var fractional: usize = 0;
    var point = false;
    while (cursor.current) |byte| {
        if (std.ascii.isDigit(byte)) {
            if (byte == '0' and leading_zeroes == digits) leading_zeroes += 1;
            digits += 1;
            if (point) fractional += 1;
        } else if (result.kind == .finite and byte == '.' and !point) {
            point = true;
        } else break;
        try cursor.advance();
    }
    if (result.kind == .finite) {
        if (digits == 0) return error.ParseValue;
        var exponent: i128 = 0;
        if (cursor.current == 'e' or cursor.current == 'E') {
            try cursor.advance();
            const negative = cursor.current == '-';
            if (negative or cursor.current == '+') try cursor.advance();
            var count: usize = 0;
            while (cursor.current) |byte| {
                if (!std.ascii.isDigit(byte)) return error.ParseValue;
                // Inputs far outside libmpdec's exponent range cannot become
                // valid after subtracting the finite input's fractional size.
                exponent = exponent * 10 + byte - '0';
                if (exponent > @as(i128, std.math.maxInt(i64)) + source.len) return error.ParseValue;
                count += 1;
                try cursor.advance();
            }
            if (count == 0) return error.ParseValue;
            if (negative) exponent = -exponent;
        }
        exponent -= fractional;
        if (exponent < min_exponent or exponent > max_exponent) return error.ParseValue;
        result.exponent = @intCast(exponent);
    }
    if (cursor.current != null) return error.ParseValue;
    const significant = digits - leading_zeroes;
    const count = if (result.kind == .finite) @max(significant, 1) else significant;
    const coefficient = try allocator.alloc(u8, count);
    if (significant == 0) {
        if (count != 0) coefficient[0] = '0';
    } else {
        var copy: Cursor = .{ .source = source };
        var observed: usize = 0;
        var written: usize = 0;
        while (observed < digits) {
            // The first pass validated every scalar, so this cannot fail.
            copy.advance() catch unreachable;
            if (copy.current) |byte| {
                if (std.ascii.isDigit(byte)) {
                    if (observed >= leading_zeroes) {
                        coefficient[written] = byte;
                        written += 1;
                    }
                    observed += 1;
                }
            }
        }
        std.debug.assert(written == count);
    }
    result.digits = coefficient;
    return result;
}
