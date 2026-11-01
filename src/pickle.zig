//! `pickle.dumps(tuple(demos))` (protocol 4, CPython's C pickler) for the
//! `dspy.Example(augmented=True, **inputs, **outputs)` demos that DSPy's
//! `BootstrapFewShot` hashes when one predictor ran several times for a
//! single example, and the resulting trace pick:
//!
//!     rng = random.Random(Hasher.hash(tuple(demos)))   # sha256 hex digest
//!     demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
//!
//! Pickle memoizes by object identity. DSZig models identity by value: equal
//! value strings are one object (values flow unchanged between hops), equal
//! field names are one object, and a value never shares an object with a
//! field name, except one-character strings, which CPython caches. Lists and
//! dicts are always distinct objects. Two hops that independently produce
//! equal strings longer than one character are the case this cannot see.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

const pyrandom = @import("pyrandom.zig");
const value = @import("value.zig");
const Record = value.Record;

pub const Error = Allocator.Error || error{Unsupported};

const frame_size_target = 64 * 1024;
const frame_header_size = 9;
const batch_size = 1000;

const op = struct {
    const proto = 0x80;
    const frame = 0x95;
    const memoize = 0x94;
    const binget = 'h';
    const long_binget = 'j';
    const short_binunicode = 0x8c;
    const binunicode = 'X';
    const binunicode8 = 0x8d;
    const binint1 = 'K';
    const binint2 = 'M';
    const binint = 'J';
    const long1 = 0x8a;
    const binfloat = 'G';
    const newtrue = 0x88;
    const newfalse = 0x89;
    const none = 'N';
    const empty_dict = '}';
    const empty_list = ']';
    const empty_tuple = ')';
    const mark = '(';
    const setitem = 's';
    const setitems = 'u';
    const append = 'a';
    const appends = 'e';
    const tuple1 = 0x85;
    const tuple_ = 't';
    const stack_global = 0x93;
    const newobj = 0x81;
    const build = 'b';
    const stop = '.';
};

const Namespace = enum { structure, key, value };

const Pickler = struct {
    gpa: Allocator,
    buf: std.ArrayList(u8) = .empty,
    frame_start: ?usize = null,
    memo_len: u32 = 0,
    memos: [3]std.StringHashMapUnmanaged(u32) = @splat(.empty),
    class: ?u32 = null,

    fn deinit(p: *Pickler) void {
        for (&p.memos) |*m| m.deinit(p.gpa);
        p.buf.deinit(p.gpa);
    }

    fn write(p: *Pickler, bytes: []const u8) Allocator.Error!void {
        if (p.frame_start == null) {
            p.frame_start = p.buf.items.len;
            try p.buf.appendNTimes(p.gpa, 0, frame_header_size);
        }
        try p.buf.appendSlice(p.gpa, bytes);
    }

    fn byte(p: *Pickler, b: u8) Allocator.Error!void {
        try p.write(&.{b});
    }

    fn commitFrame(p: *Pickler) void {
        const start = p.frame_start orelse return;
        const len = p.buf.items.len - start - frame_header_size;
        if (len >= 4) {
            p.buf.items[start] = op.frame;
            std.mem.writeInt(u64, p.buf.items[start + 1 ..][0..8], len, .little);
        } else {
            std.mem.copyForwards(u8, p.buf.items[start..], p.buf.items[start + frame_header_size ..]);
            p.buf.shrinkRetainingCapacity(p.buf.items.len - frame_header_size);
        }
        p.frame_start = null;
    }

    /// The C pickler ends a frame after any `save()` that grows it past 64 KiB.
    fn boundary(p: *Pickler) void {
        const start = p.frame_start orelse return;
        if (p.buf.items.len - start - frame_header_size >= frame_size_target) p.commitFrame();
    }

    fn memoize(p: *Pickler) Allocator.Error!void {
        try p.byte(op.memoize);
        p.memo_len += 1;
    }

    fn get(p: *Pickler, index: u32) Allocator.Error!void {
        if (index < 256) {
            try p.write(&.{ op.binget, @intCast(index) });
        } else {
            var b: [5]u8 = .{ op.long_binget, 0, 0, 0, 0 };
            std.mem.writeInt(u32, b[1..5], index, .little);
            try p.write(&b);
        }
    }

    fn saveStr(p: *Pickler, ns: Namespace, s: []const u8) Error!void {
        const memo = &p.memos[@backingInt(if (isCachedChar(s)) .value else ns)];
        if (memo.get(s)) |index| {
            try p.get(index);
            p.boundary();
            return;
        }
        if (!std.unicode.utf8ValidateSlice(s)) return error.Unsupported;
        if (s.len <= 0xff) {
            try p.write(&.{ op.short_binunicode, @intCast(s.len) });
            try p.write(s);
        } else if (s.len >= frame_size_target) {
            // Large payloads go out unframed, after committing the frame.
            p.commitFrame();
            if (s.len > std.math.maxInt(u32)) {
                var h: [9]u8 = undefined;
                h[0] = op.binunicode8;
                std.mem.writeInt(u64, h[1..9], s.len, .little);
                try p.buf.appendSlice(p.gpa, &h);
            } else {
                var h: [5]u8 = undefined;
                h[0] = op.binunicode;
                std.mem.writeInt(u32, h[1..5], @intCast(s.len), .little);
                try p.buf.appendSlice(p.gpa, &h);
            }
            try p.buf.appendSlice(p.gpa, s);
        } else {
            var h: [5]u8 = undefined;
            h[0] = op.binunicode;
            std.mem.writeInt(u32, h[1..5], @intCast(s.len), .little);
            try p.write(&h);
            try p.write(s);
        }
        try memo.put(p.gpa, s, p.memo_len);
        try p.memoize();
        p.boundary();
    }

    fn saveInt(p: *Pickler, x: i64) Allocator.Error!void {
        if (x >= 0 and x <= 0xff) {
            try p.write(&.{ op.binint1, @intCast(x) });
        } else if (x >= 0 and x <= 0xffff) {
            var b: [3]u8 = undefined;
            b[0] = op.binint2;
            std.mem.writeInt(u16, b[1..3], @intCast(x), .little);
            try p.write(&b);
        } else if (x >= std.math.minInt(i32) and x <= std.math.maxInt(i32)) {
            var b: [5]u8 = undefined;
            b[0] = op.binint;
            std.mem.writeInt(i32, b[1..5], @intCast(x), .little);
            try p.write(&b);
        } else {
            // `encode_long`: little-endian two's complement, minimal length.
            const magnitude: u64 = @abs(x);
            const nbits: usize = 64 - @clz(magnitude);
            var n: usize = (nbits >> 3) + 1;
            var raw: [9]u8 = undefined;
            std.mem.writeInt(i72, &raw, x, .little);
            if (x < 0 and n > 1 and raw[n - 1] == 0xff and raw[n - 2] & 0x80 != 0) n -= 1;
            try p.write(&.{ op.long1, @intCast(n) });
            try p.write(raw[0..n]);
        }
        p.boundary();
    }

    fn saveValue(p: *Pickler, v: Value) Error!void {
        switch (v) {
            .null => try p.byte(op.none),
            .bool => |b| try p.byte(if (b) op.newtrue else op.newfalse),
            .integer => |x| return p.saveInt(x),
            .float => |f| {
                var b: [9]u8 = undefined;
                b[0] = op.binfloat;
                std.mem.writeInt(u64, b[1..9], @bitCast(f), .big);
                try p.write(&b);
            },
            .number_string => return error.Unsupported,
            .string => |s| return p.saveStr(.value, s),
            .array => |a| {
                try p.byte(op.empty_list);
                try p.memoize();
                if (a.items.len == 1) {
                    try p.saveValue(a.items[0]);
                    try p.byte(op.append);
                } else {
                    var i: usize = 0;
                    while (i < a.items.len) {
                        try p.byte(op.mark);
                        const end = @min(a.items.len, i + batch_size);
                        for (a.items[i..end]) |item| try p.saveValue(item);
                        try p.byte(op.appends);
                        i = end;
                    }
                }
            },
            .object => |o| try p.saveDict(.value, &o),
        }
        p.boundary();
    }

    fn saveDict(p: *Pickler, keys: Namespace, d: *const Record) Error!void {
        try p.byte(op.empty_dict);
        try p.memoize();
        const names = d.keys();
        const values = d.values();
        if (names.len == 1) {
            try p.saveStr(keys, names[0]);
            try p.saveValue(values[0]);
            try p.byte(op.setitem);
            return;
        }
        var i: usize = 0;
        while (i < names.len) {
            try p.byte(op.mark);
            const end = @min(names.len, i + batch_size);
            for (names[i..end], values[i..end]) |k, x| {
                try p.saveStr(keys, k);
                try p.saveValue(x);
            }
            try p.byte(op.setitems);
            i = end;
        }
    }

    /// `dspy.Example` via `copyreg.__newobj__` plus its `__dict__` state.
    fn saveExample(p: *Pickler, store: *const Record) Error!void {
        if (p.class) |index| {
            try p.get(index);
        } else {
            try p.saveStr(.structure, "dspy.primitives.example");
            try p.saveStr(.structure, "Example");
            try p.byte(op.stack_global);
            p.class = p.memo_len;
            try p.memoize();
        }
        p.boundary();
        try p.byte(op.empty_tuple);
        p.boundary();
        try p.byte(op.newobj);
        try p.memoize();
        try p.byte(op.empty_dict);
        try p.memoize();
        try p.byte(op.mark);
        try p.saveStr(.structure, "_store");
        try p.saveDict(.key, store);
        p.boundary();
        try p.saveStr(.structure, "_demos");
        try p.byte(op.empty_list);
        try p.memoize();
        p.boundary();
        try p.saveStr(.structure, "_input_keys");
        try p.byte(op.none);
        p.boundary();
        try p.byte(op.setitems);
        p.boundary();
        try p.byte(op.build);
        p.boundary();
    }
};

/// One-character Latin-1 strings and `""` are CPython singletons.
fn isCachedChar(s: []const u8) bool {
    if (s.len <= 1) return true;
    return s.len == 2 and (s[0] == 0xc2 or s[0] == 0xc3) and s[1] & 0xc0 == 0x80;
}

/// `pickle.dumps(tuple(dspy.Example(**d) for d in demos))`; each demo's
/// keys in order (`augmented` first). Caller owns the bytes.
pub fn dumpDemos(gpa: Allocator, demos: []const Record) Error![]u8 {
    var p: Pickler = .{ .gpa = gpa };
    defer p.deinit();
    try p.buf.appendSlice(gpa, &.{ op.proto, 4 });
    if (demos.len == 0) {
        try p.byte(op.empty_tuple);
    } else {
        if (demos.len > 3) try p.byte(op.mark);
        for (demos) |*d| try p.saveExample(d);
        try p.byte(if (demos.len > 3) op.tuple_ else op.tuple1 + @as(u8, @intCast(demos.len - 1)));
        try p.memoize();
    }
    p.boundary();
    try p.byte(op.stop);
    p.commitFrame();
    return p.buf.toOwnedSlice(gpa);
}

/// `dspy.utils.hasher.Hasher.hash(tuple(demos))`: sha256 hex of the pickle.
pub fn hashDemos(gpa: Allocator, demos: []const Record) Error![64]u8 {
    const bytes = try dumpDemos(gpa, demos);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// Index of the trace DSPy keeps among several same-predictor demos.
pub fn pickTrace(gpa: Allocator, demos: []const Record) Error!usize {
    assert(demos.len > 1);
    const hex = try hashDemos(gpa, demos);
    var rng: pyrandom.Random = .initStr(&hex);
    if (rng.random() < 0.5) return rng.randbelow(demos.len - 1);
    return demos.len - 1;
}
