//! Genuine Python builtin identities that cannot be represented by JSON tags.
//!
//! Constructors borrow child values and byte buffers. Set/dictionary boxes own
//! their deduplicated storage, but not its children; a string-only dictionary
//! owns only its ObjectMap storage. Destroy borrowed boxes with Custom.deinit,
//! and borrowed ordinary dictionaries with ObjectMap.deinit, not freeJsonValue.
//! Custom.clone deep-owns every child and buffer and is safe with a general
//! allocator. Owning clone preserves observed container order; it is an
//! ownership operation, not Python copy.deepcopy (which can reorder a set).
//! Borrowed buffers and item/key positions must remain immutable and alive.
//! Raw/serialized projections borrow leaves, like Code/File dumps. Use an arena
//! for mixed-borrowed serialization projections; clone to obtain deep ownership.
//! Raw identity and root Ellipsis projections alias the original box: do not
//! deinitialize those projections independently; clone them to own a copy.
//!
//! Set layout follows 64-bit CPython 3.11 and defaults to PYTHONHASHSEED=0.
//! Explicit u32-seed constructors use CPython's LCG and SipHash13, without global
//! mutable state. CPython address hashes (None/Ellipsis/NaNs) necessarily use
//! native identities, not a different Python process's addresses. Value.float
//! has no object identity token, so separately stored NaNs remain distinct.
//! Opaque classes may supply CustomVTable.python_hash. Without it, class-only
//! equality admits a coherent native hash but unproved Source iteration order;
//! cross-value equality without a hash callback is rejected. Shared/cyclic
//! alias-graph preservation is outside the existing runtime ownership model.

const std = @import("std");
const runtime = @import("runtime_value.zig");
const pyjson = @import("pyjson.zig");
const value_ops = @import("value.zig");
const ownership = @import("inference.zig");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Value = runtime.Value;
const PythonValue = runtime.PythonValue;
const PythonEntry = runtime.PythonEntry;
const Kind = std.meta.Tag(PythonValue);
const SipHash = std.crypto.auth.siphash.SipHash64(1, 3);

pub const Error = runtime.CustomError;

const absent = std.math.maxInt(usize);
const Slot = struct { index: usize = absent, hash: u64 = 0 };
const Box = struct {
    payload: PythonValue,
    vtable: *const runtime.CustomVTable,
    owns_storage: bool = false,
    table: []const Slot = &.{},
    owns_table: bool = false,
    hash_seed: u32 = 0,
};

/// Borrowed metadata, never a JSON-shape heuristic or a converted value.
pub fn view(v: Value) ?PythonValue {
    if (v != .custom) return null;
    const callback = v.custom.vtable.python orelse return null;
    return callback(v.custom.ptr);
}

pub fn bytes(a: Allocator, data: []const u8) Error!Value {
    return boxed(a, .{ .bytes = data }, &bytes_vtable);
}

/// Immutable, allocation-free bytes for comptime annotations. Ordinary owning
/// clones still use the heap-backed bytes protocol and can be freed normally.
pub fn staticBytes(comptime data: []const u8) Value {
    const Storage = struct {
        const vtable: runtime.CustomVTable = blk: {
            var result = bytes_vtable;
            result.clone = cloneStaticBytes;
            result.deinit = deinitStaticBytes;
            break :blk result;
        };
        const box: Box = .{ .payload = .{ .bytes = data }, .vtable = &vtable };
    };
    return .{ .custom = .{ .ptr = &Storage.box, .vtable = &Storage.vtable } };
}

fn cloneStaticBytes(a: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
    var result = try clone(a, ptr);
    result.vtable = &bytes_vtable;
    runtimeBox(result.ptr).vtable = &bytes_vtable;
    return result;
}

fn deinitStaticBytes(_: Allocator, _: *const anyopaque, _: bool) void {}

pub fn tuple(a: Allocator, items: []const Value) Error!Value {
    const hashable = for (items) |item| {
        if (!hashAdmissible(item)) break false;
    } else true;
    return boxed(a, .{ .tuple = items }, if (hashable) &tuple_vtable else &unhashable_tuple_vtable);
}

pub fn set(a: Allocator, items: []const Value) Error!Value {
    return setWithHashSeed(a, items, 0);
}

pub fn frozenSet(a: Allocator, items: []const Value) Error!Value {
    return frozenSetWithHashSeed(a, items, 0);
}

/// Reproduce a CPython process configured with this integer PYTHONHASHSEED.
pub fn setWithHashSeed(a: Allocator, items: []const Value, seed: u32) Error!Value {
    return makeSet(a, items, seed, false);
}

pub fn frozenSetWithHashSeed(a: Allocator, items: []const Value, seed: u32) Error!Value {
    return makeSet(a, items, seed, true);
}

pub fn complex(a: Allocator, real: f64, imag: f64) Error!Value {
    return boxed(a, .{ .complex = .{ .real = real, .imag = imag } }, &complex_vtable);
}

pub fn ellipsis(a: Allocator) Error!Value {
    return boxed(a, .ellipsis, &ellipsis_vtable);
}

/// Equal keys retain the FIRST key object and insertion position, and the LAST
/// value. In particular True, 1, 1.0 and 1+0j denote one dictionary key.
pub fn dictionary(a: Allocator, pairs: []const PythonEntry) Error!Value {
    const all_strings = for (pairs) |pair| {
        if (pair.key != .string) break false;
    } else true;
    if (all_strings) {
        var out: runtime.ObjectMap = .empty;
        errdefer out.deinit(a);
        try out.ensureTotalCapacity(a, pairs.len);
        for (pairs) |pair| {
            const entry = out.getOrPutAssumeCapacity(pair.key.string);
            entry.value_ptr.* = pair.value;
        }
        return .{ .object = out };
    }

    var entries: std.ArrayList(PythonEntry) = .empty;
    defer entries.deinit(a);
    var builder: Table = .{ .allocator = a };
    defer builder.deinit();
    var unicode_keys = true;
    for (pairs) |pair| {
        if (!hashAdmissible(pair.key)) return error.ParseValue;
        const hash = try hashValue(pair.key, 0);
        if (unicode_keys and pair.key != .string) {
            const minimum = std.math.mul(usize, builder.used, 3) catch return error.OutOfMemory;
            try builder.resizeDictionary(minimum);
            unicode_keys = false;
        }
        const lookup = builder.findEntries(entries.items, pair.key, hash);
        if (lookup.found) {
            entries.items[builder.slots[lookup.slot].index].value = pair.value;
        } else {
            try entries.append(a, pair);
            try builder.add(lookup.slot, entries.items.len - 1, hash, true);
        }
    }
    // A hashable foreign custom value may compare equal to a retained string.
    // The retained key, not the last supplied key, decides the dictionary tag.
    const strings_survive = for (entries.items) |entry| {
        if (entry.key != .string) break false;
    } else true;
    if (strings_survive) return dictionary(a, entries.items);

    const data = try entries.toOwnedSlice(a);
    errdefer a.free(data);
    const result = try boxed(a, .{ .dictionary = data }, &dictionary_vtable);
    const box = runtimeBox(result.custom.ptr);
    box.owns_storage = true;
    box.table = builder.take();
    box.owns_table = true;
    return result;
}

fn boxed(a: Allocator, payload: PythonValue, vtable: *const runtime.CustomVTable) Allocator.Error!Value {
    const box = try a.create(Box);
    box.* = .{ .payload = payload, .vtable = vtable };
    return .{ .custom = .{ .ptr = box, .vtable = vtable } };
}

fn runtimeBox(ptr: *const anyopaque) *Box {
    return @ptrCast(@alignCast(@constCast(ptr)));
}

fn ownBox(v: Value) ?*const Box {
    if (v != .custom or v.custom.vtable.python != pythonView) return null;
    return runtimeBox(v.custom.ptr);
}

fn makeVTable(comptime kind: Kind, comptime hashable: bool) runtime.CustomVTable {
    const name = switch (kind) {
        .bytes => "bytes",
        .tuple => "tuple",
        .set => "set",
        .frozen_set => "frozenset",
        .complex => "complex",
        .ellipsis => "ellipsis",
        .dictionary => "dict",
    };
    return .{
        .type_name = "python.builtins." ++ name,
        .model_name = name,
        .hashable = hashable,
        .bool_value = truthy,
        .python = pythonView,
        .raw_value = rawValue,
        .serialized = serialized,
        .write_str = writeRuntimeStr,
        .write_repr = writeRuntimeRepr,
        .write_json = writeRuntimeJson,
        .write_serialized_json = writeSerializedJson,
        .clone = clone,
        .deinit = deinit,
        .eql = equal,
        .eql_value = equalValue,
    };
}

const bytes_vtable = makeVTable(.bytes, true);
const tuple_vtable = makeVTable(.tuple, true);
const unhashable_tuple_vtable = makeVTable(.tuple, false);
const set_vtable = makeVTable(.set, false);
const frozen_set_vtable = makeVTable(.frozen_set, true);
const complex_vtable = makeVTable(.complex, true);
const ellipsis_vtable = makeVTable(.ellipsis, true);
const dictionary_vtable = makeVTable(.dictionary, false);

fn pythonView(ptr: *const anyopaque) PythonValue {
    return runtimeBox(ptr).payload;
}

fn truthy(ptr: *const anyopaque) bool {
    return switch (pythonView(ptr)) {
        .bytes => |data| data.len != 0,
        .tuple, .set, .frozen_set => |items| items.len != 0,
        .dictionary => |pairs| pairs.len != 0,
        .complex => |c| c.real != 0 or c.imag != 0,
        .ellipsis => true,
    };
}

fn hashAdmissible(v: Value) bool {
    return switch (v) {
        .array, .object => false,
        .custom => |custom| if (view(v)) |python| switch (python) {
            .set, .dictionary => false,
            .tuple => |items| for (items) |item| {
                if (!hashAdmissible(item)) break false;
            } else true,
            else => true,
        } else custom.vtable.hashable,
        else => true,
    };
}

/// CPython's distinct no-deletion set/dict probe and resize algorithms. Retained
/// indices permit allocation-free equality. Temporary dict hashes preserve
/// entry-order reindexing during growth without recomputing caller callbacks.
const Table = struct {
    allocator: Allocator,
    slots: []Slot = &.{},
    used: usize = 0,
    entry_hashes: std.ArrayList(u64) = .empty,

    fn deinit(self: *Table) void {
        self.allocator.free(self.slots);
        self.entry_hashes.deinit(self.allocator);
    }

    fn take(self: *Table) []Slot {
        const out = self.slots;
        self.slots = &.{};
        return out;
    }

    const Lookup = struct { slot: usize, found: bool };

    fn findValues(self: *const Table, items: []const Value, key: Value, hash: u64) Lookup {
        return self.find(items, key, hash, false);
    }

    fn findEntries(self: *const Table, pairs: []const PythonEntry, key: Value, hash: u64) Lookup {
        return self.find(pairs, key, hash, true);
    }

    fn find(self: *const Table, items: anytype, key: Value, hash: u64, comptime entries: bool) Lookup {
        if (self.slots.len == 0) return .{ .slot = 0, .found = false };
        const mask = self.slots.len - 1;
        var index: usize = @as(usize, @truncate(hash)) & mask;
        var perturb = hash;
        while (true) {
            const probes: usize = if (entries) 0 else if (index + 9 <= mask) 9 else 0;
            for (0..probes + 1) |offset| {
                const slot = self.slots[index + offset];
                if (slot.index == absent) return .{ .slot = index + offset, .found = false };
                const prior = if (entries) items[slot.index].key else items[slot.index];
                if (slot.hash == hash and keyEqual(prior, key))
                    return .{ .slot = index + offset, .found = true };
            }
            perturb >>= 5;
            index = (index *% 5 +% 1 +% @as(usize, @truncate(perturb))) & mask;
        }
    }

    fn add(self: *Table, slot: usize, index: usize, hash: u64, comptime entries: bool) Allocator.Error!void {
        if (entries) {
            std.debug.assert(index == self.used);
            std.debug.assert(self.entry_hashes.items.len == self.used);
            const resized = self.slots.len == 0 or self.used >= self.slots.len * 2 / 3;
            if (resized) {
                const minimum = std.math.mul(usize, self.used, 3) catch return error.OutOfMemory;
                try self.resizeDictionary(minimum);
            }
            try self.entry_hashes.append(self.allocator, hash);
            if (resized) {
                self.insertClean(.{ .index = index, .hash = hash }, true);
            } else {
                self.slots[slot] = .{ .index = index, .hash = hash };
            }
            self.used += 1;
            return;
        }
        if (self.slots.len == 0) {
            self.slots = try self.allocator.alloc(Slot, 8);
            @memset(self.slots, .{});
            self.insertClean(.{ .index = index, .hash = hash }, false);
        } else {
            self.slots[slot] = .{ .index = index, .hash = hash };
        }
        self.used += 1;
        if (self.used * 5 < (self.slots.len - 1) * 3) return;
        const multiplier: usize = if (self.used > 50_000) 2 else 4;
        const min_used = std.math.mul(usize, self.used, multiplier) catch return error.OutOfMemory;
        var new_size: usize = 8;
        while (new_size <= min_used)
            new_size = std.math.mul(usize, new_size, 2) catch return error.OutOfMemory;
        const replacement = try self.allocator.alloc(Slot, new_size);
        @memset(replacement, .{});
        const old = self.slots;
        self.slots = replacement;
        for (old) |entry| if (entry.index != absent) self.insertClean(entry, false);
        self.allocator.free(old);
    }

    fn resizeDictionary(self: *Table, minimum: usize) Allocator.Error!void {
        var size: usize = 8;
        while (size < minimum)
            size = std.math.mul(usize, size, 2) catch return error.OutOfMemory;
        // The native representation already stores general entries. A Unicode
        // kind conversion at the same capacity needs no new native storage.
        if (size == self.slots.len) return;
        const replacement = try self.allocator.alloc(Slot, size);
        @memset(replacement, .{});
        const old = self.slots;
        self.slots = replacement;
        for (self.entry_hashes.items, 0..) |hash, index|
            self.insertClean(.{ .index = index, .hash = hash }, true);
        self.allocator.free(old);
    }

    fn insertClean(self: *Table, entry: Slot, comptime entries: bool) void {
        const mask = self.slots.len - 1;
        var index: usize = @as(usize, @truncate(entry.hash)) & mask;
        var perturb = entry.hash;
        while (true) {
            const probes: usize = if (entries) 0 else if (index + 9 <= mask) 9 else 0;
            for (0..probes + 1) |offset| {
                if (self.slots[index + offset].index == absent) {
                    self.slots[index + offset] = entry;
                    return;
                }
            }
            perturb >>= 5;
            index = (index *% 5 +% 1 +% @as(usize, @truncate(perturb))) & mask;
        }
    }
};

fn makeSet(a: Allocator, items: []const Value, seed: u32, frozen: bool) Error!Value {
    var builder: Table = .{ .allocator = a };
    defer builder.deinit();
    for (items, 0..) |item, index| {
        if (!hashAdmissible(item)) return error.ParseValue;
        const hash = try hashValue(item, seed);
        const lookup = builder.findValues(items, item, hash);
        if (!lookup.found) try builder.add(lookup.slot, index, hash, false);
    }
    const data = try a.alloc(Value, builder.used);
    errdefer a.free(data);
    var count: usize = 0;
    for (builder.slots) |*slot| {
        if (slot.index == absent) continue;
        data[count] = items[slot.index];
        slot.index = count;
        count += 1;
    }
    const result = try boxed(a, if (frozen) .{ .frozen_set = data } else .{ .set = data }, if (frozen) &frozen_set_vtable else &set_vtable);
    const box = runtimeBox(result.custom.ptr);
    box.owns_storage = true;
    box.hash_seed = seed;
    box.table = builder.take();
    box.owns_table = true;
    return result;
}

const hash_modulus: u64 = (1 << 61) - 1;
const none_identity: u8 = 71;
const ellipsis_identity: u8 = 93;

fn pointerHash(ptr: *const anyopaque) u64 {
    return validHash(std.math.rotr(u64, @as(u64, @intFromPtr(ptr)), 4));
}

fn validHash(hash: u64) u64 {
    return if (hash == std.math.maxInt(u64)) hash - 1 else hash;
}

fn signedHash(magnitude: u64, negative: bool) u64 {
    return validHash(if (negative) 0 -% magnitude else magnitude);
}

fn floatHash(f: f64, identity: ?*const anyopaque) u64 {
    if (std.math.isInf(f)) return signedHash(314159, f < 0);
    if (std.math.isNan(f)) {
        // Embedded floats cannot carry Python's distinct float-object pointers.
        // A coherent native hash still keeps unequal NaNs as separate entries.
        return if (identity) |ptr| pointerHash(ptr) else 0;
    }
    const bits: u64 = @bitCast(f);
    const exponent: i32 = @intCast((bits >> 52) & 0x7ff);
    const significand = (bits & 0x000f_ffff_ffff_ffff) | if (exponent == 0) @as(u64, 0) else 0x0010_0000_0000_0000;
    const shift: u6 = @intCast(@mod(if (exponent == 0) @as(i32, -1074) else exponent - 1075, 61));
    const magnitude = ((significand << shift) & hash_modulus) | (significand >> @as(u6, @intCast(61 - @as(u7, shift))));
    return signedHash(magnitude, (bits >> 63) != 0);
}

fn hashKey(seed: u32) [16]u8 {
    var key: [16]u8 = @splat(0);
    if (seed != 0) {
        var state = seed;
        for (&key) |*byte| {
            state = state *% 214013 +% 2531011;
            byte.* = @truncate(state >> 16);
        }
    }
    return key;
}

fn bytesHash(data: []const u8, seed: u32) u64 {
    if (data.len == 0) return 0;
    const key = hashKey(seed);
    return validHash(SipHash.toInt(data, &key));
}

fn stringHash(text: []const u8, seed: u32) error{ParseValue}!u64 {
    if (text.len == 0) return 0;
    // CPython hashes the compact Unicode storage, not UTF-8. Its kind is one,
    // two or four native-endian bytes per code point, including lone surrogates.
    var maximum: u21 = 0;
    var index: usize = 0;
    while (index < text.len) {
        const point = try nextPoint(text, &index);
        maximum = @max(maximum, point);
    }
    if (maximum < 128) return bytesHash(text, seed);
    const width: usize = if (maximum <= 0xff) 1 else if (maximum <= 0xffff) 2 else 4;
    const key = hashKey(seed);
    var hash = SipHash.init(&key);
    index = 0;
    while (index < text.len) {
        const point: u32 = try nextPoint(text, &index);
        var encoded: [4]u8 = undefined;
        std.mem.writeInt(u32, &encoded, point, .little);
        hash.update(encoded[0..width]);
    }
    return validHash(hash.finalInt());
}

fn nextPoint(text: []const u8, index: *usize) error{ParseValue}!u21 {
    const width = std.unicode.utf8ByteSequenceLength(text[index.*]) catch return error.ParseValue;
    if (width > text.len - index.*) return error.ParseValue;
    const result = std.unicode.wtf8Decode(text[index.*..][0..width]) catch return error.ParseValue;
    index.* += width;
    return result;
}

fn tupleHash(items: []const Value, seed: u32) error{ParseValue}!u64 {
    var acc: u64 = 2870177450012600261;
    for (items) |item| {
        acc +%= (try hashValue(item, seed)) *% 14029467366897019727;
        acc = std.math.rotl(u64, acc, 31);
        acc *%= 11400714785074694791;
    }
    acc +%= @as(u64, @intCast(items.len)) ^ (2870177450012600261 ^ 3527539);
    return if (acc == std.math.maxInt(u64)) 1546275796 else acc;
}

fn hashValue(v: Value, seed: u32) error{ParseValue}!u64 {
    if (view(v)) |python| return switch (python) {
        .bytes => |data| bytesHash(data, seed),
        .tuple => |items| try tupleHash(items, seed),
        .frozen_set => |items| blk: {
            var acc: u64 = 0;
            for (items) |item| {
                const hash = try hashValue(item, seed);
                acc ^= ((hash ^ 89869747) ^ (hash << 16)) *% 3644798167;
            }
            acc ^= (@as(u64, @intCast(items.len)) +% 1) *% 1927868237;
            acc ^= (acc >> 11) ^ (acc >> 25);
            acc = acc *% 69069 +% 907133923;
            break :blk if (acc == std.math.maxInt(u64)) 590923713 else acc;
        },
        .complex => |c| validHash(floatHash(c.real, v.custom.ptr) +% 1000003 *% floatHash(c.imag, v.custom.ptr)),
        .ellipsis => pointerHash(&ellipsis_identity),
        .set, .dictionary => error.ParseValue,
    };
    return switch (v) {
        .null => pointerHash(&none_identity),
        .bool => |b| @intFromBool(b),
        .integer => |i| signedHash(@abs(i) % hash_modulus, i < 0),
        .float => |f| floatHash(f, null),
        .string => |text| stringHash(text, seed),
        .number_string => |text| blk: {
            if (!std.json.isNumberFormattedLikeAnInteger(text))
                break :blk floatHash(std.fmt.parseFloat(f64, text) catch return error.ParseValue, null);
            if (text.len == 0) return error.ParseValue;
            const negative = text[0] == '-';
            const digits = text[@intFromBool(negative)..];
            if (digits.len == 0) return error.ParseValue;
            var acc: u64 = 0;
            for (digits) |digit| {
                if (!std.ascii.isDigit(digit)) return error.ParseValue;
                acc = @intCast((@as(u128, acc) * 10 + (digit - '0')) % hash_modulus);
            }
            break :blk signedHash(acc, negative);
        },
        .custom => |custom| blk: {
            if (!custom.vtable.hashable) break :blk error.ParseValue;
            if (custom.vtable.python_hash) |hash|
                break :blk validHash(@bitCast(try hash(custom.ptr, seed)));
            if (custom.vtable.enumeration) |inspect|
                break :blk try stringHash(inspect(custom.ptr).member_name, seed);
            // Pydantic's frozen hash is hash(itemgetter(*model_fields)(dict)).
            // Read actual native fields without allocating a raw model dump.
            if (custom.get(@import("file.zig").File)) |file| {
                var fields: [3]Value = undefined;
                inline for (.{ "file_data", "file_id", "filename" }, 0..) |name, index|
                    fields[index] = if (@field(file.*, name)) |text| .{ .string = text } else .null;
                break :blk try tupleHash(&fields, seed);
            }
            if (custom.get(@import("bounding_box.zig").BoundingBox)) |bounds| {
                var fields: [4]Value = undefined;
                inline for (.{ "top", "bottom", "left", "right" }, 0..) |name, index|
                    fields[index] = .{ .float = @field(bounds.*, name) };
                break :blk try tupleHash(&fields, seed);
            }
            // A class-only fallback is coherent only if equality cannot cross
            // value/class boundaries. It is not a claim about Source hash order.
            if (custom.vtable.eql_value != null) break :blk error.ParseValue;
            break :blk bytesHash(custom.vtable.type_name, seed);
        },
        .array, .object => error.ParseValue,
    };
}

fn keyEqual(a: Value, b: Value) bool {
    // PyObject_RichCompareBool performs this identity shortcut for membership
    // and container elements, even when an object's own equality is false.
    if (a == .custom and b == .custom and a.custom.ptr == b.custom.ptr) return true;
    if (ownBox(a)) |box| return equalValue(box, b);
    if (ownBox(b)) |box| return equalValue(box, a);
    return value_ops.pyEqual(a, b);
}

fn equal(left: *const anyopaque, right: *const anyopaque) bool {
    const other = runtimeBox(right);
    return equalValue(left, .{ .custom = .{ .ptr = other, .vtable = other.vtable } });
}

fn equalValue(ptr: *const anyopaque, other: Value) bool {
    const self = runtimeBox(ptr);
    if (other == .custom and other.custom.ptr == ptr and self.payload != .complex) return true;
    const b = view(other);
    return switch (self.payload) {
        .bytes => |data| if (b) |python| python == .bytes and std.mem.eql(u8, data, python.bytes) else false,
        .tuple => |items| blk: {
            const python = b orelse break :blk false;
            if (python != .tuple or items.len != python.tuple.len) break :blk false;
            for (items, python.tuple) |a_item, b_item| if (!keyEqual(a_item, b_item)) break :blk false;
            break :blk true;
        },
        .set, .frozen_set => |items| blk: {
            const python = b orelse break :blk false;
            const others = switch (python) {
                .set, .frozen_set => |values| values,
                else => break :blk false,
            };
            if (items.len != others.len) break :blk false;
            if (ownBox(other)) |owner| {
                const table: Table = .{ .allocator = undefined, .slots = @constCast(owner.table) };
                for (items) |item| {
                    // CPython looks each LEFT key up in the RIGHT set. Operand
                    // direction remains observable for asymmetric custom ==.
                    const hash = hashValue(item, owner.hash_seed) catch break :blk false;
                    if (!table.findValues(others, item, hash).found) break :blk false;
                }
            } else {
                for (items) |item| {
                    const hash = hashValue(item, 0) catch break :blk false;
                    var found = false;
                    for (others) |candidate| {
                        const candidate_hash = hashValue(candidate, 0) catch break :blk false;
                        if (candidate_hash == hash and keyEqual(candidate, item)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) break :blk false;
                }
            }
            break :blk true;
        },
        .complex => |c| blk: {
            if (b) |python| {
                if (python != .complex) break :blk false;
                break :blk c.real == python.complex.real and c.imag == python.complex.imag;
            }
            if (c.imag != 0) break :blk false;
            break :blk value_ops.pyEqual(.{ .float = c.real }, other);
        },
        .ellipsis => b != null and b.? == .ellipsis,
        .dictionary => |pairs| blk: {
            if (other == .object) {
                if (pairs.len != other.object.count()) break :blk false;
                const table: Table = .{ .allocator = undefined, .slots = @constCast(self.table) };
                for (other.object.keys(), other.object.values()) |key, child| {
                    // A retained custom key may compare equal to str without
                    // BEING str. Lookup must respect its supplied Python hash
                    // and equality, just as an arbitrary-key dict lookup does.
                    const candidate: Value = .{ .string = key };
                    const hash = hashValue(candidate, 0) catch break :blk false;
                    const lookup = table.findEntries(pairs, candidate, hash);
                    if (!lookup.found or !keyEqual(pairs[self.table[lookup.slot].index].value, child)) break :blk false;
                }
                break :blk true;
            }
            const python = b orelse break :blk false;
            if (python != .dictionary or pairs.len != python.dictionary.len) break :blk false;
            for (pairs) |pair| {
                const hash = hashValue(pair.key, 0) catch break :blk false;
                var match_index: usize = absent;
                if (ownBox(other)) |owner| {
                    const table: Table = .{ .allocator = undefined, .slots = @constCast(owner.table) };
                    const lookup = table.findEntries(python.dictionary, pair.key, hash);
                    if (lookup.found) match_index = owner.table[lookup.slot].index;
                } else {
                    for (python.dictionary, 0..) |candidate, index| {
                        const candidate_hash = hashValue(candidate.key, 0) catch break :blk false;
                        if (candidate_hash == hash and keyEqual(candidate.key, pair.key)) {
                            match_index = index;
                            break;
                        }
                    }
                }
                if (match_index == absent or !keyEqual(pair.value, python.dictionary[match_index].value)) break :blk false;
            }
            break :blk true;
        },
    };
}

fn rawValue(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
    return .{ .custom = .{ .ptr = ptr, .vtable = runtimeBox(ptr).vtable } };
}

fn cloneItems(a: Allocator, items: []const Value) Allocator.Error![]Value {
    const out = try a.alloc(Value, items.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |item| ownership.freeJsonValue(a, item);
        a.free(out);
    }
    for (items, out) |item, *copy| {
        copy.* = try ownership.cloneJsonValue(a, item);
        initialized += 1;
    }
    return out;
}

fn clone(a: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
    const source = runtimeBox(ptr);
    const copy = try a.create(Box);
    errdefer a.destroy(copy);
    copy.* = .{ .payload = source.payload, .vtable = source.vtable, .hash_seed = source.hash_seed };
    switch (source.payload) {
        .bytes => |data| copy.payload = .{ .bytes = try a.dupe(u8, data) },
        .tuple, .set, .frozen_set => |items| {
            const children = try cloneItems(a, items);
            errdefer {
                for (children) |item| ownership.freeJsonValue(a, item);
                a.free(children);
            }
            switch (source.payload) {
                .tuple => copy.payload = .{ .tuple = children },
                .set => copy.payload = .{ .set = children },
                .frozen_set => copy.payload = .{ .frozen_set = children },
                else => unreachable,
            }
            copy.owns_storage = true;
            if (source.table.len != 0) {
                copy.table = try a.dupe(Slot, source.table);
                copy.owns_table = true;
            }
        },
        .dictionary => |pairs| {
            const entries = try a.alloc(PythonEntry, pairs.len);
            var initialized: usize = 0;
            errdefer {
                for (entries[0..initialized]) |entry| {
                    ownership.freeJsonValue(a, entry.key);
                    ownership.freeJsonValue(a, entry.value);
                }
                a.free(entries);
            }
            for (pairs, entries) |entry, *destination| {
                const key = try ownership.cloneJsonValue(a, entry.key);
                errdefer ownership.freeJsonValue(a, key);
                destination.* = .{ .key = key, .value = try ownership.cloneJsonValue(a, entry.value) };
                initialized += 1;
            }
            copy.payload = .{ .dictionary = entries };
            copy.owns_storage = true;
            if (source.table.len != 0) {
                copy.table = try a.dupe(Slot, source.table);
                copy.owns_table = true;
            }
        },
        .complex, .ellipsis => {},
    }
    return .{ .ptr = copy, .vtable = copy.vtable, .owns_data = true };
}

fn deinit(a: Allocator, ptr: *const anyopaque, owns_data: bool) void {
    const box = runtimeBox(ptr);
    switch (box.payload) {
        .bytes => |data| if (owns_data) a.free(data),
        .tuple, .set, .frozen_set => |items| {
            if (owns_data) for (items) |item| ownership.freeJsonValue(a, item);
            if (box.owns_storage) a.free(items);
        },
        .dictionary => |pairs| {
            if (owns_data) for (pairs) |entry| {
                ownership.freeJsonValue(a, entry.key);
                ownership.freeJsonValue(a, entry.value);
            };
            if (box.owns_storage) a.free(pairs);
        },
        .complex, .ellipsis => {},
    }
    if (box.owns_table) a.free(box.table);
    a.destroy(box);
}

fn writeRuntimeStr(w: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
    try writePythonRepr(w, runtimeBox(ptr).payload, 0);
}

fn writeRuntimeRepr(w: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
    try writePythonRepr(w, runtimeBox(ptr).payload, depth);
}

fn integerTextTooLong(text: []const u8) bool {
    return text.len > 4300 and std.json.isNumberFormattedLikeAnInteger(text) and
        text.len - @as(usize, @intFromBool(text[0] == '-')) > 4300;
}

fn writeValueRepr(w: *Writer, v: Value, depth: usize) runtime.CustomWriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (view(v)) |python| return writePythonRepr(w, python, depth);
    switch (v) {
        .array => |array| {
            try w.writeByte('[');
            for (array.items, 0..) |item, index| {
                if (index != 0) try w.writeAll(", ");
                try writeValueRepr(w, item, depth + 1);
            }
            try w.writeByte(']');
        },
        .object => |object| {
            try w.writeByte('{');
            for (object.keys(), object.values(), 0..) |key, item, index| {
                if (index != 0) try w.writeAll(", ");
                try pyjson.writeReprString(w, key);
                try w.writeAll(": ");
                try writeValueRepr(w, item, depth + 1);
            }
            try w.writeByte('}');
        },
        .number_string => |text| {
            if (integerTextTooLong(text)) return error.ParseValue;
            try pyjson.writeRepr(w, v, @intCast(depth));
        },
        else => try pyjson.writeRepr(w, v, @intCast(depth)),
    }
}

fn writePythonRepr(w: *Writer, python: PythonValue, depth: usize) runtime.CustomWriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    switch (python) {
        .bytes => |data| try writeBytesRepr(w, data),
        .tuple => |items| {
            try w.writeByte('(');
            for (items, 0..) |item, index| {
                if (index != 0) try w.writeAll(", ");
                try writeValueRepr(w, item, depth + 1);
            }
            if (items.len == 1) try w.writeByte(',');
            try w.writeByte(')');
        },
        .set, .frozen_set => |items| {
            const frozen = python == .frozen_set;
            if (items.len == 0) return w.writeAll(if (frozen) "frozenset()" else "set()");
            if (frozen) try w.writeAll("frozenset(");
            try w.writeByte('{');
            for (items, 0..) |item, index| {
                if (index != 0) try w.writeAll(", ");
                try writeValueRepr(w, item, depth + 1);
            }
            try w.writeAll(if (frozen) "})" else "}");
        },
        .complex => |c| {
            const real_zero = c.real == 0 and !std.math.signbit(c.real);
            if (!real_zero) {
                try w.writeByte('(');
                try writeComplexComponent(w, c.real, false);
                if (std.math.isNan(c.imag) or !std.math.signbit(c.imag)) try w.writeByte('+');
            }
            try writeComplexComponent(w, c.imag, false);
            try w.writeAll(if (real_zero) "j" else "j)");
        },
        .ellipsis => try w.writeAll("Ellipsis"),
        .dictionary => |pairs| {
            try w.writeByte('{');
            for (pairs, 0..) |entry, index| {
                if (index != 0) try w.writeAll(", ");
                try writeValueRepr(w, entry.key, depth + 1);
                try w.writeAll(": ");
                try writeValueRepr(w, entry.value, depth + 1);
            }
            try w.writeByte('}');
        },
    }
}

fn writeBytesRepr(w: *Writer, data: []const u8) Writer.Error!void {
    const quote: u8 = if (std.mem.indexOfScalar(u8, data, '\'') != null and std.mem.indexOfScalar(u8, data, '"') == null) '"' else '\'';
    try w.writeByte('b');
    try w.writeByte(quote);
    const hex = "0123456789abcdef";
    for (data) |byte| {
        if (byte == quote or byte == '\\') {
            try w.writeByte('\\');
            try w.writeByte(byte);
        } else switch (byte) {
            '\t' => try w.writeAll("\\t"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            0x20...0x7e => try w.writeByte(byte),
            else => try w.writeAll(&.{ '\\', 'x', hex[byte >> 4], hex[byte & 15] }),
        }
    }
    try w.writeByte(quote);
}

/// Complex repr uses float's shortest digits without its forced trailing .0.
/// Pydantic JSON mode instead uses Rust Display: positional digits and NaN.
fn writeComplexComponent(w: *Writer, f: f64, json_mode: bool) Writer.Error!void {
    if (std.math.isNan(f)) return w.writeAll(if (json_mode) "NaN" else "nan");
    var buffer: [64]u8 = undefined;
    var fixed: Writer = .fixed(&buffer);
    try pyjson.writeFloat(&fixed, f, .python);
    var text = fixed.buffered();
    if (std.mem.endsWith(u8, text, ".0")) text = text[0 .. text.len - 2];
    if (!json_mode or std.math.isInf(f)) return w.writeAll(text);
    const e_index = std.mem.indexOfScalar(u8, text, 'e') orelse return w.writeAll(text);
    var mantissa = text[0..e_index];
    if (mantissa[0] == '-') {
        try w.writeByte('-');
        mantissa = mantissa[1..];
    }
    const exponent = std.fmt.parseInt(i32, text[e_index + 1 ..], 10) catch unreachable;
    var digits_buffer: [32]u8 = undefined;
    var count: usize = 0;
    for (mantissa) |byte| if (byte != '.') {
        digits_buffer[count] = byte;
        count += 1;
    };
    const digits = digits_buffer[0..count];
    const decimal_point = exponent + 1;
    if (decimal_point <= 0) {
        try w.writeAll("0.");
        try w.splatByteAll('0', @intCast(-decimal_point));
        try w.writeAll(digits);
    } else if (decimal_point >= @as(i32, @intCast(count))) {
        try w.writeAll(digits);
        try w.splatByteAll('0', @as(usize, @intCast(decimal_point)) - count);
    } else {
        const point: usize = @intCast(decimal_point);
        try w.writeAll(digits[0..point]);
        try w.writeByte('.');
        try w.writeAll(digits[point..]);
    }
}

fn writeComplexJsonText(w: *Writer, c: @FieldType(PythonValue, "complex")) Writer.Error!void {
    if (c.real != 0) {
        try writeComplexComponent(w, c.real, true);
        if (std.math.isNan(c.imag) or !std.math.signbit(c.imag)) try w.writeByte('+');
    }
    try writeComplexComponent(w, c.imag, true);
    try w.writeByte('j');
}

/// Complete strict Pydantic JSON-mode projection. Only the shared top-level
/// boundary handles PythonSerializeFallback, retaining the ORIGINAL input for
/// str() rather than replacing an individual invalid child with text.
fn serialized(a: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
    if (depth >= pyjson.max_depth) return error.ParseValue;
    const self = runtimeBox(ptr);
    return switch (self.payload) {
        .bytes => |data| if (std.unicode.utf8ValidateSlice(data))
            .{ .string = data }
        else
            error.PythonSerializeFallback,
        .tuple, .set, .frozen_set => |items| blk: {
            var out: runtime.Array = .init(a);
            errdefer out.deinit();
            try out.ensureTotalCapacity(items.len);
            for (items) |item|
                out.appendAssumeCapacity(try value_ops.serializedValueDepth(a, item, depth + 1));
            break :blk .{ .array = out };
        },
        .complex => |c| blk: {
            var out: Writer.Allocating = .init(a);
            defer out.deinit();
            writeComplexJsonText(&out.writer, c) catch return error.OutOfMemory;
            break :blk .{ .string = try out.toOwnedSlice() };
        },
        .ellipsis => .{ .custom = .{ .ptr = ptr, .vtable = self.vtable } },
        .dictionary => |pairs| try serializedDictionary(a, pairs, depth),
    };
}

fn serializedDictionary(a: Allocator, pairs: []const PythonEntry, depth: u32) runtime.SerializationError!Value {
    var out: runtime.ObjectMap = .empty;
    errdefer out.deinit(a);
    var owned_keys: std.ArrayList([]const u8) = .empty;
    defer owned_keys.deinit(a);
    errdefer for (owned_keys.items) |key| a.free(key);
    try out.ensureTotalCapacity(a, pairs.len);
    for (pairs) |pair| {
        var owned = false;
        const key = if (pair.key == .string) pair.key.string else blk: {
            var buffer: Writer.Allocating = .init(a);
            defer buffer.deinit();
            writeKeyText(&buffer.writer, pair.key) catch |err| return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                error.ParseValue => error.ParseValue,
                error.PythonSerializeFallback, error.JsonTooDeep => error.PythonSerializeFallback,
            };
            owned = true;
            break :blk try buffer.toOwnedSlice();
        };
        const entry = out.getOrPutAssumeCapacity(key);
        if (entry.found_existing) {
            if (owned) a.free(key);
        } else if (owned) {
            owned_keys.append(a, key) catch {
                a.free(key);
                return error.OutOfMemory;
            };
        }
        // Convert even a value that a later COERCED key will overwrite. Source
        // observes errors in insertion order before the final key collision.
        entry.value_ptr.* = try value_ops.serializedValueDepth(a, pair.value, depth + 1);
    }
    return .{ .object = out };
}

const KeyError = error{ PythonSerializeFallback, JsonTooDeep, ParseValue };

/// Opaque callbacks expose push-based text, not a pull iterator. Builtin keys
/// never use this adapter. It reads a bounded window without heap allocation;
/// only an opaque class key larger than a window needs callback re-rendering.
const TextWindowWriter = struct {
    destination: []u8,
    skip: usize,
    written: usize = 0,
    total: usize = 0,
    writer: Writer = .{ .vtable = &.{ .drain = drain }, .buffer = &.{} },

    fn consume(self: *TextWindowWriter, bytes_: []const u8) Writer.Error!void {
        self.total = std.math.add(usize, self.total, bytes_.len) catch return error.WriteFailed;
        const skipped = @min(self.skip, bytes_.len);
        self.skip -= skipped;
        const available = bytes_[skipped..];
        const count = @min(self.destination.len - self.written, available.len);
        @memcpy(self.destination[self.written..][0..count], available[0..count]);
        self.written += count;
    }

    fn drain(writer: *Writer, data: []const []const u8, splat: usize) Writer.Error!usize {
        const self: *TextWindowWriter = @alignCast(@fieldParentPtr("writer", writer));
        const before = self.total;
        for (data[0 .. data.len - 1]) |part| try self.consume(part);
        for (0..splat) |_| try self.consume(data[data.len - 1]);
        return self.total - before;
    }
};

/// Pydantic flattens tuple keys with literal commas (no brackets or quoting).
/// The iterator permits exact collision comparison and streaming escaping of
/// arbitrarily long keys without allocating a temporary concatenated string.
const KeyIterator = struct {
    pending: ?Value,
    frames: [pyjson.max_depth]Frame = undefined,
    depth: usize = 0,
    buffer: [768]u8 = undefined,
    opaque_key: ?runtime.Custom = null,
    opaque_offset: usize = 0,

    const Frame = struct { items: []const Value, next: usize };

    fn init(key: Value) KeyIterator {
        return .{ .pending = key };
    }

    fn next(self: *KeyIterator) KeyError!?[]const u8 {
        while (true) {
            if (self.opaque_key) |custom| {
                var window: TextWindowWriter = .{
                    .destination = &self.buffer,
                    .skip = self.opaque_offset,
                };
                // Current frozen native models match Source json_key: File's
                // text serializer and BoundingBox's fallback=str. Preserve a
                // TypeFormatError from File even when it is a dictionary key.
                custom.writeStr(&window.writer) catch |err| return switch (err) {
                    error.ParseValue, error.WriteFailed => error.ParseValue,
                    error.JsonTooDeep => error.JsonTooDeep,
                };
                self.opaque_offset += window.written;
                if (self.opaque_offset >= window.total) self.opaque_key = null;
                if (window.written == 0) continue;
                return self.buffer[0..window.written];
            }
            if (self.pending) |key| {
                self.pending = null;
                if (view(key)) |python| switch (python) {
                    .tuple => |items| {
                        if (items.len == 0) continue;
                        if (self.depth == self.frames.len) return error.JsonTooDeep;
                        self.frames[self.depth] = .{ .items = items, .next = 1 };
                        self.depth += 1;
                        self.pending = items[0];
                        continue;
                    },
                    .bytes => |data| return if (std.unicode.utf8ValidateSlice(data)) data else error.PythonSerializeFallback,
                    .complex => |c| {
                        var fixed: Writer = .fixed(&self.buffer);
                        writeComplexJsonText(&fixed, c) catch unreachable;
                        return fixed.buffered();
                    },
                    .ellipsis => return "Ellipsis",
                    .set, .frozen_set, .dictionary => return error.PythonSerializeFallback,
                };
                return switch (key) {
                    .string => |text| text,
                    .null => "None",
                    .bool => |b| if (b) "true" else "false",
                    .integer => |i| std.fmt.bufPrint(&self.buffer, "{d}", .{i}) catch unreachable,
                    .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text)) blk: {
                        if (integerTextTooLong(text)) return error.PythonSerializeFallback;
                        break :blk text;
                    } else self.floatText(std.fmt.parseFloat(f64, text) catch return error.PythonSerializeFallback),
                    .float => |f| self.floatText(f),
                    .custom => |custom| {
                        if (custom.vtable.enumeration) |inspect| {
                            self.pending = inspect(custom.ptr).value;
                        } else {
                            self.opaque_key = custom;
                            self.opaque_offset = 0;
                        }
                        continue;
                    },
                    .array, .object => error.PythonSerializeFallback,
                };
            }
            if (self.depth == 0) return null;
            const frame = &self.frames[self.depth - 1];
            if (frame.next == frame.items.len) {
                self.depth -= 1;
                continue;
            }
            self.pending = frame.items[frame.next];
            frame.next += 1;
            return ",";
        }
    }

    fn floatText(self: *KeyIterator, f: f64) []const u8 {
        // TypeAdapter(dict)'s default ser_json_inf_nan="null" maps all
        // nonfinite dictionary keys to Python None's key spelling.
        if (!std.math.isFinite(f)) return "None";
        var fixed: Writer = .fixed(&self.buffer);
        pyjson.writeFloat(&fixed, f, .python) catch unreachable;
        return fixed.buffered();
    }
};

fn writeKeyText(w: *Writer, key: Value) (Writer.Error || KeyError)!void {
    var iterator = KeyIterator.init(key);
    while (try iterator.next()) |chunk| try w.writeAll(chunk);
}

fn keyTextEqual(a: Value, b: Value) KeyError!bool {
    var left = KeyIterator.init(a);
    var right = KeyIterator.init(b);
    var left_chunk: []const u8 = "";
    var right_chunk: []const u8 = "";
    var left_end = false;
    var right_end = false;
    while (true) {
        while (left_chunk.len == 0 and !left_end) {
            left_chunk = (try left.next()) orelse blk: {
                left_end = true;
                break :blk "";
            };
        }
        while (right_chunk.len == 0 and !right_end) {
            right_chunk = (try right.next()) orelse blk: {
                right_end = true;
                break :blk "";
            };
        }
        if (left_end or right_end) return left_end and right_end;
        const length = @min(left_chunk.len, right_chunk.len);
        if (!std.mem.eql(u8, left_chunk[0..length], right_chunk[0..length])) return false;
        left_chunk = left_chunk[length..];
        right_chunk = right_chunk[length..];
    }
}

fn writeQuotedKey(w: *Writer, key: Value) (Writer.Error || KeyError)!void {
    try w.writeByte('"');
    var escaped: pyjson.StringContentsWriter = .{ .destination = w };
    try writeKeyText(&escaped.writer, key);
    try w.writeByte('"');
}

fn writeRuntimeJson(jws: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
    // This is plain Python json.dumps, NOT DSPy's Pydantic conversion. Tuples
    // are JSON arrays; bytes, sets, complex and Ellipsis are not serializable.
    // Stringify's callback error set cannot express TypeError, so unsupported
    // values report WriteFailed rather than inventing a successful JSON value.
    switch (runtimeBox(ptr).payload) {
        .tuple => |items| try jws.write(items),
        .dictionary => |pairs| {
            try jws.beginObject();
            for (pairs) |pair| {
                if (pair.key == .string) {
                    try jws.objectField(pair.key.string);
                } else {
                    try jws.beginObjectFieldRaw();
                    try jws.writer.writeByte('"');
                    switch (pair.key) {
                        .null => try jws.writer.writeAll("null"),
                        .bool => |b| try jws.writer.writeAll(if (b) "true" else "false"),
                        .integer => |i| try jws.writer.print("{d}", .{i}),
                        .float => |f| try pyjson.writeFloat(jws.writer, f, .json),
                        .number_string => |text| {
                            if (integerTextTooLong(text)) return error.WriteFailed;
                            if (std.json.isNumberFormattedLikeAnInteger(text)) {
                                try jws.writer.writeAll(text);
                            } else {
                                const f = std.fmt.parseFloat(f64, text) catch return error.WriteFailed;
                                try pyjson.writeFloat(jws.writer, f, .json);
                            }
                        },
                        else => return error.WriteFailed,
                    }
                    try jws.writer.writeByte('"');
                    jws.endObjectFieldRaw();
                }
                try jws.write(pair.value);
            }
            try jws.endObject();
        },
        else => return error.WriteFailed,
    }
}

const SerializedWriteError = runtime.CustomWriteError || error{PythonSerializeFallback};

fn writeSerializedJson(w: *Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
    writePythonSerialized(w, runtimeBox(ptr).payload, style, depth) catch |err| return switch (err) {
        error.PythonSerializeFallback => error.ParseValue,
        else => |other| other,
    };
}

fn separator(w: *Writer, style: runtime.JsonStyle, depth: usize, first: bool) Writer.Error!void {
    switch (style) {
        .compact => if (!first) try w.writeAll(", "),
        .minified => if (!first) try w.writeByte(','),
        .indent2, .orjson_indent2 => {
            if (!first) try w.writeByte(',');
            try w.writeByte('\n');
            try w.splatByteAll(' ', depth * 2);
        },
    }
}

fn close(w: *Writer, style: runtime.JsonStyle, depth: usize, byte: u8) Writer.Error!void {
    if (style == .indent2 or style == .orjson_indent2) {
        try w.writeByte('\n');
        try w.splatByteAll(' ', depth * 2);
    }
    try w.writeByte(byte);
}

fn writeSerializedValue(w: *Writer, v: Value, style: runtime.JsonStyle, depth: usize) SerializedWriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (view(v)) |python| return writePythonSerialized(w, python, style, depth);
    switch (v) {
        .array => |array| try writeSerializedItems(w, array.items, style, depth),
        .object => |object| {
            if (object.count() == 0) return w.writeAll("{}");
            try w.writeByte('{');
            for (object.keys(), object.values(), 0..) |key, item, index| {
                try separator(w, style, depth + 1, index == 0);
                try pyjson.writeJsonString(w, key);
                try w.writeAll(if (style == .minified) ":" else ": ");
                try writeSerializedValue(w, item, style, depth + 1);
            }
            try close(w, style, depth, '}');
        },
        .custom => |custom| try custom.writeSerializedJson(w, style, depth),
        else => try pyjson.writeJsonProjection(w, v, style, @intCast(depth)),
    }
}

fn writeSerializedItems(w: *Writer, items: []const Value, style: runtime.JsonStyle, depth: usize) SerializedWriteError!void {
    if (items.len == 0) return w.writeAll("[]");
    try w.writeByte('[');
    for (items, 0..) |item, index| {
        try separator(w, style, depth + 1, index == 0);
        try writeSerializedValue(w, item, style, depth + 1);
    }
    try close(w, style, depth, ']');
}

fn writePythonSerialized(w: *Writer, python: PythonValue, style: runtime.JsonStyle, depth: usize) SerializedWriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    switch (python) {
        .bytes => |data| {
            if (!std.unicode.utf8ValidateSlice(data)) return error.PythonSerializeFallback;
            try pyjson.writeJsonString(w, data);
        },
        .tuple, .set, .frozen_set => |items| try writeSerializedItems(w, items, style, depth),
        .complex => |c| {
            try w.writeByte('"');
            try writeComplexJsonText(w, c);
            try w.writeByte('"');
        },
        // Root TypeAdapter(ellipsis) retains the object; nested inference with
        // fallback=str converts it to text. Plain JSON must reject the root.
        .ellipsis => if (depth == 0) return error.ParseValue else try pyjson.writeJsonString(w, "Ellipsis"),
        .dictionary => |pairs| {
            if (pairs.len == 0) return w.writeAll("{}");
            // Collision resolution below looks ahead. First visit every key
            // and value in Source order, including overwritten values, so a
            // later invalid byte key cannot hide an earlier TypeFormatError.
            var preflight: TextWindowWriter = .{ .destination = &.{}, .skip = 0 };
            for (pairs) |entry| {
                try writeKeyText(&preflight.writer, entry.key);
                try writeSerializedValue(&preflight.writer, entry.value, style, depth + 1);
            }
            try w.writeByte('{');
            var first = true;
            for (pairs, 0..) |entry, index| {
                // JSON-mode key coercion can create a NEW collision: integer 1
                // and string "1", or tuple (1,2) and string "1,2". Emit the
                // first key's position with the last matching value.
                var seen = false;
                for (pairs[0..index]) |prior| {
                    if (try keyTextEqual(prior.key, entry.key)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;
                var last_value = entry.value;
                for (pairs[index + 1 ..]) |later| {
                    if (try keyTextEqual(entry.key, later.key)) last_value = later.value;
                }
                try separator(w, style, depth + 1, first);
                first = false;
                try writeQuotedKey(w, entry.key);
                try w.writeAll(if (style == .minified) ":" else ": ");
                try writeSerializedValue(w, last_value, style, depth + 1);
            }
            try close(w, style, depth, '}');
        },
    }
}

const testing = std.testing;

fn expectRepr(expected: []const u8, v: Value) !void {
    var output: Writer.Allocating = .init(testing.allocator);
    defer output.deinit();
    try pyjson.writeRepr(&output.writer, v, 0);
    try testing.expectEqualStrings(expected, output.written());
}

fn expectSerialized(expected: []const u8, v: Value) !void {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const result = try value_ops.serializedValue(scratch.allocator(), v);
    var output: Writer.Allocating = .init(testing.allocator);
    defer output.deinit();
    try pyjson.writeJson(&output.writer, result, .compact);
    try testing.expectEqualStrings(expected, output.written());
}

/// Native counterpart of the actual StringKey class in the independent Source
/// oracle. This exercises the full custom-key protocol, not a JSON lookalike.
const SourceStringKey = struct {
    text: []const u8,
    ordered: bool = false,

    const vtable: runtime.CustomVTable = .{
        .type_name = @typeName(SourceStringKey),
        .model_name = "StringKey",
        .hashable = true,
        .python_hash = hash,
        .raw_value = raw,
        .serialized = serialize,
        .write_str = writeStr,
        .write_repr = writeRepr,
        .write_json = writeJson,
        .write_serialized_json = SourceStringKey.writeSerializedJson,
        .clone = cloneKey,
        .deinit = deinitKey,
        .eql = equalKey,
        .eql_value = equalKeyValue,
    };

    fn get(ptr: *const anyopaque) *const SourceStringKey {
        return @ptrCast(@alignCast(ptr));
    }

    fn toRuntime(a: Allocator, text: []const u8) Allocator.Error!Value {
        return create(a, text, false);
    }

    fn toOrderedRuntime(a: Allocator, text: []const u8) Allocator.Error!Value {
        return create(a, text, true);
    }

    fn create(a: Allocator, text: []const u8, ordered: bool) Allocator.Error!Value {
        const key = try a.create(SourceStringKey);
        key.* = .{ .text = text, .ordered = ordered };
        return .{ .custom = .{ .ptr = key, .vtable = &vtable } };
    }

    fn hash(ptr: *const anyopaque, seed: u32) error{ParseValue}!i64 {
        if (get(ptr).ordered) return 0;
        return @as(i64, @bitCast(try stringHash(get(ptr).text, seed)));
    }

    fn raw(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
        return .{ .custom = .{ .ptr = ptr, .vtable = &vtable } };
    }

    fn serialize(a: Allocator, ptr: *const anyopaque, _: u32) runtime.SerializationError!Value {
        var output: Writer.Allocating = .init(a);
        defer output.deinit();
        writeStr(&output.writer, ptr) catch |err| return switch (err) {
            error.WriteFailed => error.OutOfMemory,
            error.ParseValue, error.JsonTooDeep => error.ParseValue,
        };
        return .{ .string = try output.toOwnedSlice() };
    }

    fn writeStr(w: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
        try w.writeAll("StringKey(");
        try pyjson.writeReprString(w, get(ptr).text);
        if (get(ptr).ordered) try w.writeAll(", ordered=True");
        try w.writeByte(')');
    }

    fn writeRepr(w: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
        if (depth >= pyjson.max_depth) return error.JsonTooDeep;
        try writeStr(w, ptr);
    }

    fn writeJson(_: *std.json.Stringify, _: *const anyopaque) Writer.Error!void {
        return error.WriteFailed;
    }

    fn writeSerializedJson(w: *Writer, ptr: *const anyopaque, _: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
        if (depth >= pyjson.max_depth) return error.JsonTooDeep;
        try w.writeByte('"');
        var escaped: pyjson.StringContentsWriter = .{ .destination = w };
        try writeStr(&escaped.writer, ptr);
        try w.writeByte('"');
    }

    fn cloneKey(a: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
        const copy = try a.create(SourceStringKey);
        errdefer a.destroy(copy);
        copy.* = .{ .text = try a.dupe(u8, get(ptr).text), .ordered = get(ptr).ordered };
        return .{ .ptr = copy, .vtable = &vtable, .owns_data = true };
    }

    fn deinitKey(a: Allocator, ptr: *const anyopaque, owns_data: bool) void {
        const key = get(ptr);
        if (owns_data) a.free(key.text);
        a.destroy(key);
    }

    fn equalKey(left: *const anyopaque, right: *const anyopaque) bool {
        const a = get(left);
        const b = get(right);
        if (a.ordered != b.ordered) return false;
        return if (a.ordered) std.mem.order(u8, a.text, b.text) != .gt else std.mem.eql(u8, a.text, b.text);
    }

    fn equalKeyValue(ptr: *const anyopaque, other: Value) bool {
        if (other == .custom) {
            if (other.custom.get(SourceStringKey)) |key|
                return equalKey(ptr, key);
        }
        if (get(ptr).ordered) return false;
        return value_ops.pyEqual(.{ .string = get(ptr).text }, other);
    }
};

test "Python custom keys equal ordinary dictionary string keys through their hash protocol" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const key = try SourceStringKey.toRuntime(a, "x");
    const ordinary = try dictionary(a, &.{.{ .key = .{ .string = "x" }, .value = .{ .integer = 1 } }});
    const arbitrary = try dictionary(a, &.{.{ .key = key, .value = .{ .integer = 1 } }});
    try testing.expect(value_ops.pyEqual(arbitrary, ordinary));
    try testing.expect(value_ops.pyEqual(ordinary, arbitrary));
    const custom_first = try dictionary(a, &.{
        .{ .key = key, .value = .{ .integer = 1 } },
        .{ .key = .{ .string = "x" }, .value = .{ .integer = 2 } },
    });
    try testing.expectEqual(@as(usize, 1), view(custom_first).?.dictionary.len);
    try testing.expectEqual(key.custom.ptr, view(custom_first).?.dictionary[0].key.custom.ptr);
    try expectRepr("{StringKey('x'): 2}", custom_first);
    try expectSerialized("{\"StringKey('x')\": 2}", custom_first);
    const string_first = try dictionary(a, &.{
        .{ .key = .{ .string = "x" }, .value = .{ .integer = 1 } },
        .{ .key = key, .value = .{ .integer = 2 } },
    });
    try testing.expect(string_first == .object);
    try expectRepr("{'x': 2}", string_first);
    try expectRepr("{StringKey('x')}", try set(a, &.{ key, .{ .string = "x" } }));
    try expectRepr("{'x'}", try set(a, &.{ .{ .string = "x" }, key }));

    var no_hash = SourceStringKey.vtable;
    no_hash.python_hash = null;
    const unregistered: Value = .{ .custom = .{ .ptr = key.custom.ptr, .vtable = &no_hash } };
    try testing.expectError(error.ParseValue, set(a, &.{unregistered}));
    try testing.expectError(error.ParseValue, dictionary(a, &.{.{ .key = unregistered, .value = .null }}));

    const owned = try ownership.cloneJsonValue(testing.allocator, arbitrary);
    defer ownership.freeJsonValue(testing.allocator, owned);
    try testing.expect(value_ops.pyEqual(owned, ordinary));
    try testing.expect(value_ops.pyEqual(ordinary, owned));
}

test "Python set and dictionary equality preserve asymmetric lookup operand direction" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const lower = try SourceStringKey.toOrderedRuntime(a, "a");
    const upper = try SourceStringKey.toOrderedRuntime(a, "b");
    try testing.expect(value_ops.pyEqual(lower, upper));
    try testing.expect(!value_ops.pyEqual(upper, lower));
    const lower_dict = try dictionary(a, &.{.{ .key = lower, .value = .{ .integer = 1 } }});
    const upper_dict = try dictionary(a, &.{.{ .key = upper, .value = .{ .integer = 1 } }});
    try testing.expect(!value_ops.pyEqual(lower_dict, upper_dict));
    try testing.expect(value_ops.pyEqual(upper_dict, lower_dict));
    const lower_set = try set(a, &.{lower});
    const upper_set = try set(a, &.{upper});
    try testing.expect(!value_ops.pyEqual(lower_set, upper_set));
    try testing.expect(value_ops.pyEqual(upper_set, lower_set));
    const lower_frozen = try frozenSet(a, &.{lower});
    try testing.expect(!value_ops.pyEqual(lower_frozen, upper_set));
    try testing.expect(value_ops.pyEqual(upper_set, lower_frozen));
    try expectRepr("{StringKey('a', ordered=True)}", try set(a, &.{ lower, upper }));
    try expectRepr("{StringKey('b', ordered=True), StringKey('a', ordered=True)}", try set(a, &.{ upper, lower }));
}

test "CPython dictionaries use dict probes growth and Unicode conversion rather than set layout" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const Fixture = struct {
        prefix: []const []const u8,
        sequence: []const u8,
        expected_values: []const i64,
    };
    // Actual CPython observations: cross_value_keys.dictionary_probes in the
    // Source oracle. Each key has hash zero and asymmetric lexical <= equality.
    const fixtures = [_]Fixture{
        .{ .prefix = &.{}, .sequence = "zycbad", .expected_values = &.{ 1, 2, 6, 4, 5 } },
        .{
            .prefix = &.{},
            .sequence = "zyxwvutsrqponmlkjihgfedcbaf",
            .expected_values = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 22, 23, 24, 25, 26 },
        },
        .{ .prefix = &.{ "s", "t", "u" }, .sequence = "zycbad", .expected_values = &.{ 1, 2, 3, 4, 5, 9, 7, 8 } },
    };
    for (fixtures) |fixture| {
        var pairs: std.ArrayList(PythonEntry) = .empty;
        defer pairs.deinit(a);
        for (fixture.prefix) |text| try pairs.append(a, .{
            .key = .{ .string = text },
            .value = .{ .integer = @intCast(pairs.items.len + 1) },
        });
        for (fixture.sequence, 0..) |_, index| try pairs.append(a, .{
            .key = try SourceStringKey.toOrderedRuntime(a, fixture.sequence[index .. index + 1]),
            .value = .{ .integer = @intCast(pairs.items.len + 1) },
        });
        const actual = try dictionary(a, pairs.items);
        const retained = view(actual).?.dictionary;
        try testing.expectEqual(fixture.expected_values.len, retained.len);
        for (retained, fixture.expected_values, 0..) |entry, expected, index| {
            try testing.expectEqual(expected, entry.value.integer);
            if (index < fixture.prefix.len) {
                try testing.expectEqualStrings(fixture.prefix[index], entry.key.string);
            } else {
                try testing.expectEqual(pairs.items[index].key.custom.ptr, entry.key.custom.ptr);
            }
        }
        const owned = try ownership.cloneJsonValue(testing.allocator, actual);
        defer ownership.freeJsonValue(testing.allocator, owned);
        try testing.expect(value_ops.pyEqual(actual, owned));
        try testing.expect(value_ops.pyEqual(owned, actual));
    }
}

test "Python bytes retain identity and exact byte repr" {
    const cases = [_][2][]const u8{
        .{ "", "b''" },
        .{ "'", "b\"'\"" },
        .{ "\"", "b'\"'" },
        .{ "'\"", "b'\\'\"'" },
        .{ "\x00\t\n\r\\\x7f\xff", "b'\\x00\\t\\n\\r\\\\\\x7f\\xff'" },
        .{ "雪", "b'\\xe9\\x9b\\xaa'" },
    };
    for (cases) |case| {
        const actual = try bytes(testing.allocator, case[0]);
        defer actual.custom.deinit(testing.allocator);
        try testing.expect(view(actual).? == .bytes);
        try testing.expectEqualStrings(case[0], view(actual).?.bytes);
        try testing.expect(actual.custom.vtable.hashable);
        try testing.expect(!actual.custom.vtable.is_type);
        try testing.expectEqual(case[0].len != 0, actual.custom.truthy());
        try testing.expect(!value_ops.pyEqual(actual, .{ .string = case[0] }));
        try testing.expect(!value_ops.pyEqual(.{ .string = case[0] }, actual));
        try expectRepr(case[1], actual);
    }
}

test "Python tuple repr identity truth and recursive hash admission" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const empty = try tuple(a, &.{});
    const one = try tuple(a, &.{.{ .integer = 1 }});
    const equal_one = try tuple(a, &.{.{ .bool = true }});
    const list: Value = .{ .array = .init(a) };
    const unhashable = try tuple(a, &.{list});
    try testing.expect(!empty.custom.truthy());
    try testing.expect(one.custom.truthy());
    try testing.expect(one.custom.vtable.hashable);
    try testing.expect(!unhashable.custom.vtable.hashable);
    try testing.expect(value_ops.pyEqual(one, equal_one));
    try testing.expect(!value_ops.pyEqual(empty, list));
    try testing.expect(!value_ops.pyEqual(list, empty));
    try expectRepr("()", empty);
    try expectRepr("(1,)", one);
    try expectRepr("([],)", unhashable);
    try testing.expectError(error.ParseValue, set(a, &.{unhashable}));
    try testing.expectError(error.ParseValue, frozenSet(a, &.{list}));
    try testing.expectError(error.ParseValue, dictionary(a, &.{.{ .key = unhashable, .value = .null }}));
    // A representable container identity is reflexive even with an embedded
    // NaN; distinct embedded float-object identities remain a runtime limit.
    const nan_tuple = try tuple(a, &.{.{ .float = std.math.nan(f64) }});
    try testing.expect(value_ops.pyEqual(nan_tuple, nan_tuple));
}

test "CPython seed-zero set layout dedup and frozen cross equality" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const numeric = [_]Value{
        .{ .integer = 9 }, .{ .integer = -1 }, .{ .integer = 1 }, .{ .integer = 4 },
        .{ .integer = 0 }, .{ .integer = 32 }, .{ .integer = 8 }, .{ .integer = 3 },
    };
    const actual = try set(a, &numeric);
    const frozen = try frozenSet(a, &numeric);
    try expectRepr("{0, 1, 32, 3, 4, 8, 9, -1}", actual);
    try expectRepr("frozenset({0, 1, 32, 3, 4, 8, 9, -1})", frozen);
    try testing.expect(value_ops.pyEqual(actual, frozen));
    try testing.expect(value_ops.pyEqual(frozen, actual));
    try testing.expect(!actual.custom.vtable.hashable);
    try testing.expect(frozen.custom.vtable.hashable);
    try testing.expectEqual(@as(i64, -7633075888884608186), @as(i64, @bitCast(try hashValue(frozen, 0))));
    const collisions = try set(a, &.{
        .{ .integer = 0 },  .{ .integer = 8 },  .{ .integer = 16 }, .{ .integer = 24 },
        .{ .integer = 32 }, .{ .integer = 40 }, .{ .integer = 48 },
    });
    try expectRepr("{0, 32, 8, 40, 16, 48, 24}", collisions);
    const one_complex = try complex(a, 1, 0);
    const equal_numeric = try set(a, &.{
        .{ .bool = true }, .{ .integer = 1 }, .{ .float = 1 }, one_complex,
        .{ .bool = false }, .{ .integer = 0 }, .{ .float = -0.0 },
    });
    try expectRepr("{False, True}", equal_numeric);
    try testing.expectEqual(@as(usize, 2), view(equal_numeric).?.set.len);
    try expectRepr("set()", try set(a, &.{}));
    const empty_frozen = try frozenSet(a, &.{});
    try expectRepr("frozenset()", empty_frozen);
    try testing.expect(!empty_frozen.custom.truthy());
    try testing.expectEqual(@as(u64, 133146708735736), try hashValue(empty_frozen, 0));
}

test "CPython Unicode storage and explicit seed determine genuine set order" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const strings = [_]Value{
        .{ .string = "x" }, .{ .string = "y" }, .{ .string = "z" },
        .{ .string = "é" }, .{ .string = "雪" }, .{ .string = "😀" },
    };
    try expectRepr("{'y', 'x', 'é', 'z', '😀', '雪'}", try set(a, &strings));
    try expectRepr("{'z', '雪', 'x', '😀', 'y', 'é'}", try setWithHashSeed(a, &strings, 1));
    const byte_items = [_]Value{
        try bytes(a, "x"), try bytes(a, "y"), try bytes(a, "z"), try bytes(a, "\xff"),
    };
    try expectRepr("{b'y', b'x', b'z', b'\\xff'}", try set(a, &byte_items));
    try expectRepr("{b'x', b'z', b'y', b'\\xff'}", try setWithHashSeed(a, &byte_items, 1));
    const string_hashes = [_]struct { text: []const u8, hash: i64 }{
        .{ .text = "", .hash = 0 },
        .{ .text = "x", .hash = -3368204716105132637 },
        .{ .text = "é", .hash = 6047309291227476195 },
        .{ .text = "雪", .hash = -3030076488861377292 },
        .{ .text = "😀", .hash = -3536540696076613844 },
        .{ .text = "\xed\xa0\x80", .hash = 5926611903945376553 },
        .{ .text = "a雪😀", .hash = 6087975628016107452 },
    };
    for (string_hashes) |case|
        try testing.expectEqual(case.hash, @as(i64, @bitCast(try stringHash(case.text, 0))));
    const float_hashes = [_]struct { number: f64, hash: i64 }{
        .{ .number = 0, .hash = 0 },
        .{ .number = -0.0, .hash = 0 },
        .{ .number = 0.5, .hash = 1152921504606846976 },
        .{ .number = -1, .hash = -2 },
        .{ .number = 1e-5, .hash = 2170758078822671208 },
        .{ .number = 1e16, .hash = 10000000000000000 },
        .{ .number = 1e100, .hash = 1822893315824342674 },
        .{ .number = 0x1p-1074, .hash = 16777216 },
        .{ .number = std.math.inf(f64), .hash = 314159 },
        .{ .number = -std.math.inf(f64), .hash = -314159 },
    };
    for (float_hashes) |case|
        try testing.expectEqual(case.hash, @as(i64, @bitCast(floatHash(case.number, null))));
}

test "Python dictionaries retain original equal key and insertion order" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const actual = try dictionary(a, &.{
        .{ .key = .{ .bool = true }, .value = .{ .string = "bool" } },
        .{ .key = .{ .float = 1 }, .value = .{ .string = "float" } },
        .{ .key = try complex(a, 1, 0), .value = .{ .string = "complex" } },
        .{ .key = .{ .integer = 0 }, .value = .{ .string = "zero" } },
    });
    const pairs = view(actual).?.dictionary;
    try testing.expectEqual(@as(usize, 2), pairs.len);
    try testing.expect(pairs[0].key == .bool and pairs[0].key.bool);
    try expectRepr("{True: 'complex', 0: 'zero'}", actual);
    const reverse = try dictionary(a, &.{
        .{ .key = .{ .integer = 0 }, .value = .{ .string = "zero" } },
        .{ .key = .{ .integer = 1 }, .value = .{ .string = "complex" } },
    });
    try testing.expect(value_ops.pyEqual(actual, reverse));
    const ordinary = try dictionary(a, &.{
        .{ .key = .{ .string = "x" }, .value = .{ .integer = 1 } },
        .{ .key = .{ .string = "y" }, .value = .{ .integer = 2 } },
        .{ .key = .{ .string = "x" }, .value = .{ .integer = 3 } },
    });
    try testing.expect(ordinary == .object);
    try expectRepr("{'x': 3, 'y': 2}", ordinary);
    const list: Value = .{ .array = .init(a) };
    try testing.expectError(error.ParseValue, dictionary(a, &.{.{ .key = list, .value = .null }}));
    try testing.expectError(error.ParseValue, set(a, &.{actual}));
    const tuple_key = try tuple(a, &.{.{ .integer = 1 }});
    const tuple_key_equal = try tuple(a, &.{.{ .bool = true }});
    const tuple_dict = try dictionary(a, &.{
        .{ .key = tuple_key, .value = .{ .string = "first" } },
        .{ .key = tuple_key_equal, .value = .{ .string = "last" } },
    });
    try testing.expectEqual(@as(usize, 1), view(tuple_dict).?.dictionary.len);
    try expectRepr("{(1,): 'last'}", tuple_dict);
}

test "Python complex source repr numeric equality and Pydantic text differ" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const cases = [_]struct { real: f64, imag: f64, repr: []const u8, json: []const u8 }{
        .{ .real = 0, .imag = 0, .repr = "0j", .json = "\"0j\"" },
        .{ .real = -0.0, .imag = 0, .repr = "(-0+0j)", .json = "\"0j\"" },
        .{ .real = 0, .imag = -0.0, .repr = "-0j", .json = "\"-0j\"" },
        .{ .real = -0.0, .imag = -0.0, .repr = "(-0-0j)", .json = "\"-0j\"" },
        .{ .real = 1, .imag = 2, .repr = "(1+2j)", .json = "\"1+2j\"" },
        .{ .real = 1, .imag = -2, .repr = "(1-2j)", .json = "\"1-2j\"" },
        .{ .real = 0, .imag = 1e16, .repr = "1e+16j", .json = "\"10000000000000000j\"" },
        .{ .real = 1e-5, .imag = -1e-5, .repr = "(1e-05-1e-05j)", .json = "\"0.00001-0.00001j\"" },
        .{ .real = std.math.inf(f64), .imag = -std.math.inf(f64), .repr = "(inf-infj)", .json = "\"inf-infj\"" },
        .{ .real = std.math.nan(f64), .imag = std.math.nan(f64), .repr = "(nan+nanj)", .json = "\"NaN+NaNj\"" },
    };
    for (cases) |case| {
        const actual = try complex(a, case.real, case.imag);
        try expectRepr(case.repr, actual);
        try expectSerialized(case.json, actual);
        try testing.expectEqual(case.real != 0 or case.imag != 0, actual.custom.truthy());
    }
    const one = try complex(a, 1, -0.0);
    try testing.expect(value_ops.pyEqual(one, .{ .bool = true }));
    try testing.expect(value_ops.pyEqual(.{ .integer = 1 }, one));
    try testing.expect(value_ops.pyEqual(one, .{ .number_string = "1" }));
    const rounded = try complex(a, 9007199254740992, 0);
    try testing.expect(!value_ops.pyEqual(rounded, .{ .number_string = "9007199254740993" }));
    const nan = try complex(a, std.math.nan(f64), 0);
    try testing.expect(!value_ops.pyEqual(nan, nan));
    const dedup_nan = try set(a, &.{ nan, nan });
    try testing.expectEqual(@as(usize, 1), view(dedup_nan).?.set.len);
    const other_nan = try complex(a, std.math.nan(f64), 0);
    const separate = try set(a, &.{ nan, other_nan });
    try testing.expectEqual(@as(usize, 2), view(separate).?.set.len);
}

test "Builtin JSON conversion rejects plain unsupported values and preserves top-level fallback" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const valid = try bytes(a, "雪");
    try expectSerialized("\"雪\"", valid);
    const invalid = try bytes(a, "\xff");
    try testing.expectError(error.PythonSerializeFallback, invalid.custom.serialize(a));
    const original = try tuple(a, &.{ invalid, .{ .integer = 7 } });
    try expectSerialized("\"(b'\\\\xff', 7)\"", original);
    var buffer: [256]u8 = undefined;
    var writer: Writer = .fixed(&buffer);
    try testing.expectError(error.WriteFailed, std.json.Stringify.value(valid, .{}, &writer));
    writer.end = 0;
    try testing.expectError(error.ParseValue, invalid.custom.writeSerializedJson(&writer, .compact, 0));
    const ellipse = try ellipsis(a);
    const same = try value_ops.serializedValue(a, ellipse);
    try testing.expectEqual(ellipse.custom.ptr, same.custom.ptr);
    try testing.expect(ellipse.custom.truthy());
    try testing.expect(ellipse.custom.vtable.hashable);
    try expectRepr("Ellipsis", ellipse);
    writer.end = 0;
    try testing.expectError(error.ParseValue, ellipse.custom.writeSerializedJson(&writer, .compact, 0));
    const nested = try tuple(a, &.{ valid, ellipse });
    try expectSerialized("[\"雪\", \"Ellipsis\"]", nested);
    writer.end = 0;
    try nested.custom.writeSerializedJson(&writer, .compact, 0);
    try testing.expectEqualStrings("[\"雪\", \"Ellipsis\"]", writer.buffered());
    const plain_tuple = try tuple(a, &.{.{ .integer = 1 }});
    writer.end = 0;
    try std.json.Stringify.value(plain_tuple, .{}, &writer);
    try testing.expectEqualStrings("[1]", writer.buffered());
    var no_memory: std.heap.FixedBufferAllocator = .init(&.{});
    const raw = try valid.custom.rawValue(no_memory.allocator());
    try testing.expectEqual(valid.custom.ptr, raw.custom.ptr);
}

test "Pydantic dictionary key conversion merges new collisions without replacing order" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const source = try dictionary(a, &.{
        .{ .key = .{ .integer = 1 }, .value = .{ .string = "integer" } },
        .{ .key = .{ .string = "1" }, .value = .{ .string = "string" } },
        .{ .key = .null, .value = .{ .string = "null" } },
        .{ .key = .{ .string = "None" }, .value = .{ .string = "text" } },
    });
    try expectSerialized("{\"1\": \"string\", \"None\": \"text\"}", source);
    var buffer: [256]u8 = undefined;
    var writer: Writer = .fixed(&buffer);
    try source.custom.writeSerializedJson(&writer, .compact, 0);
    try testing.expectEqualStrings("{\"1\": \"string\", \"None\": \"text\"}", writer.buffered());
    const key = try tuple(a, &.{ .{ .integer = 1 }, .{ .integer = 2 } });
    const flattened = try dictionary(a, &.{
        .{ .key = key, .value = .{ .integer = 1 } },
        .{ .key = .{ .string = "1,2" }, .value = .{ .integer = 2 } },
    });
    try expectSerialized("{\"1,2\": 2}", flattened);
    const invalid_key = try frozenSet(a, &.{.{ .integer = 1 }});
    const invalid = try dictionary(a, &.{.{ .key = invalid_key, .value = .{ .integer = 7 } }});
    try testing.expectError(error.PythonSerializeFallback, invalid.custom.serialize(a));
    try expectSerialized("\"{frozenset({1}): 7}\"", invalid);
    const nonfinite = try dictionary(a, &.{
        .{ .key = .{ .float = std.math.nan(f64) }, .value = .{ .integer = 1 } },
        .{ .key = .{ .float = std.math.inf(f64) }, .value = .{ .integer = 2 } },
        .{ .key = .null, .value = .{ .integer = 3 } },
    });
    try expectSerialized("{\"None\": 3}", nonfinite);
}

test "Builtin owning clone survives every borrowed source allocation" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    const a = scratch.allocator();
    const key = try bytes(a, try a.dupe(u8, "key"));
    const binary = try bytes(a, try a.dupe(u8, "\xffdata"));
    var list: runtime.Array = .init(a);
    try list.append(.{ .string = try a.dupe(u8, "text") });
    try list.append(.{ .number_string = try a.dupe(u8, "18446744073709551616") });
    const nested = try tuple(a, &.{ binary, .{ .array = list } });
    const frozen = try frozenSet(a, &.{ .{ .integer = 2 }, .{ .integer = 3 } });
    const source = try dictionary(a, &.{
        .{ .key = key, .value = nested },
        .{ .key = .{ .integer = 1 }, .value = frozen },
    });
    const copied = ownership.cloneJsonValue(testing.allocator, source) catch |err| {
        scratch.deinit();
        return err;
    };
    scratch.deinit();
    defer ownership.freeJsonValue(testing.allocator, copied);
    try expectRepr("{b'key': (b'\\xffdata', ['text', 18446744073709551616]), 1: frozenset({2, 3})}", copied);
    try testing.expect(copied.custom.owns_data);
    const second = try ownership.cloneJsonValue(testing.allocator, copied);
    defer ownership.freeJsonValue(testing.allocator, second);
    try testing.expect(value_ops.pyEqual(copied, second));
}

test "Arbitrary dictionary model keys stream Source text and long-key collisions" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const File = @import("file.zig").File;
    const BoundingBox = @import("bounding_box.zig").BoundingBox;
    const bounds = try (BoundingBox{ .top = 0, .bottom = 1, .left = 0, .right = 1 }).toRuntime(a);
    try testing.expectEqual(@as(i64, -5021629377307685470), @as(i64, @bitCast(try hashValue(bounds, 0))));
    const geometric = try dictionary(a, &.{.{ .key = bounds, .value = .{ .integer = 7 } }});
    try expectSerialized("{\"top=0.0 bottom=1.0 left=0.0 right=1.0\": 7}", geometric);

    const identifier = try a.alloc(u8, 2000);
    @memset(identifier, 'x');
    const file = try (File{ .file_id = identifier }).toRuntime(a);
    var source_key: Writer.Allocating = .init(a);
    try source_key.writer.writeAll("<<CUSTOM-TYPE-START-IDENTIFIER>>[{\"type\": \"document\", \"media_type\": \"application/pdf\", \"file_id\": \"");
    try source_key.writer.writeAll(identifier);
    try source_key.writer.writeAll("\"}]<<CUSTOM-TYPE-END-IDENTIFIER>>");
    const mapping = try dictionary(a, &.{
        .{ .key = file, .value = .{ .integer = 1 } },
        .{ .key = .{ .string = source_key.written() }, .value = .{ .integer = 2 } },
    });
    try testing.expectEqual(@as(usize, 2), view(mapping).?.dictionary.len);
    const projected = try value_ops.serializedValue(a, mapping);
    try testing.expectEqual(@as(usize, 1), projected.object.count());
    try testing.expectEqual(@as(i64, 2), projected.object.get(source_key.written()).?.integer);
    var output: Writer.Allocating = .init(a);
    try mapping.custom.writeSerializedJson(&output.writer, .compact, 0);
    var expected: Writer.Allocating = .init(a);
    try expected.writer.writeByte('{');
    try pyjson.writeJsonString(&expected.writer, source_key.written());
    try expected.writer.writeAll(": 2}");
    try testing.expectEqualStrings(expected.written(), output.written());
}

test "Builtin ownership clone keeps set order rather than reconstructing Python deepcopy" {
    const a = testing.allocator;
    const original = try set(a, &.{ .{ .integer = 943 }, .{ .integer = -417 } });
    defer original.custom.deinit(a);
    try expectRepr("{-417, 943}", original);
    const copy = try ownership.cloneJsonValue(a, original);
    defer ownership.freeJsonValue(a, copy);
    try expectRepr("{-417, 943}", copy);
    try testing.expect(value_ops.pyEqual(original, copy));
}

test "Builtin repr and dictionary keys retain the Source decimal conversion limit" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    const digits = try a.alloc(u8, 4301);
    @memset(digits, '9');
    const large: Value = .{ .number_string = digits };
    const container = try tuple(a, &.{large});
    var output: Writer.Allocating = .init(a);
    try testing.expectError(error.ParseValue, pyjson.writeRepr(&output.writer, container, 0));
    const keyed = try dictionary(a, &.{.{ .key = large, .value = .null }});
    try testing.expectError(error.PythonSerializeFallback, keyed.custom.serialize(a));
    try testing.expectError(error.ParseValue, value_ops.serializedValue(a, keyed));
}

test "Native containers cannot reset boxed builtin representation depth" {
    var scratch: std.heap.ArenaAllocator = .init(testing.allocator);
    defer scratch.deinit();
    const a = scratch.allocator();
    var nested = try tuple(a, &.{.{ .integer = 1 }});
    for (0..pyjson.max_depth - 1) |_| {
        var list: runtime.Array = .init(a);
        try list.append(nested);
        nested = .{ .array = list };
    }
    var output: Writer.Allocating = .init(a);
    try testing.expectError(error.JsonTooDeep, pyjson.writeRepr(&output.writer, nested, 0));
    try testing.expectError(error.ParseValue, value_ops.serializedValue(a, nested));
}

fn exerciseBuiltinOwnership(a: Allocator) !void {
    const binary = try bytes(a, "borrowed 雪");
    defer binary.custom.deinit(a);
    const point = try complex(a, -0.0, std.math.inf(f64));
    defer point.custom.deinit(a);
    const ellipse = try ellipsis(a);
    defer ellipse.custom.deinit(a);
    const sequence = try tuple(a, &.{ binary, .{ .string = "text" }, point });
    defer sequence.custom.deinit(a);
    var inputs: [40]Value = undefined;
    for (&inputs, 0..) |*item, index| item.* = .{ .integer = @intCast(index * 8) };
    const members = try set(a, &inputs);
    defer members.custom.deinit(a);
    const frozen = try frozenSet(a, &.{ binary, sequence, ellipse });
    defer frozen.custom.deinit(a);
    const mapping = try dictionary(a, &.{
        .{ .key = sequence, .value = members },
        .{ .key = binary, .value = frozen },
        .{ .key = ellipse, .value = point },
    });
    defer mapping.custom.deinit(a);
    const copied = try ownership.cloneJsonValue(a, mapping);
    defer ownership.freeJsonValue(a, copied);
    try testing.expect(value_ops.pyEqual(mapping, copied));
    var growing_pairs: [27]PythonEntry = undefined;
    for ([_][]const u8{ "s", "t", "u" }, 0..) |text, index|
        growing_pairs[index] = .{ .key = .{ .string = text }, .value = .null };
    for (growing_pairs[3..], 0..) |*pair, index|
        pair.* = .{ .key = .{ .integer = @intCast(index) }, .value = .null };
    const growing = try dictionary(a, &growing_pairs);
    defer growing.custom.deinit(a);
    const growing_copy = try ownership.cloneJsonValue(a, growing);
    defer ownership.freeJsonValue(a, growing_copy);
    try testing.expect(value_ops.pyEqual(growing, growing_copy));
    const ordinary = try dictionary(a, &.{.{ .key = .{ .string = "x" }, .value = binary }});
    var storage = ordinary.object;
    defer storage.deinit(a);
}

test "Builtin construction and recursive clone release all partial allocations" {
    var stable = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable.allocator(), exerciseBuiltinOwnership, .{});
}
