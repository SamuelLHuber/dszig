const std = @import("std");
const Allocator = std.mem.Allocator;
const ParseOptions = std.json.ParseOptions;
const ParseError = std.json.ParseError;
const Writer = std.Io.Writer;

pub const ObjectMap = std.array_hash_map.String(Value);
pub const Array = std.array_list.Managed(Value);
pub const JsonStyle = enum { compact, minified, indent2, orjson_indent2 };
pub const CustomError = error{ParseValue} || Allocator.Error;
pub const SerializationError = CustomError || error{PythonSerializeFallback};
pub const CustomWriteError = error{ ParseValue, JsonTooDeep } || Writer.Error;

/// Code's base annotation accepts concrete language-specialized instances.
/// This borrowed view avoids allocating a raw model dump to project its text.
pub const CodeMetadata = struct {
    language: []const u8,
    content: *const fn (*const anyopaque) []const u8,
};

/// Borrowed builtin payloads preserve Python's non-JSON value identities.
pub const EnumValue = struct {
    /// Same nominal token as CustomVTable.type_identity, never a display name.
    class_identity: *const anyopaque,
    class_name: []const u8,
    member_name: []const u8,
    value: Value,
};
pub const PythonEntry = struct { key: Value, value: Value };
pub const PythonValue = union(enum) {
    bytes: []const u8,
    bytearray: []const u8,
    tuple: []const Value,
    set: []const Value,
    frozen_set: []const Value,
    complex: struct { real: f64, imag: f64 },
    decimal: @import("decimal.zig").Decimal,
    ellipsis: void,
    dictionary: []const PythonEntry,
};

/// Concrete decision model metadata, not a JSON-shape discriminator.
pub const DecisionKind = enum { noul, score, choice };
pub const DecisionMetadata = struct {
    kind: DecisionKind,
    field: *const fn (*const anyopaque, []const u8) ?Value,
    /// Static nominal ancestors, nearest parent first. Typed get() remains exact.
    ancestors: []const *const anyopaque = &.{},
};

/// Nominal identity cannot use @typeName: Zig elides struct-valued generic
/// arguments in that display string. Mutable per-type storage cannot be merged.
pub fn typeIdentity(comptime T: type) *const anyopaque {
    const Identity = struct {
        pub const Type = T;
        var token: u8 = 0;
    };
    return &Identity.token;
}

pub const CustomVTable = struct {
    type_name: []const u8,
    /// Exact native storage layout. Null never permits typed get().
    type_identity: ?*const anyopaque = null,
    /// An erased representation may retain a class without claiming its layout.
    /// Null uses type_identity; typed get() never consults this override.
    class_identity: ?*const anyopaque = null,
    model_name: []const u8,
    /// DSPy Type objects can supply bare text for a native tool result;
    /// ordinary models whose JSON serializer returns a string stay JSON-quoted.
    is_type: bool = false,
    /// Whether Python's object protocol admits this class as a dictionary key.
    /// Majority rejects mutable/unhashable models before comparing their values.
    hashable: bool = false,
    /// Classes with cross-value equality must supply a compatible Python hash.
    python_hash: ?*const fn (*const anyopaque, u32) error{ParseValue}!i64 = null,
    bool_value: ?*const fn (*const anyopaque) error{ParseValue}!bool = null,
    /// Python __float__, without numeric identity or numeric equality.
    float_value: ?*const fn (*const anyopaque) CustomError!f64 = null,
    /// Actual Python int-subclass payload, not an arbitrary __int__ conversion.
    /// The immutable borrowed result is .integer or canonical integer
    /// .number_string; it must never contain another custom object or a float.
    integer_value: ?*const fn (*const anyopaque) Value = null,
    decision: ?DecisionMetadata = null,
    code: ?CodeMetadata = null,
    python: ?*const fn (*const anyopaque) PythonValue = null,
    /// Python iteration of actual classes, separate from builtin identity.
    /// The returned slice borrows its elements and belongs to the allocator.
    iterate: ?*const fn (Allocator, *const anyopaque) CustomError![]Value = null,
    enumeration: ?*const fn (*const anyopaque) EnumValue = null,
    raw_value: *const fn (Allocator, *const anyopaque) Allocator.Error!Value,
    /// Complete strict JSON-mode projection; nested callbacks preserve depth.
    /// PythonSerializeFallback is handled only by serialize_for_json's root.
    serialized: *const fn (Allocator, *const anyopaque, u32) SerializationError!Value,
    write_str: *const fn (*Writer, *const anyopaque) CustomWriteError!void,
    write_repr: *const fn (*Writer, *const anyopaque, u32) CustomWriteError!void,
    write_json: *const fn (*std.json.Stringify, *const anyopaque) Writer.Error!void,
    write_serialized_json: *const fn (*Writer, *const anyopaque, JsonStyle, usize) CustomWriteError!void,
    clone: *const fn (*CloneContext, *const anyopaque) Allocator.Error!Custom,
    deinit: *const fn (Allocator, *const anyopaque, bool) void,
    eql: *const fn (*const anyopaque, *const anyopaque) CustomError!bool,
    /// Optional Python equality against a different value tag (e.g. Reasoning
    /// compares equal to its content string, but not to a lookalike dictionary).
    eql_value: ?*const fn (*const anyopaque, Value) CustomError!bool = null,
};

/// Native objects use their class's clone/free protocol. Model boxes are owned;
/// immutable class-owned singletons may retain their static box across clones.
/// owns_data describes whether ordinary model fields need deep destruction.
/// A borrowed handle owns neither its box nor the fields behind that box.
/// Memoized clones share one box: each owning handle holds one counted
/// reference, and the class destructor runs when the last one is released.
pub const Custom = struct {
    ptr: *const anyopaque,
    vtable: *const CustomVTable,
    owns_data: bool = false,
    owns_box: bool = true,
    /// Only owning handles hold a reference. Borrowing never increments it.
    shared: ?*SharedCustom = null,

    /// The source box and its fields must outlive every use of this handle.
    /// Clone still invokes the original class protocol and returns owned storage.
    pub fn borrowed(self: Custom) Custom {
        var result = self;
        result.owns_data = false;
        result.owns_box = false;
        result.shared = null;
        return result;
    }

    pub fn get(self: Custom, comptime T: type) ?*const T {
        if (self.vtable.type_identity != typeIdentity(T)) return null;
        return @ptrCast(@alignCast(self.ptr));
    }

    pub fn classIdentity(self: Custom) ?*const anyopaque {
        return self.vtable.class_identity orelse self.vtable.type_identity;
    }

    pub fn sameType(self: Custom, other: Custom) bool {
        if (self.classIdentity()) |identity| return other.classIdentity() == identity;
        return self.vtable == other.vtable;
    }

    pub fn truthy(self: Custom) error{ParseValue}!bool {
        if (self.vtable.bool_value) |value| return value(self.ptr);
        if (self.integerValue()) |integer| return switch (integer) {
            .integer => |number| number != 0,
            .number_string => |text| for (text) |byte| {
                if (byte >= '1' and byte <= '9') break true;
            } else false,
            else => unreachable,
        };
        return true;
    }

    /// Borrow the exact builtin integer base without discarding this object's
    /// class identity at storage, validation or serialization boundaries.
    pub fn integerValue(self: Custom) ?Value {
        const inspect = self.vtable.integer_value orelse return null;
        const result = inspect(self.ptr);
        std.debug.assert(result == .integer or result == .number_string);
        return result;
    }

    pub fn rawValue(self: Custom, allocator: Allocator) Allocator.Error!Value {
        return self.vtable.raw_value(allocator, self.ptr);
    }

    pub fn serialize(self: Custom, allocator: Allocator) SerializationError!Value {
        return self.vtable.serialized(allocator, self.ptr, 0);
    }

    pub fn writeStr(self: Custom, writer: *Writer) CustomWriteError!void {
        return self.vtable.write_str(writer, self.ptr);
    }

    pub fn writeRepr(self: Custom, writer: *Writer) CustomWriteError!void {
        return self.vtable.write_repr(writer, self.ptr, 0);
    }

    pub fn writeSerializedJson(self: Custom, writer: *Writer, style: JsonStyle, depth: usize) CustomWriteError!void {
        return self.vtable.write_serialized_json(writer, self.ptr, style, depth);
    }

    pub fn clone(self: Custom, allocator: Allocator) Allocator.Error!Custom {
        var context = CloneContext.init(allocator);
        defer context.deinit();
        return context.cloneCustom(self);
    }

    fn retain(self: Custom) Custom {
        std.debug.assert(self.owns_box);
        if (self.shared) |shared| {
            // The caller already holds a reference, so the count cannot reach
            // zero concurrently and the increment needs no ordering.
            const before = shared.references.fetchAdd(1, .monotonic);
            std.debug.assert(before > 0);
            std.debug.assert(before < std.math.maxInt(usize));
        }
        return self;
    }

    pub fn deinit(self: Custom, allocator: Allocator) void {
        if (!self.owns_box) {
            std.debug.assert(!self.owns_data);
            return;
        }
        if (self.shared) |shared| {
            std.debug.assert(shared.original.ptr == self.ptr);
            shared.release(allocator);
        } else {
            self.vtable.deinit(allocator, self.ptr, self.owns_data);
        }
    }

    pub fn eql(self: Custom, other: Custom) CustomError!bool {
        if (!self.sameType(other)) return false;
        // Two opaque layouts do not become compatible because both tokens are
        // null. Without a token, only the same vtable witnesses a shared layout.
        const same_layout = if (self.vtable.type_identity) |identity|
            other.vtable.type_identity == identity
        else
            self.vtable == other.vtable;
        if (!same_layout) {
            if (self.vtable.eql_value) |compare| return compare(self.ptr, .{ .custom = other });
            if (other.vtable.eql_value) |compare| return compare(other.ptr, .{ .custom = self });
            return false;
        }
        return self.vtable.eql(self.ptr, other.ptr);
    }
};

/// Heap-owned state retains the callback-produced handle, whose class destructor
/// runs once. It holds no allocator and no pointer into a temporary clone
/// context: as for every Custom, the releasing caller supplies the allocator
/// that cloned the box. Physical ptr/vtable/class identity is unchanged.
/// Aliases may be released from different threads, so the count is atomic. The
/// box behind it gains no synchronization of its own.
const SharedCustom = struct {
    original: Custom,
    references: std.atomic.Value(usize),

    fn release(self: *SharedCustom, allocator: Allocator) void {
        std.debug.assert(self.original.shared == null);
        std.debug.assert(self.original.owns_box);
        // The last releaser must observe every write made under other handles.
        const before = self.references.fetchSub(1, .acq_rel);
        std.debug.assert(before > 0);
        if (before == 1) {
            self.original.deinit(allocator);
            allocator.destroy(self);
        }
    }
};

/// Memoizes custom boxes across one tree clone. Native arrays and objects remain
/// independently owned trees; recursive container graph/cycle cloning is not implied.
/// Every source box must stay alive until deinit: the memo is keyed by source
/// address, so a box freed and reallocated mid-clone would alias a stranger.
pub const CloneContext = struct {
    allocator: Allocator,
    memo: std.AutoHashMapUnmanaged(Key, Custom) = .empty,

    const Key = struct {
        ptr: *const anyopaque,
        vtable: *const CustomVTable,
    };

    pub fn init(allocator: Allocator) CloneContext {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *CloneContext) void {
        var values = self.memo.valueIterator();
        while (values.next()) |custom| custom.deinit(self.allocator);
        self.memo.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn cloneCustom(self: *CloneContext, source: Custom) Allocator.Error!Custom {
        const key: Key = .{ .ptr = source.ptr, .vtable = source.vtable };
        if (self.memo.get(key)) |existing| return existing.retain();

        var result = try source.vtable.clone(self, source.ptr);
        errdefer result.deinit(self.allocator);
        std.debug.assert(result.owns_data);
        std.debug.assert(result.owns_box);
        std.debug.assert(source.sameType(result));
        // Class-owned immutable singletons already have stable lifetime and a
        // no-op destructor. Do not wrap their exact-source clone in heap state.
        const singleton = result.ptr == source.ptr and result.vtable == source.vtable;
        if (singleton) {
            // Every clone yields this same box, so it needs no memo entry either.
            std.debug.assert(result.shared == null);
            return result;
        }
        // A callback that delegated to this context already returned a counted
        // reference. A fresh box gets its one control block here.
        if (result.shared == null) {
            const shared = try self.allocator.create(SharedCustom);
            shared.* = .{ .original = result, .references = .init(1) };
            result.shared = shared;
        }
        // Insertion can allocate after child callbacks have grown the memo.
        // The callback result remains owned locally until insertion succeeds.
        try self.memo.putNoClobber(self.allocator, key, result);
        return result.retain();
    }

    pub fn cloneValue(self: *CloneContext, source: Value) Allocator.Error!Value {
        return switch (source) {
            .custom => |custom| .{ .custom = try self.cloneCustom(custom) },
            .string => |text| .{ .string = try self.allocator.dupe(u8, text) },
            .number_string => |text| .{ .number_string = try self.allocator.dupe(u8, text) },
            .array => |array| blk: {
                var out = Array.init(self.allocator);
                errdefer freeValue(self.allocator, .{ .array = out });
                try out.ensureTotalCapacity(array.items.len);
                for (array.items) |item|
                    out.appendAssumeCapacity(try self.cloneValue(item));
                break :blk .{ .array = out };
            },
            .object => |object| blk: {
                var out: ObjectMap = .empty;
                errdefer freeValue(self.allocator, .{ .object = out });
                try out.ensureTotalCapacity(self.allocator, object.count());
                var iterator = object.iterator();
                while (iterator.next()) |entry| {
                    const key = try self.allocator.dupe(u8, entry.key_ptr.*);
                    errdefer self.allocator.free(key);
                    const child = try self.cloneValue(entry.value_ptr.*);
                    out.putAssumeCapacity(key, child);
                }
                break :blk .{ .object = out };
            },
            else => source,
        };
    }
};

/// Releases a cloned native tree without depending on its former clone context.
pub fn freeValue(allocator: Allocator, input: Value) void {
    switch (input) {
        .custom => |custom| custom.deinit(allocator),
        .string, .number_string => |text| allocator.free(text),
        .array => |array| {
            var owned = array;
            for (owned.items) |item| freeValue(allocator, item);
            owned.deinit();
        },
        .object => |object| {
            var owned = object;
            var iterator = owned.iterator();
            while (iterator.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                freeValue(allocator, entry.value_ptr.*);
            }
            owned.deinit(allocator);
        },
        else => {},
    }
}

/// JSON values retain their usual representation; custom models retain class identity.
/// Parsed strings, keys, and number_string data are allocated like std.json.Value,
/// including when ParseOptions.allocate is alloc_if_needed. Leaky parsing requires
/// an arena because discarded duplicate values and parsed number text are retained.
pub const Value = union(enum) {
    null,
    bool: bool,
    integer: i64,
    float: f64,
    number_string: []const u8,
    string: []const u8,
    array: Array,
    object: ObjectMap,
    custom: Custom,

    pub fn parseFromNumberSlice(s: []const u8) Value {
        if (!std.json.isNumberFormattedLikeAnInteger(s)) {
            const f = std.fmt.parseFloat(f64, s) catch unreachable;
            return if (std.math.isFinite(f)) .{ .float = f } else .{ .number_string = s };
        }
        if (std.fmt.parseInt(i64, s, 10)) |i| {
            return .{ .integer = i };
        } else |err| switch (err) {
            error.Overflow => return .{ .number_string = s },
            error.InvalidCharacter => unreachable,
        }
    }

    pub fn dump(value: Value) void {
        const stderr = std.debug.lockStderr(&.{});
        defer std.debug.unlockStderr();
        std.json.Stringify.value(value, .{}, &stderr.file_writer.interface) catch return;
    }

    pub fn jsonStringify(value: Value, jws: *std.json.Stringify) Writer.Error!void {
        switch (value) {
            .null => try jws.write(null),
            .bool => |inner| try jws.write(inner),
            .integer => |inner| try jws.write(inner),
            .float => |inner| try jws.write(inner),
            .number_string => |inner| {
                if (std.json.isNumberFormattedLikeAnInteger(inner) and
                    inner.len - @intFromBool(inner.len > 0 and inner[0] == '-') > 4300)
                    return error.WriteFailed;
                try jws.print("{s}", .{inner});
            },
            .string => |inner| try jws.write(inner),
            .array => |inner| try jws.write(inner.items),
            .object => |inner| {
                try jws.beginObject();
                var it = inner.iterator();
                while (it.next()) |entry| {
                    try jws.objectField(entry.key_ptr.*);
                    try jws.write(entry.value_ptr.*);
                }
                try jws.endObject();
            },
            .custom => |inner| try inner.vtable.write_json(jws, inner.ptr),
        }
    }

    pub fn jsonParse(allocator: Allocator, source: anytype, options: ParseOptions) ParseError(@TypeOf(source.*))!Value {
        // The stack grammar is (.array | .object .string)*. It keeps parsing
        // iterative, including deeply nested JSON, as in the pinned std parser.
        var stack = Array.init(allocator);
        defer stack.deinit();

        while (true) {
            std.debug.assert(stack.items.len == 0 or
                stack.items[stack.items.len - 1] == .array or
                (stack.items.len >= 2 and stack.items[stack.items.len - 2] == .object and
                    stack.items[stack.items.len - 1] == .string));

            // std.json.Value always owns dynamic strings, even if the enclosing
            // parse options permit other Zig types to borrow input strings.
            switch (try source.nextAllocMax(allocator, .alloc_always, options.max_value_len.?)) {
                .allocated_string => |s| {
                    return try handleCompleteValue(&stack, allocator, source, .{ .string = s }, options) orelse continue;
                },
                .allocated_number => |s| {
                    const value: Value = if (options.parse_numbers) parseFromNumberSlice(s) else .{ .number_string = s };
                    return try handleCompleteValue(&stack, allocator, source, value, options) orelse continue;
                },
                .null => return try handleCompleteValue(&stack, allocator, source, .null, options) orelse continue,
                .true => return try handleCompleteValue(&stack, allocator, source, .{ .bool = true }, options) orelse continue,
                .false => return try handleCompleteValue(&stack, allocator, source, .{ .bool = false }, options) orelse continue,
                .object_begin => {
                    switch (try source.nextAllocMax(allocator, .alloc_always, options.max_value_len.?)) {
                        .object_end => return try handleCompleteValue(&stack, allocator, source, .{ .object = .empty }, options) orelse continue,
                        .allocated_string => |key| {
                            try stack.appendSlice(&.{
                                .{ .object = .empty },
                                .{ .string = key },
                            });
                        },
                        else => unreachable,
                    }
                },
                .array_begin => try stack.append(.{ .array = .init(allocator) }),
                .array_end => return try handleCompleteValue(&stack, allocator, source, stack.pop().?, options) orelse continue,
                else => unreachable,
            }
        }
    }
};

fn handleCompleteValue(
    stack: *Array,
    allocator: Allocator,
    source: anytype,
    value_: Value,
    options: ParseOptions,
) ParseError(@TypeOf(source.*))!?Value {
    if (stack.items.len == 0) return value_;
    var value = value_;
    while (true) {
        std.debug.assert(stack.items[stack.items.len - 1] == .array or
            (stack.items.len >= 2 and stack.items[stack.items.len - 2] == .object and
                stack.items[stack.items.len - 1] == .string));
        switch (stack.items[stack.items.len - 1]) {
            .string => |key| {
                _ = stack.pop();
                const object = &stack.items[stack.items.len - 1].object;
                const gop = try object.getOrPut(allocator, key);
                if (gop.found_existing) {
                    switch (options.duplicate_field_behavior) {
                        .use_first => {},
                        .@"error" => return error.DuplicateField,
                        .use_last => {
                            gop.value_ptr.* = value;
                        },
                    }
                } else {
                    gop.value_ptr.* = value;
                }

                // Consume the next key before restoring the stack invariant.
                switch (try source.nextAllocMax(allocator, .alloc_always, options.max_value_len.?)) {
                    .object_end => {
                        value = stack.pop().?;
                        if (stack.items.len == 0) return value;
                        continue;
                    },
                    .allocated_string => |next_key| {
                        try stack.append(.{ .string = next_key });
                        return null;
                    },
                    else => unreachable,
                }
            },
            .array => |*array| {
                try array.append(value);
                return null;
            },
            else => unreachable,
        }
    }
}
