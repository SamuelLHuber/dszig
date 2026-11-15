const std = @import("std");
const Allocator = std.mem.Allocator;
const ParseOptions = std.json.ParseOptions;
const ParseError = std.json.ParseError;
const Writer = std.Io.Writer;

pub const ObjectMap = std.array_hash_map.String(Value);
pub const Array = std.array_list.Managed(Value);
pub const JsonStyle = enum { compact, minified, indent2, orjson_indent2 };
pub const CustomError = error{ParseValue} || Allocator.Error;
pub const CustomWriteError = error{ ParseValue, JsonTooDeep } || Writer.Error;

/// Code's base annotation accepts concrete language-specialized instances.
/// This borrowed view avoids allocating a raw model dump to project its text.
pub const CodeMetadata = struct {
    language: []const u8,
    content: *const fn (*const anyopaque) []const u8,
};

pub const CustomVTable = struct {
    type_name: []const u8,
    model_name: []const u8,
    /// DSPy Type objects can supply bare text for a native tool result;
    /// ordinary models whose JSON serializer returns a string stay JSON-quoted.
    is_type: bool = false,
    /// Whether Python's object protocol admits this class as a dictionary key.
    /// Majority rejects mutable/unhashable models before comparing their values.
    hashable: bool = false,
    bool_value: ?*const fn (*const anyopaque) bool = null,
    code: ?CodeMetadata = null,
    raw_value: *const fn (Allocator, *const anyopaque) Allocator.Error!Value,
    serialized: *const fn (Allocator, *const anyopaque) CustomError!Value,
    write_str: *const fn (*Writer, *const anyopaque) CustomWriteError!void,
    write_repr: *const fn (*Writer, *const anyopaque) CustomWriteError!void,
    write_json: *const fn (*std.json.Stringify, *const anyopaque) Writer.Error!void,
    write_serialized_json: *const fn (*Writer, *const anyopaque, JsonStyle, usize) CustomWriteError!void,
    clone: *const fn (Allocator, *const anyopaque) Allocator.Error!Custom,
    deinit: *const fn (Allocator, *const anyopaque, bool) void,
    eql: *const fn (*const anyopaque, *const anyopaque) bool,
    /// Optional Python equality against a different value tag (e.g. Reasoning
    /// compares equal to its content string, but not to a lookalike dictionary).
    eql_value: ?*const fn (*const anyopaque, Value) bool = null,
};

/// A boxed native model. The box is always owned; owns_data describes its fields.
pub const Custom = struct {
    ptr: *const anyopaque,
    vtable: *const CustomVTable,
    owns_data: bool = false,

    pub fn get(self: Custom, comptime T: type) ?*const T {
        if (!std.mem.eql(u8, self.vtable.type_name, @typeName(T))) return null;
        return @ptrCast(@alignCast(self.ptr));
    }

    pub fn truthy(self: Custom) bool {
        return if (self.vtable.bool_value) |value| value(self.ptr) else true;
    }

    pub fn rawValue(self: Custom, allocator: Allocator) Allocator.Error!Value {
        return self.vtable.raw_value(allocator, self.ptr);
    }

    pub fn serialize(self: Custom, allocator: Allocator) CustomError!Value {
        return self.vtable.serialized(allocator, self.ptr);
    }

    pub fn writeStr(self: Custom, writer: *Writer) CustomWriteError!void {
        return self.vtable.write_str(writer, self.ptr);
    }

    pub fn writeRepr(self: Custom, writer: *Writer) CustomWriteError!void {
        return self.vtable.write_repr(writer, self.ptr);
    }

    pub fn writeSerializedJson(self: Custom, writer: *Writer, style: JsonStyle, depth: usize) CustomWriteError!void {
        return self.vtable.write_serialized_json(writer, self.ptr, style, depth);
    }

    pub fn clone(self: Custom, allocator: Allocator) Allocator.Error!Custom {
        const result = try self.vtable.clone(allocator, self.ptr);
        std.debug.assert(result.owns_data);
        std.debug.assert(std.mem.eql(u8, self.vtable.type_name, result.vtable.type_name));
        return result;
    }

    pub fn deinit(self: Custom, allocator: Allocator) void {
        self.vtable.deinit(allocator, self.ptr, self.owns_data);
    }

    pub fn eql(self: Custom, other: Custom) bool {
        if (!std.mem.eql(u8, self.vtable.type_name, other.vtable.type_name)) return false;
        return self.vtable.eql(self.ptr, other.ptr);
    }
};

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
            .number_string => |inner| try jws.print("{s}", .{inner}),
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
