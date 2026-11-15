//! DSPy 3.5 decision classes. Runtime boxes retain the concrete class; plain
//! dictionaries never acquire model identity. Constructors/decoders borrow
//! storage from their arena; clone() and owning runtime clones own every field.
//!
//! Factories are finite compile-time declarations. Python's mutable runtime
//! subclasses and its 256-entry LRU eviction/recreation identity are not modeled.
const std = @import("std");
const runtime = @import("runtime_value.zig");
const values = @import("value.zig");
const types = @import("types.zig");
const python = @import("python_value.zig");
const pyjson = @import("pyjson.zig");
const inference = @import("inference.zig");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Value = runtime.Value;
const Integer = @import("integer.zig").Integer;

pub const Kind = runtime.DecisionKind;
pub const ModelInfo = struct {
    kind: Kind,
    criteria_json: []const u8,
    labels: []const []const u8,
    values: []const Value,
    validate: *const fn (Allocator, Value) runtime.CustomError!Value,
};
pub const Option = struct { value: Value, description_json: []const u8 };
pub const ValidationMode = enum { declared, strict, lax };

const Configuration = struct {
    kind: Kind,
    configured: bool = false,
    criteria_json: []const u8 = "null",
    model_name: []const u8,
    description: []const u8,
    labels: []const []const u8 = &.{},
    members: []const Value = &.{},
};

pub const Noul = DecisionType(.{
    .kind = .noul,
    .model_name = "Noul",
    .description = "A Boolean decision.",
    .labels = &.{ "true", "false" },
    .members = &.{ .{ .bool = true }, .{ .bool = false } },
});
pub const Score = DecisionType(.{
    .kind = .score,
    .model_name = "Score",
    .description = "Continuous score from 0 to -1; intermediate values are allowed. Rubric (level index, description): [].",
});
pub const Choice = DecisionType(.{
    .kind = .choice,
    .model_name = "Choice",
    .description = "Select exactly one of the declared Literal values.",
});

pub fn isDecision(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "dszig_decision_type");
}

/// Native routing is DIRECT only. Bare Score/Choice have no declared answer
/// space and are ordinary models, just as decision_type() treats them upstream.
pub fn modelInfo(comptime T: type) ?ModelInfo {
    if (comptime !isDecision(T)) return null;
    if (T.kind != .noul and !T.configured) return null;
    return .{
        .kind = T.kind,
        .criteria_json = T.criteria_json,
        .labels = T.labels,
        .values = T.members,
        .validate = T.validate,
    };
}

/// Custom-type discovery recurses through annotation arguments, not fields.
pub fn annotationType(comptime T: type) ?type {
    if (isDecision(T)) return T;
    if (comptime types.isDict(T)) return annotationType(T.dszig_dict_value);
    return switch (@typeInfo(T)) {
        .optional => |o| annotationType(o.child),
        .pointer => |p| if (p.size == .slice and T != []const u8 and T != []u8) annotationType(p.child) else null,
        else => null,
    };
}

pub fn NoulWithCriteria(comptime options: []const Option) type {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        if (options.len == 0 or options.len > 2) @compileError("dszig: Noul requires distinct True/False option pairs");
        var sorted: [options.len]Option = options[0..options.len].*;
        for (sorted, 0..) |option, i| {
            if (option.value != .bool) @compileError("dszig: Noul criteria require exact bool values");
            for (sorted[0..i]) |previous| if (previous.value.bool == option.value.bool)
                @compileError("dszig: Noul criteria require distinct True/False values");
        }
        if (sorted.len == 2 and !sorted[0].value.bool) std.mem.swap(Option, &sorted[0], &sorted[1]);
        var criteria_json: []const u8 = "{";
        var name: []const u8 = "Noul[";
        for (sorted, 0..) |option, i| {
            const desc = descriptionSnapshot(option.description_json);
            criteria_json = criteria_json ++ (if (i == 0) "" else ", ") ++
                (if (option.value.bool) "\"true\": " else "\"false\": ") ++ desc.json;
            name = name ++ (if (i == 0) "" else ", ") ++ "(" ++ scalarText(option.value, .repr) ++ ", " ++ desc.repr ++ ")";
        }
        criteria_json = criteria_json ++ "}";
        break :blk DecisionType(.{
            .kind = .noul,
            .configured = true,
            .criteria_json = criteria_json,
            .model_name = name ++ "]",
            .description = "A Boolean decision. Criteria: " ++ criteria_json ++ ".",
            .labels = &.{ "true", "false" },
            .members = &.{ .{ .bool = true }, .{ .bool = false } },
        });
    };
}

pub fn ScoreWithLevels(comptime descriptions_json: []const u8) type {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        const levels = jsonSnapshot(descriptions_json);
        if (levels.shape != .array or levels.items.len < 2)
            @compileError("dszig: Score requires a JSON array of at least two ordered descriptions");
        var name: []const u8 = "Score[";
        var rubric: []const u8 = "[";
        var label_array: [levels.items.len][]const u8 = undefined;
        var member_array: [levels.items.len]Value = undefined;
        for (levels.items, 0..) |level, i| {
            if (!level.isDescription() or level.shape == .null)
                @compileError("dszig: Score descriptions must be non-null JSON strings, objects, or arrays");
            label_array[i] = std.fmt.comptimePrint("{d}", .{i});
            member_array[i] = .{ .integer = i };
            name = name ++ (if (i == 0) "" else ", ") ++ level.repr;
            rubric = rubric ++ (if (i == 0) "" else ", ") ++ "[" ++ label_array[i] ++ ", " ++ level.json ++ "]";
        }
        const final_labels = label_array;
        const final_members = member_array;
        break :blk DecisionType(.{
            .kind = .score,
            .configured = true,
            .criteria_json = levels.json,
            .model_name = name ++ "]",
            .labels = &final_labels,
            .members = &final_members,
            .description = std.fmt.comptimePrint("Continuous score from 0 to {d}; intermediate values are allowed. Rubric (level index, description): ", .{levels.items.len - 1}) ++ rubric ++ "].",
        });
    };
}

pub fn ChoiceWithOptions(comptime options: []const Option) type {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        if (options.len == 0) @compileError("dszig: Choice requires at least one option pair");
        var label_array: [options.len][]const u8 = undefined;
        var member_array: [options.len]Value = undefined;
        var criteria_json: []const u8 = "{";
        var name: []const u8 = "Choice[";
        var descriptions: []const u8 = "[";
        for (options, 0..) |option, i| {
            const member = optionValue(option.value);
            const label = scalarText(member, .str);
            for (label_array[0..i]) |previous| if (std.mem.eql(u8, previous, label))
                @compileError("dszig: Choice option values must have distinct Python string labels");
            const desc = descriptionSnapshot(option.description_json);
            label_array[i] = label;
            member_array[i] = member;
            criteria_json = criteria_json ++ (if (i == 0) "" else ", ") ++ scalarText(.{ .string = label }, .json) ++ ": " ++ desc.json;
            name = name ++ (if (i == 0) "" else ", ") ++ "(" ++ scalarText(member, .repr) ++ ", " ++ desc.repr ++ ")";
            descriptions = descriptions ++ (if (i == 0) "" else ", ") ++ "[" ++ scalarText(member, .json) ++ ", " ++ desc.json ++ "]";
        }
        const final_labels = label_array;
        const final_members = member_array;
        break :blk DecisionType(.{
            .kind = .choice,
            .configured = true,
            .criteria_json = criteria_json ++ "}",
            .model_name = name ++ "]",
            .labels = &final_labels,
            .members = &final_members,
            .description = "Select exactly one value. Options: " ++ descriptions ++ "].",
        });
    };
}

fn optionValue(comptime input: Value) Value {
    return switch (input) {
        .null, .bool, .integer, .string => input,
        .number_string => |s| blk: {
            if (std.mem.indexOfAny(u8, s, ".eE") != null) @compileError("dszig: Choice values must be exact str/int/bool/None");
            const snapshot = jsonSnapshot(s);
            if (snapshot.shape != .number) @compileError("dszig: invalid Choice integer");
            break :blk if (std.fmt.parseInt(i64, snapshot.json, 10)) |small| .{ .integer = small } else |_| .{ .number_string = snapshot.json };
        },
        else => @compileError("dszig: Choice values must be exact str/int/bool/None"),
    };
}

fn DecisionType(comptime config: Configuration) type {
    return struct {
        const Self = @This();
        pub const dszig_decision_type = true;
        pub const kind = config.kind;
        pub const configured = config.configured;
        pub const criteria_json = config.criteria_json;
        pub const model_name = config.model_name;
        pub const schema_name = definitionName(model_name);
        pub const description = config.description;
        pub const labels = config.labels;
        pub const members = config.members;
        pub const ValueType = switch (kind) {
            .noul => bool,
            .score => f64,
            .choice => Value,
        };
        pub const LiteralType = if (kind == .choice and configured) @import("literal.zig").Literal(members) else void;
        pub const schema_description_json = if (kind == .score)
            (if (configured) "" else scalarText(.{ .string = score_schema_description }, .json))
        else
            scalarText(.{ .string = description }, .json);
        pub const field_names: []const []const u8 = switch (kind) {
            .noul => &.{ "confidence", "value", "probability" },
            .score => &.{ "confidence", "value", "probabilities", "level" },
            .choice => &.{ "confidence", "value", "probabilities" },
        };

        confidence: f64,
        value: ValueType,
        probability: if (kind == .noul) ?f64 else void = if (kind == .noul) null else {},
        /// Score uses genuine Python int-keyed dictionaries; Choice uses string
        /// keys. This is evidence, not a second representation of the model.
        probabilities: if (kind == .noul) void else ?Value = if (kind == .noul) {} else null,
        level: if (kind == .score) ?Integer else void = if (kind == .score) null else {},
        runtime_origin: ?runtime.Custom = null,

        pub fn criteria(arena: Allocator) runtime.CustomError!Value {
            if (comptime std.mem.eql(u8, criteria_json, "null")) return .null;
            return @import("json_repair.zig").parseJson(arena, criteria_json) catch |err| switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseValue,
            };
        }

        pub fn fromValue(arena: Allocator, input: Value) runtime.CustomError!Self {
            return decode(arena, input, .declared, false);
        }
        pub fn fromXmlValue(arena: Allocator, input: Value) runtime.CustomError!Self {
            return decode(arena, input, .lax, false);
        }
        pub fn validate(arena: Allocator, input: Value) runtime.CustomError!Value {
            return validateMode(arena, input, .declared);
        }
        pub fn validateStrict(arena: Allocator, input: Value) runtime.CustomError!Value {
            return validateMode(arena, input, .strict);
        }
        pub fn validateLax(arena: Allocator, input: Value) runtime.CustomError!Value {
            return validateMode(arena, input, .lax);
        }
        pub fn validateMode(arena: Allocator, input: Value, mode: ValidationMode) runtime.CustomError!Value {
            return (try decode(arena, input, mode, false)).toRuntime(arena);
        }
        pub fn validateJson(arena: Allocator, text: []const u8, mode: ValidationMode) runtime.CustomError!Value {
            const input = @import("json_repair.zig").parseJson(arena, text) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseValue,
            };
            return (try decode(arena, input, mode, true)).toRuntime(arena);
        }

        fn decode(arena: Allocator, input: Value, mode: ValidationMode, json_input: bool) runtime.CustomError!Self {
            if (input == .custom) {
                if (input.custom.get(Self)) |instance| {
                    var result = instance.*;
                    result.runtime_origin = input.custom;
                    return result;
                }
                if (comptime !configured) {
                    if (input.custom.vtable.decision) |metadata| {
                        if (metadata.kind != kind) return error.ParseValue;
                        var result = try projectInstance(metadata, input.custom.ptr);
                        result.runtime_origin = input.custom;
                        return result;
                    }
                }
            }
            try checkFields(input, field_names);
            const confidence = try probabilityNumber(arena, getInput(input, "confidence") orelse return error.ParseValue, mode);
            const raw_value = getInput(input, "value") orelse return error.ParseValue;
            var result: Self = .{ .confidence = confidence, .value = undefined };
            switch (kind) {
                .noul => {
                    result.value = if (mode == .lax)
                        (try types.validate(bool, arena, raw_value)).bool
                    else if (raw_value == .bool) raw_value.bool else return error.ParseValue;
                    const evidence = getInput(input, "probability") orelse .null;
                    result.probability = if (evidence == .null) null else try probabilityNumber(arena, evidence, mode);
                },
                .score => {
                    result.value = try number(arena, raw_value, mode);
                    if (!std.math.isFinite(result.value)) return error.ParseValue;
                    if (comptime configured) {
                        if (result.value < 0 or result.value > @as(f64, @floatFromInt(members.len - 1))) return error.ParseValue;
                    }
                    const evidence = getInput(input, "probabilities") orelse .null;
                    result.probabilities = if (evidence == .null) null else try probabilityMap(arena, evidence, .score, mode, json_input);
                    const level = getInput(input, "level") orelse .null;
                    if (level != .null) {
                        const integer = if (mode == .lax) try values.coerceInteger(arena, level) else try strictInteger(level);
                        if (!try nonnegativeInteger(integer)) return error.ParseValue;
                        if (comptime configured) {
                            if (!(values.floatAtLeast(@floatFromInt(members.len - 1), integer) catch return error.ParseValue)) return error.ParseValue;
                        }
                        result.level = .{ .value = integer };
                    }
                },
                .choice => {
                    result.value = if (comptime configured)
                        try LiteralType.validate(arena, raw_value)
                    else
                        try bareChoiceValue(arena, raw_value, mode);
                    const evidence = getInput(input, "probabilities") orelse .null;
                    result.probabilities = if (evidence == .null) null else try probabilityMap(arena, evidence, .choice, mode, json_input);
                },
            }
            return result;
        }

        fn projectInstance(metadata: runtime.DecisionMetadata, ptr: *const anyopaque) runtime.CustomError!Self {
            const confidence = metadata.field(ptr, "confidence") orelse return error.ParseValue;
            const value = metadata.field(ptr, "value") orelse return error.ParseValue;
            if (confidence != .float) return error.ParseValue;
            var result: Self = .{ .confidence = confidence.float, .value = undefined };
            switch (kind) {
                .noul => {
                    if (value != .bool) return error.ParseValue;
                    result.value = value.bool;
                    const probability = metadata.field(ptr, "probability") orelse .null;
                    result.probability = if (probability == .null) null else if (probability == .float) probability.float else return error.ParseValue;
                },
                .score => {
                    if (value != .float) return error.ParseValue;
                    result.value = value.float;
                    const probabilities = metadata.field(ptr, "probabilities") orelse .null;
                    result.probabilities = if (probabilities == .null) null else probabilities;
                    const level = metadata.field(ptr, "level") orelse .null;
                    result.level = if (level == .null) null else .{ .value = level };
                },
                .choice => {
                    result.value = value;
                    const probabilities = metadata.field(ptr, "probabilities") orelse .null;
                    result.probabilities = if (probabilities == .null) null else probabilities;
                },
            }
            return result;
        }

        pub fn fieldValue(self: Self, comptime name: []const u8) Value {
            if (comptime std.mem.eql(u8, name, "confidence")) return .{ .float = self.confidence };
            if (comptime std.mem.eql(u8, name, "value")) return switch (kind) {
                .noul => .{ .bool = self.value },
                .score => .{ .float = self.value },
                .choice => self.value,
            };
            if (comptime kind == .noul) {
                if (comptime std.mem.eql(u8, name, "probability")) return if (self.probability) |p| .{ .float = p } else .null;
            } else {
                if (comptime std.mem.eql(u8, name, "probabilities")) return self.probabilities orelse .null;
            }
            if (comptime kind == .score) {
                if (comptime std.mem.eql(u8, name, "level")) return if (self.level) |level| level.value else .null;
            }
            @compileError("dszig: unknown decision field " ++ name);
        }

        /// model_dump/round_trip in Python mode: nullable fields are omitted,
        /// except value=None. Raw iteration below deliberately keeps all fields.
        pub fn toValue(self: Self, arena: Allocator) Allocator.Error!Value {
            if (self.runtime_origin) |origin| return origin.rawValue(arena);
            var object: runtime.ObjectMap = .empty;
            try object.ensureTotalCapacity(arena, field_names.len);
            inline for (field_names) |name| {
                const field = self.fieldValue(name);
                if (field != .null or comptime std.mem.eql(u8, name, "value")) object.putAssumeCapacity(name, field);
            }
            return .{ .object = object };
        }

        pub fn toRuntime(self: Self, arena: Allocator) Allocator.Error!Value {
            if (self.runtime_origin) |origin| return .{ .custom = origin };
            const box = try arena.create(Self);
            box.* = self;
            return .{ .custom = .{ .ptr = box, .vtable = &runtime_vtable } };
        }

        /// Decisions override serialization, NOT Type.format(). Calling this
        /// inherited operation has no implementation in the actual Source.
        pub fn format(_: Self) error{NotImplemented}!Value {
            return error.NotImplemented;
        }
        pub fn truthy(self: Self) bool {
            return if (comptime kind == .noul) self.value else true;
        }
        pub fn float(self: Self) error{ParseValue}!f64 {
            return if (comptime kind == .score) self.value else error.ParseValue;
        }
        pub fn eql(self: Self, other: Self) runtime.CustomError!bool {
            if (self.runtime_origin) |left| {
                if (other.runtime_origin) |right| return left.eql(right);
                if (!std.mem.eql(u8, left.vtable.type_name, @typeName(Self))) return false;
            } else if (other.runtime_origin) |right| {
                if (!std.mem.eql(u8, right.vtable.type_name, @typeName(Self))) return false;
            }
            inline for (field_names) |name| if (!try values.pyEqual(self.fieldValue(name), other.fieldValue(name))) return false;
            return true;
        }

        pub const Field = struct { name: []const u8, value: Value };
        pub const Iterator = struct {
            model: *const Self,
            index: usize = 0,
            pub fn next(self: *Iterator) ?Field {
                inline for (field_names, 0..) |name, index| {
                    if (self.index == index) {
                        self.index += 1;
                        return .{ .name = name, .value = self.model.fieldValue(name) };
                    }
                }
                return null;
            }
        };
        pub fn iterator(self: *const Self) Iterator {
            return .{ .model = self };
        }

        pub fn writeStr(self: Self, writer: *Writer) runtime.CustomWriteError!void {
            if (self.runtime_origin) |origin| return origin.writeStr(writer);
            return self.writeDisplay(writer, false, 0);
        }
        pub fn writeRepr(self: Self, writer: *Writer) runtime.CustomWriteError!void {
            if (self.runtime_origin) |origin| return origin.writeRepr(writer);
            return self.writeDisplay(writer, true, 0);
        }
        fn writeDisplay(self: Self, writer: *Writer, repr: bool, depth: u32) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            if (repr) try writer.writeAll(model_name ++ "(");
            inline for (field_names, 0..) |name, index| {
                if (index != 0) try writer.writeAll(if (repr) ", " else " ");
                try writer.writeAll(name ++ "=");
                try pyjson.writeRepr(writer, self.fieldValue(name), depth + 1);
            }
            if (repr) try writer.writeByte(')');
        }

        pub fn jsonStringify(self: Self, writer: *std.json.Stringify) Writer.Error!void {
            if (self.runtime_origin) |origin| return origin.vtable.write_json(writer, origin.ptr);
            try writer.beginObject();
            inline for (field_names) |name| {
                const field = self.fieldValue(name);
                if (field != .null or comptime std.mem.eql(u8, name, "value")) {
                    try writer.objectField(name);
                    if (comptime kind != .noul and std.mem.eql(u8, name, "probabilities")) {
                        try writeProbabilityJson(writer, field);
                    } else {
                        try writer.write(field);
                    }
                }
            }
            try writer.endObject();
        }

        /// Owning clones are independent of both the original arena and any
        /// borrowed base projection. deinit() is ONLY for such owning clones.
        pub fn clone(self: Self, allocator: Allocator) Allocator.Error!Self {
            if (self.runtime_origin) |origin| {
                const owned = try origin.clone(allocator);
                // The same valid class was already projected by fromValue.
                var result = projectInstance(owned.vtable.decision.?, owned.ptr) catch unreachable;
                result.runtime_origin = owned;
                return result;
            }
            var result = self;
            result.runtime_origin = null;
            if (comptime kind == .choice) result.value = .null;
            if (comptime kind != .noul) result.probabilities = null;
            if (comptime kind == .score) result.level = null;
            errdefer result.deinit(allocator);
            if (comptime kind == .choice) result.value = try inference.cloneJsonValue(allocator, self.value);
            if (comptime kind != .noul) {
                if (self.probabilities) |map| result.probabilities = try inference.cloneJsonValue(allocator, map);
            }
            if (comptime kind == .score) {
                if (self.level) |level| result.level = .{ .value = try inference.cloneJsonValue(allocator, level.value) };
            }
            return result;
        }
        pub fn deinit(self: Self, allocator: Allocator) void {
            if (self.runtime_origin) |origin| {
                origin.deinit(allocator);
                return;
            }
            if (comptime kind == .choice) inference.freeJsonValue(allocator, self.value);
            if (comptime kind != .noul) {
                if (self.probabilities) |map| inference.freeJsonValue(allocator, map);
            }
            if (comptime kind == .score) {
                if (self.level) |level| inference.freeJsonValue(allocator, level.value);
            }
        }

        const runtime_vtable: runtime.CustomVTable = .{
            .type_name = @typeName(Self),
            .model_name = model_name,
            .is_type = true,
            .type_identity = runtime.typeIdentity(Self),
            .hashable = false,
            .bool_value = runtimeBool,
            .float_value = if (kind == .score) runtimeFloat else null,
            .decision = .{ .kind = kind, .field = runtimeField },
            .iterate = iterateRuntime,
            .raw_value = rawRuntimeValue,
            .serialized = serializedRuntimeValue,
            .write_str = writeRuntimeStr,
            .write_repr = writeRuntimeRepr,
            .write_json = writeRuntimeJson,
            .write_serialized_json = writeRuntimeSerializedJson,
            .clone = cloneRuntime,
            .deinit = deinitRuntime,
            .eql = equalRuntime,
        };
        fn actual(ptr: *const anyopaque) *const Self {
            return @ptrCast(@alignCast(ptr));
        }
        fn runtimeBool(ptr: *const anyopaque) bool {
            return actual(ptr).truthy();
        }
        fn runtimeFloat(ptr: *const anyopaque) runtime.CustomError!f64 {
            return actual(ptr).float();
        }
        fn runtimeField(ptr: *const anyopaque, name: []const u8) ?Value {
            inline for (field_names) |field| if (std.mem.eql(u8, name, field)) return actual(ptr).fieldValue(field);
            return null;
        }
        fn iterateRuntime(arena: Allocator, ptr: *const anyopaque) runtime.CustomError![]Value {
            const result = try arena.alloc(Value, field_names.len);
            var initialized: usize = 0;
            errdefer {
                for (result[0..initialized]) |item| item.custom.deinit(arena);
                arena.free(result);
            }
            inline for (field_names, 0..) |name, index| {
                const pair = try arena.dupe(Value, &.{ .{ .string = name }, actual(ptr).fieldValue(name) });
                errdefer arena.free(pair);
                result[index] = try python.tupleFromOwnedSlice(arena, pair);
                initialized += 1;
            }
            return result;
        }
        fn rawRuntimeValue(arena: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return actual(ptr).toValue(arena);
        }
        fn serializedRuntimeValue(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
            if (depth >= pyjson.max_depth) return error.ParseValue;
            var object: runtime.ObjectMap = .empty;
            try object.ensureTotalCapacity(arena, field_names.len);
            inline for (field_names) |name| {
                const raw = actual(ptr).fieldValue(name);
                if (raw != .null or comptime std.mem.eql(u8, name, "value")) {
                    const field = if (comptime kind != .noul and std.mem.eql(u8, name, "probabilities"))
                        try serializedProbabilities(arena, raw, depth + 1)
                    else
                        try values.serializedValueDepth(arena, raw, depth + 1);
                    object.putAssumeCapacity(name, field);
                }
            }
            return .{ .object = object };
        }
        fn writeRuntimeStr(writer: *Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            try actual(ptr).writeStr(writer);
        }
        fn writeRuntimeRepr(writer: *Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
            try actual(ptr).writeDisplay(writer, true, depth);
        }
        fn writeRuntimeJson(writer: *std.json.Stringify, ptr: *const anyopaque) Writer.Error!void {
            try writer.write(actual(ptr).*);
        }
        fn writeRuntimeSerializedJson(writer: *Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
            if (depth >= pyjson.max_depth) return error.JsonTooDeep;
            try writer.writeByte('{');
            var first = true;
            inline for (field_names) |name| {
                const field = actual(ptr).fieldValue(name);
                if (field != .null or comptime std.mem.eql(u8, name, "value")) {
                    try separator(writer, style, depth + 1, first);
                    first = false;
                    try writer.writeAll("\"" ++ name);
                    try writer.writeAll(if (style == .minified) "\":" else "\": ");
                    if (comptime kind != .noul and std.mem.eql(u8, name, "probabilities")) {
                        try writeProbabilities(writer, field, style, depth + 1);
                    } else {
                        try pyjson.writeJsonProjection(writer, field, style, @intCast(depth + 1));
                    }
                }
            }
            if (!first and (style == .indent2 or style == .orjson_indent2)) {
                try writer.writeByte('\n');
                try writer.splatByteAll(' ', depth * 2);
            }
            try writer.writeByte('}');
        }
        fn cloneRuntime(allocator: Allocator, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
            const copy = try allocator.create(Self);
            errdefer allocator.destroy(copy);
            copy.* = try actual(ptr).clone(allocator);
            return .{ .ptr = copy, .vtable = &runtime_vtable, .owns_data = true };
        }
        fn deinitRuntime(allocator: Allocator, ptr: *const anyopaque, owns_data: bool) void {
            if (owns_data) actual(ptr).deinit(allocator);
            allocator.destroy(actual(ptr));
        }
        fn equalRuntime(left: *const anyopaque, right: *const anyopaque) runtime.CustomError!bool {
            return left == right or try actual(left).eql(actual(right).*);
        }
    };
}

fn getInput(input: Value, name: []const u8) ?Value {
    if (input == .object) return input.object.get(name);
    if (python.view(input)) |view| {
        if (view == .dictionary) {
            for (view.dictionary) |entry| if (entry.key == .string and std.mem.eql(u8, entry.key.string, name)) return entry.value;
        }
    }
    return null;
}
fn checkFields(input: Value, allowed: []const []const u8) error{ParseValue}!void {
    if (input == .object) {
        for (input.object.keys()) |key| {
            for (allowed) |name| {
                if (std.mem.eql(u8, name, key)) break;
            } else return error.ParseValue;
        }
        return;
    }
    const view = python.view(input) orelse return error.ParseValue;
    if (view != .dictionary) return error.ParseValue;
    for (view.dictionary) |entry| {
        if (entry.key != .string) return error.ParseValue;
        for (allowed) |name| {
            if (std.mem.eql(u8, name, entry.key.string)) break;
        } else return error.ParseValue;
    }
}
fn number(arena: Allocator, input: Value, mode: ValidationMode) runtime.CustomError!f64 {
    if (input == .custom) {
        if (input.custom.vtable.float_value) |convert| return convert(input.custom.ptr);
    }
    if (mode == .strict and input != .float and input != .integer and input != .number_string) return error.ParseValue;
    return (try types.validate(f64, arena, input)).float;
}
fn probabilityNumber(arena: Allocator, input: Value, mode: ValidationMode) runtime.CustomError!f64 {
    const result = try number(arena, input, mode);
    if (!(result >= 0 and result <= 1)) return error.ParseValue;
    return result;
}
fn strictInteger(input: Value) error{ParseValue}!Value {
    return switch (input) {
        .integer => input,
        .number_string => values.parseInteger(input),
        else => error.ParseValue,
    };
}
fn nonnegativeInteger(input: Value) runtime.CustomError!bool {
    return switch (input) {
        .integer => |i| i >= 0,
        .number_string => |s| s.len > 0 and (s[0] != '-' or try values.pyEqual(input, .{ .integer = 0 })),
        else => false,
    };
}
fn bareChoiceValue(arena: Allocator, input: Value, mode: ValidationMode) runtime.CustomError!Value {
    switch (input) {
        .null, .bool, .integer, .string => return input,
        .number_string => return strictInteger(input),
        else => {},
    }
    if (input == .custom) {
        if (input.custom.vtable.enumeration != null) return types.validate([]const u8, arena, input);
        if (python.view(input)) |view| {
            if ((view == .bytes or view == .bytearray) and mode != .strict)
                return types.validate([]const u8, arena, input);
        }
    }
    if (mode == .strict) return error.ParseValue;
    // The bare annotation is a union: failure of the integer branch must
    // still try its bool branch, which has distinct Decimal float coercion.
    return values.coerceInteger(arena, input) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => types.validate(bool, arena, input),
    };
}
test "bare Choice tries bool after exact Decimal integer coercion refuses" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const input = try python.decimal(a, "0.9999999999999999999999999999");
    try std.testing.expectError(error.ParseValue, values.coerceInteger(a, input));
    const default = try bareChoiceValue(a, input, .declared);
    try std.testing.expect(default == .bool and default.bool);
    const lax = try bareChoiceValue(a, input, .lax);
    try std.testing.expect(lax == .bool and lax.bool);
    try std.testing.expectError(error.ParseValue, bareChoiceValue(a, input, .strict));
}

fn probabilityMap(arena: Allocator, input: Value, kind: Kind, mode: ValidationMode, json_input: bool) runtime.CustomError!Value {
    const pairs = if (input == .object) blk: {
        const result = try arena.alloc(runtime.PythonEntry, input.object.count());
        for (input.object.keys(), input.object.values(), result) |key, value, *out| out.* = .{ .key = .{ .string = key }, .value = value };
        break :blk result;
    } else blk: {
        const view = python.view(input) orelse return error.ParseValue;
        if (view != .dictionary) return error.ParseValue;
        break :blk try arena.dupe(runtime.PythonEntry, view.dictionary);
    };
    defer arena.free(pairs);
    for (pairs) |*entry| {
        entry.key = if (kind == .score)
            if (mode == .strict and !json_input) try strictInteger(entry.key) else try values.coerceInteger(arena, entry.key)
        else if (mode == .strict and entry.key != .string)
            return error.ParseValue
        else
            try types.validate([]const u8, arena, entry.key);
        const probability = try probabilityNumber(arena, entry.value, mode);
        entry.value = .{ .float = probability };
    }
    return python.dictionary(arena, pairs);
}
/// A declared Probability's JSON-mode serializer emits actual numeric values
/// as floats, even on an existing model whose fields bypassed validation.
fn probabilityJsonScalar(input: Value) Value {
    return switch (input) {
        .integer => |i| .{ .float = @floatFromInt(i) },
        .bool => |b| .{ .float = @floatFromInt(@intFromBool(b)) },
        .number_string => |s| .{ .float = std.fmt.parseFloat(f64, s) catch return input },
        else => input,
    };
}
fn serializedProbabilities(arena: Allocator, input: Value, depth: u32) runtime.SerializationError!Value {
    if (depth >= pyjson.max_depth) return error.ParseValue;
    if (input == .object) {
        var result: runtime.ObjectMap = .empty;
        try result.ensureTotalCapacity(arena, input.object.count());
        for (input.object.keys(), input.object.values()) |key, item|
            result.putAssumeCapacity(key, try values.serializedValueDepth(arena, probabilityJsonScalar(item), depth + 1));
        return .{ .object = result };
    }
    if (python.view(input)) |view| {
        if (view == .dictionary) {
            const pairs = try arena.dupe(runtime.PythonEntry, view.dictionary);
            defer arena.free(pairs);
            for (pairs) |*entry| entry.value = probabilityJsonScalar(entry.value);
            const temporary = try python.dictionary(arena, pairs);
            // A genuine non-string-keyed dict always serializes to a different
            // JSON object; its transient table owns no child data.
            if (temporary == .custom) {
                defer temporary.custom.deinit(arena);
                return values.serializedValueDepth(arena, temporary, depth);
            }
            return values.serializedValueDepth(arena, temporary, depth);
        }
    }
    return values.serializedValueDepth(arena, input, depth);
}
fn writeProbabilityJson(writer: *std.json.Stringify, input: Value) Writer.Error!void {
    if (input == .object) {
        try writer.beginObject();
        for (input.object.keys(), input.object.values()) |key, item| {
            try writer.objectField(key);
            try writer.write(probabilityJsonScalar(item));
        }
        return writer.endObject();
    }
    if (python.view(input)) |view| {
        if (view == .dictionary) {
            try writer.beginObject();
            for (view.dictionary) |entry| {
                try writer.beginObjectFieldRaw();
                python.writeQuotedKey(writer.writer, entry.key) catch return error.WriteFailed;
                writer.endObjectFieldRaw();
                try writer.write(probabilityJsonScalar(entry.value));
            }
            return writer.endObject();
        }
    }
    try writer.write(input);
}
fn writeProbabilities(writer: *Writer, input: Value, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
    if (depth >= pyjson.max_depth) return error.JsonTooDeep;
    if (input != .object) {
        if (python.view(input)) |view| {
            if (view == .dictionary) {
                try writer.writeByte('{');
                for (view.dictionary, 0..) |entry, index| {
                    try separator(writer, style, depth + 1, index == 0);
                    python.writeQuotedKey(writer, entry.key) catch |err| return switch (err) {
                        error.PythonSerializeFallback => error.ParseValue,
                        else => |other| other,
                    };
                    try writer.writeAll(if (style == .minified) ":" else ": ");
                    try pyjson.writeJsonProjection(writer, probabilityJsonScalar(entry.value), style, @intCast(depth + 1));
                }
                return closeObject(writer, style, depth, view.dictionary.len == 0);
            }
        }
        return pyjson.writeJsonProjection(writer, input, style, @intCast(depth));
    }
    try writer.writeByte('{');
    for (input.object.keys(), input.object.values(), 0..) |key, item, index| {
        try separator(writer, style, depth + 1, index == 0);
        try pyjson.writeJsonString(writer, key);
        try writer.writeAll(if (style == .minified) ":" else ": ");
        try pyjson.writeJsonProjection(writer, probabilityJsonScalar(item), style, @intCast(depth + 1));
    }
    return closeObject(writer, style, depth, input.object.count() == 0);
}
fn closeObject(writer: *Writer, style: runtime.JsonStyle, depth: usize, empty: bool) Writer.Error!void {
    if (!empty and (style == .indent2 or style == .orjson_indent2)) {
        try writer.writeByte('\n');
        try writer.splatByteAll(' ', depth * 2);
    }
    try writer.writeByte('}');
}
fn separator(writer: *Writer, style: runtime.JsonStyle, depth: usize, first: bool) Writer.Error!void {
    switch (style) {
        .compact => if (!first) try writer.writeAll(", "),
        .minified => if (!first) try writer.writeByte(','),
        .indent2, .orjson_indent2 => {
            if (!first) try writer.writeByte(',');
            try writer.writeByte('\n');
            try writer.splatByteAll(' ', depth * 2);
        },
    }
}

// Source factories snapshot JSON before constructing their class names. This
// immutable compile-time reader avoids runtime allocations/type generation,
// reusing the existing Python scalar JSON/repr writers. std's Scanner validates
// numeric grammar; strings use WTF-8 because Python permits lone surrogates.
const Snapshot = struct {
    shape: enum { null, scalar, number, string, array, object },
    json: []const u8,
    repr: []const u8,
    text: []const u8 = "",
    items: []const Snapshot = &.{},
    fn isDescription(self: Snapshot) bool {
        return self.shape == .null or self.shape == .string or self.shape == .array or self.shape == .object;
    }
};
fn descriptionSnapshot(comptime text: []const u8) Snapshot {
    const result = comptime jsonSnapshot(text);
    if (!result.isDescription()) @compileError("dszig: decision descriptions must be JSON strings, objects, arrays, or null");
    return result;
}
fn jsonSnapshot(comptime text: []const u8) Snapshot {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        var index: usize = 0;
        const result = readSnapshot(text, &index, 0);
        whitespace(text, &index);
        if (index != text.len) @compileError("dszig: invalid decision description JSON");
        break :blk result;
    };
}
fn whitespace(comptime text: []const u8, index: *usize) void {
    while (index.* < text.len and std.mem.indexOfScalar(u8, " \t\r\n", text[index.*]) != null) index.* += 1;
}
fn readSnapshot(comptime text: []const u8, index: *usize, depth: usize) Snapshot {
    if (depth >= pyjson.max_depth) @compileError("dszig: decision description exceeds native JSON depth");
    whitespace(text, index);
    if (index.* == text.len) @compileError("dszig: incomplete decision description JSON");
    switch (text[index.*]) {
        '"' => {
            const decoded = readString(text, index);
            return .{ .shape = .string, .text = decoded, .json = scalarText(.{ .string = decoded }, .json), .repr = scalarText(.{ .string = decoded }, .repr) };
        },
        '[' => {
            index.* += 1;
            var items: []const Snapshot = &.{};
            whitespace(text, index);
            if (index.* < text.len and text[index.*] != ']') {
                while (true) {
                    const item = readSnapshot(text, index, depth + 1);
                    items = items ++ [_]Snapshot{item};
                    whitespace(text, index);
                    if (index.* == text.len) @compileError("dszig: incomplete decision description array");
                    if (text[index.*] == ']') break;
                    if (text[index.*] != ',') @compileError("dszig: invalid decision description array");
                    index.* += 1;
                }
            }
            if (index.* == text.len or text[index.*] != ']') @compileError("dszig: incomplete decision description array");
            index.* += 1;
            var json: []const u8 = "[";
            var repr: []const u8 = "[";
            for (items, 0..) |item, i| {
                json = json ++ (if (i == 0) "" else ", ") ++ item.json;
                repr = repr ++ (if (i == 0) "" else ", ") ++ item.repr;
            }
            return .{ .shape = .array, .json = json ++ "]", .repr = repr ++ "]", .items = items };
        },
        '{' => {
            index.* += 1;
            const Pair = struct { key: []const u8, value: Snapshot };
            var pairs: []const Pair = &.{};
            whitespace(text, index);
            if (index.* < text.len and text[index.*] != '}') {
                while (true) {
                    whitespace(text, index);
                    const key = readString(text, index);
                    whitespace(text, index);
                    if (index.* == text.len or text[index.*] != ':') @compileError("dszig: invalid decision description object");
                    index.* += 1;
                    const item = readSnapshot(text, index, depth + 1);
                    var replaced: []const Pair = &.{};
                    var found = false;
                    for (pairs) |pair| {
                        const same = std.mem.eql(u8, key, pair.key);
                        found = found or same;
                        replaced = replaced ++ [_]Pair{if (same) .{ .key = key, .value = item } else pair};
                    }
                    pairs = if (found) replaced else pairs ++ [_]Pair{.{ .key = key, .value = item }};
                    whitespace(text, index);
                    if (index.* == text.len) @compileError("dszig: incomplete decision description object");
                    if (text[index.*] == '}') break;
                    if (text[index.*] != ',') @compileError("dszig: invalid decision description object");
                    index.* += 1;
                }
            }
            if (index.* == text.len or text[index.*] != '}') @compileError("dszig: incomplete decision description object");
            index.* += 1;
            var json: []const u8 = "{";
            var repr: []const u8 = "{";
            for (pairs, 0..) |pair, i| {
                json = json ++ (if (i == 0) "" else ", ") ++ scalarText(.{ .string = pair.key }, .json) ++ ": " ++ pair.value.json;
                repr = repr ++ (if (i == 0) "" else ", ") ++ scalarText(.{ .string = pair.key }, .repr) ++ ": " ++ pair.value.repr;
            }
            return .{ .shape = .object, .json = json ++ "}", .repr = repr ++ "}" };
        },
        else => {
            const start = index.*;
            while (index.* < text.len and std.mem.indexOfScalar(u8, " \t\r\n,]}", text[index.*]) == null) index.* += 1;
            const token = text[start..index.*];
            if (std.mem.eql(u8, token, "null")) return .{ .shape = .null, .json = "null", .repr = "None" };
            if (std.mem.eql(u8, token, "true")) return .{ .shape = .scalar, .json = "true", .repr = "True" };
            if (std.mem.eql(u8, token, "false")) return .{ .shape = .scalar, .json = "false", .repr = "False" };
            var scanner = std.json.Scanner.initCompleteInput(undefined, token);
            const numeric = scanner.next() catch @compileError("dszig: invalid decision description number");
            if (numeric != .number or (scanner.next() catch @compileError("dszig: invalid decision description number")) != .end_of_document)
                @compileError("dszig: invalid decision description JSON token");
            const value: Value = if (std.mem.indexOfAny(u8, token, ".eE") == null)
                if (std.fmt.parseInt(i64, token, 10)) |small| .{ .integer = small } else |_| .{ .number_string = token }
            else
                .{ .float = std.fmt.parseFloat(f64, token) catch @compileError("dszig: invalid decision description number") };
            if (value == .float and !std.math.isFinite(value.float)) @compileError("dszig: decision descriptions cannot contain nonfinite numbers");
            return .{ .shape = .number, .json = scalarText(value, .json), .repr = scalarText(value, .repr) };
        },
    }
}
fn readString(comptime text: []const u8, index: *usize) []const u8 {
    if (index.* == text.len or text[index.*] != '"') @compileError("dszig: decision description object requires JSON string keys");
    index.* += 1;
    var result: []const u8 = "";
    while (index.* < text.len) {
        const byte = text[index.*];
        index.* += 1;
        if (byte == '"') return result;
        if (byte < 0x20) @compileError("dszig: unescaped JSON control character");
        if (byte != '\\') {
            if (byte < 0x80) {
                result = result ++ [_]u8{byte};
                continue;
            }
            const start = index.* - 1;
            const len = std.unicode.utf8ByteSequenceLength(byte) catch @compileError("dszig: invalid decision description UTF-8");
            if (start + len > text.len) @compileError("dszig: truncated decision description UTF-8");
            _ = std.unicode.wtf8Decode(text[start..][0..len]) catch @compileError("dszig: invalid decision description UTF-8");
            result = result ++ text[start..][0..len];
            index.* = start + len;
            continue;
        }
        if (index.* == text.len) @compileError("dszig: incomplete JSON string escape");
        const escape = text[index.*];
        index.* += 1;
        if (escape == 'u') {
            var point: u21 = readHex4(text, index);
            if (point >= 0xd800 and point <= 0xdbff and index.* + 6 <= text.len and text[index.*] == '\\' and text[index.* + 1] == 'u') {
                var lookahead = index.* + 2;
                const low = readHex4(text, &lookahead);
                if (low >= 0xdc00 and low <= 0xdfff) {
                    point = 0x10000 + (point - 0xd800) * 0x400 + low - 0xdc00;
                    index.* = lookahead;
                }
            }
            var encoded: [4]u8 = undefined;
            const count = std.unicode.wtf8Encode(point, &encoded) catch unreachable;
            const final = encoded[0..count].*;
            result = result ++ final;
        } else {
            const decoded: u8 = switch (escape) {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'b' => 8,
                'f' => 12,
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                else => @compileError("dszig: invalid JSON string escape"),
            };
            result = result ++ [_]u8{decoded};
        }
    }
    @compileError("dszig: unterminated decision description JSON string");
}
fn readHex4(comptime text: []const u8, index: *usize) u21 {
    if (index.* + 4 > text.len) @compileError("dszig: incomplete JSON Unicode escape");
    for (text[index.*..][0..4]) |byte| {
        if (!std.ascii.isDigit(byte) and !(byte >= 'a' and byte <= 'f') and !(byte >= 'A' and byte <= 'F'))
            @compileError("dszig: invalid JSON Unicode escape");
    }
    const result = std.fmt.parseInt(u16, text[index.*..][0..4], 16) catch @compileError("dszig: invalid JSON Unicode escape");
    index.* += 4;
    return result;
}
const ScalarMode = enum { str, repr, json };
fn scalarText(comptime value: Value, comptime mode: ScalarMode) []const u8 {
    return comptime blk: {
        const size = switch (value) {
            .string, .number_string => |s| s.len * 10 + 64,
            else => 512,
        };
        var buffer: [size]u8 = undefined;
        var writer: Writer = .fixed(&buffer);
        switch (mode) {
            .str => pyjson.writeStr(&writer, value) catch @compileError("dszig: invalid decision option str"),
            .repr => pyjson.writeRepr(&writer, value, 0) catch @compileError("dszig: invalid decision description repr"),
            .json => pyjson.writeJsonAscii(&writer, value) catch @compileError("dszig: invalid decision description JSON"),
        }
        const final = buffer[0..writer.buffered().len].*;
        break :blk &final;
    };
}
fn definitionName(comptime name: []const u8) []const u8 {
    return comptime blk: {
        @setEvalBranchQuota(1_000_000);
        // get_defs_ref splits even punctuation INSIDE a configured class name.
        // Its colon-ID and dotted-qualifier removal therefore also affects
        // description dictionaries/strings, before normalize_name is applied.
        var short: []const u8 = "";
        var start: usize = 0;
        for (name, 0..) |byte, index| {
            if (byte == '[' or byte == ']' or byte == ',') {
                short = short ++ shortComponent(name[start..index]) ++ [_]u8{byte};
                start = index + 1;
            }
        }
        short = short ++ shortComponent(name[start..]);
        var result: []const u8 = "";
        var iterator = (std.unicode.Utf8View.init(short) catch @compileError("dszig: invalid decision model name")).iterator();
        while (iterator.nextCodepoint()) |point| {
            result = result ++ if (point == '.') "__" else if (point <= 0x7f and (std.ascii.isAlphanumeric(@intCast(point)) or point == '_' or point == '-')) &[_]u8{@intCast(point)} else "_";
        }
        break :blk result;
    };
}
fn shortComponent(comptime component: []const u8) []const u8 {
    return comptime blk: {
        const text = component[0 .. std.mem.lastIndexOfScalar(u8, component, ':') orelse component.len];
        // re.sub(r'(?:[^.[\\]]+\\.)+((?:[^.[\\]]+))', r'\\1', text)
        var result: []const u8 = "";
        var index: usize = 0;
        while (index < text.len) {
            if (text[index] == '.') {
                result = result ++ ".";
                index += 1;
                continue;
            }
            var last_start = index;
            var end = index;
            while (end < text.len and text[end] != '.') end += 1;
            while (end + 1 < text.len and text[end] == '.' and text[end + 1] != '.') {
                last_start = end + 1;
                end = last_start;
                while (end < text.len and text[end] != '.') end += 1;
            }
            result = result ++ text[last_start..end];
            index = end;
        }
        break :blk result;
    };
}

fn testInput(arena: Allocator, text: []const u8) !Value {
    return std.json.parseFromSliceLeaky(Value, arena, text, .{});
}

test "decision declared strict fields differ from explicit lax validation" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const loose_bool = try testInput(a, "{\"value\":\"true\",\"confidence\":\"0.5\"}");
    try std.testing.expectError(error.ParseValue, Noul.validate(a, loose_bool));
    const noul = try Noul.validateLax(a, loose_bool);
    try std.testing.expect(noul.custom.get(Noul).?.value);
    try std.testing.expectError(error.ParseValue, Noul.validateStrict(a, loose_bool));
    const valid = try testInput(a, "{\"value\":false,\"confidence\":\"0.5\",\"probability\":true}");
    const checked = (try Noul.validate(a, valid)).custom.get(Noul).?;
    try std.testing.expectEqual(@as(f64, 0.5), checked.confidence);
    try std.testing.expectEqual(@as(?f64, 1), checked.probability);
    try std.testing.expect(!checked.truthy());
    const Rubric = ScoreWithLevels("[\"low\", \"mid\", \"high\"]");
    for ([_][]const u8{
        "{\"value\":3,\"confidence\":0.5}",
        "{\"value\":1,\"confidence\":1.1}",
        "{\"value\":1,\"confidence\":0.5,\"level\":true}",
        "{\"value\":1,\"confidence\":0.5,\"level\":3}",
        "{\"value\":1,\"confidence\":0.5,\"probabilities\":{\"0\":-0.1}}",
        "{\"value\":1,\"confidence\":0.5,\"extra\":0}",
    }) |text| try std.testing.expectError(error.ParseValue, Rubric.validate(a, try testInput(a, text)));
    const fractional_level = try testInput(a, "{\"value\":true,\"confidence\":0.5,\"level\":1.0}");
    try std.testing.expectError(error.ParseValue, Rubric.validate(a, fractional_level));
    const score = (try Rubric.validateLax(a, fractional_level)).custom.get(Rubric).?;
    try std.testing.expectEqual(@as(f64, 1), score.value);
    try std.testing.expectEqual(@as(i64, 1), score.level.?.value.integer);
    // The class validates individual evidence probabilities, not a normalized
    // distribution or rubric-key set; those are DecisionState's boundaries.
    const evidence = try Rubric.validate(a, try testInput(a, "{\"value\":1.5,\"confidence\":0,\"probabilities\":{\"99\":0}}"));
    try std.testing.expect(evidence.custom.get(Rubric).?.probabilities != null);
    const coerced = try Rubric.validate(a, try testInput(a, "{\"value\":1,\"confidence\":0.75,\"probabilities\":{\"0\":\"0.25\",\"1\":1}}"));
    const probabilities = python.view(coerced.custom.get(Rubric).?.probabilities.?).?.dictionary;
    try std.testing.expectEqual(@as(f64, 0.25), probabilities[0].value.float);
    try std.testing.expectEqual(@as(f64, 1), probabilities[1].value.float);
}

test "configured Choice retains typed Literal results and exact wide integers" {
    const Mixed = ChoiceWithOptions(&.{
        .{ .value = .{ .integer = 1 }, .description_json = "\"integer\"" },
        .{ .value = .{ .bool = true }, .description_json = "{\"typed\":\"bool\"}" },
        .{ .value = .null, .description_json = "null" },
    });
    const Wide = ChoiceWithOptions(&.{
        .{ .value = .{ .number_string = "100000000000000000000000000000000000000000000000000" }, .description_json = "\"wide\"" },
        .{ .value = .{ .integer = 0 }, .description_json = "\"zero\"" },
    });
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const integer = (try Mixed.validate(a, try testInput(a, "{\"value\":1,\"confidence\":0.5}"))).custom.get(Mixed).?;
    const boolean = (try Mixed.validate(a, try testInput(a, "{\"value\":true,\"confidence\":0.5}"))).custom.get(Mixed).?;
    const numeric = (try Mixed.validate(a, try testInput(a, "{\"value\":1.0,\"confidence\":0.5}"))).custom.get(Mixed).?;
    try std.testing.expect(integer.value == .integer);
    try std.testing.expect(boolean.value == .bool);
    try std.testing.expect(numeric.value == .bool);
    try std.testing.expectError(error.ParseValue, Wide.validate(a, try testInput(a, "{\"value\":100000000000000000000000000000000000000000000000000,\"confidence\":0.5}")));
    const arbitrary = (try Choice.validate(a, try testInput(a, "{\"value\":100000000000000000000000000000000000000000000000000,\"confidence\":0.5}"))).custom.get(Choice).?;
    try std.testing.expect(arbitrary.value == .number_string);
    const wide_level = (try Score.validate(a, try testInput(a, "{\"value\":-3,\"confidence\":0.5,\"level\":100000000000000000000000000000000000000000000000000}"))).custom.get(Score).?;
    try std.testing.expect(wide_level.level.?.value == .number_string);
}

test "decision boxes preserve class identity and direct routing eligibility" {
    const First = NoulWithCriteria(&.{.{ .value = .{ .bool = true }, .description_json = "\"blocked\"" }});
    const Second = NoulWithCriteria(&.{.{ .value = .{ .bool = false }, .description_json = "\"open\"" }});
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const source = try testInput(a, "{\"value\":true,\"confidence\":0.5}");
    const configured = try First.validate(a, source);
    const base = try Noul.validate(a, configured);
    try std.testing.expectEqual(configured.custom.ptr, base.custom.ptr);
    try std.testing.expectError(error.ParseValue, Second.validate(a, configured));
    try std.testing.expect(!try values.pyEqual(configured, try Noul.validate(a, source)));
    try std.testing.expect(!try values.pyEqual(configured, source));
    try std.testing.expect(modelInfo(First) != null);
    try std.testing.expect(modelInfo(?First) == null);
    try std.testing.expect(modelInfo([]const First) == null);
    try std.testing.expect(modelInfo(bool) == null);
    try std.testing.expect(modelInfo(Score) == null);
    try std.testing.expect(modelInfo(Choice) == null);
    try std.testing.expect(!configured.custom.vtable.hashable);
    try std.testing.expectError(error.ParseValue, python.set(a, &.{configured}));
}

test "decision raw iteration retains defaults while structured dumps filter nulls" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const choice = try (Choice{ .value = .null, .confidence = 0.5 }).toRuntime(a);
    const raw = try choice.custom.rawValue(a);
    try std.testing.expect(raw.object.get("value").? == .null);
    try std.testing.expect(raw.object.get("probabilities") == null);
    const fields = try choice.custom.vtable.iterate.?(a, choice.custom.ptr);
    try std.testing.expectEqual(@as(usize, 3), fields.len);
    try std.testing.expect(python.view(fields[2]).?.tuple[1] == .null);
    try std.testing.expectError(error.NotImplemented, choice.custom.get(Choice).?.format());
    const forged = try (Score{ .value = std.math.nan(f64), .confidence = std.math.nan(f64) }).toRuntime(a);
    const python_dump = try forged.custom.rawValue(a);
    try std.testing.expect(std.math.isNan(python_dump.object.get("confidence").?.float));
    const json_dump = try values.serializedValue(a, forged);
    try std.testing.expect(json_dump.object.get("confidence").? == .null);
    try std.testing.expect(json_dump.object.get("value").? == .null);
    var writer: Writer.Allocating = .init(a);
    try values.formatFieldValue(&writer.writer, .{
        .name = "answer",
        .kind = .json,
        .type_name = "Choice",
        .desc = "",
        .prefix = "",
    }, choice);
    const formatted = try testInput(a, writer.written());
    try std.testing.expect(try values.pyEqual(raw, formatted));
}

test "decision owning clones outlive input arenas and preserve base projections" {
    const Selected = ChoiceWithOptions(&.{
        .{ .value = .{ .string = "borrowed 日本" }, .description_json = "{\"why\":[\"text\",null]}" },
        .{ .value = .null, .description_json = "null" },
    });
    const owned = blk: {
        var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer memory.deinit();
        const a = memory.allocator();
        const original = try Selected.validate(a, try testInput(a, "{\"value\":\"borrowed 日本\",\"confidence\":0.75,\"probabilities\":{\"borrowed 日本\":0.8}}"));
        const base = try Choice.fromValue(a, original);
        break :blk try base.clone(std.testing.allocator);
    };
    defer owned.deinit(std.testing.allocator);
    const actual = try owned.toRuntime(std.testing.allocator);
    try std.testing.expect(actual.custom.get(Selected) != null);
    try std.testing.expectEqualStrings("borrowed 日本", actual.custom.get(Selected).?.value.string);
    try std.testing.expectEqual(@as(f64, 0.8), actual.custom.get(Selected).?.probabilities.?.object.get("borrowed 日本").?.float);
    const again = try actual.custom.clone(std.testing.allocator);
    defer again.deinit(std.testing.allocator);
    try std.testing.expect(try actual.custom.eql(again));
}

fn exerciseDecisionOwnership(allocator: Allocator, original: Value) !void {
    const copy = try original.custom.clone(allocator);
    defer copy.deinit(allocator);
    try std.testing.expect(try original.custom.eql(copy));
    const fields = try copy.vtable.iterate.?(allocator, copy.ptr);
    defer allocator.free(fields);
    defer for (fields) |field| field.custom.deinit(allocator);
    try std.testing.expect(fields.len >= 3);
}

test "decision clone and iterator unwind every partial allocation" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const score = try Score.validate(a, try testInput(a, "{\"value\":2.5,\"confidence\":0.75,\"probabilities\":{\"100000000000000000000000000000000000000000000000000\":0.5},\"level\":100000000000000000000000000000000000000000000000000}"));
    const choice = try Choice.validate(a, try testInput(a, "{\"value\":\"borrowed 日本\",\"confidence\":0.75,\"probabilities\":{\"日本\":0.5}}"));
    var stable = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(stable.allocator(), exerciseDecisionOwnership, .{score});
    try std.testing.checkAllAllocationFailures(stable.allocator(), exerciseDecisionOwnership, .{choice});
}

test "nested decision conversion and feedback retain actual model identity" {
    const Rubric = ScoreWithLevels("[\"low\",\"high\"]");
    const Bundle = struct { scores: []const ?Rubric, flags: types.Dict(Noul) };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const checked = try types.validate(Bundle, a, try testInput(a, "{\"scores\":[{\"value\":0.5,\"confidence\":1},null],\"flags\":{\"a\":{\"value\":false,\"confidence\":0.5}}}"));
    const decoded = try values.fromValue(a, Bundle, checked);
    const encoded = try values.toValue(a, Bundle, decoded);
    const score = encoded.object.get("scores").?.array.items[0];
    try std.testing.expect(score.custom.get(Rubric) != null);
    try std.testing.expectEqual(@as(f64, 0.5), (try types.validate(f64, a, score)).float);
    try std.testing.expect(!try values.pyEqual(score, .{ .float = 0.5 }));
    const masked = try @import("feedback.zig").maskValue(a, score);
    try std.testing.expectEqualStrings("<non-serializable: " ++ Rubric.model_name ++ ">", masked.string);
    const dictionary = try score.custom.rawValue(a);
    try std.testing.expect((try @import("feedback.zig").masker(Rubric).?(a, dictionary)) == null);
    const schema = try testInput(a, types.schema(Bundle));
    const definitions = schema.object.get("$defs").?.object;
    const rubric_schema = definitions.get(Rubric.schema_name).?.object;
    try std.testing.expect(rubric_schema.get("properties").?.object.get("probabilities") == null);
    try std.testing.expect(rubric_schema.get("properties").?.object.get("level") == null);
}
const score_schema_description = "Experimental: This class may change or be removed in a future release without warning.\n\n" ++
    "A continuous score with a declared rubric and confidence.\n\n" ++
    "    Declare ``Score[\"poor\", \"fair\", \"excellent\"]`` in increasing order.\n" ++
    "    Predict obtains a distribution and confidence from the backend and\n" ++
    "    averages the level indices using those probabilities.\n" ++
    "    ``probabilities`` retains the raw distribution, keyed by rubric index.\n" ++
    "    ``level`` is the zero-based ordinal selected by the predictor's cuts on the mean\n" ++
    "    level index. It does not change the continuous value or provider confidence.\n" ++
    "    The backend does not generate the derived value or level independently.\n\n" ++
    "    Use the configured Score as the field annotation. Access ``.value`` or\n" ++
    "    call ``float(result)`` to obtain the continuous numeric value.\n    ";
