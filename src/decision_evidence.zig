//! Direct decision annotation bindings and the closed generative evidence models.
const std = @import("std");
const decision = @import("decision.zig");
const runtime = @import("runtime_value.zig");
const types = @import("types.zig");
const pyjson = @import("pyjson.zig");
const Value = runtime.Value;
const Allocator = std.mem.Allocator;

pub const Binding = struct {
    model: decision.ModelInfo,
    rich: bool,
    /// Explicit Annotated metadata activates evidence without Predict.fields.
    automatic: bool,
    evidence_name: []const u8,
    evidence_schema: []const u8,
    validate_evidence: *const fn (Allocator, Value) types.Error!Value,
    validate_evidence_strict: *const fn (Allocator, Value) types.Error!Value,
    evidence_value: *const fn (Allocator, Value) types.Error!Value,
};

pub fn annotation(comptime T: type, comptime metadata: []const type) ?Binding {
    if (comptime decision.annotationError(T, metadata)) |message| @compileError(message);
    const automatic = comptime activation: {
        for (metadata) |M| if (decision.isDecision(M)) break :activation true;
        break :activation false;
    };
    if (comptime decision.modelInfo(T) != null) return bind(T, true);
    if (T == bool) {
        inline for (metadata) |M| {
            if (comptime decision.isDecision(M) and M.kind == .noul) {
                var binding = bind(M, false);
                binding.automatic = true;
                return binding;
            }
        }
        var binding = bind(decision.Noul, false);
        binding.automatic = automatic;
        return binding;
    }
    if (comptime @import("literal.zig").isLiteral(T)) {
        comptime {
            for (metadata) |M| {
                if (decision.isDecision(M) and M.kind == .choice) {
                    if (decision.modelInfo(M) != null) {
                        var binding = bind(M, false);
                        binding.automatic = true;
                        return binding;
                    }
                    break;
                }
            }
            for (T.dszig_literal_values) |v| switch (v) {
                .string, .integer, .number_string, .bool, .null => {},
                else => return null,
            };
            for (T.display_values, 0..) |label, i| {
                for (T.display_values[0..i]) |prior| if (std.mem.eql(u8, label, prior)) return null;
            }
            var options: [T.dszig_literal_values.len]decision.Option = undefined;
            for (T.dszig_literal_values, &options) |v, *option| option.* = .{ .value = v, .description_json = "\"\"" };
            var binding = bind(decision.ChoiceWithOptions(&options), false);
            binding.automatic = automatic;
            return binding;
        }
    }
    if (@typeInfo(T) == .@"enum") {
        comptime {
            for (metadata) |M| {
                if (decision.isDecision(M) and M.kind == .choice and M.configured) {
                    var binding = bind(M, false);
                    binding.automatic = true;
                    return binding;
                }
            }
            const names = std.meta.fieldNames(T);
            var options: [names.len]decision.Option = undefined;
            for (names, &options) |name, *option| option.* = .{ .value = .{ .string = name }, .description_json = "\"\"" };
            var binding = bind(decision.ChoiceWithOptions(&options), false);
            binding.automatic = automatic;
            return binding;
        }
    }
    return null;
}

fn bind(comptime T: type, comptime rich: bool) Binding {
    const info = decision.modelInfo(T).?;
    const name = switch (info.kind) {
        .noul => "Noul",
        .score => "Score",
        .choice => "Choice",
    };
    return .{
        .model = info,
        .rich = rich,
        .automatic = rich,
        .evidence_name = name ++ "Evidence",
        .evidence_schema = evidenceSchema(info, name, true),
        .validate_evidence = evidenceType(T).validate,
        .validate_evidence_strict = evidenceType(T).validateStrict,
        .evidence_value = struct {
            fn project(arena: Allocator, input: Value) types.Error!Value {
                if (input != .custom) return error.ParseValue;
                const model = input.custom.get(evidenceType(T)) orelse return error.ParseValue;
                return model.toAliasedValue(arena);
            }
        }.project,
    };
}

pub const FieldMetadata = struct {
    name: []const u8,
    alias: ?[]const u8 = null,
    title: ?[]const u8 = null,
};

/// A nominal class for each concrete decision annotation, not a JSON shape.
pub fn evidenceType(comptime T: type) type {
    return Generated(T, false);
}

pub fn probabilitiesType(comptime T: type) type {
    return Generated(T, true);
}

test "owning evidence clones retain nested probability classes after source destruction" {
    const Choice = decision.ChoiceWithOptions(&.{
        .{ .value = .{ .string = "日本" }, .description_json = "\"first\"" },
        .{ .value = .{ .string = "other" }, .description_json = "\"second\"" },
    });
    const Evidence = evidenceType(Choice);
    const Probabilities = probabilitiesType(Choice);
    const inference = @import("inference.zig");
    const a = std.testing.allocator;
    const owned = snapshot: {
        var source: std.heap.ArenaAllocator = .init(a);
        defer source.deinit();
        const input = try Evidence.validateJson(source.allocator(),
            \\{"probabilities":{"日本":0.6,"other":0.4},"confidence":0.8}
        );
        const copy = try inference.cloneJsonValue(a, input);
        const original = input.custom.get(Evidence).?.field("probabilities").?.custom.get(Probabilities).?;
        @constCast(original).fields.getPtr("option_0").?.* = .{ .float = 0.1 };
        break :snapshot copy;
    };
    defer inference.freeJsonValue(a, owned);
    const copied = owned.custom.get(Evidence).?;
    const nested = copied.field("probabilities").?.custom.get(Probabilities).?;
    try std.testing.expectEqual(@as(f64, 0.6), nested.field("option_0").?.float);
    try std.testing.expectEqual(@as(f64, 0.4), nested.field("option_1").?.float);
    try std.testing.expectEqual(@as(f64, 0.8), copied.field("confidence").?.float);
    var output: std.heap.ArenaAllocator = .init(a);
    defer output.deinit();
    const wire = try copied.toAliasedValue(output.allocator());
    try std.testing.expectEqual(@as(f64, 0.6), wire.object.get("probabilities").?.object.get("日本").?.float);
}

test "Python evidence dumps unwrap nested models without mutating their raw fields" {
    const Choice = decision.ChoiceWithOptions(&.{
        .{ .value = .{ .string = "left" }, .description_json = "\"Left\"" },
        .{ .value = .{ .string = "right" }, .description_json = "\"Right\"" },
    });
    const Evidence = evidenceType(Choice);
    const Probabilities = probabilitiesType(Choice);
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const boxed = try Evidence.validateJson(a,
        \\{"probabilities":{"left":0.6,"right":0.4},"confidence":0.8}
    );
    const model = boxed.custom.get(Evidence).?;
    const nested = model.field("probabilities").?.custom.get(Probabilities).?;
    var by_field = try model.toPythonValue(a, false);
    const named = &by_field.object.getPtr("probabilities").?.object;
    try std.testing.expectEqual(@as(f64, 0.6), named.get("option_0").?.float);
    try std.testing.expect(!named.contains("left"));
    const by_alias = try model.toPythonValue(a, true);
    const aliased = by_alias.object.get("probabilities").?.object;
    try std.testing.expectEqual(@as(f64, 0.6), aliased.get("left").?.float);
    try std.testing.expect(!aliased.contains("option_0"));
    named.getPtr("option_0").?.* = .{ .float = 0.2 };
    try std.testing.expectEqual(@as(f64, 0.6), nested.field("option_0").?.float);
    try std.testing.expect(model.toValue().object.get("probabilities").?.custom.get(Probabilities).? == nested);
}

fn Generated(comptime T: type, comptime probabilities: bool) type {
    const info = decision.modelInfo(T) orelse @compileError("dszig: decision has no configured evidence type");
    if (probabilities and info.kind == .noul) @compileError("dszig: Noul evidence has no probabilities model");
    const family = switch (info.kind) {
        .noul => "Noul",
        .score => "Score",
        .choice => "Choice",
    };
    return struct {
        const Self = @This();
        pub const decision_type = T;
        pub const is_probabilities = probabilities;
        pub const model_name = family ++ (if (probabilities) "Probabilities" else "Evidence");
        pub const model_fields: []const FieldMetadata = metadata: {
            if (!probabilities) break :metadata if (info.kind == .noul)
                &.{.{ .name = "noul" }}
            else
                &.{ .{ .name = "probabilities" }, .{ .name = "confidence" } };
            var result: [info.labels.len]FieldMetadata = undefined;
            for (info.labels, 0..) |label, i| result[i] = .{
                .name = std.fmt.comptimePrint("option_{d}", .{i}),
                .alias = label,
            };
            const final = result;
            break :metadata &final;
        };
        fields: runtime.ObjectMap,

        pub fn field(self: Self, name: []const u8) ?Value {
            return self.fields.get(name);
        }

        pub fn toValue(self: Self) Value {
            return .{ .object = self.fields };
        }

        /// Python-mode model_dump: nested generated probability models become
        /// fresh dictionaries while raw field reflection keeps their identity.
        /// These models have no Json[T] fields, so round_trip has the same view.
        pub fn toPythonValue(self: Self, arena: Allocator, by_alias: bool) Allocator.Error!Value {
            var result: runtime.ObjectMap = .empty;
            errdefer result.deinit(arena);
            try result.ensureTotalCapacity(arena, model_fields.len);
            for (model_fields) |metadata| {
                const field_value = self.fields.get(metadata.name) orelse continue;
                const projected = if (comptime !probabilities and info.kind != .noul) blk: {
                    if (field_value == .custom) {
                        if (field_value.custom.get(probabilitiesType(T))) |nested|
                            break :blk try nested.toPythonValue(arena, by_alias);
                    }
                    break :blk field_value;
                } else field_value;
                const key = if (by_alias) metadata.alias orelse metadata.name else metadata.name;
                result.putAssumeCapacity(key, projected);
            }
            return .{ .object = result };
        }

        pub fn toAliasedValue(self: Self, arena: Allocator) types.Error!Value {
            if (comptime info.kind == .noul) return self.toValue();
            var result: runtime.ObjectMap = .empty;
            if (probabilities) {
                for (model_fields) |metadata| {
                    if (self.fields.get(metadata.name)) |value| try result.put(arena, metadata.alias.?, value);
                }
            } else {
                if (self.fields.get("probabilities")) |value| {
                    const projected = if (value == .custom and value.custom.get(probabilitiesType(T)) != null)
                        try value.custom.get(probabilitiesType(T)).?.toAliasedValue(arena)
                    else
                        value;
                    try result.put(arena, "probabilities", projected);
                }
                if (self.fields.get("confidence")) |value| try result.put(arena, "confidence", value);
            }
            return .{ .object = result };
        }

        pub fn schema(comptime by_alias: bool) []const u8 {
            return comptime if (probabilities) probabilityModelSchema(info, family, by_alias) else evidenceSchema(info, family, by_alias);
        }

        pub fn validate(arena: Allocator, input: Value) types.Error!Value {
            return validateMode(arena, input, false);
        }

        pub fn validateStrict(arena: Allocator, input: Value) types.Error!Value {
            return validateMode(arena, input, true);
        }

        pub fn validateJson(arena: Allocator, text: []const u8) types.Error!Value {
            const input = @import("json_repair.zig").parseJson(arena, text) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseValue,
            };
            return validate(arena, input);
        }

        fn validateMode(arena: Allocator, input: Value, strict: bool) types.Error!Value {
            if (input == .custom and input.custom.get(Self) != null) return input;
            if (input != .object) return error.ParseValue;
            var fields: runtime.ObjectMap = .empty;
            if (probabilities) {
                if (input.object.count() != info.labels.len) return error.ParseValue;
                for (model_fields) |metadata| {
                    const raw_value = input.object.get(metadata.alias.?) orelse return error.ParseValue;
                    const parsed = try probabilityMode(arena, raw_value, strict);
                    try fields.put(arena, metadata.name, parsed);
                }
            } else if (info.kind == .noul) {
                if (input.object.count() != 1) return error.ParseValue;
                try fields.put(arena, "noul", try probabilityMode(arena, input.object.get("noul") orelse return error.ParseValue, strict));
            } else {
                if (input.object.count() != 2) return error.ParseValue;
                const raw_value = input.object.get("probabilities") orelse return error.ParseValue;
                try fields.put(arena, "probabilities", try probabilitiesType(T).validateMode(arena, raw_value, strict));
                try fields.put(arena, "confidence", try probabilityMode(arena, input.object.get("confidence") orelse return error.ParseValue, strict));
            }
            const model = try arena.create(Self);
            model.* = .{ .fields = fields };
            return .{ .custom = .{ .ptr = model, .vtable = &vtable } };
        }

        fn actual(ptr: *const anyopaque) *const Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn raw(_: Allocator, ptr: *const anyopaque) Allocator.Error!Value {
            return actual(ptr).toValue();
        }

        fn serialized(arena: Allocator, ptr: *const anyopaque, depth: u32) runtime.SerializationError!Value {
            return @import("value.zig").serializedValueDepth(arena, actual(ptr).toValue(), depth);
        }

        fn writeFields(writer: *std.Io.Writer, ptr: *const anyopaque, depth: u32, repr: bool) runtime.CustomWriteError!void {
            if (depth > 512) return error.JsonTooDeep;
            const data = actual(ptr).fields;
            for (data.keys(), data.values(), 0..) |name, value, i| {
                if (i > 0) try writer.writeAll(if (repr) ", " else " ");
                try writer.writeAll(name);
                try writer.writeByte('=');
                try pyjson.writeRepr(writer, value, depth + 1);
            }
        }

        fn writeStr(writer: *std.Io.Writer, ptr: *const anyopaque) runtime.CustomWriteError!void {
            try writeFields(writer, ptr, 0, false);
        }

        fn writeRepr(writer: *std.Io.Writer, ptr: *const anyopaque, depth: u32) runtime.CustomWriteError!void {
            try writer.writeAll(model_name ++ "(");
            try writeFields(writer, ptr, depth, true);
            try writer.writeByte(')');
        }

        fn writeJson(writer: *std.json.Stringify, ptr: *const anyopaque) std.Io.Writer.Error!void {
            try writer.write(actual(ptr).toValue());
        }

        fn writeSerializedJson(writer: *std.Io.Writer, ptr: *const anyopaque, style: runtime.JsonStyle, depth: usize) runtime.CustomWriteError!void {
            try pyjson.writeJsonProjection(writer, actual(ptr).toValue(), style, @intCast(depth));
        }

        fn clone(context: *runtime.CloneContext, ptr: *const anyopaque) Allocator.Error!runtime.Custom {
            const arena = context.allocator;
            const result = try arena.create(Self);
            errdefer arena.destroy(result);
            const data = try context.cloneValue(actual(ptr).toValue());
            result.* = .{ .fields = data.object };
            return .{ .ptr = result, .vtable = &vtable, .owns_data = true };
        }

        fn deinit(arena: Allocator, ptr: *const anyopaque, owns_data: bool) void {
            if (owns_data) @import("inference.zig").freeJsonValue(arena, actual(ptr).toValue());
            arena.destroy(actual(ptr));
        }

        fn equal(a: *const anyopaque, b: *const anyopaque) runtime.CustomError!bool {
            return @import("value.zig").recordEqual(&actual(a).fields, &actual(b).fields);
        }

        fn iterate(arena: Allocator, ptr: *const anyopaque) runtime.CustomError![]Value {
            const data = actual(ptr).fields;
            const result = try arena.alloc(Value, data.count());
            for (data.keys(), data.values(), result) |name, value, *item| {
                const pair = try arena.alloc(Value, 2);
                pair[0] = .{ .string = name };
                pair[1] = value;
                item.* = try @import("python_value.zig").tupleFromOwnedSlice(arena, pair);
            }
            return result;
        }

        const vtable: runtime.CustomVTable = .{
            .type_name = @typeName(Self),
            .type_identity = runtime.typeIdentity(Self),
            .model_name = model_name,
            .raw_value = raw,
            .model_dump = raw,
            .serialized = serialized,
            .write_str = writeStr,
            .write_repr = writeRepr,
            .write_json = writeJson,
            .write_serialized_json = writeSerializedJson,
            .clone = clone,
            .deinit = deinit,
            .eql = equal,
            .iterate = iterate,
        };
    };
}

fn probabilityMode(arena: Allocator, input: Value, strict: bool) types.Error!Value {
    if (strict and input != .integer and input != .float and input != .number_string) {
        if (input != .custom) return error.ParseValue;
        if (input.custom.vtable.float_value == null) {
            const builtin = @import("python_value.zig").view(input) orelse return error.ParseValue;
            if (builtin != .float) return error.ParseValue;
        }
    }
    return probability(arena, input);
}

test "strict probability evidence accepts Decimal float protocol without FloatOperation" {
    const python = @import("python_value.zig");
    const environment = @import("decimal_environment.zig");
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    var context: environment.Context = .{};
    context.traps.insert(.float_operation);
    var local: environment.LocalContext = .{};
    local.enter(&context);
    defer local.deinit();
    const Evidence = evidenceType(decision.Noul);
    var input: runtime.ObjectMap = .empty;
    try input.put(a, "noul", try python.decimal(a, "0.5"));
    const checked = try Evidence.validateStrict(a, .{ .object = input });
    try std.testing.expectEqual(@as(f64, 0.5), checked.custom.get(Evidence).?.field("noul").?.float);
    try std.testing.expectEqual(@as(usize, 0), local.context.flags.count());
    try input.put(a, "noul", .{ .bool = true });
    try std.testing.expectError(error.ParseValue, Evidence.validateStrict(a, .{ .object = input }));
    try input.put(a, "noul", .{ .string = "0.5" });
    try std.testing.expectError(error.ParseValue, Evidence.validateStrict(a, .{ .object = input }));
}

fn probability(arena: Allocator, input: Value) types.Error!Value {
    const parsed = try types.validate(f64, arena, input);
    if (parsed != .float or !(parsed.float >= 0 and parsed.float <= 1)) return error.ParseValue;
    return parsed;
}

fn quote(comptime text: []const u8) []const u8 {
    return comptime result: {
        var buffer: [text.len * 6 + 2]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buffer);
        pyjson.writeJsonString(&writer, text) catch unreachable;
        const bytes = writer.buffered().*;
        break :result &bytes;
    };
}

fn probabilitySchema(comptime title: []const u8, comptime description: ?[]const u8) []const u8 {
    return "{\"type\": \"number\"" ++ (if (description) |d| ", \"description\": " ++ quote(d) else "") ++
        ", \"maximum\": 1, \"minimum\": 0, \"title\": " ++ quote(title) ++ "}";
}

fn probabilityModelSchema(comptime info: decision.ModelInfo, comptime name: []const u8, comptime by_alias: bool) []const u8 {
    return comptime blk: {
        @setEvalBranchQuota(2_000_000);
        var declared: [info.labels.len][]const u8 = undefined;
        for (info.labels, 0..) |label, i| declared[i] = if (by_alias) label else std.fmt.comptimePrint("option_{d}", .{i});
        var labels = declared;
        for (1..labels.len) |i| {
            var j = i;
            while (j > 0 and std.mem.lessThan(u8, labels[j], labels[j - 1])) : (j -= 1) {
                std.mem.swap([]const u8, &labels[j], &labels[j - 1]);
            }
        }
        var properties: []const u8 = "{";
        for (labels, 0..) |label, i| {
            properties = properties ++ (if (i == 0) "" else ", ") ++ quote(label) ++ ": " ++ probabilitySchema(types.title(label), null);
        }
        var required: []const u8 = "[";
        for (declared, 0..) |label, i| required = required ++ (if (i == 0) "" else ", ") ++ quote(label);
        break :blk "{\"type\": \"object\", \"additionalProperties\": false, \"properties\": " ++ properties ++
            "}, \"required\": " ++ required ++ "], \"title\": " ++ quote(name ++ "Probabilities") ++ "}";
    };
}

fn evidenceSchema(comptime info: decision.ModelInfo, comptime name: []const u8, comptime by_alias: bool) []const u8 {
    comptime {
        @setEvalBranchQuota(2_000_000);
        if (info.kind == .noul) return "{\"type\": \"object\", \"additionalProperties\": false, \"properties\": {\"noul\": " ++
            probabilitySchema("Noul", "Probability that the answer is true.") ++
            "}, \"required\": [\"noul\"], \"title\": \"NoulEvidence\"}";
        const probabilities_schema = probabilityModelSchema(info, name, by_alias);
        return "{\"type\": \"object\", \"$defs\": {" ++ quote(name ++ "Probabilities") ++ ": " ++ probabilities_schema ++
            "}, \"additionalProperties\": false, \"properties\": {\"confidence\": " ++
            probabilitySchema("Confidence", "Confidence in the decision, from 0 to 1.") ++
            ", \"probabilities\": {\"$ref\": " ++ quote("#/$defs/" ++ name ++ "Probabilities") ++
            ", \"description\": \"Probability of each option; sum to one.\"}}, \"required\": [\"probabilities\", \"confidence\"], \"title\": " ++ quote(name ++ "Evidence") ++ "}";
    }
}
