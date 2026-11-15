//! DSPy 3.5 `DecisionState`: a per-operation decision configuration snapshot,
//! its parameter validation, backend questions and backend-independent decoding.
//!
//! Effective configurations are mutable after construction exactly like the
//! Source public dictionary: `question` and `decode` read the current values
//! without revalidating them. Arithmetic uses Python operand rules for the
//! values actually present (bools are numbers in arithmetic but not valid
//! parameters; strings, containers and models fail).
const std = @import("std");
const adapter = @import("adapter.zig");
const signature = @import("signature.zig");
const decision_evidence = @import("decision_evidence.zig");
const runtime = @import("runtime_value.zig");
const values = @import("value.zig");
const json_repair = @import("json_repair.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;
const ObjectMap = runtime.ObjectMap;
const Record = values.Record;
const Kind = @import("decision.zig").Kind;

pub const Error = error{ InvalidDecision, OutOfMemory };

/// Nesting bound for configuration JSON checks. Source fails through Python
/// recursion limits instead; both refuse the configuration.
const max_json_depth = 512;

pub const Entry = struct {
    field: signature.Field,
    binding: decision_evidence.Binding,
};

pub const State = struct {
    view: adapter.View,
    /// Effective per-output configuration objects, in output order.
    fields: Record,
    /// Active decision outputs, in output order.
    types: []const Entry,

    pub fn init(
        arena: Allocator,
        view: adapter.View,
        configs: Value,
        system_one: bool,
        declared: ?adapter.View,
    ) Error!State {
        if (configs != .object) return error.InvalidDecision;
        var entries: std.ArrayList(Entry) = .empty;
        var fields: Record = .empty;
        for (view.outputs) |field| {
            // Native boxes carry geometry, not calibratable decision values.
            if (system_one and field.native_bounding_box) continue;
            const configured = configs.object.contains(field.name);
            const binding = field.decision orelse {
                if (system_one) return error.InvalidDecision;
                continue;
            };
            if (!system_one and !configured and !binding.automatic) continue;
            if (declared) |original| {
                if (configured) try preserveAnswerSpace(arena, binding, original.findOutput(field.name));
            }
            var config: ObjectMap = .empty;
            switch (binding.model.kind) {
                .noul => try config.put(arena, "threshold", .{ .float = 0.5 }),
                .score => {
                    var cuts: runtime.Array = .init(arena);
                    const levels = binding.model.labels.len;
                    if (levels > 0) {
                        try cuts.ensureTotalCapacity(levels - 1);
                        for (0..levels - 1) |i| cuts.appendAssumeCapacity(.{ .float = @as(f64, @floatFromInt(i)) + 0.5 });
                    }
                    try config.put(arena, "cuts", .{ .array = cuts });
                },
                .choice => {
                    var weights: ObjectMap = .empty;
                    try weights.ensureTotalCapacity(arena, binding.model.labels.len);
                    for (binding.model.labels) |label| weights.putAssumeCapacity(label, .{ .float = 1.0 });
                    try config.put(arena, "weights", .{ .object = weights });
                },
            }
            try entries.append(arena, .{ .field = field, .binding = binding });
            try fields.put(arena, field.name, .{ .object = config });
        }
        for (configs.object.keys()) |name| {
            if (!fields.contains(name)) return error.InvalidDecision;
        }
        var it = configs.object.iterator();
        while (it.next()) |configured| {
            if (configured.value_ptr.* != .object) return error.InvalidDecision;
            const copy = try values.cloneRecord(arena, &configured.value_ptr.object);
            const target = &fields.getPtr(configured.key_ptr.*).?.object;
            var items = copy.iterator();
            while (items.next()) |item| try target.put(arena, item.key_ptr.*, item.value_ptr.*);
        }
        const state: State = .{ .view = view, .fields = fields, .types = entries.items };
        try state.validateParameters();
        return state;
    }

    pub fn validateParameters(self: *const State) Error!void {
        for (self.types) |entry| {
            const model = entry.binding.model;
            const config = self.fields.get(entry.field.name) orelse return error.InvalidDecision;
            if (config != .object) return error.InvalidDecision;
            const parameter = switch (model.kind) {
                .noul => "threshold",
                .score => "cuts",
                .choice => "weights",
            };
            const setting = config.object.get(parameter) orelse return error.InvalidDecision;
            for (config.object.keys()) |key| {
                if (!eql(key, "instructions") and !eql(key, "criteria") and !eql(key, parameter)) return error.InvalidDecision;
            }
            try validateQuestionConfig(model.kind, model.labels, &config.object);
            switch (model.kind) {
                .noul => {
                    const threshold = exactNumber(setting) orelse return error.InvalidDecision;
                    if (!(threshold.f >= 0 and threshold.f <= 1)) return error.InvalidDecision;
                },
                .score => {
                    const levels = model.labels.len;
                    if (levels < 2 or levels > 10 or setting != .array or setting.array.items.len != levels - 1)
                        return error.InvalidDecision;
                    const upper: f64 = @floatFromInt(levels - 1);
                    for (setting.array.items) |cut| {
                        const n = exactNumber(cut) orelse return error.InvalidDecision;
                        if (!(n.f > 0 and n.f < upper)) return error.InvalidDecision;
                    }
                    const cuts = setting.array.items;
                    for (cuts[0 .. cuts.len - 1], cuts[1..]) |a, b| {
                        if (exactNumber(a).?.f >= exactNumber(b).?.f) return error.InvalidDecision;
                    }
                },
                .choice => {
                    if (setting != .object) return error.InvalidDecision;
                    for (setting.object.keys()) |key| {
                        if (labelIndex(model.labels, key) == null) return error.InvalidDecision;
                    }
                    for (setting.object.values()) |weight| {
                        const n = exactNumber(weight) orelse return error.InvalidDecision;
                        // Every Python int is below math.inf, even beyond binary64.
                        const valid = if (n.integer) n.f >= 0 else n.f >= 0 and n.f < std.math.inf(f64);
                        if (!valid) return error.InvalidDecision;
                    }
                    for (model.labels) |label| {
                        const weight = if (setting.object.get(label)) |w| exactNumber(w).?.f else 1.0;
                        if (weight > 0) break;
                    } else return error.InvalidDecision;
                },
            }
        }
    }

    /// Independent copy of the effective criteria. Bare Noul returns null.
    pub fn getCriteria(self: *const State, arena: Allocator, name: []const u8) Error!Value {
        const entry = self.entryFor(name) orelse return error.InvalidDecision;
        const config = try self.configFor(name);
        if (config.get("criteria")) |criteria| return values.cloneValue(arena, criteria);
        var criteria = try parseCriteria(arena, entry.binding.model.criteria_json);
        if (entry.binding.model.kind == .choice and criteria == .object) {
            for (criteria.object.values()) |*description| {
                if (description.* == .string and description.string.len == 0) description.* = .null;
            }
        }
        return criteria;
    }

    pub fn question(self: *const State, arena: Allocator, name: []const u8) Error!Value {
        const entry = self.entryFor(name) orelse return error.InvalidDecision;
        const config = try self.configFor(name);
        var desc = entry.field.desc;
        if (desc.len == name.len + 3 and std.mem.startsWith(u8, desc, "${") and
            desc[desc.len - 1] == '}' and eql(desc[2 .. desc.len - 1], name)) desc = "";
        const instructions = config.get("instructions");
        if (instructions == null and values.pyStrip(desc).len == 0) return error.InvalidDecision;
        var result: ObjectMap = .empty;
        try result.put(arena, "instructions", if (instructions) |value|
            try values.cloneValue(arena, value)
        else
            .{ .string = try arena.dupe(u8, desc) });
        try result.put(arena, "type", .{ .string = @tagName(entry.binding.model.kind) });
        const criteria = try self.getCriteria(arena, name);
        if (criteria != .null or config.contains("criteria")) try result.put(arena, "criteria", criteria);
        return .{ .object = result };
    }

    /// Decision outputs only, in output order. Answers are backend evidence
    /// keyed by output name; Score/Choice probabilities use string labels.
    pub fn decode(self: *const State, arena: Allocator, answers: *const Record) Error!Record {
        var outputs: Record = .empty;
        for (self.types) |entry| {
            const name = entry.field.name;
            const model = entry.binding.model;
            const answer = answers.get(name) orelse return error.InvalidDecision;
            const config = try self.configFor(name);
            var input: ObjectMap = .empty;
            switch (model.kind) {
                .noul => {
                    const probability = try member(answer, "noul");
                    const threshold = config.get("threshold") orelse return error.InvalidDecision;
                    const p = try toFloat(probability);
                    const at_least = values.floatAtLeast(p, threshold) catch return error.InvalidDecision;
                    const t = try toFloat(threshold);
                    const other = 1 - t;
                    const denominator = if (other > t) other else t;
                    if (denominator == 0) return error.InvalidDecision;
                    try input.put(arena, "value", .{ .bool = at_least });
                    try input.put(arena, "probability", probability);
                    try input.put(arena, "confidence", .{ .float = @abs(p - t) / denominator });
                },
                .score => {
                    const probabilities = try scoreDistribution(answer, model.labels.len);
                    const entries = @import("python_value.zig").view(probabilities).?.dictionary;
                    var total: f64 = 0;
                    var weighted: f64 = 0;
                    for (entries) |pair| {
                        const p = try toFloat(pair.value);
                        total += p;
                        weighted += (try toFloat(pair.key)) * p;
                    }
                    if (total <= 0) return error.InvalidDecision;
                    const mean = weighted / total;
                    const cuts = config.get("cuts") orelse return error.InvalidDecision;
                    var level: i64 = 0;
                    switch (cuts) {
                        .array => |items| for (items.items) |cut| {
                            if (values.floatAtLeast(mean, cut) catch return error.InvalidDecision) level += 1;
                        },
                        // Python iterates dictionary keys and string characters,
                        // which compare against a float only when empty.
                        .object => |object| if (object.count() != 0) return error.InvalidDecision,
                        .string => |text| if (text.len != 0) return error.InvalidDecision,
                        else => return error.InvalidDecision,
                    }
                    try input.put(arena, "value", .{ .float = mean });
                    try input.put(arena, "confidence", try member(answer, "confidence"));
                    try input.put(arena, "probabilities", probabilities);
                    try input.put(arena, "level", .{ .integer = level });
                },
                .choice => {
                    const probabilities = try distribution(answer, model.labels);
                    var total: f64 = 0;
                    for (probabilities.object.values()) |raw| total += try toFloat(raw);
                    if (total <= 0) return error.InvalidDecision;
                    var raw_winner: usize = 0;
                    var raw_best = try toFloat(probabilities.object.get(model.labels[0]).?);
                    for (model.labels[1..], 1..) |label, i| {
                        const p = try toFloat(probabilities.object.get(label).?);
                        if (p > raw_best) {
                            raw_winner = i;
                            raw_best = p;
                        }
                    }
                    const weights = config.get("weights") orelse return error.InvalidDecision;
                    if (weights != .object) return error.InvalidDecision;
                    // Weighted ties prefer the raw winner, then declaration order.
                    var selected: usize = 0;
                    var best: Value = undefined;
                    for (model.labels, 0..) |label, i| {
                        const weight: Value = weights.object.get(label) orelse .{ .float = 1.0 };
                        const score = values.numericProduct(arena, probabilities.object.get(label).?, weight) catch |err|
                            return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidDecision;
                        const order = if (i == 0) null else values.numericOrder(score, best) catch return error.InvalidDecision;
                        if (i == 0 or order == .gt or (order == .eq and i == raw_winner and selected != raw_winner)) {
                            selected = i;
                            best = score;
                        }
                    }
                    const positive = values.numericOrder(best, .{ .integer = 0 }) catch return error.InvalidDecision;
                    if (positive == .lt or positive == .eq) return error.InvalidDecision;
                    try input.put(arena, "value", model.values[selected]);
                    try input.put(arena, "confidence", try member(answer, "confidence"));
                    try input.put(arena, "probabilities", probabilities);
                },
            }
            const box = model.validate(arena, .{ .object = input }) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidDecision,
            };
            const result = if (entry.binding.rich) box else blk: {
                if (box != .custom) return error.InvalidDecision;
                const metadata = box.custom.vtable.decision orelse return error.InvalidDecision;
                break :blk metadata.field(box.custom.ptr, "value") orelse return error.InvalidDecision;
            };
            try outputs.put(arena, name, result);
        }
        return outputs;
    }

    fn entryFor(self: *const State, name: []const u8) ?*const Entry {
        for (self.types) |*entry| if (eql(entry.field.name, name)) return entry;
        return null;
    }

    fn configFor(self: *const State, name: []const u8) Error!*const ObjectMap {
        const config = self.fields.getPtr(name) orelse return error.InvalidDecision;
        if (config.* != .object) return error.InvalidDecision;
        return &config.object;
    }
};

fn validateQuestionConfig(kind: Kind, labels: []const []const u8, config: *const ObjectMap) Error!void {
    for ([_][]const u8{ "instructions", "criteria" }) |label| {
        if (config.get(label)) |entry| {
            if (!descriptionShape(entry) or !finiteJson(entry, 0)) return error.InvalidDecision;
        }
    }
    const criteria = config.get("criteria") orelse return;
    const valid = switch (kind) {
        .noul => criteria == .null or (criteria == .object and for (criteria.object.keys()) |key| {
            if (!eql(key, "true") and !eql(key, "false")) break false;
        } else true),
        .choice => criteria == .object and criteria.object.count() == labels.len and for (criteria.object.keys()) |key| {
            if (labelIndex(labels, key) == null) break false;
        } else true,
        .score => criteria == .array and criteria.array.items.len == labels.len and
            criteria.array.items.len >= 2 and criteria.array.items.len <= 10 and for (criteria.array.items) |item|
        {
            if (item == .null) break false;
        } else true,
    };
    if (!valid) return error.InvalidDecision;
    const entries: []const Value = switch (criteria) {
        .object => |object| object.values(),
        .array => |array| array.items,
        else => &.{},
    };
    for (entries) |entry| if (!descriptionShape(entry)) return error.InvalidDecision;
}

/// Python isinstance(entry, (str, dict, list, NoneType)).
fn descriptionShape(entry: Value) bool {
    return switch (entry) {
        .string, .object, .array, .null => true,
        else => false,
    };
}

/// Strict Pydantic JsonValue plus json.dumps(allow_nan=False), including
/// CPython's 4300-digit integer string conversion limit.
fn finiteJson(entry: Value, depth: usize) bool {
    if (depth > max_json_depth) return false;
    return switch (entry) {
        .null, .bool, .integer, .string => true,
        .float => |f| std.math.isFinite(f),
        .number_string => |text| std.json.isNumberFormattedLikeAnInteger(text) and
            text.len - @intFromBool(text.len > 0 and text[0] == '-') <= 4300,
        .array => |array| for (array.items) |item| {
            if (!finiteJson(item, depth + 1)) break false;
        } else true,
        .object => |object| for (object.values()) |item| {
            if (!finiteJson(item, depth + 1)) break false;
        } else true,
        .custom => false,
    };
}

fn preserveAnswerSpace(arena: Allocator, binding: decision_evidence.Binding, original: ?signature.Field) Error!void {
    const field = original orelse return error.InvalidDecision;
    const declared = field.decision orelse return error.InvalidDecision;
    // Distinct kinds have distinct value annotations (bool, float, Literal).
    if (declared.model.kind != binding.model.kind) return error.InvalidDecision;
    const criteria = try parseCriteria(arena, binding.model.criteria_json);
    const original_criteria = try parseCriteria(arena, declared.model.criteria_json);
    if (!values.pyEqual(criteria, original_criteria)) return error.InvalidDecision;
    if (criteria == .object and original_criteria == .object) {
        for (criteria.object.keys(), original_criteria.object.keys()) |a, b| {
            if (!eql(a, b)) return error.InvalidDecision;
        }
    }
    if (binding.model.kind == .choice) {
        // typing.Literal equality: order-insensitive (type, value) sets.
        const members = binding.model.values;
        const original_members = declared.model.values;
        if (members.len != original_members.len) return error.InvalidDecision;
        for (members) |a| {
            for (original_members) |b| {
                if (sameLiteral(a, b)) break;
            } else return error.InvalidDecision;
        }
        for (original_members) |b| {
            for (members) |a| {
                if (sameLiteral(a, b)) break;
            } else return error.InvalidDecision;
        }
    }
}

const PythonType = enum { int, bool, str, none, other };

fn pythonType(v: Value) PythonType {
    return switch (v) {
        .integer => .int,
        .number_string => |text| if (std.json.isNumberFormattedLikeAnInteger(text)) .int else .other,
        .bool => .bool,
        .string => .str,
        .null => .none,
        else => .other,
    };
}

fn sameLiteral(a: Value, b: Value) bool {
    const kind = pythonType(a);
    return kind != .other and kind == pythonType(b) and values.pyEqual(a, b);
}

fn parseCriteria(arena: Allocator, text: []const u8) Error!Value {
    return json_repair.parseJson(arena, text) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidDecision,
    };
}

/// A Python value whose exact type is int or float (bool excluded).
const Number = struct { f: f64, integer: bool };

fn exactNumber(v: Value) ?Number {
    return switch (v) {
        .integer => |i| .{ .f = @floatFromInt(i), .integer = true },
        .float => |f| .{ .f = f, .integer = false },
        .number_string => |text| .{
            .f = std.fmt.parseFloat(f64, text) catch return null,
            .integer = std.json.isNumberFormattedLikeAnInteger(text),
        },
        else => null,
    };
}

/// Python's implicit float conversion of an arithmetic operand.
fn toFloat(v: Value) Error!f64 {
    return values.numericFloat(v) catch error.InvalidDecision;
}

fn member(answer: Value, key: []const u8) Error!Value {
    if (answer != .object) return error.InvalidDecision;
    return answer.object.get(key) orelse error.InvalidDecision;
}

/// Source DecisionState consumes integer-keyed Score distributions. String
/// aliases belong to DecisionAdapter and must be converted before this boundary.
fn scoreDistribution(answer: Value, levels: usize) Error!Value {
    const probabilities = try member(answer, "probabilities");
    const python = @import("python_value.zig").view(probabilities) orelse return error.InvalidDecision;
    if (python != .dictionary or python.dictionary.len != levels) return error.InvalidDecision;
    for (python.dictionary) |pair| {
        for (0..levels) |i| {
            if (values.pyEqual(pair.key, .{ .integer = @intCast(i) })) break;
        } else return error.InvalidDecision;
    }
    return probabilities;
}

/// The answer keys must be exactly the declared labels; insertion order is kept.
fn distribution(answer: Value, labels: []const []const u8) Error!Value {
    const probabilities = try member(answer, "probabilities");
    if (probabilities != .object or probabilities.object.count() != labels.len) return error.InvalidDecision;
    for (probabilities.object.keys()) |key| {
        if (labelIndex(labels, key) == null) return error.InvalidDecision;
    }
    return probabilities;
}

fn labelIndex(labels: []const []const u8, key: []const u8) ?usize {
    for (labels, 0..) |label, i| if (eql(label, key)) return i;
    return null;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test "Choice weights retain exact integers and distinguish integer zero from float zero" {
    const Choice = @import("decision.zig").ChoiceWithOptions(&.{
        .{ .value = .{ .string = "billing" }, .description_json = "\"Payment issue\"" },
        .{ .value = .{ .string = "technical" }, .description_json = "\"Product bug\"" },
    });
    const Sig = signature.Make(struct {}, struct { category: Choice }, "", &.{});
    const view: adapter.View = .{ .instructions = Sig.instructions, .inputs = Sig.input_fields_list, .outputs = Sig.output_fields_list };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const a = memory.allocator();
    const config = try json_repair.parseJson(a,
        \\{"category":{"weights":{"billing":1267650600228229401496703205376,"technical":1267650600228229401496703205377}}}
    );
    const state = try State.init(a, view, config, false, null);
    const equal_probabilities = try json_repair.parseJson(a,
        \\{"category":{"probabilities":{"billing":1,"technical":1},"confidence":0.8}}
    );
    const selected = try state.decode(a, &equal_probabilities.object);
    try std.testing.expectEqualStrings("technical", selected.get("category").?.custom.get(Choice).?.value.string);

    const zeros: [400]u8 = @splat('0');
    const huge = "1" ++ zeros;
    const huge_config = try json_repair.parseJson(a, "{\"category\":{\"weights\":{\"billing\":" ++ huge ++ ",\"technical\":1}}}");
    const huge_state = try State.init(a, view, huge_config, false, null);
    var probabilities = try json_repair.parseJson(a,
        \\{"category":{"probabilities":{"billing":0,"technical":1},"confidence":0.8}}
    );
    const integer_zero = try huge_state.decode(a, &probabilities.object);
    try std.testing.expectEqualStrings("technical", integer_zero.get("category").?.custom.get(Choice).?.value.string);
    probabilities.object.getPtr("category").?.object.getPtr("probabilities").?.object.getPtr("billing").?.* = .{ .float = 0.0 };
    try std.testing.expectError(error.InvalidDecision, huge_state.decode(a, &probabilities.object));
}
