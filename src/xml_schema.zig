//! Validation for runtime schema-backed Fields without a native Zig validator.
//! This is the same primitive/model/list/dictionary/union schema vocabulary
//! emitted by signature types. Custom annotations still supply Field.validate.
//!
//! Upstream (DSPy 3.5.0): dspy/adapters/xml_adapter.py
const std = @import("std");
const value = @import("value.zig");
const types = @import("types.zig");
const unicode = @import("xml_unicode.zig");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Error = value.ParseError;
const max_depth = @import("pyjson.zig").max_depth;

fn get(schema: Value, name: []const u8) Value {
    return if (schema == .object) schema.object.get(name) orelse .null else .null;
}
fn text(v: Value) []const u8 {
    return if (v == .string) v.string else "";
}
fn typeIs(schema: Value, name: []const u8) bool {
    return std.mem.eql(u8, text(get(schema, "type")), name);
}
fn branches(schema: Value) Value {
    const any = get(schema, "anyOf");
    return if (any == .array) any else get(schema, "oneOf");
}

pub fn parse(arena: Allocator, schema: Value, input: Value) Error!Value {
    var candidate = input;
    if (input == .string and !typeIs(schema, "string")) {
        const alternatives = branches(schema);
        var nullable = false;
        var string_branch = false;
        if (alternatives == .array) for (alternatives.array.items) |branch| {
            nullable = nullable or typeIs(branch, "null");
            string_branch = string_branch or typeIs(branch, "string");
        };
        // DSPy's parse_value deliberately keeps Optional[str] text literal.
        if (!(nullable and string_branch)) {
            const trimmed = unicode.strip(input.string);
            candidate = std.json.parseFromSliceLeaky(Value, arena, trimmed, .{ .duplicate_field_behavior = .use_last }) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => if (std.mem.eql(u8, trimmed, "None"))
                    Value.null
                else if (std.mem.eql(u8, trimmed, "True"))
                    Value{ .bool = true }
                else if (std.mem.eql(u8, trimmed, "False"))
                    Value{ .bool = false }
                else
                    input,
            };
        }
    }
    return validate(arena, schema, get(schema, "$defs"), candidate, 0);
}

fn exactType(schema: Value, input: Value) bool {
    const tag = text(get(schema, "type"));
    if (std.mem.eql(u8, tag, "string")) return input == .string;
    if (std.mem.eql(u8, tag, "integer")) return input == .integer or
        (input == .number_string and std.mem.indexOfAny(u8, input.number_string, ".eE") == null);
    if (std.mem.eql(u8, tag, "number")) return input == .float;
    if (std.mem.eql(u8, tag, "boolean")) return input == .bool;
    if (std.mem.eql(u8, tag, "null")) return input == .null;
    if (std.mem.eql(u8, tag, "object")) return input == .object;
    if (std.mem.eql(u8, tag, "array")) return input == .array;
    return false;
}

fn validate(arena: Allocator, schema: Value, definitions: Value, input: Value, depth: usize) Error!Value {
    if (depth >= max_depth or schema != .object) return error.ParseValue;
    const ref = get(schema, "$ref");
    if (ref == .string) {
        if (!std.mem.startsWith(u8, ref.string, "#/$defs/")) return error.ParseValue;
        const target = get(definitions, ref.string[8..]);
        if (target != .object) return error.ParseValue;
        return validate(arena, target, definitions, input, depth + 1);
    }
    // These need annotation-specific validators rather than silently ignoring
    // validation-bearing JSON Schema features outside the native vocabulary.
    for ([_][]const u8{
        "allOf",            "not",                   "if",            "pattern",           "format",           "patternProperties",
        "dependentSchemas", "unevaluatedProperties", "minimum",       "maximum",           "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf",       "minLength",             "maxLength",     "minItems",          "maxItems",         "uniqueItems",
        "contains",         "minProperties",         "maxProperties", "dependentRequired",
    }) |key| {
        if (schema.object.contains(key)) return error.ParseValue;
    }
    const alternatives = branches(schema);
    if (alternatives == .array) {
        // Pydantic's smart union mode prefers a branch matching the actual
        // Python value type over an earlier branch requiring coercion.
        for ([_]bool{ true, false }) |want_exact| {
            for (alternatives.array.items) |branch| {
                if (exactType(branch, input) != want_exact) continue;
                const result = validate(arena, branch, definitions, input, depth + 1) catch |err| switch (err) {
                    error.ParseValue => continue,
                    error.OutOfMemory => return error.OutOfMemory,
                };
                return result;
            }
        }
        return error.ParseValue;
    }
    const enumeration = get(schema, "enum");
    if (enumeration == .array) {
        for (enumeration.array.items) |member| if (try value.pyEqual(member, input)) return member;
        return error.ParseValue;
    }
    if (schema.object.contains("const")) {
        const member = get(schema, "const");
        return if (try value.pyEqual(member, input)) member else error.ParseValue;
    }
    const schema_type = get(schema, "type");
    if (schema_type == .null) return input;
    if (typeIs(schema, "null")) return if (input == .null) input else error.ParseValue;
    if (typeIs(schema, "string")) return if (input == .string) input else error.ParseValue;
    if (typeIs(schema, "integer")) return value.coerceInteger(arena, input);
    if (typeIs(schema, "number")) return types.validate(f64, arena, input);
    if (typeIs(schema, "boolean")) return types.validate(bool, arena, input);
    if (typeIs(schema, "array")) {
        if (input != .array) return error.ParseValue;
        const prefix = get(schema, "prefixItems");
        const items = get(schema, "items");
        var result: @import("runtime_value.zig").Array = .init(arena);
        try result.ensureTotalCapacity(input.array.items.len);
        for (input.array.items, 0..) |item, i| {
            const child = if (prefix == .array and i < prefix.array.items.len) prefix.array.items[i] else items;
            if (child == .bool and !child.bool) return error.ParseValue;
            result.appendAssumeCapacity(if (child == .object) try validate(arena, child, definitions, item, depth + 1) else item);
        }
        return .{ .array = result };
    }
    if (typeIs(schema, "object")) {
        if (input != .object) return error.ParseValue;
        const property_names = get(schema, "propertyNames");
        if (property_names == .object) {
            for (input.object.keys()) |key| _ = try validate(arena, property_names, definitions, .{ .string = key }, depth + 1);
        } else if (property_names == .bool and !property_names.bool and input.object.count() > 0) {
            return error.ParseValue;
        } else if (property_names != .null and property_names != .bool) {
            return error.ParseValue;
        }
        const properties = get(schema, "properties");
        const required = get(schema, "required");
        if (required == .array) for (required.array.items) |name| {
            if (name != .string or !input.object.contains(name.string)) return error.ParseValue;
        };
        var result: value.Record = .empty;
        if (properties == .object) {
            var it = properties.object.iterator();
            while (it.next()) |entry| {
                if (input.object.get(entry.key_ptr.*)) |item| {
                    try result.put(arena, entry.key_ptr.*, try validate(arena, entry.value_ptr.*, definitions, item, depth + 1));
                } else if (entry.value_ptr.* == .object and entry.value_ptr.object.contains("default")) {
                    try result.put(arena, entry.key_ptr.*, get(entry.value_ptr.*, "default"));
                }
            }
        }
        const additional = get(schema, "additionalProperties");
        var it = input.object.iterator();
        while (it.next()) |entry| {
            if (properties == .object and properties.object.contains(entry.key_ptr.*)) continue;
            if (additional == .bool and !additional.bool) return error.ParseValue;
            if (additional == .null and properties == .object) continue;
            const item = if (additional == .object) try validate(arena, additional, definitions, entry.value_ptr.*, depth + 1) else entry.value_ptr.*;
            try result.put(arena, entry.key_ptr.*, item);
        }
        return .{ .object = result };
    }
    return error.ParseValue;
}
