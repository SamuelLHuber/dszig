//! DSPy-compatible program state: `program.save("x.json")` /
//! `program.load("x.json")` with `save_program=False`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = @import("runtime_value.zig").Value;
const Writer = std.Io.Writer;

const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const value = @import("value.zig");
const Record = value.Record;

/// Dependency versions DSZig targets; written as DSPy writes them.
pub const dependency_versions = [_][2][]const u8{
    .{ "python", "3.11" },
    .{ "dspy", "3.5.0" },
    .{ "cloudpickle", "3.1" },
};

pub const LoadError = error{ MissingState, InvalidState } || Allocator.Error;
pub const SaveError = error{InvalidValue} || Allocator.Error;

/// `module.dump_state()` plus the `metadata` key added by `save`.
/// Read-only: `program` may be a const pointer.
pub fn dumpState(arena: Allocator, program: anytype) SaveError!Value {
    var root: Record = .empty;
    // The parameter walk yields mutable pointers; nothing here writes through them.
    for (try predict.namedParameters(arena, @constCast(program))) |p| {
        try root.put(arena, p.name, switch (p.param) {
            .predict => |s| try dumpPredict(arena, s),
            .retrieve => |r| blk: {
                var obj: Record = .empty;
                try obj.put(arena, "k", .{ .integer = r.k });
                break :blk .{ .object = obj };
            },
        });
    }
    var versions: Record = .empty;
    for (dependency_versions) |kv| try versions.put(arena, kv[0], .{ .string = kv[1] });
    var metadata: Record = .empty;
    try metadata.put(arena, "dependency_versions", .{ .object = versions });
    try root.put(arena, "metadata", .{ .object = metadata });
    return .{ .object = root };
}

fn dumpPredict(arena: Allocator, s: *const predict.PredictState) SaveError!Value {
    var obj: Record = .empty;
    try obj.put(arena, "traces", .{ .array = .init(arena) });
    try obj.put(arena, "train", .{ .array = .init(arena) });
    var demos: @import("runtime_value.zig").Array = .init(arena);
    for (s.demos) |demo| {
        const serialized = value.serializedValue(arena, .{ .object = demo }) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.ParseValue => error.InvalidValue,
        };
        try demos.append(serialized);
    }
    try obj.put(arena, "demos", .{ .array = demos });
    var fields: @import("runtime_value.zig").Array = .init(arena);
    for (s.prefixes, s.descs) |prefix, desc| {
        var f: Record = .empty;
        try f.put(arena, "prefix", .{ .string = prefix });
        try f.put(arena, "description", .{ .string = desc });
        try fields.append(.{ .object = f });
    }
    var sig: Record = .empty;
    try sig.put(arena, "instructions", .{ .string = s.instructions });
    try sig.put(arena, "fields", .{ .array = fields });
    try obj.put(arena, "signature", .{ .object = sig });
    try obj.put(arena, "lm", .null);
    if (s.hasDecisionFields()) {
        _ = @import("decision_state.zig").State.init(arena, try s.view(arena), s.decision_fields, false, null) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidValue,
        };
        try obj.put(arena, "fields", try value.cloneValue(arena, s.decision_fields));
    }
    return .{ .object = obj };
}

/// Write the state exactly as DSPy's `orjson` OPT_INDENT_2 | OPT_APPEND_NEWLINE.
pub fn save(arena: Allocator, program: anytype, w: *Writer) (pyjson.Error || SaveError)!void {
    try pyjson.writeJson(w, try dumpState(arena, program), .indent2);
    try w.writeByte('\n');
}

pub fn saveAlloc(arena: Allocator, program: anytype) (error{JsonTooDeep} || SaveError)![]const u8 {
    var aw: Writer.Allocating = .init(arena);
    save(arena, program, &aw.writer) catch |err| return switch (err) {
        // The allocating writer only fails when its allocator does.
        error.WriteFailed => error.OutOfMemory,
        error.ParseValue => error.InvalidValue,
        else => |e| e,
    };
    return aw.written();
}

/// `module.load_state(state)`: parameters commit in traversal order.
/// Loaded strings and demos are copied into `arena`.
pub fn loadState(arena: Allocator, program: anytype, state: Value) LoadError!void {
    if (state != .object) return error.InvalidState;
    for (try predict.namedParameters(arena, program)) |p| {
        const saved = state.object.get(p.name) orelse return error.MissingState;
        if (saved != .object) return error.InvalidState;
        switch (p.param) {
            .predict => |s| try loadPredict(arena, s, &saved.object),
            .retrieve => |r| if (saved.object.get("k")) |k| {
                if (k != .integer) return error.InvalidState;
                r.k = k.integer;
            },
        }
    }
}

pub fn load(arena: Allocator, program: anytype, json: []const u8) (LoadError || error{InvalidJson})!void {
    const state = std.json.parseFromSliceLeaky(Value, arena, json, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidJson,
    };
    try loadState(arena, program, state);
}

fn loadPredict(arena: Allocator, s: *predict.PredictState, saved: *const Record) LoadError!void {
    var restored = s.*;
    const sig = saved.get("signature") orelse return error.MissingState;
    if (sig != .object) return error.InvalidState;
    const instructions = sig.object.get("instructions") orelse return error.MissingState;
    if (instructions != .string) return error.InvalidState;
    restored.instructions = try arena.dupe(u8, instructions.string);
    const fields = sig.object.get("fields") orelse return error.MissingState;
    if (fields != .array) return error.InvalidState;
    const descs = try arena.dupe([]const u8, restored.descs);
    const prefixes = try arena.dupe([]const u8, restored.prefixes);
    const n = @min(fields.array.items.len, restored.fields.len);
    for (fields.array.items[0..n], 0..) |f, i| {
        if (f != .object) return error.InvalidState;
        const prefix = f.object.get("prefix") orelse return error.MissingState;
        const desc = f.object.get("description") orelse return error.MissingState;
        if (prefix != .string or desc != .string) return error.InvalidState;
        prefixes[i] = try arena.dupe(u8, prefix.string);
        descs[i] = try arena.dupe(u8, desc.string);
    }
    try restored.forkSignatureClass(arena);
    restored.descs = descs;
    restored.prefixes = prefixes;
    const decision_fields: Value = saved.get("fields") orelse .{ .object = .empty };
    _ = @import("decision_state.zig").State.init(arena, try restored.view(arena), decision_fields, false, null) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidState,
    };
    restored.decision_fields = try value.cloneValue(arena, decision_fields);
    const demos = saved.get("demos") orelse return error.MissingState;
    if (demos != .array) return error.InvalidState;
    const out = try arena.alloc(Record, demos.array.items.len);
    for (demos.array.items, out) |d, *o| {
        if (d != .object) return error.InvalidState;
        o.* = try value.cloneRecord(arena, &d.object);
    }
    restored.demos = out;
    s.* = restored;
    if (saved.contains("extended_signature")) return error.InvalidState;
}
