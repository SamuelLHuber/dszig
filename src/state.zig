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

const lm = @import("lm.zig");
const testing = std.testing;

pub const LoadError = error{ MissingState, InvalidState, ParseValue } || lm.StateError;
pub const SaveError = error{ InvalidValue, ParseValue } || lm.StateError;

pub const LoadOptions = struct {
    lm_loader: ?lm.LM.StateLoader = null,
    /// Preserve endpoint-routing keys only for explicitly trusted state.
    allow_unsafe_lm_state: bool = false,
};

/// `module.dump_state()` plus the `metadata` key added by `save`.
/// Read-only: `program` may be a const pointer.
pub fn dumpState(arena: Allocator, program: anytype) SaveError!Value {
    const P = @typeInfo(@TypeOf(program)).pointer.child;
    var root: Record = .empty;
    // The synthetic optimizer path "self" is not part of standalone Predict state.
    if (@hasDecl(P, "is_dszig_predict")) {
        root = try dumpPredict(arena, &program.state);
    } else {
        // The parameter walk yields mutable pointers; nothing here writes through them.
        for (try predict.namedParameters(arena, @constCast(program))) |p| {
            try root.put(arena, p.name, switch (p.param) {
                .predict => |s| .{ .object = try dumpPredict(arena, s) },
                .retrieve => |r| blk: {
                    var obj: Record = .empty;
                    try obj.put(arena, "k", .{ .integer = r.k });
                    break :blk .{ .object = obj };
                },
            });
        }
    }
    var versions: Record = .empty;
    for (dependency_versions) |kv| try versions.put(arena, kv[0], .{ .string = kv[1] });
    var metadata: Record = .empty;
    try metadata.put(arena, "dependency_versions", .{ .object = versions });
    try root.put(arena, "metadata", .{ .object = metadata });
    return .{ .object = root };
}

fn dumpPredict(arena: Allocator, s: *const predict.PredictState) SaveError!Record {
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
    try obj.put(arena, "lm", if (s.lm) |model| try model.dumpState(arena) else .null);
    if (try s.hasDecisionFields()) {
        _ = @import("decision_state.zig").State.init(arena, try s.view(arena), s.decision_fields, false, null) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidValue,
        };
        try obj.put(arena, "fields", try value.cloneValue(arena, s.decision_fields));
    }
    return obj;
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
pub fn loadState(arena: Allocator, program: anytype, state: Value, options: LoadOptions) LoadError!void {
    if (state != .object) return error.InvalidState;
    const P = @typeInfo(@TypeOf(program)).pointer.child;
    if (@hasDecl(P, "is_dszig_predict")) {
        return loadPredict(arena, &program.state, &state.object, options);
    }
    for (try predict.namedParameters(arena, program)) |p| {
        const saved = state.object.get(p.name) orelse return error.MissingState;
        if (saved != .object) return error.InvalidState;
        switch (p.param) {
            .predict => |s| try loadPredict(arena, s, &saved.object, options),
            .retrieve => |r| if (saved.object.get("k")) |k| {
                if (k != .integer) return error.InvalidState;
                r.k = k.integer;
            },
        }
    }
}

pub fn load(arena: Allocator, program: anytype, json: []const u8, options: LoadOptions) (LoadError || error{InvalidJson})!void {
    const state = std.json.parseFromSliceLeaky(Value, arena, json, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidJson,
    };
    try loadState(arena, program, state, options);
}

test "OpenAI program state restores owned configuration without saved credentials or untrusted routing" {
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    var source_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer source_memory.deinit();
    var target_memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer target_memory.deinit();
    const source = source_memory.allocator();
    const target = target_memory.allocator();
    var client: std.http.Client = .{ .allocator = testing.allocator, .io = testing.io };
    defer client.deinit();
    var model: lm.OpenAI = .{
        .client = &client,
        .base_url = "https://saved.invalid/v1",
        .model = try source.dupe(u8, "trained-model"),
        .api = .responses,
        .api_key = "must-not-be-saved",
        .max_tokens = 73,
        .supports_reasoning = true,
        .max_retries = 2,
        .timeout_ns = 123456789,
    };
    var program: predict.Predict(Sig) = .{};
    program.state.lm = model.lm();
    program.state.lm.?.model_type = .text;
    const json = try saveAlloc(source, &program);
    try testing.expect(std.mem.indexOf(u8, json, "must-not-be-saved") == null);
    var loader: lm.OpenAI.StateLoader = .{
        .client = &client,
        .base_url = "https://trusted.invalid/v1",
        .api_key = "injected-credential",
    };
    var safe: predict.Predict(Sig) = .{};
    try load(target, &safe, json, .{ .lm_loader = loader.loader() });
    var trusted: predict.Predict(Sig) = .{};
    try load(target, &trusted, json, .{ .lm_loader = loader.loader(), .allow_unsafe_lm_state = true });
    // No loaded string may still point into the parsed source state/model.
    _ = source_memory.reset(.free_all);
    const safe_model: *lm.OpenAI = @ptrCast(@alignCast(safe.state.lm.?.ptr));
    const trusted_model: *lm.OpenAI = @ptrCast(@alignCast(trusted.state.lm.?.ptr));
    try testing.expectEqualStrings("https://trusted.invalid/v1", safe_model.base_url);
    try testing.expectEqualStrings("https://saved.invalid/v1", trusted_model.base_url);
    try testing.expectEqualStrings("injected-credential", safe_model.api_key.?);
    try testing.expectEqualStrings("trained-model", safe_model.model);
    try testing.expectEqual(@as(?u32, 73), safe_model.max_tokens);
    try testing.expect(safe.state.lm.?.supports_reasoning);
    try testing.expectEqual(lm.OpenAI.Api.responses, safe_model.api);
    try testing.expectEqual(@import("call_result.zig").ModelType.text, safe.state.lm.?.model_type);
    try testing.expectEqual(@import("call_result.zig").ModelType.text, trusted.state.lm.?.model_type);
    try testing.expectEqual(@as(u8, 2), safe_model.max_retries);
    try testing.expectEqual(@as(?u64, 123456789), safe_model.timeout_ns);
}

test "state LM failures retain source partial commits and null clears an existing binding" {
    const Sig = @import("signature.zig").Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: i64 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var program: predict.Predict(Sig) = .{};
    var model: lm.ScriptedLM = .{ .responses = &.{} };
    var saved = try dumpState(arena, &program);
    const sig = saved.object.getPtr("signature").?;
    try sig.object.put(arena, "instructions", .{ .string = "changed before failed LM restore" });
    var nonempty: Record = .empty;
    try nonempty.put(arena, "__lm_class__", .{ .string = "unknown.NativeLM" });
    try saved.object.put(arena, "lm", .{ .object = nonempty });
    program.state.lm = model.lm();
    try testing.expectError(error.MissingLMLoader, loadState(arena, &program, saved, .{}));
    try testing.expectEqualStrings("changed before failed LM restore", program.state.instructions);
    try testing.expect(program.state.lm.?.sameInstance(model.lm()));
    const falsy = [_]Value{ .null, .{ .bool = false }, .{ .integer = 0 }, .{ .string = "" }, .{ .array = .init(arena) }, .{ .object = .empty } };
    for (falsy) |item| {
        program.state.lm = model.lm();
        try saved.object.put(arena, "lm", item);
        try loadState(arena, &program, saved, .{});
        try testing.expect(program.state.lm == null);
    }
    program.state.lm = model.lm();
    try testing.expectError(error.UnsupportedLMState, saveAlloc(arena, &program));
}

fn restoreLM(arena: Allocator, model: Value, options: LoadOptions) LoadError!?lm.LM {
    const truthy = switch (model) {
        .null => false,
        .bool => |b| b,
        .integer => |n| n != 0,
        .float => |n| n != 0,
        .number_string => |text| (std.fmt.parseFloat(f64, text) catch return error.InvalidLMState) != 0,
        .string => |text| text.len != 0,
        .array => |items| items.items.len != 0,
        .object => |items| items.count() != 0,
        .custom => |custom| try custom.truthy(),
    };
    if (!truthy) return null;
    if (model != .object) return error.InvalidLMState;
    const loader = options.lm_loader orelse return error.MissingLMLoader;
    if (options.allow_unsafe_lm_state) return try loader.restore(arena, model);
    var unsafe_count: u32 = 0;
    for ([_][]const u8{ "api_base", "base_url", "model_list" }) |key|
        unsafe_count += @intFromBool(model.object.contains(key));
    if (unsafe_count == 0) return try loader.restore(arena, model);
    var sanitized: Record = .empty;
    try sanitized.ensureTotalCapacity(arena, model.object.count() - unsafe_count);
    for (model.object.keys(), model.object.values()) |key, item| {
        const unsafe_key = std.mem.eql(u8, key, "api_base") or
            std.mem.eql(u8, key, "base_url") or std.mem.eql(u8, key, "model_list");
        if (!unsafe_key) sanitized.putAssumeCapacity(key, item);
    }
    return try loader.restore(arena, .{ .object = sanitized });
}

fn loadPredict(arena: Allocator, s: *predict.PredictState, saved: *const Record, options: LoadOptions) LoadError!void {
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
    // Source commits non-LM fields before attempting LM reconstruction.
    s.* = restored;
    const model = saved.get("lm") orelse return error.MissingState;
    s.lm = try restoreLM(arena, model, options);
    if (saved.contains("extended_signature")) return error.InvalidState;
}
