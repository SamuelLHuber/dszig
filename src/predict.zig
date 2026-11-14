//! Modules: `Predict`, `ChainOfThought`, `Retrieve`, the explicit execution
//! `Context`, the call `Trace`, and DSPy-named parameter discovery.
//!
//! A program is a plain struct with a `compiled: bool` field, `Input` and
//! `Output` decls, and `pub fn forward(self: *const P, ctx: Context, input:
//! P.Input) !P.Output`. Nothing is global: allocator, LM, adapter, retriever
//! and trace are all passed in `Context`.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const adapter_mod = @import("adapter.zig");
const lm_mod = @import("lm.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");
const tool_mod = @import("tool.zig");
const inference = @import("inference.zig");
const Adapter = adapter_mod.Adapter;
const Field = signature.Field;
const Record = value.Record;

pub const Error = adapter_mod.Error || lm_mod.Error || value.ConvertError || error{NoRetriever};

pub const Context = struct {
    /// Per-call scratch: prompts, completions, parsed outputs.
    arena: Allocator,
    lm: lm_mod.LM,
    adapter: Adapter = .xml,
    /// When set, every predictor call is appended (deep-copied).
    trace: ?*Trace = null,
    retriever: ?Retriever = null,
    lm_options: lm_mod.Options = .{},
    /// ChatAdapter and XMLAdapter retry once with JSONAdapter on parse failure.
    json_fallback: bool = true,
    /// Implicit XML and JSON enable native History; explicit Chat opts in.
    use_native_function_calling: ?bool = null,
};

pub const Trace = struct {
    arena: Allocator,
    steps: std.ArrayList(Step) = .empty,

    pub const Step = struct {
        predictor: *const PredictState,
        inputs: Record,
        outputs: Record,
    };

    pub fn record(t: *Trace, predictor: *const PredictState, inputs: *const Record, outputs: *const Record) adapter_mod.Error!void {
        var serialized_inputs = try value.cloneRecord(t.arena, inputs);
        for (predictor.fields[0..predictor.input_count]) |f| {
            if (!f.native_tools) continue;
            const catalog = inputs.get(f.name) orelse continue;
            try serialized_inputs.put(t.arena, f.name, try tool_mod.formatCatalogValue(t.arena, catalog));
        }
        try t.steps.append(t.arena, .{
            .predictor = predictor,
            .inputs = serialized_inputs,
            .outputs = try value.cloneRecord(t.arena, outputs),
        });
    }

    /// Structural equality (predictor identity, inputs, outputs, order).
    pub fn eql(a: *const Trace, b: *const Trace) bool {
        if (a.steps.items.len != b.steps.items.len) return false;
        for (a.steps.items, b.steps.items) |x, y| {
            if (x.predictor != y.predictor) return false;
            if (!orderedRecordEqual(&x.inputs, &y.inputs)) return false;
            if (!orderedRecordEqual(&x.outputs, &y.outputs)) return false;
        }
        return true;
    }
};

fn orderedRecordEqual(a: *const Record, b: *const Record) bool {
    if (a.count() != b.count()) return false;
    for (a.keys(), b.keys()) |ka, kb| if (!std.mem.eql(u8, ka, kb)) return false;
    return value.recordEqual(a, b);
}

/// Type-erased, mutable state of one predictor: what DSPy saves and what
/// optimizers change. Slices point to static or caller-arena memory and are
/// replaced, never mutated in place, so copying a program is a cheap clone.
pub const PredictState = struct {
    fields: []const Field,
    input_count: usize,
    instructions: []const u8,
    descs: []const []const u8,
    prefixes: []const []const u8,
    demos: []const Record = &.{},

    pub fn view(state: *const PredictState, arena: Allocator) Allocator.Error!adapter_mod.View {
        assert(state.descs.len == state.fields.len);
        const fields = try arena.dupe(Field, state.fields);
        for (fields, state.descs, state.prefixes) |*f, d, p| {
            f.desc = d;
            f.prefix = p;
        }
        return .{
            .instructions = state.instructions,
            .inputs = fields[0..state.input_count],
            .outputs = fields[state.input_count..],
        };
    }

    pub fn reset(state: *PredictState) void {
        state.demos = &.{};
    }

    /// Run one LM call: format, complete, parse (with JSON fallback), trace.
    pub fn call(state: *const PredictState, ctx: Context, inputs: *const Record) Error!Record {
        const view_ = try state.view(ctx.arena);
        const outputs = runAdapter(ctx, ctx.adapter, view_, state.demos, inputs) catch |err| switch (err) {
            error.AdapterParse => if (ctx.adapter != .json and ctx.json_fallback)
                try runAdapter(ctx, .json, view_, state.demos, inputs)
            else
                return err,
            else => return err,
        };
        if (ctx.trace) |t| try t.record(state, inputs, &outputs);
        return outputs;
    }

    /// `Predict(..., n=n)`: one LM call for `n` completions, all parsed with
    /// the same adapter (DSPy falls back to JSON for the whole call). Like
    /// DSPy, a temperature that is unset or at most 0.15 becomes 0.7.
    pub fn callMany(state: *const PredictState, ctx_in: Context, inputs: *const Record, n: u32) Error![]const Record {
        assert(n >= 1);
        var ctx = ctx_in;
        ctx.lm_options.n = n;
        if (n > 1 and (ctx.lm_options.temperature orelse 0) <= 0.15) ctx.lm_options.temperature = 0.7;
        const view_ = try state.view(ctx.arena);
        const outputs = runAdapterMany(ctx, ctx.adapter, view_, state.demos, inputs) catch |err| switch (err) {
            error.AdapterParse => if (ctx.adapter != .json and ctx.json_fallback)
                try runAdapterMany(ctx, .json, view_, state.demos, inputs)
            else
                return err,
            else => return err,
        };
        if (ctx.trace) |t| try t.record(state, inputs, &outputs[0]);
        return outputs;
    }
};

fn runAdapterMany(ctx: Context, adapter: Adapter, original: adapter_mod.View, demos: []const Record, inputs: *const Record) Error![]const Record {
    var options = ctx.lm_options;
    var request_tools: []const inference.FunctionTool = &.{};
    var tool_input: ?Field = null;
    var tool_output: ?Field = null;
    for (original.inputs) |f| if (f.native_tools) {
        tool_input = f;
        break;
    };
    for (original.outputs) |f| if (f.native_tool_calls) {
        tool_output = f;
        break;
    };
    // DSPy's implicit XML default enables native History rendering even if
    // this LM cannot execute native function declarations.
    const native_policy = ctx.use_native_function_calling orelse (ctx.adapter != .chat);
    const use_native_tools = native_policy and ctx.lm.supports_function_calling and tool_output != null;
    if (!use_native_tools) {
        options.tool_choice = null;
        options.parallel_tool_calls = null;
    }
    const effort = options.reasoning_effort;
    const native = ctx.lm.supports_reasoning and
        !(options.reasoning_effort_present and effort == null) and
        !(effort != null and (std.mem.eql(u8, effort.?, "none") or std.mem.eql(u8, effort.?, "off")));
    var needs_preprocess = false;
    for (original.outputs) |f| needs_preprocess = needs_preprocess or f.native_reasoning;
    var processed = original;
    processed.use_native_function_calling = native_policy;
    if (use_native_tools) {
        const field = tool_input orelse return error.AdapterParse;
        request_tools = tool_mod.functionToolsFromValue(ctx.arena, inputs.get(field.name) orelse return error.AdapterParse) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.AdapterParse,
        };
        var fields: std.ArrayList(Field) = .empty;
        try fields.ensureTotalCapacity(ctx.arena, original.inputs.len);
        for (original.inputs) |f| if (!f.native_tools) {
            fields.appendAssumeCapacity(f);
        };
        processed.inputs = fields.items;
    }
    var processed_demos = demos;
    if (needs_preprocess or use_native_tools) {
        var fields: std.ArrayList(Field) = .empty;
        try fields.ensureTotalCapacity(ctx.arena, original.outputs.len);
        for (original.outputs) |f| {
            if (use_native_tools and f.native_tool_calls) continue;
            if (f.native_reasoning and native) {
                if (options.reasoning_effort == null) options.reasoning_effort = "low";
                continue;
            }
            var wire = f;
            if (f.native_reasoning) wire.name = try fallbackName(ctx.arena, original, f);
            fields.appendAssumeCapacity(wire);
        }
        processed.outputs = fields.items;
        const mapped = try ctx.arena.alloc(Record, demos.len);
        for (demos, mapped) |demo, *dst| {
            dst.* = demo;
            for (original.outputs) |f| {
                if (!f.native_reasoning or native) continue;
                const name = try fallbackName(ctx.arena, original, f);
                if (std.mem.eql(u8, name, f.name)) continue;
                if (dst.*.get(f.name)) |v| {
                    // Copy only records whose keys actually change.
                    if (dst.*.keys().ptr == demo.keys().ptr) dst.* = try value.cloneRecord(ctx.arena, &demo);
                    _ = dst.*.orderedRemove(f.name);
                    const text = if (v == .object) v.object.get("content") orelse v else v;
                    try dst.*.put(ctx.arena, name, text);
                }
            }
        }
        processed_demos = mapped;
    }
    if (adapter == .json and ctx.lm.supports_response_format)
        options.response_format = try adapter_mod.responseFormat(ctx.arena, processed, ctx.lm.supports_response_schema);
    // Explicit-null intent has now been resolved into the effective signature.
    options.reasoning_effort_present = false;
    const messages = try adapter.format(ctx.arena, processed, processed_demos, inputs);
    const completions = try ctx.lm.completeResponses(ctx.arena, .{ .messages = messages, .options = options, .tools = request_tools });
    const outputs = try ctx.arena.alloc(Record, completions.len);
    for (completions, outputs) |c, *o| {
        const has_tool_calls = use_native_tools and c.tool_calls.len > 0;
        const parsed = if (has_tool_calls) blk: {
            if (c.text.len == 0 or processed.outputs.len == 0) break :blk Record.empty;
            break :blk adapter.parse(ctx.arena, processed, c.text) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => Record.empty,
            };
        } else try adapter.parse(ctx.arena, processed, c.text);
        if (!needs_preprocess and !use_native_tools) {
            o.* = parsed;
            continue;
        }
        o.* = .empty;
        try o.ensureTotalCapacity(ctx.arena, original.outputs.len);
        for (original.outputs) |f| {
            const v = if (f.native_tool_calls and has_tool_calls)
                tool_mod.callsFromNative(ctx.arena, c.tool_calls) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.AdapterParse,
                }
            else if (f.native_reasoning and native)
                if (c.reasoning) |r| if (r.len > 0) std.json.Value{ .string = r } else .null else .null
            else
                parsed.get(if (f.native_reasoning) try fallbackName(ctx.arena, original, f) else f.name) orelse
                    (try signature.outputDefault(ctx.arena, f)) orelse
                    if (has_tool_calls or f.native_tool_calls) @as(std.json.Value, .null) else return error.AdapterParse;
            o.putAssumeCapacity(f.name, v);
        }
    }
    return outputs;
}

fn fallbackName(arena: Allocator, view_: adapter_mod.View, f: Field) Allocator.Error![]const u8 {
    const base = f.reasoning_fallback orelse return f.name;
    if (!view_.hasField(base)) return base;
    var suffix: usize = 2;
    while (suffix <= view_.inputs.len + view_.outputs.len + 2) : (suffix += 1) {
        const name = try std.fmt.allocPrint(arena, "{s}_{d}", .{ base, suffix });
        if (!view_.hasField(name)) return name;
    }
    unreachable;
}

fn runAdapter(ctx: Context, adapter: Adapter, view_: adapter_mod.View, demos: []const Record, inputs: *const Record) Error!Record {
    assert(ctx.lm_options.n == 1);
    return (try runAdapterMany(ctx, adapter, view_, demos, inputs))[0];
}

/// `dspy.Predict(Signature)`.
pub fn Predict(comptime Sig: type) type {
    comptime assert(signature.isSignature(Sig));
    return struct {
        const Self = @This();
        pub const Signature = Sig;
        pub const Input = Sig.Input;
        pub const Output = Sig.Output;
        pub const is_dszig_predict = true;

        const default_descs = blk: {
            var d: [Sig.fields.len][]const u8 = undefined;
            for (Sig.fields, 0..) |f, i| d[i] = f.desc;
            break :blk d;
        };
        const default_prefixes = blk: {
            var p: [Sig.fields.len][]const u8 = undefined;
            for (Sig.fields, 0..) |f, i| p[i] = f.prefix;
            break :blk p;
        };

        state: PredictState = .{
            .fields = Sig.fields,
            .input_count = Sig.input_fields_list.len,
            .instructions = Sig.instructions,
            .descs = &default_descs,
            .prefixes = &default_prefixes,
        },
        compiled: bool = false,

        pub fn forward(self: *const Self, ctx: Context, input: Input) Error!Output {
            const inputs = try value.toRecord(ctx.arena, Input, input);
            const outputs = try self.state.call(ctx, &inputs);
            return value.fromRecord(ctx.arena, Output, &outputs);
        }
    };
}

/// A predictor over a comptime field list, for modules that build their own
/// signatures (`ReAct`).
pub fn PredictFields(comptime fields: []const Field, comptime input_count: usize, comptime instructions: []const u8) type {
    comptime assert(input_count < fields.len);
    return struct {
        pub const is_dszig_predict = true;

        const default_descs = blk: {
            var d: [fields.len][]const u8 = undefined;
            for (fields, 0..) |f, i| d[i] = f.desc;
            break :blk d;
        };
        const default_prefixes = blk: {
            var p: [fields.len][]const u8 = undefined;
            for (fields, 0..) |f, i| p[i] = f.prefix;
            break :blk p;
        };

        state: PredictState = .{
            .fields = fields,
            .input_count = input_count,
            .instructions = instructions,
            .descs = &default_descs,
            .prefixes = &default_prefixes,
        },
        compiled: bool = false,
    };
}

/// DSPy 3.5 CoT uses native reasoning, or an explicit plan on other models.
pub fn ChainOfThought(comptime Sig: type) type {
    return ChainOfThoughtWithOptions(Sig, .{});
}

pub const ChainOfThoughtOptions = struct {
    /// Explicit annotation requests a normal output, like rationale_field_type=str.
    rationale_field_type: ?type = null,
    rationale_description: ?[]const u8 = null,
};

pub fn ChainOfThoughtWithOptions(comptime Sig: type, comptime options: ChainOfThoughtOptions) type {
    const use_default = options.rationale_field_type == null;
    const Base = Sig.PrependOutput("reasoning", options.rationale_field_type orelse ?signature.Reasoning);
    const extended_fields = comptime blk: {
        var fields = Base.fields[0..Base.fields.len].*;
        const f = &fields[Sig.input_fields_list.len];
        f.desc = options.rationale_description orelse if (use_default) "A short plan for solving the task." else "${reasoning}";
        if (use_default) {
            f.kind = .str;
            f.type_name = "str";
            f.schema = "";
            f.xml_schema = "";
            f.validate = null;
            f.optional_str = false;
            f.xml_literal = false;
            f.nullable = false;
            f.reasoning_type = true;
            f.native_reasoning = true;
            f.reasoning_fallback = "plan";
        }
        break :blk fields;
    };
    return struct {
        const Self = @This();
        pub const Extended = struct {
            pub const Input = Sig.Input;
            pub const Output = Base.Output;
            pub const fields: []const Field = &extended_fields;
            pub const input_fields_list = fields[0..Sig.input_fields_list.len];
            pub const output_fields_list = fields[Sig.input_fields_list.len..];
            pub const instructions = Sig.instructions;
            pub const is_dszig_signature = true;
        };
        pub const Input = Sig.Input;
        pub const Output = Extended.Output;

        predict: Predict(Extended) = .{},
        compiled: bool = false,

        pub fn forward(self: *const Self, ctx: Context, input: Input) Error!Output {
            return self.predict.forward(ctx, input);
        }
    };
}

pub const Retriever = struct {
    ptr: *const anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        search: *const fn (ptr: *const anyopaque, arena: Allocator, query: []const u8, k: u32) Allocator.Error![]const []const u8,
    };

    pub fn search(r: Retriever, arena: Allocator, query: []const u8, k: u32) Allocator.Error![]const []const u8 {
        return r.vtable.search(r.ptr, arena, query, k);
    }
};

/// Deterministic lexical retriever: ranks passages by the number of distinct
/// lowercase query terms they contain; ties keep corpus order.
pub const InMemoryRetriever = struct {
    passages: []const []const u8,

    pub fn retriever(self: *const InMemoryRetriever) Retriever {
        return .{ .ptr = self, .vtable = &.{ .search = search } };
    }

    fn search(ptr: *const anyopaque, arena: Allocator, query: []const u8, k: u32) Allocator.Error![]const []const u8 {
        const self: *const InMemoryRetriever = @ptrCast(@alignCast(ptr));
        const Scored = struct { index: usize, score: usize };
        const scored = try arena.alloc(Scored, self.passages.len);
        for (self.passages, scored, 0..) |p, *s, i| s.* = .{ .index = i, .score = overlap(query, p) };
        std.sort.insertion(Scored, scored, {}, struct {
            fn lessThan(_: void, a: Scored, b: Scored) bool {
                return a.score > b.score or (a.score == b.score and a.index < b.index);
            }
        }.lessThan);
        const n = @min(k, scored.len);
        const out = try arena.alloc([]const u8, n);
        for (out, scored[0..n]) |*o, s| o.* = self.passages[s.index];
        return out;
    }

    fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
        if (needle.len > haystack.len) return false;
        var i: usize = 0;
        while (i + needle.len <= haystack.len) : (i += 1) {
            if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
        }
        return false;
    }

    fn overlap(query: []const u8, passage: []const u8) usize {
        var score: usize = 0;
        var terms = std.mem.tokenizeAny(u8, query, " \t\n.,;:!?\"'()");
        var seen: [32][]const u8 = undefined;
        var n_seen: usize = 0;
        outer: while (terms.next()) |term| {
            if (term.len < 3) continue;
            for (seen[0..n_seen]) |s| if (std.ascii.eqlIgnoreCase(s, term)) continue :outer;
            if (n_seen < seen.len) {
                seen[n_seen] = term;
                n_seen += 1;
            }
            if (containsIgnoreCase(passage, term)) score += 1;
        }
        return score;
    }
};

/// `dspy.Retrieve(k)`.
pub const Retrieve = struct {
    k: u32 = 3,

    pub fn forward(self: *const Retrieve, ctx: Context, query: []const u8) Error![]const []const u8 {
        const r = ctx.retriever orelse return error.NoRetriever;
        return r.search(ctx.arena, query, self.k);
    }
};

pub const Param = union(enum) {
    predict: *PredictState,
    retrieve: *Retrieve,
};

pub const NamedParam = struct { name: []const u8, param: Param };

/// DSPy's Signature.equals compares instructions and named field metadata,
/// not annotation types, field order, or signature class identity.
pub fn sameSignature(a: *const PredictState, b: *const PredictState) bool {
    if (a == b) return true;
    if (a.fields.len != b.fields.len or a.input_count != b.input_count) return false;
    if (!std.mem.eql(u8, a.instructions, b.instructions)) return false;
    if (a.descs.len != a.fields.len or b.descs.len != b.fields.len or
        a.prefixes.len != a.fields.len or b.prefixes.len != b.fields.len) return false;
    for (a.fields, 0..) |field, i| {
        const j = if (a.fields.ptr == b.fields.ptr) i else for (b.fields, 0..) |other, index| {
            if (std.mem.eql(u8, field.name, other.name)) break index;
        } else return false;
        if ((i < a.input_count) != (j < b.input_count)) return false;
        if (!std.mem.eql(u8, a.descs[i], b.descs[j])) return false;
        if (!std.mem.eql(u8, a.prefixes[i], b.prefixes[j])) return false;
    }
    return true;
}

/// `module.named_parameters()`: DSPy names (`self`, `predict`,
/// `generate_answer.predict`, `hops[0]`), skipping compiled sub-modules.
pub fn namedParameters(arena: Allocator, program: anytype) Allocator.Error![]NamedParam {
    const P = @typeInfo(@TypeOf(program)).pointer.child;
    // Wrappers replacing a module's forward method retain that module's
    // parameter namespace instead of exposing their capture/storage fields.
    if (@hasDecl(P, "dszig_parameter_target")) {
        return namedParameters(arena, &@field(program, P.dszig_parameter_target));
    }
    var out: std.ArrayList(NamedParam) = .empty;
    if (@hasDecl(P, "is_dszig_predict")) {
        try out.append(arena, .{ .name = "self", .param = .{ .predict = &program.state } });
    } else {
        try walk(arena, &out, "", program);
    }
    return out.items;
}

/// `module.named_predictors()`.
pub fn namedPredictors(arena: Allocator, program: anytype) Allocator.Error![]NamedParam {
    const all = try namedParameters(arena, program);
    var n: usize = 0;
    for (all) |p| {
        if (p.param == .predict) {
            all[n] = p;
            n += 1;
        }
    }
    return all[0..n];
}

fn walk(arena: Allocator, out: *std.ArrayList(NamedParam), prefix: []const u8, module: anytype) Allocator.Error!void {
    const M = @typeInfo(@TypeOf(module)).pointer.child;
    inline for (@typeInfo(M).@"struct".field_names) |field_name| {
        const name = try joinName(arena, prefix, field_name);
        try visit(arena, out, name, &@field(module, field_name));
    }
}

fn visit(arena: Allocator, out: *std.ArrayList(NamedParam), name: []const u8, ptr: anytype) Allocator.Error!void {
    const T = @typeInfo(@TypeOf(ptr)).pointer.child;
    if (T == Retrieve) return out.append(arena, .{ .name = name, .param = .{ .retrieve = ptr } });
    switch (@typeInfo(T)) {
        .@"struct" => {
            if (@hasDecl(T, "is_dszig_predict")) {
                return out.append(arena, .{ .name = name, .param = .{ .predict = &ptr.state } });
            }
            if (!isModule(T)) return;
            if (@hasDecl(T, "dszig_parameter_target")) {
                const Target = discoveryTargetType(T);
                if (!@hasDecl(Target, "is_dszig_predict") and ptr.compiled) return;
                return visit(arena, out, name, &@field(ptr, T.dszig_parameter_target));
            }
            if (ptr.compiled) return;
            try walk(arena, out, name, ptr);
        },
        .array => for (ptr, 0..) |*item, i| {
            try visit(arena, out, try std.fmt.allocPrint(arena, "{s}[{d}]", .{ name, i }), item);
        },
        else => {},
    }
}

fn discoveryTargetType(comptime T: type) type {
    if (@hasDecl(T, "dszig_parameter_target")) {
        return discoveryTargetType(@FieldType(T, T.dszig_parameter_target));
    }
    return T;
}

fn isModule(comptime T: type) bool {
    return @hasField(T, "compiled") and @hasDecl(T, "forward");
}

fn joinName(arena: Allocator, prefix: []const u8, name: []const u8) Allocator.Error![]const u8 {
    if (prefix.len == 0) return name;
    return std.fmt.allocPrint(arena, "{s}.{s}", .{ prefix, name });
}

/// Reset every reachable predictor (DSPy `reset_copy` minus the copy:
/// programs are values, so `var copy = program;` is the copy).
pub fn resetDemos(arena: Allocator, program: anytype) Allocator.Error!void {
    for (try namedPredictors(arena, program)) |p| p.param.predict.reset();
}

const testing = std.testing;

const QA = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

const Inner = struct {
    generate_answer: ChainOfThought(QA) = .{},
    compiled: bool = false,
    pub const Input = QA.Input;
    pub fn forward(_: *const @This(), _: Context, _: Input) Error!void {}
};

const Outer = struct {
    retrieve: Retrieve = .{},
    hops: [2]Predict(QA) = .{ .{}, .{} },
    inner: Inner = .{},
    frozen: Inner = .{ .compiled = true },
    compiled: bool = false,
};

test "named parameters follow DSPy naming and freeze rules" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var program: Outer = .{};
    const params = try namedParameters(arena.allocator(), &program);
    const expected = [_][]const u8{ "retrieve", "hops[0]", "hops[1]", "inner.generate_answer.predict" };
    try testing.expectEqual(expected.len, params.len);
    for (expected, params) |e, p| try testing.expectEqualStrings(e, p.name);
    var p: Predict(QA) = .{};
    try testing.expectEqualStrings("self", (try namedParameters(arena.allocator(), &p))[0].name);
}

test "predict and chain of thought run through the adapter with trace" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{
        "<plan>\nParis is the capital.\n</plan>\n\n<answer>\nParis\n</answer>",
        "not parseable",
        "{\"plan\": \"r\", \"answer\": \"Lyon\"}",
    } };
    var trace: Trace = .{ .arena = arena.allocator() };
    const ctx: Context = .{ .arena = arena.allocator(), .lm = script.lm(), .trace = &trace };
    const cot: ChainOfThought(QA) = .{};
    const out = try cot.forward(ctx, .{ .question = "Capital of France?" });
    try testing.expectEqualStrings("Paris", out.answer);
    try testing.expectEqualStrings("Paris is the capital.", out.reasoning.?.content);
    const fallback = try cot.forward(ctx, .{ .question = "x" });
    try testing.expectEqualStrings("Lyon", fallback.answer);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].predictor == &cot.predict.state);
}

test "typed Predict retains wide integer inputs and outputs across adapters" {
    const Wide = signature.Signature(struct {
        pub const Input = struct { n: u64, nested: struct { id: u64 } };
        pub const Output = struct { answer: u64 };
    });
    const InspectingLM = struct {
        response: []const u8,

        fn complete(ptr: *anyopaque, a: Allocator, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const text = request.messages[request.messages.len - 1].content;
            if (std.mem.indexOf(u8, text, "10000000000000000000") == null or
                std.mem.indexOf(u8, text, "18446744073709551615") == null or
                std.mem.indexOf(u8, text, "\"18446744073709551615\"") != null or
                std.mem.indexOf(u8, text, "\"10000000000000000000\"") != null) return error.LMFailure;
            const out = try a.alloc(lm_mod.Completion, 1);
            out[0] = .{ .text = self.response };
            return out;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    inline for (.{
        .{ Adapter.chat, "[[ ## answer ## ]]\n18446744073709551615\n\n[[ ## completed ## ]]" },
        .{ Adapter.json, "{\"answer\":18446744073709551615}" },
        .{ Adapter.xml, "<answer>18446744073709551615</answer>" },
    }) |case| {
        var model: InspectingLM = .{ .response = case[1] };
        const predict: Predict(Wide) = .{};
        const out = try predict.forward(.{
            .arena = memory.allocator(),
            .lm = .{ .ptr = &model, .vtable = &.{ .complete = InspectingLM.complete } },
            .adapter = case[0],
            .json_fallback = false,
        }, .{ .n = 10_000_000_000_000_000_000, .nested = .{ .id = std.math.maxInt(u64) } });
        try testing.expectEqual(std.math.maxInt(u64), out.answer);
    }
}

test "typed Predict preserves custom nullable Reasoning null and empty content" {
    const Custom = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { analysis: ?signature.Reasoning, answer: []const u8 };
    });
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    // Optional custom Reasoning remains a text output even on a reasoning LM:
    // upstream native adaptation only handles a direct Reasoning annotation.
    inline for (.{ false, true }) |supports_native| {
        inline for (.{ false, true }) |disabled| {
            inline for (.{
                .{ Adapter.json, "{\"analysis\":null,\"answer\":\"ok\"}", @as(?[]const u8, null) },
                .{ Adapter.json, "{\"analysis\":\"\",\"answer\":\"ok\"}", @as(?[]const u8, "") },
                .{ Adapter.json, "{\"analysis\":{\"content\":\"why\"},\"answer\":\"ok\"}", @as(?[]const u8, "why") },
                .{ Adapter.xml, "<analysis /><answer>ok</answer>", @as(?[]const u8, null) },
                .{ Adapter.xml, "<analysis></analysis><answer>ok</answer>", @as(?[]const u8, "") },
                .{ Adapter.xml, "<analysis><content>why</content></analysis><answer>ok</answer>", @as(?[]const u8, "why") },
            }) |case| {
                var script: lm_mod.ScriptedLM = .{ .responses = &.{case[1]} };
                var lm = script.lm();
                lm.supports_reasoning = supports_native;
                var trace: Trace = .{ .arena = memory.allocator() };
                const predict: Predict(Custom) = .{};
                const out = try predict.forward(.{
                    .arena = memory.allocator(),
                    .lm = lm,
                    .adapter = case[0],
                    .json_fallback = false,
                    .lm_options = if (disabled) .{ .reasoning_effort = "none" } else .{},
                    .trace = &trace,
                }, .{ .question = "Explain?" });
                try testing.expectEqualStrings("ok", out.answer);
                if (case[2]) |expected| {
                    try testing.expectEqualStrings(expected, out.analysis.?.content);
                    try testing.expectEqualStrings(expected, trace.steps.items[0].outputs.get("analysis").?.string);
                } else {
                    try testing.expect(out.analysis == null);
                    try testing.expect(trace.steps.items[0].outputs.get("analysis").? == .null);
                }
            }
        }
    }
}

test "default CoT fallback validates required Reasoning without changing native null injection" {
    var memory: std.heap.ArenaAllocator = .init(testing.allocator);
    defer memory.deinit();
    inline for (.{
        .{ Adapter.json, "{\"plan\":null,\"answer\":\"ok\"}", @as(?[]const u8, null) },
        .{ Adapter.json, "{\"plan\":{},\"answer\":\"ok\"}", @as(?[]const u8, null) },
        .{ Adapter.json, "{\"plan\":7,\"answer\":\"ok\"}", @as(?[]const u8, null) },
        .{ Adapter.json, "{\"plan\":\"\",\"answer\":\"ok\"}", @as(?[]const u8, "") },
        .{ Adapter.json, "{\"plan\":{\"content\":\"why\"},\"answer\":\"ok\"}", @as(?[]const u8, "why") },
        .{ Adapter.xml, "<plan /><answer>ok</answer>", @as(?[]const u8, null) },
        .{ Adapter.xml, "<plan></plan><answer>ok</answer>", @as(?[]const u8, null) },
        .{ Adapter.xml, "<plan><content></content></plan><answer>ok</answer>", @as(?[]const u8, "") },
    }) |case| {
        var script: lm_mod.ScriptedLM = .{ .responses = &.{case[1]} };
        const cot: ChainOfThought(QA) = .{};
        const ctx: Context = .{
            .arena = memory.allocator(),
            .lm = script.lm(),
            .adapter = case[0],
            .json_fallback = false,
        };
        if (case[2]) |expected| {
            const out = try cot.forward(ctx, .{ .question = "Explain?" });
            try testing.expectEqualStrings(expected, out.reasoning.?.content);
            try testing.expectEqualStrings("ok", out.answer);
        } else {
            try testing.expectError(error.AdapterParse, cot.forward(ctx, .{ .question = "Explain?" }));
        }
    }
}

test "native reasoning preserves absent summaries and records actual content" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const Native = struct {
        summary: ?[]const u8 = null,

        fn complete(ptr: *anyopaque, a: Allocator, request: lm_mod.Request) lm_mod.Error![]const lm_mod.Completion {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (request.options.reasoning_effort == null or
                !std.mem.eql(u8, request.options.reasoning_effort.?, "low")) return error.LMFailure;
            // Native reasoning must never be requested as a text output as well.
            if (std.mem.indexOf(u8, request.messages[0].content, "`reasoning`") != null or
                std.mem.indexOf(u8, request.messages[0].content, "`plan`") != null) return error.LMFailure;
            const out = try a.alloc(lm_mod.Completion, 1);
            out[0] = .{ .text = "<answer>\nParis\n</answer>", .reasoning = self.summary };
            return out;
        }
    };
    var native: Native = .{};
    var trace: Trace = .{ .arena = arena.allocator() };
    const ctx: Context = .{
        .arena = arena.allocator(),
        .lm = .{ .ptr = &native, .vtable = &.{ .complete = Native.complete }, .supports_reasoning = true },
        .trace = &trace,
    };
    const cot: ChainOfThought(QA) = .{};
    const without = try cot.forward(ctx, .{ .question = "Capital?" });
    try testing.expectEqualStrings("Paris", without.answer);
    try testing.expectEqual(@as(?signature.Reasoning, null), without.reasoning);
    try testing.expect(trace.steps.items[0].outputs.get("reasoning").? == .null);
    native.summary = "Paris is France's capital.";
    const with = try cot.forward(ctx, .{ .question = "Capital?" });
    try testing.expectEqualStrings(native.summary.?, with.reasoning.?.content);
    try testing.expectEqualStrings(native.summary.?, trace.steps.items[1].outputs.get("reasoning").?.string);
    native.summary = "";
    const empty = try cot.forward(ctx, .{ .question = "Capital?" });
    try testing.expectEqual(@as(?signature.Reasoning, null), empty.reasoning);
    try testing.expect(trace.steps.items[2].outputs.get("reasoning").? == .null);
}

test "CoT plan fallback avoids task field collisions and honors disabled native reasoning" {
    const WithPlan = signature.Signature(struct {
        pub const Input = struct { question: []const u8, plan: []const u8 };
        pub const Output = struct { answer: []const u8, plan_2: []const u8 };
    });
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var script: lm_mod.ScriptedLM = .{ .responses = &.{
        "<plan_3>\nCheck the facts.\n</plan_3>\n\n<answer>\nParis\n</answer>\n\n<plan_2>\nDone.\n</plan_2>",
    } };
    var lm = script.lm();
    lm.supports_reasoning = true;
    const cot: ChainOfThought(WithPlan) = .{};
    const out = try cot.forward(.{
        .arena = arena.allocator(),
        .lm = lm,
        .lm_options = .{ .reasoning_effort = "none" },
    }, .{ .question = "Capital?", .plan = "Use evidence." });
    try testing.expectEqualStrings("Check the facts.", out.reasoning.?.content);
    try testing.expectEqualStrings("Done.", out.plan_2);
    try testing.expectEqualStrings("Paris", out.answer);
}

test "signature equality compares named roles and metadata rather than order or annotations" {
    const A = signature.Signature(struct {
        pub const instructions = "identity";
        pub const Input = struct { question: []const u8, count: i64 };
        pub const Output = struct { answer: []const u8 };
    });
    const B = signature.Signature(struct {
        pub const instructions = "identity";
        pub const Input = struct { count: []const u8, question: i64 };
        pub const Output = struct { answer: f64 };
    });
    const C = signature.Signature(struct {
        pub const instructions = "identity";
        pub const Input = struct { question: []const u8, answer: []const u8 };
        pub const Output = struct { count: i64 };
    });
    const a: Predict(A) = .{};
    var b: Predict(B) = .{};
    const c: Predict(C) = .{};
    try testing.expect(sameSignature(&a.state, &b.state));
    try testing.expect(sameSignature(&b.state, &a.state));
    try testing.expect(!sameSignature(&a.state, &c.state));
    b.state.instructions = "changed";
    try testing.expect(!sameSignature(&a.state, &b.state));
    b.state.instructions = "identity";
    b.state.descs = &.{ "changed", b.state.descs[1], b.state.descs[2] };
    try testing.expect(!sameSignature(&a.state, &b.state));
}

test "in-memory retriever ranks by overlap" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const corpus: InMemoryRetriever = .{ .passages = &.{ "Berlin is in Germany.", "Paris is the capital of France.", "France has wine." } };
    var script: lm_mod.ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .arena = arena.allocator(), .lm = script.lm(), .retriever = corpus.retriever() };
    const r: Retrieve = .{ .k = 2 };
    const got = try r.forward(ctx, "What is the capital of France?");
    try testing.expectEqualStrings("Paris is the capital of France.", got[0]);
    try testing.expectEqualStrings("France has wine.", got[1]);
}
