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
const Adapter = adapter_mod.Adapter;
const Field = signature.Field;
const Record = value.Record;

pub const Error = adapter_mod.Error || lm_mod.Error || value.ConvertError || error{NoRetriever};

pub const Context = struct {
    /// Per-call scratch: prompts, completions, parsed outputs.
    arena: Allocator,
    lm: lm_mod.LM,
    adapter: Adapter = .chat,
    /// When set, every predictor call is appended (deep-copied).
    trace: ?*Trace = null,
    retriever: ?Retriever = null,
    lm_options: lm_mod.Options = .{},
    /// DSPy's ChatAdapter retries once with JSONAdapter on parse failure.
    json_fallback: bool = true,
};

pub const Trace = struct {
    arena: Allocator,
    steps: std.ArrayList(Step) = .empty,

    pub const Step = struct {
        predictor: *const PredictState,
        inputs: Record,
        outputs: Record,
    };

    pub fn record(t: *Trace, predictor: *const PredictState, inputs: *const Record, outputs: *const Record) Allocator.Error!void {
        try t.steps.append(t.arena, .{
            .predictor = predictor,
            .inputs = try value.cloneRecord(t.arena, inputs),
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
            error.AdapterParse => if (ctx.adapter == .chat and ctx.json_fallback)
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
            error.AdapterParse => if (ctx.adapter == .chat and ctx.json_fallback)
                try runAdapterMany(ctx, .json, view_, state.demos, inputs)
            else
                return err,
            else => return err,
        };
        if (ctx.trace) |t| try t.record(state, inputs, &outputs[0]);
        return outputs;
    }
};

fn runAdapterMany(ctx: Context, adapter: Adapter, view_: adapter_mod.View, demos: []const Record, inputs: *const Record) Error![]const Record {
    const messages = try adapter.format(ctx.arena, view_, demos, inputs);
    var options = ctx.lm_options;
    options.json_mode = adapter == .json;
    const completions = try ctx.lm.completeMany(ctx.arena, .{ .messages = messages, .options = options });
    const outputs = try ctx.arena.alloc(Record, completions.len);
    for (completions, outputs) |c, *o| o.* = try adapter.parse(ctx.arena, view_, c);
    return outputs;
}

fn runAdapter(ctx: Context, adapter: Adapter, view_: adapter_mod.View, demos: []const Record, inputs: *const Record) Error!Record {
    const messages = try adapter.format(ctx.arena, view_, demos, inputs);
    var options = ctx.lm_options;
    options.json_mode = adapter == .json;
    const completion = try ctx.lm.complete(ctx.arena, .{ .messages = messages, .options = options });
    return adapter.parse(ctx.arena, view_, completion);
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

/// `dspy.ChainOfThought(Signature)`: prepends a `reasoning: str` output.
pub fn ChainOfThought(comptime Sig: type) type {
    return struct {
        const Self = @This();
        pub const Extended = Sig.PrependOutput("reasoning", []const u8);
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

/// `module.named_parameters()`: DSPy names (`self`, `predict`,
/// `generate_answer.predict`, `hops[0]`), skipping compiled sub-modules.
pub fn namedParameters(arena: Allocator, program: anytype) Allocator.Error![]NamedParam {
    var out: std.ArrayList(NamedParam) = .empty;
    const P = @typeInfo(@TypeOf(program)).pointer.child;
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
            if (ptr.compiled) return;
            try walk(arena, out, name, ptr);
        },
        .array => for (ptr, 0..) |*item, i| {
            try visit(arena, out, try std.fmt.allocPrint(arena, "{s}[{d}]", .{ name, i }), item);
        },
        else => {},
    }
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
        "[[ ## reasoning ## ]]\nParis is the capital.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
        "not parseable",
        "{\"reasoning\": \"r\", \"answer\": \"Lyon\"}",
    } };
    var trace: Trace = .{ .arena = arena.allocator() };
    const ctx: Context = .{ .arena = arena.allocator(), .lm = script.lm(), .trace = &trace };
    const cot: ChainOfThought(QA) = .{};
    const out = try cot.forward(ctx, .{ .question = "Capital of France?" });
    try testing.expectEqualStrings("Paris", out.answer);
    try testing.expectEqualStrings("Paris is the capital.", out.reasoning);
    const fallback = try cot.forward(ctx, .{ .question = "x" });
    try testing.expectEqualStrings("Lyon", fallback.answer);
    try testing.expectEqual(@as(usize, 2), trace.steps.items.len);
    try testing.expect(trace.steps.items[0].predictor == &cot.predict.state);
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
