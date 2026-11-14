//! `dspy.COPRO` (DSPy 3.4.1): coordinate ascent over each predictor's
//! instruction and output prefix. Depth 0 scores `breadth - 1` proposals plus
//! the current instruction; every later depth asks for `breadth` more given
//! the best attempts so far, in increasing score order. COPRO draws no random
//! numbers, so the same LM answers give the same program.
//!
//! Not supported: `track_stats`.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const Metric = optimize.Metric;
const Record = value.Record;

const optimizer_preamble = "You are an instruction optimizer for large language models. ";
const proposed_instruction_desc = "The improved instructions for the language model";
const proposed_prefix_desc = "The string at the end of the prompt, which will help the model start solving the task";
const Proposal = struct { proposed_instruction: []const u8, proposed_prefix_for_output_field: []const u8 };

pub const BasicGenerateInstruction = signature.Signature(struct {
    pub const instructions = optimizer_preamble ++ "I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative.";
    pub const Input = struct { basic_instruction: []const u8 };
    pub const Output = Proposal;
    pub const descriptions = .{
        .basic_instruction = "The initial instructions before optimization",
        .proposed_instruction = proposed_instruction_desc,
        .proposed_prefix_for_output_field = proposed_prefix_desc,
    };
});

pub const GenerateInstructionGivenAttempts = signature.Signature(struct {
    pub const instructions = optimizer_preamble ++ "I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.\n\nYour task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative.";
    pub const Input = struct { attempted_instructions: []const u8 };
    pub const Output = Proposal;
    pub const descriptions = .{
        .proposed_instruction = proposed_instruction_desc,
        .proposed_prefix_for_output_field = proposed_prefix_desc,
    };
});

pub const COPRO = struct {
    metric: Metric,
    breadth: u32 = 10,
    depth: usize = 3,
    init_temperature: f64 = 1.4,
    max_errors: usize = 10,

    /// One entry of `best_program.candidate_programs`, without the program.
    pub const Candidate = struct {
        /// Index into the program's named predictors.
        predictor: usize,
        instruction: []const u8,
        prefix: []const u8,
        score: f64,
        depth: usize,
    };

    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            /// Sorted by score, descending, stable, duplicates dropped.
            candidates: []const Candidate,
            total_calls: usize,
        };
    }

    const Proposed = struct { instruction: []const u8, prefix: []const u8 };

    fn Evaluated(comptime P: type) type {
        return struct {
            candidate: Candidate,
            program: P,

            fn greater(_: void, a: @This(), b: @This()) bool {
                return a.candidate.score > b.candidate.score;
            }
        };
    }

    /// `prompt` is DSPy's `prompt_model` (defaults to `ctx`). Every
    /// candidate program is scored on `trainset`. Results live in `out`.
    pub fn compile(
        self: COPRO,
        comptime P: type,
        out: Allocator,
        gpa: Allocator,
        ctx: Context,
        prompt: ?Context,
        student: P,
        trainset: []const Example,
    ) !Result(P) {
        assert(self.breadth > 1);
        assert(self.depth > 0);
        assert(trainset.len > 0);
        const E = Evaluated(P);
        var clone = student;
        const preds = try predict.namedPredictors(out, &clone);
        assert(preds.len > 0);
        const all = try out.alloc(std.ArrayList(Proposed), preds.len);
        const latest = try out.alloc([]const Proposed, preds.len);
        const evaluated = try out.alloc(std.ArrayList(E), preds.len);
        for (preds, all, latest, evaluated) |p, *a, *l, *e| {
            const state = p.param.predict;
            var inputs: Record = .empty;
            try inputs.put(out, "basic_instruction", .{ .string = state.instructions });
            a.* = .empty;
            try self.propose(BasicGenerateInstruction, out, prompt orelse ctx, &inputs, self.breadth - 1, a);
            try a.append(out, .{ .instruction = state.instructions, .prefix = state.prefixes[state.prefixes.len - 1] });
            l.* = a.items;
            e.* = .empty;
        }

        var total_calls: usize = 0;
        for (0..self.depth) |d| {
            for (preds, 0..) |p, p_i| {
                const candidates = if (preds.len > 1) all[p_i].items else latest[p_i];
                for (candidates) |c| {
                    const instruction = value.pyStrip(std.mem.trim(u8, c.instruction, "\""));
                    const prefix = value.pyStrip(std.mem.trim(u8, c.prefix, "\""));
                    try setSignature(out, p.param.predict, instruction, prefix);
                    const score = (try optimize.evaluate(P, gpa, ctx, &clone, trainset, self.metric, self.max_errors)).score;
                    total_calls += 1;
                    const entry: E = .{
                        .candidate = .{ .predictor = p_i, .instruction = instruction, .prefix = prefix, .score = score, .depth = d },
                        .program = clone,
                    };
                    for (evaluated[p_i].items) |*e| {
                        if (!std.mem.eql(u8, e.candidate.instruction, instruction)) continue;
                        if (!std.mem.eql(u8, e.candidate.prefix, prefix)) continue;
                        if (e.candidate.score < score) e.* = entry;
                        break;
                    } else try evaluated[p_i].append(out, entry);
                }
                var best = &evaluated[p_i].items[0];
                for (evaluated[p_i].items[1..]) |*e| {
                    if (e.candidate.score > best.candidate.score) best = e;
                }
                try setSignature(out, p.param.predict, best.candidate.instruction, best.candidate.prefix);
            }
            if (d == self.depth - 1) break;
            for (preds, 0..) |_, p_i| {
                const tried = try self.attempts(out, evaluated[p_i].items);
                var inputs: Record = .empty;
                try inputs.put(out, "attempted_instructions", .{ .array = tried });
                var fresh: std.ArrayList(Proposed) = .empty;
                try self.propose(GenerateInstructionGivenAttempts, out, prompt orelse ctx, &inputs, self.breadth, &fresh);
                latest[p_i] = fresh.items;
                try all[p_i].appendSlice(out, fresh.items);
            }
        }

        var flat: std.ArrayList(E) = .empty;
        for (evaluated) |e| try flat.appendSlice(out, e.items);
        std.sort.insertion(E, flat.items, {}, E.greater);
        const kept = try dropDuplicates(P, out, flat.items);
        const candidates = try out.alloc(Candidate, kept.len);
        for (candidates, kept) |*c, e| c.* = e.candidate;
        return .{ .program = kept[0].program, .candidates = candidates, .total_calls = total_calls };
    }

    /// `Predict(Sig, n=n, temperature=init_temperature)(**inputs).completions`.
    fn propose(
        self: COPRO,
        comptime Sig: type,
        out: Allocator,
        base: Context,
        inputs: *const Record,
        n: u32,
        into: *std.ArrayList(Proposed),
    ) !void {
        var c = base;
        c.arena = out;
        c.trace = null;
        c.lm_options.temperature = self.init_temperature;
        const predictor: predict.Predict(Sig) = .{};
        const outputs = try predictor.state.callMany(c, inputs, n);
        for (outputs) |o| try into.append(out, .{
            .instruction = o.get("proposed_instruction").?.string,
            .prefix = o.get("proposed_prefix_for_output_field").?.string,
        });
    }

    /// The best `breadth` attempts, lowest score first, as
    /// `Instruction #k`, `Prefix #k`, `Resulting Score #k` lines.
    fn attempts(self: COPRO, out: Allocator, evaluated: anytype) !std.json.Array {
        const E = @TypeOf(evaluated[0]);
        const sorted = try out.dupe(E, evaluated);
        std.sort.insertion(E, sorted, {}, E.greater);
        const shortest = @min(sorted.len, self.breadth);
        var list: std.json.Array = .init(out);
        var i = shortest;
        while (i > 0) {
            i -= 1;
            const k = shortest - i;
            const c = sorted[i].candidate;
            try list.append(.{ .string = try std.fmt.allocPrint(out, "Instruction #{d}: {s}", .{ k, c.instruction }) });
            try list.append(.{ .string = try std.fmt.allocPrint(out, "Prefix #{d}: {s}", .{ k, c.prefix }) });
            var aw: Writer.Allocating = .init(out);
            aw.writer.print("Resulting Score #{d}: ", .{k}) catch return error.OutOfMemory;
            pyjson.writeFloat(&aw.writer, c.score, .python) catch return error.OutOfMemory;
            try list.append(.{ .string = aw.written() });
        }
        return list;
    }

    /// `signature.with_instructions(...).with_updated_fields(last, prefix=...)`.
    fn setSignature(out: Allocator, state: *predict.PredictState, instruction: []const u8, prefix: []const u8) Allocator.Error!void {
        const prefixes = try out.dupe([]const u8, state.prefixes);
        prefixes[prefixes.len - 1] = prefix;
        try state.forkSignatureClass(out);
        state.instructions = instruction;
        state.prefixes = prefixes;
    }

    /// `_drop_duplicates`: among consecutive equal scores, drop a program
    /// whose instructions and last fields equal an earlier kept one's.
    fn dropDuplicates(comptime P: type, out: Allocator, sorted: []const Evaluated(P)) ![]const Evaluated(P) {
        var kept: std.ArrayList(Evaluated(P)) = .empty;
        var batch: std.ArrayList(P) = .empty;
        var batch_score: f64 = -1;
        for (sorted) |e| {
            var repeat = false;
            if (e.candidate.score == batch_score) {
                for (batch.items) |b| {
                    if (try samePrompts(P, out, e.program, b)) {
                        repeat = true;
                        break;
                    }
                }
                if (!repeat) try batch.append(out, e.program);
            } else {
                batch.clearRetainingCapacity();
                try batch.append(out, e.program);
                batch_score = e.candidate.score;
            }
            if (!repeat) try kept.append(out, e);
        }
        return kept.items;
    }

    fn samePrompts(comptime P: type, out: Allocator, a_in: P, b_in: P) !bool {
        var a = a_in;
        var b = b_in;
        const pa = try predict.namedPredictors(out, &a);
        const pb = try predict.namedPredictors(out, &b);
        for (pa, pb) |x, y| {
            const sx = x.param.predict;
            const sy = y.param.predict;
            if (!std.mem.eql(u8, sx.instructions, sy.instructions)) return false;
            const last = sx.fields.len - 1;
            if (!std.mem.eql(u8, sx.prefixes[last], sy.prefixes[last])) return false;
            if (!std.mem.eql(u8, sx.descs[last], sy.descs[last])) return false;
        }
        return true;
    }
};

const testing = std.testing;
const lm_mod = @import("lm.zig");

const TestQA = signature.Signature(struct {
    pub const Input = struct { question: []const u8 };
    pub const Output = struct { answer: []const u8 };
});

const TestProgram = struct {
    generate: predict.Predict(TestQA) = .{},

    pub const Input = TestQA.Input;
    pub const Output = TestQA.Output;

    pub fn forward(self: *const TestProgram, ctx: Context, input: Input) predict.Error!Output {
        return self.generate.forward(ctx, input);
    }
};

fn yesMetric(_: Allocator, _: *const Example, prediction: *const Record, _: ?*const predict.Trace) f64 {
    const answer = prediction.get("answer") orelse return 0;
    return if (answer == .string and std.mem.eql(u8, answer.string, "yes")) 1 else 0;
}

test "COPRO climbs from the basic instruction to the best proposal" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var keyword: lm_mod.KeywordLM = .{
        .rules = &.{
            .{ .needle = "increasing order", .scope = .any, .response = "<proposed_instruction>\n\"Say YES loudly.\"\n</proposed_instruction>\n\n<proposed_prefix_for_output_field>\nAnswer:\n</proposed_prefix_for_output_field>" },
            .{ .needle = "``signature``", .scope = .any, .response = "<proposed_instruction>\nSay yes.\n</proposed_instruction>\n\n<proposed_prefix_for_output_field>\nReply:\n</proposed_prefix_for_output_field>" },
            .{ .needle = "Say YES loudly.", .scope = .any, .response = "<answer>\nyes\n</answer>" },
        },
        .default = "<answer>\nno\n</answer>",
    };
    const ctx: Context = .{ .arena = a, .lm = keyword.lm() };
    const trainset = [_]Example{try Example.fromJson(a, "{\"question\": \"ok?\", \"answer\": \"yes\"}", &.{"question"})};
    const optimizer: COPRO = .{ .metric = yesMetric, .breadth = 3, .depth = 2 };
    const result = try optimizer.compile(TestProgram, a, testing.allocator, ctx, null, .{}, &trainset);
    try testing.expectEqualStrings("Say YES loudly.", result.program.generate.state.instructions);
    try testing.expectEqualStrings("Answer:", result.program.generate.state.prefixes[1]);
    // Depth 0: two identical proposals plus the basic instruction; depth 1:
    // three identical proposals, all scored.
    try testing.expectEqual(@as(usize, 6), result.total_calls);
    try testing.expectEqual(@as(usize, 3), result.candidates.len);
    try testing.expectEqual(@as(f64, 100), result.candidates[0].score);
    try testing.expectEqual(@as(usize, 1), result.candidates[0].depth);
}
