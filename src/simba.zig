//! `dspy.SIMBA` (DSPy 3.4.1): stochastic introspective mini-batch ascent.
//! Each step samples trajectories of the best programs on a mini-batch at
//! temperature 1.0 (one rollout id per candidate), buckets them per example
//! by score spread, and builds new programs from the most variable buckets by
//! appending a demo or an LM-written rule; the best of them is kept. Python's
//! `random.Random(seed)` and NumPy's `default_rng(seed)` are ported, so the
//! same LM answers give the same program.
//!
//! The program's class source (`inspect.getsource`) is `Options.program_code`.
//! With `teacher_settings`, the first trajectory sampler runs under the
//! teacher (its LM keeps its temperature), as `prepare_models_for_resampling`
//! does; DSZig does not mutate the teacher LM's `rollout_id` afterwards.
//! Not supported: parallel threads, metrics returning a
//! `dspy.Prediction` with feedback. `np.exp` is Zig's `@exp`.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const adapter = @import("adapter.zig");
const nprandom = @import("nprandom.zig");
const optimize = @import("optimize.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const Metric = optimize.Metric;
const Record = value.Record;
const Value = std.json.Value;

const program_desc = "The outputs of the program that we are analyzing";
const trajectory_desc = "The trajectory of the program's execution, showing each module's I/O";
const reward_desc = "The reward value assigned to the program's outputs";
const reward_info_desc = "Additional information that might be helpful to understanding the assigned reward value.";

pub const OfferFeedback = signature.Signature(struct {
    pub const instructions =
        "You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules\n" ++
        "build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive\n" ++
        "similar inputs in the future.\n\n" ++
        "The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.\n\n" ++
        "In your advice:\n" ++
        "- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.\n" ++
        "- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.\n" ++
        "- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.\n" ++
        "- Ensure each unique module name appears exactly once as a key in the advice dictionary.";
    pub const Input = struct {
        program_code: []const u8,
        modules_defn: []const u8,
        program_inputs: []const u8,
        oracle_metadata: []const u8,
        worse_program_trajectory: []const u8,
        worse_program_outputs: []const u8,
        worse_reward_value: f64,
        worse_reward_info: []const u8,
        better_program_trajectory: []const u8,
        better_program_outputs: []const u8,
        better_reward_value: f64,
        better_reward_info: []const u8,
        module_names: []const []const u8,
    };
    pub const Output = struct { discussion: []const u8, module_advice: signature.StrDict };
    pub const descriptions = .{
        .program_code = "The code of the program that we are analyzing",
        .modules_defn = "The definition of each module in the program, including its I/O",
        .program_inputs = "The inputs to the program that we are analyzing",
        .oracle_metadata = "Any (hidden) metadata about the training set instance we're analyzing",
        .worse_program_trajectory = trajectory_desc,
        .worse_program_outputs = program_desc,
        .worse_reward_value = reward_desc,
        .worse_reward_info = reward_info_desc,
        .better_program_trajectory = trajectory_desc,
        .better_program_outputs = program_desc,
        .better_reward_value = reward_desc,
        .better_reward_info = reward_info_desc,
        .module_names = "The names of the modules in the program, for which we seek advice",
        .discussion = "Discussing blame of where each module went wrong, if it did",
        .module_advice = "For each module, describe very concretely: If the module receives ${description of input or patterns therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act like the successful trajectory rather than the lower-scoring trajectory.",
    };
});

pub const SIMBA = struct {
    metric: Metric,
    bsize: usize = 32,
    num_candidates: usize = 6,
    max_steps: usize = 8,
    max_demos: usize = 4,
    demo_input_field_maxlen: usize = 100_000,
    temperature_for_sampling: f64 = 0.2,
    temperature_for_candidates: f64 = 0.2,
    teacher_settings: ?optimize.TeacherSettings = null,

    pub const Options = struct {
        seed: u32 = 0,
        /// `inspect.getsource(type(program))`, shown to the rule writer.
        program_code: []const u8 = "",
        /// DSPy's `prompt_model` (defaults to `ctx`).
        prompt: ?Context = null,
    };

    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            /// `candidate_programs` scores on the full trainset, descending.
            scores: []const f64,
            /// `trial_logs[i]["train_score"]`, when set.
            train_scores: []const ?f64,
        };
    }

    const Run = struct {
        prediction: ?Record,
        steps: []const predict.Trace.Step,
        names: []const predict.NamedParam,
        /// `null` is DSPy's masked `"N/A"`.
        score: ?f64,
        example: *const Example,

        fn greater(_: void, a: Run, b: Run) bool {
            return a.score.? > b.score.?;
        }
    };

    const Bucket = struct {
        runs: []Run,
        gap: f64,
        max: f64,
        avg_gap: f64,

        fn greater(_: void, a: Bucket, b: Bucket) bool {
            if (a.gap != b.gap) return a.gap > b.gap;
            if (a.max != b.max) return a.max > b.max;
            return a.avg_gap > b.avg_gap;
        }
    };

    fn State(comptime P: type) type {
        return struct {
            const Self = @This();
            programs: std.ArrayList(*P) = .empty,
            scores: std.ArrayList([]const f64) = .empty,

            fn average(s: *const @This(), idx: usize) f64 {
                const list = s.scores.items[idx];
                if (list.len == 0) return 0.0;
                var sum: f64 = 0;
                for (list) |x| sum += x;
                return sum / @as(f64, @floatFromInt(list.len));
            }

            /// `top_k_plus_baseline(k)`.
            fn topK(s: *const @This(), out: Allocator, k: usize) Allocator.Error![]const usize {
                const order = try out.alloc(usize, s.programs.items.len);
                for (order, 0..) |*o, i| o.* = i;
                std.sort.insertion(usize, order, s, struct {
                    fn greater(st: *const Self, a: usize, b: usize) bool {
                        return st.average(a) > st.average(b);
                    }
                }.greater);
                const top = order[0..@min(k, order.len)];
                if (top.len > 0 and std.mem.indexOfScalar(usize, top, 0) == null) top[top.len - 1] = 0;
                var unique: std.ArrayList(usize) = .empty;
                for (top) |t| {
                    if (std.mem.indexOfScalar(usize, unique.items, t) == null) try unique.append(out, t);
                }
                return unique.items;
            }

            /// `softmax_sample(rng, program_idxs, temperature)`.
            fn softmaxSample(s: *const @This(), out: Allocator, rng: *pyrandom.Random, idxs: []const usize, temperature: f64) Allocator.Error!usize {
                assert(idxs.len > 0);
                const exps = try out.alloc(f64, idxs.len);
                var sum: f64 = 0;
                for (exps, idxs) |*e, idx| {
                    e.* = @exp(s.average(idx) / temperature);
                    sum += e.*;
                }
                if (sum <= 0) return idxs[rng.randbelow(idxs.len)];
                // random.choices(idxs, weights=probs): bisect_right over the
                // running sums, with hi = n - 1.
                var cum: f64 = 0;
                for (exps) |*e| {
                    cum += e.* / sum;
                    e.* = cum;
                }
                const x = rng.random() * (exps[exps.len - 1] + 0.0);
                var lo: usize = 0;
                var hi: usize = idxs.len - 1;
                while (lo < hi) {
                    const mid = (lo + hi) / 2;
                    if (x < exps[mid]) hi = mid else lo = mid + 1;
                }
                return idxs[lo];
            }
        };
    }

    pub fn compile(
        self: SIMBA,
        comptime P: type,
        out: Allocator,
        ctx: Context,
        student: P,
        trainset: []const Example,
        options: Options,
    ) !Result(P) {
        assert(self.bsize > 0);
        assert(self.num_candidates > 0);
        assert(trainset.len >= self.bsize);
        var rng = pyrandom.Random.init(options.seed);
        var rng_np = nprandom.Generator.init(options.seed);
        var st: State(P) = .{};
        try st.programs.append(out, try clone(P, out, &student));
        try st.scores.append(out, &.{});
        var winning: std.ArrayList(*P) = .empty;
        try winning.append(out, st.programs.items[0]);
        const data_indices = try out.alloc(usize, trainset.len);
        for (data_indices, 0..) |*d, i| d.* = i;
        rng.shuffle(usize, data_indices);
        var instance_idx: usize = 0;
        const train_scores = try out.alloc(?f64, self.max_steps);
        @memset(train_scores, null);
        const start_rollout: i64 = if (ctx.lm_options.rollout_id) |r|
            if (r == .integer) r.integer else return error.InvalidRolloutId
        else
            0;
        var base = ctx;
        base.trace = null;

        for (0..self.max_steps) |_| {
            if (instance_idx + self.bsize > trainset.len) {
                rng.shuffle(usize, data_indices);
                instance_idx = 0;
            }
            const batch = data_indices[instance_idx..][0..self.bsize];
            instance_idx += self.bsize;

            const top = try st.topK(out, self.num_candidates);
            const runs = try out.alloc(Run, self.num_candidates * self.bsize);
            for (0..self.num_candidates) |model| {
                var c = base;
                c.lm_options.rollout_id = .{ .integer = std.math.add(i64, start_rollout, @as(i64, @intCast(model))) catch return error.InvalidRolloutId };
                if (model == 0 and self.teacher_settings != null) {
                    self.teacher_settings.?.apply(&c);
                } else {
                    c.lm_options.temperature = 1.0;
                }
                for (batch, 0..) |ex_i, j| {
                    const chosen = try st.softmaxSample(out, &rng, top, self.temperature_for_sampling);
                    const copy = try clone(P, out, st.programs.items[chosen]);
                    runs[model * self.bsize + j] = try self.wrap(P, out, c, copy, &trainset[ex_i]);
                }
            }

            const all_scores = try out.alloc(f64, runs.len);
            for (all_scores, runs) |*s, r| s.* = r.score.?;
            const p10 = nprandom.percentile(try out.dupe(f64, all_scores), 10);
            const p90 = nprandom.percentile(try out.dupe(f64, all_scores), 90);

            const buckets = try out.alloc(Bucket, self.bsize);
            for (buckets, 0..) |*b, j| {
                const bucket_runs = try out.alloc(Run, self.num_candidates);
                for (bucket_runs, 0..) |*r, m| r.* = runs[j + m * self.bsize];
                std.sort.insertion(Run, bucket_runs, {}, Run.greater);
                const max_score = bucket_runs[0].score.?;
                const min_score = bucket_runs[bucket_runs.len - 1].score.?;
                var sum: f64 = 0;
                for (bucket_runs) |r| sum += r.score.?;
                const avg = sum / @as(f64, @floatFromInt(bucket_runs.len));
                b.* = .{ .runs = bucket_runs, .gap = max_score - min_score, .max = max_score, .avg_gap = max_score - avg };
            }
            std.sort.insertion(Bucket, buckets, {}, Bucket.greater);

            var candidates: std.ArrayList(*P) = .empty;
            for (buckets) |bucket| {
                const src = try st.softmaxSample(out, &rng, try st.topK(out, self.num_candidates), self.temperature_for_candidates);
                const system = try clone(P, out, st.programs.items[src]);
                const preds = try predict.namedPredictors(out, system);
                var num_demos: usize = 0;
                for (preds) |p| num_demos = @max(num_demos, p.param.predict.demos.len);
                const max_demos_tmp: usize = if (self.max_demos > 0) self.max_demos else 3;
                const lam = @as(f64, @floatFromInt(num_demos)) / @as(f64, @floatFromInt(max_demos_tmp));
                var num_drop: usize = @max(rng_np.poisson(lam), @intFromBool(num_demos >= max_demos_tmp));
                num_drop = @min(num_drop, num_demos);
                const drop = try out.alloc(usize, num_drop);
                for (drop) |*d| d.* = rng.randbelow(num_demos);
                for (preds) |p| {
                    var kept: std.ArrayList(Record) = .empty;
                    for (p.param.predict.demos, 0..) |d, i| {
                        if (std.mem.indexOfScalar(usize, drop, i) == null) try kept.append(out, d);
                    }
                    p.param.predict.demos = kept.items;
                }
                const n_strategies: usize = if (self.max_demos > 0) 2 else 1;
                const pick = rng.randbelow(n_strategies);
                if (n_strategies == 2 and pick == 0) {
                    try self.appendDemo(out, bucket.runs, preds, p10);
                } else {
                    self.appendRule(P, out, base, system, bucket.runs, p10, p90, options) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => continue,
                    };
                }
                try candidates.append(out, system);
                if (candidates.items.len >= self.num_candidates + 1) break;
            }

            var best: ?usize = null;
            var best_score: f64 = 0;
            const candidate_scores = try out.alloc([]const f64, candidates.items.len);
            for (candidates.items, candidate_scores, 0..) |cand, *scores, i| {
                const list = try out.alloc(f64, batch.len);
                for (batch, list) |ex_i, *s| s.* = (try self.wrap(P, out, base, cand, &trainset[ex_i])).score.?;
                scores.* = list;
                const avg = mean(list);
                if (best == null or avg > best_score) {
                    best = i;
                    best_score = avg;
                }
            }
            if (best) |b| try winning.append(out, try clone(P, out, candidates.items[b]));
            for (candidates.items, candidate_scores) |cand, scores| {
                try st.programs.append(out, cand);
                try st.scores.append(out, scores);
            }
        }

        const m = winning.items.len - 1;
        const n = self.num_candidates + 1;
        var idxs: std.ArrayList(usize) = .empty;
        for (0..n) |i| {
            const idx: usize = if (m < 1) 0 else @intFromFloat(roundHalfEven(@as(f64, @floatFromInt(i * m)) / @as(f64, @floatFromInt(n - 1))));
            if (std.mem.indexOfScalar(usize, idxs.items, idx) == null) try idxs.append(out, idx);
        }
        const scores = try out.alloc(f64, idxs.items.len);
        const programs = try out.alloc(*P, idxs.items.len);
        var best_idx: usize = 0;
        for (idxs.items, scores, programs, 0..) |w, *s, *prog, i| {
            prog.* = try clone(P, out, winning.items[w]);
            const list = try out.alloc(f64, trainset.len);
            for (trainset, list) |*ex, *x| x.* = (try self.wrap(P, out, base, prog.*, ex)).score.?;
            s.* = mean(list);
            if (i != 0) train_scores[i - 1] = s.*;
            if (s.* > scores[best_idx]) best_idx = i;
        }
        const sorted = try out.dupe(f64, scores);
        std.sort.insertion(f64, sorted, {}, std.sort.desc(f64));
        return .{ .program = programs[best_idx].*, .scores = sorted, .train_scores = train_scores };
    }

    /// `wrap_program(program, metric)(example)`.
    fn wrap(self: SIMBA, comptime P: type, out: Allocator, ctx: Context, program: *P, example: *const Example) Allocator.Error!Run {
        var trace: predict.Trace = .{ .arena = out };
        var c = ctx;
        c.arena = out;
        c.trace = &trace;
        const names = try predict.namedPredictors(out, program);
        const prediction: ?Record = optimize.runProgram(P, c, program, example) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        const score = if (prediction) |*p| self.metric(out, example, p, null) else 0.0;
        return .{ .prediction = prediction, .steps = trace.steps.items, .names = names, .score = score, .example = example };
    }

    /// `append_a_demo`: the best run's last trace step per predictor becomes
    /// an augmented demo of that predictor.
    fn appendDemo(self: SIMBA, out: Allocator, bucket: []Run, preds: []const predict.NamedParam, p10: f64) Allocator.Error!void {
        const good = bucket[0];
        if (good.score.? <= p10) return;
        var names: std.ArrayList([]const u8) = .empty;
        var demos: std.ArrayList(Record) = .empty;
        for (good.steps) |step| {
            const name = nameOf(good.names, step.predictor) orelse continue;
            var demo: Record = .empty;
            try demo.put(out, "augmented", .{ .bool = true });
            var it = step.inputs.iterator();
            while (it.next()) |e| try demo.put(out, e.key_ptr.*, try self.truncate(out, e.value_ptr.*));
            var ot = step.outputs.iterator();
            while (ot.next()) |e| try demo.put(out, e.key_ptr.*, e.value_ptr.*);
            for (names.items, 0..) |existing, i| {
                if (std.mem.eql(u8, existing, name)) {
                    demos.items[i] = demo;
                    break;
                }
            } else {
                try names.append(out, name);
                try demos.append(out, demo);
            }
        }
        for (names.items, demos.items) |name, demo| {
            for (preds) |p| {
                if (!std.mem.eql(u8, p.name, name)) continue;
                const old = p.param.predict.demos;
                const grown = try out.alloc(Record, old.len + 1);
                @memcpy(grown[0..old.len], old);
                grown[old.len] = demo;
                p.param.predict.demos = grown;
            }
        }
    }

    fn truncate(self: SIMBA, out: Allocator, v: Value) Allocator.Error!Value {
        if (self.demo_input_field_maxlen == 0) return v;
        const text = try pyStr(out, v);
        const len = std.unicode.utf8CountCodepoints(text) catch text.len;
        if (len <= self.demo_input_field_maxlen) return v;
        var view = std.unicode.Utf8View.initUnchecked(text).iterator();
        var end: usize = 0;
        for (0..self.demo_input_field_maxlen) |_| end += (view.nextCodepointSlice() orelse break).len;
        return .{ .string = try std.fmt.allocPrint(out, "{s}\n\t\t... <TRUNCATED FOR BREVITY>", .{text[0..end]}) };
    }

    /// `append_a_rule`: ask the prompt model to contrast the best and worst
    /// runs and append its advice to each predictor's instructions.
    fn appendRule(
        self: SIMBA,
        comptime P: type,
        out: Allocator,
        ctx: Context,
        system: *P,
        bucket: []Run,
        p10: f64,
        p90: f64,
        options: Options,
    ) !void {
        _ = self;
        const good = &bucket[0];
        const bad = &bucket[bucket.len - 1];
        if (good.score.? <= p10 or bad.score.? >= p90) return;
        if (good.score.? <= bad.score.?) {
            if (good.score.? > p90) try mask(out, bad) else try mask(out, good);
        }
        const preds = try predict.namedPredictors(out, system);
        var module_names: std.json.Array = .init(out);
        for (preds) |p| try module_names.append(.{ .string = p.name });
        const example = good.example;
        var labels: Record = .empty;
        var it = example.record.iterator();
        while (it.next()) |e| if (!example.isInput(e.key_ptr.*)) try labels.put(out, e.key_ptr.*, e.value_ptr.*);

        const better_outputs = good.prediction orelse return error.StrategyFailed;
        const worse_outputs = bad.prediction orelse Record.empty;
        const ProgramSignature = signature.Signature(struct {
            pub const Input = struct {};
            pub const Output = P.Output;
        });
        const better_feedback_outputs = try maskReasoningFields(out, ProgramSignature.output_fields_list, &better_outputs);
        const worse_feedback_outputs = try maskReasoningFields(out, ProgramSignature.output_fields_list, &worse_outputs);
        var inputs: Record = .empty;
        try inputs.put(out, "program_code", .{ .string = options.program_code });
        try inputs.put(out, "modules_defn", .{ .string = try inspectModules(out, preds) });
        try inputs.put(out, "program_inputs", try dumps(out, .{ .object = try example.inputs(out) }));
        try inputs.put(out, "oracle_metadata", try dumps(out, .{ .object = labels }));
        try inputs.put(out, "better_program_trajectory", try dumps(out, try trajectory(out, good)));
        try inputs.put(out, "better_program_outputs", try dumps(out, .{ .object = better_feedback_outputs }));
        try inputs.put(out, "worse_program_trajectory", try dumps(out, try trajectory(out, bad)));
        try inputs.put(out, "worse_program_outputs", try dumps(out, .{ .object = worse_feedback_outputs }));
        try inputs.put(out, "worse_reward_value", try reward(out, bad.score));
        try inputs.put(out, "better_reward_value", try reward(out, good.score));
        try inputs.put(out, "worse_reward_info", .{ .string = "{}" });
        try inputs.put(out, "better_reward_info", .{ .string = "{}" });
        try inputs.put(out, "module_names", try dumps(out, .{ .array = module_names }));

        var c = options.prompt orelse ctx;
        c.arena = out;
        c.trace = null;
        const feedback: predict.Predict(OfferFeedback) = .{};
        const outputs = try feedback.state.call(c, &inputs);
        const advice = outputs.get("module_advice").?.object;
        for (preds) |p| {
            const tip = advice.get(p.name) orelse continue;
            p.param.predict.instructions = try std.fmt.allocPrint(out, "{s}\n\n{s}", .{ p.param.predict.instructions, tip.string });
        }
    }
};

fn mean(list: []const f64) f64 {
    var sum: f64 = 0;
    for (list) |x| sum += x;
    return sum / @as(f64, @floatFromInt(list.len));
}

fn clone(comptime P: type, out: Allocator, program: *const P) Allocator.Error!*P {
    const copy = try out.create(P);
    copy.* = program.*;
    return copy;
}

fn nameOf(names: []const predict.NamedParam, state: *const predict.PredictState) ?[]const u8 {
    for (names) |n| if (n.param.predict == state) return n.name;
    return null;
}

fn mask(out: Allocator, run: anytype) Allocator.Error!void {
    var na: Record = .empty;
    try na.put(out, "N/A", .{ .string = "Prediction not available" });
    run.steps = &.{};
    run.score = null;
    run.prediction = na;
}

fn reward(out: Allocator, score: ?f64) Allocator.Error!Value {
    const s = score orelse return .{ .string = "N/A" };
    return dumps(out, .{ .float = s });
}

/// `[{"module_name", "inputs", "outputs"}]` for the run's trace steps.
fn trajectory(out: Allocator, run: anytype) Allocator.Error!Value {
    var list: std.json.Array = .init(out);
    for (run.steps) |step| {
        const name = nameOf(run.names, step.predictor) orelse continue;
        var item: Record = .empty;
        try item.put(out, "module_name", .{ .string = name });
        const fields = step.predictor.fields;
        try item.put(out, "inputs", .{ .object = try maskReasoningFields(out, fields[0..step.predictor.input_count], &step.inputs) });
        try item.put(out, "outputs", .{ .object = try maskReasoningFields(out, fields[step.predictor.input_count..], &step.outputs) });
        try list.append(.{ .object = item });
    }
    return .{ .array = list };
}

/// SIMBA's `recursive_mask` sees typed Reasoning objects before DSPy's custom
/// JSON serializer runs. Restore that distinction from field metadata without
/// mutating the serialized prediction/trace or masking ordinary string fields.
fn maskReasoningFields(out: Allocator, fields: []const signature.Field, record: *const Record) Allocator.Error!Record {
    var masked = record.*;
    var copied = false;
    for (fields) |field| {
        if (!field.reasoning_type) continue;
        const item = record.get(field.name) orelse continue;
        if (item == .null) continue;
        if (!copied) {
            masked = try record.clone(out);
            copied = true;
        }
        masked.getPtr(field.name).?.* = .{ .string = "<non-serializable: Reasoning>" };
    }
    return masked;
}

/// `orjson.dumps(v, option=OPT_INDENT_2).decode()`.
fn dumps(out: Allocator, v: Value) Allocator.Error!Value {
    var aw: Writer.Allocating = .init(out);
    pyjson.writeJson(&aw.writer, v, .indent2) catch return error.OutOfMemory;
    return .{ .string = aw.written() };
}

fn pyStr(out: Allocator, v: Value) Allocator.Error![]const u8 {
    if (v == .string) return v.string;
    var aw: Writer.Allocating = .init(out);
    pyjson.writeStr(&aw.writer, v) catch return error.OutOfMemory;
    return aw.written();
}

/// `inspect_modules(program)`.
pub fn inspectModules(out: Allocator, preds: []const predict.NamedParam) ![]const u8 {
    const separator_bytes: [80]u8 = @splat('-');
    const separator: []const u8 = &separator_bytes;
    var parts: std.ArrayList([]const u8) = .empty;
    try parts.append(out, separator);
    for (preds) |p| {
        const view = try p.param.predict.view(out);
        try parts.append(out, try std.fmt.allocPrint(out, "Module {s}", .{p.name}));
        try parts.append(out, "\n\tInput Fields:");
        try parts.append(out, try indentLines(out, try adapter.fieldDescriptionString(out, view.inputs)));
        try parts.append(out, "\tOutput Fields:");
        try parts.append(out, try indentLines(out, try adapter.fieldDescriptionString(out, view.outputs)));
        const instructions = try indentLines(out, try adapter.dedent(out, view.instructions));
        try parts.append(out, try std.fmt.allocPrint(out, "\tOriginal Instructions: {s}", .{instructions}));
        try parts.append(out, separator);
    }
    var aw: Writer.Allocating = .init(out);
    for (parts.items, 0..) |part, i| {
        if (i > 0) aw.writer.writeByte('\n') catch return error.OutOfMemory;
        aw.writer.writeAll(std.mem.trim(u8, part, "\n")) catch return error.OutOfMemory;
    }
    return aw.written();
}

/// `("\n" + "\t" * 2).join([""] + text.splitlines())`.
fn indentLines(out: Allocator, text: []const u8) Allocator.Error![]const u8 {
    var aw: Writer.Allocating = .init(out);
    var i: usize = 0;
    while (i < text.len) {
        const start = i;
        var end = i;
        var next = text.len;
        while (end < text.len) : (end += 1) {
            if (lineBreakLen(text[end..])) |n| {
                next = end + n;
                break;
            }
        }
        aw.writer.print("\n\t\t{s}", .{text[start..end]}) catch return error.OutOfMemory;
        i = next;
    }
    return aw.written();
}

/// Length of the `str.splitlines` line boundary starting `s`, if any.
fn lineBreakLen(s: []const u8) ?usize {
    switch (s[0]) {
        '\r' => return if (s.len > 1 and s[1] == '\n') 2 else 1,
        '\n', 0x0b, 0x0c, 0x1c, 0x1d, 0x1e => return 1,
        else => {},
    }
    if (std.mem.startsWith(u8, s, "\xc2\x85")) return 2;
    if (std.mem.startsWith(u8, s, "\xe2\x80\xa8") or std.mem.startsWith(u8, s, "\xe2\x80\xa9")) return 3;
    return null;
}

/// Python's `round(x)` for non-negative `x`: ties to even.
fn roundHalfEven(x: f64) f64 {
    assert(x >= 0);
    const floor = @floor(x);
    const diff = x - floor;
    if (diff < 0.5) return floor;
    if (diff > 0.5) return floor + 1;
    return if (@mod(floor, 2) == 0) floor else floor + 1;
}

test "feedback masks typed Reasoning without masking ordinary strings or mutating traces" {
    const Sig = signature.Signature(struct {
        pub const Input = struct {};
        pub const Output = struct {
            analysis: ?signature.Reasoning,
            direct: signature.Reasoning,
            empty: ?signature.Reasoning,
            reasoning: []const u8,
            absent: ?signature.Reasoning,
        };
    });
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const out = arena.allocator();
    const original = try value.toRecord(out, Sig.Output, .{
        .analysis = .{ .content = "Actual reasoning." },
        .direct = .{ .content = "Direct reasoning." },
        .empty = .{ .content = "" },
        .reasoning = "An ordinary string.",
        .absent = null,
    });
    const masked = try maskReasoningFields(out, Sig.output_fields_list, &original);
    try std.testing.expectEqualStrings("<non-serializable: Reasoning>", masked.get("analysis").?.string);
    try std.testing.expectEqualStrings("<non-serializable: Reasoning>", masked.get("direct").?.string);
    try std.testing.expectEqualStrings("<non-serializable: Reasoning>", masked.get("empty").?.string);
    try std.testing.expectEqualStrings("An ordinary string.", masked.get("reasoning").?.string);
    try std.testing.expect(masked.get("absent").? == .null);
    try std.testing.expectEqualStrings("Actual reasoning.", original.get("analysis").?.string);
    try std.testing.expectEqualStrings("Direct reasoning.", original.get("direct").?.string);
    try std.testing.expectEqualStrings("", original.get("empty").?.string);
}

test "roundHalfEven matches Python round" {
    try std.testing.expectEqual(@as(f64, 0), roundHalfEven(0.5));
    try std.testing.expectEqual(@as(f64, 2), roundHalfEven(1.5));
    try std.testing.expectEqual(@as(f64, 2), roundHalfEven(2.5));
    try std.testing.expectEqual(@as(f64, 1), roundHalfEven(0.6666));
}
