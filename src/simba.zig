//! `dspy.SIMBA` (DSPy 3.5.0): stochastic introspective mini-batch ascent.
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
//! DSPy `num_threads` maps to `rollouts`: trajectory, candidate and final
//! evaluations then run on a startup-created native pool, results kept in
//! submission order, with every random draw made before execution as in DSPy.
//! Metrics can return either a numeric score or a score with rule-writing metadata.
//! `np.exp` is Zig's `@exp`.
//!
//! Upstream (DSPy 3.5.0): dspy/teleprompt/simba.py

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const adapter = @import("adapter.zig");
const lm_mod = @import("lm.zig");
const nprandom = @import("nprandom.zig");
const optimize = @import("optimize.zig");
const parallel = @import("parallel.zig");
const predict = @import("predict.zig");
const pyjson = @import("pyjson.zig");
const pyrandom = @import("pyrandom.zig");
const signature = @import("signature.zig");
const value = @import("value.zig");

const Context = predict.Context;
const Example = optimize.Example;
const LM = lm_mod.LM;
const Metric = optimize.Metric;
const Record = value.Record;
const Value = @import("runtime_value.zig").Value;

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
    metric: ?Metric = null,
    /// Set exactly one of `metric` and `feedback_metric`.
    feedback_metric: ?FeedbackMetric = null,
    bsize: usize = 32,
    num_candidates: usize = 6,
    max_steps: usize = 8,
    max_demos: usize = 4,
    demo_input_field_maxlen: usize = 100_000,
    temperature_for_sampling: f64 = 0.2,
    temperature_for_candidates: f64 = 0.2,
    teacher_settings: ?optimize.TeacherSettings = null,
    /// Borrowed model for OfferFeedback only. Null uses the compile settings LM.
    prompt_model: ?LM = null,
    /// DSPy `num_threads`; null runs rollouts serially.
    rollouts: ?Rollouts = null,

    /// A `parallel.Parallel(SIMBA.Run, limits).runner()` plus per-attempt
    /// contexts; other settings come from each rollout. Teacher and
    /// predictor-bound LMs are shared and must be thread-safe, as must the
    /// metric. Predictions and traces are allocated from `out`, which must
    /// then be thread-safe (an ArenaAllocator over a thread-safe child is).
    pub const Rollouts = parallel.Rollouts(Run);

    /// Return metadata backed by the supplied arena or immutable static data.
    /// The score is separate, so `metadata` contains only the additional fields.
    pub const FeedbackMetric = *const fn (
        Allocator,
        *const Example,
        ?*const optimize.RecordPrediction,
    ) value.ParseError!Feedback;

    pub const Feedback = struct {
        score: f64,
        metadata: Record = .empty,
    };

    pub const Options = struct {
        seed: u32 = 0,
        /// `inspect.getsource(type(program))`, shown to the rule writer.
        program_code: []const u8 = "",
    };

    /// Capture the explicit prompt model, or the current settings LM if omitted.
    /// Only the LM handle is retained; its backing model must outlive compile.
    /// An unconfigured handle leaves feedback to the compile settings LM.
    pub fn init(config: SIMBA, settings: Context, prompt_model: ?LM) SIMBA {
        var self = config;
        const model = prompt_model orelse settings.lm;
        self.prompt_model = if (model.sameInstance(LM.unconfigured)) null else model;
        return self;
    }

    pub fn ScoredProgram(comptime P: type) type {
        return struct {
            program: *const P,
            score: f64,

            fn higher(_: void, a: @This(), b: @This()) bool {
                return a.score > b.score;
            }
        };
    }

    pub fn Result(comptime P: type) type {
        return struct {
            program: P,
            /// Final programs ranked by full-trainset score, with stable ties.
            /// Program snapshots live in the compile allocator.
            candidate_programs: []const ScoredProgram(P),
            /// Scores in the same order as candidate_programs.
            scores: []const f64,
            /// `trial_logs[i]["train_score"]`, when set.
            train_scores: []const ?f64,
        };
    }

    pub const Run = struct {
        /// The program's Prediction as the metric saw it. Rule prompts
        /// serialize `fields()` only, so its usage never reaches an LM.
        prediction: ?optimize.RecordPrediction,
        steps: []const predict.Trace.Step,
        names: []const predict.NamedParam,
        metadata: Record = .empty,
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
            fn softmaxSample(s: *const @This(), out: Allocator, rng: *pyrandom.Random, idxs: []const usize, temperature: f64) (Allocator.Error || error{InvalidSamplingTemperature})!usize {
                assert(idxs.len > 0);
                // Python divides before calling NumPy exp, so either signed
                // zero fails at the sampling stage instead of becoming NaN.
                if (temperature == 0) return error.InvalidSamplingTemperature;
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
                const total = cum + 0.0;
                // Python rejects nonfinite cumulative weights before drawing.
                if (!std.math.isFinite(total)) return error.InvalidSamplingTemperature;
                const x = rng.random() * total;
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
        if ((self.metric == null) == (self.feedback_metric == null))
            return error.InvalidMetricConfiguration;
        if (trainset.len < self.bsize) return error.InvalidOptions;
        if (self.max_steps > 0) {
            if (self.bsize == 0 or self.num_candidates == 0) return error.InvalidOptions;
        }
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

        for (0..self.max_steps) |_| {
            if (instance_idx + self.bsize > trainset.len) {
                rng.shuffle(usize, data_indices);
                instance_idx = 0;
            }
            const batch = data_indices[instance_idx..][0..self.bsize];
            instance_idx += self.bsize;

            // Resolve the baseline binding only when sampling. Candidate
            // scoring and final evaluation keep the original program bindings.
            const baseline_lm = try predict.getLMFromPredictors(try predict.namedPredictors(out, st.programs.items[0]));
            const teacher = self.teacher_settings orelse optimize.TeacherSettings{};
            const has_teacher_settings = teacher.lm != null or teacher.adapter != null or
                teacher.max_errors != null or teacher.max_trace_size != null;
            // The first sampler sets a rollout on the selected model. When
            // that is the baseline, its later evaluations see the same ID.
            // Native LM handles are borrowed; model configuration stays local.
            const baseline_engine = baseline_lm orelse base.lm;
            if (has_teacher_settings and (teacher.lm orelse baseline_engine).sameInstance(baseline_engine)) {
                base.lm_options.rollout_id = .{ .integer = start_rollout };
            }
            const top = try st.topK(out, self.num_candidates);
            const runs = try out.alloc(Run, self.num_candidates * self.bsize);
            const rollout_jobs = try out.alloc(Job(P), runs.len);
            for (0..self.num_candidates) |model| {
                var c = base;
                c.lm_options.rollout_id = .{ .integer = std.math.add(i64, start_rollout, @as(i64, @intCast(model))) catch return error.InvalidRolloutId };
                c.before_settings_rollout = null;
                const use_teacher = model == 0 and has_teacher_settings;
                // A nonempty settings record reserves the first sampler,
                // but SIMBA reads only its LM (not adapter or error settings).
                const selected_lm = if (use_teacher) teacher.lm orelse baseline_lm else baseline_lm;
                if (selected_lm) |selected| c.lm = selected;
                if (!use_teacher) c.lm_options.temperature = 1.0;
                for (batch, 0..) |ex_i, j| {
                    const chosen = try st.softmaxSample(out, &rng, top, self.temperature_for_sampling);
                    const copy = try clone(P, out, st.programs.items[chosen]);
                    if (selected_lm) |selected| try predict.setLM(out, copy, selected);
                    rollout_jobs[model * self.bsize + j] = .{ .ctx = c, .shared_lm = selected_lm != null, .program = copy, .example = &trainset[ex_i] };
                }
            }
            try self.runAll(P, out, rollout_jobs, runs);

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
                const strategy: anyerror!void = if (n_strategies == 2 and pick == 0)
                    self.appendDemo(out, bucket.runs, preds, p10)
                else
                    self.appendRule(P, out, base, system, bucket.runs, p10, p90, options);
                strategy catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => continue,
                };
                try candidates.append(out, system);
                if (candidates.items.len >= self.num_candidates + 1) break;
            }

            var best: ?usize = null;
            var best_score: f64 = 0;
            const candidate_runs = try self.runPrograms(P, out, base, candidates.items, trainset, batch);
            const candidate_scores = try out.alloc([]const f64, candidates.items.len);
            for (candidate_scores, 0..) |*scores, i| {
                const list = try out.alloc(f64, batch.len);
                for (list, candidate_runs[i * batch.len ..][0..batch.len]) |*s, r| s.* = r.score.?;
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
        // With no winning step, the source's repeated baseline indices
        // collapse to one candidate, even when num_candidates is zero.
        const n = if (m == 0) 1 else try std.math.add(usize, self.num_candidates, 1);
        var idxs: std.ArrayList(usize) = .empty;
        for (0..n) |i| {
            const idx: usize = if (m < 1) 0 else @intFromFloat(roundHalfEven(@as(f64, @floatFromInt(i * m)) / @as(f64, @floatFromInt(n - 1))));
            if (std.mem.indexOfScalar(usize, idxs.items, idx) == null) try idxs.append(out, idx);
        }
        const ranked = try out.alloc(ScoredProgram(P), idxs.items.len);
        const programs = try out.alloc(*P, idxs.items.len);
        var best_idx: usize = 0;
        for (idxs.items, programs) |w, *prog| prog.* = try clone(P, out, winning.items[w]);
        const all = try out.alloc(usize, trainset.len);
        for (all, 0..) |*index, i| index.* = i;
        const final_runs = try self.runPrograms(P, out, base, programs, trainset, all);
        for (ranked, programs, 0..) |*row, program, i| {
            const list = try out.alloc(f64, trainset.len);
            for (list, final_runs[i * trainset.len ..][0..trainset.len]) |*x, r| x.* = r.score.?;
            const score = mean(list);
            row.* = .{ .program = program, .score = score };
            if (i != 0) train_scores[i - 1] = score;
            if (score > ranked[best_idx].score) best_idx = i;
        }
        const best = programs[best_idx].*;
        std.sort.insertion(ScoredProgram(P), ranked, {}, ScoredProgram(P).higher);
        const scores = try out.alloc(f64, ranked.len);
        for (scores, ranked) |*score, row| score.* = row.score;
        return .{
            .program = best,
            .candidate_programs = ranked,
            .scores = scores,
            .train_scores = train_scores,
        };
    }

    fn Job(comptime P: type) type {
        return struct {
            ctx: Context,
            /// Explicit sampler LMs are shared; otherwise each attempt brings its own.
            shared_lm: bool,
            program: *P,
            example: *const Example,
        };
    }

    /// Every program on `indices` of trainset, program-major, under ctx.
    fn runPrograms(
        self: *const SIMBA,
        comptime P: type,
        out: Allocator,
        ctx: Context,
        programs: []const *P,
        trainset: []const Example,
        indices: []const usize,
    ) ![]Run {
        const jobs = try out.alloc(Job(P), programs.len * indices.len);
        for (programs, 0..) |program, p| {
            for (indices, 0..) |index, i| jobs[p * indices.len + i] = .{
                .ctx = ctx,
                .shared_lm = false,
                .program = program,
                .example = &trainset[index],
            };
        }
        const runs = try out.alloc(Run, jobs.len);
        try self.runAll(P, out, jobs, runs);
        return runs;
    }

    /// DSPy `dspy.Parallel(...)(exec_pairs)`: ordered results.
    fn runAll(self: *const SIMBA, comptime P: type, out: Allocator, jobs: []const Job(P), runs: []Run) !void {
        assert(jobs.len == runs.len);
        if (jobs.len == 0) return;
        const rollouts = self.rollouts orelse {
            for (jobs, runs) |job, *run| run.* = try self.wrap(P, out, job.ctx, job.program, job.example);
            return;
        };
        const Call = struct {
            simba: *const SIMBA,
            out: Allocator,
            job: *const Job(P),

            fn invoke(ptr: *anyopaque, attempt: parallel.Attempt, run: *Run) anyerror!void {
                const call: *const @This() = @ptrCast(@alignCast(ptr));
                try attempt.cancellation.check();
                const c = try call.simba.rollouts.?.context(call.job.ctx, attempt, call.job.shared_lm);
                run.* = try call.simba.wrap(P, call.out, c, call.job.program, call.job.example);
            }
        };
        const calls = try out.alloc(Call, jobs.len);
        const tasks = try out.alloc(parallel.TaskFor(Run), jobs.len);
        for (calls, tasks, jobs) |*call, *task, *job| {
            call.* = .{ .simba = self, .out = out, .job = job };
            task.* = .{ .ptr = call, .invoke = Call.invoke };
        }
        try rollouts.run(tasks, runs);
    }

    /// `wrap_program(program, metric)(example)`.
    fn wrap(self: SIMBA, comptime P: type, out: Allocator, ctx: Context, program: *P, example: *const Example) Allocator.Error!Run {
        var trace: predict.Trace = .{ .arena = out };
        var c = ctx;
        c.arena = out;
        c.trace = &trace;
        const names = try predict.namedPredictors(out, program);
        const prediction: ?optimize.RecordPrediction = optimize.runProgram(P, c, program, example) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        // Source invokes the metric even after program failure, with None.
        // Scoring failure must not discard a successful prediction or trace.
        const feedback = self.scoreRun(out, example, if (prediction) |*p| p else null) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => Feedback{ .score = 0.0 },
        };
        return .{
            .prediction = prediction,
            .steps = trace.steps.items,
            .names = names,
            .score = feedback.score,
            .metadata = feedback.metadata,
            .example = example,
        };
    }

    fn scoreRun(
        self: SIMBA,
        out: Allocator,
        example: *const Example,
        prediction: ?*const optimize.RecordPrediction,
    ) value.ParseError!Feedback {
        if (self.feedback_metric) |metric| return metric(out, example, prediction);
        return .{ .score = try self.metric.?(out, example, prediction, null) };
    }

    /// `append_a_demo`: the best run's last trace step per predictor becomes
    /// an augmented demo of that predictor.
    fn appendDemo(self: SIMBA, out: Allocator, bucket: []Run, preds: []const predict.NamedParam, p10: f64) optimize.TraceDemoError!void {
        const good = bucket[0];
        if (good.score.? <= p10) return;
        const NamedDemo = struct { name: []const u8, demo: Record };
        var demos: std.ArrayList(NamedDemo) = .empty;
        defer demos.deinit(out);
        errdefer for (demos.items) |*item| item.demo.deinit(out);
        for (good.steps) |step| {
            var demo = try optimize.traceDemo(out, step, self.demo_input_field_maxlen);
            const name = nameOf(good.names, step.predictor) orelse {
                demo.deinit(out);
                continue;
            };
            for (demos.items) |*existing| {
                if (std.mem.eql(u8, existing.name, name)) {
                    existing.demo.deinit(out);
                    existing.demo = demo;
                    break;
                }
            } else {
                errdefer demo.deinit(out);
                try demos.append(out, .{ .name = name, .demo = demo });
            }
        }
        for (demos.items) |item| {
            for (preds) |p| {
                if (!std.mem.eql(u8, p.name, item.name)) continue;
                const old = p.param.predict.demos;
                const grown = try out.alloc(Record, old.len + 1);
                @memcpy(grown[0..old.len], old);
                grown[old.len] = item.demo;
                p.param.predict.demos = grown;
            }
        }
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
        const good = &bucket[0];
        const bad = &bucket[bucket.len - 1];
        if (good.score.? <= p10 or bad.score.? >= p90) return;
        if (good.score.? <= bad.score.?) {
            if (good.score.? > p90) try mask(out, bad) else try mask(out, good);
        }
        const preds = try predict.namedPredictors(out, system);
        var module_names: @import("runtime_value.zig").Array = .init(out);
        for (preds) |p| try module_names.append(.{ .string = p.name });
        const example = good.example;
        var labels: Record = .empty;
        var it = example.record.iterator();
        while (it.next()) |e| if (!example.isInput(e.key_ptr.*)) try labels.put(out, e.key_ptr.*, e.value_ptr.*);

        const better_outputs = (good.prediction orelse return error.StrategyFailed).fields().*;
        const worse_outputs = if (bad.prediction) |prediction| prediction.fields().* else Record.empty;
        var inputs: Record = .empty;
        try inputs.put(out, "program_code", .{ .string = options.program_code });
        try inputs.put(out, "modules_defn", .{ .string = try inspectModules(out, preds) });
        try inputs.put(out, "program_inputs", try dumps(out, .{ .object = try example.inputs(out) }));
        try inputs.put(out, "oracle_metadata", try dumps(out, .{ .object = labels }));
        try inputs.put(out, "better_program_trajectory", try dumps(out, try trajectory(out, good)));
        try inputs.put(out, "better_program_outputs", try dumps(out, .{ .object = better_outputs }));
        try inputs.put(out, "worse_program_trajectory", try dumps(out, try trajectory(out, bad)));
        try inputs.put(out, "worse_program_outputs", try dumps(out, .{ .object = worse_outputs }));
        try inputs.put(out, "worse_reward_value", try reward(out, bad.score));
        try inputs.put(out, "better_reward_value", try reward(out, good.score));
        try inputs.put(out, "worse_reward_info", try rewardInfo(out, bad.metadata));
        try inputs.put(out, "better_reward_info", try rewardInfo(out, good.metadata));
        try inputs.put(out, "module_names", try dumps(out, .{ .array = module_names }));

        var c = ctx;
        c.arena = out;
        c.lm = self.prompt_model orelse ctx.lm;
        // OfferFeedback is a fresh predictor, not a call to the student.
        c.lm_override = null;
        c.lm_override_present = false;
        c.trace = null;
        const feedback: predict.Predict(OfferFeedback) = .{};
        const outputs = try feedback.state.call(c, &inputs);
        const advice = outputs.get("module_advice").?.object;
        for (preds) |p| {
            const tip = advice.get(p.name) orelse continue;
            const instruction = try std.fmt.allocPrint(out, "{s}\n\n{s}", .{ p.param.predict.instructions, tip.string });
            try p.param.predict.forkSignatureClass(out);
            p.param.predict.instructions = instruction;
        }
    }
};

fn mean(list: []const f64) f64 {
    if (list.len == 0) return 0;
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
    run.prediction = try optimize.RecordPrediction.init(out, na);
}

fn rewardInfo(out: Allocator, metadata: Record) (@import("feedback.zig").Error || error{ParseValue})!Value {
    if (metadata.count() == 0) return .{ .string = "{}" };
    return dumps(out, .{ .object = metadata });
}

fn reward(out: Allocator, score: ?f64) (@import("feedback.zig").Error || error{ParseValue})!Value {
    const s = score orelse return .{ .string = "N/A" };
    return dumps(out, .{ .float = s });
}

/// `[{"module_name", "inputs", "outputs"}]` for the run's trace steps.
fn trajectory(out: Allocator, run: anytype) Allocator.Error!Value {
    var list: @import("runtime_value.zig").Array = .init(out);
    for (run.steps) |step| {
        const name = nameOf(run.names, step.predictor) orelse continue;
        var item: Record = .empty;
        try item.put(out, "module_name", .{ .string = name });
        try item.put(out, "inputs", .{ .object = step.inputs });
        try item.put(out, "outputs", .{ .object = step.outputs });
        try list.append(.{ .object = item });
    }
    return .{ .array = list };
}

/// `orjson.dumps(recursive_mask(v), option=OPT_INDENT_2).decode()`.
fn dumps(out: Allocator, v: Value) (@import("feedback.zig").Error || error{ParseValue})!Value {
    const masked = try @import("feedback.zig").maskValue(out, v);
    var aw: Writer.Allocating = .init(out);
    pyjson.writeJson(&aw.writer, masked, .orjson_indent2) catch |err| return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        else => |failure| failure,
    };
    return .{ .string = aw.written() };
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
    const serialized = try dumps(out, .{ .object = original });
    const masked = (try std.json.parseFromSliceLeaky(Value, out, serialized.string, .{})).object;
    try std.testing.expectEqualStrings("<non-serializable: Reasoning>", masked.get("analysis").?.string);
    try std.testing.expectEqualStrings("<non-serializable: Reasoning>", masked.get("direct").?.string);
    try std.testing.expectEqualStrings("<non-serializable: Reasoning>", masked.get("empty").?.string);
    try std.testing.expectEqualStrings("An ordinary string.", masked.get("reasoning").?.string);
    try std.testing.expect(masked.get("absent").? == .null);
    try std.testing.expectEqualStrings("Actual reasoning.", (try signature.Reasoning.fromValue(original.get("analysis").?)).content);
    try std.testing.expectEqualStrings("Direct reasoning.", (try signature.Reasoning.fromValue(original.get("direct").?)).content);
    try std.testing.expectEqualStrings("", (try signature.Reasoning.fromValue(original.get("empty").?)).content);
}

test "SIMBA metric failures keep successful predictions and traces but allocation failures remain fatal" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = predict.Predict(Sig);
    const FailingMetric = struct {
        fn failure(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            return error.ParseValue;
        }

        fn allocationFailure(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            return error.OutOfMemory;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{ "<answer>ok</answer>", "<answer>ok</answer>" } };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm(), .adapter = .xml };
    const example = try Example.fromJson(arena, "{\"question\":\"q\",\"answer\":\"ok\"}", &.{"question"});
    var program: P = .{};
    var optimizer: SIMBA = .{ .metric = FailingMetric.failure };
    const run = try optimizer.wrap(P, arena, ctx, &program, &example);
    try std.testing.expectEqual(@as(f64, 0), run.score.?);
    try std.testing.expectEqualStrings("ok", run.prediction.?.get("answer").?.string);
    try std.testing.expectEqual(@as(usize, 1), run.steps.len);
    try std.testing.expectEqualStrings("q", run.steps[0].inputs.get("question").?.string);
    try std.testing.expectEqualStrings("ok", run.steps[0].outputs.get("answer").?.string);
    try std.testing.expectEqual(@as(usize, 1), script.index);

    optimizer.metric = FailingMetric.allocationFailure;
    try std.testing.expectError(error.OutOfMemory, optimizer.wrap(P, arena, ctx, &program, &example));
    try std.testing.expectEqual(@as(usize, 2), script.index);
}

test "SIMBA lets metrics score missing predictions after program failure" {
    const Program = struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };

        pub fn forward(_: *const @This(), _: Context, _: Input) error{ParseValue}!Output {
            return error.ParseValue;
        }
    };
    const MissingPredictionMetric = struct {
        fn score(_: Allocator, example: *const Example, prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            if (prediction != null) return error.ParseValue;
            return example.record.get("fallback").?.float;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const example = try Example.fromJson(arena, "{\"question\":\"q\",\"fallback\":0.375}", &.{"question"});
    var program: Program = .{};
    const optimizer: SIMBA = .{ .metric = MissingPredictionMetric.score };
    const run = try optimizer.wrap(Program, arena, .{ .arena = arena, .lm = script.lm() }, &program, &example);
    try std.testing.expect(run.prediction == null);
    try std.testing.expectEqual(@as(f64, 0.375), run.score.?);
    try std.testing.expectEqual(@as(usize, 0), run.steps.len);
    try std.testing.expectEqual(@as(usize, 0), script.index);
}

test "SIMBA binds trace demos before predictor lookup and drops invalid candidates before scoring" {
    const P = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const TraceCase = enum { output_base, truncated_input_base, mapped_duplicate, unmapped_duplicate };
    const Program = struct {
        predictor: P = .{},
        trace_case: TraceCase = .output_base,
        compiled: bool = false,

        const unmapped: P = .{};
        pub const Input = P.Input;
        pub const Output = struct { reward: f64 };

        pub fn forward(self: *const @This(), ctx: Context, input: Input) !Output {
            if (ctx.lm_options.rollout_id) |rollout_id| {
                if (rollout_id.integer != 0) return .{ .reward = 0 };
                const trace = ctx.trace.?;
                var inputs = try value.toRecord(ctx.arena, Input, input);
                var outputs: Record = .empty;
                try outputs.put(ctx.arena, "answer", .{ .string = "first" });
                try trace.record(&self.predictor.state, &inputs, &outputs);
                const base = try Example.fromJson(ctx.arena, "{\"answer\":\"stale\",\"question\":\"old\",\"augmented\":false,\"inherited\":7}", &.{});
                const base_owner = if (self.trace_case == .truncated_input_base) &inputs else &outputs;
                try base_owner.put(ctx.arena, "base", .{ .object = base.record });
                try outputs.put(ctx.arena, "answer", .{ .string = "teacher" });
                try trace.record(&self.predictor.state, &inputs, &outputs);
                if (self.trace_case == .mapped_duplicate or self.trace_case == .unmapped_duplicate) {
                    const malformed = try value.toRecord(ctx.arena, struct { augmented: bool }, .{ .augmented = false });
                    const state = if (self.trace_case == .mapped_duplicate) &self.predictor.state else &unmapped.state;
                    try trace.record(state, &inputs, &malformed);
                }
                return .{ .reward = 1 };
            }
            return .{ .reward = if (self.predictor.state.demos.len == 0) 0 else 1 };
        }
    };
    const MetricCheck = struct {
        var calls: usize = 0;

        fn score(_: Allocator, _: *const Example, pred: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            calls += 1;
            return pred.?.get("reward").?.float;
        }
    };

    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm() };
    const train = try Example.fromJson(arena, "{\"question\":\"é长rest\"}", &.{"question"});
    const optimizer: SIMBA = .{
        .metric = MetricCheck.score,
        .max_steps = 1,
        .num_candidates = 2,
        .bsize = 1,
        .demo_input_field_maxlen = 2,
    };
    // Seed 2 selects append_a_demo after two rollout draws and one source draw.
    for ([_]TraceCase{ .output_base, .truncated_input_base, .mapped_duplicate, .unmapped_duplicate }) |trace_case| {
        MetricCheck.calls = 0;
        const result = try optimizer.compile(Program, arena, ctx, .{ .trace_case = trace_case }, &.{train}, .{ .seed = 2 });
        if (trace_case == .output_base or trace_case == .truncated_input_base) {
            // Two rollouts, one candidate mini-batch evaluation, two final evaluations.
            try std.testing.expectEqual(@as(usize, 5), MetricCheck.calls);
            try std.testing.expectEqualSlices(f64, &.{ 1, 0 }, result.scores);
            try std.testing.expectEqual(@as(usize, 1), result.program.predictor.state.demos.len);
            const demo = result.program.predictor.state.demos[0];
            try std.testing.expect(!demo.contains("base"));
            try std.testing.expect(demo.get("augmented").?.bool);
            if (trace_case == .output_base) {
                try std.testing.expectEqual(@as(i64, 7), demo.get("inherited").?.integer);
            } else {
                // A truncated input base is a string by the time Example binds it.
                try std.testing.expect(!demo.contains("inherited"));
            }
            try std.testing.expectEqualStrings("teacher", demo.get("answer").?.string);
            try std.testing.expectEqualStrings("é长\n\t\t... <TRUNCATED FOR BREVITY>", demo.get("question").?.string);
        } else {
            // Two rollouts and one final baseline evaluation; no candidate scoring.
            try std.testing.expectEqual(@as(usize, 3), MetricCheck.calls);
            try std.testing.expectEqualSlices(f64, &.{0}, result.scores);
            try std.testing.expectEqual(@as(usize, 0), result.program.predictor.state.demos.len);
            try std.testing.expectEqual(@as(?f64, null), result.train_scores[0]);
        }
    }
}

test "SIMBA scores only the baseline when no search steps are requested" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = predict.Predict(Sig);
    const MetricCheck = struct {
        fn score(_: Allocator, _: *const Example, result: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            const answer = (result orelse return 0).get("answer") orelse return 0;
            return if (std.mem.eql(u8, answer.string, "yes")) 1 else 0;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{"<answer>yes</answer>"} };
    const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm() };
    const train = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"ok?\",\"answer\":\"yes\"}", &.{"question"}),
    };

    var optimizer: SIMBA = .{
        .metric = MetricCheck.score,
        .max_steps = 0,
        .num_candidates = 0,
        .bsize = 1,
    };
    optimizer.temperature_for_sampling = 0.0;
    optimizer.temperature_for_candidates = 0.0;
    const scored = try optimizer.compile(P, arena, ctx, .{}, &train, .{});
    try std.testing.expectEqualSlices(f64, &.{1}, scored.scores);
    try std.testing.expectEqual(@as(usize, 1), script.index);
    optimizer.bsize = 0;
    optimizer.num_candidates = 1;
    const empty = try optimizer.compile(P, arena, ctx, .{}, &.{}, .{});
    try std.testing.expectEqualSlices(f64, &.{0}, empty.scores);
    try std.testing.expectEqual(@as(usize, 1), script.index);

    optimizer.max_steps = 1;
    try std.testing.expectError(error.InvalidOptions, optimizer.compile(P, arena, ctx, .{}, &.{}, .{}));
    optimizer.bsize = 1;
    optimizer.num_candidates = 0;
    try std.testing.expectError(error.InvalidOptions, optimizer.compile(P, arena, ctx, .{}, &train, .{}));
}

test "SIMBA rejects zero temperatures at the respective sampling stages" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = predict.Predict(Sig);
    const CountingMetric = struct {
        var calls: usize = 0;
        fn score(_: Allocator, _: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) error{}!f64 {
            calls += 1;
            return 1;
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const train = [_]Example{
        try Example.fromJson(arena, "{\"question\":\"ok?\",\"answer\":\"yes\"}", &.{"question"}),
    };
    for ([_]bool{ true, false }) |zero_before_rollout| {
        CountingMetric.calls = 0;
        var script: @import("lm.zig").ScriptedLM = .{ .responses = &.{"<answer>yes</answer>"} };
        const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm() };
        const optimizer: SIMBA = .{
            .metric = CountingMetric.score,
            .max_steps = 1,
            .num_candidates = 1,
            .bsize = 1,
            .temperature_for_sampling = if (zero_before_rollout) 0.0 else 0.2,
            .temperature_for_candidates = if (zero_before_rollout) 0.2 else -0.0,
        };
        const result = optimizer.compile(P, arena, ctx, .{}, &train, .{});
        const expected_calls: usize = if (zero_before_rollout) 0 else 1;
        try std.testing.expectEqual(expected_calls, script.index);
        try std.testing.expectEqual(expected_calls, CountingMetric.calls);
        try std.testing.expectError(error.InvalidSamplingTemperature, result);
    }
}

test "SIMBA rejects nonfinite sampling weights at the respective sampling stages" {
    const P = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const CountingMetric = struct {
        var calls: usize = 0;

        fn score(_: Allocator, example: *const Example, _: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) error{}!f64 {
            calls += 1;
            return @floatFromInt(example.record.get("reward").?.integer);
        }
    };
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const train = try Example.fromJson(arena, "{\"question\":\"ok?\",\"reward\":1000}", &.{"question"});
    const nan = std.math.nan(f64);
    const responses: [5][]const u8 = @splat("<answer>yes</answer>");
    const cases = [_]struct {
        sampling: f64,
        candidates: f64,
        steps: usize,
        num_candidates: usize,
        calls: usize,
    }{
        .{ .sampling = nan, .candidates = 0.2, .steps = 1, .num_candidates = 1, .calls = 0 },
        .{ .sampling = 0.2, .candidates = nan, .steps = 1, .num_candidates = 1, .calls = 1 },
        // The first batch registers a score of 1000, so exp(1000 / 0.2)
        // overflows at the respective sampling stage of the second batch.
        .{ .sampling = 0.2, .candidates = 1e6, .steps = 2, .num_candidates = 2, .calls = 3 },
        .{ .sampling = 1e6, .candidates = 0.2, .steps = 2, .num_candidates = 2, .calls = 5 },
    };
    for (cases) |case| {
        CountingMetric.calls = 0;
        var script: @import("lm.zig").ScriptedLM = .{ .responses = &responses };
        const ctx: Context = .{ .workspace = @import("workspace.zig").forTests(), .arena = arena, .lm = script.lm() };
        const optimizer: SIMBA = .{
            .metric = CountingMetric.score,
            .max_steps = case.steps,
            .num_candidates = case.num_candidates,
            .bsize = 1,
            .max_demos = 0,
            .temperature_for_sampling = case.sampling,
            .temperature_for_candidates = case.candidates,
        };
        try std.testing.expectError(error.InvalidSamplingTemperature, optimizer.compile(P, arena, ctx, .{}, &.{train}, .{}));
        try std.testing.expectEqual(case.calls, script.index);
        try std.testing.expectEqual(case.calls, CountingMetric.calls);
    }
}

test "SIMBA sampler settings preserve bound model precedence and the ambient adapter" {
    const Sig = signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    });
    const P = predict.Predict(Sig);
    const MetricCheck = struct {
        fn score(_: Allocator, _: *const Example, prediction: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            return if (prediction) |p| if (std.mem.eql(u8, p.get("answer").?.string, "yes")) 1 else 0 else 0;
        }
    };
    const lm = @import("lm.zig");
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const train = [_]Example{try Example.fromJson(arena, "{\"question\":\"ok?\",\"answer\":\"yes\"}", &.{"question"})};
    for (0..4) |scenario| {
        var base_history: lm.History = .{ .arena = arena };
        var teacher_history: lm.History = .{ .arena = arena };
        var base: lm.KeywordLM = .{ .rules = &.{}, .default = "<answer>yes</answer>", .history = &base_history };
        var teacher: lm.KeywordLM = .{ .rules = &.{}, .default = "<answer>yes</answer>", .history = &teacher_history };
        var student: P = .{};
        if (scenario == 3) student.state.lm = base.lm();
        const optimizer: SIMBA = .{
            .metric = MetricCheck.score,
            .max_steps = 1,
            .bsize = 1,
            .num_candidates = 2,
            .teacher_settings = switch (scenario) {
                0 => null,
                1 => .{},
                2 => .{ .adapter = .json },
                else => .{ .lm = teacher.lm() },
            },
        };
        const result = try optimizer.compile(P, arena, .{
            .arena = arena,
            .lm = if (scenario == 3) lm.LM.unconfigured else base.lm(),
            .lm_options = .{ .temperature = 0.25 },
            .json_fallback = false,
        }, student, &train, .{});
        try std.testing.expectEqual(@as(usize, if (scenario == 3) 1 else 0), teacher_history.requests.items.len);
        const first = if (scenario == 3) teacher_history.requests.items[0] else base_history.requests.items[0];
        const sampling_calls: usize = if (scenario == 3) 1 else 2;
        const second = base_history.requests.items[sampling_calls - 1];
        try std.testing.expectEqual(@as(i64, 0), first.options.rollout_id.?.integer);
        try std.testing.expectEqual(@as(i64, 1), second.options.rollout_id.?.integer);
        try std.testing.expectEqual(@as(?f64, if (scenario < 2) 1 else 0.25), first.options.temperature);
        try std.testing.expectEqual(@as(?f64, 1), second.options.temperature);
        try std.testing.expect(std.mem.indexOf(u8, first.messages[first.messages.len - 1].content.text, "<question>") != null);
        try std.testing.expectEqual(@as(usize, 3), base_history.requests.items.len - sampling_calls);
        for (base_history.requests.items[sampling_calls..]) |evaluation| {
            if (scenario == 2) {
                try std.testing.expectEqual(@as(i64, 0), evaluation.options.rollout_id.?.integer);
            } else {
                try std.testing.expect(evaluation.options.rollout_id == null);
            }
        }
        try std.testing.expectEqual(@as(f64, 1), result.scores[0]);
        if (scenario == 3) try std.testing.expect(result.program.state.lm.?.sameInstance(base.lm()));
    }
}

test "SIMBA captures only the feedback LM and leaves compile routing and trace intact" {
    const lm = lm_mod;
    const Predictor = predict.Predict(signature.Signature(struct {
        pub const Input = struct { question: []const u8 };
        pub const Output = struct { answer: []const u8 };
    }));
    const Program = struct {
        predictor: Predictor = .{},

        pub const Input = Predictor.Input;
        pub const Output = Predictor.Output;

        pub fn forward(self: *const @This(), ctx: Context, input: Input) !Output {
            return self.predictor.forward(ctx, input);
        }

        fn score(_: Allocator, _: *const Example, pred: ?*const optimize.RecordPrediction, _: ?*const predict.Trace) value.ParseError!f64 {
            return if (pred) |p| @floatFromInt(@intFromBool(std.mem.eql(u8, p.get("answer").?.string, "yes"))) else 0;
        }
    };

    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const train = [_]Example{try Example.fromJson(arena, "{\"question\":\"ok?\"}", &.{"question"})};
    const config: SIMBA = .{ .metric = Program.score, .bsize = 1, .num_candidates = 2, .max_steps = 1, .max_demos = 0 };
    const yes = "<answer>yes</answer>";
    const no = "<answer>no</answer>";
    const feedback = "<discussion>Compare the answers.</discussion><module_advice><predictor>Give the better answer.</predictor></module_advice>";
    const Case = enum { explicit, captured, fallback, unconfigured_explicit };

    for ([_]Case{ .explicit, .captured, .fallback, .unconfigured_explicit }) |case| {
        var history: lm.History = .{ .arena = arena };
        var task_history: lm.History = .{ .arena = arena };
        var compile_history: lm.History = .{ .arena = arena };
        var explicit: lm.ScriptedLM = .{ .responses = &.{feedback}, .history = &history };
        var captured: lm.ScriptedLM = .{ .responses = &.{feedback}, .history = &history };
        var task: lm.ScriptedLM = .{ .responses = &.{ yes, no, yes, no, yes }, .history = &task_history };
        var current: lm.ScriptedLM = .{ .responses = &.{feedback}, .history = &compile_history };
        var construction: Context = .{
            .arena = arena,
            .lm = if (case == .fallback) LM.unconfigured else captured.lm(),
            .adapter = .json,
            .lm_options = .{ .temperature = 0.81 },
            .lm_override = task.lm(),
        };
        const optimizer = config.init(construction, switch (case) {
            .explicit => explicit.lm(),
            .unconfigured_explicit => LM.unconfigured,
            else => null,
        });
        construction.lm = current.lm();
        var trace: predict.Trace = .{ .arena = arena };
        const ctx: Context = .{
            .arena = arena,
            .lm = construction.lm,
            .lm_override = task.lm(),
            .lm_override_present = true,
            .adapter = .xml,
            .lm_options = .{ .temperature = 0.37, .rollout_id = .{ .integer = 6 } },
            .trace = &trace,
            .json_fallback = false,
        };

        const result = try optimizer.compile(Program, arena, ctx, .{}, &train, .{});
        try std.testing.expectEqualSlices(f64, &.{ 1, 0 }, result.scores);
        try std.testing.expect(std.mem.endsWith(u8, result.program.predictor.state.instructions, "\n\nGive the better answer."));
        try std.testing.expect(result.program.predictor.state.lm == null);
        try std.testing.expectEqual(@as(usize, 0), trace.steps.items.len);

        try std.testing.expectEqual(@as(usize, @intFromBool(case == .explicit)), explicit.index);
        try std.testing.expectEqual(@as(usize, @intFromBool(case == .captured)), captured.index);
        const fallback = case == .fallback or case == .unconfigured_explicit;
        try std.testing.expectEqual(@as(usize, @intFromBool(fallback)), current.index);
        const feedback_requests = if (fallback) compile_history.requests.items else history.requests.items;
        try std.testing.expectEqual(@as(usize, 1), feedback_requests.len);
        try std.testing.expectEqual(@as(usize, 0), if (fallback) history.requests.items.len else compile_history.requests.items.len);
        const request = feedback_requests[0];
        try std.testing.expect(std.mem.indexOf(u8, request.messages[request.messages.len - 1].content.text, "<module_names>") != null);
        try std.testing.expectEqual(@as(?f64, 0.37), request.options.temperature);
        try std.testing.expectEqual(@as(i64, 6), request.options.rollout_id.?.integer);

        try std.testing.expectEqual(@as(usize, 5), task.index);
        try std.testing.expectEqual(@as(usize, 5), task_history.requests.items.len);
        for (task_history.requests.items, 0..) |call, i| {
            try std.testing.expect(std.mem.indexOf(u8, call.messages[call.messages.len - 1].content.text, "<question>") != null);
            const has_advice = std.mem.indexOf(u8, call.messages[0].content.text, "Give the better answer.") != null;
            try std.testing.expectEqual(i == 2 or i == 4, has_advice);
            try std.testing.expectEqual(@as(?f64, if (i < 2) 1 else 0.37), call.options.temperature);
            try std.testing.expectEqual(@as(i64, if (i == 1) 7 else 6), call.options.rollout_id.?.integer);
        }
    }
}

test "roundHalfEven matches Python round" {
    try std.testing.expectEqual(@as(f64, 0), roundHalfEven(0.5));
    try std.testing.expectEqual(@as(f64, 2), roundHalfEven(1.5));
    try std.testing.expectEqual(@as(f64, 2), roundHalfEven(2.5));
    try std.testing.expectEqual(@as(f64, 1), roundHalfEven(0.6666));
}
