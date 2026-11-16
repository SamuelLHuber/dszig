//! Source BootstrapFinetune orchestration over real startup-created training jobs.
//! The student is mutated only after all jobs and cleanup have joined successfully.
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const predict = @import("predict.zig");
const trace = @import("bootstrap_trace.zig");
const data = @import("bootstrap_finetune_data.zig");
const training = @import("training.zig");
const lm = @import("lm.zig");
const adapter = @import("adapter.zig");
const optimize = @import("optimize.zig");
const Value = @import("runtime_value.zig").Value;

/// A borrowed, typed native program. Teachers need matching predictor paths, not
/// matching Zig types or signatures. No program copy or callback training facade.
pub const Teacher = union(enum) {
    /// A None element in Source's teacher list means the current student.
    student,
    program: struct { ptr: *const anyopaque, vtable: *const VTable },

    const VTable = struct {
        predictors: *const fn (*const anyopaque, Allocator) Allocator.Error![]predict.NamedParam,
        collect: *const fn (
            *const anyopaque,
            Allocator,
            predict.Context,
            []const optimize.Example,
            trace.Options,
        ) anyerror![]trace.TraceData,
    };

    pub fn init(comptime P: type, program: *const P) Teacher {
        const Implementation = struct {
            fn predictors(
                ptr: *const anyopaque,
                arena: Allocator,
            ) Allocator.Error![]predict.NamedParam {
                const typed: *const P = @ptrCast(@alignCast(ptr));
                return predict.namedPredictors(arena, @constCast(typed));
            }

            fn collect(
                ptr: *const anyopaque,
                arena: Allocator,
                ctx: predict.Context,
                dataset: []const optimize.Example,
                options: trace.Options,
            ) anyerror![]trace.TraceData {
                const typed: *const P = @ptrCast(@alignCast(ptr));
                return trace.bootstrapTraceData(P, arena, ctx, typed, dataset, options);
            }

            const vtable: VTable = .{ .predictors = predictors, .collect = collect };
        };
        return .{ .program = .{ .ptr = program, .vtable = &Implementation.vtable } };
    }
};

pub const AdapterEntry = struct { model: lm.LM, adapter: ?adapter.Adapter };
pub const KwargsEntry = struct { model: lm.LM, kwargs: ?Value };

/// A global value or an explicit identity map. Unlike a global null, a missing
/// key in an explicit map is an error, just as with Source's ordinary dict.
pub const Adapters = union(enum) {
    global: ?adapter.Adapter,
    per_lm: []const AdapterEntry,

    fn get(self: Adapters, model: lm.LM, fallback: adapter.Adapter) !adapter.Adapter {
        switch (self) {
            .global => |selected| return selected orelse fallback,
            .per_lm => |entries| {
                for (entries) |entry| {
                    if (entry.model.sameInstance(model)) return entry.adapter orelse fallback;
                }
                // Source treats {} as the adapter itself, not an empty LM map.
                if (entries.len == 0) return error.CannotInferFinetuneFormat;
                return error.MissingFinetuneAdapter;
            },
        }
    }
};

pub const TrainKwargs = union(enum) {
    global: ?Value,
    per_lm: []const KwargsEntry,

    fn get(self: TrainKwargs, model: lm.LM) !?Value {
        switch (self) {
            .global => |kwargs| return kwargs,
            .per_lm => |entries| {
                for (entries) |entry| {
                    if (entry.model.sameInstance(model)) return entry.kwargs;
                }
                if (entries.len == 0) return null;
                return error.MissingFinetuneKwargs;
            },
        }
    }
};

pub const Limits = struct {
    max_teachers: u32,
    max_examples: u32,
    max_predictors: u32,
    data: data.Limits,
};

/// Initialize the pointed-to Job at startup. Slots and names must remain unmoved
/// and alive while compiled programs borrow their model handles. Reusing a slot
/// explicitly invalidates its previous compiled model, not the HTTP resources.
pub const Slot = struct {
    job: *training.Job,
    model: lm.OpenAI = undefined,
    name: [1024]u8 = undefined,

    fn retain(self: *Slot, trained: *const lm.OpenAI) !void {
        if (trained.model.len > self.name.len) return error.TrainingIdentifierTooLong;
        @memcpy(self.name[0..trained.model.len], trained.model);
        trained.copyWithModel(&self.model, self.name[0..trained.model.len]);
        assert(self.model.model.ptr == self.name[0..].ptr);
    }
};

const Group = struct {
    source: lm.LM,
    pred_ind: ?usize,
    request: training.Request,
    trained: *const lm.OpenAI = undefined,
};

pub const BootstrapFinetune = struct {
    metric: ?optimize.Metric = null,
    multitask: bool = true,
    train_kwargs: TrainKwargs = .{ .global = null },
    adapters: Adapters = .{ .global = null },
    exclude_demos: bool = false,
    /// Zero selects the Source settings default (eight).
    num_threads: u32 = 8,
    limits: Limits,

    /// `out` retains traces, formatted data and discovery scratch for this compile;
    /// provide bounded startup backing, distinct from attempt scratch. The program,
    /// datasets, LM configs and both arenas live until this call joins every job.
    /// Null teachers means the actual student; an explicit student alias refuses.
    /// Empty teacher lists are valid and produce empty training datasets.
    /// For concurrent traces pass a startup pool in trace_options.rollouts with
    /// the desired worker count; without one collection is serial.
    pub fn compile(
        self: *const BootstrapFinetune,
        comptime P: type,
        out: Allocator,
        io: std.Io,
        ctx: predict.Context,
        student: *P,
        dataset: []const optimize.Example,
        teachers: ?[]const Teacher,
        slots: []Slot,
        trace_options: trace.Options,
    ) !void {
        if (dataset.len > self.limits.max_examples) return error.FinetuneExampleLimit;
        const initial = try predict.namedPredictors(out, student);
        if (initial.len > self.limits.max_predictors) return error.FinetunePredictorLimit;
        try self.validateTeachers(out, initial, teachers);
        const num_threads = if (self.num_threads == 0) 8 else self.num_threads;
        var options = trace_options;
        options.metric = self.metric;
        if (options.rollouts) |rollouts| {
            if (rollouts.runner.workers > num_threads) return error.FinetuneTraceThreadLimit;
        }
        const rows = try self.collect(P, out, ctx, student, dataset, teachers, options);
        // Source validates bound models only AFTER collecting every teacher.
        const predictors = try predict.namedPredictors(out, student);
        if (predictors.len > self.limits.max_predictors) return error.FinetunePredictorLimit;
        const groups = try self.prepare(out, io, ctx.adapter, predictors, rows);
        if (groups.len > num_threads) return error.FinetuneThreadLimit;
        if (groups.len > slots.len) return error.FinetuneSlotLimit;
        const used = slots[0..groups.len];
        try validateSlots(used);
        try runTraining(groups, used);
        // All groups succeeded before a single student binding is changed.
        for (predictors, 0..) |predictor, pred_ind| {
            const state = predictor.param.predict;
            const index = groupIndex(groups, state.lm.?, if (self.multitask) null else pred_ind).?;
            state.lm = used[index].model.lm();
            if (self.exclude_demos) state.demos = &.{};
        }
        student.compiled = true;
    }

    fn validateTeachers(
        self: *const BootstrapFinetune,
        arena: Allocator,
        student: []const predict.NamedParam,
        teachers: ?[]const Teacher,
    ) !void {
        const count = if (teachers) |items| items.len else 1;
        if (count > self.limits.max_teachers) return error.FinetuneTeacherLimit;
        if (teachers) |items| {
            // Validate the entire list before executing even its first teacher.
            for (items) |teacher| {
                const program = switch (teacher) {
                    .student => continue,
                    .program => |program| program,
                };
                const predictors = try program.vtable.predictors(program.ptr, arena);
                if (predictors.len != student.len) return error.ProgramMismatch;
                for (student, predictors) |left, right| {
                    if (!std.mem.eql(u8, left.name, right.name)) return error.ProgramMismatch;
                }
                for (student) |left| {
                    for (predictors) |right| {
                        if (left.param.predict == right.param.predict) {
                            return error.SharedFinetunePredictor;
                        }
                    }
                }
            }
        }
    }

    fn collect(
        self: *const BootstrapFinetune,
        comptime P: type,
        arena: Allocator,
        ctx: predict.Context,
        student: *const P,
        dataset: []const optimize.Example,
        teachers: ?[]const Teacher,
        options: trace.Options,
    ) ![]trace.TraceData {
        _ = self;
        if (teachers) |items| {
            var rows: std.ArrayList(trace.TraceData) = .empty;
            for (items) |teacher| {
                const collected = switch (teacher) {
                    .student => try trace.bootstrapTraceData(
                        P,
                        arena,
                        ctx,
                        student,
                        dataset,
                        options,
                    ),
                    .program => |program| try program.vtable.collect(
                        program.ptr,
                        arena,
                        ctx,
                        dataset,
                        options,
                    ),
                };
                try rows.appendSlice(arena, collected);
            }
            return rows.items;
        }
        return trace.bootstrapTraceData(P, arena, ctx, student, dataset, options);
    }

    fn prepare(
        self: *const BootstrapFinetune,
        arena: Allocator,
        io: std.Io,
        fallback: adapter.Adapter,
        predictors: []const predict.NamedParam,
        rows: []const trace.TraceData,
    ) ![]Group {
        var groups: std.ArrayList(Group) = .empty;
        try groups.ensureTotalCapacity(arena, predictors.len);
        for (predictors, 0..) |predictor, pred_ind| {
            const model = predictor.param.predict.lm orelse return error.FinetuneMissingLM;
            const position: ?usize = if (self.multitask) null else pred_ind;
            if (groupIndex(groups.items, model, position) != null) continue;
            const selected_adapter = try self.adapters.get(model, fallback);
            const prepared = try data.prepare(
                arena,
                io,
                rows,
                position,
                selected_adapter,
                self.metric != null,
                self.exclude_demos,
                self.limits.data,
            );
            groups.appendAssumeCapacity(.{
                .source = model,
                .pred_ind = position,
                .request = .{
                    .data = prepared.data,
                    .format = prepared.format,
                    .train_kwargs = try self.train_kwargs.get(model),
                },
            });
        }
        assert(groups.items.len <= predictors.len);
        return groups.items;
    }
};

fn groupIndex(groups: []const Group, model: lm.LM, pred_ind: ?usize) ?usize {
    for (groups, 0..) |group, index| {
        if (group.pred_ind == pred_ind and group.source.sameInstance(model)) return index;
    }
    return null;
}

fn validateSlots(slots: []const Slot) !void {
    for (slots, 0..) |slot, index| {
        for (slots[0..index]) |previous| {
            if (slot.job == previous.job) return error.DuplicateFinetuneJob;
        }
    }
}

/// All starts precede every wait. On any failure every started job is canceled
/// and joined, unlike Source's first-failure escape. All cleanup is attempted;
/// its first failure takes precedence and retained IDs prevent unsafe reuse.
fn runTraining(groups: []Group, slots: []Slot) !void {
    assert(groups.len == slots.len);
    var started: usize = 0;
    var failure: ?anyerror = null;
    for (groups, slots) |group, slot| {
        group.source.finetune(slot.job, group.request) catch |err| {
            failure = err;
            break;
        };
        started += 1;
    }
    if (failure == null) {
        for (groups, slots) |*group, slot| {
            group.trained = slot.job.wait() catch |err| {
                failure = err;
                break;
            };
        }
    }
    var cleanup_failure: ?anyerror = null;
    if (failure != null) {
        for (slots[0..started]) |slot| {
            slot.job.cancelAndWait() catch |err| {
                if (err != error.TrainingJobComplete and cleanup_failure == null) {
                    cleanup_failure = err;
                }
            };
        }
    }
    for (slots[0..started]) |slot| {
        slot.job.release() catch |err| {
            if (cleanup_failure == null) cleanup_failure = err;
        };
    }
    if (cleanup_failure) |err| return err;
    if (failure) |err| return err;
    assert(started == groups.len);
    // A reused slot may hold an input LM's borrowed model name. Do not write
    // any result storage until ALL work and cleanup have stopped reading it.
    for (groups, slots) |group, *slot| try slot.retain(group.trained);
}
