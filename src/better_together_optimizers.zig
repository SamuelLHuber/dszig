//! Borrowed native BetterTogether optimizer adapters. Registry entries are pointers
//! to these adapters, or to any caller-defined adapter with the same compile
//! protocol; no runtime optimizer tag or Python argument dictionary is needed.
//! Datasets, optimizer state, pools, jobs and result storage remain caller-owned.
const std = @import("std");
const Allocator = std.mem.Allocator;
const predict = @import("predict.zig");
const optimize = @import("optimize.zig");
const bootstrap_finetune = @import("bootstrap_finetune.zig");
const bootstrap_trace = @import("bootstrap_trace.zig");
const mipro = @import("mipro.zig");
const copro = @import("copro.zig");
const gepa = @import("gepa.zig");
const infer_rules = @import("infer_rules.zig");
const simba = @import("simba.zig");
const bootstrap_optuna = @import("bootstrap_optuna.zig");

/// Source passes the teacher argument raw: BetterTogether's own top-level value
/// is always a normalized list, while a per-key override may be None, a single
/// program or a list. Adapters must not coerce a list (even a singleton) to P.
pub const TeacherArgument = union(enum) {
    none,
    program: bootstrap_finetune.Teacher.Program,
    list: []const bootstrap_finetune.Teacher,

    /// Source BootstrapFewShot and optimizers delegating to it call deepcopy on
    /// the teacher; a Python list has no such method, even empty or singleton.
    /// These optimizers execute the teacher as P, so an erased program of a
    /// different nominal type is refused rather than reinterpreted.
    fn typed(self: TeacherArgument, comptime P: type) !?*const P {
        return switch (self) {
            .none => null,
            .program => |program| program.get(P) orelse error.OptimizerTeacherTypeUnsupported,
            .list => error.OptimizerTeacherListUnsupported,
        };
    }

    /// Source BootstrapFinetune wraps a non-list teacher (including None) in a
    /// one-element list. Null selects the native actual-student default.
    pub fn normalized(
        self: TeacherArgument,
        single: *[1]bootstrap_finetune.Teacher,
    ) ?[]const bootstrap_finetune.Teacher {
        switch (self) {
            .none => return null,
            .program => |program| {
                single[0] = .{ .program = program };
                return single;
            },
            .list => |list| return list,
        }
    }
};

/// Standard arguments after BetterTogether's per-step overrides. Adapters set
/// `accepts_valset` / `accepts_teacher` to mirror the Source compile signature:
/// an explicit override of a declared-false argument is rejected before launch.
/// Result-owned allocations use out; temporary work uses gpa where supported.
pub fn Request(comptime P: type) type {
    return struct {
        out: Allocator,
        gpa: Allocator,
        io: std.Io,
        ctx: predict.Context,
        student: P,
        /// Actual current program, retained by the caller. In-place optimizers
        /// must use this pointer so failures preserve observable partial edits.
        student_ref: *P,
        trainset: []const optimize.Example,
        valset: ?[]const optimize.Example,
        teacher: TeacherArgument,
        step_index: usize,
    };
}

pub const RandomSearch = struct {
    pub const source_name = "BootstrapFewShotWithRandomSearch";
    pub const accepts_valset = true;
    pub const accepts_teacher = true;
    optimizer: optimize.BootstrapFewShotWithRandomSearch,

    /// Source accepts trainset, valset and teacher. Native does not expose Source
    /// restrict/labeled_sample; this adapter does not invent those options.
    pub fn compile(self: *RandomSearch, comptime P: type, request: Request(P)) !P {
        // Preserve Source's delayed list error: stop_at_score may finish at
        // the reset/labeled candidate before any bootstrap consumes teacher.
        const result = switch (request.teacher) {
            .list => |teachers| try self.optimizer.compile(P, request.out, request.gpa, request.ctx, request.student, teachers, request.trainset, request.valset),
            .none, .program => try self.optimizer.compile(P, request.out, request.gpa, request.ctx, request.student, try teacherValue(P, request.teacher), request.trainset, request.valset),
        };
        return result.program;
    }
};

pub const LabeledFewShot = struct {
    pub const source_name = "LabeledFewShot";
    pub const accepts_valset = false;
    pub const accepts_teacher = false;
    optimizer: optimize.LabeledFewShot,

    /// Source does not accept teacher or valset, so standard values are ignored.
    /// The native optimizer implements Source's default sample=true only.
    pub fn compile(self: *LabeledFewShot, comptime P: type, request: Request(P)) !P {
        return self.optimizer.compile(P, request.out, request.student, request.trainset);
    }
};

pub const BootstrapFewShot = struct {
    pub const source_name = "BootstrapFewShot";
    pub const accepts_valset = false;
    pub const accepts_teacher = true;
    optimizer: optimize.BootstrapFewShot,

    /// Source accepts teacher but not valset.
    pub fn compile(self: *BootstrapFewShot, comptime P: type, request: Request(P)) !P {
        const result = try self.optimizer.compile(P, request.out, request.gpa, request.ctx, request.student, try teacherValue(P, request.teacher), request.trainset);
        return result.program;
    }
};

pub const MIPROv2 = struct {
    pub const source_name = "MIPROv2";
    pub const accepts_valset = true;
    pub const accepts_teacher = true;
    optimizer: mipro.MIPROv2,
    prompt: ?predict.Context = null,
    /// Standard valset comes from Request, not this options object's valset.
    compile_options: mipro.MIPROv2.Options = .{},

    pub fn compile(self: *MIPROv2, comptime P: type, request: Request(P)) !P {
        var options = self.compile_options;
        options.valset = request.valset;
        // The real proposal-demo loop consumes a teacher only when it reaches
        // bootstrap. Preserve both ignored-list and late-failure behavior.
        const result = switch (request.teacher) {
            .list => |teachers| try self.optimizer.compile(P, request.out, request.gpa, request.ctx, self.prompt, request.student, teachers, request.trainset, options),
            .none, .program => try self.optimizer.compile(P, request.out, request.gpa, request.ctx, self.prompt, request.student, try teacherValue(P, request.teacher), request.trainset, options),
        };
        return result.program;
    }
};

pub const COPRO = struct {
    pub const source_name = "COPRO";
    pub const accepts_valset = false;
    pub const accepts_teacher = false;
    optimizer: copro.COPRO,
    prompt: ?predict.Context = null,

    /// Source scores on trainset, ignores standard teacher/valset, and accepts
    /// eval_kwargs. Native evaluation settings live in the typed optimizer.
    pub fn compile(self: *COPRO, comptime P: type, request: Request(P)) !P {
        const result = try self.optimizer.compile(P, request.out, request.gpa, request.ctx, self.prompt, request.student, request.trainset);
        return result.program;
    }
};

pub const GEPA = struct {
    pub const source_name = "GEPA";
    pub const accepts_valset = true;
    pub const accepts_teacher = true;
    optimizer: gepa.GEPA,
    /// Request supplies standard valset; reflection is a native typed option.
    compile_options: gepa.GEPA.Options = .{},

    pub fn compile(self: *GEPA, comptime P: type, request: Request(P)) !P {
        // GEPA's actual compile signature accepts teacher but asserts None.
        // Parent also performs Source's special override prevalidation.
        if (request.teacher != .none) return error.GEPATeacherUnsupported;
        var options = self.compile_options;
        options.valset = request.valset;
        const result = try self.optimizer.compile(P, request.out, request.ctx, request.student, request.trainset, options);
        return result.program;
    }
};

pub const InferRules = struct {
    pub const source_name = "InferRules";
    pub const accepts_valset = true;
    pub const accepts_teacher = true;
    /// Construct once with native InferRules.init. Borrowing preserves the
    /// induction program's signature identity, edits and RNG across compiles.
    optimizer: *infer_rules.InferRules,

    pub fn compile(self: *InferRules, comptime P: type, request: Request(P)) !P {
        // The borrowed teacher itself is passed: Source instruction writes reach
        // every module sharing its signature classes, including the teacher.
        _ = try request.teacher.typed(P);
        const teacher: ?*P = switch (request.teacher) {
            .none => null,
            .program => |program| program.getMutable(P) orelse return error.ImmutableTeacherProgram,
            .list => unreachable,
        };
        const result = try self.optimizer.compile(P, request.out, request.gpa, request.ctx, request.student_ref, teacher, request.trainset, request.valset);
        // Source can return None (no candidates or no score beating -inf).
        // The registry protocol requires P: report that result, never substitute
        // the original program and pretend optimization selected it.
        return result.program orelse error.OptimizerReturnedNull;
    }
};

pub const SIMBA = struct {
    pub const source_name = "SIMBA";
    pub const accepts_valset = false;
    pub const accepts_teacher = false;
    optimizer: simba.SIMBA,
    compile_options: simba.SIMBA.Options = .{},

    /// Source does not accept standard teacher or valset.
    pub fn compile(self: *SIMBA, comptime P: type, request: Request(P)) !P {
        const result = try self.optimizer.compile(P, request.out, request.ctx, request.student, request.trainset, self.compile_options);
        return result.program;
    }
};

pub const BootstrapFewShotWithOptuna = struct {
    pub const source_name = "BootstrapFewShotWithOptuna";
    pub const accepts_valset = true;
    pub const accepts_teacher = true;
    optimizer: bootstrap_optuna.BootstrapFewShotWithOptuna,
    /// Source requires this compile argument (it has no default).
    max_demos: usize,

    pub fn compile(self: *BootstrapFewShotWithOptuna, comptime P: type, request: Request(P)) !P {
        const result = try self.optimizer.compile(P, request.out, request.gpa, request.io, request.ctx, request.student, try teacherValue(P, request.teacher), self.max_demos, request.trainset, request.valset);
        return result.program;
    }
};

/// Default weight adapter. Each attempted native compile consumes a fresh slot
/// set, including on failure. Retain slot models/names while any original,
/// teacher or current program borrows them; copying this adapter does not create
/// fresh slots. BetterTogether's ordinary candidate deepcopy clones its LMs,
/// rather than borrowing these slots. Startup jobs may be shared ACROSS sets:
/// native BFF joins and releases them before returning, then copies trained
/// names into each set's distinct slots.
pub const BootstrapFinetune = struct {
    pub const source_name = "BootstrapFinetune";
    pub const accepts_valset = false;
    pub const accepts_teacher = true;
    /// The actual current pointer preserves Source's in-place teacher identity
    /// validation inside BootstrapFinetune; no copied-program precheck is needed.
    pub const returns_in_place = true;
    optimizer: bootstrap_finetune.BootstrapFinetune,
    trace_options: bootstrap_trace.Options = .{},
    slot_sets: []const []bootstrap_finetune.Slot,
    next_slots: usize = 0,

    pub fn compile(self: *BootstrapFinetune, comptime P: type, request: Request(P)) !P {
        if (self.next_slots >= self.slot_sets.len) return error.FinetuneSlotSetsExhausted;
        try self.validateStorage();
        const slots = self.slot_sets[self.next_slots];
        for (slots) |*slot| {
            if (slot.used) return error.FinetuneSlotAlreadyUsed;
        }
        // Traces, formatting and group discovery expire after native BFF has
        // joined every job. Retained models borrow original LM configuration
        // and the separate slots, not this per-attempt scratch.
        var scratch: std.heap.ArenaAllocator = .init(request.gpa);
        defer scratch.deinit();
        const arena = scratch.allocator();
        // Compute the exact number of LM/predictor training groups without
        // tracing or starting jobs. No heap allocation is used for alias checks.
        const predictors = try predict.namedPredictors(arena, request.student_ref);
        var required: usize = 0;
        for (predictors, 0..) |predictor, index| {
            const model = predictor.param.predict.lm orelse return error.FinetuneMissingLM;
            var duplicate = false;
            if (self.optimizer.multitask) {
                for (predictors[0..index]) |previous| {
                    if (previous.param.predict.lm.?.sameInstance(model)) {
                        duplicate = true;
                        break;
                    }
                }
            }
            if (!duplicate) required += 1;
        }
        if (required > slots.len) return error.FinetuneSlotLimit;
        self.next_slots += 1;
        var single: [1]bootstrap_finetune.Teacher = undefined;
        const teachers = request.teacher.normalized(&single);
        try self.optimizer.compile(P, arena, request.io, request.ctx, request.student_ref, request.trainset, teachers, slots, self.trace_options);
        return request.student_ref.*;
    }

    fn validateStorage(self: *const BootstrapFinetune) !void {
        // Check the WHOLE schedule before any training, so a later step cannot
        // overwrite an earlier candidate through partially overlapping slices.
        for (self.slot_sets, 0..) |slots, set_index| {
            const bytes = std.mem.sliceAsBytes(slots);
            for (self.slot_sets[0..set_index]) |previous| {
                if (overlaps(bytes.ptr, bytes.len, std.mem.sliceAsBytes(previous).ptr, std.mem.sliceAsBytes(previous).len)) {
                    return error.OverlappingFinetuneSlots;
                }
            }
            for (slots, 0..) |*slot, slot_index| {
                for (slots[0..slot_index]) |*previous| {
                    if (slot.job == previous.job) return error.DuplicateFinetuneJob;
                }
            }
            // A name/model slot must not occupy a referenced job or its
            // mutable request scratch, even when that job belongs to a
            // different step. Shared job identity alone is allowed.
            for (self.slot_sets) |other_slots| {
                for (other_slots) |*other| {
                    if (overlaps(bytes.ptr, bytes.len, other.job, @sizeOf(@TypeOf(other.job.*))) or
                        overlaps(bytes.ptr, bytes.len, other.job.buffers.body.ptr, other.job.buffers.body.len) or
                        overlaps(bytes.ptr, bytes.len, other.job.buffers.scratch.ptr, other.job.buffers.scratch.len))
                    {
                        return error.OverlappingFinetuneSlots;
                    }
                }
            }
        }
    }
};

/// A by-value teacher for optimizers that, like Source, work on their own copy.
fn teacherValue(comptime P: type, teacher: TeacherArgument) !?P {
    return if (try teacher.typed(P)) |typed| typed.* else null;
}

/// Subtraction avoids overflowing an end address. Empty slices never overlap.
fn overlaps(left: anytype, left_len: usize, right: anytype, right_len: usize) bool {
    if (left_len == 0 or right_len == 0) return false;
    const a = @intFromPtr(left);
    const b = @intFromPtr(right);
    return if (a <= b) b - a < left_len else a - b < right_len;
}
