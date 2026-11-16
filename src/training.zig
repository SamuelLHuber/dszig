//! OpenAI training jobs. Initialize immovable workers and bounded buffers at
//! startup; inputs and model resources remain borrowed until wait/cancel joins.
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const lm = @import("lm.zig");
const http = @import("http.zig");
const Value = @import("runtime_value.zig").Value;
const pyjson = @import("pyjson.zig");

pub const Format = enum { chat, completion, grpo_chat };
pub const Status = enum {
    not_started,
    pending,
    running,
    succeeded,
    failed,
    cancelled,

    pub fn terminal(self: Status) bool {
        return switch (self) {
            .succeeded, .failed, .cancelled => true,
            else => false,
        };
    }
};

pub const Request = struct {
    data: []const Value,
    format: ?Format,
    train_kwargs: ?Value = null,
};

pub const Limits = struct {
    max_rows: u32,
    max_polls: u32,
    total_timeout_ns: u64,
    request_timeout_ns: u64,
    cleanup_timeout_ns: u64,
    poll_interval_ns: u64,
};

pub const Buffers = struct {
    /// Multipart/JSON request storage, reused only after its request joins.
    body: []u8,
    /// URL/header/response parsing scratch. HTTP client allocations are separate.
    scratch: []u8,
};

const Text = struct {
    bytes: [1024]u8 = undefined,
    len: u16 = 0,

    fn get(self: *const Text) []const u8 {
        assert(self.len <= self.bytes.len);
        return self.bytes[0..self.len];
    }

    fn set(self: *Text, text: []const u8) !void {
        if (text.len == 0 or text.len > self.bytes.len) return error.TrainingIdentifierTooLong;
        @memcpy(self.bytes[0..text.len], text);
        self.len = @intCast(text.len);
    }
};

/// One persistent startup worker. wait/cancelAndWait form a completion barrier;
/// result and snapshots must not be read through pointers while starting again.
/// The supplied client/Io may allocate internally: this is not a global-heap proof.
pub const Job = struct {
    io: Io,
    address: *const Job,
    limits: Limits,
    buffers: Buffers,
    thread: ?std.Thread = null,
    mutex: Io.Mutex = .init,
    changed: Io.Condition = .init,
    live: bool = true,
    busy: bool = false,
    submitted: bool = false,
    cancellation: std.atomic.Value(bool) = .init(false),
    source: lm.OpenAI = undefined,
    request: Request = undefined,
    result_model: lm.OpenAI = undefined,
    trained_model: Text = .{},
    file_id: Text = .{},
    job_id: Text = .{},
    event_id: Text = .{},
    events_seen: u32 = 0,
    observed_status: Status = .not_started,
    failure: ?anyerror = null,

    pub fn init(self: *Job, io: Io, buffers: Buffers, limits: Limits, config: std.Thread.SpawnConfig) !void {
        if (limits.max_rows == 0 or limits.max_polls == 0 or limits.total_timeout_ns == 0 or
            limits.request_timeout_ns == 0 or limits.cleanup_timeout_ns == 0 or
            buffers.body.len == 0 or buffers.scratch.len == 0) return error.InvalidTrainingLimits;
        const body_start = @intFromPtr(buffers.body.ptr);
        const scratch_start = @intFromPtr(buffers.scratch.ptr);
        assert(body_start + buffers.body.len <= scratch_start or scratch_start + buffers.scratch.len <= body_start);
        self.* = .{ .io = io, .address = self, .limits = limits, .buffers = buffers };
        self.thread = try std.Thread.spawn(config, worker, .{self});
        assert(self.thread != null);
    }

    /// Stops and joins the worker even when remote cleanup fails. Cleanup errors
    /// retain remote IDs and must be reported by the owner, not silently dropped.
    pub fn deinit(self: *Job) !void {
        assert(self.address == self);
        self.cancellation.store(true, .release);
        self.mutex.lockUncancelable(self.io);
        while (self.busy) self.changed.waitUncancelable(self.io, &self.mutex);
        self.live = false;
        self.changed.broadcast(self.io);
        self.mutex.unlock(self.io);
        if (self.thread) |thread| thread.join();
        self.thread = null;
        assert(!self.busy);
        if (self.submitted) try self.cleanup();
    }

    /// Does not create a thread or copy the dataset. Retain inputs until wait.
    pub fn start(self: *Job, source: *const lm.OpenAI, request: Request) !void {
        assert(self.address == self);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        assert(self.live);
        if (self.busy) return error.TrainingJobBusy;
        if (self.job_id.len != 0 or self.file_id.len != 0) return error.TrainingResourcesRetained;
        if (request.data.len > self.limits.max_rows) return error.TrainingRowLimit;
        try checkInputs(source, request, self.buffers.body.len);
        self.source = source.*;
        self.request = request;
        self.file_id.len = 0;
        self.job_id.len = 0;
        self.event_id.len = 0;
        self.trained_model.len = 0;
        self.events_seen = 0;
        self.observed_status = .not_started;
        self.failure = null;
        self.submitted = true;
        self.cancellation.store(false, .release);
        self.busy = true;
        self.changed.signal(self.io);
    }

    /// The returned model borrows this job's inline name until start/deinit.
    /// copyWithModel into caller storage with an owned name to retain it longer.
    pub fn wait(self: *Job) !*lm.OpenAI {
        assert(self.address == self);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.submitted) return error.TrainingNotStarted;
        while (self.busy) self.changed.waitUncancelable(self.io, &self.mutex);
        if (self.failure) |err| return err;
        assert(self.trained_model.len > 0);
        return &self.result_model;
    }

    /// Cooperative cancellation includes remote cleanup and the local barrier.
    /// A blocked request is bounded by request_timeout_ns, not merely inactivity.
    pub fn cancelAndWait(self: *Job) !void {
        assert(self.address == self);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.busy) return error.TrainingJobComplete;
        self.cancellation.store(true, .release);
        while (self.busy) self.changed.waitUncancelable(self.io, &self.mutex);
        if (self.failure) |err| {
            if (err != error.TrainingCanceled) return err;
        } else return error.TrainingJobComplete;
    }

    /// Explicitly release completed/failed remote resources before slot reuse.
    /// Failure preserves remaining IDs, so another start cannot abandon them.
    pub fn release(self: *Job) !void {
        assert(self.address == self);
        self.mutex.lockUncancelable(self.io);
        const busy = self.busy;
        self.mutex.unlock(self.io);
        if (busy) return error.TrainingJobBusy;
        if (self.submitted) try self.cleanup();
    }

    /// Fresh remote status using caller-owned scratch, safe during worker polls.
    /// After cancellation clears the remote ID, this returns not_started.
    pub fn status(self: *Job, arena: Allocator) !Status {
        assert(self.address == self);
        self.mutex.lockUncancelable(self.io);
        if (!self.submitted) {
            self.mutex.unlock(self.io);
            return .not_started;
        }
        const id = self.job_id;
        const source = self.source;
        self.mutex.unlock(self.io);
        if (id.len == 0) return .not_started;
        const path = try std.fmt.allocPrint(arena, "/fine_tuning/jobs/{s}", .{id.get()});
        const response = try send(self.io, &source, arena, self.limits, deadline(self.io, self.limits.request_timeout_ns), .GET, path, null, null);
        defer source.client.allocator.free(response.body);
        const parsed = try std.json.parseFromSlice(std.json.Value, arena, response.body, .{});
        defer parsed.deinit();
        return parseStatus(try stringField(parsed.value, "status"));
    }

    fn worker(self: *Job) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        // Persistent lifecycle loop; every submitted operation is bounded.
        while (self.live) {
            if (!self.busy) {
                self.changed.waitUncancelable(self.io, &self.mutex);
                continue;
            }
            self.mutex.unlock(self.io);
            const failure: ?anyerror = if (self.run()) null else |err| err;
            self.mutex.lockUncancelable(self.io);
            self.failure = failure;
            self.busy = false;
            self.changed.broadcast(self.io);
        }
        assert(!self.busy);
    }

    fn check(self: *Job, end: u64) !void {
        if (self.cancellation.load(.acquire)) return error.TrainingCanceled;
        if (now(self.io) >= end) return error.TrainingTimedOut;
    }

    fn run(self: *Job) !void {
        self.runTraining() catch |err| {
            if (err == error.TrainingCanceled or err == error.TrainingTimedOut) try self.cleanup();
            return err;
        };
    }

    fn runTraining(self: *Job) !void {
        const end = deadline(self.io, self.limits.total_timeout_ns);
        const format = self.request.format orelse return error.UnsupportedTrainingFormat;
        if (format == .grpo_chat) return error.UnsupportedTrainingFormat;
        try self.check(end);
        try self.upload(end);
        try self.check(end);
        try self.create(end);
        for (0..self.limits.max_polls) |_| {
            try self.check(end);
            const terminal = try self.poll(end);
            if (terminal) {
                try self.check(end);
                self.source.copyWithModel(&self.result_model, self.trained_model.get());
                return;
            }
            const remaining = end -| now(self.io);
            try Io.sleep(self.io, .fromNanoseconds(@min(remaining, self.limits.poll_interval_ns)), .awake);
        }
        // Poll exhaustion is also a bounded cancellation, never an abandoned job.
        try self.cleanup();
        return error.TrainingPollLimit;
    }

    fn upload(self: *Job, end: u64) !void {
        var writer: Io.Writer = .fixed(self.buffers.body);
        const boundary = "dszig-finetune-jsonl";
        try writer.writeAll("--" ++ boundary ++ "\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nfine-tune\r\n");
        try writer.writeAll("--" ++ boundary ++ "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"training.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n");
        for (self.request.data) |row| {
            try pyjson.writeJsonProjection(&writer, row, .minified, 0);
            try writer.writeByte('\n');
        }
        try writer.writeAll("\r\n--" ++ boundary ++ "--\r\n");
        var scratch = std.heap.FixedBufferAllocator.init(self.buffers.scratch);
        const response = try send(self.io, &self.source, scratch.allocator(), self.limits, end, .POST, "/files", writer.buffered(), "multipart/form-data; boundary=" ++ boundary);
        defer self.source.client.allocator.free(response.body);
        const parsed = try std.json.parseFromSlice(std.json.Value, scratch.allocator(), response.body, .{});
        defer parsed.deinit();
        var id: Text = .{};
        try setIdentifier(&id, try stringField(parsed.value, "id"));
        self.mutex.lockUncancelable(self.io);
        self.file_id = id;
        self.mutex.unlock(self.io);
    }

    fn create(self: *Job, end: u64) !void {
        var writer: Io.Writer = .fixed(self.buffers.body);
        try writer.writeAll("{\"model\":");
        const configured = self.source.finetuning_model orelse "";
        const model = if (configured.len != 0) configured else self.source.model;
        // Source removes every occurrence, not just the leading prefix.
        var scratch = std.heap.FixedBufferAllocator.init(self.buffers.scratch);
        try writeModel(&writer, model);
        try writer.writeAll(",\"training_file\":");
        try pyjson.writeJsonString(&writer, self.file_id.get());
        try writer.writeAll(",\"hyperparameters\":");
        var options = self.request.train_kwargs orelse self.source.train_kwargs;
        if (options == .null or (options == .object and options.object.count() == 0)) options = self.source.train_kwargs;
        if (options != .object) return error.InvalidTrainingOptions;
        try pyjson.writeJsonProjection(&writer, options, .minified, 0);
        try writer.writeByte('}');
        const response = try send(self.io, &self.source, scratch.allocator(), self.limits, end, .POST, "/fine_tuning/jobs", writer.buffered(), "application/json");
        defer self.source.client.allocator.free(response.body);
        const parsed = try std.json.parseFromSlice(std.json.Value, scratch.allocator(), response.body, .{});
        defer parsed.deinit();
        var id: Text = .{};
        try setIdentifier(&id, try stringField(parsed.value, "id"));
        self.mutex.lockUncancelable(self.io);
        self.job_id = id;
        self.observed_status = .pending;
        self.mutex.unlock(self.io);
    }

    fn poll(self: *Job, end: u64) !bool {
        try self.events(end);
        var scratch = std.heap.FixedBufferAllocator.init(self.buffers.scratch);
        const arena = scratch.allocator();
        const path = try std.fmt.allocPrint(arena, "/fine_tuning/jobs/{s}", .{self.job_id.get()});
        const response = try send(self.io, &self.source, arena, self.limits, end, .GET, path, null, null);
        defer self.source.client.allocator.free(response.body);
        const parsed = try std.json.parseFromSlice(std.json.Value, arena, response.body, .{});
        defer parsed.deinit();
        const state = try parseStatus(try stringField(parsed.value, "status"));
        self.mutex.lockUncancelable(self.io);
        self.observed_status = state;
        self.mutex.unlock(self.io);
        switch (state) {
            .succeeded => {
                try self.trained_model.set(try stringField(parsed.value, "fine_tuned_model"));
                return true;
            },
            .failed => return error.TrainingFailed,
            .cancelled => return error.TrainingCanceled,
            else => return false,
        }
    }

    fn events(self: *Job, end: u64) !void {
        var scratch = std.heap.FixedBufferAllocator.init(self.buffers.scratch);
        const arena = scratch.allocator();
        const path = try std.fmt.allocPrint(arena, "/fine_tuning/jobs/{s}/events?limit=1", .{self.job_id.get()});
        const response = try send(self.io, &self.source, arena, self.limits, end, .GET, path, null, null);
        defer self.source.client.allocator.free(response.body);
        const parsed = try std.json.parseFromSlice(std.json.Value, arena, response.body, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidTrainingResponse;
        const data = parsed.value.object.get("data") orelse return error.InvalidTrainingResponse;
        if (data != .array or data.array.items.len > 1) return error.InvalidTrainingResponse;
        if (data.array.items.len == 0) return;
        const id = try stringField(data.array.items[0], "id");
        if (!std.mem.eql(u8, self.event_id.get(), id)) {
            try self.event_id.set(id);
            self.events_seen += 1;
            assert(self.events_seen <= self.limits.max_polls);
        }
    }

    fn cancelRemote(self: *Job, arena: Allocator, end: u64) !void {
        const path = try std.fmt.allocPrint(arena, "/fine_tuning/jobs/{s}", .{self.job_id.get()});
        const response = try send(self.io, &self.source, arena, self.limits, end, .GET, path, null, null);
        defer self.source.client.allocator.free(response.body);
        const parsed = try std.json.parseFromSlice(std.json.Value, arena, response.body, .{});
        defer parsed.deinit();
        const state = try parseStatus(try stringField(parsed.value, "status"));
        if (!state.terminal()) {
            const cancel_path = try std.fmt.allocPrint(arena, "{s}/cancel", .{path});
            const canceled = try send(self.io, &self.source, arena, self.limits, end, .POST, cancel_path, "{}", "application/json");
            self.source.client.allocator.free(canceled.body);
        }
        self.mutex.lockUncancelable(self.io);
        self.job_id.len = 0;
        self.mutex.unlock(self.io);
    }

    fn cleanup(self: *Job) !void {
        const end = deadline(self.io, self.limits.cleanup_timeout_ns);
        var scratch = std.heap.FixedBufferAllocator.init(self.buffers.scratch);
        const arena = scratch.allocator();
        if (self.job_id.len != 0) try self.cancelRemote(arena, end);
        if (self.file_id.len != 0) {
            const path = try std.fmt.allocPrint(arena, "/files/{s}", .{self.file_id.get()});
            const response = try send(self.io, &self.source, arena, self.limits, end, .DELETE, path, null, null);
            self.source.client.allocator.free(response.body);
            self.mutex.lockUncancelable(self.io);
            self.file_id.len = 0;
            self.mutex.unlock(self.io);
        }
    }
};

/// Bound serialization work as well as output size. A huge string containing
/// only removed provider markers must not evade the request buffer's work cap.
fn checkInputs(source: *const lm.OpenAI, request: Request, capacity: usize) !void {
    const configured = source.finetuning_model orelse "";
    const model = if (configured.len != 0) configured else source.model;
    if (model.len > 1024) return error.TrainingInputLimit;
    var remaining = capacity;
    try checkJsonBudget(request.data, &remaining);
    var options = request.train_kwargs orelse source.train_kwargs;
    if (options == .null or (options == .object and options.object.count() == 0)) options = source.train_kwargs;
    try checkJsonBudget((&options)[0..1], &remaining);
}

/// Iterative JSON traversal: at most capacity input bytes/nodes, with a fixed
/// depth stack. No user class serializer runs inside a training worker.
fn checkJsonBudget(values: []const Value, remaining: *usize) !void {
    const Frame = struct { values: []const Value, keys: ?[]const []const u8 = null, index: usize = 0 };
    var stack: [pyjson.max_depth]Frame = undefined;
    var depth: usize = 1;
    stack[0] = .{ .values = values };
    while (depth != 0) {
        const frame = &stack[depth - 1];
        if (frame.index == frame.values.len) {
            depth -= 1;
            continue;
        }
        if (remaining.* == 0) return error.TrainingInputLimit;
        remaining.* -= 1;
        if (frame.keys) |keys| try checkTextBudget(keys[frame.index], remaining);
        const item = frame.values[frame.index];
        frame.index += 1;
        const children: ?Frame = switch (item) {
            .string => blk: {
                try checkTextBudget(item.string, remaining);
                break :blk null;
            },
            .number_string => blk: {
                try checkTextBudget(item.number_string, remaining);
                break :blk null;
            },
            .array => .{ .values = item.array.items },
            .object => .{ .values = item.object.values(), .keys = item.object.keys() },
            .custom => return error.UnsupportedTrainingValue,
            .null, .bool, .integer, .float => null,
        };
        if (children) |child| {
            if (child.values.len == 0) continue;
            if (depth == stack.len) return error.TrainingInputLimit;
            stack[depth] = child;
            depth += 1;
        }
    }
    assert(depth == 0);
}

fn checkTextBudget(text: []const u8, remaining: *usize) !void {
    if (text.len > remaining.*) return error.TrainingInputLimit;
    remaining.* -= text.len;
    if (!std.unicode.utf8ValidateSlice(text)) return error.UnsupportedTrainingValue;
}

/// Remove Source's provider markers while streaming, without a temporary copy.
fn writeModel(writer: *Io.Writer, model: []const u8) !void {
    try writer.writeByte('"');
    var escaped: pyjson.StringContentsWriter = .{ .destination = writer };
    var start: usize = 0;
    var index: usize = 0;
    while (index < model.len) {
        if (std.mem.startsWith(u8, model[index..], "openai/")) {
            try escaped.writer.writeAll(model[start..index]);
            index += "openai/".len;
            start = index;
        } else index += 1;
    }
    assert(start <= model.len);
    try escaped.writer.writeAll(model[start..]);
    try writer.writeByte('"');
}

fn stringField(value: std.json.Value, key: []const u8) ![]const u8 {
    if (value != .object) return error.InvalidTrainingResponse;
    const field = value.object.get(key) orelse return error.InvalidTrainingResponse;
    if (field != .string) return error.InvalidTrainingResponse;
    return field.string;
}

fn setIdentifier(target: *Text, input: []const u8) !void {
    for (input) |byte| {
        if (!std.ascii.isAlphanumeric(byte) and byte != '_' and byte != '-') return error.InvalidTrainingResponse;
    }
    try target.set(input);
}

test "training rejects unsupported formats and recovers the joined worker slot" {
    var body: [1024]u8 = undefined;
    var scratch: [4096]u8 = undefined;
    var job: Job = undefined;
    try job.init(std.testing.io, .{ .body = &body, .scratch = &scratch }, .{
        .max_rows = 1,
        .max_polls = 1,
        .total_timeout_ns = 1000000000,
        .request_timeout_ns = 100000000,
        .cleanup_timeout_ns = 100000000,
        .poll_interval_ns = 0,
    }, .{});
    defer job.deinit() catch unreachable;
    try std.testing.expectError(error.TrainingNotStarted, job.wait());
    try std.testing.expectEqual(Status.not_started, try job.status(std.testing.allocator));
    var source: lm.OpenAI = .{ .client = undefined, .base_url = "", .model = "unused" };
    try std.testing.expectError(error.UnsupportedFinetuning, lm.LM.unconfigured.finetune(&job, .{ .data = &.{}, .format = .chat }));
    try source.lm().finetune(&job, .{ .data = &.{}, .format = .grpo_chat });
    try std.testing.expectError(error.UnsupportedTrainingFormat, job.wait());
    try source.lm().finetune(&job, .{ .data = &.{}, .format = null });
    try std.testing.expectError(error.UnsupportedTrainingFormat, job.wait());
    try std.testing.expectError(error.TrainingRowLimit, job.start(&source, .{ .data = &.{ .null, .null }, .format = .chat }));
    var oversized: [2048]u8 = undefined;
    @memset(&oversized, 'x');
    try std.testing.expectError(error.TrainingInputLimit, source.lm().finetune(&job, .{
        .data = &.{.{ .string = &oversized }},
        .format = .completion,
    }));
    try source.lm().finetune(&job, .{ .data = &.{}, .format = .grpo_chat });
    try std.testing.expectError(error.UnsupportedTrainingFormat, job.wait());
}

test "training status rejects unknown states and recognizes terminal boundaries" {
    const cases = .{
        .{ "validating_files", Status.pending }, .{ "queued", Status.pending },
        .{ "running", Status.running },          .{ "succeeded", Status.succeeded },
        .{ "failed", Status.failed },            .{ "cancelled", Status.cancelled },
    };
    inline for (cases) |case| try std.testing.expectEqual(case[1], try parseStatus(case[0]));
    try std.testing.expectError(error.InvalidTrainingResponse, parseStatus("unknown"));
    try std.testing.expect(!Status.not_started.terminal());
    try std.testing.expect(!Status.running.terminal());
    try std.testing.expect(Status.failed.terminal());
    try std.testing.expect(Status.cancelled.terminal());
    try std.testing.expect(Status.succeeded.terminal());
}

fn parseStatus(input: []const u8) !Status {
    if (std.mem.eql(u8, input, "validating_files") or std.mem.eql(u8, input, "queued")) return .pending;
    if (std.mem.eql(u8, input, "running")) return .running;
    if (std.mem.eql(u8, input, "succeeded")) return .succeeded;
    if (std.mem.eql(u8, input, "failed")) return .failed;
    if (std.mem.eql(u8, input, "cancelled")) return .cancelled;
    return error.InvalidTrainingResponse;
}

fn now(io: Io) u64 {
    const ns = Io.Clock.awake.now(io).nanoseconds;
    assert(ns >= 0);
    assert(ns <= std.math.maxInt(u64));
    return @intCast(ns);
}

fn deadline(io: Io, duration_ns: u64) u64 {
    return now(io) +| duration_ns;
}

/// Whole-request timer complements transport inactivity timers. Both competing
/// I/O tasks are canceled AND joined before request/scratch storage is reused.
fn send(io: Io, source: *const lm.OpenAI, arena: Allocator, limits: Limits, end: u64, method: std.http.Method, path: []const u8, body: ?[]const u8, content_type: ?[]const u8) !http.Response {
    const remaining = end -| now(io);
    if (remaining == 0) return error.TrainingTimedOut;
    const url = try std.fmt.allocPrint(arena, "{s}{s}", .{ std.mem.trimEnd(u8, source.base_url, "/"), path });
    const authorization = if (source.api_key) |key| try std.fmt.allocPrint(arena, "Bearer {s}", .{key}) else null;
    const transport: http.Transport = .{
        .client = source.client,
        .max_response_bytes = source.max_response_bytes,
        // Creating remote work is not retried: an ambiguous response may hide
        // a committed upload/job, and an automatic retry could duplicate it.
        .max_retries = 0,
        .timeout_ns = source.timeout_ns,
        .connect_timeout_ns = source.connect_timeout_ns,
        .header_provider = source.header_provider,
    };
    const input: http.Request = .{ .url = url, .method = method, .body = body, .content_type = content_type, .authorization = authorization };
    const Result = union(enum) { response: anyerror!http.Response, timer: Io.Cancelable!void };
    var results: [2]Result = undefined;
    var select = Io.Select(Result).init(io, &results);
    defer while (select.cancel()) |result| {
        switch (result) {
            .response => |value| if (value) |response| {
                source.client.allocator.free(response.body);
            } else |_| {},
            .timer => {},
        }
    };
    try select.concurrent(.response, http.Transport.send, .{ &transport, arena, input });
    const timeout: Io.Timeout = .{ .duration = .{ .raw = .fromNanoseconds(@min(remaining, limits.request_timeout_ns)), .clock = .awake } };
    try select.concurrent(.timer, Io.Timeout.sleep, .{ timeout, io });
    const completed = try select.await();
    switch (completed) {
        .timer => |result| {
            try result;
            return error.TrainingTimedOut;
        },
        .response => |result| {
            const response = try result;
            const code = @backingInt(response.status);
            if (code < 200 or code >= 300) {
                source.client.allocator.free(response.body);
                return error.TrainingHttpFailure;
            }
            return response;
        },
    }
}
