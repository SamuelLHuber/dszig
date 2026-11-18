//! Public history for canonical LM calls. Owners store shared entry handles,
//! not reconstructed requests, responses or output graphs.
const std = @import("std");
const sdk = @import("sdk_response.zig");
const json = @import("sdk_json.zig");
const result_mod = @import("call_result.zig");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub const global_capacity = 10_000;
pub const Error = sdk.Error || error{ InvalidHistoryCapacity, HistoryEntropyUnavailable };

/// Initialize with caller-owned slots. Synchronize access to owners and any
/// mutable SDK payloads externally. Slots need not be initialized or zeroed.
pub const Owner = struct {
    slots: []*Entry,
    head: usize = 0,
    count: usize = 0,

    pub fn init(slots: []*Entry) Owner {
        return .{ .slots = slots };
    }

    pub fn clear(self: *Owner) void {
        while (self.count != 0) self.pop();
        self.head = 0;
    }

    pub fn len(self: *const Owner) usize {
        assert(self.count <= self.slots.len);
        return self.count;
    }

    /// Borrow until eviction/clear, or retain for an independent lease.
    pub fn at(self: *const Owner, index: usize) ?*Entry {
        if (index >= self.len()) return null;
        return self.slots[(self.head + index) % self.slots.len];
    }

    fn pop(self: *Owner) void {
        assert(self.count > 0);
        assert(self.head < self.slots.len);
        self.slots[self.head].release();
        self.head = (self.head + 1) % self.slots.len;
        self.count -= 1;
    }

    fn append(self: *Owner, entry: *Entry, limit: usize) void {
        assert(limit > 0);
        assert(limit <= self.slots.len);
        // Pop exactly one even if the configured limit was lowered.
        if (self.count >= limit) self.pop();
        self.slots[(self.head + self.count) % self.slots.len] = entry.retain();
        self.count += 1;
        assert(self.count <= self.slots.len);
    }
};

pub const Capture = struct {
    allocator: Allocator,
    global: *Owner,
    client: *Owner,
    modules: []const *Owner = &.{},
    owner_limit: usize,

    pub fn validate(self: Capture) error{InvalidHistoryCapacity}!void {
        if (self.global.slots.len != global_capacity) return error.InvalidHistoryCapacity;
        if (self.owner_limit == 0) return;
        if (self.owner_limit > self.client.slots.len) return error.InvalidHistoryCapacity;
        for (self.modules) |owner| {
            if (self.owner_limit > owner.slots.len) return error.InvalidHistoryCapacity;
        }
    }

    /// Publish a validated capture after generation and usage accounting.
    /// Fallible work happens before inserting into any owner.
    /// If present, prompt must be text owned by request.
    pub fn publish(
        self: Capture,
        io: std.Io,
        request: *const sdk.Request,
        prompt: ?[]const u8,
        result: *const result_mod.CallResult,
        model_type: result_mod.ModelType,
    ) Error!void {
        assert(self.global.slots.len == global_capacity);
        assert(result.responses.len > 0);
        const messages = try request.historyMessages(self.allocator);
        errdefer messages.free();
        const timestamp = std.Io.Clock.real.now(io);
        const uuid = try newUuid(io);
        const entry = try self.allocator.create(Entry);
        errdefer self.allocator.destroy(entry);
        const many = if (result.responses.len == 1) null else try self.allocator.dupe(*const sdk.Response, result.responses);
        entry.* = .{
            .allocator = self.allocator,
            .request = request.retain(),
            .prompt = prompt,
            .responses = many orelse (&entry.single_response)[0..1],
            .single_response = result.responses[0],
            .messages = messages,
            .outputs = result.outputs.retain(),
            .usage = result.usage.retain(),
            .cost = result.cost.retain(),
            .cost_details = result.cost_details.retain(),
            .model_type = model_type,
            .timestamp = timestamp,
            .uuid = uuid,
        };
        for (entry.responses) |response| _ = response.retain();
        defer entry.release();
        self.global.append(entry, global_capacity);
        if (self.owner_limit == 0) return;
        self.client.append(entry, self.owner_limit);
        for (self.modules) |owner| owner.append(entry, self.owner_limit);
    }
};

pub const Entry = struct {
    allocator: Allocator,
    references: std.atomic.Value(u32) = .init(1),
    request: *const sdk.Request,
    /// Borrowed from the retained request for string prompt calls.
    prompt: ?[]const u8,
    responses: []const *const sdk.Response,
    single_response: *const sdk.Response,
    messages: sdk.PayloadValue,
    outputs: sdk.PayloadValue,
    usage: sdk.PayloadValue,
    cost: sdk.PayloadValue,
    cost_details: sdk.PayloadValue,
    model_type: result_mod.ModelType,
    timestamp: std.Io.Timestamp,
    uuid: [36]u8,
    annotations: std.StringArrayHashMapUnmanaged(sdk.PayloadValue) = .empty,

    pub const max_annotations = 64;
    pub const max_annotation_key_bytes = 256;
    pub const AnnotationError = Allocator.Error || error{ AnnotationLimit, AnnotationKeyTooLong };

    /// Borrow until this key is replaced/removed or the entry is released.
    /// Retain the value before any of those operations to keep using it.
    pub fn annotation(self: *const Entry, key: []const u8) ?sdk.PayloadValue {
        if (key.len > max_annotation_key_bytes) return null;
        return self.annotations.get(key);
    }

    /// All owners see writes on the same entry. Retain owned graphs as-is;
    /// clone borrowed graphs so annotation lifetime does not depend on input.
    pub fn setAnnotation(self: *Entry, key: []const u8, value: sdk.PayloadValue) AnnotationError!void {
        if (key.len > max_annotation_key_bytes) return error.AnnotationKeyTooLong;
        assert(self.annotations.count() <= max_annotations);
        if (self.annotations.getPtr(key)) |current| {
            const replacement = try self.ownAnnotation(value);
            current.free();
            current.* = replacement;
            return;
        }
        if (self.annotations.count() == max_annotations) return error.AnnotationLimit;
        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);
        const owned_value = try self.ownAnnotation(value);
        errdefer owned_value.free();
        try self.annotations.ensureUnusedCapacity(self.allocator, 1);
        self.annotations.putAssumeCapacityNoClobber(owned_key, owned_value);
    }

    pub fn removeAnnotation(self: *Entry, key: []const u8) bool {
        if (key.len > max_annotation_key_bytes) return false;
        const removed = self.annotations.fetchSwapRemove(key) orelse return false;
        self.allocator.free(removed.key);
        removed.value.free();
        return true;
    }

    fn ownAnnotation(self: *const Entry, value: sdk.PayloadValue) Allocator.Error!sdk.PayloadValue {
        if (value.storage != null) return value.retain();
        var clone: json.CloneContext = .{ .allocator = self.allocator };
        defer clone.deinit();
        return clone.clone(value);
    }

    pub fn retain(self: *Entry) *Entry {
        const previous = self.references.fetchAdd(1, .monotonic);
        assert(previous > 0);
        assert(previous < std.math.maxInt(u32));
        return self;
    }

    pub fn release(self: *Entry) void {
        const previous = self.references.fetchSub(1, .acq_rel);
        assert(previous > 0);
        if (previous != 1) return;
        for (self.annotations.keys(), self.annotations.values()) |key, value| {
            self.allocator.free(key);
            value.free();
        }
        self.annotations.deinit(self.allocator);
        self.messages.free();
        self.outputs.free();
        self.usage.free();
        self.cost.free();
        self.cost_details.free();
        self.request.release();
        for (self.responses) |response| response.release();
        if (self.responses.len > 1) self.allocator.free(self.responses);
        self.allocator.destroy(self);
    }

    /// When formatted for display, use the caller's real-clock instant in UTC.
    /// The string has no zone suffix, and omits a zero fractional part.
    pub fn timestampIso(self: *const Entry, buffer: *[26]u8) error{InvalidHistoryTimestamp}![]const u8 {
        const epoch = std.time.epoch;
        const ns = self.timestamp.nanoseconds;
        const seconds = @divFloor(ns, std.time.ns_per_s);
        if (seconds < epoch.clr or seconds >= 253402300800) return error.InvalidHistoryTimestamp;
        // Gregorian calendars repeat every 400 years. Normalize to one cycle
        // supported by the standard POSIX-epoch conversion.
        const cycle_seconds: i96 = 146097 * @as(i96, epoch.secs_per_day);
        const calendar: epoch.EpochSeconds = .{ .secs = @intCast(@mod(seconds, cycle_seconds)) };
        const year_day = calendar.getEpochDay().calculateYearDay();
        const month_day = year_day.calculateMonthDay();
        const day = calendar.getDaySeconds();
        const year: u16 = @intCast(@as(i96, year_day.year) + 400 * @divFloor(seconds, cycle_seconds));
        const micros: u32 = @intCast(@divFloor(@mod(ns, std.time.ns_per_s), std.time.ns_per_us));
        const date = std.fmt.bufPrint(buffer, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}", .{
            year,                  month_day.month.numeric(), @as(u8, month_day.day_index) + 1,
            day.getHoursIntoDay(), day.getMinutesIntoHour(),  day.getSecondsIntoMinute(),
        }) catch unreachable;
        assert(date.len == 19);
        if (micros == 0) return date;
        const fraction = std.fmt.bufPrint(buffer[date.len..], ".{d:0>6}", .{micros}) catch unreachable;
        assert(fraction.len == 7);
        return buffer;
    }
};

fn newUuid(io: std.Io) error{HistoryEntropyUnavailable}![36]u8 {
    var bytes: [16]u8 = undefined;
    io.randomSecure(&bytes) catch return error.HistoryEntropyUnavailable;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = std.fmt.bytesToHex(bytes, .lower);
    var uuid: [36]u8 = undefined;
    const formatted = std.fmt.bufPrint(&uuid, "{s}-{s}-{s}-{s}-{s}", .{
        hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..],
    }) catch unreachable;
    assert(formatted.len == uuid.len);
    return uuid;
}
