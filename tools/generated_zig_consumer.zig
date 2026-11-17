//! Linux qualification consumer. A caller supplies an empty delegated lane,
//! a private workspace, the pinned compiler, and its immutable closure module.
//! `generated` exercises a real HTTP provider (a fixture is not live inference).
//! Other cases exercise failures followed by successful reuse of the SAME engine.
const std = @import("std");
const builtin = @import("builtin");
const dszig = @import("dszig");
const closure = @import("compiler_closure");
const native = dszig.generated_zig;
const linux = std.os.linux;

var engine: native.Executor = undefined;
var tool_scratch: [256 * 1024]u8 = undefined;

const Case = enum {
    generated,
    success,
    unknown_tool,
    invalid_argument,
    invalid_header,
    oversize_header,
    partial_frame,
    extra_after_finish,
    unread_reply,
    finish_then_crash,
    finish_then_hang,
    file_escape,
    network_escape,
    process_escape,
    memory_escape,
    no_finish,
    invalid_source,
    compiler_error,
    diagnostics_flood,
};

const Options = struct {
    lane: [:0]const u8,
    workspace: [:0]const u8,
    compiler: [:0]const u8,
    case: Case,
    endpoint: ?[]const u8,
    timeout_ms: u32,
};

const lookup_tool: dszig.Tool = .{
    .name = "lookup",
    .desc = "Return twice the integer input as the answer.",
    .args = &.{dszig.react.Arg.typed("input", i64)},
    .func = lookupValue,
};

pub fn main(init: std.process.Init) !void {
    if (comptime builtin.os.tag != .linux) return error.UnsupportedPlatform;
    const options = try arguments(try init.minimal.args.toSlice(init.arena.allocator()));
    const descriptor = linux.openat(linux.AT.FDCWD, options.lane, .{ .DIRECTORY = true, .CLOEXEC = true, .NOFOLLOW = true }, 0);
    if (descriptor > std.math.maxInt(i32)) return error.OpenLaneFailed;
    defer _ = linux.syscall1(.close, descriptor);
    var lane = try native.Lane.configure(@intCast(descriptor), .{});
    var tool_state: ToolState = .{ .lane = &lane };
    const capabilities = [_]native.Capability{
        .{ .name = lookup_tool.name, .context = &tool_state, .call = lookup },
    };
    try engine.init(&.{ .lane = &lane, .worker = .{
        .workspace = options.workspace,
        .compiler = options.compiler,
        .read_only_roots = &closure.paths,
    }, .tools = &capabilities, .timeout_ms = options.timeout_ms });

    if (options.case == .generated) {
        try generated(init, options.endpoint orelse return error.MissingEndpoint);
        if (tool_state.calls != 1) return error.WrongToolCalls;
        if (!try lane.empty()) return error.LaneNotJoined;
        std.debug.print("case=generated result=42 calls=1 lane_empty=yes\n", .{});
        return;
    }
    try exercise(options.case, &lane, &tool_state.calls);
}

fn arguments(args: []const [:0]const u8) !Options {
    if (args.len < 6 or args.len > 7) return error.InvalidArguments;
    return .{
        .lane = args[1],
        .workspace = args[2],
        .compiler = args[3],
        .case = std.meta.stringToEnum(Case, args[4]) orelse return error.InvalidCase,
        .timeout_ms = try std.fmt.parseInt(u32, args[5], 10),
        .endpoint = if (args.len == 7) args[6] else null,
    };
}

fn generated(init: std.process.Init, endpoint: []const u8) !void {
    var client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer client.deinit();
    var provider: dszig.lm.OpenAI = .{
        .client = &client,
        .base_url = endpoint,
        .model = "native-zig-proof",
        .api_key = init.environ_map.get("OPENAI_API_KEY") orelse "fixture-only",
    };
    const generator: native.Generator = .{};
    const result = native.generateAndRun(&generator, &engine, .{
        .arena = init.arena.allocator(),
        .lm = provider.lm(),
        .json_fallback = false,
    }, .{
        .task = "Call lookup with the provided input and finish with its exact JSON reply.",
        .input_json = "{\"input\":21}",
        .capabilities = "lookup accepts {\"input\":integer} and returns {\"answer\":integer}.",
    }) catch |err| {
        std.debug.print("generated_error={s} diagnostics={s}\n", .{ @errorName(err), engine.diagnostics() });
        return err;
    };
    if (!std.mem.eql(u8, result, "{\"answer\":42}")) return error.WrongResult;
}

fn exercise(case: Case, lane: *const native.Lane, calls: *u32) !void {
    const result = engine.run(program(case), "{\"input\":21}");
    if (case == .success) {
        if (!std.mem.eql(u8, try result, "{\"answer\":42}")) return error.WrongResult;
        if (calls.* != 1) return error.WrongToolCalls;
    } else {
        if (result) |_| return error.UnexpectedPublication else |err| {
            std.debug.print("case={s} refusal={s} diagnostics={s}\n", .{ @tagName(case), @errorName(err), engine.diagnostics() });
            try expected(case, err);
        }
        const expected_calls: u32 = if (case == .unread_reply) 1 else 0;
        if (calls.* != expected_calls) return error.UnexpectedToolEffect;
    }
    if (!try lane.empty()) return error.LaneNotJoined;
    const before = calls.*;
    const recovered = try engine.run(program(.success), "{\"input\":21}");
    if (!std.mem.eql(u8, recovered, "{\"answer\":42}")) return error.WrongRecoveryResult;
    if (calls.* != before + 1) return error.WrongRecoveryToolCalls;
    if (!try lane.empty()) return error.RecoveryLaneNotJoined;
    std.debug.print("case={s} recovery=42 lane_empty=yes\n", .{@tagName(case)});
}

fn expected(case: Case, failure: anyerror) !void {
    const accepted = switch (case) {
        .unknown_tool => failure == error.UnknownTool,
        .invalid_argument => failure == error.ToolFailed,
        .invalid_header => failure == error.InvalidFrameVersion,
        .oversize_header => failure == error.FrameCapacityExceeded,
        .partial_frame => failure == error.PartialFrame,
        .extra_after_finish => failure == error.UnexpectedFrame,
        .unread_reply => failure == error.ReceiveFailed,
        .finish_then_crash, .file_escape, .network_escape, .process_escape, .memory_escape, .no_finish => failure == error.ChildFailed,
        .finish_then_hang => failure == error.DeadlineExceeded,
        .invalid_source, .compiler_error => failure == error.ChildFailed,
        .diagnostics_flood => failure == error.DiagnosticsOverflow,
        .generated, .success => false,
    };
    if (!accepted) return failure;
}

const ToolState = struct { lane: *const native.Lane, calls: u32 = 0 };

/// Only the trusted namespace PID1 and its policy-confined runtime may remain
/// when a capability runs. Compiler descendants must already have been reaped.
fn runtimeMembership(lane: *const native.Lane) native.CapabilityError!void {
    const descriptor = linux.openat(lane.descriptor, "cgroup.procs", .{ .CLOEXEC = true, .NOFOLLOW = true }, 0);
    if (descriptor > std.math.maxInt(i32)) return error.ToolFailed;
    defer _ = linux.syscall1(.close, descriptor);
    var buffer: [256]u8 = undefined;
    const count = linux.syscall3(.read, descriptor, @intFromPtr(&buffer), buffer.len);
    if (count >= buffer.len) return error.ToolFailed;
    var members = std.mem.tokenizeScalar(u8, buffer[0..count], '\n');
    var processes: u16 = 0;
    while (members.next()) |_| processes += 1;
    if (processes != 2) return error.ToolFailed;
}

fn lookup(raw: *anyopaque, argument: []const u8, output: []u8, _: u64) native.CapabilityError!u32 {
    // Fixed-size typed normalization plus one bounded in-memory cgroup read.
    // No unbounded native callback work; the executor checks its deadline around it.
    const state: *ToolState = @ptrCast(@alignCast(raw));
    try runtimeMembership(state.lane);
    var scratch = std.heap.FixedBufferAllocator.init(&tool_scratch);
    const arena = scratch.allocator();
    const parsed = std.json.parseFromSliceLeaky(dszig.Value, arena, argument, .{}) catch
        return error.ToolFailed;
    if (parsed != .object) return error.ToolFailed;
    const answer = lookup_tool.call(.{ .arena = arena }, &parsed.object) catch
        return error.ToolFailed;
    var writer: std.Io.Writer = .fixed(output);
    std.json.Stringify.value(answer, .{}, &writer) catch return error.ToolFailed;
    state.calls += 1;
    return @intCast(writer.end);
}

fn lookupValue(context: dszig.Context, args: *const dszig.value.Record) dszig.react.ToolError!dszig.Value {
    const input = args.get("input") orelse return error.InvalidArguments;
    if (input != .integer) return error.InvalidArguments;
    const answer = std.math.mul(i64, input.integer, 2) catch return error.InvalidArguments;
    var result: dszig.value.Record = .empty;
    try result.put(context.arena, "answer", .{ .integer = answer });
    return .{ .object = result };
}

const prefix = "const guest = @import(\"guest\");\npub fn run(ctx: *guest.Context) !void {\n";
fn program(case: Case) []const u8 {
    return switch (case) {
        .generated, .success => prefix ++
            "const reply = try ctx.tool(\"lookup\", try ctx.input());\n" ++
            "try ctx.finish(reply);\n}\n",
        .unknown_tool => prefix ++ "try ctx.finish(try ctx.tool(\"forbidden\", \"{}\"));\n}\n",
        .invalid_argument => prefix ++ "try ctx.finish(try ctx.tool(\"lookup\", \"{}\"));\n}\n",
        .invalid_header => prefix ++ "_ = ctx;\n" ++
            "const bytes = \"\\x02\\x04\\x00\\x00\\x00\\x00\\x00\\x00\";\n" ++
            "_ = guest.std.os.linux.write(1, bytes.ptr, bytes.len);\n}\n",
        .oversize_header => prefix ++ "_ = ctx;\n" ++
            "const bytes = \"\\x01\\x04\\x00\\x00\\x00\\x01\\x00\\x01\";\n" ++
            "_ = guest.std.os.linux.write(1, bytes.ptr, bytes.len);\n}\n",
        .partial_frame => prefix ++ "_ = ctx;\n" ++
            "const bytes = \"\\x01\\x04\\x00\\x00\\x00\\x00\\x00\\x20abc\";\n" ++
            "_ = guest.std.os.linux.write(1, bytes.ptr, bytes.len);\n}\n",
        .extra_after_finish => prefix ++ "try ctx.finish(\"{}\");\n" ++
            "_ = guest.std.os.linux.write(1, \"x\", 1);\n}\n",
        .unread_reply => prefix ++
            "const bytes = \"\\x01\\x02\\x00\\x00\\x00\\x00\\x00\\x14\\x00\\x06lookup{\\\"input\\\":21}\";\n" ++
            "_ = guest.std.os.linux.write(1, bytes.ptr, bytes.len);\n" ++
            "var reply_prefix: [9]u8 = undefined;\nvar offset: usize = 0;\n" ++
            "while (offset < 9) {\n" ++
            "const count = guest.std.os.linux.read(0, reply_prefix[offset..].ptr, 9 - offset);\n" ++
            "if (count == 0 or count > 9 - offset) return error.ReplyReadFailed;\n" ++
            "offset += count;\n}\ntry ctx.finish(\"{}\");\n}\n",
        .finish_then_crash => prefix ++ "try ctx.finish(\"{}\");\n@panic(\"controlled\");\n}\n",
        .finish_then_hang => prefix ++ "try ctx.finish(\"{}\");\n" ++
            "while (true) { guest.std.atomic.spinLoopHint(); }\n}\n",
        .file_escape => prefix ++ "_ = ctx;\n" ++
            "_ = guest.std.os.linux.openat(-100, \"/etc/passwd\", .{}, 0);\n}\n",
        .network_escape => prefix ++ "_ = ctx;\n" ++
            "_ = guest.std.os.linux.socket(2, 1, 0);\n}\n",
        .process_escape => prefix ++ "_ = ctx;\n" ++
            "_ = guest.std.os.linux.syscall5(.clone, 17, 0, 0, 0, 0);\n}\n",
        .memory_escape => prefix ++ "_ = ctx;\n" ++
            "_ = guest.std.os.linux.syscall6(.mmap, 0, 4096, 3, 0x22, ~@as(usize, 0), 0);\n}\n",
        .no_finish => prefix ++ "_ = ctx;\n}\n",
        .invalid_source => "const guest = @import(\"std\");\n" ++
            "pub fn run(ctx: *guest.Context) !void { _ = ctx; }\n",
        .compiler_error => prefix ++ "_ = ctx;\nundefined_symbol();\n}\n",
        .diagnostics_flood => prefix ++ "_ = ctx;\n" ++
            "const message: [32768]u8 = @splat('x');\n@compileError(&message);\n}\n",
    };
}
