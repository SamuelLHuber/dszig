//! Fixed scratch memory reserved once by the caller (TigerStyle static
//! allocation). Allocations bump an index into the caller's buffer; scopes
//! release in LIFO order, so nothing grows at runtime and exhaustion is
//! `error.OutOfMemory`. Only the innermost open scope may allocate: an outer
//! scope that allocates while an inner one is open would have its memory
//! released by the inner close, so it trips an assertion instead. A
//! Workspace is used by one thread at a time; each concurrent worker owns one.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

pub const Workspace = struct {
    fixed: std.heap.FixedBufferAllocator,
    depth: u32 = 0,
    used_peak: usize = 0,

    pub fn init(buffer: []u8) Workspace {
        return .{ .fixed = .init(buffer) };
    }

    pub fn capacity(ws: *const Workspace) usize {
        return ws.fixed.buffer.len;
    }

    pub fn used(ws: *const Workspace) usize {
        return ws.fixed.end_index;
    }

    /// Peak bytes in use, including any open scope.
    pub fn peak(ws: *const Workspace) usize {
        return @max(ws.used_peak, ws.fixed.end_index);
    }

    /// Everything allocated through the scope is released by `close`;
    /// results that outlive it belong in another allocator. Close scopes in
    /// reverse order of opening.
    pub fn open(ws: *Workspace) Scope {
        ws.depth += 1;
        return .{ .ws = ws, .mark = ws.fixed.end_index, .depth = ws.depth };
    }

    pub const Scope = struct {
        ws: *Workspace,
        mark: usize,
        depth: u32,

        pub fn allocator(scope: *Scope) Allocator {
            return .{ .ptr = scope, .vtable = &vtable };
        }

        pub fn close(scope: *Scope) void {
            const ws = scope.ws;
            assert(ws.depth == scope.depth);
            assert(scope.mark <= ws.fixed.buffer.len);
            ws.depth -= 1;
            ws.used_peak = @max(ws.used_peak, ws.fixed.end_index);
            // Freeing the newest pre-scope allocation can leave end_index below the mark.
            ws.fixed.end_index = @min(ws.fixed.end_index, scope.mark);
            scope.* = undefined;
        }

        const vtable: Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

        fn innermost(ptr: *anyopaque) *Workspace {
            const scope: *Scope = @ptrCast(@alignCast(ptr));
            assert(scope.ws.depth == scope.depth);
            return scope.ws;
        }

        fn alloc(ptr: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
            const ws = innermost(ptr);
            return ws.fixed.allocator().rawAlloc(len, alignment, ret_addr);
        }

        fn resize(ptr: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
            const ws = innermost(ptr);
            return ws.fixed.allocator().rawResize(memory, alignment, new_len, ret_addr);
        }

        fn remap(ptr: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            const ws = innermost(ptr);
            return ws.fixed.allocator().rawRemap(memory, alignment, new_len, ret_addr);
        }

        fn free(ptr: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
            const scope: *Scope = @ptrCast(@alignCast(ptr));
            // Releasing an outer allocation early is harmless; close resets anyway.
            scope.ws.fixed.allocator().rawFree(memory, alignment, ret_addr);
        }
    };
};

/// Test-only workspace, reserved statically once. Serial test code shares it;
/// nested scopes stay LIFO. Concurrent workers bind their own workspaces.
pub fn forTests() *Workspace {
    if (!@import("builtin").is_test) @compileError("Workspace.forTests is for tests");
    const state = struct {
        var buffer: [16 * 1024 * 1024]u8 = undefined;
        var workspace: ?Workspace = null;
    };
    if (state.workspace == null) state.workspace = .init(&state.buffer);
    return &state.workspace.?;
}

const testing = std.testing;

test "scopes release in LIFO order and never grow past the buffer" {
    var buffer: [256]u8 = undefined;
    var ws: Workspace = .init(&buffer);
    var outer = ws.open();
    const kept = try outer.allocator().alloc(u8, 64);
    @memset(kept, 'k');
    {
        var inner = ws.open();
        defer inner.close();
        _ = try inner.allocator().alloc(u8, 128);
        try testing.expectEqual(@as(usize, 192), ws.used());
        try testing.expectError(error.OutOfMemory, inner.allocator().alloc(u8, 128));
    }
    try testing.expectEqual(@as(usize, 64), ws.used());
    const want: [64]u8 = @splat('k');
    try testing.expectEqualSlices(u8, &want, kept);
    outer.close();
    try testing.expectEqual(@as(usize, 0), ws.used());
    try testing.expectEqual(@as(usize, 192), ws.peak());
    try testing.expectEqual(@as(usize, 256), ws.capacity());
}

test "closing after freeing the newest pre-scope allocation keeps the lower index" {
    var buffer: [64]u8 = undefined;
    var ws: Workspace = .init(&buffer);
    var outer = ws.open();
    const first = try outer.allocator().alloc(u8, 16);
    var inner = ws.open();
    outer.allocator().free(first);
    inner.close();
    try testing.expectEqual(@as(usize, 0), ws.used());
    outer.close();
}
