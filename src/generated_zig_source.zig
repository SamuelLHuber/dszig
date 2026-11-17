//! Conservative generated-source syntax policy, NOT a security boundary.
//! guest.std exposes OS facilities. The parent must isolate and deadline both
//! compilation and execution. No source check proves compile-time termination.
//! All parser allocations use caller-provisioned scratch; reset it between calls.
const std = @import("std");
const Ast = std.zig.Ast;
const assert = std.debug.assert;

pub const source_bytes_max: u32 = 64 * 1024;
/// Includes documentation comments and EOF; ordinary comments consume byte work.
pub const tokens_max: u32 = 4096;
pub const nodes_max: u32 = 4096;
/// Bounds delimiter nesting only. Undelimited parser recursion (prefix operators,
/// nested if/while without braces) is bounded by tokens_max, so the validating
/// thread needs a stack sized for a token-deep std.zig parse.
pub const nesting_max: u32 = 64;
pub const work_max: u32 = 128 * 1024;
/// Provision at startup, then pass a FixedBufferAllocator over this storage.
/// Exhaustion is an ordinary error, not permission to fall back to a heap.
pub const scratch_bytes_recommended: u32 = 2 * 1024 * 1024;

pub const Error = error{
    OutOfMemory,
    SourceTooLarge,
    TooManyTokens,
    TooManyNodes,
    NestingTooDeep,
    WorkLimitExceeded,
    InvalidSyntax,
    ForbiddenSyntax,
    ForbiddenBuiltin,
    InvalidImport,
    InvalidEntrypoint,
};

/// Requires the canonical root declarations const guest = @import("guest");
/// and pub fn run(ctx: *guest.Context) !void { ... }. Private helper functions
/// and variables are allowed; other public root declarations are not. Escaped
/// identifiers and root test/comptime declarations are intentionally unsupported.
/// The compiler still checks nominal types, name resolution and body semantics.
pub fn validate(allocator: std.mem.Allocator, source: [:0]const u8) Error!void {
    if (source.len > source_bytes_max) return error.SourceTooLarge;
    var work: u32 = @intCast(source.len);
    try preflight(source, &work);
    var tree = try Ast.parse(allocator, source, .{ .mode = .zig, .recover = false });
    defer tree.deinit(allocator);
    if (tree.errors.len != 0) return error.InvalidSyntax;
    assert(tree.tokens.len <= tokens_max);
    if (tree.nodes.len > nodes_max) return error.TooManyNodes;
    try spend(&work, @intCast(tree.nodes.len));
    const import_node = try validateImports(&tree);
    try validateRoot(&tree, import_node, &work);
    assert(work <= work_max);
}

fn spend(work: *u32, amount: u32) Error!void {
    assert(work.* <= work_max);
    if (amount > work_max - work.*) return error.WorkLimitExceeded;
    work.* += amount;
    assert(work.* <= work_max);
}

fn preflight(source: [:0]const u8, work: *u32) Error!void {
    assert(source.len <= source_bytes_max);
    var tokenizer = std.zig.Tokenizer.init(source);
    var delimiters: [nesting_max]std.zig.Token.Tag = undefined;
    var depth: u32 = 0;
    for (0..tokens_max) |_| {
        const token = tokenizer.next();
        try spend(work, 1);
        const text = source[token.loc.start..token.loc.end];
        switch (token.tag) {
            .invalid => return error.InvalidSyntax,
            .keyword_asm,
            .keyword_extern,
            .keyword_export,
            .keyword_linksection,
            .keyword_addrspace,
            .keyword_callconv,
            .keyword_threadlocal,
            => return error.ForbiddenSyntax,
            .identifier => {
                if (text.len > 0 and text[0] == '@') return error.ForbiddenSyntax;
            },
            .builtin => {
                if (!std.mem.eql(u8, text, "@import") and !allowedBuiltin(text)) {
                    return error.ForbiddenBuiltin;
                }
            },
            .l_paren, .l_brace, .l_bracket => {
                if (depth == nesting_max) return error.NestingTooDeep;
                delimiters[depth] = token.tag;
                depth += 1;
            },
            .r_paren, .r_brace, .r_bracket => {
                if (depth == 0) return error.InvalidSyntax;
                depth -= 1;
                const expected: std.zig.Token.Tag = switch (token.tag) {
                    .r_paren => .l_paren,
                    .r_brace => .l_brace,
                    .r_bracket => .l_bracket,
                    else => unreachable,
                };
                if (delimiters[depth] != expected) return error.InvalidSyntax;
            },
            .eof => {
                if (token.loc.start != source.len) return error.InvalidSyntax;
                if (depth != 0) return error.InvalidSyntax;
                return;
            },
            else => {},
        }
    }
    return error.TooManyTokens;
}

/// Deny by default, including future compiler builtins. Direct integer/pointer
/// conversion and @bitCast are absent, but @ptrCast and guest.std byte helpers
/// can still reinterpret stored integers as pointers. This is a language subset,
/// not a proof of memory or syscall safety.
fn allowedBuiltin(name: []const u8) bool {
    const allowed = std.StaticStringMap(void).initComptime(.{
        .{ "@abs", {} },             .{ "@addWithOverflow", {} }, .{ "@alignCast", {} },
        .{ "@alignOf", {} },         .{ "@as", {} },              .{ "@bitReverse", {} },
        .{ "@bitOffsetOf", {} },     .{ "@bitSizeOf", {} },       .{ "@byteSwap", {} },
        .{ "@ceil", {} },            .{ "@clz", {} },             .{ "@compileError", {} },
        .{ "@constCast", {} },       .{ "@ctz", {} },             .{ "@divExact", {} },
        .{ "@divFloor", {} },        .{ "@divTrunc", {} },        .{ "@enumFromInt", {} },
        .{ "@errorCast", {} },       .{ "@errorFromInt", {} },    .{ "@errorName", {} },
        .{ "@exp", {} },             .{ "@exp2", {} },            .{ "@field", {} },
        .{ "@FieldType", {} },       .{ "@floatCast", {} },       .{ "@floatFromInt", {} },
        .{ "@floor", {} },           .{ "@hasDecl", {} },         .{ "@hasField", {} },
        .{ "@intCast", {} },         .{ "@intFromBool", {} },     .{ "@intFromEnum", {} },
        .{ "@intFromError", {} },    .{ "@intFromFloat", {} },    .{ "@log", {} },
        .{ "@log2", {} },            .{ "@log10", {} },           .{ "@max", {} },
        .{ "@memcpy", {} },          .{ "@memmove", {} },         .{ "@memset", {} },
        .{ "@min", {} },             .{ "@mod", {} },             .{ "@mulAdd", {} },
        .{ "@mulWithOverflow", {} }, .{ "@offsetOf", {} },        .{ "@popCount", {} },
        .{ "@ptrCast", {} },         .{ "@reduce", {} },          .{ "@rem", {} },
        .{ "@round", {} },           .{ "@select", {} },          .{ "@shlExact", {} },
        .{ "@shlWithOverflow", {} }, .{ "@shrExact", {} },        .{ "@shuffle", {} },
        .{ "@sin", {} },             .{ "@sizeOf", {} },          .{ "@splat", {} },
        .{ "@sqrt", {} },            .{ "@subWithOverflow", {} }, .{ "@tagName", {} },
        .{ "@This", {} },            .{ "@truncate", {} },        .{ "@trunc", {} },
        .{ "@typeName", {} },        .{ "@typeInfo", {} },        .{ "@TypeOf", {} },
        .{ "@unionInit", {} },       .{ "@Vector", {} },          .{ "@cos", {} },
        .{ "@tan", {} },             .{ "@divCeil", {} },         .{ "@panic", {} },
        .{ "@backingInt", {} },      .{ "@fromBackingInt", {} },  .{ "@Int", {} },
    });
    return allowed.has(name);
}

fn validateImports(tree: *const Ast) Error!Ast.Node.Index {
    assert(tree.nodes.len <= nodes_max);
    var found: ?Ast.Node.Index = null;
    for (0..tree.nodes.len) |index| {
        const node: Ast.Node.Index = @fromBackingInt(@intCast(index));
        var buffer: [2]Ast.Node.Index = undefined;
        const params = tree.builtinCallParams(&buffer, node) orelse continue;
        const token = tree.nodeMainToken(node);
        if (!std.mem.eql(u8, tree.tokenSlice(token), "@import")) continue;
        if (found != null) return error.InvalidImport;
        if (params.len != 1) return error.InvalidImport;
        if (tree.nodeTag(params[0]) != .string_literal) return error.InvalidImport;
        const target = tree.tokenSlice(tree.nodeMainToken(params[0]));
        if (!std.mem.eql(u8, target, "\"guest\"")) return error.InvalidImport;
        found = node;
    }
    return found orelse error.InvalidImport;
}

fn validateRoot(tree: *const Ast, import_node: Ast.Node.Index, work: *u32) Error!void {
    var guest_found = false;
    var run_found = false;
    for (tree.rootDecls()) |node| {
        try spend(work, 1);
        if (tree.fullVarDecl(node)) |decl| {
            const name = tree.tokenSlice(decl.ast.mut_token + 1);
            if (decl.visib_token != null) return error.InvalidEntrypoint;
            if (std.mem.eql(u8, name, "main")) return error.InvalidEntrypoint;
            if (std.mem.eql(u8, name, "run")) return error.InvalidEntrypoint;
            if (!std.mem.eql(u8, name, "guest")) continue;
            if (guest_found) return error.InvalidImport;
            if (decl.ast.init_node.unwrap() != import_node) return error.InvalidImport;
            const canonical = [_][]const u8{
                "const", "guest", "=", "@import", "(", "\"guest\"", ")", ";",
            };
            if (!tokensEqual(tree, decl.ast.mut_token, &canonical)) return error.InvalidImport;
            guest_found = true;
            continue;
        }
        if (tree.nodeTag(node) != .fn_decl) return error.InvalidEntrypoint;
        var buffer: [1]Ast.Node.Index = undefined;
        const function = tree.fullFnProto(&buffer, node).?;
        const name = tree.tokenSlice(function.name_token orelse return error.InvalidEntrypoint);
        if (std.mem.eql(u8, name, "main")) return error.InvalidEntrypoint;
        if (!std.mem.eql(u8, name, "run")) {
            if (function.visib_token != null) return error.InvalidEntrypoint;
            continue;
        }
        if (run_found) return error.InvalidEntrypoint;
        const public_token = function.visib_token orelse return error.InvalidEntrypoint;
        const canonical = [_][]const u8{
            "pub", "fn", "run",  "(", "ctx", ":", "*", "guest", ".", "Context",
            ")",   "!",  "void", "{",
        };
        if (!tokensEqual(tree, public_token, &canonical)) return error.InvalidEntrypoint;
        run_found = true;
    }
    if (!guest_found) return error.InvalidImport;
    if (!run_found) return error.InvalidEntrypoint;
}

fn tokensEqual(tree: *const Ast, start: Ast.TokenIndex, expected: []const []const u8) bool {
    assert(expected.len <= 14);
    assert(start < tree.tokens.len);
    if (expected.len > tree.tokens.len - start) return false;
    for (expected, 0..) |text, offset| {
        if (!std.mem.eql(u8, tree.tokenSlice(start + @as(u32, @intCast(offset))), text)) {
            return false;
        }
    }
    return true;
}

const valid_prefix = "const guest = @import(\"guest\");\n";
const valid_run = "pub fn run(ctx: *guest.Context) !void { _ = ctx; }\n";

fn testValidate(source: [:0]const u8) Error!void {
    var scratch: [scratch_bytes_recommended]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&scratch);
    try validate(fixed.allocator(), source);
}

test "accept canonical entrypoint, private helpers and normal fixed storage builtins" {
    try testValidate(valid_prefix ++
        "const std = guest.std;\n" ++
        "fn helper(bytes: []u8) void { @memset(bytes, 0); }\n" ++
        "pub fn run(ctx: *guest.Context) !void {\n" ++
        " var buffer: [32]u8 = undefined; helper(&buffer);\n" ++
        " const ptr: [*]u8 = @ptrCast(&buffer); _ = ptr;\n" ++
        " try ctx.finish(&buffer);\n" ++
        "}\n");
}

test "comments and string contents are not executable syntax" {
    try testValidate(valid_prefix ++
        "// @import(\"std\") asm extern export @ptrFromInt(1)\n" ++
        "const text = \"@cImport extern export asm\";\n" ++ valid_run);
}

test "reject foreign imports, duplicate imports and noncanonical guest bindings" {
    const cases = [_][:0]const u8{
        "const guest = @import(\"std\");\n" ++ valid_run,
        valid_prefix ++ "const other = @import(\"guest\");\n" ++ valid_run,
        "const other = @import(\"guest\");\n" ++ valid_run,
        "var guest = @import(\"guest\");\n" ++ valid_run,
        "const guest = @import(\"gu\\x65st\");\n" ++ valid_run,
        "const guest = @import(\"gu\" ++ \"est\");\n" ++ valid_run,
    };
    for (cases) |source| try std.testing.expectError(error.InvalidImport, testValidate(source));
}

test "reject pointer forging, foreign interfaces and unknown builtins" {
    const calls = [_][:0]const u8{
        "@ptrFromInt(1)", "@intFromPtr(ctx)",         "@fieldParentPtr(\"x\", ctx)",
        "@cImport({})",   "@cInclude(\"a.h\")",       "@cDefine(\"x\", \"y\")",
        "@cUndef(\"x\")", "@embedFile(\"secret\")",   "@extern(u8, .{})",
        "@bitCast(1)",    "@setRuntimeSafety(false)", "@futureBuiltin()",
    };
    inline for (calls) |call| {
        try std.testing.expectError(error.ForbiddenBuiltin, testValidate(
            valid_prefix ++ "pub fn run(ctx: *guest.Context) !void { _ = " ++ call ++ "; }",
        ));
    }
}

test "reject assembly, extern, export and escaped identifiers" {
    const cases = [_][:0]const u8{
        valid_prefix ++ "extern fn foreign() void;\n" ++ valid_run,
        valid_prefix ++ "export fn foreign() void {}\n" ++ valid_run,
        valid_prefix ++ "pub fn run(ctx: *guest.Context) !void { asm (\"\"); }",
        valid_prefix ++ "pub fn @\"run\"(ctx: *guest.Context) !void { _ = ctx; }",
        valid_prefix ++ "fn naked() callconv(.naked) void {}\n" ++ valid_run,
        valid_prefix ++ "var placed: u8 linksection(\".init\") = 0;\n" ++ valid_run,
        valid_prefix ++ "threadlocal var slot: u8 = 0;\n" ++ valid_run,
    };
    for (cases) |source| try std.testing.expectError(error.ForbiddenSyntax, testValidate(source));
}

test "require a genuine canonical public run definition and no alternate root entry" {
    const cases = [_][:0]const u8{
        valid_prefix,
        valid_prefix ++ "const run = struct {};",
        valid_prefix ++ "fn run(ctx: *guest.Context) !void { _ = ctx; }",
        valid_prefix ++ "pub fn run(ctx: *guest.Context) void { _ = ctx; }",
        valid_prefix ++ "pub fn run(ctx: *const guest.Context) !void { _ = ctx; }",
        valid_prefix ++ "pub fn run(ctx: *guest.Context, x: u8) !void { _ = ctx; _ = x; }",
        valid_prefix ++ "pub fn run(ctx: *guest.Context) !void;",
        valid_prefix ++ valid_run ++ "pub fn main() void {}",
        valid_prefix ++ valid_run ++ "fn main() void {}",
        valid_prefix ++ valid_run ++ "pub const another = 1;",
        valid_prefix ++ valid_run ++ valid_run,
    };
    for (cases) |source| {
        try std.testing.expectError(error.InvalidEntrypoint, testValidate(source));
    }
}

test "reject malformed syntax and embedded NUL" {
    try std.testing.expectError(error.InvalidSyntax, testValidate(valid_prefix ++ "fn ("));
    const embedded_nul = valid_prefix ++ valid_run ++ "\x00";
    try std.testing.expectError(error.InvalidSyntax, testValidate(embedded_nul));
    try std.testing.expectError(error.InvalidSyntax, testValidate(valid_prefix ++ "const x = ;"));
}

test "bound bytes, tokens, nesting, work and caller scratch" {
    const oversized: [source_bytes_max + 1:0]u8 = @splat(' ');
    try std.testing.expectError(error.SourceTooLarge, testValidate(&oversized));
    const many_tokens: [tokens_max:0]u8 = @splat(';');
    try std.testing.expectError(error.TooManyTokens, testValidate(&many_tokens));
    const deep: [nesting_max + 1:0]u8 = @splat('(');
    try std.testing.expectError(error.NestingTooDeep, testValidate(&deep));
    var work: u32 = work_max;
    try std.testing.expectError(error.WorkLimitExceeded, spend(&work, 1));
    var empty: [0]u8 = .{};
    var fixed = std.heap.FixedBufferAllocator.init(&empty);
    const valid = valid_prefix ++ valid_run;
    try std.testing.expectError(error.OutOfMemory, validate(fixed.allocator(), valid));
}

test "token bound keeps undelimited parser recursion finite" {
    const chain: [tokens_max - 64:0]u8 = @splat('!');
    try testValidate(valid_prefix ++
        "pub fn run(ctx: *guest.Context) !void { _ = ctx; _ = " ++ chain ++ "true; }");
}
