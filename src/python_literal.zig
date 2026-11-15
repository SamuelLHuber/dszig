//! CPython 3.11 `ast.literal_eval` for a UTF-8 source string, without execution.
//!
//! This is Python literal grammar, not JSON repair. Parentheses retain AST
//! shape: `1+(2j)` is valid, but `1+(-2j)`, `--1`, and `-(1+2j)` are not.
//! Only the actual empty `set()` call is admitted, including a parenthesized
//! or NFKC-equivalent identifier. `Ellipsis` is a name, not the `...` literal.
//!
//! Returned values deep-own their data in `a`, including every builtin box;
//! source storage and temporary parser storage may be destroyed immediately.
//! They use inference.freeJsonValue's ordinary recursive ownership convention.
//! There are no aliases/cycles in a literal expression. A scratch arena avoids
//! repeatedly cloning nested borrowed builtin constructors and unwinds *all*
//! temporary allocations on syntax, hashability, depth, and allocation errors.
//!
//! Integers outside i64 retain canonical decimal number_string data, including
//! arbitrarily large base-2/8/16 literals. Only nonzero decimal integer tokens
//! have CPython's default 4300-digit limit (underscores do not count). Float
//! literals use correctly rounded binary64, including overflow and signed zero.
//! Escaped lone surrogates are retained as WTF-8; literal source surrogates and
//! other invalid UTF-8 are rejected, as ast.parse's UTF-8 encoding rejects them.
//! Universal newlines apply even inside raw/triple strings. Unknown escapes are
//! preserved as in Python 3.11; compiler warnings are not parser return values.
//!
//! ParseValue covers Python syntax and literal_eval's unsupported AST shapes.
//! LiteralEvaluation denotes uncaught TypeError/OverflowError (unhashable keys
//! or elements, integer-to-complex overflow), which must NOT retain the source
//! as a successful string. Python's >200-open-delimiter refusal is SyntaxError,
//! caught by DSPy's original-string fallback.
//! IntegerTooLong stays distinct; the caller can map CPython's caught decimal
//! conversion SyntaxError separately. JsonTooDeep is a genuine native AST-depth
//! refusal: expression bodies have a deterministic 3000-node ceiling. Unary
//! +, -, and ~ chains are traversed iteratively, even when literal_eval would
//! reject their AST shape. Complete syntax is checked before reporting depth.
//!
//! This ceiling is NOT CPython's exact cutoff. Python-ast.c scales the current
//! recursion limit and already-used Python frames by three; a default limit
//! of 1000 does not make every caller's remaining AST budget 3000. Native
//! parsing does not emulate arbitrary caller stacks or recursionlimit changes.
//! Near-cutoff results, composite parser-resource refusals, general unsupported
//! Python AST error precedence, exception classes/messages, and object interning
//! therefore remain compatibility limits, not claims of full DSPy parity.
//!
//! Grounded in CPython v3.11.17 Lib/ast.py, Parser/tokenizer.c,
//! Parser/string_parser.c, and Modules/unicodedata.c. Unicode 14 names/aliases
//! and the small NFKC-to-`set` table come from the actual pinned interpreter;
//! regenerate with tools/dspy_reference/python_literal_oracle.py --write-unicode.

const std = @import("std");
const runtime = @import("runtime_value.zig");
const python = @import("python_value.zig");
const inference = @import("inference.zig");
const Allocator = std.mem.Allocator;
const Value = runtime.Value;

pub const Error = error{ ParseValue, LiteralEvaluation, JsonTooDeep, IntegerTooLong } || Allocator.Error;
pub const max_depth = 200;
pub const max_ast_depth = 3000;
const max_int_digits = 4300;

pub fn parse(a: Allocator, source: []const u8) Error!Value {
    if (std.mem.indexOfScalar(u8, source, 0) != null or !std.unicode.utf8ValidateSlice(source)) return error.ParseValue;
    var temporary = std.heap.ArenaAllocator.init(a);
    defer temporary.deinit();
    // literal_eval strips only leading spaces/tabs, before Python tokenization.
    var start: usize = 0;
    while (start < source.len and (source[start] == ' ' or source[start] == '\t')) start += 1;
    var parser: Parser = .{ .a = temporary.allocator(), .lexer = .{ .source = source, .index = start } };
    try parser.advance();
    while (parser.token == .newline) try parser.advance();
    const first = try parser.expression();
    const result = if (parser.token == .comma) try parser.tupleTail(first, .eof) else first;
    while (parser.token == .newline) try parser.advance();
    if (parser.token != .eof) return error.ParseValue;
    return inference.cloneJsonValue(a, try result.value());
}

const StringToken = struct {
    content: []const u8,
    raw: bool,
    bytes: bool,
};

const Token = union(enum) {
    eof,
    newline,
    comma,
    colon,
    lparen,
    rparen,
    lbracket,
    rbracket,
    lbrace,
    rbrace,
    plus,
    minus,
    invert,
    ellipsis,
    number: []const u8,
    name: []const u8,
    string: StringToken,
};

const Lexer = struct {
    source: []const u8,
    index: usize = 0,
    brackets: usize = 0,
    line_start: bool = true,

    fn at(self: *const Lexer, offset: usize) ?u8 {
        return if (offset < self.source.len - self.index) self.source[self.index + offset] else null;
    }

    fn newline(self: *Lexer) void {
        const cr = self.source[self.index] == '\r';
        self.index += 1;
        if (cr and self.at(0) == '\n') self.index += 1;
    }

    fn next(self: *Lexer) Error!Token {
        while (self.index < self.source.len) {
            if (self.line_start) {
                var column: usize = 0;
                var continuation_column: usize = 0;
                var had_continuation = false;
                while (self.at(0)) |c| {
                    switch (c) {
                        ' ' => column += 1,
                        '\t' => column = (column / 8 + 1) * 8,
                        '\x0c' => column = 0,
                        '\\' => {
                            // A continuation in indentation is not the same
                            // as a continuation after an expression token.
                            if (continuation_column == 0) continuation_column = column;
                            self.index += 1;
                            if (self.at(0) != '\n' and self.at(0) != '\r') return error.ParseValue;
                            self.newline();
                            if (self.index == self.source.len) return error.ParseValue;
                            had_continuation = true;
                            continue;
                        },
                        else => break,
                    }
                    self.index += 1;
                }
                if (continuation_column != 0) column = continuation_column;
                // Blank/comment-only lines do not generate indentation.
                const c = self.at(0);
                if (self.brackets == 0 and column != 0 and
                    (c != null or had_continuation) and c != '#' and c != '\n' and c != '\r') return error.ParseValue;
                self.line_start = false;
                if (c == null) break;
            }
            const c = self.source[self.index];
            switch (c) {
                ' ', '\t', '\x0c' => self.index += 1,
                '#' => {
                    while (self.at(0)) |comment| {
                        if (comment == '\n' or comment == '\r') break;
                        self.index += 1;
                    }
                },
                '\n', '\r' => {
                    self.newline();
                    self.line_start = true;
                    if (self.brackets == 0) return .newline;
                },
                '\\' => {
                    self.index += 1;
                    if (self.at(0) != '\n' and self.at(0) != '\r') return error.ParseValue;
                    self.newline();
                    if (self.index == self.source.len) return error.ParseValue;
                    // tok_get's continuation resumes the same logical line;
                    // indentation is not rechecked after an existing token.
                },
                else => break,
            }
        }
        if (self.index == self.source.len) return .eof;
        if (try self.string()) |literal| return .{ .string = literal };
        const start = self.index;
        const c = self.source[self.index];
        if (digit(c, 10) != null or (c == '.' and decimalDigit(self.at(1)))) {
            try self.number();
            if (self.at(0)) |after| {
                // CPython verifies the numeric token before converting it.
                // In particular, 4301 digits followed by 'x' is an invalid
                // token, not an IntegerTooLong conversion failure.
                if (std.ascii.isDigit(after) or
                    ((std.ascii.isAlphabetic(after) or after == '_') and
                        !numericKeywordSuffix(self.source[self.index..]))) return error.ParseValue;
            }
            return .{ .number = self.source[start..self.index] };
        }
        if (identifierByte(c)) {
            self.index += 1;
            while (self.at(0)) |next_byte| {
                if (!identifierByte(next_byte) and !std.ascii.isDigit(next_byte)) break;
                self.index += 1;
            }
            return .{ .name = self.source[start..self.index] };
        }
        self.index += 1;
        return switch (c) {
            ',' => .comma,
            ':' => .colon,
            '+' => .plus,
            '-' => .minus,
            '~' => .invert,
            '(', '[', '{' => blk: {
                if (self.brackets == max_depth) return error.ParseValue;
                self.brackets += 1;
                break :blk switch (c) { '(' => .lparen, '[' => .lbracket, else => .lbrace };
            },
            ')', ']', '}' => blk: {
                if (self.brackets == 0) return error.ParseValue;
                self.brackets -= 1;
                break :blk switch (c) { ')' => .rparen, ']' => .rbracket, else => .rbrace };
            },
            '.' => blk: {
                if (self.at(0) != '.' or self.at(1) != '.') return error.ParseValue;
                self.index += 2;
                break :blk .ellipsis;
            },
            else => error.ParseValue,
        };
    }

    fn string(self: *Lexer) Error!?StringToken {
        var quote_pos = self.index;
        while (quote_pos < self.source.len and quote_pos - self.index < 2 and std.ascii.isAlphabetic(self.source[quote_pos])) quote_pos += 1;
        if (quote_pos == self.source.len or (self.source[quote_pos] != '\'' and self.source[quote_pos] != '"')) return null;
        const prefix = self.source[self.index..quote_pos];
        var raw = false;
        var bytes = false;
        switch (prefix.len) {
            0 => {},
            1 => switch (std.ascii.toLower(prefix[0])) {
                'r' => raw = true,
                'b' => bytes = true,
                'u' => {},
                else => return null,
            },
            2 => {
                const first = std.ascii.toLower(prefix[0]);
                const second = std.ascii.toLower(prefix[1]);
                if (!((first == 'b' and second == 'r') or (first == 'r' and second == 'b'))) return null;
                raw = true;
                bytes = true;
            },
            else => return null,
        }
        const quote = self.source[quote_pos];
        const triple = self.source.len - quote_pos >= 3 and self.source[quote_pos + 1] == quote and self.source[quote_pos + 2] == quote;
        self.index = quote_pos + @as(usize, if (triple) 3 else 1);
        const content = self.index;
        while (self.at(0)) |c| {
            if (c == quote and (!triple or (self.at(1) == quote and self.at(2) == quote))) {
                const end = self.index;
                self.index += @as(usize, if (triple) 3 else 1);
                return .{ .content = self.source[content..end], .raw = raw, .bytes = bytes };
            }
            if (c == '\\') {
                self.index += 1;
                const escaped = self.at(0) orelse return error.ParseValue;
                if (escaped == '\n' or escaped == '\r') self.newline() else self.index += 1;
            } else if (c == '\n' or c == '\r') {
                if (!triple) return error.ParseValue;
                self.newline();
            } else self.index += 1;
        }
        return error.ParseValue;
    }

    fn digits(self: *Lexer, base: u8) Error!void {
        if (digit(self.at(0) orelse return error.ParseValue, base) == null) return error.ParseValue;
        self.index += 1;
        while (self.at(0)) |c| {
            if (digit(c, base) != null) {
                self.index += 1;
            } else if (c == '_') {
                self.index += 1;
                if (digit(self.at(0) orelse return error.ParseValue, base) == null) return error.ParseValue;
                self.index += 1;
            } else break;
        }
    }

    fn number(self: *Lexer) Error!void {
        if (self.at(0) == '0') {
            const base: u8 = switch (self.at(1) orelse 0) {
                'b', 'B' => 2,
                'o', 'O' => 8,
                'x', 'X' => 16,
                else => 10,
            };
            if (base != 10) {
                self.index += 2;
                if (self.at(0) == '_') self.index += 1;
                return self.digits(base);
            }
        }
        if (self.at(0) == '.') {
            self.index += 1;
            try self.digits(10);
        } else {
            try self.digits(10);
            if (self.at(0) == '.') {
                self.index += 1;
                if (decimalDigit(self.at(0))) try self.digits(10) else if (self.at(0) == '_') return error.ParseValue;
            }
        }
        if (self.at(0) == 'e' or self.at(0) == 'E') {
            const exponent = self.index;
            self.index += 1;
            if (self.at(0) == '+' or self.at(0) == '-') {
                self.index += 1;
                try self.digits(10);
            } else if (decimalDigit(self.at(0))) {
                try self.digits(10);
            } else {
                // In 3.11 `1else ...` still tokenizes with a warning, unlike
                // an invalid identifier suffix. literal_eval rejects the
                // subsequent AST, but conversion-limit precedence remains.
                self.index = exponent;
                return;
            }
        }
        if (self.at(0) == 'j' or self.at(0) == 'J') self.index += 1;
    }
};

// Grouping does not change a Python AST node's kind. Containers, calls, and a
// converted BinOp cannot masquerade as Constant in a surrounding unary/BinOp.
const Node = struct {
    const Kind = union(enum) {
        constant: Value,
        signed: Value,
        converted: Value,
        set_name,
        unsupported,
        // Defer semantic failures until the whole expression's syntax and AST
        // shape are known: `+{[1]}` is ValueError, not the set's TypeError.
        evaluation_failed,
        recursion_failed,
    };

    kind: Kind,
    // Height of this expression's AST, not the number of sibling nodes or
    // parentheses. No allocated AST or native frame per unary operator.
    depth: usize,

    fn init(kind: Kind, depth: usize) Node {
        return .{ .kind = if (depth > max_ast_depth) .recursion_failed else kind, .depth = depth };
    }

    fn value(self: Node) Error!Value {
        return switch (self.kind) {
            .constant, .signed, .converted => |v| v,
            .set_name, .unsupported => error.ParseValue,
            .evaluation_failed => error.LiteralEvaluation,
            .recursion_failed => error.JsonTooDeep,
        };
    }

    fn valueOrFailure(self: Node, failure: *Failure) ?Value {
        const next: Failure = switch (self.kind) {
            .constant, .signed, .converted => |v| return v,
            .set_name, .unsupported => .unsupported,
            .evaluation_failed => .evaluation,
            .recursion_failed => .recursion,
        };
        failure.record(next);
        return null;
    }

    fn number(self: Node, allow_signed: bool) ?Value {
        const v = switch (self.kind) {
            .constant => |v| v,
            .signed => |v| if (allow_signed) v else return null,
            else => return null,
        };
        if (v == .integer or v == .number_string or v == .float or complexParts(v) != null) return v;
        return null;
    }
};

const Failure = enum {
    none,
    unsupported,
    evaluation,
    recursion,

    fn record(self: *Failure, next: Failure) void {
        // Python constructs the complete AST before literal evaluation, so
        // resource refusal takes precedence over any deferred semantic failure.
        if (self.* == .none or next == .recursion) self.* = next;
    }

    fn kind(self: Failure) Node.Kind {
        return switch (self) {
            .none => unreachable,
            .unsupported => .unsupported,
            .evaluation => .evaluation_failed,
            .recursion => .recursion_failed,
        };
    }
};

const Parser = struct {
    a: Allocator,
    lexer: Lexer,
    token: Token = .eof,

    fn advance(self: *Parser) Error!void {
        self.token = try self.lexer.next();
    }

    fn take(self: *Parser, comptime tag: std.meta.Tag(Token)) Error!void {
        if (self.token != tag) return error.ParseValue;
        try self.advance();
    }

    fn expression(self: *Parser) Error!Node {
        const left = try self.signed();
        if (self.token != .plus and self.token != .minus) return left;
        const subtract = self.token == .minus;
        try self.advance();
        const right = try self.signed();
        const depth = @max(left.depth, right.depth) + 1;
        if (depth > max_ast_depth) return Node.init(.recursion_failed, depth);
        const left_number = left.number(true) orelse return Node.init(.unsupported, depth);
        const right_number = right.number(false) orelse return Node.init(.unsupported, depth);
        const imag = complexParts(right_number) orelse return Node.init(.unsupported, depth);
        // Both AST/type checks precede integer-to-complex conversion: a huge
        // integer plus an ordinary integer is ValueError, not OverflowError.
        const real = realNumber(left_number) catch |err| switch (err) {
            error.ParseValue => return Node.init(.unsupported, depth),
            error.LiteralEvaluation => return Node.init(.evaluation_failed, depth),
            else => return err,
        };
        const real_result: f64 = if (subtract) real - imag.real else real + imag.real;
        const imag_result: f64 = if (subtract) @as(f64, 0.0) - imag.imag else @as(f64, 0.0) + imag.imag;
        return Node.init(.{ .converted = try python.complex(self.a, real_result, imag_result) }, depth);
    }

    fn signed(self: *Parser) Error!Node {
        const operator = self.token;
        var count: usize = 0;
        while (self.token == .plus or self.token == .minus or self.token == .invert) {
            count += 1;
            try self.advance();
        }
        // Parse the operand even after exhausting the native budget. Missing
        // operands, unmatched delimiters and trailing garbage remain syntax
        // failures, rather than being mistaken for an uncaught RecursionError.
        const operand = try self.primary();
        if (count == 0) return operand;
        const depth = operand.depth + count;
        if (depth > max_ast_depth) return Node.init(.recursion_failed, depth);
        // literal_eval admits exactly one USub/UAdd over a numeric Constant.
        // More prefixes and Invert are valid Python ASTs, not valid literals.
        if (count != 1 or operator == .invert) return Node.init(.unsupported, depth);
        const v = operand.number(false) orelse return Node.init(.unsupported, depth);
        if (operator == .plus) return Node.init(.{ .signed = v }, depth);
        return Node.init(.{ .signed = switch (v) {
            .integer => |n| .{ .integer = -n },
            .float => |n| .{ .float = -n },
            .number_string => |n| blk: {
                if (std.mem.eql(u8, n, "9223372036854775808")) break :blk .{ .integer = std.math.minInt(i64) };
                const text = try self.a.alloc(u8, n.len + 1);
                text[0] = '-';
                @memcpy(text[1..], n);
                break :blk .{ .number_string = text };
            },
            .custom => blk: {
                const parts = complexParts(v).?;
                break :blk try python.complex(self.a, -parts.real, -parts.imag);
            },
            else => unreachable,
        } }, depth);
    }

    fn primary(self: *Parser) Error!Node {
        const node = switch (self.token) {
            .number => |text| blk: {
                const v = try self.number(text);
                try self.advance();
                break :blk Node.init(.{ .constant = v }, 1);
            },
            .string => Node.init(.{ .constant = try self.strings() }, 1),
            .ellipsis => blk: {
                try self.advance();
                break :blk Node.init(.{ .constant = try python.ellipsis(self.a) }, 1);
            },
            .name => |name| blk: {
                const result = if (std.mem.eql(u8, name, "None")) Node.init(.{ .constant = .null }, 1) else if (std.mem.eql(u8, name, "True"))
                    Node.init(.{ .constant = .{ .bool = true } }, 1)
                else if (std.mem.eql(u8, name, "False"))
                    Node.init(.{ .constant = .{ .bool = false } }, 1)
                else if (isSetName(name))
                    Node.init(.set_name, 1)
                else
                    return error.ParseValue;
                try self.advance();
                break :blk result;
            },
            .lparen => blk: {
                try self.advance();
                if (self.token == .rparen) {
                    try self.advance();
                    break :blk Node.init(.{ .converted = try python.tuple(self.a, &.{}) }, 1);
                }
                const first = try self.expression();
                const result = if (self.token == .comma) try self.tupleTail(first, .rparen) else first;
                try self.take(.rparen);
                break :blk result;
            },
            .lbracket => try self.list(),
            .lbrace => try self.braces(),
            else => return error.ParseValue,
        };
        if (node.kind == .set_name and self.token == .lparen) {
            try self.advance();
            try self.take(.rparen);
            return Node.init(.{ .converted = try python.set(self.a, &.{}) }, node.depth + 1);
        }
        return node;
    }

    fn tupleTail(self: *Parser, first: Node, comptime close: std.meta.Tag(Token)) Error!Node {
        var items: std.ArrayList(Value) = .empty;
        var failure: Failure = .none;
        var child_depth = first.depth;
        if (first.valueOrFailure(&failure)) |v| try items.append(self.a, v);
        while (self.token == .comma) {
            try self.advance();
            if (self.token == close or (close == .eof and self.token == .newline)) break;
            const node = try self.expression();
            child_depth = @max(child_depth, node.depth);
            if (node.valueOrFailure(&failure)) |v| try items.append(self.a, v);
        }
        const depth = child_depth + 1;
        if (failure != .none) return Node.init(failure.kind(), depth);
        return Node.init(.{ .converted = try python.tuple(self.a, items.items) }, depth);
    }

    fn list(self: *Parser) Error!Node {
        try self.take(.lbracket);
        var items: runtime.Array = .init(self.a);
        var failure: Failure = .none;
        var child_depth: usize = 0;
        while (self.token != .rbracket) {
            const node = try self.expression();
            child_depth = @max(child_depth, node.depth);
            if (node.valueOrFailure(&failure)) |v| try items.append(v);
            if (self.token != .comma) break;
            try self.advance();
        }
        try self.take(.rbracket);
        const depth = child_depth + 1;
        if (failure != .none) return Node.init(failure.kind(), depth);
        return Node.init(.{ .converted = .{ .array = items } }, depth);
    }

    fn braces(self: *Parser) Error!Node {
        try self.take(.lbrace);
        if (self.token == .rbrace) {
            try self.advance();
            return Node.init(.{ .converted = try python.dictionary(self.a, &.{}) }, 1);
        }
        const first = try self.expression();
        var failure: Failure = .none;
        var child_depth = first.depth;
        if (self.token == .colon) {
            var entries: std.ArrayList(runtime.PythonEntry) = .empty;
            var key = first.valueOrFailure(&failure);
            while (true) {
                try self.take(.colon);
                const node = try self.expression();
                child_depth = @max(child_depth, node.depth);
                const value = node.valueOrFailure(&failure);
                if (key) |k| {
                    if (value) |v| try entries.append(self.a, .{ .key = k, .value = v });
                }
                if (self.token != .comma) break;
                try self.advance();
                if (self.token == .rbrace) break;
                const next = try self.expression();
                child_depth = @max(child_depth, next.depth);
                key = next.valueOrFailure(&failure);
            }
            try self.take(.rbrace);
            const depth = child_depth + 1;
            if (failure != .none) return Node.init(failure.kind(), depth);
            const value = python.dictionary(self.a, entries.items) catch |err| switch (err) {
                error.ParseValue => return Node.init(.evaluation_failed, depth),
                error.OutOfMemory => return error.OutOfMemory,
            };
            return Node.init(.{ .converted = value }, depth);
        }
        var items: std.ArrayList(Value) = .empty;
        if (first.valueOrFailure(&failure)) |v| try items.append(self.a, v);
        while (self.token == .comma) {
            try self.advance();
            if (self.token == .rbrace) break;
            const node = try self.expression();
            child_depth = @max(child_depth, node.depth);
            if (node.valueOrFailure(&failure)) |v| try items.append(self.a, v);
        }
        try self.take(.rbrace);
        const depth = child_depth + 1;
        if (failure != .none) return Node.init(failure.kind(), depth);
        const value = python.set(self.a, items.items) catch |err| switch (err) {
            error.ParseValue => return Node.init(.evaluation_failed, depth),
            error.OutOfMemory => return error.OutOfMemory,
        };
        return Node.init(.{ .converted = value }, depth);
    }

    fn number(self: *Parser, token: []const u8) Error!Value {
        var text = token;
        const imaginary = text[text.len - 1] == 'j' or text[text.len - 1] == 'J';
        if (imaginary) text = text[0 .. text.len - 1];
        const base: u8 = if (text.len >= 2 and text[0] == '0') switch (text[1]) {
            'b', 'B' => 2,
            'o', 'O' => 8,
            'x', 'X' => 16,
            else => 10,
        } else 10;
        if (base == 10 and (imaginary or std.mem.indexOfAny(u8, text, ".eE") != null)) {
            const n = std.fmt.parseFloat(f64, text) catch return error.ParseValue;
            return if (imaginary) python.complex(self.a, 0.0, n) else .{ .float = n };
        }
        if (imaginary) return error.ParseValue;
        var magnitude = if (base != 10) text[2..] else text;
        if (base != 10 and magnitude[0] == '_') magnitude = magnitude[1..];
        var decimal_digits: usize = 0;
        if (base == 10) {
            for (magnitude) |c| {
                if (c == '_') continue;
                decimal_digits += 1;
                if (magnitude[0] == '0' and c != '0') return error.ParseValue;
            }
            if (magnitude[0] == '0') return .{ .integer = 0 };
            if (decimal_digits > max_int_digits) return error.IntegerTooLong;
        }
        // The lexer already enforced Python's underscore grammar. Zig's
        // numeric parsers admit those internal underscores without a copy.
        if (std.fmt.parseInt(i64, magnitude, base)) |n| return .{ .integer = n } else |err| switch (err) {
            error.Overflow => {},
            else => return error.ParseValue,
        }
        if (base != 10) return .{ .number_string = try baseToDecimal(self.a, magnitude, base) };
        if (decimal_digits == magnitude.len) return .{ .number_string = magnitude };
        const canonical = try self.a.alloc(u8, decimal_digits);
        var index: usize = 0;
        for (magnitude) |c| if (c != '_') {
            canonical[index] = c;
            index += 1;
        };
        return .{ .number_string = canonical };
    }

    fn strings(self: *Parser) Error!Value {
        const first = self.token.string;
        const bytes = first.bytes;
        try self.advance();
        if (self.token != .string and plainString(first)) {
            if (bytes) for (first.content) |c| {
                if (c >= 0x80) return error.ParseValue;
            };
            return if (bytes) python.bytes(self.a, first.content) else .{ .string = first.content };
        }
        var data: std.ArrayList(u8) = .empty;
        try decodeString(self.a, &data, first);
        while (self.token == .string) {
            const part = self.token.string;
            if (part.bytes != bytes) return error.ParseValue;
            try decodeString(self.a, &data, part);
            try self.advance();
        }
        return if (bytes) python.bytes(self.a, data.items) else .{ .string = data.items };
    }
};

fn decimalDigit(c: ?u8) bool {
    return if (c) |value| value >= '0' and value <= '9' else false;
}

fn digit(c: u8, base: u8) ?u8 {
    const n: u8 = switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => return null,
    };
    return if (n < base) n else null;
}

fn identifierByte(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_' or c >= 0x80;
}

fn numericKeywordSuffix(text: []const u8) bool {
    // tokenizer.c verify_end_of_number: these legacy keyword adjacencies
    // produce warnings, not lexical errors, in Python 3.11. They do not
    // become executable expressions: the literal parser still rejects them.
    for ([_][]const u8{ "and", "else", "for", "not", "or" }) |keyword| {
        if (std.mem.startsWith(u8, text, keyword) and
            (text.len == keyword.len or
                (!identifierByte(text[keyword.len]) and !std.ascii.isDigit(text[keyword.len])))) return true;
    }
    // CPython's special i-lookahead intentionally has no identifier-boundary
    // test after these two bytes; reproduce that tokenization, not a fix.
    return text.len >= 2 and text[0] == 'i' and
        (text[1] == 'f' or text[1] == 'n' or text[1] == 's');
}

const Complex = struct { real: f64, imag: f64 };

fn complexParts(v: Value) ?Complex {
    const builtin = python.view(v) orelse return null;
    return switch (builtin) {
        .complex => |n| .{ .real = n.real, .imag = n.imag },
        else => null,
    };
}

fn realNumber(v: Value) Error!f64 {
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        .number_string => |text| blk: {
            const n = std.fmt.parseFloat(f64, text) catch return error.ParseValue;
            // Unlike an overflowing *float token*, int-to-double conversion
            // used by Python's complex arithmetic raises OverflowError.
            if (!std.math.isFinite(n)) return error.LiteralEvaluation;
            break :blk n;
        },
        else => error.ParseValue,
    };
}

fn baseToDecimal(a: Allocator, text: []const u8, base: u8) Allocator.Error![]const u8 {
    // Little-endian base-1e9 limbs: a digit multiply fits comfortably in u64.
    // There is deliberately no decimal conversion limit for power-of-two bases.
    const limb_base: u64 = 1_000_000_000;
    var limbs: std.ArrayList(u32) = .empty;
    try limbs.append(a, 0);
    for (text) |c| {
        if (c == '_') continue;
        var carry: u64 = digit(c, base).?;
        for (limbs.items) |*limb| {
            carry += @as(u64, limb.*) * base;
            limb.* = @intCast(carry % limb_base);
            carry /= limb_base;
        }
        if (carry != 0) try limbs.append(a, @intCast(carry));
    }
    var output: std.ArrayList(u8) = .empty;
    try output.ensureTotalCapacity(a, limbs.items.len * 9);
    var buffer: [10]u8 = undefined;
    output.appendSliceAssumeCapacity(std.fmt.bufPrint(&buffer, "{d}", .{limbs.items[limbs.items.len - 1]}) catch unreachable);
    var i = limbs.items.len - 1;
    while (i != 0) {
        i -= 1;
        output.appendSliceAssumeCapacity(std.fmt.bufPrint(&buffer, "{d:0>9}", .{limbs.items[i]}) catch unreachable);
    }
    return output.items;
}

fn appendCodepoint(a: Allocator, out: *std.ArrayList(u8), cp: u32) Allocator.Error!void {
    // std.unicode.utf8Encode correctly rejects surrogates; Python strings do
    // not, so use the lossless WTF-8 convention shared by the JSON decoder.
    if (cp < 0x80) {
        try out.append(a, @intCast(cp));
    } else if (cp < 0x800) {
        try out.appendSlice(a, &.{ @intCast(0xc0 | (cp >> 6)), @intCast(0x80 | (cp & 0x3f)) });
    } else if (cp < 0x10000) {
        try out.appendSlice(a, &.{ @intCast(0xe0 | (cp >> 12)), @intCast(0x80 | ((cp >> 6) & 0x3f)), @intCast(0x80 | (cp & 0x3f)) });
    } else {
        try out.appendSlice(a, &.{ @intCast(0xf0 | (cp >> 18)), @intCast(0x80 | ((cp >> 12) & 0x3f)), @intCast(0x80 | ((cp >> 6) & 0x3f)), @intCast(0x80 | (cp & 0x3f)) });
    }
}

fn hexEscape(text: []const u8, index: *usize, count: usize) Error!u32 {
    if (text.len - index.* < count) return error.ParseValue;
    var cp: u32 = 0;
    for (text[index.*..][0..count]) |c| cp = cp * 16 + (digit(c, 16) orelse return error.ParseValue);
    index.* += count;
    if (cp > 0x10ffff) return error.ParseValue;
    return cp;
}

fn plainString(token: StringToken) bool {
    return std.mem.indexOfScalar(u8, token.content, '\r') == null and
        (token.raw or std.mem.indexOfScalar(u8, token.content, '\\') == null);
}

fn decodeString(a: Allocator, out: *std.ArrayList(u8), token: StringToken) Error!void {
    const text = token.content;
    if (token.bytes) for (text) |c| {
        if (c >= 0x80) return error.ParseValue;
    };
    if (plainString(token)) return out.appendSlice(a, text);
    var index: usize = 0;
    while (index < text.len) {
        const c = text[index];
        index += 1;
        if (c == '\r') {
            if (index < text.len and text[index] == '\n') index += 1;
            try out.append(a, '\n');
        } else if (c != '\\' or token.raw) {
            try out.append(a, c);
        } else {
            if (index == text.len) return error.ParseValue;
            const escaped = text[index];
            index += 1;
            switch (escaped) {
                '\n' => {},
                '\r' => if (index < text.len and text[index] == '\n') {
                    index += 1;
                },
                '\\', '\'', '"' => try out.append(a, escaped),
                'a' => try out.append(a, 7),
                'b' => try out.append(a, 8),
                'f' => try out.append(a, 12),
                'n' => try out.append(a, 10),
                'r' => try out.append(a, 13),
                't' => try out.append(a, 9),
                'v' => try out.append(a, 11),
                '0'...'7' => {
                    var cp: u32 = escaped - '0';
                    var count: usize = 1;
                    while (count < 3 and index < text.len and digit(text[index], 8) != null) : (count += 1) {
                        cp = cp * 8 + digit(text[index], 8).?;
                        index += 1;
                    }
                    if (token.bytes) try out.append(a, @truncate(cp)) else try appendCodepoint(a, out, cp);
                },
                'x' => try appendCodepointOrByte(a, out, try hexEscape(text, &index, 2), token.bytes),
                'u', 'U' => {
                    if (token.bytes) {
                        try out.appendSlice(a, &.{ '\\', escaped });
                    } else try appendCodepoint(a, out, try hexEscape(text, &index, if (escaped == 'u') 4 else 8));
                },
                'N' => {
                    if (token.bytes) {
                        try out.appendSlice(a, "\\N");
                    } else {
                        if (index == text.len or text[index] != '{') return error.ParseValue;
                        index += 1;
                        const start = index;
                        while (index < text.len and text[index] != '}') index += 1;
                        if (index == text.len) return error.ParseValue;
                        const cp = unicodeName(text[start..index]) orelse return error.ParseValue;
                        index += 1;
                        try appendCodepoint(a, out, cp);
                    }
                },
                else => try out.appendSlice(a, &.{ '\\', escaped }),
            }
        }
    }
}

fn appendCodepointOrByte(a: Allocator, out: *std.ArrayList(u8), cp: u32, bytes: bool) Allocator.Error!void {
    if (bytes) try out.append(a, @intCast(cp)) else try appendCodepoint(a, out, cp);
}

// Generated from pinned CPython, not host/locale data. Header: magic[8], three
// little-endian u32 counts; name records (cp, pool_offset, length); NFKC records
// (cp, packed ASCII); CJK ranges (first,last); then the ASCII name pool. Explicit
// lengths and complete name comparisons avoid probabilistic hash-only lookup.
const unicode_data = @embedFile("python_literal_unicode.bin");
const name_count = dataWord(8);
const set_component_count = dataWord(12);
const cjk_range_count = dataWord(16);
const set_components_start = 20 + name_count * 12;
const cjk_ranges_start = set_components_start + set_component_count * 8;
const name_pool_start = cjk_ranges_start + cjk_range_count * 8;

fn dataWord(offset: usize) u32 {
    return std.mem.readInt(u32, unicode_data[offset..][0..4], .little);
}

fn isSetName(name: []const u8) bool {
    if (std.mem.eql(u8, name, "set")) return true;
    var source_index: usize = 0;
    var target_index: usize = 0;
    const target = "set";
    while (source_index < name.len) {
        const n = std.unicode.utf8ByteSequenceLength(name[source_index]) catch return false;
        if (name.len - source_index < n) return false;
        const cp = std.unicode.utf8Decode(name[source_index..][0..n]) catch return false;
        source_index += n;
        var low: usize = 0;
        var high: usize = set_component_count;
        var component: u32 = 0;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const entry = set_components_start + middle * 8;
            const candidate = dataWord(entry);
            if (cp < candidate) {
                high = middle;
            } else if (cp > candidate) {
                low = middle + 1;
            } else {
                component = dataWord(entry + 4);
                break;
            }
        }
        if (component == 0) return false;
        while (component != 0) {
            if (target_index == target.len or @as(u8, @truncate(component)) != target[target_index]) return false;
            target_index += 1;
            component >>= 8;
        }
    }
    return target_index == target.len;
}

fn compareName(input: []const u8, stored: []const u8) std.math.Order {
    for (input[0..@min(input.len, stored.len)], stored[0..@min(input.len, stored.len)]) |a, b| {
        const folded = std.ascii.toUpper(a);
        if (folded < b) return .lt;
        if (folded > b) return .gt;
    }
    return std.math.order(input.len, stored.len);
}

fn unicodeName(name: []const u8) ?u32 {
    if (std.mem.startsWith(u8, name, "CJK UNIFIED IDEOGRAPH-")) {
        const suffix = name[22..];
        if (suffix.len != 4 and suffix.len != 5) return null;
        var cp: u32 = 0;
        for (suffix) |c| {
            if (!std.ascii.isDigit(c) and !(c >= 'A' and c <= 'F')) return null;
            cp = cp * 16 + digit(c, 16).?;
        }
        for (0..cjk_range_count) |i| {
            const offset = cjk_ranges_start + i * 8;
            if (cp >= dataWord(offset) and cp <= dataWord(offset + 4)) return cp;
        }
        return null;
    }
    if (std.mem.startsWith(u8, name, "HANGUL SYLLABLE ")) {
        var index: usize = 16;
        const l = hangulPart(name, &index, &hangul_l) orelse return null;
        const v = hangulPart(name, &index, &hangul_v) orelse return null;
        const t = hangulPart(name, &index, &hangul_t) orelse return null;
        if (index != name.len) return null;
        return @as(u32, @intCast(0xac00 + (l * 21 + v) * 28 + t));
    }
    var low: usize = 0;
    var high: usize = name_count;
    while (low < high) {
        const middle = low + (high - low) / 2;
        const entry = 20 + middle * 12;
        const offset = name_pool_start + dataWord(entry + 4);
        const stored = unicode_data[offset..][0..dataWord(entry + 8)];
        switch (compareName(name, stored)) {
            .lt => high = middle,
            .gt => low = middle + 1,
            .eq => return dataWord(entry),
        }
    }
    return null;
}

const hangul_l = [_][]const u8{ "G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "", "J", "JJ", "C", "K", "T", "P", "H" };
const hangul_v = [_][]const u8{ "A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "YI", "I" };
const hangul_t = [_][]const u8{ "", "G", "GG", "GS", "N", "NJ", "NH", "D", "L", "LG", "LM", "LB", "LS", "LT", "LP", "LH", "M", "B", "BS", "S", "SS", "NG", "J", "C", "K", "T", "P", "H" };

fn hangulPart(name: []const u8, index: *usize, choices: []const []const u8) ?usize {
    var best: ?usize = null;
    var length: usize = 0;
    for (choices, 0..) |choice, i| {
        if ((best == null or choice.len > length) and std.mem.startsWith(u8, name[index.*..], choice)) {
            best = i;
            length = choice.len;
        }
    }
    index.* += length;
    return best;
}

// Expected values below were observed independently with the pinned Python
// oracle. These tests are intentionally not JSON round trips: each assertion
// examines the native builtin kind or exact integer/binary64/string payload.
test "Python literals preserve exact integers across bases and underscores" {
    const testing = std.testing;
    const cases = [_]struct { source: []const u8, expected: i64 }{
        .{ .source = "0_00_0", .expected = 0 },
        .{ .source = "0X_FF", .expected = 255 },
        .{ .source = "0b_0101", .expected = 5 },
        .{ .source = "0O12_34", .expected = 668 },
        .{ .source = "+1_000", .expected = 1000 },
        .{ .source = "9223372036854775807", .expected = std.math.maxInt(i64) },
        .{ .source = "-9223372036854775808", .expected = std.math.minInt(i64) },
    };
    for (cases) |case| {
        const result = try parse(testing.allocator, case.source);
        defer inference.freeJsonValue(testing.allocator, result);
        try testing.expect(result == .integer);
        try testing.expectEqual(case.expected, result.integer);
    }
    const wide = [_]struct { source: []const u8, expected: []const u8 }{
        .{ .source = "9223372036854775808", .expected = "9223372036854775808" },
        .{ .source = "-9223372036854775809", .expected = "-9223372036854775809" },
        .{ .source = "0x8000000000000000", .expected = "9223372036854775808" },
        .{ .source = "0x10000000000000000", .expected = "18446744073709551616" },
        .{ .source = "0o2000000000000000000000", .expected = "18446744073709551616" },
        .{ .source = "0b1_0000000000000000000000000000000000000000000000000000000000000000", .expected = "18446744073709551616" },
    };
    for (wide) |case| {
        const result = try parse(testing.allocator, case.source);
        defer inference.freeJsonValue(testing.allocator, result);
        try testing.expect(result == .number_string);
        try testing.expectEqualStrings(case.expected, result.number_string);
    }
}

test "Python literal floats and complex arithmetic retain Source binary64 bits" {
    const testing = std.testing;
    const floats = [_]struct { source: []const u8, bits: u64 }{
        .{ .source = "0.", .bits = 0 },
        .{ .source = "+.5", .bits = 0x3fe0000000000000 },
        .{ .source = "-0.0", .bits = 0x8000000000000000 },
        .{ .source = "-1e-9999", .bits = 0x8000000000000000 },
        .{ .source = "1.7976931348623157e308", .bits = 0x7fefffffffffffff },
        .{ .source = "1e309", .bits = 0x7ff0000000000000 },
        .{ .source = "5e-324", .bits = 1 },
        .{ .source = "2.2250738585072014e-308", .bits = 0x0010000000000000 },
        .{ .source = "2.4703282292062327e-324", .bits = 0 },
    };
    for (floats) |case| {
        const result = try parse(testing.allocator, case.source);
        defer inference.freeJsonValue(testing.allocator, result);
        try testing.expect(result == .float);
        try testing.expectEqual(case.bits, @as(u64, @bitCast(result.float)));
    }
    const complexes = [_]struct { source: []const u8, real: u64, imag: u64 }{
        .{ .source = "-0j", .real = 0x8000000000000000, .imag = 0x8000000000000000 },
        .{ .source = "-0.0-0j", .real = 0x8000000000000000, .imag = 0 },
        .{ .source = "1-0j", .real = 0x3ff0000000000000, .imag = 0 },
        .{ .source = "1e999+1e999j", .real = 0x7ff0000000000000, .imag = 0x7ff0000000000000 },
    };
    for (complexes) |case| {
        const result = try parse(testing.allocator, case.source);
        defer inference.freeJsonValue(testing.allocator, result);
        const parts = complexParts(result) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(case.real, @as(u64, @bitCast(parts.real)));
        try testing.expectEqual(case.imag, @as(u64, @bitCast(parts.imag)));
    }
}

test "Python literal strings decode raw adjacent triple named and surrogate forms" {
    const testing = std.testing;
    const cases = [_]struct { source: []const u8, expected: []const u8 }{
        .{ .source = "'a' 'b'", .expected = "ab" },
        .{ .source = "r'a\\n' 'b'", .expected = "a\\nb" },
        .{ .source = "'''a\rb\r\nc'''", .expected = "a\nb\nc" },
        .{ .source = "'a'\\\n'b'", .expected = "ab" },
        .{ .source = "'\\N{CJK UNIFIED IDEOGRAPH-03400}\\N{HANGUL SYLLABLE A}\\N{LF}'", .expected = "\u{3400}\u{c544}\n" },
        .{ .source = "'\\ud800\\udfff'", .expected = "\xed\xa0\x80\xed\xbf\xbf" },
    };
    for (cases) |case| {
        const result = try parse(testing.allocator, case.source);
        defer inference.freeJsonValue(testing.allocator, result);
        try testing.expect(result == .string);
        try testing.expectEqualStrings(case.expected, result.string);
    }
    const bytes_result = try parse(testing.allocator, "b'\\0\\377\\400\\777'");
    defer inference.freeJsonValue(testing.allocator, bytes_result);
    const builtin = python.view(bytes_result) orelse return error.TestUnexpectedResult;
    try testing.expect(builtin == .bytes);
    try testing.expectEqualStrings("\x00\xff\x00\xff", builtin.bytes);
}

test "Python literal containers preserve builtin kinds deduplication and key types" {
    const testing = std.testing;
    const result = try parse(testing.allocator, "(True, b'\\x41', {1, True, 1.0, 1+0j}, {2:'x'}, ..., set())");
    defer inference.freeJsonValue(testing.allocator, result);
    const builtin = python.view(result) orelse return error.TestUnexpectedResult;
    try testing.expect(builtin == .tuple);
    try testing.expectEqual(@as(usize, 6), builtin.tuple.len);
    const items = builtin.tuple;
    try testing.expect(items[0] == .bool and items[0].bool);
    const byte_value = python.view(items[1]) orelse return error.TestUnexpectedResult;
    try testing.expect(byte_value == .bytes);
    try testing.expectEqualStrings("A", byte_value.bytes);
    const unique = python.view(items[2]) orelse return error.TestUnexpectedResult;
    try testing.expect(unique == .set);
    try testing.expectEqual(@as(usize, 1), unique.set.len);
    try testing.expect(unique.set[0] == .integer and unique.set[0].integer == 1);
    const dictionary = python.view(items[3]) orelse return error.TestUnexpectedResult;
    try testing.expect(dictionary == .dictionary);
    try testing.expectEqual(@as(usize, 1), dictionary.dictionary.len);
    try testing.expectEqual(@as(i64, 2), dictionary.dictionary[0].key.integer);
    try testing.expectEqualStrings("x", dictionary.dictionary[0].value.string);
    const ellipsis_value = python.view(items[4]) orelse return error.TestUnexpectedResult;
    try testing.expect(ellipsis_value == .ellipsis);
    const empty_set = python.view(items[5]) orelse return error.TestUnexpectedResult;
    try testing.expect(empty_set == .set);
    try testing.expectEqual(@as(usize, 0), empty_set.set.len);

    const object = try parse(testing.allocator, "{'a': 1, 'b': 2, 'a': 3}");
    defer inference.freeJsonValue(testing.allocator, object);
    try testing.expect(object == .object);
    try testing.expectEqual(@as(usize, 2), object.object.count());
    try testing.expectEqualStrings("a", object.object.keys()[0]);
    try testing.expectEqualStrings("b", object.object.keys()[1]);
    try testing.expectEqual(@as(i64, 3), object.object.get("a").?.integer);
}

test "Python empty set calls accept parenthesized and normalized identifiers" {
    const testing = std.testing;
    for ([_][]const u8{ "set()", "(set)()", "ｓｅｔ()", "𝓼𝓮𝓽()" }) |source| {
        const result = try parse(testing.allocator, source);
        defer inference.freeJsonValue(testing.allocator, result);
        const builtin = python.view(result) orelse return error.TestUnexpectedResult;
        try testing.expect(builtin == .set);
        try testing.expectEqual(@as(usize, 0), builtin.set.len);
    }
}

test "Python literal parse results own source and nested builtin payloads" {
    const testing = std.testing;
    const source = try testing.allocator.dupe(u8, "(b'payload', {'key': ['value', (1+2j,)]})");
    defer testing.allocator.free(source);
    const result = try parse(testing.allocator, source);
    defer inference.freeJsonValue(testing.allocator, result);
    @memset(source, '?');
    try testing.expect(result == .custom and result.custom.owns_data);
    const items = python.view(result).?.tuple;
    try testing.expect(items[0] == .custom and items[0].custom.owns_data);
    try testing.expectEqualStrings("payload", python.view(items[0]).?.bytes);
    const children = items[1].object.get("key").?.array.items;
    try testing.expectEqualStrings("value", children[0].string);
    const nested = python.view(children[1]).?.tuple;
    const parts = complexParts(nested[0]) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(f64, 1), parts.real);
    try testing.expectEqual(@as(f64, 2), parts.imag);
}

test "Python literals reject executable and malformed AST shapes without repair" {
    const testing = std.testing;
    for ([_][]const u8{
        "07", "1__2", "1+-2j", "--1", "-(1+2j)", "1+2", "1j+2j",
        "1x", "1_0x", "0x1g", "0b18", "0o78", "1jz", "1e2jz",
        "+{[1]}", "1+{[1]}", "[+{[1]}]", "{[]:1} blah",
        "b'é'", "f'abc'", "[x for x in []]",
        "__import__('os')", "set(())", "Ellipsis", "\\\n 1", "1\\\n",
        "'\\N{KEYCAP DIGIT ONE}'", "'\\N{cjk unified ideograph-4E00}'",
    }) |source| try testing.expectError(error.ParseValue, parse(testing.allocator, source));
}

test "Python literal evaluation failures are not caught syntax fallbacks" {
    const testing = std.testing;
    for ([_][]const u8{ "{[]:1}", "{[1]}", "{set()}", "{(1,[]):2}", "({[]:1},)", "[{[]:1}]" }) |source| {
        try testing.expectError(error.LiteralEvaluation, parse(testing.allocator, source));
    }
    var huge: [309 + 3]u8 = @splat('9');
    @memcpy(huge[309..], "+1j");
    try testing.expectError(error.LiteralEvaluation, parse(testing.allocator, &huge));
    const finite = try parse(testing.allocator, huge[1..]);
    defer inference.freeJsonValue(testing.allocator, finite);
    const parts = complexParts(finite) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 0x7fe1ccf385ebc8a0), @as(u64, @bitCast(parts.real)));

    // A rejected outer AST never evaluates its failing BinOp operand.
    const enclosed = try std.mem.concat(testing.allocator, u8, &.{ "+(", &huge, ")" });
    defer testing.allocator.free(enclosed);
    try testing.expectError(error.ParseValue, parse(testing.allocator, enclosed));
    // Likewise a non-complex RHS fails the AST/type check before conversion.
    huge[huge.len - 1] = ' ';
    try testing.expectError(error.ParseValue, parse(testing.allocator, &huge));
}

test "Python unary AST resource budget traverses prefixes without native recursion" {
    const testing = std.testing;
    for ([_]u8{ '+', '-', '~' }) |operator| {
        var source: [max_ast_depth + 1]u8 = @splat(operator);
        // The native policy allows 3000 expression-body nodes. CPython's
        // caller-frame-dependent cutoff near this boundary is not identical.
        for ([_]usize{ 2000, max_ast_depth - 1, max_ast_depth }) |count| {
            source[count] = '1';
            try testing.expectError(
                if (count == max_ast_depth) error.JsonTooDeep else error.ParseValue,
                parse(testing.allocator, source[0 .. count + 1]),
            );
            source[count] = operator;
        }
    }
}

test "Python AST depth refusal defers to syntax and precedes literal evaluation" {
    const testing = std.testing;
    const prefix: [max_ast_depth]u8 = @splat('+');
    try testing.expectError(error.ParseValue, parse(testing.allocator, &prefix));
    for ([_]struct { head: []const u8, tail: []const u8, expected: Error }{
        .{ .head = "", .tail = "1 garbage", .expected = error.ParseValue },
        .{ .head = "[", .tail = "1", .expected = error.ParseValue },
        .{ .head = "", .tail = "1]", .expected = error.ParseValue },
        .{ .head = "[", .tail = "1,] garbage", .expected = error.ParseValue },
        .{ .head = "[", .tail = "1,]", .expected = error.JsonTooDeep },
        .{ .head = "(", .tail = "1,)", .expected = error.JsonTooDeep },
        .{ .head = "{", .tail = "1:0}", .expected = error.JsonTooDeep },
        .{ .head = "", .tail = "{[1]}", .expected = error.JsonTooDeep },
        .{ .head = "[", .tail = "1,~1]", .expected = error.JsonTooDeep },
        .{ .head = "[", .tail = "1,{[1]}]", .expected = error.JsonTooDeep },
    }) |case| {
        const source = try std.mem.concat(testing.allocator, u8, &.{ case.head, &prefix, case.tail });
        defer testing.allocator.free(source);
        try testing.expectError(case.expected, parse(testing.allocator, source));
    }
    const nested = try std.mem.concat(testing.allocator, u8, &.{ prefix[0..2000], "(", prefix[0..1000], "1)" });
    defer testing.allocator.free(nested);
    try testing.expectError(error.JsonTooDeep, parse(testing.allocator, nested));
    const siblings = try std.mem.concat(testing.allocator, u8, &.{ "[", prefix[0..2000], "1,", prefix[0..2000], "1]" });
    defer testing.allocator.free(siblings);
    try testing.expectError(error.ParseValue, parse(testing.allocator, siblings));
}

test "Python literal AST budget refusal unwinds temporary allocations" {
    const testing = std.testing;
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), struct {
        fn run(a: Allocator) !void {
            const tail = "[(b'payload',), ...]";
            var source: [max_ast_depth + tail.len]u8 = undefined;
            @memset(source[0..max_ast_depth], '+');
            @memcpy(source[max_ast_depth..], tail);
            const result = parse(a, &source) catch |err| switch (err) {
                error.JsonTooDeep => return,
                else => return err,
            };
            defer inference.freeJsonValue(a, result);
            return error.TestUnexpectedResult;
        }
    }.run, .{});
}

test "Python literal depth and decimal integer limits match the Source boundaries" {
    const testing = std.testing;
    var nested: [max_depth * 2]u8 = undefined;
    @memset(nested[0..max_depth], '[');
    @memset(nested[max_depth..], ']');
    const result = try parse(testing.allocator, &nested);
    defer inference.freeJsonValue(testing.allocator, result);
    const excessive_depth: [max_depth + 1]u8 = @splat('[');
    try testing.expectError(error.ParseValue, parse(testing.allocator, &excessive_depth));

    const ones: [max_int_digits + 1]u8 = @splat('1');
    const wide = try parse(testing.allocator, ones[0..max_int_digits]);
    defer inference.freeJsonValue(testing.allocator, wide);
    try testing.expect(wide == .number_string);
    try testing.expectEqualStrings(ones[0..max_int_digits], wide.number_string);
    try testing.expectError(error.IntegerTooLong, parse(testing.allocator, &ones));
    var invalid_suffix: [max_int_digits + 2]u8 = @splat('1');
    invalid_suffix[invalid_suffix.len - 1] = 'x';
    try testing.expectError(error.ParseValue, parse(testing.allocator, &invalid_suffix));
    const zeros: [max_int_digits + 1]u8 = @splat('0');
    const zero = try parse(testing.allocator, &zeros);
    defer inference.freeJsonValue(testing.allocator, zero);
    try testing.expect(zero == .integer and zero.integer == 0);
}

test "Python literal parsing unwinds every allocation failure" {
    const testing = std.testing;
    // Arena in-place growth otherwise makes fault sites address-dependent.
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), struct {
        fn run(a: Allocator) !void {
            const result = try parse(a, "[('x', b'abc', {1,2}), {('k',): [1+2j, ...]}, set(), 'a' 'b']");
            defer inference.freeJsonValue(a, result);
        }
    }.run, .{});
}

test "Python literal evaluation failure unwinds allocations before hashability errors" {
    const testing = std.testing;
    var stable_backing = testing.FailingAllocator.init(testing.allocator, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(stable_backing.allocator(), struct {
        fn run(a: Allocator) !void {
            const result = parse(a, "[('x', b'abc'), {(): [1+2j]}, {[]: 1}]") catch |err| switch (err) {
                error.LiteralEvaluation => return,
                else => return err,
            };
            defer inference.freeJsonValue(a, result);
            return error.TestUnexpectedResult;
        }
    }.run, .{});
}
