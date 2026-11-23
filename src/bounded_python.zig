//! A fixed-capacity interpreter for a narrow Python profile. Compilation finishes
//! before seeding globals or executing any instructions. No guest operation allocates.
//! Statements are simple assignments, indented `while` suites, `print(expr)`,
//! `llm_query(expr)`, and terminal `SUBMIT(name=expr, ...)`. Expressions support
//! scalar literals, names, grouping, calls to `llm_query`, unary `+`/`-`, and
//! `+ - * // % == != < <= > >=` without comparison chaining. Arithmetic is
//! checked i64 arithmetic (booleans are 0/1); no containers or general calls.
const std = @import("std");

pub const source_bytes_max = 8192;
pub const bindings_max = 64;
pub const name_max = 64;
pub const string_max = 4096;
pub const stack_max = 64;
pub const output_max = 8192;
pub const fuel_max: u32 = 1_000_000;

comptime {
    std.debug.assert(source_bytes_max <= std.math.maxInt(u16));
    std.debug.assert(name_max <= std.math.maxInt(u8));
    std.debug.assert(string_max <= std.math.maxInt(u16));
    std.debug.assert(string_max <= source_bytes_max);
}

pub const Value = union(enum) {
    none,
    boolean: bool,
    integer: i64,
    string: []const u8,
};
pub const Binding = struct { name: []const u8, value: Value };
pub const Host = struct {
    ptr: *anyopaque,
    // The response is borrowed only until the interpreter copies it. The host
    // must not reenter or reset this machine from the callback.
    query: *const fn (*anyopaque, []const u8) anyerror![]const u8,
};
pub const Options = struct {
    /// One unit per instruction, including host calls and loop backedges.
    /// Compilation and byte copies are bounded independently of fuel.
    fuel: u32 = 100_000,
};
pub const Result = union(enum) {
    observation: []const u8,
    submit: []const Binding,
};
pub const Error = error{
    SyntaxError,
    SourceLimit,
    BindingLimit,
    NameLimit,
    StringLimit,
    StackLimit,
    OutputLimit,
    FuelLimit,
    FuelExhausted,
    NameError,
    TypeError,
    IntegerOverflow,
    DivisionByZero,
    HostUnavailable,
};

const OwnedBinding = struct {
    name: [name_max]u8,
    name_len: u8,
    bytes: [string_max]u8,
    value: union(enum) { none, boolean: bool, integer: i64, string: u16 },

    fn set_name(self: *OwnedBinding, name: []const u8) void {
        @memmove(self.name[0..name.len], name);
        self.name_len = @intCast(name.len);
    }

    fn set_value(self: *OwnedBinding, value: Value) Error!void {
        self.value = switch (value) {
            .none => .none,
            .boolean => |bit| .{ .boolean = bit },
            .integer => |number| .{ .integer = number },
            .string => |bytes| blk: {
                if (bytes.len > string_max) return error.StringLimit;
                @memmove(self.bytes[0..bytes.len], bytes);
                break :blk .{ .string = @intCast(bytes.len) };
            },
        };
    }

    fn get_name(self: *const OwnedBinding) []const u8 {
        return self.name[0..self.name_len];
    }

    fn view(self: *const OwnedBinding) Binding {
        return .{
            .name = self.get_name(),
            .value = switch (self.value) {
                .none => .none,
                .boolean => |bit| .{ .boolean = bit },
                .integer => |number| .{ .integer = number },
                .string => |len| .{ .string = self.bytes[0..len] },
            },
        };
    }
};

pub const Machine = struct {
    globals: [bindings_max]OwnedBinding = undefined,
    globals_len: usize = 0,
    seeded: bool = false,
    output: [output_max]u8 = undefined,
    submitted: [bindings_max]OwnedBinding = undefined,
    submit_views: [bindings_max]Binding = undefined,
    compiler: Compiler = .{},
    execution: Execution = .{},

    pub fn init(self: *Machine) void {
        self.reset();
    }

    pub fn reset(self: *Machine) void {
        self.globals_len = 0;
        self.seeded = false;
    }

    /// Result slices are borrowed until the next run or reset. Runtime errors
    /// preserve completed assignments; a compile error does not seed or execute.
    /// Inputs seed globals once per reset, after successful compilation and
    /// input validation. Later runs use the persisted globals instead.
    pub fn run(
        self: *Machine,
        source: []const u8,
        inputs: []const Binding,
        options: Options,
        host: ?Host,
    ) anyerror!Result {
        if (source.len > source_bytes_max) return error.SourceLimit;
        if (options.fuel > fuel_max) return error.FuelLimit;
        self.compiler.literals_len = 0;
        self.compiler.ops_len = 0;
        self.compiler.blocks_len = 0;
        @memcpy(self.compiler.source[0..source.len], source);
        try self.compiler.compile(source.len);
        if (!self.seeded) try self.seed(inputs);
        self.execution.len = 0;
        self.execution.output_len = 0;
        self.execution.submit_len = 0;
        return self.execution.run(self, &self.compiler, options.fuel, host);
    }

    fn seed(self: *Machine, inputs: []const Binding) Error!void {
        if (inputs.len > bindings_max) return error.BindingLimit;
        for (inputs, 0..) |input, index| {
            if (input.name.len > name_max) return error.NameLimit;
            if (!valid_name(input.name)) return error.SyntaxError;
            if (input.value == .string and input.value.string.len > string_max) {
                return error.StringLimit;
            }
            for (inputs[0..index]) |other| {
                if (std.mem.eql(u8, input.name, other.name)) return error.SyntaxError;
            }
        }
        for (inputs) |input| try self.store(input.name, input.value);
        self.seeded = true;
    }

    fn lookup(self: *const Machine, name: []const u8) ?usize {
        for (self.globals[0..self.globals_len], 0..) |*global, index| {
            if (std.mem.eql(u8, global.get_name(), name)) return index;
        }
        return null;
    }

    fn store(self: *Machine, name: []const u8, value: Value) Error!void {
        if (self.lookup(name)) |index| return self.globals[index].set_value(value);
        if (self.globals_len == bindings_max) return error.BindingLimit;
        const target = &self.globals[self.globals_len];
        try target.set_value(value);
        target.set_name(name);
        self.globals_len += 1;
    }
};

const Span = struct {
    start: u16 = 0,
    end: u16 = 0,

    fn slice(self: Span, bytes: []const u8) []const u8 {
        return bytes[self.start..self.end];
    }
};

const Kind = enum {
    end,
    name,
    number,
    string,
    none,
    true_,
    false_,
    while_,
    print,
    query,
    submit,
    assign,
    left,
    right,
    comma,
    colon,
    add,
    sub,
    mul,
    floor_div,
    mod,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    pos,
    neg,
};

const Token = struct {
    kind: Kind,
    span: Span = .{},
};

fn identifier_start(byte: u8) bool {
    return std.ascii.isAlphabetic(byte) or byte == '_';
}

fn identifier_byte(byte: u8) bool {
    return identifier_start(byte) or std.ascii.isDigit(byte);
}

fn name_kind(name: []const u8) Error!Kind {
    const names = .{
        .{ "None", Kind.none },     .{ "True", Kind.true_ },
        .{ "False", Kind.false_ },  .{ "while", Kind.while_ },
        .{ "print", Kind.print },   .{ "llm_query", Kind.query },
        .{ "SUBMIT", Kind.submit },
    };
    inline for (names) |pair| {
        if (std.mem.eql(u8, name, pair[0])) return pair[1];
    }
    const unsupported = [_][]const u8{
        "and",    "as",   "assert", "async",  "await",  "break",   "class",    "continue",
        "def",    "del",  "elif",   "else",   "except", "finally", "for",      "from",
        "global", "if",   "import", "in",     "is",     "lambda",  "nonlocal", "not",
        "or",     "pass", "raise",  "return", "try",    "with",    "yield",
    };
    for (unsupported) |word| if (std.mem.eql(u8, name, word)) return error.SyntaxError;
    return .name;
}

fn valid_name(name: []const u8) bool {
    if (name.len == 0 or !identifier_start(name[0])) return false;
    for (name[1..]) |byte| if (!identifier_byte(byte)) return false;
    return (name_kind(name) catch return false) == .name;
}

const Op = union(enum) {
    integer: i64,
    boolean: bool,
    none,
    string: Span,
    load: Span,
    store: Span,
    capture: Span,
    unary: Kind,
    binary: Kind,
    jump: u16,
    jump_false: u16,
    print,
    query,
    discard,
    submit,
};

const Compiler = struct {
    source: [source_bytes_max]u8 = undefined,
    literals: [source_bytes_max]u8 = undefined,
    literals_len: u16 = 0,
    // Every emitted instruction is backed by at least one source byte,
    // including the backedge of a while (whose header spans several bytes).
    ops: [source_bytes_max]Op = undefined,
    ops_len: u16 = 0,
    blocks: [stack_max]Block = undefined,
    blocks_len: usize = 0,

    const Block = struct { head: u16, exit: u16, indent: ?usize = null };

    fn compile(self: *Compiler, source_len: usize) Error!void {
        const source = self.source[0..source_len];
        if (!std.unicode.utf8ValidateSlice(source)) return error.SyntaxError;
        if (std.mem.indexOfScalar(u8, source, 0) != null) return error.SyntaxError;
        var offset: usize = 0;
        while (offset < source.len) {
            const end = std.mem.indexOfScalarPos(u8, source, offset, '\n') orelse source.len;
            const start = offset;
            offset = if (end < source.len) end + 1 else end;
            const line_end = if (end > start and source[end - 1] == '\r') end - 1 else end;
            var cursor = start;
            while (cursor < line_end and source[cursor] == ' ') cursor += 1;
            if (cursor == line_end or source[cursor] == '#') continue;
            if (source[cursor] == '\t') return error.SyntaxError;
            try self.indent(cursor - start);
            var line: Line = .{ .compiler = self, .cursor = cursor, .end = line_end };
            try line.advance();
            try line.statement();
            if (line.token.kind != .end) return error.SyntaxError;
        }
        try self.indent(0);
    }

    fn indent(self: *Compiler, width: usize) Error!void {
        if (self.blocks_len != 0) {
            const top = &self.blocks[self.blocks_len - 1];
            if (top.indent == null) {
                const parent = if (self.blocks_len == 1) 0 else self.blocks[self.blocks_len - 2].indent.?;
                if (width <= parent) return error.SyntaxError;
                top.indent = width;
                return;
            }
        }
        while (self.blocks_len != 0) {
            const top = self.blocks[self.blocks_len - 1];
            if (width == top.indent.?) return;
            if (width > top.indent.?) return error.SyntaxError;
            _ = try self.emit(.{ .jump = top.head });
            self.ops[top.exit] = .{ .jump_false = self.ops_len };
            self.blocks_len -= 1;
        }
        if (width != 0) return error.SyntaxError;
    }

    fn emit(self: *Compiler, op: Op) Error!u16 {
        if (self.ops_len == self.ops.len) return error.SourceLimit;
        const index = self.ops_len;
        self.ops[index] = op;
        self.ops_len += 1;
        return index;
    }
};

const Line = struct {
    compiler: *Compiler,
    cursor: usize,
    end: usize,
    token: Token = .{ .kind = .end },

    fn lex(self: *Line) Error!Token {
        const source = &self.compiler.source;
        while (self.cursor < self.end) : (self.cursor += 1) {
            switch (source[self.cursor]) {
                ' ', '\t' => {},
                else => break,
            }
        }
        const start = self.cursor;
        if (start == self.end or source[start] == '#') return .{ .kind = .end };
        const byte = source[start];
        self.cursor += 1;
        if (identifier_start(byte)) {
            while (self.cursor < self.end and identifier_byte(source[self.cursor])) {
                self.cursor += 1;
            }
            const name = source[start..self.cursor];
            if (name.len > name_max) return error.NameLimit;
            return self.make_token(try name_kind(name), start);
        }
        if (std.ascii.isDigit(byte)) {
            while (self.cursor < self.end and std.ascii.isDigit(source[self.cursor])) {
                self.cursor += 1;
            }
            if (byte == '0') {
                for (source[start + 1 .. self.cursor]) |digit| {
                    if (digit != '0') return error.SyntaxError;
                }
            }
            return self.make_token(.number, start);
        }
        const kind: Kind = switch (byte) {
            '\'', '"' => return self.lex_string(byte),
            '(' => .left,
            ')' => .right,
            ',' => .comma,
            ':' => .colon,
            '+' => .add,
            '-' => .sub,
            '*' => .mul,
            '%' => .mod,
            '/' => if (self.consume_byte('/')) .floor_div else return error.SyntaxError,
            '=' => if (self.consume_byte('=')) .eq else .assign,
            '!' => if (self.consume_byte('=')) .ne else return error.SyntaxError,
            '<' => if (self.consume_byte('=')) .le else .lt,
            '>' => if (self.consume_byte('=')) .ge else .gt,
            else => return error.SyntaxError,
        };
        return self.make_token(kind, start);
    }

    fn lex_string(self: *Line, quote: u8) Error!Token {
        const start = self.compiler.literals_len;
        const source = &self.compiler.source;
        while (self.cursor < self.end) {
            var byte = source[self.cursor];
            self.cursor += 1;
            if (byte == quote) {
                return .{
                    .kind = .string,
                    .span = .{ .start = start, .end = self.compiler.literals_len },
                };
            }
            if (byte == '\r') return error.SyntaxError;
            if (byte == '\\') {
                if (self.cursor == self.end) return error.SyntaxError;
                byte = switch (source[self.cursor]) {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\', '\'', '"' => |escape| escape,
                    else => return error.SyntaxError,
                };
                self.cursor += 1;
            }
            if (self.compiler.literals_len - start == string_max) return error.StringLimit;
            // Each decoded byte consumed at least one byte of the bounded source.
            if (self.compiler.literals_len == self.compiler.literals.len) return error.SourceLimit;
            self.compiler.literals[self.compiler.literals_len] = byte;
            self.compiler.literals_len += 1;
        }
        return error.SyntaxError;
    }

    fn consume_byte(self: *Line, byte: u8) bool {
        if (self.cursor == self.end or self.compiler.source[self.cursor] != byte) return false;
        self.cursor += 1;
        return true;
    }

    fn make_token(self: *const Line, kind: Kind, start: usize) Token {
        return .{ .kind = kind, .span = .{ .start = @intCast(start), .end = @intCast(self.cursor) } };
    }

    fn advance(self: *Line) Error!void {
        self.token = try self.lex();
    }

    fn expect(self: *Line, kind: Kind) Error!Token {
        const token = self.token;
        if (token.kind != kind) return error.SyntaxError;
        try self.advance();
        return token;
    }

    fn statement(self: *Line) Error!void {
        const compiler = self.compiler;
        switch (self.token.kind) {
            .name => {
                const name = (try self.expect(.name)).span;
                _ = try self.expect(.assign);
                try self.expression();
                _ = try compiler.emit(.{ .store = name });
            },
            .while_ => {
                try self.advance();
                const head = compiler.ops_len;
                try self.expression();
                _ = try self.expect(.colon);
                if (compiler.blocks_len == stack_max) return error.StackLimit;
                const exit = try compiler.emit(.{ .jump_false = 0 });
                compiler.blocks[compiler.blocks_len] = .{ .head = head, .exit = exit };
                compiler.blocks_len += 1;
            },
            .print => {
                try self.advance();
                _ = try self.expect(.left);
                try self.expression();
                _ = try self.expect(.right);
                _ = try compiler.emit(.print);
            },
            .query => {
                try self.call_statement();
                _ = try compiler.emit(.discard);
            },
            .submit => try self.submit(),
            else => return error.SyntaxError,
        }
    }

    fn call_statement(self: *Line) Error!void {
        _ = try self.expect(.query);
        _ = try self.expect(.left);
        try self.expression();
        _ = try self.expect(.right);
        _ = try self.compiler.emit(.query);
    }

    fn submit(self: *Line) Error!void {
        _ = try self.expect(.submit);
        _ = try self.expect(.left);
        var names: [bindings_max]Span = undefined;
        var count: usize = 0;
        while (self.token.kind != .right) {
            if (count == names.len) return error.BindingLimit;
            const name = (try self.expect(.name)).span;
            const source = &self.compiler.source;
            for (names[0..count]) |prior| {
                if (std.mem.eql(u8, name.slice(source), prior.slice(source))) {
                    return error.SyntaxError;
                }
            }
            names[count] = name;
            count += 1;
            _ = try self.expect(.assign);
            try self.expression();
            _ = try self.compiler.emit(.{ .capture = name });
            if (self.token.kind != .comma) break;
            try self.advance();
        }
        _ = try self.expect(.right);
        _ = try self.compiler.emit(.submit);
    }

    fn expression(self: *Line) Error!void {
        var pending: Operators = .{};
        var needs_operand = true;
        // Each iteration consumes at least one source byte or returns. Leave
        // one extra iteration to accept the delimiter after the last operand.
        for (0..source_bytes_max + 1) |_| {
            if (needs_operand) {
                needs_operand = try self.operand(&pending);
                continue;
            }
            const kind = self.token.kind;
            if (kind == .right and pending.groups != 0) {
                try pending.close(self.compiler);
            } else if (precedence(kind) != 0) {
                try pending.binary(self.compiler, kind);
                needs_operand = true;
            } else {
                if (pending.groups != 0) return error.SyntaxError;
                while (pending.len != 0) try pending.emit_top(self.compiler);
                return;
            }
            try self.advance();
        }
        unreachable; // The source byte cap was checked before lexing.
    }

    // Returns true if this prefix still needs an operand.
    fn operand(self: *Line, pending: *Operators) Error!bool {
        const token = self.token;
        const compiler = self.compiler;
        try self.advance();
        const op: Op = switch (token.kind) {
            .name => .{ .load = token.span },
            .number => .{ .integer = std.fmt.parseInt(i64, token.span.slice(&compiler.source), 10) catch return error.IntegerOverflow },
            .string => .{ .string = token.span },
            .true_, .false_ => .{ .boolean = token.kind == .true_ },
            .none => .none,
            .left, .query => {
                if (token.kind == .query) _ = try self.expect(.left);
                try pending.push(token.kind);
                pending.groups += 1;
                return true;
            },
            .add, .sub => {
                // The one negative literal whose magnitude is outside i64.
                if (token.kind == .sub and self.token.kind == .number and
                    std.mem.eql(u8, self.token.span.slice(&compiler.source), "9223372036854775808"))
                {
                    try self.advance();
                    _ = try compiler.emit(.{ .integer = std.math.minInt(i64) });
                    return false;
                }
                try pending.push(if (token.kind == .add) .pos else .neg);
                return true;
            },
            else => return error.SyntaxError,
        };
        _ = try compiler.emit(op);
        return false;
    }
};

fn precedence(kind: Kind) u8 {
    return switch (kind) {
        .eq, .ne, .lt, .le, .gt, .ge => 1,
        .add, .sub => 2,
        .mul, .floor_div, .mod => 3,
        .neg, .pos => 4,
        else => 0,
    };
}

const Operators = struct {
    stack: [stack_max]Kind = undefined,
    len: usize = 0,
    groups: usize = 0,

    fn push(self: *Operators, kind: Kind) Error!void {
        if (self.len == self.stack.len) return error.StackLimit;
        self.stack[self.len] = kind;
        self.len += 1;
    }

    fn emit_top(self: *Operators, compiler: *Compiler) Error!void {
        self.len -= 1;
        const kind = self.stack[self.len];
        _ = try compiler.emit(switch (kind) {
            .neg, .pos => .{ .unary = kind },
            else => .{ .binary = kind },
        });
    }

    fn binary(self: *Operators, compiler: *Compiler, kind: Kind) Error!void {
        const priority = precedence(kind);
        while (self.len != 0) {
            const top = precedence(self.stack[self.len - 1]);
            if (top < priority) break;
            // Python comparison chains have short-circuit semantics, not
            // left association. Reject them instead of silently changing meaning.
            if (top == 1 and priority == 1) return error.SyntaxError;
            try self.emit_top(compiler);
        }
        try self.push(kind);
    }

    fn close(self: *Operators, compiler: *Compiler) Error!void {
        std.debug.assert(self.groups != 0);
        while (self.len != 0) {
            const kind = self.stack[self.len - 1];
            if (kind == .left or kind == .query) {
                self.len -= 1;
                self.groups -= 1;
                if (kind == .query) _ = try compiler.emit(.query);
                return;
            }
            try self.emit_top(compiler);
        }
        unreachable;
    }
};

const Execution = struct {
    stack: [stack_max]Value = undefined,
    // A host result shares the popped operand's slot; the next write to that
    // slot cannot invalidate any remaining operand. No loop grows this storage.
    replies: [stack_max][string_max]u8 = undefined,
    len: usize = 0,
    output_len: usize = 0,
    submit_len: usize = 0,

    fn run(
        self: *Execution,
        machine: *Machine,
        compiler: *const Compiler,
        fuel_limit: u32,
        host: ?Host,
    ) anyerror!Result {
        var pc: u16 = 0;
        var fuel = fuel_limit;
        while (pc < compiler.ops_len) {
            if (fuel == 0) return error.FuelExhausted;
            fuel -= 1;
            const op = compiler.ops[pc];
            pc += 1;
            switch (op) {
                .integer => |value| try self.push(.{ .integer = value }),
                .boolean => |value| try self.push(.{ .boolean = value }),
                .none => try self.push(.none),
                .string => |span| try self.push(.{ .string = span.slice(&compiler.literals) }),
                .load => |span| {
                    const index = machine.lookup(span.slice(&compiler.source)) orelse
                        return error.NameError;
                    try self.push(machine.globals[index].view().value);
                },
                .store => |span| try machine.store(span.slice(&compiler.source), self.pop()),
                .capture => |span| try self.capture(machine, span.slice(&compiler.source)),
                .unary => |kind| {
                    const value = try integer(self.pop());
                    const result = if (kind == .neg) try checked(@subWithOverflow(0, value)) else value;
                    try self.push(.{ .integer = result });
                },
                .binary => |kind| {
                    const rhs = self.pop();
                    const lhs = self.pop();
                    try self.push(try binary_value(kind, lhs, rhs));
                },
                .jump => |target| pc = target,
                .jump_false => |target| if (!truthy(self.pop())) {
                    pc = target;
                },
                .print => try self.print(machine),
                .query => try self.query(host),
                .discard => _ = self.pop(),
                .submit => return .{ .submit = machine.submit_views[0..self.submit_len] },
            }
        }
        std.debug.assert(self.len == 0);
        return .{ .observation = machine.output[0..self.output_len] };
    }

    fn push(self: *Execution, value: Value) Error!void {
        if (self.len == self.stack.len) return error.StackLimit;
        self.stack[self.len] = value;
        self.len += 1;
    }

    fn pop(self: *Execution) Value {
        // The compiler emits stack-balanced statements; guests cannot directly
        // supply bytecode or jump to the interior of an expression.
        std.debug.assert(self.len > 0);
        self.len -= 1;
        return self.stack[self.len];
    }

    fn capture(self: *Execution, machine: *Machine, name: []const u8) Error!void {
        if (self.submit_len == bindings_max) return error.BindingLimit;
        const result = &machine.submitted[self.submit_len];
        try result.set_value(self.pop());
        result.set_name(name);
        machine.submit_views[self.submit_len] = result.view();
        self.submit_len += 1;
    }

    fn print(self: *Execution, machine: *Machine) Error!void {
        var formatted: [21]u8 = undefined;
        const bytes: []const u8 = switch (self.pop()) {
            .none => "None",
            .boolean => |value| if (value) "True" else "False",
            .integer => |value| std.fmt.bufPrint(&formatted, "{d}", .{value}) catch unreachable,
            .string => |text| text,
        };
        const remaining = machine.output[self.output_len..];
        if (bytes.len >= remaining.len) return error.OutputLimit;
        @memcpy(remaining[0..bytes.len], bytes);
        remaining[bytes.len] = '\n';
        self.output_len += bytes.len + 1;
    }

    fn query(self: *Execution, host: ?Host) anyerror!void {
        const prompt = switch (self.pop()) {
            .string => |text| text,
            else => return error.TypeError,
        };
        const callback = host orelse return error.HostUnavailable;
        const response = try callback.query(callback.ptr, prompt);
        if (response.len > string_max) return error.StringLimit;
        const slot = self.replies[self.len][0..response.len];
        @memmove(slot, response);
        try self.push(.{ .string = slot });
    }
};

fn integer(value: Value) Error!i64 {
    return switch (value) {
        .integer => |number| number,
        .boolean => |bit| @intFromBool(bit),
        else => error.TypeError,
    };
}

fn truthy(value: Value) bool {
    return switch (value) {
        .none => false,
        .boolean => |bit| bit,
        .integer => |number| number != 0,
        .string => |text| text.len != 0,
    };
}

fn equal(lhs: Value, rhs: Value) bool {
    return switch (lhs) {
        .none => rhs == .none,
        .string => |text| rhs == .string and std.mem.eql(u8, text, rhs.string),
        .integer, .boolean => (integer(lhs) catch unreachable) == (integer(rhs) catch return false),
    };
}

fn checked(result: struct { i64, u1 }) Error!i64 {
    if (result[1] != 0) return error.IntegerOverflow;
    return result[0];
}

fn binary_value(kind: Kind, lhs: Value, rhs: Value) Error!Value {
    if (kind == .eq or kind == .ne) return .{ .boolean = equal(lhs, rhs) == (kind == .eq) };
    const left = try integer(lhs);
    const right = try integer(rhs);
    return switch (kind) {
        .add => .{ .integer = try checked(@addWithOverflow(left, right)) },
        .sub => .{ .integer = try checked(@subWithOverflow(left, right)) },
        .mul => .{ .integer = try checked(@mulWithOverflow(left, right)) },
        .floor_div, .mod => blk: {
            if (right == 0) return error.DivisionByZero;
            if (left == std.math.minInt(i64) and right == -1) {
                if (kind == .floor_div) return error.IntegerOverflow;
                break :blk .{ .integer = 0 };
            }
            if (kind == .floor_div) break :blk .{ .integer = @divFloor(left, right) };
            // Python's remainder follows the divisor's sign, even when negative.
            const magnitude: i64 = @intCast(@abs(left) % @abs(right));
            const remainder = if (left < 0) -magnitude else magnitude;
            break :blk .{ .integer = if (remainder != 0 and ((remainder < 0) != (right < 0)))
                remainder + right
            else
                remainder };
        },
        .lt => .{ .boolean = left < right },
        .le => .{ .boolean = left <= right },
        .gt => .{ .boolean = left > right },
        .ge => .{ .boolean = left >= right },
        else => unreachable,
    };
}

test "bounded Python caps source, fuel, expression stack, bindings and output" {
    const t = std.testing;
    var machine: Machine = undefined;
    machine.init();
    const spaces: [source_bytes_max + 1]u8 = @splat(' ');
    const opens: [stack_max + 1:0]u8 = @splat('(');
    const name: [name_max + 1:0]u8 = @splat('a');
    const bytes: [string_max + 1:0]u8 = @splat('a');
    try t.expectEqualStrings("", (try machine.run(spaces[0..source_bytes_max], &.{}, .{}, null)).observation);
    try t.expectError(error.SourceLimit, machine.run(&spaces, &.{}, .{}, null));
    try t.expectError(error.FuelLimit, machine.run("", &.{}, .{ .fuel = fuel_max + 1 }, null));
    try t.expectError(error.FuelExhausted, machine.run("while True:\n    x = 0", &.{}, .{ .fuel = 19 }, null));
    try t.expectError(error.StackLimit, machine.run("x = " ++ opens ++ "0", &.{}, .{}, null));
    try t.expectError(error.NameLimit, machine.run(name ++ " = 0", &.{}, .{}, null));
    try t.expectError(error.StringLimit, machine.run("x = '" ++ bytes ++ "'", &.{}, .{}, null));
    const max_string: [string_max:0]u8 = @splat('a');
    const too_much_output = "x = '" ++ max_string ++ "'\nprint(x)\nprint(x)";
    try t.expectError(error.OutputLimit, machine.run(too_much_output, &.{}, .{}, null));
    machine.reset();
    const oversized: [bindings_max + 1]Binding = @splat(.{ .name = "x", .value = .none });
    try t.expectError(error.BindingLimit, machine.run("", &oversized, .{}, null));
}

test "bounded Python validates the whole source before seeding or calling the host" {
    const t = std.testing;
    const Callback = struct {
        fn query(ptr: *anyopaque, prompt: []const u8) anyerror![]const u8 {
            const called: *bool = @ptrCast(@alignCast(ptr));
            called.* = true;
            return prompt;
        }
    };
    var called = false;
    const host: Host = .{ .ptr = &called, .query = Callback.query };
    const seed = [_]Binding{.{ .name = "seed", .value = .{ .integer = 1 } }};
    var machine: Machine = .{};
    const invalid = [_][]const u8{
        "x = llm_query('x')\nimport os",
        "x = llm_query('x')\nwhile True:\n",
        "x = llm_query('x')\nwhile True:\n    x = 1\n  x = 2",
        "x = llm_query('x')\nSUBMIT(a=1, a=2)",
        "x = llm_query('x')\nSUBMIT(answer=(1+2)",
        "x = llm_query('x')\nprint(1 < 2 < 3)",
    };
    for (invalid) |source| {
        try t.expectError(error.SyntaxError, machine.run(source, &seed, .{}, host));
        try t.expect(!called);
    }
    try t.expectError(error.NameError, machine.run("print(seed)", &.{}, .{}, null));
    try t.expectError(error.NameError, machine.run("print(x)", &.{}, .{}, null));
    try t.expectError(error.DivisionByZero, machine.run("print(1 // 0)", &.{}, .{}, null));
    try t.expectError(error.IntegerOverflow, machine.run("print(9223372036854775807 + 1)", &.{}, .{}, null));
    try t.expectError(error.IntegerOverflow, machine.run("print(-9223372036854775808 // -1)", &.{}, .{}, null));
}

test "bounded Python persists owned values and repeats deterministically after reset" {
    const t = std.testing;
    const Callback = struct {
        bytes: [1]u8 = undefined,

        fn query(ptr: *anyopaque, prompt: []const u8) anyerror![]const u8 {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (prompt.len != 1) return error.BadTestPrompt;
            self.bytes[0] = prompt[0];
            return &self.bytes;
        }
    };
    var machine: Machine = .{};
    var callback: Callback = .{};
    const host: Host = .{ .ptr = &callback, .query = Callback.query };
    var input_bytes = [_]u8{'o'};
    const seed = [_]Binding{
        .{ .name = "n", .value = .{ .integer = 3 } },
        .{ .name = "original", .value = .{ .string = &input_bytes } },
    };
    const source =
        \\total = 0
        \\while n > 0:
        \\    total = total + n
        \\    n = n - 1
        \\saved = llm_query('a')
        \\print(llm_query('b') != llm_query('c'))
    ;
    for (0..2) |_| {
        input_bytes[0] = 'o';
        try t.expectEqualStrings("True\n", (try machine.run(source, &seed, .{}, host)).observation);
        input_bytes[0] = 'X';
        callback.bytes[0] = 'X';
        const report =
            \\print(original)
            \\print(saved)
            \\print(n)
            \\print(-7 // 3)
            \\print(7 % -3)
        ;
        try t.expectEqualStrings("o\na\n0\n-3\n-2\n", (try machine.run(report, &seed, .{}, null)).observation);
        const result = (try machine.run(
            "SUBMIT(answer=total, copied=original)\nllm_query('X')",
            &seed,
            .{},
            null,
        )).submit;
        try t.expectEqual(@as(usize, 2), result.len);
        try t.expectEqualStrings("answer", result[0].name);
        try t.expectEqual(@as(i64, 6), result[0].value.integer);
        try t.expectEqualStrings("copied", result[1].name);
        try t.expectEqualStrings("o", result[1].value.string);
        machine.reset();
        try t.expectError(error.NameError, machine.run("print(total)", &.{}, .{}, null));
        machine.reset();
    }
}
