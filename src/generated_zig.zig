//! Native Zig code generation through a genuine Predict(Code[zig]), followed by
//! isolated compilation/execution. This is not DSPy's arbitrary Python RLM/Flex
//! environment. The Linux executor fails closed when its prerequisites are absent.
const signature = @import("signature.zig");
const predict = @import("predict.zig");
const code = @import("code.zig");
const executor = @import("generated_zig_executor_linux.zig");

pub const Executor = executor.Executor;
pub const Options = executor.Options;
pub const Capability = executor.Capability;
pub const CapabilityError = executor.CapabilityError;
pub const Error = predict.Error || executor.Error;
pub const Lane = @import("generated_zig_cgroup_linux.zig").Lane;
pub const Limits = @import("generated_zig_cgroup_linux.zig").Limits;
pub const Source = @import("generated_zig_source.zig");
pub const ZigCode = code.CodeWithLanguage("zig");

pub const CodeSignature = signature.Signature(struct {
    pub const instructions =
        "Write a native Zig program to solve the task using the supplied JSON input and capabilities. " ++
        "The only import is exactly const guest = @import(\"guest\");. " ++
        "The only public entrypoint is exactly pub fn run(ctx: *guest.Context) !void. " ++
        "ctx.input() returns ![]const u8 containing input_json. " ++
        "ctx.tool(name, argument_json) returns ![]const u8 containing the capability's JSON reply; " ++
        "the reply borrows fixed storage until the next tool call. " ++
        "ctx.finish(result_json) sends the final JSON value; call it exactly once and return. " ++
        "Use only the exact registered capability names. " ++
        "guest.std provides pure algorithms, JSON, formatting and fixed-buffer allocators. " ++
        "All storage must be fixed-size; no heap allocation, files, sockets, processes or threads. " ++
        "No asm, extern, export, alternate imports, integer-to-pointer casts or safety disabling. " ++
        "Keep loops and recursion bounded. The source limit is 64 KiB, 4096 tokens and nodes, " ++
        "and 64 nested delimiters. There are no runtime dependencies beyond the guest module. " ++
        "The returned code must implement the complete task, not pseudocode or placeholders.";
    pub const Input = struct {
        task: []const u8,
        input_json: []const u8,
        /// Include exact names, argument/result shapes and native-tool semantics.
        capabilities: []const u8,
    };
    pub const Output = struct { code: ZigCode };
});

/// Ordinary discoverable Predict state: optimizer bindings, demos, adapters,
/// tracing and per-call LM options use the existing native predictor machinery.
pub const Generator = predict.Predict(CodeSignature);

/// No execution retries, interpreter, alternate model or unsafe fallback.
/// The executor and registered bounded native capability contexts are initialized
/// at startup. The resulting JSON borrows executor storage until its next run.
pub fn generateAndRun(generator: *const Generator, engine: *Executor, context: predict.Context, input: CodeSignature.Input) Error![]const u8 {
    const generated = try generator.forward(context, input);
    return engine.run(generated.code.code, input.input_json);
}
