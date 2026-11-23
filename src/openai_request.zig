//! Canonical SDK configuration on OpenAI Chat and Responses wires.
//! The owning request survives transport; this module only builds wire views.
const std = @import("std");
const sdk = @import("sdk_response.zig");
const runtime = @import("runtime_value.zig");
const content = @import("openai_content.zig");
const graph = @import("json_graph.zig");
const Allocator = std.mem.Allocator;
const Value = graph.Value;
const Object = graph.Record;
const Array = graph.Array;
pub const Error = content.Error;
pub const Dialect = content.Dialect;
pub const Adaptations = enum { note, silent, refuse };
pub const Compat = struct {
    content_policy: content.Policy,
    max_tokens_field: enum { max_tokens, max_completion_tokens, max_output_tokens },
    reasoning_format: enum { none, responses_reasoning, reasoning_effort, openrouter, deepseek, kimi, qwen, qwen_chat_template, zai },
    cache_control: enum { none, openai, openai_implicit, anthropic } = .openai,
    strict_tools: enum { omit, include } = .omit,
    builtin_tools: enum { reject, openai, verbatim, groq },
    user_field: enum { user, user_id, safety_identifier },
    forced_tool_choice: enum { send, reject } = .send,
    json_schema: enum { send, reject } = .send,
    reasoning_off: enum { send, lowest } = .send,
    reasoning_efforts: ?[]const sdk.ReasoningEffort = null,
    routing: ?Value = null,
    stream_usage: enum { include, omit } = .include,
    assistant_after_tool_result: enum { insert, omit } = .omit,
    token_scoring: enum { none, logprob_token_ids } = .none,
    reasoning_usage: enum { in_completion, separate } = .in_completion,

    pub fn defaults(dialect: Dialect) Compat {
        return if (dialect == .chat) .{
            .content_policy = .{},
            .max_tokens_field = .max_completion_tokens,
            .reasoning_format = .reasoning_effort,
            .builtin_tools = .reject,
            .user_field = .user,
        } else .{
            .content_policy = .{ .instruction_role = .developer, .tool_result_media = .native },
            .max_tokens_field = .max_output_tokens,
            .reasoning_format = .responses_reasoning,
            .builtin_tools = .openai,
            .user_field = .safety_identifier,
        };
    }
};
/// A borrowed, partial Responses policy. Null inherits; auto explicitly resets
/// to the dialect default. Routing and extensions retain their graph roots:
/// constructor validation checks the outer object, not its JSON serializability.
/// Use clone when the policy must outlive the caller's storage.
pub const ResponsesCompat = struct {
    developer_role: ?enum { auto, developer, system } = null,
    max_output_tokens_field: ?enum { auto, max_output_tokens, max_completion_tokens, max_tokens } = null,
    reasoning_format: ?enum {
        auto,
        none,
        responses_reasoning,
        reasoning_effort,
        openrouter,
        deepseek,
        qwen,
        qwen_chat_template,
        zai,
    } = null,
    tool_result_name: ?enum { auto, include, omit } = null,
    strict_tools: ?enum { auto, include, omit } = null,
    cache_control: ?enum { auto, none, openai, openai_implicit, anthropic } = null,
    commentary_phase: ?enum { auto, omit, tag } = null,
    edit_image_field: ?enum { auto, array, indexed } = null,
    builtin_tools: ?enum { auto, openai, verbatim } = null,
    tool_result_media: ?enum { auto, native, images, reject } = null,
    routing: ?Value = null,
    extensions: ?Value = null,

    /// Returns the shared immutable partial policy, not a resolved copy.
    pub fn preset(name: []const u8) error{UnknownResponsesPreset}!*const ResponsesCompat {
        return (try ResponsesPreset.fromName(name)).policy();
    }

    pub fn init(options: ResponsesCompat) Error!ResponsesCompat {
        try options.validate();
        return options;
    }

    fn validate(self: *const ResponsesCompat) Error!void {
        // Both ordinary records and genuine Python dictionaries satisfy the
        // Source outer-dict check, including dictionaries with non-string keys.
        inline for (.{ "routing", "extensions" }) |name| {
            if (@field(self, name)) |value| {
                if (value != .object and value != .dictionary) return error.LMFailure;
            }
        }
    }

    /// Borrow scalar storage while importing container headers. Unknown
    /// constructor fields refuse, unlike ignored request-extension knobs.
    pub fn fromValue(arena: Allocator, input: runtime.Value) Error!ResponsesCompat {
        if (input != .object) return error.LMFailure;
        if (input.object.count() > @typeInfo(ResponsesCompat).@"struct".field_names.len)
            return error.LMFailure;
        var result: ResponsesCompat = .{};
        var context: graph.ImportContext = .init(arena);
        defer context.deinit();
        for (input.object.keys()) |name| {
            var recognized = false;
            inline for (@typeInfo(ResponsesCompat).@"struct".field_names) |field_name| {
                if (std.mem.eql(u8, name, field_name)) recognized = true;
            }
            if (!recognized) return error.LMFailure;
        }
        inline for (@typeInfo(ResponsesCompat).@"struct".field_names) |name| {
            if (input.object.get(name)) |value| {
                if (value != .null) {
                    const T = @typeInfo(@FieldType(ResponsesCompat, name)).optional.child;
                    @field(result, name) = if (comptime T == Value)
                        context.fromValue(value) catch |err| return mapError(err)
                    else if (value == .string)
                        std.meta.stringToEnum(T, value.string) orelse return error.LMFailure
                    else
                        return error.LMFailure;
                }
            }
        }
        return init(result);
    }

    pub fn clone(self: *const ResponsesCompat, arena: Allocator) Allocator.Error!ResponsesCompat {
        var custom: runtime.CloneContext = .init(arena);
        defer custom.deinit();
        var context = graph.CloneContext.init(arena, arena, &custom);
        defer context.deinit();
        var result = self.*;
        inline for (.{ "routing", "extensions" }) |name| {
            if (@field(self, name)) |value| @field(result, name) = try context.clone(value);
        }
        return result;
    }

    /// Owning ordinary record projection for native provider persistence.
    pub fn toValue(self: *const ResponsesCompat, arena: Allocator) Error!runtime.Value {
        var object_value: runtime.ObjectMap = .empty;
        try object_value.ensureTotalCapacity(arena, @typeInfo(ResponsesCompat).@"struct".field_names.len);
        inline for (@typeInfo(ResponsesCompat).@"struct".field_names) |name| {
            const encoded: runtime.Value = if (@field(self, name)) |value| blk: {
                break :blk if (comptime @TypeOf(value) == Value)
                    graph.toValue(arena, value) catch |err| return mapError(err)
                else
                    .{ .string = @tagName(value) };
            } else .null;
            object_value.putAssumeCapacity(name, encoded);
        }
        return .{ .object = object_value };
    }

    pub fn resolve(self: *const ResponsesCompat) Error!Compat {
        try self.validate();
        var result = Compat.defaults(.responses);
        inline for (.{
            .{ "developer_role", "instruction_role" },
            .{ "tool_result_name", "tool_result_name" },
            .{ "tool_result_media", "tool_result_media" },
            .{ "commentary_phase", "commentary_phase" },
        }) |names| {
            if (@field(self, names[0])) |value| {
                const target = &@field(result.content_policy, names[1]);
                target.* = try policyWord(
                    @TypeOf(target.*),
                    .{ .string = @tagName(value) },
                    target.*,
                    target.*,
                );
            }
        }
        inline for (.{
            .{ "max_output_tokens_field", "max_tokens_field" },
            .{ "reasoning_format", "reasoning_format" },
            .{ "strict_tools", "strict_tools" },
            .{ "cache_control", "cache_control" },
            .{ "builtin_tools", "builtin_tools" },
        }) |names| {
            if (@field(self, names[0])) |value| {
                const target = &@field(result, names[1]);
                target.* = try policyWord(
                    @TypeOf(target.*),
                    .{ .string = @tagName(value) },
                    target.*,
                    target.*,
                );
            }
        }
        result.routing = self.routing;
        // Source retains extensions as compatibility metadata, not body
        // overrides. Image-edit policy likewise does not alter generation.
        return result;
    }
};

/// Partial Chat policy. Slices and graph roots are borrowed. Compatibility
/// extensions and assistant-after-tool are retained policy metadata: the
/// Source Chat generation adapter does not use them to change the wire.
pub const ChatCompat = struct {
    instruction_role: ?enum { auto, developer, system } = null,
    max_tokens_field: ?enum { auto, max_completion_tokens, max_tokens } = null,
    stream_usage: ?enum { auto, include, omit } = null,
    tool_result_name: ?enum { auto, include, omit } = null,
    assistant_after_tool_result: ?enum { auto, insert, omit } = null,
    thinking_format: ?enum {
        auto,
        none,
        reasoning_effort,
        openrouter,
        deepseek,
        kimi,
        qwen,
        qwen_chat_template,
    } = null,
    thinking_replay: ?enum { auto, native, as_text, omit } = null,
    assistant_reasoning_content: ?enum { auto, include_empty, omit } = null,
    strict_tools: ?enum { auto, include, omit } = null,
    builtin_tools: ?enum { auto, reject, groq } = null,
    tool_result_media: ?enum { auto, native, images, reject } = null,
    cache_control: ?enum { auto, none, openai, openai_implicit, anthropic } = null,
    user_field: ?enum { auto, user, user_id, safety_identifier } = null,
    forced_tool_choice: ?enum { auto, send, reject } = null,
    reasoning_off: ?enum { auto, send, lowest } = null,
    json_schema: ?enum { auto, send, reject } = null,
    token_scoring: ?enum { auto, none, logprob_token_ids } = null,
    reasoning_efforts: ?[]const sdk.ReasoningEffort = null,
    reasoning_usage: ?enum { auto, in_completion, separate } = null,
    routing: ?Value = null,
    extensions: ?Value = null,
    model_overrides: []const Override = &.{},

    pub const Override = struct {
        prefix: []const u8,
        knobs: []const Knob,
    };

    pub const Knob = struct {
        name: Name,
        value: union(enum) {
            inherit,
            word: []const u8,
            levels: []const sdk.ReasoningEffort,
        },

        pub const Name = enum {
            instruction_role,
            max_tokens_field,
            stream_usage,
            thinking_format,
            thinking_replay,
            assistant_reasoning_content,
            strict_tools,
            cache_control,
            user_field,
            forced_tool_choice,
            json_schema,
            reasoning_efforts,
            tool_result_media,
            reasoning_off,
            reasoning_usage,
        };
    };

    pub fn preset(name: []const u8) error{UnknownChatPreset}!*const ChatCompat {
        return (try chatPreset(name)).policy();
    }

    pub fn init(options: ChatCompat) Error!ChatCompat {
        try options.validate();
        return options;
    }

    fn validate(self: *const ChatCompat) Error!void {
        try validateEfforts(self.reasoning_efforts);
        for (self.model_overrides) |entry| {
            if (entry.prefix.len == 0) return error.LMFailure;
            var seen: std.EnumSet(Knob.Name) = .empty;
            for (entry.knobs) |knob| {
                if (seen.contains(knob.name)) return error.LMFailure;
                seen.insert(knob.name);
            }
        }
        inline for (.{ "routing", "extensions" }) |name| {
            if (@field(self, name)) |value| {
                if (value != .object and value != .dictionary) return error.LMFailure;
            }
        }
    }

    fn validateEfforts(efforts: ?[]const sdk.ReasoningEffort) Error!void {
        if (efforts) |levels| {
            for (levels) |level| {
                if (level == .off) return error.LMFailure;
            }
        }
    }

    /// Prefixes are case-sensitive and ordered, not longest-match. Values are
    /// interpreted only for the selected entry, as in Source's for_model.
    pub fn forModel(self: *const ChatCompat, model: []const u8) Error!ChatCompat {
        try self.validate();
        for (self.model_overrides) |entry| {
            if (!std.mem.startsWith(u8, model, entry.prefix)) continue;
            var result = self.*;
            result.model_overrides = &.{};
            for (entry.knobs) |knob| try result.applyKnob(knob);
            try validateEfforts(result.reasoning_efforts);
            return result;
        }
        return self.*;
    }

    fn applyKnob(self: *ChatCompat, knob: Knob) Error!void {
        switch (knob.name) {
            .reasoning_efforts => self.reasoning_efforts = switch (knob.value) {
                .inherit => null,
                .levels => |levels| levels,
                .word => return error.LMFailure,
            },
            inline else => |name| {
                const target = &@field(self, @tagName(name));
                const T = @typeInfo(@TypeOf(target.*)).optional.child;
                target.* = switch (knob.value) {
                    .inherit => null,
                    .word => |word| std.meta.stringToEnum(T, word) orelse return error.LMFailure,
                    .levels => return error.LMFailure,
                };
            },
        }
    }

    pub fn clone(self: *const ChatCompat, arena: Allocator) Allocator.Error!ChatCompat {
        var result = self.*;
        var custom: runtime.CloneContext = .init(arena);
        defer custom.deinit();
        var context = graph.CloneContext.init(arena, arena, &custom);
        defer context.deinit();
        inline for (.{ "routing", "extensions" }) |name| {
            if (@field(self, name)) |value| @field(result, name) = try context.clone(value);
        }
        if (self.reasoning_efforts) |levels| result.reasoning_efforts =
            try arena.dupe(sdk.ReasoningEffort, levels);
        if (self.model_overrides.len == 0) return result;
        const overrides = try arena.alloc(Override, self.model_overrides.len);
        for (self.model_overrides, overrides) |entry, *target| {
            const knobs = try arena.dupe(Knob, entry.knobs);
            for (knobs) |*knob| {
                knob.value = switch (knob.value) {
                    .inherit => .inherit,
                    .word => |word| .{ .word = try arena.dupe(u8, word) },
                    .levels => |levels| .{ .levels = try arena.dupe(sdk.ReasoningEffort, levels) },
                };
            }
            target.* = .{ .prefix = try arena.dupe(u8, entry.prefix), .knobs = knobs };
        }
        result.model_overrides = overrides;
        return result;
    }

    pub fn toValue(self: *const ChatCompat, arena: Allocator) Error!runtime.Value {
        var result: runtime.ObjectMap = .empty;
        const fields = @typeInfo(ChatCompat).@"struct".field_names;
        try result.ensureTotalCapacity(arena, fields.len);
        inline for (fields) |name| {
            const field = @field(self, name);
            const encoded = if (comptime @TypeOf(field) == []const Override)
                try overridesToValue(arena, field)
            else if (field) |value|
                if (comptime @TypeOf(value) == Value)
                    graph.toValue(arena, value) catch |err| return mapError(err)
                else if (comptime @TypeOf(value) == []const sdk.ReasoningEffort)
                    try effortsToValue(arena, value)
                else
                    runtime.Value{ .string = @tagName(value) }
            else
                .null;
            result.putAssumeCapacity(name, encoded);
        }
        return .{ .object = result };
    }

    pub fn fromValue(arena: Allocator, input: runtime.Value) Error!ChatCompat {
        if (input != .object) return error.LMFailure;
        var result: ChatCompat = .{};
        var context: graph.ImportContext = .init(arena);
        defer context.deinit();
        for (input.object.keys(), input.object.values()) |key, encoded| {
            var recognized = false;
            inline for (@typeInfo(ChatCompat).@"struct".field_names) |name| {
                if (std.mem.eql(u8, key, name)) {
                    recognized = true;
                    const field = &@field(result, name);
                    if (comptime @TypeOf(field.*) == []const Override) {
                        field.* = try overridesFromValue(arena, encoded);
                    } else if (encoded != .null) {
                        const T = @typeInfo(@TypeOf(field.*)).optional.child;
                        field.* = if (comptime T == Value)
                            context.fromValue(encoded) catch |err| return mapError(err)
                        else if (comptime T == []const sdk.ReasoningEffort)
                            try effortsFromValue(arena, encoded)
                        else if (encoded == .string)
                            std.meta.stringToEnum(T, encoded.string) orelse return error.LMFailure
                        else
                            return error.LMFailure;
                    }
                }
            }
            if (!recognized) return error.LMFailure;
        }
        return init(result);
    }

    fn overridesToValue(arena: Allocator, overrides: []const Override) Error!runtime.Value {
        var rows = std.array_list.Managed(runtime.Value).init(arena);
        try rows.ensureTotalCapacityPrecise(overrides.len);
        for (overrides) |entry| {
            var knobs: runtime.ObjectMap = .empty;
            for (entry.knobs) |knob| {
                try knobs.put(arena, @tagName(knob.name), switch (knob.value) {
                    .inherit => .null,
                    .word => |word| .{ .string = try arena.dupe(u8, word) },
                    .levels => |levels| try effortsToValue(arena, levels),
                });
            }
            var pair = std.array_list.Managed(runtime.Value).init(arena);
            try pair.appendSlice(&.{
                .{ .string = try arena.dupe(u8, entry.prefix) },
                .{ .object = knobs },
            });
            rows.appendAssumeCapacity(.{ .array = pair });
        }
        return .{ .array = rows };
    }

    fn overridesFromValue(arena: Allocator, encoded: runtime.Value) Error![]const Override {
        if (encoded != .array) return error.LMFailure;
        const rows = try arena.alloc(Override, encoded.array.items.len);
        for (rows, encoded.array.items) |*target, row| {
            if (row != .array or row.array.items.len != 2) return error.LMFailure;
            const prefix, const fields = .{ row.array.items[0], row.array.items[1] };
            if (prefix != .string or fields != .object) return error.LMFailure;
            const knobs = try arena.alloc(Knob, fields.object.count());
            for (knobs, fields.object.keys(), fields.object.values()) |*knob, name, value| {
                knob.* = .{
                    .name = std.meta.stringToEnum(Knob.Name, name) orelse return error.LMFailure,
                    .value = switch (value) {
                        .null => .inherit,
                        .string => |word| .{ .word = word },
                        .array => .{ .levels = try effortsFromValue(arena, value) },
                        else => return error.LMFailure,
                    },
                };
            }
            target.* = .{ .prefix = prefix.string, .knobs = knobs };
        }
        return rows;
    }

    fn effortsToValue(arena: Allocator, levels: []const sdk.ReasoningEffort) Error!runtime.Value {
        var words = std.array_list.Managed(runtime.Value).init(arena);
        try words.ensureTotalCapacityPrecise(levels.len);
        for (levels) |level| words.appendAssumeCapacity(.{ .string = @tagName(level) });
        return .{ .array = words };
    }

    fn effortsFromValue(arena: Allocator, encoded: runtime.Value) Error![]const sdk.ReasoningEffort {
        if (encoded != .array) return error.LMFailure;
        const levels = try arena.alloc(sdk.ReasoningEffort, encoded.array.items.len);
        for (levels, encoded.array.items) |*level, word| {
            if (word != .string) return error.LMFailure;
            level.* = std.meta.stringToEnum(sdk.ReasoningEffort, word.string) orelse
                return error.LMFailure;
        }
        return levels;
    }

    pub fn resolve(self: *const ChatCompat) Error!Compat {
        try self.validate();
        var result = Compat.defaults(.chat);
        inline for (.{
            "instruction_role",            "tool_result_name",  "thinking_replay",
            "assistant_reasoning_content", "tool_result_media",
        }) |name| {
            if (@field(self, name)) |value| {
                const target = &@field(result.content_policy, name);
                target.* = try resolveWord(@TypeOf(target.*), value, target.*);
            }
        }
        inline for (.{
            "max_tokens_field", "stream_usage",       "assistant_after_tool_result",
            "strict_tools",     "builtin_tools",      "cache_control",
            "user_field",       "forced_tool_choice", "reasoning_off",
            "json_schema",      "token_scoring",      "reasoning_usage",
        }) |name| {
            if (@field(self, name)) |value| {
                const target = &@field(result, name);
                target.* = try resolveWord(@TypeOf(target.*), value, target.*);
            }
        }
        if (self.thinking_format) |value| {
            const target = &result.reasoning_format;
            target.* = try resolveWord(@TypeOf(target.*), value, target.*);
        }
        result.reasoning_efforts = self.reasoning_efforts;
        result.routing = self.routing;
        return result;
    }

    fn resolveWord(comptime T: type, word: anytype, default: T) Error!T {
        return policyWord(T, .{ .string = @tagName(word) }, default, default);
    }
};

const ChatPreset = @import("openai_chat_presets.zig").Table(ChatCompat);

pub fn chatPreset(name: []const u8) error{UnknownChatPreset}!ChatPreset {
    return presetLookup(ChatPreset, ChatPreset.names, name) orelse error.UnknownChatPreset;
}

/// The endpoint name and policy identity are separate: LM Studio shares
/// Ollama's policy, but its default port is not Ollama's.
pub const ResponsesPreset = enum {
    openai,
    openrouter,
    ollama,
    lmstudio,
    vllm,
    sglang,
    qwen,
    deepseek,
    zai,
    meta,
    moonshotai,

    const names = std.StaticStringMap(ResponsesPreset).initComptime(.{
        .{ "openai", .openai },
        .{ "openrouter", .openrouter },
        .{ "ollama", .ollama },
        .{ "lmstudio", .lmstudio },
        .{ "vllm", .vllm },
        .{ "sglang", .sglang },
        .{ "qwen", .qwen },
        .{ "deepseek", .deepseek },
        .{ "zai", .zai },
        .{ "meta", .meta },
        .{ "moonshotai", .moonshotai },
        .{ "openai_chat", .openai },
        .{ "chat", .openai },
        .{ "chat_completions", .openai },
        .{ "responses", .openai },
        .{ "openai_responses", .openai },
        .{ "lm_studio", .lmstudio },
        .{ "dashscope_qwen", .qwen },
        .{ "z_ai", .zai },
    });

    pub fn fromName(name: []const u8) error{UnknownResponsesPreset}!ResponsesPreset {
        return presetLookup(ResponsesPreset, names, name) orelse error.UnknownResponsesPreset;
    }

    const policies = std.enums.EnumArray(ResponsesPreset, ResponsesCompat).init(.{
        .openai = openai_policy,
        .openrouter = .{
            .developer_role = .developer,
            .max_output_tokens_field = .max_tokens,
            .reasoning_format = .openrouter,
            .tool_result_name = .omit,
            .strict_tools = .omit,
            .cache_control = .openai,
            .tool_result_media = .reject,
        },
        .ollama = localPolicy(.none),
        .lmstudio = .{}, // policy() uses the Ollama entry instead.
        .vllm = localPolicy(.reasoning_effort),
        .sglang = localPolicy(.reasoning_effort),
        .qwen = localPolicy(.qwen),
        .deepseek = localPolicy(.deepseek),
        .zai = localPolicy(.zai),
        .meta = blk: {
            var value = openai_policy;
            value.cache_control = .openai_implicit;
            value.commentary_phase = .tag;
            value.edit_image_field = .indexed;
            value.builtin_tools = .verbatim;
            value.tool_result_media = .native;
            break :blk value;
        },
        .moonshotai = blk: {
            var value = openai_policy;
            value.cache_control = .openai_implicit;
            value.builtin_tools = .verbatim;
            value.tool_result_media = .images;
            break :blk value;
        },
    });

    const openai_policy: ResponsesCompat = .{
        .developer_role = .developer,
        .max_output_tokens_field = .max_output_tokens,
        .reasoning_format = .responses_reasoning,
        .tool_result_name = .omit,
        .strict_tools = .omit,
        .cache_control = .openai,
    };

    fn localPolicy(reasoning: @FieldType(ResponsesCompat, "reasoning_format")) ResponsesCompat {
        return .{
            .developer_role = .system,
            .max_output_tokens_field = .max_tokens,
            .reasoning_format = reasoning,
            .tool_result_name = .omit,
            .strict_tools = .omit,
            .cache_control = .none,
            .tool_result_media = .reject,
        };
    }

    pub fn policy(self: ResponsesPreset) *const ResponsesCompat {
        return policies.getPtrConst(if (self == .lmstudio) .ollama else self);
    }

    pub const default_openai_url = "https://api.openai.com/v1";

    pub fn defaultUrl(self: ResponsesPreset) error{MissingResponsesPresetBaseUrl}![]const u8 {
        return switch (self) {
            .openai => default_openai_url,
            .openrouter => "https://openrouter.ai/api/v1",
            .ollama => "http://localhost:11434/v1",
            .lmstudio => "http://localhost:1234/v1",
            .vllm => "http://localhost:8000/v1",
            .sglang => "http://localhost:30000/v1",
            .meta => "https://api.meta.ai/v1",
            .moonshotai => "https://api.moonshot.ai/v1",
            .qwen, .deepseek, .zai => error.MissingResponsesPresetBaseUrl,
        };
    }
};

/// Responses policy overrides are selected before credentials, just like the
/// wire fields. Null inherits; "auto" resets a knob to the dialect default.
pub fn compatibility(
    dialect: Dialect,
    bound: ?*const ResponsesCompat,
    extensions: ?sdk.JsonValue,
) Error!Compat {
    var resolved = if (dialect == .responses and bound != null)
        try bound.?.resolve()
    else
        Compat.defaults(dialect);
    if (dialect == .chat) return resolved;
    const extra = extensions orelse return resolved;
    var patch = extra.root.object.get("openai_responses_compat") orelse Value.null;
    if (patch == .null) patch = extra.root.object.get("openai_compat") orelse .null;
    if (patch == .null) {
        const nested = extra.root.object.get("compat") orelse Value.null;
        if (nested == .object) {
            patch = nested.object.get("openai_responses") orelse .null;
            if (!(graph.truthy(patch) catch return error.LMFailure))
                patch = nested.object.get("openai") orelse .null;
        }
    }
    if (patch != .object) return resolved;
    const defaults = Compat.defaults(.responses);
    inline for (.{
        .{ "developer_role", "instruction_role" },
        .{ "tool_result_name", "tool_result_name" },
        .{ "tool_result_media", "tool_result_media" },
        .{ "commentary_phase", "commentary_phase" },
    }) |names| {
        if (patch.object.get(names[0])) |value| {
            const target = &@field(resolved.content_policy, names[1]);
            target.* = try policyWord(@TypeOf(target.*), value, @field(defaults.content_policy, names[1]), target.*);
        }
    }
    inline for (.{
        .{ "max_output_tokens_field", "max_tokens_field" },
        .{ "reasoning_format", "reasoning_format" },
        .{ "strict_tools", "strict_tools" },
        .{ "cache_control", "cache_control" },
        .{ "builtin_tools", "builtin_tools" },
    }) |names| {
        if (patch.object.get(names[0])) |value| {
            // These two enum domains also contain Chat-only policies.
            if (comptime std.mem.eql(u8, names[0], "reasoning_format")) {
                if (value == .string and std.mem.eql(u8, value.string, "kimi")) return error.LMFailure;
            }
            if (comptime std.mem.eql(u8, names[0], "builtin_tools")) {
                if (value == .string and
                    (std.mem.eql(u8, value.string, "reject") or std.mem.eql(u8, value.string, "groq"))) return error.LMFailure;
            }
            const target = &@field(resolved, names[1]);
            target.* = try policyWord(@TypeOf(target.*), value, @field(defaults, names[1]), target.*);
        }
    }
    // Image-edit policy has no effect on a generation request, but the Source
    // compatibility constructor still validates it before any credentials.
    if (patch.object.get("edit_image_field")) |value|
        _ = try policyWord(enum { array, indexed }, value, .array, .array);
    inline for (.{ "routing", "extensions" }) |name| {
        if (patch.object.get(name)) |value| {
            if (value != .null and value != .object) return error.LMFailure;
            if (comptime std.mem.eql(u8, name, "routing")) {
                if (value != .null) resolved.routing = value;
            }
        }
    }
    return resolved;
}

fn presetLookup(comptime T: type, comptime names: std.StaticStringMap(T), name: []const u8) ?T {
    var key: [names.max_len]u8 = undefined;
    // Unicode lowercase maps only U+212A outside ASCII into an ASCII
    // spelling. All admitted names are ASCII keys.
    if (name.len > key.len * 3) return null;
    var remaining = name;
    var length: usize = 0;
    while (remaining.len != 0) : (length += 1) {
        if (length == key.len) return null;
        const byte = remaining[0];
        if (byte < 128) {
            key[length] = switch (byte) {
                '-', ' ', '.' => '_',
                else => std.ascii.toLower(byte),
            };
            remaining = remaining[1..];
        } else {
            const kelvin_sign = "\u{212a}";
            if (!std.mem.startsWith(u8, remaining, kelvin_sign)) return null;
            key[length] = 'k';
            remaining = remaining[kelvin_sign.len..];
        }
    }
    return names.get(key[0..length]);
}

fn policyWord(comptime T: type, value: Value, default: T, inherited: T) Error!T {
    if (value == .null) return inherited;
    if (value != .string) return error.LMFailure;
    if (std.mem.eql(u8, value.string, "auto")) return default;
    return std.meta.stringToEnum(T, value.string) orelse error.LMFailure;
}

pub const Options = struct {
    dialect: Dialect,
    compat: Compat,
    adaptations: Adaptations,
    max_media_bytes: usize,
    stream: bool = false,
};
pub const Wire = struct {
    body: []const u8,
    adaptations: []const sdk.Adaptation,
};

/// Prepared wrappers snapshot field selection while retaining the original
/// nested JSON nodes. Credentials may mutate those nodes before serialization.
pub const Prepared = struct {
    payload: sdk.JsonValue,
    adaptations: []const sdk.Adaptation,

    pub fn body(self: *const Prepared, arena: Allocator) Error![]const u8 {
        var encoded: std.Io.Writer.Allocating = .init(arena);
        self.payload.writeJsonMinified(&encoded.writer) catch |err| return mapError(err);
        return encoded.written();
    }

    pub fn prepareBody(context: *anyopaque, arena: Allocator) anyerror![]const u8 {
        const self: *const Prepared = @ptrCast(@alignCast(context));
        return self.body(arena);
    }
};

pub fn build(arena: Allocator, io: std.Io, request: *const sdk.Request, options: Options) Error!Wire {
    const prepared = try prepare(arena, io, request, options);
    return .{ .body = try prepared.body(arena), .adaptations = prepared.adaptations };
}

pub fn prepare(arena: Allocator, io: std.Io, request: *const sdk.Request, options: Options) Error!Prepared {
    var builder: Builder = .{ .arena = arena, .options = options, .payload = try newObject(arena) };
    const breakpoint = try builder.breakpointIndex(request);
    try builder.probabilities(request);
    const cache = request.fields.config.cache;
    const stable = options.compat.cache_control == .openai and cache != null and
        cache.?.mode != .off and cache.?.prefix == .stable;
    const messages = try content.messages(arena, io, request, .{
        .dialect = options.dialect,
        .policy = options.compat.content_policy,
        .max_media_bytes = options.max_media_bytes,
        .breakpoint_index = breakpoint,
        .stable_prefix = stable,
    });
    try builder.put("model", .{ .string = wireModel(request.fields.model, options.dialect) });
    try builder.put(if (options.dialect == .chat) "messages" else "input", graph.fromValue(arena, .{ .array = messages }) catch |err| return mapError(err));
    if (options.dialect == .responses) {
        try builder.put("stream", .{ .bool = options.stream });
        if (!stable) {
            if (request.fields.system) |system| {
                if (content.systemPresent(system))
                    try builder.put("instructions", .{ .string = try content.systemText(arena, system) });
            }
        }
    } else if (options.stream) {
        try builder.put("stream", .{ .bool = true });
        if (options.compat.stream_usage == .include) {
            const usage = try object(arena, &.{.{ "include_usage", .{ .bool = true } }});
            try builder.put("stream_options", usage);
        }
    }
    try builder.controls(&request.fields.config);
    try builder.tools(request);
    try builder.format(request.fields.config.response_format);
    if (request.fields.config.reasoning) |reasoning| try builder.reasoning(&reasoning);
    try builder.cachePayload(request, breakpoint, stable);
    if (options.compat.routing) |routing| try builder.put("provider", routing);
    const config = &request.fields.config;
    if (config.service_tier) |tier| try builder.put("service_tier", .{ .string = tier });
    if (config.user_id) |user| try builder.put(@tagName(options.compat.user_field), .{ .string = user });
    if (config.store) |store| try builder.put("store", .{ .bool = store });
    if (config.extensions) |extensions| {
        const extra = extensions.root;
        var fields = extra.object.iterator();
        while (fields.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key, "prompt_caching") or std.mem.eql(u8, key, "cache") or
                std.mem.eql(u8, key, "compat") or std.mem.eql(u8, key, "openai_compat") or
                std.mem.eql(u8, key, if (options.dialect == .chat) "openai_chat_compat" else "openai_responses_compat")) continue;
            try builder.put(key, entry.value_ptr.*);
        }
    }
    return .{ .payload = .borrowed(.{ .object = builder.payload }), .adaptations = builder.records.items };
}

pub fn wireModel(model: []const u8, dialect: Dialect) []const u8 {
    const colon = std.mem.indexOfScalar(u8, model, ':') orelse return model;
    if (colon + 1 == model.len) return model;
    const prefix = model[0..colon];
    if (dialect == .responses and std.mem.eql(u8, prefix, "openai")) return model[colon + 1 ..];
    if (dialect == .chat and (std.mem.eql(u8, prefix, "openai_chat") or std.mem.eql(u8, prefix, "openai-chat")))
        return model[colon + 1 ..];
    return model;
}

const Builder = struct {
    arena: Allocator,
    options: Options,
    payload: *Object,
    records: std.ArrayList(sdk.Adaptation) = .empty,

    fn put(self: *Builder, name: []const u8, value: Value) Error!void {
        try self.payload.put(self.arena, name, value);
    }

    fn adapt(self: *Builder, field: []const u8, action: sdk.AdaptationAction, reason: []const u8, asked: Value, applied: Value) Error!void {
        if (self.options.adaptations == .refuse) return error.LMFailure;
        try self.records.append(self.arena, .{
            .field = field,
            .action = action,
            .reason = reason,
            .asked = .borrowed(asked),
            .applied = .borrowed(applied),
        });
    }

    fn controls(self: *Builder, config: *const sdk.Config) Error!void {
        const chat = self.options.dialect == .chat;
        if (config.max_tokens != .null) try self.put(@tagName(self.options.compat.max_tokens_field), scalar(config.max_tokens));
        if (config.temperature) |n| try self.put("temperature", .{ .float = n });
        if (config.top_p) |n| try self.put("top_p", .{ .float = n });
        if (!chat and config.stop.len != 0) {
            const stop = try strings(self.arena, config.stop);
            try self.adapt("config.stop", .client_side, "the Responses wire has no stop field; the reply is streamed and the connection " ++
                "closed at the first stop sequence (whether the provider then stops generating, " ++
                "and billing, is its own behaviour); the usage report rides only the final frame, " ++
                "so it is not reported when the cut happens (never estimated)", stop, try strings(self.arena, config.stop));
        }
        if (config.top_k != .null) try self.adapt("config.top_k", .dropped, if (chat) "the Chat Completions wire has no top_k (Anthropic and Gemini carry it; servers " ++
            "that accept it take it through extensions)" else "the Responses wire has no top_k (Anthropic and Gemini carry it)", scalar(config.top_k), .null);
        inline for (.{ "seed", "frequency_penalty", "presence_penalty" }) |name| {
            const value = scalar(@field(config, name));
            if (value != .null) {
                if (chat) try self.put(name, value) else try self.adapt("config." ++ name, .dropped, "the Responses wire has no " ++ name ++ " field (the Chat Completions dialect carries it)", value, .null);
            }
        }
        if (chat and config.stop.len != 0) try self.put("stop", try strings(self.arena, config.stop));
        if (config.logprobs != .null) {
            if (chat) {
                try self.put("logprobs", .{ .bool = true });
                if ((@import("value.zig").numericOrder(config.logprobs, .{ .integer = 0 }) catch return error.LMFailure) == .gt)
                    try self.put("top_logprobs", scalar(config.logprobs));
            } else {
                try self.put("top_logprobs", scalar(config.logprobs));
                try self.put("include", try strings(self.arena, &.{"message.output_text.logprobs"}));
            }
        }
    }

    fn probabilities(self: *Builder, request: *const sdk.Request) Error!void {
        const config = &request.fields.config;
        const policy = config.probabilities orelse return;
        if (policy == .off) return;
        const format_value = config.response_format orelse return;
        const format_config = format_value.root;
        if (!isString(format_config.object.get("type") orelse .null, "json_schema")) return;
        const schema = graph.toValue(self.arena, format_config.object.get("schema") orelse return) catch |err| return mapError(err);
        if (!try @import("typesafe.zig").hasMeasurableJudgments(self.arena, schema)) return;
        if (policy == .required) return error.LMFailure;
        const streaming_scores = self.options.dialect == .chat and self.options.stream and
            self.options.compat.token_scoring == .logprob_token_ids;
        const reason = if (streaming_scores)
            "stream() uses generated JSON, not candidate scoring; the answer carries an unmeasured pick only; use complete() for scoring"
        else
            "this wire cannot measure a distribution over the declared keys; the answer carries the pick only";
        try self.adapt("config.probabilities", .dropped, reason, .{ .string = @tagName(policy) }, .null);
    }

    fn tools(self: *Builder, request: *const sdk.Request) Error!void {
        if (request.fields.tools.len != 0) {
            const tools_wire = try newArray(self.arena);
            for (request.fields.tools) |*tool| {
                const entry = try newObject(self.arena);
                switch (tool.*) {
                    .function => |function| {
                        const fields = if (self.options.dialect == .chat) try newObject(self.arena) else entry;
                        if (self.options.dialect == .responses) try fields.put(self.arena, "type", .{ .string = "function" });
                        try fields.put(self.arena, "name", .{ .string = function.name });
                        if (function.description) |description| {
                            if (description.len != 0) try fields.put(self.arena, "description", .{ .string = description });
                        }
                        const parameters = function.parameters.?.root;
                        try fields.put(self.arena, "parameters", parameters);
                        if (self.options.compat.strict_tools == .include) try fields.put(self.arena, "strict", .{ .bool = false });
                        if (self.options.dialect == .chat) {
                            try entry.put(self.arena, "type", .{ .string = "function" });
                            try entry.put(self.arena, "function", .{ .object = fields });
                        }
                    },
                    .builtin => |builtin| {
                        try entry.put(self.arena, "type", .{ .string = try self.builtinType(builtin.name) });
                        if (builtin.config) |config| {
                            const extra = config.root;
                            var iterator = extra.object.iterator();
                            while (iterator.next()) |field| try entry.put(self.arena, field.key_ptr.*, field.value_ptr.*);
                        }
                    },
                }
                try tools_wire.append(.{ .object = entry });
            }
            try self.put("tools", .{ .array = tools_wire });
        }
        const choice = request.fields.config.tool_choice orelse return;
        var selected = try self.toolChoice(request, &choice);
        if (self.options.dialect == .chat and self.options.compat.forced_tool_choice == .reject and
            (choice.mode != .auto or choice.allowed.len != 0))
        {
            if (choice.mode == .required) return error.LMFailure;
            if (choice.mode == .none) {
                try self.adapt("config.tool_choice.mode", .client_side, "this server ignores tool_choice='none'; no tools were sent, which is the same outcome", .{ .string = "none" }, .{ .string = "no tools sent" });
                _ = self.payload.swapRemove("tools");
            } else {
                const kept = try newArray(self.arena);
                const names = try newArray(self.arena);
                if (self.payload.get("tools")) |tools_wire| {
                    for (tools_wire.array.items) |tool| {
                        const function = tool.object.get("function") orelse continue;
                        const name = function.object.get("name").?.string;
                        for (choice.allowed) |allowed| {
                            if (std.mem.eql(u8, name, allowed)) {
                                try kept.append(tool);
                                try names.append(.{ .string = name });
                                break;
                            }
                        }
                    }
                }
                try self.adapt("config.tool_choice.allowed", .client_side, "this server ignores tool_choice allowlists; only the allowed tools were sent, " ++
                    "which is what the allowlist means", try strings(self.arena, choice.allowed), .{ .array = names });
                try self.put("tools", .{ .array = kept });
            }
            selected = .{ .string = "auto" };
        }
        try self.put("tool_choice", selected);
        if (choice.parallel) |parallel| try self.put("parallel_tool_calls", .{ .bool = parallel });
    }

    fn builtinType(self: *const Builder, name: []const u8) Error![]const u8 {
        switch (self.options.compat.builtin_tools) {
            .reject => return error.LMFailure,
            .verbatim => return name,
            .groq => {
                if (std.mem.eql(u8, name, "web_search")) return "browser_search";
                if (std.mem.eql(u8, name, "code_execution")) return "code_interpreter";
                return error.LMFailure;
            },
            .openai => {
                inline for (.{ .{ "web_search", "web_search_preview" }, .{ "code_execution", "code_interpreter" }, .{ "computer_use", "computer_use_preview" } }) |pair|
                    if (std.mem.eql(u8, name, pair[0])) return pair[1];
                return name;
            },
        }
    }

    fn namedTool(self: *const Builder, tool: *const sdk.RequestTool) Error!Value {
        return switch (tool.*) {
            .builtin => |builtin| if (self.options.dialect == .chat) error.LMFailure else object(self.arena, &.{.{ "type", .{ .string = try self.builtinType(builtin.name) } }}),
            .function => |function| blk: {
                const named = try object(self.arena, &.{.{ "name", .{ .string = function.name } }});
                break :blk if (self.options.dialect == .chat)
                    try object(self.arena, &.{ .{ "type", .{ .string = "function" } }, .{ "function", named } })
                else
                    try object(self.arena, &.{ .{ "type", .{ .string = "function" } }, .{ "name", .{ .string = function.name } } });
            },
        };
    }

    fn toolChoice(self: *const Builder, request: *const sdk.Request, choice: *const sdk.ToolChoice) Error!Value {
        if (choice.mode == .none) return .{ .string = "none" };
        if (choice.allowed.len == 0) return .{ .string = @tagName(choice.mode) };
        const selected = try newArray(self.arena);
        for (choice.allowed) |name| {
            const tool = for (request.fields.tools) |*candidate| {
                const candidate_name = switch (candidate.*) {
                    inline else => |value| value.name,
                };
                if (std.mem.eql(u8, name, candidate_name)) break candidate;
            } else return error.LMFailure;
            try selected.append(try self.namedTool(tool));
        }
        if (choice.mode == .required and selected.items.len == 1) return selected.items[0];
        const subset = try object(self.arena, &.{ .{ "mode", .{ .string = @tagName(choice.mode) } }, .{ "tools", .{ .array = selected } } });
        return if (self.options.dialect == .chat)
            object(self.arena, &.{ .{ "type", .{ .string = "allowed_tools" } }, .{ "allowed_tools", subset } })
        else
            object(self.arena, &.{ .{ "type", .{ .string = "allowed_tools" } }, .{ "mode", .{ .string = @tagName(choice.mode) } }, .{ "tools", .{ .array = selected } } });
    }

    fn format(self: *Builder, format_value: ?sdk.JsonValue) Error!void {
        const value = format_value orelse return;
        const config = value.root;
        if (self.options.dialect == .chat and self.options.compat.json_schema == .reject and
            !isString(config.object.get("type").?, "json_object"))
        {
            try self.adapt("config.response_format", .dropped, "this server accepts response_format type 'json_schema' " ++
                "and does not apply it; use {'type': 'json_object'} and describe the shape in the prompt", config, .null);
            return;
        }
        const fields = try newObject(self.arena);
        if (isString(config.object.get("type").?, "json_object")) {
            try fields.put(self.arena, "type", .{ .string = "json_object" });
        } else {
            if (self.options.dialect == .responses) try fields.put(self.arena, "type", .{ .string = "json_schema" });
            const name = config.object.get("name") orelse Value{ .string = "response" };
            try fields.put(self.arena, "name", name);
            try fields.put(self.arena, "schema", config.object.get("schema").?);
            if (config.object.get("strict")) |strict| try fields.put(self.arena, "strict", strict);
            if (self.options.dialect == .chat) {
                const nested = try object(self.arena, &.{ .{ "type", .{ .string = "json_schema" } }, .{ "json_schema", .{ .object = fields } } });
                try self.put("response_format", nested);
                return;
            }
        }
        if (self.options.dialect == .responses)
            try self.put("text", try object(self.arena, &.{.{ "format", .{ .object = fields } }}))
        else
            try self.put("response_format", .{ .object = fields });
    }

    fn reasoning(self: *Builder, config: *const sdk.Reasoning) Error!void {
        const chat = self.options.dialect == .chat;
        const dialect = self.options.compat.reasoning_format;
        // Dropping the whole dial precedes every subfield adaptation.
        if (chat and dialect == .none) {
            try self.adapt("config.reasoning", .dropped, "this server has no reasoning dial on its wire (compat thinking_format='none'); " ++
                "the model reasons at its own default; pass the server's own knob through extensions", try object(self.arena, &.{.{ "effort", if (config.effort) |value| Value{ .string = @tagName(value) } else .null }}), .null);
            return;
        }
        var selected = config.effort;
        if (chat and selected == .off and self.options.compat.reasoning_off == .lowest) {
            const levels = self.options.compat.reasoning_efforts;
            selected = if (levels != null and levels.?.len != 0) levels.?[0] else .low;
            try self.adapt("config.reasoning.effort", .substituted, "this model cannot stop reasoning and the server accepts 'none' and reasons anyway " ++
                "(a paid no-op); the lowest level was sent", .{ .string = "off" }, .{ .string = @tagName(selected.?) });
        }
        const off = selected == .off;
        var effort: Value = if (off) .{ .string = "none" } else if (selected) |value| .{ .string = @tagName(value) } else .null;
        if (!off) {
            if (config.thinking_budget != .null) try self.adapt("config.reasoning.thinking_budget", .dropped, if (chat) "the Chat Completions wire has no thinking token budget; effort carries the intent" else "this wire has no thinking token budget; effort carries the intent " ++
                "(Anthropic's manual class and Gemini take a budget)", scalar(config.thinking_budget), .null);
            if (config.summary == .concise or config.summary == .detailed) {
                if (chat or dialect != .responses_reasoning) try self.adapt("config.reasoning.summary", .substituted, if (chat) "the Chat Completions wire has no summary detail levels; 'auto' is what it shows" else "this wire has no summary detail levels; 'auto' is what it shows", .{ .string = @tagName(config.summary.?) }, .{ .string = "auto" });
            }
        }
        if (chat and !off) {
            if (self.options.compat.reasoning_efforts) |levels| {
                const nearest = try nearestEffort(selected, levels);
                if (nearest != selected) {
                    try self.adapt("config.reasoning.effort", .clamped, try self.effortReason(selected, levels), effort, .{ .string = @tagName(nearest) });
                    effort = .{ .string = @tagName(nearest) };
                }
            }
            if (self.options.compat.builtin_tools == .groq and config.summary != null)
                try self.put("reasoning_format", .{ .string = "parsed" });
        }
        switch (dialect) {
            .responses_reasoning => {
                const fields = try newObject(self.arena);
                try fields.put(self.arena, "effort", effort);
                if (!off) if (config.summary) |summary| try fields.put(self.arena, "summary", .{ .string = @tagName(summary) });
                try self.put("reasoning", .{ .object = fields });
            },
            .reasoning_effort, .kimi => {
                if (off and dialect == .kimi)
                    try self.put("thinking", try object(self.arena, &.{.{ "type", .{ .string = "disabled" } }}))
                else
                    try self.put("reasoning_effort", effort);
            },
            .openrouter => try self.put("reasoning", if (off)
                try object(self.arena, &.{.{ "enabled", .{ .bool = false } }})
            else
                try object(self.arena, &.{.{ "effort", effort }})),
            .deepseek => {
                try self.put("thinking", try object(self.arena, &.{.{ "type", .{ .string = if (off) "disabled" else "enabled" } }}));
                if (!off) try self.put("reasoning_effort", effort);
            },
            .qwen, .zai => try self.put("enable_thinking", .{ .bool = !off }),
            .qwen_chat_template => {
                const fields = try newObject(self.arena);
                try fields.put(self.arena, "enable_thinking", .{ .bool = !off });
                if (!off) try fields.put(self.arena, "preserve_thinking", .{ .bool = true });
                try self.put("chat_template_kwargs", .{ .object = fields });
            },
            .none => {},
        }
    }

    fn effortReason(self: *const Builder, asked: ?sdk.ReasoningEffort, levels: []const sdk.ReasoningEffort) Error![]const u8 {
        var reason: std.Io.Writer.Allocating = .init(self.arena);
        if (asked) |level| {
            reason.writer.print("this server has no '{s}' level (it accepts ", .{@tagName(level)}) catch return error.OutOfMemory;
        } else {
            reason.writer.writeAll("this server has no None level (it accepts ") catch return error.OutOfMemory;
        }
        for (levels, 0..) |level, index| {
            if (index != 0) reason.writer.writeAll(", ") catch return error.OutOfMemory;
            reason.writer.writeAll(@tagName(level)) catch return error.OutOfMemory;
        }
        reason.writer.writeAll(") and would have accepted the word silently") catch return error.OutOfMemory;
        return reason.written();
    }

    fn breakpointIndex(self: *Builder, request: *const sdk.Request) Error!?usize {
        const cache = request.fields.config.cache orelse return null;
        if (cache.mode == .off or cache.prefix_until_index == .null or self.options.compat.cache_control != .openai) return null;
        const messages = request.fields.messages;
        const end: i64 = @as(i64, @intCast(messages.len)) - 1;
        const asked = if (cache.prefix_until_index == .integer) @min(cache.prefix_until_index.integer, end) else end;
        var index = asked;
        while (index >= 0) : (index -= 1) {
            const message = &messages[@intCast(index)];
            if (message.role == .assistant or message.role == .tool or message.parts.len == 0) continue;
            if (message.parts[message.parts.len - 1] != .text) continue;
            if (index != asked) try self.adapt("config.cache.prefix_until_index", .substituted, try std.fmt.allocPrint(self.arena, "message {d} is a {s} message or does not end with " ++
                "text; the Responses wire marks text blocks of user/developer messages only, so the " ++
                "mark moved to the nearest eligible message before it", .{ asked, @tagName(messages[@intCast(asked)].role) }), .{ .integer = asked }, .{ .integer = index });
            return @intCast(index);
        }
        try self.adapt("config.cache.prefix_until_index", .dropped, try std.fmt.allocPrint(self.arena, "no user/developer message ending with text at or before message {d}; the Responses " ++
            "wire marks text blocks only (implicit caching still applies)", .{asked}), .{ .integer = asked }, .null);
        return null;
    }

    fn cachePayload(self: *Builder, request: *const sdk.Request, breakpoint: ?usize, stable: bool) Error!void {
        const cache = request.fields.config.cache orelse return;
        const control = self.options.compat.cache_control;
        if (control != .openai and control != .openai_implicit) {
            if (cache.resource != null) return error.LMFailure;
            if (cache.key) |key| try self.adapt("config.cache.key", .dropped, "this server has no cache affinity field; implicit caching still applies", .{ .string = key }, .null);
            if (cache.retention == .long) try self.adapt("config.cache.retention", .dropped, "this server has no in-request cache lifetime knob; implicit caching still applies", .{ .string = "long" }, .null);
            return;
        }
        const model_has_options = hasCacheOptions(wireModel(request.fields.model, self.options.dialect));
        if (control == .openai and cache.mode == .off) {
            if (model_has_options) try self.put("prompt_cache_options", try object(self.arena, &.{.{ "mode", .{ .string = "explicit" } }}));
            return;
        }
        if (cache.mode != .off) {
            if (cache.key) |key| if (key.len != 0) {
                try self.put("prompt_cache_key", .{ .string = key });
            };
            if (cache.retention == .long) try self.put("prompt_cache_retention", .{ .string = "24h" });
        }
        const system_present = if (request.fields.system) |system| content.systemPresent(system) else false;
        if (control == .openai and model_has_options and (breakpoint != null or (stable and system_present)))
            try self.put("prompt_cache_options", try object(self.arena, &.{.{ "mode", .{ .string = "explicit" } }}));
        if (cache.resource != null) return error.LMFailure;
    }
};

pub fn nearestEffort(asked: ?sdk.ReasoningEffort, levels: []const sdk.ReasoningEffort) Error!sdk.ReasoningEffort {
    std.debug.assert(asked != .off);
    const wanted = @backingInt(asked orelse .minimal);
    var nearest: ?sdk.ReasoningEffort = null;
    var distance: u8 = std.math.maxInt(u8);
    for (levels) |level| {
        if (level == .off) return error.LMFailure;
        const ordinal = @backingInt(level);
        const delta = if (ordinal >= wanted) ordinal - wanted else wanted - ordinal;
        if (delta < distance or (delta == distance and ordinal < @backingInt(nearest.?))) {
            nearest = level;
            distance = delta;
        }
    }
    const result = nearest orelse return error.LMFailure;
    std.debug.assert(result != .off);
    return result;
}

fn hasCacheOptions(model: []const u8) bool {
    if (model.len < 7 or !std.ascii.eqlIgnoreCase(model[0..4], "gpt-")) return false;
    var end: usize = 4;
    while (end < model.len and std.ascii.isDigit(model[end])) : (end += 1) {}
    if (end == 4 or end == model.len or model[end] != '.') return false;
    const major = std.fmt.parseInt(u64, model[4..end], 10) catch std.math.maxInt(u64);
    const start = end + 1;
    end = start;
    while (end < model.len and std.ascii.isDigit(model[end])) : (end += 1) {}
    if (end == start) return false;
    const minor = std.fmt.parseInt(u64, model[start..end], 10) catch std.math.maxInt(u64);
    return major > 5 or (major == 5 and minor >= 6);
}

fn newObject(arena: Allocator) Error!*Object {
    const result = try arena.create(Object);
    result.* = .empty;
    return result;
}
fn newArray(arena: Allocator) Error!*Array {
    const result = try arena.create(Array);
    result.* = .init(arena);
    return result;
}
fn object(arena: Allocator, fields: []const struct { []const u8, Value }) Error!Value {
    const result = try newObject(arena);
    for (fields) |field| try result.put(arena, field[0], field[1]);
    return .{ .object = result };
}
fn strings(arena: Allocator, input: []const []const u8) Error!Value {
    const values = try newArray(arena);
    try values.ensureTotalCapacity(input.len);
    for (input) |text| try values.append(.{ .string = text });
    return .{ .array = values };
}
fn scalar(input: anytype) Value {
    if (@TypeOf(input) == runtime.Value) return switch (input) {
        inline .null, .bool, .integer, .float, .number_string, .string => |value, tag| @unionInit(Value, @tagName(tag), value),
        else => unreachable, // Config constructors admit only numeric scalars here.
    };
    return switch (@typeInfo(@TypeOf(input))) {
        .optional => if (input) |value| scalar(value) else .null,
        .float => .{ .float = input },
        else => @compileError("not an SDK configuration scalar"),
    };
}
fn isString(value: Value, expected: []const u8) bool {
    return value == .string and std.mem.eql(u8, value.string, expected);
}
fn mapError(err: anyerror) Error {
    return switch (err) {
        error.OutOfMemory, error.WriteFailed => error.OutOfMemory,
        else => error.LMFailure,
    };
}

test "OpenAI tool wire omits empty descriptions in both dialects" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    // DSPy 3.5.0's OpenAIChatLM and OpenAILM both use MAP-17
    // (providers/common.py:tool_description): omit None and "", not whitespace.
    const request = try sdk.Request.fromValue(arena, try std.json.parseFromSliceLeaky(runtime.Value, arena,
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"next"}]}],"tools":[{"type":"function","name":"missing","parameters":{"type":"object"}},{"type":"function","name":"empty","description":"","parameters":{"type":"object"}},{"type":"function","name":"whitespace","description":" ","parameters":{"type":"object"}},{"type":"function","name":"described","description":"Lookup café.","parameters":{"type":"object"}}]}
    , .{}));
    defer request.release();
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "missing", null },
        .{ "empty", null },
        .{ "whitespace", " " },
        .{ "described", "Lookup café." },
    };
    for ([_]Dialect{ .chat, .responses }) |dialect| {
        const wire = try build(arena, std.testing.io, request, .{
            .dialect = dialect,
            .compat = Compat.defaults(dialect),
            .adaptations = .note,
            .max_media_bytes = 1024,
        });
        const payload = try std.json.parseFromSliceLeaky(runtime.Value, arena, wire.body, .{});
        const tools_wire = payload.object.get("tools").?.array.items;
        try std.testing.expectEqual(cases.len, tools_wire.len);
        for (tools_wire, cases) |tool, expected| {
            try std.testing.expectEqualStrings("function", tool.object.get("type").?.string);
            const function = if (dialect == .chat) tool.object.get("function").? else tool;
            try std.testing.expectEqualStrings(expected[0], function.object.get("name").?.string);
            const parameters = function.object.get("parameters").?;
            try std.testing.expectEqualStrings("object", parameters.object.get("type").?.string);
            const description = function.object.get("description");
            if (expected[1]) |text| {
                try std.testing.expectEqualStrings(text, description.?.string);
            } else {
                try std.testing.expect(description == null);
            }
        }
    }
}

test "Chat compatibility drops the whole dial after client-side tool and schema controls" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    const request = try sdk.Request.fromValue(arena, try std.json.parseFromSliceLeaky(runtime.Value, arena,
        \\{"model":"fixture","messages":[{"role":"user","parts":[{"type":"text","text":"next"}]}],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"config":{"tool_choice":{"mode":"none"},"response_format":{"type":"json_schema","schema":{"type":"object"}},"reasoning":{"effort":"high","thinking_budget":7,"summary":"detailed"}}}
    , .{}));
    defer request.release();
    var compat = Compat.defaults(.chat);
    compat.forced_tool_choice = .reject;
    compat.json_schema = .reject;
    compat.reasoning_format = .none;
    const options: Options = .{ .dialect = .chat, .compat = compat, .adaptations = .note, .max_media_bytes = 1024 };
    const built = try build(arena, std.testing.io, request, options);
    const payload = try std.json.parseFromSliceLeaky(runtime.Value, arena, built.body, .{});
    try std.testing.expect(!payload.object.contains("tools"));
    try std.testing.expect(!payload.object.contains("response_format"));
    try std.testing.expect(!payload.object.contains("reasoning_effort"));
    try std.testing.expectEqualStrings("auto", payload.object.get("tool_choice").?.string);
    try std.testing.expectEqual(@as(usize, 3), built.adaptations.len);
    for (built.adaptations, [_][]const u8{ "config.tool_choice.mode", "config.response_format", "config.reasoning" }) |adaptation, field|
        try std.testing.expectEqualStrings(field, adaptation.field);
    try std.testing.expectEqual(sdk.AdaptationAction.client_side, built.adaptations[0].action);
    try std.testing.expectEqual(sdk.AdaptationAction.dropped, built.adaptations[1].action);
    try std.testing.expectEqual(sdk.AdaptationAction.dropped, built.adaptations[2].action);
    const asked = try built.adaptations[2].asked.toValue(arena);
    try std.testing.expectEqualStrings("high", asked.object.get("effort").?.string);
    var refusing = options;
    refusing.adaptations = .refuse;
    try std.testing.expectError(error.LMFailure, build(arena, std.testing.io, request, refusing));
}

test "Chat effort clamps ties downward but off uses the first declared level" {
    var memory: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer memory.deinit();
    const arena = memory.allocator();
    var compat = Compat.defaults(.chat);
    compat.reasoning_efforts = &.{ .high, .low };
    compat.reasoning_off = .lowest;
    for ([_]sdk.ReasoningEffort{ .medium, .off }, [_][]const u8{ "low", "high" }, [_]sdk.AdaptationAction{ .clamped, .substituted }) |effort, expected, action| {
        const request = try sdk.Request.fromFields(arena, &.{
            .model = "fixture",
            .messages = &.{.{ .role = .user, .parts = &.{.{ .text = .{ .text = "next" } }} }},
            .config = .{ .reasoning = .{ .effort = effort } },
        });
        defer request.release();
        const options: Options = .{ .dialect = .chat, .compat = compat, .adaptations = .note, .max_media_bytes = 1024 };
        const built = try build(arena, std.testing.io, request, options);
        const payload = try std.json.parseFromSliceLeaky(runtime.Value, arena, built.body, .{});
        try std.testing.expectEqualStrings(expected, payload.object.get("reasoning_effort").?.string);
        try std.testing.expectEqual(@as(usize, 1), built.adaptations.len);
        try std.testing.expectEqual(action, built.adaptations[0].action);
        try std.testing.expectEqualStrings(@tagName(effort), (try built.adaptations[0].asked.toValue(arena)).string);
        try std.testing.expectEqualStrings(expected, (try built.adaptations[0].applied.toValue(arena)).string);
        var empty = options;
        empty.compat.reasoning_efforts = &.{};
        try std.testing.expectError(error.LMFailure, build(arena, std.testing.io, request, empty));
    }
}
