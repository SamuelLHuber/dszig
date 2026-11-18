//! Named Chat Completions policies and their independent endpoint roots.
const std = @import("std");

pub fn Table(comptime ChatCompat: type) type {
    return enum {
        openai,
        ollama,
        lmstudio,
        groq,
        openrouter,
        xai,
        vllm,
        sglang,
        deepseek,
        qwen,
        bedrock,
        bedrock_mantle,
        zai,
        meta,
        moonshotai,
        deepinfra,
        together,
        fireworks,
        parasail,

        const Preset = @This();
        pub const names = std.StaticStringMap(Preset).initComptime(.{
            .{ "openai", .openai },
            .{ "ollama", .ollama },
            .{ "lmstudio", .lmstudio },
            .{ "groq", .groq },
            .{ "openrouter", .openrouter },
            .{ "xai", .xai },
            .{ "vllm", .vllm },
            .{ "sglang", .sglang },
            .{ "deepseek", .deepseek },
            .{ "qwen", .qwen },
            .{ "bedrock", .bedrock },
            .{ "bedrock_mantle", .bedrock_mantle },
            .{ "zai", .zai },
            .{ "meta", .meta },
            .{ "moonshotai", .moonshotai },
            .{ "deepinfra", .deepinfra },
            .{ "together", .together },
            .{ "fireworks", .fireworks },
            .{ "parasail", .parasail },
            .{ "openai_chat", .openai },
            .{ "chat", .openai },
            .{ "chat_completions", .openai },
            .{ "responses", .openai },
            .{ "openai_responses", .openai },
            .{ "lm_studio", .lmstudio },
            .{ "dashscope_qwen", .qwen },
            .{ "z_ai", .zai },
        });

        const policies = std.enums.EnumArray(Preset, ChatCompat).init(.{
            .openai = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .tool_result_media = .reject,
                .cache_control = .openai,
            }),
            .ollama = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .reasoning_effort,
                .tool_result_media = .reject,
                .cache_control = .none,
                .reasoning_efforts = &.{ .minimal, .low, .medium, .high, .xhigh, .max },
            }),
            .lmstudio = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .none,
                .tool_result_media = .reject,
                .cache_control = .none,
            }),
            .groq = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .reasoning_effort,
                .builtin_tools = .groq,
                .tool_result_media = .reject,
                .cache_control = .none,
            }),
            .openrouter = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .openrouter,
                .tool_result_media = .reject,
                .cache_control = .openai,
            }),
            .xai = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .deepseek,
                .tool_result_media = .images,
                .cache_control = .none,
                .reasoning_usage = .separate,
            }),
            .vllm = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .reasoning_effort,
                .tool_result_media = .reject,
                .cache_control = .none,
                .token_scoring = .logprob_token_ids,
            }),
            .sglang = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .reasoning_effort,
                .tool_result_media = .reject,
                .cache_control = .none,
            }),
            .deepseek = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .deepseek,
                .thinking_replay = .native,
                .assistant_reasoning_content = .include_empty,
                .tool_result_media = .reject,
                .cache_control = .none,
                .user_field = .user_id,
            }),
            .qwen = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .qwen,
                .cache_control = .none,
            }),
            .bedrock = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .tool_result_media = .reject,
                .cache_control = .none,
                .user_field = .user,
                .forced_tool_choice = .send,
                .json_schema = .send,
                .model_overrides = &.{
                    .{ .prefix = "openai.gpt-oss", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "reject" },
                        },
                        .{
                            .name = .json_schema,
                            .value = .{ .word = "reject" },
                        },
                    } },
                    .{ .prefix = "google.gemma", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "reject" },
                        },
                    } },
                },
            }),
            .bedrock_mantle = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .cache_control = .none,
                .user_field = .user,
                .forced_tool_choice = .send,
                .json_schema = .send,
                .model_overrides = &.{
                    .{ .prefix = "openai.gpt-oss", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "reject" },
                        },
                        .{
                            .name = .json_schema,
                            .value = .{ .word = "reject" },
                        },
                    } },
                },
            }),
            .zai = common(.{
                .max_tokens_field = .max_tokens,
                .thinking_format = .deepseek,
                .thinking_replay = .native,
                .tool_result_media = .images,
                .cache_control = .none,
                .user_field = .user_id,
                .forced_tool_choice = .reject,
                .json_schema = .reject,
            }),
            .meta = common(.{
                .instruction_role = .developer,
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .tool_result_media = .reject,
                .cache_control = .openai_implicit,
                .user_field = .safety_identifier,
            }),
            .moonshotai = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .kimi,
                .thinking_replay = .native,
                .tool_result_media = .images,
                .cache_control = .openai_implicit,
                .user_field = .safety_identifier,
                .reasoning_efforts = &.{ .low, .high, .max },
            }),
            .deepinfra = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .thinking_replay = .native,
                .tool_result_media = .reject,
                .cache_control = .none,
                .forced_tool_choice = .reject,
                .model_overrides = &.{
                    .{ .prefix = "openai/gpt-oss", .knobs = &.{
                        .{
                            .name = .reasoning_off,
                            .value = .{ .word = "lowest" },
                        },
                    } },
                    .{ .prefix = "deepseek-ai/DeepSeek-V3.2", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "deepseek-ai/DeepSeek-V4-Flash", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "deepseek-ai/DeepSeek-V4.1-Flash", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "zai-org/GLM-5.3-Flash", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "moonshotai/Kimi-K2.6", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "meta-llama/Llama-4-Scout-17B-16E-Instruct", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "Qwen/Qwen3.6-27B", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "Qwen/Qwen3-Next-80B-A3B-Instruct", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "nvidia/NVIDIA-Nemotron-3.5-Lightning", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "ibm-granite/granite-4.2-8b", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "XiaomiMiMo/MiMo-V2.6-Flash", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "tencent/Hy3", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "google/gemini-3.1-flash-lite", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                    .{ .prefix = "anthropic/claude-haiku-4-5", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "send" },
                        },
                    } },
                },
            }),
            .together = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .thinking_replay = .native,
                .tool_result_media = .reject,
                .cache_control = .none,
                .model_overrides = &.{
                    .{ .prefix = "openai/gpt-oss", .knobs = &.{
                        .{
                            .name = .forced_tool_choice,
                            .value = .{ .word = "reject" },
                        },
                        .{
                            .name = .reasoning_efforts,
                            .value = .{ .levels = &.{ .low, .medium, .high } },
                        },
                        .{
                            .name = .reasoning_off,
                            .value = .{ .word = "lowest" },
                        },
                    } },
                    .{ .prefix = "zai-org/GLM-5.3", .knobs = &.{
                        .{
                            .name = .reasoning_off,
                            .value = .{ .word = "lowest" },
                        },
                    } },
                },
            }),
            .fireworks = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .thinking_replay = .native,
                .tool_result_media = .images,
                .cache_control = .none,
            }),
            .parasail = common(.{
                .max_tokens_field = .max_completion_tokens,
                .thinking_format = .reasoning_effort,
                .thinking_replay = .native,
                .tool_result_media = .images,
                .cache_control = .none,
            }),
        });

        fn common(partial: ChatCompat) ChatCompat {
            var result = partial;
            result.instruction_role = partial.instruction_role orelse .system;
            result.stream_usage = .include;
            result.tool_result_name = .omit;
            result.strict_tools = .omit;
            return result;
        }

        pub fn policy(self: Preset) *const ChatCompat {
            return policies.getPtrConst(self);
        }

        pub fn defaultUrl(self: Preset) error{MissingChatPresetBaseUrl}![]const u8 {
            return switch (self) {
                .openai => "https://api.openai.com/v1",
                .ollama => "http://localhost:11434/v1",
                .lmstudio => "http://localhost:1234/v1",
                .groq => "https://api.groq.com/openai/v1",
                .openrouter => "https://openrouter.ai/api/v1",
                .xai => "https://api.x.ai/v1",
                .vllm => "http://localhost:8000/v1",
                .sglang => "http://localhost:30000/v1",
                .deepseek => "https://api.deepseek.com",
                .zai => "https://api.z.ai/api/paas/v4",
                .meta => "https://api.meta.ai/v1",
                .moonshotai => "https://api.moonshot.ai/v1",
                .deepinfra => "https://api.deepinfra.com/v1/openai",
                .together => "https://api.together.ai/v1",
                .fireworks => "https://api.fireworks.ai/inference/v1",
                .parasail => "https://api.parasail.io/v1",
                .qwen, .bedrock, .bedrock_mantle => error.MissingChatPresetBaseUrl,
            };
        }
    };
}
