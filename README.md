# DSZig

A native Zig port of selected [DSPy 3.5.0](https://github.com/stanfordnlp/dspy/releases/tag/3.5.0)
APIs, with **no Zig package dependencies**. Signatures are checked at compile time;
allocators and execution context are explicit.

**This is not a feature-complete or fully identical implementation of DSPy.** The
cross-runtime gates cover the behaviors below, not every DSPy capability. In
particular, the Python-executing RLM/Flex stack, fine-tuning, embeddings,
multimodal content, and streaming are not implemented. See [Compatibility limits](#compatibility-limits).

## Example

```zig
const dszig = @import("dszig");

const QA = dszig.Signature(struct {
    pub const instructions = "Answer questions with short factoid answers.";
    pub const Input = struct {
        context: []const []const u8,
        question: []const u8,
    };
    pub const Output = struct { answer: []const u8 };
    pub const descriptions = .{ .answer = "often between 1 and 5 words" };
});

// Allocator, LM, adapter, retriever and trace are caller-owned.
const ctx: dszig.Context = .{ .arena = arena, .lm = client.lm() };
const result = try (dszig.ChainOfThought(QA){}).forward(ctx, .{
    .context = &.{"Paris is the capital of France."},
    .question = "What is the capital of France?",
});
// result.answer is a string; result.reasoning is ?dszig.Reasoning.
// A provider can omit a native reasoning summary, so reasoning can be null.
```

[`examples/rag.zig`](examples/rag.zig) and
[`examples/bootstrap.zig`](examples/bootstrap.zig) are runnable RAG and
few-shot optimization examples:

```sh
devenv shell
zig build rag
zig build bootstrap -- rag.json
```

The library targets Zig 0.17.0. `devenv.nix` and CI pin the compiler; the reference
harness separately pins CPython 3.11, DSPy 3.5.0 and every Python dependency in
`tools/dspy_reference/uv.lock`. Python is reference tooling, not a library runtime
dependency. `build.zig.zon` has `.dependencies = .{}`.

## DSPy 3.5 contracts

This migration is not a relabeling of the previous 3.4.1 implementation:

- **XML is the default adapter.** `Adapter.xml` uses typed field tags, literal
  string leaves, nested model/list/dictionary XML, nullable self-closing tags,
  and DSPy's formatting and parsing rules. Explicit `Adapter.chat` and
  `Adapter.json` remain available. Missing output defaults are applied in
  signature order; an absent value and an explicit null are distinct.
- **Typed integers retain their range.** Values outside signed 64-bit range
  remain unquoted JSON integers rather than being rounded through `f64` or
  narrowed to `i64`. The same representation is used by typed predictions,
  nested fields, ReAct tool arguments/results and numeric majority voting.
- **ChainOfThought uses native reasoning or a plan fallback.** Native reasoning
  fields are removed from the prompt for an advertised capable LM. Otherwise
  the default wire field is `plan`, with collision-safe names such as `plan_2`.
  The typed result is `?Reasoning { content }`; DSPy's custom serialization emits
  the content string, not a `{content: ...}` object. Use
  `ChainOfThoughtWithOptions(Sig, .{ .rationale_field_type = []const u8 })` when
  explicitly requesting a normal string rationale, as DSPy permits.
  A user-declared `?Reasoning` output remains a nullable model field; it does not
  inherit ChainOfThought's native-reasoning routing. Required Reasoning values
  are validated even when the provider response schema describes them as strings.
- **ReAct uses structured History and typed `submit`.** There is no legacy
  `finish`/`extract` agent implementation. Calls include turn budgets, multiple
  tool calls/results, error observations, final-submit attempts and termination
  reasons. History retains tool-call IDs and per-call provider continuation.
  The implicit XML adapter's native-History policy is distinct from whether a
  particular LM supports native function declarations. Set
  `Context.use_native_function_calling` to override the policy.
- **Multiple candidates are separate requests.** An `n > 1` generation emits
  individual `n = 1` LM calls with DSPy's tagged candidate rollout identity.
  Rollout IDs preserve JSON integer, string and array identities rather than
  collapsing them to an integer. Replay does not excuse a different candidate
  request identity.
- **Bootstrap trace selection is positional.** Repeated predictor traces are
  selected using `random.Random(f"{predictor_name}:{prior_trace_count}")`, not
  the obsolete pickle hash. The independent oracle exercises 144 cases,
  including Unicode and predictor names longer than 64 bytes. An explicit
  `metric_threshold = 0.0` accepts a zero score, unlike an unset threshold.
- **Native request metadata is retained.** Chat Completions and Responses
  transports support typed response formats, function declarations, tool
  choices, call IDs, actual summaries, provider continuation and raw response
  data. Capabilities are explicit; selecting JSONAdapter does not invent a
  provider capability. Cached, recorded and replayed responses retain native
  data, and cache keys include request metadata.

DSPy itself drops response-level thinking continuation when converting it to a
content-only `Reasoning` value inside ReAct. DSZig follows that high-level
boundary rather than inventing hidden History fields. The low-level LM API can
retain the actual continuation and raw provider response independently.

## Implemented surfaces

| DSPy surface | Native Zig surface and coverage |
|---|---|
| Class-style signatures and typed fields | `Signature`, `Dict`, scalar/string/list/optional/model fields, literal enums, field descriptions/defaults, compile-time rejection of invalid signatures |
| Chat, JSON and XML adapters | `Adapter.chat`, `.json`, `.xml`; independent prompt/parse fixtures, strict response-schema construction, JSON fallback, typed XML and Unicode key handling |
| Prediction and reasoning | `Predict`, `ChainOfThought`, explicit `Context`, typed outputs, serialized `Trace`, named predictor/parameter discovery |
| Retrieval | `Retrieve`, a retriever vtable and lexical `InMemoryRetriever` |
| Evaluation and few-shot search | `optimize.evaluate`, `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, teacher context, zero thresholds and positional multi-use trace selection |
| Instruction optimization | `MIPROv2`, its program-aware proposer and an Optuna 4.5.0 TPE port; `COPRO`; `SIMBA`; supported instance-frontier `GEPA` paths against gepa 0.1.4 |
| Agents and tools | `ReAct`, comptime `Tool`, `History`, `ToolCalls`, tool results, native declarations and text tool calls, typed submit, retained history |
| Voting | `Ensemble`, `ensemble.majority`, Unicode text normalization; exact Python numeric equality, NaN object identity and source-order fallback behavior |
| LM boundary | `LM`, native `lm.OpenAI` Chat Completions/Responses, scripted/keyword LMs, recording/replay wrappers and typed native protocol metadata |
| Cache | `CachedLM`, per-candidate/request identity, LRU eviction, rich-response JSON dump/load; failed calls are not cached |
| State | `state.save`, `saveAlloc`, `load`, with DSPy 3.5.0 dependency metadata and program JSON artifact comparisons |

A named surface in this table is not a claim that every Python option or provider
route is implemented. The limits below are part of the compatibility contract.

## Cross-runtime verification

```sh
# Native modules, independent golden fixtures, behavior, full-suite replay,
# positional trace selection, Optuna and compile-fail cases:
zig build test examples --summary all
zig build test examples -Doptimize=ReleaseSafe --summary all

# Regenerate every oracle with actual locked DSPy 3.5.0, require unchanged
# fixtures, and check that DSPy loads/re-saves the Zig-generated artifact:
tools/dspy_reference/check.sh

# Exercise the actual native replay executable, without a network or API key:
zig build replay -- tests/fixtures/dspy_3_5_0_mock_suite.json
```

`tools/dspy_reference` uses real DSPy adapters, predictors, optimizers, evaluation
and state serialization. Only provider responses in deterministic fixtures are
scripted. The independently recorded full-suite fixture contains **687 model
requests, eight adapter cases and nine pipelines**:

- RAG + BootstrapFewShot;
- dates + BootstrapFewShot and random search;
- dates + MIPROv2, with and without the program-aware proposer;
- dates + COPRO;
- RAG ReAct + BootstrapFewShot;
- dates + SIMBA and GEPA.

Replay requires exact request metadata and consumption, output values, evaluation
scores, saved artifact bytes, instruction candidates, trial logs, candidate
scores and SIMBA train scores. It is **mock-provider behavioral evidence**, not
proof of real-model performance or complete DSPy equivalence.

Local verification on 2026-11-14 passed **110/110 tests and all 28 build steps**
in both Debug and ReleaseSafe. The actual replay executable consumed the entire
fixture and matched every adapter case and pipeline. The independent real-DSPy
reference gate also passed. These results establish only the coverage described
above, not the missing capabilities listed below.

The existing `tests/fixtures/live/*gpt-4o-mini.json` recordings were made against
**DSPy 3.4.1**. They remain unchanged as historical artifacts. They are not
relabeled or used as evidence that the new 3.5 protocol matches an old protocol.
Fresh 3.5 live recordings must identify their actual model, capabilities and
`target_dspy` before either runtime replays them.

## Real inference

The live executable reads `OPENAI_API_KEY`, or `CODEX_LB_API_KEY` for the local
homelab configuration. Selecting `CODEX_LB_API_KEY` requires an explicit
`OPENAI_BASE_URL`; the homelab key is never sent to the public default endpoint.
It never reads or mutates personal agent settings and never embeds credentials
in transcripts.

For an OpenAI-compatible Responses endpoint:

```sh
export OPENAI_BASE_URL=http://codexlb.homelab:2455/v1
export DSZIG_LIVE_API=responses
# Set DSZIG_LIVE_MODEL to an actual model served by the authenticated endpoint.
# Supply CODEX_LB_API_KEY in the environment, not in source or committed files.
zig build live -- --smoke
zig build live -- tests/fixtures/live/dszig_3.5.0.json
```

`DSZIG_LIVE_REASONING=1`, `DSZIG_LIVE_RESPONSE_SCHEMA=1` and
`DSZIG_LIVE_FUNCTION_CALLING=1` advertise the corresponding capabilities; enable
only capabilities the actual served model supports. Defaults use Chat
Completions, the public OpenAI URL and `gpt-4o-mini`.

For a fresh full-suite transcript recorded by either runtime, replay in the other:

```sh
zig build replay -- path/to/dspy_3.5.0.json
uv run --project tools/dspy_reference --locked \
  tools/dspy_reference/check_live.py path/to/dszig_3.5.0.json
```

An authentication failure, a deterministic replay, or a local scripted HTTP
provider does not count as successful inference from the configured homelab.

## Compatibility limits

The following prevent a claim that DSZig can do everything DSPy can:

- **RLM, Flex and interpreters:** no PythonInterpreter/Pyodide/Deno,
  LocalInterpreter, InterpreterPool or complete Python execution environment.
  DSPy's RLM/Flex execute arbitrary Python and can import real DSPy inside the
  guest. A restricted expression evaluator is not an identical implementation.
- **Other modules and optimizers:** no general BestOfN, Refine, Parallel, KNN,
  MultiChainComparison, AvatarOptimizer, BetterTogether, BootstrapFinetune,
  KNNFewShot, BootstrapFewShotWithOptuna, InferRules or bootstrap_trace_data.
- **Providers and media:** no complete lm15 provider catalog/routing/auth/error
  taxonomy, Embedder/embedding retrievers/ColBERTv2, Image/Audio/BoundingBox/File,
  rich media tool results, provider training lifecycle or native streaming.
- **Dynamic Python APIs:** no general Python Module/BaseModule/Example/Prediction/
  Completions object model, dynamic signature syntax, arbitrary Pydantic custom
  models/validators, custom Type hierarchy, TwoStepAdapter, generic callbacks,
  settings singleton, usage accounting, asyncify/syncify or streamify equivalents.
- **Type and parsing boundaries:** tagged unions beyond optional types and some
  Pydantic model defaults are not native signature types. JSON repair is not
  implemented; strict JSON and the existing limited Python-literal parser are
  not equivalent to `json_repair` for malformed output. XML golden cases that
  use a schema-backed runtime view do not prove a native Zig union type exists.
- **Optimizer options:** unsupported GEPA custom code/instruction proposers,
  Flex components, other frontier/selection policies, format-failure feedback,
  logging integrations, and some MIPROv2/SIMBA/COPRO statistics/logging options.
  Evaluation runs sequentially rather than implementing DSPy's parallel and
  async execution. SIMBA uses Zig `@exp`, not NumPy's implementation.
- **Persistence and errors:** cache files/key hashing are not DSPy's diskcache
  format; LM/provider/training state is not a complete DSPy save/load port.
  Python exception tracebacks are not reconstructed from native Zig errors.
- **Upstream limits are not repaired silently:** stock DSPy 3.5.0 Retrieve still
  rejects the `json_mode` argument supplied by module state dumping. The RAG
  reference uses a small Retrieve subclass accepting that argument; this is an
  explicit upstream workaround, not evidence that stock DSPy can save that RAG.
  DSPy's strict JSON response-schema transformation can also close nested
  dictionary objects, excluding otherwise valid nonempty dictionaries. DSZig
  preserves that upstream behavior; schema-capable nested dictionary outputs
  are not claimed to be free of this source limitation.

The finite parity gate should grow with real capabilities. Missing APIs must not
be represented by stubs, ignored options, fabricated results, or a claim of full
compatibility based only on existing examples.
