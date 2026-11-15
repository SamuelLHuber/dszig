# DSZig

A native Zig port of selected [DSPy 3.5.0](https://github.com/stanfordnlp/dspy/releases/tag/3.5.0)
APIs, with **no Zig package dependencies**. Signatures are checked at compile time;
allocators and execution context are explicit.

**This is not a feature-complete or fully identical implementation of DSPy.** The
cross-runtime gates cover the behaviors below, not every DSPy capability. In
particular, the Python-executing RLM/Flex stack, fine-tuning, images/audio,
full provider SDK coverage, and streaming are not implemented. Native File and
bounded Responses document support are implemented, not general multimodal
parity. See [Compatibility limits](#compatibility-limits).

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
| Class-style signatures and typed fields | `Signature`, `Dict`, `Code`, `CodeWithLanguage`, `BoundingBox`, `File`, scalar/string/list/optional/model fields, literal enums, field descriptions/defaults, compile-time rejection of invalid signatures |
| Runtime values and documents | `Value`, `Array`, `ObjectMap`, boxed native Code/File/Reasoning/BoundingBox identity, recursive typed-class serialization, text/canonical-document content union and bounded Responses `input_file` translation |
| Chat, JSON, XML and two-stage adapters | `Adapter.chat`, `.json`, `.xml`, `.two_step`; `TwoStepAdapter` with a separate extraction LM; independent prompt/parse fixtures, strict response-schema construction, JSON fallback, typed XML and Unicode key handling |
| Prediction and reasoning | `Predict`, `ChainOfThought`, `BestOfN`, `Refine`, `MultiChainComparison`, explicit `Context`, typed outputs, serialized `Trace`, named predictor/parameter discovery |
| Retrieval and embeddings | `Retrieve`, lexical `InMemoryRetriever`, HTTP `ColBERTv2` with shared mutable-graph memory caching, custom/OpenAI/Azure `Embedder`, float32 `KNN`, per-query `KNNFewShot`, and brute-force `Embeddings`/`EmbeddingsWithScores` with NumPy persistence |
| Evaluation and few-shot search | `optimize.evaluate`, `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `BootstrapFewShotWithOptuna`, teacher context, zero thresholds and positional multi-use trace selection |
| Instruction optimization | `InferRules`; `MIPROv2`, its program-aware proposer and an Optuna 4.5.0 TPE port; `COPRO`; `SIMBA`; supported instance-frontier `GEPA` paths against gepa 0.1.4 |
| Agents and tools | `ReAct`, comptime `Tool`, `History`, `ToolCalls`, tool results, native declarations and text tool calls, typed submit, retained history |
| Voting | `Ensemble`, `ensemble.majority`, Unicode text normalization; exact Python numeric equality, NaN object identity and source-order fallback behavior |
| LM boundary | `LM`, native `lm.OpenAI` Chat Completions/Responses, shared bounded `http.Transport`, scripted/keyword LMs, recording/replay wrappers and typed native protocol metadata |
| Cache | `CachedLM`, per-candidate/request identity, LRU eviction, rich-response JSON dump/load; failed calls are not cached |
| State | `state.save`, `saveAlloc`, `load`, with DSPy 3.5.0 dependency metadata and program JSON artifact comparisons |

A named surface in this table is not a claim that every Python option or provider
route is implemented. The limits below are part of the compatibility contract.

### LM text and non-text results

`lm.Completion.text` is `?[]const u8`: `null` means that no text part was
returned, not an empty answer. Use `LM.completeResponses` to retain that
distinction together with reasoning, tool calls, continuation and raw provider
data. Recording/replay entries and completion JSON metadata retain nullable
text. The text-only `complete`, `completeMany` and `OpenAI.parseResponse`
helpers return `LMFailure` for an absent text result rather than inventing `""`.
Chat's no-tool empty/null/missing content fallback is separately normalized to
`""`, as in the source; a Chat tool-only result retains `null`.

The Responses parser follows the observed lm15 behavior: root/item statuses
and message roles do not discard returned text, `output_text` and `text`
blocks concatenate, and an empty output can use the top-level text fallback.
Refusal-only, reasoning-only and tool-only responses have absent text;
provider error objects fail. Missing tool-call IDs fall back to the item ID
or a canonical-part-index ID. Arguments are normalized to an object and
serialized with Python JSON spacing and ASCII escaping; malformed argument
text can remain a `partial_json` object. Chat and Responses apply DSPy's
provenance-sensitive repair: only malformed original argument strings correlated
with a real provider ID can replace the fallback, and only dictionary repairs
are accepted. Legitimate `partial_json` parameters and synthesized-ID fallbacks
are retained. Nonfinite tool input numbers and CPython's oversized-integer
failure propagate as `LMFailure`, not silently repaired values.

The zero-dependency `json_repair` module exposes `parseJson` (CPython-style
whole-input parsing) and `loads` (the pinned `json_repair` 0.64.2 default repair
behavior). A native executable matched **390 comparisons over 195 actual Python
parser observations**, including ordered object pairs, exact integers, binary64
bits and lone-surrogate WTF-8 strings. Another matched **26 actual DSPy LM HTTP
observations**, including exact canonical argument JSON, nullable text, real and
synthesized IDs, legitimate `partial_json` objects and failure categories.
The development oracle is `tools/dspy_reference/tool_argument_repair_oracle.py`.
This does not prove every malformed input, optional upstream parser setting or
Python exception identity; the native parser has a 512-level nesting bound.
JSONAdapter also repairs the whole completion before trying DSPy's first
balanced-brace fallback for nonobject results, then filters and validates output
fields. A native consumer matched **141 actual DSPy JSONAdapter calls** across
string, required-pair and integer outputs, including malformed objects, comments,
multiple objects, nonobject roots, exact large integers, nonfinite numbers and
lone surrogates. Integer string fields use repair before conversion and accept
Pydantic's boolean-to-integer coercion. This does not establish every shared
`parse_value` annotation or Python `ast.literal_eval` fallback.

Pinned native parser consumers matched **42 Responses and 4 Chat actual DSPy
3.5 localhost HTTP observations**, comparing visible text, reasoning and tool
calls, including null versus empty text, summary coercion, Unicode argument
escaping, normalization and fallback IDs. The repeatable development oracle is
`tools/dspy_reference/lm_response_oracle.py`. This is protocol-fixture evidence,
not authenticated model inference, provider streaming or complete lm15
compatibility.

### Native runtime values and model identity

Public runtime records, tool results and JSON-shaped APIs use **`dszig.Value`**,
not `std.json.Value`. `dszig.Array` is the managed list of these values;
`dszig.ObjectMap` is their ordered string-keyed map. The familiar `.null`, `.bool`,
`.integer`, `.float`, `.number_string`, `.string`, `.array` and `.object` tags
are joined by `.custom`, which retains a boxed native class instead of erasing
it into a lookalike dictionary. `std.json` parsing and Stringify work directly
with `dszig.Value`; parsing JSON creates ordinary values, never class identity.
A custom value exposes its type through `custom.get(dszig.File)`,
`custom.get(dszig.Reasoning)` or `custom.get(dszig.BoundingBox)` and retains
class-specific validation, formatting, equality, truthiness and cloning.

Typed conversion boxes these three model families; parsing ordinary JSON does
not. A raw lookalike dictionary remains a dictionary until explicit typed
validation. Native class validation preserves an existing box, subject to the
model's actual validators. Owning clones retain their class and copy borrowed
strings; generic container copies do not reproduce Python's memoized alias graph.

- `Reasoning{ .content = ... }` borrows its content. `toRuntime(arena)` boxes it;
  `toValue(arena)` returns the raw `{ "content": ... }` fields. Its ordinary
  serialization is the content string, `str` is bare content and `repr` is the
  quoted Python string. It compares equal to the same content string in either
  order, but not to a lookalike dictionary. Empty Reasoning is false; nonempty
  content is true. It remains unhashable, so `majority` rejects it.
- `BoundingBox` keeps structured geometry in both raw and ordinary serialization.
  Its Python `str`/`repr` are model displays, distinct from its JSON `format`.
  Raw dictionary validation checks finite edges and positive area; validation
  of an existing box reruns the source's ordering check, not field validators.
  JSON-mode serialization converts unvalidated nonfinite fields to null, while
  raw fields and `format` retain them. Frozen File and BoundingBox instances
  are admissible majority-vote keys.
- Refine's recursive mask preserves actual Reasoning content; SIMBA masks that
  same object as `<non-serializable: Reasoning>`. Both preserve ordinary strings
  and dictionaries. Class identity, not an annotation alone, controls this
  distinction.

#### File constructors

`File` has three nullable UTF-8 fields: `file_data`, `file_id` and `filename`.
Its constructors are explicit:

- `File.init(source, keywords)` constructs a fresh typed File without allocating.
  `source` is an optional runtime value; `keywords` is an ordinary object.
  Positional File instances omit null fields before keyword merging, unlike
  dictionaries, which retain explicit null keys. Overlapping fields are
  rejected. Copying an all-null instance with no keywords fails construction,
  whereas validating that same instance preserves it.
- `File.fromValue(input)` accepts a data-URI string, an object with recognized
  File fields, or an actual runtime File. It borrows validated/stripped strings;
  it does not read paths or fetch URLs. Unknown object fields and invalid field
  types are rejected. Runtime class validation preserves an existing File box,
  whereas explicitly converting a raw object to a typed File constructs a File.
- `File.fromFileId(id, filename)` accepts nullable strings and borrows their
  validated/stripped contents.
- `File.fromBytes(arena, bytes, .{ .mime_type = ..., .filename = ... })` allocates
  an encoded data URI and optional filename in the arena. Its default MIME type
  is `application/octet-stream`.
- `File.fromPath(arena, io, path, options)` reads a regular file, following
  symlinks; it uses the basename and MIME guess unless overridden, with
  `application/octet-stream` as the unknown-type fallback. The resulting data URI
  and filename belong to the arena.

Construction validates fields, **not usable media**. Nullable/empty fields and
malformed inline data can construct successfully, then fail in `format`,
`serialize` or adapter rendering. Empty byte input likewise encodes successfully
but fails at formatting. A nonempty `file_id` takes precedence over inline data
and formats as an `application/pdf` document; otherwise formatting validates a
nonempty base64 data URI. The filename is not included in that canonical part.

`file.toRuntime(arena)` allocates a File box but **borrows its field strings**;
`value.toValue(arena, dszig.File, file)` uses this identity-preserving path too.
`file.toValue(arena)` instead returns the raw model dump: an ordinary object
containing all three nullable fields, with borrowed strings and no media marker.
Generic `std.json` Stringify also emits that raw File object. DSPy-style
`pyjson` serialization emits the custom-type marker. These are intentionally
different representations.

```zig
// arena and owner are caller-provided allocators with suitable lifetimes.
const file = try dszig.File.fromFileId("file-report", "report.pdf");
const runtime = try file.toRuntime(arena); // .custom; strings remain borrowed
const raw = try file.toValue(arena);       // .object; not an actual File
const owned = try dszig.inference.cloneJsonValue(owner, runtime);
defer dszig.inference.freeJsonValue(owner, owned);
// owned retains a File class and owns copies of all non-null field strings.
const serialized = try dszig.value.serializedValue(arena, runtime); // marker string
_ = raw;
_ = serialized;
```

Copying a `Value` is not a deep clone. Keep borrowed fields, containers and the
box allocator alive while using them. `inference.cloneJsonValue` makes an owning
recursive copy, including the File box and its fields; release it with
`inference.freeJsonValue` using the same allocator. `value.serializedValue`
recursively serializes actual custom classes inside arrays/objects, copying
containers only where changed. Unchanged subtrees remain borrowed. Ordinary
dictionaries with identical File keys stay dictionaries and are not Python-equal
to an actual File; they do not acquire class identity by serialization.

`adapter.Message.content` is `adapter.Content`, a union of `.text` and `.parts`
(canonical text/document dictionaries, not provider-specific blocks).
`File.format(arena)` returns a canonical document array; `serialize(arena)`
wraps it in `<<CUSTOM-TYPE-START-IDENTIFIER>>` / `<<CUSTOM-TYPE-END-IDENTIFIER>>`.
Rendered adapter messages expand these markers once, including JSON-escaped
payloads, into canonical content parts. This is a rendered-message boundary,
not a general JSON decoder: `Content.fromValue` does not expand markers in a
string. Literal marker text in a rendered message can trigger expansion even
without an actual File. Missing closing markers stay text; payloads that are
not lists remain text, while malformed canonical part dictionaries fail.
Canonical documents require a media type and exactly one of `data`, `url`,
`file_id` or `path`; this is not the full Python DocumentPart object API.

The native OpenAI Responses transport translates supported non-assistant
documents to `input_file` blocks using a file ID, URL, inline base64 data or a
local path read at request construction. Assistant documents are rejected.
Chat Completions rejects document content before HTTP, following the pinned
source preflight rather than inventing an unsupported route. These implemented
paths do not establish full provider/media parity or a completed post-cutover
provider rerun.

### Other native API contracts

`BoundingBox.init(top, bottom, left, right)` constructs normalized image
geometry with finite edges in `[0, 1]` and strictly positive width/height.
Invalid edges are rejected, never clamped or reordered. Strict JSON/model
decoding rejects string/bool edges, missing fields and extra properties;
integer JSON numbers become floats. Structured XML follows the source's
explicit lax override for decoded containers, while JSON literal leaves retain
strict validation. `format(&writer)` emits the source's JSON spacing and float
rendering without allocating an intermediate object; `toValue(arena)` returns
structured geometry rather than a media marker. A native executable matched
50 actual DSPy validation cases, the complete schema, all three exact adapter
prompts/parses, and typed Predict results. Native boxes are Zig values: use
`const` for immutable bindings; direct field mutation does not reproduce
Python's frozen-model assignment exception.

`TwoStepAdapter` uses a natural-language first-stage prompt and an independent
ChatAdapter extraction stage. Configure it with borrowed model handles:

```zig
const two_step: dszig.TwoStepAdapter = .{
    .extraction_model = extraction_client.lm(),
    .extraction_options = .{ .temperature = 0.125 },
};
const ctx = two_step.context(base_ctx);
```

Both clients and `two_step` must outlive `ctx`. Main predictor options do not
leak into extraction. Extraction retries with JSON on a parse failure, but
failure never retries the main LM. Input/demo formatting follows Python `str`,
including declared model, Code, Reasoning, History and Tool representations.
The upstream `text` output/input collision is preserved rather than repaired.
DSPy deprecated this adapter in 3.5; it remains part of the pinned 3.5 surface.
An actual native executable matched **10 actual DSPy cases and 24 LM requests**
exactly, including prompts, rollout/temperature settings, candidate outputs,
output field order and retained traces. It also matched a typed first-stage
prompt containing models, Code, Reasoning and History.

`BootstrapFewShot` accepts zero rounds: no teacher attempts run, but labeled
demos are still selected. A zero error budget propagates the first teacher
failure instead of rejecting the option or swallowing that failure.
Teacher signature matching follows DSPy's named field roles, descriptions,
prefixes and instructions; annotation types and field order do not decide
signature equality.

`optimize.evaluate` follows Python's exact `round(score, 2)`, including
ties-to-even, binary decimal boundaries such as `2.675 → 2.67`, signed zero
and nonfinite scores. It does not round an already-rounded `score * 100`.

`BootstrapFewShotWithOptuna.compile` takes explicit `std.Io` and a required
`max_demos` bootstrap limit. Each candidate selects exactly one demo per
predictor. The optimizer retains every trial and uses Optuna 4.5's independent
integer TPE, distinct from MIPROv2's categorical/multivariate sampler. An
explicit seed reproduces the source-backed NumPy RandomState and integer TPE
sequence probes; an unset seed obtains system entropy. The numerical portability
limit below still applies. An actual native sampler executable matched all
134 independent source suggestions, including stable objective ties, 300-row
history, missing/failed observations, queued values and changing bounds.
A native optimizer executable exercised bootstrap,
20 candidate evaluations and the optimized typed predictor through 128 LM
calls, selecting the same first best trial, score and demos as real DSPy.

`InferRules.init(arena, options)` constructs a persistent rules-induction program.
Keep that arena alive for the optimizer's lifetime. `compile` bootstraps once,
evaluates rule candidates and returns an optional winning program; all-NaN or
negative-infinity scores can leave it null. Its induction RNG, instruction
metadata and teacher-settings binding persist across calls. An explicit borrowed
`teacher_settings_ref` models shared settings mutation; value settings are
captured by value. Candidate evaluation snapshots are independent, while
instruction mutations propagate across matching native signature-class identities
in the reachable caller/teacher/candidate graph, including an earlier winning
program. Class identity survives program copies and remains distinct from
structural signature equality. Instruction-replacement optimizers and state
restoration create fresh class identities.
An actual native executable exercised rule induction, selection and the returned
typed predictor against the independently observed DSPy behavior.

`BestOfN(P).init(program, options)` takes a typed reward callback, an attempt
count and a threshold. Its output is optional: DSPy returns no prediction when
no reward exceeds negative infinity, including all-NaN rewards or no attempts.
Each attempt has its own program value and rollout ID at temperature 1.0;
only the winning program trace is retained, excluding reward-function calls.
The signed failure budget persists across calls, including DSPy's unusual
index-versus-remaining-budget failure boundary. Program values follow the same
immutable-slice ownership convention as the existing optimizers.

`Refine(P).init(program, options)` additionally takes `module_code` and
`reward_code`: the actual source text that Python obtains through introspection
must be supplied explicitly in Zig. It generates real `OfferFeedback` predictions
between attempts, applies per-signature hints through the adapter, and selects
the first strictly best result. Shared signature classes use the last matching
predictor name; empty advice installs no wrapper. Feedback calls use the outer
LM/options and remain in its trace; reward calls do not enter the selected
program trace. A native executable matched **14 actual DSPy cases and all
66 recorded prompts and rollout/temperature settings exactly**, together with
outputs, outer traces and persistent failure budgets. Coverage includes distinct versus shared
signature classes, missing/empty advice, null thresholds, NaN/infinite rewards,
and small-decimal feedback serialization.
Thresholds are nullable `dszig.Value` scalars; for example,
`const threshold: ?dszig.Value = .{ .integer = 100 };`.
`.integer = 100` and `.float = 100` intentionally generate different feedback
strings. Exact integer
thresholds beyond machine widths use `.number_string`; reward comparisons do
not round them to a floating-point threshold. Feedback retains DSPy's
`<non-serializable: int>` masking beyond orjson's integer range.

Construct agents with `ReAct(Sig, tools).init(arena)` or
`react.ReActIters(...).init(arena)`. Each agent gets a fresh derived signature
class; program copies retain its identity. The arena must outlive the agent and
its copies. This prevents separate agents from receiving each other's Refine
advice. A two-agent native run matched DSPy's complete prediction and feedback,
including all five exact prompts and independently assigned hints.

Typed `ReAct.forward` returns the signature's outputs plus `History` and
`termination_reason`, including forced-submit results. A reproduced Refine/ReAct
regression verifies that feedback masks History, Tool and ToolCalls objects
while retaining the original structured traces. An actual native executable
matched DSPy's complete final prediction, all nine feedback inputs and all three
prompts, rollout IDs and temperatures exactly.
The same executable also matched actual DSPy selections for positive and negative
`2**200 + 1` thresholds and the resulting large-integer feedback mask.

`Code` defaults to Python; `CodeWithLanguage("JAVA")` keeps the supplied language
in its description and lowercases only the Markdown fence. Validation extracts
the first multiline fenced match before considering inline fences; unfenced
source remains untrimmed. Native typed predictions, nested collections, adapters,
provider schemas, state demos and ReAct submit preserve Code's raw-string wire
representation while retaining its model validation.
Erased records retain genuine boxed Code classes rather than ordinary strings
or annotation-based reconstructions. `CodeWithLanguageIdentity("JAVA", Tag)`
permits distinct same-language classes using separate explicit type tags;
`CodeWithLanguage` alone remains canonical for the same language. Validating a
specialized box as base Code preserves the box and concrete class; validating
it as a different specialized class fails. Ordinary and round-trip dumps remain
strings, while Python-style `str` is `code='…'` and nested repr includes the
concrete model name. Owning clones copy the code and retain its class.
A native consumer matched **281 actual DSPy validations, 210 class/rendering/
serialization/hashability comparisons and 75 symmetric equality cases** from
`code_identity_oracle.py`.
Automatic fresh runtime class creation, Python constructor calling conventions
and generic memoized deepcopy graphs are not provided by these explicit Zig APIs.
A native executable also exercised a typed Code prediction, fenced output,
language-specialized instance preservation through the base validator,
raw-string traces and saved-demo restoration.
Further native checks matched 12 actual XML predictions across direct, optional,
dictionary and nested-list Code annotations, including exact input/output
structure formatting. Five composed/decomposed Unicode and control-character
schema prompts also matched exactly. Provider schemas reject unequal colliding
definitions rather than silently aliasing them; equivalent definitions share
the same source-matching schema.

`Embeddings` and `EmbeddingsWithScores` embed the corpus once, normalize float32
vectors when requested and expose explicit batched queries. Their signed `k`
follows `[:k]`, including empty results for zero, rather than KNN's reverse
slice. Scores widen the float32 calculation to float64 like NumPy `tolist()`,
preserving comparisons such as `0.800000011920929 > 0.8`. `save`, `load` and
`fromSaved` use DSPy's `config.json`/`corpus_embeddings.npy` layout; supported
float32 arrays include NPY versions 1–3, both byte orders and C/Fortran storage.
The embedder and initial corpus are borrowed; initialized retrievers own their
vectors and loaded configuration and require `deinit`.
An actual native executable loaded six independently generated Python saves
without recomputing the corpus and matched their retrieval results. Real DSPy
then loaded both normalized and raw native saves and reproduced the same indices,
passages, float64 scores and query-only embedding calls.
Eight further load/save round trips matched actual NumPy output bytes, including
scalar, vector, empty and degenerate Fortran headers. Saves canonicalize arrays
that are also C-contiguous to `fortran_order: False`, preserving nondegenerate
Fortran storage.

`MultiChainComparison(Sig, M)` accepts exactly `M` ordered completion records
and the original typed inputs. It constructs the student-attempt fields and
returns typed `rationale` plus the original outputs. Default temperature is
0.7; explicit context options override it. Signature collisions, first-line
formatting, ignored input tracing and predictor state roundtrips have
source-backed coverage. These modules were also exercised through an actual
native executable, independently of the existing nine-pipeline replay fixture.

`ColBERTv2.init(arena, &client, &cache, options)` supports GET and POST retrieval.
The caller owns the `std.http.Client` and `request_cache.Cache`; both must outlive
the retriever. `forward(query, k)` accepts signed `i64` counts and returns a
leased result: call `result.deinit()` when finished. GET preserves DSPy's double
negative slicing and synthesized `long_text`; POST slices once and preserves
the server's `long_text`, including missing or non-string values.
`result.simplify(arena)` returns graph values borrowing that lease, while
`retriever()` adapts to the string-only `Context.retriever` interface and copies
returned strings into the context arena. Retrieval failures propagate through
`Retrieve`, RAG and Ensemble before generation.

The two ColBERT cache layers retain the same first raw payload. Returned rows
are shallow copies: initial top-level edits are private, but nested edits reach
cached data, including list appends that reallocate. Cache hits are memoized
deep copies, isolating their later mutations. Result leases survive replacement,
eviction and cache destruction. The cache mutex protects entries, not mutable
graphs; callers must serialize graph mutations against concurrent readers.
Memory caching can be disabled or bounded with a positive `max_entries`;
zero capacity fails on insertion. GET's `k > 100` guard follows cache lookup,
so preseeded inner or outer hits bypass it, as upstream does.

`Embedder.initCustom` accepts a native callback returning float32, float64 or
rectangular JSON vectors; `initHosted` supports `openai/` and `azure/` models
through the existing `std.http` transport. Hosted environment values are explicit,
not read implicitly from the process. `embed(arena, texts, options)` returns a
contiguous float32 matrix owned by the caller. Batch-size/caching overrides,
merged request options, SDK response-index sorting and base64 vectors are
supported. Its bounded per-instance cache retains successful raw batches before
final float32 conversion, including the reference's conversion-error behavior.
It is not DSPy's shared Python disk cache.

Hosted timeouts preserve per-operation inactivity rather than imposing a total
request deadline: the SDK defaults are 600 seconds for reads/writes and 5 seconds
for connection setup; explicit null disables both. Azure token callbacks refresh
before every attempt, including retries, and callback failures retain their
native error identity. Request-local clients keep timed connections from escaping
their I/O adapter's lifetime and preserve independently owned copies of the
caller's initialized TLS trust store. Deadline tasks remain concurrent even when
the async worker budget is exhausted. External transport cancellation is not
retried; the low-level JSON transport preserves `Canceled`. The shared transport
still bounds response bytes.

`KNN.init(arena, k, trainset, &embedder)` embeds training inputs once;
`call(arena, ordered_kwargs)` uses raw float32 dot products without normalization.
`KNNFewShot(P)` retrieves and bootstraps again on each forward. Its compiled
program borrows the optimizer, original student and optional teacher, so their
addresses must remain stable and their lifetimes must cover every forward.
Predictor discovery exposes the visible student's namespace, not hidden captures.
Bootstrap mapping checks the live student/teacher instructions, descriptions and
prefixes before reporting an empty training neighborhood.
Both optimizers and embedding output arenas must outlive the values borrowing
them. NumPy sorting/BLAS portability limits are explicit below.

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

Local verification on 2026-11-15 of the File/Reasoning/BoundingBox native
runtime-value and nullable-completion cutover passed **274/274 tests** in both
Debug and ReleaseSafe, with all 28 build steps, including the actual
RAG/bootstrap examples. Its actual replay executable consumed all **687**
deterministic mock calls and matched every adapter case and pipeline; the
independent real-DSPy reference gate also passed.

At the preceding 261-test File checkpoint, the actual
`zig build live -- --smoke` CLI completed once through Chat
Completions and once through Responses against an isolated localhost HTTP
fixture, returning `4` and writing each protocol's transcript. A separate
negative run exited before HTTP when a dummy `CODEX_LB_API_KEY` was supplied
without `OPENAI_BASE_URL`. These are native CLI/transport/configuration checks,
**not authenticated homelab inference or live-model parity**.

Separate post-cutover pinned native consumer smokes matched the actual DSPy
File oracle for **8 shapes, 64 exact prompts (including TwoStep), 198 parses,
24 typed Predicts, 26 marker cases and 8 persistence/masking/loaded-demo
behaviors**. An identity consumer matched six actual source/native tool-result
cases: actual File versus plain lookalike dictionary, repeated nested instances
of each, escaped file IDs and late invalid-File formatting. It also checked
class-validation pointer identity, a deep-owned clone surviving source-arena
destruction and allocator cleanup, and exact escaped streamed marker JSON.
That last check exercises the JSON writer, not provider streaming.
The post-cutover provider consumer also matched **all 14 actual HTTP request
bodies byte-for-byte and all 17 typed predictions**, including cache behavior
and retained provider JSON. It still reports six unresolved cross-API
comparisons: five compare native replay text with the source's structured SDK
output view, and one compares a private SDK Chat display serializer with public
Chat preflight. Those are not successful SDK-equivalence checks.
A separate constructor executable compared **393/429 constructor cases,
47/51 filesystem cases and 5/6 constructor-I/O controls**, with **5,107 assertions
and zero failures**. This includes nine actual File-instance construction,
validation and copy cases: native instances are reconstructed as boxed Files,
not substituted with lookalike dictionaries, and recorded same-instance
validation checks compare native box identity. Across exercised cases it
compared 3,671 source operations and 329 additional streaming-writer outputs.
Explicitly skipped dispatches and Python DocumentPart operations remain limits.
These smokes do not establish exhaustive constructor or DocumentPart coverage,
full File parity or full provider/SDK compatibility.

The subsequent Reasoning/BoundingBox native-object consumer matched **592**
assertions against independently executed DSPy 3.5 source receipts: 63
Reasoning validation rows, eight constructor view sets, equality and nested
JSON/tool boundaries, five valid and four unvalidated BoundingBox cases, plus
Refine/SIMBA masking, model truthiness and majority hashability. It rebuilt
actual native boxes rather than substituting their dictionaries. The checks
explicitly exclude tuple repr/dispatch, shared deepcopy aliases, NaN
cross-instance float-object identity and Reasoning's Python string operators.

Additional native executable checks exercised hosted OpenAI/Azure embeddings,
base64/index ordering, cache reuse, vector KNN, repeated KNNFewShot predictions
and retained traces against a scripted localhost HTTP provider. Azure checks
covered refreshed retry credentials, explicit Authorization precedence and
callback errors. Transport checks covered idle headers/body, a 903 ms dripping
body under a 120 ms inactivity timeout, disabled/zero timeouts, and the existing
Chat Completions/Responses CLIs. Further actual HTTP checks exercised an exhausted
async worker budget and cancellation without duplicate attempts. A private-CA
HTTPS check covered untimed and timed clients followed by a fresh untimed TLS
handshake, proving the caller's trust remained usable. An exhaustive native
representation digest matched actual pinned CPython `repr` for all 1,112,064 Unicode scalar values.
The shared HTTP transport was also exercised with an actual bodiless gzip GET,
raw non-success status inspection, per-retry header refresh and override,
unchanged callback errors, response-size rejection, an inactivity timeout and
the actual live CLI against a scripted localhost provider. Generation retains
its existing HTTP-200-only success policy; status-inspecting consumers use the
same bounded transfer and timeout implementation without generation retries.
An actual native ColBERT executable also matched the pinned DSPy oracle's
**81 HTTP requests byte-for-byte**, with 87 value/error-category/alias checks:
GET/POST and signed slicing, Unicode URLs and JSON bodies, typed/missing
`long_text`, malformed-response cache timing, two-layer promotion, nested
mutation isolation, eviction, distinct request keys and preseeded cap bypass.
A separate HTTP consumer exercised negative-`k` Retrieve, RAG and Ensemble,
preserved retrieval errors before LM invocation, and verified copied strings
after cache destruction. Five scripted generation prompts were used, not a
real inference endpoint. Redirect checks matched 57 actual requests across
GET/POST 301/302/303/307/308, relative Unicode redirects, cross-port credential
stripping and redirect-loop termination. URI requoting matched pinned requests
across all 1,112,064 Unicode scalar values; this is not full requests/session
equivalence.
These are protocol and formatting checks, not homelab or real-model evidence.

The independent optimizer source probes can also be run directly:

```sh
uv run --project tools/dspy_reference --locked tools/dspy_reference/infer_rules_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/bootstrap_optuna_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/code_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/embeddings_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/refine_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/refine_react_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/two_step_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/colbertv2_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/bounding_box_oracle.py
uv run --project tools/dspy_reference --locked tools/dspy_reference/file_oracle.py
```

These execute actual pinned DSPy/Optuna. They cover rule metadata/settings
persistence, source alias mutations and failure timing, independent integer TPE
suggestions and optimizer probes, Code validation/adapters, dense retrieval and
NumPy persistence, Refine feedback/selection and ReAct object masking, and actual
two-stage extraction/fallback. The File oracle additionally records constructor,
filesystem, adapter, marker, persistence and provider observations; source
observations alone are not native coverage. Native behavior tests preserve
the corresponding source-backed consumer boundaries; the probes do not add
real-model inference evidence.

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
- **Other modules and optimizers:** no Parallel,
  AvatarOptimizer, BetterTogether, BootstrapFinetune or bootstrap_trace_data.
- **Optimizer execution and shared identity:** `InferRules` and
  `BootstrapFewShotWithOptuna` currently support serial evaluation
  (`num_threads = null` or `1`), rejecting other explicit
  settings. InferRules' signature-alias propagation does not update arbitrary external
  instances outside the reachable program graph. Canonical Zig signature/Code
  types and default CoT construction do not reproduce every fresh dynamic Python
  class identity; Refine shares this boundary. Mid-bootstrap mutation of a
  borrowed teacher-settings object is not reflected after Bootstrap's entry
  snapshot.
- **Runtime model identity:** File, Reasoning and BoundingBox are genuine boxed
  classes, not a complete Python object runtime. Other model families still
  use their existing representations. Generic arrays/maps do not preserve
  shared container/class aliases through memoized deepcopy. Python tuple
  identity/dispatch, NaN float-object identity across distinct or cloned boxes,
  Reasoning indexing/iteration/string-method delegation, arbitrary attribute
  mutation and Python frozen-assignment exceptions are not reproduced.
- **Providers and media:** native File construction, adapter/typed prediction
  handling, File tool-result markers and bounded Responses document translation
  are implemented as described above. There is still no complete lm15 provider
  catalog/routing/auth/error taxonomy, full provider SDK, embedding providers
  beyond OpenAI/Azure, Image/Audio, general rich-media tool results, provider
  training lifecycle or native streaming. Document support is not general
  multimodal support, and Chat Completions does not accept document parts.
- **File and DocumentPart boundaries:** the constructor smoke explicitly leaves
  606 successful source Python DocumentPart object/repr/bytes operations
  unimplemented; canonical document dictionaries do not replace those
  operations. Constructor evidence still excludes bytearray/memoryview/buffer
  and tuple dispatch, bytes-valued field coercion, nonstring helper parameters,
  bytes-path/PathLike dispatch, duplicate arguments and positional arity.
  The nine File-instance construction/validation/copy cases are now exercised,
  but do not close those remaining dispatch gaps.
  Native errors do not reproduce Python exception classes/messages, and finite
  constructor/formatting cases do not establish full File parity.
- **ColBERT HTTP/cache boundaries:** no Python diskcache/key interoperability,
  requests cookie jar, proxy/netrc/IDNA/session behavior or complete Python
  exception type/message reconstruction. Native JSON rejects bare NaN/Infinity
  and unpaired surrogate escapes accepted by the Python JSON backend. HTTP
  response bytes are bounded explicitly; graph-to-JSON conversion rejects cycles
  and nesting beyond 1,000 levels, while cache graph cloning preserves aliases
  and cycles. Native cache zero-capacity rejection occurs on insertion rather
  than Python's constructor.
- **Dynamic Python APIs:** no general Python Module/BaseModule/Example/Prediction/
  Completions object model, dynamic signature syntax, arbitrary Pydantic custom
  models/validators, custom Type hierarchy, generic callbacks,
  settings singleton, usage accounting, asyncify/syncify or streamify equivalents.
- **Type and parsing boundaries:** Unequal response-schema definitions with
  colliding names fail closed; equivalent definitions can share a name.
  Python's identity-dependent automatic renaming is not reproduced. Tagged
  unions beyond optional types and some
  Pydantic model defaults are not native signature types. JSON repair covers the
  pinned default parser and correlated Chat/Responses tool arguments, not every
  upstream option or every adapter parsing path. Its nesting bound is 512 and
  Python exception classes/messages are not reconstructed (for example, some
  fenced nonstring-key assertion failures become `ParseValue`). XML golden cases that
  use a schema-backed runtime view do not prove a native Zig union type exists.
  Python-style rendering recovers declared native type metadata and retained
  native custom classes, not arbitrary Python objects erased into ordinary
  `Value`/`Record` dictionaries; a nested dynamic model inside a History record
  therefore cannot retain its original Python class repr merely from its keys.
- **Optimizer options:** unsupported GEPA custom code/instruction proposers,
  Flex components, other frontier/selection policies, format-failure feedback,
  logging integrations, and some MIPROv2/SIMBA/COPRO statistics/logging options.
  Evaluation runs sequentially rather than implementing DSPy's parallel and
  async execution. SIMBA uses Zig `@exp`, not NumPy's implementation.
- **Integer TPE numerical portability:** the independent sampler uses a native
  extended-precision logarithm before binary64 rounding to reproduce the observed
  source acquisition/selection boundaries. A separate 10,009-value primitive
  diagnostic still differs from Darwin libm/NumPy at six inputs. Matching the
  observed trial sequences does not prove every possible history has identical
  floating-point decisions; no libc dependency or input-specific correction hides
  that limit.
- **Dense retrieval boundaries:** FAISS construction/indexed saves are unsupported;
  the default threshold of 20,000 corpus items therefore fails explicitly rather
  than silently changing index algorithms. Retrieval caching and Python
  Unbatchify's background scheduler are not implemented. Persistence accepts
  float32 arrays, not arbitrary NumPy dtypes/ranks. Scalar sorting and the observed
  NEON einsum arithmetic are not all-platform NumPy CPU-dispatch equivalence.
- **Embedding numerical portability:** KNN ports NumPy 2.4.6's scalar quicksort,
  including unstable ties and its heapsort fallback. Other NumPy SIMD dispatches
  can order ties differently. Native strict float32 dot accumulation is not
  bitwise identical to BLAS: the locked Apple Accelerate reference returns 2
  for `[1e8, 1, -1e8, 1]` dotted with ones, whereas the scalar kernel returns 1;
  reference accumulation also changes with matrix shape. Cancellation-sensitive
  neighbor rankings are therefore a known incompatibility, not a parity claim.
  Python string representations use the pinned CPython 3.11 Unicode 14 tables;
  classification can differ from other Python/Unicode releases.
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
