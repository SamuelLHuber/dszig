# DSZig

A native Zig port of selected [DSPy 3.5.0](https://github.com/stanfordnlp/dspy/releases/tag/3.5.0)
APIs, with **no Zig package dependencies**. Signatures are checked at compile time;
allocators and execution context are explicit.

**This is not a feature-complete or fully identical implementation of DSPy.** The
cross-runtime gates cover the behaviors below, not every DSPy capability. In
particular, the Python-executing RLM/Flex stack, images/audio,
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
| Class-style signatures and typed fields | `Signature`, `Dict`, `Code`, `CodeWithLanguage`, `BoundingBox`, `File`, `Citations` and `Citations.Citation`, scalar/string/list/fixed-array/optional/model fields, string literal enums and mixed scalar/Enum/bytes `Literal`, field descriptions/defaults, compile-time rejection of invalid signatures |
| Runtime values and documents | `Value`, `Array`, `ObjectMap`, boxed native Code/File/Reasoning/BoundingBox/Citations/decision/evidence identity, recursive typed-class serialization, text/canonical-document content union and bounded Responses `input_file` translation |
| Chat, JSON, XML and two-stage adapters | `Adapter.chat`, `.json`, `.xml`, `.two_step`; `TwoStepAdapter` with a separate extraction LM; independent prompt/parse fixtures, strict response-schema construction, JSON fallback, typed XML and Unicode key handling |
| Prediction and reasoning | `Predict`, `ChainOfThought`, `BestOfN`, `Refine`, `MultiChainComparison`, explicit `Context`, per-predictor LM bindings and module LM aggregation, typed outputs, serialized `Trace`, named predictor/parameter discovery |
| Parallel composition | Startup-created `Parallel(Result, limits)` workers, typed program/Example/tuple bindings, ordered result slots, speculative straggler retries, bounded accumulated failure reports and joined cooperative cancellation |
| Typed decisions | `Noul`, configured `Score`/`Choice`, generative evidence classes, criteria/state snapshots, native Annotated metadata, owning evidence collectors and TypeSafe SystemOne prediction; finite coverage and unsupported rows below |
| Retrieval and embeddings | `Retrieve`, lexical `InMemoryRetriever`, HTTP `ColBERTv2` with shared mutable-graph memory caching, custom/OpenAI/Azure `Embedder`, float32 `KNN`, per-query `KNNFewShot`, and brute-force `Embeddings`/`EmbeddingsWithScores` with NumPy persistence |
| Evaluation and few-shot search | `optimize.evaluate`, concurrent `parallel.Evaluator`, `bootstrapTraceData` with real parse-failure capture and ordered native rollouts, `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `BootstrapFewShotWithOptuna`, teacher context, zero thresholds and positional multi-use trace selection |
| Fine-tuning | Chat/XML `Adapter.formatFinetuneData`; native `BootstrapFinetune` over bounded OpenAI HTTP jobs, startup workers, retained model names and joined failure cleanup |
| Optimizer composition | Native `BetterTogether`, real default prompt/weight optimization, typed custom registries, retained candidate LM snapshots and joined lifecycle cleanup |
| Instruction optimization | `InferRules`; `MIPROv2`, its program-aware proposer and an Optuna 4.5.0 TPE port; `COPRO`; `SIMBA`; supported instance-frontier `GEPA` paths against gepa 0.1.4 |
| Agents and tools | `ReAct`, bounded native `Avatar` and `AvatarOptimizer`, comptime `Tool`, `History`, `ToolCalls`, tool results, native declarations and text tool calls, typed submit, retained history |
| Generated native execution | `generated_zig.Generator` (`Predict(Code[zig])`) and fixed-storage Linux `Executor`: restricted compiler, kernel-confined guest, exact native capability RPC, pending output and whole-tree join; not Python RLM/Flex |
| Voting | `Ensemble`, `ensemble.majority`, Unicode text normalization; exact Python numeric equality, NaN object identity and source-order fallback behavior |
| Answer metrics | `metrics.emScore`, `f1Score`, `hotpotF1Score`, `precisionScore`, multi-reference `em`/`f1`/`hotpotF1`, thresholded and record-based answer matching, DPR-token passage matching |
| LM boundary | `LM`, native `lm.OpenAI` Chat Completions/Responses and `lm.TypeSafe` SystemOne, shared bounded `http.Transport`, scripted/keyword LMs, recording/replay wrappers and typed native protocol metadata |
| Cache | `CachedLM`, per-candidate/request identity, LRU eviction, rich-response JSON dump/load; failed calls are not cached |
| State | `state.save`, `saveAlloc`, `load`/`loadState` with explicit load options, native OpenAI configuration reconstruction, DSPy 3.5.0 dependency metadata and program JSON artifact comparisons |

A named surface in this table is not a claim that every Python option or provider
route is implemented. The limits below are part of the compatibility contract.

### Fixed-size native signature fields

A field `[N]T` is a homogeneous fixed-length Python tuple annotation, not a
variable-length list. For example, `[2]i64` renders as `tuple[int, int]` with
`prefixItems` and equal `minItems`/`maxItems`; `[0]T` is the empty tuple.
Arrays compose with optional, list, dictionary, model and other array fields.
Validation applies the element codec and rejects missing or extra elements.
Typed decoding writes into the fixed array, including its sentinel when present.

Erased values retain genuine runtime tuple metadata. Chat/JSON/XML formatting
and typed Predict work through the same field codecs; XML prompts use a JSON
tuple body, matching DSPy's tuple annotation behavior rather than its nested-list
prompt syntax. Owning clones copy tuple children; source-backed tuples and their
children retain the existing arena lifetime requirements. The nested conversion
still provisions runtime tuple storage: static Zig output arrays alone do **not**
make the full prediction path allocation-free. This does not add heterogeneous
tuple annotations or native union annotations.

A native probe matched six actual DSPy 3.5 schemas/annotation names, nine
validation cases and all 12 Chat/JSON/XML prompt messages, then executed all
three typed Predict paths. Additional empty-rich-tuple, sentinel, model-feedback
and clone-lifetime checks passed. A separate native Evaluate probe ran 8,192
fixed-array-input/scalar-output calls in each of Debug and ReleaseSafe with a
zero-byte general allocator backing, checksum 262,144 and no LM calls. That
narrow execution proof does not cover nested runtime output allocation.

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
Pydantic's boolean-to-integer coercion.

A separate native consumer matched **5,166 actual DSPy `parse_value` calls**
across 21 annotations, preserving model/builtin kinds, ordered dictionaries,
exact integers and binary64 bits. Its original 4,920-case baseline improved from
4,012 matches, 624 differences and 284 missing inputs to 4,920 matches with none
missing. The additional 246 cases exercise `Any`. These finite observations do
not establish every annotation or input.

Plain Python Enum classes use `Enum` / `EnumWithIdentity`, not Zig enums (which
represent string Literals). Their members retain canonical alias and class
identity through runtime values and owning clones. A native consumer matched
**3,064 actual DSPy/Pydantic comparisons**, including separate same-name classes,
value-before-name resolution, scalar validation, schemas, serialization and
typed Predict outputs through Chat, JSON and XML. `Integer` is width-independent
storage for an int annotation's actual validated result: Pydantic's lax Enum
path can return a noninteger member value without revalidating it. Native
fixed-width integer types still impose their explicit representational limits.

Mixed scalar annotations use `Literal(&.{ .{ .integer = 1 }, .{ .bool = true },
.{ .string = "ready" }, .null })`; the typed result's `.value` retains its actual
runtime primitive. Direct DSPy membership preserves the input's type (for
example, `True` for `Literal[1]`), while Pydantic validation chooses exact
bool/int/string buckets before numeric equality fallbacks. Nested lists and
optionals use the Pydantic path. A native executable matched **1,164 Source
observations**: 1,140 validation/parse calls, 15 schemas, three actual typed
Predict outputs/call counts and all six exact Chat/JSON/XML prompt messages.
Genuine Enum members are also valid arguments, for example
`Literal(&.{ comptime Color.member("red") catch unreachable, .{ .string = "red" } })`.
Validation retains the member's class identity: a raw `"red"` cannot stand in
for an Enum-only literal, and a member from an independent same-name class is
rejected. When both the member and string are allowed, each retains its own
type. Schemas project Enum values recursively, but parsing does not erase their
identity. Another native smoke matched **753 Enum-literal observations**:
720 actual validation/parse calls, 12 schemas, 12 annotation names, three typed
Predict outputs/call counts and six exact adapter prompt messages. It also
replayed the scalar observations unchanged. General Enum schemas also classify
genuine bytes, mixed bytes/text values and recursively wrapped Enum members
after projection; all-string projected values retain `type: "string"`.

Bytes arguments use `Literal(&.{python_value.staticBytes("hi")})`. These
allocation-free immutable values retain bytes identity, not string identity;
`Literal[b"hi", "hi"]` keeps both alternatives even though its JSON schema
contains two `"hi"` entries. Owning clones copy the byte buffer and free normally.
Another actual native executable matched **503 bytes-literal observations**:
480 DSPy/Pydantic validation/parse calls, eight annotation names, six valid
schemas, three typed Predict outputs/call counts and six exact Chat/JSON/XML
prompt messages. It also replayed all previous scalar/Enum observations unchanged.
Non-UTF-8 bytes validate, but cannot generate a JSON schema: native schema
generation rejects them at compile time, whereas Python raises
`UnicodeDecodeError` when schema generation is requested. Both refusal paths
were exercised; their failure timing is not identical. Dynamic annotation
construction and shared NaN identity remain unsupported.

The zero-dependency `python_literal` parser implements the supported CPython
3.11 literal grammar without execution, and `python_value` retains genuine
bytes, mutable bytearrays, tuples, sets, frozensets, complex values, Ellipsis and arbitrary-key
dictionaries rather than replacing them with JSON lookalikes. Native comparison
against **175,826 actual `ast.literal_eval` calls** matched 175,820 and differed
on six near-recursion-limit cases. Python's cutoff depends on its caller frame
depth; the native AST-depth ceiling is deterministic. This is a real parity
limit, not a passing check. Generic Python execution is still unsupported.

`python_value.bytearray(allocator, data)` copies its input into a genuine,
unhashable mutable builtin box. `bytearraySet`, `bytearrayAppend`,
`bytearrayExtend`, `bytearrayPop`, `bytearrayClear` and `bytearrayReverse` operate
on that box: copied `Value` handles remain aliases, while an owning clone has
independent storage. Self-extension and subslices remain valid when growth
reallocates. Use the original allocator for mutation/deinitialization; do not
retain borrowed views or serialized leaves across mutation. Text validation
copies valid UTF-8 into an immutable snapshot, including File fields. Numeric,
bool and typed-list validation reject bytearrays rather than treating them as
bytes.

DSPy root serialization uses `str(bytearray)`, while nested valid UTF-8 becomes
text; invalid nested UTF-8 falls back to the whole original container's string.
An independent actual-source receipt covers 265 inputs, including all 256 octets,
plus mutable aliases, growth, copying, hashing refusal and metric iteration.
Four further source cases exercise immutable text/File snapshots. Native Debug
and ReleaseSafe each matched **7,893 observations with zero mismatches**.
The initial run exposed three apostrophe-representation mismatches, fixed by
matching CPython's bytearray-specific escaping rather than changing expectations.
Error comparisons cover acceptance/refusal, not Python exception identity or
identical `TypeAdapter(bytearray)` schema-error timing. This is not the complete
Python bytearray method/buffer/memoryview API. `python_value_oracle.py
--bytearray-only` regenerates the primary development receipt.

After these cutovers, including Literal and native citation integration, the
pinned Debug and ReleaseSafe gates each passed all 344 tests and 28 build steps,
including actual RAG/bootstrap executions. Native
687-request replay and the pinned Python fixture/artifact/replay gate also
passed. These gates do not erase the literal-recursion and NaN-identity
differences, or replace authenticated inference proof.

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
`custom.get(dszig.Reasoning)`, `custom.get(dszig.BoundingBox)` or
`custom.get(dszig.Citations)` and retains class-specific validation, formatting,
equality, truthiness and cloning. Typed boxes use nominal per-type identity
tokens, not `@typeName` display strings, which can elide different generic
configurations. Custom vtables that support typed `get(T)` provide
`type_identity = runtime_value.typeIdentity(T)`; opaque callback objects do not
acquire a typed identity from their display name.

Equality is fallible: `value.pyEqual`, `value.recordEqual` and custom vtable
`eql`/`eql_value` callbacks return `ParseValue || OutOfMemory` error unions.
Callers must propagate comparison failures rather than convert them to `false`.
The same contract extends through class equality, content/request matching,
trace equality and replay; strict JSON matching still distinguishes booleans
from numbers. Container-element comparison uses `python_value.containerEqual`:
the same genuine boxed Decimal NaN is equal to itself as a list/dictionary/tuple
element without invoking Decimal equality or raising `InvalidOperation`. Direct
Decimal equality remains different: a quiet NaN is unequal to itself, while a
signaling NaN can fail. This does not add missing float-object identity.

### Native integer-subclass enums

`dszig.IntEnum(name, entries)` creates genuine immutable integer-subclass
members alongside the existing nominal `dszig.Enum` classes.
`IntEnumWithIdentity(name, entries, Identity)` supplies an explicit nominal class
identity. Aliases share a singleton. Numeric equality and hashing cross class
boundaries without making those classes identical: lax validation can resolve
an equal member of another class, while strict validation requires the actual
declared class. Reflection exposes a nominal identity token, not a type-name
string.

Exact borrowed integer payloads participate in bool/int/float/string and Decimal
validation, arithmetic, ordering and Python dictionary key equality. Numeric
operations retain integers above binary64's exact range and support genuine
complex operands; Decimal/float arithmetic remains refused. Member `str` is
numeric, while `repr` retains its class and canonical member name.

Declaration-time builtin floats truncate toward zero without routing large
integers through binary64 arithmetic. Base-ten strings accept signs, leading
zeros, digit separators and Unicode decimal digits, subject to CPython's default
4,300-digit string limit. Genuine builtin bytes use the same base-ten grammar
with ASCII digits and whitespace only, not Unicode text coercion. Conversion
precedes singleton alias resolution.

The following independent frozen-source observations matched in both native
Debug and ReleaseSafe, with byte-identical output between modes:

| Native consumer | Compared / Source rows | Outcome/value/identity/signal observations | Normalized exception identities unobserved |
| --- | ---: | ---: | ---: |
| IntEnum runtime operations | 2,294 / 2,324 | 24,577 | 657 |
| IntEnum scalar validation | 35 / 35 | 379 | 6 |
| Builtin IntEnum declarations | 19 / 19 | 134 | 0 |
| Builtin bytes IntEnum declarations | 7 / 7 | 50 | 0 |
| Configured Decimal policies | 368 / 368 | 3,950 | 233 |

The historical runtime consumer exited nonzero with 30 sequence-repetition rows
unsupported. `value.numericProduct` now repeats immutable text and genuine bytes
with integer/IntEnum multipliers in either order. A focused native consumer
compares all 30 previously unsupported frozen Source rows: 24 results and six
ordinary-Enum refusals. This does not rerun or expand the historical 24,577
observations above. One-fold results may borrow their input; repeated bytes own
their new buffer. Use caller-provided startup storage through a fixed allocator
when runtime heap allocation is prohibited. Index and output-size overflow are
refused before allocation, including out-of-range counts on empty sequences.
Unsafe unbounded repetitions were separately left unattempted by the Source
probe; normalized refusals do not establish identical Python exception classes.
The declaration probe includes fractional and maximum finite floats, signed zero, subnormals,
Unicode/separated strings, exact wide integers and the 4,300-digit boundary.
Seventeen independently compiled Debug programs also refused the source-observed
malformed strings, over-limit strings and nonfinite floats; compile-time
refusals do not establish Python exception identity. A separate new Source
probe observed seven builtin-bytes declarations and their complete schema;
native Debug and ReleaseSafe matched all 50 observations. Sixteen independent
native compilation probes refused the Source-observed invalid byte inputs,
including Unicode digits/whitespace, malformed separators and the 4,301-digit
boundary. Native compile-time refusal is not Python runtime exception parity.
Declaration-time Decimal, tuple/base arguments, arbitrary Python `int(value)` protocols, custom
mixins, configurable Python digit limits and dynamic Python class construction
are not implemented. Normalized native errors are not proof of identical Python
exception classes.

### Native Decimal values and arithmetic

`dszig.Decimal` stores a genuine runtime Decimal box; it is not a string or a
lookalike dictionary. `python_value.decimal(allocator, text)` constructs an exact
coefficient/exponent value, preserving signed zero, trailing-zero quantum and
NaN payloads. Default typed Decimal validation is finite-only and distinct from
the constructor. `DecimalWithOptions(.{ .allow_inf_nan = true })` admits genuine
nonfinite values without changing their runtime class. `Decimal.validate`
follows Python-input coercion; `validateStrict` requires a genuine Decimal
instance before applying the configured finite/nonfinite policy, retaining
accepted identity and rejecting other kinds before conversion or signaling.

`validateJson` follows JSON-input coercion, including its different shortest
plain float spelling and lax constructor-array path. `validateJsonStrict`
accepts JSON scalars but refuses constructor arrays before conversion. Quoted
JSON text retains its original quantum. Configured malformed-input coercion
preserves the actual `InvalidOperation` flag/trap transition: an untrapped
constructor can produce a quiet NaN, accepted only by the nonfinite policy.
`schemaSerialization()` provides the actual string-only serialization schema,
distinct from the validation schema. These policies, identities and all nine
signals are covered by the configured-policy observations above.

`dszig.decimal` exposes the explicitly managed context environment.
`LocalContext.enter` copies a context into the stable local guard; `deinit`
restores the previously borrowed context object. Contexts provide precision,
eight rounding modes, exponent bounds, capitals/clamp settings, and nine sticky
signals with independent traps. Native contextual add/subtract/multiply/divide
and absolute value return owned exact metadata; quiet `copyAbs` does not round
or signal. Runtime numeric helpers preserve Decimal/integer arithmetic and
refuse Decimal/float arithmetic rather than silently converting it to binary64.
DecisionState uses those same helpers before final model validation.

Generated decision evidence retains genuine nested probability models in its
raw fields. `toPythonValue(allocator, by_alias)` instead returns fresh
Python-mode dump dictionaries, recursively projecting those nested generated
models without revalidating their fields. Aliased backend projection remains a
separate API.

Native Debug and ReleaseSafe executables matched all **30,100** frozen independent
CPython/DSPy arithmetic, context and configured-decision rows: **1,462,558**
observations, including **9,465** independently executed model-dump views.
The receipt includes both successful and refused operations. Exact Decimal
signal exception classes were compared for 1,989 refusals; **9,665 normalized
exception classes remain unobserved**, not proven equivalent. A separate
genuine-Decimal container probe went from six mismatches to zero across
16 Source rows and 160 outcome/signal observations.

This is finite evidence, not complete Decimal or DSPy parity. Other Decimal
operators, general Python numeric protocols, arbitrary mutable alias graphs,
full Pydantic policies and Python exception reconstruction are not established
by those runs. The separate validation consumer matched **18,016 of 21,837
Source rows** and **24,752 actual dump calls**, with 1,521,288 observations and
zero mismatches in both native modes. This includes actual strict Enum and
Decimal boundaries, the declared numeric IntEnum class, configured nonfinite
validation and both Decimal schema modes, not synthetic consumer refusals.
It still exits nonzero for incomplete coverage:
3,120 rows are unsupported, 701 are unconsumed (including 699 preserved
invalid-baseline cases), and 35,392 Source dump frames are unobserved. Warning
emission, 1,196 identity frames and 11,331 normalized exception classes remain
unobserved. Those gaps are not passing observations.

A separate integer-division consumer matched 679 actual CPython rows through
both the direct exact-ratio helper and public `numericQuotient`, checking 1,358
outcome/binary64-bit observations. Its 53 refusal rows do not compare Python
exception identity. The public SIMBA smoke also exercised a failed program
whose metric received a genuine null prediction exactly once and returned the
Source fallback score 0.375 without any LM requests; this is not whole-SIMBA
or live-inference parity.

Typed conversion boxes the supported native model families; parsing ordinary JSON does
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

- `File.init(arena, source, keywords)` constructs a fresh typed File.
  `source` is an optional runtime value; `keywords` is an ordinary object.
  Strings are borrowed; actual Python bytes allocate a base64 data URI in the arena.
  Positional File instances omit null fields before keyword merging, unlike
  dictionaries, which retain explicit null keys. Overlapping fields are
  rejected. Copying an all-null instance with no keywords fails construction,
  whereas validating that same instance preserves it.
- `File.fromValue(arena, input)` accepts a data-URI string, an object with recognized
  File fields, an actual runtime File, or actual Python bytes. Bytes allocate a
  base64 data URI; other inputs borrow validated/stripped strings;
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

Native optimizer execution decodes `P.Input` directly from the example's
borrowed record, without allocating a temporary input dictionary. Only fields
explicitly named by `input_keys` are eligible: an existing training label
cannot silently become an input. Role checks and typed conversions follow
declaration order, preserving which failure is reported first.

Evaluation and bootstrap provision their outer prediction dictionaries once
before their example/attempt loops, using inline storage sized from the native
output field count. They reuse that storage without charging it to each
example's scratch budget. The scratch reset and error-budget policies remain
unchanged. SIMBA and GEPA retain predictions, so those paths still materialize
separate outer dictionaries rather than aliasing reusable storage.

For explicit startup preparation, initialize `value.RecordStorage(field_count)`
**in place and do not move it**, and reserve a stable `P.Output` slot. Call
`optimize.runProgramInto(P, ctx, program, example, &output, &storage.record)`.
The record must have capacity for every declared output field. Its backing
must not alias program inputs or outputs. Both the typed output slot and its
borrowed children must remain alive until the prediction is discarded or
reused. Evaluation and bootstrap reserve their typed slots before their loops.

Prepared output records also borrow runtime boxes retained by scalar wrappers
(including Decimal) and raw `Value` fields. Releasing those borrowed handles
cannot destroy the caller's scalar object; cloning still produces an
independently owned box and data. This does not change nested container
allocation or aliasing limits.

Direct Code, BoundingBox, Reasoning, File, Citation, Citations and decision
fields, including optional fields, borrow those typed slots instead of
allocating model boxes. Nominal class identity, projected specializations and
class-specific serializers remain intact. Borrowed runtime handles own neither
their boxes nor their children; releasing a handle cannot destroy the typed
slot. Cloning still invokes the original class's owning clone protocol.

Reuse invalidates the previous prediction. Any input, program or conversion
failure leaves the prepared record empty, including a conversion failure
after earlier fields were written. Metrics keep the ordinary `Record`
interface, including mutation of existing values through `getPtr`.

`optimize.runProgram` remains the materialization API for separately retained
outer dictionaries; both execution paths share input decoding and field codecs.
Use `value.cloneRecord` with an appropriate owning lifetime when retaining
borrowed model or nested values beyond their source lifetime; copying only
the outer dictionary does not extend a borrowed model's lifetime.
Nested container conversions, retained predictions, program execution and
metrics can still allocate.
This is not a whole-runtime allocator-free proof.

`optimize.Metric` and GEPA feedback callbacks can return `ParseValue` or
`OutOfMemory`. Evaluate and bootstrap apply their existing error budgets to
metric failures as well as program failures; bootstrap restores temporary
teacher demos before scoring. SIMBA retains a successful prediction and trace
when scoring fails, with a zero score. GEPA uses its configured failure score,
with an empty prediction when trace capture is disabled versus no prediction
or trajectory when capture is enabled. Reflective feedback failures propagate.
Allocation failures remain fatal. `optimize.Metric` receives a nullable
`?*const Record` prediction: SIMBA calls it with null after a program failure,
rather than skipping scoring or supplying an empty record. A callback may
assign its own fallback score; callbacks that require an actual prediction
must explicitly reject null.

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

Schema field titles share Pydantic's Unicode-aware
`name.title().replace("_", " ").strip()` behavior, including expanding title
mappings, Greek final sigma and lone surrogates. An independent native executable
matched the actual pinned Pydantic helper across all 1,114,112 Python code points
(SHA-256 `df75e4398df1eecf9e9a9a6e091e1cfb7074d205448e0b4624cf6bca52e2dbe5`)
and 22 runtime/compile-time context cases. This is finite single-code-point and
context evidence, not proof for every possible string.

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

### Answer metrics and passage boundaries

`dszig.metrics` shares DSPy's NFD/lowercase, ASCII punctuation, English article
and whitespace normalization with `ensemble.majority`. `emScore` treats two
empty normalized answers as equal; token F1 and precision return zero instead.
Repeated tokens use multiset intersection, not set intersection. `hotpotF1Score`
preserves the special `yes`, `no` and `noanswer` mismatch rule.

Each metric takes an arena allocator first. `em`, `f1` and `hotpotF1` accept a
prediction and a reference slice and return `error.EmptyAnswers` for an empty
reference list. `answerMatch(arena, prediction, answers, frac)` uses exact match
when `frac >= 1`, including thresholds above one; lower thresholds compare the
best token F1. NaN thresholds compare false after normal reference validation.

`passageMatch(arena, passages, answers)` accepts string slices and searches
contiguous DPR token sequences, not byte substrings. Thus `Paris` does not match
`Parisian`. `passageMatchValue` is the runtime-value counterpart: both arguments
follow Python iteration, so a direct answer string is iterated as characters and
a dictionary as keys. An answer normalized
to no tokens matches any passage, including an empty one, but an empty passage
list never matches. `dszig.dpr.normalize` exposes the underlying DPR tokenization:
NFD followed by runs of Unicode letters/numbers/marks or individual nonseparator,
noncontrol characters, then lowercase **per token**. The character classes come
from the locked `regex` package, not CPython's older Unicode category tables;
both still use CPython's NFD and lowercase data.

`answerExactMatch(arena, &example, &prediction, frac)` and
`answerPassageMatch(arena, &example, &prediction)` accept native `Example` and
`Record` values. The latter retains Python's context iteration and laziness:
a string context is iterated as characters, a dictionary as keys, and an empty
context does not inspect individual elements of an answer list. Unicode helpers
preserve Python lone-surrogate code points using WTF-8 rather than replacing
each encoded byte.

These are native typed APIs, not Python's dynamic argument/exception model.
F1/precision return numeric results without DSPy's ambient timestamped stdout
diagnostic for two empty token lists. Python exception classes/messages, implicit
printing and arbitrary Python iterators are not claimed identical.

The independent `answer_metrics_oracle.py` observed **2,832 actual DSPy calls**
across all 13 metric/tokenization functions. Native executables in both Debug
and ReleaseSafe matched every accepted value and error acceptance result,
including 391 source errors. The source records 73 diagnostic stdout messages;
these were not compared or counted as reproduced side effects. An initial
31 passage-iteration mismatches were fixed in the production runtime-value API
without filtering cases or changing source expectations.

A separate exhaustive sweep over all **1,114,112 Python code points**, including
lone surrogates, matched both actual-source hashes. Each normalized output is
encoded as a u32 little-endian byte length plus UTF-8/WTF-8 bytes; each DPR output
starts with its u32 token count and then the same length/byte encoding per token:

- `normalize_text`: `104e026eca607a2e5ec6d9e3980220d8afbd36f42a89b15f8a128e085c50aa1d`
- `DPR_normalize`: `94bb69feba0d503671fda76e3b43b3c99345c295f69e2ed9fb2dd65a72eb256d`

This is exhaustive single-code-point evidence, not proof for every string or
every DSPy capability. The shared WTF-8 repair first failed three actual-source
surrogate comparisons and then passed the same unchanged expectations.
Pinned Debug/ReleaseSafe each passed **334/334 tests and 28/28 build steps**,
actual RAG/bootstrap, native **687-request replay** and the Python
fixture/artifact/reference gate, including regeneration of the DPR tables.

### Typed decisions, evidence and SystemOne

`Noul`, `Score` and `Choice` are genuine native model classes. Configure graded
and categorical decisions with `ScoreWithLevels` and `ChoiceWithOptions`;
`NoulWithCriteria` configures boolean criteria. Descriptions are JSON values,
not a text-only approximation of DSPy's criteria:

```zig
const Quality = dszig.ScoreWithLevels("[\"poor\",\"fair\",\"excellent\"]");
const Category = dszig.ChoiceWithOptions(&.{
    .{ .value = .{ .string = "billing" }, .description_json = "\"Payment issue\"" },
    .{ .value = .{ .string = "technical" }, .description_json = "\"Product bug\"" },
});
const Judge = dszig.Signature(struct {
    pub const Input = struct { text: []const u8 };
    pub const Output = struct {
        flag: dszig.Noul,
        rating: Quality,
        category: Category,
    };
    pub const instructions = "Judge the supplied input.";
});
```

`Predict(Judge)` selects decision adaptation automatically. Its typed outputs
retain boolean/score/typed-choice values, confidence and model-specific
probability/level fields. Generative decisions work through the ordinary
Chat/JSON/XML LM path. Labeled task examples are incorporated into decision
instructions rather than passed to the evidence signature as ordinary demos.
Completion extras and their insertion order survive decoding.

Plain `bool` and scalar `Literal` outputs are not automatically decisions.
Explicit compatible metadata uses the existing description declaration:
`.flag = .{ .text = "Evaluate flag.", .metadata = &.{ConfiguredNoul} }`.
A matching configured Choice can similarly annotate a Literal output. These
native Annotated forms produce the primitive output while retaining the
configured decision model for evidence and criteria. Float/Score metadata and
arbitrary runtime annotations are not implemented.

`Predict.state.decision_fields` is the mutable decision configuration, separate
from signature-field metadata. `getCriteria(arena, name)` returns an owning
copy; `setCriteria(arena, name, value)` validates before committing an owning
copy. State dump/load retains decision settings and DSPy's partial-commit
behavior when a later legacy signature restore fails. Direct
`decision_state.State.decode` consumes integer-keyed Python dictionaries for
Score distributions; the generative Predict path validates string aliases and
converts them before recording and decoding.

`decision_evidence.evidenceType(Model)` and `probabilitiesType(Model)` expose
the genuine generated evidence classes. They provide strict/lax/JSON
validation, nominal class identity, field metadata, raw attributes,
`toAliasedValue(arena)` wire projection, schemas, iteration and owning clones.
Raw probability attributes are named `option_0`, `option_1`, etc.; wire aliases
retain the actual labels. `Context.decision_evidence` can point to an initialized
`DecisionEvidence` collector (`.arena` and `.io` required). Recording takes
locked owning snapshots before model decoding. Native SystemOne validates all
BoundingBoxes first and records boxes before decisions; invalid geometry
records neither.

`lm.TypeSafe` / `TypeSafe` uses the existing bounded native HTTP transport for
`POST /v1/systemone`. Its LM advertises decision-request capability, retaining
structured state, questions and response schema through recording/replay/cache
wrappers. OpenAI refuses direct native decision requests rather than silently
dropping them. The TypeSafe box route sends the experimental images header;
this is **not** an Image/assets implementation or the complete provider SDK.
Usage/request-id/convenience views, Python exception identity, asynchronous
execution and streaming remain outside this native decision implementation.

The frozen independent DSPy 3.5 source receipt contains **15,218 actual rows
and 17 model definitions**. Two separate consumers report the following finite
coverage; their entries/observations overlap and must not be added together:

| Independent consumer | Fully supported entries | Unsupported entries | Unconsumed entries | Compared observations | Mismatches |
|---|---:|---:|---:|---:|---:|
| Decision models and views | 8,216 | 1,152 | 5,867 | 81,843 | 0 |
| Evidence/state/Predict pipeline | 1,879 | 501 | 12,838 | 15,751 | 0 |

Both consumers deliberately **exit nonzero for incomplete coverage**.
Partially compared unsupported rows contribute observations, not a claim that
their entire operation is supported. Six candidate-client rollout observations
are unobserved: the source engine receipt does not capture those client-only
options. Decimal, dynamic subclasses, arbitrary forged model state (including
bytearray values in statically typed Noul/Score fields) and other unsupported
operations remain in the receipt, not replaced with JSON lookalikes. ScriptedLM
now refuses non-JSON provider data at response construction, before adapter
decoding or evidence recording; this is not a complete SDK response-object API. Bare Score/Choice evidence refusal also differs in failure timing:
native compile-time rejection versus Python runtime rejection. These results
do not establish complete decision-model or DSPy equivalence.

A focused actual-source/native smoke reproduced seven mismatches in 15 Choice
weight cases before repair and zero afterward, including adjacent integers above
`2^53`/`2^100` and integer `0 * 10^400` versus float `0.0 * 10^400` overflow.
Another 13 actual Python products and 14 numeric comparisons exercise the shared
exact-integer arithmetic. A real localhost TypeSafe HTTP/Predict smoke matched
all three request bodies **byte-for-byte** against the actual SDK provider,
including route, fixture authorization, JSON and experimental box headers.
Typed valid output, geometry refusal before evidence recording and boolean
probability refusal were exercised. This is HTTP/codec/prediction evidence,
**not authenticated homelab inference**.

Both decision consumers produced the same counts in Debug and ReleaseSafe,
including the deliberate incomplete-coverage failures. The bytearray change also
passed pinned **355/355 tests and 28/28 build steps in each mode**, actual
RAG/bootstrap execution, the complete 687-request native replay and the
independent Python fixture/artifact/reference gate. Unicode and DPR tables
regenerated unchanged.

### Citations and provider references

`Citations.Citation` and `Citations` are genuine native model types, not aliases
for dictionaries. They retain typed validation, raw/model serialization,
Python-style display, equality, truthiness, iteration and owning clones.
`Citations` supports negative indexing and slicing; slices borrow their Citation
elements. `fromDictList` is a separate iterable constructor, not the annotation
validator. An owning clone copies nested fields and survives the original arena;
borrowed iterator values do not recreate Python garbage-collector lifetimes.

LM completions carry canonical `inference.CitationPart` references independently
of their text, tools and reasoning. Recording, replay and caching retain them.
Responses annotation extraction follows the actual SDK's text/title/URL
fallbacks and Unicode-code-point spans, including Python integer conversion.
The pinned Chat Completions reader ignores message annotations; it does not
share Responses extraction. Chat text blocks and reasoning/refusal parts retain
their actual nullable-text behavior and canonical tool-ID indices.

After normal adapter parsing, nonempty canonical references replace an exact
direct `Citations` output. They do not repair invalid text, bypass parsing or
override optional/container annotations. Citation fields remain in prompts.
Generated JSON response schemas undergo the separate DSPy request-builder
closure step; caller-supplied raw response-format dictionaries remain untouched.

Independent installed-source oracles and native executables matched:

- **10,484 observations across 1,572 citation model/protocol/schema/adapter/Predict
  rows**, including exact prompts and complete generated request bodies;
- **143 provider cases**, comparing acceptance, nullable text, reasoning,
  citation fields and decoded tool calls;
- **175 generated-request schema cases**, including nonmutation, nullable
  defaults, nested unions and source-specific reference resolution.

The model consumer deliberately exits nonzero for **150 unsupported input rows**
requiring genuine Decimal, bytearray or dynamic subclass/lookalike-model types.
Provider stream readers/events, LiteLLM-specific SDK views, listener scheduling
and streamify cases are also unconsumed by this model proof. Those rows are
retained, not converted to fake dictionaries or counted as matches. Python
exception classes/messages, arbitrary mutable aliases and all Python class
operations are not identical.

Actual `zig build live -- --smoke` runs through localhost Responses and Chat
HTTP returned answer `4`. The recording retained canonical Responses references
and omitted Chat annotations exactly as the pinned reader does. This is real
HTTP/codec/CLI evidence, **not authenticated homelab inference**.

All three differential consumers produced the same comparison results in Debug
and ReleaseSafe, including the explicit unsupported-row failure. The pinned
344-test/28-step Debug and ReleaseSafe gates, actual RAG/bootstrap, 687-request
replay and independent Python fixture/artifact/reference gate also passed.

## Native LM bindings and program state

`Predict.state.lm` is an optional borrowed LM handle. A predictor selects an
explicit `Context.lm_override`, then its own binding, then `Context.lm`.
Set `lm_override_present = true` with a null override to bypass the binding
and select the context fallback, matching DSPy's explicit `lm=None`.
A bound predictor does not require a context fallback; an entirely unconfigured
call returns `error.NoLM`. Capability selection, JSON fallback, native decisions
and candidate fanout all use the selected handle.

`predict.setLM(arena, &program, model)` binds every discovered predictor;
passing null clears the bindings. `predict.getLM` returns the unique borrowed
handle, or null if every predictor is unbound. Mixed bound/unbound predictors,
distinct LM identities, and modules without predictors return `error.MultipleLMs`.
Program value copies retain model identity without owning the model.
`PredictState.reset` and `predict.resetPredictors` clear both demos and bindings.

Program save/dump now stores a bound LM's configuration rather than silently
writing null. `lm.OpenAI` supports native configuration persistence, including
its API, model, capability flags, generation token limit, response bound, retries
and timeouts, plus `finetuning_model`, `train_kwargs` and `launch_kwargs`.
Nested training/launch options are JSON objects owned by the startup load arena.
The native class discriminator and transport settings are **not
Python LM-state interchange**. API keys, HTTP clients, trust stores and header
providers are never serialized. Custom engines must provide `LM.VTable.dump_state`
and an application-owned `LM.StateLoader`; inference-only scripted, replay, cache
and recording handles do not automatically gain persistence.

Loading has an explicit fourth argument:

```zig
var loader: dszig.lm.OpenAI.StateLoader = .{
    .client = &http_client,
    .base_url = trusted_base_url,
    .api_key = credential,
};
try dszig.state.load(startup_arena, &program, saved_json, .{
    .lm_loader = loader.loader(),
});
```

Use `.{}` when loading programs without bound models. A nonempty LM state
without a loader fails with `error.MissingLMLoader`, never a fake restored model.
By default, loading removes the top-level `api_base`, `base_url` and `model_list`
routing keys and uses the explicitly injected endpoint. Preserve saved routing
only with `.allow_unsafe_lm_state = true` for trusted files. Credentials always
come from the loader. Native loaders use static class dispatch, not serialized
function pointers or dynamic Python imports.

The startup arena owns reconstructed models and copied configuration strings;
the injected client, trust and authentication resources must outlive the loaded
program and every joined call using it. Restore constructs each predictor's LM
separately, just as the source does: shared identity before saving need not remain
shared after loading. A failed LM restore retains the source's earlier signature,
demo and decision-state commits, while leaving the previous LM binding intact.

An independent native executable matched **24 complete normalized observations**
from actual DSPy 3.5: binding/override/null precedence, fanout, copy/reset behavior,
module identity errors, secret redaction, trusted/untrusted routing, reconstruction
and failure transitions. Native error enums are mapped to corresponding source
refusals for that comparison, not claimed to be identical Python exceptions.
Debug and ReleaseSafe also exercised a restored model through actual localhost
HTTP and typed Predict, returning `42`; the fixture verified model, token limit
and injected authentication. This is not live-model inference, training-job,
whole-runtime static-memory, or complete LM/provider save/load parity evidence.

## Native OpenAI training jobs

`LM.finetune(&job, request)` starts genuine OpenAI-compatible HTTP training in a
caller-owned `training.Job`. Unsupported engines return `error.UnsupportedFinetuning`.
Initialize the immovable job with `Job.init(io, buffers, limits, thread_config)` at
startup: it creates one persistent worker. Supply disjoint bounded body/scratch
buffers and explicit row, poll, whole-job, whole-request, cleanup and poll-interval
limits. The client, authentication provider, dataset and model configuration are
borrowed until completion; do not mutate them concurrently.

Requests accept `.chat` or `.completion` data. Like the source provider, this
checks the format tag, not training-row schema. The worker uploads JSONL with
purpose `fine-tune`, creates `/fine_tuning/jobs`, polls status and the latest
event, and obtains the succeeded job's trained model. A configured nonempty
`finetuning_model` overrides the inference model. Empty/absent per-job
`train_kwargs` fall back to the LM's defaults. OpenAI's provider prefix is removed
from the wire training model; the returned trained name is not prefixed again.
`launch_kwargs` persist as configuration; this OpenAI provider does not launch a
local training engine.

`Job.wait()` joins the submitted operation and returns its actual model or error.
The returned OpenAI borrows the job's inline model-name storage until reuse.
`OpenAI.copyWithModel(&target, model_name)` copies configuration without allocation,
retaining inference clients/authentication/options and borrowing the supplied name.
Copy that name into caller-owned stable storage before retaining a result across
job reuse. `Job.status(arena)` queries fresh remote status with separate caller
scratch; do not read mutable job fields while work is running.

`cancelAndWait()` requests cancellation, cleans up known remote resources and
waits for the local completion barrier. Every request races a whole-request timer;
both sides are canceled and joined before scratch reuse. The supplied Io backend
and authentication callbacks must cooperate with cancellation; a non-cooperating
caller callback can delay the barrier. `release()` cancels any
remaining nonterminal remote job and deletes the training file. It must succeed
before another `start`: failed cleanup retains IDs and refuses reuse.
`deinit()` cancels, joins the startup worker, and reports remaining cleanup errors.
Report those errors rather than silently dropping them. A response lost after a
remote commit can leave an unknown remote object: no automatic create retry or
claim of guaranteed remote rollback is made.

This intentionally does not reproduce the source's unbounded polling or its
cancel-during-poll hang. Native failure results are errors, not Python exception
objects returned as successful Future values. Polling avoids redundant SDK
existence/estimate retrieves; event IDs are deduplicated against the immediately
preceding ID. Native limits include 1,024-byte retained identifiers/model names
and the supplied response/body capacities. Before submission, an iterative
fixed-depth traversal bounds JSON input bytes/nodes by body capacity, including
hyperparameters; huge inputs cannot evade work bounds by producing tiny output.
Non-JSON custom runtime values and invalid UTF-8 refuse rather than executing a
user serializer on the worker. HTTP-client and Io internals may allocate:
bounded job buffers are not whole-runtime static-memory proof.

The actual DSPy provider was exercised with an in-process scripted SDK; the
OpenAI SDK itself is absent from the pinned reference environment. Native
localhost HTTP exercised upload/create/poll/events, trained-model typed Predict,
failure, remote cancellation, poll/request bounds, cancellation and slot reuse.
These are finite protocol/lifecycle checks, not paid training, authenticated
model inference, SDK/error identity, all providers, training-job persistence,
`BetterTogether`, DST/PBT or full DSPy parity.

## Native BootstrapFinetune

`BootstrapFinetune.compile(P, out, io, ctx, &student, trainset, teachers, slots,
trace_options)` collects actual native traces, formats training rows and starts
real `LM.finetune` jobs. It mutates the supplied student; it neither resets nor
copies it, and an already compiled root is accepted. Predictor bindings are
required even if `ctx.lm` supplies inference: like Source, that validation occurs
**after** teacher trace collection.

Initialize every `slots[i].job` at startup using the training API above. Supply
explicit limits for teachers, examples, predictors, emitted training rows and
visited trace steps. `out` holds discovery, retained traces and training JSON;
use distinct bounded startup backing for it and attempt scratch. The optimizer
does not own a heap or create worker threads. Without `trace_options.rollouts`,
collection is serial; with it, collection uses that startup pool and its private
attempt contexts. `num_threads` (default eight; zero selects eight) bounds job
count and refuses a trace pool whose exposed worker count exceeds it.
`parallel.Runner` exposes that startup `workers` count as well as task/attempt
capacities.

`teachers = null` uses the actual student. Explicit lists accept heterogeneous
native programs through `bootstrap_finetune.Teacher.init(T, &teacher)`, including
different signatures with the same predictor names and count. Explicit shared
predictor identities are refused, including an explicit student alias.
A `.student` list element represents Source's `None`; an empty list collects
nothing and still prepares empty datasets for the bound student models.

Groups use LM identity and, unless `multitask` is true, predictor index. Selection
uses the **trace position**, not the traced predictor identity; formatting uses
that step's own signature and demonstrations. With a metric, nonzero numeric
scores survive, including negative values and NaN. Each group uses a fresh
Python-compatible `Random(0)` shuffle. `exclude_demos` removes formatter demos
and, after success, student demos.

`adapters` and `train_kwargs` accept a `.global` value or explicit `.per_lm`
identity entries. Missing keys in nonempty maps refuse. A null adapter uses
`ctx.adapter`, not an LM property. As in Source, an empty adapter dictionary
cannot infer a format; empty training kwargs use the provider's defaults.
Chat/XML prepare chat data. JSON accepts an empty selection but its formatter
refuses selected rows; TwoStep cannot infer a training format even when empty.
A selected failed prediction is not an answer dictionary: Chat/XML refuse its
missing mapping protocol when outputs are declared, rather than inventing
answers from its diagnostic attributes.

All starts precede waits. On failure, every started job is canceled and joined,
including jobs Source abandons on its first failure. All releases are attempted;
cleanup errors take precedence and retain IDs so those jobs cannot be reused
unsafely. The student is rebound only after every job and cleanup succeeds.
Each source LM is killed immediately before its training job starts, as in
Source. A native lifecycle hook must join its workers before returning and
support repeated kills, including shared-LM groups and error cleanup.
OpenAI-compatible remote API LMs need no local lifecycle hook; this does not
implement a separate local inference engine.

A slot owns its retained trained-model name and copied OpenAI configuration.
Slots must remain unmoved and alive while compiled bindings use them; reusing a
slot explicitly invalidates its previous compiled model. Original clients,
authentication and nested model configuration remain borrowed. Result storage
is written only after **all** workers and cleanup have stopped reading inputs,
so a reused slot cannot overwrite another running job's borrowed model name.
Formatting/tracing scratch can be reclaimed after compile joins.

This is native-program orchestration over the implemented OpenAI provider,
not arbitrary Python module execution, every provider's training API, paid
training, every BetterTogether optimizer option, or whole-runtime
allocation-freedom. The bounded training/HTTP limitations above still apply.

The native HTTP consumer exercised 23 scenarios in both Debug and ReleaseSafe,
including parallel traces, heterogeneous and mixed student/teacher lists,
separate/shared LM groups, negative/NaN scores, late missing-LM validation,
adapter/config errors, thread/slot limits and joined job/start failure cleanup.
Each mode made 183 localhost requests with 14 uploads/job creations. Complete
training rows matched actual DSPy compile/trace output for six selected
comparisons per mode, including all fields and shuffle order; remote group
arrival order was not compared. Compiled typed predictions returned `42` after
the 1 MiB compile backing was overwritten. Actual Source had 106 finite oracle
cases; its training jobs were scripted real futures, not provider/SDK/network
work, and those 106 cases were not all matched to native execution.

## Few-shot compile options

`LabeledFewShot.compile(P, out, student, trainset, options)` accepts
`.{ .sample = true }` for CPython-compatible sampling or
`.{ .sample = false }` for prefix selection. Its signed `i128` count preserves
Python negative slicing in prefix mode. A negative sampled count fails only
when a nonempty dataset has a predictor; empty data returns a reset copy first.
Both modes own deep-copied demos in `out`. DSPy aliases the original Examples,
so native ownership is an intentional difference, not alias-identity parity.

`BootstrapFewShotWithRandomSearch.compile(..., valset, options)` accepts
`restrict: ?[]const i64` and `labeled_sample: bool`. Restrictions are a borrowed
seed set: duplicates and out-of-range entries are ignored, candidates retain
ascending seed order, and no matching seed returns `RestrictMatchesNoSeed`
before candidate work. `labeled_sample` changes only the labeled seed `-2`.
The reset, labeled and fixed bootstrap seeds permit zero bootstrapped demos;
a consumed random seed instead returns `InvalidBootstrapRange`. Early stopping
can avoid that failure. Teacher-list refusal is likewise deferred until a
bootstrap candidate actually consumes the teacher.

BetterTogether's `optimizers.LabeledFewShot` and `optimizers.RandomSearch`
expose `compile_options`; `Defaults.prompt_compile_options` configures the
default `"p"` step. These are typed native options, not arbitrary Python kwargs.
Random-search candidate reports still contain seed and score, not the Source
candidate programs and per-example subscores.

The Debug and ReleaseSafe public-API consumers each matched 203 value
observations and 13 refusal
categories across 32 labeled and 30 search Source cases, separately recording
13 ownership differences. Its BetterTogether paths passed 67 checks.
Eight additional actual Source zero-bootstrap cases matched 31 value
observations and three refusal categories after reproducing and fixing the
former eager native assertion. These execute real optimizer methods without
model inference; categories do not reproduce Python exception classes/messages.

## Bounded native Avatar

`Avatar(Sig)` and `AvatarWithOptions(Sig, .{ .actions_max = 8,
.tools_max = 32 })` build native actors with statically generated signature
layouts. Initialize with `Program.init(arena, tools)`. Tools use
`avatar.Tool`/`avatar.Runner`, distinct from ReAct's root-level `Tool`.
`forward(ctx, input)` runs on a copy; mutable `call(ctx, input, options)`
preserves actor evolution after failures. Successful calls restore the actor
and return the task outputs plus ordered `actions`. After an evolved failure,
another mutable call refuses before LM use until `reset()`.

The action capacity is a real safety bound: the next decision may finish, but
cannot execute another tool. Exhaustion returns `AvatarActionsExhausted`,
never a fabricated final answer. Source's `max_iters` loop does not provide
that bound; native call options preserve its observed behavior within the
explicit capacity. Native tool callbacks are borrowed, and evolved signature
class identities are static per layout rather than freshly generated Python
classes. Typed inputs require all declared fields.

`AvatarOptimizer.compile(P, out, gpa, ctx, rng, student, trainset)` evaluates
native actors serially, partitions examples by score, samples positives before
negatives from the explicit CPython-compatible generator, then runs comparator
and feedback-instruction Predicts. It applies instructions only on a strict
improvement and returns the newly proposed instruction without evaluating that
proposal again. Reports use raw mean scores, not Evaluate percentages.
Failed examples score zero except allocation failure, which propagates.
Returned metadata/instructions live in `out`; tool callbacks remain borrowed.

DSPy 3.5's original Avatar and AvatarOptimizer constructors fail because they
refer to removed `dspy.TypedPredictor`. This port follows saved observations
of unchanged Source methods on explicitly initialized real `dspy.Predict`
instances, not working original constructors or unmodified end-to-end Source
execution. Debug and ReleaseSafe native HTTP smokes each completed 22 actual
requests, including tool execution, two optimizer iterations and execution of
the optimized actor; both modes closed and joined the fixture. This is
localhost transport evidence, not authenticated model inference or
whole-runtime static-memory proof.

Final Debug and ReleaseSafe gates each passed 496 tests and 34 build steps,
including examples. The saved actor-method receipt covers 39 rows: each mode
passed 7,674 Source-backed value/metadata/message comparisons and 19
refusal-category comparisons, separately reporting 226 native invariants, nine
deliberate differences and 65 unsupported-category reports. Two requests
belonging to missing typed input remain explicitly unconsumed; three requests
are unconsumed only in the supplemental capacity-limited replay. These are not
parity matches. The unchanged expectations first exposed 136 prompt mismatches,
fixed by propagating DSPy model-module metadata into shared XML annotation
routing: DSPy-owned BaseModels use JSON leaves while ordinary models retain
nested XML.

The optimizer-method consumer accounts for all 25 saved rows: 20 matches,
four refusal categories and one deliberate serial-context difference. All 13
ordered sampling rows match. Its 2,677 checks include exact feedback inputs and
prompts, but are checks, not 2,677 independent Source observations. Actor prompt
bytes absent from that optimizer receipt, Python exception identity, dynamic
class/object identity and Source thread scheduling are not established by it.

## Native BetterTogether

`BetterTogether.compile(P, out, gpa, io, ctx, &student, trainset, options,
defaults)` runs real native prompt and weight optimizers. Its default strategy
is exactly `"p -> w -> p"`: `BootstrapFewShotWithRandomSearch`,
`BootstrapFinetune`, then prompt search again. `defaults` supplies the weight
limits and fresh slot sets; optional typed `prompt` and `weight` configurations
replace the corresponding defaults.

Set every student predictor's LM explicitly before compiling; `ctx.lm` alone
does not satisfy that requirement. Already compiled students refuse. Explicit
teachers use `bootstrap_finetune.Teacher.init(T, &teacher)`. Top-level
`options.teachers` is normalized as a list, including the Source distinction
between an implicit student and an explicit `.student` (`None`) list element.
Teacher paths/counts, not native signature equality, determine compatibility.

`compileWith(..., options, .{ .p = &prompt, .w = &weight, ... })` accepts an
anonymous registry of borrowed native adapters. Exact field names are strategy
keys, separated by `" -> "`. Builtin adapters cover RandomSearch, LabeledFewShot,
BootstrapFewShot, MIPROv2, COPRO, GEPA, InferRules, SIMBA, Optuna and
BootstrapFinetune. Custom entries implement
`compile(self, comptime P, better_together.optimizers.Request(P)) !P`.
In-place entries declare `returns_in_place = true` and mutate `student_ref`;
others return a value that is published only after restoration and launch
succeed. InferRules borrows its initialized optimizer to retain induction state.

`options.optimizer_compile_args` supplies per-key dataset and teacher
overrides. `valset = .{ .value = null }` explicitly removes validation for that
optimizer call; omission preserves the standard value. The singular `teacher`
override is `.none`, `.program` or `.list`. For example, a single typed teacher
is `.{ .program = bootstrap_finetune.Teacher.init(P, &teacher).program }`.
Unlike a top-level singleton list, that value works with prompt optimizers
expecting one program. Explicit arguments unsupported by a builtin optimizer
refuse before launch; optimizer-specific settings remain typed on the adapter,
not a Python keyword dictionary. Prompt adapters currently require a single
teacher of the same nominal `P`; heterogeneous erased teachers remain supported
by BootstrapFinetune. Lists are not silently coerced to single programs.

The result contains `program`, `candidate_programs` (program, optional score
and strategy prefix), and `flag_compilation_error_occurred`. Validation ties
retain the earliest candidate; absent or empty validation selects the latest
successful candidate. Evaluation failures score zero. A failed optimizer,
snapshot or replacement launch stops composition and retains earlier
candidates. If any predictor loses its LM, all previous bindings are restored
positionally. Named LMs relaunch when the model-name list changes, not merely
when handle identities change.

Candidate snapshots clone LMs, preserving shared LM identity within each
candidate and distinct clones across candidates. A custom LM must implement
the real `clone` hook or compilation refuses. OpenAI snapshots own model/URL
strings and nested training/launch configuration, but borrow the initialized
HTTP client, authentication and callback resources. They have no Python
engine store to copy. Signature-class instruction mutations propagate through
the reachable current/original programs, candidates and explicit teachers,
not arbitrary global objects. Const teachers remain readable, but an operation
requiring instruction writes refuses rather than casting away their constness.

Supply explicit limits for strategy bytes/steps, examples, predictors and
teachers. `out` is caller-owned arena or fixed startup backing retaining the
result graph; release it as a unit only after all candidates are finished.
Every weight attempt needs distinct, unused slot/name storage while any
current/original/teacher binding borrows it. Startup jobs may be shared across
sequential slot sets because training and cleanup join before reuse.
Lifecycle cleanup attempts every unique handle even after a hook error.
The composition seed is an optional native `u32`; omitted seeds use secure
randomness. Attempt/provider allocators and HTTP resources retain their existing
contracts: bounded metadata backing does **not** prove whole-runtime
post-initialization allocation freedom or generated-code process isolation.

The HTTP consumer exercised 23 scenarios in both Debug and ReleaseSafe,
including unmodified default prompt search, repeated weight steps, teacher
overrides and alias propagation, GEPA's empty sampled holdout, slot refusal,
failed training and a provider requiring shutdown before training.
It made 637 localhost requests and created
13 training jobs per mode. Eighteen matched actual DSPy scenarios passed
122 comparisons per mode of selection, candidate metadata, failure boundaries
and complete SDK training rows. Error categories and model-name equality
relations were compared, not
identical exception types or fixture-specific generated model strings.
Candidate clone invariants were checked separately. Eight additional lifecycle
control cases exercised failure cleanup with synchronous resource tokens; those
are not inference, actual worker-join or subprocess-isolation evidence.

## Isolated generated native Zig

`dszig.generated_zig` composes a genuine `Predict(CodeWithLanguage("zig"))`
with compilation and execution of native Zig. `Generator` has ordinary
discoverable predictor state, demos, LM bindings, adapters and tracing.
`generateAndRun(&generator, &executor, context, input)` generates code through
that predictor, then executes it; it adds no execution retry or alternate model.
`input` contains `task`, `input_json` and `capabilities` (exact native names and
argument/result descriptions). The returned JSON borrows executor storage until
its next run.

This is **not** an arbitrary Python interpreter or an implementation of Python
RLM/Flex. There is no unsafe in-process or Darwin fallback. Execution requires
64-bit aarch64/x86_64 Linux, working unprivileged user/PID/network/mount namespaces,
Landlock ABI 7 or newer, seccomp, and an owned delegated cgroup-v2 domain with
memory and pids controllers. Supply the pinned Zig compiler and its complete
immutable `/nix/store/...` runtime closure. Nix is used to prepare that closure,
not invoked for individual executions. Missing prerequisites refuse execution.

At startup, create a private owner-only directory directly under `/tmp/dszig-*`
and an **empty, owned** delegated child cgroup. Open its directory with CLOEXEC
and initialize `generated_zig.Lane.configure(fd, limits)`. This verifies the
cgroup-v2 filesystem, configures and reads back `memory.max`, `memory.swap.max=0`,
`memory.oom.group=1` and `pids.max`. The executor stays outside that lane.
Initialize the large `Executor` **in place** with `Options`:

- `lane`: a borrowed initialized lane.
- `worker`: `workspace`, absolute `compiler`, exact `read_only_roots`, and
  `artifact_bytes` (default 128 MiB, maximum 256 MiB).
- `tools`: a startup registry of exact names, native context pointers and bounded
  callbacks. A callback writes one JSON value into the supplied fixed buffer and
  returns its byte length; its final argument is an absolute monotonic deadline.
- `timeout_ms`: one absolute deadline covering preparation, compiler and runtime,
  not an inactivity timeout (default 60 seconds, maximum 120 seconds).
- `max_calls`, `max_frame_bytes`, `max_work_bytes`: cumulative RPC limits;
  hard maxima are 256 calls, 8 MiB framed bytes and 16 MiB validation work.

Options, closure paths, lane and native callback contexts must remain stable for
the executor's lifetime. Each executor/lane has one exclusive owner; do not run
the same executor concurrently or share its lane with another active executor.
An earlier result expires when the next run starts: copy it into caller-owned
startup-provisioned storage before using it as that next run's input.
The lane retains the kernel's effective page-rounded memory bound, never a value
larger than requested. Registered native callbacks are **trusted bounded host
code**, not sandboxed code: they must honor the supplied deadline. The executor
checks before/after dispatch but cannot preempt a noncooperative native callback.
`Tool.call(context, &arguments)` is the shared typed-argument normalization path
also used by ReAct; the qualification consumer uses it for native capability RPC.

The compiler and guest have different kernel policies. A startup-gated direct
PID1 child receives its UID/GID mappings and aggregate cgroup membership before
work begins. It mounts a fresh quota-limited tmpfs and proc, installs resource
limits, and validates source inside the child using fixed parser scratch.
Compiler subprocesses can read/execute only the immutable closure and
read/write the private workspace; sockets, new namespaces and tracing are
denied. Compiler output goes only to bounded diagnostics. The supervisor kills
and reaps remaining compiler descendants **before** starting the runtime.
The statically compiled guest enters a second syscall allowlist before generated
code: only stdin reads, stdout/stderr writes, basic exit and signal-return/mask
operations remain. Heap mappings, file access, sockets and process creation
terminate it.

Generated source has a 64 KiB / 4096-token / 4096-node limit and must contain
exactly `const guest = @import("guest");` plus the sole public entrypoint
`pub fn run(ctx: *guest.Context) !void`. Other imports, assembly, extern/export
and unsupported builtins are rejected. This source guard is **not the sandbox**:
`guest.std` reaches operating-system APIs, so the kernel policy remains required.
Guest methods use fixed buffers:

```zig
const guest = @import("guest");
pub fn run(ctx: *guest.Context) !void {
    const reply = try ctx.tool("lookup", try ctx.input());
    try ctx.finish(reply);
}
```

Arguments, replies and final output must be complete bounded UTF-8 JSON values.
Names are matched exactly; unknown tools never dispatch. JSON depth, total work,
call count and frame bytes are checked independently by the host. A finish frame
is pending, not success: publication requires successful process exit, the direct
namespace PID1 reaped, the lane empty, and both streams drained through EOF with
no trailing or partial frame. Finish-then-crash, finish-then-hang and extra bytes
revoke output. An unread/reset protocol stream also prevents publication, even
when the child exits zero. Ordinary source/compile failures retain diagnostics and
report `ChildFailed` after joining, rather than hiding them as socket resets.
Failure attempts both recursive cgroup kill and direct PID1 kill and joins before
reuse. If cleanup cannot be proved within its bounded attempts, the executor returns
`ExecutorQuarantined`, retains its child handle and refuses subsequent runs;
it never reports a clean join or reuses that storage.

The executor/session/guest use startup-provisioned fixed storage; this does not
claim the ordinary model client, all optimizers or the whole runtime are
allocation-free. Authenticated model inference remains a separate gate.
A localhost HTTP provider fixture proves transport, typed `Code[zig]` decoding,
actual isolated compilation and native execution, **not** real-model inference.

`tools/generated_zig_consumer.zig` is the Linux qualification CLI. Supply a
`compiler_closure` module containing `pub const paths = [_][:0]const u8{ ... };`
from the compiler's actual recursive Nix closure. Build with the pinned compiler:

```sh
zig build-exe -O ReleaseSafe --dep dszig --dep compiler_closure \
  -Mroot=tools/generated_zig_consumer.zig \
  -O ReleaseSafe -Mdszig=src/root.zig \
  -O ReleaseSafe -Mcompiler_closure=/absolute/path/compiler-closure.zig \
  -femit-bin=/tmp/generated-zig-consumer
```

Run `/tmp/generated-zig-consumer LANE WORKSPACE COMPILER CASE TIMEOUT_MS [BASE_URL]`.
`success` and the malformed-frame/tool/OS/finalization cases each verify successful
reuse of the same executor; `generated` calls an explicit OpenAI-compatible
endpoint. The CLI's default key is explicitly fixture-only; use an explicit
endpoint and `OPENAI_API_KEY` for an authorized real provider. Its native tool
checks that only the supervisor and runtime remain at dispatch: no compiler
descendant may survive into the runtime phase.

The finite qualification for this slice passed the pinned repository's
512 tests / 34 build steps plus examples in both Debug and ReleaseSafe.
On aarch64 Linux, 18 distinct execution/refusal cases per host mode verify
same-engine recovery, including diagnostic overflow and a successful child exit
with unread reply bytes. One HTTP fixture request per mode also generated typed
Zig, compiled it, called the native typed tool and returned `42`; fixture serving
and non-daemon request threads were closed and joined. Guests are always compiled
ReleaseSafe, independently of the host mode.

Separate real kernel controls exercised aggregate OOM group termination,
1 MiB tmpfs exhaustion, an eight-process cap, three/eight live pidfds becoming
dead after PID1 join, direct-kill fallback after cgroup control failure and
non-page-aligned memory limits. An immutable compiler wrapper deliberately left
an acknowledged live descendant; runtime tool dispatch still saw only PID1 and
the runtime. Planted removal of compiler reaping failed the process invariant,
and planted acceptance of an unclean reset failed with `UnexpectedPublication`.
x86_64 Linux received semantic-only checking, not runtime qualification; Darwin
was actually run and refused with `UnsupportedPlatform`. These results are
finite native proof, not full DSPy parity or authenticated inference.

## Native parallel composition

`Parallel(Result, limits)` runs native programs or callbacks on a persistent worker
pool. Initialize it **in place and do not move it**. The compile-time limits specify
`workers`, `tasks`, total `attempts` (including speculative duplicates), and retained
`failures`. Threads are created only by `init(io, spawn_config)` and joined by
`deinit()`. Queues, scheduling metadata and typed output slots are inline; dispatch
does not use an allocator. A pool permits one forward at a time; use distinct
startup-created pools for nested or concurrent batches.

`Pool.Program(P)` binds a borrowed `*const P`, `arguments` and per-attempt `contexts`.
Its `task()` is a native task descriptor:

- `.arguments = .{ .input = typed_input }` passes `P.Input` directly. Named structs,
  tuple structs and a whole borrowed `*const Example` are ordinary native inputs.
- `.arguments = .{ .example = &example }` decodes a struct input from
  `Example.input_keys`, using the same declaration-order role checks and codecs as
  evaluation, without materializing an input or output dictionary.
- Custom `Pool.Task` callbacks receive attempt/task/worker indices, a cancellation
  token and stable `*Result` storage. `Result` may be optional or a tagged union;
  a successful null is distinct from a failed slot.

Provide at least `limits.attempts` contexts. **Copying Context does not isolate its
pointees.** Scratch, mutable LM state, traces and mutable retrievers must be private
to each simultaneously running attempt or explicitly synchronized. Predictor-bound
LMs are shared through program value copies and override `Context.lm`: supplying
a private fallback per attempt does not isolate a bound model. Use an explicit
per-attempt `lm_override`, private program bindings, or a synchronized model.
Do not mutate shared program configuration while calls run. In particular, do not freeze or
globally roll back live InferRules instruction identities to make concurrency
appear safe. Scratch-backed children must survive until the returned results are
discarded; resetting a worker arena between tasks would invalidate earlier outputs.

`forward(tasks, options)` returns input-ordered `.results`: each slot is `.failed`
or `.value = *const Result`. Output slots are valid until the next forward or
deinitialization, and borrowed children retain their own lifetime requirements.
Failures below `max_errors` leave a slot available for a later successful duplicate.
The first accepted success wins; raised errors from losing attempts still count
toward the budget while the run is active. Zero errors allowed means the first
actual error cancels execution. With `return_failed_tasks = true`, `.failures` is
a borrowed handle whose `items()` observes accumulated failed descriptors across
calls. Repeated failures of an unresolved task retain the last accepted error for
each occurrence, as in DSPy's failed-index/exception-map pairing. Its frame pointers
must stay alive until `clearFailures()` or deinitialization. A full failure buffer
returns `FailureCapacity` without partially appending that call's failures.

`straggler_timeout_ns` is a speculative retry threshold, **not a task deadline**;
zero disables retries. As in DSPy, a duplicate future can itself be retried while
few attempts remain pending. Unlike the source's unbounded chain, exhausting the
startup attempt capacity returns `AttemptCapacity`. Terminal success/error/cancel
decisions are fixed before joining late callbacks; losing callbacks cannot rewrite
accepted output slots or retroactively change a terminal decision.

Every return, including cancellation and capacity/error-budget failures, waits for
all running callbacks. `cancel()` can be called from another thread; custom
callbacks must check their cancellation token and bound their work/I/O.
`Program(P)` checks cancellation before entering `P.forward`; the program itself
must finish or cooperate through its own inputs/I/O policy. This is trusted native
execution, not a sandbox for generated code, and a non-cooperating callback delays
return. DSPy's executor can return while abandoned originals still run; that
lifetime hazard is deliberately not copied.

Run the actual native program example:

```sh
devenv shell -- zig build parallel
```

It compares every one of 8,192 parallel typed arithmetic outputs to serial native
execution, using zero-byte per-attempt scratch and no LM calls, and reports both
elapsed times. This is a CPU dispatch example, not authenticated inference or a
whole-runtime allocation/performance proof. The ordinary test suite also exercises
actual concurrent typed Predict with separate scripted LMs and scratch, joined
stragglers/cancellation, and seeded scheduling properties.

With speculative retries disabled (`straggler_timeout_ns = 0`), workers avoid
per-task clock reads and notify the owner only on a terminal decision or a late
completion needed by the join barrier. Timed runs retain start/completion
notifications so retry deadlines are recalculated.

A local ReleaseSafe A/B of `examples/parallel.zig` used 20 interleaved runs per
binary, each checking 8,192 results against serial execution. Median pool time
fell from 20.54 ms to 15.19 ms; serial medians were 43.63 ms and 43.49 ms.
Both versions retained the same checksum, zero scratch bytes and zero LM calls.
This is a short CPU-workload observation, not a general throughput or tail-latency
guarantee: the candidate's slowest sample was 54.84 ms versus 24.70 ms for the
baseline, on a non-isolated workstation.

`parallel.Evaluator(P, limits)` is the native counterpart of
`dspy.Evaluate(num_threads=...)`. After initialization, set `failure_score`
(default zero) to score recoverable program or metric failures; error-budget
cancellation and out-of-memory still abort. Configure it only between joined
evaluations. `init(io, spawn_config, backing)` creates the
workers, one prediction record and typed output slot per attempt, and a
fixed-buffer scratch allocator over each borrowed `backing[attempt]`.
`evaluate(contexts, &program, devset, metric, options)` runs program plus metric per
example with the same `Options` as `Parallel` (`return_failed_tasks` must be false).
Each attempt's context supplies its own LM/retriever pointees; its arena is
replaced by the reset attempt scratch and its trace by null. The program and metric
must be thread-safe. Scores are summed after the join in devset order, not
completion order, with failed slots scoring `failure_score`, as in DSPy.
At the default zero failure score the sum also matches serial `optimize.evaluate`.
`errors` counts failed examples, not losing
duplicate attempts. As with `Parallel`, reaching `max_errors` returns
`error.ExecutionCanceled` (DSPy's executor raises "Execution cancelled"), while
serial `optimize.evaluate` returns the triggering error. Out-of-memory, including
exhausted attempt scratch, cancels the run and returns `error.OutOfMemory`, as
serially.

### Trace bootstrapping

`bootstrapTraceData(P, out, ctx, &program, dataset, options)` is the native
counterpart of DSPy 3.5's `bootstrap_trace_data`. It runs the original program
without patching or mutating it. Rows retain their dataset index, borrowed
example, successful prediction or `FailedPrediction`, ordered trace and nullable
score. The score is absent (`null`) when no metric was supplied.

```zig
const rows = try dszig.bootstrapTraceData(
    Program, out, ctx, &program, dataset,
    .{ .metric = metric, .failure_score = 0.5 },
);
```

Adapter failures retain the actual completion and partial result from the
production parser, including JSON fallback's **final** failed response.
Missing-field payloads include all present fields and defaults, even after an
earlier missing field. Chat/XML invalid values carry no partial result; JSON
retains converted predecessors and raw invalid/following values. Empty
completions fail before output defaults can turn them into a success.
TwoStep extraction failures retain the original main response and signature,
not the extraction response. Invalid native decision boxes retain the source's
ASCII-escaped JSON answer text.

Failed-format rewards are
`format_failure_score + (failure_score - format_failure_score) * present / expected`,
or `format_failure_score` with no present fields. Failed predictions bypass the
metric. Successful predictions call it with a null trace, as under DSPy's
Evaluate. The appended failed trace selects the **first predictor with the same
signature class** and records top-level program inputs, not that predictor's
inputs. A newly derived evidence signature with no matching program predictor
follows the source's unhandled-failure policy rather than inventing a match.

`capture_crashes` captures non-LM program errors, retaining preceding successful
trace steps; its completion text is the native error name, not a fabricated
Python exception class/message. `LMFailure` and `ScriptExhausted` are never
captured as crashes. Unhandled program/metric rows are skipped with
`raise_on_error = false`; otherwise all evaluation work finishes before
`error.UnpackPredictionAndTrace` is returned. `unhandled` can retain the first
cause, stage and dataset index. Out-of-memory and cancellation abort instead.

For DSPy's `num_threads`, supply `rollouts` with a startup-created
`Parallel(bootstrap_trace.Run, limits).runner()` and exclusive per-attempt
contexts. Results are collected in dataset order only after every callback
joins. Each context supplies its own scratch arena, LM and retriever; the
program and metric must be thread-safe. `out` is used only by the calling
thread, never concurrently. It owns retained predictions, completion text and
successful trace children; rows and failed-step input values borrow the dataset,
and predictor pointers borrow the program. Neither may expire while rows live.
An optional `scratch_reset` callback reclaims attempt storage after joined
results have been retained; do not reset or alias that storage with `out`.

Native `callbacks` expose evaluation start/end events with a correlated
process-local `u64` call ID and `callback_metadata`. Handler errors are logged
and ignored. End receives the aggregate evaluation, including configured
failure scores for unhandled rows; a later unpack failure does not retroactively
turn a successful evaluation event into an error. An empty dataset, memory
failure or cancellation reaches end as an error. `log_format_failures` enables
format-failure warnings. There is no no-op `capture_failed_parses` option:
upstream leaves that argument unused and always captures actual parse errors.

Low-level callers can opt into `Context.parse_failure` or
`Adapter.parseWithFailure` without using trace bootstrapping. Captures borrow
the completion and parse arena; use one per concurrent context. If a program
handles a predictor error and later raises its own `error.AdapterParse` without
another predictor call, explicitly `reset()` its capture: Zig error values do
not preserve Python exception-instance identity.

An independent actual DSPy 3.5 comparison covered 15 direct parser cases,
24 trace-bootstrap cases and 17 policy cases, including serial/parallel ordered
results, duplicate signature classes, fallback, metric failure and crash policy.
A separate comparison covered five evaluation callback lifecycles, TwoStep
wrapping and two native decision-box failures. Four configured evidence-signature
lookup-failure policies also matched. These finite checks do not supply
fine-tuning, BetterTogether or whole-library static-memory parity.

### Fine-tuning training records

`Adapter.formatFinetuneData(arena, io, view, demos, &inputs, &outputs)` formats
one successful call as OpenAI-shaped `{"messages": [...]}` training data.
Chat and XML reuse their existing system, demo, history and request rendering,
then append the assistant answer. Missing declared answer fields become Python
`None` in Chat and self-closing tags in XML, including incomplete ordinary
history events. Explicitly filtered native-tool history retains its filtered
field set. Native-tool history keeps assistant `tool_calls`, paired
`tool_call_id` values, error-result text and null content for calls without
assistant text. Formatting does not validate the answer as a prediction or
submit a training job.

This mirrors the actual DSPy adapter boundary: JSON reports
`error.FinetuneNotImplemented`, and TwoStep reports
`error.MissingFinetuneFormatter` because its source class has no such method.
Both refuse before formatting inputs or accessing media. These are source
limitations, not fallbacks to another adapter.

Training/display serialization is deliberately separate from canonical lm15
serialization and the live OpenAI Chat HTTP policy. Native documents become
OpenAI `file` blocks, with `file_id` or inline `file_data`; generated filenames
use the pinned CPython MIME-extension ordering, not the caller's filename.
Raw canonical local-path documents are read through the explicit `std.Io`,
and file errors propagate; URL documents report `error.UnsupportedFeature`.
`Content.toOpenAIChatValue(arena, io)` exposes this training/display projection.
It does **not** enable document requests on the live Chat transport.

Containers, encoded data and generated names use the supplied arena. Other
string leaves borrow input, demo or history storage; clone before retaining
the record beyond those lifetimes. A fixed-capacity arena bounds formatter
memory, not all allocations inside the caller's I/O implementation.

An independent DSPy 3.5 comparison matched 60 cases: 26 complete training
records and 34 source refusals or file errors. It includes missing/null answers,
incomplete demos/history, Code, Reasoning, fixed tuples, optional fields and
in-memory/local-file documents. A separate comparison matched 1,086 actual
source document blocks across MIME types. Sixteen additional complete tool
dialogues matched native/text policies, result errors, null assistant text and
mismatched call IDs. These are finite formatter proofs,
not provider training, `BootstrapFinetune`, full DSPy parity or whole-library
static-memory proof.

### Concurrent optimizer evaluation

Optimizers take DSPy's `num_threads` as `concurrent_evaluation`:
`evaluator.concurrent(&contexts, options)` binds an initialized evaluator to its
nominal program type, per-attempt contexts and options. A different program type
is rejected before invocation, even when generic type display names collide.
`BootstrapFewShotWithRandomSearch`,
`BootstrapFewShotWithOptuna`, `MIPROv2`, `COPRO` and `InferRules` then evaluate
candidates through it, each call using the optimizer's own `max_errors`; a
program-type mismatch returns `error.ProgramTypeMismatch`, and null keeps serial
`optimize.evaluate`. Candidate evaluation is sequential between candidates, as in
DSPy; only examples within one evaluation run concurrently. `BootstrapFewShot`
has no DSPy thread option and stays serial.

SIMBA's `num_threads` is `rollouts`: a `Parallel(SIMBA.Run, limits).runner()` and
per-attempt contexts supplying LM and retriever pointees. Trajectory sampling,
candidate scoring and final trainset scoring then run on the pool in submission
order, with every random draw made beforehand, as DSPy does. Because SIMBA
retains predictions and traces, they are allocated from `out`, which must then be
thread-safe (an `ArenaAllocator` over a thread-safe child is); a configured
teacher LM is shared and must be thread-safe. GEPA's `rollouts` field
(`Parallel(GEPA.Run, limits)`) runs each minibatch and valset evaluation the
same way, under the same `out` requirement. Golden tests check that concurrent
SIMBA and GEPA (including merge) issue DSPy's exact multiset of requests and
select DSPy's programs and scores.

## Native seeded evaluation campaigns

The native simulation harness executes real `dszig.optimize.evaluate` calls with
a typed arithmetic program and metric, without Python, a network or an API key.
Generated cases vary row values, scores, program/metric failures and error
budgets. An independent oracle checks the outcome, program-call count,
ties-to-even percentage rounding, unchanged dataset/input roles, scratch
reclamation and absence of unexpected LM calls.

Run these commands from the repository root with the pinned development tools:

```sh
# Four fixed seeds, 512 cases each, plus rounding/error boundaries,
# wrong-answer detection and shrinking regression tests:
devenv shell -- zig build simulation-test

# A reproducible generated campaign (default: 10,000 cases):
devenv shell -- zig build simulate -- --seed 0x1234 --cases 512

# Enter the shell BEFORE redirecting: its startup banner is not replay JSON.
devenv shell

# Deliberately corrupt actual program answers to prove detection and shrinking.
# EXPECTED NONZERO EXIT: stdout contains only the concrete failing Replay JSON;
# stderr contains the negative-control label, violation and seed/index details.
zig build simulate -- --wrong-answer --seed 0x1234 --cases 512 > replay.json

# Reexecute that concrete counterexample, in the same negative-control mode.
# EXPECTED NONZERO EXIT again, reporting the score violation:
zig build simulate -- --replay replay.json --wrong-answer
```

Normal campaigns report seed, replay version and case count when no violation
is found. `--seed` accepts a `u64` with base prefixes; `--cases` is bounded to
1–1,000,000. On the first violation, the driver shrinks the case while retaining
the same violation, with at most 4,096 shrink attempts, and emits a versioned
concrete replay rather than seed-only evidence. Preserve that JSON for failures
in normal mode too, and replay it without `--wrong-answer`. Replay inputs are
limited to 16 KiB and must have a supported version and valid scenario.
`--replay` cannot be combined with explicit `--seed` or `--cases`.

This is a **seeded sequential evaluation campaign**, not full optimizer
coverage, concurrent deterministic simulation testing, a static-allocation
proof or additional DSPy parity evidence. The harness initializes fixed
execution backing and reuses example storage, but the current evaluation core
still makes allocator calls against that backing. The wrong-answer switch is
an explicit test-only negative control, not a production-runtime modification.

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

The committed `dspy_3.5.0.json` and `dszig_3.5.0.json` recordings use the public
OpenAI endpoint with `gpt-4o-mini`: 679 DSPy calls and 680 native calls.
`devenv shell -- dz-live` completed both authenticated recordings, both
cross-runtime replays and the pinned reference gate. The nine pipeline scores
and serialized artifacts matched on replay; GEPA improved the date task from
0% to 100%, while COPRO and SIMBA remained at 0% in this run. This is finite
live-model coverage, not proof of every optimizer option or full DSPy parity.
The recorder preserves lm15's already-flat JSON-schema metadata and also
normalizes nested Chat-wire schema dictionaries.

## Real inference

The live executable reads `OPENAI_API_KEY`, or `CODEX_LB_API_KEY` for the local
homelab configuration. Selecting `CODEX_LB_API_KEY` requires an explicit
`OPENAI_BASE_URL`; the homelab key is never sent to the public default endpoint.
It never reads or mutates personal agent settings and never embeds credentials
in transcripts.

Credentials are provisioned through SecretSpec, profile `live`, stored in the OS
keyring (`tools/live/secretspec.toml`; it is not at the repository root because
devenv auto-loads a root manifest into every shell):

```sh
secretspec --file tools/live/secretspec.toml set OPENAI_API_KEY --profile live --reason "dszig live inference"
# Optional: OPENAI_BASE_URL, DSZIG_LIVE_MODEL, DSZIG_LIVE_API (chat_completions|responses)
devenv shell -- dz-live-smoke   # one real native CoT request
devenv shell -- dz-live         # DSPy and DSZig recordings, cross-replayed
```

The DSPy recorder (`tools/dspy_reference/live.py`) reads the same variables,
passing `OPENAI_BASE_URL` as `api_base` and mapping `chat_completions` to DSPy's
`chat` model type.

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
  The separate Linux generated-Zig executor above compiles native programs and
  capability RPC; it does not claim arbitrary Python execution or full RLM/Flex
  behavior.
- **Other modules and optimizers:** bounded native Avatar/AvatarOptimizer follow
  Source methods, with broken Source constructors and the explicit native
  capacity, class-identity, callback-ownership and serial-execution boundaries
  described above. Native BetterTogether
  composes implemented optimizers, with the typed argument, teacher, ownership
  and provider limits described above. Trace bootstrapping, Chat/XML fine-tuning
  data formatting, bounded OpenAI training jobs and BootstrapFinetune are implemented
  with the provider, capacity and ownership limits described above.
  Native Parallel uses explicit typed
  tasks and contexts, not Python callback objects or ambient thread-local settings.
  Its startup capacities and cooperative join contract are described above.
- **Optimizer execution and shared identity:** InferRules' signature-alias propagation does not update arbitrary external
  instances outside the reachable program graph. Canonical Zig signature/Code
  types and default CoT construction do not reproduce every fresh dynamic Python
  class identity; Refine shares this boundary. Mid-bootstrap mutation of a
  borrowed teacher-settings object is not reflected after Bootstrap's entry
  snapshot.
- **Runtime model identity:** Code, File, Reasoning, BoundingBox, Citations,
  Noul/Score/Choice and generated decision evidence have genuine boxed classes,
  not a complete Python object runtime. Other model
  families still use their existing representations. Generic arrays/maps do not
  preserve shared container/class aliases through memoized deepcopy. Complete
  Python builtin identity/dispatch, NaN float-object identity across distinct or
  cloned boxes, Reasoning indexing/string-method delegation, arbitrary attribute
  mutation and Python frozen-assignment exceptions are not reproduced. The
  citation proof retains 150 unsupported Decimal/bytearray/dynamic-model rows;
  its finite passing observations do not establish complete Citations parity.
- **Providers and media:** native File construction, adapter/typed prediction
  handling, File tool-result markers and bounded Responses document translation
  are implemented as described above. There is still no complete lm15 provider
  catalog/routing/auth/error taxonomy, full provider SDK, embedding providers
  beyond OpenAI/Azure, Image/Audio, general rich-media tool results, non-OpenAI
  training providers or native streaming. Document support is not general
  multimodal support, and Chat Completions does not accept document parts.
- **File and DocumentPart boundaries:** the constructor smoke explicitly leaves
  606 successful source Python DocumentPart object/repr/bytes operations
  unimplemented; canonical document dictionaries do not replace those
  operations. The bytearray slice now exercises valid/invalid UTF-8 field
  coercion and immutable snapshots (also accepting immutable bytes fields).
  The older constructor receipt still excludes bytearray/memoryview/buffer
  and tuple dispatch, nonstring helper parameters,
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
  Pydantic model defaults are not native signature types. `Literal` supports
  null/bool/int/float/string, genuine Enum-member and bytes arguments, not
  dynamic annotation construction. Non-UTF-8 bytes validate, but schema
  generation fails at compile time rather than Python's deferred
  `UnicodeDecodeError`. Enum factories are
  comptime class definitions, not arbitrary runtime class creation or Enum
  mixins. Python literal lexical nesting is bounded at 200; its separate native
  AST-depth ceiling is 3,000, not Python's caller-stack-dependent recursion
  budget. Generic runtime values still cannot preserve shared NaN float-object
  identity or arbitrary mutable alias/cycle graphs. A builtin observation smoke
  matched 405 checks but differed on two same-NaN container-equality cases;
  those failures are not counted as parity. Native ownership clones preserve a
  set's stored iteration snapshot, not CPython deepcopy's reconstruction order.
  JSON repair covers the
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
  Concurrent evaluation and SIMBA/GEPA rollouts use explicit startup-created
  native pools; Python ambient settings and general async execution are not
  implemented. SIMBA uses Zig `@exp`, not NumPy's implementation.
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
  format. Native OpenAI configuration and bound predictors support program
  save/load with injected runtime resources; Python class reconstruction, all
  native engine codecs, provider training state and complete DSPy LM save/load
  remain unsupported.
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
