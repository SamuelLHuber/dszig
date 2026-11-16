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
| Class-style signatures and typed fields | `Signature`, `Dict`, `Code`, `CodeWithLanguage`, `BoundingBox`, `File`, `Citations` and `Citations.Citation`, scalar/string/list/optional/model fields, string literal enums and mixed scalar/Enum/bytes `Literal`, field descriptions/defaults, compile-time rejection of invalid signatures |
| Runtime values and documents | `Value`, `Array`, `ObjectMap`, boxed native Code/File/Reasoning/BoundingBox/Citations/decision/evidence identity, recursive typed-class serialization, text/canonical-document content union and bounded Responses `input_file` translation |
| Chat, JSON, XML and two-stage adapters | `Adapter.chat`, `.json`, `.xml`, `.two_step`; `TwoStepAdapter` with a separate extraction LM; independent prompt/parse fixtures, strict response-schema construction, JSON fallback, typed XML and Unicode key handling |
| Prediction and reasoning | `Predict`, `ChainOfThought`, `BestOfN`, `Refine`, `MultiChainComparison`, explicit `Context`, typed outputs, serialized `Trace`, named predictor/parameter discovery |
| Parallel composition | Startup-created `Parallel(Result, limits)` workers, typed program/Example/tuple bindings, ordered result slots, speculative straggler retries, bounded accumulated failure reports and joined cooperative cancellation |
| Typed decisions | `Noul`, configured `Score`/`Choice`, generative evidence classes, criteria/state snapshots, native Annotated metadata, owning evidence collectors and TypeSafe SystemOne prediction; finite coverage and unsupported rows below |
| Retrieval and embeddings | `Retrieve`, lexical `InMemoryRetriever`, HTTP `ColBERTv2` with shared mutable-graph memory caching, custom/OpenAI/Azure `Embedder`, float32 `KNN`, per-query `KNNFewShot`, and brute-force `Embeddings`/`EmbeddingsWithScores` with NumPy persistence |
| Evaluation and few-shot search | `optimize.evaluate`, concurrent `parallel.Evaluator`, `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `BootstrapFewShotWithOptuna`, teacher context, zero thresholds and positional multi-use trace selection |
| Instruction optimization | `InferRules`; `MIPROv2`, its program-aware proposer and an Optuna 4.5.0 TPE port; `COPRO`; `SIMBA`; supported instance-frontier `GEPA` paths against gepa 0.1.4 |
| Agents and tools | `ReAct`, comptime `Tool`, `History`, `ToolCalls`, tool results, native declarations and text tool calls, typed submit, retained history |
| Voting | `Ensemble`, `ensemble.majority`, Unicode text normalization; exact Python numeric equality, NaN object identity and source-order fallback behavior |
| Answer metrics | `metrics.emScore`, `f1Score`, `hotpotF1Score`, `precisionScore`, multi-reference `em`/`f1`/`hotpotF1`, thresholded and record-based answer matching, DPR-token passage matching |
| LM boundary | `LM`, native `lm.OpenAI` Chat Completions/Responses and `lm.TypeSafe` SystemOne, shared bounded `http.Transport`, scripted/keyword LMs, recording/replay wrappers and typed native protocol metadata |
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
4,300-digit string limit. Conversion precedes singleton alias resolution.

The following independent frozen-source observations matched in both native
Debug and ReleaseSafe, with byte-identical output between modes:

| Native consumer | Compared / Source rows | Outcome/value/identity/signal observations | Normalized exception identities unobserved |
| --- | ---: | ---: | ---: |
| IntEnum runtime operations | 2,294 / 2,324 | 24,577 | 657 |
| IntEnum scalar validation | 35 / 35 | 379 | 6 |
| Builtin IntEnum declarations | 19 / 19 | 134 | 0 |
| Configured Decimal policies | 368 / 368 | 3,950 | 233 |

The runtime consumer deliberately exits nonzero for incomplete coverage:
sequence repetition is not implemented by the numeric helpers. Unsafe unbounded
repetitions were separately left unattempted by the Source probe. The declaration
probe includes fractional and maximum finite floats, signed zero, subnormals,
Unicode/separated strings, exact wide integers and the 4,300-digit boundary.
Seventeen independently compiled Debug programs also refused the source-observed
malformed strings, over-limit strings and nonfinite floats; compile-time
refusals do not establish Python exception identity. Declaration-time bytes,
Decimal, tuple/base arguments, arbitrary Python `int(value)` protocols, custom
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

## Benchmarks against DSPy

`dz-bench` (in `devenv shell`) builds ReleaseFast and has both runtimes replay
the same full-suite transcript: eight adapter cases and nine optimizer
pipelines, 687 LM calls. Replay removes network and model time, so these are
framework-overhead numbers: prompt formatting, parsing, optimizer logic,
evaluation, serialization and the replay lookup. Each run is verified first
(every request matched, every call consumed, identical outputs, scores and
artifacts), so accuracy is equal by construction on this transcript.
Segments end where DSPy starts the next pipeline's first LM call.
DSPy's own progress printing and logging stay enabled (stdout to /dev/null,
logging disabled). Peak heap is `tracemalloc` for DSPy and a counting
allocator around every DSZig allocation. Measured 2026-11-15 on an 8-vCPU
Intel Xeon Linux VM:

Transcript: `tests/fixtures/dspy_3_5_0_mock_suite.json`, 687 LM calls; DSZig fast x20, dspy 3.5.0 (Python 3.11.17) x5. Medians; no network.

| Segment | DSPy ms | DSZig ms | Speedup |
|---|---:|---:|---:|
| cases | 22.9 | 0.16 | 141x |
| rag | 52.5 | 0.24 | 222x |
| dates | 115.6 | 0.43 | 266x |
| dates_search | 346.0 | 1.37 | 252x |
| dates_mipro | 396.2 | 1.56 | 254x |
| dates_mipro_aware | 211.1 | 0.85 | 249x |
| rag_react | 161.2 | 1.48 | 109x |
| dates_copro | 367.5 | 1.14 | 324x |
| dates_simba | 282.1 | 2.10 | 134x |
| dates_gepa | 264.2 | 1.64 | 161x |
| total | 2282.9 | 10.97 | 208x |

| Metric | DSPy | DSZig |
|---|---:|---:|
| Throughput (LM calls/s of framework work) | 301 | 62,639 |
| Per-call overhead p50 (us) | 2,925.3 | 11.5 |
| Per-call overhead p99 (us) | 8,045.9 | 88.0 |
| Startup: `import dspy` vs none (ms) | 365 | 0 |
| Peak heap, one run (MiB) | 6.5 (tracemalloc) | 2.8 (counting allocator) |
| Peak RSS, process (MiB) | 123 | 18 |

Real-model accuracy and latency come from the live transcripts (see Real
inference); there the model dominates latency in both runtimes.

## Feature map

Every public `dspy.*` name in DSPy 3.5.0, from
`tools/dspy_reference/features.json`. `native` means implemented and compared
against DSPy-generated fixtures or oracles; `partial` means implemented with
the limits in Compatibility limits; `missing` means not implemented.
`dz-upstream` (also run by `tools/dspy_reference/check.sh`) fails when DSPy
exports a name without a row or changes a source file a row was reviewed
against (pinned by SHA-256), so a DSPy release shows exactly what to port.
Regenerate this table with
`uv run --locked upstream_report.py --markdown dszig.json dspy.json`.

| DSPy 3.5.0 | Status | DSZig source | Equality checks | Replay benchmark (DSPy → DSZig) |
|---|---|---|---|---|
| `dspy.Adapter` (`adapters/base.py`) | partial | `adapter.zig` | format_cases.json, xml.json, dspy_3_5_0_mock_suite.json | cases: 23 ms → 0.16 ms |
| `dspy.AdapterParseError` (`utils/exceptions.py`) | partial | `adapter.zig` | error.AdapterParse outcome compared in suite cases | – |
| `dspy.Audio` (`adapters/types/audio.py`) | missing | – | – | – |
| `dspy.AvatarOptimizer` (`teleprompt/avatar_optimizer.py`) | missing | – | – | – |
| `dspy.BaseLM` (`clients/base_lm.py`) | partial | `lm.zig` | lm_response_oracle.py, request_schema_oracle.py | – |
| `dspy.BaseModule` (`primitives/base_module.py`) | partial | `predict.zig` | named predictor discovery tests | – |
| `dspy.BestOfN` (`predict/best_of_n.py`) | native | `best_of_n.zig` | behavior tests vs DSPy source | – |
| `dspy.BetterTogether` (`teleprompt/bettertogether.py`) | missing | – | – | – |
| `dspy.BootstrapFewShot` (`teleprompt/bootstrap.py`) | native | `optimize.zig` | bootstrap_rag.json, bootstrap_multihop.json, trace_selection.json, live replay | rag: 52 ms → 0.24 ms; dates: 116 ms → 0.43 ms; rag_react: 161 ms → 1.48 ms |
| `dspy.BootstrapFewShotWithOptuna` (`teleprompt/teleprompt_optuna.py`) | partial | `bootstrap_optuna.zig`, `tpe.zig` | bootstrap_optuna_oracle.py, optuna_tpe.json | – |
| `dspy.BootstrapFewShotWithRandomSearch` (`teleprompt/random_search.py`) | native | `optimize.zig` | suite replay candidates | dates_search: 346 ms → 1.37 ms |
| `dspy.BootstrapFinetune` (`teleprompt/bootstrap_finetune.py`) | missing | – | – | – |
| `dspy.BootstrapRS` (`teleprompt/random_search.py`) | native | `optimize.zig` | alias of BootstrapFewShotWithRandomSearch | dates_search: 346 ms → 1.37 ms |
| `dspy.BoundingBox` (`adapters/types/bounding_box.py`) | native | `bounding_box.zig` | bounding_box_oracle.py, bounding_box_identity_oracle.py | – |
| `dspy.COPRO` (`teleprompt/copro_optimizer.py`) | native | `copro.zig` | suite replay candidates and scores | dates_copro: 368 ms → 1.14 ms |
| `dspy.ChainOfThought` (`predict/chain_of_thought.py`) | native | `predict.zig` | format_cases.json, xml.json, dspy_3_5_0_mock_suite.json | cases: 23 ms → 0.16 ms |
| `dspy.ChatAdapter` (`adapters/chat_adapter.py`) | native | `adapter.zig` | format_cases.json, xml.json, dspy_3_5_0_mock_suite.json | cases: 23 ms → 0.16 ms |
| `dspy.Code` (`adapters/types/code.py`) | native | `code.zig` | code_oracle.py, code_identity_oracle.py | – |
| `dspy.CodeExecutionError` (`primitives/code_interpreter.py`) | missing | – | – | – |
| `dspy.CodeInterpreter` (`primitives/code_interpreter.py`) | missing | – | – | – |
| `dspy.CodeInterpreterError` (`primitives/code_interpreter.py`) | missing | – | – | – |
| `dspy.ColBERTv2` (`dsp/colbertv2.py`) | native | `colbertv2.zig` | colbertv2_oracle.py (81 HTTP requests) | – |
| `dspy.Completions` (`primitives/prediction.py`) | partial | `predict.zig` | suite replay | – |
| `dspy.ContextWindowExceededError` (`utils/exceptions.py`) | partial | `lm.zig` | – | – |
| `dspy.DSPyError` (`utils/exceptions.py`) | partial | `lm.zig` | – | – |
| `dspy.Embedder` (`clients/embedding.py`) | native | `embedder.zig` | embeddings_oracle.py | – |
| `dspy.Embeddings` (`retrievers/embeddings.py`) | native | `embeddings.zig` | embeddings_oracle.py | – |
| `dspy.EmbeddingsWithScores` (`retrievers/embeddings.py`) | native | `embeddings.zig` | embeddings_oracle.py | – |
| `dspy.Ensemble` (`teleprompt/ensemble.py`) | native | `ensemble.zig` | ensemble.json | – |
| `dspy.Evaluate` (`evaluate/evaluate.py`) | partial | `optimize.zig`, `parallel.zig` | bootstrap_rag_eval scores, suite replay (serial); `parallel.Evaluator` devset-order sums vs serial in tests/parallel.zig | rag: 52 ms → 0.24 ms; dates: 116 ms → 0.43 ms |
| `dspy.Example` (`primitives/example.py`) | partial | `predict.zig` | suite replay | – |
| `dspy.File` (`adapters/types/file.py`) | native | `file.zig` | file_oracle.py | – |
| `dspy.FinalOutput` (`primitives/code_interpreter.py`) | missing | – | – | – |
| `dspy.Flex` (`predict/flex/flex.py`) | missing | – | – | – |
| `dspy.GEPA` (`teleprompt/gepa/gepa.py`) | partial | `gepa.zig` | gepa.json, gepa_merge.json, suite replay | dates_gepa: 264 ms → 1.64 ms |
| `dspy.History` (`adapters/types/history.py`) | native | `history.zig` | react.json | – |
| `dspy.Image` (`adapters/types/image.py`) | missing | – | – | – |
| `dspy.InferRules` (`teleprompt/infer_rules.py`) | native | `infer_rules.zig` | infer_rules_oracle.py | – |
| `dspy.InputField` (`signatures/field.py`) | partial | `signature.zig` | compile-fail cases, format_cases.json | – |
| `dspy.InterpreterPool` (`primitives/interpreter_pool.py`) | missing | – | – | – |
| `dspy.JSONAdapter` (`adapters/json_adapter.py`) | native | `adapter.zig`, `json_repair.zig` | format_cases.json, xml.json, dspy_3_5_0_mock_suite.json, adapter_repair_oracle.py | cases: 23 ms → 0.16 ms |
| `dspy.KNN` (`predict/knn.py`) | native | `knn.zig` | knn_oracle.py | – |
| `dspy.KNNFewShot` (`teleprompt/knn_fewshot.py`) | native | `knn_fewshot.zig` | knn_oracle.py | – |
| `dspy.LM` (`clients/lm.py`) | partial | `lm.zig`, `http.zig` | lm_response_oracle.py, live transcripts (Chat Completions/Responses only) | – |
| `dspy.LMAuthError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMBillingError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMCollectionLimitError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMConfigurationError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMInvalidRequestError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMLockTimeoutError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMNotConfiguredError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMProviderError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMRateLimitError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMServerError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMStreamAssemblyError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMTimeoutError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMTransportError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMUnexpectedError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMUnsupportedFeatureError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LMUnsupportedModelError` (`utils/exceptions.py`) | partial | `lm.zig` | Zig error set, not Python exception classes | – |
| `dspy.LabeledFewShot` (`teleprompt/vanilla.py`) | native | `optimize.zig` | golden tests | – |
| `dspy.LocalInterpreter` (`primitives/local_interpreter.py`) | missing | – | – | – |
| `dspy.MIPROv2` (`teleprompt/mipro_optimizer_v2.py`) | native | `mipro.zig`, `tpe.zig` | suite replay instructions and trial logs, optuna_tpe.json | dates_mipro: 396 ms → 1.56 ms; dates_mipro_aware: 211 ms → 0.85 ms |
| `dspy.Module` (`primitives/module.py`) | partial | `predict.zig` | comptime program structs | – |
| `dspy.MultiChainComparison` (`predict/multi_chain_comparison.py`) | native | `multi_chain_comparison.zig` | behavior tests vs DSPy source | – |
| `dspy.OldField` (`signatures/field.py`) | missing | – | – | – |
| `dspy.OldInputField` (`signatures/field.py`) | missing | – | – | – |
| `dspy.OldOutputField` (`signatures/field.py`) | missing | – | – | – |
| `dspy.OutputField` (`signatures/field.py`) | partial | `signature.zig` | compile-fail cases, format_cases.json | – |
| `dspy.Parallel` (`predict/parallel.py`) | partial | `parallel.zig`, `parallel_state.zig` | parallel.json (DSPy ParallelExecutor order, failures, max_errors cancel) on the native pool; tests/parallel.zig (typed program/Example bindings, straggler duplicates, cancellation joins, seeded scheduling); no progress bar, SIGINT or thread-local settings; attempts bounded (`AttemptCapacity`) | – |
| `dspy.Predict` (`predict/predict.py`) | native | `predict.zig` | format_cases.json, xml.json, dspy_3_5_0_mock_suite.json | cases: 23 ms → 0.16 ms |
| `dspy.Prediction` (`primitives/prediction.py`) | partial | `predict.zig` | suite replay | – |
| `dspy.Provider` (`clients/provider.py`) | missing | – | – | – |
| `dspy.PythonInterpreter` (`primitives/python_interpreter.py`) | missing | – | – | – |
| `dspy.RLM` (`predict/rlm.py`) | missing | – | – | – |
| `dspy.ReAct` (`predict/react.py`) | native | `react.zig`, `tool.zig` | react.json, refine_react_oracle.py, suite replay | rag_react: 161 ms → 1.48 ms |
| `dspy.ReActV2` (`predict/react_v2.py`) | missing | – | – | – |
| `dspy.Reasoning` (`adapters/types/reasoning.py`) | native | `reasoning.zig` | reasoning_identity_oracle.py | – |
| `dspy.Refine` (`predict/refine.py`) | native | `refine.zig` | refine_oracle.py | – |
| `dspy.Retrieve` (`retrievers/retrieve.py`) | native | `predict.zig` | bootstrap_rag.json, suite replay | rag: 52 ms → 0.24 ms |
| `dspy.SIMBA` (`teleprompt/simba.py`) | native | `simba.zig`, `nprandom.zig` | simba.json, suite replay train scores | dates_simba: 282 ms → 2.10 ms |
| `dspy.SandboxSerializable` (`primitives/sandbox_serializable.py`) | missing | – | – | – |
| `dspy.Signature` (`signatures/signature.py`) | partial | `signature.zig` | compile-fail cases, format_cases.json, nested.json | – |
| `dspy.SignatureMeta` (`signatures/signature.py`) | partial | `signature.zig` | comptime signatures | – |
| `dspy.Tool` (`adapters/types/tool.py`) | native | `tool.zig` | react.json, tool_argument_repair_oracle.py | – |
| `dspy.ToolCallResults` (`adapters/types/tool.py`) | native | `tool.zig` | react.json | – |
| `dspy.ToolCalls` (`adapters/types/tool.py`) | native | `tool.zig` | react.json | – |
| `dspy.TrainingJob` (`clients/provider.py`) | missing | – | – | – |
| `dspy.TwoStepAdapter` (`adapters/two_step_adapter.py`) | native | `adapter.zig` | two_step_oracle.py | – |
| `dspy.Type` (`adapters/types/base_type.py`) | partial | `types.zig` | nested.json | – |
| `dspy.XMLAdapter` (`adapters/xml_adapter.py`) | native | `xml.zig`, `xml_schema.zig` | xml.json | cases: 23 ms → 0.16 ms |
| `dspy.asyncify` (`utils/asyncify.py`) | missing | – | – | – |
| `dspy.bootstrap_trace_data` (`teleprompt/bootstrap_trace.py`) | missing | – | – | – |
| `dspy.configure` (`dsp/utils/settings.py`) | partial | `predict.zig` | explicit Context instead of global settings | – |
| `dspy.configure_cache` (`clients/__init__.py`) | partial | `cache.zig`, `request_cache.zig` | cache tests | – |
| `dspy.configure_dspy_loggers` (`utils/logging_utils.py`) | missing | – | – | – |
| `dspy.context` (–) | partial | `predict.zig` | explicit Context | – |
| `dspy.disable_litellm_logging` (`clients/__init__.py`) | missing | – | no litellm | – |
| `dspy.disable_logging` (`utils/logging_utils.py`) | missing | – | – | – |
| `dspy.enable_litellm_logging` (`clients/__init__.py`) | missing | – | no litellm | – |
| `dspy.enable_logging` (`utils/logging_utils.py`) | missing | – | – | – |
| `dspy.ensure_signature` (`signatures/signature.py`) | partial | `signature.zig` | comptime | – |
| `dspy.infer_prefix` (`signatures/signature.py`) | partial | `signature.zig` | format_cases.json | – |
| `dspy.inspect_history` (`clients/base_lm.py`) | missing | – | – | – |
| `dspy.is_retryable_lm_error` (`utils/exceptions.py`) | partial | `http.zig` | transport retry tests | – |
| `dspy.load` (`utils/saving.py`) | native | `state.zig` | check_artifact.py (DSPy loads and re-saves DSZig artifacts) | – |
| `dspy.load_settings` (`dsp/utils/settings.py`) | missing | – | – | – |
| `dspy.majority` (`predict/aggregation.py`) | native | `ensemble.zig` | ensemble.json | – |
| `dspy.make_signature` (`signatures/signature.py`) | partial | `signature.zig` | comptime | – |
| `dspy.resolve_interpreter_factory` (`primitives/code_interpreter.py`) | missing | – | – | – |
| `dspy.settings` (–) | partial | `predict.zig` | explicit Context | – |
| `dspy.streamify` (`streaming/streamify.py`) | missing | – | – | – |
| `dspy.syncify` (`utils/syncify.py`) | missing | – | – | – |
| `dspy.track_usage` (–) | missing | – | – | – |


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
to each simultaneously running attempt or explicitly synchronized. Do not mutate
shared program configuration while calls run. In particular, do not freeze or
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

`parallel.Evaluator(P, limits)` is the native counterpart of
`dspy.Evaluate(num_threads=...)`. `init(io, spawn_config, backing)` creates the
workers, one prediction record and typed output slot per attempt, and a
fixed-buffer scratch allocator over each borrowed `backing[attempt]`.
`evaluate(contexts, &program, devset, metric, options)` runs program plus metric per
example with the same `Options` as `Parallel` (`return_failed_tasks` must be false).
Each attempt's context supplies its own LM/retriever pointees; its arena is
replaced by the reset attempt scratch and its trace by null. The program and metric
must be thread-safe. Scores are summed after the join in devset order, not
completion order, with failed slots scoring 0, so the sum matches serial
`optimize.evaluate` and DSPy. `errors` counts failed examples, not losing
duplicate attempts. As with `Parallel`, reaching `max_errors` returns
`error.ExecutionCanceled` (DSPy's executor raises "Execution cancelled"), while
serial `optimize.evaluate` returns the triggering error. Out-of-memory, including
exhausted attempt scratch, cancels the run and returns `error.OutOfMemory`, as
serially.

Optimizers take DSPy's `num_threads` as `concurrent_evaluation`:
`evaluator.concurrent(&contexts, options)` binds an initialized evaluator to its
program type, per-attempt contexts and options. `BootstrapFewShotWithRandomSearch`,
`BootstrapFewShotWithOptuna`, `MIPROv2`, `COPRO` and `InferRules` then evaluate
candidates through it, each call using the optimizer's own `max_errors`; a
program-type mismatch returns `error.ProgramTypeMismatch`, and null keeps serial
`optimize.evaluate`. Candidate evaluation is sequential between candidates, as in
DSPy; only examples within one evaluation run concurrently. `BootstrapFewShot`
has no DSPy thread option and stays serial; SIMBA and GEPA batch rollouts remain
serial.

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

### DSPy 3.5.0 real-model parity (gpt-4o-mini)

`tests/fixtures/live/dspy_3.5.0_gpt-4o-mini.json` (DSPy 3.5.0, 684 real calls)
and `tests/fixtures/live/dszig_3.5.0_gpt-4o-mini.json` (DSZig, 685 real calls)
were each recorded against the public OpenAI API through the Responses API, with
response-schema and function-calling capabilities enabled. Each recording is
replayed through the other runtime: every request must byte-match a recorded
request, every recorded call must be consumed, and case outputs, scores,
artifacts and optimizer traces must be equal. `check.sh` runs both directions
for every committed `*_3.5.0*.json` transcript without a key.

| Pipeline | DSPy 3.5.0 | DSZig |
|---|---|---|
| RAG + BootstrapFewShot | 100% → 100% | 100% → 100% |
| Dates + BootstrapFewShot | 0% → 100% | 0% → 100% |
| Dates + BootstrapFewShotWithRandomSearch | 0% → 100% | 0% → 100% |
| Dates + MIPROv2 | 0% → 100% | 0% → 100% |
| Dates + MIPROv2 (program-aware proposer) | 0% → 100% | 0% → 100% |
| RAG ReAct + BootstrapFewShot | 100% → 100% | 100% → 100% |
| Dates + COPRO | 0% → 0% | 0% → 0% |
| Dates + SIMBA (small configuration) | 0% → 0% | 0% → 0% |
| Dates + GEPA | 0% → 100% | 0% → 100% |

Accuracy is at par on the same model: equal scores in both independent runs and
identical results when either runtime replays the other's completions. The one
call-count difference comes from sampling, not protocol: the replay checks
prove each runtime makes exactly the other's requests given the same completions.
DSPy 3.5 routes `openai/*` models through the Responses API, so `dz-live` records
DSZig with `DSZIG_LIVE_API=responses`; a Chat Completions recording returns native
tool calls where Responses returns text and does not cross-replay.

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
- **Other modules and optimizers:** no AvatarOptimizer, BetterTogether,
  BootstrapFinetune or bootstrap_trace_data. Native Parallel uses explicit typed
  tasks and contexts, not Python callback objects or ambient thread-local settings.
  Its startup capacities and cooperative join contract are described above.
- **Optimizer execution and shared identity:** SIMBA and GEPA batch rollouts
  run serially. InferRules' signature-alias propagation does not update arbitrary external
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
  beyond OpenAI/Azure, Image/Audio, general rich-media tool results, provider
  training lifecycle or native streaming. Document support is not general
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
