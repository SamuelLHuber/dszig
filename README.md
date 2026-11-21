# DSZig

A native Zig port of selected [DSPy 3.5.0](https://github.com/stanfordnlp/dspy/releases/tag/3.5.0)
APIs, with **no Zig package dependencies**. Signatures are checked at compile time;
allocators and execution context are explicit.

**This is not a feature-complete or fully identical implementation of DSPy.** The
cross-runtime gates cover the behaviors below, not every DSPy capability. In
particular, the Python-executing RLM/Flex stack and full provider SDK coverage
are not implemented. Native image, encoded audio, and document inputs and
OpenAI streaming cover the paths described below, not general multimodal or
streaming parity. See [Compatibility limits](#compatibility-limits).

## Install

Use the repository-pinned Zig 0.17.0 toolchain. From your Zig project,
check out the release using Git credentials with access to this repository:

```sh
git clone --depth 1 --branch 'v3.5.0+dszig.1' https://github.com/SamuelLHuber/dszig.git vendor/dszig
```

Add `.dszig = .{ .path = "vendor/dszig" },` to `.dependencies` in
`build.zig.zon`. For a shared project, vendor the checked-out files or pin
the same commit as a Git submodule.

In `build.zig`, import the dependency's `dszig` module:

```zig
const dep = b.dependency("dszig", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("dszig", dep.module("dszig"));
```

The version core tracks the upstream DSPy release. `+dszig.1` identifies
DSZig's first build of that version; pin an exact tag because SemVer ignores
build metadata for precedence.

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

The library targets Zig 0.17.0. The repository toolchain and CI installer pin
the compiler and apply a Darwin-only standard-library fix for interrupted TCP
connects. Use `devenv shell` on macOS to pick up that fix; an unpatched Zig
0.17.0 can intermittently fail HTTP requests before sending them. The reference
harness separately pins CPython 3.11, DSPy 3.5.0 and every Python dependency in
`tools/dspy_reference/uv.lock`. Python is reference tooling, not a library runtime
dependency. `build.zig.zon` has `.dependencies = .{}`.

### Darwin HTTP transport verification

The intermittent ReleaseSafe OpenAI loopback failure was reproduced before
request transmission: Darwin returned `EINTR`, then `EISCONN` on the same
socket and thread. The pinned Zig stdlib treated the second result as an
unexpected failure. The Darwin toolchain patch accepts it only after an
interrupted connect and checks for cancellation before accepting it.

The retained loopback regression reads the full POST and tests HTTP 201 and
400 in both buffered and SSE calls, including context overflow and no events
on streamed refusals. Pinned DSPy 3.5.0 confirmed those status-handling
outcomes over loopback. A disposable fault injection turned a real successful
connect into `EINTR`; the real retry returned `EISCONN`. The unpatched binary
failed, the patched binary passed, and a first-call `EISCONN` still failed.
The patched regression passed 500 uninstrumented ReleaseSafe runs; the full
ReleaseSafe build passed 847 tests and all 40 steps. The real `live --smoke`
and `live --stream-smoke` commands passed over loopback HTTP 201. No paid
inference was needed; the full Debug and offline reference suites were not
rerun. No local blocker remains. At the next Zig upgrade, run the interrupted
connect control against upstream before deciding whether to carry the patch.

### Gemini buffered continuation verification

A locked DSPy 3.5.0 `GeminiLM.complete` loopback returned an empty thinking part
for a signed `text: null` response and sent the signature back on the next
turn. The native `Gemini.completeRequest` regression reproduced the missing
signature on the second HTTP request before the fix and passed afterward.
All 465 tests rooted at `src/gemini.zig` passed, as did the format check.
The full Debug test and examples build passed (875 tests, 40 steps), and
deterministic replay consumed all 687 requests across eight adapter cases
and nine pipelines. The broader verification run was interrupted during
ReleaseSafe (exit 143), so neither that build nor the final reference
script is confirmed. No local blocker remains for the buffered regression;
hosted Gemini inference was not exercised.

### Gemini streamed continuation verification

The locked DSPy 3.5.0 `GeminiLM.stream` emitted an empty thinking delta and a
continuation delta at the same index for a signed `text: null` part. Its
assembled response replayed that signature on the next HTTP request. DSZig
now distinguishes a present null `text` field from a missing field in
GenerateContent SSE. Before the fix, the retained two-turn regression saw
only the answer in the second POST; after the fix, it matched the reference
body byte for byte, including the empty signed thinking part.

A separate public `Gemini.streamRequest` loopback smoke observed start,
empty thinking, continuation, answer, and stop in order, and matched the
reference request. Formatting and all 466 tests rooted at `src/gemini.zig`
passed in Debug. The targeted ReleaseSafe run did not finish before the
command deadline; full ReleaseSafe and the offline reference suite remain
unconfirmed. No hosted inference was needed for this codec regression.
Next, run the full ReleaseSafe test and examples gate with a separate
deadline, then finish the offline reference gate.

### Gemini streamed part correlation verification

Pinned DSPy 3.5.0 `GeminiLM.stream` reserved a part index for inline PDF
without emitting a media delta. DSZig now preserves that slot: text on
either side stays in separate parts and subsequent function calls keep
their correct correlation IDs. The retained HTTP regression saw delta
indexes `0, 0, 1` before the fix and `0, 2, 3` afterward, with the assembled
call identified as `tool_call_3`.

Formatting and all 467 Debug tests rooted at `src/gemini.zig` passed.
A standalone `Gemini.streamRequest` consumer also passed over loopback,
matching the request body and preserving the same indexes after inline
video. PDF and video still produce no GenerateContent SSE media delta.
There is no local blocker for this fix; hosted inference was not needed.
The full ReleaseSafe test and examples build and the offline reference
gate remain unconfirmed. Run ReleaseSafe next, then the offline gate.

### Gemini streamed tool-argument verification

GenerateContent SSE now sends compact, ASCII-escaped JSON in
`ToolCallDelta.input`, matching the locked DSPy 3.5.0 `GeminiLM.stream`.
The retained HTTP regression failed on the raw argument bytes before the
fix and passed afterward. It also checks that the assembled call retains
the Unicode key and value and the exact integer `9007199254740993`.

Formatting and all 467 Debug tests rooted at `src/gemini.zig` passed.
A standalone public `Gemini.streamRequest` loopback smoke matched the
reference POST body and argument bytes and observed start, tool delta, and
end in order. No local blocker remains for this change; hosted inference
was not needed. The next recommended check is the full ReleaseSafe test
and examples build, followed by the offline reference gate; neither was
rerun here.

### Gemini generation HTTP status verification

Gemini buffered and SSE generation now pass sub-400 HTTP responses to their
response decoders, matching pinned DSPy 3.5.0 instead of rejecting every
status other than 200. DSPy accepted 201 in both modes over loopback and
refused 400. The retained native regression failed on 201 before the fix
and passed afterward; it also verifies that 400 produces no streaming
events even when the body would otherwise decode successfully.

Formatting and all 468 Debug tests rooted at `src/gemini.zig` passed.
A standalone consumer exercised `Gemini.completeRequest` and
`Gemini.streamRequest` over loopback, matching both reference POST bodies
and the buffered answer and SSE start/text/end sequence on 201. No local
blocker remains for this fix; hosted inference was not needed. The full
ReleaseSafe test and examples build and the offline reference gate were
not rerun; run ReleaseSafe next, then the offline gate.

### Gemini context-window failure verification

Gemini buffered generation and GenerateContent SSE now return
`error.ContextWindowExceeded` for HTTP 400 responses whose JSON error message
reports a context-length failure. Callers can distinguish this case from
`error.LMFailure` and shorten the prompt or history. Ordinary HTTP 400
responses and explicit Google `API_KEY_INVALID` errors still return
`error.LMFailure`; a failed streaming request emits no events.

Pinned DSPy 3.5.0 classified the same loopback token-limit response as
`ContextLengthError` in both modes, and classified an ordinary HTTP 400 as
`InvalidRequestError`. The retained native HTTP regression reproduced the
generic error before the fix and passed afterward in both modes. Formatting
and all 468 Debug tests rooted at `src/gemini.zig` passed. An independent
ReleaseSafe consumer exercised `Gemini.completeRequest` and
`Gemini.streamRequest` over loopback, checked both POSTs for the overflow and
invalid-key cases, and observed no retries or stream events.

### Gemini context-window failures beyond HTTP 400

Gemini now checks rejected GenerateContent responses for context overflow
before retrying, regardless of HTTP status. Callers receive
`error.ContextWindowExceeded` without resending the same oversized prompt.
An explicit Google `API_KEY_INVALID` error takes precedence and returns
`error.LMFailure` without retrying. Ordinary transient failures still retry.

Pinned DSPy 3.5.0 returned `ContextLengthError` on HTTP 413, 429, and 503 in
both buffered and SSE loopback calls, and `AuthError` for the explicit
invalid-key case. The retained native regression failed on HTTP 413 before
the fix and passed afterward. It also checks that an ordinary 503 retries
and succeeds, and that rejected streams emit no events. Formatting and all
468 Debug tests rooted at `src/gemini.zig` passed. A standalone ReleaseSafe
consumer exercised `Gemini.completeRequest` and `Gemini.streamRequest` for
the three overflow statuses and the invalid-key case, verifying all eight
POSTs byte for byte with no retries or stream events.

No local blocker remains for this change. The full Gemini error taxonomy
and hosted inference were not verified. Next, run the full ReleaseSafe test
and examples build, then the offline DSPy reference gate; neither was rerun
in this iteration.

### Gemini terminal RPC failures on retryable HTTP statuses

GenerateContent now stops retrying when a rejected response reports a
terminal Google RPC status, even if the HTTP status alone would retry.
`INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `PERMISSION_DENIED`,
`UNAUTHENTICATED`, and `NOT_FOUND` return `error.LMFailure` in buffered and
SSE requests. Context-window detection and explicit invalid-key precedence
are unchanged, and a transient `UNAVAILABLE` HTTP 503 still retries.

Pinned DSPy 3.5.0 classified all five HTTP 503 envelopes as terminal over
loopback in buffered and SSE calls. Its public LM sent one request for each
terminal case and two for the transient control. The retained native HTTP
regression observed an unexpected retry before the fix. Afterward, the
Debug test and examples build passed (878 tests, 40 steps), as did formatting.
A standalone ReleaseSafe consumer exercised `Gemini.completeRequest` and
`Gemini.streamRequest` and verified four exact POSTs, no retries, and no
events for permission denial and context overflow.

No local blocker remains. Hosted Gemini inference and the full error
taxonomy were not verified. Next, run the full ReleaseSafe test and
examples gate, then the offline reference gate; neither was rerun here.

### Gemini retries for timeouts and transient RPC failures

Buffered and streaming GenerateContent now retry HTTP 408 and recognized
transient Google RPC errors (`RESOURCE_EXHAUSTED`, `INTERNAL`, `UNAVAILABLE`,
and `DEADLINE_EXCEEDED`) even when the HTTP status alone would not retry.
Terminal RPC errors still take precedence, and the configured retry budget
still applies. Rejected attempts do not emit stream events.

Pinned DSPy 3.5.0 recovered on a second loopback POST in both modes for HTTP
408 and for all four transient RPC errors carried in HTTP 400. It did not
retry a plain HTTP 400 or a permission-denied HTTP 408. The retained native
HTTP regression failed on 408 before the fix and passed afterward. A standalone
ReleaseSafe consumer verified both public Gemini methods, exact POST bodies,
stream event order, and zero and exhausted retry budgets across 18 cases.

Formatting passed. The full ReleaseSafe test and examples build passed all
878 tests and 40 steps. No local blocker remains for this change. Hosted
inference and the complete error taxonomy were not verified. The next
recommended item is the offline DSPy reference gate
(`devenv shell -- tools/dspy_reference/check.sh`), which was not run here.

### Gemini error envelope verification

Buffered and streaming GenerateContent now recognize context-limit messages
when the HTTP error's `error` field is a string or an array, not just when
it is an object. An explicitly non-string object `message` falls back to
the HTTP status instead of using the RPC status to decide whether to retry.
A missing message still defaults to an empty string.

Pinned DSPy 3.5.0 returned `ContextLengthError` for the string and array
envelopes on HTTP 413, and `InvalidRequestError` for an HTTP 400 envelope
with `status: "UNAVAILABLE"` and `message: null`. Each buffered and streaming
call made one loopback request. The retained native HTTP regression failed
before the fix and passed afterward. Formatting and the Debug test and
examples build passed (878 tests, 40 steps). A standalone ReleaseSafe
consumer exercised both public Gemini methods and verified six exact POSTs,
no retries, and no stream events.

No local blocker remains for this fix. Hosted inference was not needed.
The next recommended item is the offline DSPy reference gate
(`devenv shell -- tools/dspy_reference/check.sh`), which was not run here.

### Gemini context-window failures inside a stream

When Gemini accepts the POST but reports a context-limit error in a later
SSE frame, `LM.completeResponses` now returns `error.ContextWindowExceeded`
instead of `error.LMFailure`. Callers can shorten the prompt and try again.
The stream sink still receives the structured provider error after any
partial text. There is no automatic retry, end event, or completed response.
Ordinary in-stream invalid requests still return `error.LMFailure`.

Pinned DSPy 3.5.0's public Gemini stream and response collector confirmed
both cases over loopback. The retained native HTTP regression failed on
the generic error before the fix and passed afterward. Formatting and the
full Debug test and examples build passed (878 tests, 40 steps). The focused
regression also passed in ReleaseSafe, and a separate ReleaseSafe consumer
verified both cases, including the exact POSTs and event order. No local
blocker remains; hosted inference was not needed.
The full ReleaseSafe build and offline DSPy reference gate were not rerun
for that change.

### Gemini request IDs on streamed errors

GenerateContent SSE error events now expose `provider_error.request_id` so
callers can correlate a failed stream with provider or gateway logs. Header
names are case-insensitive, and the first value for each recognized name
is considered in DSPy order (`x-request-id` before `request-id`). If no
recognized header supplies an ID, the field is null.

Locked DSPy 3.5.0 confirmed the header precedence on a real loopback POST.
The native loopback regression failed before the fix and passed in Debug
and ReleaseSafe. It covers the raw stream and collected completion, including
a no-header control. A standalone ReleaseSafe consumer verified one exact
POST, partial text followed by a context-length error with the request ID,
and `error.ContextWindowExceeded` from the collector. Formatting and the full
Debug test and examples build passed (878 tests, 40 steps). Hosted inference
was not needed; the full ReleaseSafe build was not run.

The offline reference gate, started before this fix, regenerated the fixtures,
round-tripped the native artifact, and matched the 687-call mock suite in
both runtimes. It also matched a 679-call recording in Zig, but its
240-second deadline expired during the later replay checks. The gate has
not passed end to end. Next, run the remaining recording checks separately
with a longer deadline, reusing the completed fixture and mock comparisons.

### Gemini retry hints on streamed errors

GenerateContent SSE now exposes `provider_error.retry_after` in seconds.
Callers can use the hint to decide when to retry; DSZig does not replay a
stream after delivering an in-stream error. Header names are case-insensitive,
and the first value for each name is used. A valid numeric `Retry-After`
takes precedence over `Retry-After-Ms`, then `X-MS-Retry-After-MS`. Invalid
and nonfinite values fall through; zero is retained. HTTP-date hints and
other rate-limit headers are not exposed.

Pinned DSPy 3.5.0 and standalone Debug and ReleaseSafe consumers agreed
on five loopback cases: `2.5`, `0.1255`, `0`, and two absent hints. Each native
consumer verified five exact POSTs and one rate-limit event per stream.
The retained HTTP regression failed before the fix and passed in both
build modes. Formatting and the full Debug test and examples build passed
(878 tests, 40 steps). No hosted inference was needed. The full ReleaseSafe
build and the remaining offline recording comparisons were not run here.
Next, finish the offline recording comparisons noted above without rerunning
the completed fixture and mock checks.

### Gemini non-string thought signatures

Buffered and streamed GenerateContent now convert non-null thinking and
answer signatures to strings instead of rejecting the response. The retained
two-turn HTTP regression failed on the buffered call before the fix and
passed in both Debug and ReleaseSafe afterward. Pinned DSPy 3.5.0 and a
standalone ReleaseSafe consumer matched all four POSTs byte for byte:
both modes replayed `9007199254740993` without rounding and `false` as
`"False"`, then returned the second-turn answer.

Formatting and the full Debug test and examples build passed (879 tests,
40 steps). No local blocker remains, and no hosted inference was needed.
The full ReleaseSafe build and the outstanding offline recording comparisons
were not run here. Next, finish those recording comparisons with a separate
deadline, reusing the completed fixture and mock checks.

### Gemini function-call continuation verification

Buffered and streamed GenerateContent now preserve truthy non-string
function-call IDs and names by converting them to strings. Falsy IDs still
use the generated correlator. A truthy part-level thought signature takes
precedence over the function-call signature; a falsy outer value falls back
to the inner value, which is retained even when it is zero.

The two-turn HTTP regression failed before the fix and passed afterward in
Debug and ReleaseSafe. A standalone ReleaseSafe consumer matched pinned
DSPy 3.5.0's request bodies byte for byte across four loopback POSTs and
replayed both tool calls on the next turn, including the exact ID
`9007199254740993` and the fallback signature `"0"`. Formatting and the full
Debug test and examples build passed (880 tests, 40 steps).

No local blocker remains for this fix. Hosted inference was not needed.
The full ReleaseSafe build and the outstanding offline recording comparisons
were not run. Next, finish those recording comparisons with a separate
deadline, reusing the completed fixture and mock checks.

### Gemini content-filter finish reasons

Buffered and streamed GenerateContent now report `content_filter` rather
than `stop` for case variants of Gemini's content-filter finish reasons.
Exact uppercase refusal codes such as `SAFETY` still take the existing
error path without publishing the rejected text.

Pinned DSPy 3.5.0 and a standalone ReleaseSafe consumer agreed over
loopback on `safety`, `Recitation`, `SAFETY`, and `MAX_TOKENS` in both
modes. The consumer verified all eight POST bodies and the SSE event
sequence. The retained native HTTP regression failed before the fix and
passed in the full Debug test and examples build (880 tests, 40 steps).
Formatting passed.

No local blocker remains for the fix; hosted inference was not needed.
The full ReleaseSafe build and remaining offline recording comparisons
were not run. Next, finish those recording comparisons with a separate
deadline, reusing the completed fixture and mock checks.

### Gemini buffered response identifiers

`Gemini.completeRequest` now exposes truthy GenerateContent `responseId`
values as strings in the canonical response, so callers can correlate
responses even when a gateway returns a numeric ID. Falsy values still
produce no ID. This does not change streamed response IDs.

Locked DSPy 3.5.0 and a standalone ReleaseSafe consumer agreed on seven
loopback responses and their exact POST bodies. The retained HTTP regression
failed before the fix because `9007199254740993` was dropped; it now checks
that all digits survive. Formatting passed. Debug and ReleaseSafe test and
example builds each passed 880 tests and all 40 steps. Native deterministic
replay consumed all 687 requests across eight adapter cases and nine pipelines.

No local blocker remains for this fix. Hosted inference was not needed, and
the full offline reference gate was not run. Next, replay the two existing
DSPy 3.5.0 live recordings in both runtimes offline, reusing the completed
fixture and mock checks.

### Gemini streamed finish markers

GenerateContent SSE now emits an end event for truthy `finishReason` values
even when a gateway sends a non-string marker. An unknown marker falls back
to `stop` unless the frame contains a tool call. Falsy markers such as `0`
and `false` do not end the stream on their own.

Pinned DSPy 3.5.0 and a standalone ReleaseSafe `Gemini.streamRequest`
consumer agreed on event order and finish reasons across six loopback
responses and sent identical POST bodies. The retained regression reproduced
the missing end event before the fix. Formatting and the full Debug test
and examples build passed (880 tests, 40 steps).

No local blocker remains for this fix. Hosted inference was not needed.
The full ReleaseSafe build was not rerun at that checkpoint. Both saved
3.5.0 recordings have since replayed offline in both runtimes: each matched
eight adapter cases and nine pipelines, consuming all 679 DSPy-recorded
and 680 DSZig-recorded requests, respectively.

### Gemini prompt-block verification

Buffered GenerateContent now rejects truthy `promptFeedback.blockReason`
values, including numeric markers, instead of returning a successful
candidate from the same response. Buffered and SSE generation share the
block check: zero and `BLOCK_REASON_UNSPECIFIED` do not block a response.

Pinned DSPy 3.5.0 rejected numeric `1` over loopback in both modes. The
retained native HTTP regression reproduced a successful buffered response
before the fix and now checks that it returns `error.LMFailure`, that SSE
emits only an invalid-request event, and that both nonblocking controls
still return the candidate. Formatting and the full Debug test and examples
build passed (880 tests, 40 steps). A standalone ReleaseSafe consumer
exercised `Gemini.completeRequest` and `Gemini.streamRequest` and matched
the six reference POST bodies byte for byte.

No local blocker remains for this fix; hosted inference was not needed.
The full ReleaseSafe test and examples build and the full offline reference
gate were not rerun. Run the ReleaseSafe build next; the saved recording
comparisons above need not be repeated for this change.

### Gemini generated media from gateways

GenerateContent now preserves inline media when a gateway returns truthy
non-string data, converting it to text rather than dropping it from a
buffered response or emitting an empty SSE delta. Buffered media also
uses the same conversion for file URIs and MIME types, with
`application/octet-stream` as the default for a falsy MIME type. Falsy data
still produces no buffered part; streamed images and audio still emit an
empty delta for falsy data. SSE does not add document or file-URI deltas.

Pinned DSPy 3.5.0 and a standalone ReleaseSafe consumer agreed over loopback
on five buffered parts and four SSE deltas, including numeric image data
`1234`, ordinary audio, and the empty-image control. Both native POSTs
matched the reference byte for byte. The retained HTTP regression found
two buffered parts instead of three before the fix and passed afterward.
Formatting and the full Debug test and examples build passed (881 tests,
40 steps).

No local blocker remains; hosted inference was not needed. Next, run the
full ReleaseSafe test and examples build. The full offline reference gate
was not rerun.

### Gemini replay of opaque thought signatures

GenerateContent now preserves truthy `thought_signature` values from a
canonical request's continuation records, rather than silently dropping
non-string values. The value keeps its JSON type, and an unsigned or falsy
thinking signature still leaves out both `thoughtSignature` and `thought`.
This changes request encoding only; response signatures are still decoded
as strings.

Pinned DSPy 3.5.0's `GeminiLM.complete` sent the exact integer
`9007199254740993` on a thinking part, omitted a false signature on a text
part, and retained an object signature on a tool call. The native loopback
regression reproduced both dropped values before the fix and matched the
reference POST byte for byte afterward in Debug and ReleaseSafe. A separate
ReleaseSafe SDK consumer verified exact POSTs from `Gemini.completeRequest`
and `Gemini.streamRequest`, including the buffered answer and SSE event
order. Formatting and the full Debug test and examples build passed
(881 tests, 40 steps).

No local blocker remains. Hosted Gemini acceptance of non-string signatures
was not tested. Next, run the full ReleaseSafe test and examples build; the
full offline reference gate was also not rerun for this change.

### Gemini malformed candidate envelopes

GenerateContent now returns `error.LMFailure` for a nonempty candidate
object where Gemini should send a candidate array. Previously, a buffered
call could report a successful empty answer, and SSE could emit a successful
end event if the frame also carried usage. Both paths still accept an empty
candidate object as an empty result, matching pinned DSPy 3.5.0.

The reference rejected the malformed envelope over loopback in both modes.
The retained native regression failed before the fix and now covers both
public APIs, including the empty-object control. A standalone ReleaseSafe
consumer verified all six loopback POST paths and bodies, rejection without
SSE events, and normal results for the empty-object and valid-array controls.
Formatting and the full Debug and ReleaseSafe test and examples builds
passed (881 tests and 40 steps in each mode).

No local blocker remains. Hosted Gemini was not exercised; loopback was
sufficient for this decoding fix. The next verification item is the full
offline DSPy reference gate, which was not rerun here.

### Gemini non-iterable content parts

Buffered and streamed GenerateContent now return `error.LMFailure` when
`candidates[0].content.parts` is a truthy number or boolean. Previously an
HTTP 200 carrying such a value could look like a successful empty answer.
The malformed SSE frame emits no events. `0`, `false`, and `null` still
behave like an empty list.

Pinned DSPy 3.5.0 rejected `1` and `true` over loopback in both modes and
accepted the three empty controls and a valid parts array. The retained
native HTTP regression reproduced the false success before the fix and
passed afterward. A standalone ReleaseSafe consumer exercised
`Gemini.completeRequest` and `Gemini.streamRequest`, verifying all 12
POST paths and bodies, the response and event outcomes, and no retries.
Formatting, the focused ReleaseSafe regression, and the full Debug test
and examples build passed (881 tests, 40 steps).

No local blocker remains. Hosted inference was not needed. The full
ReleaseSafe build and offline reference gate were not rerun for this
change. Next, run `devenv shell -- tools/dspy_reference/check.sh`.

### Gemini terminal stream usage verification

GenerateContent now validates terminal-frame usage before invoking stream
callbacks for that frame. A malformed token count returns `error.LMFailure`
without publishing the frame's text or an end event. Text from earlier valid
frames remains delivered, and a valid terminal frame still delivers its
text and reports usage at stream exhaustion.

Pinned DSPy 3.5.0 and a standalone ReleaseSafe `Gemini.streamRequest`
consumer agreed on three loopback cases: a malformed first frame, malformed
usage after valid text, and a valid terminal frame. Their POST paths and
bodies matched byte for byte. The retained regression reproduced premature
start and text callbacks before the fix. Formatting, the focused ReleaseSafe
HTTP regression, and the full Debug test and examples build passed (881
tests, 40 steps).

No local blocker remains. Hosted inference was not needed. The full
ReleaseSafe build and offline reference gate were not run in this iteration.
Next, run `devenv shell -- tools/dspy_reference/check.sh`.

### Gemini output audio token accounting

Buffered and streamed GenerateContent now select the audio usage breakdown
before validating its shape. A nonempty but malformed `candidatesTokensDetails`
leaves output audio tokens unknown rather than reporting a count from
`responseTokensDetails`. An empty primary list still falls back to the alternate
field; a nonempty primary list is authoritative, even when it has no audio rows.

Locked DSPy 3.5.0 and a standalone ReleaseSafe consumer agreed on four
loopback cases in both modes. The native consumer verified eight exact POST
bodies, unchanged output token totals, the buffered answer, and SSE event order.
The retained HTTP regression reported an integer instead of null before the
fix and passed afterward in Debug and ReleaseSafe. Formatting passed, and the
full Debug test and examples build passed (882 tests, 40 steps).

No local blocker remains for this fix. Hosted inference was not needed.
The full ReleaseSafe build and offline reference gate were not rerun.
Next, run `devenv shell -- tools/dspy_reference/check.sh`.

### Gemini malformed audio token counts

Buffered and streamed GenerateContent now reject explicit null or string
`tokenCount` values in `AUDIO` breakdown rows instead of treating them as
reported zero or parsing them as integers. Omitted counts still contribute
zero, valid counts still sum across rows, and non-audio rows are ignored.
Malformed terminal SSE frames publish no events from that frame.

Locked DSPy 3.5.0 rejected both malformed counts over loopback. A standalone
ReleaseSafe consumer matched all eight reference POST paths and bodies and
verified the buffered and streamed results for both malformed cases and both
valid controls. The retained HTTP regression failed before the fix on an
explicit null and passed afterward. Formatting and the full Debug test and
examples build passed (882 tests, 40 steps).

No local blocker remains. Hosted inference was not needed. The full
ReleaseSafe build and offline reference gate were not rerun. Next, finish
the offline DSPy reference gate (`devenv shell -- tools/dspy_reference/check.sh`).

### Gemini audio token aggregation

Buffered and streamed GenerateContent now sum audio breakdown rows before
validating the final usage counters, matching locked DSPy 3.5.0. Fractional
rows can add up to a valid integer count, including across prompt and
tool-use breakdowns, and boolean rows contribute zero or one. A final
fractional count, explicit null row count, or string row count still fails.

The retained HTTP regression reproduced the fractional-row failure before
the fix and passed after it in Debug and ReleaseSafe. A standalone ReleaseSafe
consumer exercised the public Gemini SDK over loopback and verified exact
POST paths and bodies, buffered results, and SSE results against locked DSPy.
The full Debug test and examples build passed (882 tests, 40 steps). The
offline reference gate also completed, including unchanged fixture regeneration
and the 687-request mock replay in both runtimes.

No local blocker remains. Hosted inference was not needed or exercised.
Next, run the full ReleaseSafe test and examples build; only the targeted
ReleaseSafe checks ran in this iteration.

### Gemini text inside tool results

Buffered and streaming GenerateContent now accept text inside a canonical
tool result even when that text carries continuation metadata. Gemini's
`functionResponse` receives the text, including empty lines, but not the
nested metadata. Assistant thought signatures still go back on the wire.

The retained HTTP regression failed with `error.LMFailure` before the fix
and passed afterward. A standalone ReleaseSafe consumer exercised
`Gemini.completeRequest` and `Gemini.streamRequest` over loopback:
both POST targets and bodies matched locked DSPy 3.5.0 byte for byte,
and the consumer received the expected answer and SSE start/text/end
sequence. Formatting and the full ReleaseSafe test and examples build
passed (882 tests, 40 steps), completing the previous checkpoint's
outstanding ReleaseSafe check.

No local blocker remained. Hosted inference was not needed, and the full
offline reference gate was not rerun.

### Gemini continuation on tool results

Buffered and streaming GenerateContent now accept canonical tool results
that carry their own continuation metadata. As in locked DSPy 3.5.0, the
metadata stays off the `functionResponse` wire object, while assistant
thought signatures and the tool result's content are still sent.

The retained HTTP regression returned `error.LMFailure` before the fix and
passed afterward in Debug and ReleaseSafe. A standalone ReleaseSafe consumer
exercised `Gemini.completeRequest` and `Gemini.streamRequest` over loopback:
both POST targets and bodies matched DSPy byte for byte, and the consumer
received the buffered answer and the SSE start/text/end sequence. Formatting
and the full Debug test and examples build passed (882 tests, 40 steps).

No local blocker remains. Hosted inference was not needed; the full
ReleaseSafe build and offline reference gate were not rerun for this change.
Next, verify whether GenerateContent should also omit continuation metadata
on image and document parts rather than rejecting those parts.

### Gemini continuation on media

Buffered and streaming GenerateContent no longer reject media solely because
it carries SDK continuation metadata. The metadata stays off `inlineData` and
`fileData`, matching locked DSPy 3.5.0. Assistant thought signatures still go
back on the wire. Local media paths were not part of that change.

The retained HTTP regression failed with `error.LMFailure` before the fix
and passed afterward with an inline image and a PDF attached to a tool
result. A standalone ReleaseSafe consumer exercised `Gemini.completeRequest`
and `Gemini.streamRequest` over loopback. Both POST targets and bodies
matched DSPy byte for byte, and the consumer received the buffered answer
and SSE start/text/end sequence. Formatting and the full Debug test and
examples build passed (882 tests, 40 steps).

No local blocker remains. Hosted inference was not needed. The full
ReleaseSafe build and offline reference gate were not rerun for that iteration.

### Gemini local media in GenerateContent

`Gemini.completeRequest` and GenerateContent SSE via `Gemini.streamRequest`
now accept caller-provided local media paths, including image and document
attachments in function results. The file is read when the request is encoded
and sent as base64 `inlineData`. Supply only trusted paths: the file contents
are sent to the configured endpoint. Native reads are capped at 16 MiB per
file. Gemini Live WebSocket paths are covered separately below.

Locked DSPy 3.5.0 and a standalone ReleaseSafe Zig consumer sent identical
POST targets and bodies over loopback, including a PDF tool attachment and
an image whose bytes changed between the buffered and streaming calls. The
retained regression failed with `error.LMFailure` before the fix and now
passes. It also checks that a missing file fails before credentials are
requested or any stream events are emitted. Formatting passed, and the full
test and examples builds passed in both Debug and ReleaseSafe (883 tests
and 40 steps in each mode).

The full offline reference gate (`devenv shell -- tools/dspy_reference/check.sh`)
passed after this change. DSPy regenerated the fixtures unchanged and
round-tripped the native RAG artifact byte for byte. Both runtimes matched
eight adapter cases and nine pipelines per recording, consuming all 687
mock requests and all 679 DSPy-recorded and 680 DSZig-recorded requests.
No local blocker remains. These checks did not call a hosted model or
revalidate provider credentials.

### Gemini Live local media

Gemini Live now reads local SDK media paths when encoding a streamed request
or a persistent `Gemini.LiveSession.send(.turn)` call. Each read uses the same
16 MiB per-file cap as GenerateContent. Audio-native requests also accept
local WAV and video files as realtime input; WAV sample data is sent as PCM
with its sample rate. Only supply trusted paths, because the file contents
are sent to the configured endpoint. This does not add file uploads or
change the inline-only `.audio` and `.image` session shortcuts.

The retained loopback WebSocket regression matched seven input frames
byte for byte against the locked DSPy 3.5.0 encoder, including WAV-to-PCM
conversion and a file changed between calls. It also received a streamed
reply and verified that a deleted file fails on the next session turn.
Formatting and the full Debug and ReleaseSafe test and examples builds
passed (884 tests and 40 steps in each mode). No local blocker remains.
The WebSocket proof used loopback; hosted Gemini inference was not tested.
The full offline gate (`devenv shell -- tools/dspy_reference/check.sh`)
also passed after this change. DSPy regenerated the fixtures unchanged
and round-tripped the native RAG artifact byte for byte. Both runtimes
matched eight adapter cases and nine pipelines per recording, consuming
all 687 mock requests and all 679 DSPy-recorded and 680 DSZig-recorded
requests.
+
+Audio-native streamed turns now match DSPy 3.5.0 when the conversation
+includes earlier tool calls/results or URL-only audio/video. The encoder
+omits those parts without fetching the URLs. URL-only media still disables
+automatic activity detection in setup, but only actual audio/video frames
+trigger activity markers. If no input remains, the turn sends empty
+realtime text. Persistent session turns and non-audio-native history
+encoding are unchanged.
+
+The loopback regression failed with `LMFailure` before the fix and now
+matches the pinned encoder's input frames for mixed history and URL-only
+turns. Both turns receive a reply and close without an extra input frame.
+Formatting and the full Debug build passed (884 tests across 40 steps).
+Hosted Gemini inference remains unverified.

### Gemini Live inline WAV

Audio-native `Gemini.streamRequest` now accepts inline WAV data URIs and
base64 with whitespace accepted by DSPy 3.5.0. It decodes the WAV and sends
PCM with the file's sample rate. It also accepts nonzero unused base64 pad
bits, as DSPy does, without relaxing the alphabet or padding shape.

The retained WebSocket regression failed with `LMFailure` before the fix
and passed afterward. A standalone ReleaseSafe consumer matched all three
input frames from the pinned DSPy encoder byte for byte, received a reply,
and closed the WebSocket. Formatting and the full Debug test and examples
build passed (884 tests, 40 steps).

The ReleaseSafe test and examples build passed all 40 steps on rerun.
The first run passed 883 of 884 tests; the Gemini HTTP/RPC retry test failed
without diagnostics. The same binary passed with the original seed, and the
isolated retry test passed 200 consecutive runs. The cause remains unknown.
If it recurs, capture the failing case, HTTP status and transport error
before changing retry behavior.

The locked DSPy 3.5.0 reference gate passed, including all 687 deterministic
requests, eight adapter cases and nine pipelines. Both current-version
OpenAI recordings also replayed offline. No new hosted inference was run;
Gemini Live remains verified over loopback only.

### Gemini Live explicit audio output

When a streamed Live request explicitly selects audio output and includes
an audio or video part, DSZig now disables automatic activity detection
even on models without native audio input framing. Other detection settings
are preserved, and those models still send media in `clientContent` turns.

The retained loopback WebSocket regression failed before the fix. It now
matches the pinned DSPy 3.5.0 setup and input frames byte for byte, receives
a reply, and closes the stream. Formatting passed, and the Debug and
ReleaseSafe test and examples builds each passed 884 tests across 40 steps.
The broader offline reference gate was not rerun for this change. Hosted
Gemini inference remains unverified.
The next recommended Live parity check is structured system prompts.

### Gemini Live structured system prompts

Streamed Live requests and persistent sessions now preserve an explicitly
supplied sequence of system prompt parts even when it renders as empty text.
An omitted system prompt still omits `systemInstruction`, and blank lines
between text parts are preserved.

The pinned DSPy 3.5.0 encoders produced the expected setup frames for both
Live paths. The retained regression failed before the fix and passed after it
over loopback WebSocket, checking exact setup frames, a streamed reply, and
a persistent session turn. Formatting and the full Debug and ReleaseSafe
test and examples builds passed (884 tests and 40 steps in each mode).

The full offline reference gate subsequently passed: DSPy regenerated the
fixtures unchanged, round-tripped the DSZig artifact byte for byte, and replay
matched all eight adapter cases and nine pipelines across 687 mock requests.
Hosted Gemini inference was not exercised.

### Gemini Live default output modality

`Gemini.streamRequest` defaults a missing `generationConfig.responseModalities`
to `["TEXT"]` on non-audio-native Live requests. It keeps other generation
settings and preserves an existing modality key (even `null`) unless audio
or image output takes precedence. Persistent `Gemini.live` sessions do not
acquire this default. The earlier omission matched the reference setup
builder, but not DSPy 3.5.0's public `GeminiLM.stream`, which adds `TEXT`
before sending the frame.

### Gemini Live model-name escaping

Streamed requests and persistent Gemini Live sessions now percent-encode the
model resource name in the WebSocket setup frame. They preserve `/`, `:`, and
`@`, encode spaces and non-ASCII characters, and add `models/` only when it
is missing. Ordinary model names are unchanged.

The loopback regression reproduced the unescaped setup frame before the fix.
Afterward, `Gemini.streamRequest` and `Gemini.live` each sent all three setup
frames byte for byte against the pinned DSPy 3.5.0 encoders. The stream
received a reply, and the persistent session sent its first turn.
Formatting, the focused ReleaseSafe regression,
and the full Debug test and examples build passed (884 tests, 40 steps).
Hosted Gemini and the broader offline gate were not rerun for this change.

### Gemini Live setup rejection precedes turn preparation

`Gemini.streamRequest` now waits for the WebSocket setup acknowledgement
before encoding the turn or reading local media. Previously, a missing
local file could make the request fail before the server saw the setup.
If the server rejects setup, both `Gemini.streamRequest` and `Gemini.live`
return `error.LMFailure` without sending a turn or publishing stream events.

Pinned DSPy 3.5.0 raised `InvalidRequestError` on the same loopback refusal
in both public paths. The retained Zig regression failed before the fix
because the missing file prevented the connection. After the fix, a separate
ReleaseSafe consumer checked three exact setup frames, no turn after either
refusal, and a successful streamed turn using a file created just before
the acknowledgement. Formatting, the focused ReleaseSafe regression, and
all 884 Debug tests and 40 test and example build steps passed.

The full ReleaseSafe test and examples build passed with the pinned Zig
toolchain and `-j1`: 884 tests and 40 build steps. The initial build ran
out of disk space while compiling `tests/behavior.zig`; the successful
retry started with an empty local Zig cache.

The offline DSPy 3.5.0 reference gate also passed. Fixture regeneration
produced no differences, and DSPy loaded and re-saved the 1,571-byte
native RAG artifact byte for byte. Both runtimes matched eight adapter
cases and nine pipelines per recording, replaying all 687 deterministic
mock requests and all 679 and 680 requests from the two saved model
transcripts. No fresh hosted inference was exercised.

### Gemini Live output transcription indexes

`Gemini.streamRequest` now reports output transcription at part index `0`,
matching DSPy 3.5.0. Before the fix it omitted the index, leaving consumers
without the correlator used by other text deltas for that part.

The retained WebSocket regression failed before the fix and passed afterward.
Pinned DSPy and a standalone ReleaseSafe Zig consumer both observed text at
indexes `1, 0, 0` in arrival order on the same loopback scenario. Formatting
and all 474 Debug tests rooted at `src/gemini.zig` passed. No local blocker
remains for this fix. Hosted Gemini and the full build and reference gates
were not rerun.

### Gemini Live mixed audio and transcription

A retained WebSocket regression now checks the response assembled by
`gemini.lm().completeResponses` with a stream sink. Two PCM chunks at part
index `0`, one without base64 padding, become a single WAV part. Transcription
and subsequent text at index `0` join ahead of that audio part; text at
index `1` follows it in the assembled response.

Pinned DSPy 3.5.0 and standalone Zig consumers in Debug and ReleaseSafe
matched the five deltas in arrival order, the stop reason, and all three
assembled parts, including the exact WAV bytes. The focused regression
passed in both modes without a production code change. Hosted Gemini and
the full repository gates were not rerun.

### Gemini Live cumulative usage

The mixed audio and transcription regression now checks that the assembled
response uses the latest nonempty usage snapshot, not a sum or field-by-field
merge. It covers usage nested under `serverContent`, precedence for top-level
usage and Live's `responseTokenCount`, and an empty update followed by a
terminal frame without usage.

Pinned DSPy 3.5.0 and DSZig agreed over loopback WebSocket: 5 input tokens,
6 output tokens, and 11 total, with cache-read, reasoning, and output-audio
counts preserved and the earlier input-audio count absent. Formatting and
the focused Zig regression passed in Debug and ReleaseSafe. No runtime
change was needed. Hosted Gemini and the full repository gates were not rerun.

### Gemini Live terminal-frame usage

Gemini Live now validates a frame's usage before delivering its tool-call
deltas. A terminal frame with invalid usage returns `error.LMFailure`
without exposing its tool call or emitting an end event. With valid usage,
the terminal snapshot replaces the previous one, and top-level metadata
takes precedence over metadata nested under `serverContent`.

The native WebSocket regression exercised `gemini.lm().completeResponses`
with a stream sink. Reinstating the old ordering in an isolated copy made
the malformed case expose one tool call instead of zero. The public DSPy
3.5.0 stream with scripted transport agreed on event delivery and on the
valid case's 5 input, 2 output, and 7 total tokens. All 474 tests rooted at
`src/gemini.zig` passed in Debug, and the focused regression passed in
ReleaseSafe. Hosted Gemini and the full repository gates were not rerun.

### Gemini Live interruption usage

A retained loopback WebSocket regression exercises `Gemini.live` and the
public session interrupt and receive methods. It verifies that partial
transcription and its usage arrive before the interruption event, that
top-level usage wins over nested usage, and that a frame marked both
interrupted and complete reports its usage only on `turn_end`.

Pinned DSPy 3.5.0's public Live session agreed over scripted transport.
Formatting and the focused Zig regression passed in Debug and ReleaseSafe
without a runtime change. Hosted Gemini and the full repository gates were
not rerun.

### Gemini Live usage across a tool exchange

The loopback regression now sends a tool result using the received call ID
before reading the queued usage event. It verifies that the tool-call frame's
9 tokens remain a separate `usage` event and that `turn_end` reports only the
15 tokens from the terminal frame, not their sum.

Pinned DSPy 3.5.0 agreed on the event sequence and client payloads over
scripted transport. The focused Zig regression passed in Debug and ReleaseSafe
without a runtime change. Hosted Gemini and the full repository gates were
not rerun for that regression.

### Repository verification after the Gemini changes

The full ReleaseSafe test and examples build passed all 884 tests and 40
build steps. The locked DSPy 3.5.0 offline gate also passed: fixtures
regenerated unchanged, DSPy round-tripped the DSZig artifact byte for byte,
and both runtimes replayed all 687 mock requests and both saved OpenAI
transcripts. The mock replay matched eight adapter cases and nine pipelines.

No runtime change was needed for this verification step. No fresh hosted
inference was run. Next, verify authenticated Gemini Live text and tool
round-tripping when Gemini credentials are available; loopback coverage
does not qualify the hosted service.

### Gemini Live inline audio from gateways

Streamed completions and persistent Gemini Live sessions now convert truthy
non-string inline audio data to text instead of silently emitting an empty
chunk. This matches the public DSPy 3.5.0 Live APIs over scripted transport.
The native loopback regression failed before the fix and passed in Debug
and ReleaseSafe afterward. It checks the streamed audio delta, exact WAV
assembly, and audio delivery through `LiveSession.recv`.

The full ReleaseSafe test and examples build passed all 884 tests and 40
build steps after this change. The locked DSPy 3.5.0 offline gate also
passed: fixtures regenerated unchanged, DSPy round-tripped the DSZig
artifact byte for byte, and both runtimes replayed all 687 mock requests
and both saved OpenAI transcripts.

Hosted Gemini Live text and tool round-tripping remains unverified until
`GEMINI_API_KEY` or `GOOGLE_API_KEY` is available in the launch environment.

### Gemini Live developer turns

Non-audio-native `Gemini.streamRequest` now preserves developer messages
as one `[developer]`-prefixed text part per turn, matching locked DSPy 3.5.0.
Previously the WebSocket transcript silently sent those instructions as
ordinary user parts. Live transcript turns and GenerateContent now share
the same message encoder. Audio-native framing and persistent session sends
are unchanged.

The retained loopback regression failed on the transcript before the fix
and passed afterward. A standalone ReleaseSafe SDK consumer matched the
reference setup and transcript frames byte for byte, received start/text/end,
and closed the socket. Formatting and the full Debug tests and examples
passed (884 tests, 40 build steps). The full ReleaseSafe and offline reference
gates were not rerun for this change.

### Gemini Live default modality verification

The locked DSPy 3.5.0 public `GeminiLM.stream` and `GeminiLM.live` methods
confirmed their different defaults over a scripted socket. The retained
native WebSocket regression failed before the fix on the missing `TEXT`
setting. It now checks the setup frames through `Gemini.streamRequest` and
`Gemini.live`, including preserved generation options, a present null
modality, and explicit image output, and receives a streamed reply.

Formatting and the full Debug and ReleaseSafe test and examples builds
passed (884 tests and 40 build steps per mode). The native replay consumed
all 687 mock requests and matched eight adapter cases and nine pipelines.
The broader offline reference gate was not rerun. Hosted inference remains blocked: neither
`GEMINI_API_KEY` nor `GOOGLE_API_KEY` was available. The next recommended
step is an authenticated Gemini Live text and tool round-trip once a key
is available.

### SDK inline media base64 validation

The canonical SDK now rejects inline base64 with only interior non-ASCII
whitespace, matching DSPy 3.5.0. Leading or trailing Python whitespace, or
an ASCII whitespace separator anywhere in the payload, still enables
whitespace removal for validation. Accepted data is not rewritten.

The retained `Request.fromFields` regression reproduced the false acceptance
before the fix. Afterward it matched seven locked DSPy `AudioPart` cases.
A standalone ReleaseSafe consumer exercised `Gemini.completeRequest` over
loopback: malformed audio was rejected before a POST, while a valid control
received the expected answer and matched DSPy's POST target and JSON body.

Formatting, the focused regression, and the full Debug test and examples
build passed (885 tests, 40 steps). The full ReleaseSafe build and offline
reference gate were not rerun. No local blocker remains for this fix.
Next, verify an authenticated Gemini Live text and tool round-trip when
`GEMINI_API_KEY` or `GOOGLE_API_KEY` is available.

### Gemini Live context-window failures

When Gemini Live reports a context-limit error after partial output,
`gemini.lm().completeResponses` now returns `error.ContextWindowExceeded`
instead of `error.LMFailure`, so callers can shorten the prompt or history
and try again. The stream sink still receives the partial text and provider
error, with no end event or completed response. Ordinary invalid requests
still return `error.LMFailure`.

Pinned DSPy 3.5.0's public Live stream and response collector confirmed
both cases over a scripted socket. The retained Zig WebSocket regression
failed before the fix and passed afterward, exercising both the raw stream
and the collector. A standalone ReleaseSafe consumer verified the exact
loopback setup and input frames, the error event sequence, the typed error,
and connection closure. The focused regression passed in Debug and
ReleaseSafe. Formatting and the full Debug test and examples build passed
(885 tests, 40 steps). The full ReleaseSafe and offline reference gates
were not rerun.

Hosted Gemini Live remains unverified because neither `GEMINI_API_KEY` nor
`GOOGLE_API_KEY` was available. The next recommended step is an authenticated
Live text and tool round-trip when a key is available.

### Gemini Live malformed model-turn parts

`Gemini.streamRequest` and `Gemini.LiveSession.recv` now return
`error.LMFailure` when `modelTurn.parts` is a truthy number or boolean.
Previously, a malformed terminal frame could look like a successfully
completed empty turn. `0`, `false`, and `null` still mean no parts.

Locked DSPy 3.5.0's public Live APIs confirmed these cases over a scripted
socket. The retained loopback WebSocket regression reproduced the false
success before the fix, then passed in Debug and ReleaseSafe. It tests
both Zig APIs, including the empty controls and a valid text reply. All
475 Debug tests rooted at `src/gemini.zig` passed. A standalone ReleaseSafe
consumer also verified rejection through `gemini.lm().completeResponses`
and `Gemini.LiveSession.recv`, with both setup and input frames checked
byte for byte.

The full repository and offline reference gates were not rerun. No hosted
inference was needed for this fix. The next recommended step remains an
authenticated Gemini Live text and tool round-trip once `GEMINI_API_KEY`
or `GOOGLE_API_KEY` is available.

### Gemini Live malformed tool-call lists

`Gemini.streamRequest` and `Gemini.LiveSession.recv` now return
`error.LMFailure` when `toolCall.functionCalls` is a truthy number or
boolean. Previously a malformed terminal frame could silently complete
without delivering the tool call. `0`, `false`, and `null` still mean no
calls; a valid list still delivers its tool call before the end event.

Locked DSPy 3.5.0 confirmed the same acceptance and event ordering through
its public Live APIs over a scripted socket. The retained loopback
regression reproduced the false success before the fix and passed after
it. A standalone ReleaseSafe consumer also verified that
`gemini.lm().completeResponses` returns `error.LMFailure` without publishing
a completed turn; the loopback peer checked the setup and input frames and
observed the connection close.

Formatting, the focused Debug regression, and the full ReleaseSafe tests
and examples passed (885 tests, 40 build steps). The offline reference gate
was not rerun. No hosted inference was needed for this fix. The next
recommended step is an authenticated Gemini Live text and tool round-trip
when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available.

### Gemini Live malformed model-turn entries

`Gemini.streamRequest` and `Gemini.LiveSession.recv` now return
`error.LMFailure` if a `modelTurn.parts` array contains a null, boolean,
or number. A malformed terminal frame no longer delivers its earlier text
as if the turn succeeded. This does not change how a missing or falsy
`parts` field is handled, and valid text followed by an empty object still
completes normally.

Locked DSPy 3.5.0 confirmed the rejection and event ordering through its
public Live APIs over a scripted socket. The retained loopback regression
reproduced the false success before the fix and passed in Debug after it.
A standalone Debug consumer confirmed that `gemini.lm().completeResponses`
returns `error.LMFailure` without publishing a delta or end event; the
loopback peer verified setup, input, and connection close.

Formatting passed. The full ReleaseSafe test and examples run timed out
before reporting a verdict, as did the standalone ReleaseSafe build; the
offline reference gate was not rerun. No hosted inference was needed for
this fix. Next, run an authenticated Gemini Live text and tool round-trip
to verify compatibility with the hosted service.

### Gemini Live malformed part keys

`Gemini.streamRequest` and `Gemini.LiveSession.recv` now return
`error.LMFailure` when a string or array inside `modelTurn.parts` matches a
recognized part key (`text`, `functionCall`, or `inlineData`) but cannot
supply the corresponding field. Previously, the decoder could skip that
entry, publish earlier deltas from the same frame, and report a
successful turn. Unrecognized string and array entries are still ignored.

Locked DSPy 3.5.0 confirmed rejection and accepted controls through its
public Live APIs over a scripted socket. The retained Zig WebSocket
regression reproduced the false success before the fix and passed after
it in Debug and ReleaseSafe. Formatting passed. An independent
ReleaseSafe consumer calling `gemini.lm().completeResponses` received
`error.LMFailure` with no delta or end event; the peer verified the exact
setup and input frames and observed the client disconnect.

The full repository and offline reference gates were not rerun. Hosted
inference remains unverified because neither `GEMINI_API_KEY` nor
`GOOGLE_API_KEY` was available. The next recommended step is an
authenticated Gemini Live text and tool round-trip once a key is available.

### Gemini refusal history replay

Buffered and SSE GenerateContent now replay an assistant refusal as plain
text, without forwarding refusal continuation metadata. Previously a
canonical request containing the refusal failed locally with
`error.LMFailure`, blocking the next turn.

Locked DSPy 3.5.0 confirmed the wire projection through `GeminiLM.complete`
and `GeminiLM.stream` over loopback HTTP. The retained Zig regression
reproduced the buffered failure and now checks the exact request body and
successful reply for both generation modes. Formatting and the focused
Debug and ReleaseSafe tests passed. A standalone public SDK consumer also
sent a refusal in the transcript over loopback and received the next reply.

The full repository and offline reference gates were not rerun. Hosted
Gemini Live verification still requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`;
the next recommended step is an authenticated text and tool round-trip.

### SDK empty system part lists

Canonical requests now reject an explicitly empty system part list with
`error.InvalidResponse` in `Request.fromFields`, `Request.fromValue`, and
`Request.fromPayload`, matching locked DSPy 3.5.0. Use `null` (or omit `system`
in a record) when there is no system prompt. A nonempty list containing a
blank text part remains valid.

The retained regression reproduced acceptance before the fix and passed
afterward across all three entry points. A standalone ReleaseSafe consumer
rejected the empty list before contacting the provider, then used
`Gemini.completeRequest` for the blank-part control. It received the same
answer as DSPy over loopback, with identical POST targets and bodies and
one request per client. Formatting and the full Debug test and examples
build passed (886 tests, 40 steps).

No local blocker remains for this fix. The full ReleaseSafe build and
offline reference gate were not rerun. The next recommended step is an
authenticated Gemini Live text and tool round-trip when `GEMINI_API_KEY`
or `GOOGLE_API_KEY` is available; neither was present in the launch
environment.

### Gemini Live tools with no declared parameters

`Gemini.live` now sends `{"type":"object","properties":{}}` as the parameter
schema when a function tool omits `parameters`. An explicitly supplied `{}`
is still sent verbatim. Canonical GenerateContent requests already apply
the default; persistent Live sessions bypass that request constructor.

Locked DSPy 3.5.0 `GeminiLM.live` confirmed both declarations in a scripted
tool exchange. The retained native loopback regression failed on the setup
frame before the fix; afterward it matched the setup and completed the
tool-call/result exchange over WebSocket, then received the answer and turn
usage. Zig formatting, the focused WebSocket regression, and the full Debug
test and examples build passed (886 tests, 40 steps). ReleaseSafe and the
offline reference gate were not rerun for this change.

The next recommended step is an authenticated Gemini Live text and tool
round-trip. Hosted verification remains blocked: neither `GEMINI_API_KEY`
nor `GOOGLE_API_KEY` was available in the project shell.

### Gemini provider-prefixed model names

Gemini generation now strips one leading `gemini:` binding prefix when it
has a nonempty model suffix. Buffered requests, GenerateContent SSE, and
request-based Live streams use the resulting model for routing and response
metadata without rewriting the caller's request. Foreign prefixes, later
colons, and `gemini:` with no suffix remain unchanged. The persistent
`Gemini.live` session now follows the same rule; see the verification below.

Locked DSPy 3.5.0 confirmed four model spellings through buffered and SSE
loopback POSTs and confirmed the Live setup model through a scripted
socket. The retained native HTTP regression failed before the fix on
`/v1beta/models/gemini:models/gemini-3-flash:generateContent`. Afterward it
matched all eight POST targets and bodies. The retained Live WebSocket test
also passed with a prefixed request. A standalone ReleaseSafe consumer
matched all three POSTs and verified the model and answer in buffered, raw
stream, and collected stream calls.

Formatting, the focused HTTP and WebSocket regressions in Debug and
ReleaseSafe, and the full Debug tests and examples passed (887 tests, 40
build steps). The full ReleaseSafe and offline reference gates were not
rerun. Hosted Gemini Live remains blocked by unavailable credentials.
Next, verify an authenticated Live text and tool round-trip when
`GEMINI_API_KEY` or `GOOGLE_API_KEY` is available.

### Gemini malformed token score telemetry

Buffered and SSE GenerateContent now return `error.LMFailure` when a
token-score collection is a truthy non-iterable scalar, or when a
`topCandidates` object is indexed as a list. Previously, those responses
could look successful while silently dropping score data. Malformed SSE
frames publish no events from that frame. Falsy collections still mean no
scores, and valid scores still reach callers.

Locked DSPy 3.5.0 confirmed six cases in both modes over loopback HTTP,
including unused malformed alternatives and a valid `-0.5` score. The
retained regression reproduced silent success before the fix and passed
in Debug and ReleaseSafe afterward. A standalone ReleaseSafe SDK consumer
verified both public APIs and four exact POST targets and bodies. Formatting
and the full Debug tests and examples passed (887 tests, 40 build steps).
The full ReleaseSafe build and broader offline reference gate were not
rerun.

No local blocker remains for this fix. No hosted inference was needed.
The next recommended step is an authenticated Gemini Live text and tool
round-trip; neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` was available in
the project shell.

### Gemini Live peer-initiated shutdown

When the server closes a persistent `Gemini.LiveSession`, DSZig echoes
the WebSocket Close once. Further reads and text sends report
`error.PeerClosed`, and `session.close()` releases resources without
sending a second frame. This follows
[RFC 6455 §5.5.1](https://www.rfc-editor.org/rfc/rfc6455#section-5.5.1).

The retained loopback regression reads a partial transcript, then checks
the masked Close echo and expects TCP EOF. Before the fix, it saw a second
Close instead. Formatting and the full Debug test and examples build passed
(887 tests, 40 steps). A standalone ReleaseSafe consumer using the public
`Gemini.live` API also passed against an independent loopback server, which
observed one exact Close echo with code 1001 and its reason, then EOF.

No hosted inference was needed. The full ReleaseSafe and offline reference
gates were not rerun; a direct DSPy Live shutdown comparison was unavailable
because the local reference environment lacks the optional `websockets`
package. The next recommended step is still an authenticated Gemini Live
text and tool round-trip once `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available.

### Gemini Live caller-initiated stream stop

When a `Gemini.streamRequest` sink returns `.stop` on a Live connection,
DSZig sends a masked WebSocket Close with normal status (1000) before
disconnecting. The call still returns `.stopped` without reading more
events or waiting for the peer to reply.

Locked DSPy 3.5.0's public `GeminiLM.stream` called the scripted socket's
`close()` exactly once when the caller closed the iterator after either
the start or the first text delta, without reading any further frames.
The retained Zig loopback regression previously saw TCP EOF before a
Close frame. It now checks both stop points for one masked Close followed
by EOF, with no terminal frame from the peer. Formatting and the focused
Debug regression passed. A standalone ReleaseSafe consumer using the public
`Gemini.streamRequest` API also passed against an independent loopback
peer that checked the setup and input frames and observed the same Close
and EOF.

The full repository and offline reference gates were not rerun. No hosted
inference was needed. The next recommended step is an authenticated Gemini
Live text and tool round-trip; neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY`
was available in the project shell.

### Gemini Live rejects malformed WebSocket Close payloads

`Gemini.LiveSession.recv()` now reports `error.InvalidFrame` for a Close
payload with an invalid status or a non-UTF-8 reason. Following
[RFC 6455 §§5.5.1, 7.1.7, and 8.1](https://www.rfc-editor.org/rfc/rfc6455#section-5.5.1),
the client sends a best-effort Close with status 1002 (protocol error) or
1007 (invalid UTF-8) and disconnects without waiting for a reply. Valid
Close payloads are still echoed once and reported as `error.PeerClosed`.

Before the fix, the retained loopback regression saw the malformed UTF-8
payload echoed and received `error.PeerClosed`. It now checks malformed
UTF-8, a reserved status, a one-byte payload, and normal and empty Close
payloads. Formatting and the full Debug test and examples build passed
(887 tests, 40 steps). An independent ReleaseSafe consumer used the public
`Gemini.live` API and confirmed the two failure replies and TCP EOF before
session cleanup, plus a normal echo of a private-use status.

No local blocker remains. The full ReleaseSafe build and offline DSPy gate
were not rerun for this transport change. The next recommended step is an
authenticated Gemini Live text and tool round-trip; the project shell had
neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY`.

### Gemini Live closes rejected streamed setups

When Gemini rejects setup on a request-based Live stream, `Gemini.streamRequest`
now sends a best-effort masked WebSocket Close (1000) before disconnecting.
It still returns `error.LMFailure` without preparing the turn, publishing
events, or waiting for a Close reply. The persistent `Gemini.live` API is
unchanged.

Locked DSPy 3.5.0's public `GeminiLM.stream` closed its scripted socket once
on setup rejection without sending a turn. The retained native loopback
regression saw TCP EOF instead of a Close before the fix. Afterward it
observed one masked Close followed by EOF. Formatting, the focused Debug
regression, and the full Debug test and examples build passed (887 tests,
40 steps). An independent ReleaseSafe SDK consumer also verified the
Close, EOF, `error.LMFailure`, and zero stream events over loopback.

The full ReleaseSafe and offline reference gates were not rerun. No hosted
inference was needed for this fix. The next recommended step is an
authenticated Gemini Live text and tool round-trip once `GEMINI_API_KEY`
or `GOOGLE_API_KEY` is available; neither was present in the project shell.

### Gemini Live closes streams after provider errors

Request-based Gemini Live streams now send a best-effort masked WebSocket
Close (1000) after an in-band provider error, even when response assembly
returns `error.LMFailure` or `error.ContextWindowExceeded`. They still expose
the error event after any partial output and do not wait for a Close reply.
Persistent `Gemini.LiveSession` behavior is unchanged.

Locked DSPy 3.5.0 `GeminiLM.stream` emitted start, partial text, and an error,
then closed its scripted socket once for both invalid-request and context-limit
errors. Before the fix, the retained Zig loopback regression saw TCP EOF
without a Close when collecting an invalid-request stream. It now checks both
error kinds through `Gemini.streamRequest` and `gemini.lm().completeResponses`,
including exactly one masked Close followed by EOF.

Formatting and the full Debug test and examples build passed (887 tests,
40 steps). A separate ReleaseSafe SDK consumer verified both error returns,
event order, and the Close followed by EOF against an independent loopback
peer. The full ReleaseSafe build and offline reference gate were not rerun.
No hosted inference was needed for this fix. The next recommended step is
an authenticated Gemini Live text and tool round-trip once `GEMINI_API_KEY`
or `GOOGLE_API_KEY` is available; neither was present in the project shell.

### Gemini Live disconnects on invalid WebSocket frames

`Gemini.LiveSession.recv()` now fails the connection on malformed WebSocket
framing or non-UTF-8 text. Following
[RFC 6455 §§7.1.7, 8.1, and 10.7](https://www.rfc-editor.org/rfc/rfc6455#section-7.1.7),
the client sends a best-effort masked Close with status 1002 for framing
errors or 1007 for invalid UTF-8, disconnects, and returns `error.InvalidFrame`.
Subsequent reads and sends report `error.PeerClosed`.

Before the fix, a session that rejected invalid UTF-8 still sent the next
application message. The retained public `Gemini.live` loopback regression
now checks invalid text, a reserved header bit, and an unexpected continuation
after partial output. The peer verifies the matching Close and TCP EOF before
explicit session cleanup. Existing valid and invalid Close cases remain covered.

Formatting, the WebSocket unit test, and the dedicated loopback test passed
in the pinned toolchain; the loopback passed in Debug and ReleaseSafe.
The full ReleaseSafe test and examples build timed out, and the broader Live
media test separately returned `error.LMFailure` in ReleaseSafe before the
protocol regression was split out. The offline DSPy gate was not run.
Next, isolate that broader ReleaseSafe Live failure. Authenticated Live text
and tool verification still requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

### Gemini Live closes when turn preparation fails

After a successful WebSocket upgrade, request-based `Gemini.streamRequest`
now attempts a Close on every exit, including when an acknowledged setup is
followed by an unreadable local media file or a malformed provider payload.
Cleanup preserves the original result, does not wait for a peer reply, and
does not send a second Close if the WebSocket layer already sent one.
Persistent `Gemini.LiveSession` behavior is unchanged.

Pinned DSPy 3.5.0's `GeminiLM.stream` closed its scripted socket once when an
acknowledged setup was followed by a missing image, without sending a turn
or yielding an event. The retained native loopback regression saw TCP EOF
before any Close on the old code. It now observes a single masked Close
(1000) followed by EOF and verifies `error.LMFailure` with zero stream
events. Existing malformed-response cases verify the same close behavior.

Formatting and the full Debug test and examples build passed (889 tests,
40 steps). An independent ReleaseSafe consumer exercised the public SDK
against a loopback peer and confirmed the Close and EOF without a peer
reply. The previously failing broader Live media test also passed a focused
ReleaseSafe run before this change; the earlier failure did not recur in that
run. The full ReleaseSafe build and offline DSPy gate were not rerun.
Next, run the full ReleaseSafe gate, then the offline DSPy gate.
Authenticated Live text and tool verification remains blocked: neither
`GEMINI_API_KEY` nor `GOOGLE_API_KEY` was present in the project shell.

### Gemini Live disconnects when a message exceeds the receive budget

`Gemini.LiveSession.recv()` still returns `error.MessageTooLarge` when a
WebSocket message exceeds the receive budget, but now sends a best-effort
masked Close with status 1009
([RFC 6455 §7.4.1](https://www.rfc-editor.org/rfc/rfc6455#section-7.4.1))
and disconnects without draining the payload. Subsequent sends and receives
report `error.PeerClosed`; callers still call `close()` to release storage.

Before the fix, the retained loopback regression showed that the session
could send another application message after reporting the size error.
The regression now checks both a single oversized frame and a fragmented
message that crosses the budget on continuation. Formatting and the focused
regression passed in Debug and ReleaseSafe. A standalone ReleaseSafe consumer
using `Gemini.live` confirmed the SDK errors and, through an independent
loopback peer, one masked Close and TCP EOF while the remaining payload
was withheld.

No local blocker remains for this fix. The full ReleaseSafe test and examples
gate is the next recommended check, followed by the offline DSPy gate;
neither was rerun in that iteration. Authenticated Gemini Live text and tool
verification still requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

### Gemini Live closes rejected persistent setups

When Gemini rejects setup after a successful WebSocket upgrade, `Gemini.live`
now attempts a masked Close with status 1000 before disconnecting, just as
the request-based streaming path does. It still returns `error.LMFailure`
without opening a session or waiting for a Close reply. Locked DSPy 3.5.0
confirmed the setup refusal through `GeminiLM.live` with a scripted socket;
the Close behavior is a native transport guarantee, guided by
[RFC 6455 §7.1.7](https://www.rfc-editor.org/rfc/rfc6455#section-7.1.7).

The retained loopback regression failed before the fix: it saw TCP EOF
instead of a Close frame. It now passes in Debug and ReleaseSafe. A
standalone ReleaseSafe consumer and independent loopback peer also observed
`error.LMFailure`, exactly one masked Close (1000), and then EOF without
sending a Close reply. Formatting passed, and the full ReleaseSafe test and
examples gate passed all 889 tests and 40 build steps.

The next recommended local check is the locked offline DSPy reference gate,
which was not rerun. Authenticated Gemini Live text and tool verification
is still blocked: neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` was available
in the project shell.

### Gemini Live releases the connection on peer Close

`Gemini.LiveSession.recv()` now echoes a valid peer Close in a masked frame
and disconnects before returning `error.PeerClosed`, as permitted by
[RFC 6455 §5.5.1](https://www.rfc-editor.org/rfc/rfc6455#section-5.5.1).
It also disconnects if writing the reply fails. Callers still call
`close()` to release session storage.

The retained `Gemini.live` loopback regression now waits for TCP EOF
before calling `close()`. Previously that wait timed out even though
the peer had received the Close reply. Both an empty Close and a Close
with a status and reason now pass in Debug and ReleaseSafe. Formatting
and all 479 Debug tests rooted at `src/gemini.zig` passed.

No local blocker remains for this fix. The full repository build and
locked offline DSPy reference gate were not rerun; the offline gate is
the next recommended local check. Authenticated Live text and tool
verification still requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`;
neither was available in the project shell.

### Gemini Live releases the connection on transport EOF

If the peer disconnects without a WebSocket Close frame,
`Gemini.LiveSession.recv()` preserves the initial `error.EndOfStream`
and immediately releases the connection. Further receives and text sends
return `error.PeerClosed`; callers still call `close()` to free session
storage. This follows [RFC 6455 §7.2.1](https://www.rfc-editor.org/rfc/rfc6455#section-7.2.1)
for abnormal transport closure.

The retained loopback regression reproduced a successful text send after
EOF before the fix. It now passes in Debug and ReleaseSafe for EOF both
between frames and inside a frame payload. An independent public
`Gemini.live` consumer received partial text and then the expected errors
in both cases. The peer saw TCP EOF with no intervening WebSocket bytes
before the consumer called `close()`. Formatting and all 479 Debug tests
rooted at `src/gemini.zig` passed.

No local blocker remains for the native fix. The locked DSPy Live probe
could not run because its reference environment lacks `websockets`.
Hosted Gemini Live was not exercised; it requires Gemini credentials.
Next local check: `devenv shell -- tools/dspy_reference/check.sh`.
The full repository build and offline reference gate were not rerun.

### Gemini Live disconnects after a WebSocket read timeout

`Gemini.LiveSession.recv()` preserves the original WebSocket receive error
and disconnects rather than trying to resume a partially consumed frame.
A read timeout still surfaces as `error.ReadFailed`; subsequent sends and
receives return `error.PeerClosed`. Call `close()` to release session storage.
This is native transport behavior, following the fail-connection guidance in
[RFC 6455 §7.1.7](https://www.rfc-editor.org/rfc/rfc6455#section-7.1.7).

The retained public API regression withholds the rest of a frame without
closing the peer socket. Before the fix, a text send succeeded after the
timeout and the peer received a new frame. After the fix, the peer saw TCP
EOF before application cleanup, with no additional WebSocket bytes.
Formatting, all 479 Debug tests rooted at `src/gemini.zig`, the targeted
ReleaseSafe regression, and an independent ReleaseSafe `Gemini.live` smoke
against a loopback peer passed.

No local blocker remains for this transport fix. The full repository and
offline reference gates were not rerun. Authenticated Gemini Live still
requires provider credentials and was not exercised. Next local check:
`devenv shell -- tools/dspy_reference/check.sh`.

### Gemini Live disconnects after a WebSocket write failure

`Gemini.LiveSession.send()` preserves the original transport write error and
disconnects. A failed write may already have sent part of a frame, so the
client sends neither another message nor a Close frame on that connection
([RFC 6455 §§7.1.7 and 7.2.1](https://www.rfc-editor.org/rfc/rfc6455#section-7.1.7)).
Subsequent text sends and receives report `error.PeerClosed`, even if there
was a queued event. Call `close()` to release session storage. Errors while
preparing local input do not close the connection.

The retained public API regression forces a write deadline after setup.
Before the fix, the next send succeeded and the peer received the first
byte of a WebSocket frame. Afterward, the peer saw TCP EOF before application
cleanup, with no additional frame bytes. Formatting, the focused ReleaseSafe
regression, and the full Debug test and examples build (890 tests, 40 steps)
passed. An independent ReleaseSafe consumer also observed `error.WriteFailed`
followed by `error.PeerClosed` on receive and send after a real loopback TCP
reset.

### Offline replay of saved DSPy 3.5.0 recordings

Both saved OpenAI recordings passed through the native `zig build replay`
CLI and locked DSPy 3.5.0's `check_live.py`. Each runtime consumed all 679
DSPy-recorded requests and all 680 DSZig-recorded requests. For each
recording, eight adapter cases and nine pipelines matched, including
scores, optimizer trials, and saved artifacts. No runtime code or
recordings needed to change.

This closes the pending recording comparison without repeating the
completed fixture and deterministic mock checks. The full ReleaseSafe
build and offline reference gate were not rerun, and no new hosted
inference was performed. Next, verify an authenticated Gemini Live text
and tool round-trip once `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available;
neither was configured in the project shell.

### Gemini Live rejects ambiguous WebSocket upgrades

Both Gemini Live APIs now require a single `Upgrade` response header whose
value is `websocket`, ignoring case, as specified by
[RFC 6455 §4.1](https://www.rfc-editor.org/rfc/rfc6455#section-4.1).
The `Connection` header still accepts a list containing the `Upgrade`
token. Invalid upgrades disconnect before the client sends Gemini setup.

Before the fix, the loopback peer saw the first setup-frame byte after
returning `Upgrade: h2c, websocket`. The retained regression now checks that
both `Gemini.live` and `Gemini.streamRequest` send no frame for a protocol
list or duplicate `Upgrade` headers. A standalone ReleaseSafe consumer
confirmed both rejections and completed a text turn using a valid mixed-case
`Upgrade` and a multi-token `Connection` header. Formatting and the full
Debug test and examples build passed (891 tests, 40 steps).

The full ReleaseSafe build and offline DSPy gate were not rerun. Hosted
Gemini Live remains unverified: neither Gemini key was configured in the
project shell. Next, verify an authenticated Live text and tool round-trip
when a key is available.

### Gemini Live accepts informational responses before the upgrade

Both `Gemini.live` and `Gemini.streamRequest` now skip up to 16 interim HTTP
1xx responses before validating the WebSocket upgrade. An interim response
such as `103 Early Hints` no longer prevents a valid `101 Switching Protocols`
from opening the connection. A non-101 final response still fails the
handshake. This follows [RFC 9110 §15.2](https://www.rfc-editor.org/rfc/rfc9110#section-15.2)
and preserves the upgrade checks in [RFC 6455 §4.1](https://www.rfc-editor.org/rfc/rfc6455#section-4.1).

The retained loopback regression failed with `error.InvalidUpgrade` before
the fix. It now completes a text turn through both public APIs after a 103
and a 100. A standalone ReleaseSafe consumer and independent loopback peer
confirmed a successful turn after 16 interim responses, plus rejection
without sending setup after 17 or after a final 403. Formatting and the
full Debug test and examples build passed (892 tests, 40 steps).

No local blocker remains. The full ReleaseSafe build and the offline DSPy
gate were not rerun. The next recommended local check is the full
ReleaseSafe test and examples build. An authenticated Gemini Live text and
tool round-trip remains blocked until `GEMINI_API_KEY` or `GOOGLE_API_KEY`
is available.

### Gemini Live requires HTTP/1.1 for WebSocket upgrades

`Gemini.live` and `Gemini.streamRequest` now reject an `HTTP/1.0 101`
response before sending Gemini setup, even when the upgrade headers and
accept key are valid. WebSocket Upgrade uses HTTP/1.1
([RFC 9110 §7.8](https://www.rfc-editor.org/rfc/rfc9110#section-7.8),
[RFC 6455 §4.2.2](https://www.rfc-editor.org/rfc/rfc6455#section-4.2.2)).

The retained loopback regression observed the first setup byte before the
fix and now verifies disconnect without a frame through both public APIs.
A standalone ReleaseSafe client independently confirmed rejection with
peer EOF and completed a text turn over HTTP/1.1. Formatting, the
focused Debug regression, and the full ReleaseSafe test and examples
build passed (892 tests, 40 steps).

No local blocker remains. The locked offline DSPy gate was not rerun and
is the next recommended local check. Authenticated Gemini Live text and
tool verification still requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`;
neither was available in the project shell.

### Gemini Live closes upgraded connections on local setup failure

If `Gemini.live` cannot prepare setup after a successful WebSocket upgrade,
it now attempts one masked Close with status 1000 before disconnecting. It
preserves the original error and does not wait for a Close reply. This
uses the Close frame defined in [RFC 6455 §7.1.2](https://www.rfc-editor.org/rfc/rfc6455#section-7.1.2),
consistent with the cleanup already used by `Gemini.streamRequest`.

The retained loopback regression calls the public API with invalid setup
extensions. Before the fix the peer saw EOF without a Close; afterward it
saw a masked Close followed by EOF, with no setup frame, while the caller
still received `error.LMFailure`. Formatting, the focused Debug regression
and all nine tests in the focused ReleaseSafe run passed.

The locked offline DSPy gate passed before the fix: fixture regeneration
was unchanged, DSPy round-tripped the native artifact, and both runtimes
replayed the 687-request mock transcript and the 679- and 680-request saved
recordings. The full post-fix ReleaseSafe test and examples build passed
892 of 893 tests; the `typed Predict store follows provider policy over HTTP
and distinguishes cache and replay requests` test failed without output.
Isolate that failure next before treating the full gate as green.
Authenticated Gemini Live text and tool verification remains blocked:
neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` was available.

### Gemini Live rejects HTTP body framing during WebSocket upgrade

`Gemini.live` and `Gemini.streamRequest` now reject `Content-Length` and
`Transfer-Encoding` on informational HTTP responses, including the `101`
upgrade. [RFC 9110 §8.6](https://www.rfc-editor.org/rfc/rfc9110#section-8.6)
and [RFC 9112 §6.1](https://www.rfc-editor.org/rfc/rfc9112#section-6.1)
forbid these headers on all 1xx responses. Valid interim responses can
still precede an upgrade.

Before the fix, the retained loopback regression saw the client start
sending its setup frame after a `101` with `Content-Length: 0`. The
regression now checks both headers on interim and final responses and
verifies that the peer receives EOF before any setup frame. Formatting
and the focused Debug and ReleaseSafe tests passed, including the valid
interim-response control. A standalone ReleaseSafe consumer confirmed
`InvalidUpgrade` for sessions and `LMFailure` with no events for streams.

No local blocker remains for this fix. The full build and offline reference
gate were not rerun in that iteration. Hosted Gemini Live verification still
needs credentials.

### ReleaseSafe predictor-store verification

The previously failing typed `Predict.call` store regression passed 200
isolated ReleaseSafe runs. The full ReleaseSafe test and examples build
also passed (893 tests, 40 steps). The historical failure did not recur,
so its cause remains unknown and provider behavior was not changed. The
expected store wire fields and Anthropic adaptations were checked against
the locked DSPy 3.5.0 provider source.

The retained regression now identifies the provider, store setting, and
whether a failure occurred during loopback, cache lookup, or replay. A
disposable wrong-wire assertion failed as expected and printed
`store loopback failed (chat, disabled)`. After the diagnostic change,
formatting passed and the ReleaseSafe gate succeeded again, rerunning all
98 behavior tests and reusing the other successful test results.

Next, run the locked offline DSPy gate against the current tree; it was not
rerun here. Authenticated Gemini Live text and tool verification remains
blocked until `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available.

### Persistent Gemini Live model bindings

`Gemini.live` now strips one leading `gemini:` binding prefix before
encoding its WebSocket setup, just like request-based Gemini calls. An
application can reuse `gemini:models/gemini-2.5-flash` without sending the
invalid resource `models/gemini:models/gemini-2.5-flash` to the provider.

The retained WebSocket regression reproduced that exact setup mismatch
before the fix and passed afterward in Debug and ReleaseSafe. A standalone
ReleaseSafe consumer used the public session API over loopback. Its setup
and text turn matched locked DSPy 3.5.0 on a scripted socket using the bare
model resource. The native session received text and `turn_end`, then
sent a normal Close. Formatting and the full Debug test and examples
build passed (893 tests, 40 steps).

The locked offline DSPy gate passed on the starting tree, including
unchanged fixture regeneration and replay of the mock suite and both
saved DSPy 3.5.0 recordings. It was not rerun after this session-only
change. The full ReleaseSafe build was not rerun either; the focused
regression and standalone smoke passed in that mode. Next, verify an
authenticated Gemini Live text and tool round-trip when `GEMINI_API_KEY`
or `GOOGLE_API_KEY` is available. No hosted inference was run here.

### Gemini Live rejects sends before preparing input after disconnect

`Gemini.LiveSession.send()` now returns `error.PeerClosed` as soon as it
detects a disconnected transport, before encoding the turn or reading a
local media file. On a connected session, a local input error still leaves
the connection usable. After the application calls `close()`, sends
continue to return `error.SessionClosed`. The locked DSPy 3.5.0 session
also rejects sends on an explicitly closed session before encoding input;
DSZig separately reports a transport disconnect, consistent with
[RFC 6455 §7.1.4](https://www.rfc-editor.org/rfc/rfc6455#section-7.1.4).

The retained loopback regression reproduced `error.LMFailure` instead of
`error.PeerClosed` for a media turn after disconnect. It now checks both
outcomes and confirms that the peer receives no extra frame. Formatting,
the focused ReleaseSafe regression, and the full Debug test and examples
build passed (893 tests, 40 steps). A standalone ReleaseSafe consumer
also completed a text exchange between a missing-file error and a peer
Close, then confirmed `PeerClosed` before cleanup and `SessionClosed`
afterward. Its peer observed one masked Close followed by EOF.

No local blocker remains for this fix. The full ReleaseSafe build and
offline DSPy gate were not rerun. Next, verify an authenticated Gemini
Live text and tool round-trip when `GEMINI_API_KEY` or `GOOGLE_API_KEY`
is available; neither was configured in the project shell.

### Gemini Live Unicode text frames

`Gemini.streamRequest` and `Gemini.live` now ASCII-escape Unicode in their
WebSocket JSON, matching locked DSPy 3.5.0. Ordinary text decodes unchanged,
including supplementary characters. Python-style lone surrogates are
preserved as JSON escapes rather than emitted as invalid UTF-8 bytes.

The retained loopback regression reproduced the setup-frame mismatch
before the fix. It now verifies the setup and turn frames and completes
a text exchange through both public APIs. Formatting, the full Debug test
and examples build (894 tests, 40 steps), and the focused ReleaseSafe run
passed. An independent ReleaseSafe session smoke also verified that the
peer decoded both frames as UTF-8, recovered the original strings, and
observed a normal Close and EOF.

No local blocker remains. The full ReleaseSafe build and offline DSPy gate
were not rerun. Next, verify an authenticated Gemini Live text and tool
round-trip when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available; neither
was configured in the project shell. No hosted inference was used here.

### Gemini Live ignores binary messages that are not UTF-8

`Gemini.live` and `Gemini.streamRequest` now skip binary WebSocket messages
that are not valid UTF-8 before interpreting them as JSON. Previously, a
raw surrogate encoded in a binary message could reach the caller as text
or be treated as a setup error. Valid binary JSON still works, including a
Unicode character split across WebSocket fragments. The WebSocket protocol
handling for invalid text messages is unchanged.

Locked DSPy 3.5.0 skipped the malformed messages during setup and the turn
through both public APIs on a scripted socket. The retained native
loopback regression reproduced the stray text before the fix and now
checks that both APIs complete the turn with only the valid reply.
Formatting, the focused ReleaseSafe run (11 tests), and the full Debug
test and examples build (895 tests, 40 steps) passed. A standalone
ReleaseSafe session also completed a loopback turn; the peer observed a
normal Close and TCP EOF.

No local blocker remains. The full ReleaseSafe build and offline DSPy gate
were not rerun. The next recommended step is an authenticated Gemini Live
text and tool round-trip once `GEMINI_API_KEY` or `GOOGLE_API_KEY` is
available; neither was configured in the project shell.

### Gemini streamed provider error classification

Gemini Live and GenerateContent SSE now classify in-band errors using the
provider's actual message. If the message is missing, the provider code is
still displayed, but text such as `Token limit exceeded` in that code alone
no longer causes `LM.completeResponses` to return
`error.ContextWindowExceeded`.

Locked DSPy 3.5.0's public `GeminiLM.stream` confirmed the distinction on a
scripted socket. The retained WebSocket regression failed before the fix
and now checks both `Gemini.streamRequest` and `LM.completeResponses`,
including a real context-limit message as a positive control. Formatting
and the full ReleaseSafe test and examples build passed (895 tests, 40
steps). A standalone ReleaseSafe consumer also verified the corrected
error event over loopback GenerateContent SSE.

No local blocker remains for this fix. The full offline DSPy gate was not
rerun. The next recommended step is an authenticated Gemini Live text and
tool round-trip; neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` was available
in the project shell.

### Gemini Live object-shaped output parts

When a gateway sends `modelTurn.parts` as an object such as `{"text":"lost"}`
instead of a parts array, Gemini Live now returns `error.LMFailure` rather than
silently discarding the text and reporting a successful empty turn. Objects
with only unrecognized keys are still ignored, matching locked DSPy 3.5.0.

The locked reference confirmed both cases through public `GeminiLM.stream`
and `GeminiLM.live` calls on a scripted socket. The native loopback regression
reproduced the false success before the fix and passed afterward through
`Gemini.streamRequest`, `LM.completeResponses`, and `Gemini.LiveSession.recv`.
Formatting, the focused ReleaseSafe test, and the full Debug test and examples
build (895 tests, 40 steps) passed. The full ReleaseSafe build and offline DSPy
gate were not rerun.

No local blocker remains. The next recommended step is an authenticated Gemini
Live text and tool round-trip; neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY`
was available in the project shell.

### Gemini streamed judgment answers

`gemini.lm().completeResponses` now exposes a single JSON-object answer as
a canonical data part when the response schema contains a judgment, matching
buffered Gemini responses. Text callbacks still receive the original deltas,
and a schema without judgments leaves the assembled answer as text.

Locked DSPy 3.5.0 confirmed both cases by materializing a public
`GeminiLM.stream` over a scripted Live socket. The retained native regression
uses a loopback WebSocket and splits the answer across two frames. It failed
before the fix because the judgment came back as text, then passed in
ReleaseSafe. Formatting and the full ReleaseSafe test and examples build
passed (896 tests, 40 steps). The offline DSPy gate was not rerun.

No hosted calls were needed. The next recommended step remains an
authenticated Gemini Live text and tool round-trip; neither `GEMINI_API_KEY`
nor `GOOGLE_API_KEY` was available in the project shell.

### Gemini Live duplicate tool names

Persistent `Gemini.live` now returns `error.InvalidResponse` for duplicate
tool names before opening a connection, matching DSPy 3.5.0 `LiveConfig`.
The loopback regression saw duplicate declarations reach the peer before the
fix. It now verifies that the peer's first connection carries distinct
declarations and completes a text and tool round-trip.

The focused ReleaseSafe checks passed (12 tests), as did the full ReleaseSafe
test and examples build (896 tests, 40 steps) and the locked offline DSPy gate.
A fresh Zig project also fetched the versioned package from a local release
archive and ran an exact-match metric through the exported `dszig` module.
No hosted requests were made. Authenticated Gemini Live remains unverified
and is not a release requirement. GitHub-hosted jobs did not start because
of an account billing or spending-limit issue. Clear that issue and run the
Linux and macOS matrix before merging; the local pass does not replace it.

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
  On a capable LM, scalar `reasoning_effort = "none"` or an explicitly
  present null uses that fallback; `"off"` keeps the native field out of the
  prompt while turning off provider reasoning.
  Prior rationale in `History` uses that same wire name without mutating
  the caller's history.
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
  Zero and negative iteration limits skip ordinary tool turns and go straight
  to forced submission. Generated tool-call IDs retain the signed limit.
  The implicit XML adapter's native-History policy is distinct from whether a
  particular LM supports native function declarations. Set
  `Context.use_native_function_calling` to override the policy.
  If a response contains native tool calls, `Predict` restores them into the
  first declared `ToolCalls` output even when the request did not declare
  tools. Negotiating native tools removes only that first output from the
  text signature; any additional `ToolCalls` output stays a text field.
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
| Runtime values and documents | `Value`, `Array`, `ObjectMap`, boxed native Code/File/Document/Reasoning/BoundingBox/Citations/decision/evidence identity, recursive typed-class serialization, text/canonical-document content union and bounded Responses `input_file` translation |
| Chat, JSON, XML and two-stage adapters | `Adapter.chat`, `.json`, `.xml`, `.two_step`; `TwoStepAdapter` with a separate extraction LM; independent prompt/parse fixtures, strict response-schema construction, JSON fallback, typed XML, top-level XML output union retries and Unicode key handling |
| Prediction and reasoning | `Predict`, `ChainOfThought`, `BestOfN`, `Refine`, `MultiChainComparison`, explicit `Context`, per-predictor LM bindings and module LM aggregation, typed outputs, serialized `Trace`, named predictor/parameter discovery |
| Parallel composition | Startup-created `Parallel(Result, limits)` workers, typed program/Example/tuple bindings, ordered result slots, speculative straggler retries, bounded accumulated failure reports and joined cooperative cancellation |
| Typed decisions | `Noul`, configured `Score`/`Choice`, generative evidence classes, criteria/state snapshots, native Annotated metadata, owning evidence collectors and TypeSafe SystemOne prediction; finite coverage and unsupported rows below |
| Retrieval and embeddings | `Retrieve`, lexical `InMemoryRetriever`, HTTP `ColBERTv2` with shared mutable-graph memory caching, custom/OpenAI/Azure `Embedder`, float32 `KNN`, per-query `KNNFewShot`, and brute-force `Embeddings`/`EmbeddingsWithScores` with NumPy persistence |
| Evaluation and few-shot search | `optimize.evaluate`, concurrent `parallel.Evaluator`, `bootstrapTraceData` with real parse-failure capture and ordered native rollouts, `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `BootstrapFewShotWithOptuna`, teacher context, zero thresholds and positional multi-use trace selection |
| Fine-tuning | Chat/XML `Adapter.formatFinetuneData`; native `BootstrapFinetune` over bounded OpenAI HTTP jobs, startup workers, retained model names and joined failure cleanup |
| Optimizer composition | Native `BetterTogether`, real default prompt/weight optimization, typed custom registries, retained candidate LM snapshots and joined lifecycle cleanup |
| Instruction optimization | `InferRules`; `MIPROv2`, its program-aware proposer and an Optuna 4.5.0 TPE port; `COPRO`; `SIMBA`; native `GEPA` with instance and named-objective frontiers |
| Decision calibration | `ReAnchor` fits Noul thresholds, Score cuts, and Choice weights against the metric; rejects gains that fail the fold check |
| Agents and tools | `ReAct` (also available under DSPy's deprecated `ReActV2` name), bounded native `Avatar` and `AvatarOptimizer`, comptime `Tool`, `History`, `ToolCalls`, tool results, native declarations and text tool calls, typed submit, retained history |
| Generated native execution | `generated_zig.Generator` (`Predict(Code[zig])`) and fixed-storage Linux `Executor`: restricted compiler, kernel-confined guest, exact native capability RPC, pending output and whole-tree join; not Python RLM/Flex |
| Voting | `Ensemble`, `EnsembleWithReducer`, `ensemble.majority`, `ensemble.majorityWithOptions` with inferred field and custom or disabled normalization; Python numeric equality and source-order fallback |
| Answer metrics | `metrics.emScore`, `f1Score`, `hotpotF1Score`, `precisionScore`, multi-reference `em`/`f1`/`hotpotF1`, thresholded and record-based answer matching, DPR-token passage matching |
| LM boundary | `LM`, native `lm.OpenAI` Chat Completions/Responses and `lm.TypeSafe` SystemOne, shared bounded `http.Transport`, scripted/keyword LMs, recording/replay wrappers and typed native protocol metadata |
| Cache | `CachedLM`, whole public candidate-group/request identity, LRU eviction and rich-response JSON dump/load; engine-group failures are not cached |
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

### Retries for custom canonical engines

A synchronous custom SDK engine can opt in with
`LM.VTable.canonical_retry.max_retries` (additional attempts, default zero).
Return `error.RetryableLMFailure` only when replaying that attempt is safe.
The optional `backoff` callback runs before each retry, receives the same
candidate request and a one-based retry number, and can wait using the
engine's I/O or propagate cancellation. Without a callback, retries are
immediate.

Only the failing candidate is retried. Completed candidates are not replayed,
and a failed group is not cached. Other engine errors, adapter calls, streams,
and post-generation failures are not retried by this policy. Built-in HTTP
providers keep their existing transport retries; the custom policy does not
add another retry layer. This is an explicit native policy, not DSPy's default
retry count, error taxonomy, or delay selection.

### LM text and non-text results

`lm.Completion.text` is `?[]const u8`: `null` means that no text part was
returned, not an empty answer. Use `LM.completeResponses` to retain that
distinction together with reasoning, tool calls, continuation and raw provider
data. Recording/replay entries and completion JSON metadata retain nullable
text. The text-only `complete`, `completeMany` and `OpenAI.parseResponse`
helpers return `LMFailure` for an absent text result rather than inventing `""`.
Chat's no-tool empty/null/missing content fallback is separately normalized to
`""`, as in the source; a Chat tool-only result retains `null`.

### Response metadata and exact token usage

`lm.Completion.metadata` retains canonical `id`, `model`, `finish_reason` and
`usage` independently of raw `provider_data`. Legacy native engines can leave
these facts unknown. Chat Completions, Responses and TypeSafe read their real
response fields; model fallback uses the actual wire request, and TypeSafe's
response ID comes from `x-typesafe-request-id`, not its body. The five canonical
finish reasons are `stop`, `length`, `tool_call`, `content_filter` and `error`.

`lm.Usage` contains eight optional exact counters. Unknown is `.null`, not zero.
Use its checked constructor before passing hand-built counts to an engine:

```zig
const usage = try dszig.lm.Usage.init(arena, .{
    .input_tokens = .{ .integer = 13 },
    .output_tokens = .{ .integer = 7 },
});
// usage.total_tokens is the exact integer 20.
```

Counters retain integers beyond machine-word precision; finite integral builtin
floats become exact integers. Bool, strings, fractional/nonfinite floats,
ordinary Enum and Decimal counters refuse. Genuine IntEnum counters retain
their identity. Construction enforces nonnegative counts, part/whole relations
and explicit total consistency, inferring a total only when both addends are
known. Provider ingestion separately follows the SDK's settlement order:
inconsistent wire wholes may become unknown before constructing canonical
usage; raw provider counts remain unchanged in `provider_data`.

`Usage.fromValue` reads the eight SDK names and ignores unknown dictionary
keys. `toValue` produces the sparse SDK dictionary; `dspyValue` produces
prompt/completion totals and nested detail dictionaries, preserving reported
zero and unknown details. `init`/`fromValue` borrow input number text and boxes;
converted wide floats and computed wide totals use the supplied arena.
`clone`/`free` manage independent owned snapshots, and both projections own
their keys and values. A native object dictionary is required; arbitrary Python
dictionary objects and runtime-created numeric subclasses are not modeled.

Completion cloning, scripted responses, recording/replay, cache persistence and
native live transcripts retain this metadata. Scripted metadata is supplied
separately from `provider_data`, never inferred from fixture expectations.
Explicit SDK-style choice selection retains the response-wide usage on each
choice; the normal LM boundary refuses an unexpected multi-choice response.

Independent actual Source probes cover 276 Usage constructor/serde inputs and
30 Chat response inputs. Native Debug and ReleaseSafe each matched 1,595
targeted Usage observations and 165 Chat observations without mismatches.
The integrated checkpoint passed **551/551 tests and 38/38 build steps in each
mode**, its actual RAG/bootstrap/8,192-run Parallel examples, and all 687 existing
native replay requests. An isolated wrong-field metadata encoder produced 27
mismatches and a nonzero independent comparison exit with unchanged Source bytes.
One unknown-keyword constructor row remains unobserved at runtime because the
native constructor is statically typed; Python exception classes/messages and
the Usage probe's complete Response objects were not compared. The Chat
comparison exercised actual localhost HTTP; native live Chat/Responses CoT
smokes returned `4`, and a TypeSafe HTTP smoke retained its header, wire-model
fallback and exact counts through recording, replay and cache dump/load after
the original response arena was destroyed. This is finite protocol evidence,
not authenticated inference.

LM calls can account canonical response usage into an explicit native tracker;
Module calls can also create a scope and attach its totals to their returned
`Prediction`. The collector and automatic-call contracts below are separate
from costs/pricing, a complete Python Response/Prediction object model,
asynchronous execution and streaming.

For Chat servers such as xAI, set `OpenAI.chat_reasoning_usage = .separate`
when reasoning tokens are reported outside `completion_tokens`; the default
`.in_completion` keeps the inclusive counter unchanged. Separate accounting
adds exact integers before SDK settlement. If reasoning is absent, output stays
unknown unless the reported total equals prompt plus completion. Responses
continues to use its own inclusive output counter. The policy survives native
LM state save/restore and model copies.

A fresh 20-case actual Source HTTP corpus exposed nine native response
mismatches before this policy was integrated. Debug and ReleaseSafe now each
match all 16 complete SDK responses, four acceptance/refusal outcomes and all
20 ordered HTTP path/auth/body records through both `completeRequest` and the
native LM caller. Canonical SDK request snapshots and both routes' credential
call counts also match. Cases include sums beyond 64 bits, missing counters, inconsistent totals,
negative/integral-float counts and malformed reasoning counts. These are local
protocol proofs for the bound Chat usage policy, not authenticated homelab
inference. The current Debug and ReleaseSafe gates each pass all 38 build steps
and 652 tests, alongside the 687-request replay and actual examples.

`UsageTracker` records arbitrary provider fields in `usage_tracker.Graph`
values, separately from SDK `lm.Usage` validation. Initialize its fixed model
and entry capacities against shared, stable-address `usage_tracker.Storage`:

```zig
var storage: dszig.usage_tracker.Storage = .{ .arena = arena, .io = io };
var tracker: dszig.UsageTracker = undefined;
try tracker.init(&storage, .{ .models = 16, .entries = 1024 });
try tracker.addUsage("configured/model", try dszig.usage_tracker.Graph.parse(
    arena, "{\\\"prompt_tokens\\\":13,\\\"is_byok\\\":true}",
));
const totals = try tracker.getTotalTokens(arena);
```

Entries copy their root dictionary, flatten only direct genuine BaseModels
with default Python-mode `model_dump`, and borrow ordinary nested graph values.
Totals recompute in entry order: integer/float counts sum exactly as Python
does, booleans never sum, and first nonnumeric values remain unchanged. Nested
dictionary merging preserves Source's asymmetric iterable-pair conversion and
refusal ordering. Neither recorded entries nor totals are deep snapshots;
`Graph.clone` makes an owning arena copy when needed.

`UsageScope.init` takes the same storage and an explicit context-local
`?*UsageTracker` slot. Its `run(body, args)` rolls up on success and error,
restoring the prior slot even when rollup fails; `finish` supports explicit
lifetime management. Rollup copies entry references once, grouped by model,
so later child additions do not reach the parent. Related scopes share one
allocator/collector lock. Their graph inputs still require external
synchronization; serializer callbacks must not reenter collection. Storage,
input leaves and output arenas must remain alive for their borrowed users.
This is an explicit native context, not Python settings or garbage collection.

Model/entry capacities do not grow. Merge uses a fixed stack of at most 1,000
child merges and a 1,048,576-step budget shared across a totals snapshot.
Capacity refusal is explicit; child rollup preflights capacity without partial
publication. These are native safety/resource contracts, not Python limits or
Python MemoryError transition equivalence.

Both Debug and ReleaseSafe native CLIs consumed the unchanged **635-row**
actual Source tracker corpus with **zero mismatches or unconsumed rows**:
525 accepted results, 108 refusals and two scope-only rows. Comparisons include
ordered fields, genuine scalar/container kinds, direct model snapshots, alias
observations and scope restoration; Python exception classes/messages are not
compared. Four independently corrupted result/alias/scope outputs each failed
the comparator. A separate real four-thread smoke retained all 128 entries,
observed ten consistent snapshots and joined every worker. Those collector
proofs do not establish all Python models/protocols, authenticated inference or
whole-runtime static allocation.

### Canonical token logprobs

Set `lm.Options.logprobs` to a nonnegative integer `Value`: `.null` leaves
scoring off, `.{ .integer = 0 }` requests chosen-token scores, and positive
values request alternatives. Integral builtin floats normalize to integers;
booleans, negative counts and fractional counts refuse before engine execution.
Provider caps are not imposed locally. Chat sends `logprobs: true`, adding
`top_logprobs` only for positive counts; Responses sends `top_logprobs` and
`include: ["message.output_text.logprobs"]`.

`lm.Completion.logprobs` is an optional sequence of native `TokenLogprob`
records. Unknown and reported-empty sequences remain distinct. Records retain
the token, binary64 logprob, optional byte sequence, optional vocabulary ID and
ordered `top` alternatives. SDK byte entries are nonnegative integer `Value`s,
not narrowed to octets: the Source permits values greater than 255 and 64 bits.
Vocabulary IDs may be signed. The SDK permits nonfinite scores; the native
implementation does not clamp, sort or invent them. `logprobs_complete` retains
the provider/editor completeness fact separately from score presence.

Both OpenAI dialects use the shared Source-backed entry decoder. Responses
concatenates scores in text-block order; Chat refusal-token scores remain only
in raw provider data. Chosen and alternative tokens use Python string/float
coercion at the wire boundary, while canonical SDK record validation remains
strict. Shared Python float-text parsing handles Unicode decimal digits,
separators, signed zero and nonfinite spellings; ordinary ASCII decimal text
allocates no conversion scratch. JSON repair reuses that parser rather than
maintaining a second decimal grammar.

Owning cloning, `ScriptedLM`, recording/replay, both transcript writers and
in-memory/persisted `CachedLM` retain scores and completeness. Native JSON
persistence uses a reserved `$dszig_f64` bit tag **only for nonfinite scores**;
the canonical `lm.logprobs.toValue` view still returns float values. The native
persistence reader alone decodes that tag. This is not Python diskcache or
canonical SDK JSON-format interoperability.

The new independent Source corpus has 51 canonical observations; Debug and
ReleaseSafe compared **48** with zero mismatches, including genuine native
IntEnum/Enum inputs. Three Python constructor/child-alias observations remain
unconsumed, not passed. A separate **64-row** actual provider/configuration
corpus exposed 38 native mismatches before score integration, then matched all
42 response rows and 22 request rows in both modes. Actual cache dump/load
additionally exposed four nonfinite-score failures before the lossless native
encoding repaired them. A supplemental **24-row** actual Source probe covered
previously unobserved float-text boundaries, including signed NaN, enormous
exponents and a 5,000-digit string mantissa. Both modes matched all rows and
the 14 accepted scores' exact binary64 bits; 12 ASCII cases also converted
with a zero-byte allocator. Independently corrupted score and request-knob
expectations each produced real nonzero comparator exits in both modes.
These are finite observations, not full Python dataclass identity, arbitrary
iterable or streaming-materialization parity.

Actual localhost HTTP consumers in Debug and ReleaseSafe made eight
score-configured requests with Source-matching request knobs and returned
canonical scores. Both real `zig build live -- --smoke` dialects also returned
answer `4`; the fixture served ten total requests and its worker joined.
This is real HTTP/codec/CLI evidence, **not authenticated homelab inference**.
At the token-logprob checkpoint, native gates passed **596/596 tests and
38/38 build steps per mode**.

### Ordered SDK artifacts and public call results

`dszig.sdk_response` exposes a stable, reference-counted `Response` object,
its typed `ResponseFields`, `Message` and twelve-variant `Part` records.
`Response.fromFields(allocator, &fields)` copies immutable field storage,
retains already-owned JSON views and snapshots borrowed JSON views.
`fromValue`/`fromNativeValue` construct owning objects from wire projections.
The returned `*const Response` exposes its immutable header through `.fields`;
`retain` preserves object identity, `release` drops a reference, and `clone`
makes an independent deep copy. The object's allocator must outlive all its
references. Raw owning field records retain their `clone`/`free` methods.

Tool-result content has its own presentational union, so nested tool
calls/results, thinking and refusals cannot inhabit it. Owning readers validate
roles, nonempty identifiers, JSON payloads, media addresses, probability
distributions and adaptations. Partial constructors unwind on allocation
failure. `toValue` is an arena projection; graph-backed JSON payloads are
materialized into that arena. Native JSON persistence additionally retains
empty score presence and nonfinite score bits.

`DataPart.value`, `DataPart.probabilities`, `ResponseFields.provider_data` and
`Completion.provider_data` use `sdk_response.JsonValue`. Its `root` is the
existing `json_graph.Value`, not a flat container-header snapshot.
`JsonValue.borrowed(root)` explicitly borrows a graph until an enclosing SDK
clone; `fromValue(allocator, wire_value)` owns a converted wire value.
`fromBorrowedValue(arena, wire_value)` imports parsed wire containers into the
call arena and borrows their scalar leaves. Neither wire conversion can recover
identity already erased by flat JSON headers. Construct shared graph nodes
directly when aliases matter.

SDK response and completion clones share one graph identity memo across data,
probabilities and provider metadata. Their owned graph payloads use the same
reference-counted arena storage as request-cache payloads. Free owning SDK
records with their documented `free` method and response objects with `release`;
do not independently free their payload fields. Convenience data text streams
directly from graph nodes using
the existing Python-compatible JSON scalar formatter, without a flattened
temporary tree. Validation and both JSON writers use a fixed 64-frame traversal
with u32 container indexes; depth overflow and cycles refuse rather than recurse.
Null, bool, integer and float JSON round trips also passed with a zero-byte
allocator in both modes; that is not a whole-runtime static-memory guarantee.

`dszig.CallResult.native(allocator, response, options)` retains the supplied
response object rather than copying its header. It provides the public
string-or-dictionary output view, separate from the unchanged ordered artifact.
`outputs` is now an owning `sdk_response.PayloadValue` whose `root.array` is the
actual mutable list. Use `CallResult.init(allocator)` for an empty result, mutate
through `outputs.root.array`, and use `outputs.retain()` when that same list must
outlive the result while its allocator remains alive. `clone` instead owns an
independent, memoized graph copy in the supplied allocator.
`usage`, `cost`, and `cost_details` also use `sdk_response.PayloadValue`: inspect
their `root` graph or call `toValue(arena)` for a flat borrowed projection.
Each empty result has independent usage and detail dictionaries. These are
Python-capable payloads, not JSON-only fields; even a nonnumeric mutable `cost`
can be retained and cloned, although summing known costs still requires numbers.
Retain an individual payload when it must outlive its result. When replacing a
field, obtain the replacement lease first and free the previous lease.
LM call methods use their documented arena for execution storage; clone results
into longer-lived storage before destroying that arena. As with SDK objects,
a retained payload's backing allocator must outlive every reference, including
cache references. Releasing an original payload handle is safe; destroying its
backing arena while a cache retains that opaque graph is not. Keep shared SDK
payloads in startup-created or otherwise sufficiently long-lived storage.
Options select `chat`, `auto`, `responses` or `text` and requested logprobs.
Visible text concatenates text and JSON data parts; reasoning, tool calls and
citations remain independent. Text mode preserves provider text-score offsets
and unknown scores. `dump`/`load` preserve costs and SDK usage while a cache
load exposes empty public usage. `call_result.combine` retains the original
response objects in order, merges
usage, promotes mixed outputs, preserves first response-bearing model selection
and sums only fully known costs. Its new list retains the original dictionary
rows and mutable wrapped values, not deep copies of them. Appending to an input
list does not append to the combined list; mutating a shared row does remain
visible. Shallow payload dependencies keep these rows alive after input results
are freed, with allocation-free iterative final release rather than recursive
destruction. Combined cost-detail candidates likewise retain the original
metadata values, including repeated and empty dictionaries. Aggregate usage
gets fresh dictionary headers while ordinary nested values remain shallow.
Convenience outputs and native usage dictionaries are built directly into graph
storage. `CallResult.clone` uses one object/graph memo across SDK responses and
all four public payload fields: repeated objects remain repeated, and shared
containers and mutable scalar boxes stay shared inside an independent copy.
`CallResult.dump(allocator)` returns an owning canonical graph record
(`sdk_response.PayloadValue`); free that record with `record.free()`.
`CallResult.load(allocator, record)` retains an owning record's original cost
and cost-detail roots. Its output conversion copies each dictionary/list
occurrence independently, but leaves tuples and other objects unchanged; output
values are not restricted to lists. Missing details and loaded public usage get
fresh dictionaries. Explicitly borrowed records are snapshotted first, as at the
SDK ownership boundary. Convert flat parsed input with
`PayloadValue.fromPythonValue` before loading; that owning import cannot restore
dictionary/list identity already erased by JSON or by-value runtime headers.

Dump uses Source's distinct `plain` conversion: dictionary/list/tuple occurrences
get fresh containers, tuples become lists, and mutable noncontainer leaves stay
shared. Immutable text and dictionary-key bytes belong to the record; only
untouched noncontainer leaves and non-scalar cost roots retain public payload
owners. BaseModels follow dictionary construction through model iteration, not
`model_dump`: empty `Reasoning` becomes `{}`, nonempty `Reasoning` in projected
metadata refuses, and `Code` exposes its `code` field. Cost is deliberately not
projected. Canonical SDK response records retain opaque payload roots and copy
immutable headers, so a dump can outlive its result and SDK objects.
`Response.toPayload` and `Response.fromPayload` provide the corresponding owning
graph serde boundary; existing `toValue` projections still borrow their leaves.
Invalid numeric combinations return `InvalidCost`, not a fabricated zero
or unknown bill.

Debug and ReleaseSafe each compared all **57** frozen actual Source result
cases: 54 accepted and three refused combinations, without claiming Python
exception identity. The SDK projection consumed the corresponding **45** raw
response rows, not 45 additional independent cases. Before the response-object
cutover, allocation-failure sweeps
covered **331 SDK** and **2,741 result** sites per mode. Nine distinct shapes
also survived actual scripted generation, recording, cache hits after the
source arena was destroyed, cache dump/load and replay. The injected adaptation
in that carrier check is a native lifecycle invariant, not a Source observation.

`lm.Completion.message` and `.adaptations`, scripted/replay carriers and both
live transcript codecs retain supplied artifacts. OpenAI Chat and Responses
parsers construct ordered parts while reading the wire, retaining adjacent text,
refusals, generated media, citation adjacency and part-local reasoning replay
state. Nine newly observed actual Source messages matched in Debug and
ReleaseSafe, including the previously failing missing-message path, owning
clones and the completion codec. These are separate observations from the
57-case standalone result corpus.

Actual OpenAI HTTP calls additionally fold a single JSON-object text part into
a canonical data part when the request schema declares any judgment. Mixed
ordinary/judgment properties and scores above Jev's ceiling are allowed here;
this is not TypeSafe's stricter request validation. Multiple text parts,
non-object answers and malformed JSON remain text without repair. Other parts
and the replaced text's continuation stay unchanged. Nonfinite object values
fail SDK data construction rather than silently remaining text. Debug and
ReleaseSafe each exercised **26 new frozen Source cases over localhost HTTP**:
24 accepted responses and two refusals, comparing whole canonical SDK responses
without claiming identical exception classes. The original message path produced
12 mismatches; the request-aware path produced none. Extending comparison to
whole responses exposed a separate canonical-serde bug: unknown `Usage` must be
omitted, not serialized as `{}`. The fix preserves the actual nonnullable SDK
`Usage` object and retains reported zeros. Six additional frozen Source cases
(null, empty and zero-valued usage in both dialects) matched whole responses
over HTTP in both modes. All request schemas were checked and server threads
joined per mode.
This proves scripted transport/codec behavior, not authenticated model inference.

Responses also retain unknown output/content blocks in the ordered
`provider_data._lm15_unmapped` index, with their original wire positions and
Python-style type labels. The five recognized provider-executed tool item kinds
are deliberately excluded. Newly observed unknown blocks replace an existing
index; an existing index is retained when no unknown blocks occur. Reasoning
continuation state stringifies truthy values and omits falsy values, rather than
requiring the provider to send strings. **36 additional frozen Source cases**
matched complete canonical SDK responses over localhost HTTP in both Debug and
ReleaseSafe, including nested unknown blocks, index replacement/order, all five
tool exclusions and seven reasoning-state shapes. All 36 request schemas were
checked and server threads joined in each mode. These are value/codec
observations, not shared SDK-object lifetime or authenticated-inference proof.

After the graph-backed payload cutover, all those conserved provider
observations were re-compared without re-executing Source. Whole SDK values,
expected refusals and all request schemas still matched in both modes.

The actual `live -- --smoke` CLI additionally completed four localhost requests:
Chat Completions and Responses, each with and without native reasoning. Each
returned `4` and wrote a real transcript. The production `live_suite.parseCalls`
reader then replayed all four artifacts through both `LM.callResult` and typed
ChainOfThought in Debug and ReleaseSafe, preserving complete SDK messages,
provider data and canonical metadata and consuming every recorded request.
Responses artifacts included the unmapped-block index and non-string reasoning
continuation state. This is fixture-backed CLI/writer/reader proof, not a
live-model result.

TypeSafe also retains its canonical data part, including selected values,
per-field distributions and classification method; opaque box edges retain their
original numeric kinds. Both modes matched two accepted messages and one
refusal from the conserved original TypeSafe Source receipt, without rerunning
that oracle or counting these as additional independent Source cases.

`LM.callResult(arena, request)` executes once and retains the actual canonical
`Completion.response` object published by the engine. `LM.model_type` selects the client output
mode independently of the HTTP wire dialect. Physical groups finalize before
cache publication; invalid combined costs cannot publish through a fresh cache
or an enclosing cache over a preexisting legacy entry. A cache hit has empty
public usage but retains the original SDK usage and cost. Candidate groups combine their
responses, outputs, costs and provenance in order. Custom engines using this API
must supply their genuine canonical response object; joined text or an
unpublished parser message is not silently reconstructed by the LM.

Both modes compared the full results and dumps of **36 native and nine cached**
rows from the conserved 57-case Source receipt through this public LM/cache path.
These overlap the standalone corpus, not 45 new independent Source observations.
The cached rows inject the receipt's prepriced cost/provenance at the engine
boundary; actual catalogue pricing is exercised separately below. Public
global/client/module response history and arbitrary Python class/path identity
remain unfinished boundaries. The conserved Source history probe exposed a
mutable-payload alias gap in the public LM path; graph-backed data/provider
payloads now preserve that internal alias through actual `LM.callResult`.
A new native consumer in Debug and ReleaseSafe matched two assertions from that
saved Source mutation row: appending 100 values produced 101 items through both
references, after the engine's source arena was destroyed. An independently
owned result clone remained isolated. A permanent actual-LM/cache regression
also covers probability/provider aliases, cache destruction, independent empty
containers and the unchanged output-text snapshot; the allocation-failure
completion regression now includes shared data/provider roots.

Stable response objects now preserve identity through direct `CallResult.native`
and `combine`, including repeated references. Native consumers in Debug and
ReleaseSafe destroyed constructor storage, original response handles and input
results before checking retained objects; an independent clone then outlived
those objects while preserving cross-response payload aliases. Actual JSON
serialization emitted the canonical response, not allocator/refcount internals.
Permanent regressions cover these transitions and exhaustive clone-allocation
failure cleanup. Both modes passed 609 tests; the same conserved 68 HTTP cases
and four actual CLI artifacts were re-compared after this object cutover, not
counted as new Source observations.

OpenAI, TypeSafe, scripted and replay engines publish their canonical response
before returning. `lm.publishResponse` explicitly distinguishes borrowed parser
artifacts from owning clones: it snapshots borrowed artifacts without freeing
static provider leaves, while releasing replaced owning fields. Canonical
metadata, scores and provider data then belong to the class; the completion
retains borrowed views of those fields. Parser-only calls can still expose an
unpublished message without inventing a configured model.
Recording reads the published message, cache clones own independent response
objects, and native cache deserialization restores canonical objects from their
recorded fields. The LM no longer fabricates a response during finalization.

An earlier native-only Debug/ReleaseSafe consumer checked the engine object's
identity through direct calls and cache misses, cache persistence without another
engine call, shared payload mutation and independent clone lifetime. Its repeated
cached-response identity check was not Source equivalence and has since been
replaced by the canonical reconstruction behavior described below.
Its general-purpose allocator reported no ownership leaks. The actual
HTTP path re-compared the same conserved 68 Source cases per mode with no
mismatches; all request schemas validated and serving threads joined. Four actual
CLI runs and their production replay/typed-CoT readers also passed. These remain
localhost transport and native lifetime proofs, not new Source observations or
authenticated inference.

`sdk_response.Request` now provides a stable, retained SDK request object,
separate from `lm.Request`'s execution-adapter view. Its constructor owns message,
system and tool headers while retaining owned JSON payload aliases across message
data and configuration. An independent clone copies that whole graph together.
Typed generation configuration includes reasoning budgets, cache intents, tool
selection, exact integer knobs and opaque response schemas; contradictory options,
duplicate tool names and selections outside the request's tools are refused.

Actual Debug/ReleaseSafe consumers verified retained request identity, constructor
payload sharing, an independent clone surviving the originals, exact wide negative
seeds, budget grading, tool-schema preservation and serialization round trips with
no allocator leaks. The request projection also matched the corresponding field
of the conserved Source history case in both modes; this is **not** a match of
that complete history case. Three permanent regressions cover ownership,
contradictory options and exhaustive construction/clone allocation-failure cleanup.
The native gates passed 627 tests in each mode, all 687 replay requests and the
actual RAG, bootstrap and parallel examples.

`Request.snapshot(allocator, io, maximum_file_bytes)` freezes local media before
cache lookup or engine execution. With no paths it retains the **same** request;
otherwise it returns a new request with path-backed media replaced by base64,
including system parts and nested tool results. Existing owned JSON payloads
remain shared. Each call reads current file bytes, while a completed snapshot
remains unchanged if the file is overwritten or removed. The explicit per-file
byte cap is a native safety bound, not Source's unbounded `read_bytes` behavior.
Actual Debug/ReleaseSafe executables verified these transitions, retained payloads
after the original request and caller references were freed, independent cloning,
and oversized/missing/empty-file refusal with clean allocator shutdown. Three
permanent regressions also cover inclusive byte limits, every media kind and
exhaustive allocation-failure cleanup after partial replacement. These are native
controls informed by static Source inspection, not additional Source observations.

`Request.toPayload(allocator)` returns an owning canonical request record, using
the same record projection as SDK responses. Each invocation creates fresh typed
dictionary/list headers but retains the exact opaque roots stored in message
data, function parameters, builtin configuration, response schemas and extensions.
Mutations through those shared payloads remain visible. Call `record.free()` when
finished; the record can outlive the original request and input handles, subject
to the payload backing-allocator lifetime contract above. This is not a deep copy.

One actual Source observation checks all five payload paths, fresh typed headers,
shared nested mutation and the complete serialized record. Native Debug and
ReleaseSafe consumers match that complete observation with clean allocator
shutdown. A permanent exhaustive allocation-failure regression also releases both
the request and input before mutating the retained record.

`Request.fromPayload(allocator, record)` reconstructs an owning, validated SDK
request from a canonical graph record. Each import owns fresh typed headers but
retains the record's opaque payload roots, including aliases across messages,
tools and configuration and between separate imports of the same owning record.
Explicitly borrowed records are instead snapshotted once per import: aliases
within each import survive, but separate imports do not share mutable storage.
Release the returned request with `request.release()`; the payload backing
allocator must still outlive all retained references.

A separate actual Source `request_from_dict` observation checks all five opaque
paths, fresh headers, aliases across two imports, nested mutation and the complete
re-serialized record. Actual Debug and ReleaseSafe consumers match the whole
observation with clean allocator shutdown. An exhaustive allocation-failure
regression frees the input record and parse arena before mutating imported
requests, and checks both retained owning records and isolated borrowed snapshots.
These record operations do not by themselves integrate canonical requests into
public LM calls or history. The TypeSafe and OpenAI physical SDK boundaries are
described below; the public LM/history input-side cutover is still separate.

Tool-call inputs now use the same `sdk_response.JsonValue` graph as data and
provider payloads. Construct them with an owned payload or an explicit borrowed
graph; the SDK reader requires the input object rather than inventing a default.
Construction retains owned input identity, including sharing between multiple
tool calls, data parts and provider data. Independent clones preserve that sharing
within the clone without sharing mutable nodes with the original.
An actual native executable first reproduced lost tool-input sharing, then passed
in Debug and ReleaseSafe after the migration. It exercised mutation, independent
cloning, original-input destruction, the actual `LM.callResult` response identity
and Python-compatible ASCII tool-argument serialization, with clean allocator
shutdown. Two permanent regressions cover container growth, escaped clone lifetime
and exhaustive allocation-failure cleanup through public result construction.
Both modes also re-compared the same 36 conserved HTTP cases with no mismatches;
four current live-CLI localhost runs passed across Chat/Responses and plain/native
reasoning, with serving threads joined. These are not new Source observations or
authenticated inference.
Continuation data now uses that same owned `JsonValue` graph. Owned constructor
inputs remain shared across message/part continuations, tool inputs, data parts
and provider data; independent clones preserve sharing within the copy.
Standalone continuation cloning also uses one shared graph memo. Missing data
in SDK and metadata readers creates a fresh object per continuation; explicit
null or non-object data is refused at those decoding boundaries.
Canonical JSON equality compares graph nodes directly with a bounded, nonrecursive
stack rather than allocating flat projections. It keeps array order, ignores
object insertion order and distinguishes boolean/integer/float kinds.
An actual executable reproduced a lost continuation mutation before the repair.
The repaired Debug/ReleaseSafe consumers passed eight lifecycle controls, including
cache reuse, escaped result lifetime, shared list growth and independent defaults,
with clean allocator shutdown. Three additional permanent regressions cover
standalone clone lifetime, reader/allocation-failure cleanup and bounded equality.
Both modes re-compared the same 36 conserved HTTP cases without mismatches.
Four current localhost CLI runs passed; the existing transcript reader and typed
CoT replay also consumed all four artifacts in both modes, comparing canonical
message, provider-data and metadata values. These are native controls and frozen
Source comparisons, not new Source observations or authenticated inference.
Adaptation `asked` and `applied` now use `sdk_response.PayloadValue`, the same
graph/owner representation without the JSON-field restriction.
`PayloadValue.fromPythonValue` imports supported native Python values without
projecting custom objects to JSON. Owned inputs retain shared containers across
adaptations, data and provider fields; one clone memo preserves those aliases
inside an independent copy. Tuple, bytes, mutable bytearray, Decimal, nonfinite
float and cyclic-container payloads remain genuine graph values. Decoding these
two fields uses the full Python protocol; JSON-only SDK fields remain strict.
Cyclic payloads can be cloned but refuse flat wire projection. Neither this work
nor the earlier controls establishes full Python class or garbage-collection
equivalence.
Four newly observed Source adaptation scenarios matched the native executable in
Debug and ReleaseSafe: shared mutable data, genuine non-JSON kinds, cyclic copies
and absent defaults. The executable also exercised actual LM/cache calls and
escaped ownership. Its strict-data negative control exposed a missing nested
constructor check; `Response.fromFields` now validates owned part/continuation
headers and adaptation records before publication. Non-JSON data/provider inputs
are refused while those same values remain valid adaptation payloads.
Three new permanent regressions cover custom/mutable clones, owning readers with
exhaustive allocation failures, and cyclic copies versus projection refusal;
the existing public LM/cache alias regression also covers adaptation sharing.
The final native gates passed 627 tests per mode and all 687 replay requests.
Both modes re-compared the conserved 36-case HTTP subset without mismatches.
Four fresh localhost CLI runs and both-mode artifact-reader/typed replay passed.
Those HTTP/CLI runs are not new Source observations or authenticated inference.

Five new actual Source output scenarios matched a native executable in Debug
and ReleaseSafe: native/combined row identity and nested mutation, independent
deepcopy with repeated-row identity, mixed mutable bytearray promotion, cycles,
and independent default lists. The native before-proof failed on a lost combined
row mutation. Additional native controls cover retained lists after result
destruction, a 10,000-owner dependency chain, and actual LM/cache calls with
caller mutation and escaped output ownership. Those controls are not additional
Source observations. Permanent regressions cover shallow-row lifetime, cyclic
copies, exhaustive allocation failures, and iterative dependency release.
At the shared SDK/output memo checkpoint, native gates passed 627 tests per mode
and all 687 replay requests. Both modes re-compared the conserved 57-result
corpus with 2,184 exercised allocation-failure sites. The preceding output-list
checkpoint also re-compared the conserved 36-case HTTP subset without mismatches;
four localhost CLI runs and both-mode artifact replay passed there.
These are overlapping native re-comparisons, not new Source executions or
authenticated inference. Arbitrary class projection is outside this finite
output-list proof; usage/cost metadata is covered separately below.

Owning `CallResult.clone` now uses one graph memo across its SDK responses and
public outputs, rather than independently copying those two views. A dictionary
referenced by both an SDK provider field and repeated output rows remains one
independent dictionary in the copy. One additional actual Source deepcopy
scenario observed this cross-field alias and mutations in both directions.
The unchanged native driver first failed with `LostSDKOutputAlias`, then passed
in Debug and ReleaseSafe, including after the original owners were destroyed.
Existing lifetime and exhaustive allocation-failure regressions also exercise
that shared memo.

Public usage, cost, and cost-details fields now retain mutable Python-capable
graphs too. Three frozen actual Source metadata scenarios matched complete
native observation rows in Debug and ReleaseSafe: independent default
dictionaries, shallow combined candidates with repeated aliases and independent
usage sums, and one deepcopy spanning repeated output rows, usage, details and a
genuine mutable bytearray cost. The native before-proof failed on a lost
candidate metadata mutation. A separate native control retained metadata after
the source result and SDK response were destroyed. Existing SDK lifetime
regressions now cover SDK/public-metadata sharing; one additional regression
covers whole-result mutable-class copies and exhaustive allocation failures.
A native loader control caught independent imports splitting a shared bytearray;
one common import memo fixed that failure without changing the driver.
Both modes re-compared all 57 conserved result rows with 1,537 exercised
allocation-failure sites, the five conserved output rows, and the unchanged
cross-SDK/output alias driver. These are not additional Source executions.
That checkpoint passed 628 tests and all 687 replay requests per mode; the
actual RAG, bootstrap and parallel examples also ran successfully.

A subsequent actual Source SDK Adaptation/deepcopy probe observed repeated
tuples, arbitrary-key dictionaries, sets and frozensets. All four complete rows
matched native Debug and ReleaseSafe after the unchanged native driver first
failed on a lost imported tuple alias. Genuine runtime containers now use a
memoized graph importer. At that checkpoint the flat loader also shared its
import context across fields. A native loader control retained a shared tuple
across outputs, cost and details after the input storage was destroyed. These imports are owning snapshots, not
preservation of the original caller's Python objects. Expanded SDK allocation
fault coverage checks nested aliases, distinct equal containers and independent
mutable leaves; result loading has its own cross-field lifetime/fault regression.
Graph cloning looks up existing containers before reserving memo capacity, so
alias hits do not attempt unnecessary allocations.
The resulting Debug and ReleaseSafe gates each passed 629 tests, all 38 build
steps and all 687 replay requests; actual RAG, bootstrap and parallel examples
also ran successfully.

The canonical graph-record cutover then compared **seven** actual Source rows
in both native modes: original cost/details aliases, independently copied
output dictionaries/lists, tuple-to-list dump projections, non-list outputs,
BaseModel iteration and refusals, plus SDK owner escape and Code field
iteration. The native before-proof failed when mutation of the original cost
was invisible after loading. The owner-escape consumer now releases the
original result, response and input owners, loads the surviving record, releases
that record, and observes shared cost/tool-input and adaptation/bytearray
mutations through the loaded result. Permanent regressions include exhaustive
allocation-failure coverage for SDK serde ownership.
Both modes again matched all **57** frozen result rows, now exercising **1,686**
allocation-failure sites per mode with the new ownership paths. This count is
not additive to earlier sweeps. At that checkpoint, Debug and ReleaseSafe gates
each passed **632 tests**, all **38 build steps**, and all **687 replay requests**;
actual RAG, bootstrap and parallel examples also ran successfully.

Public `LM.callResult` calls through `CachedLM` now store the owning canonical record itself,
rather than deep-copying SDK payloads during insertion and reconstructing outputs
from them on a hit. Like Source's memory cache, lookup deep-copies the whole record
once, then loads each response occurrence separately. A repeated engine response
remains the same object on a miss but becomes distinct SDK headers on a hit;
shared opaque payloads remain shared within the hit's independent copy.
Mutations made to the original SDK payload between insertion and lookup remain
visible in that copy, while the already-dumped convenience outputs stay at their
insertion snapshot. Public usage is empty without clearing SDK usage.

A new actual Source cached-group observation exposed both the former premature
payload copy and the incorrect repeated SDK-header identity. The unchanged native
consumer failed before repair, then matched the complete observation in Debug and
ReleaseSafe with clean allocator shutdown. Both modes also re-compared the seven
canonical rows and unchanged 57-row corpus above; those are overlapping
re-comparisons, not additional Source executions. One new permanent regression
covers mutation timing, shared payloads with separate SDK headers, independent
successive hits, execution-arena teardown for private projected values, empty
public usage, JSON persistence and exhaustive allocation-failure cleanup.
Immutable projected strings/keys are now owned directly, so cache snapshots need
not retain their former execution arena merely for those bytes.

Native cache persistence now stores `{completions, record}` per key. Completion
records preserve the existing native SDK codecs, including nonfinite score tags;
the optional public record stores the output and metadata snapshots. Loading
reconstitutes SDK records from those completion artifacts. The old array-only
native cache shape is no longer accepted. This remains DSZig's JSON persistence,
not DSPy's disk-cache format, arbitrary Python-value persistence or preservation
of graph aliases through JSON. Native text-only calls without public results
continue to cache their native completion carriers.

Current Debug and ReleaseSafe qualification each passed **637 tests** and all
**38 build steps**; both actual replay CLI runs consumed all **687 requests**.
The eight new Source public-call observations also capture request/response
identity, candidate seeds, local-media snapshots, refusal ordering and
global/client/module history controls. They are research evidence for the next
integration step, not eight implemented native parity cases.

This does **not** implement response history. Passing the actual SDK Request
through public calls and the remaining engines, binding returned lists into
actual history entries, and module/global/client history ownership still require
integration. TypeSafe and OpenAI now accept canonical requests at their physical
boundaries; that alone is not the public LM/history cutover.
Arbitrary third-party Python models and mutable fields outside those listed
above—including constructor aliasing between owned and borrowed views—remain
unfinished. Canonical records preserve owned graph identities; explicitly
borrowed views and flat input still cross an owning snapshot boundary.
These finite comparisons do not establish full Python class/garbage-collection
equivalence or authenticated inference.

### Advisory model pricing

`model_prices.Catalogue.bundled(startup_arena)` loads the exact DSPy 3.5 bundled
LiteLLM 1.108.3 catalogue into caller-owned startup storage. It does not fetch
prices or read environment variables. The unmodified snapshot and its upstream
licence ship together. `Catalogue.init(arena, json, source_info)` accepts an
explicitly obtained catalogue; immutable lookups allocate no memory and keep
namespace order, provider-matched bare entries and ordered aliases distinct.
Capability-only rules affect whether a model is known, but never supply prices
or inherit another entry's rates.

`lm_cost.estimate(arena, &catalogue, &facts)` calculates advisory USD cost from
validated canonical `lm.Usage`, actual response model/provider data, the pricing
provider and any declared metadata namespaces. It preserves reported zero and
other provider-reported JSON amounts; unknown pricing is null, not zero.
Estimates separate seven token dimensions using exact integer subtraction,
per-dimension/tier strict long-context thresholds and Source's float operation
order. Served tiers override requested tiers. The direct facts API also carries
hosted-tool, generated-media and long-cache-retention facts. Responses
`output_image`/`output_audio` parts now supply the generated-media fact after
the SDK's truthy payload fallback and shared base64-shape validation. Malformed
payloads fail generation even if the response reports zero cost. Valid media
counts toward synthetic tool IDs and does not fabricate an empty text part.
The fact survives cloning, recording/replay and cache persistence; this is not
a full generated-media SDK object API or general multimodal transport support.

Enable pricing on the LM handle with **dedicated advisory storage**, disjoint
from generation and recording arenas:

```zig
const prices = try dszig.model_prices.Catalogue.bundled(startup_arena);
var priced_lm = model.lm();
priced_lm.pricing = .{
    .catalogue = &prices,
    .arena = billing_arena,
    .provider = "openai", // Explicit billing provider, not inferred from HTTP protocol.
};
var recording: dszig.lm.RecordingLM = .{
    .inner = priced_lm,
    .arena = recording_arena,
};
const measured_lm = recording.lm();
```

Each physical completion is priced once, before recording or cache publication.
The configured model comes from the actual engine getter, and response facts
come from the actual completion. Engine headers may be immutable: execution owns
its headers rather than writing through an engine's const slice. Recording
clones the already-known `cost` and `details` into `calls.items[i].billing`;
it does not run another estimator. Cache persistence and replay retain those
facts. A cache hit creates neither another physical recording nor another bill.
Missing configured identity or exhausted advisory storage leaves billing absent
without losing generation or required recording. Known billing, like other
required response evidence, still requires ordinary storage when cloned.

A native Debug/ReleaseSafe executable exercised single and grouped public
results, immutable engine headers, destruction of original generation storage,
cache hits after pricing storage exhaustion, dump/load, recorded replay and a
fresh successful generation with unavailable advisory storage. It checked the
previously Source-observed exact cost bits, preserved SDK usage and empty public
usage on hits, with no cache-owner leaks. This does not implement Python's
global/client/module response history or its mutable alias/lifetime semantics.

Independent pinned Source pricing observations covered **4,420 cases**:
3,469 estimates, 944 unknowns and seven reported bills. Debug and ReleaseSafe
matched all **8,840 cost/provenance observations**, including every non-reserved
bundled catalogue entry, namespace/alias precedence, zero/reporting, tier and
threshold boundaries, exact wide counts and float-conversion overflow.
Separate probes matched 55 model-rule truthiness/cost cases (165 observations),
660 actual regular-expression matches, and three oversized threshold captures
(six observations). The latter first exposed six native mismatches, then
matched after preserving CPython's default 4,300-digit conversion guard.
These probes overlap semantically; their counts are not a disjoint total.

Actual typed Predict → cache → RecordingLM execution preserved the bill across
a cache hit, and zero dedicated pricing capacity preserved the completed
generation. Isolated cost/provenance observation controls produced 3,476/4,420
real mismatches respectively in both modes. Ordinary Source-backed pricing
used 2,552 caller-arena bytes before fixed-stack rate keys and borrowed ASCII
thresholds, then 1,800 bytes with identical cost bits in both modes. That is one
allocation probe, not whole-runtime static-memory or throughput evidence.
A separate actual Responses parsing/pricing probe covered 21 new cases
(17 accepted responses and four media-validation refusals). Before repair it
exposed 37 native mismatches; afterward Debug and ReleaseSafe matched the
Source acceptance, text, tool IDs and cost/provenance, with media facts retained
through cloning, replay and cache dump/load. An earlier probe failed before
emitting its accumulated successful rows; its complete failure receipt is
retained, those lost observations are not counted, and it was not rerun.
At that media checkpoint, native gates passed **592/592 tests and 38/38 build
steps per mode**, actual examples and all **687 replay requests**; the later
token-logprob extension is documented separately above.

Only the bundled capability-pattern texts have bounded native matchers;
other patterns are explicitly refused, not interpreted by a general regex
engine. Remote refresh/shrink validation and full merged capability dictionaries
are not provided. Native catalogue JSON also refuses NaN, Infinity and BOM
accepted by Python's loader. Prices are pinned advisory metadata, not a promise
of current provider billing.

### LM accounting and per-call Predictions

Set `Context.track_usage = .{ .storage = &storage, .limits = limits }` to enable
automatic accounting on `dszig.call(&program, ctx, input)`, `Predict.call`,
`ChainOfThought.call` or `Retrieve.call`. Retrieval's canonical result has a
`passages` field; its low-level `forward` still returns the passage slice.
A reflected field struct becomes a native
`Prediction(Fields)`; an existing Prediction retains its instance identity.
An optional keeps null, a tuple transforms only its first element, and other
outputs remain bare. Read or mutate output fields through `fields()`, and use
`getLMUsage()` / `setLMUsage()` for that instance's optional usage graph.
Copying a Prediction handle shares its state; `clone(arena)` owns independent
fields, completions and usage while preserving shared genuine custom objects
across those projections.

With tracking enabled and no active tracker, a canonical Module call opens
one parentless scope, finishes it on success or failure, and attaches an owning
totals snapshot only after success. An empty field struct receives no usage.
Totals are still computed for unattachable outputs, so invalid merges and
resource failures are not hidden. Any active `Context.usage_tracker`
suppresses automatic scopes and attachment, regardless of how the caller
configured it. This matches Source for both context-local and globally
configured trackers; only explicit child `UsageScope`s roll up to a parent.
Direct typed `forward` remains a low-level operation without a per-instance
usage attachment.

Prediction construction without completions has `completions() == null`.
Predict's canonical call retains its parsed completion rows separately from
the first output header. Evaluation preserves completion rows when the program
provides `forwardPrediction`; a plain typed `forward` has no such metadata.
The prepared record's first header is separate from its completion headers,
and an owning clone retains both across scratch reset and prepared-state reuse.
This is a typed row view, **not** Python's mutable column-based Completions
class: dynamic extra columns, ragged column lists, empty-row indexing and
Python shallow-copy behavior are not implemented. Native trace-step field
projections do not carry per-step usage.

At the LM boundary, usage is keyed by the configured model, not the response's
wire-model name. Successful uncached public fanout combines candidate usage
before adding one entry to the active tracker; a failed group accounts each
retained successful candidate separately. The floating-point order matters.
Cache hits retain SDK metadata but perform no physical accounting.
`CachedLM` stores the whole public candidate group before successful final
accounting: a tracker refusal can therefore leave cached responses, and a
later hit returns without another charge. Physical recording/replay still
uses the actual tagged `n=1` requests. Transient tracker, accounting-error,
recording-chain and execution pointers never reach engines or persisted
request records. A direct `LM.completeResponses` caller supplies its tracker
through `Request.usage_tracker`.

`Parallel.Program(P)` uses the canonical call result; choose the pool type
with `parallel.ProgramResult(P)`. The utility itself creates no scope. Each
task/attempt deep-forks an active tracker, including its seeded entries,
without merging charges back into the original. `parallel.Evaluator` follows
the same policy. Forks have stable, distinct headers, inherited capacities and
independent mutable entry graphs, preserving aliases within each snapshot.
They share Storage's lifetime and allocator lock, and remain allocated until
that storage is reset. Without an active tracker, each tracking-enabled
Module task gets its own automatic scope. An enclosing automatically scoped
Module can consequently receive empty totals even while workers incur usage.
SIMBA/GEPA `Rollouts.context` also forks per task and can fail when snapshot
storage is exhausted. Raw Task callbacks receive no Context and do not
implicitly fork one. Automatic-scope tracker headers, like forks, remain
address-stable until their Storage is reset.

Current parent-owned Debug and ReleaseSafe gates each passed **572/572 tests
and 38/38 steps**, including actual RAG, bootstrap and joined Parallel
examples; deterministic replay consumed all **687 requests, eight adapter
cases and nine pipelines**. Both native modes consumed the frozen, disjoint
15-row LM/cache and 19-row Module corpora: 2,602 and 3,647 Source-backed
comparisons respectively, with zero mismatches. Their 30/22 native invariants
and 15 LM Source-integrity checks are separate, not additional Source matches.
The slices leave 630/668 observations unobserved, including client-only
options, public Python history fields, exception representations and
thread-local override-map details.

Both native modes also consumed the separate **53-case, 212-request**
threaded Source corpus with zero mismatches: **41,453 Source-backed checks**
and 212 separately counted native invariants per mode. This covers the
tracking/active-tracker/explicit-child matrix, direct and enclosing Module
calls, evaluation metrics and two deliberate engine failures. Actual tracker
alias relations include automatic scopes and retained per-task copies.
Thread checks compare task-local affinity, caller separation, worker count
and the barrier-forced first overlap; later worker assignments and request
arrival order are scheduler-dependent and are not compared. Python exception
representations and Source's `shutdown(wait=False)` followed by harness joins
differ from native error names and the pool's joined `deinit`. Seven
independently corrupted threaded outputs per mode were all refused.
The eight frozen Prediction-edge Source cases remain a separate, unconsumed
corpus; the row view above must not be reported as full column-class parity.
These are finite proofs, not full DSPy, authenticated inference, current CI
or whole-runtime static-memory qualification.

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
construction remains unsupported. Unboxed Literal floats do not carry object
identity; explicit runtime float boxes below preserve it.

The zero-dependency `python_literal` parser implements the supported CPython
3.11 literal grammar without execution, and `python_value` retains genuine
bytes, mutable bytearrays, identity-bearing floats, tuples, sets, frozensets, complex values, Ellipsis and arbitrary-key
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
| IntEnum runtime operations | 2,324 / 2,324 | 24,901 | 663 |
| IntEnum scalar validation | 35 / 35 | 379 | 6 |
| Builtin IntEnum declarations | 19 / 19 | 134 | 0 |
| Builtin bytes IntEnum declarations | 7 / 7 | 50 | 0 |
| Configured Decimal policies | 368 / 368 | 3,950 | 233 |

The historical runtime consumer exited nonzero with 30 sequence-repetition rows
unsupported. `value.numericProduct` now repeats immutable text and genuine bytes
with integer/IntEnum multipliers in either order. After a focused comparison of
those 30 rows, the complete frozen runtime/scalar/Decimal-policy corpus was
recompared against the current native implementation in Debug and ReleaseSafe.
All rows are now consumed with zero mismatches and byte-identical mode output;
the table reports these current observations, not their sum with old runs.
No Source oracle was rerun. The restored historical driver was verified against
its original archive hash before removing its obsolete repetition skip and
adding genuine bytes-result encoding. One-fold results may borrow their input;
repeated bytes own their new buffer. Use caller-provided startup storage through a fixed allocator
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
strings. One value/record/model clone memoizes custom boxes across its fields
and nested containers: repeated references share one independent cloned box.
Each owning alias retains a counted reference; releasing one does not invalidate
the others. Ordinary handle assignment is still a borrowed copy, not an implicit
retain. Use the allocator that created the clone when releasing owned handles.
Native array/object metadata remains copied by value, so this is not Python
container-graph or cyclic-deepcopy equivalence.

`runtime.CloneContext` exposes that same scope for multiple related values.
Custom vtable clone callbacks receive the context and must use
`context.cloneValue`/`cloneCustom` for children, preserving the common memo.
Source boxes must remain alive for the complete context lifetime. The memo is
temporary; returned owners do not retain a pointer to it. Non-singleton custom
copies provision a shared-control allocation and memo storage through the caller's
allocator; this is not whole-runtime startup-only memory proof.

An independent new DSPy 3.5 boundary probe observed shared bytearray preservation
through direct fields, lists, tuples and non-string-key dictionaries. Native
Debug and ReleaseSafe executables each matched the four distinct scenarios,
including mutation visibility and original isolation, with a leak-checking
allocator. Permanent regressions also exercise source-arena destruction,
independent second copies, early alias release and allocation-failure cleanup.
The integrated pinned gates passed **527/527 tests and 36/36 steps in each mode**,
actual RAG/bootstrap/Parallel examples and all 687 replay requests. These are new
finite clone observations, not a rerun or expansion of the frozen model corpus.

`python_value.float(allocator, number)` boxes an actual builtin float, distinct
from a model that merely implements `__float__`. `python_value.sameIdentity`
compares logical immutable identity: independently owned copies of one float,
Decimal, complex number or bytes object retain that identity after the source
arena is destroyed, while separately constructed same-bit NaNs stay distinct.
Direct NaN equality remains false (signaling Decimal NaN can raise); container
membership/equality applies the identity shortcut first. Hashing remains coherent
across those copies. Identity minting is bounded and allocation-free; exhaustion
refuses construction rather than degrading to address identity. Physical boxes
and their ownership remain independent. Unboxed `Value.float` and native `f64`
projections do not acquire identity, and mutable container graphs remain outside
this protocol.

A new independent Source probe observed 27 boolean results across float,
Decimal and complex NaNs. Native Debug and ReleaseSafe CLIs matched all 27 using
actual lists, dictionaries and decision models, then retained identity across
source-arena destruction and another copy. A separate actual-source class probe
exposed six incorrect cross-class matches among float, Decimal and complex;
per-kind nominal tokens now distinguish those classes without changing their
shared physical layout or numeric equality. Both modes matched all 12 class
observations, including copy preservation. The integrated scalar/mutable-field
checkpoint passed **538/538 tests and 36/36 steps in each mode**, the actual
RAG/bootstrap/Parallel examples and all 687 replay requests. The separate model
corpus re-consumption below remains incomplete; these finite results do not
establish full DSPy parity.

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

`boxed.custom.modelDump(arena)` is the separate default **Python-mode**
BaseModel view. It uses an explicit class serializer rather than guessing from
raw fields or reusing JSON mode: Reasoning/Code become strings, Citation omits
null fields, evidence unwraps nested probability models, and unvalidated
BoundingBox nonfinite fields remain floats. Nested containers are fresh
snapshots; bytes, Enum, Decimal and other nonmodel leaves retain their native
identities. The result borrows leaf storage from its source; use
`value.cloneValue` for independent ownership. Ordinary objects, builtin boxes
and stream namespaces are not implicitly BaseModels.

Projection uses a fixed 64-frame stack, counts model serializers toward its
depth limit, and refuses after 1,048,576 work steps. Storage comes only from the
supplied arena, which can be backed by startup-provisioned memory. These native
limits are not Python's serialization limits. An independent actual-source
probe matched five complete outputs, including a nested tuple/list snapshot,
invalid-UTF8 bytes, genuine Decimal/Enum values and an unvalidated NaN box. This is
not by itself a proof of the automatic Module/Prediction integration above.

`json_graph` also retains genuine scalar boxes, tuples, sets, frozen sets and
arbitrary-key Python dictionaries without JSON coercion. Its pointer containers
preserve mutable aliases across appends; memoized `clone` retains shared/cyclic
containers while copying custom boxes and immutable dictionary keys into the
destination arena. `fromValue` snapshots runtime container headers but borrows
scalar leaves/keys: it is not a lease or a way to make later runtime-header
mutations visible. Build and mutate graph containers directly for that behavior.
`toValue` copies strings, borrows custom leaves/non-string keys, and refuses
cycles or unhashable keys. Both conversion directions use explicit fixed stacks,
at most 1,000 nested containers and 1,048,576 visited values; these are native
resource bounds, not Python limits. Cloning remains iterative and preserves
deeper graphs. Graphs are externally synchronized, arena-lifetime objects, not
garbage-collected Python objects; custom clone protocols must support that arena
lifetime. Low-level set/dictionary mutation must retain unique, hashable members
or keys. Set materialization preserves the observed iteration order and hash
seed rather than reinserting in iteration order, which can change collision
layout. The startup-backed roundtrip consumer additionally clones and restores
the five frozen Python-mode outputs described above; this is overlapping
evidence, not five additional independent Source cases.

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
Chat Completions accepts canonical text parts: a lone part stays a string,
multiple user parts stay ordered text blocks, assistant parts join with
newlines, and tool-result text joins with newlines. Visible reasoning retains the
default thinking-before-text policy. This generation policy is distinct from
the SDK training/display projection below. Chat Completions still rejects
document content before HTTP, following the pinned source preflight rather
than inventing an unsupported route. These implemented
paths do not establish full provider/media parity or a completed post-cutover
provider rerun.

### Other native API contracts

Chat parsing preserves DSPy's inline-header behavior: if a header is indented,
the parser slices the original line using the character count of the trimmed
header match. Unicode whitespace counts in characters, not UTF-8 bytes.

Typed JSON predictions with nested integer fields accept leading zero padding
and separators between whole-part digits, but reject fractional separators
such as `"1.0_0"`. The 4,300-digit limit applies to the integer after removing
padding and separators. Pinned DSPy 3.5.0 `Predict` confirmed the accepted and
rejected cases, and a retained native `Predict` regression failed before the
fix. A standalone native prediction also verified exact signed values at and
beyond the `i64` boundary. Formatting passed; Debug and ReleaseSafe each
passed all 874 tests and 40 build steps. The offline reference gate passed,
including the 687-request mock replay. No local blocker remains; no hosted
inference was needed. Next, verify the scalar-text path separately: DSPy
repairs those strings before validation, so a scalar can accept a spelling
that nested validation rejects.

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
Compile on a mutable optimizer instance. Its `error_count` accumulates teacher
and metric failures across calls; `report.errors` describes only the current
successful compile. Reaching the budget propagates the triggering error, but
does not prevent a later compile from succeeding without another failure.
Out-of-memory propagates immediately without consuming this budget.
When `BootstrapFewShot.max_errors` is null, the budget comes from the outer
`Context.max_errors` (default 10), not from `teacher_settings.max_errors`.
An explicit optimizer limit, including zero, takes precedence.
Set `teacher_settings.max_trace_size` to override trace retention for teacher
attempts without changing student inference. A nonpositive limit leaves the
teacher metric with an empty trace and produces no augmented demos; the metric
can still accept the example.
Set `teacher_settings_ref` to share a live settings object. Bootstrap reads it
at the start of each attempt, so an update in a teacher or metric callback
affects the next attempt, including a retry. Clearing an override restores the
outer context's setting on the next attempt. The object must remain alive
through compile and must not be mutated concurrently.
Teacher signature matching follows DSPy's named field roles, descriptions,
prefixes and instructions; annotation types and field order do not decide
signature equality.
`BootstrapFewShot` and `BootstrapFewShotWithRandomSearch` accept a distinct
typed teacher program if its named predictors match the student. The
bootstrap metric sees the teacher's own output and trace, and only the
student's type is returned. With `BootstrapFewShotWithOptuna`, call
`compileWithTeacher` for a distinct typed teacher; trials still score the
student. This does not change BetterTogether's erased teacher restrictions.

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
**in place and do not move it**, and reserve a stable `P.Output` slot.
Adopt its record once into a stable
`optimize.RecordPrediction.State{ .output = storage.record }`, then call
`optimize.runProgramInto(P, ctx, program, example, &output, &state)`.
Use `state.output`, not the original record header, after adoption. The record
must have capacity for every declared output field; its backing must not alias
program inputs or outputs. The state, typed output slot and borrowed children
must remain alive and unmoved until the prediction is discarded or reused.
Evaluation and bootstrap reserve these slots before their loops.

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
after earlier fields were written. The returned `RecordPrediction` borrows
that state and exposes its ordinary record through `fields()`, including
mutation of existing values through `getPtr`. Its optional `getLMUsage()`
belongs to that invocation. Copies of the handle observe the next reuse;
`prediction.clone(arena)` retains independent fields and usage across both
state reuse and scratch reset.

`optimize.runProgram` materializes a separately retained `RecordPrediction`;
both execution paths share input decoding and field codecs. Copying only a
record header does not extend a borrowed model's lifetime. Use the
Prediction's owning clone when retaining the whole result, or
`value.cloneRecord` when deliberately retaining only its field projection.
Nested container conversions, retained predictions, program execution and
metrics can still allocate.
This is not a whole-runtime allocator-free proof.

`optimize.Metric` and GEPA feedback callbacks can return `ParseValue` or
`OutOfMemory`. Evaluate and bootstrap apply their existing error budgets to
metric failures as well as program failures. Bootstrap restores temporary
teacher demos before scoring a successful teacher call. A demo comparison
failure counts against the same budget as a teacher call failure. If the budget
permits continuing, the compile-local teacher keeps each completed filter for
the next attempt, but not a partially filtered list. The caller's teacher is
unchanged. Bootstrap and SIMBA apply `Example` keyword binding when
turning a trace into a demo: explicit fields override inherited `base` fields,
but duplicate explicit keywords fail. Bootstrap propagates a binding failure
outside the teacher error budget; SIMBA skips that strategy candidate before
scoring it. SIMBA resolves a unique predictor-bound student LM for trajectory
sampling. Nonempty `teacher_settings` reserves the first sampler and may
select its LM; it does not override the adapter or trace limit. The other samplers use
temperature `1.0`. Trajectory-specific bindings do not modify the student or
the returned program. If the first sampler uses the baseline LM, its rollout
ID applies to later evaluations in that compile; native LM handles do not
mutate a caller's model-wide defaults. SIMBA retains a successful prediction and trace
when scoring fails, with a zero score. GEPA uses its configured failure score,
with an empty prediction when trace capture is disabled versus no prediction
or trajectory when capture is enabled. Reflective feedback failures propagate.
Allocation failures remain fatal. `optimize.Metric` receives a nullable
`?*const optimize.RecordPrediction`: SIMBA calls it with null after a program
failure, rather than skipping scoring or supplying an empty record. A callback may
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
captured by value. Bootstrap reads the current binding each attempt; the
induction program reads the object it captured at initialization each time it
generates rules. Candidate evaluation snapshots are independent, while
instruction mutations propagate across matching native signature-class identities
in the reachable caller/teacher/candidate graph, including an earlier winning
program. Class identity survives program copies and remains distinct from
structural signature equality. Instruction-replacement optimizers and state
restoration create fresh class identities.
An actual native executable exercised rule induction, selection and the returned
typed predictor against the independently observed DSPy behavior.

`BestOfN(P).init(program, options)` takes a typed reward callback, an attempt
count and a required numeric `dszig.Value` threshold. Use `.{ .float = 0.8 }`
for a floating threshold, `.{ .integer = 100 }` for an exact integer, or
`.{ .number_string = "90071992547409931234" }` for integers beyond `i64`.
Comparisons do not round integer thresholds to binary64.
Its output is optional: DSPy returns no prediction when
no reward exceeds negative infinity, including all-NaN rewards or no attempts.
Each attempt has its own program value and gives its predictors a rollout ID
and a default temperature of 1.0. Unrelated predictors keep their ambient
settings even when they use the same LM handle. Stored and per-call predictor
config can override sampling defaults. A per-call LM override selects the
model for prediction, not the retry's starting rollout. Only the winning
program trace is retained, excluding reward-function calls.
Cancellation and an explicit streaming stop propagate immediately without
consuming the failure budget or publishing a partial winner.
The signed failure budget persists across calls, including DSPy's unusual
index-versus-remaining-budget failure boundary. Program values follow the same
immutable-slice ownership convention as the existing optimizers.

`Refine(P).init(program, options)` additionally takes `module_code` and
`reward_code`: the actual source text that Python obtains through introspection
must be supplied explicitly in Zig. It generates real `OfferFeedback` predictions
between attempts, applies per-signature hints through the adapter, and selects
the first strictly best result. Shared signature classes use the last matching
predictor name; empty advice installs no wrapper. Feedback calls use the outer
LM/options, without the task predictor's overrides, and remain in the outer
trace; reward calls do not enter the selected program trace. Cancellation and
streaming stop bypass the failure budget here too. A native executable matched
**14 actual DSPy cases and all 66 recorded prompts and rollout/temperature
settings exactly**, together with outputs, outer traces and persistent failure
budgets. Coverage includes distinct versus shared
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
`termination_reason`, including forced-submit results. If OpenAI generation
returns HTTP 400 with `error.code` set to `context_length_exceeded`, ReAct
keeps its history and tries one forced submit. If that call also exceeds the
window, `error.ContextWindowExceeded` is returned. An ordinary HTTP 400 does
not trigger recovery. This is not a general provider error taxonomy.

A reproduced Refine/ReAct
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
and cyclic or shared container-identity graphs are not provided by these explicit
Zig APIs; custom boxes within a native tree do share the clone context above.
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

For concurrent single-query retrieval, initialize an `Embeddings.Batcher` or
`EmbeddingsWithScores.Batcher` in place and call its `forward` from client
threads. One worker batches up to 32 queries, waiting up to 100 ms from the
start of collection by default. Pending work is bounded to one additional
batch; full queues return `BatchQueueFull`. `close` finishes active work,
rejects queued requests and joins the worker. Join client threads before
reclaiming the batcher, and do not use the borrowed retriever or embedder
directly while the batcher owns access. Output allocations use each caller's
own allocator; returned passages still borrow the corpus.

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
If the options cannot be encoded as a cache key, the batch runs without cache
lookup or storage. This does not suppress backend or allocation errors.
It is not DSPy's shared Python disk cache, and key-encoding limits can differ.

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
For a teacher with a different Zig program type, call
`optimizer.compileWithTeacher(Teacher, arena, gpa, ctx, &student, &teacher)`.
The result still uses the student's input and output types.
Predictor discovery exposes the visible student's namespace, not hidden captures.
Bootstrap mapping checks the live student/teacher instructions, descriptions and
prefixes before reporting an empty training neighborhood.
Both optimizers and embedding output arenas must outlive the values borrowing
them. NumPy sorting/BLAS portability limits are explicit below.

### Ensemble voting and reduction

`Ensemble(P, n).forward` votes on the last output field. Use `forwardAll` for
the unreduced outputs, or `EnsembleWithReducer(P, n, reduce)` to return a
caller-defined result. The reducer gets the context and every output in
execution order, including an empty slice for an empty ensemble. Null or zero
`size` runs every member in order without using the RNG; a positive size
samples without replacement and requires a caller-owned RNG. Negative and
oversized samples fail before running members. Ensemble members share one Zig
program type.

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

Declare a nominal native subclass with
`dszig.decision.Subclass(Base, name, Identity)`, for example:

```zig
const Risk = dszig.decision.Subclass(dszig.Noul, "Risk", struct {});
```

Use a distinct identity tag for independent declarations, even if their names
match. Subclasses inherit criteria, values and native-adaptation eligibility.
A parent accepts its descendants without erasing their concrete boxed class;
children reject parent and sibling instances. Owning clones preserve the
concrete class after the source arena is destroyed. Ancestry is compile-time
metadata, bounded to 64 ancestors, not a runtime allocation or class registry.
Consumer-declared subclasses also retain their non-DSPy ownership for XML:
their structured fields use nested XML rather than the literal JSON leaves
used by DSPy-owned decision models. This is a static native declaration API,
not Python's mutable runtime class construction.

`Model.modelConstruct(arena, fields)` preserves unvalidated keyword fields in
a genuine erased instance of that model class. Required fields may be absent,
and wrong-typed fields retain their actual values, including Decimal and
bytearray boxes. Optional evidence fields default to null. Raw iteration and
Python-mode dumps retain those values; JSON-mode serialization applies the
declared field projections without revalidating the instance.
Value-level validation preserves accepted instance identity, while typed
`Model.fromValue` refuses fields that cannot safely inhabit the typed layout.
`custom.get(Model)` never reinterprets erased raw storage as a typed pointer.
Owning runtime clones retain the concrete class and deep-copy present fields.

For dynamic attributes, `Model.mutable(arena, fields, mode)` validates constructor
fields once and returns a genuine erased mutable instance.
`Model.setField(instance, name, value)` then stores the actual unvalidated value;
`Model.deleteField(instance, name)` removes an existing field. Unknown assignment
and absent deletion refuse. Existing aliases of the instance see the mutation.
Assigned values borrow caller storage, which must outlive their use by the model.
A mutated owning clone retains displaced owned fields until model destruction,
so earlier borrowed field aliases remain valid. This retirement ledger has at
most one slot per declared field and allocates nothing during assignment or
deletion. Copies include current fields, not retired ownership bookkeeping.
Use these APIs for dynamic model behavior rather than erasing a boxed scalar
into a strongly typed `f64` field and expecting Python object identity.

The erased object boolean protocol is fallible:
`try object.custom.truthy()` propagates a refusal, such as an unvalidated Noul
whose `value` is not a bool. Strongly typed model `.truthy()` methods remain
ordinary bool-returning methods.

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
configured decision model for evidence and criteria. Choice criteria must match
the Literal's members **including their Python types**, independently of order;
integer `1` is not boolean `True` for this check. Native string-enum fields use
the same metadata constraints and configured evidence binding. Float/Score
metadata is rejected, matching Source's refusal rather than silently accepting
it. Invalid static signature metadata is a compile-time error, whereas Source
raises during runtime TypeAdapter construction.

`dszig.types.validateAnnotated(T, metadata, arena, input)` and
`dszig.types.schemaAnnotated(T, metadata)` expose standalone validation and
schema generation with that same metadata slice. They return `error.ParseValue`
for invalid combinations before validating the input. Valid values retain the
underlying primitive or model; there is no Annotated wrapper box. Noul/Choice
metadata replaces the schema description without losing the base schema's
constraints or definitions. Score metadata has no description hook. Arbitrary
runtime Python annotations and Python's Annotated descriptor-cache identity are
not implemented.

Explicit Noul/Score/Choice class metadata activates evidence for supported
boolean and Literal outputs, including bare metadata classes. Routing still
follows the base field: a boolean uses Noul; a Literal uses a compatible
configured Choice or its own declared members. The metadata class need not be
that selected class. Fields without decision metadata remain ordinary outputs.
An independent nine-case actual Source state probe exposed 18 incorrect
selection/configuration/criteria observations before repair; native Debug and
ReleaseSafe CLIs then matched all 36 recorded fields. This is finite state
routing evidence, not Python descriptor-cache or complete pipeline equivalence.
The Annotated checkpoint passed **540/540 tests in both Debug and ReleaseSafe**;
the integrated build now passes **38/38 steps**, including two permanent
compile-refusal regressions for invalid typed Choice and float/Score signatures.
Actual RAG/bootstrap/8,192-run Parallel examples and all 687 replay requests
passed. An isolated schema-comparator mutation produced ten mismatches and
`ProofMismatch`; incomplete ordinary consumers still fail with
`UnsupportedSourceRows`, not success.

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

#### Canonical OpenAI physical requests

`lm.OpenAI.requestBody(arena, request)` accepts an owning `sdk.Request` and
returns wire bytes and adaptation records.
`completeRequest(arena, request)` executes that same request and returns an
owning `sdk.Response`; release it when finished. The SDK request's model and
configuration take precedence over the provider's native prediction defaults.
The native prediction entry point constructs a canonical request before entering
the same codec and HTTP transport.

Wire preparation selects fields before refreshing credentials, but retains
references to the selected nested schemas, tool parameters and extension values.
Serialization follows credential refresh: in-place mutations reach the wire,
while replacing a selected top-level entry does not reselect it. Parsing reads
the same current request again. This ordering is observable when credential
callbacks mutate JSON payloads.

Canonical Chat responses refuse multiple returned choices instead of silently
discarding all but the first. An empty choice list becomes an empty text part;
unknown content/tool blocks and finish reasons remain in `_lm15_unmapped`
provider metadata. The explicit all-choice `parseCompletions` helper remains
available separately.

Responses client-side stops use an SSE consumer, not buffered text truncation.
The `completeRequest` path authenticates and serializes the discarded nonstreaming
request before preparing the streaming request. A matched stop closes the unread
connection without waiting for terminal usage; absent usage remains unknown.
Text across part indexes forms one stop stream, while materialized parts retain
canonical index ordering. Final item copies complete fragments without repeating
them, and conflicting copies refuse rather than choosing a version. Complete
streams retain terminal usage and provider data; incomplete transport streams
without a terminal event refuse. The existing response-byte limit also bounds
SSE consumption and synthesized reasoning-summary separators.

Frozen Source comparisons cover 116 whole request-body/refusal cases, actual
HTTP execution, schema mutation and replacement clocks, canonical Chat choice
boundaries, and Responses stop timing, final copies, scores, generated media,
continuations and large part indexes. These are finite localhost protocol
proofs, not authenticated inference.

`OpenAI.streamRequest(backing, request, sink)` delivers typed SSE events for
Responses and Chat Completions synchronously through `OpenAI.StreamSink`.
The callback may return
`.stop` to close the connection without another event. Event data is borrowed
only for that invocation; copy anything that must outlive it. Delta part
indexes are decimal strings so large provider indexes remain exact; a null
index attaches continuation state to the message rather than a part.
Provider error frames are events; an opening, transport, or decoding failure
returns a Zig error. `.exhausted` means delivery finished, not necessarily
that a terminal event arrived. Bare terminators carry neither usage nor a
finish reason. Responses handles stops locally and emits one `.end` with
`.stop` and no usage when cut; Chat forwards stop sequences to the server.
Chat keeps the latest usage-bearing frame as end provider data, ahead of
finish-only frames. Required vLLM candidate scoring is refused before
credentials are fetched; `if_available` streams an unmeasured pick.

This direct provider call prepares a streaming request without the discarded
nonstreaming preflight used by `completeRequest`. It does not assemble a
response, cache results, or publish history. It does not add streaming to
other providers or the general LM interface. Localhost comparisons in Debug
and ReleaseSafe matched eight frozen DSPy Responses and ten Chat event
sequences, exact request bodies, paths, and authorization, plus one Chat
pre-request refusal. Both modes delivered deltas with the tail withheld and
closed the connection before the terminal frame on callback stops and
Responses client-side stop matches.

`lm.responseToEvents(backing, response, sink)` visits a completed SDK
response through the same callback, without HTTP. It emits scores and
their completeness on the first text delta, preserves image addresses and
continuation scopes, and serializes tool arguments with Python JSON
spelling. Unsupported parts and incomplete scores without a text carrier
return explicit errors after the events already delivered; a callback stop
does not inspect later parts.

`lm.materializeStream(backing, request, events)` folds up to 65,536 typed
events already supplied by the caller. It distinguishes a complete response,
a missing end, an event after the end, an unnamed tool, and a provider error.
Missing-end and unnamed-tool results can carry a partial response; an event
after the end preserves the response at the first end. Release the result
with `.release()`. Diagnostic strings borrow the input; the response is owned.
This API does not read from a transport, manage a source iterator, or run a
DSPy program.

`lm.materializeStreamSource` applies the same rules to a synchronous source.
It closes the source exactly once, even when reading, folding, or copying
diagnostics fails. An optional observer can stop before the end without
another read, returning the partial result with `stopped = true`. After an
end, one more read distinguishes a trailing event from source exhaustion.
Read failures after completion and close failures are reported separately
from the primary outcome. Keep event payloads valid until close; the returned
response and diagnostic strings no longer depend on source storage. Release
the report with `.release(backing)`. This API does not turn an HTTP push
stream into a pull iterator or run a DSPy program.

`lm.ChatStreamListener` projects interim text for one ChatAdapter wire field
from a sequence of text chunks. It uses three disjoint caller-provided buffers
and reports `BufferTooSmall` instead of allocating. Feed text deltas to
`.receive()`, then call `.finalize()` when the source ends. The result is
borrowed until the next call. A chunk containing both field markers suppresses
interim output, as in DSPy; it does not prove a cache lookup occurred. This
listener does not parse the final prediction or route chunks to predictors.

`lm.JsonStreamListener` and `lm.XmlStreamListener` also accept text chunks
and write fragments into caller-provided storage. JSON emits raw value text,
without unescaping it. It bounds nesting to 64 levels and top-level keys
to 256 bytes and does not support escaped secondary keys or JSON repair.
XML uses the declared output fields to locate literal boundaries; pass a
provider terminal chunk to `.finishChunk()` (or call `.finish()` for a
terminal without text). Nested targets emit raw text rather than parsed
children. A literal XML listener cannot skip a non-self-closing structured
predecessor. Both listeners leave final prediction parsing to the adapter.

Set `Context.stream_sink` to receive raw events during a normal native
program call. Despite the `OpenAI.StreamSink` type name, Anthropic Messages
and Gemini GenerateContent also use this event interface. The same HTTP
request emits events and assembles the completion for adapter parsing, so the
call still returns its ordinary prediction.
Cache hits return their stored completion without emitting events. On a miss,
each physical generation has its own event sequence. Returning `.stop` aborts
the active generation with `error.StreamStopped` and does not cache a partial
completion. Event payloads are borrowed only during the callback. This is
synchronous raw delivery; use `dszig.callStream` for named field routing.

Bind `dszig.stream.Listener` descriptors with `dszig.stream.Session.init`
before invoking `dszig.callStream`. An implicit binding must identify one
predictor by its declared output field; to disambiguate, provide the
predictor's state pointer. Only selected predictors request streaming.
Listeners use the adapter's prepared output names, including text fallbacks
for reasoning. A declared `Citations` listener emits native citation deltas
as JSON-list fragments and can still receive the rendered text field.
If Chat or XML parsing fails after publishing a field fragment, the call
returns the parse error instead of issuing a JSON fallback request. A parse
failure before publication can still trigger fallback.
Each listener needs disjoint fragment buffers, and XML also
needs storage for the prepared output field table. Keep the program, binding
arena and buffers alive throughout the session.

Fragments are delivered synchronously and borrowed only during the callback.
Returning `.stop` aborts the call. `allow_reuse` resets a listener for
another invocation of its predictor in the same program call. Each new
`callStream` starts with clean listener state. The returned typed prediction
is always available; `emit_prediction` reports whether the streaming caller
should also publish it, after all final fragments. A cache hit can therefore
return a prediction without any fragments. A session cannot be shared across
concurrent calls, and selected predictor generations cannot overlap. This
does not implement arbitrary custom streamable types or routed
TwoStep/native decision streaming.

Set `Context.status_sink` to observe synchronous module, adapter LM, and
tool calls. Its default provider emits DSPy's tool progress messages and
omits `finish` starts and `Completed.` results. A custom provider can inspect
borrowed inputs, outputs, and failures, or suppress any event by returning
an empty message. Status messages can coexist with routed field fragments.
They do not change the underlying result or error.

`dszig.AsyncProgram(Program)` runs a native program in a concurrent Zig
`Io` task. Start it with `.start()`, pull events with `.next()`, and call
`.cancel()` if you stop before exhaustion. With field listeners it emits
routed fragments; without them it emits raw OpenAI events. It can also emit
status messages and the typed prediction. Each event is borrowed until the
next pull or cancellation; the producer waits for that acknowledgement
instead of copying or buffering an entire transcript. Exhaustion or
cancellation joins the task. Keep the program, input backing, client,
allocator, and listener storage alive until then. The iterator has one
consumer and requires an `Io` backend that supports concurrency. It does
not run Python coroutines or expose their cancellation API.

Responses requests honor `config.extensions.openai_responses_compat`, its
`openai_compat` alias and the nested `compat` forms. Null inherits; `"auto"`
selects the dialect default. Direct empty objects win over aliases, whereas
empty nested Responses policies fall back to the generic OpenAI policy.
Invalid recognized policies refuse before credential refresh or HTTP; unknown
fields are ignored as in Source. Fifteen fresh actual Source observations match
in Debug and ReleaseSafe: eight complete SDK responses and ordered HTTP requests,
seven refusals, and all request snapshots and credential counts.

`lm.OpenAI.responses_compat` accepts a typed `lm.OpenAI.ResponsesCompat`, built
with `.init(.{ ... })`. Its twelve optional fields preserve partial policy:
null inherits, while `auto` explicitly resets to the Responses default. Bound
policy is applied before request overrides, so a null routing override retains
the bound object but an empty object replaces it. Routing and compatibility
extensions borrow graph roots; owning LM clones copy both roots together and
retain shared aliases. Native state save/restore owns its ordinary record
projection. Compatibility `extensions` remain metadata, not request-body
overrides, and the image-edit knob does not affect generation requests.

Seventeen fresh actual Source cases matched thirteen complete SDK responses,
four refusal outcomes, all thirteen ordered HTTP path/auth/body byte records,
actual request snapshots and credential counts in Debug and ReleaseSafe.
Ten of those frozen cases were also consumed through the native LM caller:
seven whole SDK responses and HTTP records plus three refusal outcomes matched.
Its unused input SDK record is not an executed SDK-request mutation clock.
The corpus includes inherited/reset/replaced controls, empty routing, constructor
versus request-policy errors, Source's nonfinite bound-routing serialization,
genuine dictionaries with non-string keys, UTF-8 tuple values and a set that
refuses during serialization after credential refresh. The allocation-free graph
writer shares Python dictionary-key encoding without loosening finite-JSON
admission. Permanent regressions check that distinction and build real request
bodies after the source arena is destroyed, preserving independent state
snapshots and shared aliases within an owning clone.

`try provider.bindResponsesPreset("LM Studio", null)` selects the Responses
dialect, binds the Ollama-compatible policy and uses LM Studio's separate root
(`http://localhost:1234/v1`). Pass a non-default URL as the second argument to
target a different deployment. Qwen, DeepSeek and Z.AI have no default Responses
root: binding them without an override refuses without changing the provider.
The exact OpenAI cloud default selects the preset's own default, as in DSPy;
it is not a fallback for missing roots. `lm.OpenAI.ResponsesCompat.preset(name)`
returns the shared partial policy without changing any endpoint.

Both Debug and ReleaseSafe matched 19 frozen DSPy metadata cases for partial
policies, available generation controls and default/explicit address handling.
A separate actual localhost run matched 20 complete SDK responses and ordered
HTTP path/auth/body byte records, two unknown-name refusals, and all request
snapshots and credential counts. It covered every canonical Responses preset,
aliases, Unicode case normalization and a request override of a bound preset.
It did not contact the default server addresses or exercise endpoint profiles.

`try provider.bindChatPreset("xai", null)` selects Chat Completions, its
default root and xAI's separate reasoning-token accounting. Qwen, Bedrock and
Bedrock Mantle require an explicit non-default root; a failed bind leaves the
previous client settings intact. To bind a custom policy, set
`lm.OpenAI.chat_compat` to a partial `lm.OpenAI.ChatCompat`. A matching
case-sensitive model prefix applies only the first matching override, using
the request's model rather than the client's fallback model. `auto` resets a
knob to the Chat default. A bound Chat policy also replaces the standalone
`chat_reasoning_usage` override. Native LM clones and state restore own the
policy's borrowed inputs. Chat does not treat request compatibility extension
keys as policy overrides.

For Azure OpenAI **v1** generation, use `lm.OpenAI.initAzure(arena, &client,
options)` with `.endpoint`, `.deployment`, and `.credential`. The endpoint
can be the Azure resource root, a gateway prefix, or the full `/openai/v1`
base. It defaults to Responses, like DSPy's Azure `model_type="auto"`.
Set `.api = .chat_completions` for Chat Completions.
The deployment name goes in the request's `model`; no legacy `api-version`
parameter or `/deployments/` path is added. Use `.credential =
.{ .api_key = key }` for a static credential: opaque subscription keys use
`api-key`, while JWT-shaped and `ya29.` token strings use `Authorization: Bearer …`.
Use `.credential = .{ .entra = provider }` to supply or refresh a bearer
token on each attempt. Keep the startup arena and borrowed
credentials alive while the LM is in use. Do not rebind a credential-bearing
LM to another host. Entra credential discovery and Azure management endpoints
are not included.

`completeRequest` uses MAP-14 candidate scoring on Chat policies that declare
`logprob_token_ids` (including vLLM) when `probabilities` is `required` or
`if_available` and a JSON schema declares measurable judgments. It tokenizes
the conversation and candidate answers with the server’s chat template, then
scores their shared prefixes in one Completions request. Ordinary properties
and box judgments still come from structured generation. If the server omits
requested token scores, `required` fails; `if_available` can fall back to
strict structured generation when the adaptation policy permits it. Tool
execution, stored-cache resources and unsupported privacy or billing controls
are refused before credentials or HTTP are used. The explicit score-prompt
array is capped at 16 MiB.

A localhost comparison against 14 frozen DSPy scoring cases matched all three
complete SDK responses, 11 refusal outcomes, 64 ordered HTTP path/auth/body
records, and all 68 credential calls in both Debug and ReleaseSafe.

Debug and ReleaseSafe matched 27 frozen DSPy preset cases, including 20
model-specific resolutions. A separate localhost comparison matched 46
complete SDK responses and byte-exact HTTP path/auth/body records, one
unknown-name refusal, and all request snapshots and credential counts.
This did not contact hosted servers or authenticate to the homelab.

The canonical Chat wire builder also implements client-side tool exclusion and
allowlists for servers that ignore forced choices, recorded JSON-schema
dropping, whole-reasoning-dial dropping before subfield adaptations, nearest
effort clamping with lower-level tie breaking, first-declared-level substitution
for reasoning off, and Groq reasoning visibility. A separate 29-row actual
Source corpus matched 20 complete wire bodies and ordered published adaptation
lists, nine acceptance/refusal outcomes, and request-state outcomes in both
native modes. This is a direct wire-builder comparison, not a whole-response or
native HTTP execution proof for bound presets; those presets are not exposed
yet. Canonical SDK request record import separately preserves legacy reasoning
defaults, budget aliases and off normalization without weakening genuine typed
constructor validation.

The flattened native caller now replays multiple no-summary reasoning states
as separate canonical thinking parts instead of silently sending only the first
opaque state. A nonempty flattened summary with multiple states still refuses:
the flattened representation cannot identify which state owns that summary.
Three fresh two-turn Source comparisons matched both complete SDK responses and
all six ordered HTTP requests in Debug and ReleaseSafe, including absent
summaries, empty summaries and a single summarized state.

`lm.TypeSafe` / `TypeSafe` uses the existing bounded native HTTP transport for
`POST /v1/systemone`. Both physical entry points now take an actual canonical
SDK request: `requestBody(arena, request)` returns wire bytes, header selection
and adaptation records;
`completeRequest(arena, request)` returns the actual owning SDK response.
Release that response when finished; its allocator and any retained payload
backing allocators must outlive it. The native prediction LM constructs its SDK
request **before** entering the same codec/transport/parser, rather than
reconstructing a request after legacy execution. Native decision state,
questions and schema still survive recording/replay/cache wrappers, and decision
calls forward the same explicit tracker and accounting-error capture. OpenAI
still refuses direct native decision requests rather than silently dropping them.

For a public call, `LM.callCanonicalResult(arena, io, request, options)` sends
one SDK request through `CachedLM` and the normal usage and pricing finalizer.
The configured model must match `request.fields.model`. Local media is frozen
before cache lookup with an explicit file-size limit, and the engine uses the
request's own config without applying prediction defaults. `use_cache = false`
bypasses `CachedLM`; `rollout_id` separates otherwise identical calls. The
existing backing-allocator rule applies to response payloads retained by the
cache. OpenAI and TypeSafe support this entry point; custom engines can provide
`LM.VTable.complete_canonical`. The legacy `RecordingLM` transcript does not
encode SDK requests and rejects these calls before generation.

Set `CanonicalCall.history` to a `lm.public_history.Capture` to record successful
canonical calls, including cache hits, after usage accounting. Initialize the
global owner with 10,000 pointer slots and supply client and module owners
with enough slots for `owner_limit`. Each owner retains the same entry, with
the base request, the ordered `Entry.responses`, and shared output and usage
graphs. A zero owner limit skips the client and
modules; a null capture disables recording altogether. Lowering a limit
evicts one old entry per insertion, and repeated module owners are processed
in order. Retain an entry before eviction if you need to use it afterward.
Keep all backing allocators alive for as long as references survive and
synchronize owner access and graph mutations externally. Timestamps use the
injected real clock and display in UTC without a zone suffix; UUIDs use its
secure randomness API. This option does not record legacy adapter calls.
Use `Entry.setAnnotation`, `annotation`, and `removeAnnotation` to share
caller-added fields across the same history entry. An entry supports up to
64 annotation keys of at most 256 bytes each. Keys are copied; values with
owned payloads are retained, and borrowed values are cloned. An annotation
lookup borrows the value until replacement, removal, or entry release.
Annotations do not replace typed history fields or populate Python `kwargs`.

`LM.callPromptResult(arena, io, prompt, options)` takes UTF-8 text and a
normalized SDK `Config`. It sends one user message per candidate, up to
`LM.PromptCall.max_candidates`, adding the zero-based candidate index to a
non-null seed. Cache entries distinguish candidate counts, and one history
entry retains the original prompt and all ordered responses. The returned
`CallResult.outputs` and the history entry share the same mutable output list.
Supply the resolved generation config explicitly, or use
`LM.callPromptOptionsResult(arena, io, prompt, &options, execution)` to merge
`LM.PromptOptions.defaults` and `.overrides` JSON objects. This second API uses
generic Chat generation spellings (and integer `logprobs` for `.text`), replaces
aliases as a group, and treats null overrides as removing defaults. Connection
settings belong on the LM; caching and rollout identity belong in `execution`.
Provider-specific ingress dialects, Python model classes as schemas, and dynamic
`kwargs` history are not supported by this API. For provider-specific options,
pass a normalized SDK `Config` or `Request` instead.

The canonical TypeSafe codec derives questions from judgment schemas, preserving
instructions, choice descriptions and ordered-level fallback labels. State is
one user text/data part, or an ordered mixture containing images. Inline/local
PNG and JPEG images become named `assets`; image and box requests send the
experimental header. Local reads have an inclusive, configurable `max_media_bytes`
bound (16 MiB by default), unlike Source's unbounded reads. System prompts,
conversation, unsupported media, tools and multiple-response extensions refuse.
Extensions overwrite derived payload fields in order, and header selection uses
the final payload, including Source's assets/box short-circuit behavior.

Dropped configuration knobs and defaulted question descriptions produce canonical
adaptation records. `adaptations = .note` exposes them, `.silent` preserves wire
behavior while hiding them on responses, and `.refuse` rejects dropped controls
before HTTP but allows defaults. Model fallback uses the canonical request model,
not the native prediction binding; response usage, raw answers and the actual
request-ID header are retained.

Both native modes matched **32 frozen Source wire observations**: 15 complete
successful bodies/header decisions/adaptation lists and 17 accept/refuse outcomes.
Errors remain native `LMFailure`, not identical Python exception classes. Actual
localhost HTTP also exercised both SDK and native prediction paths: five requests
per mode matched the corresponding conserved/new Source bodies. Fresh Source
image calls verify note/silent behavior and unchanged caller records; a separate
complete call matches the full SDK record, including raw provider data and all
15 adaptation records (14 dropped controls and one defaulted description).
A further actual Source credential callback changes a boolean schema to a choice
after the old questions have been serialized. Native execution first failed on
the stale schema, then matched the complete Source response, mutated caller
record, credential-call count and original wire bytes in both modes. Response
parsing now rereads the actual SDK request rather than retaining a build-time
schema snapshot; it recognizes the current judgments without repeating admission
rules. Permanent regressions cover that parser/admission distinction and the
inclusive local-image byte cap without changing the caller's request.
These are finite protocol proofs, not authenticated
TypeSafe inference or the complete provider SDK. Public LM/history integration,
asynchronous execution and streaming remain unfinished.

The frozen independent DSPy 3.5 source receipt contains **15,218 actual rows
and 17 model definitions**. Both consumers were re-run in native Debug and
ReleaseSafe after SDK metadata and canonical Prediction integration, without
rerunning Source. The model consumer now observes singleton and genuine boxed
builtin identity where the runtime exposes it. The pipeline consumer supplies
metadata from the actual recorded SDK responses, not expected prediction usage.
These separate consumers report finite coverage; their entries/observations
must not be added together:

| Independent consumer | Fully supported entries | Unsupported entries | Unconsumed entries | Compared observations | Mismatches |
|---|---:|---:|---:|---:|---:|
| Decision models and views | 9,114 | 0 | 6,121 | 95,462 | 0 |
| Evidence/state/Predict pipeline | 2,095 | 285 | 12,838 | 18,154 | 0 |

Both consumers deliberately **exit nonzero for incomplete coverage**.
Partially compared unsupported rows contribute observations, not a claim that
their entire operation is supported. Six candidate-client rollout observations
are unobserved: the source engine receipt does not capture those client-only
options. The native catalogue now includes all 17 declarations, including the
three simple subclasses, and consumers reconstruct genuine Decimal and
bytearray inputs rather than JSON lookalikes. Unvalidated model construction
now preserves raw state without unsafe typed casts. The current model ledger
has zero unsupported observations and 8,087 unconsumed observations. Native
identity checks include `None`/bool singletons, genuine Decimal/bytes/bytearray
boxes, and independently observable copied evidence; erased int/string object
identity is still not inferred from equality. The source oracle explicitly
guards dictionary inputs before its value-identity check: parsing a raw JSON
string does not turn that original input into a dictionary. All 304
Annotated validation rows were exercised (322 acceptance/result observations),
as were all 16 schema rows (26 comparisons). Six schema-error exception-identity
observations remain unconsumed; native errors are not Python exception objects.
The actual Source descriptor for the reordered Choice case already has the
original member order because of Python's Annotated cache; the native regression
separately verifies order-independent typed-set compatibility. The earlier
unconsumed-row counter did not separately budget 17 newly compared
acceptance/result observations, so historical observation totals are not a
conservation ledger. The current model ledger retains all 103,549 budgeted
observations, compared or explicitly unconsumed.

The pipeline separately checks 42 native invariants; first-row value/class
equality is **not** counted as Source's stronger object-identity observation.
Canonical usage is read from the real Prediction, and SDK response ID, model,
finish reason and sparse usage are checked through actual recording. Source
history usage/model values can be compared to those retained SDK facts and the
configured LM getter. `lm.sdk.Request.historyMessages(allocator)` provides the
canonical display projection: text-only messages join with newlines, structured
parts retain their mutable graph payloads, and local file paths are not read.
The returned payload owns its display headers and must be freed by the caller.
In Debug and ReleaseSafe it matched three frozen DSPy display records, including
a mutation of a shared data part after the display was created.
This does not implement Python's public history publication, owner retention, or
cost/pricing/model-type entries. The separate advisory physical-request billing
API does not turn those incomplete pipeline rows into matches.
Dynamic Completions, Python Prediction class
identity, private adapter entrypoints, warnings and async/streaming remain gaps.
Isolated observation-corruption controls were compiled and executed in both
modes: boxed identity, singleton identity, SDK usage loss and Prediction usage
loss each produced actual mismatches and a nonzero mismatch exit, not merely
the normal incomplete-coverage exit. Source bytes were unchanged.

`adapter.formatFieldContent(arena, field, input)` implements the canonical
`format_field_value(..., assume_text=False)` text block. It uses the existing
field serializer, with both the returned dictionary and rendered text owned by
the supplied arena. Pass the result directly to `adapter.Content.parts`; no
second custom-value representation is introduced. All six frozen formatting
rows now consume all 36 observations, including the text-block view and the
distinction between a genuine model and a lookalike dictionary. Erasing the
observed block text in an isolated consumer produced six real mismatches in
each native mode. These replace six previously unconsumed observations, not
additional Source execution or a larger observation budget.

A separate Debug/ReleaseSafe native CLI exercised owned text after input
mutation, content cloning, SDK projection, actual Chat/Responses request-body
serialization for user and assistant roles, and rendering/container allocation
refusal with fixed-capacity caller storage. It made no network requests.
The role-policy regression and complete native gates passed: 574/574 tests
and 38/38 build steps per mode, plus all 687 replay requests. Fixed-capacity
smoke storage does not establish whole-runtime startup-only allocation.

Scalar-only genuine decisions now stream through the existing JSON writer
without allocating a temporary model dictionary. Eligibility inspects the
actual decision fields; nested containers and custom values keep the
materialized serializer, preserving whole-object string fallback rather than
emitting partial JSON. Two newly observed Source formatting boundaries used
genuine invalid-UTF-8 bytearrays directly and inside a list. Both native modes
matched both results; bypassing materialization produced two actual mismatches
in each mode. These are separate observations, not a rerun of the frozen
15,218-row corpus.

A ReleaseSafe native microbenchmark used runtime input `0.5`, seven alternating
samples of 100,000 Score renders per path, fixed caller scratch storage and
matching output checksums. The table reports median elapsed time and the
scratch high-water mark **per render**, not total process memory:

| Scalar Score rendering path | Scratch bytes per render | Time for 100,000 renders (ms) |
|---|---:|---:|
| Materialized projection | 608 | 29.72 |
| Scalar streaming | 0 | 22.86 |

This is one rendering boundary, not general DSPy throughput or whole-runtime
allocation freedom. The independent frozen model and pipeline consumers were
recompiled and executed in both modes after this change; their counters above
were unchanged, including explicit nonzero incomplete-coverage exits.

All five frozen assignment/deletion scenarios exercised
real native mutable operations:
110 observations matched, with five missing-argument views still unconsumed.
Python type-object reflection, mutable runtime class construction, broader scalar
identity/reflection operations and other unconsumed operations remain explicit
gaps. An independent focused Source probe and native
Debug/ReleaseSafe consumers additionally matched all 24 XML prompt structures
and routing decisions across direct, optional, list and optional-list-item
annotations of the three roots and three subclasses. A separate eight-case
Source probe matched 40 exact raw str/repr/Python-dump/JSON observations in each
mode. It exposed and verified the repair for Score's Python-string key fallback
versus Choice's JSON-key spelling; Python serialization warnings remain
unobserved. ScriptedLM now refuses non-JSON provider data at response
construction, before adapter decoding or evidence recording; this is not a
complete SDK response-object API. Bare Score/Choice evidence refusal also
differs in failure timing:
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

Both current decision consumers produced the same counts in Debug and
ReleaseSafe, including the deliberate incomplete-coverage failures. The raw
model and fallible-boolean changes passed pinned **525/525 tests and 36/36 build
steps in each mode**, actual RAG/bootstrap and 8,192-run Parallel examples,
plus the complete 687-request native replay. A separate opaque-layout smoke
reproduced incorrect equality before repair and safe bidirectional equality
afterward, without permitting typed extraction. An isolated old-XML-routing control failed on the
actual `SubNoul` prompt instead of passing the new surface comparison.
The older bytearray checkpoint additionally ran the independent Python
fixture/artifact/reference gate and unchanged Unicode/DPR regeneration;
those Python gates were not rerun for these native class changes.

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
When calling `PredictState.call` or `callMany` with a record, omitted input
fields use their declared signature defaults in the prompt and trace. An
explicit `null` stays `null`; the caller's record is not modified.
Typed programs run from an `Example` also use declared input defaults for
omitted arguments. Fields not marked as inputs are not passed to the program,
even if a label has the same name as an input.

When `Context.trace` is set, `Context.max_trace_size` (default 10,000)
controls predictor appends. A successful call at or above the limit removes
one oldest entry before recording the new entry. A nonpositive limit leaves
the trace unchanged. This applies to predictor calls inside optimizers as
well, but not to explicit appends or winner-trace merges. It limits retained
step headers, not the memory already allocated in the trace arena.
A per-call `_trace` entry in a dynamic predictor input record is evaluated after
successful inference. A falsey value skips the append; a truthy value allows it
but is omitted from the stored inputs. The caller's input record is unchanged.

Set `Context.demos_override` on an individual predictor call to replace its
stored demonstrations without changing the program. Use `&.{}` for a zero-shot
call or `null` to use the stored demonstrations. In a composed program, pass
the override only to the child calls that should receive it.

For DSPy-style predictor configuration, set `predictor.state.config` to a
borrowed, immutable `PredictConfig`. Its supplied fields override
`Context.lm_options`. Set `Context.predict_config` on an individual call to
override selected fields again, without modifying the predictor. An omitted
field inherits; for nullable fields use a typed null (e.g.
`@as(?dszig.Value, null)` for `rollout_id`) to clear an inherited setting.
Configuration lives in the program definition and is not included in
state-only snapshots.

Set `PredictConfig.cache = false` to bypass both lookup and storage in an
installed `CachedLM`, for a stored predictor or a single call. Set it to `true`
to use that cache, or to `@as(?bool, null)` to clear an inherited override.
The bypass does not replace an existing entry or change request identity.
This controls the native response cache, not the provider's prompt cache, and
does not install caching on an uncached LM.

Verification: pinned DSPy 3.5.0 `Predict` with a local counting engine
confirmed that bypass skips lookup and storage, and that a per-call `true`
or explicit null can supersede a stored `false`. The native public
`Predict.call` smoke returned `original`, `fresh`, `original` from two
physical calls. The retained regressions cover precedence, cache preservation,
and ReAnchor's refusal when caching is required. Debug and ReleaseSafe each
passed 864 tests and 40 build steps; strict replay consumed all 687 requests
across eight adapter cases and nine pipelines. Formatting passed. No local
blocker remained; the full Python reference gate and authenticated inference
were not rerun for this client-side change.

Set `PredictConfig.max_tokens` to cap generated output for a stored predictor
or a single call. The per-call setting overrides the stored predictor, then
`Context.lm_options.max_tokens`, then the provider default. Omit the setting to
inherit; use `@as(?u32, null)` to remove the configured cap for that request.
OpenAI sends the effective cap as `max_completion_tokens` for Chat Completions
and `max_output_tokens` for Responses under the default wire policies. The
override participates in cache and replay identity. It is not a total context
limit, and TypeSafe decision calls reject it.

Verification: pinned DSPy 3.5.0 `Predict` sent caps of 24, 48, no cap, and
96 on each OpenAI endpoint over loopback. The retained native regression
checks the same cases plus a context cap, cache hits for each distinct
setting, and out-of-order replay after option metadata round-trip. A standalone
native `Predict` sent 24 on its overridden call and 96 on the next call.
`zig build live -- --smoke` also passed over loopback without an override.
Formatting passed. Debug passed all 40 build steps, with all 89 behavior
tests rerun and 776 other passing results reused; ReleaseSafe passed all 865
tests and 40 steps. Strict replay matched all 687 requests across eight adapter
cases and nine pipelines. No local blocker remains. The full Python reference
gate, other providers' token-cap wire behavior, and authenticated inference
were not checked for this change.

Set `PredictConfig.stop = &.{"STOP"}` to stop generation for a stored predictor
or a single call. A per-call list replaces the stored list, which otherwise
overrides `Context.lm_options.stop`. Omit the setting to inherit or use
`.stop = &.{}` to disable inherited stops. The effective list participates in
cache and replay identity, including its order. TypeSafe decision calls reject
nonempty stop lists.

OpenAI Chat sends the list as `stop`. OpenAI Responses instead streams and
closes the connection at the first match, including a match split across
deltas. A cut does not report terminal usage, and closing the connection does
not guarantee that the provider stops generating or billing.

Verification: pinned DSPy 3.5.0 `Predict` confirmed per-call, stored, cleared
and LM-default stop lists with a local engine; its vendored OpenAI codecs
establish the wire behavior. A standalone native typed `Predict` passed over
loopback in Debug and ReleaseSafe: Chat received both sequences in order,
and Responses returned the typed answer and closed the connection before the
peer sent any terminal frame. Retained regressions cover precedence, cleared
stops, cache separation, out-of-order replay, and token-score coverage on a
Responses cut. Formatting passed. Debug and ReleaseSafe each passed all 866
tests and 40 build steps; strict replay consumed all 687 requests across eight
adapter cases and nine pipelines. No local blocker remains. The full Python
reference gate, other providers' wire behavior, and authenticated inference
were not run for this change.

Set `PredictConfig.top_p` for nucleus sampling. A per-call setting overrides
the stored predictor setting, which overrides `Context.lm_options.top_p`.
Omit it to inherit or use `.top_p = @as(?f64, null)` to leave sampling
unspecified on that request. OpenAI Chat Completions and Responses send
`top_p` when set, including `0`; the canonical generation config accepts
the inclusive range `[0, 1]`. The effective value participates in response
cache and replay matching. Typed TypeSafe decision calls reject it.

Verification: pinned DSPy 3.5.0 `Predict` sent `0`, `0.5`, no field, `0.75`,
and `1` on each OpenAI endpoint over loopback and rejected `1.1` and NaN
before HTTP. A standalone native typed `Predict.call` confirmed call
override, stored value, explicit clearing, and pre-HTTP refusal of both
invalid values on both endpoints. The retained loopback regression also
checks context inheritance, the upper endpoint, cache separation, and
out-of-order replay after metadata round-trip. Formatting passed. Debug and
ReleaseSafe each passed 866 tests and all 40 steps; native and pinned Python
replay each consumed all 687 deterministic requests. No local blocker remains.
Other providers' wire behavior, authenticated inference, and the full
Python reference gate were not checked for this change.

Set `PredictConfig.seed = .{ .integer = 42 }` for best-effort seeded sampling.
A per-call seed overrides the stored predictor seed, which overrides
`Context.lm_options.seed`. Omit the setting to inherit, or set
`.seed = @as(dszig.Value, .null)` to clear it. Integral floats normalize to
integers; booleans and fractional floats are rejected. Candidate fanout uses
`seed + candidate index` without narrowing to `i64`; the effective seed is
part of the cache and replay identity. Native TypeSafe decision calls reject
a non-null seed.

OpenAI Chat Completions sends the seed, including zero. OpenAI Responses has
no seed field: it drops the seed under the normal adaptation policy, and
refuses the request under `.refuse`. Even where supported, a seed does not
guarantee repeatable hosted outputs.

Verification: pinned DSPy 3.5.0 `Predict` and the retained native typed
`Predict.call` regression exercised eight physical loopback requests per
endpoint. They covered override precedence, clearing, zero, a negative seed,
candidate fanout across the `i64` boundary, Responses drop records, and
rejection of boolean and fractional seeds. The native regression also checks
cache separation and out-of-order replay after JSON metadata round-trip. A
standalone typed consumer confirmed candidate order, clearing, and strict
Responses refusal without an HTTP request. Formatting passed; Debug and
ReleaseSafe each passed all 867 tests and 40 build steps. Native and pinned
Python replay each consumed all 687 deterministic requests. No local blocker
remains. Other providers' seed behavior, authenticated inference, and the
full Python fixture-regeneration gate were not checked.

Set `PredictConfig.frequency_penalty` and `PredictConfig.presence_penalty`
independently to control repetition. Values must be finite and in `[-2, 2]`.
For each penalty, the per-call setting overrides the stored predictor setting,
which overrides `Context.lm_options`. Omit a setting to inherit it, or pass
`@as(?f64, null)` to clear it. Zero is an explicit setting. Both penalties
participate in cache and replay identity.

OpenAI Chat Completions forwards both penalties. OpenAI Responses omits them
and records a drop for each supplied penalty by default; `.refuse` rejects
such a request. Native TypeSafe decision predictions reject non-null penalties.

Verification: pinned DSPy 3.5.0 and a standalone native typed consumer matched
eight loopback requests across both OpenAI dialects. The retained typed
`Predict.call` regression additionally checks each penalty independently for
cache separation, out-of-order replay after JSON metadata round-trip, drop
records, strict Responses refusal, and rejection of invalid numbers.
Debug and ReleaseSafe each passed all 868 tests and 40 build steps. Native
and pinned Python replay each consumed all 687 deterministic requests.
No local blockers remain. This change was not checked against authenticated
providers or non-OpenAI wire endpoints; the full fixture-regeneration gate
was not run.

Set `PredictConfig.top_k = .{ .integer = 40 }` to limit sampling to the top
40 candidates. The per-call setting overrides the stored predictor setting,
which overrides `Context.lm_options.top_k`. Omit it to inherit, or set
`.top_k = @as(dszig.Value, .null)` to clear it. The cutoff must be a positive
integer; integral floats normalize to integers and booleans are rejected.
The effective cutoff participates in cache and replay identity.

Native Anthropic Messages sends `top_k`; Gemini GenerateContent sends
`generationConfig.topK`. Both OpenAI dialects omit it and record a drop
by default, or reject the request under `.refuse`. Native TypeSafe decision
predictions reject a non-null cutoff.

Verification: pinned DSPy 3.5.0 and the retained native typed `Predict.call`
regression exercised five loopback requests per dialect, covering precedence,
clearing, integral floats, integers beyond `i64`, invalid values, and OpenAI
drop/refuse behavior. The native regression also verifies cache separation
and reverse-order replay after a JSON metadata round trip. A standalone typed
consumer that previously failed to compile sent `top_k: 7` to `/v1/messages`
and parsed the answer. Formatting passed; Debug and ReleaseSafe each passed
869 tests and 40 build steps. Native and pinned Python replay each consumed
all 687 deterministic requests. No local blockers remain. Authenticated
inference and the full fixture-regeneration gate were not run.

Set `PredictConfig.store = false` to request that the provider not store a
retrievable response. The per-call setting overrides the stored predictor
setting, which overrides `Context.lm_options.store`. Omit it to inherit, or
set `.store = @as(?bool, null)` to clear it and leave the provider default.
OpenAI Chat Completions, OpenAI Responses, and Gemini GenerateContent send
explicit `false` and `true` as JSON booleans and omit an unset value.
Anthropic Messages has no retrievable response object, so it omits the wire
field. Its default `.note` policy reports `false` as `satisfied` and `true`
as `dropped`; `.refuse` rejects `true` before transport and accepts `false`.
Native TypeSafe decision predictions reject an explicit `store` setting.

This setting is not a guarantee about provider logging or data retention.
It does not disable the local `cache` or recording; the effective `store`
value participates in cache and replay identity.

Verification: pinned DSPy 3.5.0 and a standalone typed Zig consumer matched
eight loopback requests across both OpenAI dialects, covering context and
predictor defaults, explicit clearing, and a per-call `false` overriding a
stored `true`. The retained typed regression also checks cache separation
and reverse-order replay after a JSON metadata round trip. Formatting passed;
Debug and ReleaseSafe each passed 870 tests and 40 build steps. Native and
pinned Python replay each consumed all 687 deterministic requests. No local
blockers remained in that OpenAI slice. Provider-side retention,
authenticated inference, and full fixture regeneration were not checked.

Set `PredictConfig.user_id` to a nonempty, opaque identifier for end-user
attribution. The per-call value overrides the stored predictor value, which
overrides `Context.lm_options.user_id`. Omit the setting to inherit it, or
pass `@as(?[]const u8, null)` to clear it. The effective identifier separates
local cache entries and is retained in recorded request metadata, so use a
pseudonymous identifier rather than an email address or other personal data.

OpenAI Chat Completions sends `user` by default (or the field selected by
Chat compatibility policy), OpenAI Responses sends `safety_identifier`, and
Anthropic Messages sends `metadata.user_id`. Gemini GenerateContent omits
the identifier and records a drop by default; `.refuse` rejects the request.
Native TypeSafe decision predictions reject a non-null identifier.

Verification: pinned DSPy 3.5.0 provider calls established the wire mapping,
omission and Gemini adaptation behavior over loopback HTTP. The retained
native typed `Predict.call` regression covers five provider configurations,
override precedence, explicit clearing, empty-identifier rejection, Gemini
drop/refusal, cache separation, and reverse-order replay after a JSON metadata
round trip. A standalone typed consumer sent `safety_identifier` to a
loopback Responses endpoint and parsed its answer. Formatting passed; Debug
and ReleaseSafe each passed all 871 tests and 40 build steps. Native and pinned
Python replay each consumed all 687 deterministic requests. No local blocker
remains; authenticated provider behavior and full fixture regeneration were
not checked.

Set `PredictConfig.service_tier = "priority"` to request a provider service
tier. The per-call setting overrides the stored predictor setting, which
overrides `Context.lm_options.service_tier`. Omit it to inherit, or pass
`@as(?[]const u8, null)` to clear it. Tier names are nonempty,
provider-defined strings, not a native enum. The effective tier separates
cache entries and is retained in replay metadata. When pricing is enabled,
the requested tier is a fallback if the response does not report the served tier.

Both OpenAI dialects and Anthropic Messages send `service_tier`; Gemini
GenerateContent sends top-level `serviceTier`. TypeSafe System One omits it
and records a drop by default, or refuses it before transport under `.refuse`.

Verification: pinned DSPy 3.5.0 and the retained native typed regression each
exercised 20 loopback requests across those five paths. The native regression
also verifies override precedence, explicit clearing, empty-string rejection,
cache separation, reverse-order replay after a JSON metadata round trip, and
served-versus-requested pricing precedence. A standalone typed executable sent
`service_tier: "priority"` to a loopback Responses endpoint and parsed the
answer. Formatting passed; Debug and ReleaseSafe each passed all 872 tests and
40 build steps. Native and pinned Python replay each consumed all 687
deterministic requests. No local blocker remains. Hosted tier availability
and billing were not checked, and the full fixture-regeneration gate was not run.

The buffered Anthropic/Gemini `PredictConfig.store` check matched pinned
DSPy 3.5.0 across seven loopback POSTs and a pre-transport Anthropic refusal.
The retained typed `Predict.call` regression covers Gemini's explicit
`false`, `true`, and omission, Anthropic's `satisfied` and `dropped` records,
and a per-call `false` overriding stored `true` under Anthropic `.refuse`.
It also checks distinct cache and replay entries even when Anthropic sends
the same wire body. No provider behavior change was needed. The focused test
and formatting check passed. Debug and ReleaseSafe each passed all 40 build
steps, with 96 behavior tests rerun in each mode and other test targets reused
from cache. No local blocker remains; hosted retention was not tested, and
the offline fixture-regeneration gate was not rerun.

Anthropic SSE verification: pinned DSPy 3.5.0 and the retained native typed
`Predict.call` regression each exercised five loopback POSTs and one refusal
before transport. All successful POSTs set `stream: true` and omit `store`.
The stream start publishes `satisfied` for `false` and `dropped` for `true`
under `.note`; explicit clearing and `.silent` leave the adaptation list
empty. Under `.refuse`, `true` emits no events and does not reach header
preparation; a per-call `false` overrides stored `true`, publishes `satisfied`,
and returns the typed answer. A standalone native consumer confirmed the
strict-policy pair with exactly one POST. No runtime change was needed.

Formatting passed. Debug and ReleaseSafe each passed all 40 build steps and
97 behavior tests, with the other test targets reused from cache. The first
ReleaseSafe run failed the existing output-cap/nucleus-sampling test without
diagnostic output; it passed on rerun, and the cause is unresolved. No local
blocker remains for this slice. Hosted retention and fixture regeneration were
not checked. Next, verify that typed streaming sends `false`, `true`, and
omission on OpenAI and Gemini; this slice covers only Anthropic SSE.

For OpenAI inference, set `Context.lm_options.prediction` to the predicted-output
object. A record-based `Predict` call also moves a `prediction` argument with
`type: "content"` and a `content` key into the LM options, overriding an
explicit hint. The argument is removed before signature defaults are applied and
is not recorded as a trace input. Other values named `prediction` remain
ordinary inputs. The hint participates in cache and replay identity; whether
the hosted model accepts it depends on the endpoint and model.

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

The public handle's `model_type` is persisted separately from the provider's
wire `api`: restoring text/chat output semantics must not switch a Responses
endpoint. Four new frozen Source dump/reconstruction observations (`auto`,
`chat`, `text`, `responses`) matched native program save/load in Debug and
ReleaseSafe, with Responses request generation unchanged in every case. The
Source BaseLM reconstruction used explicit custom-class trust; native loading
does not import Source classes. The existing native lifetime regression also
checks that a text-mode handle survives both trusted and sanitized restoration
after the source arena is destroyed.

Custom `LM.StateLoader` implementations provide `ptr` and `restore_engine`;
call `loader.restore(arena, saved_state)` to restore the engine and apply the
saved public mode. Program loading uses that same entry point.

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

`dszig.Anthropic.StateLoader` and `dszig.Gemini.StateLoader` use the same
pattern. They save model settings but never save or restore the endpoint,
credentials, HTTP client, or credential callback. The loader supplies those
resources, even when `allow_unsafe_lm_state` is enabled. For Gemini Vertex or
Express, supply the corresponding `door` and fully constructed `base_url`
together. The runtime resources must outlive the loaded program. These are
native provider records, not Python provider records or saved Live sessions.

For a program with bound predictors from several native providers, pass
`dszig.state.ProviderLoaders.loader()` instead. Set `openai`, `anthropic`, or
`gemini` to pointers to the application-configured loaders for the providers you
allow. An unknown or unconfigured provider fails with `UnsupportedLMState`.
Each binding is restored with its own provider's runtime endpoint and credentials.
The provider loaders need to exist during loading; the resources they inject
must outlive the loaded program.

Use `.{}` when loading programs without bound models. A nonempty LM state
without a loader fails with `error.MissingLMLoader`, never a fake restored model.
By default, loading removes the top-level `api_base`, `base_url` and `model_list`
routing keys and uses the explicitly injected endpoint. Loaders that support
saved routing, such as OpenAI's, can preserve it only with
`.allow_unsafe_lm_state = true` for trusted files. Credentials always
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
caller callback can delay the barrier. `release()` cancels any remaining
nonterminal remote job and deletes the training file. A 404 when looking up the
job means it is already absent, so `release()` skips cancellation and still
deletes the file. A 404 on file deletion likewise clears the file ID. Other
cleanup failures retain the remaining IDs and refuse another `start`.
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

Random search inherits `Context.max_errors` unless its own `max_errors` is
set. The same limit governs each candidate's bootstrap and validation runs,
with a fresh failure count for each run.

`BootstrapFewShotWithRandomSearch.compile(..., valset, options)` accepts
`restrict: ?[]const i64` and `labeled_sample: bool`. Restrictions are a borrowed
seed set: duplicates and out-of-range entries are ignored, candidates retain
ascending seed order, and no matching seed returns `RestrictMatchesNoSeed`
before candidate work. Restricted runs visit only matching seeds; a large gap
between seeds does not add iterations. `labeled_sample` changes only the
labeled seed `-2`.
`BootstrapFewShot` and random search accept signed `i128` demo limits. A
nonpositive bootstrap limit produces no traces for the fixed bootstrap seed;
a shuffled seed instead returns `InvalidBootstrapRange` when reached. A
negative labeled limit slices the training prefix at seed `-2` when
`labeled_sample = false`, and otherwise fails if sampled on a nonempty
predictor dataset. Bootstrap also samples labels when preparing an uncompiled
teacher; a compiled teacher skips that step, and a negative limit adds no raw
demos to the student. Early stopping can avoid a later seed's failure.
Teacher-list refusal is likewise deferred until a bootstrap candidate
actually consumes the teacher. Limits beyond `i128` are not supported.
Like `BootstrapFewShot`, random search and `BootstrapFewShotWithOptuna`
accept a borrowed `teacher_settings_ref`. Each bootstrap attempt reads it,
so updates carry across attempts and across random-search candidates.

BetterTogether's `optimizers.LabeledFewShot` and `optimizers.RandomSearch`
expose `compile_options`; `Defaults.prompt_compile_options` configures the
default `"p"` step. These are typed native options, not arbitrary Python kwargs.
Random search returns each runnable candidate with its seed, score, and
per-example subscores, in stable descending score order.

The offline reference gate runs the locked DSPy few-shot option oracle and a
native public-API consumer. They compare candidate seeds and scores, selected
demos, evaluation and metric call counts, and refusal categories. The corpus
includes negative limits, wide positive limits, and defaults exercised through
BetterTogether. These comparisons run the real optimizer methods without
hosted model inference; the refusal categories do not stand in for Python
exception identities or messages.

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
A zero-iteration compile preserves the actor as-is, even after a failed call.
For positive iterations, an evolved actor returns `error.AvatarActorEvolved`;
call `reset()` before training it again.
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
replace the corresponding defaults. `options.seed` accepts signed 64-bit
integers and follows CPython's shuffle order, including for negative seeds.
Python integers outside the signed 64-bit range are not supported here.

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

## Bounded model-generated Python

`dszig.bounded_python_rlm.initProgram(Sig, arena, &session, options)` runs
an RLM action loop in which the model writes Python and a native Zig VM
executes it. Initialize a stable, caller-owned
`bounded_python_rlm.Session` in place with a fuel budget and an explicit
sub-model-query flag. A session rejects overlapping use, preserves bindings
between turns, and resets them when the invocation ends. Use one session
per active invocation. The program completes only on a `SUBMIT` that passes
the signature's output validation; when the turn budget expires it returns
`error.RLMSubmissionRequired` instead of asking the model for a final answer.

The supported profile is scalar assignment, indented `while` loops, `None`,
booleans, strings, checked signed 64-bit arithmetic (`+`, `-`, `*`, `//`, `%`),
single comparisons, grouping, `print(value)`, and `SUBMIT` keyword arguments.
`llm_query(text)` is available only with an explicit host capability and a
separate RLM call budget. There are no collections, general functions, imports,
filesystem or network APIs in the guest. The VM compiles the entire action
before executing it. Syntax errors do not perform earlier statements within
that action; runtime errors may leave already completed assignments in
the current session. Neither error publishes a partial result.

The VM uses fixed storage in the machine; the RLM controller and LM transport
still use their caller-provided resources. Limits are 8 KiB of source and
printed output per action, 64 globals, 64 submitted fields, 64 bytes per name,
4 KiB per string and 64 operand slots. The default budget is 100,000 bytecode
instructions per action, with a hard maximum of 1,000,000. Parsing and per-op
byte copies have separate fixed-size bounds. Fuel does not interrupt an
enabled external model call: use provider deadlines. This in-process
interpreter is not an OS isolation boundary or a full Python/Flex runtime.

```sh
devenv shell -- zig build bounded-python-sim -- --seed 42
devenv shell -- dz-live-python-rlm-smoke .zig-cache/bounded-python-receipt.json
```

The simulation drives the real VM with an independent arithmetic oracle and
scripted host responses. It checks ordered requests, reset/replay and
host-denial behavior; `--inject-wrong-result` must fail with `SubmitMismatch`.
The authenticated smoke checks the model's Python on an input seen during
generation and then reruns the exact source offline with a second, unseen
input. Neither a printed answer nor a hard-coded submission can pass it.

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

Use `optimize.evaluateDetailed` for serial evaluation with a configurable
`failure_score` and ordered per-example results. It retains successful
predictions in the output arena; a recoverable program or metric error
produces a row with no prediction and the configured score. Rows borrow the
original examples. The aggregate-only evaluator does not retain predictions.
`parallel.Evaluator.evaluateDetailed` has the same result shape. It clones
only the winning predictions after all attempts join, so rows survive
subsequent evaluations and evaluator teardown.

`DetailedEvaluation.writeJson` and `.writeCsv` write the evaluation table to a
caller-owned writer. Supply the metric column name explicitly. Both preserve
example order and prefix overlapping fields with `example_` and `pred_`; CSV
orders columns by first appearance. JSON accepts plain JSON-serializable
values; it does not dispatch Python model serialization methods.

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
candidates through it, each call using the optimizer's effective `max_errors`; a
program-type mismatch returns `error.ProgramTypeMismatch`, and null keeps serial
`optimize.evaluate`. MIPROv2 rejects a mismatched evaluator before demo
bootstrapping or instruction proposal. As in DSPy, candidates are evaluated
sequentially; only examples within one evaluation run concurrently. `BootstrapFewShot`
has no DSPy thread option and stays serial.
`InferRules` inherits the outer `Context.max_errors` for bootstrapping and
candidate evaluation unless its own `max_errors` is set. Teacher settings do
not replace that limit.
`BetterTogether` uses the outer limit for baseline and candidate validation
unless its `max_errors` option is set; this option does not override each
nested optimizer's own limit.
`COPRO` inherits the caller context's limit for candidate evaluation. Setting
`COPRO.max_errors` is a native override, not a translation of DSPy's
`eval_kwargs["max_errors"]`, which DSPy 3.5 rejects during evaluation.
`BootstrapFewShotWithOptuna` inherits the same caller setting for both
bootstrap and trial evaluation; its explicit `max_errors` is a native override.
For standalone serial `optimize.evaluateDetailed`, a null
`EvaluationOptions.max_errors` also inherits the caller context.

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

# Regenerate the golden fixtures with locked DSPy 3.5.0, check artifact
# round-tripping, and replay the mock and available 3.5.0 recordings in both
# runtimes without making model requests:
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

JSONAdapter's fallback object search now takes linear time without allocating
scratch storage. It still selects the leftmost balanced brace group, including
braces inside quoted strings, as DSPy 3.5.0's regex does. A local DSPy `Predict`
probe confirmed the first-object, nested-object, and no-match outcomes. The
retained typed Zig regression covers all three, including a 128 KiB unmatched
brace prefix. The same standalone ReleaseSafe `Predict` smoke went from 11.49 s
to 0.15 s elapsed locally and returned the same answers and parse error.
Debug and ReleaseSafe each passed 849 tests and all 40 steps, including examples;
native replay matched all 687 requests, eight adapter cases, and nine pipelines.
No blocker remained for object extraction at that checkpoint.

For truncated JSON strings, escape decoding now updates brace and regex-class
tracking in place instead of rescanning the accumulated string for every
escape. A pinned DSPy 3.5.0 `Predict` with json-repair 0.64.2 confirmed the
decoded output on 1 KiB and 256 KiB completions. The same standalone native
ReleaseSafe `Predict` smoke recovered identical bytes before and after on
256 KiB and 1 MiB completions; combined elapsed time fell from 26.23 s to
0.20 s locally, excluding compilation. The retained typed regression checks
the entire decoded 1 MiB completion, including escaped braces, brackets,
backslash runs, and a Unicode tail.

Debug and ReleaseSafe each passed 850 tests and all 40 steps, including
examples; native replay again matched all 687 requests, eight adapter cases,
and nine pipelines. No blocker remains for this escape-decoding change. The
full reference gate and authenticated inference were not rerun. Next, measure
whether repeated closing-brace lookahead still rescans long truncated strings
without a closing quote.

For that no-closing-quote path, JSON repair now reuses the colon search miss
and the next unescaped closing-brace position instead of rescanning the same
suffix. Pinned DSPy 3.5.0 `Predict` with json-repair 0.64.2 confirmed the
recovered text on a 24 KiB completion and the escape-parity, nearby-brace,
and unescaped-colon boundaries. The retained typed regression checks every
output byte with 65,536 closing braces and covers those boundary cases.
The same standalone native ReleaseSafe `Predict` smoke returned identical
answers before and after on 24, 96, and 192 KiB inputs; combined elapsed time
fell from 8.45 s to 0.16 s locally, excluding compilation.

Debug and ReleaseSafe each passed 851 tests and all 40 steps, including
examples. Native replay consumed all 687 requests and matched all eight
adapter cases and nine pipelines. No blocker remains for this path. The
full reference gate and authenticated inference were not rerun. Next,
measure the separate path where many braces look ahead to the same closing
quote followed by a long whitespace suffix.

JSON repair now classifies the text after each shared closing quote once per
string scan. Pinned DSPy 3.5.0 `Predict` with json-repair 0.64.2 confirmed the
recovered text for an escaped value followed by a long whitespace suffix and
no object close. The retained typed regression checks all output bytes with
32,768 closing braces and covers Unicode whitespace and a later quote with
and without a following colon. The standalone native ReleaseSafe `Predict`
smoke returned identical answers on all three inputs (4,096 to 32,768 braces);
combined elapsed time fell from 3.33 s to 0.16 s, excluding compilation.

Debug and ReleaseSafe each passed 852 tests and all 40 steps, including
examples. Native replay consumed all 687 requests and matched all eight adapter
cases and nine pipelines. No blocker remains for this path. The full reference
gate and authenticated inference were not rerun. Next, measure whether repeated
`]` characters in a malformed quoted array value rescan for the same distant
closing quote.

JSON repair now reuses the next unescaped quote when scanning closing brackets
inside a quoted array item. Pinned DSPy 3.5.0 `Predict` with json-repair 0.64.2
confirmed the decoded text and the missing-quote, odd/even backslash, and
multiple-item boundaries. The retained typed regression checks every output
byte with 32,768 brackets and covers those boundaries. The same standalone
native ReleaseSafe `Predict` returned identical answers on three inputs
(4,096 to 32,768 brackets); combined elapsed time fell from 2.18 s to 0.16 s
locally, excluding compilation.

Debug and ReleaseSafe each passed 853 tests and all 40 build steps, including
examples. Native replay consumed all 687 requests and matched all eight
adapter cases and nine pipelines. No blocker remains for this change. The
full reference gate and authenticated inference were not rerun. Next, measure
whether repeated colons inside a malformed quoted object key rescan for the
same later quotes.

JSON repair now reuses the quote lookahead across colons in a malformed quoted
object key. Pinned DSPy 3.5.0 `Predict` with json-repair 0.64.2 confirmed the
decoded key and value with 8,192 colons, as well as escaped-quote, single-quote,
multiple-key, and refusal cases. The retained typed `Predict` regression
checks the full key with 32,768 colons and covers those boundaries. A standalone
ReleaseSafe `Predict` returned identical dictionaries before and after on
three inputs (4,096 to 16,384 colons); in two warm runs, combined runtime
fell from about 0.53 s to 0.006 s, excluding compilation.

Formatting passed, and Debug and ReleaseSafe each passed 854 tests and all
40 build steps, including examples. Native replay consumed all 687 requests
and matched all eight adapter cases and nine pipelines. No blocker remains
for this path. The full reference gate and authenticated inference were not
rerun. Next, measure malformed keys with mixed quote delimiters to see whether
different first quotes can still trigger repeated scans to the same second
quote.

Malformed JSON keys with opening smart quotes now reuse a shared closing-quote
search instead of rescanning the same suffix after each opening quote.
Pinned DSPy 3.5.0 `Predict` with json-repair 0.64.2 established the expected
dictionary for 8,192 repetitions and the outcomes for escaped delimiters,
nested quote spans, multiple keys, and malformed input. The retained typed
`Predict` regression checks the full decoded key at 32,768 repetitions and
those boundary cases. The standalone ReleaseSafe `Predict` smoke returned
identical dictionaries before and after on three inputs (4,096 to 16,384
repetitions); combined elapsed time fell from 0.62 s to 0.16 s in one local
comparison, excluding compilation.

Formatting passed, and Debug and ReleaseSafe each passed 855 tests and all
40 build steps. Native mock replay matched all 687 requests, eight adapter
cases, and nine pipelines. The full offline DSPy reference gate passed,
including replay of the existing 3.5.0 recordings. No blocker remains for
this change; no new authenticated inference was needed or run. Next, measure
whether repeated quotes in a JSON string containing a regex character class
rescan its growing prefix and revisit the same closing bracket.

JSON repair now tracks whether the current regex character class contains
whitespace as it decodes the string, and reuses the next unescaped closing
bracket (or end of input) across bare quotes. Pinned DSPy 3.5.0 `Predict` with
json-repair 0.64.2 confirmed the decoded answer at 8,192 bare quotes and on
six boundary cases. The retained typed `Predict` regression checks every
output byte at 32,768 bare quotes and covers Unicode whitespace, new opening
brackets, odd/even backslash runs, a missing close, and consecutive classes.
A standalone ReleaseSafe `Predict` returned byte-identical answers before
and after on three inputs with 8,192 to 65,536 bare quotes; combined elapsed
time fell from 26.83 s to 0.17 s locally, excluding compilation.

Formatting passed, and Debug and ReleaseSafe each passed 856 tests and all
40 build steps, including examples. Native mock replay consumed all 687
requests and matched all eight adapter cases and nine pipelines. The full
offline DSPy reference gate passed, including replay of existing 3.5.0
recordings. No blocker remains for this change, and no new authenticated
inference was needed or run. Next, measure whether the separate quote-before-
comma repair path repeatedly scans past the same distant quote to distinguish
an embedded comma from the next object member.

JSON repair now classifies the text after a shared closing quote once when
deciding whether embedded commas begin an object member. Pinned DSPy 3.5.0
`Predict` with json-repair 0.64.2 confirmed the full answer at 4,096 commas
and the boundary behavior for actual members, escaped quotes, a second
quoted span, and end of input. The retained typed `Predict` regression
checks every output byte at 32,768 commas followed by a long Unicode
whitespace suffix. A standalone ReleaseSafe `Predict` returned identical
answers before and after on three inputs with 4,096 to 32,768 commas;
combined local runtime fell from 1.60 s to under 0.01 s, excluding compilation.

Formatting passed, and Debug and ReleaseSafe each passed 857 tests and all
40 build steps, including examples. Native mock replay consumed all 687
requests and matched all eight adapter cases and nine pipelines. The full
offline DSPy reference gate also passed, including replay of existing 3.5.0
recordings. No blocker remains for this change; no authenticated inference
was needed or run. Next, measure whether malformed values with many braces
before a Markdown fence repeatedly scan the same post-fence suffix.

JSON repair now reuses the search for a closing quote across embedded Markdown
fences that look ahead to the same suffix. A pinned DSPy 3.5.0 `Predict` with
json-repair 0.64.2 confirmed the decoded answer at 8,192 fences and five
boundary cases covering wrapper fences, escaped quotes, nonclosing quotes,
and post-fence containers. The retained typed `Predict` regression checks
every decoded byte at 32,768 fences and covers those same boundaries. A
standalone ReleaseSafe `Predict` returned identical bytes before and after on
three inputs with 4,096 to 32,768 fences; combined local elapsed time fell
from 9.56 s to 0.17 s, excluding compilation.

Formatting passed. Debug and ReleaseSafe each passed 863 tests and all 40
build steps, including examples. Native mock replay consumed all 687 requests
and matched all eight adapter cases and nine pipelines. The full offline
DSPy reference gate passed, including replay of existing 3.5.0 recordings.
No blocker remains for this quote-search change, and no new authenticated
inference was needed. Next, measure the separate post-fence container scan
when repeated fences precede unterminated inline containers.

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
The later independent clone-context proof above addresses shared custom-box
aliases, without relabeling these older excluded observations as passed.

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
the actual live CLI against a scripted localhost provider. OpenAI generation
passes HTTP statuses below 400 to its JSON or SSE parser, matching DSPy’s
provider status check. A valid response body is still required. Other callers
retain their own status policies.
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

A separate native Chat Completions run on public OpenAI used SecretSpec and
`gpt-4o-mini` for 683 recorded calls. Pinned DSPy 3.5.0 strictly replayed that
transcript and reproduced all eight adapter cases and nine pipeline scores
and artifacts. COPRO and SIMBA completed but did not improve the date task.
Separate native live checks exercised asynchronous field streaming on both
Chat Completions and Responses. These runs do not qualify the homelab endpoint.

Additional authenticated runs on `https://api.openai.com/v1` verified
ChainOfThought, native ReAct `submit` (one typed tool call), and asynchronous
field streaming through both Chat Completions and Responses with
`gpt-4o-mini`. In both streaming dialects the routed deltas reconstructed the
answer and preceded the correct final prediction. A three-request
Responses run accepted one BootstrapFewShot trace, verified the demo in
the held-out request, and returned the correct typed answer both before
and after compilation. A separate native KNN run using
`text-embedding-3-small` returned the expected nearest example and reused
the cached query embedding on the second lookup.

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
# Optional: DSZIG_LIVE_MODEL, DSZIG_LIVE_API (chat_completions|responses)
devenv shell -- dz-live-smoke           # one real native CoT request
devenv shell -- dz-live-stream-smoke    # one streamed typed prediction
devenv shell -- dz-live-document-smoke  # one typed File prediction on Responses
devenv shell -- dz-live-tool-smoke /tmp/dszig-tool-smoke.json
devenv shell -- dz-live-optimization-smoke /tmp/dszig-bootstrap-smoke.json
devenv shell -- dz-live-verify /tmp/dszig-release-receipts # directory must not exist
devenv shell -- dz-live                 # DSPy and DSZig recordings, cross-replayed
```

These managed commands use `OPENAI_API_KEY` and pin `OPENAI_BASE_URL` to
`https://api.openai.com/v1`, regardless of inherited endpoint settings.
The streaming smoke routes live SSE text through the XML field listener,
checks the reconstructed field against the typed prediction, and writes a
compact receipt. Its default is Chat Completions; the optional
`DSZIG_LIVE_API=responses` setting exercises the Responses transport instead.
The optimization smoke makes two model calls: it bootstraps one verified CoT
demo, saves and reloads the optimized program state, and checks that the
held-out request actually includes the reloaded demo and returns the expected
answer. Its receipt contains both requests and their responses.
The tool smoke checks a native ReAct round trip: the model calls a local
shipment lookup, then uses its result in a separate `submit` call. It requires
both native function calls in order, verifies the successful lookup result in
the trace, and checks the submitted code, which is not in the initial prompt.
The document smoke forces Responses, attaches a synthetic text file, and
checks the model's extracted code before writing its receipt. Both DSZig and
the pinned DSPy 3.5.0 returned the expected code with `gpt-4o-mini` on the
official endpoint.

`dz-live-verify` runs thirteen native workloads: ChainOfThought, typed SSE
streaming, strict JSON-schema output, ReAct tool execution, and BootstrapFewShot
with state reload on both Chat Completions and Responses; typed document input
on Responses; hosted `text-embedding-3-small`
retrieval through KNN with a verified repeat-query cache hit; and native
reasoning with `gpt-5-mini` on Responses. The reasoning workload requires
both the correct typed answer and a nonempty typed reasoning summary matching
the provider summary. The schema workload checks that the strict request
contains the Unicode property `résultat_pair` with the title `Résultat Pair`,
and that the typed result is correct without adapter fallback.
The gate enables capabilities only where a workload requires them, fails on
the first error, and writes each receipt to a new directory without
modifying the reference recordings. It publishes `verification.json` only
after all workloads pass. Individual transcripts can be left by failed
workloads for diagnosis and are not a pass signal. A passing run verifies
those model and provider paths; use `dz-check` separately for the pinned
DSPy reference gate.
This gate does not qualify other providers or certify that an arbitrary DSPy
program can be migrated. For a standalone check with the
same strict summary assertion, use `--reasoning-smoke` with an authenticated
`zig build live` invocation. The following diagnostic variant instead reports
whether a summary arrived without requiring one:

```sh
DSZIG_LIVE_REASONING=1 DSZIG_LIVE_API=responses DSZIG_LIVE_MODEL=gpt-5-mini \
  devenv shell -- dz-live-smoke /tmp/dszig-reasoning-smoke.json
```

The diagnostic variant reports `native summary: yes` when the provider
returns one. Like DSPy, DSZig's prediction bridge requests `summary="auto"` when
native reasoning is enabled; a provider can still omit a summary. Direct
canonical SDK calls retain their own reasoning configuration.
The DSPy recorder (`tools/dspy_reference/live.py`) passes that URL as
`api_base` and maps `chat_completions` to DSPy's `chat` model type.
Direct `zig build live` invocations can still select a compatible endpoint.
Unknown flags and extra arguments fail before credential lookup or network use;
they never fall through to the full, paid recording suite.

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

With native function calling enabled and no `ToolCalls` output, `Predict`
leaves explicit `tool_choice` and `parallel_tool_calls` for the LM to interpret.
Disabling native calling strips both options, as does declaring a `ToolCalls`
output on a model without function-calling support.

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

The locked reference gate checks native counterparts for all 15 entries in
DSPy 3.5.0's `dspy.teleprompt.__all__` and all 12 in `dspy.predict.__all__`.
The explicit mappings are `bootstrap_trace_data` to `bootstrapTraceData`,
`majority` to `ensemble.majority`, and `RLM` to `CustomRLM`. The last is a
migration surface for a caller-supplied runtime, not a Python interpreter.
This declaration check is separate from behavior and artifact replays. It
does not establish option or provider parity, nor cover private exports.
Before deploying a native Zig rewrite, verify it with your own data, metrics,
saved state, and intended model and provider.
Run `dz-check` for the offline reference and regression gate, then
`dz-live-verify` for the bounded authenticated OpenAI gate. For a migrated
application, also compare held-out results and reloaded state against the
original program, and measure latency and failure behavior on the deployment
target. The bundled live smoke is not an application acceptance test.

- **RLM, Flex and interpreters:** the [bounded Python guest](#bounded-model-generated-python)
  executes a limited Python profile in Zig. It is not PythonInterpreter/Pyodide/Deno,
  LocalInterpreter, InterpreterPool, or a complete Python environment.
  DSPy's RLM/Flex can import real DSPy inside the guest; DSZig's bounded guest
  cannot import packages. Port compatible guest logic to the bounded profile,
  or supply a separate runtime through `CustomRLM`.
  The separate Linux generated-Zig executor above compiles native programs and
  capability RPC; it does not claim arbitrary Python execution or full RLM/Flex
  behavior. A `History` loaded from tagged DSPy REPL JSON can replay recorded
  action and observation turns through XML, Chat and JSON adapters without
  executing code. Observation truncation counts Unicode characters, and
  replayed code and observations cannot create provider media parts. This
  preserves DSPy's slicing for zero and negative signed 64-bit observation
  limits. Replay does not create or resume an interpreter session.
  `CustomRLM(Sig)` adds a synchronous action/observation loop for a trusted,
  caller-supplied runtime factory. Give it an action objective describing what
  the runtime supports. The constructor rejects input names reserved by the
  sandbox and output names reserved for result metadata. Each call opens one
  fresh session, passes model actions to it serially, and closes it on success
  or failure. Invalid typed submissions report missing fields or all
  field-validation failures in one observation; diagnostics use native
  validation errors rather than Pydantic's exact messages. Typed submissions
  and extraction at the iteration limit return outputs, tagged history, final
  reasoning, and the current trajectory. Action parsing unwraps nested bare
  fences, accepts `python`, `py`, `python3`, and `py3` labels, and feeds an
  unsupported-label error back without executing the action. Recoverable
  execution failures become observations; runtime failures propagate.
  `forwardWithHistory` continues the
  conversation, not the runtime state. Variable previews count Unicode code
  points, group the displayed count with commas, and show the first and last
  500 code points when the rendered value exceeds 1,000. List and dictionary
  previews use `list` and `dict` rather than their parameterized annotations.
  Optional inputs display `NoneType` or the present value's type; scalar
  literals display their scalar type rather than the `Literal` annotation.
  An optional `history_processor` runs before each action and can edit or trim a private
  draft, or supply a replacement history that DSZig copies. `forwardWithOptions`
  can override the processor or the `interpreter_factory` for a single call.
  The factory override
  does not change the configured action instructions. Callback errors in the
  history processor roll back edits, except allocation failure and cancellation,
  which propagate. Trimming prompt history does not
  change the returned trajectory or the trajectory supplied to extraction.
  A runtime callback receives a borrowed `Runtime.Execution`: read the input
  record through `variables` and call `query` to dispatch a text-only sub-LM
  request. It uses `sub_lm` when configured, otherwise the context LM.
  `max_llm_calls` is shared across that invocation's actions; empty prompts do
  not consume it, but failed dispatched calls do. `queryBatch` reserves all
  calls before dispatch, accepts empty text prompts, preserves input order,
  and converts native `LMFailure` to `[ERROR] LMFailure` per item. Other
  errors fail the batch. It runs serially by default; pass `BatchConcurrency`
  with a concurrency-capable `Io`, a thread-safe LM and a thread-safe arena
  to run up to eight requests at once. Workers are joined before return and
  their text is retained by that arena. `queryParts` accepts ordered canonical
  text, image, audio and document parts; `queryContentBatch` accepts a mix of
  text and part-list prompts with the same batch semantics. Parts are not
  interpreted as adapter markers, and local paths are rejected before any
  calls are charged. Actual media support depends on the sub-LM and its
  provider. The loop does not execute arbitrary Python, provide guest DSPy,
  or sandbox runtime callbacks. The caller must enforce runtime time and
  memory limits.
- **Other modules and optimizers:** bounded native Avatar/AvatarOptimizer follow
  Source methods, with broken Source constructors and the explicit native
  capacity, class-identity, callback-ownership and serial-execution boundaries
  described above. GRPO is not implemented. It is not in DSPy 3.5.0's public
  teleprompt exports, but is importable from `dspy.teleprompt.grpo` and
  requires a reinforcement job that accepts rewarded batches and activates
  updated policies. DSPy's bundled OpenAI provider does not implement that
  job; OpenAI supervised fine-tuning is not a substitute. Native BetterTogether
  composes implemented optimizers, with the typed argument, teacher, ownership
  and provider limits described above. Trace bootstrapping, Chat/XML fine-tuning
  data formatting, bounded OpenAI training jobs and BootstrapFinetune are implemented
  with the provider, capacity and ownership limits described above.
  Native Parallel uses explicit typed
  tasks and contexts, not Python callback objects or ambient thread-local settings.
  Its startup capacities and cooperative join contract are described above.
- **ReAnchor:** Compile with a cached LM or explicitly set `require_cache = false`.
  With cache checking enabled, concurrent scoring must use the same effective
  LM instance as serial evidence collection for each decision predictor, and
  neither path may bypass its cache.
  A custom evaluator must expose its attempt contexts to enable this check.
  The returned program and calibration report are retained in the compile
  allocator. Calibration observes decision evidence serially and can use a
  bound concurrent evaluator for scoring. Set `ReAnchor.log_dir` to a trusted
  directory path to atomically replace `report.json` after successful evaluation,
  or use `Report.writeJson` with a caller-owned writer. A late validation error
  leaves the training report available; a write error leaves the completed
  report in memory and propagates instead of returning a calibrated program.
  Decimal and arbitrary-precision evidence values are not implemented.
- **Optimizer execution and shared identity:** InferRules' signature-alias propagation does not update arbitrary external
  instances outside the reachable program graph. Canonical Zig signature/Code
  types and default CoT construction do not reproduce every fresh dynamic Python
  class identity; Refine shares this boundary.
- **Runtime model identity:** Code, File, Image, Audio, Reasoning, BoundingBox,
  Citations, Noul/Score/Choice and generated decision evidence have genuine boxed classes,
  not a complete Python object runtime. Other model
  families still use their existing representations. A clone scope preserves
  shared custom-box aliases, but generic arrays/maps do not preserve shared
  container identity or cycles through memoized deepcopy. Complete
  Python builtin identity/dispatch, NaN float-object identity across distinct or
  cloned boxes, Reasoning indexing/string-method delegation, arbitrary attribute
  mutation and Python frozen-assignment exceptions are not reproduced. The
  citation proof retains 150 unsupported Decimal/bytearray/dynamic-model rows;
  its finite passing observations do not establish complete Citations parity.
- **Providers and media:** `dszig.Anthropic` supports Messages generation
  with text turns, function calls and text tool results, and
  explicit JSON schemas (`supports_response_schema = true` for a compatible
  model). Canonical judgment schemas use the Messages `anyOf` dialect.
  A single JSON-object answer becomes a canonical data part. Data parts replay
  as compact JSON text in messages, system instructions, and tool results.
  Messages returns the selected value, not a probability distribution; `if_available` records
  that limitation and `required` refuses measurable judgments before sending.
  Supply an existing `std.http.Client` and an API key or a
  per-attempt `header_provider`; use a trusted `base_url` ending in `/v1`.
  Set `system_prefix` to prepend application instructions before the caller's
  system blocks, preserving any cache markers on those blocks.
  For a compatible endpoint, `model_prefixes` restricts model names before
  credentials or HTTP, avoiding silent substitution of out-of-prefix names.
  Set `sampling_params = .reject` for servers that ignore temperature, top-p,
  and top-k. These canonical settings are omitted with an adaptation, or
  refused before HTTP under the `.refuse` adaptation policy.
  Use `parallel_tool_calls = .reject` for servers that ignore the parallel-tool
  control. An explicit parallel preference is then omitted and reported, or
  refused in strict mode; the tool choice and declarations still apply.
  Use `structured_output = .reject` for endpoints that ignore response
  formats. Raw canonical requests then omit the format with an adaptation
  (or fail under `.refuse`), and typed predictors do not advertise native
  response schemas on those endpoints.
  Set `cache_control = .none` for endpoints with implicit caching. Explicit
  marks are omitted, unsupported long retention is reported, and references
  to stored cache resources still fail before HTTP.
  `Anthropic.bindPreset` selects the documented Messages wire policy and
  default endpoint for `.anthropic`, `.deepseek`, `.meta`, or `.moonshotai`.
  Pass a trusted URL override for a proxy. Binding does not discover or
  change credentials; supply credentials for the selected server.
  For compatible Claude models, set `supports_reasoning = true` to advertise
  native reasoning to typed predictors. Canonical requests support manual
  budgets and adaptive effort; `max_tokens` is the visible budget on manual
  models and the total budget on adaptive models. Select `thinking_format`
  `.deepseek`, `.adaptive`, or `.effort` for compatible endpoints whose dial
  is effort-only. Those dialects send an explicit disabled block for an off
  request and leave absent reasoning settings untouched. An optional
  `reasoning_efforts` list clamps unsupported levels to the nearest supported
  level (ties go lower); strict adaptation policy refuses the clamp.
  Canonical assistant
  messages preserve signed and redacted thinking for subsequent requests.
  The Messages implementation has been compared against DSPy over loopback,
  not against a hosted Claude model. `tool_result_media = .images` admits
  images but not documents in tool results; `.reject` admits neither.
  Both policies fail before HTTP for unsupported attachments, without
  changing normal user-message media. `Anthropic.streamRequest` emits
  callback-borrowed canonical SSE events with early cancellation, including
  tool fragments and reasoning continuation. Use `Anthropic.lm()` with the
  regular program streaming APIs to route typed fields and get a final
  prediction; cache hits return the prediction without replaying live fields.
  Messages accepts image and document blocks, including in tool results,
  from inline base64, URLs or existing provider file IDs. The transport does
  not read local paths implicitly. Canonical `CacheConfig` supports ephemeral
  system and message-block markers, including one-hour retention; stored cache
  resources are not supported. Canonical requests can declare hosted web
  search and code execution. Buffered responses expose citations in message
  order and retain hosted tool blocks in raw provider data rather than treating
  them as client tool calls. Unrecognized blocks are recorded by wire index
  under `provider_data._lm15_unmapped` without discarding usable parts. Set
  `thinking_replay = .unsigned` only on endpoints that accept unsigned thinking
  blocks. Signed and redacted
  continuation still takes precedence. Additional media kinds are not supported.
  `Gemini.lm()` and `Gemini.completeRequest` support nonstreaming text,
  JSON/schema-constrained generation, and function calls with text results
  on the Gemini Developer API, with `x-goog-api-key` authentication and
  canonical token accounting. `Gemini.initVertex` also supports project-scoped
  Vertex AI generation and SSE. Pass `.project`, `.model`, and either an API
  key or an OAuth header provider; `.location` defaults to `global`. The OAuth
  callback supplies `Authorization: Bearer …` on each attempt. Use
  `Gemini.initVertexExpress` for projectless, query-key authentication.
  Both constructors accept a trusted endpoint root or gateway prefix.
  Use HTTPS outside local tests, and redact Express query strings in proxy
  logs. Native credential discovery and hosted Vertex authentication have
  not been verified.
  Gemini converts judgment branches to enum
  constraints and folds labels into the schema description. A buffered answer
  with one JSON-object text part becomes a canonical data part. Gemini
  returns the pick, not a distribution over judgment keys: `if_available`
  records that limitation and `required` refuses measurable judgments.
  Canonical data parts can be sent back as compact JSON text, including in
  system instructions and developer turns. `extensions.output` can request
  image or audio output; other wire extensions override top-level request
  fields. Gemini has no end-user attribution field, so `user_id` is recorded
  as dropped or rejected under the refusal policy. The codec maps requested
  token logprobs in buffered and streamed responses when a model serves them. Buffered
  generation maps grounding sources to citations and preserves hosted code
  execution output in raw provider data. Unrecognized buffered parts are
  recorded by wire index under `provider_data._lm15_unmapped` without
  discarding usable parts. Nonstreaming reasoning supports
  Gemini 2.5 budgets and Gemini 3 effort levels, with thought-signature replay
  for reasoning, text, and function calls. Buffered replies preserve a signed
  thinking part even when `text` is null: the next turn sends an empty string
  with the original signature. Set `Gemini.supports_reasoning` when the
  configured model supports native reasoning for typed prediction. Gemini
  accepts inline bytes, URIs, and file IDs for media, including image and
  document attachments in function results. GenerateContent also accepts
  trusted local paths, reads at most 16 MiB per file, and sends the bytes as
  inline base64. This does not extend to Live WebSockets. Canonical function
  results also accept structured data and citations; Gemini replays citation
  text rather than provider citation metadata. It decodes returned image, audio,
  and document parts. Generation supports implicit caching and references to
  existing `cachedContents` resources; it does not create or manage those
  resources. `Gemini.streamRequest` delivers canonical SSE events
  with text, thought signatures, function calls, and inline image/audio deltas.
  On normal exhaustion, a GenerateContent stream with a terminal frame emits
  one end event, including usage that arrived after the finish reason. A callback
  can still stop reading without waiting for the response tail. Raw stream
  errors preserve Gemini diagnostics and distinguish model and context-length
  failures from generic provider errors. GenerateContent SSE also exposes
  `provider_error.request_id` when a recognized response header provides one.
  In-stream errors also expose `provider_error.retry_after` when a recognized
  numeric retry header supplies a hint in seconds.
  HTTP 400 context-length errors
  return `error.ContextWindowExceeded` in both buffered and SSE generation.
  Typed Gemini
  streaming supports `AsyncProgram` field listeners, final predictions, and
  cancellation. Canonical streaming can run a Gemini Live WebSocket turn
  (`config.extensions.transport = "live"`) with text or conversation history,
  inline audio/video input for audio-native models, function declarations, and
  text, audio, image, and function-call deltas. Non-audio conversation history
  can include earlier function calls and results. It uses the configured Gemini
  API key. `Gemini.live(&session, options)` opens a persistent `Gemini.LiveSession`
  for text, turns, inline base64 audio/image, and function results. Use `send`
  and `recv` serially, keep the session at a stable address, and call `close`
  when finished. Received events borrow storage until the next `recv` or
  `close`. Canonical Live requests and persistent `.turn` inputs support
  bounded local media reads. Vertex Live and file uploads are not supported.
  Native File construction, adapter/typed prediction
  handling, File tool-result markers and bounded Responses document translation
  are implemented as described above. There is still no complete lm15 provider
  catalog/routing/auth/error taxonomy, full provider SDK, embedding providers
  beyond OpenAI/Azure, general rich-media tool results, non-OpenAI training
  providers, or other providers' streaming protocols. Completed SDK responses
  can be projected to events,
  and bounded event sequences or synchronous sources can be assembled.
  Bounded Chat, JSON, and XML text field listeners are available separately.
  Native programs can receive raw OpenAI events or routed field fragments
  synchronously or through a bounded Zig `Io` iterator, with optional program
  status messages.
  Typed `Image` and `Audio` can produce canonical input parts from encoded media.
  `Image.fromEncodedBytes` and both `fromPath` helpers accept explicit byte
  limits; path helpers require caller-trusted paths. `Image.fromUrl` and
  `Audio.fromUrl` explicitly download caller-trusted HTTP(S) media with a
  required decompressed byte cap, configurable inactivity timeout and redirect
  limit, and the caller's TLS trust. These helpers do not enforce an SSRF
  allowlist; use only trusted URLs and redirects. The runtime never downloads
  media implicitly or transcodes encoded audio. `Audio.fromWavSamples` encodes
  typed interleaved `i16`, `i32`, `f32`, or `f64` slices as PCM16 WAV with a byte
  cap, without NumPy or libsndfile. Other sample container formats are not
  supported. Live public OpenAI
  Chat Completions and Responses checks identified the dominant color of an
  inline PNG through typed `Predict`. When a Chat backend requires text-only
  tool rows, the prediction path replaces each tool image with a placeholder
  and sends the images together in a labeled user message after the tool
  outputs. Public OpenAI Chat Completions identified a purple PNG sent this
  way. Responses prediction requests similarly move audio out of the tool
  result into a user attachment, leaving native image and document results
  in the tool row. Raw SDK tool rows are not rewritten. This does not verify
  hosted audio, media fine-tuning, or other rich-media tool results.
  Chat Completions does not accept document parts.
- **Document, File, and DocumentPart boundaries:** Use `dszig.Document` for
  DSPy's experimental title/context framing, for example
  `.{ .data = "Water boils at 100 °C at standard pressure.", .title = "Facts" }`
  in a typed signature field. Plain text becomes text parts. Set
  `.media_type = .@"application/pdf"` with base64 PDF data to send a canonical
  document part. PDF input needs a provider route that accepts document
  parts, such as OpenAI Responses. This does not enable provider-native
  citation marking or the Python `DocumentPart` object interface.
  The constructor smoke explicitly leaves
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
- **Logprob boundaries:** read provider token scores on a typed result with
  `prediction.completionLogprobs(0)` for the first completion, or pass the index
  of another completion. Null means neither a score sequence nor incomplete
  coverage is known (or the completion does not exist). A present empty
  `tokens` slice with `reported = true` means reported empty. If an edit
  leaves no whole scored tokens but coverage is known to be incomplete,
  the row has empty `tokens`, `reported = false`, and `complete = false`.
  `complete = true` does not promise the provider supplied scores.
  Score metadata survives `Prediction.clone` and the
  optimizer's record projection. With no declared `logprobs` field, a batch
  that mixes reported and unreported scores fails, matching DSPy's rejection
  of an uneven completion column.
  Read the same per-completion metadata with `step.completionLogprobs(index)`
  on `Trace.Step` and on steps returned by `bootstrapTraceData`. Scores are
  copied into the trace's storage, including when BestOfN or Refine transfers
  the winning trace. They do not enter parsed output records, demonstrations,
  or fine-tuning data. Raw `forward` still returns only typed output fields.
  DSPy 3.5.0 overwrites a declared `logprobs` output with provider scores,
  even when the declared field is a string and the score sequence is empty.
  DSZig preserves the field when all scores are absent. If any completion
  reports scores, typed prediction returns `error.ConflictingCompletionLogprobs`
  after parsing, before publishing a trace step and without retrying the LM.
  Rename the output field when requesting provider scores and read the scores
  through `completionLogprobs(index)`.
  An offline DSPy 3.5.0 `Predict` probe confirmed that its trace contains the
  returned prediction, with ordered scores and distinct empty and absent
  scores, and that it records no step on mixed-presence refusal. The native
  public `Predict.forward` smoke reproduced missing trace scores before the
  change and read them after scratch teardown afterward. The retained
  regression exercises adapter fallback, `bootstrapTraceData` after scratch
  teardown, and demo extraction without scores. The pinned Debug and
  ReleaseSafe gates each passed 847 tests and 40 build steps; strict replay
  consumed all 687 requests across eight adapter cases and nine pipelines.
  No local verification blocker remained. The full Python reference suite
  and new authenticated inference were not run.
  The collision regression reproduced silent success with a reported-empty
  sequence before the fix. The public `Predict.call` and `Predict.forward`
  smoke verified that absent scores preserve the declared string, while
  reported-empty scores and a mixed batch are refused without an extra LM
  request or a trace step. Debug passed 848 tests and all 40 build steps.
  ReleaseSafe passed all 40 steps after the combined command timed out;
  the rerun executed all 77 behavior tests and reused cached results for
  the other suites. Native replay matched all 687 requests, eight adapter
  cases and nine pipelines. No blocker remains for this collision. The full
  Python reference gate was not rerun, and no live inference was needed.
  Pinned DSPy 3.5.0 materialized three loopback Responses streams that
  stopped after the same text: a cut inside the first scored token left no
  scores and incomplete coverage; a token-boundary cut kept one whole score;
  an unscored cut did not mark coverage incomplete. The reference peer
  observed all three connections close before it sent any terminal frame.
  Before the fix, the first case lost its coverage flag in `Predict.call`.
  The retained native regression drives canonical `OpenAI.completeRequest`
  over HTTP into a typed prediction via the public custom LM interface and
  checks both the cloned result and trace after scratch teardown. A standalone
  native consumer returned `four` with zero tokens, `reported = false`, and
  `complete = false` in both the prediction and trace. Debug and ReleaseSafe
  each passed 858 tests and all 40 build steps, including examples. Native
  replay consumed all 687 requests and matched eight adapter cases and nine
  pipelines. The full Python reference gate was not rerun, and no live
  inference was needed. Native typed LM options still cannot configure stop
  sequences directly. Next,
  compare and expose stop configuration through the normal typed Responses
  streaming path, including cache and replay identity.
  These observations do not qualify all providers or streaming behavior.
  Native SDK records do not implement Python dataclass identity or
  arbitrary constructor behavior. Native nonfinite-score persistence uses an
  explicit bit tag, not Python cache/SDK JSON interchange. `LM.callResult`
  also exposes scores at the LM boundary.
- **Pricing/history boundaries:** canonical calls can record shared entries
  in explicitly supplied global, client and module owners. This does not
  implement automatic module discovery, legacy adapter-call history, or
  Python's mutable history dictionaries and dynamic keyword records.
  Custom engines lacking a canonical SDK response object cannot use
  `LM.callResult`; their existing completion APIs remain available.
  Catalogue loading is explicit and local; arbitrary
  regex rules and automatic remote refresh/shrink validation remain unsupported.
- **Dynamic Python APIs:** no general Python Module/BaseModule/Example/Prediction/
  Completions object model, dynamic signature syntax, arbitrary Pydantic custom
  models/validators, custom Type hierarchy, generic callbacks,
  settings singleton, or arbitrary Python asyncify/syncify/streamify behavior.
  Native Prediction handles, explicit UsageTracker scopes and automatic LM/Module-call
  accounting are available with the lifetime, synchronization and resource
  contracts above; they are not the full dynamic Python object model.
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
  budget. Explicit builtin float/Decimal/complex boxes now retain immutable NaN
  identity through owning copies, but unboxed floats and arbitrary mutable
  alias/cycle graphs do not. A historical builtin observation smoke matched
  405 checks but differed on two same-NaN cases; that old receipt was not rerun
  or relabeled as passing. The new scalar/mutable-model evidence is reported
  separately above. Native ownership clones preserve a
  set's stored iteration snapshot, not CPython deepcopy's reconstruction order.
  JSON repair covers the
  pinned default parser and correlated Chat/Responses tool arguments, not every
  upstream option or every adapter parsing path. Its nesting bound is 512 and
  Python exception classes/messages are not reconstructed (for example, some
  fenced nonstring-key assertion failures become `ParseValue`). Chat completion
  field headers and typed signature names use the pinned Python Unicode `\w`
  rule. Use Zig's `@"éclair"` identifier syntax for non-ASCII fields. JSON
  response schemas preserve those property names and use Pydantic-style Unicode
  titles (for example, `éclair` becomes `Éclair`). A Chat line with a symbol or
  combining mark in the header name stays in the current field's text.
  XML golden cases that
  use a schema-backed runtime view do not prove a native Zig union type exists.
  Python-style rendering recovers declared native type metadata and retained
  native custom classes, not arbitrary Python objects erased into ordinary
  `Value`/`Record` dictionaries; a nested dynamic model inside a History record
  therefore cannot retain its original Python class repr merely from its keys.
- **Optimizer options:** BootstrapFewShotWithRandomSearch uses the DSPy seed
  range `[-3, num_candidate_programs)`, so negative limits can exclude some
  or all of the fixed candidates. Native compilation reports `NoCandidates`
  for an empty range with no restriction.
  GEPA requires exactly one budget: `auto`,
  `max_full_evals`, or `max_metric_calls`. Zero counts as a supplied budget;
  GEPA still evaluates the seed program before checking whether to stop.
  The native full-evaluation budget rejects overflow instead of wrapping.
  GEPA returns `error.EmptyTrainset` for empty training data and
  `error.NoComponents` for programs with no native predictors, before calling
  the metric or model.
  GEPA returns the best candidate indices for each
  validation example, including ties. Set `track_best_outputs = true` to also
  retain their outputs in `best_outputs_valset`, in candidate discovery order.
  Read ordinary predictions through `entry.output.prediction` (formerly
  `entry.prediction`), or handle `entry.output.failed_parse` for the raw
  completion and format reward. A child with no output can still update the
  score frontier without replacing or appending to the retained outputs. A
  failed seed evaluated through `Evaluate` retains an empty prediction.
  The results live in the compilation allocator and do not require additional
  model calls. Set `instruction_proposer` to supply a native callback
  instead of the built-in reflection prompt. It receives the current named
  instructions, selected component names, and available examples grouped by
  component. Ordinary inputs and outputs are rendered as Python-style strings,
  declared DSPy-type inputs use native trace values, and direct `History` inputs
  become `Context` text. The callback can use the configured reflection `Context`;
  an empty update skips child evaluation. GEPA retries a callback error once
  (allocation errors propagate) and copies the returned instruction text.
  The callback sees native trace representations, not Python objects; original
  model identity erased from untyped fields cannot be recovered. Custom code
  proposers, Flex components, and logging integrations remain unsupported.
  A metric can return named `Feedback.objective_scores`, with unique names per
  row. GEPA averages each objective over the validation rows that supply it
  and returns the aggregates, best scores, and tied leaders. `frontier_type`
  selects `.instance` (default), `.objective` (named aggregates), `.hybrid`
  (instances and named aggregates), or `.cartesian` (per-row objectives).
  Non-instance frontiers require at least one objective per evaluated validation
  batch. The winning program is still chosen by mean validation score.
  `acceptance_criterion` defaults to `.strict_improvement`; set
  `.improvement_or_equal` to validate reflective mutations whose minibatch
  scores tie their parent's. `.custom` supplies a callback that sees both
  minibatch evaluations and the current search state before child validation.
  It runs once per evaluated reflective mutation; errors abort compilation.
  Merge acceptance is unchanged.
  `sampling_strategy` defaults to `.single`; use `.{ .same_parent = n }` for
  several minibatches from one parent, `.{ .independent = n }` to select a
  parent for each task, or `.{ .pxn = .{ .parents = p, .mutations = n } }`
  to sample `n` minibatches per parent. Built-in batches contain 1–32 tasks.
  A `.custom` proposal sampler can fill up to 32 ordered `(parent_idx,
  minibatch_ids)` tasks or return zero to skip the iteration. It can use the
  configured candidate selector and minibatch sampler or supply its own
  training row IDs, including repeats. Each task must use an existing parent
  and a nonempty minibatch of at most `max_minibatch_size` rows (default 4096).
  `selection_strategy` defaults to `.all`; `.best` and `.{ .top_k = k }`
  rank accepted proposals by their minibatch improvement, preserving task
  order on ties. Identical selected children are validated once per iteration.
  The validation policy sees the same search state for every selected child;
  the evaluated children are then added in selection order. The metric
  budget is checked between iterations, so a whole batch can cross it.
  Tasks run in order within each phase; the optional rollout runner can
  parallelize examples within a task, not across tasks. A `.custom`
  selection strategy receives the evaluated proposals and can return their
  indices in any order, even if they fail acceptance. `shouldAccept` lazily
  memoizes the criterion for each proposal in that batch. Unknown indices and
  repeated candidates are ignored.
  Set `batch_reflection` to pass eligible jobs in sampling order to a native
  reflection backend. Return one instruction-update slice per job; an empty
  slice skips that child. A callback failure or wrong result count falls back
  to individual built-in reflection attempts, skipping only failed jobs.
  Allocation failure and cancellation abort compilation. An explicit
  `instruction_proposer` takes precedence. This is the underlying GEPA batch
  extension, not an automatic provider batch API in `dspy.GEPA`.
  Set `cache_evaluation = true` to reuse a candidate's scored rows during
  validation and merge. As in DSPy, training and validation share numeric
  row IDs for these lookups even if the datasets differ. Traced reflection
  still runs fresh and updates the cache; only misses in validation and merge
  consume the metric budget. The cache is independent of the LM cache and is
  retained across compiles only with a GEPA checkpoint. Read `Result.val_ids`
  for score order when enabled.
  Candidate selection supports `pareto`, `current_best`,
  `epsilon_greedy` (epsilon 0.1), and `top_k_pareto` (top five). Top-five
  selection restricts the validation frontiers before removing dominated
  candidates and falls back to the best aggregate candidate if none remain.
  A custom `candidate_selection_strategy` receives a read-only snapshot of
  candidate instructions, validation subscores, ancestry, and active frontier
  leaders before each reflection minibatch. It returns a zero-based candidate
  index; invalid indices fail before evaluating the minibatch.
  Set `batch_sampler` to choose the ordered training indices for each
  reflection minibatch, and leave `reflection_minibatch_size` unset. Repeated
  indices are evaluated in order; empty, out-of-range, and over-capacity
  batches are rejected before rollout.
  The default epoch sampler uses size three if `reflection_minibatch_size`
  is unset or zero. It pads to whole minibatches and cycles through them on
  repeated proposals in an iteration.
  Set `val_evaluation_policy` to choose an ordered subset of validation rows
  for each accepted candidate. The seed still uses all rows unless the policy
  supplies `seed_batch`. With a custom policy, `result.val_ids` pairs each
  score with its original validation row. Rows that no candidate evaluated
  have no Pareto leaders or best outputs. Switching back to full validation on
  restart keeps `result.val_ids` populated if any archived candidate was
  evaluated on only a subset. Merge sampling with partial coverage uses
  only IDs scored by both parents, with `merge_val_overlap_floor`
  controlling the required overlap.
  The optional `get_best_program` and `get_valset_score` validation hooks run
  on accepted children and at optimization end, in DSPy 3.5's diagnostic
  order. Callback errors abort compilation, and invalid best indices fail
  explicitly. Diagnostic scores and picks do not replace the raw aggregates
  or change the returned program; external experiment loggers are not invoked.
  A custom `component_selector` can inspect borrowed candidate instructions,
  validation scores, and the current minibatch's traced rollouts before
  returning predictor names in reflection order. An empty selection skips
  reflection; unknown or repeated names raise `InvalidComponentSelection`.
  The selector is not called when a perfect minibatch is skipped.
  GEPA passes the captured trace to predictor feedback, not to the program
  scoring metric, even when scoring the same traced rollout.
  GEPA warns at most once per compile if the predictor feedback score differs
  from the program score. `warn_on_score_mismatch = false` suppresses that
  diagnostic without changing feedback or selection. The result records the seed.
  `stop_callbacks` can stop a compile after seed evaluation or between
  complete iterations. Native callbacks see a borrowed, read-only state and
  run in caller order until one returns true, before the metric-budget check.
  They do not interrupt an in-flight evaluation or replace the metric budget.
  If the student returns `error.Canceled` during a rollout, `GEPA.compile`
  now returns that error instead of recording a failed prediction and
  continuing. With the native rollout pool, cancellation stops queued work
  and joins callbacks already running; those callbacks are not forcibly
  interrupted. Ordinary rollout failures still use `failure_score`.
  Pinned DSPy 3.5.0 propagated `asyncio.CancelledError` from seed evaluation.
  The native regression previously returned a successful result with two
  zero scores; it now checks cancellation in both serial and pooled
  compilation. A standalone consumer also skipped the third of three
  rollouts and successfully reused the same pool for another compile.
  Debug and ReleaseSafe each passed 859 tests and 40 build steps, including
  examples. The offline reference gate and native replay passed, including
  all 687 mock requests, eight adapter cases, and nine pipelines. No live
  model was needed and no blocker remains for student rollouts.
  A custom reflection LM can return `error.Canceled` from generation.
  `GEPA.compile` propagates it without retrying the proposal or evaluating a
  child, including when a failed batch backend falls back to built-in
  reflection. Ordinary proposal failures retain their retry and skip behavior.
  Pinned DSPy 3.5.0 / GEPA 0.1.4 and a standalone native `GEPA.compile`
  consumer both stopped after two task calls and one reflection call.
  The retained regression returned a successful result before the fix;
  it now passes for both direct reflection and batch fallback. The focused
  Debug check and all 23 tests in the focused ReleaseSafe run passed.
  The full Debug build and native mock replay hit `NoSpaceLeft`; the full
  ReleaseSafe build and offline reference gate were not run. No live model
  was needed. After freeing build space, rerun those gates, then extend
  cancellation propagation to hosted OpenAI reflection over HTTP.
  With `GEPA.Options.log_dir` configured, creating `gepa.stop` in that
  directory also requests a stop at the next iteration boundary. GEPA checks
  the marker after caller callbacks and before the metric budget. It leaves the
  marker in place; remove it before restarting to continue optimization.
  `GEPA.Options.log_dir` also saves seed validation outputs
  under `generated_best_outputs_valset/task_<row>/iter_0_prog_0.json`, even
  without `track_best_outputs`. A failed seed row saves `{}`. When tracking is
  enabled, later strict per-row improvements save outputs with their iteration
  and candidate number. A reflective child's parse failure saves a JSON string
  with the `FailedPrediction` diagnostic, including the raw completion, even
  when validation reuses a cached reflection row. A child failure without
  a prediction or parse diagnostic does not write a file.
  Ties remain in memory without writing new files. These writes are atomic,
  observe `log_dir.max_file_bytes`, and do not add model calls. To direct
  predictions elsewhere or save them without run logs, set
  `GEPA.validation_output_dir = .{ .io = io, .path = path }` explicitly.
  Use a dedicated trusted directory for each run; prediction files may contain
  sensitive data.
  To resume a GEPA compile in a new process, set `GEPA.Options.checkpoint` to
  `.{ .io = io, .path = checkpoint_path, .revision = "experiment-1" }` and
  compile again with the same student, ordered datasets, and scoring options.
  Change the revision when other student or scoring configuration changes;
  use a new checkpoint path when starting over. Version 26 checks digests
  of the ordered training and effective validation rows (including input
  keys, labels and record order), each named predictor's ordered field names,
  input/output split, live descriptions and prefixes, field kinds, annotation
  names, schema bytes, nullability, literal choices, declared defaults and
  declared decision bindings (including criteria and answer spaces) by value,
  stored demos, `PredictState.config`, `PredictState.decision_fields`
  (including criteria overrides and decoding settings), its effective task LM's
  reasoning, function-calling, native decision, response-format, and
  response-schema capabilities and name (when reported by `LM.modelName()`),
  and the compile context's
  `demos_override`, `predict_config`, and base `lm_options`. Parallel runs check
  effective predictor LMs in attempt-slot order instead of checking the
  shadowed base LM. Serial and parallel task configurations are distinct.
  With `TwoStepAdapter`,
  it also checks `extraction_options` by value and checks the extraction LM's
  name and those capabilities. It records the compile context's
  `adapter`, `json_fallback`, and effective `use_native_function_calling`
  policy. An explicit setting equal to the adapter default is compatible.
  The checkpoint also requires the exact same `failure_score`.
  An empty demo override is distinct from null.
  At either config layer, an empty `PredictConfig` inherits all
  settings, just like no config; an explicit clear is distinct from
  inheriting that option. A mismatch
  returns `IncompatibleGepaCheckpoint` before rerunning the seed or
  rewriting the checkpoint. Versions 1–25 are not accepted by the version 26
  reader; use a fresh checkpoint path when upgrading.
  Update the revision for same-name or unnamed LM changes (including provider
  routing), rollout retriever changes, custom validator or model behavior changes
  not represented by field metadata and default values, and other unchecked
  configuration. Apart from the registered Code bindings below, the fingerprint
  does not identify nominal classes; it does not track object aliasing.
  Opaque or over-deep defaults that cannot be fingerprinted
  return `UnsupportedGepaFieldDefaultValue` with checkpointing enabled.
  With checkpointing
  enabled, boxed custom values or value nesting deeper than 128 return
  `UnsupportedGepaValidationValue` for validation rows,
  `UnsupportedGepaTrainingValue` for a separate training set,
  `UnsupportedGepaDemoValue` for stored or override demos,
  `UnsupportedGepaPredictConfigValue` for stored or override configs (including
  decision settings that cannot be fingerprinted), or
  `UnsupportedGepaLMOptionsValue` for base or TwoStep extraction LM options.
  The metric budget can increase. On a compatible restart, seed validation
  runs again, but its scores and calls do not replace the archived scores
  or count against the resumed budget.
  The bounded, versioned native JSON file is atomically replaced between
  iterations. Keep it private: it can contain instructions and predictions.
  Use one writer per path. The checkpoint preserves boxed `Reasoning`, base
  `Code`, valid `BoundingBox`, `Citations`, `Citations.Citation`, and validated
  base `Noul`, `Score`, and `Choice` values in stored prediction fields and
  completions, including cached evaluations. `Noul` preserves its boolean
  value, confidence, and optional provider probability. `Score` preserves its
  numeric value, confidence, optional level, and integer-keyed probabilities.
  `Choice` preserves the type of its selected option, confidence, and
  string-keyed provider probabilities.
  Tuples, including fixed-size Zig arrays converted to runtime values, are
  restored as tuples even inside ReAct tool results. Plain lists and
  dictionaries keep their types. Citation records preserve quoted text,
  source metadata, and exact integer offsets and document indexes within the
  checkpoint's JSON limits. For a specialized Code class, set
  `Checkpoint.code_classes` to `&.{GEPA.CheckpointCode.bind(MyCode, "my-code")}`.
  Register each concrete
  class separately, even if several use the same language. Keep the same
  ordered bindings and IDs on restart; a change in the registered IDs, type
  names, or languages is incompatible. Ordinary strings matching the code
  text remain strings. For a `NoulWithCriteria` class, set
  `Checkpoint.noul_classes` to `&.{GEPA.CheckpointNoul.bind(MyNoul, "my-noul")}`.
  For a `ScoreWithLevels` class, set `Checkpoint.score_classes` to
  `&.{GEPA.CheckpointScore.bind(MyScore, "my-rubric")}`. For a
  `ChoiceWithOptions` class, set `Checkpoint.choice_classes` to
  `&.{GEPA.CheckpointChoice.bind(MyChoice, "my-category")}`. Use the same
  ordered bindings, IDs, and criteria on restart. Base `Noul`, `Score`, and
  `Choice` do not need bindings. The file remains version 26.
  Existing files remain readable with
  the same registrations; older builds cannot resume files containing markers
  they do not recognize.
  Unregistered specialized Code and other unsupported boxed custom values in
  stored predictions return `UnsupportedGepaCheckpointValue`. Callback-owned
  state is not saved.
  The file does not import Python pickle.
  Set `GEPA.Options.log_dir = .{ .io = io, .path = path }` to write
  `candidates.json` and, after the first iteration, `run_log.json`. When the
  run finds a new candidate, it also writes `candidate_tree.html`: an offline
  lineage report with linked parents, validation scores, frontier roles, and
  escaped instructions. It does not load scripts, a CDN, or a graph renderer.
  Failure to write this optional report warns without aborting optimization.
  The records include selected parents, sampled row IDs, before/after scores,
  and accepted candidates. Each file is individually bounded and atomically
  replaced; the files are not a transactional bundle. Use a fresh, private
  output directory per run, with one writer and no checkpoint file named
  `candidates.json`, `run_log.json`, `candidate_tree.html`, or `gepa.stop`. Keep logging
  enabled across restarts to carry the earlier iterations forward from the
  checkpoint. These files do not add model calls or invoke external
  experiment loggers.
  Debug and ReleaseSafe matched the parsed candidate and run logs from DSPy
  3.5.0 for three two-process restart workloads and a four-proposal iteration
  with rejected proposals and reversed validation order.
  The validation-file regression reproduced a missing seed file before the
  fix. Pinned DSPy 3.5.0/GEPA 0.1.4 and a standalone native `GEPA.compile`
  consumer then agreed on the parsed seed and improved prediction files,
  with four metric calls. The regression also covers tracking disabled,
  the explicit directory override, and preservation of an existing file
  when a replacement exceeds the byte limit. Debug and ReleaseSafe each
  passed 845 tests and 40 build steps; native replay matched all 687 mock
  requests, eight adapter cases, and nine pipelines. The full reference
  gate stopped with `NoSpaceLeft` in the Zig cache before completing;
  the targeted DSPy artifact probe passed separately.
  A follow-up failure probe reproduced native `{}` in place of the expected
  child parse diagnostic. A standalone native `GEPA.compile` consumer then
  matched all five parsed validation files from the pinned DSPy/GEPA probe
  in each mode: ten metric calls with fresh validation and eight with cache
  reuse. The fixture also checks a failed seed row and no extra file for a tied
  failure. Both runtimes retain the regression. The focused Debug run passed
  all 20 tests, and the native replay matched 687 requests, eight adapter
  cases, and nine pipelines. The integrated Debug run and focused
  ReleaseSafe command ended with SIGTERM before reporting a final result;
  the full reference gate was not reached in this follow-up.
  The in-memory tracking follow-up reproduced an extra empty prediction for
  a failed child tied with the seed. The retained regression now checks the
  full distinction: an empty failed-seed prediction, a child parse diagnostic,
  and no output for a tied child whose metric failed. The pinned DSPy 3.5.0
  oracle and a standalone native `GEPA.compile` consumer agreed with fresh
  validation (ten calls) and cache reuse (eight). Focused Debug and ReleaseSafe
  each passed 20 tests; integrated Debug passed 846 tests and 40 build steps
  including examples. Native replay matched 687 requests, eight adapter cases,
  and nine pipelines. The full reference gate and live inference were not run
  for this change; the targeted oracle needed no external model.
  GEPA merge subsamples now use the last rollout's score for every occurrence
  of a repeated validation ID, even with caching disabled. Each actual rollout
  still counts against the budget. Before the fix, an earlier success masked
  a later parse failure in the uncached subsample score. Public `GEPA.compile`
  regressions in both runtimes also check the tracked output and artifact:
  cache reuse preserves the subsample's empty prediction (19 evaluations),
  while fresh full validation records its own parse diagnostic (25 evaluations).
  The pinned DSPy/GEPA checks regenerated both GEPA fixtures byte-for-byte.
  Focused Debug and ReleaseSafe each passed 21 tests; integrated Debug passed
  847 tests and 40 build steps including examples. Native replay matched 687
  requests, eight adapter cases, and nine pipelines. Integrated ReleaseSafe
  timed out before its summary, so the chained full reference gate was not
  reached. No live model was needed.
  On restart, `cache_evaluation = false` can now discard cached rollouts
  without discarding archived candidate scores or tracked outputs. Without a
  custom validation policy, archived candidates must cover the full valset;
  their rows are restored in valset order. Native checkpoint files remain
  separate from DSPy's pickle state.
  The retained cross-process `GEPA.compile` regression previously failed with
  `IncompatibleGepaCheckpoint` when caching was turned off. Pinned DSPy
  3.5.0/GEPA 0.1.4 and the native consumer now agree on candidate scores,
  selection, and tracked and saved parse diagnostics: keeping the cache
  retains the archived failure (six budgeted evaluations); disabling it
  observes the fresh failure (eight). Focused Debug and ReleaseSafe each
  passed 21 tests. Native replay matched all 687 requests, eight adapter
  cases, and nine pipelines. The full reference gate, integrated builds,
  and live inference were not run for this change.
  With standard full validation, restarting with `cache_evaluation = true`
  from an uncached checkpoint now preserves archived candidate scores and
  outputs and starts with an empty evaluation cache. Neither archived
  validation rows nor the rerun seed populate that cache.
  The retained two-process `GEPA.compile` regression reproduced
  `IncompatibleGepaCheckpoint` before the fix. Afterward, native Debug and
  ReleaseSafe matched pinned DSPy 3.5.0/GEPA 0.1.4: the fresh seed scores
  did not replace the archived scores, the newly cached reflection row was
  reused, and the previously uncached row produced a fresh parse diagnostic
  in memory and on disk. The resumed run used five model calls and ended at
  seven budgeted evaluations. The 21 focused tests passed in each mode, and
  native replay matched all 687 mock requests, eight adapter cases, and nine
  pipelines. Integrated builds, the full reference gate, and live inference
  were not run for this change.
  A child candidate's NaN score on a previously unseen validation row now
  stays in `Result.val_subscores` without creating a Pareto leader or a
  tracked best output for that row. The old behavior made that candidate
  the row's leader and prevented a later finite score from replacing it.
  The cross-process `GEPA.compile` regression keeps the same custom subset
  policy and checks ordered row IDs, leaders, tracked outputs, and saved
  parse diagnostics with caching retained, disabled, and newly enabled
  (seven, nine, and eight budgeted evaluations respectively). A further
  restart reconstructs the same frontier without charging the rerun seed.
  Debug and ReleaseSafe matched pinned DSPy 3.5.0/GEPA 0.1.4 in all 11
  restart-oracle phases. The integrated Debug build passed 847 tests and
  all 40 steps including examples; native replay matched all 687 requests,
  eight adapter cases, and nine pipelines. Integrated ReleaseSafe, the full
  reference gate, and live inference were not run. This does not qualify
  changing the validation policy itself across a restart.
  Sparse GEPA merges now sample shared validation rows in CPython 3.11 set
  intersection order, not numeric order. In the cross-process regression, one
  parent is restored from a checkpoint and the other is evaluated under the
  same subset policy. Their shared rows are 0 and 8. The old sort changed
  the five merge evaluations from the reference's `0, 8, 0, 0, 0` to
  `8, 0, 8, 8, 8`, even though lineage and final scores matched in this case.
  The public `GEPA.compile` regression now matches ordered requests, candidate
  scores, lineage, winner, and metric budget (7 before restart, 20 after) in
  Debug and ReleaseSafe. The Debug build passed all 847 tests and 40 steps.
  The full offline reference gate passed, including fixture regeneration,
  all 687 mock requests, and both existing DSPy 3.5.0 live transcripts in
  offline replay. The full ReleaseSafe build was interrupted by SIGTERM;
  the focused ReleaseSafe restart check passed separately. No new live
  inference was needed.
  The offline reference gate now includes a cache-enabled, two-process
  `GEPA.compile` merge. All five initially uncached sample occurrences
  (`0, 8, 0, 0, 0`) are evaluated; subsequent candidate validation reuses
  rows 8 and 0 and evaluates only rows 1 and 2. The repeated row 0 returns
  a different score on its last rollout, so the check distinguishes reusing
  that final score (0.75) from reusing the first score or deduplicating misses.
  Native and pinned DSPy 3.5.0/GEPA 0.1.4 agree on ordered calls, validation
  scores, lineage, winner, and budget (6 before restart, 17 after). All 15
  restart-oracle phases passed in Debug; both new phases passed in
  ReleaseSafe. No optimizer change was needed. The broader build and
  reference gates were not rerun, and live inference was not needed.
  The restart oracle now also covers a rejected merge. In a three-process
  `GEPA.compile` run, the first merge attempt evaluated five sample
  occurrences of rows 0 and 8 and was rejected. After the next restart, a
  successful reflection scheduled the same merge again. The retry reused
  both checkpointed rows, made no merge LM calls, spent no merge metric
  budget, and was rejected again. DSPy 3.5.0/GEPA 0.1.4 and native agreed
  on ordered calls, merge sample IDs and scores, candidate lineage, winner,
  and the cumulative metric budget (6, 15, then 19).
  All 17 restart-oracle phases passed in Debug; the four cache-enabled
  phases passed in ReleaseSafe. Removing just the two rejected-merge rows
  from a disposable checkpoint caused the retry to attempt acceptance,
  confirming the regression detects cache loss. No optimizer change was
  needed. The broader build and reference gates were not rerun, and live
  inference was not needed.
  GEPA can now resume an uncached full-validation checkpoint with a custom
  subset policy on the same ordered valset. The new policy controls the
  rerun seed and future validation batches, not the archived scores or
  outputs. The two-process `GEPA.compile` regression previously failed
  with `IncompatibleGepaCheckpoint`. It now matches pinned DSPy 3.5.0 /
  GEPA 0.1.4 on ordered calls, scores, frontier leaders, tracked outputs,
  and the winner. The new candidate evaluated rows `2, 1, 0`; the original
  candidate remained the leader on the omitted row. The cumulative budget
  went from 6 to 11, excluding the two rerun seed calls.
  All 19 restart-oracle phases passed in Debug and ReleaseSafe. The
  integrated Debug build passed 847 tests and 40 steps including examples.
  The offline reference gate passed, including unchanged fixture
  regeneration and replay of all 687 mock requests, eight adapter cases,
  and nine pipelines.
  The reverse switch, from a subset policy to uncached full validation,
  now resumes partial archives rather than returning `InvalidGepaCheckpoint`.
  In the two-process `GEPA.compile` regression, the checkpoint retained
  seed scores for rows `2, 0` in that order, even though all four seed
  predictions changed on the unbudgeted rerun. The new candidate evaluated
  all four rows in valset order and won. The cumulative budget rose from
  2 to 8. Ordered calls, row IDs, scores, Pareto leaders, and best outputs
  (including a parse failure) matched pinned DSPy 3.5.0 / GEPA 0.1.4.
  The offline reference gate passed all 21 restart phases in Debug, and
  both new phases passed in ReleaseSafe. Integrated Debug and ReleaseSafe
  each passed 847 tests and 40 steps including examples; native replay
  matched all 687 requests, eight adapter cases, and nine pipelines.
  No blocker remains for the verified uncached transition. Live inference
  was not needed.
  The restart oracle now also resumes a sparse checkpoint with full
  validation and merges a restored parent covering three rows with a new
  parent covering all nine. The subsample was `1, 0, 8, 1, 0`, drawn only
  from their shared rows. The accepted merge then evaluated all nine rows
  in order and won; the cumulative metric budget went from 7 to 32.
  Pinned DSPy 3.5.0 / GEPA 0.1.4 and native `GEPA.compile` agreed on
  ordered calls, archived and new scores, lineage, winner, and merge-log
  samples and scores. No optimizer change was needed. The full offline
  reference gate passed all 22 restart phases in Debug; the three uncached
  merge phases also passed in ReleaseSafe. No live inference was needed.
  The restart oracle also tests enabling `cache_evaluation` while resuming
  that same uncached sparse checkpoint with full validation. The archived
  scores and row order stay intact. Full validation reuses reflection row 2;
  the winning merge reuses subsample rows 0, 1, and 8. Both candidate score
  lists report cache hits before misses. The cumulative budget rises from
  7 to 28 rather than 32, with the nine rerun seed evaluations uncharged.
  Native `GEPA.compile` matched pinned DSPy 3.5.0 / GEPA 0.1.4 on ordered
  calls, scores, lineage, winner, and merge logs without an optimizer change.
  All 23 restart phases passed in Debug, and the four merge phases branching
  from the uncached checkpoint passed in ReleaseSafe. The broader build and
  full reference gate were not rerun; no external model was needed.
  The restart regression now branches from the same cached sparse checkpoint
  after a rejected merge. Keeping the cache reuses the rejected scores and
  issues no merge LM calls. Switching to full validation with
  `cache_evaluation = false` instead reevaluates all five sampled occurrences,
  accepts the merge on the new scores, then validates it on all nine rows.
  The archived scores and their row order stay intact. The cumulative budget
  rises from 15 to 40, excluding the nine rerun seed evaluations.
  Native `GEPA.compile` matched pinned DSPy 3.5.0 / GEPA 0.1.4 on ordered
  calls, scores, lineage, winner, and merge logs without an optimizer change.
  All 24 restart phases passed in Debug; the five cached-checkpoint merge
  phases passed in ReleaseSafe. The broader build and full reference gate
  were not rerun; no live model was needed. No blocker remains for this case.
  The restart regression now also covers turning evaluation caching off,
  saving a checkpoint, and turning it back on. Resuming `GEPA.compile`
  preserves the archived candidate scores but does not revive the rejected
  merge's cached scores. The retry makes five fresh merge calls and reuses
  those new scores for validation; the merge wins. The cumulative metric
  budget moves from 15 to 19 to 28, excluding rerun seed calls. Native and
  pinned DSPy 3.5.0 / GEPA 0.1.4 agree on ordered calls, scores, lineage,
  winner, and merge logs. Reinjecting just the two stale rows into a
  disposable native checkpoint suppressed all five merge calls and kept
  the old winner, confirming that the regression catches stale cache reuse.
  The full offline reference gate passed all 26 restart phases in Debug,
  unchanged fixture regeneration, and mock replay. No optimizer change
  was needed and no blocker remains for this case. The integrated build,
  ReleaseSafe variant, and live inference were not run for this iteration.
  GEPA now rejects reordered same-length validation sets on resume. In the
  retained cross-process `GEPA.compile` regression, pinned DSPy 3.5.0 /
  GEPA 0.1.4 ran the reordered seed and kept the archived scores and outputs
  at their old row IDs; the previous native build also accepted the restart.
  Native now refuses before any LM calls and leaves the checkpoint byte-for-byte
  unchanged. The same checkpoint still resumes with the original order.
  Debug passed 847 tests and all 40 build steps, including examples. The
  offline reference gate passed the restart oracle, unchanged fixture
  regeneration, all 687 mock requests, and both existing live transcripts
  in offline replay. The focused ReleaseSafe restart smoke passed; the full
  ReleaseSafe build and new live inference were not run.
  The cross-process `GEPA.compile` regression now also keeps validation fixed
  while reordering the training rows or changing only a training label.
  DSPy 3.5.0 / GEPA 0.1.4 accepted the reordered training set and reflected
  on a different row at the same index, changing the winning candidate.
  Native previously accepted the restart too. It now rejects both training
  changes before any model call, leaves the checkpoint byte-for-byte intact,
  and still resumes with the original training set.
  Debug and ReleaseSafe each passed 847 tests and 40 build steps, including
  examples. The restart regression passed in both modes. The full offline
  reference gate passed, including unchanged fixture regeneration, all 687
  mock requests, and both existing live transcripts in offline replay. No
  new live inference was needed.
  A cross-process `GEPA.compile` regression now changes only the answer in a
  stored predictor demo, keeping the revision and datasets fixed. Pinned DSPy
  3.5.0 / GEPA 0.1.4 used the changed demo in four model calls but kept the
  archived seed scores and outputs. Before the fix, native also accepted the
  restart and made four calls. Version 4 now refuses before any model call,
  leaves the checkpoint byte-for-byte unchanged, and resumes with freshly
  reconstructed demos of identical content.
  The Debug build passed 847 tests and 40 steps including examples. The full
  offline reference gate passed, including unchanged fixture regeneration,
  all 687 mock requests, and both existing live transcripts in offline replay.
  The focused demo restart regression passed in ReleaseSafe; the full
  ReleaseSafe build and new live inference were not run in that iteration.
  The next cross-process `GEPA.compile` regression changed only the answer in
  `Context.demos_override`, leaving the stored demos, datasets, and revision
  unchanged. Pinned DSPy 3.5.0 / GEPA 0.1.4, using the explicit `Predict`
  `demos` keyword, made four calls with the new override and retained the old
  scores and outputs. Before the fix, native also accepted the restart and
  made four calls. Version 5 now refuses before any model call or checkpoint
  write, distinguishes an empty override from no override, and resumes with
  a separately reconstructed override of identical content.
  Debug and ReleaseSafe each passed 847 tests and 40 steps including examples.
  The targeted restart regression passed in both modes. The full offline
  reference gate passed, including unchanged fixture regeneration, all 687
  mock requests, and both historical 3.5.0 live transcripts in offline replay.
  No new live inference was needed.
  The next cross-process `GEPA.compile` regression changed only the
  `Context.predict_config` temperature from 0.25 to 0.75. Pinned DSPy
  3.5.0 / GEPA 0.1.4, using the explicit `Predict` `config` keyword, made
  four calls at the new temperature but kept the archived seed scores and
  outputs. The previous native build also accepted the restart and made
  four calls. Version 6 now refuses before any model call and leaves the
  checkpoint byte-for-byte unchanged. A separately reconstructed config
  with the same settings resumes and matches the reference request
  temperatures, scores, outputs, leaders, and metric budget. Version 5
  checkpoints without the digest are refused; start a new run instead.
  The focused regression passed in Debug and ReleaseSafe; the integrated
  builds in both modes passed 847 tests and all 40 steps including examples.
  The full offline reference gate passed, with unchanged fixtures, 687 mock
  requests, and both historical live transcripts replayed offline. No new
  inference was needed.
  The next cross-process `GEPA.compile` regression changed only the stored
  `PredictState.config` temperature from 0.25 to 0.75, without a context
  override. Pinned DSPy 3.5.0 / GEPA 0.1.4 made four calls at the new
  temperature while retaining archived seed scores and outputs; the old
  native build also made four calls. Version 7 now refuses before any model
  call and leaves the checkpoint byte-for-byte unchanged. A separately
  reconstructed config with identical settings resumes and matches the
  reference temperatures, scores, outputs, leaders, and metric budget.
  The offline reference gate passed the cross-process regression and version
  6 refusal check, unchanged fixture regeneration, all 687 mock calls, and
  both historical live transcripts in offline replay. Debug and ReleaseSafe
  each passed 847 tests and all 40 steps including examples. No new inference
  was needed.
  The cross-process `GEPA.compile` regression now also changes only the
  base `Context.lm_options.temperature` from 0.25 to 0.75. Pinned DSPy
  3.5.0 / GEPA 0.1.4, with the temperature set on the settings LM, made
  four calls at 0.75 while retaining the archived seed scores and outputs.
  Before the fix, native accepted the restart and made four calls. Version
  8 fingerprints base task LM options by value and refuses the changed
  restart before any model call, leaving the checkpoint byte-for-byte
  unchanged. With the original options, restart matches the reference
  temperatures, scores, tracked outputs, frontier leaders, winner, and
  metric budget. The regression also verifies version 7 refusal.
  Debug and ReleaseSafe each passed 847 tests and 40 build steps including
  examples. The restart regression passed in both modes. The full offline
  reference gate passed, including unchanged fixtures, all 687 mock
  requests, and both historical live transcripts in offline replay. No
  new inference was needed.
  The cross-process `GEPA.compile` regression now changes only the task
  adapter from XML to Chat. Pinned DSPy 3.5.0 / GEPA 0.1.4 made four calls
  with Chat prompts but kept the archived XML seed scores and outputs.
  The previous native build also accepted the restart and made four calls.
  Version 9 now rejects the change before any model call and leaves the
  checkpoint byte-for-byte unchanged. With XML unchanged, the resumed
  prompts' adapter format, scores, tracked outputs, frontier, winner, and
  metric budget match the reference. The regression also verifies refusal
  of version 8 checkpoints.
  Debug and ReleaseSafe each passed 847 tests and 40 build steps including
  examples. The restart regression passed in both modes. The full offline
  reference gate passed, including unchanged fixtures, all 687 mock
  requests, and both historical live transcripts in offline replay. No
  new inference was needed.
  The next cross-process `GEPA.compile` regression replaced the predictor's
  bound task model while leaving the settings LM unchanged. Pinned DSPy
  3.5.0 / GEPA 0.1.4 and the previous native build each made four calls with
  the replacement but kept the original seed scores and outputs. Version 10
  rejects the changed name before any model call and preserves the checkpoint
  byte-for-byte. A separately constructed LM with the original name resumes
  and matches the reference scores, tracked outputs, frontier, winner, and
  metric budget. The regression also verifies version 9 refusal.
  Debug and ReleaseSafe each passed 847 tests and 40 build steps including
  examples, and the targeted restart regression passed in both modes. The
  full offline reference gate passed, including unchanged fixture regeneration,
  all 687 mock requests, and both existing live transcripts in offline replay.
  No new inference was needed.
  The cross-process `GEPA.compile` regression now enables JSON fallback
  without changing the XML adapter. Pinned DSPy 3.5.0 / GEPA 0.1.4 retried
  the malformed seed response with JSON and kept the archived scores and
  outputs. The previous native build also accepted the restart and made
  three model calls. Version 11 rejects the change before any model call
  and leaves the checkpoint byte-for-byte unchanged. With fallback still
  disabled, restart matches the reference prompt formats, scores, tracked
  outputs, frontier, winner, and metric budget. The regression also checks
  refusal of version 10 checkpoints.
  Debug and ReleaseSafe each passed 847 tests and all 40 build steps,
  including examples, and the restart regression passed in both modes.
  The full offline reference gate passed, including unchanged fixture
  regeneration, all 687 mock requests, and both existing live transcripts
  in offline replay. No new inference was needed.
  The cross-process `GEPA.compile` regression now disables native function
  calling for a task with `Tool` inputs and a `ToolCalls` output, keeping the
  XML adapter and LM fixed. Pinned DSPy 3.5.0 / GEPA 0.1.4 reran two seed
  requests without tool declarations but kept the archived scores and
  outputs. The previous native build also accepted the restart and made two
  model calls. Version 12 refuses before inference and leaves the checkpoint
  byte-for-byte unchanged. A checkpoint created with explicit `true` resumes
  under the equivalent XML default and matches the reference tool
  declarations, adapter format, scores, outputs, frontier, winner and budget.
  Version 11 checkpoints are rejected.
  Debug and ReleaseSafe each passed 847 tests and all 40 build steps,
  including examples. The restart regression passed in both modes. The
  full offline reference gate passed, including unchanged fixture
  regeneration, all 687 mock requests, and both existing live transcripts
  in offline replay. No new inference was needed.
  A cross-process `GEPA.compile` regression now changes the bound task LM's
  function-calling capability while keeping its name and the XML adapter's
  native tool policy fixed. Pinned DSPy 3.5.0 / GEPA 0.1.4 reran two seed
  requests without tool declarations and retained the archived scores and
  outputs. The previous native build also made two calls. Version 13 refuses
  before any model call and leaves the checkpoint byte-for-byte unchanged.
  A separately constructed bound LM with the original capability still
  resumes and matches the reference, even when the unused settings LM's
  capability changes. Version 12 checkpoints are refused.
  Debug and ReleaseSafe each passed 847 tests and all 40 build steps,
  including examples. The restart regression passed in both modes, and the
  full offline reference gate passed, including unchanged fixture regeneration
  and offline replay. No fresh live inference was needed.
  The cross-process `GEPA.compile` regression now disables native reasoning
  on the bound LM of a `ChainOfThought` task without changing its name.
  Pinned DSPy 3.5.0 / GEPA 0.1.4 reran two seed requests with an explicit
  `<plan>` output but retained the archived scores and outputs. The previous
  native build also made two calls. Version 14 rejects the change before
  any model call and preserves the checkpoint byte-for-byte. An unchanged
  bound LM resumes and matches the reference prompt format, tracked outputs,
  scores, frontier, winner, and metric budget even when the unused settings
  LM changes capability. Version 13 checkpoints are refused.
  Debug and ReleaseSafe each passed 847 tests and all 40 build steps, including
  examples. The restart regression passed in both modes, and the full offline
  reference gate passed with unchanged fixtures and replay. No new inference
  was needed. That regression uses native responses without reasoning summaries.
  A separate two-process `GEPA.compile` regression now uses nonempty native
  summaries. Before the fix, writing the first checkpoint failed with
  `UnsupportedGepaCheckpointValue`. Version 15 restores the archived summary
  as a `Reasoning` instance even when fresh seed calls return different text.
  Debug and ReleaseSafe matched pinned DSPy 3.5.0 / GEPA 0.1.4 on ordered
  calls, archived outputs, scores, frontier leaders, winner, and the
  four-evaluation budget. Integrated Debug passed 847 tests and all 40
  steps, including examples. The offline reference gate passed the restart
  oracle, unchanged fixture regeneration, artifact round-trip, and all
  687 mock requests. The combined command reached its timeout after mock
  replay; the two remaining historical transcripts passed separately in
  both runtimes. No new live inference or integrated ReleaseSafe build ran.
  A cross-process `GEPA.compile` regression now replaces only the
  `TwoStepAdapter` extraction LM. Pinned DSPy 3.5.0 / GEPA 0.1.4
  extracted `high` instead of `low` for both fresh seed rows but retained
  the archived scores and outputs. The previous native build accepted the
  change and made two main and two extraction calls. Version 16 rejects
  the replacement before either LM runs and leaves the checkpoint
  byte-for-byte unchanged. A separately constructed extractor with the
  original identity resumes and matches the reference's ordered main and
  extraction calls, archived outputs, scores, frontier, winner, and metric
  budget. The regression also verifies that version 15 is refused.
  Debug and ReleaseSafe each passed 847 tests and all 40 build steps,
  including examples. The full offline reference gate passed the new
  regression, unchanged fixture regeneration, all 687 mock requests, and
  replay of both existing live transcripts. No new inference was needed.
  A second cross-process regression changes only the extraction temperature
  from 0.25 to 0.75, with the same extractor name. DSPy 3.5.0 / GEPA 0.1.4
  used the new temperature and extracted `high` for both fresh seed rows,
  while keeping the archived scores and outputs. The old native build
  accepted the change and made two main and two extraction calls.
  Version 17 includes the active adapter's extraction options in the
  checkpoint digest, alongside the base LM options. The regression passed
  in Debug and ReleaseSafe: a changed temperature was refused before either
  LM ran, the checkpoint stayed byte-for-byte unchanged, and an unchanged
  temperature resumed with matching request order, extraction temperatures,
  scores, outputs, frontier, winner, and budget. Both integrated builds passed
  847 tests and all 40 steps. The offline reference gate passed fixture
  regeneration, artifact round-trip, all 687 mock requests, and both recorded
  live transcripts in both runtimes. No new live inference was needed.
  The parallel-rollout restart regression found that DSPy 3.5.0 / GEPA 0.1.4
  reran both seed rows with the replacement LM but kept archived results.
  The old native build likewise made two calls after only the rollout LM
  changed. Version 18 checks each attempt slot's effective predictor LM
  identity. The public `GEPA.compile` regression verifies refusal before any
  model calls, byte-for-byte preservation of the checkpoint, and refusal of
  version 17. An unchanged rollout LM resumes even if the shadowed base LM
  changes, matching the reference's seed calls, scores, outputs, frontier,
  winner, and metric budget. The regression passed in Debug and ReleaseSafe;
  both integrated builds passed 847 tests and 40 steps. The offline reference
  gate passed, including fixture regeneration and all 687 mock requests.
  No new live inference was needed. Same-name and unnamed LM changes,
  retrievers, and unchecked scoring settings still rely on the caller's
  revision.
  The next cross-process `GEPA.compile` regression changed `failure_score`
  from 0.25 to 0.75. Pinned DSPy 3.5.0 / GEPA 0.1.4 kept the seed
  candidate's archived 0.25 parse-failure score but scored a new candidate's
  parse failure at 0.75, changing the row's frontier leaders. The previous
  native build also accepted the restart and made six model calls. Version
  19 refuses the change before any model call and preserves the checkpoint
  byte-for-byte. The retained regression also verifies that an unchanged
  score resumes with the reference's calls, scores, outputs, frontier,
  winner, and metric budget, and that a version 18 checkpoint is refused.
  The regression passed in Debug and ReleaseSafe. Both integrated builds
  passed 847 tests and 40 steps, and the offline reference gate passed
  fixture regeneration and both runtimes' replay of all 687 mock requests
  and the saved 3.5.0 live transcripts. No new live inference was needed.
  Caller-owned metric and retriever changes still require a revision change.
  The cross-process `GEPA.compile` regression now changes only a predictor's
  answer-field description, keeping its name and instructions fixed. Pinned
  DSPy 3.5.0 / GEPA 0.1.4 sent four prompts with the revised description but
  kept the old validation scores and outputs. The previous native build also
  accepted the restart and made four calls. Version 20 rejects it before
  any model call, leaves the checkpoint byte-for-byte unchanged, and rejects
  version 19 checkpoints. With the original description reconstructed by
  value, the native resume matches the reference's ordered calls, scores,
  outputs, frontier, winner, and metric budget. The regression passed in Debug
  and ReleaseSafe. Both integrated builds passed 847 tests and 40 steps,
  including examples. The offline reference gate regenerated the fixtures and
  both runtimes matched all 687 mock requests and both saved 3.5.0 transcripts.
  No new live inference was needed.
  A cross-process `GEPA.compile` regression changed only the `answer`
  annotation from string to integer. Pinned DSPy 3.5.0 / GEPA 0.1.4 put an
  archived string and a newly evaluated integer on the same validation row's
  frontier. The previous native build also accepted the restart and made six
  model calls. Version 21 rejects the change before any model call and leaves
  the checkpoint byte-for-byte unchanged. The regression also checks version
  20 refusal and compares the unchanged restart's ordered calls, scores,
  outputs, frontier, winner, and metric budget against the reference.
  This passed in Debug and ReleaseSafe, as did all 847 tests and 40 build
  steps in each mode. The full offline reference gate regenerated unchanged
  fixtures, and both runtimes replayed all 687 mock requests and both saved
  DSPy 3.5.0 transcripts. No new live inference was needed.
  A cross-process `GEPA.compile` regression changed only the default answer
  from `low` to `high`. The scripted XML response omitted `answer` on one
  validation row. DSPy 3.5.0 / GEPA 0.1.4 retained the seed's archived score
  of 0 on that row but scored the new candidate 1. The old native build
  accepted the restart and made six model calls. Version 22 rejects it before
  any model call, preserves the checkpoint byte-for-byte, and refuses version
  21 checkpoints. With the original default, the native resume matches the
  reference's ordered calls, scores, outputs, frontier, winner and metric
  budget. The retained restart regression passed in Debug and ReleaseSafe.
  Both integrated builds passed 847 tests and 40 steps, including examples.
  The offline reference gate regenerated unchanged fixtures, round-tripped
  the program artifact, and both runtimes replayed all 687 mock requests
  and both saved DSPy 3.5.0 transcripts. No live inference was needed.
  A cross-process `GEPA.compile` regression then changed only a runtime
  decision criterion, from excluding to including broken cases. Pinned DSPy
  3.5.0 / GEPA 0.1.4 used the new rule in six prompts but compared the new
  candidate's score of 1 against the seed's archived 0 on the affected row.
  The previous native build accepted the restart and made six model calls.
  Version 23 fingerprints each predictor's stored decision configuration and
  refuses the changed criterion before any model call, leaving the checkpoint
  byte-for-byte unchanged. It also refuses version 22 checkpoints. With the
  original criterion reconstructed by value, the restart matches the
  reference's prompt criteria, ordered calls, scores, outputs, frontier,
  winner, and metric budget.
  The retained regression passed in Debug and ReleaseSafe. Both integrated
  builds passed 847 tests and 40 build steps, including examples. The offline
  reference gate regenerated unchanged fixtures, round-tripped the program
  artifact, and replayed all 687 mock requests and both saved DSPy 3.5.0
  transcripts in both runtimes. No new live inference was needed.
  A cross-process `GEPA.compile` regression changed only the criterion on a
  boolean output's `Annotated` Noul binding, without setting
  `PredictState.decision_fields`. DSPy 3.5.0 / GEPA 0.1.4 used the new criterion
  in six calls and compared new candidate scores with the archived seed scores.
  The previous native build also accepted the restart and made six calls.
  Version 24 rejects the change before any model call and preserves the
  checkpoint byte-for-byte. The regression also verifies version 23 refusal
  and matches the reference's ordered calls, criteria, scores, tracked
  outputs, frontier, winner, and metric budget when the criterion is unchanged.
  The regression passed in Debug and ReleaseSafe. The integrated Debug build
  passed 847 tests and all 40 steps, including examples. Native replay
  consumed all 687 requests and matched eight adapter cases and nine pipelines.
  The full reference gate and integrated ReleaseSafe build were not rerun;
  no live inference was needed. Caller-owned metrics, retrievers and custom
  validation behavior not represented in field metadata still require a
  revision change.
  A cross-process `GEPA.compile` regression changed only the bound task LM's
  `supports_decision_requests` flag, keeping the model name, criterion, and
  datasets fixed. Pinned DSPy 3.5.0 / GEPA 0.1.4 made six native judgment
  calls and scored the new candidate against archived generative seed scores.
  The old native build also accepted the restart and made six model calls.
  Version 25 includes the flag in the effective LM digest.
  The retained regression passed in Debug and ReleaseSafe: the changed run
  was refused before any model call, the checkpoint stayed byte-for-byte
  unchanged, version 24 was refused, and the unchanged resume matched
  ordered calls, criteria, scores, tracked outputs, frontier, winner, and
  metric budget. Debug passed 847 tests and all 40 build steps. ReleaseSafe
  initially failed the OpenAI non-200 buffered/streamed test; the rerun
  passed all 40 steps, with 721 tests rerun and the other test suites cached
  from the first run. The cause of that first failure remains undiagnosed.
  The full offline reference gate passed fixture regeneration, artifact
  round-trip, all 687 mock requests, and both saved 3.5.0 transcripts. No new
  live inference was needed.
  A cross-process `GEPA.compile` regression then changed either
  `supports_response_schema` or `supports_response_format` on the same-name
  bound task LM. Pinned DSPy 3.5.0 / GEPA 0.1.4 switched from JSON-object
  requests to strict schema requests or requests with no response format,
  respectively, and compared fresh candidate scores with the archived
  JSON-object seed scores. The old native build accepted both changes and
  made six model calls in each case. Version 26 fingerprints both flags and
  rejects either change, and version 25 checkpoints, before any model call
  or checkpoint write. With the capabilities unchanged, the native resume
  matched request order and formats, scores, tracked outputs, frontier,
  winner, and metric budget.
  The regression passed in Debug and ReleaseSafe. The integrated Debug
  build passed 847 tests and all 40 steps, including examples. The full
  offline reference gate passed fixture regeneration, artifact round-trip,
  all 687 mock requests, and both saved 3.5.0 transcripts. The integrated
  ReleaseSafe build was not rerun for that checkpoint change; no new live
  inference was needed. No blocker remains for that checkpoint case.
  GEPA checkpoints now retain integer predictions outside the `i64` range
  as exact JSON numbers, without converting them to floats. The checkpoint
  format remains version 26; the runtime's 4,300-digit serialization limit
  still applies, and arbitrary non-integer raw number strings remain unsupported.
  The old build failed to save the first checkpoint with
  `UnsupportedGepaCheckpointValue`. A two-process `GEPA.compile` regression
  matched pinned DSPy 3.5.0 / GEPA 0.1.4 on ordered calls, exact archived and
  new outputs, scores, frontier, winner, and metric budget. Native completion
  records also retained the exact integers. The regression passed in Debug
  and ReleaseSafe, the full GEPA restart oracle passed in Debug, and the
  integrated Debug build passed 859 tests and all 40 steps. Native mock
  replay matched all 687 requests, eight adapter cases, and nine pipelines.
  The full ReleaseSafe build, broader reference gate, and live inference
  were not run for that integer change.
  GEPA now saves and restores unparameterized `Code` outputs as typed values.
  The old build rejected the initial save with `UnsupportedGepaCheckpointValue`.
  The retained two-process `GEPA.compile` regression matched DSPy 3.5.0 /
  GEPA 0.1.4 on call order, archived and new outputs, scores, frontier, winner,
  and metric budget. Native checks confirmed the exact Code type in prediction
  fields and completions while identical plain text stayed a string. An
  existing version 26 checkpoint resumed successfully, and the same non-Code
  run still wrote byte-identical checkpoint data.
  The regression passed in Debug and ReleaseSafe. Both integrated builds
  passed 859 tests and all 40 steps, including examples. The offline reference
  gate passed fixture regeneration, artifact round-trip, and replay of all
  687 mock requests and both saved 3.5.0 transcripts in both runtimes. No new
  live inference was needed.
  GEPA can now checkpoint registered language-specialized Code classes.
  Before the change, a typed Java Code output blocked the first save with
  `UnsupportedGepaCheckpointValue`. The retained two-process `GEPA.compile`
  regression uses two distinct classes with the same language and model
  name. Pinned DSPy 3.5.0 / GEPA 0.1.4 preserved their distinction across
  restart, though its reconstructed archived classes were not identical to
  the new process's declarations. Native restore uses the exact registered
  Zig types in prediction fields and completions, leaving identical plain
  strings unboxed. The runs matched call order, archived and new outputs,
  scores, frontier, winner, and metric budget. Swapped bindings and an
  invalid class index were both refused before any model call, and neither
  refusal rewrote the checkpoint.
  The focused Debug regression and full offline reference gate passed,
  including the 687 mock requests and both recorded 3.5.0 transcripts.
  Integrated Debug and ReleaseSafe passed 859 tests and all 40 steps each.
  No new live inference was needed and there is no blocker for registered Code.
  GEPA now saves and restores typed `BoundingBox` predictions, including boxes
  nested in lists. Previously, the first checkpoint write failed with
  `UnsupportedGepaCheckpointValue`. The two-process `GEPA.compile` regression
  matched pinned DSPy 3.5.0 / GEPA 0.1.4 on ordered calls, archived and new
  geometry, scores, frontier, winner, and metric budget. Native checks
  confirmed the type in prediction fields and completions while an identical
  dictionary stayed plain. Invalid geometry and overlapping type markers were
  refused before any model call and did not rewrite the checkpoint. The
  focused restart regression passed in Debug and ReleaseSafe.
  Integrated Debug and ReleaseSafe passed 859 tests and all 40 build steps
  each. The offline reference gate passed, including unchanged fixture
  regeneration, artifact round-trip, and replay of the 687 mock requests and
  both saved 3.5.0 transcripts in both runtimes. No live inference was needed.
  GEPA now checkpoints `Citations` and individual `Citations.Citation` values.
  The old build failed on the first save with `UnsupportedGepaCheckpointValue`.
  The retained two-process `GEPA.compile` regression matched DSPy 3.5.0 /
  GEPA 0.1.4 on call order, archived and new citation records, scores,
  frontier, winner, and metric budget. It covers provider-supplied references,
  a nested text-parsed citation with an integer beyond `i64`, and a plain
  dictionary that remains unboxed. Native checks cover both prediction fields
  and completions. Corrupt integer fields and overlapping or out-of-range
  type markers were refused before model calls without rewriting the file.
  Integrated Debug and ReleaseSafe builds each passed 859 tests and all 40
  steps. The full offline reference gate passed, including the citation
  restart regression, unchanged fixture regeneration, artifact round-trip,
  all 687 mock requests, and both recorded 3.5.0 live transcripts in offline
  replay. No new live inference was needed.
  GEPA now checkpoints validated base `Noul` values with provider-derived
  evidence. Previously, `GEPA.compile` failed on its first checkpoint write
  with `UnsupportedGepaCheckpointValue`. The retained two-process regression
  matched DSPy 3.5.0 / GEPA 0.1.4 on ordered judgment calls, archived and new
  outputs, validation scores, frontier, winner, and metric budget. Native
  checks verified the exact type, boolean value, confidence, and probability
  in prediction fields and completions. Invalid probabilities and overlapping
  or out-of-range type markers were refused before any model call without
  rewriting the file. Integrated Debug and ReleaseSafe each passed 859 tests
  and all 40 build steps. The full offline reference gate passed, including
  fixture regeneration, artifact round-trip, and replay of all 687 mock
  requests and both recorded 3.5.0 transcripts. No live inference was needed.
  GEPA now also checkpoints registered `NoulWithCriteria` classes. The former
  build failed on the first save with `UnsupportedGepaCheckpointValue`.
  Native restart restores the exact registered type in prediction fields
  and completions; DSPy's cloudpickle restores an equivalent dynamic
  subclass with the same name and criteria, not the fresh process's class
  object. The two-process `GEPA.compile` regression passed in Debug and
  ReleaseSafe against pinned DSPy 3.5.0 / GEPA 0.1.4: ordered judgment calls,
  archived and new evidence, scores, frontier, winner, and metric budget matched.
  A changed binding and four corrupt checkpoint variants were refused before
  any model call without rewriting the file. Both integrated builds passed
  859 tests and 40 build steps each. The full offline reference gate passed
  unchanged fixture regeneration, artifact round-trip, and both runtimes'
  replays of the 687 mock requests and both saved 3.5.0 transcripts. Its first
  attempt ran out of disk space; the retry completed. No live inference was
  needed.
  GEPA now checkpoints validated base `Score` and registered
  `ScoreWithLevels` values, retaining integer probability keys and the exact
  registered Zig type in prediction fields and completions. The retained
  two-process `GEPA.compile` regression matched pinned DSPy 3.5.0 / GEPA
  0.1.4 on ordered judgment calls, evidence, scores, frontier, winner, and
  metric budget (4 before restart, 8 after). A changed binding and six corrupt
  checkpoints were rejected before any model call without rewriting the file.
  The base `Score` regression also round-trips keys and a level beyond 64 bits
  without boxing an identical plain dictionary.
  Debug and ReleaseSafe each passed 861 tests and all 40 build steps,
  including examples. The cross-process regression passed in both modes;
  the full offline reference gate passed unchanged fixture regeneration,
  artifact round-trip, and replay of the mock suite and both recorded 3.5.0
  transcripts. No new live inference was needed.
  GEPA now checkpoints validated base `Choice` and registered
  `ChoiceWithOptions` values. Before the fix, `GEPA.compile` failed at the
  first save with `UnsupportedGepaCheckpointValue`. In Debug and ReleaseSafe,
  the retained two-process regression matched pinned DSPy 3.5.0 / GEPA 0.1.4
  on ordered judgment calls, archived and new choices, scores, frontier,
  winner, and metric budget (4 before restart, 8 after). It checked that
  integer and boolean options and string probability labels survived in
  prediction fields and completions. A changed binding and six corrupt
  checkpoints were refused before any model call without rewriting the file.
  A native regression also preserved wide integers and null, and left an
  identical plain dictionary unboxed.
  Debug and ReleaseSafe each passed 862 tests and all 40 build steps,
  including examples. The offline gate passed the restart oracle, unchanged
  fixture regeneration, artifact round-trip, and both runtimes' 687-request
  mock replay. The combined command timed out before the saved transcript
  replays; both saved 3.5.0 transcripts passed separately in both runtimes.
  No new live inference was needed. Erased mutable decision boxes remain
  unsupported.
  GEPA now saves ReAct histories containing tuple-valued tool results.
  Previously the first checkpoint save failed with
  `UnsupportedGepaCheckpointValue`. The retained two-process `GEPA.compile`
  regression matched pinned DSPy 3.5.0 / GEPA 0.1.4 on ordered calls,
  archived and new histories (including tool IDs, arguments, results, and
  reasoning), validation scores, winner, and metric budget (2 before
  restart, 6 after). A tuple and a list with identical elements kept their
  distinct types. A malformed tuple was refused before any model call
  without rewriting the checkpoint. The file remains version 26.
  Debug and ReleaseSafe each passed 862 tests and all 40 build steps,
  including examples, and the focused regression passed in both modes.
  The full offline reference gate passed, including the 687-request mock
  replay and both saved 3.5.0 transcripts in both runtimes. No new live
  inference was needed.
  `History.format` now uses an empty object when a native tool continuation
  omits `data`, matching DSPy 3.5.0. Previously replay failed with
  `AdapterParse`. A two-process `GEPA.compile` regression now saves and
  restores ReAct's native tool turns and replays the archived history,
  comparing call/result IDs, arguments, results, and ordered continuation
  payloads against DSPy 3.5.0 / GEPA 0.1.4. It exercises the omitted-data
  case on a copy of the restored turn and does not change checkpoint
  version 26.
  The regression passed in Debug and ReleaseSafe. Both integrated builds
  passed 862 tests and all 40 steps, including examples. The full offline
  reference gate passed, including unchanged fixture regeneration, artifact
  round-trip, all 687 mock requests, and both saved 3.5.0 transcript replays
  in both runtimes.
  Restored ReAct histories now preserve tool-result names through request
  serialization and OpenAI Responses replay. A retained two-process
  `GEPA.compile` regression posts the archived history to a scripted
  localhost `/v1/responses` endpoint. The old build sent the correct IDs,
  arguments and outputs but dropped `lookup` and `submit` from the result
  rows even with `tool_result_name = .include`. The fixed build matched
  pinned DSPy 3.5.0 / GEPA 0.1.4 on the ordered function-call and result rows
  for both `.include` and `.omit`. Checkpoint version 26 is unchanged.
  The deterministic transcript now records those names and compares them
  strictly; the two older live recordings are unchanged and do not assert
  names in their flat request rows.
  Debug and ReleaseSafe each passed 863 tests and all 40 steps. The full
  offline reference gate passed the loopback restart regression, fixture
  regeneration, artifact round-trip, and both runtimes' replays of all 687
  mock requests and both saved 3.5.0 transcripts. No local blocker remains;
  no authenticated provider was called. Next, have the loopback provider
  produce the original calls before checkpointing, with different item IDs
  and call IDs, to test response decoding as well as restored request encoding.
  `proposal_options` configures the built-in reflection proposer: override its
  base instructions, supply additional guidance, select an adapter, or limit
  an instruction by Unicode character count. An oversized draft gets one
  compression call per proposal attempt; empty or still-oversized results
  fail rather than being truncated. `gepa.ConfiguredInstructionProposer`
  also runs standalone on reflection records. Pass already-loaded
  `gepa.ReflectionSkill` records as `proposal_options.skills` to include
  trusted reference material in caller order (up to 64 skills and 256 KiB
  of input text). The proposer does not read files or parse YAML; resolve
  those at application startup. Built-in reflection labels a declared History
  input `Context`. Set `proposal_options.truncate_history_outputs = true` to
  shorten recorded tool-result values after 500 Unicode characters, preserving
  the original History and leaving ordinary JSON fields alone. For standalone
  proposal records, call `prepareHistory` before inserting a typed History.
  Tagged REPL observations in a typed History retain their head and tail
  under the smaller of 500 Unicode characters and the entry's own limit.
  Compaction changes the stored observation but not the entry's replay limit.
  Standalone `REPLHistory` model values and History models erased inside
  arbitrary JSON containers are not compacted.
  `add_format_failure_as_feedback = true` includes captured predictor response
  parse failures in the reflection dataset, with the completion text and the
  active adapter's output structure. Such rollouts keep the configured failure
  score and do not call the application metric. Unattributed program errors are
  not treated as predictor parse failures.
  COPRO `track_stats` exposes per-predictor, per-depth score summaries in
  `Result.stats`. `best` summarizes up to ten retained instruction/prefix
  pairs; `latest` includes repeat evaluations and accumulates the newest
  batches across predictors within the same depth, as in DSPy 3.5.
  Summarizing NaN or infinite evaluation scores with tracking enabled returns
  `error.NonFiniteScoreStatistics` after the batch has been evaluated, matching
  the refusal by the pinned Python statistics code. Tracking disabled does not
  impose this restriction on candidate scores.
  Within one predictor, a repeat evaluation of an instruction/prefix pair keeps
  the earlier evaluation only if its score is greater than or equal to the new
  score. A NaN on either side makes that comparison false.
  `Result.candidate_programs` pairs each score-ordered candidate with its runnable
  program snapshot. Equal-scoring candidates with identical prompt text can
  remain separate, as in DSPy; native custom pointers retain Zig copy semantics.
  Invalid breadth and programs without predictors fail before model calls.
  With zero depth, COPRO generates initial proposals then returns
  `error.NoCandidates`. With an empty training set at positive depth it
  generates initial proposals, then the first evaluation returns `error.EmptyDevset`.
  Tracking does not add evaluations.
  Bootstrap retries refresh the settings LM with temperature `1.0` and the
  round's rollout ID. Predictors with an explicitly selected LM do not inherit
  that refresh. A teacher program returning `error.Canceled` aborts
  BootstrapFewShot immediately without charging the error budget.
  `BestOfN` and `Refine` use the wrapped program's unique LM
  binding for attempts, falling back to the settings LM. Mixed bindings return
  `error.MultipleLMs`; reward callbacks still receive the original context.
  Native programs with no discoverable predictors may run directly with the
  attempt options in `Context`; DSPy 3.5 rejects predictor-free programs here.
  In `BestOfN`, a nonnumeric threshold spends the same failure budget as
  a program or reward failure. The best prediction is selected before the
  threshold comparison, so a tolerated comparison error does not discard it.
  Use a numeric threshold; unlike `Refine`, `BestOfN` does not treat null as
  a disabled threshold.
  `BootstrapFewShotWithRandomSearch.Result.candidates` retains runnable
  programs and their per-validation-example scores in original example order.
  Candidates are sorted by aggregate score, with ties preserving seed order.
  With `stop_at_score`, candidate storage grows as seeds are evaluated instead
  of reserving space for the entire seed range. Early-stopped and restricted
  searches accept an `i128` upper bound even when it exceeds the `i64` range
  of individual candidate seeds. An exhaustive unrestricted search fails
  with `CandidateCapacityExceeded` if its seed range is not representable. An
  early-stopped search that reaches an unrepresentable seed fails likewise.
  The candidate programs and scores use the caller's output allocator and
  do not trigger extra evaluations.
  Serial and concurrent evaluation reject an empty validation set with
  `error.EmptyDevset` instead of invoking the program or scheduling work.
  MIPROv2 `track_stats` retains native program values in trial order with
  stable score-ranked full and minibatch indices in `Result.tracked_programs`.
  Trial logs expose cumulative evaluation counts and distinguish minibatch
  logging mode from a batch that covered the whole validation set. Tracking
  makes no extra evaluations. This is not Python object deepcopy or LM-call
  counter parity; opaque program state can still be borrowed.
  Set `MIPROv2.log_dir = .{ .io = io, .path = path }` to write state-only JSON
  for the baseline, each trial, and periodic full evaluations under
  `evaluated_programs/`. `TrialLog.program_path` identifies the file even
  when `track_stats` is disabled. Use a caller-trusted directory dedicated to
  the run; matching filenames are atomically replaced, and previous snapshots
  remain if a later evaluation or write fails.
  A standalone `Predict` saves its fields at the JSON root, like DSPy.
  Standalone DSZig snapshots from the older format can be converted by moving
  the fields from `self` to the root alongside `metadata`.
  MIPROv2 returns `InvalidOptions` if its trial schedule or evaluation count
  exceeds the native counter range.
  A NaN objective retains its trial number and log but does not enter TPE's
  completed-trial history. A NaN baseline or periodic full reevaluation
  returns `error.NaNStudyObservation`, since those scores are inserted as
  completed observations.
  `view_data_batch_size = 0` summarizes an empty initial view without
  refinement, matching DSPy rather than disabling the data-aware proposer.
  MIPROv2 compile options can override both demo limits without changing the
  optimizer defaults. A zero seed is explicit; only null inherits the default.
  For DSPy-style model defaults, construct `MIPROv2` with
  `try config.init(settings)`. Each omitted `prompt_model` or `task_model`
  independently captures `settings.lm` at construction. Keep the underlying
  LMs alive through compilation. Compile contexts still supply the adapter and
  other execution settings. Without `init`, unbound roles use the compile
  contexts, and an omitted proposal context defaults to the task context.
  An unset `MIPROv2.max_errors` inherits `Context.max_errors` for both
  demo bootstrapping and trial evaluation; an explicit limit overrides it.
  Exhausting the limit aborts demo bootstrapping, but scores the affected
  trial evaluation as zero.
  `MIPROv2.teacher_settings_ref` can borrow settings for the duration of
  compilation. Teacher attempts read the current settings; this does not
  change the task or proposal model.
  In manual mode (`auto = null`), `num_instruct_candidates` and
  `num_fewshot_candidates` independently override the `num_candidates` budget.
  Automatic presets keep their own budgets.
  A zero instruction-candidate count with a nonempty predictor graph returns
  `error.InvalidOptions` rather than starting trial evaluation.
  With zero demo candidates, the few-shot-aware proposer rejects the missing
  demo set. If it is disabled, a manual few-shot search rejects an empty
  categorical demo space before sampling a trial.
  Construct SIMBA with `config.init(construction_context, explicit_prompt_model)`
  to capture the rule-writing LM before compile. A null argument selects the
  construction context's LM; an unconfigured model leaves the choice to
  compile time. The model handle is borrowed and its backing object must
  outlive the optimizer. The compile context still supplies the adapter and
  inference settings, and the feedback request does not enter the task trace.
  Task rollouts and evaluations keep their normal model selection. To migrate
  from `SIMBA.Options.prompt`, pass its LM to the constructor and configure the
  adapter on the compile context rather than passing a separate feedback context.
  SIMBA `Result.candidate_programs` retains the final native program values
  alongside their full-trainset scores, descending and stable across ties.
  It reuses the final evaluations and does not deep-copy opaque program state.
  With `max_steps = 0`, SIMBA evaluates only the baseline, including when
  `num_candidates = 0`. An empty training set requires `bsize = 0` and gives
  that baseline a score of zero. Positive-step searches require nonzero
  `bsize` and `num_candidates`; invalid configurations return `InvalidOptions`.
  A zero `temperature_for_sampling` fails with `InvalidSamplingTemperature`
  before rollouts. A zero `temperature_for_candidates` fails at candidate
  sampling, after the first batch of rollouts. Nonfinite cumulative sampling
  weights fail at the corresponding draw, before consuming RNG state. Neither
  temperature is used for sampling when `max_steps = 0`.
  Set `feedback_metric` instead of `metric` to return a numeric score with
  additional fields for the rule writer. The metadata is masked and rendered
  in the better/worse reward inputs; it does not participate in ranking.
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
  than silently changing index algorithms. Like DSPy, dense retrieval rejects
  retrieval-level caching. Background batching is explicit and bounded in Zig;
  it does not reproduce Python Unbatchify's idle polling or unbounded queue.
  Persistence accepts
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
  format. Bound OpenAI, Anthropic and Gemini predictors support native
  configuration save/load with injected runtime resources. Python class
  reconstruction, other engine codecs, provider training state and complete
  DSPy LM save/load remain unsupported.
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
