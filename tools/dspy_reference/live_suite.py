"""The live-validation suite, shared by live.py (record real-model calls) and
check_live.py (replay a recorded transcript). Mirrors examples/live_suite.zig.

A transcript is {"runtime", "model", "calls": [{"messages", "completion",
...}], "cases": [...], "rag" and "dates": {"baseline", "optimized", "artifact"}}."""

import json
import pathlib
import tempfile

import dspy
from dspy._vendor.lm15.serde import request_to_dict, response_from_dict, response_to_dict
from dspy.lm15 import CitationPart, ContinuationState, Message, TextPart, ThinkingPart, ToolCallPart, ToolResultPart

import reference as ref

TEMPERATURE = 0.0
DATES = json.loads((pathlib.Path(__file__).resolve().parents[2] / "examples" / "dates.json").read_text(encoding="utf-8"))


class Dates(dspy.Signature):
    """Answer the question with the date it asks for."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class DateQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(Dates)

    def forward(self, question):
        return self.generate(question=question)


def exact_metric(example, pred, trace=None):
    return pred.answer.strip() == example.answer


def date_examples(split):
    return [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in DATES[split]]


def cases():
    qa_demos = [
        dspy.Example(augmented=True, context=["a fact"], question="q1?", reasoning=dspy.Reasoning(content="r1"), answer="a1"),
        dspy.Example(question="q2?", answer="a2"),
    ]
    qa_inputs = {"context": ["Paris is the capital of France.", "multi\nline «x»"], "question": "What is the capital of France?"}
    typed_demos = [
        dspy.Example(text="good", tags=["x"], flag=True, count=3, ratio=0.5, label="pos", items=["a", "b"]),
        dspy.Example(text="bad", flag=False, label="neg"),
    ]
    typed_inputs = {"text": "great movie", "tags": ["review", "film"]}
    return [
        ("cot_qa_chat", lambda: dspy.ChainOfThought(ref.QA), dspy.ChatAdapter(), qa_demos, qa_inputs),
        ("cot_qa_json", lambda: dspy.ChainOfThought(ref.QA), dspy.JSONAdapter(), qa_demos, qa_inputs),
        ("typed_chat", lambda: dspy.Predict(ref.Typed), dspy.ChatAdapter(), typed_demos, typed_inputs),
        ("typed_json", lambda: dspy.Predict(ref.Typed), dspy.JSONAdapter(), typed_demos, typed_inputs),
        ("indented_chat", lambda: dspy.Predict(ref.Indented), dspy.ChatAdapter(), [], {"text": "meh"}),
        ("cot_qa_xml", lambda: dspy.ChainOfThought(ref.QA), dspy.XMLAdapter(), qa_demos, qa_inputs),
        ("typed_xml", lambda: dspy.Predict(ref.Typed), dspy.XMLAdapter(), typed_demos, typed_inputs),
        ("indented_xml", lambda: dspy.Predict(ref.Indented), dspy.XMLAdapter(), [], {"text": "meh"}),
    ]


def run_suite(lm):
    """Run every case and every optimizer pipeline on `lm`."""
    out = {"cases": []}
    for name, make, adapter, demos, inputs in cases():
        dspy.configure(lm=lm, adapter=adapter)
        module = make()
        (module.predict if hasattr(module, "predict") else module).demos = demos
        try:
            pred = module(**inputs)
            out["cases"].append({"name": name, "outputs": ref.serialize_object(dict(pred.items()))})
        except Exception as e:  # noqa: BLE001 - parse failures are a result, compared across runtimes.
            out["cases"].append({"name": name, "error": type(e).__name__})
    dspy.configure(lm=lm, rm=ref.in_memory_rm(ref.CORPUS), adapter=None)
    rag_optimizer = dspy.BootstrapFewShot(metric=ref.keyword_metric, max_bootstrapped_demos=2, max_labeled_demos=3)
    out["rag"] = pipeline(ref.RAG, rag_optimizer, ref.trainset(), ref.trainset(), ref.keyword_metric)
    dates_optimizer = dspy.BootstrapFewShot(metric=exact_metric, max_bootstrapped_demos=4, max_labeled_demos=8)
    out["dates"] = pipeline(DateQA, dates_optimizer, date_examples("train"), date_examples("dev"), exact_metric)
    search = dspy.BootstrapFewShotWithRandomSearch(
        metric=exact_metric, max_bootstrapped_demos=2, max_labeled_demos=2, num_candidate_programs=4, num_threads=1
    )
    out["dates_search"] = pipeline(DateQA, search, date_examples("train"), date_examples("dev"), exact_metric)
    mipro = RecordedMIPROv2(
        metric=exact_metric, auto=None, num_candidates=4, max_bootstrapped_demos=2, max_labeled_demos=2, num_threads=1
    )
    out["dates_mipro"] = pipeline(
        DateQA, mipro, date_examples("train"), date_examples("dev"), exact_metric,
        valset=date_examples("dev"), num_trials=MIPRO_TRIALS, minibatch_size=MIPRO_MINIBATCH, program_aware_proposer=False,
    )
    aware = RecordedMIPROv2(
        metric=exact_metric, auto=None, num_candidates=2, max_bootstrapped_demos=2, max_labeled_demos=2, num_threads=1
    )
    out["dates_mipro_aware"] = pipeline(
        DateQA, aware, date_examples("train"), date_examples("dev"), exact_metric,
        valset=date_examples("dev"), num_trials=MIPRO_AWARE_TRIALS, minibatch_size=MIPRO_MINIBATCH,
    )
    out["rag_react"] = pipeline(
        rag_agent, dspy.BootstrapFewShot(metric=ref.keyword_metric, max_bootstrapped_demos=2, max_labeled_demos=0),
        ref.trainset(), ref.trainset(), ref.keyword_metric,
    )
    copro = dspy.COPRO(metric=exact_metric, breadth=COPRO_BREADTH, depth=COPRO_DEPTH)
    out["dates_copro"] = pipeline(
        DateQA, copro, date_examples("train"), date_examples("dev"), exact_metric,
        eval_kwargs={"num_threads": 1, "display_progress": False},
    )
    simba = dspy.SIMBA(
        metric=float_metric, bsize=SIMBA_BSIZE, num_candidates=SIMBA_CANDIDATES, max_steps=SIMBA_STEPS,
        max_demos=SIMBA_DEMOS, num_threads=1,
    )
    out["dates_simba"] = pipeline(DateQA, simba, date_examples("train"), date_examples("dev"), float_metric)
    gepa = dspy.GEPA(
        metric=gepa_metric, max_metric_calls=GEPA_BUDGET, reflection_lm=dspy.settings.lm, num_threads=1, seed=0,
        track_stats=True,
    )
    out["dates_gepa"] = pipeline(
        DateQA, gepa, date_examples("train"), date_examples("dev"), float_metric, valset=date_examples("dev"),
    )
    return out
MIPRO_TRIALS = 12
MIPRO_MINIBATCH = 4
MIPRO_AWARE_TRIALS = 4
COPRO_BREADTH = 3
COPRO_DEPTH = 2
SIMBA_BSIZE = 4
SIMBA_CANDIDATES = 2
SIMBA_STEPS = 2
SIMBA_DEMOS = 2
GEPA_BUDGET = 60


def float_metric(example, pred, trace=None):
    return float(exact_metric(example, pred))


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float_metric(gold, pred)
    if pred_name is None or score:
        return score
    return dspy.Prediction(score=score, feedback=f"Expected {gold.answer}.")


def corpus_search(query: str) -> list[str]:
    """Search the corpus for passages relevant to the query."""
    return [p.long_text for p in ref.in_memory_rm(ref.CORPUS)(query, k=2)]


def rag_agent():
    return dspy.ReAct("question -> answer", tools=[corpus_search], max_iters=REACT_MAX_ITERS)


REACT_MAX_ITERS = 4
class RecordedMIPROv2(dspy.MIPROv2):
    """Keeps the proposed instruction candidates for comparison."""

    def _propose_instructions(self, *args, **kwargs):
        self.instruction_candidates = super()._propose_instructions(*args, **kwargs)
        return self.instruction_candidates


def trial_rows(logs, n_predictors):
    """`trial_logs` as [trial_num, chosen indices or None, score, full_eval]."""
    rows = []
    for num in sorted(logs):
        log = logs[num]
        keys = [k for i in range(n_predictors) for k in (f"{i}_predictor_instruction", f"{i}_predictor_demos") if k in log]
        params = [log[k] for k in keys] or None
        full = "mb_score" not in log
        rows.append([num, params, log["full_eval_score"] if full else log["mb_score"], full])
    return rows


def pipeline(make, optimizer, trainset, devset, metric, **compile_kwargs):
    """Evaluate on devset, compile on trainset, evaluate again."""
    evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=False, num_threads=1)
    result = {"baseline": evaluate(make()).score}
    compiled = optimizer.compile(make(), trainset=trainset, **compile_kwargs)
    result["optimized"] = evaluate(compiled).score
    programs = getattr(compiled, "candidate_programs", [])
    result["candidates"] = [[c["seed"], c["score"]] for c in programs if "seed" in c]
    result["scores"] = [c["score"] for c in programs if "seed" not in c]
    proposed = getattr(optimizer, "instruction_candidates", {})
    result["instructions"] = [list(proposed[i]) for i in sorted(proposed)]
    n_predictors = len(compiled.predictors())
    result["trials"] = trial_rows(compiled.trial_logs, n_predictors) if isinstance(optimizer, dspy.MIPROv2) else []
    logs = compiled.trial_logs if isinstance(optimizer, dspy.SIMBA) else {}
    result["train_scores"] = [logs[k].get("train_score") for k in sorted(logs)]
    if isinstance(optimizer, dspy.GEPA):
        r = compiled.detailed_results
        result["instructions"] = [[p.signature.instructions for _, p in c.named_predictors()] for c in r.candidates]
        result["scores"] = list(r.val_aggregate_scores)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "program.json"
        compiled.save(str(path))
        result["artifact"] = path.read_text(encoding="utf-8")
    return result



def normalized_continuation(states):
    return [{"provider": state.provider, "kind": state.kind, "data": state.data} for state in states]


def restore_continuation(states):
    return tuple(ContinuationState(provider=state["provider"], kind=state["kind"], data=state["data"])
                 for state in states)


def normalized_tool_calls(parts):
    return [{"id": part.id, "name": part.name,
             "arguments": json.dumps(part.input, ensure_ascii=False, separators=(",", ":")),
             "continuation": normalized_continuation(part.continuation)}
            for part in parts]


def message_continuation(message):
    states = list(message.continuation)
    for part in message.parts:
        if not isinstance(part, ToolCallPart):
            states.extend(part.continuation)
    return normalized_continuation(states)


def text_parts(parts):
    if any(not isinstance(part, TextPart) for part in parts):
        raise TypeError("the shared transcript text projection cannot represent non-text media")
    return "".join(part.text for part in parts)


def normalized_messages(request):
    messages = []
    if request.system is not None:
        system = request.system if isinstance(request.system, str) else text_parts(request.system)
        messages.append({"role": "system", "content": system, "tool_calls": [], "tool_call_id": None,
                         "reasoning": None, "continuation": []})
    for message in request.messages:
        if any(not isinstance(part, (TextPart, ThinkingPart, ToolCallPart, ToolResultPart)) for part in message.parts):
            raise TypeError("the shared transcript cannot represent this lm15 part; canonical data is required")
        results = message.parts_of(ToolResultPart)
        if results:
            if len(results) != len(message.parts):
                raise TypeError("the shared transcript cannot flatten mixed tool-result content")
            for result in results:
                messages.append({"role": message.role, "content": text_parts(result.content), "tool_calls": [],
                                 "tool_call_id": result.id, "reasoning": None,
                                 "continuation": normalized_continuation((*message.continuation, *result.continuation))})
        else:
            messages.append({"role": message.role, "content": "".join(part.text for part in message.parts_of(TextPart)),
                             "tool_calls": normalized_tool_calls(message.parts_of(ToolCallPart)),
                             "tool_call_id": None,
                             "reasoning": "".join(part.text for part in message.parts_of(ThinkingPart)) or None,
                             "continuation": message_continuation(message)})
    return messages


def request_metadata(request, rollout):
    """Shared Zig/Python text/native metadata; canonical request stays separate."""
    messages = normalized_messages(request)
    format_ = request.config.response_format
    if format_ is not None and format_.get("type") == "json_schema":
        format_ = {"type": "json_schema", **format_["json_schema"]}
    choice = request.config.tool_choice
    tools = []
    for tool in request.tools:
        if tool.type != "function":
            raise TypeError(f"shared transcript cannot represent provider builtin tool {tool.name!r}")
        tools.append({"name": tool.name, "description": tool.description or "", "parameters": tool.parameters, "strict": None})
    return {
        "messages": messages,
        "tools": tools,
        "options": {
            "temperature": request.config.temperature,
            "rollout_id": json.loads(json.dumps(rollout, ensure_ascii=False)),
            "n": 1,
            "reasoning_effort": request.config.reasoning.effort if request.config.reasoning is not None else None,
            # Predictor-only explicit-null intent is consumed before this
            # normalized engine Request; it is not a wire generation option.
            "reasoning_effort_present": False,
            "response_format": format_,
            "tool_choice": {"mode": choice.mode, "allowed": list(choice.allowed)} if choice is not None else None,
            "parallel_tool_calls": choice.parallel if choice is not None else None,
        },
    }


def transcript_call(request, response, rollout):
    """Keep canonical native protocol data as well as the shared Zig projection."""
    reasoning = "".join(part.text for part in response.message.parts_of(ThinkingPart)) or None
    calls = normalized_tool_calls(response.message.parts_of(ToolCallPart))
    citations = [
        {"type": "citation", "text": part.text, "title": part.title, "url": part.url,
         "continuation": normalized_continuation(part.continuation)}
        for part in response.message.parts_of(CitationPart)
    ]
    continuation = message_continuation(response.message)
    return {
        **request_metadata(request, rollout),
        "completion": response.text or "",
        **({"reasonings": [reasoning]} if reasoning is not None else {}),
        "tool_calls": [calls] if calls else [],
        **({"citations": [citations]} if citations else {}),
        "continuation": [continuation] if continuation else [],
        "provider_data": [response.provider_data] if response.provider_data is not None else [],
        "request": request_to_dict(request),
        "response": response_to_dict(response, include_provider_data=True),
        "response_model": response.model,
    }


class RecordingLM(dspy.LM):
    """A real dspy.LM that also appends every call to `self.calls`. `copy()`
    is shallow, so rollout copies share the list."""

    def __init__(self, model, **kwargs):
        self._execution_controls = ref.ExecutionControls()
        callbacks = [*kwargs.pop("callbacks", []), self._execution_controls]
        super().__init__(model, temperature=TEMPERATURE, cache=False, num_retries=8, callbacks=callbacks, **kwargs)
        self.calls = []

    def update_history(self, entry):
        super().update_history(entry)
        self.calls.append(transcript_call(entry["request"], entry["response"], ref.rollout_id(self)))


class ReplayLM(ref.ReferenceLM):
    """Answers each request with the first unused recorded call whose
    messages are identical; any other request is a mismatch."""

    def __init__(self, calls, supports_reasoning=False, capabilities=None):
        super().__init__(model="dszig/replay", model_type="chat", temperature=TEMPERATURE, max_tokens=None, cache=False)
        self._reasoning_supported = supports_reasoning
        self._capabilities = capabilities or {}
        self.calls = calls
        self.used = [False] * len(calls)
        self.misses = []

    @property
    def supports_reasoning(self):
        return self._reasoning_supported

    @property
    def supports_function_calling(self):
        return self._capabilities.get("supports_function_calling", False)

    @property
    def supports_response_schema(self):
        return self._capabilities.get("supports_response_schema", False)

    @property
    def supported_params(self):
        params = set(self._capabilities.get("supported_params", []))
        if self._capabilities.get("supports_response_format", False):
            params.add("response_format")
        return params

    def complete(self, request):
        metadata = request_metadata(request, ref.rollout_id(self))
        want = metadata["messages"]
        for i, call in enumerate(self.calls):
            if call["messages"] != want or call.get("tools", []) != metadata["tools"]:
                continue
            if call["options"] != metadata["options"]:
                continue
            if "request" in call:
                recorded = {**call["request"], "model": request.model}
                if recorded != request_to_dict(request):
                    continue
            # One canonical engine call is one candidate, with its exact
            # adapter-assigned rollout identity and n=1.
            if self.used[i]:
                continue
            text = call["completion"]
            self.used[i] = True
            if "response" in call:
                return response_from_dict(call["response"])
            native_calls = call.get("tool_calls", [])
            citations = call.get("citations", [])
            continuations = call.get("continuation", [])
            reasoning = (call.get("reasonings") or [None])[0]
            provider_data = call.get("provider_data", [])
            raw = provider_data[0] if provider_data else None
            if native_calls or citations or continuations or reasoning is not None or raw is not None:
                from dataclasses import replace
                parts = []
                states = restore_continuation(continuations[0]) if continuations else ()
                if reasoning is not None or states:
                    parts.append(ThinkingPart(reasoning or "", continuation=states))
                if text:
                    parts.append(TextPart(text))
                if native_calls:
                    parts.extend(ToolCallPart(id=c["id"], name=c["name"], input=json.loads(c["arguments"]),
                                              continuation=restore_continuation(c.get("continuation", [])))
                                 for c in native_calls[0])
                if citations:
                    parts.extend(CitationPart(text=c.get("text"), title=c.get("title"), url=c.get("url"),
                                              continuation=restore_continuation(c.get("continuation", [])))
                                 for c in citations[0])
                if not parts:
                    parts.append(TextPart(text))
                return replace(ref.text_response(self.model, text), message=Message.assistant(parts), provider_data=raw)
            return ref.text_response(self.model, text)
        self.misses.append(want)
        raise RuntimeError("replay: request has no recorded twin")
