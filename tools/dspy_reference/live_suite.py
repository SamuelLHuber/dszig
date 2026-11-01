"""The live-validation suite, shared by live.py (record real-model calls) and
check_live.py (replay a recorded transcript). Mirrors examples/live_suite.zig.

A transcript is {"runtime", "model", "calls": [{"messages", "completion",
...}], "cases": [...], "rag" and "dates": {"baseline", "optimized", "artifact"}}."""

import json
import pathlib
import tempfile
from types import SimpleNamespace

import dspy

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
        dspy.Example(augmented=True, context=["a fact"], question="q1?", reasoning="r1", answer="a1"),
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
            out["cases"].append({"name": name, "outputs": dict(pred.items())})
        except Exception as e:  # noqa: BLE001 - parse failures are a result, compared across runtimes.
            out["cases"].append({"name": name, "error": type(e).__name__})
    dspy.configure(lm=lm, rm=ref.in_memory_rm(ref.CORPUS), adapter=dspy.ChatAdapter())
    rag_optimizer = dspy.BootstrapFewShot(metric=ref.keyword_metric, max_bootstrapped_demos=2, max_labeled_demos=3)
    out["rag"] = pipeline(ref.RAG, rag_optimizer, ref.trainset(), ref.trainset(), ref.keyword_metric)
    dates_optimizer = dspy.BootstrapFewShot(metric=exact_metric, max_bootstrapped_demos=4, max_labeled_demos=8)
    out["dates"] = pipeline(DateQA, dates_optimizer, date_examples("train"), date_examples("dev"), exact_metric)
    search = dspy.BootstrapFewShotWithRandomSearch(
        metric=exact_metric, max_bootstrapped_demos=2, max_labeled_demos=2, num_candidate_programs=4
    )
    out["dates_search"] = pipeline(DateQA, search, date_examples("train"), date_examples("dev"), exact_metric)
    return out


def pipeline(make, optimizer, trainset, devset, metric):
    """Evaluate on devset, compile on trainset, evaluate again."""
    evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=False, num_threads=1)
    result = {"baseline": evaluate(make()).score}
    compiled = optimizer.compile(make(), trainset=trainset)
    result["optimized"] = evaluate(compiled).score
    result["candidates"] = [[c["seed"], c["score"]] for c in getattr(compiled, "candidate_programs", [])]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "program.json"
        compiled.save(str(path))
        result["artifact"] = path.read_text(encoding="utf-8")
    return result


def completion_text(output):
    return output if isinstance(output, str) else output["text"]


class RecordingLM(dspy.LM):
    """A real dspy.LM that also appends every call to `self.calls`. `copy()`
    is shallow, so rollout copies share the list."""

    def __init__(self, model, **kwargs):
        super().__init__(model, temperature=TEMPERATURE, cache=False, num_retries=8, **kwargs)
        self.calls = []

    def update_history(self, entry):
        super().update_history(entry)
        kwargs = {**self.kwargs, **entry["kwargs"]}
        self.calls.append({
            "messages": [dict(m) for m in entry["messages"]],
            "completion": completion_text(entry["outputs"][0]),
            "temperature": kwargs.get("temperature"),
            "rollout_id": kwargs.get("rollout_id"),
            "response_format": json.dumps(kwargs["response_format"], default=str) if "response_format" in kwargs else None,
            "response_model": entry.get("response_model"),
        })


class ReplayLM(dspy.BaseLM):
    """Answers each request with the first unused recorded call whose
    messages are identical; any other request is a mismatch."""

    def __init__(self, calls):
        super().__init__(model="dszig/replay", model_type="chat", temperature=TEMPERATURE, max_tokens=None, cache=False)
        self.calls = calls
        self.used = [False] * len(calls)
        self.misses = []

    def forward(self, prompt=None, messages=None, **kwargs):
        want = [{"role": m["role"], "content": m["content"]} for m in messages]
        for i, call in enumerate(self.calls):
            if not self.used[i] and call["messages"] == want:
                self.used[i] = True
                message = SimpleNamespace(content=call["completion"], tool_calls=None, reasoning_content=None)
                choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
                return SimpleNamespace(choices=[choice], usage={}, model=self.model)
        self.misses.append(want)
        raise RuntimeError("replay: request has no recorded twin")

    def copy(self, **kwargs):
        return self
