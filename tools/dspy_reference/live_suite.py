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
            **({"completions": [completion_text(o) for o in entry["outputs"]]} if len(entry["outputs"]) > 1 else {}),
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
                texts = call.get("completions", [call["completion"]])
                choices = [
                    SimpleNamespace(
                        message=SimpleNamespace(content=text, tool_calls=None, reasoning_content=None),
                        finish_reason="stop",
                        index=i,
                    )
                    for i, text in enumerate(texts)
                ]
                return SimpleNamespace(choices=choices, usage={}, model=self.model)
        self.misses.append(want)
        raise RuntimeError("replay: request has no recorded twin")

    def copy(self, **kwargs):
        return self
