"""Golden GEPA fixture from DSPy 3.4.1 (gepa 0.1.4) on a deterministic mock
LM: task answers are right when the instruction carries the advice (or the
question length is a multiple of 3), and the reflection LM proposes a
sequence of instructions, so reflection, acceptance, rejection, the Pareto
front and merge sampling all run.

Usage: uv run gen_gepa.py [out_dir]  (default: ../../tests/fixtures)"""

import contextlib
import hashlib
import io
import json
import pathlib
import sys
from types import SimpleNamespace

import dspy

import live_suite

ADVICE = "Answer with an ISO date."
KEEP = "Keep the question."
PROPOSALS = ["Be brief.", "Think about the calendar. " + ADVICE, "Answer politely.", ADVICE + " Never guess."]


class MockLM(dspy.BaseLM):
    def __init__(self, golds, calls):
        super().__init__(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
        self.golds = golds
        self.calls = calls
        self.reflections = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        digest = hashlib.sha256(json.dumps(plain, ensure_ascii=False).encode()).hexdigest()
        self.calls.append([kwargs.get("rollout_id"), kwargs.get("temperature"), digest])
        system, user = messages[0]["content"], messages[-1]["content"]
        if "examples_with_feedback" in system and "Restate the question." in user:
            text = json.dumps({"new_instruction": "Restate the question. " + KEEP})
        elif "examples_with_feedback" in system:
            text = json.dumps({"new_instruction": "  " + PROPOSALS[self.reflections % len(PROPOSALS)] + "\n"})
            self.reflections += 1
        elif "`restated`" in system:
            q = user.split("[[ ## question ## ]]\n", 1)[1].split("\n\n", 1)[0]
            keep = len(q) % 5 != 0 if KEEP in system else len(q) % 2 == 1
            restated = q if keep else q + " Please."
            text = f"[[ ## restated ## ]]\n{restated}\n\n[[ ## completed ## ]]"
        else:
            q = user.split("[[ ## question ## ]]\n", 1)[1].split("\n\n", 1)[0]
            good = (ADVICE in system or len(q) % 3 == 0) and q in self.golds
            answer = self.golds[q] if good else "unknown"
            text = f"[[ ## reasoning ## ]]\nr\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
        message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)], usage={}, model="mock")


class Restate(dspy.Signature):
    """Restate the question."""

    question: str = dspy.InputField()
    restated: str = dspy.OutputField()


class TwoStep(dspy.Module):
    def __init__(self):
        super().__init__()
        self.restate = dspy.Predict(Restate)
        self.generate = dspy.ChainOfThought(live_suite.Dates)

    def forward(self, question):
        return self.generate(question=self.restate(question=question).restated)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float(live_suite.exact_metric(gold, pred))
    if pred_name is None:
        return score
    return dspy.Prediction(score=score, feedback=None if score else f"Expected {gold.answer}.")


def run(out, name, program, budget):
    golds = {q: a for split in ("train", "dev") for q, a in live_suite.DATES[split]}
    calls = []
    lm = MockLM(golds, calls)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    gepa = dspy.GEPA(metric=gepa_metric, max_metric_calls=budget, reflection_lm=lm, num_threads=1, seed=0, track_stats=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        compiled = gepa.compile(program, trainset=live_suite.date_examples("train"), valset=live_suite.date_examples("dev"))
    r = compiled.detailed_results
    path = out / f"{name}_program.json"
    compiled.save(str(path))
    artifact = path.read_text(encoding="utf-8")
    path.unlink()
    data = {
        "calls": calls,
        "candidates": [[p.signature.instructions for _, p in c.named_predictors()] for c in r.candidates],
        "parents": r.parents,
        "val_aggregate_scores": r.val_aggregate_scores,
        "val_subscores": [[s[k] for k in sorted(s)] if isinstance(s, dict) else s for s in r.val_subscores],
        "discovery_eval_counts": r.discovery_eval_counts,
        "best_idx": r.best_idx,
        "total_metric_calls": r.total_metric_calls,
        "num_full_val_evals": r.num_full_val_evals,
        "artifact": artifact,
    }
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    run(out, "gepa", live_suite.DateQA(), 150)
    run(out, "gepa_merge", TwoStep(), 400)


if __name__ == "__main__":
    main()
