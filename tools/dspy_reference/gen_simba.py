"""Golden SIMBA fixture from DSPy 3.4.1 on a deterministic mock LM whose
answers depend on the rollout id, the demos and the instructions, so every
SIMBA path runs (demos, rules, drops, buckets, final selection). Needs NumPy,
so run it inside the devenv shell.

Usage: uv run gen_simba.py [out_dir]  (default: ../../tests/fixtures)"""

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
FEEDBACK = (
    "[[ ## discussion ## ]]\nThe worse run guessed.\n\n"
    '[[ ## module_advice ## ]]\n{"generate.predict": "' + ADVICE + '"}\n\n[[ ## completed ## ]]'
)


class MockLM(dspy.BaseLM):
    def __init__(self, golds, calls):
        super().__init__(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
        self.golds = golds
        self.calls = calls

    def forward(self, prompt=None, messages=None, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        digest = hashlib.sha256(json.dumps(plain, ensure_ascii=False).encode()).hexdigest()
        rollout = kwargs.get("rollout_id")
        self.calls.append([rollout, kwargs.get("temperature"), digest])
        system, user = messages[0]["content"], messages[-1]["content"]
        if "module_advice" in system:
            text = FEEDBACK
        else:
            q = user.split("[[ ## question ## ]]\n", 1)[1].split("\n\n", 1)[0]
            good = ADVICE in system or (len(messages) > 2 and len(q) % 3 == 0)
            good = good or (rollout is not None and (len(q) + rollout) % 2 == 0)
            answer = self.golds[q] if good else "unknown"
            text = f"[[ ## reasoning ## ]]\nr\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
        message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)], usage={}, model="mock")

    def copy(self, **kwargs):
        new = MockLM(self.golds, self.calls)
        new.kwargs = {**self.kwargs, **kwargs}
        return new


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    golds = {q: a for split in ("train", "dev") for q, a in live_suite.DATES[split]}
    calls = []
    dspy.configure(lm=MockLM(golds, calls), adapter=dspy.ChatAdapter())
    simba = dspy.SIMBA(metric=live_suite.float_metric, bsize=4, num_candidates=3, max_steps=3, max_demos=2, num_threads=1)
    with contextlib.redirect_stdout(io.StringIO()):
        compiled = simba.compile(live_suite.DateQA(), trainset=live_suite.date_examples("train"), seed=0)
    path = out / "simba_program.json"
    compiled.save(str(path))
    artifact = path.read_text(encoding="utf-8")
    path.unlink()
    data = {
        "calls": calls,
        "scores": [c["score"] for c in compiled.candidate_programs],
        "train_scores": [compiled.trial_logs[k].get("train_score") for k in sorted(compiled.trial_logs)],
        "artifact": artifact,
    }
    (out / "simba.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
