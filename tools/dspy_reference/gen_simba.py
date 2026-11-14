"""Golden SIMBA fixture from DSPy 3.5.0 on a deterministic engine whose
answers depend on the rollout id, the demos and the instructions, so every
SIMBA path runs (demos, rules, drops, buckets, final selection). NumPy is
included in the locked reference environment.

Usage: uv run gen_simba.py [out_dir]  (default: ../../tests/fixtures)"""

import contextlib
import hashlib
import io
import json
import pathlib
import sys

import dspy

import live_suite
import reference as ref

ADVICE = "Answer with an ISO date."
FEEDBACK = (
    "<discussion>\nThe worse run guessed.\n</discussion>\n\n"
    "<module_advice>\n<generate.predict>\n" + ADVICE + "\n</generate.predict>\n</module_advice>"
)


class MockLM(ref.ReferenceLM):
    def __init__(self, golds, calls):
        super().__init__(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
        self.golds = golds
        self.calls = calls

    def complete(self, request):
        messages = ref.messages_for(request)
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        digest = hashlib.sha256(json.dumps(plain, ensure_ascii=False).encode()).hexdigest()
        rollout = ref.rollout_id(self)
        self.calls.append([rollout, request.config.temperature, digest])
        system, user = messages[0]["content"], messages[-1]["content"]
        if "module_advice" in system:
            text = FEEDBACK
        else:
            q = user.split("<question>\n", 1)[1].split("\n</question>", 1)[0]
            good = ADVICE in system or (len(messages) > 2 and len(q) % 3 == 0)
            good = good or (rollout is not None and (len(q) + rollout) % 2 == 0)
            answer = self.golds[q] if good else "unknown"
            text = ref.xml("r", answer)
        return ref.text_response(self.model, text)


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    golds = {q: a for split in ("train", "dev") for q, a in live_suite.DATES[split]}
    calls = []
    dspy.configure(lm=MockLM(golds, calls), adapter=None)
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
