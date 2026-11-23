"""Golden GEPA fixture from DSPy 3.5.0 (gepa 0.1.4) on a deterministic mock
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
import tempfile

import dspy
from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal
from gepa.strategies.proposal_sampling import SameParentSampling

import live_suite
import reference as ref

ADVICE = "Answer with an ISO date."
KEEP = "Keep the question."
PROPOSALS = ["Be brief.", "Think about the calendar. " + ADVICE, "Answer politely.", ADVICE + " Never guess."]


class MockLM(ref.ReferenceLM):
    def __init__(self, golds, calls):
        super().__init__(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
        self.golds = golds
        self.calls = calls
        self.reflections = 0

    def complete(self, request):
        messages = ref.messages_for(request)
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        digest = hashlib.sha256(json.dumps(plain, ensure_ascii=False).encode()).hexdigest()
        self.calls.append([ref.rollout_id(self), request.config.temperature, digest])
        system, user = messages[0]["content"], messages[-1]["content"]
        if "examples_with_feedback" in system and "Restate the question." in user:
            text = json.dumps({"new_instruction": "Restate the question. " + KEEP})
        elif "examples_with_feedback" in system:
            text = json.dumps({"new_instruction": "  " + PROPOSALS[self.reflections % len(PROPOSALS)] + "\n"})
            self.reflections += 1
        elif "`restated`" in system:
            q = user.split("<question>\n", 1)[1].split("\n</question>", 1)[0]
            keep = len(q) % 5 != 0 if KEEP in system else len(q) % 2 == 1
            restated = q if keep else q + " Please."
            text = f"<restated>\n{restated}\n</restated>"
        else:
            q = user.split("<question>\n", 1)[1].split("\n</question>", 1)[0]
            good = (ADVICE in system or len(q) % 3 == 0) and q in self.golds
            answer = self.golds[q] if good else "unknown"
            text = ref.xml("r", answer)
        return ref.text_response(self.model, text)


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
    dspy.configure(lm=lm, adapter=None)
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


def check_batch_reflection():
    # Exercise the pinned GEPA engine directly. DSPy's DspyAdapter owns its
    # instruction proposer, so ordinary dspy.GEPA bypasses this extension.
    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            score = {"seed": 0.0, "weaker": 0.5, "winner": 1.0}[candidate["p"]]
            return EvaluationBatch(batch, [score] * len(batch), batch if capture_traces else None)

        def make_reflective_dataset(self, candidate, evaluation, components):
            return {name: [{"row": row} for row in evaluation.trajectories] for name in components}

    class Reflector:
        def __init__(self, mode):
            self.mode = mode
            self.events = []

        def reflect_many(self, jobs):
            self.events.append(("batch", [data["p"][0]["row"] for _, data, _ in jobs]))
            if self.mode == "failed":
                raise RuntimeError("batch failed")
            texts = ("weaker",) if self.mode == "short" else ("weaker", "winner")
            return [(ReflectionProposal({"p": text}), self) for text in texts]

        def reflect(self, candidate, data, components):
            row = data["p"][0]["row"]
            self.events.append(("single", row))
            if row == "q0":
                raise RuntimeError("single job failed")
            return ReflectionProposal({"p": "winner"}), self

    for mode in ("success", "failed", "short"):
        reflector = Reflector(mode)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = optimize(
                {"p": "seed"}, ["q0", "q1"], adapter=Adapter(),
                reflection_strategy=reflector, sampling_strategy=SameParentSampling(2),
                reflection_minibatch_size=1, max_metric_calls=5, use_merge=False,
            )
        expected = [("batch", ["q0", "q1"])]
        if mode != "success":
            expected += [("single", "q0"), ("single", "q1")]
        assert reflector.events == expected, (mode, reflector.events)
        assert result.best_candidate == {"p": "winner"}, (mode, result.best_candidate)
        assert result.total_metric_calls == (10 if mode == "success" else 7), mode


def check_validation_output_artifacts():
    class Signature(dspy.Signature):
        """Seed."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class ArtifactLM(ref.ReferenceLM):
        def complete(self, request):
            system = ref.messages_for(request)[0]["content"]
            answer = "high" if "Winner." in system else "low"
            return ref.text_response(self.model, f"<answer>{answer}</answer>")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return float(pred.answer == "high")

    def propose(*, components_to_update, **kwargs):
        return dict.fromkeys(components_to_update, "Winner.")

    row = dspy.Example(question="q").with_inputs("question")
    for track in (False, True):
        with tempfile.TemporaryDirectory() as path:
            lm = ArtifactLM(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
            with dspy.context(lm=lm, adapter=dspy.XMLAdapter()), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                compiled = dspy.GEPA(
                    metric=metric, max_metric_calls=4, instruction_proposer=propose,
                    reflection_minibatch_size=1, use_merge=False, num_threads=1,
                    log_dir=path, track_stats=True, track_best_outputs=track,
                    gepa_kwargs={"use_cloudpickle": True},
                ).compile(dspy.Predict(Signature), trainset=[row], valset=[row])
            root = pathlib.Path(path) / "generated_best_outputs_valset"
            observed = {f.relative_to(root).as_posix(): json.loads(f.read_text()) for f in root.rglob("*.json")}
            expected = {"task_0/iter_0_prog_0.json": {"answer": "low"}}
            if track:
                expected["task_0/iter_1_prog_1.json"] = {"answer": "high"}
            assert observed == expected, (track, observed)
            assert compiled.detailed_results.total_metric_calls == 4


def check_failed_validation_output_artifacts():
    class Signature(dspy.Signature):
        """Seed."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class ArtifactLM(ref.ReferenceLM):
        def complete(self, request):
            messages = ref.messages_for(request)
            child = "Winner." in messages[0]["content"]
            if child and "broken-case" in messages[-1]["content"]:
                text = "not-an-answer"
            else:
                text = "<answer>high</answer>" if child else "<answer>low</answer>"
            return ref.text_response(self.model, text)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        if gold.question == "seed-crash":
            raise ValueError("deliberate metric failure")
        return float(pred.answer == "high")

    def propose(*, components_to_update, **kwargs):
        return dict.fromkeys(components_to_update, "Winner.")

    rows = [dspy.Example(question=q).with_inputs("question") for q in ("broken-case", "good-case", "seed-crash")]
    for cached, calls in ((False, 10), (True, 8)):
        with tempfile.TemporaryDirectory() as path:
            lm = ArtifactLM(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
            with dspy.context(lm=lm, adapter=dspy.XMLAdapter()), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                compiled = dspy.GEPA(
                    metric=metric, max_metric_calls=7, instruction_proposer=propose,
                    reflection_minibatch_size=2, use_merge=False, num_threads=1,
                    log_dir=path, track_stats=True, track_best_outputs=True, failure_score=0.25,
                    gepa_kwargs={"use_cloudpickle": True, "cache_evaluation": cached},
                ).compile(dspy.Predict(Signature), trainset=rows[:2], valset=rows)
            root = pathlib.Path(path) / "generated_best_outputs_valset"
            observed = {f.relative_to(root).as_posix(): json.loads(f.read_text()) for f in root.rglob("*.json")}
            assert observed == {
                "task_0/iter_0_prog_0.json": {"answer": "low"},
                "task_0/iter_1_prog_1.json": "FailedPrediction(completion_text='not-an-answer', format_reward=0.25)",
                "task_1/iter_0_prog_0.json": {"answer": "low"},
                "task_1/iter_1_prog_1.json": {"answer": "high"},
                "task_2/iter_0_prog_0.json": {},
            }, (cached, observed)
            result = compiled.detailed_results
            assert result.per_val_instance_best_candidates[2] == {0, 1}
            assert result.best_outputs_valset[2] == [(0, dspy.Prediction())]
            assert [(idx, dict(pred)) for idx, pred in result.best_outputs_valset[1]] == [(1, {"answer": "high"})]
            [(idx, failure)] = result.best_outputs_valset[0]
            assert idx == 1
            assert failure.completion_text == "not-an-answer"
            assert failure.format_reward == 0.25
            assert compiled.detailed_results.total_metric_calls == calls


def check_merge_parse_failure():
    class Left(dspy.Signature):
        """Left seed."""

        question: str = dspy.InputField()
        tag: str = dspy.OutputField()

    class Right(dspy.Signature):
        """Right seed."""

        question: str = dspy.InputField()
        tag: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Program(dspy.Module):
        def __init__(self):
            super().__init__()
            self.left = dspy.Predict(Left)
            self.right = dspy.Predict(Right)

        def forward(self, question):
            return self.right(question=question, tag=self.left(question=question).tag)

    class SeedSelector:
        def select_candidate_idx(self, state):
            return 0

    class Sampler:
        def next_minibatch_ids(self, loader, state):
            return [len(state.program_candidates) - 1]

    class MergeLM(ref.ReferenceLM):
        def complete(self, request):
            messages = ref.messages_for(request)
            system, user = messages[0]["content"], messages[-1]["content"]
            q = user.split("<question>", 1)[1].split("</question>", 1)[0].strip()
            if "Left seed." in system or "Left upgraded." in system:
                tag = "upgraded" if "Left upgraded." in system else "base"
                return ref.text_response(self.model, f"<tag>{tag}</tag>")
            left = "<tag>\nupgraded\n</tag>" in user
            right = "Right upgraded." in system
            if left and right:
                self.merge_rows.append(q)
                if len(self.merge_rows) == 5:
                    self.broken = q
                    return ref.text_response(self.model, "bad-subsample")
                if q == getattr(self, "broken", None):
                    return ref.text_response(self.model, "bad-validation")
                return ref.text_response(self.model, "<answer>merged</answer>")
            high = (left and q == "q0") or (right and q == "q1")
            return ref.text_response(self.model, "<answer>high</answer>" if high else "<answer>low</answer>")

    class MergeEvents:
        def on_evaluation_end(self, event):
            if event["candidate_idx"] is None and event["parent_ids"] and len(event["parent_ids"]) == 2:
                self.scores = event["scores"]

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return {"merged": 1.0, "high": 0.1875, "low": 0.125}[pred.answer]

    def propose(*, components_to_update, **kwargs):
        return {name: f"{name.capitalize()} upgraded." for name in components_to_update}

    def row(question):
        return dspy.Example(question=question).with_inputs("question")

    for cached in (False, True):
        with tempfile.TemporaryDirectory() as path:
            lm = MergeLM(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
            lm.merge_rows = []
            events = MergeEvents()
            rows = [row(f"q{i}") for i in range(4)]
            adapter = dspy.XMLAdapter(use_json_adapter_fallback=False)
            with dspy.context(lm=lm, adapter=adapter), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                compiled = dspy.GEPA(
                    metric=metric, max_metric_calls=17, instruction_proposer=propose,
                    candidate_selection_strategy=SeedSelector(),
                    reflection_minibatch_size=None, num_threads=1, log_dir=path,
                    track_stats=True, track_best_outputs=True, failure_score=0.25,
                    gepa_kwargs={
                        "use_cloudpickle": True, "cache_evaluation": cached, "batch_sampler": Sampler(),
                        "merge_val_overlap_floor": 4, "callbacks": [events],
                    },
                ).compile(Program(), trainset=rows, valset=rows)
            result = compiled.detailed_results
            assert result.best_idx == 3 and set(result.parents[3]) == {1, 2}, (result.best_idx, result.parents, result.val_subscores)
            assert len(set(lm.merge_rows[:4])) == 4
            assert len(lm.merge_rows) == (5 if cached else 9), lm.merge_rows
            assert events.scores == [0.25 if q == lm.broken else 1.0 for q in lm.merge_rows[:5]], (lm.merge_rows, events.scores)
            assert result.total_metric_calls == (19 if cached else 25), result.total_metric_calls
            row_id = int(lm.broken[1:])
            assert result.val_subscores[3][row_id] == 0.25
            artifact = pathlib.Path(path) / "generated_best_outputs_valset" / f"task_{row_id}/iter_3_prog_3.json"
            [(idx, output)] = result.best_outputs_valset[row_id]
            assert idx == 3
            expected = {} if cached else "FailedPrediction(completion_text='bad-validation', format_reward=0.25)"
            assert (dict(output) if cached else repr(output)) == expected
            assert json.loads(artifact.read_text()) == expected


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    run(out, "gepa", live_suite.DateQA(), 150)
    run(out, "gepa_merge", TwoStep(), 400)
    check_batch_reflection()
    check_validation_output_artifacts()
    check_failed_validation_output_artifacts()
    check_merge_parse_failure()


if __name__ == "__main__":
    main()
