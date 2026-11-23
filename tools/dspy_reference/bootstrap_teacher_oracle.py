"""DSPy 3.5.0 teacher-compatibility oracle; no LM or fixture writes.

From the repository root:
    uv run --project tools/dspy_reference --locked tools/dspy_reference/bootstrap_teacher_oracle.py
"""

import contextlib
from decimal import Decimal, InvalidOperation, localcontext
import importlib.metadata
import io
import json
from unittest.mock import patch

import dspy
import optuna
from dspy.teleprompt.simba_utils import append_a_demo
from dspy.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna
from dspy.utils import DummyLM

from fewshot_options_oracle import search_case


class Task(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Student(dspy.Module):
    def __init__(self):
        self.z = dspy.Predict(Task)
        self.a = dspy.Predict(Task)

    def forward(self, question, hint=None):
        selected = [int(p.demos[0]["answer"]) if p.demos else -1 for p in (self.z, self.a)]
        if dspy.settings.trace is not None:
            for predictor in (self.z, self.a):
                dspy.settings.trace.append((predictor, {"question": question}, {"answer": question}))
        return dspy.Prediction(answer=question, z=selected[0], a=selected[1])


class Teacher(dspy.Module):
    def __init__(self):
        self.z = dspy.Predict(Task)
        self.a = dspy.Predict(Task)

    def forward(self, question, hint):
        dspy.settings.trace.extend([
            (self.z, {"question": question}, {"answer": "7"}),
            (self.a, {"question": question}, {"answer": "8"}),
        ])
        return dspy.Prediction(gate=True, first=len(self.z.demos), second=len(self.a.demos), hint=hint)


def metric(example, prediction, trace=None):
    if trace is not None:
        return float(
            prediction.gate and prediction.first == prediction.second == 0
            and prediction.hint == "teacher input" and len(trace) == 2
        )
    return float((prediction.z, prediction.a) == (7, 8))


def check_demo_keywords():
    train = [dspy.Example(question="x").with_inputs("question")]
    for outputs, expected in (
        ({"answer": "ordinary"}, [("augmented", True), ("question", "x"), ("answer", "ordinary")]),
        ({"base": {"tag": "inherited", "question": "old"}}, [
            ("tag", "inherited"), ("question", "x"), ("augmented", True),
        ]),
        ({"augmented": False}, None),
    ):
        calls = []

        class Traced(dspy.Module):
            def __init__(self):
                super().__init__()
                self.p = dspy.Predict(Task)

            def forward(self, question):
                calls.append("teacher")
                dspy.settings.trace.append((self.p, {"question": question}, outputs))
                return dspy.Prediction(answer="ok")

        def accept(*_):
            calls.append("metric")
            return True

        boot = dspy.BootstrapFewShot(
            metric=accept, max_bootstrapped_demos=1, max_labeled_demos=0, max_errors=3,
        )
        system = Traced()
        p = system.p

        def run_simba():
            assert append_a_demo(None)(
                [{"score": 1, "trace": [(p, {"question": "x"}, outputs)]}], system,
                predictor2name={id(p): "p"}, name2predictor={"p": p}, batch_10p_score=0,
            )

        def run_bootstrap():
            return boot.compile(Traced(), trainset=train)

        if expected is None:
            for run in (run_bootstrap, run_simba):
                try:
                    run()
                except TypeError as exc:
                    assert "augmented" in str(exc)
                else:
                    raise AssertionError("duplicate demo keyword accepted")
            assert not p.demos
        else:
            assert list(run_bootstrap().p.demos[0].items()) == expected
            run_simba()
            assert list(p.demos[0].items()) == expected
        assert calls == ["teacher", "metric"]
        assert boot.error_count == 0


def check_demo_comparison_errors():
    bad = dspy.Example(question="bad", metadata=Decimal("sNaN")).with_inputs("question")
    distinct = dspy.Example(question="bad", metadata=Decimal("sNaN")).with_inputs("question")
    good = dspy.Example(question="good").with_inputs("question")
    calls = []

    class Traced(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = dspy.Predict(Task)
            self.second = dspy.Predict(Task)

        def forward(self, question):
            calls.append((question, len(self.first.demos), len(self.second.demos)))
            dspy.settings.trace.append((self.second, {"question": question}, {"answer": "ok"}))
            return dspy.Prediction(answer="ok")

    teacher = Traced()
    teacher.first.demos = [bad]
    teacher.second.demos = [bad, distinct]
    with localcontext() as ctx:
        ctx.traps[InvalidOperation] = True
        optimizer = dspy.BootstrapFewShot(max_labeled_demos=0, max_errors=2)
        result = optimizer.compile(Traced(), teacher=teacher, trainset=[bad, good])
        assert calls == [("good", 0, 2)]
        assert optimizer.error_count == 1
        assert optimizer.validation[0] is bad and len(optimizer.validation) == 1
        assert [demo.question for demo in result.second.demos] == ["good"]
        assert len(teacher.first.demos) == 1 and len(teacher.second.demos) == 2

        calls.clear()
        optimizer = dspy.BootstrapFewShot(max_labeled_demos=0, max_errors=1)
        try:
            optimizer.compile(Traced(), teacher=teacher, trainset=[bad, good])
        except InvalidOperation:
            assert calls == [] and optimizer.error_count == 1
        else:
            raise AssertionError("bootstrap exceeded the demo comparison error budget")


def check_trace_controls():
    class Pair(dspy.Module):
        def __init__(self):
            super().__init__()
            self.left = dspy.Predict(Task)
            self.right = dspy.Predict(Task)

        def forward(self, question):
            self.left(question="good")
            return self.right(question="bad")

    row = dspy.Example(question="q").with_inputs("question")
    for caller, teacher, expected in (
        (2, None, ["good", "bad"]),
        (2, 1, ["bad"]),
        (2, 0, []),
        (2, -1, []),
        (0, 2, ["good", "bad"]),
    ):
        traces = []
        answers = ["left", "right", "held-out left", "held-out right"]
        lm = DummyLM([{"answer": answer} for answer in answers])

        def accept(_, prediction, trace):
            assert prediction.answer == "right"
            traces.append([inputs["question"] for _, inputs, _ in trace])
            return True

        caller_trace = []
        with dspy.context(lm=lm, trace=caller_trace, max_trace_size=caller):
            optimizer = dspy.BootstrapFewShot(
                metric=accept, max_bootstrapped_demos=1, max_labeled_demos=0, max_errors=1,
                teacher_settings={} if teacher is None else {"max_trace_size": teacher},
            )
            program = optimizer.compile(Pair(), trainset=[row])
            assert optimizer.validation == []
            assert traces == [expected] and caller_trace == [] and len(lm.history) == 2
            assert [demo.answer for demo in program.left.demos] == (["left"] if "good" in expected else [])
            assert [demo.answer for demo in program.right.demos] == (["right"] if "bad" in expected else [])
            assert program(question="held out").answer == "held-out right"
            assert len(caller_trace) == caller

    trace = []
    lm = DummyLM([{"answer": answer} for answer in "ABCD"])
    with dspy.context(lm=lm, trace=trace):
        predictor = dspy.Predict(Task)
        for kwargs, answer, retained in zip(
            [{}, {"_trace": False}, {"_trace": ""}, {"_trace": 1}], "ABCD", [1, 1, 1, 2], strict=True,
        ):
            assert predictor(question="q", **kwargs).answer == answer and len(trace) == retained
    assert len(lm.history) == 4
    assert trace[-1][1] == {"question": "q"} and trace[-1][2].answer == "D"


def main():
    assert importlib.metadata.version("dspy") == "3.5.0"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        check_demo_comparison_errors()
    print("ok: DSPy bootstrap counts demo comparison errors and preserves earlier demo filters")
    with contextlib.redirect_stdout(io.StringIO()):
        check_trace_controls()
    print("ok: DSPy scopes teacher trace limits and consumes predictor trace controls")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for restrict, seed, score in ((None, -3, 320), ([-2], -2, 360)):
            result = search_case("wide", restrict, candidates=2**100, labeled_sample=False, stop_at_score=0)
            assert result["error"] is None, result
            assert result["candidates"] == [{"seed": seed, "score": score}], result
            assert result["forward_calls"] == result["metric_calls"] == 5, result
    print("ok: DSPy random search stops at an early seed with a 2**100 upper bound")
    assert optuna.__version__ == "4.5.0"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train = [dspy.Example(question="0", hint="teacher input", answer="0").with_inputs("question", "hint")]
    student, teacher = Student(), Teacher()
    studies = []
    create_study = optuna.create_study

    def observe_study(*args, **kwargs):
        study = create_study(*args, **kwargs)
        studies.append(study)
        return study

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bootstrap = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=1, max_labeled_demos=1)
        check_demo_keywords()
        compiled = bootstrap.compile(student, teacher=teacher, trainset=train)
        assert compiled._compiled and not teacher.z.demos and not teacher.a.demos
        demos = [dict(p.demos[0]) for p in (compiled.z, compiled.a)]
        assert demos == [dict(augmented=True, question="0", answer=x) for x in ("7", "8")]
        mismatch = Teacher()
        mismatch.a.signature = mismatch.a.signature.with_instructions("Different instruction.")
        for dataset in ([], train):
            try:
                bootstrap.compile(student, teacher=mismatch, trainset=dataset)
            except AssertionError as exc:
                assert "same signatures" in str(exc)
            else:
                raise AssertionError("signature mismatch accepted")
        fallback = dspy.BootstrapFewShot(metric=lambda *_: 1, max_bootstrapped_demos=1).compile(student, trainset=train)
        assert fallback.z.demos[0].answer == "0"
        randomized = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric, max_bootstrapped_demos=1, max_labeled_demos=1, num_candidate_programs=1, num_threads=1,
        ).compile(student, teacher=teacher, trainset=train)
        ranked = [(row["seed"], row["score"], row["subscores"]) for row in randomized.candidate_programs]
        assert ranked == [(-1, 100, [1]), (0, 100, [1]), (-3, 0, [0]), (-2, 0, [0])]
        assert randomized is randomized.candidate_programs[0]["program"]
        with patch.object(optuna, "create_study", observe_study):
            tuned = BootstrapFewShotWithOptuna(metric=metric, max_labeled_demos=1, num_candidate_programs=2).compile(
                student, teacher=teacher, max_demos=1, trainset=train,
            )
        study, = studies
        trials = [(trial.value, trial.params) for trial in study.trials]
        assert trials == [(100, dict(demo_index_for_z=0, demo_index_for_a=0))] * 2
        assert study.best_trial.number == 0
        assert [p.demos[0]["answer"] for p in (tuned.z, tuned.a)] == ["7", "8"]
    print(json.dumps({"demos": demos, "random_search": ranked, "optuna_trials": trials, "best_trial": 0}))


if __name__ == "__main__":
    main()
