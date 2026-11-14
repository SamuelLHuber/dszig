"""Independent locked-DSPy 3.5 / Optuna 4.5 BootstrapFewShotWithOptuna oracle.

Run from the repository root:
  tools/dspy_reference/.venv/bin/python tools/dspy_reference/bootstrap_optuna_oracle.py

Only the oracle injects a seeded create_study sampler. The real optimizer,
bootstrap, evaluation, Trial.suggest_int and integer TPE implementations execute
unchanged. Deterministic programs record genuine DSPy traces without using a
model; these receipts are not authenticated inference evidence. This script
prints evidence and never rewrites native tests or verification fixtures.
"""

import contextlib
import hashlib
import io
import json
import math
import platform
import warnings
from unittest.mock import patch

import dspy
import numpy as np
import optuna
from optuna.distributions import IntDistribution
from optuna.samplers._tpe import _truncnorm
from optuna.samplers._tpe._erf import _erf_right_non_big
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.trial import TrialState, create_trial

from dspy.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna


optuna.logging.set_verbosity(optuna.logging.ERROR)


def sequence(seed, params, count=36, tied=False, history=0, missing=False, queued=False):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    for i in range(history):
        values = {name: low + ((i * (j + 3)) % (high - low + 1))
                  for j, (name, low, high) in enumerate(params)
                  if not (missing and j == 1 and i % 7 == 0)}
        study.add_trial(create_trial(params=values,
                                     distributions={name: IntDistribution(low, high)
                                                    for name, low, high in params if name in values},
                                     value=float((i * 17) % 101)))
    # Failed observations must not count toward startup or the Parzen history.
    if missing:
        for _ in range(3):
            study.add_trial(create_trial(state=TrialState.FAIL))
    rows = []
    for i in range(history, history + count):
        fixed = {params[0][0]: params[0][2]} if queued and i in (0, 11) else {}
        if fixed:
            study.enqueue_trial(fixed)
        trial = study.ask()
        chosen = [trial.suggest_int(name, low, high) for name, low, high in params]
        score = 1.0 if tied else float((sum((j + 3) * x for j, x in enumerate(chosen)) + i * 7) % 31)
        study.tell(trial, score)
        rows.append({"params": chosen, "value": score, "fixed": fixed})
    return {"seed": seed, "params": params, "history": history, "missing": missing,
            "failed_history": 3 if missing else 0, "trials": rows}


def dynamic_sequence():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=41))
    rows = []
    for i in range(36):
        low, high = -3 + i % 3, 7 + i % 5
        trial = study.ask()
        x = trial.suggest_int("x", low, high)
        score = float((x * 13 + i * 7) % 19)
        study.tell(trial, score)
        rows.append({"low": low, "high": high, "x": x, "value": score})
    return rows


def fixed_probe():
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=5))
    study.enqueue_trial({"x": 99})
    trial = study.ask()
    first = trial.suggest_int("x", 0, 5)
    repeated = trial.suggest_int("x", 1, 3)
    following = trial.suggest_int("y", 0, 3)
    study.tell(trial, 1.0)
    assert [first, repeated, following] == [99, 99, 0]
    return [first, repeated, following]


def numerical_probe():
    sampler = optuna.samplers.TPESampler(seed=19)
    observations = np.array([0, 1, 1, 4, 5])
    mpe = _ParzenEstimator({"x": observations}, {"x": IntDistribution(0, 5)},
                          sampler._parzen_estimator_parameters)
    distribution = mpe._mixture_distribution.distributions[0]
    a = np.array([-40.0, -20.0, -9.0, -3.0, -1.0, 0.0, 1.0, 8.0])
    b = a + np.array([0.1, 0.5, 1.0, 0.3, 2.0, 1.0, 2.0, 0.1])
    q = np.array([0.0, 1e-12, 0.01, 0.25, 0.5, 0.75, 1 - 1e-12, 1.0])
    erf_inputs = np.array([0.25, 0.5, 1.0]) / math.sqrt(2)
    tied_mpe = _ParzenEstimator({"x": np.array([0, 1])}, {"x": IntDistribution(0, 1)},
                               sampler._parzen_estimator_parameters)
    prior_masses = _truncnorm._log_gauss_mass(np.array([-0.5, 0.0, -0.5]),
                                            np.array([0.0, 0.5, 0.5]))
    log1p_grid = np.concatenate((
        np.linspace(-0.999999999999, -0.000000000001, 10001),
        np.array([0.0, -0.0, 1e-20, -1e-20, 1e-100, -1e-100, 1.0, 1e20]),
    ))
    log1p_values = np.log1p(log1p_grid)
    log1p_boundary = -2.0 * _truncnorm._ndtr_single(-0.5)
    return {"observations": observations.tolist(), "mus": distribution.mu.tolist(),
            "sigmas": distribution.sigma.tolist(), "weights": mpe._mixture_distribution.weights.tolist(),
            "pdf": np.exp(mpe.log_pdf({"x": np.arange(6)})).tolist(),
            "samples": mpe.sample(np.random.RandomState(19), 24)["x"].astype(int).tolist(),
            "mass_a": a.tolist(), "mass_b": b.tolist(),
            "log_mass": _truncnorm._log_gauss_mass(a, b).tolist(),
            "q": q.tolist(), "ppf": _truncnorm.ppf(q, a, b).tolist(),
            "log1p_grid": {
                "count": len(log1p_grid),
                "sha256_f64le": hashlib.sha256(log1p_values.astype("<f8").tobytes()).hexdigest(),
                "boundary_input": log1p_boundary.hex(),
                "boundary_result": float(np.log1p(log1p_boundary)).hex(),
            },
            "erf_boundary": {
                "inputs": [float(x).hex() for x in erf_inputs],
                "actual_scalar": [math.erf(float(x)).hex() for x in erf_inputs],
                "source_rational": [float(x).hex() for x in _erf_right_non_big(erf_inputs)],
                "cdf_minus_half": _truncnorm._ndtr_single(-0.5).hex(),
                "logcdf_minus_half": _truncnorm._log_ndtr_single(-0.5).hex(),
                "left_right_normalizer": [float(x).hex() for x in prior_masses],
                "tied_log_pdf": [float(x).hex() for x in tied_mpe.log_pdf({"x": np.array([0, 1])})],
            }}


class DemoProgram(dspy.Module):
    """Value program with deliberately nonalphabetic predictor declaration order."""

    def __init__(self, fail=False, fail_after_first=False):
        super().__init__()
        self.z = dspy.Predict("question -> answer")
        self.a = dspy.Predict("question -> answer")
        self.fail = fail
        self.fail_after_first = fail_after_first

    def forward(self, question):
        selected = [int(p.demos[0]["answer"]) if p.demos else -1 for p in (self.z, self.a)]
        if self.fail and question == "fail" and (not self.fail_after_first or selected != [0, 2]):
            raise RuntimeError("oracle evaluation failure")
        for p in (self.z, self.a):
            dspy.settings.trace.append((p, {"question": question}, {"answer": question}))
        return dspy.Prediction(answer=question, z=selected[0], a=selected[1])


def metric(example, prediction, trace=None):
    if trace is not None:
        return 1.0
    return float((prediction.z * 2 + prediction.a) % 5 == int(example.question) % 5)


def tied_metric(example, prediction, trace=None):
    return 1.0


def nan_metric(example, prediction, trace=None):
    return 1.0 if trace is not None else math.nan


def mixed_nan_metric(example, prediction, trace=None):
    return 1.0 if trace is not None or prediction.z != 0 else math.nan


def teacher_metric(example, prediction, trace=None):
    return 1.0 if trace is None else float(prediction.z == 88 and prediction.a == 77)


def reset_teacher_metric(example, prediction, trace=None):
    return 1.0 if trace is None else float(prediction.z == -1 and prediction.a == -1)


def examples(values):
    return [dspy.Example(question=str(i), answer=str(i)).with_inputs("question") for i in values]


def optimizer_probe(seed=9, val_kind="explicit", max_demos=3, max_rounds=1,
                    count=20, tie=False, teacher=False, compiled=False,
                    empty_train=False, empty_demos=False, nan=False, failure=False,
                    mismatch=False, failure_count=10, later_failure=False, max_labeled_demos=5,
                    teacher_sensitive=False, reset_sensitive=False, mixed_nan=False):
    studies = []
    create_study = optuna.create_study

    def seeded_study(*args, **kwargs):
        assert "sampler" not in kwargs
        kwargs["sampler"] = optuna.samplers.TPESampler(seed=seed)
        study = create_study(*args, **kwargs)
        studies.append(study)
        return study

    student = DemoProgram(fail=failure, fail_after_first=later_failure)
    student._compiled = compiled
    student.z.demos = [{"question": "original", "answer": "99"}]
    teacher_program = DemoProgram() if teacher else None
    if teacher_program is not None:
        teacher_program.z.demos = [{"question": "teacher", "answer": "88"}]
        teacher_program.a.demos = [{"question": "teacher", "answer": "77"}]
        teacher_program._compiled = teacher_sensitive
        if mismatch:
            teacher_program.a.signature = teacher_program.a.signature.with_instructions("different")
    trainset = [] if empty_train else examples(range(6))
    valset = {"none": None, "empty": [], "explicit": examples([0, 0, 1])}[val_kind]
    if failure:
        valset = examples(["fail"] * failure_count)
    selected_metric = (teacher_metric if teacher_sensitive else reset_teacher_metric if reset_sensitive else
                       mixed_nan_metric if mixed_nan else nan_metric if nan else tied_metric if tie else metric)
    opt = BootstrapFewShotWithOptuna(metric=selected_metric,
                                    max_bootstrapped_demos=99,
                                    max_labeled_demos=0 if empty_demos else max_labeled_demos,
                                    max_rounds=max_rounds, num_candidate_programs=count, num_threads=1)
    error = None
    best = None
    with patch.object(optuna, "create_study", seeded_study):
        try:
            best = opt.compile(student, teacher=teacher_program, max_demos=0 if empty_demos else max_demos,
                               trainset=trainset, valset=valset)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    rows = []
    if studies:
        for trial in studies[0].trials:
            program = trial.user_attrs.get("program")
            rows.append({"number": trial.number, "state": trial.state.name,
                         "params": trial.params, "value": trial.value,
                         "demos": None if program is None else
                         [[dict(demo) for demo in p.demos] for _, p in program.named_predictors()]})
    candidates = getattr(opt, "compiled_teleprompter", None)
    return {"seed": seed, "validation": val_kind, "max_demos": max_demos, "max_rounds": max_rounds,
            "error": error, "original_demo": student.z.demos,
            "teacher_demo": None if teacher_program is None else teacher_program.z.demos,
            "predictor_names": [] if candidates is None else [n for n, _ in candidates.named_predictors()],
            "candidate_demos": None if candidates is None else
            [[dict(demo) for demo in p.demos] for _, p in candidates.named_predictors()],
            "best_number": None if best is None else studies[0].best_trial.number,
            "best_compiled": None if best is None else best._compiled,
            "trials": rows}


def main():
    assert dspy.__version__ == "3.5.0", dspy.__version__
    assert optuna.__version__ == "4.5.0", optuna.__version__
    # Capture DSPy's informational output/progress, keeping stdout pure JSON.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenarios = [
            sequence(0, [("z", 0, 5), ("a", -7, 13), ("fixed", 4, 4)]),
            sequence(9, [("x", 0, 1), ("fixed", -4, -4)], tied=True),
            sequence(19, [("z", -5, 8), ("a", 10, 23), ("fixed", 7, 7)], count=20, queued=True),
            sequence(19, [("z", -5, 8), ("a", 10, 23), ("fixed", 7, 7)], count=6, history=300, missing=True),
        ]
        probes = {
            "explicit": optimizer_probe(),
            "defaults": optimizer_probe(count=16, max_labeled_demos=16),
            "none": optimizer_probe(val_kind="none"),
            "empty": optimizer_probe(val_kind="empty"),
            "tie": optimizer_probe(tie=True, teacher=True),
            "zero_rounds": optimizer_probe(max_rounds=0),
            "zero_trials": optimizer_probe(count=0),
            "empty_train": optimizer_probe(empty_train=True, count=0),
            "compiled_empty_train": optimizer_probe(compiled=True, empty_train=True),
            "mismatch_empty_train": optimizer_probe(teacher=True, mismatch=True, empty_train=True),
            "empty_demos": optimizer_probe(empty_demos=True),
            "nan": optimizer_probe(nan=True, count=3),
            "mixed_nan": optimizer_probe(mixed_nan=True, count=30),
            "evaluation_failure": optimizer_probe(failure=True, count=3),
            "partial_failure": optimizer_probe(failure=True, failure_count=1, tie=True, count=3),
            "later_failure": optimizer_probe(failure=True, later_failure=True, tie=True, count=3),
            "preserved_teacher": optimizer_probe(teacher=True, teacher_sensitive=True,
                                                max_labeled_demos=0, max_demos=1, count=1),
            "reset_teacher": optimizer_probe(reset_sensitive=True, max_labeled_demos=0, max_demos=1, count=1),
        }
        assert probes["none"]["trials"] == probes["empty"]["trials"]
        assert probes["tie"]["best_number"] == 0
        assert probes["explicit"]["best_compiled"] is False
        assert [t["value"] for t in probes["partial_failure"]["trials"]] == [0.0, 0.0, 0.0]
        assert [t["state"] for t in probes["later_failure"]["trials"]] == ["COMPLETE", "FAIL"]
        assert probes["preserved_teacher"]["trials"][0]["value"] == 100.0
        assert probes["reset_teacher"]["trials"][0]["value"] == 100.0
        receipt = {"dspy": dspy.__version__, "optuna": optuna.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "sampler": {"multivariate": False,
                   "n_startup_trials": 10, "n_ei_candidates": 24}, "sequences": scenarios,
                   "dynamic": dynamic_sequence(), "fixed": fixed_probe(),
                   "numerical": numerical_probe(), "optimizer": probes}
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
