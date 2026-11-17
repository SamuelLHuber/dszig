"""Independent locked-DSPy 3.5 LabeledFewShot / RandomSearch compile-options oracle.

Run from the repository root:
  devenv shell -- uv run --project tools/dspy_reference --locked \
    python tools/dspy_reference/fewshot_options_oracle.py RECEIPT.json

The real LabeledFewShot.compile(sample=...) and
BootstrapFewShotWithRandomSearch.compile(restrict=..., labeled_sample=...)
execute unchanged. Deterministic value programs record genuine DSPy traces
without a model, so the receipt is not inference evidence. DSPy's own stdout and
stderr are left untouched for the caller to conserve; the JSON receipt goes to
the path argument. This script never rewrites native tests or fixtures.
"""

import json
import platform
import sys

import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot

COUNTS = {"forward": 0, "metric": 0}


class DemoProgram(dspy.Module):
    """Two predictors in deliberately nonalphabetic declaration order."""

    def __init__(self):
        super().__init__()
        self.z = dspy.Predict("question -> answer")
        self.a = dspy.Predict("question -> answer")

    def forward(self, question):
        COUNTS["forward"] += 1
        first = [int(p.demos[0]["answer"]) if p.demos else -1 for p in (self.z, self.a)]
        for p in (self.z, self.a):
            dspy.settings.trace.append((p, {"question": question}, {"answer": question}))
        return dspy.Prediction(answer=question, z=first[0], a=first[1],
                               nz=len(self.z.demos), na=len(self.a.demos))


class EmptyProgram(dspy.Module):
    """No predictors: Source never reaches rng.sample, even for negative k."""

    def forward(self, question):
        COUNTS["forward"] += 1
        return dspy.Prediction(answer=question)


def metric(example, prediction, trace=None):
    COUNTS["metric"] += 1
    if trace is not None:
        return 1.0
    # Integer-valued and candidate-sensitive, so ranking does not depend on float sums.
    return float((2 * prediction.z + prediction.a + prediction.nz + int(example.question)) % 7)


def tied_metric(example, prediction, trace=None):
    COUNTS["metric"] += 1
    return 1.0


def teacher_metric(example, prediction, trace=None):
    """Bootstrap accepts a trace only when the supplied teacher actually ran."""
    COUNTS["metric"] += 1
    if trace is None:
        return float(prediction.nz)
    return float(prediction.z == 88 and prediction.a == 77)


METRICS = {"metric": metric, "tied": tied_metric, "teacher": teacher_metric}


def examples(values):
    return [dspy.Example(question=str(i), answer=str(i)).with_inputs("question") for i in values]


def answers(program):
    return {name: [demo["answer"] for demo in p.demos] for name, p in program.named_predictors()}


def failure(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def student_program(empty=False):
    student = EmptyProgram() if empty else DemoProgram()
    if not empty:
        # reset_copy must discard this demo in every successful compile.
        student.z.demos = [{"question": "original", "answer": "99"}]
    return student


def labeled_case(k, sample, train=6, empty_program=False):
    student = student_program(empty_program)
    trainset = examples(range(train))
    row = {"k": str(k), "sample": sample, "train": train, "empty_program": empty_program,
           "error": None, "demos": None, "prefix_aliases_trainset": None}
    try:
        compiled = LabeledFewShot(k=k).compile(student, trainset=trainset, sample=sample)
        row["demos"] = answers(compiled)
        predictors = compiled.predictors()
        if predictors and predictors[0].demos:
            # Source assigns the caller's Example objects; native owns copies.
            row["prefix_aliases_trainset"] = predictors[0].demos[0] is trainset[int(row["demos"]["z"][0])]
    except Exception as exc:
        row["error"] = failure(exc)
    row["original_demos"] = None if empty_program else [d["answer"] for d in student.z.demos]
    return row


def search_case(name, restrict, labeled_sample=True, teacher="none", metric_name="metric",
                stop_at_score=None, candidates=4, train=6, val=(0, 1, 2, 3, 4),
                max_labeled_demos=3, max_bootstrapped_demos=2):
    COUNTS["forward"] = COUNTS["metric"] = 0
    student = student_program()
    teacher_value = None
    if teacher == "list":
        teacher_value = [DemoProgram()]
    elif teacher == "empty_list":
        teacher_value = []
    elif teacher == "program":
        teacher_value = DemoProgram()
        teacher_value.z.demos = [{"question": "teacher", "answer": "88"}]
        teacher_value.a.demos = [{"question": "teacher", "answer": "77"}]
        teacher_value._compiled = True
    trainset = examples(range(train))
    valset = None if val is None else examples(val)
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=METRICS[metric_name], max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos, num_candidate_programs=candidates,
        num_threads=1, stop_at_score=stop_at_score)
    row = {"name": name, "restrict": None if restrict is None else list(restrict),
           "labeled_sample": labeled_sample, "teacher": teacher, "metric": metric_name,
           "stop_at_score": stop_at_score, "num_candidate_programs": candidates, "train": train,
           "val": None if val is None else list(val), "max_labeled_demos": max_labeled_demos,
           "max_bootstrapped_demos": max_bootstrapped_demos,
           "error": None, "candidates": None, "best_demos": None}
    try:
        best = optimizer.compile(student, teacher=teacher_value, trainset=trainset, valset=valset,
                                 restrict=restrict, labeled_sample=labeled_sample)
        row["candidates"] = [{"seed": c["seed"], "score": c["score"]} for c in best.candidate_programs]
        row["best_demos"] = answers(best)
    except Exception as exc:
        row["error"] = failure(exc)
    row["forward_calls"] = COUNTS["forward"]
    row["metric_calls"] = COUNTS["metric"]
    row["original_demos"] = [d["answer"] for d in student.z.demos]
    return row


def main():
    assert dspy.__version__ == "3.5.0", dspy.__version__
    counts = [0, 1, 3, 6, 16, 2**63, -1, -2, -5, -6, -7, -100, -(2**127)]
    labeled = [labeled_case(k, sample) for sample in (True, False) for k in counts]
    labeled += [labeled_case(k, sample, train=0) for sample in (True, False) for k in (3, -1)]
    labeled += [labeled_case(-1, sample, empty_program=True) for sample in (True, False)]
    search = [
        search_case("all_sampled", None),
        search_case("all_prefix", None, labeled_sample=False),
        search_case("labeled_sampled", [-2]),
        search_case("labeled_prefix", [-2], labeled_sample=False),
        search_case("prefix_ignored_outside_labeled_seed", [-3, -1, 1], labeled_sample=False),
        search_case("duplicates_out_of_range_unordered", [3, -3, 3, -3, 99, -99, 4, -4]),
        search_case("bootstrap_subset", [2, 0, -1]),
        search_case("empty_restrict", []),
        search_case("no_intersection", [4, -4, 100]),
        search_case("no_candidates_shuffled_seed", [0], candidates=0),
        search_case("no_candidates_fixed_seeds", [-1, -2, -3, 0], candidates=0),
        search_case("teacher_list_unconsumed", [-3, -2], teacher="list"),
        search_case("teacher_empty_list_unconsumed", [-2], teacher="empty_list", labeled_sample=False),
        search_case("teacher_list_consumed_after_labeled", [-2, -1], teacher="list"),
        search_case("teacher_list_consumed_first", [1], teacher="list"),
        search_case("teacher_list_early_stop", None, teacher="list", stop_at_score=0.0),
        search_case("restricted_early_stop", [1, 2, 3], stop_at_score=0.0),
        search_case("threshold_stop_sampled", None, stop_at_score=360.0),
        search_case("threshold_stop_prefix", None, labeled_sample=False, stop_at_score=360.0),
        search_case("threshold_stop_restricted", [3, 2, 0, -3], stop_at_score=360.0),
        search_case("teacher_program_consumed", [-1, 0], teacher="program", metric_name="teacher"),
        search_case("teacher_program_default", [-1, 0], metric_name="teacher"),
        search_case("stable_ties", [2, 0, -2, 3], metric_name="tied"),
        search_case("empty_valset_prefix", [-3, -2], labeled_sample=False, val=()),
        search_case("missing_valset_prefix", [-3, -2], labeled_sample=False, val=None),
        search_case("empty_trainset_reset", [-3, -2], train=0),
        search_case("empty_trainset_bootstrap", [-1], train=0),
        search_case("labeled_prefix_wide", [-2], labeled_sample=False, max_labeled_demos=16),
        search_case("labeled_sampled_wide", [-2], max_labeled_demos=16),
        search_case("labeled_zero", [-2], labeled_sample=False, max_labeled_demos=0),
    ]
    receipt = {"dspy": dspy.__version__, "python": platform.python_version(),
               "platform": platform.platform(), "labeled": labeled, "search": search,
               "counts": {"labeled": len(labeled), "search": len(search),
                          "labeled_errors": sum(r["error"] is not None for r in labeled),
                          "search_errors": sum(r["error"] is not None for r in search)}}
    with open(sys.argv[1], "x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
