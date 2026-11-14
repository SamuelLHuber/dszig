"""Exercise the installed DSPy 3.5 InferRules implementation without network.

Run: tools/dspy_reference/.venv/bin/python tools/dspy_reference/infer_rules_oracle.py
The recording engine answers real DSPy adapter requests; optimizer control flow,
CoT expansion, prompt formatting, copying, evaluation, RNG and retries are DSPy's.
"""

import contextlib
import hashlib
import io
import json
import math
import threading

import dspy
from dspy.teleprompt.infer_rules import InferRules, RulesInductionProgram
from dspy.utils.callback import BaseCallback

import reference as ref


class Capture(BaseCallback):
    def __init__(self):
        self.local = threading.local()

    def on_adapter_format_start(self, call_id, instance, inputs):
        self.local.signature = inputs["signature"]
        self.local.inputs = inputs["inputs"]


class RecordingLM(ref.ReferenceLM):
    def __init__(self, label, capture, state):
        super().__init__(model="mock/infer-rules-" + label, temperature=0.25,
                         max_tokens=1000, cache=False)
        self.label, self.capture, self.state = label, capture, state

    def __call__(self, prompt=None, **options):
        # Typed pre-provider failures are thrown at the LM call boundary. Errors
        # thrown by an engine are wrapped by DSPy 3.5, and are a separate probe.
        if ("natural_language_rules" in self.capture.local.signature.output_fields
                and self.state.get("failures")):
            failure = self.state["failures"].pop(0)
            if failure is not None:
                row = self.record(prompt)
                row["rollout_id"] = options.get("rollout_id", self.kwargs.get("rollout_id"))
                row["error"] = type(failure).__name__
                raise failure
        return super().__call__(prompt, **options)

    def record(self, request):
        signature = self.capture.local.signature
        inputs = self.capture.local.inputs
        inducing = "natural_language_rules" in signature.output_fields
        row = {
            "lm": self.label,
            "kind": "induce" if inducing else "task",
            "instructions": signature.instructions,
            "fields": [{"name": name, "description": field.json_schema_extra["desc"],
                        "prefix": field.json_schema_extra["prefix"]}
                       for name, field in signature.fields.items()],
            "inputs": dict(inputs),
            "messages": ref.messages_for(request),
            "temperature": request.config.temperature,
            "rollout_id": ref.rollout_id(self),
        }
        self.state["calls"].append(row)
        return row

    def complete(self, request):
        row = self.record(request)
        signature = self.capture.local.signature
        inputs = self.capture.local.inputs
        if "natural_language_rules" in signature.output_fields:
            if self.state.get("engine_failure") is not None:
                raise self.state.pop("engine_failure")
            index = self.state.setdefault("induction_count", 0)
            self.state["induction_count"] = index + 1
            rule = self.state.get("rule_text", "\u2003 rule-" + str(index) + " \u00a0")
            row["rule"] = rule
            text = "<plan>Use the examples.</plan>\n<natural_language_rules>\n" + rule + "\n</natural_language_rules>"
        elif "draft" in signature.output_fields:
            text = "<draft>generated:" + inputs["question"] + "</draft>"
        else:
            good = self.label == "teacher" or "Please adhere" not in signature.instructions
            good = good or any(rule in signature.instructions for rule in self.state.get("good_rules", []))
            text = "<answer>" + ("yes" if good else "no") + "</answer>"
        row["completion"] = text
        return ref.text_response(self.model, text)


class First(dspy.Signature):
    """Original first."""
    question: str = dspy.InputField()
    draft: str = dspy.OutputField()


class Second(dspy.Signature):
    """Original second."""
    draft: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        # Fresh signature classes keep independent probes independent. Within
        # one compile, real DSPy deepcopy still shares these signature classes.
        self.first = dspy.Predict(First.with_instructions("Original first."))
        self.second = dspy.Predict(Second.with_instructions("Original second."))

    def forward(self, question):
        draft = self.first(question=question).draft
        return self.second(draft=draft, question=question)


def instructions(program):
    return [predictor.signature.instructions for predictor in program.predictors()]


def snapshot(program):
    return {"instructions": instructions(program),
            "demos": [[dict(demo) for demo in predictor.demos] for predictor in program.predictors()]}


class ObservedInferRules(InferRules):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.induction_snapshots, self.evaluations, self.evaluated_programs = [], [], []

    def induce_natural_language_rules(self, predictor, trainset):
        self.induction_snapshots.append({"instruction": predictor.signature.instructions,
                                         "examples": [dict(example) for example in trainset]})
        return super().induce_natural_language_rules(predictor, trainset)

    def evaluate_program(self, program, dataset):
        row = {"program": snapshot(program), "validation": [dict(example) for example in dataset]}
        self.evaluations.append(row)
        self.evaluated_programs.append(program)
        row["score"] = super().evaluate_program(program, dataset)
        return row["score"]


def examples(count):
    return [dspy.Example(answer="yes", ignored="discard", question="q" + str(index),
                         draft="labeled:" + str(index)).with_inputs("question") for index in range(count)]


def metric(example, prediction, trace=None):
    return prediction.answer == example.answer


@contextlib.contextmanager
def environment(state=None, adapter=None):
    state = state if state is not None else {"calls": []}
    capture = Capture()
    student = RecordingLM("student", capture, state)
    teacher = RecordingLM("teacher", capture, state)
    # Silence only presentation; the real Evaluate and BootstrapFewShot run.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            dspy.context(lm=student, adapter=adapter or dspy.XMLAdapter(), callbacks=[capture], num_threads=1):
        yield state, teacher


def defaults_probe():
    optimizer = InferRules()
    assert optimizer.num_candidates == optimizer.num_rules == 10
    assert optimizer.num_threads is None
    return {"num_candidates": optimizer.num_candidates, "num_rules": optimizer.num_rules,
            "num_threads": optimizer.num_threads}


def rules_probe():
    with environment(adapter=dspy.ChatAdapter()) as (state, teacher):
        program = RulesInductionProgram(7, teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
        outputs = [program("first"), program("second"), program("third")]
        assert outputs == ["rule-0", "rule-1", "rule-2"]
        calls = state["calls"]
        assert [row["rollout_id"] for row in calls] == [906691059, 413653999, 813847339]
        assert all(row["lm"] == "teacher" and row["temperature"] == 1.0 for row in calls)
        assert [row["name"] for row in calls[0]["fields"]] == ["examples_text", "plan", "natural_language_rules"]
        assert [row["description"] for row in calls[0]["fields"]] == [
            "Text containing examples", "A short plan for solving the task.", "Induced natural language rules"]
        assert "a list of 7 concise" in calls[0]["instructions"]
        return {"outputs": outputs, "calls": calls}


def formatting_probe():
    class Formatting(dspy.Signature):
        question: object = dspy.InputField()
        nested: object = dspy.InputField()
        flag: object = dspy.OutputField()
        answer: object = dspy.OutputField()

    predictor = dspy.Predict(Formatting)
    trainset = [dspy.Example(answer=None, unused="skip", nested={"yes": False}, question=[True, "can't"], flag=True),
                dspy.Example(unused="missing all"),
                dspy.Example(flag=False, question="\u2003 raw \u00a0")]
    optimizer = InferRules()
    demos = optimizer.get_predictor_demos(trainset, predictor)
    result = optimizer.format_examples(demos, predictor.signature)
    expected = ("Input Fields:\nnested: {'yes': False}\nquestion: [True, \"can't\"]\n\n=========\n"
                "Output Fields:\nanswer: None\nflag: True\n\n"
                "Input Fields:\n\n\n=========\nOutput Fields:\n\n\n"
                "Input Fields:\nquestion: \u2003 raw \u00a0\n\n=========\nOutput Fields:\nflag: False\n\n")
    assert result == expected, repr(result)
    return {"demos": demos, "text": result}


def compile_probe():
    with environment({"calls": [], "good_rules": ["rule-3", "rule-5"]}) as (state, teacher):
        optimizer = ObservedInferRules(num_candidates=3, num_rules=7, num_threads=1, metric=metric,
                                      max_bootstrapped_demos=1, max_labeled_demos=0,
                                      teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
        student = Program()
        best = optimizer.compile(student, trainset=examples(5))
        assert [row["score"] for row in optimizer.evaluations] == [0.0, 100.0, 100.0]
        assert best is optimizer.evaluated_programs[1]  # Strict comparison keeps the first 100-point candidate.
        selected_snapshot = optimizer.evaluations[1]["program"]
        assert selected_snapshot["instructions"] == [
            "Original first.\n\nPlease adhere to the following rules when making your prediction:\nrule-2",
            "Original second.\n\nPlease adhere to the following rules when making your prediction:\nrule-3"]
        # Record, do not hide, the pinned source's signature-class aliasing bug:
        # later candidates mutate the selected object, bootstrap and caller.
        final_alias = instructions(optimizer.evaluated_programs[2])
        assert instructions(best) == instructions(optimizer.student) == instructions(student) == final_alias
        source_aliasing = {"selected_candidate": 1, "selected_snapshot": selected_snapshot,
                           "returned": snapshot(best), "bootstrap_final": snapshot(optimizer.student),
                           "caller_final": snapshot(student)}
        assert [len(row["validation"]) for row in optimizer.evaluations] == [3, 3, 3]
        assert all(row["instruction"] in ("Original first.", "Original second.") for row in optimizer.induction_snapshots)
        induction_calls = [row for row in state["calls"] if row["kind"] == "induce"]
        assert len(induction_calls) == 6
        assert all("labeled:0" in row["inputs"]["examples_text"] for row in induction_calls)
        assert all("generated:" not in row["inputs"]["examples_text"] for row in induction_calls)
        assert all("q2" not in row["inputs"]["examples_text"] for row in induction_calls)
        # Reusing the optimizer continues the same Random(0) stream.
        second = optimizer.compile(Program(), trainset=examples(2), valset=examples(1))
        repeated = [row["rollout_id"] for row in state["calls"] if row["kind"] == "induce"]
        assert repeated[:8] == [906691059, 413653999, 813847339, 955892128,
                                451585301, 43469773, 278009742, 548977048]
        assert second is not None
        return {"source_signature_aliasing": source_aliasing,
                "induction_snapshots": optimizer.induction_snapshots,
                "evaluations": optimizer.evaluations, "calls": state["calls"]}


def persistent_rules_configuration_probe():
    with environment() as (state, teacher):
        original_settings = {"lm": teacher, "adapter": dspy.XMLAdapter()}
        optimizer = ObservedInferRules(num_candidates=1, num_rules=7, metric=metric, num_threads=1,
                                      max_bootstrapped_demos=0, max_labeled_demos=0,
                                      teacher_settings=original_settings)
        rules_program = optimizer.rules_induction_program
        original_signature = rules_program.rules_induction.predict.signature
        assert optimizer.teacher_settings is rules_program.teacher_settings is original_settings
        # The constructor has already captured 7, even before its first compile.
        optimizer.num_rules = 99
        optimizer.compile(Program(), trainset=examples(2), valset=examples(1))
        first = [row for row in state["calls"] if row["kind"] == "induce"]
        assert len(first) == 2 and all("a list of 7 concise" in row["instructions"] for row in first)

        replacement = RecordingLM("replacement", teacher.capture, state)
        optimizer.num_rules = -2
        optimizer.teacher_settings = {"lm": replacement, "adapter": dspy.XMLAdapter()}
        rules_program.rules_induction.predict.signature.instructions = "Caller-owned induction instructions."
        optimizer.compile(Program(), trainset=examples(2), valset=examples(1))
        second = [row for row in state["calls"] if row["kind"] == "induce"][2:]
        assert len(second) == 2
        assert all(row["lm"] == "teacher" and row["instructions"] == "Caller-owned induction instructions." for row in second)
        assert optimizer.rules_induction_program is rules_program
        assert rules_program.rules_induction.predict.signature is original_signature

        # Reassignment did not retarget the captured dictionary; mutating the
        # original dictionary itself remains observable through the shared ref.
        original_settings["lm"] = replacement
        optimizer.compile(Program(), trainset=examples(2), valset=examples(1))
        third = [row for row in state["calls"] if row["kind"] == "induce"][4:]
        assert len(third) == 2
        assert all(row["lm"] == "replacement" and row["instructions"] == "Caller-owned induction instructions." for row in third)
        assert rules_program.teacher_settings is original_settings
        assert optimizer.teacher_settings is not original_settings
        inducing = [row for row in state["calls"] if row["kind"] == "induce"]
        assert [row["rollout_id"] for row in inducing] == [
            906691059, 413653999, 813847339, 955892128, 451585301, 43469773]
        return {"rules_program_preserved": optimizer.rules_induction_program is rules_program,
                "signature_class_preserved": rules_program.rules_induction.predict.signature is original_signature,
                "captured_original_settings": rules_program.teacher_settings is original_settings,
                "calls": inducing}


def split_and_null_probe():
    rows = []
    for label, count, candidates, validation, score in [
            ("explicit-validation", 3, 1, examples(1), 1.0),
            ("zero-candidates-empty-validation", 2, 0, [], 1.0),
            ("negative-candidates", 2, -2, [], 1.0),
            ("all-nan", 2, 2, examples(1), float("nan")),
            ("all-negative-infinity", 2, 2, examples(1), -math.inf)]:
        with environment() as (state, teacher):
            optimizer = ObservedInferRules(num_candidates=candidates, num_threads=1,
                                          metric=lambda *args: score, max_bootstrapped_demos=0,
                                          max_labeled_demos=0,
                                          teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
            result = optimizer.compile(Program(), trainset=examples(count), valset=validation)
            assert (result is not None) == (label == "explicit-validation")
            assert len(optimizer.trainset) == count
            rows.append({"case": label, "returned_null": result is None,
                         "training_count": len(optimizer.trainset),
                         "induction_calls": sum(row["kind"] == "induce" for row in state["calls"]),
                         "scores": [str(row["score"]) for row in optimizer.evaluations]})
    for label, trainset, valset, expected_error, expected_calls in [
            ("explicit-empty-validation", examples(2), [], ValueError, 2),
            ("omitted-validation-single-training-example", examples(1), None, UnboundLocalError, 0)]:
        with environment() as (state, teacher):
            optimizer = ObservedInferRules(num_candidates=1, metric=metric, num_threads=1,
                                          max_bootstrapped_demos=0, max_labeled_demos=0,
                                          teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
            try:
                optimizer.compile(Program(), trainset=trainset, valset=valset)
            except expected_error as exc:
                rows.append({"case": label, "error": type(exc).__name__, "message": str(exc),
                             "induction_calls": len(state["calls"])})
            else:
                raise AssertionError("Expected " + expected_error.__name__)
            assert len(state["calls"]) == expected_calls
    return rows


def shrink_probe():
    BadRequestError = type("BadRequestError", (Exception,), {})
    failures = [ValueError("too long"), BadRequestError("too long"),
                RuntimeError("wrapped ContextWindowExceededError from provider"), None]
    rows = []
    with environment({"calls": [], "failures": failures}) as (state, teacher):
        optimizer = InferRules(teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
        result = optimizer.induce_natural_language_rules(dspy.Predict(First), examples(4))
        assert result == "rule-0"
        counts = [row["inputs"]["examples_text"].count("Input Fields:") for row in state["calls"]]
        assert counts == [4, 3, 2, 1], counts
        assert [row["rollout_id"] for row in state["calls"]] == [906691059, 413653999, 813847339, 955892128]
        rows.append({"case": "accepted-errors", "counts": counts, "result": result, "calls": state["calls"]})
    for label, failure, count, error_type in [("unaccepted", LookupError("not context"), 3, AssertionError),
                                              ("single", ValueError("too long"), 1, RuntimeError),
                                              ("empty", ValueError("too long"), 0, RuntimeError)]:
        with environment({"calls": [], "failures": [failure]}) as (state, teacher):
            optimizer = InferRules(teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
            try:
                optimizer.induce_natural_language_rules(dspy.Predict(First), examples(count))
            except error_type as exc:
                assert len(state["calls"]) == 1
                rows.append({"case": label, "error": type(exc).__name__, "message": str(exc),
                             "calls": state["calls"]})
            else:
                raise AssertionError("Expected " + error_type.__name__)
    return rows


def reserved_and_strip_probe():
    class Reserved(dspy.Signature):
        question: str = dspy.InputField()
        dspy_secret: str = dspy.InputField()
        answer: str = dspy.OutputField()

    optimizer = InferRules()
    predictor = dspy.Predict(Reserved)
    demos = optimizer.get_predictor_demos(
        [dspy.Example(dspy_secret="hidden", question="visible", answer="yes")], predictor)
    text = optimizer.format_examples(demos, predictor.signature)
    assert demos == [{"question": "visible", "answer": "yes"}]
    assert text == "Input Fields:\nquestion: visible\n\n=========\nOutput Fields:\nanswer: yes\n\n"
    spaces = [codepoint for codepoint in range(0x110000) if chr(codepoint).isspace()]
    padding = "".join(map(chr, spaces))
    with environment({"calls": [], "rule_text": padding + "x\u200b" + padding}) as (state, teacher):
        program = RulesInductionProgram(-2, teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
        stripped = program("strip")
        assert stripped == "x\u200b"
        assert "a list of -2 concise" in state["calls"][0]["instructions"]
    return {"reserved_demos": demos, "reserved_text": text,
            "whitespace_codepoints": spaces, "stripped": stripped}


def missing_metric_probe():
    with environment() as (state, teacher):
        optimizer = InferRules(num_threads=1)
        program = Program()
        score = optimizer.evaluate_program(program, examples(1))
        assert score == 0.0
        assert len(state["calls"]) == 2
        optimizer.max_errors = 1
        try:
            optimizer.evaluate_program(program, examples(1))
        except Exception as exc:
            assert len(state["calls"]) == 4
            return {"below_limit_score": score, "at_limit_error": type(exc).__name__,
                    "at_limit_message": str(exc), "task_calls": len(state["calls"])}
        raise AssertionError("Missing metric must count as evaluation failure")


def partial_failure_probe():
    rows = []
    for shared in (False, True):
        class Shared(dspy.Signature):
            """Shared task."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        class SharedProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.first = dspy.Predict(Shared)
                self.second = dspy.Predict(Shared)

            def forward(self, question):
                self.first(question=question)
                return self.second(question=question)

        with environment({"calls": [], "failures": [None, LookupError("not context")]}) as (state, teacher_lm):
            student = SharedProgram() if shared else Program()
            optimizer = ObservedInferRules(num_candidates=2, metric=metric, num_threads=1,
                                          max_bootstrapped_demos=0, max_labeled_demos=0,
                                          teacher_settings={"lm": teacher_lm, "adapter": dspy.XMLAdapter()})
            try:
                optimizer.compile(student, trainset=examples(2), valset=examples(1))
            except AssertionError:
                expected = ["Shared task.\n\nPlease adhere to the following rules when making your prediction:\nrule-0"] * 2 if shared else [
                    "Original first.\n\nPlease adhere to the following rules when making your prediction:\nrule-0",
                    "Original second."]
                assert instructions(student) == instructions(optimizer.student) == expected
                assert len(state["calls"]) == 2
                assert not optimizer.evaluations
                assert [row["rollout_id"] for row in state["calls"]] == [906691059, 413653999]
                rows.append({"shared_signature": shared, "error": "AssertionError",
                             "caller_final": snapshot(student), "bootstrap_final": snapshot(optimizer.student),
                             "induction_snapshots": optimizer.induction_snapshots, "calls": state["calls"]})
            else:
                raise AssertionError("Second-predictor failure must escape without evaluating")
    return rows


def alias_probe():
    rows = []
    for shared in (False, True):
        class Shared(dspy.Signature):
            """Shared task."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        class SharedProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.first = dspy.Predict(Shared)
                self.second = dspy.Predict(Shared)

            def forward(self, question):
                self.first(question=question)
                return self.second(question=question)

        with environment({"calls": [], "good_rules": ["rule-1"]}) as (state, teacher_lm):
            student = SharedProgram() if shared else Program()
            teacher = SharedProgram() if shared else None
            optimizer = ObservedInferRules(num_candidates=3, metric=metric, num_threads=1,
                                          max_bootstrapped_demos=0, max_labeled_demos=0,
                                          teacher_settings={"lm": teacher_lm, "adapter": dspy.XMLAdapter()})
            best = optimizer.compile(student, teacher=teacher, trainset=examples(2), valset=examples(1))
            assert [row["score"] for row in optimizer.evaluations] == [100.0, 0.0, 0.0]
            assert best is optimizer.evaluated_programs[0]
            expected = [item + "\n\nPlease adhere to the following rules when making your prediction:\nrule-" + str(index)
                        for item, index in (("Shared task.", 5), ("Shared task.", 5))] if shared else [
                "Original first.\n\nPlease adhere to the following rules when making your prediction:\nrule-4",
                "Original second.\n\nPlease adhere to the following rules when making your prediction:\nrule-5"]
            assert instructions(best) == instructions(student) == instructions(optimizer.student) == expected
            if shared:
                assert instructions(teacher) == expected
                assert optimizer.evaluations[0]["program"]["instructions"] == [
                    "Shared task.\n\nPlease adhere to the following rules when making your prediction:\nrule-1"] * 2
                assert optimizer.induction_snapshots[1]["instruction"].endswith("\nrule-0")
            rows.append({"shared_signature": shared, "selected_candidate": 0,
                         "evaluations": optimizer.evaluations, "induction_snapshots": optimizer.induction_snapshots,
                         "returned": snapshot(best), "caller_final": snapshot(student),
                         "bootstrap_final": snapshot(optimizer.student),
                         "teacher_final": snapshot(teacher) if teacher is not None else None})
    return rows


def wrapped_engine_error_probe():
    with environment({"calls": [], "engine_failure": ValueError("too long")}) as (state, teacher):
        optimizer = InferRules(teacher_settings={"lm": teacher, "adapter": dspy.XMLAdapter()})
        try:
            optimizer.induce_natural_language_rules(dspy.Predict(First), examples(3))
        except AssertionError as exc:
            assert len(state["calls"]) == 1
            assert type(exc.__context__).__name__ == "LMUnexpectedError"
            return {"source_error": type(exc).__name__, "wrapped_error": type(exc.__context__).__name__,
                    "calls": len(state["calls"])}
        raise AssertionError("Engine-wrapped ValueError must not be retried")


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def main():
    assert dspy.__version__ == "3.5.0", dspy.__version__
    receipt = finite_json({"dspy": dspy.__version__, "defaults": defaults_probe(),
                           "rules": rules_probe(), "formatting": formatting_probe(),
                           "compile": compile_probe(), "split_and_null": split_and_null_probe(),
                           "shrink": shrink_probe(), "aliases": alias_probe(),
                           "wrapped_engine_error": wrapped_engine_error_probe(),
                           "reserved_and_strip": reserved_and_strip_probe(),
                           "missing_metric": missing_metric_probe(),
                           "partial_failure": partial_failure_probe(),
                           "persistent_rules_configuration": persistent_rules_configuration_probe()})
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"sha256": hashlib.sha256(payload.encode()).hexdigest(), "receipt": receipt},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
