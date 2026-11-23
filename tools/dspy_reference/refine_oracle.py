"""Exercise installed DSPy 3.5 Refine, including real feedback adapter calls.

Run with tools/dspy_reference/.venv/bin/python tools/dspy_reference/refine_oracle.py.
No network or replacement Refine control flow is used.
"""

import contextlib
import hashlib
import inspect
import io
import json
import math
import threading

import dspy
from dspy.predict.refine import Refine, recursive_mask
from dspy.utils.callback import BaseCallback

import reference as ref


class QA(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class SeparateQA(QA):
    """Answer the question."""


class Capture(BaseCallback):
    def __init__(self):
        self.local = threading.local()

    def on_adapter_format_start(self, call_id, instance, inputs):
        self.local.signature = inputs["signature"]
        self.local.inputs = inputs["inputs"]


class RecordingLM(ref.ReferenceLM):
    def __init__(self, capture, state):
        super().__init__(model="mock/refine", temperature=0.25, max_tokens=1000,
                         rollout_id=7, cache=False)
        self.capture, self.state = capture, state

    def complete(self, request):
        signature = self.capture.local.signature
        inputs = self.capture.local.inputs
        feedback = "advice" in signature.output_fields
        row = {
            "kind": "feedback" if feedback else "task",
            "rollout_id": ref.rollout_id(self),
            "temperature": request.config.temperature,
            "fields": [{"name": name, "description": field.json_schema_extra["desc"],
                        "prefix": field.json_schema_extra["prefix"]}
                       for name, field in signature.fields.items()],
            "instructions": signature.instructions,
            "inputs": dict(inputs),
            "messages": ref.messages_for(request),
        }
        self.state["calls"].append(row)
        if feedback:
            index = self.state.setdefault("feedback_count", 0)
            self.state["feedback_count"] = index + 1
            advice = self.state["advice"][min(index, len(self.state["advice"]) - 1)]
            text = "<discussion>Try a concrete correction.</discussion>\n<advice>" + json.dumps(advice) + "</advice>"
        else:
            text = "<answer>" + str(ref.rollout_id(self)) + "</answer>"
        row["completion"] = text
        return ref.text_response(self.model, text)


class Pair(dspy.Module):
    def __init__(self, separate=False):
        super().__init__()
        self.first = dspy.Predict(QA)
        self.last = dspy.Predict(SeparateQA if separate else QA)

    def forward(self, question):
        first = self.first(question=question)
        return self.last(question=first.answer)


def score(inputs, prediction):
    return float(prediction.answer)


def nan_score(inputs, prediction):
    return float("nan")


def negative_infinity(inputs, prediction):
    return -float("inf")


def small_score(inputs, prediction):
    return -1e-5


def large_score(inputs, prediction):
    return float(2**200 if prediction.answer == "7" else 2**201)


def negative_large_score(inputs, prediction):
    return -float(2**200)


class FailingPair(Pair):
    def forward(self, question):
        raise ValueError("native-parity-program-failure")


def run(name, *, separate=False, n=3, threshold=100.0, advice=None,
        reward=score, fail_count=None, repeat=1, failing=False):
    capture = Capture()
    state = {"calls": [], "advice": advice if advice is not None else [
        {"first": "first-specific", "last": "last-specific"}
    ]}
    lm = RecordingLM(capture, state)
    module = FailingPair(separate) if failing else Pair(separate)
    wrapper = Refine(module, n, reward, threshold, fail_count)
    outcomes, diagnostics, outer_trace = [], io.StringIO(), []
    with contextlib.redirect_stdout(diagnostics), dspy.context(
        lm=lm, adapter=dspy.XMLAdapter(), callbacks=[capture], trace=outer_trace,
        max_trace_size=10000
    ):
        for _ in range(repeat):
            try:
                result = wrapper(question="original-question")
                outcomes.append(None if result is None else dict(result))
            except Exception as error:
                outcomes.append({"error": type(error).__name__, "message": str(error)})
        outer_trace = dspy.settings.trace.copy()
    row = {"name": name, "outcomes": outcomes, "remaining_fail_count": wrapper.fail_count,
           "calls": state["calls"],
           "outer_trace": [{"inputs": dict(inputs), "outputs": dict(outputs)}
                              for _, inputs, outputs in outer_trace],
           "module_code": wrapper.module_code, "reward_code": wrapper.reward_fn_code}
    task_calls = [call for call in row["calls"] if call["kind"] == "task"]
    if name == "shared-class":
        assert [call["inputs"].get("hint_") for call in task_calls] == [
            None, None, "last-specific", "last-specific", "last-specific", "last-specific"
        ]
        assert outcomes == [{"answer": "9"}]
        assert len(outer_trace) == 4
        assert ["advice" in outputs for _, _, outputs in outer_trace] == [True, True, False, False]
    if name == "separate-classes":
        assert [call["inputs"].get("hint_") for call in task_calls] == [
            None, None, "first-specific", "last-specific", "first-specific", "last-specific"
        ]
    if name == "empty-advice":
        assert all("hint_" not in call["inputs"] for call in task_calls)
    if name == "unmapped-advice":
        assert all(call["inputs"]["hint_"] == "N/A" for call in task_calls[2:])
    if name == "threshold":
        assert outcomes == [{"answer": "7"}] and len(row["calls"]) == 2
    if name in ("nan", "negative-infinity"):
        assert outcomes == [None]
        assert len(outer_trace) == 2
        assert all("advice" in outputs for _, _, outputs in outer_trace)
    return row


def main():
    dspy.configure(lm=RecordingLM(Capture(), {"calls": [], "advice": [{}]}))
    rows = [
        run("shared-class"), run("separate-classes", separate=True),
        run("empty-advice", advice=[{}]),
        run("unmapped-advice", advice=[{"other": "not-for-this-program"}]),
        run("threshold", threshold=7.0), run("null-threshold", threshold=None),
        run("nan", reward=nan_score), run("negative-infinity", reward=negative_infinity),
        run("zero-attempts", n=0), run("negative-attempts", n=-1),
        run("persistent-failures", n=4, failing=True, repeat=2),
        run("zero-failure-budget-is-default", n=4, failing=True, fail_count=0),
        run("negative-failure-budget", n=3, failing=True, fail_count=-1),
        run("feedback-float-small", threshold=1e-5, reward=small_score),
        run("integer-threshold-above-float", n=2, threshold=2**200 + 1, reward=large_score),
        run("negative-integer-threshold", n=2, threshold=-(2**200 + 1), reward=negative_large_score),
    ]
    assert rows[-2]["outcomes"] == [{"answer": "8"}]
    assert rows[-1]["outcomes"] == [{"answer": "7"}]
    feedback = next(call for call in rows[-2]["calls"] if call["kind"] == "feedback")
    assert feedback["inputs"]["target_threshold"] == '"<non-serializable: int>"'
    masks = {
        "reasoning": recursive_mask({"nested": [dspy.Reasoning(content="reason")]}),
        "code": recursive_mask({"nested": [dspy.Code(code="x = 1")]}),
        "model": recursive_mask({"nested": [QA(question="q", answer="a")]}),
        "large_integer": recursive_mask({"large": 1 << 128}),
    }
    output = {"dspy_version": dspy.__version__, "cases": rows, "masks": masks}
    encoded = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    print(json.dumps({"sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                      "output": output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
