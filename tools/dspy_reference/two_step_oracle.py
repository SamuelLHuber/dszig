"""Record installed DSPy 3.5 TwoStepAdapter behavior without network.

The source adapter, Predict, candidate fanout and extraction fallback all run
unchanged. Run with the reference virtualenv's Python interpreter.
"""

import hashlib
import inspect
import json
import warnings

import dspy
import pydantic
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.utils import serialize_for_json
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.utils.callback import BaseCallback

import reference as ref


class QA(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Count(dspy.Signature):
    """Count the items."""
    question: str = dspy.InputField()
    count: int = dspy.OutputField()
    flags: list[bool] = dspy.OutputField()


class Collision(dspy.Signature):
    question: str = dspy.InputField()
    text: str = dspy.OutputField()
    answer: str = dspy.OutputField()


class CollisionLast(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    text: str = dspy.OutputField()


class Thought(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


class Payload(pydantic.BaseModel):
    name: str
    count: int


class Formatted(dspy.Signature):
    """Preserve Python string formatting."""
    model: Payload = dspy.InputField()
    code: dspy.Code = dspy.InputField()
    reasoning: Reasoning = dspy.InputField()
    models: list[Payload] = dspy.InputField()
    codes: list[dspy.Code] = dspy.InputField()
    reasons: list[Reasoning] = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


class Capture(BaseCallback):
    def __init__(self):
        self.signature = None

    def on_adapter_format_start(self, call_id, instance, inputs):
        self.signature = inputs["signature"]


class RecordingLM(ref.ReferenceLM):
    def __init__(self, kind, completions, capture, calls):
        super().__init__(model=f"mock/{kind}", temperature=0.25 if kind == "main" else 0.125,
                         max_tokens=1000, cache=False)
        self.kind, self.completions = kind, list(completions)
        self.capture, self.calls = capture, calls

    def complete(self, request):
        signature = self.capture.signature
        self.calls.append({
            "kind": self.kind,
            "messages": ref.messages_for(request),
            "temperature": request.config.temperature,
            "rollout_id": ref.rollout_id(self),
            "instructions": signature.instructions,
            "inputs": list(signature.input_fields),
            "outputs": list(signature.output_fields),
        })
        text = self.completions.pop(0)
        self.calls[-1]["completion"] = text
        return ref.text_response(self.model, text)


def run_case(name, signature, main, extraction, *, n=1, demos=(), inputs=None):
    calls, capture, trace = [], Capture(), []
    main_lm = RecordingLM("main", main, capture, calls)
    extraction_lm = RecordingLM("extract", extraction, capture, calls)
    adapter = TwoStepAdapter(extraction_lm)
    program = dspy.Predict(signature, n=n, temperature=0.75, rollout_id=23)
    program.demos = list(demos)
    with dspy.context(lm=main_lm, adapter=adapter, trace=trace, callbacks=[capture]):
        try:
            prediction = program(**(inputs or {"question": "What?"}))
            result = {"ok": True, "prediction": prediction.toDict(),
                      "completions": serialize_for_json(dict(prediction.completions.items()))}
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__}
            if result["error"] != "AdapterParseError":
                raise
    return {"name": name, "signature": signature.__name__, "n": n,
            "demos": serialize_for_json(list(demos)),
            "inputs": serialize_for_json(inputs or {"question": "What?"}),
            "calls": calls, "result": result,
            "trace": [{"inputs": row[1], "outputs": row[2].toDict()} for row in trace]}


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cases = [
            run_case("basic", QA, ["Paris is the answer."], ["[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"]),
            run_case("empty_main", QA, [""], []),
            run_case("candidates", QA, ["first", "second"], ["[[ ## answer ## ]]\none", "[[ ## answer ## ]]\ntwo"], n=2),
            run_case("fallback", Count, ["There are two items."], ["not structured", '{"count":2,"flags":[true,false]}']),
            run_case("failure", QA, ["main text"], ["bad chat", "bad json"]),
            run_case("text_collision", Collision, ["main text"], ["[[ ## text ## ]]\nverbatim\n\n[[ ## answer ## ]]\nyes"]),
            run_case("text_collision_last", CollisionLast, ["main text"], ["[[ ## text ## ]]\nverbatim\n\n[[ ## answer ## ]]\nyes"]),
            run_case("empty_extraction", QA, ["main text"], ["", '{"answer":"fallback"}']),
            run_case("demos", QA, ["main"], ["[[ ## answer ## ]]\nyes"], demos=[
                {"question": "incomplete", "answer": None}, {"question": "complete", "answer": "known"}, {"answer": "only"}]),
            run_case("reasoning", Thought, ["freeform"], ["[[ ## reasoning ## ]]\nwhy\n\n[[ ## answer ## ]]\nyes"]),
        ]
        payload = Payload(name="O'Reilly\n", count=3)
        code, reason = dspy.Code(code="x = 1"), Reasoning(content="why")
        inputs = {"model": payload, "code": code, "reasoning": reason,
                  "models": [payload], "codes": [code], "reasons": [reason],
                  "history": dspy.History(messages=[{"question": "old", "answer": "past"}])}
        adapter = TwoStepAdapter(ref.KeywordLM([], "unused"))
        formatting = {"inputs": serialize_for_json(inputs),
                      "messages": ref.messages_for(adapter.format(Formatted, [], inputs)),
                      "str": {k: str(v) for k, v in inputs.items()},
                      "repr": {k: repr(v) for k, v in inputs.items()}}
        source = inspect.getsourcefile(TwoStepAdapter)
        print(json.dumps({"dspy_version": dspy.__version__,
                          "source_sha256": hashlib.sha256(open(source, "rb").read()).hexdigest(),
                          "cases": cases, "formatting": formatting}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
