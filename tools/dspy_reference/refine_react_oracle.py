"""Run actual DSPy Refine(ReAct), retaining outputs, feedback and LM prompts."""

import contextlib
import copy
import io
import json

import dspy
from dspy.adapters.utils import serialize_for_json

import reference as ref
import refine_oracle as refine


class RecordingLM(ref.ReferenceLM):
    def __init__(self, capture):
        super().__init__(model="mock/refine-react", temperature=0.25,
                         rollout_id=7, max_tokens=1000, cache=False)
        self.capture, self.feedback, self.calls = capture, [], []
        self.advice = {}

    def complete(self, request):
        self.calls.append({"messages": ref.messages_for(request),
                           "temperature": request.config.temperature,
                           "rollout_id": ref.rollout_id(self),
                           "hint": self.capture.local.inputs.get("hint_")})
        if "advice" in self.capture.local.signature.output_fields:
            self.feedback.append(dict(self.capture.local.inputs))
            text = '<discussion>Improve.</discussion><advice>' + json.dumps(self.advice) + '</advice>'
        else:
            answer = str(ref.rollout_id(self))
            text = '<next_thought>ready</next_thought><tool_calls>' + json.dumps({
                "tool_calls": [{"name": "submit", "args": {"answer": answer}}]
            }) + '</tool_calls>'
        return ref.text_response(self.model, text)


class AgentPair(dspy.Module):
    def __init__(self):
        super().__init__()
        self.first = dspy.ReAct(refine.QA, tools=[])
        self.last = dspy.ReAct(refine.QA, tools=[])

    def forward(self, question):
        first = self.first(question=question)
        return self.last(question=first.answer)


def independent_agents():
    capture = refine.Capture()
    lm = RecordingLM(capture)
    lm.advice = {"first.react": "first-specific", "last.react": "last-specific"}
    module = AgentPair()
    assert module.first.react.signature is not module.last.react.signature
    assert copy.deepcopy(module).first.react.signature is module.first.react.signature
    wrapper = dspy.Refine(module, 2, refine.score, 100)
    trace, diagnostics = [], io.StringIO()
    with contextlib.redirect_stdout(diagnostics), dspy.context(
        lm=lm, adapter=dspy.XMLAdapter(), callbacks=[capture], trace=trace
    ):
        prediction = wrapper(question="q")
    assert [call["hint"] for call in lm.calls] == [
        None, None, None, "first-specific", "last-specific"
    ]
    assert prediction.answer == "8"
    return {"prediction": prediction.toDict(), "feedback": lm.feedback,
            "calls": lm.calls}


def main():
    capture = refine.Capture()
    lm = RecordingLM(capture)
    dspy.configure(lm=lm)
    wrapper = dspy.Refine(dspy.ReAct(refine.QA, tools=[]), 2, refine.score, 100)
    trace, diagnostics = [], io.StringIO()
    with contextlib.redirect_stdout(diagnostics), dspy.context(
        lm=lm, adapter=dspy.XMLAdapter(), callbacks=[capture], trace=trace
    ):
        prediction = wrapper(question="q")
    trajectory = json.loads(lm.feedback[0]["program_trajectory"])
    inputs, outputs = trajectory[0]["inputs"], trajectory[0]["outputs"]
    assert inputs["history"] == "<non-serializable: History>"
    assert inputs["tools"] == ["<non-serializable: Tool>"]
    assert outputs["tool_calls"] == "<non-serializable: ToolCalls>"
    assert outputs["next_thought"] == "ready"
    assert prediction.answer == "8"
    print(json.dumps({"dspy_version": dspy.__version__,
                      "prediction": prediction.toDict(), "feedback": lm.feedback,
                      "calls": lm.calls,
                      "independent_agents": independent_agents(),
                      "trace": [{"inputs": serialize_for_json(inputs), "outputs": outputs.toDict()}
                                for _, inputs, outputs in trace]}, indent=2))


if __name__ == "__main__":
    main()
