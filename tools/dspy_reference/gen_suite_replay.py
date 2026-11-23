"""Record the complete real-DSPy suite with an explicitly deterministic mock LM.

This proves only the recorded deterministic cross-runtime cases, not live-model
or complete public-API parity. Usage: uv run --locked gen_suite_replay.py [out_dir]
"""

import json
import pathlib
import sys
import threading

import dspy
from dspy.utils.callback import BaseCallback

import live_suite
import reference as ref

ADVICE = "Answer with an ISO date."


class CapturePrompt(BaseCallback):
    """Capture the real adapter's processed signature, not a duplicate parser."""

    def __init__(self):
        self.local = threading.local()

    def on_adapter_format_start(self, call_id, instance, inputs):
        self.local.adapter = instance
        self.local.signature = inputs["signature"]
        self.local.inputs = inputs["inputs"]


class SuiteLM(ref.ReferenceLM):
    def __init__(self, capture):
        super().__init__(model="mock/reference-suite", model_type="chat", temperature=0.0, cache=False)
        self.capture = capture
        self.calls = []
        self.proposals = {"instruction": 0, "reflection": 0}
        self.golds = {q: a for split in ("train", "dev") for q, a in live_suite.DATES[split]}
        self.golds.update({row["question"]: row["answer"] for row in ref.TRAINSET})

    def complete(self, request):
        state = self.capture.local
        signature, inputs = state.signature, state.inputs
        question = inputs.get("question")
        history = inputs.get("history")
        if question is None and history is not None:
            question = next((event["question"] for event in history.messages if "question" in event), None)
        fields = {}
        for name in signature.output_fields:
            if name in ("plan", "reasoning", "next_thought"):
                value = "Use the supplied evidence."
            elif name == "answer":
                # Instruction optimization can improve this deliberately imperfect model.
                good = ADVICE in signature.instructions or question is not None and len(question) % 3 == 0
                value = self.golds.get(question, "Paris") if good else "unknown"
            elif name == "tool_calls":
                if history is None or not history.messages:
                    calls = [{"name": "corpus_search", "args": {"query": question}}]
                else:
                    calls = [{"name": "submit", "args": {"answer": self.golds.get(question, "unknown")}}]
                value = dspy.ToolCalls.from_dict_list(calls)
            elif name == "proposed_instruction":
                index = self.proposals["instruction"]
                self.proposals["instruction"] += 1
                value = ADVICE + (" Be precise." if index % 2 else " Use the calendar.")
            elif name == "proposed_prefix_for_output_field":
                value = "Answer:"
            elif name == "new_instruction":
                index = self.proposals["reflection"]
                self.proposals["reflection"] += 1
                value = ["Be brief.", ADVICE, "Answer politely.", ADVICE + " Never guess."][index % 4]
            elif name == "module_advice":
                value = {"generate.predict": ADVICE}
            elif name == "discussion":
                value = "The worse run guessed."
            elif name == "observations":
                value = "COMPLETE: Questions ask for historical dates."
            elif name in ("summary", "program_description", "module_description"):
                value = "Answer questions with historical dates."
            elif name == "flag":
                value = True
            elif name == "count":
                value = 42
            elif name == "ratio":
                value = 0.25
            elif name == "label":
                value = "pos"
            elif name == "items":
                value = ["one", "two"]
            else:
                raise ValueError(f"Unscripted real DSPy output field: {name}")
            fields[name] = value
        text = state.adapter.format_assistant_message_content(signature, fields)
        response = ref.text_response(self.model, text)
        self.calls.append(live_suite.transcript_call(request, response, ref.rollout_id(self)))
        return response


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    capture = CapturePrompt()
    lm = SuiteLM(capture)
    dspy.configure(callbacks=[capture])
    result = live_suite.run_suite(lm)
    transcript = {"target_dspy": dspy.__version__, "runtime": f"dspy {dspy.__version__} deterministic mock",
                  "model": lm.model, "supports_reasoning": lm.supports_reasoning,
                  "supports_response_format": "response_format" in lm.supported_params,
                  "supports_response_schema": lm.supports_response_schema,
                  "supports_function_calling": lm.supports_function_calling,
                  "records_tool_result_names": True,
                  "calls": lm.calls, **result}
    (out / "dspy_3_5_0_mock_suite.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
