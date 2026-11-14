"""Golden ReAct evidence from real DSPy 3.5.0 with a scripted text engine.

Records actual requests, typed history/tool IDs/results, final outputs and
termination reasons. Usage: uv run --locked gen_react.py [out_dir]
"""

import json
import pathlib
import sys

import dspy

import reference as ref


def search(query: str) -> list[str]:
    """Search the corpus.
    Returns passages."""
    return ["Apollo 11 landed on the Moon in 1969.", "multi\nline «x»"]


def add(a: int, b: int = 2) -> int:
    return a + b


def turn(thought, calls):
    return (f"<next_thought>\n{thought}\n</next_thought>\n\n<tool_calls>\n"
            + json.dumps({"tool_calls": calls}, ensure_ascii=False) + "\n</tool_calls>")


def call(name, args, id=None):
    return {"name": name, "args": args, **({"id": id} if id is not None else {})}


SUBMIT = call("submit", {"answer": "1969"})
CASES = [
    ("react_tools", [search, add], 5, [
        turn("Search first", [call("search", {"query": "moon landing"})]),
        turn("Add one", [call("add", {"a": 1969, "b": 1})]),
        turn("Done", [SUBMIT]),
    ]),
    ("react_max_iters", [search], 1, [
        turn("Search", [call("search", {"query": "moon"})]),
        turn("Out of steps", [call("search", {"query": "ignored in forced submit"}), SUBMIT]),
    ]),
    ("react_batched", [search, add], 3, [
        turn("Search and calculate", [call("search", {"query": "moon"}, "search-id"),
                                      call("add", {"a": 1968, "b": 1}, "add-id")]),
        turn("Done", [call("submit", {"answer": "1969"}, "submit-id")]),
    ]),
    ("react_empty_tool_calls", [search], 3, [turn("No call", []), turn("Submit now", [SUBMIT])]),
    # XMLAdapter's real JSON fallback also fails before ReAct forces submit.
    ("react_parse_error", [search], 3, ["not XML", "not JSON", turn("Recover", [SUBMIT])]),
    ("react_unknown_tool", [search], 3, [turn("Unknown", [call("missing", {})]), turn("Done", [SUBMIT])]),
    ("react_zero_iters", [search], 0, [turn("Submit immediately", [SUBMIT])]),
]


class ScriptedLM(ref.ReferenceLM):
    def __init__(self, answers):
        super().__init__(model="mock/react-3.5", model_type="chat", temperature=0.0, cache=False)
        self.answers = iter(answers)
        self.calls = []

    def complete(self, request):
        text = next(self.answers)
        self.calls.append({"messages": ref.messages_for(request), "completion": text})
        return ref.text_response(self.model, text)


def run(name, tools, max_iters, answers):
    lm = ScriptedLM(answers)
    dspy.configure(lm=lm, adapter=None)
    agent = dspy.ReAct("question -> answer", tools=tools, max_iters=max_iters)
    pred = agent(question="When did Apollo 11 land?")
    # Do not permit a successful final answer to hide unused scripted calls.
    if next(lm.answers, None) is not None:
        raise AssertionError(f"{name}: unconsumed scripted completion")
    return {
        "name": name,
        "max_iters": max_iters,
        "tools": [tool.__name__ for tool in tools],
        "calls": lm.calls,
        "history": pred.history.model_dump(mode="json"),
        "termination_reason": pred.termination_reason,
        "outputs": ref.serialize_object({k: v for k, v in pred.items() if k not in ("history", "termination_reason")}),
    }


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    cases = [run(*case) for case in CASES]
    (out / "react.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
