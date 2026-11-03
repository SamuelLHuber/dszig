"""Golden ReAct fixture from DSPy 3.4.1: every prompt of scripted agent runs.
Tool errors are left out: DSPy puts a Python traceback in the observation.

Usage: uv run gen_react.py [out_dir]  (default: ../../tests/fixtures)"""

import json
import pathlib
import sys

import dspy
from dspy.utils.dummies import DummyLM


def search(query: str) -> list[str]:
    """Search the corpus.
    Returns passages."""
    return ["Apollo 11 landed on the Moon in 1969.", "multi\nline «x»"]


def add(a: int, b: int = 2) -> int:
    return a + b


CASES = [
    (
        "react_tools",
        [search, add],
        5,
        [
            {"next_thought": "Search first", "next_tool_name": "search", "next_tool_args": '{"query": "moon landing"}'},
            {"next_thought": "Add one", "next_tool_name": "add", "next_tool_args": '{"a": 1969, "b": 1}'},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": "{}"},
            {"reasoning": "Found 1969", "answer": "1969"},
        ],
    ),
    (
        "react_max_iters",
        [search],
        1,
        [
            {"next_thought": "Search", "next_tool_name": "search", "next_tool_args": '{"query": "moon"}'},
            {"reasoning": "Out of steps", "answer": "1969"},
        ],
    ),
]


def run(name, tools, max_iters, answers):
    lm = DummyLM(answers)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    agent = dspy.ReAct("question -> answer", tools=tools, max_iters=max_iters)
    pred = agent(question="When did Apollo 11 land?")
    trajectory = pred.trajectory
    return {
        "name": name,
        "max_iters": max_iters,
        "calls": [{"messages": h["messages"], "completion": h["outputs"][0]} for h in lm.history],
        "trajectory": trajectory,
        "outputs": {k: v for k, v in pred.items() if k != "trajectory"},
    }


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    cases = [run(*c) for c in CASES]
    (out / "react.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
