"""Golden fixture for dspy.majority and normalize_text (DSPy 3.4.1).

Usage: uv run gen_ensemble.py [out_dir]  (default: ../../tests/fixtures)"""

import json
import pathlib
import sys

import dspy
from dspy.evaluate import normalize_text

NORMALIZE = [
    "The,  Eiffel  Tower!",
    "an apple a day",
    "A",
    "",
    "!!!",
    "Tab\tand\nnewline\x0bvt",
    "the_the the-the",
    "x-ray's (an) anthem theater",
    "  The  ",
    "ANd THE An",
]

MAJORITY = [
    ["Paris", "paris.", "Berlin"],
    ["a1", "b1", "B1!", "A1."],
    ["", "!!", "x"],
    ["", "!!"],
    ["The cat", "cat", "a dog", "dog", "Dog!"],
    ["only"],
]


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    majority = []
    for values in MAJORITY:
        winner = dspy.majority([dspy.Prediction(reasoning="r", answer=v) for v in values])
        majority.append({"values": values, "answer": winner.answer})
    data = {"normalize": [[s, normalize_text(s)] for s in NORMALIZE], "majority": majority}
    (out / "ensemble.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
