"""Golden fixture for dspy.majority and normalize_text (DSPy 3.5.0).

Usage: uv run gen_ensemble.py [out_dir]  (default: ../../tests/fixtures)"""

import json
import pathlib
import random
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
    "Café ÉCOLE naïve résumé",
    "ΟΔΥΣΣΕΥΣ ΣΑ Σ ΑΣ'Σ ὈΔΥΣΣΕΎΣ",
    "İstanbul Straße ﬁne Ǆemal",
    "The\u00a0Ａ\u2003the—an—theme a…b",
    "한국어 THE Ⅻ x\u0301\u0323 Å Ω",
    "ΑΣ\u00ad ΑΣ\u0301 Σ\u0301 ΑΣ\u0345Α",
]


def random_strings():
    """Seeded random strings over every code point, weighted to tricky ones."""
    rng = random.Random(0)
    tricky = "ΣσςΑα\u0300\u0301\u0323\u0345\u00ad\u02bc'İıÅÉ한ﬁ \u00a0\u2003\u3000aAnNtThHeE_-.,"
    out = []
    for _ in range(400):
        chars = []
        for _ in range(rng.randint(1, 10)):
            if rng.random() < 0.6:
                chars.append(rng.choice(tricky))
            else:
                cp = rng.randrange(0x110000)
                chars.append(chr(cp if not 0xD800 <= cp <= 0xDFFF else 0x20))
        out.append("".join(chars))
    return out

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
    data = {"normalize": [[s, normalize_text(s)] for s in NORMALIZE + random_strings()], "majority": majority}
    (out / "ensemble.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
