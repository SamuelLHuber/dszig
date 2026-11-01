"""Generate tests/fixtures/pickle.json: BootstrapFewShot's multi-hop trace pick.

For several same-predictor demo lists, records `Hasher.hash(tuple(demos))`
(sha256 of `pickle.dumps`) and which demo DSPy 3.4.1 keeps:

    rng = random.Random(Hasher.hash(tuple(demos)))
    demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]

Equal value strings are made the same object (as values flowing between hops
are), and so are equal field names, but a value never shares an object with a
field name: the identity model DSZig assumes (one-character strings are
CPython singletons either way). Run: uv run --locked gen_pickle.py"""

import json
import pathlib
import random
import sys

import dspy
from dspy.utils.hasher import Hasher

OUT = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "pickle.json"


def canonical(v, cache):
    if isinstance(v, str):
        return cache.setdefault(v, v)
    if isinstance(v, list):
        return [canonical(x, cache) for x in v]
    if isinstance(v, dict):
        return {canonical(k, cache): canonical(x, cache) for k, x in v.items()}
    return v


def pick(demos):
    keys, values = {}, {}
    examples = tuple(
        dspy.Example(augmented=True, **{keys.setdefault(k, k): canonical(v, values) for k, v in d.items()}) for d in demos
    )
    digest = Hasher.hash(examples)
    rng = random.Random(digest)
    index = rng._randbelow(len(demos) - 1) if rng.random() < 0.5 else len(demos) - 1
    return digest, index


def cases():
    """Each demo list round-trips through JSON, so its strings are fresh
    objects, as parsed LM outputs are, not interned source constants."""
    for demos in literal_cases():
        yield json.loads(json.dumps(demos))


def literal_cases():
    yield [{"question": "Q1?", "reasoning": "r", "answer": "a"}, {"question": "Q1?", "reasoning": "r2", "answer": "b"}]
    yield [{"question": "answer", "a": "a", "answer": "question"}, {"question": "", "a": "é", "answer": "answer"}]
    yield [
        {"context": ["Paris is in France.", "Ünïcödé — ✓ 日本"], "question": "Where?", "n": 3, "big": 2**40, "neg": -70000,
         "small": 255, "mid": 65535, "f": 0.1, "ok": False, "none": None, "nested": {"k": "v"}, "answer": "x" * 300},
        {"context": [], "question": "Where?", "n": -1, "big": -(2**63), "neg": 256, "small": 0, "mid": 65536, "f": -2.5e300,
         "ok": True, "none": None, "nested": {}, "answer": "Paris is in France."},
        {"context": ["one"], "question": "Where?", "n": 2**31 - 1, "big": 2**31, "neg": -(2**31), "small": 1, "mid": -1,
         "f": 1e-310, "ok": True, "none": None, "nested": {"a": 1, "b": [1, 2]}, "answer": ""},
    ]
    yield [{"question": f"q{i % 7}", "passage": f"passage number {i}", "answer": f"a{i}"} for i in range(150)]
    yield [{"question": "long", "context": [f"{i:04d} " + "lorem ipsum " * 80 for i in range(120)], "answer": "y" * 70000},
           {"question": "long", "context": ["short"], "answer": "z"}]
    rng = random.Random(1234)
    for n in range(20):
        size = 2 + n % 5
        yield [{"question": f"hop question {n}", "query": f"query {n}.{j}", "answer": str(rng.random())} for j in range(size)]


def main():
    path = pathlib.Path(sys.argv[1]) / "pickle.json" if len(sys.argv) > 1 else OUT
    out = []
    for demos in cases():
        digest, index = pick(demos)
        out.append({"demos": demos, "sha256": digest, "pick": index})
    path.write_text(json.dumps({"dspy": dspy.__version__, "cases": out}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(out)} cases, picks {[c['pick'] for c in out]}")


if __name__ == "__main__":
    main()
