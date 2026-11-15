"""DSPy 3.5.0 ParallelExecutor reference cases (dspy/utils/parallelizer.py).

Each case maps `i -> i * i`, raising for `i % fail_mod == fail_mod - 1`. Records
ordered results (None for failures), failed indices and whether the executor
cancelled after `max_errors` failures. Multi-thread cancellation depends on
scheduling, so threaded cases only use limits that are never reached.
"""

import json
import logging
import pathlib
import sys

from dspy.utils.parallelizer import ParallelExecutor

CASES = [
    {"threads": 1, "max_errors": 10, "n": 10, "fail_mod": 3},
    {"threads": 1, "max_errors": 2, "n": 10, "fail_mod": 3},
    {"threads": 1, "max_errors": 1, "n": 4, "fail_mod": 1},
    {"threads": 1, "max_errors": 5, "n": 0, "fail_mod": 2},
    {"threads": 4, "max_errors": 100, "n": 37, "fail_mod": 5},
    {"threads": 8, "max_errors": 10, "n": 3, "fail_mod": 1000},
]


def run(case):
    def square(i):
        if i % case["fail_mod"] == case["fail_mod"] - 1:
            raise ValueError(f"item {i}")
        return i * i

    executor = ParallelExecutor(num_threads=case["threads"], max_errors=case["max_errors"], disable_progress_bar=True)
    try:
        results = executor.execute(square, list(range(case["n"])))
        cancelled = False
    except Exception as e:
        assert str(e) == "Execution cancelled due to errors or interruption.", e
        results = None
        cancelled = True
    return {**case, "results": results, "failed": sorted(executor.failed_indices), "cancelled": cancelled}


def main():
    logging.disable(logging.CRITICAL)
    out = pathlib.Path(sys.argv[1])
    data = {"dspy": __import__("dspy").__version__, "cases": [run(c) for c in CASES]}
    (out / "parallel.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
