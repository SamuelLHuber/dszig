"""Write DSPy 3.4.1's `get_dspy_source_code(DateQA())`, the program code the
MIPROv2 program-aware proposer sends, and `inspect.getsource(DateQA)`, which
SIMBA's rule writer sends; DSZig embeds both in examples/live_suite.zig.

    uv run --locked gen_program_code.py [out_dir]"""

import contextlib
import inspect
import io
import pathlib
import sys

from dspy.propose.utils import get_dspy_source_code

import live_suite

OUT = pathlib.Path(__file__).resolve().parents[2] / "examples"


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        code = get_dspy_source_code(live_suite.DateQA())
    (out / "dates_program_code.txt").write_text(code, encoding="utf-8")
    (out / "dates_class_source.txt").write_text(inspect.getsource(live_suite.DateQA), encoding="utf-8")
if __name__ == "__main__":
    main()
