"""Run the live suite against a real model with DSPy 3.4.1 and write the
transcript (every request and real completion) for DSZig to replay.

    OPENAI_API_KEY=... uv run --locked live.py [out.json] [model]"""

import json
import pathlib
import sys

import live_suite

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "live" / "dspy_gpt-4o-mini.json"


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-4o-mini"
    lm = live_suite.RecordingLM(model)
    result = live_suite.run_suite(lm)
    transcript = {"runtime": "dspy 3.4.1", "model": model, "calls": lm.calls, **result}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scores = "; ".join(f"{p} {result[p]['baseline']}% -> {result[p]['optimized']}%" for p in ("rag", "dates", "dates_search", "dates_mipro", "dates_mipro_aware", "dates_copro"))
    print(f"wrote {out}: {len(lm.calls)} real calls; {scores}", file=sys.stderr)


if __name__ == "__main__":
    main()
