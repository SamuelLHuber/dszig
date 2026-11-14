"""Run the live suite against a real model with DSPy 3.5.0 and write the
transcript (every request and real completion) for DSZig to replay.

    OPENAI_API_KEY=... uv run --locked live.py [out.json] [model]"""

import json
import os
import pathlib
import sys

import dspy

import live_suite

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "live" / "dspy_3.5.0_gpt-4o-mini.json"


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(os.environ.get("DSZIG_LIVE_TRANSCRIPT", DEFAULT_OUT))
    model = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DSZIG_LIVE_MODEL", "openai/gpt-4o-mini")
    lm = live_suite.RecordingLM(model, model_type=os.environ.get("DSZIG_LIVE_API", "auto"))
    result = live_suite.run_suite(lm)
    transcript = {"target_dspy": dspy.__version__, "runtime": f"dspy {dspy.__version__}", "model": model,
                  "supports_reasoning": lm.supports_reasoning,
                  "supports_response_format": "response_format" in lm.supported_params,
                  "supports_response_schema": lm.supports_response_schema,
                  "supports_function_calling": lm.supports_function_calling,
                  "supported_params": sorted(lm.supported_params),
                  "calls": lm.calls, **result}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scores = "; ".join(f"{p} {result[p]['baseline']}% -> {result[p]['optimized']}%" for p in ("rag", "dates", "dates_search", "dates_mipro", "dates_mipro_aware", "dates_copro", "rag_react", "dates_simba", "dates_gepa"))
    print(f"wrote {out}: {len(lm.calls)} real calls; {scores}", file=sys.stderr)


if __name__ == "__main__":
    main()
