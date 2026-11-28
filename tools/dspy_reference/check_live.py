"""Replay a live transcript (recorded by DSPy's live.py or by DSZig's
`zig build live`) through the locked DSPy runtime. Passes only if DSPy sends exactly the
recorded requests, consumes every recorded call, and reproduces the recorded
case outputs, scores and saved artifact byte for byte.

    uv run --locked check_live.py transcript.json"""

import json
import sys

import live_suite
from tqdm import tqdm

PIPELINES = ("rag", "dates", "dates_search", "dates_mipro", "dates_mipro_aware", "dates_copro", "rag_react", "dates_simba", "dates_gepa")


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        transcript = json.load(f)
    if transcript.get("target_dspy") != live_suite.dspy.__version__:
        sys.exit(f"transcript targets {transcript.get('target_dspy', transcript.get('runtime'))!r}, "
                 f"not locked DSPy {live_suite.dspy.__version__}; historical evidence cannot be relabeled")
    sampling = live_suite.settings(transcript)
    lm = live_suite.ReplayLM(transcript["calls"], supports_reasoning=transcript["supports_reasoning"],
                            capabilities=transcript, temperature=sampling["temperature"])
    result = live_suite.run_suite(lm, copro_init_temperature=sampling["copro_init_temperature"])
    failures = []
    if lm.misses:
        failures.append(f"{len(lm.misses)} requests had no recorded twin")
    if not all(lm.used):
        failures.append(f"{lm.used.count(False)} recorded calls were never requested")
    if len(transcript["cases"]) != len(result["cases"]):
        failures.append(f"expected {len(transcript['cases'])} adapter cases, got {len(result['cases'])}")
    for want, got in zip(transcript["cases"], result["cases"]):
        # Error types are runtime-specific names; only the outcome must agree.
        if (want["name"] != got["name"]
                or ("error" in want) != ("error" in got)
                or ("outputs" in want) != ("outputs" in got)
                or want.get("outputs") != got.get("outputs")):
            failures.append(f"case {want['name']}: recorded {want} != replayed {got}")
    for pipeline in PIPELINES:
        for key in ("baseline", "optimized", "artifact", "candidates", "instructions", "trials", "scores", "train_scores"):
            if transcript[pipeline][key] != result[pipeline][key]:
                failures.append(f"{pipeline} {key} differs")
    if failures:
        sys.stderr.write("\n".join(failures) + "\n")
        sys.exit(1)
    scores = "; ".join(
        f"{p} {result[p]['baseline']}% -> {result[p]['optimized']}% ({len(result[p]['artifact'])}-byte artifact)" for p in PIPELINES
    )
    print(f"ok: DSPy {live_suite.dspy.__version__} replayed {len(lm.calls)} {transcript['runtime']} calls ({transcript['model']}) identically; {scores}")


if __name__ == "__main__":
    # Progress bars are useful, but their daemon monitor is not needed for
    # replay. Avoid leaving it waiting on Python locks during finalization.
    tqdm.monitor_interval = 0
    main()
