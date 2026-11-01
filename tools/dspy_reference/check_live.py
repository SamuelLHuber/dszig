"""Replay a live transcript (recorded by DSPy's live.py or by DSZig's
`zig build live`) through DSPy 3.4.1. Passes only if DSPy sends exactly the
recorded requests, consumes every recorded call, and reproduces the recorded
case outputs, scores and saved artifact byte for byte.

    uv run --locked check_live.py transcript.json"""

import json
import sys

import live_suite


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        transcript = json.load(f)
    lm = live_suite.ReplayLM(transcript["calls"])
    result = live_suite.run_suite(lm)
    failures = []
    if lm.misses:
        failures.append(f"{len(lm.misses)} requests had no recorded twin")
    if not all(lm.used):
        failures.append(f"{lm.used.count(False)} recorded calls were never requested")
    for want, got in zip(transcript["cases"], result["cases"], strict=True):
        # Error types are runtime-specific names; only the outcome must agree.
        if ("error" in want) != ("error" in got) or want.get("outputs") != got.get("outputs"):
            failures.append(f"case {want['name']}: recorded {want} != replayed {got}")
    for key in ("baseline", "optimized", "artifact"):
        if transcript["rag"][key] != result["rag"][key]:
            failures.append(f"rag {key} differs")
    if failures:
        sys.stderr.write("\n".join(failures) + "\n")
        sys.exit(1)
    rag = result["rag"]
    print(
        f"ok: DSPy 3.4.1 replayed {len(lm.calls)} {transcript['runtime']} calls ({transcript['model']}) identically; "
        f"baseline {rag['baseline']}%, optimized {rag['optimized']}%, artifact {len(rag['artifact'])} bytes"
    )


if __name__ == "__main__":
    main()
