"""Time and measure DSPy 3.5.0 replaying a full-suite transcript, for
comparison with DSZig's examples/bench.zig on the same transcript.

Both runtimes run the identical suite (eight adapter cases, nine optimizer
pipelines) against a replay LM, so no network or model latency is included:
the numbers are framework overhead only (prompt formatting, parsing,
optimizer logic, evaluation, serialization, and the replay lookup itself).
A timestamp is taken as each LM call starts; pipeline boundaries are the
call indices at which each pipeline begins.

    uv run --locked bench.py transcript.json [reps] > dspy_bench.json"""

import contextlib
import json
import logging
import os
import resource
import sys
import time
import tracemalloc

t_import = time.perf_counter()
import dspy  # noqa: E402

import live_suite  # noqa: E402

import_seconds = time.perf_counter() - t_import

PIPELINES = ("rag", "dates", "dates_search", "dates_mipro", "dates_mipro_aware", "rag_react", "dates_copro",
             "dates_simba", "dates_gepa")
COMPARED = ("baseline", "optimized", "artifact", "candidates", "instructions", "trials", "scores", "train_scores")


class TimedReplayLM(live_suite.ReplayLM):
    def __init__(self, *args, clock_start, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock_start = clock_start
        self.stamps = []

    def complete(self, request):
        self.stamps.append(time.perf_counter() - self.clock_start)
        return super().complete(request)


def run_once(transcript, boundaries):
    start = time.perf_counter()
    sampling = live_suite.settings(transcript)
    lm = TimedReplayLM(transcript["calls"], supports_reasoning=transcript["supports_reasoning"],
                       capabilities=transcript, temperature=sampling["temperature"], clock_start=start)
    original = live_suite.pipeline

    def timed_pipeline(*args, **kwargs):
        boundaries.append(len(lm.stamps))
        return original(*args, **kwargs)

    live_suite.pipeline = timed_pipeline
    try:
        result = live_suite.run_suite(lm, copro_init_temperature=sampling["copro_init_temperature"])
    finally:
        live_suite.pipeline = original
    total = time.perf_counter() - start
    if lm.misses or not all(lm.used):
        sys.exit(f"replay diverged: {len(lm.misses)} misses, {lm.used.count(False)} unused calls")
    for want, got in zip(transcript["cases"], result["cases"], strict=True):
        if ("error" in want) != ("error" in got) or want.get("outputs") != got.get("outputs"):
            sys.exit(f"case {want['name']} differs")
    for p in PIPELINES:
        for key in COMPARED:
            if transcript[p][key] != result[p][key]:
                sys.exit(f"{p} {key} differs")
    return total, lm.stamps, result


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    logging.disable(logging.CRITICAL)
    with open(path, encoding="utf-8") as f:
        transcript = json.load(f)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    totals, stamps, boundaries = [], [], []
    # DSPy's optimizers print progress to stdout; keep it (as shipped) but off the JSON.
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        for _ in range(reps):
            b = []
            total, s, result = run_once(transcript, b)
            totals.append(total)
            stamps.append(s)
            boundaries = b
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tracemalloc.start()
        run_once(transcript, [])
        _, heap_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    json.dump({
        "runtime": f"dspy {dspy.__version__}",
        "python": sys.version.split()[0],
        "transcript": path,
        "calls": len(transcript["calls"]),
        "reps": reps,
        "import_seconds": import_seconds,
        "total_seconds": totals,
        "call_seconds": stamps,
        "pipeline_starts": dict(zip(PIPELINES, boundaries, strict=True)),
        "scores": {p: [result[p]["baseline"], result[p]["optimized"]] for p in PIPELINES},
        "maxrss_kb_after_import": rss_before,
        "maxrss_kb": rss_after,
        "heap_peak_bytes": heap_peak,
    }, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
