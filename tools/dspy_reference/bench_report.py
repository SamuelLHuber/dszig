"""Compare examples/bench.zig and bench.py results on the same transcript.

    python3 bench_report.py dszig_bench.json dspy_bench.json

Prints a Markdown table of medians per suite segment (adapter cases, then
each optimizer pipeline), throughput, per-call overhead percentiles and
memory. Segments are delimited by the LM-call index at which DSPy begins
each pipeline; both runtimes make the identical call sequence."""

import json
import statistics
import sys


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def segments(run, starts, total):
    names = list(starts)
    bounds = [0.0] + [run[starts[n]] for n in names] + [total]
    out = {"cases": bounds[1] - bounds[0]}
    for i, n in enumerate(names):
        out[n] = bounds[i + 2] - bounds[i + 1]
    return out


def summarize(data, starts):
    per = [segments(run, starts, total) for run, total in zip(data["call_seconds"], data["total_seconds"], strict=True)]
    med = {k: statistics.median(p[k] for p in per) for k in per[0]}
    med["total"] = statistics.median(data["total_seconds"])
    gaps = [b - a for run in data["call_seconds"] for a, b in zip(run, run[1:])]
    return med, gaps


def main():
    zig, py = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
    if zig["calls"] != py["calls"]:
        sys.exit(f"call counts differ: dszig {zig['calls']} vs dspy {py['calls']}")
    starts = py["pipeline_starts"]
    zm, zg = summarize(zig, starts)
    pm, pg = summarize(py, starts)
    print(f"Transcript: `{py['transcript']}`, {py['calls']} LM calls; DSZig {zig['optimize']} x{zig['reps']}, "
          f"{py['runtime']} (Python {py['python']}) x{py['reps']}. Medians; no network.\n")
    print("| Segment | DSPy ms | DSZig ms | Speedup |")
    print("|---|---:|---:|---:|")
    for k in pm:
        print(f"| {k} | {pm[k] * 1e3:.1f} | {zm[k] * 1e3:.2f} | {pm[k] / zm[k]:.0f}x |")
    print()
    print("| Metric | DSPy | DSZig |")
    print("|---|---:|---:|")
    print(f"| Throughput (LM calls/s of framework work) | {py['calls'] / pm['total']:,.0f} | {zig['calls'] / zm['total']:,.0f} |")
    for q in (0.5, 0.99):
        print(f"| Per-call overhead p{int(q * 100)} (us) | {pct(pg, q) * 1e6:,.1f} | {pct(zg, q) * 1e6:,.1f} |")
    print(f"| Startup: `import dspy` vs none (ms) | {py['import_seconds'] * 1e3:,.0f} | 0 |")
    print(f"| Peak heap, one run (MiB) | {py['heap_peak_bytes'] / 2**20:,.1f} (tracemalloc) | {zig['heap_peak_bytes'] / 2**20:,.1f} (requested) |")
    print(f"| Heap reserved incl. arena chunks, one run (MiB) | n/a | {zig['heap_reserved_peak_bytes'] / 2**20:,.1f} |")
    print(f"| Peak RSS, process (MiB) | {py['maxrss_kb'] / 1024:,.0f} | {zig['maxrss_kb'] / 1024:,.0f} |")


if __name__ == "__main__":
    main()
