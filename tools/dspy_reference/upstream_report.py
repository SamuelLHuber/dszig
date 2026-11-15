"""Track DSZig against upstream DSPy: every public `dspy.*` name must have a
row in features.json, and every upstream source file a row points at must be
unchanged since the row was last reviewed.

    uv run --locked upstream_report.py            # check; exit 1 on drift
    uv run --locked upstream_report.py --update   # re-pin upstream paths/hashes
    uv run --locked upstream_report.py --markdown [dszig_bench.json dspy_bench.json]

After a DSPy release: bump the lock, run the check, port or classify every new
name and changed file, then --update."""

import hashlib
import inspect
import json
import os
import pathlib
import statistics
import sys

import dspy

FEATURES = pathlib.Path(__file__).with_name("features.json")
ROOT = pathlib.Path(dspy.__file__).parent


def public_api():
    out = {}
    for name in sorted(n for n in dir(dspy) if not n.startswith("_")):
        obj = getattr(dspy, name)
        if inspect.ismodule(obj):
            continue
        try:
            path = pathlib.Path(inspect.getsourcefile(obj)).resolve()
            rel = path.relative_to(ROOT.resolve()).as_posix()
        except (TypeError, ValueError):
            rel = None
        out[name] = rel
    return out


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() if rel else None


def check(data):
    api = public_api()
    rows = {r["name"]: r for r in data["features"]}
    problems = []
    if data["dspy"] != dspy.__version__:
        problems.append(f"features.json reviewed against DSPy {data['dspy']}, locked DSPy is {dspy.__version__}")
    problems += [f"new upstream name without a feature row: dspy.{n} ({api[n]})" for n in api if n not in rows]
    problems += [f"feature row for a name upstream no longer exports: dspy.{n}" for n in rows if n not in api]
    for n, rel in api.items():
        row = rows.get(n)
        if row and (row.get("upstream") != rel or row.get("sha256") != sha(rel)):
            problems.append(f"upstream source changed since review: dspy.{n} ({rel})")
    return problems


def update(data):
    api = public_api()
    for row in data["features"]:
        rel = api.get(row["name"])
        row["upstream"], row["sha256"] = rel, sha(rel)
    data["dspy"] = dspy.__version__
    FEATURES.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def bench_segments(paths):
    if not paths:
        return {}
    import bench_report

    zig, py = (json.load(open(p, encoding="utf-8")) for p in paths)
    zm, _ = bench_report.summarize(zig, py["pipeline_starts"])
    pm, _ = bench_report.summarize(py, py["pipeline_starts"])
    return {k: (pm[k], zm[k]) for k in pm}


def markdown(data, paths):
    seg = bench_segments(paths)
    print("| DSPy 3.5.0 | Status | DSZig source | Equality checks | Replay benchmark (DSPy → DSZig) |")
    print("|---|---|---|---|---|")
    for r in data["features"]:
        bench = "; ".join(
            f"{s}: {seg[s][0] * 1e3:.0f} ms → {seg[s][1] * 1e3:.2f} ms" for s in r.get("bench", []) if s in seg
        ) or "–"
        up = f"`{r['upstream']}`" if r.get("upstream") else "–"
        print(f"| `dspy.{r['name']}` ({up}) | {r['status']} | {', '.join(f'`{s}`' for s in r['dszig']) or '–'} "
              f"| {r['checks'] or '–'} | {bench} |")


def main():
    data = json.loads(FEATURES.read_text(encoding="utf-8"))
    if "--update" in sys.argv:
        update(data)
        return
    if "--markdown" in sys.argv:
        markdown(data, sys.argv[sys.argv.index("--markdown") + 1:])
        return
    problems = check(data)
    for p in problems:
        print(p)
    n = len(data["features"])
    counts = {s: sum(r["status"] == s for r in data["features"]) for s in ("native", "partial", "missing")}
    print(f"{n} public DSPy {dspy.__version__} names mapped: {counts}; {len(problems)} drift problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
