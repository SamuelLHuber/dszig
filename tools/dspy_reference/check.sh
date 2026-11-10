#!/usr/bin/env bash
# Cross-runtime gate against real DSPy 3.4.1 (uv.lock pins every package by
# version and sha256):
#   1. regenerating the golden fixtures with DSPy (and the Optuna 4.5.0 TPE
#      trace MIPROv2 relies on, and the multi-hop trace-pick pickle hashes)
#      reproduces the committed ones;
#   2. DSPy loads the DSZig-compiled program and re-saves it byte-for-byte;
#   3. DSPy replays the committed real-model transcripts (its own and the one
#      DSZig recorded with `zig build live`) and reproduces every request,
#      output, score and compiled artifact. No API key needed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cd "$here"
uv sync --locked --quiet
uv run --locked gen_fixtures.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_optuna.py "$tmp/fixtures"
uv run --locked gen_pickle.py "$tmp/fixtures" >/dev/null
uv run --locked gen_react.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_ensemble.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_simba.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_gepa.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_nested.py "$tmp/fixtures" 2>/dev/null
diff -r --exclude=live "$root/tests/fixtures" "$tmp/fixtures"
echo "ok: DSPy 3.4.1 regenerates tests/fixtures unchanged"
uv run --locked gen_unicode.py "$tmp/src"
diff "$root/src/unicode_tables.zig" "$tmp/src/unicode_tables.zig"
echo "ok: CPython 3.11 regenerates src/unicode_tables.zig unchanged"
uv run --locked gen_program_code.py "$tmp/program_code"
diff "$root/examples/dates_program_code.txt" "$tmp/program_code/dates_program_code.txt"
diff "$root/examples/dates_class_source.txt" "$tmp/program_code/dates_class_source.txt"
echo "ok: DSPy 3.4.1 regenerates examples/dates_program_code.txt and dates_class_source.txt unchanged"

(cd "$root" && zig build bootstrap -- "$tmp/dszig_rag.json" >/dev/null)
uv run --locked check_artifact.py "$tmp/dszig_rag.json"

for transcript in "$root"/tests/fixtures/live/*.json; do
  uv run --locked check_live.py "$transcript" 2>/dev/null
done
