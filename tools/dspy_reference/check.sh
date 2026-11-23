#!/usr/bin/env bash
# Cross-runtime gate against real DSPy 3.5.0 (uv.lock pins every package by
# version and sha256):
#   1. regenerating the golden fixtures with DSPy (and the Optuna 4.5.0 TPE
#      trace MIPROv2 relies on, and positional multi-hop trace selections)
#      reproduces the committed ones;
#   2. DSPy loads the DSZig-compiled program and re-saves it byte-for-byte;
#   3. Both runtimes replay the independently recorded 3.5.0 deterministic mock
#      suite and reproduce every request, output, score and compiled artifact.
#      Both also replay current-version live recordings when present, offline.
#      Immutable 3.4.1 live recordings remain historical, not 3.5.0 proof.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cd "$here"
uv sync --locked --quiet
uv run --locked check_optimizer_exports.py
uv run --locked canonical_retry_oracle.py
uv run --locked probe_simba_prompt_lm.py
uv run --locked bootstrap_teacher_oracle.py
if ! uv run --locked fewshot_options_oracle.py "$tmp/fewshot_options.json" >"$tmp/fewshot_options.log" 2>&1; then
  cat "$tmp/fewshot_options.log" >&2
  exit 1
fi
(cd "$root" && zig run --dep dszig -Mroot=tools/fewshot_options_consumer.zig -Mdszig=src/root.zig -- "$tmp/fewshot_options.json")
(cd "$root" && zig build-exe --dep dszig -Mroot=tools/gepa_restart_consumer.zig -Mdszig=src/root.zig -femit-bin="$tmp/gepa_restart_consumer")
uv run --locked gepa_restart_oracle.py "$tmp/gepa_restart_consumer"
uv run --locked gen_fixtures.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_optuna.py "$tmp/fixtures"
uv run --locked gen_trace_selection.py "$tmp/fixtures"
uv run --locked gen_react.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_ensemble.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_parallel.py "$tmp/fixtures"
uv run --locked gen_simba.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_gepa.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_nested.py "$tmp/fixtures" 2>/dev/null
uv run --locked gen_xml.py "$tmp/fixtures"
uv run --locked gen_suite_replay.py "$tmp/fixtures" 2>/dev/null
diff -r --exclude=live "$root/tests/fixtures" "$tmp/fixtures"
echo "ok: DSPy 3.5.0 regenerates tests/fixtures unchanged"
uv run --locked gen_unicode.py "$tmp/src"
diff "$root/src/unicode_tables.zig" "$tmp/src/unicode_tables.zig"
echo "ok: CPython 3.11 regenerates src/unicode_tables.zig unchanged"
uv run --locked gen_dpr_unicode.py "$tmp/src/dpr_unicode.zig"
diff "$root/src/dpr_unicode.zig" "$tmp/src/dpr_unicode.zig"
echo "ok: DSPy 3.5.0/regex regenerates src/dpr_unicode.zig unchanged"
uv run --locked gen_program_code.py "$tmp/program_code"
diff "$root/examples/dates_program_code.txt" "$tmp/program_code/dates_program_code.txt"
diff "$root/examples/dates_class_source.txt" "$tmp/program_code/dates_class_source.txt"
echo "ok: DSPy 3.5.0 regenerates examples/dates_program_code.txt and dates_class_source.txt unchanged"

(cd "$root" && zig build bootstrap -- "$tmp/dszig_rag.json" >/dev/null)
uv run --locked upstream_report.py
uv run --locked check_artifact.py "$tmp/dszig_rag.json"

mock_transcript="$root/tests/fixtures/dspy_3_5_0_mock_suite.json"
(cd "$root" && zig build replay -- "$mock_transcript")
uv run --locked check_live.py "$mock_transcript"
shopt -s nullglob
live_transcripts=("$root"/tests/fixtures/live/*_3.5.0*.json)
if ((${#live_transcripts[@]} == 0)); then
  echo "note: no DSPy 3.5.0 live-model transcript is present; deterministic mock replay is not live proof"
fi
for transcript in "${live_transcripts[@]}"; do
  (cd "$root" && zig build replay -- "$transcript")
  uv run --locked check_live.py "$transcript"
  (cd "$root" && zig build replay -- "$transcript")
done
