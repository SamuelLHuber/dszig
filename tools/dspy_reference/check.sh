#!/usr/bin/env bash
# Cross-runtime gate against real DSPy 3.4.1 (uv.lock pins every package by
# version and sha256):
#   1. regenerating the golden fixtures with DSPy reproduces the committed ones;
#   2. DSPy loads the DSZig-compiled program and re-saves it byte-for-byte.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cd "$here"
uv sync --locked --quiet
uv run --locked gen_fixtures.py "$tmp/fixtures" 2>/dev/null
diff -r "$root/tests/fixtures" "$tmp/fixtures"
echo "ok: DSPy 3.4.1 regenerates tests/fixtures unchanged"

(cd "$root" && zig build bootstrap -- "$tmp/dszig_rag.json" >/dev/null)
uv run --locked check_artifact.py "$tmp/dszig_rag.json"
