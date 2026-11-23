#!/usr/bin/env bash
# Cut a DSZig release from an up-to-date, clean main: enforce the version
# scheme, run every local gate, then create and push the annotated tag. The
# tag push starts .github/workflows/release.yml, which reruns CI and publishes.
# Run inside `devenv shell` (dz-release). Pass --no-push to stop after tagging.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
push=1
if [ "${1:-}" = "--no-push" ]; then push=0; elif [ $# -ne 0 ]; then echo "usage: cut.sh [--no-push]" >&2; exit 2; fi

version="$(sed -n 's/^ *\.version = "\(.*\)",$/\1/p' build.zig.zon)"
tag="v$version"
git fetch --quiet --tags origin main
test "$(git rev-parse --abbrev-ref HEAD)" = main || { echo "error: release from main" >&2; exit 1; }
test -z "$(git status --porcelain)" || { echo "error: working tree is not clean" >&2; exit 1; }
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" || { echo "error: main is not origin/main" >&2; exit 1; }
python3 tools/release/check_version.py --tag "$tag" --new

zig fmt --check .
zig build test examples simulation-test -Doptimize=ReleaseSafe --summary line
zig build test examples simulation-test -Doptimize=Debug --summary line
bash tools/dspy_reference/check.sh

core="${version%%+*}"
git tag -a "$tag" -m "DSZig $version (DSPy $core, Zig $(zig version))"
echo "tagged $tag at $(git rev-parse --short HEAD)"
if [ "$push" = 1 ]; then git push origin "refs/tags/$tag"; fi
