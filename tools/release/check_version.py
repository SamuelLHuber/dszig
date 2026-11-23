#!/usr/bin/env python3
"""Enforce DSZig's release version scheme: v<DSPy version>+dszig.<N>.

The version core is the upstream DSPy release DSZig targets (as RKE2 and K3s
track Kubernetes); +dszig.N counts DSZig builds of that DSPy version, starting
at 1 and restarting at 1 when the DSPy target changes. build.zig.zon, the
suite's target_dspy, the pinned DSPy reference and the README install line
must all agree. Released tags are immutable.

    check_version.py                 # repository is self-consistent
    check_version.py --tag TAG       # TAG names the repository version
    check_version.py --tag TAG --new # ...and is the next unreleased build
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEME = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\+dszig\.([1-9]\d*)")


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def one(pattern, text, what):
    found = re.findall(pattern, text, re.MULTILINE)
    if len(found) != 1:
        sys.exit(f"error: expected exactly one {what}, found {len(found)}")
    return found[0]


def tags():
    out = subprocess.run(["git", "-C", str(ROOT), "tag", "--list", "v*"], check=True, capture_output=True, text=True)
    return out.stdout.split()


def released_tags():
    # A local tag left by `cut.sh --no-push` is not a release; origin decides.
    out = subprocess.run(["git", "-C", str(ROOT), "ls-remote", "--tags", "--refs", "origin", "v*"], check=True, capture_output=True, text=True)
    return [line.split("refs/tags/", 1)[1] for line in out.stdout.splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--new", action="store_true", help="require TAG to be unreleased and the next build number")
    args = parser.parse_args()
    if args.new and not args.tag:
        parser.error("--new requires --tag")

    version = one(r'^\s*\.version = "([^"]+)",', read("build.zig.zon"), "build.zig.zon .version")
    match = SCHEME.fullmatch(version)
    if not match:
        sys.exit(f"error: build.zig.zon version {version!r} is not <DSPy MAJOR.MINOR.PATCH>+dszig.<N>")
    core, build = version.split("+dszig.")
    sources = {
        "examples/live_suite.zig target_dspy": one(r'^pub const target_dspy = "([^"]+)";', read("examples/live_suite.zig"), "target_dspy"),
        "tools/dspy_reference/pyproject.toml dspy pin": one(r'"dspy(?:\[[^\]]*\])?==([^"]+)"', read("tools/dspy_reference/pyproject.toml"), "dspy pin"),
        "tools/dspy_reference/uv.lock dspy": one(r'^name = "dspy"\nversion = "([^"]+)"', read("tools/dspy_reference/uv.lock"), "locked dspy"),
    }
    for where, pinned in sources.items():
        if pinned != core:
            sys.exit(f"error: version core {core} differs from {where} {pinned}")
    install = f"--branch 'v{version}'"
    if install not in read("README.md"):
        sys.exit(f"error: README install command must use {install}")

    if args.tag:
        if args.tag != f"v{version}":
            sys.exit(f"error: tag {args.tag} does not name repository version v{version}")
        existing = tags()
        for tag in existing:
            if not SCHEME.fullmatch(tag[1:]):
                sys.exit(f"error: existing tag {tag} breaks the version scheme")
        if args.new:
            released = released_tags()
            if args.tag in released:
                sys.exit(f"error: {args.tag} is already released; released tags are immutable, bump +dszig.N")
            if args.tag in existing:
                sys.exit(f"error: unpushed local tag {args.tag} exists; push it or delete it with git tag -d")
            builds = [int(t.split("+dszig.")[1]) for t in released if t.startswith(f"v{core}+dszig.")]
            expected = max(builds, default=0) + 1
            if int(build) != expected:
                sys.exit(f"error: next DSZig build of DSPy {core} is +dszig.{expected}, not +dszig.{build}")
    print(f"ok: DSZig {version} targets DSPy {core}")


if __name__ == "__main__":
    main()
