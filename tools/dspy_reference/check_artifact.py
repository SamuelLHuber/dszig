"""Load a DSZig-written program state into DSPy, re-save it, require identical bytes.

Usage: uv run check_artifact.py path/to/dszig_rag.json
"""

import pathlib
import sys
import tempfile

import reference as ref


def main():
    src = pathlib.Path(sys.argv[1])
    original = src.read_bytes()
    program = ref.RAG()
    program.load(str(src))
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "resaved.json"
        program.save(str(out))
        resaved = out.read_bytes()
    if resaved != original:
        sys.stderr.write("DSPy re-save differs from DSZig artifact\n")
        sys.exit(1)
    print(f"ok: DSPy 3.4.1 loaded and re-saved {src} byte-for-byte ({len(original)} bytes)")


if __name__ == "__main__":
    main()
