"""Check native declaration coverage of the pinned optimizer and predictor inventories.

A declaration or migration equivalent is not proof of behavioral parity.
"""

import json
from pathlib import Path
import subprocess
import tempfile

import dspy
from dspy import predict, teleprompt


def main() -> None:
    if dspy.__version__ != "3.5.0":
        raise RuntimeError(f"This check requires DSPy 3.5.0, found {dspy.__version__}")
    root = Path(__file__).resolve().parents[2]
    inventories = (
        (teleprompt, {"bootstrap_trace_data": ("dszig", "bootstrapTraceData")}),
        # RLM has a native migration surface, not a bundled Python interpreter.
        (predict, {"majority": ("dszig.ensemble", "majority"), "RLM": ("dszig", "CustomRLM")}),
    )
    checks = [
        'const dszig = @import("dszig");',
        "comptime {",
    ]
    for module, equivalents in inventories:
        for source in module.__all__:
            namespace, native = equivalents.get(source, ("dszig", source))
            # Names and diagnostics are ASCII; JSON quoting is Zig-compatible here.
            message = f"missing native counterpart {namespace}.{native} for {module.__name__}.{source}"
            checks.append(
                f"    if (!@hasDecl({namespace}, {json.dumps(native)})) @compileError({json.dumps(message)});"
            )
    checks.append("}")
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "optimizer_exports.zig"
        probe.write_text("\n".join(checks) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "zig", "build-obj", "-fno-emit-bin", "--dep", "dszig",
                f"-Mroot={probe}", f"-Mdszig={root / 'src/root.zig'}",
            ],
            cwd=root,
            check=True,
        )
    for module, _ in inventories:
        print(f"ok: {len(module.__all__)} exports in {module.__name__} have native counterparts (inventory only)")


if __name__ == "__main__":
    main()
