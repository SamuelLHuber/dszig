#!/usr/bin/env python3
"""Observe actual scalar Literal validation, schemas and typed DSPy calls."""
import argparse
import json
from pathlib import Path
from typing import Literal, Optional

import dspy
from pydantic import TypeAdapter
from dspy.adapters.utils import parse_value
import reference as ref
import tool_argument_repair_oracle as repair
from parse_value_repair_oracle import enc

ARGUMENTS = [
    (1, True), (True, 1), (0, False), (False, 0), (1, "a", None),
    (1.0, 1), (1, 1.0), (10**50,), (0, 10**50), (None,),
    ("a", "b"), ("a'b", 'a"b', "日本"), (2.5, -0.0),
    (False, 0, 0.0, "0"), (1, 1, True, True),
]
INPUTS = [None, False, True, -1, 0, 1, 2, 2.5, -0.0, 0.0, 1.0,
          10**50, -(10**50), float("inf"), float("nan"), [], {},
          "", "1", "0", "True", "False", "None", "null", "a", " a ",
          "'a'", '"a"', "Literal['a']", "str[1]", "Literal[True]",
          "1.0", "2.5", str(10**50), "a'b", 'a"b', "日本", b"a"]


def observe(fn, x):
    try:
        return {"ok": True, "value": enc(fn(x))}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, schemas = [], []
    for index, values in enumerate(ARGUMENTS):
        annotation = Literal[values]
        adapter = TypeAdapter(annotation)
        schemas.append({"index": index, "schema": adapter.json_schema()})
        for x in INPUTS:
            rows.append({"index": index, "input": enc(x),
                         "pydantic": observe(adapter.validate_python, x),
                         "dspy": observe(lambda v: parse_value(v, annotation), x)})
    class Typed(dspy.Signature):
        question: str = dspy.InputField()
        choice: Literal[1, True, "ready", None] = dspy.OutputField()
        choices: list[Literal[1, True, "ready", None]] = dspy.OutputField()
        optional: Optional[Literal[1, True, "ready", None]] = dspy.OutputField()

    predictions = []
    for adapter, response in [
        (dspy.ChatAdapter(), '[[ ## choice ## ]]\nTrue\n[[ ## choices ## ]]\n[1.0, true, "ready", null]\n[[ ## optional ## ]]\nnull\n[[ ## completed ## ]]'),
        (dspy.JSONAdapter(), '{"choice":1.0,"choices":[1.0,true,"ready",null],"optional":null}'),
        (dspy.XMLAdapter(), '<choice>1.0</choice><choices><item>1.0</item><item>true</item><item>ready</item><item>null</item></choices><optional>null</optional>'),
    ]:
        lm = ref.KeywordLM([], response)
        with dspy.context(lm=lm, adapter=adapter):
            result = dspy.Predict(Typed)(question="Decode")
        predictions.append({"adapter": type(adapter).__name__, "response": response,
                            "outputs": enc(result.toDict()), "requests": lm.calls})
    receipt = {"evidence_kind": "actual_dspy_scalar_literal_calls", "provenance": repair.provenance(),
               "rows": rows, "schemas": schemas, "predictions": predictions}
    args.output.write_text(json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n")
    print(f"Observed {len(rows)*2} scalar Literal calls, {len(schemas)} schemas and {len(predictions)} typed predictions")


if __name__ == "__main__":
    main()
