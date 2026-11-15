#!/usr/bin/env python3
"""Observe DSPy 3.5 JSONAdapter repair, field conversion and error boundaries.

Development oracle only; the native runtime never launches Python. Results are
actual adapter calls, not expectations inferred from the repair library alone.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import dspy
from dspy.adapters import json_adapter
from dspy.adapters import utils

import tool_argument_repair_oracle as repair_oracle

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")


class Text(dspy.Signature):
    a: str = dspy.OutputField()


class Pair(dspy.Signature):
    a: str = dspy.OutputField()
    b: str = dspy.OutputField()


class Integer(dspy.Signature):
    a: int = dspy.OutputField()


COMPLETIONS = [
    '{"a":"Paris"}', '{ "a" : "雪😀" }', '{"a":"x","extra":7}',
    '{"a":"old","a":"new"}', '{"a":"x","b":"y"}',
    '{"a":1}', '{"a":1.5}', '{"a":true}', '{"a":null}',
    '{"a":[1,"x",null]}', '{"a":{"x":"y"}}',
    '{"a":"Paris"', '{a:"Paris"}', "{'a': 'Paris'}", '{"a":"Paris",}',
    '{"a" "Paris"}', '{"a": Paris}', '{"a":"Paris"} trailing',
    'Sure: {"a":"Paris"} done.', '```json\n{"a":"Paris"}\n```',
    '{/* note */ "a": "Paris"}', '{"a":"Paris" // note\n}',
    '[{"a":"first"},{"a":"second"}]',
    'prefix {"a":"first"} middle {"a":"second"}',
    '[0,{"a":"nested"}]', '[[{"a":"deep"}]]',
    '{"a":"literal { brace"}', '{"a":"literal } brace"}',
    '[{"a":"unbalanced { text"}]', '[{"a":"unbalanced } text"}]',
    '{"a":"escaped \\" quote"}', '{"a":"\\ud800"}',
    '{"a":NaN}', '{"a":Infinity}', '{"a":1e400}',
    '{"a":123456789012345678901234567890}',
    '{"a":"7"}', '{"a":"7.0"}', '{"a":"true"}',
    '{"a":false,"b":[]}', '{}', '[]', 'null', 'true', '7', 'hello', '',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    adapter = dspy.JSONAdapter()
    rows = []
    for label, signature in (("text", Text), ("pair", Pair), ("integer", Integer)):
        for index, completion in enumerate(COMPLETIONS):
            result = repair_oracle.attempt(lambda: adapter.parse(signature, completion))
            rows.append({"id": f"{label}/{index}", "signature": label,
                         "completion": completion, "result": result})
    sources = []
    for module in (json_adapter, utils):
        path = Path(inspect.getfile(module))
        sources.append({"name": module.__name__, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = {"evidence_kind": "actual_dspy_adapter_calls", "provenance": repair_oracle.provenance(),
               "adapter_sources": sources, "cases": rows}
    data = json.dumps(receipt, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
        print(f"Observed {len(rows)} actual DSPy adapter calls; sha256={hashlib.sha256(data.encode()).hexdigest()}")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
