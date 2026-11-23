#!/usr/bin/env python3
"""Observe actual DSPy parse_value before extending shared native field parsing.

Tagged observations retain class identity, tuples/bytes/sets, ordered dict keys,
exact integers, binary64 bits and surrogate code points rather than serializing
all Python objects into misleading lookalike JSON objects.
"""
import argparse
import enum
import hashlib
import inspect
import json
import struct
from pathlib import Path
from typing import Any, Literal

import dspy
from dspy.adapters import utils
import tool_argument_repair_oracle as repair_oracle

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")


class Color(enum.Enum):
    red = "red"
    blue = "blue"


def decimal(value):
    # Do not change sys.int_max_str_digits: it changes the observed parser.
    if -10**4000 < value < 10**4000:
        return str(value)
    negative = value < 0
    value = abs(value)
    chunks = []
    while value:
        value, remainder = divmod(value, 10**9)
        chunks.append(remainder)
    return ("-" if negative else "") + str(chunks[-1]) + "".join(f"{n:09d}" for n in reversed(chunks[:-1]))


def enc(value):
    if isinstance(value, dspy.Code):
        return {"model": "Code", "language": value.language, "code": value.code}
    if isinstance(value, dspy.Reasoning):
        return {"model": "Reasoning", "content": value.content}
    if isinstance(value, dspy.File):
        return {"model": "File", "fields": enc({name: getattr(value, name) for name in ("file_data", "file_id", "filename")})}
    if isinstance(value, dspy.BoundingBox):
        return {"model": "BoundingBox", "fields": enc({name: getattr(value, name) for name in ("top", "left", "bottom", "right")})}
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__name__, "value": enc(value.value)}
    if value is Ellipsis:
        return {"ellipsis": True}
    if value is None:
        return {"null": True}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": decimal(value)}
    if isinstance(value, float):
        return {"float": repr(value), "bits": struct.pack(">d", value).hex()}
    if isinstance(value, str):
        return repair_oracle.enc(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex": [enc(value.real), enc(value.imag)]}
    if isinstance(value, (list, tuple, set, frozenset)):
        key = {list: "array", tuple: "tuple", set: "set", frozenset: "frozenset"}[type(value)]
        return {key: [enc(item) for item in value]}
    if isinstance(value, dict):
        return {"object": [[enc(key), enc(item)] for key, item in value.items()]}
    raise TypeError(f"Unsupported observation type: {type(value)!r}")


ANNOTATIONS = [
    ("any", Any),
    ("str", str), ("bool", bool), ("int", int), ("float", float),
    ("str_list", list[str]), ("int_list", list[int]),
    ("object", dict[str, Any]), ("str_dict", dict[str, str]), ("int_dict", dict[str, int]),
    ("optional_str", str | None), ("optional_int", int | None),
    ("literal", Literal["red", "blue"]), ("enum", Color),
    ("code", dspy.Code), ("optional_code", dspy.Code | None), ("code_list", list[dspy.Code]),
    ("reasoning", dspy.Reasoning), ("optional_reasoning", dspy.Reasoning | None),
    ("file", dspy.File), ("bbox", dspy.BoundingBox),
]


def inputs():
    yield from repair_oracle.CORPUS
    for index, value in enumerate([
        None, False, True, 0, 1, 2, -1, 1.0, 1.5, float("nan"), float("inf"),
        [], ["x"], [1, True, "2"], {}, {"x": "y"}, {"x": 1}, {"code": "```x```"},
        {"content": "thought"}, {"file_id": "file-1"},
        {"top": 0, "left": 0, "bottom": 1, "right": 1},
        dspy.Code(code="```x```"), dspy.Code.model_construct(code="```unfiltered```"),
        dspy.Reasoning(content="thought"), dspy.File(file_id="file-1"),
        dspy.BoundingBox(top=0, left=0, bottom=1, right=1),
        ("x", "y"), b"7",
    ]):
        yield f"actual/{index}", value
    for index, text in enumerate([
        "True", "False", "None", "0x10", "0o10", "0b10", "1_000", "+7", "07",
        "('x', 'y')", "{'x', 'y'}", "set()", "b'7'", "1+2j", "...",
        "Literal['red']", "str[blue]", "RED", "Color.red", "'red'",
        "{'code': '```x```'}", "{'content': 'thought'}", "{'file_id': 'file-1'}",
    ]):
        yield f"python_or_annotation/{index}", text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    corpus = list(inputs())
    for label, annotation in ANNOTATIONS:
        for case_id, value in corpus:
            row = {"annotation": label, "id": case_id, "input": enc(value)}
            try:
                result = utils.parse_value(value, annotation)
            except Exception as exc:
                row["result"] = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
            else:
                row["result"] = {"ok": True, "value": enc(result), "same_instance": result is value}
            rows.append(row)
    path = Path(inspect.getfile(utils))
    receipt = {"evidence_kind": "actual_dspy_parse_value_calls", "provenance": repair_oracle.provenance(),
               "parse_value_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": rows}
    data = json.dumps(receipt, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
        print(f"Observed {len(rows)} actual DSPy parse_value calls; sha256={hashlib.sha256(data.encode()).hexdigest()}")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
