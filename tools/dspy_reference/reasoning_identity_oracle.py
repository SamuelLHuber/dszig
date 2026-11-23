"""Observe actual installed DSPy 3.5 Reasoning identity and serialization; never run native code.

Every expected value is produced by the pinned Python libraries. Tool-result text uses the
actual private adapter helper ``dspy.adapters.base._tool_result_content``, not a serializer proxy.
"""

import argparse
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import dspy
import pydantic
from pydantic import TypeAdapter

from dspy.adapters import base as adapter_base
from dspy.adapters.types import base_type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.utils import serialize_for_json


CONTENTS = ["", "think", "  spaced\n  ", "a'b", 'a"b', "a'\"b\\", "雪😀é",
            "\x00\x08\x0c\n\r\t\x1f\x7f\u2028雪😀"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_receipt(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def encode(value):
    """Explicitly tag actual Reasoning objects so JSON never erases class identity."""
    if isinstance(value, Reasoning):
        return {"python_type": "Reasoning", "content": value.content}
    if isinstance(value, (dspy.File, dspy.BoundingBox)):
        return {"python_type": type(value).__name__, "fields": value.model_dump(round_trip=True)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {"python_type": "dict", "items": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, list):
        return {"python_type": "list", "items": [encode(item) for item in value]}
    if isinstance(value, tuple):
        return {"python_type": "tuple", "items": [encode(item) for item in value]}
    if isinstance(value, BaseException):
        return {"python_type": type(value).__name__, "message": str(value)}
    raise TypeError(f"Oracle needs an explicit encoding for {type(value)!r}")


def error(exc):
    result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, pydantic.ValidationError):
        result["validation_errors"] = json.loads(exc.json(include_url=False))
    return result


def observe(fn):
    try:
        return {"ok": True, "value": encode(fn())}
    except Exception as exc:
        return error(exc)


def object_views(value):
    return {
        "type": type(value).__name__,
        "content": value.content,
        "str": str(value),
        "repr": repr(value),
        "format": value.format(),
        "serialize_model": value.serialize_model(),
        "model_dump": encode(value.model_dump()),
        "model_dump_json_mode": encode(value.model_dump(mode="json")),
        "model_dump_round_trip": encode(value.model_dump(round_trip=True)),
        "model_dump_json": value.model_dump_json(),
        "model_dump_json_round_trip": value.model_dump_json(round_trip=True),
        "serialize_for_json": encode(serialize_for_json(value)),
        "tool_result_content": encode(adapter_base._tool_result_content(value)),
        "len": len(value),
    }


def validation_cases():
    actual = Reasoning(content="actual 雪")
    inputs = [
        ("actual", actual),
        *[(f"string:{index}", content) for index, content in enumerate(CONTENTS)],
        *[(f"dict:{index}", {"content": content}) for index, content in enumerate(CONTENTS)],
        ("dict_extra", {"content": "kept", "extra": {"content": "ignored"}, "python": [1, 2]}),
        ("null", None), ("false", False), ("integer", 42), ("float", 1.5), ("list", []),
        ("empty_dict", {}), ("other_key", {"other": "text"}), ("content_null", {"content": None}),
        ("content_false", {"content": False}), ("content_integer", {"content": 42}),
        ("content_list", {"content": []}), ("content_dict", {"content": {"content": "nested"}}),
        ("content_actual", {"content": Reasoning(content="nested")}),
    ]
    adapters = {"model_validate": Reasoning.model_validate, "type_adapter": TypeAdapter(Reasoning).validate_python}
    rows = []
    for entrypoint, validate in adapters.items():
        for case_id, value in inputs:
            row = {"id": case_id, "entrypoint": entrypoint, "input": encode(value)}
            try:
                result = validate(value)
                row.update({"ok": True, "result": object_views(result), "same_instance": result is value})
                if isinstance(value, dict):
                    row["input_unchanged"] = encode(value)
            except Exception as exc:
                row.update(error(exc))
            rows.append(row)
    optional = TypeAdapter(Reasoning | None)
    rows.append({"id": "optional_null", "entrypoint": "type_adapter_optional",
                 **observe(lambda: optional.validate_python(None))})
    return rows


def constructor_cases():
    rows = []
    for index, content in enumerate(CONTENTS):
        value = Reasoning(content=content)
        rows.append({"id": f"keyword:{index}", "views": object_views(value)})
    for case_id, fn in [("missing", lambda: Reasoning()), ("null", lambda: Reasoning(content=None)),
                        ("extra", lambda: Reasoning(content="x", extra="ignored"))]:
        rows.append({"id": case_id, **observe(fn)})
    return rows


def equality_cases():
    content = "same 雪"
    actual = Reasoning(content=content)
    rows = []
    comparands = [("same_instance", actual), ("equal_instance", Reasoning(content=content)),
                  ("different_instance", Reasoning(content="other")), ("string", content),
                  ("different_string", "other"), ("dict", {"content": content}), ("null", None),
                  ("false", False), ("zero", 0), ("list_actual", [actual]), ("list_string", [content])]
    for case_id, other in comparands:
        rows.append({"id": case_id, "other": encode(other), "reasoning_eq_other": actual == other,
                     "other_eq_reasoning": other == actual, "reasoning_ne_other": actual != other,
                     "other_ne_reasoning": other != actual, "identity": actual is other})
    rows.append({"id": "nested_lists", "actual_vs_string": [actual] == [content],
                 "string_vs_actual": [content] == [actual], "actual_vs_dict": [actual] == [{"content": content}],
                 "dict_vs_actual": [{"content": content}] == [actual]})
    rows.append({"id": "nested_dicts", "actual_vs_string": {"r": actual} == {"r": content},
                 "string_vs_actual": {"r": content} == {"r": actual}})
    return rows


def nested_cases():
    actual = Reasoning(content="nested 雪\n'\"")
    value = {"values": [actual, actual.content, {"content": actual.content}, None],
             "tuple": (actual,), "dict": {"reasoning": actual}}
    list_value = [actual, actual.content, {"content": actual.content}, None]
    return {
        "input": encode(value),
        "str": str(value),
        "repr": repr(value),
        "serialize_for_json": encode(serialize_for_json(value)),
        "json_dumps_serialized": json.dumps(serialize_for_json(value), ensure_ascii=False),
        "type_adapter_json": TypeAdapter(type(value)).dump_python(value, mode="json"),
        "type_adapter_json_round_trip": TypeAdapter(type(value)).dump_python(value, mode="json", round_trip=True),
        "type_adapter_json_bytes": TypeAdapter(type(value)).dump_json(value).decode(),
        "type_adapter_json_bytes_round_trip": TypeAdapter(type(value)).dump_json(value, round_trip=True).decode(),
        "tool_result_content": encode(adapter_base._tool_result_content(value)),
        "list_tool_result_content": encode(adapter_base._tool_result_content(list_value)),
        "string_tool_result_content": encode(adapter_base._tool_result_content(actual.content)),
        "dict_tool_result_content": encode(adapter_base._tool_result_content({"content": actual.content})),
        "original_after_serialization": encode(value),
    }


def deepcopy_cases():
    actual = Reasoning(content="copy 雪")
    graph = {"first": actual, "second": actual, "items": [actual, actual]}
    copied = copy.deepcopy(graph)
    shallow = copy.copy(actual)
    deep = copy.deepcopy(actual)
    typed = TypeAdapter(list[Reasoning]).validate_python([actual, "new", actual])
    nested_typed = TypeAdapter(dict[str, list[Reasoning | None]]).validate_python(
        {"items": [actual, {"content": "dict", "extra": 1}, None, actual]})
    return {
        "direct_copy_equal": shallow == actual, "direct_copy_identity": shallow is actual,
        "direct_deepcopy_equal": deep == actual, "direct_deepcopy_identity": deep is actual,
        "graph_values_equal": copied == graph,
        "graph_first_new_identity": copied["first"] is not actual,
        "graph_alias_preserved": copied["first"] is copied["second"] is copied["items"][0] is copied["items"][1],
        "typed_list": encode(typed),
        "typed_list_identity": [typed[0] is actual, typed[2] is actual, typed[0] is typed[2]],
        "typed_nested": encode(nested_typed),
        "typed_nested_identity": [nested_typed["items"][0] is actual, nested_typed["items"][3] is actual],
    }


def protocol_cases():
    import orjson
    from dspy.predict.refine import recursive_mask as refine_mask
    from dspy.teleprompt.simba_utils import recursive_mask as simba_mask

    reasoning = Reasoning(content="reason 雪")
    box = dspy.BoundingBox(top=0, bottom=1, left=0, right=1)
    file = dspy.File(file_id="file-protocol")
    invalid_file = dspy.File(file_data=None)
    inputs = [
        ("actual", reasoning), ("plain_string", reasoning.content),
        ("plain_dict", {"content": reasoning.content}),
        ("nested", {"items": [reasoning, reasoning.content, {"content": reasoning.content},
                              box, box.model_dump(), file, invalid_file, None]}),
    ]
    masks = []
    for name, value in inputs:
        refined, simba = refine_mask(value), simba_mask(value)
        masks.append({"id": name, "input": encode(value),
                      "refine": encode(refined), "simba": encode(simba),
                      "refine_json": orjson.dumps(refined, option=orjson.OPT_INDENT_2).decode(),
                      "simba_json": orjson.dumps(simba, option=orjson.OPT_INDENT_2).decode()})
    truthiness = [{"id": name, "input": encode(value), "value": bool(value)}
                  for name, value in [("empty", Reasoning(content="")), ("space", Reasoning(content=" ")),
                                      ("nul", Reasoning(content="\0")), ("file", file), ("bbox", box)]]
    votes = []
    for name, value in [("reasoning", reasoning), ("file", file), ("bbox", box)]:
        try:
            result = dspy.majority([{"answer": value, "marker": "first"},
                                    {"answer": value, "marker": "second"}], field="answer")
            outcome = {"ok": True, "marker": result.marker}
        except Exception as exc:
            outcome = {"ok": False, "error": type(exc).__name__}
        votes.append({"id": name, "input": encode(value), "result": outcome})
    return {"masks": masks, "truthiness": truthiness, "votes": votes,
            "sources": {"refine": source_receipt(inspect.getsourcefile(refine_mask)),
                        "simba": source_receipt(inspect.getsourcefile(simba_mask))}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert dspy.__version__ == "3.5.0", dspy.__version__
    receipt = {
        "format_version": 1,
        "dspy": dspy.__version__,
        "pydantic": pydantic.__version__,
        "python": sys.version,
        "sources": {"reasoning": source_receipt(inspect.getsourcefile(Reasoning)),
                    "base_type": source_receipt(inspect.getsourcefile(base_type)),
                    "adapter_base": source_receipt(inspect.getsourcefile(adapter_base))},
        "is_type": issubclass(Reasoning, base_type.Type),
        "schema": Reasoning.model_json_schema(),
        "constructors": constructor_cases(),
        "validation": validation_cases(),
        "equality": equality_cases(),
        "nested": nested_cases(),
        "copies": deepcopy_cases(),
        "protocol": protocol_cases(),
    }
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "validation_cases": len(receipt["validation"]),
                          "sha256": digest(encoded.encode("utf-8"))}))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
