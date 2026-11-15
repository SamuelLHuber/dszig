"""Observe actual DSPy 3.5 BoundingBox object identity versus raw dictionaries.

Run from the repository root:
  tools/dspy_reference/.venv/bin/python tools/dspy_reference/bounding_box_identity_oracle.py [--output PATH]

Every observation calls the installed dspy.BoundingBox, Pydantic and DSPy's own
serialize_for_json/_tool_result_content. No native implementation is read and no
expectation is written by hand. Inputs a native consumer must rebuild are listed
as constructor keyword fields plus how the source built them (validated
constructor or model_construct); the consumer must create genuine native boxes.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import dspy
import pydantic
from pydantic import TypeAdapter
from dspy.adapters import base as adapter_base
from dspy.adapters import utils as adapter_utils
from dspy.adapters.base import _tool_result_content
from dspy.adapters.types import base_type, bounding_box
from dspy.adapters.utils import serialize_for_json

Box = dspy.BoundingBox
ADAPTER = TypeAdapter(Box)

# Validated constructor inputs. Integer edges show strict-float int acceptance;
# signed zero and inexact binary64 values exercise exact float display.
VALID = [
    ("ordinary", {"top": 0.125, "bottom": 0.875, "left": 0.25, "right": 0.75}),
    ("integer_edges", {"top": 0, "bottom": 1, "left": 0, "right": 1}),
    ("signed_zero", {"top": -0.0, "bottom": 1.0, "left": 0.0, "right": 1.0}),
    ("inexact", {"top": 1 / 3, "bottom": 0.5, "left": 1e-5, "right": 0.1 + 0.2}),
    ("tiny", {"top": 0.0, "bottom": 1.0, "left": 0.0, "right": 1e-20}),
]

# model_construct bypasses validation entirely, like a native struct literal
# passed to toRuntime without init. Validation of such an instance shows which
# source validators still run for an existing object.
CONSTRUCTED = [
    ("out_of_range", {"top": 2.0, "bottom": 3.0, "left": 0.0, "right": 1.0}),
    ("reversed", {"top": 0.9, "bottom": 0.1, "left": 0.0, "right": 1.0}),
    ("degenerate", {"top": 0.5, "bottom": 0.5, "left": 0.0, "right": 1.0}),
    ("nan_top", {"top": math.nan, "bottom": 0.5, "left": 0.0, "right": 1.0}),
]


def enc(value):
    """Lossless JSON encoding of observed Python values; class stays explicit."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"python_float": repr(value)}
    if isinstance(value, pydantic.BaseModel):
        return {"python_type": type(value).__name__,
                "round_trip": enc(value.model_dump(round_trip=True))}
    if isinstance(value, dict):
        return {str(key): enc(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [enc(item) for item in value]
    raise TypeError(f"Oracle needs an explicit encoding for {type(value)!r}")


def attempt(function):
    try:
        return {"ok": True, "value": enc(function())}
    except Exception as exc:
        result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, pydantic.ValidationError):
            result["error_types"] = [item["type"] for item in exc.errors()]
        return result


def identity(make, original):
    """Whether validation returned the same object, and its outcome."""
    try:
        result = make()
    except Exception as exc:
        failure = {"ok": False, "error": type(exc).__name__}
        if isinstance(exc, pydantic.ValidationError):
            failure["error_types"] = [item["type"] for item in exc.errors()]
        return failure
    return {"ok": True, "same_object": result is original, "python_type": type(result).__name__,
            "repr": repr(result), "equal": result == original}


def surfaces(box):
    """Every display and serialization surface of one actual instance."""
    return {
        "python_type": type(box).__name__,
        "is_type_subclass": isinstance(box, base_type.Type),
        "fields": {name: {"value": enc(getattr(box, name)), "python_type": type(getattr(box, name)).__name__}
                   for name in ("top", "bottom", "left", "right")},
        "model_dump": attempt(box.model_dump),
        "model_dump_json_mode": attempt(lambda: box.model_dump(mode="json")),
        "model_dump_round_trip": attempt(lambda: box.model_dump(round_trip=True)),
        "model_dump_json": attempt(box.model_dump_json),
        "model_dump_json_round_trip": attempt(lambda: box.model_dump_json(round_trip=True)),
        "str": attempt(lambda: str(box)),
        "repr": attempt(lambda: repr(box)),
        "format": attempt(box.format),
        "str_equals_format": attempt(lambda: str(box) == box.format()),
        "serialize_for_json": attempt(lambda: serialize_for_json(box)),
        "serialize_for_json_python_type": attempt(lambda: type(serialize_for_json(box)).__name__),
        # The real tool boundary; never a direct wrapped-serializer call.
        "tool_result_content": attempt(lambda: _tool_result_content(box)),
        # Separate API boundary: the wrapped serializer itself requires a handler.
        "direct_serialize_model_without_handler": attempt(box.serialize_model),
    }


def equality(box, dictionary, fresh):
    return {
        "model_eq_dict": box == dictionary,
        "dict_eq_model": dictionary == box,
        "model_ne_dict": box != dictionary,
        "model_eq_fresh": box == fresh,
        "fresh_eq_model": fresh == box,
        "fresh_is_model": fresh is box,
        "list_model_eq_list_dict": [box] == [dictionary],
        "list_dict_eq_list_model": [dictionary] == [box],
        "dict_of_model_eq_dict_of_dict": {"b": box} == {"b": dictionary},
        "dict_of_dict_eq_dict_of_model": {"b": dictionary} == {"b": box},
        "list_model_eq_list_fresh": [box] == [fresh],
        "model_eq_json_text": box == box.format(),
        "json_text_eq_model": box.format() == box,
        "model_eq_str_text": box == str(box),
    }


def nested(box, dictionary):
    containers = {
        "list": [box, dictionary, box],
        "dict": {"actual": box, "raw": dictionary},
        "nested": {"boxes": [box, dictionary], "pair": {"inner": [box]}},
    }
    observed = {}
    for name, container in containers.items():
        observed[name] = {
            "str": attempt(lambda c=container: str(c)),
            "repr": attempt(lambda c=container: repr(c)),
            "serialize_for_json": attempt(lambda c=container: serialize_for_json(c)),
            "tool_result_content": attempt(lambda c=container: _tool_result_content(c)),
            "type_adapter_dump_python": attempt(lambda c=container: TypeAdapter(type(c)).dump_python(c)),
            "type_adapter_dump_json_mode": attempt(lambda c=container: TypeAdapter(type(c)).dump_python(c, mode="json")),
        }
    raw = {"list": [dictionary], "dict": {"raw": dictionary}}
    observed["raw_only"] = {name: {"repr": repr(value), "tool_result_content": attempt(lambda v=value: _tool_result_content(v))}
                            for name, value in raw.items()}
    observed["raw_dictionary_tool_result_content"] = attempt(lambda: _tool_result_content(dictionary))
    return observed


def copies(box):
    deep = copy.deepcopy(box)
    shallow = copy.copy(box)
    holder = [box, box]
    deep_holder = copy.deepcopy(holder)
    return {
        "deepcopy_is_original": deep is box,
        "deepcopy_equal": deep == box,
        "deepcopy_repr": repr(deep),
        "copy_is_original": shallow is box,
        "copy_equal": shallow == box,
        "deepcopy_list_shares_one_copy": deep_holder[0] is deep_holder[1],
        "deepcopy_list_item_is_original": deep_holder[0] is box,
        "model_copy_is_original": box.model_copy() is box,
        "model_copy_equal": box.model_copy() == box,
    }


def validations(box, dictionary):
    return {
        "model_validate_instance": identity(lambda: Box.model_validate(box), box),
        "type_adapter_instance": identity(lambda: ADAPTER.validate_python(box), box),
        "type_adapter_instance_lax": identity(lambda: ADAPTER.validate_python(box, strict=False), box),
        "model_validate_dict": identity(lambda: Box.model_validate(dictionary), box),
        "type_adapter_dict": identity(lambda: ADAPTER.validate_python(dictionary), box),
        "type_adapter_dict_returns_dict": isinstance(dictionary, dict),
        "list_adapter_instance_item": attempt(
            lambda: TypeAdapter(list[Box]).validate_python([box])[0] is box),
        "list_adapter_dict_item_type": attempt(
            lambda: type(TypeAdapter(list[Box]).validate_python([dictionary])[0]).__name__),
        "untyped_dict_stays_dict": attempt(lambda: type(TypeAdapter(dict).validate_python(dictionary)).__name__),
        "untyped_any_list_keeps_dict": attempt(
            lambda: type(TypeAdapter(list).validate_python([dictionary])[0]).__name__),
        "untyped_any_list_keeps_instance": attempt(
            lambda: TypeAdapter(list).validate_python([box])[0] is box),
    }


def valid_case(name, fields):
    constructed = Box(**fields)
    dictionary = dict(fields)
    fresh = Box(**fields)
    return {
        "name": name,
        "native_input": {"build": "validated_constructor", "fields": enc(fields)},
        "constructor": attempt(lambda: Box(**fields)),
        "surfaces": surfaces(constructed),
        "raw_dictionary": {"python_type": type(dictionary).__name__, "repr": repr(dictionary),
                           "str": str(dictionary), "json_dumps": json.dumps(dictionary)},
        "equality": equality(constructed, dictionary, fresh),
        "validation": validations(constructed, dictionary),
        "copies": copies(constructed),
        "nested": nested(constructed, constructed.model_dump()),
    }


def constructed_case(name, fields):
    box = Box.model_construct(**fields)
    twin = Box.model_construct(**fields)
    return {
        "name": name,
        "native_input": {"build": "model_construct", "fields": enc(fields)},
        "repr": attempt(lambda: repr(box)),
        "validation": {
            "model_validate_instance": identity(lambda: Box.model_validate(box), box),
            "type_adapter_instance": identity(lambda: ADAPTER.validate_python(box), box),
            "type_adapter_instance_lax": identity(lambda: ADAPTER.validate_python(box, strict=False), box),
            "type_adapter_dict": attempt(lambda: ADAPTER.validate_python(dict(fields))),
        },
        # Python equality compares the __dict__ field objects; NaN equals
        # itself only through float object identity, not IEEE comparison.
        "equality": {"self": box == box, "separately_constructed": box == twin,
                     "deepcopy": box == copy.deepcopy(box), "dict": box == dict(fields)},
        "serialize_for_json": attempt(lambda: serialize_for_json(box)),
        "tool_result_content": attempt(lambda: _tool_result_content(box)),
        "format": attempt(box.format),
    }


def source_receipt(module):
    path = Path(inspect.getsourcefile(module))
    return {"module": module.__name__, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if dspy.__version__ != "3.5.0":
        raise RuntimeError(f"This oracle requires actual DSPy 3.5.0, installed {dspy.__version__}")
    receipt = {
        "format_version": 1,
        "dspy": dspy.__version__,
        "pydantic": pydantic.__version__,
        "python": sys.version,
        "sources": [source_receipt(module) for module in
                    (bounding_box, base_type, adapter_base, adapter_utils, pydantic.main)],
        "class": {"name": Box.__name__, "is_type_subclass": issubclass(Box, base_type.Type),
                  "frozen": Box.model_config.get("frozen"), "strict": Box.model_config.get("strict"),
                  "extra": Box.model_config.get("extra"),
                  "serialize_model_signature": str(inspect.signature(Box.serialize_model))},
        "valid": [valid_case(name, fields) for name, fields in VALID],
        "constructed": [constructed_case(name, fields) for name, fields in CONSTRUCTED],
        "limits": [
            "Python float object identity (NaN self-equality through identity) has no native counterpart.",
            "Python container alias graphs are recorded, not claimed for generic native arrays/maps.",
            "Frozen assignment exceptions are a Python object-protocol feature, not a native value field.",
        ],
    }
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({"output": str(args.output), "valid": len(VALID), "constructed": len(CONSTRUCTED),
                          "sha256": hashlib.sha256(encoded.encode()).hexdigest()}))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
