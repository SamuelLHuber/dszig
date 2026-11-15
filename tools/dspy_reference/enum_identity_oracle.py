#!/usr/bin/env python3
"""Observe plain Enum identity, DSPy resolution and Pydantic serialization."""
import argparse
import copy
import enum
import dspy
import reference as ref
import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import TypeAdapter
from dspy.adapters.utils import parse_value, serialize_for_json
from dspy.predict.refine import recursive_mask
from parse_value_repair_oracle import enc
import tool_argument_repair_oracle as repair

CLASSES = {
    "color_a": enum.Enum("Color", {"red": "red", "blue": "blue"}),
    "color_b": enum.Enum("Color", {"red": "red", "blue": "blue"}),
    "collision": enum.Enum("Collision", {"first": "second", "second": "value2", "alias": "second"}),
    "integer": enum.Enum("IntegerEnum", {"zero": 0, "one": 1, "big": 10**50, "alias_one": True}),
    "float": enum.Enum("FloatEnum", {"zero": -0.0, "fraction": 2.5, "alias_zero": 0.0}),
    "mixed": enum.Enum("MixedEnum", {"yes": True, "one": 1, "text": "a"}),
    "null": enum.Enum("NullEnum", {"nothing": None}),
    "nonfinite": enum.Enum("NonfiniteEnum", {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")}),
}


def tagged(v):
    if isinstance(v, enum.Enum):
        label = next(label for label, cls in CLASSES.items() if type(v) is cls)
        return {"enum_class": label, "name": v.name, "value": enc(v.value)}
    if isinstance(v, list):
        return {"array": [tagged(item) for item in v]}
    if isinstance(v, dict):
        return {"object": [[tagged(key), tagged(value)] for key, value in v.items()]}
    return enc(v)


def observe(call, argument):
    try:
        result = call(argument)
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {"ok": True, "value": tagged(result), "same_instance": result is argument}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inputs = [None, False, True, 0, 1, 2, 2.5, 10**50, "red", "blue", " red ", '"red"',
              "Color.red", "RED", "first", "second", "value2", "alias", "zero", "one", "1",
              "big", "fraction", "yes", "text", "a", "nothing", "nan", "positive", [], {}]
    inputs.extend(member for cls in CLASSES.values() for member in cls)
    scalar_types = {
        "str": str, "int": int, "float": float, "bool": bool,
        "optional_str": Optional[str], "optional_int": Optional[int],
        "str_list": list[str], "int_list": list[int],
        "str_dict": dict[str, str], "int_dict": dict[str, int],
        "literal": Literal["red", "blue"],
    }
    scalar_inputs = inputs + [[member] for cls in CLASSES.values() for member in cls] + [
        {"x": member} for cls in CLASSES.values() for member in cls
    ]
    scalar_validation = []
    for label, annotation in scalar_types.items():
        adapter = TypeAdapter(annotation)
        for index, item in enumerate(scalar_inputs):
            scalar_validation.append({
                "annotation": label, "input_index": index, "input": tagged(item),
                "dspy": observe(lambda v: parse_value(v, annotation), item),
                "pydantic": observe(adapter.validate_python, item),
            })
    validation = []
    views = []
    schemas = []
    equality = []
    for label, cls in CLASSES.items():
        adapter = TypeAdapter(cls)
        schemas.append({"class": label, "schema": enc(adapter.json_schema())})
        for index, item in enumerate(inputs):
            validation.append({"class": label, "input_index": index, "input": tagged(item),
                               "dspy": observe(lambda v: parse_value(v, cls), item),
                               "pydantic": observe(adapter.validate_python, item)})
        for member_name, item in cls.__members__.items():
            views.append({"class": label, "requested_name": member_name, "value": tagged(item),
                          "str": str(item), "repr": repr(item), "bool": bool(item),
                          "deepcopy_same_instance": copy.deepcopy(item) is item,
                          "raw": tagged(adapter.dump_python(item)),
                          "serialized": enc(serialize_for_json(item)),
                          "json": adapter.dump_json(item).decode(),
                          "mask_same_instance": recursive_mask(item) is item,
                          "mask": tagged(recursive_mask(item))})
    for left_label, left_cls in CLASSES.items():
        for left in left_cls:
            for right_label, right_cls in CLASSES.items():
                for right in right_cls:
                    equality.append({"left": tagged(left), "right": tagged(right), "equal": left == right,
                                     "same_instance": left is right})
    class Typed(dspy.Signature):
        question: str = dspy.InputField()
        color: CLASSES["color_a"] = dspy.OutputField()
        number: int = dspy.OutputField()
        payload: Any = dspy.OutputField()

    predictions = []
    for adapter, response in [
        (dspy.ChatAdapter(), "[[ ## color ## ]]\nred\n[[ ## number ## ]]\n100000000000000000000000000000000000000000000000000\n[[ ## payload ## ]]\nb'hi'\n[[ ## completed ## ]]"),
        (dspy.JSONAdapter(), '{"color":"red","number":100000000000000000000000000000000000000000000000000,"payload":"b\\u0027hi\\u0027"}'),
        (dspy.XMLAdapter(), "<color>red</color><number>100000000000000000000000000000000000000000000000000</number><payload>b'hi'</payload>"),
    ]:
        lm = ref.KeywordLM([], response)
        with dspy.context(lm=lm, adapter=adapter):
            result = dspy.Predict(Typed)(question="Decode")
        predictions.append({"adapter": type(adapter).__name__, "response": response,
                            "outputs": tagged(result.toDict()), "requests": lm.calls})
    receipt = {"evidence_kind": "actual_dspy_enum_calls", "provenance": repair.provenance(),
               "schemas": schemas, "validation": validation, "scalar_validation": scalar_validation,
               "views": views, "equality": equality, "predictions": predictions}
    data = json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
        print(f"Observed {len(validation) * 2} enum validations, {len(views)} views, "
              f"{len(equality)} equality pairs, {len(schemas)} schemas; "
              f"{len(scalar_validation) * 2} scalar validations")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
