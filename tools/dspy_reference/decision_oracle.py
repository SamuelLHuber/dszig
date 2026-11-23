#!/usr/bin/env python3
"""Observe installed DSPy 3.5 decisions through actual Python calls only.

Use tools/dspy_reference/.venv/bin/python decision_oracle.py --output RECEIPT.
The development oracle does not import or execute the native implementation.
Scripted engines use the established reference.ReferenceLM pattern: actual DSPy
adapters, Predict, BaseLM and lm15 requests, without external model inference.
Python values carry explicit python_type tags; *_json fields are raw wire JSON.
Every Source failure remains an observation with its actual exception details.
"""

import argparse
import asyncio
import copy
import decimal
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Annotated, Literal, Optional, get_args

import dspy
import pydantic
from pydantic import BaseModel, TypeAdapter

import reference as ref
from dspy._vendor.lm15 import serde
from dspy.adapters.decision import DecisionAdapter, evidence_type, record_evidence, resolve_adapter
from dspy.adapters.decision_state import DecisionState
from dspy.adapters.types.bounding_box import BoundingBox
from dspy.adapters.types.decision import Choice, Noul, Score, decision_type
from dspy.adapters.utils import format_field_value, get_annotation_name, parse_value, serialize_for_json
from dspy.lm15 import Message, Request, Response, Usage
from dspy.predict.predict import serialize_object

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")

MODEL_IDS = {}
MODELS = []
ROWS = []


def qualname(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def kind_name(cls):
    return "noul" if issubclass(cls, Noul) else "score" if issubclass(cls, Score) else "choice"


def enc(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"python_type": "float", "repr": repr(value)}
    if isinstance(value, (bytes, bytearray)):
        return {"python_type": type(value).__name__, "hex": bytes(value).hex()}
    if isinstance(value, decimal.Decimal):
        return {"python_type": "Decimal", "str": str(value)}
    if isinstance(value, BaseModel):
        result = {"python_type": type(value).__name__, "class": qualname(type(value)),
                  "fields": [[k, enc(v)] for k, v in value.__dict__.items()]}
        if type(value) in MODEL_IDS:
            result["model"] = MODEL_IDS[type(value)]
            result["python_type"] = (kind_name(type(value)).capitalize()
                                     if type(value).__bases__[0] in (Noul, Score, Choice)
                                     or type(value) in (Noul, Score, Choice) else type(value).__name__)
        return result
    if isinstance(value, dspy.Prediction):
        return {"python_type": "Prediction", "fields": [[k, enc(value[k])] for k in value.keys()]}
    if isinstance(value, dspy.Example):
        return {"python_type": "Example", "fields": [[k, enc(v)] for k, v in value.items()]}
    if isinstance(value, Request):
        return {"python_type": "lm15.Request", "request_json": serde.request_to_dict(value)}
    if isinstance(value, Response):
        return {"python_type": "lm15.Response", "response_json": serde.response_to_dict(value, include_provider_data=True)}
    if isinstance(value, dict):
        return {"python_type": "dict", "items": [[enc(k), enc(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"python_type": "tuple", "items": [enc(v) for v in value]}
    if isinstance(value, list):
        return [enc(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"python_type": type(value).__name__, "items": sorted((enc(v) for v in value), key=repr)}
    if isinstance(value, type):
        return {"python_type": "type", "class": qualname(value), "model": MODEL_IDS.get(value)}
    if isinstance(value, BaseException):
        return {"python_type": type(value).__name__, "message": str(value)}
    if value is Ellipsis:
        return {"python_type": "ellipsis"}
    if hasattr(value, "__origin__") or type(value).__module__ == "typing":
        return {"python_type": "typing", "repr": repr(value)}
    raise TypeError(f"Oracle needs explicit encoding for {type(value)!r}")


def error_value(exc):
    result = {"type": type(exc).__name__, "module": type(exc).__module__, "message": str(exc)}
    if isinstance(exc, pydantic.ValidationError):
        result["validation_errors"] = [
            {k: enc(v) if k != "ctx" else {str(a): str(b) for a, b in v.items()}
             for k, v in item.items()} for item in exc.errors(include_url=False)]
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        result["cause"] = {"type": type(cause).__name__, "message": str(cause)}
    return result


def observe(fn, raw_json=False):
    try:
        value = fn()
    except Exception as exc:
        return {"ok": False, "error": error_value(exc)}
    # Encoding errors are harness defects, never misreported as Source errors.
    return {"ok": True, "value_json" if raw_json else "value": value if raw_json else enc(value)}


def add(group, case_id, fn, args=None, model=None, raw_json=False, **extra):
    row = {"group": group, "id": f"{group}/{case_id}", "input": enc(args), **extra}
    if model is not None:
        row["model"] = MODEL_IDS[model]
    row["result"] = observe(fn, raw_json=raw_json)
    ROWS.append(row)
    return row


def register(model_id, cls, origin="configured"):
    if cls in MODEL_IDS:
        raise RuntimeError(f"Registering the same Source class twice: {model_id}")
    MODEL_IDS[cls] = model_id
    criteria = cls.criteria()
    kind = kind_name(cls)
    labels = ["true", "false"] if kind == "noul" else (
        list(range(len(criteria))) if kind == "score" and criteria is not None else
        list(criteria) if kind == "choice" and criteria is not None else [])
    values = [True, False] if kind == "noul" else (
        list(get_args(cls.model_fields["value"].annotation)) if kind == "choice" else labels)
    MODELS.append({"id": model_id, "kind": kind, "origin": origin, "class": qualname(cls),
                   "criteria_json": cls._criteria_json, "criteria": enc(criteria),
                   "labels_json": labels, "values": enc(values), "description": cls.description(),
                   "annotation": repr(cls.model_fields["value"].annotation)})
    return cls


def setup_models():
    models = {
        "bare_noul": register("bare_noul", Noul, "bare"),
        "bare_score": register("bare_score", Score, "bare"),
        "bare_choice": register("bare_choice", Choice, "bare"),
        "noul_binary": register("noul_binary", Noul[(False, ["usable", None]), (True, {"text": "bloqué 日本", "n": 1})]),
        "noul_true": register("noul_true", Noul[(True, "blocked")]),
        "score_three": register("score_three", Score["poor", "fair", "excellent"]),
        "score_json": register("score_json", Score[{"a": ["é", 1]}, ["good", {"why": None}]]),
        "choice_text": register("choice_text", Choice[("billing", "Payment issue"), ("technical", "Product bug")]),
        "choice_typed": register("choice_typed", Choice[(1, "integer"), (True, {"typed": "bool"}),
                                                        (None, None), ("ready", ["ready", "日本"])]),
        "choice_int": register("choice_int", Choice[(1, "one"), (0, "zero")]),
        "choice_bool": register("choice_bool", Choice[(True, "one"), (False, "zero")]),
        "choice_none": register("choice_none", Choice[(None, "empty"), ("None?", "not empty")]),
        "choice_wide": register("choice_wide", Choice[(10**50, "wide"), (0, "zero")]),
    }
    class SubNoul(Noul):
        pass
    class SubScore(Score):
        pass
    class SubChoice(Choice):
        pass
    models.update({"sub_noul": register("sub_noul", SubNoul, "dynamic_subclass"),
                   "sub_score": register("sub_score", SubScore, "dynamic_subclass"),
                   "sub_choice": register("sub_choice", SubChoice, "dynamic_subclass")})
    return models


def model_views(model):
    adapter = TypeAdapter(type(model))
    operations = {
        "str": lambda: str(model), "repr": lambda: repr(model), "format": model.format,
        "serialize_model_without_handler": model.serialize_model,
        "model_dump": model.model_dump,
        "model_dump_round_trip": lambda: model.model_dump(round_trip=True),
        "model_dump_json_mode": lambda: model.model_dump(mode="json"),
        "model_dump_json_mode_round_trip": lambda: model.model_dump(mode="json", round_trip=True),
        "model_dump_json": model.model_dump_json,
        "model_dump_json_round_trip": lambda: model.model_dump_json(round_trip=True),
        "typeadapter_dump_python": lambda: adapter.dump_python(model),
        "typeadapter_dump_json": lambda: adapter.dump_json(model),
        "serialize_for_json": lambda: serialize_for_json(model),
        "serialize_object": lambda: serialize_object(model),
        "bool": lambda: bool(model), "float": lambda: float(model),
        "iter": lambda: list(iter(model)), "hash_admissible": lambda: (hash(model), True)[1],
        "self_eq": lambda: model == model, "deepcopy_eq": lambda: copy.deepcopy(model) == model,
        "dict_eq": lambda: model == model.model_dump(round_trip=True),
    }
    return {name: observe(fn) for name, fn in operations.items()}


def model_observation(case_id, cls, entrypoint, args, fn, input_json=None):
    row = {"group": "models", "id": f"models/{MODEL_IDS[cls]}/{case_id}/{entrypoint}",
           "model": MODEL_IDS[cls], "entrypoint": entrypoint, "input": enc(args)}
    if input_json is not None:
        row["input_json_text"] = input_json
    try:
        result = fn()
    except Exception as exc:
        row["result"] = {"ok": False, "error": error_value(exc)}
    else:
        row["result"] = {"ok": True, "value": enc(result)}
        row["identity"] = {"result_is_input": result is args, "exact_class": type(result) is cls,
                           "result_is_value_input": isinstance(args, dict) and result.value is args.get("value")}
        row["views"] = model_views(result)
    ROWS.append(row)


SCALARS = [
    ("none", None), ("false", False), ("true", True), ("zero_int", 0), ("one_int", 1),
    ("negative_int", -1), ("two_int", 2), ("zero_float", 0.0), ("negative_zero", -0.0),
    ("one_float", 1.0), ("half", 0.5), ("fraction", 1.5), ("nan", float("nan")),
    ("positive_inf", float("inf")), ("negative_inf", float("-inf")),
    ("wide_int", 10**50), ("negative_wide_int", -(10**50)), ("u64", 2**64),
    ("large_float", 1e30), ("empty", ""), ("true_text", "true"), ("false_text", "False"),
    ("one_text", "1"), ("half_text", "0.5"), ("padded", " 1 "), ("unicode", "日本"),
    ("billing", "billing"), ("ready", "ready"), ("nan_text", "nan"), ("inf_text", "Infinity"),
    ("bytes", b"1"), ("invalid_bytes", b"\xff"), ("bytearray", bytearray(b"1")),
    ("decimal", decimal.Decimal("0.5")), ("decimal_nan", decimal.Decimal("NaN")),
    ("list", [1]), ("dict", {"x": 1}), ("tuple", (1,)),
]


def model_cases(models):
    for model_id, cls in models.items():
        kind = kind_name(cls)
        default = True if kind == "noul" else 1.0 if kind == "score" else (
            get_args(cls.model_fields["value"].annotation)[0] if cls.criteria() is not None else "billing")
        baseline = {"value": default, "confidence": 0.75}
        inputs = [("minimal", baseline), ("missing_all", {}), ("missing_value", {"confidence": 0.5}),
                  ("missing_confidence", {"value": default}), ("extra", {**baseline, "extra": 1}),
                  ("confidence_null", {**baseline, "confidence": None}),
                  ("reordered", {"confidence": 0.75, "value": default})]
        # Named cases remain stable if later Source cases are appended.
        inputs += [(f"value/{name}", {**baseline, "value": value}) for name, value in SCALARS]
        inputs += [(f"confidence/{name}", {**baseline, "confidence": value}) for name, value in SCALARS]
        if kind == "noul":
            inputs += [(f"probability/{name}", {**baseline, "probability": value}) for name, value in SCALARS]
        elif kind == "score":
            inputs += [(f"level/{name}", {**baseline, "level": value}) for name, value in SCALARS]
            inputs += [("probabilities/" + name, {**baseline, "probabilities": value}) for name, value in [
                ("null", None), ("empty", {}), ("int_keys", {0: 0.25, 1: 0.75}),
                ("str_keys", {"0": "0.25", "1": True}), ("bool_key", {True: 0.5}),
                ("bad_key", {"x": 0.5}), ("wide_key", {10**50: 1}),
                ("negative_key", {-1: 1}), ("nan", {0: float("nan")}),
                ("over_one", {0: 1.1}), ("negative", {0: -0.1}), ("pairs", [(0, 1)]),
                ("list", [0.5, 0.5]), ("decimal", {0: decimal.Decimal("0.5")})]]
        else:
            inputs += [("probabilities/" + name, {**baseline, "probabilities": value}) for name, value in [
                ("null", None), ("empty", {}), ("valid", {"billing": 0.25, "technical": 0.75}),
                ("unknown", {"unknown": 1}), ("int_key", {1: 0.5}), ("bool_key", {True: 0.5}),
                ("bytes_key", {b"billing": 1}), ("null_key", {None: 1}),
                ("nan", {"billing": float("nan")}), ("over_one", {"billing": 1.1}),
                ("negative", {"billing": -0.1}), ("list", [0.5, 0.5])]]
        # Full validation matrices on canonical types; dynamic subclasses remain genuine but bounded.
        if model_id.startswith("sub_"):
            inputs = inputs[:7] + inputs[7:12]
        for name, value in inputs:
            methods = {
                "constructor": lambda v=value: cls(**v),
                "model_validate": lambda v=value: cls.model_validate(v),
                "model_validate_strict": lambda v=value: cls.model_validate(v, strict=True),
                "model_validate_lax": lambda v=value: cls.model_validate(v, strict=False),
                "typeadapter_python": lambda v=value: TypeAdapter(cls).validate_python(v),
            }
            for method, fn in methods.items():
                model_observation(name, cls, method, value, fn)
            try:
                text = json.dumps(value, ensure_ascii=True)
            except (TypeError, ValueError):
                continue
            for method, fn in {
                "model_validate_json": lambda t=text: cls.model_validate_json(t),
                "model_validate_json_strict": lambda t=text: cls.model_validate_json(t, strict=True),
                "typeadapter_json": lambda t=text: TypeAdapter(cls).validate_json(t),
            }.items():
                model_observation(name, cls, method, value, fn, text)
        # The mixed wide/small Literal genuinely refuses its oversized member.
        # Keep every refusal above; use its separately valid zero for identity calls.
        identity_baseline = {**baseline, "value": 0} if model_id == "choice_wide" else baseline
        original = cls(**identity_baseline)
        for name, value in [("existing", original), ("copy", copy.copy(original)),
                            ("deepcopy", copy.deepcopy(original)), ("list_pairs", list(baseline.items())),
                            ("none", None), ("primitive", default), ("raw_json", json.dumps(baseline))]:
            for method, fn in {
                "model_validate": lambda v=value: cls.model_validate(v),
                "typeadapter_python": lambda v=value: TypeAdapter(cls).validate_python(v),
                "constructor_positional": lambda v=value: cls(v),
                "parse_value": lambda v=value: parse_value(v, cls),
            }.items():
                model_observation(f"boundary/{name}", cls, method, value, fn)
        for mode in ("validation", "serialization"):
            add("schemas", f"{model_id}/{mode}", lambda m=mode: cls.model_json_schema(mode=m),
                {"mode": mode}, cls, raw_json=True)
        add("annotation_names", model_id, lambda: get_annotation_name(cls), model=cls)


def factory_cases(models):
    noul_cases = [
        ("empty", ()), ("single_true", (True, "blocked")), ("single_false", (False, None)),
        ("both", ((False, "usable"), (True, "blocked"))),
        ("reverse", ((True, "blocked"), (False, "usable"))),
        ("duplicate", ((True, "one"), (True, "two"))),
        ("integer", ((1, "one"), (0, "zero"))), ("list_outer", [(True, "yes")]),
        ("list_pair", ([True, "yes"],)), ("malformed_pair", ((True,),)),
    ]
    score_cases = [("empty", ()), ("one", ("only",)), ("string", "only"), ("list", ["poor", "good"]),
                   ("two", ("poor", "good")), ("three", ("poor", "fair", "good")),
                   ("null", (None, "good")), ("integer", (1, "good")),
                   ("json", ({"x": [1, True]}, ["good", None])), ("eleven", tuple(map(str, range(11))))]
    choice_cases = [("empty", ()), ("single", ("x", "desc")),
                    ("typed", ((1, "int"), (True, "bool"), (None, "null"), ("ready", "text"))),
                    ("str_int_ambiguous", ((1, "int"), ("1", "str"))),
                    ("str_bool_ambiguous", ((True, "bool"), ("True", "str"))),
                    ("str_none_ambiguous", ((None, "null"), ("None", "str"))),
                    ("same_value", (("a", "one"), ("a", "two"))),
                    ("bool_int", ((False, "bool"), (0, "int"))),
                    ("float", ((1.0, "float"), (2, "int"))), ("bytes", ((b"a", "bytes"),)),
                    ("bytearray", ((bytearray(b"a"), "bytes"),)),
                    ("decimal", ((decimal.Decimal("1"), "decimal"),)),
                    ("wide", ((10**50, "wide"), (-(10**50), "negative"))),
                    ("empty_label", (("", "empty"), ("x", "other"))),
                    ("list_outer", [("x", "desc")]), ("bad_pair", (("a",),))]
    descriptions = [("none", None), ("string", "日本"), ("dict", {"b": 1, "a": [True, None]}),
                    ("list", ["é", {"x": None}]), ("bool", False), ("int", 1), ("float", 0.5),
                    ("nan_nested", {"x": float("nan")}), ("inf_nested", [float("inf")]),
                    ("nonstring_key", {1: "x"}), ("bytes_nested", {"x": b"a"}),
                    ("decimal_nested", [decimal.Decimal("0.5")]), ("tuple", ("a",)),
                    ("bytearray", bytearray(b"a"))]
    for name, desc in descriptions:
        noul_cases.append(("description/" + name, (True, desc)))
        score_cases.append(("description/" + name, (desc, "other")))
        choice_cases.append(("description/" + name, ("x", desc)))
    for kind, cls, cases in [("noul", Noul, noul_cases), ("score", Score, score_cases), ("choice", Choice, choice_cases)]:
        for name, options in cases:
            def run():
                result = cls[options]
                return {"class": qualname(result), "criteria_json": result._criteria_json,
                        "criteria": result.criteria(), "description": result.description(),
                        "same_again": result is cls[options],
                        "value_annotation": repr(result.model_fields["value"].annotation)}
            add("factories", kind + "/" + name, run, options, entrypoint="__class_getitem__")
    def snapshots():
        desc = {"nested": ["first"], "日本": None}
        cls = Choice[("snapshot", desc), ("other", "other")]
        before = cls.criteria()
        desc["nested"].append("mutated")
        returned = cls.criteria()
        returned["snapshot"]["nested"].append("returned mutation")
        return {"before": before, "after": cls.criteria(), "criteria_json": cls._criteria_json,
                "different_after_argument_mutation": cls is not Choice[("snapshot", desc), ("other", "other")],
                "different_description_key_order": Choice[("x", {"a": 1, "b": 2})] is not Choice[("x", {"b": 2, "a": 1})],
                "noul_order_same": Noul[(True, "x"), (False, "y")] is Noul[(False, "y"), (True, "x")],
                "score_order_distinct": Score["x", "y"] is not Score["y", "x"],
                "choice_order_distinct": Choice[("x", "x"), ("y", "y")] is not Choice[("y", "y"), ("x", "x")],
                "bool_int_cache_distinct": Choice[(True, "x")] is not Choice[(1, "x")],
                "false_zero_cache_distinct": Choice[(False, "x")] is not Choice[(0, "x")]}
    add("factory_identity", "snapshot_order_typed_cache", snapshots)
    def eviction():
        original = Choice[("eviction_original", "x")]
        instance = original(value="eviction_original", confidence=0.5)
        for i in range(257):
            Choice[(f"eviction_{i}", "x")]
        recreated = Choice[("eviction_original", "x")]
        return {"same_class_after_eviction": original is recreated,
                "same_name": qualname(original) == qualname(recreated),
                "instance_equal_after_eviction": instance == recreated(value="eviction_original", confidence=0.5),
                "existing_instance_validate_identity": original.model_validate(instance) is instance,
                "cross_class_validate": observe(lambda: recreated.model_validate(instance)),
                "genuine_original": instance}
    add("factory_identity", "cache_eviction", eviction, features=["cache_eviction", "dynamic_class"])


def protocol_cases(models):
    for kind, ids, values in [
        ("noul", ["bare_noul", "noul_binary", "noul_true", "sub_noul"], [True, False]),
        ("score", ["bare_score", "score_three", "sub_score"], [1.0, 0.0]),
        ("choice", ["bare_choice", "choice_int", "choice_bool", "sub_choice"], [1, True, "1", None]),
    ]:
        instances = []
        for model_id in ids:
            for value in values:
                cls = models[model_id]
                try:
                    instance = cls(value=value, confidence=0.5)
                except Exception:
                    continue
                instances.append((f"{model_id}/{type(value).__name__}/{repr(value)}", instance))
        for left_id, left in instances:
            for right_id, right in instances:
                add("equality", f"{kind}/{left_id}/{right_id}",
                    lambda a=left, b=right: {"left_eq_right": a == b, "right_eq_left": b == a,
                                           "same_object": a is b, "same_class": type(a) is type(b)},
                    (left, right))
    cls = models["score_three"]
    for name, mutate in [
        ("value_out_of_range", lambda m: setattr(m, "value", 99)),
        ("confidence_nonfinite", lambda m: setattr(m, "confidence", float("nan"))),
        ("unknown_field", lambda m: setattr(m, "unknown", 1)),
        ("delete_value", lambda m: delattr(m, "value")),
        ("probabilities_late_invalid", lambda m: setattr(m, "probabilities", {"bad": -1})),
    ]:
        model = cls(value=1, confidence=0.5)
        mutation = observe(lambda: mutate(model))
        add("model_mutation", name, lambda: model, model=cls, mutation=mutation, views=model_views(model))
    for name, value in [("nonfinite", float("nan")), ("large", 99), ("string", "1")]:
        model = cls.model_construct(value=value, confidence=0.5, probabilities=None, level=None)
        add("model_construct", name, lambda: model, {"value": value}, cls, views=model_views(model))


def make_signature(name, outputs, descriptions=True, instructions="Judge the supplied input. Preserve typed decisions."):
    fields = {"text": (str, dspy.InputField(desc="Input text, including 日本."))}
    for field_name, annotation in outputs.items():
        field = dspy.OutputField(desc=f"Evaluate {field_name}." if descriptions else None)
        fields[field_name] = (annotation, field)
    return dspy.Signature(fields, instructions=instructions).with_updated_fields("text", prefix="Text:")


def state_view(state):
    return {"types": {name: MODEL_IDS.get(kind, {"class": qualname(kind), "criteria_json": kind._criteria_json})
                      for name, kind in state.types.items()},
            "fields": state.fields,
            "criteria": {name: state.get_criteria(name) for name in state.types},
            "questions": {name: observe(lambda n=name, k=kind: state._question(n, state.signature.output_fields[n], k))
                          for name, kind in state.types.items()}}


def state_cases(models):
    n, s, c = models["noul_binary"], models["score_three"], models["choice_text"]
    sig = make_signature("DecisionTask", {"flag": n, "rating": s, "category": c})
    native_sig = make_signature("NativeTask", {"flag": Annotated[bool, n],
                                                "category": Annotated[Literal["billing", "technical"], c]})
    signatures = {
        "rich": sig, "native_metadata": native_sig,
        "native_plain": make_signature("NativePlain", {"flag": bool, "category": Literal["billing", "technical"]}),
        "bare": make_signature("Bare", {"flag": Noul, "rating": Score, "category": Choice}),
        "nested": make_signature("Nested", {"flags": list[n], "optional": Optional[s]}),
        "box": make_signature("Box", {"flag": n, "box": BoundingBox}),
        "ordinary": make_signature("Ordinary", {"answer": str}),
    }
    for name, signature in signatures.items():
        for system_one in (False, True):
            for config_name, fields in [("empty", {}), ("flag", {"flag": {}})]:
                add("state_init", f"{name}/{system_one}/{config_name}",
                    lambda a=signature, b=fields, so=system_one: state_view(DecisionState(a, b, system_one=so)),
                    {"fields": fields, "system_one": system_one}, signature=name)
        for field_name, field in signature.output_fields.items():
            add("decision_type", f"{name}/{field_name}", lambda f=field: decision_type(f),
                {"annotation": repr(field.annotation), "metadata": field.metadata})
    invalids = [
        ("fields_none", None), ("fields_list", []), ("unknown", {"missing": {}}),
        ("config_none", {"flag": None}), ("config_list", {"flag": []}),
        ("unknown_parameter", {"flag": {"bogus": 1}}), ("wrong_parameter", {"flag": {"weights": {}}}),
    ]
    for name, value in SCALARS:
        invalids.append(("threshold/" + name, {"flag": {"threshold": value}}))
    for name, value in [
        ("default", [0.5, 1.5]), ("zero", [0, 1]), ("end", [1, 2]), ("reversed", [1.5, 0.5]),
        ("equal", [1, 1]), ("empty", []), ("short", [0.5]), ("tuple", (0.5, 1.5)),
        ("false", [False, 1]), ("strings", ["0.5", "1.5"]), ("nan", [float("nan"), 1]),
        ("inf", [0.5, float("inf")]), ("none", None), ("boolean", False),
    ]:
        invalids.append(("cuts/" + name, {"rating": {"cuts": value}}))
    for name, value in [
        ("empty", {}), ("partial_zero", {"billing": 0}), ("all_zero", {"billing": 0, "technical": 0}),
        ("partial", {"billing": 2}), ("unknown", {"x": 1}), ("int_key", {1: 1}),
        ("negative", {"billing": -1}), ("nan", {"billing": float("nan")}),
        ("inf", {"billing": float("inf")}), ("false", {"billing": False}),
        ("string", {"billing": "1"}), ("none", None), ("list", []), ("boolean", False),
        ("decimal", {"billing": decimal.Decimal("1")}),
    ]:
        invalids.append(("weights/" + name, {"category": {"weights": value}}))
    for name, value in SCALARS:
        invalids.append(("instructions/" + name, {"flag": {"instructions": value}}))
    for target, cases in {
        "flag": [("null", None), ("empty", {}), ("true_only", {"true": "yes"}),
                 ("typed_key", {True: "yes"}), ("unknown", {"yes": "x"}), ("numeric_desc", {"true": 1})],
        "rating": [("valid", ["low", {"mid": 1}, ["high"]]), ("null", None), ("dict", {}),
                   ("short", ["low", "high"]), ("null_entry", ["low", None, "high"]),
                   ("numeric_entry", ["low", 1, "high"])],
        "category": [("valid", {"billing": None, "technical": ["bug"]}), ("partial", {"billing": "x"}),
                     ("extra", {"billing": "x", "technical": "y", "other": None}),
                     ("numeric", {"billing": 1, "technical": "y"}), ("null", None)],
    }.items():
        for name, value in cases:
            invalids.append((f"criteria/{target}/{name}", {target: {"criteria": value}}))
    for name, fields in invalids:
        add("state_config", name, lambda f=fields: state_view(DecisionState(sig, f)), fields)
    for name, fields in [("null_instructions", {"flag": {"instructions": None}}),
                         ("empty_instructions", {"flag": {"instructions": ""}}), ("none", {})]:
        signature = make_signature("Undescribed", {"flag": n}, descriptions=False)
        add("questions", name, lambda f=fields: state_view(DecisionState(signature, f)), fields)
    def mutation():
        fields = {"rating": {"cuts": [0.25, 1.75], "instructions": {"steps": ["first"]}}}
        state = DecisionState(sig, fields)
        fields["rating"]["cuts"][0] = 99
        fields["rating"]["instructions"]["steps"].append("external")
        question = state._question("rating", sig.output_fields["rating"], s)
        question["instructions"]["steps"].append("question")
        criteria = state.get_criteria("rating")
        criteria.append("external")
        return state_view(state)
    add("state_mutation", "snapshot_inputs_and_returned_values", mutation)
    # Signature override validation includes order and typed Literal answer spaces.
    for name, replacement in [
        ("same", sig), ("instructions", sig.with_instructions("Different instructions")),
        ("choice_reordered", make_signature("Reordered", {"flag": n, "rating": s,
             "category": Choice[("technical", "Product bug"), ("billing", "Payment issue")]})),
        ("criteria_changed", make_signature("Changed", {"flag": n, "rating": s,
             "category": Choice[("billing", "different"), ("technical", "Product bug")]})),
        ("missing", make_signature("Missing", {"flag": n, "rating": s})),
    ]:
        add("signature_override", name,
            lambda a=replacement: state_view(DecisionState(a, {"category": {}}, declared_signature=sig)),
            {"configured": "category"})
    return sig, native_sig


def decoding_cases(models, sig):
    baseline = {"flag": {"noul": 0.75}, "rating": {"probabilities": {0: 0.25, 1: 0.25, 2: 0.5}, "confidence": 0.8},
                "category": {"probabilities": {"billing": 0.4, "technical": 0.6}, "confidence": 0.9}}
    for field_name, cls in [("flag", models["noul_binary"]), ("rating", models["score_three"]),
                            ("category", models["choice_text"])]:
        signature = make_signature("Single", {field_name: cls})
        cases = [("baseline", baseline[field_name]), ("missing", {}), ("null", None),
                 ("extra", {**baseline[field_name], "extra": True})]
        if field_name == "flag":
            cases += [("probability/" + name, {"noul": value}) for name, value in SCALARS]
        elif field_name == "rating":
            cases += [("distribution/" + name, {"probabilities": value, "confidence": 0.8}) for name, value in [
                ("empty", {}), ("missing", {0: 0.5, 1: 0.5}), ("extra", {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.4}),
                ("zero", {0: 0, 1: 0, 2: 0}), ("unnormalized", {0: 1, 1: 1, 2: 1}),
                ("string_keys", {"0": 0.2, "1": 0.3, "2": 0.5}),
                ("reversed", {2: 0.5, 1: 0.25, 0: 0.25}),
                ("negative", {0: -0.1, 1: 0.1, 2: 1}),
                ("nan", {0: float("nan"), 1: 0, 2: 1}), ("inf", {0: 0, 1: 0, 2: float("inf")}),
                ("string_values", {0: "0.1", 1: "0.2", 2: "0.7"}),
                ("false_values", {0: False, 1: True, 2: False}),
                ("boolean_key", {False: 0.2, True: 0.3, 2: 0.5}),
                ("float_key", {0.0: 0.2, 1.0: 0.3, 2.0: 0.5}),
                ("list", [0.2, 0.3, 0.5]), ("pairs", [(0, 0.2), (1, 0.3), (2, 0.5)]),
                ("none", None), ("scalar", False),
            ]]
            for name, probabilities in [
                ("ascending", {0: 1.0, 1: 1e-16, 2: 1e-16}),
                ("descending", {2: 1e-16, 1: 1e-16, 0: 1.0}),
                ("near_cut", {0: 0.25, 1: 0.5, 2: 0.25}),
            ]:
                cases.append(("sum_order/" + name, {"probabilities": probabilities, "confidence": 1}))
        else:
            cases += [("distribution/" + name, {"probabilities": value, "confidence": 0.8}) for name, value in [
                ("tie", {"billing": 0.5, "technical": 0.5}),
                ("reversed_tie", {"technical": 0.5, "billing": 0.5}),
                ("empty", {}), ("missing", {"billing": 1}), ("extra", {"billing": 0.5, "technical": 0.5, "x": 0}),
                ("zero", {"billing": 0, "technical": 0}), ("unnormalized", {"billing": 1, "technical": 1}),
                ("negative", {"billing": -0.1, "technical": 1}),
                ("nan", {"billing": float("nan"), "technical": 0.5}),
                ("inf", {"billing": float("inf"), "technical": 0.5}),
                ("strings", {"billing": "0.5", "technical": "0.5"}),
                ("false", {"billing": False, "technical": True}), ("list", [0.5, 0.5]), ("null", None),
            ]]
        for name, answer in cases:
            for boundary in ("state", "adapter"):
                state = DecisionState(signature, {})
                fn = (lambda a=answer: state._decode({field_name: copy.deepcopy(a)})) if boundary == "state" else (
                    lambda a=answer: DecisionAdapter(dspy.JSONAdapter(), state, False)._decode([{field_name: copy.deepcopy(a)}]))
                add("decode", f"{field_name}/{name}/{boundary}", fn, answer, entrypoint=boundary)
        if field_name == "flag":
            for threshold in (0, 0.5, 0.75, 1):
                state = DecisionState(signature, {field_name: {"threshold": threshold}})
                for probability in (0, 0.5, 0.75, 1):
                    add("decode_threshold", f"{threshold}/{probability}",
                        lambda p=probability: state._decode({field_name: {"noul": p}}),
                        {"threshold": threshold, "probability": probability})
        if field_name == "category":
            for name, probabilities, weights in [
                ("weighted_tie_prefers_raw", {"billing": 0.25, "technical": 0.5}, {"billing": 2}),
                ("raw_tie_declaration_order", {"technical": 0.5, "billing": 0.5}, {}),
                ("zero_raw_winner", {"billing": 0.4, "technical": 0.6}, {"technical": 0}),
                ("no_positive_mass", {"billing": 0, "technical": 1}, {"technical": 0}),
            ]:
                state = DecisionState(signature, {field_name: {"weights": weights}})
                add("decode_weights", name, lambda p=probabilities: state._decode(
                    {field_name: {"probabilities": p, "confidence": 0.8}}),
                    {"probabilities": probabilities, "weights": weights})
    for model_id in ("choice_typed", "choice_bool", "choice_int", "choice_none", "choice_wide"):
        cls = models[model_id]
        labels = list(cls.criteria())
        signature = make_signature("Typed", {"answer": cls})
        state = DecisionState(signature, {})
        for winner in labels:
            answer = {"probabilities": {label: int(label == winner) for label in labels}, "confidence": 1}
            add("decode_typed", model_id + "/" + winner,
                lambda a=answer: state._decode({"answer": a}), answer, cls)
    for name, completion in [
        ("missing_output", {"flag": {"noul": 1}}), ("extra_output", {**baseline, "extra": "kept"}),
        ("mixed_score_keys", {**baseline, "rating": {"probabilities": {"0": 0.2, 0: 0.2, "1": 0.3, "2": 0.3}, "confidence": 0.5}}),
        ("invalid_later_field", {**baseline, "category": {"probabilities": {}, "confidence": 0.5}}),
    ]:
        state = DecisionState(sig, {})
        def run(a=completion):
            with record_evidence() as log:
                result = observe(lambda: DecisionAdapter(dspy.JSONAdapter(), state, False)._decode([a]))
            return {"decode": result, "evidence": [(name, item) for _, name, item in log], "completion_after": a}
        add("decode_boundaries", name, run, completion)


def evidence_cases(models):
    for model_id, cls in models.items():
        for mode in ("validation", "serialization"):
            add("evidence_schemas", f"{model_id}/{mode}",
                lambda c=cls, m=mode: evidence_type(c).model_json_schema(mode=m),
                {"mode": mode}, cls, raw_json=True)
        add("evidence_identity", model_id,
            lambda c=cls: evidence_type(c) is evidence_type(c), model=cls)
        try:
            evidence = evidence_type(cls)
        except Exception:
            continue
        if issubclass(cls, Noul):
            values = [(name, {"noul": value}) for name, value in SCALARS]
            values += [("missing", {}), ("extra", {"noul": 0.5, "extra": 1})]
        else:
            labels = list(map(str, range(len(cls.criteria())))) if issubclass(cls, Score) else list(cls.criteria())
            probabilities = dict.fromkeys(labels, 1 / len(labels))
            values = [("baseline", {"probabilities": probabilities, "confidence": 0.5}),
                      ("missing", {}), ("extra", {"probabilities": probabilities, "confidence": 0.5, "extra": 1}),
                      ("missing_probability", {"probabilities": {}, "confidence": 0.5}),
                      ("extra_probability", {"probabilities": {**probabilities, "extra": 0}, "confidence": 0.5}),
                      ("list_distribution", {"probabilities": list(probabilities.values()), "confidence": 0.5}),
                      ("unnormalized", {"probabilities": dict.fromkeys(labels, 1), "confidence": 0.5}),
                      ("zero_sum", {"probabilities": dict.fromkeys(labels, 0), "confidence": 0.5})]
            for name, scalar in SCALARS:
                values.append(("confidence/" + name, {"probabilities": probabilities, "confidence": scalar}))
                values.append(("probability/" + name, {"probabilities": {**probabilities, labels[0]: scalar}, "confidence": 0.5}))
        for name, value in values:
            for method, fn in {
                "python": lambda v=value: evidence.model_validate(v),
                "strict": lambda v=value: evidence.model_validate(v, strict=True),
            }.items():
                add("evidence_validate", f"{model_id}/{name}/{method}", fn, value, cls)


class ScriptedLM(ref.ReferenceLM):
    """Actual BaseLM executions with deterministic, in-process lm15 engines."""
    def __init__(self, completion="", answers=None, structured=False, asynchronous=False):
        super().__init__(model="dszig/decision-reference", model_type="chat", cache=False, num_retries=0,
                         temperature=0, max_tokens=1000)
        self.completion = completion
        self.answers = answers
        self.calls = []
        self.supports_decision_requests = answers is not None
        self.engine.supports_response_schema = structured
        if asynchronous:
            class AsyncEngine:
                async def complete(inner, request):
                    return self.complete(request)
            self._async_engine_spec = AsyncEngine()

    def complete(self, request):
        call = {"request_json": serde.request_to_dict(request)}
        if not self.supports_decision_requests:
            call["messages_json"] = ref.messages_for(request)
        self.calls.append(call)
        response = Response(id="decision-scripted", model=self.model, message=Message.assistant(self.completion),
                            finish_reason="stop", usage=Usage(input_tokens=13, output_tokens=7, total_tokens=20),
                            provider_data={"typesafe": {"answers": copy.deepcopy(self.answers)}} if self.answers is not None else None)
        call["response_json"] = serde.response_to_dict(response, include_provider_data=True)
        return response


def signature_view(signature):
    return {"instructions": signature.instructions,
            "fields": [{"name": name, "annotation": repr(field.annotation),
                        "metadata": enc(field.metadata), "json_schema_extra": enc(field.json_schema_extra)}
                       for name, field in signature.fields.items()],
            "schema_json": signature.model_json_schema()}


def completion_for(adapter_name, answers):
    if adapter_name == "chat":
        return "\n".join(f"[[ ## {name} ## ]]\n{json.dumps(answer, ensure_ascii=False)}" for name, answer in answers.items()) + "\n[[ ## completed ## ]]"
    if adapter_name == "json":
        return json.dumps(answers, ensure_ascii=False)
    return "".join(f"<{name}>{json.dumps(answer, ensure_ascii=False)}</{name}>" for name, answer in answers.items())


ADAPTERS = {"chat": dspy.ChatAdapter, "json": dspy.JSONAdapter, "xml": dspy.XMLAdapter}


def adapter_predict_cases(models, sig, native_sig):
    answers = {"flag": {"noul": 0.75},
               "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.8},
               "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}
    configured = {"flag": {"threshold": 0.8, "instructions": {"prompt": ["Is it blocked?", "日本"]},
                            "criteria": {"true": {"reason": "blocked"}, "false": ["usable"]}},
                  "rating": {"cuts": [0.25, 1.25], "criteria": ["low", {"middle": "fair"}, ["high"]]},
                  "category": {"weights": {"billing": 2}, "instructions": ["Route the issue"],
                               "criteria": {"billing": None, "technical": "bug"}}}
    demo = dspy.Example(text="demonstration", flag=models["noul_binary"](value=True, confidence=1),
                        rating=models["score_three"](value=2, confidence=1),
                        category=models["choice_text"](value="technical", confidence=1), ignored="omit").with_inputs("text")
    for signature_name, signature in [("rich", sig), ("native", native_sig)]:
        relevant_answers = {name: answers[name] for name in signature.output_fields}
        for config_name, fields in [("defaults", {}), ("configured", {k: v for k, v in configured.items() if k in signature.output_fields})]:
            for demo_name, demos in [("no_demos", []), ("demos", [demo])]:
                for adapter_name, adapter_cls in ADAPTERS.items():
                    for structured in ([False, True] if adapter_name == "json" else [False]):
                        suffix = f"{signature_name}/{config_name}/{demo_name}/{adapter_name}/structured_{structured}"
                        state = DecisionState(signature, fields)
                        adapter = DecisionAdapter(adapter_cls(), state, False)
                        completion = completion_for(adapter_name, relevant_answers)
                        lm = ScriptedLM(completion, structured=structured)
                        prepared = adapter._prepare(lm, signature, demos, {"text": "issue 日本", "ignored": "kept_until_adapter"}, {"temperature": 0.2})
                        add("generative_prepare", suffix,
                            lambda p=prepared: {"signature": signature_view(p["signature"]), "demos": p["demos"],
                                                "inputs": p["inputs"], "lm_kwargs": p["lm_kwargs"]},
                            {"fields": fields, "demos": demos}, adapter=adapter_name)
                        predictor = dspy.Predict(signature, temperature=0.2)
                        predictor.fields = copy.deepcopy(fields)
                        predictor.demos = demos
                        def run():
                            with dspy.context(lm=lm, adapter=adapter_cls(), track_usage=True), record_evidence() as log:
                                prediction = predictor(text="issue 日本", ignored="discard")
                            return {"outputs": prediction.toDict(),
                                    "completions": prediction.completions.toDict(),
                                    "usage": prediction.get_lm_usage(),
                                    "evidence": [{"predictor_is_current": owner is predictor, "field": name, "answer": evidence}
                                                 for owner, name, evidence in log]}
                        row = add("predict", suffix, run, {"fields": fields, "completion": completion, "demos": demos},
                                  adapter=adapter_name, signature=signature_name)
                        row["calls_json"] = lm.calls
                        row["history"] = enc(lm.history)
    # Native bool/Literal are opt-in on generative backends; explicit field config enables evidence.
    plain = make_signature("Plain", {"flag": bool, "category": Literal["billing", "technical"]})
    for adapter_name, adapter_cls in ADAPTERS.items():
        lm = ScriptedLM(completion_for(adapter_name, {"flag": answers["flag"], "category": answers["category"]}))
        predictor = dspy.Predict(plain)
        predictor.fields = {"flag": {"threshold": 0.8}, "category": {"weights": {"billing": 2}}}
        def run():
            with dspy.context(lm=lm, adapter=adapter_cls()):
                return predictor(text="plain native").toDict()
        row = add("predict_plain_configured", adapter_name, run, predictor.fields)
        row["calls_json"] = lm.calls
    # Nested annotations remain real nested models and emit upstream warnings, not evidence decoding.
    nested = make_signature("Nested", {"flags": list[models["noul_binary"]]})
    for adapter_name, adapter_cls in ADAPTERS.items():
        completion = completion_for(adapter_name, {"flags": [{"value": True, "confidence": 0.75}]})
        lm = ScriptedLM(completion)
        predictor = dspy.Predict(nested)
        def run():
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with dspy.context(lm=lm, adapter=adapter_cls()):
                    output = predictor(text="nested").toDict()
            return {"outputs": output, "warnings": [{"type": item.category.__name__, "message": str(item.message)} for item in captured]}
        row = add("nested_predict", adapter_name, run)
        row["calls_json"] = lm.calls
    # Actual async Predict and adapter paths, using a deterministic async engine.
    for adapter_name, adapter_cls in ADAPTERS.items():
        lm = ScriptedLM(completion_for(adapter_name, answers), asynchronous=True)
        predictor = dspy.Predict(sig)
        async def run_async():
            with dspy.context(lm=lm, adapter=adapter_cls()), record_evidence() as log:
                result = await predictor.acall(text="async")
            return {"outputs": result.toDict(), "evidence": [(name, item) for _, name, item in log]}
        row = add("async_predict", adapter_name, lambda: asyncio.run(run_async()))
        row["calls_json"] = lm.calls
    for name, fields in [("null_instructions", {"flag": {"instructions": None}}),
                         ("empty_instructions", {"flag": {"instructions": ""}})]:
        signature = make_signature("NoDesc", {"flag": Noul}, descriptions=False)
        lm = ScriptedLM('{"flag":{"noul":0.5}}')
        predictor = dspy.Predict(signature)
        predictor.fields = fields
        def run():
            with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
                return predictor(text="no desc").toDict()
        row = add("predict_question_boundaries", name, run, fields)
        row["calls_json"] = lm.calls


def system_one_cases(models, sig, native_sig):
    answers = {"flag": {"noul": 0.75}, "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.8},
               "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}
    fields = {"flag": {"threshold": 0.8}, "rating": {"cuts": [0.25, 1.25]}, "category": {"weights": {"billing": 2}}}
    for name, signature in [("rich", sig), ("native", native_sig),
                            ("native_plain", make_signature("Plain", {"flag": bool, "category": Literal["billing", "technical"]})),
                            ("box", make_signature("Boxes", {"flag": Noul, "box": BoundingBox}))]:
        supplied = {k: answers[k] for k in signature.output_fields if k in answers}
        if "box" in signature.output_fields:
            supplied["box"] = {"box": {"top": 0.1, "left": 0.2, "bottom": 0.8, "right": 0.9}}
        for config_name, config in [("defaults", {}), ("configured", {k: v for k, v in fields.items() if k in signature.output_fields})]:
            for demo_name, demos in [("no_demos", []), ("demos", [dspy.Example(text="demo", flag=False, category="billing", ignored="drop")])]:
                lm = ScriptedLM(answers=supplied)
                predictor = dspy.Predict(signature)
                predictor.fields = copy.deepcopy(config)
                predictor.demos = demos
                state = DecisionState(signature, config, system_one=True)
                adapter = DecisionAdapter(dspy.XMLAdapter(), state, True)
                add("system_one_prepare", f"{name}/{config_name}/{demo_name}",
                    lambda: adapter._prepare(lm, signature, demos, {"text": "日本 issue", "ignored": "discard"}, {}),
                    {"fields": config, "demos": demos})
                def run():
                    with dspy.context(lm=lm), record_evidence() as log:
                        prediction = predictor(text="日本 issue", ignored="discard")
                    return {"outputs": prediction.toDict(), "evidence": [{"predictor_is_current": owner is predictor,
                            "field": field_name, "answer": item} for owner, field_name, item in log]}
                row = add("system_one_predict", f"{name}/{config_name}/{demo_name}", run, config)
                row["calls_json"] = lm.calls
    for name, supplied in [
        ("missing_answers", {}), ("invalid_box", {"flag": {"noul": 1}, "box": {"box": {"top": 0.9, "left": 0.2, "bottom": 0.1, "right": 0.8}}}),
        ("missing_box", {"flag": {"noul": 1}}),
    ]:
        signature = make_signature("BoxError", {"flag": Noul, "box": BoundingBox})
        lm = ScriptedLM(answers=supplied)
        predictor = dspy.Predict(signature)
        def run():
            with dspy.context(lm=lm), record_evidence() as log:
                result = observe(lambda: predictor(text="invalid"))
            return {"prediction": result, "evidence": [(n, item) for _, n, item in log]}
        row = add("system_one_errors", name, run, supplied)
        row["calls_json"] = lm.calls
    empty_choice = Choice[("", "empty"), ("x", "nonempty")]
    for name, signature, config in [
        ("generation_options", sig, {"temperature": 0.2}),
        ("empty_choice_label", make_signature("Empty", {"answer": empty_choice}), {}),
        ("unsupported_str", make_signature("Unsupported", {"answer": str}), {}),
        ("bare_score", make_signature("BareScore", {"answer": Score}), {}),
    ]:
        lm = ScriptedLM(answers=answers)
        predictor = dspy.Predict(signature, **config)
        def run():
            with dspy.context(lm=lm):
                return predictor(text="boundary")
        row = add("system_one_errors", name, run, config)
        row["calls_json"] = lm.calls
    for system_one in (False, True):
        state = DecisionState(sig, {}, system_one=system_one)
        adapter = DecisionAdapter(dspy.JSONAdapter(), state, system_one)
        lm = ScriptedLM(answers=answers) if system_one else ScriptedLM("{}")
        def run():
            with dspy.context(send_stream=object()):
                return adapter._prepare(lm, sig, [], {"text": "stream"}, {})
        add("streaming_boundary", str(system_one), run)
    lm = ScriptedLM(answers=answers, asynchronous=True)
    predictor = dspy.Predict(sig)
    async def run_async():
        with dspy.context(lm=lm), record_evidence() as log:
            prediction = await predictor.acall(text="async system one")
        return {"outputs": prediction.toDict(), "evidence": [(name, item) for _, name, item in log]}
    row = add("system_one_async", "predict", lambda: asyncio.run(run_async()))
    row["calls_json"] = lm.calls


def persistence_cases(models, sig):
    predictor = dspy.Predict(sig)
    predictor.demos = [dspy.Example(text="demo", flag=models["noul_binary"](value=True, confidence=1))]
    steps = [
        ("initial", lambda: predictor.dump_state()),
        ("get_default", lambda: {name: predictor.get_criteria(name) for name in sig.output_fields}),
        ("set_flag", lambda: predictor.set_criteria("flag", {"true": {"reason": ["blocked"]}})),
        ("set_rating", lambda: predictor.set_criteria("rating", ["low", "middle", {"high": 1}])),
        ("set_category", lambda: predictor.set_criteria("category", {"billing": None, "technical": ["bug"]})),
        ("bad_flag", lambda: predictor.set_criteria("flag", {"yes": "bad"})),
        ("bad_rating", lambda: predictor.set_criteria("rating", ["one", "two"])),
        ("bad_category", lambda: predictor.set_criteria("category", {"billing": "partial"})),
        ("bad_output", lambda: predictor.set_criteria("unknown", None)),
        ("dump", lambda: predictor.dump_state()), ("dump_python_mode", lambda: predictor.dump_state(json_mode=False)),
    ]
    for name, fn in steps:
        row = add("predict_persistence", name, fn)
        row["fields_after"] = enc(copy.deepcopy(predictor.fields))
    def snapshot():
        criteria = {"billing": {"nested": ["first"]}, "technical": None}
        predictor.set_criteria("category", criteria)
        criteria["billing"]["nested"].append("external")
        returned = predictor.get_criteria("category")
        returned["billing"]["nested"].append("returned")
        dumped = predictor.dump_state()
        dumped["fields"]["category"]["criteria"]["billing"]["nested"].append("dump")
        return {"criteria": predictor.get_criteria("category"), "fields": predictor.fields}
    add("predict_persistence", "independent_criteria_and_dump", snapshot)
    dumped = predictor.dump_state()
    for name, state in [("valid", dumped), ("legacy_missing_fields", {k: v for k, v in dumped.items() if k != "fields"}),
                        ("bad_fields", {**dumped, "fields": {"flag": {"threshold": False}}}),
                        ("unknown_fields", {**dumped, "fields": {"unknown": {}}}),
                        ("legacy_extended_signature", {**dumped, "extended_signature": {}})]:
        restored = dspy.Predict(sig)
        restored.fields = {"flag": {"threshold": 0.9}}
        before = restored.dump_state()
        result = observe(lambda s=state: restored.load_state(copy.deepcopy(s)) is restored)
        add("predict_load", name, lambda: {"load": result, "before": before, "after": restored.dump_state()}, state)
    predictor.fields["flag"]["threshold"] = False
    add("predict_persistence", "invalid_direct_fields_dump", predictor.dump_state, predictor.fields)


def annotated_cases(models):
    cases = [
        ("bool_noul", lambda: Annotated[bool, models["noul_binary"]]),
        ("literal_choice", lambda: Annotated[Literal["billing", "technical"], models["choice_text"]]),
        ("literal_choice_reordered", lambda: Annotated[Literal["technical", "billing"], models["choice_text"]]),
        ("literal_choice_type_mismatch", lambda: Annotated[Literal[True, False], models["choice_int"]]),
        ("literal_choice_members_mismatch", lambda: Annotated[Literal["billing", "other"], models["choice_text"]]),
        ("literal_bare_choice", lambda: Annotated[Literal[1, True, None], Choice]),
        ("float_score", lambda: Annotated[float, models["score_three"]]),
        ("optional_noul", lambda: Optional[models["noul_binary"]]),
    ]
    for name, fn in cases:
        annotation = fn()
        for mode in ("validation", "serialization"):
            add("annotated_schema", f"{name}/{mode}", lambda m=mode: TypeAdapter(annotation).json_schema(mode=m),
                {"annotation": annotation, "mode": mode}, raw_json=True)
        for scalar_name, value in SCALARS:
            add("annotated_validate", f"{name}/{scalar_name}", lambda v=value: TypeAdapter(annotation).validate_python(v),
                {"annotation": annotation, "value": value})


def prediction_snapshot(predictor, lm, adapter, text):
    with dspy.context(lm=lm, adapter=adapter, track_usage=True), record_evidence() as log:
        prediction = predictor(text=text)
    # Completions exposes items(), not toDict(). The initial accessor-error rows
    # remain intact; these independent calls observe the actual successful outputs.
    metadata_keys = ("usage", "cost", "model", "response_model", "model_type", "cost_details")
    return {
        "outputs": prediction.toDict(),
        "completions": dict(prediction.completions.items()),
        "completion_count": len(prediction.completions),
        "usage": prediction.get_lm_usage(),
        "evidence": [{"predictor_is_current": owner is predictor, "field": name, "answer": item}
                     for owner, name, item in log],
        "metadata": [{key: history.get(key) for key in metadata_keys} for history in lm.history],
    }


def supplemental_prediction_cases(models):
    sig = make_signature("SuccessfulRich", {"flag": models["noul_binary"], "rating": models["score_three"],
                                           "category": models["choice_text"]})
    native = make_signature("SuccessfulNative", {"flag": Annotated[bool, models["noul_binary"]],
               "category": Annotated[Literal["billing", "technical"], models["choice_text"]]})
    answers = {"flag": {"noul": 0.75},
               "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.8},
               "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}
    configured = {"flag": {"threshold": 0.8, "instructions": {"prompt": ["Is it blocked?", "日本"]},
                            "criteria": {"true": {"reason": "blocked"}, "false": ["usable"]}},
                  "rating": {"cuts": [0.25, 1.25], "criteria": ["low", {"middle": "fair"}, ["high"]]},
                  "category": {"weights": {"billing": 2}, "instructions": ["Route the issue"],
                               "criteria": {"billing": None, "technical": "bug"}}}
    demo = dspy.Example(text="demonstration", flag=models["noul_binary"](value=True, confidence=1),
                        rating=models["score_three"](value=2, confidence=1),
                        category=models["choice_text"](value="technical", confidence=1)).with_inputs("text")
    for signature_name, signature in [("rich", sig), ("native", native)]:
        supplied = {name: answers[name] for name in signature.output_fields}
        for config_name, fields in [("defaults", {}), ("configured", {
                key: value for key, value in configured.items() if key in signature.output_fields})]:
            for demo_name, demos in [("no_demos", []), ("demos", [demo])]:
                for adapter_name, adapter_cls in ADAPTERS.items():
                    for structured in ([False, True] if adapter_name == "json" else [False]):
                        completion = completion_for(adapter_name, supplied)
                        lm = ScriptedLM(completion, structured=structured)
                        predictor = dspy.Predict(signature, temperature=0.2)
                        predictor.fields = copy.deepcopy(fields)
                        predictor.demos = demos
                        row = add("predict_outputs",
                            f"{signature_name}/{config_name}/{demo_name}/{adapter_name}/structured_{structured}",
                            lambda: prediction_snapshot(predictor, lm, adapter_cls(), "issue 日本"),
                            {"fields": fields, "completion": completion, "demos": demos},
                            adapter=adapter_name, signature=signature_name)
                        row["calls_json"] = lm.calls

    # Compare actual unconfigured versus opt-in native bool/Literal prompts.
    plain = make_signature("PlainNativeComparison", {"flag": bool, "category": Literal["billing", "technical"]})
    for name, fields, supplied in [
        ("unconfigured", {}, {"flag": True, "category": "technical"}),
        ("configured", {"flag": {"threshold": 0.8}, "category": {"weights": {"billing": 2}}},
            {"flag": answers["flag"], "category": answers["category"]}),
        ("only_flag_configured", {"flag": {"threshold": 0.8}}, {"flag": answers["flag"], "category": "technical"}),
    ]:
        for adapter_name, adapter_cls in ADAPTERS.items():
            lm = ScriptedLM(completion_for(adapter_name, supplied))
            predictor = dspy.Predict(plain)
            predictor.fields = copy.deepcopy(fields)
            resolved = resolve_adapter(lm, adapter_cls(), plain, fields, plain)
            row = add("native_configuration_prompts", f"{name}/{adapter_name}",
                lambda: prediction_snapshot(predictor, lm, adapter_cls(), "native plain 日本"),
                {"fields": fields}, adapter=adapter_name, resolved_adapter=type(resolved).__name__)
            row["calls_json"] = lm.calls


def supplemental_model_cases(models):
    options = tuple((label, {"label": label}) for label in
                    ("é", "ΐ", "ǆ", "ß", "ΟΣ", "they're", "snake_case", "_leading", "two__words"))
    unicode_choice = register("choice_unicode", Choice[options])
    for mode in ("validation", "serialization"):
        add("schemas", f"choice_unicode/{mode}",
            lambda m=mode: unicode_choice.model_json_schema(mode=m), {"mode": mode}, unicode_choice, raw_json=True)
        for by_alias in (True, False):
            add("evidence_schemas", f"choice_unicode/{mode}/by_alias_{by_alias}",
                lambda m=mode, a=by_alias: evidence_type(unicode_choice).model_json_schema(mode=m, by_alias=a),
                {"mode": mode, "by_alias": by_alias}, unicode_choice, raw_json=True)
    labels = list(unicode_choice.criteria())
    probabilities = dict.fromkeys(labels, 0)
    probabilities["ß"] = 1
    evidence = evidence_type(unicode_choice).model_validate({"probabilities": probabilities, "confidence": 0.8})
    add("evidence_aliases", "choice_unicode", lambda: {
        "raw": evidence, "by_alias": evidence.model_dump(by_alias=True),
        "by_field": evidence.model_dump(by_alias=False),
        "fields": [(name, field.alias, field.title) for name, field in type(evidence.probabilities).model_fields.items()],
    }, {"probabilities": probabilities}, unicode_choice)
    signature = make_signature("UnicodeChoice", {"answer": unicode_choice})
    state = DecisionState(signature, {})
    add("decode_unicode", "typed_winner", lambda: state._decode({
        "answer": {"probabilities": probabilities, "confidence": 0.8}}), probabilities, unicode_choice)
    for adapter_name, adapter_cls in ADAPTERS.items():
        answers = {"answer": {"probabilities": probabilities, "confidence": 0.8}}
        lm = ScriptedLM(completion_for(adapter_name, answers))
        predictor = dspy.Predict(signature)
        row = add("unicode_predict", adapter_name,
            lambda: prediction_snapshot(predictor, lm, adapter_cls(), "Unicode labels"),
            answers, unicode_choice, adapter=adapter_name)
        row["calls_json"] = lm.calls

    for model_id in ("bare_noul", "bare_score", "bare_choice", "noul_binary", "score_three", "choice_text"):
        cls = models[model_id]
        value = True if issubclass(cls, Noul) else 1 if issubclass(cls, Score) else "billing"
        model = cls(value=value, confidence=0.75)
        field = make_signature("Format", {"answer": cls}).output_fields["answer"]
        add("formatting_boundaries", model_id, lambda: {
            "type_format": observe(model.format),
            "format_field_value": format_field_value(field, model),
            "format_field_value_not_assume_text": format_field_value(field, model, assume_text=False),
            "lookalike_dict_format": format_field_value(field, {"value": value, "confidence": 0.75}),
            "serialize_for_json": serialize_for_json(model),
            "raw_iter": list(model),
        }, model, cls)

    # Independent genuine cache-eviction classes for the other two factories.
    for name, factory in [("noul", lambda text: Noul[(True, text)]),
                          ("score", lambda text: Score[text, "other"])]:
        def run(f=factory):
            original = f("eviction_original")
            value = True if issubclass(original, Noul) else 0
            instance = original(value=value, confidence=0.5)
            for i in range(257):
                f(f"eviction_{i}")
            recreated = f("eviction_original")
            return {"same_class_after_eviction": original is recreated,
                    "same_qualified_name": qualname(original) == qualname(recreated),
                    "equal_instances": instance == recreated(value=value, confidence=0.5),
                    "instance": instance, "cross_validate": observe(lambda: recreated.model_validate(instance))}
        add("factory_cache_eviction", name, run, features=["cache_eviction", "dynamic_class"])


def supplemental_score_order_cases(models):
    import itertools
    import struct

    cls = models["score_three"]
    signature = make_signature("OrderedScore", {"rating": cls})
    for dataset, data in [("tiny_tail", [(0, 1.0), (1, 1e-16), (2, 1e-16)]),
                           ("decimal", [(0, 0.1), (1, 0.2), (2, 0.7)]),
                           ("subnormal", [(0, 5e-324), (1, 1.0), (2, 5e-324)])]:
        for permutation in itertools.permutations(data):
            probabilities = dict(permutation)
            state = DecisionState(signature, {"rating": {"cuts": [0.5, 1.5]}})
            name = dataset + "/" + "".join(str(i) for i, _ in permutation)
            def run():
                result = state._decode({"rating": {"probabilities": probabilities, "confidence": 0.8}})["rating"]
                return {
                    "probability_keys": list(probabilities),
                    "sum": sum(probabilities.values()),
                    "weighted_sum": sum(i * p for i, p in probabilities.items()),
                    "result": result, "value_f64_le": struct.pack("<d", result.value).hex(),
                    "result_probability_keys": list(result.probabilities),
                }
            add("score_insertion_order", name, run, probabilities, cls)


def supplemental_identity_cases(models):
    for ids, value in [
        (("bare_noul", "noul_binary", "noul_true", "sub_noul"), True),
        (("bare_score", "score_three", "score_json", "sub_score"), 1),
        (("bare_choice", "choice_int", "choice_bool", "sub_choice"), 1),
    ]:
        for source_id in ids:
            source = models[source_id](value=value, confidence=0.5)
            for target_id in ids:
                target = models[target_id]
                for entrypoint, fn in {
                    "model_validate": lambda t=target, v=source: t.model_validate(v),
                    "model_validate_strict": lambda t=target, v=source: t.model_validate(v, strict=True),
                    "typeadapter_python": lambda t=target, v=source: TypeAdapter(t).validate_python(v),
                }.items():
                    model_observation("cross_class/" + source_id, target, entrypoint, source, fn)

    rich = make_signature("IdentityRich", {"flag": models["noul_binary"], "rating": models["score_three"],
                                         "category": models["choice_text"]})
    native = make_signature("IdentityNative", {"flag": Annotated[bool, models["noul_binary"]],
                "category": Annotated[Literal["billing", "technical"], models["choice_text"]]})
    baseline = {"flag": {"noul": 0.75},
                "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.8},
                "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}
    for name, signature in [("rich", rich), ("native", native)]:
        for configured in (False, True):
            for adapter_name, adapter_cls in ADAPTERS.items():
                supplied = {name: baseline[name] for name in signature.output_fields}
                lm = ScriptedLM(completion_for(adapter_name, supplied))
                predictor = dspy.Predict(signature)
                if configured:
                    predictor.fields = {"flag": {"threshold": 0.8}, "category": {"weights": {"billing": 2}}}
                def run():
                    with dspy.context(lm=lm, adapter=adapter_cls()):
                        result = predictor(text="Keep genuine result identity")
                    return {"prediction": result, "items": dict(result.items()),
                            "items_are_completion_objects": {
                                key: item is result.completions[key][0] for key, item in result.items()},
                            "toDict": result.toDict()}
                row = add("prediction_identity", f"{name}/{configured}/{adapter_name}", run, predictor.fields)
                row["calls_json"] = lm.calls

    signature = make_signature("EvidenceSnapshots", {"flag": models["noul_binary"]})
    state = DecisionState(signature, {})
    adapter = DecisionAdapter(dspy.JSONAdapter(), state, False)
    def collectors():
        completion = {"flag": {"noul": 0.75}}
        with record_evidence() as outer:
            first = adapter._decode([completion])
            with record_evidence() as inner:
                second = adapter._decode([{"flag": {"noul": 0.25}}])
            completion["flag"]["noul"] = 0
        return {"outer": [(name, item) for _, name, item in outer],
                "inner": [(name, item) for _, name, item in inner],
                "outputs": [first, second], "source_after": completion,
                "collectors_are_distinct": outer is not inner}
    add("evidence_recording", "nested_independent_snapshots", collectors)

    for levels in (2, 10, 11):
        cls = Score[tuple(f"level_{i}" for i in range(levels))]
        signature = make_signature("ScoreLimits", {"rating": cls})
        add("state_score_limits", str(levels),
            lambda: state_view(DecisionState(signature, {})), {"levels": levels})


def mutable_state_boundary_cases(models):
    signature = make_signature("MutableState", {"flag": models["noul_binary"], "rating": models["score_three"],
                                               "category": models["choice_text"]})
    answers = {"flag": {"noul": 0.75},
               "rating": {"probabilities": {0: 0.25, 1: 0.25, 2: 0.5}, "confidence": 0.8},
               "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}
    # Construction validates a snapshot; later direct public-dict mutations are
    # independent Source operations and are not silently repaired by this oracle.
    mutations = [
        ("threshold_false", "flag", "threshold", False),
        ("threshold_string", "flag", "threshold", "0.5"),
        ("threshold_out_of_range", "flag", "threshold", 2),
        ("threshold_nan", "flag", "threshold", float("nan")),
        ("cuts_false", "rating", "cuts", False),
        ("cuts_strings", "rating", "cuts", ["0.5", "1.5"]),
        ("cuts_unordered", "rating", "cuts", [1.5, 0.5]),
        ("weights_false", "category", "weights", False),
        ("weights_nonnumeric", "category", "weights", {"billing": "2"}),
        ("weights_all_zero", "category", "weights", {"billing": 0, "technical": 0}),
        ("criteria_mismatch", "category", "criteria", {"other": "late mutation"}),
        ("instructions_false", "flag", "instructions", False),
    ]
    for name, output, parameter, value in mutations:
        state = DecisionState(signature, {})
        state.fields[output][parameter] = copy.deepcopy(value)
        def run():
            before = copy.deepcopy(state.fields)
            decoded = observe(lambda: state._decode(copy.deepcopy(answers)))
            questions = {name: observe(lambda n=name, kind=kind:
                state._question(n, signature.output_fields[n], kind)) for name, kind in state.types.items()}
            revalidate = observe(state._validate_parameters)
            return {"fields_before": before, "decode": decoded, "questions": questions,
                    "explicit_revalidation": revalidate, "fields_after": state.fields}
        add("state_late_mutation", name, run, {output: {parameter: value}})

    for name, fields in [("false", False), ("zero", 0), ("empty_text", ""), ("none", None), ("empty_list", [])]:
        predictor = dspy.Predict(signature)
        predictor.fields = fields
        lm = ScriptedLM(completion_for("json", answers))
        def run():
            dumped = observe(predictor.dump_state)
            with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
                prediction = observe(lambda: predictor(text="invalid public field state"))
            return {"dump_state": dumped, "prediction": prediction, "fields_after": predictor.fields}
        row = add("predict_fields_boundary", name, run, fields)
        row["calls_json"] = lm.calls

    for name, support in [("true", True), ("one", 1), ("false", False), ("none", None)]:
        lm = ScriptedLM()
        lm.supports_decision_requests = support
        add("backend_selection", name, lambda: {
            "resolved_adapter": type(resolve_adapter(lm, dspy.JSONAdapter(), signature, {})).__name__,
            "system_one": resolve_adapter(lm, dspy.JSONAdapter(), signature, {}).system_one,
        }, {"supports_decision_requests": support})

    for system_one in (False, True):
        lm = ScriptedLM(answers=answers) if system_one else ScriptedLM(completion_for("json", answers))
        predictor = dspy.Predict(signature)
        def run():
            with dspy.context(lm=lm, adapter=dspy.JSONAdapter(), send_stream=object()):
                return predictor(text="streaming boundary")
        row = add("predict_streaming_boundary", str(system_one), run)
        row["calls_json"] = lm.calls


def raw_metadata_boundary_cases(models):
    signature = make_signature("RawMetadata", {"flag": models["noul_binary"],
        "rating": models["score_three"], "category": models["choice_text"]})
    answers = {"flag": {"noul": 0.75},
               "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.8},
               "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}

    class MetadataLM(ScriptedLM):
        def __init__(self, metadata):
            super().__init__(answers={})
            self.metadata = metadata

        def complete(self, request):
            call = {"request_json": serde.request_to_dict(request)}
            self.calls.append(call)
            # Construct the real lm15 Response before reading typesafe metadata.
            # Invalid JSON metadata can fail here, before decision decoding.
            response = Response(id="decision-metadata", model=self.model,
                message=Message.assistant(""), finish_reason="stop",
                usage=Usage(input_tokens=3, output_tokens=1, total_tokens=4),
                provider_data=copy.deepcopy(self.metadata))
            call["response_json"] = serde.response_to_dict(response, include_provider_data=True)
            return response

    cases = [
        ("valid", {"typesafe": {"answers": answers}}),
        ("provider_none", None), ("provider_empty", {}),
        ("typesafe_none", {"typesafe": None}), ("typesafe_list", {"typesafe": []}),
        ("missing_answers", {"typesafe": {}}), ("answers_none", {"typesafe": {"answers": None}}),
        ("answers_list", {"typesafe": {"answers": []}}),
        ("flag_none", {"typesafe": {"answers": {**answers, "flag": None}}}),
        ("flag_primitive", {"typesafe": {"answers": {**answers, "flag": True}}}),
        ("missing_score_confidence", {"typesafe": {"answers": {**answers,
            "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}}}}}),
        ("missing_choice_confidence", {"typesafe": {"answers": {**answers,
            "category": {"probabilities": {"billing": 0.25, "technical": 0.5}}}}}),
        ("score_duplicate_normalized_keys", {"typesafe": {"answers": {**answers,
            "rating": {"probabilities": {"0": 0.25, "00": 0.25, "2": 0.5}, "confidence": 0.8}}}}),
        ("nonfinite_provider_probability", {"typesafe": {"answers": {**answers, "flag": {"noul": float("nan")}}}}),
        ("decimal_provider_probability", {"typesafe": {"answers": {**answers, "flag": {"noul": decimal.Decimal("0.5")}}}}),
        ("bytearray_provider_probability", {"typesafe": {"answers": {**answers, "flag": {"noul": bytearray(b"1")}}}}),
    ]
    for name, metadata in cases:
        lm = MetadataLM(metadata)
        predictor = dspy.Predict(signature)
        def run():
            with dspy.context(lm=lm), record_evidence() as log:
                result = observe(lambda: predictor(text="raw metadata boundary"))
            return {"prediction": result, "evidence": [(name, item) for _, name, item in log]}
        row = add("raw_metadata_boundaries", name, run, metadata)
        row["calls_json"] = lm.calls


def labeled_demo_cases(models):
    signature = make_signature("LabeledDemos", {"flag": models["noul_binary"],
        "rating": models["score_three"], "category": models["choice_text"]})
    answers = {"flag": {"noul": 0.75},
               "rating": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "confidence": 0.8},
               "category": {"probabilities": {"billing": 0.25, "technical": 0.5}, "confidence": 0.9}}
    demos = [
        ("primitive_labels", dspy.Example(text="demo", flag=True, rating=2, category="technical", ignored="omit")),
        ("evidence_labels", dspy.Example(text="demo", **answers)),
        ("partial_labels", dspy.Example(text="demo", flag=False)),
        ("null_labels", dspy.Example(text="demo", flag=None, rating=None, category=None)),
        ("no_input_label", dspy.Example(flag=True, rating=0, category="billing")),
    ]
    for name, demo in demos:
        for backend in ("chat", "json", "xml", "system_one"):
            adapter_name = "xml" if backend == "system_one" else backend
            adapter = ADAPTERS[adapter_name]()
            lm = ScriptedLM(answers=answers) if backend == "system_one" else ScriptedLM(completion_for(adapter_name, answers))
            predictor = dspy.Predict(signature)
            predictor.demos = [demo]
            row = add("labeled_demos", f"{name}/{backend}",
                lambda: prediction_snapshot(predictor, lm, adapter, "real query 日本"), demo, adapter=adapter_name)
            row["calls_json"] = lm.calls
    # Actual Predict owns n-candidate fanout; no fabricated completion arrays.
    for adapter_name, adapter_cls in ADAPTERS.items():
        lm = ScriptedLM(completion_for(adapter_name, answers))
        predictor = dspy.Predict(signature, n=2)
        row = add("decision_candidates", adapter_name,
            lambda: prediction_snapshot(predictor, lm, adapter_cls(), "two candidates"),
            {"n": 2}, adapter=adapter_name)
        row["calls_json"] = lm.calls


def existing_instance_boundary_cases(models):
    for model_id in ("bare_noul", "noul_binary", "bare_score", "score_three",
                     "bare_choice", "choice_text", "choice_typed"):
        cls = models[model_id]
        kind = kind_name(cls)
        default = True if kind == "noul" else 1 if kind == "score" else (
            "ready" if model_id == "choice_typed" else "billing")
        baseline = {"value": default, "confidence": 0.75}
        evidence = ({"probability": 2} if kind == "noul" else
                    {"probabilities": {0: 2}, "level": True} if kind == "score" else
                    {"probabilities": {"unknown": 2}})
        cases = [
            ("noncanonical_value", {**baseline, "value": 1 if kind == "noul" else "outside"}),
            ("nonfinite_confidence", {**baseline, "confidence": float("nan")}),
            ("null_confidence", {**baseline, "confidence": None}),
            ("invalid_evidence", {**baseline, **evidence}),
            ("genuine_decimal_value", {**baseline, "value": decimal.Decimal("0.5")}),
            ("genuine_bytearray_value", {**baseline, "value": bytearray(b"1")}),
        ]
        for case_id, values in cases:
            model_observation("unvalidated/" + case_id, cls, "model_construct", values,
                              lambda v=values: cls.model_construct(**v))
            instance = cls.model_construct(**values)
            for entrypoint, fn in {
                "existing_model_validate": lambda v=instance: cls.model_validate(v),
                "existing_model_validate_strict": lambda v=instance: cls.model_validate(v, strict=True),
                "existing_typeadapter": lambda v=instance: TypeAdapter(cls).validate_python(v),
                "model_copy_shallow": lambda v=instance: v.model_copy(),
                "model_copy_deep": lambda v=instance: v.model_copy(deep=True),
            }.items():
                model_observation("unvalidated/" + case_id, cls, entrypoint, instance, fn)
            def alias_views(v=instance):
                shallow, deep = v.model_copy(), v.model_copy(deep=True)
                return {
                    "validate_is_original": cls.model_validate(v) is v,
                    "shallow_is_original": shallow is v, "deep_is_original": deep is v,
                    "shallow_same_value": shallow.value is v.value,
                    "deep_same_value": deep.value is v.value,
                    "shallow_equal": shallow == v, "deep_equal": deep == v,
                    "shallow_evidence_identity": getattr(shallow, "probabilities", None) is getattr(v, "probabilities", None),
                    "deep_evidence_identity": getattr(deep, "probabilities", None) is getattr(v, "probabilities", None),
                }
            add("existing_model_identity", f"{model_id}/{case_id}", alias_views, instance, cls)


def supplemental_cases(models):
    supplemental_prediction_cases(models)
    supplemental_model_cases(models)
    supplemental_score_order_cases(models)
    supplemental_identity_cases(models)
    mutable_state_boundary_cases(models)
    raw_metadata_boundary_cases(models)
    labeled_demo_cases(models)
    existing_instance_boundary_cases(models)


def source_receipt(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def receipt_counts(rows):
    counts = {}
    for row in rows:
        group = counts.setdefault(row["group"], {"rows": 0, "ok": 0, "error": 0, "view_observations": 0})
        group["rows"] += 1
        group["ok" if row["result"]["ok"] else "error"] += 1
        group["view_observations"] += len(row.get("views", {}))
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true",
                        help="Retain every existing row verbatim and add independently executed supplemental cases.")
    args = parser.parse_args()
    previous = None
    if args.append:
        previous_text = args.output.read_bytes()
        previous = json.loads(previous_text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = setup_models()
        if previous is None:
            model_cases(models)
            factory_cases(models)
            protocol_cases(models)
            annotated_cases(models)
            sig, native_sig = state_cases(models)
            decoding_cases(models, sig)
            evidence_cases(models)
            adapter_predict_cases(models, sig, native_sig)
            system_one_cases(models, sig, native_sig)
            persistence_cases(models, sig)
        supplemental_cases(models)
    if previous is not None:
        old_ids = {row["id"] for row in previous["rows"]}
        ROWS[:] = previous["rows"] + [row for row in ROWS if row["id"] not in old_ids]
        old_models = {model["id"] for model in previous["models"]}
        MODELS[:] = previous["models"] + [model for model in MODELS if model["id"] not in old_models]
    ids = [row["id"] for row in ROWS]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate Source observation IDs")
    modules = {
        "decision_models": sys.modules["dspy.adapters.types.decision"],
        "decision_state": sys.modules["dspy.adapters.decision_state"],
        "decision_adapter": sys.modules["dspy.adapters.decision"],
        "predict": sys.modules["dspy.predict.predict"],
        "base_type": sys.modules["dspy.adapters.types.base_type"],
        "adapter_utils": sys.modules["dspy.adapters.utils"],
        "chat_adapter": sys.modules["dspy.adapters.chat_adapter"],
        "json_adapter": sys.modules["dspy.adapters.json_adapter"],
        "xml_adapter": sys.modules["dspy.adapters.xml_adapter"],
        "base_lm": sys.modules["dspy.clients.base_lm"],
        "lm_execution": sys.modules["dspy.clients.execution"],
        "prediction": sys.modules["dspy.primitives.prediction"],
        "lm15_types": sys.modules["dspy._vendor.lm15.types"],
        "lm15_serde": serde,
    }
    receipt = {
        "format_version": 1,
        "evidence_kind": "actual installed DSPy/Pydantic calls and scripted Source BaseLM/lm15 engines; no external inference or native reconstruction",
        "dspy": dspy.__version__, "pydantic": pydantic.VERSION,
        "pydantic_core": importlib.metadata.version("pydantic-core"), "python": sys.version.split()[0],
        "oracle": source_receipt(__file__),
        "sources": {name: source_receipt(inspect.getsourcefile(module)) for name, module in modules.items()},
        "harness": source_receipt(ref.__file__),
        "encoding": "python_type tags preserve genuine classes, typed dict keys, bytes/bytearray/Decimal/nonfinite; *_json is raw JSON-native data; ensure_ascii=True",
        "models": MODELS, "counts": receipt_counts(ROWS), "rows": ROWS,
    }
    if previous is not None:
        receipt["prior_receipt"] = {
            "sha256": hashlib.sha256(previous_text).hexdigest(),
            "rows": len(previous["rows"]), "oracle": previous["oracle"],
            "retained_rows_verbatim": True,
        }
    text = json.dumps(receipt, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
    args.output.write_text(text)
    print(json.dumps({"output": str(args.output), "rows": len(ROWS), "models": len(MODELS),
                      "source_error_rows": sum(not row["result"]["ok"] for row in ROWS),
                      "sha256": hashlib.sha256(text.encode()).hexdigest(), "counts": receipt["counts"]}))


if __name__ == "__main__":
    main()
