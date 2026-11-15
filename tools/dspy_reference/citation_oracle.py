#!/usr/bin/env python3
"""Observe installed DSPy 3.5 Citations behavior; never run native code.

Run from the repository root with the pinned reference virtualenv:

    devenv shell -- tools/dspy_reference/.venv/bin/python tools/dspy_reference/citation_oracle.py OUTPUT

Every observed value comes from the installed Python libraries. Provider rows
use the actual lm15 response readers and actual dspy.LM HTTP to a localhost
fixture: protocol evidence, not authenticated model inference.

Encoding: JSON-native wire data (requests, prompt messages, lm15 serde dicts)
is kept raw under keys ending in ``_json``. Python values use ``python_type``
tags so class identity is never inferred from a dictionary shape. The receipt
is written with ensure_ascii=True so lone surrogates survive as escapes.
"""

import argparse
import copy
import decimal
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
import threading
import types
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import dspy
import pydantic
from pydantic import BaseModel, TypeAdapter

from dspy.adapters.base import prompt_to_openai_messages
from dspy.adapters.types.citation import Citations
from dspy.clients import lm15_boundary
from dspy.clients.call_result import AttributeDict, attributes, convenience_output, response_text
from dspy.clients.engines.streaming import EngineChunk, ListenerBridge
from dspy._vendor.lm15 import serde
from dspy._vendor.lm15.providers import openai as openai_provider
from dspy._vendor.lm15.providers import openai_chat as openai_chat_provider
from dspy._vendor.lm15.providers.base import HttpResponse
from dspy._vendor.lm15.sse import SSEEvent
from dspy.lm15 import (CitationDelta, CitationPart, Message, Request, Response, StreamDeltaEvent,
                       StreamEndEvent, StreamStartEvent, TextPart, Usage)
from dspy.streaming.messages import StreamResponse

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")

warnings.simplefilter("ignore")
Citation = Citations.Citation


class SubCitation(Citations.Citation):
    """Distinct subclass used only to observe actual class handling."""


class SubCitations(Citations):
    """Distinct subclass used only to observe actual class handling."""


class LookalikeModel(BaseModel):
    cited_text: str


# ─── encoding ──────────────────────────────────────────────────────────────


def qualname(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def enc(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"python_type": "float", "repr": repr(value)}
    if isinstance(value, (bytes, bytearray)):
        return {"python_type": type(value).__name__, "hex": bytes(value).hex()}
    if isinstance(value, decimal.Decimal):
        return {"python_type": "Decimal", "str": str(value)}
    if isinstance(value, (Citations, Citation)):
        name = "Citations" if type(value) is Citations else "Citation" if type(value) is Citation else type(value).__qualname__
        return {"python_type": name, "class": qualname(type(value)),
                "fields": [[key, enc(item)] for key, item in value.__dict__.items()]}
    if isinstance(value, BaseModel):
        return {"python_type": type(value).__qualname__, "class": qualname(type(value)),
                "fields": [[key, enc(item)] for key, item in value.__dict__.items()]}
    if isinstance(value, Response):
        return {"python_type": "lm15.Response", "response_json": serde.response_to_dict(value)}
    if type(value).__name__.endswith("Part") and hasattr(value, "type") and type(value).__module__.startswith("dspy._vendor.lm15"):
        return {"python_type": "lm15." + type(value).__name__, "part_json": serde.part_to_dict(value)}
    if isinstance(value, (StreamDeltaEvent, StreamEndEvent, StreamStartEvent)):
        return {"python_type": "lm15." + type(value).__name__, "event_json": serde.stream_event_to_dict(value)}
    if isinstance(value, StreamResponse):
        return {"python_type": "StreamResponse", "predict_name": value.predict_name,
                "signature_field_name": value.signature_field_name, "chunk": enc(value.chunk),
                "is_last_chunk": value.is_last_chunk}
    if isinstance(value, dspy.Prediction):
        return {"python_type": "Prediction", "fields": [[key, enc(value[key])] for key in value.keys()]}
    if isinstance(value, EngineChunk):
        return {"python_type": "EngineChunk", "items": [[enc(k), enc(v)] for k, v in value.items()]}
    if isinstance(value, AttributeDict):
        return {"python_type": "AttributeDict", "items": [[enc(k), enc(v)] for k, v in value.items()]}
    if isinstance(value, types.SimpleNamespace):
        return {"python_type": "SimpleNamespace", "attributes": [[k, enc(v)] for k, v in vars(value).items()]}
    if isinstance(value, dict):
        return {"python_type": "dict", "items": [[enc(k), enc(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"python_type": "tuple", "items": [enc(item) for item in value]}
    if isinstance(value, list):
        return [enc(item) for item in value]
    if isinstance(value, slice):
        return {"python_type": "slice", "start": enc(value.start), "stop": enc(value.stop), "step": enc(value.step)}
    if isinstance(value, type):
        return {"python_type": "type", "class": qualname(value)}
    if isinstance(value, BaseException):
        return {"python_type": type(value).__name__, "message": str(value)}
    if value is Optional:
        return {"python_type": "typing", "repr": repr(value)}
    raise TypeError(f"Oracle needs an explicit encoding for {type(value)!r}")


def error_value(exc):
    result = {"type": type(exc).__name__, "module": type(exc).__module__, "message": str(exc)}
    if isinstance(exc, pydantic.ValidationError):
        result["validation_errors"] = [
            {key: enc(val) if key != "ctx" else {str(k): str(v) for k, v in val.items()} for key, val in err.items()}
            for err in exc.errors(include_url=False)]
    for name in ("lm_response", "provider", "feature", "code"):
        val = getattr(exc, name, None)
        if isinstance(val, (str, int)):
            result[name] = val
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        result["cause"] = {"type": type(cause).__name__, "message": str(cause)}
    return result


def observe(fn, encoder=enc):
    try:
        return {"ok": True, "value": encoder(fn())}
    except Exception as exc:  # Source failures are observations, not harness errors.
        return {"ok": False, "error": error_value(exc)}


def model_views(model):
    views = {
        "class": lambda: qualname(type(model)),
        "str": lambda: str(model), "repr": lambda: repr(model),
        "format": model.format, "serialize_model": model.serialize_model,
        "model_dump": model.model_dump,
        "model_dump_round_trip": lambda: model.model_dump(round_trip=True),
        "model_dump_json_mode": lambda: model.model_dump(mode="json"),
        "model_dump_json_mode_round_trip": lambda: model.model_dump(mode="json", round_trip=True),
        "model_dump_json": model.model_dump_json,
        "model_dump_json_round_trip": lambda: model.model_dump_json(round_trip=True),
        "bool": lambda: bool(model),
        "iter": lambda: list(iter(model)),
        "hash_admissible": lambda: (hash(model), True)[1],
        "copy_eq": lambda: copy.deepcopy(model) == model,
        "self_eq": lambda: model == model,
    }
    if isinstance(model, Citations):
        views["len"] = lambda: len(model)
        views["fields_citations_identity_preserved"] = lambda: [a is b for a, b in zip(model, model.citations)]
    return {name: observe(fn) for name, fn in views.items()}


def model_row(group, case_id, entrypoint, fn, args):
    row = {"group": group, "id": case_id, "entrypoint": entrypoint, "input": enc(args)}
    try:
        result = fn()
    except Exception as exc:
        row["result"] = {"ok": False, "error": error_value(exc)}
        return row
    row["result"] = {"ok": True, "value": enc(result)}
    if isinstance(result, (Citations, Citation)):
        row["views"] = model_views(result)
        row["identity"] = {"result_is_input": result is args if not isinstance(args, tuple) else None}
        if isinstance(result, Citations):
            originals = args.get("citations") if isinstance(args, dict) and isinstance(args.get("citations"), list) else (
                args if isinstance(args, list) else [])
            row["identity"]["element_is_input"] = [
                any(item is original for original in originals) for item in result.citations]
    return row


# ─── model cases ───────────────────────────────────────────────────────────

FIELD_VALUES = {
    "str_fields": ["x", "", "é😀", "\ud800", "a\x00b", "\u2028", 0, 1, 1.5, True, None, b"bytes",
                   bytearray(b"ab"), b"\xff", ["x"], {"x": 1}, decimal.Decimal("1")],
    "int_fields": [0, -1, 1, 2**63 - 1, 2**63, 2**64, 10**30, -(10**30), "1", " 1 ", "+1", "-1", "1_000", "1.0",
                   "1e3", "１", "0x10", "", "x", 1.0, 1.5, -0.0, float("nan"), float("inf"), True, False, None,
                   b"1", bytearray(b"2"), decimal.Decimal("3"), decimal.Decimal("3.5"), [1], {"x": 1},
                   # Appended later so earlier positional case ids stay stable.
                   1e30, -1e30, float(-(2**63)), float(2**63), 1e308, float(2**64), float(2**53 + 2), -1e308,
                   float(-(2**63) - 2048), 5e-324, -2.0],
}
STRING_FIELDS = ("cited_text", "type", "document_title", "url", "supported_text")
INT_FIELDS = ("document_index", "start_char_index", "end_char_index")


def citation_inputs():
    full = {"type": "char_location", "cited_text": "The sky", "document_index": 0, "document_title": "Weather",
            "url": "https://example.invalid/a", "start_char_index": 0, "end_char_index": 7,
            "supported_text": "Sky is blue."}
    cases = {"minimal": {"cited_text": "x"}, "full": full, "empty_dict": {}, "missing_cited_text": {"document_index": 0},
             "extra_key": {"cited_text": "x", "extra": 1}, "explicit_nulls": {"cited_text": "x", **{k: None for k in INT_FIELDS},
                                                                              "document_title": None, "url": None,
                                                                              "supported_text": None},
             "type_null": {"cited_text": "x", "type": None}, "reordered": dict(reversed(list(full.items())))}
    for name in STRING_FIELDS:
        for index, value in enumerate(FIELD_VALUES["str_fields"]):
            cases[f"{name}/{index}"] = {**({"cited_text": "x"} if name != "cited_text" else {}), name: value}
    for name in INT_FIELDS:
        for index, value in enumerate(FIELD_VALUES["int_fields"]):
            cases[f"{name}/{index}"] = {"cited_text": "x", name: value}
    return cases


def json_able(value):
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def citation_rows():
    rows = []
    adapter = TypeAdapter(Citation)
    for name, data in citation_inputs().items():
        rows.append(model_row("citation", f"{name}/constructor", "Citation(**input)", lambda d=data: Citation(**d), data))
        rows.append(model_row("citation", f"{name}/model_validate", "Citation.model_validate",
                              lambda d=data: Citation.model_validate(d), data))
        rows.append(model_row("citation", f"{name}/type_adapter", "TypeAdapter(Citation).validate_python",
                              lambda d=data: adapter.validate_python(d), data))
        if json_able(data):
            text = json.dumps(data, ensure_ascii=False)
            rows.append(model_row("citation", f"{name}/model_validate_json", "Citation.model_validate_json",
                                  lambda t=text: Citation.model_validate_json(t), text))
    existing = Citation(cited_text="x", document_index=1)
    others = {"existing": existing, "subclass": SubCitation(cited_text="x"), "lookalike_model": LookalikeModel(cited_text="x"),
              "citations_instance": Citations(citations=[existing]), "none": None, "string": "x", "list": [], "int": 1,
              "tuple_pairs": (("cited_text", "x"),)}
    for name, data in others.items():
        rows.append(model_row("citation", f"nondict/{name}/model_validate", "Citation.model_validate",
                              lambda d=data: Citation.model_validate(d), data))
        rows.append(model_row("citation", f"nondict/{name}/type_adapter", "TypeAdapter(Citation).validate_python",
                              lambda d=data: adapter.validate_python(d), data))
    for name, text in {"surrogate_escape": '{"cited_text":"\\ud800"}', "pair_escape": '{"cited_text":"\\ud83d\\ude00"}',
                       "int_float_text": '{"cited_text":"x","document_index":1.0}',
                       "int_string_text": '{"cited_text":"x","document_index":"7"}', "malformed": '{"cited_text":',
                       "array": '[{"cited_text":"x"}]', "bigint": '{"cited_text":"x","document_index":123456789012345678901234567890}'}.items():
        rows.append(model_row("citation", f"json_text/{name}", "Citation.model_validate_json",
                              lambda t=text: Citation.model_validate_json(t), text))
    return rows


def citations_inputs():
    c0 = Citation(cited_text="A", document_index=0)
    sub = SubCitation(cited_text="S")
    return {
        "list_dicts": [{"cited_text": "A", "document_index": 0}, {"cited_text": "B", "url": "u"}],
        "list_empty": [],
        "list_missing_cited_text": [{"document_index": 0}],
        "list_mixed_missing": [{"cited_text": "A"}, {"document_index": 0}],
        "list_existing_instance": [c0],
        "list_dict_and_instance": [{"cited_text": "A"}, c0],
        "list_lax_int": [{"cited_text": "A", "document_index": "1", "start_char_index": 2.0}],
        "list_bad_int": [{"cited_text": "A", "document_index": "x"}],
        "list_extra": [{"cited_text": "A", "extra": True}],
        "list_strings": ["A"],
        "list_none_cited": [{"cited_text": None}],
        "dict_citations_list": {"citations": [{"cited_text": "A"}]},
        "dict_citations_instances": {"citations": [c0, sub]},
        "dict_citations_mixed": {"citations": [{"cited_text": "A"}, c0, "raw"]},
        "dict_citations_empty": {"citations": []},
        "dict_citations_tuple": {"citations": ({"cited_text": "A"},)},
        "dict_citations_none": {"citations": None},
        "dict_citations_string": {"citations": "A"},
        "dict_citations_dict": {"citations": {"cited_text": "A"}},
        "dict_citations_and_cited_text": {"citations": [{"cited_text": "A"}], "cited_text": "B"},
        "dict_citations_extra": {"citations": [{"cited_text": "A"}], "extra": 1},
        "dict_single_citation": {"cited_text": "A", "document_index": 3},
        "dict_single_citation_bad": {"cited_text": "A", "document_index": "x"},
        "dict_empty": {},
        "dict_other": {"text": "A"},
        "tuple_dicts": ({"cited_text": "A"},),
        "string": "A",
        "json_string": '[{"cited_text":"A"}]',
        "none": None,
        "int": 1,
        "bool": True,
        "bytes": b"A",
        "existing_citations": Citations(citations=[c0]),
        "subclass_citations": SubCitations(citations=[c0]),
        "existing_citation": c0,
        "lookalike_model": LookalikeModel(cited_text="A"),
        "list_unicode": [{"cited_text": "é😀\ud800"}],
    }


def citations_rows():
    rows = []
    adapter = TypeAdapter(Citations)
    for name, data in citations_inputs().items():
        rows.append(model_row("citations", f"{name}/model_validate", "Citations.model_validate",
                              lambda d=data: Citations.model_validate(d), data))
        rows.append(model_row("citations", f"{name}/type_adapter", "TypeAdapter(Citations).validate_python",
                              lambda d=data: adapter.validate_python(d), data))
        rows.append(model_row("citations", f"{name}/constructor_citations_kw", "Citations(citations=input)",
                              lambda d=data: Citations(citations=d), {"citations": data}))
        if isinstance(data, dict):
            rows.append(model_row("citations", f"{name}/constructor_kwargs", "Citations(**input)",
                                  lambda d=data: Citations(**d), data))
        rows.append(model_row("citations", f"{name}/from_dict_list", "Citations.from_dict_list",
                              lambda d=data: Citations.from_dict_list(d), data))
        if json_able(data):
            text = json.dumps(data, ensure_ascii=False)
            rows.append(model_row("citations", f"{name}/model_validate_json", "Citations.model_validate_json",
                                  lambda t=text: Citations.model_validate_json(t), text))
    return rows


def from_part_rows():
    rows = []
    specs = {"text_only": {"text": "T"}, "title_only": {"title": "Ti"}, "url_only": {"url": "https://u"},
             "all": {"text": "T", "title": "Ti", "url": "https://u"}, "title_url": {"title": "Ti", "url": "https://u"},
             "unicode": {"text": "é😀", "title": "\ud800"}, "empty_text": {"text": "", "title": "Ti"},
             "all_none": {}, "int_text": {"text": 1}, "whitespace": {"text": " "}}
    for name, kwargs in specs.items():
        part = observe(lambda k=kwargs: CitationPart(**k), lambda p: p)
        row = {"group": "from_part", "id": name, "input": enc(kwargs)}
        if not part["ok"]:
            row["citation_part"] = part
            row["result"] = None
        else:
            row["citation_part"] = {"ok": True, "value": enc(part["value"])}
            row.update({k: v for k, v in model_row("from_part", name, "Citation.from_part",
                                                     lambda p=part["value"]: Citation.from_part(p), kwargs).items()
                        if k in ("result", "views")})
        rows.append(row)
    return rows


def protocol_rows():
    c0 = Citation(cited_text="A", document_index=0)
    c1 = Citation(cited_text="B", url="u")
    values = {
        "c0": c0, "c0_copy": Citation(cited_text="A", document_index=0),
        "c0_explicit_type": Citation(type="char_location", cited_text="A", document_index=0),
        "c0_other_type": Citation(type="page_location", cited_text="A", document_index=0),
        "c0_subclass": SubCitation(cited_text="A", document_index=0), "c1": c1, "c0_format": c0.format(),
        "c0_lookalike": LookalikeModel(cited_text="A"),
        "a": Citations(citations=[c0, c1]), "a_copy": Citations(citations=[Citation(cited_text="A", document_index=0), c1]),
        "a_subclass": SubCitations(citations=[c0, c1]), "b": Citations(citations=[c1, c0]),
        "empty": Citations(citations=[]), "list_c0_c1": [c0, c1], "a_format": Citations(citations=[c0, c1]).format(),
    }
    rows = []
    names = list(values)
    for left in names:
        for right in names:
            if isinstance(values[left], (Citation, Citations)) or isinstance(values[right], (Citation, Citations)):
                rows.append({"group": "protocol", "id": f"eq/{left}/{right}", "operation": "==",
                             "input": [enc(values[left]), enc(values[right])],
                             "result": observe(lambda l=values[left], r=values[right]: l == r)})
    a = values["a"]
    for index in [-3, -2, -1, 0, 1, 2, True, False, 2**64, -(2**64), "0", 1.0, None, slice(0, 1), slice(None, None, -1),
                  slice(5, None), slice(-1, None), slice(None, None, 0), slice(True, None)]:
        rows.append({"group": "protocol", "id": f"index/{enc(index)!r}", "operation": "Citations[index]",
                     "input": [enc(a), enc(index)], "result": observe(lambda i=index: a[i])})
        if isinstance(index, int) and not isinstance(index, bool) and -2 <= index <= 1:
            rows[-1]["identity_is_element"] = a[index] is a.citations[index]
    for name in ("a", "empty", "c0", "c0_subclass", "a_subclass"):
        rows.append({"group": "protocol", "id": f"views/{name}", "operation": "views", "input": enc(values[name]),
                     "views": model_views(values[name])})
    rows.append({"group": "protocol", "id": "contains/c0", "operation": "c0 in a", "input": [enc(a), enc(c0)],
                 "result": observe(lambda: c0 in a)})
    rows.append({"group": "protocol", "id": "list(a)/identity", "operation": "list(iter(a)) is elements",
                 "input": enc(a), "result": observe(lambda: [x is y for x, y in zip(list(a), a.citations)])})
    nested = Citations.model_validate({"citations": [c0]})
    rows.append({"group": "protocol", "id": "nested_existing_identity", "operation": "Citations.model_validate({'citations':[c0]}).citations[0] is c0",
                 "input": enc({"citations": [c0]}), "result": observe(lambda: nested.citations[0] is c0)})
    rows.append({"group": "protocol", "id": "validate_existing_identity", "operation": "Citations.model_validate(a) is a",
                 "input": enc(a), "result": observe(lambda: Citations.model_validate(a) is a)})
    rows.append({"group": "protocol", "id": "type_adapter_existing_identity", "operation": "TypeAdapter(Citations).validate_python(a) is a",
                 "input": enc(a), "result": observe(lambda: TypeAdapter(Citations).validate_python(a) is a)})
    for name in ("description", "is_streamable"):
        rows.append({"group": "protocol", "id": f"classmethod/{name}", "operation": f"Citations.{name}()",
                     "input": None, "result": observe(getattr(Citations, name))})
    return rows


def schema_rows():
    specs = {
        "Citations.model_json_schema": lambda: Citations.model_json_schema(),
        "Citations.model_json_schema_serialization": lambda: Citations.model_json_schema(mode="serialization"),
        "Citation.model_json_schema": lambda: Citation.model_json_schema(),
        "Citation.model_json_schema_serialization": lambda: Citation.model_json_schema(mode="serialization"),
        "TypeAdapter(Optional[Citations])": lambda: TypeAdapter(Optional[Citations]).json_schema(),
        "TypeAdapter(list[Citation])": lambda: TypeAdapter(list[Citation]).json_schema(),
        "TypeAdapter(list[Citations])": lambda: TypeAdapter(list[Citations]).json_schema(),
    }
    return [{"group": "schema", "id": name, "result": observe(fn, lambda v: v)} for name, fn in specs.items()]


# ─── adapters and typed Predict ────────────────────────────────────────────


def signature(kind):
    annotation = {"direct": Citations, "optional": Optional[Citations], "list_citation": list[Citation],
                  "list_citations": list[Citations]}[kind]
    return dspy.Signature({"question": (str, dspy.InputField()), "answer": (str, dspy.OutputField()),
                           "citations": (annotation, dspy.OutputField())}, "Answer with citations.")


def demo_value(kind):
    c = Citation(cited_text="A", document_index=0, start_char_index=0, end_char_index=1)
    return {"direct": Citations(citations=[c]), "optional": Citations(citations=[c]), "list_citation": [c],
            "list_citations": [Citations(citations=[c])]}[kind]


ADAPTERS = {"chat": dspy.ChatAdapter, "json": dspy.JSONAdapter, "xml": dspy.XMLAdapter}
SIGNATURE_KINDS = ("direct", "optional", "list_citation", "list_citations")


def prompt_json(prompt):
    return {"system": prompt.system if isinstance(prompt.system, (str, type(None))) else [serde.part_to_dict(p) for p in prompt.system],
            "messages": prompt_to_openai_messages(prompt)}


def demo_completion(adapter_name, kind):
    """The assistant demo message the actual adapter renders for a citation value."""
    adapter = ADAPTERS[adapter_name]()
    prompt = adapter.format(signature(kind), [{"question": "dq", "answer": "A", "citations": demo_value(kind)}],
                            {"question": "q"})
    messages = prompt_to_openai_messages(prompt)
    assistant = next(m for m in messages if m["role"] == "assistant")
    return assistant["content"] if isinstance(assistant["content"], str) else None


PAYLOADS = {
    "list_dicts": [{"cited_text": "A", "document_index": 0, "start_char_index": 0, "end_char_index": 1}],
    "single_dict": {"cited_text": "A"},
    "wrapped": {"citations": [{"cited_text": "A"}]},
    "empty_list": [],
    "missing_cited_text": [{"document_index": 0}],
    "lax_int": [{"cited_text": "A", "document_index": "1"}],
    "bad_int": [{"cited_text": "A", "document_index": "x"}],
    "unicode": [{"cited_text": "é😀"}],
    "nested_list": [[{"cited_text": "A"}]],
    "not_json": "not citations",
    "null": None,
}


def completion(adapter_name, payload):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if adapter_name == "chat":
        return f"[[ ## answer ## ]]\nA\n\n[[ ## citations ## ]]\n{text}\n\n[[ ## completed ## ]]"
    if adapter_name == "json":
        return json.dumps({"answer": "A", "citations": payload}, ensure_ascii=False)
    return f"<answer>\nA\n</answer>\n<citations>\n{text}\n</citations>"


def missing_completion(adapter_name):
    return {"chat": "[[ ## answer ## ]]\nA\n\n[[ ## completed ## ]]", "json": '{"answer": "A"}',
            "xml": "<answer>\nA\n</answer>"}[adapter_name]


def adapter_rows():
    rows = []
    for adapter_name, adapter_cls in ADAPTERS.items():
        for kind in SIGNATURE_KINDS:
            sig = signature(kind)
            for demo in (False, True):
                demos = [{"question": "dq", "answer": "A", "citations": demo_value(kind)}] if demo else []
                rows.append({"group": "adapter_format", "id": f"{adapter_name}/{kind}/demo={demo}", "adapter": adapter_name,
                             "signature": kind, "demos": enc(demos),
                             "result": observe(lambda s=sig, d=demos, a=adapter_cls: a().format(s, d, {"question": "q"}),
                                               prompt_json)})
            completions = {name: completion(adapter_name, payload) for name, payload in PAYLOADS.items()}
            completions["missing_field"] = missing_completion(adapter_name)
            generated = observe(lambda a=adapter_name, k=kind: demo_completion(a, k), lambda v: v)
            if generated["ok"] and generated["value"] is not None:
                completions["source_demo_roundtrip"] = generated["value"]
            for name, text in completions.items():
                rows.append({"group": "adapter_parse", "id": f"{adapter_name}/{kind}/{name}", "adapter": adapter_name,
                             "signature": kind, "completion": text,
                             "result": observe(lambda s=sig, t=text, a=adapter_cls: a().parse(s, t))})
    return rows


# ─── localhost HTTP fixture ────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({"path": self.path, "body_json": body})
        if body.get("stream"):
            chunks = []
            for event in self.server.sse:
                head = f"event: {event['type']}\n" if self.server.sse_event_names and isinstance(event, dict) and "type" in event else ""
                chunks.append(f"{head}data: {event if isinstance(event, str) else json.dumps(event)}\n\n")
            payload = "".join(chunks).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            payload = json.dumps(self.server.payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def http_lm(server, model_type):
    return dspy.LM("openai/gpt-4o", engine="lm15", model_type=model_type,
                   api_base=f"http://127.0.0.1:{server.server_port}/v1", api_key="source-fixture-not-a-secret",
                   cache=False, num_retries=0, timeout=5, max_tokens=None, temperature=None)


def responses_body(blocks, extra_items=()):
    return {"id": "resp-citation", "model": "gpt-4o", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": blocks}, *extra_items]}


def chat_body(message):
    return {"id": "chat-citation", "model": "gpt-4o", "choices": [{"index": 0, "message": {"role": "assistant", **message},
                                                                   "finish_reason": "stop"}]}


URL_ANNOTATION = {"type": "url_citation", "url": "https://example.invalid/a", "title": "Source A", "start_index": 0,
                  "end_index": 1}


def annotation_cases():
    text = "é😀ab"
    base = "Answer text"
    cases = {}

    def add(name, annotations, block_text=base, ctype="output_text"):
        cases[name] = responses_body([{"type": ctype, "text": block_text, "annotations": annotations}])

    add("url_citation_full", [URL_ANNOTATION])
    add("indices_only_ascii", [{"type": "url_citation", "url": "u", "start_index": 0, "end_index": 6}])
    for start, end in [(0, 1), (1, 2), (2, 4), (0, 4), (3, 4), (4, 4), (2, 2), (3, 5), (-1, 2), (2, 1), (0, 5)]:
        add(f"unicode_indices/{start}-{end}", [{"type": "url_citation", "start_index": start, "end_index": end}], text)
    add("unicode_indices_with_url", [{"url": "u", "start_index": 1, "end_index": 2}], text)
    for key in ("text", "snippet", "cited_text", "quote"):
        add(f"text_alias/{key}", [{key: "T-" + key, "start_index": 0, "end_index": 1}])
    add("text_alias_precedence", [{"quote": "Q", "cited_text": "C", "snippet": "S", "text": "T"}])
    add("text_alias_empty_falls_through", [{"text": "", "snippet": "S"}])
    for index, value in enumerate([0, 1, False, True, [], ["x"], {}, {"k": 1}, 1.5, "\ud800", " "]):
        add(f"text_value/{index}", [{"text": value, "url": "u"}])
    for key in ("url", "uri"):
        add(f"url_alias/{key}", [{key: "https://" + key}])
    add("url_precedence", [{"url": "U", "uri": "I"}])
    add("url_empty_falls_to_uri", [{"url": "", "uri": "I"}])
    add("url_zero_falls_to_uri", [{"url": 0, "uri": "I"}])
    add("url_false_alone", [{"url": False}])
    add("url_int", [{"url": 7}])
    for key in ("title", "filename", "file_id"):
        add(f"title_alias/{key}", [{key: "T-" + key}])
    add("title_precedence", [{"file_id": "F", "filename": "N", "title": "T"}])
    add("title_zero_falls_through", [{"title": 0, "filename": "N"}])
    add("title_list", [{"title": ["x"]}])
    # `or` chains: the LAST operand is kept even when falsy, then _str_or_none stringifies it.
    for name, annotation in {"url_last_zero": {"url": "", "uri": 0}, "url_last_false": {"url": False, "uri": False},
                             "uri_only_zero": {"uri": 0}, "url_zero_only": {"url": 0},
                             "url_last_empty_list": {"url": None, "uri": []},
                             "title_last_zero": {"title": "", "filename": None, "file_id": 0},
                             "title_last_false": {"title": False, "filename": "", "file_id": False},
                             "title_only_false": {"title": False}, "title_last_empty_dict": {"file_id": {}}}.items():
        add(f"or_chain/{name}", [annotation])
    for name, value in {"zero": 0, "false": False, "true": True, "one": 1, "float": 1.5, "empty_string": "",
                        "empty_dict": {}, "tuple_like_list": [[URL_ANNOTATION]]}.items():
        add(f"annotations_value/{name}", value)
    for name, (start, end) in {"nd_digits": ("١", "٣"), "fullwidth": ("０", "３"), "underscores": ("0_0", "1_0"),
                               "neg_zero_string": ("-0", "2"), "neg_zero_float": (-0.0, 2), "fraction_trunc": (0.9, 2.9),
                               "negative_fraction_trunc": (-0.5, 2), "bool_start": (False, 2), "bool_end": (0, True),
                               "nan_both": (float("nan"), float("nan")), "nan_start_inf_end": (float("nan"), float("inf")),
                               "none_start_inf_end": (None, float("-inf")), "inf_start_none_end": (float("inf"), None),
                               "string_start_inf_end": ("x", float("inf")), "huge_string": ("0", "9" * 50),
                               "exponent_string": ("0", "2e0"), "plus_string": ("+0", "+2"), "spaces": (" 0\n", "\t2 ")}.items():
        add(f"index_parse/{name}", [{"url": "u", "start_index": start, "end_index": end}], "é😀ab")
    for index, (start, end) in enumerate([("1", "3"), (" 1 ", "3"), (1.9, 3.2), (True, 3), (0, True), ("x", 3),
                                          (None, 3), (0, None), (0, 10**30), (float("nan"), 3), (-0.0, 2.0)]):
        add(f"index_value/{index}", [{"start_index": start, "end_index": end}])
    add("index_inf_overflow", [{"start_index": float("inf"), "end_index": 3}])
    add("index_inf_with_text_alias", [{"text": "T", "start_index": float("inf"), "end_index": 3}])
    add("no_fields", [{"type": "url_citation"}])
    add("empty_annotation", [{}])
    for kind in ("file_citation", "container_file_citation", "file_path", "unknown"):
        add(f"type/{kind}", [{"type": kind, "file_id": "file-1", "filename": "a.txt", "start_index": 0, "end_index": 2}])
    add("non_dict_entries", ["x", 1, None, [URL_ANNOTATION], URL_ANNOTATION])
    add("annotations_dict", {"url": "u"})
    add("annotations_string", "url")
    add("annotations_null", None)
    add("annotations_missing", [])
    add("ordering_multiple", [{"url": "u1", "title": "First"}, {"url": "u2", "title": "Second"}, {"text": "Third"}])
    add("non_string_text_block", [{"start_index": 0, "end_index": 3}], 12345)
    add("null_text_block", [{"start_index": 0, "end_index": 1, "url": "u"}], None)
    add("plain_text_ctype", [URL_ANNOTATION], base, "text")
    cases["refusal_annotations_ignored"] = responses_body([{"type": "refusal", "refusal": "no", "annotations": [URL_ANNOTATION]}])
    cases["two_blocks_order"] = responses_body([
        {"type": "output_text", "text": "first", "annotations": [{"url": "a"}]},
        {"type": "output_text", "text": "second", "annotations": [{"url": "b"}, {"url": "c"}]}])
    cases["synthesized_tool_ids"] = responses_body(
        [{"type": "output_text", "text": "first", "annotations": [{"url": "a"}, {"url": "b"}]}],
        [{"type": "function_call", "name": "probe", "arguments": "{}"},
         {"type": "message", "content": [{"type": "output_text", "text": "x", "annotations": [{"url": "c"}]}]},
         {"type": "function_call", "name": "probe", "arguments": "{}"}])
    cases["reasoning_then_citations_then_tool"] = {
        "id": "r", "model": "gpt-4o", "output": [{"type": "reasoning", "summary": [{"text": "think"}]},
                                                  {"type": "message", "content": [{"type": "output_text", "text": "t",
                                                                                   "annotations": [{"url": "a"}]}]},
                                                  {"type": "function_call", "name": "probe", "arguments": "{}"}]}
    return cases


def chat_cases():
    return {
        "chat_message_annotations": chat_body({"content": "Answer text", "annotations": [
            {"type": "url_citation", "url_citation": {"url": "https://example.invalid/a", "title": "A", "start_index": 0,
                                                      "end_index": 6}}]}),
        "chat_message_annotations_flat": chat_body({"content": "Answer text", "annotations": [URL_ANNOTATION]}),
        "chat_content_list_annotations": chat_body({"content": [{"type": "text", "text": "Answer", "annotations": [URL_ANNOTATION]}]}),
        "chat_provider_specific_citations": chat_body({"content": "Answer text", "provider_specific_fields": {
            "citations": [[{"cited_text": "A", "document_title": "D", "url": "u"}]]}}),
        "chat_tool_synthesized_ids_with_annotations": chat_body({"content": "Answer text", "annotations": [URL_ANNOTATION],
                                                                 "tool_calls": [{"type": "function", "function": {"name": "probe", "arguments": "{}"}}]}),
        **chat_content_cases(),
    }


def chat_content_cases():
    """Text-list content, nullable text and reasoning/refusal/tool part ordering in the actual Chat reader."""
    tool = {"type": "function", "function": {"name": "probe", "arguments": "{}"}}
    image = {"type": "image_url", "image_url": {"url": "https://example.invalid/i.png"}}
    messages = {
        "two_text_blocks": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        "text_blocks_first_empty": {"content": [{"type": "text", "text": ""}, {"type": "text", "text": "b"}]},
        "text_blocks_all_empty": {"content": [{"type": "text", "text": ""}, {"type": "text", "text": ""}]},
        "text_block_null_text": {"content": [{"type": "text", "text": None}]},
        "text_block_missing_text": {"content": [{"type": "text"}]},
        "text_block_int_text": {"content": [{"type": "text", "text": 7}]},
        "content_empty_list": {"content": []},
        "nontext_only": {"content": [image]},
        "nontext_only_with_tool": {"content": [image], "tool_calls": [tool]},
        "nontext_then_text": {"content": [image, {"type": "text", "text": "b"}]},
        "null_content_reasoning": {"content": None, "reasoning_content": "think"},
        "empty_content_reasoning": {"content": "", "reasoning_content": "think"},
        "missing_content_reasoning": {"reasoning_content": "think"},
        "reasoning_empty_string": {"content": "a", "reasoning_content": ""},
        "reasoning_empty_list": {"content": "a", "reasoning_content": []},
        "reasoning_nonempty_list": {"content": "a", "reasoning_content": ["bad"]},
        "reasoning_integer": {"content": "a", "reasoning_content": 42},
        "reasoning_alias": {"content": "a", "reasoning": "alias"},
        "reasoning_empty_falls_to_alias": {"content": "a", "reasoning_content": "", "reasoning": "alias"},
        "reasoning_precedence": {"content": "a", "reasoning_content": "primary", "reasoning": "alias"},
        "reasoning_alias_only_null_content": {"content": None, "reasoning": "alias"},
        "null_content_refusal": {"content": None, "refusal": "no"},
        "empty_content_refusal_reasoning": {"content": "", "refusal": "no", "reasoning_content": "think"},
        "tool_after_multi_text_blocks": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                                         "tool_calls": [tool, tool]},
        "tool_after_empty_text_blocks": {"content": [{"type": "text", "text": ""}, {"type": "text", "text": ""}],
                                         "tool_calls": [tool]},
        "tool_after_reasoning_text_refusal": {"content": [{"type": "text", "text": "a"}], "reasoning_content": "think",
                                              "refusal": "no", "tool_calls": [tool]},
        "tool_only_null_content_reasoning": {"content": None, "reasoning_content": "think", "tool_calls": [tool]},
    }
    return {f"chat_content/{name}": chat_body(message) for name, message in messages.items()}


def litellm_boundary_cases():
    """dspy.clients.lm15_boundary.response_value reads SDK provider_specific_fields.citations (LiteLLM engine)."""
    citation_lists = {
        "group_list": [[{"cited_text": "A", "document_title": "D", "url": "u"}]],
        "flat_dicts": [{"text": "T", "title": "Ti"}],
        "supported_text_fallback": [[{"supported_text": "S"}]],
        "empty_cited_text_fallback": [[{"cited_text": "", "text": "T"}]],
        "all_missing": [[{"other": 1}]],
        "non_string": [[{"cited_text": 1}]],
        "empty_group": [[]],
        "two_groups": [[{"cited_text": "A"}], [{"cited_text": "B"}, {"url": "u"}]],
    }
    return {name: chat_body({"content": "Answer text", "provider_specific_fields": {"citations": value}})
            for name, value in citation_lists.items()}


def response_observation(response):
    return {"response": enc(response), "citations": [enc(part) for part in response.citations],
            "response_text": response_text(response),
            "parse_lm_response": observe(lambda: Citations.parse_lm_response(response)),
            "convenience_output": observe(lambda: convenience_output(response))}


REQUEST = Request(model="gpt-4o", messages=(Message.user("probe"),))
# dspy.LM requires Request.model to equal its own provider-qualified model string.
LM_REQUEST = Request(model="openai/gpt-4o", messages=(Message.user("probe"),))


def provider_rows(server):
    rows = []
    responses_reader = openai_provider.OpenAILM(api_key="conversion-only", transport=lm15_boundary._NoTransport())
    for name, body in annotation_cases().items():
        wire_text = json.dumps(body)  # Python default: NaN/Infinity literals, ASCII escapes.
        row = {"group": "provider_responses", "id": name, "body": enc(body), "body_json": body if json_able(body) else None,
               "wire_text": wire_text}
        row["direct_parse_response"] = observe(lambda t=wire_text: responses_reader.parse_response(REQUEST, HttpResponse(
            status=200, reason="OK", headers=[], body=t.encode())), response_observation)
        # The fixture serves json.dumps(body), byte-identical to wire_text.
        server.payload = body
        server.requests.clear()
        row["dspy_lm_http_request"] = observe(lambda: http_lm(server, "responses")(LM_REQUEST), response_observation)
        row["dspy_lm_http_string_prompt"] = observe(lambda: http_lm(server, "responses")(prompt="probe"))
        row["http_request_count"] = len(server.requests)
        rows.append(row)
    for name, body in chat_cases().items():
        row = {"group": "provider_chat", "id": name, "body_json": body}
        row["response_from_openai_chat"] = observe(
            lambda b=body: openai_chat_provider.response_from_openai_chat(b, model="gpt-4o"), response_observation)
        server.payload = body
        server.requests.clear()
        row["dspy_lm_http_request"] = observe(lambda: http_lm(server, "chat")(LM_REQUEST), response_observation)
        row["http_request_count"] = len(server.requests)
        row["chat_history_ingest"] = observe(lambda b=body: openai_chat_provider.request_from_openai_chat(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "q"}, b["choices"][0]["message"]]}),
            lambda r: [[serde.part_to_dict(p) for p in m.parts] for m in r.messages])
        rows.append(row)
    for name, body in litellm_boundary_cases().items():
        rows.append({"group": "provider_litellm_boundary", "id": name, "body_json": body,
                     "response_value": observe(lambda b=body: lm15_boundary.response_value(b, "chat", REQUEST),
                                               response_observation)})
    return rows


def sse(data):
    return SSEEvent(event=data.get("type") if isinstance(data, dict) else None,
                    data=data if isinstance(data, str) else json.dumps(data))


def responses_stream_events(text, annotations, final_annotations=None):
    final_annotations = annotations if final_annotations is None else final_annotations
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text,
                                                                 "annotations": final_annotations}]}
    events = [{"type": "response.created", "response": {"id": "resp-stream", "model": "gpt-4o"}},
              {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "content": []}},
              {"type": "response.output_text.delta", "output_index": 0, "delta": text}]
    events += [{"type": "response.output_text.annotation.added", "output_index": 0, "annotation": a} for a in annotations]
    events += [{"type": "response.output_item.done", "output_index": 0, "item": item},
               {"type": "response.completed", "response": {"id": "resp-stream", "model": "gpt-4o", "status": "completed",
                                                           "output": [item]}}]
    return events


def stream_provider_rows():
    rows = []
    reader = openai_provider.OpenAILM(api_key="conversion-only", transport=lm15_boundary._NoTransport())
    single = {"url_full": {"type": "response.output_text.annotation.added", "output_index": 0, "annotation": URL_ANNOTATION},
              "output_index_3": {"type": "response.output_text.annotation.added", "output_index": 3, "annotation": {"url": "u"}},
              "output_index_string": {"type": "response.output_text.annotation.added", "output_index": "2", "annotation": {"url": "u"}},
              "output_index_null": {"type": "response.output_text.annotation.added", "output_index": None, "annotation": {"url": "u"}},
              "indices_without_source_text": {"type": "response.output_text.annotation.added", "annotation": {"start_index": 0, "end_index": 1}},
              "text_alias": {"type": "response.output_text.annotation.added", "annotation": {"snippet": "S"}},
              "empty_annotation": {"type": "response.output_text.annotation.added", "annotation": {}},
              "non_dict_annotation": {"type": "response.output_text.annotation.added", "annotation": "u"},
              "missing_annotation": {"type": "response.output_text.annotation.added"},
              "int_text": {"type": "response.output_text.annotation.added", "annotation": {"text": 0}},
              "file_citation": {"type": "response.output_text.annotation.added", "annotation": {"type": "file_citation", "file_id": "f"}}}
    for name, event in single.items():
        rows.append({"group": "provider_responses_stream_event", "id": name, "event_json": event,
                     "parse_stream_events": observe(lambda e=event: list(reader.parse_stream_events(REQUEST, sse(e))))})
    sequences = {"one_annotation": responses_stream_events("Answer text", [URL_ANNOTATION]),
                 "two_annotations": responses_stream_events("é😀ab", [{"url": "u1"}, {"title": "T2", "start_index": 1, "end_index": 2}]),
                 "final_only_annotation": responses_stream_events("Answer", [], [URL_ANNOTATION])}
    for name, events in sequences.items():
        def run(events=events):
            stream_reader = reader.stream_reader(REQUEST)
            return [list(stream_reader.feed(sse(event))) for event in events]
        rows.append({"group": "provider_responses_stream_reader", "id": name, "events_json": events,
                     "feed": observe(run)})
    chat_reader = openai_chat_provider.OpenAIChatLM(api_key="conversion-only", transport=lm15_boundary._NoTransport())
    chat_events = {"delta_annotations": {"id": "c", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "A", "annotations": [URL_ANNOTATION]}}]},
                   "delta_provider_specific_citation": {"id": "c", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"provider_specific_fields": {"citation": {"text": "T"}}}}]}}
    for name, event in chat_events.items():
        rows.append({"group": "provider_chat_stream_event", "id": name, "event_json": event,
                     "parse_stream_events": observe(lambda e=event: list(chat_reader.parse_stream_events(REQUEST, sse(e))))})
    return rows


# ─── canonical responses ───────────────────────────────────────────────────


def canonical_rows():
    usage = Usage()
    specs = {
        "no_citation": (TextPart(text="x"),),
        "one_text": (TextPart(text="x"), CitationPart(text="T")),
        "title_only": (CitationPart(title="Ti"),),
        "url_only": (CitationPart(url="https://u"),),
        "title_url": (CitationPart(title="Ti", url="https://u"),),
        "all_fields": (CitationPart(text="T", title="Ti", url="https://u"),),
        "ordering": (CitationPart(url="a"), TextPart(text="x"), CitationPart(title="b"), CitationPart(text="c")),
        "unicode": (CitationPart(text="é😀\ud800"),),
    }
    rows = []
    for name, parts in specs.items():
        response = Response(id="r", model="m", message=Message(role="assistant", parts=parts), finish_reason="stop",
                            usage=usage)
        rows.append({"group": "canonical", "id": name, **response_observation(response)})
    return rows


# ─── typed Predict ─────────────────────────────────────────────────────────


def predict_rows(server):
    rows = []
    annotations = [URL_ANNOTATION, {"type": "url_citation", "title": "Second", "start_index": 2, "end_index": 4}]
    for adapter_name, adapter_cls in ADAPTERS.items():
        for kind in ("direct", "optional", "list_citation"):
            generated = observe(lambda a=adapter_name, k=kind: demo_completion(a, k), lambda v: v)
            valid = generated["value"] if generated["ok"] and generated["value"] else completion(adapter_name, PAYLOADS["list_dicts"])
            texts = {"valid": valid, "invalid_citations": completion(adapter_name, PAYLOADS["bad_int"]),
                     "missing_citations": missing_completion(adapter_name), "empty_text": ""}
            for text_name, text in texts.items():
                for ann_name, ann in (("annotations", annotations), ("no_annotations", []), ("malformed_annotation", [{}])):
                    if kind != "direct" and ann_name == "malformed_annotation":
                        continue
                    for model_type in ("responses", "chat"):
                        if model_type == "responses":
                            body = responses_body([{"type": "output_text", "text": text, "annotations": ann}])
                        else:
                            body = chat_body({"content": text, "annotations": [
                                {"type": "url_citation", "url_citation": {"url": "https://example.invalid/a", "title": "A",
                                                                          "start_index": 0, "end_index": 1}}] if ann_name == "annotations" else []})
                        server.payload = body
                        server.requests.clear()

                        def call(sig=signature(kind), mt=model_type, a=adapter_cls):
                            with dspy.context(lm=http_lm(server, mt), adapter=a()):
                                return dspy.Predict(sig)(question="q")
                        result = observe(call)
                        rows.append({"group": "predict", "id": f"{adapter_name}/{kind}/{text_name}/{ann_name}/{model_type}",
                                     "adapter": adapter_name, "signature": kind, "model_type": model_type,
                                     "completion": text, "response_body_json": body, "result": result,
                                     "requests_json": list(server.requests)})
    return rows


# ─── streaming ─────────────────────────────────────────────────────────────


def bridge_chunk(delta):
    bridge = ListenerBridge("gpt-4o", predict_id=7)
    bridge.chunk(StreamStartEvent(id="s", model="gpt-4o"))
    return bridge.chunk(StreamDeltaEvent(delta=delta))


def stream_chunk_inputs():
    chunks = {}
    for name, kwargs in {"text": {"text": "T"}, "title": {"title": "Ti"}, "url": {"url": "u"},
                         "all": {"text": "T", "title": "Ti", "url": "u"}, "unicode": {"text": "é😀\ud800"},
                         "part_index": {"text": "T", "part_index": 4}}.items():
        chunks[f"bridge/{name}"] = bridge_chunk(CitationDelta(**kwargs))
    chunks["bridge/text_delta"] = bridge_chunk(dspy.lm15.TextDelta(text="hello"))
    ns = types.SimpleNamespace

    def choice(delta, extra=()):
        return attributes({"choices": [{"index": 0, "delta": delta, "finish_reason": None}, *extra]})
    chunks.update({
        "attrdict/citation_text_int": choice({"provider_specific_fields": {"citation": {"text": 1}}}),
        "attrdict/citation_empty_strings": choice({"provider_specific_fields": {"citation": {"text": "", "title": ""}}}),
        "attrdict/citation_empty_dict": choice({"provider_specific_fields": {"citation": {}}}),
        "attrdict/citation_none": choice({"provider_specific_fields": {"citation": None}}),
        "attrdict/citation_list": choice({"provider_specific_fields": {"citation": [{"text": "T"}]}}),
        "attrdict/citation_string": choice({"provider_specific_fields": {"citation": "T"}}),
        "attrdict/citation_aliases_ignored": choice({"provider_specific_fields": {"citation": {"cited_text": "C", "document_title": "D"}}}),
        "attrdict/citation_extra_key": choice({"provider_specific_fields": {"citation": {"text": "T", "extra": 1}}}),
        "attrdict/psf_empty": choice({"provider_specific_fields": {}}),
        "attrdict/psf_none": choice({"provider_specific_fields": None}),
        "attrdict/no_psf": choice({"content": "x"}),
        "attrdict/second_choice_only": choice({"content": "x"}, [{"index": 1, "delta": {"provider_specific_fields": {"citation": {"text": "T"}}}}]),
        "attrdict/first_of_two": choice({"provider_specific_fields": {"citation": {"text": "first"}}},
                                        [{"index": 1, "delta": {"provider_specific_fields": {"citation": {"text": "second"}}}}]),
        "attrdict/choices_empty": attributes({"choices": []}),
        "attrdict/choices_none": attributes({"choices": None}),
        "plain_dict_chunk": {"choices": [{"delta": {"provider_specific_fields": {"citation": {"text": "T"}}}}]},
        "namespace/citation": ns(choices=[ns(delta=ns(provider_specific_fields={"citation": {"text": "T"}}))]),
        "namespace/tuple_choices": ns(choices=(ns(delta=ns(provider_specific_fields={"citation": {"url": "u"}})),)),
        "namespace/psf_namespace": ns(choices=[ns(delta=ns(provider_specific_fields=ns(citation={"text": "T"})))]),
        "namespace/citation_namespace": ns(choices=[ns(delta=ns(provider_specific_fields={"citation": ns(text="T")}))]),
        "namespace/delta_dict": ns(choices=[ns(delta={"provider_specific_fields": {"citation": {"text": "T"}}})]),
        "namespace/choice_dict": ns(choices=[{"delta": ns(provider_specific_fields={"citation": {"text": "T"}})}]),
        "namespace/no_choices_attr": ns(other=1),
        "none": None,
        "string": "chunk",
    })
    return chunks


def stream_rows(server):
    rows = []
    chunks = stream_chunk_inputs()
    for name, chunk in chunks.items():
        rows.append({"group": "parse_stream_chunk", "id": name, "input": enc(chunk),
                     "result": observe(lambda c=chunk: Citations.parse_stream_chunk(c))})
    sequence = ["bridge/text", "bridge/all", "attrdict/first_of_two", "attrdict/citation_text_int", "bridge/text_delta"]
    for adapter_name, adapter_cls in ADAPTERS.items():
        for kind in SIGNATURE_KINDS:
            def run(sig=signature(kind), a=adapter_cls):
                listener = dspy.streaming.StreamListener("citations", predict=dspy.Predict(sig), predict_name="p")
                with dspy.context(adapter=a()):
                    return [listener.receive(chunks[name]) for name in sequence]
            rows.append({"group": "listener_receive", "id": f"{adapter_name}/{kind}", "adapter": adapter_name,
                         "signature": kind, "chunk_ids": sequence, "result": observe(run)})
    streams = {"responses/one_annotation": ("responses", responses_stream_events, [URL_ANNOTATION]),
               "responses/two_annotations": ("responses", responses_stream_events, [{"url": "u1"}, {"title": "T2"}]),
               "responses/no_annotation": ("responses", responses_stream_events, [])}
    for adapter_name, adapter_cls in ADAPTERS.items():
        for kind in ("direct", "optional"):
            generated = observe(lambda a=adapter_name, k=kind: demo_completion(a, k), lambda v: v)
            text = generated["value"] if generated["ok"] and generated["value"] else completion(adapter_name, PAYLOADS["list_dicts"])
            for stream_name, (model_type, builder, ann) in streams.items():
                server.sse = builder(text, ann)
                server.sse_event_names = True
                server.requests.clear()

                def run(sig=signature(kind), a=adapter_cls, mt=model_type):
                    program = dspy.Predict(sig)
                    stream = dspy.streamify(program, stream_listeners=[dspy.streaming.StreamListener("citations")],
                                            async_streaming=False)
                    with dspy.context(lm=http_lm(server, mt), adapter=a()):
                        return [item for item in stream(question="q")
                                if isinstance(item, (StreamResponse, dspy.Prediction))]
                rows.append({"group": "streamify", "id": f"{adapter_name}/{kind}/{stream_name}", "adapter": adapter_name,
                             "signature": kind, "sse_json": server.sse, "result": observe(run),
                             "requests_json": list(server.requests)})
    return rows


# ─── main ──────────────────────────────────────────────────────────────────


def source_receipt(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def counts(rows):
    out = {}
    for row in rows:
        group = out.setdefault(row["group"], {"rows": 0, "ok": 0, "error": 0})
        group["rows"] += 1
        primary = next((row[k] for k in ("result", "direct_parse_response", "response_from_openai_chat", "response_value",
                                          "parse_stream_events", "feed", "citation_part", "parse_lm_response")
                        if isinstance(row.get(k), dict)), None)
        if primary is not None and "ok" in primary:
            group["ok" if primary["ok"] else "error"] += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests, server.payload, server.sse, server.sse_event_names = [], None, [], True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = (citation_rows() + citations_rows() + from_part_rows() + protocol_rows() + schema_rows()
                + adapter_rows() + canonical_rows() + provider_rows(server) + stream_provider_rows()
                + predict_rows(server) + stream_rows(server))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    ids = [(row["group"], row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case ids")
    sources = {name: source_receipt(inspect.getsourcefile(module)) for name, module in {
        "citation": sys.modules["dspy.adapters.types.citation"], "adapter_base": sys.modules["dspy.adapters.base"],
        "base_type": sys.modules["dspy.adapters.types.base_type"], "openai_responses": openai_provider,
        "openai_chat": openai_chat_provider, "lm15_boundary": lm15_boundary,
        "lm15_types": sys.modules["dspy._vendor.lm15.types"], "listener_bridge": sys.modules["dspy.clients.engines.streaming"],
        "stream_listener": sys.modules["dspy.streaming.streaming_listener"],
        "call_result": sys.modules["dspy.clients.call_result"]}.items()}
    receipt = {
        "format_version": 1,
        "evidence_kind": "actual installed DSPy/Pydantic/lm15 calls and dspy.LM HTTP to localhost fixtures; not inference",
        "dspy": dspy.__version__, "pydantic": pydantic.VERSION,
        "pydantic_core": importlib.metadata.version("pydantic-core"), "python": sys.version.split()[0],
        "oracle": source_receipt(__file__), "sources": sources,
        "encoding": "python_type tags; *_json keys hold raw JSON-native wire data; ensure_ascii=True",
        "counts": counts(rows), "rows": rows,
    }
    Path(args.output).write_text(json.dumps(receipt, ensure_ascii=True, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": args.output, "rows": len(rows), "counts": receipt["counts"]}))


if __name__ == "__main__":
    main()
