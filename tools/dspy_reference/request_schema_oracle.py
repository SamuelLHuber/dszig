#!/usr/bin/env python3
"""Observe DSPy 3.5's generated-model request schema closure, not raw formats.

Development oracle only: the Zig runtime never imports Python or this script.
All expected values and failures come from the installed Source function.
"""
import hashlib
import inspect
import json
import pathlib
import random
import sys

import dspy
from dspy.adapters.types.citation import Citations
from dspy.clients.requests import close_object_schemas

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")

cases = []
def add(name, schema):
    cases.append((name, schema))

add("citation_model", Citations.model_json_schema())
add("nested_citation_model", Citations.Citation.model_json_schema())
for value in (None, True, False, 7, "schema", [], {}):
    add("root_" + repr(value), value)
for default in (None, False, 0, "", [], {}):
    add("default_" + repr(default), {"type": "string", "default": default})
for keyword in ("items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"):
    add("schema_child_" + keyword, {keyword: {"type": "object", "properties": {"x": {"default": None}}}})
for keyword in ("items", "prefixItems", "anyOf", "oneOf", "allOf"):
    add("schema_array_" + keyword, {keyword: [{"type": "object", "properties": {"x": {"default": None}}}, {"type": "null"}]})
for keyword in ("properties", "patternProperties", "$defs", "definitions"):
    add("schema_map_" + keyword, {keyword: {"x": {"type": "object", "properties": {"y": {"default": None}}}}})
add("preserve_schema_data", {"default": {"type": "object", "default": None}, "examples": [{"default": None}], "x-meta": {"default": None}})
add("required_property_order", {"properties": {"b": {}, "a": {}}, "required": ["a"]})
for additional in (True, False, {"type": "object"}):
    add("additional_" + repr(additional), {"type": "object", "additionalProperties": additional})
add("allof_single", {"title": "outer", "allOf": [{"type": "object", "title": "inner", "properties": {"x": {}}}]})
add("ref_only_default_null", {"$ref": "bad", "default": None})
add("ref_only", {"$ref": "bad"})
add("ref_external_sibling", {"$ref": "https://example.invalid/schema", "title": "x"})
add("ref_missing_sibling", {"$ref": "#/$defs/missing", "title": "x"})
add("ref_sibling_overrides", {"$defs": {"x": {"type": "object", "title": "base", "properties": {"a": {"default": None}}}}, "items": {"$ref": "#/$defs/x", "title": "override", "description": "sibling"}})
add("ref_scalar_target", {"$defs": {"x": 7}, "$ref": "#/$defs/x", "title": "keep"})
add("ref_null_target", {"$defs": {"x": None}, "$ref": "#/$defs/x", "title": "refuse"})
add("ref_literal_segment", {"$defs": {"~1": {"type": "string"}}, "$ref": "#/$defs/~1", "title": "literal"})
add("ref_not_pointer_unescaped", {"$defs": {"/": {"type": "string"}}, "$ref": "#/$defs/~1", "title": "refuse"})

rng = random.Random(350)
def node(depth):
    if not depth:
        return rng.choice([{"type": "string", "default": None}, {"type": "integer", "default": 0}, {"type": "null"}, {}])
    kind = rng.randrange(5)
    if kind == 0:
        return {"type": "object", "properties": {"a": node(depth - 1), "b": node(depth - 1)}, "required": ["a"]}
    if kind == 1:
        return {"type": "array", "items": node(depth - 1)}
    if kind == 2:
        return {"anyOf": [node(depth - 1), {"type": "null"}], "default": None}
    if kind == 3:
        return {"allOf": [node(depth - 1)], "title": "outer"}
    return {"not": node(depth - 1), "if": {"type": "object"}, "then": node(depth - 1), "else": {"type": "null"}}
for index in range(128):
    add(f"composed_{index}", node(4))

rows = []
for name, schema in cases:
    before = json.dumps(schema, ensure_ascii=True)
    try:
        value = close_object_schemas(schema)
        row = {"id": name, "input": schema, "ok": True, "value": value}
    except Exception as error:
        row = {"id": name, "input": schema, "ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
    if json.dumps(schema, ensure_ascii=True) != before:
        raise AssertionError("Source mutated the input schema")
    rows.append(row)
source = pathlib.Path(inspect.getsourcefile(close_object_schemas))
receipt = {"dspy": dspy.__version__, "evidence_kind": "actual installed generated-model request close_object_schemas; not raw-format rewriting or inference", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": rows}
pathlib.Path(sys.argv[1]).write_text(json.dumps(receipt, ensure_ascii=True, indent=2) + "\n")
print(f"Request schema closure: {len(rows)} actual Source observations, {sum(not r['ok'] for r in rows)} errors")
