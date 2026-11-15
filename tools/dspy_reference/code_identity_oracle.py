"""Observe actual installed DSPy 3.5 Code class identity and serialization; never run native code.

Run: tools/dspy_reference/.venv/bin/python tools/dspy_reference/code_identity_oracle.py [--output PATH]

Every expected value is produced by the pinned Python libraries. Actual Code instances are
tagged with an explicit class label so JSON never erases their concrete class. Two separate
``dspy.Code["JAVA"]`` subscripts create two FRESH Python classes (``java_a``/``java_b``) with the
same name and language; they are recorded as distinct classes, never aliased. The native Zig port
canonicalizes ``CodeWithLanguage("JAVA")`` to one static type, so its class identity corresponds
to only ONE of those fresh classes per program location; that remains an explicit global
boundary, reported under ``boundaries`` rather than hidden.
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
from dspy.adapters import utils as adapter_utils
from dspy.adapters.types import base_type
from dspy.adapters.types import code as code_module
from dspy.adapters.utils import format_field_value, serialize_for_json


Base = dspy.Code
JavaA = dspy.Code["JAVA"]
JavaB = dspy.Code["JAVA"]  # A second, independently created class.
Python = dspy.Code["python"]
UnicodeLanguage = dspy.Code["İΣ"]

CLASSES = [("base", Base), ("java_a", JavaA), ("java_b", JavaB), ("python", Python),
           ("unicode", UnicodeLanguage)]
LABELS = {id(cls): label for label, cls in CLASSES}

CONTENTS = ["", "x = 1", "x = 'a\"b'\n", "雪😀é", "\x00\x08\n\t\x7f\u2028",
            "```java\nrun();\n```"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_receipt(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def label(cls):
    try:
        return LABELS[id(cls)]
    except KeyError:
        raise TypeError(f"Oracle needs an explicit class label for {cls!r}") from None


def encode(value):
    """Explicitly tag Code instances with their concrete class label and object identity."""
    if isinstance(value, Base):
        cls = type(value)
        return {"python_type": "Code", "class_label": label(cls), "class_name": cls.__name__,
                "language": cls.language, "code": value.code}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {"python_type": "dict", "items": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, list):
        return {"python_type": "list", "items": [encode(item) for item in value]}
    if isinstance(value, tuple):
        return {"python_type": "tuple", "items": [encode(item) for item in value]}
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


def hash_view(value):
    try:
        hash(value)
        return {"hashable": True}
    except TypeError as exc:
        return {"hashable": False, "error": type(exc).__name__, "message": str(exc)}


def object_views(value):
    return {
        "value": encode(value),
        "str": str(value),
        "repr": repr(value),
        "format_builtin": format(value),
        "fstring": f"{value}",
        "format": value.format(),
        "serialize_model": value.serialize_model(),
        "model_dump": encode(value.model_dump()),
        "model_dump_json_mode": encode(value.model_dump(mode="json")),
        "model_dump_round_trip": encode(value.model_dump(round_trip=True)),
        "model_dump_json_mode_round_trip": encode(value.model_dump(mode="json", round_trip=True)),
        "model_dump_json": value.model_dump_json(),
        "model_dump_json_round_trip": value.model_dump_json(round_trip=True),
        "serialize_for_json": encode(serialize_for_json(value)),
        "json_dumps_serialized": json.dumps(serialize_for_json(value), ensure_ascii=False),
        "tool_result_content": encode(adapter_base._tool_result_content(value)),
        "format_field_value_input": encode(format_field_value(dspy.InputField(), value)),
        "bool": bool(value),
        "hash": hash_view(value),
        "dict": encode(dict(value.__dict__)),
    }


def class_rows():
    rows = []
    for name, cls in CLASSES:
        rows.append({
            "class_label": name, "class_name": cls.__name__, "qualname": cls.__qualname__,
            "module": cls.__module__, "language": cls.language,
            "is_type": issubclass(cls, base_type.Type), "is_code": issubclass(cls, Base),
            "bases": [base.__name__ for base in cls.__bases__],
            "base_is_dspy_code": cls.__bases__[0] is Base if cls is not Base else None,
            "hash_is_none": cls.__hash__ is None,
            "description": cls.description(),
            "schema": cls.model_json_schema(),
        })
    identity = [{"left": a, "right": b, "same_class": ca is cb, "left_subclass_of_right": issubclass(ca, cb)}
                for a, ca in CLASSES for b, cb in CLASSES]
    return {"classes": rows, "identity": identity}


def validation_cases():
    actual = {name: cls(code="```java\nrun();\n```") for name, cls in CLASSES}
    inputs = [
        *[(f"actual:{name}", value) for name, value in actual.items()],
        *[(f"string:{index}", content) for index, content in enumerate(CONTENTS)],
        ("fenced_multiline", "prefix ```python\n  print(1)\n``` suffix"),
        ("fenced_inline", "```first``` gap ```second```"),
        ("unclosed", " \t```python\n body\n\u2003"),
        ("dict", {"code": "```x```"}),
        ("dict_extra", {"code": " ```y``` ", "extra": True, "language": "go"}),
        ("dict_actual", {"code": actual["java_a"]}),
        ("empty_dict", {}), ("other_key", {"other": "x"}), ("code_null", {"code": None}),
        ("code_integer", {"code": 7}), ("code_list", {"code": ["x"]}),
        ("null", None), ("false", False), ("integer", 7), ("float", 1.5), ("list", ["x"]),
        ("reasoning", dspy.Reasoning(content="x")),
    ]
    rows = []
    for annotation_label, cls in CLASSES:
        for entrypoint, validate in [("model_validate", cls.model_validate),
                                     ("type_adapter", TypeAdapter(cls).validate_python)]:
            for case_id, value in inputs:
                row = {"annotation": annotation_label, "entrypoint": entrypoint, "id": case_id}
                try:
                    row["input"] = encode(value)
                except TypeError:
                    row["input"] = {"python_type": type(value).__name__, "repr": repr(value)}
                try:
                    result = validate(value)
                    row.update({"ok": True, "result": encode(result), "same_instance": result is value})
                    if isinstance(value, dict):
                        row["input_unchanged"] = encode(value)
                except Exception as exc:
                    row.update(error(exc))
                rows.append(row)
    optional = TypeAdapter(Base | None)
    rows.append({"annotation": "optional_base", "entrypoint": "type_adapter", "id": "null",
                 **observe(lambda: optional.validate_python(None))})
    return rows


def constructor_cases():
    rows = []
    for name, cls in CLASSES:
        for case_id, fn in [("keyword_fenced", lambda cls=cls: cls(code="```py\nx\n```")),
                            ("keyword_plain", lambda cls=cls: cls(code="x")),
                            ("keyword_empty", lambda cls=cls: cls(code="")),
                            ("missing", lambda cls=cls: cls()),
                            ("null", lambda cls=cls: cls(code=None)),
                            ("extra", lambda cls=cls: cls(code="x", extra=1))]:
            rows.append({"class_label": name, "id": case_id, **observe(fn)})
    return rows


def view_cases():
    rows = []
    for name, cls in CLASSES:
        for index, content in enumerate(CONTENTS):
            instance = cls.model_validate(content)
            rows.append({"class_label": name, "content_index": index, "input": content,
                         "views": object_views(instance)})
    return rows


def equality_cases():
    content = "x = 1"
    left = {name: cls(code=content) for name, cls in CLASSES}
    comparands = [
        *[(f"{name}_equal", cls(code=content)) for name, cls in CLASSES],
        *[(f"{name}_different", cls(code="other")) for name, cls in CLASSES],
        ("string", content), ("dict", {"code": content}), ("null", None), ("empty_string", ""),
    ]
    rows = []
    for left_label, value in left.items():
        rows.append({"left": left_label, "right": "same_instance", "eq": value == value,
                     "ne": value != value, "identity": True})
        for right_label, other in comparands:
            rows.append({"left": left_label, "right": right_label, "other": encode(other),
                         "left_eq_right": value == other, "right_eq_left": other == value,
                         "left_ne_right": value != other, "right_ne_left": other != value,
                         "identity": value is other})
    java = left["java_a"]
    rows.append({"left": "nested", "list_actual_vs_string": [java] == [content],
                 "list_string_vs_actual": [content] == [java],
                 "list_actual_vs_dict": [java] == [{"code": content}],
                 "list_base_vs_java": [left["base"]] == [java],
                 "list_java_a_vs_java_b": [java] == [left["java_b"]],
                 "dict_actual_vs_string": {"c": java} == {"c": content}})
    return rows


def nested_cases():
    base = Base(code="nested 雪\n'\"")
    java = JavaA(code="run();")
    value = {"values": [base, java, base.code, {"code": base.code}, None],
             "tuple": (java,), "dict": {"code": java}}
    adapter = TypeAdapter(dict)
    return {
        "input": encode(value),
        "str": str(value),
        "repr": repr(value),
        "serialize_for_json": encode(serialize_for_json(value)),
        "json_dumps_serialized": json.dumps(serialize_for_json(value), ensure_ascii=False),
        "type_adapter_python": encode(adapter.dump_python(value)),
        "type_adapter_python_round_trip": encode(adapter.dump_python(value, round_trip=True)),
        "type_adapter_json": encode(adapter.dump_python(value, mode="json")),
        "type_adapter_json_round_trip": encode(adapter.dump_python(value, mode="json", round_trip=True)),
        "type_adapter_json_bytes": adapter.dump_json(value).decode(),
        "type_adapter_json_bytes_round_trip": adapter.dump_json(value, round_trip=True).decode(),
        "tool_result_content": encode(adapter_base._tool_result_content(value)),
        "list_tool_result_content": encode(adapter_base._tool_result_content([base, java])),
        "string_tool_result_content": encode(adapter_base._tool_result_content(base.code)),
        "dict_tool_result_content": encode(adapter_base._tool_result_content({"code": base.code})),
        "original_after_serialization": encode(value),
    }


def copy_cases():
    java = JavaA(code="```java\ncopy 雪\n```")
    base = Base(code="base")
    graph = {"first": java, "second": java, "items": [java, base, java]}
    copied = copy.deepcopy(graph)
    shallow, deep = copy.copy(java), copy.deepcopy(java)
    model_copy = java.model_copy()
    typed_base = TypeAdapter(list[Base]).validate_python([java, "```new```", base, java])
    typed_java = TypeAdapter(list[JavaA]).validate_python([java, {"code": "```x```"}, java])
    nested = TypeAdapter(dict[str, list[Base | None]]).validate_python(
        {"items": [java, {"code": "dict", "extra": 1}, None, base]})
    return {
        "shallow": encode(shallow), "shallow_identity": shallow is java,
        "shallow_class_same": type(shallow) is JavaA, "shallow_equal": shallow == java,
        "deep": encode(deep), "deep_identity": deep is java,
        "deep_class_same": type(deep) is JavaA, "deep_equal": deep == java,
        "deep_code_unfiltered": deep.code == java.code,
        "model_copy": encode(model_copy), "model_copy_class_same": type(model_copy) is JavaA,
        "graph": encode(copied), "graph_values_equal": copied == graph,
        "graph_first_new_identity": copied["first"] is not java,
        "graph_alias_preserved": copied["first"] is copied["second"] is copied["items"][0] is copied["items"][2],
        "graph_class_labels": [label(type(item)) for item in copied["items"]],
        "typed_base_list": encode(typed_base),
        "typed_base_identity": [typed_base[0] is java, typed_base[2] is base, typed_base[3] is java],
        "typed_java_list": encode(typed_java),
        "typed_java_identity": [typed_java[0] is java, typed_java[2] is java],
        "typed_java_b_rejects_java_a": observe(lambda: TypeAdapter(list[JavaB]).validate_python([java])),
        "typed_nested": encode(nested),
        "typed_nested_identity": [nested["items"][0] is java, nested["items"][3] is base],
    }


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
        "sources": {"code": source_receipt(inspect.getsourcefile(code_module)),
                    "base_type": source_receipt(inspect.getsourcefile(base_type)),
                    "adapter_base": source_receipt(inspect.getsourcefile(adapter_base)),
                    "adapter_utils": source_receipt(inspect.getsourcefile(adapter_utils)),
                    "pydantic_main": source_receipt(inspect.getsourcefile(pydantic.main))},
        "classes": class_rows(),
        "constructors": constructor_cases(),
        "validation": validation_cases(),
        "views": view_cases(),
        "equality": equality_cases(),
        "nested": nested_cases(),
        "copies": copy_cases(),
        "boundaries": {
            "fresh_same_language_classes_distinct": JavaA is not JavaB,
            "fresh_same_language_names_equal": JavaA.__name__ == JavaB.__name__,
        },
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
