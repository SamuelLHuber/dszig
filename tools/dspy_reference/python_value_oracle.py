#!/usr/bin/env python3
"""Observe genuine CPython 3.11 and DSPy 3.5 builtin-value behavior.

This is a development oracle, never a native runtime dependency. The receipt
retains builtin classes, ordered keys, exact integers and binary64 bits. Set
iteration/repr observations are process-specific unless PYTHONHASHSEED is fixed.
Each row also observes reconstruction from its encoded container iteration order.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
import struct
import sys
from pathlib import Path

import dspy
import pydantic
from dspy.adapters import utils
from pydantic import TypeAdapter

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("This oracle requires CPython 3.11")


class BrokenFile(dspy.File):
    """A real Source TypeFormatError; never substitutes for native behavior."""

    def format(self):
        return [{"type": "file"}]


class StringKey:
    """A genuine hashable Python key with cross-value equality."""

    def __init__(self, text, ordered=False):
        self.text = text
        self.ordered = ordered

    def __hash__(self):
        return 0 if self.ordered else hash(self.text)

    def __eq__(self, other):
        if isinstance(other, StringKey):
            if self.ordered != other.ordered:
                return False
            return self.text <= other.text if self.ordered else self.text == other.text
        return False if self.ordered else self.text == other

    def __repr__(self):
        suffix = ", ordered=True" if self.ordered else ""
        return f"StringKey({self.text!r}{suffix})"


def decimal(value):
    negative = value < 0
    value = abs(value)
    chunks = []
    while value:
        value, remainder = divmod(value, 10**9)
        chunks.append(remainder)
    if not chunks:
        return "0"
    return ("-" if negative else "") + str(chunks[-1]) + "".join(
        f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


def enc(value):
    if value is None:
        return {"null": True}
    if value is Ellipsis:
        return {"ellipsis": True}
    if type(value) is bool:
        return {"bool": value}
    if type(value) is int:
        return {"int": decimal(value)}
    if type(value) is float:
        return {"float": repr(value), "bits": struct.pack(">d", value).hex()}
    if type(value) is complex:
        return {"complex": [enc(value.real), enc(value.imag)]}
    if type(value) is str:
        return {"string_codepoints": [ord(char) for char in value]}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) in (list, tuple, set, frozenset):
        return {type(value).__name__: [enc(item) for item in value]}
    if type(value) is dict:
        return {"dictionary": [[enc(key), enc(item)] for key, item in value.items()]}
    raise TypeError(f"Unexpected Source value {type(value)!r}")


def attempt(function):
    try:
        return {"ok": True, "value": enc(function())}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def rebuild_from_encoding(value):
    """Apply genuine Python constructors in the encoded child iteration order."""
    kind = type(value)
    if kind in (list, tuple, set, frozenset):
        return kind(rebuild_from_encoding(item) for item in value)
    if kind is dict:
        return dict((rebuild_from_encoding(key), rebuild_from_encoding(item))
                    for key, item in value.items())
    if kind is float:
        return struct.unpack(">d", struct.pack(">d", value))[0]
    if kind is complex:
        return complex(value.real, value.imag)
    return value


def observe(value):
    return {"type": type(value).__name__, "value": enc(value),
            "str": str(value), "repr": repr(value), "truth": bool(value),
            "hash": attempt(lambda: hash(value)),
            "plain_json": attempt(lambda: json.dumps(value)),
            "pydantic_json": attempt(lambda: TypeAdapter(type(value)).dump_python(value, mode="json")),
            "pydantic_json_with_fallback": attempt(lambda: TypeAdapter(type(value)).dump_python(value, mode="json", fallback=str)),
            "dspy_serialized": attempt(lambda: utils.serialize_for_json(value)),
            "deepcopy": attempt(lambda: copy.deepcopy(value))}


def cases():
    yield "bytes/empty", b""
    yield "bytes/ascii", b"plain"
    yield "bytes/single_quote", b"'"
    yield "bytes/double_quote", b'"'
    yield "bytes/both_quotes", b"'\""
    yield "bytes/all_octets", bytes(range(256))
    yield "bytes/utf8", "snow \u96ea".encode()
    yield "bytes/invalid_utf8", b"\xff\xfe\xc0\x80"
    yield "bytes/nul", b"a\x00b"
    yield "tuple/empty", ()
    yield "tuple/single", (1,)
    yield "tuple/unhashable_child", ([],)
    yield "tuple/nested", (b"x", (1, True), {"x": [1]})
    yield "set/empty", set()
    yield "set/integers", set([9, -1, 1, 4, 0, 32, 8, 3])
    yield "set/collisions", set([0, 8, 16, 24, 32, 40, 48])
    yield "set/deepcopy_reorders", set([943, -417])
    yield "set/strings", set(["x", "y", "z", "\u00e9", "\u96ea", "\U0001f600"])
    yield "set/bytes", set([b"x", b"y", b"z", b"\xff"])
    yield "set/equal_numeric", set([True, 1, 1.0, complex(1, 0), False, 0, -0.0])
    yield "set/tuples", set([(1, 2), (2, 1), (), ("x",)])
    yield "frozenset/empty", frozenset()
    yield "frozenset/integers", frozenset([9, -1, 1, 4, 0, 32, 8, 3])
    yield "frozenset/deepcopy_reorders", frozenset([943, -417])
    yield "frozenset/nested", frozenset([frozenset([1, 2]), frozenset([3]), ()])
    for real, imag in [
        (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0),
        (1.0, 2.0), (1.0, -2.0), (0.0, 1e16), (1e-5, -1e-5),
        (float("inf"), float("-inf")), (float("nan"), float("nan")),
        (-float("nan"), -float("nan")),
    ]:
        value = complex(real, imag)
        yield "complex/" + struct.pack(">dd", real, imag).hex(), value
    yield "ellipsis", Ellipsis
    yield "tuple/ellipsis", (Ellipsis,)
    yield "list/ellipsis", [Ellipsis]
    yield "set/ellipsis", {Ellipsis}
    yield "frozenset/ellipsis", frozenset([Ellipsis])
    yield "dict/string_keys", dict([("x", 1), ("y", 2), ("x", 3)])
    yield "dict/numeric_replacement", dict([(True, "bool"), (1.0, "float"), (complex(1, 0), "complex"), (0, "zero")])
    yield "dict/float_retained", dict([(1.0, "float"), (True, "bool")])
    yield "dict/arbitrary_keys", {b"x": 1, (1, 2): b"y", frozenset([3]): 4, Ellipsis: None}
    yield "dict/stringification_collision", {1: "integer", "1": "string", None: "null", "None": "text"}
    yield "dict/invalid_utf8_value", {"x": b"\xff", "sibling": 7}
    yield "dict/invalid_utf8_key", {b"\xff": "x", "sibling": 7}
    yield "dict/tuple_key", {(1, 2): "x"}
    yield "dict/bytes_key", {b"x": "y"}
    yield "dict/complex_key", {complex(1, 2): "x"}
    yield "dict/frozenset_key", {frozenset([1, 2]): "x"}
    yield "dict/nested_invalid_utf8", {"x": (b"\xff",)}
    yield "dict/empty_tuple_key", {(): 1}
    yield "dict/nested_tuple_key", {(1, (2, 3)): 1}
    yield "dict/flattened_tuple_collision", {(1, 2): 1, "1,2": 2}
    yield "dict/ellipsis_key", {Ellipsis: 1}
    yield "dict/nonfinite_keys", {float("nan"): 1, float("inf"): 2, -float("inf"): 3, None: 4}
    yield "dict/finite_float_keys", {-0.0: 0, 1e-5: 1, 1e16: 2, 1e20: 3}
    for index, text in enumerate(["", "x", "\u00e9", "\u96ea", "\U0001f600", "\ud800", "a\u96ea\U0001f600"]):
        yield f"string/hash/{index}", text
    for index, value in enumerate([0.0, -0.0, 0.5, -1.0, 1e-5, 1e16, 1e100, 5e-324, float("inf"), -float("inf")]):
        yield f"float/hash/{index}", value
    yield "float/nan", float("nan")
    yield "integer/wide", 10**100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for name, value in cases():
        row = {"id": name, **observe(value),
               "reconstructed_from_encoding": observe(rebuild_from_encoding(value))}
        rows.append(row)
    equalities = []
    left_nan = float("nan")
    right_nan = float("nan")
    pairs = [
        (1, True), (1.0, True), (complex(1, -0.0), True),
        (complex(9007199254740992, 0), 9007199254740993),
        (complex(1e100, 0), 10**100), (complex(0, float("nan")), 0),
        (b"x", "x"), ((1,), [1]), ((True, b"x"), (1.0, b"x")),
        (set([1, 2]), frozenset([2.0, True])),
        ({(1,): "x"}, {(True,): "x"}),
        (left_nan, left_nan), (left_nan, right_nan),
        ((left_nan,), (left_nan,)), ((left_nan,), (right_nan,)),
        (set([left_nan]), set([left_nan])),
        (set([left_nan]), set([right_nan])),
    ]
    for left, right in pairs:
        equalities.append({"left": enc(left), "right": enc(right),
                           "equal": left == right, "reverse_equal": right == left})
    failures = []
    for name, function in [
        ("set/list", lambda: set([[]])),
        ("set/dict", lambda: set([{}])),
        ("set/mutable_set", lambda: set([set()])),
        ("set/tuple_with_list", lambda: set([([],)])),
        ("frozenset/list", lambda: frozenset([[]])),
        ("dictionary/list_key", lambda: dict([([], 1)])),
        ("dictionary/tuple_with_list_key", lambda: dict([(([],), 1)])),
    ]:
        failures.append({"id": name, "result": attempt(function)})
    model_keys = []
    for key in [dspy.File(file_id="id"), dspy.File(file_id="x" * 2000),
                dspy.BoundingBox(top=0, left=0, bottom=1, right=1)]:
        model_keys.append({"type": type(key).__name__, "repr": repr(key),
                           "hash": attempt(lambda: hash(key)),
                           "dspy_serialized": attempt(lambda: utils.serialize_for_json({key: 7}))})
    broken = BrokenFile(file_id="id")
    ordered_failures = []
    for name, value in [
        ("type_error_before_invalid_utf8_key", {0: broken, b"\xff": 1}),
        ("invalid_utf8_key_before_type_error", {b"\xff": 1, 0: broken}),
        ("overwritten_value_still_serialized", {1: broken, "1": "replacement"}),
    ]:
        ordered_failures.append({"id": name, "repr": repr(value),
                                 "result": attempt(lambda: utils.serialize_for_json(value))})
    large_integer = 10**4300
    integer_conversion_limits = [
        {"id": "tuple_repr", "result": attempt(lambda: repr((large_integer,)))},
        {"id": "dict_repr", "result": attempt(lambda: repr({large_integer: 1}))},
        {"id": "dict_serialization", "result": attempt(lambda: utils.serialize_for_json({large_integer: 1}))},
    ]
    key = StringKey("x")
    custom_first = {key: 1, "x": 2}
    string_first = {"x": 1, key: 2}
    cross_value_keys = {
        "key_equals_string": key == "x",
        "string_equals_key": "x" == key,
        "hash_matches_string": hash(key) == hash("x"),
        "dictionary_equals_string_dictionary": {key: 1} == {"x": 1},
        "string_dictionary_equals_dictionary": {"x": 1} == {key: 1},
        "custom_first_retained_type": type(next(iter(custom_first))).__name__,
        "string_first_retained_type": type(next(iter(string_first))).__name__,
        "custom_first_repr": repr(custom_first),
        "string_first_repr": repr(string_first),
        "custom_first_set_repr": repr(set([key, "x"])),
        "string_first_set_repr": repr(set(["x", key])),
        "serialized_custom_first": attempt(lambda: utils.serialize_for_json(custom_first)),
    }
    lower = StringKey("a", ordered=True)
    upper = StringKey("b", ordered=True)
    cross_value_keys["asymmetric"] = {
        "scalar_forward": lower == upper,
        "scalar_reverse": upper == lower,
        "dictionary_forward": {lower: 1} == {upper: 1},
        "dictionary_reverse": {upper: 1} == {lower: 1},
        "set_forward": {lower} == {upper},
        "set_reverse": {upper} == {lower},
        "frozen_cross_forward": frozenset([lower]) == {upper},
        "frozen_cross_reverse": {upper} == frozenset([lower]),
        "dedup_forward_repr": repr(set([lower, upper])),
        "dedup_reverse_repr": repr(set([upper, lower])),
    }
    dictionary_probes = []
    for name, prefix, sequence in [
        ("five_collision_keys", [], "zycbad"),
        ("entry_order_growth", [], "zyxwvutsrqponmlkjihgfedcbaf"),
        ("unicode_to_general", ["s", "t", "u"], "zycbad"),
    ]:
        keys = prefix + [StringKey(text, ordered=True) for text in sequence]
        inputs = list(zip(keys, range(1, len(keys) + 1)))
        value = dict(inputs)
        def descriptor(item):
            return ({"string": item} if type(item) is str else
                    {"StringKey": {"text": item.text, "ordered": item.ordered}})
        dictionary_probes.append({
            "id": name,
            "input_keys": [descriptor(item) for item in keys],
            "input_values": [item for _, item in inputs],
            "retained_keys": [descriptor(item) for item in value],
            "retained_values": list(value.values()),
            "repr": repr(value),
            "dspy_serialized": attempt(lambda: utils.serialize_for_json(value)),
        })
    cross_value_keys["dictionary_probes"] = dictionary_probes
    source = Path(inspect.getfile(utils))
    receipt = {"evidence_kind": "actual_cpython_builtin_and_dspy_serialization_calls",
               "python": sys.version, "dspy": dspy.__version__, "pydantic": pydantic.__version__,
               "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random"),
               "int_max_str_digits": sys.get_int_max_str_digits(),
               "hash_info": {key: getattr(sys.hash_info, key) for key in
                             ("width", "modulus", "inf", "imag", "algorithm", "hash_bits", "seed_bits", "cutoff")},
               "dspy_utils_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "cases": rows, "equalities": equalities, "unhashable_failures": failures,
               "model_key_observations": model_keys, "ordered_serialization_failures": ordered_failures,
               "integer_conversion_limits": integer_conversion_limits,
               "cross_value_keys": cross_value_keys}
    text = json.dumps(receipt, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"Observed {len(rows)} builtin values, {len(equalities)} equality pairs, {len(failures)} unhashable failures; sha256={hashlib.sha256(text.encode()).hexdigest()}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
