#!/usr/bin/env python3
"""Independent CPython 3.11 ast.literal_eval observations and Unicode data.

No native implementation is imported or executed. Integer serialization uses
small decimal chunks, never sys.set_int_max_str_digits. Set observations are
canonically sorted because Python sets have no iteration-order contract; dicts
retain insertion order. --write-unicode regenerates the literal parser's name
and identifier tables from the *running pinned interpreter*, including aliases
through the actual ucnhash C API (not a separately downloaded Unicode version).
--dspy-boundary additionally records actual DSPy repair-first Any parsing, with
its original-string fallback eligibility and uncaught evaluation failures.

Recursion observations use this process's actual Python frames and default
recursionlimit=1000. CPython's AST cutoff depends on both; it is not identical
to the native deterministic 3000-expression-node ceiling. That native policy
does not emulate arbitrary caller stacks, nondefault recursion limits, or every
composite C-parser resource refusal. No recursion settings are mutated here.

Usage:
  tools/dspy_reference/.venv/bin/python -I tools/dspy_reference/python_literal_oracle.py --output /tmp/dszig-python-literal-oracle.json
  tools/dspy_reference/.venv/bin/python -I tools/dspy_reference/python_literal_oracle.py --write-unicode src/python_literal_unicode.bin --output /tmp/dszig-python-literal-oracle.json
  tools/dspy_reference/.venv/bin/python -I tools/dspy_reference/python_literal_oracle.py --all-unicode-names --dspy-boundary --output /tmp/dszig-python-literal-full-oracle.json
"""

import argparse
import ast
import ctypes
import hashlib
import inspect
import json
import random
import struct
import sys
import unicodedata
import warnings
from pathlib import Path


def decimal(value):
    if -10**100 < value < 10**100:
        return str(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    chunks = []
    while value:
        value, remainder = divmod(value, 10**9)
        chunks.append(remainder)
    return sign + str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))


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
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            return {"string_codepoints": [ord(c) for c in value]}
        return {"string": value}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) in (list, tuple):
        return {"array" if type(value) is list else "tuple": [enc(item) for item in value]}
    if type(value) is set:
        items = [enc(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True))
        return {"set": items}
    if type(value) is dict:
        return {"object": [[enc(key), enc(item)] for key, item in value.items()]}
    raise TypeError(f"Unrecognized literal result type {type(value)!r}")


def category(exc):
    if isinstance(exc, MemoryError):
        return "OutOfMemory"
    if isinstance(exc, RecursionError):
        return "JsonTooDeep"
    if isinstance(exc, (TypeError, OverflowError)):
        return "LiteralEvaluation"
    if "Exceeds the limit (4300 digits)" in str(exc):
        return "IntegerTooLong"
    return "ParseValue"


def observe(source):
    # Parsing failures and observation/serialization bugs must not be conflated.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = ast.literal_eval(source)
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "category": category(exc),
                "caught_by_dspy_literal_fallback": isinstance(exc, (ValueError, SyntaxError)),
                "message": str(exc)}
    return {"ok": True, "value": enc(value)}


def set_name_mappings():
    for cp in range(0x110000):
        char = chr(cp)
        normalized = unicodedata.normalize("NFKC", char)
        # NFKC does not delete scalars. An identifier normalizing to exactly
        # 'set' can therefore use only these components; no general-purpose
        # identifier/normalization database is needed by literal_eval.
        if 0 < len(normalized) <= 3 and set(normalized) <= set("set") and char.isidentifier():
            yield cp, normalized


def unicode_tables():
    if sys.version_info[:2] != (3, 11) or unicodedata.unidata_version != "14.0.0":
        raise RuntimeError("Unicode generation requires pinned CPython 3.11 / Unicode 14.0.0")
    names = {}
    cjk_ranges = []
    for cp in range(0x110000):
        name = unicodedata.name(chr(cp), None)
        if name is None:
            continue
        if name.startswith("CJK UNIFIED IDEOGRAPH-"):
            if cjk_ranges and cjk_ranges[-1][1] == cp - 1:
                cjk_ranges[-1][1] = cp
            else:
                cjk_ranges.append([cp, cp])
        elif not name.startswith("HANGUL SYLLABLE "):
            names[name] = cp

    getname_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    getcode_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_uint32), ctypes.c_int)

    class NameAPI(ctypes.Structure):
        _fields_ = [("getname", getname_type), ("getcode", getcode_type)]

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(unicodedata._ucnhash_CAPI, b"unicodedata._ucnhash_CAPI")
    api = ctypes.cast(pointer, ctypes.POINTER(NameAPI)).contents
    buffer = ctypes.create_string_buffer(256)
    aliases = 0
    # CPython stores aliases and named sequences in private-use plane 15.
    # getcode(..., with_named_seq=0) is the exact Unicode-escape restriction.
    for cp in range(0xF0000, 0x100000):
        if not api.getname(cp, buffer, len(buffer), 1):
            continue
        name = buffer.value
        resolved = ctypes.c_uint32()
        if api.getcode(name, len(name), ctypes.byref(resolved), 0):
            names[name.decode("ascii")] = resolved.value
            aliases += 1
    if aliases != 470:
        raise RuntimeError(f"Unexpected pinned Unicode alias count: {aliases}")
    return sorted(names.items()), list(set_name_mappings()), cjk_ranges


def write_unicode(path):
    names, mappings, cjk_ranges = unicode_tables()
    pool = bytearray()
    entries = bytearray()
    for name, cp in names:
        data = name.encode("ascii")
        entries += struct.pack("<III", cp, len(pool), len(data))
        pool += data
    encoded_mappings = b"".join(struct.pack("<II", cp, int.from_bytes(text.encode("ascii"), "little"))
                                for cp, text in mappings)
    encoded_ranges = b"".join(struct.pack("<II", first, last) for first, last in cjk_ranges)
    data = struct.pack("<8sIII", b"PYLIT14\0", len(names), len(mappings), len(cjk_ranges))
    data += entries + encoded_mappings + encoded_ranges + pool
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "names_including_aliases": len(names), "aliases": 470,
            "set_identifier_components": len(mappings), "cjk_ranges": cjk_ranges}


def corpus(all_unicode_names=False):
    cases = [
        "", " ", "\n", "# comment", "None", "True", "False", "...", "Ellipsis", "none", "true", "false",
        "0", "00", "000", "0_0", "0_00_0", "07", "00_1", "0_1", "+0", "-0", "+1", "-1",
        "9223372036854775807", "9223372036854775808", "-9223372036854775808", "-9223372036854775809",
        "0x10", "0X_FF", "0b_0101", "0B10_01", "0o_77", "0O12_34", "0x", "0b", "0o", "0x__f",
        "0b2", "0o8", "0xfg", "0x1.0", "0x1p2", "0x1j", "1__2", "1_", "_1", "1_.0", "1._0",
        "1e_2", "1e+_2", "1_e2", "1e", "1e+", "1e2_", "1_2.3_4e+5_6", "00.1", "00e1", "0_1.0",
        ".5", "1.", "1.e2", "1.e+2j", "1e309", "1e-9999", "-0.0", "+0.0", "-1e-9999", "nan", "inf",
        "0j", "-0j", "+0j", "00j", "01j", "2J", "1.5j", ".5j", "1e999j", "1e-999j", "-1e999j",
        "1+2j", "1-2j", "-1+2j", "+1-2j", "1.5+2.5j", "-0.0+0j", "-0.0-0j", "0+0j", "0-0j",
        "1e999+1e999j", "1e999-1e999j", "1+0j", "1+(2j)", "((1))+((2j))", "1+-2j", "1-(-2j)",
        "1++2j", "1j+2", "1j+2j", "1+2", "1-2", "1+2j+3j", "-(1+2j)", "+(1+2j)", "--1", "+-1",
        "-(-1)", "+(1)", "-(1j)", "+True", "-False", "+None", "~1", "1*2", "1/2", "1**2", "1|2",
        "()", "(1)", "(1,)", "(1,2)", "1,2", "1,", "1,2,", "(())", "(1,,2)", "[,]", "[1,2,]",
        "[]", "[None, True, False, ...]", "[1, (2,), {'x': 3}]", "{}", "{1}", "{1,2,1,}", "{...}",
        "set()", "set ()", "(set)()", "(((set))) ( )", "set(\n#x\n)", "set", "set(1)", "set(())", "set(,)" ,
        "set(x=1)", "set(*())", "set(**{})", "set()()", "set.__call__()", "frozenset()", "list()", "dict()",
        "{1: 'a', True: 'b', 1.0: 'c', 1+0j: 'd'}", "{'a':1,'b':2,'a':3}", "{None:1, ...:2}",
        "{(1,2):[3],b'x':set()}", "{():1}", "{[]:1}", "{{}:1}", "{set():1}", "{[1]}", "{{1}}", "{([] ,)}",
        "{**{}}", "{1, **{}}", "{1:2,3}", "{1,2:3}", "[x for x in []]", "{x for x in []}", "(x for x in [])",
        "__import__('os')", "(lambda: 1)()", "[1][0]", "(x:=1)", "1 if True else 2", "True and False",
        "  \t1", "\n1", "\n 1", "\n\t1", "\n  # blank\n1", "\n \n1", "\f1", "\f 1", "1\f", "1\v",
        "1 #comment\n", "1\n#comment", "1\n2", "1\n,2", "1,\n2", "1,\n", "1 #c\n,", "1;", "1;2",
        "1\r", "1\r\n", "[1,\r2]", "[1,#c\r\n2]", "set\n()", "'a'\n'b'", "('a'\n'b')",
        "1\\\n+2j", "\\\n 1", "1\\\r\n+2j", "1\\ +2j", "1\\\n#x\n", "1\u00a0", "\ufeff1", "1\0", "#\0\n1",
        "ｓｅｔ()", "𝓼𝓮𝓽()", "ＳＥＴ()", "Ｔｒｕｅ", "Ｎｏｎｅ", "Truex", "None0", "set0()",
        "'a' 'b'", "b'a' b'b'", "b'a' 'b'", "'a' b'b'", "u'a' r'b'", "''", "b''", "''''''", '""""""',
        "'unterminated", "'''unterminated", "'a\nb'", "'''a\nb'''", "'''a\rb\r\nc'''", "r'''a\rb'''",
        r"'\a\b\f\n\r\t\v\\\'\"'", r"'\0\00\000\1\12\123\377\400\777\7777'",
        r"b'\0\377\400\777'", r"'\x00\xff'", r"'\x0'", r"'\xz1'", r"'\u0000\u00ff\ud800\udfff\U0001f642'",
        r"'\ud83d\ude42'", r"'\U0010ffff'", r"'\U00110000'", r"'\u123'", r"'\U0000000'", r"'\q\8\9\/\é'",
        r"b'\u0041\U00000042\N{LATIN CAPITAL LETTER C}'", "b'é'", "br'é'", "'é🙂'", "'\ud800'",
        r"'\N{LATIN CAPITAL LETTER A}'", r"'\N{latin small letter a}'", r"'\N{LF}'", r"'\N{BOM}'", r"'\N{VS256}'",
        r"'\N{HANGUL SYLLABLE GA}'", r"'\N{HANGUL SYLLABLE HIH}'", r"'\N{hangul syllable ga}'",
        r"'\N{CJK UNIFIED IDEOGRAPH-4E00}'", r"'\N{CJK UNIFIED IDEOGRAPH-20000}'", r"'\N{CJK UNIFIED IDEOGRAPH-3134A}'",
        r"'\N{CJK UNIFIED IDEOGRAPH-4e00}'", r"'\N{cjk unified ideograph-4E00}'", r"'\N{CJK UNIFIED IDEOGRAPH-3134B}'",
        r"'\N{KEYCAP DIGIT ONE}'", r"'\N{}'", r"'\N{UNKNOWN}'", r"'\N{ LF}'", r"'\N{LF }'", r"'\N'", r"'\N{LF'",
        r"'\N{CJK UNIFIED IDEOGRAPH-03400}'", r"'\N{CJK UNIFIED IDEOGRAPH-04E00}'",
        r"'\N{HANGUL SYLLABLE A}'", r"'\N{HANGUL SYLLABLE GGAEG}'",
        "\\\n1", "\\\n\t1", "\\\n\f1", "\\\n\f 1", "\\\n\\\n1", "\\\n\\\n 1",
        "\\\n#x\n1", "\\\n#x\n 1", "1\\\n  +2j", "1\\\n\f +2j", "1\\\n", "1\\\n  ",
        "1\\\n#x", "1\n\\\n", "1\n\\\n ", "1\n#x\\\n", "(1)\\\n", "[1]\\\n",
        "set\\\n ()", "set\\\n#x\n()", "set\\\n\n()", "[1\\\n#x\n,2]", "'a'\\\n'b'",
        "r'a\\\rb'", "'a\\\rb'", "0b0001", "0o0001", "0x0001", "00_00j",
    ]
    for index, source in enumerate(cases):
        yield f"literal/{index}", source

    for prefix in ("", "r", "R", "u", "U", "b", "B", "br", "Br", "bR", "BR", "rb", "rB", "Rb", "RB",
                   "f", "F", "fr", "rf", "ur", "ru", "bu", "ub", "rr", "bb", "uuu"):
        for quote in ("'", '"', "'''", '"""'):
            for body in ("abc", r"a\nb", r"a\x41b", "a\\\nb", r"a\'b", "é", "a\r\nb"):
                yield f"string/{prefix}/{quote}/{body!r}", prefix + quote + body + quote

    for n in (4299, 4300, 4301, 5000):
        yield f"int/decimal/{n}", "1" * n
        yield f"int/decimal_underscores/{n}", "_".join("1" * n)
        yield f"int/zero/{n}", "0" * n
        for suffix in ("x", "and True", "else 2", "ifx", "in []", "not in []", "#x", "+1"):
            yield f"int/token_boundary/{n}/{suffix}", "1" * n + suffix
    for index, source in enumerate(("1x", "1_0x", "0x1g", "0b18", "0o78", "1jz", "1e2jz")):
        yield f"number/invalid_suffix/{index}", source
    for base, char in (("0x", "f"), ("0o", "7"), ("0b", "1")):
        for n in (30, 300, 4301, 10000):
            yield f"int/{base}/{n}", base + char * n
    for n in (198, 199, 200, 201, 202):
        yield f"depth/list/{n}", "[" * n + "]" * n
        yield f"depth/group/{n}", "(" * n + "0" + ")" * n
        yield f"depth/tuple/{n}", "(" * n + "0" + ",)" * n
    for n in (307, 308, 309, 400):
        yield f"complex/huge_real/{n}", "9" * n + "+1j"
    for index, source in enumerate((
        "+{[1]}", "1+{[1]}", "[+{[1]}]", "{[]:1} blah",
        "+(" + "9" * 309 + "+1j)", "9" * 309 + "+1",
        "[" + "9" * 309 + "+1j]", "(" + "9" * 309 + "+1j,)",
        "({[]:1},)", "[{[]:1}]", "0x" + "f" * 300 + "+1j",
    )):
        yield f"evaluation_order/{index}", source

    yield from recursion_corpus()

    rng = random.Random(31117)
    for index in range(300):
        value = rng.getrandbits(rng.randrange(0, 2048))
        base = rng.choice((2, 8, 10, 16))
        text = {2: bin, 8: oct, 10: str, 16: hex}[base](value)
        if rng.randrange(2):
            text = "-" + text
        yield f"generated/integer/{index}", text
    for index in range(300):
        digits = str(rng.randrange(10**17))
        exponent = rng.randrange(-400, 401)
        text = rng.choice(("", "+", "-")) + digits + "e" + str(exponent)
        if rng.randrange(3) == 0:
            text += "j"
        yield f"generated/float/{index}", text
    for cp, text in set_name_mappings():
        source = "set".replace(text, chr(cp), 1)
        if source != "set":
            yield f"identifier/{cp:x}", source + "()"
    if all_unicode_names:
        names, _, cjk_ranges = unicode_tables()
        for name, cp in names:
            yield f"unicode/name/{cp:x}/{name}", "'\\N{" + name + "}'"
            yield f"unicode/lowercase_name/{cp:x}/{name}", "'\\N{" + name.lower() + "}'"
        for cp in range(0xAC00, 0xD7A4):
            name = unicodedata.name(chr(cp))
            yield f"unicode/hangul/{cp:x}", "'\\N{" + name + "}'"
        for first, last in cjk_ranges:
            for cp in range(first, last + 1):
                name = unicodedata.name(chr(cp))
                yield f"unicode/cjk/{cp:x}", "'\\N{" + name + "}'"


def recursion_corpus():
    for operator in ("+", "-", "~"):
        for count in (2000, 2970, 2980, 2990, 2999, 3000, 3001):
            yield f"recursion/unary/{operator}/{count}", operator * count + "1"
    for label, source in (
        ("missing_operand", "+" * 3000),
        ("trailing_name", "+" * 3000 + "1 garbage"),
        ("list", "[" + "+" * 3000 + "1,]"),
        ("tuple", "(" + "+" * 3000 + "1,)"),
        ("before_evaluation", "+" * 3000 + "{[1]}"),
        ("before_unsupported", "[" + "+" * 3000 + "1,~1]"),
        ("nested_unary", "+" * 2000 + "(" + "-" * 1000 + "1)"),
        ("separate_unary", "[" + "+" * 2000 + "1," + "-" * 2000 + "1]"),
    ):
        yield f"recursion/{label}", source


def dspy_boundary_receipt():
    # Optional independent integration observations. Never substitute the
    # standalone literal result for what DSPy's repair-first path returned.
    import importlib.metadata
    from typing import Any

    import dspy
    import json_repair
    from dspy.adapters import utils

    repair_version = importlib.metadata.version("json-repair")
    if dspy.__version__ != "3.5.0" or repair_version != "0.64.2":
        raise RuntimeError("Boundary observations require pinned DSPy 3.5.0 / json-repair 0.64.2")
    sources = [
        "", " ", "None", "True", "False", "...", "Ellipsis", "set()", "set ()", "(set)()", "ｓｅｔ()",
        "b'7'", "rb'\\x41'", "b'\\377\\400'", "1+2j", "-1+2j", "-0j",
        "0x10", "0o10", "0b10", "1_000", "+7", "07", "1x",
        "(1,2)", "(1,)", "()", "[]", "{}", "{1,2}", "{(1,2):'x'}",
        "'\\N{LF}'", "'\\ud800'", "Literal['red']", "str[blue]", "__import__('os')",
    ]
    sources.extend("9" * n + "+1j" for n in (307, 308, 309, 400))
    sources.extend(("0x" + "f" * 300 + "+1j", "(set(),)", "{[1]}", "{[]:1}",
                    "+{[1]}", "1+{[1]}", "[+{[1]}]", "{[]:1} blah", "+(" + "9" * 309 + "+1j)",
                    "9" * 309 + "+1", "[" + "9" * 309 + "+1j]", "(" + "9" * 309 + "+1j,)",
                    "({[]:1},)", "[{[]:1}]"))
    for depth in (200, 201):
        sources.extend("(" * depth + body + ")" * depth for body in ("...", "None", "b'x'", "set()"))
    sources.extend(source for _, source in recursion_corpus())
    rows = []
    for index, source in enumerate(sources):
        row = {"id": f"dspy_any/{index}", "source": source, "annotation": "Any"}
        try:
            candidate = json_repair.loads(source)
        except Exception as exc:
            row["repair_return"] = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        else:
            row["repair_return"] = {"ok": True, "value": enc(candidate)}
            row["literal_fallback_eligible"] = candidate == "" and source != ""
            if row["literal_fallback_eligible"]:
                row["standalone_literal"] = observe(source)
        try:
            value = utils.parse_value(source, Any)
        except Exception as exc:
            row["actual_parse_value"] = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        else:
            row["actual_parse_value"] = {"ok": True, "value": enc(value)}
        rows.append(row)
    utils_path = Path(inspect.getfile(utils))
    return {"evidence_kind": "actual_dspy_parse_value_any_calls", "dspy": dspy.__version__,
            "json_repair": repair_version, "recursion_limit": sys.getrecursionlimit(),
            "utils_source_sha256": hashlib.sha256(utils_path.read_bytes()).hexdigest(),
            "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-unicode", type=Path)
    parser.add_argument("--all-unicode-names", action="store_true")
    parser.add_argument("--dspy-boundary", action="store_true",
                        help="also observe actual repair-first DSPy parse_value calls with Any")
    args = parser.parse_args()
    if (sys.version_info[:3] != (3, 11, 17) or sys.get_int_max_str_digits() != 4300
            or sys.getrecursionlimit() != 1000):
        raise RuntimeError("This oracle requires pinned Python 3.11.17, 4300 integer digits, recursionlimit=1000")
    unicode_receipt = write_unicode(args.write_unicode) if args.write_unicode else None
    rows = [{"id": label, "source": source, "result": observe(source)}
            for label, source in corpus(args.all_unicode_names)]
    ast_path = Path(inspect.getfile(ast))
    receipt = {"evidence_kind": "actual_cpython_ast_literal_eval_calls", "python": sys.version,
               "unicode": unicodedata.unidata_version, "int_max_str_digits": sys.get_int_max_str_digits(),
               "recursion_limit": sys.getrecursionlimit(),
               "recursion_cutoff_note": "Actual CPython cutoff depends on current caller frames and recursionlimit; "
                                        "the deterministic native 3000-expression-node ceiling is not identical "
                                        "and does not emulate arbitrary caller stacks or recursionlimit changes.",
               "ast_source": str(ast_path), "ast_source_sha256": hashlib.sha256(ast_path.read_bytes()).hexdigest(),
               "unicode_extension_sha256": hashlib.sha256(Path(unicodedata.__file__).read_bytes()).hexdigest(),
               "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "set_order": "canonically sorted typed observations; not iteration-order evidence",
               "generated_unicode": unicode_receipt, "cases": rows,
               "dspy_boundary": dspy_boundary_receipt() if args.dspy_boundary else None}
    data = json.dumps(receipt, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
        accepted = sum(row["result"]["ok"] for row in rows)
        print(f"Observed {len(rows)} actual ast.literal_eval calls ({accepted} accepted); "
              f"sha256={hashlib.sha256(data.encode()).hexdigest()}")
        if unicode_receipt:
            print(json.dumps(unicode_receipt, sort_keys=True))
        if receipt["dspy_boundary"]:
            boundary_rows = receipt["dspy_boundary"]["cases"]
            fallbacks = sum(row.get("literal_fallback_eligible", False) for row in boundary_rows)
            print(f"Observed {len(boundary_rows)} actual DSPy parse_value calls; {fallbacks} eligible literal fallbacks")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
