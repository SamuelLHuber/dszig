#!/usr/bin/env python3
"""Observe installed json-repair defaults and DSPy 3.5 raw-ID tool argument repair.

Run from the repository root with the reference virtualenv:
  tools/dspy_reference/.venv/bin/python tools/dspy_reference/tool_argument_repair_oracle.py [--output PATH]
  tools/dspy_reference/.venv/bin/python tools/dspy_reference/tool_argument_repair_oracle.py --zig-unicode-tables

Every result comes from the pinned Python libraries; nothing here is native or
hand-written expectation. The LM section sends actual dspy.LM requests to a
localhost fixture: protocol evidence, not authenticated inference.

Values use a lossless tagged encoding: exact decimal ints, float repr plus
binary64 bits, ordered object pairs, and code points for strings that contain
lone surrogates (native output encodes those as WTF-8).
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import struct
import sys
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import json_repair

sys.path.insert(0, "tools/dspy_reference")
import file_oracle as fo  # noqa: E402

if dspy.__version__ != "3.5.0":
    raise RuntimeError("This oracle requires DSPy 3.5.0")
if importlib.metadata.version("json-repair") != "0.64.2":
    raise RuntimeError("This oracle requires json-repair 0.64.2")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def provenance():
    package = Path(json_repair.__file__).resolve().parent
    dist = importlib.metadata.distribution("json-repair")
    sources = sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)
    from dspy.clients import lm15_boundary
    boundary = Path(lm15_boundary.__file__).resolve()
    license_files = sorted(str(f) for f in dist.files or [] if "LICENSE" in str(f))
    return {
        "python": sys.version,
        "unicodedata": unicodedata.unidata_version,
        "dspy": dspy.__version__,
        "json_repair": dist.version,
        "json_repair_record_sha256": sha256(Path(dist.locate_file(f"json_repair-{dist.version}.dist-info/RECORD")).read_bytes()),
        "json_repair_license": [{"path": name, "sha256": sha256(Path(dist.locate_file(name)).read_bytes())} for name in license_files],
        "json_repair_sources": [{"path": str(p.relative_to(package)), "sha256": sha256(p.read_bytes())} for p in sources],
        "dspy_lm15_boundary": {"path": "dspy/clients/lm15_boundary.py", "sha256": sha256(boundary.read_bytes())},
        "unicode_tables_sha256": sha256(zig_unicode_tables().encode()),
    }


def enc(value):
    if value is None:
        return {"null": True}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        return {"float": repr(value), "bits": struct.pack(">d", value).hex()}
    if isinstance(value, str):
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            return {"string_codepoints": [ord(c) for c in value]}
        return {"string": value}
    if isinstance(value, list):
        return {"array": [enc(item) for item in value]}
    if isinstance(value, dict):
        return {"object": [[enc(k), enc(v)] for k, v in value.items()]}
    raise TypeError(f"Unexpected json-repair result type {type(value)!r}")


def attempt(function):
    try:
        return {"ok": True, "value": enc(function())}
    except RecursionError as exc:
        return {"ok": False, "error": "RecursionError", "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__,
                "json_decode_error": isinstance(exc, json.JSONDecodeError), "message": str(exc)}


# Branch-oriented default-parser corpus. Labels name the source helper/branch
# intended; actual reachability and results are whatever the source does.
CORPUS = [
    # json.loads fast path and Python JSON numeric/string semantics.
    ("valid/object", '{"a": 1, "b": [true, false, null]}'),
    ("valid/duplicate_key_order", '{"a": 1, "b": 2, "a": 3}'),
    ("valid/whitespace", ' \t\r\n{"a":1}\n '),
    ("valid/scalar_string", '"text"'),
    ("valid/nan", "NaN"), ("valid/infinity", "Infinity"), ("valid/neg_infinity", "-Infinity"),
    ("valid/nan_in_object", '{"x": NaN, "y": -Infinity}'),
    ("valid/exp_overflow", "1e400"), ("valid/neg_exp_overflow", "-1e400"), ("valid/underflow", "1e-400"),
    ("valid/neg_zero_int", "-0"), ("valid/neg_zero_float", "-0.0"),
    ("valid/i64_max", "9223372036854775807"), ("valid/i64_overflow", "9223372036854775808"),
    ("valid/i64_min", "-9223372036854775808"), ("valid/i64_underflow", "-9223372036854775809"),
    ("valid/u64_over", "18446744073709551616"),
    ("valid/digits_4300", "1" * 4300), ("valid/digits_4301", "1" * 4301),
    ("valid/digits_4301_in_object", '{"a": ' + "1" * 4301 + "}"),
    ("valid/float_precision", "0.1"), ("valid/float_2_53_plus", "9007199254740993.0"),
    ("valid/exponent_int_form", "1E2"),
    ("valid/escapes", r'"\" \\ \/ \b \f \n \r \t \u00e9 \u96ea"'),
    ("valid/surrogate_pair", r'"\ud83d\ude00"'),
    ("valid/lone_high", r'"\ud83d"'), ("valid/lone_low", r'"\ude00"'),
    ("valid/high_then_nonlow", r'"\ud83d\u0041"'),
    ("valid/raw_unicode", '{"é": "雪😀\u2028"}'),
    ("invalid/high_then_bad_hex", r'"\ud83d\uZZZZ"'),
    ("invalid/control_char", '"a\nb"'), ("invalid/bom", "\ufeff{}"),
    ("invalid/trailing_data", '{"a":1} x'), ("invalid/leading_zero", "01"),
    ("invalid/plus", "+1"), ("invalid/dot_end", "1."), ("invalid/exp_empty", "1e"),
    ("invalid/lowercase_nan", "nan"),
    # Top-level dispatch, prefixes, multiple values, comments.
    ("top/empty", ""), ("top/whitespace_only", "   "), ("top/prose_only", "hello world"),
    ("top/prose_prefix", 'Here you go: {"a": 1}'),
    ("top/prose_suffix", '{"a": 1} thanks'),
    ("top/valid_suffix_after_prefix", 'text {"a": [1, 2]} more'),
    ("top/multiple_objects", '{"a": 1} {"b": 2}'),
    ("top/repeated_same_shape", '{"a": 1} {"a": 2}'),
    ("top/repeated_comma_separated", '{"a": 1}, {"a": 2}'),
    ("top/empty_then_value", '{} {"a": 1}'),
    ("top/array_then_object", '[1] {"a": 1}'),
    ("top/line_comment", '// note\n{"a": 1}'), ("top/hash_comment", '# note\n{"a": 1}'),
    ("top/block_comment", '/* note */ {"a": 1}'), ("top/block_short_close", '/*/ {"a": 1}'),
    ("top/unclosed_block", '/* never'), ("top/stray_slash", '/ {"a": 1}'),
    ("top/comments_run", '# a\n// b\n/* c */ {"a": 1}'),
    ("top/missing_open_brace", '"a": 1, "b": 2}'), ("top/missing_open_brace_spaces", '  "a": 1}'),
    ("top/missing_open_prose_quote", '"note" then "a": 1}'),
    ("top/missing_open_big_int", '"a": ' + "1" * 4301 + "}"),
    ("top/code_fence", '```json\n{"a": 1}\n```'),
    ("top/code_fence_prose", 'Answer:\n```json\n{"a": [1, 2]}\n```\nDone'),
    ("top/llm_block_false", '{"a": "```json false```"}'),
    # Tuples and parenthesized values.
    ("tuple/explicit", "(1, 2)"), ("tuple/single_group", "(1)"), ("tuple/single_trailing", "(1,)"),
    ("tuple/empty", "()"), ("tuple/nested", '{"a": (1, (2, 3))}'),
    ("tuple/prose_paren", 'note (clarification): {"a": 1}'),
    ("tuple/top_literal", "(true, None)"), ("tuple/top_trailer", "(1, 2) x"),
    ("tuple/after_newline", 'text\n(1, 2)'), ("tuple/unclosed", "(1, 2"),
    ("tuple/quoted_paren", "('a)', 2)"),
    # Objects.
    ("object/single_quotes", "{'a': 'b'}"), ("object/unquoted_key", "{a: 1}"),
    ("object/unquoted_value", '{"a": hello world}'), ("object/trailing_comma", '{"a": 1,}'),
    ("object/missing_comma", '{"a": 1 "b": 2}'), ("object/missing_colon", '{"a" 1}'),
    ("object/colon_before_key", '{: "a": 1}'), ("object/missing_value", '{"a": , "b": 2}'),
    ("object/missing_value_end", '{"a":}'), ("object/incomplete_key", '{"a'),
    ("object/incomplete_value", '{"a": "b'), ("object/incomplete_after_colon", '{"a":'),
    ("object/unclosed", '{"a": 1'), ("object/unclosed_nested", '{"a": {"b": [1, 2'),
    ("object/extra_close", '{"a": 1}}'), ("object/extra_close_nested", '[{"a": 1}}]'),
    ("object/empty_key", '{"": 1}'), ("object/empty_key_unquoted", '{: 1}'),
    ("object/comment_in_value", '{"a": 1 // c\n, "b": 2}'), ("object/hash_comment_value", '{"a": # c\n 1}'),
    ("object/block_comment_key", '{/* c */ "a": 1}'),
    ("object/empty_body_to_array", '{"a", "b"}'), ("object/empty_body_comment", '{ /* x */ }'),
    ("object/escaped_keys", '{\\"a\\": \\"b\\"}'), ("object/escaped_single_keys", "{\\'a\\': 1}"),
    ("object/comma_after_close", '{"a": 1}, "b": 2'),
    ("object/array_continuation", '{"rows": [[1, 2]], [3, 4]}'),
    ("object/array_continuation_tail", '{"rows": [[1, 2], 3, 4] [5, 6]}'),
    ("object/array_continuation_flat", '{"a": [1, 2] [3]}'),
    ("object/duplicate_in_array", '[{"a": 1, "a": 2}]'),
    ("object/duplicate_no_comma", '[{"a": 1 "a": 2}]'),
    ("object/bracket_in_array_object", '[{"a": 1]'),
    ("object/literal_values", '{"a": True, "b": FALSE, "c": None, "d": nul}'),
    ("object/literal_prefix", '{"a": nullable}'),
    ("object/key_literal", '{true: 1, null: 2}'),
    ("object/smart_quotes", '{“a”: “b”}'), ("object/right_smart_open", '{”a”: 1}'),
    ("object/low_smart_quote", '{"a": "„quoted” text"}'),
    ("object/nested_quote", '{"a": "say "hi" now"}'), ("object/nested_quote_end", '{"a": "x "y""}'),
    ("object/quote_then_member", '{"a": "b" c", "d": 1}'),
    ("object/value_comma_member", '{"a": "one, two", "b": 3}'),
    ("object/value_comma_bare_member", '{"a": "x, b: 2}'),
    ("object/value_comma_prose", '{"a": "x, floof: explanation"}'),
    ("object/value_comma_container", '{"a": "x, [1, 2]"}'),
    ("object/value_brace_text", '{"a": "f(x) { return 1; }"}'),
    ("object/value_inline_object", '{"a": "see {b: 1} here"}'),
    ("object/value_missing_close", '{"a": "b, "c": "d"}'),
    ("object/value_close_brace", '{"a": "b}'), ("object/value_close_brace_more", '{"a": "b} x"}'),
    ("object/regex_class", '{"a": "[^"]+", "b": 1}'),
    ("object/fence_after_brace", '{"a": "x"}\n```'),
    ("object/fence_snippet", '{"a": "code: {x}\n```js\nf()\n``` end"}'),
    ("object/doubled_quotes", '{"a": ""b""}'), ("object/doubled_empty", '{"a": """}'),
    ("object/doubled_quote_key", '{""a"": 1}'), ("object/leading_quote_span", '{"a": ""b" c", "d": 1}'),
    ("object/key_colon_inside", '{"a:b": 1}'), ("object/key_colon_unclosed", '{"a: 1}'),
    ("object/key_missing_close_comma", '{"a: "b", "c": 1}'),
    ("object/key_comment_corner", '{a b: 1}'),
    ("object/nonalnum_prefix", '{"a": @@@value}'),
    # Arrays.
    ("array/unclosed", "[1, 2"), ("array/trailing_comma", "[1, 2,]"), ("array/missing_comma", "[1 2]"),
    ("array/extra_commas", "[,1,,2,]"), ("array/ellipsis", '[1, 2, ...]'), ("array/ellipsis_quoted", '["..."]'),
    ("array/unquoted_strings", "[a, b c]"), ("array/string_colon_object", '["a": 1]'),
    ("array/object_value_scalar_close", '{"a": [1, "b": 2}'),
    ("array/brace_close", "[1, 2}"), ("array/empty_items", '[[], {}, ""]'),
    ("array/mixed_quotes", "['a', \"b\"]"), ("array/quoted_even_section", '["a "b" c"]'),
    ("array/quoted_odd", '["a "b c"]'), ("array/close_without_quote", '["a, b]'),
    ("array/comment", "[1, // c\n 2]"), ("array/hash_comment", "[1, # c\n 2]"),
    ("array/nested_unclosed", "[[1, [2"), ("array/literals", "[true, False, NONE, none]"),
    # Numbers.
    ("number/currency_comma", '{"a": 1,000}'), ("number/array_comma", "[1,000]"),
    ("number/underscore", '{"a": 1_000}'), ("number/fraction_slash", '{"a": 1/2}'),
    ("number/trailing_minus", '{"a": 1-}'), ("number/trailing_e", '{"a": 1e}'),
    ("number/exp_plus", '{"a": 1e+5}'), ("number/leading_dot", '{"a": .5}'),
    ("number/trailing_dot", '{"a": 5.}'), ("number/neg_leading_dot", '{"a": -.5}'),
    ("number/minus_only", '{"a": -}'), ("number/double_dot", '{"a": 1.2.3}'),
    ("number/inner_minus", '{"a": 1-2}'), ("number/alpha_suffix", '{"a": 12abc}'),
    ("number/big", '{"a": 123456789012345678901234567890}'),
    ("number/leading_zeros", '{"a": 007}'), ("number/neg_zero", '{"a": -0}'),
    ("number/overflow", '{"a": 1e400}'), ("number/digits_4301_repair", '{"a": ' + "1" * 4301 + "x}"),
    ("number/digits_4301_bare", "[" + "1" * 4301 + ", ]"),
    ("number/unicode_digit", '{"a": ²}'), ("number/arabic_digit", '{"a": ٣}'),
    # Strings and escapes inside repair.
    ("string/escape_n", '{"a": "x\\n'), ("string/stray_escapes", "{'a': 'x\\'y'}"),
    ("string/escape_double_backslash", '{"a": "x\\\\\\\\y", }'),
    ("string/unicode_escape", "{'a': '\\u00e9'}"), ("string/hex_escape", "{'a': '\\x41'}"),
    ("string/bad_unicode_escape", "{'a': '\\uZZ'}"), ("string/escaped_smart", '{"a": "x\\“y"}'),
    ("string/lone_surrogate_escape", "{'a': '\\ud800'}"),
    ("string/raw_newline_end", '{"a": "x\n"}'), ("string/trailing_space", '{"a": "x   '),
    ("string/unicode_key", "{é: 雪}"), ("string/unicode_value", '{"a": 雪 😀}'),
    ("string/emoji_boundary", '{"😀": "😀", "b": "é'),
    ("string/backtick_key", '{"a": "x, `b`: 1}'),
    # Default-path nesting.
    ("nesting/deep_array_200", "[" * 200 + "]" * 200),
    ("nesting/deep_unclosed_200", "[" * 200),
    ("nesting/deep_object_100", '{"a":' * 100 + "1" + "}" * 100),
]


def zig_unicode_tables():
    """CPython's str.isalpha/str.isdigit as sorted inclusive Zig ranges."""
    def ranges(predicate):
        out, start = [], None
        for cp in range(0x110000):
            hit = predicate(chr(cp))
            if hit and start is None:
                start = cp
            elif not hit and start is not None:
                out.append((start, cp - 1))
                start = None
        if start is not None:
            out.append((start, 0x10FFFF))
        return out

    lines = [f"// Generated by tool_argument_repair_oracle.py --zig-unicode-tables from CPython {sys.version.split()[0]} (Unicode {unicodedata.unidata_version})."]
    for name, predicate in (("alpha_ranges", str.isalpha), ("digit_ranges", str.isdigit)):
        lines.append(f"const {name} = [_]Range{{")
        lines.extend(f"    .{{ .lo = 0x{lo:x}, .hi = 0x{hi:x} }}," for lo, hi in ranges(predicate))
        lines.append("};")
    return "\n".join(lines) + "\n"


def parser_rows():
    rows = []
    for case_id, text in CORPUS:
        rows.append({"id": case_id, "input": enc(text),
                     "json_loads": attempt(lambda: json.loads(text)),
                     "json_repair_loads": attempt(lambda: json_repair.loads(text))})
    return rows


# Actual DSPy LM tool normalization plus lm15_boundary.repair_tool_arguments.
MALFORMED = '{"query": "Paris", "limit": 2'
LM_CASES = [
    ("chat/real_id_malformed", "chat", [{"id": "call_1", "name": "probe", "arguments": MALFORMED}]),
    ("chat/missing_id_malformed", "chat", [{"name": "probe", "arguments": MALFORMED}]),
    ("chat/null_id_malformed", "chat", [{"id": None, "name": "probe", "arguments": MALFORMED}]),
    ("chat/real_id_valid", "chat", [{"id": "call_1", "name": "probe", "arguments": '{"query": "Paris"}'}]),
    ("chat/legit_partial_json", "chat", [{"id": "call_1", "name": "probe", "arguments": '{"partial_json": "{bad"}'}]),
    ("chat/legit_partial_json_self", "chat", [{"id": "call_1", "name": "probe", "arguments": json.dumps({"partial_json": MALFORMED})}]),
    ("chat/repair_to_array", "chat", [{"id": "call_1", "name": "probe", "arguments": "[1, 2"}]),
    ("chat/repair_to_string", "chat", [{"id": "call_1", "name": "probe", "arguments": "hello"}]),
    ("chat/repair_empty", "chat", [{"id": "call_1", "name": "probe", "arguments": ""}]),
    ("chat/valid_nonobject", "chat", [{"id": "call_1", "name": "probe", "arguments": "42"}]),
    ("chat/valid_nan", "chat", [{"id": "call_1", "name": "probe", "arguments": '{"x": NaN}'}]),
    ("chat/valid_big_int", "chat", [{"id": "call_1", "name": "probe", "arguments": '{"x": 123456789012345678901234567890}'}]),
    ("chat/int_4301_digits", "chat", [{"id": "call_1", "name": "probe", "arguments": '{"x": ' + "1" * 4301 + "}"}]),
    ("chat/lone_surrogate", "chat", [{"id": "call_1", "name": "probe", "arguments": '{"x": "\\ud800"'}]),
    ("chat/unicode_malformed", "chat", [{"id": "call_1", "name": "probe", "arguments": "{'é': '雪😀'"}]),
    ("chat/two_calls_same_raw", "chat", [{"id": "call_1", "name": "probe", "arguments": MALFORMED},
                                         {"id": "call_2", "name": "other", "arguments": MALFORMED}]),
    ("chat/nonstring_arguments", "chat", [{"id": "call_1", "name": "probe", "arguments": {"query": "x"}}]),
    ("responses/call_id_malformed", "responses", [{"call_id": "call_1", "name": "probe", "arguments": MALFORMED}]),
    ("responses/item_id_malformed", "responses", [{"id": "item_1", "name": "probe", "arguments": MALFORMED}]),
    ("responses/empty_call_id_item_id", "responses", [{"call_id": "", "id": "item_1", "name": "probe", "arguments": MALFORMED}]),
    ("responses/synthesized_id", "responses", [{"name": "probe", "arguments": MALFORMED}]),
    ("responses/legit_partial_json", "responses", [{"call_id": "call_1", "name": "probe", "arguments": '{"partial_json": "{bad"}'}]),
    ("responses/repair_to_array", "responses", [{"call_id": "call_1", "name": "probe", "arguments": "[1, 2"}]),
    ("responses/valid_nan", "responses", [{"call_id": "call_1", "name": "probe", "arguments": '{"x": NaN}'}]),
    ("responses/int_4301_digits", "responses", [{"call_id": "call_1", "name": "probe", "arguments": '{"x": ' + "1" * 4301 + "}"}]),
    ("responses/prose_wrapped", "responses", [{"call_id": "call_1", "name": "probe", "arguments": 'Sure: {"a": 1} ok'}]),
]


def lm_payload(model_type, calls):
    if model_type == "chat":
        tool_calls = []
        for call in calls:
            row = {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            if "id" in call:
                row["id"] = call["id"]
            tool_calls.append(row)
        return {"id": "repair-probe", "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}]}
    return {"id": "repair-probe", "model": "gpt-4o",
            "output": [{"type": "function_call", **call} for call in calls]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.dumps(self.server.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def lm_rows():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rows = []
    try:
        for case_id, model_type, calls in LM_CASES:
            server.payload = lm_payload(model_type, calls)
            model = dspy.LM("openai/gpt-4o", engine="lm15", model_type=model_type,
                            api_base=f"http://127.0.0.1:{server.server_port}/v1",
                            api_key="source-fixture-not-a-secret", cache=False, num_retries=0,
                            timeout=2, max_tokens=None, temperature=None)
            raw = [c["arguments"] for c in calls if isinstance(c.get("arguments"), str)]
            rows.append({"id": case_id, "model_type": model_type, "response": server.payload,
                         "raw_json_loads": [attempt(lambda r=r: json.loads(r)) for r in raw],
                         "raw_json_repair_loads": [attempt(lambda r=r: json_repair.loads(r)) for r in raw],
                         "result": fo.observe(lambda: model(prompt="probe"))})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--zig-unicode-tables", action="store_true")
    args = parser.parse_args()
    if args.zig_unicode_tables:
        sys.stdout.write(zig_unicode_tables())
        return
    document = {
        "evidence_kind": "actual installed json_repair.loads defaults and dspy.LM HTTP to localhost; not inference",
        "provenance": provenance(),
        "parser_cases": parser_rows(),
        "lm_cases": lm_rows(),
    }
    text = json.dumps(document, ensure_ascii=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
