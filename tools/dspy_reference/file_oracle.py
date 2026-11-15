"""Observe actual installed DSPy 3.5 File and lm15 behavior; never run native code.

All expected values in the receipt are produced by the installed Python libraries.
MIME data is copied after the same lazy initialization used by File.from_path.
"""

import argparse
import base64
import builtins
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from unittest import mock

import dspy
import pydantic
from pydantic import BaseModel, TypeAdapter

import reference as ref
from dspy.adapters.types.base_type import (
    CUSTOM_TYPE_END_IDENTIFIER,
    CUSTOM_TYPE_START_IDENTIFIER,
    Type,
    parts_from_text,
    split_data_uri,
)
from dspy._vendor.lm15 import serde
from dspy._vendor.lm15.types import Message, Request, document


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def source_receipt(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def json_value(value):
    """Retain Python-only inputs explicitly instead of inventing JSON equivalents."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"python_float": repr(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"python_type": type(value).__name__, "base64": base64.b64encode(raw).decode("ascii"),
                "hex": raw.hex(), "bytes": len(raw)}
    if isinstance(value, os.PathLike):
        return {"python_type": type(value).__name__, "path": os.fsdecode(os.fspath(value))}
    if isinstance(value, BaseModel):
        return {"python_type": type(value).__name__, "round_trip": json_value(value.model_dump(round_trip=True))}
    if dataclasses.is_dataclass(value):
        return {"python_type": type(value).__name__,
                "fields": {field.name: json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, type):
        result = {"python_class": f"{value.__module__}.{value.__qualname__}"}
        if issubclass(value, BaseModel):
            result["json_schema"] = value.model_json_schema()
        return result
    if isinstance(value, dspy.Example):
        # toDict() invokes File's prompt serializer and would erase the typed result.
        return {"python_type": type(value).__name__,
                "fields": {key: json_value(item) for key, item in value.items(include_dspy=True)}}
    if isinstance(value, BaseException):
        return {"python_type": type(value).__name__, "message": str(value)}
    raise TypeError(f"Oracle needs an explicit encoding for {type(value)!r}")


def observe(fn):
    try:
        return {"ok": True, "value": json_value(fn())}
    except Exception as exc:
        return error_observation(exc)


def mime_receipt():
    inited_before = mimetypes.inited
    # Do not reset or substitute a custom database: this is File.from_path's actual global DB.
    mimetypes.guess_type("source-oracle.pdf")
    names = ("types_map", "common_types", "suffix_map", "encodings_map")
    maps = {name: dict(sorted(getattr(mimetypes, name).items())) for name in names}
    known = []
    for filename in mimetypes.knownfiles:
        path = Path(filename)
        known.append({"path": filename, "exists": path.is_file(),
                      **({"source": source_receipt(path)} if path.is_file() else {})})
    urls = ["report.pdf", "REPORT.PDF", "notes.txt", "notes.TXT", "sheet.csv", "paper.docx",
            "archive.tar.gz", "archive.TAR.GZ", "archive.tar.Gz", "archive.TGZ", "archive.tgz",
            "archive.TBZ2", "drawing.SVGZ", "file.Z", "file.z", "file.gz", "file.br", "file.xz",
            "file.unknown-dszig", ".pdf", "without_extension", "trailing.", "file.pdf?x=1",
            "file.pdf#section", "https://example.invalid/file.pdf?x=1#section", "a:file.pdf",
            "aa:file.pdf", "data:text/plain;base64,QQ==", "data:;base64,QQ==", "data:foo,QQ==",
            "data:foo=bar,QQ==", "data:application/pdf,", "data:application/pdf", "DATA:text/plain,hi",
            "subdir/café.雪", "subdir/file.PDF ", "file.svg.gz", "file.svg.GZ", "file.pic", "file.webp"]
    guesses = [{"input": url, "strict": strict,
                "result": observe(lambda url=url, strict=strict: mimetypes.guess_type(url, strict=strict))}
               for url in urls for strict in (True, False)]
    # Include each actual extension and every suffix/encoding transformation, not merely examples.
    extension_guesses = {name: [{"extension": extension,
                                "result": list(mimetypes.guess_type("source" + extension)),
                                "uppercase_result": list(mimetypes.guess_type("source" + extension.upper()))}
                               for extension in mapping]
                         for name, mapping in maps.items()}
    return {"format_version": 1, "dspy": dspy.__version__, "python": sys.version,
            "python_executable": sys.executable, "platform": sys.platform,
            "inited_before_capture": inited_before, "inited_after_capture": mimetypes.inited,
            "initialization": "mimetypes.guess_type('source-oracle.pdf') on the imported global database",
            "sources": {"mimetypes": source_receipt(mimetypes.__file__),
                        "file": source_receipt(inspect.getsourcefile(dspy.File))},
            "knownfiles": list(mimetypes.knownfiles), "knownfile_receipts": known,
            **maps, "counts": {name: len(mapping) for name, mapping in maps.items()},
            "map_sha256": {name: digest(encoded_json(mapping)) for name, mapping in maps.items()},
            "guess_type_cases": guesses, "extension_guesses": extension_guesses,
            "ordered_entries": {name: list(getattr(mimetypes, name).items()) for name in names},
            "inverse_types_map": {
                "strict": mimetypes._db.types_map_inv[True], "common": mimetypes._db.types_map_inv[False]},
            "guess_extension_cases": [
                {"media_type": media_type, "strict": strict,
                 "extension": mimetypes.guess_extension(media_type, strict=strict),
                 "all_extensions": mimetypes.guess_all_extensions(media_type, strict=strict)}
                for media_type in sorted(set(maps["types_map"].values()) | set(maps["common_types"].values()))
                for strict in (True, False)]}


def error_observation(exc):
    result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, pydantic.ValidationError):
        result["validation_errors"] = json_value(exc.errors(include_url=False))
    for name in ("provider", "feature", "code"):
        value = getattr(exc, name, None)
        if value is not None:
            result[name] = json_value(value)
    return result


def file_observation(case_id, entrypoint, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    row = {"id": case_id, "entrypoint": entrypoint, "args": json_value(args), "kwargs": json_value(kwargs)}
    functions = {"File": dspy.File, "model_validate": dspy.File.model_validate,
                 "model_validate_json": dspy.File.model_validate_json,
                 "type_adapter": TypeAdapter(dspy.File).validate_python,
                 "from_bytes": dspy.File.from_bytes, "from_file_id": dspy.File.from_file_id,
                 "from_path": dspy.File.from_path}
    try:
        model = functions[entrypoint](*args, **kwargs)
    except Exception as exc:
        row["construction"] = error_observation(exc)
        row["first_failure"] = "construction"
        return row
    row["construction"] = {"ok": True, "value": json_value(model)}
    operations = {
        "model_dump_round_trip": lambda: model.model_dump(round_trip=True),
        "model_dump_json_round_trip": lambda: model.model_dump_json(round_trip=True),
        "format": lambda: [serde.part_to_dict(part) for part in model.format()],
        "format_python_parts": model.format,
        "format_repr": lambda: repr(model.format()),
        "serialize_model": model.serialize_model,
        "str": lambda: str(model),
        "repr": lambda: repr(model),
        "model_dump": model.model_dump,
        "model_dump_json": model.model_dump_json,
        "type_adapter_dump_json_round_trip": lambda: TypeAdapter(dspy.File).dump_json(model, round_trip=True),
        "type_adapter_dump_json": lambda: TypeAdapter(dspy.File).dump_json(model),
        "part_bytes": lambda: [part.bytes for part in model.format()],
    }
    row["operations"] = {name: observe(fn) for name, fn in operations.items()}
    row["failed_operations"] = [name for name, result in row["operations"].items() if not result["ok"]]
    row["first_failure"] = next(iter(row["failed_operations"]), None)
    return row


def constructor_cases():
    cases = []

    def add(case_id, entrypoint="File", args=(), kwargs=None):
        cases.append(file_observation(case_id, entrypoint, args, kwargs))

    valid_uri = "data:text/plain;base64,SGVsbG8="
    strings = {
        "valid_text": valid_uri,
        "valid_pdf": "data:application/pdf;base64,JVBERi0=",
        "valid_empty_media": "data:;base64,QQ==",
        "empty_payload": "data:application/pdf;base64,",
        "empty_data_head": "data:",
        "missing_comma": "data:text/plain;base64",
        "missing_base64": "data:text/plain,Hello",
        "uppercase_scheme": "DATA:text/plain;base64,QQ==",
        "uppercase_base64": "data:text/plain;BASE64,QQ==",
        "base64_parameter_after": "data:text/plain;base64;charset=utf-8,QQ==",
        "charset_before_base64": "data:text/plain;charset=utf-8;base64,QQ==",
        "bare_base64_media": "data:base64,QQ==",
        "extra_semicolon": "data:text/plain;;base64,QQ==",
        "media_spaces": "data: text/plain ;base64,QQ==",
        "media_unicode": "data:application/雪;base64,QQ==",
        "media_extra_colon_repr": "data:application/a:b;base64,QQ==",
        "payload_nonbmp_repr": "data:text/plain;base64,雪😀",
        "payload_unpadded": "data:text/plain;base64,QQ",
        "payload_three_padding": "data:text/plain;base64,QQ===",
        "payload_padding_only": "data:text/plain;base64,====",
        "payload_invalid_alphabet": "data:text/plain;base64,!!!!",
        "payload_urlsafe": "data:application/octet-stream;base64,__8=",
        "payload_noncanonical_pad_bits": "data:text/plain;base64,QR==",
        "payload_whitespace_only": "data:text/plain;base64, \n\t",
        "payload_internal_whitespace": "data:text/plain;base64,Q Q = =",
        "payload_ascii_newlines": "data:text/plain;base64,Q\nQ==\r\n",
        "payload_unicode_whitespace": "data:text/plain;base64,Q\u00a0Q==",
        "payload_whitespace_enables_unicode_split": "data:text/plain;base64,Q\u00a0 Q==",
        "payload_nested_uri": "data:text/plain;base64,data:application/pdf;base64,QQ==",
        "payload_multiple_comma": "data:text/plain;base64,QQ==,tail",
        "leading_ascii_whitespace": " \t" + valid_uri,
        "leading_unicode_whitespace": "\u2003" + valid_uri,
        "trailing_ascii_whitespace": valid_uri + "\r\n ",
        "trailing_unicode_whitespace": valid_uri + "\u3000",
        "empty_string": "",
        "base64_not_uri": "QQ==",
        "path_string": "/tmp/dszig-file-input.pdf",
        "url_string": "https://example.invalid/document.pdf",
        "file_url": "file:///tmp/dszig-file-input.pdf",
        "json_object_string": '{"file_id":"file-json"}',
        "marker_string": str(dspy.File.from_file_id("file-marker")),
    }
    for name, value in strings.items():
        for entrypoint in ("File", "model_validate", "type_adapter"):
            add(f"string/{name}/{entrypoint}", entrypoint, (value,))
        add(f"field/{name}", kwargs={"file_data": value})
        cases[-1]["split_data_uri_input"] = observe(lambda value=value: split_data_uri(value))
    byte_values = [b"", b"Hello", b"\x00\xff\x80\x01", bytes(range(256))]
    for index, value in enumerate(byte_values):
        for entrypoint in ("File", "model_validate", "from_bytes"):
            add(f"bytes/{index}/{entrypoint}", entrypoint, (value,))
    for value in (bytearray(b"Hello"), memoryview(b"Hello"), None, 0, 1.5, True, [], (valid_uri,)):
        add(f"unsupported/{type(value).__name__}/File", args=(value,))
        add(f"unsupported/{type(value).__name__}/validate", "model_validate", (value,))
    for index, value in enumerate((bytearray(b"Hello"), memoryview(b"\x00\xff"), None, "Hello", [0, 1], 12)):
        add(f"from_bytes/dtype/{index}", "from_bytes", (value,))
    records = {
        "missing_all": {},
        "unrecognized_only": {"url": "https://example.invalid/doc.pdf"},
        "filename_only": {"filename": "only.txt"},
        "filename_null": {"filename": None},
        "data_null": {"file_data": None},
        "id_null": {"file_id": None},
        "all_null": {"file_data": None, "file_id": None, "filename": None},
        "id": {"file_id": "file-123"},
        "data": {"file_data": valid_uri},
        "data_empty": {"file_data": ""},
        "id_empty": {"file_id": ""},
        "id_whitespace": {"file_id": " \t\u2003"},
        "both": {"file_data": valid_uri, "file_id": "file-priority", "filename": "display.txt"},
        "both_bad_data": {"file_data": "not a data URI", "file_id": "file-priority"},
        "both_null_data": {"file_data": None, "file_id": "file-priority"},
        "both_empty_id": {"file_data": valid_uri, "file_id": ""},
        "both_whitespace_id": {"file_data": valid_uri, "file_id": " \t\u00a0"},
        "extra": {"file_id": "file-123", "extra": 1},
        "bad_data_field": {"file_data": 12, "file_id": "file-priority"},
        "bad_id_field": {"file_id": 12},
        "bad_filename_field": {"file_id": "file-123", "filename": 12},
        "bool_filename": {"file_id": "file-123", "filename": False},
        "list_data": {"file_data": []},
        "dict_id": {"file_id": {}},
        "utf8_field_bytes": {"file_data": valid_uri.encode(), "file_id": b" file-bytes ", "filename": b" bytes.txt "},
        "invalid_utf8_field_bytes": {"file_data": b"\xff"},
        "data_uri_keyword_non_uri": {"file_data": "existing-file.txt"},
    }
    for name, value in records.items():
        for entrypoint in ("File", "model_validate", "type_adapter"):
            add(f"object/{name}/{entrypoint}", entrypoint, (value,))
        add(f"keywords/{name}", kwargs=value)
        if not any(isinstance(item, bytes) for item in value.values()):
            add(f"json/object/{name}", "model_validate_json", (json.dumps(value, ensure_ascii=False),))
    for field in ("file_data", "file_id", "filename"):
        for index, bad in enumerate((False, 1, 1.5, [], {}, bytearray(b" trimmed "), b"\xff")):
            value = {"file_id": "file-valid", field: bad}
            add(f"field_types/{field}/{index}", "model_validate", (value,))
    whitespace = [*range(0x09, 0x0E), *range(0x1C, 0x21), 0x85, 0xA0, 0x1680, 0x180E,
                  *range(0x2000, 0x200C), 0x2028, 0x2029, 0x202F, 0x205F, 0x2060, 0x3000, 0xFEFF]
    for point in whitespace:
        char = chr(point)
        add(f"unicode_trim/U+{point:04X}", kwargs={"file_id": char + "file-trim" + char,
                                               "filename": char + "name.txt" + char,
                                               "file_data": char + valid_uri + char})
    for index, filename in enumerate(("", " simple.txt ", "O'Reilly.pdf", 'say "hello".txt',
                                    "'both\".txt", "line\nbreak\tname.txt", "雪😀.pdf",
                                    "e\u0301é\x00\x1f\x7f\u2028.txt")):
        add(f"filename_repr/{index}", "from_file_id", ("file-'\"雪\n",), {"filename": filename})
    for index, mime in enumerate(("text/plain", "", None, " application/pdf ", "text/plain;charset=utf-8",
                                 "application/雪", "application/vnd.api+json", 42)):
        add(f"from_bytes/mime/{index}", "from_bytes", (b"\x00\xff",), {"filename": "source.bin", "mime_type": mime})
    for index, value in enumerate(("", " \t\u2003", " file-123 ", None, 1, False, b" file-bytes ")):
        add(f"from_file_id/{index}", "from_file_id", (value,), {"filename": " source.txt "})
    original = dspy.File(file_data=valid_uri, file_id="file-copy", filename="copy.txt")
    nullable = dspy.File(file_id=None)
    for name, value in (("ordinary", original), ("nullable", nullable)):
        for entrypoint in ("File", "model_validate", "type_adapter"):
            add(f"copy/{name}/{entrypoint}", entrypoint, (value,))
        add(f"copy/{name}/identity", "model_validate", (value,))
        cases[-1]["same_instance"] = dspy.File.model_validate(value) is value
    overlap_cases = [
        ("uri_data_overlap", (valid_uri,), {"file_data": valid_uri}),
        ("bytes_data_overlap", (b"Hello",), {"file_data": valid_uri}),
        ("object_filename_overlap", ({"file_id": "file-123", "filename": None},), {"filename": "other.txt"}),
        ("object_multiple_overlap", ({"file_id": "file-123", "file_data": valid_uri},),
         {"file_data": valid_uri, "file_id": "file-other"}),
        ("copy_optional_keyword", (dspy.File(file_id="file-copy"),), {"filename": "additional.txt"}),
        ("id_keyword_from_uri", (valid_uri,), {"file_id": "file-keyword"}),
        ("two_positionals", (valid_uri, "other"), {}),
        ("zero_positionals", (), {}),
    ]
    for name, args, kwargs in overlap_cases:
        add(f"constructor_args/{name}", args=args, kwargs=kwargs)
    for index, value in enumerate((valid_uri, "ordinary.txt", None, [], {"file_id": "file-json"})):
        add(f"json/top_level/{index}", "model_validate_json", (json.dumps(value),))
    return cases


def prepare_fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    names = ["plain.txt", "report.pdf", "REPORT.PDF", "archive.tar.gz", "archive.TAR.GZ", "archive.TGZ",
             "drawing.svgz", "drawing.SVGZ", "data.json", "table.csv", "sheet.xlsx", "paper.docx",
             "picture.webp", "movie.mp4", "record.wav", "source.py", "file.pic", "file.unknown-dszig",
             "no_extension", ".pdf", "trailing.", "query.pdf?x=1", "hash.pdf#section", "space.PDF ",
             "雪😀.txt", "empty.bin", "non_utf8.bin"]
    chunk_lengths = {f"chunk-{length}.bin": length for length in (3071, 3072, 3073, 6144, 6145)}
    names.extend(chunk_lengths)
    manifest = []
    for name in names:
        data = (bytes(index % 256 for index in range(chunk_lengths[name])) if name in chunk_lengths
                else b"" if name == "empty.bin" else bytes(range(256)) if name == "non_utf8.bin"
                else b"source-file\x00\xff\n")
        path = root / name
        if path.exists():
            if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
                raise RuntimeError(f"Refusing to overwrite an unrelated fixture: {path}")
        else:
            path.write_bytes(data)
        manifest.append({"path": str(path), "relative_path": name, "content": json_value(data), "sha256": digest(data),
                         **({"payload_rule": "bytes(i % 256 for i in range(length))"} if name in chunk_lengths else {})})
    directory = root / "a_directory.pdf"
    directory.mkdir(exist_ok=True)
    for name, target in (("link.txt", "report.pdf"), ("dangling.pdf", "not-created.pdf"), ("directory-link.pdf", "a_directory.pdf")):
        path = root / name
        if path.is_symlink():
            if os.readlink(path) != target:
                raise RuntimeError(f"Refusing to replace an unrelated symlink: {path}")
        elif path.exists():
            raise RuntimeError(f"Refusing to replace an unrelated fixture: {path}")
        else:
            path.symlink_to(target)
        manifest.append({"path": str(path), "relative_path": name, "symlink_target": target})
    return manifest


def filesystem_cases(root, manifest):
    cases = [file_observation(f"path/{row['relative_path']}", "from_path", (row["path"],)) for row in manifest]
    path = str(root / "report.pdf")
    for name, kwargs in (
        ("explicit_filename", {"filename": "custom.txt"}),
        ("empty_filename", {"filename": ""}),
        ("whitespace_filename", {"filename": " \u2003"}),
        ("mime_override", {"mime_type": "text/plain"}),
        ("empty_mime_override", {"mime_type": ""}),
        ("filename_no_mime_influence", {"filename": "misleading.json"}),
        ("both_overrides", {"filename": "custom.docx", "mime_type": "application/x-source"}),
        ("nonstring_filename", {"filename": 42}),
        ("nonstring_mime", {"mime_type": 42}),
    ):
        cases.append(file_observation(f"path_options/{name}", "from_path", (path,), kwargs))
    for name, value in (("missing", str(root / "not-created.pdf")), ("directory", str(root / "a_directory.pdf")),
                        ("pathlib", Path(path)), ("bytes_path", os.fsencode(path)), ("none", None),
                        ("empty", ""), ("nul", path + "\x00")):
        cases.append(file_observation(f"path_edge/{name}", "from_path", (value,)))
    io_cases = []
    real_open, real_io_open, real_stat = builtins.open, io.open, os.stat
    factories = [
        ("positional_existing", lambda: dspy.File(path)),
        ("validate_existing", lambda: dspy.File.model_validate(path)),
        ("adapter_validate_existing", lambda: TypeAdapter(dspy.File).validate_python(path)),
        ("keyword_existing", lambda: dspy.File(file_data=path)),
        ("positional_pathlib_existing", lambda: dspy.File(Path(path))),
        ("from_path_existing", lambda: dspy.File.from_path(path)),
    ]
    for name, fn in factories:
        with mock.patch.object(builtins, "open", wraps=real_open) as opens, \
             mock.patch.object(io, "open", wraps=real_io_open) as io_opens, \
             mock.patch.object(os, "stat", wraps=real_stat) as stats:
            result = observe(fn)
            calls = {"builtins_open": opens.call_count, "io_open": io_opens.call_count, "os_stat": stats.call_count}
        io_cases.append({"id": name, "input": path, "result": result, "filesystem_calls": calls})
    return {"root": str(root), "manifest": manifest, "cases": cases, "constructor_io": io_cases}


class FileBundle(BaseModel):
    """A labeled collection of original attachments."""

    title: str
    primary: dspy.File
    attachments: list[dspy.File]
    optional: dspy.File | None = None


def signature_shapes():
    first = dspy.File.from_bytes(b"Hello", filename="user filename.txt", mime_type="text/plain")
    second = dspy.File.from_file_id("file-uploaded", filename="uploaded.pdf")
    returned = dspy.File.from_file_id("file-returned", filename="return.txt")
    bundle = FileBundle(title="雪 collection", primary=first, attachments=[second, first], optional=None)
    output_bundle = FileBundle(title="returned", primary=returned, attachments=[returned], optional=None)
    records = [
        ("direct", dspy.File, first, returned),
        ("optional", dspy.File | None, first, returned),
        ("optional_null", dspy.File | None, None, None),
        ("list", list[dspy.File], [first, second], [returned]),
        ("dict", dict[str, dspy.File], {"inline": first, "uploaded": second}, {"returned": returned}),
        ("nested_list", list[list[dspy.File]], [[first], [second, first]], [[returned], []]),
        ("list_optional", list[dspy.File | None], [first, None, second], [None, returned]),
        ("model", FileBundle, bundle, output_bundle),
    ]
    return [
        {"id": name, "annotation": annotation, "input": value, "output": output,
         "signature": dspy.Signature(
             {"attachment": (annotation, dspy.InputField(desc="Original attachment; preserve its contents.")),
              "result": (annotation, dspy.OutputField(desc="The returned attachment."))},
             "Read the attachment and return the requested file value.")}
        for name, annotation, value, output in records
    ]


def raw_value(value):
    if isinstance(value, BaseModel):
        return value.model_dump(round_trip=True)
    if isinstance(value, dict):
        return {key: raw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [raw_value(item) for item in value]
    return value


def prompt_value(prompt):
    return {"system": json_value(prompt.system), "canonical_messages": serde.messages_to_json(prompt.messages),
            "messages": ref.messages_for(prompt)}


def parse_observation(adapter, signature, text):
    try:
        result = adapter.parse(signature, text)
    except Exception as exc:
        return error_observation(exc)
    operations = {"round_trip": observe(lambda: raw_value(result)),
                  "python_models": observe(lambda: result),
                  "serialized_output": observe(lambda: importlib.import_module("dspy.predict.predict").serialize_object(result))}
    return {"ok": True, "value": json_value(raw_value(result)), "operations": operations}


def adapter_cases(shapes):
    cases = []
    for shape in shapes:
        signature = shape["signature"]
        raw = raw_value(shape["output"])
        candidates = [("ordinary", raw), ("empty_object", {}), ("missing_result", None)]
        if shape["id"] in {"direct", "optional"}:
            candidates.extend([
                ("data_uri", "data:text/plain;base64,QQ=="), ("file_id_only", {"file_id": "file-parse"}),
                ("filename_only_late_failure", {"filename": "display-only.pdf"}),
                ("null_record_late_failure", {"file_id": None}), ("empty_payload_late_failure", {"file_data": "data:text/plain;base64,"}),
                ("invalid_data_late_failure", {"file_data": "ordinary.txt"}),
                ("id_priority", {"file_data": "not URI", "file_id": "file-priority"}),
                ("extra_property", {"file_id": "file-parse", "extra": True}),
                ("path_never_read", "/tmp/dszig-file-input.pdf"), ("null", None), ("empty_string", ""),
            ])
        elif shape["id"] == "list":
            candidates.extend([("empty_list", []), ("uri_list", ["data:text/plain;base64,QQ=="]),
                               ("nullable_item", [None]), ("bad_record", [{}])])
        for name, adapter in (("chat", dspy.ChatAdapter()), ("json", dspy.JSONAdapter()), ("xml", dspy.XMLAdapter())):
            row = {"id": f"{shape['id']}/{name}", "shape": shape["id"], "adapter": name,
                   "input": json_value(shape["input"]), "output": json_value(shape["output"]),
                   "annotation_schema": TypeAdapter(shape["annotation"]).json_schema(),
                   "xml_uses_nested": dspy.XMLAdapter._uses_nested_xml(shape["annotation"]),
                   "custom_types": json_value(Type.extract_custom_type_from_annotation(shape["annotation"]))}
            inputs, outputs = {"attachment": shape["input"]}, {"result": shape["output"]}
            row["format_user_message_content"] = observe(lambda: adapter.format_user_message_content(signature, inputs))
            row["format_assistant_message_content"] = observe(lambda: adapter.format_assistant_message_content(signature, outputs))
            row["format"] = observe(lambda: prompt_value(adapter.format(signature, demos=[], inputs=inputs)))
            row["format_with_demo"] = observe(
                lambda: prompt_value(adapter.format(signature, demos=[{**inputs, **outputs}], inputs=inputs)))
            parses = []
            for candidate_name, value in candidates:
                is_missing = candidate_name == "missing_result"
                if name == "chat":
                    text = ("[[ ## completed ## ]]" if is_missing else
                            f"[[ ## result ## ]]\n{json.dumps(value, ensure_ascii=False)}\n\n[[ ## completed ## ]]")
                elif name == "json":
                    text = json.dumps({} if is_missing else {"result": value}, ensure_ascii=False)
                else:
                    text = "" if is_missing else f"<result>{json.dumps(value, ensure_ascii=False)}</result>"
                parses.append({"id": candidate_name, "raw_output": text, "source_value": json_value(value),
                               "result": parse_observation(adapter, signature, text)})
            if name == "xml":
                for candidate_name, text in (
                    ("source_nested_xml", adapter._value_to_xml(raw, "result")),
                    ("self_closing", "<result/>"),
                    ("paired_empty", "<result></result>"),
                    ("nested_file_fields", "<result><file_id> file-nested </file_id><filename> return.txt </filename></result>"),
                    ("unquoted_data_uri", "<result>data:text/plain;base64,QQ==</result>"),
                    ("null_literal", "<result>null</result>"),
                ):
                    parses.append({"id": candidate_name, "raw_output": text,
                                   "result": parse_observation(adapter, signature, text)})
            row["parses"] = parses
            cases.append(row)
    return cases


class CapabilityFixture:
    """Capabilities supplied as fixture inputs; never an inference backend."""

    def __init__(self, enabled):
        self.supports_function_calling = enabled
        self.supports_response_schema = enabled
        self.supports_reasoning = enabled
        self.supported_params = {"response_format"} if enabled else set()

    def complete(self, request):
        raise RuntimeError("Capability-only oracle fixture must not execute inference")


def schema_and_flags(shapes):
    schemas = {"file_validation": dspy.File.model_json_schema(mode="validation"),
               "file_serialization": dspy.File.model_json_schema(mode="serialization"),
               "model_config": dict(dspy.File.model_config), "signatures": {}}
    preprocess = []
    for shape in shapes:
        signature = shape["signature"]
        schemas["signatures"][shape["id"]] = {
            "signature": signature.model_json_schema(),
            "output_annotation": TypeAdapter(shape["annotation"]).json_schema(),
            "structured_outputs": observe(lambda: importlib.import_module("dspy.adapters.json_adapter")
                                          ._get_structured_outputs_response_format(signature).model_json_schema()),
        }
        for enabled in (False, True):
            lm = dspy.BaseLM(model="source/capability-fixture", engine=CapabilityFixture(enabled), cache=False)
            for name, adapter in (("chat", dspy.ChatAdapter(native_response_types=[dspy.File])),
                                  ("json", dspy.JSONAdapter()),
                                  ("xml", dspy.XMLAdapter(native_response_types=[dspy.File]))):
                options = {}
                try:
                    adapted = adapter._call_preprocess(lm, options, signature, {"attachment": shape["input"]})
                    result = {"ok": True, "same_signature": adapted is signature,
                              "input_fields": list(adapted.input_fields), "output_fields": list(adapted.output_fields),
                              "lm_kwargs": json_value(options)}
                except Exception as exc:
                    result = error_observation(exc)
                preprocess.append({"shape": shape["id"], "adapter": name,
                                   "configured_fixture_capabilities": enabled, "result": result})
    exemplar = dspy.File.from_file_id("file-flags")
    frozen = observe(lambda: setattr(exemplar, "filename", "mutated.txt"))
    return {"schemas": schemas, "preprocess": preprocess,
            "type_flags": {"description": dspy.File.description(), "is_streamable": dspy.File.is_streamable(),
                           "parse_stream_chunk": observe(lambda: dspy.File.parse_stream_chunk(None)),
                           "parse_lm_response": observe(lambda: dspy.File.parse_lm_response(
                               ref.text_response("source/fixture", "not a native File response"))),
                           "frozen_assignment": frozen}}


def marker(payload):
    return CUSTOM_TYPE_START_IDENTIFIER + payload + CUSTOM_TYPE_END_IDENTIFIER


def content_part_cases():
    first = dspy.File.from_bytes(b"Hello", filename="not in canonical part.txt", mime_type="text/plain")
    second = dspy.File.from_file_id("file-parts", filename="not in canonical id.pdf")
    first_marker, second_marker = str(first), str(second)
    inputs = [
        ("empty_text", ""), ("plain_text", "before 雪\ntext"), ("one_file", first_marker),
        ("surrounded", "before\n" + first_marker + "\nafter"),
        ("two_adjacent", first_marker + second_marker),
        ("two_separated", first_marker + " text 雪 " + second_marker),
        ("json_list_embedding", json.dumps([first_marker, second_marker], ensure_ascii=False)),
        ("json_dict_embedding", json.dumps({"file": first_marker}, ensure_ascii=False)),
        ("escaped_marker", json.dumps(first_marker, ensure_ascii=False)[1:-1]),
        ("unmatched_start", "prefix" + CUSTOM_TYPE_START_IDENTIFIER + "not closed"),
        ("unmatched_end", CUSTOM_TYPE_END_IDENTIFIER + "suffix"),
        ("payload_empty", marker("")),
        ("payload_whitespace", marker(" \n\t ")),
        ("payload_invalid_json", marker("[invalid]")),
        ("payload_object_not_list", marker('{"type":"document","media_type":"application/pdf","file_id":"file-object"}')),
        ("payload_string_not_list", marker('"quoted"')),
        ("payload_empty_list", marker("[]")),
        ("payload_scalars", marker('[1, true, null, "literal", ["nested"]]')),
        ("payload_text_coalescing", "before" + marker('[{"type":"text","text":"middle"},"after"]') + "end"),
        ("payload_text_continuation", "before" + marker(json.dumps([
            {"type": "text", "text": "middle", "continuation": [{"provider": "source", "kind": "fixture", "data": {}}]},
            {"type": "text", "text": "after"}])) + "end"),
        ("payload_unknown_type", marker('[{"type":"not-real"}]')),
        ("payload_missing_type", marker('[{"file_id":"file-missing"}]')),
        ("payload_document_no_media", marker('[{"type":"document","file_id":"file-missing-media"}]')),
        ("payload_document_with_filename", marker(
            '[{"type":"document","media_type":"application/pdf","file_id":"file-extra","filename":"ignored.pdf"}]')),
        ("payload_document_empty_data", marker('[{"type":"document","media_type":"text/plain","data":""}]')),
        ("payload_document_url", marker(
            '[{"type":"document","media_type":"application/pdf","url":"https://example.invalid/doc.pdf"}]')),
    ]
    cases = []
    for case_id, text in inputs:
        def expand():
            parts = parts_from_text(text)
            return {"parts": [serde.part_to_dict(part) for part in parts],
                    "part_types": [type(part).__name__ for part in parts]}
        cases.append({"id": case_id, "input": text, "result": observe(expand)})
    return cases


class NoTransport:
    def __init__(self):
        self.calls = 0

    def stream(self, request):
        self.calls += 1
        raise RuntimeError("Provider-lowering oracle must not perform network I/O")


def transport_value(request):
    return {"method": request.method, "url": request.url, "headers": request.headers,
            "body_utf8": request.body.decode("utf-8"), "body_json": json.loads(request.body),
            "body_bytes": len(request.body), "body_sha256": digest(request.body),
            "connect_timeout": request.connect_timeout, "read_timeout": request.read_timeout,
            "write_timeout": request.write_timeout}


def provider_cases():
    from dspy._vendor.lm15.providers.openai import OpenAILM
    from dspy._vendor.lm15.providers.openai_chat import OpenAIChatLM
    from dspy._vendor.lm15.providers.anthropic import AnthropicLM
    from dspy._vendor.lm15.providers.gemini import GeminiLM
    from dspy.clients.lm15_boundary import history_messages, request_kwargs, snapshot_request
    from dspy.lm15 import TextPart

    transport = NoTransport()
    options = {"api_key": "source-oracle-not-a-secret", "base_url": "http://127.0.0.1:1/v1", "transport": transport}
    providers = [("openai_responses", OpenAILM(**options), "gpt-4o"),
                 ("openai_chat", OpenAIChatLM(**options), "gpt-4o"),
                 ("anthropic", AnthropicLM(**options), "claude-sonnet-4-5"),
                 ("gemini", GeminiLM(**options), "gemini-2.5-flash")]
    files = [
        ("inline_pdf", dspy.File.from_bytes(b"%PDF-", filename="original display.pdf", mime_type="application/pdf")),
        ("inline_text", dspy.File.from_bytes(b"Hello", filename="display.csv", mime_type="text/plain")),
        ("inline_octet", dspy.File.from_bytes(b"\x00\xff")),
        ("inline_plus_subtype", dspy.File.from_bytes(b"{}", mime_type="application/vnd.api+json")),
        ("inline_charset", dspy.File(file_data="data:text/plain;charset=utf-8;base64,QQ==")),
        ("inline_empty_media_default", dspy.File(file_data="data:;base64,QQ==")),
        ("inline_media_trailing_slash", dspy.File.from_bytes(b"A", mime_type="text/")),
        ("inline_media_without_slash", dspy.File.from_bytes(b"A", mime_type="odd")),
        ("inline_whitespace_payload", dspy.File(file_data="data:text/plain;base64,Q Q = =")),
        ("id", dspy.File.from_file_id("file-uploaded", filename="display.pdf")),
        ("id_priority", dspy.File(file_data="bad uri never consumed", file_id="file-priority", filename="display.txt")),
    ]
    cases = []
    for case_id, value in files:
        parts = tuple(value.format())
        request = Request(model="gpt-4o", messages=(Message.user((TextPart("before\n"), *parts, TextPart("\nafter"))),),
                          system="Keep the exact attachment order.")
        row = {"id": case_id, "file": json_value(value), "canonical_request": serde.request_to_dict(request),
               "history_messages": history_messages(request),
               "snapshot_is_same_request": snapshot_request(request) is request,
               "sdk_boundary": {kind: observe(lambda kind=kind: request_kwargs(request, kind))
                                for kind in ("chat", "responses", "text")}, "providers": []}
        for provider_name, provider, model in providers:
            for role in ("user", "assistant", "developer"):
                routed = dataclasses.replace(request, model=model, messages=(Message(role=role, parts=request.messages[0].parts),))
                for stream in (False, True):
                    before = transport.calls
                    result = observe(lambda: transport_value(provider.build_request(routed, stream=stream)))
                    row["providers"].append({"provider": provider_name, "role": role, "stream": stream,
                                             "result": result, "transport_calls": transport.calls - before})
        cases.append(row)
    text = str(files[0][1])
    ordinary = [
        ("plain_text", (Message.user("Plain text 雪\nwith a second line."),), None),
        ("empty_text", (Message.user(""),), None),
        ("text_turns", (Message.user("question"), Message.assistant("answer"), Message.user("next")), "system instructions"),
        ("literal_assistant_file_marker", (Message.user("question"), Message.assistant(text), Message.user("next")), None),
        ("literal_user_file_marker", (Message.user(text),), None),
        ("literal_system_file_marker", (Message.user("question"),), text),
        ("canonical_assistant_document", (Message.user("question"), Message.assistant(tuple(files[0][1].format()))), None),
    ]
    ordinary_cases = []
    for case_id, messages, system in ordinary:
        request = Request(model="gpt-4o", messages=messages, system=system)
        row = {"id": case_id, "canonical_request": serde.request_to_dict(request),
               "sdk_boundary": {kind: observe(lambda kind=kind: request_kwargs(request, kind))
                                for kind in ("chat", "responses", "text")}, "providers": []}
        for provider_name, provider, model in providers:
            routed = dataclasses.replace(request, model=model)
            for stream in (False, True):
                before = transport.calls
                row["providers"].append({
                    "provider": provider_name, "stream": stream,
                    "result": observe(lambda: transport_value(provider.build_request(routed, stream=stream))),
                    "transport_calls": transport.calls - before})
        ordinary_cases.append(row)
    return {"credential": "source-oracle-not-a-secret (literal fixture input, not a real credential)",
            "cases": cases, "text_and_assistant_cases": ordinary_cases, "total_transport_calls": transport.calls}


def persistence_cases(shapes, root):
    from dspy.predict.predict import serialize_object
    from dspy.predict.refine import recursive_mask
    import orjson

    cases = []
    for shape in shapes:
        signature = shape["signature"]
        demo = dspy.Example(attachment=shape["input"], result=shape["output"]).with_inputs("attachment")
        program = dspy.Predict(signature)
        program.demos = [demo]
        row = {"id": shape["id"], "original_demo": json_value(demo),
               "serialize_object_input": observe(lambda: serialize_object(shape["input"])),
               "recursive_mask_input": observe(lambda: recursive_mask(shape["input"])),
               "recursive_mask_demo": observe(lambda: recursive_mask(demo)),
               "recursive_mask_container": observe(lambda: recursive_mask(
                   {"attachment": shape["input"], "result": shape["output"], "ordinary": "retained"})),
               "orjson_input": observe(lambda: orjson.dumps(shape["input"])),
               "dump_state": observe(program.dump_state),
               "dump_state_non_json": observe(lambda: program.dump_state(json_mode=False))}
        try:
            state = program.dump_state()
            loaded = dspy.Predict(signature)
            loaded.load_state(state)
            row["load_state"] = {"ok": True, "demos": json_value(loaded.demos), "dump_state": json_value(loaded.dump_state()),
                                 "demo_value_types": {key: type(value).__name__ for key, value in loaded.demos[0].items()},
                                 "demo_input_is_same_object": loaded.demos[0]["attachment"] is shape["input"]}
            row["loaded_demo_prompts"] = {
                name: observe(lambda adapter=adapter: prompt_value(
                    adapter.format(signature, demos=loaded.demos, inputs={"attachment": shape["input"]})))
                for name, adapter in (("chat", dspy.ChatAdapter()), ("json", dspy.JSONAdapter()), ("xml", dspy.XMLAdapter()))
            }
        except Exception as exc:
            row["load_state"] = error_observation(exc)
        path = root / f"predict-{shape['id']}.json"
        try:
            program.save(path)
            data = path.read_bytes()
            restored = dspy.Predict(signature)
            restored.load(path)
            row["save_load_json"] = {"ok": True, "file": source_receipt(path), "saved_utf8": data.decode("utf-8"),
                                     "saved_state": json.loads(data), "loaded_demos": json_value(restored.demos),
                                     "loaded_dump_state": json_value(restored.dump_state())}
        except Exception as exc:
            row["save_load_json"] = error_observation(exc)
        cases.append(row)
    bad = dspy.File(filename="source accepts this but formatting fails")
    program = dspy.Predict(shapes[0]["signature"])
    program.demos = [{"attachment": bad, "result": bad}]
    cases.append({"id": "late_failure_demo", "input": json_value(bad), "dump_state": observe(program.dump_state),
                  "recursive_mask": observe(lambda: recursive_mask(bad))})
    return cases


def marker_boundary_cases(shape):
    from dspy.adapters.base import Prompt, _tool_result_content, user_message
    from dspy.clients.lm15_boundary import history_messages, request_kwargs
    from dspy.clients.requests import build_request
    from dspy.lm15 import TextPart

    value = shape["input"]
    text = str(value)
    lm = dspy.BaseLM(model="source/capability-fixture", engine=CapabilityFixture(False), cache=False)
    cases = []

    def canonical_content(content):
        return content if isinstance(content, str) else [serde.part_to_dict(part) for part in content]

    for case_id, fn in (
        ("Message.user_literal", lambda: serde.message_to_dict(Message.user(text))),
        ("Message.assistant_literal", lambda: serde.message_to_dict(Message.assistant(text))),
        ("adapter_user_message", lambda: serde.message_to_dict(user_message(text))),
        ("request_builder_string_literal", lambda: serde.request_to_dict(build_request(lm, text))),
        ("request_builder_system_literal", lambda: serde.request_to_dict(build_request(
            lm, Prompt(system=text, messages=(Message.user("ordinary"),))))),
        ("request_builder_prompt_assistant_literal", lambda: serde.request_to_dict(build_request(
            lm, Prompt(messages=(Message.assistant(text),))))),
        ("history_display_literal", lambda: history_messages(Request(
            model="source", system=text, messages=(Message.user(text), Message.assistant(text))))),
        ("tool_direct_File", lambda: canonical_content(_tool_result_content(value))),
        ("tool_marker_string", lambda: canonical_content(_tool_result_content(text))),
        ("tool_nested_File", lambda: canonical_content(_tool_result_content({"files": [value]}))),
        ("tool_plain_string", lambda: canonical_content(_tool_result_content("plain 雪 text"))),
        ("tool_scalar", lambda: canonical_content(_tool_result_content(42))),
    ):
        cases.append({"id": case_id, "marker_input": text, "result": observe(fn)})
    history_signature = dspy.Signature(
        {"attachment": (dspy.File, dspy.InputField()), "history": (dspy.History, dspy.InputField()),
         "result": (dspy.File, dspy.OutputField())},
        "Read attachments. Literal system marker: " + text)
    history = dspy.History(messages=[{"attachment": value, "result": shape["output"]}])
    for name, adapter in (("chat", dspy.ChatAdapter()), ("json", dspy.JSONAdapter()), ("xml", dspy.XMLAdapter())):
        def format_history():
            prompt = adapter.format(history_signature, demos=[], inputs={"attachment": value, "history": history})
            request = Request(model="gpt-4o", messages=prompt.messages, system=prompt.system)
            return {"prompt": prompt_value(prompt), "canonical_request": serde.request_to_dict(request),
                    "responses_payload": request_kwargs(request, "responses")}
        cases.append({"id": f"History.format/{name}", "result": observe(format_history)})

    calls = dspy.ToolCalls.model_validate({
        "tool_calls": [{"id": "source-tool-id", "name": "read_attachment", "args": {}}],
        "tool_call_results": {"tool_call_results": [
            {"call_id": "source-tool-id", "name": "read_attachment", "value": value, "is_error": False}]}})
    tool_signature = dspy.Signature(
        {"attachment": (str, dspy.InputField()), "calls": (dspy.ToolCalls, dspy.OutputField())},
        "Read a tool attachment.")
    for native in (False, True):
        adapter = dspy.ChatAdapter(use_native_function_calling=native)
        tool_history = dspy.History(messages=[{"attachment": "tool input", "calls": calls}])

        def format_tool_history():
            messages = tool_history.format(adapter, tool_signature)
            request = Request(model="gpt-4o", messages=messages)
            return {"canonical_messages": serde.messages_to_json(messages),
                    "responses_payload": request_kwargs(request, "responses"),
                    "chat_payload": observe(lambda: request_kwargs(request, "chat"))}
        cases.append({"id": f"History.tool_results/native={native}", "result": observe(format_tool_history)})
    return cases


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port=0):
        super().__init__(("127.0.0.1", port), FixtureHandler)
        self.receipts = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path != "/__oracle__/requests":
            self.send_error(404)
            return
        data = encoded_json(self.server.receipts)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.loads(data)
        self.server.receipts.append({"method": self.command, "path": self.path,
                                     "headers": list(self.headers.items()), "body_utf8": data.decode("utf-8"),
                                     "body_json": body, "body_sha256": digest(data), "body_bytes": len(data)})
        text = '<result>{"file_id":"file-http-fixture","filename":"fixture display.txt"}</result>'
        if self.path.endswith("/responses"):
            response = {"id": "source-response", "object": "response", "model": body["model"],
                        "output": [{"type": "message", "id": "source-message", "status": "completed", "role": "assistant",
                                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}}
        elif self.path.endswith("/chat/completions"):
            response = {"id": "source-chat", "object": "chat.completion", "model": body["model"],
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
        else:
            self.send_error(404)
            return
        encoded = encoded_json(response)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def lm_http_cases(shape):
    from dspy.clients.backend_selection import select_backend
    from dspy.clients.cache import Cache

    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_cache = dspy.cache
    real_connect = socket.socket.connect

    def local_only(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1") or address[1] != server.server_port:
            raise RuntimeError(f"Oracle refuses non-fixture network destination: {address!r}")
        return real_connect(sock, address)

    cases = []
    try:
        with mock.patch.object(socket.socket, "connect", local_only):
            for case_id, options in (
                ("openai_default_auto", {}),
                ("openai_responses_auto_engine", {"model_type": "responses"}),
                ("openai_auto_lm15_engine", {"engine": "lm15"}),
                ("openai_chat_auto_engine", {"model_type": "chat"}),
                ("openai_chat_lm15_engine", {"model_type": "chat", "engine": "lm15"}),
            ):
                dspy.cache = Cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=None)
                lm = dspy.LM("openai/gpt-4o", api_key="source-oracle-not-a-secret", api_base=server.url + "/v1",
                             num_retries=0, timeout=2.0, cache=True, **options)
                row = {"id": case_id, "constructor": {"model": "openai/gpt-4o", "api_base": server.url + "/v1",
                                                     "api_key": "source-oracle-not-a-secret", **options}, "calls": []}
                try:
                    text_signature = dspy.Signature(
                        {"attachment": (str, dspy.InputField()), "result": (str, dspy.OutputField())},
                        "Return the requested plain text.")
                    demo = {"attachment": shape["input"], "result": shape["output"]}
                    values = [("inline", shape["input"], shape["signature"], []),
                              ("same_inline_different_filename", dspy.File.from_bytes(
                                  b"Hello", filename="renamed display.pdf", mime_type="text/plain"), shape["signature"], []),
                              ("uploaded_id", dspy.File.from_file_id("file-uploaded", filename="display.txt"), shape["signature"], []),
                              ("inline_with_file_demo", shape["input"], shape["signature"], [demo]),
                              ("plain_text", "Plain text 雪\nwith no media.", text_signature, [])]
                    for value_name, value, signature, demos in values:
                        program = dspy.Predict(signature)
                        program.demos = demos
                        prompt = dspy.XMLAdapter().format(signature, demos=demos, inputs={"attachment": value})
                        request = Request(model=lm.model, messages=prompt.messages, system=prompt.system)
                        selection = select_backend(lm, request=request)
                        resolution = selection.resolution
                        selected = {"native": selection.native, "model_type": selection.model_type,
                                    "resolution": None if resolution is None else {
                                        key: getattr(resolution, key) for key in ("provider", "model", "adapter", "declared")}}
                        before = len(server.receipts)
                        with dspy.context(lm=lm, adapter=dspy.XMLAdapter()):
                            result = observe(lambda: program(attachment=value))
                        row["calls"].append({"id": value_name, "input": json_value(value), "demos": json_value(demos),
                                             "canonical_prompt": {"system": json_value(prompt.system),
                                                                  "messages": serde.messages_to_json(prompt.messages)},
                                             "selection": selected, "result": result, "http_requests": server.receipts[before:],
                                             "http_request_count": len(server.receipts) - before,
                                             "cache_entries_after": len(dspy.cache.memory_cache),
                                             "history_after": json_value(lm.history)})
                    row["engines"] = [
                        {"class": f"{type(engine).__module__}.{type(engine).__qualname__}",
                         "providers": {name: f"{type(provider).__module__}.{type(provider).__qualname__}"
                                       for name, provider in getattr(engine, "_providers", {}).items()}}
                        for engine in lm._engine_store.values()
                    ]
                except Exception as exc:
                    row["setup_or_capture_error"] = error_observation(exc)
                finally:
                    lm.close()
                cases.append(row)
    finally:
        dspy.cache = original_cache
        server.shutdown()
        server.server_close()
        thread.join()
    return {"evidence_kind": "Actual installed DSPy HTTP calls to a localhost response fixture; not model inference",
            "fixture_url": server.url, "cases": cases, "all_requests": server.receipts,
            "total_http_requests": len(server.receipts)}


def provenance():
    names = ["dspy.adapters.types.file", "dspy.adapters.types.base_type", "dspy._vendor.lm15.types",
             "dspy._vendor.lm15.serde", "dspy._vendor.lm15.providers.common",
             "dspy._vendor.lm15.providers.openai", "dspy._vendor.lm15.providers.openai_chat",
             "dspy._vendor.lm15.providers.anthropic", "dspy._vendor.lm15.providers.gemini",
             "dspy.clients.lm15_boundary", "dspy.clients.lm", "dspy.clients.base_lm",
             "dspy.clients.cache", "dspy.clients.execution", "dspy.clients.requests", "dspy.clients.model_metadata",
             "dspy.clients.backend_selection", "dspy.clients.engines.lm15_engine",
             "dspy.clients.engines.litellm_engine", "dspy.adapters.base", "dspy.adapters.chat_adapter",
             "dspy.adapters.json_adapter", "dspy.adapters.xml_adapter", "dspy.adapters.utils",
             "dspy.predict.predict", "dspy.predict.refine", "dspy.primitives.base_module", "dspy.primitives.example",
             "dspy.adapters.types.history", "dspy.adapters.types.tool",
             "pydantic", "pydantic_core._pydantic_core", "orjson.orjson", "mimetypes", "base64", "json"]
    modules = {name: source_receipt(importlib.import_module(name).__file__) for name in names}
    distributions = {}
    for name in ("dspy", "pydantic", "pydantic_core", "orjson", "httpx", "litellm"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = {"installed": False}
            continue
        files = [file for file in dist.files or () if str(file).endswith((".dist-info/METADATA", ".dist-info/RECORD"))]
        distributions[name] = {"version": dist.version, "metadata": [source_receipt(dist.locate_file(file)) for file in files]}
    return {"dspy": dspy.__version__, "python": sys.version, "python_executable": source_receipt(sys.executable),
            "platform": sys.platform, "modules": modules, "distributions": distributions,
            "oracle": source_receipt(__file__), "reference": source_receipt(ref.__file__),
            "lock": source_receipt(Path(__file__).with_name("uv.lock")),
            "offline_environment": {"LITELLM_LOCAL_MODEL_COST_MAP": os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")},
            "model_metadata": {
                "source_info": importlib.import_module("dspy.clients.model_metadata").source_info(),
                "bundled_snapshot": source_receipt(Path(importlib.import_module("dspy.clients.model_metadata").__file__)
                                                  .with_name("snapshot.json.gz")),
                "bundled_provenance": source_receipt(Path(importlib.import_module("dspy.clients.model_metadata").__file__)
                                                    .with_name("provenance.json"))}}


def receipt_counts(receipt):
    adapters = receipt["adapters"]
    constructors = receipt["constructors"]
    paths = receipt["filesystem"]["cases"]
    return {"constructor_cases": len(constructors), "constructor_successes": sum(row["construction"]["ok"] for row in constructors),
            "constructor_operations": sum(len(row.get("operations", {})) for row in constructors),
            "filesystem_cases": len(paths), "filesystem_operations": sum(len(row.get("operations", {})) for row in paths),
            "constructor_io_cases": len(receipt["filesystem"]["constructor_io"]),
            "adapter_cases": len(adapters), "adapter_parse_cases": sum(len(row["parses"]) for row in adapters),
            "schema_shapes": len(receipt["schema_and_flags"]["schemas"]["signatures"]),
            "preprocess_cases": len(receipt["schema_and_flags"]["preprocess"]),
            "content_part_cases": len(receipt["content_parts"]),
            "provider_file_cases": len(receipt["providers"]["cases"]),
            "provider_build_requests": sum(len(row["providers"]) for row in receipt["providers"]["cases"]),
            "text_assistant_provider_build_requests": sum(len(row["providers"])
                                                          for row in receipt["providers"]["text_and_assistant_cases"]),
            "marker_boundary_cases": len(receipt["marker_boundaries"]),
            "provider_transport_calls": receipt["providers"]["total_transport_calls"],
            "persistence_cases": len(receipt["persistence"]), "lm_routing_cases": len(receipt["lm_http"]["cases"]),
            "actual_localhost_http_requests": receipt["lm_http"]["total_http_requests"]}


def full_receipt(root, mime):
    manifest = prepare_fixtures(root)
    shapes = signature_shapes()
    receipt = {"format_version": 1, "evidence_kind": "Independent actual installed DSPy 3.5 source observations",
               "provenance": provenance(),
               "mime_source": {"counts": mime["counts"], "map_sha256": mime["map_sha256"],
                               "knownfiles": mime["knownfiles"], "sources": mime["sources"]},
               "constructors": constructor_cases(), "filesystem": filesystem_cases(root, manifest),
               "schema_and_flags": schema_and_flags(shapes), "adapters": adapter_cases(shapes),
               "content_parts": content_part_cases(), "providers": provider_cases(),
               "marker_boundaries": marker_boundary_cases(shapes[0]),
               "persistence": persistence_cases(shapes, root), "lm_http": lm_http_cases(shapes[0])}
    receipt["counts"] = receipt_counts(receipt)
    # Case IDs and captured-operation coverage, not handwritten expected DSPy behavior.
    for key in ("constructors", "adapters", "content_parts", "marker_boundaries", "persistence"):
        ids = [row["id"] for row in receipt[key]]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate source case IDs in {key}")
    return receipt


def write_receipt(path, value):
    data = encoded_json(value)
    path.write_bytes(data)
    return {"output": str(path), "bytes": len(data), "sha256": digest(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mime-output", type=Path)
    parser.add_argument("--mime-only", action="store_true", help="Capture only the initialized MIME source database")
    parser.add_argument("--fixture-dir", type=Path, help="Create and retain reusable source files in this directory")
    parser.add_argument("--serve", action="store_true", help="Serve the reusable localhost provider response fixture")
    parser.add_argument("--port", type=int, default=0, help="Fixture HTTP port; zero selects an available port")
    args = parser.parse_args()
    if dspy.__version__ != "3.5.0":
        raise RuntimeError(f"This oracle requires actual DSPy 3.5.0, installed {dspy.__version__}")
    if args.serve:
        if args.output or args.mime_output or args.mime_only or args.fixture_dir:
            parser.error("--serve cannot be combined with source receipt/fixture-file options")
        with FixtureServer(args.port) as server:
            print(json.dumps({"url": server.url, "requests_url": server.url + "/__oracle__/requests",
                              "evidence_kind": "Local deterministic response fixture, not model inference"}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    if args.port:
        parser.error("--port requires --serve")
    if args.output is not None and args.mime_output is not None and args.output.resolve() == args.mime_output.resolve():
        parser.error("--output and --mime-output must be different files")
    mime = mime_receipt()
    summaries = []
    if args.mime_output is not None:
        summaries.append(write_receipt(args.mime_output, mime))
    # Official offline metadata mode: no remote model-map refresh and no substitute capability data.
    with mock.patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "true"}):
        if args.mime_only:
            result = mime
        elif args.fixture_dir is not None:
            result = full_receipt(args.fixture_dir.resolve(), mime)
        else:
            with tempfile.TemporaryDirectory(prefix="dszig-file-oracle-") as root:
                result = full_receipt(Path(root), mime)
    if args.output is not None:
        summaries.append(write_receipt(args.output, result))
    if summaries:
        print(json.dumps({"receipts": summaries, "counts": result["counts"]}))
    else:
        sys.stdout.buffer.write(encoded_json(result))


if __name__ == "__main__":
    main()
