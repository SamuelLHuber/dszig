"""Exercise the installed DSPy 3.5 Code implementation without network.

Run: tools/dspy_reference/.venv/bin/python tools/dspy_reference/code_oracle.py
All receipts come from DSPy/Pydantic calls; no native implementation is read.
"""

import hashlib
import importlib.metadata
import inspect
import json
import pathlib
import platform
import unicodedata

import dspy
import pydantic
from dspy.adapters.json_adapter import _get_structured_outputs_response_format
from dspy.adapters.types import code as code_module
from dspy.adapters.types import base_type as base_type_module
from dspy.adapters.utils import (
    _get_json_schema,
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    serialize_for_json,
)


import reference as ref


Java = dspy.Code["JAVA"]
UnicodeLanguage = dspy.Code["İΣ"]


class Bundle(pydantic.BaseModel):
    primary: Java
    optional: dspy.Code | None
    snippets: list[dspy.Code]
    files: dict[str, Java]


class Direct(dspy.Signature):
    source: dspy.Code = dspy.InputField()
    result: Java = dspy.OutputField()


class Nested(dspy.Signature):
    source: list[dict[str, Java]] = dspy.InputField()
    result: Java | None = dspy.OutputField()
    snippets: list[dspy.Code] = dspy.OutputField()
    bundle: Bundle = dspy.OutputField()


class OnlyOptional(dspy.Signature):
    result: dspy.Code | None = dspy.OutputField()


CASES = [
    ("unfenced", " \tprint(1)\r\n\u2003"),
    ("multiline", "prefix ```python\n  print(1)\n``` suffix"),
    ("inline", "prefix ``` print(1) ``` suffix"),
    ("two_inline", "```first``` gap ```second```"),
    ("inline_before_multiline", "```inline```\nprose\n```python\nsecond\n```"),
    ("inline_same_line_before_multiline", "```inline``` gap ```python\nsecond\n```"),
    ("empty_language", "```\n  body\n```"),
    ("empty_inline", "``````"),
    ("empty_multiline", "```python\n```"),
    ("unclosed", " \t```python\n body\n\u2003"),
    ("opening_at_end", "before ```"),
    ("carriage_return_only", "```python\r body\r```"),
    ("crlf", "before ```Py\r\n  first\r\n  second\r\n``` after"),
    ("unicode_whitespace", "```python\n\u0085\u00a0\u1680\u2000\u2003\u2028\u2029\u202f\u205f\u3000body\u3000\u001c\n```"),
    ("non_whitespace_edges", "```\n\u200bbody\ufeff\n```"),
    ("fence_with_newline_after_close", "```first```\nsecond```"),
    ("non_ascii_line_separator", "```language\u2028 body```"),
    ("four_backticks", "````python\nbody\n````"),
    ("json_looking", '{"code":"inside"}'),
    ("fenced_json_object", '```json\n{"code":"inside"}\n```'),
    ("fenced_json_list", '```python\n[1,2]\n```'),
    ("quoted_string", '"quoted"'),
    ("invalid_object_text", '{"code":7}'),
    ("json_scalar_looking", "123"),
    ("null_looking", "null"),
]


def normalized(value):
    return serialize_for_json(value)


def attempt(operation):
    try:
        result = operation()
        return {"ok": True, "value": normalized(result)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def source_receipt(module):
    path = pathlib.Path(inspect.getsourcefile(module))
    return {"file": str(path.relative_to(pathlib.Path(dspy.__file__).parent)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    valid = dspy.Code.model_validate("```python\nprint(1)\n```")
    specialized = Java.model_validate("run();")
    direct_values = ["plain", {"code": "```java\nrun();\n```", "ignored": 7},
                     {}, {"other": "x"}, {"code": 7}, {"code": None},
                     {"code": ["x"]}, None, True, 7, ["x"]]
    bundle_input = {"primary": "```java\nrun();\n```", "optional": None,
                    "snippets": ["```x```", {"code": " ```y``` "}],
                    "files": {"main": {"code": "```java\nmain();\n```", "extra": True}}}
    bundle = Bundle.model_validate(bundle_input)
    predictor = dspy.Predict(Direct)
    predictor.demos = [dspy.Example(source=valid, result=Java.model_validate("run();")).with_inputs("source")]
    demo_state = predictor.dump_state()
    restored = dspy.Predict(Direct)
    restored.load_state(demo_state)
    descriptions = []
    for annotation in (dspy.Code, Java, UnicodeLanguage, list[Java], Java | None,
                       dict[str, Java], list[dict[str, Java]], Bundle):
        signature = dspy.Signature({"result": (annotation, dspy.OutputField())})
        descriptions.append({"annotation": get_annotation_name(annotation),
                             "field_description": get_field_description_string(signature.output_fields),
                             "schema": pydantic.TypeAdapter(annotation).json_schema(),
                             "prompt_schema": _get_json_schema(annotation)})
    adapters = {}
    for adapter in (dspy.ChatAdapter(), dspy.JSONAdapter(), dspy.XMLAdapter()):
        label = type(adapter).__name__
        if label == "ChatAdapter":
            candidates = ["[[ ## result ## ]]\n```java\nrun();\n```\n[[ ## completed ## ]]",
                          '[[ ## result ## ]]\n{"code":"```java\\nrun();\\n```","extra":4}',
                          '[[ ## result ## ]]\n{"code":7}',
                          "[[ ## result ## ]]\n123", "[[ ## result ## ]]\nnull",
                          "[[ ## result ## ]]\n"]
            optional = ["[[ ## result ## ]]\nnull", "[[ ## result ## ]]\n", ""]
        elif label == "JSONAdapter":
            candidates = [json.dumps({"result": "```java\nrun();\n```"}),
                          json.dumps({"result": {"code": "```java\nrun();\n```", "extra": 4}}),
                          '{"result":{"code":7}}', '{"result":7}', '{"result":null}',
                          '{"result":""}', '{"result":{}}', '{}']
            optional = ['{"result":null}', '{"result":""}', '{}', '{"result":7}']
        else:
            candidates = ["<result>```java\nrun();\n```</result>",
                          "<result><code>```java\nrun();\n```</code></result>",
                          "<result><code>run();</code><extra>4</extra></result>",
                          "<result><wrong>x</wrong></result>", "<result />",
                          "<result></result>", "<result>123</result>",
                          "<result>null</result>", "<result>if (a < b) x = '<tag>';</result>"]
            optional = ["<result />", "<result></result>", "", "<result>null</result>",
                        "<result><code></code></result>"]
        nested_outputs = {"result": None, "snippets": bundle.snippets, "bundle": bundle}
        nested_text = adapter.format_assistant_message_content(Nested, nested_outputs)
        model = ref.KeywordLM([], candidates[0])
        with dspy.context(lm=model, adapter=adapter):
            predicted = dspy.Predict(Direct)(source=valid)
        adapters[label] = {
            "direct": [{"completion": text, **attempt(lambda text=text: adapter.parse(Direct, text))}
                       for text in candidates],
            "optional": [{"completion": text, **attempt(lambda text=text: adapter.parse(OnlyOptional, text))}
                         for text in optional],
            "input": adapter.format_user_message_content(Direct, {"source": valid}),
            "assistant": adapter.format_assistant_message_content(Direct, {"result": Java.model_validate("run();")}),
            "structure": adapter.format_field_structure(Direct),
            "requirements": adapter.user_message_output_requirements(Direct),
            "nested_assistant": nested_text,
            "nested_round_trip": attempt(lambda: adapter.parse(Nested, nested_text)),
            "predict": {"output": predicted.toDict(), "result_class": type(predicted.result).__name__,
                        "requests": model.calls},
        }
    output = {
        "versions": {"python": platform.python_version(), "dspy": importlib.metadata.version("dspy"),
                     "pydantic": pydantic.__version__, "unicode": unicodedata.unidata_version},
        "sources": [source_receipt(module) for module in (
            code_module, base_type_module, inspect.getmodule(parse_value),
            inspect.getmodule(dspy.JSONAdapter), inspect.getmodule(dspy.XMLAdapter))],
        "classes": {"default_language": dspy.Code.language, "specialized_name": Java.__name__,
                    "specialized_language": Java.language,
                    "repeated_specialization_is_same_class": dspy.Code["JAVA"] is Java,
                    "description": dspy.Code.description(), "java_description": Java.description(),
                    "unicode_description": UnicodeLanguage.description()},
        "identity": {"validate_input": dspy.Code.validate_input(valid) is valid,
                     "model_validate": dspy.Code.model_validate(valid) is valid,
                     "specialized_to_base": {
                         "validate_input": dspy.Code.validate_input(specialized) is specialized,
                         "model_validate": dspy.Code.model_validate(specialized) is specialized,
                         "class": type(dspy.Code.model_validate(specialized)).__name__,
                         "language": dspy.Code.model_validate(specialized).language,
                     },
                     "other_language": attempt(lambda: UnicodeLanguage.model_validate(specialized)),
                     "fresh_same_language_class": attempt(lambda: dspy.Code["JAVA"].model_validate(specialized))},
        "filter": [{"name": name, "input": text,
                    "code": dspy.Code.model_validate(text).code,
                    "serialized": dspy.Code.model_validate(text).model_dump(),
                    "parse_value": attempt(lambda text=text: parse_value(text, dspy.Code))}
                   for name, text in CASES],
        "validation": [{"input": value, **attempt(lambda value=value: dspy.Code.model_validate(value))}
                       for value in direct_values],
        "serialization": {"format": valid.format(), "serialize_model": valid.serialize_model(),
                          "model_dump": valid.model_dump(), "model_dump_json": valid.model_dump_json(),
                          "round_trip_dump": valid.model_dump(round_trip=True),
                          "field": format_field_value(Direct.input_fields["source"], valid),
                          "nested": normalized(bundle),
                          "nested_round_trip": bundle.model_dump(round_trip=True),
                          "prediction": dspy.Prediction(result=valid).toDict(),
                          "demo_state": demo_state,
                          "restored_demos": restored.demos},
        "nested_validation": [{"name": name, **attempt(operation)} for name, operation in [
            ("valid", lambda: Bundle.model_validate(bundle_input)),
            ("invalid_optional", lambda: Bundle.model_validate({**bundle_input, "optional": 7})),
            ("invalid_list", lambda: Bundle.model_validate({**bundle_input, "snippets": [{"code": 7}]})),
            ("invalid_dict", lambda: Bundle.model_validate({**bundle_input, "files": {"bad": {}}})),
        ]],
        "language_schemas": [
            {"language": language, "name": annotation.__name__, "description": annotation.description(),
             "optional": pydantic.TypeAdapter(annotation | None).json_schema(),
             "provider": _get_structured_outputs_response_format(
                 dspy.Signature({"result": (annotation, dspy.OutputField())})).model_json_schema()}
            for language in ("python", "C++", "C#", "İΣ", "with space", 'q"uote', "", "a.b")
            for annotation in (dspy.Code[language],)
        ],
        "types": descriptions,
        "adapters": adapters,
        "provider_schemas": {"direct": _get_structured_outputs_response_format(Direct).model_json_schema(),
                             "nested": _get_structured_outputs_response_format(Nested).model_json_schema()},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=normalized))


if __name__ == "__main__":
    main()
