"""Generate XML formatting/parsing evidence with the installed DSPy 3.5.0.

No expected outputs are implemented here: both prompts and outcomes come from
DSPy's XMLAdapter. Usage: uv run --locked gen_xml.py [out_dir]
"""

import json
import pathlib
import sys
from typing import Literal

import dspy
from dspy.adapters.utils import get_annotation_name

import gen_nested
import reference as ref


class Nullable(dspy.Signature):
    """Return nullable values without conflating null and empty text."""

    question: str = dspy.InputField()
    text: str | None = dspy.OutputField()
    number: int | None = dspy.OutputField()
    empty: str = dspy.OutputField()


class NestedOptional(dspy.Signature):
    """Return optional people and dictionary values."""

    question: str = dspy.InputField()
    person: gen_nested.Person | None = dspy.OutputField()
    people: list[gen_nested.Person | None] = dspy.OutputField()
    codes: dict[str, int | None] = dspy.OutputField()


class LiteralUnion(dspy.Signature):
    """Classify and return either a count or literal text."""

    question: str = dspy.InputField()
    label: Literal["pos", "neg"] = dspy.OutputField()
    value: int | str = dspy.OutputField()


SCHEMAS = {
    "primitive": ref.Typed,
    "nullable": Nullable,
    "nested_optional": NestedOptional,
    "extract": gen_nested.Extract,
    "tally": gen_nested.Tally,
    "literal_union": LiteralUnion,
}

PRIMITIVE = (
    "<flag>\nTrue\n</flag>\n<count>\n42\n</count>\n<ratio>\n0.25\n</ratio>\n"
    "<label>\npos\n</label>\n<items>\n<item>\none\n</item>\n<item>\ntwo\n</item>\n</items>"
)
PERSON = (
    "<name>\nBo\n</name>\n<age>\n2\n</age>\n<tags />\n"
    "<address>\n<city>\nX\n</city>\n<zip_code>\n9\n</zip_code>\n</address>\n"
    "<role>\nuser\n</role>\n<score>\n1\n</score>\n<nickname />"
)
EXTRACT = (
    "<person>\n" + PERSON + "\n</person>\n<ids>\n<item>\n1\n</item>\n<item>\n2\n</item>\n</ids>\n"
    "<pairs>\n<item>\n<item>\n1\n</item>\n</item>\n<item />\n</pairs>\n<maybe />\n<age />"
)
TALLY = (
    '<scores>\n<entry key="a&amp;&quot;b">\n2.5\n</entry>\n</scores>\n'
    "<groups>\n<x>\n<item>\np\n</item>\n</x>\n<y />\n</groups>\n"
    "<homes>\n<bo>\n<city>\nX\n</city>\n<zip_code>\n9\n</zip_code>\n</bo>\n</homes>\n"
    "<deep>\n<k>\n<z>\n2\n</z>\n</k>\n</deep>"
)
NULLABLE = "<text />\n<number />\n<empty></empty>"
NESTED_OPTIONAL = (
    "<person />\n<people>\n<item />\n<item>\n" + PERSON + "\n</item>\n</people>\n"
    '<codes>\n<entry key="a&amp;b" />\n<x>\n7\n</x>\n</codes>'
)


def parse_case(name, schema, completion):
    row = {"name": name, "schema": schema, "completion": completion}
    try:
        row["outputs"] = ref.serialize_object(dspy.XMLAdapter().parse(SCHEMAS[schema], completion))
    except dspy.AdapterParseError as error:
        row["error"] = type(error).__name__
    return row


def pydantic_schema(annotation):
    from pydantic import TypeAdapter
    return TypeAdapter(annotation).json_schema()


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    ada = gen_nested.Person(name="Ada", age=36, tags=["math"], address=gen_nested.Address(city="London", zip_code=1),
                            role="admin", score=0.5)
    formats = [
        ("primitive", {"text": "great movie", "tags": ["review", "film"]}, PRIMITIVE),
        ("nullable", {"question": "null or empty?"}, NULLABLE),
        ("nested_optional", {"question": "optional people"}, NESTED_OPTIONAL),
        ("extract", {"people": [ada], "note": None}, EXTRACT),
        ("tally", {"counts": {"b": 2, "a": 1}, "inventory": gen_nested.Inventory(
            items={"x": 1}, where={"h": gen_nested.Address(city="Oslo", zip_code=5)})}, TALLY),
        ("literal_union", {"question": "classify"}, "<label>\npos\n</label>\n<value>\n42\n</value>"),
    ]
    format_cases = []
    for schema, inputs, completion in formats:
        row = parse_case(schema, schema, completion)
        row["inputs"] = ref.serialize_object(inputs)
        row["messages"] = ref.messages_for(dspy.XMLAdapter().format(SCHEMAS[schema], demos=[], inputs=inputs))
        format_cases.append(row)
    parse_cases = [parse_case("primitive", "primitive", PRIMITIVE),
                   parse_case("null_self_close", "nullable", NULLABLE),
                   parse_case("empty_not_null", "nullable", "<text></text>\n<number />\n<empty></empty>"),
                   parse_case("nullable_missing", "nullable", "<empty>\nx\n</empty>"),
                   parse_case("missing_required", "nullable", "<text />\n<number />"),
                   parse_case("unknown_field", "nullable", NULLABLE + "\n<unknown>\nx\n</unknown>"),
                   parse_case("repeated_scalar", "nullable", NULLABLE + "\n<empty>\nx\n</empty>"),
                   parse_case("literal_inner_close", "nullable", "<text>\na\n</text> literal\nb\n</text>\n<number />\n<empty></empty>"),
                   parse_case("fenced", "nullable", "```xml\n" + NULLABLE + "\n```"),
                   parse_case("leading_thought", "nullable", "<think>private</think>\n" + NULLABLE),
                   parse_case("unclosed_thought", "nullable", "<think>private\n" + NULLABLE),
                   parse_case("multiline_close_then_prose", "nullable", "<text>\nvalue</text> prose\n<number />\n<empty></empty>"),
                   parse_case("root_entry_key", "nullable", '<entry key="text">x</entry>\n<number />\n<empty></empty>'),
                   parse_case("nested_optional", "nested_optional", NESTED_OPTIONAL),
                   parse_case("extract", "extract", EXTRACT),
                   parse_case("repeated_array_wrapper", "extract", EXTRACT.replace(
                       "<ids>\n<item>\n1\n</item>\n<item>\n2\n</item>\n</ids>",
                       "<ids>\n1\n</ids>\n<ids>\n2\n</ids>")),
                   parse_case("dictionary_escaped_key", "tally", TALLY),
                   parse_case("literal_valid", "literal_union", "<label>\nneg\n</label>\n<value>\nhello\n</value>"),
                   parse_case("literal_invalid", "literal_union", "<label>\nother\n</label>\n<value>\n42\n</value>")]
    data = {"dspy": dspy.__version__, "schemas": {name: {
        "instructions": signature.instructions,
        "json_schema": signature.model_json_schema(),
        "input_fields": list(signature.input_fields), "output_fields": list(signature.output_fields),
        "field_metadata": {field_name: {
            "annotation_name": get_annotation_name(field.annotation),
            "description": field.json_schema_extra.get("desc"),
            "prefix": field.json_schema_extra.get("prefix"),
            "json_schema": pydantic_schema(field.annotation),
        } for field_name, field in signature.fields.items()},
    } for name, signature in SCHEMAS.items()}, "format_cases": format_cases, "parse_cases": parse_cases}
    (out / "xml.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
