"""Golden fixture for nested field types (Pydantic models, list[T], Optional)
in DSPy 3.4.1: ChatAdapter and JSONAdapter prompts plus one parsed answer.

Usage: uv run gen_nested.py [out_dir]  (default: ../../tests/fixtures)"""

import json
import pathlib
import sys
from typing import Literal, Optional

import dspy
import pydantic


class Address(pydantic.BaseModel):
    city: str
    zip_code: int


class Person(pydantic.BaseModel):
    name: str
    age: int
    tags: list[str]
    address: Address
    role: Literal["admin", "user"]
    score: float
    nickname: Optional[str] = None


class Extract(dspy.Signature):
    """Extract people."""

    people: list[Person] = dspy.InputField()
    note: Optional[str] = dspy.InputField()
    person: Person = dspy.OutputField()
    ids: list[int] = dspy.OutputField()
    pairs: list[list[int]] = dspy.OutputField()
    maybe: Optional[str] = dspy.OutputField()
    age: Optional[int] = dspy.OutputField()


class Inventory(pydantic.BaseModel):
    items: dict[str, int]
    where: dict[str, Address]


class Tally(dspy.Signature):
    """Tally things."""

    counts: dict[str, int] = dspy.InputField()
    inventory: Inventory = dspy.InputField()
    scores: dict[str, float] = dspy.OutputField()
    groups: dict[str, list[str]] = dspy.OutputField()
    homes: dict[str, Address] = dspy.OutputField()
    deep: dict[str, dict[str, int]] = dspy.OutputField()


DICT_COMPLETION = (
    '[[ ## scores ## ]]\n{"b": 1, "a": "2.5"}\n\n[[ ## groups ## ]]\n{"x": ["p"], "y": []}\n\n'
    '[[ ## homes ## ]]\n{"bo": {"city": "X", "zip_code": "9"}}\n\n[[ ## deep ## ]]\n{"k": {"z": 2.0}}\n\n'
    "[[ ## completed ## ]]"
)

COMPLETION = (
    '[[ ## person ## ]]\n{"name": "Bo", "age": 2.0, "tags": [], "address": {"city": "X", "zip_code": "9"}, '
    '"role": "user", "score": 1, "extra": true}\n\n[[ ## ids ## ]]\n[1, 2]\n\n[[ ## pairs ## ]]\n[[1], []]\n\n'
    "[[ ## maybe ## ]]\nNone\n\n[[ ## age ## ]]\nNone\n\n[[ ## completed ## ]]"
)


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    ada = Person(name="Ada", age=36, tags=["math"], address=Address(city="London", zip_code=1), role="admin", score=0.5)
    inputs = {"people": [ada], "note": None}
    cases = []
    for name, adapter in (("chat", dspy.ChatAdapter()), ("json", dspy.JSONAdapter())):
        cases.append({"adapter": name, "messages": adapter.format(Extract, demos=[], inputs=inputs)})
    parsed = dspy.ChatAdapter().parse(Extract, COMPLETION)
    outputs = {k: pydantic.TypeAdapter(type(v)).dump_python(v, mode="json") for k, v in parsed.items()}
    dict_inputs = {"counts": {"b": 2, "a": 1}, "inventory": Inventory(items={"x": 1}, where={"h": Address(city="Oslo", zip_code=5)})}
    dict_cases = []
    for name, adapter in (("chat", dspy.ChatAdapter()), ("json", dspy.JSONAdapter())):
        dict_cases.append({"adapter": name, "messages": adapter.format(Tally, demos=[], inputs=dict_inputs)})
    parsed = dspy.ChatAdapter().parse(Tally, DICT_COMPLETION)
    dict_outputs = {k: pydantic.TypeAdapter(Tally.output_fields[k].annotation).dump_python(v, mode="json") for k, v in parsed.items()}
    data = {
        "cases": cases, "completion": COMPLETION, "outputs": outputs,
        "dict_cases": dict_cases, "dict_completion": DICT_COMPLETION, "dict_outputs": dict_outputs,
    }
    (out / "nested.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
