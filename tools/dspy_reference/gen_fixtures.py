"""Regenerate tests/fixtures from DSPy 3.4.1. Run: uv run gen_fixtures.py"""

import json
import pathlib
import sys

import dspy

import reference as ref

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def run_case(name, module, adapter, demos, inputs, completion):
    lm = ref.KeywordLM([], completion)
    dspy.configure(lm=lm, adapter=adapter)
    predictor = module.predict if hasattr(module, "predict") else module
    predictor.demos = demos
    pred = module(**inputs)
    return {
        "name": name,
        "messages": lm.calls[0]["messages"],
        "completion": completion,
        "outputs": dict(pred.items()),
    }


def format_cases():
    qa_demos = [
        dspy.Example(augmented=True, context=["a fact"], question="q1?", reasoning="r1", answer="a1"),
        dspy.Example(question="q2?", answer="a2"),
    ]
    qa_inputs = {"context": ["Paris is the capital of France.", "multi\nline «x»"], "question": "What is the capital of France?"}
    typed_demos = [
        dspy.Example(text="good", tags=["x"], flag=True, count=3, ratio=0.5, label="pos", items=["a", "b"]),
        dspy.Example(text="bad", flag=False, label="neg"),
    ]
    typed_inputs = {"text": "great movie", "tags": ["review", "film"]}
    typed_chat = (
        "[[ ## flag ## ]]\nTrue\n\n[[ ## count ## ]]\n42\n\n[[ ## ratio ## ]]\n0.25\n\n"
        "[[ ## label ## ]]\n'pos'\n\n[[ ## items ## ]]\n[\"one\", \"two\"]\n\n[[ ## completed ## ]]"
    )
    typed_json = '{"flag": true, "count": 7, "ratio": 1, "label": "neg", "items": ["z"]}'
    return [
        run_case("cot_qa_chat", dspy.ChainOfThought(ref.QA), dspy.ChatAdapter(), qa_demos, qa_inputs,
                 ref.chat("Paris is the capital.", "Paris")),
        run_case("cot_qa_json", dspy.ChainOfThought(ref.QA), dspy.JSONAdapter(), qa_demos, qa_inputs,
                 '```json\n{"reasoning": "Because.", "answer": "Paris"}\n```'),
        run_case("typed_chat", dspy.Predict(ref.Typed), dspy.ChatAdapter(), typed_demos, typed_inputs, typed_chat),
        run_case("typed_json", dspy.Predict(ref.Typed), dspy.JSONAdapter(), typed_demos, typed_inputs, typed_json),
        run_case("indented_chat", dspy.Predict(ref.Indented), dspy.ChatAdapter(), [], {"text": "meh"},
                 "[[ ## label ## ]]\nLiteral['neg']\n\n[[ ## completed ## ]]"),
    ]


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"wrote {OUT / name}", file=sys.stderr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("format_cases.json", json.dumps(format_cases(), indent=2, ensure_ascii=False) + "\n")
    compiled, lm = ref.compile_rag()
    path = OUT / "bootstrap_rag.json"
    compiled.save(str(path))
    print(f"wrote {path}", file=sys.stderr)
    evaluate = dspy.Evaluate(devset=ref.trainset(), metric=ref.keyword_metric, display_progress=False)
    baseline = evaluate(ref.RAG()).score
    optimized = evaluate(compiled).score
    multihop, _ = ref.compile_multihop()
    multihop.save(str(OUT / "bootstrap_multihop.json"))
    write("bootstrap_rag_eval.json", json.dumps({"baseline": baseline, "optimized": optimized}, indent=2) + "\n")


if __name__ == "__main__":
    main()
