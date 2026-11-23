"""Actual locked-DSPy KNN/KNNFewShot probes; no native expectations are rewritten.

Run from the repository root with:
  tools/dspy_reference/.venv/bin/python tools/dspy_reference/knn_oracle.py

The JSON records NumPy's installed sort/BLAS behavior, including deliberately
nonportable cancellation results. It is development evidence, not a dependency
of the native Zig runtime or a claim of BLAS bitwise equivalence.
"""

import contextlib
import hashlib
import io
import json
import platform
import re
import struct

import dspy
import numpy as np

import reference as ref


def examples(count):
    return [dspy.Example(question=str(i), answer=str(i), index=i).with_inputs("question") for i in range(count)]


def retrieve_scores(scores, k):
    trainset = examples(len(scores))
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0] if text == "question: query" else [scores[int(text.removeprefix("question: "))]]
                for text in texts]

    retriever = dspy.KNN(k, trainset, dspy.Embedder(embed, caching=False))
    result = retriever(question="query")
    indices = [example.index for example in result]
    expected = np.asarray(scores, dtype=np.float32).argsort()[-k:][::-1].tolist()
    assert indices == expected
    assert all(example is trainset[index] for example, index in zip(result, indices))
    assert len(calls) == (len(scores) + 199) // 200 + 1
    return indices


def text_probe():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[i + 1.0, 1.0] for i, _ in enumerate(texts)]

    trainset = [
        dspy.Example(answer="excluded", z=True, a=[1, "two", None], unused="skip", nested={"yes": False}, number=1e20)
        .with_inputs("number", "nested", "a", "z"),
        dspy.Example(answer="other", z=False, a=[]).with_inputs("a", "z"),
    ]
    retriever = dspy.KNN(0, trainset, dspy.Embedder(embed, caching=False))
    result = retriever(number=1e-7, nested={"yes": None}, z=False, a=[True, "can't"])
    expected = [
        ["z: True | a: [1, 'two', None] | nested: {'yes': False} | number: 1e+20", "z: False | a: []"],
        ['number: 1e-07 | nested: {\'yes\': None} | z: False | a: [True, "can\'t"]'],
    ]
    assert calls == expected
    assert result[0] is trainset[1] and result[1] is trainset[0]
    retriever(z=True)
    assert len(calls) == 3  # Training embeddings are not recomputed.
    return calls


def reserved_key_probe():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0] for _ in texts]

    example = dspy.Example(answer="output", dspy_metadata="hidden", question="visible") \
        .with_inputs("dspy_metadata", "question")
    retriever = dspy.KNN(1, [example], dspy.Embedder(embed, caching=False))
    result = retriever(dspy_metadata="query-visible", question="query")
    assert calls == [["question: visible"], ["dspy_metadata: query-visible | question: query"]]
    assert result[0] is example and result[0].get("dspy_metadata") == "hidden"
    return {"calls": calls, "retained_metadata": result[0].get("dspy_metadata")}


def unicode_text_probe():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0] for _ in texts]

    example = dspy.Example(values=["\u0085", "\u00a0", "\u200b", "\u2028",
                                   "\u2029", "\ufeff", "日本語", "é"]).with_inputs("values")
    retriever = dspy.KNN(1, [example], dspy.Embedder(embed, caching=False))
    retriever(values=["\u200b", "日本語"], text="\u200b")
    assert calls == [
        ["values: ['\\x85', '\\xa0', '\\u200b', '\\u2028', '\\u2029', '\\ufeff', '日本語', 'é']"],
        ["values: ['\\u200b', '日本語'] | text: \u200b"],
    ]
    return calls


def numeric_probe():
    trainset = examples(2)
    count = [0]

    def raw_dot(texts):
        count[0] += 1
        return [[10.0, 0.0], [1.0, 1.0]] if count[0] == 1 else [[1.0, 1.0]]

    raw = dspy.KNN(1, trainset, dspy.Embedder(raw_dot, caching=False))
    assert raw(question="query")[0] is trainset[0]  # Cosine would select 1.
    count[0] = 0

    def rounded(texts):
        count[0] += 1
        return np.array([[1.0 + 2.0**-25], [1.0]], dtype=np.float64) if count[0] == 1 else [[1.0]]

    f32 = dspy.KNN(1, trainset, dspy.Embedder(rounded, caching=False))
    assert f32.trainset_vectors.dtype == np.float32
    assert f32(question="query")[0] is trainset[1]
    cancellation = []
    for width in (3, 4, 5, 8, 16, 17, 32, 64, 257):
        row = np.asarray(([1e8, 1.0, -1e8, 1.0] * ((width + 3) // 4))[:width], dtype=np.float32)
        query = np.ones((width, 1), dtype=np.float32)
        serial = np.float32(0.0)
        for component in row:
            serial = np.float32(serial + component)
        cancellation.append({
            "width": width,
            "one_row_dot": float(np.dot(row.reshape(1, width), query).squeeze()),
            "two_row_dot": np.dot(np.stack([row, row]), query).ravel().tolist(),
            "strict_serial_f32": float(serial),
        })
    return {"raw_dot_index": 0, "f32_rounding_index": 1, "cancellation": cancellation}


def shape_probe():
    cases = [
        ("empty", [], [[1.0, 2.0]], [], 1, "ValueError", None),
        ("empty_zero", [], [[]], [], 1, "IndexError", None),
        ("empty_zero_negative", [], [[]], [], -1, None, []),
        ("zero_width", [[], []], [[]], examples(2), 0, None, [1, 0]),
        ("width_mismatch", [[1.0, 2.0], [2.0, 3.0]], [[1.0]], examples(2), 1, "ValueError", None),
        ("extra_training_row", [[1.0], [2.0], [3.0]], [[1.0]], examples(2), 1, "IndexError", None),
        ("fewer_training_rows", [[1.0]], [[1.0]], examples(2), 1, None, [0]),
        ("two_query_rows", [[1.0, 2.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], examples(2), 1, "TypeError", None),
        ("two_query_rows_sliced_empty", [[1.0], [2.0]], [[1.0], [2.0]], examples(2), -2, None, []),
        ("one_training_two_query_rows", [[1.0]], [[1.0], [2.0]], examples(2), 0, None, [1, 0]),
        ("empty_query", [[1.0, 2.0], [2.0, 3.0]], [], examples(2), 1, "ValueError", None),
        ("zero_width_empty_query", [[], []], [], examples(2), 0, None, [1, 0]),
        ("empty_training_vectors", [], [[]], examples(2), 0, None, [0]),
    ]
    observations = []
    for name, training, query, trainset, k, error, selected in cases:
        count = [0]

        def embed(texts):
            count[0] += 1
            # Empty trainsets do not call the custom backend at initialization.
            return training if trainset and count[0] == 1 else query

        retriever = dspy.KNN(k, trainset, dspy.Embedder(embed, caching=False))
        try:
            result = [example.index for example in retriever(question="query")]
        except Exception as exc:
            assert type(exc).__name__ == error, (name, type(exc).__name__, error)
            observations.append({"case": name, "error": type(exc).__name__, "message": str(exc)})
        else:
            assert error is None and result == selected, (name, result, selected)
            observations.append({"case": name, "indices": result})
    return observations


def optimizer_capture_probe():
    calls = []

    class Program(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, question):
            calls.append([demo.question for demo in self.qa.demos])
            return self.qa(question=question)

    trainset = [dspy.Example(question=color, answer=color).with_inputs("question")
                for color in ("red", "blue")]
    vectorizer = dspy.Embedder(
        lambda texts: [[1.0, 0.0] if "red" in text else [0.0, 1.0] for text in texts],
        caching=False)
    optimizer = dspy.KNNFewShot(1, trainset, vectorizer,
                              max_bootstrapped_demos=0, max_labeled_demos=1)
    original = Program()
    compiled = optimizer.compile(original)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            dspy.context(lm=ref.KeywordLM([], "<answer>\nx\n</answer>")):
        compiled(question="red")
        optimizer.few_shot_bootstrap_args["max_labeled_demos"] = 0
        compiled(question="red")
        optimizer.KNN.k = -2
        try:
            compiled(question="red")
        except UnboundLocalError as exc:
            failure = {"error": type(exc).__name__, "message": str(exc)}
        else:
            raise AssertionError("The source unexpectedly snapshotted optimizer state")
    assert calls == [["red"], []], calls
    return {"demos": calls, "after_k_change": failure}


def teacher_signature_probe():
    class Program(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.qa(question=question)

    lm = ref.KeywordLM([], "<answer>\nx\n</answer>")
    student, teacher = Program(), Program()
    original = teacher.qa.signature
    optimizer = dspy.KNNFewShot(
        1, examples(2), dspy.Embedder(lambda texts: [[1.0] for _ in texts], caching=False),
        max_bootstrapped_demos=0, max_labeled_demos=0)
    compiled = optimizer.compile(student, teacher=teacher)
    cases = []
    for mutation, signature in [
        ("instructions", original.with_instructions("A changed teacher instruction.")),
        ("desc", original.with_updated_fields("question", desc="A changed teacher description.")),
        ("prefix", original.with_updated_fields("question", prefix="Changed question:")),
    ]:
        # This is the original borrowed teacher, mutated after wrapper compile.
        teacher.qa.signature = signature
        for k in (1, -2):
            optimizer.KNN.k = k
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                        dspy.context(lm=lm, adapter=dspy.XMLAdapter()):
                    compiled(question="query")
            except AssertionError as exc:
                assert "same signatures" in str(exc), str(exc)
                cases.append({"mutation": mutation, "k": k,
                              "error": type(exc).__name__, "message": str(exc)})
            else:
                raise AssertionError("The source accepted mismatched student/teacher signatures")
    teacher.qa.signature = original.with_instructions((" " + original.instructions)[1:])
    assert student.qa.signature.equals(teacher.qa.signature)
    optimizer.KNN.k = -2
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                dspy.context(lm=lm, adapter=dspy.XMLAdapter()):
            compiled(question="query")
    except UnboundLocalError as exc:
        equal_metadata_empty = {"error": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("The source unexpectedly accepted an empty bootstrap neighborhood")
    assert not lm.calls, lm.calls
    return {"cases": cases, "equal_metadata_empty": equal_metadata_empty, "lm_calls": len(lm.calls)}


def empty_bootstrap_probe():
    trainset = examples(2)
    embedder = dspy.Embedder(lambda texts: [[1.0] for _ in texts], caching=False)
    optimizer = dspy.KNNFewShot(-2, trainset, embedder,
                              max_bootstrapped_demos=0, max_labeled_demos=0)
    program = optimizer.compile(dspy.Predict("question -> answer"))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                dspy.context(lm=ref.KeywordLM([], "<answer>\nx\n</answer>")):
            program(question="query")
    except UnboundLocalError as exc:
        return {"error": type(exc).__name__, "message": str(exc)}
    raise AssertionError("The source unexpectedly accepted an empty bootstrap neighborhood")


def fewshot_probe():
    calls, embeddings, forwards = [], [], []

    class LM(ref.ReferenceLM):
        def __init__(self, label):
            super().__init__(model="mock/" + label, model_type="chat", temperature=0.0, max_tokens=1000, cache=False)
            self.label = label

        def complete(self, request):
            messages = ref.messages_for(request)
            questions = [re.search(r"<question>\s*(.*?)\s*</question>", m["content"], re.S).group(1)
                         for m in messages if m["role"] == "user"]
            answers = [re.search(r"<answer>\s*(.*?)\s*</answer>", m["content"], re.S).group(1)
                       for m in messages if m["role"] == "assistant"]
            calls.append({"lm": self.label, "questions": questions, "answers": answers})
            answer = self.label + ":" + questions[-1] + ":" + str(len(answers))
            return ref.text_response(self.model, "<answer>\n" + answer + "\n</answer>")

    class Program(dspy.Module):
        def __init__(self, marker):
            super().__init__()
            self.qa = dspy.Predict("question -> answer")
            self.marker, self.counter = marker, 7

        def forward(self, question):
            forwards.append([self.marker, self.counter, question])
            self.counter += 1
            return self.qa(question=question + "|" + self.marker)

    def embed(texts):
        embeddings.append(list(texts))
        return [[1.0, 0.0] if "red" in text else [0.0, 1.0] for text in texts]

    trainset = [dspy.Example(question=color + "-train", answer=color + "-gold").with_inputs("question")
                for color in ("red", "blue")]
    student, teacher = Program("original"), Program("teacher")
    student.qa.demos = [dspy.Example(question="old", answer="old")]
    teacher._compiled = True
    teacher.qa.demos = [dspy.Example(question="teacher-old", answer="teacher-old")]
    optimizer = dspy.KNNFewShot(1, trainset, dspy.Embedder(embed, caching=False),
                              max_bootstrapped_demos=1, max_labeled_demos=0,
                              metric=lambda example, prediction, trace: len(trace) == 1,
                              teacher_settings={"lm": LM("teacher")})
    compiled = optimizer.compile(student, teacher=teacher)
    visible = {"marker": compiled.marker, "counter": compiled.counter, "compiled": compiled._compiled,
               "predictors": [name for name, _ in compiled.named_predictors()], "demos": len(compiled.qa.demos)}
    assert visible == {"marker": "original", "counter": 7, "compiled": False, "predictors": ["qa"], "demos": 0}
    student.marker, teacher.marker, compiled.marker = "captured-mutated", "teacher-mutated", "visible-mutated"
    compiled.qa.demos = [dspy.Example(question="visible-only", answer="visible-only")]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            dspy.context(lm=LM("student"), adapter=dspy.XMLAdapter(), trace=[]):
        outputs = [compiled(question=color + "-query").answer for color in ("red", "blue")]
        trace = [{"question": step[1]["question"], "answer": step[2]["answer"]} for step in dspy.settings.trace]
    assert outputs == ["student:red-query|captured-mutated:1", "student:blue-query|captured-mutated:1"]
    assert forwards == [["teacher-mutated", 7, "red-train"], ["captured-mutated", 7, "red-query"],
                        ["teacher-mutated", 7, "blue-train"], ["captured-mutated", 7, "blue-query"]]
    assert [call["lm"] for call in calls] == ["teacher", "student", "teacher", "student"]
    assert calls[1]["questions"] == ["red-train|teacher-mutated", "red-query|captured-mutated"]
    assert calls[3]["questions"] == ["blue-train|teacher-mutated", "blue-query|captured-mutated"]
    assert [step["question"] for step in trace] == ["red-query|captured-mutated", "blue-query|captured-mutated"]
    assert student.counter == teacher.counter == compiled.counter == 7
    assert not compiled._compiled and compiled.qa.demos[0].question == "visible-only"
    # A separate closure without a teacher must still capture its own original.
    second = Program("second")
    second.qa.demos = [dspy.Example(question="second-old", answer="second-old")]
    independent = optimizer.compile(second)
    second.marker, independent.marker = "second-captured", "second-visible"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            dspy.context(lm=LM("student"), adapter=dspy.XMLAdapter()):
        fallback = independent(question="red-query").answer
    assert fallback == "student:red-query|second-captured:1"
    assert forwards[-2:] == [["second-captured", 7, "red-train"], ["second-captured", 7, "red-query"]]
    assert calls[-2]["questions"] == ["second-old", "red-train|second-captured"]
    assert independent.qa.demos == [] and compiled.marker == "visible-mutated"
    return {"visible_initial": visible, "outputs": outputs, "fallback_output": fallback,
            "embeddings": embeddings, "forwards": forwards, "calls": calls, "trace": trace}


def main():
    assert dspy.__version__ == "3.5.0", dspy.__version__
    sorting = []
    for count in (17, 20, 33, 64):
        for label, scores in (("equal", [1.0] * count),
                              ("ties", [(i * 7) % 5 for i in range(count)]),
                              ("nan", [float("nan") if i % 7 == 0 else i % 3 - 1 for i in range(count)]),
                              ("zeros", [-0.0 if i % 2 else 0.0 for i in range(count)])):
            sorting.append({"case": label, "count": count, "indices": retrieve_scores(scores, 0)})
    pipe_indices = retrieve_scores([min(i, 256 - i) for i in range(257)], 0)
    pipe_encoding = b"".join(struct.pack("<Q", index) for index in reversed(pipe_indices))
    sorting.append({"case": "organ_pipe_depth_fallback", "count": 257,
                    "indices": pipe_indices, "ascending_u64le_sha256": hashlib.sha256(pipe_encoding).hexdigest()})
    special = [2.0, float("nan"), 2.0, -0.0, 0.0, float("inf"), -float("inf"), float("nan")]
    slices = {str(k): retrieve_scores(special, k) for k in (0, 1, 3, -1, -3, -8, -9, 9)}
    print(json.dumps({"dspy": dspy.__version__, "numpy": np.__version__, "machine": platform.machine(),
                      "system": platform.system(), "text": text_probe(), "sorting": sorting,
                      "reserved_keys": reserved_key_probe(), "unicode_text": unicode_text_probe(),
                      "slices": slices, "numeric": numeric_probe(), "shapes": shape_probe(),
                      "fewshot": fewshot_probe(), "empty_bootstrap": empty_bootstrap_probe(),
                      "live_optimizer": optimizer_capture_probe(),
                      "teacher_signature": teacher_signature_probe()},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
