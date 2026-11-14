"""Pinned DSPy 3.5 / NumPy 2.4.6 Embeddings observations.

Run with tools/dspy_reference/.venv/bin/python. Optional --write-fixtures DIR
writes genuine Python save directories for a native consumer; --native-save DIR
loads a native save with actual DSPy and NumPy. No native output is used to
construct expectations, and this tool does not rewrite native fixtures/tests.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import tempfile

import dspy
import numpy as np
from dspy.retrievers.embeddings import Embeddings, EmbeddingsWithScores


def number(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return value


def array(value):
    value = np.asarray(value)
    result = {"dtype": value.dtype.str, "shape": list(value.shape)}
    if value.dtype.kind == "f":
        result["values"] = [number(x) for x in value.ravel(order="C")]
        unsigned = np.dtype(f"u{value.dtype.itemsize}").newbyteorder(value.dtype.byteorder)
        result["bits"] = [int(x) for x in value.view(unsigned).ravel(order="C")]
    else:
        result["values"] = value.tolist()
    return result


def failure(exc):
    return {"error": type(exc).__name__, "message": str(exc)}


def batch_result(results):
    return [{"passages": passages, "indices": indices,
             "scores": [number(score) for score in scores],
             "score_bits": array(np.asarray(scores, dtype=np.float32))["bits"]}
            for passages, indices, scores in results]


@contextlib.contextmanager
def managed(retriever):
    try:
        yield retriever
    finally:
        # Unbatchify.close prints an unrelated thread-lifecycle message.
        with contextlib.redirect_stdout(io.StringIO()):
            retriever.search_fn.close()


class Backend:
    def __init__(self, corpus_vectors, query_vectors, *, first_is_corpus=True):
        self.first_is_corpus = first_is_corpus
        self.corpus_vectors = corpus_vectors
        self.query_vectors = query_vectors
        self.calls = []
        self.embedder = dspy.Embedder(self.embed, caching=False)

    def embed(self, texts):
        first = self.first_is_corpus and not self.calls
        self.calls.append(list(texts))
        return self.corpus_vectors if first else self.query_vectors


def retrieve_case(name, corpus_vectors, query_vectors, *, corpus=None,
                  queries=None, k=5, normalize=True, **options):
    if corpus is None:
        corpus = [f"passage-{i}" for i in range(len(corpus_vectors))]
    if queries is None:
        queries = [f"query-{i}" for i in range(len(query_vectors))]
    # Real Embedder returns (0,) without calling its backend for an empty corpus.
    backend = Backend(corpus_vectors, query_vectors, first_is_corpus=bool(corpus))
    observation = {"name": name, "k": k, "normalize": normalize}
    try:
        with managed(Embeddings(corpus, backend.embedder, k=k,
                                normalize=normalize, **options)) as retriever:
            observation["corpus_embeddings"] = array(retriever.corpus_embeddings)
            observation["results"] = batch_result(retriever._batch_forward(queries))
    except Exception as exc:
        observation.update(failure(exc))
    observation["calls"] = backend.calls
    return observation


def ranking_cases():
    result = []
    corpus = [[3.0, 0.0], [0.0, 4.0], [0.0, 0.0], [-3.0, 0.0]]
    for normalize in (True, False):
        for k in (5, 2, 0, -1, -4, -9, 99):
            result.append(retrieve_case(f"slice-{normalize}-{k}", corpus,
                                       [[3.0, 4.0], [0.0, 0.0]],
                                       k=k, normalize=normalize))
    scores = [float("nan") if i % 7 == 0 else float(i % 3) for i in range(33)]
    result.append(retrieve_case("nan-ties-33", [[x] for x in scores], [[1.0]],
                               normalize=False, k=33))
    result.append(retrieve_case("equal-ties-17", [[1.0]] * 17, [[1.0]],
                               normalize=False, k=17))
    result.append(retrieve_case("signed-zero-ties-17",
                               [[0.0 if i % 2 else -0.0] for i in range(17)],
                               [[1.0]], normalize=False, k=17))
    result.append(retrieve_case("nonfinite-normalization",
                               [[0.0, -0.0], [float("inf"), 1.0],
                                [float("nan"), 2.0], [1e-30, 0.0],
                                [1e30, 1e30]], [[1.0, 0.0]], k=5))
    backend = Backend([[1.0 + 2.0**-25], [1.0]], [[1.0]])
    with managed(Embeddings(["rounded", "one"], backend.embedder,
                            normalize=False)) as retriever:
        result.append({"name": "embedder-float64-coerced-float32",
                       "corpus_embeddings": array(retriever.corpus_embeddings),
                       "results": batch_result(retriever._batch_forward(["query"]))})
    return result


def shape_cases():
    return [
        retrieve_case("empty-corpus-normalized", [], [[1.0]], corpus=[]),
        retrieve_case("empty-corpus-raw", [], [[1.0]], corpus=[], normalize=False),
        retrieve_case("empty-query-normalized", [[1.0]], [], queries=[]),
        retrieve_case("empty-query-raw", [[1.0]], [], queries=[], normalize=False),
        retrieve_case("zero-width", [[], []], [[]], k=2),
        retrieve_case("query-width-one-broadcast", [[1.0, 2.0]], [[1.0]]),
        retrieve_case("corpus-width-one-broadcast", [[2.0]], [[1.0, 2.0]], normalize=False),
        retrieve_case("width-mismatch", [[1.0, 2.0]], [[1.0, 2.0, 3.0]]),
        retrieve_case("zero-width-query-one", [[], []], [[1.0]], normalize=False),
        retrieve_case("one-width-query-zero", [[1.0]], [[]], normalize=False),
        retrieve_case("extra-corpus-row", [[1.0], [3.0]], [[1.0]],
                      corpus=["only"], normalize=False),
        retrieve_case("extra-corpus-row-sliced-empty", [[1.0], [3.0]], [[1.0]],
                      corpus=["only"], normalize=False, k=0),
        retrieve_case("fewer-corpus-rows", [[1.0]], [[1.0]], corpus=["first", "second"]),
        retrieve_case("extra-query-rows", [[1.0]], [[1.0], [2.0]], queries=["one"],
                      normalize=False),
        retrieve_case("no-query-rows", [[1.0]], [], queries=["one"], normalize=False),
        retrieve_case("ragged-corpus", [[1.0], [1.0, 2.0]], [[1.0]]),
        retrieve_case("cache-rejected-before-embedding", [[1.0]], [[1.0]], cache=True),
        retrieve_case("faiss-threshold-after-embedding", [[1.0]], [[1.0]],
                      brute_force_threshold=1),
        retrieve_case("empty-faiss-dimension-before-import", [], [[1.0]], corpus=[],
                      normalize=False, brute_force_threshold=0),
    ]


def runtime_cases():
    backend = Backend([[3.0, 0.0], [0.0, 4.0]], [[3.0, 4.0]])
    with managed(EmbeddingsWithScores(["east", "north"], backend.embedder, k=1)) as retriever:
        first = retriever.forward("single")
        first_value = {"passages": first.passages, "indices": first.indices,
                       "scores": first.scores, "fields": list(first.keys()),
                       "score_types": [type(score).__name__ for score in first.scores],
                       "above_decimal_point_eight": first.scores[0] > 0.8,
                       "equal_decimal_point_eight": first.scores[0] == 0.8}
        retriever.k = -1
        retriever.normalize = False  # Does not renormalize stored corpus vectors.
        retriever.corpus[1] = "renamed"
        second = retriever.forward("mutated")
        mutated = {"passages": second.passages, "indices": second.indices,
                   "scores": second.scores}
        retriever.corpus_embeddings = np.asarray([[4.0, 0.0], [0.0, 1.0]], np.float32)
        replaced = batch_result(retriever._batch_forward(["replaced"]))
        return {"first": first_value, "mutated": mutated, "replaced_vectors": replaced,
                "calls": backend.calls}


def arithmetic_cases():
    records = []
    rng = np.random.RandomState(418)
    for width in (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 129, 257):
        row = np.asarray(([1e8, 1.0, -1e8, 1.0] * ((width + 3) // 4))[:width], np.float32)
        for query_rows, corpus_rows in ((1, 1), (1, 2), (2, 1), (3, 17)):
            query = np.ones((query_rows, width), np.float32)
            corpus = np.tile(row, (corpus_rows, 1))
            scores = np.einsum("qd,kd->qk", query, corpus, order="C")
            records.append({"width": width, "query_rows": query_rows,
                            "corpus_rows": corpus_rows, "scores": array(scores)})
    norm_records = []
    for width in (1, 3, 8, 17, 128, 129, 257):
        vectors = (rng.standard_normal((3, width)) * 10).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.maximum(norms, 1e-10)
        norm_records.append({"width": width, "input": array(vectors),
                             "norms": array(norms), "normalized": array(normalized)})
    # Inputs and exact output bits allow a consumer to distinguish fused SIMD
    # products from serial dot accumulation without relying only on cancellation.
    random_query = rng.standard_normal((3, 33)).astype(np.float32)
    random_corpus = rng.standard_normal((17, 33)).astype(np.float32)
    return {"cancellation": records, "normalization": norm_records,
            "fused_inputs": {"queries": array(random_query), "corpus": array(random_corpus),
                             "scores": array(np.einsum("qd,kd->qk", random_query,
                                                       random_corpus, order="C"))}}


class TextBackend:
    def __init__(self):
        self.calls = []
        self.embedder = dspy.Embedder(self.embed, caching=False)

    def embed(self, texts):
        self.calls.append(list(texts))
        table = {"east": [3.0, 0.0], "north": [0.0, 4.0],
                 "zero": [0.0, 0.0], "query": [3.0, 4.0]}
        return [table.get(text, [3.0, 4.0]) for text in texts]


def saved_observation(path):
    backend = TextBackend()
    with managed(EmbeddingsWithScores.from_saved(str(path), backend.embedder)) as loaded:
        calls_after_load = list(backend.calls)
        result = loaded.forward("query")
        return {"config": json.loads((path / "config.json").read_text()),
                "array": array(np.load(path / "corpus_embeddings.npy", allow_pickle=False)),
                "calls_after_load": calls_after_load, "calls_after_query": backend.calls,
                "result": {"passages": result.passages, "indices": result.indices,
                           "scores": [number(x) for x in result.scores]}}


def persistence_cases(root):
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for normalize in (True, False):
        backend = TextBackend()
        path = root / ("normalized" if normalize else "raw")
        with managed(EmbeddingsWithScores(["east", "north", "zero"], backend.embedder,
                                         k=-1, normalize=normalize)) as retriever:
            retriever.save(str(path))
            assert backend.calls == [["east", "north", "zero"]]
        result[path.name] = saved_observation(path)
        assert result[path.name]["calls_after_load"] == []
        assert result[path.name]["calls_after_query"] == [["query"]]

    raw = root / "raw"
    with managed(EmbeddingsWithScores.from_saved(str(raw), TextBackend().embedder)) as loaded:
        # Both normal NumPy memory layouts and byte orders are real np.load paths.
        original = loaded.corpus_embeddings
        for name, vectors in (
            ("fortran-f32", np.asfortranarray(original)),
            ("big-endian-f32", original.astype(">f4")),
            ("float64", original.astype(np.float64)),
            ("rank-one-empty", np.asarray([], np.float32)),
            ("rank-two-empty", np.empty((0, 2), np.float32)),
        ):
            loaded.corpus_embeddings = vectors
            path = root / name
            loaded.save(str(path))
            try:
                result[name] = saved_observation(path)
            except Exception as exc:
                result[name] = {"array": array(vectors), **failure(exc)}
        loaded.corpus_embeddings = original
        for name, version, vectors in (
            ("npy-v2-fortran-big", (2, 0), np.asfortranarray(original.astype(">f4"))),
            ("npy-v3", (3, 0), original),
        ):
            path = root / name
            loaded.save(str(path))
            with (path / "corpus_embeddings.npy").open("wb") as stream:
                np.lib.format.write_array(stream, vectors, version=version, allow_pickle=False)
            result[name] = saved_observation(path)

    result.update(fortran_header_cases(root / "fortran-header"))

    # load replaces an existing instance without invoking its new embedder.
    backend = TextBackend()
    with managed(Embeddings(["east"], backend.embedder)) as loaded:
        replacement = TextBackend()
        assert loaded.load(str(raw), replacement.embedder) is loaded
        result["load-existing"] = {"old_calls": backend.calls,
                                   "new_calls": replacement.calls,
                                   "k": loaded.k, "normalize": loaded.normalize,
                                   "corpus": loaded.corpus}

    invalid = root / "invalid"
    invalid.mkdir(exist_ok=True)
    np.save(invalid / "corpus_embeddings.npy", np.zeros((1, 2), np.float32))
    errors = []
    valid = {"k": 5, "normalize": True, "corpus": ["east"], "has_faiss_index": False}
    for field in valid:
        config = {key: value for key, value in valid.items() if key != field}
        (invalid / "config.json").write_text(json.dumps(config))
        try:
            candidate = Embeddings.from_saved(str(invalid), TextBackend().embedder)
        except Exception as exc:
            errors.append({"missing": field, **failure(exc)})
        else:
            with managed(candidate):
                raise AssertionError(f"Missing config field accepted: {field}")
    result["invalid-config"] = errors
    # Do not leave a deliberately invalid config among the reusable fixtures.
    (invalid / "config.json").write_text(json.dumps(valid))
    corrupt = root / "corrupt-array"
    corrupt.mkdir(exist_ok=True)
    (corrupt / "config.json").write_text(json.dumps(
        {"k": -2, "normalize": False, "corpus": ["new"], "has_faiss_index": False}))
    (corrupt / "corpus_embeddings.npy").write_bytes(b"invalid-npy")
    backend = TextBackend()
    with managed(Embeddings(["east"], backend.embedder, k=3)) as loaded:
        replacement = TextBackend()
        try:
            loaded.load(str(corrupt), replacement.embedder)
        except Exception as exc:
            result["partial-mutation-on-array-failure"] = {
                **failure(exc), "k": loaded.k, "normalize": loaded.normalize,
                "corpus": loaded.corpus, "old_array": array(loaded.corpus_embeddings),
                "old_calls": backend.calls, "new_calls": replacement.calls,
                "embedder_replaced": loaded.embedder is replacement.embedder}
        else:
            raise AssertionError("Corrupt array was unexpectedly accepted")
    for name, path in (("directory-missing", root / "missing"),
                       ("config-missing", root / "missing-config"),
                       ("array-missing", root / "missing-array")):
        if name != "directory-missing":
            path.mkdir(exist_ok=True)
        if name == "array-missing":
            (path / "config.json").write_text("not json")
        try:
            candidate = Embeddings.from_saved(str(path), TextBackend().embedder)
        except Exception as exc:
            result[name] = failure(exc)
        else:
            with managed(candidate):
                raise AssertionError(f"Missing file accepted: {name}")
    return result


def fortran_header_cases(root):
    # Hand-written headers claim fortran_order=True. DSPy's real np.load/np.save
    # canonicalize degenerate shapes (one row or column) and rank one to False.
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, shape in (("degenerate-row", (1, 3)), ("degenerate-column", (3, 1)),
                        ("nondegenerate-matrix", (2, 3)), ("scalar", ()),
                        ("vector", (3,)), ("empty-rows", (0, 3)),
                        ("empty-columns", (3, 0)), ("unit-matrix", (1, 1))):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        data = np.arange(int(np.prod(shape)), dtype="<f4").reshape(shape, order="F")
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": True,
                                                      "shape": shape})
        stream.write(data.tobytes(order="F"))
        (path / "corpus_embeddings.npy").write_bytes(stream.getvalue())
        (path / "config.json").write_text(json.dumps(
            {"k": 5, "normalize": False, "corpus": [f"{name}-{i}" for i in range(shape[0] if shape else 1)],
             "has_faiss_index": False}))
        with managed(EmbeddingsWithScores.from_saved(str(path), TextBackend().embedder)) as loaded:
            loaded.save(str(path / "resaved"))
        result[name] = {"input_hex": stream.getvalue().hex(),
                        "saved_hex": (path / "resaved" / "corpus_embeddings.npy").read_bytes().hex(),
                        "array": array(np.load(path / "corpus_embeddings.npy", allow_pickle=False))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixtures", type=Path)
    parser.add_argument("--native-save", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert dspy.__version__ == "3.5.0", dspy.__version__
    assert np.__version__ == "2.4.6", np.__version__
    with np.errstate(all="ignore"):
        observations = {"dspy": dspy.__version__, "numpy": np.__version__,
                        "machine": platform.machine(), "ranking": ranking_cases(),
                        "shapes": shape_cases(), "runtime": runtime_cases(),
                        "arithmetic": arithmetic_cases()}
        if args.write_fixtures:
            observations["persistence"] = persistence_cases(args.write_fixtures)
        else:
            with tempfile.TemporaryDirectory(prefix="dszig-embeddings-oracle-") as temporary:
                observations["persistence"] = persistence_cases(Path(temporary))
        if args.native_save:
            observations["native_save"] = saved_observation(args.native_save)
    encoded = json.dumps(observations, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                          "ranking_cases": len(observations["ranking"]),
                          "shape_cases": len(observations["shapes"])}))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
