"""Actual pinned DSPy 3.5 ColBERTv2 observations over real localhost HTTP.

Run with tools/dspy_reference/.venv/bin/python. This oracle invokes DSPy's
unmodified ColBERTv2 and Cache; its server only supplies HTTP response fixtures.
No requests are mocked, no native output supplies expectations, and no native
fixtures/tests are rewritten. Disk caching and authenticated services are not
exercised. Optional --output PATH saves the complete source receipt.

--serve [--port PORT] serves the SAME response fixtures for a native consumer.
It prints a ready JSON object with its base URL. GET /__oracle__/requests returns
the received request records without itself being recorded. Importing Server
and Handler also permits an in-process consumer; no DSPy calls run on import.
"""

import argparse
import contextlib
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import inspect
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

import dspy
from dspy.clients.cache import Cache

colbert = importlib.import_module("dspy.dsp.colbertv2")


def rows():
    return [
        {"text": text, "long_text": f"server-{text}", "score": 1.0 / (i + 1),
         "meta": {"tags": ["seed"]}, "labels": ["base"],
         "docid": 9007199254740993 + i, "huge": 18446744073709551619 + i}
        for i, text in enumerate(("alpha", "beta", "gamma", "delta", "epsilon"))
    ]


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port=0):
        super().__init__(("127.0.0.1", port), Handler)
        self.receipts = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.respond()

    def do_POST(self):
        self.respond()

    def respond(self):
        parts = urlsplit(self.path)
        if self.command == "GET" and parts.path == "/__oracle__/requests":
            records = self.server.receipts[:]
            self.send_payload(200, json.dumps({"requests": records, "total": len(records)}).encode("utf-8"))
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        params = json.loads(raw) if raw else parse_qs(parts.query, keep_blank_values=True)
        self.server.receipts.append({
            "method": self.command, "target": self.path,
            "content_type": self.headers.get("Content-Type"),
            "body_ascii": raw.decode("ascii"), "body_hex": raw.hex(),
            "parameters": params,
        })
        path = parts.path
        status = 200
        body = {"topk": rows()}
        if path == "/http-error":
            status, body = 418, {"error": "HTTP status takes precedence"}
        elif path == "/error-message":
            body = {"error": "failed", "message": "index unavailable"}
        elif path == "/error-default":
            body = {"error": 1}
        elif path == "/error-null-message":
            body = {"error": True, "message": None}
        elif path == "/error-false":
            body = {"error": [], "topk": rows()}
        elif path == "/missing-topk":
            body = {"message": "missing passages"}
        elif path == "/nonobject":
            body = []
        elif path == "/topk-null":
            body = {"topk": None}
        elif path == "/row-missing-text":
            body = {"topk": [{"long_text": "already supplied", "meta": {"tags": []}}]}
        elif path == "/missing-long-text":
            body = {"topk": [{"text": "only text", "meta": {"tags": []}}]}
        elif path == "/typed-long-text":
            body = {"topk": [{"text": {"tokens": [1]}, "long_text": 23},
                              {"text": False, "long_text": None},
                              {"text": ["nested"], "long_text": ["server"]}]}
        elif path == "/row-scalar":
            body = {"topk": [7]}
        encoded = b"not JSON" if path == "/bad-json" else json.dumps(body).encode("utf-8")
        self.send_payload(status, encoded)

    def send_payload(self, status, encoded):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class AuditedCache(Cache):
    """Observe real Cache calls without changing its cache/ownership semantics."""

    def __init__(self, *, enabled=True, max_entries=1000000):
        super().__init__(False, enabled, None, memory_max_entries=max_entries)
        self.events = []
        self.requests = {}

    def get(self, request, ignored_args_for_cache_key=None):
        value = super().get(request, ignored_args_for_cache_key)
        self.events.append({"op": "get", "function": request["_fn_identifier"],
                            "hit": value is not None})
        self.requests[request["_fn_identifier"]] = copy.deepcopy(request)
        return value

    def put(self, request, value, ignored_args_for_cache_key=None, enable_memory_cache=True):
        super().put(request, value, ignored_args_for_cache_key, enable_memory_cache)
        self.events.append({"op": "put", "function": request["_fn_identifier"]})
        self.requests[request["_fn_identifier"]] = copy.deepcopy(request)

    def request_for(self, layer, *, post=False):
        name = "colbertv2_post_request_v2" if post else "colbertv2_get_request_v2"
        if layer == "outer":
            name += "_wrapped"
        return self.requests[f"{colbert.__name__}.{name}"]


@contextlib.contextmanager
def cache_scope(**options):
    original = dspy.cache
    cache = AuditedCache(**options)
    dspy.cache = cache
    try:
        yield cache
    finally:
        dspy.cache = original


def error_result(exc, server):
    return {"error": type(exc).__name__,
            "message": str(exc).replace(server.url, "http://<server>")}


def invoke(server, client, query, k, *, simplify=False):
    before = len(server.receipts)
    try:
        result = {"value": copy.deepcopy(client(query, k=k, simplify=simplify))}
    except Exception as exc:
        result = error_result(exc, server)
    result["requests"] = copy.deepcopy(server.receipts[before:])
    return result


def ordinary_cases(server):
    observations = []
    with cache_scope(enabled=False):
        for post in (False, True):
            client = colbert.ColBERTv2(server.url + "/rows", post_requests=post)
            for k in (2, 0, -1, -2, -4, -9, -(2**63), 100, 101):
                result = invoke(server, client, "ordinary /?&=+%", k)
                result.update({"method": "POST" if post else "GET", "k": k})
                if not post and k == 101:
                    assert result["error"] == "AssertionError" and not result["requests"], result
                else:
                    expected = rows()[:k]
                    if not post:
                        expected = [{**row, "long_text": row["text"]} for row in expected][:k]
                    assert result["value"] == expected, result
                    assert len(result["requests"]) == 1, result
                observations.append(result)
            simplified = invoke(server, client, "simplify", 2, simplify=True)
            assert simplified["value"] == (["server-alpha", "server-beta"] if post else ["alpha", "beta"])
            observations.append({"method": "POST" if post else "GET", "simplify": simplified})
    return observations


def encoding_cases(server):
    query = 'café 雪 😀 /?&=+%~\"\\\n\t\x00\x1f\x7f\u2028'
    url = server.url + "/café/%7e?existing=é%20%2B&blank=#ignored"
    result = {}
    with cache_scope(enabled=False):
        for post in (False, True):
            observed = invoke(server, colbert.ColBERTv2(url, post_requests=post), query, -1)
            request = observed["requests"][0]
            assert request["target"].startswith("/caf%C3%A9/~?existing=%C3%A9%20%2B&blank="), request
            if post:
                expected = ('{"query": "caf\\u00e9 \\u96ea \\ud83d\\ude00 /?&=+%~\\"\\\\\\n\\t'
                            '\\u0000\\u001f\\u007f\\u2028", "k": -1}')
                assert request["body_ascii"] == expected, request
                assert request["content_type"] == "application/json; charset=utf-8", request
                assert request["parameters"] == {"query": query, "k": -1}, request
            else:
                expected_suffix = ("&query=caf%C3%A9+%E9%9B%AA+%F0%9F%98%80+%2F%3F%26%3D%2B%25~"
                                   "%22%5C%0A%09%00%1F%7F%E2%80%A8&k=-1")
                assert request["target"].endswith(expected_suffix), request
                assert request["content_type"] is None and request["body_ascii"] == "", request
                assert request["parameters"]["query"] == [query], request
            result["POST" if post else "GET"] = observed
    constructors = []
    for port in (None, "", 0, "0", 123, -7, "8080/path"):
        client = colbert.ColBERTv2("http://example.invalid/base", port=port)
        expected = "http://example.invalid/base" + (f":{port}" if port else "")
        assert client.url == expected
        constructors.append({"port": port, "url": client.url})
    result["constructors"] = constructors
    return result


def long_text_cases(server):
    observations = {}
    with cache_scope() as cache:
        get = colbert.ColBERTv2(server.url + "/typed-long-text")
        post = colbert.ColBERTv2(server.url + "/typed-long-text", post_requests=True)
        get_rows = get("typed", k=3)
        assert get_rows[0].text is get_rows[0].long_text
        get_rows[0].long_text["tokens"].append(2)
        assert get_rows[0].text == {"tokens": [1, 2]}
        observations["GET"] = invoke(server, get, "typed", 3, simplify=True)
        observations["POST"] = invoke(server, post, "typed", 3, simplify=True)
        assert observations["GET"]["value"] == [{"tokens": [1, 2]}, False, ["nested"]]
        assert observations["POST"]["value"] == [23, None, ["server"]]
        observations["get_text_aliases_long_text"] = True
        observations["events"] = cache.events
    with cache_scope():
        for post in (False, True):
            client = colbert.ColBERTv2(server.url + "/missing-long-text", post_requests=post)
            normal = invoke(server, client, "absent", 1)
            simplified = invoke(server, client, "absent", 1, simplify=True)
            assert "long_text" not in normal["value"][0] if post else normal["value"][0]["long_text"] == "only text"
            if post:
                assert simplified["error"] == "KeyError" and not simplified["requests"], simplified
            else:
                assert simplified["value"] == ["only text"], simplified
            observations["missing_POST" if post else "missing_GET"] = {
                "normal": normal, "simplify": simplified}
    return observations


def error_cases(server):
    cases = {"/http-error": "HTTPError", "/error-message": "ValueError",
             "/error-default": "ValueError", "/error-null-message": "ValueError",
             "/missing-topk": "ValueError", "/bad-json": "JSONDecodeError",
             "/nonobject": "AttributeError", "/topk-null": "TypeError",
             "/row-scalar": "TypeError"}
    observations = []
    for post in (False, True):
        for path, expected_error in cases.items():
            with cache_scope() as cache:
                client = colbert.ColBERTv2(server.url + path, post_requests=post)
                first = invoke(server, client, "errors", 1)
                second = invoke(server, client, "errors", 1)
                assert first["error"] == second["error"] == expected_error, (path, first, second)
                # POST request succeeds and caches its raw sliced list before
                # __call__ fails to turn a scalar row into a dotdict.
                assert len(second["requests"]) == (0 if post and path == "/row-scalar" else 1), second
                observations.append({"path": path, "method": "POST" if post else "GET",
                                     "first": first, "second": second, "events": cache.events})
        with cache_scope(enabled=False):
            truthy = invoke(server, colbert.ColBERTv2(server.url + "/error-false", post_requests=post), "falsy", 1)
            assert truthy["value"][0]["text"] == "alpha", truthy
            missing = invoke(server, colbert.ColBERTv2(server.url + "/row-missing-text", post_requests=post), "no-text", 1)
            if post:
                assert missing["value"][0]["long_text"] == "already supplied", missing
            else:
                assert missing["error"] == "KeyError", missing
            observations.append({"method": "POST" if post else "GET", "falsy_error": truthy,
                                 "missing_text": missing})
    return observations


def cache_alias_cases(server):
    observations = {}
    for post in (False, True):
        with cache_scope() as cache:
            client = colbert.ColBERTv2(server.url + "/rows", post_requests=post)
            independent = colbert.ColBERTv2(server.url + "/rows", post_requests=post)
            before = len(server.receipts)
            first = client("alias", k=2)
            inner_key = cache.cache_key(cache.request_for("inner", post=post))
            outer_key = cache.cache_key(cache.request_for("outer", post=post))
            same_raw = cache.memory_cache[inner_key] is cache.memory_cache[outer_key]
            assert same_raw
            first[0].meta["tags"].extend([f"first-{i}" for i in range(40)])
            first[0].labels.extend(range(40))
            first[0].long_text = "caller top-level"
            first[0].text = "caller text"
            first[0].added = "not cached"
            first[0].pop("score")
            hit = independent("alias", k=2)
            expected_long_text = "server-alpha" if post else "alpha"
            assert hit[0].long_text == expected_long_text and hit[0].text == "alpha"
            assert "added" not in hit[0] and "score" in hit[0]
            assert hit[0].meta["tags"] == ["seed"] + [f"first-{i}" for i in range(40)]
            assert hit[0].labels == ["base"] + list(range(40))
            hit[0].meta["tags"].append("hit-only")
            hit[0].labels.append("hit-only")
            last = client("alias", k=2)
            assert "hit-only" not in last[0].meta["tags"] and "hit-only" not in last[0].labels
            assert len(server.receipts) - before == 1
            observations["POST" if post else "GET"] = {
                "first": copy.deepcopy(first), "last": copy.deepcopy(last),
                "raw_layers_share_object": same_raw, "events": copy.deepcopy(cache.events),
                "requests": copy.deepcopy(server.receipts[before:])}
    return observations


def cache_layer_cases(server):
    result = {}
    with cache_scope() as cache:
        client = colbert.ColBERTv2(server.url + "/rows")
        original = client("promotion", k=1)
        inner_key = cache.cache_key(cache.request_for("inner"))
        outer_key = cache.cache_key(cache.request_for("outer"))
        cache.memory_cache.pop(outer_key)
        cache.events.clear()
        before = len(server.receipts)
        promoted = client("promotion", k=1)
        assert not server.receipts[before:]
        assert [(event["op"], event.get("hit")) for event in cache.events] == [
            ("get", False), ("get", True), ("put", None)]
        promoted[0].meta["tags"].append("promoted")
        original[0].meta["tags"].append("inner-only")
        outer_hit = client("promotion", k=1)
        assert outer_hit[0].meta["tags"] == ["seed", "promoted"]
        cache.memory_cache.pop(outer_key)
        inner_hit = client("promotion", k=1)
        assert inner_hit[0].meta["tags"] == ["seed", "inner-only"]
        result["inner_promotion"] = {"outer_hit": copy.deepcopy(outer_hit),
                                     "inner_hit": copy.deepcopy(inner_hit), "events": cache.events}
    with cache_scope(max_entries=1) as cache:
        client = colbert.ColBERTv2(server.url + "/rows")
        before = len(server.receipts)
        first = client("one-entry", k=1)
        hit = client("one-entry", k=1)
        assert first == hit and len(server.receipts) - before == 1
        assert len(cache.memory_cache) == 1
        result["one_entry"] = {"events": cache.events, "network_calls": 1}
    with cache_scope(max_entries=2) as cache:
        client = colbert.ColBERTv2(server.url + "/rows")
        before = len(server.receipts)
        outstanding = client("evicted", k=1)
        client("replacement", k=1)
        outstanding[0].meta["tags"].append("alive after eviction")
        assert outstanding[0].text == "alpha"
        reloaded = client("evicted", k=1)
        assert reloaded[0].meta["tags"] == ["seed"]
        cache.reset_memory_cache()
        outstanding[0].labels.append("alive after reset")
        assert outstanding[0].labels == ["base", "alive after reset"]
        assert len(server.receipts) - before == 3
        result["eviction"] = {"outstanding": copy.deepcopy(outstanding), "reloaded": copy.deepcopy(reloaded),
                              "network_calls": 3, "events": cache.events}
    with cache_scope(enabled=False) as cache:
        client = colbert.ColBERTv2(server.url + "/rows")
        before = len(server.receipts)
        first = client("disabled", k=1)
        first[0].meta["tags"].append("not cached")
        second = client("disabled", k=1)
        assert second[0].meta["tags"] == ["seed"] and len(server.receipts) - before == 2
        assert not cache.memory_cache
        result["disabled"] = {"network_calls": 2, "events": cache.events}
    try:
        AuditedCache(max_entries=0)
    except ValueError as exc:
        result["zero_capacity"] = error_result(exc, server)
    else:
        raise AssertionError("Actual DSPy accepted zero memory cache capacity")
    return result


def cache_key_and_cap_cases(server):
    result = {}
    with cache_scope() as cache:
        get = colbert.ColBERTv2(server.url + "/rows")
        post = colbert.ColBERTv2(server.url + "/rows", post_requests=True)
        before = len(server.receipts)
        get("keys", k=1)
        get("keys", k=1)
        get("keys", k=2)
        get("different", k=1)
        post("keys", k=1)
        colbert.ColBERTv2(server.url + "/rows?exact=%61")("keys", k=1)
        colbert.ColBERTv2(server.url + "/rows?exact=a")("keys", k=1)
        assert len(server.receipts) - before == 6
        result["distinct_keys"] = {"requests": copy.deepcopy(server.receipts[before:]), "events": cache.events}
    result["prepopulated_cap"] = {}
    for layer in ("outer", "inner"):
        with cache_scope() as cache:
            client = colbert.ColBERTv2(server.url + "/rows")
            client("seed request shape", k=1)
            request = copy.deepcopy(cache.request_for(layer))
            request["positional_arg_1"] = "prepopulated-101"
            request["positional_arg_2"] = 101
            cache.put(request, [{"text": "seeded", "long_text": "seeded"}])
            cache.events.clear()
            observed = invoke(server, client, "prepopulated-101", 101)
            assert observed["value"] == [{"text": "seeded", "long_text": "seeded"}]
            assert not observed["requests"]
            result["prepopulated_cap"][layer] = {**observed, "events": cache.events}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true", help="Serve reusable HTTP response fixtures; do not run DSPy cases")
    parser.add_argument("--port", type=int, default=0, help="Local fixture port; 0 selects an available port")
    args = parser.parse_args()
    assert dspy.__version__ == "3.5.0", dspy.__version__
    if args.serve:
        if args.output:
            parser.error("--output saves source observations and cannot be combined with --serve")
        with Server(args.port) as server:
            print(json.dumps({"url": server.url, "requests_url": server.url + "/__oracle__/requests"}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    if args.port:
        parser.error("--port requires --serve")
    source = Path(inspect.getsourcefile(colbert))
    cache_source = Path(inspect.getsourcefile(Cache))
    server = Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        observations = {
            "dspy": dspy.__version__,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cache_source_sha256": hashlib.sha256(cache_source.read_bytes()).hexdigest(),
            "ordinary": ordinary_cases(server), "encoding": encoding_cases(server),
            "long_text": long_text_cases(server), "errors": error_cases(server),
            "cache_alias": cache_alias_cases(server), "cache_layers": cache_layer_cases(server),
            "cache_keys_and_cap": cache_key_and_cap_cases(server),
        }
        observations["total_http_requests"] = len(server.receipts)
        observations["all_requests"] = server.receipts
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    encoded = json.dumps(observations, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({"output": str(args.output),
                          "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                          "source_sha256": observations["source_sha256"],
                          "total_http_requests": observations["total_http_requests"]}))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
