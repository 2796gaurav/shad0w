"""The OpenAI-compatible proxy: table answers, upstream fallback with logging, pass-through, streaming, metrics."""
import json
import threading
import urllib.request

import pytest

from shad0w.proxy import proxy
from shad0w.shadow import shadow_compile, write_certificate

from .mock_llm import MockLLM
from .test_shadow import SCHEMA, synth


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("px")
    rows, _ = synth(3000, 21, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(folder / "intent" / "bundle"))
    write_certificate(str(folder / "intent" / "bundle"), cert)
    mock = MockLLM()
    srv = proxy(mock.url, port=0, folder=str(folder), audit_rate=0.0, model="mock-1", cost_per_call=0.001)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", mock, folder, srv
    srv.shutdown()
    mock.close()


def call(base, path, body=None, headers=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def chat(text, **kw):
    return {"model": kw.pop("model", "mock-1"), "messages": [{"role": "system", "content": "classify"},
                                                             {"role": "user", "content": text}], **kw}


def test_certified_decision_is_answered_by_the_table(setup):
    base, mock, *_ = setup
    n0 = len(mock.requests)
    st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks"), {"X-Shad0w-Question": "intent"})
    out = json.loads(b)
    assert st == 200 and h["x-shad0w-source"] == "table" and len(mock.requests) == n0
    assert out["object"] == "chat.completion" and out["choices"][0]["message"]["content"] == "lost_card"
    assert out["shad0w"]["certified"] is True


def test_uncertain_decision_goes_upstream_and_is_logged(setup):
    base, mock, folder, _ = setup
    n0 = len(mock.requests)
    st, h, b = call(base, "/v1/chat/completions", chat("zebra quantum lettuce"), {"X-Shad0w-Question": "intent"})
    assert st == 200 and h["x-shad0w-source"] == "teacher" and len(mock.requests) == n0 + 1
    assert json.loads(b)["choices"][0]["message"]["content"] == "balance"  # the upstream reply, untouched
    rows = [json.loads(l) for l in (folder / "intent" / "log.jsonl").read_text().splitlines()]
    assert rows[-1]["text"] == "zebra quantum lettuce" and rows[-1]["intent"] == "balance"


def test_model_prefix_routes_and_rewrites_model(setup):
    base, mock, *_ = setup
    st, h, b = call(base, "/v1/chat/completions", chat("zebra quantum lettuce", model="shad0w/intent"))
    assert st == 200 and mock.requests[-1]["body"]["model"] == "mock-1"


def test_json_schema_reply_shape(setup):
    base, *_ = setup
    rf = {"type": "json_schema", "json_schema": {"name": "intent", "schema": {"type": "object", "properties": {
        "label": {"type": "string", "enum": ["refund", "lost_card", "balance", "transfer"]}}}}}
    st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks", response_format=rf))
    assert h["x-shad0w-source"] == "table"
    assert json.loads(json.loads(b)["choices"][0]["message"]["content"]) == {"label": "lost_card"}


def test_streaming_from_table_and_upstream(setup):
    base, *_ = setup
    st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks", stream=True), {"X-Shad0w-Question": "intent"})
    assert st == 200 and "text/event-stream" in h["content-type"] and b"lost_card" in b and b.endswith(b"data: [DONE]\n\n")
    st, h, b = call(base, "/v1/chat/completions", chat("zebra quantum", stream=True), {"X-Shad0w-Question": "intent"})
    assert st == 200 and b"[DONE]" in b and b"bal" in b


def test_non_decisions_pass_through(setup):
    base, mock, *_ = setup
    n0 = len(mock.requests)
    st, _, b = call(base, "/v1/chat/completions", chat("write me a poem"))
    assert st == 200 and len(mock.requests) == n0 + 1
    st, _, b = call(base, "/v1/models")
    assert st == 200 and json.loads(b)["data"][0]["id"] == "mock-1"


def test_upstream_errors_are_relayed(setup):
    base, *_ = setup
    st, _, b = call(base, "/v1/chat/completions", chat("boom zebra"), {"X-Shad0w-Question": "intent"})
    assert st == 401 and b"bad key" in b


def test_new_question_starts_log_only(setup):
    base, mock, folder, _ = setup
    rf = {"type": "json_schema", "json_schema": {"name": "topic", "schema": {"type": "object", "properties": {
        "answer": {"type": "string", "enum": ["refund", "lost_card", "balance", "transfer"]}}}}}
    st, h, b = call(base, "/v1/chat/completions", chat("refund please", response_format=rf))
    assert h["x-shad0w-source"] == "teacher"
    assert json.loads((folder / "topic" / "log.jsonl").read_text().splitlines()[0])["topic"] == "refund"


def test_observability_endpoints(setup):
    base, *_ = setup
    call(base, "/v1/chat/completions", chat("hi block my card thanks"), {"X-Shad0w-Question": "intent"})
    st, _, b = call(base, "/v1/stats")
    s = json.loads(b)
    assert st == 200 and s["mode"] == "proxy" and s["questions"]["intent"]["table"] >= 1 and s["cost_saved"] > 0
    st, h, b = call(base, "/metrics")
    text = b.decode()
    assert 'shad0w_decisions_total{question="intent",source="table"}' in text and "shad0w_latency_seconds_bucket" in text
    st, h, b = call(base, "/")
    assert st == 200 and b"<title>shad0w" in b and h["content-type"].startswith("text/html")
