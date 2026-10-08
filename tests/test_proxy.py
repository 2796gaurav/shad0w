"""The OpenAI-compatible proxy: table answers, upstream fallback with logging, pass-through, streaming, metrics,
tools, yes/no, the Decisions API and System One paths (no marking), the playground, and settings."""
import json
import threading
import urllib.request

import pytest

from shad0w.proxy import proxy
from shad0w.shadow import shadow_compile, write_certificate

from .mock_llm import MockLLM
from .test_shadow import SCHEMA, synth

OPTS = ["refund", "lost_card", "balance", "transfer"]


def start(folder, mock, **kw):
    srv = proxy(mock.url, port=0, folder=str(folder), audit_rate=0.0, model="mock-1", **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("px")
    rows, _ = synth(3000, 21, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(folder / "intent" / "bundle"))
    write_certificate(str(folder / "intent" / "bundle"), cert)
    mock = MockLLM()
    base, srv = start(folder, mock, cost_per_call=0.001, capture=["header", "model", "tools", "json_schema", "decisions"],
                      schema={"spam": {"type": "yesno"}})
    yield base, mock, folder, srv
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
    assert st == 200 and h["x-shad0w-source"] == "table" and h["x-shad0w-certified"] == "1" and len(mock.requests) == n0
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
        "label": {"type": "string", "enum": OPTS}}}}}
    st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks", response_format=rf))
    assert h["x-shad0w-source"] == "table"
    assert json.loads(json.loads(b)["choices"][0]["message"]["content"]) == {"label": "lost_card"}


def test_unmarked_json_schema_passes_through_by_default(tmp_path):
    mock = MockLLM()
    base, srv = start(tmp_path, mock)
    try:
        rf = {"type": "json_schema", "json_schema": {"name": "topic", "schema": {"type": "object", "properties": {
            "answer": {"type": "string", "enum": OPTS}}}}}
        st, h, b = call(base, "/v1/chat/completions", chat("refund please", response_format=rf))
        assert st == 200 and "x-shad0w-source" not in h and len(mock.requests) == 1
        assert not (tmp_path / "topic").exists()
        assert srv.gateway.capture == {"header", "model", "tools", "decisions"}
    finally:
        srv.shutdown()
        mock.close()


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
        "answer": {"type": "string", "enum": OPTS}}}}}
    st, h, b = call(base, "/v1/chat/completions", chat("refund please", response_format=rf))
    assert h["x-shad0w-source"] == "teacher"
    assert json.loads((folder / "topic" / "log.jsonl").read_text().splitlines()[0])["topic"] == "refund"


def test_options_header_sets_options(setup):
    base, mock, folder, _ = setup
    st, h, b = call(base, "/v1/chat/completions", chat("free money click here"),
                    {"X-Shad0w-Question": "promo", "X-Shad0w-Options": "refund, lost_card, balance, transfer"})
    assert h["x-shad0w-source"] == "teacher"
    assert json.loads((folder / "promo" / "log.jsonl").read_text().splitlines()[-1])["promo"] == "balance"


def test_yesno_question_logs_bools(setup):
    base, mock, folder, _ = setup
    def spam(text):
        return {"model": "mock-1", "messages": [{"role": "system", "content": "Is this spam? yes or no"},
                                                {"role": "user", "content": text}]}
    st, h, b = call(base, "/v1/chat/completions", spam("free money click here"), {"X-Shad0w-Question": "spam"})
    assert st == 200 and h["x-shad0w-source"] == "teacher"
    st, h, b = call(base, "/v1/chat/completions", spam("what is my balance"), {"X-Shad0w-Question": "spam"})
    rows = [json.loads(l) for l in (folder / "spam" / "log.jsonl").read_text().splitlines()]
    assert rows[-2]["spam"] is True and rows[-1]["spam"] is False


def test_tools_request_returns_tool_calls(setup):
    base, mock, folder, _ = setup
    tools = [{"type": "function", "function": {"name": "intent", "parameters": {"type": "object", "properties": {
        "intent": {"type": "string", "enum": OPTS}}, "required": ["intent"]}}}]
    tc = {"type": "function", "function": {"name": "intent"}}
    n0 = len(mock.requests)
    st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks", tools=tools, tool_choice=tc))
    out = json.loads(b)
    assert h["x-shad0w-source"] == "table" and len(mock.requests) == n0 and h["x-shad0w-question"] == "intent"
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls" and msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "intent"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"intent": "lost_card"}
    st, h, b = call(base, "/v1/chat/completions", chat("zebra quantum lettuce", tools=tools, tool_choice=tc))
    assert h["x-shad0w-source"] == "teacher" and json.loads(b)["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads((folder / "intent" / "log.jsonl").read_text().splitlines()[-1])["intent"] == "balance"
    st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks", tools=tools, tool_choice=tc, stream=True))
    assert b'"finish_reason": "tool_calls"' in b and b"lost_card" in b
    st, h, b = call(base, "/v1/chat/completions", chat("zebra quantum lettuce", tools=tools, tool_choice=tc, stream=True))
    assert st == 200 and b"[DONE]" in b


DECQ = [{"type": "choice", "name": "intent", "instructions": "Which intent?",
         "choices": [{"value": o, "description": None} for o in OPTS]},
        {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
        {"type": "score", "name": "severity", "instructions": "How bad?", "levels": [{"label": "low"}, {"label": "high"}]}]


def test_decisions_endpoint_zero_code(setup):
    base, mock, folder, _ = setup
    n0 = len(mock.requests)
    st, h, b = call(base, "/v1/decisions", {"model": "gpt-6-luna", "input": "hi block my card thanks", "questions": DECQ})
    out = json.loads(b)
    assert st == 200 and h["x-shad0w-source"] == "mixed" and out["object"] == "decision"
    assert [a["name"] for a in out["answers"]] == ["intent", "urgent", "severity"]
    a = out["answers"][0]
    assert a["type"] == "choice" and a["choice"] == "lost_card" and a["shad0w"]["source"] == "table" and a["shad0w"]["certified"] is True
    assert abs(sum(p["probability"] for p in a["probabilities"]) - 1) < 1e-3 and 0 < a["confidence"] <= 1
    assert out["answers"][1]["type"] == "predicate" and out["answers"][1]["shad0w"]["source"] == "upstream"
    assert out["answers"][2]["type"] == "score" and out["answers"][2]["shad0w"]["source"] == "upstream"
    assert len(mock.requests) == n0 + 1  # only the unanswered questions went upstream
    sent = mock.requests[-1]["body"]
    assert [q["name"] for q in sent["questions"]] == ["urgent", "severity"] and sent["input"] == "hi block my card thanks"
    assert json.loads((folder / "urgent" / "log.jsonl").read_text().splitlines()[-1])["urgent"] is False
    st, h, b = call(base, "/v1/decisions", {"model": "gpt-6-luna", "input": "zebra quantum lettuce", "questions": DECQ[:1]})
    out = json.loads(b)
    assert h["x-shad0w-source"] == "teacher" and out["answers"][0]["choice"] == "balance" and out["answers"][0]["shad0w"]["source"] == "upstream"
    assert json.loads((folder / "intent" / "log.jsonl").read_text().splitlines()[-1])["intent"] == "balance"
    st, h, b = call(base, "/v1/decisions", {"model": "gpt-6-luna", "input": "boom", "questions": DECQ[:1]})
    assert st == 401
    st, h, b = call(base, "/v1/decisions", {"model": "gpt-6-luna", "input": "hi", "questions": [{"type": "nope"}]})
    assert st == 400


def test_systemone_endpoint(setup):
    base, mock, folder, _ = setup
    req = {"model": "kev-4b", "state": "hi block my card thanks", "questions": {
        "intent": {"type": "choice", "instructions": "?", "criteria": {o: o for o in OPTS}},
        "urgent": {"type": "noul", "instructions": "urgent?"}}}
    n0 = len(mock.requests)
    st, h, b = call(base, "/v1/systemone", req)
    out = json.loads(b)
    assert st == 200 and out["answers"]["intent"]["choice"] == "lost_card" and out["answers"]["intent"]["shad0w"]["source"] == "table"
    assert out["answers"]["urgent"]["type"] == "noul" and "latency_ms" in out and len(mock.requests) == n0 + 1
    assert list(mock.requests[-1]["body"]["questions"]) == ["urgent"]


def test_playground(setup):
    base, mock, *_ = setup
    st, h, b = call(base, "/v1/playground", {"question": "intent", "text": "hi block my card thanks", "ask_llm": True})
    out = json.loads(b)
    assert st == 200 and out["explain"]["answer"] == "lost_card" and out["explain"]["would_serve"] is True
    assert out["llm"]["answer"] == "lost_card" and out["llm"]["model"] == "mock-1"
    st, h, b = call(base, "/", None)
    assert b"Playground" in b


def test_gateway_settings_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAD0W_CANARY", "0")
    rows, _ = synth(2000, 22, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(tmp_path / "intent" / "bundle"))
    mock = MockLLM()
    base, srv = start(tmp_path, mock)
    try:
        assert srv.gateway.settings.canary == 0.0
        st, h, b = call(base, "/v1/chat/completions", chat("hi block my card thanks"), {"X-Shad0w-Question": "intent"})
        assert h["x-shad0w-source"] == "teacher"
        st, _, b = call(base, "/v1/stats")
        assert json.loads(b)["settings"]["intent"]["canary"] == 0.0
    finally:
        srv.shutdown()
        mock.close()


def test_trace_file_receives_every_decision(tmp_path):
    mock = MockLLM()
    base, srv = start(tmp_path, mock, trace=str(tmp_path / "trace.jsonl"))
    try:
        call(base, "/v1/chat/completions", chat("refund me"), {"X-Shad0w-Question": "intent"})
        call(base, "/v1/chat/completions", chat("refund me again"), {"X-Shad0w-Question": "intent"})
        srv.gateway._trace._f.flush()
        assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2
    finally:
        srv.shutdown()
        mock.close()


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
