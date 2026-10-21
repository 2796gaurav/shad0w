"""The proxy on the two decision-model paths with vendor shapes (TypeSafe / Jev on /v1/decisions, llama.cpp / Kev on
/v1/systemone): known question types are served from a certified table, unknown types (box, ...) and non-text
states / inputs go upstream untouched and are never answered from a table."""
import json
import threading
import urllib.request

import pytest

from shad0w.proxy import proxy, state_non_text
from shad0w.shadow import shadow_compile, write_certificate

from .mock_llm import MockLLM
from .test_shadow import SCHEMA, synth

OPTS = ["refund", "lost_card", "balance", "transfer"]
CERTIFIED = "hi block my card thanks"  # served by the table in tests/test_proxy.py too


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pxs1")
    rows, _ = synth(3000, 21, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(folder / "intent" / "bundle"))
    write_certificate(str(folder / "intent" / "bundle"), cert)
    mock = MockLLM()
    srv = proxy(mock.url, port=0, folder=str(folder), audit_rate=0.0, model="mock-1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", mock, folder, srv
    srv.shutdown()
    mock.close()


def call(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _log_lines(folder, q):
    p = folder / q / "log.jsonl"
    return len(p.read_text().splitlines()) if p.exists() else 0


def test_llamacpp_systemone_choice_is_served_from_the_table(setup):
    base, mock, folder, _ = setup
    n0 = len(mock.requests)
    req = {"model": "kev-4b", "state": CERTIFIED,
           "questions": {"intent": {"type": "choice", "instructions": "intent?", "criteria": OPTS}}}  # list-form criteria
    st, h, b = call(base, "/v1/systemone", req)
    out = json.loads(b)
    assert st == 200 and h["x-shad0w-source"] == "table" and len(mock.requests) == n0
    a = out["answers"]["intent"]
    assert a["type"] == "choice" and a["choice"] == "lost_card" and a["shad0w"]["certified"] is True
    assert set(a["probabilities"]) <= set(OPTS) and "latency_ms" in out


def test_noul_without_a_table_goes_upstream_and_is_merged(setup):
    base, mock, folder, _ = setup
    n0 = len(mock.requests)
    req = {"model": "kev-4b", "state": CERTIFIED, "questions": {
        "intent": {"type": "choice", "criteria": {o: o for o in OPTS}},
        "urgent": {"type": "noul", "instructions": "is it urgent?"}}}
    st, h, b = call(base, "/v1/systemone", req)
    out = json.loads(b)
    assert st == 200 and h["x-shad0w-source"] == "mixed" and len(mock.requests) == n0 + 1
    assert list(mock.requests[-1]["body"]["questions"]) == ["urgent"]  # only the pending question went upstream
    assert out["answers"]["intent"]["shad0w"]["source"] == "table"
    assert out["answers"]["urgent"]["type"] == "noul" and out["answers"]["urgent"]["shad0w"]["source"] == "upstream"


@pytest.mark.parametrize("path,req", [
    ("/v1/systemone", {"model": "kev-4b", "state": CERTIFIED, "questions": {
        "intent": {"type": "choice", "criteria": {o: o for o in OPTS}},
        "where": {"type": "box", "instructions": "where is the card?"}}}),
    ("/v1/decisions", {"model": "jev-1", "input": CERTIFIED, "questions": [
        {"type": "choice", "name": "intent", "choices": [{"value": o} for o in OPTS]},
        {"type": "box", "name": "where", "instructions": "where?"}]}),
    ("/v1/systemone", {"model": "kev-4b", "state": CERTIFIED, "questions": {"intent": {"type": "segment", "criteria": OPTS}}}),
])
def test_unknown_question_types_are_forwarded_untouched(setup, path, req):
    """A request with a question type the proxy does not know is relayed as is: the upstream's reply (here the
    mock's 400 for the unknown type) comes back unchanged, nothing is served, nothing is logged."""
    base, mock, folder, srv = setup
    n0, p0 = len(mock.requests), srv.gateway.metrics.passthrough
    lines0 = _log_lines(folder, "intent")
    st, h, b = call(base, path, req)
    assert len(mock.requests) == n0 + 1 and mock.requests[-1]["body"] == req  # forwarded byte-for-byte in content
    assert "x-shad0w-source" not in h and srv.gateway.metrics.passthrough == p0 + 1
    assert st == 400 and b"unknown type" in b  # the mock upstream's own answer, relayed
    assert _log_lines(folder, "intent") == lines0 and not (folder / "where").exists()


@pytest.mark.parametrize("state", [
    {"text": CERTIFIED, "image": "data:image/png;base64,AAAA"},
    {"parts": [{"type": "text", "text": CERTIFIED}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
    [{"type": "input_text", "text": CERTIFIED}, {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
    "data:image/png;base64,AAAA",
    {"screenshot": "....", "prompt": CERTIFIED},
])
def test_non_text_state_is_never_served_from_a_table(setup, state):
    base, mock, folder, srv = setup
    n0, k0 = len(mock.requests), srv.gateway.metrics.non_text
    lines0 = _log_lines(folder, "intent")
    req = {"model": "kev-4b", "state": state, "questions": {"intent": {"type": "choice", "criteria": {o: o for o in OPTS}}}}
    st, h, b = call(base, "/v1/systemone", req)
    assert st == 200 and "x-shad0w-source" not in h and len(mock.requests) == n0 + 1
    assert mock.requests[-1]["body"] == req and srv.gateway.metrics.non_text == k0 + 1
    assert "shad0w" not in json.loads(b)  # the upstream's reply, untouched
    assert _log_lines(folder, "intent") == lines0


def test_typesafe_decisions_with_image_input_is_forwarded(setup):
    base, mock, folder, srv = setup
    n0, k0 = len(mock.requests), srv.gateway.metrics.non_text
    req = {"model": "jev-1", "input": [{"role": "user", "content": [{"type": "input_text", "text": CERTIFIED},
                                                                   {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}],
           "questions": [{"type": "choice", "name": "intent", "choices": [{"value": o} for o in OPTS]}]}
    st, h, b = call(base, "/v1/decisions", req)
    assert "x-shad0w-source" not in h and len(mock.requests) == n0 + 1 and srv.gateway.metrics.non_text == k0 + 1
    # the same text alone is still served
    st, h, b = call(base, "/v1/decisions", {**req, "input": CERTIFIED})
    assert h["x-shad0w-source"] == "table" and json.loads(b)["answers"][0]["choice"] == "lost_card"


def test_structured_text_state_is_still_a_decision(setup):
    """A dict state without any image / audio / file part is text (JSON-dumped), as before."""
    base, mock, folder, srv = setup
    req = {"model": "kev-4b", "state": {"customer": "c1", "message": "zebra quantum lettuce"},
           "questions": {"intent": {"type": "choice", "criteria": {o: o for o in OPTS}}}}
    st, h, b = call(base, "/v1/systemone", req)
    assert st == 200 and h["x-shad0w-source"] in ("teacher", "table")


def test_state_non_text_rules():
    assert not state_non_text({"state": "plain text"})
    assert not state_non_text({"state": {"a": 1, "b": ["x", {"type": "text", "text": "hi"}]}})
    assert not state_non_text({"state": "data:text/plain;base64,aGk="})
    assert state_non_text({"state": "data:image/png;base64,AAAA"})
    assert state_non_text({"state": {"nested": {"deeper": [{"audio": "..."}]}}})
    assert state_non_text({"state": [{"type": "file", "file_id": "f1"}]})
    assert not state_non_text({"input": "no state key"}) and not state_non_text(None)
