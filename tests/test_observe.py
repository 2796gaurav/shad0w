"""Observability and the one-call API: metrics, Prometheus text, hooks, trace, decision(), train(), auto_train, async."""
import asyncio
import json
import time

import pytest

import shad0w
from shad0w.observe import Metrics

from .mock_llm import MockLLM, classify
from .test_shadow import VOCAB, synth


def test_metrics_and_prometheus():
    m = Metrics(cost_per_call=0.002)
    m.set_alpha("intent", 0.05)
    for _ in range(8):
        m.record("intent", "table", "refund", 3e-6, 0.99, None, "x")
    m.record("intent", "teacher", "balance", 0.4, 0.5, "low_confidence", "y", teacher_s=0.4)
    m.record_audit("intent", False)
    s = m.snapshot()
    q = s["questions"]["intent"]
    assert s["decisions"] == 9 and abs(s["offload"] - 8 / 9) < 1e-9 and s["llm_calls_saved"] == 8
    assert abs(s["cost_saved"] - 0.016) < 1e-9 and abs(s["time_saved_s"] - 3.2) < 1e-6
    assert q["flags"] == {"low_confidence": 1} and q["table_p50_us"] == 3.0 and q["llm_p50_ms"] == 400.0
    assert q["audits"] == 1 and 0 < q["audit_disagreement_upper"] < 1
    p = m.prometheus()
    assert 'shad0w_decisions_total{question="intent",source="table"} 8' in p
    assert 'shad0w_latency_seconds_bucket{question="intent",source="table",le="+Inf"} 8' in p
    assert "# TYPE shad0w_latency_seconds histogram" in p and 'shad0w_alpha{question="intent"} 0.05' in p


def test_redacted_text():
    m = Metrics(keep_text=False)
    m.record("q", "table", "a", 1e-6, text="secret")
    assert m.snapshot()["recent"][0]["text"] is None


def test_decision_end_to_end_with_mock_llm(tmp_path):
    """decision() -> log-only -> train() -> table answers, hooks and trace see everything."""
    mock = MockLLM()
    try:
        seen = []
        intent = shad0w.decision("intent", options={k: None for k in VOCAB}, llm="mock-1", base_url=mock.url,
                                 folder=str(tmp_path / "intent"), trace=str(tmp_path / "trace.jsonl"),
                                 on_decision=lambda d, t: seen.append(d.source), audit_rate=0)
        rows, _ = synth(1200, 31, teacher_noise=0.0)
        for r in rows:
            assert intent(r["text"]).source == "teacher"
        assert intent.log_rows() == 1200 and set(seen) == {"teacher"}
        cert = intent.train(min_rows=1000)
        assert cert["certified_share_on_calibration"] > 0.5
        n0 = len(mock.requests)
        d = intent("hi block my card thanks")
        assert d.source == "table" and d.answer == "lost_card" and len(mock.requests) == n0
        trace = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert len(trace) == 1201 and trace[-1]["source"] == "table"
        assert intent.log_rows() == 1200  # table answers never enter the training log
        s = intent.stats()
        assert s["table"] == 1 and s["log_rows"] == 1200 and s["alpha"] == 0.05
        assert (tmp_path / "intent" / "bundle" / "schema.json").exists()
        # a fresh decision() on the same folder picks the trained bundle up with no options needed
        again = shad0w.decision("intent", llm=classify, folder=str(tmp_path / "intent"))
        assert again("hi block my card thanks").source == "table"
        ex = again.explain("zebra lettuce")
        assert ex["certified"] is False and "your model" in ex["why"] and len(ex["top"]) == 3
    finally:
        mock.close()


def test_train_learns_options_from_the_log(tmp_path):
    log = tmp_path / "log.jsonl"
    rows, _ = synth(600, 32, teacher_noise=0.0)
    log.write_text("".join(json.dumps({"text": r["text"], "intent": r["intent"], "source": "teacher"}) + "\n" for r in rows))
    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=classify, question="intent", log=str(log))
    assert sh.model is None
    with pytest.raises(ValueError, match="need 1000"):
        sh.train()
    sh.train(min_rows=500)
    assert sorted(sh.model.questions["intent"].options) == sorted(VOCAB)


def test_auto_train(tmp_path):
    rows, _ = synth(700, 33, teacher_noise=0.0)
    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=classify, question="intent", log=str(tmp_path / "l.jsonl"),
                       schema={"type": "choice", "criteria": {k: None for k in VOCAB}}, auto_train=600, audit_rate=0)
    sh.train = (lambda orig: (lambda **kw: orig(min_rows=500)))(sh.train)
    for r in rows:
        sh.decide(r["text"])
    for _ in range(600):
        if sh.model is not None:
            break
        time.sleep(0.1)
    assert sh.model is not None and sh.decide("hi block my card thanks").source == "table"


def test_async_decide(tmp_path):
    rows, _ = synth(1200, 34, teacher_noise=0.0)
    calls = []

    async def ateacher(text):
        calls.append(text)
        await asyncio.sleep(0)
        return classify(text)

    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=ateacher, question="intent", log=str(tmp_path / "l.jsonl"),
                       schema={"type": "choice", "criteria": {k: None for k in VOCAB}}, audit_rate=0)

    async def run():
        out = [await sh.adecide(r["text"]) for r in rows]
        assert {d.source for d in out} == {"teacher"} and len(calls) == 1200
        await asyncio.to_thread(sh.train)
        d = await sh.adecide("hi block my card thanks")
        assert d.source == "table" and len(calls) == 1200
        sync_t = await shad0w.Shadow(None, teacher=classify, question="intent").adecide("refund me")
        assert sync_t.source == "teacher" and sync_t.answer == "refund"

    asyncio.run(run())


def test_hook_errors_do_not_break_decisions(tmp_path):
    def bad(d, t):
        raise RuntimeError("x")

    sh = shad0w.Shadow(None, teacher=classify, question="intent", on_decision=bad)
    assert sh.decide("refund me").answer == "refund"


def test_llm_latency_default_feeds_time_saved():
    m = Metrics(llm_latency_ms=300)
    m.record("q", "table", "a", 2e-6)
    s = m.snapshot()
    assert s["llm_latency_assumed"] is True and abs(s["time_saved_s"] - 0.3) < 1e-9 and s["llm_mean_ms"] == 300.0
    m.record("q", "teacher", "b", 0.5, teacher_s=0.5)
    s = m.snapshot()
    assert s["llm_latency_assumed"] is False and s["llm_mean_ms"] == 500.0
