"""0.3 cascade behaviour: Decision fields, rollout knobs, uniform audits, honest calibration, the train gate,
peek/record, the decide decorator, the drift override, and the complete= fix."""
import json
import os
from typing import Literal

import pytest

import shad0w
from shad0w.shadow import read_certificate, shadow_compile

from .mock_llm import classify
from .test_shadow import SCHEMA, VOCAB, synth


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    rows, _ = synth(3000, 51, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    path = str(tmp_path_factory.mktemp("policy"))
    m.save(path)
    from shad0w.shadow import write_certificate
    write_certificate(path, cert)
    return path


CERTIFIED = "hi block my card thanks"


def test_decision_fields(bundle):
    sh = shad0w.Shadow(bundle, teacher=classify, audit_rate=0)
    d = sh.decide(CERTIFIED)
    assert d.source == "table" and d.question == "intent" and 0 < d.threshold < 1 and d.top is None
    d = sh.decide(CERTIFIED, probabilities=True)
    assert d.top[0][0] == "lost_card" and abs(sum(p for _, p in d.top) - 1) < 1e-3
    sh2 = shad0w.Shadow(bundle, teacher=classify, audit_rate=0, probabilities=True)
    assert sh2.decide(CERTIFIED).top is not None
    e = sh.explain("zebra lettuce")
    assert e["question"] == "intent" and e["would_serve"] is False and e["mode"] == "serve"


def test_complete_kw_reaches_llm_teacher(tmp_path):
    seen = []
    intent = shad0w.decision("intent", options=list(VOCAB), llm="anything/x", complete=lambda m: (seen.append(m), "refund")[1],
                             folder=str(tmp_path / "i"), audit_rate=0)
    assert intent("money back please").answer == "refund" and seen


@pytest.mark.parametrize("knobs,expect", [
    ({"mode": "shadow"}, "shadow"), ({"mode": "off"}, "off"), ({"canary": 0.0}, "canary"),
    ({"never_serve": ["lost_card"]}, "never_serve"), ({"min_confidence": 1.0}, "min_confidence"),
])
def test_policy_knobs_defer(bundle, knobs, expect):
    sh = shad0w.Shadow(bundle, teacher=classify, audit_rate=0, seed=1, **knobs)
    d = sh.decide(CERTIFIED)
    assert d.source == "teacher" and d.answer == "lost_card" and d.flag == expect and d.certified is False
    if expect == "shadow":
        s = sh.stats()
        assert s["would_serve"] == 1 and s["shadow_disagreements"] == 0 and s["mode"] == "shadow"


def test_force_threshold_serves_uncertified(bundle):
    sh = shad0w.Shadow(bundle, teacher=classify, audit_rate=0, force_threshold=0.0)
    d = sh.decide("zebra quantum lettuce")
    assert d.source == "table" and d.certified is False and d.flag == "manual_threshold"


def test_canary_half_serves_some(bundle):
    sh = shad0w.Shadow(bundle, teacher=classify, audit_rate=0, canary=0.5, seed=3)
    out = [sh.decide(CERTIFIED).source for _ in range(60)]
    assert 5 < out.count("table") < 55


def test_explain_and_adecide_touch_guard_once(bundle):
    import asyncio
    sh = shad0w.Shadow(bundle, teacher=classify, audit_rate=0)
    g = sh.model.questions["intent"].guard
    n0 = len(g.buf)
    sh.explain(CERTIFIED)
    sh.peek(CERTIFIED) and None
    assert len(g.buf) == n0 + 1  # peek observes once, explain never
    asyncio.run(sh.adecide(CERTIFIED))
    assert len(g.buf) == n0 + 2
    asyncio.run(sh.adecide("zebra lettuce"))
    assert len(g.buf) == n0 + 3


def test_uniform_audit_rows_from_deferred_traffic(tmp_path, bundle):
    log = tmp_path / "l.jsonl"
    sh = shad0w.Shadow(None, teacher=classify, question="intent", log=str(log), audit_rate=1.0)
    sh.decide("refund me")
    assert json.loads(log.read_text().splitlines()[0])["source"] == "audit"
    sh = shad0w.Shadow(bundle, teacher=classify, log=str(log), audit_rate=0.0)
    sh.decide("zebra lettuce")
    assert json.loads(log.read_text().splitlines()[-1])["source"] == "teacher"


def test_train_prefers_audit_rows_and_restores_counter(tmp_path):
    rows, _ = synth(1500, 52, teacher_noise=0.0)
    log = tmp_path / "l.jsonl"
    lines = [json.dumps({"text": r["text"], "intent": r["intent"], "source": "audit" if i < 300 else "teacher"}) for i, r in enumerate(rows)]
    log.write_text("\n".join(lines) + "\n")
    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=classify, question="intent", log=str(log),
                       schema={"type": "choice", "criteria": {k: None for k in VOCAB}}, audit_rate=0)
    cert = sh.train()
    assert cert["accepted"] is True and cert["calibration"] == "uniform-audit"
    assert cert["n_calibration"] == 300 and cert["n_fit"] == 1200
    assert read_certificate(str(tmp_path / "b"))["questions"]["intent"]["calibration"] == "uniform-audit"
    sh.decide("zebra lettuce 1")
    sh.decide("zebra lettuce 2")
    again = shad0w.Shadow(str(tmp_path / "b"), teacher=classify, question="intent", log=str(log), auto_train=5000, audit_rate=0)
    assert again._since_train == 2


def test_train_gate_keeps_old_bundle(tmp_path, monkeypatch):
    rows, _ = synth(1200, 53, teacher_noise=0.0)
    log = tmp_path / "l.jsonl"
    log.write_text("".join(json.dumps({"text": r["text"], "intent": r["intent"], "source": "teacher"}) + "\n" for r in rows))
    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=classify, question="intent", log=str(log),
                       schema={"type": "choice", "criteria": {k: None for k in VOCAB}}, audit_rate=0)
    first = sh.train()
    assert first["accepted"] and first["certified_share_on_calibration"] > 0.3
    thr = sh.model.questions["intent"].threshold
    import shad0w.cascade as cascade_mod
    real = cascade_mod.__dict__.get("shadow_compile")  # imported inside train(), so patch the source module

    def worse(schema, records, **kw):
        from shad0w import shadow as sm
        m, cert = sm.__dict__["shadow_compile_orig"](schema, records, **kw)
        for q in cert["questions"].values():
            q["certified_share_on_calibration"] = 0.0
            q["threshold"] = None
        return m, cert

    from shad0w import shadow as sm
    sm.shadow_compile_orig = sm.shadow_compile
    monkeypatch.setattr(sm, "shadow_compile", worse)
    rej = sh.train(gate=True)
    assert rej["accepted"] is False and "kept the old" in rej["reason"]
    assert sh.model.questions["intent"].threshold == thr
    assert read_certificate(str(tmp_path / "b"))["questions"]["intent"]["certified_share_on_calibration"] > 0.3
    ok = sh.train(gate=False)
    assert ok["accepted"] is True
    del sm.shadow_compile_orig
    assert real is None or True


def test_peek_and_record(bundle, tmp_path):
    log = tmp_path / "l.jsonl"
    sh = shad0w.Shadow(bundle, teacher=None, log=str(log), audit_rate=0)
    d = sh.peek(CERTIFIED)
    assert d is not None and d.source == "table" and d.answer == "lost_card"
    assert sh.peek("zebra lettuce") is None
    d = sh.record("zebra lettuce", "balance", teacher_s=0.2)
    assert d.source == "teacher" and d.answer == "balance" and d.flag == "low_confidence"
    row = json.loads(log.read_text().splitlines()[-1])
    assert row["text"] == "zebra lettuce" and row["intent"] == "balance"
    s = sh.stats()
    assert s["table"] == 1 and s["teacher"] == 1 and s["llm_p50_ms"] == 200.0


def test_decide_decorator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @shad0w.decide()
    def route(text) -> Literal["refund", "lost_card", "balance", "transfer"]:
        return classify(text)

    assert route("refund me") == "refund" and route.shadow.question == "route"
    assert os.path.exists(tmp_path / "shad0w" / "route" / "log.jsonl")
    assert route.shadow.schema["criteria"] == {k: None for k in ("refund", "lost_card", "balance", "transfer")}

    @shad0w.decide(name="spam", folder=str(tmp_path / "s"))
    def is_spam(text) -> bool:
        return "free money" in text

    assert is_spam("free money now") is True and is_spam.shadow.schema == {"type": "yesno"}
    with pytest.raises(TypeError):
        @shad0w.decide()
        def bad(text) -> str:
            return text


def test_drift_override_and_flag(bundle):
    sh = shad0w.Shadow(bundle, teacher=classify, audit_rate=0, drift_window=20, drift_margin=0.0)
    g = sh.model.questions["intent"].guard
    assert g.window == 20 and g.margin == 0.0
    for i in range(40):
        sh.decide(f"zz qq ww {i} xx")  # low-confidence traffic pushes the drift estimate up
    assert sh.decide(CERTIFIED).flag == "drift"


def test_decision_reads_env_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAD0W_MODE", "off")
    monkeypatch.setenv("SHAD0W_FOLDER", str(tmp_path / "root"))
    intent = shad0w.decision("intent", options=list(VOCAB), llm=classify)
    assert intent.settings.mode == "off" and intent.log.startswith(str(tmp_path / "root"))
    assert intent("refund me").flag == "no_bundle"
