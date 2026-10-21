"""The unseen-item share (how much of an input the table has never seen) and distil(): teacher labels on a corpus
you supply, used for fitting only."""
import json

import numpy as np

import shad0w
from shad0w.shadow import read_jsonl, shadow_compile

from .test_shadow import SCHEMA, VOCAB, synth


def test_unseen_share_is_reported_and_agrees_between_python_and_c(tmp_path):
    rows, _ = synth(1500, 3, teacher_noise=0.0)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05, student="linear")
    out = str(tmp_path / "b")
    m.save(out)
    py = shad0w.load(out, native=False)
    seen = py.decide("please block my card today")["answers"]["intent"]
    junk = py.decide("xqzv plortk vemblu qwxxz")["answers"]["intent"]
    assert 0.0 <= seen["unseen"] < 0.5 < junk["unseen"] <= 1.0
    from shad0w.native import available
    if available():
        nat = shad0w.load(out, native=True)
        assert nat.native
        for t in ("please block my card today", "xqzv plortk vemblu qwxxz", "", "refund"):
            a, b = py.decide(t)["answers"]["intent"], nat.decide(t)["answers"]["intent"]
            assert abs(a["unseen"] - b["unseen"]) < 1e-6 and a["choice"] == b["choice"]


def test_max_unseen_defers_junk(tmp_path):
    rows, _ = synth(1500, 4, teacher_noise=0.0)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05, student="linear")
    out = str(tmp_path / "b")
    m.save(out)
    shad0w.shadow.write_certificate(out, cert)
    calls = []

    def teacher(t):
        calls.append(t)
        return "balance"
    plain = shad0w.Shadow(out, teacher=teacher, question="intent")  # the default: off
    assert plain.peek("please block my card today").flag is None
    # a certified answer whose input carries injected junk: still certified by default, deferred under max_unseen
    probe = next(t for t in ("block my card please xq zv", "block my card please xqzv plortk", "block my card xqzv plortk vemb")
                 if plain.peek(t) is not None and plain.peek(t).flag is None
                 and m.decide(t)["answers"]["intent"]["unseen"] > 0.2)
    strict = shad0w.Shadow(out, teacher=teacher, question="intent", max_unseen=0.2)
    d = strict(probe)
    assert d.source == "teacher" and d.flag == "unseen_items" and calls == [probe]
    assert "never seen" in d.why


def test_distil_rows_fit_only(tmp_path):
    from shad0w.llm import LLMTeacher

    def complete(msgs):
        text = msgs[-1]["content"]
        for opt, words in VOCAB.items():
            if any(w in text for w in words):
                return opt
        return "balance"
    t = LLMTeacher(SCHEMA, "openai/x", question="intent", api_key="sk-test", complete=complete)
    sh = shad0w.Shadow(None, teacher=t, question="intent", log=str(tmp_path / "l.jsonl"), schema=SCHEMA["intent"])
    corpus = [r["text"] for r in synth(40, 11)[0]] + ["", "   "]
    rep = sh.distil(corpus, concurrency=4)
    assert rep == {"asked": 40, "logged": 40, "failed": 0}
    rows = read_jsonl(str(tmp_path / "l.jsonl"))
    assert len(rows) == 40 and all(r["source"] == "distil" and r["intent"] in VOCAB for r in rows)
    real, _ = synth(1200, 12, teacher_noise=0.01)
    with open(tmp_path / "l.jsonl", "a") as f:
        for r in real:
            f.write(json.dumps({**r, "source": "teacher"}) + "\n")
    cert = sh.train(out=str(tmp_path / "b"), min_rows=100)
    assert cert["n_fit_only"] == 40 and cert["n_records"] == 1200  # distilled rows never reach calibration
    assert cert["student"] in ("linear", "deep") and "student_choice" in cert
    assert np.isfinite(cert["threshold"]) if cert["threshold"] is not None else True


def test_majority_vote_calibration_is_recorded():
    rows, _ = synth(1500, 13, teacher_noise=0.0)
    cal = [{**r, "teacher_votes": 3} for r in synth(400, 14, teacher_noise=0.0)[0]]
    _, cert = shadow_compile(SCHEMA, rows, alpha=0.05, cal_records=cal, student="linear")
    assert cert["questions"]["intent"]["calibration"] == "uniform-audit (majority of 3 teacher votes)"
