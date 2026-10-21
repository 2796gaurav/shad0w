"""The live monitor: spot checks feed an anytime-valid lower bound on the served answers' disagreement; when it
rises above alpha the table stops serving ("audit_breach"), and on a table that is as good as certified it does not."""
import numpy as np

import shad0w
from shad0w.audit import BettingUCB
from shad0w.shadow import shadow_compile, write_certificate

from .test_shadow import SCHEMA, synth


def _lower_crosses(rate, alpha, steps, rng, delta=0.1):
    m = BettingUCB(delta)
    for _ in range(steps):
        m.update(1.0 - float(rng.random() < rate))
        if m.n >= 20 and 1.0 - m.ucb() > alpha:
            return True, m.n
    return False, m.n


def test_monitor_rarely_fires_when_the_table_holds():
    rng = np.random.default_rng(0)
    fired = sum(_lower_crosses(0.04, 0.05, 3000, rng)[0] for _ in range(60))
    assert fired / 60 <= 0.1


def test_monitor_fires_quickly_when_the_table_breaks():
    rng = np.random.default_rng(1)
    hits = [_lower_crosses(0.25, 0.05, 3000, rng) for _ in range(20)]
    assert all(h for h, _ in hits) and np.median([n for _, n in hits]) < 200


def test_shadow_stops_serving_after_a_breach(tmp_path):
    rows, _ = synth(3000, 21, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(tmp_path / "b"))
    write_certificate(str(tmp_path / "b"), cert)
    state = {"broken": False}

    def teacher(text):  # the model behind the table changes its mind about everything
        from .mock_llm import classify
        return "refund" if state["broken"] else classify(text)

    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=teacher, question="intent", audit_rate=1.0, seed=0)
    texts = [r["text"] for r in synth(600, 5)[0]]
    served_before = sum(sh(t).source == "table" for t in texts[:200])
    sh.flush()
    assert served_before > 100 and not sh.stats()["audit_breach"]
    state["broken"] = True
    for t in texts[200:]:
        sh(t)
    sh.flush()
    st = sh.stats()
    assert st["audit_breach"] and st["audit_disagreement_lower_anytime"] > 0.05
    d = sh(texts[0])
    assert d.source == "teacher" and d.flag == "audit_breach"
    sh.reload()  # a new (or reloaded) table starts a new watch
    assert not sh.stats()["audit_breach"]


def test_teacher_self_agreement_flags_a_noisy_teacher():
    import random

    from shad0w.audit import teacher_self_agreement
    rows = [{"text": f"message {i}", "intent": "refund"} for i in range(200)]
    steady = teacher_self_agreement(lambda t: "refund", rows, "intent", n=80)
    assert steady["self_agreement"] == 1.0 and not steady["blocks_certificate"]
    rng = random.Random(0)
    noisy = teacher_self_agreement(lambda t: "refund" if rng.random() < 0.7 else "balance", rows, "intent", n=80)
    assert noisy["blocks_certificate"] and noisy["noise_lower"] > 0.05
