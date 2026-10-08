"""Shadow mode: compile from a teacher's answers, certify agreement, re-certify, and check the bound empirically."""
import json
import os
import random
import subprocess
import sys

import shad0w
from shad0w.shadow import CERT_FILE, certify_bundle, read_certificate, shadow_compile, write_certificate

VOCAB = {
    "refund": ["refund", "money back", "return my payment", "reimburse", "charged twice"],
    "lost_card": ["lost my card", "card stolen", "cannot find my card", "block my card", "card missing"],
    "balance": ["balance", "how much money", "account total", "funds available", "what do i have"],
    "transfer": ["transfer", "send money", "wire to", "move funds", "pay my friend"],
}
FILLER = ["please", "today", "asap", "hi", "thanks", "i need help", "urgent", "my account", "again", "now"]
SCHEMA = {"intent": {"type": "choice", "criteria": {k: None for k in VOCAB}}}


def synth(n, seed, teacher_noise=0.1):
    """Synthetic traffic: text from an intent's vocabulary; the 'teacher' answers correctly 90% of the time."""
    rng = random.Random(seed)
    keys = list(VOCAB)
    rows, gold = [], []
    for _ in range(n):
        y = rng.choice(keys)
        text = " ".join([rng.choice(FILLER), rng.choice(VOCAB[y]), rng.choice(FILLER)])
        t = y if rng.random() > teacher_noise else rng.choice(keys)
        rows.append({"text": text, "intent": t})
        gold.append(y)
    return rows, gold


def test_shadow_compile_and_certificate(tmp_path):
    rows, _ = synth(3000, 0)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05, teacher="synthetic-teacher")
    q = cert["questions"]["intent"]
    assert cert["certified_against"] == "teacher" and cert["teacher"] == "synthetic-teacher"
    assert q["n_calibration"] == 900 and q["n_fit"] == 2100
    assert len(q["data_sha256"]) == 64
    assert 0.0 <= q["certified_share_on_calibration"] <= 1.0
    out = str(tmp_path / "bundle")
    m.save(out)
    write_certificate(out, cert)
    assert os.path.exists(os.path.join(out, CERT_FILE))
    m2 = shad0w.load(out, native=False)
    r = m2.decide("hi block my card thanks")["answers"]["intent"]
    assert r["choice"] in VOCAB and isinstance(r["certified"], bool)


def test_bound_holds_on_fresh_teacher_traffic():
    """Empirical check of the guarantee: on fresh traffic from the same distribution, disagreement with the
    teacher among certified answers stays at or under alpha (allowing the delta=0.1 failure probability)."""
    rows, _ = synth(4000, 1, teacher_noise=0.02)
    alpha = 0.05
    m, cert = shadow_compile(SCHEMA, rows, alpha=alpha)
    fresh, _ = synth(4000, 2, teacher_noise=0.02)
    served = dis = 0
    for r in fresh:
        a = m.decide(r["text"])["answers"]["intent"]
        if a["certified"]:
            served += 1
            dis += a["choice"] != r["intent"]
    assert served > 0
    assert dis / served <= alpha + 0.02


def test_refuses_to_certify_when_bound_is_unreachable():
    """A teacher that flips 10% of answers at random (independent of the input) disagrees with ANY student on
    ~7.5% of confident inputs, so a 5% bound is unreachable: the certificate must serve nothing."""
    rows, _ = synth(4000, 7, teacher_noise=0.10)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05)
    assert cert["questions"]["intent"]["certified_share_on_calibration"] == 0.0
    assert not m.decide("hi block my card thanks")["answers"]["intent"]["certified"]


def test_recertify_on_fresh_traffic(tmp_path):
    rows, _ = synth(2500, 3)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05, teacher="t1")
    out = str(tmp_path / "b")
    m.save(out)
    write_certificate(out, cert)
    fresh, _ = synth(1500, 4)
    c2 = certify_bundle(out, fresh, alpha=0.02)
    assert c2["alpha"] == 0.02 and "recertified_utc" in c2
    assert read_certificate(out)["questions"]["intent"]["n_records"] == 1500


def test_unknown_teacher_answer_is_rejected():
    rows, _ = synth(300, 5)
    rows[0]["intent"] = "not_an_option"
    try:
        shadow_compile(SCHEMA, rows)
    except ValueError as e:
        assert "not_an_option" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_cli_shadow_and_report(tmp_path):
    rows, _ = synth(2000, 6)
    data = tmp_path / "teacher.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA))
    out = tmp_path / "bundle"
    cmd = [sys.executable, "-m", "shad0w", "shadow", "--schema", str(schema), "--data", str(data), "--teacher", "cli-test",
           "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert r.returncode == 0, r.stderr
    assert "certified share" in r.stdout
    r = subprocess.run([sys.executable, "-m", "shad0w", "report", "--bundle", str(out)], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(__file__)))
    assert '"certified_against": "teacher"' in r.stdout


def test_cal_records_are_never_fit_on():
    rows, _ = synth(2000, 7, teacher_noise=0.01)
    cal, _ = synth(400, 8, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05, cal_records=cal)
    q = cert["questions"]["intent"]
    assert q["n_fit"] == 2000 and q["n_calibration"] == 400 and q["calibration"] == "uniform-audit"
    m2, cert2 = shadow_compile(SCHEMA, rows, alpha=0.05)
    assert cert2["questions"]["intent"]["calibration"] == "held-out-split"
    try:
        shadow_compile(SCHEMA, rows, alpha=0.05, cal_records=cal[:50])
        raise AssertionError("expected a ValueError for too few calibration records")
    except ValueError:
        pass


def test_yesno_string_labels():
    from shad0w.shadow import _encode_labels
    enc = _encode_labels("yesno", ["no", "yes"], ["no", "false", False, "yes", True, "1", "No", "YES"])
    assert enc.tolist() == [0, 0, 0, 1, 1, 1, 0, 1]
    rows = [{"text": f"free money click here {i}" if i % 2 else f"hello there friend {i}", "spam": "yes" if i % 2 else "no"}
            for i in range(400)]
    m, cert = shadow_compile({"spam": {"type": "yesno"}}, rows, alpha=0.1)
    assert m.decide("free money click here now")["answers"]["spam"]["answer"] is True
    assert m.decide("hello there friend today")["answers"]["spam"]["answer"] is False
