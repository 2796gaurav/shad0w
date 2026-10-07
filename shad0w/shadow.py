"""Shadow mode: learn a decision from the answers of a model you already run (the "teacher"), and certify
how often the compiled table disagrees with that teacher. No human labels.

    records = [{"text": "...", "intent": "lost_card"}, ...]     # the teacher's answers, logged in shadow
    model, cert = shadow_compile(schema, records, alpha=0.05, teacher="my-llm-v3")
    model.save("bundle/"); write_certificate("bundle/", cert)

The certificate (Learn-then-Test, fixed-sequence over confidence thresholds, Clopper-Pearson, level delta):
    on inputs drawn like the calibration slice, among answers the table serves itself (certified=True),
    the rate of disagreement with the teacher is at most alpha, with probability at least 1 - delta.
It is a bound against the TEACHER, not against the truth, and it is marginal over the calibration distribution:
a shift in the mix of traffic can break it, which is what audits (shad0w.audit) and re-certification are for.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

import numpy as np

from .api import DriftGuard, Model, Question, load
from .features import items
from .mathutil import softmax

CERT_FILE = "certificate.json"


def _hash_records(texts, labels) -> str:
    h = hashlib.sha256()
    for t, y in zip(texts, labels):
        h.update(str(t).encode("utf-8")); h.update(b"\x00"); h.update(str(y).encode("utf-8")); h.update(b"\x01")
    return h.hexdigest()


def _encode_labels(qtype, options, ys):
    if qtype == "choice":
        idx = {o: i for i, o in enumerate(options)}
        unknown = sorted({str(v) for v in ys if v not in idx})
        if unknown:
            raise ValueError(f"teacher answers not in the schema options: {unknown[:5]}")
        return np.array([idx[v] for v in ys])
    return np.array([int(bool(v)) for v in ys])


def _certify(q: Question, texts, y, alpha: float, delta: float):
    """Fit the certified threshold on teacher-labelled calibration texts (never used for fitting)."""
    from .reliability import SelectiveRiskController
    P = np.stack([softmax(q.rx.logits(items(t))) for t in texts])
    conf, agree = P.max(1), P.argmax(1) == y
    thr = float(SelectiveRiskController(alpha, delta).fit(conf, agree).threshold)
    served = conf >= thr
    q.threshold, q.alpha, q.calibrated = thr, alpha, True
    q.guard = DriftGuard(conf, agree)
    return {
        "n_calibration": int(len(y)),
        "agreement_with_teacher": float(agree.mean()),
        "threshold": thr if np.isfinite(thr) else None,
        "certified_share_on_calibration": float(served.mean()),
        "disagreement_on_certified_calibration": float((~agree[served]).mean()) if served.any() else None,
    }


def shadow_compile(schema: dict, records: list[dict], alpha: float = 0.05, delta: float = 0.1,
                   cal_fraction: float = 0.3, max_cal: int = 3000, teacher: str = "unspecified",
                   seed: int = 0, r_min: float = 8.0):
    """Compile every question in `schema` from the teacher's logged answers in `records` and certify it.
    Returns (Model, certificate dict). Each question needs at least ~1,000 records for a useful certificate."""
    from .compiler.distill import compile_schema

    rng = np.random.default_rng(seed)
    model = Model(meta={"mode": "shadow", "teacher": teacher, "alpha": alpha, "delta": delta})
    cert = {"format": "shad0w-certificate/1", "mode": "shadow", "certified_against": "teacher", "teacher": teacher,
            "alpha": alpha, "delta": delta,
            "statement": ("On inputs drawn like the calibration slice, among answers this bundle serves with "
                          "certified=true, the rate of disagreement with the teacher is at most alpha with "
                          "probability at least 1 - delta. The bound is against the teacher, not the truth, and "
                          "does not survive a shift in the traffic mix without re-certification."),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "questions": {}}
    for name, spec in schema.items():
        qtype = spec.get("type", "choice")
        options = list(spec["criteria"].keys()) if qtype == "choice" else ["no", "yes"]
        rows = [r for r in records if name in r and r.get("text")]
        if len(rows) < 100:
            raise ValueError(f"question {name!r}: {len(rows)} teacher records; need at least 100 (1,000+ recommended)")
        texts = [r["text"] for r in rows]
        y = _encode_labels(qtype, options, [r[name] for r in rows])
        perm = rng.permutation(len(texts))
        n_cal = int(min(max_cal, max(100, cal_fraction * len(texts))))
        cal, fit = perm[:n_cal], perm[n_cal:]
        comp = compile_schema([texts[i] for i in fit], y[fit], options)
        q = Question(name, qtype, options, comp.rx, float("inf"), alpha, DriftGuard(np.ones(2), np.ones(2, bool)), r_min, False)
        stats = _certify(q, [texts[i] for i in cal], y[cal], alpha, delta)
        model.questions[name] = q
        cert["questions"][name] = {"type": qtype, "options": options, "n_records": len(texts), "n_fit": int(len(fit)),
                                   "data_sha256": _hash_records(texts, [r[name] for r in rows]), **stats}
    return model, cert


def certify_bundle(bundle: str, records: list[dict], alpha: float | None = None, delta: float = 0.1,
                   teacher: str | None = None):
    """Re-certify an existing bundle on FRESH teacher-labelled records (e.g. the latest week of shadow traffic).
    The records must not have been used to compile the bundle."""
    m = load(bundle, native=False)
    old = read_certificate(bundle) or {}
    alpha = alpha if alpha is not None else old.get("alpha", m.meta.get("alpha", 0.05))
    cert = {**{k: v for k, v in old.items() if k != "questions"}, "alpha": alpha, "delta": delta,
            "teacher": teacher or old.get("teacher", m.meta.get("teacher", "unspecified")),
            "recertified_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "questions": {}}
    cert.setdefault("format", "shad0w-certificate/1"); cert.setdefault("certified_against", "teacher")
    for name, q in m.questions.items():
        rows = [r for r in records if name in r and r.get("text")]
        if not rows:
            continue
        texts = [r["text"] for r in rows]
        y = _encode_labels(q.qtype, q.options, [r[name] for r in rows])
        cert["questions"][name] = {"type": q.qtype, "options": q.options, "n_records": len(texts),
                                   "data_sha256": _hash_records(texts, [r[name] for r in rows]),
                                   **_certify(q, texts, y, alpha, delta)}
    m.save(bundle)
    write_certificate(bundle, cert)
    return cert


def write_certificate(bundle: str, cert: dict):
    os.makedirs(bundle, exist_ok=True)
    with open(os.path.join(bundle, CERT_FILE), "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)


def read_certificate(bundle: str):
    p = os.path.join(bundle, CERT_FILE)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
