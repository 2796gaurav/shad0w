"""Shadow mode: learn a decision from the answers of a model you already run (the "teacher"), and certify
how often the compiled table disagrees with that teacher. No human labels.

    records = [{"text": "...", "intent": "lost_card"}, ...]     # the teacher's answers, logged in shadow
    model, cert = shadow_compile(schema, records, alpha=0.05, teacher="my-llm-v3")
    model.save("bundle/"); write_certificate("bundle/", cert)

The certificate (Learn-then-Test over confidence thresholds with Clopper-Pearson bounds; the default procedure "auto"
runs fixed-sequence and Bonferroni at delta/2 each and keeps the lower threshold):
    on inputs drawn like the calibration slice, among answers the table serves itself (certified=True),
    the rate of disagreement with the teacher is at most alpha, with probability at least 1 - delta.
It is a bound against the TEACHER, not against the truth, and it is marginal over the calibration distribution:
a shift in the mix of traffic can break it, which is what audits (shad0w.audit) and re-certification are for.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os

import numpy as np

from .api import DriftGuard, Model, Question, load, to_bool
from .features import items
from .mathutil import softmax

CERT_FILE = "certificate.json"


_log = logging.getLogger("shad0w")

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
    return np.array([int(to_bool(v)) for v in ys])


def _certify(q: Question, texts, y, alpha: float, delta: float, drift_window: int = 500, drift_margin: float = 0.03):
    """Fit the certified threshold on teacher-labelled calibration texts (never used for fitting)."""
    from .reliability import SelectiveRiskController
    P = np.stack([softmax(q.rx.logits(items(t))) for t in texts])
    conf, agree = P.max(1), P.argmax(1) == y
    ctl = SelectiveRiskController(alpha, delta).fit(conf, agree)
    thr = float(ctl.threshold)
    served = conf >= thr
    q.threshold, q.alpha, q.calibrated = thr, alpha, True
    q.guard = DriftGuard(conf, agree, drift_window, drift_margin)
    return {
        "n_calibration": int(len(y)),
        "agreement_with_teacher": float(agree.mean()),
        "threshold": thr if np.isfinite(thr) else None,
        "certified_share_on_calibration": float(served.mean()),
        "disagreement_on_certified_calibration": float((~agree[served]).mean()) if served.any() else None,
        "procedure": f"learn-then-test/clopper-pearson/{ctl.procedure}" + (f" ({ctl.chosen})" if ctl.chosen else ""),
    }


def shadow_compile(schema: dict, records: list[dict], alpha: float = 0.05, delta: float = 0.1,
                   cal_fraction: float = 0.3, max_cal: int = 3000, teacher: str = "unspecified",
                   seed: int = 0, r_min: float = 8.0, cal_records: list[dict] | None = None,
                   drift_window: int = 500, drift_margin: float = 0.03, max_mb: float | None = None):
    """Compile every question in `schema` from the teacher's logged answers in `records` and certify it.
    Returns (Model, certificate dict). Each question needs at least ~1,000 records for a useful certificate.

    cal_records: an optional separate calibration set (for example the uniform spot-check sample of live traffic).
    When given, every row of `records` is used for fitting and only `cal_records` certify; the certificate then
    says calibration="uniform-audit" instead of "held-out-split"."""
    from .compiler.distill import compile_schema, max_features_for

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
        ext = [r for r in (cal_records or []) if name in r and r.get("text")]
        if ext:
            if len(ext) < 100:
                raise ValueError(f"question {name!r}: {len(ext)} calibration records; need at least 100")
            if len(ext) > max_cal:
                ext = [ext[i] for i in sorted(rng.permutation(len(ext))[:max_cal])]
            fit = np.arange(len(texts))
            cal_texts, cal_y = [r["text"] for r in ext], _encode_labels(qtype, options, [r[name] for r in ext])
            how = "uniform-audit"
        else:
            perm = rng.permutation(len(texts))
            n_cal = int(min(max_cal, max(100, cal_fraction * len(texts)), len(texts) // 2))  # always leave rows to fit on
            cal, fit = perm[:n_cal], perm[n_cal:]
            cal_texts, cal_y = [texts[i] for i in cal], y[cal]
            how = "held-out-split"
        comp = compile_schema([texts[i] for i in fit], y[fit], options, max_features=max_features_for(max_mb, len(options)))
        q = Question(name, qtype, options, comp.rx, float("inf"), alpha, DriftGuard(np.ones(2), np.ones(2, bool)), r_min, False)
        stats = _certify(q, cal_texts, cal_y, alpha, delta, drift_window, drift_margin)
        model.questions[name] = q
        cert["questions"][name] = {"type": qtype, "options": options, "n_records": len(texts), "n_fit": int(len(fit)),
                                   "data_sha256": _hash_records(texts, [r[name] for r in rows]), "calibration": how, **stats}
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
        if not rows:  # it keeps serving on its old threshold, so keep (and mark) its old certificate entry
            if name in old.get("questions", {}):
                cert["questions"][name] = {**old["questions"][name], "recertified": False}
                _log.warning("certify_bundle: no fresh records for %r; it keeps its previous certificate", name)
            continue
        texts = [r["text"] for r in rows]
        y = _encode_labels(q.qtype, q.options, [r[name] for r in rows])
        cert["questions"][name] = {"type": q.qtype, "options": q.options, "n_records": len(texts),
                                   "data_sha256": _hash_records(texts, [r[name] for r in rows]),
                                   **_certify(q, texts, y, alpha, delta, q.guard.window, q.guard.margin)}
    m.save(bundle)
    write_certificate(bundle, cert)
    return cert


def gate_reason(new: dict, old: dict | None) -> str | None:
    """Why a retrained question should NOT replace the current one (None = accept it).

    The new table must certify something and at least 80% of the old table's certified share. When the option set
    changed, the old table answers a different question, so it is no baseline and the new one is always accepted."""
    if not old or old.get("threshold") is None:
        return None
    if set(old.get("options") or []) != set(new.get("options") or []):
        return None
    new_share, old_share = new["certified_share_on_calibration"], old.get("certified_share_on_calibration", 0.0)
    if new["threshold"] is None or new_share < 0.8 * old_share:
        return (f"new bundle certifies {new_share:.1%} of calibration traffic vs {old_share:.1%} before; "
                "kept the old one (retrain='always' or no --gate to replace anyway)")
    return None


def write_certificate(bundle: str, cert: dict):
    os.makedirs(bundle, exist_ok=True)
    with open(os.path.join(bundle, CERT_FILE), "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)


def read_certificate(bundle: str):
    p = os.path.join(bundle, CERT_FILE)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def read_jsonl(path: str) -> list[dict]:
    """Rows of a JSON Lines file. A malformed line (say, half-written when a process was killed) is skipped with a
    warning naming its line number, so one bad line never blocks training."""
    rows, bad = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad.append(i)
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad.append(i)
    if bad:
        _log.warning("%s: skipped %d malformed line(s): %s", path, len(bad), ", ".join(map(str, bad[:10])) + (" ..." if len(bad) > 10 else ""))
    return rows
