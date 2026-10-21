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

from .api import DriftGuard, Model, Question, load, question_items, to_bool
from .features import TEXT_KIND, as_items, check_kind, using_kind
from .mathutil import softmax

CERT_FILE = "certificate.json"


_log = logging.getLogger("shad0w")

def _input(r: dict):
    """A log row's input: its text, or the item ids an image was turned into (rows written by an image decision)."""
    t = r.get("text")
    return t if t else r.get("items")


def _coverage_at(P: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Share of rows a confidence threshold could serve with empirical disagreement <= alpha (a selection score)."""
    order = np.argsort(-P.max(1))
    wrong = np.cumsum(P.argmax(1)[order] != y[order])
    ok = np.flatnonzero(wrong <= alpha * np.arange(1, len(y) + 1))
    return (ok.max() + 1) / len(y) if len(ok) else 0.0


def _choose_static(texts, y, options, rows, alpha, seed, max_mb, model: str = "potion-8M"):
    """static="auto": train with and without static word vectors on 80% of the fitting rows, keep the one that would
    serve more of the other 20% at disagreement <= alpha. Calibration rows are not used, so the certificate keeps its
    full delta. Returns the static model name or None."""
    from .compiler.distill import compile_schema, max_features_for
    from .static import word_vectors
    rng = np.random.default_rng(seed + 1)
    sel = rng.random(len(texts)) < 0.2
    if sel.sum() < 50 or (~sel).sum() < 100:
        return None
    tr = np.flatnonzero(~sel)
    te = np.flatnonzero(sel)
    best, best_cov = None, -1.0
    for v in (None, model):
        dense = word_vectors([texts[i] for i in tr], model=v) if v else None
        comp = compile_schema([texts[i] for i in tr], y[tr], options, max_features=max_features_for(max_mb, len(options)),
                              soft=_soft_targets([rows[i] for i in tr], options, y[tr]), student="linear", dense=dense)
        P = np.stack([softmax(comp.rx.logits(as_items(texts[i]))) for i in te])
        cov = _coverage_at(P, y[te], alpha)
        _log.info("static=auto: %s serves %.1f%% of held-out fitting rows at alpha", v or "n-grams only", 100 * cov)
        if cov > best_cov + 0.01:  # the static table must earn its place by more than noise
            best, best_cov = v, cov
    return best


def _soft_targets(rows: list[dict], options: list, y: np.ndarray):
    """(n, K) fitting targets: a row's logged teacher probabilities ("probs") where it has them, else its one-hot
    answer; None when no row has probabilities (then training is exactly the hard-label fit)."""
    if not any(isinstance(r.get("probs"), dict) for r in rows):
        return None
    idx = {str(o): i for i, o in enumerate(options)}
    Y = np.zeros((len(rows), len(options)), np.float32)
    for i, r in enumerate(rows):
        p = r.get("probs")
        if isinstance(p, dict):
            for k, v in p.items():
                j = idx.get(str(k))
                if j is not None and isinstance(v, (int, float)) and v > 0:
                    Y[i, j] = v
        s = Y[i].sum()
        if s > 0:
            Y[i] /= s
        else:
            Y[i, y[i]] = 1.0
    return Y


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
    P = np.stack([softmax(q.rx.logits(question_items(q, t))) for t in texts])
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
                   drift_window: int = 500, drift_margin: float = 0.03, max_mb: float | None = None,
                   image: dict | None = None, static: str | None = None, fit_extra: list[dict] | None = None,
                   student: str = "auto", hidden: int | None = None, text_kind: int = TEXT_KIND):
    """Compile every question in `schema` from the teacher's logged answers in `records` and certify it.
    Returns (Model, certificate dict). Each question needs at least ~1,000 records for a useful certificate.

    student: "linear" (additive table, exact radius), "deep" (a hidden layer of `hidden` units: models interactions
    between items, no radius) or "auto" (default): both are trained and the one that would serve more held-out
    FITTING rows at alpha is kept (linear on a tie); the certificate is then computed on the calibration rows.

    fit_extra: rows used for fitting only, never for calibration (synthetic examples written by an LLM: they are not
    drawn like live traffic, so they must not certify anything).

    static: None, a static model ("potion-8M") whose word vectors become extra items (text questions only), or "auto":
    train with and without them, certify each at delta / 2 (a union bound keeps the guarantee at delta) and keep the
    one that certifies the larger share of calibration traffic.

    cal_records: an optional separate calibration set (for example the uniform spot-check sample of live traffic).
    When given, every row of `records` is used for fitting and only `cal_records` certify; the certificate then
    says calibration="uniform-audit" instead of "held-out-split".

    text_kind: the text featurizer (shad0w.features): 2 (default: script-aware, so Chinese, Japanese, Thai, ... get
    character words; identical to 1 on ASCII text) or 1 (the featurizer of older bundles). It is recorded in the
    manifest meta and the certificate, and the Python, C and JS runtimes all use it."""
    from .compiler.distill import compile_schema, max_features_for

    text_kind = check_kind(text_kind)
    rng = np.random.default_rng(seed)
    model = Model(meta={"mode": "shadow", "teacher": teacher, "alpha": alpha, "delta": delta,
                        **({"image": image} if image else {}), **({"text_kind": text_kind} if text_kind != 1 else {})})
    cert = {"format": "shad0w-certificate/1", "mode": "shadow", "certified_against": "teacher", "teacher": teacher,
            "alpha": alpha, "delta": delta, "text_kind": text_kind,
            "statement": ("On inputs drawn like the calibration slice, among answers this bundle serves with "
                          "certified=true, the rate of disagreement with the teacher is at most alpha with "
                          "probability at least 1 - delta. The bound is against the teacher, not the truth, and "
                          "does not survive a shift in the traffic mix without re-certification."),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            **({"image": image} if image else {}), "questions": {}}
    for name, spec in schema.items():
        qtype = spec.get("type", "choice")
        options = list(spec["criteria"].keys()) if qtype == "choice" else ["no", "yes"]
        rows = [r for r in records if name in r and _input(r)]
        if len(rows) < 100:
            raise ValueError(f"question {name!r}: {len(rows)} teacher records; need at least 100 (1,000+ recommended)")
        texts = [_input(r) for r in rows]
        y = _encode_labels(qtype, options, [r[name] for r in rows])
        ext = [r for r in (cal_records or []) if name in r and _input(r)]
        if ext:
            if len(ext) < 100:
                raise ValueError(f"question {name!r}: {len(ext)} calibration records; need at least 100")
            if len(ext) > max_cal:
                ext = [ext[i] for i in sorted(rng.permutation(len(ext))[:max_cal])]
            fit = np.arange(len(texts))
            cal_texts, cal_y = [_input(r) for r in ext], _encode_labels(qtype, options, [r[name] for r in ext])
            how = "uniform-audit"
            votes = sorted({int(r["teacher_votes"]) for r in ext if r.get("teacher_votes")})
            if votes:  # `shad0w votes`: calibration answers are a majority of k teacher answers
                how += f" (majority of {votes[0]}" + (f"-{votes[-1]}" if len(votes) > 1 else "") + " teacher votes)"
        else:
            perm = rng.permutation(len(texts))
            n_cal = int(min(max_cal, max(100, cal_fraction * len(texts)), len(texts) // 2))  # always leave rows to fit on
            cal, fit = perm[:n_cal], perm[n_cal:]
            cal_texts, cal_y = [texts[i] for i in cal], y[cal]
            how = "held-out-split"
        extra = [r for r in (fit_extra or []) if name in r and _input(r) and r[name] is not None]
        extra_y = _encode_labels(qtype, options, [r[name] for r in extra]) if extra else np.zeros(0, int)
        fit_rows_all = [rows[i] for i in fit] + extra
        fit_y_all = np.concatenate([y[fit], extra_y]).astype(int)
        fit_texts = [texts[i] for i in fit] + [_input(r) for r in extra]
        is_text = all(isinstance(t, str) for t in fit_texts)
        choice = static if (static and is_text) else None
        if choice == "auto":  # chosen on a held-out slice of the FITTING rows: calibration stays untouched, full delta
            with using_kind(text_kind):
                choice = _choose_static(fit_texts, fit_y_all, options, fit_rows_all, alpha, seed, max_mb)
        dense = None
        if choice is not None:  # static word vectors: trained as an extra input, folded into the word rows
            from .static import word_vectors
            dense = word_vectors(fit_texts, model=choice, kind=text_kind)
        with using_kind(text_kind):  # compile_schema hashes the texts itself
            comp = compile_schema(fit_texts, fit_y_all, options, max_features=max_features_for(max_mb, len(options)),
                                  soft=_soft_targets(fit_rows_all, options, fit_y_all), student=student, hidden=hidden,
                                  alpha=alpha, dense=dense)
        q = Question(name, qtype, options, comp.rx, float("inf"), alpha, DriftGuard(np.ones(2), np.ones(2, bool)),
                     r_min, False, text_kind=text_kind)
        stats = _certify(q, cal_texts, cal_y, alpha, delta, drift_window, drift_margin)
        stats["static"] = choice
        if choice is not None:
            stats["static_mode"] = "folded"
        stats["student"] = comp.student
        if comp.student == "deep":
            stats["hidden"] = comp.rx.hidden
        if comp.choice:
            stats["student_choice"] = comp.choice
        if static == "auto" and is_text:
            stats["procedure"] += "; static=auto chosen on held-out fitting rows"
        model.questions[name] = q
        cert["questions"][name] = {"type": qtype, "options": options, "n_records": len(texts), "n_fit": int(len(fit)),
                                   "n_fit_only": len(extra),
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
        rows = [r for r in records if name in r and _input(r)]
        if not rows:  # it keeps serving on its old threshold, so keep (and mark) its old certificate entry
            if name in old.get("questions", {}):
                cert["questions"][name] = {**old["questions"][name], "recertified": False}
                _log.warning("certify_bundle: no fresh records for %r; it keeps its previous certificate", name)
            continue
        texts = [_input(r) for r in rows]
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


def cascade_compile(name: str, spec: dict, tiers: list, records: list[dict], alpha: float = 0.05, delta: float = 0.1,
                    cal_fraction: float = 0.3, max_cal: int = 3000, teacher: str = "unspecified", seed: int = 0,
                    cal_records: list[dict] | None = None, drift_window: int = 500, drift_margin: float = 0.03,
                    student: str = "auto", hidden: int | None = None):
    """Compile one table per image tier (an ImageSpec on the log's "items", an EmbedSpec on its "embed" vectors) and
    certify the CASCADE with one certificate (reliability.CascadeController): among answers any tier serves,
    disagreement with the teacher <= alpha with probability >= 1 - delta. Tier k is question `name` (k = 0) or
    `name@k`. An EmbedSpec's bucket edges are fitted on the fitting rows only."""
    from .compiler.distill import compile_schema
    from .reliability import CascadeController
    from .vision import EmbedSpec

    rng = np.random.default_rng(seed)
    qtype = spec.get("type", "choice")
    options = list(spec["criteria"].keys()) if qtype == "choice" else ["no", "yes"]
    rows = [r for r in records if name in r and r.get("items")]
    if len(rows) < 100:
        raise ValueError(f"question {name!r}: {len(rows)} teacher records; need at least 100 (1,000+ recommended)")
    y = _encode_labels(qtype, options, [r[name] for r in rows])
    ext = [r for r in (cal_records or []) if name in r and r.get("items")]
    if len(ext) >= 100:
        fit_rows, fit_y = rows, y
        cal_rows = ext if len(ext) <= max_cal else [ext[i] for i in sorted(rng.permutation(len(ext))[:max_cal])]
        cal_y, how = _encode_labels(qtype, options, [r[name] for r in cal_rows]), "uniform-audit"
    else:
        perm = rng.permutation(len(rows))
        n_cal = int(min(max_cal, max(100, cal_fraction * len(rows)), len(rows) // 2))
        cal_rows, fit_rows = [rows[i] for i in perm[:n_cal]], [rows[i] for i in perm[n_cal:]]
        cal_y, fit_y, how = y[perm[:n_cal]], y[perm[n_cal:]], "held-out-split"
    fitted, questions, confs, agrees = [], [], [], []
    for k, t in enumerate(tiers):
        qname = name if k == 0 else f"{name}@{k}"
        if isinstance(t, EmbedSpec):
            have = [i for i, r in enumerate(fit_rows) if r.get("embed")]
            if len(have) < 100:
                raise ValueError(f"tier {k} ({t.model}): {len(have)} logged embeddings; need at least 100")
            t = t.fit([fit_rows[i]["embed"] for i in have])
            inp = lambda r, t=t: t.items(r["embed"]) if r.get("embed") else None
        else:
            have = list(range(len(fit_rows)))
            inp = lambda r: r["items"]
        fitted.append(t)
        comp = compile_schema([inp(fit_rows[i]) for i in have], fit_y[have], options, student=student, hidden=hidden,
                              alpha=alpha)
        q = Question(qname, qtype, options, comp.rx, float("inf"), alpha, DriftGuard(np.ones(2), np.ones(2, bool)), 0.0, True)
        c, a = np.full(len(cal_rows), np.nan), np.zeros(len(cal_rows), bool)
        for i, r in enumerate(cal_rows):
            its = inp(r)
            if its is not None:
                p = softmax(q.rx.logits(as_items(its)))
                c[i], a[i] = p.max(), p.argmax() == cal_y[i]
        m = np.isfinite(c)
        q.guard = DriftGuard(c[m], a[m], drift_window, drift_margin) if m.sum() >= 2 else q.guard
        questions.append(q)
        confs.append(c)
        agrees.append(a)
    ctl = CascadeController(alpha, delta, seed=seed).fit(np.stack(confs), np.stack(agrees))
    for q, t in zip(questions, ctl.thresholds):
        q.threshold = float(t)
    tier_meta = [{"question": q.name, "spec": t.to_dict()} for q, t in zip(questions, fitted)]
    model = Model(meta={"mode": "shadow", "teacher": teacher, "alpha": alpha, "delta": delta,
                        "image": fitted[0].to_dict(), "tiers": tier_meta},
                  questions={q.name: q for q in questions})
    total = float(ctl.share.sum())
    entry = {"type": qtype, "options": options, "n_records": len(rows), "n_fit": len(fit_rows),
             "data_sha256": _hash_records([json.dumps(r.get("items")) for r in rows], [r[name] for r in rows]),
             "calibration": how, "n_calibration": len(cal_rows),
             "threshold": None if not np.isfinite(ctl.thresholds[0]) else float(ctl.thresholds[0]),
             "certified_share_on_calibration": total,
             "disagreement_on_certified_calibration": ctl.risk,
             "procedure": "learn-then-test/clopper-pearson/cascade-fixed-sequence",
             "tiers": [{"question": q.name, "kind": t.to_dict()["kind"], "student": "deep" if q.rx.hidden else "linear",
                        "threshold": float(th) if np.isfinite(th) else None, "share": float(sh)}
                       for q, t, th, sh in zip(questions, fitted, ctl.thresholds, ctl.share)]}
    cert = {"format": "shad0w-certificate/1", "mode": "shadow", "certified_against": "teacher", "teacher": teacher,
            "alpha": alpha, "delta": delta,
            "statement": ("On inputs drawn like the calibration slice, among answers this cascade serves with "
                          "certified=true (from any tier), the rate of disagreement with the teacher is at most alpha "
                          "with probability at least 1 - delta. The bound is against the teacher, not the truth."),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "image": fitted[0].to_dict(), "questions": {name: entry}}
    return model, cert
