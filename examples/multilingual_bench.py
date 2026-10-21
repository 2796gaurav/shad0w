"""Text featurizer kind 1 vs kind 2 (script-aware items) on Chinese, Japanese and Thai, with English as the control.

    pip install "shad0wllm[compile]" datasets
    python examples/multilingual_bench.py                         # downloads the sets from the Hugging Face hub
    python examples/multilingual_bench.py --data DIR --sets tnews,livedoor   # DIR/<set>/data.jsonl: {"text", "label"}

Kind 1 hashes runs of non-ASCII bytes as words, so a whole Chinese / Japanese / Thai sentence (no spaces) is one
"word" and only its byte trigrams generalise. Kind 2 makes each Han / kana / Thai character a word of its own, with
character bigrams; on ASCII both kinds give identical items, so English numbers must not move.

The teacher is the dataset's labels (an upper bound: a perfectly consistent teacher). For each set and kind: train on
up to 10k rows, fit the certified threshold on 2k calibration rows (Learn-then-Test, alpha 5 %, delta 10 %), and report
on 2k unseen test rows: agreement with the teacher, the share certified, and the disagreement on that share.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shad0w.compiler.distill import compile_schema  # noqa: E402
from shad0w.features import items, using_kind  # noqa: E402
from shad0w.mathutil import softmax  # noqa: E402
from shad0w.reliability import SelectiveRiskController  # noqa: E402

SETS = {  # name: (language, loader)
    "tnews": "zh", "livedoor": "ja", "massive_ja": "ja", "wongnai": "th", "ag_news": "en", "banking77": "en",
}


def _hub(name):
    from datasets import load_dataset
    if name == "tnews":
        ds = load_dataset("clue/clue", "tnews")
        names = ds["train"].features["label"].names
        return [(r["sentence"], names[r["label"]]) for s in ("train", "validation") for r in ds[s]]
    if name == "livedoor":
        ds = load_dataset("llm-book/livedoor-news-corpus", revision="refs/convert/parquet")
        out = []
        for s in ds:
            f = ds[s].features["category"]
            for r in ds[s]:
                y = f.names[r["category"]] if hasattr(f, "names") else r["category"]
                out.append(((r.get("title") or "") + "\n" + (r.get("content") or ""), y))
        return out
    if name == "massive_ja":
        ds = load_dataset("mteb/amazon_massive_intent", "ja")
        return [(r["text"], r["label"]) for s in ds for r in ds[s]]
    if name == "wongnai":
        ds = load_dataset("Wongnai/wongnai_reviews")
        return [(r["review_body"], r["star_rating"]) for s in ds for r in ds[s]]
    if name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news")
        return [(r["text"], r["label"]) for s in ("train", "test") for r in ds[s]]
    if name == "banking77":
        ds = load_dataset("legacy-datasets/banking77")
        return [(r["text"], r["label"]) for s in ("train", "test") for r in ds[s]]
    raise ValueError(name)


def load_rows(name, data_dir):
    path = os.path.join(data_dir, name, "data.jsonl") if data_dir else ""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [(r["text"], str(r["label"])) for r in rows]
    return [(t, str(y)) for t, y in _hub(name) if t and str(t).strip()]


def split(rows, n_train, n_cal, n_test, seed=0):
    perm = np.random.default_rng(seed).permutation(len(rows))
    n_eval = n_cal + n_test
    if len(rows) < n_eval + 1000:  # a small set: shrink calibration and test alike
        n_cal = n_test = max(300, (len(rows) - 1000) // 4)
        n_eval = n_cal + n_test
    tr = perm[n_eval:n_eval + n_train]
    return [rows[i] for i in tr], [rows[i] for i in perm[:n_cal]], [rows[i] for i in perm[n_cal:n_eval]]


def run(train, cal, test, kind, student, alpha, delta):
    labels = sorted({y for _, y in train})
    idx = {y: i for i, y in enumerate(labels)}
    keep = lambda part: [(t, idx[y]) for t, y in part if y in idx]  # noqa: E731
    cal, test = keep(cal), keep(test)
    t0 = time.time()
    with using_kind(kind):
        comp = compile_schema([t for t, _ in train], np.array([idx[y] for _, y in train]), labels, student=student,
                              alpha=alpha)
    fit_s = time.time() - t0

    def probs(part):
        return np.stack([softmax(comp.rx.logits(items(t, kind=kind))) for t, _ in part])

    Pc, Pt = probs(cal), probs(test)
    yc, yt = np.array([y for _, y in cal]), np.array([y for _, y in test])
    thr = float(SelectiveRiskController(alpha, delta).fit(Pc.max(1), Pc.argmax(1) == yc).threshold)
    served = Pt.max(1) >= thr
    agree = Pt.argmax(1) == yt
    return {"kind": kind, "student": comp.student if student == comp.student else f"{student}:{comp.student}", "agreement": float(agree.mean()),
            "certified": float(served.mean()), "risk_on_certified": float((~agree[served]).mean()) if served.any() else None,
            "threshold": thr if np.isfinite(thr) else None, "rows": int(len(comp.rx.keys)), "fit_s": round(fit_s, 1),
            "items_per_text": float(np.mean([len(items(t, kind=kind)) for t, _ in test[:500]])),
            "n": [len(train), len(cal), len(test)], "K": len(labels)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--data", help="directory of <set>/data.jsonl files (else downloaded)")
    ap.add_argument("--students", default="linear,auto")
    ap.add_argument("--train", type=int, default=10_000)
    ap.add_argument("--cal", type=int, default=2_000)
    ap.add_argument("--test", type=int, default=2_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--out", help="write every result as JSON Lines here")
    a = ap.parse_args()
    results = []
    print(f"{'set':<11}{'lang':<5}{'student':<12}{'kind':>5}{'agree':>8}{'certified':>11}{'risk@cert':>11}{'rows':>9}"
          f"{'items/text':>12}{'fit s':>7}")
    for name in a.sets.split(","):
        rows = load_rows(name, a.data)
        train, cal, test = split(rows, a.train, a.cal, a.test)
        same = sum(items(t, kind=1).tolist() == items(t, kind=2).tolist() for t, _ in test)
        for student in a.students.split(","):
            for kind in (1, 2):
                r = {"set": name, "lang": SETS.get(name, "?"), **run(train, cal, test, kind, student, a.alpha, a.delta),
                     "test_texts_with_identical_items": same / len(test)}
                results.append(r)
                risk = "-" if r["risk_on_certified"] is None else f"{100 * r['risk_on_certified']:.1f}%"
                print(f"{name:<11}{r['lang']:<5}{r['student']:<12}{kind:>5}{100 * r['agreement']:>7.1f}%"
                      f"{100 * r['certified']:>10.1f}%{risk:>11}{r['rows']:>9,}{r['items_per_text']:>12.0f}{r['fit_s']:>7.0f}",
                      flush=True)
        print(f"{'':<11}test texts whose items are identical under kind 1 and 2: {100 * same / len(test):.1f}%", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
