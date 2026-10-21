"""shad0w under Jevstiller's benchmark protocol, so the two can be read side by side.

Jevstiller (github.com/tomerglick57/Jevstiller) distils a repeated Jev decision into bge-small embeddings + a logistic
head, and routes to the local model only when a Clopper-Pearson bound (picked strictest-first on a held-out IID
calibration set) says the disagreement with Jev stays within `1 - target_agreement`. Its README reports, on 2,000
held-out rows per task with target_agreement = 0.98 (a 2% disagreement budget) at 95% confidence:

    Banking77  71.9% answered locally, 99.50% agreement      CLINC150  69.0%, 99.65%      AG News  80.2%, 99.40%

and about 15 ms p50 per local answer on CPU (Jev itself ~300 ms). This script runs shad0w the same way:

  * the dataset is shuffled (seed 0); the first 2,000 rows are the held-out evaluation set, the rest is the stream
    the table is compiled and certified from (shad0w.shadow.shadow_compile: fit / calibration split inside);
  * the teacher is the dataset's own labels ("oracle" in Jevstiller's harness), so agreement = accuracy here. State
    that when you quote the numbers: with an LLM teacher both tools certify agreement with it, not accuracy;
  * coverage = share of the 2,000 evaluation rows the table answers (certified); disagreement on served = share of
    those answers that differ from the teacher; p50 latency = one decide() call on the Python path and on the C core
    (shad0w.native, when the library is built).
Reported at Jevstiller's setting (alpha 2%, 95% confidence) and at shad0w's default (alpha 5%, 90% confidence).

    pip install "shad0wllm[compile]" datasets
    python examples/jevstiller_compare.py [--datasets banking77,clinc150,ag_news] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time

import numpy as np

DATASETS = {
    # name: (HF id, config, text field, label field, extra note)
    "banking77": ("mteb/banking77", None, "text", "label_text", "77 intents"),
    "clinc150": ("DeepPavlov/clinc_oos", "plus", "text", "label_text", "150 intents + oos kept as a 151st option"),
    "ag_news": ("fancyzhx/ag_news", None, "text", "label", "4 sections; stream capped at 30,000 rows"),
}
EVAL_ROWS = 2000
SETTINGS = (("jevstiller: alpha 2%, 95% conf", 0.02, 0.05), ("shad0w default: alpha 5%, 90% conf", 0.05, 0.10))


def load_rows(name: str, cap: int | None = None):
    from datasets import load_dataset
    hf, cfg, tf, lf, _ = DATASETS[name]
    ds = load_dataset(hf, cfg) if cfg else load_dataset(hf)
    rows = []
    for split in ds:
        d = ds[split]
        names = getattr(d.features[lf], "names", None)
        for t, y in zip(d[tf], d[lf]):
            rows.append({"text": t, "label": str(names[y]) if names else str(y)})
    random.Random(0).shuffle(rows)
    if cap:
        rows = rows[:EVAL_ROWS + cap]
    return rows


def p50_us(model, texts, warm: int = 50) -> float:
    for t in texts[:warm]:
        model.decide(t)
    ts = []
    for t in texts:
        t0 = time.perf_counter_ns()
        model.decide(t)
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return ts[len(ts) // 2] / 1e3


def run(name: str) -> dict:
    from shad0w.api import load
    from shad0w.native import available
    from shad0w.shadow import shadow_compile
    rows = load_rows(name, cap=30000 if name == "ag_news" else None)
    ev, stream = rows[:EVAL_ROWS], rows[EVAL_ROWS:]
    options = sorted({r["label"] for r in rows})
    schema = {"intent": {"type": "choice", "criteria": {o: None for o in options}}}
    records = [{"text": r["text"], "intent": r["label"]} for r in stream]
    out = {"dataset": name, "note": DATASETS[name][4], "options": len(options), "stream_rows": len(stream),
           "eval_rows": len(ev), "teacher": "dataset labels (oracle)", "settings": {}}
    for label, alpha, delta in SETTINGS:
        t0 = time.time()
        model, cert = shadow_compile(schema, records, alpha=alpha, delta=delta, teacher="truth-labels")
        train_s = time.time() - t0
        q = cert["questions"]["intent"]
        with tempfile.TemporaryDirectory() as d:
            model.save(d)
            res = {"train_s": round(train_s, 1), "calibration_rows": q["n_calibration"],
                   "certified_share_on_calibration": q["certified_share_on_calibration"], "threshold": q["threshold"]}
            for path, native in (("python", False), ("c", True)):
                if native and not available():
                    res["p50_us_c"] = None
                    continue
                m = load(d, native=native)
                served = agree = 0
                for r in ev:
                    dec = m.decide(r["text"])["answers"]["intent"]
                    if dec.get("certified"):
                        served += 1
                        agree += dec["choice"] == r["label"]
                if path == "python":
                    res["coverage"] = served / len(ev)
                    res["disagreement_on_served"] = (1 - agree / served) if served else None
                    res["agreement_on_served"] = (agree / served) if served else None
                res[f"p50_us_{path}"] = round(p50_us(m, [r["text"] for r in ev[:500]]), 1)
        out["settings"][label] = res
        print(f"{name:10s} {label:36s} coverage {res['coverage']:.1%}  disagreement on served "
              f"{(res['disagreement_on_served'] or 0):.2%}  p50 py {res['p50_us_python']} us  c {res.get('p50_us_c')} us",
              flush=True)
    return out


def table(results: list[dict]) -> str:
    lines = ["| dataset | setting | coverage | disagreement on served | p50 Python | p50 C core | cal rows |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        for label, s in r["settings"].items():
            lines.append(f"| {r['dataset']} | {label} | {s['coverage']:.1%} | {(s['disagreement_on_served'] or 0):.2%} | "
                         f"{s['p50_us_python']} us | {s.get('p50_us_c') or 'n/a'} us | {s['calibration_rows']} |")
    lines.append("| Jevstiller README (bge-small, CPU) | banking77: alpha 2%, 95% conf | 71.9% | 0.50% | ~15 ms | - | - |")
    lines.append("| Jevstiller README | clinc150 (rare classes deferred) | 69.0% | 0.35% | ~15 ms | - | - |")
    lines.append("| Jevstiller README | ag_news | 80.2% | 0.60% | ~15 ms | - | - |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default="banking77,clinc150", help="comma-separated: " + ",".join(DATASETS))
    ap.add_argument("--out", help="write the results as JSON here")
    a = ap.parse_args()
    results = [run(n.strip()) for n in a.datasets.split(",") if n.strip()]
    print()
    print(table(results))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=float)
        print(f"\nwrote {os.path.abspath(a.out)}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
