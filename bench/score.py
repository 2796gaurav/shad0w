"""Score the saved predictions (bench/results/preds) into bench/results/results.json. Needs no model, no network and
no GPU: anyone can re-check every number from the committed prediction files.

Per (task, system):
  accuracy            on the test rows, against the dataset's true labels, with a 95% Wilson interval
  p50_us / p99_us     wall time of ONE decision (batch size 1), measured around the call from Python
  certified_coverage  the same certifier for every system: Learn-then-Test with Clopper-Pearson
                      (shad0w.reliability.SelectiveRiskController, alpha = 5%, delta = 10%) picks a confidence threshold
                      on the calibration rows; coverage = share of test rows at or above it, served_error = error on them
  cascade             shad0w-auto answers what it certifies, the decision model answers the rest: accuracy and
                      mean time per decision of the pair, on the decision model's test rows
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from shad0w.reliability import SelectiveRiskController  # noqa: E402

ALPHA, DELTA = 0.05, 0.10


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        rows = [json.loads(line) for line in f]
    return meta, rows


def score_one(meta, rows):
    cal = [r for r in rows if r["split"] == "cal"]
    test = [r for r in rows if r["split"] == "test"]
    ok = np.array([r["pred"] == r["y"] for r in test])
    us = np.array([r["us"] for r in test])
    out = {"system": meta["system"], "kind": meta["kind"], "info": meta["info"], "n_test": len(test), "n_cal": len(cal),
           "n_train": meta.get("n_train", 0), "subset": bool(meta.get("subset")),
           "accuracy": round(float(ok.mean()), 4), "accuracy_ci95": wilson(int(ok.sum()), len(ok)),
           "p50_us": round(float(np.percentile(us, 50)), 2), "p99_us": round(float(np.percentile(us, 99)), 2),
           "mean_us": round(float(us.mean()), 2), "train_s": meta.get("train_s"), "size_mb": round(meta.get("size_bytes", 0) / 2**20, 2)}
    if meta.get("c_core"):
        out["c_core_p50_us"] = round(meta["c_core"]["p50_ns"] / 1000, 3)
        out["c_core_mean_us"] = round(meta["c_core"]["mean_ns"] / 1000, 3)
    if cal:
        ctl = SelectiveRiskController(ALPHA, DELTA).fit(np.array([r["conf"] for r in cal]),
                                                        np.array([r["pred"] == r["y"] for r in cal]))
        conf = np.array([r["conf"] for r in test])
        served = conf >= ctl.threshold
        out["certified_threshold"] = None if not np.isfinite(ctl.threshold) else round(float(ctl.threshold), 6)
        out["certified_coverage"] = round(float(served.mean()), 4)
        out["served_error"] = None if not served.any() else round(float((~ok[served]).mean()), 4)
    return out, test


NAMES = {"shad0w-auto": "shad0w", "tfidf-logreg": "TF-IDF + logreg", "potion8m-logreg": "potion-8M + logreg",
         "minilm-logreg": "MiniLM-L6 + logreg", "bge-small-logreg": "bge-small + logreg", "julia-1": "Julia-1", "laya": "Laya",
         "kev-0.8b": "Kev-0.8B", "kev-4b": "Kev-4B", "lev": "Lev", "clef-flash": "Clef-flash", "decider-0.8b": "Decider-0.8B",
         "gliner2.5-decide": "GLiNER2.5-Decide"}


def _us(u):
    if u is None:
        return "–"
    return f"{u:.1f} µs" if u < 1000 else (f"{u / 1000:.1f} ms" if u < 1e6 else f"{u / 1e6:.1f} s")


def summary_md(out) -> str:
    """The summary table that bench/README.md and the main README carry between their bench markers."""
    hw = out["hardware"]
    lines = [f"Measured on {hw['cpu']}, {hw['date']}. Decision models run zero-shot on the CPU only with {hw['dm_threads']} "
             "threads; shad0w runs on one thread and learns from each task's training split. Accuracy is against the true "
             "labels; certified = share of test rows served under the same 5% Learn-then-Test certificate.", "",
             "| task | shad0w: accuracy / certified / p50 | most accurate decision model | fastest decision model | "
             "shad0w in front of the most accurate |", "|---|---|---|---|---|"]
    for t, v in out["tasks"].items():
        S = v["systems"]
        a = S.get("shad0w-auto")
        dms = [r for r in S.values() if r["kind"] == "zero-shot"]
        if not a or not dms:
            continue
        best = max(dms, key=lambda r: r["accuracy"])
        fast = min(dms, key=lambda r: r["p50_us"])
        c = v["cascades"].get(best["system"])
        cell = lambda r: (f"{NAMES.get(r['system'], r['system'])}: {100 * r['accuracy']:.1f}%, {_us(r['p50_us'])}"  # noqa: E731
                          + (f" ({r['n_test']} rows)" if r.get("subset") else ""))
        lines.append(f"| {t} | {100 * a['accuracy']:.1f}% / {100 * (a.get('certified_coverage') or 0):.0f}% / {_us(a['p50_us'])} | "
                     f"{cell(best)} | {cell(fast)} | " + ("–" if not c else f"{100 * c['accuracy']:.1f}%, {c['speedup']:,}× faster") + " |")
    return "\n".join(lines)


def write_block(path, text):
    if not os.path.exists(path):
        return
    src = open(path, encoding="utf-8").read()
    a, b = "<!-- bench:start -->", "<!-- bench:end -->"
    if a in src and b in src:
        src = src[:src.index(a) + len(a)] + "\n" + text + "\n" + src[src.index(b):]
        open(path, "w", encoding="utf-8").write(src)


def main():
    hw = json.load(open(os.path.join(HERE, "results", "hardware.json")))
    probes = {os.path.basename(p)[len("single-thread-"):-5]: json.load(open(p))
              for p in glob.glob(os.path.join(HERE, "results", "preds", "single-thread-*.json"))}
    tasks = {}
    for tdir in sorted(glob.glob(os.path.join(HERE, "results", "preds", "*/"))):
        task = os.path.basename(os.path.dirname(tdir))
        res, tests = {}, {}
        for p in sorted(glob.glob(os.path.join(tdir, "*.jsonl.gz"))):
            meta, rows = read(p)
            s, test = score_one(meta, rows)
            if s["system"] in probes and task in probes[s["system"]]:
                s["single_thread_p50_us"] = probes[s["system"]][task]["p50_us"]
            res[s["system"]], tests[s["system"]] = s, test
        auto = "shad0w-auto" if "shad0w-auto" in res else None  # the library default: student="auto"
        cascades = {}
        if auto:
            st = tests[auto]
            thr = res[auto].get("certified_threshold")
            for name, r in res.items():
                if r["kind"] != "zero-shot":
                    continue
                dm = tests[name]
                n = len(dm)
                served = np.array([thr is not None and st[i]["conf"] >= thr for i in range(n)])
                pred = np.array([st[i]["pred"] if served[i] else dm[i]["pred"] for i in range(n)])
                y = np.array([dm[i]["y"] for i in range(n)])
                t_us = np.array([st[i]["us"] + (0.0 if served[i] else dm[i]["us"]) for i in range(n)])
                cascades[name] = {"front": auto, "n": n, "served_by_shad0w": round(float(served.mean()), 4),
                                  "accuracy": round(float((pred == y).mean()), 4),
                                  "decision_model_alone_accuracy": r["accuracy"],
                                  "mean_us": round(float(t_us.mean()), 1), "decision_model_alone_mean_us": r["mean_us"],
                                  "speedup": (lambda v: round(v, 1) if v < 100 else round(v))(float(r["mean_us"] / max(t_us.mean(), 1e-9)))}
        tasks[task] = {"systems": res, "shad0w_auto": auto, "cascades": cascades}
    out = {"benchmark": "shad0w decision bench", "version": 1, "alpha": ALPHA, "delta": DELTA, "hardware": hw, "tasks": tasks}
    json.dump(out, open(os.path.join(HERE, "results", "results.json"), "w"), indent=1)
    md = summary_md(out)
    open(os.path.join(HERE, "results", "SUMMARY.md"), "w", encoding="utf-8").write(md + "\n")
    for path in (os.path.join(HERE, "README.md"), os.path.join(os.path.dirname(HERE), "README.md")):
        write_block(path, md)
    for t, v in tasks.items():
        print(t)
        for s, r in sorted(v["systems"].items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"  {s:18s} acc {r['accuracy']:.3f}  cert {r.get('certified_coverage', float('nan')):.3f}  "
                  f"p50 {r['p50_us']:>10.1f} us  n={r['n_test']}")
        for s, c in v["cascades"].items():
            print(f"  cascade {c['front']} -> {s}: acc {c['accuracy']:.3f} (alone {c['decision_model_alone_accuracy']:.3f}), "
                  f"{c['speedup']}x faster, shad0w served {c['served_by_shad0w']:.0%}")


if __name__ == "__main__":
    main()
