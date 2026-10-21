"""Run the shad0w decision benchmark: every system answers the same calibration and test rows of every task, one
decision at a time on the CPU, and every answer is written to bench/results/preds/<task>/<system>.jsonl.gz.

    python bench/run.py                         # everything (several hours: the large decision models are slow on CPU)
    python bench/run.py --systems shad0w-linear,shad0w-deep,tfidf-logreg --tasks sst2
    python bench/run.py --dm-threads 5          # threads given to llama.cpp decision models (default: performance cores)

Answers that already exist are reused (delete a file to re-run it). Then `python bench/score.py` turns the
predictions into bench/results/results.json and `python bench/render.py` into bench/site/index.html.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "1")  # serving-time timing is single-threaded for in-process systems

from bench import systems as S  # noqa: E402
from bench.tasks import TASKS, load  # noqa: E402

PRED = os.path.join(HERE, "results", "preds")
N_CAL, N_TEST = 500, 500
LARGE_DM = {"kev-4b", "lev", "clef-flash"}  # too slow on a CPU for the full rows: test subset only, no calibration
LARGE_ROWS = 150
SLOW_DM = {"kev-0.8b", "decider-0.8b"}  # about 1-5 s per decision on a CPU: the first 150 test and calibration rows
SLOW_S = 1.0  # any other decision model averaging more than this per decision on a task's first 5 rows gets the subset

LEARNED = {"shad0w-auto": lambda: S.Shad0w("auto"), "shad0w-linear": lambda: S.Shad0w("linear"),
           "shad0w-deep": lambda: S.Shad0w("deep"),
           "tfidf-logreg": S.TfidfLR,
           "potion8m-logreg": lambda: S.EmbedLR("minishlab/potion-base-8M", "potion8m-logreg"),
           "minilm-logreg": lambda: S.EmbedLR("sentence-transformers/all-MiniLM-L6-v2", "minilm-logreg"),
           "bge-small-logreg": lambda: S.EmbedLR("BAAI/bge-small-en-v1.5", "bge-small-logreg")}
# The 4B-9B decision models (Kev-4B, Lev, Clef-flash) take seconds per decision on a CPU; they can be run with
# --systems kev-4b,... but are not part of the default run.
DMS = ["julia-1", "laya", "kev-0.8b", "decider-0.8b", "gliner2.5-decide"]
BASELINES = ["tfidf-logreg", "potion8m-logreg", "minilm-logreg", "bge-small-logreg"]  # optional: --systems ...


def hardware(dm_threads: int) -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    import numpy
    info = {"cpu": sysctl("machdep.cpu.brand_string") or platform.processor(), "cores": os.cpu_count(),
            "performance_cores": sysctl("hw.perflevel0.physicalcpu"), "memory_gb": None, "os": platform.platform(),
            "python": platform.python_version(), "numpy": numpy.__version__, "dm_threads": dm_threads,
            "date": time.strftime("%Y-%m-%d")}
    mem = sysctl("hw.memsize")
    if mem and mem.isdigit():
        info["memory_gb"] = round(int(mem) / 2**30)
    try:
        import shad0w
        info["shad0w"] = shad0w.__version__
    except Exception:
        pass
    return info


def done(task, name):
    return os.path.exists(os.path.join(PRED, task, f"{name}.jsonl.gz"))


def save(task, name, rows, meta):
    os.makedirs(os.path.join(PRED, task), exist_ok=True)
    with gzip.open(os.path.join(PRED, task, f"{name}.jsonl.gz"), "wt", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_system(sysobj, task, workdir, rows_cal, rows_test, extra=None):
    sysobj.fit(task, workdir)
    out = []
    for split, rows in (("cal", rows_cal), ("test", rows_test)):
        texts = [t for t, _ in rows]
        res = S._timed(sysobj.answer, texts)
        for i, ((t, y), ((yp, conf, _p), us)) in enumerate(zip(rows, res)):
            out.append({"split": split, "i": i, "y": int(y), "pred": int(yp), "conf": round(float(conf), 7), "us": round(us, 2)})
    meta = {"system": sysobj.name, "kind": sysobj.kind, "info": sysobj.info, "train_s": round(getattr(sysobj, "train_s", 0.0), 2),
            "size_bytes": int(getattr(sysobj, "size_bytes", 0)), "n_cal": len(rows_cal), "n_test": len(rows_test),
            "n_train": len(task.train) if sysobj.kind == "learned" else 0, **(extra or {})}
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--systems", default=",".join(["shad0w-auto", "shad0w-linear", "shad0w-deep"] + DMS),
                    help="default: shad0w and the decision models; add " + ",".join(BASELINES) + " for classic baselines")
    ap.add_argument("--dm-threads", type=int, default=None, help="threads for llama.cpp decision models")
    ap.add_argument("--single-thread-probe", type=int, default=50,
                    help="also time this many test rows of each decision model on ONE thread (0 = skip)")
    a = ap.parse_args()
    perf = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout.strip()
    dm_threads = a.dm_threads or (int(perf) if perf.isdigit() else max(1, (os.cpu_count() or 2) // 2))
    tasks = [t for t in a.tasks.split(",") if t]
    names = [s for s in a.systems.split(",") if s]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(hardware(dm_threads), open(os.path.join(HERE, "results", "hardware.json"), "w"), indent=1)
    workdir = tempfile.mkdtemp(prefix="shad0w-bench-")
    loaded = {t: load(t, N_CAL, N_TEST) for t in tasks}
    for name in [n for n in names if n in LEARNED]:
        for t in tasks:
            if done(t, name):
                continue
            task = loaded[t]
            t0 = time.time()
            sysobj = LEARNED[name]()
            rows, meta = run_system(sysobj, task, workdir, task.cal, task.test)
            if isinstance(sysobj, S.Shad0w):
                meta["c_core"] = sysobj.c_core_ns([x for x, _ in task.test])
            save(t, name, rows, meta)
            print(f"{t:17s} {name:18s} done in {time.time() - t0:6.0f}s", flush=True)
    for name in [n for n in names if n in S.GGUF or n in S.PY_DM]:
        todo = [t for t in tasks if not done(t, name)]
        if not todo:
            continue
        make = (lambda th, port=18820, n=name: S.PyDecisionModel(n, th, port + 2)) if name in S.PY_DM else \
            (lambda th, port=18820, n=name: S.DecisionModel(n, th, port))
        dm = make(dm_threads).start()
        try:
            for t in todo:
                task = loaded[t]
                t0 = time.time()
                large, slow = name in LARGE_DM, name in SLOW_DM
                if not (large or slow):  # the same rule for every model: over 1 s per decision on this task -> subset
                    dm.fit(task, workdir)
                    t1 = time.perf_counter()
                    for x, _ in task.test[:5]:
                        dm.answer(x)
                    slow = (time.perf_counter() - t1) / 5 > SLOW_S
                cal = [] if large else (task.cal[:LARGE_ROWS] if slow else task.cal)
                test = task.test[:LARGE_ROWS] if (large or slow) else task.test
                rows, meta = run_system(dm, task, workdir, cal, test, {"subset": large or slow})
                save(t, name, rows, meta)
                print(f"{t:17s} {name:18s} done in {time.time() - t0:6.0f}s", flush=True)
        finally:
            dm.stop()
        if a.single_thread_probe and name not in LARGE_DM:
            probe = os.path.join(PRED, f"single-thread-{name}.json")
            if not os.path.exists(probe):
                one = make(1, 18821).start()
                try:
                    res = {}
                    for t in tasks:
                        one.fit(loaded[t], workdir)
                        tm = [us for _, us in S._timed(one.answer, [x for x, _ in loaded[t].test[:a.single_thread_probe]])]
                        res[t] = {"n": len(tm), "p50_us": float(sorted(tm)[len(tm) // 2])}
                    json.dump(res, open(probe, "w"), indent=1)
                finally:
                    one.stop()


if __name__ == "__main__":
    main()
