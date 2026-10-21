"""End-to-end check of an installed shad0w, with no network and no API key.

    python examples/end_to_end.py [workdir]

A stand-in "LLM" labels movie reviews (it understands negation). The script then walks the whole product:
decision() logs its answers, train() builds and certifies a table, the table serves through the C core, the same
bundle decides identically in JavaScript (if node is installed), `shad0w train --student deep` builds a deep table
from the same log, and `shad0w proxy` serves the System One request shape from it. It exits non-zero on any mismatch.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import shad0w

REPO_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "index.js")
POS = ["good", "great", "lovely", "fun", "brilliant", "moving"]
NEG = ["bad", "dull", "awful", "boring", "messy", "weak"]
FILL = ["the film", "this movie", "the plot", "the acting", "it", "the ending"]


def cli() -> list[str]:
    """The installed `shad0w` command. Not `python -m shad0w`: this script works in a folder that has a `shad0w/` data
    folder, and with the current folder first on the import path, `-m` would import that folder instead of the package."""
    exe = os.path.join(os.path.dirname(sys.executable), "shad0w.exe" if os.name == "nt" else "shad0w")
    return [exe] if os.path.exists(exe) else [shutil.which("shad0w") or "shad0w"]


def main(work: str) -> int:
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    shutil.rmtree("shad0w", ignore_errors=True)
    rng = random.Random(0)
    truth: dict[str, str] = {}
    calls = {"n": 0}
    failures = []

    def review():
        pos, neg = rng.random() < 0.5, rng.random() < 0.35
        t = f"{rng.choice(FILL)} was {'not ' if neg else ''}{rng.choice(POS if pos else NEG)} " \
            f"{rng.choice(['really', 'honestly', 'overall', ''])}".strip()
        truth[t] = "positive" if pos != neg else "negative"
        return t

    def llm(text):
        calls["n"] += 1
        return truth[text]

    sent = shad0w.decision("sentiment", options={"positive": "liked it", "negative": "did not"}, llm=llm, audit_rate=0.0)
    for _ in range(3000):
        sent(review())
    cert = sent.train()
    print(f"1. logged {sent.log_rows()} answers; trained a {cert['student']} table, certified "
          f"{cert['certified_share_on_calibration']:.1%} of calibration traffic")
    if not sent.model.native:
        print("   (the C core is not available: serving through numpy)")
    before = calls["n"]
    test = [review() for _ in range(2000)]
    ds = [sent(t) for t in test]
    served = [(d, truth[t]) for d, t in zip(ds, test) if d.source == "table"]
    wrong = sum(d.answer != y for d, y in served)
    lat = sorted(d.latency_us for d, _ in served) or [float("nan")]
    print(f"2. served {len(served) / len(test):.1%} locally, disagreement {wrong / max(1, len(served)):.1%}, "
          f"p50 {lat[len(lat) // 2]:.1f} µs; LLM calls {calls['n'] - before}")
    if served and wrong / len(served) > 0.05:
        failures.append("served disagreement above alpha")

    bundles = ["shad0w/sentiment/bundle"]
    shad0w_cli = cli()
    r = subprocess.run(shad0w_cli + ["train", "--log", "shad0w/sentiment/log.jsonl", "--out", "deep/sentiment/bundle",
                                     "--question", "sentiment", "--student", "deep"], capture_output=True, text=True)
    print(f"3. shad0w train --student deep: exit {r.returncode}")
    if r.returncode:
        failures.append("cli train failed: " + r.stderr[-300:])
    else:
        bundles.append("deep/sentiment/bundle")

    probe = test[:300] + ["the film was not good", "it was not bad", ""]
    for b in bundles:
        c_model = shad0w.load(b, native=True)
        py_model = shad0w.load(b, native=False)
        mism = sum(1 for t in probe
                   if [c_model.decide(t)["answers"]["sentiment"][k] for k in ("choice", "certified")]
                   != [py_model.decide(t)["answers"]["sentiment"][k] for k in ("choice", "certified")])
        line = f"4. {b}: hidden={py_model.questions['sentiment'].rx.hidden}; C vs numpy mismatches {mism}"
        if shutil.which("node") and os.path.exists(REPO_JS):
            json.dump(probe, open("probe.json", "w"))
            js = ("const {Bundle}=require(process.argv[1]);const fs=require('fs');(async()=>{const b=await "
                  "Bundle.load(process.argv[2]);const p=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));"
                  "console.log(JSON.stringify(p.map(t=>{const a=b.decide(t).answers.sentiment;return [a.choice,a.certified]})))})()")
            out = subprocess.run(["node", "-e", js, REPO_JS, b, "probe.json"], capture_output=True, text=True)
            if out.returncode:
                failures.append("node failed: " + out.stderr[-300:])
            else:
                jsd = json.loads(out.stdout)
                jm = sum(1 for t, j in zip(probe, jsd)
                         if [py_model.decide(t)["answers"]["sentiment"][k] for k in ("choice", "certified")] != j)
                line += f", JavaScript vs numpy mismatches {jm}"
                if jm:
                    failures.append(f"{b}: JS mismatches")
        print(line)
        if mism:
            failures.append(f"{b}: C mismatches")

    port = 18731
    proxy = subprocess.Popen(shad0w_cli + ["proxy", "--upstream", "http://127.0.0.1:9", "--port", str(port), "--dir", "deep"
                                           if len(bundles) > 1 else "shad0w"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        body = {"model": "any", "state": "the acting was not dull honestly",
                "questions": {"sentiment": {"type": "choice", "criteria": {"positive": "liked it", "negative": "did not"}}}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        try:
            a = json.loads(urllib.request.urlopen(req, timeout=5).read())["answers"]["sentiment"]
            print(f"5. proxy /v1/systemone: {a['choice']} from {a['shad0w']['source']} (certified {a['shad0w']['certified']})")
            if a["shad0w"]["source"] != "table":
                failures.append("proxy did not serve from the table")
        except urllib.error.HTTPError as e:
            failures.append(f"proxy HTTP {e.code}: {e.read()[:200]!r}")
    finally:
        proxy.terminate()
        proxy.communicate(timeout=5)
    print("OK" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="shad0w-e2e-")))
