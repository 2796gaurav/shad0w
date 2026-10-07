"""The JavaScript runtime must make exactly the same decisions as Python on the same .s0 file (skipped without node)."""
import json
import os
import pathlib
import random
import shutil
import subprocess

import numpy as np
import pytest

from shad0w.compiler.distill import compile_schema
from shad0w.features import items

ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python(tmp_path):
    rng = random.Random(0)
    words = {0: ["refund", "money back", "reimburse"], 1: ["lost card", "stolen", "block card"], 2: ["balance", "funds", "total"]}
    texts, y = [], []
    for _ in range(600):
        k = rng.randrange(3)
        texts.append(f"{rng.choice(['hi', 'please', 'now'])} {rng.choice(words[k])} {rng.choice(['ok', 'thanks', 'asap'])} नमस्ते")
        y.append(k)
    comp = compile_schema(texts, np.array(y), ["refund", "lost_card", "balance"])
    table = tmp_path / "t.s0"
    comp.rx.save(str(table))
    probe = texts[:200] + ["", "UPPER case & punctuation!!", "ünïcödé text", "a" * 400]
    (tmp_path / "texts.txt").write_text("\n".join(t if t else " " for t in probe), encoding="utf-8")
    preds = [int(np.argmax(comp.rx.logits(items(t if t else " ")))) for t in probe]
    (tmp_path / "preds.txt").write_text("\n".join(map(str, preds)), encoding="utf-8")
    r = subprocess.run(["node", os.path.join(ROOT, "js", "bench.js"), str(table), str(tmp_path / "texts.txt"),
                        str(tmp_path / "preds.txt")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '"parity":1' in r.stdout, r.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_bundle_matches_python_certified_flags(tmp_path):
    from shad0w import load
    from shad0w.shadow import shadow_compile

    from .test_shadow import SCHEMA, synth

    rows, _ = synth(3000, 21, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(tmp_path / "b"))
    probe = [r["text"] for r in synth(150, 22)[0]] + ["", "weather in paris", "ünïcode café"]
    (tmp_path / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    script = ("const {Bundle}=require(process.argv[1]);const fs=require('fs');(async()=>{const b=await Bundle.load(process.argv[2]);"
              "const p=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));"
              "console.log(JSON.stringify(p.map(t=>{const a=b.decide(t).answers.intent;return [a.choice,a.certified]})))})()")
    r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js"), str(tmp_path / "b"), str(tmp_path / "probe.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    js = json.loads(r.stdout)
    py = load(str(tmp_path / "b"), native=False)
    for t, (choice, cert) in zip(probe, js):
        a = py.decide(t)["answers"]["intent"]
        assert a["choice"] == choice and a["certified"] == cert, t
    assert any(c for _, c in js), "some answers should be certified"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_esm_entry_point():
    mjs = pathlib.Path(ROOT, "js", "index.mjs").resolve().as_uri()  # a file URL, so Windows paths work too
    code = f"import {{Bundle, items}} from '{mjs}'; console.log(typeof Bundle, items('hi').length)"
    r = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["function", "3"]  # "hi": one word item + two character trigrams


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_esm_build_is_in_sync():
    mjs = pathlib.Path(ROOT, "js", "index.mjs")
    before = mjs.read_text(encoding="utf-8")
    r = subprocess.run(["node", os.path.join(ROOT, "js", "build.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert mjs.read_text(encoding="utf-8") == before, "js/index.mjs was stale: run `node js/build.mjs` and commit it"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_shadow_and_openai_teacher(tmp_path):
    """JS decision() against the mock LLM: log-only, then a trained bundle answers locally; matchOption parity."""
    from shad0w.llm import match_option
    from shad0w.shadow import shadow_compile

    from .mock_llm import MockLLM
    from .test_shadow import SCHEMA, synth

    rows, _ = synth(3000, 41, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(tmp_path / "b"))
    mock = MockLLM("chatty")
    try:
        script = """
const {decision, matchOption} = require(process.argv[1]);
(async () => {
  const rows = [];
  const opts = {options: ["refund","lost_card","balance","transfer"], llm: "mock-1", baseURL: process.argv[2],
                log: (r) => rows.push(r), auditRate: 0};
  const before = await decision("intent", {...opts, bundle: process.argv[3] + "/missing"});
  const a = await before.decide("hi block my card thanks");
  const after = await decision("intent", {...opts, bundle: process.argv[3]});
  const b = await after.decide("hi block my card thanks");
  const probes = JSON.parse(process.argv[4]);
  console.log(JSON.stringify({a, b, rows, m: probes.map(p => matchOption(p, ["refund","lost_card","balance","transfer"]))}));
})().catch(e => { console.error(e); process.exit(1); });
"""
        probes = ["lost_card", " Lost Card. ", '{"answer": "refund"}', "The answer is transfer.", "refund or transfer", "nope"]
        r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js"), mock.url, str(tmp_path / "b"), json.dumps(probes)],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out["a"]["source"] == "teacher" and out["a"]["answer"] == "lost_card" and out["a"]["flag"] == "no_bundle"
        assert out["b"]["source"] == "table" and out["b"]["answer"] == "lost_card"
        assert out["rows"] == [{"text": "hi block my card thanks", "intent": "lost_card", "source": "teacher", "ts": out["rows"][0]["ts"]}]
        assert out["m"] == [match_option(p, ["refund", "lost_card", "balance", "transfer"]) for p in probes]
        assert "response_format" in mock.requests[0]["body"]
    finally:
        mock.close()
