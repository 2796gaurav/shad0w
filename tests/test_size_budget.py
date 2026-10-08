"""max_mb keeps a table within its size budget and still certifies; the JavaScript fast path (no per-option
probabilities, hash-table lookup) gives exactly the same answers and confidences as the full path."""
import json
import os
import shutil
import subprocess

import pytest

from shad0w.compiler.distill import max_features_for
from shad0w.shadow import shadow_compile

from .test_shadow import SCHEMA, synth

ROOT = os.path.dirname(os.path.dirname(__file__))


def table_bytes(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".s0"))


def test_budget_formula_matches_the_file_format():
    for k, mb in ((4, 0.05), (77, 1.5), (500, 4)):
        f = max_features_for(mb, k)
        assert 20 + 4 * f + f * k + 8 * k + 4 * k * k <= mb * 2**20 or f == 1000
    assert max_features_for(None, 10) is None


def test_max_mb_caps_the_table_and_still_certifies(tmp_path):
    import random
    rng = random.Random(0)
    noise = ["".join(rng.choice("bcdfghjklmnprstvz") + rng.choice("aeiou") for _ in range(3)) for _ in range(4000)]
    rows, _ = synth(3000, 61, teacher_noise=0.01)
    rows = [{**r, "text": r["text"] + " " + " ".join(rng.sample(noise, 3))} for r in rows]  # a large vocabulary
    full, cf = shadow_compile(SCHEMA, rows, alpha=0.05)
    small, cs = shadow_compile(SCHEMA, rows, alpha=0.05, max_mb=0.01)  # 0.01 MB: forces pruning to 1,000 rows
    full.save(str(tmp_path / "full"))
    small.save(str(tmp_path / "small"))
    assert table_bytes(tmp_path / "small") < table_bytes(tmp_path / "full")
    assert small.questions["intent"].rx.keys.shape[0] == max_features_for(0.01, 4)
    assert cs["questions"]["intent"]["certified_share_on_calibration"] > 0.5
    for t in [r["text"] for r in synth(50, 62)[0]]:  # pruning keeps the decisions sensible
        assert small.decide(t)["answers"]["intent"]["choice"] in SCHEMA["intent"]["criteria"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_fast_path_equals_full_path(tmp_path):
    rows, _ = synth(3000, 63, teacher_noise=0.01)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    m.save(str(tmp_path / "b"))
    probe = [r["text"] for r in synth(300, 64)[0]] + ["", "ünïcödé text", "A" * 300 + " tail"]
    (tmp_path / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    script = """
const {Bundle} = require(process.argv[1]); const fs = require("fs");
(async () => {
  const b = await Bundle.load(process.argv[2]); const p = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
  const out = p.map((t) => { const a = b.decide(t).answers.intent, f = b.decide(t, {probabilities: false}).answers.intent;
    return [a.choice === f.choice, a.confidence === f.confidence, a.certified === f.certified, f.probabilities === undefined]; });
  console.log(JSON.stringify(out));
})();
"""
    r = subprocess.run(["node", "-e", script, os.path.join(ROOT, "js", "index.js"), str(tmp_path / "b"), str(tmp_path / "probe.json")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert all(all(x) for x in json.loads(r.stdout))
