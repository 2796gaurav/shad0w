"""Regressions for the independent review of the deep table, the student choice and the JS loader."""
import json
import os
import random
import shutil
import struct
import subprocess

import numpy as np
import pytest

from shad0w.compiler.distill import compile_schema
from shad0w.features import items
from shad0w.mathutil import softmax

ROOT = os.path.dirname(os.path.dirname(__file__))


def texts_k(n=1500, k=4, seed=0):
    rng = random.Random(seed)
    words = [[f"w{c}a", f"w{c}b", f"w{c}c"] for c in range(k)]
    t, y = [], []
    for _ in range(n):
        c = rng.randrange(k)
        t.append(f"hi {rng.choice(words[c])} {rng.choice(['ok', 'now', 'pls'])}")
        y.append(c)
    return t, np.array(y)


def test_an_unused_option_keeps_the_held_out_slice():
    t, y = texts_k()
    a = compile_schema(t, y, list("abcd"), student="auto", hidden=8)
    b = compile_schema(t, y, list("abcde"), student="auto", hidden=8)  # "e" never answered
    # the choice is scored on the same held-out rows in both cases, not on training rows
    assert a.choice["linear"]["agreement"] == pytest.approx(b.choice["linear"]["agreement"], abs=0.02)


def test_fit_only_rows_do_not_count_toward_min_rows(tmp_path):
    import shad0w
    from tests.test_shadow import SCHEMA, synth
    log = tmp_path / "l.jsonl"
    rows, _ = synth(150, 1)
    with open(log, "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "source": "teacher"}) + "\n")
        for r in synth(1150, 2)[0]:
            f.write(json.dumps({**r, "source": "distil"}) + "\n")
    sh = shad0w.Shadow(None, teacher=lambda x: "balance", question="intent", log=str(log), schema=SCHEMA["intent"])
    with pytest.raises(ValueError, match="fit-only"):
        sh.train(out=str(tmp_path / "b"), min_rows=1000)


@pytest.mark.parametrize("student", ["linear", "deep"])
def test_static_vectors_respect_the_size_budget(student):
    t, y = texts_k(1200)
    rng = np.random.default_rng(0)
    from shad0w.features import MASK, fnv1a
    dense = {fnv1a(b"w" + f"extra{i}".encode()) & MASK: rng.normal(0, 1, 4).astype(np.float32) for i in range(3000)}
    c = compile_schema(t, y, list("abcd"), student=student, hidden=8, max_features=1000, dense=dense)
    assert len(c.rx.keys) <= 1000


def test_deep_table_python_and_c_agree_bit_for_bit(tmp_path):
    from shad0w.native import NativeReflex, available
    if not available():
        pytest.skip("C core not built")
    t, y = texts_k(2000, 6)
    c = compile_schema(t, y, list("abcdef"), student="deep", hidden=32)
    c.rx.save(str(tmp_path / "t.s0"))
    nat = NativeReflex(str(tmp_path / "t.s0"))
    probe = t[:500] + ["", "zz unknown", "w1a w2b w3c w4a"]
    worst = 0
    for x in probe:
        i, p, _ = nat.decide(x)
        q = softmax(c.rx.logits(items(x))).astype(np.float32)
        assert i == int(np.argmax(q)), x
        # the scores are summed in the same order (bit-identical logits); the final exp is the platform's (glibc expf,
        # Apple's, numpy's SIMD exp), which may round the last bit differently, so allow 2 ulp on the probabilities
        worst = max(worst, int(np.max(np.abs(p.view(np.int32).astype(np.int64) - q.view(np.int32).astype(np.int64)))))
    assert worst <= 2, worst


def test_empty_item_bag_scores_like_c():
    t, y = texts_k(600)
    c = compile_schema(t, y, list("abcd"), student="linear")
    assert np.isfinite(c.rx.logits(np.zeros(0, np.uint32))).all()
    assert np.array_equal(c.rx.logits(np.zeros(0, np.uint32)), c.rx.logits(items("")))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("mutate", ["unsorted", "big_key", "nan_scale", "inf_bias", "nan_tail"])
def test_js_refuses_tables_python_and_c_refuse(tmp_path, mutate):
    t, y = texts_k(600)
    c = compile_schema(t, y, list("abcd"), student="linear")
    p = tmp_path / "t.s0"
    c.rx.save(str(p))
    d = bytearray(p.read_bytes())
    F, K = struct.unpack_from("<II", d, 8)
    keys_at, scale_at = 20, 20 + 4 * F + F * K
    if mutate == "unsorted":
        d[keys_at:keys_at + 8] = d[keys_at + 4:keys_at + 8] + d[keys_at:keys_at + 4]
    elif mutate == "big_key":
        struct.pack_into("<I", d, keys_at + 4 * (F - 1), 1 << 22)
    elif mutate == "nan_scale":
        struct.pack_into("<f", d, scale_at, float("nan"))
    elif mutate == "inf_bias":
        struct.pack_into("<f", d, scale_at + 4 * K, float("inf"))
    else:
        struct.pack_into("<f", d, len(d) - 4, float("nan"))
    p.write_bytes(bytes(d))
    from shad0w.api import _load_reflex
    with pytest.raises(ValueError):
        _load_reflex(str(p))
    js = ("const {Table}=require(process.argv[1]);const fs=require('fs');"
          "try{new Table(fs.readFileSync(process.argv[2]));console.log('loaded')}catch(e){console.log('refused')}")
    r = subprocess.run(["node", "-e", js, os.path.join(ROOT, "js", "index.js"), str(p)], capture_output=True, text=True)
    assert r.stdout.strip() == "refused", r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_lone_surrogate_hashes_like_python():
    probe = ["ab\ud800cd", "x\udc00y", "ok 😀 fine", "\ud800"]
    py = [sorted(int(h) for h in items(x)) for x in probe]
    js = ("const {items}=require(process.argv[1]);const p=JSON.parse(process.argv[2]);"
          "console.log(JSON.stringify(p.map(t=>Array.from(items(t)).sort((a,b)=>a-b))))")
    r = subprocess.run(["node", "-e", js, os.path.join(ROOT, "js", "index.js"), json.dumps(probe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == py
