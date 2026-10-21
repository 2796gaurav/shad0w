"""The deep student (table format 2: int8 rows -> hidden layer -> dense): Python, C and JavaScript agree, the
loader refuses corrupt files, the radius is honestly absent, and `student="auto"` picks it only when it earns it."""
import os
import random
import shutil
import struct
import subprocess

import numpy as np
import pytest

from shad0w.api import _load_reflex
from shad0w.compiler.distill import compile_schema, fit_dan, max_features_for
from shad0w.features import items
from shad0w.mathutil import softmax
from shad0w.reflex import Reflex, build_vocab

ROOT = os.path.dirname(os.path.dirname(__file__))
LABELS = ["refund", "lost_card", "balance"]


def xor_texts(n=2000, seed=0):
    """A task a linear table cannot learn: the label is the XOR of two cue words, so no additive weighting works
    (every unigram and the cue bigram are equally frequent under both labels)."""
    rng = random.Random(seed)
    texts, y = [], []
    fill = ["please", "hello", "today", "thanks", "quickly", "again", "now", "ok"]
    for _ in range(n):
        a, b = rng.randrange(2), rng.randrange(2)
        w1 = ["alpha", "beta"][a]
        w2 = ["gamma", "delta"][b]
        mid = " ".join(rng.choice(fill) for _ in range(rng.randrange(1, 4)))
        texts.append(f"{w1} {mid} {w2} {rng.choice(fill)}")
        y.append(a ^ b)
    return texts, np.array(y)


@pytest.fixture(scope="module")
def deep():
    texts, y = xor_texts()
    return texts, y, compile_schema(texts, y, ["no", "yes"], student="deep", hidden=16, seed=0)


def test_deep_learns_what_linear_cannot(deep):
    texts, y, comp = deep
    assert comp.student == "deep" and comp.rx.hidden == 16 and comp.rx.version == 2
    lin = compile_schema(texts, y, ["no", "yes"], student="linear", seed=0)
    probe, py = xor_texts(400, seed=1)
    acc_deep = np.mean([int(np.argmax(comp.rx.logits(items(t)))) == k for t, k in zip(probe, py)])
    acc_lin = np.mean([int(np.argmax(lin.rx.logits(items(t)))) == k for t, k in zip(probe, py)])
    assert acc_deep > 0.95 > 0.7 > acc_lin
    assert comp.choice["chosen"] == "deep" and comp.choice["deep"]["coverage"] > 0.9


def test_auto_prefers_linear_when_it_is_as_good():
    rng = random.Random(0)
    words = {0: ["refund", "money back"], 1: ["lost card", "stolen"], 2: ["balance", "funds"]}
    texts, y = [], []
    for _ in range(900):
        k = rng.randrange(3)
        texts.append(f"{rng.choice(['hi', 'please'])} {rng.choice(words[k])} {rng.choice(['ok', 'thanks'])}")
        y.append(k)
    comp = compile_schema(texts, np.array(y), LABELS, student="auto", hidden=16)
    assert comp.student == "linear" and comp.rx.version == 1 and comp.rx.hidden == 0
    assert set(comp.choice) == {"linear", "deep", "chosen"}
    with pytest.raises(ValueError):
        compile_schema(texts, np.array(y), LABELS, student="huge")


def test_roundtrip_and_native_and_js_agree(deep, tmp_path):
    texts, y, comp = deep
    path = tmp_path / "t.s0"
    comp.rx.save(str(path))
    rx2 = _load_reflex(str(path))
    assert rx2.hidden == 16 and rx2.W2.shape == (16, 2) and rx2.G is None
    probe = xor_texts(300, seed=2)[0] + ["", "UPPER!!", "ünïcödé", "a" * 400]
    for t in probe:
        its = items(t if t else " ")
        np.testing.assert_allclose(rx2.logits(its), comp.rx.logits(its), rtol=0, atol=1e-6)
    assert np.isnan(comp.rx.radius(items("alpha gamma")))
    from shad0w.native import NativeReflex, available
    if available():
        nat = NativeReflex(str(path))
        assert nat.hidden == 16
        for t in probe:
            yn, p, r = nat.decide(t if t else " ")
            p_py = softmax(comp.rx.logits(items(t if t else " ")))
            assert yn == int(np.argmax(p_py))
            np.testing.assert_allclose(p, p_py, atol=1e-5)
            assert np.isnan(r)
    if shutil.which("node"):
        preds = [int(np.argmax(comp.rx.logits(items(t if t else " ")))) for t in probe]
        (tmp_path / "texts.txt").write_text("\n".join(t if t else " " for t in probe), encoding="utf-8")
        (tmp_path / "preds.txt").write_text("\n".join(map(str, preds)), encoding="utf-8")
        r = subprocess.run(["node", os.path.join(ROOT, "js", "bench.js"), str(path), str(tmp_path / "texts.txt"),
                            str(tmp_path / "preds.txt")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert '"parity":1' in r.stdout, r.stdout


@pytest.mark.parametrize("mutate", ["version", "hidden", "truncate", "pad"])
def test_corrupt_deep_tables_are_refused(deep, tmp_path, mutate):
    comp = deep[2]
    p = tmp_path / "t.s0"
    comp.rx.save(str(p))
    data = bytearray(p.read_bytes())
    if mutate == "version":
        data[4:8] = struct.pack("<I", 3)
    elif mutate == "hidden":
        data[20:24] = struct.pack("<I", 5000)
    elif mutate == "truncate":
        data = data[:-5]
    elif mutate == "pad":
        data += b"\0" * 2
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        _load_reflex(str(p))
    from shad0w.native import NativeReflex, available
    if available():
        with pytest.raises(OSError):
            NativeReflex(str(p))


def test_deep_table_in_a_bundle_never_claims_a_radius(deep, tmp_path):
    from shad0w import load
    from shad0w.api import DriftGuard, Model, Question
    texts, y, comp = deep
    q = Question("q", "yesno", ["no", "yes"], comp.rx, 0.0, 0.05, DriftGuard(np.ones(2), np.ones(2, bool)), r_min=2.0)
    m = Model(questions={"q": q})
    m.save(str(tmp_path / "b"))
    m2 = load(str(tmp_path / "b"), native=False)
    r = m2.decide("alpha please gamma")["answers"]["q"]
    assert r["radius"] is None and r["certified"]
    r = m2.decide("alpha please gamma", exposed=True)["answers"]["q"]
    assert r["flag"] == "low_radius" and not r["certified"]  # no radius: never robust enough for an exposed channel


def test_compile_handles_soft_targets_and_size_budget():
    texts, y = xor_texts(1200)
    soft = np.eye(2, dtype=np.float32)[y] * 0.9 + 0.05
    comp = compile_schema(texts, y, ["no", "yes"], student="deep", hidden=8, soft=soft, max_features=1000)
    assert comp.rx.version == 2 and len(comp.rx.keys) <= 1000
    assert max_features_for(1.0, 2, hidden=8) < max_features_for(1.0, 2)


def test_numpy_fallback_matches_shape():
    from shad0w.compiler.distill import _fit_dan_numpy
    texts, y = xor_texts(600)
    its = [items(t) for t in texts]
    vocab = build_vocab(its)
    idx = np.arange(len(y))
    W1, b1, W2, b2, ep, _ = _fit_dan_numpy(its, vocab, y, 2, 8, idx[:500], idx[500:], 60, 3e-3, 0.0, 0.1, 0, 10)
    assert W1.shape == (8, len(vocab)) and W2.shape == (8, 2) and b2.shape == (2,)
    rx = Reflex.compile(vocab, W1, b1, ["no", "yes"], W2=W2, b2=b2)
    acc = np.mean([int(np.argmax(rx.logits(its[i]))) == y[i] for i in idx[500:]])
    assert acc > 0.8
    torch_out = fit_dan(its, vocab, y, 2, 8, idx[:500], idx[500:])
    assert torch_out[0].shape == W1.shape
