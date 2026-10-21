"""Static word vectors as items (shad0w.static): C and numpy agree exactly, the table round-trips through a bundle,
and static="auto" keeps whichever variant certifies more (each certified at delta / 2)."""
import numpy as np
import pytest

from shad0w import native
from shad0w.features import _words, fnv1a, items
from shad0w.static import StaticTable

from .test_shadow import SCHEMA, synth

TEXTS = ["my card was stolen", "I want my money back please", "what is my balance", "", "!!!", "send money to a friend",
         "unknownword zzz qqq", "Card STOLEN again, block it now"]


def _table(texts, dims=16, seed=0):
    rng = np.random.default_rng(seed)
    words = sorted({w for t in texts for w in _words(t)})[:-1]  # leave one word unknown on purpose
    hs = np.array(sorted({fnv1a(b"w" + w) for w in words}), dtype=np.uint32)
    V = rng.integers(-127, 128, (len(hs), dims)).astype(np.int8)
    return StaticTable(hs, V, rng.normal(0, 30, (dims, 3)).astype(np.float64).cumsum(1), 4)


def test_items_shape_and_determinism():
    t = _table(TEXTS)
    for x in TEXTS:
        its = t.items(x)
        assert its.dtype == np.uint32 and len(its) == t.dims and (its < 2**22).all()
        assert np.array_equal(its, t.items(x))


@pytest.mark.skipif(not native.available(), reason="C core not built")
def test_c_and_numpy_agree(tmp_path):
    from shad0w.mathutil import softmax
    from shad0w.reflex import Reflex

    t = _table(TEXTS)
    t.save(str(tmp_path / "q.s0w"))
    assert np.array_equal(StaticTable.load(str(tmp_path / "q.s0w")).V, t.V)
    its = [np.concatenate([items(x), t.items(x)]) for x in TEXTS]
    vocab = {int(h): i for i, h in enumerate(sorted({int(h) for s in its for h in s}))}
    rng = np.random.default_rng(1)
    rx = Reflex.compile(vocab, rng.standard_normal((4, len(vocab))).astype(np.float32), np.zeros(4, np.float32), list("abcd"))
    rx.save(str(tmp_path / "q.s0"))
    nat = native.NativeReflex(str(tmp_path / "q.s0"), static=str(tmp_path / "q.s0w"))
    assert nat.has_static
    for x, it in zip(TEXTS, its):
        y, p, _ = nat.decide(x)
        assert y == int(np.argmax(rx.logits(it))) and np.allclose(p, softmax(rx.logits(it)), atol=1e-5)


def test_bundle_roundtrip_with_static(tmp_path):
    import shad0w
    from shad0w.api import Model, Question

    t = _table(TEXTS)
    from shad0w.shadow import shadow_compile
    rows, _ = synth(600, 3)
    m, _ = shadow_compile(SCHEMA, rows, alpha=0.05)
    q = m.questions["intent"]
    q.static = t  # attach a table, as training with static= does (the rx stays valid: unseen items score 0)
    m.save(str(tmp_path / "b"))
    m2 = shad0w.load(str(tmp_path / "b"))
    q2 = m2.questions["intent"]
    assert isinstance(q2.static, StaticTable) and np.array_equal(q2.static.hashes, t.hashes)
    for x in TEXTS[:4]:
        a = m.decide(x)["answers"]["intent"]
        b = m2.decide(x)["answers"]["intent"]
        assert a["choice"] == b["choice"] and abs(a["confidence"] - b["confidence"]) < 1e-5
    assert isinstance(Model, type) and isinstance(Question, type)


def test_static_auto_end_to_end():
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")
    from shad0w import static as st
    try:
        st._load_model("potion-8M")
    except Exception as e:  # not cached and offline: nothing to test here
        pytest.skip(f"potion-8M not available: {e}")
    from shad0w.shadow import shadow_compile
    rows, _ = synth(1500, 4, teacher_noise=0.01)
    m, cert = shadow_compile(SCHEMA, rows, alpha=0.05, static="auto")
    c = cert["questions"]["intent"]
    assert c["static"] in (None, "potion-8M") and "static=auto chosen on held-out fitting rows" in c["procedure"]
    assert c["certified_share_on_calibration"] > 0.5


def test_soft_labels_are_used_for_fitting_only(tmp_path):
    """Rows that carry the teacher's probabilities are fitted on them; the certificate still uses hard answers."""
    import json

    import shad0w
    from shad0w.shadow import _soft_targets, shadow_compile
    rows, _ = synth(1500, 8, teacher_noise=0.01)
    opts = list(SCHEMA["intent"]["criteria"])
    soft = [{**r, "probs": {o: (0.85 if o == r["intent"] else 0.05) for o in opts}} for r in rows]
    Y = _soft_targets(soft[:3], opts, np.array([opts.index(r["intent"]) for r in soft[:3]]))
    assert Y.shape == (3, 4) and np.allclose(Y.sum(1), 1) and (Y.argmax(1) == [opts.index(r["intent"]) for r in soft[:3]]).all()
    assert _soft_targets(rows[:3], opts, np.zeros(3, int)) is None
    m_hard, c_hard = shadow_compile(SCHEMA, rows, alpha=0.05)
    m_soft, c_soft = shadow_compile(SCHEMA, soft, alpha=0.05)
    assert c_soft["questions"]["intent"]["certified_share_on_calibration"] > 0.5
    # a teacher that reports probabilities gets them logged
    def teacher(text):
        from .mock_llm import classify
        a = classify(text)
        return {"answer": a, "probabilities": {o: (0.9 if o == a else 0.1 / 3) for o in opts}}
    sh = shad0w.Shadow(None, teacher=teacher, question="intent", log=str(tmp_path / "l.jsonl"), schema=SCHEMA["intent"])
    assert sh("block my card please").answer == "lost_card"
    row = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert row["intent"] == "lost_card" and abs(row["probs"]["lost_card"] - 0.9) < 1e-6


def test_synthesize_writes_fit_only_rows(tmp_path):
    import json

    import shad0w
    from shad0w.llm import LLMTeacher
    from shad0w.shadow import read_jsonl

    from .test_shadow import VOCAB

    def complete(msgs):
        prompt = msgs[-1]["content"]
        opt = next(o for o in VOCAB if f'"{o}"' in prompt)
        if "Write" in prompt:  # generation: one message per line, with list noise the parser strips
            return "\n".join(f"{i}. please {VOCAB[opt][i % len(VOCAB[opt])]} today" for i in range(1, 6))
        return opt
    t = LLMTeacher(SCHEMA, "openai/x", question="intent", api_key="sk-test", complete=complete)
    sh = shad0w.Shadow(None, teacher=t, question="intent", log=str(tmp_path / "l.jsonl"), schema=SCHEMA["intent"])
    rep = sh.synthesize(per_option=5)
    assert rep["written"] == 20 and set(rep["per_option"].values()) == {5}
    rows = read_jsonl(str(tmp_path / "l.jsonl"))
    assert all(r["source"] == "synthetic" and not r["text"][0].isdigit() for r in rows)
    real, _ = synth(1200, 9, teacher_noise=0.01)
    with open(tmp_path / "l.jsonl", "a") as f:
        for r in real:
            f.write(json.dumps({**r, "source": "teacher"}) + "\n")
    cert = sh.train(out=str(tmp_path / "b"), min_rows=100)
    assert cert["n_fit_only"] == 20 and cert["n_records"] == 1200  # synthetic rows never reach calibration
