
import numpy as np
import pytest

from shad0w.features import items
from shad0w.mathutil import softmax
from shad0w.native import available as _native_available
from shad0w.reflex import Reflex, bag_matrix, build_vocab, fit_lr
from shad0w.reliability import SelectiveRiskController, conformal_sets, conformal_threshold

TEXTS = ["I lost my card!!", "Where is my REFUND?", "transfer 50€ to Ana", "", "   ", "don't charge me twice",
         "naïve café crème", "a", "x" * 400, "card card card", "Why was my top-up declined? (again)"]
LABELS = ["card", "refund", "transfer"]
Y = [0, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2]


@pytest.fixture(scope="module")
def reflex(tmp_path_factory):
    its = [items(t) for t in TEXTS]
    vocab = build_vocab(its)
    W, b = fit_lr(bag_matrix(its, vocab), np.array(Y), 10.0, K=3)
    rx = Reflex.compile(vocab, W, b, LABELS, temperature=1.3)
    path = str(tmp_path_factory.mktemp("a") / "m.s0")
    rx.save(path)
    return rx, path


@pytest.mark.skipif(not _native_available(), reason="C core not built")
def test_native_matches_python(reflex):
    from shad0w.native import NativeReflex
    rx, path = reflex
    nat = NativeReflex(path)
    for t in TEXTS + ["unseen words entirely", "Lost CARD refund??"]:
        y, p, r = nat.decide(t)
        pp = softmax(rx.logits(items(t)))
        assert y == int(pp.argmax())
        assert np.allclose(p, pp, atol=1e-5)
        assert r == rx.radius(items(t)) or (np.isinf(r) and np.isinf(rx.radius(items(t))))


def test_radius_is_exact(reflex):
    rx, _ = reflex
    Tf = rx.T.astype(np.float32) * rx.scale
    for t in TEXTS:
        its = items(t)
        r = rx.radius(its)
        if not np.isfinite(r) or r > 50:
            continue
        y = int(rx.logits(its).argmax())
        # the optimal attack: r copies of the best key for the binding rival flips; r-1 does not
        flipped = []
        for k in (int(r) - 1, int(r)):
            best = False
            for j in range(len(LABELS)):
                if j == y:
                    continue
                gains = Tf[:, j] - Tf[:, y]
                key = rx.keys[int(np.argmax(gains))] if gains.max() > 0 else np.uint32(0xFFFFFFFF)
                adv = np.concatenate([its, np.full(k, key, dtype=np.uint32)])
                best |= int(rx.logits(adv).argmax()) != y
            flipped.append(best)
        assert flipped == [False, True]


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    K, n = 10, 20000
    logits = rng.normal(size=(n, K)) * 2
    p = softmax(logits)
    y = np.array([rng.choice(K, p=pi) for pi in p])
    q = conformal_threshold(p[:5000], y[:5000], 0.1)
    S = conformal_sets(p[5000:], q)
    cov = S[np.arange(n - 5000), y[5000:]].mean()
    assert 0.88 <= cov <= 0.92


def test_selective_risk_guarantee():
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(200):
        conf = rng.random(3000)
        correct = rng.random(3000) < 0.6 + 0.4 * conf
        src = SelectiveRiskController(0.05, 0.1).fit(conf[:1500], correct[:1500])
        m = src.accept(conf[1500:])
        if m.any() and (~correct[1500:][m]).mean() > 0.05:
            hits += 1
    assert hits / 200 <= 0.15  # delta=0.1 plus sampling slack
