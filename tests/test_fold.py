"""Static word vectors folded into the table: the dense block trained alongside the items is written into the word
rows exactly, words with a vector but no training occurrence get rows, and both students support it."""
import random

import numpy as np
import pytest

from shad0w.compiler.distill import compile_schema
from shad0w.features import MASK, fnv1a, items

LABELS = ["a", "b", "c"]
TRAIN_WORDS = {0: ["apple", "pear", "plum"], 1: ["hammer", "wrench", "drill"], 2: ["river", "lake", "sea"]}
NEW_WORDS = {0: ["cherry", "mango"], 1: ["pliers", "saw"], 2: ["pond", "ocean"]}  # never in training texts


def wid(w: str) -> int:
    return fnv1a(b"w" + w.encode()) & MASK


def vectors(dim=8):
    """A vector per word that encodes its class in a fixed direction (like a static embedding would, roughly)."""
    rng = np.random.default_rng(0)
    out = {}
    for k in range(3):
        for w in TRAIN_WORDS[k] + NEW_WORDS[k]:
            v = rng.normal(0, 0.3, dim).astype(np.float32)
            v[k] += 2.0
            out[wid(w)] = v
    return out


def corpus(n=1500, seed=0):
    rng = random.Random(seed)
    fill = ["please", "now", "ok", "the", "my", "thanks"]
    texts, y = [], []
    for _ in range(n):
        k = rng.randrange(3)
        texts.append(f"{rng.choice(fill)} {rng.choice(TRAIN_WORDS[k])} {rng.choice(fill)}")
        y.append(k)
    return texts, np.array(y)


@pytest.mark.parametrize("student", ["linear", "deep"])
def test_fold_gives_unseen_words_their_meaning(student):
    texts, y = corpus()
    dense = vectors()
    comp = compile_schema(texts, y, LABELS, student=student, hidden=16, dense=dense)
    assert comp.student == student
    for h in dense:  # every word with a vector has a row, seen in training or not
        assert h in set(comp.rx.keys.tolist())
    probe = [(f"please {w} now", k) for k in range(3) for w in NEW_WORDS[k]]
    acc = np.mean([int(np.argmax(comp.rx.logits(items(t)))) == k for t, k in probe])
    assert acc == 1.0, acc
    plain = compile_schema(texts, y, LABELS, student=student, hidden=16)
    acc_plain = np.mean([int(np.argmax(plain.rx.logits(items(t)))) == k for t, k in probe])
    assert acc_plain < 0.7  # without the vectors, unseen words carry nothing


def test_fold_is_exact_for_the_linear_table():
    """Fitting with the dense block and serving the folded table give the same logits (up to int8 rounding)."""
    from scipy.sparse import csr_matrix, hstack

    from shad0w.compiler.distill import _fold
    from shad0w.reflex import bag_matrix, build_vocab, fit_lr
    texts, y = corpus(600, 1)
    dense = vectors(4)
    its = [items(t) for t in texts]
    vocab = build_vocab(its)
    X = bag_matrix(its, vocab)
    Zs = np.zeros((len(its), 4), np.float32)
    for r, s in enumerate(its):
        vs = [dense[int(h)] for h in s if int(h) in dense]
        if vs:
            Zs[r] = np.sum(vs, 0) / len(s)
    Xd = hstack([X, csr_matrix(Zs)]).tocsr()
    W, b = fit_lr(Xd, y, 100.0, K=3)
    z_model = Xd @ W.T + b
    vocab2, W2 = _fold(vocab, W[:, :len(vocab)], W[:, len(vocab):], dense)
    cols2 = np.array(sorted(vocab2, key=vocab2.get))
    X2 = bag_matrix(its, vocab2)
    z_fold = X2 @ W2.T + b
    assert list(cols2) == sorted(cols2)  # keys in ascending hash order, as the table format requires
    np.testing.assert_allclose(z_fold, z_model, atol=1e-4)
