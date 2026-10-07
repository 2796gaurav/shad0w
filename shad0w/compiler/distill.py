"""Distiller + certifier: (texts, labels) -> compiled Reflex artifact + correctness head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from shad0w.features import items_batch
from shad0w.mathutil import softmax
from shad0w.reflex import Reflex, bag_matrix, build_vocab, fit_lr
from shad0w.reliability import CorrectnessHead, confidence_features, fit_temperature


@dataclass
class Compiled:
    rx: Reflex
    head: CorrectnessHead
    C: float

    def predict(self, texts):
        its = items_batch(texts)
        p = np.stack([softmax(self.rx.logits(i)) for i in its])
        c = self.head(confidence_features(p, n_tokens=np.array([len(i) for i in its])))
        return p, c, its


def compile_schema(texts, y, labels, C: float | None = None, seed: int = 0,
                   C_grid=(100.0, 300.0, 1000.0, 3000.0), idf: str | bool = "auto") -> Compiled:
    """idf: fold inverse document frequency into the table (T'[h] = idf_h * T[h]); runtime and
    the exact radius are unchanged. 'auto' enables it when texts average > 100 items
    (long documents, where plain term frequency is swamped by common items)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    K = len(labels)
    its = items_batch(texts)
    long_docs = np.mean([len(i) for i in its]) > 100
    use_idf = long_docs if idf == "auto" else bool(idf)
    vocab = build_vocab(its, min_count=2 if long_docs else 1)
    X = bag_matrix(its, vocab)
    w_idf = None
    if use_idf:
        from scipy.sparse import diags
        df = np.zeros(len(vocab))
        for s in its:
            for h in set(s.tolist()):
                j = vocab.get(h)
                if j is not None:
                    df[j] += 1
        w_idf = (np.log((1 + len(its)) / (1 + df)) + 1).astype(np.float32)
        X = X @ diags(w_idf)
        if C is None and long_docs:
            C_grid = (1e3, 1e4, 1e5, 1e6)
    if C is None:
        hold = rng.random(len(y)) < 0.15
        scores = []
        for c in C_grid:
            W, b = fit_lr(X[~hold], y[~hold], c, K=K)
            scores.append(((X[hold] @ W.T + b).argmax(1) == y[hold]).mean())
        C = C_grid[int(np.argmax(scores))]
    oof = np.zeros((len(y), K), np.float32)
    n_splits = max(2, min(5, np.bincount(y, minlength=K).min()))
    for a, b_ in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        W, b = fit_lr(X[a], y[a], C, K=K)  # always K outputs, even if a class is absent here
        oof[b_] = X[b_] @ W.T + b
    T = fit_temperature(oof, y)
    W, b = fit_lr(X, y, C, K=K)
    p_oof = softmax(oof / T)
    head = CorrectnessHead().fit(confidence_features(p_oof, n_tokens=np.array([len(i) for i in its])),
                                 p_oof.argmax(1) == y)
    if w_idf is not None:
        W = W * w_idf[None, :]  # fold idf into the table: runtime stays sum(T[h]) / n
    return Compiled(Reflex.compile(vocab, W, b, labels, temperature=T), head, C)
