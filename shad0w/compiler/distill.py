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


# Above this many table cells (features x options), training picks C on a subsample (the full grid, so quality holds)
# and uses 3 folds instead of 5 for the temperature: the fits dominate training time on many-option questions.
LARGE_PROBLEM = 2_000_000
C_SEARCH_ROWS = 4000


def max_features_for(max_mb: float | None, n_options: int) -> int | None:
    """How many feature rows fit a table of `max_mb` megabytes. The .s0 file is a 20-byte header, then per row a
    4-byte key and one int8 per option, then 8 bytes per option (scale, bias) and a K x K float32 matrix (for radii)."""
    if not max_mb:
        return None
    k = n_options
    fixed = 20 + 8 * k + 4 * k * k
    return max(1000, int((max_mb * 2**20 - fixed) / (k + 4)))


def compile_schema(texts, y, labels, C: float | None = None, seed: int = 0,
                   C_grid=(100.0, 300.0, 1000.0, 3000.0), idf: str | bool = "auto",
                   max_features: int | None = None) -> Compiled:
    """idf: fold inverse document frequency into the table (T'[h] = idf_h * T[h]); runtime and
    the exact radius are unchanged. 'auto' enables it when texts average > 100 items
    (long documents, where plain term frequency is swamped by common items).

    max_features: keep only this many feature rows (the ones with the largest weight for any option) and refit on
    them, so the table fits a size budget. The certificate is computed afterwards on the final table, so it stays
    valid; a smaller table usually certifies a little less traffic."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    K = len(labels)
    its = items_batch(texts)
    long_docs = np.mean([len(i) for i in its]) > 100
    use_idf = long_docs if idf == "auto" else bool(idf)
    vocab = build_vocab(its, min_count=2 if long_docs else 1)
    if max_features and len(vocab) > 4 * max_features:
        # a size budget far below the vocabulary: keep the most frequent items before fitting, so memory stays
        # bounded (W is options x features), then prune to the budget by weight after the fit
        df = {}
        for s_ in its:
            for h in set(s_.tolist()):
                if h in vocab:
                    df[h] = df.get(h, 0) + 1
        top = sorted(sorted(df, key=lambda h: -df[h])[:4 * max_features])
        vocab = {h: i for i, h in enumerate(top)}
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
    large = len(vocab) * K > LARGE_PROBLEM
    if C is None:
        hold = rng.random(len(y)) < 0.15
        fit_idx, hold_idx = np.flatnonzero(~hold), np.flatnonzero(hold)
        if large and len(fit_idx) > C_SEARCH_ROWS:
            fit_idx = np.sort(rng.choice(fit_idx, C_SEARCH_ROWS, replace=False))
        scores = []
        for c in C_grid:
            W, b = fit_lr(X[fit_idx], y[fit_idx], c * len(y) / max(1, len(fit_idx)) if large else c, K=K)
            scores.append(((X[hold_idx] @ W.T + b).argmax(1) == y[hold_idx]).mean())
        C = C_grid[int(np.argmax(scores))]
    oof = np.zeros((len(y), K), np.float32)
    n_splits = max(2, min(3 if large else 5, np.bincount(y, minlength=K).min()))
    for a, b_ in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        W, b = fit_lr(X[a], y[a], C, K=K)  # always K outputs, even if a class is absent here
        oof[b_] = X[b_] @ W.T + b
    T = fit_temperature(oof, y)
    W, b = fit_lr(X, y, C, K=K)
    if max_features and len(vocab) > max_features:
        keep = np.sort(np.argsort(-np.abs(W).max(0))[:max_features])  # columns stay in ascending-hash order
        col_hash = np.empty(len(vocab), dtype=np.int64)
        for h, j in vocab.items():
            col_hash[j] = h
        vocab = {int(col_hash[j]): i for i, j in enumerate(keep)}
        X = X[:, keep]
        if w_idf is not None:
            w_idf = w_idf[keep]
        W, b = fit_lr(X, y, C, K=K)
    p_oof = softmax(oof / T)
    head = CorrectnessHead().fit(confidence_features(p_oof, n_tokens=np.array([len(i) for i in its])),
                                 p_oof.argmax(1) == y)
    if w_idf is not None:
        W = W * w_idf[None, :]  # fold idf into the table: runtime stays sum(T[h]) / n
    return Compiled(Reflex.compile(vocab, W, b, labels, temperature=T), head, C)
