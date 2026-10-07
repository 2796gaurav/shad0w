"""The shad0w table: a tokenizer-free, compiled, certified decision table.

Training fits a multinomial logistic model on mean-form bags of hashed items
(x_i = count_i / n_items), so the score is purely additive:

    z_j(x) = s_j * sum_{h in x} T[h, j] / n + b_j        T int8

which makes the exact insertion radius of `robust_cert` apply at item level.
`save()` writes the binary artifact consumed by csrc/reflex.c.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .features import items_batch
from .mathutil import softmax


def bag_matrix(item_lists, vocab: dict[int, int]):
    from scipy.sparse import csr_matrix

    rows, cols, vals = [], [], []
    for r, its in enumerate(item_lists):
        n = len(its)
        for h in its:
            c = vocab.get(int(h))
            if c is not None:
                rows.append(r)
                cols.append(c)
                vals.append(1.0 / n)
    return csr_matrix((vals, (rows, cols)), shape=(len(item_lists), len(vocab)), dtype=np.float32)


def build_vocab(item_lists, min_count: int = 1) -> dict[int, int]:
    counts: dict[int, int] = {}
    for its in item_lists:
        for h in set(its.tolist()):
            counts[h] = counts.get(h, 0) + 1
    keys = sorted(h for h, c in counts.items() if c >= min_count)
    return {h: i for i, h in enumerate(keys)}


def fit_lr_sklearn(X, y, C: float):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=3000, tol=1e-5)
    clf.fit(X, y)
    return clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)


def fit_lr(X, y, C: float, K: int | None = None, iters: int = 300):
    """Same objective as sklearn's L2 multinomial LR (sum CE + ||W||^2 / (2C)), solved
    with full-batch L-BFGS in torch on a sparse matrix: multithreaded and much faster."""
    import torch

    y = np.asarray(y)
    K = K or int(y.max()) + 1
    Xc = X.tocoo()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch: "sparse CSR support is in beta"
        Xt = torch.sparse_coo_tensor(np.vstack([Xc.row, Xc.col]), Xc.data.astype(np.float32),
                                     size=X.shape, check_invariants=False).coalesce().to_sparse_csr()
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Xt @ W + b
        loss = torch.nn.functional.cross_entropy(logits, yt, reduction="sum") + (W * W).sum() / (2 * C)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy().T.copy(), b.detach().numpy().copy()


@dataclass
class Reflex:
    keys: np.ndarray  # (F,) uint32 sorted bucket ids
    T: np.ndarray  # (F, K) int8
    scale: np.ndarray  # (K,) float32
    bias: np.ndarray  # (K,) float32
    temperature: float
    labels: list
    G: np.ndarray | None = None  # (K, K) max_h (T[h,j]-T[h,y]) * scale, for radii

    @classmethod
    def compile(cls, vocab, W, b, labels, temperature=1.0) -> Reflex:
        keys = np.array(sorted(vocab, key=vocab.get), dtype=np.uint32)
        Tf = W.T  # (F, K)
        scale = np.abs(Tf).max(0) / 127.0 + 1e-12
        T = np.clip(np.round(Tf / scale), -127, 127).astype(np.int8)
        r = cls(keys, T, scale.astype(np.float32), b.astype(np.float32), float(temperature), list(labels))
        r.G = r._pairwise()
        return r

    def _pairwise(self):
        Tf = self.T.astype(np.float32) * self.scale
        K = Tf.shape[1]
        G = np.zeros((K, K), np.float32)
        for y in range(K):
            G[y] = (Tf - Tf[:, y:y + 1]).max(0)
        return np.maximum(G, 0.0)  # inserting an unseen item (zero row) is always available

    @property
    def nbytes(self):
        return self.keys.nbytes + self.T.nbytes + self.scale.nbytes + self.bias.nbytes + self.G.nbytes

    def rows(self, its: np.ndarray):
        pos = np.searchsorted(self.keys, its)
        pos = np.minimum(pos, len(self.keys) - 1)
        hit = self.keys[pos] == its
        return pos[hit], len(its)

    def logits(self, its: np.ndarray) -> np.ndarray:
        pos, n = self.rows(its)
        acc = self.T[pos].sum(0, dtype=np.int32).astype(np.float32) * self.scale / n
        return (acc + self.bias) / self.temperature

    def radius(self, its: np.ndarray) -> float:
        """Exact minimum number of inserted items that flips the decision."""
        pos, n = self.rows(its)
        s = self.T[pos].sum(0, dtype=np.int32).astype(np.float32) * self.scale
        y = int(np.argmax(s / n + self.bias))
        D = s - s[y]
        db = self.bias - self.bias[y]
        den = self.G[y] + db
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.floor((-D - db * n) / den) + 1
        k[(den <= 0) | (np.arange(len(s)) == y)] = np.inf
        return float(k.min())

    def predict_proba(self, texts) -> np.ndarray:
        return np.stack([softmax(self.logits(i)) for i in items_batch(texts)])

    def save(self, path: str):
        K = len(self.bias)
        F = len(self.keys)
        with open(path, "wb") as f:
            f.write(b"S0RX")
            f.write(struct.pack("<IIIf", 1, F, K, self.temperature))
            f.write(self.keys.astype("<u4").tobytes())
            f.write(np.ascontiguousarray(self.T).tobytes())
            f.write(self.scale.astype("<f4").tobytes())
            f.write(self.bias.astype("<f4").tobytes())
            f.write(np.ascontiguousarray(self.G, dtype="<f4").tobytes())
