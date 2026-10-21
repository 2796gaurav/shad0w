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

from .features import MASK, fnv1a, items_batch
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


# Measured on an M-series laptop, 100 L-BFGS iterations: Fashion-MNIST ferns (0.84M weights) scipy 14 s vs torch 43 s;
# AG News (1.0M) 21 s vs 39 s; BANKING77 (1.9M weights, 77 options) 37 s vs 20 s. Torch wins once the multithreaded
# dense gradient (n x K) dominates.
TORCH_MIN_PARAMS = 1_500_000


def fit_lr(X, y, C: float, K: int | None = None, iters: int = 300, init=None):
    """Same objective as sklearn's L2 multinomial LR (sum CE + ||W||^2 / (2C)), solved with full-batch L-BFGS:
    in torch when it is installed and the table has >= TORCH_MIN_PARAMS weights (multithreaded), else in scipy.
    y: class indices (n,), or a (n, K) matrix of target probabilities (soft labels: the teacher's probabilities).
    init: optional (W, b) to start from (a nearby solution: another C, or the same data minus a fold). The objective is
    convex, so the optimum is the same; a good start only takes fewer iterations."""
    y_ = np.asarray(y)
    k_ = K or (y_.shape[1] if y_.ndim == 2 else int(y_.max()) + 1)
    if X.shape[1] * k_ < TORCH_MIN_PARAMS:  # small problems: scipy's L-BFGS-B was 2-3x faster (no tensor overhead)
        return fit_lr_scipy(X, y, C, K, iters, init)
    try:
        import torch  # noqa: F401
    except ImportError:
        return fit_lr_scipy(X, y, C, K, iters, init)
    return fit_lr_torch(X, y, C, K, iters, init)


def fit_lr_scipy(X, y, C: float, K: int | None = None, iters: int = 300, init=None):
    """The same convex objective as fit_lr_torch, minimised with scipy's L-BFGS-B."""
    from scipy.optimize import minimize

    y = np.asarray(y)
    soft = y.ndim == 2
    K = K or (y.shape[1] if soft else int(y.max()) + 1)
    Y = y.astype(np.float64) if soft else None
    X = X.tocsr().astype(np.float64)
    XT = X.T.tocsr()
    n, F = X.shape
    rows = np.arange(n)

    def f(theta):
        W = theta[:F * K].reshape(F, K)
        b = theta[F * K:]
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        s = E.sum(1)
        if soft:
            loss = float((Y.sum(1) * np.log(s)).sum() - (Y * Z).sum() + (W * W).sum() / (2 * C))
            P = E / s[:, None] * Y.sum(1, keepdims=True) - Y
        else:
            loss = float(np.log(s).sum() - Z[rows, y].sum() + (W * W).sum() / (2 * C))
            P = E / s[:, None]
            P[rows, y] -= 1.0
        g = np.concatenate([(XT @ P + W / C).ravel(), P.sum(0)])
        return loss, g

    x0 = np.zeros(F * K + K) if init is None else np.concatenate([np.asarray(init[0], np.float64).T.ravel(),
                                                                   np.asarray(init[1], np.float64)])
    res = minimize(f, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxcor": _history(F * K), "gtol": 1e-7, "ftol": 1e-12})
    W = res.x[:F * K].reshape(F, K)
    return W.T.astype(np.float32).copy(), res.x[F * K:].astype(np.float32).copy()


def _history(n_params: int) -> int:
    """L-BFGS history length: 20 normally; 5 for huge tables, where 2 x 20 parameter-sized vectors would not fit
    in memory (1,000 options x 100k features is 100M parameters)."""
    return 20 if n_params < 10_000_000 else 5


def fit_lr_torch(X, y, C: float, K: int | None = None, iters: int = 300, init=None):
    import torch

    y = np.asarray(y)
    soft = y.ndim == 2
    K = K or (y.shape[1] if soft else int(y.max()) + 1)
    Xc = X.tocoo()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch: "sparse CSR support is in beta"
        Xt = torch.sparse_coo_tensor(np.vstack([Xc.row, Xc.col]), Xc.data.astype(np.float32),
                                     size=X.shape, check_invariants=False).coalesce().to_sparse_csr()
    yt = torch.tensor(y, dtype=torch.float32 if soft else torch.long)  # cross_entropy takes probabilities too
    if init is None:
        W = torch.zeros(X.shape[1], K, requires_grad=True)
        b = torch.zeros(K, requires_grad=True)
    else:
        W = torch.tensor(np.asarray(init[0], np.float32).T.copy(), requires_grad=True)
        b = torch.tensor(np.asarray(init[1], np.float32).copy(), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=_history(X.shape[1] * K), line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Xt @ W + b
        loss = torch.nn.functional.cross_entropy(logits, yt, reduction="sum") + (W * W).sum() / (2 * C)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy().T.copy(), b.detach().numpy().copy()


EMPTY_ITEM = fnv1a(b"e") & MASK  # the item id of an empty input, as in features.items and the C core


def dense_layer(h: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """h @ W2 + b2 in float32, summed in the C core's order (start from b2, add hidden unit j's row in turn, skip
    zeros), so the deep table's confidences match the C core bit for bit, not just to 1e-7: a confidence that sits on
    the certified threshold must give the same certified flag in Python (where it was certified) and in C (where it
    is served)."""
    z = b2.astype(np.float32).copy()
    for j in np.flatnonzero(h):
        z += np.float32(h[j]) * W2[j]
    return z


@dataclass
class Reflex:
    """A table of int8 rows over hashed items, in one of two shapes:

    linear (format 1)   rows are K wide: z = (sum rows / n * scale + bias) / temperature. The exact insertion
                        radius (`radius`) exists because the score is additive in the items.
    deep (format 2)     rows are H wide (a hidden layer): h = relu(sum rows / n * scale + bias), z = (h @ W2 + b2) /
                        temperature. The same lookups and adds as a K = H linear table plus one H x K product, so
                        the cost is the same few microseconds, but the student can model interactions between items
                        (negation in sentiment, word pairs in intents). It has no exact radius (`radius` is NaN), so
                        exposed mode and the guard's allow path need a linear table."""
    keys: np.ndarray  # (F,) uint32 sorted bucket ids
    T: np.ndarray  # (F, K) int8, or (F, H) for a deep table
    scale: np.ndarray  # (K,) or (H,) float32
    bias: np.ndarray  # (K,) or (H,) float32
    temperature: float
    labels: list
    G: np.ndarray | None = None  # (K, K) max_h (T[h,j]-T[h,y]) * scale, for radii (linear tables only)
    W2: np.ndarray | None = None  # (H, K) float32: the dense layer of a deep table
    b2: np.ndarray | None = None  # (K,) float32

    @classmethod
    def compile(cls, vocab, W, b, labels, temperature=1.0, W2=None, b2=None) -> Reflex:
        """W: (K, F) weights of a linear table, or (H, F) first-layer weights with W2 (H, K), b2 (K,) for a deep one."""
        keys = np.array(sorted(vocab, key=vocab.get), dtype=np.uint32)
        Tf = W.T  # (F, K) or (F, H)
        scale = np.abs(Tf).max(0) / 127.0 + 1e-12
        T = np.clip(np.round(Tf / scale), -127, 127).astype(np.int8)
        # the temperature is stored as float32 in the file: round it now so a table scores the same before and after
        # save/load (a confidence exactly at the threshold must not flip on reload)
        r = cls(keys, T, scale.astype(np.float32), np.asarray(b, np.float32), float(np.float32(temperature)), list(labels))
        if W2 is not None:
            r.W2 = np.ascontiguousarray(W2, dtype=np.float32)
            r.b2 = np.asarray(b2, dtype=np.float32)
            if r.W2.shape != (T.shape[1], len(r.b2)):
                raise ValueError(f"W2 must be (H, K) = ({T.shape[1]}, {len(r.b2)}), got {r.W2.shape}")
        else:
            r.G = r._pairwise()
        return r

    @property
    def hidden(self) -> int:
        """Width of the hidden layer; 0 for a linear table."""
        return 0 if self.W2 is None else int(self.T.shape[1])

    @property
    def K(self) -> int:
        return int(self.T.shape[1]) if self.W2 is None else int(self.W2.shape[1])

    @property
    def version(self) -> int:
        return 2 if self.W2 is not None else 1

    def _pairwise(self):
        Tf = self.T.astype(np.float32) * self.scale
        K = Tf.shape[1]
        G = np.zeros((K, K), np.float32)
        for y in range(K):
            G[y] = (Tf - Tf[:, y:y + 1]).max(0)
        return np.maximum(G, 0.0)  # inserting an unseen item (zero row) is always available

    @property
    def nbytes(self):
        n = self.keys.nbytes + self.T.nbytes + self.scale.nbytes + self.bias.nbytes
        return n + (self.W2.nbytes + self.b2.nbytes if self.W2 is not None else self.G.nbytes)

    def rows(self, its: np.ndarray):
        pos = np.searchsorted(self.keys, its)
        pos = np.minimum(pos, len(self.keys) - 1)
        hit = self.keys[pos] == its
        return pos[hit], len(its)

    def logits(self, its: np.ndarray) -> np.ndarray:
        if len(its) == 0:  # as the C core does: an empty input is the single 'e' item
            its = np.array([EMPTY_ITEM], dtype=np.uint32)
        pos, n = self.rows(its)
        acc = self.T[pos].sum(0, dtype=np.int32).astype(np.float32) * self.scale / np.float32(n) + self.bias
        if self.W2 is not None:
            acc = dense_layer(np.maximum(acc, np.float32(0)), self.W2, self.b2)
        return acc / np.float32(self.temperature)

    def radius(self, its: np.ndarray) -> float:
        """Exact minimum number of inserted items that flips the decision (NaN for a deep table: not additive)."""
        if self.W2 is not None:
            return float("nan")
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
        """Format 1 (linear): "S0RX", u32 version=1, u32 F, u32 K, f32 temperature, u32 keys[F], int8 T[F*K],
        f32 scale[K], f32 bias[K], f32 G[K*K].
        Format 2 (deep): "S0RX", u32 version=2, u32 F, u32 K, f32 temperature, u32 H, u32 keys[F], int8 T[F*H],
        f32 scale[H], f32 bias[H], f32 W2[H*K], f32 b2[K]."""
        K, F, H = self.K, len(self.keys), self.hidden
        with open(path, "wb") as f:
            f.write(b"S0RX")
            f.write(struct.pack("<IIIf", self.version, F, K, self.temperature))
            if H:
                f.write(struct.pack("<I", H))
            f.write(self.keys.astype("<u4").tobytes())
            f.write(np.ascontiguousarray(self.T).tobytes())
            f.write(self.scale.astype("<f4").tobytes())
            f.write(self.bias.astype("<f4").tobytes())
            if H:
                f.write(np.ascontiguousarray(self.W2, dtype="<f4").tobytes())
                f.write(self.b2.astype("<f4").tobytes())
            else:
                f.write(np.ascontiguousarray(self.G, dtype="<f4").tobytes())
