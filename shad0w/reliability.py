"""Reliability layer: calibration, a correctness head, conformal sets, and
selective-risk control with a finite-sample guarantee.

Guarantee (under exchangeability of calibration and test data):
  * conformal_sets: P(y in C(x)) >= 1 - alpha
  * SelectiveRiskController: with prob >= 1 - delta over the calibration draw,
    error among *accepted* predictions <= alpha (Learn-then-Test, binomial tail,
    fixed-sequence testing from the most to least confident threshold).
"""

from __future__ import annotations

import math

import numpy as np

from .mathutil import softmax

# ---------- calibration ----------

def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(20), 200)):
        p = softmax(logits / t)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return float(best_t)


# ---------- correctness head: can reorder confidences, unlike a temperature ----------

def confidence_features(p: np.ndarray, X: np.ndarray | None = None,
                        centroids: np.ndarray | None = None,
                        n_tokens: np.ndarray | None = None) -> np.ndarray:
    srt = np.sort(p, axis=1)[:, ::-1]
    top1, top2 = srt[:, 0], srt[:, 1]
    ent = -(p * np.log(p + 1e-12)).sum(1) / math.log(p.shape[1])
    feats = [top1, top1 - top2, ent, np.log(top1 + 1e-6) - np.log(top2 + 1e-6)]
    if X is not None and centroids is not None:
        pred = p.argmax(1)
        sims = X @ centroids.T  # cosine (X and centroids are normalised)
        feats.append(sims[np.arange(len(pred)), pred])  # agreement with predicted class
        feats.append(sims.max(1))  # in-distribution-ness
    if n_tokens is not None:
        feats.append(np.log1p(n_tokens))
    return np.stack(feats, axis=1).astype(np.float32)


class CorrectnessHead:
    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, max_iter=2000)

    def fit(self, F: np.ndarray, correct: np.ndarray) -> CorrectnessHead:
        c = correct.astype(int)
        self.const = None
        if c.min() == c.max():  # all correct (or all wrong): nothing to rank, fall back to a constant
            self.const = float(c[0]) * 0.999 + 0.0005
            return self
        self.lr.fit(self.scaler.fit_transform(F), c)
        return self

    def __call__(self, F: np.ndarray) -> np.ndarray:
        if self.const is not None:
            return np.full(len(F), self.const)
        return self.lr.predict_proba(self.scaler.transform(F))[:, 1]


# ---------- conformal prediction sets ----------

def conformal_threshold(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float,
                        mondrian: bool = False) -> np.ndarray | float:
    """LAC score r = 1 - p_y. Returns q_hat (scalar) or per-class q_hat (Mondrian)."""
    r = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]

    def q(scores):
        n = len(scores)
        if n == 0:
            return 1.0
        k = math.ceil((n + 1) * (1 - alpha))
        return 1.0 if k > n else float(np.sort(scores)[k - 1])

    if not mondrian:
        return q(r)
    K = p_cal.shape[1]
    return np.array([q(r[y_cal == c]) for c in range(K)])


def conformal_sets(p: np.ndarray, qhat) -> np.ndarray:
    return (1.0 - p) <= (qhat if np.isscalar(qhat) else qhat[None, :])


# ---------- selective risk control (Learn-then-Test) ----------

def binom_ucb(errors: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper confidence bound on an error rate."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    from scipy.stats import beta as _beta
    return float(_beta.ppf(1 - delta, errors + 1, n - errors))


def _tolerant_start(alpha: float, delta: float, n: int) -> int:
    """Smallest k at which k answers with floor(alpha * k / 2) disagreements still certify alpha."""
    for k in range(1, n + 1):
        if binom_ucb(int(alpha * k / 2), k, delta) <= alpha:
            return k
    return n


class SelectiveRiskController:
    """Pick the lowest confidence threshold whose certified selective error <= alpha.

    Fixed-sequence testing over thresholds sorted from strict to lenient controls
    the family-wise error at delta without a multiplicity penalty."""

    def __init__(self, alpha: float = 0.05, delta: float = 0.1):
        self.alpha, self.delta = alpha, delta
        self.threshold = np.inf

    def fit(self, conf: np.ndarray, correct: np.ndarray, grid: int = 50) -> SelectiveRiskController:
        order = np.argsort(-conf)
        c_sorted, ok = conf[order], correct[order].astype(bool)
        errs = np.cumsum(~ok)
        n = len(c_sorted)
        # Where the strict -> lenient sequence starts. It depends only on (n, alpha, delta), never on the data,
        # so fixed-sequence testing keeps its guarantee. It starts late enough that a run with disagreement at
        # half of alpha can still certify: a single unlucky disagreement among the very top answers must not end
        # the test (with a start at 50 answers, one such disagreement certified nothing at 99% agreement).
        n_min = math.ceil(math.log(self.delta) / math.log(1 - self.alpha))  # zero-error run that certifies alpha
        n_min = max(n_min, math.ceil(0.05 * n), min(50, n), min(_tolerant_start(self.alpha, self.delta, n), n))
        self.threshold = np.inf
        # pre-registered grid of coverage levels, tested strict -> lenient; stop at first failure
        for k in np.unique(np.linspace(min(n_min, n), n, grid).astype(int)):
            if binom_ucb(int(errs[k - 1]), int(k), self.delta) > self.alpha:
                break
            self.threshold = c_sorted[k - 1]
        return self

    def accept(self, conf: np.ndarray) -> np.ndarray:
        return conf >= self.threshold


# ---------- metrics ----------

def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf)
    err = (~correct[order].astype(bool)).astype(float)
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risks.mean())


def oracle_automation(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Largest coverage whose empirical selective error <= alpha (test-set oracle;
    comparable to how decision-model vendors report 'automation at 5% error')."""
    order = np.argsort(-conf)
    err = np.cumsum(~correct[order].astype(bool)) / np.arange(1, len(conf) + 1)
    ok = np.where(err <= alpha)[0]
    return float((ok.max() + 1) / len(conf)) if len(ok) else 0.0
