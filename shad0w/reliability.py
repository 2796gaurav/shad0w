"""Reliability layer: calibration, a correctness head, conformal sets, and
selective-risk control with a finite-sample guarantee.

Guarantee (under exchangeability of calibration and test data):
  * conformal_sets: P(y in C(x)) >= 1 - alpha
  * SelectiveRiskController: with prob >= 1 - delta over the calibration draw,
    error among *accepted* predictions <= alpha (Learn-then-Test, Clopper-Pearson,
    fixed-sequence and Bonferroni families combined by a union bound).
"""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=65536)
def binom_ucb_betting(errors: int, n: int, delta: float, grid: int = 4000) -> float:
    """Upper confidence bound on an error rate from a betting (test-martingale) argument, at a fixed n.

    For each candidate rate m a gambler bets, round after round, that the loss x_i in {0, 1} falls below m: with a
    bet lambda in [0, 1 / (1 - m)) the capital K_n(m) = prod_i (1 + lambda (m - x_i)) is a nonnegative martingale
    with K_0 = 1 when the true rate is m, so P(K_n(m) >= 1 / delta) <= delta (Markov / Ville). The bound is the
    largest m the gambler could not reject. This is the hedged-capital bound of Waudby-Smith & Ramdas (JRSS-B 2024)
    with the bet mixed uniformly over its whole legal range (a mixture of martingales is a martingale) instead of
    their predictable plug-in, so the bound depends only on (errors, n), never on the order of the losses: in a
    selective-risk test the served rows are sorted by confidence, which makes the plug-in's step-by-step means
    non-identical. For 0/1 losses the mixed capital has a closed form via the incomplete beta function
    (substituting q = m - lambda m (1 - m): K_n(m) = int_0^m q^k (1 - q)^(n - k) dq / (m^(k + 1) (1 - m)^(n - k))),
    so the bound is exact, not simulated.

    Valid at every n, like Clopper-Pearson, and usually looser on 0/1 losses (Clopper-Pearson inverts the exact
    binomial tail, which no test martingale can beat; measured in tests/test_betting.py). Offered for comparison with
    betting-based certificates and because the same machinery extends to losses in [0, 1] and to sequential
    monitoring (shad0w.audit.BettingUCB)."""
    if n == 0 or errors >= n:
        return 1.0
    if errors < 0:
        raise ValueError("errors must be >= 0")
    from scipy.special import betainc, betaln
    k = int(errors)
    m = np.linspace(0.0, 1.0, grid + 1)[1:-1]  # candidate rates in (0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logK = betaln(k + 1, n - k + 1) + np.log(betainc(k + 1, n - k + 1, m)) - (k + 1) * np.log(m) - (n - k) * np.log1p(-m)
    accepted = np.flatnonzero(~(logK >= math.log(1.0 / delta)))  # rates the gambler could not reject (NaN: keep)
    if len(accepted) == 0:
        return float(m[0])
    j = int(accepted.max()) + 1  # the bound is the supremum of the accepted set, rounded up one grid step
    return float(m[j]) if j < len(m) else 1.0


def _tolerant_start(alpha: float, delta: float, n: int, bound=binom_ucb) -> int:
    """Smallest k at which k answers with floor(alpha * k / 2) disagreements still certify alpha."""
    for k in range(1, n + 1):
        if bound(int(alpha * k / 2), k, delta) <= alpha:
            return k
    return n


BONFERRONI_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # coverage levels, fixed in advance
PROCEDURES = ("auto", "fixed-sequence", "bonferroni", "betting")
BOUNDS = {"clopper-pearson": binom_ucb, "betting": binom_ucb_betting}


def _served(c_sorted, k):
    """How many calibration rows a threshold at the k-th confidence actually serves: rows tied with it are served
    too, so they must be counted in the test (otherwise tied confidences would serve answers the bound never saw)."""
    return int(np.searchsorted(-c_sorted, -c_sorted[k - 1], side="right"))


class SelectiveRiskController:
    """Pick the lowest confidence threshold whose certified selective error <= alpha, with probability >= 1 - delta.

    Two valid procedures over pre-registered coverage levels:
      fixed-sequence  tests levels strict -> lenient and stops at the first failure. No multiplicity penalty, so it
                      is the tightest when disagreement rises as confidence falls; but chance disagreements among the
                      top answers can end it early, which happens when a teacher makes errors regardless of input.
      bonferroni      tests 12 levels independently at delta / 12 and keeps the most lenient that passes. Robust to
                      flat, input-independent teacher noise, at the price of a multiplicity penalty.
    auto (default) runs both at delta / 2 each and keeps the more lenient threshold. By the union bound the result
    is valid at delta. In simulation it kept ~98% of fixed-sequence's offload where that is strong and raised
    offload from 39% to ~100% under LLM-like flat noise (tests/test_certificate_validity.py).

    A fourth, optional procedure:
      betting         fixed-sequence testing with the betting bound (binom_ucb_betting) in place of Clopper-Pearson.
                      Same guarantee; kept as an option for comparison with betting-based certificates (it is not
                      tighter than Clopper-Pearson on 0/1 disagreement, see tests/test_betting.py)."""

    def __init__(self, alpha: float = 0.05, delta: float = 0.1, procedure: str = "auto"):
        if procedure not in PROCEDURES:
            raise ValueError(f"procedure must be one of {PROCEDURES}")
        self.alpha, self.delta, self.procedure = alpha, delta, procedure
        self.threshold = np.inf
        self.chosen = None  # which procedure produced the threshold

    def fit(self, conf: np.ndarray, correct: np.ndarray, grid: int = 50) -> SelectiveRiskController:
        order = np.argsort(-np.asarray(conf))
        c_sorted, ok = np.asarray(conf)[order], np.asarray(correct)[order].astype(bool)
        errs = np.cumsum(~ok)
        if self.procedure == "fixed-sequence":
            self.threshold, self.chosen = self._fixed_sequence(c_sorted, errs, self.delta, grid), "fixed-sequence"
        elif self.procedure == "betting":
            self.threshold = self._fixed_sequence(c_sorted, errs, self.delta, grid, bound=binom_ucb_betting)
            self.chosen = "betting"
        elif self.procedure == "bonferroni":
            self.threshold, self.chosen = self._bonferroni(c_sorted, errs, self.delta), "bonferroni"
        else:
            t1 = self._fixed_sequence(c_sorted, errs, self.delta / 2, grid)
            t2 = self._bonferroni(c_sorted, errs, self.delta / 2)
            self.threshold = min(t1, t2)
            self.chosen = None if not np.isfinite(self.threshold) else ("fixed-sequence" if t1 <= t2 else "bonferroni")
        return self

    def _fixed_sequence(self, c_sorted, errs, delta, grid, bound=binom_ucb):
        n = len(c_sorted)
        if n == 0:
            return np.inf
        # Where the strict -> lenient sequence starts depends only on (n, alpha, delta), never on the data, so
        # fixed-sequence testing keeps its guarantee. It starts late enough that a run with disagreement at half of
        # alpha can still certify, so one unlucky disagreement among the very top answers does not end the test.
        n_min = math.ceil(math.log(delta) / math.log(1 - self.alpha))  # zero-error run that certifies alpha
        n_min = max(n_min, math.ceil(0.05 * n), min(50, n), min(_tolerant_start(self.alpha, delta, n, bound), n))
        thr = np.inf
        for k in np.unique(np.linspace(min(n_min, n), n, grid).astype(int)):  # strict -> lenient, stop at failure
            k = _served(c_sorted, k)
            if bound(int(errs[k - 1]), int(k), delta) > self.alpha:
                break
            thr = c_sorted[k - 1]
        return thr

    def _bonferroni(self, c_sorted, errs, delta):
        n = len(c_sorted)
        ks = sorted({max(1, int(round(level * n))) for level in BONFERRONI_LEVELS}) if n else []
        thr = np.inf
        for k in ks:
            k = _served(c_sorted, k)
            if binom_ucb(int(errs[k - 1]), k, delta / len(ks)) <= self.alpha:
                thr = min(thr, c_sorted[k - 1])  # the most lenient level that passes
        return thr

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


# ---------- one certificate for a cascade of tables ----------

class CascadeController:
    """Certify a cascade: tier 0 answers when its confidence clears its threshold, else tier 1, ..., else the teacher.

    One guarantee for the whole cascade, with no per-tier split of alpha: with probability >= 1 - delta, among the
    answers ANY tier serves, the rate of disagreement with the teacher is at most alpha (the same statement as
    SelectiveRiskController, which is the one-tier case).

    How: the calibration rows are split at random. The first part ("path") turns one risk level s into a threshold per
    tier: the most lenient threshold at which that tier's own disagreement on the path rows it would answer stays
    <= s. The second part tests the cascade at s = 0, ..., s_max, strict to lenient, with Clopper-Pearson at delta
    and stops at the first failure (fixed-sequence testing). The path depends only on the first part, so the test
    on the second part is valid as is (Learn-then-Test, Angelopoulos et al. 2021; cf. risk-controlled early exit,
    Jazbec et al. 2024).

    conf[k] (n,) is tier k's confidence on each calibration row (NaN where tier k is not available) and agree[k] (n,)
    whether its answer equals the teacher's."""

    def __init__(self, alpha: float = 0.05, delta: float = 0.1, path_fraction: float = 0.3, levels: int = 40,
                 seed: int = 0):
        self.alpha, self.delta, self.path_fraction, self.levels, self.seed = alpha, delta, path_fraction, levels, seed
        self.thresholds: np.ndarray | None = None
        self.share = None  # per tier: share of the test rows it answered at the certified level
        self.risk = None  # realised disagreement on served test rows at the certified level

    @staticmethod
    def _route(conf, thr):
        """Index of the tier that answers each row (-1 = the teacher)."""
        T, n = conf.shape
        out = np.full(n, -1)
        for k in range(T - 1, -1, -1):  # the first tier that clears its threshold wins
            ok = np.nan_to_num(conf[k], nan=-np.inf) >= thr[k]
            out[ok] = k
        return out

    def _thresholds_at(self, conf, agree, s):
        thr = np.full(conf.shape[0], np.inf)
        for k in range(conf.shape[0]):
            c = conf[k]
            m = np.isfinite(c)
            if not m.any():
                continue
            order = np.argsort(-c[m])
            cs, bad = c[m][order], np.cumsum(~agree[k][m][order])
            n_min = max(20, int(0.02 * m.sum()))  # never set a threshold on a handful of rows
            ok = [j for j in range(n_min, len(cs) + 1) if bad[j - 1] <= s * j]
            if ok:
                thr[k] = cs[max(ok) - 1]
        return thr

    def fit(self, conf, agree) -> CascadeController:
        conf = np.asarray(conf, dtype=float)
        agree = np.asarray(agree, dtype=bool)
        if conf.ndim == 1:
            conf, agree = conf[None], agree[None]
        T, n = conf.shape
        rng = np.random.default_rng(self.seed)
        path = rng.random(n) < self.path_fraction
        test = ~path
        self.thresholds = np.full(T, np.inf)
        n_test = int(test.sum())
        # Where the test sequence starts is chosen on the path rows only: the first level whose cascade would serve
        # enough rows to be able to pass (as SelectiveRiskController starts at _tolerant_start). That keeps
        # fixed-sequence testing valid: every level from the start on is tested, and the first failure ends it.
        start = min(_tolerant_start(self.alpha, self.delta, max(n_test, 1)), max(n_test, 1))
        seq = []
        for s in np.linspace(0.0, 2 * self.alpha, self.levels):  # strict -> lenient
            thr = self._thresholds_at(conf[:, path], agree[:, path], s)
            path_served = int((self._route(conf[:, path], thr) >= 0).sum())
            if seq or path_served * n_test / max(int(path.sum()), 1) >= start:
                seq.append(thr)
        best = None
        for thr in seq:
            r = self._route(conf[:, test], thr)
            served = r >= 0
            k = int(served.sum())
            wrong = int((~agree[:, test][r[served], np.flatnonzero(served)]).sum()) if k else 0
            if k == 0 or binom_ucb(wrong, k, self.delta) > self.alpha:
                break
            best = (thr, r, wrong, k)
        if best is not None:
            thr, r, wrong, k = best
            self.thresholds = thr
            self.share = np.array([(r == j).mean() for j in range(T)])
            self.risk = wrong / k
        else:
            self.share = np.zeros(T)
        return self

    def route(self, conf) -> np.ndarray:
        conf = np.asarray(conf, dtype=float)
        return self._route(conf if conf.ndim == 2 else conf[:, None], self.thresholds)
