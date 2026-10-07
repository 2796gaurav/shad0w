"""Anchored prototype refinement with Sinkhorn-balanced assignment (label-names-only pseudo-labelling).

E: (N, d) L2-normalised text embeddings; L: (K, d) L2-normalised option embeddings.
  C_k <- L_k
  repeat: soft-assign texts to prototypes with Sinkhorn balancing (prevents collapse onto a few options),
          C_k <- normalise(beta * L_k + (1 - beta) * mean of the most confident members of k)
Related work: X-Class, LOTClass, PESCO (label-name-only classification); SeLa / OTTER (balanced assignment).
"""
from __future__ import annotations

import numpy as np


def sinkhorn(S, temp=0.05, iters=30, prior=None):
    """Balanced soft assignment. prior: target class marginals (K,); default uniform."""
    Q = np.exp((S - S.max(1, keepdims=True)) / temp)
    col = None if prior is None else np.asarray(prior, dtype=np.float64) * S.shape[0]
    for _ in range(iters):
        Q /= Q.sum(0, keepdims=True)
        if col is not None:
            Q *= col[None, :]
        Q /= Q.sum(1, keepdims=True)
    return Q


def estimate_prior(S, K, mix=0.5):
    """Class prior from the zero-shot assignment, smoothed half-way toward uniform."""
    f = np.bincount(S.argmax(1), minlength=K).astype(np.float64) + 0.5
    f /= f.sum()
    return (1 - mix) * f + mix * np.full(K, 1.0 / K)


def refine(E, L, beta=0.3, iters=10, top=0.5, balance=True, prior=None):
    """Returns (assignment, confidence, prototypes). prior: None (uniform), "auto" (estimated) or a (K,) array."""
    C = L.copy()
    if isinstance(prior, str) and prior == "auto":
        prior = estimate_prior(E @ L.T, len(L))
    for _ in range(iters):
        S = E @ C.T
        Q = sinkhorn(S, prior=prior) if balance else np.exp(S / 0.05)
        a = Q.argmax(1)
        conf = Q.max(1) / Q.sum(1)
        newC = []
        for k in range(len(L)):
            ix = np.where(a == k)[0]
            if len(ix) == 0:
                newC.append(L[k])
                continue
            ix = ix[np.argsort(-conf[ix])[: max(1, int(top * len(ix)))]]
            c = beta * L[k] + (1 - beta) * E[ix].mean(0)
            newC.append(c / np.linalg.norm(c))
        C = np.stack(newC)
    S = E @ C.T
    Q = sinkhorn(S, prior=prior) if balance else np.exp(S / 0.05)
    return Q.argmax(1), Q.max(1) / Q.sum(1), C
