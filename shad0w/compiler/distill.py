"""Distiller + certifier: (texts, labels) -> compiled Reflex artifact + correctness head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from shad0w.features import items_batch
from shad0w.mathutil import softmax
from shad0w.reflex import Reflex, bag_matrix, build_vocab, dense_layer, fit_lr
from shad0w.reliability import CorrectnessHead, confidence_features, fit_temperature


@dataclass
class Compiled:
    rx: Reflex
    head: CorrectnessHead
    C: float
    student: str = "linear"  # "linear" or "deep" (a hidden layer; see Reflex)
    choice: dict | None = None  # how the student was chosen: held-out coverage of each candidate

    def predict(self, texts):
        its = items_batch(texts)
        p = np.stack([softmax(self.rx.logits(i)) for i in its])
        c = self.head(confidence_features(p, n_tokens=np.array([len(i) for i in its])))
        return p, c, its


# Above this many table cells (features x options), training picks C on a subsample (the full grid, so quality holds)
# and uses 3 folds instead of 5 for the temperature: the fits dominate training time on many-option questions.
LARGE_PROBLEM = 2_000_000
C_SEARCH_ROWS = 4000
STUDENTS = ("auto", "linear", "deep")
HIDDEN = None  # default width of the deep student's hidden layer: hidden_for(K)


def hidden_for(n_options: int) -> int:
    """Hidden width from the option count: 32 for a few options, up to 128 for many. Measured (labels as teacher):
    32 = 64 = 128 on SST-2, AG News and Fashion-MNIST; BANKING77 (77 options) certified 54 / 70 / 77% at 32 / 64 / 128."""
    return int(min(128, max(32, 2 * n_options)))


def max_features_for(max_mb: float | None, n_options: int, hidden: int = 0) -> int | None:
    """How many feature rows fit a table of `max_mb` megabytes. A linear .s0 file is a 20-byte header, then per row a
    4-byte key and one int8 per option, then 8 bytes per option (scale, bias) and a K x K float32 matrix (for radii).
    A deep table (hidden > 0) has rows `hidden` wide, 8 bytes per hidden unit and a float32 hidden x K layer + K."""
    if not max_mb:
        return None
    k = n_options
    if hidden:
        fixed = 24 + 8 * hidden + 4 * hidden * k + 4 * k
        return max(1000, int((max_mb * 2**20 - fixed) / (hidden + 4)))
    fixed = 20 + 8 * k + 4 * k * k
    return max(1000, int((max_mb * 2**20 - fixed) / (k + 4)))


def _bags(item_lists, vocab):
    """Flat (indices, offsets, weights) of the bags, weights = 1 / n_items of the bag (all items, hits or not), so a
    model trained on them scores exactly as the table serves: sum of the rows found / number of items."""
    idx, w, off = [], [], [0]
    for its in item_lists:
        n = max(1, len(its))
        cols = [vocab[int(h)] for h in its if int(h) in vocab]
        idx.extend(cols)
        w.extend([1.0 / n] * len(cols))
        off.append(len(idx))
    return (np.asarray(idx, dtype=np.int64), np.asarray(off, dtype=np.int64), np.asarray(w, dtype=np.float32))


BAG_DROPOUT = 0.3  # with static vectors: share of training rows whose item rows are hidden (see fit_dan)
DAN_STEPS = 4000  # mini-batch steps the deep student trains for at most (epochs follow from the row count)


def _budget(n_rows: int, epochs: int | None, patience: int | None, batch: int = 256):
    """Epochs and patience from a step budget: a small log (5 batches per epoch) needs hundreds of epochs, a large
    one a few dozen; early stopping ends it sooner when the held-out accuracy stops improving."""
    per_epoch = max(1, -(-n_rows // batch))
    if epochs is None:
        epochs = int(min(400, max(20, -(-DAN_STEPS // per_epoch))))
    if patience is None:
        patience = max(4, epochs // 5)
    return epochs, patience


def fit_dan(item_lists, vocab, tgt, K: int, H: int, fit_idx, hold_idx, epochs: int | None = None, lr: float = 3e-3,
            weight_decay: float = 0.0, dropout: float = 0.1, seed: int = 0, patience: int | None = None, dense=None):
    """Train the deep student: rows (F x H) summed over the bag / n_items, + b1, relu, dropout (training only),
    dense H x K + b2, cross-entropy on `tgt` (class indices or (n, K) probabilities), Adam in mini-batches. The epoch
    with the lowest cross-entropy on `hold_idx` wins (early stopping), so hold rows never shape the weights.
    dense: optional (n, D) float32 per-row features that are additive in the items (the mean of static word vectors,
    already divided by the item count): they get a projection P (H x D) that `_fold` later writes into the word rows.
    Returns (W1 (H, F), b1 (H,), W2 (H, K), b2 (K,), best_epoch, P or None)."""
    epochs, patience = _budget(len(fit_idx), epochs, patience)
    try:
        import torch
    except ImportError:
        return _fit_dan_numpy(item_lists, vocab, tgt, K, H, fit_idx, hold_idx, epochs, lr, weight_decay, dropout, seed,
                              patience, dense)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx, off, w = _bags(item_lists, vocab)
    idx_t, w_t = torch.from_numpy(idx), torch.from_numpy(w)
    tgt = np.asarray(tgt)
    soft = tgt.ndim == 2
    y_hard = tgt.argmax(1) if soft else tgt
    emb = torch.nn.EmbeddingBag(len(vocab), H, mode="sum")
    torch.nn.init.normal_(emb.weight, std=0.05)
    out = torch.nn.Linear(H, K)
    proj = torch.nn.Linear(dense.shape[1], H, bias=False) if dense is not None else None
    dense_t = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32)) if dense is not None else None
    params = list(emb.parameters()) + list(out.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    tt = torch.tensor(tgt, dtype=torch.float32 if soft else torch.long)

    def batch(rows):
        parts = [idx_t[off[r]:off[r + 1]] for r in rows]
        wp = [w_t[off[r]:off[r + 1]] for r in rows]
        o = torch.tensor(np.cumsum([0] + [len(p) for p in parts[:-1]]), dtype=torch.long)
        return torch.cat(parts), o, torch.cat(wp)

    def forward(rows, train):
        i, o, ww = batch(rows)
        pre = emb(i, o, per_sample_weights=ww)
        if proj is not None:
            if train:  # bag dropout: some rows see only the vectors, so the projection must carry meaning on its own
                keep = (torch.rand(len(rows), 1) >= BAG_DROPOUT).float() / (1 - BAG_DROPOUT)
                pre = pre * keep
            pre = pre + proj(dense_t[rows])
        h = torch.relu(pre)
        if train and dropout > 0:
            h = torch.nn.functional.dropout(h, dropout, training=True)
        return out(h)

    def hold_score():
        """Held-out cross-entropy (lower is better): keeps improving after accuracy saturates, and it is what the
        temperature and the certificate's confidence ranking depend on."""
        emb.eval(); out.eval()
        with torch.no_grad():
            z = torch.cat([forward(hold_idx[i:i + 4096], False) for i in range(0, len(hold_idx), 4096)])
            return float(torch.nn.functional.cross_entropy(z, torch.as_tensor(y_hard[hold_idx], dtype=torch.long)))

    best, best_state, since = np.inf, None, 0
    fit_idx = np.asarray(fit_idx)
    for ep in range(epochs):
        emb.train(); out.train()
        perm = fit_idx[rng.permutation(len(fit_idx))]
        for i in range(0, len(perm), 256):
            rows = perm[i:i + 256]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(forward(rows, True), tt[rows])
            loss.backward()
            opt.step()
        loss_h = hold_score()
        if loss_h < best - 1e-5:
            best, since = loss_h, 0
            best_state = (emb.weight.detach().numpy().T.copy(), np.zeros(H, np.float32),
                          out.weight.detach().numpy().T.copy(), out.bias.detach().numpy().copy(), ep,
                          None if proj is None else proj.weight.detach().numpy().copy())
        else:
            since += 1
            if since >= patience:
                break
    W1, b1, W2, b2, ep, P = best_state
    return W1.astype(np.float32), b1, W2.astype(np.float32), b2.astype(np.float32), ep, P


def _fit_dan_numpy(item_lists, vocab, tgt, K, H, fit_idx, hold_idx, epochs, lr, weight_decay, dropout, seed, patience,
                   dense=None):
    """The same model and training loop without torch (slower; for installs with numpy only)."""
    rng = np.random.default_rng(seed)
    idx, off, w = _bags(item_lists, vocab)
    tgt = np.asarray(tgt)
    soft = tgt.ndim == 2
    y_hard = tgt.argmax(1) if soft else tgt
    Y = tgt.astype(np.float32) if soft else np.eye(K, dtype=np.float32)[tgt]
    F = len(vocab)
    E = rng.normal(0, 0.05, (F, H)).astype(np.float32)
    W2 = (rng.normal(0, 1, (H, K)) / np.sqrt(H)).astype(np.float32)
    b2 = np.zeros(K, np.float32)
    Pm = None if dense is None else (rng.normal(0, 1, (H, dense.shape[1])) / np.sqrt(dense.shape[1])).astype(np.float32)
    params = [("E", E), ("W2", W2), ("b2", b2)] + ([("P", Pm)] if Pm is not None else [])
    m = {k: np.zeros_like(val) for k, val in params}
    v = {k: np.zeros_like(val) for k, val in params}
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0

    def pooled(rows, keep=None):
        """keep: optional (n, 1) bag-dropout mask on the item rows (training with static vectors, as in fit_dan)."""
        P = np.zeros((len(rows), H), np.float32)
        cols, rid, ww = [], [], []
        for j, r in enumerate(rows):
            s = slice(off[r], off[r + 1])
            cols.append(idx[s]); ww.append(w[s]); rid.append(np.full(off[r + 1] - off[r], j))
        cols, rid, ww = np.concatenate(cols), np.concatenate(rid), np.concatenate(ww)
        np.add.at(P, rid, E[cols] * ww[:, None])
        if keep is not None:
            P *= keep
        if Pm is not None:
            P += dense[rows] @ Pm.T
        return P, cols, rid, ww

    def adam(name, param, grad):
        nonlocal step
        m[name] = beta1 * m[name] + (1 - beta1) * grad
        v[name] = beta2 * v[name] + (1 - beta2) * grad * grad
        param -= lr * (m[name] / (1 - beta1 ** step)) / (np.sqrt(v[name] / (1 - beta2 ** step)) + eps)

    def hold_score():
        z = np.concatenate([np.maximum(pooled(hold_idx[i:i + 4096])[0], 0) @ W2 + b2 for i in range(0, len(hold_idx), 4096)])
        z = z - z.max(1, keepdims=True)
        return float((np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), y_hard[hold_idx]]).mean())

    best, best_state, since = np.inf, None, 0
    fit_idx = np.asarray(fit_idx)
    for ep in range(epochs):
        perm = fit_idx[rng.permutation(len(fit_idx))]
        for i in range(0, len(perm), 256):
            rows = perm[i:i + 256]
            # bag dropout with static vectors: some rows see only the vectors, so the projection carries meaning alone
            keep = ((rng.random((len(rows), 1)) >= BAG_DROPOUT).astype(np.float32) / (1 - BAG_DROPOUT)) if Pm is not None else None
            P, cols, rid, ww = pooled(rows, keep)
            Hd = np.maximum(P, 0)
            mask = (rng.random(Hd.shape) >= dropout).astype(np.float32) / (1 - dropout) if dropout > 0 else 1.0
            Ha = Hd * mask
            Z = Ha @ W2 + b2
            Z -= Z.max(1, keepdims=True)
            Pr = np.exp(Z); Pr /= Pr.sum(1, keepdims=True)
            G = (Pr - Y[rows]) / len(rows)
            step += 1
            gW2 = Ha.T @ G + weight_decay * W2
            gb2 = G.sum(0)
            gH = (G @ W2.T) * mask * (P > 0)
            gE = np.zeros_like(E)
            np.add.at(gE, cols, (gH * keep if keep is not None else gH)[rid] * ww[:, None])
            adam("E", E, gE + weight_decay * E); adam("W2", W2, gW2); adam("b2", b2, gb2)
            if Pm is not None:
                adam("P", Pm, gH.T @ dense[rows])
        loss_h = hold_score()
        if loss_h < best - 1e-5:
            best, since = loss_h, 0
            best_state = (E.T.copy(), np.zeros(H, np.float32), W2.copy(), b2.copy(), ep, None if Pm is None else Pm.copy())
        else:
            since += 1
            if since >= patience:
                break
    return best_state


def _hold_coverage(z_hold, y_hold, alpha: float) -> dict:
    """How a candidate student would do on the held-out fitting rows: agreement and oracle coverage at alpha (the
    largest share whose empirical error stays <= alpha; a ranking proxy for what the certificate will allow)."""
    from shad0w.reliability import oracle_automation
    p = softmax(z_hold)
    correct = p.argmax(1) == y_hold
    return {"agreement": float(correct.mean()), "coverage": float(oracle_automation(p.max(1), correct, alpha))}


def compile_schema(texts, y, labels, C: float | None = None, seed: int = 0,
                   C_grid=(100.0, 300.0, 1000.0, 3000.0), idf: str | bool = "auto",
                   max_features: int | None = None, soft=None, student: str = "auto", hidden: int = HIDDEN,
                   alpha: float = 0.05, dense: dict | None = None, oof_folds: int = 0) -> Compiled:
    """dense: optional {word item id: float32 vector} (static word vectors, see shad0w.static.word_vectors). The mean
    vector of a text's known words is an extra additive input during training; afterwards its weights are folded
    into the word rows EXACTLY (W . mean(E[w]) = mean(W . E[w])), and words that have a vector but never appeared in
    the training texts get rows too. Serving stays a plain table lookup.

    oof_folds: 0 (default) takes the linear table's temperature from the held-out fitting slice (2 fits); k > 1 uses
    k-fold out-of-fold logits instead (k + 1 fits, the behaviour before the deep student; slower, same guarantee).

    student: "linear" (the additive int8 table, exact radius), "deep" (rows feed a hidden layer of `hidden` units;
    models item interactions such as negation, no radius) or "auto": train both and keep the one whose held-out
    fitting rows show the larger oracle coverage at `alpha` (linear on a tie, so the radius survives). The certificate
    is computed afterwards on the calibration rows, whichever student is kept.

    soft: optional (n, K) target probabilities (the teacher's), used for fitting instead of one-hot targets; the
    temperature, the choice of C and the certificate still use the hard answers `y`.

    idf: fold inverse document frequency into the table (T'[h] = idf_h * T[h]); runtime and
    the exact radius are unchanged. 'auto' enables it when texts average > 100 items
    (long documents, where plain term frequency is swamped by common items).

    max_features: keep only this many feature rows (the ones with the largest weight for any option) and refit on
    them, so the table fits a size budget. The certificate is computed afterwards on the final table, so it stays
    valid; a smaller table usually certifies a little less traffic."""
    if student not in STUDENTS:
        raise ValueError(f"student must be one of {STUDENTS}")
    hidden = hidden or hidden_for(len(labels))
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    K = len(labels)
    tgt = y if soft is None else np.asarray(soft, dtype=np.float32)
    if soft is not None and tgt.shape != (len(y), K):
        raise ValueError(f"soft targets must be ({len(y)}, {K}), got {tgt.shape}")
    its = items_batch(texts)
    # long texts (many items each) get idf and a wider C grid; item ids given directly (an image's fixed-size bag)
    # are not documents, so they never do
    is_text = all(isinstance(t, str) for t in texts)
    long_docs = is_text and np.mean([len(i) for i in its]) > 100
    if not is_text and C is None and tuple(C_grid) == (100.0, 300.0, 1000.0, 3000.0):
        # item ids (an image's ferns, an embedding's buckets): dense, fixed-size bags that often want weaker
        # regularisation than text n-grams; on CLIP buckets the best C was 1e3-1e4 (certified 37% -> 50%)
        C_grid = (100.0, 300.0, 1000.0, 3000.0, 1e4, 1e5)
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
    Zs = None
    if dense:
        D = len(next(iter(dense.values())))
        Zs = np.zeros((len(its), D), np.float32)
        for r, s_ in enumerate(its):
            vs = [dense[int(h)] for h in s_ if int(h) in dense]
            if vs:
                Zs[r] = np.sum(vs, axis=0) / max(1, len(s_))
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
    if Zs is not None:  # the dense block rides along as extra columns of the fitting matrix
        from scipy.sparse import csr_matrix, hstack
        X = hstack([X, csr_matrix(Zs)]).tocsr()
    F0 = len(vocab)  # item columns; columns F0.. are the dense block
    large = len(vocab) * K > LARGE_PROBLEM
    # one held-out slice of the fitting rows: it ranks C, stops the deep student early and chooses between students
    hold = rng.random(len(y)) < 0.15
    fit_idx, hold_idx = np.flatnonzero(~hold), np.flatnonzero(hold)
    if len(hold_idx) < 20 or len(np.unique(y[fit_idx])) < len(np.unique(y)):  # (an unused option is fine)
        fit_idx, hold_idx = np.arange(len(y)), np.arange(len(y))  # too few rows to hold any out
    # the held-out slice in two halves: `tune` picks C and stops the deep student early, `judge` compares the two
    # students. Scoring a student on the rows that tuned it would favour whichever was tuned harder.
    half = rng.random(len(hold_idx)) < 0.5
    tune_idx, judge_idx = hold_idx[half], hold_idx[~half]
    if len(tune_idx) < 10 or len(judge_idx) < 10:
        tune_idx = judge_idx = hold_idx
    choice: dict = {}
    lin_hold = None  # (W, b) of a linear fit that never saw the hold rows, for the comparison
    need_linear = student in ("auto", "linear")
    if C is None and need_linear:
        search_idx = fit_idx
        if large and len(search_idx) > C_SEARCH_ROWS:
            search_idx = np.sort(rng.choice(search_idx, C_SEARCH_ROWS, replace=False))
        scores, prev, fits = [], None, {}
        for c in sorted(C_grid):  # each C starts from the previous solution (warm start: same optimum, fewer steps)
            W, b = fit_lr(X[search_idx], tgt[search_idx], c * len(y) / max(1, len(search_idx)) if large else c, K=K,
                          init=prev, iters=100)  # only ranks C by hold-out accuracy: it need not converge fully
            prev = fits[c] = (W, b)
            scores.append(((X[tune_idx] @ W.T + b).argmax(1) == y[tune_idx]).mean())
        C_grid = sorted(C_grid)
        C = C_grid[int(np.argmax(scores))]
        lin_hold = fits[C]
    elif C is None:
        C = 1000.0
    deep = None
    if student in ("auto", "deep"):
        # the deep student trains on plain mean bags (idf is a trick for the linear table: its hidden layer learns
        # its own item weights), so nothing is folded into its rows afterwards
        W1, b1, W2, b2, ep, P = fit_dan(its, vocab, tgt, K, hidden, fit_idx, tune_idx, seed=seed, dense=Zs)
        deep = (W1, b1, W2, b2, P)
        zd = _dan_logits(its, vocab, W1, b1, W2, b2, judge_idx, Zs, P)
        choice["deep"] = {**_hold_coverage(zd, y[judge_idx], alpha), "hidden": int(hidden), "epochs": int(ep + 1)}
    if need_linear:
        if lin_hold is None:
            lin_hold = fit_lr(X[fit_idx], tgt[fit_idx], C, K=K, iters=100)
        zl = X[judge_idx] @ lin_hold[0].T + lin_hold[1]
        choice["linear"] = {**_hold_coverage(zl, y[judge_idx], alpha), "C": float(C)}
    if student == "auto":
        # the deep student must beat the linear table clearly on held-out coverage (its tie-breaker is the exact
        # radius, which only a linear table has); when neither certifies anything, the better agreement decides
        dl, ln = choice["deep"], choice["linear"]
        better = (dl["coverage"] > ln["coverage"] + 0.01) or (dl["coverage"] == ln["coverage"] == 0.0 and
                                                              dl["agreement"] > ln["agreement"] + 0.01)
        student = "deep" if better else "linear"
    choice["chosen"] = student
    n_tok = np.array([len(i) for i in its])
    if student == "deep":
        W1, b1, W2, b2, P = deep
        if max_features:  # the budget was sized for K-wide rows; a deep row is `hidden` wide
            max_features = max(1000, int(max_features * (K + 4) / (hidden + 4)))
        if max_features and len(vocab) > max_features:
            # keep the most frequent items and refit on them. On SST-2 (deep, H = 32) frequency kept more than weight
            # did at a similar size: 58.5% certified at 20k rows vs 50.7% at 16.7k rows by largest |weight|
            dfc = np.zeros(len(vocab), np.int64)
            for s_ in its:
                cols = {vocab[int(h)] for h in s_ if int(h) in vocab}
                dfc[list(cols)] += 1
            keep = np.sort(np.argsort(-dfc, kind="stable")[:max_features])
            vocab = _prune_vocab(vocab, keep)
            W1, b1, W2, b2, _, P = fit_dan(its, vocab, tgt, K, hidden, fit_idx, tune_idx, seed=seed, dense=Zs)
        z_hold = _dan_logits(its, vocab, W1, b1, W2, b2, hold_idx, Zs, P)
        T = fit_temperature(z_hold, y[hold_idx])  # hold rows never trained the weights: honest logits
        p_hold = softmax(z_hold / T)
        head = CorrectnessHead().fit(confidence_features(p_hold, n_tokens=n_tok[hold_idx]), p_hold.argmax(1) == y[hold_idx])
        if P is not None:
            vocab, W1 = _fold(vocab, W1, P, dense, budget=max_features)
        rx = Reflex.compile(vocab, W1, b1, labels, temperature=T, W2=W2, b2=b2)
        return Compiled(rx, head, C, "deep", choice)
    if oof_folds and len(hold_idx) < len(y):  # the old way: k-fold out-of-fold logits for the temperature
        oof = np.zeros((len(y), K), np.float32)
        n_splits = max(2, min(3 if large else oof_folds, np.bincount(y, minlength=K).min()))
        W, b = fit_lr(X, tgt, C, K=K)  # the table itself; each fold below starts from it (a fold is most of the data)
        full = (W, b)
        for a, b_ in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
            Wf, bf = fit_lr(X[a], tgt[a], C, K=K, init=full)  # always K outputs, even if a class is absent here
            oof[b_] = X[b_] @ Wf.T + bf
        T = fit_temperature(oof, y)
        cal_rows = np.arange(len(y))
    else:
        # Default: the temperature and the correctness head come from the held-out fitting slice, scored by a fit on
        # the other rows; the table is then refit on every row from that solution. Two fits instead of six. The
        # temperature only rescales confidences, and the certificate is computed afterwards on calibration rows the
        # table never saw, so its guarantee does not depend on how the temperature was chosen.
        Wh, bh = fit_lr(X[fit_idx], tgt[fit_idx], C, K=K, init=lin_hold)
        oof = np.zeros((len(y), K), np.float32)
        oof[hold_idx] = X[hold_idx] @ Wh.T + bh
        T = fit_temperature(oof[hold_idx], y[hold_idx])
        W, b = fit_lr(X, tgt, C, K=K, init=(Wh, bh))
        cal_rows = hold_idx
    if max_features and len(vocab) > max_features:
        keep = np.sort(np.argsort(-np.abs(W[:, :F0]).max(0))[:max_features])  # columns stay in ascending-hash order
        vocab = _prune_vocab(vocab, keep)
        X = X[:, np.concatenate([keep, np.arange(F0, X.shape[1])])]  # the dense block (if any) always stays
        if w_idf is not None:
            w_idf = w_idf[keep]
        F0 = len(vocab)
        if oof_folds:
            W, b = fit_lr(X, tgt, C, K=K)
        else:  # the pruned table gets its own temperature, from the same held-out slice
            Wh, bh = fit_lr(X[fit_idx], tgt[fit_idx], C, K=K)
            oof[hold_idx] = X[hold_idx] @ Wh.T + bh
            T = fit_temperature(oof[hold_idx], y[hold_idx])
            W, b = fit_lr(X, tgt, C, K=K, init=(Wh, bh))
    p_oof = softmax(oof[cal_rows] / T)
    head = CorrectnessHead().fit(confidence_features(p_oof, n_tokens=n_tok[cal_rows]), p_oof.argmax(1) == y[cal_rows])
    Wp, W = W[:, F0:], W[:, :F0]
    if w_idf is not None:
        W = W * w_idf[None, :]  # fold idf into the table: runtime stays sum(T[h]) / n
    if Zs is not None:
        vocab, W = _fold(vocab, W, Wp, dense, budget=max_features)
    return Compiled(Reflex.compile(vocab, W, b, labels, temperature=T), head, C, "linear", choice or None)


def _fold(vocab: dict, Wm: np.ndarray, Wp: np.ndarray, dense: dict, budget: int | None = None):
    """Write the dense block's weights into the word rows: row[w] += Wp @ E[w] for every word with a vector, adding
    rows for words that never appeared in training. Exact because the dense input is the mean of E over the same
    bag the rows are averaged over. Returns (vocab in ascending-hash order, (R, F') weights)."""
    extra = [int(h) for h in dense if int(h) not in vocab]
    if budget is not None:  # a size budget: new rows only while it lasts, in the order given (training words first)
        extra = extra[:max(0, budget - len(vocab))]
        keep_ = set(extra)
        dense = {h: v for h, v in dense.items() if int(h) in vocab or int(h) in keep_}
    keys = np.array(sorted(list(vocab) + extra), dtype=np.int64)
    new_vocab = {int(h): i for i, h in enumerate(keys)}
    W = np.zeros((Wm.shape[0], len(keys)), np.float32)
    old_cols = np.array([vocab[h] for h in vocab], dtype=np.int64)
    W[:, [new_vocab[h] for h in vocab]] = Wm[:, old_cols]
    hs = list(dense)
    E = np.stack([dense[h] for h in hs]).astype(np.float32)  # (M, D)
    W[:, [new_vocab[int(h)] for h in hs]] += (Wp @ E.T).astype(np.float32)
    return new_vocab, W


def _prune_vocab(vocab: dict, keep: np.ndarray) -> dict:
    """The vocabulary restricted to the kept columns, renumbered in ascending-hash order (keep is sorted)."""
    col_hash = np.empty(len(vocab), dtype=np.int64)
    for h, j in vocab.items():
        col_hash[j] = h
    return {int(col_hash[j]): i for i, j in enumerate(keep)}


def _dan_logits(item_lists, vocab, W1, b1, W2, b2, rows, dense=None, P=None):
    """Float32 logits of the deep student on `rows`, computed as the table will serve them (plus the dense block
    through its projection P while it is not yet folded into the rows)."""
    out = np.empty((len(rows), len(b2)), np.float32)
    Tf = W1.T  # (F, H)
    for k, r in enumerate(rows):
        its = item_lists[r]
        cols = [vocab[int(h)] for h in its if int(h) in vocab]
        acc = Tf[cols].sum(0) / np.float32(max(1, len(its))) + b1 if cols else b1.copy()
        if P is not None:
            acc = acc + dense[r] @ P.T
            out[k] = np.maximum(acc, 0) @ W2 + b2  # not yet folded: training-time scores only
        else:
            out[k] = dense_layer(np.maximum(acc, np.float32(0)).astype(np.float32), W2, b2)
    return out
