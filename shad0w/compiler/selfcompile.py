"""Label-free compile ("logs" mode): option names (+ optional one-line criteria) and UNLABELLED texts -> labels.

1. Embed texts and options with a frozen local sentence encoder (shad0w/compiler/encoders.py).
2. Anchored prototype refinement with Sinkhorn balancing (shad0w/compiler/prototypes.py). The class prior is either
   uniform or estimated from the zero-shot assignment; the one the hashed-table view agrees with more is kept
   (label-free model selection).
3. One co-training round with the hashed-n-gram table.
Returns one pseudo-label per text; the caller compiles the table on them. Certificates for this mode still
need REAL labels (shad0w calibrate): certificates fitted on pseudo-labels are not valid.
"""
from __future__ import annotations

import numpy as np

from .encoders import DEFAULT


def pseudo_label(texts, options, spec=None, beta=0.3, encoder=DEFAULT, prior="select", return_info=False):
    from .distill import compile_schema
    from .encoders import Encoder
    from .prototypes import refine

    crit = (spec or {}).get("criteria") or {}
    opt_text = [f"{o.replace('_', ' ')}: {crit[o]}" if crit.get(o) else o.replace("_", " ") for o in options]
    enc = Encoder(encoder)
    E, L = enc.encode(list(texts), "state"), enc.encode(opt_text, "option")

    def one(pr):
        a, conf, _ = refine(E, L, beta=beta, balance=True, prior=pr)
        keep = conf >= np.quantile(conf, 0.3)
        comp = compile_schema([t for t, k in zip(texts, keep) if k], a[keep], options, C=300.0)
        pr_, _, _ = comp.predict(list(texts))
        agree = pr_.argmax(1) == a
        out = np.where(agree, a, pr_.argmax(1) if agree.mean() < 0.5 else a)
        return out, float(agree.mean())

    if prior == "select":
        cands = {"uniform": one(None), "auto": one("auto")}
        chosen = max(cands, key=lambda k: cands[k][1])
        out, info = cands[chosen][0], {"prior": chosen, "agreement": {k: v[1] for k, v in cands.items()}}
    else:
        out, agree = one(prior)
        info = {"prior": "uniform" if prior is None else str(prior), "agreement": {"chosen": agree}}
    return (out, info) if return_info else out
