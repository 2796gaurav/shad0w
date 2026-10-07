"""shad0w model API: self-contained, no network, no external model at runtime.

    import shad0w
    schema = {"intent": {"type": "choice", "instructions": "...",
                         "criteria": {"refund": "money back", "lost_card": None, ...}},
              "urgent": {"type": "yesno", "instructions": "Is this urgent?"}}
    model = shad0w.compile(schema, labeled={"intent": (texts, labels), "urgent": (texts, [0/1,...])})
    model.save("bundle/");  model = shad0w.load("bundle/")
    model.decide("I lost my card, please block it!")
    -> {"answers": {"intent": {"choice": "lost_card", "probabilities": {...}, "confidence": 0.97,
                               "certified": True, "radius": 6, "flag": None}, ...}}

Every answer is always returned. `certified` means: the answer clears the threshold that
held the target error rate on calibration data AND the drift guard has not raised.
`flag` explains an uncertified answer: "low_confidence", "low_radius" (exposed channels),
"drift". Optional `feedback()` feeds the anytime-valid auditor when labels exist.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from .features import items
from .mathutil import softmax
from .reflex import Reflex

FORMAT_VERSION = 1  # manifest.json
TABLE_VERSION = 1  # .s0 header


def _load_reflex(path) -> Reflex:
    """Read a .s0 table, refusing unknown versions, impossible shapes and truncated or padded files."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"S0RX":
        raise ValueError(f"{path}: not a shad0w table")
    ver, F_, K, temp = struct.unpack_from("<IIIf", data, 4)
    if ver != TABLE_VERSION:
        raise ValueError(f"{path}: table format version {ver} is not supported (this shad0w reads {TABLE_VERSION})")
    if not (2 <= K <= 1024 and 0 < F_ <= (1 << 22)) or not (temp > 0 and math.isfinite(temp)):
        raise ValueError(f"{path}: corrupt header (F={F_}, K={K}, temperature={temp})")
    if len(data) != 20 + 4 * F_ + F_ * K + 8 * K + 4 * K * K:
        raise ValueError(f"{path}: truncated or padded table")
    o = 20
    keys = np.frombuffer(data, "<u4", F_, o).copy(); o += 4 * F_
    T = np.frombuffer(data, np.int8, F_ * K, o).reshape(F_, K).copy(); o += F_ * K
    scale = np.frombuffer(data, "<f4", K, o).copy(); o += 4 * K
    bias = np.frombuffer(data, "<f4", K, o).copy(); o += 4 * K
    G = np.frombuffer(data, "<f4", K * K, o).reshape(K, K).copy()
    if F_ > 1 and not bool(np.all(keys[1:] > keys[:-1])):
        raise ValueError(f"{path}: table keys are not sorted")
    return Reflex(keys, T, scale, bias, temp, [], G)


class DriftGuard:
    """Label-free monitor (ATC-style, Garg et al. ICLR'22). On calibration data, pick the
    confidence threshold t whose below-t mass equals the observed error rate; on live
    traffic, estimated error = fraction of answers below t. Raises when a sliding window's
    estimate exceeds the calibration estimate by `margin` (absolute)."""

    def __init__(self, conf_cal: np.ndarray, correct_cal: np.ndarray, window: int = 500, margin: float = 0.03):
        err = 1.0 - float(np.mean(correct_cal))
        self.t = float(np.quantile(conf_cal, err)) if 0 < err < 1 else 0.0
        self.base = err
        self.window, self.margin = window, margin
        self._init_buf()

    def _init_buf(self):
        from collections import deque
        self.buf, self.low = deque(maxlen=self.window), 0  # ring buffer of "below t" bits
        self._lock = threading.Lock()  # the HTTP server and app threads share one guard

    def update(self, conf: float) -> bool:
        b = conf < self.t
        with self._lock:
            if len(self.buf) == self.window:
                self.low -= self.buf[0]
            self.buf.append(b)
            self.low += b
            return self.raised

    @property
    def estimate(self) -> float:
        return self.low / len(self.buf) if self.buf else self.base

    @property
    def raised(self) -> bool:
        return len(self.buf) >= self.window // 2 and self.estimate > self.base + self.margin

    def state(self):
        return {"t": self.t, "base": self.base, "window": self.window, "margin": self.margin}

    @classmethod
    def from_state(cls, s):
        g = cls.__new__(cls)
        g.t, g.base, g.window, g.margin = s["t"], s["base"], s["window"], s["margin"]
        g._init_buf()
        return g


@dataclass
class Question:
    name: str
    qtype: str  # "choice" | "yesno"
    options: list[str]
    rx: Reflex
    threshold: float
    alpha: float
    guard: DriftGuard
    r_min: float = 0.0
    calibrated: bool = True  # False when thresholds come from pseudo-labels (label-free compile)
    _native: object = None
    _native_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def decide(self, text: str, its=None, exposed: bool = False, options: list[str] | None = None) -> dict:
        """options: an optional subset of the compiled options to decide among (request-time narrowing,
        as the /v1/decide wire format allows); probabilities are renormalised over the subset."""
        if self._native is not None:
            with self._native_lock:
                y, p, r = self._native.decide(text)
        else:
            its = items(text) if its is None else its
            p = softmax(self.rx.logits(its))
            y, r = int(np.argmax(p)), self.rx.radius(its)
        if options and self.qtype == "choice":
            idx = [self.options.index(o) for o in options if o in self.options]
            if idx and len(idx) < len(self.options):
                sub = np.asarray(p)[idx]
                sub = sub / max(float(sub.sum()), 1e-12)
                p = np.zeros_like(np.asarray(p)); p[idx] = sub
                y = int(idx[int(np.argmax(sub))])
        conf = float(np.max(p))
        drift = self.guard.update(conf)
        flag = None
        if not self.calibrated:
            flag = "uncalibrated"
        elif conf < self.threshold:
            flag = "low_confidence"
        elif exposed and r < self.r_min:
            flag = "low_radius"
        elif drift:
            flag = "drift"
        out = {"confidence": conf, "certified": flag is None, "flag": flag,
               "radius": None if math.isinf(r) else int(r)}
        if self.qtype == "yesno":
            out["probability"] = float(p[1])
            out["answer"] = bool(y == 1)
        else:
            out["choice"] = self.options[y]
            keep = set(options) if options else None
            out["probabilities"] = {o: round(float(q), 6) for o, q in zip(self.options, p) if keep is None or o in keep}
        return out


@dataclass
class Model:
    questions: dict[str, Question] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def decide(self, state, exposed: bool = False, questions: dict | None = None) -> dict:
        """questions: optional wire-format questions {name: {"criteria": {...}}}; when given, each named question
        is narrowed to the requested criteria keys (a subset of its compiled options)."""
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        native = all(q._native is not None for q in self.questions.values())
        its = None if native else items(text)  # Python path: hash once, share across questions
        out = {}
        for n, q in self.questions.items():
            if questions is not None and n not in questions:
                continue
            opts = None
            if questions and isinstance(questions.get(n), dict):
                crit = questions[n].get("criteria")
                if isinstance(crit, dict):
                    opts = list(crit.keys())
                elif isinstance(crit, list):
                    opts = list(crit)
            out[n] = q.decide(text, its, exposed, opts)
        return {"answers": out}

    def calibrate(self, question: str, texts, labels, alpha: float | None = None, delta: float = 0.1):
        """Turn the guarantee on with REAL labelled examples (~300 suffice in our tests).
        Certificates fitted on pseudo-labels or synthetic data were invalid in every test."""
        from .reliability import SelectiveRiskController
        q = self.questions[question]
        alpha = q.alpha if alpha is None else alpha
        y = np.array([q.options.index(v) if q.qtype == "choice" else int(bool(v)) for v in labels])
        P = np.stack([softmax(q.rx.logits(items(t))) for t in texts])
        conf, correct = P.max(1), P.argmax(1) == y
        q.threshold = float(SelectiveRiskController(alpha, delta).fit(conf, correct).threshold)
        q.alpha, q.guard, q.calibrated = alpha, DriftGuard(conf, correct), True
        return {"n": len(y), "accuracy": float(correct.mean()), "threshold": q.threshold,
                "certified_share": float((conf >= q.threshold).mean())}

    @property
    def native(self) -> bool:
        """True when every question decides through the C core."""
        return bool(self.questions) and all(q._native is not None for q in self.questions.values())

    def use_native(self, bundle_dir: str):
        from .native import NativeReflex
        for n, q in self.questions.items():
            q._native = NativeReflex(os.path.join(bundle_dir, f"{n}.s0"))
        return self

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        manifest = {"format": FORMAT_VERSION, "meta": self.meta, "questions": {}}
        for n, q in self.questions.items():
            q.rx.save(os.path.join(path, f"{n}.s0"))
            manifest["questions"][n] = {"type": q.qtype, "options": q.options, "threshold": q.threshold,
                                        "alpha": q.alpha, "r_min": q.r_min, "guard": q.guard.state(),
                                        "calibrated": q.calibrated}
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)


def load(path: str, native: bool = True) -> Model:
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    if man.get("format", 1) != FORMAT_VERSION:
        raise ValueError(f"{path}: bundle format {man.get('format')} is not supported by this shad0w")
    m = Model(meta=man.get("meta", {}))
    for n, q in man["questions"].items():
        rx = _load_reflex(os.path.join(path, f"{n}.s0"))
        rx.labels = q["options"]
        m.questions[n] = Question(n, q["type"], q["options"], rx, q["threshold"], q["alpha"],
                                  DriftGuard.from_state(q["guard"]), q.get("r_min", 0.0),
                                  q.get("calibrated", True))
    if native:
        try:
            m.use_native(path)
        except OSError:
            pass  # no C core: the numpy path makes the same decisions
    return m


def compile(schema: dict, labeled: dict | None = None, unlabeled: list[str] | None = None, encoder: str | None = None,
            alpha: float = 0.05, delta: float = 0.1, r_min: float = 8.0, seed: int = 0) -> Model:
    """Compile a typed schema into a self-contained shad0w model.

    labeled[q] = (texts, labels) where labels are option names (choice) or 0/1 (yesno).
    Without labels for a question, `unlabeled` texts + option names are used (LLM-free
    anchored-prototype compile; needs the [logs] extra; see shad0w/compiler/selfcompile.py).
    """
    from .compiler.distill import compile_schema
    from .reliability import SelectiveRiskController

    rng = np.random.default_rng(seed)
    model = Model(meta={"alpha": alpha, "delta": delta})
    for name, spec in schema.items():
        qtype = spec.get("type", "choice")
        options = list(spec["criteria"].keys()) if qtype == "choice" else ["no", "yes"]
        if labeled and name in labeled:
            texts, ys = labeled[name]
            y = np.array([options.index(v) if qtype == "choice" else int(bool(v)) for v in ys])
        elif unlabeled is not None:
            from .compiler.selfcompile import pseudo_label
            texts = list(unlabeled)
            y = pseudo_label(texts, options, spec, **({"encoder": encoder} if encoder else {}))
        else:
            raise ValueError(f"question {name!r}: give labeled data or unlabeled texts")
        perm = rng.permutation(len(texts))
        n_cal = max(int(0.2 * len(texts)), min(300, len(texts) // 2))
        cal, fit_i = perm[:n_cal], perm[n_cal:]
        comp = compile_schema([texts[i] for i in fit_i], y[fit_i], options)
        p_cal = np.stack([softmax(comp.rx.logits(items(texts[i]))) for i in cal])
        conf, correct = p_cal.max(1), p_cal.argmax(1) == y[cal]
        thr = float(SelectiveRiskController(alpha, delta).fit(conf, correct).threshold)
        model.questions[name] = Question(name, qtype, options, comp.rx, thr, alpha,
                                         DriftGuard(conf, correct), r_min,
                                         calibrated=bool(labeled and name in labeled))
    return model
