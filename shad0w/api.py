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

from .features import as_items, check_kind, items
from .mathutil import softmax
from .reflex import Reflex

FORMAT_VERSION = 1  # manifest.json
TABLE_VERSION = 1  # .s0 header of a linear table; 2 = a deep table (hidden layer), see Reflex.save
TABLE_VERSIONS = (1, 2)


def question_items(q, x) -> np.ndarray:
    """The items question `q` sees for one input: a text's n-grams (plus its static word items when the question was
    trained with them), or item ids given directly."""
    its = as_items(x, q.text_kind)
    if q.static is not None and isinstance(x, str):
        its = np.concatenate([its, q.static.items(x)])
    return its


def to_bool(v) -> bool:
    """A yes/no label as a bool: True/False, 1/0, or the strings yes/no, true/false, y/n, 1/0 (case-insensitive)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("yes", "true", "y", "1")


def _load_reflex(path) -> Reflex:
    """Read a .s0 table, refusing unknown versions, impossible shapes and truncated or padded files."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"S0RX":
        raise ValueError(f"{path}: not a shad0w table")
    ver, F_, K, temp = struct.unpack_from("<IIIf", data, 4)
    if ver not in TABLE_VERSIONS:
        raise ValueError(f"{path}: table format version {ver} is not supported (this shad0w reads {TABLE_VERSIONS})")
    if not (2 <= K <= 1024 and 0 < F_ <= (1 << 22)) or not (temp > 0 and math.isfinite(temp)):
        raise ValueError(f"{path}: corrupt header (F={F_}, K={K}, temperature={temp})")
    o = 20
    H = 0
    if ver == 2:
        if len(data) < 24:
            raise ValueError(f"{path}: truncated table")
        (H,) = struct.unpack_from("<I", data, 20)
        if not (1 <= H <= 1024):
            raise ValueError(f"{path}: corrupt header (H={H})")
        o = 24
    W_ = H or K  # row width
    expect = o + 4 * F_ + F_ * W_ + 8 * W_ + (4 * H * K + 4 * K if H else 4 * K * K)
    if len(data) != expect:
        raise ValueError(f"{path}: truncated or padded table")
    keys = np.frombuffer(data, "<u4", F_, o).copy(); o += 4 * F_
    T = np.frombuffer(data, np.int8, F_ * W_, o).reshape(F_, W_).copy(); o += F_ * W_
    scale = np.frombuffer(data, "<f4", W_, o).copy(); o += 4 * W_
    bias = np.frombuffer(data, "<f4", W_, o).copy(); o += 4 * W_
    if H:
        W2 = np.frombuffer(data, "<f4", H * K, o).reshape(H, K).copy(); o += 4 * H * K
        b2 = np.frombuffer(data, "<f4", K, o).copy()
        G, tail = None, (W2, b2)
    else:
        G = np.frombuffer(data, "<f4", K * K, o).reshape(K, K).copy()
        W2 = b2 = None
        tail = (G,)
    if F_ > 1 and not bool(np.all(keys[1:] > keys[:-1])):
        raise ValueError(f"{path}: table keys are not sorted")
    if F_ and int(keys[-1]) >= (1 << 22):
        raise ValueError(f"{path}: table key out of range")
    if not (np.isfinite(scale).all() and np.isfinite(bias).all() and all(np.isfinite(t).all() for t in tail)):
        raise ValueError(f"{path}: table has non-finite weights")
    return Reflex(keys, T, scale, bias, temp, [], G, W2, b2)


class DriftGuard:
    """Label-free monitor (ATC-style, Garg et al. ICLR'22). On calibration data, pick the
    confidence threshold t whose below-t mass equals the observed error rate; on live
    traffic, estimated error = fraction of answers below t. Raises when a sliding window's
    estimate exceeds the calibration estimate by `margin` (absolute), or by 3 standard errors of the two estimates
    when that is larger: with a few hundred calibration rows the estimates alone differ by a few points, and a
    guard that fires on that noise sends certified traffic to the LLM for nothing."""

    def __init__(self, conf_cal: np.ndarray, correct_cal: np.ndarray, window: int = 500, margin: float = 0.03):
        err = 1.0 - float(np.mean(correct_cal))
        self.t = float(np.quantile(conf_cal, err)) if 0 < err < 1 else 0.0
        self.base = err
        self.n_cal = len(correct_cal)
        self.window, self.margin = window, margin
        self._init_buf()

    @property
    def margin_used(self) -> float:
        """The margin actually applied: `margin`, or 3 standard errors of (calibration, window) estimates."""
        if not self.n_cal:
            return self.margin
        b = min(max(self.base, 0.01), 0.99)
        se = math.sqrt(b * (1 - b) * (1.0 / self.n_cal + 1.0 / max(self.window, 1)))
        return max(self.margin, 3.0 * se)

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
        return len(self.buf) >= self.window // 2 and self.estimate > self.base + self.margin_used

    def reconfigure(self, window: int | None = None, margin: float | None = None) -> None:
        """Change the window or margin at runtime (the bundle on disk keeps the values it was trained with)."""
        if window is not None:
            self.window = int(window)
        if margin is not None:
            self.margin = float(margin)
        self._init_buf()

    def state(self):
        return {"t": self.t, "base": self.base, "window": self.window, "margin": self.margin, "n_cal": self.n_cal}

    @classmethod
    def from_state(cls, s):
        g = cls.__new__(cls)
        g.t, g.base, g.window, g.margin = s["t"], s["base"], s["window"], s["margin"]
        g.n_cal = s.get("n_cal", 0)  # bundles from before 0.4 have no n_cal: the plain margin applies
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
    static: object = None  # shad0w.static.StaticTable: static word vectors added as items (texts only)
    text_kind: int = 1  # text featurizer (shad0w.features): 1 for bundles without manifest meta "text_kind"
    _native: object = None
    _native_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def decide(self, text: str, its=None, exposed: bool = False, options: list[str] | None = None,
               probabilities: bool = True, observe: bool = True) -> dict:
        """options: an optional subset of the compiled options to decide among (request-time narrowing,
        as the /v1/decide wire format allows); probabilities are renormalised over the subset.
        probabilities=False skips building the per-option dict (the slowest part of a call on the C path).
        observe=False answers without feeding the drift guard (for explain / dry runs / pre-checks)."""
        if not isinstance(text, str):  # item ids (an image's, see shad0w.vision) instead of text
            its = as_items(text) if its is None else its
            empty = len(its) == 0
        else:
            empty = not text.strip()
        native = self._native is not None and (isinstance(text, str) or getattr(self._native, "_has_items", False))
        if native and isinstance(text, str) and self.static is not None and not getattr(self._native, "has_static", False):
            native = False  # a C build without the static table loaded: the numpy path gives the same answer
        if native and isinstance(text, str):
            with self._native_lock:
                y, p, r = self._native.decide(text)
                unseen = self._native.unseen
        elif native:
            with self._native_lock:
                y, p, r = self._native.decide_items(its)
                unseen = self._native.unseen
        else:
            its = items(text, kind=self.text_kind) if its is None else its
            if self.static is not None and isinstance(text, str):
                its = np.concatenate([its, self.static.items(text)])
            p = softmax(self.rx.logits(its))
            y, r = int(np.argmax(p)), self.rx.radius(its)
            hits, n_all = self.rx.rows(its)
            unseen = 1.0 - len(hits) / n_all if n_all else 1.0
        # The certificate covers the table's own answer at its full-option confidence. A narrowed request may pick
        # a different option or renormalise the confidence upwards; neither is covered, so certification is always
        # judged on the full-option answer and confidence.
        y_full = y
        conf = float(p[y])
        unknown = False
        if options is not None and self.qtype == "choice":  # [] = none of the requested options is known
            idx = [self.options.index(o) for o in options if o in self.options]
            unknown = not idx
            if idx and len(idx) < len(self.options):
                sub = np.asarray(p)[idx]
                tot = float(sub.sum())  # renormalise even when the subset's mass is tiny (no clamp)
                sub = sub / tot if tot > 0 else np.full(len(idx), 1.0 / len(idx))
                p = np.zeros_like(np.asarray(p)); p[idx] = sub
                y = int(idx[int(np.argmax(sub))])
        drift = self.guard.update(conf) if observe else self.guard.raised
        flag = None
        if not self.calibrated:
            flag = "uncalibrated"
        elif unknown:
            flag = "options_changed"
        elif empty:  # nothing to go on: the table would only echo its most common answer
            flag = "empty_input"
        elif y != y_full or not (conf >= self.threshold):  # fail closed on NaN
            flag = "low_confidence"
        elif exposed and not (r >= self.r_min):  # a deep table has no radius (NaN): never robust enough when exposed
            flag = "low_radius"
        elif drift:
            flag = "drift"
        out = {"confidence": conf if y == y_full else float(p[y]), "certified": flag is None, "flag": flag,
               "radius": None if not math.isfinite(r) else int(r), "unseen": round(float(unseen), 4)}
        if self.qtype == "yesno":
            out["probability"] = float(p[1])
            out["answer"] = bool(y == 1)
        else:
            out["choice"] = self.options[y]
            if probabilities:
                pl = np.round(p, 6).tolist()
                if options:
                    keep = set(options)
                    out["probabilities"] = {o: q for o, q in zip(self.options, pl) if o in keep}
                else:
                    out["probabilities"] = dict(zip(self.options, pl))
        return out


@dataclass
class Model:
    questions: dict[str, Question] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def decide(self, state, exposed: bool = False, questions: dict | None = None, probabilities: bool = True,
               observe: bool = True) -> dict:
        """questions: optional wire-format questions {name: {"criteria": {...}}}; when given, each named question
        is narrowed to the requested criteria keys (a subset of its compiled options)."""
        spec = self.image_spec
        if isinstance(state, np.ndarray) and state.dtype == np.uint32 and state.ndim == 1:  # item ids from the caller
            text = its = state
        elif spec is not None and not isinstance(state, (str, dict, list)):  # an image: pixels -> item ids
            from .vision import image_items
            text = its = image_items(state, spec)
        else:
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            native = all(q._native is not None for q in self.questions.values())
            kinds = {q.text_kind for q in self.questions.values()}
            # Python path: hash once, share across questions (they share one featurizer in every bundle shad0w writes)
            its = None if native or len(kinds) != 1 else items(text, kind=kinds.pop())
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
            out[n] = q.decide(text, its, exposed, opts, probabilities, observe)
        return {"answers": out}

    def calibrate(self, question: str, texts, labels, alpha: float | None = None, delta: float = 0.1):
        """Turn the guarantee on with REAL labelled examples (~300 suffice in our tests).
        Certificates fitted on pseudo-labels or synthetic data were invalid in every test."""
        from .reliability import SelectiveRiskController
        q = self.questions[question]
        alpha = q.alpha if alpha is None else alpha
        y = np.array([q.options.index(v) if q.qtype == "choice" else int(to_bool(v)) for v in labels])
        P = np.stack([softmax(q.rx.logits(question_items(q, t))) for t in texts])
        conf, correct = P.max(1), P.argmax(1) == y
        q.threshold = float(SelectiveRiskController(alpha, delta).fit(conf, correct).threshold)
        q.alpha, q.guard, q.calibrated = alpha, DriftGuard(conf, correct), True
        return {"n": len(y), "accuracy": float(correct.mean()), "threshold": q.threshold,
                "certified_share": float((conf >= q.threshold).mean())}

    @property
    def image_spec(self):
        """The ImageSpec this bundle's images are featurised with (None for a text bundle)."""
        d = self.meta.get("image")
        if not d:
            return None
        if getattr(self, "_image_spec", None) is None:
            from .vision import ImageSpec
            self._image_spec = ImageSpec.from_dict(d)
        return self._image_spec

    @property
    def native(self) -> bool:
        """True when every question decides through the C core."""
        return bool(self.questions) and all(q._native is not None for q in self.questions.values())

    def use_native(self, bundle_dir: str):
        from .native import NativeReflex
        for n, q in self.questions.items():
            sw = os.path.join(bundle_dir, f"{n}.s0w") if q.static is not None else None
            q._native = NativeReflex(os.path.join(bundle_dir, f"{n}.s0"), static=sw, text_kind=q.text_kind)
        return self

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        kinds = {q.text_kind for q in self.questions.values()}
        if len(kinds) > 1:
            raise ValueError(f"questions of one bundle must share a text_kind, got {sorted(kinds)}")
        meta = dict(self.meta)
        if kinds and kinds != {1}:
            meta["text_kind"] = kinds.pop()  # absent = 1, so older runtimes reading a kind-1 bundle see no change
        elif kinds:
            meta.pop("text_kind", None)
        manifest = {"format": FORMAT_VERSION, "meta": meta, "questions": {}}
        for n, q in self.questions.items():
            q.rx.save(os.path.join(path, f"{n}.s0"))
            thr = q.threshold if math.isfinite(q.threshold) else None  # JSON has no Infinity; null = certifies nothing
            manifest["questions"][n] = {"type": q.qtype, "options": q.options, "threshold": thr,
                                        "alpha": q.alpha, "r_min": q.r_min, "guard": q.guard.state(),
                                        "calibrated": q.calibrated}
            if q.static is not None:
                q.static.save(os.path.join(path, f"{n}.s0w"))
                manifest["questions"][n]["static"] = {"model": q.static.model, "file": f"{n}.s0w"}
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, allow_nan=False)


def load(path: str, native: bool = True) -> Model:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    if man.get("format", 1) != FORMAT_VERSION:
        raise ValueError(f"{path}: bundle format {man.get('format')} is not supported by this shad0w")
    m = Model(meta=man.get("meta", {}))
    text_kind = check_kind(m.meta.get("text_kind", 1))  # an unknown featurizer is refused, never guessed
    for n, q in man["questions"].items():
        rx = _load_reflex(os.path.join(path, f"{n}.s0"))
        rx.labels = q["options"]
        m.questions[n] = Question(n, q["type"], q["options"], rx, math.inf if q["threshold"] is None else q["threshold"], q["alpha"],
                                  DriftGuard.from_state(q["guard"]), q.get("r_min", 0.0),
                                  q.get("calibrated", True), text_kind=text_kind)
        if q.get("static"):
            from .static import StaticTable
            m.questions[n].static = StaticTable.load(os.path.join(path, q["static"]["file"]), q["static"]["model"])
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
            y = np.array([options.index(v) if qtype == "choice" else int(to_bool(v)) for v in ys])
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
