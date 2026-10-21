"""Streams: certified reuse of the previous answer across video frames, in microseconds, no neural network.

Consecutive frames of a camera, a screen or a game mostly carry the same answer. A vision LLM re-answers each one
from scratch; this module answers a frame from what was already decided when that is certifiably safe, in a cascade
of three tiers over the same shad0w tables:

  tier 0  reuse   "is the answer still the one served on the current keyframe?" decided from DELTA items between
                  this frame and the keyframe (dHash Hamming distance bands, fern-code agreement, thumbnail
                  difference, age, and the keyframe's answer), scored by a yes/no table;
  tier 1  table   the ordinary frame table (shad0w.vision.ImageSpec items), as for a single image;
  else    teacher the vision model, whose answer is served and becomes reusable.

Why keyframes and not "the previous frame": comparing to the previous frame lets a slow drift pass unnoticed one
step at a time (the answer goes stale without any single delta looking large), and makes the comparison pairs seen
at run time depend on the thresholds, so no offline calibration can describe them. A keyframe is taken by a rule
that knows nothing about thresholds: the first frame, then whenever the current keyframe is `block` frames old or the
dHash has moved by more than `hamming` bits. At a keyframe tier 0 is not available (the table or the teacher answers
and the frame becomes the keyframe); between keyframes tier 0 may reuse the keyframe's answer. The same rule runs on
the calibration clips, so the (keyframe, frame) pairs of calibration are drawn exactly as the ones at run time.

One certificate for the cascade (reliability.CascadeController, Learn-then-Test over one risk level with Clopper-
Pearson tests, strict to lenient). Two things make a stream different from a bag of images and are handled here:
  * adjacent frames are not exchangeable, so calibration counts ONE frame per block of `block` frames (drawn at
    random inside the block) and reports disagreement on transition frames (where the teacher's answer changed since
    the previous labelled frame) separately;
  * a reused answer is only as good as the keyframe's served answer, which at run time is the frame table's when it
    cleared its threshold at the keyframe and the teacher's otherwise. Whether the table did so depends on the very
    threshold under test, so the test step re-derives tier 0's agreement for every candidate (the path step, which
    only picks the candidate sequence, uses the teacher's keyframe answer). Propagated table errors are inside the
    bound, not beside it.

Per-frame cost: thumbnail, dHash, fern codes and a few dozen numpy ops; about 100-300 us a frame here. A C port needs
nothing new beyond s0_image_items: keep the keyframe's thumbnail (size x size x channels bytes), its 64-bit dHash and
fern codes, and add a popcount, a loop over the fern codes and over the thumbnail, and one more table lookup: a few
microseconds on top of the image decision.

    model, cert = stream_compile(frames_by_clip, answers_by_clip, options, block=15)   # teacher-labelled clips
    s = Stream(model, teacher=my_vlm)
    for frame in camera:
        d = s.decide(frame)        # d.answer, d.source in {"reuse", "table", "teacher"}, d.certified, d.latency_us
    s.reset()                      # at a clip boundary: the next frame is a keyframe
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .api import DriftGuard, Model, Question, load
from .features import MASK
from .mathutil import softmax
from .reliability import CascadeController, _tolerant_start, binom_ucb
from .vision import _PAIRS, ImageSpec, _mix, _pairs, image_items_py, pixels, thumbnail

_log = logging.getLogger("shad0w")

REUSE_SUFFIX = "@reuse"
DELTA_PREFIX = 0xA0000000  # item-space prefix of delta items (vision uses 0xF0, 0xE0, 0xB0, 0xD0)
HD_BINS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 33)                      # dHash Hamming distance (of 64 bits)
HB_BINS = (0, 1, 2, 4, 7)                                            # per 16-bit band
MAD_BINS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)  # mean |thumbnail difference|
MAX_BINS = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192)                  # max |thumbnail difference|
MOVED_BINS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # share of thumbnail pixels moved by > 24
AGE_BINS = (1, 2, 4, 8, 16, 32, 64)                                  # frames since the keyframe
FERN_LEVELS = 16                                                     # fern-code agreement, quantised
MOVED_AT = 24


# ---------- one frame's state: thumbnail, dHash, fern codes and the tier-1 items, computed once ----------------

@dataclass
class FrameState:
    thumb: np.ndarray   # (size, size, channels) uint8, exactly vision.thumbnail
    dhash: int          # the 64-bit difference hash
    codes: np.ndarray   # (ferns,) fern codes
    items: np.ndarray   # (n_items,) uint32, exactly vision.image_items


_SAMPLES: dict = {}


def _samples(n: int, S: int) -> np.ndarray:
    """(S, 2) source indices per thumbnail cell, the two samples vision.thumbnail (ferns/2) takes per axis; a cell
    one pixel wide repeats its pixel, so a fixed 2 x 2 gather and `// 4` give the reference's integer means."""
    key = (n, S)
    out = _SAMPLES.get(key)
    if out is None:
        out = np.empty((S, 2), dtype=np.int64)
        for i in range(S):
            lo, hi = i * n // S, (i + 1) * n // S
            hi = max(hi, lo + 1)
            out[i] = (lo, lo + max(1, (hi - lo) // 2) if hi - lo >= 2 else lo)
        _SAMPLES[key] = out
    return out


def _thumb(a: np.ndarray, spec: ImageSpec) -> np.ndarray:
    """vision.thumbnail without its per-cell Python loop (gather 2 x 2 samples per cell, then reduce)."""
    if spec.version < 2:
        return thumbnail(a, spec)
    h, w = a.shape[:2]
    S, C = spec.size, spec.channels
    src = a if a.ndim == 3 else a[:, :, None]
    g = src[_samples(h, S)][:, :, _samples(w, S)]  # (S, 2, S, 2, C_in)
    if C == 1:
        if g.shape[-1] >= 3:
            g16 = g[..., :3].astype(np.int32)
            g = ((g16[..., 0] + 2 * g16[..., 1] + g16[..., 2]) >> 2)[..., None]
        else:
            g = g[..., :1].astype(np.int32)
    else:
        g = (g[..., :3] if g.shape[-1] >= 3 else np.repeat(g[..., :1], 3, axis=-1)).astype(np.int32)
    return (g.sum(axis=(1, 3)) // 4).astype(np.uint8)


def _block_means(x: np.ndarray, R: int, Cc: int) -> np.ndarray:
    """Integer means of an (S, S[, C]) array over an R x Cc grid of blocks with the edges r * S // R, c * S // Cc."""
    S = x.shape[0]
    rs = np.fromiter((r * S // R for r in range(R)), dtype=np.int64, count=R)
    cs = np.fromiter((c * S // Cc for c in range(Cc)), dtype=np.int64, count=Cc)
    sums = np.add.reduceat(np.add.reduceat(x.astype(np.int64), rs, axis=0), cs, axis=1)
    sizes = np.outer(np.diff(np.append(rs, S)), np.diff(np.append(cs, S)))
    return sums // (sizes if x.ndim == 2 else sizes[:, :, None])


def frame_state(frame, spec: ImageSpec | None = None) -> FrameState:
    """Everything a stream needs from one frame, computed once (tests check items == vision.image_items)."""
    spec = spec or ImageSpec()
    a = pixels(frame)
    if spec.version < 2:  # the older kind (4 x 4 samples per cell): the reference path, plus a dHash for the stream
        th = thumbnail(a, spec)
        its = image_items_py(a, spec)
        grey = th[:, :, 0].astype(np.uint32) if spec.channels == 1 else (
            (th[:, :, 0].astype(np.uint32) + 2 * th[:, :, 1] + th[:, :, 2]) >> 2)
        bands = _bands_of(grey)
        return FrameState(th, _dhash64(bands), _codes(th, spec), its)
    th = _thumb(a, spec)
    codes = _codes(th, spec)
    fern = (np.arange(spec.ferns, dtype=np.uint64) << np.uint64(spec.bits)) | codes.astype(np.uint64)
    out = [_mix(np.uint64(0xF0000000) ^ fern)]
    G, L, C = spec.grid, spec.levels, spec.channels
    if G:
        q = (_block_means(th, G, G).astype(np.uint64) * np.uint64(L)) >> np.uint64(8)
        j = np.arange(G * G * C, dtype=np.uint64)
        out.append(_mix(np.uint64(0xE0000000) ^ (j * np.uint64(L) + q.reshape(-1))))
    t = th.astype(np.uint32)
    grey = t[:, :, 0] if C == 1 else (t[:, :, 0] + 2 * t[:, :, 1] + t[:, :, 2]) >> 2
    bands = _bands_of(grey)  # the stream always needs the dHash, even when the spec has no dHash items
    if spec.dhash:
        out.append(_mix(np.uint64(0xB0000000) ^ bands))
    its = (np.concatenate(out) & np.uint64(MASK)).astype(np.uint32)
    return FrameState(th, _dhash64(bands), codes, its)


def _codes(th: np.ndarray, spec: ImageSpec) -> np.ndarray:
    if spec not in _PAIRS:
        _PAIRS[spec] = _pairs(spec)
    pa, pb = _PAIRS[spec]
    flat = th.reshape(-1).astype(np.int32)
    bits = flat[pa] > flat[pb] + spec.margin
    return (bits.astype(np.uint32) << np.arange(spec.bits, dtype=np.uint32)).sum(1).astype(np.uint32)


def _bands_of(grey: np.ndarray) -> np.ndarray:
    """vision._dhash_bands from a grey thumbnail (uint32, (S, S)), vectorised."""
    g = _block_means(grey, 8, 9)
    bits = (g[:, :8] > g[:, 1:]).reshape(4, 16).astype(np.uint64)
    return (np.arange(4, dtype=np.uint64) << np.uint64(16)) | (bits << np.arange(16, dtype=np.uint64)).sum(1)


def _dhash64(bands: np.ndarray) -> int:
    b = [int(x) & 0xFFFF for x in bands]
    return b[0] | (b[1] << 16) | (b[2] << 32) | (b[3] << 48)


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _bin(bins, x) -> int:
    return int(np.searchsorted(bins, x, side="right"))


def delta_items(cur: FrameState, key: FrameState, answer_id: int, age: int, hd: int | None = None) -> np.ndarray:
    """The tier-0 items of `cur` against keyframe `key` whose served answer is option `answer_id`, `age` frames old.
    Each is (tag, value) hashed with vision._mix under DELTA_PREFIX; the pairs with the answer id let the table learn
    that some answers persist more than others (an empty scene stays empty longer than a passing bicycle)."""
    x = cur.dhash ^ key.dhash
    hd = _popcount(x) if hd is None else hd
    d = np.abs(cur.thumb.astype(np.int16) - key.thumb.astype(np.int16))
    fa = int((cur.codes == key.codes).sum()) * FERN_LEVELS // (len(cur.codes) + 1)
    b_hd, b_mad, b_max = _bin(HD_BINS, hd), _bin(MAD_BINS, float(d.mean())), _bin(MAX_BINS, int(d.max()))
    b_mv, b_age = _bin(MOVED_BINS, float((d > MOVED_AT).mean())), _bin(AGE_BINS, age)
    vals = [(0, b_hd), (1, _popcount(x & 0xFFFF)), (1, 16 | _popcount((x >> 16) & 0xFFFF)),
            (1, 32 | _popcount((x >> 32) & 0xFFFF)), (1, 48 | _popcount(x >> 48)), (2, fa), (3, b_mad),
            (4, b_max), (5, b_mv), (6, b_age), (7, answer_id), (8, (answer_id << 8) | b_hd),
            (9, (answer_id << 8) | b_mad), (10, (answer_id << 8) | fa), (11, (answer_id << 8) | b_mv)]
    payload = np.fromiter(((t << 20) | v for t, v in vals), dtype=np.uint64, count=len(vals))
    return (_mix(np.uint64(DELTA_PREFIX) ^ payload) & np.uint64(MASK)).astype(np.uint32)


# ---------- the run-time object ---------------------------------------------------------------------------------

@dataclass
class Decision:
    """One frame's answer. `certified` is True only for answers the certificate covers (reused or from the frame
    table at or above its threshold); a teacher's answer is the reference, not a certified one."""
    answer: object
    source: str            # "reuse" | "table" | "teacher"
    tier: int | None       # 0 = reuse, 1 = frame table, None = teacher
    certified: bool
    confidence: float | None
    latency_us: float
    frame: int             # index in the stream since the last reset
    keyframe: int | None   # index of the keyframe this frame was compared with
    flag: str | None = None

    def to_dict(self) -> dict:
        return {"answer": self.answer, "source": self.source, "tier": self.tier, "certified": self.certified,
                "confidence": self.confidence, "latency_us": self.latency_us, "frame": self.frame,
                "keyframe": self.keyframe, "flag": self.flag}


@dataclass
class _Key:
    index: int
    state: FrameState
    answer: object
    answer_id: int | None  # None: the answer is not a known option or not covered, so it must not be reused


class Stream:
    """Decide a sequence of frames with a stream bundle (see stream_compile). `teacher(frame) -> answer` is called
    when neither tier certifies; without one the frame table's best guess is served with certified=False."""

    def __init__(self, model_or_bundle, question: str | None = None, spec: ImageSpec | None = None, teacher=None,
                 block: int | None = None, hamming: int | None = None, reuse: bool = True, observe: bool = True):
        self.model = load(model_or_bundle, native=False) if isinstance(model_or_bundle, str) else model_or_bundle
        meta = self.model.meta.get("stream", {})
        names = [n for n in self.model.questions if not n.endswith(REUSE_SUFFIX)]
        if question is None:
            question = meta.get("question") or (names[0] if names else None)
        if question not in self.model.questions:
            raise ValueError(f"no question {question!r} in this bundle (it has {sorted(self.model.questions)})")
        self.name = question
        self.q1: Question = self.model.questions[question]
        self.q0: Question | None = self.model.questions.get(question + REUSE_SUFFIX) if reuse else None
        self.spec = spec or self.model.image_spec or ImageSpec()
        self.block = int(block if block is not None else meta.get("block", 15))
        self.hamming = int(hamming if hamming is not None else meta.get("hamming", 12))
        self.teacher, self.observe = teacher, observe
        self._options = {o: i for i, o in enumerate(self.q1.options)}
        self.reset()
        self._counts = {"reuse": 0, "table": 0, "teacher": 0, "uncertified": 0, "frames": 0, "keyframes": 0,
                        "changes": 0, "latency_us": 0.0}

    def reset(self) -> None:
        """A clip boundary: forget the keyframe, so the next frame is answered by the table or the teacher."""
        self._key: _Key | None = None
        self._n = 0
        self._last = None

    @property
    def reuse_enabled(self) -> bool:
        """True when the bundle has a reuse table whose threshold certifies something."""
        return self.q0 is not None and math.isfinite(self.q0.threshold)

    def decide(self, frame) -> Decision:
        t0 = time.perf_counter()
        st = frame_state(frame, self.spec)
        i, key = self._n, self._key
        self._n += 1
        hd = age = None
        if key is not None:
            age = i - key.index
            hd = _popcount(st.dhash ^ key.state.dhash)
        new_key = key is None or age >= self.block or hd > self.hamming  # the rule knows nothing of thresholds
        out = None
        if not new_key and key.answer_id is not None and self.q0 is not None:
            p = softmax(self.q0.rx.logits(delta_items(st, key.state, key.answer_id, age, hd)))[1]
            raised = self.q0.guard.update(float(p)) if self.observe else self.q0.guard.raised
            if p >= self.q0.threshold and not raised:
                out = Decision(key.answer, "reuse", 0, True, float(p), 0.0, i, key.index)
        if out is None:
            d = self.q1.decide(st.items, its=st.items, probabilities=False, observe=self.observe)
            ans = d["choice"] if self.q1.qtype == "choice" else d["answer"]
            if d["certified"]:
                out = Decision(ans, "table", 1, True, d["confidence"], 0.0, i, key.index if not new_key else None)
            elif self.teacher is not None:
                ans = self.teacher(frame)
                if self.q1.qtype == "yesno" and not isinstance(ans, bool):
                    ans = str(ans).strip().lower() in ("yes", "true", "y", "1")
                out = Decision(ans, "teacher", None, False, None, 0.0, i, key.index if not new_key else None)
            else:
                out = Decision(ans, "table", 1, False, d["confidence"], 0.0, i, key.index if not new_key else None,
                               d.get("flag"))
        if new_key:  # covered answers only: a known option served by the table (certified) or by the teacher
            aid = self._answer_id(out.answer) if (out.source == "teacher" or out.certified) else None
            self._key = _Key(i, st, out.answer, aid)
            self._counts["keyframes"] += 1
        out.latency_us = (time.perf_counter() - t0) * 1e6
        c = self._counts
        c["frames"] += 1
        c[out.source if out.certified or out.source == "teacher" else "uncertified"] += 1
        c["latency_us"] += out.latency_us
        if self._last is not None and out.answer != self._last:
            c["changes"] += 1
        self._last = out.answer
        return out

    def _answer_id(self, ans) -> int | None:
        if self.q1.qtype == "yesno":
            return int(bool(ans))
        return self._options.get(ans)

    def stats(self) -> dict:
        """Counts and shares since construction (reset() keeps them: it is a clip boundary, not a new stream)."""
        c, n = self._counts, max(self._counts["frames"], 1)
        return {"frames": c["frames"], "keyframes": c["keyframes"], "answer_changes": c["changes"],
                "reuse": c["reuse"], "table": c["table"], "teacher": c["teacher"], "uncertified": c["uncertified"],
                "share": {k: c[k] / n for k in ("reuse", "table", "teacher", "uncertified")},
                "mean_latency_us": c["latency_us"] / n, "block": self.block, "hamming": self.hamming,
                "reuse_enabled": self.q0 is not None}

    compile = None  # set below: Stream.compile(...) is stream_compile(...)


# ---------- compiling and certifying from teacher-labelled clips -------------------------------------------------

@dataclass
class _Rows:
    """Calibration rows (one per block) in the form the certificate needs."""
    conf1: np.ndarray            # frame table confidence
    agree1: np.ndarray           # frame table answer == teacher
    conf0_t: np.ndarray          # reuse confidence when the keyframe carries the TEACHER's answer (NaN: no tier 0)
    conf0_k: np.ndarray          # ... when it carries the frame TABLE's answer
    agree0_t: np.ndarray         # teacher's keyframe answer == teacher's answer on this frame
    agree0_k: np.ndarray         # table's keyframe answer == teacher's answer on this frame
    key_conf1: np.ndarray        # frame table confidence at the keyframe (decides which of the two applies)
    transition: np.ndarray       # the teacher's answer differs from its answer on the previous labelled frame
    clip: np.ndarray             # clip index of each row (the path/test split is by clip)
    extra: dict = field(default_factory=dict)


def _keyframes(states: list[FrameState], block: int, hamming: int) -> list[int]:
    """The keyframe index of every frame of one clip under the run-time rule (label-free, threshold-free)."""
    keys, key = [], -1
    for i, st in enumerate(states):
        if key < 0 or i - key >= block or _popcount(st.dhash ^ states[key].dhash) > hamming:
            key = i
        keys.append(key)
    return keys


def _under(rows: _Rows, thr, m=slice(None)):
    """(conf, agree), each (2, n), of rows `m` under threshold vector `thr`: tier 0's are the ones with the keyframe's
    answer served by the frame table where the table cleared thr[1] at the keyframe, else with the teacher's."""
    by_table = np.nan_to_num(rows.key_conf1[m], nan=-np.inf) >= thr[1]
    c0 = np.where(by_table, rows.conf0_k[m], rows.conf0_t[m])
    a0 = np.where(by_table, rows.agree0_k[m], rows.agree0_t[m])
    return np.stack([c0, rows.conf1[m]]), np.stack([a0, rows.agree1[m]])


def _thresholds_at(conf, agree, s):
    """CascadeController._thresholds_at, vectorised: per tier, the most lenient threshold at which the tier's own
    disagreement on the rows it would answer stays <= s (never set on fewer than 20 rows)."""
    thr = np.full(conf.shape[0], np.inf)
    for k in range(conf.shape[0]):
        c = conf[k]
        m = np.isfinite(c)
        n_min = max(20, int(0.02 * m.sum()))
        if m.sum() < n_min:
            continue
        order = np.argsort(-c[m])
        cs, bad = c[m][order], np.cumsum(~agree[k][m][order])
        j = np.arange(n_min, len(cs) + 1)
        ok = np.flatnonzero(bad[j - 1] <= s * j)
        if len(ok):
            thr[k] = cs[j[ok[-1]] - 1]
    return thr


PROCEDURES = ("auto", "single", "cascade")


def certify_rows(rows: _Rows, alpha: float, delta: float, levels: int = 40, path_fraction: float = 0.3,
                 seed: int = 0, procedure: str = "auto") -> tuple[CascadeController, dict]:
    """The stream's certificate from calibration rows. Two pre-registered candidates are tested at delta / 2 each
    and the one serving the larger share is kept; by the union bound the result is valid at delta (the device of
    SelectiveRiskController's "auto"):

      single tier  "reuse-only" (the frame table switched off, so every keyframe carries the teacher's answer; or
                   "table-only" when the bundle has no reuse tier): one confidence, so SelectiveRiskController's
                   Learn-then-Test on ALL rows, no split. With one row per block, rows are scarce (a few hundred),
                   and a split would cost a third of them;
      cascade      CascadeController's Learn-then-Test with its path step on whole clips (adjacent rows of one clip
                   are not independent) and its test step re-deriving tier 0's confidence and agreement for every
                   candidate, since at run time a reused answer is the frame table's when the table cleared the
                   candidate threshold at the keyframe and the teacher's otherwise.

    procedure="single" or "cascade" tests only that candidate, at the full delta; it must be chosen BEFORE looking
    at the calibration rows (for example because the frame table alone is known to certify nothing on this camera).

    The single-tier candidate also protects reuse from a frame table that is confident and wrong on new clips, which
    would otherwise poison the keyframes and sink a reuse tier that is fine on its own."""
    from .reliability import SelectiveRiskController
    ctl = CascadeController(alpha, delta, path_fraction, levels, seed)
    n = len(rows.conf1)
    has0 = bool(np.isfinite(rows.conf0_t).any())
    if procedure not in PROCEDURES:
        raise ValueError(f"procedure must be one of {PROCEDURES}")
    variants = ["reuse-only", "cascade"] if has0 else ["table-only"]
    if has0 and procedure == "single":
        variants = ["reuse-only"]
    elif has0 and procedure == "cascade":
        variants = ["cascade"]
    d_v = delta / len(variants)
    rng = np.random.default_rng(seed)
    clips = np.unique(rows.clip)
    path_clips = set(clips[rng.random(len(clips)) < path_fraction].tolist()) if len(clips) > 1 else set()
    if len(clips) > 1 and not path_clips:
        path_clips = {int(clips[rng.integers(len(clips))])}
    path = np.isin(rows.clip, list(path_clips))
    test = ~path
    n_test = int(test.sum())
    info = {"n_calibration": n, "n_path": int(path.sum()), "n_test": n_test, "n_path_clips": len(path_clips),
            "n_test_clips": int(len(clips) - len(path_clips)), "variants": variants}
    cands = {}
    # single tier, all rows
    tier = 0 if has0 else 1
    c1 = rows.conf0_t if has0 else rows.conf1
    a1 = rows.agree0_t if has0 else rows.agree1
    m = np.isfinite(c1)
    if m.sum() and variants[0] != "cascade":
        t = float(SelectiveRiskController(alpha, d_v).fit(c1[m], a1[m]).threshold)
        if np.isfinite(t):
            thr = np.full(2, np.inf)
            thr[tier] = t
            cands[variants[0]] = (thr, np.ones(n, bool))
    # cascade, path / test split by clip
    if has0 and path.any():
        conf_t = np.stack([rows.conf0_t, rows.conf1])
        agree_t = np.stack([rows.agree0_t, rows.agree1])
        start = min(_tolerant_start(alpha, d_v, max(n_test, 1)), max(n_test, 1))
        seq = []
        for s in np.linspace(0.0, 2 * alpha, levels):  # strict -> lenient; the path rows only pick the sequence
            thr = _thresholds_at(conf_t[:, path], agree_t[:, path], s)
            path_served = int((ctl._route(conf_t[:, path], thr) >= 0).sum())
            if seq or path_served * n_test / max(int(path.sum()), 1) >= start:
                seq.append(thr)
        last = None
        for thr in seq:
            c, a = _under(rows, thr, test)
            r = ctl._route(c, thr)
            served = r >= 0
            k = int(served.sum())
            wrong = int((~a[r[served], np.flatnonzero(served)]).sum()) if k else 0
            if k == 0 or binom_ucb(wrong, k, d_v) > alpha:
                break
            last = thr
        if last is not None:
            cands["cascade"] = (last, test)
    best = None
    for v, (thr, mask) in cands.items():
        c, a = _under(rows, thr, mask)
        r = ctl._route(c, thr)
        share = float((r >= 0).mean())
        if best is None or share > best[0]:
            best = (share, v, thr, mask, r, a)
    info["variant"] = None if best is None else best[1]
    if best is None:
        ctl.thresholds, ctl.share, ctl.risk = np.array([np.inf, np.inf]), np.zeros(2), None
        info.update(served_share=0.0, transition_disagreement=None)
        return ctl, info
    _, v, thr, mask, r, a = best
    served = r >= 0
    ctl.thresholds = thr
    ctl.share = np.array([(r == j).mean() for j in range(2)])
    ctl.risk = float((~a[r[served], np.flatnonzero(served)]).mean()) if served.any() else None
    tr = rows.transition[mask]
    ts = served & tr
    info.update(served_share=float(served.mean()), evaluated_on="all rows" if mask.all() else "test clips",
                transition_rows=int(tr.sum()), transition_served=int(ts.sum()),
                transition_disagreement=(float((~a[r[ts], np.flatnonzero(ts)]).mean()) if ts.any() else None),
                transition_served_share=float(ts.sum() / tr.sum()) if tr.any() else None,
                transition_reuse_share=float(((r == 0) & tr).sum() / tr.sum()) if tr.any() else None)
    return ctl, info
    thr, r, wrong, k, a = best
    ctl.thresholds = thr
    ctl.share = np.array([(r == j).mean() for j in range(2)])
    ctl.risk = wrong / k
    tr = rows.transition[test]
    served = r >= 0
    ts = served & tr
    info.update(served_share=float(served.mean()), transition_rows=int(tr.sum()), transition_served=int(ts.sum()),
                transition_disagreement=(float((~a[r[ts], np.flatnonzero(ts)]).mean()) if ts.any() else None),
                transition_served_share=float(ts.sum() / tr.sum()) if tr.any() else None,
                transition_reuse_share=float(((r == 0) & tr).sum() / tr.sum()) if tr.any() else None)
    return ctl, info


def _student(student: str) -> dict:
    """compile_schema's `student` argument where this shad0w has one (older distill.py versions do not)."""
    import inspect

    from .compiler.distill import compile_schema
    return {"student": student} if "student" in inspect.signature(compile_schema).parameters else {}


def stream_compile(frames_by_clip, answers_by_clip, options, spec: ImageSpec | None = None, alpha: float = 0.05,
                   delta: float = 0.1, block: int = 15, hamming: int = 12, cal_clips=None, cal_fraction: float = 0.4,
                   name: str = "frame", teacher: str = "unspecified", seed: int = 0, reuse: bool = True,
                   qtype: str = "choice", levels: int = 40, drift_window: int = 500, drift_margin: float = 0.03,
                   C: float | None = None, student: str = "linear", procedure: str = "auto"):
    """Compile a stream bundle from teacher-labelled clips and certify it. Returns (Model, certificate dict) in the
    shape of shadow.cascade_compile.

    frames_by_clip[c] is a sequence of frames (uint8 arrays or anything vision.pixels accepts) at the rate the stream
    will run at; answers_by_clip[c][i] is the teacher's answer on frame i, or None where it gave none (a refusal, a
    content filter): such frames still shape the keyframes but never fit or calibrate. An answer outside `options`
    counts as None. Clips are split into fitting and calibration clips (`cal_clips`: indices; else a `cal_fraction`
    share at random, at least 2); the two tables are fitted on the fitting clips only and the certificate uses one
    row per block of `block` frames of the calibration clips. reuse=False compiles the frame table alone (the same
    certificate without tier 0), which is the baseline the reuse tier must beat. C: the tables' regularisation
    (None: chosen by held-out search, which is most of the compile time). student: the frame table's student
    ("linear", the int8 table the C core serves; "deep" / "auto" where this shad0w's distill offers them); the
    reuse table, 15 items a frame, is always linear. procedure: see certify_rows ("auto" tests the reuse-only and
    the cascade candidates at delta / 2 each)."""
    from .compiler.distill import compile_schema
    spec = spec or ImageSpec()
    if qtype == "yesno":
        options = ["no", "yes"]
    options = [str(o) for o in options]
    idx = {o: i for i, o in enumerate(options)}
    n_clips = len(frames_by_clip)
    if n_clips != len(answers_by_clip) or n_clips < 2:
        raise ValueError("need at least 2 clips with one answer list per clip")
    rng = np.random.default_rng(seed)
    if n_clips < 3:
        raise ValueError("need at least 3 clips: one to fit on and two to calibrate on")
    if cal_clips is None:
        perm = rng.permutation(n_clips)
        n_cal = int(min(max(2, round(cal_fraction * n_clips)), n_clips - 1))
        cal_set = set(perm[:n_cal].tolist())
    else:
        cal_set = set(int(c) for c in cal_clips)
        if len(cal_set) < 2 or len(cal_set) >= n_clips:
            raise ValueError("cal_clips must name at least 2 clips and leave at least one clip to fit on")
    # per clip: states, answer ids (-1 = unknown), keyframes under the run-time rule
    clips = []
    n_unknown = 0
    for c, (frames, answers) in enumerate(zip(frames_by_clip, answers_by_clip)):
        states = [frame_state(f, spec) for f in frames]
        if len(answers) != len(states):
            raise ValueError(f"clip {c}: {len(states)} frames but {len(answers)} answers")
        y = np.full(len(states), -1, dtype=int)
        for i, a in enumerate(answers):
            if qtype == "yesno" and isinstance(a, bool):
                a = "yes" if a else "no"
            j = idx.get(str(a)) if a is not None else None
            if j is None:
                n_unknown += a is not None
            else:
                y[i] = j
        clips.append((states, y, _keyframes(states, block, hamming)))
    fit_ix = [c for c in range(n_clips) if c not in cal_set]
    cal_ix = sorted(cal_set)
    # tier 1: the frame table on every labelled fitting frame
    fit_items = [st.items for c in fit_ix for st, yy in zip(clips[c][0], clips[c][1]) if yy >= 0]
    fit_y = np.array([yy for c in fit_ix for yy in clips[c][1] if yy >= 0])
    if len(fit_y) < 50:
        raise ValueError(f"{len(fit_y)} labelled fitting frames; need at least 50")
    if len(np.unique(fit_y)) < 2:
        raise ValueError("the teacher gave a single answer on every fitting frame: nothing to compile")
    comp1 = compile_schema(fit_items, fit_y, options, C=C, **_student(student))
    q1 = Question(name, qtype, options, comp1.rx, float("inf"), alpha, DriftGuard(np.ones(2), np.ones(2, bool)),
                  0.0, True)

    def conf_pred(st):
        p = softmax(q1.rx.logits(st.items))
        return float(p.max()), int(p.argmax())

    # tier 0: the reuse table on (keyframe, frame) pairs of the fitting clips, labelled "same teacher answer?"
    q0 = None
    if reuse:
        d_items, d_y = [], []
        for c in fit_ix:
            states, y, keys = clips[c]
            for i, (st, yy, k) in enumerate(zip(states, y, keys)):
                if k != i and yy >= 0 and y[k] >= 0:
                    d_items.append(delta_items(st, states[k], int(y[k]), i - k))
                    d_y.append(int(yy == y[k]))
        d_y = np.array(d_y, dtype=int)
        if len(d_y) >= 50 and 0 < d_y.sum() < len(d_y):
            comp0 = compile_schema(d_items, d_y, ["no", "yes"], C=C, **_student("linear"))
            q0 = Question(name + REUSE_SUFFIX, "yesno", ["no", "yes"], comp0.rx, float("inf"), alpha,
                          DriftGuard(np.ones(2), np.ones(2, bool)), 0.0, True)
        else:
            _log.warning("stream_compile: %d reuse rows (%d 'same'): the reuse tier is not compiled",
                         len(d_y), int(d_y.sum()))
    # calibration rows: one labelled frame per block of each calibration clip
    R = {k: [] for k in ("conf1", "agree1", "conf0_t", "conf0_k", "agree0_t", "agree0_k", "key_conf1", "transition",
                         "clip")}
    n_blocks = n_labelled_cal = 0
    flick_n = flick_d = 0
    for c in cal_ix:
        states, y, keys = clips[c]
        cp = {}
        prev = -1
        trans = np.zeros(len(y), bool)
        for i in range(len(y)):
            if y[i] >= 0:
                trans[i] = prev >= 0 and y[i] != prev
                if prev >= 0 and i > 0 and y[i - 1] >= 0 and _popcount(states[i].dhash ^ states[i - 1].dhash) <= 2:
                    flick_n += 1
                    flick_d += y[i] != y[i - 1]
                prev = y[i]
        n_labelled_cal += int((y >= 0).sum())
        for b0 in range(0, len(y), block):
            lab = [i for i in range(b0, min(b0 + block, len(y))) if y[i] >= 0]
            n_blocks += 1
            if not lab:
                continue
            i = int(rng.choice(lab))
            if i not in cp:
                cp[i] = conf_pred(states[i])
            c1, p1 = cp[i]
            k = keys[i]
            row = {"conf1": c1, "agree1": p1 == y[i], "transition": trans[i], "clip": c, "conf0_t": np.nan,
                   "conf0_k": np.nan, "agree0_t": False, "agree0_k": False, "key_conf1": np.nan}
            if q0 is not None and k != i and y[k] >= 0:
                if k not in cp:
                    cp[k] = conf_pred(states[k])
                kc, kp = cp[k]
                row["key_conf1"] = kc
                row["conf0_t"] = softmax(q0.rx.logits(delta_items(states[i], states[k], int(y[k]), i - k)))[1]
                row["agree0_t"] = y[k] == y[i]
                row["conf0_k"] = (row["conf0_t"] if kp == y[k]
                                  else softmax(q0.rx.logits(delta_items(states[i], states[k], kp, i - k)))[1])
                row["agree0_k"] = kp == y[i]
            for kk, v in row.items():
                R[kk].append(v)
    rows = _Rows(**{k: np.asarray(v, dtype=float if k.startswith("conf") or k == "key_conf1" else
                                  (int if k == "clip" else bool)) for k, v in R.items()})
    if len(rows.conf1) < 20:
        raise ValueError(f"{len(rows.conf1)} calibration rows (one per block of {block} frames); "
                         "need more clips or a smaller block")
    ctl, info = certify_rows(rows, alpha, delta, levels, seed=seed, procedure=procedure)
    q1.threshold = float(ctl.thresholds[1])
    m = np.isfinite(rows.conf1)
    q1.guard = DriftGuard(rows.conf1[m], rows.agree1[m], drift_window, drift_margin)
    questions = {name: q1}
    if q0 is not None:
        q0.threshold = float(ctl.thresholds[0])
        m0 = np.isfinite(rows.conf0_t)
        if m0.sum() >= 2:
            q0.guard = DriftGuard(rows.conf0_t[m0], rows.agree0_t[m0], drift_window, drift_margin)
        questions[q0.name] = q0
    model = Model(meta={"mode": "stream", "teacher": teacher, "alpha": alpha, "delta": delta, "image": spec.to_dict(),
                        "stream": {"question": name, "block": block, "hamming": hamming,
                                   "reuse": q0.name if q0 is not None else None, "options": options}},
                  questions=questions)
    shares = ctl.share if ctl.share is not None else np.zeros(2)
    tiers = [{"question": q.name, "tier": t, "source": src, "threshold": None if not np.isfinite(th) else float(th),
              "share": float(sh)}
             for q, t, src, th, sh in ((q0, 0, "reuse", ctl.thresholds[0], shares[0]),
                                       (q1, 1, "table", ctl.thresholds[1], shares[1]))
             if q is not None]
    served = float(shares.sum())
    sha = hashlib.sha256()
    for c in range(n_clips):
        sha.update(json.dumps(clips[c][1].tolist()).encode())
    entry = {"type": qtype, "options": options, "n_clips": n_clips, "n_fit_clips": len(fit_ix),
             "n_calibration_clips": len(cal_ix), "n_fit": int(len(fit_y)),
             "n_fit_reuse_rows": int(len(d_y)) if reuse else 0, "n_unknown_answers": int(n_unknown),
             "n_labelled_calibration_frames": n_labelled_cal, "n_blocks": n_blocks, "block": block, "hamming": hamming,
             "data_sha256": sha.hexdigest(), "calibration": "held-out-clips/one-row-per-block",
             "threshold": tiers[-1]["threshold"], "certified_share_on_calibration": served,
             "disagreement_on_certified_calibration": ctl.risk, "teacher_share_on_calibration": 1.0 - served,
             "transition_disagreement": {k: info.get(k) for k in (
                 "transition_rows", "transition_served", "transition_disagreement", "transition_served_share",
                 "transition_reuse_share")},
             "teacher_flicker_on_near_identical_frames": (flick_d / flick_n) if flick_n else None,
             "n_near_identical_pairs": flick_n,
             "procedure": ("learn-then-test/clopper-pearson/stream-cascade-fixed-sequence (path split by clip; "
                           "tier-0 agreement re-derived per candidate threshold from the keyframe's served answer) "
                           f"+ single-tier learn-then-test on all rows; procedure={procedure} over "
                           f"{info['variants']}, kept: {info.get('variant')}"),
             **{k: info[k] for k in ("n_calibration", "n_path", "n_test", "n_path_clips", "n_test_clips")},
             "tiers": tiers}
    cert = {"format": "shad0w-certificate/1", "mode": "stream", "certified_against": "teacher", "teacher": teacher,
            "alpha": alpha, "delta": delta, "block": block,
            "statement": ("On clips like the calibration clips, played at the same frame rate and decided with this "
                          "bundle's keyframe rule, among frames this stream answers itself with certified=true (a "
                          "reused answer or the frame table's), the rate of disagreement with the teacher is at most "
                          "alpha with probability at least 1 - delta, counting one frame per block of `block` frames "
                          "(adjacent frames are not independent). The bound is against the teacher, not the truth; "
                          "it is marginal over frames, so transition frames, where the teacher's answer changes, are "
                          "reported separately and may fare worse. A different scene, frame rate or camera needs "
                          "re-certification."),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "image": spec.to_dict(), "questions": {name: entry}}
    return model, cert


Stream.compile = staticmethod(stream_compile)
