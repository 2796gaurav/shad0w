"""A microsecond guardrail in front of an LLM: block the obvious, never wave anything through on the table's word.

    gate = shad0w.guard.Guard(shad0w.decision("injection", options={"attack": "...", "safe": "..."}, llm=...))
    v = gate.check(user_text)          # Verdict(action="block" | "ask_model", ...)
    if v.action == "block": refuse()
    else: v = gate.confirm(user_text)  # your LLM decides (and is logged, so the table keeps learning)

Design (asymmetric, from the guardrail literature: adaptive attackers break every cheap detector, 2510.09023;
character smuggling bypasses commercial guards 60-100% of the time, 2504.11168):

  * normalise first: Unicode NFKC, case folding, zero-width / bidi / tag-block characters removed, homoglyphs and
    leetspeak folded, s p a c e d letters joined, base64 / hex runs decoded and appended. The table sees the
    normalised text.
  * anything the normaliser had to undo is an ANOMALY: hiding a payload is itself the signal, so an anomalous input
    is never decided by the table alone. It goes to the model (or is blocked outright with block_anomalies=True).
  * the table may BLOCK, under its own BLOCK-ONLY certificate (Guard.calibrate): among inputs the table blocks,
    disagreement with your LLM <= alpha with probability >= 1 - delta, on traffic like the calibration traffic. It is
    fitted only on calibration rows the table would block, so it is valid for exactly the decision the Guard makes
    (the bundle's ordinary certificate covers both answers together, which a block-only gate does not need).
    It may never ALLOW: a "safe" answer from the table is a hint, and the input still goes to
    your LLM, unless you opt in with allow_safe=True for low-stakes routing (then the exposed-mode robustness radius
    must also clear r_min, so a few inserted words cannot flip it; only a linear table has that radius, so compile with
    student="linear" for allow_safe: a deep table never allows).
  * every block is cheap and logged; a sample of them is re-checked by the LLM (the Shadow's spot checks), and the
    live breach monitor stops the table if blocks start disagreeing with your LLM.
"""
from __future__ import annotations

import base64
import binascii
import re
import time
import unicodedata
from dataclasses import dataclass, field

# characters that render as nothing or reorder text: stripped, and their presence is an anomaly
_INVISIBLE = re.compile("[­͏؜ᅟᅠ឴឵᠎​-‏‪-‮⁠-⁯"
                        "ㅤ︀-️﻿ﾠ\U000e0000-\U000e007f\U000e0100-\U000e01ef]")
# a small confusables map (Cyrillic / Greek letters that look Latin), folded to ASCII
_HOMOGLYPHS = str.maketrans({
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c", "т": "t", "у": "y",
    "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ɡ": "g", "ո": "n", "ս": "u", "ᴠ": "v", "ԝ": "w",
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "ı": "i", "ℓ": "l", "∅": "o"})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_WORD = re.compile(r"(?<![\w@$])[a-z0-9@$]{3,}(?![\w@$])")
_LEET_INSIDE = re.compile(r"[a-z][0134579@$]+[a-z]")  # a substitution between letters: n0r, pr3v, h4ck
_SKIP = re.compile(r"\S+@\S+\.\S+|https?://\S+|www\.\S+|[A-Za-z0-9+/]{24,}={0,2}")  # emails, URLs, encoded runs
_SPACED = re.compile(r"\b(?:[a-z]\s){3,}[a-z]\b")
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}){12,}\b")


def normalize(text: str) -> tuple[str, dict]:
    """(normalised text, anomalies). anomalies counts what had to be undone; empty means the text was plain."""
    an: dict[str, int] = {}
    t = str(text)
    n = len(_INVISIBLE.findall(t))
    if n:
        an["invisible"] = n
        t = _INVISIBLE.sub("", t)
    k = unicodedata.normalize("NFKC", t)
    if k != t:
        an["compatibility_forms"] = sum(a != b for a, b in zip(k, t)) + abs(len(k) - len(t))
    t = k.casefold()
    h = t.translate(_HOMOGLYPHS)
    if h != t:
        an["homoglyphs"] = sum(a != b for a, b in zip(h, t))
    t = h
    if any(ord(c) > 127 and unicodedata.category(c).startswith("L") for c in t):
        scripts = {unicodedata.name(c, "?").split(" ")[0] for c in t if c.isalpha()}
        if len(scripts) > 1:
            an["mixed_scripts"] = len(scripts)
    plain = _SKIP.sub(" ", t)
    leet = {w for w in _LEET_WORD.findall(plain) if _LEET_INSIDE.search(w)}
    subs = sum(sum(ch in "0134579@$" for ch in w) for w in leet)
    if leet and (subs >= 2):
        an["leetspeak"] = len(leet)
        t = _LEET_WORD.sub(lambda m: m.group(0).translate(_LEET) if m.group(0) in leet else m.group(0), t)
    sp = _SPACED.findall(t)
    if sp:
        an["spaced_letters"] = len(sp)
        t = _SPACED.sub(lambda m: m.group(0).replace(" ", ""), t)
    decoded = []
    for m in _B64.findall(text):
        try:
            d = base64.b64decode(m + "=" * (-len(m) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if sum(c.isprintable() for c in d) > 0.9 * len(d) and len(d) >= 8:
            decoded.append(d)
    for m in _HEX.findall(text):
        try:
            d = bytes.fromhex(m).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if sum(c.isprintable() for c in d) > 0.9 * len(d):
            decoded.append(d)
    if decoded:
        an["encoded_payload"] = len(decoded)
        t = t + " " + " ".join(normalize(d)[0] for d in decoded)  # scan what the encoding hid, too
    return t, an


@dataclass
class Verdict:
    action: str                 # "block" (the table is certain it is an attack) | "allow" | "ask_model"
    source: str                 # "table" | "model" | "normaliser"
    answer: object = None       # the table's or the model's answer ("attack" / "safe" / ...)
    confidence: float | None = None
    anomalies: dict = field(default_factory=dict)
    latency_us: float = 0.0
    reason: str = ""
    tokens_seen: int | None = None   # StreamGuard: how many whitespace tokens of the reply had arrived at this verdict
    total_tokens: int | None = None  # StreamGuard: the reply's length when known (check(), finish())


class Guard:
    """Asymmetric guardrail around a Shadow whose question has an attack label (default "attack")."""

    def __init__(self, shadow, attack: str = "attack", safe: str = "safe", allow_safe: bool = False,
                 block_anomalies: bool = False, r_min: float = 3.0, max_chars: int = 8000):
        self.sh, self.attack, self.safe = shadow, attack, safe
        self.allow_safe, self.block_anomalies, self.r_min, self.max_chars = allow_safe, block_anomalies, r_min, max_chars

    @property
    def calibrated(self) -> bool:
        m = self.sh.model
        return m is not None and self.sh.question in (m.meta.get("guard") or {})

    @property
    def block_threshold(self) -> float | None:
        """The block-only certified confidence; inf when calibration certified nothing (then the table blocks
        nothing); None when not calibrated (then the bundle's ordinary certificate decides blocks, which bounds
        disagreement over BOTH answers together and can be looser for blocks alone: call calibrate())."""
        if not self.calibrated:
            return None
        t = self.sh.model.meta["guard"][self.sh.question].get("threshold")
        return float("inf") if t is None else float(t)

    def calibrate(self, texts, answers, alpha: float | None = None, delta: float = 0.1, save: bool = True) -> dict:
        """Block-only certificate from FRESH teacher-labelled inputs (never trained on): the lowest confidence t such
        that, among inputs the table would block (answer = attack, confidence >= t, no anomaly), disagreement with the
        teacher is <= alpha with probability >= 1 - delta (Learn-then-Test, Clopper-Pearson). Saved in the bundle."""
        import numpy as np

        from .reliability import SelectiveRiskController
        q = self.sh.model.questions[self.sh.question]
        alpha = q.alpha if alpha is None else alpha
        conf, ok = [], []
        for t, a in zip(texts, answers):
            norm, an = normalize(t)
            if an:
                continue  # anomalous inputs are never blocked by the table, so they are not part of this decision
            r = q.decide(norm, observe=False)
            if r.get("choice") == self.attack:
                conf.append(r["confidence"])
                ok.append(str(a) == self.attack)
        ctl = SelectiveRiskController(alpha, delta).fit(np.array(conf), np.array(ok, bool)) if conf else None
        thr = None if ctl is None or not np.isfinite(ctl.threshold) else float(ctl.threshold)
        conf, ok = np.array(conf), np.array(ok, bool)
        served = conf >= thr if thr is not None else np.zeros(len(conf), bool)
        rep = {"threshold": thr, "alpha": alpha, "delta": delta, "n_calibration": len(texts), "n_would_block": int(len(conf)),
               "blocked_on_calibration": int(served.sum()),
               "disagreement_on_blocked": float((~ok[served]).mean()) if served.any() else None}
        self.sh.model.meta.setdefault("guard", {})[self.sh.question] = rep
        if save and self.sh.bundle_path:
            self.sh.model.save(self.sh.bundle_path)
        return rep

    def check(self, text: str) -> Verdict:
        """Microseconds: block, allow (only with allow_safe), or ask_model. Never calls the LLM."""
        t0 = time.perf_counter()
        norm, an = normalize(text)
        if len(text) > self.max_chars:
            an = {**an, "too_long": len(text)}
        if an:
            act = "block" if self.block_anomalies else "ask_model"
            return Verdict(act, "normaliser", None, None, an, (time.perf_counter() - t0) * 1e6,
                           "the input hides or reshapes text (" + ", ".join(an) + "): " +
                           ("blocked" if act == "block" else "the table does not decide it"))
        thr = self.block_threshold
        if thr is not None:  # the block-only certificate
            q = self.sh.model.questions[self.sh.question]
            r = q.decide(norm, observe=True, probabilities=False)
            ans, conf = r.get("choice", r.get("answer")), r["confidence"]
            us = (time.perf_counter() - t0) * 1e6
            if str(ans) == self.attack and conf >= thr and r["flag"] not in ("drift", "options_changed"):
                return Verdict("block", "table", ans, conf, an, us, f"certified block ({conf:.3f} >= {thr:.3f})")
            if not (self.allow_safe and str(ans) == self.safe and r["certified"]):
                return Verdict("ask_model", "table", ans, conf, an, us, "the table never allows on its own")
            d = self.sh.peek(norm, probabilities=False)
        else:
            d = self.sh.peek(norm, probabilities=False)
        us = (time.perf_counter() - t0) * 1e6
        if d is None:
            return Verdict("ask_model", "table", None, None, an, us, "the table is not certified on this input")
        if str(d.answer) == self.attack and thr is None:
            return Verdict("block", "table", d.answer, d.confidence, an, us, "certified attack (bundle certificate)")
        if self.allow_safe and str(d.answer) == self.safe:
            q = self.sh.model.questions[self.sh.question]
            r = q.decide(norm, exposed=True, observe=False)["radius"]  # inserted words needed to flip it (None = inf)
            # a deep table has no exact radius (reported None as well): it may never allow (compile with student="linear")
            deep = getattr(q.rx, "W2", None) is not None
            r = 0.0 if deep else (float("inf") if r is None else r)
            if r >= self.r_min:
                return Verdict("allow", "table", d.answer, d.confidence, an, us, f"certified safe, radius {r}")
        return Verdict("ask_model", "table", d.answer, d.confidence, an, us, "the table never allows on its own")

    def confirm(self, text: str) -> Verdict:
        """The LLM decides (logged by the Shadow, so the table keeps learning). Use it after check() says ask_model."""
        t0 = time.perf_counter()
        norm, an = normalize(text)
        teacher = self.sh.teacher
        # the LLM judges the ORIGINAL input (obfuscation and all); the log keeps the normalised text the table reads
        ans, _ = self.sh._ask(lambda _norm: teacher(text), norm, self.sh._src(), self.sh.question or "decision")
        act = "block" if str(ans) == self.attack else "allow"
        return Verdict(act, "model", ans, None, an, (time.perf_counter() - t0) * 1e6, "your model decided")

    def __call__(self, text: str) -> Verdict:
        v = self.check(text)
        return v if v.action != "ask_model" else self.confirm(text)


# ---------- streaming output guard: certified prefix blocks ----------

DEFAULT_GRID = (16, 32, 64, 128, 256)


def prefix_grid(n_tokens: int, grid=DEFAULT_GRID, step: int = 256) -> list[int]:
    """The token counts at which a reply of n_tokens whitespace tokens is checked: the grid points below n_tokens,
    then every `step` tokens after the last one, then the full reply."""
    pts = [g for g in grid if g < n_tokens]
    g = (grid[-1] if grid else 0) + step
    while g < n_tokens:
        pts.append(g)
        g += step
    pts.append(max(int(n_tokens), 0))
    return pts


class StreamGuard(Guard):
    """Check an LLM reply WHILE it streams, blocking on a certified prefix.

        sg = StreamGuard(shad0w.decision("reply_safety", options={"unsafe": "...", "safe": "..."}, llm=...))
        sg.calibrate(full_replies, teacher_labels)       # fresh replies, teacher labels on the FULL reply
        s = sg.session()
        for chunk in llm_stream:
            v = s.feed(chunk)                            # Verdict("block") at the first grid point that blocks, else None
            if v: cut the stream; break
            forward(chunk)
        else:
            v = s.finish()                               # the full reply, checked once more

    The reply is checked at a fixed grid of whitespace-token counts (16, 32, 64, 128, 256, then every 256) and once
    complete. Each check is one table decision (microseconds). The guard only ever BLOCKS; a prefix the table calls
    safe says nothing, the stream simply continues, and the final full reply is still yours to confirm.

    Certificate (StreamGuard.calibrate). The deployed rule is one decision per REPLY: block iff at some grid point
    the table answers `unsafe` with confidence >= t (checked in order, so the first such point cuts the stream).
    Calibration therefore scores each fresh teacher-labelled reply by the maximum unsafe-confidence over ITS grid
    (the same prefixes the live guard would see) and runs the block-only Learn-then-Test on those reply-level
    scores: among replies the guard blocks at any prefix, the teacher called the full reply safe at most alpha of
    the time, with probability >= 1 - delta, on traffic like the calibration traffic. No union bound over the grid
    is needed because the unit of exchangeability is the reply and the rule is a single threshold on a single
    per-reply score; pooling prefix rows (several correlated rows per reply, the stream stopping at the first hit)
    or a per-grid-point delta/len(grid) split would certify a different event than the one the guard produces.

    tokens = whitespace-separated words (model tokens differ by a factor ~1.3; the grid is a convention, and the
    certificate holds for whatever grid was calibrated). `context` (the user prompt) can be prepended to what the
    table reads; calibrate with the same `contexts` then."""

    def __init__(self, shadow, unsafe: str = "unsafe", safe: str = "safe", grid=DEFAULT_GRID, step: int = 256,
                 block_anomalies: bool = False, max_chars: int = 32000):
        super().__init__(shadow, attack=unsafe, safe=safe, allow_safe=False, block_anomalies=block_anomalies,
                         max_chars=max_chars)
        self.grid, self.step = tuple(int(g) for g in grid), int(step)

    unsafe = property(lambda self: self.attack)

    # -- the certificate ----------------------------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        m = self.sh.model
        return m is not None and self.sh.question in (m.meta.get("guard_stream") or {})

    @property
    def block_threshold(self) -> float | None:
        """Certified reply-level threshold; inf when calibration certified nothing; None when never calibrated
        (an uncalibrated StreamGuard blocks nothing: the bundle's certificate was never tested on prefixes)."""
        if not self.calibrated:
            return None
        t = self.sh.model.meta["guard_stream"][self.sh.question].get("threshold")
        return float("inf") if t is None else float(t)

    def _score(self, text: str, context: str | None) -> tuple[float, object, dict]:
        """(block score, table answer, anomalies) for one prefix: the unsafe-confidence when the table answers unsafe
        on a plain prefix, else 0 (never blockable)."""
        q = self.sh.model.questions[self.sh.question]
        norm, an = normalize(text if context is None else context + "\n" + text)
        if len(text) > self.max_chars:
            an = {**an, "too_long": len(text)}
        if an:
            return 0.0, None, an
        r = q.decide(norm, observe=False, probabilities=False)
        ans = r.get("choice", r.get("answer"))
        if str(ans) == self.attack and r["flag"] not in ("drift", "options_changed"):
            return float(r["confidence"]), ans, an
        return 0.0, ans, an

    def reply_score(self, text: str, context: str | None = None) -> tuple[float, int, int]:
        """(max block score over the reply's grid, the grid point reaching it first, n_tokens)."""
        words = str(text).split()
        best, at = 0.0, 0
        for g in prefix_grid(len(words), self.grid, self.step):
            s = self._score(" ".join(words[:g]), context)[0]
            if s > best:
                best, at = s, g
        return best, at, len(words)

    def calibrate(self, texts, answers, alpha: float | None = None, delta: float = 0.1, save: bool = True,
                  contexts=None) -> dict:
        """Reply-level block-only certificate from FRESH teacher-labelled full replies (see the class doc)."""
        import numpy as np

        from .reliability import SelectiveRiskController
        q = self.sh.model.questions[self.sh.question]
        alpha = q.alpha if alpha is None else alpha
        texts, answers = list(texts), list(answers)
        contexts = list(contexts) if contexts is not None else [None] * len(texts)
        score, ok, frac = [], [], []
        for t, a, c in zip(texts, answers, contexts):
            s, at, n = self.reply_score(t, c)
            if s > 0:  # a reply the table would block at some threshold; the others are never blocked
                score.append(s)
                ok.append(str(a) == self.attack)
                frac.append(at / max(n, 1))
        score, ok, frac = np.array(score, float), np.array(ok, bool), np.array(frac, float)
        ctl = SelectiveRiskController(alpha, delta).fit(score, ok) if len(score) else None
        thr = None if ctl is None or not np.isfinite(ctl.threshold) else float(ctl.threshold)
        served = score >= thr if thr is not None else np.zeros(len(score), bool)
        rep = {"threshold": thr, "alpha": alpha, "delta": delta, "grid": list(self.grid), "step": self.step,
               "unit": "reply (max unsafe-confidence over its prefix grid)", "with_context": contexts[0] is not None,
               "n_calibration": len(texts), "n_would_block": int(len(score)), "blocked_on_calibration": int(served.sum()),
               "disagreement_on_blocked": float((~ok[served]).mean()) if served.any() else None,
               "mean_fraction_seen_on_blocked": float(frac[served].mean()) if served.any() else None}
        self.sh.model.meta.setdefault("guard_stream", {})[self.sh.question] = rep
        if save and self.sh.bundle_path:
            self.sh.model.save(self.sh.bundle_path)
        return rep

    # -- checking -----------------------------------------------------------------------------------------------
    def session(self, context: str | None = None) -> StreamSession:
        return StreamSession(self, context)

    def check(self, text: str, context: str | None = None) -> Verdict:
        """A whole reply, checked as the stream would have been: the first grid point that blocks wins."""
        s = self.session(context)
        v = s.feed(str(text))
        return v if v is not None else s.finish()

    def stream(self, chunks, context: str | None = None):
        """Wrap a chunk iterator: yields (chunk, verdict) with verdict None while the reply passes; a block verdict
        ends the iteration (the chunk it came with is yielded, so you can cut there); the final verdict comes with
        an empty chunk."""
        s = self.session(context)
        for ch in chunks:
            v = s.feed(ch)
            yield ch, v
            if v is not None:
                return
        yield "", s.finish()


class StreamSession:
    """One reply being streamed through a StreamGuard (see StreamGuard)."""

    def __init__(self, guard: StreamGuard, context: str | None = None):
        self.g, self.context = guard, context
        self.text, self.n_tokens = "", 0
        self._next = 0  # index of the next grid point to check (beyond the fixed grid: multiples of step)
        self.checks: list[tuple[int, float, float]] = []  # (tokens, score, us) per prefix check
        self.verdict: Verdict | None = None

    def _grid_point(self, i: int) -> int:
        g = self.g.grid
        return g[i] if i < len(g) else (g[-1] if g else 0) + self.g.step * (i - len(g) + 1)

    def _check_prefix(self, n: int, words=None) -> Verdict | None:
        t0 = time.perf_counter()
        thr = self.g.block_threshold
        if thr is None or self.g.sh.model is None:
            return None
        words = self.text.split() if words is None else words
        text = " ".join(words[:n]) if n < len(words) else " ".join(words)
        s, ans, an = self.g._score(text, self.context)
        us = (time.perf_counter() - t0) * 1e6
        self.checks.append((n, s, us))
        if an and self.g.block_anomalies:
            return Verdict("block", "normaliser", None, None, an, us, "the reply hides or reshapes text (" + ", ".join(an) + ")",
                           tokens_seen=n)
        if s > 0 and s >= thr:
            return Verdict("block", "table", ans, s, an, us, f"certified block at {n} tokens ({s:.3f} >= {thr:.3f})",
                           tokens_seen=n)
        return None

    def feed(self, chunk: str) -> Verdict | None:
        """Add a chunk; returns a block Verdict at the first grid point the table blocks, else None."""
        if self.verdict is not None:
            return self.verdict
        if not chunk:
            return None
        cont = bool(self.text) and not self.text[-1].isspace() and not chunk[0].isspace()
        n_new = len(chunk.split())
        if cont and n_new:
            n_new -= 1  # the chunk's first piece continues the last word
        self.text += chunk
        self.n_tokens += n_new
        words = None
        while True:
            g = self._grid_point(self._next)
            # the g-th word must be complete (calibration saw whole words): wait when the chunk ended inside it
            if self.n_tokens < g or (self.n_tokens == g and not self.text[-1].isspace()):
                return None
            if words is None:
                words = self.text.split()
            v = self._check_prefix(g, words)
            self._next += 1
            if v is not None:
                self.verdict = v
                return v

    def finish(self) -> Verdict:
        """The complete reply: a certified block, or ask_model (the guard never allows on the table's word)."""
        if self.verdict is not None:
            return self.verdict
        n = self.n_tokens
        t0 = time.perf_counter()
        v = self._check_prefix(n)
        if v is None:
            thr = self.g.block_threshold
            reason = ("not calibrated: call StreamGuard.calibrate()" if thr is None else
                      "no certified block at any prefix; the table never allows on its own")
            s = self.checks[-1][1] if self.checks else 0.0
            v = Verdict("ask_model", "table", None, s if s > 0 else None, {}, (time.perf_counter() - t0) * 1e6, reason,
                        tokens_seen=n)
        v.total_tokens = n
        v.latency_us = float(sum(us for _, _, us in self.checks))
        self.verdict = v
        return v

    @property
    def blocked(self) -> bool:
        return self.verdict is not None and self.verdict.action == "block"
