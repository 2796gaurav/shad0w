"""Put shad0w in front of the model you already call, in one line.

    import shad0w

    def ask_llm(text):                       # your existing model call: returns one of the schema's options
        ...

    sh = shad0w.Shadow("bundle/", teacher=ask_llm, log="teacher_log.jsonl")
    d = sh.decide("my card was stolen yesterday")
    d.answer, d.source                       # "lost_card", "table"  (or "teacher" when the table defers)

Day one, before any bundle exists, run `shad0w.Shadow(None, teacher=ask_llm, log=...)`: every call goes to
your model and is logged. Once enough traffic is logged, run `shad0w shadow` to compile and certify a bundle.
Then point `Shadow` at the bundle: certified answers come from the table and the rest still go to your
model, logged, so the next compile has more data.

A small random share of certified answers (`audit_rate`, default 1%) is also sent to your model. These
spot checks are logged with `"source": "audit"` and summarised by `stats()`, so you can watch live
disagreement with your model, not only the calibration-time certificate.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .api import Model, load

__all__ = ["Shadow", "Decision", "cascade"]


@dataclass(frozen=True)
class Decision:
    answer: Any  # the option name (choice) or a bool (yesno)
    source: str  # "table" (certified, served locally) or "teacher" (deferred to your model)
    confidence: float | None  # the table's confidence; None when no table was consulted
    certified: bool
    flag: str | None  # why the table deferred: low_confidence, low_radius, drift, uncalibrated, no_bundle
    latency_us: float  # end-to-end time of this call

    def __str__(self) -> str:
        return str(self.answer)


class Shadow:
    """A certified cascade around one question: the table when it is certified, your model otherwise.

    bundle      path to a compiled bundle, a loaded `Model`, or None (log-only: every call goes to the teacher)
    teacher     callable(text) -> answer, your existing model call
    question    which question of the bundle to answer (default: the bundle's only question, or "decision")
    log         JSON Lines file that receives every teacher answer in the format `shad0w shadow` reads
    audit_rate  share of certified answers also sent to the teacher as spot checks (0 disables)
    exposed     treat inputs as adversarial: also defer answers with a small robustness radius
    """

    def __init__(self, bundle: str | Model | None, teacher: Callable[[str], Any], question: str | None = None,
                 log: str | None = None, audit_rate: float = 0.01, exposed: bool = False, seed: int | None = None):
        if not callable(teacher):
            raise TypeError("teacher must be a callable: text -> answer")
        if not 0.0 <= audit_rate <= 1.0:
            raise ValueError("audit_rate must be in [0, 1]")
        self.teacher, self.log, self.audit_rate, self.exposed = teacher, log, audit_rate, exposed
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._counts = {"table": 0, "teacher": 0, "audits": 0, "audit_disagreements": 0}
        self.model: Model | None = None
        self.question = question
        self.reload(bundle)

    def reload(self, bundle: str | Model | None) -> Shadow:
        """Swap in a new bundle (for example after `shad0w certify`) without restarting."""
        model = load(bundle) if isinstance(bundle, (str, os.PathLike)) else bundle
        if model is not None:
            names = list(model.questions)
            if self.question is None:
                if len(names) != 1:
                    raise ValueError(f"bundle has questions {names}; pass question=...")
                self.question = names[0]
            elif self.question not in model.questions:
                raise ValueError(f"question {self.question!r} is not in the bundle ({names})")
        self.model = model
        return self

    def decide(self, text: str) -> Decision:
        t0 = time.perf_counter()
        name = self.question or "decision"
        if self.model is None:
            answer = self._ask(text, "teacher", name)
            self._count("teacher")
            return Decision(answer, "teacher", None, False, "no_bundle", _us(t0))
        r = self.model.decide(text, exposed=self.exposed, questions={name: {}}, probabilities=False)["answers"][name]
        local = r["choice"] if "choice" in r else r["answer"]
        if r["certified"]:
            if self.audit_rate and self._rng.random() < self.audit_rate:
                truth = self._ask(text, "audit", name)
                with self._lock:
                    self._counts["audits"] += 1
                    self._counts["audit_disagreements"] += int(_norm(truth) != _norm(local))
            self._count("table")
            return Decision(local, "table", r["confidence"], True, None, _us(t0))
        answer = self._ask(text, "teacher", name)
        self._count("teacher")
        return Decision(answer, "teacher", r["confidence"], False, r["flag"], _us(t0))

    __call__ = decide

    def stats(self, delta: float = 0.1) -> dict:
        """Live counters plus an upper confidence bound on disagreement among served answers, from the audits.

        The bound is a one-sided Clopper-Pearson bound at level 1 - delta. It holds for the audited stream so far,
        not for each moment of continuous monitoring; for that, use shad0w.audit.Auditor."""
        with self._lock:
            c = dict(self._counts)
        total = c["table"] + c["teacher"]
        c["offload"] = c["table"] / total if total else 0.0
        n, k = c["audits"], c["audit_disagreements"]
        c["audit_disagreement"] = k / n if n else None
        c["audit_disagreement_upper"] = _cp_upper(k, n, delta) if n else None
        return c

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def _ask(self, text: str, source: str, name: str):
        answer = self.teacher(text)
        if self.log:
            row = json.dumps({"text": text, name: answer, "source": source, "ts": round(time.time(), 3)},
                             ensure_ascii=False, default=str)
            with self._lock, open(self.log, "a", encoding="utf-8") as f:
                f.write(row + "\n")
        return answer


def cascade(bundle: str | Model | None, question: str | None = None, log: str | None = None,
            audit_rate: float = 0.01, exposed: bool = False):
    """Decorator form of `Shadow`: wraps your model call and returns the answer.

        @shad0w.cascade("bundle/", log="teacher_log.jsonl")
        def classify(text): return ask_llm(text)

        classify("block my card")      # answer from the table when certified, from ask_llm otherwise
        classify.shadow.stats()        # offload and live audit disagreement
    """
    def wrap(fn: Callable[[str], Any]):
        sh = Shadow(bundle, teacher=fn, question=question, log=log, audit_rate=audit_rate, exposed=exposed)

        def call(text: str):
            return sh.decide(text).answer

        call.shadow = sh  # type: ignore[attr-defined]
        call.__name__, call.__doc__, call.__wrapped__ = fn.__name__, fn.__doc__, fn  # type: ignore[attr-defined]
        return call
    return wrap


def _us(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e6


def _norm(v):
    return v if isinstance(v, str) else bool(v)


def _cp_upper(k: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper bound without scipy: bisection on the binomial CDF, P(X <= k; n, p) = delta."""
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:  # summed in log space so large n cannot underflow
        lp, lq = math.log(p), math.log1p(-p)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                   for i in range(k + 1))

    lo, hi = max(k / n, 1e-12), 1.0 - 1e-12
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi
