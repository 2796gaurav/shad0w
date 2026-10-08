"""Put shad0w in front of the model you already call, in one line.

    import shad0w

    intent = shad0w.decision("intent", options={"refund": "wants money back", "lost_card": "card lost or stolen"},
                             llm="openai/gpt-6-luna")
    intent("my card was stolen yesterday")    # Decision(answer='lost_card', source='teacher', ...)  logged
    intent.train()                            # once ~1,000 answers are logged: compile + certify, hot-swap
    intent("my card was stolen yesterday")    # Decision(answer='lost_card', source='table', latency_us=9, ...)

Or wrap any function you already have: `shad0w.Shadow("bundle/", teacher=ask_llm, log="teacher_log.jsonl")`.

Day one, before any bundle exists, every call goes to your model and is logged. After `train()` (or the
`shad0w train` command), certified answers come from the table and the rest still go to your model, logged,
so the next training run has more data.

Spot checks: a small random share of decisions (`audit_rate`, default 1%) is also sent to your model. For a
certified answer that happens in the background; for a deferred one the answer you got anyway is marked
`"source": "audit"`. Together those rows are a uniform sample of live traffic: `stats()` turns the certified
ones into a live disagreement bound, and `train()` certifies on all of them when there are enough.

Rollout knobs (code > SHAD0W_* env > shad0w.toml > defaults; see shad0w.config): mode="shadow" computes the
table's answer but always returns your model's; canary=0.1 lets the table answer 10% of what it could;
never_serve=["fraud"] keeps some labels with your model; min_confidence raises the bar above the certificate.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import math
import os
import random
import threading
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import config as _config
from .api import Model, load
from .observe import Metrics, TraceWriter, emit_span, log, otel_enabled

__all__ = ["Shadow", "Decision", "cascade", "decision", "decide", "explain", "explain_model"]

MIN_UNIFORM_CAL = 100  # audit rows needed before train() certifies on them instead of a random split


@dataclass(frozen=True)
class Decision:
    answer: Any  # the option name (choice) or a bool (yesno)
    source: str  # "table" (served locally) or "teacher" (deferred to your model)
    confidence: float | None  # the table's confidence; None when no table was consulted
    certified: bool
    flag: str | None  # why the table deferred (or why a served answer is not certified); see explain()
    latency_us: float  # end-to-end time of this call
    question: str = ""
    threshold: float | None = None  # the certified confidence threshold of the loaded table
    top: list[tuple[str, float]] | None = None  # options by probability, when probabilities were requested

    def __str__(self) -> str:
        return str(self.answer)


_FLAG_WORDS = {
    None: "certified: answered by the table",
    "no_bundle": "no trained table yet: asked your model",
    "low_confidence": "table not sure enough to stay inside the certified bound: asked your model",
    "low_radius": "input could be flipped by a few edits (exposed mode): asked your model",
    "drift": "traffic looks different from calibration: asked your model until the window recovers or you re-train",
    "uncalibrated": "table has no certificate: asked your model",
    "min_confidence": "below your min_confidence floor (stricter than the certificate): asked your model",
    "never_serve": "this label is on never_serve: asked your model",
    "canary": "held back by the canary share: asked your model",
    "shadow": "shadow mode: the table's answer was recorded, your model's was returned",
    "off": "mode=off: the table was not consulted",
    "manual_threshold": "served under force_threshold: NOT covered by the certificate",
}


def explain(flag: str | None) -> str:
    """The reason behind a decision, in plain words."""
    return _FLAG_WORDS.get(flag, flag or "")


def explain_model(model: Model | None, name: str, text: str, exposed: bool = False) -> dict:
    """What a table thinks of `text`, without calling any model and without touching the drift guard."""
    if model is None or name not in model.questions:
        return {"text": text, "question": name, "certified": False, "flag": "no_bundle", "why": explain("no_bundle")}
    q = model.questions[name]
    r = q.decide(text, exposed=exposed, observe=False)
    top = sorted((r.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:3]
    return {"text": text, "question": name, "answer": r.get("choice", r.get("answer")), "confidence": r["confidence"],
            "threshold": q.threshold, "alpha": q.alpha, "certified": r["certified"], "flag": r["flag"],
            "why": explain(r["flag"]), "top": top}


class Shadow:
    """A certified cascade around one question: the table when it is certified, your model otherwise.

    bundle       path to a compiled bundle, a loaded `Model`, or None (log-only: every call goes to the teacher).
                 A path that does not exist yet is fine: the wrapper starts log-only and `train()` creates it.
    teacher      callable(text) -> answer, your existing model call (or `shad0w.llm_teacher(...)`)
    question     which question of the bundle to answer (default: the bundle's only question, or "decision")
    log          JSON Lines file that receives every teacher answer in the format `shad0w train` reads
    seed         random seed for spot checks and the canary
    schema       the question's options ({"type": "choice", "criteria": {...}}); taken from the teacher or the
                 bundle when omitted, else learned from the answers in the log at training time
    on_decision  callable(decision, text) run after every decision (send it to your tracing / analytics)
    trace        JSON Lines file (or a shared `TraceWriter`) that receives EVERY decision, table and teacher
    metrics      a shared `shad0w.observe.Metrics` (default: a private one; see `stats()` and `metrics.prometheus()`)
    probabilities  fill `Decision.top` on every call (slower; default False)
    config       path to a shad0w.toml (default: ./shad0w.toml or $SHAD0W_CONFIG)

    Every other keyword is a setting (see `shad0w.config.DEFAULTS`): alpha, delta, min_rows, audit_rate,
    auto_train, retrain, mode, canary, never_serve, min_confidence, force_threshold, drift_window, drift_margin,
    exposed, cost_per_call, llm_latency_ms, cal_fraction, max_cal. Unset ones come from the environment, the
    config file, then the defaults.
    """

    def __init__(self, bundle: str | Model | None, teacher: Callable[[str], Any] | None, question: str | None = None,
                 log: str | None = None, *, seed: int | None = None, schema: dict | None = None,
                 on_decision: Callable[[Decision, str], Any] | None = None, trace: str | TraceWriter | None = None,
                 metrics: Metrics | None = None, probabilities: bool = False, config: str | None = None,
                 settings: _config.Settings | None = None, **kw):
        if teacher is not None and not callable(teacher):
            raise TypeError("teacher must be a callable: text -> answer")
        self.question = question if question is not None else getattr(teacher, "question", None)
        self.settings = settings or _config.resolve(self.question or "decision", path=config, **kw)
        s = self.settings
        self.teacher, self.log = teacher, log
        self.audit_rate, self.alpha, self.exposed = s.audit_rate, s.alpha, s.exposed
        self.auto_train = s.auto_train or None
        self.on_decision, self.probabilities = on_decision, probabilities
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._train_lock = threading.Lock()
        self._counts = {"table": 0, "teacher": 0, "audits": 0, "audit_disagreements": 0, "would_serve": 0,
                        "shadow_disagreements": 0}
        self._since_train = 0
        self._otel = otel_enabled()
        self.metrics = metrics or Metrics(cost_per_call=s.cost_per_call, llm_latency_ms=s.llm_latency_ms)
        trace = trace if trace is not None else s.trace
        self.trace = trace if isinstance(trace, TraceWriter) else (TraceWriter(trace) if trace else None)
        self.model: Model | None = None
        self._audits: set = set()  # spot checks in flight (they run off the request path)
        if schema is None and hasattr(teacher, "criteria"):  # an llm_teacher knows the options
            schema = {"type": teacher.qtype, "criteria": dict(teacher.criteria)}
        self.schema = schema
        self.bundle_path = os.fspath(bundle) if isinstance(bundle, (str, os.PathLike)) else None
        exists = self.bundle_path is None or os.path.exists(os.path.join(self.bundle_path, "manifest.json"))
        self.reload(bundle if exists else None)

    # -- bundle -----------------------------------------------------------------------------------------------
    def reload(self, bundle: str | Model | None = None) -> Shadow:
        """Swap in a new bundle (for example after `shad0w train`) without restarting. With no argument,
        reloads the bundle path this wrapper was created with."""
        if bundle is None and self.bundle_path and os.path.exists(os.path.join(self.bundle_path, "manifest.json")):
            bundle = self.bundle_path
        model = load(bundle) if isinstance(bundle, (str, os.PathLike)) else bundle
        if model is not None:
            names = list(model.questions)
            if self.question is None:
                if len(names) != 1:
                    raise ValueError(f"bundle has questions {names}; pass question=...")
                self.question = names[0]
            elif self.question not in model.questions:
                raise ValueError(f"question {self.question!r} is not in the bundle ({names})")
            q = model.questions[self.question]
            s = self.settings
            if (s.drift_window, s.drift_margin) != (_config.DEFAULTS["drift_window"], _config.DEFAULTS["drift_margin"]):
                q.guard.reconfigure(s.drift_window, s.drift_margin)
            if s.force_threshold is not None and math.isfinite(q.threshold) and s.force_threshold < q.threshold:
                log.warning("force_threshold=%s is below the certified threshold %.3f for %r: answers between them are "
                            "served with certified=False", s.force_threshold, q.threshold, self.question)
            self.metrics.set_alpha(self.question, q.alpha)
            if self.schema is None:
                self.schema = {"type": q.qtype, "criteria": {o: None for o in q.options}} if q.qtype == "choice" else {"type": "yesno"}
            log.info("loaded bundle for %r: %d options, alpha=%s, C core=%s", self.question, len(q.options), q.alpha,
                     model.native)
            if self.auto_train and isinstance(bundle, (str, os.PathLike)):
                from .shadow import read_certificate
                cert = read_certificate(os.fspath(bundle)) or {}
                cq = (cert.get("questions") or {}).get(self.question, {})
                n_rec = cq.get("n_records")
                if n_rec:
                    if cq.get("calibration") == "uniform-audit":
                        n_rec += cq.get("n_calibration", 0)
                    self._since_train = max(0, self.log_rows() - int(n_rec))
        with self._lock:
            self.model = model
        return self

    # -- deciding ---------------------------------------------------------------------------------------------
    def _table(self, text: str, probabilities: bool | None = None, observe: bool = True):
        """(raw table result, local answer) for one text; the model must be loaded."""
        name = self.question or "decision"
        r = self.model.decide(text, exposed=self.exposed, questions={name: {}},
                              probabilities=self.probabilities if probabilities is None else probabilities,
                              observe=observe)["answers"][name]
        return r, (r["choice"] if "choice" in r else r["answer"])

    def _policy(self, r: dict, local) -> tuple[bool, bool, str | None]:
        """(serve from the table?, certified?, flag) after the rollout knobs are applied."""
        s = self.settings
        if s.mode == "off":
            return False, False, "off"
        certified, flag = r["certified"], r["flag"]
        if certified and s.min_confidence is not None and r["confidence"] < s.min_confidence:
            certified, flag = False, "min_confidence"
        if certified and s.never_serve and local in s.never_serve:
            certified, flag = False, "never_serve"
        if certified and s.canary < 1.0 and self._rng.random() >= s.canary:
            certified, flag = False, "canary"
        if not certified and flag == "low_confidence" and s.force_threshold is not None and r["confidence"] >= s.force_threshold:
            return True, False, "manual_threshold"
        if s.mode == "shadow":
            return False, False, ("shadow" if certified else flag)
        return certified, certified, flag

    def _mk(self, answer, source: str, r: dict | None, certified: bool, flag: str | None, t0: float) -> Decision:
        name = self.question or "decision"
        top = None
        if r is not None and "probabilities" in r:
            top = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])
        thr = self.model.questions[name].threshold if self.model is not None else None
        return Decision(answer, source, None if r is None else r["confidence"], certified, flag, _us(t0), name,
                        None if thr is None or not math.isfinite(thr) else thr, top)

    def _src(self) -> str:
        """Label for a deferred row: "audit" with probability audit_rate, so audits sample all traffic uniformly."""
        return "audit" if self.audit_rate and self._rng.random() < self.audit_rate else "teacher"

    def _spot_check(self, teacher, text: str, name: str, local) -> None:
        if self.audit_rate and teacher is not None and self._rng.random() < self.audit_rate:
            th = threading.Thread(target=self._audit, args=(teacher, text, name, local), daemon=True, name="shad0w-audit")
            with self._lock:
                self._audits.add(th)
            th.start()

    def decide(self, text: str, teacher: Callable[[str], Any] | None = None, *, probabilities: bool | None = None) -> Decision:
        """The table's answer when certified, otherwise your model's (logged). `teacher` overrides the
        wrapper's teacher for this one call."""
        t0 = time.perf_counter()
        ns0 = time.time_ns() if self._otel else 0
        name = self.question or "decision"
        teacher = teacher or self.teacher
        model = self.model
        if model is None:
            answer, ts = self._ask(teacher, text, self._src(), name)
            return self._done(self._mk(answer, "teacher", None, False, "no_bundle", t0), text, name, ts, ns0)
        r, local = self._table(text, probabilities)
        serve, certified, flag = self._policy(r, local)
        if serve:
            d = self._mk(local, "table", r, certified, flag, t0)
            if certified:
                self._spot_check(teacher, text, name, local)
            return self._done(d, text, name, None, ns0)
        answer, ts = self._ask(teacher, text, self._src(), name)
        self._shadow_count(flag, r, answer, local)
        return self._done(self._mk(answer, "teacher", r, False, flag, t0), text, name, ts, ns0)

    __call__ = decide

    def _shadow_count(self, flag, r, answer, local):
        if self.settings.mode == "shadow":
            with self._lock:
                if flag == "shadow":
                    self._counts["would_serve"] += 1
                    self._counts["shadow_disagreements"] += int(_norm(answer) != _norm(local))

    def peek(self, text: str, *, probabilities: bool | None = None) -> Decision | None:
        """The table's answer when it would be served (counted and traced like any table decision), else None.
        Never calls your model: integrations use it, then `record()` the answer they fetched themselves."""
        if self.model is None:
            return None
        t0 = time.perf_counter()
        name = self.question or "decision"
        r, local = self._table(text, probabilities)
        serve, certified, flag = self._policy(r, local)
        if not serve:
            return None
        d = self._mk(local, "table", r, certified, flag, t0)
        if certified:
            self._spot_check(self.teacher, text, name, local)
        return self._done(d, text, name, None, time.time_ns() if self._otel else 0)

    def record(self, text: str, answer, *, teacher_s: float | None = None) -> Decision:
        """Log an answer you obtained from your model yourself and count it as a teacher decision."""
        t0 = time.perf_counter()
        name = self.question or "decision"
        self._record(text, self._src(), name, answer)
        if self.model is None:
            d = self._mk(answer, "teacher", None, False, "no_bundle", t0)
        else:
            r, local = self._table(text, observe=False)
            flag = self._policy(r, local)[2]
            self._shadow_count(flag, r, answer, local)
            d = self._mk(answer, "teacher", r, False, flag, t0)
        return self._done(d, text, name, teacher_s, time.time_ns() if self._otel else 0)

    def flush(self, timeout: float | None = None) -> None:
        """Wait for spot checks still in flight (they run in the background so they never slow an answer)."""
        with self._lock:
            pending = list(self._audits)
        for th in pending:
            if isinstance(th, threading.Thread):
                th.join(timeout)

    async def adecide(self, text: str, teacher: Callable[[str], Any] | None = None) -> Decision:
        """`decide` for async code. The table path never leaves the event loop; an async teacher is awaited,
        a sync teacher runs in a worker thread."""
        teacher = teacher or self.teacher
        if not _is_async(teacher):
            model = self.model
            if model is not None:
                r, local = self._table(text, observe=False)
                if self._policy(r, local)[0]:
                    return self.decide(text, teacher)  # microseconds; observes the guard exactly once
            return await asyncio.to_thread(self.decide, text, teacher)
        t0 = time.perf_counter()
        name = self.question or "decision"
        model = self.model
        flag, r = "no_bundle", None
        if model is not None:
            r, local = self._table(text)
            serve, certified, flag = self._policy(r, local)
            if serve:
                d = self._mk(local, "table", r, certified, flag, t0)
                if certified and self.audit_rate and self._rng.random() < self.audit_rate:
                    task = asyncio.ensure_future(self._aaudit(teacher, text, name, local))
                    self._audits.add(task)
                    task.add_done_callback(self._audits.discard)
                return self._done(d, text, name, None, 0)
        t = time.perf_counter()
        answer = await teacher(text)
        took = time.perf_counter() - t
        self._record(text, self._src(), name, answer)
        if r is not None:
            self._shadow_count(flag, r, answer, local)
        return self._done(self._mk(answer, "teacher", r, False, flag, t0), text, name, took, 0)

    async def _aaudit(self, teacher, text, name, local):
        try:
            truth = await teacher(text)
        except Exception as e:
            log.warning("audit call failed for %r: %s", name, e)
            return
        self._record(text, "audit", name, truth)
        self._audit_result(name, local, truth, text)

    def explain(self, text: str) -> dict:
        """What the table thinks of `text`, without calling your model: top options, confidence, the
        certified threshold, the reason in plain words, and what the rollout knobs would do with it."""
        out = explain_model(self.model, self.question or "decision", text, self.exposed)
        if self.model is not None:
            r = self.model.questions[self.question or "decision"].decide(text, exposed=self.exposed, observe=False)
            local = r["choice"] if "choice" in r else r["answer"]
            serve, certified, flag = self._policy(r, local)
            out.update({"would_serve": serve, "policy_flag": flag, "mode": self.settings.mode})
        return out

    def _audit(self, teacher, text, name, local):
        try:
            truth, _ = self._ask(teacher, text, "audit", name)
        except Exception as e:  # a failing spot check must not fail a certified answer
            log.warning("audit call failed for %r: %s", name, e)
            return
        self._audit_result(name, local, truth, text)
        with self._lock:
            self._audits.discard(threading.current_thread())

    def _audit_result(self, name, local, truth, text):
        if truth is None:
            return
        dis = _norm(truth) != _norm(local)
        with self._lock:
            self._counts["audits"] += 1
            self._counts["audit_disagreements"] += int(dis)
        self.metrics.record_audit(name, dis)
        if dis:
            log.info("audit disagreement on %r: table=%r model=%r text=%r", name, local, truth, text[:80])

    def _done(self, d: Decision, text: str, name: str, teacher_s: float | None, ns0: int) -> Decision:
        with self._lock:
            self._counts[d.source] += 1
        self.metrics.record(name, d.source, d.answer, d.latency_us / 1e6, d.confidence, d.flag, text, teacher_s)
        if log.isEnabledFor(10):  # DEBUG
            log.debug("%s -> %r via %s conf=%s flag=%s %.1fus | %s", name, d.answer, d.source,
                      None if d.confidence is None else round(d.confidence, 3), d.flag, d.latency_us, text[:80])
        if self.trace:
            self.trace.write({"ts": round(time.time(), 3), "question": name, "text": text, "answer": d.answer,
                              "source": d.source, "confidence": d.confidence, "flag": d.flag,
                              "latency_us": round(d.latency_us, 2)})
        if self._otel:
            emit_span(name, d, ns0, time.time_ns(), getattr(self.teacher, "name", None))
        if self.on_decision is not None:
            try:
                self.on_decision(d, text)
            except Exception as e:
                log.warning("on_decision hook failed: %s", e)
        return d

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
        q = self.metrics.snapshot(delta)["questions"].get(self.question or "decision", {})
        c.update({k: q.get(k) for k in ("table_p50_us", "table_p99_us", "llm_p50_ms", "llm_p99_ms", "flags")})
        c["alpha"] = self.model.questions[self.question].alpha if self.model is not None else None
        c["threshold"] = self.model.questions[self.question].threshold if self.model is not None else None
        c["mode"], c["canary"] = self.settings.mode, self.settings.canary
        c["log_rows"] = self.log_rows()
        return c

    def _ask(self, teacher, text: str, source: str, name: str):
        if teacher is None:
            raise RuntimeError("this Shadow has no teacher; pass teacher=... (or call with decide(text, teacher=...))")
        t = time.perf_counter()
        answer = teacher(text)
        took = time.perf_counter() - t
        self._record(text, source, name, answer)
        return answer, took

    def _record(self, text: str, source: str, name: str, answer) -> None:
        """Append one teacher answer to the log (the training data) and trigger auto_train when due."""
        if self.log and answer is not None:
            row = json.dumps({"text": text, name: answer, "source": source, "ts": round(time.time(), 3)},
                             ensure_ascii=False, default=str)
            due = False
            with self._lock, open(self.log, "a", encoding="utf-8") as f:
                f.write(row + "\n")
                if self.auto_train and source != "table":
                    self._since_train += 1
                    due = self._since_train >= self.auto_train
            if due and not self._train_lock.locked():
                with self._lock:
                    self._since_train = 0
                threading.Thread(target=self._train_quietly, daemon=True, name="shad0w-train").start()

    # -- training ---------------------------------------------------------------------------------------------
    def log_rows(self) -> int:
        if not self.log or not os.path.exists(self.log):
            return 0
        with open(self.log, "rb") as f:
            return sum(1 for line in f if line.strip())

    def train(self, out: str | None = None, alpha: float | None = None, delta: float | None = None,
              min_rows: int | None = None, *, gate: bool = False) -> dict:
        """Compile and certify a bundle from this wrapper's log, save it and start serving from it.
        Returns the certificate for this question (plus "accepted"). Needs `pip install "shad0wllm[compile]"`.

        gate=True keeps the current bundle when the new one certifies less than 80% of its share (or nothing);
        that is what auto_train does under retrain="gated"."""
        from .shadow import read_certificate, read_jsonl, shadow_compile, write_certificate
        s = self.settings
        out = out or self.bundle_path
        if not out:
            raise ValueError("pass out='bundle/' (where to save the trained bundle)")
        if not self.log or not os.path.exists(self.log):
            raise ValueError("nothing to train on: this Shadow has no log file yet")
        name = self.question or "decision"
        alpha = s.alpha if alpha is None else alpha
        delta = s.delta if delta is None else delta
        min_rows = s.min_rows if min_rows is None else min_rows
        with self._train_lock:
            rows, schema = rows_for_training(read_jsonl(self.log), name, self.schema)
            if len(rows) < min_rows:
                raise ValueError(f"{len(rows)} usable answers logged for {name!r}; need {min_rows} "
                                 f"(pass min_rows=... to try with fewer; 100 is the hard minimum)")
            cal = [r for r in rows if r.get("source") == "audit"]
            fit_rows, cal_rows = rows, None
            if len(cal) >= MIN_UNIFORM_CAL:
                fit_rows = [r for r in rows if r.get("source") != "audit"]
                cal_rows = cal
                log.info("certifying %r on %d uniform spot-check rows (fitting on the other %d)", name, len(cal), len(fit_rows))
            else:
                log.info("certifying %r on a random held-out split (%d spot-check rows, need %d for a uniform sample)",
                         name, len(cal), MIN_UNIFORM_CAL)
            t = time.time()
            try:
                model, cert = shadow_compile({name: schema}, fit_rows, alpha=alpha, delta=delta,
                                             cal_fraction=s.cal_fraction, max_cal=s.max_cal,
                                             teacher=getattr(self.teacher, "name", "unspecified"),
                                             cal_records=cal_rows, drift_window=s.drift_window, drift_margin=s.drift_margin)
            except ModuleNotFoundError as e:
                if (e.name or "").split(".")[0] in ("sklearn", "scipy", "torch"):
                    raise ImportError('training needs: pip install "shad0wllm[compile]" (scipy, scikit-learn)') from e
                raise
            q = cert["questions"][name]
            old = (read_certificate(out) or {}).get("questions", {}).get(name) if gate else None
            if old and old.get("threshold") is not None and self.model is not None:
                new_share, old_share = q["certified_share_on_calibration"], old.get("certified_share_on_calibration", 0.0)
                if q["threshold"] is None or new_share < 0.8 * old_share:
                    reason = (f"new bundle certifies {new_share:.1%} of calibration traffic vs {old_share:.1%} before; "
                              "kept the old one (retrain='always' to replace anyway)")
                    log.warning("train %r rejected: %s", name, reason)
                    return {**q, "accepted": False, "reason": reason}
            model.save(out)
            write_certificate(out, cert)
            with open(os.path.join(out, "schema.json"), "w", encoding="utf-8") as f:
                json.dump({name: schema}, f, indent=2)
            self.bundle_path, self.schema = out, schema
            self.reload(out)
            with self._lock:
                self._since_train = 0
            log.info("trained %r on %d answers in %.1fs: certified share %.1f%% at alpha=%s (%s)", name, len(rows),
                     time.time() - t, 100 * q["certified_share_on_calibration"], alpha, q.get("calibration"))
            return {**q, "accepted": True}

    def _train_quietly(self):
        try:
            self.train(gate=(self.settings.retrain == "gated"))
        except Exception as e:
            log.warning("auto_train skipped: %s", e)

    def close(self):
        if self.trace:
            self.trace.close()


def rows_for_training(records: list[dict], name: str, schema: dict | None):
    """Teacher answers usable for `name` (answers outside the options are dropped) and the schema to train with.
    Without a schema the options are the answers seen in the log."""
    rows = [r for r in records if name in r and r.get("text") and r[name] is not None and r.get("source") != "table"]
    if schema is None:
        vals = {r[name] for r in rows}
        if vals <= {True, False}:
            schema = {"type": "yesno"}
        else:
            counts = {}
            for r in rows:
                counts[str(r[name])] = counts.get(str(r[name]), 0) + 1
            schema = {"type": "choice", "criteria": {k: None for k in sorted(counts, key=lambda k: -counts[k])}}
    if schema.get("type", "choice") == "choice":
        opts = set(schema["criteria"])
        if len(opts) < 2:
            raise ValueError(f"{name!r}: need at least 2 different answers to train, saw {sorted(opts)}")
        rows = [r for r in rows if str(r[name]) in opts]
        rows = [{**r, name: str(r[name])} for r in rows]
    else:
        from .api import to_bool
        rows = [{**r, name: to_bool(r[name])} for r in rows]
    return rows, schema


_TEACHER_KW = ("base_url", "api_key", "client", "system", "temperature", "timeout", "retries", "headers", "structured",
               "max_tokens", "complete")


def decision(name: str, options: Any = None, llm: str | Callable[[str], Any] | None = None, *,
             bundle: str | None = None, log: str | None = None, folder: str | None = None, **kw) -> Shadow:
    """One call from nothing to a certified cascade.

        intent = shad0w.decision("intent", options={"refund": "wants money back", ...}, llm="openai/gpt-6-luna")
        intent("my card was stolen")     # your LLM answers and is logged; later the table answers
        intent.train()                   # when ~1,000 answers are logged

    options  list of option names, {name: description}, or a schema entry; optional when the bundle exists
    llm      "provider/model" (openai, anthropic, gemini, groq, ollama, openai-decisions, systemone, ...; see
             shad0w.llm.PROVIDERS) or any callable(text) -> answer
    folder   where the log and bundle live (default: <settings.folder>/<name>/); bundle= and log= override each
    Other keywords go to `Shadow` (audit_rate, auto_train, mode, canary, on_decision, trace, alpha, ...) and, for a
    "provider/model" string, to `llm_teacher` (base_url, api_key, client, complete, system, temperature, ...)."""
    from .llm import LLMTeacher, normalize_options
    llm_kw = {k: kw.pop(k) for k in list(kw) if k in _TEACHER_KW}
    settings = _config.resolve(name, path=kw.get("config"), **{k: v for k, v in kw.items() if k in _config.DEFAULTS})
    kw["settings"] = settings
    if folder is None:
        folder = os.path.join(settings.folder, name)
    bundle = bundle or os.path.join(folder, "bundle")
    log_path = log or os.path.join(folder, "log.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    schema = None
    if options is not None:
        _, qtype, crit, instr = normalize_options(options, name)
        schema = {"type": qtype, "criteria": crit} if qtype == "choice" else {"type": "yesno"}
        if instr:
            schema["instructions"] = instr
    elif os.path.exists(os.path.join(bundle, "schema.json")):
        with open(os.path.join(bundle, "schema.json"), encoding="utf-8") as f:
            schema = json.load(f).get(name)
    if isinstance(llm, str):
        if schema is None:
            raise ValueError("pass options=... so the LLM knows what to choose from")
        teacher = LLMTeacher({name: schema}, llm, question=name, **llm_kw)
    else:
        teacher = llm
    shadow_kw = {k: v for k, v in kw.items() if k not in _config.DEFAULTS}
    return Shadow(bundle, teacher=teacher, question=name, log=log_path, schema=schema, **shadow_kw)


def cascade(bundle: str | Model | None, question: str | None = None, log: str | None = None,
            audit_rate: float | None = None, exposed: bool | None = None, **kw):
    """Decorator form of `Shadow`: wraps your model call and returns the answer.

        @shad0w.cascade("bundle/", log="teacher_log.jsonl")
        def classify(text): return ask_llm(text)

        classify("block my card")      # answer from the table when certified, from ask_llm otherwise
        classify.shadow.stats()        # offload and live audit disagreement
    """
    def wrap(fn: Callable[[str], Any]):
        sh = Shadow(bundle, teacher=fn, question=question, log=log, audit_rate=audit_rate, exposed=exposed, **kw)
        return _wrapped(fn, sh)
    return wrap


def decide(fn: Callable | None = None, /, *, name: str | None = None, folder: str | None = None, **kw):
    """Decorator: the function's return annotation names the options, its body is the teacher.

        @shad0w.decide()
        def route(text) -> Literal["billing", "tech", "sales"]:
            return ask_llm(text)

        route("my invoice is wrong")     # "billing": from the table when certified, else from ask_llm (logged)
        route.shadow.train()             # once ~1,000 answers are logged

    A `-> bool` annotation makes a yes/no question. Files live in <folder>/<name>/ like `decision()`.
    """
    def wrap(f: Callable):
        hints = typing.get_type_hints(f)
        ret = hints.get("return")
        if ret is bool:
            options = {"type": "yesno"}
        elif typing.get_origin(ret) is typing.Literal:
            options = [str(v) for v in typing.get_args(ret)]
        else:
            raise TypeError(f"{f.__name__}: annotate the return type with Literal[...] (the options) or bool")
        sh = decision(name or f.__name__, options=options, llm=f, folder=folder, **kw)
        return _wrapped(f, sh)
    return wrap(fn) if fn is not None else wrap


def _wrapped(fn, sh: Shadow):
    @functools.wraps(fn)
    def call(text: str):
        return sh.decide(text).answer

    call.shadow = sh  # type: ignore[attr-defined]
    return call


def _is_async(fn) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(type(fn).__call__)


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
