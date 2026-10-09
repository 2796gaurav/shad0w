"""Put shad0w in front of the model you already call, in one line.

    import shad0w

    intent = shad0w.decision("intent", options={"refund": "wants money back", "lost_card": "card lost or stolen"},
                             llm="openai/gpt-6-luna")
    intent("my card was stolen yesterday")    # Decision(answer='lost_card', source='teacher', ...)  logged
    intent.train()                            # once ~1,000 answers are logged: compile + certify, hot-swap
    intent("my card was stolen yesterday")    # Decision(answer='lost_card', source='table', latency_us=9, ...)

The key is yours to pass: `api_key="sk-..."` (or a function that returns it), `api_key_env="MY_KEY_VAR"`, or once for
the process with `shad0w.configure(api_key_env=...)`; otherwise the provider's usual variable (OPENAI_API_KEY, ...).

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

_TEACHER_KW = ("base_url", "api_key", "api_key_env", "client", "system", "temperature", "timeout", "retries", "headers",
               "structured", "max_tokens", "complete")

_NO_FALLBACK = object()  # sentinel: no fallback configured (decide() needs a teacher for what the table defers)
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

    @property
    def why(self) -> str:
        """Why this answer came from where it did, in plain words."""
        w = explain(self.flag)
        if self.source == "fallback":
            w = w.replace("asked your model", "used your fallback (no LLM configured)")
            if self.flag == "no_bundle":
                w = "no trained table yet: used your fallback (no LLM configured)"
        return w


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
    "options_changed": "your options include ones the table never learned: asked your model until you retrain",
    "option_removed": "the table picked an option you removed: asked your model",
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
    fallback     what to answer when the table defers and there is no teacher (running without an LLM): a value such
                 as "needs_review", or callable(text) -> answer (rules, a queue). Fallback answers are never logged
                 as training data and come back with source="fallback".
    llm          instead of teacher=: a "provider/model" string (Shadow builds the LLM teacher itself; needs the
                 options via schema= or an existing bundle) or a callable(text) -> answer
    api_key      the LLM key: a string or a zero-argument function (called on every request). Never printed.
    api_key_env  the NAME of the environment variable that holds the key
    base_url     an OpenAI-compatible server for the llm string

    Every other keyword is a setting (see `shad0w.config.DEFAULTS`): alpha, delta, min_rows, audit_rate,
    auto_train, retrain, mode, canary, never_serve, min_confidence, force_threshold, drift_window, drift_margin,
    exposed, cost_per_call, llm_latency_ms, cal_fraction, max_cal. Unset ones come from the environment, the
    config file, then the defaults.
    """

    def __init__(self, bundle: str | Model | None, teacher: Callable[[str], Any] | None = None, question: str | None = None,
                 log: str | None = None, *, llm: str | Callable[[str], Any] | None = None, api_key: Any = None,
                 api_key_env: str | None = None, base_url: str | None = None, seed: int | None = None,
                 schema: dict | None = None, on_decision: Callable[[Decision, str], Any] | None = None,
                 trace: str | TraceWriter | None = None, metrics: Metrics | None = None, probabilities: bool = False,
                 config: str | None = None, settings: _config.Settings | None = None, fallback: Any = _NO_FALLBACK, **kw):
        llm_kw = {k: kw.pop(k) for k in list(kw) if k in _TEACHER_KW}
        unknown = set(kw) - set(_config.DEFAULTS)
        if unknown or (settings is not None and kw):
            known = set(_config.DEFAULTS) | set(_TEACHER_KW) | _SHADOW_PARAMS
            raise _config.unknown_keywords(set(kw) if settings is not None else unknown, known, "Shadow()")
        llm_kw.update({k: v for k, v in (("api_key", api_key), ("api_key_env", api_key_env), ("base_url", base_url))
                       if v is not None})
        if llm is not None and teacher is not None:
            raise TypeError("pass teacher= or llm=, not both")
        if callable(llm):
            teacher, llm = llm, None
        if llm_kw and llm is None:
            raise TypeError(f"{sorted(llm_kw)} only apply when llm is a \"provider/model\" string; your LLM is a "
                            "function, so it holds its own key and URL (pass them where you build that function)")
        if llm is not None and not isinstance(llm, str):
            raise TypeError('llm must be a "provider/model" string or a callable(text) -> answer')
        if teacher is not None and not callable(teacher):
            raise TypeError("teacher must be a callable: text -> answer")
        self.question = question if question is not None else getattr(teacher, "question", None)
        self.settings = settings or _config.resolve(self.question or "decision", path=config, **kw)
        s = self.settings
        self.teacher, self.log = teacher, log
        self.fallback = fallback
        self.audit_rate, self.alpha, self.exposed = s.audit_rate, s.alpha, s.exposed
        self.auto_train = s.auto_train or None
        self.on_decision, self.probabilities = on_decision, probabilities
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._train_lock = threading.Lock()
        self._counts = {"table": 0, "teacher": 0, "fallback": 0, "audits": 0, "audit_disagreements": 0, "would_serve": 0,
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
        self._cert: dict = {}
        self.bundle_path = os.fspath(bundle) if isinstance(bundle, (str, os.PathLike)) else None
        exists = self.bundle_path is None or os.path.exists(os.path.join(self.bundle_path, "manifest.json"))
        self.reload(bundle if exists else None)
        if llm is not None:
            self.teacher = self._llm_teacher(llm, llm_kw)

    def _llm_teacher(self, llm: str, llm_kw: dict):
        """Build the LLM teacher for a "provider/model" string from the options and the key settings."""
        from .llm import LLMTeacher
        name = self.question or "decision"
        if not self.schema:
            raise ValueError(f"llm={llm!r} needs the options: pass options=... (decision) or schema=... (Shadow)")
        s = self.settings
        kw = dict(llm_kw)
        kw.setdefault("base_url", s.base_url)
        return LLMTeacher({name: self.schema}, llm, question=name, settings_key_env=s.api_key_env, **kw)

    # -- options ----------------------------------------------------------------------------------------------
    @property
    def schema(self) -> dict | None:
        return self._schema

    @schema.setter
    def schema(self, value: dict | None) -> None:
        self._schema = value
        self._diff_options()

    def _diff_options(self) -> None:
        """Compare the options you configure now with the ones the loaded table learned (after `rename`).

        Added options: the table cannot answer them, so its other answers are not covered by the certificate
        either (it was certified on a world without them). on_new_option="defer" (default) sends everything to
        your LLM until you retrain. Removed options: answers naming them are never served."""
        self.options_added, self.options_removed = (), ()
        model, schema = getattr(self, "model", None), getattr(self, "_schema", None)
        if model is None or not schema or schema.get("type", "choice") != "choice" or not schema.get("criteria"):
            return
        q = model.questions.get(self.question or "decision")
        if q is None or q.qtype != "choice":
            return
        ren = self.settings.rename_map
        known = {ren.get(o, o) for o in q.options}
        wanted = {str(o) for o in schema["criteria"]}
        self.options_added, self.options_removed = tuple(sorted(wanted - known)), tuple(sorted(known - wanted))
        if self.options_added or self.options_removed:
            log.warning("options for %r changed since training: added %s, removed %s. %s Retrain to pick them up "
                        "(the log keeps collecting your LLM's answers).", self.question, list(self.options_added),
                        list(self.options_removed),
                        "Deferring every decision to your LLM until then." if self.options_added and
                        self.settings.on_new_option == "defer" else "Answers naming removed options go to your LLM.")

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
            if isinstance(bundle, (str, os.PathLike)):
                from .shadow import read_certificate
                self._cert = ((read_certificate(os.fspath(bundle)) or {}).get("questions") or {}).get(self.question, {})
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
        self._diff_options()
        return self

    # -- deciding ---------------------------------------------------------------------------------------------
    def _table(self, text: str, probabilities: bool | None = None, observe: bool = True):
        """(raw table result, local answer) for one text; the model must be loaded."""
        name = self.question or "decision"
        r = self.model.decide(text, exposed=self.exposed, questions={name: {}},
                              probabilities=self.probabilities if probabilities is None else probabilities,
                              observe=observe)["answers"][name]
        ren = self.settings.rename
        if ren and "choice" in r:
            ren = dict(ren)
            r = {**r, "choice": ren.get(r["choice"], r["choice"])}
            if "probabilities" in r:
                r["probabilities"] = {ren.get(k, k): v for k, v in r["probabilities"].items()}
        return r, (r["choice"] if "choice" in r else r["answer"])

    def _policy(self, r: dict, local) -> tuple[bool, bool, str | None]:
        """(serve from the table?, certified?, flag) after the rollout knobs are applied."""
        s = self.settings
        if s.mode == "off":
            return False, False, "off"
        if self.options_added and s.on_new_option == "defer":
            return False, False, "options_changed"
        if self.options_removed and local in self.options_removed:
            return False, False, "option_removed"
        certified, flag = r["certified"], r["flag"]
        if certified and s.min_confidence is not None and r["confidence"] < s.min_confidence:
            certified, flag = False, "min_confidence"
        if certified and s.never_serve and local in s.never_serve:
            certified, flag = False, "never_serve"
        if certified and s.canary < 1.0 and self._rng.random() >= s.canary:
            certified, flag = False, "canary"
        forced = (not certified and flag == "low_confidence" and s.force_threshold is not None
                  and r["confidence"] >= s.force_threshold)
        if s.mode == "shadow":  # shadow mode never serves, not even a forced threshold
            return False, False, ("shadow" if certified or forced else flag)
        if forced:
            return True, False, "manual_threshold"
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
        return self._decide_with(text, teacher or self.teacher, probabilities, None)

    def _decide_with(self, text: str, teacher, probabilities, pre) -> Decision:
        """decide(), optionally with the table's result already computed (pre = (t0, r, local, serve, certified, flag))."""
        ns0 = time.time_ns() if self._otel else 0
        name = self.question or "decision"
        if pre is None:
            t0 = time.perf_counter()
            if self.model is None:
                return self._defer(text, teacher, name, None, "no_bundle", None, t0, ns0)
            r, local = self._table(text, probabilities)
            serve, certified, flag = self._policy(r, local)
        else:
            t0, r, local, serve, certified, flag = pre
        if serve:
            d = self._mk(local, "table", r, certified, flag, t0)
            if certified:
                self._spot_check(teacher, text, name, local)
            return self._done(d, text, name, None, ns0)
        return self._defer(text, teacher, name, r, flag, local, t0, ns0)

    def _defer(self, text, teacher, name, r, flag, local, t0, ns0) -> Decision:
        """The table does not answer: ask the teacher (logged), or use the fallback when there is no teacher."""
        if teacher is None and self.fallback is not _NO_FALLBACK:
            answer = self.fallback(text) if callable(self.fallback) else self.fallback
            return self._done(self._mk(answer, "fallback", r, False, flag, t0), text, name, None, ns0)
        try:
            answer, ts = self._ask(teacher, text, self._src(), name)
        except Exception as e:
            _add_why(e, flag)
            raise
        if r is not None:
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

    async def adecide(self, text: str, teacher: Callable[[str], Any] | None = None, *,
                      probabilities: bool | None = None) -> Decision:
        """`decide` for async code. The table path never leaves the event loop; an async teacher is awaited,
        a sync teacher runs in a worker thread."""
        teacher = teacher or self.teacher
        return await self._adecide_pre(text, teacher, self._pre(text, probabilities))

    def _pre(self, text: str, probabilities: bool | None = None):
        """The table's part of a decision, computed once: (t0, r, local, serve, certified, flag), or None."""
        if self.model is None:
            return None
        t0 = time.perf_counter()
        r, local = self._table(text, probabilities)
        serve, certified, flag = self._policy(r, local)
        return (t0, r, local, serve, certified, flag)

    def _served(self, pre, teacher) -> bool:
        """True when the decision finishes without calling the teacher (table answer or fallback)."""
        return pre is not None and (pre[3] or (teacher is None and self.fallback is not _NO_FALLBACK))

    async def _adecide_pre(self, text: str, teacher, pre) -> Decision:
        if not _is_async(teacher):
            if self._served(pre, teacher):  # the table ran on the event loop (microseconds), once
                return self._decide_with(text, teacher, None, pre)
            return await asyncio.to_thread(self._decide_with, text, teacher, None, pre)
        name = self.question or "decision"
        flag, r, local = "no_bundle", None, None
        t0 = time.perf_counter()
        if pre is not None:
            t0, r, local, serve, certified, flag = pre
            if serve:
                d = self._mk(local, "table", r, certified, flag, t0)
                if certified and self.audit_rate and self._rng.random() < self.audit_rate:
                    task = asyncio.ensure_future(self._aaudit(teacher, text, name, local))
                    self._audits.add(task)
                    task.add_done_callback(self._audits.discard)
                return self._done(d, text, name, None, 0)
        t = time.perf_counter()
        try:
            answer = await teacher(text)
        except Exception as e:
            _add_why(e, flag)
            raise
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

    def decide_many(self, texts, concurrency: int = 8, teacher: Callable[[str], Any] | None = None) -> list[Decision]:
        """Decide a batch, in order. The table answers first (microseconds each, no threads); only the texts it
        defers go to your LLM, `concurrency` at a time in worker threads. The first LLM error is raised."""
        texts = list(texts)
        teacher = teacher or self.teacher
        pres = [self._pre(t) for t in texts]
        out: list[Decision | None] = [None] * len(texts)
        todo = []
        for i, (t, pre) in enumerate(zip(texts, pres)):
            if self._served(pre, teacher):
                out[i] = self._decide_with(t, teacher, None, pre)
            else:
                todo.append(i)
        if todo:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, min(int(concurrency), len(todo))),
                                    thread_name_prefix="shad0w-batch") as pool:
                futs = {i: pool.submit(self._decide_with, texts[i], teacher, None, pres[i]) for i in todo}
                for i, f in futs.items():
                    out[i] = f.result()
        return out  # type: ignore[return-value]

    async def adecide_many(self, texts, concurrency: int = 8,
                           teacher: Callable[[str], Any] | None = None) -> list[Decision]:
        """`decide_many` for async code: table answers on the event loop, deferred texts as tasks (at most
        `concurrency` LLM calls in flight). Order is preserved."""
        texts = list(texts)
        teacher = teacher or self.teacher
        pres = [self._pre(t) for t in texts]
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def one(t, pre):
            if self._served(pre, teacher):
                return await self._adecide_pre(t, teacher, pre)
            async with sem:
                return await self._adecide_pre(t, teacher, pre)
        return list(await asyncio.gather(*(one(t, p) for t, p in zip(texts, pres))))

    def warm_start(self, rows_or_path, text: str = "text", label: str | None = None, *, source: str = "import") -> dict:
        """Import answers you already have (an old LLM's log, human labels) into this decision's log, so `train()`
        can start from them instead of from zero.

        rows_or_path  a list of dicts, or a .jsonl / .json / .csv file
        text, label   the field names holding the text and the answer (label defaults to the decision's name)

        Labels are checked against the options: rows with an unknown label (or no text) are skipped and counted.
        Rows are written with source="import". Returns the counts."""
        if not self.log:
            raise ValueError("warm_start needs a log file: pass log=... (decision() sets one up for you)")
        name = self.question or "decision"
        label = label or name
        rows = _read_rows(rows_or_path)
        schema = self.schema or {}
        yesno = schema.get("type") == "yesno"
        opts = list(schema.get("criteria") or {}) if schema and not yesno else None
        ren = self.settings.rename_map
        from .llm import enum_option, match_option
        good, unknown, empty = [], {}, 0
        for r in rows:
            t, y = r.get(text), r.get(label)
            if not isinstance(t, str) or not t.strip() or y is None or (isinstance(y, str) and not y.strip()):
                empty += 1
                continue
            y = enum_option(y)
            if yesno:
                v = match_option(y if isinstance(y, (str, bool)) else str(y), ["yes", "no"])
                if v is None:
                    unknown[str(y)] = unknown.get(str(y), 0) + 1
                    continue
                y = v == "yes"
            elif opts is not None:
                y = ren.get(str(y), str(y))
                v = y if y in opts else match_option(y, opts)
                if v is None:
                    unknown[y] = unknown.get(y, 0) + 1
                    continue
                y = v
            good.append(json.dumps({"text": t, name: y, "source": source, "ts": round(time.time(), 3)},
                                   ensure_ascii=False, default=str))
        if good:
            with self._lock, open(self.log, "a", encoding="utf-8") as f:
                f.write("\n".join(good) + "\n")
        total = self.log_rows()
        out = {"imported": len(good), "skipped_unknown_label": sum(unknown.values()), "skipped_no_text": empty,
               "unknown_labels": dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:10]), "log_rows": total,
               "min_rows": self.settings.min_rows, "ready_to_train": total >= self.settings.min_rows}
        log.info("warm_start %r: imported %d rows (%d unknown labels, %d without text); log has %d of %d needed",
                 name, len(good), out["skipped_unknown_label"], empty, total, self.settings.min_rows)
        return out

    # -- what it is -------------------------------------------------------------------------------------------
    def _state(self) -> dict:
        name = self.question or "decision"
        n_opts = None
        if self.schema and self.schema.get("type", "choice") == "choice":
            n_opts = len(self.schema.get("criteria") or {})
        elif self.schema:
            n_opts = 2
        st = {"question": name, "options": n_opts, "mode": self.settings.mode}
        if self.model is None:
            st["state"] = f"logging {self.log_rows():,}/{self.settings.min_rows:,} answers (no table yet)"
        else:
            q = self.model.questions[name]
            share = self._cert.get("certified_share_on_calibration")
            st["state"] = (f"serving: certifies {share:.1%} at alpha={q.alpha}" if share is not None
                           else f"serving at alpha={q.alpha}")
        t = self.teacher
        st["llm"] = None if t is None else (getattr(t, "name", None) or getattr(t, "__name__", None) or type(t).__name__)
        st["key"] = getattr(t, "key_source", None) if t is not None else None
        return st

    def __repr__(self) -> str:
        st = self._state()
        parts = [f"{st['question']!r}", f"options={st['options']}", st["state"]]
        parts.append(f"llm={st['llm']}" if st["llm"] else "llm=None")
        if st["key"]:
            parts.append(f"key: {st['key']}")
        if st["mode"] != "serve":
            parts.append(f"mode={st['mode']}")
        return "Shadow(" + ", ".join(parts) + ")"

    def _repr_html_(self) -> str:
        import html
        st = self._state()
        rows = [("question", st["question"]), ("options", st["options"]), ("state", st["state"]),
                ("LLM", st["llm"] or "none"), ("key", st["key"] or "-"), ("mode", st["mode"])]
        body = "".join(f"<tr><th style='text-align:left;padding-right:1em'>{html.escape(str(k))}</th>"
                       f"<td>{html.escape(str(v))}</td></tr>" for k, v in rows)
        return f"<table><caption style='text-align:left'><b>shad0w.Shadow</b></caption>{body}</table>"

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
            self._audit_result(name, local, truth, text)
        except Exception as e:  # a failing spot check must not fail a certified answer
            log.warning("audit call failed for %r: %s", name, e)
        finally:
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
        c["options_added"], c["options_removed"] = list(self.options_added), list(self.options_removed)
        return c

    def _ask(self, teacher, text: str, source: str, name: str):
        if teacher is None:
            raise RuntimeError("the table did not answer and this Shadow has no teacher: pass teacher=... (or llm=...), "
                               "or fallback=... to run without an LLM")
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
        from .shadow import gate_reason, read_certificate, read_jsonl, shadow_compile, write_certificate
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
            rows, schema = rows_for_training(read_jsonl(self.log), name, self.schema, rename=s.rename_map)
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
                                             cal_records=cal_rows, drift_window=s.drift_window, drift_margin=s.drift_margin,
                                             max_mb=s.max_mb)
            except ModuleNotFoundError as e:
                if (e.name or "").split(".")[0] in ("sklearn", "scipy", "torch"):
                    raise ImportError('training needs: pip install "shad0wllm[compile]" (scipy, scikit-learn)') from e
                raise
            q = cert["questions"][name]
            old = (read_certificate(out) or {}).get("questions", {}).get(name) if gate and self.model is not None else None
            reason = gate_reason(q, old)
            if reason:
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


def rows_for_training(records: list[dict], name: str, schema: dict | None, rename: dict | None = None):
    """Teacher answers usable for `name` (answers outside the options are dropped) and the schema to train with.
    Without a schema the options are the answers seen in the log. `rename` maps old labels in the log to new ones,
    so renaming an option never throws its history away."""
    rows = [r for r in records if name in r and r.get("text") and r[name] is not None and r.get("source") != "table"]
    if rename:
        rows = [{**r, name: rename.get(r[name], r[name])} if isinstance(r[name], str) else r for r in rows]
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
        counts = {o: 0 for o in schema["criteria"]}
        for r in rows:
            counts[r[name]] += 1
        thin = {o: n for o, n in counts.items() if n < 20}
        if thin and rows:
            log.warning("%r: options with fewer than 20 logged answers %s; the table will rarely answer them "
                        "(those go to your LLM) until more are logged", name, thin)
    else:
        from .api import to_bool
        rows = [{**r, name: to_bool(r[name])} for r in rows]
    return rows, schema


_SHADOW_PARAMS = {"seed", "schema", "on_decision", "trace", "metrics", "probabilities", "config", "settings", "fallback",
                  "llm", "api_key", "api_key_env", "base_url", "question", "teacher", "log", "bundle", "folder", "options",
                  "name"}


def decision(name: str = "decision", options: Any = None, llm: str | Callable[[str], Any] | None = None, *,
             api_key: Any = None, api_key_env: str | None = None, base_url: str | None = None,
             bundle: str | None = None, log: str | None = None, folder: str | None = None, **kw) -> Shadow:
    """One call from nothing to a certified cascade.

        intent = shad0w.decision("intent", options={"refund": "wants money back", ...},
                                 llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])
        intent("my card was stolen")     # your LLM answers and is logged; later the table answers
        intent.train()                   # when ~1,000 answers are logged

    name         what this decision is called (default "decision"). It names the folder (shad0w/<name>/), the field
                 in the log, the question inside the bundle, the dashboard and metrics label, and the question name on
                 the Decisions API, so several decisions can live side by side: decision("intent"), decision("urgent").
    options      list of option names, {name: description}, an Enum class, Literal["a", "b"], bool (yes/no), or a
                 schema entry; optional when the bundle exists
    llm          "provider/model" (openai, anthropic, gemini, groq, ollama, openai-decisions, systemone, ...; see
                 shad0w.llm.PROVIDERS) or any callable(text) -> answer. Default: shad0w.configure(llm=...), then
                 SHAD0W_LLM / `llm` in shad0w.toml.
    api_key      the key for that LLM: a string, or a zero-argument function called on every request (rotating or
                 vault keys). It is never printed, logged or pickled ('sk-…3f9a' at most).
    api_key_env  the NAME of the environment variable holding the key (e.g. "OPENAI_API_KEY"). Key precedence:
                 api_key > api_key_env > shad0w.configure(...) > SHAD0W_API_KEY_ENV / shad0w.toml > the provider's
                 usual variable.
    base_url     an OpenAI-compatible server (default: configure(), SHAD0W_BASE_URL / shad0w.toml, the provider's)
    bundle, log  where the trained table and the log live (default: <folder>/bundle and <folder>/log.jsonl)
    folder       default: <settings.folder>/<name>/
    Other keywords go to `Shadow` (audit_rate, auto_train, mode, canary, on_decision, trace, fallback, alpha, ...) and,
    for a "provider/model" string, to `LLMTeacher` (client, complete, system, temperature, timeout, retries, headers).
    api_key, base_url and the other LLM keywords with a function llm raise TypeError: the function holds its own."""
    from .llm import normalize_options
    known = set(_config.DEFAULTS) | set(_TEACHER_KW) | _SHADOW_PARAMS
    unknown = set(kw) - known
    if unknown:
        raise _config.unknown_keywords(unknown, known, "decision()")
    if "settings" in kw:
        raise TypeError("decision() resolves its own settings; pass them as keywords (alpha=..., mode=...)")
    settings = _config.resolve(name, path=kw.get("config"), **{k: v for k, v in kw.items()
                                                              if k in _config.DEFAULTS and k not in _TEACHER_KW})
    kw = {k: v for k, v in kw.items() if k not in _config.DEFAULTS or k in _TEACHER_KW}
    if llm is None and settings.llm:
        llm = settings.llm
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
    if isinstance(llm, str) and schema is None:
        raise ValueError("pass options=... so the LLM knows what to choose from")
    return Shadow(bundle, question=name, log=log_path, schema=schema, settings=settings, llm=llm, api_key=api_key,
                  api_key_env=api_key_env, base_url=base_url, **kw)


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
    Or let shad0w call the LLM for you; then the body is never run:

        @shad0w.decide(llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])
        def route(text) -> Literal["billing", "tech", "sales"]: ...
    """
    llm = kw.pop("llm", None)

    def wrap(f: Callable):
        from .llm import _is_enum, enum_member, enum_option
        hints = typing.get_type_hints(f)
        ret = hints.get("return")
        if ret is bool:
            options = {"type": "yesno"}
        elif typing.get_origin(ret) is typing.Literal:
            options = [str(v) for v in typing.get_args(ret)]
        elif _is_enum(ret):
            options = ret
        else:
            raise TypeError(f"{f.__name__}: annotate the return type with Literal[...], an Enum (the options) or bool")
        body = llm if llm is not None else f
        if _is_enum(ret):
            teacher = body if isinstance(body, str) else functools.wraps(body)(lambda text: enum_option(body(text)))
            sh = decision(name or f.__name__, options=options, llm=teacher, folder=folder, **kw)
            return _wrapped(f, sh, lambda a: enum_member(ret, a))
        sh = decision(name or f.__name__, options=options, llm=body, folder=folder, **kw)
        return _wrapped(f, sh)
    return wrap(fn) if fn is not None else wrap


def _wrapped(fn, sh: Shadow, convert=None):
    @functools.wraps(fn)
    def call(text: str):
        a = sh.decide(text).answer
        return convert(a) if convert is not None else a

    call.shadow = sh  # type: ignore[attr-defined]
    return call


def _add_why(e: Exception, flag) -> None:
    """Say why the LLM was asked in a TeacherError, so a failure reads as a sentence."""
    from .llm import TeacherError
    if isinstance(e, TeacherError) and e.args and "why your LLM was asked" not in str(e.args[0]):
        e.args = (f"{e.args[0]} (why your LLM was asked: {explain(flag)})", *e.args[1:])


def _read_rows(src) -> list[dict]:
    """Rows from a list of dicts or a .jsonl / .json / .csv file."""
    if isinstance(src, (str, os.PathLike)):
        p = os.fspath(src)
        if p.lower().endswith(".csv"):
            import csv
            with open(p, encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        with open(p, encoding="utf-8") as f:
            if p.lower().endswith(".json"):
                data = json.load(f)
                return data if isinstance(data, list) else data.get("rows", [])
            return [json.loads(line) for line in f if line.strip()]
    return [dict(r) for r in src]


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
