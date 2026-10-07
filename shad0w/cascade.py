"""Put shad0w in front of the model you already call, in one line.

    import shad0w

    intent = shad0w.decision("intent", options={"refund": "wants money back", "lost_card": "card lost or stolen"},
                             llm="openai/gpt-4o-mini")
    intent("my card was stolen yesterday")    # Decision(answer='lost_card', source='teacher', ...)  logged
    intent.train()                            # once ~1,000 answers are logged: compile + certify, hot-swap
    intent("my card was stolen yesterday")    # Decision(answer='lost_card', source='table', latency_us=9, ...)

Or wrap any function you already have: `shad0w.Shadow("bundle/", teacher=ask_llm, log="teacher_log.jsonl")`.

Day one, before any bundle exists, every call goes to your model and is logged. After `train()` (or the
`shad0w train` command), certified answers come from the table and the rest still go to your model, logged,
so the next training run has more data.

A small random share of certified answers (`audit_rate`, default 1%) is also sent to your model. These spot
checks are logged with `"source": "audit"` and summarised by `stats()`, so you can watch live disagreement
with your model, not only the calibration-time certificate.
"""
from __future__ import annotations

import asyncio
import inspect
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
from .observe import Metrics, TraceWriter, emit_span, log, otel_enabled

__all__ = ["Shadow", "Decision", "cascade", "decision"]


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


_FLAG_WORDS = {
    None: "certified: answered by the table",
    "no_bundle": "no trained table yet: asked your model",
    "low_confidence": "table not sure enough to stay inside the certified bound: asked your model",
    "low_radius": "input could be flipped by a few edits (exposed mode): asked your model",
    "drift": "traffic looks different from calibration: asked your model until re-certified",
    "uncalibrated": "table has no certificate: asked your model",
}


def explain(flag: str | None) -> str:
    """The reason behind a decision, in plain words."""
    return _FLAG_WORDS.get(flag, flag or "")


class Shadow:
    """A certified cascade around one question: the table when it is certified, your model otherwise.

    bundle       path to a compiled bundle, a loaded `Model`, or None (log-only: every call goes to the teacher).
                 A path that does not exist yet is fine: the wrapper starts log-only and `train()` creates it.
    teacher      callable(text) -> answer, your existing model call (or `shad0w.llm_teacher(...)`)
    question     which question of the bundle to answer (default: the bundle's only question, or "decision")
    log          JSON Lines file that receives every teacher answer in the format `shad0w train` reads
    audit_rate   share of certified answers also sent to the teacher as spot checks (0 disables)
    exposed      treat inputs as adversarial: also defer answers with a small robustness radius
    schema       the question's options ({"type": "choice", "criteria": {...}}); taken from the teacher or the
                 bundle when omitted, else learned from the answers in the log at training time
    on_decision  callable(decision, text) run after every decision (send it to your tracing / analytics)
    trace        JSON Lines file that receives EVERY decision (table and teacher), for debugging and dashboards
    metrics      a shared `shad0w.observe.Metrics` (default: a private one; see `stats()` and `metrics.prometheus()`)
    auto_train   retrain in the background every N new teacher answers (needs `pip install "shad0wllm[compile]"`)
    alpha        certified disagreement bound used by `train()` (default 0.05)
    """

    def __init__(self, bundle: str | Model | None, teacher: Callable[[str], Any] | None, question: str | None = None,
                 log: str | None = None, audit_rate: float = 0.01, exposed: bool = False, seed: int | None = None,
                 schema: dict | None = None, on_decision: Callable[[Decision, str], Any] | None = None,
                 trace: str | None = None, metrics: Metrics | None = None, auto_train: int | None = None,
                 alpha: float = 0.05):
        if teacher is not None and not callable(teacher):
            raise TypeError("teacher must be a callable: text -> answer")
        if not 0.0 <= audit_rate <= 1.0:
            raise ValueError("audit_rate must be in [0, 1]")
        self.teacher, self.log, self.audit_rate, self.exposed = teacher, log, audit_rate, exposed
        self.on_decision, self.alpha, self.auto_train = on_decision, alpha, auto_train
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._train_lock = threading.Lock()
        self._counts = {"table": 0, "teacher": 0, "audits": 0, "audit_disagreements": 0}
        self._since_train = 0
        self._otel = otel_enabled()
        self.metrics = metrics or Metrics()
        self.trace = TraceWriter(trace) if trace else None
        self.model: Model | None = None
        self.question = question if question is not None else getattr(teacher, "question", None)
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
            self.metrics.set_alpha(self.question, q.alpha)
            if self.schema is None:
                self.schema = {"type": q.qtype, "criteria": {o: None for o in q.options}} if q.qtype == "choice" else {"type": "yesno"}
            log.info("loaded bundle for %r: %d options, alpha=%s, C core=%s", self.question, len(q.options), q.alpha,
                     model.native)
        with self._lock:
            self.model = model
        return self

    # -- deciding ---------------------------------------------------------------------------------------------
    def decide(self, text: str, teacher: Callable[[str], Any] | None = None) -> Decision:
        """The table's answer when certified, otherwise your model's (logged). `teacher` overrides the
        wrapper's teacher for this one call."""
        t0 = time.perf_counter()
        ns0 = time.time_ns() if self._otel else 0
        name = self.question or "decision"
        teacher = teacher or self.teacher
        model = self.model
        if model is None:
            answer, ts = self._ask(teacher, text, "teacher", name)
            d = Decision(answer, "teacher", None, False, "no_bundle", _us(t0))
            return self._done(d, text, name, ts, ns0)
        r = model.decide(text, exposed=self.exposed, questions={name: {}}, probabilities=False)["answers"][name]
        local = r["choice"] if "choice" in r else r["answer"]
        if r["certified"]:
            d = Decision(local, "table", r["confidence"], True, None, _us(t0))
            if self.audit_rate and teacher is not None and self._rng.random() < self.audit_rate:
                th = threading.Thread(target=self._audit, args=(teacher, text, name, local), daemon=True, name="shad0w-audit")
                with self._lock:
                    self._audits.add(th)
                th.start()
            return self._done(d, text, name, None, ns0)
        answer, ts = self._ask(teacher, text, "teacher", name)
        d = Decision(answer, "teacher", r["confidence"], False, r["flag"], _us(t0))
        return self._done(d, text, name, ts, ns0)

    __call__ = decide

    def flush(self, timeout: float | None = None) -> None:
        """Wait for spot checks still in flight (they run in the background so they never slow an answer)."""
        with self._lock:
            pending = list(self._audits)
        for th in pending:
            th.join(timeout)

    async def adecide(self, text: str, teacher: Callable[[str], Any] | None = None) -> Decision:
        """`decide` for async code. The table path never leaves the event loop; an async teacher is awaited,
        a sync teacher runs in a worker thread."""
        teacher = teacher or self.teacher
        if not _is_async(teacher):
            model = self.model
            if model is not None:
                name = self.question or "decision"
                r = model.decide(text, exposed=self.exposed, questions={name: {}}, probabilities=False)["answers"][name]
                if r["certified"]:
                    return self.decide(text, teacher)  # microseconds; a sampled audit runs in a background thread
            return await asyncio.to_thread(self.decide, text, teacher)
        t0 = time.perf_counter()
        name = self.question or "decision"
        model = self.model
        flag, conf = "no_bundle", None
        if model is not None:
            r = model.decide(text, exposed=self.exposed, questions={name: {}}, probabilities=False)["answers"][name]
            local = r["choice"] if "choice" in r else r["answer"]
            if r["certified"]:
                d = Decision(local, "table", r["confidence"], True, None, _us(t0))
                if self.audit_rate and self._rng.random() < self.audit_rate:
                    task = asyncio.ensure_future(self._aaudit(teacher, text, name, local))
                    self._audits.add(task)
                    task.add_done_callback(self._audits.discard)
                return self._done(d, text, name, None, 0)
            flag, conf = r["flag"], r["confidence"]
        t = time.perf_counter()
        answer = await teacher(text)
        took = time.perf_counter() - t
        self._record(text, "teacher", name, answer)
        return self._done(Decision(answer, "teacher", conf, False, flag, _us(t0)), text, name, took, 0)

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
        certified threshold and the reason in plain words."""
        if self.model is None:
            return {"text": text, "certified": False, "flag": "no_bundle", "why": explain("no_bundle")}
        name = self.question or "decision"
        q = self.model.questions[name]
        r = q.decide(text, exposed=self.exposed)
        top = sorted((r.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:3]
        return {"text": text, "answer": r.get("choice", r.get("answer")), "confidence": r["confidence"],
                "threshold": q.threshold, "alpha": q.alpha, "certified": r["certified"], "flag": r["flag"],
                "why": explain(r["flag"]), "top": top}

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
            with self._lock, open(self.log, "a", encoding="utf-8") as f:
                f.write(row + "\n")
            if self.auto_train and source == "teacher":
                self._since_train += 1
                if self._since_train >= self.auto_train and not self._train_lock.locked():
                    self._since_train = 0
                    threading.Thread(target=self._train_quietly, daemon=True, name="shad0w-train").start()

    # -- training ---------------------------------------------------------------------------------------------
    def log_rows(self) -> int:
        if not self.log or not os.path.exists(self.log):
            return 0
        with open(self.log, "rb") as f:
            return sum(1 for line in f if line.strip())

    def train(self, out: str | None = None, alpha: float | None = None, delta: float = 0.1, min_rows: int = 1000) -> dict:
        """Compile and certify a bundle from this wrapper's log, save it and start serving from it.
        Returns the certificate for this question. Needs `pip install "shad0wllm[compile]"`."""
        from .shadow import read_jsonl, shadow_compile, write_certificate
        out = out or self.bundle_path
        if not out:
            raise ValueError("pass out='bundle/' (where to save the trained bundle)")
        if not self.log or not os.path.exists(self.log):
            raise ValueError("nothing to train on: this Shadow has no log file yet")
        name = self.question or "decision"
        with self._train_lock:
            rows, schema = rows_for_training(read_jsonl(self.log), name, self.schema)
            if len(rows) < min_rows:
                raise ValueError(f"{len(rows)} usable answers logged for {name!r}; need {min_rows} "
                                 f"(pass min_rows=... to try with fewer; 100 is the hard minimum)")
            t = time.time()
            a = self.alpha if alpha is None else alpha
            model, cert = shadow_compile({name: schema}, rows, alpha=a, delta=delta,
                                         teacher=getattr(self.teacher, "name", "unspecified"))
            model.save(out)
            write_certificate(out, cert)
            with open(os.path.join(out, "schema.json"), "w", encoding="utf-8") as f:
                json.dump({name: schema}, f, indent=2)
            self.bundle_path, self.schema = out, schema
            self.reload(out)
            q = cert["questions"][name]
            log.info("trained %r on %d answers in %.1fs: certified share %.1f%% at alpha=%s", name, len(rows),
                     time.time() - t, 100 * q["certified_share_on_calibration"], a)
            return q

    def _train_quietly(self):
        try:
            self.train()
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
    return rows, schema


def decision(name: str, options: Any = None, llm: str | Callable[[str], Any] | None = None, *,
             bundle: str | None = None, log: str | None = None, folder: str | None = None, **kw) -> Shadow:
    """One call from nothing to a certified cascade.

        intent = shad0w.decision("intent", options={"refund": "wants money back", ...}, llm="openai/gpt-4o-mini")
        intent("my card was stolen")     # your LLM answers and is logged; later the table answers
        intent.train()                   # when ~1,000 answers are logged

    options  list of option names, {name: description}, or a schema entry; optional when the bundle exists
    llm      "provider/model" (openai, anthropic, gemini, groq, ollama, ... see shad0w.llm.PROVIDERS) or any
             callable(text) -> answer
    folder   where the log and bundle live (default: ./shad0w/<name>/); bundle= and log= override each path
    Other keywords go to `Shadow` (audit_rate, auto_train, on_decision, trace, alpha, ...) and, for a
    "provider/model" string, to `llm_teacher` (base_url, api_key, client, system, temperature, ...)."""
    from .llm import LLMTeacher, normalize_options
    folder = folder or os.path.join("shad0w", name)
    bundle = bundle or os.path.join(folder, "bundle")
    log_path = log or os.path.join(folder, "log.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    llm_kw = {k: kw.pop(k) for k in list(kw) if k in ("base_url", "api_key", "client", "system", "temperature", "timeout",
                                                       "retries", "headers", "structured", "max_tokens")}
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
    return Shadow(bundle, teacher=teacher, question=name, log=log_path, schema=schema, **kw)


def cascade(bundle: str | Model | None, question: str | None = None, log: str | None = None,
            audit_rate: float = 0.01, exposed: bool = False, **kw):
    """Decorator form of `Shadow`: wraps your model call and returns the answer.

        @shad0w.cascade("bundle/", log="teacher_log.jsonl")
        def classify(text): return ask_llm(text)

        classify("block my card")      # answer from the table when certified, from ask_llm otherwise
        classify.shadow.stats()        # offload and live audit disagreement
    """
    def wrap(fn: Callable[[str], Any]):
        sh = Shadow(bundle, teacher=fn, question=question, log=log, audit_rate=audit_rate, exposed=exposed, **kw)

        def call(text: str):
            return sh.decide(text).answer

        call.shadow = sh  # type: ignore[attr-defined]
        call.__name__, call.__doc__, call.__wrapped__ = fn.__name__, fn.__doc__, fn  # type: ignore[attr-defined]
        return call
    return wrap


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
