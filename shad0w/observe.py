"""Watch shad0w work: counters, latency, savings, live audit bound, recent decisions, logs and hooks.

One `Metrics` object can be shared by many `Shadow` wrappers (the proxy does that). It feeds:

    metrics.snapshot()          plain dict: offload, calls saved, p50/p99 per source, flags, audit bound, recent
    metrics.prometheus()        Prometheus text format (served at GET /metrics by `shad0w serve` and `shad0w proxy`)
    GET /                       a live dashboard page (dashboard.html, polls /v1/stats)

Logging: the "shad0w" logger. Set SHAD0W_LOG=debug to print one line per decision, SHAD0W_LOG=info for
training and reload events. Set SHAD0W_OTEL=1 to emit one OpenTelemetry span per decision when the
opentelemetry API is installed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque

log = logging.getLogger("shad0w")

if os.environ.get("SHAD0W_LOG") and not log.handlers:  # opt-in only: a library does not configure logging by default
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s shad0w %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(os.environ["SHAD0W_LOG"].upper())
    log.propagate = False

# seconds; table answers land in the first few buckets, LLM calls in the last ones
BUCKETS = (2.5e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
SOURCES = ("table", "teacher", "deferred")


class _Hist:
    __slots__ = ("counts", "sum", "n", "recent")

    def __init__(self):
        self.counts = [0] * (len(BUCKETS) + 1)
        self.sum, self.n = 0.0, 0
        self.recent = deque(maxlen=2048)  # for percentiles

    def add(self, seconds: float):
        i = 0
        while i < len(BUCKETS) and seconds > BUCKETS[i]:
            i += 1
        self.counts[i] += 1
        self.sum += seconds
        self.n += 1
        self.recent.append(seconds)

    def pct(self, q: float) -> float | None:
        if not self.recent:
            return None
        s = sorted(self.recent)
        return s[min(len(s) - 1, int(q * len(s)))]


class Metrics:
    """Thread-safe decision counters shared by `Shadow`, `shad0w serve` and `shad0w proxy`.

    cost_per_call  what one LLM call costs you (any unit, e.g. dollars); turns calls saved into money saved
    keep_text      keep the input text in the recent-decisions list shown on the dashboard (False to redact)
    """

    def __init__(self, cost_per_call: float | None = None, keep_text: bool = True, recent: int = 50):
        self.cost_per_call, self.keep_text = cost_per_call, keep_text
        self.started = time.time()
        self._lock = threading.Lock()
        self._q: dict[str, dict] = {}
        self._recent = deque(maxlen=recent)
        self.passthrough = 0
        self.errors = 0

    def _question(self, name: str) -> dict:
        q = self._q.get(name)
        if q is None:
            q = self._q[name] = {"count": dict.fromkeys(SOURCES, 0), "flags": {}, "answers": {}, "audits": 0,
                                 "audit_disagreements": 0, "lat": {s: _Hist() for s in SOURCES}, "alpha": None}
        return q

    def record(self, question: str, source: str, answer, latency_s: float, confidence: float | None = None,
               flag: str | None = None, text: str | None = None, teacher_s: float | None = None):
        """One decision. `latency_s` is the end-to-end time; `teacher_s` the part spent in the LLM, if any."""
        with self._lock:
            q = self._question(question)
            q["count"][source] = q["count"].get(source, 0) + 1
            if flag:
                q["flags"][flag] = q["flags"].get(flag, 0) + 1
            key = str(answer)
            q["answers"][key] = q["answers"].get(key, 0) + 1
            q["lat"].setdefault(source, _Hist()).add(teacher_s if teacher_s is not None and source == "teacher" else latency_s)
            self._recent.append({"ts": round(time.time(), 3), "question": question, "source": source, "answer": answer,
                                 "confidence": None if confidence is None else round(confidence, 4), "flag": flag,
                                 "latency_us": round(latency_s * 1e6, 1),
                                 "text": (text[:120] if self.keep_text and text is not None else None)})

    def record_audit(self, question: str, disagree: bool):
        with self._lock:
            q = self._question(question)
            q["audits"] += 1
            q["audit_disagreements"] += int(disagree)

    def set_alpha(self, question: str, alpha: float | None):
        with self._lock:
            self._question(question)["alpha"] = alpha

    def snapshot(self, delta: float = 0.1) -> dict:
        """Everything the dashboard and `shad0w watch` show, as plain JSON-able data."""
        from .cascade import _cp_upper
        with self._lock:
            out_q, tot = {}, {"table": 0, "teacher": 0, "deferred": 0, "audits": 0, "audit_disagreements": 0}
            llm_s_sum, llm_n = 0.0, 0
            for name, q in self._q.items():
                c = dict(q["count"])
                served = sum(c.values())
                th, tl = q["lat"]["table"], q["lat"]["teacher"]
                n, k = q["audits"], q["audit_disagreements"]
                out_q[name] = {
                    "decisions": served, **c, "offload": c["table"] / served if served else 0.0,
                    "flags": dict(q["flags"]), "answers": dict(sorted(q["answers"].items(), key=lambda kv: -kv[1])[:20]),
                    "audits": n, "audit_disagreements": k, "audit_disagreement": k / n if n else None,
                    "audit_disagreement_upper": _cp_upper(k, n, delta) if n else None, "alpha": q["alpha"],
                    "table_p50_us": _us(th.pct(0.5)), "table_p99_us": _us(th.pct(0.99)),
                    "llm_p50_ms": _ms(tl.pct(0.5)), "llm_p99_ms": _ms(tl.pct(0.99)),
                }
                for s in tot:
                    tot[s] += c.get(s, 0) if s in c else q.get(s, 0)
                llm_s_sum += tl.sum
                llm_n += tl.n
            recent = list(self._recent)[::-1]
        served = tot["table"] + tot["teacher"] + tot["deferred"]
        mean_llm = llm_s_sum / llm_n if llm_n else None
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "decisions": served, "table": tot["table"], "teacher": tot["teacher"], "deferred": tot["deferred"],
            "offload": tot["table"] / served if served else 0.0,
            "llm_calls_saved": tot["table"],
            "llm_mean_ms": None if mean_llm is None else round(mean_llm * 1e3, 2),
            "time_saved_s": None if mean_llm is None else round(tot["table"] * mean_llm, 3),
            "cost_per_call": self.cost_per_call,
            "cost_saved": None if self.cost_per_call is None else round(tot["table"] * self.cost_per_call, 6),
            "audits": tot["audits"], "audit_disagreements": tot["audit_disagreements"],
            "passthrough": self.passthrough, "errors": self.errors,
            "questions": out_q, "recent": recent,
        }

    def prometheus(self) -> str:
        """Prometheus text exposition format, version 0.0.4."""
        snap = self.snapshot()
        L = []

        def metric(name, kind, help_, rows):
            L.append(f"# HELP {name} {help_}")
            L.append(f"# TYPE {name} {kind}")
            for labels, v in rows:
                if v is None:
                    continue
                lab = ",".join(f'{k}="{_esc(val)}"' for k, val in labels.items())
                L.append(f"{name}{{{lab}}} {_num(v)}" if lab else f"{name} {_num(v)}")

        qs = snap["questions"]
        metric("shad0w_decisions_total", "counter", "Decisions by question and source (table, teacher, deferred).",
               [({"question": q, "source": s}, v[s]) for q, v in qs.items() for s in SOURCES])
        metric("shad0w_flags_total", "counter", "Why the table did not answer, by question and flag.",
               [({"question": q, "flag": f}, n) for q, v in qs.items() for f, n in v["flags"].items()])
        metric("shad0w_offload_ratio", "gauge", "Share of decisions answered by the certified table.",
               [({"question": q}, v["offload"]) for q, v in qs.items()])
        metric("shad0w_audits_total", "counter", "Certified answers spot-checked against the LLM.",
               [({"question": q}, v["audits"]) for q, v in qs.items()])
        metric("shad0w_audit_disagreements_total", "counter", "Spot checks where the LLM disagreed with the table.",
               [({"question": q}, v["audit_disagreements"]) for q, v in qs.items()])
        metric("shad0w_audit_disagreement_upper", "gauge", "90% upper bound on live disagreement among served answers.",
               [({"question": q}, v["audit_disagreement_upper"]) for q, v in qs.items()])
        metric("shad0w_alpha", "gauge", "Certified disagreement bound of the loaded bundle.",
               [({"question": q}, v["alpha"]) for q, v in qs.items()])
        metric("shad0w_llm_calls_saved_total", "counter", "LLM calls answered by the table instead.", [({}, snap["llm_calls_saved"])])
        metric("shad0w_time_saved_seconds_total", "counter", "Estimated LLM time saved (calls saved x mean LLM latency).",
               [({}, snap["time_saved_s"])])
        if snap["cost_saved"] is not None:
            metric("shad0w_cost_saved_total", "counter", "Estimated LLM cost saved (calls saved x cost per call).", [({}, snap["cost_saved"])])
        metric("shad0w_passthrough_total", "counter", "Proxy requests forwarded untouched (not decisions).", [({}, snap["passthrough"])])
        metric("shad0w_errors_total", "counter", "Requests that failed.", [({}, snap["errors"])])
        L.append("# HELP shad0w_latency_seconds End-to-end decision latency (LLM time for teacher answers).")
        L.append("# TYPE shad0w_latency_seconds histogram")
        with self._lock:
            for q, st in self._q.items():
                for s, h in st["lat"].items():
                    if not h.n:
                        continue
                    acc = 0
                    for b, c in zip(BUCKETS, h.counts):
                        acc += c
                        L.append(f'shad0w_latency_seconds_bucket{{question="{_esc(q)}",source="{s}",le="{b:g}"}} {acc}')
                    L.append(f'shad0w_latency_seconds_bucket{{question="{_esc(q)}",source="{s}",le="+Inf"}} {h.n}')
                    L.append(f'shad0w_latency_seconds_sum{{question="{_esc(q)}",source="{s}"}} {_num(h.sum)}')
                    L.append(f'shad0w_latency_seconds_count{{question="{_esc(q)}",source="{s}"}} {h.n}')
        return "\n".join(L) + "\n"


class TraceWriter:
    """Append every decision to a JSON Lines file. Kept apart from the teacher log on purpose:
    table answers must never become training data for the next compile."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, row: dict):
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            self._f.write(line + "\n")

    def close(self):
        with self._lock:
            self._f.close()


def otel_enabled() -> bool:
    return os.environ.get("SHAD0W_OTEL", "").lower() in ("1", "true", "yes")


_tracer = None


def emit_span(question: str, d, start_ns: int, end_ns: int, model: str | None = None):
    """One OpenTelemetry span per decision, when SHAD0W_OTEL=1 and opentelemetry-api is importable."""
    global _tracer
    try:
        if _tracer is None:
            from opentelemetry import trace
            _tracer = trace.get_tracer("shad0w")
        span = _tracer.start_span("shad0w.decide", start_time=start_ns)
        span.set_attribute("shad0w.question", question)
        span.set_attribute("shad0w.source", d.source)
        span.set_attribute("shad0w.certified", d.certified)
        span.set_attribute("shad0w.answer", str(d.answer))
        if d.confidence is not None:
            span.set_attribute("shad0w.confidence", float(d.confidence))
        if d.flag:
            span.set_attribute("shad0w.flag", d.flag)
        if model:
            span.set_attribute("gen_ai.request.model", model)
        span.end(end_time=end_ns)
    except Exception:  # tracing must never break a decision
        pass


def _us(s):
    return None if s is None else round(s * 1e6, 2)


def _ms(s):
    return None if s is None else round(s * 1e3, 2)


def _num(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "NaN" if math.isnan(v) else ("+Inf" if v > 0 else "-Inf")
    return repr(v) if isinstance(v, float) else str(v)


def _esc(s) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def dashboard_html() -> bytes:
    from importlib.resources import files
    return files("shad0w").joinpath("dashboard.html").read_bytes()
