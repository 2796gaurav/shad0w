"""Reference HTTP server, stdlib only.

  POST /v1/decide     {"state": "...", "exposed": false, "questions": {...}?}  ->  {"answers": {...}, "latency_us": ...}
  POST /v1/decisions  OpenAI Decisions API shape (input + questions[]) -> answers[] with probabilities and confidence
  POST /v1/systemone  System One shape (state + questions{}) as spoken by Jev / Kev / Laya -> answers{} + latency_ms
  POST /v1/playground {"question": "intent", "text": "..."}  -> what the table thinks and why (no model call)
  GET  /v1/health
  GET  /v1/stats      JSON counters (offload, flags, latency)      GET /metrics  Prometheus      GET /  dashboard

Questions a bundle does not know (and score questions) come back with shad0w.flag = "unsupported".
No authentication: keep it on localhost or behind your own front end.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import wire
from .api import Model, load
from .cascade import explain_model
from .observe import Metrics, dashboard_html

MAX_BODY = 1 << 20  # 1 MiB


def decision_answers(model: Model, dialect: str, text: str, qs: list[wire.WireQ], exposed: bool = False):
    """Answer decision-format questions from a loaded model. Returns (answers list|dict, flat results by name)."""
    known = {q.name: q for q in qs if q.qtype != "score" and q.name in model.questions}
    narrowed = {}
    for name, q in known.items():
        mq = model.questions[name]
        narrowed[name] = {"criteria": [o for o in q.options if o in mq.options]} if q.qtype == "choice" else {}
    out = model.decide(text, exposed=exposed, questions=narrowed, probabilities=True)["answers"] if known else {}
    answers = [] if dialect == "decisions" else {}
    for q in qs:
        a = out.get(q.name)
        if a is None:
            ans = {"type": {"yesno": "predicate" if dialect == "decisions" else "noul"}.get(q.qtype, q.qtype),
                   "shad0w": {"source": "none", "certified": False, "flag": "unsupported"}}
            if dialect == "decisions":
                ans["name"] = q.name
        elif q.qtype == "yesno":
            ans = wire.format_answer(dialect, q, yes=a["answer"], probability=a["probability"], confidence=a["confidence"],
                                     certified=a["certified"], flag=a["flag"])
        else:
            ans = wire.format_answer(dialect, q, choice=a["choice"], probabilities=a.get("probabilities"),
                                     confidence=a["confidence"], certified=a["certified"], flag=a["flag"])
        if dialect == "decisions":
            answers.append(ans)
        else:
            answers[q.name] = ans
    return answers, out


def serve(bundle: str, host: str = "127.0.0.1", port: int = 8010, metrics: Metrics | None = None, run: bool = True):
    model = load(bundle)
    metrics = metrics or Metrics()
    for n, q in model.questions.items():
        metrics.set_alpha(n, q.alpha)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: one TCP connection per client, not per request
        disable_nagle_algorithm = True  # headers and body go out as separate writes: without this, +40 ms per reply

        def _send(self, code, obj, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p == "/v1/health":
                return self._send(200, {"ok": True, "questions": list(model.questions)})
            if p == "/v1/stats":
                return self._send(200, {"mode": "serve", **metrics.snapshot()})
            if p == "/metrics":
                return self._send(200, metrics.prometheus().encode(), "text/plain; version=0.0.4")
            if p in ("/", "/dashboard"):
                return self._send(200, dashboard_html(), "text/html; charset=utf-8")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?", 1)[0]
            dialect = wire.dialect_of(p)
            if p not in ("/v1/decide", "/v1/playground") and dialect is None:
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("content-length", 0))
                if n < 0:
                    return self._send(400, {"error": "invalid content-length"})
                if n > MAX_BODY:
                    return self._send(413, {"error": f"request body over {MAX_BODY} bytes"})
                req = json.loads(self.rfile.read(n))
                if p == "/v1/playground":
                    name = req.get("question") or (next(iter(model.questions)) if len(model.questions) == 1 else None)
                    if name not in model.questions:
                        return self._send(400, {"error": f"question must be one of {list(model.questions)}"})
                    return self._send(200, {"explain": explain_model(model, name, str(req.get("text", "")),
                                                                     bool(req.get("exposed", False))), "llm": None})
                t = time.perf_counter_ns()
                if dialect:
                    text, qs = wire.parse(dialect, req)
                    answers, flat = decision_answers(model, dialect, text, qs, bool(req.get("exposed", False)))
                    us = (time.perf_counter_ns() - t) / 1e3
                    for qn, a in flat.items():
                        metrics.record(qn, "table" if a["certified"] else "deferred", a.get("choice", a.get("answer")),
                                       us / 1e6, a["confidence"], a["flag"], text)
                    return self._send(200, wire.format_response(dialect, req, answers, us, {"source": "table"}))
                if not isinstance(req, dict) or "state" not in req:
                    return self._send(400, {"error": 'body needs "state": the text (or JSON) to decide on, '
                                                     'e.g. {"state": "my card was stolen"}'})
                out = model.decide(req["state"], exposed=bool(req.get("exposed", False)), questions=req.get("questions"))
                out["latency_us"] = (time.perf_counter_ns() - t) / 1e3
                for qn, a in out["answers"].items():
                    metrics.record(qn, "table" if a["certified"] else "deferred", a.get("choice", a.get("answer")),
                                   out["latency_us"] / 1e6, a["confidence"], a["flag"], req["state"] if isinstance(req["state"], str) else None)
                self._send(200, out)
            except Exception as e:  # malformed request
                metrics.errors += 1
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    class S(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 1024  # listen backlog; the default of 5 drops SYNs under concurrency (1 s / 3 s retransmits)

    srv = S((host, port), H)
    srv.metrics = metrics  # type: ignore[attr-defined]
    if not run:
        return srv
    print(f"shad0w serving {bundle} on http://{host}:{port}/v1/decide  (also /v1/decisions, /v1/systemone)"
          f"   dashboard http://{host}:{port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:  # Ctrl-C: stop quietly, like `shad0w proxy`
        print("\nshad0w serve stopped", flush=True)
    finally:
        srv.server_close()
