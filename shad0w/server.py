"""Reference HTTP server, stdlib only.

  POST /v1/decide   {"state": "...", "exposed": false, "questions": {...}?}  ->  {"answers": {...}, "latency_us": ...}
  GET  /v1/health
  GET  /v1/stats      JSON counters (offload, flags, latency)      GET /metrics  Prometheus      GET /  dashboard

`/v1/systemone` is accepted as an alias, so clients of open decision models that speak that wire format work as-is.
No authentication: keep it on localhost or behind your own front end.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import load
from .observe import Metrics, dashboard_html

MAX_BODY = 1 << 20  # 1 MiB


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
            if self.path not in ("/v1/decide", "/v1/systemone"):
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("content-length", 0))
                if n > MAX_BODY:
                    return self._send(413, {"error": f"request body over {MAX_BODY} bytes"})
                req = json.loads(self.rfile.read(n))
                t = time.perf_counter_ns()
                out = model.decide(req["state"], exposed=bool(req.get("exposed", False)), questions=req.get("questions"))
                out["latency_us"] = (time.perf_counter_ns() - t) / 1e3
                for n, a in out["answers"].items():
                    metrics.record(n, "table" if a["certified"] else "deferred", a.get("choice", a.get("answer")),
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
    print(f"shad0w serving {bundle} on http://{host}:{port}/v1/decide   dashboard http://{host}:{port}/", flush=True)
    srv.serve_forever()
