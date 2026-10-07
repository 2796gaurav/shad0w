"""Reference HTTP server, stdlib only.

  POST /v1/decide   {"state": "...", "exposed": false, "questions": {...}?}  ->  {"answers": {...}, "latency_us": ...}
  GET  /v1/health

`/v1/systemone` is accepted as an alias, so clients of open decision models that speak that wire format work as-is.
No authentication: keep it on localhost or behind your own front end.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import load

MAX_BODY = 1 << 20  # 1 MiB


def serve(bundle: str, host: str = "127.0.0.1", port: int = 8010):
    model = load(bundle)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: one TCP connection per client, not per request

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/health":
                return self._send(200, {"ok": True, "questions": list(model.questions)})
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
                self._send(200, out)
            except Exception as e:  # malformed request
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    print(f"shad0w serving {bundle} on http://{host}:{port}/v1/decide")
    class S(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 1024  # listen backlog; the default of 5 drops SYNs under concurrency (1 s / 3 s retransmits)

    S((host, port), H).serve_forever()
