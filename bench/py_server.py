"""Serve a Python-only decision model behind the same POST /v1/systemone shape llama.cpp speaks, so the benchmark asks
every decision model the same way. CPU only (CUDA and MPS are hidden), one request at a time.

    python bench/py_server.py --backend decider --model Mapika/decider-0.8b --port 18822 --threads 5
    python bench/py_server.py --backend gliner2 --model fastino/GLiNER2.5-Decide --port 18822 --threads 5

decider  needs `pip install git+https://github.com/Mapika/decider` (Python >= 3.11)
gliner2  needs `pip install "gliner2[local]"`
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def load(backend: str, model: str, threads: int):
    import torch
    torch.set_num_threads(threads)
    if backend == "decider":
        from decider.infer import Decider
        try:
            d = Decider(model, device="cpu")
        except TypeError:
            d = Decider(model)

        def decide(state, q):
            out = d.system_one({"text": state}, {"q": q})
            return out["answers"]["q"] if "answers" in out else out["q"]
        return decide
    if backend == "gliner2":
        from gliner2 import AutoExtractor
        m = AutoExtractor.from_pretrained(model)
        try:
            m = m.to("cpu")
        except Exception:
            pass

        def decide(state, q):
            crit = q["criteria"]
            labels = crit if isinstance(crit, dict) else {c: None for c in crit}
            spec = {"q": {"labels": {k: (v or k) for k, v in labels.items()}, "prompt": q.get("instructions")}}
            res = m.classify_text(state, spec, include_confidence=True)
            r = res.get("q", res)
            if isinstance(r, dict) and "label" in r:
                return {"choice": r["label"], "confidence": float(r.get("confidence", 0.0))}
            if isinstance(r, list) and r and isinstance(r[0], dict):
                top = max(r, key=lambda x: x.get("confidence", 0.0))
                return {"choice": top["label"], "confidence": float(top.get("confidence", 0.0)),
                        "probabilities": {x["label"]: float(x.get("confidence", 0.0)) for x in r}}
            return {"choice": r if isinstance(r, str) else str(r), "confidence": 0.0}
        return decide
    raise SystemExit(f"unknown backend {backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=("decider", "gliner2"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18822)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    decide = load(a.backend, a.model, a.threads)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
                (name, q), = req["questions"].items()
                ans = decide(req["state"], q)
                self._send(200, {"model": a.model, "answers": {name: {"type": "choice", **ans}}})
            except Exception as e:
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"[:300]}})

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
