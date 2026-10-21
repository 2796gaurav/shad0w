"""Serve the Doodle Duel page with a live model lane: the table guesses in the browser on every stroke, and every
2 s the drawing also goes to a vision model through this server (which holds the key, never the page).

    export OPENAI_API_KEY=...                  # any OpenAI-compatible vision model
    export OPENAI_BASE_URL=https://...         # optional: another provider's OpenAI-compatible endpoint
    python examples/doodle_live.py --page doodle-duel.html --labels cat,house,tree,car,sun,fish,star,apple,bicycle,umbrella,clock,airplane

Then open http://127.0.0.1:8765. Only 127.0.0.1 is bound: the server spends your API budget for whoever reaches it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_client(model: str | None):
    from openai import OpenAI  # reads OPENAI_API_KEY and, if set, OPENAI_BASE_URL (any OpenAI-compatible server)
    return OpenAI(), model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True, help="the built Doodle Duel HTML file")
    ap.add_argument("--labels", required=True, help="comma-separated words the model chooses from (the table's options)")
    ap.add_argument("--model", help="vision model name (default: $OPENAI_MODEL or gpt-4.1-mini)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    labels = [w.strip() for w in a.labels.split(",") if w.strip()]
    client, model = make_client(a.model)
    page = open(a.page, "rb").read()
    prompt = ("This is a quick doodle from a drawing game (like Pictionary). Which ONE of these words is it: "
              + ", ".join(labels) + ". Reply with exactly one word from the list.")

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/ask":
                return self._send(200, json.dumps({"ok": True, "model": model}).encode())
            self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, b"{}")
            n = int(self.headers.get("content-length") or 0)
            if n > 2_000_000:
                return self._send(413, b'{"error": "image too large"}')
            try:
                url = json.loads(self.rfile.read(n))["png"]
                if not str(url).startswith("data:image/png;base64,"):
                    raise ValueError("expected a PNG data URL")
                t = time.time()
                r = client.chat.completions.create(model=model, temperature=0, max_tokens=4, messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url, "detail": "low"}}]}])
                ans = (r.choices[0].message.content or "").strip().lower().strip(".")
                self._send(200, json.dumps({"answer": ans if ans in labels else None, "raw": ans,
                                            "model_ms": round((time.time() - t) * 1000)}).encode())
            except Exception as e:  # the page shows "?"; the error class is enough to debug, never the key
                self._send(502, json.dumps({"error": type(e).__name__}).encode())

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"Doodle Duel on http://127.0.0.1:{a.port} (model: {model})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
