"""A tiny OpenAI-compatible chat server for offline tests. It answers like a keyword classifier."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .test_shadow import VOCAB


def classify(text: str) -> str:
    for k, phrases in VOCAB.items():
        if any(p in text for p in phrases):
            return k
    return "balance"


class MockLLM:
    """modes: structured (json_schema ok), plain (400 on response_format), flaky (one 500 first), chatty (prose)."""

    def __init__(self, mode="structured"):
        self.mode, self.requests, self.fail_next = mode, [], mode == "flaky"
        mock = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj, ctype="application/json"):
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    return self._send(200, {"object": "list", "data": [{"id": "mock-1", "object": "model"}]})
                self._send(404, {"error": "nf"})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["content-length"])))
                mock.requests.append({"body": req, "headers": {k.lower(): v for k, v in self.headers.items()}})
                if mock.fail_next:
                    mock.fail_next = False
                    return self._send(500, {"error": {"message": "overloaded"}})
                if "response_format" in req and mock.mode == "plain":
                    return self._send(400, {"error": {"message": "response_format json_schema is not supported"}})
                text = [m for m in req["messages"] if m["role"] == "user"][-1]["content"]
                if "boom" in text:
                    return self._send(401, {"error": {"message": "bad key"}})
                ans = classify(text)
                if "response_format" in req:
                    content = json.dumps({"answer": ans})
                elif mock.mode == "chatty":
                    content = f"Sure! The answer is {ans}."
                else:
                    content = ans
                if req.get("stream"):
                    chunks = [{"choices": [{"index": 0, "delta": {"content": c}}]} for c in (content[:3], content[3:])]
                    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    return self._send(200, body.encode(), "text/event-stream")
                self._send(200, {"id": "x", "object": "chat.completion", "model": req.get("model"),
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                              "finish_reason": "stop"}]})

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
