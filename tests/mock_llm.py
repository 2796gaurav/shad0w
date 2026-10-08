"""A tiny OpenAI-compatible server for offline tests. It answers like a keyword classifier.

Besides chat completions it speaks the two decision-model formats (POST /v1/decisions in OpenAI's shape and
POST /v1/systemone in the System One shape) and answers tool-shaped chat requests with tool_calls."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from shad0w import wire

from .test_shadow import VOCAB


def classify(text: str, options=None) -> str:
    for k, phrases in VOCAB.items():
        if any(p in text for p in phrases) and (options is None or k in options):
            return k
    if options is not None and "balance" not in options:
        return list(options)[-1]
    return "balance"


def _spam(text: str) -> bool:
    return "free money" in text or "click here" in text


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
                mock.requests.append({"body": req, "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "path": self.path})
                if mock.fail_next:
                    mock.fail_next = False
                    return self._send(500, {"error": {"message": "overloaded"}})
                dialect = wire.dialect_of(self.path)
                if dialect:
                    return self._decisions(dialect, req)
                if "response_format" in req and mock.mode == "plain":
                    return self._send(400, {"error": {"message": "response_format json_schema is not supported"}})
                text = [m for m in req["messages"] if m["role"] == "user"][-1]["content"]
                if "boom" in text:
                    return self._send(401, {"error": {"message": "bad key"}})
                sysm = next((m["content"] for m in req["messages"] if m["role"] == "system"), "")
                if "spam" in sysm.lower() or "spam" in str(req.get("response_format", "")):
                    ans = "yes" if _spam(text) else "no"
                else:
                    ans = classify(text)
                tools = req.get("tools")
                if tools:
                    fn = tools[0]["function"]
                    param = next(iter(fn["parameters"]["properties"]))
                    call = {"id": "call_mock1", "type": "function", "function": {"name": fn["name"], "arguments": json.dumps({param: ans})}}
                    if req.get("stream"):
                        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, **call}]}}]},
                                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}]
                        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                        return self._send(200, body.encode(), "text/event-stream")
                    return self._send(200, {"id": "x", "object": "chat.completion", "model": req.get("model"),
                                            "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                                                         "finish_reason": "tool_calls"}]})
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

            def _decisions(self, dialect, req):
                try:
                    text, qs = wire.parse(dialect, req)
                except ValueError as e:
                    return self._send(400, {"error": {"message": str(e)}})
                if "boom" in text:
                    return self._send(401, {"error": {"message": "bad key"}})
                answers = [] if dialect == "decisions" else {}
                for q in qs:
                    if q.qtype == "choice":
                        opts = q.options
                        ans = classify(text, opts)
                        probs = {o: (0.9 if o == ans else 0.1 / max(1, len(opts) - 1)) for o in opts}
                        if dialect == "decisions":
                            a = {"type": "choice", "name": q.name, "choice": ans, "confidence": 0.9,
                                 "probabilities": [{"value": o, "probability": p} for o, p in probs.items()]}
                        else:
                            a = {"type": "choice", "choice": ans, "confidence": 0.9, "probabilities": probs}
                    elif q.qtype == "yesno":
                        p = 0.95 if ("urgent" in text or _spam(text)) else 0.05
                        a = ({"type": "predicate", "name": q.name, "probability": p} if dialect == "decisions"
                             else {"type": "noul", "probability": p, "confidence": max(p, 1 - p)})
                    else:
                        levels = q.options or ["low", "high"]
                        if dialect == "decisions":
                            a = {"type": "score", "name": q.name, "score": 0.0, "confidence": 0.8,
                                 "probabilities": [{"label": lv, "value": i, "probability": float(i == 0)} for i, lv in enumerate(levels)]}
                        else:
                            a = {"type": "score", "score": 0.0, "confidence": 0.8}
                    if dialect == "decisions":
                        answers.append(a)
                    else:
                        answers[q.name] = a
                if dialect == "decisions":
                    return self._send(200, {"object": "decision", "model": req.get("model"), "answers": answers,
                                            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}})
                self._send(200, {"model": req.get("model"), "answers": answers, "latency_ms": 1})

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
