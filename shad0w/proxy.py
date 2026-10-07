"""An OpenAI-compatible gateway: point your app's base_url here and nothing else changes.

    shad0w proxy --upstream https://api.openai.com/v1            # then, in your app:
    client = OpenAI(base_url="http://localhost:8010/v1")

Requests are forwarded untouched, except the ones you mark as a decision. A request is a decision when
  - its model is "shad0w/<question>" (forwarded upstream as --model), or
  - it carries the header  X-Shad0w-Question: <question>, or
  - its response_format.json_schema.name is <question>.
For a decision, the text is the last user message. If the question has a trained bundle and the table is
certified for this text, the proxy answers itself in microseconds with a normal chat.completion (plus a
"shad0w" field and x-shad0w-* headers). Otherwise the request goes upstream as usual, the reply is returned
untouched, and the answer is logged so the table can learn it.

Everything lives under --dir (default ./shad0w):  <dir>/<question>/log.jsonl  and  <dir>/<question>/bundle/
Train a question with `shad0w train --dir <dir> --question <q>` (the proxy reloads the new bundle on its own),
or start the proxy with --auto-train N.

GET /  live dashboard · GET /v1/stats  JSON · GET /metrics  Prometheus · GET /v1/health
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .cascade import Shadow
from .llm import match_option
from .observe import Metrics, dashboard_html, log

MAX_BODY = 8 << 20  # chat requests can carry long histories
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate", "accept-encoding"}


class Upstream(Exception):
    """The upstream call did not produce a usable decision; carries the response to relay as-is."""

    def __init__(self, status: int, headers: list, body: bytes, streamed: bool = False):
        super().__init__(f"upstream {status}")
        self.status, self.headers, self.body, self.streamed = status, headers, body, streamed


class Gateway:
    """The proxy's state: one `Shadow` per question, shared metrics, upstream settings."""

    def __init__(self, upstream: str, folder: str = "shad0w", model: str | None = None, api_key: str | None = None,
                 audit_rate: float = 0.01, auto_train: int | None = None, alpha: float = 0.05,
                 cost_per_call: float | None = None, keep_text: bool = True, timeout: float = 120.0,
                 schema: dict | None = None, bundles: dict | None = None):
        self.upstream = upstream.rstrip("/")
        self.folder, self.model, self.api_key = folder, model, api_key
        self.audit_rate, self.auto_train, self.alpha, self.timeout = audit_rate, auto_train, alpha, timeout
        self.schema = schema or {}
        self.bundles = dict(bundles or {})
        self.metrics = Metrics(cost_per_call=cost_per_call, keep_text=keep_text)
        self._shadows: dict[str, Shadow] = {}
        self._mtimes: dict[str, float] = {}
        self._checked: dict[str, float] = {}
        self._lock = threading.Lock()
        for q in set(self.bundles) | set(self.schema) | set(_existing_questions(folder)):
            self.shadow(q)

    def paths(self, q: str) -> tuple[str, str]:
        d = os.path.join(self.folder, q)
        return self.bundles.get(q, os.path.join(d, "bundle")), os.path.join(d, "log.jsonl")

    def shadow(self, q: str) -> Shadow:
        with self._lock:
            sh = self._shadows.get(q)
            if sh is None:
                if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", q):
                    raise ValueError(f"question name {q!r}: use letters, digits, _ . - (max 64)")
                bundle, log_path = self.paths(q)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                sh = Shadow(bundle, teacher=None, question=q, log=log_path, audit_rate=self.audit_rate,
                            schema=self.schema.get(q), metrics=self.metrics, auto_train=self.auto_train, alpha=self.alpha)
                self._shadows[q] = sh
                self._mtimes[q] = _mtime(bundle)
            return sh

    def maybe_reload(self, q: str, sh: Shadow):
        """Pick up a bundle retrained by `shad0w train` (checked at most once a second)."""
        now = time.monotonic()
        if now - self._checked.get(q, 0) < 1.0:
            return
        self._checked[q] = now
        bundle, _ = self.paths(q)
        m = _mtime(bundle)
        if m and m != self._mtimes.get(q):
            try:
                sh.reload(bundle)
                self._mtimes[q] = m
                log.info("reloaded bundle for %r", q)
            except Exception as e:  # half-written bundle: try again next second
                log.warning("reload of %r failed: %s", q, e)

    def options(self, q: str, sh: Shadow, req: dict) -> list[str] | None:
        if sh.model is not None:
            return list(sh.model.questions[q].options)
        if sh.schema and sh.schema.get("type", "choice") == "choice":
            return list(sh.schema["criteria"])
        return _enum_from_request(req)


def _existing_questions(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
            and (os.path.exists(os.path.join(folder, d, "log.jsonl")) or os.path.isdir(os.path.join(folder, d, "bundle")))]


def _mtime(bundle: str) -> float:
    p = os.path.join(bundle, "manifest.json")
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def _schema_of(req: dict) -> dict | None:
    rf = req.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        return (rf.get("json_schema") or {})
    return None


def _enum_from_request(req: dict) -> list[str] | None:
    js = _schema_of(req)
    props = ((js or {}).get("schema") or {}).get("properties") or {}
    enums = [p["enum"] for p in props.values() if isinstance(p, dict) and isinstance(p.get("enum"), list)]
    return [str(x) for x in enums[0]] if len(enums) == 1 else None


def _answer_field(req: dict) -> str | None:
    """The JSON property that holds the answer when the client asked for structured output."""
    props = (((_schema_of(req) or {}).get("schema") or {}).get("properties") or {})
    enum_props = [k for k, p in props.items() if isinstance(p, dict) and "enum" in p]
    if len(enum_props) == 1:
        return enum_props[0]
    return next(iter(props)) if len(props) == 1 else None


def question_of(req: dict, headers) -> str | None:
    h = headers.get("x-shad0w-question")
    if h:
        return h.strip()
    m = str(req.get("model", ""))
    if m.startswith("shad0w/"):
        return m.split("/", 1)[1].split("@", 1)[0]
    js = _schema_of(req)
    return js.get("name") if js and js.get("name") else None


def last_user_text(req: dict) -> str | None:
    for msg in reversed(req.get("messages") or []):
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return None


def completion(req: dict, q: str, d, model_name: str) -> dict:
    """A chat.completion carrying the table's answer, shaped like the client asked (text, JSON object, JSON schema)."""
    ans = d.answer
    field = _answer_field(req)
    rf = req.get("response_format") or {}
    if field:
        prop = (((_schema_of(req) or {}).get("schema") or {}).get("properties") or {}).get(field, {})
        val = ans if not isinstance(ans, bool) else (ans if prop.get("type") == "boolean" else ("yes" if ans else "no"))
        content = json.dumps({field: val})
    elif isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        content = json.dumps({q: ans})
    else:
        content = ans if isinstance(ans, str) else ("yes" if ans else "no")
    return {
        "id": "chatcmpl-shad0w-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
        "model": model_name, "system_fingerprint": "shad0w",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": None},
                     "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "shad0w": {"question": q, "source": d.source, "answer": ans, "confidence": d.confidence,
                   "certified": d.certified, "latency_us": round(d.latency_us, 2)},
    }


def make_handler(gw: Gateway):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True  # avoid the 40 ms delayed-ACK stall between header and body writes

        # -- helpers ------------------------------------------------------------------------------------------
        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj, extra=None):
            self._send(code, json.dumps(obj, default=str).encode(), extra=extra)

        def _upstream_url(self):
            path = self.path
            if path.startswith("/v1/"):
                path = path[3:]
            return gw.upstream + path

        def _forward(self, body: bytes | None, relay: bool = True, rewrite_model: dict | None = None):
            """Send this request upstream. relay=True streams the response straight to the client and returns
            (status, collected bytes); relay=False returns (status, headers, bytes) without writing anything."""
            if rewrite_model is not None:
                body = json.dumps(rewrite_model).encode()
            req = urllib.request.Request(self._upstream_url(), data=body, method=self.command)
            for k, v in self.headers.items():
                if k.lower() not in HOP and not k.lower().startswith("x-shad0w"):
                    req.add_header(k, v)
            req.add_header("accept-encoding", "identity")
            if gw.api_key:
                req.add_header("authorization", f"Bearer {gw.api_key}")
            try:
                resp = urllib.request.urlopen(req, timeout=gw.timeout)
                status = resp.status
            except urllib.error.HTTPError as e:
                resp, status = e, e.code
            except (urllib.error.URLError, OSError) as e:
                gw.metrics.errors += 1
                msg = {"error": {"message": f"shad0w proxy: cannot reach upstream {gw.upstream}: {e}", "type": "upstream_unreachable"}}
                if relay:
                    self._json(502, msg)
                    return 502, b""
                raise Upstream(502, [("content-type", "application/json")], json.dumps(msg).encode()) from None
            headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in HOP]
            ctype = resp.headers.get("content-type", "")
            if not relay:
                data = resp.read()
                resp.close()
                return status, headers, data
            if "text/event-stream" in ctype:
                return status, self._relay_stream(resp, status, headers)
            data = resp.read()
            resp.close()
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return status, data

        def _relay_stream(self, resp, status, headers) -> bytes:
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            got = []
            while True:
                line = resp.readline()
                if not line:
                    break
                got.append(line)
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            resp.close()
            return b"".join(got)

        # -- routes -------------------------------------------------------------------------------------------
        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p in ("/", "/dashboard"):
                return self._send(200, dashboard_html(), "text/html; charset=utf-8")
            if p == "/v1/stats":
                return self._json(200, {"mode": "proxy", "upstream": gw.upstream, **gw.metrics.snapshot(),
                                        "trained": {q: sh.model is not None for q, sh in gw._shadows.items()},
                                        "log_rows": {q: sh.log_rows() for q, sh in gw._shadows.items()}})
            if p == "/metrics":
                return self._send(200, gw.metrics.prometheus().encode(), "text/plain; version=0.0.4")
            if p == "/v1/health":
                return self._json(200, {"ok": True, "upstream": gw.upstream, "questions": sorted(gw._shadows)})
            gw.metrics.passthrough += 1
            self._forward(None)

        def do_DELETE(self):
            gw.metrics.passthrough += 1
            self._forward(None)

        def do_POST(self):
            n = int(self.headers.get("content-length", 0) or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": {"message": f"request body over {MAX_BODY} bytes"}})
            body = self.rfile.read(n) if n else b""
            p = self.path.split("?", 1)[0]
            req = None
            if p.endswith("/chat/completions"):
                try:
                    req = json.loads(body or b"{}")
                except ValueError:
                    req = None
            q = question_of(req, self.headers) if isinstance(req, dict) else None
            text = last_user_text(req) if q else None
            if not q or text is None:
                gw.metrics.passthrough += 1
                return self._forward(body)
            try:
                self._decide(q, req, text)
            except Upstream as u:
                if not u.streamed:
                    self.send_response(u.status)
                    for k, v in u.headers:
                        self.send_header(k, v)
                    self.send_header("content-length", str(len(u.body)))
                    self.end_headers()
                    self.wfile.write(u.body)
            except ValueError as e:
                gw.metrics.errors += 1
                self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})

        def _decide(self, q, req, text):
            sh = gw.shadow(q)
            gw.maybe_reload(q, sh)
            upstream_req = dict(req)
            if str(req.get("model", "")).startswith("shad0w/"):
                m = req["model"].split("@", 1)
                upstream_req["model"] = m[1] if len(m) == 2 else gw.model
                if not upstream_req["model"]:
                    raise ValueError("model 'shad0w/<question>' needs `shad0w proxy --model <upstream model>` "
                                     "(or 'shad0w/<question>@<model>')")
            stream = bool(req.get("stream"))
            opts = gw.options(q, sh, req)
            field = _answer_field(req) or "answer"
            sent = {}
            me = threading.current_thread()

            def via_upstream(_text):
                if threading.current_thread() is not me:  # a background spot check: never touches the client
                    status, _, data = self._forward(None, relay=False, rewrite_model={**upstream_req, "stream": False})
                    if status != 200:
                        raise Upstream(status, [], data)
                    return _parse(_content(data), opts, field)
                if stream:
                    status, data = self._forward(None, relay=True, rewrite_model=upstream_req)
                    sent["streamed"] = True
                    if status != 200:
                        raise Upstream(status, [], b"", streamed=True)
                    return _parse("".join(_sse_content(data)), opts, field)
                status, headers, data = self._forward(None, relay=False, rewrite_model=upstream_req)
                sent["resp"] = (status, headers, data)
                if status != 200:
                    raise Upstream(status, headers, data)
                return _parse(_content(data), opts, field)

            d = sh.decide(text, teacher=via_upstream)
            hdr = {"x-shad0w-source": d.source, "x-shad0w-question": q,
                   "x-shad0w-latency-us": f"{d.latency_us:.1f}"}
            if d.source == "table":
                hdr["x-shad0w-confidence"] = f"{d.confidence:.4f}"
                out = completion(req, q, d, req.get("model") or "shad0w")
                if stream:
                    return self._stream_table(out, hdr)
                return self._json(200, out, hdr)
            if stream:
                return  # already relayed chunk by chunk
            status, headers, data = sent["resp"]
            self.send_response(status)
            for k, v in headers:
                self.send_header(k, v)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _stream_table(self, out, hdr):
            msg = out["choices"][0]["message"]["content"]
            base = {k: out[k] for k in ("id", "created", "model", "system_fingerprint")}
            ch = {**base, "object": "chat.completion.chunk"}

            def chunk(delta, finish=None, **extra):
                return {**ch, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

            chunks = [chunk({"role": "assistant", "content": ""}), chunk({"content": msg}),
                      chunk({}, "stop", shad0w=out["shad0w"])]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self._send(200, body.encode(), "text/event-stream", hdr)

        def log_message(self, *a):
            if log.isEnabledFor(10):
                log.debug("http " + (a[0] % a[1:]))

    return H


def _content(data: bytes) -> str:
    try:
        return json.loads(data)["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def _sse_content(data: bytes):
    for line in data.decode("utf-8", "replace").splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                ch = json.loads(line[5:].strip())
                delta = ch["choices"][0].get("delta") or {}
                if delta.get("content"):
                    yield delta["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                continue


def _parse(content: str, options: list[str] | None, field: str):
    if options:
        return match_option(content, options, field)
    s = content.strip()  # options unknown yet: keep short single-line answers, learn the option set from them
    try:
        v = json.loads(s)
        if isinstance(v, dict) and len(v) == 1:
            v = next(iter(v.values()))
        s = v if isinstance(v, str) else s
    except ValueError:
        pass
    s = s.strip().strip('"').strip()
    return s if s and len(s) <= 64 and "\n" not in s else None


def proxy(upstream: str, host: str = "127.0.0.1", port: int = 8010, **kw):
    gw = Gateway(upstream, **kw)

    class S(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 1024

    srv = S((host, port), make_handler(gw))
    port = srv.server_address[1]
    print(f"shad0w proxy on http://{host}:{port}/v1  ->  {gw.upstream}\n"
          f"  dashboard http://{host}:{port}/   metrics http://{host}:{port}/metrics   data {os.path.abspath(gw.folder)}\n"
          "  mark a decision with header 'X-Shad0w-Question: <name>' or model 'shad0w/<name>'", flush=True)
    srv.gateway = gw  # type: ignore[attr-defined]
    return srv
