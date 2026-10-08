"""An OpenAI-compatible gateway: point your app's base_url here and nothing else changes.

    shad0w proxy --upstream https://api.openai.com/v1            # then, in your app:
    client = OpenAI(base_url="http://localhost:8010/v1")

Zero-code paths (the request already names the question and its options):
  - POST /v1/decisions   the OpenAI Decisions API (client.decisions.create(...)): every choice / predicate
                         question becomes a shad0w question; certified ones are answered here, the rest are
                         forwarded (only those) and their answers logged.
  - POST /v1/systemone   the System One format of Jev, Kev, Laya, Ollaya and llama.cpp. Same treatment.

Chat completions are forwarded untouched, except the ones recognised as a decision (`--capture`, default
header, model, tools, decisions):
  - header     X-Shad0w-Question: <question>            (options: the bundle, --schema, the request's json_schema
                                                         enum or tool enum, or X-Shad0w-Options: a,b,c)
  - model      model = "shad0w/<question>[@upstream-model]"
  - tools      a forced tool_choice (or a single tool) whose parameters hold exactly one enum property
  - json_schema  (opt in with --capture json_schema) response_format.json_schema.name is the question
For a decision, the text is the last user message. If the question has a trained bundle and the table is
certified for this text, the proxy answers itself in microseconds with a normal chat.completion (or tool_calls
when the request used tools), plus a "shad0w" field and x-shad0w-* headers. Otherwise the request goes
upstream as usual, the reply is returned untouched, and the answer is logged so the table can learn it.

Everything lives under --dir (default ./shad0w):  <dir>/<question>/log.jsonl  and  <dir>/<question>/bundle/
Train a question with `shad0w train --dir <dir> --question <q>` (the proxy reloads the new bundle on its own),
or start the proxy with --auto-train N. Settings come from flags, SHAD0W_* variables or shad0w.toml, with
[questions.<name>] sections applying per question.

GET /  live dashboard (with a playground) · GET /v1/stats  JSON · GET /metrics  Prometheus · GET /v1/health
POST /v1/playground {"question", "text", "ask_llm": false, "log": false}  what the table thinks (and your LLM)
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

from . import config as _config
from . import wire
from .cascade import Shadow
from .llm import LLMTeacher, match_option
from .observe import Metrics, TraceWriter, dashboard_html, log

MAX_BODY = 8 << 20  # chat requests can carry long histories
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate", "accept-encoding"}
_SHADOW_KEYS = ("alpha", "delta", "min_rows", "cal_fraction", "max_cal", "audit_rate", "auto_train", "retrain", "mode",
                "canary", "never_serve", "min_confidence", "force_threshold", "drift_window", "drift_margin", "exposed")


class Upstream(Exception):
    """The upstream call did not produce a usable decision; carries the response to relay as-is."""

    def __init__(self, status: int, headers: list, body: bytes, streamed: bool = False):
        super().__init__(f"upstream {status}")
        self.status, self.headers, self.body, self.streamed = status, headers, body, streamed


class Gateway:
    """The proxy's state: one `Shadow` per question, shared metrics, upstream settings.

    Keyword settings (alpha, audit_rate, auto_train, mode, canary, never_serve, capture, timeout, trace, ...) follow
    shad0w.config: a flag beats SHAD0W_* variables, which beat shad0w.toml, which beats the defaults."""

    def __init__(self, upstream: str, folder: str | None = None, model: str | None = None, api_key: str | None = None,
                 keep_text: bool = True, schema: dict | None = None, bundles: dict | None = None,
                 config: str | None = None, **settings):
        self.settings = _config.resolve(path=config, **settings)
        self._given = {k: v for k, v in settings.items() if v is not None}
        self._config_path = config
        s = self.settings
        self.upstream = upstream.rstrip("/")
        self.folder = folder or s.folder
        self.model, self.api_key = model, api_key
        self.audit_rate, self.auto_train, self.alpha, self.timeout = s.audit_rate, s.auto_train, s.alpha, s.timeout
        self.capture = set(s.capture)
        self.schema = schema or {}
        self.bundles = dict(bundles or {})
        self.metrics = Metrics(cost_per_call=s.cost_per_call, keep_text=keep_text, llm_latency_ms=s.llm_latency_ms)
        self._trace = TraceWriter(s.trace) if isinstance(s.trace, str) else s.trace
        self._shadows: dict[str, Shadow] = {}
        self._teachers: dict[tuple, LLMTeacher] = {}
        self._mtimes: dict[str, float] = {}
        self._checked: dict[str, float] = {}
        self._lock = threading.Lock()
        for q in set(self.bundles) | set(self.schema) | set(_existing_questions(self.folder)):
            self.shadow(q)

    def paths(self, q: str) -> tuple[str, str]:
        d = os.path.join(self.folder, q)
        return self.bundles.get(q, os.path.join(d, "bundle")), os.path.join(d, "log.jsonl")

    def shadow(self, q: str, schema: dict | None = None) -> Shadow:
        with self._lock:
            sh = self._shadows.get(q)
            if sh is None:
                if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", q):
                    raise ValueError(f"question name {q!r}: use letters, digits, _ . - (max 64)")
                bundle, log_path = self.paths(q)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                qs = _config.resolve(q, path=self._config_path, **self._given)
                sh = Shadow(bundle, teacher=None, question=q, log=log_path, schema=self.schema.get(q) or schema,
                            metrics=self.metrics, trace=self._trace, settings=qs)
                self._shadows[q] = sh
                self._mtimes[q] = _mtime(bundle)
            elif schema is not None and _option_set(sh.schema) != _option_set(schema):
                sh.schema = schema  # the request names the options: it is the source of truth (and flags changes)
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

    def options(self, q: str, sh: Shadow, req: dict, headers=None) -> list[str] | None:
        """The options to parse your LLM's reply against: the request's own (json_schema / tool enum / header) when
        it names them, else the configured schema, else the table's."""
        named = options_of(req, headers)
        if named:
            if _option_set(sh.schema) != set(named):
                sh.schema = {"type": "choice", "criteria": {o: None for o in named}}
            return named
        if sh.schema and sh.schema.get("type", "choice") == "choice" and sh.schema.get("criteria"):
            return list(sh.schema["criteria"])
        if sh.model is not None:
            return list(sh.model.questions[q].options)
        if sh.schema:
            if sh.schema.get("type", "choice") == "choice":
                return list(sh.schema["criteria"])
            return ["yes", "no"]
        return options_of(req, headers)

    def teacher_for(self, q: str, sh: Shadow, dialect: str, model: str | None, api_key: str | None) -> LLMTeacher:
        """A cached decision-model teacher for spot checks on the decisions / systemone paths."""
        key = (q, dialect, model, bool(api_key))
        with self._lock:
            t = self._teachers.get(key)
            if t is None:
                provider = "openai-decisions" if dialect == "decisions" else "systemone"
                t = self._teachers[key] = LLMTeacher({q: sh.schema or {"type": "choice", "criteria": {}}},
                                                     f"{provider}/{model or 'default'}", question=q,
                                                     base_url=self.upstream, api_key=api_key or "", timeout=self.timeout)
            return t


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


def _tool_enum(req: dict) -> tuple[str, str, list[str]] | None:
    """(function name, parameter, enum values) when the request forces one tool whose parameters hold exactly one
    enum property (or offers a single such tool)."""
    tools = req.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    tc = req.get("tool_choice")
    name = None
    if isinstance(tc, dict):
        name = ((tc.get("function") or {}).get("name")) if tc.get("type", "function") == "function" else None
        if name is None:
            return None
    elif tc in (None, "auto", "required") and len(tools) == 1:
        name = ((tools[0].get("function") or {}).get("name"))
    if not name:
        return None
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") != name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        enums = [(k, p["enum"]) for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list)]
        if len(enums) == 1:
            return name, enums[0][0], [str(x) for x in enums[0][1]]
    return None


def _option_set(schema: dict | None):
    if not schema:
        return None
    if schema.get("type", "choice") == "yesno":
        return {"__yesno__"}
    return {str(o) for o in (schema.get("criteria") or {})}


def options_of(req: dict, headers=None) -> list[str] | None:
    """Options named by the request itself: a json_schema enum, a tool enum, or the X-Shad0w-Options header."""
    e = _enum_from_request(req)
    if e:
        return e
    t = _tool_enum(req)
    if t:
        return t[2]
    h = headers.get("x-shad0w-options") if headers is not None else None
    if h:
        opts = [x.strip() for x in h.split(",") if x.strip()]
        return opts if len(opts) >= 2 else None
    return None


def _answer_field(req: dict) -> str | None:
    """The JSON property that holds the answer when the client asked for structured output (or a tool)."""
    props = (((_schema_of(req) or {}).get("schema") or {}).get("properties") or {})
    enum_props = [k for k, p in props.items() if isinstance(p, dict) and "enum" in p]
    if len(enum_props) == 1:
        return enum_props[0]
    if len(props) == 1:
        return next(iter(props))
    t = _tool_enum(req)
    return t[1] if t else None


def question_of(req: dict, headers, capture=("header", "model", "tools", "decisions")) -> str | None:
    h = headers.get("x-shad0w-question")
    if h and "header" in capture:
        return h.strip()
    m = str(req.get("model", ""))
    if m.startswith("shad0w/") and "model" in capture:
        return m.split("/", 1)[1].split("@", 1)[0]
    if "tools" in capture:
        t = _tool_enum(req)
        if t:
            return t[0]
    if "json_schema" in capture:
        js = _schema_of(req)
        return js.get("name") if js and js.get("name") else None
    return None


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
    """A chat.completion carrying the table's answer, shaped like the client asked (text, JSON object, JSON schema,
    or a tool call)."""
    ans = d.answer
    field = _answer_field(req)
    rf = req.get("response_format") or {}
    tool = _tool_enum(req)
    if tool:
        val = ans if not isinstance(ans, bool) else ("yes" if ans else "no")
        message = {"role": "assistant", "content": None, "refusal": None,
                   "tool_calls": [{"id": "call_shad0w_" + uuid.uuid4().hex[:16], "type": "function",
                                   "function": {"name": tool[0], "arguments": json.dumps({tool[1]: val})}}]}
        finish = "tool_calls"
    else:
        if field:
            prop = (((_schema_of(req) or {}).get("schema") or {}).get("properties") or {}).get(field, {})
            val = ans if not isinstance(ans, bool) else (ans if prop.get("type") == "boolean" else ("yes" if ans else "no"))
            content = json.dumps({field: val})
        elif isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
            content = json.dumps({q: ans})
        else:
            content = ans if isinstance(ans, str) else ("yes" if ans else "no")
        message = {"role": "assistant", "content": content, "refusal": None}
        finish = "stop"
    return {
        "id": "chatcmpl-shad0w-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
        "model": model_name, "system_fingerprint": "shad0w",
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "shad0w": {"question": q, "source": d.source, "answer": ans, "confidence": d.confidence,
                   "certified": d.certified, "flag": d.flag, "latency_us": round(d.latency_us, 2)},
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

        def _relay_upstream(self, u: Upstream, extra=None):
            if u.streamed:
                return
            self.send_response(u.status)
            for k, v in u.headers:
                self.send_header(k, v)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(u.body)))
            self.end_headers()
            self.wfile.write(u.body)

        # -- routes -------------------------------------------------------------------------------------------
        def do_GET(self):
            p = self.path.split("?", 1)[0]
            if p in ("/", "/dashboard"):
                return self._send(200, dashboard_html(), "text/html; charset=utf-8")
            if p == "/v1/stats":
                return self._json(200, {"mode": "proxy", "upstream": gw.upstream, **gw.metrics.snapshot(),
                                        "trained": {q: sh.model is not None for q, sh in gw._shadows.items()},
                                        "log_rows": {q: sh.log_rows() for q, sh in gw._shadows.items()},
                                        "settings": {q: {"mode": sh.settings.mode, "canary": sh.settings.canary,
                                                         "alpha": sh.settings.alpha} for q, sh in gw._shadows.items()}})
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
            try:
                if p == "/v1/playground":
                    return self._playground(json.loads(body or b"{}"))
                dialect = wire.dialect_of(p)
                if dialect and "decisions" in gw.capture:
                    try:
                        req = json.loads(body or b"{}")
                    except ValueError:
                        req = None
                    if isinstance(req, dict) and not req.get("stream"):
                        return self._decisions(dialect, req)
                    gw.metrics.passthrough += 1
                    return self._forward(body)
                req = None
                if p.endswith("/chat/completions"):
                    try:
                        req = json.loads(body or b"{}")
                    except ValueError:
                        req = None
                q = question_of(req, self.headers, gw.capture) if isinstance(req, dict) else None
                text = last_user_text(req) if q else None
                if not q or text is None:
                    gw.metrics.passthrough += 1
                    return self._forward(body)
                self._decide(q, req, text)
            except Upstream as u:
                self._relay_upstream(u)
            except ValueError as e:
                gw.metrics.errors += 1
                self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})

        # -- chat completions marked as decisions ---------------------------------------------------------------
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
            opts = gw.options(q, sh, req, self.headers)
            field = _answer_field(req) or "answer"
            yesno = bool(sh.schema) and sh.schema.get("type") == "yesno"
            sent = {}
            me = threading.current_thread()

            def finish(ans):
                if yesno and isinstance(ans, str):
                    return ans.strip().lower() in ("yes", "true", "y", "1")
                return ans

            def via_upstream(_text):
                if threading.current_thread() is not me:  # a background spot check: never touches the client
                    status, _, data = self._forward(None, relay=False, rewrite_model={**upstream_req, "stream": False})
                    if status != 200:
                        raise Upstream(status, [], data)
                    return finish(_parse(_reply_text(data), opts, field))
                if stream:
                    status, data = self._forward(None, relay=True, rewrite_model=upstream_req)
                    sent["streamed"] = True
                    if status != 200:
                        raise Upstream(status, [], b"", streamed=True)
                    return finish(_parse("".join(_sse_content(data)), opts, field))
                status, headers, data = self._forward(None, relay=False, rewrite_model=upstream_req)
                sent["resp"] = (status, headers, data)
                if status != 200:
                    raise Upstream(status, headers, data)
                return finish(_parse(_reply_text(data), opts, field))

            d = sh.decide(text, teacher=via_upstream)
            hdr = {"x-shad0w-source": d.source, "x-shad0w-question": q,
                   "x-shad0w-latency-us": f"{d.latency_us:.1f}"}
            if d.source == "table":
                hdr["x-shad0w-confidence"] = f"{d.confidence:.4f}"
                hdr["x-shad0w-certified"] = "1" if d.certified else "0"
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
            msg = out["choices"][0]["message"]
            base = {k: out[k] for k in ("id", "created", "model", "system_fingerprint")}
            ch = {**base, "object": "chat.completion.chunk"}

            def chunk(delta, finish=None, **extra):
                return {**ch, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

            if msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                chunks = [chunk({"role": "assistant", "content": None}),
                          chunk({"tool_calls": [{"index": 0, "id": tc["id"], "type": "function",
                                                 "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}),
                          chunk({}, "tool_calls", shad0w=out["shad0w"])]
            else:
                chunks = [chunk({"role": "assistant", "content": ""}), chunk({"content": msg["content"]}),
                          chunk({}, "stop", shad0w=out["shad0w"])]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self._send(200, body.encode(), "text/event-stream", hdr)

        # -- decision-model formats: no marking needed -----------------------------------------------------------
        def _decisions(self, dialect, req):
            t0 = time.perf_counter()
            text, qs = wire.parse(dialect, req)
            served: dict[str, object] = {}
            pending: list[wire.WireQ] = []
            for q in qs:
                if q.qtype == "score":
                    pending.append(q)
                    continue
                schema = ({"type": "yesno", **({"instructions": q.instructions} if q.instructions else {})} if q.qtype == "yesno"
                          else {"type": "choice", "criteria": dict(q.criteria), **({"instructions": q.instructions} if q.instructions else {})})
                sh = gw.shadow(q.name, schema=schema)
                gw.maybe_reload(q.name, sh)
                d = sh.peek(text, probabilities=True)
                if d is None:
                    pending.append(q)
                else:
                    served[q.name] = d
                    if d.certified and sh.audit_rate and sh.teacher is None:
                        self._spot_check(dialect, req, sh, q, text, d)
            up: dict[str, dict] = {}
            if pending:
                status, headers, data = self._forward(None, relay=False, rewrite_model=wire.subset(dialect, req, [q.name for q in pending]))
                if status != 200:
                    raise Upstream(status, headers, data)
                try:
                    up = wire.parse_answers(dialect, json.loads(data))
                except ValueError:
                    raise Upstream(status, headers, data) from None
                for q in pending:
                    if q.qtype != "score" and q.name in up:
                        v = wire.answer_value(q, up[q.name])
                        if v is not None:
                            gw.shadow(q.name).record(text, v)
            answers = [] if dialect == "decisions" else {}
            for q in qs:
                if q.name in served:
                    d = served[q.name]
                    probs = dict(d.top or [])
                    if q.qtype == "yesno":
                        p_yes = probs.get("yes", 1.0 if d.answer else 0.0) if probs else (1.0 if d.answer else 0.0)
                        a = wire.format_answer(dialect, q, yes=bool(d.answer), probability=p_yes, confidence=d.confidence,
                                               certified=d.certified, flag=d.flag)
                    else:
                        a = wire.format_answer(dialect, q, choice=d.answer, probabilities=probs, confidence=d.confidence,
                                               certified=d.certified, flag=d.flag)
                else:
                    a = up.get(q.name) or {"type": q.raw.get("type"), "name": q.name, "shad0w": {"source": "upstream", "flag": "missing"}}
                    if isinstance(a, dict):
                        a = {**a, "shad0w": {"source": "upstream", "certified": False, "flag": None}}
                if dialect == "decisions":
                    answers.append(a)
                else:
                    answers[q.name] = a
            source = "table" if not pending else ("mixed" if served else "teacher")
            out = wire.format_response(dialect, req, answers, (time.perf_counter() - t0) * 1e6,
                                       {"table": sorted(served), "upstream": [q.name for q in pending]})
            self._json(200, out, {"x-shad0w-source": source})

        def _spot_check(self, dialect, req, sh: Shadow, q: wire.WireQ, text: str, d):
            """Re-ask the upstream decision model for a served answer, in the background, at audit_rate."""
            if sh._rng.random() >= sh.audit_rate:
                return
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and not k.lower().startswith("x-shad0w")}
            sub = wire.subset(dialect, req, [q.name])
            url = self._upstream_url()

            def run():
                try:
                    r = urllib.request.Request(url, data=json.dumps(sub).encode(), method="POST")
                    for k, v in headers.items():
                        r.add_header(k, v)
                    if gw.api_key:
                        r.add_header("authorization", f"Bearer {gw.api_key}")
                    with urllib.request.urlopen(r, timeout=gw.timeout) as resp:
                        v = wire.answer_value(q, wire.parse_answers(dialect, json.loads(resp.read())).get(q.name, {}))
                    if v is not None:
                        sh._record(text, "audit", q.name, v)
                        sh._audit_result(q.name, d.answer, v, text)
                except Exception as e:
                    log.warning("spot check failed for %r: %s", q.name, e)
            threading.Thread(target=run, daemon=True, name="shad0w-audit").start()

        # -- playground ---------------------------------------------------------------------------------------
        def _playground(self, req: dict):
            name = req.get("question")
            if not name:
                names = sorted(gw._shadows)
                if len(names) != 1:
                    return self._json(400, {"error": {"message": f"pass question (one of {names})"}})
                name = names[0]
            text = str(req.get("text", ""))
            sh = gw.shadow(name)
            gw.maybe_reload(name, sh)
            out = {"explain": sh.explain(text), "llm": None}
            if req.get("ask_llm") and gw.model:
                opts = gw.options(name, sh, {}, self.headers) or []
                schema = sh.schema or {"type": "choice", "criteria": {o: None for o in opts}}
                from .llm import _prompt
                system = _prompt(name, schema.get("type", "choice"), schema.get("criteria", {}), schema.get("instructions"))
                body = {"model": gw.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                        "temperature": 0, "max_tokens": 50}
                self.path = "/v1/chat/completions"
                t = time.perf_counter()
                status, _, data = self._forward(None, relay=False, rewrite_model=body)
                took = time.perf_counter() - t
                if status == 200:
                    ans = _parse(_reply_text(data), opts or None, "answer")
                    out["llm"] = {"answer": ans, "latency_ms": round(took * 1e3, 1), "model": gw.model}
                    if req.get("log") and ans is not None:
                        sh.record(text, ans, teacher_s=took)
                else:
                    out["llm"] = {"error": data[:300].decode("utf-8", "replace"), "status": status}
            self._json(200, out)

        def log_message(self, *a):
            if log.isEnabledFor(10):
                log.debug("http " + (a[0] % a[1:]))

    return H


def _reply_text(data: bytes) -> str:
    """The assistant's text, or the arguments of its first tool call, from a chat.completion body."""
    try:
        msg = json.loads(data)["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        return ""
    if msg.get("content"):
        return msg["content"]
    calls = msg.get("tool_calls") or []
    if calls:
        return (calls[0].get("function") or {}).get("arguments") or ""
    return ""


_content = _reply_text  # backwards-compatible name


def _sse_content(data: bytes):
    for line in data.decode("utf-8", "replace").splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            try:
                ch = json.loads(line[5:].strip())
                delta = ch["choices"][0].get("delta") or {}
                if delta.get("content"):
                    yield delta["content"]
                for tc in delta.get("tool_calls") or []:
                    args = (tc.get("function") or {}).get("arguments")
                    if args:
                        yield args
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
          "  decisions API: POST /v1/decisions and /v1/systemone need no marking\n"
          "  chat: mark a decision with header 'X-Shad0w-Question: <name>', model 'shad0w/<name>', or a forced tool", flush=True)
    srv.gateway = gw  # type: ignore[attr-defined]
    return srv
