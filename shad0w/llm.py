"""Turn any OpenAI-compatible chat model into a shad0w teacher in one line. Standard library only.

    teacher = shad0w.llm_teacher({"refund": "wants money back", "lost_card": "card lost or stolen"},
                                 model="openai/gpt-4o-mini")
    teacher("my card was stolen")          # -> "lost_card"

`model` is "provider/model-name". The provider picks the base URL and the environment variable that holds
the key. Any other OpenAI-compatible server works with base_url=... (and a model name without a prefix).
Pass client=OpenAI(...) to reuse an openai SDK client (Azure, proxies, custom auth) instead of the built-in
HTTP call, or complete=fn to use anything else (LiteLLM, LangChain, a local model): fn(messages) -> reply text.

The model is asked for a JSON object whose one field is constrained to your options (structured outputs).
Servers that do not support that get the same prompt as plain text. Either way the reply is mapped back to
exactly one of your options, or the call raises TeacherError, so nothing unexpected reaches your log.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

__all__ = ["llm_teacher", "LLMTeacher", "TeacherError", "PROVIDERS", "normalize_options", "match_option"]

# provider -> (base URL, environment variable holding the API key; None for local servers)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "azure": ("", "AZURE_OPENAI_API_KEY"),  # needs base_url="https://<resource>.openai.azure.com/openai/v1"
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "lmstudio": ("http://localhost:1234/v1", None),
}


class TeacherError(RuntimeError):
    """The LLM call failed, or its reply did not match any option. Nothing is logged for that input."""


def normalize_options(options: Any, question: str = "decision") -> tuple[str, str, dict[str, str | None], str | None]:
    """Accept a list of option names, {option: description}, one schema entry {"type":..,"criteria":..},
    or a whole schema {question: entry}. Returns (question, qtype, criteria, instructions)."""
    if isinstance(options, (list, tuple)):
        return question, "choice", {str(o): None for o in options}, None
    if not isinstance(options, dict) or not options:
        raise TypeError("options must be a list of names, a {name: description} dict or a shad0w schema")
    if "criteria" in options or options.get("type") in ("choice", "yesno"):
        entry = options
    elif all(isinstance(v, dict) and ("criteria" in v or v.get("type") in ("choice", "yesno")) for v in options.values()):
        if len(options) != 1 and question not in options:
            raise ValueError(f"schema has questions {list(options)}; pass question=...")
        question = question if question in options else next(iter(options))
        entry = options[question]
    else:
        return question, "choice", {str(k): (None if v is None else str(v)) for k, v in options.items()}, None
    qtype = entry.get("type", "choice")
    crit = {"yes": None, "no": None} if qtype == "yesno" else {str(k): v for k, v in entry["criteria"].items()}
    return question, qtype, crit, entry.get("instructions")


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def match_option(reply: Any, options: list[str], field: str | None = None):
    """Map an LLM reply (raw text, JSON text, or parsed JSON) to exactly one option, else None."""
    if isinstance(reply, str):
        s = reply.strip()
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)  # fenced code
        if s[:1] in "{[\"":
            try:
                reply = json.loads(s)
            except ValueError:
                reply = s
        else:
            reply = s
    if isinstance(reply, dict):
        for k in ([field] if field else []) + ["answer", "label", "choice", "intent", "category", "class"]:
            if k in reply:
                return match_option(reply[k], options)
        vals = [v for v in reply.values() if isinstance(v, (str, bool))]
        return match_option(vals[0], options) if len(vals) == 1 else None
    if isinstance(reply, list):
        return match_option(reply[0], options) if len(reply) == 1 else None
    if isinstance(reply, bool):
        reply = "yes" if reply else "no"
    if not isinstance(reply, str) or not reply:
        return None
    if reply in options:
        return reply
    canon = {_canon(o): o for o in options}
    c = _canon(reply)
    if c in canon:
        return canon[c]
    if {"yes", "no"} <= set(options) and c in ("true", "false", "y", "n"):
        return "yes" if c in ("true", "y") else "no"
    # an option named inside a short sentence ("The answer is lost_card."), only when exactly one fits
    hits = [o for k, o in canon.items() if k and re.search(rf"(^|_){re.escape(k)}(_|$)", c)]
    hits = [h for h in hits if not any(h != g and _canon(h) in _canon(g) for g in hits)]  # prefer the longest
    return hits[0] if len(hits) == 1 else None


def _prompt(question: str, qtype: str, criteria: dict[str, str | None], instructions: str | None) -> str:
    if qtype == "yesno":
        head = instructions or f"Answer the question '{question}' about the user's message with yes or no."
        return f"{head}\nReply with JSON: {{\"answer\": \"yes\"}} or {{\"answer\": \"no\"}}. Nothing else."
    lines = [instructions or f"Classify the user's message ({question}). Pick exactly one option."]
    lines.append("Options:")
    for name, desc in criteria.items():
        lines.append(f"- {name}" + (f": {desc}" if desc else ""))
    lines.append('Reply with JSON: {"answer": "<option>"} using one option name exactly as written. Nothing else.')
    return "\n".join(lines)


class LLMTeacher:
    """A callable teacher backed by an OpenAI-compatible chat completions endpoint. Thread-safe."""

    def __init__(self, options: Any, model: str = "openai/gpt-4o-mini", *, question: str | None = None,
                 base_url: str | None = None, api_key: str | None = None, client: Any = None,
                 system: str | None = None, temperature: float = 0.0, timeout: float = 30.0, retries: int = 2,
                 headers: dict | None = None, structured: bool | None = None, max_tokens: int = 50,
                 complete: Any = None):
        self.question, self.qtype, self.criteria, instructions = normalize_options(options, question or "decision")
        self.options = list(self.criteria)
        self._complete = complete
        if complete is not None:
            client, base_url = None, base_url or "-"
        provider, _, name = model.partition("/") if "/" in model else ("", "", model)
        if provider not in PROVIDERS:  # "meta-llama/Llama-3-8b" on a custom base_url: keep the full name
            provider, name = "", model
        self.provider, self.model = provider or "custom", name
        default_url, key_env = PROVIDERS.get(provider, ("", None))
        self.base_url = (base_url or os.environ.get("SHAD0W_BASE_URL") or default_url).rstrip("/")
        if client is None and not self.base_url:
            raise ValueError(f"model {model!r}: unknown provider; pass base_url=... (known: {', '.join(PROVIDERS)})")
        self.api_key = api_key if api_key is not None else (os.environ.get(key_env) if key_env else None)
        if client is None and complete is None and key_env and not self.api_key:
            raise ValueError(f"{provider}: set {key_env} or pass api_key=...")
        self.client, self.temperature, self.timeout, self.retries = client, temperature, timeout, retries
        self.headers = dict(headers or {})
        self.max_tokens = max_tokens
        self.system = system or _prompt(self.question, self.qtype, self.criteria, instructions)
        self._structured = True if structured is None else structured
        self._dropped: set[str] = set()
        self.name = f"{self.provider}/{self.model}"
        self.calls = 0
        self.failures = 0

    @property
    def response_format(self) -> dict:
        enum = ["yes", "no"] if self.qtype == "yesno" else self.options
        return {"type": "json_schema", "json_schema": {
            "name": re.sub(r"[^a-zA-Z0-9_-]", "_", self.question)[:64] or "decision", "strict": True,
            "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": enum}},
                       "required": ["answer"], "additionalProperties": False}}}

    def messages(self, text: str) -> list[dict]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": text}]

    def parse(self, content: str):
        """The option for a raw reply (None when it matches nothing)."""
        opt = match_option(content, ["yes", "no"] if self.qtype == "yesno" else self.options, "answer")
        if opt is None:
            return None
        return (opt == "yes") if self.qtype == "yesno" else opt

    def __call__(self, text: str):
        content = self.complete(text)
        ans = self.parse(content)
        if ans is None:
            self.failures += 1
            raise TeacherError(f"{self.name} replied {content[:200]!r}, which matches none of {self.options}")
        return ans

    def complete(self, text: str) -> str:
        """Raw reply text for one input, with retries and the structured-output fallback."""
        self.calls += 1
        body = {"model": self.model, "messages": self.messages(text), "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        for k in self._dropped:
            body.pop(k, None)
        last, attempt, adjusted = None, 0, 0
        while True:
            if self._structured:
                body["response_format"] = self.response_format
            else:
                body.pop("response_format", None)
            try:
                return self._send(body)
            except _DropParam as e:  # e.g. reasoning models refuse temperature / max_tokens: drop it, go again
                body.pop(e.param, None)
                self._dropped.add(e.param)
                last, adjusted = e, adjusted + 1
                if adjusted > 3:
                    break
                continue
            except _Unsupported as e:  # this server rejects json_schema: remember, retry at once as plain text
                if not self._structured:
                    last = e
                    break
                self._structured = False
                continue
            except _Retryable as e:
                last = e
            if attempt >= self.retries:
                break
            time.sleep(min(8.0, getattr(last, "wait", None) or 0.5 * 2 ** attempt))
            attempt += 1
        self.failures += 1
        raise TeacherError(f"{self.name}: {last}") from last

    def _send(self, body: dict) -> str:
        if self._complete is not None:
            try:
                return str(self._complete(body["messages"]) or "")
            except Exception as e:
                raise _Retryable(str(e)) from None
        if self.client is not None:
            return self._send_sdk(body)
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(), method="POST")
        req.add_header("content-type", "application/json")
        if self.api_key:
            req.add_header("authorization", f"Bearer {self.api_key}")
            if self.provider == "azure":
                req.add_header("api-key", self.api_key)
        for k, v in self.headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode("utf-8", "replace")
            if e.code in (400, 422):
                _classify_400(body, detail)
            if e.code == 429 or e.code >= 500:
                wait = e.headers.get("retry-after") if e.headers else None
                raise _Retryable(f"HTTP {e.code}: {detail}", float(wait) if wait and wait.replace(".", "").isdigit() else None) from None
            raise TeacherError(f"{self.name}: HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise _Retryable(f"cannot reach {self.base_url}: {e}") from None
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise TeacherError(f"{self.name}: unexpected response {str(data)[:300]}") from None

    def _send_sdk(self, body: dict) -> str:
        kw = {k: v for k, v in body.items() if k != "messages"}
        try:
            r = self.client.chat.completions.create(messages=body["messages"], **kw)
        except Exception as e:  # the SDK raises its own types; classify by status code and message
            status = getattr(e, "status_code", None)
            msg = str(e)
            if status in (400, 422):
                _classify_400(body, msg)
            if status is None or status == 429 or status >= 500:
                raise _Retryable(msg) from None
            raise TeacherError(f"{self.name}: {msg}") from None
        return r.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"LLMTeacher({self.name!r}, question={self.question!r}, options={len(self.options)})"


def llm_teacher(options: Any, model: str = "openai/gpt-4o-mini", **kw) -> LLMTeacher:
    """A teacher callable(text) -> option backed by any OpenAI-compatible LLM. See LLMTeacher for keywords."""
    return LLMTeacher(options, model, **kw)


class _Retryable(Exception):
    def __init__(self, msg, wait=None):
        super().__init__(msg)
        self.wait = wait


class _Unsupported(Exception):
    pass


class _DropParam(Exception):
    def __init__(self, param):
        super().__init__(f"server rejected {param}")
        self.param = param


def _classify_400(body: dict, msg: str):
    m = msg.lower()
    for p in ("temperature", "max_tokens"):
        if p in body and p in m:
            raise _DropParam(p)
    if "response_format" in body and _mentions_format(msg):
        raise _Unsupported(msg)


def _mentions_format(msg: str) -> bool:
    m = msg.lower()
    return any(w in m for w in ("response_format", "json_schema", "structured", "schema", "not supported", "unsupported"))
