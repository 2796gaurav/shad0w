"""LiteLLM adapter: answer certified decisions from the table, let LiteLLM call the model for the rest.

    from shad0w.integrations.litellm import completion, Shad0wLogger
    intent = shad0w.decision("intent", options=[...], llm=None)        # a Shadow; the teacher is LiteLLM itself

    # SDK: drop-in for litellm.completion
    resp = completion(intent, model="gpt-6-luna", messages=msgs)       # table answer when certified, else LiteLLM

    # Proxy: litellm config.yaml -> callbacks: my_hooks.handler, with my_hooks.py:  handler = Shad0wLogger({"intent": intent})
    # A request is a decision when its metadata names the question: extra_body={"metadata": {"shad0w_question": "intent"}}

Certified answers go back through LiteLLM's `mock_response`, so they come out as ordinary ModelResponse objects.
"""
from __future__ import annotations

from typing import Any

from ._common import answer_text, last_user_text, to_answer

try:  # the base class only exists when LiteLLM is installed; the hooks work duck-typed without it
    from litellm.integrations.custom_logger import CustomLogger as _Base  # type: ignore
except Exception:  # pragma: no cover - depends on the environment
    _Base = object


def _reply(resp: Any) -> str | None:
    try:
        return resp["choices"][0]["message"]["content"] if isinstance(resp, dict) else resp.choices[0].message.content
    except (KeyError, IndexError, AttributeError, TypeError):
        return None


def completion(shadow, **kw):
    """`litellm.completion(**kw)`, except that a certified table answer is returned without calling the model."""
    import litellm
    text = last_user_text(kw.get("messages"))
    d = shadow.peek(text) if text else None
    if d is not None:
        return litellm.completion(**kw, mock_response=answer_text(d.answer))
    resp = litellm.completion(**kw)
    ans = to_answer(shadow, _reply(resp))
    if text and ans is not None:
        shadow.record(text, ans)
    return resp


class Shad0wLogger(_Base):
    """LiteLLM proxy callback. `shadows` maps question names to Shadows; `question_of(data)` may pick one."""

    def __init__(self, shadows: dict, question_of=None):
        if _Base is not object:
            super().__init__()
        self.shadows, self.question_of = dict(shadows), question_of

    def _question(self, data: dict) -> str | None:
        if self.question_of:
            return self.question_of(data)
        q = (data.get("metadata") or {}).get("shad0w_question")
        return q or (next(iter(self.shadows)) if len(self.shadows) == 1 else None)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        q, text = self._question(data), last_user_text(data.get("messages"))
        sh = self.shadows.get(q) if q else None
        if sh is None or not text:
            return data
        d = sh.peek(text)
        if d is not None:
            data["mock_response"] = answer_text(d.answer)
            data.setdefault("metadata", {})["shad0w"] = {"question": q, "source": "table", "confidence": d.confidence}
        return data

    async def async_log_success_event(self, kwargs: dict, response_obj, start_time, end_time):
        meta = (kwargs.get("litellm_params") or {}).get("metadata") or kwargs.get("metadata") or {}
        if "shad0w" in meta:  # served by the table: nothing to learn
            return
        q, text = self._question({"metadata": meta, **kwargs}), last_user_text(kwargs.get("messages"))
        sh = self.shadows.get(q) if q else None
        ans = to_answer(sh, _reply(response_obj)) if sh is not None else None
        if text and ans is not None:
            sh.record(text, ans)
