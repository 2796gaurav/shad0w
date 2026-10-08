"""Helpers shared by the framework adapters (standard library only)."""
from __future__ import annotations

from typing import Any

from ..llm import match_option


def options_of(shadow) -> list[str]:
    """The options a Shadow decides among, from its schema (yes/no questions answer "yes" or "no")."""
    s = shadow.schema or {}
    if s.get("type", "choice") == "yesno":
        return ["yes", "no"]
    return list(s.get("criteria") or {})


def to_answer(shadow, reply: Any):
    """Map an LLM reply to one of the Shadow's options (a bool for yes/no questions), or None."""
    a = match_option(reply, options_of(shadow))
    if a is not None and (shadow.schema or {}).get("type") == "yesno":
        return a == "yes"
    return a


def last_user_text(messages: Any) -> str | None:
    """The last user message's text from an OpenAI-style list, a LangChain message list, a dict or a string."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        for k in ("text", "input", "question", "content"):
            if isinstance(messages.get(k), str):
                return messages[k]
        return last_user_text(messages.get("messages"))
    for m in reversed(list(messages or [])):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None) or getattr(m, "role", None)
        if role not in ("user", "human"):
            continue
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", None))
    return None


def answer_text(answer) -> str:
    return ("yes" if answer else "no") if isinstance(answer, bool) else str(answer)
