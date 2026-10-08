"""Framework adapters with fakes only: no LiteLLM or LangChain needed."""
import asyncio
import json
import sys
import types

import pytest

import shad0w
from shad0w.integrations import langchain as lc
from shad0w.integrations import litellm as ll
from shad0w.integrations._common import last_user_text

from .test_policy import CERTIFIED, bundle  # noqa: F401  (fixture)
from .test_shadow import SCHEMA

UNKNOWN = "zebra lettuce violin"


def _shadow(bundle, tmp_path):  # noqa: F811
    return shad0w.Shadow(bundle, teacher=None, question="intent", log=str(tmp_path / "log.jsonl"), audit_rate=0,
                         schema=SCHEMA["intent"])


def _rows(tmp_path):
    p = tmp_path / "log.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _resp(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_last_user_text_shapes():
    assert last_user_text("hi") == "hi"
    assert last_user_text([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]) == "u"
    assert last_user_text([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]) == "a\nb"
    assert last_user_text({"text": "t"}) == "t"
    msg = types.SimpleNamespace(type="human", content="lc")
    assert last_user_text([msg]) == "lc"


def test_litellm_completion_sdk(bundle, tmp_path, monkeypatch):  # noqa: F811
    calls = []

    def fake_completion(**kw):
        calls.append(kw)
        return _resp(kw.get("mock_response", '{"intent": "refund"}'))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=fake_completion))
    sh = _shadow(bundle, tmp_path)
    r = ll.completion(sh, model="m", messages=[{"role": "user", "content": CERTIFIED}])
    assert calls[-1]["mock_response"] == "lost_card" and r["choices"][0]["message"]["content"] == "lost_card"
    r = ll.completion(sh, model="m", messages=[{"role": "user", "content": UNKNOWN}])
    assert "mock_response" not in calls[-1]
    assert _rows(tmp_path)[-1]["intent"] == "refund"


def test_litellm_proxy_hooks(bundle, tmp_path):  # noqa: F811
    sh = _shadow(bundle, tmp_path)
    h = ll.Shad0wLogger({"intent": sh})
    data = asyncio.run(h.async_pre_call_hook(None, None, {"messages": [{"role": "user", "content": CERTIFIED}]}, "completion"))
    assert data["mock_response"] == "lost_card" and data["metadata"]["shad0w"]["source"] == "table"
    data = asyncio.run(h.async_pre_call_hook(None, None, {"messages": [{"role": "user", "content": UNKNOWN}]}, "completion"))
    assert "mock_response" not in data
    asyncio.run(h.async_log_success_event({"messages": data["messages"], "litellm_params": {"metadata": {}}},
                                          _resp("balance"), 0, 1))
    assert _rows(tmp_path)[-1]["intent"] == "balance"
    # served-by-table calls are not learned again
    n = len(_rows(tmp_path))
    asyncio.run(h.async_log_success_event({"messages": [{"role": "user", "content": CERTIFIED}],
                                           "litellm_params": {"metadata": {"shad0w": {}}}}, _resp("lost_card"), 0, 1))
    assert len(_rows(tmp_path)) == n


def test_litellm_question_from_metadata(bundle, tmp_path):  # noqa: F811
    sh = _shadow(bundle, tmp_path)
    h = ll.Shad0wLogger({"intent": sh, "other": sh})
    plain = {"messages": [{"role": "user", "content": CERTIFIED}]}
    assert "mock_response" not in asyncio.run(h.async_pre_call_hook(None, None, dict(plain), "completion"))
    tagged = {**plain, "metadata": {"shad0w_question": "intent"}}
    assert asyncio.run(h.async_pre_call_hook(None, None, tagged, "completion"))["mock_response"] == "lost_card"


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, x):
        self.calls += 1
        return types.SimpleNamespace(content=self.reply)

    async def ainvoke(self, x):
        return self.invoke(x)


class Box:
    def __init__(self, f, af):
        self.invoke, self.ainvoke = f, af


def test_langchain_runnable(bundle, tmp_path):  # noqa: F811
    sh = _shadow(bundle, tmp_path)
    llm = FakeLLM("refund")
    r = lc.shad0w_runnable(sh, llm, factory=Box)
    assert r.invoke(CERTIFIED) == "lost_card" and llm.calls == 0
    assert r.invoke([{"role": "user", "content": UNKNOWN}]) == "refund" and llm.calls == 1
    assert asyncio.run(r.ainvoke({"text": UNKNOWN})) == "refund"
    assert r.shadow is sh and _rows(tmp_path)[-1]["intent"] == "refund"


def test_langchain_unparseable_reply_not_logged(bundle, tmp_path):  # noqa: F811
    sh = _shadow(bundle, tmp_path)
    r = lc.shad0w_runnable(sh, FakeLLM("I cannot say"), factory=Box)
    assert r.invoke(UNKNOWN) is None and _rows(tmp_path) == []


def test_langchain_default_factory_needs_langchain(bundle, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", None)
    with pytest.raises(ImportError):
        lc.shad0w_runnable(_shadow(bundle, tmp_path), FakeLLM("x"))
