"""LangChain adapter: a Runnable that answers from the table when certified and asks your chat model otherwise.

    from shad0w.integrations.langchain import shad0w_runnable
    intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=None)
    chain = shad0w_runnable(intent, ChatOpenAI(model="gpt-6-luna").bind(...))   # or any prompt | llm chain
    chain.invoke("my card was stolen")      # -> "lost_card"
    await chain.ainvoke([HumanMessage("...")])

The model's reply is mapped to one option and logged, so the next `intent.train()` learns from it.
"""
from __future__ import annotations

from ._common import last_user_text, to_answer


def _content(out):
    return getattr(out, "content", out)


def shad0w_runnable(shadow, llm, *, factory=None):
    """Wrap `llm` (anything with .invoke, optionally .ainvoke) as a Runnable returning the option.
    `factory` builds the Runnable from (func, afunc); default `langchain_core.runnables.RunnableLambda`."""

    def run(x):
        text = last_user_text(x)
        d = shadow.peek(text) if text else None
        if d is not None:
            return d.answer
        ans = to_answer(shadow, _content(llm.invoke(x)))
        if text and ans is not None:
            shadow.record(text, ans)
        return ans

    async def arun(x):
        text = last_user_text(x)
        d = shadow.peek(text) if text else None
        if d is not None:
            return d.answer
        out = await llm.ainvoke(x) if hasattr(llm, "ainvoke") else llm.invoke(x)
        ans = to_answer(shadow, _content(out))
        if text and ans is not None:
            shadow.record(text, ans)
        return ans

    if factory is None:
        from langchain_core.runnables import RunnableLambda
        factory = lambda f, af: RunnableLambda(f, afunc=af)  # noqa: E731
    r = factory(run, arun)
    try:
        r.shadow = shadow
    except AttributeError:  # pydantic models may refuse new attributes
        pass
    return r
