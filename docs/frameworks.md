---
title: Frameworks
description: Recipes for LangChain, LangGraph, LiteLLM, the Vercel AI SDK, Pydantic AI, FastAPI, Django and Cloudflare Workers.
---
# Frameworks

Copy-paste recipes for putting shad0w in front of the LLM call your framework already makes. shad0w needs one thing from a framework: a way to ask the LLM for an answer. Every recipe below is that, plus a line or two.

## Pick a recipe

| You use | Easiest route | Code change |
|---|---|---|
| any Python function that asks an LLM | [`@shad0w.decide`](#any-function-the-decorator) | a decorator |
| LangChain / LangGraph | [`llm_teacher(complete=...)` or `shad0w_runnable`](#langchain-langgraph) | a few lines |
| LiteLLM (SDK or proxy) | [`shad0w.integrations.litellm`](#litellm) | one import, or a callback |
| Vercel AI SDK | [`shad0wMiddleware` / `decisionModel`](#vercel-ai-sdk) | wrap the model |
| Pydantic AI and other decision-model clients | [the proxy](#pydantic-ai-and-other-decision-model-clients) | `base_url` only |
| FastAPI, Django, Flask | [create once, call per request](#fastapi) | a few lines |
| any OpenAI SDK, any language | [the proxy](#anything-with-an-openai-client) | `base_url` only |

Every recipe has the same behaviour: the table (shad0w's small learned model) answers what it is **certified** on (tested on answers it never trained on, with a bound on how often it disagrees with your LLM); everything else goes to your LLM and the answer is logged for the next `train()`.

## Any function: the decorator

If you already have a function that asks your LLM, annotate its return type with the options:

```python
from typing import Literal
import shad0w

@shad0w.decide()
def route(text: str) -> Literal["billing", "tech", "sales", "other"]:
    return my_llm_call(text)          # Instructor, Pydantic AI, LangChain, raw SDK: anything

route("my invoice is wrong")          # from the table once trained and certified, else from my_llm_call (logged)
route.shadow.train()                  # after ~1,000 logged answers
```

`-> bool` makes a yes/no question, and an `Enum` return type returns enum members. To let shad0w call the LLM itself, pass it to the decorator: `@shad0w.decide(llm="openai/gpt-6-luna")`, and leave the body as `...`. See [Python](python.html#decorators).

## LangChain / LangGraph

Use the chat model you already have as the teacher. shad0w builds the prompt and maps the reply to one option:

```python
import shad0w
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

llm = ChatOpenAI(model="gpt-6-luna", temperature=0)
teacher = shad0w.llm_teacher(["billing", "tech", "sales", "other"], question="team",
                             complete=lambda messages: llm.invoke(messages).content)
team = shad0w.decision("team", llm=teacher)

route = RunnableLambda(lambda x: team(x["text"]).answer)        # use it in a chain...

def pick_next(state):                                           # ...or as a LangGraph conditional edge
    return team(state["messages"][-1].content).answer
graph.add_conditional_edges("triage", pick_next, {"billing": "billing_agent", "tech": "tech_agent", "sales": "sales_agent", "other": "human"})
```

Or wrap a chain you already have (your own prompt and model) as a Runnable. The table answers what it is certified on; otherwise the chain runs, its reply is mapped to one option and logged:

```python
import shad0w
from shad0w.integrations.langchain import shad0w_runnable

team = shad0w.decision("team", options=["billing", "tech", "sales", "other"])   # no llm: the chain supplies it
router = shad0w_runnable(team, prompt | llm)        # .invoke(text | {"text": ...} | messages), .ainvoke(...) too
router.invoke({"text": "my invoice is wrong"})      # -> "billing"
```

With `shad0w_runnable`, shad0w never calls the model itself, so there are no background spot checks; the first recipe has them.

## LiteLLM

**In-process, any of LiteLLM's providers as the teacher:**

```python
import litellm, shad0w

complete = lambda messages: litellm.completion(model="anthropic/claude-haiku-4-5", messages=messages,
                                               temperature=0).choices[0].message.content
intent = shad0w.decision("intent", llm=shad0w.llm_teacher(["refund", "lost_card", "other"], question="intent", complete=complete))
```

**Keep calling LiteLLM** and let shad0w short-circuit certified answers. They come back as normal `ModelResponse` objects, through LiteLLM's `mock_response`:

```python
import shad0w
from shad0w.integrations.litellm import completion

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"])
resp = completion(intent, model="gpt-6-luna", messages=[{"role": "user", "content": "my card was stolen"}])
```

**LiteLLM proxy.** Register a callback in a file next to your `config.yaml`:

```python
# shad0w_hooks.py
import shad0w
from shad0w.integrations.litellm import Shad0wLogger

handler = Shad0wLogger({"intent": shad0w.decision("intent", options=["refund", "lost_card", "other"])})
```

```yaml
# config.yaml
litellm_settings:
  callbacks: shad0w_hooks.handler
```

A request is a decision when its metadata names the question, `extra_body={"metadata": {"shad0w_question": "intent"}}`, or when the handler has only one question. `Shad0wLogger(shadows, question_of=fn)` lets you pick the question from the request yourself. Answers from the model are logged for the next `train()`. These adapters are tested against stand-ins for LiteLLM's interfaces, not against a running LiteLLM proxy.

**Or put the shad0w proxy in front of a LiteLLM proxy.** It speaks OpenAI, so nothing else changes:

```bash
shad0w proxy --upstream http://localhost:4000 --port 8010
```

## Vercel AI SDK

Wrap the model: certified answers skip it, everything else is logged for training.

```ts
import { wrapLanguageModel, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { decision, shad0wMiddleware } from "shad0wllm";

const intent = await decision("intent", { options: ["refund", "lost_card", "balance", "other"],
                                          bundle: "shad0w/intent/bundle", log: "shad0w/intent/log.jsonl" });
const model = wrapLanguageModel({ model: openai("gpt-6-luna"), middleware: shad0wMiddleware(intent, { specificationVersion: "v3" }) });

export async function POST(req: Request) {
  const { message } = await req.json();
  const { text } = await generateText({ model, system: "Answer with one of: refund, lost_card, balance, other.", prompt: message });
  return Response.json({ intent: text });
}
```

`specificationVersion` must match your `ai` major version (ai 5 → `"v2"`, ai 6 → `"v3"`, ai 7 → `"v4"`). With `experimental_decide`, use `decisionModel(intent, { fallback: openai.decisionModel("gpt-6-luna") })` as the model: the table answers the questions it certifies and the fallback answers the rest. See [JavaScript](javascript.html#vercel-ai-sdk).

## Pydantic AI and other decision-model clients

Point the client's base URL at `shad0w proxy`. Requests to `/v1/decisions` (OpenAI Decisions API) and `/v1/systemone` (Jev, Kev, Laya) are recognised with no marking. See [Decisions API & System One](decisions-api.html).

## FastAPI

```python
import shad0w
from fastapi import FastAPI

app = FastAPI()
intent = shad0w.decision("intent", options=["refund", "lost_card", "balance", "other"], llm="openai/gpt-6-luna",
                         auto_train=2000)                  # retrain in the background every 2,000 new answers

@app.post("/route")
async def route(body: dict):
    d = await intent.adecide(body["text"])                 # never blocks the event loop on a table answer
    return {"intent": d.answer, "source": d.source, "why": d.why, "us": d.latency_us}

@app.get("/shad0w")
def stats():
    return intent.stats()
```

For Prometheus, share one `shad0w.Metrics()` across your deciders (`metrics=` keyword) and expose `metrics.prometheus()` on `/metrics`. See [Monitoring](observability.html).

## Django / Flask

Create the decider once, at import time (module level). It is thread-safe. Call `intent(text)` in your views.

## Cloudflare Workers, Deno, Bun

See [JavaScript](javascript.html). The same `decision()` works everywhere `fetch` does.

## Anything with an OpenAI client

Run the [proxy](proxy.html) and set `base_url`. That covers Ruby, Go, Java, .NET, PHP and every other language with an OpenAI SDK.
