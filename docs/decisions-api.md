---
title: Decisions API & System One (zero code)
description: Point an OpenAI Decisions API client, or a Jev / Kev / Laya (System One) client, at shad0w and change nothing else.
---
# Decisions API & System One: zero code

If your app already calls a **decision model**, shad0w can sit in front of it with one change: the client's `base_url`. This page shows how, what you get back, and what it saves.

## TL;DR

```bash
pip install "shad0wllm[compile]"
shad0w proxy --upstream https://api.openai.com/v1        # listens on http://localhost:8010
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8010/v1")      # the only change in your app
```

Your requests keep working as before. Every answer the decision model gives is logged. After about 1,000 answers per question, run `shad0w train --dir shad0w`, and the proxy answers the questions it is **certified** on by itself, in microseconds. ("Certified" = tested on answers it never trained on, and allowed to answer only where it disagrees with the model on at most α of them; α = 5% by default. See [The guarantee](guarantee.html).)

<div class="viz" data-viz="proxy">Your app (OpenAI SDK, base_url changed) sends a request to the shad0w proxy. Requests to /v1/decisions and /v1/systemone are recognised with no marking: questions the table is certified on are answered locally, the rest are forwarded upstream in one smaller request and their answers logged. Everything else is forwarded untouched.</div>

## Which APIs this covers

Both API families already describe a decision completely in every request: the question's name and its finite list of answers. That is everything shad0w needs, so there is nothing to mark.

| API | Endpoint | Models | Question types shad0w learns | Always forwarded |
|---|---|---|---|---|
| OpenAI Decisions API (public beta since 6 Oct 2026) | `POST /v1/decisions` | `gpt-6-luna` | `choice`, `predicate` (yes/no) | `score` |
| System One | `POST /v1/systemone` | TypeSafe's Jev; the open Kev, Laya, Julia, Clef and others | `choice`, `noul` (yes/no) | `score` |

`score` questions (an ordinal scale) always go upstream: shad0w does not learn them.

## OpenAI Decisions API

```bash
shad0w proxy --upstream https://api.openai.com/v1
```

```python
from openai import OpenAI                                # openai >= 3.26

client = OpenAI(base_url="http://localhost:8010/v1")     # the only change
d = client.decisions.create(
    model="gpt-6-luna",
    input="my card was stolen yesterday",
    questions=[{
        "type": "choice", "name": "intent", "instructions": "Which support intent is this?",
        "choices": [{"value": "lost_card", "description": "card lost or stolen"},
                    {"value": "refund", "description": "wants money back"},
                    {"value": "other", "description": "anything else"}],
    }],
)
```

The key: by default the proxy forwards each client's own `Authorization` header, so nothing changes there either. To keep the key in the proxy instead, start it with `--api-key-env OPENAI_API_KEY` or `--api-key-file /run/secrets/openai`. See [The proxy](proxy.html#keys).

## System One servers (Jev, Kev, Laya, ...)

Point the proxy at any server that implements `/v1/systemone`, then point your client at the proxy:

```bash
shad0w proxy --upstream http://127.0.0.1:8081/v1         # the System One server's base URL
```

```python
import requests

r = requests.post("http://localhost:8010/v1/systemone", json={
    "model": "kev",
    "state": "my card was stolen yesterday",
    "questions": {
        "intent": {"type": "choice", "instructions": "Which support intent is this?",
                   "criteria": {"lost_card": "card lost or stolen", "refund": "wants money back", "other": "anything else"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    },
}).json()
print(r["answers"]["intent"]["choice"], r["answers"]["urgent"]["noul"])
```

**Run an open decision model on your laptop.** llama.cpp serves the System One endpoint for the open GGUF builds (Kev 0.8B/4B/9B, Laya, Julia-1, Clef-Flash, Nimble, OpenJev and lev from `ggml-org` on Hugging Face):

```bash
llama-server -hf ggml-org/Kev-0.8B-GGUF --port 8081      # then: shad0w proxy --upstream http://127.0.0.1:8081/v1
```

This exact setup was tested end to end on an Apple-silicon Mac with shad0w 0.3.0 and llama.cpp 0.6.0. Ollama's library also lists decision models such as `laya`, `tev1`, `nimble` and `clef-flash`, served at `/v1/systemone` from Ollama 0.40; that route was not tested here.

<div class="callout warn">llama.cpp implements <code>/v1/systemone</code> but not <code>/v1/decisions</code>. A Decisions-API request that the table cannot fully answer is forwarded upstream, so against llama.cpp it comes back as the server's 404, unchanged.</div>

## What happens to a request

Each `choice` and yes/no question in the request goes to the table first. A question is created the first time its name is seen, with the options taken from the request.

| Table certified for... | What the proxy does | You pay for | `x-shad0w-source` |
|---|---|---|---|
| every question | answers by itself: no upstream call, no tokens, microseconds | nothing | `table` |
| some questions | sends only the other questions upstream, in one request, and merges the answers back in the original order | the smaller request | `mixed` |
| none (or no table yet) | forwards the request upstream as it is | the full request | `teacher` |

Every answer that came from upstream is logged per question in `shad0w/<question>/log.jsonl`, so `shad0w train --dir shad0w` can learn from it. Streaming requests (`"stream": true`) are forwarded untouched.

Spot checks work as everywhere else: `--audit-rate` (default `0.01`) re-asks the upstream model about that share of the table's answers, in the background, to measure live agreement. The [rollout settings](rollout.html) (`--mode`, `--canary`, `--never-serve`, `--min-confidence`) apply to these endpoints too.

To stop intercepting these endpoints, leave `decisions` out of the capture list: `--capture header,model,tools`.

## What comes back

The response has the same shape whether the model or the table answered it, so your parsing code does not change. shad0w adds a few fields:

| Where | Field | Meaning |
|---|---|---|
| each answer | `shad0w.source` | `"table"` (answered locally) or `"upstream"` (answered by the model) |
| each answer | `shad0w.certified` | `true` when the table's answer is covered by the certificate |
| each answer | `shad0w.flag` | why an answer is not certified (`null` when it is); see [flags](python.html#flags) |
| top level | `shad0w.latency_us` | time spent in the proxy, including any upstream call |
| top level | `shad0w.table`, `shad0w.upstream` | which question names were answered where |
| header | `x-shad0w-source` | `table`, `mixed` or `teacher` (see the table above) |

A table answer carries the usual fields: `choice`, `confidence` and `probabilities` for a choice; `probability` (Decisions API) or `noul` (System One) for yes/no. Its `usage` is all zeros.

## Use a decision model as the teacher, from code

You can also go the other way: let shad0w call a decision model directly, with no proxy, and learn from its answers. ("Teacher" = the model whose answers the table learns.)

```python
import shad0w

intent = shad0w.decision(
    "intent",
    options={"lost_card": "card lost or stolen", "refund": "wants money back", "other": "anything else"},
    llm="systemone/kev",                       # or "openai-decisions/gpt-6-luna" (reads OPENAI_API_KEY)
    base_url="http://127.0.0.1:8081",          # the System One server; "/v1" is added when missing
)
intent("my card was stolen yesterday")         # Kev answers, the answer is logged
intent.train()                                 # later: the table answers the certified share itself
```

Yes/no questions (`options={"type": "yesno", "instructions": "Is it urgent?"}`) are sent as `noul` to System One and as `predicate` to the Decisions API. In JavaScript the same presets are `llm: "systemone/kev"` and `llm: "openai-decisions/gpt-6-luna"`; see [JavaScript](javascript.html#decision-models).

## `shad0w serve` speaks both too

A trained bundle served with `shad0w serve --bundle shad0w/intent/bundle` answers `POST /v1/decisions` and `POST /v1/systemone` from the table alone. There is no LLM behind `serve`, so it returns the table's answer with `certified` true or false and leaves the choice to you. Questions the bundle does not know, and `score` questions, come back with the flag `unsupported`. See [HTTP](http.html#decisions).

## Is this worth it next to a decision model?

Decision models are already fast and cheap, so the case is narrower than against a chat LLM:

- The table answers in microseconds on your own CPU, with no network hop and no per-token price. A hosted decision costs about $0.030 per 1,000 at gpt-6-luna's list price, or $0.0126 with Jev.
- It carries a written bound on how often it disagrees with the model it learned from.
- It needs about 1,000 logged answers per question first, and it only answers the share it is certified on. The rest still goes to the decision model.

The [comparison page](compare.html) puts numbers on this.
