---
title: The proxy (zero code changes)
description: An OpenAI-compatible gateway. Point base_url at it, mark which calls are decisions, and shad0w does the rest.
---
# The proxy: zero code changes

`shad0w proxy` sits between your app and any OpenAI-compatible API. Your code keeps using its OpenAI SDK; only the `base_url` changes. This page covers starting it, marking which calls are decisions, keys, what comes back, and every flag.

## TL;DR

```bash
pip install "shad0wllm[compile]"
shad0w proxy --upstream https://api.openai.com/v1 --auto-train 1000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8010/v1")             # 1. the only change to your client
r = client.chat.completions.create(
    model="gpt-6-luna", messages=[{"role": "user", "content": "my card was stolen"}],
    extra_headers={"X-Shad0w-Question": "intent", "X-Shad0w-Options": "refund,lost_card,balance,other"})  # 2. mark it
print(r.choices[0].message.content)                                   # lost_card
```

At first every marked call goes to the upstream LLM and its answer is logged. With `--auto-train 1000`, the proxy trains a table (shad0w's small learned model) after 1,000 answers. From then on it answers the calls it is **certified** on by itself, in microseconds. ("Certified" = tested on answers it never trained on, and allowed to answer only where it disagrees with your LLM on at most α of them; α = 5% by default.) Everything that is not a marked decision is forwarded untouched.

<div class="viz" data-viz="proxy">Your app (OpenAI SDK, base_url changed) sends every request to the shad0w proxy. Marked chat completions (a header, a shad0w/ model name or a forced enum tool) and Decisions API / System One requests are decisions: the table answers the certified ones locally; the rest go to the upstream LLM and their answers are logged. Unmarked requests are forwarded untouched.</div>

## 1. Start it

```bash
shad0w proxy --upstream https://api.openai.com/v1
```

```text
shad0w proxy on http://127.0.0.1:8010/v1  ->  https://api.openai.com/v1
  dashboard http://127.0.0.1:8010/   metrics http://127.0.0.1:8010/metrics   data /home/you/app/shad0w
  decisions API: POST /v1/decisions and /v1/systemone need no marking
  chat: mark a decision with header 'X-Shad0w-Question: <name>', model 'shad0w/<name>', or a forced tool
  upstream key: each client sends its own
```

The upstream can be OpenAI, Groq, Together, OpenRouter, Mistral, a LiteLLM proxy, Ollama (`http://localhost:11434/v1`), vLLM, or anything else that speaks the chat completions API.

## 2. Point your client at it

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8010/v1")
```

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8010/v1" });
```

## Keys {#keys}

| How | Flag | Use it when |
|---|---|---|
| Forward each client's own key (default) | none | clients already hold their keys. Each client's key is used for its own requests and for the spot checks on them, so two clients never borrow each other's key. |
| The proxy's key, from an environment variable | `--api-key-env OPENAI_API_KEY` | the key lives with the proxy, not the clients |
| The proxy's key, from a file | `--api-key-file /run/secrets/openai` | a Docker or Kubernetes secret. The file is re-read on every request, so a rotated secret needs no restart. |

There is no flag that takes the key itself: it would show up in `ps` and shell history. The two flags cannot be combined. The startup banner says where the upstream key comes from, masked: `upstream key: set via OPENAI_API_KEY (sk-…3f9a)`.

## 3. Mark which calls are decisions {#mark}

**Using the OpenAI Decisions API or a System One model (Jev, Kev, Laya)? Nothing to mark.** Requests to `/v1/decisions` and `/v1/systemone` already name each question and list its answers. See [Decisions API & System One](decisions-api.html).

For chat completions, pick whichever is easiest:

```python
msgs = [{"role": "system", "content": "Classify the support message."},
        {"role": "user", "content": "my card was stolen"}]

# a header
client.chat.completions.create(model="gpt-6-luna", messages=msgs, extra_headers={"X-Shad0w-Question": "intent"})

# a model name: "shad0w/<question>@<upstream model>" (or "shad0w/<question>" with --model on the proxy)
client.chat.completions.create(model="shad0w/intent@gpt-6-luna", messages=msgs)

# a forced tool whose parameters hold exactly one enum: the tool's name is the question
client.chat.completions.create(model="gpt-6-luna", messages=msgs,
    tools=[{"type": "function", "function": {"name": "intent", "parameters": {"type": "object",
        "properties": {"intent": {"type": "string", "enum": ["refund", "lost_card", "balance", "other"]}}}}}],
    tool_choice={"type": "function", "function": {"name": "intent"}})
```

Which shapes count as decisions is set by `--capture` (or `capture` in [Configuration](configuration.html)):

| Capture | Recognises | Default |
|---|---|---|
| `header` | the `X-Shad0w-Question: <name>` request header | on |
| `model` | `model="shad0w/<name>"` or `"shad0w/<name>@<upstream model>"` | on |
| `tools` | a forced tool (or the only tool) with exactly one enum parameter; the table's answer comes back as a `tool_calls` message | on |
| `decisions` | `POST /v1/decisions` and `POST /v1/systemone` | on |
| `json_schema` | a `response_format` JSON schema with a `name`; the name is the question | **off**: turn on with `--capture json_schema` |

`json_schema` is off by default because many apps use named schemas for things that are not decisions. Turning it on treats every such request as one. `--capture` takes a comma-separated list and can be repeated; it replaces the default list.

**Where the options come from**, first match wins: the request's own enum (JSON schema or tool parameter), the `X-Shad0w-Options: refund,lost_card,balance,other` header, `--schema`, then a trained table. Without any of these the proxy learns the option set from the short answers your LLM gives.

**The text shad0w learns from is the last user message.** Put the thing to classify there and keep instructions in the system message.

`model="shad0w/<name>"` with no `@<model>` and no `--model` flag is refused with a 400, because the proxy would not know which upstream model to call.

## What happens to a decision

| Situation | What the proxy does |
|---|---|
| No trained table yet | forwards to the upstream, returns its reply untouched, logs the answer to `shad0w/<question>/log.jsonl` |
| Table certified for this text | answers itself in microseconds with a normal `chat.completion` |
| Table not sure | forwards, returns the reply, logs the answer (training data for next time) |
| `--mode shadow` | always forwards and returns the LLM's reply; counts what the table would have answered |
| Upstream error | returns the upstream's error as-is |

A table answer is shaped like the request asked for:

- plain text: the option name;
- JSON object mode: `{"intent": "lost_card"}`;
- your JSON schema: `{"<your field>": "lost_card"}`;
- a forced tool: a `tool_calls` message with `finish_reason: "tool_calls"`;
- `stream: true`: server-sent events, for text and tool calls;
- `/v1/decisions` and `/v1/systemone`: the same JSON shape the upstream would return ([details](decisions-api.html#what-comes-back)).

Its `usage` is all zeros, and it carries an extra `shad0w` object: `question`, `source`, `answer`, `confidence`, `certified`, `flag`, `latency_us`.

**Response headers** on a marked chat completion:

| Header | When | Value |
|---|---|---|
| `x-shad0w-source` | always | `table` (answered locally) or `teacher` (your LLM answered) |
| `x-shad0w-question` | always | the question name |
| `x-shad0w-latency-us` | always | time spent in the proxy, in microseconds |
| `x-shad0w-confidence` | table answers | the table's confidence, 0 to 1 |
| `x-shad0w-certified` | table answers | `1` or `0` |

On `/v1/decisions` and `/v1/systemone`, `x-shad0w-source` is `table`, `mixed` or `teacher`. Request headers starting with `X-Shad0w-` are never forwarded upstream.

## 4. Train

```bash
shad0w stats --log shad0w/intent/log.jsonl        # how many answers, which options, ready yet?
shad0w train --dir shad0w --question intent       # the running proxy picks up the new table within a second
```

Or let the proxy do it: `--auto-train 1000` retrains a question every 1,000 new answers. By default (`retrain = "gated"`) the new table replaces the old one only if it certifies at least four fifths (0.8) of the old one's share. Training needs `pip install "shad0wllm[compile]"`. See [Training](training.html).

## Flags

Every flag can also come from `shad0w.toml` or a `SHAD0W_*` variable, and a `[questions.<name>]` section sets values for one question; see [Configuration](configuration.html). A flag beats both.

| Flag | Type | Default | What it does | Change it when |
|---|---|---|---|---|
| `--upstream` | URL | `https://api.openai.com/v1` | the real API | you use another provider or a local server |
| `--host`, `--port` | str, int | `127.0.0.1`, `8010` | where the proxy listens | it runs in a container (`--host 0.0.0.0`) or the port is taken |
| `--dir` | path | `shad0w` | logs and tables live in `<dir>/<question>/` | several apps share a machine |
| `--bundle Q=PATH` | repeatable | | use the table at PATH for question Q | the table was trained elsewhere |
| `--model` | str | | upstream model for `model="shad0w/<question>"` | you use the model-name marking |
| `--schema` | path | | a `schema.json` naming each question's options | requests do not list the options |
| `--api-key-env NAME` | str | | send the key held in this variable upstream, instead of the client's | see [Keys](#keys) |
| `--api-key-file PATH` | path | | send the key held in this file upstream, re-read on every request | see [Keys](#keys) |
| `--capture` | list | `header,model,tools,decisions` | which request shapes count as decisions | see [above](#mark) |
| `--mode` | `serve` / `shadow` / `off` | `serve` | `shadow`: always return the LLM's answer, count what the table would have done; `off`: never consult the table | rolling out; see [Rollout](rollout.html) |
| `--canary` | 0 to 1 | `1` | share of certified answers the table may serve | rolling out gradually, e.g. `0.1` |
| `--never-serve` | comma list | | labels that always go upstream | sensitive labels such as `fraud` |
| `--min-confidence` | 0 to 1 | | a confidence floor on top of the certificate (only makes serving stricter) | you want extra margin |
| `--audit-rate` | 0 to 1 | `0.01` | share of decisions spot-checked against the upstream in the background | more (or less) live measurement |
| `--auto-train` | int | `0` (off) | retrain a question every N new answers | you want it hands-off |
| `--min-rows` | int | `1000` | logged answers needed before auto-train runs (100 at least) | few options, clean data |
| `--alpha` | 0 to 1 | `0.05` | most disagreement with your LLM allowed among table answers (α) | stricter (`0.02`) or looser (`0.1`) |
| `--delta` | 0 to 1 | `0.1` | chance the certificate is wrong because of an unlucky sample (δ; `0.1` = 90% confidence) | rarely |
| `--timeout` | seconds | `120` | upstream timeout | slow models |
| `--trace` | path | | JSON Lines file receiving every decision | auditing, offline analysis |
| `--cost-per-call` | float | | your cost per LLM call, to show money saved on the dashboard | you want the dashboard figure |
| `--llm-latency-ms` | float | | LLM latency to assume until one is measured, to show time saved | you want the dashboard figure |
| `--no-text` | flag | | keep request text off the dashboard | texts are sensitive |
| `--config` | path | `./shad0w.toml` (or `$SHAD0W_CONFIG`) | the config file to read | it lives elsewhere |

## Routes

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | decisions when marked (above); otherwise forwarded |
| `POST /v1/decisions`, `POST /v1/systemone` | decision-model formats, no marking needed |
| `GET /` | live dashboard, with a **Playground** |
| `GET /v1/stats` | JSON: counts, share answered, flags, latency, per-question settings and log sizes |
| `GET /metrics` | Prometheus text format |
| `GET /v1/health` | `{"ok": true, "upstream": ..., "questions": [...]}` |
| `POST /v1/playground` | `{"question", "text", "ask_llm": false, "log": false}`: what the table thinks of a text, and with `ask_llm` (needs `--model`) what your LLM says |
| anything else | forwarded untouched (embeddings, images, `GET /v1/models`, ...) |

The dashboard's Playground uses that last route: type a text, pick a question, and see the table's answer, its confidence against the threshold, and why; with `--model` set, also ask the upstream LLM and compare the two side by side.

<div class="callout warn">The proxy has no authentication of its own. Keep it on localhost or inside your network.</div>
