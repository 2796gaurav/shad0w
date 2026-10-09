---
title: Connect your LLM
description: Use OpenAI, Anthropic, Gemini, Groq, Ollama, vLLM, a decision model or any OpenAI-compatible server as shad0w's teacher, and pass its API key safely.
---
# Connect your LLM

This page shows how to point shad0w at the LLM you already use, how to give it the API key, and what shad0w sends to that LLM.

The LLM that answers today is shad0w's **teacher**: the table (shad0w's small learned model) learns its answers and is certified against them. Name it with a `"provider/model"` string, or pass any function.

## TL;DR

```python
import os, shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"],
                         llm="anthropic/claude-haiku-4-5",
                         api_key=os.environ["ANTHROPIC_API_KEY"])
```

## Providers

| Provider | `llm=` | Key read from (by default) |
|---|---|---|
| OpenAI | `openai/gpt-6-luna` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Groq | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |
| Together | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| OpenRouter | `openrouter/openai/gpt-6-luna` | `OPENROUTER_API_KEY` |
| Mistral | `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| Fireworks | `fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct` | `FIREWORKS_API_KEY` |
| xAI | `xai/grok-3-mini` | `XAI_API_KEY` |
| Ollama (local) | `ollama/llama3.1` | none (`http://localhost:11434/v1`) |
| vLLM (local) | `vllm/<model>` | none (`http://localhost:8000/v1`) |
| LM Studio (local) | `lmstudio/<model>` | none (`http://localhost:1234/v1`) |
| OpenAI Decisions API | `openai-decisions/gpt-6-luna` | `OPENAI_API_KEY` |
| System One decision models (Jev, Kev, Laya, llama.cpp, ...) | `systemone/<model>` + `base_url="http://host:port"` | none, or `api_key=` for a hosted one |

Model names are examples; use any model your provider serves. If the key is missing, `decision()` raises a `ValueError` right away that says which variable it looked for. `shad0w doctor --llm openai/gpt-6-luna` checks the key without calling the model.

## API keys {#api-keys}

Pass the key itself, the name of the variable that holds it, or a function that fetches it:

```python
import os, shad0w

opts = ["refund", "lost_card", "other"]

# 1. the key itself
intent = shad0w.decision("intent", options=opts, llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])

# 2. the NAME of an environment variable (read on every request, so a rotated value is picked up)
intent = shad0w.decision("intent", options=opts, llm="openai/gpt-6-luna", api_key_env="SUPPORT_BOT_OPENAI_KEY")

# 3. a zero-argument function, called on every request (vaults, rotating keys)
intent = shad0w.decision("intent", options=opts, llm="openai/gpt-6-luna", api_key=lambda: my_vault.get("openai"))

# 4. once for the whole process
shad0w.configure(llm="openai/gpt-6-luna", api_key_env="SUPPORT_BOT_OPENAI_KEY")
intent = shad0w.decision("intent", options=opts)         # uses the defaults above
```

When several are set, the first match wins:

| # | Source | Example |
|---|---|---|
| 1 | `api_key=` on the call (a string or a function) | `api_key=os.environ["OPENAI_API_KEY"]` |
| 2 | `api_key_env=` on the call | `api_key_env="SUPPORT_BOT_OPENAI_KEY"` |
| 3 | `shad0w.configure(api_key=...)`, then `shad0w.configure(api_key_env=...)` | at app start-up |
| 4 | the `api_key_env` setting, from `SHAD0W_API_KEY_ENV` or `shad0w.toml` | `api_key_env = "OPENAI_API_KEY"` |
| 5 | the provider's usual variable | `OPENAI_API_KEY` (see the table above) |

<div class="viz" data-viz="keys">Where the key comes from, first match wins: api_key= on the call, then api_key_env= on the call, then shad0w.configure(), then the api_key_env setting (SHAD0W_API_KEY_ENV or shad0w.toml), then the provider's own variable such as OPENAI_API_KEY. Wherever it is shown, the key is masked as sk-…3f9a.</div>

**The key never leaks.** It does not appear in `repr()`, logs, traces, pickles, `shad0w doctor` or `shad0w config` output, or error messages. At most you see `sk-…3f9a`, enough to tell two keys apart. To check where a key came from:

```python
print(intent.teacher.key_source)     # set via SUPPORT_BOT_OPENAI_KEY (sk-…3f9a)
print(intent)                        # Shadow('intent', options=3, ..., llm=openai/gpt-6-luna, key: set via ... (sk-…3f9a))
```

**Rules that keep keys out of the wrong places:**

- `shad0w.toml` never holds a key. Put the variable's *name* there (`api_key_env = "OPENAI_API_KEY"`); an `api_key = ...` line is an error.
- `api_key`, `api_key_env` and `base_url` only apply to a `"provider/model"` string. With a function as `llm`, the function holds its own key, so passing them raises `TypeError`.
- A mistyped keyword raises `TypeError` with a suggestion: `apikey` → `did you mean 'api_key'?`.
- `shad0w.configure()` with no arguments clears every process-wide default.

In JavaScript it works the same way: `apiKey` (a string or `() => string | Promise<string>`), `apiKeyEnv`, and `configure({ llm, apiKey, apiKeyEnv, baseURL })`. See [JavaScript](javascript.html#keys). For the proxy, see [`--api-key-env` and `--api-key-file`](proxy.html#keys).

## Any other OpenAI-compatible server

Give the model name and the server's base URL:

```python
intent = shad0w.decision("intent", options=["refund", "lost_card", "other"],
                         llm="my-model", base_url="https://llm.internal/v1", api_key_env="INTERNAL_LLM_KEY")
```

`base_url` also overrides a known provider's URL (for example, a regional endpoint). To set it for every teacher at once, use `shad0w.configure(base_url=...)`, `SHAD0W_BASE_URL` or `base_url` in `shad0w.toml`. The first one set wins, in that order, before the provider's default.

## Decision models as the teacher

Decision models answer with a typed choice instead of text, so there is no reply to parse.

- `openai-decisions/<model>` calls `POST /v1/decisions` (the OpenAI Decisions API).
- `systemone/<model>` calls `POST /v1/systemone` on the server in `base_url` (`/v1` is added when missing).
- Yes/no questions are sent as `predicate` and `noul` questions respectively.

```python
import shad0w

intent = shad0w.decision("intent",
                         options={"lost_card": "card lost or stolen", "refund": "wants money back", "other": "anything else"},
                         llm="systemone/kev", base_url="http://127.0.0.1:8081")   # e.g. llama-server -hf ggml-org/Kev-0.8B-GGUF --port 8081
```

To use a decision model with no code change at all, put `shad0w proxy` in front of it instead: see [Decisions API & System One](decisions-api.html).

## Reuse an SDK client

Pass an `openai` SDK client (its proxies and custom auth then apply). shad0w calls `client.chat.completions.create(...)`:

```python
from openai import OpenAI
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm="gpt-6-luna", client=OpenAI())
```

## Anything else: give it a function

**A function that returns the reply text.** shad0w builds the prompt and maps the reply to an option, exactly as for a built-in provider:

```python
import shad0w

def complete(messages):                      # messages = [{"role": "system", ...}, {"role": "user", ...}]
    return my_llm(messages)                  # return the reply text, e.g. "lost_card" or '{"answer": "lost_card"}'

teacher = shad0w.llm_teacher(["refund", "lost_card", "other"], question="intent", complete=complete)
intent = shad0w.decision("intent", llm=teacher)
```

**A function that returns the answer.** No prompt, no parsing: whatever it returns is the answer (it should be one of your options):

```python
intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=lambda text: my_classifier(text))
```

This is also how you use LangChain, LiteLLM or any SDK: see [Frameworks](frameworks.html).

## LLM settings

These keywords go to `decision()`, `Shadow(llm=...)` or `shad0w.llm_teacher(...)`. They only apply to a `"provider/model"` string (or to `llm_teacher`).

| Keyword | Type | Default | What it does | Change it when |
|---|---|---|---|---|
| `api_key` | `str` or `() -> str` | see [API keys](#api-keys) | the key, or a function called on every request | you hold the key in code or a vault |
| `api_key_env` | `str` | the provider's variable | the *name* of the variable that holds the key | one app uses several keys |
| `base_url` | `str` | the provider's URL | an OpenAI-compatible server | self-hosted or proxied models |
| `client` | `openai.OpenAI` | none | send requests through this SDK client | you need the SDK's auth or transport |
| `complete` | `fn(messages) -> str` | none | send requests through your own function | any other SDK or framework |
| `system` | `str` | built from your options | replaces the system message | you have a tuned prompt |
| `temperature` | `float` | `0.0` | sampling temperature | rarely; dropped automatically if the model refuses it |
| `max_tokens` | `int` | `50` | reply length cap | long option names; dropped automatically if refused |
| `timeout` | `float` (seconds) | `30.0` | per attempt | slow local models |
| `retries` | `int` | `2` | extra attempts on rate limits, server errors and network errors | flaky upstreams |
| `headers` | `dict` | `{}` | extra HTTP headers on every request | gateways that need them |
| `structured` | `bool` | `True` | ask for JSON-schema structured output first | the server misbehaves with `response_format` |

## What shad0w sends

One chat request per decision: a system message listing your options (with their descriptions), then the user's text, at `temperature=0`. It asks for **structured output**: a JSON object whose `answer` must be one of your options. If the server does not support that, shad0w notices once and switches to plain text.

Every reply is mapped back to exactly one option. These all work:

- an exact match, or one that differs only in case or spacing (`Lost Card` → `lost_card`);
- an option named inside a short sentence (`The answer is lost_card.`), when exactly one option fits;
- `{"answer": "refund"}`, also inside a fenced code block.

If no option matches, the call raises `shad0w.TeacherError` and **nothing is logged**, so bad replies never become training data. The error says what the model replied and why it was asked.

Other built-in behaviour:

- **Retries.** Rate limits (429) and server errors (5xx) are retried with backoff, honouring `Retry-After`.
- **Rejected parameters.** If a model rejects `temperature` or `max_tokens` (as reasoning models do), that parameter is dropped and the request retried.

## Yes/no questions

```python
import shad0w

spam = shad0w.decision("spam", options={"type": "yesno", "instructions": "Is this message spam?"},
                       llm="openai/gpt-6-luna")
spam("WIN A FREE CRUISE, click now").answer      # True
```

`options=bool` does the same without instructions. The answer is a Python `bool`.
