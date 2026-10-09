---
title: Get started in 5 minutes
description: Put shad0w in front of your LLM, let it learn, and serve certified answers in microseconds.
---
# Get started in 5 minutes

<p class="lead">shad0w learns the answers your LLM gives to a small, fixed-choice question (which intent, which team, spam or not). Once it has seen enough of them, it answers the ones it is <b>certified</b> on by itself, from a 1.5 MB table, in about 2.4 µs. Everything else still goes to your LLM.</p>

This page takes you from `pip install` to your first answer served by the table. It needs Python 3.10 or newer.

<div class="viz" data-viz="flow">A message comes in. shad0w looks it up in its table. If the table is sure (certified), it answers in microseconds and your LLM is never called. If it is not sure, the message goes to your LLM as before, and the LLM's answer is logged so the next training run can learn it.</div>

## TL;DR

```python
# pip install "shad0wllm[compile]"
import os, shad0w

intent = shad0w.decision("intent",
                         options=["refund", "lost_card", "balance", "other"],
                         llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])

print(intent("someone stole my card at the airport").answer)   # lost_card  (your LLM answered, it was logged)
# ... after about 1,000 calls:
intent.train()                                                  # the table now answers what it is certified on
```

Three words you will see everywhere:

| Word | Meaning |
|---|---|
| **the table** | shad0w's small learned model. It lives in a folder (the *bundle*) and needs no GPU and no network. |
| **certified** | the table was tested on answers it never trained on, and only answers where it disagrees with your LLM on at most α of them (α = 5% by default, at 90% confidence). See [The guarantee](guarantee.html). |
| **defer** | the table is not sure enough, so the message goes to your LLM instead. |

## 1. Install

```bash
pip install "shad0wllm[compile]"
shad0w doctor                       # checks Python, the C core, training extras and your API key
```

The package installs as `shad0wllm`, imports as `shad0w`, and adds a `shad0w` command. It runs on Linux, macOS and Windows.

| Install | What you get | When |
|---|---|---|
| `pip install shad0wllm` | serving only (numpy + the bundled C core) | a server that only loads a trained table |
| `pip install "shad0wllm[compile]"` | + training (scipy, scikit-learn) | you call `train()` |
| `pip install "shad0wllm[torch]"` | + torch, for faster training | logs with tens of thousands of rows |

<div class="callout tip"><b>Already using the OpenAI Decisions API, or a decision model such as Jev, Kev or Laya?</b> You can skip the code on this page: run <code>shad0w proxy</code> and move your client's <code>base_url</code>. See <a href="decisions-api.html">Decisions API &amp; System One</a>.</div>

## 2. Wrap the question you already ask your LLM

```python
import os, shad0w

intent = shad0w.decision(
    "intent",                                   # the decision's name: also its folder, shad0w/intent/
    options={
        "refund":    "the customer wants money back",
        "lost_card": "a card is lost or stolen",
        "balance":   "the customer asks about their balance",
        "other":     "anything else",
    },
    llm="openai/gpt-6-luna",
    api_key=os.environ["OPENAI_API_KEY"],      # or leave it out: OPENAI_API_KEY is read by default
)

d = intent("someone stole my card at the airport")
print(d.answer, d.source)                       # lost_card teacher
print(d.why)                                    # no trained table yet: asked your model
```

`intent(...)` returns a `Decision`. `d.answer` is the option. `d.source` says who answered: `"teacher"` (your LLM) or `"table"`. `d.why` says why, in plain words. All fields are listed in [Python](python.html#decision).

On day one there is no table yet, so every call goes to your LLM. shad0w writes each `(text, answer)` pair to `shad0w/intent/log.jsonl`. That log is the training data.

<div class="callout tip"><b>Always include an <code>other</code> option.</b> The table can only answer with one of your options. Without a catch-all, an off-topic message ("tell me a joke") gets mapped to the closest real intent. With <code>other</code>, it lands there, or goes to your LLM.</div>

**Not on OpenAI?** Use `anthropic/…`, `gemini/…`, `groq/…`, `ollama/llama3.1`, a decision model (`systemone/…`, `openai-decisions/…`) or any OpenAI-compatible server. See [Connect your LLM](connect-your-llm.html), which also covers every way to pass the key.

**No key at hand?** Any function that returns one of the options works as the LLM. This runs offline:

```python
import shad0w

def my_llm(text):                                # stands in for your real LLM call
    return "lost_card" if "card" in text else "other"

intent = shad0w.decision("intent", options=["refund", "lost_card", "balance", "other"], llm=my_llm)
print(intent("my card is gone").answer)          # lost_card
```

## 3. Train once about 1,000 answers are logged

```python
result = intent.train()        # needs shad0wllm[compile]; compiles, certifies, then serves from the new table
print(result["accepted"], result["certified_share_on_calibration"], result["threshold"])
```

```text
True 0.68 0.91          # example output: the table may answer 68% of messages like the ones it was tested on
```

`train()` holds back part of the log, trains on the rest, and certifies on the held-back part. It returns the certificate; the fields are explained in [Training](training.html). From now on:

```python
intent("my card was stolen")
# Decision(answer='lost_card', source='table', confidence=0.99, certified=True, flag=None, latency_us=9.6, ...)

intent("can you explain compound interest like I'm five?")
# Decision(answer='other', source='teacher', confidence=0.41, certified=False, flag='low_confidence', ...)
```

The second message went to your LLM because the table was not sure enough (`flag='low_confidence'`). That answer is logged too, so the next `train()` knows more. To retrain automatically, pass `auto_train=1000` (every 1,000 new answers).

## 4. Look at what it is doing

```bash
shad0w try --bundle shad0w/intent/bundle "my card was stolen" "explain interest"   # what the table answers, and why
shad0w stats --log shad0w/intent/log.jsonl                                          # what has been logged; ready to train?
shad0w status                                                                       # one line per decision folder
```

```python
print(intent)          # Shadow('intent', options=4, serving: certifies 68.0% at alpha=0.05, llm=openai/gpt-6-luna, key: ...)
intent.stats()         # {'table': 812, 'teacher': 388, 'offload': 0.68, 'audit_disagreement_upper': 0.041, ...}
```

`offload` is the share of calls the table answered. `audit_disagreement_upper` comes from spot checks: shad0w re-asks your LLM about 1% of the table's answers in the background and measures how often the two disagree. See [Monitoring](observability.html) for the live dashboard, Prometheus metrics and hooks.

Try a trained table yourself. This one knows 77 banking intents:

<div class="viz" data-viz="try">Interactive demo: type a banking message and see the table's top choices, its confidence, and whether it would answer or send the message to the LLM.</div>

## What next

- **Roll it out safely:** [shadow → canary → serve](rollout.html).
- **Tune it:** α, spot checks, labels that must never be served: [Configuration](configuration.html).
- **Change no code at all:** run the [OpenAI-compatible proxy](proxy.html) and point your client's `base_url` at it.
- **Have old LLM answers already?** Import them with [`warm_start`](python.html#warm-start) and train today.
- **Something not working:** `shad0w doctor`, then [Troubleshooting](troubleshooting.html).
- **JavaScript, browsers, edge:** [JavaScript](javascript.html).
- **Any language:** [HTTP](http.html) or [C](c.html).
- **LangChain, LiteLLM, Vercel AI SDK, FastAPI:** [Frameworks](frameworks.html).
- **Keep it fresh:** [Training & retraining](training.html).
- **What "certified" means exactly:** [The guarantee](guarantee.html).
