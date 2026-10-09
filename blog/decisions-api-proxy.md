---
title: Zero-code System Zero for the OpenAI Decisions API
description: Put a certified reflex in front of client.decisions.create with three commands and no code change. Shadow mode first, the dashboard, the agreement check, the cost maths and the limits.
date: 2026-10-09
tags: decisions api, proxy, zero code, llm cost
---

OpenAI's Decisions API is already a System 1: you send a question and a fixed list of answers, and `gpt-6-luna` returns a choice with probabilities. It is fast and cheap per call. But it still computes every answer from scratch, including the ten-thousandth "my card was stolen".

`shad0w proxy` adds the reflex below it. Each Decisions API request already names its question and lists its answers, so there is nothing to mark. You move `base_url` and change nothing else.

## The proxy in three commands

```bash
pip install shad0wllm
shad0w proxy --upstream https://api.openai.com/v1      # dashboard at http://localhost:8010
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

Your API key is forwarded to OpenAI as usual. To keep it in the proxy instead, add `--api-key-env OPENAI_API_KEY`.

At first shad0w has learned nothing. Every request goes to OpenAI and the answer is logged per question. Once a question has about 1,000 logged answers, train it:

```bash
shad0w train --dir shad0w                                # the running proxy picks up the new shad0w
```

Or let the proxy do it with `--auto-train 1000`. Training needs `pip install "shad0wllm[compile]"`.

## Watch first: shadow mode

You do not have to trust shad0w on day one. Start the proxy with `--mode shadow`:

```bash
shad0w proxy --upstream https://api.openai.com/v1 --mode shadow
```

Now shad0w decides every call, but OpenAI's answer is always the one returned. shad0w counts two things: `would_serve`, the calls shad0w would have answered, and `shadow_disagreements`, how many of those differed from OpenAI. Read them at `/v1/stats` or on the dashboard. Deferred calls carry the flag `shadow`.

Move to serving when `shadow_disagreements / would_serve` sits at or below your α over a few thousand calls. α is the most disagreement you allow: 5% means shad0w may differ from OpenAI on at most 1 in 20 answers it gives. The [rollout guide](../docs/rollout.html) adds a canary step in between.

## What you get back

The response has the same shape whether OpenAI or shad0w answered: an `answers` list with `choice`, `confidence` and `probabilities` per question. shad0w's answers also carry a `shad0w` object (`source`, `certified`, `flag`), and every response has an `x-shad0w-source` header: `table` when shad0w answered, `teacher` when OpenAI did, or `mixed`.

`mixed` is the interesting one. If a request asks several questions and only some are certified, only the rest go to OpenAI, in one smaller request. The answers are merged back in the original order.

## The dashboard

Open http://localhost:8010. It updates every second: share answered by shad0w, LLM calls saved, time saved, shad0w vs LLM speed per question, why shad0w deferred to your LLM, and the live safety bound. That bound comes from background spot checks (`--audit-rate`, 1% by default) and shows a ⚠ with "re-train" when it rises above α. Add `--cost-per-call` to see money saved. A built-in playground lets you type a text and see shad0w's answer and confidence.

## Agreement, re-checked

We ran the Decisions API shape end to end: the openai SDK, the proxy, an upstream, then fresh messages shad0w had never seen. Agreement was checked by asking the upstream again directly.

shad0w answered 92.7% of fresh traffic and agreed with the upstream on 98.2% of those. Be clear about what this is: the upstream was a **local mock of the Decisions API wire format**, not OpenAI. It proves the request and response shapes round-trip and the certifier works through them. It does not tell you how much of *your* traffic gpt-6-luna's answers will let shad0w answer. Shadow mode tells you that, on your own traffic, for free.

Every run is in [honest numbers](honest-numbers.html).

## Cost maths at a million calls a day

Start from list prices and measured rates, then do the multiplication yourself:

- gpt-6-luna lists at about $0.030 per 1,000 decisions. Multiply by 1,000 for a million calls a day. With the proxy, you pay that only on the share shad0w passes to OpenAI.
- For a chat LLM the saving is larger. Shadowing a hosted LLM on BANKING77, calling the LLM for everything cost $0.520 per 1,000 decisions; with shad0w in front it cost $0.204. Again, multiply each by 1,000 for a million a day.
- shad0w's own answers cost nothing per call. They run on the CPU you already pay for.

Against a decision model the case is narrower, and we say so. The money is small per call. The stronger reasons are latency (microseconds, no network hop) and a written bound on disagreement.

## Limits

- **`score` questions always go upstream.** shad0w does not learn ordinal scales.
- **Cold start.** Each question needs about 1,000 logged answers, and more with many options.
- **Changing options.** Add an option and every decision for that question defers until you retrain. See [adding a category](adding-a-category.html).
- **Traffic shift.** The bound covers traffic like what was logged. The drift guard steps back when that changes.
- **No authentication.** The proxy has none. Keep it on localhost or inside your network.
- **Not measured against the real API.** See above.

## Vercel AI SDK and Pydantic AI

**Pydantic AI and other decision-model clients:** point the client's base URL at the proxy. `/v1/decisions` and `/v1/systemone` requests are recognised with no marking.

**Vercel AI SDK:** no proxy needed. With `experimental_decide`, use shad0w as the model and gpt-6-luna as the fallback:

```ts
decisionModel(intent, { fallback: openai.decisionModel("gpt-6-luna") })
```

shad0w answers the questions it certifies and the fallback answers the rest. See [Frameworks](../docs/frameworks.html).

## FAQ

**Does it work with Jev, Kev or Laya?** Yes. They speak `/v1/systemone`, which the proxy recognises the same way. Point `--upstream` at the System One server. See [Decisions API & System One](../docs/decisions-api.html).

**Can I run it against llama.cpp?** For `/v1/systemone`, yes. llama.cpp does not implement `/v1/decisions`, so a Decisions API request shad0w cannot answer comes back as the server's 404, unchanged.

**How do I turn it off?** Drop `decisions` from the capture list (`--capture header,model,tools`), or run with `--mode off`.

**Is shad0w compared with the truth?** No. It is compared with the model it learned from. If gpt-6-luna is wrong, shad0w is wrong the same way.

<div class="callout"><b>Start in shadow mode.</b> <code>pip install shad0wllm</code>, then <code>shad0w proxy --upstream https://api.openai.com/v1 --mode shadow</code>. You lose nothing and learn what the reflex would save.</div>
