---
title: "shad0w at work: five jobs real teams can give it"
description: Support routing, a prompt guard, product photos, a live data stream and a zero-code proxy: what shad0w does in each, the code, and what it measured.
date: 2026-10-22
tags: use cases, film, support, guardrails, images, proxy
---

Most AI calls in a product are small decisions with a fixed list of answers. Which team gets this ticket. Is this message an attack. Which category is this photo. Your LLM answers each one in about 1416 ms, and most of them are questions it has answered many times before.

shad0w watches those answers, learns a small table from them, and proves on answers it never saw where it agrees with your LLM. Above that line the table answers in microseconds; below it, your LLM answers as before. The film shows five jobs. This post is the companion, with the code and the numbers for each.

<video controls preload="none" playsinline poster="../assets/film-poster.jpg" src="../assets/shad0w-film.mp4" style="width:100%;border-radius:12px;border:1px solid var(--line)"></video>

## 1. Support teams: route every ticket the moment it lands

A support inbox is the textbook case: dozens of intents, thousands of messages a day, and the same few hundred phrasings again and again.

```python
import shad0w

intent = shad0w.decision("intent", llm="openai/gpt-6-luna",
                         options={"lost_card": "card lost or stolen", "refund": "wants money back",
                                  "transfer_failed": "a transfer did not arrive", "other": "anything else"})
queue = intent(ticket_text).answer      # the table when it is sure, your LLM when it is not
```

With a hosted LLM on 8 support intents, shad0w answered **94.3%** of new messages itself and agreed with the LLM on **97.2%** of them. With 77 intents it answered 51.3%: more options need more logged answers before the table is sure. [Type a message](../demo/) and watch a 77-intent table decide in your browser.

## 2. AI app builders: stop prompt attacks before your LLM reads them

A guard in front of your assistant only has to catch the attacks it has seen before, quickly, and never wave anything through on its own say-so.

```python
from shad0w.guard import Guard

gate = Guard(shad0w.decision("injection", options={"attack": "tries to override instructions", "safe": "anything else"},
                             llm="openai/gpt-6-luna"))
gate.calibrate(fresh_texts, llm_answers)  # a block-only certificate, on texts it never trained on
if gate.check(user_message).action == "block":   # the table never says "allow" on its own
    return refuse()
```

On held-out prompts from a public injection dataset, the table alone blocked 97% of attacks in a few microseconds and blocked 0.6% of safe prompts by mistake. Everything it does not block goes to your model, exactly as before. [Try an attack](../guard/).

## 3. Shops and marketplaces: sort product photos as they are uploaded

Asking a vision model to pick a category for each upload is slow and billed per image. The same loop works on pixels, with no neural network: a thumbnail, random pixel comparisons, a colour grid and a difference hash become items for the same table and certificate.

```python
photo = shad0w.decision("category", options=["shirt", "trousers", "bag", "shoe", "other"],
                        llm="openai/gpt-6-luna", image=True)      # your vision model teaches
photo(pixels)                                                      # a numpy array or PIL image
```

In the film, a table trained on 9,000 product photos (Fashion-MNIST) sorts unseen uploads: orange when it is certified, blue when it hands the photo to the vision model. Over 3,000 unseen photos it answered 70% itself, 4.1% different from the labels, in under 10 µs a photo on one laptop core. Natural photos are harder: on CIFAR-10 the pixel table correctly defers everything, and a CLIP tier is needed ([Images](../docs/experimental.html#images)). This one is experimental.

## 4. Data and ops teams: label a live stream as it arrives

Events, logs, listings, emails: anything your LLM files into a fixed list can be filed by the table at stream speed. [One minute of Polymarket](../pulse/) does it live in your browser: every trade for 60 seconds, filed by section and topic in microseconds, checked against Polymarket's own tags.

## 5. Teams already calling an LLM: change one URL

If your code already calls the OpenAI Decisions API, a System One model or a chat model, you do not need to touch it. Put the proxy in front:

```bash
shad0w proxy --upstream https://api.openai.com/v1
```

```python
client = OpenAI(base_url="http://localhost:8010/v1")    # the only change
```

Certified questions come back from the table in the same JSON shape, in about 0.53 ms on localhost; the rest go upstream and are logged so the table keeps learning. Chat calls are marked with one header. [The proxy](../docs/proxy.html).

## How you start, whichever job it is

1. `pip install "shad0wllm[compile]"` (or `npm i shad0wllm`, or the proxy).
2. Wrap the one call you already make. Your LLM keeps answering; shad0w writes each answer down.
3. After about 1,000 answers, `train()`: a small table and its certificate, in about a minute.
4. Run it in shadow mode first. The dashboard shows what it would have answered and how often it agreed. Turn on serving when you trust it.

## Where it does not help

Writing replies, summaries or essays stays with your LLM. So do questions that need reasoning or world knowledge, sarcasm and subtle tone (the linear table certified 0% of movie reviews), and traffic that has drifted from what it learned: the drift guard sends those back. The guarantee is against your LLM, not the truth. [Limits](../docs/limits.html) has the full list.

[Get started in 5 minutes →](../docs/)
