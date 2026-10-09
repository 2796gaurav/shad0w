---
title: Changing your options
description: What happens when you add, remove or rename an option after training, and how shad0w keeps serving safely until it relearns.
---
# Changing your options

<p class="lead">Products change: you add an intent, retire a team, rename a label. This page shows what the table does after each kind of change, and the one command that brings it back to full speed.</p>

**TL;DR.** Change the `options` list in your code and keep running. shad0w notices the difference with what the table learned and plays safe until you retrain. A rename needs no retraining at all.

```python
import os
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "balance", "transfer"],   # "transfer" is new
                         llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])
print(intent.options_added)     # ('transfer',): every call goes to your LLM (and is logged) until you retrain
# ... later, once your LLM has answered enough "transfer" messages:
intent.train()                  # learns all four options and serves again under a fresh certificate
```

**The table** is the small model shad0w trains from your LLM's answers; **certified** means tested on answers it never trained on and allowed to serve only within α disagreement. The rule on this page: **the table only answers with options it learned, from a list you still offer, under the certificate it was given.** Everything else goes to your LLM, and your LLM's answers keep being logged so the next training run learns the new list.

<div class="viz" data-viz="options">Add an option: the table cannot know it, so every decision goes to your LLM (flag options_changed) until you retrain. Remove an option: the table keeps answering with the others; anything it would label with the removed option goes to your LLM (flag option_removed). Rename an option: the table keeps answering, using the new name, with no retraining.</div>

## At a glance

| You… | What the table does until you retrain | Flag on deferred decisions |
|---|---|---|
| **add** an option | Nothing. Every decision goes to your LLM. | `options_changed` |
| **remove** an option | Keeps answering with the options that remain. Never answers with the removed one. | `option_removed` |
| **rename** an option (with `rename=`) | Keeps answering, using the new name. | — |
| **retrain** | Learns the current list, gets a new certificate, and serves again. | — |

The examples below assume a table trained on `refund`, `lost_card` and `balance`.

## Adding an option

```python
intent("send 50 to my sister")
# Decision(answer='transfer', source='teacher', flag='options_changed', ...)   your LLM answered, and it was logged
```

Why everything goes to your LLM, and not only messages about the new option: the table cannot recognise a message about something it has never seen. It would label "send 50 to my sister" with its closest old option, possibly with high confidence. In our tests, before this protection existed, a table served more than half of a new option's messages with an old label. The certificate was earned on a world without the new option, so it no longer covers any answer.

Nothing is wasted while you wait: every LLM answer, including the new option, goes to the log. Retrain once the new option has at least 20 logged answers (shad0w warns below that):

```python
intent.train()
```

`auto_train` does this for you if you turned it on. The retrain gate never blocks this case: an old table that answers a different question is not a baseline for the new one.

**Prefer to keep serving the old options?** `on_new_option = "serve"` keeps the table answering with what it knows. Messages about the new option reach your LLM only when the table happens to be unsure. The certificate no longer describes those answers (they come back with `certified=False` and the flag `new_options_served`), so use this only when a short dip in agreement costs less than a burst of LLM calls.

## Removing an option

```python
import os
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card"],      # "balance" retired
                         llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])
print(intent.options_removed)    # ('balance',)
```

Anything the table would answer as `balance` goes to your LLM (flag `option_removed`), which now picks among the options you still offer. Everything else keeps being served: removing an option doesn't change what your LLM says about the others. At the next training run, old log rows labelled `balance` are left out automatically.

## Renaming an option

```python
import os
import shad0w

intent = shad0w.decision("intent", options=["refund", "card_lost", "balance"],
                         llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"],
                         rename={"lost_card": "card_lost"})
print(intent("my card was stolen").answer)    # 'card_lost', straight from the table, no retraining
```

A rename is the one change that needs no new certificate: the table's decisions are the same, only the name differs. The next `intent.train()` rewrites the old name in the log as it reads it, so no history is lost. The same in `shad0w.toml`:

```toml
[questions.intent]
rename = { lost_card = "card_lost" }
```

From the command line, pass a schema that lists the new name, so the old bundle's option list is not reused:

```bash
shad0w train --log shad0w/intent/log.jsonl --out shad0w/intent/bundle --rename lost_card=card_lost
```

## Merging and splitting

- **Merge two options**: rename both to the same new name **and** list only the merged name in `options`. The table answers with the merged name at once; retrain soon so the certificate describes the merged option.
- **Split one option into two**: that adds at least one option, so the table defers until it relearns from your LLM's new answers.

## Through the proxy

With the OpenAI Decisions API or System One, each request lists its options, and the proxy compares them with the table's on every request. Add a choice to your request and that question goes to the upstream model until the table is retrained; nothing else in your code changes. On the chat path, the options come from your JSON-schema enum, a forced tool's enum, or the `X-Shad0w-Options` header. See [The proxy](proxy.html).

## In JavaScript

```js
import { decision } from "shad0wllm";

const intent = await decision("intent", {
  options: ["refund", "card_lost", "balance", "transfer"], llm: "openai/gpt-6-luna",
  bundle: "shad0w/intent/bundle", rename: { lost_card: "card_lost" },
});
console.log(intent.optionsAdded);   // ["transfer"]: every decision goes to the LLM until the bundle is retrained
```

`onNewOption: "serve"` is the JavaScript spelling of `on_new_option = "serve"`.

## Checking where you stand

```bash
shad0w stats --log shad0w/intent/log.jsonl    # answers per option, and whether there are enough to train
shad0w status                                 # every decision: rows, certified share, next step
```

In Python, `intent.stats()` includes `options_added` and `options_removed`. The dashboard counts the `options_changed` and `option_removed` flags under "why deferred". Until a new option has enough examples the table rarely answers it, which is safe: those messages go to your LLM.
