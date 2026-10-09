---
title: Adding a category without breaking production
description: A trained shortcut and a changing list of options are a dangerous mix. The rule shad0w follows, and what happens when you add, remove or rename an option.
date: 2026-10-08
updated: 2026-10-09
tags: production, lifecycle, safety, release notes
---

Product adds a new intent, `transfer`, to the list your LLM chooses from. The LLM uses it from the first request. shad0w learns from your LLM's answers, so a shad0w trained last week on the old list cannot: it has never seen a transfer, so it picks its closest old label, sometimes with high confidence.

Before 0.3.1 we tested exactly this. More than half of the new option's messages went out with a wrong old label, marked certified. Every message about a removed option went out with the deleted label. The certificate was right about the world it was computed in. The world changed.

## The rule

From 0.3.1 on, shad0w compares the options you offer now with the options shad0w learned, on every load, and in the proxy on every request:

> shad0w only answers with options it learned, from a list you still offer, under the certificate it was given.

<div class="lifecycle" aria-label="Option lifecycle: add defers everything until retrain; remove defers only that option; rename applies at once">
  <div class="lc"><b>add</b><span>defer everything</span><small>shad0w never learned it</small></div>
  <div class="lc"><b>remove</b><span>never serve it</span><small>the rest keep serving</small></div>
  <div class="lc"><b>rename</b><span>applies at once</span><small>same decisions, new name</small></div>
  <div class="lc ok"><b>retrain</b><span>new certificate</span><small>serving resumes</small></div>
</div>

- **Add** an option and every decision goes to your LLM, flagged `options_changed`, until you retrain. All of them, because shad0w cannot tell which messages belong to the new option. The LLM's answers keep flowing into the log, so the next `train()` learns the full list.
- **Remove** one and shad0w keeps answering the rest. An answer that would use the removed option goes to your LLM, flagged `option_removed`.
- **Rename** one and nothing needs retraining:

```python
intent = shad0w.decision("intent", options=["refund", "card_lost", "balance"], llm=...,
                         rename={"lost_card": "card_lost"})
```

- **Retrain** and shad0w gets a new certificate and answers again.

## Why not keep serving?

You can: `on_new_option = "serve"`. It is not the default because shad0w cannot tell you *which* of its answers were really about the new option, and those come back confident and wrong. A day of extra LLM calls costs money. Silent wrong answers cost trust.

The JavaScript package and the proxy follow the same rules. [Changing your options](../docs/options.html) covers merging, splitting and checking where you stand.

<div class="callout"><b>Upgrade:</b> <code>pip install -U shad0wllm</code> · <code>npm i shad0wllm@latest</code>. The protection is on by default.</div>
