---
title: "Can a table drive a browser? We tested computer use from raw HTML"
description: An LLM web agent ran a shopping errand 300 times. A shad0w table learned its steps from the pages' HTML and replayed 100 new errands with zero LLM calls. Then the site changed, success fell to 44%, and two cheap checks brought it back to 100%. On websites it had never seen, it failed, and said so.
date: 2026-10-21
tags: agents, computer use, experiments, guardrails
---

Computer-use agents are slow and expensive because every step is a big-model call: look at the page, pick the next click, wait a second or two, repeat. Yet a lot of agent work is the same errand on the same site, again and again.

So we asked a narrow question: **if an agent only reads a page's HTML (no screenshots), can a shad0w table take over the steps it has seen many times, and hand the rest back?** We tested three things, and wrote down where each one broke. You can [run the first two in your browser](../webagent/).

<figure class="post-video">
<video controls playsinline preload="metadata" poster="../assets/webagent-demo-poster.jpg" src="../assets/webagent-demo.mp4" aria-label="Screen recording of the web agent playground: the table runs an errand, gets stuck when the site changes, and finishes once the novelty check is on"></video>
<figcaption>71 seconds, recorded from the <a href="../webagent/">playground page</a>. The shop is on the left; on the right is the text the agent reads from the page's HTML. Orange steps are the table (certified, microseconds); blue steps are handed to the LLM. In the browser the LLM's turns are played by a script; the numbers below are from the real LLM.</figcaption>
</figure>

## 1. A repeated errand on one site: it works

We built a small shop website with the things that trip agents up, randomised on every visit:

- a cookie banner;
- a login wall at checkout;
- out-of-stock variants, where the agent must stop and report;
- a declined payment that needs a retry;
- search results in a different order every time.

The errand is "buy 2 desk lamps in brass and ship them to Pune". The agent reads each page as plain text built from the HTML (title, heading, fields and whether they match the task, buttons) and picks one action from a fixed list.

An LLM agent ran **300 errands, 3,284 steps**, at about **2 seconds a step**, and got every errand right. A shad0w table learned from those steps: **82 KB**, certified at α = 5%. Then we ran 100 errands neither of them had seen, three ways:

| | errands done right | LLM calls per errand | steps answered by the table |
|---|---|---|---|
| LLM only | 100% | 10.99 | 0% |
| shad0w, LLM when not certified | 100% | 0 | 100% |
| same, but "Place order" always goes to the LLM | 100% | 1.06 | 90% |

About **11 µs a step** in Python instead of about 2 seconds. On a site you know, the errand becomes almost free.

## 2. Then the site changed: the certificate didn't save it

Websites change. We added a newsletter pop-up after the first search and a terms checkbox that must be ticked before an order goes through. We did not retrain the table.

**Success fell from 100% to 44%.** The table kept answering "certified" and kept clicking a product hidden under the pop-up, or pressing "Place order" without the terms, until it gave up after 25 steps.

This is the most useful result here. A certificate says the table rarely disagrees with its teacher **on pages like the ones it was tested on**. A changed site isn't that, and the table can be wrong and certain.

Our first fix failed too. "If the page's HTML didn't change, ask the LLM" never fired, because the results list reshuffles and the suggested products change on every render, so the HTML always looked new. Real sites do this with ads, timestamps and recommendations.

Two checks did work:

- **Progress check:** compare a summary of the page that leaves out link lists (title, heading, dialogs, messages, fields, buttons). If a table action changed nothing, the next step goes to the LLM.
- **Novelty check:** if the page has an element or pop-up the table never saw in training, the LLM decides.

| on the changed site | 100 errands | 100 fresh errands |
|---|---|---|
| no checks | 44% right | – |
| progress check | 100% right, 1.58 LLM calls each, 89% of steps by the table | 100%, 1.78, 89% |
| novelty check | 100%, 3.32, 73% | 100%, 3.61, 72% |
| both | 100%, 3.32, 73% | 100%, 3.60, 72% |

We designed the checks after seeing the first 100 failures, so we ran them again on 100 fresh errands. With both checks on and the site unchanged, the table still answered every step with zero LLM calls: the checks cost nothing until something changes.

One caution: the progress check only catches wrong actions that change nothing. A wrong action that does something (ordering the wrong item) would pass it. Keep irreversible actions such as paying, deleting or sending behind the LLM or a person.

## 3. Websites it has never seen: it fails, and knows it

The hard part of computer use is a new website: which of ~600 elements should I click? We used the public [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) training set: **1,009 tasks, 7,775 steps, 73 websites**. We held out whole tasks (on websites seen in training) and whole websites.

The table scored every element from its tag, attributes, text and the task words it shares:

| | right element first time | in its top 50 |
|---|---|---|
| new tasks, known websites | 12.4% | 65.6% |
| websites it never saw | 6.8% | 58.5% |

A plain word-overlap rule gets 8.9%. **The certificate refused to serve a single step**, even with a 20% error budget. That refusal is correct: a hashed table should not be choosing elements on sites it hasn't learned.

We also asked an LLM to label 4,911 pages by type (search results, form, product page, …) and trained a table on its labels. With a 10% error budget it agreed with the LLM on 69% of pages from known sites (certifying 27% of them, at 5.1% disagreement) and on 51% of pages from new sites, where always guessing the most common type gets 41%. On new sites its few certified answers missed their 10% error budget (11.8%, on 17 pages). Page type across many different websites is too fuzzy for a table, and certificates don't travel to new websites.

## What this means

- **Works:** a repeated workflow on a site you know (RPA-style errands, back-office forms, test scripts). Record it with your LLM agent, then replay it at table speed, and hand anything unusual back.
- **Doesn't:** finding the way on websites it hasn't learned. That stays the big model's job.
- **Always add the two checks.** A certificate alone doesn't protect you when the site changes.
- **Limits of this test:** it is one synthetic shop. The LLM made no mistakes on it, so this shows a table copying a reliable agent, not fixing a bad one. The Mind2Web splits come from its public training set, not its official test sets.

[Run the errand in your browser](../webagent/): it's the same table, and the page text is checked to match the Python version on 1,412 pages. Tick "The site changed" and watch it fail, then turn on a check.
