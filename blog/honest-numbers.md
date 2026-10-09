---
title: 94.3%, 51.3%, 0%: honest numbers from shadowing a hosted LLM
description: We put shad0w in front of a hosted LLM through the official OpenAI client and measured how much shad0w answered, how often it agreed with the LLM, how fast, and where it answered nothing at all.
date: 2026-10-09
tags: benchmarks, evaluation, comparison, honesty
---

Benchmarks for developer tools usually measure the happy path on the author's laptop. We tried to do better:

- a real hosted LLM, called through the official OpenAI client and the proxy, exactly as an app would;
- public datasets;
- every competing method put through the **same** safety check;
- the losses published next to the wins.

Every figure here comes from a result file in the [benchmark kit](../docs/benchmarks.html) and is inserted by the site build. None is typed by hand.

The three numbers in the title: shad0w answered 94.3% of fresh traffic on a task with 8 intents, 51.3% on a task with 77 intents, and 0% on sentiment. The last one is the right answer.

One term first. **α** is the most disagreement you allow. α = 5% means shad0w may differ from your LLM on at most 1 in 20 answers it gives. All runs here use α = 5%.

## How much shad0w answers depends on how many options you have

The set-up a user would have: the official `openai` SDK talks to `shad0w proxy`, which talks to the LLM. We logged the LLM's answers, trained shad0w, then sent fresh messages shad0w had never seen. To check agreement, we asked the LLM again directly.

- **8 intents.** shad0w learned from 1,000 logged answers. It answered 94.3% of fresh traffic, and agreed with the LLM on 97.2% of those. The LLM took 1416 ms per call.
- **77 intents.** shad0w learned from 4,999 logged answers. It answered 51.3% of fresh traffic, and agreed with the LLM on 95.9% of those. The LLM took 1397 ms per call.
- **8 intents, local Ollama model as the LLM.** shad0w answered 96.0% and agreed on 97.6%.
- **Decisions API shape.** shad0w answered 92.7% and agreed on 98.2%. This is not the real Decisions API. It is a local mock of its wire format, to check that the request and response shapes round-trip.

With 8 intents shad0w answered nearly everything. With 77 intents it answered about half. More options means fewer messages shad0w is sure about, and more logged answers before it is sure at all. shad0w's answers came back in milliseconds at the client, instead of 1397 ms from the LLM.

## It starts at zero

On BANKING77 (77 options), shad0w answered 0% of traffic after 2,000 logged answers, 40% after 5,000, and 60% after 10,000.

This is the honest version of "set and forget". A task with a few options gets going far sooner. Until then, shad0w just logs and forwards. It costs you nothing, and it saves nothing.

## Speed spans almost six orders of magnitude

Time for one decision, middle value of many runs, on an idle laptop CPU:

- shad0w in C: 1.0 µs. In Python: 2.4 µs. In JavaScript: 10 µs.
- shad0w through the proxy, as the openai SDK sees it: 0.53 ms, almost all of it HTTP.
- Decision models on the same machine: Laya 421M 91.6 ms, Kev-0.8B 271 ms.
- The hosted LLM: 1416 ms.

## Sentiment: 0% is correct behaviour

On SST-2, shad0w answered 0% and sent every message to the LLM. shad0w reads words, not sarcasm. On its own it gets about 79% of messages right, nowhere near good enough to promise it agrees with the LLM. The safety check saw that and refused to let shad0w answer. Saving nothing is the right outcome when the alternative is silent wrong answers.

## How often shad0w differed from the LLM: 3.0–6.0% against a 5% limit

We set α = 5%: among the answers shad0w gives, it may differ from the LLM on at most 1 in 20. On fresh traffic, how often shad0w actually gave a different answer than the LLM was 3.0–6.0% across tasks. So yes, one task went over 5%.

That is allowed, and here is why. The promise holds with 90% confidence. It is computed on a sample, and a sample can be kind. Roughly one promise in ten may overshoot by design. Our simulation found 0.0–8.1% of certificates exceeded α in simulation (δ allows 10%).

The bigger cause is traffic shift. On CLINC150 the test traffic had about ten times the off-topic share of the logged traffic. The promise covers "traffic like the traffic shad0w was checked on", and this was not. In live use the drift guard flags this after a few hundred decisions and sends those messages to the LLM. The benchmark runs without the guard, so it shows the unguarded number. Lesson: log off-topic traffic at its real rate and always offer an `other` option.

## Against a semantic cache, under the same test

On BANKING77 we gave shad0w and its alternatives the same LLM answers to learn from, and the same safety check at α = 5%. Everyone had to prove their answers, not just make them.

- **shad0w answered 61%** of traffic. How often it actually gave a different answer than the LLM: 3.0%. Cost: $0.204 per 1,000 decisions, against $0.520 for calling the LLM every time.
- **A semantic cache answered 0%.** Its nearest-neighbour answers looked plausible but could not prove they stayed within α. An exact-match cache also answered 0%.
- **An embedding classifier answers more.** A small head on bge-small answered 75%, against 61% for shad0w. It costs 7.2 ms per decision and an embedding model at run time. If coverage is all you want, use one, and put the same safety check on it.
- **Local decision models replace your LLM instead of shadowing it.** Kev-0.8B takes 271 ms per decision and Laya 421M 91.6 ms. They give probabilities, but no limit on how often they differ from your LLM.
- **shad0w answers in microseconds, anywhere.** No model at run time, 1.5 MB for 77 options. That buys browser and edge use.

## What we did not measure

- Hosted Jev, Kev-4B and pplx-decider: we had no credits.
- The real OpenAI Decisions API: only a local mock of its wire format.
- Long-running production traffic. Every run here is a benchmark replay.
- Inconsistent LLMs in depth. A small local model that answered the same message differently on different days capped what shad0w could learn. shad0w is only as consistent as the LLM it shadows.

## Reproduce

The benchmark kit has setup and run scripts and a results template. Every figure on this site rebuilds from its output:

```bash
bash bench/setup.sh && bash bench/run_all.sh
```

Commands, hardware and the source file behind every number are on the [benchmarks page](../docs/benchmarks.html).

<div class="callout"><b>Run it on your own traffic.</b> <code>shad0w proxy --upstream https://api.openai.com/v1 --mode shadow</code> answers nothing itself and counts what shad0w would have done. See <a href="decisions-api-proxy.html">zero-code System Zero</a>.</div>
