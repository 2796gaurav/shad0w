---
title: System Zero: the reflex layer below your LLM
description: shad0w is the shadow of your LLM. It watches your LLM, learns to answer like it, and takes the repeat traffic, so the LLM only gets what shad0w is not sure about. What System Zero is, why a cache is not enough, and where shad0w stays silent.
date: 2026-10-08
updated: 2026-10-09
tags: system zero, architecture, decision models, llm cost
---

shad0w is the shadow of your LLM. It watches your LLM answer, and learns to answer the same way. Most repeat traffic then goes to shad0w instead of the large LLM. Your LLM only gets what shad0w is not sure about.

That is a reflex. Pull your hand off a hot stove and you move before you feel it. The signal never reaches the part of your brain that thinks. It goes to your spinal cord and straight back out.

Your LLM stack has no reflex, so every repeat question takes over a second.

## Three layers

<div class="layers" aria-label="Three layers: System 2 reasons in seconds, System 1 decides in milliseconds, System 0 answers repeats in microseconds">
  <div class="layer l2"><b>System 2</b><span>the LLM thinks</span><em>1416 ms per call</em></div>
  <div class="layer l1"><b>System 1</b><span>a decision model picks</span><em>271 ms (Kev-0.8B, local)</em></div>
  <div class="layer l0"><b>System 0</b><span>shad0w answers what it has learned</span><em>1.0 µs in C · 2.4 µs in Python</em></div>
  <i class="pulse p1"></i><i class="pulse p2"></i><i class="pulse p3"></i>
</div>

- **System 2 is the general LLM.** You describe the task in a prompt and it reasons its way to an answer. Flexible and slow: 1416 ms for one fixed-choice decision from a hosted LLM in our runs.
- **System 1 is the decision model.** OpenAI's Decisions API, TypeSafe's Jev, and open models such as Kev and Laya take a question and a fixed list of answers, and return a probability for each in one pass. Tens to hundreds of milliseconds.
- **System 0 is the reflex.** It does not reason. It remembers how the layers above answered questions like this one, and answers at once, but only when it can prove the answer will match.

Systems 2 and 1 both compute every answer from scratch. Ask either one "which intent is *my card got stolen*?" a million times and you pay for a million computations of the same thing.

## How much of your traffic is a repeat?

Look at what apps send to their models. A support bot routes messages into a few dozen intents. A moderation step asks "spam or not?" An agent asks "which tool next?" between steps. These are the same few questions, about wording that repeats.

We measured three intent and topic tasks, with a hosted LLM as the model shad0w learned from. shad0w answered **61–86%** of the traffic by itself, in microseconds. The other 14–39% still went to the LLM, as it should.

## Why a cache is not enough

A cache only knows exact repeats. Real users rarely type the same sentence twice. We put an exact-match cache and a semantic cache (nearest logged answer by embedding) through the same safety check as shad0w. The check uses α, the most disagreement you allow: α = 5% means the shortcut may differ from your LLM on at most 1 in 20 answers it gives. Both caches answered 0%: neither could prove it stayed within that limit.

## Why a fine-tune is too much

You could fine-tune a classifier instead. It can even answer more: a small head on an embedding model passed the check on 75% of traffic in our test, against 61% for shad0w. But it needs an embedding model at run time, takes 7.2 ms per decision, and is one more model to own, label for and retrain. A reflex should be smaller than the thing it protects. shad0w is 1.5 MB for 77 options and runs in a browser tab.

## Defer by default

A reflex that fires at the wrong time is worse than none. So System Zero follows four rules:

1. **It learns from the layer above.** No labelling project. Your LLM or decision model is the teacher.
2. **It only answers what it can prove.** A certificate limits how often it disagrees with your LLM. Everything else goes up.
3. **It lives where the request is.** In your process, a browser, an edge worker. No network hop.
4. **Every pass upward teaches it.** Each answer your LLM gives becomes training data, so the share shad0w answers grows.

The default is to defer. Serving is the exception that has to be earned.

## The numbers

- A decision costs 1.0 µs in C, 10 µs in JavaScript and 2.4 µs in Python (5.2 µs with per-option probabilities).
- We set α = 5%: shad0w may differ from your LLM on at most 1 in 20 of the answers it gives, a promise made with 90% confidence. How often shad0w actually gave a different answer than your LLM in our runs: 3.0–6.0%.
- In front of a decision model it still pays: with Kev-0.8B as the teacher, the average time per decision fell from 382 ms to 115 ms, and accuracy moved +1.0 points.

The [honest numbers](honest-numbers.html) post has every result, including the losses.

## Where shad0w stays silent

- **Tone.** On SST-2 sentiment it answered 0%. shad0w reads words, not sarcasm, so everything went to the LLM. That is correct.
- **Long, varied text.** On LEDGAR legal clauses it answered 13%.
- **Cold start.** With 77 options it still answered nothing after 2,000 logged answers. It got going somewhere before 5,000.
- **Shifted traffic.** When users start asking new kinds of questions, the guarantee no longer applies. shad0w watches its own confidence and sends everything to your LLM until it has relearned.

It also cannot write a reply or answer a question that needs knowledge of the world. That is what the layers above are for.

## shad0w is a System Zero

shad0w is a tiny model of hashed words and characters that it learns from your LLM's answers. It certifies itself with Learn-then-Test, and runs the same model in Python, JavaScript and C. A proxy puts it in front of chat models, the OpenAI Decisions API and System One servers with no code change.

<div class="callout"><b>Try the reflex.</b> The <a href="../demo/">playground</a> runs real shad0w models in your browser. Type a message and watch the decision arrive before your finger leaves the key.</div>
