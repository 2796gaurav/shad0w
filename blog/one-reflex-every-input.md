---
title: "shad0w 0.4: one reflex for text, images and guards"
description: shad0w 0.4 takes its learned table and certificate from text to pixels, prompt-injection guards and video frames. What works, what doesn't, and what to try.
date: 2026-10-21
tags: release, images, guardrails, benchmarks, multimodal
---

Most AI calls in a product are small, fixed-choice decisions: which intent, which team, spam or not, which category is this photo, is this message an attack. They repeat. Until 0.3, shad0w handled that for text. In 0.4 the same loop runs on pixels, video frames and prompt-injection checks, and it gained a hidden layer that reads negation.

Everything below was measured. Each section also says what didn't work.

## Text: a table that reads "not good"

A linear table adds one score per word, so "not good" can never mean the opposite of "good". The new **deep table** keeps the same lookups but runs them through one small hidden layer. Training builds both and keeps the deep one only when it certifies more held-out rows.

- **Sentiment (SST-2):** 0% → **74%** of reviews answered with a certificate. Sentiment used to be the textbook "send it all to the LLM" case.
- **AG News topics:** 81% → 83%.
- **BANKING77 intents:** 81% → 87%.
- **Cost:** about 1–6 µs in C. A deep table has no exact robustness radius, so the prompt guard never uses it to allow anything.

## Against open decision models, on a CPU

The [Decision Bench](../bench/) puts shad0w next to five open decision models (Julia-1, Laya, Kev-0.8B, Decider-0.8B, GLiNER2.5-Decide) on six public tasks, with the same rows and the same 5% certifier for every system. On one CPU thread shad0w answers in 2–10 µs; the decision models take 14 ms to 5.6 s on the CPU. shad0w was the more accurate on 4 of 6 tasks. Laya won on news topics and sentiment, and put behind shad0w it got both faster and more accurate. The models answer zero-shot while shad0w learned each task, and on a GPU the models would be far faster. Every prediction is saved, so you can re-score it without running anything.

## Images: pixels in, a certified label out, no neural network

An image becomes items the way a sentence does. A 32×32 thumbnail, 256 random pixel comparisons, a coarse grid of where the ink is and a perceptual hash each give the table a word to read. In C that takes 4–7 µs per image.

**[Doodle Duel](../doodle/)** lets you feel it. Draw, and the table guesses on every stroke in your browser. It learned from 72,000 Quick, Draw! doodles, finished and half-drawn.

- **Finished doodles it never saw:** it named 87.9% correctly, certified 75.1%, and 97.6% of its certified answers were right.
- **"I know!" mid-drawing:** right 88.3% of the time. That's lower than on finished drawings, because guessing after every stroke gives it more chances to be wrong. We say so on the page.

**Where it doesn't work.** On natural photos (CIFAR-10) the pixel table certifies nothing and passes everything to your vision model, as it should. An optional CLIP tier certified 73% there, at about 28 ms instead of a vision model's 2.6 s.

## Guardrails: block in microseconds, never allow

Many prompt-injection attempts are copies of copies. **[The prompt guard](../guard/)** blocks the ones it recognises, with a certificate, in microseconds. Everything else goes to your model, including anything that merely *looks* safe.

**On 2,060 held-out prompts from a public dataset:** it blocked 97.4% of attacks and blocked 0.6% of safe prompts by mistake.

**Why it never allows.** Try the encoded and full-width examples on the page. The bare table reads them as safe, even certified. A guard that let traffic through on the table's word would wave them in. In the library, `Guard` also normalises the input first (zero-width characters, homoglyphs, leetspeak, base64), and anything it had to undo goes to your model. It catches repeats cheaply; it is not a defence against an adaptive attacker on its own.

## Video frames and structured fields (experimental)

- **Video frames.** Adjacent frames usually carry the same answer. A small table decides whether the answer still holds since the last keyframe. On one surveillance dataset that reuse answered about two thirds of frames at 2–4% disagreement. But only in 2 of 3 splits, and only at a block size chosen after the fact. The pre-registered setting certified nothing, mostly because the teacher model contradicts itself on identical frames. Measure your teacher's flicker first.
- **Structured fields.** One certified table per field fills what it is sure of and asks your LLM for the rest. Per-field guarantees are not record-level, so a record-level option splits α across fields: 60% of records were served whole, with 5.2% having any wrong field.

## Try it

Everything in the [playground](../playground/) runs in your browser: the doodle game, the prompt guard, a live minute of Polymarket trades, the 77-intent message box, and the games.

```bash
pip install "shad0wllm[compile]"   # or: npm i shad0wllm
```

The [changelog](../docs/changelog.html) has every change, and [the docs](../docs/deep-table.html) explain the deep table and [the experimental layers](../docs/experimental.html).
