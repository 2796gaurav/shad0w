---
title: The deep table
description: Since 0.4 a shad0w table can read word interactions such as negation, with one hidden layer and the same microsecond lookups. When it helps, and its costs.
---
# The deep table

<p class="lead">A linear table adds one score per word, so "not good" can never mean the opposite of "good". The deep table keeps the same lookups but turns each row into a small hidden vector, then applies a ReLU and one dense layer. It learns negation and word pairs, and still answers in microseconds.</p>

**TL;DR**: you don't have to do anything. With `student = "auto"` (the default) training builds both and keeps the deep table only when it certifies more held-out rows. It made sentiment, the classic weak spot, work.

## What changed, measured

Measured through the full pipeline at α = 5% (labels as the teacher):

| task | linear table: certified | deep table: certified | disagreement on served | deep table | C p50 / p99 |
|---|---|---|---|---|---|
| SST-2 sentiment (32k rows) | 0% | **74%** | 4.3% | H = 32, 2.7 MB | 1 / 5 µs |
| AG News topics (29k rows) | 81% | **83%** | 4.5% | H = 32, 5.7 MB | 6 / 13 µs |
| BANKING77 intents (77 options) | 81% | **87%** | 3.0% | H = 128, 3.1 MB | 4 / 13 µs |
| prompt-injection set (LLM labels) | 97.0% | 98.1% | 3.3% | auto kept linear | |

## How it works

Each word or n-gram of the input is hashed to a row, exactly as before. In a linear table a row holds one score per option and the rows are added. In a deep table a row holds `H` numbers (32 to 128, chosen from the number of options); they are added, passed through a ReLU, and one small dense layer turns them into option scores. That one non-linearity is what lets "not" change the meaning of "good".

It is file format 2. The C core, numpy and JavaScript read it and compute the same scores bit for bit: deep tables are summed in the C core's order everywhere. Only the final exp is each platform's own, so a probability can differ in its last bit; answers and certified flags match.

## Choosing

| `student` | What you get |
|---|---|
| `"auto"` (default) | Both are trained; the held-out fitting slice is split in two, one half tunes, the other judges which certifies more. The certificate records the choice. |
| `"linear"` | Always the additive table, with its exact robustness radius |
| `"deep"` | Always the deep table |

```python
intent = shad0w.decision("sentiment", options=["positive", "negative"], llm="openai/gpt-6-luna", student="auto")
```
```bash
shad0w train --dir shad0w --question sentiment --student deep --hidden 64
```

## What it costs

- **Time.** A deep row is `H` bytes instead of one per option, so long texts cost more: about 1–4 µs on short texts and 6 µs on news articles in C.
- **Size.** `max_mb` caps a deep table by keeping its most frequent items, but that costs coverage: SST-2 fell from 74% to 58.5% at 0.7 MB.
- **No exact radius.** A deep table reports no robustness radius, so exposed mode defers it and the [prompt-injection guard](experimental.html#guardrails) never allows on it. Set `student = "linear"` where you need the radius.

## Also new in 0.4

- **Script-aware text** (the default for new bundles). Every Chinese and Japanese character becomes its own item, plus character bigrams; Latin-1, Greek and Cyrillic capitals are folded. ASCII hashes exactly as before, and older bundles keep their old text handling everywhere. Japanese intents (MASSIVE-ja): 30.7% → 52.0% certified; Japanese news 93.0% → 95.3%.
- **Unseen words.** Every answer now reports `unseen`, the share of the input's items the table has never seen. `max_unseen = 0.3` sends mostly-unseen inputs (junk, another language) to your LLM.
- **`distil(texts)`**. `sh.distil(texts)` asks your LLM about a corpus you supply and logs the answers as fit-only rows: extra context for the deep table on nuanced questions, never used for the certificate.
- **Faster training.** Linear tables take their temperature from a held-out slice: 2 fits instead of 6 (AG News 218 s → 100 s, BANKING77 140 s → 65 s, same or better coverage).
- **Folded word vectors.** `static="potion-8M"` now folds the word vectors into the table rows exactly, so serving needs no extra file. Mixed results (a small SST-2 log 0% → 15% certified, BANKING77 84% → 79%), so it stays opt-in.

See also: [Decision Bench](../bench/), where `shad0w-auto` chose the deep table on three of six tasks (AG News, BANKING77, CLINC150).
