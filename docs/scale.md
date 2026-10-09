---
title: "Scale: options, size and speed"
description: How many options a shad0w table supports, how its size grows, how long training takes, and how to keep a large table small with max_mb.
---
# Scale: options, size and speed

This page tells you how big a question can get, how big its table gets, and what to change when it grows. "The table" is shad0w's small learned model; "options" are the answers a question can choose from.

**Short answers:**

- **Options:** 2 to 1,024 per question. Above that, split the question in two.
- **Speed:** microseconds per decision, from 10 to 500 options in our runs.
- **Size:** grows with (patterns learned) × (options). Cap it with `max_mb`.
- **Growth over time:** none on its own. Each training run replaces the table.

## Measured

Synthetic traffic, so only the option count changes: each option has its own distinctive words, the LLM stand-in is perfectly consistent, and the same certifier runs everywhere. Measured on 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39, 2026-10-08.

| options | logged answers | size budget | patterns kept | table size | training | p50 per decision | certified share | agreement |
|---|---|---|---|---|---|---|---|---|
| 10 | 400 | none | 2,267 | 0.03 MB | 4 s | 10.2 µs | 90.0% | 97.5% |
| 50 | 2,000 | none | 8,693 | 0.46 MB | 27 s | 7.1 µs | 100.0% | 98.3% |
| 150 | 6,000 | none | 24,026 | 3.62 MB | 5 min | 7.2 µs | 94.4% | 93.6% |
| 150 | 6,000 | 1 MB | 6,216 | 1.00 MB | 6 min | 7.6 µs | 94.4% | 93.7% |
| 500 | 15,000 | 4 MB | 8,322 | 4.97 MB | 21 min | 10.7 µs | 57.3% | 80.2% |

*Synthetic traffic with one consistent teacher; α = 5%. Training time is wall clock on the machine below. The 500-option row: measured before the 0.3.1 size-budget fix (the K x K matrix was not counted), so it overshoots its 4 MB budget.* Source: `results/b9_scale.json`.

What the columns mean:

- **options**: answers the question can choose from.
- **logged answers**: LLM answers the table was trained and certified on.
- **size budget**: the `max_mb` setting (`none` = no cap).
- **patterns kept**: word and character patterns stored in the table after training.
- **table size**: the `.s0` file on disk.
- **training**: wall-clock time of `train()`, including certification.
- **p50 per decision**: median time for one decision in Python with the C core.
- **certified share**: share of calibration traffic the table may answer at α = 5%.
- **agreement**: how often the table matches the LLM on all calibration traffic, served or not.

What it shows:

- **Speed barely moves with the option count.** A decision adds up one small row per pattern found in the message, with one byte per option. From 10 to 500 options, per-decision time stayed between 7.1 and 10.7 µs.
- **Size is about (patterns × options) bytes.** Each pattern stores one byte per option plus a 4-byte key. With 150 options and no budget, the table was 3.6 MB.
- **A size budget is nearly free.** With `max_mb=1`, the same 150-option table was 1.00 MB and certified the same share (94.4% vs 94.4%). Most patterns carry almost no weight. Training keeps the informative ones and refits on them, and the certificate is computed on the smaller table.
- **Training is the slow part.** Its time grows with options × patterns. The largest run, 500 options with a 4 MB budget (5.0 MB, 57.3% certified), took 21 minutes. That row was measured before a fix in 0.3.1 that counts the option-by-option matrix in the budget, which is why it overshot.

## How the size adds up

A table file holds:

| Part | Bytes |
|---|---|
| header | 20 |
| per pattern | 4 (key) + 1 per option |
| per option | 8 (scale and bias) |
| option × option matrix (used for the robustness radius) | 4 × options² |

The last part matters only for very large questions. At 1,024 options it alone is about 4.2 million bytes, so a `max_mb` budget must be larger than that. shad0w always keeps at least 1,000 patterns, whatever the budget.

## Does the table grow over time?

No. Each `train()` builds a new table from the log and replaces the old one. Its size follows the vocabulary of your logged traffic and the number of options, not how long you have been running.

Size grows when:

- users bring new wording (new products, phrasings, languages);
- you add options.

It levels off once the vocabulary settles, and `max_mb` caps it either way.

The **log** does keep growing: one line per LLM answer. Rotate or trim it as you would any log. A few tens of thousands of recent answers per question are plenty to train on.

## Recommendations

| Options per question | Suggested set-up |
|---|---|
| up to ~100 | defaults; tables are small |
| ~100–400 | `max_mb` between 1 and 4; log several thousand answers before expecting a large certified share |
| ~400–1,024 | `max_mb` between 4 and 16; train on a machine with a few GB of RAM; expect training to take tens of minutes |
| more than 1,024 | split into two questions: coarse first (`team`), then fine (`intent` within that team) |

A two-level split often certifies *more* than one huge question. The coarse question has few options and certifies early. Each fine question only has to tell a few dozen intents apart.

```python
import shad0w

team = shad0w.decision("team", options=["billing", "cards", "transfers"], max_mb=1)
cards = shad0w.decision("cards_intent", options=["lost_card", "card_arrival", "pin_blocked"], max_mb=1)
# route with team(text) first, then ask the matching fine question
```

## Memory

Training holds an options × patterns weight matrix in memory. Above about 10 million weights, shad0w keeps a shorter optimiser history. With a `max_mb` budget far below the vocabulary, it first keeps only the most frequent patterns, then fits. Both keep memory bounded on a laptop. Serving needs only the table itself.

## Many questions

Each question has its own table and its own certificate. Ten questions of 50 options each are ten small, fast tables, not one large one. Pass one `shad0w.Metrics()` to all of them to see them on one dashboard ([observability](observability.html)).
