---
title: Benchmarks & evidence
description: Every number shad0w quotes, with the dataset, the LLM, the hardware, the date and the result file behind it, plus where shad0w loses.
---
# Benchmarks & evidence

This page shows every measurement behind the numbers on this site: what was run, on what data, against which LLM, and what each column means. It also lists where shad0w loses. Every figure comes from `numbers.json`, which the benchmark kit generates from the result files named under each table.

**Which version was measured.** The result files were produced with shad0w 0.3.0 on 8 October 2026. Before the 0.3.2 release the latency runs, the certificate simulation and the BANKING77 and CLINC150 comparisons were repeated with 0.3.2 on the same machine; they matched within noise, and JavaScript came out a little faster than the figure shown here.

Hardware: Apple M5 Pro (15 cores, 24 GB), macOS, unless a section says otherwise.

## How to read these numbers

A few terms used in every table:

- **The table**: shad0w's small learned model, trained on the LLM's logged answers.
- **α (alpha)**: the most disagreement with the LLM you allow among the answers the table gives. The default, 5%, is 1 in 20.
- **δ (delta)**: the chance that a certificate is wrong because of an unlucky calibration sample. The default δ = 0.1 means 90% confidence.
- **Answered by the table** (or "served"): the share of *fresh* traffic, never seen in training or calibration, that the table answered without the LLM.
- **Realised disagreement**: on that fresh traffic, how often the table's answer differed from the LLM's. It may exceed α in up to δ of certifications ([why](guarantee.html)).
- **Accuracy vs gold**: agreement with the dataset's human labels. It is reported separately because **agreement with the LLM is not accuracy**: the table copies the LLM, mistakes included.
- **Cascade**: the table plus the LLM for everything the table defers. Cascade accuracy is what your users get.

## Headline: the same certifier for every method

**Setup.** BANKING77 (77 intents). The LLM, gpt-4.1-mini (OpenAI), labelled 10,000 messages; those answers are the log. Every method was trained on the same log, certified with the **same** procedure at α = 5% and δ = 0.1, and tested on the same 3,080 fresh messages, over 3 seeds.

| method | answered without the LLM @ α = 5% | disagreement with the LLM on those | accuracy vs gold (cascade) | p50 per decision | cost per 1k decisions | needs | guarantee |
|---|---|---|---|---|---|---|---|
| shad0w table | 61% | 3.0% | 76.5% | 4.0 µs | $0.204 | your LLM's logged answers | certified ≤ α vs your LLM |
| semantic cache (bge-small, nearest logged answer) | 0% | — | 76.3% | 7.5 ms | $0.520 | embedding model + logged answers | same certifier applied |
| embedding kNN router (bge-small, k=10) | 69% | 2.9% | 76.9% | 7.6 ms | $0.161 | embedding model + logged answers | same certifier applied |
| small fine-tuned head (LR on bge-small) | 75% | 3.4% | 77.1% | 7.2 ms | $0.129 | embedding model + logged answers | same certifier applied |
| exact-match cache | 0% | — | 76.3% | 1.2 µs | $0.520 | logged answers | exact repeats only |
| keep calling the LLM | 0% | 0.0% | 76.3% | — | $0.520 | API or GPU | it is the reference |
| Kev-0.8B decision model (local) | 100% (it replaces your LLM) | 23% vs gpt-4.1-mini | 80.0% | 271 ms | $0 (your hardware) | a GPU/MLX machine, ~1-3 GB | calibrated probabilities, no bound |
| Julia-1 decision model (local) | 100% (it replaces your LLM) | 41% vs gpt-4.1-mini | 61.6% | 11.9 ms | $0 (your hardware) | llama.cpp, a laptop | calibrated probabilities, no bound |
| Laya 421M decision model (local) | 100% (it replaces your LLM) | 58% vs gpt-4.1-mini | 38.0% | 91.6 ms | $0 (your hardware) | a GPU or fast CPU, 0.8 GB | calibrated probabilities, no bound |
| hosted decision model (OpenAI Decisions API, gpt-6-luna) | 100% (it replaces your LLM) | not measured (no API key) | not measured | ~150 ms claimed | $0.030 per 1k | API key | none (probabilities only) |
| hosted decision model (TypeSafe Jev 1.13) | 100% (it replaces your LLM) | not measured (no credits) | not measured | 70–500 ms claimed | $0.0126 per 1k | API key | none (probabilities only) |

Source: `results/b6_compare_openai_banking77.json` (learned methods) and `results/b7_systemone.json` (local decision models). Hosted decision-model rows use vendor-stated latency and prices.

What each column means:

- **answered without the LLM @ α = 5%**: share of fresh traffic the method answered itself, with its certified threshold. Decision models answer everything, because they replace your LLM.
- **disagreement with the LLM on those**: realised disagreement on the answers it gave. For decision models: how often they differ from gpt-4.1-mini (OpenAI) over all messages.
- **accuracy vs gold (cascade)**: method plus LLM fallback, against human labels. The LLM alone scores the "keep calling the LLM" row.
- **p50 per decision**: median time of the method itself, not counting LLM calls for deferred messages.
- **cost per 1k decisions**: LLM spend per 1,000 messages for the share still sent to the LLM ($0.520 per 1,000 when every message goes to it). Local models cost only your hardware.
- **needs** and **guarantee**: what you must run, and what bound you get.

On 3 intent and topic tasks (BANKING77, CLINC150, AG News) the table answered 61–86% of fresh traffic at α = 5%, with realised disagreement of 3.0–6.0%.

## The α sweep

**Setup.** The same BANKING77 run, at four values of α. "n/a" means nothing was served, so there is nothing to measure.

| α (max disagreement) | answered by the table | realised disagreement |
|---|---|---|
| 1% | 0% | n/a |
| 2% | 20% | 0.4% |
| 5% | 61% | 3.0% |
| 10% | 80% | 7.4% |

*BANKING77. Lower α = safer, fewer calls saved.* Source: `results/evidence.json cmp.banking77.openai.shad0w`.

A lower α buys safety with calls: at the strictest setting the table answers nothing.

## It grows with traffic

**Setup.** The same BANKING77 run, trained on the first *N* logged answers, at α = 5%.

| LLM answers logged | answered by the table @ α = 5% |
|---|---|
| 250 | 0% |
| 500 | 0% |
| 1,000 | 0% |
| 2,000 | 0% |
| 5,000 | 40% |
| 10,000 | 60% |

*BANKING77.* Source: `results/b6_compare_openai_banking77.json`.

With 77 options, the table certified nothing until it had seen a few thousand answers. Simpler questions start earlier: on AG News (4 options) it began serving after about 1,000.

## Latency per runtime

**Setup.** The 77-intent BANKING77 table, single thread, median (p50) of many calls. Decision-model and LLM rows are measured on the same machine for comparison.

| runtime | p50 per decision |
|---|---|
| shad0w · C | 1.0 µs |
| shad0w · JavaScript | 10 µs |
| shad0w · Python | 2.4 µs |
| shad0w · via the proxy (HTTP) | 0.53 ms |
| Laya 421M (decision model) | 91.6 ms |
| Kev-0.8B (decision model) | 271 ms |
| gpt-4.1-mini (OpenAI) | 1416 ms |

Source: `results/b3_latency.json`, `results/b7_systemone.json`, `results/b4_e2e_openai_gpt-4.1-mini_8.json`.

- **C**: the bundled C core, called directly.
- **Python**: `Model.decide` through the C core, without per-option probabilities. With them it is 5.2 µs; without the C core (numpy only) 44 µs.
- **JavaScript**: `Bundle.decide` in Node, one question.
- **via the proxy**: a full HTTP round trip from the openai SDK to `shad0w proxy` and back on localhost, for a message the table answers.

## End to end through the proxy

**Setup.** A real app path: the openai Python SDK → `shad0w proxy` → the LLM. The proxy logs the LLM's answers, trains, then answers fresh messages. Agreement is checked by re-asking the LLM directly.

| run | LLM p50 | answered by the table (fresh traffic) | agreement with the LLM on those | answers logged | fresh messages |
|---|---|---|---|---|---|
| gpt-4.1-mini (OpenAI) · 77 intents | 1397 ms | 51.3% | 95.9% | 4,999 | 1,000 |
| gpt-4.1-mini (OpenAI) · 8 intents | 1416 ms | 94.3% | 97.2% | 1,000 | 300 |
| local Ollama model · 8 intents | 1960 ms | 96.0% | 97.6% | 1,000 | 300 |
| Decisions API shape (mock upstream) | 174 ms | 92.7% | 98.2% | 1,000 | 300 |

*The openai SDK → shad0w proxy → the LLM, on this machine; agreement re-asks the LLM directly.* Source: `results/b4_*.json`.

- **LLM p50**: median LLM latency seen by the client.
- **answered by the table (fresh traffic)**: share of new messages answered by the proxy without calling the LLM.
- **agreement with the LLM on those**: how often those answers match what the LLM says when asked directly.
- **answers logged / fresh messages**: training log size and test size.

In the latency benchmark, a table answer through the proxy reached the client in 0.53 ms. The LLM's median in the 77-intent run was 1397 ms.

## Decision models head to head

**Setup.** Decision models answer typed questions directly, instead of your LLM. Here each one answered the same BANKING77 prompts. The last column asks: if we put the same certifier on each, how much could it serve while staying within 5% of the reference LLM?

| engine | p50 | p99 | accuracy vs gold | answered @ α = 5% (same certifier) |
|---|---|---|---|---|
| shad0w (in process) | 0.01 ms | 0.01 ms | 74.1% | 55% |
| shad0w (HTTP /v1/systemone) | 0.25 ms | 0.29 ms | 74.1% | 55% |
| Kev-0.8B (llama.cpp) | 271 ms | 508 ms | 80.0% | 0% |
| Julia-1 (llama.cpp) | 11.9 ms | 13.3 ms | 61.6% | 0% |
| Laya 421M (Ollama) | 91.6 ms | 299 ms | 38.0% | 0% |
| Laya 421M (llama.cpp) | 42.4 ms | 47.3 ms | 37.9% | 0% |

*BANKING77, 77 options, the same 2,000 test prompts for every engine, teacher gpt-4.1-mini (77% accurate). Decision models answer everything themselves; the last column is what the same certifier would let them serve while staying within 5% of gpt-4.1-mini.* Source: `results/b7_systemone.json`.

The local decision models are more different from the reference LLM than α allows, so the certifier lets none of them serve on its own. They are still useful as the LLM shad0w learns from: with Kev-0.8B as the teacher, the table answered 70% of BANKING77 traffic.

## At game speed

A game frame is a hard deadline. In the [snake arena](../arena/), shad0w (trained on a hosted LLM's answers), three System One decision models running on the Apple M5 Pro's 16-core GPU (Laya, Kev-0.8B, Kev-4B) and the hosted LLM played the same 8 seeded games at every speed from one move a second to 120. A move lands on the first tick after its answer arrives; until then the snake keeps going straight. Every decision was a real call, timed as it happened.

- shad0w scored 28.0 food per game at every speed, deciding in about 1.6 µs per move from a 2 KB table.
- The hosted LLM scored 25.9 with no clock, and 0.1 at 10 moves a second (about 912 ms per answer).
- Laya (about 16 ms) and Kev-0.8B (about 23 ms) make most frames up to 30 moves a second; their low scores come from how they play without examples, not from time (compare the no-clock column).

The arena page replays every game and lists memory, model size and cost per lane. Pong, learned the same way from 3,000 answers, decides in about 1.5 µs per frame (Node on the Apple M5 Pro) and is [playable in the browser](../pong/). Sources: `arena/results/arena_summary.json`, `arena/results/pong_eval.json`; how to rerun: `arena/README.md`.

## Certificate validity by simulation

**Setup.** 1,000 calibration sets per row, drawn from known populations, for each procedure, α and calibration size. "flat" means the LLM makes random mistakes regardless of the message; "rising" means disagreement rises as the table's confidence falls. The column that matters is **share of certificates above α**: it must stay at or below δ = 10%. "upper CI" is the upper end of a 95 percent confidence interval on that share, given 1,000 draws.

| procedure | population | α | calibration n | share of certificates above α | upper CI |
|---|---|---|---|---|---|
| auto | flat | 1% | 1000 | 0.0% | 0.0% |
| auto | flat | 1% | 2000 | 0.0% | 0.0% |
| auto | flat | 1% | 3000 | 0.0% | 0.0% |
| auto | flat | 1% | 500 | 0.0% | 0.0% |
| auto | flat | 2% | 1000 | 1.3% | 2.0% |
| auto | flat | 2% | 2000 | 1.0% | 1.6% |
| auto | flat | 2% | 3000 | 0.5% | 0.9% |
| auto | flat | 2% | 500 | 1.5% | 2.2% |
| auto | flat | 5% | 1000 | 0.0% | 0.0% |
| auto | flat | 5% | 2000 | 0.0% | 0.0% |
| auto | flat | 5% | 3000 | 0.0% | 0.0% |
| auto | flat | 5% | 500 | 0.0% | 0.0% |
| auto | flat | 10% | 1000 | 0.0% | 0.0% |
| auto | flat | 10% | 2000 | 0.0% | 0.0% |
| auto | flat | 10% | 3000 | 0.0% | 0.0% |
| auto | flat | 10% | 500 | 0.0% | 0.0% |
| auto | rising | 1% | 1000 | 0.0% | 0.0% |
| auto | rising | 1% | 2000 | 0.8% | 1.4% |
| auto | rising | 1% | 3000 | 1.1% | 1.8% |
| auto | rising | 1% | 500 | 0.0% | 0.0% |
| auto | rising | 2% | 1000 | 1.8% | 2.6% |
| auto | rising | 2% | 2000 | 2.6% | 3.6% |
| auto | rising | 2% | 3000 | 2.4% | 3.4% |
| auto | rising | 2% | 500 | 0.0% | 0.0% |
| auto | rising | 5% | 1000 | 3.1% | 4.2% |
| auto | rising | 5% | 2000 | 3.3% | 4.4% |
| auto | rising | 5% | 3000 | 4.5% | 5.8% |
| auto | rising | 5% | 500 | 2.4% | 3.4% |
| auto | rising | 10% | 1000 | 0.0% | 0.0% |
| auto | rising | 10% | 2000 | 0.0% | 0.0% |
| auto | rising | 10% | 3000 | 0.0% | 0.0% |
| auto | rising | 10% | 500 | 0.0% | 0.0% |
| bonferroni | flat | 1% | 1000 | 0.0% | 0.0% |
| bonferroni | flat | 1% | 2000 | 0.0% | 0.0% |
| bonferroni | flat | 1% | 3000 | 0.0% | 0.0% |
| bonferroni | flat | 1% | 500 | 0.0% | 0.0% |
| bonferroni | flat | 2% | 1000 | 0.2% | 0.5% |
| bonferroni | flat | 2% | 2000 | 0.1% | 0.3% |
| bonferroni | flat | 2% | 3000 | 0.0% | 0.0% |
| bonferroni | flat | 2% | 500 | 0.3% | 0.6% |
| bonferroni | flat | 5% | 1000 | 0.0% | 0.0% |
| bonferroni | flat | 5% | 2000 | 0.0% | 0.0% |
| bonferroni | flat | 5% | 3000 | 0.0% | 0.0% |
| bonferroni | flat | 5% | 500 | 0.0% | 0.0% |
| bonferroni | flat | 10% | 1000 | 0.0% | 0.0% |
| bonferroni | flat | 10% | 2000 | 0.0% | 0.0% |
| bonferroni | flat | 10% | 3000 | 0.0% | 0.0% |
| bonferroni | flat | 10% | 500 | 0.0% | 0.0% |
| bonferroni | rising | 1% | 1000 | 0.0% | 0.0% |
| bonferroni | rising | 1% | 2000 | 0.1% | 0.3% |
| bonferroni | rising | 1% | 3000 | 0.1% | 0.3% |
| bonferroni | rising | 1% | 500 | 0.0% | 0.0% |
| bonferroni | rising | 2% | 1000 | 0.0% | 0.0% |
| bonferroni | rising | 2% | 2000 | 0.0% | 0.0% |
| bonferroni | rising | 2% | 3000 | 0.0% | 0.0% |
| bonferroni | rising | 2% | 500 | 0.1% | 0.3% |
| bonferroni | rising | 5% | 1000 | 0.5% | 0.9% |
| bonferroni | rising | 5% | 2000 | 0.2% | 0.5% |
| bonferroni | rising | 5% | 3000 | 0.2% | 0.5% |
| bonferroni | rising | 5% | 500 | 0.3% | 0.6% |
| bonferroni | rising | 10% | 1000 | 0.0% | 0.0% |
| bonferroni | rising | 10% | 2000 | 0.0% | 0.0% |
| bonferroni | rising | 10% | 3000 | 0.0% | 0.0% |
| bonferroni | rising | 10% | 500 | 0.0% | 0.0% |
| fixed_sequence | flat | 1% | 1000 | 0.1% | 0.3% |
| fixed_sequence | flat | 1% | 2000 | 0.1% | 0.3% |
| fixed_sequence | flat | 1% | 3000 | 0.0% | 0.0% |
| fixed_sequence | flat | 1% | 500 | 0.0% | 0.0% |
| fixed_sequence | flat | 2% | 1000 | 3.5% | 4.6% |
| fixed_sequence | flat | 2% | 2000 | 4.2% | 5.4% |
| fixed_sequence | flat | 2% | 3000 | 4.0% | 5.2% |
| fixed_sequence | flat | 2% | 500 | 4.2% | 5.4% |
| fixed_sequence | flat | 5% | 1000 | 0.0% | 0.0% |
| fixed_sequence | flat | 5% | 2000 | 0.0% | 0.0% |
| fixed_sequence | flat | 5% | 3000 | 0.0% | 0.0% |
| fixed_sequence | flat | 5% | 500 | 0.0% | 0.0% |
| fixed_sequence | flat | 10% | 1000 | 0.0% | 0.0% |
| fixed_sequence | flat | 10% | 2000 | 0.0% | 0.0% |
| fixed_sequence | flat | 10% | 3000 | 0.0% | 0.0% |
| fixed_sequence | flat | 10% | 500 | 0.0% | 0.0% |
| fixed_sequence | rising | 1% | 1000 | 2.1% | 3.0% |
| fixed_sequence | rising | 1% | 2000 | 3.3% | 4.4% |
| fixed_sequence | rising | 1% | 3000 | 2.3% | 3.2% |
| fixed_sequence | rising | 1% | 500 | 0.0% | 0.0% |
| fixed_sequence | rising | 2% | 1000 | 2.9% | 3.9% |
| fixed_sequence | rising | 2% | 2000 | 4.8% | 6.1% |
| fixed_sequence | rising | 2% | 3000 | 4.2% | 5.4% |
| fixed_sequence | rising | 2% | 500 | 4.2% | 5.4% |
| fixed_sequence | rising | 5% | 1000 | 6.8% | 8.4% |
| fixed_sequence | rising | 5% | 2000 | 5.7% | 7.1% |
| fixed_sequence | rising | 5% | 3000 | 8.1% | 9.8% |
| fixed_sequence | rising | 5% | 500 | 5.4% | 6.8% |
| fixed_sequence | rising | 10% | 1000 | 0.0% | 0.0% |
| fixed_sequence | rising | 10% | 2000 | 0.0% | 0.0% |
| fixed_sequence | rising | 10% | 3000 | 0.0% | 0.0% |
| fixed_sequence | rising | 10% | 500 | 0.0% | 0.0% |

*δ = 0.1 allows up to 10%.* Source: `results/b8_certificate_sim.json`.

Result: 0.0–8.1% of certificates exceeded α in simulation (δ allows 10%). The default is `auto`.

## Earlier runs (0.1, embedding-model teacher)

**Setup.** Earlier release, earlier certificate procedure, with an embedding model or Kev-0.8B as the teacher instead of an LLM. Kept for the wider range of tasks.

| task | teacher | answered by the table @ α = 5% | realised disagreement | max over seeds | cascade acc − teacher acc |
|---|---|---|---|---|---|
| BANKING77 · Kev-0.8B teacher | Kev-0.8B (local GPU, 77 options) | 71% | 5.7% | 6.5% | +0.8 |
| BANKING77 | EmbeddingGemma-300m zero-shot | 71% | 4.1% | 4.7% | +0.2 |
| HWU64 | EmbeddingGemma-300m zero-shot | 69% | 5.4% | 5.8% | -0.8 |
| CLINC150 | EmbeddingGemma-300m zero-shot | 65% | 4.9% | 5.1% | +0.2 |
| MASSIVE | EmbeddingGemma-300m zero-shot | 64% | 4.9% | 5.7% | -0.7 |
| DBpedia | EmbeddingGemma-300m zero-shot | 54% | 4.0% | 4.3% | +0.4 |
| AG News | EmbeddingGemma-300m zero-shot | 45% | 3.9% | 4.1% | +0.4 |
| LEDGAR (legal clauses) | EmbeddingGemma-300m zero-shot | 13% | 4.3% | 4.5% | +0.0 |
| SST-2 (sentiment) | EmbeddingGemma-300m zero-shot | 0% | 0.0% | 0.0% | +0.0 |
| toxic (yes/no) | EmbeddingGemma-300m zero-shot | 0% | 0.0% | 0.0% | +0.0 |
| tweet sentiment | EmbeddingGemma-300m zero-shot | 2% | 6.3% | 18.9% | -0.1 |

*Earlier runs (pre-0.1.1 certificate procedure), 3 seeds, 7k fit / 3k calibration rows. Two tasks landed above α on the mean; the bound allows that in up to δ = 10% of certifications.* Source: `legacy/benchmarks/results/phase31_fidelity.json`.

"cascade acc − teacher acc" is the change in accuracy against gold, in points, from putting the table in front of the teacher.

## Where shad0w loses

- **Other learned methods served more on BANKING77.** The kNN router answered 69% and the fine-tuned head 75%, against the table's 61%. Both run an embedding model on every message (7.2 ms vs 4.0 µs) and ship a much larger model. If milliseconds are fine and you already host embeddings, they are a reasonable choice.
- **CLINC150: realised disagreement above α, because the traffic mix moved.** CLINC150's test split has far more out-of-scope messages than its training split, and the training split is what the log looks like here. The certificate was computed on mostly in-scope traffic, then served a test set full of off-topic messages. Realised disagreement went above α on every seed (the top of the 3.0–6.0% range), and 5% of out-of-scope messages were answered with a real intent. This is the "traffic like the calibration traffic" condition failing, not a bug: calibrated on a test-like mix, the bound held again at a lower served share. In a live deployment the drift guard flags this shift after a few hundred decisions and defers those messages; this benchmark compares methods without the guard, so it shows the unguarded number. What to do: [limits](limits.html#it-assumes-traffic-like-the-calibration-traffic).
- **AG News: two of three seeds landed slightly above α at α = 5%** (5.1% and 5.6%; the third 3.5%), though the mean stayed below it. This is the δ allowance at work, and why spot checks exist.
- **Sentiment (SST-2): 0% served.** The table tops out near 79% accuracy on sentiment, so it never certifies and defers everything.
- **Legal clauses (LEDGAR): 13% served** in an earlier run. Long texts where one clause decides are hard for a word-pattern table.
- **A noisy LLM is hard to copy.** With a small local model (qwen3.5 4B through Ollama) as the LLM on 77 intents, the table certified little or nothing: when the LLM is inconsistent, no threshold keeps disagreement under α.
- **Accuracy is capped by your LLM.** Kev-0.8B alone scored 80.0% against gold, more than the table-plus-gpt-4.1-mini (OpenAI) cascade (76.5%). The table can only be as accurate as the model it copies.
- **Cold start.** Nothing is served until enough answers are logged and `train()` has run (see the growth table above).

## Reproduce

```bash
bash bench/setup.sh && bash bench/run_all.sh      # writes results/*.json, results/SUMMARY.md, results/evidence.json
cd shad0w-studio && python make_numbers.py --check && python build_site.py
```

The OpenAI steps skip themselves when `OPENAI_API_KEY` is unset. `BENCHMARK.md` in the kit lists each benchmark, its options and its runtime.

## Provenance of every number

<!-- numbers:begin provenance -->
| number | value | source | dataset | teacher | hardware | date | status |
|---|---|---|---|---|---|---|---|
| `cert.violation` | 0.0–8.1% of certificates exceeded α in simulation (δ allows 10%) | `results/b8_certificate_sim.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `curve.dataset` | BANKING77 | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `curve.first_n` | 250 | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `curve.first_pct` | 0% | `results/b6_compare_openai_banking77.json` | banking77 | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `curve.last_n` | 10,000 | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `curve.last_pct` | 60% | `results/b6_compare_openai_banking77.json` | banking77 | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `demo.agreement` | 81.4% | `site-src/demo-data/replay.json + bundle/certificate.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.dis` | 3.3% | `site-src/demo-data/replay.json + bundle/certificate.json` | BANKING77 test | bge-small zero-shot | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.n_cal` | 3,000 | `site-src/demo-data/replay.json + bundle/certificate.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.n_records` | 10,003 | `site-src/demo-data/replay.json + bundle/certificate.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.offload` | 63.3% | `site-src/demo-data/replay.json + bundle/certificate.json` | BANKING77 test | bge-small zero-shot | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.share_cal` | 61.2% | `site-src/demo-data/replay.json + bundle/certificate.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.teacher_acc` | 68% | `site-src/demo-data/replay.json + bundle/certificate.json` | BANKING77 test | bge-small zero-shot | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `demo.threshold` | 0.886 | `site-src/demo-data/replay.json + bundle/certificate.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `e2e.agreement` | 95.9% | `results/b4_e2e_openai_gpt-4.1-mini_77.json` | BANKING77 | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `e2e.llm_ms` | 1397 ms | `results/b4_e2e_openai_gpt-4.1-mini_77.json` |  | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `e2e.n_cal` | 1,499 | `results/b4_e2e_openai_gpt-4.1-mini_77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `e2e.n_log` | 4,999 | `results/b4_e2e_openai_gpt-4.1-mini_77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `e2e.offload` | 51.3% | `results/b4_e2e_openai_gpt-4.1-mini_77.json` | BANKING77 fresh traffic | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `e2e.run` | gpt-4.1-mini (OpenAI) · 77 intents | `results/b4_e2e_openai_gpt-4.1-mini_77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `film.length_s` | 102 | `video/film.html (EVENTS-JSON duration)` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `fit.scipy_s` | 217 s | `results/b2_fit_compare.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `fit.torch_s` | 40 s | `results/b2_fit_compare.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `headline.alpha_pct` | 5% | `default alpha` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.conf_pct` | 90% | `default delta = 0.1` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.dis_range` | 3.0–6.0% | `results/b6_compare_openai_banking77.json` | banking77, clinc150, ag_news, sst2 | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.dis_sentence` | disagrees with your LLM on at most α = 5% of what it serves (90% confidence); realised 3.0–6.0% in our runs | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.llm_calls_left` | 14–39% | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.served_range` | 61–86% | `results/b6_compare_openai_banking77.json` | banking77, clinc150, ag_news | gpt-4.1-mini (OpenAI) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.tasks_n` | 3 | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `headline.teacher` | gpt-4.1-mini (OpenAI) | `results/b6_compare_openai_banking77.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `hero.us` | 2.4 µs | `results/b3_latency.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `hero.us_n` | 2.4 | `results/b3_latency.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `kevteacher.offload` | 70% | `results/b7_systemone.json` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `lat.bundle_mb` | 1.5 MB | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.c_us` | 1.0 µs | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.http_ms` | 0.17 ms | `results/b3_latency.json` |  |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.js_p99_us` | 38 µs | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.js_us` | 10 µs | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.kev_ms` | 271 ms | `results/evidence.json s1.llama_Kev_0_8B` |  | Kev-0.8B (llama.cpp GGUF) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `lat.laya_ms` | 91.6 ms | `results/evidence.json s1.laya` |  | Laya 421M | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `lat.llm_model` | gpt-4.1-mini (OpenAI) | `results/evidence.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `lat.llm_ms` | 1416 ms | `results/b4_e2e_openai_gpt-4.1-mini_8.json` | banking77 | gpt-4.1-mini (OpenAI) | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.proxy_ms` | 0.53 ms | `results/b3_latency.json` |  |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.py_fast_us` | 2.4 µs | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.py_numpy_us` | 44 µs | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.py_us` | 5.2 µs | `results/b3_latency.json` | banking77 |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `lat.runtime_sentence` | A decision costs 1.0 µs in C, 10 µs in JavaScript and 2.4 µs in Python (5.2 µs with per-option probabilities). | `results/b3_latency.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `ledgar.served` | 13% | `legacy/benchmarks/results/phase31_fidelity.json (gemma/ledgar)` | LEDGAR (legal clauses) | EmbeddingGemma-300m | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `legacy.fid.range_dis` | 3.9–5.7% (above 5% on 2 of 7 tasks; max over seeds 6.5%) | `legacy/benchmarks/results/phase31_fidelity.json` | 7 tasks | EmbeddingGemma-300m / Kev-0.8B | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `legacy.fid.range_offload` | 45–71% | `legacy/benchmarks/results/phase31_fidelity.json` | 7 intent/topic tasks | EmbeddingGemma-300m zero-shot (6 tasks), Kev-0.8B (1 task) | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `legacy.fid.tasks_n` | 7 | `legacy/benchmarks/results/phase31_fidelity.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `mean.acc_delta` | +1.0 | `results/b7_systemone.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `mean.cascade_ms` | 115 ms | `results/b7_systemone.json` | BANKING77 | Kev-0.8B | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `mean.llm_ms` | 382 ms | `results/b7_systemone.json` | BANKING77 | Kev-0.8B | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `mean.measured` | measured end to end | `results/b7_systemone.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `mean.teacher` | Kev-0.8B (local, MLX) | `results/b7_systemone.json` |  |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `oos.silent` | 5% | `results/b6_compare_openai_clinc150.json` | clinc150 | gpt-4.1-mini (OpenAI) | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `price.gpt41mini_per_1k` | $0.520 | `results/b6_compare_*.json` |  |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `price.jev_per_1k` | $0.0126 | `https://openrouter.ai/docs/guides/community/jev` |  |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `price.luna_per_1k` | $0.030 | `https://developers.openai.com/api/docs/guides/decisions` |  |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `scale.hardware` | Measured on 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39, 2026-10-08 | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.k150_budget_mb` | 1.00 MB | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.k150_cert_budget` | 94.4% | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.k150_cert_full` | 94.4% | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.k150_full_mb` | 3.6 MB | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.k150_options` | 150 | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.largest` | 500 options with a 4 MB budget (5.0 MB, 57.3% certified) | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.largest_train` | 21 minutes | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `scale.p50_range` | 7.1 and 10.7 µs | `results/b9_scale.json` | synthetic, one consistent teacher |  | 12th Gen Intel(R) Core(TM) i7-1255U (12 threads), Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39 | 2026-10-08 | new |
| `sst2.served` | 0% | `results/b6_compare_openai_sst2.json` | sst2 | gpt-4.1-mini (OpenAI) | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
| `sst2.table_acc` | 79% | `legacy/benchmarks/results/phase29_hybrid.json (table vs gold, SST-2)` | SST-2 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-01 | legacy |
| `sweep.a01.dis` | n/a | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a01.served` | 0% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a02.dis` | 0.4% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a02.served` | 20% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a05.dis` | 3.0% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a05.served` | 61% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a10.dis` | 7.4% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `sweep.a10.served` | 80% | `results/evidence.json cmp.banking77.openai.shad0w` | BANKING77 |  | Apple M5 Pro (15 cores, 24 GB), macOS | 2026-10-08 | new |
| `tests.count` | 149 | `results/b1_tests.txt` |  |  | Apple M5 Pro, 15 cores, 24.0 GB, macOS-26.6.2-arm64-arm-64bit | 2026-10-08 | new |
<!-- numbers:end provenance -->
