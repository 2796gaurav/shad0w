---
title: How it compares
description: shad0w next to semantic caches, embedding routers, small fine-tuned models, LLM-only, and decision models (OpenAI Decisions API, Jev, Kev, Laya), measured with the same certifier.
---
# How it compares

This page puts shad0w next to the other ways to answer a fixed-choice question without calling your LLM every time, measured the same way, and says when to pick which.

## How the comparison was run

- **Same data.** BANKING77 (77 intents). gpt-4.1-mini (OpenAI) answered 10,000 messages; that log is the training data for every learned method.
- **Same split and the same certifier.** Every learned method was certified with the same Learn-then-Test procedure at α = 5% and δ = 0.1. So "answered without the LLM" means the same thing on every row: the share it may answer while staying within α of the LLM ([the guarantee](guarantee.html)).
- **Same fresh test set**, never seen in training, over 3 seeds.
- **Decision models are a different kind of thing.** They replace your LLM for every request and return calibrated probabilities, but no bound against anything. They are measured on the same prompts, and they can also be the LLM shad0w learns from.

<!-- numbers:begin compare_table -->
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
<!-- numbers:end compare_table -->

The columns, in plain words:

| Column | Meaning |
|---|---|
| answered without the LLM @ α = 5% | Share of fresh messages the method answered itself. Decision models answer all of them. |
| disagreement with the LLM on those | How often those answers differed from the LLM's. Must stay near or under α for certified methods. |
| accuracy vs gold (cascade) | The method plus the LLM for everything it defers, scored against human labels. This is what your users get. |
| p50 per decision | Median time of the method itself on one message. |
| cost per 1k decisions | LLM spend per 1,000 messages for what still goes to the LLM, or the vendor price for hosted decision models. |
| needs | What you must run in production. |
| guarantee | What bound you get on its answers. |

Local decision-model rows come from `results/b7_systemone.json`; hosted rows use vendor-stated latency and prices. Everything else, with hardware and dates: [benchmarks](benchmarks.html).

## What each row is

- **shad0w table**: a hashed word and character-pattern table with 8-bit weights, trained on your LLM's logged answers and certified. Answers in microseconds on a CPU; no model to host.
- **semantic cache**: embed the message (bge-small) and return the logged answer of the nearest logged message when similarity clears a threshold. We put the same certifier on the similarity, so the comparison is fair. Needs an embedding call per message.
- **embedding kNN router**: like the cache, but a weighted vote over the 10 nearest logged messages.
- **small fine-tuned head**: logistic regression on bge-small embeddings, the cheapest "fine-tune a small model" stand-in. Needs the embedding model at serving time.
- **exact-match cache**: free, but only answers exact repeats.
- **keep calling the LLM**: the reference every other row is measured against.
- **decision models**: models built to answer typed questions directly: the OpenAI Decisions API (`gpt-6-luna`, $0.030 per 1,000 decisions), TypeSafe Jev (hosted), and the open Kev, Laya and Julia-1 models run locally. They answer every message, in milliseconds to a second.

## Which one to pick

| You want | Pick |
|---|---|
| Microseconds, no model to host, a written bound against your LLM | shad0w in front of your LLM |
| The largest share answered, and you already run an embedding model | an embedding router or small head, ideally with the same certifier |
| One engine that answers everything, with good accuracy | a decision model, with shad0w in front of it for the repeat traffic |
| Free answers for exact repeats only | an exact-match cache |

shad0w and decision models combine. Use the decision model as the LLM shad0w learns from: `llm="openai-decisions/gpt-6-luna"` for the hosted API, or `llm="systemone/kev-0.8b"` with `base_url="http://localhost:8080"` for a local llama.cpp server. The table then answers the certified share in microseconds and the decision model handles the rest. See [Decisions API & System One](decisions-api.html).

## Where shad0w loses

- **Share answered:** on BANKING77, the kNN router and the fine-tuned head answered more traffic at the same α. They pay for it with an embedding model on every message.
- **Sentiment and tone** (SST-2): the table reaches 79% accuracy at best, certifies 0% and defers everything. That is the right behaviour; a decision model or your LLM is the right tool.
- **Long legal clauses** (LEDGAR, 100 labels): 13% served in an earlier run.
- **Accuracy:** the table can only be as accurate as the LLM it copies. A decision model answers *every* message from its own knowledge; shad0w answers only the certified share and hands the rest back.
- **Cold start:** nothing is served until enough answers are logged and `train()` has run (about 1,000 for simple questions, a few thousand for 77 options).

The full list, including the CLINC150 traffic-shift case: [benchmarks](benchmarks.html#where-shad0w-loses).
