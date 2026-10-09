---
title: Limits
description: Where shad0w does not help, what its guarantee does not cover, and what to do about each case.
---
# Limits

This page lists where shad0w stops helping and what to do in each case. shad0w is careful engineering on well-known ideas: a fastText-style table of word patterns and a Learn-then-Test certificate. Knowing where it stops is what makes the guarantee useful.

In short: it picks one answer from a fixed list, it learns wording rather than meaning, it copies your LLM (mistakes included), and its promise holds only for traffic like the traffic it was tested on.

## It picks; it doesn't write

shad0w answers fixed-choice questions: one answer from a list per question. Chat, summaries, extraction and reasoning stay with your LLM. For several labels on one message, ask one yes/no question per label.

## Wording, not meaning

The table learns from words and character patterns. It is strong on intents, routes, topics and policies that have distinctive wording. It is weak on:

- **sentiment, tone, sarcasm and emotion:** about 79% accuracy on SST-2, even trained on perfect labels;
- **world knowledge and reasoning:** "is this claim true?";
- **long documents where one clause decides:** legal clause types reached only 13% served in our tests.

On these, shad0w certifies little or nothing and sends the traffic to your LLM. That is safe, but it saves you nothing.

## It copies your LLM

The guarantee bounds disagreement with your LLM, not errors against the truth. A certified table in front of a wrong LLM is wrong the same way. For a bound against the truth, certify on about 300 human labels with `shad0w calibrate` ([training](training.html)).

A noisy LLM is also hard to copy. If your LLM gives different answers to near-identical messages, no confidence threshold keeps disagreement under α, and the table serves little.

## It assumes traffic like the calibration traffic

The certificate is computed on a slice of your logged traffic, so it describes traffic with that same mix. A big change in what users send can break it.

The most common way this goes wrong is the share of off-topic messages. In our CLINC150 run, the logged traffic had very few out-of-scope messages and the test traffic had many more. The table was certified on the first mix and served the second, and its realised disagreement went above α. Some messages that belonged in `other` were answered with a real intent. Details: [where shad0w loses](benchmarks.html#where-shad0w-loses).

What protects you:

- **Log real traffic, off-topic part included, at its real rate.** Don't filter the log down to "good" examples before training.
- **Always offer an `other` option**, so your LLM has somewhere to put messages that fit no intent, and the table learns that answer too. If answering a real intent by mistake is costly, add `never_serve=["other"]` so off-topic messages always go to your LLM.
- **Watch the drift guard.** It notices when more messages than at calibration fall below its confidence level, which is what a jump in off-topic traffic looks like. It then flags answers `drift` and defers them until the mix settles or you retrain. Spot checks catch slower shifts. See [the guarantee](guarantee.html#how-it-stays-honest-at-runtime).
- **Retrain when the mix changes**, for example after a launch or a new channel.

## It changes when your LLM changes

The certificate is tied to the LLM that produced the log. After a model or prompt change, rotate the log, let the new LLM answer for a while, and retrain. shad0w cannot detect a prompt change on its own.

## It needs data first

Nothing is served until at least `min_rows` answers (default 1,000) are logged and `train()` has run. Questions with many options need more: with 77 options, our runs needed a few thousand. See [benchmarks](benchmarks.html#it-grows-with-traffic).

## It is not the most accurate engine

If you need the most accurate answer every time, call your best model. A decision model answers every message from its own knowledge; Kev-0.8B scored 80.0% against gold on BANKING77, at 271 ms per decision. shad0w belongs **in front of** that model, answering the certified share in microseconds, not instead of it.

## Size limits

| | |
|---|---|
| Options per question | 2 to 1,024 |
| Patterns per table | up to 4,194,304 (2²²) |
| Table size | about (options + 4) bytes per pattern; cap it with `max_mb` ([scale](scale.html)) |
| Text | any UTF-8; words and character patterns are hashed, so any language works |

## No built-in authentication

`shad0w serve` and `shad0w proxy` have no authentication. They listen on `127.0.0.1` by default. Keep them on localhost or behind your own gateway.
