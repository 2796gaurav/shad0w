---
title: Training & retraining
description: How shad0w learns from your LLM's logged answers, when to train, what a training run does, and how to keep the table fresh.
---
# Training & retraining

<p class="lead">shad0w trains its table from the answers your LLM already gives. You don't label anything. This page covers what is logged, when to train, what a training run does, and how to keep the table current as traffic changes.</p>

**TL;DR**

```bash
shad0w status                                   # rows logged vs needed, certified share, next step
shad0w train --dir shad0w --question intent     # train + certify; a running app picks it up
```

```python
intent.train()                                  # the same from Python: train, certify, hot-swap
```

Words used on this page. **The table**: the small model shad0w trains from your LLM's answers. **Certified**: tested on logged answers it never trained on, and allowed to serve only where it differs from your LLM on at most **α** of them (default 5%, with 90% confidence). **Defer**: send to your LLM. **Spot check**: a decision also re-asked to your LLM to measure live agreement.

## The loop

<div class="viz" data-viz="lifecycle" data-step="train">Log: every answer your LLM gives is appended to the log. Train: once enough answers are logged, a table is fitted on most of them. Certify: the table is tested on answers it never saw, and a confidence cutoff is chosen so it disagrees with your LLM on at most α of what it serves. Serve: the table answers what it is certified on, in microseconds; the rest go to your LLM and are logged. Spot-check: a small random share is re-asked to your LLM, giving a live agreement figure and a fair sample for the next certificate.</div>

1. **Log.** Every LLM answer is appended to `shad0w/<question>/log.jsonl`.
2. **Train.** Once `min_rows` answers are logged (default 1,000), a table is fitted on them.
3. **Certify.** The table is tested on answers held back from training. The certifier picks the confidence cutoff that keeps disagreement within α.
4. **Serve.** The table answers what it is certified on; everything else goes to your LLM and keeps feeding the log.
5. **Spot-check.** A small random share (`audit_rate`, default 1%) is also sent to your LLM, to measure live agreement and to certify the next table fairly.

## The log is the training data

Every time your LLM answers, shad0w appends one line:

```json
{"text": "my card was stolen", "intent": "lost_card", "source": "teacher", "ts": 1791370000.1}
```

`source` is `teacher` (your LLM answered), `audit` (a spot check) or `import` (brought in with `warm_start` or `shad0w import`). Answers that match none of your options are never logged, and neither are the table's own answers, so the table never trains on itself.

### Start from answers you already have

```bash
shad0w import --from openai-chat --file export.jsonl --question intent      # chat completions request/response pairs
shad0w import --from openai-decisions --file export.jsonl --question intent # OpenAI Decisions API pairs
```

The file holds one `{"request": ..., "response": ...}` per line; OpenAI Batch output works as is. From Python, `intent.warm_start("old_answers.jsonl")` reads `.jsonl`, `.json` or `.csv` with a `text` column and one column named after the question. Both check every label against your options and report what they skipped.

Any JSON Lines file with a `text` field and one field per question can also be trained on directly with `shad0w train --log`:

```json
{"text": "app crashes on login", "team": "mobile", "urgent": true}
```

### Log the traffic you really get

The certificate holds for traffic that looks like the log, so the log should be your real traffic at its real proportions:

- keep off-topic and odd messages in it, at the rate they really arrive;
- give every question an `other` option, so those messages have an answer to learn;
- don't hand-pick or deduplicate the log into a cleaner set than production.

If off-topic messages are rare in the log but common in production, the table is certified on the wrong mix and can disagree with your LLM more than α. The drift guard flags this kind of shift. Worked example: [CLINC150 in the benchmarks](benchmarks.html#where-shad0w-loses).

## When to train

```bash
shad0w status
```

```text
intent    1,500/1,000 rows  no table yet  -> run: shad0w train --dir shad0w --question intent
```

`shad0w stats --log shad0w/intent/log.jsonl` shows the answers per option:

```text
shad0w/intent/log.jsonl: 1,500 answers logged (teacher 1,484, audit 16)

intent: 1,500 answers, 4 different
  balance                              391  ########################################
  lost_card                            382  #######################################
  refund                               374  ######################################
  other                                353  ####################################
  -> ready to train
```

- **About 1,000 answers** gets you started; 100 is the hard minimum.
- **More is better.** In our BANKING77 run the share answered by the table went from 0% at 250 answers to 60% at 10,000.
- **Rare options** need examples too. shad0w warns about options with fewer than 20 answers; the table rarely answers them, which is safe: those messages go to your LLM.

## Train

```python
intent.train()                          # train + certify + hot-swap; returns the certificate (+ "accepted")
intent.train(alpha=0.02, min_rows=500)  # a stricter bound; try with fewer rows
intent.train(gate=True)                 # keep the current table unless the new one is at least as useful
```

```bash
shad0w train --dir shad0w --question intent                                # the decision() / proxy folder layout
shad0w train --log shad0w/intent/log.jsonl --out shad0w/intent/bundle      # any log; questions and options are detected
shad0w train --log log.jsonl --out bundle --schema schema.json --alpha 0.02 --max-mb 1
```

```text
intent: 1,500 usable answers, 4 options
trained in 1.6s -> shad0w/intent/bundle
  intent: the table will answer 100.0% of traffic like this, disagreeing with your model on at most 5% of those (with 90% confidence; calibration: held-out-split)
```

(That share is from a small synthetic example. On real traffic, expect less; see [Benchmarks](benchmarks.html).)

Training needs `pip install "shad0wllm[compile]"` (scipy and scikit-learn). A run of 1,000–10,000 answers takes seconds to a minute on a laptop CPU. With torch installed (`pip install "shad0wllm[torch]"`) large logs train faster; without it, scipy solves the same problem more slowly. Every `shad0w train` flag is on the [Command line](cli.html#train) page.

A training run:

1. picks a **calibration set**: the spot-check rows when there are at least 100 of them (see below), otherwise a random `cal_fraction` of the answers (default 30%, at most `max_cal` = 3,000);
2. **fits** the table on the rest;
3. **certifies** it on the calibration set and picks the confidence cutoff;
4. writes the bundle: `manifest.json`, one `.s0` table per question, `certificate.json` and `schema.json`.

The certificate records which calibration set it used: `"calibration": "uniform-audit"` or `"held-out-split"`. Read it with `shad0w report --bundle shad0w/intent/bundle`.

## Why retraining certifies on spot checks

Before the first table exists, the log holds every call. Afterwards it holds mostly the calls the table **deferred**: the hard ones. A certificate computed on those would describe a harder mix than your real traffic.

Spot checks fix that. At rate `audit_rate`, an answer the table served is re-asked to your LLM in the background, and an answer your LLM gave anyway is marked as a spot check. The rows marked `"source": "audit"` are a fair, uniform sample of live traffic, and training certifies on them once there are at least 100.

With fewer, training falls back to a random held-out split of the whole log, and the bound then speaks for a harder mix than live traffic. During the first weeks of serving, set `audit_rate = 0.05` to collect the sample faster, or stay in `mode = "shadow"`, where the table answers nobody and the log stays representative ([Rollout](rollout.html)).

## Keep it fresh

Traffic changes: new products, new phrasings, new kinds of users. Three ways to stay current:

- **Retrain on a schedule.** The log keeps growing with every deferred call and every spot check. Run `shad0w train --dir shad0w --question intent` weekly. A running proxy picks up the new table within a second; a `decision()` in your app picks it up when you call `intent.reload()` (or `intent.train()` from inside the app, which swaps it in directly).
- **Retrain automatically.** `auto_train = 2000` (or `decision(..., auto_train=2000)`, or `shad0w proxy --auto-train 2000`) retrains in the background every 2,000 new logged answers. It is **off by default**. With `retrain = "gated"` (default), the new table replaces the current one only if it certifies at least 80% of the current certified share; otherwise the old one stays and a warning is logged. `shad0w train --gate` does the same from the command line and exits with code 3 when it keeps the old table. The gate never blocks a run after your options changed: the old table answers a different question.
- **Re-certify only.** `shad0w certify --bundle shad0w/intent/bundle --data fresh.jsonl` keeps the table and refreshes its cutoff on fresh answers it never trained on.

Signs it is time:

- the dashboard's live bound creeps toward α;
- `flag="drift"` shows up and stays;
- the share answered by the table drops;
- `shad0w status` says `serving; N new answers since training`.

## Choosing α

α is the most disagreement with your LLM you accept among the table's answers; a smaller α answers less but agrees more. On BANKING77 with a small open teacher model (68% accurate), the table answered 63.3% of test traffic at α = 5% with 3.3% realised disagreement (the [playground](../demo/) table). [Tuning](tuning.html#alpha) shows how the share changes with α and how to try values on your own log.

## Human labels instead of an LLM

Have labelled data? Compile from it directly, and certify against the truth instead of against your LLM:

```bash
shad0w compile   --schema schema.json --data labels.jsonl --out bundle/
shad0w calibrate --bundle bundle/ --data real_labels.jsonl          # ~300 held-out labels per question
```

`schema.json` names the options: `{"intent": {"type": "choice", "criteria": {"refund": "wants money back", "lost_card": null}}}`. See [Running without an LLM](without-llm.html).
