---
title: Tuning
description: What each knob does in plain words (α, δ, min_rows, audit_rate, min_confidence, force_threshold, canary, never_serve), why the defaults tune themselves, and recipes for common goals.
---
# Tuning

<p class="lead">shad0w has one number that sets safety (α) and a few knobs that decide how much it answers and how carefully it rolls out. This page explains each in plain words, why you usually don't need to touch them, and what to change for common goals.</p>

**TL;DR.** Keep the defaults. Pick α once for how costly a wrong answer is, roll out with `mode` and `canary`, and keep spot checks on. Everything else adjusts itself.

```toml
# shad0w.toml: the only lines most teams ever write
alpha = 0.05          # at most 1 in 20 table answers may differ from your LLM
audit_rate = 0.01     # 1 in 100 decisions is double-checked against your LLM
mode = "serve"        # "shadow" while you watch, then "serve"
```

Words used on this page. **The table**: the small model shad0w trains from your LLM's logged answers. **Certified**: tested on logged answers it never trained on, and allowed to serve only where it is expected to differ from your LLM on at most α of them. **Defer**: send this one to your LLM. **Spot check**: a decision also re-asked to your LLM, to measure live agreement. **Drift**: live traffic no longer looks like the traffic the table was tested on.

## Why you don't need to fine-tune

Most systems that put a cheap model in front of an expensive one ask you to pick a confidence cutoff and keep re-picking it. shad0w does that part for you, and checks its own work:

| What would normally be tuned by hand | What shad0w does instead |
|---|---|
| The confidence cutoff above which the cheap model answers | **The certifier picks it.** At every training run it tests the new table on held-out answers and chooses the lowest cutoff that keeps disagreement within α (at 90% confidence). You set α, not the cutoff. |
| Deciding whether a retrained model is better | **Gated retraining.** With `retrain = "gated"` (default), a new table replaces the current one only if it certifies at least 80% of the current one's share. A bad run never lands. |
| Remembering to retrain | **`auto_train`** retrains in the background every N new answers. It is **off by default** (`0`): turn it on when you want it, e.g. `auto_train = 2000`. |
| Checking that it is still right in production | **Spot checks** (`audit_rate`, default 1%) re-ask your LLM on a random sample and compute a live upper bound on disagreement, shown on the dashboard. |
| Noticing that traffic changed | **The drift guard** watches how often the table is unsure on recent traffic. When that rises, it sends answers to your LLM until traffic looks familiar again or you retrain. |

So the knobs below are for **policy** (how strict, how fast to roll out, which labels are off-limits), not for keeping the table accurate.

## The knobs, in plain words

### α: how much disagreement you accept {#alpha}

`alpha` (default `0.05`) is the most the table may disagree with your LLM, counted over the answers it serves. `0.05` means at most 1 in 20.

A smaller α makes the certifier pick a higher confidence cutoff, so the table answers fewer messages but agrees more often. A larger α does the opposite. Click through the values below: the figure shows the real certifier output on the demo table.

<div class="viz" data-viz="alpha">On the demo table (BANKING77, 77 intents), a stricter α means the table answers a smaller share of traffic: lower α moves the confidence cutoff up, and only the answers above it are served. Realised disagreement stays near or below α.</div>

On a 77-intent task with gpt-4.1-mini (OpenAI) as the teacher, the table answered 0% of traffic at α = 1%, 20% at 2%, 61% at 5% and 80% at 10%. Realised disagreement was 0.4%, 3.0% and 7.4% for the last three.

How to pick:

- **5%** suits routing and triage, where a wrong answer costs a retry or a hand-off.
- **1–2%** where a wrong answer is expensive (refunds, compliance).
- **10%** only when a wrong answer costs almost nothing.

Try several on your own log before deciding. Training into a scratch folder leaves the live table alone:

```bash
shad0w train --log shad0w/intent/log.jsonl --out /tmp/a02 --alpha 0.02
shad0w train --log shad0w/intent/log.jsonl --out /tmp/a05 --alpha 0.05
```

Each prints the share the table would answer at that α.

**There is one α per question, not per label.** The certificate controls a single number: disagreement over everything the table serves. For a label that must be stricter, use `never_serve`, or give that question its own section with a lower `alpha` ([Configuration](configuration.html#where-a-setting-can-come-from)).

### δ: how sure the certificate is

`delta` (default `0.1`) is the chance that the certificate itself is wrong because the held-out answers happened to be unusually easy. `0.1` means 90% confidence. You rarely change it: `0.05` (95%) makes the cutoff a little stricter and the share answered a little smaller.

### min_rows: when training may start

`min_rows` (default `1000`) is how many logged LLM answers training waits for. Below it, `train()` refuses. 100 is the hard minimum, for experiments. More answers almost always certify a larger share: in our BANKING77 run the share went from 0% at 250 answers to 60% at 10,000.

### audit_rate: spot checks

`audit_rate` (default `0.01`) is the share of all decisions that are also checked against your LLM. When the table answered, your LLM is asked in the background, so the user never waits. When your LLM answered anyway, the row is simply marked as a spot check. These rows are a fair sample of live traffic. They feed the live safety bound on the dashboard, and once there are 100 of them, training certifies on them ([why](training.html#why-retraining-certifies-on-spot-checks)).

Raise it to `0.05` for the first weeks of serving. Keep it above zero afterwards: without spot checks you would learn about a drop in agreement only from your users.

### min_confidence and force_threshold: moving the cutoff by hand

Both override the certified cutoff, in opposite directions, and they are deliberately not symmetric:

| | `min_confidence` | `force_threshold` |
|---|---|---|
| Direction | Stricter: the table needs at least this confidence **and** the certificate | Looser: the table serves above this confidence **even where the certificate says no** |
| Guarantee | Still holds (the table serves a subset of what was certified) | Voided for those answers: they carry `certified=False` and flag `manual_threshold`; a warning is logged at start-up |
| Flag on deferred answers | `min_confidence` | — |
| Use it for | A safety margin, e.g. `0.97` | Experiments only |

### canary, mode and never_serve: rollout

- **`mode`**: `serve` (default) answers from the table; `shadow` lets the table decide but always returns your LLM's answer and counts how often they would have differed; `off` skips the table entirely.
- **`canary`** (default `1.0`): the share of certified answers the table may actually serve. `0.1` serves one in ten; the rest go to your LLM, flagged `canary`, and keep feeding the log.
- **`never_serve`** (default `[]`): labels that always go to your LLM, however sure the table is.

The [Rollout guide](rollout.html) walks through `shadow` → `canary` → `serve`.

### exposed: untrusted input

`exposed = true` also defers answers that a few character edits could flip (flag `low_radius`). Use it where users may try to game the decision, such as moderation or fraud screens.

## Recipes

### "It answers too little"

The share the table answers depends on how **consistent** your LLM is and how **much** traffic it has seen. In order of impact:

1. **Log more traffic.** The certified share grows with the log (see `min_rows` above).
2. **Offer an `other` option** and keep off-topic traffic in the log at its real rate. Without it, off-topic messages get forced into real options, and the table becomes less sure everywhere.
3. **Make your LLM consistent.** Temperature 0, a fixed prompt, and option descriptions that don't overlap. An LLM that answers the same message differently on two calls caps what any table can certify.
4. **Merge options your LLM confuses.** If `card_lost` and `card_stolen` lead to the same action, merge them with `rename` ([Changing your options](options.html)).
5. **Raise α**, if your product tolerates it. Measure the gain first with `shad0w train --out /tmp/... --alpha 0.1`.

### "It must be stricter"

| Need | Setting |
|---|---|
| Fewer disagreements overall | lower `alpha` (`0.01`–`0.02`) |
| One label must always see your LLM | `never_serve = ["fraud", "self_harm"]` |
| A margin above the certified cutoff | `min_confidence = 0.97` |
| Untrusted input | `exposed = true` |
| A stricter bound for one question only | a `[questions.<name>]` section with its own `alpha` |
| Try it without risk first | `mode = "shadow"`, then `canary = 0.1` |

### "The table is too big"

Size grows with vocabulary × options ([Scale](scale.html)). Cap it with `max_mb`:

```python
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"],
                         llm=lambda text: "other", max_mb=1)     # or SHAD0W_MAX_MB=1, or shad0w train --max-mb 1
```

Training keeps the most informative features and refits on them; the certificate is computed on the capped table. On 150 options, `max_mb=1` gave a 1.00 MB table that certified 94.4% of calibration traffic, the same as the uncapped 3.6 MB table (94.4%). Very small budgets on many options eventually cost coverage: check the printed share.

### "My LLM changed"

The certificate is against the LLM that produced the log. A new model or a new prompt is a new teacher:

1. Point `folder=` at a new place so the new LLM's answers go to a fresh log. Keep the old table serving, or set `mode = "shadow"` to see how often the new LLM disagrees with it.
2. Once the new log has `min_rows` answers, train: the new table is certified against the new LLM.

### "Drift fires on bursty traffic"

A long run of one hard message type (an outage brings a flood of "card not working") can fill the drift window with unsure answers even if the overall mix is unchanged. Raise `drift_window` to `2000` so one burst is a smaller part of it, or raise `drift_margin` a little. See [Troubleshooting](troubleshooting.html#drift).

## Should I fine-tune a small model instead?

They solve different problems and combine well.

| | Fine-tune a small model | shad0w |
|---|---|---|
| What you get | a smaller model that imitates your LLM | a table that answers what it can show it gets right, and defers the rest |
| Cost to build | labelled data or a distillation run, GPU hours | your LLM's logged answers, seconds to minutes on a CPU |
| Per decision | milliseconds, on a GPU or a fast CPU | microseconds, in your process, a browser or a worker |
| When it is wrong | silently | measured: the certificate bounds disagreement, and unsure inputs go to your LLM |
| Best at | tasks that need real language understanding | high-volume, recognisable decisions |

A common set-up uses both: the fine-tuned model is the teacher, and shad0w sits in front of it.

## Knobs you rarely need

`cal_fraction` and `max_cal` (how much of the log is held back to certify), `drift_window` and `drift_margin` (how quickly the drift guard reacts), `retrain` and `on_new_option`. Their defaults suit almost everyone; [Configuration](configuration.html#every-setting) lists them all.
