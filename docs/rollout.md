---
title: "Rollout: shadow → canary → serve"
description: Put shad0w into production in steps (log only, shadow, canary, serve), what to watch at each one, and how to go back.
---
# Rollout: shadow → canary → serve

<p class="lead">The certificate is a promise made on past traffic. A careful rollout checks it on live traffic before the table answers anyone. Each step is one setting, so moving between steps needs no code change: only a config change and a restart.</p>

**TL;DR**

```toml
# shad0w.toml, step by step
mode = "shadow"        # 1. the table decides, your LLM still answers every user
# mode = "serve"       # 2. then: the table answers ...
# canary = 0.1         #    ... 10% of what it is certified on
# canary = 1.0         # 3. then: everything it is certified on
audit_rate = 0.05      # spot checks: keep them on at every step
```

Words used on this page. **The table**: the small model shad0w trains from your LLM's answers. **Certified**: tested on answers it never trained on and allowed to serve only where it differs from your LLM on at most α (default 5%). **Defer**: send to your LLM. **Spot check**: a decision also re-asked to your LLM to measure live agreement.

<div class="viz" data-viz="rollout">off: the table is not consulted; your LLM answers everything. shadow: the table decides every call, but users get your LLM's answer; shad0w counts would_serve and shadow_disagreements. canary: the table answers a share (for example 10%) of what it is certified on; the rest go to your LLM, flagged canary. serve: the table answers everything it is certified on. Every LLM answer is logged at every step.</div>

| Step | Setting | What users get | What is logged and counted |
|---|---|---|---|
| 0. Log only | no table yet | your LLM's answer | every LLM answer |
| 1. Shadow | `mode = "shadow"` | your LLM's answer | every LLM answer; `would_serve`, `shadow_disagreements` |
| 2. Canary | `mode = "serve"`, `canary = 0.1` | the table's answer for 10% of certified calls, your LLM's for the rest | LLM answers (flag `canary`); spot checks |
| 3. Serve | `mode = "serve"`, `canary = 1.0` | the table's answer whenever it is certified | LLM answers for deferred calls; spot checks |
| Off | `mode = "off"` | your LLM's answer | every LLM answer (flag `off`) |

The same settings work as keywords (`decision(..., mode="shadow")`), environment variables (`SHAD0W_MODE=shadow`, `SHAD0W_CANARY=0.1`) and proxy flags (`shad0w proxy --mode shadow --canary 0.1`). See [Configuration](configuration.html#mode).

## 0. Log only

Before a table exists, every call goes to your LLM and the answer is written to the log. Nothing changes for your users.

```bash
shad0w status                                 # rows logged vs min_rows, and the next step
shad0w train --dir shad0w --question intent    # once ready: build and certify the table
```

Read the certified share that `train` prints. If it is close to 0%, the table is not sure enough to help on this question yet: log more answers, or see [Troubleshooting](troubleshooting.html#the-table-answers-nothing-or-very-little).

## 1. Shadow

```toml
mode = "shadow"
audit_rate = 0.05      # a larger fair sample early on; it also keeps the next certificate honest
```

The table now decides every call, but your LLM's answer is the one returned. shad0w counts two things:

- **`would_serve`**: calls the table would have answered;
- **`shadow_disagreements`**: of those, how many differed from your LLM.

```python
import shad0w

intent = shad0w.decision("intent", mode="shadow", llm=lambda text: "other")   # your real llm= here
# ... after some traffic:
s = intent.stats()
print(s["would_serve"], s["shadow_disagreements"])
```

On `shad0w proxy` and `shad0w serve`, the dashboard and `/v1/stats` show the same. Deferred calls carry the flag `shadow`.

**Move on when** `shadow_disagreements / would_serve` sits at or below α over a few thousand calls, and `would_serve` is a share worth having.

## 2. Canary

```toml
mode = "serve"
canary = 0.1           # the table answers 10% of what it is certified on
```

Real users now get table answers, but only for a tenth of the eligible calls. The rest go to your LLM with the flag `canary` and keep feeding the log.

Watch the **live bound** on the dashboard: a 90% upper bound on disagreement among the table's answers, computed from spot checks. It reads "live bound … ≤ α …" while it holds, and "⚠ live bound … > α …: re-train" when it does not.

**Move on when** the live bound stays at or below α as the number of spot checks grows.

## 3. Serve

```toml
mode = "serve"
canary = 1.0
audit_rate = 0.01      # keep spot checks on: they are your early warning
auto_train = 2000      # optional: retrain in the background every 2,000 new answers
retrain = "gated"      # keep the old table unless the new one is at least as useful
```

The table answers everything it is certified on. Keep `audit_rate` above zero. Without spot checks you would learn about a drop in agreement only from your users.

## Going back

Every step can be undone with a setting and a restart of the process that reads it, with no code change:

- `SHAD0W_MODE=off` in the environment: the table is skipped everywhere in that process. Environment variables beat `shad0w.toml`, so this is the fastest kill switch.
- `mode = "shadow"` or `canary = 0` in `shad0w.toml`: back to watching.
- `never_serve = ["fraud"]`: labels that must never be answered by the table. It works at every step.

## What to watch

| Signal | Where | Act when |
|---|---|---|
| Share answered by the table | dashboard, `stats()["offload"]`, `/v1/stats`, `shad0w_offload_ratio` | It falls: traffic changed. Retrain. |
| Live bound vs α | dashboard, `stats()["audit_disagreement_upper"]`, `shad0w_audit_disagreement_upper` vs `shad0w_alpha` | It stays above α. Lower `canary` or go back to `shadow`, then retrain. |
| `drift` flags | dashboard "why deferred", `stats()["flags"]`, `shad0w_flags_total{flag="drift"}` | They stay: inputs look unlike the calibration data. Retrain on the new traffic. |
| New answers since the last training | `shad0w status` | It suggests `shad0w train ...`: a retrain will usually certify more. |

More on each signal in [Monitoring](observability.html).
