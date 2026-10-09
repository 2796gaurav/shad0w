---
title: Monitoring
description: The live dashboard, Prometheus metric names, logs, hooks, traces and OpenTelemetry spans, and what to alert on.
---
# Monitoring

<p class="lead">At any moment you should be able to see how much the table answers, how fast, why it sent calls to your LLM, and whether it still agrees with your LLM. This page shows where each of those lives: the dashboard, Prometheus, logs, traces and OpenTelemetry.</p>

**TL;DR**

```bash
shad0w proxy --upstream https://api.openai.com/v1 --cost-per-call 0.0006   # or: shad0w serve --bundle B
# dashboard  http://localhost:8010/          Prometheus  http://localhost:8010/metrics
# JSON       http://localhost:8010/v1/stats  terminal    shad0w watch
```

```python
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=lambda text: "other")
intent("my card was stolen")
print(intent.stats()["offload"])        # share answered by the table, in this process
```

Words used on this page. **The table**: the small model shad0w trains from your LLM's answers. **Certified**: allowed to serve, because it was tested to differ from your LLM on at most α of its answers. **Defer**: send to your LLM. **Spot check**: a decision also re-asked to your LLM (`audit_rate`, default 1%). **Drift**: live traffic no longer looks like the traffic the table was tested on.

## The dashboard

`shad0w proxy` and `shad0w serve` both serve a live page at `/`, refreshed every second. It shows:

- the **share answered by the table**, with a sparkline;
- **LLM calls saved**, **time saved** (calls saved × your LLM's measured mean latency, or `--llm-latency-ms` until one is measured) and **money saved** (with `--cost-per-call`);
- per question: **table vs LLM latency** (p50), **why the table deferred** (counts per flag, see below), and the **live safety bound**;
- the latest decisions with source, confidence and latency (`shad0w proxy --no-text` hides the message text);
- a **Playground**: pick a question, type a message, and see the table's answer, its confidence against the certified cutoff, the reason and the top options. On `shad0w proxy` started with `--model`, it can also ask your LLM and show both answers side by side. Nothing typed there is logged unless you ask. The same is `POST /v1/playground` ([HTTP](http.html)).

**The live safety bound** is a 90% upper bound on how often the table disagrees with your LLM, computed from spot checks on the answers it served. The dashboard shows "no spot checks yet", then "live bound … ≤ α …" while it holds, or "⚠ live bound … > α …: re-train" when it does not.

In a terminal, `shad0w watch` shows the same counters (`--url` for another host, `--every` for the refresh in seconds). `shad0w stats --url http://localhost:8010` prints one JSON snapshot.

## Why a call went to your LLM: flags

Every decision has a `flag`. `None` means the table answered under the certificate. Otherwise:

| Flag | Meaning |
|---|---|
| `no_bundle` | no trained table yet |
| `low_confidence` | the table was not sure enough to stay inside the certified bound |
| `drift` | recent traffic looks unlike the calibration data; clears by itself or after retraining |
| `low_radius` | `exposed` mode: a few character edits could flip the answer |
| `uncalibrated` | the table has no certificate |
| `min_confidence` | below your `min_confidence` floor |
| `never_serve` | the label is on `never_serve` |
| `canary` | held back by the `canary` share |
| `shadow` | `mode = "shadow"`: the table's answer was recorded, your LLM's returned |
| `off` | `mode = "off"` |
| `options_changed` | your options include ones the table never learned |
| `option_removed` | the table picked an option you removed |
| `manual_threshold` | served under `force_threshold`: **not** covered by the certificate |

`decision.why` gives the same reason as a sentence. `shad0w.explain("drift")` turns any flag into words.

## Prometheus

`GET /metrics` on `shad0w proxy` or `shad0w serve`, in Prometheus text format:

| Metric | Type | Labels | What it counts |
|---|---|---|---|
| `shad0w_decisions_total` | counter | `question`, `source` | decisions by source: `table`, `teacher` (your LLM answered), `deferred` (`shad0w serve` only: the table was not sure and there is no LLM) |
| `shad0w_flags_total` | counter | `question`, `flag` | why the table did not answer (flags above) |
| `shad0w_offload_ratio` | gauge | `question` | share of decisions answered by the table |
| `shad0w_audits_total` | counter | `question` | table answers spot-checked against your LLM |
| `shad0w_audit_disagreements_total` | counter | `question` | spot checks where your LLM disagreed |
| `shad0w_audit_disagreement_upper` | gauge | `question` | the live safety bound (90% upper bound); absent until the first spot check |
| `shad0w_alpha` | gauge | `question` | α of the loaded table |
| `shad0w_llm_calls_saved_total` | counter | — | LLM calls answered by the table instead |
| `shad0w_time_saved_seconds_total` | counter | — | calls saved × mean LLM latency |
| `shad0w_cost_saved_total` | counter | — | calls saved × `cost_per_call` (only when it is set) |
| `shad0w_passthrough_total` | counter | — | proxy requests forwarded untouched (not decisions) |
| `shad0w_errors_total` | counter | — | requests that failed |
| `shad0w_latency_seconds` | histogram | `question`, `source` | end-to-end decision latency (LLM time for `teacher`) |

```text
shad0w_decisions_total{question="intent",source="table"} 8123
shad0w_decisions_total{question="intent",source="teacher"} 3877
shad0w_flags_total{question="intent",flag="low_confidence"} 3790
shad0w_offload_ratio{question="intent"} 0.677
shad0w_audit_disagreement_upper{question="intent"} 0.041
shad0w_alpha{question="intent"} 0.05
shad0w_latency_seconds_bucket{question="intent",source="table",le="2.5e-05"} 8090
```

Alerts worth having:

```text
shad0w_audit_disagreement_upper > on(question) shad0w_alpha     # the live bound broke α: retrain or lower canary
delta(shad0w_offload_ratio[1d]) < -0.1                           # the table answers much less: traffic changed
increase(shad0w_flags_total{flag="drift"}[1h]) > 0               # drift: retrain on recent traffic
```

### In your own app

Share one `Metrics` object between decisions and expose it on your own `/metrics` route:

```python
import shad0w

metrics = shad0w.Metrics(cost_per_call=0.0006)
intent  = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=lambda t: "other", metrics=metrics)
urgent  = shad0w.decision("urgent", options=bool, llm=lambda t: False, metrics=metrics)
intent("my card was stolen"); urgent("my card was stolen")
print(metrics.prometheus())     # text for your /metrics route
print(metrics.snapshot()["decisions"])    # the same data as a dict
```

`Metrics` also takes `keep_text=False` (keep message text out of `recent`) and `llm_latency_ms=` (assumed LLM latency until one is measured).

## Logs

shad0w logs to the standard `shad0w` logger and prints nothing unless you ask:

```bash
SHAD0W_LOG=info  python app.py     # training, reloads, option changes, spot-check disagreements
SHAD0W_LOG=debug python app.py     # plus one line per decision
```

```text
14:02:11 shad0w DEBUG intent -> 'lost_card' via table conf=0.993 flag=None 9.4us | my card was stolen
14:02:11 shad0w DEBUG intent -> 'other' via teacher conf=0.41 flag=low_confidence 612034.2us | explain interest
14:05:40 shad0w INFO trained 'intent' on 1204 answers in 3.9s: certified share 68.2% at alpha=0.05 (held-out-split)
```

Or attach your own handler: `logging.getLogger("shad0w")`.

## Hooks and traces

```python
import shad0w

def send(decision, text):             # Langfuse, Datadog, Sentry, your warehouse...
    print({"answer": decision.answer, "source": decision.source,
           "us": decision.latency_us, "flag": decision.flag})

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=lambda t: "other",
                         on_decision=send, trace="decisions.jsonl")
intent("my card was stolen")
```

`on_decision` runs after every decision; an exception in it is logged and never breaks the decision. `trace=` (or the `trace` setting) writes **every** decision, table and LLM, as one JSON line:

```json
{"ts": 1791525795.7, "question": "intent", "text": "my card was stolen", "answer": "other", "source": "teacher", "confidence": null, "flag": "no_bundle", "latency_us": 41.2}
```

The trace is kept apart from the training log on purpose: table answers never become training data.

## OpenTelemetry

```bash
pip install opentelemetry-api opentelemetry-sdk
SHAD0W_OTEL=1 python app.py
```

Each decision becomes one span named `shad0w.decide`, with attributes `shad0w.question`, `shad0w.source`, `shad0w.answer`, `shad0w.certified`, `shad0w.confidence` (when the table was consulted), `shad0w.flag` (when set) and `gen_ai.request.model` (your LLM's name, when known). Configure the exporter as usual for your OpenTelemetry SDK; if the API is not installed, nothing happens.

## Explain one decision

```bash
shad0w try --bundle shad0w/intent/bundle "the ATM ate my card"
```

```text
the ATM ate my card
  ✓ table  card_swallowed  (confidence 0.962, needs 0.911)  10.2 µs
     certified: answered by the table
     top: card_swallowed 0.96, lost_card 0.03, other 0.01
```

Without a message, `shad0w try` reads messages interactively. From Python, `intent.explain(text)` returns the same as a dict (`answer`, `confidence`, `threshold`, `flag`, `why`, `top`, `would_serve`) without calling your LLM.
