---
title: HTTP server
description: Serve a trained table over HTTP to any language, in shad0w's own format or the Decisions API and System One shapes, with metrics and a live dashboard.
---
# HTTP server

`shad0w serve` puts a trained table on an HTTP port, so any language can ask it. It answers from the table alone: there is no LLM behind it, so each answer says whether it is **certified** and you decide what to do with the rest. ("Certified" = tested on answers the table never trained on, and covered by its bound on disagreement with your LLM; see [The guarantee](guarantee.html).)

Want a server that also calls your LLM and logs its answers for training? Use the [proxy](proxy.html) instead.

## TL;DR

```bash
shad0w serve --bundle shad0w/intent/bundle --port 8010
curl -s localhost:8010/v1/decide -d '{"state": "my card was stolen"}'
```

```json
{"answers": {"intent": {"choice": "lost_card", "confidence": 0.993, "certified": true, "flag": null, "radius": 6,
                        "probabilities": {"refund": 0.002, "lost_card": 0.993, "...": 0.005}}},
 "latency_us": 11.2}
```

Serve `choice` when `certified` is true; otherwise ask your LLM.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | the trained table folder |
| `--host` | `127.0.0.1` | where to listen; `0.0.0.0` inside a container, with an access token |
| `--token-env NAME` / `--token-file PATH` | `$SHAD0W_PROXY_TOKEN` | access token every route but `/v1/health` needs: `X-Shad0w-Token: <token>`, `Authorization: Bearer <token>`, or the browser's login prompt for the dashboard |
| `--port` | `8010` | the port |
| `--cost-per-call` | | what one LLM call costs you, for the dashboard's money-saved figure |
| `--llm-latency-ms` | | your LLM's latency, for the dashboard's time-saved figure |

## Routes

| Route | What it does |
|---|---|
| `POST /v1/decide` | shad0w's own format (below) |
| `POST /v1/decisions` | the OpenAI Decisions API shape |
| `POST /v1/systemone` | the System One shape (Jev, Kev, Laya clients) |
| `POST /v1/playground` | what the table thinks of a text, without counting it |
| `GET /` | live dashboard |
| `GET /v1/stats` | JSON: counts, share answered, deferral reasons, latency percentiles, recent decisions |
| `GET /metrics` | Prometheus text format ([Monitoring](observability.html)) |
| `GET /v1/health` | `{"ok": true, "questions": [...]}` |

## `POST /v1/decide`

**Request**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `state` | string (or JSON object) | yes | the text to decide on. An object is serialised to JSON first. |
| `questions` | object | no | answer only these questions; `{"intent": {"criteria": ["refund", "lost_card"]}}` also narrows `intent` to that subset of its options |
| `exposed` | boolean | no | treat the input as adversarial: also withhold certification from answers a few inserted words could flip |

**Response**: one entry per question under `answers`, plus `latency_us` (time inside the server).

| Field | Question type | Meaning |
|---|---|---|
| `choice` | choice | the table's best option |
| `answer` | yes/no | `true` or `false` |
| `confidence` | both | the table's confidence in that answer, 0 to 1 |
| `certified` | both | `true`: serve it. `false`: ask your LLM. |
| `flag` | both | why it is not certified: `low_confidence`, `drift`, `low_radius` (exposed only), `uncalibrated`; `null` when certified. See [flags](python.html#flags). |
| `radius` | both | how many inserted words it takes to flip the answer (higher is more robust; `null` when none can) |
| `probabilities` | choice | every option's probability (only the requested ones when narrowed) |
| `probability` | yes/no | the probability of yes |

The table always gives its best guess; `certified` is what tells you whether to trust it.

## The Decisions API and System One shapes {#decisions}

The same table also answers in the two decision-model formats, so a client written for them works against `shad0w serve` unchanged.

**OpenAI Decisions API**, `POST /v1/decisions`:

```bash
curl -s localhost:8010/v1/decisions -d '{"model": "any", "input": "my card was stolen",
  "questions": [{"type": "choice", "name": "intent", "instructions": "...",
                 "choices": [{"value": "refund"}, {"value": "lost_card"}, {"value": "balance"}]}]}'
```

```json
{"object": "decision", "model": "any",
 "answers": [{"type": "choice", "name": "intent", "choice": "lost_card", "confidence": 0.999,
              "probabilities": [{"value": "refund", "probability": 0.0001}, {"value": "lost_card", "probability": 0.999}, "..."],
              "shad0w": {"source": "table", "certified": true, "flag": null}}],
 "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "shad0w": {"latency_us": 38.0, "source": "table"}}
```

**System One** (Jev, Kev, Laya clients), `POST /v1/systemone`:

```bash
curl -s localhost:8010/v1/systemone -d '{"model": "any", "state": "my card was stolen",
  "questions": {"intent": {"type": "choice", "instructions": "...", "criteria": {"refund": "", "lost_card": "", "balance": ""}}}}'
```

```json
{"model": "any", "answers": {"intent": {"type": "choice", "choice": "lost_card", "confidence": 0.999,
                                       "probabilities": {"refund": 0.0001, "lost_card": 0.999, "balance": 0.0},
                                       "shad0w": {"source": "table", "certified": true, "flag": null}}},
 "latency_ms": 0.04, "shad0w": {"latency_us": 38.0, "source": "table"}}
```

- The question `name` must match a question in the bundle (the decision's name, e.g. `intent`).
- The listed options narrow the decision to that subset.
- Yes/no questions are `predicate` (answered with `probability`) and `noul` (answered with `noul`, the probability of yes, plus `answer`) respectively.
- A question the bundle does not know, and any `score` question, comes back with `shad0w.flag = "unsupported"`. For the version that forwards those to a real model, use the [proxy](proxy.html).
- Streaming is not supported here.

## Playground

```bash
curl -s localhost:8010/v1/playground -d '{"question": "intent", "text": "my card was stolen"}'
```

It returns what the table thinks without recording a decision: answer, confidence, threshold, α, whether it is certified, the reason in words, and the top three options. `question` may be left out when the bundle has one question. The dashboard's Playground card uses it.

## Limits

- HTTP/1.1 keep-alive and one thread per connection; it adds about 0.17 ms per request on localhost.
- Request bodies over 1 MiB get a 413. A malformed request gets a 400 with `{"error": "..."}`; an unknown route a 404.
- Off this machine, set an access token (above); without one it starts with a warning.
