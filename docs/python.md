---
title: Python
description: The Python API - decision(), Decision, Shadow, configure, warm_start, batches, async, decorators and the low-level model.
---
# Python

The complete Python API: `shad0w.decision()` and the `Decision` it returns first, then the `Shadow` class underneath it, batches, async, decorators and the low-level model.

## TL;DR

```python
import os, shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "balance", "other"],
                         llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])

d = intent("my card was stolen")      # the table answers if it is certified, otherwise your LLM (logged)
print(d.answer, d.source, d.why)
intent.train()                        # once about 1,000 answers are logged
```

## Install

| Install | Adds | Needed for |
|---|---|---|
| `pip install shad0wllm` | numpy and the bundled C core | serving a trained table |
| `pip install "shad0wllm[compile]"` | scipy, scikit-learn | `train()` |
| `pip install "shad0wllm[torch]"` | torch | faster training on large logs (optional) |

Python 3.10 or newer. Reading `shad0w.toml` on 3.10 needs `pip install tomli`.

## `shad0w.decision(...)` {#decision-fn}

```python
shad0w.decision(name="decision", options=None, llm=None, *, api_key=None, api_key_env=None, base_url=None,
                bundle=None, log=None, folder=None, **kw) -> Shadow
```

One call from nothing to a working decision. It returns a `Shadow`; call it like a function.

| Argument | Type | Default | What it does |
|---|---|---|---|
| `name` | `str` | `"decision"` | what the decision is called. It names the folder, the field in the log, the question inside the table, the dashboard label and the Decisions-API question, so several decisions can run side by side (`"intent"`, `"urgent"`). |
| `options` | see [below](#options) | `None` | the possible answers. Optional once a trained table exists (its options are used). |
| `llm` | `str` or `fn(text) -> answer` | `configure(llm=...)`, then `SHAD0W_LLM`, then `llm` in `shad0w.toml` | `"provider/model"` (see [Connect your LLM](connect-your-llm.html)), a `shad0w.llm_teacher(...)`, or any function. Omit it to run [without an LLM](without-llm.html). |
| `api_key` | `str` or `() -> str` | the provider's variable | the key, or a function called on every request. Never printed. See [API keys](connect-your-llm.html#api-keys). |
| `api_key_env` | `str` | | the *name* of the environment variable that holds the key |
| `base_url` | `str` | the provider's URL | an OpenAI-compatible server |
| `folder` | path | `shad0w/<name>/` | where the log and the table live |
| `bundle` | path | `<folder>/bundle` | the table folder. It may not exist yet: `train()` creates it. |
| `log` | path | `<folder>/log.jsonl` | the JSON Lines file your LLM's answers are written to (the training data) |
| `**kw` | | | anything `Shadow` takes ([below](#shadow)), any [setting](configuration.html) (`alpha=0.02`, `mode="shadow"`, `auto_train=1000`, ...), and the [LLM settings](connect-your-llm.html#llm-settings) (`timeout`, `retries`, `system`, ...) |

Errors you may see when creating one:

- `TypeError` for an unknown keyword, with a suggestion (`apikey` → `did you mean 'api_key'?`);
- `TypeError` for `api_key`, `api_key_env` or `base_url` with a function `llm` (the function holds its own key);
- `ValueError` when the LLM's key is not set, naming the variable it looked for;
- `ValueError` for `llm="provider/model"` without `options` and without a trained table.

### Options {#options}

| You pass | Options are | Example |
|---|---|---|
| a list | the names | `["refund", "lost_card", "other"]` |
| a dict | the keys; values are descriptions the LLM sees | `{"refund": "wants money back", "other": "anything else"}` |
| `Literal[...]` | the literal values | `Literal["refund", "lost_card"]` |
| a string `Enum` (`StrEnum`, `class X(str, Enum)`) | the members' **values** | `class Intent(StrEnum): REFUND = "refund"` |
| any other `Enum` | the members' **names**; string values become descriptions | `class Intent(Enum): refund = "wants money back"` |
| `bool` | yes/no; answers are `True` / `False` | `options=bool` |
| a yes/no schema entry | yes/no, with instructions for the LLM | `{"type": "yesno", "instructions": "Is this spam?"}` |

```python
import enum, shad0w

class Intent(enum.StrEnum):
    REFUND = "refund"
    LOST_CARD = "lost_card"
    OTHER = "other"

intent = shad0w.decision("intent", options=Intent, llm="openai/gpt-6-luna")
```

Logs always hold the option string. [`@shad0w.decide`](#decorators) with `-> Intent` returns enum members.

### What it prints

`print(intent)` says what it is doing, and never shows the key:

```text
Shadow('intent', options=3, logging 240/1,000 answers (no table yet), llm=openai/gpt-6-luna, key: set via OPENAI_API_KEY (sk-…3f9a))
Shadow('intent', options=3, serving: certifies 68.0% at alpha=0.05, llm=openai/gpt-6-luna, key: set via OPENAI_API_KEY (sk-…3f9a))
```

In Jupyter it renders as a small table.

### Process-wide defaults

```python
import shad0w

shad0w.configure(llm="openai/gpt-6-luna", api_key_env="OPENAI_API_KEY", alpha=0.02)
intent = shad0w.decision("intent", options=["refund", "lost_card", "other"])   # uses them
shad0w.configure()                                                             # no arguments: clear them all
```

`configure()` takes `llm`, `api_key`, `api_key_env`, `base_url` and any [setting](configuration.html). Each call adds to the earlier ones. Its values rank below the keywords of one call and above `SHAD0W_*` variables and `shad0w.toml`. It returns the current defaults, with the key masked.

## `Decision`: what you get back {#decision}

Every call returns a frozen `Decision`. `str(d)` is the answer.

```python
d = intent("my card was stolen")
d
# Decision(answer='lost_card', source='table', confidence=0.99, certified=True, flag=None, latency_us=9.6,
#          question='intent', threshold=0.91, top=None)
d.why     # 'certified: answered by the table'
```

| Field | Type | Meaning |
|---|---|---|
| `answer` | `str` or `bool` | the option (a `bool` for yes/no questions) |
| `source` | `str` | who answered: `"table"` (locally, in microseconds), `"teacher"` (your LLM; the answer was logged) or `"fallback"` (your `fallback`, never logged) |
| `certified` | `bool` | `True` when the answer came from the table **and** is covered by the certificate. Serve-as-is answers. |
| `confidence` | `float` or `None` | the table's confidence in its top option, 0 to 1. `None` when no table was consulted. Also filled when your LLM answered: it shows how close the table was. |
| `threshold` | `float` or `None` | the certified threshold of the loaded table: the table answers at or above it. `None` without a table. |
| `flag` | `str` or `None` | why the table did not answer, or why a table answer is not certified. `None` for a certified answer. See [flags](#flags). |
| `why` | `str` | the flag in plain words, e.g. `"table not sure enough to stay inside the certified bound: asked your model"` |
| `latency_us` | `float` | the whole call in microseconds, including any LLM call |
| `question` | `str` | the decision's name |
| `top` | `list[(option, probability)]` or `None` | every option by probability, highest first; filled when `probabilities=True` |

### Flags {#flags}

| `flag` | Meaning | What to do |
|---|---|---|
| `None` | certified: answered by the table | nothing |
| `no_bundle` | no trained table yet | keep logging, then `train()` |
| `low_confidence` | below the certified threshold | normal; more data raises the share answered |
| `drift` | traffic looks different from what the table was certified on; clears when it looks familiar again | retrain if it persists |
| `low_radius` | (`exposed=True` only) a few inserted words could flip the answer | normal for adversarial inputs |
| `uncalibrated` | the table has no certificate | train with `train()` or `shad0w train` |
| `shadow` | `mode="shadow"`: the table would have answered; your LLM's answer was returned | switch to `serve` when ready ([Rollout](rollout.html)) |
| `canary` | certified, but held back by `canary` | raise `canary` |
| `never_serve` | the label is on `never_serve` | intended |
| `min_confidence` | certified, but below your `min_confidence` floor | intended |
| `options_changed` | your options include ones the table never learned | retrain ([Changing your options](options.html)) |
| `option_removed` | the table picked an option you removed | retrain |
| `manual_threshold` | served under `force_threshold`: **not** covered by the certificate (`certified=False`) | avoid in production |
| `off` | `mode="off"`: the table was not consulted | intended |

## Start from answers you already have {#warm-start}

```python
intent.warm_start("old_llm_answers.jsonl")                        # or .csv, .json, or a list of dicts
intent.warm_start(rows, text="message", label="category")         # other field names
# {'imported': 4210, 'skipped_unknown_label': 12, 'skipped_no_text': 0, 'unknown_labels': {'refunds': 12},
#  'log_rows': 4210, 'min_rows': 1000, 'ready_to_train': True}
intent.train()
```

| Argument | Default | Meaning |
|---|---|---|
| `rows_or_path` | | a list of dicts, or a `.jsonl`, `.json` or `.csv` file |
| `text` | `"text"` | the field that holds the text |
| `label` | the decision's name | the field that holds the answer |
| `source` | `"import"` | the `source` written to each log row |

Each label is checked against the options (spelling and case are forgiven, e.g. `Lost Card` → `lost_card`); unknown labels and empty texts are skipped and counted. Human labels work too: the certificate is then against those labels. For exported API traffic, see `shad0w import` on the [command line](cli.html).

## Batches

```python
results = intent.decide_many(texts, concurrency=8)             # list of Decision, same order as texts
results = await intent.adecide_many(texts, concurrency=8)      # the same, for async code
```

The table answers first, in microseconds each; only the texts it defers go to your LLM, at most `concurrency` at a time. The first LLM error is raised.

## Async

```python
import asyncio, shad0w

async def ask(text):                  # your async LLM call; returns one of the options
    await asyncio.sleep(0.1)
    return "lost_card" if "card" in text else "other"

intent = shad0w.decision("intent", options=["lost_card", "other"], llm=ask)

async def main():
    d = await intent.adecide("my card was stolen")
    print(d.answer, d.source)

asyncio.run(main())
```

The table path never leaves the event loop. An async LLM function is awaited and its spot checks run as tasks; a sync one runs in a worker thread.

## `shad0w.Shadow` {#shadow}

`decision()` builds a `Shadow` with the folder layout filled in. Use `Shadow` directly to choose every path yourself or to wrap an existing table:

```python
import shad0w

sh = shad0w.Shadow("bundle/", teacher=lambda text: "lost_card", log="log.jsonl")
d = sh("my card was stolen")          # same as sh.decide(text)
```

| Argument | Type | Default | What it does |
|---|---|---|---|
| `bundle` | path, `Model` or `None` | | the table. `None` = log only. A folder that does not exist yet is fine: `train()` creates it. |
| `teacher` | `fn(text) -> answer` | | your LLM call; an async one needs `adecide()` |
| `llm`, `api_key`, `api_key_env`, `base_url` | | | instead of `teacher`: a `"provider/model"` string and its key (needs `schema=` or a table for the options) |
| `question` | `str` | the table's only question | which question to answer |
| `log` | path | | JSON Lines file for your LLM's answers (the training data) |
| `schema` | `dict` | from the teacher or the table | the options, e.g. `{"type": "choice", "criteria": {"refund": None, ...}}` |
| `fallback` | value or `fn(text)` | | the answer when the table defers and there is no teacher (`source="fallback"`, never logged) |
| `probabilities` | `bool` | `False` | fill `Decision.top` on every call (slower) |
| `on_decision` | `fn(decision, text)` | | called after every decision (tracing, analytics) |
| `trace` | path | | JSON Lines file that receives **every** decision |
| `metrics` | `shad0w.Metrics` | a private one | share one across several deciders for one dashboard |
| `seed` | `int` | | random seed for spot checks and the canary |
| `config` | path | `./shad0w.toml` | the config file to read |
| any setting | | see [Configuration](configuration.html) | `alpha`, `delta`, `min_rows`, `audit_rate`, `auto_train`, `retrain`, `mode`, `canary`, `never_serve`, `min_confidence`, `force_threshold`, `drift_window`, `drift_margin`, `exposed`, `rename`, `on_new_option`, `max_mb`, `cost_per_call`, `llm_latency_ms`, ... |

Settings you do not pass come from `shad0w.configure()`, then `SHAD0W_*` environment variables, then `shad0w.toml`, then the defaults. `shad0w config` prints the result and where each value came from.

### Methods

| Method | Returns | What it does |
|---|---|---|
| `sh(text)`, `sh.decide(text, teacher=None, *, probabilities=None)` | `Decision` | the table's answer when certified, otherwise your LLM's (logged). `teacher=` overrides the LLM for this one call. |
| `await sh.adecide(text, teacher=None)` | `Decision` | the same, for async code |
| `sh.decide_many(texts, concurrency=8)` | `list[Decision]` | a batch: table first, deferred texts to your LLM in parallel |
| `await sh.adecide_many(texts, concurrency=8)` | `list[Decision]` | the same, for async code |
| `sh.peek(text)` | `Decision` or `None` | the table's answer if it would serve it; never calls your LLM |
| `sh.record(text, answer)` | `Decision` | log an answer you got from your LLM yourself (integrations use `peek` + `record`) |
| `sh.explain(text)` | `dict` | top-3 options, confidence vs threshold, the reason in words, and whether it would serve; never calls your LLM |
| `sh.train(out=None, alpha=None, delta=None, min_rows=None, *, gate=False)` | `dict` | compile and certify from the log, save, and start serving; returns the certificate plus `accepted`. `gate=True` keeps the current table unless the new one certifies at least 0.8 of its share. |
| `sh.warm_start(rows_or_path, text="text", label=None)` | `dict` | import answers or labels you already have ([above](#warm-start)) |
| `sh.stats(delta=0.1)` | `dict` | counts, `offload`, spot-check disagreement and its upper bound, p50/p99 latency, flags, `log_rows` |
| `sh.reload(bundle=None)` | `Shadow` | pick up a table retrained elsewhere (e.g. by `shad0w train`) |
| `sh.flush(timeout=None)` | | wait for spot checks still running in the background (tests, shutdown) |
| `sh.log_rows()` | `int` | rows in the log |
| `sh.close()` | | close the trace file |

Useful attributes: `sh.teacher` (with `.key_source` for a `"provider/model"` LLM), `sh.settings`, `sh.schema`, `sh.model`, `sh.options_added`, `sh.options_removed`.

## Decorators {#decorators}

The return annotation names the options and the function body is the teacher:

```python
from typing import Literal
import shad0w

@shad0w.decide()
def route(text) -> Literal["billing", "tech", "sales"]:
    return "billing"                 # your LLM call goes here

route("my invoice is wrong")         # "billing": from the table once trained and certified, else from the body (logged)
route.shadow.train()                 # route.shadow is the Shadow; train after ~1,000 logged answers
```

- `-> bool` makes a yes/no question.
- `-> Intent` (an `Enum`) makes the enum's options and returns enum members; the body may return a member or its option string.
- Files live in `shad0w/<function name>/`, like `decision()`. `@shad0w.decide(name="...", folder="...")` changes that, and any other `decision()` keyword works too.

Or let shad0w call the LLM. Then the body is never run:

```python
@shad0w.decide(llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])
def route(text) -> Literal["billing", "tech", "sales"]: ...
```

To wrap an existing table and log, use `@shad0w.cascade`:

```python
@shad0w.cascade("bundle/", log="log.jsonl")
def classify(text):
    return ask_llm(text)

classify("block my card")            # the answer; classify.shadow.stats() for counters
```

## Several questions per text

The simple way: one `decision()` per question (`shad0w.decision("intent", ...)`, `shad0w.decision("urgent", ...)`). For one table with several questions, compile from a log that has one field per question (`shad0w train` does this), then use the low-level model:

```python
m = shad0w.load("bundle/")
m.decide("I lost my card, block it now!")
# {"answers": {"intent": {"choice": "lost_card", "confidence": 0.97, "certified": True, "flag": None, "radius": 6,
#                         "probabilities": {...}},
#              "urgent": {"answer": True, "probability": 0.93, "confidence": 0.93, "certified": True, ...}}}
```

| Call | What it does |
|---|---|
| `shad0w.load(path, native=True)` | load a table folder; `native=False` forces pure numpy (same decisions) |
| `m.decide(text, exposed=False, questions=None, probabilities=True)` | every question's answer; `questions={"intent": {"criteria": ["refund", "lost_card"]}}` narrows one to a subset |
| `m.decide(text, probabilities=False)` | skips the per-option dictionary: the fastest path (2.4 µs through the C core on an Apple M5 Pro) |
| `m.native` | `True` when every question decides through the C core |

The low-level model applies no rollout settings and calls no LLM: serve an answer when `certified` is true, otherwise ask your LLM.

## Thread safety

`Shadow`, `Model` and the HTTP servers are safe to share across threads. The C core is guarded by a lock per question. Create a decider once (at import time in a web app) and reuse it.
