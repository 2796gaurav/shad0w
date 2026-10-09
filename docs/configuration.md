---
title: Configuration
description: Every shad0w setting with its default, type, SHAD0W_* variable and shad0w.toml key, how the layers combine, and a complete commented shad0w.toml.
---
# Configuration

<p class="lead">This page lists every setting shad0w has, where you can set it, and which value wins when it is set in more than one place. You need none of it to start: every setting has a safe default.</p>

**TL;DR**

```bash
shad0w config --init            # write a commented shad0w.toml (every key, all commented out)
shad0w config                   # every effective value and where it came from
shad0w config --question intent # the same, with [questions.intent] applied
```

```python
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"],
                         llm=lambda text: "refund",       # any function works; see Connect your LLM
                         alpha=0.02, mode="shadow")       # settings are plain keywords
print(intent.settings.alpha, intent.settings.mode)        # 0.02 shadow: what this decision uses
```

A few words used below. **The table** is the small model shad0w trains from your LLM's answers. **Certified** means the table was tested on logged answers it never trained on, and serves only answers where it is expected to differ from your LLM on at most **α** of them (with 90% confidence by default). **Defer** means "send this one to your LLM".

## Where a setting can come from

Each setting can be set in five places. The highest one that sets it wins:

<div class="viz" data-viz="precedence">Highest first: (1) a keyword in code or a CLI flag, such as alpha=0.02 or --alpha 0.02; (2) a SHAD0W_* environment variable, such as SHAD0W_ALPHA=0.02; (3) a [questions.intent] section in shad0w.toml, which applies to that one question; (4) the top level of shad0w.toml; (5) the built-in default. shad0w.configure(...) sits between the keywords of one call and the environment.</div>

| Rank | Where | Example | Applies to |
|---|---|---|---|
| 1 | A keyword in code, or a CLI flag | `decision("intent", ..., alpha=0.02)`, `shad0w train --alpha 0.02` | that one call or process |
| 1b | `shad0w.configure(...)` in code | `shad0w.configure(alpha=0.02)` | every `decision()` made after it in this process |
| 2 | An environment variable | `SHAD0W_ALPHA=0.02` | every question in the process |
| 3 | A `[questions.<name>]` section of `shad0w.toml` | `[questions.intent]` then `alpha = 0.02` | one question |
| 4 | The top level of `shad0w.toml` | `alpha = 0.02` | every question |
| 5 | The default | `0.05` | — |

Things that surprise people:

- **An environment variable beats a per-question section.** `SHAD0W_ALPHA=0.1` overrides `alpha = 0.02` under `[questions.fraud]`. Use environment variables for whole-process switches (`SHAD0W_MODE=off`), and the file for per-question tuning.
- **`shad0w.configure()` with no arguments clears** everything set with it before.
- **The config file** is `./shad0w.toml` in the folder your program runs from, or the path in `SHAD0W_CONFIG`, or the `config=` keyword / `--config` flag. A path given explicitly must exist (`config file ... not found`); a missing `./shad0w.toml` is simply skipped.
- **Values are checked when they are read.** A bad value stops start-up with a message that names the setting, for example `mode must be one of ('serve', 'shadow', 'off'), got 'on'`. A misspelt keyword in code gets a suggestion: `decision() got unexpected keyword argument(s): 'alpah' (did you mean 'alpha'?)`. Unknown keys in `shad0w.toml` are ignored, so check spelling with `shad0w config`.
- Reading `shad0w.toml` needs Python 3.11+, or `pip install tomli` on 3.10.

To see the result for one question from Python, without the keywords of any one call:

```python
import shad0w
from shad0w.config import explain

print(shad0w.settings("intent"))                       # a frozen Settings object
for key, value, source in explain("intent"):
    print(key, value, source)                          # e.g.  alpha 0.02 toml:shad0w.toml [questions.intent]
```

## How environment variables are written

The variable is `SHAD0W_` plus the setting name in capitals: `alpha` → `SHAD0W_ALPHA`, `never_serve` → `SHAD0W_NEVER_SERVE`.

| Setting type | Write it as | Example |
|---|---|---|
| number | the number | `SHAD0W_ALPHA=0.02`, `SHAD0W_MIN_ROWS=500` |
| true / false | `1`, `true`, `yes` or `on` for true; anything else is false | `SHAD0W_EXPOSED=1` |
| list | comma-separated | `SHAD0W_NEVER_SERVE=fraud,self_harm` |
| rename map | `old=new` pairs, comma-separated | `SHAD0W_RENAME=lost_card=card_lost,topup=top_up` |
| "not set" | `none`, `null` or empty | `SHAD0W_MIN_CONFIDENCE=none` |
| `auto_train` | a number; `off`, `false` or `no` mean 0 | `SHAD0W_AUTO_TRAIN=off` |

## Every setting

In `shad0w.toml` the key is the setting's name (`alpha = 0.02`). Ranges are enforced: a value outside them is an error.

### The guarantee

| Setting | Env var | Type | Default | What it does | Change it when |
|---|---|---|---|---|---|
| <a id="alpha"></a>`alpha` | `SHAD0W_ALPHA` | number, 0–1 | `0.05` | The most the table may disagree with your LLM on the answers it serves. `0.05` = 1 in 20. Lower is safer and serves less. | A wrong answer is costly: try `0.02` or `0.01`. Retries are cheap and you want more answered: `0.1`. See [Tuning](tuning.html#alpha). |
| <a id="delta"></a>`delta` | `SHAD0W_DELTA` | number, 0–1 | `0.1` | The chance that the certificate itself is wrong because the test sample was unlucky. `0.1` = 90% confidence. | Rarely. `0.05` for 95% confidence; the table then serves a little less. |
| <a id="min_rows"></a>`min_rows` | `SHAD0W_MIN_ROWS` | whole number, ≥ 100 | `1000` | Logged LLM answers needed before training will run. | To experiment with a small log (100 is the hard minimum). |
| <a id="cal_fraction"></a>`cal_fraction` | `SHAD0W_CAL_FRACTION` | number, 0.01–0.9 | `0.3` | Share of the log held back to test (certify) the table. Those rows are never trained on. | Rarely. Only used when there are fewer than 100 spot-check rows (see [Training](training.html#why-retraining-certifies-on-spot-checks)). |
| <a id="max_cal"></a>`max_cal` | `SHAD0W_MAX_CAL` | whole number | `3000` | Upper limit on that held-back slice, so huge logs still train on most rows. | Rarely. |

### Rollout and serving

| Setting | Env var | Type | Default | What it does | Change it when |
|---|---|---|---|---|---|
| <a id="mode"></a>`mode` | `SHAD0W_MODE` | `serve`, `shadow` or `off` | `"serve"` | `serve`: the table answers what it is certified on. `shadow`: the table decides but your LLM's answer is always returned, and shad0w counts how often they would have differed. `off`: the table is not consulted. | Start new questions in `shadow`; `off` is the kill switch. See [Rollout](rollout.html). |
| <a id="canary"></a>`canary` | `SHAD0W_CANARY` | number, 0–1 | `1.0` | Share of the certified answers the table may actually serve. The rest go to your LLM (flag `canary`). | `0.1` for the first days of serving. |
| <a id="never_serve"></a>`never_serve` | `SHAD0W_NEVER_SERVE` | list of labels | `[]` | Labels that always go to your LLM, even when the table is sure (flag `never_serve`). | High-stakes labels such as `fraud` or `self_harm`. |
| <a id="min_confidence"></a>`min_confidence` | `SHAD0W_MIN_CONFIDENCE` | number 0–1, or none | none | An extra confidence floor on top of the certificate. Only ever stricter (flag `min_confidence`). | You want a margin above the certified threshold. |
| <a id="force_threshold"></a>`force_threshold` | `SHAD0W_FORCE_THRESHOLD` | number 0–1, or none | none | Serve above this confidence **even where the certificate says no**. Those answers carry `certified=False` and flag `manual_threshold`; a warning is logged at start-up. | Experiments only. It voids the guarantee for those answers. |
| <a id="exposed"></a>`exposed` | `SHAD0W_EXPOSED` | true / false | `false` | Treat inputs as adversarial: also defer answers that a few character edits could flip (flag `low_radius`). | Users may try to game the decision (moderation, fraud). |

### Spot checks, retraining and drift

| Setting | Env var | Type | Default | What it does | Change it when |
|---|---|---|---|---|---|
| <a id="audit_rate"></a>`audit_rate` | `SHAD0W_AUDIT_RATE` | number, 0–1 | `0.01` | Share of all decisions also checked against your LLM (**spot checks**). Gives the live agreement figure and an honest sample for the next certificate. | `0.05` for the first weeks of serving; `0` only if you cannot afford any extra LLM call. |
| <a id="auto_train"></a>`auto_train` | `SHAD0W_AUTO_TRAIN` | whole number | `0` (off) | Retrain in the background every N new logged LLM answers. Needs `shad0wllm[compile]`. | You want the table to keep up without a cron job, e.g. `2000`. |
| <a id="retrain"></a>`retrain` | `SHAD0W_RETRAIN` | `gated` or `always` | `"gated"` | For `auto_train`: `gated` keeps the current table unless the new one certifies at least 80% of its share; `always` swaps whatever comes out. | Almost never. |
| <a id="drift_window"></a>`drift_window` | `SHAD0W_DRIFT_WINDOW` | whole number, ≥ 10 | `500` | How many recent decisions the drift guard looks at. | Bursty traffic trips the guard: raise to `2000`. |
| <a id="drift_margin"></a>`drift_margin` | `SHAD0W_DRIFT_MARGIN` | number | `0.03` | How much the estimated error may rise before the `drift` flag sends answers to your LLM. | Too sensitive: `0.05`. |

### Options and size

| Setting | Env var | Type | Default | What it does | Change it when |
|---|---|---|---|---|---|
| <a id="on_new_option"></a>`on_new_option` | `SHAD0W_ON_NEW_OPTION` | `defer` or `serve` | `"defer"` | You added an option the table never learned. `defer` sends every decision to your LLM until you retrain; `serve` keeps serving the options it knows; those answers say `certified=False`, flag `new_options_served`, because the certificate does not cover them. | Rarely. See [Changing your options](options.html). |
| <a id="rename"></a>`rename` | `SHAD0W_RENAME` | table of `old = "new"` | `{}` | Rename labels without retraining. The table answers with the new names at once; old rows in the log are read under the new names when you next train. | You renamed or merged an option. |
| <a id="max_mb"></a>`max_mb` | `SHAD0W_MAX_MB` | number 0.01–4096, or none | none (no cap) | A size budget per table, in MB. Training keeps the most informative features; the certificate is computed on the capped table. | Many options or a large vocabulary make the table big. See [Scale](scale.html). |

### Your LLM

| Setting | Env var | Type | Default | What it does | Change it when |
|---|---|---|---|---|---|
| <a id="llm"></a>`llm` | `SHAD0W_LLM` | `"provider/model"` | none | The LLM `decision()` uses when you don't pass `llm=`. | You want one place to switch models, e.g. `"openai/gpt-6-luna"`. |
| <a id="base_url"></a>`base_url` | `SHAD0W_BASE_URL` | URL | the provider's | An OpenAI-compatible server for that LLM. | vLLM, Ollama, a gateway, a decision model. |
| <a id="api_key_env"></a>`api_key_env` | `SHAD0W_API_KEY_ENV` | variable name | the provider's (`OPENAI_API_KEY`, …) | The **name** of the environment variable that holds the key. Never the key itself. | Your key lives in a differently named variable. |

There is no `api_key` setting. A key in a file ends up in a repository sooner or later, so an `api_key = ...` line in `shad0w.toml` is refused with an error. Pass `api_key=` in code if you must (a string, or a function that returns one). See [API keys](connect-your-llm.html#api-keys).

### Files, dashboard and proxy

| Setting | Env var | Type | Default | What it does | Change it when |
|---|---|---|---|---|---|
| <a id="folder"></a>`folder` | `SHAD0W_FOLDER` | path | `"shad0w"` | Where logs and tables live: `<folder>/<question>/log.jsonl` and `<folder>/<question>/bundle/`. Relative to where your program runs. | Several apps share a machine, or a volume is mounted elsewhere. |
| <a id="trace"></a>`trace` | `SHAD0W_TRACE` | path, or none | none | A JSON Lines file that receives every decision, table and LLM. Never used for training. | You want a full audit trail. See [Monitoring](observability.html#hooks-and-traces). |
| <a id="cost_per_call"></a>`cost_per_call` | `SHAD0W_COST_PER_CALL` | number, or none | none | What one LLM call costs you. The dashboard then shows money saved. | Always worth setting on a dashboard. |
| <a id="llm_latency_ms"></a>`llm_latency_ms` | `SHAD0W_LLM_LATENCY_MS` | number, or none | none | LLM latency to assume until one is measured, so "time saved" shows from the first table answer. | Before the first LLM call is measured. |
| <a id="capture"></a>`capture` | `SHAD0W_CAPTURE` | list of `header`, `model`, `tools`, `decisions`, `json_schema` | `header, model, tools, decisions` | Proxy only: which request shapes count as decisions. | Add `json_schema` to capture every request with a named JSON schema. See [The proxy](proxy.html). |
| <a id="timeout"></a>`timeout` | `SHAD0W_TIMEOUT` | seconds | `120` | Proxy only: upstream request timeout. | Slow upstream models. |

## A complete shad0w.toml

`shad0w config --init` writes every key, commented out, with a one-line explanation. `shad0w init` writes a shorter one next to a runnable `app.py`. Here is a realistic file with every section in use:

```toml
# shad0w.toml: read from the folder your program runs in (or $SHAD0W_CONFIG).
# Precedence: code keywords / CLI flags > shad0w.configure() > SHAD0W_* environment > [questions.X] > this top level > defaults.

# --- your LLM -------------------------------------------------------------------------------------------------
llm = "openai/gpt-6-luna"          # used by decision() when llm= is not passed
api_key_env = "OPENAI_API_KEY"     # the NAME of the variable holding the key; never the key itself
# base_url = "http://localhost:8000/v1"   # an OpenAI-compatible server instead of the provider's

# --- the guarantee --------------------------------------------------------------------------------------------
alpha = 0.05                       # at most 1 in 20 table answers may differ from your LLM
delta = 0.1                        # ... with 90% confidence
min_rows = 1000                    # logged answers before training runs (100 is the minimum)

# --- rollout --------------------------------------------------------------------------------------------------
mode = "shadow"                    # watch first; switch to "serve" when the numbers look right
canary = 1.0                       # 0.1 = the table answers 10% of what it is certified on
never_serve = []                   # labels that always go to your LLM
# min_confidence = 0.97            # an extra floor; only ever stricter than the certificate

# --- spot checks and retraining -------------------------------------------------------------------------------
audit_rate = 0.05                  # higher during the first weeks; 0.01 afterwards
auto_train = 0                     # e.g. 2000 = retrain every 2,000 new answers (off by default)
retrain = "gated"                  # keep the old table unless the new one is at least as useful

# --- files and dashboard --------------------------------------------------------------------------------------
folder = "shad0w"                  # <folder>/<question>/log.jsonl and .../bundle/
# trace = "decisions.jsonl"        # every decision, table and LLM
# cost_per_call = 0.0006           # dashboard: money saved

# --- one question at a time -----------------------------------------------------------------------------------
[questions.fraud_check]
alpha = 0.01                       # stricter for a costly decision
never_serve = ["fraud"]

[questions.intent]
rename = { lost_card = "card_lost" }   # renamed an option: no retraining needed
```

Check it:

```bash
shad0w config --question fraud_check
```

```text
alpha            0.01                          toml:shad0w.toml [questions.fraud_check]
...
mode             "shadow"                      toml:shad0w.toml
never_serve      ["fraud"]                     toml:shad0w.toml [questions.fraud_check]
...
config file: shad0w.toml
llm key: set via OPENAI_API_KEY (sk-…3f9a)
```

## Other environment variables

These are not settings, so they have no `shad0w.toml` key.

| Variable | What it does |
|---|---|
| `SHAD0W_CONFIG` | Path of the config file, instead of `./shad0w.toml`. |
| `SHAD0W_LOG` | `info` for training, reloads and spot-check disagreements; `debug` adds one line per decision. See [Monitoring](observability.html#logs). |
| `SHAD0W_OTEL` | `1` to emit one OpenTelemetry span per decision. |
| `SHAD0W_NATIVE_LIB` | Path to a C core library to load instead of the bundled one. |
