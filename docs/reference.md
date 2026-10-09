---
title: Reference
description: Every public Python function and class in shad0w with exact signatures, parameters, return values and errors; the Decision fields and flags; every setting; the log, bundle, manifest and certificate file formats; the JavaScript exports; and the HTTP routes.
---
# Reference

Everything shad0w exposes, on one page. Signatures are copied from the code by introspection (shad0w 0.3.2). The settings and flags tables are generated from the package source at build time, so they always match the release.

Terms used below: **the table** is shad0w's small learned model; **your LLM** (also called the *teacher* in code) is the model whose answers it learns from; **certified** means the answer is covered by the guarantee (it differs from your LLM on at most α of answers, with 90% confidence); **defer** means the table does not answer and the message goes to your LLM. See [the guarantee](guarantee.html).

## At a glance

Everything in `shad0w.__all__`:

| Name | Kind | What it is for |
|---|---|---|
| [`decision`](#shad0wdecision) | function | Start here: one question in front of your LLM, with its own folder, log and table. |
| [`Shadow`](#shad0wshadow) | class | What `decision()` returns: decide, log, train, explain, stats. |
| [`Decision`](#decision) | dataclass | One answer, where it came from and why. |
| [`decide`](#shad0wdecide) | decorator | Options from a return annotation, the function body as your LLM. |
| [`cascade`](#shad0wcascade) | decorator | Wrap an existing function with a `Shadow` around a bundle path. |
| [`explain`](#shad0wexplain) | function | A flag in plain words. |
| [`configure`](#shad0wconfigure) | function | Process-wide defaults (LLM, key, settings). |
| [`settings`](#shad0wsettings) | function | The effective settings for a question. |
| [`Settings`](#shad0wsettings-class) | dataclass | The resolved settings object. |
| [`Secret`](#shad0wsecret) | class | An API key that never prints. |
| [`llm_teacher`, `LLMTeacher`](#shad0wllm_teacher-and-llmteacher) | function, class | Your LLM as a callable that returns one of your options. |
| [`TeacherError`](#shad0wteachererror) | exception | The LLM failed or replied with no valid option. |
| [`Metrics`](#shad0wmetrics) | class | Counters behind `stats()`, the dashboard and Prometheus. |
| [`compile`](#shad0wcompile) | function | Build a table from human labels. |
| [`load`](#shad0wload), [`Model`](#shad0wmodel) | function, class | Load a bundle and decide with it directly (no LLM, no logging). |
| [`shadow_compile`](#shad0wshadow_compile) | function | Build and certify a table from logged LLM answers, without a `Shadow`. |
| [`certify_bundle`](#shad0wcertify_bundle) | function | Re-certify a bundle on fresh LLM answers. |
| [`read_certificate`](#shad0wread_certificate) | function | Read a bundle's `certificate.json`. |
| `__version__` | string | `"0.3.2"` |

A complete offline run of the main calls (a keyword function stands in for your LLM; it reuses the table if you ran the example on [the guarantee page](guarantee.html#how-a-certificate-is-made), otherwise every call goes to the function and is logged):

```python
import shad0w

def llm(text):  # a stand-in for your LLM, so this runs offline
    return "refund" if "refund" in text else "lost_card" if "stolen" in text else "other"

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=llm)

d = intent("my card was stolen please help")
print(d.answer, d.source, d.certified, d.flag, d.why)
print([x.source for x in intent.decide_many(["a refund please", "stolen wallet", "nice weather"])])
print(intent.explain("refund please")["why"])
print(intent.stats()["offload"], repr(intent))
print(shad0w.explain("drift"))
```

---

## `shad0w.decision`

```text
shad0w.decision(name: str = 'decision', options: Any = None, llm: str | Callable[[str], Any] | None = None, *,
                api_key: Any = None, api_key_env: str | None = None, base_url: str | None = None,
                bundle: str | None = None, log: str | None = None, folder: str | None = None, **kw) -> Shadow
```

One call from nothing to a certified cascade. Day one, every call goes to your LLM and is logged. After `train()`, the table answers what it is certified for and your LLM answers the rest.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `name` | `str` | `'decision'` | What the decision is called. Names its folder (`shad0w/<name>/`), its field in the log, the question in the bundle, the dashboard label and the Decisions API question. Use one name per decision. |
| `options` | list, dict, `Enum`, `Literal[...]`, `bool`, or schema entry | `None` | The answers to choose from: `["a", "b"]`; `{"a": "description", ...}`; an `Enum` class (string enums use values, others use names with string values as descriptions); `Literal["a", "b"]`; `bool` for yes/no; or `{"type": "choice" \| "yesno", "criteria": {...}, "instructions": "..."}`. Optional when a trained bundle with `schema.json` exists. Compared with the trained table to catch [changes](options.html). |
| `llm` | `str` or callable | `None` | Your LLM: `"provider/model"` ([providers](connect-your-llm.html)), `shad0w.llm_teacher(...)`, or any `fn(text) -> answer` (sync, or async with `adecide()`). Default: `configure(llm=...)`, then the `llm` setting. Omit it to run [without an LLM](without-llm.html). |
| `api_key` | `str` or zero-argument callable | `None` | The key for a `"provider/model"` LLM. A function is called on every request (rotating or vault keys). Never printed; shows as `sk-…3f9a` at most. |
| `api_key_env` | `str` | `None` | The *name* of the environment variable holding the key. Precedence: `api_key` > `api_key_env` > `configure()` > the `api_key_env` setting > the provider's usual variable. |
| `base_url` | `str` | `None` | An OpenAI-compatible server. Default: `configure()`, `SHAD0W_BASE_URL` or the setting, then the provider's URL. |
| `bundle` | `str` | `<folder>/bundle` | Where the trained table lives. A path that does not exist yet is fine. |
| `log` | `str` | `<folder>/log.jsonl` | The JSON Lines file of LLM answers (the training data). |
| `folder` | `str` | `<folder setting>/<name>` | Base folder for `bundle` and `log` (the `folder` setting defaults to `shad0w`). |
| `**kw` | | | Any [setting](#settings) (`alpha=0.02`, `canary=0.1`, `max_mb=1`, ...); `Shadow` keywords (`fallback`, `on_decision`, `trace`, `metrics`, `probabilities`, `seed`, `config`); and, for a `"provider/model"` LLM, the `LLMTeacher` keywords (`client`, `complete`, `system`, `temperature`, `timeout`, `retries`, `headers`, `structured`, `max_tokens`). |

**Returns** a [`Shadow`](#shad0wshadow).

**Raises**

- `TypeError`: an unknown keyword (with a "did you mean" hint); `settings=` passed; key or LLM keywords together with a function `llm` (the function holds its own key); options in an unsupported form.
- `ValueError`: a `"provider/model"` LLM without `options` and without an existing `schema.json`; an invalid setting value; a `"provider/model"` LLM with no key found.

```python
import os
import shad0w

intent = shad0w.decision("intent", options={"refund": "wants money back", "lost_card": "card lost or stolen"},
                         llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"], alpha=0.05)
```

## `shad0w.Shadow`

```text
shad0w.Shadow(bundle: str | Model | None, teacher: Callable[[str], Any] | None = None, question: str | None = None,
              log: str | None = None, *, llm: str | Callable[[str], Any] | None = None, api_key: Any = None,
              api_key_env: str | None = None, base_url: str | None = None, seed: int | None = None,
              schema: dict | None = None, on_decision: Callable[[Decision, str], Any] | None = None,
              trace: str | TraceWriter | None = None, metrics: Metrics | None = None, probabilities: bool = False,
              config: str | None = None, settings: Settings | None = None, fallback: Any = <no fallback>, **kw)
```

A certified cascade around one question: the table when it is certified, your LLM otherwise. `decision()` builds one for you; build it directly when you manage paths yourself.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `bundle` | `str`, `Model` or `None` | required | A bundle folder, a loaded `Model`, or `None` (log only). A folder without `manifest.json` yet starts log-only; `train()` creates it. |
| `teacher` | callable | `None` | Your LLM as `fn(text) -> answer`. Pass this or `llm`, not both. |
| `question` | `str` | `None` | Which question of the bundle to answer. Default: the teacher's `question`, else the bundle's only question. |
| `log` | `str` | `None` | JSON Lines file that receives every LLM answer. Without it nothing is logged and `train()` cannot run. |
| `llm` | `str` or callable | `None` | Instead of `teacher`: a `"provider/model"` string (needs `schema` or a bundle for the options) or a callable. |
| `api_key`, `api_key_env`, `base_url` | | `None` | As in [`decision`](#shad0wdecision); only with a `"provider/model"` string. |
| `seed` | `int` | `None` | Random seed for spot checks and the canary. |
| `schema` | `dict` | `None` | The options, as `{"type": "choice", "criteria": {...}}` or `{"type": "yesno"}`. Taken from the teacher or the bundle when omitted. |
| `on_decision` | callable | `None` | `fn(decision, text)` called after every decision. Errors in it are logged, not raised. |
| `trace` | `str` or `TraceWriter` | `None` | JSON Lines file that receives *every* decision ([format](#trace-file)). Default: the `trace` setting. |
| `metrics` | `Metrics` | a private one | Share one `Metrics` across questions for one dashboard. |
| `probabilities` | `bool` | `False` | Fill `Decision.top` on every call (a little slower). |
| `config` | `str` | `None` | Path to a `shad0w.toml` (default `./shad0w.toml` or `$SHAD0W_CONFIG`). |
| `settings` | `Settings` | `None` | Already-resolved settings; then no setting keywords may be passed. |
| `fallback` | value or callable | no fallback | The answer when the table defers and there is no LLM: a value such as `"needs_review"`, or `fn(text) -> answer`. Never logged. |
| `**kw` | | | Any [setting](#settings), plus the `LLMTeacher` keywords for a `"provider/model"` LLM. |

**Raises** `TypeError` (unknown keyword, `teacher` and `llm` together, a non-callable teacher, key keywords with a function LLM) and `ValueError` (the bundle does not contain `question`, or has several questions and none was named).

### Methods

| Method | Returns | What it does |
|---|---|---|
| `decide(text: str, teacher: Callable[[str], Any] \| None = None, *, probabilities: bool \| None = None)` | `Decision` | The table's answer when certified; else your LLM's (logged); else the fallback. `teacher` overrides the LLM for this one call. Also `sh(text)`. |
| `adecide(text: str, teacher: Callable[[str], Any] \| None = None)` | `Decision` (awaitable) | `decide` for async code. The table runs on the event loop; an async LLM is awaited, a sync one runs in a worker thread. |
| `decide_many(texts, concurrency: int = 8, teacher: Callable[[str], Any] \| None = None)` | `list[Decision]` | A batch, in order. The table answers first; only deferred texts go to your LLM, `concurrency` at a time in threads. The first LLM error is raised. |
| `adecide_many(texts, concurrency: int = 8, teacher: Callable[[str], Any] \| None = None)` | `list[Decision]` (awaitable) | The same for async code. |
| `peek(text: str, *, probabilities: bool \| None = None)` | `Decision \| None` | The table's answer when it would be served, else `None`. Never calls your LLM. |
| `record(text: str, answer, *, teacher_s: float \| None = None)` | `Decision` | Log an answer you got yourself (a person, your own LLM call). `teacher_s` is how long it took, for latency stats. |
| `explain(text: str)` | `dict` | What the table thinks, without calling your LLM: `text`, `question`, `answer`, `confidence`, `threshold`, `alpha`, `certified`, `flag`, `why`, `top` (3 best options), plus `would_serve`, `policy_flag` and `mode` after the rollout settings. Does not touch the drift guard. |
| `train(out: str \| None = None, alpha: float \| None = None, delta: float \| None = None, min_rows: int \| None = None, *, gate: bool = False)` | `dict` | Compile and certify from the log, save to `out` (default: the bundle path), and start serving from it. Returns this question's [certificate entry](#certificatejson) plus `accepted` (and `reason` when `gate=True` kept the old table). |
| `stats(delta: float = 0.1)` | `dict` | Live counters: `table`, `teacher`, `fallback`, `offload` (table ÷ (table + teacher)), `audits`, `audit_disagreements`, `audit_disagreement`, `audit_disagreement_upper` (Clopper–Pearson upper bound at 1 − `delta`), `would_serve` and `shadow_disagreements` (shadow mode), `table_p50_us`, `table_p99_us`, `llm_p50_ms`, `llm_p99_ms`, `flags`, `alpha`, `threshold`, `mode`, `canary`, `log_rows`, `options_added`, `options_removed`. |
| `warm_start(rows_or_path, text: str = 'text', label: str \| None = None, *, source: str = 'import')` | `dict` | Import answers or labels you already have (list of dicts, `.jsonl`, `.json`, `.csv`) into the log. Labels are checked against the options. Returns `imported`, `skipped_unknown_label`, `skipped_no_text`, `unknown_labels`, `log_rows`, `min_rows`, `ready_to_train`. |
| `reload(bundle: str \| Model \| None = None)` | `Shadow` | Swap in a bundle retrained elsewhere, without restarting. No argument: reload this wrapper's bundle path. |
| `flush(timeout: float \| None = None)` | `None` | Wait for background spot checks (tests, shutdown). |
| `log_rows()` | `int` | Lines in the log. |
| `close()` | `None` | Close the trace file. |

**`train` raises** `ValueError` (no `out` and no bundle path, no log file, fewer than `min_rows` usable answers, fewer than 2 distinct answers) and `ImportError` when the training extras are missing (`pip install "shad0wllm[compile]"`).

**`decide` raises** whatever your LLM raises (`TeacherError` from `llm_teacher`, with the reason it was asked appended), and `RuntimeError` when the table defers and there is neither an LLM nor a `fallback`.

**Attributes**: `question`, `settings` (a `Settings`), `model` (the loaded `Model` or `None`), `bundle_path`, `log`, `teacher`, `metrics`, `schema`, `options_added` and `options_removed` (tuples of option names changed since training; see [options](options.html)). `repr(sh)` and the notebook view show the question, the number of options, the state, the LLM and where its key comes from (masked).

## `Decision`

```text
shad0w.Decision(answer: Any, source: str, confidence: float | None, certified: bool, flag: str | None,
                latency_us: float, question: str = '', threshold: float | None = None,
                top: list[tuple[str, float]] | None = None) -> None
```

A frozen dataclass returned by every decide call. `str(d)` is the answer.

| Field | Type | Meaning |
|---|---|---|
| `answer` | `str` or `bool` | The option name, or `True`/`False` for a yes/no question. (A function wrapped with `@shad0w.decide` and `Enum` options returns the `Enum` member instead.) |
| `source` | `str` | `"table"` (answered locally), `"teacher"` (your LLM answered, and it was logged) or `"fallback"`. |
| `confidence` | `float \| None` | The table's confidence in its top option; `None` when no table was consulted. |
| `certified` | `bool` | `True` when the answer is covered by the certificate. Only table answers can be. |
| `flag` | `str \| None` | Why the table did not answer, or why a table answer is not certified. See [flags](#flags). |
| `latency_us` | `float` | Time for this call, in microseconds, LLM call included. |
| `question` | `str` | The question's name. |
| `threshold` | `float \| None` | The certified confidence threshold of the loaded table; `None` without a table or when it certifies nothing. |
| `top` | `list[tuple[str, float]] \| None` | Options by probability, highest first, when `probabilities=True`. |
| `why` | `str` (property) | The reason for this answer in plain words. |

### Flags

| Flag | Meaning |
|---|---|
| `no_bundle` | no trained table yet: asked your model |
| `low_confidence` | table not sure enough to stay inside the certified bound: asked your model |
| `low_radius` | input could be flipped by a few edits (exposed mode): asked your model |
| `empty_input` | the message is empty: asked your model |
| `drift` | traffic looks different from calibration: asked your model until the window recovers or you re-train |
| `uncalibrated` | table has no certificate: asked your model |
| `min_confidence` | below your min_confidence floor (stricter than the certificate): asked your model |
| `never_serve` | this label is on never_serve: asked your model |
| `canary` | held back by the canary share: asked your model |
| `shadow` | shadow mode: the table's answer was recorded, your model's was returned |
| `off` | mode=off: the table was not consulted |
| `manual_threshold` | served under force_threshold: NOT covered by the certificate |
| `options_changed` | your options include ones the table never learned: asked your model until you retrain |
| `new_options_served` | served although you added options the table never learned (on_new_option="serve"): NOT covered by the certificate |
| `option_removed` | the table picked an option you removed: asked your model |

## `shad0w.decide`

```text
shad0w.decide(fn: Callable | None = None, /, *, name: str | None = None, folder: str | None = None, **kw)
```

Decorator. The function's return annotation names the options (`Literal[...]`, an `Enum` class, or `bool` for yes/no); its body is your LLM. Files live in `<folder>/<name>/` as with `decision()`; `name` defaults to the function name. Pass `llm="provider/model"` to let shad0w call the LLM instead; then the body never runs. Other keywords go to `decision()`. The wrapped function returns the answer (an `Enum` member for `Enum` options) and has a `.shadow` attribute with the `Shadow`.

**Raises** `TypeError` when the return annotation is missing or not one of those forms.

```python
from typing import Literal

import shad0w

@shad0w.decide()
def route(text) -> Literal["billing", "tech", "sales"]:
    return "billing" if "invoice" in text else "tech"   # your LLM call goes here

print(route("my invoice is wrong"))   # from your function now (logged); from the table once trained
route.shadow.flush()
```

## `shad0w.cascade`

```text
shad0w.cascade(bundle: str | Model | None, question: str | None = None, log: str | None = None,
               audit_rate: float | None = None, exposed: bool | None = None, **kw)
```

Decorator form of `Shadow` for a function you already have: `@shad0w.cascade("bundle/", log="log.jsonl")`. The wrapped function returns the answer and has a `.shadow` attribute. Other keywords go to `Shadow`.

## `shad0w.explain`

```text
shad0w.explain(flag: str | None) -> str
```

The meaning of a [flag](#flags) in plain words. `shad0w.explain(None)` is `"certified: answered by the table"`. Unknown flags come back unchanged.

## `shad0w.configure`

```text
shad0w.configure(*, llm: str | None = None, api_key=None, api_key_env: str | None = None,
                 base_url: str | None = None, **settings) -> dict
```

Process-wide defaults for every `decision()` and `llm_teacher()` created afterwards. They rank below one call's keywords and above `SHAD0W_*` variables and `shad0w.toml`. Calls add up; `shad0w.configure()` with no arguments clears everything. Returns the current defaults, with the key as a masked `Secret`.

**Raises** `TypeError` for an unknown setting and `ValueError` for an invalid value.

```python
import shad0w

shad0w.configure(llm="openai/gpt-6-luna", api_key_env="OPENAI_API_KEY", alpha=0.02)
shad0w.configure()   # clear
```

## `shad0w.settings`

```text
shad0w.settings(question: str | None = None, path: str | None = None, **overrides) -> Settings
```

The effective settings for `question` (its `[questions.<name>]` section applies), from `overrides` > `configure()` > `SHAD0W_*` > `shad0w.toml` > defaults. `path` is the config file. **Raises** `ValueError` for an invalid value (for example `alpha` outside 0–1, `min_rows` below 100, an unknown `mode`). `shad0w config` prints the same with each value's source.

```python
import shad0w

print(shad0w.settings("intent").alpha)
```

## `shad0w.Settings` (class) {: #shad0wsettings-class }

```text
shad0w.Settings(alpha: float = 0.05, delta: float = 0.1, min_rows: int = 1000, cal_fraction: float = 0.3,
                max_cal: int = 3000, audit_rate: float = 0.01, auto_train: int = 0, retrain: str = 'gated',
                mode: str = 'serve', canary: float = 1.0, never_serve: tuple = (), min_confidence: float | None = None,
                force_threshold: float | None = None, drift_window: int = 500, drift_margin: float = 0.03,
                cost_per_call: float | None = None, llm_latency_ms: float | None = None, folder: str = 'shad0w',
                exposed: bool = False, trace: str | None = None, capture: tuple = ('header', 'model', 'tools', 'decisions'),
                timeout: float = 120.0, on_new_option: str = 'defer', max_mb: float | None = None, rename: tuple = (),
                llm: str | None = None, base_url: str | None = None, api_key_env: str | None = None) -> None
```

A frozen dataclass with one field per [setting](#settings). `asdict()` returns them as a dict; `rename_map` returns `rename` as a dict. Build it with `shad0w.settings(...)` rather than directly, so validation runs.

## Settings

Every setting works as a keyword in code, as `SHAD0W_<NAME>` in the environment, and in `shad0w.toml` (top level or a `[questions.<name>]` section). The first one found wins, in that order. Explanations and examples: [configuration](configuration.html).

| Setting | Default | Environment variable | What it does |
|---|---|---|---|
| `alpha` | `0.05` | `SHAD0W_ALPHA` | Max share of table answers that may differ from your LLM. Lower = safer, fewer calls saved. |
| `delta` | `0.1` | `SHAD0W_DELTA` | Chance the certificate itself is wrong because of an unlucky sample. 0.1 = 90% confidence. |
| `min_rows` | `1000` | `SHAD0W_MIN_ROWS` | Logged LLM answers needed before training. More rows → larger certified share. |
| `cal_fraction` | `0.3` | `SHAD0W_CAL_FRACTION` | Share of logged rows held back to certify, never trained on. |
| `max_cal` | `3000` | `SHAD0W_MAX_CAL` | Upper limit on the calibration slice. |
| `audit_rate` | `0.01` | `SHAD0W_AUDIT_RATE` | Share of decisions re-asked to your LLM to measure live agreement (and to calibrate honestly). |
| `auto_train` | `0` | `SHAD0W_AUTO_TRAIN` | Retrain in the background every N new logged answers. 0 turns it off. |
| `retrain` | `gated` | `SHAD0W_RETRAIN` | 'gated' keeps the old table unless the new one certifies at least 80% of its share; 'always' swaps. |
| `mode` | `serve` | `SHAD0W_MODE` | 'serve' answers from the table; 'shadow' computes but always returns your LLM's answer; 'off' skips the table. |
| `canary` | `1.0` | `SHAD0W_CANARY` | Share of eligible traffic the table may answer (1.0 = all). Use 0.1 for a gradual rollout. |
| `never_serve` | `[]` | `SHAD0W_NEVER_SERVE` | Labels always sent to your LLM, for example fraud or self_harm. |
| `min_confidence` | `none` | `SHAD0W_MIN_CONFIDENCE` | Manual floor on confidence. Can only make serving stricter than the certificate. |
| `force_threshold` | `none` | `SHAD0W_FORCE_THRESHOLD` | Serve above this confidence even below the certified threshold. Those answers are NOT certified. |
| `drift_window` | `500` | `SHAD0W_DRIFT_WINDOW` | Decisions in the drift guard's sliding window. |
| `drift_margin` | `0.03` | `SHAD0W_DRIFT_MARGIN` | Rise in estimated error (absolute) that raises the drift flag. |
| `cost_per_call` | `none` | `SHAD0W_COST_PER_CALL` | Cost of one LLM call, for the dashboard's money-saved figure. |
| `llm_latency_ms` | `none` | `SHAD0W_LLM_LATENCY_MS` | Assumed LLM latency until measured, for the dashboard's time-saved figure. |
| `folder` | `shad0w` | `SHAD0W_FOLDER` | Where logs and bundles live. |
| `exposed` | `false` | `SHAD0W_EXPOSED` | Treat inputs as adversarial: also defer answers a few edits could flip. |
| `trace` | `none` | `SHAD0W_TRACE` | JSON Lines file that receives every decision. |
| `capture` | `header, model, tools, decisions` | `SHAD0W_CAPTURE` | Proxy: which request shapes count as decisions (header, model, tools, decisions, json_schema). |
| `timeout` | `120.0` | `SHAD0W_TIMEOUT` | Proxy: upstream request timeout in seconds. |
| `on_new_option` | `defer` | `SHAD0W_ON_NEW_OPTION` | Options added after training: 'defer' asks your LLM until you retrain; 'serve' keeps serving known ones. |
| `max_mb` | `none` | `SHAD0W_MAX_MB` | Size budget for each table in MB. Many options or a huge vocabulary? Cap it; the certificate is computed on the capped table. |
| `rename` | `{}` | `SHAD0W_RENAME` | Rename labels without retraining, e.g. {lost_card = "card_lost"}. Applied to the table and, at the next train, to the log. |
| `llm` | `none` | `SHAD0W_LLM` | Default LLM for decision(), as "provider/model", e.g. "openai/gpt-6-luna". |
| `base_url` | `none` | `SHAD0W_BASE_URL` | Base URL of an OpenAI-compatible server for that LLM (default: the provider's). |
| `api_key_env` | `none` | `SHAD0W_API_KEY_ENV` | Name of the environment variable that holds the LLM key, e.g. "OPENAI_API_KEY". Never the key itself. |

## `shad0w.Secret`

```text
shad0w.Secret(value=None, *, env: str | None = None, source: str = '')
```

An API key that never prints. Holds a string, a zero-argument function (called on every use) or the name of an environment variable (read on every use). `get()` returns the key now; `hint()` returns `sk-…3f9a`; `describe()` says where it comes from (`set via OPENAI_API_KEY (sk-…3f9a)`); `is_callable` says whether it is a function. `repr` and `str` are masked; pickling keeps only the variable name, never a literal key. **Raises** `TypeError` when `value` is neither a string nor callable.

## `shad0w.llm_teacher` and `LLMTeacher`

```text
shad0w.llm_teacher(options: Any, model: str = 'openai/gpt-6-luna', **kw) -> LLMTeacher

shad0w.LLMTeacher(options: Any, model: str = 'openai/gpt-6-luna', *, question: str | None = None,
                  base_url: str | None = None, api_key: Any = None, api_key_env: str | None = None,
                  client: Any = None, system: str | None = None, temperature: float = 0.0, timeout: float = 30.0,
                  retries: int = 2, headers: dict | None = None, structured: bool | None = None,
                  max_tokens: int = 50, complete: Any = None, settings_key_env: str | None = None)
```

Your LLM as a thread-safe callable: `teacher(text)` returns exactly one of your options (or a `bool` for yes/no). Works with any OpenAI-compatible chat API, the OpenAI Decisions API (`"openai-decisions/<model>"`) and System One servers (`"systemone/<model>"` with `base_url`).

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `options` | as in `decision()` | required | The answers to choose from. |
| `model` | `str` | `'openai/gpt-6-luna'` | `"provider/model"`. The provider picks the URL and key variable; an unknown prefix is kept as the model name and needs `base_url`. |
| `question` | `str` | `None` | The question's name (default `"decision"`). |
| `base_url` | `str` | provider's | An OpenAI-compatible server. For `systemone`, `/v1` is appended when missing. |
| `api_key`, `api_key_env` | | `None` | As in `decision()`. |
| `client` | SDK client | `None` | Reuse an `openai.OpenAI(...)` client instead of the built-in HTTP call. |
| `complete` | callable | `None` | `fn(messages) -> reply text` for anything else (LiteLLM, LangChain, a local model). |
| `system` | `str` | generated | Your own system prompt instead of the generated one. |
| `temperature` | `float` | `0.0` | Sampling temperature. |
| `timeout` | `float` | `30.0` | Seconds per request. |
| `retries` | `int` | `2` | Retries on network errors, rate limits and 5xx. |
| `headers` | `dict` | `None` | Extra HTTP headers. |
| `structured` | `bool` | `None` (= try) | Ask for JSON-schema structured output first, falling back to plain text if the server refuses. |
| `max_tokens` | `int` | `50` | Reply length cap. Dropped automatically for models that refuse it. |
| `settings_key_env` | `str` | `None` | Used internally to pass the `api_key_env` setting. |

**Attributes**: `name` (`"provider/model"`), `options`, `base_url`, `calls`, `failures`, `api_key` (a `Secret`), `key_source` (masked description). **Methods**: `complete(text)` returns the raw reply; `parse(reply)` maps a reply to an option or `None`; `messages(text)` returns the chat messages sent.

**Raises** `ValueError` at construction for an unknown provider without `base_url` or a missing key, and [`TeacherError`](#shad0wteachererror) per call.

## `shad0w.TeacherError`

A `RuntimeError`: the LLM call failed after retries, or its reply matched none of your options. Nothing is logged for that message. When raised through `Shadow.decide`, the message ends with why your LLM was asked.

## `shad0w.Metrics`

```text
shad0w.Metrics(cost_per_call: float | None = None, keep_text: bool = True, recent: int = 50,
               llm_latency_ms: float | None = None)
```

Thread-safe counters shared by `Shadow`, `shad0w serve` and `shad0w proxy`. `cost_per_call` turns saved calls into saved money; `keep_text=False` keeps message text off the dashboard; `recent` is how many recent decisions the dashboard lists; `llm_latency_ms` is assumed until an LLM latency is measured.

| Method | Returns | What it does |
|---|---|---|
| `snapshot(delta: float = 0.1)` | `dict` | Everything the dashboard shows: `decisions`, `table`, `teacher`, `offload`, `llm_calls_saved`, `time_saved_s`, `cost_saved`, `audits`, per-question details under `questions`, and `recent`. |
| `prometheus()` | `str` | Prometheus text format, version 0.0.4 ([observability](observability.html)). |
| `record(...)`, `record_audit(...)`, `set_alpha(...)` | `None` | Used by `Shadow` and the servers; call them only from your own integrations. |

## `shad0w.compile`

```text
shad0w.compile(schema: dict, labeled: dict | None = None, unlabeled: list[str] | None = None,
               encoder: str | None = None, alpha: float = 0.05, delta: float = 0.1, r_min: float = 8.0,
               seed: int = 0) -> Model
```

Build a table from **human labels**: `schema` maps question names to `{"type": "choice", "criteria": {...}}` or `{"type": "yesno"}`, and `labeled[q] = (texts, labels)`. A random 20% (at least 300 rows, at most half) is held out to certify against the labels. Without labels, `unlabeled` texts plus an encoder build a label-free table that comes back `calibrated=False` (flag `uncalibrated`) until you `calibrate` it on real labels. `r_min` is the robustness radius required in `exposed` mode. **Raises** `ValueError` when a question has neither labels nor unlabelled texts. Needs `shad0wllm[compile]`.

## `shad0w.load`

```text
shad0w.load(path: str, native: bool = True) -> Model
```

Load a bundle folder. With `native=True` it uses the bundled C core when present (same decisions, faster). **Raises** `FileNotFoundError` for a missing bundle and `ValueError` for an unsupported format version or a corrupt table.

## `shad0w.Model`

```text
shad0w.Model(questions: dict[str, Question] = <factory>, meta: dict = <factory>) -> None
```

A loaded bundle. No LLM, no logging, no rollout settings: just the tables and their thresholds.

| Method | Returns | What it does |
|---|---|---|
| `decide(state, exposed: bool = False, questions: dict \| None = None, probabilities: bool = True, observe: bool = True)` | `dict` | `{"answers": {question: {...}}}` with `choice` and `probabilities` (choice) or `answer` and `probability` (yes/no), plus `confidence`, `certified`, `flag` and `radius`. `questions={"intent": {"criteria": [...]}}` narrows to some questions and options. `observe=False` leaves the drift guard alone. |
| `calibrate(question: str, texts, labels, alpha: float \| None = None, delta: float = 0.1)` | `dict` | Re-certify one question against human labels; returns `n`, `accuracy`, `threshold`, `certified_share`. Call `save()` after. |
| `save(path: str)` | `None` | Write `manifest.json` and one `.s0` file per question. |
| `use_native(bundle_dir: str)` | `Model` | Switch to the C core for that bundle. |
| `native` (property) | `bool` | `True` when every question runs on the C core. |

## `shad0w.shadow_compile`

```text
shad0w.shadow_compile(schema: dict, records: list[dict], alpha: float = 0.05, delta: float = 0.1,
                      cal_fraction: float = 0.3, max_cal: int = 3000, teacher: str = 'unspecified', seed: int = 0,
                      r_min: float = 8.0, cal_records: list[dict] | None = None, drift_window: int = 500,
                      drift_margin: float = 0.03, max_mb: float | None = None)
```

What `Shadow.train()` calls: build and certify every question of `schema` from your LLM's answers. `records` are log rows (`{"text": ..., "<question>": answer}`). Without `cal_records`, a random `cal_fraction` (at least 100 rows, at most `max_cal`) is held out to certify; with them, all `records` train and `cal_records` certify. `teacher` is the LLM's name, recorded in the certificate. **Returns** `(Model, certificate dict)`; save them with `model.save(path)` and write the certificate as `certificate.json`. **Raises** `ValueError` for fewer than 100 rows per question or answers outside the options.

## `shad0w.certify_bundle`

```text
shad0w.certify_bundle(bundle: str, records: list[dict], alpha: float | None = None, delta: float = 0.1,
                      teacher: str | None = None)
```

Re-certify an existing bundle on **fresh** LLM answers that were not used to build it (for example last week's spot checks). Rewrites the thresholds in `manifest.json` and `certificate.json` (adding `recertified_utc`) and returns the new certificate. `alpha` defaults to the bundle's. CLI: `shad0w certify`.

## `shad0w.read_certificate`

```text
shad0w.read_certificate(bundle: str)
```

The bundle's `certificate.json` as a dict, or `None` when there is none.

---

## Files

### Bundle folder

```text
shad0w/intent/
├── log.jsonl            your LLM's answers (training data)
└── bundle/
    ├── manifest.json    questions, options, thresholds, drift guard
    ├── intent.s0        the table for question "intent" (binary)
    ├── certificate.json what was certified, on what, with which result
    └── schema.json      the options train() used (written by Shadow.train)
```

A bundle runs anywhere the files can be read: Python (`shad0w.load`), JavaScript (`Bundle.load`), C, or `shad0w serve`.

### `log.jsonl`

One JSON object per line, appended for every answer your LLM gives:

```json
{"text": "my card was stolen", "intent": "lost_card", "source": "teacher", "ts": 1760000000.123}
```

| Field | Meaning |
|---|---|
| `text` | The message. |
| `<question>` | Your LLM's answer (option name, or `true`/`false`). |
| `source` | `"teacher"` (a deferred decision), `"audit"` (a spot check: a uniform random sample used to certify at the next `train()`), or `"import"` (`warm_start` / `shad0w import`). |
| `ts` | Unix time in seconds. |

`shad0w train --log` accepts any JSON Lines file with a `text` field and one field per question.

### Trace file

With `trace=`, every decision (table and LLM) is appended as `{"ts", "question", "text", "answer", "source", "confidence", "flag", "latency_us"}`.

### `manifest.json`

```json
{"format": 1,
 "meta": {"mode": "shadow", "teacher": "openai/gpt-6-luna", "alpha": 0.05, "delta": 0.1},
 "questions": {"intent": {"type": "choice", "options": ["refund", "lost_card", "other"], "threshold": 0.886,
                          "alpha": 0.05, "r_min": 8.0, "calibrated": true,
                          "guard": {"t": 0.623, "base": 0.186, "window": 500, "margin": 0.03}}}}
```

| Field | Meaning |
|---|---|
| `format` | Bundle format version (1). Other versions are refused. |
| `meta` | How it was built: `mode`, `teacher`, `alpha`, `delta`. |
| `questions.<q>.type` | `"choice"` or `"yesno"` (options are then `["no", "yes"]`). |
| `questions.<q>.options` | Option names, in table column order. |
| `questions.<q>.threshold` | The certified confidence threshold; `null` = certifies nothing. |
| `questions.<q>.alpha` | The α it was certified at. |
| `questions.<q>.r_min` | Robustness radius required when `exposed=True`. |
| `questions.<q>.calibrated` | `false` for label-free tables (every answer is flagged `uncalibrated`). |
| `questions.<q>.guard` | Drift guard state: confidence level `t`, calibration share below it `base`, `window`, `margin`. |

### `certificate.json`

What was certified, on what data, with what result. `shad0w report --bundle B` prints it.

| Field | Meaning |
|---|---|
| `format` | `"shad0w-certificate/1"`. |
| `mode`, `certified_against` | `"shadow"` and `"teacher"`: the bound is against your LLM. |
| `teacher` | The LLM's name, as recorded at training. |
| `alpha`, `delta` | The bound and the failure probability. |
| `statement` | The guarantee in one paragraph. |
| `created_utc`, `recertified_utc` | When it was trained, and re-certified with `certify_bundle`. |
| `questions.<q>.type`, `options` | As in the manifest. |
| `questions.<q>.n_records`, `n_fit` | Log rows used, and how many of them trained the table. |
| `questions.<q>.data_sha256` | Hash of the training texts and answers, to tie the certificate to its data. |
| `questions.<q>.calibration` | `"held-out-split"` (a random slice of the log) or `"uniform-audit"` (spot-check rows). |
| `questions.<q>.n_calibration` | Calibration rows. |
| `questions.<q>.agreement_with_teacher` | Agreement with your LLM on *all* calibration rows, served or not. |
| `questions.<q>.threshold` | The certified threshold (`null` = nothing certified). |
| `questions.<q>.certified_share_on_calibration` | Share of calibration rows at or above the threshold: an estimate of how much the table will answer. |
| `questions.<q>.disagreement_on_certified_calibration` | Observed disagreement on those rows (always below α; the bound is the conservative upper limit). |
| `questions.<q>.procedure` | For example `learn-then-test/clopper-pearson/auto (fixed-sequence)`: which test set the threshold ([the math](guarantee.html#formally)). |

### `.s0` table

Little-endian binary: `b"S0RX"`, then `uint32 version` (1), `uint32 F` (patterns), `uint32 K` (options), `float32 temperature`; then `uint32 keys[F]` (sorted pattern hashes), `int8 weights[F×K]`, `float32 scale[K]`, `float32 bias[K]`, `float32 G[K×K]` (used for the robustness radius). Total size `20 + 4F + F·K + 8K + 4K²` bytes. Loaders refuse other versions, `K` outside 2–1,024, `F` above 2²², unsorted keys, and truncated or padded files.

### `schema.json`

`{"<question>": {"type": "choice", "criteria": {"refund": "wants money back", ...}, "instructions": "..."}}` (or `{"type": "yesno"}`). `decision()` reads it when you omit `options`.

---

## JavaScript

`npm install shad0wllm` gives the same decisions in Node 18+, browsers, Cloudflare Workers, Deno and Bun. Full types ship in `index.d.ts`; guide: [JavaScript & TypeScript](javascript.html).

| Export | Kind | What it does |
|---|---|---|
| `decision(name, opts)` / `decision(opts)` | function → `Promise<Shadow>` | Like Python's `decision()`: `options`, `llm` (`"provider/model"` or a function), `apiKey` (string or function), `apiKeyEnv`, `baseURL`, `bundle` (path, URL or `Bundle`), `log`, `fallback`, `rename`, `onNewOption`, `auditRate`, `onDecision`, plus teacher options. |
| `Shadow` | class | `decide(text)`, `decideMany(texts, {concurrency})`, `peek(text)`, `record(text, answer)`, `explain(text)`, `options()`, `threshold()`, `stats()`, `optionsAdded`, `optionsRemoved`, `toString()`. Every `Decision` has `answer`, `source`, `confidence`, `certified`, `flag`, `latencyUs`, `question`, `threshold`, `why`. |
| `Bundle` | class | `Bundle.load(dirOrUrl)` then `bundle.decide(text, {questions, probabilities})` returns `{answers: {q: {choice \| answer, confidence, certified, flag, probabilities \| probability}}}`. `bundle.manifest` is the manifest. |
| `Table` | class | One `.s0` table: `new Table(buffer, labels)`, `decide(text, probabilities = true)`, `F`, `K`. |
| `configure({llm, apiKey, apiKeyEnv, baseURL})` | function | Process-wide defaults; no argument clears them. |
| `openaiTeacher(opts)`, `decisionsTeacher(opts)`, `systemoneTeacher(opts)` | functions | Your LLM as a teacher: chat completions, the OpenAI Decisions API, or a System One server. |
| `shad0wMiddleware(shadow, opts)` | function | Vercel AI SDK language-model middleware. |
| `decisionModel(shadows, {fallback})` | function | An AI SDK decision model (spec v4) backed by your tables. |
| `explainFlag(flag)`, `FLAG_WORDS` | function, object | Flags in plain words. |
| `maskKey(key)` | function | `sk-…3f9a`. |
| `matchOption(reply, options, field?)` | function | Map an LLM reply to one option or `null`. |
| `items(text)`, `words(text)`, `TABLE_VERSION` | function, constant | The feature hashing shared with Python and C. |
| `PROVIDERS` | object | Provider → `[base URL, key variable]`. |

---

## HTTP routes

`shad0w serve --bundle B` and `shad0w proxy --upstream URL` speak HTTP on `127.0.0.1:8010` by default. Details and examples: [HTTP server](http.html) and [the proxy](proxy.html).

| Route | `serve` | `proxy` | What it does |
|---|---|---|---|
| `POST /v1/decide` | yes | | `{"state": text, "exposed"?, "questions"?}` → `{"answers": {...}, "latency_us"}` (the `Model.decide` shape). |
| `POST /v1/decisions` | yes | yes | OpenAI Decisions API shape. The proxy answers certified questions locally and forwards the rest. |
| `POST /v1/systemone` | yes | yes | System One shape (Jev, Kev, Laya, llama.cpp). |
| `POST /v1/chat/completions` | | yes | Answered from the table when marked as a decision (header, `model="shad0w/<question>"`, tools) and certified; otherwise forwarded and the answer logged. |
| `POST /v1/playground` | yes | yes | `{"text", "question"?}` → what the table thinks (`explain`). |
| `GET /v1/health` | yes | yes | `{"ok": true, "questions": [...]}`. |
| `GET /v1/stats` | yes | yes | `Metrics.snapshot()` as JSON (the proxy adds upstream, training state and settings). |
| `GET /metrics` | yes | yes | Prometheus text format. |
| `GET /`, `GET /dashboard` | yes | yes | The live dashboard. |
| anything else | 404 | forwarded | The proxy passes every other request to the upstream unchanged. |

Neither has authentication: keep them on localhost or behind your own gateway.

## Command line

| Command | What it does |
|---|---|
| `shad0w init` | A commented `shad0w.toml` and a runnable `app.py` using `decision()`. |
| `shad0w status` | One line per decision folder: rows vs `min_rows`, certified share, live agreement, next step. |
| `shad0w import` | Exported request/response JSON Lines into a decision's log. |
| `shad0w train` | Train and certify from a log (`--log L --out B`, or `--dir` for the folder layout). |
| `shad0w proxy` | OpenAI-compatible gateway in front of your LLM. |
| `shad0w serve` | HTTP decisions, dashboard and metrics for a bundle. |
| `shad0w try` | Type messages and see the answer, confidence vs threshold, the reason and the time. |
| `shad0w stats`, `shad0w watch` | Summarise a log; live counters from a running server or proxy. |
| `shad0w config` | Effective settings and their source; `--init` writes a starter `shad0w.toml`. |
| `shad0w doctor` | Check Python, the C core, training dependencies, Node, config, keys, upstream and bundles. |
| `shad0w report` | Print a bundle's certificate. |
| `shad0w certify` | Re-certify a bundle on fresh LLM answers (`certify_bundle`). |
| `shad0w calibrate` | Re-certify against about 300 human labels per question (`Model.calibrate`). |
| `shad0w shadow`, `shad0w compile`, `shad0w decide`, `shad0w bench` | Lower-level: build from a schema and data, decide one text, time a bundle. |

Every command takes `--help`. Flags and examples: [command line](cli.html).

## Limits

| | |
|---|---|
| Options per question | 2 to 1,024 |
| Patterns per table | up to 4,194,304 (2²²) |
| Size | about (options + 4) bytes per pattern, plus 4 × options² bytes; cap it with `max_mb` ([scale](scale.html)) |
| Text | any UTF-8; words and character patterns are hashed, so any language works |
