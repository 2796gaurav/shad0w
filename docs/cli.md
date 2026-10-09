---
title: Command line
description: Every shad0w command and every flag, with defaults, examples and exit codes.
---
# Command line

<p class="lead">Everything shad0w does from Python it also does from a terminal: start a project, import old answers, train, try, watch and serve. This page lists every command and every flag.</p>

**TL;DR**

```bash
pip install "shad0wllm[compile]"
shad0w init                                     # shad0w.toml + a runnable app.py
python app.py "my card was stolen"              # your LLM answers; the answer is logged
shad0w status                                   # rows logged, certified share, next step
shad0w train --dir shad0w --question intent     # once enough answers are logged
shad0w try --bundle shad0w/intent/bundle "my card was stolen"
```

`shad0w --help` lists the commands; `shad0w <command> --help` lists a command's flags; `shad0w --version` prints the version.

Words used on this page. **The table** (or **bundle**, the folder that holds it): the small model shad0w trains from your LLM's answers. **Certified**: tested on answers it never trained on and allowed to serve only where it differs from your LLM on at most **α**. **Log**: the JSON Lines file of your LLM's answers that training reads.

## Commands at a glance

| Command | What it does |
|---|---|
| [`init`](#init) | Write a commented `shad0w.toml` and a runnable `app.py` |
| [`status`](#status) | One line per decision: rows logged, certified share, live agreement, next step |
| [`import`](#import) | Turn exported LLM request/response pairs into a decision's log |
| [`doctor`](#doctor) | Check Python, the C core, training deps, config, your API key, an upstream, a bundle and a log |
| [`config`](#config) | Print every effective setting and where it came from, or write a starter `shad0w.toml` |
| [`proxy`](#proxy) | OpenAI-compatible gateway: answers marked decisions from the table, forwards the rest |
| [`train`](#train) | Train and certify a table from logged answers |
| [`try`](#try) | See what the table answers for a message, how sure it is, how fast, and why |
| [`stats`](#stats) | Summarise a log, or print a snapshot of a running proxy or server |
| [`watch`](#watch) | Live counters from a running proxy or server |
| [`serve`](#serve) | HTTP decisions from a bundle, with dashboard and metrics |
| [`decide`](#decide) | One raw decision as JSON |
| [`bench`](#bench) | Latency percentiles of a bundle |
| [`report`](#report) | Print a bundle's certificate |
| [`certify`](#certify) | Re-certify a bundle on fresh answers it never trained on |
| [`calibrate`](#calibrate) | Certify a bundle against human labels |
| [`compile`](#compile) | Build a table from human labels or from unlabelled text |
| [`shadow`](#shadow) | Train with an explicit schema (the older form of `train`) |

Settings that a command does not take as a flag come from `SHAD0W_*` variables, then `shad0w.toml`, then the defaults ([Configuration](configuration.html)). A flag always wins.

## Start

### init

Write a commented `shad0w.toml` (with `llm` and `api_key_env`) and a runnable `app.py` that uses `decision()`. Existing files are kept unless you pass `--force`.

| Flag | Default | What it does |
|---|---|---|
| `--dir` | `.` | Folder to write into (created if missing) |
| `--name` | `intent` | The decision's name: a Python name, used for the variable, the folder and the log field |
| `--llm` | `openai/gpt-6-luna` | `provider/model`; the provider picks `api_key_env` (none for local servers such as `ollama/...`) |
| `--options` | a card-support example (`refund`, `lost_card`, `balance`, `other`) | Comma-separated option names, at least two |
| `--force` | off | Overwrite existing files |
| `--files` | off | Write the older starter instead: `schema.json` and `teacher_log.jsonl` for `shad0w shadow` |

```bash
shad0w init --dir support --name ticket --options billing,tech,other
```

```text
wrote support/shad0w.toml, support/app.py
next:
  cd support
  export OPENAI_API_KEY=sk-...      # your key; shad0w.toml names this variable
  python app.py "my card was stolen"
  shad0w status
```

### status

One line per decision folder: rows logged against `min_rows`, the certified share at α, live agreement on spot checks since the last training, and the next step.

| Flag | Default | What it does |
|---|---|---|
| `--folder` | the `folder` setting (`shad0w`) | The decisions folder to scan |
| `--config` | `./shad0w.toml` | Config file to read |

```text
intent    1,500/1,000 rows  certified 61.2% at alpha=0.05 (threshold 0.886)  live agreement 97.0% (n=33)  -> serving
ticket      240/1,000 rows  no table yet  -> log 760 more answers
```

### import

Turn exported request/response pairs into a decision's log, so training can start from answers you already paid for. Each line of the file is `{"request": ..., "response": ...}`; OpenAI Batch output works as is. Labels that match none of the options are skipped and counted. Rows are written with `"source": "import"`.

| Flag | Default | What it does |
|---|---|---|
| `--from` | required | `openai-chat` (chat completions) or `openai-decisions` (`/v1/decisions`) |
| `--file` | required | The JSON Lines export |
| `--question` | required | The decision's name: its folder and log field |
| `--options` | the request's enum, or the answers seen | Comma-separated options |
| `--dir` | `shad0w` | Decisions folder |
| `--config` | `./shad0w.toml` | Config file to read |

```bash
shad0w import --from openai-chat --file export.jsonl --question intent
```

```text
export.jsonl: 1,500 answers imported into shad0w/intent/log.jsonl (0 exchanges without a usable answer, 0 with an unknown option)
next: shad0w train --dir shad0w --question intent
```

## Check and configure

### doctor

Checks what most often goes wrong and prints one line each: `[ok]`, `[--]` (a warning) or `[!!]` (a problem). Exits 1 when any `[!!]` line is printed, so it works in CI. What each line means: [Troubleshooting](troubleshooting.html#what-each-doctor-line-means).

| Flag | Default | What it does |
|---|---|---|
| `--llm` | the `llm` setting | `provider/model` whose API key should be set; prints where the key comes from, masked |
| `--upstream` | — | An OpenAI-compatible base URL to probe (`<url>/models`) |
| `--bundle` | — | A bundle to load; reports each question's options, threshold and α |
| `--log` | — | A log to count against `min_rows` |
| `--config` | `./shad0w.toml` | Config file to read and validate |

```bash
shad0w doctor --llm openai/gpt-6-luna --bundle shad0w/intent/bundle --log shad0w/intent/log.jsonl
```

```text
[ok] Python 3.12.14                       shad0w 0.3.2
[ok] C core                               built in
[ok] training deps (scipy, scikit-learn)  installed
[--] torch (optional, faster training)    not installed: scipy solver is used
[ok] TOML config reader                   tomllib
[ok] Node (JavaScript package)            v24.20.0
[ok] config                               defaults; mode=serve alpha=0.05 audit_rate=0.01 canary=1.0
[ok] openai API key                       key: set via OPENAI_API_KEY (sk-…3f9a)
[ok] bundle question 'intent'             4 options, threshold 0.886, alpha 0.05
[ok] certificate.json                     present
[ok] log                                  1,500 rows (ready to train)

all good
```

### config

Without flags, prints every setting, its effective value and where it came from (`default`, `env:SHAD0W_…`, `toml:…`, `toml:… [questions.X]`, `configure()`), then the config file in use and where the LLM key comes from (masked).

| Flag | Default | What it does |
|---|---|---|
| `--question` | — | Also apply that question's `[questions.<name>]` section |
| `--config` | `./shad0w.toml` or `$SHAD0W_CONFIG` | Config file to read (with `--init`: the file to write) |
| `--init` | off | Write a commented `shad0w.toml` listing every setting with its default and meaning; refuses to overwrite |
| `--explain` | on | The default action; accepted for clarity |

```bash
shad0w config --question intent
shad0w config --init
```

## Train and certify

### train

Train and certify a table from logged answers. Questions and options are detected from the log unless you pass `--schema` (or a previous bundle's `schema.json` exists). When at least 100 spot-check rows exist, they are used to certify ([why](training.html#why-retraining-certifies-on-spot-checks)).

| Flag | Default | What it does |
|---|---|---|
| `--dir` | — | The `decision()` / proxy layout: reads `<dir>/<question>/log.jsonl`, writes `<dir>/<question>/bundle` |
| `--question` | the only one in `--dir` | Which question to train |
| `--log` | — | A JSON Lines log (instead of `--dir`) |
| `--out` | — | Bundle folder to write (with `--log`) |
| `--schema` | options seen in the log | A `schema.json` naming each question's options |
| `--alpha` | the `alpha` setting (`0.05`) | Most disagreement allowed among table answers |
| `--delta` | the `delta` setting (`0.1`) | Chance the certificate itself is wrong |
| `--min-rows` | the `min_rows` setting (`1000`) | Refuse smaller logs (exit 1); 100 is the hard minimum |
| `--max-mb` | the `max_mb` setting (none) | Size budget per table, in MB |
| `--rename OLD=NEW` | — | Rename a label in the log while training; repeatable. Without `--schema`, the previous bundle's `schema.json` is reused with the names renamed |
| `--gate` | off | Keep the existing bundle unless the new one certifies at least 80% of its share (exit 3 when kept) |
| `--teacher` | `unspecified` | A name for your LLM, recorded in the certificate |
| `--config` | `./shad0w.toml` | Config file to read (including `[questions.<name>]`) |

```bash
shad0w train --dir shad0w --question intent
shad0w train --log log.jsonl --out bundle --schema schema.json --alpha 0.02 --max-mb 1
shad0w train --log log.jsonl --out bundle --rename lost_card=card_lost   # the old schema is renamed for you
```

### report

Print a bundle's certificate as JSON: α, δ, the teacher, and per question the options, the number of rows, the threshold, the certified share and the calibration set used.

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle folder |

### certify

Keep the table, and refresh its threshold and certificate on fresh answers it never trained on (for example last week's log).

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle to re-certify (rewritten in place) |
| `--data` | required | JSON Lines of fresh LLM answers (`text` plus one field per question) |
| `--alpha` | the bundle's α | Most disagreement allowed |
| `--delta` | `0.1` | Chance the certificate itself is wrong |
| `--teacher` | the bundle's | A name for your LLM |
| `--force` | off | Certify even with fewer than 100 rows (it then usually stops serving) |

### calibrate

Certify against human labels (the truth) instead of your LLM. About 300 labels per question suffice.

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle (rewritten in place) |
| `--data` | required | JSON Lines of labelled examples, held out from training |
| `--alpha` | the table's α | Most error allowed |
| `--force` | off | Calibrate even with fewer than 100 labels per question |

### compile

Build a table without an LLM log.

| Flag | Default | What it does |
|---|---|---|
| `--schema` | required | `schema.json` naming the options |
| `--data` | — | JSON Lines of human labels |
| `--unlabeled` | — | A text file, one message per line: label-free mode, uncertified until calibrated (needs `shad0wllm[logs]`) |
| `--encoder` | `bge-small` | Sentence encoder for label-free mode |
| `--alpha` | `0.05` | α recorded for later calibration |
| `--out` | required | Bundle folder to write |

### shadow

Train with an explicit schema and data file: what `train` does under the hood, kept for older scripts.

| Flag | Default | What it does |
|---|---|---|
| `--schema` | required | `schema.json` |
| `--data` | required | JSON Lines: `text` plus one field per question holding your LLM's answer |
| `--teacher` | `unspecified` | A name for your LLM |
| `--alpha` | `0.05` | Most disagreement allowed (this command does not read `shad0w.toml`) |
| `--delta` | `0.1` | Chance the certificate itself is wrong |
| `--out` | required | Bundle folder to write |

## Use a table

### try

Shows, for each message: whether the table would answer, the answer, its confidence against the certified threshold, the time in µs, the reason, and the top three options. Without messages it reads them interactively (an empty line quits).

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle folder |
| `--question` | the bundle's only question | Which question to show |
| `text …` | — | One or more messages |

```text
my card was stolen
  ✓ table  lost_card  (confidence 0.962, needs 0.911)  7.7 µs
     certified: answered by the table
     top: lost_card 0.96, card_swallowed 0.03, other 0.01
```

### decide

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle folder |
| `text` | required | One message |
| `--exposed` | off | Treat the input as adversarial (also defer answers a few edits could flip) |

Prints the raw decision for every question as JSON: choice, confidence, `certified`, `flag`, probabilities.

### bench

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle folder |
| `--texts` | required | A text file, one message per line |

Prints `{"n", "p50_us", "p99_us", "questions"}` after a short warm-up.

### serve

HTTP decisions from a bundle, without an LLM: `POST /v1/decide`, `/v1/decisions`, `/v1/systemone`, `/v1/playground`, plus the dashboard at `/`, `/v1/stats`, `/metrics` and `/v1/health`. See [HTTP](http.html). Off this machine, set an access token (`$SHAD0W_PROXY_TOKEN`, `--token-env`, `--token-file`); see [Access token](proxy.html#access).

| Flag | Default | What it does |
|---|---|---|
| `--bundle` | required | The bundle folder |
| `--host` | `127.0.0.1` | Address to listen on |
| `--port` | `8010` | Port |
| `--token-env NAME` / `--token-file PATH` | `$SHAD0W_PROXY_TOKEN` | Access token every route but `/v1/health` needs |
| `--cost-per-call` | — | Cost of one LLM call, for "money saved" |
| `--llm-latency-ms` | — | Assumed LLM latency, for "time saved" |

### proxy

An OpenAI-compatible gateway in front of your LLM. Requests marked as decisions are answered from the table when it is certified; everything else is forwarded unchanged. Full guide: [The proxy](proxy.html).

| Flag | Default | What it does |
|---|---|---|
| `--upstream` | `https://api.openai.com/v1` | The real API base URL (OpenAI, Groq, Ollama, vLLM, …) |
| `--dir` | the `folder` setting (`shad0w`) | Where logs and bundles live: `<dir>/<question>/` |
| `--model` | — | Upstream model for requests that use `model="shad0w/<question>"` |
| `--api-key-env NAME` | forward each client's own key | Send the key in this environment variable upstream |
| `--api-key-file PATH` | forward each client's own key | Send the key in this file upstream, re-read on every request (Docker / Kubernetes secrets). Not with `--api-key-env` |
| `--host` | `127.0.0.1` | Address to listen on |
| `--port` | `8010` | Port |
| `--token-env NAME` / `--token-file PATH` | `$SHAD0W_PROXY_TOKEN` | Access token every route but `/v1/health` needs; see [Access token](proxy.html#access) |
| `--insecure-open` | off | Allow a non-local `--host` with the proxy's own key and no token |
| `--audit-rate` | the `audit_rate` setting (`0.01`) | Share of decisions spot-checked against the LLM |
| `--auto-train` | the `auto_train` setting (`0`, off) | Retrain a question every N new LLM answers |
| `--alpha` | the `alpha` setting (`0.05`) | Most disagreement allowed |
| `--delta` | the `delta` setting (`0.1`) | Chance the certificate itself is wrong |
| `--min-rows` | the `min_rows` setting (`1000`) | Answers needed before auto-train runs |
| `--mode` | the `mode` setting (`serve`) | `serve`, `shadow` or `off` |
| `--canary` | the `canary` setting (`1.0`) | Share of certified answers the table may serve |
| `--never-serve` | — | Comma-separated labels that always go to the LLM |
| `--min-confidence` | — | Confidence floor on top of the certificate |
| `--capture` | the `capture` setting (`header,model,tools,decisions`) | How requests are recognised as decisions; add `json_schema`; repeatable |
| `--timeout` | the `timeout` setting (`120`) | Upstream timeout in seconds |
| `--trace` | — | JSON Lines file receiving every decision |
| `--bundle Q=PATH` | `<dir>/<Q>/bundle` | Use the bundle at PATH for question Q; repeatable |
| `--schema` | — | A `schema.json` naming each question's options |
| `--cost-per-call` | — | Cost of one LLM call, for "money saved" |
| `--llm-latency-ms` | — | Assumed LLM latency until measured, for "time saved" |
| `--no-text` | off | Keep request text out of the dashboard |
| `--config` | `./shad0w.toml` or `$SHAD0W_CONFIG` | Config file to read |

```bash
shad0w proxy --upstream https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --mode shadow
shad0w proxy --api-key-file /run/secrets/openai --token-file /run/secrets/shad0w-token --host 0.0.0.0 --canary 0.1
```

On start it prints where the key comes from, masked: `upstream key: set via OPENAI_API_KEY (sk-…3f9a)`.

## Watch

### stats

| Flag | Default | What it does |
|---|---|---|
| `--log` | — | Summarise a log: answers per source and per option, and whether it is ready to train |
| `--top` | `15` | Options shown per question |
| `--trace` | — | Also summarise a trace file: share answered by the table |
| `--url` | — | Instead of `--log`: print one JSON snapshot from a running proxy or server (`/v1/stats`) |

One of `--log` or `--url` is required.

### watch

Live counters in the terminal: decisions, share answered by the table, LLM calls, time and money saved, per-question latency and the live safety bound, and the latest decisions. Ctrl-C quits.

| Flag | Default | What it does |
|---|---|---|
| `--url` | `http://127.0.0.1:8010` | The running `shad0w proxy` or `shad0w serve` |
| `--every` | `1.0` | Refresh interval in seconds |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | `train`: not enough logged answers yet. `doctor`: at least one `[!!]` line |
| `2` | A one-line `shad0w: …` error on stderr (missing bundle or log, missing training dependencies, invalid setting, bad flag); no traceback |
| `3` | `train --gate` kept the old table |

## Files

```text
shad0w.toml                              settings (optional)
shad0w/intent/log.jsonl                  {"text": "...", "intent": "lost_card", "source": "teacher", "ts": ...}  one per LLM answer
shad0w/intent/bundle/manifest.json       options, certified threshold, alpha, drift-guard state
shad0w/intent/bundle/intent.s0           the table (int8)
shad0w/intent/bundle/certificate.json    what was certified, on how much data, when
shad0w/intent/bundle/schema.json         the options (and descriptions) it was trained with
```

A typical table is about 1.5 MB on disk.
