---
title: Troubleshooting
description: The real error messages shad0w prints, what each one means and how to fix it, plus what to do when the table answers nothing, drift fires or the proxy forwards everything.
---
# Troubleshooting

<p class="lead">Find the message you see on this page, and the fix next to it. Every message here is copied from the code. If the table simply answers less than you hoped, jump to <a href="#the-table-answers-nothing-or-very-little">The table answers nothing</a>.</p>

**TL;DR**: start with `shad0w doctor`. It checks what most often goes wrong and prints one line each.

```bash
shad0w doctor --llm openai/gpt-6-luna --bundle shad0w/intent/bundle --log shad0w/intent/log.jsonl
```

`[ok]` is fine, `[--]` is a warning, `[!!]` is a problem. It exits 1 when any `[!!]` line is printed, so it also works in CI. `SHAD0W_LOG=debug` prints one line per decision if you need to see more.

Words used on this page. **The table**: the small model shad0w trains from your LLM's answers. **Certified**: tested on answers it never trained on and allowed to serve only where it differs from your LLM on at most α. **Defer**: send to your LLM. **Drift**: live traffic no longer looks like the traffic the table was tested on.

## What each doctor line means

| Line | When it is not `[ok]` | Fix |
|---|---|---|
| **Python** | `[!!]` below 3.10 | shad0w needs Python 3.10 or newer. |
| **C core** | `[--] numpy fallback (same answers, slower)` | Answers are identical. For speed, install from a wheel (`pip install shad0wllm`) or build with a C compiler (`xcode-select --install` on macOS, `build-essential` on Linux). |
| **training deps (scipy, scikit-learn)** | `[--] missing ['scipy', 'sklearn']: pip install "shad0wllm[compile]"` | Serving needs only numpy; training needs these. |
| **torch (optional, faster training)** | `[--] not installed: scipy solver is used` | Optional. `pip install "shad0wllm[torch]"` trains large logs faster. |
| **TOML config reader** | `[--] pip install tomli (needed to read shad0w.toml on 3.10)` | Only on Python 3.10, and only if you use `shad0w.toml`. |
| **Node (JavaScript package)** | `[--] not found; only needed for the npm package and its tests` | Ignore unless you use the npm package. |
| **config** | `[!!]` and a message, e.g. `mode must be one of ('serve', 'shadow', 'off'), got 'on'` | `shad0w.toml` or a `SHAD0W_*` variable has an invalid value; the message names it. `shad0w config` shows each value and where it came from. |
| **`<provider>` API key** | `[!!] key: not set (OPENAI_API_KEY is empty); set it, or pass api_key= / api_key_env= in code` | Export the variable it names, or set `api_key_env` to the variable that holds your key. `[ok] key: none needed (local server)` is normal for `ollama`, `vllm`, `lmstudio` and `systemone`. |
| **provider `'name'`** | `[--] unknown preset; pass base_url= in code` | Not a built-in provider: pass `base_url=` for any OpenAI-compatible server. |
| **upstream reachable** | `[!!] <url>: <error>` | `--upstream` cannot be reached. Check that the URL ends in `/v1` and the server runs. An HTTP error such as 401 still counts as reachable. |
| **bundle question** | `[--] certifies nothing (threshold inf)` | The table never stayed within α on held-out answers, so it serves nothing. See [below](#the-table-answers-nothing-or-very-little). |
| **certificate.json** | `[--] missing: run shad0w train / shadow / certify` | The bundle was copied without its certificate. Train or certify again. |
| **bundle** | `[!!] <path>: <error>` | The folder is not a readable bundle. Point `--bundle` at the folder holding `manifest.json`. |
| **log** | `[--] 640 rows (need 1,000)` | Not enough logged answers yet. Keep running, or lower `min_rows` for an experiment (100 is the minimum). |

## Messages from the command line

The CLI prints one line starting with `shad0w:` instead of a traceback, and exits with code 2 (1 and 3 where noted).

| Message | Meaning and fix |
|---|---|
| `` shad0w: no bundle at X (expected manifest.json; run `shad0w train` first) `` | Wrong path, or nothing trained yet. |
| `shad0w: nothing logged yet at X` | `shad0w stats` found no log. With `decision()` the log is `shad0w/<question>/log.jsonl`, relative to where your program runs. |
| `shad0w: sklearn is not installed; training needs: pip install "shad0wllm[compile]"` | Install the training extras. |
| `not enough yet: need 1,000 (--min-rows to override; 100 is the hard minimum)` | `train` exits 1. Log more, or pass `--min-rows` to experiment. |
| `intent: new bundle certifies 40.0% of calibration traffic vs 62.0% before; kept the old one (retrain='always' or no --gate to replace anyway)` | `train --gate` found the new table worse (under 80% of the current certified share) and kept the old one. Exit 3. |
| `--dir shad0w holds questions ['intent', 'urgent']; pick one with --question` | Add `--question`. |
| `give --log and --out (or --dir [--question])` | `train` needs to know where to read and write. |
| `shad0w: config file X not found` | `SHAD0W_CONFIG` or `--config` points at a missing file. |
| `shad0w: mode must be one of ('serve', 'shadow', 'off'), got 'on'` (and similar) | An invalid setting; the message names it and the allowed values. |
| `` shad0w: shad0w.toml: found `api_key = ...`. Never put the key itself in a file: ... `` | Remove the key from the file. Put the variable's **name** in `api_key_env` instead. |
| `shad0w: --api-key-file X: no such file` / `... is empty` | `proxy --api-key-file` must point at a file holding the key. |
| `shad0w: error: NAME is not set` | `proxy --api-key-env NAME`: that variable is empty in the proxy's environment. |
| `pass --api-key-env or --api-key-file, not both` | Pick one. |
| `refusing to listen on 0.0.0.0 with the proxy's own LLM key and no access token` | Set `SHAD0W_PROXY_TOKEN` (or `--token-env` / `--token-file`) and send it from your apps; see [Access token](proxy.html#access). `--insecure-open` skips the check on a network you trust. |
| `401 invalid_access_token` | The proxy has a token: send `X-Shad0w-Token: <token>`, or use the token as the OpenAI SDK's `api_key` when the proxy holds the upstream key. `/v1/health` never needs it. |
| `shad0w.toml exists; not overwriting` | `config --init` never overwrites; delete the file or use `--config other.toml`. |

## Errors in Python

| Error | Meaning and fix |
|---|---|
| `TypeError: decision() got unexpected keyword argument(s): 'alpah' (did you mean 'alpha'?)` | A misspelt setting. |
| `ValueError: openai/gpt-6-luna: no API key (OPENAI_API_KEY is not set). Pass api_key="sk-..." ...` | Export the key, pass `api_key=` / `api_key_env=`, or call `shad0w.configure(api_key_env=...)` once. See [API keys](connect-your-llm.html#api-keys). |
| `ValueError: model 'x/y': unknown provider; pass base_url=... (known: openai, anthropic, ...)` | Use a built-in provider name, or pass `base_url=` for any OpenAI-compatible server. |
| `ValueError: pass options=... so the LLM knows what to choose from` | A `"provider/model"` LLM needs `options=` (a function LLM does not). |
| `TypeError: ['api_key'] only apply when llm is a "provider/model" string; ...` | Your LLM is a function: it holds its own key and URL. |
| `RuntimeError: the table did not answer and this Shadow has no teacher: pass teacher=... (or llm=...), or fallback=... to run without an LLM` | Give it an LLM, or a `fallback` ([Running without an LLM](without-llm.html)). |
| `ValueError: 812 usable answers logged for 'intent'; need 1000 (pass min_rows=... to try with fewer; 100 is the hard minimum)` | `train()` was called too early. |
| `ValueError: min_rows must be at least 100` | 100 is the hard minimum. |
| `ImportError: training needs: pip install "shad0wllm[compile]" (scipy, scikit-learn)` | Install the training extras. |
| `ValueError: question 'x' is not in the bundle ([...])` | The bundle was trained for another question name. |
| `train()` returns `{"accepted": False, "reason": "new bundle certifies ..."}` | Not an error: `gate=True` (or `auto_train` with `retrain = "gated"`) kept the current table. |
| warning `auto_train skipped: ...` | A background retrain could not run (too few rows, missing training extras). The current table keeps serving. |

## Your LLM fails

`TeacherError` means the call to your LLM failed after retries, or its reply matched none of your options. Nothing is logged for that message. The error ends with why the LLM was asked, for example `(why your LLM was asked: no trained table yet: asked your model)`.

| Error | Fix |
|---|---|
| `TeacherError: openai/gpt-6-luna replied '...', which matches none of [...]` | The LLM answered outside your options. A few are normal. Many mean the prompt or model needs attention: add an `other` option, describe each option, or use a larger model. |
| `TeacherError: openai/gpt-6-luna: HTTP 401: ...` / `HTTP 403` | Wrong or missing API key. `shad0w config` shows where the key comes from (masked). |
| `TeacherError: openai/gpt-6-luna: HTTP 404: ...` | Wrong `base_url` or model name. For `systemone/...` the server must implement `/v1/systemone`; for `openai-decisions/...`, `/v1/decisions`. |
| `TeacherError: openai/gpt-6-luna: cannot reach http://...: ...` | The server is down or the URL is wrong. Network errors, 429 and 5xx are retried with back-off first; this is the final error. |

## The table answers nothing, or very little

The certificate lets the table serve only what it is sure enough of to stay within α. A low share is the certificate doing its job. Common causes:

- **Too few answers.** Below about 1,000 per question, the cutoff has to be strict. Log more and retrain; the share usually grows with the log.
- **Too many options for the data.** With 150 options and 1,000 rows, most options have a handful of examples. Log more, or merge rare options into `other`.
- **The decision depends on meaning the words don't carry.** Sentiment, sarcasm and anything needing reasoning stay hard for a word-level table, and shad0w correctly sends nearly everything to your LLM. See [Limits](limits.html).
- **Your LLM is inconsistent.** If it answers near-identical messages differently, no table can agree with it reliably. Use temperature 0 and a fixed prompt.
- **α is very strict.** Compare `shad0w train --out /tmp/a05 --alpha 0.05` with your current α ([Tuning](tuning.html#alpha)).

`shad0w try --bundle shad0w/intent/bundle "some text"` shows the table's answer, its confidence, the cutoff it needs, and why it deferred.

## Drift

Answers flagged `drift` go to your LLM. The drift guard compares how often the table is unsure on recent traffic (the last `drift_window` decisions, default 500) with how often it was unsure at calibration. When the gap exceeds `drift_margin` (default 0.03), it raises the flag. It clears by itself when traffic looks familiar again. If it stays raised, your traffic changed: retrain on the new log.

Bursty traffic can trip it even when the overall mix is unchanged. A long run of one hard message type (for example many "card not working" messages during an outage) fills the window with unsure answers. In our BANKING77 test the guard stayed silent on shuffled traffic but raised during a long run of one intent, then cleared. If your traffic comes in bursts and you prefer to keep serving, raise `drift_window` (say to `2000`) so one burst is a smaller part of the window.

## The proxy forwards everything

| Symptom | Cause and fix |
|---|---|
| Responses carry no `x-shad0w-source` header | The proxy did not see a decision. Mark it with the `X-Shad0w-Question` header, a `shad0w/<question>` model name, a forced tool with one enum parameter, or use `/v1/decisions` / `/v1/systemone`. A plain named `json_schema` is captured only with `--capture json_schema`. |
| `x-shad0w-source: teacher` on every call | No table yet for that question, or it certifies nothing. `shad0w status` shows each question's state; `shad0w train --dir shad0w --question <q>` builds it, and the proxy picks it up within a second. |
| `` model 'shad0w/<question>' needs `shad0w proxy --model <upstream model>` `` | Start the proxy with `--model`, or send `shad0w/<question>@<model>`. |
| HTTP 502 `shad0w proxy: cannot reach upstream ...` | `--upstream` is down or wrong; `shad0w doctor --upstream URL` checks it. |
| A tool call comes back as text | The request offered several tools without forcing one, or the tool's parameters hold more than one enum. Force the tool with `tool_choice`, or use the header. |
| Multi-question `/v1/decisions` calls still reach the upstream | Only the questions the table can answer are removed from the upstream request; `score` questions always go upstream. |

## "module 'shad0w' has no attribute 'decision'" or "cannot import name '__version__' from 'shad0w'"

You ran Python from a folder that holds the `shad0w/` **data** folder (where `decision()` keeps logs and tables by default), with shad0w installed in editable mode (`pip install -e`). Python then imports the data folder instead of the package. Run from another folder, use the `shad0w` command instead of `python -m shad0w`, or pick another data folder with `folder=` or `SHAD0W_FOLDER`. A normal `pip install shad0wllm` is not affected.

## Yes/no questions from 0.2

Yes/no answers are stored as booleans. Logs written by shad0w 0.2 through the proxy may hold the strings `"yes"` and `"no"`; 0.3 reads them correctly, but 0.2 learned every one of them as "yes". Retrain any yes/no bundle made with 0.2 through the proxy.

## pip times out

Some company networks point pip at an internal mirror that is slow or incomplete. Install from the public index for this one command:

```bash
PIP_INDEX_URL=https://pypi.org/simple pip install "shad0wllm[compile]"
```

## Still stuck

Run with `SHAD0W_LOG=debug` to print one line per decision (question, answer, source, confidence, flag, latency), and open an issue with that output and `shad0w doctor`'s.
