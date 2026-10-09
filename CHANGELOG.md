# Changelog

## 0.3.2

API keys you can see and pass anywhere, plus a faster start: enum options, `shad0w init` / `status` / `import`, warm starts and batches.

### Security
- **The proxy could send one client's API key on another client's behalf.** `Gateway.teacher_for` cached decision-model teachers by whether a key was present, not by which key, so the first client's forwarded key was reused for every later client on the same question and model. Teachers are now cached per key, by a SHA-256 id (never the raw key). Spot checks keep using the key of the request they check.
- **A chat spot check on a keep-alive connection could carry the next request's key.** The background re-ask read the handler's headers when it ran, by which time the connection could already be serving another client's request. It now uses a snapshot of the URL, method and headers of the request it checks.

### Added
- **Explicit API keys.** `decision(..., api_key=..., api_key_env=..., base_url=...)` are named parameters (and on `Shadow`, which now builds the LLM teacher itself from `llm="provider/model"`, and on `LLMTeacher`). `api_key` may be a zero-argument function, called on every request, for rotating or vault keys. Order: `api_key` > `api_key_env` > `shad0w.configure()` > the `api_key_env` setting > the provider's variable.
- **`shad0w.configure(llm=, api_key=, api_key_env=, base_url=, **settings)`**: process-wide defaults, ranked just below one call's keywords; `configure()` with no arguments clears them. JavaScript: `configure({ llm, apiKey, apiKeyEnv, baseURL })`.
- **Keys never leak.** Keys are held in `shad0w.Secret`, which prints as `sk-…3f9a` and pickles without the key (an environment-variable name survives). `repr`, `vars()`, traces, logs, error messages (HTTP error bodies and `Bearer` tokens are redacted), `shad0w doctor` and `shad0w config` show only the mask and where the key came from.
- **Settings `llm`, `base_url`, `api_key_env`** (`SHAD0W_LLM`, `SHAD0W_BASE_URL`, `SHAD0W_API_KEY_ENV`, `shad0w.toml`). An `api_key = ...` line in `shad0w.toml` is an error that says to store the variable's name instead.
- **`shad0w proxy --api-key-file PATH`** for Docker / Kubernetes secrets, re-read on every request. The startup banner says where the upstream key comes from (masked).
- **`shad0w doctor --llm`** prints `key: set via OPENAI_API_KEY (sk-…3f9a)`; without `--llm` it checks the `llm` from `shad0w.toml`.
- **Enum options.** `options=MyEnum`, `options=Literal["a", "b"]` and `options=bool`. String enums use their values, other enums their names (string values become descriptions). `@shad0w.decide` with `-> MyEnum` returns enum members.
- **`@shad0w.decide(llm="openai/gpt-6-luna", api_key=...)`** calls the LLM for you: the return annotation names the options and the function body is never run (`def route(text) -> Literal[...]: ...`).
- **`shad0w init` writes a commented `shad0w.toml`** (with `api_key_env`) **and a runnable `app.py`** using `decision()`: `--name`, `--llm`, `--options a,b,c`. The old starter is `shad0w init --files`.
- **`Shadow.warm_start(rows_or_path, text="text", label=...)`**: import answers or human labels you already have (list of dicts, `.jsonl`, `.json`, `.csv`) as `source="import"`, checked against the options, with counts.
- **`shad0w import --from openai-chat|openai-decisions --file F --question Q`**: exported request/response JSON Lines (OpenAI Batch output too) into a decision's log.
- **`shad0w status`**: one line per decision folder with rows vs `min_rows`, certified share at alpha, live agreement on spot checks since training, and the next step.
- **`decide_many(texts, concurrency=8)` / `adecide_many`**: the table answers first; only deferred texts go to your LLM, in parallel; order kept. JavaScript: `shadow.decideMany(texts, { concurrency })`.
- **`Decision.why`**: the reason for every decision in plain words (JavaScript decisions carry `why` too). A `TeacherError` raised during a decision says why the LLM was asked.
- **`repr(shadow)` and a Jupyter view**: question, number of options, state (`logging 240/1,000` or the certified share at alpha), the LLM and where its key comes from.

### Fixed
- **Tied confidences could serve more than the certificate covered.** The threshold search tested the top *k* calibration answers but then served every answer tied with the *k*-th one. With many identical confidences (repeated messages), realised disagreement could exceed α. Each candidate is now tested on exactly the answers it would serve, in both procedures.
- **Narrowed options could turn an uncertain answer into a certified one.** Asking for a subset of the options renormalised the confidence before comparing it with the threshold. Certification is now judged on the full-option answer; a narrowed request that changes the answer, or names no known option, is never certified.
- **A table file with non-finite weights is refused** by both the C and the Python loader (it used to serve NaN confidences as certified), and every threshold check fails closed on NaN (Python and JavaScript).
- **Answers that match no option are never logged**, also from function teachers (old names from `rename` and 0/1 for yes/no still are); Enum members are logged by name/value; an async `llm` passed to the sync `decide()` raises a clear `TypeError` instead of logging a coroutine.
- **`never_serve` works for yes/no questions** (`never_serve = ["yes"]`).
- **`on_new_option = "serve"`** answers are now reported as `certified=False` with flag `new_options_served`.
- **`record()` without a teacher** no longer tags rows as spot checks (they would have claimed a uniform sample that was not one). `record(..., source=shadow._src())` lets a caller that spot-checks served answers itself keep the sample uniform; the proxy's Decisions API and System One paths do this.
- **Spot checks cover every answer the table served**, including `force_threshold` and `on_new_option="serve"` answers, not only certified ones.
- **JavaScript**: with options both added and removed under `onNewOption: "serve"`, a removed option is never served.
- **The proxy** checks the shape of chat requests up front (`messages` and `tools` must be lists of objects) and answers 400; other errors are no longer reported as a malformed request.
- **CLI**: `shad0w ... | head` exits quietly; `stats` and `calibrate` skip a half-written log line; `SHAD0W_TIMEOUT=none` means no timeout again; the refused-key message names the actual key line.
- **`shad0w serve` / proxy**: malformed request shapes get a 400 instead of a dropped connection; a negative `Content-Length` is refused.
- **`shad0w certify` / `calibrate`** refuse fewer than 100 rows unless `--force` (one row used to silently stop the table from serving).
- **`shadow_compile` with 100–120 rows** no longer crashes (it always leaves rows to fit on); re-certifying keeps the drift settings.
- **Lone UTF-16 surrogates** in a message no longer crash `decide`.
- **One malformed line in a log** is skipped with a warning naming its line number instead of blocking `train`.
- **Settings**: a bad `SHAD0W_*` value names the variable; `[questions] intent = 3` gives a clear error; `auto_train=True`, negative `auto_train` and non-positive `timeout` are refused; `exposed="no"` means False; `options` needs two distinct choices; `OPENAI_API_KEY = ...` (any `*api_key` line) in `shad0w.toml` is refused like `api_key`.
- **Empty messages are never served** (flag `empty_input`, Python and JavaScript): the table would only have echoed its most common answer.
- **`shad0w serve`** stops quietly on Ctrl-C; `POST /v1/decide` without `state` explains the body it needs.
- **An edited `shad0w.toml` could be read stale.** The parsed file was cached by modification time alone, so two edits within the same timestamp tick (common on Windows) kept the first. It is now keyed by the nanosecond time and the file size.
- **`shad0w proxy --bundle`** refuses entries without `QUESTION=` and paths with no bundle, instead of ignoring them.
- **`shad0w.toml`**: unknown keys are reported with a suggestion (`alpah` → did you mean `alpha`), and a syntax error names the file.
- **`shad0w try`** says "would ask your LLM; the table's guess" for answers it would defer; `shad0w status` also accepts `--dir`.
- **`llm_teacher()`** takes its model from `shad0w.configure(llm=...)` / `SHAD0W_LLM` when none is passed.
- **Python 3.10**: `tomli` is now a dependency there (`shad0w init` writes a `shad0w.toml`); the test suite runs on 3.10.
- **CLI**: unreachable servers, busy ports and similar OS errors print one line and exit 2; `decide`, `serve` and `bench` have help text; the generated `app.py` imports `os`.
- `shad0w calibrate` updated the threshold in `manifest.json` but not in `certificate.json`, so `shad0w report` showed a stale certificate.
- `certify_bundle` dropped questions with no fresh records from the certificate while they kept serving on their old threshold; their previous entry is now kept and marked `recertified: false`, with a warning.
- A spot check whose LLM call failed stayed in the pending list forever (a small leak).
- `adecide()` gains the `probabilities=` keyword that `decide()` has.
- **`mode="shadow"` with `force_threshold` served table answers to users.** Shadow mode now never serves; forced answers count as `would_serve`.
- `shad0w train --rename OLD=NEW` without `--schema` reused the old `schema.json` and dropped every renamed row; the old schema is now renamed too.
- `shad0w config` printed invalid values (say `SHAD0W_MODE=on`) without complaint; it now reports them and exits 2.
- `shad0w shadow` and `shad0w compile` ignored `alpha` from `shad0w.toml` / `SHAD0W_ALPHA` (they defaulted to 0.05).
- `shad0w compile` with a label that is not in the schema printed a raw `'x' is not in list`.
- JavaScript `configure({ auditRate })` was accepted but ignored.
- A subset of options with almost no probability mass was not renormalised (confidence ~1e-33 instead of probabilities that sum to 1).
- A teacher built with `complete=` reported itself as the default model in errors; it is now named after the function.
- TypeScript: `Flag` includes `options_changed` and `option_removed`; `Table.decide` and `Bundle.decide` accept `probabilities`.

### Changed
- **Wheels for Python 3.14 and musllinux (Alpine)**; release builds use cibuildwheel 3.4, npm publishes only after PyPI. The sdist now includes the test suite's mock LLM and the JavaScript sources, so `pytest` runs from it.
- **Clear errors instead of silent drops.** `api_key`, `base_url` or other LLM keywords with a function `llm` raise `TypeError` (they used to be ignored). An unknown keyword raises `TypeError` with a "did you mean" suggestion instead of a misleading missing-key error. The missing-key message names `api_key=`, `api_key_env=` and `shad0w.configure`.
- `LLMTeacher.api_key` is a `Secret` (compare with `==`, read with `.get()`); `teacher.key_source` describes it.
- `examples/quickstart.py` uses `decision()`, `decide_many()` and `train()` with an offline stand-in LLM.
- The built-in dashboard uses the shad0w palette (black, Prussian blue, orange, alabaster, white); LLM bars are striped so table and LLM never differ by colour alone.

## 0.3.1

Changing options safely, scale, running without an LLM, and a faster JavaScript runtime.

### Fixed
- **Adding or removing an option after training no longer serves wrong answers.** A table trained on the old option list answered messages about a new option with an old label, marked certified (more than half of a new option's messages in our test), and kept answering with options you had removed. shad0w now compares the options you offer with the ones the table learned. Added options send every decision to your LLM (`flag="options_changed"`) until you retrain; removed options are never served (`flag="option_removed"`); everything your LLM answers meanwhile is logged, so the next `train()` learns the new list. The retrain gate never blocks this case. The proxy applies the same rule to the options each request names, and the chat path now parses replies against the request's own enum, so a new option is learned instead of dropped.
- **`adecide()` with a canary could run a sync teacher on the event loop.** The rollout policy was drawn twice; it is now drawn once.

### Added
- **`rename={"old": "new"}`** (also `SHAD0W_RENAME`, TOML and `shad0w train --rename OLD=NEW`): rename options without retraining; the next training run renames the old rows in the log.
- **`on_new_option = "defer" | "serve"`**: what to do with options the table never learned.
- **`max_mb`** (also `shad0w train --max-mb`): a size budget per table. Training keeps the most informative patterns and refits on them; the certificate is computed on the capped table. On 150 options a 1 MB budget certified the same share as the 3.6 MB table without one.
- **`fallback=`**: run without an LLM. A value or a function answers what the table is unsure about (`source="fallback"`, never logged as training data). Train on human labels, rules or an existing classifier.
- **`decision()` name is optional** (default `"decision"`), in Python and JavaScript.
- **Large option sets train faster with the same quality**: the C search runs on a subsample above 2M table cells, the temperature uses 3 folds, and L-BFGS keeps a shorter history above 10M weights, so up to 1,024 options train on a laptop.
- `Shadow.stats()` reports `options_added` / `options_removed`; the JS `Shadow` has `optionsAdded`, `optionsRemoved`, `rename`, `onNewOption`, `fallback`.
- Benchmark `bench/b9_scale.py`: size, speed, training time and certified share from 10 to 500 options.

### Changed
- **Faster JavaScript.** Hashing works straight on the UTF-8 bytes without allocations, rows are found through a hash table (as in the C core) instead of a binary search, and `bundle.decide(text, { probabilities: false })` skips the per-option map. On a 77-option table on a laptop CPU, `Shadow.decide` (which uses the fast path) went from about 20 µs to about 8 µs per decision, and `bundle.decide` with probabilities from about 20 µs to about 15 µs. Answers are bit-identical (checked on 20,000 inputs and by the parity tests).
- The two copies of the retrain quality gate (Python and CLI) are one helper, `shad0w.shadow.gate_reason`.

## 0.3.0

Zero-code for the new decision APIs, every knob in one place, and retraining you can trust.

### Added
- **OpenAI Decisions API and System One, with no marking.** `shad0w proxy` intercepts `POST /v1/decisions` (OpenAI, `client.decisions.create(...)`) and `POST /v1/systemone` (Jev, Kev, Laya, Ollaya, llama.cpp). Each choice or predicate question becomes a shad0w question. Certified ones are answered by the table with per-option probabilities and a confidence. Only the remaining questions are forwarded upstream, and their answers are logged. Score questions always pass through. `shad0w serve` speaks both formats too.
- **Decision models as teachers.** `llm="openai-decisions/gpt-6-luna"` and `llm="systemone/<model>"` with `base_url=`.
- **Settings layer** (`shad0w.config`, `shad0w.toml`, `SHAD0W_*`). Precedence is code > environment > file > defaults, with `[questions.<name>]` sections. `shad0w config` prints every effective value and its source, and `shad0w config --init` writes a commented starter file.
- **Rollout knobs.**
  - `mode = "serve" | "shadow" | "off"`: shadow mode computes the table's answer, returns your LLM's, and counts `would_serve` and `shadow_disagreements`.
  - `canary`, `never_serve` and `min_confidence`: these can only make serving stricter.
  - `force_threshold`: serves uncertified answers with `certified=False, flag="manual_threshold"`, and warns loudly.
  - `drift_window` and `drift_margin`.
- **`shad0w doctor`.** It checks Python, the C core, the training dependencies, Node, the config, provider keys, upstream reachability, bundles and logs.
- **Proxy.**
  - Tool-shaped requests (a forced `tool_choice` whose function has one enum parameter) are recognised and answered with `tool_calls`, streaming included.
  - New `X-Shad0w-Options` header.
  - New flags: `--delta --min-rows --mode --canary --never-serve --min-confidence --capture --timeout --trace --bundle Q=PATH --llm-latency-ms --config`.
  - New `POST /v1/playground`.
- **Dashboard playground.** Type a message to see the table's answer, its confidence against the threshold, the reason, and optionally your LLM's answer side by side.
- **`Decision` fields.** It gains `question`, `threshold` and `top` (filled with `probabilities=True`).
- **`Shadow` methods.** It gains `peek(text)` (the table's answer only when it would be served) and `record(text, answer)` (log an answer you fetched yourself), for integrations.
- **`@shad0w.decide`.** A decorator whose `-> Literal[...]` (or `-> bool`) return annotation names the options; the function body is the teacher.
- **`Metrics(llm_latency_ms=...)`.** "Time saved" now shows before the first LLM call is measured.

### Changed
- **json_schema capture is opt-in.** The proxy no longer treats every request with a `response_format.json_schema.name` as a decision; use `--capture json_schema` to restore that. Unmarked calls now pass straight through, as the README always said.
- **The `/v1/systemone` response shape changed.** On `shad0w serve` it now returns the System One shape (`answers{}` with `choice`, `confidence`, `probabilities`, or `noul`/`probability` for yes/no, plus `latency_ms`) instead of the `/v1/decide` shape. `/v1/decide` is unchanged.
- **Gated retraining.** `auto_train` keeps the current bundle unless the new one certifies at least 80% of its share (`retrain = "always"` restores the old behaviour). The auto-train counter is restored from the certificate after a restart. `shad0w train --gate` does the same on the command line.
- **Audit rows sample deferred traffic too.** A deferred answer is logged as `source: "audit"` with probability `audit_rate`, so audit rows are a uniform sample of all traffic. `train()` certifies on them when at least 100 exist, and fits on the rest. The certificate gains an additive `calibration` key: `uniform-audit` or `held-out-split`. Raise `audit_rate` (e.g. 0.05) during ramp-up for an honest calibration set.
- **torch moved to `[torch]`.** `[compile]` is now scipy + scikit-learn. Training without torch uses the scipy solver and gives the same tables, more slowly. `[logs]` pulls `[torch]`.
- **`Shadow` policy arguments are keyword-only.** `audit_rate`, `exposed`, `auto_train`, `alpha` and the new knobs must be passed by keyword; defaults come from the settings layer.

### Fixed
- **The CLI never crashes on output encoding.** On Windows, piped output uses cp1252, which has no `✓` or `→`. `shad0w try` crashed with `UnicodeEncodeError`; it now prints replacement characters instead.
- **The C core loader picks this platform's build.** A source checkout holding builds for several platforms (say a macOS `_reflex*.so` next to the Linux one) could make Linux load the macOS binary. It now prefers this interpreter's extension suffix.
- `decision(..., complete=fn)` raised `TypeError`.
- Yes/no questions trained from string answers ("no") learned every label as "yes".
- `explain()` and `adecide()` fed the drift guard (twice for `adecide`).
- Missing bundles, missing logs and a missing `[compile]` produced raw tracebacks. The CLI now prints one line and exits 2.
- The auto-train counter was updated outside the lock.

## 0.2.0

Plug shad0w into any LLM in one line, or with no code change at all, and see what it is doing.

- **`shad0w.decision(name, options=..., llm="openai/gpt-6-luna")`.** One call from nothing to a working cascade. Your LLM answers and is logged; `.train()` builds and certifies the table and hot-swaps it in; after that, certified answers come from the table.
- **`shad0w.llm_teacher(...)`.** Any OpenAI-compatible chat model becomes a teacher. Built-in providers: OpenAI, Anthropic, Gemini, Groq, Together, OpenRouter, Mistral, DeepSeek, Fireworks, xAI, Ollama, vLLM and LM Studio. It uses structured outputs (an enum of your options), falls back to plain text automatically, maps every reply to exactly one option, retries, and drops parameters a model refuses. `client=` reuses an openai SDK client; `complete=` plugs in LiteLLM, LangChain or anything else. Standard library only.
- **`shad0w proxy`: an OpenAI-compatible gateway.**
  - Point `base_url` at it and mark decisions with the `X-Shad0w-Question` header, a `shad0w/<question>` model name, or a `json_schema` name.
  - Certified answers come back as a normal `chat.completion` (text, JSON object or your JSON schema, streaming too) in microseconds.
  - Everything else is forwarded untouched and logged. Bundles retrained by `shad0w train` are picked up automatically, and `--auto-train N` retrains on its own.
- **Observability.**
  - A live dashboard at `/` on `shad0w proxy` and `shad0w serve`. It shows table share, LLM calls, time and money saved, latency, deferral reasons, and the live spot-check bound vs α.
  - Prometheus at `/metrics` and JSON at `/v1/stats`.
  - `SHAD0W_LOG=debug` for per-decision logs, `on_decision=` hooks, `trace=` decision files, and OpenTelemetry spans with `SHAD0W_OTEL=1`.
- **New commands.** `shad0w train` detects questions and options from the log. `shad0w try` shows each answer, its speed and why. `shad0w stats` reports what has been logged and whether it is ready to train. `shad0w watch` shows live counters.
- **`Shadow` upgrades.** `train()`, `auto_train=N`, `adecide()` for async code, `explain(text)`, `on_decision`, `trace`, shared `metrics`, and a per-call `teacher=` override. Spot checks now run in the background, so they never slow an answer. Answers that match no option are never logged.
- **JavaScript.** `decision()`, `Shadow` and `openaiTeacher()` mirror the Python API (fetch-based, so they work in Node, browsers, Workers, Deno and Bun), with full TypeScript types.
- **C.** A public header, `shad0w.h`.
- **Faster HTTP replies.** `shad0w serve` and `shad0w proxy` now set `TCP_NODELAY`. Headers and body went out as two writes, which could stall each keep-alive reply by ~40 ms (Nagle plus delayed ACK). A table answer through the proxy now takes ~1.4 ms end to end, measured at the openai SDK client.
- **Training works without torch.** If torch is not installed, training solves the same convex objective with scipy's L-BFGS. It is slower, but needs no extra download.

## 0.1.1

- **Certificate holds up under LLM-like teachers.** Teachers that are wrong a few percent of the time regardless of the input (sampling noise) could make the strict-to-lenient fixed-sequence test stop early. A 97.6%-agreeing teacher then certified 0%. The default procedure (`auto`) now runs fixed-sequence and a Bonferroni test over 12 pre-registered coverage levels, each at δ/2, and keeps the more lenient threshold. That is still valid at δ by the union bound. In simulation, offload under flat teacher noise rose from 39% to ~100%; where disagreement rises as confidence falls, ~98% of the fixed-sequence offload was kept. The violation rate stayed within δ for every procedure. The end-to-end case above now certifies 100% with 2.3% realised disagreement on fresh traffic (bound 5%).
- `SelectiveRiskController(alpha, delta, procedure="auto" | "fixed-sequence" | "bonferroni")`. The certificate records which procedure produced the threshold.
- Release pipeline: PyPI and npm both publish through Trusted Publishing (OIDC). No tokens are stored.

## 0.1.0

First public release.

- **Shadow mode.** `shad0w shadow` compiles a table from your model's logged answers and certifies how often it disagrees with that model (Learn-then-Test, Clopper–Pearson, fixed-sequence testing). `shad0w certify` re-certifies on fresh traffic, and `shad0w report` prints the certificate.
- **`shad0w.Shadow` wrapper and `@shad0w.cascade` decorator.** Put the table in front of your existing model call in one line. The wrapper has a log-only start (no bundle yet), logs deferred answers for the next compile, and runs live spot-check audits with a Clopper–Pearson bound in `stats()`.
- **Robust certificate start.** The test sequence starts where a run with disagreement at half of α can still certify. One unlucky disagreement among the most confident answers no longer certifies nothing. A simulation test checks that the violation rate stays ≤ δ.
- **Three runtimes, identical decisions.** Python (numpy only), a C core shipped inside the wheels, and a dependency-free JavaScript package (`npm i shad0wllm`, CJS + ESM + types, with `Bundle.load` applying the certified threshold).
- **Strict table loading** in all three runtimes. Unknown versions, impossible shapes, unsorted keys and truncated or padded files are refused.
- Typed answers with confidence, an exact robustness radius, `certified` and `flag`. Also a label-free drift guard and anytime-valid auditors.
- Human-label mode (`shad0w compile --data`, `shad0w calibrate`) and label-free mode (`shad0w compile --unlabeled`).
- Fast Python path: `probabilities=False` skips the per-option dict (about 8 µs per call through the C core, versus about 15 µs with it).
- Reference HTTP server: `POST /v1/decide` (alias `/v1/systemone`) and `GET /v1/health`.
- `shad0w init` and `shad0w --version`.
