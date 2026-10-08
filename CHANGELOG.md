# Changelog

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
