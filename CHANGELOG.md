# Changelog

## 0.2.0

Plug shad0w into any LLM in one line, or with no code change at all, and see what it is doing.

- **`shad0w.decision(name, options=..., llm="openai/gpt-4o-mini")`.** One call from nothing to a working cascade. Your LLM answers and is logged; `.train()` builds and certifies the table and hot-swaps it in; after that, certified answers come from the table.
- **`shad0w.llm_teacher(...)`.** Any OpenAI-compatible chat model becomes a teacher. Built-in providers: OpenAI, Anthropic, Gemini, Groq, Together, OpenRouter, Mistral, DeepSeek, Fireworks, xAI, Azure, Ollama, vLLM and LM Studio. It uses structured outputs (an enum of your options), falls back to plain text automatically, maps every reply to exactly one option, retries, and drops parameters a model refuses. `client=` reuses an openai SDK client; `complete=` plugs in LiteLLM, LangChain or anything else. Standard library only.
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
