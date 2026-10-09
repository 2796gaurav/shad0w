<p align="center">
  <a href="https://2796gaurav.github.io/shad0w">
    <img src="https://2796gaurav.github.io/shad0w/assets/hero.svg" alt="Requests race in. The shad0w table answers most of them in microseconds; the hard ones still go to your LLM." width="100%">
  </a>
</p>

<h1 align="center">shad0w</h1>

<p align="center">
  <b>Your app asks an LLM the same kinds of small questions all day.<br>
  shad0w learns its answers and serves them in microseconds, with a written guarantee on how often it disagrees.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/shad0wllm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/shad0wllm?label=pip%20install%20shad0wllm&color=7c5cff&logo=pypi&logoColor=white&cacheSeconds=3600"></a>
  <a href="https://www.npmjs.com/package/shad0wllm"><img alt="npm" src="https://img.shields.io/npm/v/shad0wllm?label=npm%20i%20shad0wllm&color=7c5cff&logo=npm&logoColor=white&cacheSeconds=3600"></a>
  <a href="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.14" src="https://img.shields.io/badge/python-3.10%E2%80%933.14-3776ab?logo=python&logoColor=white">
  <a href="https://github.com/2796gaurav/shad0w/blob/main/LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"></a>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w"><b>Website</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/demo/"><b>Try it in your browser</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/pong/"><b>Play Pong against it</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/arena/"><b>Snake arena</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/"><b>Docs</b></a>
</p>

---

## Quick start

```bash
pip install "shad0wllm[compile]"
```

**1. Already calling the OpenAI Decisions API, or a decision model such as Jev, Kev or Laya? Change one URL.**

```bash
shad0w proxy --upstream https://api.openai.com/v1        # or any /v1/systemone server; dashboard at http://localhost:8010
```

```python
client = OpenAI(base_url="http://localhost:8010/v1")     # the only change
client.decisions.create(model="gpt-6-luna", input=text, questions=[...])   # unchanged
```

Decision requests already name each question and list its answers, so nothing needs marking. Questions the table is certified on come back from it in the same JSON shape; the rest go upstream and are logged so the table keeps learning. Details: [Decisions API & System One](https://2796gaurav.github.io/shad0w/docs/decisions-api.html).

**2. Asking a chat model to pick one option? One line of Python.**

```python
import os
import shad0w

intent = shad0w.decision(
    "intent",                 # the decision's name: its log, table, dashboard label (optional; default "decision")
    options={"refund": "wants money back", "lost_card": "card lost or stolen", "balance": "asks about balance",
             "other": "anything else"},   # always give it an "other": off-topic messages land there, not in a real intent
    llm="openai/gpt-6-luna",   # or anthropic/…, gemini/…, groq/…, ollama/llama3.1, systemone/kev, any OpenAI-compatible server
    api_key=os.environ["OPENAI_API_KEY"],   # or api_key_env="MY_KEY_VAR", or a function that returns the key
)

intent("my card was stolen")    # day one: your LLM answers and shad0w writes the answer down
intent.train()                  # after ~1,000 answers: builds and certifies a small table, in about a minute or less
intent("my card was stolen")    # from then on: answered locally when certified
# Decision(answer='lost_card', source='table', confidence=0.99, certified=True, flag=None, latency_us=..., question='intent', ...)
```

Anything the table is unsure about still goes to your LLM and gets logged, so the table keeps learning. Have a function already? `@shad0w.decide()` on `def route(text) -> Literal["billing", "tech"]` (or `-> MyEnum`) does the same. Starting fresh? `shad0w init` writes a commented `shad0w.toml` and a runnable `app.py`; `shad0w status` tells you what each decision needs next.

**3. Chat completions with no code change.** Run the same proxy and mark which calls are decisions:

```python
client = OpenAI(base_url="http://localhost:8010/v1")      # the only change
client.chat.completions.create(model="gpt-6-luna", messages=msgs,
                               extra_headers={"X-Shad0w-Question": "intent"})   # or a forced tool with one enum parameter
```

Unmarked calls (chat, embeddings, streaming) pass straight through.

## Is it for me?

**Yes, if your app calls an LLM to pick one answer from a fixed list.** For example:

| You ask the LLM… | shad0w answers it from a table |
|---|---|
| Which intent is this support message? | `refund`, `lost_card`, `balance`, … |
| Which team gets this ticket? | `billing`, `tech`, `sales` |
| Is this message spam / abusive / off-topic? | `yes` / `no` |
| Which tool should the agent call next? | `search`, `calculator`, `none` |
| What topic is this article? | `sports`, `business`, `tech`, … |

**What changes when you add it:**

| | Before | After |
|---|---|---|
| Typical decision | an LLM call: hundreds of ms | **microseconds** for the share the table is certified on ([Numbers](#numbers)) |
| Share answered without the LLM | 0% | most of it on intent and topic tasks once enough answers are logged (about 1,000 for a handful of options, 5,000 or more for dozens), and it grows with traffic ([Numbers](#numbers)) |
| Agreement with your LLM | — | disagrees on at most **α = 5%** of the answers it serves, with 90% confidence, on traffic like the traffic it learned from; **certified** before it serves anything ([where it went above](#numbers)) |
| Runs on | someone's GPU | your CPU, in-process; also in the browser and on edge workers |
| Your code | `ask_llm(text)` | `intent(text)` |

**Not a fit:** open-ended generation, questions that need reasoning or world knowledge, and labels that depend on tone or sarcasm (on sentiment the table certifies almost nothing and passes everything to your LLM, which is the correct behaviour).

## How it works

1. **Watch.** Your LLM keeps answering. Each `(text, answer)` pair is written to a log.
2. **Train.** The log becomes a hashed word/character table with 8-bit weights. There is no GPU, no tokenizer and no human labels, and it takes seconds to a minute on a laptop.
3. **Certify.** On answers the table never trained on, a [Learn-then-Test](https://arxiv.org/abs/2110.01052) procedure picks a confidence threshold. Above it, the table disagrees with your LLM at most α of the time, with 90% confidence.
4. **Serve.** Above the threshold the table answers in microseconds. Below it, your LLM answers and the answer is logged. 1% of decisions are spot-checked against the LLM, so you can see live agreement, not only the certificate, and retraining certifies on a uniform sample of real traffic.

<p align="center"><img src="https://2796gaurav.github.io/shad0w/assets/how-it-works.svg" alt="watch, train, certify, serve" width="100%"></p>

## Tune it

Nothing needs configuring. When you want to, every setting can come from code, a `SHAD0W_*` environment variable or a `shad0w.toml`, in that order of precedence:

```toml
# shad0w.toml
alpha = 0.05             # the most the table may disagree with your LLM on what it serves
audit_rate = 0.02        # share of decisions spot-checked against your LLM
mode = "shadow"          # watch first: the table decides but your LLM's answer is returned; then "serve"
canary = 0.1             # when serving, let the table answer 10% of what it is certified on
never_serve = ["fraud"]  # labels that always go to your LLM

[questions.refund_check]
alpha = 0.01             # stricter for one question
```

```bash
shad0w config            # every effective value and where it came from
shad0w doctor            # checks the C core, training deps, keys, upstream and your bundle
```

`min_confidence` raises the bar on top of the certificate. `force_threshold` lowers it, and every answer it lets through is marked `certified=False`. Rollout guide: [shadow → canary → serve](https://2796gaurav.github.io/shad0w/docs/rollout.html). All settings: [Configuration](https://2796gaurav.github.io/shad0w/docs/configuration.html).

## Change, scale, or no LLM at all

- **Your options will change. That is safe.** Add an option and the table steps back, sending every decision to your LLM until it has relearned. Remove one and the table never answers with it. Rename one with `rename={"lost_card": "card_lost"}` and nothing needs retraining. [Changing your options](https://2796gaurav.github.io/shad0w/docs/options.html)
- **Up to 1,024 options per question**, still microseconds per decision. `max_mb=1` caps the table's size; training keeps the most informative patterns and the certificate is computed on the capped table. [Scale](https://2796gaurav.github.io/shad0w/docs/scale.html)
- **No LLM?** Train on human labels, rules or an existing classifier, and serve with `fallback="needs_review"` (or a function) for what the table is unsure about. [Running without an LLM](https://2796gaurav.github.io/shad0w/docs/without-llm.html)

## Use it from anywhere

| | Install | Three lines |
|---|---|---|
| **Python** | `pip install shad0wllm` | `sh = shad0w.decision("intent", llm="openai/gpt-6-luna", options=[...])` → `sh(text).answer` |
| **OpenAI Decisions API / System One** (zero code) | `shad0w proxy --upstream <url>` | `OpenAI(base_url="http://localhost:8010/v1")`; nothing to mark |
| **Any OpenAI chat client** (zero code) | `shad0w proxy --upstream <url>` | `OpenAI(base_url="http://localhost:8010/v1")` plus the header `X-Shad0w-Question` |
| **JavaScript / TypeScript** (Node, browser, Workers, Deno, Bun) | `npm i shad0wllm` | `const intent = await decision("intent", {options, llm: "openai/gpt-6-luna", bundle: "bundle/"})` → `await intent.decide(text)` |
| **HTTP** (any language) | `shad0w serve --bundle bundle/` | `POST /v1/decide {"state": "my card was stolen"}`, or `/v1/decisions` / `/v1/systemone` |
| **C / C++ / FFI** | copy `shad0w.h` + `reflex.c` | `s0_load("intent.s0")` → `s0_decide(m, text, len, probs, &r)` |
| **LangChain, LiteLLM, Vercel AI SDK** | — | `shad0w.integrations.langchain.shad0w_runnable(sh, llm)`, `shad0w.integrations.litellm.completion(sh, ...)`, `shad0wMiddleware(sh)` |
| **Your own function** | — | `@shad0w.decide()` on `def route(text) -> Literal[...]`, or `shad0w.Shadow("bundle/", teacher=my_classify, log="log.jsonl")` |

Providers built in: `openai`, `anthropic`, `gemini`, `groq`, `together`, `openrouter`, `mistral`, `deepseek`, `fireworks`, `xai`, `ollama`, `vllm` and `lmstudio`, plus `openai-decisions` and `systemone` for decision models. Any other OpenAI-compatible server works with `base_url=`. Recipes for each are in the [docs](https://2796gaurav.github.io/shad0w/docs/).

## API keys

```python
shad0w.decision("intent", options=[...], llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"])  # the key
shad0w.decision("intent", options=[...], llm="openai/gpt-6-luna", api_key_env="SUPPORT_BOT_KEY")       # where it lives
shad0w.decision("intent", options=[...], llm="openai/gpt-6-luna", api_key=vault.current_key)           # called per request
shad0w.configure(llm="openai/gpt-6-luna", api_key_env="OPENAI_API_KEY")                                 # once per process
```

Order: `api_key` > `api_key_env` > `shad0w.configure(...)` > `api_key_env` in `shad0w.toml` / `SHAD0W_API_KEY_ENV` > the provider's usual variable (`OPENAI_API_KEY`, …). The key never appears in `repr()`, logs, pickles, errors, `shad0w doctor` or `shad0w config`: at most `sk-…3f9a`. `shad0w.toml` holds the variable's name, never the key. The proxy forwards each client's own key, or sends one from `--api-key-env` / `--api-key-file`. JavaScript: `decision("intent", { options, llm, apiKey })`, where `apiKey` may be a function.

## See what it is doing

```bash
shad0w try --bundle shad0w/intent/bundle "my card was stolen" "is it raining?"
#   ✓ table     lost_card  (confidence 1.000, needs 0.911)  14.1 µs
#   → your LLM  balance    (confidence 0.717, needs 0.911)  table not sure enough: asked your model
shad0w stats --log shad0w/intent/log.jsonl     # what has been logged, and whether it is ready to train
shad0w watch                                   # live counters from a running proxy or server
```

- **Dashboard.** `shad0w proxy` and `shad0w serve` serve a live page at `/`. It shows the share answered by the table, LLM calls and time saved (and money, with `--cost-per-call`), table vs LLM latency, why the table deferred, and the live spot-check bound against α.
- **Prometheus** at `/metrics`. **JSON** at `/v1/stats`.
- **Logs:** `SHAD0W_LOG=debug` prints one line per decision.
- **Hooks:** `on_decision=fn` for Langfuse, Datadog or your own analytics; `trace="decisions.jsonl"` records every decision; `SHAD0W_OTEL=1` emits OpenTelemetry spans.

## Numbers

<!-- numbers:start -->
Measured on public datasets with the certificate at α = 5% (full tables, sources and hardware: [benchmarks](https://2796gaurav.github.io/shad0w/docs/benchmarks.html)):

- **61–86%** of the traffic answered without the LLM on 3 intent and topic tasks, teacher gpt-4.1-mini (OpenAI). Realised disagreement with the teacher on those answers: **3.0–6.0%**; CLINC150 went above α because its test traffic has far more off-topic messages than the logged traffic (the drift guard flags this; see benchmarks, "Where shad0w loses"). End-to-end accuracy stayed within a point of the teacher alone.
- **2.4 µs** per decision in Python (C core inside), **1.0 µs** in C, **10 µs** in JavaScript, **0.53 ms** end to end through the proxy at the openai SDK client. A hosted LLM call: 1416 ms.
- The share grows with traffic: **0% → 60%** as logged answers went from 250 to 10,000 (BANKING77).
- With Kev-0.8B (local, MLX) as the teacher, mean latency fell from **382 ms to 115 ms** (measured end to end) and accuracy moved +1.0 points.
- Reproduce it on a laptop with a free teacher model (bge-small, 68% accurate) on BANKING77: the in-browser demo answers 63.3% locally with 3.3% realised disagreement; `python examples/shadow_demo.py` runs the same with the drift guard on and answers a few points less (about 59%).
- Where it loses: sentiment (SST-2) certifies 0% and defers everything; legal clauses (LEDGAR) 13%.

Every number above has a source file, hardware, teacher and date on the [benchmarks page](https://2796gaurav.github.io/shad0w/docs/benchmarks.html).
<!-- numbers:end -->

## How it compares

<!-- compare:start -->
| method | answered without the LLM @ α = 5% | disagreement with the LLM | p50 per decision | cost / 1k decisions | needs | guarantee |
|---|---|---|---|---|---|---|
| **shad0w table** | **61%** | **3.0%** | **4.0 µs** | **$0.204** | **your LLM's logged answers** | **certified ≤ α vs your LLM** |
| semantic cache (bge-small, nearest logged answer) | 0% | — | 7.5 ms | $0.520 | embedding model + logged answers | same certifier applied |
| embedding kNN router (bge-small, k=10) | 69% | 2.9% | 7.6 ms | $0.161 | embedding model + logged answers | same certifier applied |
| small fine-tuned head (LR on bge-small) | 75% | 3.4% | 7.2 ms | $0.129 | embedding model + logged answers | same certifier applied |
| exact-match cache | 0% | — | 1.2 µs | $0.520 | logged answers | exact repeats only |
| keep calling the LLM | 0% | 0.0% | — | $0.520 | API or GPU | it is the reference |
| Kev-0.8B decision model (local) | 100% (it replaces your LLM) | 23% vs gpt-4.1-mini | 271 ms | $0 (your hardware) | a GPU/MLX machine, ~1-3 GB | calibrated probabilities, no bound |
| Julia-1 decision model (local) | 100% (it replaces your LLM) | 41% vs gpt-4.1-mini | 11.9 ms | $0 (your hardware) | llama.cpp, a laptop | calibrated probabilities, no bound |
| Laya 421M decision model (local) | 100% (it replaces your LLM) | 58% vs gpt-4.1-mini | 91.6 ms | $0 (your hardware) | a GPU or fast CPU, 0.8 GB | calibrated probabilities, no bound |
| hosted decision model (OpenAI Decisions API, gpt-6-luna) | 100% (it replaces your LLM) | not measured (no API key) | ~150 ms claimed | $0.030 per 1k | API key | none (probabilities only) |
| hosted decision model (TypeSafe Jev 1.13) | 100% (it replaces your LLM) | not measured (no credits) | 70–500 ms claimed | $0.0126 per 1k | API key | none (probabilities only) |

Accuracy against the true labels on the same test set: Kev-0.8B 80.0%, gpt-4.1-mini 76.3%, shad0w's cascade 76.5%. A good open decision model makes a good teacher: with Kev-0.8B (local, MLX) as the teacher, shad0w answered 70% of the traffic itself and mean latency fell from 382 ms to 115 ms.
<!-- compare:end -->

**Works with decision models.** Jev, Kev, Laya and the OpenAI Decisions API can be shad0w's teacher (`llm="systemone/kev"`, `llm="openai-decisions/gpt-6-luna"`) or sit behind its proxy. shad0w takes the repeat traffic it is certified on; the model keeps the rest.

## Good to know

- **The guarantee is against your LLM, not the truth.** If your LLM is wrong, the table is wrong the same way. For a bound against the truth, calibrate on ~300 human labels (`shad0w calibrate`).
- **It holds for traffic like the traffic it was certified on.** Re-train when your traffic changes. The drift guard and live spot checks show you when that is.
- **Runtime installs only numpy**; the C core is inside the wheel. Training (`[compile]`) adds scipy and scikit-learn. `[torch]` adds torch for faster training on large logs; without it, training uses scipy.
- **Nothing leaves your machine** except the LLM calls you already make.

## Learn more

[Get started](https://2796gaurav.github.io/shad0w/docs/) · [Reference: every parameter](https://2796gaurav.github.io/shad0w/docs/reference.html) · [Tuning](https://2796gaurav.github.io/shad0w/docs/tuning.html) · [The guarantee](https://2796gaurav.github.io/shad0w/docs/guarantee.html) · [Benchmarks](https://2796gaurav.github.io/shad0w/docs/benchmarks.html) · [Blog: System Zero](https://2796gaurav.github.io/shad0w/blog/system-zero.html) · [Playground](https://2796gaurav.github.io/shad0w/demo/)

## Contributing

Try it on one of your decisions and [share your numbers](https://github.com/2796gaurav/shad0w/issues/new?template=results.yml), failures included. That is the most useful contribution. For code, see [CONTRIBUTING.md](https://github.com/2796gaurav/shad0w/blob/main/CONTRIBUTING.md).

<details><summary>Cite</summary>

```bibtex
@software{chauhan2026shad0w,
  author = {Chauhan, Gaurav},
  title  = {shad0w: certified microsecond decisions compiled from the model you already run},
  year   = {2026},
  url    = {https://github.com/2796gaurav/shad0w}
}
```
</details>

<p align="center">Built by <a href="https://github.com/2796gaurav">Gaurav Chauhan</a> · Apache-2.0</p>
