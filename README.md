<p align="center">
  <a href="https://2796gaurav.github.io/shad0w">
    <img src="https://2796gaurav.github.io/shad0w/assets/hero.svg" alt="Requests race in. The shad0w table answers most of them in microseconds; the hard ones still go to your LLM." width="100%">
  </a>
</p>

<h1 align="center">shad0w</h1>

<p align="center"><b>Your LLM's repeat decisions, answered in microseconds, with a guarantee on how often it disagrees.</b></p>

<p align="center">
  <a href="https://pypi.org/project/shad0wllm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/shad0wllm?label=pip%20install%20shad0wllm&color=7c5cff&logo=pypi&logoColor=white&cacheSeconds=3600"></a>
  <a href="https://www.npmjs.com/package/shad0wllm"><img alt="npm" src="https://img.shields.io/npm/v/shad0wllm?label=npm%20i%20shad0wllm&color=7c5cff&logo=npm&logoColor=white&cacheSeconds=3600"></a>
  <a href="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/2796gaurav/shad0w/blob/main/LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"></a>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w/#film"><b>▶ Watch the video</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/playground/"><b>Try it in your browser</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/bench/"><b>Decision Bench</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/"><b>Docs</b></a>
</p>

Your app asks an LLM the same small questions all day: which intent, which team, block or allow, which tool next. Each call takes a second or more and is billed. **shad0w** learns from your LLM's answers, proves where it agrees, and answers those repeats itself in microseconds on your CPU. Everything else still goes to your LLM.

## Quick start

```bash
pip install "shad0wllm[compile]"
```

```python
import shad0w

intent = shad0w.decision("intent", llm="openai/gpt-4.1-mini",
    options=["refund", "lost_card", "balance", "other"])

intent("my card was stolen")   # day one: your LLM answers, shad0w logs it
intent.train()                 # after ~1,000 answers: train and certify a small table
intent("my card was stolen")   # now answered by the table when certified
# Decision(answer='lost_card', source='table', certified=True, latency_us=...)
```

**No code change:** run the proxy and move your client's `base_url`.

```bash
shad0w proxy --upstream https://api.openai.com/v1      # dashboard at http://localhost:8010
```

```python
client = OpenAI(base_url="http://localhost:8010/v1")   # the only change
```

Decisions API and System One calls need nothing else. For chat completions, add the header `X-Shad0w-Question: intent`. More in the [5-minute guide](https://2796gaurav.github.io/shad0w/docs/).

## How it works

<p align="center"><img src="https://2796gaurav.github.io/shad0w/assets/how-it-works.svg" alt="watch, train, certify, serve" width="100%"></p>

1. **Watch.** Your LLM answers as usual; each question and answer is logged.
2. **Learn.** The log becomes a small table. No labels, no GPU, seconds to a minute.
3. **Prove.** On answers it never saw, a [Learn-then-Test](https://arxiv.org/abs/2110.01052) check finds where it differs from your LLM on at most 5% of what it answers, with 90% confidence.
4. **Serve.** Above that line the table answers; below it, your LLM does and the table keeps learning.

## Results

<!-- numbers:start -->
| | measured |
|---|---|
| Answered by shad0w instead of the LLM (3 public datasets, hosted LLM teacher) | **61–86%** |
| Disagreement with the LLM on those answers (limit: 5%) | **3.0–6.0%** |
| Time per decision | **2.4 µs** Python, 1.0 µs C, 10 µs JavaScript; a hosted LLM call: 1416 ms |
| Share answered as the log grows (BANKING77) | 0% at 250 answers, 60% at 10,000 |

Sources, hardware and dates for every number: [benchmarks](https://2796gaurav.github.io/shad0w/docs/benchmarks.html).
<!-- numbers:end -->

<details><summary><b>Decision Bench:</b> shad0w against five open decision models on six public tasks</summary>

<!-- bench:start -->
Measured on Apple M5 Pro, 2026-10-19. Decision models run zero-shot on the CPU only with 5 threads; shad0w runs on one thread and learns from each task's training split. Accuracy is against the true labels; certified = share of test rows served under the same 5% Learn-then-Test certificate.

| task | shad0w: accuracy / certified / p50 | most accurate decision model | fastest decision model | shad0w in front of the most accurate |
|---|---|---|---|---|
| ag_news | 91.2% / 66% / 10.1 µs | Laya: 93.8%, 124.4 ms | Julia-1: 82.8%, 50.1 ms | 94.2%, 2.9× faster |
| banking77 | 94.0% / 89% / 6.0 µs | Kev-0.8B: 83.3%, 2.6 s (150 rows) | Julia-1: 62.8%, 197.1 ms | 91.3%, 7.8× faster |
| clinc150 | 84.6% / 86% / 6.5 µs | Decider-0.8B: 83.3%, 5.6 s (150 rows) | Julia-1: 27.4%, 314.1 ms | 91.3%, 7.5× faster |
| prompt_injection | 98.8% / 100% / 2.4 µs | GLiNER2.5-Decide: 94.6%, 112.6 ms | Julia-1: 34.4%, 14.3 ms | 98.8%, 28,988× faster |
| sms_spam | 97.8% / 100% / 1.9 µs | Decider-0.8B: 95.3%, 3.6 s (150 rows) | Julia-1: 66.4%, 15.9 ms | 97.3%, 1,704,195× faster |
| sst2 | 83.2% / 62% / 3.0 µs | Laya: 91.6%, 49.2 ms | Julia-1: 76.8%, 15.5 ms | 92.4%, 2.5× faster |
<!-- bench:end -->

Interactive charts and caveats: [Decision Bench](https://2796gaurav.github.io/shad0w/bench/). Protocol: [bench/README.md](bench/README.md).
</details>

<details><summary><b>How it compares</b> with caches, embedding routers and decision models</summary>

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
</details>

## Good fit, bad fit

| Give it to shad0w | Keep it on your LLM |
|---|---|
| Support intent routing, ticket triage | Writing replies or summaries |
| Moderation and prompt-attack pre-filters | Sarcasm, tone, world knowledge |
| Agent tool routing, repeat web errands | Websites or tasks it has never seen |
| Tagging and topics at ingest | One-off, low-volume calls |

The guarantee is agreement with your LLM, not with the truth. It needs about 1,000 logged answers before it serves anything. Full list: [Limits](https://2796gaurav.github.io/shad0w/docs/limits.html).

## Use it from anywhere

| | Install | Call |
|---|---|---|
| **Python** | `pip install shad0wllm` | `shad0w.decision("intent", llm=..., options=[...])` |
| **JavaScript** (Node, browser, Workers, Deno, Bun) | `npm i shad0wllm` | `await (await decision("intent", {...})).decide(text)` |
| **Proxy** (no code change) | `shad0w proxy --upstream <url>` | `OpenAI(base_url="http://localhost:8010/v1")` |
| **HTTP** (any language) | `shad0w serve --bundle bundle/` | `POST /v1/decide {"state": "my card was stolen"}` |
| **C / FFI** | copy `shad0w.h` + `reflex.c` | `s0_decide(m, text, len, probs, &r)` |
| **LangChain, LiteLLM, Vercel AI SDK** | included | [Frameworks](https://2796gaurav.github.io/shad0w/docs/frameworks.html) |

Built-in providers: OpenAI, Anthropic, Gemini, Groq, Together, OpenRouter, Mistral, DeepSeek, Fireworks, xAI, Ollama, vLLM, LM Studio, the OpenAI Decisions API and System One servers. The runtime needs only numpy, and nothing leaves your machine except the LLM calls you already make.

## Learn more

- **Try it live:** [support inbox](https://2796gaurav.github.io/shad0w/demo/) · [prompt guard](https://2796gaurav.github.io/shad0w/guard/) · [web agent](https://2796gaurav.github.io/shad0w/webagent/) · [all demos](https://2796gaurav.github.io/shad0w/playground/)
- **Docs:** [get started](https://2796gaurav.github.io/shad0w/docs/) · [the guarantee](https://2796gaurav.github.io/shad0w/docs/guarantee.html) · [configuration](https://2796gaurav.github.io/shad0w/docs/configuration.html) · [observability](https://2796gaurav.github.io/shad0w/docs/observability.html) · [experimental features](https://2796gaurav.github.io/shad0w/docs/experimental.html)
- **Contribute:** try it on one of your decisions and [share your numbers](https://github.com/2796gaurav/shad0w/issues/new?template=results.yml), failures included. Code: [CONTRIBUTING.md](https://github.com/2796gaurav/shad0w/blob/main/CONTRIBUTING.md).

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

Apache-2.0. Built by <b>Gaurav Chauhan</b> · [LinkedIn](https://www.linkedin.com/in/gauravc2708/) · [GitHub](https://github.com/2796gaurav) · [gauravc2708@gmail.com](mailto:gauravc2708@gmail.com)
