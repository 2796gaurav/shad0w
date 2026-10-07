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
  <img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776ab?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"></a>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w"><b>Website</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/demo/"><b>Try it in your browser</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/"><b>Docs</b></a>
</p>

---

## Quick start

```bash
pip install "shad0wllm[compile]"
```

```python
import shad0w

intent = shad0w.decision(
    "intent",
    options={"refund": "wants money back", "lost_card": "card lost or stolen", "balance": "asks about balance"},
    llm="openai/gpt-4o-mini",   # or anthropic/…, gemini/…, groq/…, ollama/llama3.1, any OpenAI-compatible server
)

intent("my card was stolen")    # day one: your LLM answers (~500 ms) and shad0w writes the answer down
intent.train()                  # after ~1,000 answers: builds and certifies a small table, in about a minute or less
intent("my card was stolen")    # from then on: answered locally when certified
# Decision(answer='lost_card', source='table', confidence=0.99, certified=True, flag=None, latency_us=9.6)
```

That is the whole integration. Anything the table is unsure about still goes to your LLM and gets logged, so the table keeps learning.

**Prefer to change no code?** Run the proxy and point your OpenAI client at it:

```bash
shad0w proxy --upstream https://api.openai.com/v1        # dashboard at http://localhost:8010
```

```python
client = OpenAI(base_url="http://localhost:8010/v1")      # the only change
client.chat.completions.create(model="gpt-4o-mini", messages=msgs,
                               extra_headers={"X-Shad0w-Question": "intent"})   # mark which calls are decisions
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
| Typical decision | 200–1,500 ms LLM call | **~10 µs** for the share the table is certified on |
| Share answered without the LLM | 0% | **45–71%** in our tests, and it grows as more traffic is logged |
| Agreement with your LLM | — | disagrees on at most **α = 5%** of the answers it serves, **certified** before it serves anything |
| Runs on | someone's GPU | your CPU, in-process; also in the browser and on edge workers |
| Your code | `ask_llm(text)` | `intent(text)` |

**Not a fit:** open-ended generation, questions that need reasoning or world knowledge, and labels that depend on tone or sarcasm (sentiment tops out near 79%; shad0w then certifies almost nothing and passes everything to your LLM, which is the correct behaviour).

## How it works

1. **Watch.** Your LLM keeps answering. Each `(text, answer)` pair is written to a log.
2. **Train.** The log becomes a hashed word/character table with 8-bit weights. There is no GPU, no tokenizer and no human labels, and it takes seconds to a minute on a laptop.
3. **Certify.** On answers the table never trained on, a [Learn-then-Test](https://arxiv.org/abs/2110.01052) procedure picks a confidence threshold. Above it, the table disagrees with your LLM at most α of the time, with 90% confidence.
4. **Serve.** Above the threshold the table answers in microseconds. Below it, your LLM answers and the answer is logged. 1% of table answers are spot-checked against the LLM, so you can see live agreement, not only the certificate.

<p align="center"><img src="https://2796gaurav.github.io/shad0w/assets/how-it-works.svg" alt="watch, train, certify, serve" width="100%"></p>

## Use it from anywhere

| | Install | Three lines |
|---|---|---|
| **Python** | `pip install shad0wllm` | `sh = shad0w.decision("intent", llm="openai/gpt-4o-mini", options=[...])` → `sh(text).answer` |
| **Any OpenAI client** (zero code) | `shad0w proxy --upstream <url>` | `OpenAI(base_url="http://localhost:8010/v1")` plus the header `X-Shad0w-Question` |
| **JavaScript / TypeScript** (Node, browser, Workers, Deno, Bun) | `npm i shad0wllm` | `const intent = await decision("intent", {options, llm: "openai/gpt-4o-mini", bundle: "bundle/"})` → `await intent.decide(text)` |
| **HTTP** (any language) | `shad0w serve --bundle bundle/` | `POST /v1/decide {"state": "my card was stolen"}` |
| **C / C++ / FFI** | copy `shad0w.h` + `reflex.c` | `s0_load("intent.s0")` → `s0_decide(m, text, len, probs, &r)` |
| **LangChain, LiteLLM, anything else** | — | `shad0w.llm_teacher(options, complete=lambda msgs: llm.invoke(msgs).content)` |
| **Your own function** | — | `shad0w.Shadow("bundle/", teacher=my_classify, log="log.jsonl")` |

Providers built in: `openai`, `anthropic`, `gemini`, `groq`, `together`, `openrouter`, `mistral`, `deepseek`, `fireworks`, `xai`, `azure`, `ollama`, `vllm` and `lmstudio`. Keys come from the usual environment variables (`OPENAI_API_KEY`, …). Any other OpenAI-compatible server works with `base_url=`. Recipes for each are in the [docs](https://2796gaurav.github.io/shad0w/docs/).

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

Measured on public datasets, with the certificate at α = 5%:

- **45–71%** of LLM traffic answered by the table on 7 intent and topic tasks, with 3.9–5.7% realised disagreement. End-to-end accuracy stayed within 1 point of the LLM alone.
- **~1 µs** per decision in C, **7–11 µs** in JavaScript, **~10–15 µs** in Python. With a 0.8B LLM teacher, mean latency fell from **75 ms to 22 ms**.
- The share grows with traffic: **31% → 70%** as logged answers went from 2k to 100k (AG News).
- Reproduce it on a laptop with a free teacher model: `python examples/shadow_demo.py` gives 69.4% answered locally on BANKING77, 4.6% realised disagreement.

## How it compares

| | Per decision | Needs | Guarantee on what it serves |
|---|---|---|---|
| Keep calling the LLM | 200–1,500 ms | GPU or API bill | it *is* the reference |
| Semantic cache | ms | embedding model + vector store | similarity threshold only |
| Embedding router | ms | embedding model, example phrases | threshold only |
| Fine-tune a small model | 10–100 ms | GPU, training pipeline | none |
| fastText / SetFit by hand | µs–ms | **human labels** | none |
| **shad0w** | **~1–15 µs** | **your LLM's own answers** | **certified bound vs your LLM** |

This is not new science. The certificate is Learn-then-Test, in the spirit of [Trust or Escalate](https://arxiv.org/abs/2407.18370) and [BARGAIN](https://arxiv.org/abs/2509.02896), and the table is [fastText](https://arxiv.org/abs/1607.01759)-shaped on purpose. What shad0w adds is the packaging: one line from your existing LLM call to a certified, megabyte-sized table, with the same answers in Python, JavaScript and C.

## Good to know

- **The guarantee is against your LLM, not the truth.** If your LLM is wrong, the table is wrong the same way. For a bound against the truth, calibrate on ~300 human labels (`shad0w calibrate`).
- **It holds for traffic like the traffic it was certified on.** Re-train when your traffic changes. The drift guard and live spot checks show you when that is.
- **Runtime installs only numpy**; the C core is inside the wheel. Training (`[compile]`) adds scipy, scikit-learn and torch. For a smaller download, install CPU-only torch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`); without torch, training falls back to scipy, which is slower.
- **Nothing leaves your machine** except the LLM calls you already make.

## Contributing

Try it on one of your decisions and [share your numbers](https://github.com/2796gaurav/shad0w/issues/new?template=results.yml), failures included. That is the most useful contribution. For code, see [CONTRIBUTING.md](CONTRIBUTING.md).

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
