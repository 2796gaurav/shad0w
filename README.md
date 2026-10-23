<p align="center">
  <a href="https://2796gaurav.github.io/shad0w">
    <img src="https://2796gaurav.github.io/shad0w/assets/hero.svg" alt="Requests race in. The shad0w table answers most of them in microseconds; the hard ones still go to your LLM." width="100%">
  </a>
</p>

<h1 align="center">shad0w</h1>

<p align="center">
  <b>Your app asks an LLM the same small questions all day: which intent, which team, is this an attack, what is in this photo.<br>
  shad0w learns its answers and serves them in microseconds, with a written guarantee on how often it disagrees.</b>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w/#film">▶ Watch shad0w at work, in two minutes</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/shad0wllm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/shad0wllm?label=pip%20install%20shad0wllm&color=7c5cff&logo=pypi&logoColor=white&cacheSeconds=3600"></a>
  <a href="https://www.npmjs.com/package/shad0wllm"><img alt="npm" src="https://img.shields.io/npm/v/shad0wllm?label=npm%20i%20shad0wllm&color=7c5cff&logo=npm&logoColor=white&cacheSeconds=3600"></a>
  <a href="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10–3.14" src="https://img.shields.io/badge/python-3.10%E2%80%933.14-3776ab?logo=python&logoColor=white">
  <a href="https://github.com/2796gaurav/shad0w/blob/main/LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"></a>
  <a href="https://doi.org/10.5281/zenodo.23598813"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23598813.svg"></a>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w"><b>Website</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/playground/"><b>Playground</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/bench/"><b>Decision Bench</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/"><b>Docs</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/blog/"><b>Blog</b></a>
</p>

A hosted LLM takes about 1.4 s to pick one answer from a list. Most of those questions are repeats it has answered many times. shad0w watches your LLM, learns a small table from its answers, and **proves on answers it never saw** where it agrees with it. Above that line the table answers in microseconds on your CPU; below it, your LLM answers as before, and every answer it gives is more to learn from.

| Job | What shad0w decides | Try it |
|---|---|---|
| **Support routing** | the intent or team for each message, 2–10 µs | [type a message](https://2796gaurav.github.io/shad0w/demo/) |
| **Guardrails** | blocks known prompt attacks before your LLM reads them; never allows on its own word | [prompt guard](https://2796gaurav.github.io/shad0w/guard/) |
| **Product photos** | a category per image from pixels, no neural network, 4–7 µs in C (experimental) | [Doodle Duel](https://2796gaurav.github.io/shad0w/doodle/) |
| **Live streams** | a section and topic for every event as it arrives | [a minute of Polymarket](https://2796gaurav.github.io/shad0w/pulse/) |
| **Decision models** | the questions it is certified on, for the OpenAI Decisions API or System One models, with zero code | [Decision Bench](https://2796gaurav.github.io/shad0w/bench/) |

## Quick start

```bash
pip install "shad0wllm[compile]"          # JavaScript: npm i shad0wllm
```

**Zero code: put the proxy in front of your LLM and change one URL.**

```bash
shad0w proxy --upstream https://api.openai.com/v1      # dashboard at http://localhost:8010
```

```python
client = OpenAI(base_url="http://localhost:8010/v1")   # the only change
client.decisions.create(model="gpt-6-luna", input=text, questions=[...])          # decision models: nothing to mark
client.chat.completions.create(model="gpt-6-luna", messages=msgs,
                               extra_headers={"X-Shad0w-Question": "intent"})      # chat: mark the call
```

Certified questions come back from the table in the same JSON shape; the rest go upstream and are logged, so the table keeps learning. Unmarked calls pass straight through.

**One line of Python.**

```python
import shad0w

intent = shad0w.decision(
    "intent",
    options={"refund": "wants money back", "lost_card": "card lost or stolen", "other": "anything else"},
    llm="openai/gpt-6-luna",      # or anthropic/…, gemini/…, groq/…, ollama/…, systemone/kev, any OpenAI-compatible base_url
)

intent("my card was stolen")      # day one: your LLM answers; shad0w writes it down
intent.train()                    # after ~1,000 answers: a small table and its certificate, in about a minute
intent("my card was stolen")      # now: Decision(answer='lost_card', source='table', certified=True, ...)
```

**JavaScript** (Node, browsers, Workers, Deno, Bun; zero dependencies):

```js
import { decision } from "shad0wllm";
const intent = await decision("intent", { options: ["refund", "lost_card", "other"], llm: "openai/gpt-6-luna", bundle: "shad0w/intent/bundle" });
await intent.decide("my card was stolen");   // { answer: "lost_card", source: "table", certified: true, ... }
```

Starting fresh? `shad0w init` writes a commented `shad0w.toml` and a runnable `app.py`; `shad0w status` tells you what each decision needs next; `shad0w doctor` checks your setup.

## How it works

<p align="center"><img src="https://2796gaurav.github.io/shad0w/assets/how-it-works.svg" alt="watch, train, certify, serve" width="100%"></p>

1. **Watch.** Your LLM keeps answering. Each `(text, answer)` pair is logged.
2. **Learn.** About 1,000 answers become a hashed word table with small integer weights. No GPU, no labels.
3. **Prove.** On answers it never trained on, [Learn-then-Test](https://arxiv.org/abs/2110.01052) picks a confidence line. Above it the table disagrees with your LLM on at most **α = 5%** of what it answers, with 90% confidence.
4. **Serve.** Above the line the table answers; below it your LLM does. Live spot checks and a drift guard step it back when traffic changes. Roll out in stages: shadow → canary → serve.

## Results

Measured with a hosted LLM through the official OpenAI client, on public datasets ([every number, source and machine](https://2796gaurav.github.io/shad0w/docs/benchmarks.html)):

| | |
|---|---|
| Time per answer | **2.4 µs** in Python (C core inside), 1.0 µs in C, 10 µs in JavaScript; a hosted LLM call: 1416 ms |
| Answered without the LLM | **61–86%** of messages on 3 public intent and topic datasets; **94.3%** on 8 support intents |
| Disagreement with the LLM on those | **3.0–6.0%** realised, against a 5% bound |
| Against open decision models, CPU only | the most accurate on **4 of 6** tasks, at 2–10 µs against 14 ms–5.6 s ([Decision Bench](https://2796gaurav.github.io/shad0w/bench/)) |
| Where it loses | sentiment and legal clauses with the linear table; it defers rather than guess ([limits](https://2796gaurav.github.io/shad0w/docs/limits.html)) |

## Is it for me?

**Yes** if your app asks an LLM to pick one answer from a fixed list, thousands of times: intents, ticket routing, spam and moderation, agent tool choice, tagging, image categories. **No** for generated text (replies, summaries) or questions that need reasoning or world knowledge: those stay with your LLM, which is where shad0w sends them.

The guarantee is against your LLM, not the truth: if your LLM is wrong, the table is wrong the same way. Calibrate on a few hundred human labels for a bound against the truth.

## Use it from anywhere

Python · JavaScript/TypeScript · HTTP (`shad0w serve`) · C/FFI · the proxy for any OpenAI client · LangChain, LiteLLM and the Vercel AI SDK. Providers built in: OpenAI, Anthropic, Gemini, Groq, Together, OpenRouter, Mistral, DeepSeek, Fireworks, xAI, Ollama, vLLM, LM Studio, the OpenAI Decisions API and System One. The runtime needs only numpy; nothing leaves your machine except the LLM calls you already make.

## Learn more

- **Start:** [5-minute guide](https://2796gaurav.github.io/shad0w/docs/) · [Connect your LLM](https://2796gaurav.github.io/shad0w/docs/connect-your-llm.html) · [The proxy](https://2796gaurav.github.io/shad0w/docs/proxy.html) · [Decisions API & System One](https://2796gaurav.github.io/shad0w/docs/decisions-api.html)
- **Run it:** [Configuration](https://2796gaurav.github.io/shad0w/docs/configuration.html) · [Rollout](https://2796gaurav.github.io/shad0w/docs/rollout.html) · [Monitoring](https://2796gaurav.github.io/shad0w/docs/observability.html) · [Training](https://2796gaurav.github.io/shad0w/docs/training.html) · [Changing your options](https://2796gaurav.github.io/shad0w/docs/options.html)
- **Go further:** [The deep table](https://2796gaurav.github.io/shad0w/docs/deep-table.html) · [Images, guard, video, extraction](https://2796gaurav.github.io/shad0w/docs/experimental.html) · [Running without an LLM](https://2796gaurav.github.io/shad0w/docs/without-llm.html)
- **Understand:** [The guarantee](https://2796gaurav.github.io/shad0w/docs/guarantee.html) · [How it compares](https://2796gaurav.github.io/shad0w/docs/compare.html) · [FAQ](https://2796gaurav.github.io/shad0w/docs/faq.html) · [Reference](https://2796gaurav.github.io/shad0w/docs/reference.html) · [Changelog](CHANGELOG.md)

## Contributing

Try it on one of your decisions and [share your numbers](https://github.com/2796gaurav/shad0w/issues/new?template=results.yml), failures included: that is the most useful contribution. For code, see [CONTRIBUTING.md](CONTRIBUTING.md). Benchmarks reproduce from [`bench/`](bench/README.md).

<details><summary>Cite</summary>

```bibtex
@misc{chauhan2026certified,
  author = {Chauhan, Gaurav},
  title  = {Certified Microsecond Answers for an {LLM}'s Repeated Fixed-Choice Decisions},
  year   = {2026},
  doi    = {10.5281/zenodo.23599000},
  note   = {Preprint}
}

@software{chauhan2026shad0w,
  author  = {Chauhan, Gaurav},
  title   = {shad0w: certified microsecond decisions compiled from the model you already run},
  version = {0.4.0},
  year    = {2026},
  doi     = {10.5281/zenodo.23598814},
  url     = {https://github.com/2796gaurav/shad0w}
}
```
</details>

<p align="center">Built by <b>Gaurav Chauhan</b> · <a href="https://www.linkedin.com/in/gauravc2708/">LinkedIn</a> · <a href="https://github.com/2796gaurav">GitHub</a> · <a href="mailto:gauravc2708@gmail.com">gauravc2708@gmail.com</a> · Apache-2.0</p>
