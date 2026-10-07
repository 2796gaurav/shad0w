<p align="center">
  <a href="https://2796gaurav.github.io/shad0w">
    <img src="https://2796gaurav.github.io/shad0w/assets/hero.svg" alt="shad0w: your LLM's decisions, compiled into a certified table that answers in microseconds" width="100%">
  </a>
</p>

<h1 align="center">shad0w</h1>

<p align="center">
  <b>Compile the decisions your LLM already makes into a certified 2&nbsp;MB table that answers in microseconds on a CPU.</b><br>
  It learns from your model's own answers, needs no labels, and tells you in writing how often it will disagree.
</p>

<p align="center">
  <a href="https://pypi.org/project/shad0w/"><img alt="PyPI" src="https://img.shields.io/pypi/v/shad0w?color=7c5cff&label=pypi"></a>
  <a href="https://www.npmjs.com/package/shad0w"><img alt="npm" src="https://img.shields.io/npm/v/shad0w?color=7c5cff&label=npm"></a>
  <a href="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/2796gaurav/shad0w/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776ab">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2ea44f"></a>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w/demo/"><b>Live demo</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/"><b>Docs</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/#video"><b>2-minute video</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/benchmarks.html"><b>Benchmarks</b></a>
</p>

---

Your app sends small, typed decisions to a big model millions of times: *which intent is this, which queue gets this ticket, is this spam, should this agent step run*. Every call takes 20–1,500 ms and a round trip to someone else's GPU.

**shad0w** sits in front of that model like a shadow. It watches the answers, learns the decision, and compiles it into a tiny integer table. Before the table answers anything, it **certifies how often it will disagree with your model**. On public intent and topic benchmarks, it then served **45–71%** of traffic itself, at under 5% certified disagreement and in microseconds. Everything else goes to your model, exactly as before.

```python
import shad0w

sh = shad0w.Shadow("bundle/", teacher=ask_my_llm, log="teacher_log.jsonl")

sh.decide("my card was stolen yesterday")
# Decision(answer='lost_card', source='table', confidence=0.98, certified=True, latency_us=14)
```

<table>
<tr>
<td align="center"><b>~1 µs</b><br><sub>per decision in C<br>7–11 µs JS · 15 µs Python</sub></td>
<td align="center"><b>45–71%</b><br><sub>of LLM traffic served locally<br>at a certified 5% bound</sub></td>
<td align="center"><b>0 labels</b><br><sub>learns from your model's<br>own logged answers</sub></td>
<td align="center"><b>2 MB</b><br><sub>one file, no GPU, no network<br>256 KB still keeps 88.5%</sub></td>
</tr>
</table>

## How it works

<p align="center">
  <img src="https://2796gaurav.github.io/shad0w/assets/how-it-works.svg" alt="shadow → compile → certify → serve" width="100%">
</p>

1. **Shadow.** Your model keeps serving. Its answers are logged as `(text, answer)` pairs, either by the `Shadow` wrapper or by your own logging.
2. **Compile.** The log trains a hashed n‑gram table with int8 weights. There is no tokenizer and no GPU, and it takes seconds to minutes on a laptop.
3. **Certify.** On a held-out slice, a [Learn‑then‑Test](https://arxiv.org/abs/2110.01052) procedure picks the confidence threshold. Among answers the table serves, disagreement with your model is ≤ α (say 5%), with probability ≥ 1 − δ.
4. **Serve.** Certified answers come back in microseconds, in your process. Everything else goes to your model and is logged, so the share the table can take grows with traffic: **31% → 70%** as AG News shadow traffic went from 2k to 100k.

## Results

Measured on public datasets. Every number links to its raw result file on the [benchmarks page](https://2796gaurav.github.io/shad0w/docs/benchmarks.html).

| | Result |
|---|---|
| LLM traffic answered locally (7 intent/topic tasks, 2 teachers), certified α = 5% | **45–71%**, realised disagreement 3.9–5.7% (mean of 3 seeds) |
| End-to-end accuracy (table + teacher) vs teacher alone | within **1 point** on every task; **+0.7** with Kev‑0.8B as teacher |
| Mean latency of the cascade, Kev‑0.8B → BANKING77 | **75 ms → 22 ms** |
| Offload as shadow traffic grows (AG News, 2k → 100k decisions) | **31% → 70%** at α = 5%; **5% → 52%** at 2% |
| Per decision | **~1 µs** C · **7–11 µs** JavaScript · **15 µs** Python · 0.19 ms over HTTP |
| Public benchmark (local‑jev‑bench, BANKING77, 20 options) | **89.7% at 0.018 ms on 1 CPU thread**; Kev‑0.8B 88% at 57.6 ms; Clef‑flash (9B) 97% at 1,274 ms |
| Out-of-scope queries (CLINC150) separated by confidence | AUROC **0.94** |
| **Reproduce on your laptop** (`examples/shadow_demo.py`, MIT-licensed teacher, BANKING77) | **69.4%** answered locally at α = 5% (realised 4.6%); cascade 67.8% vs teacher 68.0% |

## Compared with what you would otherwise do

| | Per decision | Runs on | Needs | Bound on what it serves |
|---|---|---|---|---|
| Keep calling the LLM / decision API | 20–1,500 ms | GPU or vendor API | — | it *is* the reference |
| Semantic cache (GPTCache, gateway caches) | ms (embed + vector search) | embedding model + store | nothing | similarity threshold, no bound |
| Embedding router (e.g. semantic-router) | ms | embedding model | example utterances | threshold, no bound |
| Distil into a smaller LLM | 10–100 ms | GPU | teacher outputs | none |
| Hand-built fastText / SetFit / Model2Vec | µs–ms | CPU | **human labels** | none |
| **shad0w** | **1–15 µs** | **CPU, in-process, browser, edge** | **your model's logged answers** | **finite-sample, against your model** |

It is not new science. The certificate is Learn‑then‑Test, in the spirit of [Trust or Escalate](https://arxiv.org/abs/2407.18370) and [BARGAIN](https://arxiv.org/abs/2509.02896). The table is [fastText](https://arxiv.org/abs/1607.01759)-shaped on purpose. What is new is the packaging: one workflow from shadow logs to a certified, KB-to-MB artifact, with the same decisions in C, Python and JavaScript.

## Quick start

```bash
pip install "shad0w[compile]"        # the runtime alone is: pip install shad0w   (numpy only, C core included)
```

**1. Log your model's answers.** Wrap the call you already make. With no bundle yet, every call goes to your model and is logged:

```python
import shad0w

sh = shad0w.Shadow(None, teacher=ask_my_llm, question="intent", log="teacher_log.jsonl")
answer = sh.decide(text).answer          # = ask_my_llm(text), logged
```

**2. Compile and certify** once you have 1,000+ lines (10k–100k for high offload):

```bash
shad0w init                                        # starter schema.json: the options your model chooses from
shad0w shadow --schema schema.json --data teacher_log.jsonl --teacher my-llm-v3 --alpha 0.05 --out bundle/
shad0w report --bundle bundle/                      # the certificate, in plain words and numbers
```

**3. Serve.** Point the wrapper at the bundle. Certified answers come from the table and the rest from your model. 1% of certified answers are spot-checked against your model, so `sh.stats()` shows live disagreement:

```python
sh = shad0w.Shadow("bundle/", teacher=ask_my_llm, log="teacher_log.jsonl")
```

**4. Re-certify** on fresh traffic every week, or when traffic changes:

```bash
shad0w certify --bundle bundle/ --data last_week.jsonl
```

### Everywhere else

| | |
|---|---|
| **JavaScript** (Node, browsers, Workers, Deno, Bun), zero dependencies | `npm i shad0w` → `const b = await Bundle.load("bundle/"); b.decide(text)` |
| **HTTP** sidecar, any language | `shad0w serve --bundle bundle/` → `POST /v1/decide` |
| **C**, any language with an FFI | `shad0w/_native/reflex.c`: `s0_load`, `s0_decide` |
| **Decorator** | `@shad0w.cascade("bundle/")` on your existing classify function |
| **Human labels** instead of a teacher | `shad0w compile --data labels.jsonl` + `shad0w calibrate` (~300 labels) |
| **No teacher, no labels** | `shad0w compile --unlabeled logs.txt` (option names + raw logs, `pip install "shad0w[logs]"`) |

Integration recipes for FastAPI, LiteLLM, LangGraph, Cloudflare Workers and the browser are in the [docs](https://2796gaurav.github.io/shad0w/docs/integrations.html).

## When not to use it

The limits below are measured. Knowing them is what makes the certificate useful.

- **Sentiment, tone, emotion.** A hashed table tops out near 79% on SST‑2 even with perfect labels. shad0w certifies almost nothing there and defers everything to your model, which is the correct behaviour.
- **When you need the most accurate answer.** A 9B decision model is 7 points better on the public BANKING77 file, at ~70,000× the latency. Put shad0w *in front of* such a model, not instead of it.
- **The bound is against your model, not the truth.** A certified table in front of a wrong model is wrong in the same way. For a bound against the truth, calibrate with ~300 real labels.
- **The bound holds on traffic like the calibration slice.** A big shift in the request mix can break it. The drift guard flags confidence shifts, the audits watch live disagreement, and `shad0w certify` re-certifies.
- **Reasoning or world knowledge.** It learns *surface* decisions: intents, routes, topics, policies with distinctive wording.

## Documentation

[Getting started](https://2796gaurav.github.io/shad0w/docs/) · [Shadow mode guide](https://2796gaurav.github.io/shad0w/docs/shadow-mode.html) · [Integrations](https://2796gaurav.github.io/shad0w/docs/integrations.html) · [The certificate, explained](https://2796gaurav.github.io/shad0w/docs/certificate.html) · [API & CLI](https://2796gaurav.github.io/shad0w/docs/reference.html) · [Architecture & file format](https://2796gaurav.github.io/shad0w/docs/architecture.html) · [Benchmarks](https://2796gaurav.github.io/shad0w/docs/benchmarks.html) · [Limitations](https://2796gaurav.github.io/shad0w/docs/limitations.html) · [FAQ](https://2796gaurav.github.io/shad0w/docs/faq.html)

## Contributing

Run it on one of your decisions and [share your numbers](https://github.com/2796gaurav/shad0w/issues/new?template=results.yml), including failures. That is the most valuable contribution. Code contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{chauhan2026shad0w,
  author = {Chauhan, Gaurav},
  title  = {shad0w: certified microsecond decisions compiled from the model you already run},
  year   = {2026},
  url    = {https://github.com/2796gaurav/shad0w}
}
```

---

<p align="center">Built by <a href="https://github.com/2796gaurav">Gaurav Chauhan</a> · Apache-2.0</p>
