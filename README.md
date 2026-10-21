<p align="center">
  <a href="https://2796gaurav.github.io/shad0w">
    <img src="https://2796gaurav.github.io/shad0w/assets/hero.svg" alt="Requests race in. The shad0w table answers most of them in microseconds; the hard ones still go to your LLM." width="100%">
  </a>
</p>

<h1 align="center">shad0w</h1>

<p align="center">
  <b>Your app asks a model the same kinds of small questions all day: about text, images and video frames.<br>
  shad0w learns its answers and serves them in microseconds, with a written guarantee on how often it disagrees.</b>
</p>

<p align="center">
  <a href="https://2796gaurav.github.io/shad0w/#film">▶ Watch how it works, in two minutes</a> · <a href="https://github.com/2796gaurav/shad0w/releases/latest">download the films (2 min and 30 s, 16:9 and 9:16)</a>
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
  <a href="https://2796gaurav.github.io/shad0w/playground/"><b>Playground</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/bench/"><b>Decision Bench</b></a> ·
  <a href="https://2796gaurav.github.io/shad0w/docs/"><b>Docs</b></a>
</p>

| | What it decides | Try it |
|---|---|---|
| **Text** | intents, routing, topics, spam, sentiment (deep table), 2–10 µs on one CPU thread | [type a message](https://2796gaurav.github.io/shad0w/demo/) · [a live minute of Polymarket](https://2796gaurav.github.io/shad0w/pulse/) |
| **Images** | a label per image from pixels, no neural network, 4–7 µs in C | [Doodle Duel: draw, it guesses on every stroke](https://2796gaurav.github.io/shad0w/doodle/) |
| **Guardrails** | blocks known prompt attacks before your LLM reads them; never allows on its own word | [prompt guard](https://2796gaurav.github.io/shad0w/guard/) |
| **Video frames** | reuses a certified answer while the scene holds (experimental) | [docs](https://2796gaurav.github.io/shad0w/docs/experimental.html#video-frames) |
| **Structured fields** | fills the fields it is sure of, asks your LLM for the rest (experimental) | [docs](https://2796gaurav.github.io/shad0w/docs/experimental.html#structured-extraction) |
| **Against decision models** | six public tasks, CPU only: 2–10 µs against 14 ms–5.6 s, most accurate on 4 of 6 | [Decision Bench](https://2796gaurav.github.io/shad0w/bench/) |

And for fun, at game speed: [a CS-style duel](https://2796gaurav.github.io/shad0w/duel/), [Doom](https://2796gaurav.github.io/shad0w/doom/), [Pong](https://2796gaurav.github.io/shad0w/pong/) and [a snake arena](https://2796gaurav.github.io/shad0w/arena/).

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

Unmarked calls (chat, embeddings, streaming) pass straight through. Running the proxy on a network (a container, `--host 0.0.0.0`)? Set `SHAD0W_PROXY_TOKEN` so only your apps can use it: [Access token](https://2796gaurav.github.io/shad0w/docs/proxy.html#access).

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

**Not a fit:** open-ended generation and questions that need reasoning or world knowledge. Labels that depend on tone or negation (sentiment) used to be out of reach; the deep table below now certifies most of SST-2. When the table cannot certify something, it passes it to your LLM, which is the correct behaviour.

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

## The deep table: interactions between words, same microseconds

A linear table adds one score per word, so "not good" can never mean the opposite of "good". The **deep table** keeps
the same lookups but makes each row a hidden vector (32 to 128 wide, from the number of options), then applies a ReLU
and one small dense layer. It is file format 2, read by the C core, numpy and JavaScript, which compute the same scores bit for bit.
Training picks it only when it would certify more held-out fitting rows (`student="auto"`, the default). Otherwise
you keep the linear table and its exact robustness radius.

Measured through the full pipeline at α = 5% (labels as the teacher; the guard set is labelled by gpt-4.1-mini):

| task | linear table: certified | deep table: certified | disagreement on served | deep table | C p50 / p99 |
|---|---|---|---|---|---|
| SST-2 sentiment (32k rows) | 0% | **74%** | 4.3% | H = 32, 2.7 MB | 1 / 5 µs |
| AG News topics (29k rows) | 81% | **83%** | 4.5% | H = 32, 5.7 MB | 6 / 13 µs |
| BANKING77 intents (77 options) | 81% | **87%** | 3.0% | H = 128, 3.1 MB | 4 / 13 µs |
| prompt-injection set (gpt-4.1-mini labels) | 97.0% | 98.1% | 3.3% | auto kept linear | |

- **Cost.** A deep row is H bytes instead of K, so long texts with many items cost more: about 1–4 µs on short
  texts and 6 µs on news articles in C.
- **Size.** `max_mb` caps a deep table by keeping its most frequent items, but that costs coverage: SST-2 fell from
  74% to 58.5% at 0.7 MB.
- **No exact radius.** A deep table reports none, so exposed mode defers it and `Guard` never allows on one. Set
  `student = "linear"` where you need the radius.
- **Training is faster.** Linear tables now take their temperature from a held-out slice: 2 fits instead of 6 (AG News
  218 s → 100 s, BANKING77 140 s → 65 s, same or better coverage). The L-BFGS backend is chosen by problem size.

Related additions:
- `Shadow.distil(texts)` asks your LLM about a corpus you supply and logs the answers as fit-only rows.
- `static="potion-8M"` now folds the word vectors into the table rows exactly, so serving needs no extra file. It
  helped a small SST-2 log (4k rows: 0% → 15% certified) and hurt full logs (BANKING77 84% → 79%), so it stays opt-in.
- `max_unseen=0.3` defers inputs whose words the table has mostly never seen, such as junk or another language.
  Every answer now reports `unseen`.
- **Script-aware text** (`text_kind` 2, the default for new bundles) gives every Chinese and Japanese character its own
  item, plus character bigrams, and case-folds Latin-1, Greek and Cyrillic. ASCII hashes exactly as before, and old
  bundles keep kind 1. Japanese intents (MASSIVE-ja): 30.7% → 52.0% certified. Japanese news: 93.0% → 95.3%.

## Images (experimental)

Asking a vision model to pick one label for an image? The same loop works on pixels. Tier 0 uses no neural network:
a 32×32 thumbnail, 256 random pixel-comparison ferns, a coarse colour grid and a 64-bit difference hash become item
ids for the same table and certificate. An optional tier 1 quantises a frozen image encoder's embedding (CLIP,
SigLIP, or your own via `register_embedder`) into items. **One certificate covers the whole cascade**: among answers
any tier serves, disagreement with your vision model is at most α.

```python
from shad0w.vision import ImageSpec, EmbedSpec
photo = shad0w.decision("category", options=[...], llm="openai/gpt-4.1-mini", image=True)            # ferns only
photo = shad0w.decision("category", options=[...], llm="openai/gpt-4.1-mini",
                        image=[ImageSpec(), EmbedSpec("openai/clip-vit-base-patch32")])              # a cascade
photo(pixels)      # numpy array or PIL image; your model gets the image, the log gets item ids (+ the embedding)
photo.train()
```

| measured on one CPU core, α = 5%, unseen images | answered locally | disagreement on those | cost per image |
|---|---|---|---|
| product photos (Fashion-MNIST, 9k logged answers), ferns | **64%** | 3.1% | **4–7 µs** pixels → answer (C, any size up to 1080p) |
| natural photos (CIFAR-10), ferns | 0% (correctly defers everything) | — | 4–7 µs |
| natural photos (CIFAR-10), ferns → CLIP B/32 table, teacher gemma4 (2.6 s) | **73%** (all by the CLIP tier) | 5.5% (95% CI ≈ 3.6–8%) | ~28 ms instead of 2.6 s |

Microseconds assume pixels already in memory: decoding a JPEG or PNG costs 0.3–3 ms. The proxy decides marked
requests that carry one inline `data:` image when the question has an image bundle (or `shad0w proxy --images`);
remote image URLs are never fetched. C API: `s0_decide_image` in `shad0w.h`. Reproduce:
`python examples/vision_bench.py --dataset fashion_mnist --channels 1` (add `--labels <vlm answers> --embed <model>`
for the cascade). Not shipped, measured: k-means visual words lift CIFAR-10 agreement from 38% to 56% but still
certify 0% at α = 5%, at 20–50× the cost.

## Guardrails (experimental): block in microseconds, never allow on the table's word

`shad0w.guard.Guard` puts a learned table in front of your LLM as a prompt-injection / jailbreak gate:

```python
from shad0w.guard import Guard
gate = Guard(shad0w.decision("injection", options={"attack": "...", "safe": "..."}, llm="openai/gpt-4.1-mini"))
gate.calibrate(fresh_texts, llm_answers)          # a BLOCK-ONLY certificate: among inputs it blocks, disagreement <= alpha
v = gate.check(text)                              # microseconds: "block" or "ask_model", never "allow"
```

The input is normalised first (Unicode forms, zero-width, bidi and tag characters, homoglyphs, leetspeak, spaced letters,
base64 and hex payloads). Anything the normaliser had to undo goes to your model. On ~9,400 texts labelled by
gpt-4.1-mini (with the provider's own content-filter refusals counted as attacks), held out:

| | measured |
|---|---|
| blocked by the table, without an LLM call | 14% of all traffic, 57% of the attacks |
| blocks the model called safe | 2.6% (block-only certificate at α = 5%) |
| smuggled variants of blocked attacks (zero-width, homoglyphs, leetspeak, full-width, tags, base64) allowed | **0%**; all go to the model. A naive table that may also allow called 94–100% of them safe |
| benign inputs sent to the model because they looked obfuscated | 1.0% |
| latency, normaliser + table (Python) | ~155 µs median, against ~1.1 s for the model |

It catches the repeats of known attacks cheaply; it is not a defence against an adaptive attacker on its own, which is why
it never allows.

**Streaming replies** (`StreamGuard`): checks a reply as it streams, at 16, 32, 64, 128, 256 words and then every
256, and blocks on the first certified prefix. The certificate is per reply: among replies it blocks, the teacher
called the full reply safe at most α of the time. Honest result on BeaverTails: a word-n-gram table certifies nothing
at α = 5%. At α = 10%, with human labels as the teacher, it blocks 50% of unsafe replies after 47% of their words and
over-blocks 4.8% of safe ones. Use it as an early cut-off in front of a real guard model.

**Doodle Duel** (`examples/doodle_live.py`): a sketch game where a table learned from gpt-4.1-mini guesses the doodle on every
stroke, 25 µs per guess in the browser, certifying 41.5% of held-out doodles at 4.8% disagreement (every disagreement on
those was the model's own mistake against the drawing's true word). A version you can play in the browser, learned from Quick, Draw!'s
own labels instead of a model, is at [2796gaurav.github.io/shad0w/doodle](https://2796gaurav.github.io/shad0w/doodle/).

## Video streams, structured extraction, signed certificates (experimental)

- **Video** (`shad0w.stream.Stream`). Consecutive frames mostly carry the same answer. A small yes/no table, scored
  on the change since the last keyframe, decides whether that answer still holds; otherwise the frame table or your
  vision model answers. One certificate covers the cascade and counts one frame per block, because adjacent frames
  are not independent. Transition frames are reported separately.
  - Result on UCSD Ped2 with gpt-4.1-mini: reuse answered 65–66% of test frames at 2.3–3.6% disagreement, in 2 of 3
    clip splits, and only at a block size chosen after the fact.
  - The pre-registered setting certified nothing: the dataset is about 8 minutes of video, and the teacher changes
    its own answer on 2–9% of identical frames. `examples/stream_bench.py` measures that flicker first.
- **Structured extraction** (`shad0w.extract`). `compile({field: options}, records)` builds one certified table per
  enum field and decides all fields from one hashing pass. `decide()` returns the certified values and the fields to
  ask your LLM for; `merge()` folds in the LLM's JSON.
  - Per-field certificates are not record-level: on MultiWOZ, per-field α = 5% left 16% of records with a wrong field.
  - `record_alpha=0.10` splits α across fields (Bonferroni): 60% of records were served whole and 5.2% had any wrong
    field.
- **Option-order voting** (`llm_teacher(..., stabilise=2)`). Each input is asked in 3 option orders and the majority
  kept. On BANKING77 with gpt-4.1-mini, flips against a fourth order fell from 17.3% to 13.0%, for 3× the calls.
- **Majority-vote calibration** (`shad0w votes --votes 3`). At temperature 0.7, disagreement with the temperature-0
  answer fell from 5.6% to 4.3%. The certificate records the vote count.
- **Signed certificates.** `SHAD0W_CERT_KEY=... shad0w certify --sign` writes an HMAC; `shad0w verify --teacher
  gpt-4.1-mini` checks it and the teacher id.
- **Proxy.** Unknown System One question types (`box`, `segment`) and image or audio `state` parts now go upstream
  untouched and are never served from a table.
- **Against Jevstiller** (`examples/jevstiller_compare.py`, its protocol, labels as the teacher). At α = 2%, 95%
  confidence, the table serves 68% of BANKING77 and 82% of CLINC150 at about 20 µs. Jevstiller reports 72% and 69%
  at about 15 ms with a Jev teacher, so the setups differ.

## Safer and stronger by default

- **Live breach check.** Spot checks feed an anytime-valid lower confidence bound on served disagreement (a betting
  confidence sequence). If it rises above α the table stops serving (`flag="audit_breach"`) until you retrain; on a
  table that is as good as certified it fires in under δ of runs.
- **Quieter drift guard.** The drift alarm now needs 3 standard errors of calibration-plus-window noise, not a flat
  3 points, so a few hundred calibration rows no longer trip it on ordinary traffic.
- **Is your LLM consistent?** `shad0w doctor --log L --llm provider/model --teacher-noise 50` re-asks your LLM and
  reports how often it repeats itself. Above α, nothing can certify, whatever the table.
- **Soft labels.** A teacher that reports probabilities (decision-model endpoints do; or return
  `{"answer": ..., "probabilities": {...}}`) has them logged and fitted on. In a simulation on BANKING77 the
  certified share went from 29% to 50%. The certificate still uses the hard answers.
- **Static word vectors** (`static="potion-8M"`, opt-in, `pip install "shad0wllm[static]"` to train, nothing extra to
  serve). Mixed results: BANKING77 certified 70% → 74–80%, AG News 63% → 59%, so measure on your task.
- **Synthetic warm start** (`sh.synthesize(per_option=20)`): the LLM writes examples per option, used for fitting
  only, never for the certificate. On BANKING77 it raised agreement by 4–7 points at 300–1,000 logged rows, but did
  not make 77 options certify any sooner.

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
- Reproduce it on a laptop with a free teacher model (bge-small, 68% accurate) on BANKING77: the in-browser demo answers 63.3% locally with 3.3% realised disagreement; `python examples/shadow_demo.py` runs the same on your machine with the drift guard on (it answered between 59% and 70% locally in our runs, depending on library versions).
- Where the linear table loses: sentiment (SST-2) certifies 0% and defers everything; legal clauses (LEDGAR) 13%. The deep table (above) certified 74% of SST-2 with the labels as the teacher; the benchmark page has not been re-run with it yet.

Every number above has a source file, hardware, teacher and date on the [benchmarks page](https://2796gaurav.github.io/shad0w/docs/benchmarks.html).
<!-- numbers:end -->

## Decision Bench: shad0w against open decision models on a CPU

`bench/` is a reproducible benchmark. Six public decision tasks are answered by shad0w and by open decision models run
zero-shot on the CPU only: Julia-1, Laya and Kev-0.8B in llama.cpp, and Decider-0.8B and GLiNER2.5-Decide through their
own packages. They share the same rows, one decision at a time, and every model's confidence goes through the same 5%
certifier. Every
prediction is saved, so the numbers can be re-checked without running a model. Full tables and charts are in
`bench/site/index.html` and, interactive, at [2796gaurav.github.io/shad0w/bench](https://2796gaurav.github.io/shad0w/bench/); the protocol and caveats are in [bench/README.md](bench/README.md).

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

<p align="center">Built by <b>Gaurav Chauhan</b> · <a href="https://www.linkedin.com/in/gauravc2708/">LinkedIn</a> · <a href="https://github.com/2796gaurav">GitHub</a> · <a href="mailto:gauravc2708@gmail.com">gauravc2708@gmail.com</a> · Apache-2.0<br>
Questions, ideas or a decision you want to try it on? Open an issue or write to me.</p>
