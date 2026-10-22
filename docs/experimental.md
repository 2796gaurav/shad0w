---
title: Experimental layers
description: Images, a prompt-injection guard, video frames, structured extraction and signed certificates: how to use each, and what it measured, misses included.
---
# Experimental layers

<p class="lead">The same idea, a small table plus a certificate that says when to trust it, applied beyond text questions. Each layer below works and is tested, but is new in 0.4 and may change. Every number was measured; misses are stated.</p>

## Images

Asking a vision model to pick one label for an image? The same loop works on pixels. Tier 0 uses no neural network: a 32×32 thumbnail, 256 random pixel-comparison ferns, a coarse colour grid and a 64-bit difference hash become items for the same table and certificate. An optional tier 1 quantises a frozen image encoder's embedding (CLIP, SigLIP, or your own via `register_embedder`). One certificate covers the whole cascade.

```python
from shad0w.vision import ImageSpec, EmbedSpec
photo = shad0w.decision("category", options=[...], llm="openai/gpt-4.1-mini", image=True)            # ferns only
photo = shad0w.decision("category", options=[...], llm="openai/gpt-4.1-mini",
                        image=[ImageSpec(), EmbedSpec("openai/clip-vit-base-patch32")])              # a cascade
photo(pixels)      # numpy array or PIL image
photo.train()
```

| one CPU core, α = 5%, unseen images | answered locally | disagreement on those | cost per image |
|---|---|---|---|
| product photos (Fashion-MNIST), ferns | **70%** | 4.1% | **4–7 µs** (C; 9 µs on a 12th-gen Intel laptop) |
| natural photos (CIFAR-10), ferns | 0% (correctly defers everything) | — | 4–7 µs |
| natural photos (CIFAR-10), ferns → CLIP B/32 table | **73%** | 5.5% (95% CI ≈ 3.6–8%) | ~28 ms instead of 2.6 s |

**Try it:** [Doodle Duel](../doodle/) runs an image table in your browser and guesses your drawing on every stroke.

Microseconds assume pixels already in memory: decoding a JPEG or PNG costs 0.3–3 ms. `pip install "shad0wllm[vision]"` for image files. C API: `s0_decide_image`.

## Guardrails

`shad0w.guard.Guard` puts a learned table in front of your LLM as a prompt-injection gate that can **block** in microseconds but **never allows** on the table's word: anything it does not block goes to your model.

```python
from shad0w.guard import Guard
gate = Guard(shad0w.decision("injection", options={"attack": "...", "safe": "..."}, llm="openai/gpt-4.1-mini"))
gate.calibrate(fresh_texts, llm_answers)   # a block-only certificate: among inputs it blocks, disagreement <= alpha
v = gate.check(text)                       # "block" or "ask_model", never "allow"
```

The input is normalised first (Unicode forms, zero-width, bidi and tag characters, homoglyphs, leetspeak, spaced letters, base64 and hex payloads); anything the normaliser had to undo goes to your model. On ~9,400 held-out texts labelled by an LLM:

| | measured |
|---|---|
| blocked by the table, without an LLM call | 14% of all traffic, 57% of the attacks |
| blocks the model called safe | 2.6% (block-only certificate at α = 5%) |
| smuggled variants of blocked attacks allowed | **0%**: all go to the model |
| benign inputs sent to the model because they looked obfuscated | 1.0% |
| latency, normaliser + table (Python) | ~155 µs median, against ~1.1 s for the model |

It catches repeats of known attacks cheaply. It is not a defence against an adaptive attacker on its own, which is why it never allows.

**Try it:** [the prompt guard](../guard/) runs a guard table in your browser, block-only.

**Streaming replies** (`StreamGuard`) check a reply as it streams, at 16, 32, 64, 128, 256 words and then every 256, and block on the first certified prefix. Honest result on BeaverTails: a word table certifies nothing at α = 5%; at α = 10% it blocks 50% of unsafe replies after 47% of their words and over-blocks 4.8% of safe ones. Use it as an early cut-off in front of a real guard model.

## Video frames

`shad0w.stream.Stream` reuses a certified answer across consecutive frames: a small yes/no table, scored on the change since the last keyframe, decides whether the answer still holds; otherwise the frame table or your vision model answers. The certificate counts one frame per block, because adjacent frames are not independent.

On UCSD Ped2, reuse answered 65–66% of test frames at 2.3–3.6% disagreement in 2 of 3 clip splits, and only at a block size chosen after the fact. The pre-registered setting certified nothing: the teacher changed its own answer on 2–9% of identical frames. `examples/stream_bench.py` measures that flicker first.

## Structured extraction

`shad0w.extract.compile({field: options}, records)` builds one certified table per enum field and decides all fields from one hashing pass. `decide(text)` returns the certified `values` and the `deferred` fields; `prompt(text, fields)` asks your LLM for only those, and `merge()` folds its JSON back in.

- Per-field certificates are not record-level: on MultiWOZ, per-field α = 5% left 16% of records with a wrong field.
- `record_alpha=0.10` splits α across fields: 60% of records were served whole and 5.2% had any wrong field.

## A steadier teacher

- **Option-order voting** (`llm_teacher(..., stabilise=2)`): each input is asked in 3 option orders and the majority kept. On BANKING77, flips against a fourth order fell from 17.3% to 13.0%, for 3× the calls.
- **Majority-vote calibration** ([`shad0w votes`](cli.html#votes)): at temperature 0.7, disagreement with the temperature-0 answer fell from 5.6% to 4.3%.
- **Is your LLM consistent?** `shad0w doctor --log L --teacher-noise 50` re-asks it and reports how often it repeats itself. Above α, nothing can certify.

## Signed certificates

`SHAD0W_CERT_KEY=... shad0w certify --sign` (or `shad0w sign`) writes an HMAC of the certificate; `shad0w verify --teacher <name>` checks it and the teacher id. See [sign and verify](cli.html#sign-and-verify).
