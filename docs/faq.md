---
title: FAQ
description: Short answers to common questions about shad0w, each linking to the page with the details.
---
# FAQ

Short answers to the questions people ask first. Each one links to the page with the details.

<div class="viz" data-viz="flow">A message arrives. If shad0w's table is sure enough to stay inside its certified bound, it answers in microseconds. If not, the message goes to your LLM, and the LLM's answer is returned to you and logged so the next training run learns from it.</div>

### What does shad0w do?
It learns the fixed-choice decisions your LLM already makes (intent, route, topic, yes/no) from your LLM's logged answers. A small table then answers the messages it is sure about in microseconds, and sends the rest to your LLM. It only answers where it has been tested to differ from your LLM on at most α of answers (5% by default). See [the docs home](./).

### Is this a cache?
No. A cache answers only messages it has seen before. The table generalises to new wordings of what it learned, and its guarantee covers new messages drawn like your traffic. See [how it compares](compare.html).

### How much traffic does it take over?
On 3 public intent and topic tasks, 61–86% at α = 5% (LLM: gpt-4.1-mini (OpenAI)), with end-to-end accuracy within about a point of the LLM alone. It grows with logged traffic: 0% at 250 answers and 60% at 10,000 on BANKING77. Sentiment-like tasks get 0%, by design. See [benchmarks](benchmarks.html).

### What exactly is guaranteed?
Among the answers the table gives, at most α differ from your LLM's, with 90% confidence, on traffic like the traffic it was tested on. It is a bound against your LLM, not against the truth. See [the guarantee](guarantee.html).

### How fast is it?
A decision costs 1.0 µs in C, 10 µs in JavaScript and 2.4 µs in Python (5.2 µs with per-option probabilities). Through the proxy, a table answer reaches the client in 0.53 ms. See [benchmarks](benchmarks.html#latency-per-runtime).

### How much data does it need?
At least 1,000 logged LLM answers per question (`min_rows`) before `train()` runs. Questions with many options need more before the table answers much: a few thousand for 77 options. See [training](training.html).

### Does it need a GPU, labels or a training pipeline?
No. It needs a CPU, your LLM's answers and `pip install "shad0wllm[compile]"` for training. Training takes seconds to minutes for tens of options; serving needs only numpy. See [scale](scale.html).

### Which LLMs can it learn from?
Any. Presets cover OpenAI, Anthropic, Gemini, Groq, Together, OpenRouter, Mistral, DeepSeek, Fireworks, xAI, Ollama, vLLM and LM Studio, plus the OpenAI Decisions API (`openai-decisions/`) and System One decision models such as Jev, Kev and Laya (`systemone/`). Anything else works through `base_url=`, a `complete=` function, or a plain `fn(text) -> answer`. See [connect your LLM](connect-your-llm.html).

### Does it work with the OpenAI Decisions API?
Yes, with no code change. Run `shad0w proxy --upstream https://api.openai.com/v1` and point your client's `base_url` at it. Certified questions are answered locally in the same JSON shape; the rest go to `gpt-6-luna` and are logged. See [Decisions API & System One](decisions-api.html).

### Jev, Kev and Laya are already fast. Why add shad0w?
They take milliseconds to a second per call; the table answers its certified share in microseconds on your CPU, with no per-call price and a written bound on how often it differs from them. It needs logged answers first and never covers everything. Use the decision model as the LLM shad0w learns from. See [how it compares](compare.html).

### How do I tune it?
`alpha` sets the bound, `audit_rate` the spot checks, `mode` and `canary` the rollout, and `never_serve` the answers that always go to your LLM. Set them in code, as `SHAD0W_*` variables or in `shad0w.toml`; `shad0w config` shows the result. See [configuration](configuration.html) and [tuning](tuning.html).

### Can I set the confidence threshold myself?
You can make it stricter with `min_confidence`, and the guarantee still holds. `force_threshold` lets the table answer below the certified threshold, but those answers come back `certified=False` with the flag `manual_threshold`, because the guarantee does not cover them. See [configuration](configuration.html).

### What happens when my LLM changes (new model, new prompt)?
The certificate is against the old LLM. Rotate the log, let the new LLM answer for a while, and retrain. Until then, keep the old table or set `mode="off"` to send everything to the new LLM. See [tuning](tuning.html).

### Off-topic messages get a real intent. Why?
The table can only answer with one of your options. Add a catch-all such as `"other": "anything else"`, so your LLM labels off-topic messages `other` and the table learns that. See [limits](limits.html#it-assumes-traffic-like-the-calibration-traffic).

### Can it do multi-label or open-ended output?
No. One answer from a fixed list per question. For several labels, ask one yes/no question per label. See [limits](limits.html).

### How big is the table?
About 1.5 MB for 77 options. It grows with the number of options and the vocabulary of your traffic, not with the amount of traffic, and `max_mb` caps it. See [scale](scale.html).

### Is it safe against adversarial input?
Set `exposed=True` for user-facing channels where people may try to game it. The table then also defers any answer that a few inserted words or character patterns could flip (flag `low_radius`). For high-stakes decisions, keep your LLM in the loop. See [configuration](configuration.html).

### Is this new?
The idea of a small model learning from an LLM and deferring when unsure is not new (see Online Cascade Learning, ICML 2024, and vCache, ICLR 2026, in the [references](guarantee.html#references)). What shad0w adds is a packaged, tested version you can drop in: a finite-sample certificate on every table, microsecond answers on a CPU, and a proxy that needs no code change.

### Am I allowed to learn from my LLM's answers?
shad0w learns your own app's labels for your own questions, to answer your own traffic; it does not build a general model. Still, read your provider's terms for your use case. With open-weight models (Kev, Laya, Llama, Qwen) there is no such question.

### Does any data leave my machine?
No. Logs, training and tables are local files, and there is no telemetry. The only network traffic is the LLM calls you already make.

### What does it cost?
Nothing. It is open source under Apache-2.0.

### Where do I report a problem or share results?
On [GitHub](https://github.com/2796gaurav/shad0w/issues). Results on your own data, good or bad, are very welcome.
