---
title: The guarantee
description: What "certified" means in plain words, what can break it, how the certificate is computed (Learn-then-Test with Clopper–Pearson bounds), and how to read certificate.json.
---
# The guarantee

This page explains the one promise shad0w makes: what it says, what it is measured against, what can break it, and the math behind it. Read the first two sections before you go to production. The rest is for reviewers.

A few words used throughout:

- **The table**: shad0w's small learned model. It is trained on your LLM's logged answers.
- **α (alpha)**: the most disagreement you allow. The default, 5%, means 1 answer in 20.
- **δ (delta)**: the chance that the certificate itself is wrong because of an unlucky sample. The default is 0.1, which means 90% confidence.
- **Certified**: tested on answers the table never trained on, and passed.
- **Defer**: the table does not answer, so the message goes to your LLM.

## In plain words

Before the table answers anything, shad0w tests it on LLM answers it **never trained on**. From that test it picks a confidence threshold. When the table is at least that confident, it answers. Below it, your LLM answers.

The promise is about the answers the table gives:

> On traffic like the calibration traffic, the table's answers differ from your LLM's on **at most α** of them (5% by default). There is at most a **δ** chance (10% by default) that this is false because the test sample was unlucky.

Three things to notice:

1. **It is against your LLM, not against the truth.** The table copies your LLM. If your LLM is wrong, the table is wrong the same way. To bound errors against the truth, certify with about 300 human labels (`shad0w calibrate`, see [training](training.html)).
2. **It is a rate, not a per-message promise.** Any single answer can still differ. Over many served answers, the share that differs stays at or below α.
3. **It holds for traffic like the test traffic.** If what users send changes a lot, the promise no longer covers it. The next section lists what breaks it.

In our runs the realised disagreement on fresh traffic was 3.0–6.0% at α = 5% ([benchmarks](benchmarks.html)).

## What can break it

| What happens | Why the promise stops covering it | What shad0w does | What you do |
|---|---|---|---|
| **Traffic shifts** (a launch, a new channel, many more off-topic messages) | The certificate describes the old mix | The drift guard sees more low-confidence messages than in calibration, flags `drift` and defers to your LLM. Spot checks show live disagreement. | Retrain on fresh logs. See [limits](limits.html). |
| **Your LLM changes** (new model, new prompt) | The certificate compares the table with the *old* LLM | Nothing automatic: shad0w cannot tell your prompt changed | Rotate the log, let the new LLM answer for a while, retrain |
| **Your options change** | The table was certified on a different list of answers | Flags `options_changed` or `option_removed` and defers | Retrain. See [changing your options](options.html). |
| **You force a lower threshold** (`force_threshold`) | Answers below the certified threshold were never tested | Marks them `certified=False`, flag `manual_threshold` | Use `min_confidence` (stricter) instead |
| **Bad luck** | δ allows it in up to 10% of certifications | Spot checks measure live disagreement | Watch the spot-check bound; retrain if it sits above α |

Bad luck is not hypothetical. In earlier runs the realised rate landed above α on 2 of 7 tasks (3.9–5.7% (above 5% on 2 of 7 tasks; max over seeds 6.5%)). In the current runs, CLINC150 went above α because of a traffic shift ([where shad0w loses](benchmarks.html#where-shad0w-loses)).

## Choosing α

A lower α is safer: the table answers less and your LLM answers more. A higher α saves more calls and allows more disagreement. Pick the figure on the selector to see the trade-off on the demo table.

<div class="viz" data-viz="alpha">Lower α means the table answers less and agrees more. Measured on BANKING77 with gpt-4.1-mini as the LLM: at α = 2% the table answered 20% of fresh traffic with 0.4% disagreement; at α = 5%, 61% with 3.0%; at α = 10%, 80% with 7.4%.</div>

The figure uses the demo table. The measured run behind the numbers above:

| α (max disagreement) | answered by the table | realised disagreement |
|---|---|---|
| 1% | 0% | n/a |
| 2% | 20% | 0.4% |
| 5% | 61% | 3.0% |
| 10% | 80% | 7.4% |

*BANKING77. Lower α = safer, fewer calls saved.* Source: `results/evidence.json cmp.banking77.openai.shad0w`.

Set α per question with `alpha=` in code, `SHAD0W_ALPHA`, or `shad0w.toml` ([configuration](configuration.html)). It takes effect at the next `train()`.

## How a certificate is made

`train()` runs these steps for each question:

1. **Set aside calibration answers.** These are LLM answers the table never trains on. Once the log holds at least 100 spot-check rows, shad0w uses those: they are a uniform random sample of live traffic. Before that, it holds out a random `cal_fraction` of the log (default 0.3, at least 100 rows, at most `max_cal` = 3000).
2. **Score them.** For each calibration message, record the table's answer, its confidence, and whether it matches your LLM's answer.
3. **Sort by confidence** and find the lowest threshold at which a conservative upper bound on disagreement is still at most α (details below).
4. **Save it.** The threshold goes into `manifest.json`. The full record goes into `certificate.json` next to it.

If no threshold passes, the threshold is `null` and the table answers nothing. That is the safe outcome, not an error.

Try it offline. A keyword function stands in for your LLM:

```python
import json
import random

import shad0w

KEYWORDS = {"refund": "refund", "lost_card": "stolen", "other": "weather"}

def llm(text):  # a stand-in for your LLM, so this runs offline
    return next((label for label, word in KEYWORDS.items() if word in text), "other")

rng = random.Random(0)
fillers = ["please", "today", "my", "the", "help", "now", "account", "app", "urgent", "hi"]
texts = [" ".join(rng.sample(fillers, 3) + [rng.choice(["refund", "stolen", "weather"])]) for _ in range(1200)]

intent = shad0w.decision("intent", options=["refund", "lost_card", "other"], llm=llm)
for t in texts:
    intent(t)                      # your LLM answers; shad0w logs it

cert = intent.train()              # compile, certify, start serving
print(json.dumps({k: cert[k] for k in ("n_calibration", "certified_share_on_calibration",
                                       "disagreement_on_certified_calibration", "procedure")}, indent=2))
print(shad0w.read_certificate("shad0w/intent/bundle")["statement"])
```

On the command line, `shad0w report --bundle shad0w/intent/bundle` prints the same certificate. Every field is listed in the [reference](reference.html#certificatejson).

## How it stays honest at runtime

The certificate is computed once, at training time. Two checks run on live traffic:

- **Spot checks.** A share of decisions (`audit_rate`, default 1%) is also sent to your LLM, chosen at random across all traffic. For table answers this happens in the background, so it never slows a reply. `stats()` and the dashboard show the observed disagreement and its 90% upper bound next to α. The next `train()` certifies on these rows once there are at least 100.
- **Drift guard.** It needs no labels. At calibration it picks a confidence level below which about as many messages fall as the table got wrong. Live, it tracks the share of messages below that level over the last `drift_window` decisions (default 500). If that share rises more than `drift_margin` (default 0.03) above the calibration share, every answer is flagged `drift` and deferred until the share falls back or you retrain. It starts checking once half the window is full.

See [observability](observability.html) for where these numbers show up.

## Formally

> **Setting.** The calibration set has *n* texts never used for fitting. For each one: your LLM's answer *tᵢ*, the table's answer *ŷᵢ* and its confidence *cᵢ* (the top softmax probability). Disagreement is *eᵢ* = 1[*ŷᵢ* ≠ *tᵢ*].
>
> **Candidates.** Sort the calibration set by confidence, highest first. Candidate *k* serves the top *k* texts, so its threshold is *τₖ* = *c₍ₖ₎* (the *k*-th highest confidence). Let *Eₖ* be the number of disagreements among those *k*.
>
> **Test.** Candidate *k* passes at level *δ′* when the one-sided Clopper–Pearson upper bound is at most α: *U*(*Eₖ*, *k*, *δ′*) = Beta⁻¹(1 − *δ′*; *Eₖ* + 1, *k* − *Eₖ*) ≤ α (and *U* = 1 when *Eₖ* = *k*).
>
> **Procedure** (Learn-then-Test, the default `auto`). Run two valid procedures, each at *δ*/2, and keep the lower (more lenient) of the two thresholds. By the union bound, the result holds at *δ*.
>
> 1. *Fixed sequence.* Walk *k* from strict to lenient over 50 evenly spaced steps, starting at *k*₀, and stop at the first candidate that fails. Keep the last one that passed. *k*₀ depends only on *n*, α and *δ*/2, never on the data: it is the largest of ⌈log(*δ*/2) / log(1 − α)⌉, ⌈0.05 *n*⌉, min(50, *n*), and the smallest *k* at which ⌊α*k*/2⌋ disagreements would still pass. No multiplicity penalty, so it is tight when disagreement rises as confidence falls.
> 2. *Bonferroni.* Test 12 coverage levels fixed in advance (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, … 0.9 and 1.0 times *n*), each at (*δ*/2)/12 = *δ*/24, and keep the most lenient one that passes. It is robust when your LLM makes random mistakes regardless of the input, which can stop the fixed sequence early.
>
> **Guarantee.** With probability at least 1 − *δ* over the calibration sample, P( *ŷ*(*x*) ≠ teacher(*x*) | *c*(*x*) ≥ *τ*\* ) ≤ α, for *x* drawn exchangeably with the calibration data.
>
> **Against the truth.** If your LLM is right with probability *q* on the inputs the table serves, the table's error against the truth on those inputs is at most α + (1 − *q*).

Two honest notes on the proof:

- The candidate thresholds are read off the calibration sample at positions fixed in advance (as in Geifman and El-Yaniv's selective-guaranteed-risk method), not from a grid of confidence values chosen before seeing data. That is why the test suite also checks validity by simulation.
- Exchangeability is the real assumption. A random held-out split of the log meets it for traffic like the log. Once the table serves, the log mostly holds the messages it deferred, which are harder than average. That is why retraining switches to the spot-check rows, a uniform sample of all traffic, as soon as there are 100 of them.

**Simulation.** Benchmark 8 draws 1,000 calibration sets per setting from known populations (`tests/test_certificate_validity.py` runs a smaller version with the test suite). It counts how often a certificate's true disagreement exceeds α, for each procedure, for disagreement that rises as confidence falls and for flat LLM-like noise. Result: 0.0–8.1% of certificates exceeded α in simulation (δ allows 10%). The full table is on the [benchmarks page](benchmarks.html#certificate-validity-by-simulation).

The code is in `shad0w/reliability.py` (`SelectiveRiskController`, `binom_ucb`) and `shad0w/shadow.py` (`_certify`).

## References

- Angelopoulos et al., *Learn then Test*, 2021. [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)
- Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500)
- Jung et al., *Trust or Escalate*, ICLR 2025. [arXiv:2407.18370](https://arxiv.org/abs/2407.18370)
- Zeighami et al., *BARGAIN*, 2025. [arXiv:2509.02896](https://arxiv.org/abs/2509.02896)
- Garg et al., *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance*, ICLR 2022. [arXiv:2201.04234](https://arxiv.org/abs/2201.04234)
- Nie et al., *Online Cascade Learning for Efficient Inference over Streams*, ICML 2024. [PMLR](https://proceedings.mlr.press/v235/nie24a.html) (the closest prior work: a small model learns online from an LLM and defers when unsure; shad0w adds the finite-sample certificate, a µs runtime and the zero-code proxy)
- Schroeder et al., *vCache: Verified Semantic Prompt Caching*, ICLR 2026. [arXiv:2502.03771](https://arxiv.org/abs/2502.03771) (a cache with a user-set error bound; it replays past answers, shad0w generalises to new wording)
- Chen et al., *FrugalGPT*, 2023. [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)
- Joulin et al., *Bag of Tricks for Efficient Text Classification*, 2016. [arXiv:1607.01759](https://arxiv.org/abs/1607.01759)
