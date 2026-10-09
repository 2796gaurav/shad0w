---
title: How shad0w proves it agrees with your LLM
description: The certificate behind shad0w, in plain words: a held-out test, a confidence line, an allowance for bad luck, and what the promise does and does not cover.
date: 2026-10-07
updated: 2026-10-09
tags: certificate, statistics, learn then test, trust
---

"Our shortcut agrees with your LLM 19 times in 20" is an easy sentence to write and a hard one to mean. Agrees on *which* messages? Measured how? What if the measurement was lucky?

shad0w answers those questions with a **certificate**: one sentence, computed before shad0w answers anything, that you could show to a sceptical colleague. This post explains how it is computed, without the maths, and then where the maths comes in.

One fact up front, because it matters for trust. In our runs shadowing a hosted LLM, we allowed shad0w to differ from the LLM on at most 5% of its answers. How often shad0w actually gave a different answer than the LLM: 3.0–6.0%. That is not a contradiction: a promise made with 90% confidence is allowed to overshoot now and then. On CLINC150 it did, because test traffic looked different from the logged traffic. The last section explains why.

## Step 1: hold some answers back

shad0w is a tiny model it learns from your LLM's logged answers, but never from all of them. A slice is set aside: about a third of the log, up to 3,000 answers. Better still, it uses the uniform spot-check sample of live traffic when one exists. shad0w never sees these answers while it learns.

That slice is the exam. Because shad0w has never seen it, how shad0w does on it is an honest preview of how it will do on new traffic like it.

## Step 2: sort by confidence

For every exam question, shad0w gives an answer and a confidence. Sort the exam from most confident to least. At the confident end, shad0w almost always agrees with your LLM. Further down, mistakes creep in.

<div class="cert-demo" id="certDemo" aria-label="Interactive: pick alpha, the most disagreement you allow, and see how much of the traffic shad0w answers">
  <div class="cd-bars" id="cdBars"></div>
  <div class="cd-controls"><span>Allowed disagreement α</span>
    <button data-a="a02">2%</button><button data-a="a05" class="on">5%</button><button data-a="a10">10%</button></div>
  <div class="cd-out"><div><b id="cdServed">–</b><span>of traffic answered by shad0w</span></div><div><b id="cdDis">–</b><span>how often shad0w actually gave a different answer than your LLM</span></div></div>
  <p class="cd-note">BANKING77, 77 options, shadowing a hosted LLM, measured.</p>
</div>
<script>
(() => {
  const S = {a02: ["20%", "0.4%"], a05: ["61%", "3.0%"], a10: ["80%", "7.4%"]};
  const bars = document.getElementById("cdBars"); const n = 40;
  for (let i = 0; i < n; i++) { const b = document.createElement("i"); b.style.height = (96 - i * 2.1) + "%"; bars.appendChild(b); }
  function show(k) {
    const served = parseFloat(S[k][0]) / 100, cut = Math.round(served * n);
    bars.querySelectorAll("i").forEach((b, i) => b.classList.toggle("in", i < cut));
    document.getElementById("cdServed").textContent = S[k][0]; document.getElementById("cdDis").textContent = S[k][1];
    document.querySelectorAll("#certDemo button").forEach((x) => x.classList.toggle("on", x.dataset.a === k));
  }
  document.querySelectorAll("#certDemo button").forEach((x) => x.addEventListener("click", () => show(x.dataset.a)));
  show("a05");
})();
</script>

## Step 3: draw the line, allowing for bad luck

Now pick α, the most disagreement you allow. α = 5% means shad0w may differ from your LLM on at most 1 in 20 answers it gives. 5% is the default.

The naive move: walk down the sorted list until the mistakes so far reach 5%, and draw the line there. The trouble is that the exam is a *sample*. A lucky exam can make shad0w look better than it is. So shad0w does not ask "is the disagreement above the line 5% or less?" It asks a stricter question:

> Even if this exam was unusually kind to shad0w, could the true disagreement above this line still be above 5%?

If the answer is "plausibly, yes", the line moves up. The test behind that question is the **Clopper–Pearson bound**: given *k* mistakes in *n* answers, it gives the highest disagreement rate that is still believable at a chosen confidence. shad0w uses 90% confidence.

The line ends where that worst believable rate stays within α. Above the line, shad0w answers. Below it, your LLM does.

## Step 4: say exactly what was promised

The result is one sentence:

> On traffic like the calibration traffic, among the answers shad0w gives, it disagrees with your LLM **at most α** of the time. This statement is wrong with probability at most **10%**.

Three words in it carry weight:

- **"Like the calibration traffic."** The promise is about traffic that resembles the exam. If your users start asking new kinds of questions, it no longer applies. That is why shad0w watches its own confidence for drift, spot-checks served answers against your LLM, and retrains.
- **"Among the answers shad0w gives."** The bound covers served answers only. Deferred ones are your LLM's.
- **"Your LLM."** shad0w is compared with your LLM, not with the truth. A wrong LLM produces a shad0w that is wrong in the same way. For a bound against the truth, certify on about 300 human-labelled examples instead.

## Where the maths comes in

Picking a threshold with a test, then claiming the test's guarantee, is a known trap. Try enough lines and one will look good by chance. The framework that avoids it is called **Learn-then-Test** (Angelopoulos et al., 2021): treat each candidate line as a hypothesis, and test the hypotheses in a way that controls the chance of *any* false pass.

shad0w runs two such procedures and keeps the more lenient result. The union bound pays for that choice by running each at half the error budget:

- a **fixed-sequence** test, which walks from strict to lenient lines;
- a **Bonferroni** test over twelve pre-chosen coverage levels.

Why both? The fixed-sequence test is the more powerful when shad0w's mistakes cluster at low confidence. Bonferroni is the more robust when your LLM is a little noisy everywhere, which LLMs often are.

We checked the promise by simulation: draw many calibration samples from a population where the truth is known, certify each, and count how often a certificate overstates. The result: 0.0–8.1% of certificates exceeded α in simulation (δ allows 10%).

## What the real numbers look like

On BANKING77, shadowing a hosted LLM, at α = 5% shad0w answered 61% of fresh traffic. How often it actually gave a different answer than the LLM: 3.0%. At α = 2% it answered 20%, and differed on 0.4%.

The actual rate can land above α. A 90% promise is wrong one time in ten by design, and a different traffic mix voids it outright. On CLINC150, test traffic had about ten times the off-topic share of the logged traffic, and how often shad0w actually differed from the LLM went over α. We publish that case in the [benchmarks](../docs/benchmarks.html). A certificate you can trust is one whose failures you can see.

<div class="callout"><b>See it on your own data.</b> <code>shad0w train --log log.jsonl --out bundle/ --alpha 0.02</code> prints what you would get. <a href="../docs/guarantee.html">The guarantee</a> has the formal statement.</div>
