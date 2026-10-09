"""Quickstart (offline, no key, ~20 s): decision() in front of "your LLM", log its answers, train, serve.

    python examples/quickstart.py

With a real model you would write llm="openai/gpt-6-luna", api_key=os.environ["OPENAI_API_KEY"] (or api_key_env=...).
Here a small keyword function stands in for the LLM so the example runs anywhere: decision() accepts any
callable(text) -> answer as `llm`. Training needs: pip install "shad0wllm[compile]".
"""
import json
import os
import random
import shutil

import shad0w

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "quickstart")
shutil.rmtree(OUT, ignore_errors=True)  # start fresh on every run
options = json.load(open(os.path.join(HERE, "schema.json"), encoding="utf-8"))["intent"]["criteria"]

phrases = {
    "refund": ["i want a refund", "money back please", "charged twice, reimburse me", "return my payment"],
    "lost_card": ["i lost my card", "my card was stolen", "block my card now", "cannot find my card"],
    "balance": ["what is my balance", "how much money do i have", "show my account total", "funds available?"],
    "transfer": ["send money to my friend", "transfer 50 to savings", "wire money abroad", "move funds to account"],
}
rng = random.Random(0)


def ask_llm(text: str) -> str:
    """Stand-in for your LLM call: right ~98% of the time, like a real (imperfect) model."""
    truth = next((k for k, ps in phrases.items() if any(p in text for p in ps)), "balance")
    return truth if rng.random() > 0.02 else rng.choice(list(phrases))


def traffic(n: int) -> list[str]:
    openers = ["hi", "hello", "please", "can you help", "what do i do", "i need help", "quick question", ""]
    closers = ["", "thanks", "asap", "today", "what now", "is that possible", "on my account", "for my account"]
    return [f"{rng.choice(openers)} {rng.choice(phrases[rng.choice(list(phrases))])} {rng.choice(closers)}".strip()
            for _ in range(n)]


intent = shad0w.decision("intent", options=options, llm=ask_llm, folder=OUT, audit_rate=0.0)
print(intent)                                       # logging 0/1,000 answers (no table yet)

# Day one: every call goes to "your LLM" and is logged. decide_many runs the LLM calls 8 at a time.
intent.decide_many(traffic(4000))
print(intent)                                       # logging 4,000/1,000 answers

cert = intent.train()                               # compile + certify from the log, then hot-swap
print(f"trained: the table certifies {cert['certified_share_on_calibration']:.1%} of traffic like this, "
      f"disagreeing with your LLM on at most {intent.settings.alpha:.0%} of those")

for text in ["someone stole my card yesterday, my card was stolen", "how much money do i have", "what's the weather"]:
    d = intent(text)
    print(f"{text!r:58} -> {d.answer:10} via {d.source:7} {d.latency_us:8.1f} us  ({d.why})")
print(intent)
