"""Quickstart (no downloads, ~20 s): shadow-compile from a teacher's logged answers, certify, decide.

    python examples/quickstart.py

In real use, `teacher_log` is what your existing model (an LLM, a decision API, an in-house classifier)
answered on your traffic. Here a tiny synthetic stand-in is generated so the example runs anywhere.
"""
import json
import os
import random
import time

import shad0w
from shad0w.shadow import shadow_compile, write_certificate

HERE = os.path.dirname(os.path.abspath(__file__))
schema = json.load(open(os.path.join(HERE, "schema.json")))

phrases = {
    "refund": ["i want a refund", "money back please", "charged twice, reimburse me", "return my payment"],
    "lost_card": ["i lost my card", "my card was stolen", "block my card now", "cannot find my card"],
    "balance": ["what is my balance", "how much money do i have", "show my account total", "funds available?"],
    "transfer": ["send money to my friend", "transfer 50 to savings", "wire money abroad", "move funds to account"],
}
openers = ["hi", "hello", "please", "can you help", "what do i do", "i need help", "quick question", ""]
closers = ["", "thanks", "asap", "today", "what now", "is that possible", "on my account", "for my account"]
rng = random.Random(0)
teacher_log = []
for _ in range(4000):
    k = rng.choice(list(phrases))
    text = f"{rng.choice(openers)} {rng.choice(phrases[k])} {rng.choice(closers)}".strip()
    answer = k if rng.random() > 0.03 else rng.choice(list(phrases))   # real teachers are not perfect
    teacher_log.append({"text": text, "intent": answer})

t = time.time()
model, cert = shadow_compile(schema, teacher_log, alpha=0.05, teacher="example-teacher")
out = os.path.join(HERE, "out", "bundle")
model.save(out)
write_certificate(out, cert)
q = cert["questions"]["intent"]
print(f"compiled + certified in {time.time() - t:.1f}s -> {out}")
print(f"agreement with teacher {q['agreement_with_teacher']:.1%}; certified share {q['certified_share_on_calibration']:.1%} at alpha=5%")

m = shad0w.load(out)                                                # numpy-only (+ C core if built)
for text in ["someone stole my card yesterday", "how much is in my account", "what's the weather"]:
    a = m.decide(text)["answers"]["intent"]
    route = a["choice"] if a["certified"] else "-> defer to your model"
    print(f"{text!r:40} {a['choice']:10} conf={a['confidence']:.2f} certified={a['certified']}  {route}")

# In production, wrap the model you already call: certified answers come from the table, the rest from your model
# (logged, so the next compile has more data), and 1% of certified answers are spot-checked against your model.
sh = shad0w.Shadow(out, teacher=lambda text: "balance", log=os.path.join(HERE, "out", "teacher_log.jsonl"))
d = sh.decide("please block my card, it was stolen")
print(f"\nShadow wrapper: answer={d.answer!r} source={d.source} in {d.latency_us:.0f} us")
