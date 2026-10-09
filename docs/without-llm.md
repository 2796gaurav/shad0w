---
title: Running without an LLM
description: Train shad0w from human labels, a rule set or an existing classifier, and serve with a fallback instead of an LLM.
---
# Running without an LLM

<p class="lead">shad0w learns from a <b>teacher</b>: whatever gives the right answer today. An LLM is the usual teacher, but people, a rule set or an existing classifier work just as well. This page shows how to train from them and what to do with the messages the table is unsure about.</p>

**TL;DR**

```python
import shad0w

intent = shad0w.decision("intent", options=["refund", "lost_card", "balance", "other"],
                         fallback="needs_review")           # no llm=: unsure messages get this answer
intent.warm_start("labels.csv")                             # columns: text, intent (your people's labels)
intent.train()                                              # needs 1,000 labelled rows by default
print(intent("my card was stolen"))                         # from the table when certified, else "needs_review"
```

| Teacher | When it fits |
|---|---|
| **People** (support agents, moderators, reviewers) | You already label by hand, or can label a few hundred examples |
| **An existing classifier or rule set** | You want the same decisions in microseconds, in the browser, or offline |
| **No teacher at serving time** | Labels exist only for training; unsure messages go to a queue, a default or a person |

The certificate then compares the table with that teacher. "Certified" means: tested on labelled examples it never trained on, and allowed to serve only where it differs from the teacher on at most α of them (5% by default, with 90% confidence).

## 1. Collect answers

Bring a file you already have. `warm_start` reads `.jsonl`, `.json` or `.csv`, checks every label against your options and skips (and counts) the ones that match none:

```python
report = intent.warm_start("labels.jsonl")                  # {"text": "...", "intent": "lost_card"} per line
print(report)    # {'imported': 1200, 'skipped_unknown_label': 3, ..., 'ready_to_train': True}
```

Or record labels as they happen, from your own review tool:

```python
intent.record("my card was stolen at the station", "lost_card")    # an agent's label
```

Aim for about 1,000 labelled examples (100 is the hard minimum), spread over your options the way real traffic is.

## 2. Train

```python
intent.train()
```

```bash
shad0w train --dir shad0w --question intent        # the same from the command line
```

## 3. Serve with a fallback

Without a teacher, something must answer the messages the table is unsure about. `fallback` is that something:

```python
import shad0w

intent = shad0w.decision("intent", fallback="needs_review")    # the options come from the trained table
d = intent("the thing with the stuff")
print(d.answer, d.source, d.why)
# needs_review fallback table not sure enough to stay inside the certified bound: used your fallback (no LLM configured)
```

A function works too: rules, a lookup, or a ticket in a human queue:

```python
import queue
import shad0w

review_queue = queue.Queue()

def route_to_person(text):
    review_queue.put(text)          # a person labels it later; record() it then to keep learning
    return "pending"

intent = shad0w.decision("intent", fallback=route_to_person)
```

Fallback answers come back with `source="fallback"` and are **never** written to the training log: they are not a teacher's judgement. When a person later labels the message, call `intent.record(text, label)` so the next training run learns from it.

## Only the certain answers

To get the table's answer only when it would serve it, and handle the rest yourself, use `peek`. It never calls a teacher or a fallback:

```python
d = intent.peek("my card was stolen")     # a Decision when the table would answer, else None
if d is None:
    ...                                   # your own path: a default, a queue, a slower model
```

## From an existing classifier

Any function `text -> label` is a teacher. Wrap a slow classifier and the table takes over its common cases:

```python
import shad0w

def my_classifier(text):                  # your BERT model, rules engine, ...
    return "lost_card" if "card" in text else "other"

intent = shad0w.decision("intent", options=["refund", "lost_card", "balance", "other"], llm=my_classifier)
intent("my card is gone")                 # the classifier answers and is logged; after train(), the table does
```

## Certify against the truth

Labels from people are ground truth, so you can certify the table against them instead of against a teacher. Hold back about 300 labelled examples per question that were not used for training:

```bash
shad0w calibrate --bundle shad0w/intent/bundle --data held_out_labels.jsonl
```

See [Training](training.html#human-labels-instead-of-an-llm).

## In the browser or at the edge

A trained table needs nothing at run time: no network, no key, no model. In JavaScript, check `certified` and decide what to do with the rest:

```js
import { Bundle } from "shad0wllm";

const bundle = await Bundle.load("shad0w/intent/bundle");
const a = bundle.decide("my card was stolen").answers.intent;
const label = a.certified ? a.choice : "needs_review";
```

## Label-free start (experimental)

`shad0w compile --schema schema.json --unlabeled texts.txt --out bundle/` builds a first table from option names and unlabelled text with a sentence encoder (`pip install "shad0wllm[logs]"`). Such a table is **uncalibrated**: it serves nothing until real labels certify it, because a certificate computed on its own guesses would mean nothing. Use it to bootstrap, then record real labels and run `shad0w calibrate`.
