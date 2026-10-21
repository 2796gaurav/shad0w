"""Schema -> certified per-field tables, with partial deferral to an LLM (shad0w.extract), on MultiWOZ 2.2 user turns.

    python examples/extract_demo.py [--records 20000] [--record-alpha 0.05]

The "teacher" here is the corpus's own slot annotation (what an LLM extractor would have logged); each categorical
slot becomes a field with a "none" option (not mentioned in this turn). The demo compiles one table per field,
decides every test turn on one hashing pass, and reports: fields served per field, records served completely,
record-level disagreement, microseconds per record. Pass --llm to see partial deferral with a stand-in LLM that
answers from the annotation."""
import argparse
import json
import time

import numpy as np

from shad0w import extract

FIELDS = {"restaurant-area": ["centre", "east", "north", "south", "west", "dontcare"],
          "restaurant-pricerange": ["cheap", "moderate", "expensive", "dontcare"],
          "hotel-area": ["centre", "east", "north", "south", "west", "dontcare"],
          "hotel-pricerange": ["cheap", "moderate", "expensive", "dontcare"],
          "hotel-type": ["guesthouse", "hotel", "dontcare"],
          "hotel-parking": ["yes", "no", "dontcare"],
          "hotel-internet": ["yes", "no", "dontcare"],
          "hotel-stars": ["0", "1", "2", "3", "4", "5", "dontcare"]}
SCHEMA = {f.replace("-", "_"): v + ["none"] for f, v in FIELDS.items()}


def turns(split, limit=None):
    """User turns of restaurant / hotel dialogues: text = previous system utterance + user utterance; a field's value
    is the slot value set or changed at this turn, else "none"."""
    from datasets import load_dataset
    d = load_dataset("pfb30/multi_woz_v22", revision="refs/convert/parquet", split=split)
    out = []
    for dlg in d:
        if not ({"restaurant", "hotel"} & set(dlg["services"])):
            continue
        t = dlg["turns"]
        prev, prev_sys = {}, ""
        for i, sp in enumerate(t["speaker"]):
            if sp != 0:
                prev_sys = t["utterance"][i]
                continue
            cur = {}
            for st in t["frames"][i]["state"]:
                sv = st["slots_values"]
                for n, v in zip(sv["slots_values_name"], sv["slots_values_list"]):
                    cur[n] = v[0]
            rec = {"text": f"SYSTEM: {prev_sys}\nUSER: {t['utterance'][i]}"}
            for f, opts in FIELDS.items():
                v = cur.get(f)
                if v is None or prev.get(f) == v:
                    rec[f.replace("-", "_")] = "none"
                else:
                    v = {"free": "yes"}.get(v, v)
                    rec[f.replace("-", "_")] = v if v in opts else "none"
            out.append(rec)
            prev = cur
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--record-alpha", type=float, default=None)
    ap.add_argument("--llm", action="store_true", help="show partial deferral with a stand-in LLM")
    ap.add_argument("--student", default="auto", choices=["auto", "linear", "deep"])
    a = ap.parse_args()
    train = turns("train", a.records)
    test = turns("test")
    t0 = time.time()
    ex = extract.compile(SCHEMA, train, alpha=a.alpha, record_alpha=a.record_alpha, teacher="multiwoz-2.2 annotation",
                         student=a.student)
    print(f"compiled {len(SCHEMA)} fields from {len(train)} turns in {time.time() - t0:.0f}s; {ex.cert['extract']['statement']}")
    served = {f: 0 for f in SCHEMA}
    wrong = {f: 0 for f in SCHEMA}
    mentioned = {f: 0 for f in SCHEMA}  # turns where the field has a value (not "none"), and how many of those were served
    served_mentioned = {f: 0 for f in SCHEMA}
    full = any_wrong = full_wrong = 0
    us = []
    for r in test:
        e = ex.decide(r["text"], observe=False)
        us.append(e.latency_us)
        bad = False
        for f in SCHEMA:
            if r[f] != "none":
                mentioned[f] += 1
                served_mentioned[f] += f in e.values
        for f, v in e.values.items():
            served[f] += 1
            if v != r[f]:
                wrong[f] += 1
                bad = True
        any_wrong += bad
        full += e.complete
        full_wrong += e.complete and bad
    n = len(test)
    print(f"\n{'field':24s} {'served':>8s} {'disagree':>9s} {'mentioned':>10s} {'served when mentioned':>22s}")
    for f in SCHEMA:
        print(f"{f:24s} {served[f] / n:8.1%} {wrong[f] / max(served[f], 1):9.1%} {mentioned[f] / n:10.1%} "
              f"{served_mentioned[f] / max(mentioned[f], 1):22.1%}")
    print(f"\nrecords served completely: {full / n:.1%}; some certified field wrong: {any_wrong / n:.2%} of all records, "
          f"{full_wrong / max(full, 1):.2%} of completely served ones; {np.median(us):.0f} us per record (Python path)")
    if a.llm:
        def stand_in(prompt):  # an LLM that answers from the annotation of the turn in the prompt
            text = prompt.split("Text:\n", 1)[1]
            r = next(x for x in test if x["text"] == text)
            return json.dumps({f: r[f] for f in SCHEMA})
        asked = 0
        for r in test[:2000]:
            e = ex(r["text"], llm=stand_in)
            asked += any(s == "llm" for s in e.sources.values())
        print(f"partial deferral: the LLM was asked for some field on {asked / 2000:.1%} of 2,000 turns, "
              f"for {np.mean([len(ex.decide(r['text']).deferred) for r in test[:2000]]):.2f} fields per turn on average")


if __name__ == "__main__":
    main()
