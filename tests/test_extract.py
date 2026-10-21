"""shad0w.extract: a schema of enum fields becomes one certified table per field, decided on one hashing pass;
partial deferral hands only the uncertified fields to the LLM and merges its JSON; record_alpha splits alpha by
Bonferroni; free-text fields are always deferred; save/load round-trips."""
import json
import random

import numpy as np

from shad0w import extract

AREA = {"north": ["up north", "in the north", "north side", "northern part"], "south": ["down south", "in the south", "south side"],
        "centre": ["in the centre", "downtown", "city centre", "central"]}
PRICE = {"cheap": ["cheap", "budget", "inexpensive", "low cost"], "expensive": ["expensive", "upscale", "fancy", "high end"],
         "moderate": ["moderate", "mid range", "reasonably priced"]}
PARK = {"yes": ["with parking", "needs parking", "free parking"], "no": ["no parking needed", "without parking"]}
FILL = ["i need a place", "looking for somewhere", "find me a hotel", "book a room", "please", "thanks"]
SCHEMA = {"area": list(AREA) + ["none"], "price": list(PRICE) + ["none"], "parking": list(PARK) + ["none"]}


def synth(n, seed, noise=0.0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        rec, parts = {}, [rng.choice(FILL)]
        for f, vocab in (("area", AREA), ("price", PRICE), ("parking", PARK)):
            if rng.random() < 0.65:
                v = rng.choice(list(vocab))
                parts.append(rng.choice(vocab[v]))
            else:
                v = "none"
            rec[f] = v if rng.random() >= noise else rng.choice(SCHEMA[f])
        rng.shuffle(parts)
        rec["text"] = " ".join(parts) + " " + rng.choice(FILL)
        rows.append(rec)
    return rows


def test_per_field_tables_and_partial_deferral(tmp_path):
    ex = extract.compile(SCHEMA, synth(3000, 0), alpha=0.05)
    assert set(ex.fields) == {"area", "price", "parking"} and ex.free == []
    test = synth(400, 1)
    served = {f: 0 for f in SCHEMA}
    wrong = {f: 0 for f in SCHEMA}
    full = 0
    us = []
    for r in test:
        e = ex.decide(r["text"])
        us.append(e.latency_us)
        assert set(e.values) | set(e.deferred) == set(SCHEMA) and not (set(e.values) & set(e.deferred))
        full += e.complete
        for f, v in e.values.items():
            served[f] += 1
            wrong[f] += v != r[f]
    for f in SCHEMA:
        assert served[f] > 0.5 * len(test), (f, served)  # clean synthetic data: most of each field is certified
        assert wrong[f] <= 0.05 * served[f] + 3, (f, wrong, served)
    assert full > 0.3 * len(test)
    assert np.median(us) < 3000  # Python path, three fields on one hashing pass
    # partial deferral: a text the tables cannot place for one field; the LLM's JSON fills only that field
    e = ex.decide("book a room with parking thanks")
    e = extract.Extraction(dict(e.values), list(e.deferred), e.hints, e.confidence, e.flags)  # a copy to merge into
    certified = dict(e.values)
    llm_json = json.dumps({f: "none" for f in e.deferred} | {"ignored": "x"} | {f: "WRONG" for f in certified})
    m = ex.merge(e, llm_json)
    assert m.complete and all(m.values[f] == v for f, v in certified.items())  # certified values never overwritten
    assert all(m.sources[f] == "llm" for f in m.values if f not in certified)
    # a value matching no option stays deferred; code fences and case are forgiven
    e2 = extract.Extraction({}, ["area", "price"])
    m2 = ex.merge(e2, "```json\n{\"area\": \"North\", \"price\": \"luxury\"}\n```")
    assert m2.values == {"area": "north"} and m2.deferred == ["price"]
    # the prompt asks for the deferred fields only
    p = ex.prompt("some text", ["price"])
    assert '"price": one of' in p and "area" not in p
    # __call__ with an llm closes the loop
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return json.dumps({f: "none" for f in SCHEMA})
    out = ex("hello there", llm=llm)
    assert out.complete and len(calls) == 1
    # save / load round trip keeps fields, certificate and decisions
    ex.save(str(tmp_path / "b"))
    ex2 = extract.load(str(tmp_path / "b"), native=False)
    assert ex2.fields == ex.fields and ex2.cert["extract"]["fields"] == ["area", "price", "parking"]
    for r in test[:50]:
        assert ex2.decide(r["text"]).values == ex.decide(r["text"]).values


def test_record_level_guarantee_and_free_text_fields():
    rows = synth(4000, 2, noise=0.005)  # a slightly noisy teacher (a flat 3% noise would exceed alpha / 3 and certify nothing)
    schema = {**SCHEMA, "name": "text"}
    ex = extract.compile(schema, rows, alpha=0.05, record_alpha=0.05, delta=0.1)
    assert ex.free == ["name"] and ex.cert["extract"]["field_alpha"] == 0.05 / 3 and ex.cert["extract"]["field_delta"] == 0.1 / 3
    assert all(q.alpha == 0.05 / 3 for q in ex.model.questions.values())
    test = synth(600, 3, noise=0.005)
    any_wrong = full = 0
    for r in test:
        e = ex.decide(r["text"])
        assert "name" in e.deferred and not e.complete  # free text is always the LLM's
        enum_full = set(e.values) == set(SCHEMA)
        full += enum_full
        any_wrong += any(r[f] != v for f, v in e.values.items())
    assert any_wrong <= 0.05 * len(test) + 5, (any_wrong, full)  # the record-level bound, with slack for 600 rows
    assert full > 0.2 * len(test)
    # a schema with no enum field cannot be compiled
    try:
        extract.compile({"name": "text"}, rows)
    except ValueError as e:
        assert "no enum field" in str(e)
    else:
        raise AssertionError("expected ValueError")
