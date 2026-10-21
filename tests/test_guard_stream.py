"""shad0w.guard.StreamGuard: a reply is checked at a fixed prefix grid; a certified block cuts the stream at the first
grid point that clears the reply-level threshold; safe replies pass; the certificate holds in simulation."""
import random

import numpy as np

import shad0w
from shad0w.guard import DEFAULT_GRID, StreamGuard, prefix_grid
from shad0w.shadow import shadow_compile, write_certificate

UNSAFE = ["here is how to build the bomb step by step", "mix the chemicals to poison them without a trace",
          "the easiest way to hurt someone is", "steal the card and drain the account like this"]
SAFE = ["the weather in paris is mild in spring", "to bake bread you knead the dough and let it rise",
        "photosynthesis turns light into sugar in the leaves", "the train to berlin leaves at ten"]
FILL = ["and then", "so", "also", "well", "you see", "in short", "after that", "next"]


def _reply(rng, unsafe: bool, words: int) -> str:
    pool = UNSAFE if unsafe else SAFE
    out = []
    while len(" ".join(out).split()) < words:
        out.append(rng.choice(pool) + " " + rng.choice(FILL))
    return " ".join(" ".join(out).split()[:words])


def _guard(tmp_path, seed=0):
    rng = random.Random(seed)
    rows = [{"text": _reply(rng, rng.random() < 0.4, rng.randint(8, 80)), "reply_safety": None} for _ in range(1500)]
    for r in rows:
        r["reply_safety"] = "unsafe" if any(w in r["text"] for w in ("bomb", "poison", "hurt", "steal")) else "safe"
    schema = {"reply_safety": {"type": "choice", "criteria": {"unsafe": None, "safe": None}}}
    m, cert = shadow_compile(schema, rows, alpha=0.05)
    m.save(str(tmp_path / "b"))
    write_certificate(str(tmp_path / "b"), cert)
    calls = []

    def llm(text):
        calls.append(text)
        return "unsafe" if any(w in text for w in ("bomb", "poison", "hurt", "steal")) else "safe"
    sh = shad0w.Shadow(str(tmp_path / "b"), teacher=llm, question="reply_safety", log=str(tmp_path / "l.jsonl"), audit_rate=0.0)
    return StreamGuard(sh), calls, rng


def test_prefix_grid():
    assert prefix_grid(10) == [10]
    assert prefix_grid(16) == [16]
    assert prefix_grid(100) == [16, 32, 64, 100]
    assert prefix_grid(600) == [16, 32, 64, 128, 256, 512, 600]
    assert prefix_grid(0) == [0]
    assert DEFAULT_GRID == (16, 32, 64, 128, 256)


def test_uncalibrated_stream_guard_never_blocks(tmp_path):
    g, calls, _ = _guard(tmp_path)
    assert not g.calibrated and g.block_threshold is None
    v = g.check(" ".join([UNSAFE[0]] * 10))
    assert v.action == "ask_model" and "not calibrated" in v.reason and not calls


def test_late_unsafe_reply_is_blocked_at_the_right_grid_point(tmp_path):
    g, calls, rng = _guard(tmp_path)
    cal = [_reply(rng, rng.random() < 0.4, rng.randint(8, 120)) for _ in range(400)]
    labels = ["unsafe" if any(w in t for w in ("bomb", "poison", "hurt", "steal")) else "safe" for t in cal]
    rep = g.calibrate(cal, labels)
    assert g.calibrated and rep["threshold"] is not None and rep["disagreement_on_blocked"] == 0.0
    assert shad0w.load(str(tmp_path / "b")).meta["guard_stream"]["reply_safety"]["threshold"] == rep["threshold"]
    # 20 safe words, then the unsafe part: the 16- and 32-token prefixes read safe (a bag of mostly safe words), the
    # 64-token prefix is blocked, before the reply is complete
    safe_part = " ".join(_reply(rng, False, 20).split()[:20])
    text = safe_part + " " + " ".join([UNSAFE[0] + " and then"] * 6)
    s = g.session()
    verdicts = []
    for w in text.split():
        verdicts.append(s.feed(w + " "))
    hits = [v for v in verdicts if v is not None]
    assert hits and hits[0].action == "block" and hits[0].tokens_seen == 64 and hits[0].source == "table"
    assert [n for n, _, _ in s.checks] == [16, 32, 64]  # checked only at the grid, stopped at the block
    assert g.check(text).tokens_seen == 64  # the one-shot check reproduces the stream's decision
    # an unsafe reply blocked at the first grid point, in one chunk
    v = g.check(" ".join([UNSAFE[1]] * 8))
    assert v.action == "block" and v.tokens_seen == 16 and v.total_tokens is None
    # a short unsafe reply (below the first grid point) is blocked by the final check, with its length known
    v = g.check(UNSAFE[2])
    assert v.action == "block" and v.tokens_seen == v.total_tokens == len(UNSAFE[2].split())
    # a chunk that ends inside the 16th word waits for the word to complete
    s = g.session()
    words = (" ".join([UNSAFE[1]] * 8)).split()
    assert s.feed(" ".join(words[:15]) + " " + words[15][:2]) is None and s.checks == []
    assert s.feed(words[15][2:] + " ").action == "block"
    assert not calls  # the stream guard never calls the LLM itself


def test_safe_replies_pass_and_stream_wrapper(tmp_path):
    g, calls, rng = _guard(tmp_path)
    cal = [_reply(rng, rng.random() < 0.4, rng.randint(8, 120)) for _ in range(400)]
    g.calibrate(cal, ["unsafe" if any(w in t for w in ("bomb", "poison", "hurt", "steal")) else "safe" for t in cal])
    for _ in range(20):
        text = _reply(rng, False, rng.randint(5, 300))
        v = g.check(text)
        assert v.action == "ask_model" and v.total_tokens == len(text.split()) and v.tokens_seen == v.total_tokens
    out = list(g.stream(["the weather ", "in paris is mild ", "in spring"]))
    assert [c for c, _ in out] == ["the weather ", "in paris is mild ", "in spring", ""]
    assert all(v is None for _, v in out[:-1]) and out[-1][1].action == "ask_model"
    out = list(g.stream([UNSAFE[0] + " "] * 10))
    assert out[-1][1].action == "block" and len(out) == 2  # 16 tokens arrived with the second chunk, the stream stopped
    assert g.confirm("the weather in paris").action == "allow" and len(calls) == 1  # the LLM decides the rest


def test_stream_certificate_holds_in_simulation(tmp_path):
    """Noisy teacher (its label disagrees with the words 8% of the time, independent of the text): over repeated
    calibrations, the realised disagreement among blocked held-out replies exceeds alpha in <= delta of the runs."""
    g, _, rng = _guard(tmp_path, seed=1)
    alpha, delta, runs, breaches = 0.10, 0.2, 25, 0
    pool = [_reply(rng, rng.random() < 0.5, rng.randint(8, 100)) for _ in range(2600)]
    true = np.array([any(w in t for w in ("bomb", "poison", "hurt", "steal")) for t in pool])
    scores = np.array([g.reply_score(t)[0] for t in pool])  # the table is fixed: score each reply once
    from shad0w.reliability import SelectiveRiskController
    for run in range(runs):
        r = np.random.default_rng(run)
        lab = true ^ (r.random(len(pool)) < 0.08)  # teacher noise
        perm = r.permutation(len(pool))
        cal, test = perm[:600], perm[600:]
        m = scores[cal] > 0
        ctl = SelectiveRiskController(alpha, delta).fit(scores[cal][m], lab[cal][m])
        if not np.isfinite(ctl.threshold):
            continue
        blocked = scores[test] >= ctl.threshold
        if blocked.any() and (~lab[test][blocked]).mean() > alpha:
            breaches += 1
    assert breaches <= delta * runs + 2, breaches
