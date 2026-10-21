"""Streams: delta items agree with vision, a compiled stream reuses answers and falls back correctly, and the
stream certificate keeps its promise in simulation (200 calibration draws; realised disagreement among served
frames <= alpha in at least 1 - delta of them, as tests/test_certificate_validity.py checks for one table)."""
import numpy as np
import pytest

from shad0w import stream as S
from shad0w.stream import Stream, _Rows, certify_rows, delta_items, frame_state, stream_compile
from shad0w.vision import ImageSpec, image_digest, image_items_py, thumbnail

OPTIONS = ["empty", "red", "green", "blue"]
COLOURS = {"red": (230, 40, 40), "green": (40, 220, 40), "blue": (40, 60, 230)}
H, W = 48, 64
# no dHash items in the frame table (every frame would add four new items): the stream computes its dHash anyway
SPEC = ImageSpec(size=16, ferns=32, bits=6, grid=4, levels=4, dhash=False)


def synth_clip(rng, n=60, p_change=0.03):
    """One bright square (or none) drifting over a fixed noisy background with sensor noise; now and then it
    vanishes, appears or changes colour. The (deterministic) teacher answers its colour. Returns (frames, answers)."""
    bg = rng.integers(90, 130, (H, W, 3)).astype(np.int16)
    state = rng.choice(OPTIONS)
    y, x, dy, dx = rng.integers(6, H - 18), rng.integers(6, W - 18), rng.choice([-1, 1]), rng.choice([-1, 1])
    frames, answers = [], []
    for _ in range(n):
        if rng.random() < p_change:
            state = rng.choice([o for o in OPTIONS if o != state])
        im = bg + rng.integers(-2, 3, (H, W, 3))
        y, x = int(np.clip(y + dy, 2, H - 14)), int(np.clip(x + dx, 2, W - 14))
        dy, dx = (-dy if y in (2, H - 14) else dy), (-dx if x in (2, W - 14) else dx)
        if state != "empty":
            im[y:y + 12, x:x + 12] = COLOURS[state]
        frames.append(np.clip(im, 0, 255).astype(np.uint8))
        answers.append(state)
    return frames, answers


def synth_clips(n_clips, seed, **kw):
    rng = np.random.default_rng(seed)
    clips = [synth_clip(rng, **kw) for _ in range(n_clips)]
    return [c[0] for c in clips], [c[1] for c in clips]


class FakeTeacher:
    """Deterministic: answers from a (digest -> answer) table and counts its calls."""

    def __init__(self, frames_by_clip, answers_by_clip):
        self.table = {image_digest(f): a for fs, as_ in zip(frames_by_clip, answers_by_clip) for f, a in zip(fs, as_)}
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.table[image_digest(frame)]


def test_frame_state_matches_vision():
    rng = np.random.default_rng(0)
    for spec in (ImageSpec(), ImageSpec(channels=1), SPEC, ImageSpec(size=9, channels=1, ferns=8, grid=0),
                 ImageSpec(version=1, dhash=False), ImageSpec(size=33, grid=5, levels=7)):
        for shape in ((32, 32, 3), (28, 28), (3, 500, 4), (240, 360), (7, 5), (17, 23, 3)):
            a = rng.integers(0, 256, shape, dtype=np.uint8)
            st = frame_state(a, spec)
            assert np.array_equal(st.items, image_items_py(a, spec)), (spec, shape)
            assert np.array_equal(st.thumb, thumbnail(a, spec)), (spec, shape)
    a = rng.integers(0, 256, (240, 360), dtype=np.uint8)
    st = frame_state(a, ImageSpec(channels=1))
    same = delta_items(st, st, 1, 3)
    assert same.dtype == np.uint32 and (same < 2**22).all() and np.array_equal(same, delta_items(st, st, 1, 3))
    b = a.copy(); b[60:180, 100:260] = 255
    assert not np.array_equal(same, delta_items(frame_state(b, ImageSpec(channels=1)), st, 1, 3))
    assert not np.array_equal(same, delta_items(st, st, 2, 3))  # the keyframe's answer is part of the items


@pytest.fixture(scope="module")
def compiled():
    frames, answers = synth_clips(40, seed=1)
    model, cert = stream_compile(frames, answers, OPTIONS, spec=SPEC, block=4, hamming=10, cal_fraction=0.5, seed=0,
                                 C=300.0)  # a fixed C skips the search (most of the compile time)
    return model, cert


def test_compile_certifies_and_reports(compiled):
    model, cert = compiled
    e = cert["questions"]["frame"]
    assert cert["mode"] == "stream" and cert["block"] == 4 and "one frame per block" in cert["statement"]
    assert "against the teacher" in cert["statement"]
    assert e["calibration"] == "held-out-clips/one-row-per-block" and e["n_calibration"] <= e["n_blocks"]
    assert e["n_fit_clips"] + e["n_calibration_clips"] == 40
    assert [t["source"] for t in e["tiers"]] == ["reuse", "table"]
    assert set(e["transition_disagreement"]) >= {"transition_rows", "transition_served", "transition_disagreement"}
    assert e["transition_disagreement"]["transition_rows"] > 0
    # a deterministic teacher never flickers; what is counted here are real colour changes that barely move the grey
    # dHash (red -> green at similar luminance), which is why reuse must not rely on the dHash alone
    assert e["teacher_flicker_on_near_identical_frames"] < 0.03
    assert e["certified_share_on_calibration"] > 0.3 and e["tiers"][0]["share"] > 0.1, e  # reuse does real work
    assert model.meta["stream"]["reuse"] == "frame@reuse" and "frame@reuse" in model.questions


def test_stream_reuses_falls_back_resets_and_roundtrips(compiled, tmp_path):
    model, cert = compiled
    frames, answers = synth_clips(8, seed=2)
    teacher = FakeTeacher(frames, answers)
    s = Stream(model, teacher=teacher)
    decisions = []
    for fs in frames:
        s.reset()
        first = s.decide(fs[0])
        assert first.source != "reuse" and first.frame == 0 and first.keyframe is None
        decisions.append(first)
        decisions.extend(s.decide(f) for f in fs[1:])
    ys = [a for as_ in answers for a in as_]
    assert all(d.source in ("reuse", "table", "teacher") for d in decisions)
    assert all(d.certified == (d.source != "teacher") for d in decisions)
    assert all(d.latency_us > 0 and (d.tier in (0, 1) or d.source == "teacher") for d in decisions)
    st = s.stats()
    assert st["frames"] == len(ys) == sum(st[k] for k in ("reuse", "table", "teacher", "uncertified"))
    assert st["reuse"] > 0.1 * st["frames"] and st["teacher"] == teacher.calls
    served = [(d, y) for d, y in zip(decisions, ys) if d.certified]
    wrong = sum(d.answer != y for d, y in served)
    assert served and wrong / len(served) <= 0.1, (wrong, len(served))  # loose: 8 clips are no certificate
    # a teacher answer outside the options is served as is and is never reused
    model.save(str(tmp_path / "b"))  # before the threshold change below, which must not reach the saved bundle
    s2 = Stream(str(tmp_path / "b"), teacher=lambda f: "banana")  # its own copy of the tables
    s2.q1.threshold = np.inf  # force the teacher
    d0, d1 = s2.decide(frames[0][0]), s2.decide(frames[0][1])
    assert d0.answer == "banana" and d0.source == "teacher" and d1.source == "teacher"
    # the bundle survives save/load with the same decisions
    s3 = Stream(str(tmp_path / "b"), teacher=teacher)
    again = []
    for fs in frames:
        s3.reset()
        again.extend(s3.decide(f) for f in fs)
    assert [(d.answer, d.source) for d in again] == [(d.answer, d.source) for d in decisions]
    # without the reuse tier nothing is reused
    s4 = Stream(model, teacher=teacher, reuse=False)
    assert all(s4.decide(f).source != "reuse" for f in frames[0])


def test_no_reuse_possible_falls_back(compiled):
    model, _ = compiled
    rng = np.random.default_rng(5)
    noise = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(40)]  # every frame a new keyframe
    calls = []
    s = Stream(model, teacher=lambda f: calls.append(1) or "empty", block=4, hamming=6)
    out = [s.decide(f) for f in noise]
    assert all(d.source in ("table", "teacher") for d in out) and s.stats()["keyframes"] == 40
    assert len(calls) == s.stats()["teacher"]
    # compiling from such clips gives no reuse table, and the stream still works
    frames = [[rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(40)] for _ in range(10)]
    answers = [[OPTIONS[rng.integers(4)] for _ in range(40)] for _ in range(10)]
    m2, c2 = stream_compile(frames, answers, OPTIONS, spec=SPEC, block=5, hamming=6, cal_fraction=0.5, seed=1, C=300.0)
    assert m2.meta["stream"]["reuse"] is None and [t["source"] for t in c2["questions"]["frame"]["tiers"]] == ["table"]
    s2 = Stream(m2, teacher=lambda f: "empty")
    assert all(s2.decide(f).source != "reuse" for f in frames[0])


# ---------- the certificate's promise, by simulation ----------

def _population(rng, n, n_clips=30):
    """Rows as a stream produces them: tier 1 better when confident; tier 0 available on 80% of rows; the keyframe's
    answer came from the table (right when confident) or the teacher, which decides what a reused answer is."""
    conf1 = rng.beta(5, 1.2, n)
    agree1 = rng.random(n) > np.clip(0.9 * (1 - conf1) ** 1.5, 0, 1)
    have = rng.random(n) < 0.8
    conf0_t = np.where(have, rng.beta(6, 1, n), np.nan)
    agree0_t = rng.random(n) > 0.01 + 0.5 * (1 - np.nan_to_num(conf0_t)) ** 2
    key_conf1 = np.where(have, rng.beta(5, 1.2, n), np.nan)
    key_right = rng.random(n) > np.clip(0.9 * (1 - np.nan_to_num(key_conf1)) ** 1.5, 0, 1)
    agree0_k = np.where(key_right, agree0_t, ~agree0_t & (rng.random(n) < 0.3))
    conf0_k = np.clip(conf0_t + rng.normal(0, 0.02, n), 0, 1)
    return _Rows(conf1, agree1, conf0_t, conf0_k, agree0_t & have, agree0_k & have, key_conf1,
                 rng.random(n) < 0.1, rng.integers(0, n_clips, n))


def _true_risk(ctl, big):
    c, a = S._under(big, ctl.thresholds)
    r = ctl._route(c, ctl.thresholds)
    served = r >= 0
    if not served.any():
        return 0.0, 0.0
    return float((~a[r[served], np.flatnonzero(served)]).mean()), float(served.mean())


@pytest.mark.parametrize("procedure,n", [("auto", 1500), ("single", 1500), ("cascade", 1500), ("auto", 300)])
def test_stream_certificate_is_valid_in_simulation(procedure, n):
    rng = np.random.default_rng(11)
    big = _population(rng, 300_000)
    alpha, delta, draws = 0.05, 0.1, 200
    viol, shares = 0, []
    for i in range(draws):
        ctl, info = certify_rows(_population(rng, n), alpha, delta, seed=i, procedure=procedure)
        risk, share = _true_risk(ctl, big)
        viol += risk > alpha
        shares.append(share)
    assert viol / draws <= delta + 0.05, viol / draws  # delta plus Monte Carlo slack
    assert np.mean(shares) > (0.3 if n >= 1500 else 0.1), np.mean(shares)  # and it certifies something useful


def test_certify_rows_reports_transitions_and_refuses_when_hopeless():
    rng = np.random.default_rng(3)
    rows = _population(rng, 1500)
    ctl, info = certify_rows(rows, 0.05, 0.1)
    assert np.isfinite(ctl.thresholds).any() and info["transition_rows"] > 0 and info["transition_served"] >= 0
    assert info["variant"] in ("reuse-only", "cascade")
    for proc, want in (("single", ["reuse-only"]), ("cascade", ["cascade"])):
        assert certify_rows(rows, 0.05, 0.1, procedure=proc)[1]["variants"] == want
    with pytest.raises(ValueError):
        certify_rows(rows, 0.05, 0.1, procedure="nope")
    assert info["n_path"] + info["n_test"] == 1500 and info["n_path_clips"] + info["n_test_clips"] == 30
    bad = _population(rng, 1500)
    bad.agree1[:] = rng.random(1500) < 0.5
    bad.agree0_t[:] = bad.agree0_k[:] = rng.random(1500) < 0.5
    ctl, info = certify_rows(bad, 0.05, 0.1)
    assert not np.isfinite(ctl.thresholds).any() and info["served_share"] == 0.0
