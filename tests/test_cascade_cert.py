"""One certificate for a cascade of tables (CascadeController), checked by simulation: over many calibration draws,
the share of draws whose certified cascade has a TRUE disagreement rate above alpha among served answers must be at
most delta; and a cheap tier in front of a good one answers traffic the good one alone would have answered."""
import numpy as np

from shad0w.reliability import CascadeController, SelectiveRiskController


def _two_tiers(rng, n):
    """Tier 0: cheap, confident and right on 'easy' rows only. Tier 1: better everywhere. Teacher noise ~2%."""
    easy = rng.random(n) < 0.4
    c0 = np.where(easy, rng.beta(8, 1, n), rng.beta(2, 3, n))
    ok0 = rng.random(n) > np.where(easy, 0.03 + 0.4 * (1 - c0) ** 2, 0.6 * (1 - c0) + 0.1)
    c1 = rng.beta(5, 1.2, n)
    ok1 = rng.random(n) > np.clip(0.02 + 0.8 * (1 - c1) ** 1.5, 0, 1)
    return np.stack([c0, c1]), np.stack([ok0, ok1])


def _truth(ctl, conf, agree):
    r = ctl.route(conf)
    served = r >= 0
    if not served.any():
        return 0.0, 0.0, np.zeros(conf.shape[0])
    wrong = ~agree[r[served], np.flatnonzero(served)]
    return wrong.mean(), served.mean(), np.array([(r == k).mean() for k in range(conf.shape[0])])


def test_cascade_certificate_is_valid():
    rng = np.random.default_rng(7)
    big_c, big_ok = _two_tiers(rng, 400_000)
    alpha, delta, draws = 0.05, 0.1, 200
    viol, shares = 0, []
    for i in range(draws):
        c, ok = _two_tiers(rng, 2000)
        ctl = CascadeController(alpha, delta, seed=i).fit(c, ok)
        risk, share, _ = _truth(ctl, big_c, big_ok)
        viol += risk > alpha
        shares.append(share)
    assert viol / draws <= delta, viol / draws
    assert np.mean(shares) > 0.2  # and it certifies something useful


def test_cheap_tier_takes_traffic_and_total_coverage_holds():
    rng = np.random.default_rng(3)
    c, ok = _two_tiers(rng, 4000)
    big_c, big_ok = _two_tiers(rng, 200_000)
    ctl = CascadeController(0.05, 0.1).fit(c, ok)
    _, share, per_tier = _truth(ctl, big_c, big_ok)
    single = SelectiveRiskController(0.05, 0.1).fit(c[1], ok[1]).threshold
    single_share = (big_c[1] >= single).mean()
    assert per_tier[0] > 0.15  # the cheap tier answers a real share
    assert share >= single_share - 0.05  # without giving up the good tier's coverage


def test_cascade_handles_missing_tiers_and_nothing_certifiable():
    rng = np.random.default_rng(0)
    c, ok = _two_tiers(rng, 1500)
    c[0, ::2] = np.nan  # tier 0 unavailable on half the rows
    ctl = CascadeController(0.05, 0.1).fit(c, ok)
    assert np.isfinite(ctl.thresholds).any()
    bad = CascadeController(0.05, 0.1).fit(rng.random((2, 500)), rng.random((2, 500)) < 0.5)
    assert not np.isfinite(bad.thresholds).any() and (bad.route(rng.random((2, 10))) == -1).all()
