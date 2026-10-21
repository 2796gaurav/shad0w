"""The betting bound (binom_ucb_betting) and SelectiveRiskController(procedure="betting"): valid, and how it compares
with Clopper-Pearson."""
import time

import numpy as np
import pytest

from shad0w.reliability import PROCEDURES, SelectiveRiskController, binom_ucb, binom_ucb_betting


def test_bound_basics():
    assert binom_ucb_betting(0, 0, 0.1) == 1.0 and binom_ucb_betting(5, 5, 0.1) == 1.0
    assert 0 < binom_ucb_betting(0, 300, 0.1) < binom_ucb_betting(5, 300, 0.1) < binom_ucb_betting(15, 300, 0.1) < 1
    assert binom_ucb_betting(0, 300, 0.1) > binom_ucb_betting(0, 3000, 0.1)  # tightens with n
    assert binom_ucb_betting(10, 500, 0.01) > binom_ucb_betting(10, 500, 0.1)  # and with delta
    assert binom_ucb_betting(10, 500, 0.1) >= 10 / 500
    with pytest.raises(ValueError):
        binom_ucb_betting(-1, 10, 0.1)
    assert "betting" in PROCEDURES


def test_bound_covers_the_true_rate():
    """A level-delta bound: P(bound < p) <= delta. Checked at p = alpha for a few n (exact binomial draws)."""
    rng = np.random.default_rng(1)
    for n in (50, 300, 1000):
        ks = rng.binomial(n, 0.05, 2000)
        viol = np.mean([binom_ucb_betting(int(k), n, 0.1) < 0.05 for k in ks])
        assert viol <= 0.1 + 0.02, (n, viol)


def test_bound_is_not_tighter_than_clopper_pearson_on_binary_losses():
    """Honest comparison: on 0/1 losses the exact binomial tail wins. Documented, not hidden."""
    looser = [binom_ucb_betting(k, n, 0.1) >= binom_ucb(k, n, 0.1) - 1e-9
              for n in (100, 300, 1000, 3000) for k in range(0, int(0.12 * n), max(1, n // 50))]
    assert np.mean(looser) > 0.95


def _draw(rng, n, flat=False):
    """Calibration draw: confidences and correctness with error falling as confidence rises (or flat noise)."""
    conf = rng.beta(5, 1.5, n)
    p_err = np.full(n, 0.04) if flat else np.clip(0.6 * (1 - conf) ** 2, 0, 1)
    ok = rng.random(n) >= p_err
    return conf, ok, p_err


def _population_risk(conf, p_err, thr):
    served = conf >= thr
    return float(p_err[served].mean()) if served.any() else 0.0, float(served.mean())


@pytest.mark.parametrize("flat", [False, True])
def test_betting_procedure_controls_selective_risk(flat):
    """200 trials x n in {300, 1000, 3000}: the realised (population) selective risk at the certified threshold is
    <= alpha in at least 1 - delta of the trials. Population risk is evaluated on a large fresh sample."""
    alpha, delta, trials = 0.05, 0.1, 200
    rng = np.random.default_rng(7)
    t0 = time.time()
    for n in (300, 1000, 3000):
        viol = 0
        test_conf, _, test_p = _draw(np.random.default_rng(99), 200_000, flat)
        for _ in range(trials):
            conf, ok, _ = _draw(rng, n, flat)
            ctl = SelectiveRiskController(alpha, delta, "betting").fit(conf, ok)
            assert ctl.chosen == "betting" or not np.isfinite(ctl.threshold)
            risk, _ = _population_risk(test_conf, test_p, ctl.threshold)
            viol += risk > alpha
        assert viol / trials <= delta + 0.03, (n, flat, viol)
    assert time.time() - t0 < 30


def test_betting_certifies_less_or_equal_than_clopper_pearson():
    rng = np.random.default_rng(3)
    shares = {"fixed-sequence": [], "betting": []}
    for n in (300, 1000, 3000):
        for _ in range(5):
            conf, ok, _ = _draw(rng, n)
            for proc in shares:
                thr = SelectiveRiskController(0.05, 0.1, proc).fit(conf, ok).threshold
                shares[proc].append(float((conf >= thr).mean()))
    cp, bet = np.mean(shares["fixed-sequence"]), np.mean(shares["betting"])
    assert bet <= cp + 1e-9, (cp, bet)  # looser; the gap depends on how steeply error rises below the threshold
