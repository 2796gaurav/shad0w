"""The certificate's promise, checked by simulation: over many independent calibration draws, the share of
draws whose certified threshold has a TRUE disagreement rate above alpha must be at most delta."""
import numpy as np
import pytest

from shad0w.reliability import SelectiveRiskController


def _draw(rng, n):
    conf = rng.beta(5, 1.2, n)
    wrong = rng.random(n) < np.clip(0.9 * (1 - conf) ** 1.5, 0, 1)  # disagreement is likelier at low confidence
    return conf, ~wrong


@pytest.mark.parametrize("alpha", [0.02, 0.05])
def test_violation_rate_is_at_most_delta(alpha):
    rng = np.random.default_rng(42)
    delta = 0.1
    big_c, big_ok = _draw(rng, 1_000_000)  # stands in for the population
    order = np.argsort(-big_c)
    c_sorted, err_cum = big_c[order], np.cumsum(~big_ok[order])

    def true_risk(t):
        k = np.searchsorted(-c_sorted, -t, side="right")
        return err_cum[k - 1] / k if k else 0.0

    draws, violations, served = 300, 0, 0
    for _ in range(draws):
        c, ok = _draw(rng, 2000)
        t = SelectiveRiskController(alpha, delta).fit(c, ok).threshold
        if np.isfinite(t):
            served += 1
            violations += true_risk(t) > alpha
    assert served > draws // 2, "the bound should be reachable in this setting"
    assert violations / draws <= delta + 0.05  # delta plus Monte Carlo slack (SE ~ 0.017)
