"""The certificate's promise, checked by simulation: over many independent calibration draws, the share of
draws whose certified threshold has a TRUE disagreement rate above alpha must be at most delta.

Two populations: disagreement that rises as confidence falls, and an LLM-like teacher that is wrong ~2.5% of the
time regardless of the input (flat noise), which is where a pure fixed-sequence test tends to stop early."""
import numpy as np
import pytest

from shad0w.reliability import SelectiveRiskController


def _rising(rng, n):
    conf = rng.beta(5, 1.2, n)
    return conf, ~(rng.random(n) < np.clip(0.9 * (1 - conf) ** 1.5, 0, 1))


def _flat(rng, n):
    conf = rng.beta(6, 1, n)
    return conf, ~(rng.random(n) < 0.025 + 0.5 * (1 - conf) ** 3)


def _simulate(draw, alpha, procedure, n=2000, draws=200, delta=0.1, seed=42):
    rng = np.random.default_rng(seed)
    big_c, big_ok = draw(rng, 1_000_000)  # stands in for the population
    order = np.argsort(-big_c)
    c_sorted, err_cum = big_c[order], np.cumsum(~big_ok[order])

    def truth(t):
        k = np.searchsorted(-c_sorted, -t, side="right")
        return (err_cum[k - 1] / k if k else 0.0), k / len(big_c)

    violations, offload = 0, []
    for _ in range(draws):
        c, ok = draw(rng, n)
        t = SelectiveRiskController(alpha, delta, procedure).fit(c, ok).threshold
        risk, share = truth(t) if np.isfinite(t) else (0.0, 0.0)
        violations += risk > alpha
        offload.append(share)
    return violations / draws, float(np.mean(offload))


@pytest.mark.parametrize("procedure", ["auto", "fixed-sequence", "bonferroni"])
@pytest.mark.parametrize("draw", [_rising, _flat], ids=["rising", "flat"])
@pytest.mark.parametrize("alpha", [0.02, 0.05])
def test_violation_rate_is_at_most_delta(alpha, draw, procedure):
    rate, _ = _simulate(draw, alpha, procedure)
    assert rate <= 0.1 + 0.05  # delta plus Monte Carlo slack (SE ~ 0.02)


def test_auto_survives_flat_teacher_noise():
    """A teacher that is wrong 2.5% of the time at random must not make the certificate refuse everything."""
    _, fixed = _simulate(_flat, 0.05, "fixed-sequence")
    _, auto = _simulate(_flat, 0.05, "auto")
    assert auto > 0.85 and auto > fixed + 0.3


def test_auto_keeps_most_of_fixed_sequence_when_disagreement_rises():
    _, fixed = _simulate(_rising, 0.05, "fixed-sequence")
    _, auto = _simulate(_rising, 0.05, "auto")
    assert auto > 0.9 * fixed
