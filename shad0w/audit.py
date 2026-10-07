"""S0-Auditor: verified, shift-proof bounds on the error of ACCEPTED answers.

Three bound engines (all reset when the acceptance threshold changes):
  cp        sliding-window Clopper-Pearson on uniformly audited answers (E4 baseline;
            valid per window, not under continuous monitoring)
  bet       anytime-valid betting confidence sequence (Waudby-Smith & Ramdas) on uniformly
            audited 0/1 errors; restarts spend delta_k = 6 delta / (pi^2 k^2)
  ppi       ACTIVE prediction-powered estimate: every accepted answer contributes
              Z_t = (1-c_t) + (A_t/pi_t) (e_t - (1-c_t)),  A_t ~ Bern(pi_t),
            with pi_t proportional to sqrt(1-c_t) (audit where errors are likely);
            unbiased under any shift. Bound: asymptotic confidence sequence
            (Waudby-Smith et al.), anytime-valid in the large-sample sense.
"""

from __future__ import annotations

import math

import numpy as np

from .reliability import binom_ucb


class BettingUCB:
    """Upper confidence sequence for the mean of [0,1] variables via betting."""

    def __init__(self, delta: float, grid: int = 400):
        self.delta = delta
        self.m = np.linspace(0.0, 1.0, grid + 1)[1:]
        self.logK = np.zeros_like(self.m)
        self.n, self.s, self.s2 = 0, 0.0, 0.0
        self.rejected = np.zeros_like(self.m, dtype=bool)

    def update(self, x: float):
        mu = (self.s + 0.5) / (self.n + 1)
        var = (self.s2 - 2 * mu * self.s + (self.n) * mu * mu + 0.25) / (self.n + 1)
        lam = math.sqrt(2 * math.log(1 / self.delta) / (max(self.n, 1) * math.log(self.n + 2) * max(var, 1e-4) + 1e-12))
        lam_m = np.minimum(lam, 0.5 / np.maximum(1.0 - self.m, 1e-6) )
        # betting that x is SMALLER than m: capital grows when the true mean < m
        self.logK += np.log1p(lam_m * (self.m - x))
        self.rejected |= self.logK >= math.log(1 / self.delta)
        self.n += 1
        self.s += x
        self.s2 += x * x

    def ucb(self) -> float:
        ok = np.where(~self.rejected)[0]
        return float(self.m[ok].max()) if len(ok) else 0.0


class AsymptoticCS:
    """Asymptotic (time-uniform CLT) confidence sequence; rho tuned for n ~ n_star."""

    def __init__(self, delta: float, n_star: int = 2000):
        self.delta = delta
        self.rho2 = (-2 * math.log(delta) + math.log(-2 * math.log(delta) + 1)) / n_star
        self.n, self.s, self.s2 = 0, 0.0, 0.0

    def update(self, z: float):
        self.n += 1
        self.s += z
        self.s2 += z * z

    def ucb(self) -> float:
        if self.n < 30:
            return 1.0
        t = self.n
        mu = self.s / t
        var = max(self.s2 / t - mu * mu, 1e-8)
        r = self.rho2
        w = math.sqrt(var) * math.sqrt(2 * (t * r + 1) / (t * t * r) * math.log(math.sqrt(t * r + 1) / self.delta))
        return mu + w


class Auditor:
    """Threshold controller over a strict->lenient grid of confidence thresholds."""

    def __init__(self, grid: np.ndarray, start: int, alpha: float, delta: float, engine: str,
                 budget: float, rng: np.random.Generator, window: int = 4000):
        self.grid, self.level, self.alpha, self.delta = grid, start, alpha, delta
        self.engine, self.budget, self.rng, self.window = engine, budget, rng, window
        self.restarts = 0
        self.audits = 0
        self.mean_q = budget  # running normaliser for active sampling
        self._reset()

    def _reset(self):
        self.restarts += 1
        d = self.delta * 6 / (math.pi ** 2 * self.restarts ** 2)
        self.cp_buf = []
        self.bet = BettingUCB(d)
        self.acs = AsymptoticCS(d)
        self.since = 0

    @property
    def threshold(self):
        return self.grid[self.level]

    def _move(self, d):
        new = int(np.clip(self.level + d, 0, len(self.grid) - 1))
        if new != self.level:
            self.level = new
            self._reset()

    def _combo(self, t, conf, is_error_fn):
        """Half the budget: uniform audits -> betting bound; half: active audits -> PPI bound.
        Tighten if EITHER bound exceeds alpha; relax only when BOTH are below it."""
        half = self.budget / 2
        q = math.sqrt(max(1.0 - conf, 0.0)) + 0.05
        self.mean_q = 0.999 * self.mean_q + 0.001 * q
        pi = min(1.0, max(half * 0.2, half * q / max(self.mean_q, 1e-6)))
        a_act = self.rng.random() < pi
        a_uni = self.rng.random() < half
        e = is_error_fn() if (a_act or a_uni) else 0.0
        self.audits += int(a_act or a_uni)
        self.acs.update((1 - conf) + (a_act / pi) * (e - (1 - conf)))
        if a_uni:
            self.bet.update(e)
        u_bet, u_ppi = self.bet.ucb(), self.acs.ucb()
        hot_bet = self.bet.n >= 20 and u_bet > self.alpha and self.bet.s / self.bet.n > 0.5 * self.alpha
        hot_ppi = self.acs.n >= 200 and u_ppi > self.alpha and self.acs.s / self.acs.n > 0.5 * self.alpha
        if hot_bet or hot_ppi:
            self._move(-5)
        elif self.acs.n >= 200 and u_ppi < self.alpha and u_bet < self.alpha:
            self._move(+1)

    def step(self, t: int, conf: float, accepted: bool, is_error_fn) -> None:
        if not accepted:
            return
        self.since += 1
        if self.engine == "combo":
            return self._combo(t, conf, is_error_fn)
        if self.engine == "ppi":
            q = math.sqrt(max(1.0 - conf, 0.0)) + 0.05
            self.mean_q = 0.999 * self.mean_q + 0.001 * q
            pi = min(1.0, max(self.budget * 0.2, self.budget * q / max(self.mean_q, 1e-6)))
            a = self.rng.random() < pi
            e = is_error_fn() if a else 0.0
            if a:
                self.audits += 1
            self.acs.update((1 - conf) + (a / pi) * (e - (1 - conf)))
            u = self.acs.ucb()
            if self.acs.n >= 200 and u > self.alpha and self.acs.s / self.acs.n > 0.5 * self.alpha:
                self._move(-5)
            elif self.acs.n >= 200 and u < self.alpha:
                self._move(+1)
            return
        if self.rng.random() >= self.budget:
            return
        self.audits += 1
        e = is_error_fn()
        if self.engine == "cp":
            self.cp_buf.append((t, e))
            self.cp_buf = [x for x in self.cp_buf if x[0] > t - self.window]
            if len(self.cp_buf) >= 20 and t % 50 == 0:
                m, k = len(self.cp_buf), int(sum(x[1] for x in self.cp_buf))
                u = binom_ucb(k, m, self.delta)
                if u > self.alpha:
                    self._move(-5)
                elif u < self.alpha / 2:
                    self._move(+1)
        elif self.engine == "bet":
            self.bet.update(e)
            u = self.bet.ucb()
            if self.bet.n >= 20 and u > self.alpha and self.bet.s / self.bet.n > 0.5 * self.alpha:
                self._move(-5)
            elif u < self.alpha:
                self._move(+1)
