"""Heston stochastic-volatility simulator (full-truncation Euler, risk-neutral, r = 0)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HestonParams:
    s0: float = 1.0
    v0: float = 0.04       # initial variance (20% vol)
    kappa: float = 3.0     # mean-reversion speed
    theta: float = 0.04    # long-run variance
    xi: float = 0.5        # vol of vol
    rho: float = -0.7      # spot/vol correlation
    T: float = 0.25        # maturity in years
    n_steps: int = 63      # daily rebalancing dates
    substeps: int = 2      # Euler substeps per rebalancing interval


def simulate(p: HestonParams, n_paths: int, rng: np.random.Generator):
    """Return spot paths S of shape (n_paths, n_steps+1) and variance paths v (same shape)."""
    dt = p.T / (p.n_steps * p.substeps)
    S = np.empty((n_paths, p.n_steps + 1))
    V = np.empty((n_paths, p.n_steps + 1))
    logs = np.full(n_paths, np.log(p.s0))
    v = np.full(n_paths, p.v0)
    S[:, 0], V[:, 0] = p.s0, p.v0
    sq = np.sqrt(1.0 - p.rho ** 2)
    for t in range(p.n_steps):
        for _ in range(p.substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)                       # full truncation
            dws = p.rho * z1 + sq * z2
            logs += -0.5 * vp * dt + np.sqrt(vp * dt) * dws
            v = v + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z1
        S[:, t + 1] = np.exp(logs)
        V[:, t + 1] = np.maximum(v, 0.0)
    return S, V
