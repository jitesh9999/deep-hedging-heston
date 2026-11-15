"""Finite-difference checks of every hand-derived gradient. Run: python -m pytest -q  (or python tests/test_gradients.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deephedge.heston import HestonParams, simulate
from deephedge.hedging import cvar_loss_and_grad, features, pnl
from deephedge.blackscholes import bs_price, bs_delta
from deephedge.nn import MLP


def test_mlp_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP(3, 8, rng)
    X = rng.standard_normal((20, 3))
    c = rng.standard_normal(20)
    f = lambda: float((net.forward(X)[0] * c).sum())
    out, cache = net.forward(X)
    g = net.backward(cache, c)
    eps = 1e-6
    for k, v in net.params.items():
        idx = tuple(rng.integers(0, s) for s in v.shape)
        old = v[idx]
        v[idx] = old + eps; fp = f()
        v[idx] = old - eps; fm = f()
        v[idx] = old
        assert abs(g[k][idx] - (fp - fm) / (2 * eps)) < 1e-6 * max(1, abs(g[k][idx])), k


def test_cvar_gradient_wrt_positions():
    p = HestonParams(n_steps=10)
    S, _ = simulate(p, 400, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    delta = rng.uniform(0.2, 0.8, size=(400, p.n_steps))
    w = 0.03                                              # hold w fixed: loss is piecewise linear in delta
    loss, grad, _ = cvar_loss_and_grad(S, delta, 0.001, 0.04, 0.95, w=w)
    eps = 1e-7
    for _ in range(30):
        i, t = rng.integers(0, 400), rng.integers(0, p.n_steps)
        d = delta.copy(); d[i, t] += eps
        lp = cvar_loss_and_grad(S, d, 0.001, 0.04, 0.95, w=w)[0]
        d[i, t] -= 2 * eps
        lm = cvar_loss_and_grad(S, d, 0.001, 0.04, 0.95, w=w)[0]
        assert abs(grad[i, t] - (lp - lm) / (2 * eps)) < 1e-5 + 1e-4 * abs(grad[i, t])


def test_bs_delta_is_derivative_of_price():
    s, tau, sig = np.array([0.9, 1.0, 1.1]), 0.25, 0.2
    h = 1e-5
    fd = (bs_price(s + h, 1.0, tau, sig) - bs_price(s - h, 1.0, tau, sig)) / (2 * h)
    assert np.allclose(fd, bs_delta(s, 1.0, tau, sig), atol=1e-6)


def test_heston_martingale_and_costs_accounting():
    p = HestonParams()
    S, V = simulate(p, 40000, np.random.default_rng(3))
    assert abs(S[:, -1].mean() - 1.0) < 3e-3              # risk-neutral: E[S_T] = S_0
    assert (V >= 0).all()
    pl0, c0, _ = pnl(S, np.zeros((S.shape[0], p.n_steps)), 0.001, 0.0)
    assert np.allclose(c0, 0.0)                           # no trading -> no cost


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("passed", name)
