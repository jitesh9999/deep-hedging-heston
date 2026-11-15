"""Features, P&L accounting and the CVaR objective (with analytic gradient w.r.t. positions)."""
import numpy as np

K = 1.0  # at-the-money strike (S0 = 1)


def ewma_vol(S, p, lam=0.94):
    """Annualised EWMA volatility known at each trading date t = 0..n-1 (uses returns up to t only)."""
    dt = p.T / p.n_steps
    r = np.diff(np.log(S), axis=1)                        # (N, n)
    var = np.empty((S.shape[0], p.n_steps))
    cur = np.full(S.shape[0], p.theta * dt)               # prior = long-run variance
    for t in range(p.n_steps):
        var[:, t] = cur
        cur = lam * cur + (1 - lam) * r[:, t] ** 2
    return np.sqrt(var / dt)


def features(S, p):
    """(N*n, 3) matrix: scaled log-moneyness, time-to-maturity fraction, EWMA vol deviation."""
    N, n = S.shape[0], p.n_steps
    m = np.log(S[:, :n] / K) / 0.1
    tau = np.broadcast_to(1.0 - np.arange(n) / n, (N, n))
    sig = (ewma_vol(S, p) - 0.2) / 0.1
    return np.stack([m, tau, sig], axis=-1).reshape(N * n, 3)


def pnl(S, delta, cost, premium):
    """P&L of: sell a call for `premium`, hold `delta` shares, pay proportional costs, settle payoff.

    delta has shape (N, n) = position held over [t, t+1]; the position is liquidated at maturity.
    """
    dS = np.diff(S, axis=1)
    gains = (delta * dS).sum(axis=1)
    pos = np.concatenate([np.zeros((S.shape[0], 1)), delta, np.zeros((S.shape[0], 1))], axis=1)
    trades = np.abs(np.diff(pos, axis=1))                 # (N, n+1): trade at each date incl. liquidation
    costs = cost * (trades * S).sum(axis=1)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    return premium + gains - costs - payoff, costs, trades.sum(axis=1)


def cvar_loss_and_grad(S, delta, cost, premium, alpha=0.95, w=None):
    """CVaR_alpha of losses L = -P&L and dCVaR/d delta (shape (N, n)).

    Uses the Rockafellar-Uryasev form w + E[(L - w)+] / (1 - alpha). If w is None it is set to the
    empirical alpha-quantile, which is the exact minimiser over w (so no gradient flows through w).
    """
    N, n = delta.shape
    pl, _, _ = pnl(S, delta, cost, premium)
    L = -pl
    if w is None:
        w = np.quantile(L, alpha)
    excess = np.maximum(L - w, 0.0)
    loss = w + excess.mean() / (1.0 - alpha)
    dL = (L > w).astype(float) / (N * (1.0 - alpha))      # dCVaR / dL_i
    dS = np.diff(S, axis=1)
    pos = np.concatenate([np.zeros((N, 1)), delta, np.zeros((N, 1))], axis=1)
    sg = np.sign(np.diff(pos, axis=1))                    # sign of trade at dates 0..n
    # dL_i/d delta_t = -dS_t + c S_t sgn(trade_t) - c S_{t+1} sgn(trade_{t+1})
    d_delta_L = -dS + cost * S[:, :n] * sg[:, :n] - cost * S[:, 1:] * sg[:, 1:]
    return loss, dL[:, None] * d_delta_L, w
