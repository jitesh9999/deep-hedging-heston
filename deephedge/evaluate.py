"""Out-of-sample evaluation against Black-Scholes delta hedging, plus plots."""
import argparse, json, os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .blackscholes import bs_delta, bs_price
from .heston import HestonParams, simulate
from .hedging import K, ewma_vol, features, pnl
from .nn import MLP


def cvar(pl, a):
    L = -pl
    return float(L[L >= np.quantile(L, a)].mean())


def bs_positions(S, p, sigma):
    n = p.n_steps
    tau = p.T * (1 - np.arange(n) / n)
    return bs_delta(S[:, :n], K, tau[None, :], sigma)


def strategies(S, p, net):
    n = p.n_steps
    out = {"Unhedged": np.zeros((S.shape[0], n)),
           "BS delta (flat 20% vol)": bs_positions(S, p, 0.2),
           "BS delta (EWMA vol)": bs_positions(S, p, np.clip(ewma_vol(S, p), 0.05, 1.5)),
           "Deep hedge (MLP)": net.forward(features(S, p))[0].reshape(S.shape[0], n)}
    return out


def evaluate(net, p, cost, n_paths=50000, seed=12345):
    S, _ = simulate(p, n_paths, np.random.default_rng(seed))
    premium = float(bs_price(1.0, K, 0.25, 0.2))
    rows = {}
    for name, d in strategies(S, p, net).items():
        pl, c, turn = pnl(S, d, cost, premium)
        rows[name] = {"mean_pnl": float(pl.mean()), "std_pnl": float(pl.std()),
                      "cvar95_loss": cvar(pl, 0.95), "cvar99_loss": cvar(pl, 0.99),
                      "avg_cost": float(c.mean()), "avg_turnover": float(turn.mean())}
    return rows, S, premium


def table(rows):
    h = "| Strategy | Mean P&L | Std | CVaR95 loss | CVaR99 loss | Avg cost | Turnover |\n|---|---|---|---|---|---|---|\n"
    return h + "\n".join(f"| {k} | {v['mean_pnl']:.4f} | {v['std_pnl']:.4f} | {v['cvar95_loss']:.4f} | {v['cvar99_loss']:.4f} | {v['avg_cost']:.4f} | {v['avg_turnover']:.2f} |" for k, v in rows.items())


def main(out="results"):
    meta = json.load(open(os.path.join(out, "train_log.json")))
    cost = meta["cost"]
    net = MLP(3, meta["hidden"], np.random.default_rng(0))
    net.load(dict(np.load(os.path.join(out, "model.npz"))))
    base = HestonParams()
    stress = replace(base, v0=0.06, theta=0.06, xi=0.8, rho=-0.5)
    res, md = {}, ""
    for label, p in [("Base Heston (training dynamics)", base), ("Stressed Heston (unseen: higher vol, vol-of-vol)", stress)]:
        rows, S, premium = evaluate(net, p, cost)
        res[label] = rows
        md += f"\n### {label}\n\n{table(rows)}\n"
        if label.startswith("Base"):
            S_base, prem = S, premium
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    open(os.path.join(out, "metrics.md"), "w").write(md)
    print(md)

    # --- plots -------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1 training curve
    h = np.array(meta["history"]); k = 25
    sm = np.convolve(h, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(6, 3.4)); ax.plot(h, alpha=.25, color="C0"); ax.plot(np.arange(k - 1, len(h)), sm, color="C0")
    ax.set(xlabel="training step", ylabel="CVaR95 of hedging loss", title="Training objective (fresh Heston paths each step)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curve.png"), dpi=150); plt.close(fig)
    # 2 P&L distributions
    p = base; strat = strategies(S_base, p, net)
    fig, ax = plt.subplots(figsize=(7, 3.8)); bins = np.linspace(-0.12, 0.08, 120)
    for (name, d), c in zip(list(strat.items())[1:], ["C1", "C2", "C3"]):
        pl, _, _ = pnl(S_base, d, cost, prem); ax.hist(pl, bins=bins, histtype="step", lw=1.6, color=c, label=name, density=True)
    ax.set(xlabel="P&L per unit notional", ylabel="density", title="Out-of-sample P&L, 50,000 Heston paths")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, "pnl_distribution.png"), dpi=150); plt.close(fig)
    # 3 learned policy vs Black-Scholes delta
    m = np.linspace(-0.15, 0.15, 200); fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for frac, c in [(0.9, "C0"), (0.5, "C1"), (0.1, "C2")]:
        X = np.stack([m / 0.1, np.full_like(m, frac), np.zeros_like(m)], axis=1)   # EWMA vol = 20%
        ax.plot(m, net.forward(X)[0], color=c, label=f"MLP, {frac:.0%} of maturity left")
        ax.plot(m, bs_delta(np.exp(m), 1.0, frac * 0.25, 0.2), color=c, ls="--", alpha=.7)
    ax.set(xlabel="log-moneyness ln(S/K)", ylabel="hedge ratio", title="Learned hedge ratio (solid) vs Black-Scholes delta (dashed)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, "learned_policy.png"), dpi=150); plt.close(fig)
    # 4 one path
    i = 7; fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    a1.plot(S_base[i], color="k"); a1.axhline(K, color="grey", lw=.8, ls=":"); a1.set_ylabel("spot")
    a2.step(range(p.n_steps), strat["BS delta (flat 20% vol)"][i], where="post", label="BS delta", color="C1")
    a2.step(range(p.n_steps), strat["Deep hedge (MLP)"][i], where="post", label="Deep hedge", color="C3")
    a2.set(xlabel="trading day", ylabel="position"); a2.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "sample_path.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); main(ap.parse_args().out)
