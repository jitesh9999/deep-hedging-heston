"""Train the hedging network on freshly simulated Heston paths every step (no overfitting to a fixed set)."""
import argparse, json, time
import numpy as np
from .blackscholes import bs_price
from .heston import HestonParams, simulate
from .hedging import cvar_loss_and_grad, features
from .nn import MLP, Adam


def train(steps=1500, batch=4096, hidden=64, cost=0.001, alpha=0.95, lr=3e-3, seed=0, log_every=50):
    p = HestonParams()
    premium = float(bs_price(p.s0, 1.0, p.T, np.sqrt(p.theta)))
    rng = np.random.default_rng(seed)
    net = MLP(3, hidden, rng)
    net.params["b3"][:] = 0.0
    opt = Adam(net.params, lr=lr)
    hist, t0 = [], time.time()
    for it in range(steps):
        S, _ = simulate(p, batch, rng)
        X = features(S, p)
        out, cache = net.forward(X)
        delta = out.reshape(batch, p.n_steps)
        loss, d_delta, _ = cvar_loss_and_grad(S, delta, cost, premium, alpha)
        grads = net.backward(cache, d_delta.reshape(-1))
        cur_lr = lr * (0.5 * (1 + np.cos(np.pi * it / steps)) * 0.95 + 0.05)   # cosine decay
        opt.step(net.params, grads, lr=cur_lr)
        hist.append(float(loss))
        if it % log_every == 0 or it == steps - 1:
            print(f"step {it:5d}  CVaR{int(alpha*100)} loss {loss:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return net, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    net, hist = train(a.steps, a.batch, a.hidden, a.cost, seed=a.seed)
    np.savez(os.path.join(a.out, "model.npz"), **net.state())
    json.dump({"history": hist, "cost": a.cost, "hidden": a.hidden}, open(os.path.join(a.out, "train_log.json"), "w"))
