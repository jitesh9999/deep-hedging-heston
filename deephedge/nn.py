"""A small MLP with hand-written backpropagation and an Adam optimiser (NumPy only).

Architecture: x -> tanh(W1 x + b1) -> tanh(W2 h1 + b2) -> sigmoid(W3 h2 + b3)
The sigmoid output keeps the hedge ratio in (0, 1), the correct range for a long-call hedge.
"""
import numpy as np


class MLP:
    def __init__(self, n_in: int, hidden: int, rng: np.random.Generator):
        g = lambda i, o: rng.standard_normal((i, o)) * np.sqrt(1.0 / i)   # LeCun init for tanh
        self.params = {
            "W1": g(n_in, hidden), "b1": np.zeros(hidden),
            "W2": g(hidden, hidden), "b2": np.zeros(hidden),
            "W3": g(hidden, 1) * 0.1, "b3": np.zeros(1),
        }

    def forward(self, X):
        p = self.params
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        z3 = (h2 @ p["W3"] + p["b3"])[:, 0]
        out = 1.0 / (1.0 + np.exp(-z3))
        return out, (X, h1, h2, out)

    def backward(self, cache, d_out):
        """Gradients of a scalar loss w.r.t. all parameters, given d_loss/d_output (shape (M,))."""
        X, h1, h2, out = cache
        p = self.params
        dz3 = (d_out * out * (1.0 - out))[:, None]
        g = {"W3": h2.T @ dz3, "b3": dz3.sum(0)}
        dh2 = dz3 @ p["W3"].T
        dz2 = dh2 * (1.0 - h2 ** 2)
        g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
        dh1 = dz2 @ p["W2"].T
        dz1 = dh1 * (1.0 - h1 ** 2)
        g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
        return g

    def state(self):
        return {k: v.copy() for k, v in self.params.items()}

    def load(self, state):
        self.params = {k: v.copy() for k, v in state.items()}


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)
