"""Black-Scholes call price and delta (r = 0)."""
import numpy as np
from scipy.stats import norm


def bs_price(s, k, tau, sigma):
    s, tau, sigma = np.broadcast_arrays(np.asarray(s, float), np.asarray(tau, float), np.asarray(sigma, float))
    out = np.maximum(s - k, 0.0)
    m = tau > 0
    d1 = (np.log(s[m] / k) + 0.5 * sigma[m] ** 2 * tau[m]) / (sigma[m] * np.sqrt(tau[m]))
    d2 = d1 - sigma[m] * np.sqrt(tau[m])
    out = np.array(out, float)
    out[m] = s[m] * norm.cdf(d1) - k * norm.cdf(d2)
    return out


def bs_delta(s, k, tau, sigma):
    s, tau, sigma = np.broadcast_arrays(np.asarray(s, float), np.asarray(tau, float), np.asarray(sigma, float))
    out = (s > k).astype(float)
    m = tau > 0
    d1 = (np.log(s[m] / k) + 0.5 * sigma[m] ** 2 * tau[m]) / (sigma[m] * np.sqrt(tau[m]))
    out[m] = norm.cdf(d1)
    return out
