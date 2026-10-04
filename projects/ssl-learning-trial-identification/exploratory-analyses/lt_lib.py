"""Shared code for the learning-trial identification project.

Exact (grid) inference for the learning-curve state-space model used in
`beh_plotting_functions.plot_single_mouse_session_learning_curve`:
    x_t = x_{t-1} + N(0, sigma^2),  p_t = invlogit(x_t),  y_t ~ Bernoulli(p_t)
The latent x is discretised on X_GRID and the hidden-Markov forward-backward
recursions give the exact smoothed posterior of p_t given ALL trials, for a
fixed sigma. Uncertainty in sigma is integrated over a grid of precisions
tau = 1/sigma^2 with prior weights, using each tau's marginal likelihood
(product of forward normalisers) -> exact Bayesian posterior of the same
model, no MCMC noise. Two priors on tau:
  'orig' : tau ~ Gamma(10, 10)  (the stored model's prior; sigma ~ 1)
  'eb'   : log-uniform sigma in [0.02, 1.5] (data decide the smoothness;
           "empirical-Bayes-like" alternative curve)
The initial state keeps the stored model's N(0, 100^2) prior (flat on the grid).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

X_GRID = np.linspace(-9.0, 9.0, 181)
P_GRID = 1.0 / (1.0 + np.exp(-X_GRID))
N_TAU = 36


def tau_grid(prior: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (tau values, log prior weights) -- weights include the grid
    cell mass so that sum(exp(logw)) ~ 1."""
    if prior == "orig":
        q = (np.arange(N_TAU) + 0.5) / N_TAU
        taus = gamma.ppf(q, a=10, scale=1 / 10)
        return taus, np.full(N_TAU, -np.log(N_TAU))
    if prior == "eb":
        sig = np.exp(np.linspace(np.log(0.02), np.log(1.5), N_TAU))
        return 1 / sig**2, np.full(N_TAU, -np.log(N_TAU))
    raise ValueError(prior)


def _transition(sigma: float) -> np.ndarray:
    d = X_GRID[None, :] - X_GRID[:, None]
    T = np.exp(-0.5 * (d / sigma) ** 2)
    return T / T.sum(1, keepdims=True)


def forward_backward(y: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    """Smoothed marginals gamma[t, g] and log marginal likelihood."""
    n, G = len(y), len(X_GRID)
    T = _transition(sigma)
    lik = np.where(y[:, None] == 1, P_GRID[None, :], 1 - P_GRID[None, :])
    alpha = np.empty((n, G))
    c = np.empty(n)
    a = np.full(G, 1.0 / G) * lik[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ T) * lik[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.ones(G)
    gam = np.empty((n, G))
    gam[-1] = alpha[-1]
    for t in range(n - 2, -1, -1):
        beta = T @ (lik[t + 1] * beta) / c[t + 1]
        g = alpha[t] * beta
        gam[t] = g / g.sum()
    return gam, float(np.log(c).sum())


def fit_curve(y: np.ndarray, prior: str) -> dict:
    """Exact posterior over p_t, integrating sigma. Returns marginals
    (n x G), mean, quantile function helpers and sigma posterior."""
    y = np.asarray(y).astype(int)
    taus, logw = tau_grid(prior)
    gams, logz = [], []
    for tau in taus:
        g, lz = forward_backward(y, 1 / np.sqrt(tau))
        gams.append(g)
        logz.append(lz)
    lp = np.asarray(logz) + logw
    w = np.exp(lp - lp.max())
    w /= w.sum()
    marg = np.tensordot(w, np.stack(gams), axes=1)
    sig = 1 / np.sqrt(taus)
    return dict(marg=marg, p_mean=marg @ P_GRID, sigma_post_mean=float(w @ sig), sigma_grid=sig, sigma_weights=w,
                log_evidence=float(np.log(np.exp(lp - lp.max()).sum()) + lp.max()))


def quantiles(marg: np.ndarray, qs) -> list[np.ndarray]:
    cdf = np.cumsum(marg, axis=1)
    return [P_GRID[np.clip((cdf < q).sum(1), 0, len(P_GRID) - 1)] for q in qs]


def interp_marginals(marg: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear-in-time mixture of the source trials' marginals at t_dst
    (edges held constant) -- FA posterior at whisker trial times."""
    idx = np.searchsorted(t_src, t_dst)
    out = np.empty((len(t_dst), marg.shape[1]))
    for k, (i, t) in enumerate(zip(idx, t_dst)):
        if i <= 0:
            out[k] = marg[0]
        elif i >= len(t_src):
            out[k] = marg[-1]
        else:
            f = (t - t_src[i - 1]) / (t_src[i] - t_src[i - 1])
            out[k] = (1 - f) * marg[i - 1] + f * marg[i]
    return out


def prob_greater(marg_a: np.ndarray, marg_b: np.ndarray) -> np.ndarray:
    """Per trial P(p_a > p_b) for independent grid marginals."""
    cdf_b_strict = np.concatenate([np.zeros((marg_b.shape[0], 1)), np.cumsum(marg_b, axis=1)[:, :-1]], axis=1)
    return np.sum(marg_a * cdf_b_strict, axis=1)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out
