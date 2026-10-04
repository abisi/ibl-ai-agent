"""Joint whisker + no-stim (false-alarm) learning-curve models, exact grid inference.

Trials t = 1..N = all active whisker and no-stim trials in time order. Each trial observes a
lick y_t of one type.

CONTINUOUS model ("g + d"):
    g_t = g_{t-1} + N(0, sigma_g^2)          general lick propensity (logit), drives FA licks
    d_t = d_{t-1} + N(0, sigma_d^2)          whisker-specific drive (logit discrimination)
    no-stim trial:  y_t ~ Bernoulli(logistic(g_t))
    whisker trial:  y_t ~ Bernoulli(logistic(g_t + d_t))
    g_1 ~ N(logit 0.2, 1.5^2), d_1 ~ N(0, 1.5^2); sigma_g, sigma_d on a log-spaced grid (uniform
    prior over the grid), integrated with evidence weights.
  Inference: (g, d) on a 2-D grid; the two walks are independent so the prediction step is
  Tg^T A Td; forward-backward gives exact smoothed marginals of g and d at every trial.

SWITCHING model:
    g_t random walk as above; s_t in {naive, learned}, left-to-right (never back), switch time
    uniform over trials (time-varying hazard h_t = 1 / (N - t + 1)); d_t = d_naive or d_learned
    (constants, on a grid, with the cohort's order constraint: R+ d_learned > d_naive + DMIN,
    R- d_learned < d_naive - DMIN). Posterior over the switch trial; Bayes factor vs a no-switch
    model (d constant, same g).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

G_GRID = np.arange(-7.0, 5.0 + 1e-9, 0.15)
D_GRID = np.arange(-5.0, 9.0 + 1e-9, 0.15)
SIGMAS = np.exp(np.linspace(np.log(0.02), np.log(1.0), 6))
S_GRID = G_GRID[0] + D_GRID[0] + 0.15 * np.arange(len(G_GRID) + len(D_GRID) - 1)   # grid of g + d (whisker logit)
_KSUM = np.arange(len(G_GRID))[:, None] + np.arange(len(D_GRID))[None, :]
# Probability-scale discrimination dp = P(whisker lick) - P(FA lick) on each (g, d) cell, binned
DP_GRID = np.linspace(-1.0, 1.0, 101)
_DP = expit(G_GRID[:, None] + D_GRID[None, :]) - expit(G_GRID)[:, None]
_KDP = np.clip(np.round((_DP + 1.0) / 0.02).astype(int), 0, len(DP_GRID) - 1)
DMIN = 0.5


def _T(grid, sigma):
    d = grid[None, :] - grid[:, None]
    T = np.exp(-0.5 * (d / sigma) ** 2)
    return T / T.sum(1, keepdims=True)


def _prior(grid, mu, sd):
    p = norm.pdf(grid, mu, sd)
    return p / p.sum()


def continuous_fb(y, is_w, sg, sd):
    """Exact forward-backward on the (g, d) grid. Returns (marg_g [N,Gg], marg_d [N,Gd], logZ)."""
    N, Gg, Gd = len(y), len(G_GRID), len(D_GRID)
    Tg, Td = _T(G_GRID, sg), _T(D_GRID, sd)
    pg = expit(G_GRID)                                    # no-stim lick prob depends on g only
    pw = expit(G_GRID[:, None] + D_GRID[None, :])         # whisker lick prob on (g, d)

    def lik(t):
        if is_w[t]:
            return pw if y[t] else 1 - pw
        return np.broadcast_to((pg if y[t] else 1 - pg)[:, None], (Gg, Gd))

    alpha = np.empty((N, Gg, Gd))
    c = np.empty(N)
    a = np.outer(_prior(G_GRID, np.log(0.2 / 0.8), 1.5), _prior(D_GRID, 0.0, 1.5)) * lik(0)
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, N):
        a = (Tg.T @ alpha[t - 1] @ Td) * lik(t)
        c[t] = a.sum()
        alpha[t] = a / c[t]
    mg, md, ms = np.empty((N, Gg)), np.empty((N, Gd)), np.empty((N, Gg + Gd - 1))
    mdp = np.empty((N, len(DP_GRID)))
    beta = np.ones((Gg, Gd))
    for t in range(N - 1, -1, -1):
        if t < N - 1:
            beta = Tg @ (lik(t + 1) * beta) @ Td.T / c[t + 1]
        post = alpha[t] * beta
        post /= post.sum()
        mg[t], md[t] = post.sum(1), post.sum(0)
        ms[t] = np.bincount(_KSUM.ravel(), weights=post.ravel(), minlength=Gg + Gd - 1)   # posterior of g + d
        mdp[t] = np.bincount(_KDP.ravel(), weights=post.ravel(), minlength=len(DP_GRID))  # posterior of dp
    return mg, md, ms, mdp, float(np.log(c).sum())


def fit_continuous(y, is_w):
    """Integrate sigma_g x sigma_d over SIGMAS (uniform prior on the grid)."""
    res, logz = [], []
    for sg in SIGMAS:
        for sd in SIGMAS:
            mg, md, ms, mdp, lz = continuous_fb(y, is_w, sg, sd)
            res.append((sg, sd, mg, md, ms, mdp))
            logz.append(lz)
    logz = np.asarray(logz)
    w = np.exp(logz - logsumexp(logz))
    mg = sum(wi * r[2] for wi, r in zip(w, res))
    md = sum(wi * r[3] for wi, r in zip(w, res))
    ms = sum(wi * r[4] for wi, r in zip(w, res))
    mdp = sum(wi * r[5] for wi, r in zip(w, res))
    sg_post = np.array([[w[i * len(SIGMAS) + j] for j in range(len(SIGMAS))] for i in range(len(SIGMAS))])
    return dict(marg_g=mg, marg_d=md, marg_s=ms, marg_dp=mdp, sigma_weights=sg_post, log_evidence=float(logsumexp(logz) - np.log(len(logz))),
                sigma_g_mean=float(sum(wi * r[0] for wi, r in zip(w, res))),
                sigma_d_mean=float(sum(wi * r[1] for wi, r in zip(w, res))))


def _switch_fb(y, is_w, sg, dn, dl, switch=True):
    """Forward-backward on (g, s). Returns (logZ, P(s_t = learned), P(switch at t))."""
    N, Gg = len(y), len(G_GRID)
    Tg = _T(G_GRID, sg)
    pg = expit(G_GRID)
    pw = np.stack([expit(G_GRID + dn), expit(G_GRID + dl)], 1)      # [Gg, 2]

    def lik(t):
        if is_w[t]:
            return pw if y[t] else 1 - pw
        v = pg if y[t] else 1 - pg
        return np.stack([v, v], 1)

    h = 1.0 / (N - np.arange(N) + 1) if switch else np.zeros(N)       # uniform switch-time prior
    a = np.zeros((Gg, 2))
    a[:, 0] = _prior(G_GRID, np.log(0.2 / 0.8), 1.5)
    if switch:                                                          # switch before trial 1 allowed
        a[:, 1], a[:, 0] = a[:, 0] * h[0], a[:, 0] * (1 - h[0])
    a = a * lik(0)
    alpha = np.empty((N, Gg, 2))
    c = np.empty(N)
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, N):
        pr = Tg.T @ alpha[t - 1]
        pr = np.stack([pr[:, 0] * (1 - h[t]), pr[:, 1] + pr[:, 0] * h[t]], 1)
        a = pr * lik(t)
        c[t] = a.sum()
        alpha[t] = a / c[t]
    logz = float(np.log(c).sum())
    if not switch:
        return logz, None, None
    beta = np.ones((Gg, 2))
    p_learned, p_sw = np.empty(N), np.zeros(N)
    post = alpha[-1] * beta
    p_learned[-1] = post[:, 1].sum() / post.sum()
    for t in range(N - 1, 0, -1):
        lb = lik(t) * beta
        # switch between t-1 and t: s_{t-1}=naive, s_t=learned
        p_sw[t] = float(((Tg.T @ alpha[t - 1][:, 0]) * h[t] * lb[:, 1]).sum() / c[t])
        nb = np.stack([Tg @ (lb[:, 0] * (1 - h[t]) + lb[:, 1] * h[t]), Tg @ lb[:, 1]], 1) / c[t]
        beta = nb
        post = alpha[t - 1] * beta
        p_learned[t - 1] = post[:, 1].sum() / post.sum()
    p_sw[0] = p_learned[0]
    return logz, p_learned, p_sw


D_SWITCH = np.arange(-3.0, 7.0 + 1e-9, 0.5)
SIG_SWITCH = np.exp(np.linspace(np.log(0.02), np.log(1.0), 5))


def fit_switching(y, is_w, rg: int):
    """rg = 1 (R+: d rises) or 0 (R-: d falls). Returns switch posterior, P(learned), BF vs no switch."""
    logz, pl, psw, cfg = [], [], [], []
    for sg in SIG_SWITCH:
        for dn in D_SWITCH:
            for dl in D_SWITCH:
                if (rg == 1 and dl < dn + DMIN) or (rg == 0 and dl > dn - DMIN):
                    continue
                lz, p1, p2 = _switch_fb(y, is_w, sg, dn, dl, True)
                logz.append(lz)
                pl.append(p1)
                psw.append(p2)
                cfg.append((sg, dn, dl))
    logz = np.asarray(logz)
    w = np.exp(logz - logsumexp(logz))
    null = [_switch_fb(y, is_w, sg, d0, d0, False)[0] for sg in SIG_SWITCH for d0 in D_SWITCH]
    cfg = np.asarray(cfg)
    return dict(p_learned=np.tensordot(w, np.stack(pl), 1), p_switch=np.tensordot(w, np.stack(psw), 1),
                log10_bf=float(((logsumexp(logz) - np.log(len(logz))) - (logsumexp(null) - np.log(len(null)))) / np.log(10)),
                d_naive=float(w @ cfg[:, 1]), d_learned=float(w @ cfg[:, 2]), sigma_g=float(w @ cfg[:, 0]))


def marg_summary(marg, grid, qs=(0.1, 0.5, 0.9)):
    cdf = np.cumsum(marg, 1)
    return [grid[np.clip((cdf < q).sum(1), 0, len(grid) - 1)] for q in qs], marg @ grid
