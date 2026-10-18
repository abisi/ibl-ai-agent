"""rl2_model -- redesigned learning model (v2, 2026-10-18; user: "fix the model design ... the model is to fit behavioural data to see what
type of learning strategy mice use, explaining both auditory-to-whisker learning and the R+ / R- difference; the latter should be just a
design choice ... keep R- whisker lick as -1").

Design (one learning rule for every mouse; cohorts differ ONLY in the outcome of a whisker lick):
  Features of a trial: bias (1 on every trial), stimulus-present S (1 on auditory and whisker trials), auditory A, whisker W.
  The no-stimulus trial is the bias alone: a separate no-stimulus weight is redundant with the bias in the original design (variant
  "+NS weight" tests it).
  Lick value      V = w_b + gamma * w_S + (1 - gamma) * w_X      (X = A or W on stimulus trials; V = w_b on no-stimulus trials)
  Choice          P(lick) = eps / 2 + (1 - eps) * sigmoid(beta * V)
  Outcome         observed only when the mouse licks: r = +1 auditory lick; whisker lick +1 (R+) or -1 (R-); no-stimulus lick -1
  Learning        delta = r - V;  alpha = alpha_pos if delta > 0 else alpha_neg
                  w_b += alpha * kappa * delta;  w_S += alpha * gamma * delta;  w_X += alpha * (1 - gamma) * delta
  Initial state   w_b = w0 (lick tendency before pretraining), w_S = w_A = w_W = 0 (no stimulus knowledge)
Strategy parameters:
  gamma  generalisation: share of stimulus learning credited to the generic "a stimulus was presented" feature (1 = everything learnt
         about the tone transfers to the whisker stimulus at day 0; 0 = stimulus-specific, no transfer)
  kappa  how much each outcome also changes the general lick tendency (bias learning; 0 = none)
  alpha_pos / alpha_neg  learning from better / worse than expected outcomes
  beta, eps  choice determinism and lapses
Identifiability: while V stays far from the outcome values (+-1), only beta x alpha is constrained (a first check on 11 mice put beta at
its bound with tiny alphas, and fixing beta cost up to 75 log-likelihood units in some mice). The strategy quantities reported are
therefore scale-free: gamma, kappa (credit-assignment shares), alpha_neg / alpha_pos, and the effective learning speed beta x alpha_pos.
Fitted by maximum likelihood of each mouse's own sequence of trials and licks (all pretraining and whisker days), not by MSE on smoothed
curves; every quantity is per trial, so the latents can be aligned with neural data (session_id, trial_idx).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize

PARAMS = ["alpha_pos", "alpha_neg", "beta", "gamma", "kappa", "w0", "eps", "alpha_ns"]
BOUNDS = dict(alpha_pos=(1e-4, 1.0), alpha_neg=(1e-4, 1.0), beta=(0.1, 100.0), gamma=(0.0, 1.0), kappa=(0.0, 3.0), w0=(-2.0, 2.0),
              eps=(1e-4, 0.5), alpha_ns=(0.0, 1.0))
LOG = {"alpha_pos", "alpha_neg", "beta"}                 # sampled log-uniformly for the random starts
DEFAULT = dict(alpha_pos=0.05, alpha_neg=0.05, beta=3.0, gamma=0.5, kappa=1.0, w0=-0.3, eps=0.05, alpha_ns=0.0)
# variants: free parameters; the others are fixed to the given value (or tied, for "symmetric")
VARIANTS = {
    "full":        dict(free=["alpha_pos", "alpha_neg", "beta", "gamma", "kappa", "w0", "eps"], fixed=dict(alpha_ns=0.0)),
    "specific":    dict(free=["alpha_pos", "alpha_neg", "beta", "kappa", "w0", "eps"], fixed=dict(gamma=0.0, alpha_ns=0.0)),
    "general":     dict(free=["alpha_pos", "alpha_neg", "beta", "kappa", "w0", "eps"], fixed=dict(gamma=1.0, alpha_ns=0.0)),
    "no_bias":     dict(free=["alpha_pos", "alpha_neg", "beta", "gamma", "w0", "eps"], fixed=dict(kappa=0.0, alpha_ns=0.0)),
    "symmetric":   dict(free=["alpha_pos", "beta", "gamma", "kappa", "w0", "eps"], fixed=dict(alpha_ns=0.0), tie=dict(alpha_neg="alpha_pos")),
    "ns_weight":   dict(free=["alpha_pos", "alpha_neg", "beta", "gamma", "kappa", "w0", "eps", "alpha_ns"], fixed={}),
}
TT = {"auditory": 0, "whisker": 1, "no_stim": 2}


@njit(cache=True)
def _run(tt, lick, r, ap, an, beta, gamma, kappa, w0, eps, ans, out):
    """tt: 0 auditory, 1 whisker, 2 no stimulus; lick 0/1; r: outcome of a lick on that trial. Returns the log-likelihood; if out has rows,
    writes per trial: P(lick), V, w_b, w_S, w_A, w_W, w_N, delta (state BEFORE the trial's update; delta NaN without a lick)."""
    wb, ws, wa, ww, wn = w0, 0.0, 0.0, 0.0, 0.0
    ll = 0.0
    keep = out.shape[0] > 0
    for t in range(tt.shape[0]):
        k = tt[t]
        if k == 0:
            v = wb + gamma * ws + (1.0 - gamma) * wa
        elif k == 1:
            v = wb + gamma * ws + (1.0 - gamma) * ww
        else:
            v = wb + wn
        p = eps / 2.0 + (1.0 - eps) / (1.0 + np.exp(-beta * v))
        if p < 1e-12:
            p = 1e-12
        if p > 1.0 - 1e-12:
            p = 1.0 - 1e-12
        if keep:
            out[t, 0] = p; out[t, 1] = v; out[t, 2] = wb; out[t, 3] = ws; out[t, 4] = wa; out[t, 5] = ww; out[t, 6] = wn; out[t, 7] = np.nan
        if lick[t] == 1:
            ll += np.log(p)
            d = r[t] - v
            a = ap if d > 0 else an
            if k == 2:
                wb += a * kappa * d
                wn += ans * d
            else:
                wb += a * kappa * d
                ws += a * gamma * d
                if k == 0:
                    wa += a * (1.0 - gamma) * d
                else:
                    ww += a * (1.0 - gamma) * d
            if keep:
                out[t, 7] = d
        else:
            ll += np.log(1.0 - p)
    return ll


@njit(cache=True)
def _simulate(tt, r, ap, an, beta, gamma, kappa, w0, eps, ans, u):
    """generative run on a fixed trial sequence: licks drawn from the model (u: uniform draws); returns P(lick) per trial and the licks"""
    wb, ws, wa, ww, wn = w0, 0.0, 0.0, 0.0, 0.0
    n = tt.shape[0]
    P = np.empty(n); L = np.zeros(n, np.int64)
    for t in range(n):
        k = tt[t]
        if k == 0:
            v = wb + gamma * ws + (1.0 - gamma) * wa
        elif k == 1:
            v = wb + gamma * ws + (1.0 - gamma) * ww
        else:
            v = wb + wn
        p = eps / 2.0 + (1.0 - eps) / (1.0 + np.exp(-beta * v))
        P[t] = p
        if u[t] < p:
            L[t] = 1
            d = r[t] - v
            a = ap if d > 0 else an
            if k == 2:
                wb += a * kappa * d
                wn += ans * d
            else:
                wb += a * kappa * d
                ws += a * gamma * d
                if k == 0:
                    wa += a * (1.0 - gamma) * d
                else:
                    ww += a * (1.0 - gamma) * d
    return P, L


def outcomes(tt, cohort):
    """outcome of a lick per trial: auditory +1, whisker +1 (R+) / -1 (R-), no stimulus -1 (the only cohort-dependent quantity)"""
    return np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if cohort == "R+" else -1.0, -1.0))


def full_params(variant, x):
    v = VARIANTS[variant]
    p = {**DEFAULT, **v["fixed"], **dict(zip(v["free"], x))}
    for k, src in v.get("tie", {}).items():
        p[k] = p[src]
    return p


def _args(p):
    return p["alpha_pos"], p["alpha_neg"], p["beta"], p["gamma"], p["kappa"], p["w0"], p["eps"], p["alpha_ns"]


def loglik(p, tt, lick, r):
    return _run(tt, lick, r, *_args(p), np.empty((0, 8)))


def trace(p, tt, lick, r):
    out = np.empty((len(tt), 8))
    _run(tt, lick, r, *_args(p), out)
    return out


def simulate(p, tt, r, seed=0):
    return _simulate(tt, r, *_args(p), np.random.default_rng(seed).random(len(tt)))


def starts(variant, n, seed=0):
    rng = np.random.default_rng(seed)
    free = VARIANTS[variant]["free"]
    S = [[DEFAULT[k] if k != "alpha_ns" else 0.05 for k in free]]
    for _ in range(n - 1):
        S.append([np.exp(rng.uniform(np.log(BOUNDS[k][0]), np.log(BOUNDS[k][1]))) if k in LOG else rng.uniform(*BOUNDS[k]) for k in free])
    return np.array(S)


def fit(variant, data, n_starts=10, seed=0, init=None):
    """data: list of (tt, lick, r) sequences fitted with ONE parameter set (one mouse, or a pooled group). init: optional full parameter
    dict (e.g. another variant's optimum) added as a start, projected on this variant's free parameters and clipped to the bounds, so
    that a variant containing another as a special case cannot end worse than it. Returns dict(params, nll, k, n, ...)."""
    free = VARIANTS[variant]["free"]
    bounds = [BOUNDS[k] for k in free]

    def nll(x):
        p = full_params(variant, x)
        return -sum(loglik(p, *d) for d in data)
    best = None
    S = starts(variant, n_starts, seed)
    if init is not None:
        x_init = [float(np.clip(init.get(k, DEFAULT[k]) if k != "alpha_ns" else init.get(k, 0.0), *BOUNDS[k])) for k in free]
        S = np.vstack([S, x_init])
    for x0 in S:
        r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    n = sum(len(d[0]) for d in data)
    return dict(params=full_params(variant, best.x), nll=float(best.fun), k=len(free), n=n, converged=bool(best.success),
                bic=float(2 * best.fun + len(free) * np.log(n)), aic=float(2 * best.fun + 2 * len(free)))
