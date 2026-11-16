"""rl8_hier -- hierarchical (mouse-level) fitting of rl7_model (model v7, model_v7_design.md, 2026-10-23).
Per cohort: mouse-level parameters y_m (each parameter mapped to the real line by a scaled logit of its bounded, log where rl7 LOG)
with population prior y_m ~ N(mu, diag sd^2); cohort-level parameters fixed at the cohort fit (023); session-level rho_s integrated out
inside rl7 (around the mouse's mu_rho). EM with Laplace approximation (Huys et al. 2011):
  E-step  per mouse MAP of -loglik_m(y) + 0.5 sum((y - mu) / sd)^2, posterior covariance from the numerical Hessian
  M-step  mu = mean(y_hat), sd^2 = mean(y_hat^2 + diag(Sigma_m)) - mu^2 (floor 0.05)
Per-mouse Laplace evidence log p(D_m | mu, sd) for model comparison per mouse; iBIC = -2 sum_m log p(D_m) + 2 d log(n_trials).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import rl7_model as M

MOUSE_BASE = ["alpha_pos", "alpha_neg", "beta", "c", "s_B", "mu_rho"]
SD_FLOOR = 0.05
Y_LIM = 12.0


COHORT_KEYS = None   # lean v7b (027): {cohort: mouse-level parameters}, overrides MOUSE_BASE


def mouse_params(name, cs=None):
    if COHORT_KEYS is not None:
        return COHORT_KEYS[cs]
    return MOUSE_BASE + (["theta"] if name.split("-")[1] == "thr" else [])


def _u(k, x):
    lo, hi = M.BOUNDS[k]
    return (np.log(x), np.log(lo), np.log(hi)) if k in M.LOG else (x, lo, hi)


def to_y(keys, x):
    out = []
    for k, v in zip(keys, x):
        u, lo, hi = _u(k, max(v, 1e-12) if k in M.LOG else v)
        f = np.clip((u - lo) / (hi - lo), 1e-5, 1 - 1e-5)
        out.append(np.log(f / (1 - f)))
    return np.array(out)


def to_x(keys, y):
    out = []
    for k, v in zip(keys, y):
        f = 1.0 / (1.0 + np.exp(-v))
        _, lo, hi = _u(k, M.BOUNDS[k][0])
        u = lo + f * (hi - lo)
        out.append(float(np.exp(u)) if k in M.LOG else float(u))
    return np.array(out)


def loglik(name, keys, y, cohort_p, A):
    p = dict(cohort_p)
    p.update(dict(zip(keys, to_x(keys, y))))
    P = np.array([p.get(k, M.FIXED[k]) for k in M.P_ALL], float)
    lr, uo, lw = M.nodes(name, P)
    e = np.empty((0, 0))
    return M.kernel(*A, P, *M._modes(name), lr, uo, lw, np.ones(len(A[0]), np.int64), False, e, e, e)


def hessian(f, y, h=1e-3):
    d = len(y)
    H = np.empty((d, d))
    f0 = f(y)
    E = np.eye(d) * h
    fp = np.array([f(y + E[i]) for i in range(d)]); fm = np.array([f(y - E[i]) for i in range(d)])
    for i in range(d):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / h ** 2
        for j in range(i + 1, d):
            H[i, j] = H[j, i] = (f(y + E[i] + E[j]) - fp[i] - fp[j] + f0) / h ** 2
    w, V = np.linalg.eigh((H + H.T) / 2)
    return (V * np.maximum(w, 1e-3)) @ V.T


def map_mouse(name, A, cohort_p, mu, sd, starts, cs=None):
    keys = mouse_params(name, cs)

    def f(y):
        v = -loglik(name, keys, y, cohort_p, A) + 0.5 * np.sum(((y - mu) / sd) ** 2)
        return v if np.isfinite(v) else 1e12
    best = None
    for s in starts:
        r = minimize(f, np.clip(s, -Y_LIM, Y_LIM), method="L-BFGS-B", bounds=[(-Y_LIM, Y_LIM)] * len(keys), options=dict(maxiter=500))
        if best is None or r.fun < best.fun:
            best = r
    y = best.x
    H = hessian(f, y)
    Sig = np.linalg.inv(H)
    ll = loglik(name, keys, y, cohort_p, A)
    logprior = -0.5 * np.sum(((y - mu) / sd) ** 2) - np.sum(np.log(sd)) - 0.5 * len(y) * np.log(2 * np.pi)
    # Laplace: log p(D_m) = ll + log prior + d/2 log 2pi + 1/2 log |Sigma|
    log_ev = ll + logprior + 0.5 * len(y) * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(Sig)[1]
    return dict(y=y, Sig=Sig, ll=ll, log_ev=log_ev)


def m_step(Y, Sigs):
    mu = Y.mean(0)
    var = (Y ** 2 + np.array([np.diag(S) for S in Sigs])).mean(0) - mu ** 2
    return mu, np.sqrt(np.maximum(var, SD_FLOOR ** 2))
