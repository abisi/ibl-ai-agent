"""rl5_model -- model v4 (model_v4_design.md, 2026-10-23): stimulus network S / A / W -> lick with generalisation gamma, thirst, an
exploration term for false alarms combined with the stimulus drive as an OR, asymmetric learning rates and between-session consolidation.

One kernel runs every variant (log-likelihood, latents, or simulation). Variants:
  FA mode   E0 constant eps0*T | E1 decay with training trials | E2 decay after FA licks | E3 recent reward rate | B learned bias (v2-like)
  lr        sym (alpha- tied to alpha+) | asym
  ts        1ts (k_c fixed 1) | 2ts (k_c free: w_slow += k_c * w_fast at each session boundary, w_fast reset)
Sequence arrays (one row per trial, mice concatenated, each mouse ordered by day and start time):
  tt 0 auditory / 1 whisker / 2 no stimulus, lick 0/1, rplus 0/1, sess_start 0/1, mouse_start 0/1
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize

P_ALL = ["alpha_pos", "alpha_neg", "beta", "gamma", "c", "rho", "k_c", "eps0", "eps_inf", "tau", "alpha_eps", "eta", "alpha_R", "kappa"]
BOUNDS = dict(alpha_pos=(1e-4, 1.0), alpha_neg=(1e-4, 1.0), beta=(0.1, 30.0), gamma=(0.0, 1.0), c=(-5.0, 5.0), rho=(0.0, 0.05),
              k_c=(0.0, 2.0), eps0=(0.005, 0.8), eps_inf=(0.005, 0.8), tau=(50.0, 30000.0), alpha_eps=(1e-4, 1.0), eta=(0.0, 1.0),
              alpha_R=(1e-3, 1.0), kappa=(0.0, 3.0))
FIXED = dict(alpha_pos=0.05, alpha_neg=0.05, beta=3.0, gamma=0.0, c=0.0, rho=0.0, k_c=1.0, eps0=0.1, eps_inf=0.1, tau=1000.0,
             alpha_eps=0.0, eta=0.0, alpha_R=0.1, kappa=0.0)
DEFAULT = dict(alpha_pos=0.02, alpha_neg=0.05, beta=4.0, gamma=0.3, c=0.5, rho=0.003, k_c=1.0, eps0=0.3, eps_inf=0.1, tau=2000.0,
               alpha_eps=0.01, eta=0.3, alpha_R=0.05, kappa=0.5)
LOG = {"alpha_pos", "alpha_neg", "beta", "tau", "alpha_eps", "alpha_R"}
LAPSE = 0.02
FA_MODES = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "B": 4}
FA_PARAMS = {"E0": ["eps0"], "E1": ["eps0", "eps_inf", "tau"], "E2": ["eps0", "eps_inf", "alpha_eps"], "E3": ["eps_inf", "eta", "alpha_R"],
             "B": ["eps0", "kappa"]}
CORE = ["alpha_pos", "beta", "gamma", "c", "rho"]
LATENTS = ["p_lick", "V", "w_S", "w_A", "w_W", "w_b", "eps", "thirst", "drive", "delta"]


def model_names():
    return [f"{fa}-{lr}-{ts}" for fa in FA_MODES for lr in ("sym", "asym") for ts in ("1ts", "2ts")]


def free_params(name):
    fa, lr, ts = name.split("-")
    p = CORE + (["alpha_neg"] if lr == "asym" else []) + (["k_c"] if ts == "2ts" else []) + FA_PARAMS[fa]
    return p


def full_vector(name, x):
    """free values -> full parameter vector (P_ALL order) and dict"""
    p = dict(FIXED)
    p.update(dict(zip(free_params(name), x)))
    if name.split("-")[1] == "sym":
        p["alpha_neg"] = p["alpha_pos"]
    return np.array([p[k] for k in P_ALL], float), p


@njit(cache=True)
def kernel(tt, lick, rplus, sess_start, mouse_start, P, fa_mode, mask, u, simulate, out):
    """Returns the log-likelihood over trials with mask == 1. simulate: licks drawn from u (lick overwritten in place).
    out (n x 10 or 0 rows): latents BEFORE the trial's update (LATENTS order)."""
    ap, an, beta, gamma, c, rho, kc = P[0], P[1], P[2], P[3], P[4], P[5], P[6]
    eps0, einf, tau, aeps, eta, aR, kappa = P[7], P[8], P[9], P[10], P[11], P[12], P[13]
    keep = out.shape[0] > 0
    sf = np.zeros(4)          # fast: S, A, W, b
    ss = np.zeros(4)          # slow
    T = 1.0; Rbar = 0.0; eps_state = eps0; n = 0.0; ll = 0.0
    for t in range(tt.shape[0]):
        if mouse_start[t] == 1:
            sf[:] = 0.0; ss[:] = 0.0; eps_state = eps0; n = 0.0; T = 1.0; Rbar = 0.0
        elif sess_start[t] == 1:
            for j in range(4):
                ss[j] += kc * sf[j]
                sf[j] = 0.0
            T = 1.0; Rbar = 0.0
        k = tt[t]
        wS = sf[0] + ss[0]; wA = sf[1] + ss[1]; wW = sf[2] + ss[2]; wb = sf[3] + ss[3]
        if k < 2:
            wX = wA if k == 0 else wW
            V = gamma * wS + (1.0 - gamma) * wX
        else:
            V = 0.0
        if fa_mode == 4:
            V += wb
        if k < 2 or fa_mode == 4:
            x = beta * (T * V - c)
            drive = 1.0 / (1.0 + np.exp(-x)) if x > -50.0 else 0.0
        else:
            drive = 0.0
        if fa_mode == 1:
            eps = (einf + (eps0 - einf) * np.exp(-n / tau)) * T
        elif fa_mode == 2:
            eps = eps_state * T
        elif fa_mode == 3:
            eps = min(einf + eta * Rbar, 0.99)
        else:
            eps = eps0 * T
        p = (1.0 - 0.02) * (1.0 - (1.0 - eps) * (1.0 - drive))
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        if simulate:
            lick[t] = 1 if u[t] < p else 0
        l = lick[t]
        if mask[t] == 1:
            ll += np.log(p) if l == 1 else np.log(1.0 - p)
        d = np.nan
        water = 0.0
        if l == 1:
            if k == 0:
                o = 1.0; water = 1.0
            elif k == 1:
                o = 1.0 if rplus[t] == 1 else -1.0; water = 1.0 if rplus[t] == 1 else 0.0
            else:
                o = -0.5
            if k < 2:
                d = o - V
                a = ap if d > 0 else an
                sf[0] += a * gamma * d
                sf[1 + k] += a * (1.0 - gamma) * d
                if fa_mode == 4:
                    sf[3] += a * kappa * d
            else:
                if fa_mode == 4:
                    d = o - V
                    a = ap if d > 0 else an
                    sf[3] += a * kappa * d
                elif fa_mode == 2:
                    eps_state = einf + (eps_state - einf) * (1.0 - aeps)
            T = max(0.0, T - rho * water)
        if keep:
            out[t, 0] = p; out[t, 1] = V; out[t, 2] = wS; out[t, 3] = wA; out[t, 4] = wW; out[t, 5] = wb; out[t, 6] = eps
            out[t, 7] = T; out[t, 8] = drive; out[t, 9] = d
        if fa_mode == 3:
            Rbar += aR * (water - Rbar)
        n += 1.0
    return ll


def _transform(name):
    fp = free_params(name)
    lo = np.array([BOUNDS[k][0] for k in fp]); hi = np.array([BOUNDS[k][1] for k in fp])
    islog = np.array([k in LOG for k in fp])
    return fp, lo, hi, islog


def loglik(name, x, A, mask):
    P, _ = full_vector(name, x)
    z = np.empty(0); out = np.empty((0, 10))
    return kernel(*A, P, FA_MODES[name.split("-")[0]], mask, z, False, out)


def latents(name, p, A):
    P = np.array([p[k] for k in P_ALL], float)
    out = np.empty((len(A[0]), 10)); z = np.empty(0)
    kernel(*A, P, FA_MODES[name.split("-")[0]], np.ones(len(A[0]), np.int64), z, False, out)
    return out


def simulate(name, p, A, rng):
    tt, lick, rplus, ss, ms = A
    lick = lick.copy()
    P = np.array([p[k] for k in P_ALL], float)
    out = np.empty((len(tt), 10))
    kernel(tt, lick, rplus, ss, ms, P, FA_MODES[name.split("-")[0]], np.ones(len(tt), np.int64), rng.uniform(size=len(tt)), True, out)
    return lick, out


def fit(name, A, mask, rng, n_random=6, extra_starts=()):
    """Bounded multi-start L-BFGS-B in an unconstrained-ish space (log for rates / beta / tau). Returns (param dict, nll)."""
    fp, lo, hi, islog = _transform(name)

    def to_x(y):
        return np.where(islog, np.exp(y), y)

    def nll(y):
        v = -loglik(name, to_x(y), A, mask)
        return v if np.isfinite(v) else 1e12
    blo = np.where(islog, np.log(np.where(islog, lo, 1.0)), lo); bhi = np.where(islog, np.log(np.where(islog, hi, 1.0)), hi)
    starts = [[DEFAULT[k] for k in fp]] + [list(s) for s in extra_starts]
    for _ in range(n_random):
        starts.append([float(np.exp(rng.uniform(np.log(BOUNDS[k][0]), np.log(BOUNDS[k][1])))) if k in LOG else rng.uniform(*BOUNDS[k])
                       for k in fp])
    best = None
    for s in starts:
        s = np.clip(s, lo, hi)
        y0 = np.clip(np.where(islog, np.log(np.where(islog, s, 1.0)), s), blo, bhi)
        r = minimize(nll, y0, method="L-BFGS-B", bounds=list(zip(blo, bhi)), options=dict(maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    return full_vector(name, to_x(best.x))[1], float(best.fun)


def arrays(d):
    """sequence table (mouse_id, session_id, trial_type, lick_flag, reward_group; ordered) -> kernel arrays"""
    tt = d.trial_type.map({"auditory": 0, "whisker": 1, "no_stim": 2}).to_numpy(np.int64)
    lick = d.lick_flag.to_numpy(np.int64).copy()
    rplus = (d.reward_group == "R+").to_numpy(np.int64)
    ms = (d.mouse_id != d.mouse_id.shift()).to_numpy(np.int64)
    ss = (d.session_id != d.session_id.shift()).to_numpy(np.int64)
    return tt, lick, rplus, ss, ms
