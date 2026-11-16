"""rl6_model -- model v5 (model_v5_design.md, 2026-10-23): v4 with (1) one interpretable auditory -> whisker transfer parameter g and
(2) a within-session engagement factor on all licking, separate from the false-alarm exploration term (user: "yes go").

Stimulus nodes A, W -> lick (no shared S node). Transfer variants:
  init     V(A) = w_A, V(W) = w_W; at the mouse's first whisker trial w_W <- g * w_A (once); afterwards independent learning
  overlap  overlapping inputs: auditory x = (1, g), whisker x = (g, 1) on (A, W); V = w . x; update w += alpha * delta * x
           (ongoing generalisation in both directions; V(W) at whisker onset = w_W + g * w_A)
Choice:  P(lick) = (1 - lapse) * e_t * [1 - (1 - eps_t) * (1 - sigmoid(beta * (T_t * V - c)))]   (drive 0 on no-stimulus trials)
  e_t  engagement (variant 'eng'): e_t = e_min + (1 - e_min) * (1 - exp(-Rbar_t / r_e)); without engagement e_t = 1
  Rbar recent reward rate, Rbar <- Rbar + alpha_R * (water_t - Rbar) every trial, reset at session start to R0 (eng) or 0 (no eng)
  T_t  thirst, 1 - rho * rewards so far in the session; eps_t as rl5 (E0 constant, E1 decay with trials, E2 decay after FA licks,
       E3 eps_inf + eta * Rbar); E0-E2 scaled by T
Learning on licked stimulus trials: delta = o - V(s), alpha = alpha+ / alpha- (sym: tied). Outcomes: auditory +1, whisker +1 (R+) /
-1 (R-), FA -0.5 (used by E2 only through the lick, no weight). One timescale (v4: 2ts added little).
Model name: <transfer>-<FA>-<eng|noeng>-<sym|asym>
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize

P_ALL = ["alpha_pos", "alpha_neg", "beta", "g", "c", "rho", "eps0", "eps_inf", "tau", "alpha_eps", "eta", "alpha_R", "e_min", "r_e", "R0"]
BOUNDS = dict(alpha_pos=(1e-4, 1.0), alpha_neg=(1e-4, 1.0), beta=(0.1, 30.0), g=(0.0, 1.0), c=(-5.0, 5.0), rho=(0.0, 0.05),
              eps0=(0.005, 0.8), eps_inf=(0.005, 0.8), tau=(50.0, 30000.0), alpha_eps=(1e-4, 1.0), eta=(0.0, 1.0), alpha_R=(1e-3, 1.0),
              e_min=(0.0, 1.0), r_e=(0.01, 2.0), R0=(0.0, 1.0))
FIXED = dict(alpha_pos=0.05, alpha_neg=0.05, beta=3.0, g=0.0, c=0.0, rho=0.0, eps0=0.1, eps_inf=0.1, tau=1000.0, alpha_eps=0.0, eta=0.0,
             alpha_R=0.1, e_min=1.0, r_e=1.0, R0=0.0)
DEFAULT = dict(alpha_pos=0.02, alpha_neg=0.05, beta=4.0, g=0.5, c=0.5, rho=0.003, eps0=0.3, eps_inf=0.1, tau=2000.0, alpha_eps=0.01,
               eta=0.3, alpha_R=0.1, e_min=0.3, r_e=0.2, R0=0.5)
LOG = {"alpha_pos", "alpha_neg", "beta", "tau", "alpha_eps", "alpha_R", "r_e"}
TRANSFER = {"init": 0, "overlap": 1}
FA_MODES = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}
FA_PARAMS = {"E0": ["eps0"], "E1": ["eps0", "eps_inf", "tau"], "E2": ["eps0", "eps_inf", "alpha_eps"], "E3": ["eps_inf", "eta", "alpha_R"]}
ENG_PARAMS = ["e_min", "r_e", "alpha_R", "R0"]
CORE = ["alpha_pos", "beta", "g", "c", "rho"]
LATENTS = ["p_lick", "V", "V_A", "V_W", "w_A", "w_W", "eps", "thirst", "drive", "eng", "delta"]


def model_names():
    return [f"{tr}-{fa}-{en}-{lr}" for tr in TRANSFER for fa in FA_MODES for en in ("noeng", "eng") for lr in ("sym", "asym")]


def parts(name):
    tr, fa, en, lr = name.split("-")
    return TRANSFER[tr], FA_MODES[fa], int(en == "eng"), lr


def free_params(name):
    tr, fa, en, lr = name.split("-")
    p = CORE + (["alpha_neg"] if lr == "asym" else []) + FA_PARAMS[fa]
    if en == "eng":
        p += [k for k in ENG_PARAMS if k not in p]
    return p


def full_vector(name, x):
    p = dict(FIXED)
    p.update(dict(zip(free_params(name), x)))
    if name.endswith("-sym"):
        p["alpha_neg"] = p["alpha_pos"]
    return np.array([p[k] for k in P_ALL], float), p


@njit(cache=True)
def kernel(tt, lick, rplus, sess_start, mouse_start, P, transfer, fa_mode, eng_on, mask, u, simulate, out):
    ap, an, beta, g, c, rho = P[0], P[1], P[2], P[3], P[4], P[5]
    eps0, einf, tau, aeps, eta, aR, emin, re, R0 = P[6], P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14]
    keep = out.shape[0] > 0
    wA = 0.0; wW = 0.0; T = 1.0; Rbar = 0.0; eps_state = eps0; n = 0.0; ll = 0.0; seen_w = False
    for t in range(tt.shape[0]):
        if mouse_start[t] == 1:
            wA = 0.0; wW = 0.0; eps_state = eps0; n = 0.0; seen_w = False
        if mouse_start[t] == 1 or sess_start[t] == 1:
            T = 1.0
            Rbar = R0 if eng_on == 1 else 0.0
        k = tt[t]
        if transfer == 0 and k == 1 and not seen_w:
            wW = g * wA
        if k == 1:
            seen_w = True
        if transfer == 1:
            VA = wA + g * wW; VW = wW + g * wA
        else:
            VA = wA; VW = wW
        V = VA if k == 0 else (VW if k == 1 else 0.0)
        if k < 2:
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
        e = emin + (1.0 - emin) * (1.0 - np.exp(-Rbar / re)) if eng_on == 1 else 1.0
        p = (1.0 - 0.02) * e * (1.0 - (1.0 - eps) * (1.0 - drive))
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        if simulate:
            lick[t] = 1 if u[t] < p else 0
        l = lick[t]
        if mask[t] == 1:
            ll += np.log(p) if l == 1 else np.log(1.0 - p)
        d = np.nan
        water = 0.0
        if keep:
            out[t, 0] = p; out[t, 1] = V; out[t, 2] = VA; out[t, 3] = VW; out[t, 4] = wA; out[t, 5] = wW; out[t, 6] = eps
            out[t, 7] = T; out[t, 8] = drive; out[t, 9] = e
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
                if transfer == 1:
                    wA += a * d * (1.0 if k == 0 else g)
                    wW += a * d * (g if k == 0 else 1.0)
                elif k == 0:
                    wA += a * d
                else:
                    wW += a * d
            elif fa_mode == 2:
                eps_state = einf + (eps_state - einf) * (1.0 - aeps)
            T = max(0.0, T - rho * water)
        if keep:
            out[t, 10] = d
        Rbar += aR * (water - Rbar)
        n += 1.0
    return ll


def _args(name):
    tr, fa, en, _ = parts(name)
    return tr, fa, en


def loglik(name, x, A, mask):
    P, _ = full_vector(name, x)
    return kernel(*A, P, *_args(name), mask, np.empty(0), False, np.empty((0, len(LATENTS))))


def latents(name, p, A):
    P = np.array([p[k] for k in P_ALL], float)
    out = np.empty((len(A[0]), len(LATENTS)))
    kernel(*A, P, *_args(name), np.ones(len(A[0]), np.int64), np.empty(0), False, out)
    return out


def simulate(name, p, A, rng):
    tt, lick, rplus, ss, ms = A
    lick = lick.copy()
    P = np.array([p[k] for k in P_ALL], float)
    out = np.empty((len(tt), len(LATENTS)))
    kernel(tt, lick, rplus, ss, ms, P, *_args(name), np.ones(len(tt), np.int64), rng.uniform(size=len(tt)), True, out)
    return lick, out


def fit(name, A, mask, rng, n_random=6, extra_starts=()):
    fp = free_params(name)
    lo = np.array([BOUNDS[k][0] for k in fp]); hi = np.array([BOUNDS[k][1] for k in fp])
    islog = np.array([k in LOG for k in fp])
    blo = np.where(islog, np.log(np.where(islog, lo, 1.0)), lo); bhi = np.where(islog, np.log(np.where(islog, hi, 1.0)), hi)

    def to_x(y):
        return np.where(islog, np.exp(y), y)

    def nll(y):
        v = -loglik(name, to_x(y), A, mask)
        return v if np.isfinite(v) else 1e12
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


def nested_of(name):
    """(sub-model, extra values that make `name` reproduce it)"""
    tr, fa, en, lr = name.split("-")
    out = []
    if lr == "asym":
        out.append((f"{tr}-{fa}-{en}-sym", {}))
    if en == "eng":
        out.append((f"{tr}-{fa}-noeng-{lr}", dict(e_min=1.0, R0=0.0)))
    if fa in ("E1", "E2"):
        out.append((f"{tr}-E0-{en}-{lr}", dict(eps_inf=None, tau=3000.0, alpha_eps=1e-4)))
    return out


def arrays(d):
    tt = d.trial_type.map({"auditory": 0, "whisker": 1, "no_stim": 2}).to_numpy(np.int64)
    lick = d.lick_flag.to_numpy(np.int64).copy()
    rplus = (d.reward_group == "R+").to_numpy(np.int64)
    ms = (d.mouse_id != d.mouse_id.shift()).to_numpy(np.int64)
    ss = (d.session_id != d.session_id.shift()).to_numpy(np.int64)
    return tt, lick, rplus, ss, ms
