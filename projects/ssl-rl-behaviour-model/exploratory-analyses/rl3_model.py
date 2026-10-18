"""rl3_model -- model v3 (model_v3_design.md, decisions of 2026-10-18).
Core learning (bias with a context, stimulus-present S, auditory A, whisker W; delta rule on licks; credit kappa on the bias) and decision
(beta, constant lick cost c, disengagement lapse lambda, spontaneous licks p_sp), plus six switchable components:
  G generalisation   gamma
  M motivation       rho (thirst drop per reward; thirst reset to 1 each session, instantaneous absorption), s_B, theta (Pavlovian
                     lick bias B(T) = s_B * softplus(K * (T - theta)) / K)
  N novelty          nu, lam_nu (bonus nu * exp(-presentations / lam_nu) for the tone and the whisker stimulus)
  Z context          zeta (bias scaled by 1 - zeta at the first trial of the first whisker session)
  R average reward   alpha_R, eta (Rbar <- Rbar + alpha_R * (water - Rbar) every trial, reset each session; lick outcome o - eta * Rbar)
  P perseveration    pi, phi (choice kernel C <- C + phi * (lick - C), reset to 0.5 each session; + pi * (2C - 1))
Outcomes of a lick (fixed; the only cohort difference): auditory +1 (water), whisker +1 (water) in R+ and -1 (no water) in R-, no
stimulus -0.5 (no water).
DV = beta * T * Q(s) + B(T) + Nov(s) + pi * (2C - 1) - c;  P(lick) = (1 - lambda) * [1 - (1 - p_sp) * (1 - sigmoid(DV))].
A model is a subset of "GMNZRP" (core always on); switched-off components take their OFF values. The kernel scores only trials with
mask = 1 (leave-one-session-out) but always runs the whole sequence with the actual licks.
"""
from __future__ import annotations

import itertools

import numpy as np
from numba import njit
from scipy.optimize import minimize

PARAMS = ["alpha_pos", "alpha_neg", "kappa", "beta", "c", "lapse", "p_sp",
          "gamma", "rho", "s_B", "theta", "nu", "lam_nu", "zeta", "alpha_R", "eta", "pi", "phi"]
IDX = {k: i for i, k in enumerate(PARAMS)}
CORE = ["alpha_pos", "alpha_neg", "kappa", "beta", "c", "lapse", "p_sp"]
COMPONENTS = {"G": ["gamma"], "M": ["rho", "s_B", "theta"], "N": ["nu", "lam_nu"], "Z": ["zeta"], "R": ["alpha_R", "eta"], "P": ["pi", "phi"]}
CNAME = {"G": "generalisation", "M": "motivation", "N": "novelty", "Z": "context switch", "R": "average reward", "P": "perseveration"}
BOUNDS = dict(alpha_pos=(1e-4, 1.0), alpha_neg=(1e-4, 1.0), kappa=(0.0, 3.0), beta=(0.1, 50.0), c=(-5.0, 5.0), lapse=(0.0, 0.5),
              p_sp=(0.0, 0.5), gamma=(0.0, 1.0), rho=(0.0, 0.05), s_B=(0.0, 10.0), theta=(0.0, 1.0), nu=(0.0, 5.0), lam_nu=(1.0, 500.0),
              zeta=(0.0, 1.0), alpha_R=(1e-3, 1.0), eta=(0.0, 3.0), pi=(-3.0, 3.0), phi=(1e-3, 1.0))
OFF = dict(gamma=0.0, rho=0.0, s_B=0.0, theta=0.5, nu=0.0, lam_nu=50.0, zeta=0.0, alpha_R=0.1, eta=0.0, pi=0.0, phi=0.5)
DEFAULT = dict(alpha_pos=0.05, alpha_neg=0.05, kappa=0.5, beta=3.0, c=0.5, lapse=0.02, p_sp=0.05, gamma=0.3, rho=0.005, s_B=1.0,
               theta=0.5, nu=0.5, lam_nu=50.0, zeta=0.3, alpha_R=0.05, eta=0.5, pi=0.5, phi=0.3)
LOG = {"alpha_pos", "alpha_neg", "beta", "rho", "lam_nu", "alpha_R", "phi"}
MODELS = ["".join(c) for n in range(7) for c in itertools.combinations("GMNZRP", n)]      # 64, smallest first
K_SOFT = 20.0
TT = {"auditory": 0, "whisker": 1, "no_stim": 2}


def free_params(model):
    return CORE + [p for comp in model for p in COMPONENTS[comp]]


@njit(cache=True)
def _kernel(tt, lick, o, water, sstart, switch, P, mask, out, sim, u):
    ap, an, kap, beta, c, lapse, psp = P[0], P[1], P[2], P[3], P[4], P[5], P[6]
    gam, rho, sB, th, nu, lnu, zeta, aR, eta, pi_, phi = P[7], P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15], P[16], P[17]
    wb, ws, wa, ww = 0.0, 0.0, 0.0, 0.0
    T, Rb, Ck = 1.0, 0.0, 0.5
    nA, nW = 0.0, 0.0
    ll = 0.0
    keep = out.shape[0] > 0
    for t in range(tt.shape[0]):
        if sstart[t] == 1:
            T, Rb, Ck = 1.0, 0.0, 0.5
        if switch[t] == 1:
            wb = (1.0 - zeta) * wb
        k = tt[t]
        if k == 0:
            q = wb + gam * ws + (1.0 - gam) * wa
            nov = nu * np.exp(-nA / lnu)
        elif k == 1:
            q = wb + gam * ws + (1.0 - gam) * ww
            nov = nu * np.exp(-nW / lnu)
        else:
            q = wb
            nov = 0.0
        x = K_SOFT * (T - th)
        sp = x if x > 30.0 else np.log1p(np.exp(x))
        dv = beta * T * q + sB * sp / K_SOFT + nov + pi_ * (2.0 * Ck - 1.0) - c
        p = (1.0 - lapse) * (1.0 - (1.0 - psp) * (1.0 - 1.0 / (1.0 + np.exp(-dv))))
        if p < 1e-9:
            p = 1e-9
        if p > 1.0 - 1e-9:
            p = 1.0 - 1e-9
        if sim:
            lick[t] = 1 if u[t] < p else 0
        l = lick[t]
        if mask[t] == 1:
            ll += np.log(p) if l == 1 else np.log(1.0 - p)
        if keep:
            out[t, 0] = p; out[t, 1] = q; out[t, 2] = wb; out[t, 3] = ws; out[t, 4] = wa; out[t, 5] = ww; out[t, 6] = T
            out[t, 7] = Rb; out[t, 8] = dv; out[t, 9] = np.nan
        wat = water[t] * l
        if l == 1:
            d = (o[t] - eta * Rb) - q
            a = ap if d > 0 else an
            wb += a * kap * d
            if k != 2:
                ws += a * gam * d
                if k == 0:
                    wa += a * (1.0 - gam) * d
                else:
                    ww += a * (1.0 - gam) * d
            if keep:
                out[t, 9] = d
        if wat > 0:
            T = T - rho * wat
            if T < 0.0:
                T = 0.0
        Rb += aR * (wat - Rb)
        Ck += phi * (l - Ck)
        if k == 0:
            nA += 1.0
        elif k == 1:
            nW += 1.0
    return ll


def sequence_arrays(g):
    """per-mouse arrays from the trial table (ordered): tt, lick, outcome, water, session start, context switch"""
    tt = g.trial_type.map(TT).to_numpy(np.int64)
    rplus = g.reward_group.iloc[0] == "R+"
    o = np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if rplus else -1.0, -0.5))
    water = np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if rplus else 0.0, 0.0))
    sess = g.session_id.to_numpy()
    sstart = np.r_[1, (sess[1:] != sess[:-1]).astype(np.int64)]
    switch = np.zeros(len(g), np.int64)
    d0 = np.where(g.day.to_numpy() == 0)[0]
    if len(d0):
        switch[d0[0]] = 1
    return dict(tt=np.array(tt), lick=np.array(g.lick_flag, dtype=np.int64), o=np.array(o), water=np.array(water),
                sstart=np.array(sstart, dtype=np.int64), switch=switch)   # writable copies (pandas returns read-only views)


def full_vector(model, x):
    p = {**DEFAULT, **OFF}
    p.update(dict(zip(free_params(model), x)))
    return np.array([p[k] for k in PARAMS], float), p


def loglik(P, A, mask):
    return _kernel(A["tt"], A["lick"], A["o"], A["water"], A["sstart"], A["switch"], P, mask, np.empty((0, 10)), False, np.empty(0))


def trace(P, A):
    out = np.empty((len(A["tt"]), 10))
    _kernel(A["tt"], A["lick"], A["o"], A["water"], A["sstart"], A["switch"], P, np.ones(len(A["tt"]), np.int64), out, False, np.empty(0))
    return out


def simulate(P, A, seed=0):
    lick = np.zeros(len(A["tt"]), np.int64); out = np.empty((len(A["tt"]), 10))
    _kernel(A["tt"], lick, A["o"], A["water"], A["sstart"], A["switch"], P, np.ones(len(A["tt"]), np.int64), out, True,
            np.random.default_rng(seed).random(len(A["tt"])))
    return lick, out[:, 0]


def random_start(model, rng):
    return [np.exp(rng.uniform(np.log(BOUNDS[k][0] if BOUNDS[k][0] > 0 else 1e-4), np.log(BOUNDS[k][1]))) if k in LOG
            else rng.uniform(*BOUNDS[k]) for k in free_params(model)]


def project(model, pdict):
    """a parameter dict (another model's optimum) as a start for `model`: its free parameters, defaults for those not in pdict"""
    return [float(np.clip(pdict.get(k, DEFAULT[k]), *BOUNDS[k])) for k in free_params(model)]


def fit(model, A, mask, starts):
    """L-BFGS-B from each start; returns (params dict, nll)"""
    fp = free_params(model); bounds = [BOUNDS[k] for k in fp]

    def nll(x):
        return -loglik(full_vector(model, x)[0], A, mask)
    best = None
    for x0 in starts:
        try:
            r = minimize(nll, np.asarray(x0, float), method="L-BFGS-B", bounds=bounds)
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return full_vector(model, best.x)[1], float(best.fun)


def fit_all_models(A, mask, n_random=2, seed=0):
    """all 64 models, smallest first; each starts from the default point, n_random random points and the optima of its immediate submodels
    (so a model can never end worse than a model it contains)"""
    rng = np.random.default_rng(seed)
    res = {}
    for m in MODELS:
        starts = [project(m, DEFAULT)] + [random_start(m, rng) for _ in range(n_random)]
        for comp in m:
            sub = m.replace(comp, "")
            starts.append(project(m, res[sub][0]))
        res[m] = fit(m, A, mask, starts)
    return res
