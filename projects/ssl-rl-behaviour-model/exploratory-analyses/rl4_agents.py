"""rl4_agents -- generative agents of the whole training process (model_sim_design.md). An agent runs on a mouse's own trial sequence and
draws its own licks; the output is the mean P(lick) per trial over R runs, aggregated into the summary bins of the mouse.
Core: bias, auditory and whisker weights, delta rule after licks (alpha+, alpha-, bias credit kappa), outcomes fixed by the design
(auditory +1 water, whisker +1 water in R+ / -1 in R-, no stimulus -0.5), P(lick) = lapse/2 + (1 - lapse) * sigmoid(beta * T * Q - c),
disengagement: thirst T = 1 at session start, - rho per water reward. Components: G (gamma), N (nu, lam_nu), Z (zeta).
"""
from __future__ import annotations

import itertools

import numpy as np
from numba import njit

PARAMS = ["alpha_pos", "alpha_neg", "kappa", "beta", "c", "lapse", "rho", "gamma", "nu", "lam_nu", "zeta"]
CORE = ["alpha_pos", "alpha_neg", "kappa", "beta", "c", "lapse", "rho"]
COMP = {"G": ["gamma"], "N": ["nu", "lam_nu"], "Z": ["zeta"]}
CNAME = {"G": "generalisation", "N": "novelty", "Z": "context switch"}
BOUNDS = dict(alpha_pos=(1e-3, 0.5), alpha_neg=(1e-3, 0.5), kappa=(0.0, 2.0), beta=(0.5, 20.0), c=(-3.0, 3.0), lapse=(0.01, 0.3),
              rho=(0.0, 0.02), gamma=(0.0, 1.0), nu=(0.0, 4.0), lam_nu=(2.0, 300.0), zeta=(0.0, 1.0))
LOG = {"alpha_pos", "alpha_neg", "beta", "lam_nu"}
OFF = dict(gamma=0.0, nu=0.0, lam_nu=50.0, zeta=0.0)
MODELS = ["".join(c) for n in range(4) for c in itertools.combinations("GNZ", n)]        # "", G, N, Z, GN, GZ, NZ, GNZ
TT = {"auditory": 0, "whisker": 1, "no_stim": 2}


def lab(m):
    return m if m else "core"


def free(model):
    return CORE + [p for c in model for p in COMP[c]]


def vector(model, x):
    p = dict(OFF); p.update(dict(zip(free(model), x)))
    return np.array([p.get(k, 0.0) for k in PARAMS], float)


@njit(cache=True)
def _simulate(tt, o, water, sstart, switch, P, U, out):
    """U: runs x trials uniform draws; out: mean P(lick) per trial over runs (written)"""
    ap, an, kap, beta, c, lapse, rho, gam, nu, lnu, zeta = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]
    n = tt.shape[0]; R = U.shape[0]
    for t in range(n):
        out[t] = 0.0
    for r in range(R):
        wb, ws, wa, ww = 0.0, 0.0, 0.0, 0.0
        T = 1.0; nA = 0.0; nW = 0.0
        for t in range(n):
            if sstart[t] == 1:
                T = 1.0
            if switch[t] == 1:
                wb = (1.0 - zeta) * wb
            k = tt[t]
            if k == 0:
                q = wb + gam * ws + (1.0 - gam) * wa; nov = nu * np.exp(-nA / lnu)
            elif k == 1:
                q = wb + gam * ws + (1.0 - gam) * ww; nov = nu * np.exp(-nW / lnu)
            else:
                q = wb; nov = 0.0
            p = lapse / 2.0 + (1.0 - lapse) / (1.0 + np.exp(-(beta * T * q + nov - c)))
            out[t] += p / R
            if U[r, t] < p:
                d = o[t] - q
                a = ap if d > 0 else an
                wb += a * kap * d
                if k != 2:
                    ws += a * gam * d
                    if k == 0:
                        wa += a * (1.0 - gam) * d
                    else:
                        ww += a * (1.0 - gam) * d
                if water[t] > 0:
                    T -= rho
                    if T < 0.0:
                        T = 0.0
            if k == 0:
                nA += 1.0
            elif k == 1:
                nW += 1.0


def mouse_arrays(g):
    """trial arrays of one mouse (ordered) and its summary bins: per session, auditory / whisker / no-stimulus in the first and second
    half; on day 0 whisker trials in blocks of 10 (first 100) instead of halves. Returns arrays, bin id per trial (-1 = unused), bin table"""
    tt = g.trial_type.map(TT).to_numpy(np.int64)
    rplus = g.reward_group.iloc[0] == "R+"
    o = np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if rplus else -1.0, -0.5))
    water = np.where(tt == 0, 1.0, np.where(tt == 1, 1.0 if rplus else 0.0, 0.0))
    sess = g.session_id.to_numpy(); day = g.day.to_numpy()
    sstart = np.r_[1, (sess[1:] != sess[:-1])].astype(np.int64)
    switch = np.zeros(len(g), np.int64)
    if (day == 0).any():
        switch[np.argmax(day == 0)] = 1
    pos = g.groupby("session_id").cumcount().to_numpy(); size = g.groupby("session_id").session_id.transform("size").to_numpy()
    half = (pos >= size / 2).astype(int)
    labels = []
    d0 = day == 0
    wcount = np.full(len(g), -1)
    if d0.any():
        w0 = np.where(d0 & (tt == 1))[0]
        wcount[w0] = np.arange(len(w0))
    for i in range(len(g)):
        if d0[i] and tt[i] == 1:
            labels.append((sess[i], int(day[i]), "whisker", f"block{wcount[i] // 10:02d}") if wcount[i] < 100 else None)
        else:
            labels.append((sess[i], int(day[i]), ["auditory", "whisker", "no_stim"][tt[i]], f"half{half[i]}"))
    keys = sorted({x for x in labels if x is not None}, key=lambda z: (z[1], z[0], z[2], z[3]))
    kid = {k: i for i, k in enumerate(keys)}
    bin_id = np.array([kid[x] if x is not None else -1 for x in labels], np.int64)
    lick = g.lick_flag.to_numpy(np.int64)
    nb = len(keys)
    n = np.bincount(bin_id[bin_id >= 0], minlength=nb); k = np.bincount(bin_id[bin_id >= 0], weights=lick[bin_id >= 0], minlength=nb)
    import pandas as pd
    B = pd.DataFrame(keys, columns=["session_id", "day", "trial_type", "bin"]).assign(n=n, k=k)
    A = dict(tt=np.array(tt), o=np.array(o), water=np.array(water), sstart=np.array(sstart), switch=switch)
    return A, bin_id, B


def bin_rates(A, bin_id, nb, P, U):
    out = np.empty(len(A["tt"]))
    _simulate(A["tt"], A["o"], A["water"], A["sstart"], A["switch"], P, U, out)
    m = bin_id >= 0
    s = np.bincount(bin_id[m], weights=out[m], minlength=nb); c = np.bincount(bin_id[m], minlength=nb)
    return s / np.maximum(c, 1), out


def binom_ll(rate, n, k):
    r = np.clip(rate, 1e-3, 1 - 1e-3)
    return k * np.log(r) + (n - k) * np.log(1 - r)          # per bin


def lhs(model, n, seed):
    """Latin hypercube in the bounds (log scale for LOG parameters)"""
    rng = np.random.default_rng(seed); fp = free(model)
    X = (rng.permuted(np.tile(np.arange(n), (len(fp), 1)), axis=1).T + rng.random((n, len(fp)))) / n
    out = np.empty_like(X)
    for j, k in enumerate(fp):
        lo, hi = BOUNDS[k]
        out[:, j] = np.exp(np.log(lo) + X[:, j] * (np.log(hi) - np.log(lo))) if k in LOG else lo + X[:, j] * (hi - lo)
    return out


# ── run-level summaries and synthetic likelihood (2026-10-18: score against the distribution of simulated trajectories) ──────────
@njit(cache=True)
def _simulate_runs(tt, o, water, sstart, switch, P, U, bin_id, nb, counts):
    """like _simulate, but every run keeps its own sampled licks: counts[r, b] = licks of run r in bin b"""
    ap, an, kap, beta, c, lapse, rho, gam, nu, lnu, zeta = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]
    n = tt.shape[0]; R = U.shape[0]
    for r in range(R):
        for b in range(nb):
            counts[r, b] = 0.0
        wb, ws, wa, ww = 0.0, 0.0, 0.0, 0.0
        T = 1.0; nA = 0.0; nW = 0.0
        for t in range(n):
            if sstart[t] == 1:
                T = 1.0
            if switch[t] == 1:
                wb = (1.0 - zeta) * wb
            k = tt[t]
            if k == 0:
                q = wb + gam * ws + (1.0 - gam) * wa; nov = nu * np.exp(-nA / lnu)
            elif k == 1:
                q = wb + gam * ws + (1.0 - gam) * ww; nov = nu * np.exp(-nW / lnu)
            else:
                q = wb; nov = 0.0
            p = lapse / 2.0 + (1.0 - lapse) / (1.0 + np.exp(-(beta * T * q + nov - c)))
            if U[r, t] < p:
                if bin_id[t] >= 0:
                    counts[r, bin_id[t]] += 1.0
                d = o[t] - q
                a = ap if d > 0 else an
                wb += a * kap * d
                if k != 2:
                    ws += a * gam * d
                    if k == 0:
                        wa += a * (1.0 - gam) * d
                    else:
                        ww += a * (1.0 - gam) * d
                if water[t] > 0:
                    T -= rho
                    if T < 0.0:
                        T = 0.0
            if k == 0:
                nA += 1.0
            elif k == 1:
                nW += 1.0


def run_rates(A, bin_id, n_bin, P, U):
    """runs x bins matrix of simulated lick rates (each run with its own licks and learning path)"""
    counts = np.empty((U.shape[0], len(n_bin)))
    _simulate_runs(A["tt"], A["o"], A["water"], A["sstart"], A["switch"], P, U, bin_id, len(n_bin), counts)
    return counts / np.maximum(n_bin, 1)[None, :]


def synth_ll(Rr, obs, sel, floor=1e-3, shrink=0.2):
    """Gaussian synthetic log-likelihood (Wood 2010) of the observed bin rates obs[sel] under the simulated runs Rr[:, sel]: mean and
    covariance across runs (covariance shrunk toward its diagonal, variance floor), so path-to-path variability of learning agents and
    its correlation across sessions are part of the score"""
    X = Rr[:, sel]; y = obs[sel]
    mu = X.mean(0); S = np.cov(X, rowvar=False) if X.shape[1] > 1 else np.atleast_2d(X.var(ddof=1))
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S)) + floor * np.eye(len(y))
    sign, logdet = np.linalg.slogdet(S)
    r = y - mu
    return -0.5 * (r @ np.linalg.solve(S, r) + logdet + len(y) * np.log(2 * np.pi))
