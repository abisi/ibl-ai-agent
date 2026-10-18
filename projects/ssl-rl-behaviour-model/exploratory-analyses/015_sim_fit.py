"""015 -- Simulation-based fitting of the whole training process (model_sim_design.md; rl4_agents). For every mouse and agent:
  bank          N_BANK parameter sets (Latin hypercube), each simulated R times on the mouse's own trial sequence (fixed random draws per
                mouse: common random numbers), mean lick rate per summary bin
  full fit      best bank sets by the binomial log-likelihood of the mouse's bin counts, refined by Nelder-Mead (bounded by clipping)
  held-out      leave-one-session-out over bins: parameters chosen (bank + refinement) on the other sessions' bins, the left-out session's
                bins scored; summed = held-out LL; the day-0 fold separately
  trajectories  per bin: data rate and the full-fit simulated rate of every agent
Recovery (--recovery N): for N mice, pseudo-mice simulated (one run, own licks) from every agent's full-fit parameters for that mouse,
matched with all agents exactly as the data -> generating vs selected agent, true vs recovered parameters.
Output: fits/015_fits.csv, tables/015_bins.parquet, (recovery) fits/015_recovery.csv
Run (haas): python 015_sim_fit.py [--n-jobs 90] [--n-bank 2000] [--runs 10] [--recovery 40]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import common as C
import rl4_agents as A4


def clip(model, x):
    return np.array([np.clip(v, *A4.BOUNDS[k]) for k, v in zip(A4.free(model), x)])


def scores(model, X, Ar, bin_id, B, U):
    nb = len(B)
    R = np.array([A4.bin_rates(Ar, bin_id, nb, A4.vector(model, x), U)[0] for x in X])
    return A4.binom_ll(R, B.n.to_numpy(), B.k.to_numpy())                    # bank x bins


def refine(model, x0, Ar, bin_id, B, U, use, maxiter):
    nb = len(B); n, k = B.n.to_numpy(), B.k.to_numpy()

    def f(x):
        r = A4.bin_rates(Ar, bin_id, nb, A4.vector(model, clip(model, x)), U)[0]
        return -A4.binom_ll(r, n, k)[use].sum()
    r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3))
    x = clip(model, r.x)
    return x, -f(x)


def project(model, sub, x_sub):
    """a sub-agent's solution as a point of `model` (components the sub-agent lacks at their OFF values: identical simulation)"""
    d = dict(A4.OFF); d.update(dict(zip(A4.free(sub), x_sub)))
    return np.array([d[k] for k in A4.free(model)])


def match(Ar, bin_id, B, U, n_bank, seed, models=A4.MODELS):
    """all agents for one (real or pseudo) mouse: full fit, held-out LL (leave one session out), fitted rates per bin. Agents are fitted
    smallest first; each also starts from the optima of its immediate sub-agents (full fit and every fold), so an agent never ends worse
    than an agent it contains (common random numbers make the simulation deterministic)"""
    sess = B.session_id.to_numpy(); order = list(dict.fromkeys(sess)); d0 = B.day.to_numpy() == 0
    nk = (B.n.to_numpy(), B.k.to_numpy())
    out, xs_fold = {}, {}
    for m in models:
        subs = [m.replace(c, "") for c in m]
        X = A4.lhs(m, n_bank, seed + len(m))
        S = scores(m, X, Ar, bin_id, B, U)
        top = np.argsort(S.sum(1))[::-1][:3]
        starts = [X[i] for i in top] + [project(m, sb, out[sb]["x"]) for sb in subs]
        x_full, ll_full = max((refine(m, x0, Ar, bin_id, B, U, np.ones(len(B), bool), 400) for x0 in starts), key=lambda z: z[1])
        held, held0, xs_fold[m] = 0.0, np.nan, {}
        for s in order:
            tr = sess != s
            i0 = np.argsort(S[:, tr].sum(1))[::-1][:2]
            st = [X[i] for i in i0] + [project(m, sb, xs_fold[sb][s]) for sb in subs]
            xs = max((refine(m, x0, Ar, bin_id, B, U, tr, 200) for x0 in st), key=lambda z: z[1])[0]
            xs_fold[m][s] = xs
            r = A4.bin_rates(Ar, bin_id, len(B), A4.vector(m, xs), U)[0]
            ll = A4.binom_ll(r, *nk)[~tr].sum()
            held += ll
            if d0[~tr].any():
                held0 = ll
        rates = A4.bin_rates(Ar, bin_id, len(B), A4.vector(m, x_full), U)[0]
        out[m] = dict(x=x_full, ll_full=ll_full, ll_held=held, ll_held_day0=held0, rates=rates)
    return out


def run_mouse(args):
    mid, g, n_bank, runs = args
    seed = zlib.crc32(mid.encode())
    Ar, bin_id, B = A4.mouse_arrays(g)
    U = np.random.default_rng(seed).random((runs, len(g)))
    res = match(Ar, bin_id, B, U, n_bank, seed)
    meta = dict(mouse_id=mid, cohort=g.reward_group.iloc[0], n_trials=len(g), n_bins=len(B), n_sessions=B.session_id.nunique())
    rows = [dict(meta, model=A4.lab(m), k=len(A4.free(m)), ll_full=r["ll_full"], ll_held=r["ll_held"], ll_held_day0=r["ll_held_day0"],
                 **dict(zip(A4.free(m), r["x"]))) for m, r in res.items()]
    Bo = B.assign(mouse_id=mid, cohort=meta["cohort"], rate=B.k / B.n, **{f"sim_{A4.lab(m)}": r["rates"] for m, r in res.items()})
    return rows, Bo, res


def recover_mouse(args):
    mid, g, n_bank, runs, fitted = args
    seed = zlib.crc32(mid.encode()) + 7
    Ar, bin_id, B = A4.mouse_arrays(g)
    U = np.random.default_rng(seed).random((runs, len(g)))
    rows = []
    for gm in A4.MODELS:
        f = fitted[(mid, A4.lab(gm))]
        P = A4.vector(gm, [f[k] for k in A4.free(gm)])
        u1 = np.random.default_rng(seed + 100 + len(gm)).random((1, len(g)))
        _, p1 = A4.bin_rates(Ar, bin_id, len(B), P, u1)
        lick = (u1[0] < p1).astype(float)
        m_ = bin_id >= 0
        Bp = B.assign(k=np.bincount(bin_id[m_], weights=lick[m_], minlength=len(B)))
        res = match(Ar, bin_id, Bp, U, n_bank, seed)
        best = max(res, key=lambda m: res[m]["ll_held"])
        rows.append(dict(mouse_id=mid, generating=A4.lab(gm), selected=A4.lab(best), **{f"ll_held_{A4.lab(m)}": res[m]["ll_held"] for m in res},
                         **{f"true_{k}": f[k] for k in A4.free(gm)}, **{f"fit_{k}": v for k, v in zip(A4.free(gm), res[gm]["x"])}))
    return rows


def main(a):
    d = pd.read_parquet(C.TAB / "011_sequences.parquet")
    groups = list(d.groupby("mouse_id"))
    print(f"{len(groups)} mice, {len(A4.MODELS)} agents, bank {a.n_bank} x {a.runs} runs")
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        if not a.recovery_only:
            res = pool.map(run_mouse, [(m, g, a.n_bank, a.runs) for m, g in groups], chunksize=1)
            F = pd.DataFrame([r for x in res for r in x[0]]); F.to_csv(C.FITS / "015_fits.csv", index=False)
            pd.concat([x[1] for x in res]).to_parquet(C.TAB / "015_bins.parquet")
            H = F.pivot(index="mouse_id", columns="model", values="ll_held")
            print("held-out LL summed over mice, relative to core:"); print((H.sub(H["core"], axis=0)).sum().sort_values().round(1))
            print("best agent per mouse:"); print(H.idxmax(1).value_counts())
        if a.recovery:
            F = pd.read_csv(C.FITS / "015_fits.csv")
            fitted = {(r.mouse_id, r.model): r._asdict() for r in F.itertuples()}
            sel = [x for x in groups][:: max(1, len(groups) // a.recovery)][:a.recovery]
            rec = pool.map(recover_mouse, [(m, g, a.n_bank, a.runs, fitted) for m, g in sel], chunksize=1)
            Rr = pd.DataFrame([r for x in rec for r in x]); Rr.to_csv(C.FITS / "015_recovery.csv", index=False)
            print(pd.crosstab(Rr.generating, Rr.selected))
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=90)
    ap.add_argument("--n-bank", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--recovery", type=int, default=0)
    ap.add_argument("--recovery-only", action="store_true")
    main(ap.parse_args())
