"""017 -- Simulation-based fitting of the whole training process, scored against the DISTRIBUTION of simulated trajectories (fix of 015,
whose recovery failed: 015 compared one learning path with the mean of simulated runs, ignoring path-to-path variability).
Same agents (rl4_agents), sequences and summary bins as 015. For a parameter set, R runs on the mouse's own trial sequence, each with its
own licks and learning path, give R simulated trajectories (bin lick rates); score = Gaussian synthetic log-likelihood of the mouse's bin
rates under their mean and covariance across runs (rl4_agents.synth_ll; shrinkage toward the diagonal, variance floor).
  bank + refinement (Nelder-Mead, fixed random draws), agents smallest first, each also started from its sub-agents' optima
  held-out: leave-one-session-out (parameters chosen on the other sessions' bins, the left-out session's bins scored); day-0 fold separate
Recovery (--recovery N): pseudo-mice = one independent run of each agent with the mouse's fitted parameters; matched like the data.
Output: fits/017_fits.csv, tables/017_bins.parquet, fits/017_recovery.csv
Run (haas): python 017_sim_fit_synth.py [--n-jobs 90] [--n-bank 800] [--runs 50] [--recovery 40] [--recovery-only]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import common as C
import rl4_agents as A4


def clip(model, x):
    return np.array([np.clip(v, *A4.BOUNDS[k]) for k, v in zip(A4.free(model), x)])


def project(model, sub, x_sub):
    d = dict(A4.OFF); d.update(dict(zip(A4.free(sub), x_sub)))
    return np.array([d[k] for k in A4.free(model)])


class Mouse:
    def __init__(self, Ar, bin_id, B, U):
        self.Ar, self.bin_id, self.n = Ar, bin_id, B.n.to_numpy(float)
        self.obs = (B.k / B.n).to_numpy(float); self.U = U
        self.sess = B.session_id.to_numpy(); self.d0 = B.day.to_numpy() == 0

    def runs(self, model, x):
        return A4.run_rates(self.Ar, self.bin_id, self.n, A4.vector(model, x), self.U)

    def score(self, model, x, sel):
        return A4.synth_ll(self.runs(model, x), self.obs, sel)


def refine(Mo, model, x0, sel, maxiter):
    f = lambda x: -Mo.score(model, clip(model, x), sel)
    r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3))
    x = clip(model, r.x)
    return x, -f(x)


def match(Mo, n_bank, seed, models=A4.MODELS):
    order = list(dict.fromkeys(Mo.sess)); allb = np.ones(len(Mo.obs), bool)
    out, xf = {}, {}
    for m in models:
        subs = [m.replace(c, "") for c in m]
        X = A4.lhs(m, n_bank, seed + len(m))
        RR = [Mo.runs(m, x) for x in X]
        tot = np.array([A4.synth_ll(R, Mo.obs, allb) for R in RR])
        starts = [X[i] for i in np.argsort(tot)[::-1][:3]] + [project(m, sb, out[sb]["x"]) for sb in subs]
        x_full, ll_full = max((refine(Mo, m, x0, allb, 200) for x0 in starts), key=lambda z: z[1])
        held, held0, xf[m] = 0.0, np.nan, {}
        for s in order:
            tr = Mo.sess != s
            sc = np.array([A4.synth_ll(R, Mo.obs, tr) for R in RR])
            st = [X[i] for i in np.argsort(sc)[::-1][:2]] + [project(m, sb, xf[sb][s]) for sb in subs]
            xs = max((refine(Mo, m, x0, tr, 80) for x0 in st), key=lambda z: z[1])[0]
            xf[m][s] = xs
            ll = Mo.score(m, xs, ~tr)
            held += ll
            if Mo.d0[~tr].any():
                held0 = ll
        R = Mo.runs(m, x_full)
        out[m] = dict(x=x_full, ll_full=ll_full, ll_held=held, ll_held_day0=held0, mean=R.mean(0), q10=np.quantile(R, 0.1, 0),
                      q90=np.quantile(R, 0.9, 0))
    return out


def run_mouse(args):
    mid, g, n_bank, runs = args
    seed = zlib.crc32(mid.encode())
    Ar, bin_id, B = A4.mouse_arrays(g)
    Mo = Mouse(Ar, bin_id, B, np.random.default_rng(seed).random((runs, len(g))))
    res = match(Mo, n_bank, seed)
    meta = dict(mouse_id=mid, cohort=g.reward_group.iloc[0], n_trials=len(g), n_bins=len(B), n_sessions=B.session_id.nunique())
    rows = [dict(meta, model=A4.lab(m), k=len(A4.free(m)), ll_full=r["ll_full"], ll_held=r["ll_held"], ll_held_day0=r["ll_held_day0"],
                 **dict(zip(A4.free(m), r["x"]))) for m, r in res.items()]
    Bo = B.assign(mouse_id=mid, cohort=meta["cohort"], rate=B.k / B.n,
                  **{f"sim_{A4.lab(m)}": r["mean"] for m, r in res.items()},
                  **{f"q10_{A4.lab(m)}": r["q10"] for m, r in res.items()}, **{f"q90_{A4.lab(m)}": r["q90"] for m, r in res.items()})
    print(mid, "done", flush=True)
    return rows, Bo


def recover_mouse(args):
    mid, g, n_bank, runs, fitted = args
    seed = zlib.crc32(mid.encode()) + 7
    Ar, bin_id, B = A4.mouse_arrays(g)
    U = np.random.default_rng(seed).random((runs, len(g)))
    rows = []
    for gm in A4.MODELS:
        f = fitted[(mid, A4.lab(gm))]
        x_true = np.array([f[k] for k in A4.free(gm)])
        u1 = np.random.default_rng(seed + 100 + len(gm)).random((1, len(g)))
        r1 = A4.run_rates(Ar, bin_id, B.n.to_numpy(float), A4.vector(gm, x_true), u1)[0]            # one pseudo-mouse
        Bp = B.assign(k=np.round(r1 * B.n.to_numpy()))
        res = match(Mouse(Ar, bin_id, Bp, U), n_bank, seed)
        best = max(res, key=lambda m: res[m]["ll_held"])
        rows.append(dict(mouse_id=mid, generating=A4.lab(gm), selected=A4.lab(best),
                         **{f"ll_held_{A4.lab(m)}": res[m]["ll_held"] for m in res},
                         **{f"true_{k}": v for k, v in zip(A4.free(gm), x_true)}, **{f"fit_{k}": v for k, v in zip(A4.free(gm), res[gm]["x"])}))
    print(mid, "recovered", flush=True)
    return rows


def main(a):
    d = pd.read_parquet(C.TAB / "011_sequences.parquet")
    groups = list(d.groupby("mouse_id"))
    print(f"{len(groups)} mice, {len(A4.MODELS)} agents, bank {a.n_bank} x {a.runs} runs", flush=True)
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        if not a.recovery_only:
            res = pool.map(run_mouse, [(m, g, a.n_bank, a.runs) for m, g in groups], chunksize=1)
            F = pd.DataFrame([r for x in res for r in x[0]]); F.to_csv(C.FITS / "017_fits.csv", index=False)
            pd.concat([x[1] for x in res], ignore_index=True).to_parquet(C.TAB / "017_bins.parquet")
            H = F.pivot(index="mouse_id", columns="model", values="ll_held")
            print("held-out synthetic LL summed over mice, relative to core:"); print(H.sub(H["core"], axis=0).sum().sort_values().round(1))
            print("best agent per mouse:"); print(H.idxmax(1).value_counts(), flush=True)
        if a.recovery:
            F = pd.read_csv(C.FITS / "017_fits.csv")
            fitted = {(r.mouse_id, r.model): r._asdict() for r in F.itertuples()}
            sel = groups[:: max(1, len(groups) // a.recovery)][:a.recovery]
            rec = pool.map(recover_mouse, [(m, g, a.n_bank, a.runs, fitted) for m, g in sel], chunksize=1)
            Rr = pd.DataFrame([r for x in rec for r in x]); Rr.to_csv(C.FITS / "017_recovery.csv", index=False)
            print(pd.crosstab(Rr.generating, Rr.selected), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=90)
    ap.add_argument("--n-bank", type=int, default=800)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--recovery", type=int, default=0)
    ap.add_argument("--recovery-only", action="store_true")
    main(ap.parse_args())
