"""006 -- Fit the redesigned model (rl2_model, v2) to every mouse's own trial sequence by maximum likelihood.
Sequences: tables/001_trials.csv without free-licking sessions and without the few whisker trials on pretraining days; ordered by day and
trial start; the last TRIM trials of each session dropped (as the original targets: end-of-session disengagement); auditory pretraining
and all whisker days in one continuous sequence per mouse. Outcome of a lick: auditory +1, whisker +1 (R+) / -1 (R-), no stimulus -1.
Fits (L-BFGS-B, N_STARTS starts: the default point + random points within the bounds):
  per mouse x variant (rl2_model.VARIANTS): parameters, NLL, AIC, BIC
  pooled, variant "full": one parameter set for all mice, and one per cohort (likelihood-ratio test of a cohort-specific learning rule; the
  design question is whether the contingency alone explains the cohort difference)
Per-trial latents of the "full" fit (P(lick), V, weights, prediction error), keyed by mouse_id, session_id, trial_idx for neural alignment.
Output: tables/006_sequences.parquet, fits/006_fits.csv, fits/006_pooled.json, tables/006_latents.parquet
Run (haas): python 006_fit_v2.py [--n-jobs 60] [--n-starts 10]
"""
from __future__ import annotations

import argparse
import zlib
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import rl2_model as M

TRIM = 10


def sequences():
    d = pd.read_csv(C.TRIALS)
    d = d[(d.bhv_type != "free_licking") & ~((d.trial_type == "whisker") & (d.day < 0))]
    d = d.sort_values(["mouse_id", "day", "start_time"]).reset_index(drop=True)
    d = d[d.groupby("session_id").cumcount(ascending=False) >= TRIM].reset_index(drop=True)
    d["tt"] = d.trial_type.map(M.TT).astype(np.int64)
    d["lick"] = d.lick_flag.astype(np.int64)
    return d


def arrays(g):
    tt = g.tt.to_numpy(np.int64)
    return tt, g.lick.to_numpy(np.int64), M.outcomes(tt, g.reward_group.iloc[0])


def fit_mouse(args):
    mid, g, n_starts = args
    rows = []
    seq, sd = [arrays(g)], zlib.crc32(mid.encode())
    # nested variants: each starts also from the full optimum; full is then refitted from every nested optimum (a nested solution is a
    # point of the full space, so full can never end worse than a variant it contains); ns_weight (contains full) last, from the full optimum
    full = M.fit("full", seq, n_starts=n_starts, seed=sd)
    res = {v: M.fit(v, seq, n_starts=n_starts, seed=sd, init=full["params"]) for v in ("specific", "general", "no_bias", "symmetric")}
    for f0 in res.values():
        f1 = M.fit("full", seq, n_starts=1, seed=sd, init=f0["params"])
        if f1["nll"] < full["nll"]:
            full = f1
    res["full"] = full
    res["ns_weight"] = M.fit("ns_weight", seq, n_starts=n_starts, seed=sd, init=full["params"])
    for v in M.VARIANTS:
        f = res[v]
        rows.append(dict(mouse_id=mid, cohort=g.reward_group.iloc[0], variant=v, nll=f["nll"], k=f["k"], n=f["n"], aic=f["aic"], bic=f["bic"],
                         converged=f["converged"], **f["params"]))
    return rows


def pooled(args):
    name, data, n_starts = args
    return name, M.fit("full", data, n_starts=n_starts)


def main(a):
    d = sequences()
    d.to_parquet(C.TAB / "006_sequences.parquet")
    groups = list(d.groupby("mouse_id"))
    print(f"{len(groups)} mice, {len(d):,} trials")
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        rows = [r for rr in pool.map(fit_mouse, [(m, g, a.n_starts) for m, g in groups]) for r in rr]
        F = pd.DataFrame(rows)
        F.to_csv(C.FITS / "006_fits.csv", index=False)
        print(F.groupby("variant")[["nll", "bic"]].sum().round(1))
        data = {c: [arrays(g) for _, g in groups if g.reward_group.iloc[0] == c] for c in ("R+", "R-")}
        P = dict(pool.map(pooled, [("all", data["R+"] + data["R-"], a.n_starts), ("R+", data["R+"], a.n_starts), ("R-", data["R-"], a.n_starts)]))
    lr = 2 * (P["all"]["nll"] - P["R+"]["nll"] - P["R-"]["nll"])
    P["lrt"] = dict(stat=lr, df=P["all"]["k"], p=float(stats.chi2.sf(lr, P["all"]["k"])),
                    note="pooled fits ignore between-mouse heterogeneity: anti-conservative; the per-mouse parameter comparison is the main test")
    (C.FITS / "006_pooled.json").write_text(json.dumps(P, indent=1, default=float))
    print("pooled LRT cohort-specific vs common:", P["lrt"])
    L = []
    best = F[F.variant == "full"].set_index("mouse_id")
    for mid, g in groups:
        p = {k: best.loc[mid, k] for k in M.PARAMS}
        out = M.trace(p, *arrays(g))
        L.append(pd.DataFrame(out, columns=["p_lick", "V", "w_b", "w_S", "w_A", "w_W", "w_N", "delta"]).assign(
            mouse_id=mid, cohort=g.reward_group.iloc[0], session_id=g.session_id.to_numpy(), day=g.day.to_numpy(),
            trial_idx=g.trial_idx.to_numpy(), trial_type=g.trial_type.to_numpy(), lick=g.lick.to_numpy()))
    pd.concat(L).to_parquet(C.TAB / "006_latents.parquet")
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=60)
    ap.add_argument("--n-starts", type=int, default=10)
    main(ap.parse_args())
