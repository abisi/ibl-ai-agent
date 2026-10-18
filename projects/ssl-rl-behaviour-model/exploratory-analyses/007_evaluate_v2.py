"""007 -- Evaluation of the v2 model (rl2_model) fitted in 006.
Per mouse, on its own trial sequence:
  parameter recovery  licks simulated from the mouse's "full" fit, refitted with "full"
  model recovery      licks simulated from the mouse's fit of each strategy variant (full, specific, general, no_bias), every simulated
                      data set fitted with the four variants; best by BIC -> confusion matrix
  predictive curves   P(lick) curves, time-normalised exactly as the data targets (rl_data_utils.get_mouse_curves, no extra trim):
                        data       observed licks
                        one-step   model P(lick) given the mouse's actual history (fit quality)
                        generative mean P(lick) of N_SIM closed-loop simulations (licks drawn from the model; does the strategy
                                   reproduce the learning curves on its own?)
                        swapped    generative simulations with the OTHER cohort's whisker outcome and the mouse's own parameters
                                   (is the cohort difference produced by the contingency alone?)
Output: tables/007_param_recovery.csv, 007_model_recovery.csv, 007_curves.npz (per mouse x kind x trial type, 2000 points)
Run (haas): python 007_evaluate_v2.py [--n-jobs 60] [--n-sim 20]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd

import common as C
import rl2_model as M
import rl_data_utils as du

STRAT = ["full", "specific", "general", "no_bias"]
KINDS = ["data", "onestep", "generative", "swapped"]


def curves(g, lick):
    m = g[["mouse_id", "day", "trial_idx", "trial_type"]].copy()
    m["lick_flag"] = lick
    c = du.get_mouse_curves(m, trim_last=0)
    return None if c is None else np.stack([c[k] for k in ("auditory", "whisker", "no_stim")])


def evaluate(args):
    mid, g, F, n_sim = args
    seed = zlib.crc32(mid.encode())
    cohort = g.reward_group.iloc[0]
    tt = g.tt.to_numpy(np.int64); lick = g.lick.to_numpy(np.int64)
    r_own, r_swap = M.outcomes(tt, cohort), M.outcomes(tt, "R-" if cohort == "R+" else "R+")
    par = {v: {k: F.loc[(mid, v), k] for k in M.PARAMS} for v in STRAT}
    rec, conf = [], []
    for gen in STRAT:
        _, L = M.simulate(par[gen], tt, r_own, seed=seed + 1)
        fits = {v: M.fit(v, [(tt, L, r_own)], n_starts=10, seed=seed) for v in STRAT}
        best = min(fits, key=lambda v: fits[v]["bic"])
        conf.append(dict(mouse_id=mid, cohort=cohort, generating=gen, recovered=best, **{f"bic_{v}": fits[v]["bic"] for v in STRAT}))
        if gen == "full":
            for k in M.VARIANTS["full"]["free"]:
                rec.append(dict(mouse_id=mid, cohort=cohort, param=k, true=par["full"][k], fitted=fits["full"]["params"][k]))
            for name, f in (("beta_x_alpha_pos", lambda p: p["beta"] * p["alpha_pos"]), ("alpha_ratio", lambda p: p["alpha_neg"] / p["alpha_pos"])):
                rec.append(dict(mouse_id=mid, cohort=cohort, param=name, true=f(par["full"]), fitted=f(fits["full"]["params"])))
    out = {"data": curves(g, lick), "onestep": curves(g, M.trace(par["full"], tt, lick, r_own)[:, 0])}
    for kind, r in (("generative", r_own), ("swapped", r_swap)):
        P = np.mean([M.simulate(par["full"], tt, r, seed=seed + 100 + s)[0] for s in range(n_sim)], axis=0)
        out[kind] = curves(g, P)
    return mid, cohort, rec, conf, out


def main(a):
    d = pd.read_parquet(C.TAB / "006_sequences.parquet")
    F = pd.read_csv(C.FITS / "006_fits.csv").set_index(["mouse_id", "variant"])
    jobs = [(m, g, F, a.n_sim) for m, g in d.groupby("mouse_id")]
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        res = pool.map(evaluate, jobs)
    pd.DataFrame([x for r in res for x in r[2]]).to_csv(C.TAB / "007_param_recovery.csv", index=False)
    Mr = pd.DataFrame([x for r in res for x in r[3]]); Mr.to_csv(C.TAB / "007_model_recovery.csv", index=False)
    print(pd.crosstab(Mr.generating, Mr.recovered))
    ok = [r for r in res if all(r[4][k] is not None for k in KINDS)]
    np.savez_compressed(C.TAB / "007_curves.npz", mouse_id=np.array([r[0] for r in ok]), cohort=np.array([r[1] for r in ok]),
                        kinds=np.array(KINDS), curves=np.stack([np.stack([r[4][k] for k in KINDS]) for r in ok]))
    print("curves for", len(ok), "mice; ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=60)
    ap.add_argument("--n-sim", type=int, default=20)
    main(ap.parse_args())
