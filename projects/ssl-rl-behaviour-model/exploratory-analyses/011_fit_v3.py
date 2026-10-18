"""011 -- Fit model v3 (rl3_model; model_v3_design.md) to every mouse: all 64 models, full data and leave-one-session-out (LOSO).
Trials: tables/001_trials.csv (active trials, perf != 6), free-licking sessions and the few whisker trials on pretraining days removed;
ordered by day and trial start; no end-of-session trimming (decided 2026-10-18).
Per mouse:
  full fit      all trials scored: parameters, NLL, BIC (k = free parameters, n = trials)
  LOSO          for each session s: parameters fitted on all other sessions (the model still runs through session s with the actual licks,
                so the state is right afterwards), session s scored by one-step log-likelihood; summed over sessions = held-out LL;
                day-0 held-out LL = the first whisker session's fold
Output: tables/011_sequences.parquet, fits/011_full.csv (params per mouse x model), fits/011_loso.csv (held-out LL per mouse x model x
session), tables/011_latents_<model>.parquet for the full model (per trial: P(lick), Q, weights, thirst, Rbar, DV, prediction error)
Run (haas): python 011_fit_v3.py [--n-jobs 100] [--n-random 2] [--mice N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import zlib

import numpy as np
import pandas as pd

import common as C
import rl3_model as M


def sequences():
    d = pd.read_csv(C.TRIALS)
    d = d[(d.bhv_type != "free_licking") & ~((d.trial_type == "whisker") & (d.day < 0))]
    return d.sort_values(["mouse_id", "day", "start_time"]).reset_index(drop=True)


def run_mouse(args):
    mid, g, n_random = args
    t0 = time.time()
    A = M.sequence_arrays(g)
    n = len(g); seed = zlib.crc32(mid.encode())
    sess = g.session_id.to_numpy(); order = list(dict.fromkeys(sess))
    d0 = g.session_id[g.day == 0].iloc[0] if (g.day == 0).any() else None
    full = M.fit_all_models(A, np.ones(n, np.int64), n_random, seed)
    rows_full = [dict(mouse_id=mid, cohort=g.reward_group.iloc[0], model=m, k=len(M.free_params(m)), n=n, nll=nll,
                      bic=2 * nll + len(M.free_params(m)) * np.log(n), **p) for m, (p, nll) in full.items()]
    rows_loso = []
    for s in order:
        test = (sess == s).astype(np.int64)
        res = M.fit_all_models(A, 1 - test, n_random, seed + 1)
        for m, (p, _) in res.items():
            P, _ = M.full_vector(m, [p[k] for k in M.free_params(m)])
            rows_loso.append(dict(mouse_id=mid, cohort=g.reward_group.iloc[0], model=m, session_id=s, day=int(g.day[sess == s].iloc[0]),
                                  is_day0=s == d0, n_test=int(test.sum()), ll_test=M.loglik(P, A, test)))
    return rows_full, rows_loso, time.time() - t0


def main(a):
    d = sequences()
    d.to_parquet(C.TAB / "011_sequences.parquet")
    groups = list(d.groupby("mouse_id"))
    if a.mice:
        groups = groups[:a.mice]
    print(f"{len(groups)} mice, {sum(len(g) for _, g in groups):,} trials, {len(M.MODELS)} models")
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        res = pool.map(run_mouse, [(m, g, a.n_random) for m, g in groups], chunksize=1)
    F = pd.DataFrame([r for x in res for r in x[0]]); L = pd.DataFrame([r for x in res for r in x[1]])
    tag = "" if not a.mice else f"_test{a.mice}"
    F.to_csv(C.FITS / f"011_full{tag}.csv", index=False); L.to_csv(C.FITS / f"011_loso{tag}.csv", index=False)
    print("seconds per mouse:", np.round([x[2] for x in res], 0))
    H = L.groupby(["mouse_id", "model"]).ll_test.sum().unstack()
    print("held-out LL summed over mice (best 10):"); print(H.sum().sort_values(ascending=False).head(10).round(1))
    if not a.mice:
        full = F[F.model == "GMNZRP"].set_index("mouse_id")
        Ls = []
        for mid, g in groups:
            P = np.array([full.loc[mid, k] for k in M.PARAMS], float)
            out = M.trace(P, M.sequence_arrays(g))
            Ls.append(pd.DataFrame(out, columns=["p_lick", "Q", "w_b", "w_S", "w_A", "w_W", "thirst", "Rbar", "DV", "delta"]).assign(
                mouse_id=mid, cohort=g.reward_group.iloc[0], session_id=g.session_id.to_numpy(), day=g.day.to_numpy(),
                trial_idx=g.trial_idx.to_numpy(), trial_type=g.trial_type.to_numpy(), lick=g.lick_flag.to_numpy()))
        pd.concat(Ls).to_parquet(C.TAB / "011_latents_GMNZRP.parquet")
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--mice", type=int, default=0)
    main(ap.parse_args())
