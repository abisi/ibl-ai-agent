"""036 -- model-recovery matrix (Wilson & Collins 2019) for the final candidates (TODO 2026-10-30).

Per cohort and generating model: simulate every mouse's actual trial sequence (stimuli, R+ / R- outcomes) from the generating model's
cohort fit (session rho_s and u_s drawn from their priors), refit every candidate model on the simulated cohort (starts: the candidate's
full-data optimum, the generating parameters where they apply, defaults and random starts), and record BIC. A model is recovered when the
generating model has the lowest BIC on its own data. n_sim simulated datasets per generating model.
Output: tables/036_model_recovery.csv (one row per cohort, generator, sim, fitted model: nll, k, bic), tables/036_confusion.csv
Run (haas): python 036_model_recovery.py --models core core-pers ... --fits 027l_fits.csv 027m_fits.csv [--n-sim 2] [--n-jobs 40]
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd

import common as C
import rl7_model as M

L027 = importlib.import_module("027_lean_model")
F019 = importlib.import_module("019_fit_v4")
F023 = importlib.import_module("023_v6_prediction_check")
D = None
FITS = None


def row(variant, cs):
    return FITS[(FITS.variant == variant) & (FITS.cohort_set == cs)].sort_values("nll").iloc[0]


def params(r):
    return {k: (r[f"p_{k}"] if f"p_{k}" in r and pd.notna(r[f"p_{k}"]) else M.FIXED[k]) for k in M.P_ALL}


def job(args):
    gen, fit_v, cs, sim = args
    d = F023.subset(D, cs)
    A = M.arrays(d)
    rg = row(gen, cs)
    L027.configure(gen, cs)
    lick, _ = M.simulate(rg.model, params(rg), A, np.random.default_rng(zlib.crc32(f"036{gen}{cs}{sim}".encode())))
    As = (A[0], lick, A[2], A[3], A[4])
    name, fp = L027.configure(fit_v, cs)
    orig = M.free_params
    M.free_params = lambda n: fp
    try:
        starts = []
        for r in (row(fit_v, cs), rg):
            starts.append([float(np.clip(r.get(f"p_{q}", M.DEFAULT[q]) if pd.notna(r.get(f"p_{q}", np.nan)) else M.DEFAULT[q],
                                         *M.BOUNDS[q])) for q in fp])
        rng = np.random.default_rng(zlib.crc32(f"036fit{gen}{fit_v}{cs}{sim}".encode()))
        p, nll = M.fit(name, As, np.ones(len(lick), np.int64), rng, n_random=2, extra_starts=starts)
    finally:
        M.free_params = orig
    k = len(fp)
    return dict(cohort_set=cs, generator=gen, sim=sim, fitted=fit_v, k=k, n=len(lick), nll=nll, bic=2 * nll + k * np.log(len(lick)),
                lick_rate=float(lick.mean()), **{f"p_{q}": p[q] for q in fp})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fits", nargs="+", required=True)
    ap.add_argument("--shared-g", type=float, default=None, help="register the shared-g variants of 027 (simple-genS) first")
    ap.add_argument("--extra", nargs="*", default=[], help="tag=opt1,opt2[,g=0.12] core variants to register (027 add_core_variant)")
    ap.add_argument("--n-sim", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--out", default="036_model_recovery.csv")
    a = ap.parse_args()
    if a.shared_g is not None:
        L027.shared_g(a.shared_g)
    for e in a.extra:
        tag, opts = e.split("=", 1)
        o = opts.split(",")
        g = [float(x[2:]) for x in o if x.startswith("g=")]
        L027.add_core_variant(tag, [x for x in o if x and not x.startswith("g=")], g[0] if g else None)
    D = F019.load()
    FITS = pd.concat([pd.read_csv(C.FITS / f) for f in a.fits], ignore_index=True)
    jobs = [(g, f, cs, s) for cs in L027.COHORTS for g in a.models for s in range(a.n_sim) for f in a.models]
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        R = pd.DataFrame(pool.map(job, jobs, chunksize=1))
    R.to_csv(C.TAB / a.out, index=False)
    best = R.loc[R.groupby(["cohort_set", "generator", "sim"]).bic.idxmin()]
    Cf = best.groupby(["cohort_set", "generator"]).fitted.value_counts(normalize=True).unstack(fill_value=0).reset_index()
    Cf.to_csv(C.TAB / a.out.replace(".csv", "_confusion.csv"), index=False)
    dB = R.merge(R[R.fitted == R.generator][["cohort_set", "generator", "sim", "bic"]], on=["cohort_set", "generator", "sim"],
                 suffixes=("", "_true"))
    dB["d_bic_vs_true"] = dB.bic - dB.bic_true
    print(Cf.to_string(), flush=True)
    print(dB.groupby(["cohort_set", "generator", "fitted"]).d_bic_vs_true.mean().round(1).unstack().to_string(), flush=True)
