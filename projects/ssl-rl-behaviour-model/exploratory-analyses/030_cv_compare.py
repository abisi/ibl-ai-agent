"""030 -- held-out log-likelihood per trial (leave-mice-out cross-validation) next to BIC, for the 027 variants.

Per cohort, mice are split into K folds; the cohort-level parameters are refitted on the other folds (warm-started from the full-data
optimum) and the held-out mice are scored. Mice are conditionally independent given the shared parameters, so the held-out likelihood
is the kernel likelihood with a trial mask. Scores are per held-out mouse (nll, n trials), so variants can be compared paired over mice.
Baseline: a lick probability per trial type fitted on the training mice (pseudo-R^2 = 1 - LL_model / LL_baseline).
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
DEFAULT_VARIANTS = ["lean-abs-stim", "lean-abs-stim-hom", "lean-abs-stim-nodis", "lean-abs-stim-hom-nodis"]
FIT_FILES = ["027_fits.csv", "027b_fits.csv", "027c_fits.csv", "027d_fits.csv", "027e_fits.csv", "027f_fits.csv", "027h_fits.csv",
             "027k_fits.csv", "027l_fits.csv", "027m_fits.csv", "027n_fits.csv"]
K = 5
D = None


def full_fit(variant, cs):
    rows = [pd.read_csv(C.FITS / f) for f in FIT_FILES if (C.FITS / f).exists()]
    F = pd.concat(rows, ignore_index=True)
    r = F[(F.variant == variant) & (F.cohort_set == cs)]
    return r.sort_values("nll").iloc[0] if len(r) else None


def folds(d, cs):
    mice = np.array(sorted(d.mouse_id.unique()))
    rng = np.random.default_rng(zlib.crc32(f"030{cs}".encode()))
    rng.shuffle(mice)
    return {m: i % K for i, m in enumerate(mice)}


def job(args):
    variant, cs, k = args
    name, fp = L027.configure(variant, cs)
    orig = M.free_params
    M.free_params = lambda n: fp
    try:
        d = F023.subset(D, cs)
        A = M.arrays(d)
        fold = d.mouse_id.map(folds(d, cs)).to_numpy()
        train = (fold != k).astype(np.int64)
        starts = []
        for v in (variant, "lean-abs-stim"):
            r = full_fit(v, cs)
            if r is not None:
                starts.append([float(np.clip(r.get(f"p_{q}", M.DEFAULT[q]), *M.BOUNDS[q])) for q in fp])
        rng = np.random.default_rng(zlib.crc32(f"030{variant}{cs}{k}".encode()))
        p, nll_train = M.fit(name, A, train, rng, n_random=2, extra_starts=starts)
        x = [p[q] for q in fp]
        out = []
        for m in d.mouse_id[fold == k].unique():
            sel = (d.mouse_id == m).to_numpy()
            mask = sel.astype(np.int64)
            ll = M.loglik(name, x, A, mask)
            # baseline: per-trial-type lick probability from the training mice
            tr = d[train == 1]
            pb = tr.groupby("trial_type").lick_flag.mean().clip(1e-3, 1 - 1e-3)
            dm = d[sel]
            q = dm.trial_type.map(pb).to_numpy()
            llb = float(np.sum(np.where(dm.lick_flag == 1, np.log(q), np.log1p(-q))))
            out.append(dict(variant=variant, cohort_set=cs, fold=k, mouse_id=m, n=int(sel.sum()), nll=-ll, nll_base=-llb,
                            nll_train=nll_train, **{f"p_{q}": p[q] for q in fp}))
        return out
    finally:
        M.free_params = orig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--out", default="030_cv.csv")
    a = ap.parse_args()
    D = F019.load()
    jobs = [(v, cs, k) for v in a.variants for cs in L027.COHORTS for k in range(K)]
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        R = pd.DataFrame([r for rs in pool.map(job, jobs, chunksize=1) for r in rs])
    out = C.TAB / a.out
    if out.exists():
        old = pd.read_csv(out)
        R = pd.concat([old[~old.variant.isin(a.variants)], R], ignore_index=True)
    R.to_csv(out, index=False)
    # summary: held-out LL per trial, pseudo-R^2, BIC from the full-data fits, paired difference vs the first variant
    rows = []
    for (v, cs), g in R.groupby(["variant", "cohort_set"]):
        f = full_fit(v, cs)
        rows.append(dict(variant=v, cohort_set=cs, n_mice=len(g), ll_per_trial=-g.nll.sum() / g.n.sum(),
                         pseudo_r2=1 - g.nll.sum() / g.nll_base.sum(), bic=f.bic if f is not None else np.nan,
                         k=f.k if f is not None else np.nan))
    S = pd.DataFrame(rows)
    ref = a.variants[0]
    for i, r in S.iterrows():
        x = R[(R.variant == r.variant) & (R.cohort_set == r.cohort_set)].set_index("mouse_id")
        y = R[(R.variant == ref) & (R.cohort_set == r.cohort_set)].set_index("mouse_id")
        dd = ((y.nll - x.nll) / x.n).dropna()           # > 0: variant better than the reference on that mouse
        S.loc[i, "d_ll_per_trial_vs_ref"] = dd.mean()
        S.loc[i, "se"] = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
        S.loc[i, "n_mice_better"] = int((dd > 0).sum())
    S.to_csv(C.TAB / a.out.replace(".csv", "_summary.csv"), index=False)
    print(S.round(5).to_string(), flush=True)
