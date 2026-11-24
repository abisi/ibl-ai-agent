"""TCA on the event-aligned tensors of 005: decomposition, rank selection, model comparison (user 2026-11-22).

Input per trial / unit: firing rate (Hz, unsmoothed; 50-ms bins, sensory window 10 ms) minus that trial's baseline rate
(start - 1.0 .. start - 0.010 s), so slow baseline changes are removed. Signed data -> unconstrained CP (tensortools
cp_als; mcp_als where bins are missing). Units with mean rate < 0.5 Hz (window + baseline) dropped.
Normalisations (compared):
  rate    baseline-subtracted rate in Hz (high-rate units weigh more)
  soft    / (max over time of |trial-averaged change| + 5 Hz) per unit
  zscore  / SD of the unit's baseline-subtracted values over trials x time (user 2026-11-22, instead of min-max); not
          centred, so 0 stays the trial baseline. Every unit gets equal total variance, noise included.
TCA types = alignments (stim_long, stim_sensory, jaw, lick) x normalisations (rate, soft, zscore) = 12.
Rank selection (Williams et al. 2018, Neuron): ranks 1..20, N_REP random restarts each; reconstruction error of the
best fit vs rank and similarity of the other restarts to the best fit (kruskal_align). Chosen rank: the largest rank
whose restarts still agree (mean similarity >= SIM_MIN) and whose added component still lowers the error by at least
GAIN_MIN of the total decrease from rank 1 to rank 20. No held-out data.
Model comparison (in-sample): error of the condition-mean model (each unit's mean time course per class: context
passive / active x trial type x lick), with its parameter count, next to TCA's.

Speed (user 2026-11-22): one job per session x type x rank (not per type), one BLAS thread per worker.

Run: python 006_tca_aligned.py --sessions AB123_20240806_110231 [--n-jobs 100]
     -> OUT/artifacts/tca_aligned/<sid>/<alignment>_<norm>.npz, tables/tca_aligned_summary.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker
    os.environ.setdefault(_v, "1")   # (haas 2026-11-22: workers used 3-6 cores each, load ~390 on 112 cores)
import argparse
import importlib
import time
import traceback

import numpy as np
import pandas as pd
import tensortools as tt
from joblib import Parallel, delayed

import tca_lib as L

A = importlib.import_module("005_build_aligned")
NORMS = ["rate", "soft", "zscore"]
RANKS = range(1, 21)   # 1..20: no elbow by rank 10 in the AB123 pilot (2026-11-22)
N_REP = 10
TOL = 1e-6
MAX_ITER = 500
SIM_MIN = 0.85
GAIN_MIN = 0.05
MIN_RATE = 0.5
OUT_DIR = L.OUT / "artifacts" / "tca_aligned"


def prepare(sid, align, norm):
    d = np.load(A.OUT_DIR / sid / f"{align}.npz")
    R, base = d["rate"], d["baseline"]
    keep_u = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= MIN_RATE
    D = R[:, keep_u] - base[:, keep_u, None]
    if norm == "soft":
        D = D / (np.nanmax(np.abs(np.nanmean(D, 0, keepdims=True)), 2, keepdims=True) + 5.0)
    elif norm == "zscore":   # / SD over trials x time, not centred (0 stays the trial baseline)
        D = D / (np.nanstd(D, (0, 2), keepdims=True) + 1e-9)
    obs = ~np.isnan(D)
    t = pd.read_parquet(A.OUT_DIR / sid / "trials.parquet").set_index("trial_index").loc[d["trial_index"]]
    cls = (t.epoch.str.startswith("passive").map({True: "P", False: "A"}) + "_" + t.trial_type + "_"
           + t.lick_flag.fillna(0).astype(int).astype(str)).to_numpy()
    return np.nan_to_num(D).astype(np.float64), obs, keep_u, cls, d


def cond_mean_error(Y, obs, cls):
    """in-sample relative error of the condition-mean model (unit's mean time course per class) and its size."""
    P = np.zeros_like(Y)
    for c in np.unique(cls):
        m = cls == c
        w = obs[m]
        P[m] = (Y[m] * w).sum(0, keepdims=True) / np.maximum(w.sum(0, keepdims=True), 1)
    return np.linalg.norm((Y - P)[obs]) / np.linalg.norm(Y[obs]), len(np.unique(cls)) * Y.shape[1] * Y.shape[2]


def choose_rank(err, sim):
    """Williams et al. 2018 trade-off: largest rank with mean restart similarity >= SIM_MIN whose component still
    lowers the error by >= GAIN_MIN of the total decrease (rank 1 always admissible)."""
    R = np.array(list(RANKS))
    gain = np.r_[np.inf, -np.diff(err)] / max(err[0] - err[-1], 1e-12)
    ok = (sim >= SIM_MIN) & (gain >= GAIN_MIN)
    ok[0] = True
    return int(R[np.flatnonzero(ok)[-1]]), gain


def fit_rank(sid, align, norm, r, seed=0):
    """N_REP restarts at one rank (one job; jobs = sessions x types x ranks, for speed)."""
    t0 = time.time()
    Y, obs, keep_u, cls, d = prepare(sid, align, norm)
    rng = np.random.default_rng([seed, r, list(A.ALIGN).index(align), NORMS.index(norm)])
    M = None if obs.all() else obs
    kw = dict(verbose=False, max_iter=MAX_ITER, tol=TOL)
    fits = sorted(((tt.cp_als(Y, r, random_state=int(rng.integers(1 << 31)), **kw) if M is None else
                    tt.mcp_als(Y, r, M, random_state=int(rng.integers(1 << 31)), **kw)) for _ in range(N_REP)),
                  key=lambda f: f.obj)
    best = fits[0].factors
    best.rebalance()
    for c in range(r):   # sign convention: unit and time factors positive sum, sign on the trial factor
        for m in (1, 2):
            sg = np.sign(best.factors[m][:, c].sum()) or 1.0
            best.factors[m][:, c] *= sg
            best.factors[0][:, c] *= sg
    out = {f"r{r}_obj": np.array([f.obj for f in fits]),
           f"r{r}_sim": np.array([tt.kruskal_align(best, f.factors, permute_U=False, permute_V=False) for f in fits[1:]]),
           f"r{r}_iters": np.array([len(f.obj_hist) for f in fits]), f"r{r}_secs": time.time() - t0}
    for m, U in enumerate(best.factors):
        out[f"r{r}_f{m}"] = U.astype(np.float32)
    return (sid, align, norm, r), out


def assemble(sid, align, norm, parts):
    out = OUT_DIR / sid / f"{align}_{norm}.npz"
    Y, obs, keep_u, cls, d = prepare(sid, align, norm)
    cm_err, cm_par = cond_mean_error(Y, obs, cls)
    res = {"units_kept": np.flatnonzero(keep_u), "trial_index": d["trial_index"], "t_centres": d["t_centres"],
           "classes": cls.astype(str), "norm_Y": np.linalg.norm(Y), "shape": np.array(Y.shape),
           "condmean_err": cm_err, "condmean_params": cm_par, "tol": TOL, "max_iter": MAX_ITER}
    for p in parts:
        res.update(p)
    err = np.array([res[f"r{r}_obj"].min() for r in RANKS])
    sim = np.array([res[f"r{r}_sim"].mean() if len(res[f"r{r}_sim"]) else 1.0 for r in RANKS])
    res["rank_chosen"], res["gain"] = choose_rank(err, sim)
    res["sim_min"], res["gain_min"] = SIM_MIN, GAIN_MIN
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    rc = res["rank_chosen"]
    return dict(session_id=sid, alignment=align, norm=norm, ok=True, shape=str(Y.shape), rank_chosen=rc,
                err_chosen=round(float(err[rc - 1]), 4), sim_chosen=round(float(sim[rc - 1]), 3),
                tca_params=int(rc * sum(Y.shape)), condmean_err=round(float(cm_err), 4), condmean_params=int(cm_par),
                cpu_secs=round(sum(res[f"r{r}_secs"] for r in RANKS)),
                max_iters_hit=int(sum((res[f"r{r}_iters"] >= MAX_ITER).sum() for r in RANKS)))


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return a[:4], {"error": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--alignments", nargs="*", default=list(A.ALIGN))
    ap.add_argument("--norms", nargs="*", default=NORMS)
    ap.add_argument("--n-jobs", type=int, default=100)
    a = ap.parse_args()
    jobs = [(s, al, n, r) for s in a.sessions for al in a.alignments for n in a.norms for r in RANKS]
    print(f"{len(jobs)} jobs", flush=True)
    got = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(delayed(safe)(fit_rank, *j) for j in jobs)
    groups = {}
    for key, part in got:
        groups.setdefault(key[:3], []).append(part)
    res = []
    for key, parts in groups.items():
        errs = [p["error"] for p in parts if "error" in p]
        if errs:
            res.append(dict(session_id=key[0], alignment=key[1], norm=key[2], ok=False, error=errs[0]))
        else:
            res.append(assemble(*key, parts))
        print(res[-1], flush=True)
    f = L.OUT / "tables" / "tca_aligned_summary.csv"
    out = pd.DataFrame(res)
    if f.exists():
        out = pd.concat([pd.read_csv(f), out]).drop_duplicates(["session_id", "alignment", "norm"], keep="last")
    out.to_csv(f, index=False)


if __name__ == "__main__":
    main()
