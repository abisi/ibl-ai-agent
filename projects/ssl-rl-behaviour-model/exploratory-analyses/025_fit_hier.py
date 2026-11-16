"""025 -- model v7: hierarchical mouse-level fits of rl7 (v6) variants (model_v7_design.md; user 2026-10-23 "yes go" to: fix the
within-session shape, then mouse-specific parameters, recovery, then the R- subgroup question).
Per cohort, EM-Laplace (rl8_hier): mouse-level alpha+, alpha-, beta, c, s_B, mu_rho (+ theta for thr); cohort-level parameters fixed at
the 023 cohort fit; rho_s per session integrated out around the mouse's mu_rho.
Fit mode: per model, per-mouse MAP parameters, posterior sd, Laplace evidence; population mu/sd per EM iteration; per-trial latents
(weights, causal predictive); R+ vs R- per mouse-level parameter (mouse = unit; Mann-Whitney + Welch on the unbounded scale, and on
log(beta * alpha+)).
Recover mode (first model): (a) simulate every mouse from its MAP parameters (rho_s from its prior), refit; (b) mixed R- cohort: 11 random
R- mice with alpha- = 0.5, the rest alpha- = 0.001, refit; can the fast extinguishers be identified from the refit alpha-?
Output: fits/025_hier_mice.csv, fits/025_hier_pop.csv, tables/025_model_comparison.csv, tables/025_stats.csv, tables/025_latents/<model>.parquet,
fits/025_recovery_mice.csv, tables/025_recovery_summary.csv
Run (haas): OMP_NUM_THREADS=1 python 025_fit_hier.py --models sess-thr-inst-eng [...] [--recover] --n-jobs 60
Lean v7b (027; user 2026-10-24): --cohort-fits 027_fits.csv --variant lean-abs --lean --tag 025L: cohort parameters from the 027 variant,
mouse-level alpha+, beta, mu_rho (R+) and alpha+, alpha-, beta, mu_rho (R-) (+stim: alpha_A, alpha_W, beta, mu_rho in both); outputs
prefixed with the tag.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import rl7_model as M
import rl8_hier as H

F019 = importlib.import_module("019_fit_v4")
COHORTS = ["R+", "R-"]
LAT_DIR = C.TAB / "025_latents"
PFX = "025"
COHORT_FITS = None   # (file, variant) for the lean v7b
LEAN_KEYS = {"R+": ["alpha_pos", "beta", "mu_rho"], "R-": ["alpha_pos", "alpha_neg", "beta", "mu_rho"]}
ARR = {}          # mouse_id -> rl7 arrays (set before the pool forks)
JOB = {}          # name, cohort parameters


def cohort_params(name, cs):
    if COHORT_FITS is not None:
        F = pd.read_csv(C.FITS / COHORT_FITS[0])
        r = F[(F.variant == COHORT_FITS[1]) & (F.cohort_set == cs)].iloc[0]
        assert r.model == name
        return {k: r[f"p_{k}"] if pd.notna(r.get(f"p_{k}", np.nan)) else M.FIXED[k] for k in M.P_ALL}
    F = pd.concat([pd.read_csv(f) for f in sorted(C.FITS.glob("023*_fits.csv"))], ignore_index=True)
    r = F[(F.model == name) & (F.cohort_set == cs)].iloc[0]
    return {k: r.get(f"p_{k}", M.FIXED[k]) for k in M.P_ALL}


def e_job(args):
    mid, mu, sd, starts = args
    r = H.map_mouse(JOB["name"], ARR[mid], JOB["cohort_p"], mu, sd, starts, JOB["cs"])
    return mid, r


def em(name, cs, mice, n_jobs, max_iter=15, tol_mu=0.02):
    """EM-Laplace; the Laplace evidence is not monotone in the iterations (numerical Hessians, flat directions such as beta-c), so
    iterate until the population mean moves < tol_mu on the y scale (max_iter at most) and keep the last E-step (MAPs under the converged
    prior); the summed evidence of every iteration is in the pop table"""
    keys = H.mouse_params(name, cs)
    cp = cohort_params(name, cs)
    JOB.update(name=name, cohort_p=cp, cs=cs)
    mu = H.to_y(keys, [cp[k] for k in keys]); sd = np.full(len(keys), 1.5)
    yhat = {m: mu.copy() for m in mice}
    pop = []
    with mp.get_context("fork").Pool(n_jobs) as pool:
        for it in range(max_iter):
            res = dict(pool.map(e_job, [(m, mu, sd, [yhat[m], mu]) for m in mice], chunksize=1))
            Y = np.array([res[m]["y"] for m in mice])
            yhat = {m: res[m]["y"] for m in mice}
            tot = sum(res[m]["log_ev"] for m in mice)
            pop.append(dict(model=name, cohort_set=cs, iter=it, total_log_ev=tot, **{f"mu_{k}": v for k, v in zip(keys, mu)},
                            **{f"sd_{k}": v for k, v in zip(keys, sd)}))
            mu_new, sd = H.m_step(Y, [res[m]["Sig"] for m in mice])
            step = float(np.max(np.abs(mu_new - mu)))
            print(f"{name} {cs} iter {it}: sum log evidence {tot:.1f}, max |d mu| {step:.3f}", flush=True)
            mu = mu_new
            if step < tol_mu:
                break
    pop = pd.DataFrame(pop).assign(kept=lambda e: e.iter == e.iter.max())
    rows = []
    for m in mice:
        r = res[m]
        x = H.to_x(keys, r["y"])
        rows.append(dict(model=name, cohort_set=cs, mouse_id=m, n=len(ARR[m][0]), ll=r["ll"], log_ev=r["log_ev"],
                         **{f"p_{k}": v for k, v in zip(keys, x)}, **{f"y_{k}": v for k, v in zip(keys, r["y"])},
                         **{f"ysd_{k}": v for k, v in zip(keys, np.sqrt(np.diag(r["Sig"])))}))
    return pd.DataFrame(rows), pop


def fit_cohorts(name, D, n_jobs):
    out, pops = [], []
    for cs in COHORTS:
        d = D[D.reward_group == cs]
        if d.empty:
            continue
        ARR.clear()
        ARR.update({m: M.arrays(e.reset_index(drop=True)) for m, e in d.groupby("mouse_id", sort=False)})
        R, P = em(name, cs, list(ARR), n_jobs)
        out.append(R); pops.append(P)
    return pd.concat(out, ignore_index=True), pd.concat(pops, ignore_index=True)


def mouse_p(name, row, cs):
    p = cohort_params(name, cs)
    p.update({k: row[f"p_{k}"] for k in H.mouse_params(name, cs)})
    return p


def latents(name, D, R):
    out = []
    for (m, cs), e in D.groupby(["mouse_id", "reward_group"], sort=False):
        e = e.reset_index(drop=True)
        L = M.latents(name, mouse_p(name, R[R.mouse_id == m].iloc[0], cs), M.arrays(e))
        o = e[["mouse_id", "session_id", "trial_idx", "start_time", "day", "trial_type", "lick_flag", "reward_group"]].copy()
        o["p_lick"] = L["p_lick"]; o["lrho_smooth"] = L["lrho_smooth"]; o["p_engaged"] = L["p_engaged"]
        for i, q in enumerate(M.LATENTS):
            o[q] = L["lat"][:, i]
        out.append(o)
    return pd.concat(out, ignore_index=True)


def cohort_stats(R):
    rows = []
    for name, e in R.groupby("model"):
        e = e.assign(y_log_beta_alpha_pos=np.log(e.p_beta * e.p_alpha_pos))
        for q in [c for c in e if c.startswith("y_")]:
            a, b = e[e.cohort_set == "R+"][q].dropna(), e[e.cohort_set == "R-"][q].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            rows.append(dict(model=name, param=q[2:], median_rplus=a.median(), median_rminus=b.median(), n_rplus=len(a), n_rminus=len(b),
                             p_mwu=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return pd.DataFrame(rows)


def simulate_mice(name, D, R, rng, override=None):
    sim = D.copy()
    for (m, cs), e in D.groupby(["mouse_id", "reward_group"], sort=False):
        p = mouse_p(name, R[R.mouse_id == m].iloc[0], cs)
        if override and m in override:
            p.update(override[m])
        lk, _ = M.simulate(name, p, M.arrays(e.reset_index(drop=True)), rng)
        sim.loc[e.index, "lick_flag"] = lk
    return sim


def recover(name, D, R, n_jobs):
    rows, summ = [], []
    # (a) generated from the MAP parameters of every mouse
    sim = simulate_mice(name, D, R, np.random.default_rng(1))
    Rr, _ = fit_cohorts(name, sim, n_jobs)
    a = R.merge(Rr, on=["mouse_id", "cohort_set"], suffixes=("_true", "_refit"))
    rows.append(a.assign(scenario="map"))
    for cs in COHORTS:
        for k in H.mouse_params(name, cs):
            e = a[a.cohort_set == cs]
            summ.append(dict(scenario="map", param=k, cohort=cs, spearman=stats.spearmanr(e[f"y_{k}_true"], e[f"y_{k}_refit"])[0],
                             bias=float(np.median(e[f"y_{k}_refit"] - e[f"y_{k}_true"]))))
    st_true, st_refit = cohort_stats(R), cohort_stats(Rr)
    for _, r in st_true.iterrows():
        q = st_refit[st_refit.param == r.param]
        if q.empty:
            continue
        q = q.iloc[0]
        summ.append(dict(scenario="map", param=r.param, cohort="R+ vs R-", p_mwu_true=r.p_mwu, p_mwu_refit=q.p_mwu,
                         diff_true=r.median_rplus - r.median_rminus, diff_refit=q.median_rplus - q.median_rminus))
    # (b) mixed R- cohort: a quarter of the mice extinguish fast
    rm = sorted(R[R.cohort_set == "R-"].mouse_id)
    fast = set(np.random.default_rng(zlib.crc32(b"fast")).choice(rm, size=len(rm) // 4, replace=False))
    ov = {m: dict(alpha_neg=0.5 if m in fast else 1e-3) for m in rm}
    sim = simulate_mice(name, D, R, np.random.default_rng(2), override=ov)
    Rb, _ = fit_cohorts(name, sim[sim.reward_group == "R-"], n_jobs)
    b = Rb[Rb.cohort_set == "R-"].assign(fast=lambda e: e.mouse_id.isin(fast))
    rows.append(b.assign(scenario="mixed_rminus"))
    auc = stats.mannwhitneyu(b[b.fast].y_alpha_neg, b[~b.fast].y_alpha_neg).statistic / (b.fast.sum() * (~b.fast).sum())
    summ.append(dict(scenario="mixed_rminus", param="alpha_neg", cohort="R-", auc_fast_vs_slow=auc,
                     median_refit_fast=b[b.fast].p_alpha_neg.median(), median_refit_slow=b[~b.fast].p_alpha_neg.median()))
    print(pd.DataFrame(summ).round(3).to_string(), flush=True)
    pd.concat(rows, ignore_index=True).to_csv(C.FITS / f"{PFX}_recovery_mice.csv", index=False)
    pd.DataFrame(summ).to_csv(C.TAB / f"{PFX}_recovery_summary.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--cohort-fits", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--tag", default="025")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=60)
    a = ap.parse_args()
    PFX = a.tag
    LAT_DIR = C.TAB / f"{PFX}_latents"
    if a.cohort_fits:
        COHORT_FITS = (a.cohort_fits, a.variant)
        a.models = [pd.read_csv(C.FITS / a.cohort_fits).query("variant == @a.variant").model.iloc[0]]
    if a.lean:   # +stim: alpha_A (alpha_pos slot) and alpha_W (alpha_neg slot) act in both cohorts
        H.COHORT_KEYS = LEAN_KEYS if "+stim" not in a.models[0] else {cs: LEAN_KEYS["R-"] for cs in COHORTS}
    D = F019.load()
    LAT_DIR.mkdir(exist_ok=True)
    Rs, Ps = [], []
    for name in a.models:
        R, P = fit_cohorts(name, D, a.n_jobs)
        Rs.append(R); Ps.append(P)
        latents(name, D, R).to_parquet(LAT_DIR / f"{name}.parquet")
        pd.concat(Rs).to_csv(C.FITS / f"{PFX}_hier_mice.csv", index=False); pd.concat(Ps).to_csv(C.FITS / f"{PFX}_hier_pop.csv", index=False)
    R = pd.concat(Rs, ignore_index=True)
    n = len(D)
    S = R.groupby(["model"]).agg(sum_log_ev=("log_ev", "sum"), ll=("ll", "sum")).reset_index()
    S["d"] = [sum(2 * len(H.mouse_params(m, cs)) for cs in COHORTS) / 2 for m in S.model]   # population mean + sd per parameter, per cohort
    S["ibic"] = -2 * S.sum_log_ev + 2 * S.d * np.log(n)                          # per cohort, summed over the 2 cohorts
    S.to_csv(C.TAB / f"{PFX}_model_comparison.csv", index=False); print(S.to_string(), flush=True)
    st = cohort_stats(R); st.to_csv(C.TAB / f"{PFX}_stats.csv", index=False); print(st.round(4).to_string(), flush=True)
    if a.recover:
        recover(a.models[0], D, R[R.model == a.models[0]], a.n_jobs)
