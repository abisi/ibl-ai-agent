"""007 -- Slow drift of pre-trial population activity (Cowley et al. 2020) and its relation to learning (user 2026-10-16, "Do 4": an
unsupervised axis, not built from task labels; does it follow learning, beyond elapsed time, is it shared across areas, does it differ
R+ vs R-?).
Rates: 005 pre-trial baseline [-1.0, -0.015] s (raw), all analysed trials, units with mean >= 0.1 spikes/s, z-scored per unit across trials.
Two components per unit: a cubic time trend (regression on a cubic polynomial of tau) and the remaining slow fluctuation (residuals,
centred moving average over SMOOTH trials, edge: nearest). A first version without detrending gave a drift PC1 with r(drift, tau) ~ 0.97,
i.e. the time trend itself, collinear with the time nuisance of the behaviour test (2026-10-16); the drift is therefore the detrended
fluctuation. Drift axis = PC1 of the smoothed fluctuation; drift(t) = its projection, oriented so the mean unit loading is positive
(+ = overall higher baseline), z-scored over trials. Per session:
  f_trend    fraction of the raw baseline variance explained by the cubic time trend
  f_slow     fraction of the raw baseline variance in the slow fluctuation (smoothed detrended unit variances / raw unit variances)
  ev1        fraction of the slow fluctuation on PC1 (one dominant slow mode?)
  beta_W/A/N standardised logistic coefficient of drift on licking in whisker / auditory / no-stimulus trials, with a cubic polynomial of
             tau as nuisance (drift beyond time); p from a likelihood-ratio test; needs >= 5 licks and >= 5 no-licks
  pr_curve   day 0: partial correlation (tau-cubic residuals) of drift with the whisker learning curve (024, sigma 1) at whisker trials
Areas (>= 10 units): area drift computed the same way; r_rest = correlation with the drift of the rest of the brain (all other units, so
no shared units), minus a linear-shift null (rest-of-brain drift shifted circularly by 10-50% of trials, 50 shifts) = r_rest_ex.
Statistics: common.measure_stats (learners rule on day 0, session = unit), area level BH over areas.
Output: tables/007_sessions.csv, 007_drift.parquet (per-trial drift), 007_stats.csv, 007_area_stats.csv; figures/007_*.{png,pdf,svg}
Run (haas, prelick environment): python 007_slow_drift.py [--n-proc 25] [--plot-only]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d

import common as C

SRC = C.TAB / "005_trial_windows"
LTP = C.PROJECTS / "ssl-learning-trial-identification" / "artifacts"
SMOOTH, MIN_FR, MIN_UNITS, N_SHIFT = 21, 0.1, 10, 50
MEAS = [("f_trend", "time-trend share of\nbaseline variance"), ("f_slow", "slow-fluctuation share\nof baseline variance"),
        ("ev1", "PC1 share of the\nslow fluctuation"),
        ("beta_W", "drift → whisker lick\n(β, beyond time)"), ("beta_A", "drift → auditory lick\n(β, beyond time)"),
        ("beta_N", "drift → no-stim lick\n(β, beyond time)"), ("pr_curve", "partial r(drift,\nwhisker curve), day 0")]


def cubic(tau):
    t = (tau - tau.mean()) / (tau.std() + 1e-12)
    return np.c_[np.ones_like(t), t, t ** 2, t ** 3]


def drift(Z, tau):
    """Z: units x trials, z-scored. Splits the baseline into a cubic time trend and the remaining slow fluctuation.
    Returns the fluctuation drift d (trials; PC1 of the smoothed detrended matrix, oriented so the mean unit loading is positive, i.e.
    + = overall higher baseline; z-scored), ev1 (PC1 share of the smoothed fluctuation variance), f_slow (slow fluctuation share of the raw
    variance), f_trend (cubic-trend share of the raw variance)"""
    X = cubic(tau)
    T = (X @ np.linalg.lstsq(X, Z.T, rcond=None)[0]).T
    Rz = Z - T
    Sm = uniform_filter1d(Rz, SMOOTH, axis=1, mode="nearest")
    Sc = Sm - Sm.mean(1, keepdims=True)
    u, s, vt = np.linalg.svd(Sc, full_matrices=False)
    d = s[0] * vt[0] * (1 if u[:, 0].sum() >= 0 else -1)
    tot = Z.var(1).sum()
    return (d - d.mean()) / d.std(), s[0] ** 2 / (s ** 2).sum(), Sc.var(1).sum() / tot, T.var(1).sum() / tot


def logit_ll(X, y, n_iter=50):
    """IRLS logistic regression with a small ridge; returns log-likelihood, coefficients"""
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p) - 1e-4 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), w


def beyond_time(d, tau, y):
    if len(y) < 15 or min(y.sum(), (1 - y).sum()) < 5:
        return np.nan, np.nan
    X0 = cubic(tau)
    l0, _ = logit_ll(X0, y); l1, w = logit_ll(np.c_[X0, d], y)
    return w[-1], stats.chi2.sf(2 * max(l1 - l0, 0), 1)


def resid_cubic(y, tau):
    X = cubic(tau)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def zscore(B):
    sd = B.std(1, keepdims=True); sd[sd == 0] = 1
    return (B - B.mean(1, keepdims=True)) / sd


def session(args):
    sid, curve = args
    z = np.load(SRC / f"{sid}.npz", allow_pickle=True)
    B = z["base"].astype(float); keep = B.mean(1) >= MIN_FR
    Z = zscore(B[keep]); ag = z["area_group"][keep].astype(str)
    tau, tt, lick, wt = z["tau"], z["trial_type"], z["lick"].astype(float), z["wtrial"]
    if Z.shape[0] < MIN_UNITS:
        return None, None, []
    d, ev1, f_slow, f_trend = drift(Z, tau)
    out = dict(session_id=sid, n_units=int(keep.sum()), n_trials=len(tau), ev1=ev1, f_slow=f_slow, f_trend=f_trend)
    for k in ("W", "A", "N"):
        m = tt == k
        out[f"beta_{k}"], out[f"p_{k}"] = beyond_time(d[m], tau[m], lick[m])
    iw = np.where(tt == "W")[0]
    out["pr_curve"] = np.nan
    if curve is not None and len(curve) == len(iw):
        c = np.asarray(curve, float); ok = np.isfinite(c)
        if ok.sum() >= 15 and np.std(c[ok]) > 0:
            out["pr_curve"] = np.corrcoef(resid_cubic(d[iw][ok], tau[iw][ok]), resid_cubic(c[ok], tau[iw][ok]))[0, 1]
    tr = pd.DataFrame(dict(session_id=sid, trial_index=np.arange(len(tau)), tau=tau, trial_type=tt, lick=lick, wtrial=wt, drift=d))
    areas = []
    rng = np.random.default_rng(0)
    for a in np.unique(ag):
        m = ag == a
        if m.sum() < MIN_UNITS or (~m).sum() < MIN_UNITS:
            continue
        da, _, fa, fta = drift(Z[m], tau); dr, _, _, _ = drift(Z[~m], tau)
        r = np.corrcoef(da, dr)[0, 1]
        n = len(tau); sh = rng.integers(int(0.1 * n), int(0.5 * n) + 1, N_SHIFT)
        null = np.array([np.corrcoef(da, np.roll(dr, s))[0, 1] for s in sh])
        areas.append(dict(session_id=sid, level=a, n_units=int(m.sum()), r_rest=r, r_rest_null=null.mean(), r_rest_ex=r - null.mean(),
                          f_slow=fa, f_trend=fta))
    return out, tr, areas


def main(a):
    meta = pd.read_csv(SRC / "sessions.csv")
    meta = meta[meta.error.isna()] if "error" in meta else meta
    if not a.plot_only:
        curves = {p["session_id"]: (p["curves"] or {}).get("whisker") for p in pickle.load(open(LTP / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]}
        jobs = [(s, curves.get(s) if st == "learning" else None) for s, st in zip(meta.session_id, meta.stage)]
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            res = pool.map(session, jobs)
        keys = ["session_id", "mouse_id", "cohort", "stage"]
        S = C.population(pd.DataFrame([r[0] for r in res if r[0]]).merge(meta[keys], on="session_id"))
        S.to_csv(C.TAB / "007_sessions.csv", index=False)
        pd.concat([r[1] for r in res if r[1] is not None]).merge(meta[keys], on="session_id").to_parquet(C.TAB / "007_drift.parquet")
        A = pd.DataFrame([x for r in res for x in r[2]]).merge(meta[keys], on="session_id"); A = A[A.session_id.isin(S.session_id)]
        A.to_csv(C.TAB / "007_area_sessions.csv", index=False)
        pd.DataFrame([r for m, _ in MEAS for r in C.measure_stats(S, m, "whole brain", m)]).to_csv(C.TAB / "007_stats.csv", index=False)
        AR = []
        for lev, g in A.groupby("level"):
            for m in ("r_rest_ex", "f_slow", "f_trend"):
                for r in C.measure_stats(g, m, lev, m):
                    if r["cohort"] != "R+ vs R-":
                        n = g[g.cohort == r["cohort"]].groupby("stage").size()
                        if min(n.get("learning", 0), n.get("expert", 0)) < 3:
                            continue
                    AR.append(r)
        AR = pd.DataFrame(AR)
        for (m, c), g in AR.groupby(["measure", "cohort"]):
            for p in ("p_d0", "p_e", "p_stage_mwu"):
                AR.loc[g.index, f"q_{p}"] = C.bh(g[p].to_numpy())
        AR.to_csv(C.TAB / "007_area_stats.csv", index=False)
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    S = pd.read_csv(C.TAB / "007_sessions.csv"); R = pd.read_csv(C.TAB / "007_stats.csv"); D = pd.read_parquet(C.TAB / "007_drift.parquet")
    A = pd.read_csv(C.TAB / "007_area_sessions.csv"); AR = pd.read_csv(C.TAB / "007_area_stats.csv")
    # figure 1: one example per group (session with the largest number of trials): drift and smoothed lick probability per trial type
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
    for ax, (coh, stg) in zip(axs, C.GROUPS):
        s = S[(S.cohort == coh) & (S.stage == stg)].sort_values("n_trials").session_id.iloc[-1]
        d = D[D.session_id == s]
        ax.plot(d.tau, d.drift, color="k", lw=0.9, label="slow drift (z)")
        for k, col, lab in (("W", "#d95f02", "whisker"), ("A", "#1f78b4", "auditory"), ("N", "0.5", "no stim")):
            e = d[d.trial_type == k]
            if len(e) > 5:
                p = uniform_filter1d(e.lick.to_numpy(float), 11, mode="nearest")
                ax.plot(e.tau, (p - 0.5) * 4, color=col, lw=0.8, label=f"P(lick) {lab}")
        ax.set_title(f"{C.GLAB[(coh, stg)]}\n{s}", fontsize=5.5); ax.set_xlabel("session time (τ)")
    axs[0].set_ylabel("drift (z); P(lick): (p − 0.5) × 4"); axs[-1].legend(frameon=False, fontsize=4.5)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"007_slow_drift_examples.{ext}", dpi=300)
    plt.close(fig)
    # figure 2: session measures
    FM = [x for x in MEAS if x[0] != "beta_A"]          # auditory licks: too few misses for the logistic test in most sessions (csv only)
    fig, axs = plt.subplots(1, len(FM) + 1, figsize=(1.7 * (len(FM) + 1), 2.8))
    for ax, (m, lab) in zip(axs, FM):
        Rm = R[R.measure == m]
        C.dot_panel(ax, S, m, Rm, lab, rc=Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    Rm = AR[AR.measure == "r_rest_ex"]
    g = A.groupby(["session_id", "mouse_id", "cohort", "stage"]).r_rest_ex.mean().reset_index()
    Rg = pd.DataFrame(C.measure_stats(g, "r_rest_ex", "mean over areas", "r_rest_ex"))
    C.dot_panel(axs[-1], g, "r_rest_ex", Rg, "area drift shared with rest\n(r − shift null, mean of areas)", rc=Rg[Rg.cohort == "R+ vs R-"].iloc[0])
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"007_slow_drift_sessions.{ext}", dpi=300)
    plt.close(fig)
    # figure 3: areas, shared drift
    try:
        from ephys_utilities.allen_utils import get_custom_area_groups_colors
        acol = get_custom_area_groups_colors()
    except Exception:  # noqa: BLE001
        acol = {}
    areas = sorted(AR.level.unique())
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 0.22 * len(areas) + 1.2), sharey=True)
    for ax, m, lab in ((axs[0], "r_rest_ex", "r(area drift, rest-of-brain drift) − shift null"), (axs[1], "f_slow", "slow share of baseline variance")):
        for k, (coh, stg) in enumerate(C.GROUPS):
            g = A[(A.cohort == coh) & (A.stage == stg)].groupby("level")[m].agg(["mean", "sem", "size"]).reindex(areas)
            y = np.arange(len(areas)) + (k - 1.5) * 0.18; ok = g["size"] >= 3
            ax.errorbar(g["mean"][ok], y[ok], xerr=g["sem"][ok], fmt="o", ms=2.5, lw=0.6, color=C.COH[coh],
                        mfc="white" if stg == "learning" else C.COH[coh], label=C.GLAB[(coh, stg)])
        for yy, ar in enumerate(areas):
            for coh, dx in (("R+", 0), ("R-", 1)):
                r = AR[(AR.measure == m) & (AR.level == ar) & (AR.cohort == coh)]
                if len(r) and np.isfinite(r.p_stage_mwu.iloc[0]) and r.p_stage_mwu.iloc[0] < 0.05:
                    ax.text(1.01 + 0.06 * dx, yy, "**" if r.q_p_stage_mwu.iloc[0] < 0.05 else "*", transform=ax.get_yaxis_transform(),
                            color=C.COH[coh], va="center", fontsize=6)
        ax.axvline(0, color="0.6", lw=0.5, ls=":"); ax.set_title(lab, fontsize=6)
        ax.set_yticks(range(len(areas)), areas)
        for t_ in ax.get_yticklabels():
            t_.set_color(acol.get(t_.get_text(), "k"))
    axs[0].legend(frameon=False, fontsize=4.5)
    fig.text(0.5, 0.005, "* day 0 vs expert p < 0.05 (MWU), ** BH q < 0.05 over areas", ha="center", fontsize=5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"007_slow_drift_areas.{ext}", dpi=300)
    plt.close(fig)
    print("figures done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot()
    else:
        main(a)
