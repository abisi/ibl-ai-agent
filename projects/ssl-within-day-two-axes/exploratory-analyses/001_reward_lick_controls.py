"""001 -- Reaction-time and jaw controls of the within-session whisker-hit drift on the reward-lick CD, day 0 and expert (user
2026-10-15: "show controls for matched RTs, and show jaw kinematics similarity").
Per event (both stages): CD score of ssl-prelick-convergence within_day 002 (`cd`: fixed session axis AH - SL, 5-fold CV, held-out AH / SL
and fold-averaged WH projections, normalised so that held-out SL = 0 and AH = 1), normalised session time tau, reaction time (051 `rt`,
corrected first lick - stimulus), and pre-lick jaw opening (mm, session-median subtracted; mean over [-100, 0) ms before the corrected first
lick; 113 loader, DLC likelihood >= 0.5; > 50 % valid samples). Events are joined on class and trial start (002 `t` = 051 `trial_start`).
Per session, whisker hits (>= MIN_WH events with the quantity):
  b_cd        OLS slope of the WH CD score on tau (the 002 drift)
  b_cd_rt     slope on tau with RT as a covariate (cd ~ tau + rt)
  b_rt        slope of RT on tau (ms per session)
  d_cd, d_cd_rtm   late - early (tau median split) difference of the WH CD score, unmatched and RT-matched (20-ms RT bins, per bin
              min(n_early, n_late) events of each half drawn at random, N_DRAW draws averaged)
  b_jaw       slope of WH jaw opening on tau (mm per session); b_cd_jaw: slope on tau with jaw opening as a covariate
  s_jaw_early / s_jaw_late / d_s_jaw   WH vs AH jaw similarity, signed ROC s = 2 AUC - 1 of per-event jaw opening (0 = same distribution),
              in the early and late halves, and its change
Statistics (common.measure_stats): Wilcoxon vs 0 per cohort and stage, day 0 vs expert (MWU, Welch), R+ vs R- per stage, cohort x stage
(mouse permutation); classification of each within-session change.
Output: tables/001_rl_controls_sessions.csv, 001_rl_controls_stats.csv; figures/001_rl_controls.{png,pdf,svg}
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

m113 = importlib.import_module("113_dlc_passive_pre")
st26 = importlib.import_module("026_roc_rates_all_sessions")
m105 = m113.m105
MIN_WH, N_DRAW, RT_BIN = 6, 50, 0.02
JGRID = np.round(np.arange(-0.1, 0.0, 0.005), 4)
MEAS = [("b_cd", "WH drift on CD (score / session)"), ("b_cd_rt", "WH drift, RT covariate"), ("d_cd", "late − early (WH CD score)"),
        ("d_cd_rtm", "late − early, RT-matched"), ("b_rt", "RT drift (s / session)"), ("b_jaw", "WH jaw drift (mm / session)"),
        ("b_cd_jaw", "WH drift, jaw covariate"), ("d_s_jaw", "Δ WH vs AH jaw similarity (late − early)"),
        ("s_jaw_early", "WH vs AH jaw s, early"), ("s_jaw_late", "WH vs AH jaw s, late")]


def ols_slope(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X)); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1]


def roc_s(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1


def session(args):
    sid, E, f051 = args
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    z = np.load(f051, allow_pickle=True)
    ev = pd.DataFrame(dict(cls=z["cls"].astype(str), t=z["trial_start"].astype(float), rt=z["rt"].astype(float)))
    E = E.copy(); E["t_r"] = E.t.round(3); ev["t_r"] = ev.t.round(3)
    E = E.merge(ev[["cls", "t_r", "rt"]], on=["cls", "t_r"], how="left")
    # jaw opening in [-100, 0) ms before the corrected first lick (WH / AH: trial start + rt)
    E["jaw"] = np.nan
    try:
        _, _, series = m113.load_series(sid)
    except Exception:  # noqa: BLE001
        series = {}
    if "jaw opening" in series:
        tj, vj = series["jaw opening"]; vj = vj - np.nanmedian(vj)
        cue = E.cls.isin(["WH", "AH"]) & E.rt.notna()
        if cue.any():
            G = m113.to_grid(tj, vj, (E.t + E.rt)[cue].to_numpy(float), None, grid=JGRID)
            ok = np.isfinite(G).mean(1) > 0.5
            E.loc[cue, "jaw"] = np.where(ok, np.nanmean(G, 1), np.nan)
    W, A = E[E.cls == "WH"], E[E.cls == "AH"]
    out = dict(session_id=sid, n_WH=len(W), n_AH=len(A))
    w = W.dropna(subset=["cd", "rt"])
    if len(w) >= MIN_WH:
        out["b_cd"] = ols_slope(w.cd.to_numpy(), [w.tau.to_numpy()])
        out["b_cd_rt"] = ols_slope(w.cd.to_numpy(), [w.tau.to_numpy(), w.rt.to_numpy()])
        out["b_rt"] = ols_slope(w.rt.to_numpy(), [w.tau.to_numpy()])
        med = w.tau.median(); e_, l_ = w[w.tau <= med], w[w.tau > med]
        out["d_cd"] = l_.cd.mean() - e_.cd.mean()
        be, bl = np.floor(e_.rt / RT_BIN), np.floor(l_.rt / RT_BIN)
        common = sorted(set(be) & set(bl))
        if common:
            ds = []
            for _ in range(N_DRAW):
                ie, il = [], []
                for b in common:
                    xe, xl = np.flatnonzero(be.to_numpy() == b), np.flatnonzero(bl.to_numpy() == b); k = min(len(xe), len(xl))
                    ie += list(rng.choice(xe, k, replace=False)); il += list(rng.choice(xl, k, replace=False))
                if len(ie) >= 3:
                    ds.append(l_.cd.to_numpy()[il].mean() - e_.cd.to_numpy()[ie].mean())
            out["d_cd_rtm"] = np.mean(ds) if ds else np.nan
            out["n_rtm"] = len(ie) if ds else 0
    wj = W.dropna(subset=["cd", "jaw"])
    if len(wj) >= MIN_WH:
        out["b_jaw"] = ols_slope(wj.jaw.to_numpy(), [wj.tau.to_numpy()])
        out["b_cd_jaw"] = ols_slope(wj.cd.to_numpy(), [wj.tau.to_numpy(), wj.jaw.to_numpy()])
        aj = A.dropna(subset=["jaw"]); med = E[E.cls.isin(["WH", "AH"])].tau.median()
        se = roc_s(wj[wj.tau <= med].jaw, aj[aj.tau <= med].jaw); sl = roc_s(wj[wj.tau > med].jaw, aj[aj.tau > med].jaw)
        out.update(s_jaw_early=se, s_jaw_late=sl, d_s_jaw=sl - se)
    return out


def compute(n_proc):
    T = pd.read_parquet(C.m51.WITHIN / "slopes" / "trial_scores.parquet")
    ss = st26.all_sessions().set_index("session_id")
    jobs = []
    for sid, E in T.groupby("session_id"):
        if sid not in ss.index:
            continue
        r = ss.loc[sid]
        f = r.file.parent / f"{r.mouse}_roc_prelick{C.m51.TAG}_trials.npz"
        if f.exists():
            jobs.append((sid, E.assign(cls=E.cls.replace({"FA": "SL"}))[["cls", "t", "tau", "cd"]], f))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, jobs)
    meta = T.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    S = C.population(pd.DataFrame(res).merge(meta, on="session_id"))
    S.to_csv(C.TAB / "001_rl_controls_sessions.csv", index=False)
    return S


def figure(S, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    show = ["b_cd", "b_cd_rt", "d_cd", "d_cd_rtm", "b_cd_jaw", "b_rt", "b_jaw", "d_s_jaw"]
    fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.6), gridspec_kw=dict(wspace=0.6, hspace=1.25))
    lab = dict(MEAS)
    for ax, m, l in zip(axs.ravel(), show, "abcdefgh"):
        C.dot_panel(ax, S, m, R[R.measure == m], lab[m], l)
    fig.suptitle("Reward-lick CD, within-session whisker-hit drift: reaction-time and jaw controls (green R+, magenta R−; open day 0, "
                 "filled expert; p: Wilcoxon vs 0, Mann–Whitney day 0 vs expert)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"001_rl_controls.{ext}", dpi=300, bbox_inches="tight")


def fp(p):
    return "n/a" if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.2f}".replace("0.", "."))


def main(a):
    S = pd.read_csv(C.TAB / "001_rl_controls_sessions.csv") if a.replot else compute(a.n_proc)
    R = pd.DataFrame([r for m, l in MEAS if m in S for r in C.measure_stats(S, m, "whole brain", m)])
    R.to_csv(C.TAB / "001_rl_controls_stats.csv", index=False)
    figure(S, R)
    pd.set_option("display.width", 250)
    print(R[["measure", "cohort", "n_d0", "n_e", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20); ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
