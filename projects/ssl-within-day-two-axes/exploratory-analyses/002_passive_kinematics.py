"""002 -- Movement state around passive trials before and after the task, day 0 and expert (user 2026-10-15: jaw kinematics control of the
within-session change on the hit - miss axis). The axis window (5-35 ms after the stimulus) precedes any movement, so movement can only
act through the state at stimulus time or through slow tonic changes; both are measured here.
Sessions: those of 175 (passive whisker / auditory trials before and after the active block; learners rule on day 0, all expert sessions).
Passive pre = context passive before the first active trial; passive post = after the last active trial. Per trial (113 loader, DLC
likelihood >= 0.5, 5-ms grid):
  whisk_base   mean |whisker velocity| in [-200, 0) ms (deg/s, raw)          jaw_base    mean jaw opening in [-200, 0) ms (mm, session median 0)
  whisk_ev     mean |whisker velocity| in [0, 100) ms minus whisk_base       jaw_ev      mean jaw opening in [100, 300) ms minus jaw_base
Per session, stimulus and epoch: mean over trials (>= 3 valid); change = post - pre.
Statistics (common.measure_stats on the changes): Wilcoxon vs 0 per cohort and stage, day 0 vs expert, R+ vs R-, cohort x stage.
Output: tables/002_passive_kinematics_sessions.csv, 002_passive_kinematics_stats.csv; figures/002_passive_kinematics.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C

m113 = importlib.import_module("113_dlc_passive_pre")
GRID = np.round(np.arange(-0.2, 0.3 + 1e-9, 0.005), 4)
Q = [("whisk_base", "pre-stimulus whisking (°/s)"), ("jaw_base", "pre-stimulus jaw opening (mm)"),
     ("whisk_ev", "evoked whisking (°/s)"), ("jaw_ev", "evoked jaw opening (mm)")]


def session(args):
    sid, = args
    info, tr, series = m113.load_series(sid)
    out = []
    if tr is None or "context" not in tr:
        return out
    act = tr[tr.context == "active"]
    if act.empty:
        return out
    ep = {"pre": tr[(tr.context == "passive") & (tr.start_time < act.start_time.min())],
          "post": tr[(tr.context == "passive") & (tr.start_time > act.start_time.max())]}
    w = series.get("|whisker velocity|"); j = series.get("jaw opening")
    if j is not None:
        j = (j[0], j[1] - np.nanmedian(j[1]))
    for stim, ttype, col in (("W", "whisker_trial", "whisker_stim_time"), ("A", "auditory_trial", "auditory_stim_time")):
        for e, d in ep.items():
            d = d[d.trial_type == ttype]
            if len(d) < m113.MIN_TR:
                continue
            ev = d[col].to_numpy(float) if col in d else d.start_time.to_numpy(float)
            ev = np.where(np.isfinite(ev), ev, d.start_time.to_numpy(float))
            row = dict(session_id=sid, stim=stim, epoch=e, n_trials=len(d))
            for name, s_, b, a in (("whisk", w, (-0.2, 0.0), (0.0, 0.1)), ("jaw", j, (-0.2, 0.0), (0.1, 0.3))):
                if s_ is None:
                    continue
                G = m113.to_grid(s_[0], s_[1], ev, None, grid=GRID); ok = np.isfinite(G).mean(1) > 0.5
                if ok.sum() < m113.MIN_TR:
                    continue
                G = G[ok]; base = np.nanmean(G[:, (GRID >= b[0]) & (GRID < b[1])], 1)
                evk = np.nanmean(G[:, (GRID >= a[0]) & (GRID < a[1])], 1) - base
                row[f"{name}_base"], row[f"{name}_ev"] = float(np.nanmean(base)), float(np.nanmean(evk))
            out.append(row)
    return out


def compute(n_proc):
    S0 = pd.read_csv(C.HM / "175_across_days_passive_projection_sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, [(s,) for s in S0.session_id])
    L = pd.DataFrame([x for r in res for x in r])
    W = L.pivot_table(index=["session_id", "stim"], columns="epoch", values=[k for k, _ in Q]).reset_index()
    W.columns = [f"{a}_{b}" if b else a for a, b in W.columns]
    for k, _ in Q:
        if f"{k}_post" in W and f"{k}_pre" in W:
            W[f"d_{k}"] = W[f"{k}_post"] - W[f"{k}_pre"]
    W = W.merge(S0, on="session_id")
    W.to_csv(C.TAB / "002_passive_kinematics_sessions.csv", index=False)
    return W


def figure(W, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.8), gridspec_kw=dict(wspace=0.6, hspace=1.35))
    for i, stim in enumerate(("W", "A")):
        for ax, (k, lab), l in zip(axs[i], Q, "abcdefgh"[4 * i:]):
            Rm = R[R.measure == f"{stim} d_{k}"]
            C.dot_panel(ax, W[W.stim == stim], f"d_{k}", Rm, f"{'whisker' if stim == 'W' else 'auditory'} trials: {lab}, post − pre", l,
                        Rm[Rm.cohort == "R+ vs R-"].iloc[0])
    fig.suptitle("Movement state around passive trials, post − pre (green R+, magenta R−; open day 0, filled expert; p: Wilcoxon vs 0, "
                 "Mann–Whitney day 0 vs expert)", fontsize=6.4, x=0.01, y=1.0, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"002_passive_kinematics.{ext}", dpi=300, bbox_inches="tight")


def fp(p):
    return "n/a" if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.2f}".replace("0.", "."))


def main(a):
    W = pd.read_csv(C.TAB / "002_passive_kinematics_sessions.csv") if a.replot else compute(a.n_proc)
    R = pd.DataFrame([r for stim in ("W", "A") for k, _ in Q if f"d_{k}" in W
                      for r in C.measure_stats(W[W.stim == stim], f"d_{k}", "whole brain", f"{stim} d_{k}")])
    R.to_csv(C.TAB / "002_passive_kinematics_stats.csv", index=False)
    figure(W, R)
    pd.set_option("display.width", 250)
    print(R[["measure", "cohort", "n_d0", "n_e", "mean_d0", "mean_e", "p_d0", "p_e", "p_stage_mwu", "p_interaction", "cls"]].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20); ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
