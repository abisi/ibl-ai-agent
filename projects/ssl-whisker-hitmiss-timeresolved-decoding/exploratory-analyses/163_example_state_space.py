"""163 -- Example sessions in neural state space for the hit / miss axis (user 2026-10-08: "a similar view, finding good axes, with an
example session; for the hit / miss, only show passive data, not the active projected onto it").
Per example session (one R+, one R-, learners, learning day; shared tracked stable units, whole brain; as 135):
  responses: rate in sliding 10-ms windows (5-ms steps) from -20 to +60 ms after stimulus onset, minus the unit's epoch baseline
  (-55 to -20 ms, averaged over the epoch's trials), scaled by the unit's SD of the 5-35 ms response over all trials (135 evoked
  scaling); dead zone around the stimulus artefact excised as in 135
  axes (orthonormal): x = u, the unit hit - miss direction from the active whisker trials, 5-35 ms (z-scored as 135; active trials are
  used only to build u and are not shown); y = the passive-pre whisker pattern (5-35 ms) orthogonal to u (the state-space y axis of
  146 / 151); z = the first principal component of the passive whisker trajectories orthogonal to x and y
  curves: passive whisker (yellow) and auditory (blue) mean trajectories, before (dashed) and after (solid) the task; thick: 5-35 ms;
  dots: window means; diamonds: their projections on u; origin: no evoked response (baseline)
Example choice: among learner sessions with >= 300 tracked units, the R- session with the most negative whisker post - pre projection
change and the R+ session closest to the R+ median change (135 values); printed.
Outputs: figures/publication/163_example_state_space.{svg,pdf,png} (SVG text as text)
Run (haas, repo root): python .../163_example_state_space.py
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EA = Path(__file__).resolve().parent
SCRIPTS = str(EA.parents[2] / "scripts")
sys.path[:0] = [str(EA), SCRIPTS]
H = importlib.import_module("143_lt_split_windows_figures")
M135 = importlib.import_module("135_whisker_lick_alignment_epochs")
COL, FIGDIR = H.COL, H.FIGDIR
WC, AC = "#f7b519", "#2c2cdb"
WIN, BASE, DZ = M135.WIN, M135.BASE, M135.DZ
TS = np.arange(-0.020, 0.0501, 0.005)                       # window starts (10-ms windows)
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})


def pick():
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    A = A[(A.area == "All units") & A.skipped_reason.isna() & (A.n_units >= 300)]
    A = A[H.mouse_of(A).isin(H.learners())].copy()
    A["dW"] = A.evokedW_proj_passive_post - A.evokedW_proj_passive_pre
    rm = A[A.reward_group == "R-"].sort_values("dW").iloc[0]
    rp_ = A[A.reward_group == "R+"]; rp = rp_.iloc[(rp_.dW - rp_.dW.median()).abs().argsort().iloc[0]]
    print("examples:", rp.session_id, round(rp.dW, 4), "|", rm.session_id, round(rm.dW, 4), flush=True)
    return [(rp.session_id, rp.mouse_id, "R+"), (rm.session_id, rm.mouse_id, "R-")]


def session(sid):
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = M132.session_trials(sid, st, tt, T)
    act = trs["active"].reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    keep_act = ~licked | (rt > WIN[1]); act = act[keep_act].reset_index(drop=True); licked = licked[keep_act]
    trs = dict(trs, active=act)
    spikes = T.load_session_unit_spikes(root, sid)
    ids = importlib.import_module("tracked_units").load("stable").get(sid, np.array([], dtype=np.int64))
    units = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), ids)
    X, B, Tm = {}, {}, {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy(); one = np.ones(len(t0), bool)
        r, b = T.sliding_bin_population_matrices(spikes, units, t0, one, [WIN, BASE], dead_zone=DZ)
        X[e], B[e] = r, np.nanmean(b, 0, keepdims=True)
        Tm[e] = np.stack(T.sliding_bin_population_matrices(spikes, units, t0, one, [(s, s + 0.010) for s in TS], dead_zone=DZ), 0)
    P = np.vstack([X[e] - B[e] for e in X])
    mu, sd = np.nanmean(P, 0), np.nanstd(P, 0); ok = (sd > 0) & np.isfinite(sd)
    sd = np.where(ok, sd, 1)
    isw = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in X}
    Za = ((X["active"] - B["active"]) - mu) / sd
    wl, wn = isw["active"] & licked, isw["active"] & ~licked
    u = Za[wl].mean(0) - Za[wn].mean(0); u = np.where(ok, u, 0); u /= np.linalg.norm(u)
    E = {e: (X[e] - B[e]) / sd for e in ("passive_pre", "passive_post")}
    traj = {}
    for e in ("passive_pre", "passive_post"):
        ev = (Tm[e] - B[e][None]) / sd[None]                                   # time x trials x units
        for s_, m in (("W", isw[e]), ("A", ~isw[e])):
            traj[(e, s_)] = np.where(ok, np.nanmean(ev[:, m], 1), 0)
    pW = np.where(ok, np.nanmean(E["passive_pre"][isw["passive_pre"]], 0), 0)
    y = pW - (pW @ u) * u; y /= np.linalg.norm(y)
    M = np.vstack([v for k, v in traj.items() if k[1] == "W"]); M = M[np.isfinite(M).all(1)]; M = M - M.mean(0); M = M - np.outer(M @ u, u) - np.outer(M @ y, y)
    q = np.linalg.svd(M, full_matrices=False)[2][0]; q = q - (q @ u) * u - (q @ y) * y; q /= np.linalg.norm(q)
    Bm = np.stack([u, y, q], 1)
    wins = {k: np.where(ok, np.nanmean(E[k[0]][isw[k[0]] if k[1] == "W" else ~isw[k[0]]], 0), 0) for k in traj}
    n = {e: (int(isw[e].sum()), int((~isw[e]).sum())) for e in E}
    return traj, wins, Bm, int(ok.sum()), n, int(wl.sum()), int(wn.sum())


def panel(ax, sid, coh):
    traj, wins, Bm, nu, n, nh, nm = session(sid)
    tc = TS + 0.005
    inwin = (tc >= WIN[0]) & (tc <= WIN[1])
    for (e, s_), Y in traj.items():
        col = WC if s_ == "W" else AC
        ls = (0, (3, 1.5)) if e == "passive_pre" else "-"
        Z = Y @ Bm / np.sqrt(nu)                                               # per-unit scale (/ sqrt(N)); NaN rows = dead zone
        ax.plot(*Z.T, color=col, lw=0.8 if s_ == "W" else 0.5, alpha=0.25 if s_ == "A" else 0.7, ls=ls)
        ax.plot(*Z[inwin].T, color=col, lw=2.2 if s_ == "W" else 0.9, ls=ls, alpha=1 if s_ == "W" else 0.35)
        m = wins[(e, s_)] @ Bm / np.sqrt(nu)
        ax.scatter(*m, color=col, s=24 if s_ == "W" else 12, depthshade=False, edgecolor="white", linewidth=0.5,
                   alpha=1 if s_ == "W" else 0.6)
        if s_ == "W":
            ax.plot([m[0], m[0]], [m[1], 0], [m[2], 0], color="#b07e00", lw=0.7, ls=(0, (2, 1.5)))
            ax.scatter(m[0], 0, 0, marker="D", s=12, color="#b07e00", depthshade=False)
            ax.text(*(m + np.r_[0, 0, 0.012]), "whisker " + ("pre" if e == "passive_pre" else "post"), fontsize=6, color="#b07e00", ha="center")
    xm = max(abs(a) for a in ax.get_xlim())
    ax.quiver(0, 0, 0, xm * 0.9, 0, 0, color="black", lw=1.4, arrow_length_ratio=0.08)
    ax.text(xm * 0.95, 0, 0, "u (→ hit)", fontsize=7, weight="bold")
    ax.set_xlabel("hit − miss axis u", labelpad=-6); ax.set_ylabel("pre whisker pattern\n(orthogonal to u)", labelpad=-4)
    ax.set_zlabel("PC (remaining)", labelpad=-6)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.fill = False; a_.pane.set_edgecolor("0.85"); a_._axinfo["grid"]["color"] = (0.92, 0.92, 0.92, 1)
    ax.tick_params(labelsize=5, pad=-2); ax.view_init(elev=22, azim=-60)
    ax.set_title(f"{coh.replace('-', '−')} example: {sid}\n{nu} units; passive whisker {n['passive_pre'][0]} / {n['passive_post'][0]} "
                 f"(pre / post); u from {nh} hits, {nm} misses", color=COL[coh], fontsize=7)


def main():
    ex = pick()
    fig = plt.figure(figsize=(7.2, 3.7))
    for i, (sid, _, coh) in enumerate(ex):
        panel(fig.add_subplot(1, 2, i + 1, projection="3d"), sid, coh)
    fig.text(0.5, 0.02, "curves: passive whisker (yellow) and auditory (blue) mean trajectories, −20 to +60 ms after stimulus onset, before (dashed) and "
             "after (solid) the task; thick: 5–35 ms window; dots: window means\nu: unit hit − miss direction from active whisker trials (not shown); "
             "diamonds: projections of the whisker window means on u; units: SD per neuron (÷ √N); origin: no evoked response", ha="center", fontsize=5.6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.11, wspace=0.05)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIGDIR / f"163_example_state_space.{ext}", dpi=300)
    print("ALL DONE", FIGDIR / "163_example_state_space.png")


if __name__ == "__main__":
    main()
