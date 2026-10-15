"""167 -- How the hit - miss axis is defined, in 3D state space on real data (user 2026-10-08: "only the two conditions used for the CD,
centred on the same point, smoothed, no grid, only the explicitly named axes; before and after orthogonalisation"; "the schematic in 3D
is the thing to centre").
Example session: SSL_167_SID (default: the R- example of 166, read from 166_example_trials.csv). Harmonised scaling (135 SSL_135_HARM=1): rates in 10-ms windows
(5-ms steps) from -20 to +60 ms after stimulus onset, minus the epoch's mean baseline (-55 to -20 ms), scaled by the unit's SD of the
5-35 ms active response; dead zone around the stimulus artefact excised (gap in the curves).
  trajectories: mean of active whisker hits and of active whisker misses (Gaussian smoothing sigma = 1 step), minus the mean of the two
  first-window values (one common offset, so the miss -> hit vector stays along u)
  three named directions: u = unit hit - miss (5-35 ms window means); p = passive-pre whisker pattern; d = passive whisker change
  (post - pre); coordinates on the orthonormal basis (u, p orthogonal to u, d orthogonal to u and p), per neuron (/ sqrt(N))
  a  before orthogonalisation: p and d drawn at their true angles to u (and to each other)
  b  after orthogonalisation: the three orthogonal axes
Thick: 5-35 ms; dots: window means; orange arrow: miss -> hit (the axis); no grid, no panes; text as text (SVG).
Output: figures/publication/167_cd_definition_example.{svg,pdf,png}. Run (haas, repo root): python .../167_cd_definition_example.py
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
from mpl_toolkits.mplot3d import proj3d  # noqa: F401  (3D projection)
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")
EA = Path(__file__).resolve().parent
SCRIPTS = str(EA.parents[2] / "scripts")
sys.path[:0] = [str(EA), SCRIPTS]
M135 = importlib.import_module("135_whisker_lick_alignment_epochs")
H = importlib.import_module("143_lt_split_windows_figures")
WIN, BASE, DZ = M135.WIN, M135.BASE, M135.DZ
TS = np.arange(-0.020, 0.0501, 0.005)
_EX = EA / "166_example_trials.csv"
SID = os.environ.get("SSL_167_SID") or (pd.read_csv(_EX, usecols=["session_id", "cohort"]).query("cohort == 'R-'").session_id.iloc[0]
                                        if _EX.exists() else "MH065_20260114_154021")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})


def unit(v):
    return v / np.linalg.norm(v)


def data(sid):
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
    keep = ~licked | (rt > WIN[1]); act = act[keep].reset_index(drop=True); licked = licked[keep]
    trs = dict(trs, active=act)
    spikes = T.load_session_unit_spikes(root, sid)
    ids = importlib.import_module("tracked_units").load("stable").get(sid, np.array([], dtype=np.int64))
    units = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), ids)
    X, B, Tm = {}, {}, {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy(); one = np.ones(len(t0), bool)
        r, b = T.sliding_bin_population_matrices(spikes, units, t0, one, [WIN, BASE], dead_zone=DZ)
        X[e], B[e] = r, np.nanmean(b, 0, keepdims=True)
        if e == "active":
            Tm[e] = np.stack(T.sliding_bin_population_matrices(spikes, units, t0, one, [(s, s + 0.010) for s in TS], dead_zone=DZ), 0)
    A0 = X["active"] - B["active"]
    sd = A0.std(0); ok = sd > 1e-6
    E = {e: (X[e] - B[e])[:, ok] / sd[ok] for e in X}
    isw = {e: (trs[e].trial_type == "whisker_trial").to_numpy() for e in X}
    hit, miss = isw["active"] & licked, isw["active"] & ~licked
    ev = (Tm["active"] - B["active"][None])[:, :, ok] / sd[ok]                 # time x trials x units
    tr = {k: np.nanmean(ev[:, m], 1) for k, m in (("hit", hit), ("miss", miss))}
    win = {k: E["active"][m].mean(0) for k, m in (("hit", hit), ("miss", miss))}
    pre = E["passive_pre"][isw["passive_pre"]].mean(0); post = E["passive_post"][isw["passive_post"]].mean(0)
    return tr, win, pre, post, int(ok.sum()), int(hit.sum()), int(miss.sum())


def axis3(ax, v, lab, L, col="0.25"):
    ax.quiver(0, 0, 0, *(L * v), color=col, lw=1.0, arrow_length_ratio=0.12)
    ax.text(*(L * v * 1.12), lab, color=col, fontsize=6, ha="left" if abs(v[0]) > 0.9 else "center", va="center")


def main():
    tr, win, pre, post, N, nh, nm = data(SID)
    u = unit(win["hit"] - win["miss"])
    p, d = unit(pre), unit(post - pre)
    e2 = unit(p - (p @ u) * u)
    e3 = d - (d @ u) * u - (d @ e2) * e2; e3 = unit(e3)
    Bs = np.stack([u, e2, e3], 1)                                        # units x 3, orthonormal
    tc = TS + 0.005; inwin = (tc >= WIN[0]) & (tc <= WIN[1])
    col = {"hit": "0.1", "miss": "0.6"}
    ang = {k: np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for k, (a, b) in dict(up=(u, p), ud=(u, d), pd=(p, d)).items()}
    s0 = np.mean([tr[k][np.isfinite(tr[k]).all(1)][0] for k in ("miss", "hit")], 0)   # common start offset
    traj, pts = {}, {}
    for k in ("miss", "hit"):
        Y = (tr[k] - s0) @ Bs / np.sqrt(N)
        good = np.isfinite(Y).all(1); Ys = Y.copy(); Ys[good] = gaussian_filter1d(Y[good], 1.0, axis=0)
        traj[k] = Ys; pts[k] = (win[k] - s0) @ Bs / np.sqrt(N)
    allp = np.r_[np.vstack([t[np.isfinite(t).all(1)] for t in traj.values()]), np.zeros((1, 3))]
    L = np.abs(allp).max() * 1.15
    fig = plt.figure(figsize=(7.0, 3.4))
    panels = (("a  before orthogonalisation",
               [(np.r_[1, 0, 0], "u (hit − miss)"), (p @ Bs, f"p: pre whisker\npattern ({ang['up']:.0f}° from u)"),
                (d @ Bs, f"d: whisker post − pre\n({ang['ud']:.0f}° from u, {ang['pd']:.0f}° from p)")]),
              ("b  after orthogonalisation",
               [(np.r_[1, 0, 0], "u (hit − miss)"), (np.r_[0, 1, 0], "p $\\perp$ u"), (np.r_[0, 0, 1], "d $\\perp$ u, p")]))
    for i, (ttl, axes_) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for k in ("miss", "hit"):
            Ys = traj[k]
            ax.plot(*Ys[~inwin].T, color=col[k], lw=0.9, alpha=0.6)
            ax.plot(*Ys[inwin].T, color=col[k], lw=2.4, solid_capstyle="round")
            ax.scatter(*pts[k], color=col[k], s=22, edgecolor="white", linewidth=0.6, depthshade=False, zorder=5)
            ax.text(*pts[k], f"  {k}", color=col[k], fontsize=7)
        ax.quiver(*pts["miss"], *(pts["hit"] - pts["miss"]), color="#d94801", lw=1.0, linestyle="--", arrow_length_ratio=0.15)
        for j, (v, lab) in enumerate(axes_):
            axis3(ax, v, lab, L * (1.0 if j == 0 else 0.55))
        TRJ = list(traj.values())
        ax.scatter(0, 0, 0, color="0.5", s=6, depthshade=False)
        tips = np.vstack([np.zeros(3)] + [L * (1.0 if j == 0 else 0.55) * v * 1.12 for j, (v, _) in enumerate(axes_)]
                         + [t[np.isfinite(t).all(1)] for t in TRJ])
        lo, hi = tips.min(0), tips.max(0); pad = 0.06 * (hi - lo).max()
        lo, hi = lo - pad, hi + pad                                       # tight box, equal scaling on all three axes
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo), zoom=1.3); ax.view_init(elev=18, azim=-72); ax.set_axis_off()
        ax.set_title(ttl, fontsize=7.5, loc="left")
    fig.text(0.5, 0.03, f"{SID}, {N} units; mean of {nh} active whisker hits and {nm} misses, −20 to +60 ms after the stimulus\n"
             "(thick: 5–35 ms; gap: artefact window), common start offset; dots: 5–35 ms window means; per neuron", ha="center", fontsize=5.6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.07, wspace=0.0)
    for ext in ("svg", "pdf", "png"):
        fig.savefig((Path(__file__).resolve().parent / "figures" / "publication") / f"167_cd_definition_example.{ext}", dpi=300)
    print("ALL DONE", (Path(__file__).resolve().parent / "figures" / "publication") / "167_cd_definition_example.png", {k: round(v, 1) for k, v in ang.items()})


if __name__ == "__main__":
    main()
