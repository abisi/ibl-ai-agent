"""167 -- How the hit - miss axis is defined, on real data (user 2026-10-08: "only the two conditions used for the CD, centred on the same
point, smoothed, no grid, only the explicitly named axes; before and after orthogonalisation; also y = passive post - pre (whisker)").
Example session: the R- example of 163 / 166 (MH065, 530 tracked units). Harmonised scaling (135 SSL_135_HARM=1): rates in 10-ms windows
(5-ms steps) from -20 to +60 ms after stimulus onset, minus the epoch's mean baseline (-55 to -20 ms), scaled by the unit's SD of the
5-35 ms active response; dead zone around the stimulus artefact excised (gap in the curves).
  trajectories: mean of active whisker hits and of active whisker misses (Gaussian smoothing sigma = 1 step), minus the mean of the two
  first-window values (one common offset: the analysis subtracts one common epoch baseline, so hits and misses are not re-centred
  separately; separate centring would tilt the miss -> hit vector away from u)
  x = u, the unit hit - miss direction (5-35 ms window means)
  a  before orthogonalisation: second axis = the passive-pre whisker pattern (5-35 ms), drawn at its true angle to u
  b  after orthogonalisation: that pattern orthogonal to u
  c  alternative second axis: the passive whisker change pattern (post - pre, 5-35 ms) orthogonal to u
Coordinates per neuron (/ sqrt(N)); thick: 5-35 ms; dots: window means; arrow: miss -> hit (the axis).
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
SID = "MH065_20260114_154021"
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


def main():
    tr, win, pre, post, N, nh, nm = data(SID)
    u = unit(win["hit"] - win["miss"])
    y_pre = unit(pre); y_pre_o = unit(pre - (pre @ u) * u)
    d = post - pre; y_d = unit(d - (d @ u) * u)
    tc = TS + 0.005; inwin = (tc >= WIN[0]) & (tc <= WIN[1])
    col = {"hit": "0.1", "miss": "0.6"}
    ang = np.degrees(np.arccos(np.clip(u @ y_pre, -1, 1)))
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.9))
    panels = ((axs[0], "a  before orthogonalisation", y_pre_o, y_pre, f"passive-pre whisker pattern\n({ang:.0f}° from u)"),
              (axs[1], "b  after orthogonalisation", y_pre_o, y_pre_o, "pre whisker pattern\northogonal to u"),
              (axs[2], "c  alternative second axis", y_d, y_d, "whisker post − pre change\northogonal to u"))
    for ax, ttl, basis_y, ax2, lab2 in panels:
        pts = []
        s0 = np.mean([tr[k][np.isfinite(tr[k]).all(1)][0] for k in ("miss", "hit")], 0)   # common start offset
        for k in ("miss", "hit"):
            Y = tr[k] - s0
            Y = np.c_[Y @ u, Y @ basis_y] / np.sqrt(N)
            good = np.isfinite(Y).all(1)
            Ys = Y.copy(); Ys[good] = gaussian_filter1d(Y[good], 1.0, axis=0)
            ax.plot(*Ys[~inwin].T, color=col[k], lw=0.9, alpha=0.6)
            ax.plot(*Ys[inwin].T, color=col[k], lw=2.4, solid_capstyle="round")
            m = np.r_[(win[k] - s0) @ u, (win[k] - s0) @ basis_y] / np.sqrt(N); pts.append(m)     # 5-35 ms window means
            ax.plot(*m, "o", color=col[k], ms=5, mec="white", mew=0.6, zorder=5)
            ax.text(m[0], m[1], f"  {k}", color=col[k], fontsize=7, va="bottom")
        ax.annotate("", pts[1], pts[0], arrowprops=dict(arrowstyle="-|>", color="#d94801", lw=1.6, mutation_scale=9, shrinkA=4, shrinkB=4))
        lim = max(np.abs(np.r_[ax.get_xlim(), ax.get_ylim()])) * 1.1
        for v, lab in ((np.r_[1, 0], "u (hit − miss)"), (np.r_[ax2 @ u, ax2 @ basis_y], lab2)):
            ax.annotate("", lim * 0.85 * v, (0, 0), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.1, mutation_scale=8))
            ax.text(*(lim * 0.85 * v * 1.1), lab, color="0.25", fontsize=6, ha="center", va="center")
        ax.plot(0, 0, "o", color="0.5", ms=3)
        ax.set_xlim(-lim * 0.45, lim * 1.05); ax.set_ylim(-lim * 0.45, lim * 1.05); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl, fontsize=7.5, loc="left")
    fig.text(0.5, 0.02, f"{SID} (R−), {N} units; mean of {nh} active whisker hits and {nm} misses, −20 to +60 ms after the stimulus "
             "(thick: 5–35 ms; gap: artefact window), common start offset; dots: 5–35 ms window means; per neuron", ha="center", fontsize=5.6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.12)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(H.FIGDIR / f"167_cd_definition_example.{ext}", dpi=300)
    print("ALL DONE", H.FIGDIR / "167_cd_definition_example.png", f"angle {ang:.1f}")


if __name__ == "__main__":
    main()
