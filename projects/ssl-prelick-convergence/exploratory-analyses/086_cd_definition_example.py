"""086 -- How the reward-lick coding direction is defined, in 3D state space on real data (user 2026-10-08: "only show the two conditions
used for the CD, centred on the same point, smoothed, no grid, only the explicitly named axes; show before and after orthogonalisation";
"the schematic in 3D is the thing to centre").
Example session AB138 (R+, learning day; stable units, mean pre-lick rate >= 0.1 Hz):
  trajectories: class-mean lick-aligned activity of auditory hits (AH) and spontaneous licks (SL) from -300 ms to the lick (051 PSTHs,
  10-ms bins, Gaussian smoothing sigma = 20 ms), z-scored per unit (mean / SD of the per-trial pre-lick window rates over AH + SL), minus
  one common offset (the mean of the two classes' first 30 ms), so both start at the same point and the SL -> AH vector stays along u
  three named directions (per-trial z-scored pre-lick window [-100, 0) ms rates):
    u = unit CD = mean(AH) - mean(SL);  w = whisker-hit direction mean(WH) - mean(SL);
    d = within-day whisker-hit change mean(WH, second half of the session's WH trials) - mean(WH, first half)
  coordinates on the orthonormal basis (u, w orthogonal to u, d orthogonal to u and w), per neuron (/ sqrt(N)); WH not drawn
  a  before orthogonalisation: w and d drawn at their true angles
  b  after orthogonalisation: the three orthogonal axes
Thick: the pre-lick window; dots: window means; orange arrow: SL -> AH (the CD); no grid, no panes; text as text (SVG).
Output: within_day/<ref>/example_state_space/cd_definition_AB138.{svg,pdf,png}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../086_cd_definition_example.py
"""
import importlib
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m85 = importlib.import_module("085_example_state_space")
OUT = m85.OUT
COL = {"AH": "#2c2cdb", "FA": "#262626"}
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})


def unit(v):
    return v / np.linalg.norm(v)


def axis3(ax, v, lab, L, col="0.25"):
    ax.quiver(0, 0, 0, *(L * v), color=col, lw=1.0, arrow_length_ratio=0.12)
    ax.text(*(L * v * 1.12), lab, color=col, fontsize=6, ha="left" if abs(v[0]) > 0.9 else "center", va="center")


def main(mouse="AB138"):
    sid, raw, cls, ts, psth, edges = m85.session_data(mouse)
    m_ = np.isin(cls, ["AH", "FA"])
    mu, sd = raw[:, m_].mean(1), raw[:, m_].std(1); sd[sd == 0] = 1
    Z = (raw - mu[:, None]) / sd[:, None]                                    # units x trials (time-ordered)
    N = Z.shape[0]
    u = unit(Z[:, cls == "AH"].mean(1) - Z[:, cls == "FA"].mean(1))
    w = unit(Z[:, cls == "WH"].mean(1) - Z[:, cls == "FA"].mean(1))
    wh = np.flatnonzero(cls == "WH"); h = len(wh) // 2
    d = unit(Z[:, wh[h:]].mean(1) - Z[:, wh[:h]].mean(1))
    e2 = unit(w - (w @ u) * u); e3 = unit(d - (d @ u) * u - (d @ e2) * e2)
    Bs = np.stack([u, e2, e3], 1)
    ang = {k: np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for k, (a, b) in dict(uw=(u, w), ud=(u, d), wd=(w, d)).items()}
    t = (edges[:-1] + edges[1:]) / 2
    P = gaussian_filter1d(psth, 2.0, axis=2)                                  # sigma = 2 bins = 20 ms
    P = (P - mu[:, None, None]) / sd[:, None, None]
    base, win, sel = (t >= -0.3) & (t < -0.27), (t >= -0.1) & (t < 0), (t >= -0.3) & (t < 0)
    ci = {"AH": 1, "FA": 2}
    s0 = np.mean([P[:, ci[c], base].mean(1) for c in ci], 0)                  # common start offset
    traj = {c: (P[:, ci[c], sel].T - s0) @ Bs / np.sqrt(N) for c in ci}
    pts = {c: (P[:, ci[c], win].mean(1) - s0) @ Bs / np.sqrt(N) for c in ci}
    L = np.abs(np.vstack(list(traj.values()))).max() * 1.15
    ws = win[sel]
    fig = plt.figure(figsize=(7.0, 3.4))
    panels = (("a  before orthogonalisation",
               [(np.r_[1, 0, 0], "u (CD: AH − SL)"), (w @ Bs, f"w: whisker hit − SL\n({ang['uw']:.0f}° from u)"),
                (d @ Bs, f"d: WH late − early\n({ang['ud']:.0f}° from u, {ang['wd']:.0f}° from w)")]),
              ("b  after orthogonalisation",
               [(np.r_[1, 0, 0], "u (CD: AH − SL)"), (np.r_[0, 1, 0], "w $\\perp$ u"), (np.r_[0, 0, 1], "d $\\perp$ u, w")]))
    for i, (ttl, axes_) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for c in ("FA", "AH"):
            Y = traj[c]
            ax.plot(*Y[~ws].T, color=COL[c], lw=0.9, alpha=0.55)
            ax.plot(*Y[ws].T, color=COL[c], lw=2.4, solid_capstyle="round")
            ax.scatter(*pts[c], color=COL[c], s=22, edgecolor="white", linewidth=0.6, depthshade=False)
            ax.text(*pts[c], "  AH" if c == "AH" else "  SL", color=COL[c], fontsize=7)
        ax.quiver(*pts["FA"], *(pts["AH"] - pts["FA"]), color="#d94801", lw=1.0, linestyle="--", arrow_length_ratio=0.15)
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
    fig.text(0.5, 0.03, f"{sid} (R+), {N} units; AH and SL mean trajectories, −300 ms to the lick (thick: [−100, 0) ms), common start "
             "offset; dots: window means; per neuron", ha="center", fontsize=5.6)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.07, wspace=0.0)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"cd_definition_{mouse}.{ext}", dpi=300)
    print("ALL DONE", OUT / f"cd_definition_{mouse}.png", {k: round(v, 1) for k, v in ang.items()})


if __name__ == "__main__":
    main()
