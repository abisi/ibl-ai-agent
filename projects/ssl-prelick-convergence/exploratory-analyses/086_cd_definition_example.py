"""086 -- How the reward-lick coding direction is defined, on real data (user 2026-10-08: "only show the two conditions used for the CD,
centred on the same point, smoothed, no grid, only the explicitly named axes; show before and after orthogonalisation").
Example session AB138 (R+, learning day; stable units, mean pre-lick rate >= 0.1 Hz):
  trajectories: class-mean lick-aligned activity of auditory hits (AH) and spontaneous licks (SL) from -300 ms to the lick (051 PSTHs,
  10-ms bins, Gaussian smoothing sigma = 20 ms), z-scored per unit (mean / SD of the per-trial pre-lick window rates over AH + SL), and
  baseline-corrected per class (minus the class mean over the first 30 ms shown, -300 to -270 ms), so both start at the same point
  plane: x = u, the unit CD = mean(AH) - mean(SL) in the pre-lick window [-100, 0) ms; the named second axis w = the whisker-hit direction
  mean(WH) - mean(SL) in the same window (whisker hits themselves are not drawn)
  a  before orthogonalisation: the plane spanned by u and w, with both axes drawn at their true angle
  b  after orthogonalisation: the same plane with w replaced by its part orthogonal to u
Coordinates per neuron (/ sqrt(N)); thick: the pre-lick window; dots: window means; arrow: the CD (SL mean -> AH mean).
Output: within_day/<ref>/example_state_space/cd_definition_AB138.{svg,pdf,png} (text as text)
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../086_cd_definition_example.py
"""
import importlib
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m85 = importlib.import_module("085_example_state_space")
OUT = m85.OUT
COL = {"AH": "#2c2cdb", "FA": "#262626"}
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})


def arrow(ax, v, lab, col="0.25", L=1.0):
    ax.annotate("", L * v, (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=8))
    ax.text(*(L * v * 1.08), lab, color=col, fontsize=6.5, ha="center", va="center")


def main(mouse="AB138"):
    sid, raw, cls, ts, psth, edges = m85.session_data(mouse)
    m_ = np.isin(cls, ["AH", "FA"])
    mu, sd = raw[:, m_].mean(1), raw[:, m_].std(1); sd[sd == 0] = 1
    Z = (raw - mu[:, None]) / sd[:, None]
    N = Z.shape[0]
    t = (edges[:-1] + edges[1:]) / 2
    P = gaussian_filter1d(psth, 2.0, axis=2)                                  # sigma = 2 bins = 20 ms
    P = (P - mu[:, None, None]) / sd[:, None, None]
    base = (t >= -0.3) & (t < -0.27)                                         # first plotted bins: both start at 0
    P = P - P[:, :, base].mean(2, keepdims=True)                              # each class starts at 0
    win, sel = (t >= -0.1) & (t < 0), (t >= -0.3) & (t < 0)
    ci = {"WH": 0, "AH": 1, "FA": 2}
    wm = {c: P[:, ci[c], win].mean(1) for c in ci}
    u = wm["AH"] - wm["FA"]; u /= np.linalg.norm(u)
    w = wm["WH"] - wm["FA"]; w /= np.linalg.norm(w)
    wo = w - (w @ u) * u; wo /= np.linalg.norm(wo)
    ang = np.degrees(np.arccos(np.clip(u @ w, -1, 1)))
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.8))
    for ax, ttl, ax2, lab2 in ((axs[0], "a  before orthogonalisation", w, f"whisker-hit direction w\n({ang:.0f}° from u)"),
                               (axs[1], "b  after orthogonalisation", wo, "w orthogonal to u")):
        for c in ("FA", "AH"):
            Y = np.c_[P[:, ci[c], sel].T @ u, P[:, ci[c], sel].T @ wo] / np.sqrt(N)
            ws = win[sel]
            ax.plot(*Y[~ws].T, color=COL[c], lw=0.9, alpha=0.55)
            ax.plot(*Y[ws].T, color=COL[c], lw=2.4, solid_capstyle="round")
            mm = np.r_[wm[c] @ u, wm[c] @ wo] / np.sqrt(N)
            ax.plot(*mm, "o", color=COL[c], ms=5, mec="white", mew=0.6, zorder=5)
            ax.text(mm[0], mm[1] + 0.03 * np.ptp(ax.get_ylim()) + 0.02, "AH" if c == "AH" else "SL", color=COL[c], fontsize=7, ha="center")
        a, b = (np.r_[wm[c] @ u, wm[c] @ wo] / np.sqrt(N) for c in ("FA", "AH"))
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color="#d94801", lw=1.6, mutation_scale=9, shrinkA=4, shrinkB=4))
        ax.text(*((a + b) / 2), "CD", color="#d94801", fontsize=7, ha="right", va="bottom")
        lim = max(np.abs(np.r_[ax.get_xlim(), ax.get_ylim()])) * 1.1
        arrow(ax, np.r_[1, 0], "u (CD)", L=lim * 0.85)
        v2 = np.r_[ax2 @ u, ax2 @ wo]
        arrow(ax, v2, lab2, L=lim * 0.85)
        ax.plot(0, 0, "o", color="0.5", ms=3)
        ax.text(0, -lim * 0.08, "start", color="0.5", fontsize=5.5, ha="center", va="top")
        ax.set_xlim(-lim * 0.35, lim); ax.set_ylim(-lim * 0.35, lim); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl, fontsize=7.5, loc="left")
    fig.text(0.5, 0.02, f"{sid} (R+), {N} units; AH and SL mean trajectories, −300 ms to the lick (thick: [−100, 0) ms), baseline-corrected, "
             "per neuron", ha="center", fontsize=5.6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.15)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"cd_definition_{mouse}.{ext}", dpi=300)
    print("ALL DONE", OUT / f"cd_definition_{mouse}.png", f"angle u-w {ang:.1f}")


if __name__ == "__main__":
    main()
