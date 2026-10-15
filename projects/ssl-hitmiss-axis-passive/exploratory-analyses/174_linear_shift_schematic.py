"""174 -- Schematic of the linear-shift (shifted-label) control used for both within-day axes (user 2026-10-09: "explain the linear
shifts on the CD for the within-day analysis; make schematics of how it works"). Synthetic data only (illustration), vector output
with text as text.
  a  the problem: labels (hit / miss) change with time in the session, and neural activity drifts slowly with time too; an axis
     built from labels then partly points along the slow drift, and any later epoch (passive post) lands on the "late" side of it
  b  the shift: labels are slid by k trials against the time-ordered activity (non-wrapping: label i is paired with activity i + k,
     trials without a partner are dropped); k = 10-50 % of the trials, both directions (50 shifts); each shift keeps the label
     sequence and the activity drift intact but breaks their trial-by-trial pairing
  c  for every shift the axis is rebuilt exactly as the real one (same scaling, same cross-validation), the same test data are
     projected (passive pre / post for the hit - miss axis; whisker hits for the reward-lick CD), and the same quantity is computed
  d  the reported effect = real value - mean over shifts; what survives is tied to the trial-by-trial label, not to time
Output: figures/publication/174_linear_shift_schematic.{svg,pdf,png}. Run (haas, repo root): python .../174_linear_shift_schematic.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
H = importlib.import_module("143_lt_split_windows_figures")
FIGDIR = (Path(__file__).resolve().parent / "figures" / "publication")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 6})
HIT, MISS, DRIFT = "0.1", "0.65", "#d94801"


def labels(n, rng):
    p = np.linspace(0.8, 0.2, n)                                     # hits early, misses late (e.g. R- learning to withhold)
    return rng.random(n) < p


def panel_a(ax, y, rng):
    n = len(y); t = np.arange(n)
    drift = 0.6 * t / n + 0.08 * np.sin(t / 7)
    act = drift + 0.35 * y + rng.normal(0, 0.08, n)
    ax.vlines(t[y], 1.55, 1.85, color=HIT, lw=0.8); ax.vlines(t[~y], 1.15, 1.45, color=MISS, lw=0.8)
    ax.text(-2, 1.7, "hit", ha="right", va="center"); ax.text(-2, 1.3, "miss", ha="right", va="center")
    ax.plot(t, act, color="0.3", lw=0.8); ax.plot(t, drift + 0.17, color=DRIFT, lw=1.3)
    ax.text(n + 1, drift[-1] + 0.17, "slow drift\n(time)", color=DRIFT, va="center", fontsize=5.5)
    ax.text(n + 1, act[-5:].mean() - 0.3, "neural activity\n(one unit)", color="0.3", va="center", fontsize=5.5)
    ax.set_xlim(-14, n + 22); ax.set_ylim(-0.2, 2.0); ax.axis("off")
    ax.set_title("a  labels and activity both change with time in the session", loc="left", fontsize=6.5)
    ax.annotate("", (n, -0.12), (0, -0.12), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4"))
    ax.text(n / 2, -0.2, "active whisker trials, in time order", ha="center", va="top", fontsize=5.5, color="0.4")


def panel_b(ax, y):
    n = len(y); k = int(0.3 * n)
    rows = (("activity (time order)", None, 0, 2.2), ("real labels", y, 0, 1.45), (f"labels shifted by k = +{k}", y, k, 0.7),
            (f"labels shifted by k = −{k}", y, -k, -0.05))
    for name, lab, kk, yy in rows:
        if lab is None:
            ax.add_patch(plt.Rectangle((0, yy), n, 0.45, color="0.88", lw=0))
            ax.plot(np.arange(n), yy + 0.05 + 0.35 * np.arange(n) / n, color=DRIFT, lw=1)
        else:
            idx = np.arange(n); pos = idx + kk                        # label i drawn under activity i + k
            keep = (pos >= 0) & (pos < n)
            ax.vlines(pos[keep & lab], yy + 0.25, yy + 0.45, color=HIT, lw=0.8)
            ax.vlines(pos[keep & ~lab], yy, yy + 0.2, color=MISS, lw=0.8)
            if kk > 0:
                ax.add_patch(plt.Rectangle((0, yy - 0.03), kk, 0.51, fill=False, ls=":", lw=0.6, ec="0.5"))
                ax.text(kk / 2, yy + 0.22, "no label:\ndropped", ha="center", va="center", fontsize=4.6, color="0.4")
            if kk < 0:
                ax.add_patch(plt.Rectangle((n + kk, yy - 0.03), -kk, 0.51, fill=False, ls=":", lw=0.6, ec="0.5"))
                ax.text(n + kk / 2, yy + 0.22, "no label:\ndropped", ha="center", va="center", fontsize=4.6, color="0.4")
        ax.text(-2, yy + 0.22, name, ha="right", va="center", fontsize=5.5)
    ax.annotate("", (k + 30, 1.27), (30, 1.27), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#2c7fb8"))
    ax.text(30 + k / 2, 1.3, "slide by k", color="#2c7fb8", ha="center", va="bottom", fontsize=5)
    ax.set_xlim(-38, n + 4); ax.set_ylim(-0.3, 2.8); ax.axis("off")
    ax.set_title("b  slide the labels against the activity (non-wrapping)", loc="left", fontsize=6.5)
    ax.text(n / 2, -0.28, "k = 10–50 % of the trials, both directions: 50 shifts; each keeps ≥ 10 hits and ≥ 10 misses "
            "(reward-lick CD: ≥ 8 AH and ≥ 8 SL)", ha="center", va="top", fontsize=4.8, color="0.3")


def panel_c(ax, rng):
    ang = np.radians(22)
    u = np.r_[np.cos(ang), np.sin(ang)]
    for a_ in np.radians(rng.normal(62, 14, 14)):                      # shifted axes keep the drift component: a fan around the
        v = np.r_[np.cos(a_), np.sin(a_)]                              # time direction, partly aligned with the real axis
        ax.add_patch(FancyArrowPatch((0, 0), tuple(1.25 * v), arrowstyle="-|>", mutation_scale=6, color="0.75", lw=0.7))
    ax.add_patch(FancyArrowPatch((0, 0), tuple(1.5 * u), arrowstyle="-|>", mutation_scale=8, color="black", lw=1.4))
    ax.text(*(1.52 * u + np.r_[0.02, -0.02]), "real axis u\n(real labels)", fontsize=5.5, va="top")
    ax.text(-0.35, 1.32, "shifted axes u_k (grey): labels slid,\naxis rebuilt the same way", fontsize=5.2, color="0.45")
    pre, post = np.r_[1.25, 0.95], np.r_[0.95, 1.05]
    for p_, lab, col, dx in ((pre, "passive pre", "0.6", 0.06), (post, "passive post", "#f7b519", -0.06)):
        ax.plot(*p_, "o", ms=5, mfc=col, mec="black", mew=0.6, zorder=5)
        ax.text(p_[0] + dx, p_[1] + 0.05, lab, fontsize=5.2, ha="left" if dx > 0 else "right", va="bottom")
        q = (p_ @ u) * u; ax.plot([p_[0], q[0]], [p_[1], q[1]], color="0.4", lw=0.5, ls=":")
        ax.plot(*q, "o", ms=2.5, color="black")
    ax.text(0.55, -0.12, "same test data on every axis:\npassive pre / post (hit − miss axis)\nor whisker hits (reward-lick CD)",
            fontsize=5, color="0.3", ha="left", va="top")
    ax.set_xlim(-0.4, 2.1); ax.set_ylim(-0.55, 1.55); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("c  rebuild the axis on every shift", loc="left", fontsize=6.5)


def panel_d(ax, rng):
    null = rng.normal(-0.02, 0.012, 50); real = -0.06
    ax.hist(null, bins=12, color="0.75", edgecolor="white", lw=0.4)
    ax.axvline(null.mean(), color="0.4", lw=1, ls=(0, (2.5, 1.5))); ax.axvline(real, color="black", lw=1.5)
    yl = ax.get_ylim()[1] * 1.25; ax.set_ylim(0, yl)
    ax.text(null.mean(), yl * 0.97, "mean over 50 shifts", ha="left", va="top", fontsize=5, color="0.4")
    ax.text(real, yl * 0.97, " real axis", ha="left", va="top", fontsize=5.5)
    ax.annotate("", (real, yl * 0.55), (null.mean(), yl * 0.55), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#d94801"))
    ax.text((real + null.mean()) / 2, yl * 0.6, "reported:\nreal − shifted", ha="center", va="bottom", fontsize=5.5, color="#d94801")
    ax.set_xlabel("quantity per session (e.g. Δ projection post − pre,\nor slope of whisker hits over the session)", fontsize=5.5)
    ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_title("d  remove what time alone produces (one session, illustration)", loc="left", fontsize=6.5)


def main():
    rng = np.random.default_rng(3)
    y = labels(90, rng)
    fig = plt.figure(figsize=(7.1, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.55, wspace=0.25, left=0.08, right=0.98, top=0.93, bottom=0.12)
    panel_a(fig.add_subplot(gs[0, 0]), y, rng)
    panel_b(fig.add_subplot(gs[1, 0]), y)
    panel_c(fig.add_subplot(gs[0, 1]), rng)
    panel_d(fig.add_subplot(gs[1, 1]), rng)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIGDIR / f"174_linear_shift_schematic.{ext}", dpi=300)
    print("ALL DONE", FIGDIR / "174_linear_shift_schematic.png")


if __name__ == "__main__":
    main()
