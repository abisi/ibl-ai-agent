"""134c -- Illustration of analysis 134 (user 2026-10-02: "illustrate this"): schematics of the design and the two measures, plus
the real whole-brain results.
  a  session structure: passive_pre -> active -> passive_post; the same tracked good units in all three epochs;
  b  trial timeline: baseline -55..-20 ms (epoch-specific), dead zone -10..+5 ms, sliding 30 ms windows 5-200 ms, active trials
     with a lick before the window end excluded;
  c  measure 1 (cartoon): a unit's preferred direction from half of passive_pre, its change on the other half vs a later epoch;
     whisker-specific change = whisker change - auditory change;
  d  measure 3 (cartoon, 2-D population space): whisker axis (whisker - auditory) and lick axis (licked - unlicked whisker
     trials); R+: aligned (cos > 0); R-: anti-aligned (cos < 0);
  e  result, measure 3: whisker axis . lick axis over time, per cohort (mean +- SEM; bars: R+ vs R- MW AND Welch p < .05);
  f  result, measure 1: whisker-specific gain change post vs pre over time, per cohort.
Input: 134_stats.csv. Output: figures/134c_illustration.{pdf,png,svg}
Run: python 134c_illustration.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
COL = {"R+": "#00B400", "R-": "#C800C8"}
WC, AC = "#d95f02", "#1f5fbf"            # whisker, auditory
FS = 7


def arrow(ax, p0, p1, color, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls))


def panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for x0, w, lab, c in ((0.2, 2.3, "passive pre", "0.85"), (2.7, 4.4, "active (task)", "#fde0c5"), (7.3, 2.5, "passive post", "0.85")):
        ax.add_patch(FancyBboxPatch((x0, 2.2), w, 0.9, boxstyle="round,pad=0.02", fc=c, ec="0.4", lw=0.6))
        ax.text(x0 + w / 2, 2.65, lab, ha="center", va="center", fontsize=FS)
    rng = np.random.default_rng(1)
    for k, x in enumerate(np.sort(rng.uniform(0.3, 9.7, 46))):
        ax.plot([x, x], [1.75, 2.05], color=WC if k % 2 else AC, lw=1)
    ax.text(0.2, 1.4, "whisker (orange) and auditory (blue) trials in every epoch;\nactive: perf ≠ 6, warm-up removed, A1-trimmed",
            fontsize=FS - 1, va="top")
    for i, y in enumerate((3.35, 3.55, 3.75)):
        ax.plot([0.2, 9.8], [y, y], color="0.3", lw=0.5)
        for x in rng.uniform(0.2, 9.8, 25):
            ax.plot([x, x], [y - 0.06, y + 0.06], color="0.3", lw=0.5)
    ax.text(5, 3.95, "same good units, firing ≥ 0.5 Hz in all three epochs", ha="center", fontsize=FS - 1)


def panel_b(ax):
    ax.set_xlim(-80, 215)
    ax.set_ylim(0, 4)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.axvline(0, color="k", lw=1)
    ax.text(0, 3.75, "stimulus", ha="center", fontsize=FS - 1)
    ax.add_patch(Rectangle((-55, 2.6), 35, 0.6, color="0.6"))
    ax.text(-37.5, 3.3, "baseline\n(per epoch)", ha="center", fontsize=FS - 1.5, va="bottom")
    ax.add_patch(Rectangle((-10, 2.6), 15, 0.6, color="#d62728", alpha=0.4))
    ax.text(-2.5, 2.35, "dead\nzone", ha="center", fontsize=FS - 2, va="top")
    for k, s in enumerate(range(5, 171, 15)):
        ax.add_patch(Rectangle((s, 1.55 - 0.1 * (k % 3)), 30, 0.08, color="#1f77b4", alpha=0.7))
    ax.text(100, 1.9, "sliding 30-ms windows, 5 → 200 ms", ha="center", fontsize=FS - 1)
    ax.plot([140, 140], [0.4, 1.2], color="k", lw=1.2)
    ax.text(143, 0.75, "lick (corrected)", fontsize=FS - 1.5, va="center")
    ax.add_patch(Rectangle((125, 0.55), 30, 0.12, color="0.75"))
    ax.text(60, 0.35, "active trials with a lick before\nthe window end: excluded there", fontsize=FS - 1.5, ha="center")
    ax.set_xlabel("time from stimulus (ms)", fontsize=FS)
    ax.tick_params(labelsize=FS - 1)


def panel_c(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.1, 5.7, "one unit, whisker trials (z-scored response)", fontsize=FS - 0.5)
    ax.bar([1.2, 3.2], [1.0, 0.95], width=0.9, color=["0.75", "0.55"])
    ax.text(1.2, -0.05 + 0.2, "pre\nhalf 1", ha="center", va="top", fontsize=FS - 1.5, transform=ax.transData)
    ax.text(1.2, 1.15 + 0.1, "sign +", ha="center", fontsize=FS - 1.5)
    ax.text(3.2, 1.25, "pre half 2\n(reference)", ha="center", fontsize=FS - 1.5)
    ax.bar([5.2], [1.6], width=0.9, color=WC)
    ax.text(5.2, 1.75, "active or\npost", ha="center", fontsize=FS - 1.5)
    arrow(ax, (3.7, 1.05), (4.7, 1.55), "k", lw=1)
    ax.text(4.25, 2.6, "gain change g = sign × (later − reference)", ha="center", fontsize=FS - 1)
    ax.text(0.1, 4.6, "averaged over units and 10 random halvings;\npreferred sign and reference from DIFFERENT trials",
            fontsize=FS - 1.5)
    ax.add_patch(FancyBboxPatch((6.6, 0.3), 3.3, 3.6, boxstyle="round,pad=0.05", fc="#f7f7f7", ec="0.6", lw=0.6))
    ax.text(8.25, 3.4, "whisker-specific\nchange", ha="center", fontsize=FS, fontweight="bold", va="top")
    ax.text(8.25, 2.1, r"$g_{whisker} - g_{auditory}$", ha="center", fontsize=FS + 1, color="k")
    ax.text(8.25, 1.1, "auditory = control\n(identical in R+ and R−)", ha="center", fontsize=FS - 1.5)


def panel_d(ax):
    rng = np.random.default_rng(3)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.0, 3.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("population dimension 1", fontsize=FS - 1)
    ax.set_ylabel("population dimension 2", fontsize=FS - 1)
    A = rng.normal([-1.6, -0.3], 0.35, (25, 2))
    Wl = rng.normal([1.5, 1.2], 0.35, (25, 2))
    Wn = rng.normal([1.4, -0.6], 0.35, (25, 2))
    ax.scatter(*A.T, s=8, color=AC, alpha=0.6, lw=0, label="auditory")
    ax.scatter(*Wl.T, s=8, color=WC, alpha=0.9, lw=0, label="whisker, licked")
    ax.scatter(*Wn.T, s=8, facecolor="none", edgecolor=WC, lw=0.6, label="whisker, no lick")
    W, Am = np.r_[Wl, Wn].mean(0), A.mean(0)
    arrow(ax, tuple(Am), tuple(W), "k", lw=1.8)
    ax.text(*(Am + W) / 2 + np.array([-0.5, 0.35]), "whisker axis", fontsize=FS - 1)
    arrow(ax, tuple(Wn.mean(0)), tuple(Wl.mean(0)), "#777777", lw=1.8, ls="--")
    ax.text(Wl.mean(0)[0] + 0.15, 0.3, "lick axis", fontsize=FS - 1, color="#555555")
    ax.legend(fontsize=FS - 2, frameon=False, loc="lower left")
    ax.text(0, 3.0, "alignment = cos(whisker axis, lick axis)\n(split halves, reliability-normalised)", ha="center",
            fontsize=FS - 1, va="top")
    # inset: R+ vs R- prediction
    ins = ax.inset_axes([0.62, 0.02, 0.37, 0.36])
    ins.set_xlim(-1.3, 1.3)
    ins.set_ylim(-0.4, 1.3)
    ins.axis("off")
    arrow(ins, (0, 0), (1.0, 0.0), "k", lw=1.2)
    arrow(ins, (0, 0), (0.85, 0.5), COL["R+"], lw=1.2)
    arrow(ins, (0, 0), (-0.85, 0.5), COL["R-"], lw=1.2)
    ins.text(0.95, 0.62, "R+: lick axis\nalong whisker", fontsize=FS - 2.5, color=COL["R+"], ha="center")
    ins.text(-0.95, 0.62, "R−: opposite", fontsize=FS - 2.5, color=COL["R-"], ha="center")


def panel_result(ax, S, measure, ylabel, title):
    s = S[(S.area == "All units") & (S.measure == measure)].sort_values("win_start")
    x = s.win_start + 15
    for rg, lab in (("R+", "Rplus"), ("R-", "Rminus")):
        ax.fill_between(x, s[f"mean_{lab}"] - s[f"sem_{lab}"], s[f"mean_{lab}"] + s[f"sem_{lab}"], color=COL[rg], alpha=0.2, lw=0)
        ax.plot(x, s[f"mean_{lab}"], color=COL[rg], lw=1.6, marker="o", ms=2.5,
                label=f"{'R+' if rg == 'R+' else 'R−'} (n={int(s[f'n_{lab}'].max())})")
    both = (s.p_mw < 0.05) & (s.p_welch < 0.05)
    one = ((s.p_mw < 0.05) | (s.p_welch < 0.05)) & ~both
    for msk, c in ((both, "k"), (one, "0.65")):
        for xx in x[msk]:
            ax.plot([xx - 7.5, xx + 7.5], [1.03, 1.03], color=c, lw=3, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("window centre (ms after stimulus)", fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_title(title, fontsize=FS, pad=10)
    ax.legend(fontsize=FS - 1, frameon=False)
    ax.tick_params(labelsize=FS - 1)


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    S = pd.read_csv(OUT / "134_stats.csv")
    fig = plt.figure(figsize=(8.27, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.05, 1.0], hspace=0.55, wspace=0.3, left=0.07, right=0.98, top=0.95,
                          bottom=0.06)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    panel_d(axes[3])
    panel_result(axes[4], S, "lick_cosnorm", "whisker axis · lick axis (norm. cos)",
                 "measure 3: whisker axis vs lick axis, active epoch")
    panel_result(axes[5], S, "spec_plast", "whisker − auditory gain change (z)",
                 "measure 1: whisker-specific change, passive post vs pre")
    titles = ["session & units", "trial timeline", "measure 1: whisker-specific gain change",
              "measure 3: does the whisker code point toward licking?", "", ""]
    for ax, L, t in zip(axes, "abcdef", titles):
        ax.text(-0.06, 1.06, L, transform=ax.transAxes, fontweight="bold", fontsize=FS + 2)
        if t:
            ax.text(0.04, 1.06, t, transform=ax.transAxes, fontsize=FS, fontweight="bold")
    fig.text(0.07, 0.015, "Whole brain, tracked good units, rates with epoch-specific baseline, z-scored; bars above e–f: R+ vs R− "
             "(black: Mann-Whitney AND Welch p < .05; grey: one test); uncorrected.", fontsize=FS - 1)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"134c_illustration.{ext}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
