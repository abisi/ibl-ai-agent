"""117b -- Group figure + stats for the FULL count-matched session-halves run (117, user 2026-10-01).
Hit vs miss, whole brain, separate decoder per half, value = mean (accuracy - linear-shift null) over 5-100 ms; terminal
disengagement dropped with rule A1 before the halves are defined.
  a, b  per cohort: first vs second half, unmatched (all trials of each half) and count-matched (each half subsampled
        to the same number of hits and misses, mean of 10 subsamples); paired lines per session;
  c     change (second - first), unmatched vs matched, per session (identity line) -- how much class imbalance drives
        the unmatched change;
  d     change per cohort, matched and unmatched (mean +- SEM).
Stats (uncorrected): within cohort paired Wilcoxon AND paired t (second vs first); R+ vs R- on the change Mann-Whitney
AND Welch; matched vs unmatched change Spearman.
Outputs: figures/117b_halves_matched_full.{pdf,png,svg}, 117b_halves_matched_full_stats.csv
Run: python 117b_halves_matched_full_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
COL = {"R+": "#00B400", "R-": "#C800C8"}
FS_L, FS_M, FS_S = 8, 7, 6


def pf(p):
    return "n.a." if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}")


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS_M})
    S = pd.read_csv(OUT / "117_halves_matched_full_summary.csv")
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(8.27, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9, 0.7]))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.18, wspace=0.5)
    for i, rg in enumerate(("R+", "R-")):
        ax = axes[i]
        g = S[S.reward_group == rg]
        for j, mode in enumerate(("unmatched", "matched")):
            a, b = g[f"w_{mode}_first"].to_numpy(), g[f"w_{mode}_second"].to_numpy()
            xs = np.array([0, 1]) + 2.4 * j
            for ai, bi in zip(a, b):
                ax.plot(xs, [ai, bi], color=COL[rg], alpha=0.2, lw=0.5)
            for xx, v, face in ((xs[0], a, "white"), (xs[1], b, COL[rg])):
                ax.errorbar(xx, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", mfc=face, mec=COL[rg],
                            ecolor=COL[rg], ms=4, elinewidth=0.9, capsize=0, zorder=4)
            pw, pt = stats.wilcoxon(b, a).pvalue, stats.ttest_rel(b, a).pvalue
            ax.text(xs.mean(), 1.02, f"W {pf(pw)}\nt {pf(pt)}", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=FS_S, color=COL[rg])
            rows.append(dict(cohort=rg, mode=mode, n=len(a), first=a.mean(), second=b.mean(), change=(b - a).mean(),
                             change_sem=(b - a).std(ddof=1) / np.sqrt(len(a)), p_wilcoxon=pw, p_paired_t=pt))
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.set_xticks([0, 1, 2.4, 3.4])
        ax.set_xticklabels(["1st", "2nd", "1st", "2nd"], fontsize=FS_S)
        ax.text(0.5, -0.2, "all trials", transform=ax.get_xaxis_transform(), ha="center", fontsize=FS_S)
        ax.text(2.9, -0.2, "count-matched", transform=ax.get_xaxis_transform(), ha="center", fontsize=FS_S)
        ax.set_ylabel("acc. − null, 5-100 ms")
        ax.set_title(f"{'R+' if rg == 'R+' else 'R−'} (n = {len(g)})", color=COL[rg], fontsize=FS_M, pad=22)
        ax.tick_params(labelsize=FS_S)
    ax = axes[2]
    for rg in ("R+", "R-"):
        g = S[S.reward_group == rg]
        ax.scatter(g.change_unmatched, g.change_matched, s=9, color=COL[rg], lw=0, alpha=0.8)
    lim = np.nanmax(np.abs(S[["change_unmatched", "change_matched"]].to_numpy())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.6, ls="--")
    ax.axhline(0, color="0.85", lw=0.5)
    ax.axvline(0, color="0.85", lw=0.5)
    rho, prho = stats.spearmanr(S.change_unmatched, S.change_matched)
    ax.set_title(f"change: all trials vs matched\nSpearman ρ = {rho:.2f}, {pf(prho)}", fontsize=FS_S)
    ax.set_xlabel("2nd − 1st, all trials", fontsize=FS_S)
    ax.set_ylabel("2nd − 1st, count-matched", fontsize=FS_S)
    ax.tick_params(labelsize=FS_S)
    ax = axes[3]
    R = pd.DataFrame(rows)
    for j, mode in enumerate(("unmatched", "matched")):
        ca = S[S.reward_group == "R+"][f"change_{mode}"]
        cb = S[S.reward_group == "R-"][f"change_{mode}"]
        pm, pwl = stats.mannwhitneyu(ca, cb).pvalue, stats.ttest_ind(ca, cb, equal_var=False).pvalue
        R.loc[R["mode"] == mode, "p_cohort_mannwhitney"] = pm
        R.loc[R["mode"] == mode, "p_cohort_welch"] = pwl
        for k, (rg, c) in enumerate((("R+", ca), ("R-", cb))):
            ax.errorbar(j + (k - 0.5) * 0.3, c.mean(), yerr=c.std(ddof=1) / np.sqrt(len(c)), fmt="o", color=COL[rg],
                        ms=4, elinewidth=0.9, capsize=0)
        ax.text(j, 1.02, f"MW {pf(pm)}\nWelch {pf(pwl)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=FS_S - 0.5)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all trials", "matched"], fontsize=FS_S)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("2nd − 1st half", fontsize=FS_S)
    ax.set_title("R+ vs R−", fontsize=FS_S, pad=22)
    ax.tick_params(labelsize=FS_S)
    for ax, L in zip(axes, "abcd"):
        ax.text(-0.3, 1.2, L, transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    fig.suptitle("Hit vs miss decoding, session halves, whole brain (A1-trimmed): all trials vs count-matched halves "
                 "(paired Wilcoxon W and paired t within cohort; uncorrected)", fontsize=FS_M, y=0.99)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"117b_halves_matched_full.{ext}", dpi=300)
    plt.close(fig)
    R.to_csv(OUT / "117b_halves_matched_full_stats.csv", index=False)
    print(R.round(3).to_string(index=False))
    print("Spearman unmatched vs matched change:", round(rho, 3), prho)


if __name__ == "__main__":
    main()
