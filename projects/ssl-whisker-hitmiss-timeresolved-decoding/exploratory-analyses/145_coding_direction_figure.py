"""145 -- Publication figure for 140 (hit/miss coding direction, 5-35 ms, tracked stable units, learning stage): one row per split
(hit-median, midpoint), mean +- s.e.m. over sessions per cohort:
  a  rotation: reliability-normalised cosine between the coding directions of the two halves (1 = same direction)
  b  gain vs rotation: hit vs miss separation (d') in each half along its own direction and along the other half's direction
  c  passive whisker - auditory difference projected on the active coding direction of each half, as a fraction of the active
     hit - miss difference, passive pre vs passive post
  d  reliability-normalised cosine between the passive whisker axis (whisker - auditory) and the active coding direction
  e  noise along the coding direction: within-class variance along CD / mean variance per unit, half 1 vs half 2
  f  linear Fisher information of hits vs misses (top 10 noise PCs, bias-corrected), half 1 vs half 2
Tests: within cohort paired Wilcoxon | paired t (cohort colour) or vs reference (Wilcoxon | t); between cohorts Mann-Whitney |
Welch (black). Units: sessions. Scopes: all | learners. Uncorrected. Style: skills/ssl-figure-style.
Outputs: figures/publication/145_coding_direction_<scope>.{png,pdf,svg}, 145_stats_<scope>.csv
Run (haas, repo root): python .../145_coding_direction_figure.py [all|learners]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
SPLITS = [("hitmedian", "hit-median split"), ("mid", "midpoint split")]


def pair_panel(ax, d, ca, cb, la, lb, rows, tag, split):
    xp = {"R+": (0, 1), "R-": (2.4, 3.4)}
    a = {c: d[d.reward_group == c][ca].to_numpy(float) for c in COH}
    b = {c: d[d.reward_group == c][cb].to_numpy(float) for c in COH}
    H.mean_pair(ax, xp, a, b)
    for c in COH:
        pw, pt, n = H.paired(b[c], a[c])
        rows.append(dict(split=split, panel=tag, test=f"{cb} vs {ca} (paired Wilcoxon | paired t)", cohort=c, n=n,
                         mean_a=np.nanmean(a[c]), mean_b=np.nanmean(b[c]), p_nonparam=pw, p_param=pt))
        H.bracket(ax, *xp[c], 1.03, f"{H.pnum(pw)}|{H.pnum(pt)}", COL[c])
    chg = {c: b[c] - a[c] for c in COH}
    mw, we = H.unpaired(chg["R+"], chg["R-"])
    rows.append(dict(split=split, panel=tag, test="change R+ vs R- (Mann-Whitney | Welch)", cohort="R+ vs R-", n=np.nan,
                     mean_a=np.nanmean(chg["R+"]), mean_b=np.nanmean(chg["R-"]), p_nonparam=mw, p_param=we))
    H.bracket(ax, 0.5, 2.9, 1.2, f"{H.pnum(mw)}|{H.pnum(we)}", "k")
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels([la, lb, la, lb], fontsize=5); ax.set_xlim(-0.5, 3.9)


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    H.setup()
    D = pd.read_parquet(EA / "140_coding_direction_noise.parquet")
    D = D[D.skipped_reason.isna()]
    if scope == "learners":
        D = D[H.mouse_of(D).isin(H.learners())]
    rows = []
    fig = plt.figure(figsize=(7.4, 4.6))
    W, Hh = fig.get_size_inches()
    lefts = [0.075, 0.235, 0.41, 0.585, 0.75, 0.905]
    widths = [0.07, 0.12, 0.12, 0.12, 0.1, 0.085]
    for ri, (s, sl) in enumerate(SPLITS):
        d = D[D.split == s]
        y0 = 0.56 if ri == 0 else 0.1
        axs = [fig.add_axes([l, y0, w, 0.25]) for l, w in zip(lefts, widths)]
        # a rotation
        ax = axs[0]
        v = {}
        for k, c in enumerate(COH):
            v[c] = d[d.reward_group == c].cosnorm_between.to_numpy(float)
            ax.bar(k, np.nanmean(v[c]), 0.6, color=COL[c], alpha=0.35, lw=0, edgecolor="none")
            ax.errorbar(k, np.nanmean(v[c]), H.sem(v[c]), fmt="o", ms=3.4, color=COL[c], lw=1, capsize=0)
            pw, pt, n = H.one_sample(v[c] - 1)
            rows.append(dict(split=s, panel="rotation", test="cosnorm vs 1 (Wilcoxon | t)", cohort=c, n=n, mean_a=np.nanmean(v[c]),
                             mean_b=np.nan, p_nonparam=pw, p_param=pt))
        mw, we = H.unpaired(v["R+"], v["R-"])
        rows.append(dict(split=s, panel="rotation", test="R+ vs R- (Mann-Whitney | Welch)", cohort="R+ vs R-", n=np.nan,
                         mean_a=np.nanmean(v["R+"]), mean_b=np.nanmean(v["R-"]), p_nonparam=mw, p_param=we))
        H.bracket(ax, 0, 1, 1.03, f"{H.pnum(mw)}|{H.pnum(we)}", "k")
        ax.axhline(1, color="0.5", lw=0.5, ls=(0, (2, 2))); ax.axhline(0, color="0.7", lw=0.4)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"R+\n{len(v['R+'])}", f"R-\n{len(v['R-'])}"], fontsize=5); ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(f"{sl}\ncoding-direction similarity\nhalf 1 vs half 2 (1 = same)")
        # b gain: own axis half 1 -> 2 and cross
        ax = axs[1]
        pair_panel(ax, d, "dprime_own_1", "dprime_own_2", "half 1", "half 2", rows, "gain (own axis)", s)
        for c, xs in (("R+", (0, 1)), ("R-", (2.4, 3.4))):
            g = d[d.reward_group == c]
            m = [g.dprime_cross_1.mean(), g.dprime_cross_2.mean()]
            ax.plot(xs, m, color=COL[c], lw=0.8, ls=(0, (2, 1.5)), alpha=0.8)
            ax.plot(xs, m, "x", color=COL[c], ms=3.2, mew=0.8)
            for h in (1, 2):
                pw, pt, n = H.paired(g[f"dprime_own_{h}"].to_numpy(float), g[f"dprime_cross_{h}"].to_numpy(float))
                rows.append(dict(split=s, panel="own vs other-half axis", test=f"half {h}: own vs cross (paired Wilcoxon | paired t)", cohort=c,
                                 n=n, mean_a=g[f"dprime_cross_{h}"].mean(), mean_b=g[f"dprime_own_{h}"].mean(), p_nonparam=pw, p_param=pt))
        ax.set_ylabel("hit vs miss separation (d')\n(o own axis, x other half's axis)")
        # c passive projection on the half-2 coding direction, pre vs post
        ax = axs[2]
        pair_panel(ax, d, "frac_passive_pre_2", "frac_passive_post_2", "pre", "post", rows, "passive projection on CD half 2", s)
        ax.set_ylabel("passive whisker - auditory\non active coding direction\n(fraction of hit - miss)")
        # d passive axis vs CD (cosnorm), pre vs post, on CD half 2
        ax = axs[3]
        pair_panel(ax, d, "cosnorm_passive_pre_2", "cosnorm_passive_post_2", "pre", "post", rows, "passive axis vs CD half 2", s)
        ax.set_ylabel("passive whisker axis vs\nactive coding direction\n(normalised cosine)")
        # e noise ratio
        ax = axs[4]
        pair_panel(ax, d, "noise_ratio_1", "noise_ratio_2", "h1", "h2", rows, "noise along CD", s)
        ax.axhline(1, color="0.7", lw=0.4)
        ax.set_ylabel("noise along coding direction\n(relative to average unit)")
        # f Fisher information
        ax = axs[5]
        pair_panel(ax, d, "fisher_1", "fisher_2", "h1", "h2", rows, "Fisher information", s)
        ax.set_ylabel("linear Fisher information\n(top 10 PCs, bias-corrected)")
        for ax, l in zip(axs, "abcdef"):
            fig.text(ax.get_position().x0 - 0.36 / W, y0 + 0.25 + 0.42 / Hh, f"{l}{ri + 1}", fontsize=8.5, weight="bold")
    titles = ["Rotation", "Gain vs rotation", "Passive on active axis", "Passive axis alignment", "Noise", "Information"]
    for l, w, t in zip(lefts, widths, titles):
        fig.text(l + w / 2, 0.56 + 0.25 + 0.75 / Hh, t, ha="center", fontsize=6.5)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"145_coding_direction_{scope}.{ext}", dpi=300)
    # supplementary: whisker- and auditory-evoked patterns vs the second-half coding direction, passive pre vs post
    if "cosnorm_evokedW_passive_pre_2" in D:
        fig2, axes = plt.subplots(2, 2, figsize=(4.6, 4.2))
        fig2.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.1, hspace=0.6, wspace=0.45)
        for i, (s, sl) in enumerate(SPLITS):
            d = D[D.split == s]
            for j, (key, kl) in enumerate((("W", "whisker-evoked"), ("A", "auditory-evoked"))):
                ax = axes[i, j]
                pair_panel(ax, d, f"cosnorm_evoked{key}_passive_pre_2", f"cosnorm_evoked{key}_passive_post_2", "pre", "post", rows,
                           f"{kl} pattern vs CD half 2", s)
                ax.set_title(f"{sl}: {kl}", fontsize=6, y=1.35)
                if j == 0:
                    ax.set_ylabel("alignment with the coding\ndirection (normalised cosine)")
        fig2.suptitle("Passive evoked patterns vs the active hit / miss coding direction (2nd half); auditory = control", fontsize=6)
        for ext in ("png", "pdf", "svg"):
            fig2.savefig(FIGDIR / f"145_evoked_alignment_{scope}.{ext}", dpi=300)
    R = pd.DataFrame(rows); R.insert(0, "scope", scope)
    R.to_csv(EA / f"145_stats_{scope}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
