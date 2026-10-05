"""144 -- Publication figure for the hit-median session split (139), with the midpoint split as reference.
Figure 144_hitmedian_split_<scope>:
  row 1  the split: position of the hit-median and midpoint splits (fraction of whisker trials) and the whisker hit rate before /
         after each split, per cohort (mean +- s.e.m. over sessions)
  rows 2-3  separate count-matched decoders (row 2) and a single whole-session decoder (row 3): accuracy minus the within-half
         shift null, first vs second half, per window (hit vs miss baseline -200..-10, 5-35, 5-50, 5-100 ms; whisker vs auditory
         -100..0 ms before the first lick); hit-median split (filled) and midpoint split (open, grey frame); paired Wilcoxon |
         paired t per cohort (cohort colour), R+ vs R- on the change (Mann-Whitney | Welch, black)
  row 4  cross-half generalisation (cross minus within accuracy) per window and split
Units: sessions (learning stage, one per mouse). Scopes: all | learners. Uncorrected. Style: skills/ssl-figure-style.
Outputs: figures/publication/144_hitmedian_split_<scope>.{png,pdf,svg}, 144_stats_<scope>.csv
Run (haas, repo root): python .../144_hitmedian_split_figure.py [all|learners]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA))
import importlib  # noqa: E402

H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
WINS = [("hitmiss", "baseline", "hit/miss\nbaseline"), ("hitmiss", "5-35ms", "hit/miss\n5-35 ms"), ("hitmiss", "5-50ms", "hit/miss\n5-50 ms"),
        ("hitmiss", "5-100ms", "hit/miss\n5-100 ms"), ("modality_lick", "-100-0ms", "whisker vs auditory\n-100-0 ms pre-lick")]
SPLITS = [("hitmedian", "hit-median"), ("mid", "midpoint")]


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    H.setup()
    D = pd.read_parquet(EA / "139_hitmedian_split_whole_brain.parquet")
    D = D[D.skipped_reason.isna()]
    if scope == "learners":
        D = D[H.mouse_of(D).isin(H.learners())]
    rows = []
    fig = plt.figure(figsize=(7.4, 7.6))
    W, Hh = fig.get_size_inches()
    # row 1: split description (hit/miss rows carry it; one per session x split)
    desc = D[(D.decoding == "hitmiss") & (D.window == "5-50ms")]
    ax1 = fig.add_axes([0.08, 0.875, 0.25, 0.085]); ax2 = fig.add_axes([0.42, 0.875, 0.25, 0.085])
    for si, (s, sl) in enumerate(SPLITS):
        for k, c in enumerate(COH):
            d = desc[(desc.split == s) & (desc.reward_group == c)]
            x = si + (k - 0.5) * 0.3
            ax1.errorbar(x, d.split_frac_whisker.mean(), H.sem(d.split_frac_whisker), fmt="o", ms=3.6, color=COL[c], lw=1, capsize=0)
            for j, col in enumerate(("hit_rate_1", "hit_rate_2")):
                ax2.errorbar(si * 2.4 + k + (j - 0.5) * 0.35, d[col].mean(), H.sem(d[col]), fmt="o", ms=3.6, color=COL[c], lw=1, capsize=0,
                             mfc="white" if j == 0 else COL[c], mew=0.9)
            rows.append(dict(row="split", window="", split=s, cohort=c, test="description", n=len(d),
                             mean_a=d.split_frac_whisker.mean(), mean_b=np.nan, p_nonparam=np.nan, p_param=np.nan))
    ax1.set_xticks([0, 1]); ax1.set_xticklabels([sl for _, sl in SPLITS]); ax1.set_xlim(-0.6, 1.6)
    ax1.set_ylabel("split position\n(fraction of whisker trials)"); ax1.axhline(0.5, color="0.5", lw=0.5, ls=(0, (2, 2)))
    ax2.set_xticks([0, 1, 2.4, 3.4]); ax2.set_xlim(-0.6, 4.0); ax2.set_xticklabels(["R+", "R-", "R+", "R-"]); ax2.set_ylabel("whisker hit rate\n(open: before, filled: after)")
    ax2.text(0.5, 1.02, "hit-median", ha="center", transform=ax2.get_xaxis_transform(), fontsize=5.5)
    ax2.text(2.9, 1.02, "midpoint", ha="center", transform=ax2.get_xaxis_transform(), fontsize=5.5)
    fig.text(0.08 - 0.62 / W, 0.96 + 0.05 / Hh, "a", fontsize=9, weight="bold"); fig.text(0.42 - 0.62 / W, 0.96 + 0.05 / Hh, "b", fontsize=9, weight="bold")
    # rows 2-3: separate and single decoders
    lefts = [0.08 + i * 0.185 for i in range(len(WINS))]
    for ri, (pre, lab, y0) in enumerate((("sep", "separate decoders\n(count-matched)", 0.585), ("sgl", "single whole-session\ndecoder", 0.345))):
        for wi, (dec, w, wl) in enumerate(WINS):
            ax = fig.add_axes([lefts[wi], y0, 0.13, 0.15])
            d = D[(D.decoding == dec) & (D.window == w)]
            for si, (s, sl) in enumerate(SPLITS):
                xp = {"R+": (si * 2.6 + 0, si * 2.6 + 0.8), "R-": (si * 2.6 + 1.2, si * 2.6 + 2.0)}
                a, b = {}, {}
                for c in COH:
                    g = d[(d.split == s) & (d.reward_group == c)]
                    a[c], b[c] = g[f"{pre}_corr_1"].to_numpy(), g[f"{pre}_corr_2"].to_numpy()
                    m = [np.nanmean(a[c]), np.nanmean(b[c])]; e = [H.sem(a[c]), H.sem(b[c])]
                    ax.plot(xp[c], m, color=COL[c], lw=1.0, ls="-" if s == "hitmedian" else (0, (2, 1.5)))
                    for x_, mm, ee, filled in zip(xp[c], m, e, (False, True)):
                        ax.errorbar(x_, mm, ee, fmt="o" if s == "hitmedian" else "s", ms=3.2, color=COL[c], mfc=COL[c] if filled else "white",
                                    mew=0.8, lw=0.9, capsize=0)
                    pw, pt, n = H.paired(b[c], a[c])
                    rows.append(dict(row=pre, window=f"{dec} {w}", split=s, cohort=c, test="2nd vs 1st half (paired Wilcoxon | paired t)",
                                     n=n, mean_a=np.nanmean(a[c]), mean_b=np.nanmean(b[c]), p_nonparam=pw, p_param=pt))
                    H.bracket(ax, *xp[c], 1.03 + 0.13 * (c == "R-"), f"{H.pnum(pw)}|{H.pnum(pt)}", COL[c])
                chg = {c: b[c] - a[c] for c in COH}
                mw, we = H.unpaired(chg["R+"], chg["R-"])
                rows.append(dict(row=pre, window=f"{dec} {w}", split=s, cohort="R+ vs R-", test="change R+ vs R- (Mann-Whitney | Welch)",
                                 n=sum(np.isfinite(v).sum() for v in chg.values()), mean_a=np.nanmean(chg["R+"]), mean_b=np.nanmean(chg["R-"]),
                                 p_nonparam=mw, p_param=we))
                H.bracket(ax, si * 2.6 + 0.4, si * 2.6 + 1.6, 1.32, f"{H.pnum(mw)}|{H.pnum(we)}", "k")
            ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
            ax.set_xticks([1.0, 3.6]); ax.set_xticklabels(["hit-median", "midpoint"], fontsize=5); ax.set_xlim(-0.4, 5.0)
            if wi == 0:
                ax.set_ylabel(f"{lab}\naccuracy - within-half null")
            if ri == 0:
                ax.set_title(wl, y=1.52, fontsize=6)
                if wi == 0:
                    fig.text(lefts[0] - 0.42 / W, y0 + 0.15 + 0.5 / Hh, "c", fontsize=9, weight="bold")
            elif wi == 0:
                fig.text(lefts[0] - 0.42 / W, y0 + 0.15 + 0.5 / Hh, "d", fontsize=9, weight="bold")
    # row 4: cross-half generalisation
    for wi, (dec, w, wl) in enumerate(WINS):
        ax = fig.add_axes([lefts[wi], 0.135, 0.13, 0.12])
        d = D[(D.decoding == dec) & (D.window == w)]
        for si, (s, sl) in enumerate(SPLITS):
            v = {}
            for k, c in enumerate(COH):
                v[c] = d[(d.split == s) & (d.reward_group == c)].cross_minus_within.to_numpy()
                ax.errorbar(si + (k - 0.5) * 0.3, np.nanmean(v[c]), H.sem(v[c]), fmt="o" if s == "hitmedian" else "s", ms=3.2, color=COL[c],
                            lw=0.9, capsize=0)
                pw, pt, n = H.one_sample(v[c])
                rows.append(dict(row="cross", window=f"{dec} {w}", split=s, cohort=c, test="cross - within vs 0 (Wilcoxon | t)", n=n,
                                 mean_a=np.nanmean(v[c]), mean_b=np.nan, p_nonparam=pw, p_param=pt))
            mw, we = H.unpaired(v["R+"], v["R-"])
            rows.append(dict(row="cross", window=f"{dec} {w}", split=s, cohort="R+ vs R-", test="Mann-Whitney | Welch", n=np.nan,
                             mean_a=np.nanmean(v["R+"]), mean_b=np.nanmean(v["R-"]), p_nonparam=mw, p_param=we))
            H.bracket(ax, si - 0.15, si + 0.15, 1.03, f"{H.pnum(mw)}|{H.pnum(we)}", "k")
        ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["hit-median", "midpoint"], fontsize=5); ax.set_xlim(-0.5, 1.5)
        if wi == 0:
            ax.set_ylabel("cross-half generalisation\n(cross - within accuracy)")
            fig.text(lefts[0] - 0.42 / W, 0.255 + 0.12 / Hh, "e", fontsize=9, weight="bold")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"144_hitmedian_split_{scope}.{ext}", dpi=300)
    R = pd.DataFrame(rows); R.insert(0, "scope", scope)
    R.to_csv(EA / f"144_stats_{scope}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R[R.test != "description"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
