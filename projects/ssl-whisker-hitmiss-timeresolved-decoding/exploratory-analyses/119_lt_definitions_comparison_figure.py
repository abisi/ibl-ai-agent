"""119 -- One-figure overview of the pre/post learning-trial split decoding across the 12 LT definitions (+ half split),
from 118 (user 2026-09-30: quick overview of the differences between definitions; the LT definition may be refined).
Rows = decoding x window (hit vs miss 5-50 ms, hit vs miss 5-100 ms, whisker vs auditory -100..0 ms from the first
lick). Columns:
  a  R+: per definition, size-matched pre and post (accuracy - shift null; open = pre, filled = post; mean +- SEM
     across sessions), n sessions and paired Wilcoxon p (post vs pre) on the right;
  b  R-: same;
  c  change (post - pre, null-corrected, size-matched) R+ vs R- (mean +- SEM), Mann-Whitney p on the change;
  d  learning-trial position per definition (median and IQR of the LT, whisker-trial index, per cohort; half split =
     median trial) -- where each definition places the split.
Values are uncorrected (13 definitions x 3 windows, exploratory). Definitions keep their own session sets (n shown).
Also writes the same summary restricted to sessions valid under EVERY LT definition that covers the cohort
(figure 119_..._shared), so differences are not only due to different sessions.
Outputs: figures/119_lt_definitions_comparison{,_shared}.pdf/.png/.svg, 119_lt_definitions_stats.csv
Run: python 119_lt_definitions_comparison_figure.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
COL = {"R+": "#00B400", "R-": "#C800C8"}
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate", "half"]
ROWS = [("hitmiss", "5-50ms", "Hit vs miss\n5-50 ms"), ("hitmiss", "5-100ms", "Hit vs miss\n5-100 ms"),
        ("modality_lick", "-100-0ms", "Whisker vs auditory\n-100-0 ms pre-lick")]
FS_L, FS_M, FS_S = 8, 6, 5


def sem(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def summarise(d):
    rows = []
    for dec, w, _ in ROWS:
        for df_ in DEFS:
            g = d[(d.decoding == dec) & (d.window == w) & (d.definition == df_)]
            ch = {}
            for rg in ("R+", "R-"):
                x = g[g.reward_group == rg]
                pre, post = x.corr_pre_matched.to_numpy(), x.corr_post_matched.to_numpy()
                ch[rg] = post - pre
                p = stats.wilcoxon(post, pre).pvalue if len(x) >= 5 else np.nan
                rows.append(dict(decoding=dec, window=w, definition=df_, cohort=rg, n=len(x), pre=np.mean(pre) if len(x) else np.nan,
                                 pre_sem=sem(pre), post=np.mean(post) if len(x) else np.nan, post_sem=sem(post),
                                 change=np.mean(ch[rg]) if len(x) else np.nan, change_sem=sem(ch[rg]), p_wilcoxon=p,
                                 lt_median=x.learning_trial.median(), lt_q1=x.learning_trial.quantile(0.25),
                                 lt_q3=x.learning_trial.quantile(0.75), split_trial_median=x.n_pre.median()))
            p_mw = stats.mannwhitneyu(ch["R+"], ch["R-"]).pvalue if min(len(ch["R+"]), len(ch["R-"])) >= 3 else np.nan
            for r in rows[-2:]:
                r["p_mw_cohorts"] = p_mw
    return pd.DataFrame(rows)


def figure(S, tag, title_extra):
    fig, axes = plt.subplots(3, 4, figsize=(8.27, 8.2), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1, 0.8]))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.93, bottom=0.05, hspace=0.35, wspace=0.55)
    y = np.arange(len(DEFS))[::-1]
    for r, (dec, w, lab) in enumerate(ROWS):
        s = S[(S.decoding == dec) & (S.window == w)].set_index(["definition", "cohort"])
        for c, rg in enumerate(("R+", "R-")):
            ax = axes[r, c]
            for k, df_ in enumerate(DEFS):
                if (df_, rg) not in s.index or s.loc[(df_, rg), "n"] == 0:
                    ax.text(0.0, y[k], "not defined", fontsize=FS_S, color="0.6", va="center")
                    continue
                v = s.loc[(df_, rg)]
                ax.plot([v.pre, v.post], [y[k]] * 2, color="0.75", lw=0.8, zorder=1)
                ax.errorbar(v.pre, y[k], xerr=v.pre_sem, fmt="o", mfc="white", mec=COL[rg], ecolor=COL[rg], ms=3.2,
                            elinewidth=0.6, capsize=0, zorder=2)
                ax.errorbar(v.post, y[k], xerr=v.post_sem, fmt="o", color=COL[rg], ms=3.2, elinewidth=0.6, capsize=0,
                            zorder=3)
                ax.text(1.02, y[k], f"{int(v.n)}   {pf(v.p_wilcoxon)}", transform=ax.get_yaxis_transform(),
                        fontsize=FS_S, va="center", fontweight="bold" if v.p_wilcoxon < 0.05 else "normal")
            ax.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
            ax.axhline(y[-1] + 0.5, color="0.85", lw=0.5)          # separates the half-split reference row
            ax.set_yticks(y)
            ax.set_yticklabels(DEFS if c == 0 else [], fontsize=FS_S)
            ax.set_ylim(-0.7, len(DEFS) - 0.3)
            ax.tick_params(axis="x", labelsize=FS_S)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel("acc. − null (size-matched)", fontsize=FS_M)
            ax.text(1.02, len(DEFS) - 0.25, "n   p (W)", transform=ax.get_yaxis_transform(), fontsize=FS_S, va="bottom")
            ax.set_title(f"{'R+' if rg == 'R+' else 'R−'}: pre (open) → post (filled)", fontsize=FS_M,
                         color=COL[rg], pad=10)
            if c == 0:
                ax.text(-0.95, 0.5, lab, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=FS_M,
                        fontweight="bold")
        ax = axes[r, 2]
        for k, df_ in enumerate(DEFS):
            for rg, dy in (("R+", 0.15), ("R-", -0.15)):
                if (df_, rg) in s.index and s.loc[(df_, rg), "n"] > 0:
                    v = s.loc[(df_, rg)]
                    ax.errorbar(v.change, y[k] + dy, xerr=v.change_sem, fmt="o", color=COL[rg], ms=3, elinewidth=0.6,
                                capsize=0)
            pm = s.loc[(df_, "R+"), "p_mw_cohorts"] if (df_, "R+") in s.index else np.nan
            ax.text(1.02, y[k], pf(pm), transform=ax.get_yaxis_transform(), fontsize=FS_S, va="center",
                    fontweight="bold" if pm < 0.05 else "normal")
        ax.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
        ax.axhline(y[-1] + 0.5, color="0.85", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(-0.7, len(DEFS) - 0.3)
        ax.tick_params(labelsize=FS_S, axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("change, post − pre", fontsize=FS_M)
        ax.text(1.02, len(DEFS) - 0.25, "p (MW)", transform=ax.get_yaxis_transform(), fontsize=FS_S, va="bottom")
        ax.set_title("R+ vs R− on the change", fontsize=FS_M, pad=10)
        ax = axes[r, 3]
        for k, df_ in enumerate(DEFS):
            for rg, dy in (("R+", 0.15), ("R-", -0.15)):
                if (df_, rg) in s.index and s.loc[(df_, rg), "n"] > 0:
                    v = s.loc[(df_, rg)]
                    if df_ == "half":
                        ax.scatter(v.split_trial_median, y[k] + dy, marker="|", s=20, color=COL[rg])
                    else:
                        ax.plot([v.lt_q1, v.lt_q3], [y[k] + dy] * 2, color=COL[rg], lw=0.8)
                        ax.scatter(v.lt_median, y[k] + dy, s=8, color=COL[rg], zorder=3)
        ax.axhline(y[-1] + 0.5, color="0.85", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(-0.7, len(DEFS) - 0.3)
        ax.tick_params(labelsize=FS_S, axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("learning trial (whisker trial;\nhalf: n pre trials)", fontsize=FS_M)
        ax.set_title("split position, median (IQR)", fontsize=FS_M, pad=10)
    for ax, L in zip(axes[0], "abcd"):
        ax.text(-0.12, 1.1, L, transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    fig.suptitle("Pre/post learning-trial split decoding, whole brain, learning day, per learning-trial definition "
                 f"(separate decoder per epoch, size-matched, session-level shift null){title_extra}\n"
                 "Wilcoxon: post vs pre within cohort; MW: change R+ vs R−; uncorrected; bold p < 0.05",
                 fontsize=FS_M, y=0.995)
    FIG.mkdir(exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"119_lt_definitions_comparison{tag}{os.environ.get('SSL_LT_OUT_TAG', '')}.{ext}", dpi=300)
    plt.close(fig)


def main():
    sys.path.insert(0, str(OUT.parents[2] / "scripts"))
    d = pd.read_parquet(OUT / f"118_lt_definitions_split_whole_brain{os.environ.get('SSL_LT_OUT_TAG', '')}.parquet")
    d = d[d.skipped_reason.isna() & d.delta_matched_nullcorr.notna()].copy()   # rows without any valid shift (no null) excluded
    S = summarise(d)
    S.assign(session_set="per definition").to_csv(OUT / f"119_lt_definitions_stats{os.environ.get('SSL_LT_OUT_TAG', '')}.csv", index=False)
    figure(S, "", "")
    # shared sessions: valid under every LT definition defined for that cohort (excluding the R+-only L5w for R-)
    keep = []
    for (dec, w, rg), g in d.groupby(["decoding", "window", "reward_group"]):
        defs = [x for x in DEFS if x != "half" and not (rg == "R-" and x == "L5w lenient (R+)")]
        sets = [set(g[g.definition == x].session_id) for x in defs]
        shared = set.intersection(*sets) if sets else set()
        keep.append(g[g.session_id.isin(shared)])
    Ssh = summarise(pd.concat(keep))
    Ssh.assign(session_set="shared across definitions").to_csv(OUT / f"119_lt_definitions_stats_shared{os.environ.get('SSL_LT_OUT_TAG', '')}.csv", index=False)
    figure(Ssh, "_shared", " -- sessions shared by all definitions")
    pd.set_option("display.width", 220)
    print(S[["decoding", "window", "definition", "cohort", "n", "pre", "post", "change", "p_wilcoxon", "p_mw_cohorts"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
