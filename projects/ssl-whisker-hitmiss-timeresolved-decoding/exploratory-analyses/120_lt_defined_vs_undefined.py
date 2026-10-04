"""120 -- Per learning-trial (LT) definition: sessions WITH a definable LT (pre/post split at the LT) vs sessions WITHOUT
one (their session-half split), from 118 (user 2026-09-30: "compare non-behaving/stationary mice session halves with
gradually learning mice").
For each of the 12 LT definitions, each cohort and each decoding window (hit vs miss 5-50, 5-100 ms; whisker vs
auditory -100..0 ms from the first lick), sessions are grouped by the LT table
(ssl-learning-trial-identification/artifacts/013_learning_trials_all_methods.csv):
  learners      LT defined under the definition  -> change at the LT split (post - pre)
  learners, half  same sessions, change at the half split (control: is it the LT timing or the session type?)
  no LT         LT not defined (NaN)             -> change at the half split
Change = size-matched, shift-null-corrected accuracy, post - pre (delta_matched_nullcorr). Sessions with an LT that
fails the per-epoch class-count minimum are left out of "learners" (their half split is still in "learners, half" only
if valid). Tests: learners (LT split) vs no-LT (half), Mann-Whitney; learners' LT split vs their own half split, paired
Wilcoxon. Uncorrected (exploratory overview).
Outputs: figures/120_lt_defined_vs_undefined.pdf/.png/.svg, 120_lt_defined_vs_undefined_stats.csv
Run: python 120_lt_defined_vs_undefined.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
LT_TABLE = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "013_learning_trials_all_methods.csv"
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate"]
ROWS = [("hitmiss", "5-50ms", "Hit vs miss 5-50 ms"), ("hitmiss", "5-100ms", "Hit vs miss 5-100 ms"),
        ("modality_lick", "-100-0ms", "Whisker vs auditory, -100-0 ms pre-lick")]
COL = {"R+": "#00B400", "R-": "#C800C8"}
GROUP = {"learners": dict(f=1.0, m="o", lab="LT defined: LT split"),
         "learners_half": dict(f=0.45, m="o", lab="LT defined: half split"),
         "no_lt": dict(f=1.0, m="s", lab="no LT: half split")}
FS_L, FS_M, FS_S = 8, 6, 5


def shade(c, f):
    from matplotlib.colors import to_rgb
    r = np.array(to_rgb(c))
    return tuple(1 - f * (1 - r))


def sem(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def main():
    d = pd.read_parquet(OUT / f"118_lt_definitions_split_whole_brain{os.environ.get('SSL_LT_OUT_TAG', '')}.parquet")
    d = d[d.skipped_reason.isna() & d.delta_matched_nullcorr.notna()]   # rows without any valid shift (no null) excluded
    lt = pd.read_csv(LT_TABLE).set_index("session_id")
    rows = []
    for dec, w, _ in ROWS:
        g = d[(d.decoding == dec) & (d.window == w)]
        half = g[g.definition == "half"].set_index("session_id")
        for df_ in DEFS:
            ltd = g[g.definition == df_].set_index("session_id")
            for rg in ("R+", "R-"):
                sids = set(half.index[half.reward_group == rg])
                has_lt = {s for s in sids if s in lt.index and pd.notna(lt.loc[s, df_])}
                no_lt = sids - has_lt
                L = ltd.loc[ltd.index.isin(has_lt) & (ltd.reward_group == rg), "delta_matched_nullcorr"]
                LH = half.loc[half.index.isin(L.index), "delta_matched_nullcorr"]
                N = half.loc[half.index.isin(no_lt), "delta_matched_nullcorr"]
                p_mw = stats.mannwhitneyu(L, N).pvalue if min(len(L), len(N)) >= 3 else np.nan
                p_w = stats.wilcoxon(L.loc[LH.index], LH).pvalue if len(LH) >= 5 else np.nan
                rows.append(dict(decoding=dec, window=w, definition=df_, cohort=rg,
                                 n_learners=len(L), learners_mean=L.mean(), learners_sem=sem(L),
                                 n_learners_half=len(LH), learners_half_mean=LH.mean(), learners_half_sem=sem(LH),
                                 n_no_lt=len(N), no_lt_mean=N.mean(), no_lt_sem=sem(N),
                                 p_mw_learnersLT_vs_noLThalf=p_mw, p_wilcoxon_learners_LT_vs_half=p_w,
                                 n_lt_defined_total=len(has_lt), n_lt_failed_classcounts=len(has_lt) - len(L)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"120_lt_defined_vs_undefined_stats{os.environ.get('SSL_LT_OUT_TAG', '')}.csv", index=False)
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(3, 2, figsize=(8.27, 8.6))
    fig.subplots_adjust(left=0.22, right=0.87, top=0.925, bottom=0.07, hspace=0.32, wspace=0.95)
    y = np.arange(len(DEFS))[::-1]
    for r, (dec, w, lab) in enumerate(ROWS):
        for c, rg in enumerate(("R+", "R-")):
            ax = axes[r, c]
            s = S[(S.decoding == dec) & (S.window == w) & (S.cohort == rg)].set_index("definition")
            for k, df_ in enumerate(DEFS):
                v = s.loc[df_]
                for gname, dy in (("learners", 0.22), ("learners_half", 0.0), ("no_lt", -0.22)):
                    n = v[f"n_{gname}"]
                    if n < 2:
                        continue
                    gs = GROUP[gname]
                    ax.errorbar(v[f"{gname}_mean"], y[k] + dy, xerr=v[f"{gname}_sem"], fmt=gs["m"], ms=3,
                                color=shade(COL[rg], gs["f"]), elinewidth=0.6, capsize=0)
                txt = (f"{int(v.n_learners)}/{int(v.n_no_lt)}  {pf(v.p_mw_learnersLT_vs_noLThalf)}  "
                       f"{pf(v.p_wilcoxon_learners_LT_vs_half)}")
                bold = (v.p_mw_learnersLT_vs_noLThalf < 0.05) or (v.p_wilcoxon_learners_LT_vs_half < 0.05)
                ax.text(1.02, y[k], txt, transform=ax.get_yaxis_transform(), fontsize=FS_S, va="center",
                        fontweight="bold" if bold else "normal")
            ax.text(1.02, len(DEFS) - 0.3, "n LT/noLT  MW  W", transform=ax.get_yaxis_transform(), fontsize=FS_S,
                    va="bottom")
            ax.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
            ax.set_yticks(y)
            ax.set_yticklabels(DEFS if c == 0 else [], fontsize=FS_S)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=FS_S)
            ax.set_ylim(-0.7, len(DEFS) - 0.3)
            ax.set_xlabel("change, post − pre (acc. − null, size-matched)", fontsize=FS_M)
            ax.set_title(f"{lab}, {'R+' if rg == 'R+' else 'R−'}", fontsize=FS_M, color=COL[rg], pad=12)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker=GROUP[g]["m"], ms=4, color=shade("0.2", GROUP[g]["f"])) for g in GROUP]
    fig.legend(hs, [GROUP[g]["lab"] for g in GROUP], loc="lower center", ncol=3, fontsize=FS_S, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    for ax, L in zip(axes[:, 0], "abc"):
        ax.text(-0.75, 1.08, L, transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    fig.suptitle("Sessions with a definable learning trial (split at the LT) vs sessions without one (session-half split), "
                 "per LT definition; whole brain, learning day, mean ± SEM\nMW: LT-split learners vs half-split "
                 "no-LT sessions; W: learners' LT split vs their own half split (paired); uncorrected, bold p < 0.05",
                 fontsize=FS_M, y=0.995)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"120_lt_defined_vs_undefined{os.environ.get('SSL_LT_OUT_TAG', '')}.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
