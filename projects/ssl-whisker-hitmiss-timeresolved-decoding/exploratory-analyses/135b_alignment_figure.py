"""135b -- Figure: whisker axis . lick axis across passive_pre, active 1st / 2nd half, passive_post (5-35 ms), with "bad" R+ mice
removed (user 2026-10-02: "Repeat, removing bad R+. Make a figure. Is this a trivial result? Min number of trials?").
  a  whisker axis (whisker - auditory) . lick axis (active hits - misses), whole brain, >= 10 licked and unlicked trials;
  b  decomposition: whisker-EVOKED pattern alone . lick axis;
  c  decomposition: auditory-EVOKED pattern alone . lick axis -- if (a) were only auditory picking up reward / lick
     expectation, (c) would carry the effect and (b) would be flat;
  d  same as (a) with the lower threshold (>= 6 licked and unlicked whisker trials, SSL_MIN_CLASS=3);
  e-g  (a) for midbrain, motor areas, striatum.
Per panel: mean +- SEM per cohort, faint lines = mice; within cohort epoch effect (Friedman AND RM-ANOVA); stars = R+ vs R- on
the change from passive_pre (** MW AND Welch p < .05, * one test). R+ "bad" learners (learning_category == 'bad') excluded;
R- unchanged (user). Uncorrected.
Inputs: 135_alignment_epochs.parquet, 135_alignment_epochs_min3.parquet, 024 per-mouse table (learning_category).
Output: figures/135b_alignment_no_bad_rplus.{pdf,png,svg}, 135b_stats.csv
Run (haas): python 135b_alignment_figure.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}
EP4 = ["passive_pre", "active_1", "active_2", "passive_post"]
XL = ["passive\npre", "active\n1st half", "active\n2nd half", "passive\npost"]
FS = 6.5


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def load(tag, cat):
    d = pd.read_parquet(OUT / f"135_alignment_epochs{tag}.parquet")
    d = d[d.skipped_reason.isna()].copy()
    d["learning_category"] = d.session_id.map(cat)
    return d[~((d.reward_group == "R+") & (d.learning_category == "bad"))]


def panel(ax, g, prefix, title, rows, label):
    cols = [f"{prefix}{e}" for e in EP4]
    xs = np.arange(4)
    st = dict(panel=label, measure=prefix)
    for k, rg in enumerate(("R+", "R-")):
        x = g[g.reward_group == rg][cols].dropna()
        off = (k - 0.5) * 0.12
        for _, r in x.iterrows():
            ax.plot(xs + off, r.to_numpy(), color=COL[rg], lw=0.3, alpha=0.2)
        if len(x):
            ax.errorbar(xs + off, x.mean(), yerr=x.std(ddof=1) / np.sqrt(len(x)), color=COL[rg], lw=1.6, marker="o", ms=3,
                        capsize=0, label=f"{'R+' if rg == 'R+' else 'R−'} n={len(x)}")
        st[f"n_{rg}"] = len(x)
        for c, e in zip(cols, EP4):
            st[f"mean_{rg}_{e}"] = x[c].mean() if len(x) else np.nan
        if len(x) >= 4:
            from statsmodels.stats.anova import AnovaRM
            st[f"friedman_{rg}"] = stats.friedmanchisquare(*[x[c] for c in cols]).pvalue
            long = x.reset_index().melt(id_vars="index", value_vars=cols, var_name="ep", value_name="v")
            st[f"rmanova_{rg}"] = float(AnovaRM(long, "v", "index", within=["ep"]).fit().anova_table["Pr > F"].iloc[0])
    a, b = g[g.reward_group == "R+"], g[g.reward_group == "R-"]
    for i, (c, e) in enumerate(zip(cols, EP4)):
        if e == "passive_pre":
            continue
        da, db = (a[c] - a[cols[0]]).dropna(), (b[c] - b[cols[0]]).dropna()
        if min(len(da), len(db)) >= 3:
            pm, pw = stats.mannwhitneyu(da, db).pvalue, stats.ttest_ind(da, db, equal_var=False).pvalue
            st[f"pMW_change_{e}"], st[f"pWelch_change_{e}"] = pm, pw
            mark = "**" if (pm < 0.05 and pw < 0.05) else ("*" if (pm < 0.05 or pw < 0.05) else "")
            ax.text(i, 1.0, mark, transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    rows.append(st)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(XL, fontsize=FS - 1)
    ax.set_title(f"{title}\nR+ F {pf(st.get('friedman_R+', np.nan))}/RM {pf(st.get('rmanova_R+', np.nan))} · "
                 f"R− F {pf(st.get('friedman_R-', np.nan))}/RM {pf(st.get('rmanova_R-', np.nan))}", fontsize=FS - 0.5)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(fontsize=FS - 1.5, frameon=False, loc="lower left")


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    per = pickle.load(open(LTP / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]
    cat = {p["session_id"]: p["learning_category"] for p in per}
    d5, d3 = load("", cat), load("_min3", cat)
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.1, wspace=0.42, hspace=0.85)
    wb5, wb3 = d5[d5.area == "All units"], d3[d3.area == "All units"]
    panel(axes[0, 0], wb5, "cosnorm_", "a  whisker axis · lick axis\n(whole brain, ≥10 trials/class)", rows, "a")
    panel(axes[0, 1], wb5, "evokedW_cosnorm_", "b  whisker-EVOKED · lick axis", rows, "b")
    panel(axes[0, 2], wb5, "evokedA_cosnorm_", "c  auditory-EVOKED · lick axis", rows, "c")
    panel(axes[0, 3], wb3, "cosnorm_", "d  as (a), ≥6 trials/class", rows, "d")
    for ax, ar, L in zip(axes[1, :3], ("Midbrain", "Motor areas", "Striatum"), "efg"):
        panel(ax, d5[d5.area == ar], "cosnorm_", f"{L}  {ar}", rows, L)
    axes[1, 3].set_axis_off()
    axes[1, 3].text(0, 0.95, "Lick axis: active whisker hits − misses\n(licks after 35 ms; independent trials).\n"
                    "Whisker axis: whisker − auditory per epoch.\nEvoked: baseline-subtracted response pattern\n"
                    "of one modality (not mean-centred).\nnorm. cos: split-half reliability-normalised.\n"
                    "R+ 'bad' learners excluded.\n** / *: R+ vs R− on the change from\npassive pre (MW and Welch / one).",
                    fontsize=FS - 0.5, va="top", transform=axes[1, 3].transAxes)
    for ax in axes[:, 0]:
        ax.set_ylabel("normalised cosine", fontsize=FS)
    fig.suptitle("Is the early (5-35 ms) whisker code oriented toward or away from licking? Tracked good units, epoch-specific "
                 "baseline, z-scored; within cohort: Friedman (F) / RM-ANOVA (RM); uncorrected.", fontsize=FS + 0.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"135b_alignment_no_bad_rplus.{ext}", dpi=250)
    plt.close(fig)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "135b_stats.csv", index=False)
    pd.set_option("display.width", 300)
    print(S.round(3).T.to_string())


if __name__ == "__main__":
    main()
