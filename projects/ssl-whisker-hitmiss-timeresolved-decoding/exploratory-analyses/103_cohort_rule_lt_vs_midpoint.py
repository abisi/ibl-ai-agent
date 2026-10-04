"""103 -- Hit/miss decoding change around the learning trial vs around the session midpoint, with the
cohort-specific learning-trial definition (ssl-learning-trial-identification 020; sigma = 1 curves,
whisker-direction gate) -- user request 2026-09-25.

Comparisons (user 2026-09-25): AROUND THE LEARNING TRIAL only for learners, R+ vs R- (and each cohort's
learners LT vs their own midpoint, paired); AROUND THE MIDPOINT learners vs non-learners, per cohort and
merged. Non-learners include fast non-lickers and the one R- immediate learner. (Pseudo LTs for
non-learners were computed in the LT run but are not used.)
Decoding (095 pipeline, whole brain, disengaged trials kept): size-matched balanced accuracy minus the
session-level linear shift null, pre vs post; change = post - pre.
  LT split:  095_lt_split_shiftnull_nodisengagedrop_lt-lt_eval_whole_brain.parquet
  midpoint:  095_lt_split_shiftnull_halfsplit_nodisengagedrop_whole_brain.parquet
Stats per group x window: change at LT and at midpoint (mean, Wilcoxon + t vs 0), paired LT vs
midpoint (Wilcoxon + paired t); learners vs non-learners on change at LT and on (LT - midpoint)
(Mann-Whitney + Welch), within cohort and merged.
Outputs: figures/whole_brain/learning/103_hitmiss_cohortrule_lt_vs_mid_whole_brain.png,
103_cohortrule_lt_vs_mid_stats.csv, 103_cohortrule_between_groups.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, ttest_rel, wilcoxon

OUT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("q034", OUT / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(q034)

LT_TABLE = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "020_lt_eval.csv"
F_LT = "095_lt_split_shiftnull_nodisengagedrop_lt-lt_eval_whole_brain.parquet"
F_MID = "095_lt_split_shiftnull_halfsplit_nodisengagedrop_whole_brain.parquet"
WINDOWS = {"sensory": "sensory 5-50ms", "baseline": "baseline -200..-10ms", "sensory_minus_base": "sensory minus baseline"}
COL = {"R+": "#00B400", "R-": "#C800C8", "all": "#2c5f5b"}
GROUPS = [("R+ learners", "R+", "learner"), ("R+ non-learners", "R+", "non_learner"), ("R- learners", "R-", "learner"),
          ("R- non-learners", "R-", "non_learner"), ("R+ (all)", "R+", None), ("R- (all)", "R-", None),
          ("learners (R+ & R-)", None, "learner"), ("non-learners (R+ & R-)", None, "non_learner")]


def load(f):
    d = pd.read_parquet(OUT / f)
    d = d[d.skipped_reason.isna()].copy()
    d["d"] = (d.acc_post_matched - d.nullmean_post_matched) - (d.acc_pre_matched - d.nullmean_pre_matched)
    return d[["session_id", "window", "d"]]


def p1(x):
    x = x[~np.isnan(x)]
    return (np.nan, np.nan) if len(x) < 3 else (wilcoxon(x).pvalue, ttest_1samp(x, 0).pvalue)


def p2(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return (np.nan, np.nan) if ok.sum() < 3 else (wilcoxon(a[ok], b[ok]).pvalue, ttest_rel(a[ok], b[ok]).pvalue)


def pu(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    return (np.nan, np.nan) if min(len(a), len(b)) < 2 else (mannwhitneyu(a, b).pvalue, ttest_ind(a, b, equal_var=False).pvalue)


def main():
    tab = pd.read_csv(LT_TABLE)[["session_id", "reward_group", "group", "learning_category"]]
    m = load(F_LT).merge(load(F_MID), on=["session_id", "window"], how="outer", suffixes=("_lt", "_mid")).merge(tab, on="session_id")
    m.loc[m.group != "learner", "d_lt"] = np.nan   # user 2026-09-25: no LT comparison for non-learners (pseudo LTs unused)
    m.to_csv(OUT / "103_cohortrule_persession.csv", index=False)
    rows = []

    def add(w, comp, a, b, paired):
        pa, pb = p1(a), p1(b)
        pc = p2(a, b) if paired else pu(a, b)
        rows.append(dict(window=w, comparison=comp, n_a=int(np.sum(~np.isnan(a))), n_b=int(np.sum(~np.isnan(b))),
                         mean_a=np.nanmean(a), mean_b=np.nanmean(b), p_w_a_vs0=pa[0], p_t_a_vs0=pa[1], p_w_b_vs0=pb[0],
                         p_t_b_vs0=pb[1], test="paired Wilcoxon / paired t" if paired else "Mann-Whitney / Welch",
                         p_nonparam=pc[0], p_param=pc[1]))
        return pa, pb, pc

    fig, axes = plt.subplots(len(WINDOWS), 5, figsize=(21, 4.4 * len(WINDOWS)), constrained_layout=True, sharey="row",
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1, 1]))
    for r, (w, wl) in enumerate(WINDOWS.items()):
        g = m[m.window == w]
        L = g[g.group == "learner"]
        # A: around the learning trial, learners, R+ vs R-
        ax = axes[r, 0]
        a, b = L[L.reward_group == "R+"].d_lt.to_numpy(), L[L.reward_group == "R-"].d_lt.to_numpy()
        pa, pb, pc = add(w, "around LT (learners): R+ vs R-", a, b, False)
        for k, (v, rg) in enumerate(((a, "R+"), (b, "R-"))):
            vv = v[~np.isnan(v)]
            ax.scatter(k + np.random.default_rng(k).uniform(-0.1, 0.1, len(vv)), vv, s=14, color=COL[rg], alpha=0.7, lw=0)
            ax.errorbar(k + 0.25, vv.mean(), yerr=vv.std() / np.sqrt(len(vv)), color="k", marker="o", capsize=3)
        ax.set_xticks([0, 1], [f"R+ learners\n(n={np.sum(~np.isnan(a))})", f"R- learners\n(n={np.sum(~np.isnan(b))})"], fontsize=8)
        ax.set_title(f"AROUND THE LEARNING TRIAL | {wl}\nvs 0: R+ W p={pa[0]:.2g}, R- W p={pb[0]:.2g}\nR+ vs R-: MW p={pc[0]:.2g}, "
                     f"Welch p={pc[1]:.2g}", fontsize=8)
        ax.set_ylabel(f"{wl}\nchange post - pre\n(matched acc - shift null)", fontsize=8)
        # B, C: learners LT vs midpoint (paired), per cohort
        for c, rg in ((1, "R+"), (2, "R-")):
            ax = axes[r, c]
            Lr = L[L.reward_group == rg]
            a, b = Lr.d_lt.to_numpy(), Lr.d_mid.to_numpy()
            pa, pb, pc = add(w, f"{rg} learners: LT vs midpoint", a, b, True)
            for i in range(len(a)):
                if not (np.isnan(a[i]) or np.isnan(b[i])):
                    ax.plot([0, 1], [a[i], b[i]], color=COL[rg], alpha=0.3, lw=0.8)
            for k, v in enumerate((a, b)):
                vv = v[~np.isnan(v)]
                ax.errorbar(k, vv.mean(), yerr=vv.std() / np.sqrt(len(vv)), color="k", marker="o", capsize=3, zorder=5)
            ax.set_xticks([0, 1], ["learning trial", "midpoint"], fontsize=8)
            ax.set_title(f"{rg} learners: LT vs midpoint (paired)\nvs 0: LT p={pa[0]:.2g}, mid p={pb[0]:.2g}\n"
                         f"LT vs mid: W p={pc[0]:.2g}, t p={pc[1]:.2g}", fontsize=8)
        # D, E: around the midpoint, learners vs non-learners, per cohort and merged
        for c, (lab, rgs) in ((3, ("R+ / R- separately", ("R+", "R-"))), (4, ("merged cohorts", (None,)))):
            ax = axes[r, c]
            ticks, labs, ttl = [], [], []
            x0 = 0
            for rg in rgs:
                gg = g if rg is None else g[g.reward_group == rg]
                a, b = gg[gg.group == "learner"].d_mid.to_numpy(), gg[gg.group == "non_learner"].d_mid.to_numpy()
                pa, pb, pc = add(w, f"around midpoint ({rg or 'merged'}): learners vs non-learners", a, b, False)
                for k, v in enumerate((a, b)):
                    vv = v[~np.isnan(v)]
                    colr = COL[rg or "all"]
                    ax.scatter(x0 + k + np.random.default_rng(k).uniform(-0.1, 0.1, len(vv)), vv, s=12, color=colr,
                               alpha=0.7 if k == 0 else 0.3, lw=0)
                    ax.errorbar(x0 + k + 0.25, vv.mean(), yerr=vv.std() / np.sqrt(len(vv)), color="k", marker="o", capsize=3)
                    ticks.append(x0 + k)
                    labs.append(f"{rg or 'all'} {'L' if k == 0 else 'NL'}\n(n={len(vv)})")
                ttl.append(f"{rg or 'merged'}: L p={pa[0]:.2g}, NL p={pb[0]:.2g}; L vs NL MW p={pc[0]:.2g}, Welch p={pc[1]:.2g}")
                x0 += 2.5
            ax.set_xticks(ticks, labs, fontsize=7.5)
            ax.set_title(f"AROUND THE MIDPOINT: learners vs non-learners ({lab})\n" + "\n".join(ttl), fontsize=7.5)
        for a_ in axes[r]:
            a_.axhline(0, color="#888888", lw=1, ls=":")
            a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hit/miss decoding (whole brain, size-matched, shift null, disengaged trials kept), cohort-specific learning "
                 "trials (sigma = 1, whisker gate): around the learning trial (learners, R+ vs R-) and around the session "
                 "midpoint (learners vs non-learners). Non-learners include fast non-lickers.", fontsize=11)
    q034.savefig_retry(fig, q034.fig_dir("whole_brain") / "103_hitmiss_cohortrule_lt_vs_mid_whole_brain.png", dpi=200, bbox_inches="tight")
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "103_cohortrule_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(st.round(4).to_string())
    print(m[m.window == "sensory"].groupby(["reward_group", "group"]).agg(n_lt=("d_lt", "count"), n_mid=("d_mid", "count")).to_string())


if __name__ == "__main__":
    main()
