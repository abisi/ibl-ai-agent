"""018 -- Data check before model v4 (model_v4_design.md sec. 7): is whisker learning carried by within-session changes or by steps at
session boundaries (offline gains)? User hypothesis 2026-10-23: R- learns fast, R+ slowly over days.
Per mouse and whisker-day boundary d -> d+1 (d >= 0): lick rate per trial type in the last N trials of day d (end) and the first N trials
of day d+1 (start), and the first N trials of day d (start of the previous day).
  offline step   = start(d+1) - end(d)     (includes the thirst reset at session start)
  thirst control = the same step for auditory trials; whisker step - auditory step is the thirst-controlled whisker step
  across days    = start(d+1) - start(d)   (same thirst state)
Trials: 011 sequences (active, perf != 6, no trimming). Unit: mouse x boundary; statistics per boundary pooled with mouse as unit
(boundaries averaged within mouse). One-sample: Wilcoxon + t test; R+ vs R-: Mann-Whitney + Welch t (both reported).
Output: tables/018_boundaries.csv, tables/018_stats.csv, figures/018_session_boundaries.{png,pdf,svg}
Run (haas): python 018_session_boundaries.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

N = 80
TYPES = ["auditory", "whisker", "no_stim"]
COH = {"R+": "#00B400", "R-": "#C800C8"}


def rates(g):
    return {t: (g.lick_flag[g.trial_type == t].mean() if (g.trial_type == t).sum() >= 5 else np.nan) for t in TYPES}


def build():
    d = pd.read_parquet(C.TAB / "011_sequences.parquet")
    d = d[d.day >= 0]
    rows = []
    for mid, g in d.groupby("mouse_id"):
        sess = {day: s.sort_values("start_time") for day, s in g.groupby("day")}
        for day in sorted(sess):
            if day + 1 not in sess:
                continue
            a, b = sess[day], sess[day + 1]
            if len(a) < 2 * N or len(b) < N:
                continue
            ra_start, ra_end, rb_start = rates(a.iloc[:N]), rates(a.iloc[-N:]), rates(b.iloc[:N])
            for t in TYPES:
                rows.append(dict(mouse_id=mid, cohort=g.reward_group.iloc[0], session_id_d=a.session_id.iloc[0],
                                 session_id_d1=b.session_id.iloc[0], boundary=f"{day}->{day + 1}", trial_type=t,
                                 start_d=ra_start[t], end_d=ra_end[t], start_d1=rb_start[t]))
    B = pd.DataFrame(rows)
    B["offline"] = B.start_d1 - B.end_d
    B["across"] = B.start_d1 - B.start_d
    B["within"] = B.end_d - B.start_d
    return B


def two_tests(x):
    x = x.dropna()
    return dict(n=len(x), mean=x.mean(), median=x.median(), p_wilcoxon=stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan,
                p_ttest=stats.ttest_1samp(x, 0).pvalue if len(x) > 2 else np.nan)


def main():
    B = build()
    B.to_csv(C.TAB / "018_boundaries.csv", index=False)
    W = B.pivot_table(index=["mouse_id", "cohort", "boundary"], columns="trial_type", values=["offline", "across", "within"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    W["offline_whisker_minus_auditory"] = W.offline_whisker - W.offline_auditory
    M = W.drop(columns="boundary").groupby(["mouse_id", "cohort"]).mean().reset_index()      # mouse = unit
    qs = ["within_whisker", "offline_whisker", "across_whisker", "offline_auditory", "offline_no_stim", "offline_whisker_minus_auditory"]
    rows = []
    for q in qs:
        for c in ["R+", "R-"]:
            rows.append(dict(quantity=q, cohort=c, **two_tests(M[M.cohort == c][q])))
        x, y = M[M.cohort == "R+"][q].dropna(), M[M.cohort == "R-"][q].dropna()
        rows.append(dict(quantity=q, cohort="R+ vs R-", n=len(x) + len(y), mean=x.mean() - y.mean(), median=x.median() - y.median(),
                         p_mannwhitney=stats.mannwhitneyu(x, y).pvalue, p_welch=stats.ttest_ind(x, y, equal_var=False).pvalue))
    S = pd.DataFrame(rows)
    S.to_csv(C.TAB / "018_stats.csv", index=False)
    print(S.round(4).to_string())
    print(B.groupby(["cohort", "boundary", "trial_type"])[["start_d", "end_d", "start_d1"]].mean().round(3))
    figure(B, M)


def figure(B, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.titlesize": 6.5})
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.6), gridspec_kw=dict(wspace=0.45, hspace=0.6))
    for i, c in enumerate(["R+", "R-"]):
        for j, t in enumerate(TYPES):
            ax = axs[i, j]
            e = B[(B.cohort == c) & (B.trial_type == t)].groupby("mouse_id")[["start_d", "end_d", "start_d1"]].mean()
            for _, r in e.iterrows():
                ax.plot([0, 1, 2], r.values, color=COH[c], alpha=0.15, lw=0.6)
            m, s = e.mean(), e.sem()
            ax.errorbar([0, 1, 2], m.values, yerr=s.values, color="k", lw=1.2, capsize=2)
            ax.set_xticks([0, 1, 2], ["start d", "end d", "start d+1"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{c.replace('-', '−')} {t.replace('_', ' ')} (n = {len(e)} mice)")
            if j == 0:
                ax.set_ylabel(f"lick rate ({N} trials)")
    for i, (q, lab) in enumerate([("offline_whisker_minus_auditory", "offline step: whisker − auditory"),
                                   ("within_whisker", "within-session: whisker end − start")]):
        ax = axs[i, 3]
        for k, c in enumerate(["R+", "R-"]):
            v = M[M.cohort == c][q].dropna()
            ax.scatter(k + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=4, color=COH[c], alpha=0.6, lw=0)
            ax.hlines(v.mean(), k - 0.25, k + 0.25, color="k")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks([0, 1], ["R+", "R−"]); ax.set_title(lab)
    fig.suptitle("Whisker days: lick rate at session boundaries (mouse means over boundaries d → d+1, d ≥ 0)", y=0.99)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"018_session_boundaries.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
