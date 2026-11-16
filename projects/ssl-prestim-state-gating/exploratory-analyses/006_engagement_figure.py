"""006 -- Publication figure of 005 (movement / arousal vs choice state; the state as a slow engagement variable). Whole brain,
day-0 learners, session = unit, mean +- s.e.m. over sessions, * Wilcoxon vs 0 (vs the null where stated). Style as 003.
  A  choice (whisker lick) prediction, 5-fold CV AUC: session time, pre-stimulus movements, state, and combinations, plus the
     early (5-35 ms) sensory readout on top of every pre-stimulus variable.
  B  early whisker response, joint OLS: standardised coefficient of each pre-stimulus variable (with lick and time).
  C  the early sensory hit / miss readout vs each pre-stimulus variable (Spearman per session).
  D  state autocorrelation over trials (time-residualised) vs trial-order permutation null; lick autocorrelation.
  E  choice prediction from the current trial's state vs the mean state of the previous K trials.
  F  10-trial blocks: block state vs block whisker hit rate, auditory hit rate, early whisker response (partial rho | block
     time, minus the linear-shift null over blocks).
  G  history: the previous trial's lick and reward -> current state (partial rho).
Linear-shift nulls (user 2026-11-12; 005 *_null, median over shifts per session): grey open markers (mean +- s.e.m. over
sessions) next to each cohort in A, B, C, E, G; stars there = Wilcoxon real vs null (paired); the null level is the slow,
shared component, real - null the trial-specific one. D keeps the trial-order permutation null (a shift of a series against
itself is the autocorrelation); F is already real - shift null.
Output: figures/006_engagement.{png,pdf,svg}; tables/006_engagement_stats.csv
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
f3 = importlib.import_module("003_figures")
COH, CL, WB, TAB, FIG = f3.COH, f3.CL, f3.WB, f3.TAB, f3.FIG
OFF = {"R+": -0.13, "R-": 0.13}


def cats(ax, items, t, panel, ylab, title, ref=0.0, null=None):
    """items: (label, column name or function); null: suffix of the per-session null column (005 *_null)."""
    for j, (lab, col) in enumerate(items):
        f = (lambda d, col=col: d[col]) if isinstance(col, str) else col
        for c in COH:
            d = t[t.reward_group == c]
            what = lab.replace("\n", " ")
            x = j + OFF[c]
            if null and isinstance(col, str) and f"{col}{null}" in d:
                nv = d[f"{col}{null}"] - ref
                ax.errorbar(x + 0.07, nv.mean(), nv.sem(), fmt="o", color="0.55", mfc="white", ms=2.4, mew=0.6, elinewidth=0.6,
                            capsize=0, zorder=2)
                ax.plot([x + 0.07, x], [nv.mean(), (f(d) - ref).mean()], color="0.75", lw=0.4, zorder=1)
                diff = f(d) - d[f"{col}{null}"]
                p = f3.wil(diff)
                f3.log(panel, what + " − shift null", c, diff, p, test="Wilcoxon real vs linear-shift null")
                f3.log(panel, what + " (shift null)", c, nv, f3.wil(nv))
                f3.msem(ax, x, f(d) - ref, COH[c], ms=3, star=False, panel=panel, what=what)
                if f3.stars(p):
                    v = (f(d) - ref).dropna()
                    ax.annotate(f3.stars(p), (x, v.mean() + v.sem()), xytext=(0, 1), textcoords="offset points", ha="center",
                                va="bottom", color=COH[c], fontsize=6)
            else:
                f3.msem(ax, x, f(d) - ref, COH[c], ms=3, panel=panel, what=what)
    f3.zero(ax)
    ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylabel(ylab); ax.set_title(title)


def main():
    J = pd.read_parquet(TAB / "005_joint.parquet").query("area == @WB")
    E = pd.read_parquet(TAB / "005_engagement.parquet").query("area == @WB")
    A = pd.read_parquet(TAB / "005_acf.parquet").query("area == @WB")
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.985, top=0.92, bottom=0.14, wspace=0.6, hspace=0.95)
    ax = {k: fig.add_subplot(gs[i // 4, i % 4]) for i, k in enumerate("ABCDEFG")}

    cats(ax["A"], [("Time", "auc_time"), ("Movements", "auc_mov"), ("State", "auc_state"), ("Time + state", "auc_time_state"),
                   ("Mov. + state", "auc_mov_state"), ("Pre-stim. all", "auc_pre_state"), ("+ early readout", "auc_all")],
         J, "A", "Choice prediction\n(CV AUC − 0.5)", "What predicts the choice", ref=0.5, null="_null")
    cats(ax["B"], [("Jaw", "b_jaw_motion"), ("Nose", "b_nose_motion"), ("Whisker", "b_whisker_motion"), ("Pupil", "b_pupil_area"),
                   ("State", "b_state")],
         J, "B", "Joint coefficient on early\nwhisker response (z)", "Movement, not state,\nsets the response", null="_null")
    cats(ax["C"], [("Jaw", "rho_ro_jaw_motion"), ("Nose", "rho_ro_nose_motion"), ("Whisker", "rho_ro_whisker_motion"),
                   ("Pupil", "rho_ro_pupil_area"), ("State", "rho_ro_state"), ("Lick | all\npre-stim.", "prho_ro_lick")],
         J, "C", "ρ(early hit/miss readout,\npre-stimulus variable)", "Early hit/miss readout\ninherits pre-stimulus state",
         null="_null")

    a = ax["D"]
    for c in COH:
        t = A[A.reward_group == c]
        for col, ls in (("acf_state", "-"), ("acf_lick", ":")):
            g = t.groupby("lag")[col]
            a.plot(g.mean().index, g.mean(), color=COH[c], ls=ls, lw=0.8, label=f"{CL[c]} {'state' if col == 'acf_state' else 'lick'}")
            a.fill_between(g.mean().index, g.mean() - g.sem(), g.mean() + g.sem(), color=COH[c], alpha=0.15, lw=0, edgecolor="none")
        g = t.groupby("lag").acf_null
        a.plot(g.mean().index, g.mean(), color="0.5", lw=0.6, ls="--")
    f3.zero(a)
    a.set_xlabel("Lag (trials)"); a.set_ylabel("Autocorrelation\n(time-residualised)")
    a.legend(frameon=False, fontsize=4.5, handlelength=1.6); a.set_title("Little trial-to-trial persistence")
    for c in COH:
        e = E[E.reward_group == c]
        f3.log("D", "acf lags 1-5 − permutation null", c, e.acf_mean_1_5 - e.acf_mean_1_5_null, f3.wil(e.acf_mean_1_5 - e.acf_mean_1_5_null))
        f3.STATS.append(dict(panel="D", measure="tau (trials), median", cohort=c, n=e.tau.notna().sum(), mean=e.tau.median()))

    cats(ax["E"], [("Current", "auc_current"), ("Past 1", "auc_past1"), ("Past 3", "auc_past3"), ("Past 5", "auc_past5"),
                   ("Past 10", "auc_past10")],
         E, "E", "Whisker choice prediction\n(AUC − 0.5)", "Past trials' state\npredicts the choice", ref=0.5, null="_null")
    cats(ax["F"], [("Whisker\nhit rate", lambda t: t.blk_wh_hit - t.blk_wh_hit_null),
                   ("Auditory\nhit rate", lambda t: t.blk_aud_hit - t.blk_aud_hit_null),
                   ("Early whisker\nresponse", lambda t: t.blk_g - t.blk_g_null)],
         E, "F", "Block state vs block measure\n(partial ρ − shift null)", "10-trial blocks: no slow gating")
    cats(ax["G"], [("Previous\nlick", "prho_prev_lick"), ("Previous\nreward | lick", "prho_prev_reward")],
         E, "G", "Partial ρ with the\ncurrent trial's state", "Trial history", null="_null")

    for k, a in ax.items():
        bb = a.get_position()
        fig.text(bb.x0 - 0.06, bb.y1 + 0.035, k, fontsize=9, fontweight="bold", va="bottom")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"006_engagement.{ext}", dpi=300)
    S = pd.DataFrame(f3.STATS)
    S.to_csv(TAB / "006_engagement_stats.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(S.round(4).to_string())


if __name__ == "__main__":
    main()
