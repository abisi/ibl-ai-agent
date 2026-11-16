"""Exploration (user 2026-11-16, Figure 1 I): example mice for the reaction-time change over the first whisker hits.
Per example mouse (the two example learners of E-F, then the R+ / R- learners whose late change is closest to their
cohort median), the reaction time of each of the first 30 whisker hits of whisker day 0 (dots), its change from the median
of the first 5 hits (right axis is the same in ms), and the per-mouse median per 5-hit bin used in panel I (black line,
first bin = 0 by definition). Learners, early licks dropped, RT 0-1 s (compute.day0_hits).
Output: Figures/_explore/rt_examples_<date>.{pdf,png}.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_examples.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import compute as fig1  # noqa: E402
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"


def main():
    set_nature_rc()
    ctx = {"table": shared_behaviour.load()}
    h = fig1.day0_hits(ctx)
    h = h[(h.trial_type == "whisker_trial") & (h.hit_number <= 30)].dropna(subset=["rt_ms"])
    start = h[h.hit_number <= 5].groupby("mouse_id").rt_ms.median()
    h = h.assign(change=h.rt_ms - h.mouse_id.map(start))
    late = h[h.hit_number.between(21, 30)].groupby(["mouse_id", "cohort"]).change.mean().reset_index()
    picks = [fig1.PANELS["ex_rplus_learner"]["params"]["example"], fig1.PANELS["ex_rminus_learner"]["params"]["example"]]
    for c in ("R+", "R-"):
        g = late[late.cohort == c]
        g = g.iloc[(g.change - g.change.median()).abs().argsort()]
        picks += [m for m in g.mouse_id if m not in picks][:2]
    fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 90 * MM), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.35, hspace=0.45)
    for a, m in zip(axes.flat, picks):
        g = h[h.mouse_id == m].sort_values("hit_number")
        col = COLORS[g.cohort.iloc[0]]
        a.plot(g.hit_number, g.change, "o", ms=2, color=col, mew=0)
        b = g.assign(bin=(g.hit_number - 1) // 5).groupby("bin").change.median()
        a.plot(b.index * 5 + 3, b.values, color="k", lw=0.8, marker="s", ms=2)
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        a.axvspan(0.5, 5.5, color="0.93", lw=0, zorder=0)
        lm = late[late.mouse_id == m].change
        a.set_title(f"{m} ({g.cohort.iloc[0]}), hits 21-30: {lm.iloc[0]:+.0f} ms" if len(lm) else m, fontsize=5.5)
        a.set_ylabel("Δ RT from first 5 hits (ms)", fontsize=5)
        a.tick_params(labelsize=4.5)
    for a in axes[1]:
        a.set_xlabel("Whisker hit", fontsize=5)
        a.set_xticks([1, 10, 20, 30])
    fig.suptitle("Figure 1 I, example mice: dots = single hits (RT minus the median of hits 1-5, grey band); black = "
                 "median per 5-hit bin, as in the panel (first bin 0 by definition)", fontsize=6)
    stem = OUT / f"rt_examples_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    print(stem, picks)


if __name__ == "__main__":
    main()
