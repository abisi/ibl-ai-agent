"""Exploration (user 2026-11-09): three ways to emphasise day 0 and the R+ vs R- difference in panel b, drawn side by
side on one page, not part of Figure 1.

  Option 2: across-day context (thin lines, whisker in colour, auditory / false alarm grey) next to a day-0 close-up
            (mean +- s.e.m. and one dot per mouse, R+ next to R-, MW and Welch p).
  Option 3: R+ minus R- lick probability per day and trial type, 95% bootstrap CI over mice (10,000 resamples).
  Option 4: day 0 split into early and late trials (first and last N_EDGE whisker trials, with the auditory and
            no-stimulus trials in the same time spans), per-mouse dots, R+ vs R-.

Data: the shared behaviour table (NWB_ks4, all included mice, early licks dropped), same mice and days as panel b.
Output: Figures/_explore/panel_b_options_<date>.{pdf,png} and the CSVs behind each option.

Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_panel_b.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, two_group_tests  # noqa: E402
from paperstyle import MM, set_nature_rc  # noqa: E402
from plot import COHORTS, TRIAL_TYPES, TT_SHORT, group_color, stars  # noqa: E402

N_EDGE = 20  # whisker trials in the early and late parts of day 0
DAYS = ["-2", "-1", "0", ">0"]
DAY_LABELS = ["-2", "-1", "0", "+1-3"]
OUT = FIGURES_ROOT / "_explore"
GREY = {"auditory_trial": ("0.45", "0.7"), "no_stim_trial": ("0.15", "0.55")}  # option 2 context: R+, R-


def per_mouse_days(t):
    t = t[t.day.isin([-2, -1, 0, 1, 2, 3]) & (t.early_lick == 0)]
    s = t.groupby(["mouse_id", "cohort", "day", "session_id", "trial_type"]).lick_flag.mean().reset_index()
    d = s.groupby(["mouse_id", "cohort", "day", "trial_type"]).lick_flag.mean().reset_index()
    d["day_group"] = d.day.map(lambda x: str(x) if x <= 0 else ">0")
    return (d.groupby(["mouse_id", "cohort", "day_group", "trial_type"]).lick_flag.mean()
            .rename("lick_probability").reset_index())


def early_late(t):
    """Day 0: early = trials up to the N_EDGE-th whisker trial, late = trials from the N_EDGE-th last whisker trial."""
    rows = []
    d0 = t[(t.behavior == "whisker") & (t.day == 0) & (t.early_lick == 0)]
    for (mouse, cohort, sid), s in d0.groupby(["mouse_id", "cohort", "session_id"]):
        s = s.sort_values("start_time")
        w = s[s.trial_type == "whisker_trial"].start_time.to_numpy()
        if len(w) < 2 * N_EDGE:
            continue
        parts = {"early": s[s.start_time <= w[N_EDGE - 1]], "late": s[s.start_time >= w[-N_EDGE]]}
        for part, p in parts.items():
            for tt, q in p.groupby("trial_type"):
                rows.append({"mouse_id": mouse, "cohort": cohort, "session_id": sid, "part": part, "trial_type": tt,
                             "lick_probability": q.lick_flag.mean(), "n_trials": len(q)})
    return pd.DataFrame(rows)


def boot_diff(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def dots(ax, x, v, color, seed):
    v = np.asarray(v, float)
    j = np.random.default_rng(seed).uniform(-0.09, 0.09, len(v))
    ax.scatter(x + j, v, s=2.5, color=color, lw=0, alpha=1, zorder=2, edgecolor="none")
    ax.errorbar(x + 0.2, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color="0.15", marker="o", ms=2.2, mfc=color,
                mec="0.15", mew=0.4, capsize=0, elinewidth=0.7, zorder=3)


def cohort_pairs(ax, df, groups, key, ylabel, test_rows, label):
    """One R+ / R- dot pair per (group, trial type); p from MW (on panel) and Welch (table)."""
    x0 = 0
    ticks, ticklabels = [], []
    for g in groups:
        for tt in TRIAL_TYPES:
            vals = {}
            for k, c in enumerate(COHORTS):
                v = df[(df[key] == g) & (df.trial_type == tt) & (df.cohort == c)].lick_probability.dropna()
                dots(ax, x0 + k * 0.55, v, group_color(tt, c), seed=int(k + 10 * x0))
                vals[c] = v
            r = two_group_tests(vals["R+"], vals["R-"])
            test_rows.append({"option": label, key: g, "trial_type": tt, **r})
            y = max(vals["R+"].max(), vals["R-"].max()) + 0.04
            ax.plot([x0, x0 + 0.55], [y, y], color="0.2", lw=0.4)
            ax.text(x0 + 0.275, y, stars(r["mannwhitney_p"]), ha="center", va="bottom", fontsize=4.5)
            ticks.append(x0 + 0.275)
            ticklabels.append(TT_SHORT[tt])
            x0 += 1.5
        x0 += 0.6
    ax.set_xticks(ticks, ticklabels)
    ax.set_ylim(-0.03, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel(ylabel, labelpad=1)
    return ticks


def option2(ax_ctx, ax_d0, days, tests):
    for tt in TRIAL_TYPES:
        for k, c in enumerate(COHORTS):
            d = days[(days.trial_type == tt) & (days.cohort == c)].groupby("day_group").lick_probability
            m, s = d.mean().reindex(DAYS), d.sem().reindex(DAYS)
            col = group_color(tt, c) if tt == "whisker_trial" else GREY[tt][k]
            lw = 1.0 if tt == "whisker_trial" else 0.5
            ax_ctx.errorbar(np.arange(4) + (k - 0.5) * 0.08, m, s, color=col, marker="o", ms=1.6 if lw > 0.6 else 1,
                            lw=lw, elinewidth=0.5, capsize=0, zorder=3 if tt == "whisker_trial" else 2)
    ax_ctx.axvspan(1.6, 2.4, color="0.92", lw=0, zorder=0)
    ax_ctx.set_xticks(range(4), DAY_LABELS)
    ax_ctx.set_xlim(-0.3, 3.3)
    ax_ctx.set_ylim(-0.03, 1.03)
    ax_ctx.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax_ctx.set_xlabel("Day", labelpad=1)
    ax_ctx.set_ylabel("Lick probability", labelpad=1)
    for tt, y in zip(TRIAL_TYPES, (0.97, 0.89, 0.81)):
        ax_ctx.text(0.02, y, TT_SHORT[tt], transform=ax_ctx.transAxes, fontsize=4.5,
                    color=group_color(tt, "R+") if tt == "whisker_trial" else GREY[tt][0], va="top")
    cohort_pairs(ax_d0, days[days.day_group == "0"].assign(g="0"), ["0"], "g", "Lick probability, day 0", tests,
                 "option2_day0")
    ax_d0.set_title("Day 0", fontsize=6, pad=2)
    # day-0 band continues into the close-up
    ax_d0.set_facecolor("0.96")


def option3(ax, days, rows):
    for i, tt in enumerate(TRIAL_TYPES):
        for j, dg in enumerate(DAYS):
            d = days[(days.trial_type == tt) & (days.day_group == dg)]
            a, b = d[d.cohort == "R+"].lick_probability, d[d.cohort == "R-"].lick_probability
            if len(a) < 2 or len(b) < 2:
                continue
            m, lo, hi = boot_diff(a, b)
            r = two_group_tests(a, b)
            rows.append({"option": "option3", "day_group": dg, "trial_type": tt, "diff": m, "ci_low": lo, "ci_high": hi,
                         **r})
            x = j + (i - 1) * 0.18
            ax.plot([x, x], [lo, hi], color=group_color(tt, "R+"), lw=0.8)
            ax.plot(x, m, "o", ms=2.2, color=group_color(tt, "R+"), mec="none")
            if r["mannwhitney_p"] < 0.05:
                ax.text(x, hi + 0.02, stars(r["mannwhitney_p"]), ha="center", va="bottom", fontsize=4.5)
    ax.axhline(0, color="0.5", lw=0.4, ls=":")
    ax.axvspan(1.6, 2.4, color="0.92", lw=0, zorder=0)
    ax.set_xticks(range(4), DAY_LABELS)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("Day", labelpad=1)
    ax.set_ylabel("R+ − R- lick probability", labelpad=1)
    for tt, y in zip(TRIAL_TYPES, (0.97, 0.89, 0.81)):
        ax.text(0.02, y, TT_SHORT[tt], transform=ax.transAxes, fontsize=4.5, color=group_color(tt, "R+"), va="top")


def option4(ax, el, tests):
    per_mouse = el.groupby(["mouse_id", "cohort", "part", "trial_type"]).lick_probability.mean().reset_index()
    ticks = cohort_pairs(ax, per_mouse, ["early", "late"], "part", "Lick probability, day 0", tests, "option4")
    for k, part in enumerate(["Early", "Late"]):
        ax.text(np.mean(ticks[3 * k:3 * k + 3]), -0.2, f"{part} (first {N_EDGE} W)" if k == 0 else f"{part} (last {N_EDGE} W)",
                ha="center", va="top", fontsize=5, transform=ax.get_xaxis_transform())


def main():
    set_nature_rc()
    t = shared_behaviour.load()
    days, el = per_mouse_days(t), early_late(t)
    tests, diffs = [], []
    fig = plt.figure(figsize=(183 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.75, 0.25, 1.15, 1.6], left=0.05, right=0.99, bottom=0.22, top=0.86,
                          wspace=0.45)
    a_ctx, a_d0 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    a3, a4 = fig.add_subplot(gs[3]), fig.add_subplot(gs[4])
    option2(a_ctx, a_d0, days, tests)
    option3(a3, days, diffs)
    option4(a4, el, tests)
    n = days[days.day_group == "0"].groupby("cohort").mouse_id.nunique().to_dict()
    n_el = el.groupby("cohort").mouse_id.nunique().to_dict()
    for ax, lab in ((a_ctx, "Option 2: context + day-0 close-up"), (a3, "Option 3: R+ − R- (95% bootstrap CI)"),
                    (a4, "Option 4: day 0 early vs late")):
        ax.text(0, 1.1, lab, transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="bottom")
    fig.text(0.01, 0.01, f"Day 0: {n.get('R+')} R+ and {n.get('R-')} R- mice (options 2, 3); option 4: mice with at least "
             f"{2 * N_EDGE} whisker trials ({n_el.get('R+')} R+, {n_el.get('R-')} R-). Asterisks: Mann-Whitney R+ vs R-; "
             "Welch in the CSV. Dots, mice; black-edged dot, mean ± s.e.m.", fontsize=4.5)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"panel_b_options_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)
    pd.DataFrame(tests).to_csv(f"{stem}_tests.csv", index=False)
    pd.DataFrame(diffs).to_csv(f"{stem}_diff.csv", index=False)
    days.to_csv(f"{stem}_days.csv", index=False)
    el.to_csv(f"{stem}_early_late.csv", index=False)
    print(stem)


if __name__ == "__main__":
    main()
