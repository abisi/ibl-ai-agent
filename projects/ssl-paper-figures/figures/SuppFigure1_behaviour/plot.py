"""SuppFigure1_behaviour -- one plot_<key>(ax, df) per panel; style and helpers from the Figure 1 plot module."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("fig1_plot", Path(__file__).resolve().parents[1] / "Figure1_behaviour" / "plot.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

plot_pretraining = fig1.plot_pretraining
plot_movement_peth = fig1.plot_movement_peth
_spec2 = importlib.util.spec_from_file_location("s1_correlates_plot", Path(__file__).resolve().parent / "correlates_plot.py")
s2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(s2)
plot_lick_course, plot_rt_hits_other = s2.plot_lick_course, s2.plot_rt_hits_other
plot_rt_curves, plot_posture_curve = s2.plot_rt_curves, s2.plot_posture_curve
PREV = ["prev_whisker_hit", "prev_whisker_miss", "prev_auditory_hit", "prev_auditory_miss"]
PREV_LABEL = ["W hit", "W miss", "A hit", "A miss"]
RATE_VARS = {"aud_day0": "Auditory day 0", "fa_day0": "No-stim day 0", "aud_daym1": "Auditory day -1",
             "fa_daym1": "No-stim day -1", "wh_day1": "Whisker day +1", "wh_day2": "Whisker day +2",
             "n_pretraining_days": "Pretraining days", "n_pretraining_trials": "Pretraining trials",
             "rt_wh_day0": "Whisker RT day 0 (s)"}  # as compute.RATE_VARS


def plot_rt_change_auditory(ax, df):
    """Auditory-hit version of Figure 1 k left (user 2026-11-10)."""
    fig1.plot_rt_change(ax, df[df.kind == "bin"], "Auditory hit", df[df.kind == "fit"])


def plot_ex_rplus_nonlearner(ax, df):
    fig1._example(ax, df, "R+ non-learner", first=True)


def plot_ex_rminus_nonlearner(ax, df):
    fig1._example(ax, df, "R- non-learner")


def plot_history(ax, df):
    """Bars (planner 2026-11-09): P(lick) shift (rate given the previous stimulus trial minus the mouse's overall rate)
    and RT shift on hits, for W and A trials; mean +- s.e.m. over mice, Mann-Whitney asterisks R+ vs R-."""
    grid = fig1._subgrid(ax, 2, 2, wgap=0.45, hgap=0.75)  # rows W, A; columns P(lick) shift, RT shift
    axes = [grid[0][0], grid[0][1], grid[1][0], grid[1][1]]
    panels = [("whisker_trial", "p_lick_shift", "W trial"), ("whisker_trial", "rt_shift_s", "W hit"),
              ("auditory_trial", "p_lick_shift", "A trial"), ("auditory_trial", "rt_shift_s", "A hit")]
    for a, (tt, v, lab) in zip(axes, panels):
        for i, cond in enumerate(PREV):
            vals = {}
            for k, c in enumerate(fig1.COHORTS):
                d = df[(df.trial_type == tt) & (df.prev == cond) & (df.cohort == c)][v].dropna()
                a.bar(i + (k - 0.5) * 0.36, d.mean(), 0.34, color=fig1.group_color(tt, c), lw=0)
                a.errorbar(i + (k - 0.5) * 0.36, d.mean(), d.sem(), color="0.2", lw=0, elinewidth=0.5, capsize=0)
                vals[c] = d
            p = fig1._mw_p(vals["R+"], vals["R-"])
            if p < 0.05:
                a.text(i, 1.0, fig1.stars(p), transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.5)
        a.axhline(0, color="0.4", lw=0.4)
        a.set_ylabel("Δ P(lick)" if v == "p_lick_shift" else "Δ RT (s)", labelpad=0.5, fontsize=4.5)
        a.tick_params(labelsize=4)
        a.set_xticks(range(4), PREV_LABEL if a in axes[2:] else [], rotation=45, fontsize=4)
        a.set_xlim(-0.5, 3.5)
        a.set_title(lab, fontsize=5, pad=8)
    for a in axes[2:]:
        a.set_xlabel("Previous trial", labelpad=1, fontsize=4.5)


def plot_rates_correlation(ax, df):
    """Whisker P(lick) day 0 (y) against each measure (x): dots per mouse, OLS line with 95% band per cohort (solid if
    p < 0.05, dashed otherwise), r and p in the corner."""
    from scipy import stats
    vars_ = list(RATE_VARS)
    axes = fig1._split(ax, len(vars_), gap=0.025)
    for k, (a, v) in enumerate(zip(axes, vars_)):
        titles = []
        for j, c in enumerate(fig1.COHORTS):
            d = df[df.cohort == c][[v, "wh_day0"]].dropna()
            col = fig1.COLORS[c]
            a.scatter(d[v], d.wh_day0, s=1.2, color=col, lw=0)
            if len(d) > 3:
                r = stats.linregress(d[v], d.wh_day0)
                xs = np.linspace(d[v].min(), d[v].max(), 50)
                X = d[v].to_numpy()
                se = np.sqrt(np.sum((d.wh_day0 - (r.intercept + r.slope * X)) ** 2) / (len(X) - 2))
                band = stats.t.ppf(0.975, len(X) - 2) * se * np.sqrt(1 / len(X) + (xs - X.mean()) ** 2 / np.sum((X - X.mean()) ** 2))
                yv = r.intercept + r.slope * xs
                a.fill_between(xs, yv - band, yv + band, color=col, alpha=0.15, lw=0)
                a.plot(xs, yv, color=col, lw=0.6, ls="-" if r.pvalue < 0.05 else "--")
                titles.append((f"{c} r={r.rvalue:.2f}, p={r.pvalue:.2g}", col))
        for j, (txt, col) in enumerate(titles):  # planner: correlation annotations as title
            a.text(0.5, 1.03 + 0.11 * (len(titles) - 1 - j), txt, transform=a.transAxes, ha="center", fontsize=3.6, color=col)
        a.set_ylim(-0.05, 1.05)
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if k == 0 else [])
        a.tick_params(axis="x", labelsize=3.8)
        a.xaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(nbins=3))
        a.set_xlabel(RATE_VARS[v], fontsize=4, labelpad=1)
    axes[0].set_ylabel("Whisker P(lick) day 0", labelpad=1, fontsize=5)


def plot_day0_closeup(ax, df):
    """W | A | FA: one dot per mouse, R+ next to R-, black-edged mean +- s.e.m.; Mann-Whitney asterisks."""
    ticks = []
    for i, tt in enumerate(fig1.TRIAL_TYPES):
        vals = {}
        for k, cohort in enumerate(fig1.COHORTS):
            v = df[(df.trial_type == tt) & (df.cohort == cohort)].lick_probability.dropna().to_numpy()
            x = i * 1.5 + k * 0.55
            j = np.random.default_rng(10 * i + k).uniform(-0.09, 0.09, len(v))
            ax.scatter(x + j, v, s=1.5, color=fig1.group_color(tt, cohort), lw=0, zorder=2)
            ax.errorbar(x + 0.2, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color="0.15", marker="o", ms=2,
                        mfc=fig1.group_color(tt, cohort), mec="0.15", mew=0.4, capsize=0, elinewidth=0.6, zorder=3)
            vals[cohort] = v
        top = max(vals["R+"].max(), vals["R-"].max()) + 0.04
        fig1._bracket(ax, i * 1.5, i * 1.5 + 0.55, top, fig1._mw_p(vals["R+"], vals["R-"]))
        ticks.append(i * 1.5 + 0.275)
    ax.set_xticks(ticks, [fig1.TT_SHORT[t] for t in fig1.TRIAL_TYPES])
    ax.set_ylim(-0.03, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("P(lick)", labelpad=1)
