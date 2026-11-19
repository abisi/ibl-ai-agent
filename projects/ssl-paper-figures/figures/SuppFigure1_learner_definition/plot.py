"""SuppFigure1_learner_definition -- one plot_<key>(ax, df) per panel; style and helpers from the Figure 1 plot module.
Learners solid lines / filled markers, non-learners dashed lines / open markers; cohort and trial-type colours as Figure 1.
Stars and bars: learners versus non-learners within R+ (R- has too few non-learners to test)."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("fig1_plot", Path(__file__).resolve().parents[1] / "Figure1_behaviour" / "plot.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

COHORTS, COLORS, TRIAL_TYPES = fig1.COHORTS, fig1.COLORS, fig1.TRIAL_TYPES
LS = {True: "-", False: "--"}
GROUPS = [(c, lr) for c in COHORTS for lr in (True, False)]  # R+ learners, R+ non-learners, R- learners, R- non-learners
OFFSET = {("R+", True): 0.0, ("R+", False): 0.42, ("R-", True): 1.0, ("R-", False): 1.42}
MIN_SHOW = 3  # mice per group for a mean trace / mean marker


def _learner_entries():
    return [("Learner", "0.2"), ("Non-learner", "0.2", dict(ls="--"))]


def _sel(df, cohort, learner):
    return df[(df.cohort == cohort) & (df.learner.astype(bool) == learner)]


def _trace(a, d, x, value, color, lr):
    """Mean +- s.e.m. over mice; groups with fewer than MIN_SHOW mice (R- non-learners, 2) as thin per-mouse lines."""
    d = d.dropna(subset=[value])
    if d.mouse_id.nunique() >= MIN_SHOW:
        fig1._mean_band(a, d, x, value, color, ls=LS[lr], lw=0.7, fill=0.8)
    else:
        for _, m in d.groupby("mouse_id"):
            m = m.groupby(x)[value].mean()
            a.plot(m.index, m, color=fig1.light(color, 0.35), lw=0.35, ls=LS[lr], zorder=1)


def _four_dots(ax, df, value, cats, cat_col, color_fn, step=2.3):
    """Per category: R+ learners, R+ non-learners, R- learners, R- non-learners; one dot per mouse (non-learners open),
    mean +- s.e.m. (black edge); bracket and stars, Mann-Whitney R+ learners vs non-learners."""
    ticks = []
    for i, cat in enumerate(cats):
        d = df if cat is None else df[df[cat_col] == cat]
        vals = {}
        for k, (c, lr) in enumerate(GROUPS):
            v = _sel(d, c, lr)[value].dropna().to_numpy(float)
            vals[(c, lr)] = v
            if not len(v):
                continue
            x = i * step + OFFSET[(c, lr)]
            col = color_fn(cat, c)
            j = np.random.default_rng(10 * i + k).uniform(-0.07, 0.07, len(v))
            ax.scatter(x + j, v, s=2 if lr else 3, facecolor=col if lr else "none", edgecolor=col, lw=0 if lr else 0.4,
                       zorder=2)
            if len(v) >= MIN_SHOW:
                ax.errorbar(x + 0.17, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color="0.15", marker="o", ms=2,
                            mfc=col if lr else "white", mec="0.15", mew=0.4, capsize=0, elinewidth=0.6, zorder=3)
        a, b = vals[("R+", True)], vals[("R+", False)]
        if len(a) >= 3 and len(b) >= 3:
            top = max(a.max(), b.max())
            span = np.nanmax(df[value]) - np.nanmin(df[value])
            fig1._bracket(ax, i * step, i * step + OFFSET[("R+", False)], top + 0.05 * span, fig1._mw_p(a, b))
        ticks.append(i * step + 0.71)
    return ticks


# ---------------------------------------------------------------- panels

def plot_learners(ax, df):
    """Stacked fraction of mice per sheet category; learners/mice above."""
    df = df.set_index("cohort").loc[COHORTS]
    for i, (c, r) in enumerate(df.iterrows()):
        bottom = 0
        for cat, col in (("good", COLORS[c]), ("moderate", fig1.light(COLORS[c], 0.5)), ("bad", "0.85")):
            f = r[f"n_{cat}"] / r.mice
            ax.bar(i, f, 0.6, bottom=bottom, color=col, lw=0)
            if f > 0.06:
                ax.text(i, bottom + f / 2, int(r[f"n_{cat}"]), ha="center", va="center", fontsize=fig1.FS_NOTE)
            bottom += f
        ax.text(i, 1.02, f"{int(r.learners)}/\n{int(r.mice)}", ha="center", va="bottom", fontsize=fig1.FS_NOTE,
                linespacing=0.9)
    ax.set_xticks([0, 1], [c.replace("-", "−") for c in COHORTS])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("Fraction of mice", labelpad=1)
    fig1._legend(ax, [("Good", "0.3", dict(marker="s", lw=0, ms=2.5)), ("Moderate", "0.6", dict(marker="s", lw=0, ms=2.5)),
                      ("Bad", "0.85", dict(marker="s", lw=0, ms=2.5))], loc="upper left", bbox_to_anchor=(1.0, 1.0))


def plot_ex_rplus_nonlearner(ax, df):
    fig1._example(ax, df, "R+ non-learner", first=True)


def plot_ex_rminus_nonlearner(ax, df):
    fig1._example(ax, df, "R− non-learner")


def plot_day0_closeup(ax, df):
    ticks = _four_dots(ax, df, "lick_probability", TRIAL_TYPES, "trial_type", lambda tt, c: fig1.group_color(tt, c))
    ax.set_xticks(ticks, [fig1.TT_SHORT[t] for t in TRIAL_TYPES])
    ax.set_ylim(-0.03, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("P(lick), day 0", labelpad=1)
    ax.set_xlabel("Trial type (L, NL per cohort)", labelpad=1)


def plot_dprime_day0(ax, df):
    """User 2026-11-19: as D with d' (whisker vs FA, auditory vs FA) per mouse on day 0."""
    cats = ["whisker", "auditory"]
    tt = {"whisker": "whisker_trial", "auditory": "auditory_trial"}
    ticks = _four_dots(ax, df, "dprime", cats, "comparison", lambda c_, c: fig1.group_color(tt[c_], c))
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xticks(ticks, ["W", "A"])
    ax.set_ylabel("d′ vs FA, day 0", labelpad=1)
    ax.set_xlabel("Trial type (L, NL per cohort)", labelpad=1)


def plot_days(ax, df):
    from scipy import stats
    axes = fig1._split(ax, 3, gap=0.13)
    p = df[df.measure == "p_lick"]
    for a, c in zip(axes[:2], COHORTS):
        for tt in TRIAL_TYPES:
            for lr in (True, False):
                g = _sel(p[p.trial_type == tt], c, lr).groupby("day").value
                m, e, n = g.mean(), g.sem(), g.size()
                m, e = m[n >= MIN_SHOW], e[n >= MIN_SHOW]
                if tt == "whisker_trial":
                    m, e = m[m.index >= 0], e[e.index >= 0]
                col = fig1.group_color(tt, c)
                a.errorbar(m.index + (0 if lr else 0.15), m, e, color=col, ls=LS[lr], marker="o", ms=1.6,
                           mfc=col if lr else "white", mew=0.4, lw=0.6, elinewidth=0.5, capsize=0)
        a.axvspan(-0.4, 0.4, color="0.92", lw=0, zorder=0)
        a.set_xticks(range(-2, 4))
        a.set_ylim(0, 1.02)
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if c == "R+" else [])
        a.set_xlabel("Day", labelpad=1)
        a.set_title(c.replace("-", "−"), fontsize=fig1.FS_TITLE, pad=2, color=COLORS[c])
    axes[0].set_ylabel("P(lick)", labelpad=1)
    fig1._legend(axes[0], [(fig1.TT_SHORT[tt], fig1.group_color(tt, "R+")) for tt in TRIAL_TYPES], loc="center left",
                 bbox_to_anchor=(0, 0.6))
    dp, right = df[df.measure == "dprime"], axes[2]
    for c in COHORTS:
        for lr in (True, False):
            g = _sel(dp, c, lr).groupby("day").value
            m, e, n = g.mean(), g.sem(), g.size()
            m, e = m[n >= 2], e[n >= 2]
            right.errorbar(m.index + (0 if lr else 0.15), m, e, color=COLORS[c], ls=LS[lr], marker="o", ms=1.6,
                           mfc=COLORS[c] if lr else "white", mew=0.4, lw=0.6, elinewidth=0.5, capsize=0)
    right.axhline(0, color="0.6", lw=0.4, ls=":")
    top = right.get_ylim()[1]
    for day in range(4):
        a_, b_ = _sel(dp[dp.day == day], "R+", True).value, _sel(dp[dp.day == day], "R+", False).value
        if len(a_) >= 3 and len(b_) >= 3:
            pv = stats.mannwhitneyu(a_, b_).pvalue
            right.text(day, top, fig1.stars(pv), ha="center", va="bottom", fontsize=fig1.FS_NOTE, color=COLORS["R+"])
    right.set_xticks(range(4))
    right.set_xlim(-0.5, 3.5)
    right.set_xlabel("Whisker day", labelpad=1)
    right.set_ylabel("d′ (W vs FA)", labelpad=1)
    fig1._legend(right, _learner_entries(), loc="center right", bbox_to_anchor=(1, 0.45))


def _plot_perf(ax, df, alabel):
    axes = fig1._row_axes(ax, 3, gap_mm=[2.5, 9.0])
    curves = [("p_whisker", "whisker_trial", "Whisker"), ("p_chance", "no_stim_trial", "False alarm"),
              ("p_auditory", "auditory_trial", "Auditory")]
    for a, c in zip(axes[:2], COHORTS):
        for lr in (True, False):
            d = _sel(df, c, lr)
            for col, tt, _ in curves:
                _trace(a, d, "session_time_norm", col, fig1.group_color(tt, c), lr)
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        a.set_ylim(0, 1)
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if c == "R+" else [])
        n = [_sel(df, c, lr).mouse_id.nunique() for lr in (True, False)]
        a.set_title(f"{c.replace('-', '−')} (L {n[0]}, NL {n[1]})", fontsize=fig1.FS_TITLE, pad=2, color=COLORS[c])
    axes[0].set_ylabel("P(lick), learning curve", labelpad=1)
    fig1._legend(axes[0], [(lab, fig1.group_color(tt, "R+")) for _, tt, lab in curves], loc="upper left",
                 bbox_to_anchor=(0, 0.86))
    fig1._legend(axes[1], _learner_entries(), loc="upper left", bbox_to_anchor=(0, 0.86))
    a = axes[2]
    for c in COHORTS:
        for lr in (True, False):
            _trace(a, _sel(df, c, lr), "session_time_norm", "performance", COLORS[c], lr)
    on = df[(df.cohort == "R+") & df.cluster_sig.astype(bool)]
    if len(on):
        fig1._shaded_bars(a, on, "session_time_norm", 1.0, 0.02, label="R+ ", lw=1)
    a.axhline(0, color="0.6", lw=0.4, ls=":")
    a.set_xlim(0, 1)
    a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    a.set_ylim(-0.1, 0.45)
    a.set_yticks([0, 0.2, 0.4])
    a.set_ylabel("Δ P(lick) (W − FA)", labelpad=1)
    axes[1].set_xlabel(f"Session time from {alabel} (0) to session end (1)", labelpad=1)


def plot_perf_trial(ax, df):
    _plot_perf(ax, df, "1st W trial")


def plot_perf_hit(ax, df):
    _plot_perf(ax, df, "1st W hit")


def plot_cumulative_fa(ax, df):
    axes = fig1._split(ax, 2, gap=0.15)
    for a, (day, title) in zip(axes, ((0, "Day 0"), (-1, "Day −1"))):
        for c in COHORTS:
            for lr in (True, False):
                _trace(a, _sel(df[df.day == day], c, lr), "trial_after_anchor", "cumulative_fa", COLORS[c], lr)
        a.set_xlim(1, df.trial_after_anchor.max())
        a.set_title(title, fontsize=fig1.FS_TITLE, pad=2)
        a.set_xlabel("Trials after anchor", labelpad=1)
    hi = max(a.get_ylim()[1] for a in axes)
    for k, a in enumerate(axes):
        a.set_ylim(0, hi)
        if k:
            a.set_yticklabels([])
    axes[0].set_ylabel("Cumulative FA", labelpad=1)
    fig1._legend(axes[0], _learner_entries(), loc="upper left")


def plot_rt(ax, df):
    tts = ["whisker_trial", "auditory_trial"]
    ticks = _four_dots(ax, df, "median_reaction_time_s", tts, "trial_type", lambda tt, c: fig1.group_color(tt, c))
    ax.set_xticks(ticks, [fig1.TT_SHORT[t] for t in tts])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Median RT (s)", labelpad=1)
    ax.set_xlabel("Hits (L, NL per cohort)", labelpad=1)


DPRIME_ALIGN = [("first_whisker_trial", "1st W trial"), ("first_whisker_hit", "1st W hit")]


def _plot_dprime(ax, df, xlabel_fmt, norm):
    axes = fig1._split(ax, 2, gap=0.22)
    for a, (align, lab) in zip(axes, DPRIME_ALIGN):
        d = df[df.alignment == align]
        for c in COHORTS:
            for lr in (True, False):
                g = _sel(d, c, lr)
                n = g.groupby("time").mouse_id.nunique()
                if not norm and g.mouse_id.nunique() >= MIN_SHOW:  # minutes: while >= 5 mice remain
                    g = g[g.time.isin(n[n >= 5].index)]
                _trace(a, g, "time", "dprime", COLORS[c], lr)
        on = d[(d.cohort == "R+") & d.cluster_sig.astype(bool)]
        if len(on):
            fig1._shaded_bars(a, on, "time", 1.0, 0.02 if norm else 1, label="R+ ", lw=1)
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        if norm:
            a.set_xlim(0, 1)
            a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        else:
            a.set_xlim(0, d[(d.cohort == "R+") & d.in_test.astype(bool)].time.max())
        a.set_xlabel(xlabel_fmt.format(lab), labelpad=1)
        a.set_ylabel("d′ (W vs FA curve)", labelpad=1)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    fig1._legend(axes[0], _learner_entries(), loc="upper left")


def plot_dprime_norm(ax, df):
    _plot_dprime(ax, df, "Session time from {} (norm.)", True)


def plot_dprime_abs(ax, df):
    _plot_dprime(ax, df, "Time from {} (min)", False)


def plot_pretraining(ax, df):
    ticks = _four_dots(ax, df, "n_pretraining_trials", [None], None,
                       lambda _, c: fig1.COLORS["auditory"] if c == "R+" else fig1.light(fig1.COLORS["auditory"], 0.5))
    ax.set_xticks([OFFSET[g] for g in GROUPS], ["L", "NL", "L", "NL"])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Pretraining trials", labelpad=1)
    _cohort_labels(ax)


def plot_first_lick(ax, df):
    _four_dots(ax, df, "whisker_trials_before_first_lick", [None], None, lambda _, c: COLORS[c])
    ax.set_xticks([OFFSET[g] for g in GROUPS], ["L", "NL", "L", "NL"])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("W trials before 1st lick", labelpad=1)
    _cohort_labels(ax)


def _cohort_labels(ax):
    for c, x in (("R+", 0.21), ("R-", 1.21)):
        ax.text(x, -0.2, c.replace("-", "−"), transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=fig1.FS_TICK, color=COLORS[c])
