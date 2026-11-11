"""SuppFigure2_learning_correlates -- one plot_<key>(ax, df) per panel; Figure 1 style and helpers."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy import stats

_spec = importlib.util.spec_from_file_location("fig1_plot", Path(__file__).resolve().parents[1] / "Figure1_behaviour" / "plot.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)
COHORTS, COLORS, group_color, stars, _split = fig1.COHORTS, fig1.COLORS, fig1.group_color, fig1.stars, fig1._split

plot_first_lick = fig1.plot_first_lick


def _median_iqr(a, x, v, color, marker="o"):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    m, q1, q3 = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
    a.errorbar(x, m, [[m - q1], [q3 - m]], color=color, marker=marker, ms=2.2, lw=0, elinewidth=0.6, capsize=0)


def _hit_axis(a, df):
    nb = int(df.hit_bin.max()) + 1 if len(df) else 6
    a.set_xticks(range(nb), [f"{5 * b + 1}-{5 * b + 5}" for b in range(nb)], rotation=45, fontsize=4.2)
    a.set_xlim(-0.6, nb - 0.4)


def plot_rt_hits_other(ax, df):
    axes = _split(ax, 3, gap=0.18)
    for a, (tt, lat, title) in zip(axes, [("auditory_trial", "rt_ms", "Auditory hits, RT"),
                                          ("no_stim_trial", "rt_ms", "False alarms, RT"),
                                          ("whisker_trial", "jaw_ms", "Whisker hits, jaw onset")]):
        d = df[(df.trial_type == tt) & (df.latency == lat)]
        for k, c in enumerate(COHORTS):
            for b, g in d[d.cohort == c].groupby("hit_bin"):
                _median_iqr(a, b + (k - 0.5) * 0.25, g.value, group_color(tt, c))
        _hit_axis(a, d)
        a.set_title(title, fontsize=5.5, pad=2)
        a.set_xlabel("Hit", labelpad=0, fontsize=5)
    axes[0].set_ylabel("Latency (ms)", labelpad=1)


def plot_lick_course(ax, df):
    axes = _split(ax, 2, gap=0.15)
    for a, tt in zip(axes, ["whisker_trial", "auditory_trial"]):
        for c in COHORTS:
            g = df[(df.trial_type == tt) & (df.cohort == c)].groupby("time_s").p_lick
            a.plot(g.mean().index, g.mean(), color=group_color(tt, c), lw=0.8)
            a.fill_between(g.mean().index, g.mean() - g.sem(), g.mean() + g.sem(), color=group_color(tt, c), alpha=0.25, lw=0)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.set_xlim(-0.2, 1.5)
        a.set_ylim(-0.01, 0.5)
        a.set_title("Whisker" if tt == "whisker_trial" else "Auditory", fontsize=5.5, pad=2)
        a.set_xlabel("Time (s)", labelpad=0)
    axes[0].set_ylabel("P(lick), lick trials", labelpad=1)
    axes[1].set_yticklabels([])


def plot_rt_curves(ax, df):
    axes = _split(ax, 3, gap=0.1)
    for a, cu in zip(axes, ["whisker", "false_alarm", "difference"]):
        for i, tt in enumerate(fig1.TRIAL_TYPES):
            for k, c in enumerate(COHORTS):
                v = df[(df.curve == cu) & (df.trial_type == tt) & (df.cohort == c)].rho_vs_shift.dropna()
                x = i + (k - 0.5) * 0.35
                _median_iqr(a, x, v, group_color(tt, c))
                p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
                if p < 0.05:
                    a.text(x, np.percentile(v, 75) + 0.02, stars(p), ha="center", va="bottom", fontsize=4.2)
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        a.set_xticks(range(3), ["W", "A", "FA"])
        a.set_title(f"vs {cu.replace('_', '-')} curve", fontsize=5.5, pad=2)
    axes[0].set_ylabel("ρ(RT, curve) − shift", labelpad=1)
    for a in axes[1:]:
        a.sharey(axes[0])
        a.tick_params(labelleft=False)


def plot_bouts(ax, df):
    left, right = _split(ax, 2, gap=0.3)
    A = {"R+": COLORS["auditory"] if "auditory" in COLORS else "#2c2cdb", "R-": "#9a9aee"}
    d = df[df.measure == "n_licks"]
    for j, tt in enumerate(["whisker_trial", "auditory_trial"]):
        for k, c in enumerate(COHORTS):
            for b, g in d[(d.trial_type == tt) & (d.cohort == c)].groupby("hit_bin"):
                _median_iqr(left, b + (2 * j + k - 1.5) * 0.15, g.value, group_color(tt, c), marker="o" if j == 0 else "s")
    _hit_axis(left, d)
    left.set_ylabel("Licks per bout", labelpad=1)
    left.set_xlabel("Hit (W circles, A squares)", labelpad=0, fontsize=5)
    e = df[df.measure == "expectation_index"]
    for k, c in enumerate(COHORTS):
        for b, g in e[e.cohort == c].groupby("hit_bin"):
            _median_iqr(right, b + (k - 0.5) * 0.25, g.value, COLORS[c])
    right.axhline(1, color="0.6", lw=0.4, ls=":")
    _hit_axis(right, e)
    right.set_ylabel("Late licks, W / A hits", labelpad=1)
    right.set_xlabel("Whisker hit", labelpad=0, fontsize=5)


def plot_posture_curve(ax, df):
    metrics = [f"{bp}_pre_{k}" for bp in ("jaw", "nose", "whisker", "pupil") for k in ("posture", "movement")]
    labels = [f"{m.split('_')[0].capitalize()} {'posture' if m.endswith('posture') else 'movement'}" for m in metrics]
    for i, m in enumerate(metrics):
        vals = {}
        for k, c in enumerate(COHORTS):
            v = df[(df.metric == m) & (df.cohort == c)].spearman_vs_null.dropna()
            x = i + (k - 0.5) * 0.3
            _median_iqr(ax, x, v, COLORS[c])
            p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
            if p < 0.05:
                ax.text(x, np.percentile(v, 75) + 0.015, stars(p), ha="center", va="bottom", fontsize=4.5, color=COLORS[c])
            vals[c] = v
        p = fig1._mw_p(vals["R+"], vals["R-"])
        if p < 0.05:
            ax.text(i, 1.0, stars(p), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xticks(range(len(metrics)), labels, fontsize=4.5, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(metrics) - 0.4)
    ax.set_ylabel("ρ(pre-stim., W curve) − shift", labelpad=1)
