"""Plot helpers of SuppFigure1_behaviour panels d, i, j, k (from the deleted SuppFigure2_learning_correlates)."""

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


def plot_rt_curves(ax, df):
    axes = _split(ax, 3, gap=0.1)
    for a, cu in zip(axes, ["whisker", "false_alarm", "difference"]):
        for i, tt in enumerate(fig1.TRIAL_TYPES):
            for k, c in enumerate(COHORTS):
                v = df[(df.curve == cu) & (df.trial_type == tt) & (df.cohort == c)].rho_vs_shift.dropna()
                x = i + (k - 0.5) * 0.35
                _median_iqr(a, x, v, group_color(tt, c))
                p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
                fig1.sig_text(a, x, np.percentile(v, 75) + 0.02, p, ns=False)
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        a.set_xticks(range(3), ["W", "A", "FA"])
        a.set_title(f"vs {cu.replace('_', '-')} curve", fontsize=5.5, pad=2)
    for a in axes[1:]:
        a.sharey(axes[0])
    for a in axes:  # every axis labelled (user 2026-11-12)
        a.set_ylabel("ρ(RT, curve) − shift", labelpad=1)
    # R+ / R- colour key at the figure level (user 2026-11-16: no 'left / right' legends)


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


BODYPARTS = ("jaw", "nose", "whisker", "pupil")
DLC_NAMES = {"posture": {"jaw": "Jaw opening", "nose": "Nose position", "whisker": "Whisker angle", "pupil": "Pupil area"},
             "movement": {"jaw": "Jaw speed", "nose": "Nose speed", "whisker": "Whisker speed",
                          "pupil": "Pupil area change"}}  # DLC names = body part + quantity (planner, user 2026-11-16)


def _plot_corr_kind(ax, df, kind):
    """User 2026-11-12 / 2026-11-23 (one panel per kind): real rho per mouse (cohort colour, mean +- s.e.m.) next to each
    mouse's linear-shift-null median (grey, hollow), per body part; stars, paired Wilcoxon real vs null (cohort colour)
    and R+ vs R- on real - null (black). Then the cross-correlation with the whisker curve per body part (lags -10..10
    whisker trials, mean +- s.e.m. over mice; no shift-null line, user 2026-11-23)."""
    lag0, lag = df[df.kind == "lag0"], df[df.kind == "lag"]
    ax.axis("off")
    dots = ax.inset_axes([0.09, 0.0, 0.33, 0.78])  # user 2026-11-23: easier to read (room above for the stars)
    for i, bp in enumerate(BODYPARTS):
        m = f"{bp}_pre_{kind}"
        vals = {}
        for k, c in enumerate(COHORTS):
            g = lag0[(lag0.metric == m) & (lag0.cohort == c)]
            x = i + (k - 0.5) * 0.42
            for col, dx, face in (("spearman_null", -0.08, "white"), ("spearman", 0.08, None)):
                v = g[col].dropna()
                clr = "0.6" if face else COLORS[c]
                dots.errorbar(x + dx, v.mean(), v.sem(), color=clr, marker="o", ms=1.8, mfc=face or clr, mew=0.5,
                              lw=0, elinewidth=0.6, capsize=0, zorder=3)
            d = g.spearman_vs_null.dropna()
            p = stats.wilcoxon(d).pvalue if len(d) > 5 else np.nan
            fig1.sig_text(dots, x, 0.6, p, color=COLORS[c], transform=dots.get_xaxis_transform(), rotate_p=True,
                          ns=False)
            vals[c] = d
        p = fig1._mw_p(vals["R+"], vals["R-"])
        fig1.sig_text(dots, i, 0.6, p, transform=dots.get_xaxis_transform(), rotate_p=True, ns=False)
    dots.axhline(0, color="0.6", lw=0.4, ls=":")
    lo_, hi_ = dots.get_ylim()
    dots.set_ylim(lo_, hi_ + 0.9 * (hi_ - lo_))
    dots.set_xticks(range(4), [DLC_NAMES[kind][b] for b in BODYPARTS], fontsize=fig1.FS_TICK, rotation=35, ha="right",
                    rotation_mode="anchor")
    dots.set_xlim(-0.6, 3.6)
    dots.tick_params(labelsize=fig1.FS_TICK)
    dots.yaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(4))
    dots.set_ylabel(f"ρ (pre-stim. {kind},\nW curve), lag 0", labelpad=1, fontsize=fig1.FS_LABEL)
    fig1._legend(dots, [("Real", "0.3", dict(marker="o", ms=2, lw=0)),
                        ("Shift null", "0.6", dict(marker="o", ms=2, lw=0, mfc="white", mew=0.5))], loc="lower left")
    for i, bp in enumerate(BODYPARTS):
        r, c = divmod(i, 2)
        a = ax.inset_axes([0.56 + c * 0.235, 0.6 - r * 0.6, 0.19, 0.34])
        m = f"{bp}_pre_{kind}"
        for coh in COHORTS:
            fig1._mean_band(a, lag[(lag.metric == m) & (lag.cohort == coh)], "lag", "spearman", COLORS[coh], lw=0.6,
                            fill=0.75)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.axhline(0, color="0.8", lw=0.3)
        a.set_xlim(-10, 10)
        a.set_xticks([-10, 0, 10])
        a.tick_params(labelsize=fig1.FS_TICK, pad=1)
        a.yaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(2))
        a.set_title(DLC_NAMES[kind][bp], fontsize=fig1.FS_NOTE, pad=1.5)  # above the axes, not over the data
        if r == 1:
            a.set_xlabel("Lag (W trials)", fontsize=fig1.FS_LABEL, labelpad=0.5)
        else:
            a.set_xticklabels([])
        if c == 0:
            a.set_ylabel("ρ", fontsize=fig1.FS_LABEL, labelpad=0.5)


def plot_posture_curve(ax, df):
    _plot_corr_kind(ax, df, "posture")


def plot_movement_curve(ax, df):
    _plot_corr_kind(ax, df, "movement")
