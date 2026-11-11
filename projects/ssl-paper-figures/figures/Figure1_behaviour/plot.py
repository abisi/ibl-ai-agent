"""Figure1_behaviour -- one plot_<key>(ax, df) per panel (key = the panel's stable name in spec.yaml); each draws only
from its panel's CSV.

Style (planner notes 2026-11-05): group mean +- s.e.m. in cohort colours; individual mice (grey) only where the planner
asks for them; whisker in cohort colours, auditory blue (lighter for R-), false alarms black (R+) and grey (R-).
On-panel asterisks are Mann-Whitney R+ vs R- (both tests in the caption); bars above curves mark cluster-permutation
differences between cohorts (column cluster_sig).
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paperstyle import COLORS, TRIAL_TYPE_COLORS, light  # noqa: E402

COHORTS = ["R+", "R-"]
TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
TT_SHORT = {"whisker_trial": "W", "auditory_trial": "A", "no_stim_trial": "FA"}
INDIVIDUAL = "0.72"  # grey for individual mice
BODYPART = {"jaw_motion": "#8c510a", "nose_motion": "#01665e", "whisker_motion": "#5e3c99", "pupil_area": "#4d4d4d"}
MOVEMENT_LABELS = {"jaw_motion": "Jaw", "nose_motion": "Nose", "whisker_motion": "Whisker", "pupil_area": "Pupil"}


def group_color(trial_type, cohort):
    """Whisker: cohort colour; auditory: blue (lighter for R-); false alarm: black (R+), grey (R-)."""
    if trial_type == "auditory_trial":
        return COLORS["auditory"] if cohort == "R+" else light(COLORS["auditory"], 0.5)
    if trial_type == "no_stim_trial":
        return COLORS["no_stim"] if cohort == "R+" else "0.6"
    return COLORS[cohort]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _mw_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan


def _bracket(ax, x0, x1, y, p):
    if not np.isfinite(p):
        return
    ax.plot([x0, x1], [y, y], color="0.2", lw=0.4)
    ax.text((x0 + x1) / 2, y, stars(p), ha="center", va="bottom", fontsize=5 if p < 0.05 else 4.5)


def _bars(ax, x, v, color, width=0.6, individuals=False, seed=0, highlight=None):
    """Bar = mean, error bar = s.e.m.; optional grey individual mice (jittered), optional circled mice (highlight:
    boolean array aligned with v)."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return
    ax.bar(x, v[ok].mean(), width, color=light(color, 0.35) if individuals else color, lw=0, zorder=1)
    ax.errorbar(x, v[ok].mean(), stats.sem(v[ok]) if ok.sum() > 1 else 0, color="0.15" if individuals else "0.3",
                capsize=0, elinewidth=0.7, zorder=3)
    if individuals:
        jitter = np.random.default_rng(seed).uniform(-width / 3, width / 3, len(v))
        ax.scatter(x + jitter[ok], v[ok], s=1.5, color=INDIVIDUAL, lw=0, zorder=2)
        if highlight is not None:
            h = np.asarray(highlight, bool) & ok
            ax.scatter(x + jitter[h], v[h], s=14, facecolor="none", edgecolor="k", lw=0.6, zorder=4)


def _two_bars(ax, df, value, ylabel, colors=None, individuals=True, highlight_col=None):
    colors = colors or [COLORS[c] for c in COHORTS]
    vals = []
    for i, (cohort, color) in enumerate(zip(COHORTS, colors)):
        d = df[df.cohort == cohort]
        _bars(ax, i, d[value], color, individuals=individuals, seed=i,
              highlight=d[highlight_col] if highlight_col and highlight_col in d else None)
        vals.append(d[value].dropna())
    top = max(v.max() for v in vals) if individuals else max(v.mean() + v.sem() for v in vals)
    _bracket(ax, 0, 1, top * 1.06 if top > 0 else top + 0.05, _mw_p(*vals))
    ax.set_xticks([0, 1], COHORTS)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel, labelpad=1)


def _dots_individual(ax, x, v, color, seed=0, width=0.04, ms=2, highlight=None):
    """Distribution: grey individual mice and the group mean +- s.e.m. (colour); highlight: boolean array aligned
    with v, circled."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return
    jitter = np.random.default_rng(seed).uniform(-width, width, len(v))
    ax.scatter(x - 0.12 + jitter[ok], v[ok], s=1.5, color=INDIVIDUAL, lw=0, zorder=1)
    if highlight is not None:
        h = np.asarray(highlight, bool) & ok
        ax.scatter(x - 0.12 + jitter[h], v[h], s=12, facecolor="none", edgecolor="k", lw=0.5, zorder=4)
    ax.errorbar(x + 0.1, v[ok].mean(), stats.sem(v[ok]) if ok.sum() > 1 else 0, color=color, marker="o", ms=ms,
                capsize=0, elinewidth=0.7, zorder=3)


def _two_dots(ax, df, value, ylabel, colors=None, highlight_col=None):
    """Dot plot (planner default): per cohort, grey mice and the mean +- s.e.m.; Mann-Whitney asterisks."""
    colors = colors or [COLORS[c] for c in COHORTS]
    vals = []
    for i, (cohort, color) in enumerate(zip(COHORTS, colors)):
        d = df[df.cohort == cohort]
        _dots_individual(ax, i, d[value], color, seed=i,
                         highlight=d[highlight_col] if highlight_col and highlight_col in d else None)
        vals.append(d[value].dropna())
    top = max(v.max() for v in vals)
    _bracket(ax, 0, 1, top * 1.06 if top > 0 else top + 0.05, _mw_p(*vals))
    ax.set_xticks([0, 1], COHORTS)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel, labelpad=1)


def _subgrid(ax, nrows, ncols, wgap=0.25, hgap=0.35):
    """Replace one box's axes by an nrows x ncols grid of axes (row-major list of rows)."""
    ax.axis("off")
    w = (1 - wgap * (ncols - 1) / ncols) / ncols
    h = (1 - hgap * (nrows - 1) / nrows) / nrows
    return [[ax.inset_axes([c * (w + wgap / ncols), 1 - (r + 1) * h - r * hgap / nrows, w, h]) for c in range(ncols)]
            for r in range(nrows)]


def _mean_band(ax, df, x, value, color, ls="-", lw=0.8, fill=0.7):
    g = df.groupby(x)[value]
    m, s = g.mean(), g.sem()
    ax.fill_between(m.index, m - s, m + s, color=light(color, fill), lw=0)
    ax.plot(m.index, m, color=color, lw=lw, ls=ls)


def _sig_bar(ax, df, x):
    """Black bar at the top of the axes over x values inside a significant cluster."""
    if "cluster_sig" not in df:
        return
    xs = np.sort(df[df.cluster_sig.astype(bool)][x].unique())
    if not len(xs):
        return
    step = np.min(np.diff(np.sort(df[x].unique()))) if df[x].nunique() > 1 else 1
    for x0 in xs:
        ax.plot([x0 - step / 2, x0 + step / 2], [0.97, 0.97], transform=ax.get_xaxis_transform(), color="k", lw=1.2,
                solid_capstyle="butt")


def _split(ax, n=2, gap=0.3):
    """Replace one box's axes by n side-by-side axes."""
    ax.axis("off")
    w = (1 - gap * (n - 1)) / n
    return [ax.inset_axes([k * (w + gap), 0, w, 1]) for k in range(n)]


# ---------------------------------------------------------------- task and performance

def plot_pretraining(ax, df):
    _two_bars(ax, df, "n_pretraining_trials", "Pretraining trials",
              colors=[COLORS["auditory"], light(COLORS["auditory"], 0.5)])
    ax.set_ylim(bottom=0)


DAYS = ["-2", "-1", "0", ">0"]


CONTEXT_GREY = {"auditory_trial": ("0.45", "0.7"), "no_stim_trial": ("0.15", "0.55")}  # R+, R-


def plot_days(ax, df):
    """Planner 2026-11-09 (options 2 + 3 of explore_panel_b): left, lick probability across days with whisker in cohort
    colours and auditory / false alarm in grey, day 0 shaded; right, R+ minus R- per day and trial type, mean
    difference +- s.e.m. (sqrt of the summed squared s.e.m. of the two cohorts over mice), no CIs."""
    left, right = _split(ax, 2, gap=0.3)
    x = np.arange(len(DAYS))
    for tt in TRIAL_TYPES:
        for k, cohort in enumerate(COHORTS):
            g = df[(df.trial_type == tt) & (df.cohort == cohort)].groupby("day_group").lick_probability
            m, s = g.mean().reindex(DAYS), g.sem().reindex(DAYS)
            w = tt == "whisker_trial"
            left.errorbar(x + (k - 0.5) * 0.08, m, s, color=group_color(tt, cohort) if w else CONTEXT_GREY[tt][k],
                          marker="o", ms=1.8 if w else 1.2, lw=0.9 if w else 0.5, elinewidth=0.5, capsize=0,
                          zorder=3 if w else 2)
    for i, tt in enumerate(TRIAL_TYPES):
        for j, dg in enumerate(DAYS):
            a = df[(df.trial_type == tt) & (df.cohort == "R+") & (df.day_group == dg)].lick_probability.dropna()
            b = df[(df.trial_type == tt) & (df.cohort == "R-") & (df.day_group == dg)].lick_probability.dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            d, e = a.mean() - b.mean(), np.hypot(a.sem(), b.sem())
            xi = j + (i - 1) * 0.2
            right.errorbar(xi, d, e, color=group_color(tt, "R+"), marker="o", ms=1.8, elinewidth=0.6, capsize=0)
            p = _mw_p(a, b)
            if p < 0.05:
                right.text(xi, d + e + 0.015, stars(p), ha="center", va="bottom", fontsize=4, rotation=90)
    right.axhline(0, color="0.5", lw=0.4, ls=":")
    for a in (left, right):
        a.axvspan(1.6, 2.4, color="0.92", lw=0, zorder=0)
        a.set_xticks(x, ["-2", "-1", "0", "1-3"])
        a.tick_params(axis="x", labelsize=4.5)
        a.set_xlim(-0.4, len(DAYS) - 0.6)
        a.set_xlabel("Day", labelpad=1)
    left.set_ylim(-0.03, 1.03)
    left.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    left.set_ylabel("Lick probability", labelpad=1)
    right.set_ylabel("R+ − R-", labelpad=1)
    for tt, y, yl in zip(TRIAL_TYPES, (0.98, 0.88, 0.78), (0.78, 0.68, 0.58)):  # left: in the empty band below A
        left.text(0.03, yl, TT_SHORT[tt], transform=left.transAxes, fontsize=4.5, va="top",
                  color=group_color(tt, "R+") if tt == "whisker_trial" else CONTEXT_GREY[tt][0])
        right.text(0.03, y, TT_SHORT[tt], transform=right.transAxes, fontsize=4.5, va="top", color=group_color(tt, "R+"))


def plot_rt(ax, df):
    for i, tt in enumerate(TRIAL_TYPES):
        vals = {}
        for k, cohort in enumerate(COHORTS):
            v = df[(df.trial_type == tt) & (df.cohort == cohort)].median_reaction_time_s.dropna()
            _bars(ax, i * 1.1 + (k - 0.5) * 0.42, v, group_color(tt, cohort), width=0.38)  # planner: barplot
            vals[cohort] = v
        top = max(v.mean() + v.sem() for v in vals.values())
        _bracket(ax, i * 1.1 - 0.21, i * 1.1 + 0.21, top * 1.08, _mw_p(vals["R+"], vals["R-"]))
    ax.set_xticks([0, 1.1, 2.2], ["W", "A", "FA"])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean of median RT (s)", labelpad=1)


# ---------------------------------------------------------------- examples

def _example(ax, df, title, first=False):
    cohort = df.cohort.iloc[0]
    color = COLORS[cohort]
    x = df.whisker_trial.to_numpy()  # no performance-state bands (planner 2026-11-09)
    ax.fill_between(x, df.p_low, df.p_high, color=light(color, 0.8), lw=0, zorder=1)  # planner: more transparent CI
    ax.plot(x, df.p_mean, color=color, lw=0.8, zorder=3)
    ax.plot(x, df.p_chance, color=COLORS["no_stim"], lw=0.6, zorder=2)
    if df.p_auditory.notna().any():
        ax.plot(x, df.p_auditory, color=TRIAL_TYPE_COLORS["auditory_trial"], lw=0.6, zorder=2)
    lick = df.lick.to_numpy() == 1
    ax.scatter(x[lick], np.full(lick.sum(), 1.06), s=2, marker="|", lw=0.4, color=COLORS["R+"], zorder=4)
    ax.scatter(x[~lick], np.full((~lick).sum(), -0.06), s=2, marker="|", lw=0.4, color=COLORS["R-"], zorder=4)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, x.max() + 1)
    ax.set_xlabel("Whisker trial", labelpad=1)
    if first:  # the four examples share the y axis; label it once
        ax.set_ylabel("Lick probability", labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{title} ({df.mouse_id.iloc[0]})", fontsize=5.5, pad=2)
    ax.tick_params(length=2.5, width=0.6, labelsize=5.5)  # planner: ticks more visible


def plot_ex_rplus_learner(ax, df):
    _example(ax, df, "R+ learner", first=True)


def plot_ex_rplus_nonlearner(ax, df):
    _example(ax, df, "R+ non-learner")


def plot_ex_rminus_learner(ax, df):
    _example(ax, df, "R- learner")


def plot_ex_rminus_nonlearner(ax, df):
    _example(ax, df, "R- non-learner")


# ---------------------------------------------------------------- first lick and false alarms

def plot_rt_change(ax, df, xlabel, fit=None, display_bin=5):
    """Reaction-time change from the mouse's first 5 hits (kind 'bin' rows of compute.rt_change_from_start, one per
    mouse and hit): for display, per-mouse mean over bins of `display_bin` hits, then mean +- s.e.m. over mice, cohorts
    side by side (group_color); coloured bars, hits in a cohort-vs-0 cluster (sig_vs0); black bar, hits in an R+ vs R-
    cluster (sig_cohorts; clusters are tested hit by hit). `fit` (kind 'fit' rows): mean of the per-mouse hit-by-hit
    least-squares fits, dashed line over hits 1-30."""
    tt = df.trial_type.iloc[0]
    step = float(np.min(np.diff(np.sort(df.hit_bin_centre.unique()))))
    shown = df.assign(x=((df.hit_bin_centre - 1) // display_bin) * display_bin + (display_bin + 1) / 2)
    shown = shown.groupby(["mouse_id", "cohort", "x"]).value.mean().reset_index()
    for k, cohort in enumerate(COHORTS):
        g = shown[shown.cohort == cohort].groupby("x").value
        ax.errorbar(g.mean().index + (k - 0.5) * 0.8, g.mean(), g.sem(), color=group_color(tt, cohort), marker="o",
                    ms=1.8, lw=0.7, elinewidth=0.6, capsize=0)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    if fit is not None and len(fit):
        x = np.array([1, df.hit_bin_centre.max()])
        for r in fit.itertuples():
            ax.plot(x, r.intercept_ms + r.slope_ms_per_10_hits * (x - 1) / 10, color=group_color(tt, r.cohort), lw=0.6,
                    ls="--", zorder=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))  # headroom for the cluster bars
    bars = [(df[df.sig_cohorts.astype(bool)], "k", 0.98)]
    bars += [(df[(df.cohort == c) & df.sig_vs0.astype(bool)], group_color(tt, c), 0.93 - 0.05 * k)
             for k, c in enumerate(COHORTS)]
    for d, color, y in bars:
        for x0 in np.sort(d.hit_bin_centre.unique()):
            ax.plot([x0 - step / 2, x0 + step / 2], [y, y], transform=ax.get_xaxis_transform(), color=color, lw=1,
                    solid_capstyle="butt")
    ax.set_xlim(0, df.hit_bin_centre.max() + 1)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel("Δ RT from first 5 hits (ms)", labelpad=1)


def plot_rt_slopes(ax, mice, tt="whisker_trial"):
    """Per-mouse hit-by-hit RT slope (ms per 10 hits; kind 'mouse' rows): grey mice, cohort mean +- s.e.m.; stars
    under each cohort, Wilcoxon vs 0; bracket, R+ vs R- Mann-Whitney (user 2026-11-10)."""
    _two_dots(ax, mice, "slope_ms_per_10_hits", "RT slope (ms / 10 hits)",
              colors=[group_color(tt, c) for c in COHORTS])
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    lo = ax.get_ylim()[0]
    for i, c in enumerate(COHORTS):
        v = mice[mice.cohort == c].slope_ms_per_10_hits.dropna()
        ax.text(i, lo, stars(stats.wilcoxon(v).pvalue), ha="center", va="bottom", fontsize=4.5,
                color=group_color(tt, c))
    ax.set_xticks([0, 1], [c.replace("-", "−") for c in COHORTS])


def plot_rt_learning(ax, df):
    """Left: whisker-hit reaction time change from the mouse's first 5 hits (plot_rt_change, shown in bins of 5 hits).
    Middle: per-mouse hit-by-hit slopes (plot_rt_slopes; user 2026-11-10). Right: Spearman rho (reaction time vs
    whisker curve) minus the shift-null median, per licked trial type; mean +- s.e.m. over mice; stars, Wilcoxon vs 0."""
    left, mid, right = _split(ax, 3, gap=0.22)
    plot_rt_change(left, df[df.kind == "bin"], "Whisker hit", df[df.kind == "fit"])
    plot_rt_slopes(mid, df[df.kind == "mouse"])
    rho = df[df.kind == "rho"]
    for i, tt in enumerate(TRIAL_TYPES):
        for k, cohort in enumerate(COHORTS):
            v = rho[(rho.trial_type == tt) & (rho.cohort == cohort)].value.dropna()
            x = i + (k - 0.5) * 0.35
            right.errorbar(x, v.mean(), v.sem(), color=group_color(tt, cohort), marker="o", ms=2.2, lw=0, elinewidth=0.6,
                           capsize=0)
            p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
            if p < 0.05:
                right.text(x, v.mean() + v.sem() + 0.02, stars(p), ha="center", va="bottom", fontsize=4.5)
    right.axhline(0, color="0.6", lw=0.4, ls=":")
    right.set_xticks(range(3), ["W", "A", "FA"])
    right.set_xlim(-0.6, 2.6)
    right.set_ylabel("ρ(RT, W curve) − shift", labelpad=1)


def plot_first_lick(ax, df):
    _two_bars(ax, df, "whisker_trials_before_first_lick", "W trials before 1st lick")  # planner: barplot
    ax.set_ylim(bottom=0)


def plot_aligned_curves(ax, df):
    """Whisker | false-alarm | auditory curves (lines, mean +- s.e.m.) and data in bins of 5 trials (open circles; per
    mouse lick fraction in the bin, then mean +- s.e.m.; planner 2026-11-09), both cohorts; each trial type on its own
    trials, all aligned at the first whisker hit (whisker from trial 0 only; bins -5..-1, 0..4, 5..9, ...)."""
    x = "trial_from_first_whisker_hit"
    axes = _split(ax, 3, gap=0.12)
    for k, (a, tt) in enumerate(zip(axes, ["whisker_trial", "no_stim_trial", "auditory_trial"])):
        cur, dat = df[(df.kind == "curve") & (df.trial_type == tt)], df[(df.kind == "data") & (df.trial_type == tt)]
        for cohort in COHORTS:
            color = group_color(tt, cohort)
            _mean_band(a, cur[cur.cohort == cohort], x, "p_lick", color)
            d = dat[dat.cohort == cohort].assign(bin=lambda d_: np.floor(d_[x] / 5) * 5 + 2)  # bin centre
            g = d.groupby(["mouse_id", "bin"]).p_lick.mean().groupby("bin")
            a.errorbar(g.mean().index + (0.4 if cohort == "R-" else -0.4), g.mean(), g.sem(), color=color, marker="o",
                       ms=1.6, lw=0, elinewidth=0.4, capsize=0, mfc="white", mew=0.4)
        _sig_bar(a, cur, x)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.set_xlim(-5.5, cur[x].max() + 0.5)  # planner: -5 to +30 trials, same axis for the three types
        a.set_ylim(-0.02, 1.02)
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"] if k == 0 else [])
        a.set_title({"whisker_trial": "Whisker", "no_stim_trial": "False alarm", "auditory_trial": "Auditory"}[tt],
                    fontsize=5.5, pad=2)
        a.set_xlabel({"whisker_trial": "W trials", "no_stim_trial": "No-stim trials", "auditory_trial": "A trials"}[tt]
                     + " from 1st W hit", labelpad=1, fontsize=5)
    axes[0].set_ylabel("P(lick)", labelpad=1)
    axes[2].legend(handles=[mpl.lines.Line2D([], [], color="0.3", lw=0.8, label="Curve"),
                            mpl.lines.Line2D([], [], color="0.3", marker="o", mfc="white", ms=2, lw=0, label="Data")],
                   loc="lower right", fontsize=4.5, frameon=False, handlelength=1, borderaxespad=0)


def plot_first_hit_licks(ax, df):
    """Whisker | spontaneous licks | auditory in 1-min bins from the first whisker hit (user 2026-11-10; mean +- s.e.m.
    over mice, group_color of the trial type; spontaneous licks in FA colours); grey band, the pre window; black bar, bins
    where R+ and R- differ (BH-corrected Mann-Whitney, column cohort_sig)."""
    axes = _split(ax, 3, gap=0.2)
    spec = {"whisker": ("whisker_trial", "Whisker", "P(lick), W trials", "upper left"),
            "spont_licks": ("no_stim_trial", "Spontaneous licks", "Spont. licks / min", "lower left"),
            "auditory": ("auditory_trial", "Auditory", "P(lick), A trials", "lower right")}
    for a, (sub, (tt, title, ylabel, loc)) in zip(axes, spec.items()):
        d = df[df.sub_panel == sub].assign(t=lambda x: x.minute + 0.5)
        a.axvspan(-5, 0, color="0.93", lw=0, zorder=0)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        for cohort in COHORTS:
            _mean_band(a, d[d.cohort == cohort], "t", "value", group_color(tt, cohort))
        for m in np.sort(d[d.cohort_sig.astype(bool)].minute.unique()):
            a.plot([m, m + 1], [1.02, 1.02], transform=a.get_xaxis_transform(), color="k", lw=1.2, solid_capstyle="butt",
                   clip_on=False)
        a.set_xlim(d.minute.min(), d.minute.max() + 1)
        if sub == "spont_licks":
            a.set_ylim(bottom=0)
        else:
            a.set_ylim(-0.02, 1.04)  # headroom so the bands are not clipped at 0 and 1
            a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        a.set_xticks([-5, 0, 5, 10, 15, 20])
        a.set_title(title, fontsize=5.5, pad=7)  # above the significance bars (y = 1.02)
        a.set_ylabel(ylabel, labelpad=1)
        a.legend(handles=[mpl.lines.Line2D([], [], color=group_color(tt, c), lw=0.8, label=c) for c in COHORTS],
                 loc=loc, fontsize=4.5, frameon=False, handlelength=1, borderaxespad=0.2)
    axes[1].set_xlabel("Time from 1st W hit (min)", labelpad=1)


def plot_lick_course(ax, df):
    """Lick trials (user 2026-11-10): left, DLC jaw opening minus each trial's pre-stimulus mean (mm); right,
    P(piezo lick in the 30-ms bin); whisker and auditory on the same axes, both cohorts (group_color), mean +- s.e.m.
    over mice."""
    axes = _split(ax, 2, gap=0.38)
    spec = [("jaw", "Jaw opening", "Δ jaw opening (mm)", (-0.1, 0.3)),
            ("piezo", "Piezo licks", "P(lick in 30 ms)", (-0.2, 1.5))]
    for a, (measure, title, ylabel, xlim) in zip(axes, spec):
        for tt in ("whisker_trial", "auditory_trial"):
            for c in COHORTS:
                g = df[(df.measure == measure) & (df.trial_type == tt) & (df.cohort == c)].groupby("time_s").value
                a.plot(g.mean().index, g.mean(), color=group_color(tt, c), lw=0.7)
                a.fill_between(g.mean().index, g.mean() - g.sem(), g.mean() + g.sem(), color=group_color(tt, c),
                               alpha=0.25, lw=0)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.set_xlim(*xlim)
        a.set_title(title, fontsize=5.5, pad=2)
        a.set_xlabel("Time (s)", labelpad=0)
        a.set_ylabel(ylabel, labelpad=1)
    axes[1].set_ylim(bottom=-0.005)


def plot_cumulative_fa(ax, df):
    for cohort in COHORTS:
        color = group_color("no_stim_trial", cohort)
        for day, ls in ((0, "-"), (-1, "--")):
            _mean_band(ax, df[(df.cohort == cohort) & (df.day == day)], "trial_after_anchor", "cumulative_fa", color,
                       ls=ls, lw=0.7, fill=0.8)
    ax.set_xlim(1, df.trial_after_anchor.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Trials after anchor", labelpad=1)
    ax.set_ylabel("Cumulative FA", labelpad=1)
    ax.legend(handles=[mpl.lines.Line2D([], [], color="0.2", lw=0.8, label="Day 0"),
                       mpl.lines.Line2D([], [], color="0.2", lw=0.8, ls="--", label="Day -1")],
              loc="upper left", fontsize=4.5, frameon=False, handlelength=1.4, borderaxespad=0)


def plot_perf_session(ax, df):
    """Learners only (planner 2026-11-09), with n per cohort."""
    axes = [ax]
    lo, hi = np.inf, -np.inf
    for k, (a, grp) in enumerate(zip(axes, ["learner"])):
        d = df[df.group == grp]
        for cohort in COHORTS:
            _mean_band(a, d[d.cohort == cohort], "session_time_norm", "performance", COLORS[cohort])
        _sig_bar(a, d, "session_time_norm")
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        n = {c: d[d.cohort == c].mouse_id.nunique() for c in COHORTS}
        a.set_title(f"{grp.capitalize()}s\nR+ n = {n['R+']}, R- n = {n['R-']}", fontsize=5, pad=2, linespacing=1.1)
        y0, y1 = a.get_ylim()
        lo, hi = min(lo, y0), max(hi, y1)
    for k, a in enumerate(axes):
        a.set_ylim(lo, hi)
        if k:
            a.set_yticklabels([])
    axes[0].set_xlabel("Session time (norm.)", labelpad=1)
    axes[0].set_ylabel("Δ P(lick) (W - FA)", labelpad=1)


def plot_xcorr(ax, df):
    """Left: cross-correlation of the whisker and FA curves over lags, mean +- s.e.m.; dashed, shift-null lag-0 mean.
    Right (planner 2026-11-09): lag-0 correlation per cohort, bars mean +- s.e.m., Mann-Whitney asterisks."""
    ax, right = _split(ax, 2, gap=0.3)
    _two_bars(right, df[df.lag == 0], "r", "r at lag 0", individuals=False)
    right.axhline(0, color="0.6", lw=0.4)
    for cohort in COHORTS:
        d = df[df.cohort == cohort]
        _mean_band(ax, d, "lag", "r", COLORS[cohort], lw=0.7)
        ax.axhline(d[d.lag == 0].null_mean_lag0.mean(), color=light(COLORS[cohort], 0.4), lw=0.5, ls="--")
    ax.axvline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xlim(df.lag.min(), df.lag.max())
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("Lag (W trials)", labelpad=1)
    ax.set_ylabel("r (W, FA curves)", labelpad=1)


# ---------------------------------------------------------------- learners and states

def plot_learners(ax, df):
    df = df.set_index("cohort").loc[COHORTS]
    ax.bar([0, 1], df.fraction_learners, color=[COLORS["R+"], COLORS["R-"]], width=0.6, lw=0)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, r.fraction_learners + 0.02, f"{r.learners}/{r.mice}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks([0, 1], COHORTS)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("Fraction learners", labelpad=1)


def _state_pairs(ax, df, value):
    df = df[df.state.isin(["low", "high"])]
    for i, tt in enumerate(TRIAL_TYPES):
        for k, cohort in enumerate(COHORTS):
            g = df[(df.trial_type == tt) & (df.cohort == cohort)].groupby("state")[value]
            m, s = g.mean(), g.sem()
            x = i + np.array([-0.32, -0.12]) + 0.44 * k
            color = group_color(tt, cohort)
            ax.plot(x, [m.get("low", np.nan), m.get("high", np.nan)], color=color, lw=0.5, zorder=1)
            for xi, state in zip(x, ["low", "high"]):
                ax.errorbar(xi, m.get(state, np.nan), s.get(state, np.nan), color=color, marker="o", ms=2,
                            mfc="white" if state == "low" else color, mew=0.5, capsize=0, elinewidth=0.6, lw=0)
    ax.set_xticks([0, 1, 2], ["W", "A", "FA"])
    ax.set_xlim(-0.5, 2.5)


def plot_lick_states(ax, df):
    _state_pairs(ax, df, "lick_probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("Lick probability", labelpad=1)
    handles = [mpl.lines.Line2D([], [], color="0.3", marker="o", mfc="white", ms=2, lw=0, label="Low"),
               mpl.lines.Line2D([], [], color="0.3", marker="o", ms=2, lw=0, label="High")]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.45), fontsize=4.5, handletextpad=0.2,
              borderaxespad=0, frameon=False, ncol=1)


def plot_rt_states(ax, df):
    _state_pairs(ax, df, "median_reaction_time_s")
    ax.set_ylabel("Median RT (s)", labelpad=1)


SIGNALS = [("whisker_speed", "Whisker speed", "Δ deg/s"), ("jaw_opening", "Jaw opening", "Δ mm"),
           ("pupil_area", "Pupil area", "Δ mm²")]


def plot_movement_peth(ax, df):
    """Rows: whisker speed, jaw opening, pupil area; columns: whisker, auditory trials; stimulus-aligned; lick solid,
    no lick dashed (whisker trials; auditory hits only). Bars: R+ vs R- clusters on lick trials."""
    grid = _subgrid(ax, 3, 2, wgap=0.35, hgap=0.55)
    for r, (sig, label, unit) in enumerate(SIGNALS):
        for c, tt in enumerate(["whisker_trial", "auditory_trial"]):
            a = grid[r][c]
            d = df[(df.signal == sig) & (df.trial_type == tt) & (df["align"] == "stimulus")]
            for lick, ls in ((1, "-"), (0, "--")):
                for cohort in COHORTS:
                    dd = d[(d.lick == lick) & (d.cohort == cohort)]
                    if len(dd):
                        _mean_band(a, dd, "time_s", "value_delta", group_color(tt, cohort), ls=ls, lw=0.5,
                                   fill=0.8 if lick else 0.9)
            _sig_bar(a, d[d.lick == 1], "time_s")
            a.axvline(0, color="0.6", lw=0.3, ls=":")
            if len(d):
                a.set_xlim(d.time_s.min(), d.time_s.max())
            a.tick_params(labelsize=3.8, length=1.5, pad=1)
            if r == 0:
                a.set_title("Whisker" if tt == "whisker_trial" else "Auditory", fontsize=4.5, pad=1.5)
            if c == 0:
                a.set_ylabel(f"{label.split()[0]}\n({unit})", labelpad=1, fontsize=4)
            if r == len(SIGNALS) - 1:
                a.set_xlabel("Time (s)", labelpad=0.5, fontsize=4)


def plot_movement_peth_grid(ax, df):
    """Rows: whisker, auditory trials. Columns: each signal aligned at the stimulus (lick solid, no lick dashed) and at
    the lick (lick trials). Cohort x trial-type colours; bars, R+ vs R- clusters (lick trials; no-lick in the note)."""
    grid = _subgrid(ax, 2, 6, wgap=0.5, hgap=0.45)
    for r, tt in enumerate(["whisker_trial", "auditory_trial"]):
        for c, ((sig, label, unit), align) in enumerate([(s, al) for s in SIGNALS for al in ("stimulus", "lick")]):
            a = grid[r][c]
            d = df[(df.signal == sig) & (df.trial_type == tt) & (df["align"] == align)]  # df.align is a method
            for lick, ls in ((1, "-"), (0, "--")):
                for cohort in COHORTS:
                    dd = d[(d.lick == lick) & (d.cohort == cohort)]
                    if len(dd):
                        _mean_band(a, dd, "time_s", "value_delta", group_color(tt, cohort), ls=ls, lw=0.6,
                                   fill=0.8 if lick else 0.9)
            _sig_bar(a, d[d.lick == 1], "time_s")
            a.axvline(0, color="0.6", lw=0.4, ls=":")
            if len(d):
                a.set_xlim(d.time_s.min(), d.time_s.max())
            a.tick_params(labelsize=4.5)
            if r == 0:
                a.set_title(f"{label}\n{'stimulus' if align == 'stimulus' else 'lick'}-aligned", fontsize=5, pad=2)
            if r == 1:
                a.set_xlabel(f"Time from {'stim.' if align == 'stimulus' else 'lick'} (s)", labelpad=1, fontsize=5)
            if align == "stimulus":
                a.set_ylabel(unit, labelpad=1, fontsize=5)
        grid[r][0].text(-0.75, 0.5, "Whisker trials" if tt == "whisker_trial" else "Auditory trials",
                        transform=grid[r][0].transAxes, rotation=90, ha="center", va="center", fontsize=5.5)
    handles = [mpl.lines.Line2D([], [], color=group_color(tt, c), lw=0.8, label=f"{TT_SHORT[tt]} {c}")
               for tt in ("whisker_trial", "auditory_trial") for c in COHORTS]
    handles += [mpl.lines.Line2D([], [], color="0.3", lw=0.8, label="Lick"),
                mpl.lines.Line2D([], [], color="0.3", lw=0.8, ls="--", label="No lick")]
    grid[0][5].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=4, frameon=False,
                      handlelength=1.2, borderaxespad=0, labelspacing=0.2)


# ---------------------------------------------------------------- movements

def plot_movement_example(ax, df):
    """Upper half: whisker and false-alarm learning curves; lower half: pre-stimulus movements (z, smoothed)."""
    top, bottom = ax.inset_axes([0, 0.55, 1, 0.45]), ax.inset_axes([0, 0, 1, 0.5])
    ax.axis("off")
    x = df.whisker_trial.to_numpy()
    top.plot(x, df.p_mean, color=COLORS[df.cohort.iloc[0]], lw=0.8)
    top.plot(x, df.p_chance, color=COLORS["no_stim"], lw=0.6)
    top.set_ylim(-0.03, 1.03)
    top.set_yticks([0, 1], ["0", "1"])
    top.set_ylabel("P(lick)", labelpad=1)
    top.set_xticklabels([])
    cols = ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]
    for k, c in enumerate(cols):
        bottom.plot(x, -1.1 * k + df[f"{c}_z_smooth"].clip(-3, 3).to_numpy() / 4, color=BODYPART[c], lw=0.6)
    bottom.set_yticks([-1.1 * k for k in range(4)], [MOVEMENT_LABELS[c] for c in cols], fontsize=4.5)
    for t, c in zip(bottom.get_yticklabels(), cols):
        t.set_color(BODYPART[c])
    bottom.tick_params(axis="y", length=0, pad=1)
    bottom.spines["left"].set_visible(False)
    for a in (top, bottom):
        a.set_xlim(0, x.max() + 1)
    bottom.set_xlabel("Whisker trial", labelpad=1)


def _movement_dots(ax, df, ylabel):
    cols = ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]
    for i, c in enumerate(cols):
        for k, cohort in enumerate(COHORTS):
            v = df[(df.movement == c) & (df.cohort == cohort)].r_minus_null
            _dots_individual(ax, i + (k - 0.5) * 0.4, v, COLORS[cohort], seed=i * 2 + k)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xticks(range(4), [MOVEMENT_LABELS[c] for c in cols])
    for t, c in zip(ax.get_xticklabels(), cols):
        t.set_color(BODYPART[c])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel(ylabel, labelpad=1)


def plot_movement_perf(ax, df):
    _movement_dots(ax, df, "r with performance\n(minus shift null)")


def plot_movement_rt(ax, df):
    _movement_dots(ax, df, "r with reaction time\n(minus shift null)")


# ---------------------------------------------------------------- behavioural GLM

GLM_LABELS = {"lick_whisker_trial_n-1": "Lick n-1", "lick_whisker_trial_n-2": "Lick n-2",
              "lick_whisker_trial_n-3": "Lick n-3", "spontaneous_lick_rate": "Spont. licks",
              "time_since_reward": "Time since rew.", "cumulative_reward_fraction": "Cum. reward",
              "jaw_speed_pre": "Jaw", "nose_speed_pre": "Nose", "whisker_speed_pre": "Whisker",
              "pupil_area_pre": "Pupil", "all_movement": "All mvt"}


def plot_glm_schematic(ax, df):
    """Design matrix of the GLM example session (user 2026-11-10): predictors (rows, z-scored over the session) by
    whisker trial, the lick outcome below, and the model."""
    ax.axis("off")
    order = list(df.feature.drop_duplicates())
    z = df.pivot(index="feature", columns="whisker_trial", values="z").loc[order].to_numpy()
    y = df.drop_duplicates("whisker_trial").sort_values("whisker_trial").lick.to_numpy()
    m = ax.inset_axes([0.42, 0.34, 0.58, 0.56])
    m.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    m.set_yticks(range(len(order)), [GLM_LABELS.get(re.sub(r"_t\d+$", "", f), f) for f in order], fontsize=4)
    m.set_xticks([])
    m.tick_params(length=0, pad=1)
    for side in m.spines.values():
        side.set_linewidth(0.4)
    m.set_title("Design matrix X", fontsize=5, pad=2)
    o = ax.inset_axes([0.42, 0.24, 0.58, 0.06])
    o.imshow(y[None, :], aspect="auto", cmap=mpl.colors.ListedColormap(["white", "0.15"]), vmin=0, vmax=1,
             interpolation="nearest")
    o.set_xticks([0, len(y) - 1], ["1", str(len(y))], fontsize=4)
    o.set_yticks([0], ["Lick y"], fontsize=4)
    o.tick_params(length=0, pad=1)
    for side in o.spines.values():
        side.set_linewidth(0.4)
    o.set_xlabel("Whisker trial", fontsize=4.5, labelpad=0)
    ax.text(0.71, 0.02, "P(lick) = σ(w·x + b)", transform=ax.transAxes, ha="center", va="bottom", fontsize=5)


def plot_glm_example(ax, df):
    x = df.whisker_trial
    lick = df.lick.to_numpy() == 1
    ax.scatter(x[lick], np.full(lick.sum(), 1.06), s=2, marker="|", lw=0.4, color=COLORS["R+"])
    ax.scatter(x[~lick], np.full((~lick).sum(), -0.06), s=2, marker="|", lw=0.4, color=COLORS["R-"])
    ax.plot(x, df.observed_rolling10, color="0.6", lw=0.6, label="Data")
    if "predicted_min" in df:  # spread over the 10 cross-validation repeats
        ax.fill_between(x, df.predicted_min, df.predicted_max, color=light(COLORS[df.cohort.iloc[0]], 0.6), lw=0)
    ax.plot(x, df.predicted_p_lick, color=COLORS[df.cohort.iloc[0]], lw=0.8, label="GLM (mean over repeats)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Whisker trial", labelpad=1)
    ax.set_ylabel("P(lick)", labelpad=1)
    ax.set_title(f"{df.mouse_id.iloc[0]}, balanced accuracy {df.test_balanced_accuracy.iloc[0]:.2f}", fontsize=5.5,
                 pad=7)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.98), ncol=2, fontsize=4.5, frameon=False, handlelength=1,
              borderaxespad=0)


def plot_glm_fit(ax, df):
    _two_dots(ax, df, "test_balanced_accuracy", "Balanced accuracy", highlight_col="is_example")
    ax.axhline(0.5, color="0.6", lw=0.4, ls=":")
    ax.set_ylim(0.3, 1.05)


def _feature_rows(ax, df, value, order, xlabel):
    """Predictors on the y axis (top = first in `order`); filled = cohort mean differs from zero (Wilcoxon p < 0.05);
    asterisks at the right = R+ vs R- (Mann-Whitney p < 0.05)."""
    for i, f in enumerate(order):
        vals = {}
        for k, cohort in enumerate(COHORTS):
            v = df[(df.feature == f) & (df.cohort == cohort)][value].dropna()
            vals[cohort] = v
            nonzero = len(v) > 5 and stats.wilcoxon(v).pvalue < 0.05
            ax.errorbar(v.mean(), i + (-0.17 if k == 0 else 0.17), xerr=v.sem(), color=COLORS[cohort], marker="o",
                        ms=1.8, mfc=COLORS[cohort] if nonzero else "white", mew=0.5, capsize=0, elinewidth=0.6, lw=0)
        p = _mw_p(vals["R+"], vals["R-"])
        if p < 0.05:  # inside the axes, in a margin added on the right
            ax.text(0.995, i, stars(p), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=4.5)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.15 * (hi - lo))
    ax.axvline(0, color="0.6", lw=0.4, ls=":")
    ax.set_yticks(range(len(order)), [GLM_LABELS.get(f, f) for f in order], fontsize=4)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_xlabel(xlabel, labelpad=1)


def plot_glm_weights(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank_by_mean_abs").feature.tolist()
    _feature_rows(ax, df, "coefficient", order, "GLM weight")


def plot_glm_weights_auc(ax, df):
    """Weights (predictor names), then the drop in held-out log-likelihood and in balanced accuracy when the predictor
    is shuffled (planner 2026-11-09), same predictor order."""
    axes = _split(ax, 3, gap=0.06)
    left = axes[0]
    order = df[df.metric == "weight"].drop_duplicates("feature").sort_values("rank_by_mean_abs").feature.tolist()
    _feature_rows(left, df[df.metric == "weight"], "value", order, "Weight")
    _feature_rows(axes[1], df[df.metric == "delta_auc"], "value", order, "Δ AUC\n(shuffled)")
    _feature_rows(axes[2], df[df.metric == "delta_ll_drop"], "value", order, "Δ log-lik./trial\n(removed)")
    for a in axes[1:]:
        a.set_yticklabels([])
    for a in axes:  # narrow axes: few ticks
        a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, symmetric=a is left))


def plot_glm_shuffle(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank").feature.tolist()
    _feature_rows(ax, df, "delta_test_auc", order, "Δ AUC (shuffled)")
    ax.set_yticklabels([])  # same predictor order as the weights panel


def plot_glm_ll(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank").feature.tolist()
    _feature_rows(ax, df, "delta_ll_nats_per_trial", order, "Δ LL (nats/trial)")
    ax.set_yticklabels([])  # same predictor order as the weights panel
