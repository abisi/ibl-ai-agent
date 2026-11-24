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

# user 2026-11-17 'harmonize font sizes': two sizes only (paperstyle rc): 6 pt axis labels and titles, 5 pt ticks,
# legends and annotations
FS_LABEL, FS_TITLE, FS_TICK, FS_NOTE = 6, 6, 5, 5

COHORTS = ["R+", "R-"]
TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
TT_SHORT = {"whisker_trial": "W", "auditory_trial": "A", "no_stim_trial": "FA"}
INDIVIDUAL = "0.72"  # grey for individual mice
BODYPART = {"jaw_motion": "#8c510a", "nose_motion": "#01665e", "whisker_motion": "#5e3c99", "pupil_area": "#4d4d4d"}
MOVEMENT_LABELS = {"jaw_motion": "Jaw speed", "nose_motion": "Nose speed", "whisker_motion": "Whisker speed",
                   "pupil_area": "Pupil area"}  # DLC names = body part + quantity (planner, user 2026-11-16)


def group_color(trial_type, cohort):
    """Whisker: cohort colour; auditory: blue (lighter for R-); false alarm: black (R+), grey (R-)."""
    if trial_type == "auditory_trial":
        return COLORS["auditory"] if cohort == "R+" else light(COLORS["auditory"], 0.5)
    if trial_type == "no_stim_trial":
        return COLORS["no_stim"] if cohort == "R+" else "0.6"
    return COLORS[cohort]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def p_text(p, label="p"):
    """'p = 0.012' / 'p < 0.001' (label 'q' for corrected values)."""
    return f"{label} < 0.001" if p < 0.001 else f"{label} = {p:.2g}" if p < 0.01 else f"{label} = {p:.2f}"


def sig_text(ax, x, y, p, color="k", label="p", transform=None, va="bottom", ha="center", rotate_p=False, ns=True,
             fontsize=None):
    """User 2026-11-22: one significance style for every test: stars (n.s. when p >= 0.05), the p-value written right
    above them (below them when va='top'); rotate_p writes the p-value vertically above the stars (rows of tests)."""
    if p is None or not np.isfinite(p) or (p >= 0.05 and not ns):
        return
    fs = fontsize or FS_NOTE
    kw = dict(fontsize=fs, color=color, clip_on=False, annotation_clip=False,
              xycoords=transform if transform is not None else "data", textcoords="offset points")
    s = stars(p)
    if rotate_p:  # dense rows: one vertical label, stars then p reading upwards (downwards below the data)
        up = va != "top"
        ax.annotate(f"{s}  {p_text(p, label)}" if up else f"{p_text(p, label)}  {s}", (x, y), xytext=(0, 0),
                    ha="center", va="bottom" if up else "top", rotation=90, **kw)
        return
    ax.annotate(s, (x, y), xytext=(0, 0), ha=ha, va=va, **kw)
    if va == "center":  # stars centred on y (rows): p-value above, both centred
        ax.annotate(p_text(p, label), (x, y), xytext=(0, 2.6), ha=ha, va="bottom", fontsize=fs,
                    **{k: v for k, v in kw.items() if k != "fontsize"})
        return
    up = va != "top"
    h = (5.4 if s == "n.s." else 4.4) * (1 if up else -1)  # just clear of the stars
    ax.annotate(p_text(p, label), (x, y), xytext=(0, h), ha=ha if not rotate_p else "center",
                va="bottom" if up else "top", rotation=90 if rotate_p else 0, fontsize=fs,
                **{k: v for k, v in kw.items() if k != "fontsize"})


def _mw_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan


def _bracket(ax, x0, x1, y, p, color="0.2"):
    if not np.isfinite(p):
        return
    ax.plot([x0, x1], [y, y], color=color, lw=0.4)
    sig_text(ax, (x0 + x1) / 2, y, p, color=color if color != "0.2" else "k")


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


def _runs(xs, step):
    """Contiguous runs of sorted x values (gap <= 1.5 step) as (start, end) pairs."""
    xs = np.sort(np.asarray(xs, float))
    if not len(xs):
        return []
    cut = np.flatnonzero(np.diff(xs) > 1.5 * step) + 1
    return [(r[0], r[-1]) for r in np.split(xs, cut)]


def _p_label(ax, x, y, p, color="k"):
    """p-value of a significance bar, above its centre (planner 2026-11-12: bars annotated with their p-value)."""
    if np.isfinite(p):  # user 2026-11-22: stars with the p-value above (sig_text)
        sig_text(ax, x, y, p, color=color, transform=ax.get_xaxis_transform())


def p_shade(p):
    """Grey level of a significance bar by its p-value (user 2026-11-16): black p < 0.001, dark grey p < 0.01,
    mid grey p < 0.05."""
    return "0" if p < 0.001 else "0.35" if p < 0.01 else "0.6"


def _shaded_bars(ax, d, x, y, step, pcol="cluster_p", label="", lw=1.2):
    """One bar per run of significant x (rows of d), grey to black by the run's p-value, its p above."""
    xs = np.sort(d[x].unique())
    x0, x1 = ax.get_xlim()
    pos = ax.get_position()
    ax_mm = pos.width * ax.figure.get_figwidth() * 25.4
    placed = []  # (centre, half width, level) of labels already drawn: labels never overlap, they stack instead
    for a, b in _runs(xs, step):
        p = d[d[x].between(a, b)][pcol].min() if pcol in d else 0.0
        color = p_shade(p) if np.isfinite(p) else "k"
        ax.plot([a - step / 2, b + step / 2], [y, y], transform=ax.get_xaxis_transform(), color=color, lw=lw,
                solid_capstyle="butt", clip_on=False)
        if pcol in d and np.isfinite(p):  # user 2026-11-22: stars on the bar, (label) p-value above them
            text = label + p_text(p)
            c = (a + b) / 2
            half = 0.5 * len(text) * 0.9 / ax_mm * (x1 - x0)  # ~0.9 mm per character at 5 pt
            level = 0
            while any(lv == level and abs(c - pc) < half + ph for pc, ph, lv in placed):
                level += 1
            placed.append((c, half, level))
            yy = y + 0.005 + 0.16 * level
            ax.annotate(stars(p), (c, yy), xycoords=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=FS_NOTE, color=color, annotation_clip=False)
            ax.annotate(text, (c, yy), xycoords=ax.get_xaxis_transform(), xytext=(0, 4.4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FS_NOTE, color=color, annotation_clip=False)


def _sig_bar(ax, df, x, y=0.97):
    """Bars at the top of the axes over x values inside a significant cluster, grey to black by p (p_shade)."""
    if "cluster_sig" not in df:
        return
    d = df[df.cluster_sig.astype(bool)]
    if not len(d):
        return
    step = np.min(np.diff(np.sort(df[x].unique()))) if df[x].nunique() > 1 else 1
    _shaded_bars(ax, d, x, y, step)


def _legend(ax, entries, loc="upper right", marker=None, **kw):
    """Small frameless legend (user 2026-11-12: every panel names its cohorts / trial types / marker conventions).
    entries: (label, color) for lines, or (label, color, dict of Line2D kwargs)."""
    handles = []
    for e in entries:
        label, color, extra = (*e, {}) if len(e) == 2 else e
        style = dict(marker=marker, ms=2, lw=0) if marker else dict(lw=0.8)
        handles.append(mpl.lines.Line2D([], [], color=color, label=label, **{**style, **extra}))
    ax.legend(handles=handles, loc=loc, fontsize=kw.pop("fontsize", FS_NOTE), frameon=False, handlelength=1 if not marker else 0.8,
              handletextpad=0.3, borderaxespad=kw.pop("borderaxespad", 0.2), labelspacing=0.2, **kw)


def _cohort_entries(tt="whisker_trial"):
    return [(c.replace("-", "−"), group_color(tt, c)) for c in COHORTS]


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
    """Lick probability across days of learner mice (planner 2026-11-11: the R+ - R- difference sub-panel dropped), whisker
    in cohort colours and auditory / false alarm in grey, day 0 shaded; mean +- s.e.m. over mice. Right (planner
    2026-11-12): day-0 lick probability per trial type as bars (as panel c); brackets, W vs FA within each cohort (paired
    Wilcoxon, cohort colour) and R+ W vs R- W (Mann-Whitney, black)."""
    ax, right = _split(ax, 2, gap=0.2)  # user 2026-11-17: closer
    x = np.arange(len(DAYS))
    for tt in TRIAL_TYPES:
        for k, cohort in enumerate(COHORTS):
            g = df[(df.trial_type == tt) & (df.cohort == cohort)].groupby("day_group").lick_probability
            m, s = g.mean().reindex(DAYS), g.sem().reindex(DAYS)
            w = tt == "whisker_trial"
            ax.errorbar(x + (k - 0.5) * 0.08, m, s,  # auditory in auditory colours (planner 2026-11-11)
                        color=group_color(tt, cohort) if tt != "no_stim_trial" else CONTEXT_GREY[tt][k],
                        marker="o", ms=1.8 if w else 1.2, lw=0.9 if w else 0.5, elinewidth=0.5, capsize=0,
                        zorder=3 if w else 2)
    ax.axvspan(1.6, 2.4, color="0.92", lw=0, zorder=0)
    ax.set_xticks(x, ["-2", "-1", "0", "1-3"])
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlim(-0.4, len(DAYS) - 0.6)
    ax.set_xlabel("Day", labelpad=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("P(lick)", labelpad=1)
    _legend(ax, [*[(f"W {c}", color) for c, color in _cohort_entries()],  # in the empty band below A
                 *[(f"{TT_SHORT[tt]} {c.replace('-', '−')}", group_color(tt, c) if tt != "no_stim_trial" else CONTEXT_GREY[tt][k])
                   for tt in ("auditory_trial", "no_stim_trial") for k, c in enumerate(COHORTS)]],
            loc="center left", marker="o", bbox_to_anchor=(0, 0.6), ncol=1, fontsize=FS_NOTE)
    d0 = df[df.day_group == "0"]
    v = {}
    for i, tt in enumerate(TRIAL_TYPES):
        for k, c in enumerate(COHORTS):
            v[tt, c] = d0[(d0.trial_type == tt) & (d0.cohort == c)].set_index("mouse_id").lick_probability.dropna()
            _bars(right, i * 1.1 + (k - 0.5) * 0.42, v[tt, c], group_color(tt, c), width=0.38)
    paired = lambda c: stats.wilcoxon(*v["whisker_trial", c].align(v["no_stim_trial", c], join="inner")).pvalue
    for (x0, x1, y, p, color) in ((-0.21, 2.2 - 0.21, 1.24, paired("R+"), COLORS["R+"]),
                                  (0.21, 2.2 + 0.21, 1.5, paired("R-"), COLORS["R-"]),
                                  (-0.21, 0.21, 0.66, _mw_p(v["whisker_trial", "R+"], v["whisker_trial", "R-"]), "k"),
                                  (1.1 - 0.21, 1.1 + 0.21, 1.0,  # user 2026-11-23: auditory R+ vs R-
                                   _mw_p(v["auditory_trial", "R+"], v["auditory_trial", "R-"]), "k"),
                                  (2.2 - 0.21, 2.2 + 0.21, 0.4,  # user 2026-11-12: FA rate R+ vs R-
                                   _mw_p(v["no_stim_trial", "R+"], v["no_stim_trial", "R-"]), "k")):
        right.plot([x0, x0, x1, x1], [y - 0.02, y, y, y - 0.02], color=color, lw=0.4)
        sig_text(right, (x0 + x1) / 2, y, p, color=color)
    right.set_xticks([0, 1.1, 2.2], ["W", "A", "FA"])
    right.set_xlim(-0.5, 2.7)
    right.set_ylim(0, 1.78)  # brackets with stars and p (user 2026-11-22, auditory added 2026-11-23)
    right.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    right.set_ylabel("P(lick)", labelpad=1)
    right.set_title("Day 0", fontsize=FS_TITLE, pad=2)  # user 2026-11-16
    right.set_xlabel("Trial type", labelpad=1)  # R+ / R- colour key at the figure level (user 2026-11-16)


def plot_rt(ax, df):
    for i, tt in enumerate(TRIAL_TYPES):
        vals = {}
        for k, cohort in enumerate(COHORTS):
            v = df[(df.trial_type == tt) & (df.cohort == cohort)].median_reaction_time_s.dropna()
            _bars(ax, i * 1.1 + (k - 0.5) * 0.42, v, group_color(tt, cohort), width=0.38)  # planner: barplot
            vals[cohort] = v
        top = max(v.mean() + v.sem() for v in vals.values())
        _bracket(ax, i * 1.1 - 0.21, i * 1.1 + 0.21, top * 1.08, _mw_p(vals["R+"], vals["R-"]))
    # user / planner 2026-11-12: whisker vs auditory, cohorts pooled (paired over mice, Wilcoxon), with its p-value
    top = df.groupby(["trial_type", "cohort"]).median_reaction_time_s.agg(lambda v: v.mean() + v.sem()).max()
    w = df.pivot_table(index="mouse_id", columns="trial_type", values="median_reaction_time_s")
    pair = w[["whisker_trial", "auditory_trial"]].dropna()
    p = stats.wilcoxon(pair.whisker_trial, pair.auditory_trial).pvalue if len(pair) > 5 else np.nan
    y = top * 1.25  # user 2026-11-16: bracket from the W pair to the A pair, stars centred above, no overlap
    ax.plot([0, 0, 1.1, 1.1], [y - 0.015, y, y, y - 0.015], color="k", lw=0.4)
    sig_text(ax, 0.55, y, p)
    ax.annotate("W vs A", (0.55, y), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                fontsize=FS_NOTE)  # above the stars and p
    ax.set_xticks([0, 1.1, 2.2], ["W", "A", "FA"])
    ax.set_xlim(-1.1, 2.7)  # room for the W bracket label (no overlap with the y axis)
    ax.set_ylim(0, y * 1.3)  # headroom for the pooled bracket and its label
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.set_ylabel("Mean of median RT (s)", labelpad=1)
    ax.set_xlabel("Trial type", labelpad=1)


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
    # user 2026-11-16: hits in the cohort colour, misses grey
    # user 2026-11-23: raster ticks thicker and longer (0.7 pt wide < 1 pt trial spacing: no overlap)
    ax.scatter(x[lick], np.full(lick.sum(), 1.06), s=6, marker="|", lw=0.7, color=color, zorder=4)
    ax.scatter(x[~lick], np.full((~lick).sum(), -0.06), s=6, marker="|", lw=0.7, color="0.6", zorder=4)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, x.max() + 1)
    ax.set_xlabel("Whisker trial", labelpad=1)
    ax.set_ylabel("P(lick), learning curve", labelpad=1)  # user 2026-11-17 (was P(lick))
    if first:
        _legend(ax, [("Whisker", color), ("Auditory", TRIAL_TYPE_COLORS["auditory_trial"]),
                     ("False alarm", COLORS["no_stim"])], loc="center right")
    ax.set_title(f"{title} ({df.mouse_id.iloc[0]})", fontsize=FS_TITLE, pad=2)
    ax.tick_params(length=2.5, width=0.6, labelsize=FS_TICK)  # planner: ticks more visible


def plot_ex_rplus_learner(ax, df):
    _example(ax, df, "R+ learner", first=True)


def plot_ex_rplus_nonlearner(ax, df):
    _example(ax, df, "R+ non-learner")


def plot_ex_rminus_learner(ax, df):
    _example(ax, df, "R- learner", first=True)  # legend as in E (user 2026-11-16)


def plot_ex_rminus_nonlearner(ax, df):
    _example(ax, df, "R- non-learner")


# ---------------------------------------------------------------- first lick and false alarms

def plot_rt_change(ax, df, xlabel, fit=None, display_bin=5, mice=None):
    """Reaction-time change from the mouse's first 5 hits (kind 'bin' rows of compute.rt_change_from_start, one per
    mouse and hit): for display, per-mouse mean over bins of `display_bin` hits, then mean +- s.e.m. over mice, cohorts
    side by side (group_color); coloured bars, hits in a cohort-vs-0 cluster (sig_vs0); black bar, hits in an R+ vs R-
    cluster (sig_cohorts; clusters are tested hit by hit). `fit` (kind 'fit' rows): mean of the per-mouse hit-by-hit
    least-squares fits, dashed line over hits 1-30. `mice` (kind 'mouse' rows; planner 2026-11-11): last versus first
    bin of 5 hits per mouse (last_minus_first_ms); coloured stars next to the last bin, Wilcoxon vs 0 within cohort;
    bracket, R+ vs R- Mann-Whitney."""
    tt = df.trial_type.iloc[0]
    step = float(np.min(np.diff(np.sort(df.hit_bin_centre.unique()))))
    shown = df.assign(x=((df.hit_bin_centre - 1) // display_bin) * display_bin + (display_bin + 1) / 2)
    # per-mouse MEDIAN per display bin (user 2026-11-16): the first bin (hits 1-5) is the reference, 0 by definition
    shown = shown.groupby(["mouse_id", "cohort", "x"]).value.median().reset_index()
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
    bars = [(df[df.sig_cohorts.astype(bool)], "k", 0.98, "sig_cohorts_p")]
    bars += [(df[(df.cohort == c) & df.sig_vs0.astype(bool)], group_color(tt, c), 0.93 - 0.05 * k, "sig_vs0_p")
             for k, c in enumerate(COHORTS)]
    for d, color, y, pcol in bars:
        for x0 in np.sort(d.hit_bin_centre.unique()):
            ax.plot([x0 - step / 2, x0 + step / 2], [y, y], transform=ax.get_xaxis_transform(), color=color, lw=1,
                    solid_capstyle="butt")
        if pcol in d:
            for s0, s1 in _runs(d.hit_bin_centre.unique(), step):
                _p_label(ax, (s0 + s1) / 2, y + 0.005, d[d.hit_bin_centre.between(s0, s1)][pcol].min(), color)
    # (cohort-coloured bars kept here: they code which cohort differs from 0, not only p)
    ax.set_xlim(0, df.hit_bin_centre.max() + 1)
    if mice is not None and "last_minus_first_ms" in mice:  # user 2026-11-18: stats on the plot, not written above
        # last versus first bin of 5 hits per mouse: coloured stars right of each cohort's last bin (Wilcoxon vs 0),
        # bracket joining the two last bins (R+ vs R- Mann-Whitney)
        v = {c: mice[mice.cohort == c].last_minus_first_ms.dropna() for c in COHORTS}
        last = {}
        for k, c in enumerate(COHORTS):
            g = shown[shown.cohort == c].groupby("x").value
            m, e = g.mean(), g.sem()
            last[c] = (m.index.max() + (k - 0.5) * 0.8, m.iloc[-1], e.iloc[-1])
        hi_c = max(last, key=lambda c: last[c][1])
        for c, (x, y, e) in last.items():  # above the upper cohort's error bar, below the lower one's
            p = stats.wilcoxon(v[c]).pvalue
            up = c == hi_c
            sig_text(ax, x, y + e if up else y - e, p, color=group_color(tt, c), va="bottom" if up else "top")
        y0, y1 = sorted(y for _, y, _ in last.values())
        xb = max(x for x, _, _ in last.values()) + 2.5
        ax.plot([xb - 0.8, xb, xb, xb - 0.8], [y0, y0, y1, y1], color="k", lw=0.5)
        p = stats.mannwhitneyu(v["R+"], v["R-"]).pvalue
        sig_text(ax, xb + 0.5, (y0 + y1) / 2, p, ha="left", va="center")
        ax.set_xlim(0, xb + 3)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel("Δ RT from first 5 hits (ms)", labelpad=1)
    _legend(ax, _cohort_entries(tt), loc="upper left", marker="o")


def plot_rt_slopes(ax, mice, tt="whisker_trial"):
    """Per-mouse hit-by-hit RT slope (ms per 10 hits; kind 'mouse' rows): grey mice, cohort mean +- s.e.m.; stars
    under each cohort, Wilcoxon vs 0; bracket, R+ vs R- Mann-Whitney (user 2026-11-10)."""
    _two_dots(ax, mice, "slope_ms_per_10_hits", "RT slope (ms / 10 hits)",
              colors=[group_color(tt, c) for c in COHORTS])
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    lo = ax.get_ylim()[0]
    for i, c in enumerate(COHORTS):
        v = mice[mice.cohort == c].slope_ms_per_10_hits.dropna()
        sig_text(ax, i, lo, stats.wilcoxon(v).pvalue, color=group_color(tt, c))
    ax.set_xticks([0, 1], [c.replace("-", "−") for c in COHORTS])


def _plot_rt_aligned(a, df):
    """User 2026-11-22 ('K aligned with I'): whisker-hit RT change from the mouse's first 5 hits on the trial axis of
    Fig. 1I (kind 'bin': per-mouse median per bin 0, 1-5, ..., 26-30), mean +- s.e.m. over mice; stars above a bin, R+ vs
    R- (Mann-Whitney, Benjamini-Hochberg; kind 'bin_test'); right, late change (whisker trials 21-30, kind 'mouse'):
    coloured stars vs 0 (Wilcoxon), bracket R+ vs R- (Mann-Whitney)."""
    bins, tests, mice = df[df.kind == "bin"], df[df.kind == "bin_test"], df[df.kind == "mouse"]
    last = {}
    for k, c in enumerate(COHORTS):
        g = bins[bins.cohort == c].groupby("bin_centre").value
        m, e = g.mean(), g.sem()
        a.errorbar(m.index + (k - 0.5) * 0.6, m, e, color=COLORS[c], marker="o", ms=1.6, lw=0.6, elinewidth=0.5,
                   capsize=0, zorder=3)
        last[c] = (m.index.max() + (k - 0.5) * 0.6, m.iloc[-1], e.iloc[-1])
    a.axhline(0, color="0.6", lw=0.4, ls=":")
    lo, hi = a.get_ylim()
    a.set_ylim(lo - 0.1 * (hi - lo), hi + 0.35 * (hi - lo))  # room for the per-bin stars and p-values
    _bin_test_stars(a, tests, y=0.86)
    v = {c: mice[mice.cohort == c].late_change_ms.dropna() for c in COHORTS}
    hi_c = max(last, key=lambda c: last[c][1])
    for c, (x, y, e) in last.items():  # late change vs 0, above / below the last bin
        up = c == hi_c
        sig_text(a, x, y + e if up else y - e, stats.wilcoxon(v[c]).pvalue, color=COLORS[c],
                 va="bottom" if up else "top")
    y0, y1 = sorted(y for _, y, _ in last.values())
    xb = max(x for x, _, _ in last.values()) + 2.5
    a.plot([xb - 0.8, xb, xb, xb - 0.8], [y0, y0, y1, y1], color="k", lw=0.5)
    sig_text(a, xb + 0.6, (y0 + y1) / 2, stats.mannwhitneyu(v["R+"], v["R-"]).pvalue, ha="left", va="center")
    a.set_xlim(-1.5, xb + 3)
    a.set_xticks([0, 10, 20, 30])
    a.set_xlabel("Trials from 1st W hit", labelpad=1)
    a.set_ylabel("Δ RT from first 5 hits (ms)", labelpad=1)
    _legend(a, _cohort_entries(), loc="upper left", marker="o", bbox_to_anchor=(0, 0.8))


def _density_contours(a, x, y, color, levels=(0.25, 0.5, 0.75)):
    """Contours of a Gaussian kernel density enclosing the given fractions of the points (user 2026-11-23: density where
    the dots overlap)."""
    k = stats.gaussian_kde(np.vstack([x, y]))
    xx, yy = np.meshgrid(np.linspace(*a.get_xlim(), 80), np.linspace(*a.get_ylim(), 80))
    zz = k(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    dens = np.sort(k(np.vstack([x, y])))[::-1]  # density at the points: the level enclosing a fraction f of them
    lv = sorted(dens[min(len(dens) - 1, int(f * len(dens)))] for f in levels)
    a.contour(xx, yy, zz, levels=lv, colors=[color], linewidths=[0.4, 0.55, 0.7][:len(lv)], zorder=4)


def _plot_rt_scatter(a, d, ylim):
    """User 2026-11-23: reaction time of every whisker hit against the whisker curve at that trial, both cohorts overlaid
    (dots, hits of all mice; contours, kernel density enclosing 25, 50, 75 % of the hits); line, fixed effect of the
    linear mixed model (mouse random intercept and slope); text, rank-based LMM rho (mean within-mouse Spearman) and p."""
    a.set_xlim(0, 1.02)
    a.set_ylim(*ylim)
    for k, c in enumerate(COHORTS):
        g = d[d.cohort == c]
        a.scatter(g.curve, g.rt_ms, s=0.5, color=light(COLORS[c], 0.55), lw=0, rasterized=True, zorder=1 + k)
    for k, c in enumerate(COHORTS):
        g = d[d.cohort == c]
        _density_contours(a, g.curve.to_numpy(), g.rt_ms.to_numpy(), COLORS[c])
        x = np.array([g.curve.min(), g.curve.max()])
        a.plot(x, g.lmm_intercept.iloc[0] + g.lmm_slope.iloc[0] * x, color=COLORS[c], lw=1.4, zorder=5)
        a.text(0.03, 0.89 - 0.08 * k, f"{c.replace('-', '−')}: ρ = {g.rho_lmm.iloc[0]:.2f}, {p_text(g.rho_lmm_p.iloc[0])}",
               transform=a.transAxes, ha="left", va="top", fontsize=FS_NOTE, color=COLORS[c],
               bbox=dict(fc="white", ec="none", pad=0.2))  # opaque (EPS), readable over the dots
    a.text(0.03, 0.97, "LMM, mouse random effects", transform=a.transAxes, ha="left", va="top", fontsize=FS_NOTE)
    a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    a.set_xlabel("Whisker curve at the hit (P(lick))", labelpad=1)
    a.set_ylabel("Whisker-hit RT (ms)", labelpad=1)


def _plot_rho(right, rho, null):
    """Spearman rho (RT vs whisker curve) per licked trial type, mean +- s.e.m. over mice; grey hollow dot +- s.d., the
    cohort mean under the linear-shift null; stars, Wilcoxon of rho minus each mouse's shift-null median vs 0."""
    for i, tt in enumerate(TRIAL_TYPES):
        for k, cohort in enumerate(COHORTS):
            g = rho[(rho.trial_type == tt) & (rho.cohort == cohort)]
            v, x = g.rho.dropna(), i + (k - 0.5) * 0.35
            nd = null[(null.trial_type == tt) & (null.cohort == cohort)].rho.dropna()
            if len(nd):  # shift-null cohort mean +- s.d., left of the real dot; hollow (user 2026-11-12)
                right.errorbar(x - 0.12, nd.mean(), nd.std(), color="0.55", marker="o", ms=1.8, mfc="white", mew=0.5,
                               lw=0, elinewidth=0.6, capsize=0, zorder=2)
            right.errorbar(x, v.mean(), v.sem(), color=group_color(tt, cohort), marker="o", ms=2.2, lw=0, elinewidth=0.6,
                           capsize=0, zorder=3)
            d = g.value.dropna()  # rho minus the mouse's shift-null median
            p = stats.wilcoxon(d).pvalue if len(d) > 5 else np.nan
            top = max(v.mean() + v.sem(), nd.mean() + nd.std() if len(nd) else -np.inf)
            sig_text(right, x, top + 0.02, p, color=group_color(tt, cohort), rotate_p=True)
    right.axhline(0, color="0.6", lw=0.4, ls=":")
    right.set_xticks(range(3), ["W", "A", "FA"])
    right.set_xlim(-0.6, 2.6)
    lo, hi = right.get_ylim()
    right.set_ylim(lo, hi + 0.45 * (hi - lo))  # room for the stars, p-values and legend
    right.set_ylabel("ρ(RT, W curve)", labelpad=1)
    right.set_xlabel("Licked trial type", labelpad=1)
    _legend(right, [("Shift null", "0.55", dict(marker="o", ms=2, lw=0, mfc="white", mew=0.5))],
            loc="lower right")  # user 2026-11-23


F_MM_PER_TRIAL = 23.0 / 41.0  # mm per trial on the x axis of F (square subpanels of 23 mm, -10.5 to 30.5 trials)


def plot_rt_trials(ax, df):
    """User 2026-11-23: the previous version (whisker hits 1-30 from the start of day 0; first bin of 5 hits = 0 by
    definition), with one hit on the x axis as long as one trial on the x axis of F."""
    ax.axis("off")
    pos, fig = ax.get_position(), ax.figure
    W = pos.width * fig.get_figwidth() * 25.4
    span = 40.0  # hits 0-40: hits 1-30, the last-versus-first bracket and its label
    a = ax.inset_axes([0, 0, min(1.0, span * F_MM_PER_TRIAL / W), 1])
    plot_rt_change(a, df[df.kind == "bin"], "Whisker hit", None, mice=df[df.kind == "mouse"])
    a.set_xlim(0, span)
    lo, hi = a.get_ylim()
    a.set_ylim(lo - 0.12 * (hi - lo), hi)  # room for the stars under the lower cohort


def plot_rt_curve(ax, df):
    """User 2026-11-23: (1) whisker-hit RT against the whisker curve, cohorts overlaid, density contours, LMM lines and
    rank-LMM rho; (2) per-mouse Spearman rho per licked trial type with the shift null."""
    # user 2026-11-23: the scatter (left) removed from the figure; saved in the snapshot's _removed/ (save_removed_scatter)
    _plot_rho(ax, df[df.kind == "rho"], df[df.kind == "rho_null"])


def save_removed_scatter(df, out_dir):
    """User 2026-11-23: the RT-vs-curve scatter of the former left part of J, as its own file in <snapshot>/_removed/."""
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = df[df.kind == "scatter"]
    fig = plt.figure(figsize=(60 / 25.4, 48 / 25.4))
    a = fig.add_axes([0.2, 0.2, 0.75, 0.75])
    _plot_rt_scatter(a, sc, (0, float(np.nanpercentile(sc.rt_ms, 99.5)) * 1.3))  # room for the text inside
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"Fig1J_rt_curve_scatter.{ext}", dpi=600)
    sc.to_csv(out_dir / "Fig1J_rt_curve_scatter.csv", index=False)
    plt.close(fig)


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
                    fontsize=FS_TITLE, pad=2)
        a.set_xlabel({"whisker_trial": "W trials", "no_stim_trial": "No-stim trials", "auditory_trial": "A trials"}[tt]
                     + " from 1st W hit", labelpad=1, fontsize=FS_LABEL)
    axes[0].set_ylabel("P(lick)", labelpad=1)
    axes[2].legend(handles=[mpl.lines.Line2D([], [], color="0.3", lw=0.8, label="Curve"),
                            mpl.lines.Line2D([], [], color="0.3", marker="o", mfc="white", ms=2, lw=0, label="Data")],
                   loc="lower right", fontsize=FS_NOTE, frameon=False, handlelength=1, borderaxespad=0)


def plot_first_hit_licks(ax, df):
    """Planner 2026-11-11: whisker | auditory | false alarm, P(lick) by trial of the type from the first whisker hit in
    5-trial bins (x = bin centre; first 90% of session time); mean +- s.e.m. over mice, group_color of the trial type;
    grey band, the pre window; black bar, bins where R+ and R- differ (BH-corrected Mann-Whitney)."""
    axes = _split(ax, 3, gap=0.1)  # planner 2026-11-11 14:19: spontaneous licks dropped
    spec = {"whisker": ("whisker_trial", "Whisker", "P(lick)", "lower right"),  # every axis labelled (user 2026-11-12)
            "auditory": ("auditory_trial", "Auditory", "P(lick)", "lower right"),
            "false_alarm": ("no_stim_trial", "False alarm", "P(lick)", "upper right")}
    for a, (sub, (tt, title, ylabel, loc)) in zip(axes, spec.items()):
        d = df[df.sub_panel == sub]
        trial = (d.x_unit == "trial").all()
        # trial bins: x = centre of the bin's trials (whisker: the hit, trial 0, is its own bin at 1 by definition)
        d = d.assign(t=(d.bin_start + d.bin_end - 1) / 2 if trial else d.bin_start + 0.5)
        if trial:
            d = d[d.bin_start < 30]  # user 2026-11-12: back to 30 trials
            a.axvspan(-5.5, -0.5, color="0.93", lw=0, zorder=0)
            a.axvline(-0.5, color="0.6", lw=0.4, ls=":")
        else:
            a.axvspan(-5, 0, color="0.93", lw=0, zorder=0)
            a.axvline(0, color="0.6", lw=0.4, ls=":")
        for cohort in COHORTS:
            _mean_band(a, d[d.cohort == cohort], "t", "value", group_color(tt, cohort))
        sig = d[d.cohort_sig.astype(bool)].drop_duplicates("bin_start")
        for m, m_end in zip(sig.bin_start, sig.bin_end if trial else sig.bin_start + 1):
            x0, x1 = (m - 0.5, m_end - 0.5) if trial else (m, m_end)
            a.plot([x0, x1], [1.02, 1.02], transform=a.get_xaxis_transform(), color="k", lw=1.2, solid_capstyle="butt",
                   clip_on=False)
        if len(sig):  # planner 2026-11-12: the bars' p-value (largest BH q of the significant bins)
            q = sig.cohort_q_bh.max()
            a.text(0.98, 1.03, f"q ≤ {q:.2g}" if q >= 0.001 else "q < 0.001", transform=a.transAxes, ha="right",
                   va="bottom", fontsize=FS_NOTE)
        if trial:
            a.set_xlim(-10.5, 29.5)
            a.set_xticks([-10, 0, 10, 20])
            a.set_ylim(-0.02, 1.04)  # headroom so the bands are not clipped at 0 and 1
            a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            a.set_xlabel("Trials from 1st W hit", labelpad=1, fontsize=FS_LABEL)
        else:
            a.set_xlim(d.bin_start.min(), d.bin_start.max() + 1)
            a.set_xticks([-5, 0, 10, 20])
            a.set_ylim(bottom=0)
            a.set_xlabel("Time from 1st W hit (min)", labelpad=1, fontsize=FS_LABEL)
        a.tick_params(axis="x", labelsize=FS_TICK)
        a.set_title(title, fontsize=FS_TITLE, pad=7)  # above the significance bars (y = 1.02)
        if ylabel:
            a.set_ylabel(ylabel, labelpad=1)
        a.legend(handles=[mpl.lines.Line2D([], [], color=group_color(tt, c), lw=0.8, label=c) for c in COHORTS],
                 loc=loc, fontsize=FS_NOTE, frameon=False, handlelength=1, borderaxespad=0.2)


ALIGN_COLS = [("whisker_trial", "Whisker"), ("no_stim_trial", "False alarm"), ("auditory_trial", "Auditory")]  # FA before A


def _row_axes(ax, n, gap_mm=7.0, square=True):
    """n axes side by side inside ax, gaps in mm; square (side = box height) or filling the box width."""
    ax.axis("off")
    pos, fig = ax.get_position(), ax.figure
    W, H = pos.width * fig.get_figwidth() * 25.4, pos.height * fig.get_figheight() * 25.4
    gaps = list(gap_mm) if np.ndim(gap_mm) else [gap_mm] * (n - 1)  # one gap or one per pair
    w = (W - sum(gaps)) / n
    w = min(H, w) if square else w
    x0 = np.r_[0, np.cumsum([w + g for g in gaps])]
    return [ax.inset_axes([x0[k] / W, 0, w / W, 1]) for k in range(n)]


def _bin_test_stars(a, t, y=1.0, data_y=False):
    """User 2026-11-17: one test per bin (R+ vs R-, Mann-Whitney; Benjamini-Hochberg within the subplot): stars above
    every bin with q < 0.05, nothing elsewhere."""
    for r in t[t.mannwhitney_q_bh < 0.05].itertuples():
        sig_text(a, r.bin_centre, y, r.mannwhitney_q_bh, label="q", rotate_p=True,
                 transform=None if data_y else a.get_xaxis_transform())


PLICK_YLIM = (-0.04, 1.04)  # P(lick) axes of the aligned and session panels (user 2026-11-23: aligned)


def _plot_aligned(ax, df, alabel):
    """User 2026-11-17: whisker, false alarm, auditory P(lick) (kind 'bin'): per-mouse lick fraction per bin (-10..-6,
    -5..-1, 0 alone, 1-5, ...), mean +- s.e.m. over mice, dots connected; bars, bins where R+ and R- differ
    (Mann-Whitney per bin, Benjamini-Hochberg within the subplot)."""
    axes = _row_axes(ax, 3, gap_mm=6.0)  # squares, same size as H and J
    bins, tests = df[df.kind == "bin"], df[df.kind == "bin_test"]
    for a, (tt, title) in zip(axes, ALIGN_COLS):
        d = bins[bins.trial_type == tt]
        for k, cohort in enumerate(COHORTS):
            g = d[d.cohort == cohort].groupby("bin_centre").p_lick
            m, e = g.mean(), g.sem()
            a.errorbar(m.index + (k - 0.5) * 0.5, m, e, color=group_color(tt, cohort), marker="o", ms=1.6, lw=0.6,
                       elinewidth=0.5, capsize=0, zorder=3)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.set_xlim(-10.5, 30.5)
        a.set_xticks([-10, 0, 10, 20, 30])
        a.set_ylim(*PLICK_YLIM)  # user 2026-11-23: same y axis as the next panel; discreet stars above the axes
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        for r in tests[(tests.trial_type == tt) & (tests.mannwhitney_q_bh < 0.05)].itertuples():
            a.annotate(stars(r.mannwhitney_q_bh), (r.bin_centre, 1.0), xycoords=a.get_xaxis_transform(), ha="center",
                       va="bottom", fontsize=FS_NOTE, color="0.3", annotation_clip=False)
        a.set_xlabel(f"Trials from {alabel}", labelpad=1, fontsize=FS_LABEL)
        a.set_ylabel("P(lick)", labelpad=1, fontsize=FS_LABEL)
        a.set_title(title, fontsize=FS_TITLE, pad=7)
        a.tick_params(labelsize=FS_TICK)


def plot_aligned_trial(ax, df):
    _plot_aligned(ax, df, "1st W trial")


def plot_aligned_hit(ax, df):
    _plot_aligned(ax, df, "1st W hit")


def _delta_axis(a, d, alabel):
    """Delta P(lick) (whisker minus chance curve), both cohorts overlaid, mean +- s.e.m. over mice; cluster bars."""
    for cohort in COHORTS:
        _mean_band(a, d[d.cohort == cohort], "session_time_norm", "performance", COLORS[cohort])
    _sig_bar(a, d, "session_time_norm", y=1.0)
    a.axhline(0, color="0.6", lw=0.4, ls=":")
    a.set_xlim(0, 1)
    a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    a.set_ylim(-0.05, 0.45)  # user 2026-11-17
    a.set_yticks([0, 0.2, 0.4])
    a.set_xlabel(f"Session time from {alabel} (norm.)", labelpad=1, fontsize=FS_LABEL)
    a.set_ylabel("Δ P(lick) (W − FA)", labelpad=1, fontsize=FS_LABEL)
    a.tick_params(labelsize=FS_TICK)


def _plot_perf_three(ax, df, alabel):
    """User 2026-11-17: from the alignment (0) to the session end (1): learning curves of whisker, false-alarm (chance)
    and auditory trials, one square axis per cohort, then Delta P(lick) with both cohorts overlaid (all squares of the
    same size as G and I)."""
    axes = _row_axes(ax, 3, gap_mm=[2.5, 9.0])
    curves = [("p_whisker", "whisker_trial", "Whisker"), ("p_chance", "no_stim_trial", "False alarm"),
              ("p_auditory", "auditory_trial", "Auditory")]
    for a, cohort in zip(axes[:2], COHORTS):
        d = df[df.cohort == cohort]
        for col, tt, lab in curves:
            _mean_band(a, d.dropna(subset=[col]), "session_time_norm", col, group_color(tt, cohort), lw=0.7)
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        a.set_ylim(*PLICK_YLIM)
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        if cohort == "R+":  # R- shares the y axis
            a.set_ylabel("P(lick), learning curve", labelpad=1, fontsize=FS_LABEL)
        else:
            a.set_yticklabels([])
        a.set_title(f"{cohort.replace('-', '−')} (n = {d.mouse_id.nunique()})", fontsize=FS_TITLE, pad=7,
                    color=COLORS[cohort])
        a.tick_params(labelsize=FS_TICK)
    _legend(axes[1], [(lab, group_color(tt, "R-")) for _, tt, lab in curves], loc="center right",
            bbox_to_anchor=(1, 0.7))
    _legend(axes[0], [(lab, group_color(tt, "R+")) for _, tt, lab in curves], loc="lower left")  # user 2026-11-18
    _delta_axis(axes[2], df, alabel)
    _legend(axes[2], [(c.replace("-", "−"), COLORS[c]) for c in COHORTS], loc="upper left")  # user 2026-11-18
    axes[2].set_xlabel("")
    axes[1].set_xlabel(f"Session time from {alabel} (0) to session end (1)", labelpad=1, fontsize=FS_LABEL)


def plot_perf_trial(ax, df):
    _plot_perf_three(ax, df, "1st W trial")


def plot_perf_hit(ax, df):
    _plot_perf_three(ax, df, "1st W hit")


HITMISS_LABELS = {"whisker_speed": ("Whisker speed", "Δ deg/s"), "jaw_opening": ("Jaw opening", "Δ mm"),
                  "pupil_area": ("Pupil area", "Δ mm²"), "nose_speed": ("Nose speed", "Δ mm/s")}


def plot_movement_hitmiss(ax, df):
    """Supp. Fig. 1 (user 2026-11-16), one DLC measure: whisker hits (cohort colour) and whisker misses (grey), one axis
    per cohort, stimulus-aligned, each trial minus its pre-stimulus mean, mean +- s.e.m. over mice; bars, hit vs miss
    clusters within the cohort (sign-flip test), grey to black by p."""
    sig = df.signal.iloc[0]
    label, unit = HITMISS_LABELS[sig]
    axes = _split(ax, 2, gap=0.32)
    for a, cohort in zip(axes, COHORTS):
        d = df[df.cohort == cohort]
        for outcome, color in (("miss", "0.55"), ("hit", COLORS[cohort])):
            dd = d[d.outcome == outcome]
            if len(dd):
                _mean_band(a, dd, "time_s", "value", color, lw=0.6, fill=0.75)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.15 * (hi - lo))
        on = d[d.cluster_sig.astype(bool)].drop_duplicates("time_s")
        if len(on):
            step = np.min(np.diff(np.sort(d.time_s.unique())))
            _shaded_bars(a, on, "time_s", 0.98, step, lw=1)
        a.set_xlim(d.time_s.min(), d.time_s.max())
        a.set_xticks([0, d.time_s.max()], ["0", f"{d.time_s.max():.2g}"])
        a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        a.tick_params(labelsize=FS_TICK, pad=1)
        a.set_xlabel("Time (s)", labelpad=0.5, fontsize=FS_LABEL)
        a.set_ylabel(f"{label} ({unit})", labelpad=0.5, fontsize=FS_LABEL)
        a.set_title(cohort.replace("-", "−"), fontsize=FS_TITLE, pad=6, color=COLORS[cohort])
        _legend(a, [("Hit", COLORS[cohort]), ("Miss", "0.55")], loc="upper left", bbox_to_anchor=(0, 0.88))
    ax.set_title(f"{label}, whisker trials", fontsize=FS_TITLE, pad=13)


def plot_movement_course(ax, df):
    """Hit trials: whisker speed and jaw opening minus each trial's pre-stimulus mean; whisker and auditory trials, both
    cohorts (group_color), mean +- s.e.m. over mice. One column per body part, top -0.1 to 0.5 s, bottom zoom -0.02 to
    0.1 s (user 2026-11-23: pre-stimulus baseline 100 and 20 ms). User 2026-11-23: square axes, right-aligned in the
    panel, legend inside the jaw zoom axes. Bars, R+ vs R- clusters (whisker trials top, auditory below), grey to black."""
    ax.axis("off")
    pos, fig = ax.get_position(), ax.figure
    W, H = pos.width * fig.get_figwidth() * 25.4, pos.height * fig.get_figheight() * 25.4
    vgap, hgap = 6.0, 10.0  # mm: x tick labels of the top row; y tick labels of the right column
    side = (H - vgap) / 2  # user 2026-11-23: larger, not square: height from the box, width fills it
    wide = (W - 9.0 - hgap) / 2  # 9 mm for the left y labels
    xs = [W - 2 * wide - hgap, W - wide]  # flush right
    rows = {"full": side + vgap, "zoom": 0.0}  # bottom in mm
    windows = {"full": (-0.1, 0.5), "zoom": (-0.02, 0.1)}
    for k, (sig, label, unit) in enumerate(SIGNALS[:2]):
        ds = df[df.signal == sig]
        for row, y0 in rows.items():
            a = ax.inset_axes([xs[k] / W, y0 / H, wide / W, side / H])
            lo_t, hi_t = windows[row]
            d = ds[ds.time_s.between(lo_t - 1e-9, hi_t + 1e-9)]
            for tt in ("whisker_trial", "auditory_trial"):
                for c in COHORTS:
                    dd = d[(d.trial_type == tt) & (d.cohort == c)]
                    if len(dd):
                        _mean_band(a, dd, "time_s", "value", group_color(tt, c), lw=0.6, fill=0.75)
            a.axvline(0, color="0.6", lw=0.4, ls=":")
            a.axhline(0, color="0.8", lw=0.3)
            a.set_xlim(lo_t, hi_t)
            lo, hi = a.get_ylim()
            ticks = mpl.ticker.MaxNLocator(3).tick_values(min(lo, 0), hi)
            a.set_yticks(ticks[(ticks >= lo) & (ticks <= hi)])
            a.set_ylim(lo, hi + 0.15 * (hi - lo))  # room for the bars
            step = np.min(np.diff(np.sort(d.time_s.unique()))) if d.time_s.nunique() > 1 else 0
            for tt, y in (("whisker_trial", 0.98), ("auditory_trial", 0.91)):
                on = d[(d.trial_type == tt) & d.cluster_sig.astype(bool)]
                if len(on):
                    _shaded_bars(a, on, "time_s", y, step, label=f"{TT_SHORT[tt]} ", lw=1)
            a.set_xticks([lo_t, 0, hi_t], ["", "0", f"{hi_t:g}"])  # baseline start unlabelled: no room (caption)
            a.tick_params(labelsize=FS_TICK, pad=1)
            if row == "zoom":
                a.set_xlabel("Time (s)", labelpad=0)
            else:
                a.set_title(label, fontsize=FS_TITLE, pad=12)
            a.set_ylabel(unit, labelpad=0.5, fontsize=FS_LABEL)
            if k == 1 and row == "zoom":
                entries = [(f"{TT_SHORT[t]} {c}", color) for t in ("whisker_trial", "auditory_trial")
                           for c, color in _cohort_entries(t)]
                _legend(a, entries, loc="upper left", borderaxespad=0.1)


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
                       mpl.lines.Line2D([], [], color="0.2", lw=0.8, ls="--", label="Day −1"),
                       *[mpl.lines.Line2D([], [], color=group_color("no_stim_trial", c), lw=0.8,
                                          label=c.replace("-", "−")) for c in COHORTS]],  # user 2026-11-23: cohorts too
              loc="upper left", fontsize=FS_NOTE, frameon=False, handlelength=1.4, borderaxespad=0)


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
    ax.set_xlabel("Lag (W trials)\n← FA leads | W leads →", labelpad=1)
    ax.set_ylabel("r (W, FA curves)", labelpad=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    _legend(ax, [(c.replace("-", "−"), COLORS[c]) for c in COHORTS]
            + [("Shift null (lag 0)", "0.5", dict(ls="--", lw=0.5))], loc="upper left", ncol=1)  # user 2026-11-22


# ---------------------------------------------------------------- learners and states

def plot_learners(ax, df):
    df = df.set_index("cohort").loc[COHORTS]
    ax.bar([0, 1], df.fraction_learners, color=[COLORS["R+"], COLORS["R-"]], width=0.6, lw=0)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, r.fraction_learners + 0.02, f"{r.learners}/{r.mice}", ha="center", va="bottom", fontsize=FS_NOTE)
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
    ax.set_ylabel("P(lick)", labelpad=1)
    handles = [mpl.lines.Line2D([], [], color="0.3", marker="o", mfc="white", ms=2, lw=0, label="Low"),
               mpl.lines.Line2D([], [], color="0.3", marker="o", ms=2, lw=0, label="High")]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.45), fontsize=FS_NOTE, handletextpad=0.2,
              borderaxespad=0, frameon=False, ncol=1)


def plot_rt_states(ax, df):
    _state_pairs(ax, df, "median_reaction_time_s")
    ax.set_ylabel("Median RT (s)", labelpad=1)


SIGNALS = [("whisker_speed", "Whisker speed", "Δ deg/s"), ("jaw_opening", "Jaw opening", "Δ mm"),
           ("pupil_area", "Pupil area", "Δ mm²")]


SUPP_WINDOW = {"whisker_speed": (-0.05, 0.1), "jaw_opening": (-0.1, 0.3), "pupil_area": (-1.0, 3.0)}  # Supp. Fig. 1d planner


def plot_movement_peth(ax, df):
    """Rows: whisker speed, jaw opening, pupil area; columns: whisker, auditory trials; stimulus-aligned; lick solid,
    no lick dashed (whisker trials; auditory hits only). Bars: R+ vs R- clusters on lick trials."""
    grid = _subgrid(ax, 3, 2, wgap=0.35, hgap=0.55)
    for r, (sig, label, unit) in enumerate(SIGNALS):
        for c, tt in enumerate(["whisker_trial", "auditory_trial"]):
            a = grid[r][c]
            d = df[(df.signal == sig) & (df.trial_type == tt) & (df["align"] == "stimulus")]
            d = d[d.time_s.between(*SUPP_WINDOW[sig])]  # PETH cache v7 runs the jaw to 0.5 s (Fig. 1d)
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
            a.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
            if r == 0:
                a.set_title("Whisker" if tt == "whisker_trial" else "Auditory", fontsize=FS_TITLE, pad=1.5)
            a.set_ylabel(f"{label.split()[0]}\n({unit})", labelpad=1, fontsize=FS_LABEL)  # every axis (user 2026-11-12)
            a.set_xlabel("Time (s)", labelpad=0.5, fontsize=FS_LABEL)


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
            a.tick_params(labelsize=FS_TICK)
            if r == 0:
                a.set_title(f"{label}\n{'stimulus' if align == 'stimulus' else 'lick'}-aligned", fontsize=FS_TITLE, pad=2)
            if r == 1:
                a.set_xlabel(f"Time from {'stim.' if align == 'stimulus' else 'lick'} (s)", labelpad=1, fontsize=FS_LABEL)
            if align == "stimulus":
                a.set_ylabel(unit, labelpad=1, fontsize=FS_LABEL)
        grid[r][0].text(-0.75, 0.5, "Whisker trials" if tt == "whisker_trial" else "Auditory trials",
                        transform=grid[r][0].transAxes, rotation=90, ha="center", va="center", fontsize=FS_NOTE)
    handles = [mpl.lines.Line2D([], [], color=group_color(tt, c), lw=0.8, label=f"{TT_SHORT[tt]} {c}")
               for tt in ("whisker_trial", "auditory_trial") for c in COHORTS]
    handles += [mpl.lines.Line2D([], [], color="0.3", lw=0.8, label="Lick"),
                mpl.lines.Line2D([], [], color="0.3", lw=0.8, ls="--", label="No lick")]
    grid[0][5].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=FS_NOTE, frameon=False,
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
    bottom.set_yticks([-1.1 * k for k in range(4)], [MOVEMENT_LABELS[c] for c in cols], fontsize=FS_NOTE)
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
              "jaw_speed_pre": "Jaw speed", "nose_speed_pre": "Nose speed", "whisker_speed_pre": "Whisker speed",
              "pupil_area_pre": "Pupil area", "all_movement": "All movement"}  # DLC names (planner, user 2026-11-16)


GLM_GROUPS = [("History", ("lick_whisker_trial_n-1", "lick_whisker_trial_n-2", "lick_whisker_trial_n-3",
                             "spontaneous_lick_rate")),
              ("Reward", ("cumulative_reward_fraction", "time_since_reward")),
              ("Movement", ("jaw_speed_pre", "nose_speed_pre", "whisker_speed_pre", "pupil_area_pre"))]


def plot_glm_schematic(ax, df):
    """User 2026-11-16 (bigger, clearer): design matrix of the GLM example session, predictors grouped (history, reward,
    pre-stimulus movement; z-scored over the session for display) by whisker trial, colour bar, the lick outcome below
    (hits in the cohort colour, misses grey) and the model."""
    ax.axis("off")
    ax.set_title("GLM of whisker performance", fontsize=FS_TITLE, pad=2)
    cohort = df.cohort.iloc[0]
    feats = list(df.feature.drop_duplicates())
    base = {f: re.sub(r"_t\d+$", "", f) for f in feats}
    order = [f for _, g in GLM_GROUPS for f in feats if base[f] in g] + [f for f in feats if not any(base[f] in g for _, g in GLM_GROUPS)]
    z = df.pivot(index="feature", columns="whisker_trial", values="z").loc[order].to_numpy()
    y = df.drop_duplicates("whisker_trial").sort_values("whisker_trial").lick.to_numpy()
    pred = "predicted_p_lick" in df  # user 2026-11-22: model prediction under the lick row
    m = ax.inset_axes([0.38, 0.42, 0.44, 0.46] if pred else [0.38, 0.30, 0.44, 0.58])
    im = m.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    m.set_yticks(range(len(order)), [GLM_LABELS.get(base[f], f) for f in order], fontsize=FS_NOTE)
    m.set_xticks([])
    m.tick_params(length=0, pad=1)
    k = 0
    for name, g in GLM_GROUPS:  # white separators between groups, group names on the right
        n = sum(base[f] in g for f in order)
        if not n:
            continue
        if k:
            m.axhline(k - 0.5, color="white", lw=1.0)
        short = {"History": "Hist.", "Reward": "Rew.", "Movement": "Mov."}.get(name, name)  # fits at 5 pt
        m.text(1.03, 1 - (k + n / 2) / len(order), short, transform=m.transAxes, rotation=90, ha="left", va="center",
               fontsize=FS_NOTE, color="0.25")
        k += n
    for side in m.spines.values():
        side.set_linewidth(0.4)
    m.set_title("Design matrix X (z-scored)", fontsize=FS_TITLE, pad=2)
    cb = ax.inset_axes([0.95, 0.52, 0.025, 0.25])
    ax.figure.colorbar(im, cax=cb, ticks=[-2, 0, 2])
    cb.tick_params(labelsize=FS_TICK, length=1, pad=0.5, width=0.3)
    cb.set_title("z", fontsize=FS_TITLE, pad=1)
    o = ax.inset_axes([0.38, 0.33, 0.44, 0.05] if pred else [0.38, 0.20, 0.44, 0.06])
    o.imshow(y[None, :], aspect="auto", cmap=mpl.colors.ListedColormap(["0.8", COLORS[cohort]]), vmin=0, vmax=1,
             interpolation="nearest")
    o.set_xticks([0, len(y) - 1], ["1", str(len(y))], fontsize=FS_NOTE)
    o.set_yticks([0], ["Lick y"], fontsize=FS_NOTE)
    o.tick_params(length=0, pad=1)
    for side in o.spines.values():
        side.set_linewidth(0.4)
    if not pred:
        o.set_xlabel("Whisker trial", fontsize=FS_LABEL, labelpad=0)
        ax.text(0.63, 0.0, "P(lick) = σ(w · x + b)", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS_NOTE)
        return
    o.set_xticks([])
    q = ax.inset_axes([0.38, 0.25, 0.44, 0.05])  # held-out GLM prediction of the same session (user 2026-11-23:
    yp = df.drop_duplicates("whisker_trial").sort_values("whisker_trial").predicted_p_lick.to_numpy()  # colour strip)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("pred", ["0.9", "k"])  # user 2026-11-23: black
    q.imshow(yp[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    q.set_yticks([0], ["P(lick) = σ(w · x + b)"], fontsize=FS_NOTE)
    q.set_xticks([0, len(yp) - 1], ["1", str(len(yp))], fontsize=FS_NOTE)
    q.tick_params(length=0, pad=1)
    for side in q.spines.values():
        side.set_linewidth(0.4)
    cq = ax.inset_axes([0.88, 0.25, 0.07, 0.05])  # its colour scale, clear of the trial numbers
    cq.imshow(np.linspace(0, 1, 50)[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1)
    cq.set_yticks([])
    cq.set_xticks([0, 49], ["0", "1"], fontsize=FS_TICK)
    cq.tick_params(length=0, pad=1)
    for side in cq.spines.values():
        side.set_linewidth(0.4)
    q.set_xlabel("Whisker trial", fontsize=FS_LABEL, labelpad=0)


def plot_glm_example(ax, df):
    x = df.whisker_trial
    lick = df.lick.to_numpy() == 1
    color = COLORS[df.cohort.iloc[0]]  # user 2026-11-16: hits cohort colour, misses grey; curve vs GLM (black)
    ax.scatter(x[lick], np.full(lick.sum(), 1.06), s=2, marker="|", lw=0.4, color=color)
    ax.scatter(x[~lick], np.full((~lick).sum(), -0.06), s=2, marker="|", lw=0.4, color="0.6")
    if "curve_p_mean" in df:
        ax.plot(x, df.curve_p_mean, color=color, lw=0.8, label="Whisker curve")
    if "predicted_min" in df:  # spread over the 10 cross-validation repeats
        ax.fill_between(x, df.predicted_min, df.predicted_max, color="0.85", lw=0)  # opaque (EPS export)
    ax.plot(x, df.predicted_p_lick, color="k", lw=0.4, label="GLM")  # thinner (user 2026-11-17)
    ax.set_ylim(-0.1, 1.28)  # room for the legend inside the axes (user 2026-11-18)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Whisker trial", labelpad=1)
    ax.set_ylabel("P(lick)", labelpad=1)
    ax.set_title(f"{df.mouse_id.iloc[0]}, bal. acc. {df.test_balanced_accuracy.iloc[0]:.2f}", fontsize=FS_TITLE,
                 pad=6)
    ax.legend(loc="upper center", ncol=2, fontsize=FS_NOTE, frameon=False, handlelength=1, borderaxespad=0.1,
              columnspacing=0.8)  # inside the axes, above the lick raster (user 2026-11-18)


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
            sig_text(ax, 0.995, i, p, transform=ax.get_yaxis_transform(), ha="right", va="center")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.45 * (hi - lo))  # user 2026-11-22: room for the stars and p-values
    ax.axvline(0, color="0.6", lw=0.4, ls=":")
    ax.set_yticks(range(len(order)), [GLM_LABELS.get(f, f) for f in order], fontsize=FS_NOTE)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_xlabel(xlabel, labelpad=1)


def plot_glm_weights(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank_by_mean_abs").feature.tolist()
    _feature_rows(ax, df, "coefficient", order, "GLM weight")


def _glm_schematic_min(ax, rect, df):
    """User 2026-11-23: minimal design-matrix schematic (one strip): predictors grouped (names of the groups only), the
    lick outcome and the held-out prediction of the example session, and the model."""
    cohort = df.cohort.iloc[0]
    feats = list(df.feature.drop_duplicates())
    base = {f: re.sub(r"_t\d+$", "", f) for f in feats}
    groups = [GLM_GROUPS[i] for i in (1, 2, 0)]  # reward, movement, history: the order of the rows below
    order = [f for _, g in groups for f in feats if base[f] in g]
    z = df.pivot(index="feature", columns="whisker_trial", values="z").loc[order].to_numpy()
    tr = df.drop_duplicates("whisker_trial").sort_values("whisker_trial")
    x0, y0, w, h = rect
    m = ax.inset_axes([x0, y0 + 0.42 * h, w, 0.58 * h])
    m.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    m.set_xticks([])
    m.set_yticks([])
    k = 0
    for name, g in groups:
        n = sum(base[f] in g for f in order)
        if k:
            m.axhline(k - 0.5, color="white", lw=0.8)
        m.text(-0.01, 1 - (k + n / 2) / len(order), name, transform=m.transAxes, ha="right", va="center",
               fontsize=FS_NOTE, color="0.25")
        k += n
    o = ax.inset_axes([x0, y0 + 0.30 * h, w, 0.09 * h])
    o.imshow(tr.lick.to_numpy()[None, :], aspect="auto", interpolation="nearest", vmin=0, vmax=1,
             cmap=mpl.colors.ListedColormap(["0.8", COLORS[cohort]]))
    o.set_xticks([])
    o.set_yticks([])
    o.text(-0.01, 0.5, "Lick", transform=o.transAxes, ha="right", va="center", fontsize=FS_NOTE, color="0.25")
    q = ax.inset_axes([x0, y0, w, 0.26 * h])
    q.plot(np.arange(len(tr)), tr.predicted_p_lick.to_numpy(), color="k", lw=0.5)
    q.set_xlim(-0.5, len(tr) - 0.5)
    q.set_ylim(0, 1)
    q.axis("off")
    q.text(-0.01, 0.5, "GLM", transform=q.transAxes, ha="right", va="center", fontsize=FS_NOTE, color="0.25")
    for a_ in (m, o):
        for side in a_.spines.values():
            side.set_linewidth(0.3)
    ax.text(x0 + w + 0.015, y0 + 0.5 * h, "P(lick) = σ(w · x + b)", transform=ax.transAxes, ha="left", va="center",
            fontsize=FS_NOTE)
    ax.text(x0 + w / 2, y0 - 0.015, "Whisker trials of one session", transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_NOTE, color="0.25")


GROUP_LABELS = {"reward": "Reward\nhistory", "movement": "Movement", "history": "Lick\nhistory"}  # two lines, rotated


def _group_overlay(a, d, order, groups, own_axes=False):
    """User 2026-11-23: drop of held-out log-likelihood when a group of predictors is removed and the model refitted,
    overlaid on the single-predictor axes (same x): dark grey, at the mean y of the group's predictors (R+ circle above,
    R- square below; filled = differs from 0, Wilcoxon); right y axis = the groups; dark-grey stars right of the points,
    R+ vs R- (Mann-Whitney; p in the caption)."""
    yc = {}
    for g, members in groups.items():
        ys = [order.index(f) for f in members if f in order]
        yc[g] = float(np.mean(ys))
        vals, right = {}, -np.inf
        for k, c in enumerate(COHORTS):  # user 2026-11-23: diamonds in the cohort colours
            v = d[(d.feature == g) & (d.cohort == c)].value.dropna()
            vals[c] = v
            nonzero = len(v) > 5 and stats.wilcoxon(v).pvalue < 0.05
            a.errorbar(v.mean(), yc[g] + (-0.15 if k == 0 else 0.15), xerr=v.sem(), color=COLORS[c], marker="D", ms=2.4,
                       mfc=COLORS[c] if nonzero else "white", mew=0.6, capsize=0, elinewidth=0.8, lw=0, zorder=4)
            right = max(right, v.mean() + v.sem())
        p = _mw_p(vals["R+"], vals["R-"])
        if p < 0.05:
            a.annotate(stars(p), (right, yc[g]), xytext=(1.5, 0), textcoords="offset points", ha="left", va="center",
                       fontsize=FS_NOTE, color="0.25")
    # right y axis drawn on the same axes (a twinx of an inset axes is mispositioned). User 2026-11-23: no full spine;
    # one segment per group spanning its predictors, the group name at its centre
    a.spines["right"].set_visible(False)
    if own_axes:  # user 2026-11-23: groups in their own column right of the single predictors
        a.spines["left"].set_visible(False)
        a.set_yticks([])
    tr = a.get_yaxis_transform()
    for g, members in groups.items():
        ys = [order.index(f) for f in members if f in order]
        a.plot([1.0, 1.0], [min(ys) - 0.35, max(ys) + 0.35], transform=tr, color="0.25", lw=0.6, clip_on=False,
               solid_capstyle="butt")
        a.plot([1.0, 1.04], [yc[g], yc[g]], transform=tr, color="0.25", lw=0.5, clip_on=False)
        a.text(1.07, yc[g], GROUP_LABELS.get(g, g), transform=tr, ha="left", va="center", fontsize=FS_NOTE, rotation=90, linespacing=0.9,
               color="0.25")


def plot_glm_weights_auc(ax, df):
    """User 2026-11-23: weights (predictor names), then the drop in held-out log-likelihood per trial when a predictor is
    removed and the model refitted (same order), with the groups of predictors removed together (reward history,
    movement, lick history; dark grey, right y axis) overlaid on the same x axis. No schematic (in Supp. Fig. 1)."""
    pos, fig = ax.get_position(), ax.figure
    Wmm = pos.width * fig.get_figwidth() * 25.4
    ax.axis("off")
    widths, gap = (1.0, 1.0, 0.7), 3.0  # user 2026-11-23: weights | single predictor removed | group removed
    unit = (Wmm - 2 * gap) / sum(widths)
    x0 = np.r_[0, np.cumsum([w * unit + gap for w in widths])]
    axes = [ax.inset_axes([x0[k] / Wmm, 0, widths[k] * unit / Wmm, 1]) for k in range(3)]
    W = df[df.metric == "weight"]
    order = W.drop_duplicates("feature").sort_values("rank_by_mean_abs").feature.tolist()
    _feature_rows(axes[0], W, "value", order, "Weight")
    single, grp = df[df.metric == "delta_ll_drop"], df[df.metric == "delta_ll_group"]
    _feature_rows(axes[1], single, "value", order, "Δ log-lik./trial (nats)")
    lo = min(axes[1].get_xlim()[0], (grp.groupby(["feature", "cohort"]).value.mean()
                                     - grp.groupby(["feature", "cohort"]).value.sem()).min())
    hi = max(axes[1].get_xlim()[1], (grp.groupby(["feature", "cohort"]).value.mean()
                                     + grp.groupby(["feature", "cohort"]).value.sem()).max() * 1.6)  # room for stars
    for a in axes[1:]:  # same x range: single and group removal comparable
        a.set_xlim(lo, hi)
    groups = {g: m.split(";") for g, m in grp.drop_duplicates("feature")[["feature", "group_members"]].to_numpy()}
    axes[2].set_ylim(axes[1].get_ylim())
    axes[2].axvline(0, color="0.6", lw=0.4, ls=":")
    _group_overlay(axes[2], grp, order, groups, own_axes=True)
    axes[2].set_xlabel("Δ LL/trial\n(group)", labelpad=1)
    axes[1].set_xlabel("Δ LL/trial (nats)\n(predictor)", labelpad=1)
    axes[1].set_yticklabels([])  # same predictor order: names once (ssl-figure-style rule 6)
    for a in axes:  # user 2026-11-23: y range of the predictors only; legend above the axes
        a.set_ylim(len(order) - 0.4, -0.6)
    axes[2].tick_params(axis="y", length=0)
    _legend(axes[0], [("≠ 0", "0.3", dict(marker="o", ms=2, lw=0)),
                      ("n.s.", "0.3", dict(marker="o", ms=2, lw=0, mfc="white", mew=0.5))],
            loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, columnspacing=0.6, borderaxespad=0.1)

    axes[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, symmetric=True))
    for a in axes[1:]:  # narrow axes: two ticks
        a.set_xticks([0, 0.03], ["0", "0.03"])


def plot_glm_shuffle(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank").feature.tolist()
    _feature_rows(ax, df, "delta_test_auc", order, "Δ AUC (shuffled)")
    ax.set_yticklabels([])  # same predictor order as the weights panel


def plot_glm_ll(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank").feature.tolist()
    _feature_rows(ax, df, "delta_ll_nats_per_trial", order, "Δ LL (nats/trial)")
    ax.set_yticklabels([])  # same predictor order as the weights panel
