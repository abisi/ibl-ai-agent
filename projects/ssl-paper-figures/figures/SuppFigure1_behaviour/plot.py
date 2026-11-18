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
plot_rt_curves, plot_posture_curve = s2.plot_rt_curves, s2.plot_posture_curve
PREV = ["prev_whisker_hit", "prev_whisker_miss", "prev_auditory_hit", "prev_auditory_miss"]
PREV_LABEL = ["W hit", "W miss", "A hit", "A miss"]
RATE_VARS = {"aud_day0": "Auditory day 0", "fa_day0": "No-stim day 0", "aud_daym1": "Auditory day -1",
             "fa_daym1": "No-stim day -1", "wh_day1": "Whisker day +1", "wh_day2": "Whisker day +2",
             "n_pretraining_days": "Pretraining days", "n_pretraining_trials": "Pretraining trials",
             "rt_wh_day0": "Whisker RT day 0 (s)"}  # as compute.RATE_VARS


plot_xcorr = fig1.plot_xcorr  # moved from Figure 1 (planner 2026-11-11)
plot_learners = fig1.plot_learners  # moved from Figure 1 (planner 2026-11-11, 13:12)
plot_first_lick = fig1.plot_first_lick  # moved from Figure 1 g (user 2026-11-16)
plot_hitmiss_whisker_speed = plot_hitmiss_jaw_opening = fig1.plot_movement_hitmiss  # user 2026-11-16
plot_hitmiss_pupil_area = plot_hitmiss_nose_speed = fig1.plot_movement_hitmiss


def plot_ex_rplus_nonlearner(ax, df):
    fig1._example(ax, df, "R+ non-learner", first=True)


def plot_ex_rminus_nonlearner(ax, df):
    fig1._example(ax, df, "R- non-learner")


def plot_history(ax, df):
    """Bars (planner 2026-11-09): P(lick) shift (rate given the previous stimulus trial minus the mouse's overall rate)
    and RT shift on hits, for W and A trials; mean +- s.e.m. over mice, Mann-Whitney asterisks R+ vs R-."""
    axes = fig1._split(ax, 4, gap=0.075)  # one row (planner 2026-11-13): P(lick) shift W, A, then RT shift W, A
    panels = [("whisker_trial", "p_lick_shift", "W trial"), ("auditory_trial", "p_lick_shift", "A trial"),
              ("whisker_trial", "rt_shift_s", "W hit"), ("auditory_trial", "rt_shift_s", "A hit")]
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
        a.set_xticks(range(4), PREV_LABEL, rotation=45, fontsize=4)  # every axis labelled (user 2026-11-12)
        a.set_xlim(-0.5, 3.5)
        a.set_title(lab, fontsize=5, pad=6)
        a.set_xlabel("Previous trial", labelpad=1, fontsize=4.5)
    for a, tt in ((axes[0], "whisker_trial"), (axes[1], "auditory_trial")):
        fig1._legend(a, fig1._cohort_entries(tt), loc="lower left", marker="s")


def plot_rates_correlation(ax, df):
    """Whisker P(lick) day 0 (y) against each measure (x): dots per mouse, OLS line with 95% band per cohort (solid if
    p < 0.05, dashed otherwise), r and p in the corner."""
    from scipy import stats
    vars_ = list(RATE_VARS)
    axes = fig1._split(ax, len(vars_), gap=0.032)  # much wider sub-panels (planner 2026-11-13), y label still on each
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
        a.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        a.tick_params(labelsize=3.8)
        a.xaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(nbins=1, min_n_ticks=2))  # two ticks: no overlap
        a.xaxis.set_major_formatter(fig1.mpl.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if abs(x) >= 1000 else f"{x:g}"))
        a.set_xlabel(RATE_VARS[v], fontsize=4, labelpad=1)
        a.set_ylabel("W P(lick) day 0", labelpad=0.5, fontsize=4)


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
    ax.set_xlabel("Trial type", labelpad=1)  # R+ / R- colour key at the figure level (user 2026-11-16)


# ---------------------------------------------------------------- user 2026-11-18: reorganised Supp. Fig. 1

def plot_cumulative_fa(ax, df):
    fig1.plot_cumulative_fa(ax, df)
    ax.set_title("False alarms, day −1 vs 0", fontsize=fig1.FS_TITLE, pad=6)


def plot_days_expert(ax, df):
    """Left: lick probability per trial type on each day (-2..3), mean +- s.e.m. over mice. Right: d' on whisker days
    0..3; stars, Mann-Whitney R+ vs R- per day."""
    from scipy import stats
    left, right = fig1._split(ax, 2, gap=0.3)
    p = df[df.measure == "p_lick"]
    for tt in fig1.TRIAL_TYPES:
        for k, c in enumerate(fig1.COHORTS):
            g = p[(p.trial_type == tt) & (p.cohort == c)].groupby("day").value
            m, e = g.mean(), g.sem()
            if tt == "whisker_trial":
                m, e = m[m.index >= 0], e[e.index >= 0]
            left.errorbar(m.index + (k - 0.5) * 0.15, m, e, color=fig1.group_color(tt, c), marker="o", ms=1.6, lw=0.6,
                          elinewidth=0.5, capsize=0)
    left.axvspan(-0.4, 0.4, color="0.92", lw=0, zorder=0)
    left.set_xticks(range(-2, 4))
    left.set_ylim(0, 1.02)
    left.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    left.set_xlabel("Day", labelpad=1)
    left.set_ylabel("P(lick)", labelpad=1)
    dp = df[df.measure == "dprime"]
    top = []
    for k, c in enumerate(fig1.COHORTS):
        g = dp[dp.cohort == c].groupby("day").value
        m, e = g.mean(), g.sem()
        right.errorbar(m.index + (k - 0.5) * 0.15, m, e, color=fig1.COLORS[c], marker="o", ms=1.6, lw=0.6,
                       elinewidth=0.5, capsize=0)
        top.append(m + e)
    right.axhline(0, color="0.6", lw=0.4, ls=":")
    hi = np.nanmax(np.vstack([t.reindex(range(4)).to_numpy() for t in top]), 0)
    for day in range(4):
        a, b = dp[(dp.day == day) & (dp.cohort == "R+")].value, dp[(dp.day == day) & (dp.cohort == "R-")].value
        if len(a) > 2 and len(b) > 2:
            pv = stats.mannwhitneyu(a, b).pvalue
            if pv < 0.05:
                right.text(day, hi[day] + 0.1, fig1.stars(pv), ha="center", va="bottom", fontsize=fig1.FS_NOTE)
    right.set_xticks(range(4))
    right.set_xlim(-0.5, 3.5)
    right.set_xlabel("Whisker day", labelpad=1)
    right.set_ylabel("d′ (W vs FA)", labelpad=1)
    fig1._legend(left, [(fig1.TT_SHORT[tt], fig1.group_color(tt, "R+")) for tt in fig1.TRIAL_TYPES], loc="center left",
                 bbox_to_anchor=(0, 0.6))  # trial types; cohorts in the figure key (lighter, R-)


DPRIME_ALIGN = [("first_whisker_trial", "1st W trial"), ("first_whisker_hit", "1st W hit")]


def _plot_dprime(ax, df, xlabel_fmt, norm):
    axes = fig1._split(ax, 2, gap=0.22)
    for a, (align, lab) in zip(axes, DPRIME_ALIGN):
        d = df[(df.alignment == align) & df.in_test.astype(bool)]
        for c in fig1.COHORTS:
            fig1._mean_band(a, d[d.cohort == c], "time", "dprime", fig1.COLORS[c], lw=0.7)
        fig1._sig_bar(a, d, "time", y=1.0)
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        if norm:
            a.set_xlim(0, 1)
            a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        else:
            a.set_xlim(0, d.time.max())
        a.set_xlabel(xlabel_fmt.format(lab), labelpad=1)
        a.set_ylabel("d′ (W vs FA curve)", labelpad=1)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    fig1._legend(axes[0], [(c.replace("-", "−"), fig1.COLORS[c]) for c in fig1.COHORTS], loc="upper left")


def plot_dprime_norm(ax, df):
    _plot_dprime(ax, df, "Session time from {} (norm.)", True)


def plot_dprime_abs(ax, df):
    _plot_dprime(ax, df, "Time from {} (min)", False)


def plot_rt_change_auditory(ax, df):  # user 2026-11-18: same layout and timescale as Fig. 1K left
    fig1.plot_rt_change(ax, df[df.kind == "bin"], "Auditory hit", None, mice=df[df.kind == "mouse"])


EVOKED_COLS = [("whisker_trial", "Whisker", ("Hit", "Miss")), ("auditory_trial", "Auditory", ("Hit", "Miss")),
               ("no_stim_trial", "No stimulus", ("FA", "CR"))]


def _plot_evoked(ax, df):
    """Per trial type: lick (solid) and no-lick (dashed) trials, both cohorts (group_color), mean +- s.e.m. over mice;
    bars, R+ vs R- clusters (lick top, no lick below), grey to black by P."""
    axes = fig1._split(ax, 3, gap=0.12)
    sig = df.signal.iloc[0]
    label, unit = fig1.HITMISS_LABELS[sig]
    for k, (a, (tt, title, outcomes)) in enumerate(zip(axes, EVOKED_COLS)):
        d = df[df.trial_type == tt]
        for lick, ls in ((1, "-"), (0, "--")):
            for c in fig1.COHORTS:
                dd = d[(d.lick == lick) & (d.cohort == c)]
                if dd.mouse_id.nunique() >= 3:
                    fig1._mean_band(a, dd.rename(columns={"time_s": "t"}), "t", "value", fig1.group_color(tt, c),
                                    ls=ls, lw=0.6, fill=0.8)
        step = float(np.min(np.diff(np.sort(d.time_s.unique()))))
        for lick, y, name in ((1, 0.98, outcomes[0]), (0, 0.9, outcomes[1])):
            on = d[(d.lick == lick) & d.cluster_sig.astype(bool)]
            if len(on):
                fig1._shaded_bars(a, on, "time_s", y, step, label=f"{name} ", lw=1)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.axhline(0, color="0.8", lw=0.3)
        a.set_xlim(d.time_s.min(), d.time_s.max())
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.2 * (hi - lo))
        a.set_title(title, fontsize=fig1.FS_TITLE, pad=7)
        a.set_xlabel("Time (s)", labelpad=1)
        if k == 0:
            a.set_ylabel(f"{label} ({unit})", labelpad=1)
    ylo = min(a.get_ylim()[0] for a in axes)
    yhi = max(a.get_ylim()[1] for a in axes)
    for k, a in enumerate(axes):
        a.set_ylim(ylo, yhi)
        if k:
            a.set_yticklabels([])
    fig1._legend(axes[2], [("Lick", "0.2"), ("No lick", "0.2", dict(ls="--"))], loc="center left",
                 bbox_to_anchor=(0, 0.66))  # no-stimulus traces stay low: free space


plot_evoked_whisker_speed = plot_evoked_jaw_opening = plot_evoked_nose_speed = plot_evoked_pupil_area = _plot_evoked
