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
plot_rt_curves = s2.plot_rt_curves
plot_posture_curve, plot_movement_curve = s2.plot_posture_curve, s2.plot_movement_curve  # user 2026-11-23: two panels
PREV = ["prev_whisker_hit", "prev_whisker_miss", "prev_auditory_hit", "prev_auditory_miss"]
PREV_LABEL = ["W hit", "W miss", "A hit", "A miss"]
RATE_VARS = {"aud_day0": "Auditory day 0", "fa_day0": "No-stim day 0", "aud_daym1": "Auditory day -1",
             "fa_daym1": "No-stim day -1", "wh_day1": "Whisker day +1", "wh_day2": "Whisker day +2",
             "n_pretraining_days": "Pretraining days", "n_pretraining_trials": "Pretraining trials",
             "rt_wh_day0": "Whisker RT day 0 (s)"}  # as compute.RATE_VARS


plot_learners = fig1.plot_learners  # moved from Figure 1 (planner 2026-11-11, 13:12)
plot_first_lick = fig1.plot_first_lick  # moved from Figure 1 g (user 2026-11-16)
plot_hitmiss_whisker_speed = plot_hitmiss_jaw_opening = fig1.plot_movement_hitmiss  # user 2026-11-16
plot_hitmiss_pupil_area = plot_hitmiss_nose_speed = fig1.plot_movement_hitmiss


def plot_ex_rplus_nonlearner(ax, df):
    fig1._example(ax, df, "R+ non-learner", first=True)


def plot_ex_rminus_nonlearner(ax, df):
    fig1._example(ax, df, "R- non-learner")


HISTORY_PANELS = [("whisker_trial", "p_lick_shift", "W trials: P(lick)"), ("auditory_trial", "p_lick_shift", "A trials: P(lick)"),
                  ("whisker_trial", "rt_shift_s", "W hits: RT"), ("auditory_trial", "rt_shift_s", "A hits: RT")]


def plot_history(ax, df):
    """User 2026-11-22 (title, layout, style, more stars): shift in P(lick) given the previous stimulus trial (rate given
    the previous trial minus the mouse's rate on that trial type) and in reaction time on hits (ms), for W and A trials;
    bars mean +- s.e.m. over mice, cohorts side by side. Stars with p above: cohort colour, Wilcoxon vs 0 within the
    cohort (above positive bars, below negative ones); black, R+ vs R- (Mann-Whitney) at the top."""
    from scipy import stats
    axes = fig1._row_axes(ax, 4, gap_mm=7.0, square=False)  # user 2026-11-23: 13 mm wide, as the auditory RT axes
    for a, (tt, v, title) in zip(axes, HISTORY_PANELS):
        scale = 1000 if v == "rt_shift_s" else 1
        for i, cond in enumerate(PREV):
            vals = {}
            for k, c in enumerate(fig1.COHORTS):
                d = df[(df.trial_type == tt) & (df.prev == cond) & (df.cohort == c)][v].dropna() * scale
                x = i + (k - 0.5) * 0.38
                fig1._bars(a, x, d, fig1.group_color(tt, c), width=0.34)
                vals[c] = d
                if len(d) > 5:  # user 2026-11-23: small horizontal stars, p-values in the caption
                    m, e = d.mean(), d.sem()
                    up = m >= 0
                    p = stats.wilcoxon(d).pvalue
                    if p < 0.05:
                        a.annotate(fig1.stars(p), (x, m + e if up else m - e), xytext=(0, 0.5 if up else -0.5),
                                   textcoords="offset points", ha="center", va="bottom" if up else "top",
                                   fontsize=fig1.FS_NOTE, color=fig1.group_color(tt, c))
            p = fig1._mw_p(vals["R+"], vals["R-"])
            if p < 0.05:
                a.annotate(fig1.stars(p), (i, 0.97), xycoords=a.get_xaxis_transform(), ha="center", va="top",
                           fontsize=fig1.FS_NOTE, color="k")
        a.axhline(0, color="0.4", lw=0.4)
        lo, hi = a.get_ylim()
        a.set_ylim(lo - 0.12 * (hi - lo), hi + 0.25 * (hi - lo))  # room for the stars
        if a in (axes[0], axes[2]):  # once per measure (no clash with the neighbouring bars)
            a.set_ylabel("Δ P(lick)" if v == "p_lick_shift" else "Δ RT (ms)", labelpad=1)
        a.set_xticks(range(4), PREV_LABEL, rotation=45, ha="right", rotation_mode="anchor")
        a.set_xlim(-0.5, 3.5)
        a.set_title(title, fontsize=fig1.FS_TITLE, pad=3)
        a.set_xlabel("Previous trial", labelpad=1)
    for a, tt in ((axes[0], "whisker_trial"), (axes[1], "auditory_trial")):
        fig1._legend(a, fig1._cohort_entries(tt), loc="lower left", marker="s")
    ax.set_title("Trial history", fontsize=fig1.FS_TITLE, pad=12, loc="left")
    ax.annotate("* p < 0.05, ** p < 0.01, *** p < 0.001; colour, vs 0; black, R+ vs R−", (1.0, 1.0),
                xycoords="axes fraction", xytext=(0, 12), textcoords="offset points", ha="right", va="bottom",
                fontsize=fig1.FS_NOTE, color="0.3")  # star key on the title line, clear of the subplot titles  # user 2026-11-23: star key on the panel


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
    left.set_xticks(range(-2, 3))  # user 2026-11-23: to whisker day 2
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
    hi = np.nanmax(np.vstack([t.reindex(range(3)).to_numpy() for t in top]), 0)
    for day in range(3):
        a, b = dp[(dp.day == day) & (dp.cohort == "R+")].value, dp[(dp.day == day) & (dp.cohort == "R-")].value
        if len(a) > 2 and len(b) > 2:
            fig1.sig_text(right, day, hi[day] + 0.05, stats.mannwhitneyu(a, b).pvalue, rotate_p=True)
    right.set_ylim(top=right.get_ylim()[1] * 1.35)  # room for the stars and p-values
    right.set_xticks(range(3))
    right.set_xlim(-0.5, 2.5)
    right.set_xlabel("Whisker day", labelpad=1)
    right.set_ylabel("d′ (W vs FA)", labelpad=1)
    fig1._legend(left, [(fig1.TT_SHORT[tt], fig1.group_color(tt, "R+")) for tt in fig1.TRIAL_TYPES], loc="center left",
                 bbox_to_anchor=(0, 0.6))  # trial types; cohorts in the figure key (lighter, R-)


DPRIME_ALIGN = [("first_whisker_trial", "1st W trial"), ("first_whisker_hit", "1st W hit")]
DPRIME_YLIM = (-0.3, 1.6)  # one y range for every d' panel (user 2026-11-22: harmonise the learning-curve plots)


def _plot_dprime(ax, df, xlabel_fmt, kind):
    """d' (whisker vs chance curve) against time from the alignment, both cohorts (mean +- s.e.m. over mice); bars, R+ vs
    R- clusters with stars and p. Two square axes as Fig. 1H/J; kind 'norm' (0-1), 'abs_min' (minutes), 'trial'."""
    axes = fig1._row_axes(ax, 2, gap_mm=9.0)
    for a, (align, lab) in zip(axes, DPRIME_ALIGN):
        d = df[(df.alignment == align) & df.in_test.astype(bool)]
        for c in fig1.COHORTS:
            fig1._mean_band(a, d[d.cohort == c], "time", "dprime", fig1.COLORS[c], lw=0.7)
        a.set_ylim(*DPRIME_YLIM)
        fig1._sig_bar(a, d, "time", y=1.0)
        a.axhline(0, color="0.6", lw=0.4, ls=":")
        if kind == "norm":
            a.set_xlim(0, 1)
            a.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        else:
            a.set_xlim(0, d.time.max())
            a.xaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(3, integer=True))
        a.set_yticks([0, 0.5, 1, 1.5])
        a.set_xlabel(xlabel_fmt.format(lab), labelpad=1)
        a.set_ylabel("d′ (W vs FA curve)", labelpad=1)
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    fig1._legend(axes[0], [(c.replace("-", "−"), fig1.COLORS[c]) for c in fig1.COHORTS], loc="lower left")


def plot_dprime_trials(ax, df):
    _plot_dprime(ax, df, "W trials from {}", "trial")


def plot_dprime_norm(ax, df):
    _plot_dprime(ax, df, "Norm. time from {}", "norm")


def plot_dprime_abs(ax, df):
    _plot_dprime(ax, df, "Time from {} (min)", "abs_min")


def _sig_line(ax, x, y, p, color):
    """User 2026-11-23 ('stats nicer, horizontal and smaller'): stars and p-value on one line, just above y."""
    ax.annotate(f"{fig1.stars(p)} {fig1.p_text(p)}", (x, y), xytext=(0, 0.8), textcoords="offset points", ha="center",
                va="bottom", fontsize=fig1.FS_NOTE, color=color, annotation_clip=False)


def plot_day0_dprime(ax, df):
    """User 2026-11-23: as the per-mouse lick-probability panel, d' per mouse on whisker day 0 (whisker and auditory
    against false alarms); dots, mice; black-edged mean +- s.e.m.; Mann-Whitney R+ vs R-."""
    ticks = []
    for i, tt in enumerate(("whisker_trial", "auditory_trial")):
        vals = {}
        for k, cohort in enumerate(fig1.COHORTS):
            v = df[(df.trial_type == tt) & (df.cohort == cohort)].dprime.dropna().to_numpy()
            x = i * 1.5 + k * 0.55
            j = np.random.default_rng(10 * i + k).uniform(-0.09, 0.09, len(v))
            ax.scatter(x + j, v, s=1.5, color=fig1.group_color(tt, cohort), lw=0, zorder=2)
            ax.errorbar(x + 0.2, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color="0.15", marker="o", ms=2,
                        mfc=fig1.group_color(tt, cohort), mec="0.15", mew=0.4, capsize=0, elinewidth=0.6, zorder=3)
            vals[cohort] = v
        top = max(vals["R+"].max(), vals["R-"].max()) + 0.15
        fig1._bracket(ax, i * 1.5, i * 1.5 + 0.55, top, fig1._mw_p(vals["R+"], vals["R-"]))
        ticks.append(i * 1.5 + 0.275)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.1 * (hi - lo))
    ax.set_xticks(ticks, ["W vs FA", "A vs FA"])
    ax.set_xlim(-0.4, 2.45)
    ax.set_ylabel("d′ (day 0)", labelpad=1)
    ax.set_xlabel("Trial types", labelpad=1)


def plot_xcorr(ax, df):
    """User 2026-11-23: as Fig. 1 plot_xcorr with Spearman rho instead of Pearson r and without the shift null; left,
    rho over lags, mean +- s.e.m.; right, rho at lag 0, bars mean +- s.e.m., Mann-Whitney."""
    left, right = fig1._split(ax, 2, gap=0.3)
    fig1._two_bars(right, df[df.lag == 0], "rho", "ρ at lag 0", individuals=False)
    right.axhline(0, color="0.6", lw=0.4)
    for cohort in fig1.COHORTS:
        fig1._mean_band(left, df[df.cohort == cohort], "lag", "rho", fig1.COLORS[cohort], lw=0.7)
    left.axvline(0, color="0.6", lw=0.4, ls=":")
    left.axhline(0, color="0.6", lw=0.4)
    left.set_xlim(df.lag.min(), df.lag.max())
    left.set_xticks([-10, -5, 0, 5, 10])
    left.set_xlabel("Lag (W trials)\n← FA leads | W leads →", labelpad=1)
    left.set_ylabel("ρ (W, FA curves)", labelpad=1)
    lo, hi = left.get_ylim()
    left.set_ylim(lo, hi + 0.2 * (hi - lo))
    fig1._legend(left, [(c.replace("-", "−"), fig1.COLORS[c]) for c in fig1.COHORTS], loc="upper left")


def plot_glm_shuffle_groups(ax, df):
    """User 2026-11-23: shuffle controls of the GLM, laid out as Fig. 1K: drop in held-out log-likelihood per trial when
    a predictor is shuffled (cohort colours, predictor names), then, in its own column on the right, when a group of
    predictors is shuffled together (diamonds, at the centre of the group's predictors; group names on the right)."""
    single, grp = df[df.metric == "shuffle_single"], df[df.metric == "shuffle_group"]
    order = single.drop_duplicates("feature").sort_values("rank").feature.tolist()
    pos, fig = ax.get_position(), ax.figure
    Wmm = pos.width * fig.get_figwidth() * 25.4
    ax.axis("off")
    widths, gap = (1.0, 0.7), 3.0
    unit = (Wmm - gap) / sum(widths)
    axes = [ax.inset_axes([0, 0, widths[0] * unit / Wmm, 1]),
            ax.inset_axes([(widths[0] * unit + gap) / Wmm, 0, widths[1] * unit / Wmm, 1])]
    fig1._feature_rows(axes[0], single, "value", order, "Δ LL/trial (nats)\n(predictor)")
    m = grp.groupby(["feature", "cohort"]).value
    lo = min(axes[0].get_xlim()[0], (m.mean() - m.sem()).min())
    hi = max(axes[0].get_xlim()[1], (m.mean() + m.sem()).max() * 1.6)
    for a in axes:
        a.set_xlim(lo, hi)
        a.set_ylim(len(order) + 1.3, -0.6)  # legend band under the predictors
    axes[1].axvline(0, color="0.6", lw=0.4, ls=":")
    groups = {g: mm.split(";") for g, mm in grp.drop_duplicates("feature")[["feature", "group_members"]].to_numpy()}
    fig1._group_overlay(axes[1], grp, order, groups, own_axes=True)
    axes[1].tick_params(axis="y", length=0)
    axes[1].set_xlabel("Δ LL/trial\n(group)", labelpad=1)
    for a in axes:
        a.set_xticks([0, 0.04], ["0", "0.04"])
    ax.set_title("Shuffled predictors", fontsize=fig1.FS_TITLE, pad=12)
    fig1._legend(axes[0], [("≠ 0", "0.3", dict(marker="o", ms=2, lw=0)),
                           ("n.s.", "0.3", dict(marker="o", ms=2, lw=0, mfc="white", mew=0.5))], loc="lower left",
                 ncol=2, columnspacing=0.6, borderaxespad=0.1)


def plot_rt_change_auditory(ax, df):
    """User 2026-11-22: auditory hits on the whisker-trial axis of Fig. 1K; median RT per mouse in the early (whisker
    trials 0-5 from the first whisker hit) and late (21-30) windows, two dots per cohort (mean +- s.e.m. over mice with
    both), grey lines, mice; stars, Wilcoxon of late - early within cohort (cohort colour); bracket, R+ vs R- on the
    difference (Mann-Whitney)."""
    from scipy import stats
    d = df[(df.trial_type == "auditory_trial") & df.paired.astype(bool)]
    diff = {}
    for k, c in enumerate(fig1.COHORTS):
        w = d[d.cohort == c].pivot_table(index="mouse_id", columns="window", values="median_rt_ms")[["early", "late"]]
        x = np.array([0, 1]) + k * 2.4
        for _, r in w.iterrows():
            ax.plot(x, r.to_numpy(), color="0.85", lw=0.3, zorder=1)
        color = fig1.group_color("auditory_trial", c)
        ax.errorbar(x, w.mean(), w.sem(), color=color, marker="o", ms=2.2, lw=0.8, elinewidth=0.6, capsize=0, zorder=3)
        diff[c] = (w.late - w.early).dropna()
        top = np.nanmax(w.to_numpy())  # above every mouse
        ax.plot(x, [top * 1.04] * 2, color=color, lw=0.4)
        p = stats.wilcoxon(diff[c]).pvalue  # narrow axes: stars only here, p in the caption
        ax.annotate(fig1.stars(p), (x.mean(), top * 1.04), xytext=(0, 0.8), textcoords="offset points", ha="center",
                    va="bottom", fontsize=fig1.FS_NOTE, color=color)
    ymax = ax.get_ylim()[1]
    y = ymax * 1.1
    ax.plot([0.5, 0.5, 2.9, 2.9], [y - ymax * 0.02, y, y, y - ymax * 0.02], color="k", lw=0.4)
    _sig_line(ax, 1.7, y, fig1._mw_p(diff["R+"], diff["R-"]), "k")
    ax.set_ylim(top=ymax * 1.22)
    ax.set_xticks([0, 1, 2.4, 3.4], ["0-5", "21-30", "0-5", "21-30"], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, 3.9)
    ax.set_xlabel("W trials from 1st W hit", labelpad=1)
    ax.set_ylabel("Auditory RT (ms)", labelpad=1)  # no legend: R+ left pair, R- right pair (caption; overlapped data)


plot_glm_schematic = fig1.plot_glm_schematic  # moved from Figure 1 (user 2026-11-22)
plot_glm_example = fig1.plot_glm_example
plot_glm_fit = fig1.plot_glm_fit
