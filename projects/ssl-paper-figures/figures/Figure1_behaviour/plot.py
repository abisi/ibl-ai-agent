"""Figure1_behaviour -- one plot_<id>(ax, df) per panel; each draws only from its panel's CSV."""

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paperstyle import COLORS, TRIAL_TYPE_COLORS, light  # noqa: E402

COHORT_LS = {"R+": "-", "R-": "--"}
STATE_COLOR = {"high": None, "low_sated": "0.88"}  # high: light cohort colour


def _example(ax, df, title, first=False):
    cohort = df.cohort.iloc[0]
    color = COLORS[cohort]
    x = df.whisker_trial.to_numpy()
    # performance states as background bands (high: light cohort colour; low-sated: grey)
    for state, c in (("high", light(color, 0.85)), ("low_sated", STATE_COLOR["low_sated"])):
        on = (df.state == state).to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            ax.axvspan(x[start] - 0.5, x[stop - 1] + 0.5, ymin=0.06, ymax=0.94, color=c, lw=0, zorder=0)
    ax.fill_between(x, df.p_low, df.p_high, color=light(color, 0.6), lw=0, zorder=1)
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
    ax.set_xlabel("Whisker trial")
    if first:  # the four examples share the y axis; label it once
        ax.set_ylabel("Lick probability")
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{title} ({df.mouse_id.iloc[0]})", fontsize=5.5)


def plot_e(ax, df):
    _example(ax, df, "R+ learner", first=True)


def plot_f(ax, df):
    _example(ax, df, "R+ non-learner")


def plot_g(ax, df):
    _example(ax, df, "R- learner")


def plot_h(ax, df):
    _example(ax, df, "R- non-learner")


def _two_groups(ax, df, value, ylabel):
    for i, cohort in enumerate(["R+", "R-"]):
        v = df[df.cohort == cohort][value].dropna()
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
        ax.scatter(i + jitter, v, s=2.5, color=light(COLORS[cohort], 0.5), lw=0)
        ax.errorbar(i + 0.3, v.mean(), v.sem(), color=COLORS[cohort], marker="o", ms=2.5, capsize=0, elinewidth=0.7)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel(ylabel)


def plot_b(ax, df):
    _two_groups(ax, df, "n_pretraining_trials", "Pretraining trials")
    ax.set_ylim(bottom=0)


def plot_d(ax, df):
    bins = np.linspace(0, 1, 26)
    for tt in ["whisker_trial", "auditory_trial", "no_stim_trial"]:
        for cohort, ls in COHORT_LS.items():
            v = df[(df.trial_type == tt) & (df.cohort == cohort)].reaction_time_s
            color = COLORS[cohort] if tt == "whisker_trial" else TRIAL_TYPE_COLORS[tt]
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, ls=ls, lw=0.6)
    handles = [mpl.lines.Line2D([], [], color=COLORS["R+"], lw=0.6, label="Whisker R+"),
               mpl.lines.Line2D([], [], color=COLORS["R-"], lw=0.6, ls="--", label="Whisker R-"),
               mpl.lines.Line2D([], [], color=TRIAL_TYPE_COLORS["auditory_trial"], lw=0.6, label="Auditory"),
               mpl.lines.Line2D([], [], color=TRIAL_TYPE_COLORS["no_stim_trial"], lw=0.6, label="FA")]
    ax.legend(handles=handles, loc="upper right", fontsize=4.5, handlelength=1.2, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Reaction time (s)")
    ax.set_ylabel("Density")


def plot_i(ax, df):
    df = df.set_index("cohort").loc[["R+", "R-"]]
    ax.bar([0, 1], df.fraction_learners, color=[COLORS["R+"], COLORS["R-"]], width=0.6, lw=0)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, r.fraction_learners + 0.02, f"{r.learners}/{r.mice}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])  # narrow labels: the gap to the panel on the left is 7 mm
    ax.set_ylabel("Fraction learners", labelpad=1)


def _mean_sem(df, by, value):
    g = df.groupby(by)[value]
    return g.mean(), g.sem()


def plot_c(ax, df):
    for tt in ["whisker_trial", "auditory_trial", "no_stim_trial"]:
        for cohort, ls in COHORT_LS.items():
            d = df[(df.trial_type == tt) & (df.cohort == cohort)]
            if d.empty:
                continue
            m, s = _mean_sem(d, "day", "lick_probability")
            color = COLORS[cohort] if tt == "whisker_trial" else TRIAL_TYPE_COLORS[tt]
            ax.errorbar(m.index + (0.04 if cohort == "R-" else -0.04), m, s, color=color, ls=ls, marker="o", ms=2,
                        capsize=0, elinewidth=0.6, lw=0.8)
    ax.set_xticks([-1, 0], ["Last auditory day", "Whisker day 0"])
    ax.set_xlim(-1.4, 0.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Lick probability")


def plot_j(ax, df):
    df = df[df.state.isin(["low", "high"])]
    for i, tt in enumerate(["whisker_trial", "auditory_trial", "no_stim_trial"]):
        for k, cohort in enumerate(["R+", "R-"]):
            d = df[(df.trial_type == tt) & (df.cohort == cohort)]
            m, s = _mean_sem(d, "state", "lick_probability")
            x = i + np.array([-0.3, -0.1]) + 0.4 * k
            ax.plot(x, [m.get("low", np.nan), m.get("high", np.nan)], color=COLORS[cohort], lw=0.6, zorder=1)
            for xi, state in zip(x, ["low", "high"]):
                ax.errorbar(xi, m.get(state, np.nan), s.get(state, np.nan), color=COLORS[cohort], marker="o", ms=2.2,
                            mfc="white" if state == "low" else COLORS[cohort], capsize=0, elinewidth=0.6, lw=0)
    ax.set_xticks([0, 1, 2], ["W", "A", "FA"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("Lick probability", labelpad=1)


def _cumulative(ax, df, x, xlabel, label):
    for cohort in ["R+", "R-"]:
        m, s = _mean_sem(df[df.cohort == cohort], x, "cumulative_fa")
        ax.fill_between(m.index, m - s, m + s, color=light(COLORS[cohort], 0.7), lw=0)
        ax.plot(m.index, m, color=COLORS[cohort], lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_xlim(1, df[x].max())
    ax.set_ylim(0, 13)  # same range in both cumulative FA panels
    ax.set_yticks([0, 5, 10])
    ax.text(0.05, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=5.5)


def plot_l(ax, df):
    _cumulative(ax, df, "whisker_trial_after_anchor", "Whisker trial", "Day 0, from 1st\nwhisker lick")
    ax.set_ylabel("Cumulative FA")


def plot_m(ax, df):
    _cumulative(ax, df, "no_stim_trial_after_anchor", "No-stim trial", "Day -1, from\nmatched reward")
    ax.set_yticklabels([])


def plot_n(ax, df):
    _two_groups(ax, df, "delta_fa", "Δ early FA rate")
    ax.axhline(0, color="0.6", lw=0.4, ls=":")


def plot_k(ax, df):
    for cohort in ["R+", "R-"]:
        m, s = _mean_sem(df[df.cohort == cohort], "session_time_norm", "performance")
        ax.fill_between(m.index, m - s, m + s, color=light(COLORS[cohort], 0.7), lw=0)
        ax.plot(m.index, m, color=COLORS[cohort], lw=0.8, label=cohort)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Session time (normalised)")
    ax.set_ylabel("Whisker - chance")
    ax.legend(loc="upper left", fontsize=5, frameon=False, handlelength=1)


MOVEMENT_LABELS = {"jaw_motion": "Jaw", "nose_motion": "Nose", "whisker_motion": "Whisker", "pupil_area": "Pupil"}


def plot_r(ax, df):
    _two_groups(ax, df, "whisker_trials_before_first_lick", "W trials before 1st lick")
    ax.set_ylim(bottom=0)


def plot_o(ax, df):
    for cohort in ["R+", "R-"]:
        d = df[df.cohort == cohort]
        m, s = _mean_sem(d, "lag", "r")
        ax.fill_between(m.index, m - s, m + s, color=light(COLORS[cohort], 0.7), lw=0)
        ax.plot(m.index, m, color=COLORS[cohort], lw=0.8)
        null = d[d.lag == 0].null_mean_lag0.mean()
        ax.axhline(null, color=light(COLORS[cohort], 0.4), lw=0.5, ls="--")
    ax.axvline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xlabel("Lag (whisker trials)")
    ax.set_ylabel("r (whisker, FA curves)", labelpad=1)
    ax.set_xlim(df.lag.min(), df.lag.max())


def plot_p(ax, df):
    x = df.whisker_trial.to_numpy()
    color = COLORS[df.cohort.iloc[0]]
    ax.plot(x, df.p_mean, color=color, lw=0.8)
    ax.plot(x, df.p_chance, color=COLORS["no_stim"], lw=0.6)
    ticks = [0.5]
    labels = ["Lick prob."]
    for k, c in enumerate(["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]):
        base = -1.0 - 1.1 * k
        z = df[f"{c}_z"].clip(-3, 3).to_numpy() / 6  # +-3 SD in +-0.5
        ax.plot(x, base + z, color="0.25", lw=0.5)
        ticks.append(base)
        labels.append(MOVEMENT_LABELS[c])
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, x.max() + 1)
    ax.set_xlabel("Whisker trial")



def plot_q(ax, df):
    for i, c in enumerate(["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]):
        for k, cohort in enumerate(["R+", "R-"]):
            v = df[(df.movement == c) & (df.cohort == cohort)].r_minus_null.dropna()
            xc = i + (-0.17 if k == 0 else 0.17)
            jitter = np.random.default_rng(i).uniform(-0.06, 0.06, len(v))
            ax.scatter(xc + jitter, v, s=1.5, color=light(COLORS[cohort], 0.5), lw=0)
            ax.errorbar(xc, v.mean(), v.sem(), color=COLORS[cohort], marker="o", ms=2.2, capsize=0, elinewidth=0.7)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xticks(range(4), [MOVEMENT_LABELS[c] for c in ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("r - null", labelpad=1)


GLM_LABELS = {"lick_whisker_trial_n-1": "Lick n-1", "lick_whisker_trial_n-2": "Lick n-2",
              "lick_whisker_trial_n-3": "Lick n-3", "spontaneous_lick_rate": "Spont. licks",
              "time_since_reward": "Time since rew.", "cumulative_reward_fraction": "Cum. reward",
              "jaw_speed_pre": "Jaw", "nose_speed_pre": "Nose", "whisker_speed_pre": "Whisker",
              "pupil_area_pre": "Pupil", "all_movement": "All mvt"}


def plot_t(ax, df):
    x = df.whisker_trial
    lick = df.lick.to_numpy() == 1
    ax.scatter(x[lick], np.full(lick.sum(), 1.06), s=2, marker="|", lw=0.4, color=COLORS["R+"])
    ax.scatter(x[~lick], np.full((~lick).sum(), -0.06), s=2, marker="|", lw=0.4, color=COLORS["R-"])
    ax.plot(x, df.observed_rolling10, color="0.6", lw=0.6, label="Data")
    ax.plot(x, df.predicted_p_lick, color=COLORS[df.cohort.iloc[0]], lw=0.8, label="GLM")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Whisker trial")
    ax.set_ylabel("P(lick)", labelpad=1)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=4.5, frameon=False, handlelength=1)


def plot_u(ax, df):
    _two_groups(ax, df, "test_auc", "Test AUC")
    ax.yaxis.labelpad = 1
    ax.axhline(0.5, color="0.6", lw=0.4, ls=":")


def _feature_points(ax, df, value, order, ylabel):
    for i, f in enumerate(order):
        for k, cohort in enumerate(["R+", "R-"]):
            v = df[(df.feature == f) & (df.cohort == cohort)][value].dropna()
            ax.errorbar(i + (-0.15 if k == 0 else 0.15), v.mean(), v.sem(), color=COLORS[cohort], marker="o", ms=2,
                        capsize=0, elinewidth=0.6, lw=0)
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xticks(range(len(order)), [GLM_LABELS.get(f, f) for f in order], rotation=90)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(ylabel, labelpad=1)


def plot_v(ax, df):
    order = df.drop_duplicates("feature").sort_values("rank_by_mean_abs").feature.tolist()
    _feature_points(ax, df, "coefficient", order, "GLM weight")


def plot_w(ax, df):
    order = (df.groupby("feature").delta_test_auc.mean().sort_values(ascending=False).index.tolist())
    _feature_points(ax, df, "delta_test_auc", order, "\u0394 AUC (shuffled)")
