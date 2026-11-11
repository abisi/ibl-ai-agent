"""Example_behaviour -- one plot_<id>(ax, df) per panel; each draws only from its panel's CSV."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paperstyle import COLORS, TRIAL_TYPE_COLORS, TRIAL_TYPE_LABELS, light  # noqa: E402

TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
COHORT_STYLE = {"R+": "-", "R-": "--"}


def _mean_sem(df, by, value):
    g = df.groupby(by)[value]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem()}).reset_index()


def _band(ax, x, m, s, color, ls="-", label=None):
    ax.plot(x, m, color=color, ls=ls, label=label)
    ax.fill_between(x, m - s, m + s, color=light(color, 0.8), lw=0, edgecolor="none", zorder=1)


def _p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_b(ax, df):
    rows = {"auditory_trial": 2, "whisker_trial": 1, "no_stim_trial": 0}
    for tt, d in df.groupby("trial_type"):
        y = np.full(len(d), rows[tt])
        lick = d.lick.to_numpy() == 1
        c = TRIAL_TYPE_COLORS[tt]
        ax.scatter(d.trial_number[lick], y[lick], s=4, marker="|", color=c, lw=0.6)
        ax.scatter(d.trial_number[~lick], y[~lick] - 0.3, s=1.5, marker="|", color=light(c, 0.5), lw=0.4)
    ax.set_yticks([0, 1, 2], [TRIAL_TYPE_LABELS[t] for t in sorted(rows, key=rows.get)])
    ax.set_ylim(-0.7, 2.5)
    ax.set_xlabel("Trial number")
    ax.set_title(f"Example R+ mouse ({df.mouse_id.iloc[0]}), whisker day 0: lick (tall) / no lick (short)")


def _blocks(ax, df, title):
    x = df.first_trial + 1
    for col, tt in (("hr_w", "whisker_trial"), ("hr_a", "auditory_trial"), ("hr_n", "no_stim_trial")):
        ax.plot(x, df[col], color=TRIAL_TYPE_COLORS[tt], marker="o", ms=1.5, label=TRIAL_TYPE_LABELS[tt])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Trial number (block start)")
    ax.set_ylabel("Lick probability")
    ax.set_title(title)


def plot_c(ax, df):
    _blocks(ax, df, f"Example R+ ({df.mouse_id.iloc[0]})")
    ax.legend(loc="center right", handlelength=1)


def plot_d(ax, df):
    _blocks(ax, df, f"Example R- ({df.mouse_id.iloc[0]})")


def plot_e(ax, df):
    for cohort, ls in COHORT_STYLE.items():
        for tt in TRIAL_TYPES:
            d = df[(df.cohort == cohort) & (df.trial_type == tt)]
            if d.empty:
                continue
            ms = _mean_sem(d, "day", "lick_probability")
            color = COLORS[cohort] if tt == "whisker_trial" else TRIAL_TYPE_COLORS[tt]
            ax.errorbar(ms.day + (0.06 if cohort == "R-" else -0.06), ms["mean"], ms["sem"], color=color, ls=ls,
                        marker="o", ms=2, capsize=0, elinewidth=0.5,
                        label=f"{TRIAL_TYPE_LABELS[tt]} {cohort}")
    ax.axvline(-0.5, color="0.6", lw=0.4, ls=":")
    ax.set_xticks(range(-2, 3))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Training day (0 = first whisker day)")
    ax.set_ylabel("Lick probability")
    ax.set_title("Mean ± s.e.m. over mice; whisker in cohort colour")
    ax.legend(ncol=2, loc="center right", bbox_to_anchor=(1, 0.52), handlelength=1.5, fontsize=4.5)


def plot_f(ax, df):
    for cohort, ls in COHORT_STYLE.items():
        d = df[df.cohort == cohort]
        for col, tt in (("hr_w", "whisker_trial"), ("hr_a", "auditory_trial"), ("hr_n", "no_stim_trial")):
            ms = _mean_sem(d, "block", col)
            color = COLORS[cohort] if tt == "whisker_trial" else TRIAL_TYPE_COLORS[tt]
            _band(ax, ms.block + 1, ms["mean"], ms["sem"], color, ls)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Block of 20 trials")
    ax.set_ylabel("Lick probability")
    ax.set_title("Whisker day 0, all mice")


def _two_groups(ax, df, value, ylabel, title):
    for i, cohort in enumerate(["R+", "R-"]):
        v = df[df.cohort == cohort][value].dropna()
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
        ax.scatter(i + jitter, v, s=3, color=light(COLORS[cohort], 0.5), lw=0)
        ax.errorbar(i + 0.3, v.mean(), v.sem(), color=COLORS[cohort], marker="o", ms=2.5, capsize=0, elinewidth=0.7)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_g(ax, df):
    _two_groups(ax, df, "whisker_trials_before_first_lick", "Whisker trials before first lick", "First whisker lick")


def plot_h(ax, df):
    x = df.whisker_trial
    ax.fill_between(x, df.p_low, df.p_high, color=light(COLORS["R+"], 0.7), lw=0, edgecolor="none")
    ax.plot(x, df.p_mean, color=COLORS["R+"], label="Whisker")
    ax.plot(x, df.p_chance, color=COLORS["no_stim"], lw=0.6, label="No stimulus (chance)")
    ax.scatter(x, df.lick * 1.06 - 0.03, s=1, marker="|", color="0.4", lw=0.4)
    ax.axvline(df.learning_trial.iloc[0], color="k", lw=0.5, ls="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Whisker trial")
    ax.set_ylabel("Lick probability (80% CI)")
    ax.set_title(f"Example R+ ({df.mouse_id.iloc[0]}), learning trial")
    ax.legend(loc="center right", handlelength=1)


def plot_i(ax, df):
    for cohort, ls in COHORT_STYLE.items():
        v = np.sort(df[(df.cohort == cohort) & df.learner].learning_trial)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[cohort],
                label=f"{cohort} (n = {len(v)})")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Learning trial (whisker trial)")
    ax.set_ylabel("Cumulative fraction of learners")
    ax.set_title("Learning trial")
    ax.legend(loc="lower right", handlelength=1)


def plot_j(ax, df):
    df = df.set_index("cohort").loc[["R+", "R-"]]
    ax.bar([0, 1], df.fraction_learners, color=[COLORS["R+"], COLORS["R-"]], width=0.6, lw=0)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, r.fraction_learners + 0.02, f"{r.learners}/{r.mice}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fraction of learners")
    ax.set_title("Learners (learning-trial rule)")


def plot_k(ax, df):
    for cohort, ls in COHORT_STYLE.items():
        ms = _mean_sem(df[df.cohort == cohort], "trial_from_learning_trial", "p_mean")
        _band(ax, ms.trial_from_learning_trial, ms["mean"], ms["sem"], COLORS[cohort], label=cohort)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Whisker trials from learning trial")
    ax.set_ylabel("Whisker lick probability")
    ax.set_title("Aligned to learning trial")
    ax.legend(loc="upper left", handlelength=1)


def plot_l(ax, df):
    bins = np.linspace(0, 1, 41)
    for tt in ["whisker_trial", "auditory_trial"]:
        for cohort, ls in COHORT_STYLE.items():
            v = df[(df.trial_type == tt) & (df.cohort == cohort)].reaction_time_s
            color = COLORS[cohort] if tt == "whisker_trial" else TRIAL_TYPE_COLORS[tt]
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, ls=ls, lw=0.7,
                    label=f"{TRIAL_TYPE_LABELS[tt]} hits {cohort}")
    ax.set_xlabel("Reaction time from response-window start (s)")
    ax.set_ylabel("Density (trials pooled)")
    ax.set_title("Reaction times, whisker day 0")
    ax.legend(handlelength=1.5)


def plot_m(ax, df):
    for tt in ["whisker_trial", "auditory_trial"]:
        for cohort, ls in COHORT_STYLE.items():
            d = df[(df.trial_type == tt) & (df.cohort == cohort)]
            if d.empty:
                continue
            ms = _mean_sem(d, "day", "median_reaction_time_s")
            color = COLORS[cohort] if tt == "whisker_trial" else TRIAL_TYPE_COLORS[tt]
            ax.errorbar(ms.day + (0.06 if cohort == "R-" else -0.06), ms["mean"], ms["sem"], color=color, ls=ls,
                        marker="o", ms=2, capsize=0, elinewidth=0.5, label=f"{TRIAL_TYPE_LABELS[tt]} {cohort}")
    ax.set_xticks(range(-2, 3))
    ax.set_xlabel("Training day")
    ax.set_ylabel("Median reaction time (s)")
    ax.set_title("Mean ± s.e.m. over mice")
    ax.legend(ncol=2, loc="upper right", handlelength=1.5, fontsize=4.5)


def plot_n(ax, df):
    from scipy import stats
    for cohort in ["R+", "R-"]:
        d = df[df.cohort == cohort]
        x, y = d.learning_trial.to_numpy(float), d.day1_whisker_lick_probability.to_numpy(float)
        ax.scatter(x, y, s=4, color=light(COLORS[cohort], 0.3), lw=0)
        if len(d) < 3:
            continue
        fit = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        resid = y - (fit.intercept + fit.slope * x)
        s = np.sqrt(resid @ resid / (len(x) - 2))
        se = s * np.sqrt(1 / len(x) + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
        tq = stats.t.ppf(0.975, len(x) - 2)
        yhat = fit.intercept + fit.slope * xs
        ax.fill_between(xs, yhat - tq * se, yhat + tq * se, color=light(COLORS[cohort], 0.85), lw=0, edgecolor="none", zorder=0)
        ax.plot(xs, yhat, color=COLORS[cohort], ls="-" if fit.pvalue < 0.05 else "--",
                label=f"{cohort}: r = {fit.rvalue:.2f}, {_p_text(fit.pvalue)}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Learning trial (day 0)")
    ax.set_ylabel("Whisker lick probability, day +1")
    ax.set_title("Learning speed vs next day")
    ax.legend(loc="lower right", handlelength=1, fontsize=4.5)
