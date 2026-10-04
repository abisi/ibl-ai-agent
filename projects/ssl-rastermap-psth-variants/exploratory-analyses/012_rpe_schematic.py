"""Schematic: expected activity of reward-prediction-error (RPE) neurons per cohort (idealised TD-model predictions).

Timeline per trial (stimulus-aligned; values are illustrative):
    stimulus (0) -> first lick = action that triggers delivery (~0.3 s) -> reward contact at ~the 2nd lick
    (+~0.14 s, one inter-lick interval). Spontaneous licks: aligned so their first lick sits at the same time.
Contingencies: auditory hit rewarded in both cohorts (well learned -> reward predicted by the tone);
whisker hit rewarded in R+ only (day 0: initially unpredicted); R- whisker hits are unrewarded (omission if the
mouse expects water after licking, e.g. by generalising from auditory trials); spontaneous licks unrewarded.
Assumption: withholding on R- whisker trials is not rewarded with water.
    RPE+ neuron  = dopamine-like: burst for better-than-expected, dip for worse-than-expected
    RPE- neuron  = sign-inverted (e.g. lateral-habenula-like): burst for worse, dip for better
Early vs late in the session: the RPE at reward shrinks and moves to the earliest predictor (stimulus / action).
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rpe_roc_pilot")
T_STIM, T_L1, T_REW = 0.0, 0.30, 0.44
t = np.linspace(-0.15, 0.8, 800)
COL = {"WH early": "#d4a017", "WH late": "#8a6a00", "AH": "#1f4fbf", "SP": "0.55"}


def bump(t0, amp, w=0.025):
    return amp * np.exp(-0.5 * ((t - t0) / w) ** 2)


def rpe(stim=0.0, action=0.0, reward=0.0):
    """signed RPE time course: phasic events at stimulus, first lick, reward contact"""
    return 1.0 + bump(T_STIM + 0.05, stim) + bump(T_L1 + 0.03, action) + bump(T_REW + 0.05, reward, 0.03)


# (stim, action, reward) signed RPE per cohort x condition
PRED = {
    "R+": {"WH early": (0.1, 0.3, 1.6),     # unexpected reward after whisker hit
           "WH late": (0.9, 0.4, 0.4),      # prediction moved to stimulus/action, reward largely predicted
           "AH": (1.2, 0.2, 0.2),           # tone predicts reward; small RPE at receipt
           "SP": (0.0, 0.0, 0.0)},          # unrewarded, not expected
    "R-": {"WH early": (0.1, 0.3, -0.8),    # expected water after licking -> omission dip
           "WH late": (0.0, 0.0, -0.2),     # expectation extinguished; few whisker licks anyway
           "AH": (1.2, 0.2, 0.2),
           "SP": (0.0, 0.0, 0.0)},
}
NOTES = {
    "R+": "whisker hit: large burst at reward contact early;\nlate, response shifts to stimulus / first lick",
    "R-": "whisker hit: dip at the expected reward contact\n(omission) early; fades as expectation extinguishes",
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2), dpi=250, sharex=True)
    for c, cohort in enumerate(["R+", "R-"]):
        for r, (ntype, sign) in enumerate([("RPE+ neuron (dopamine-like)", 1), ("RPE- neuron (sign-inverted)", -1)]):
            ax = axes[r, c]
            for cond, (s, a, w) in PRED[cohort].items():
                y = 1 + sign * (rpe(s, a, w) - 1)
                ls = "--" if cond == "WH late" else "-"
                lab = {"WH early": "whisker hit, early", "WH late": "whisker hit, late", "AH": "auditory hit",
                       "SP": "spontaneous lick"}[cond]
                ax.plot(t, y, color=COL[cond], lw=1.4, ls=ls, label=lab)
            for x, txt, ha in ((T_STIM, "stimulus", "center"), (T_L1, "1st lick\n(action) ", "right"),
                               (T_REW, " reward contact\n (~2nd lick)", "left")):
                ax.axvline(x, color="0.3", lw=0.6, ls=":")
                if r == 0:
                    ax.text(x, 2.95, txt, ha=ha, va="bottom", fontsize=6.5)
            ax.axvspan(T_REW, T_REW + 0.15, color="#fdebd0", zorder=0, lw=0)
            ax.axhline(1, color="0.8", lw=0.5)
            ax.set_ylim(-0.8, 2.9); ax.set_yticks([1]); ax.set_yticklabels(["baseline"])
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if r == 0:
                ax.set_title(f"{cohort} cohort\n\n", fontsize=9)
            if c == 0:
                ax.set_ylabel(ntype, fontsize=8)
            if r == 1:
                ax.set_xlabel("time from stimulus (s)")
            if r == 0:
                ax.text(0.99, 0.02, NOTES[cohort], transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                        color="0.25")
    axes[0, 1].legend(fontsize=6.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.text(0.01, 0.005,
             "Idealised TD predictions. Shaded = receipt window used for the ROC tests. Spontaneous licks are "
             "aligned so that their first lick coincides with the hits' first lick. Assumes no water for withholding.",
             fontsize=6, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "rpe_schematic.png"); fig.savefig(OUT / "rpe_schematic.pdf")
    print("saved", OUT / "rpe_schematic.png")
