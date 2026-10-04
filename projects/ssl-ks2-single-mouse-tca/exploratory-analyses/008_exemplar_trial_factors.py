"""Gallery: for a curated set of example mice, show each TCA trial-mode
component's loading trace alongside the mouse's own raw outcome (lick_flag
per trial) and smoothed P(lick), on the same trial axis.

This is the same relationship the circular-shift test (003/005) already
quantified as an r and a p-value -- this script exists to show it, not to
add a new statistic. 8 mice are hand-picked from the full 68 to cover a
range of stories: strong P(lick)-driven mice, strong d-prime-driven mice,
both reward-group cohorts, and (for honesty) one mouse with no significant
component at all. This is a curated "best/representative examples"
selection, not a random sample -- the unbiased population-level picture is
the 55/68 and enrichment statistics in the main report, not this gallery.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tca_lib import load_session_tables, align_whisker_trials_first_hit
from behavior_lib import smoothed_plick, rolling_dprime

ARTIFACTS_DIR = Path("../artifacts/per_mouse")
RANK = 4
COMPONENT_COLORS = ["#1f6f78", "#b8433f", "#c98a2c", "#5b4b8a"]

# Hand-picked for a range of stories (see docstring); (session_id, reason).
EXEMPLARS = [
    ("MH069_20260122_111455", "strongest overall: P(lick) AND d-prime both significant"),
    ("AB121_20240813_125401", "R+, strong P(lick)-driven"),
    ("AB126_20240822_114405", "R-, strong P(lick)-driven"),
    ("MH065_20260114_154021", "R-, strong P(lick)-driven"),
    ("AB159_20250409_135813", "R-, strongest d-prime-driven example"),
    ("MH039_20250525_112720", "R-, strong d-prime-driven"),
    ("MH030_20250501_151231", "R+, strong d-prime-driven"),
    ("AB117_20240723_125437", "R+, no significant component (honest negative example)"),
]

sessions, units, trials = load_session_tables()

fig, axes = plt.subplots(len(EXEMPLARS), 2, figsize=(12, 2.5 * len(EXEMPLARS)))

for row, (session_id, reason) in enumerate(EXEMPLARS):
    d = np.load(ARTIFACTS_DIR / f"{session_id}.npz", allow_pickle=True)
    trial_factors = d["trial_factors"]  # (n_trials, RANK)
    reward_group = int(d["reward_group"])
    corr = pd.DataFrame({
        "component": d["corr_component"], "behavior": d["corr_behavior"],
        "r": d["corr_r"], "p": d["corr_p"],
    })

    trials_session = trials[trials["session_id"] == session_id]
    aligned = align_whisker_trials_first_hit(trials_session)
    plick = smoothed_plick(aligned, window=10)
    dprime = rolling_dprime(trials_session, aligned, window_trials=20)
    x = aligned["whisker_trial_id"].to_numpy()

    cohort_label = "R+" if reward_group == 1 else "R-"

    # Left panel: raw outcome + smoothed P(lick) + rolling d-prime (secondary axis).
    ax = axes[row, 0]
    ax.scatter(x, aligned["lick_flag"], s=8, color="0.75", zorder=1, label="outcome (lick_flag)")
    ax.plot(x, plick, color="k", lw=1.6, zorder=2, label="P(lick), smoothed")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel(f"{session_id}\n({cohort_label})\nP(lick) / outcome", fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(x, dprime, color="#1f6f78", lw=1.2, ls=":", zorder=2, label="d-prime")
    ax2.set_ylabel("d-prime", fontsize=8, color="#1f6f78")
    ax2.tick_params(axis="y", labelcolor="#1f6f78", labelsize=7)
    if row == 0:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower right")
    if row == len(EXEMPLARS) - 1:
        ax.set_xlabel("whisker trial id (0 = first active hit)")

    # Right panel: all 4 z-scored trial factors, significant ones bolded and
    # annotated with which behavior they track.
    ax = axes[row, 1]
    tf_z = (trial_factors - trial_factors.mean(axis=0)) / trial_factors.std(axis=0)
    sig_labels = []
    for comp in range(RANK):
        comp_corr = corr[corr["component"] == comp + 1]
        is_sig = (comp_corr["p"] < 0.05).any()
        lw = 2.2 if is_sig else 0.9
        alpha = 1.0 if is_sig else 0.35
        ax.plot(x, tf_z[:, comp], color=COMPONENT_COLORS[comp], lw=lw, alpha=alpha, label=f"comp {comp + 1}")
        if is_sig:
            hit = comp_corr[comp_corr["p"] < 0.05].sort_values("p").iloc[0]
            sig_labels.append(f"c{comp + 1}→{hit['behavior']} r={hit['r']:+.2f}")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("trial factor\n(z-scored)", fontsize=8.5)
    title = ", ".join(sig_labels) if sig_labels else "no component clears p<0.05"
    ax.set_title(title, fontsize=8, loc="left")
    if row == 0:
        ax.legend(fontsize=6.5, loc="lower right", ncol=4)
    if row == len(EXEMPLARS) - 1:
        ax.set_xlabel("whisker trial id (0 = first active hit)")

fig.tight_layout()
fig.savefig("008_exemplar_trial_factors.png", dpi=150, bbox_inches="tight")
print("saved 008_exemplar_trial_factors.png")
