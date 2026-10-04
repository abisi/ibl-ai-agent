"""Single-mouse trial-by-trial task performance during the active epoch.

Question: plot smoothed auditory hit rate, whisker hit rate, and false alarm
rate (no-stim trials) trial-by-trial during the active task, one panel per
mouse, from the ssl_ephys dataset (AB116/AB117/AB119/AB120 ephys sessions).

Outcome classification (validated against ground-truth event categories in
events.parquet -- auditory_hit_trial, whisker_miss_trial, etc.):
- auditory_trial / whisker_trial: lick_flag == 1 -> hit, == 0 -> miss
- no_stim_trial: lick_flag == 1 -> false alarm, == 0 -> correct rejection

Trials are restricted to the "active" epoch (per epochs.parquet) before
computing rates, since passive_pre/passive_post trials are not part of the
active task. Rates are smoothed with a trailing rolling mean over each trial
type's own occurrences (window=10), plotted against real trial position in
the active-epoch trial sequence.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ibl_ai_agent.data_locations import resolve_dataset_dir

WINDOW = 10  # trials, rolling mean over each trial type's own occurrences
MIN_PERIODS = 3

d = resolve_dataset_dir("ssl_ephys")
trials = pd.read_parquet(d / "metadata/trials.parquet")
epochs = pd.read_parquet(d / "metadata/epochs.parquet")
sessions = pd.read_parquet(d / "metadata/sessions.parquet")

ephys_sessions = sessions.loc[sessions["has_ephys"], "session_id"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.ravel()

for ax, session_id in zip(axes, ephys_sessions):
    subject_id = sessions.loc[sessions["session_id"] == session_id, "subject_id"].iloc[0]

    active = epochs[(epochs["session_id"] == session_id) & (epochs["epoch_name"] == "active")]
    t = trials[trials["session_id"] == session_id].sort_values("start_time").reset_index(drop=True)
    if len(active):
        lo, hi = active["start_time"].iloc[0], active["stop_time"].iloc[0]
        t = t[(t["start_time"] >= lo) & (t["start_time"] <= hi)].reset_index(drop=True)
    t["trial_index"] = np.arange(len(t))

    curves = {
        "auditory hit rate": ("auditory_trial", "tab:green", "-"),
        "whisker hit rate": ("whisker_trial", "tab:blue", "-"),
        "false alarm rate": ("no_stim_trial", "tab:red", "--"),
    }
    for label, (trial_type, color, ls) in curves.items():
        sub = t[t["trial_type"] == trial_type]
        smoothed = sub["lick_flag"].rolling(WINDOW, min_periods=MIN_PERIODS).mean()
        ax.plot(sub["trial_index"], smoothed, color=color, ls=ls, lw=1.6, label=label)

    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{subject_id} ({session_id})\nn_trials(active)={len(t)}", fontsize=9)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")

axes[0].legend(loc="upper right", fontsize=8)
for ax in axes[2:]:
    ax.set_xlabel("trial index (active epoch)")
for ax in axes[::2]:
    ax.set_ylabel("rate")

fig.suptitle(
    f"Active-task performance per mouse (rolling mean, window={WINDOW} trials of each type)",
    y=1.02,
)
fig.tight_layout()
fig.savefig("000_active_task_performance.png", dpi=150, bbox_inches="tight")
print("saved 000_active_task_performance.png")
