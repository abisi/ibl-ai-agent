"""Pilot: build one first-hit-aligned spike tensor for a single mouse from
ssl_ks2_ephys, and sanity-check it before scaling to all 68 eligible mice.

Question this answers: does the single-mouse tensor-build adapter (tca_lib.py)
reproduce what the megamouse pipeline's own sanity plots show -- a clear
passive-to-active P(lick) step, aligned at the first hit (whisker_trial_id=0)
-- and how long does one mouse take, to estimate the cost of running all 68.

Pilot mouse: AB116_20240724_102941 (used as an example session in the prior
ssl-task-performance project; 162 active + 44 passive whisker trials, well
above the >=80 active / >=1 passive threshold used to define eligibility).
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)

SESSION_ID = "AB116_20240724_102941"
TRIAL_WINDOW_PRE = 30
TRIAL_WINDOW_POST = 50
TIME_WINDOW = (-0.1, 0.2)
BIN_SIZE = 0.01

t_start = time.time()

sessions, units, trials = load_session_tables()
units_session = units[units["session_id"] == SESSION_ID].reset_index(drop=True)
trials_session = trials[trials["session_id"] == SESSION_ID]

aligned = align_whisker_trials_first_hit(trials_session, TRIAL_WINDOW_PRE, TRIAL_WINDOW_POST)
spike_times_by_cluster = load_session_spike_times(SESSION_ID, units_session)

t_build_start = time.time()
tensor, time_bins = build_spike_tensor(
    spike_times_by_cluster, units_session, aligned,
    time_window=TIME_WINDOW, bin_size=BIN_SIZE,
)
t_build = time.time() - t_build_start

print(f"session={SESSION_ID}")
print(f"n_neurons={len(units_session)}, n_trials={len(aligned)}, n_time_bins={tensor.shape[2]}")
print(f"tensor shape: {tensor.shape}")
print(f"mean spike count per bin: {tensor.mean():.4f} (spikes/{BIN_SIZE*1000:.0f}ms bin)")
print(f"tensor build time: {t_build:.1f}s")
print(f"total pilot time (incl. table/spike loading): {time.time() - t_start:.1f}s")
print(f"estimated time for all 68 mice (build step only, linear scaling): {t_build * 68 / 60:.1f} min")

# Sanity-check plot: mirrors the megamouse pipeline's own "megamouse_aligned_perf"
# plot, but for a single mouse -- P(lick) should show low passive-period licking,
# then jump toward criterion performance around the alignment trial (id=0),
# since alignment is defined by the first active hit.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

ax = axes[0]
ax.plot(aligned["whisker_trial_id"], aligned["lick_flag"], "o", ms=3, color="0.6", label="single trial")
ax.axvline(0, color="crimson", lw=1.5, label="first active hit")
context_change = aligned.index[aligned["context"] == "active"]
if len(context_change):
    ax.axvline(aligned["whisker_trial_id"].iloc[context_change[0]], color="k", ls="--", lw=1,
               label="passive->active")
ax.set_xlabel("whisker trial id (0 = first active hit)")
ax.set_ylabel("lick_flag")
ax.set_title(f"{SESSION_ID}\nn_trials={len(aligned)}")
ax.legend(fontsize=7, loc="center left")

# Population mean spike count aligned to stimulus onset, to check the tensor
# itself (not just the trial table) shows a sensible sensory-evoked bump.
ax = axes[1]
time_centers = (time_bins[:-1] + time_bins[1:]) / 2
mean_psth = tensor.mean(axis=(0, 1)) / BIN_SIZE  # -> spikes/sec, averaged over trials & neurons
ax.plot(time_centers, mean_psth, color="k", lw=1.5)
ax.axvline(0, color="crimson", lw=1, ls="--")
ax.set_xlabel("time from whisker stim onset (s)")
ax.set_ylabel("population mean firing rate (Hz)")
ax.set_title("tensor sanity check")

fig.tight_layout()
fig.savefig("000_single_mouse_tensor_build.png", dpi=150, bbox_inches="tight")
print("saved 000_single_mouse_tensor_build.png")
