"""Pilot: fit TCA independently on two mice, then match their components by
time-factor similarity + Hungarian assignment, and sanity-check the matches
by eye (overlaid time-factor shapes) before trusting the approach at scale.

TCA components are unordered and sign-ambiguous per mouse, and (unlike the
megamouse pipeline, which pools neurons across mice on a shared trial axis)
single-mouse tensors here share only the time axis across mice -- neuron
count differs per mouse, and while the trial axis has the same *length*
across mice (fixed pre/post window), trial N isn't the "same" trial for two
different mice the way time bin N is the same instant relative to stimulus
onset. So matching is restricted to the time mode, per user's chosen
approach.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensortools as tt
from scipy.optimize import linear_sum_assignment

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)

SESSION_IDS = ["AB116_20240724_102941", "AB117_20240723_125437"]
RANK = 4

sessions, units, trials = load_session_tables()


def fit_mouse(session_id):
    units_session = units[units["session_id"] == session_id].reset_index(drop=True)
    trials_session = trials[trials["session_id"] == session_id]
    aligned = align_whisker_trials_first_hit(trials_session)
    spike_times_by_cluster = load_session_spike_times(session_id, units_session)
    tensor, time_bins = build_spike_tensor(spike_times_by_cluster, units_session, aligned)
    tensor_min = tensor.min(axis=(0, 2), keepdims=True)
    tensor_max = tensor.max(axis=(0, 2), keepdims=True)
    tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)
    model = tt.ncp_hals(tensor_norm, rank=RANK, verbose=False)
    return model.factors[2], time_bins  # time-mode factors, shape (n_time_bins, RANK)


time_factors = {}
time_bins_by_mouse = {}
for sid in SESSION_IDS:
    time_factors[sid], time_bins_by_mouse[sid] = fit_mouse(sid)

mouse_a, mouse_b = SESSION_IDS
Ta, Tb = time_factors[mouse_a], time_factors[mouse_b]

# Both mice share the same time_window/bin_size, so their time axes line up
# bin-for-bin; each time factor is z-scored before correlating so matching
# depends on shape, not overall scale (CP factor scale is arbitrary anyway --
# it trades off against the other two modes' norms).
Ta_z = (Ta - Ta.mean(axis=0)) / Ta.std(axis=0)
Tb_z = (Tb - Tb.mean(axis=0)) / Tb.std(axis=0)
n_bins = Ta_z.shape[0]
similarity = (Ta_z.T @ Tb_z) / n_bins  # shape (RANK, RANK), Pearson r per pair

row_ind, col_ind = linear_sum_assignment(-similarity)  # maximize similarity

print(f"matching {mouse_a} <-> {mouse_b}")
for i, j in zip(row_ind, col_ind):
    print(f"  {mouse_a} comp {i + 1}  <->  {mouse_b} comp {j + 1}   r={similarity[i, j]:+.3f}")

fig, axes = plt.subplots(1, RANK, figsize=(3.2 * RANK, 3), sharey=True)
time_centers = (time_bins_by_mouse[mouse_a][:-1] + time_bins_by_mouse[mouse_a][1:]) / 2
for k, (i, j) in enumerate(zip(row_ind, col_ind)):
    ax = axes[k]
    ax.plot(time_centers, Ta_z[:, i], color="k", lw=1.5, label=f"{mouse_a} c{i + 1}")
    ax.plot(time_centers, Tb_z[:, j], color="tab:orange", lw=1.5, label=f"{mouse_b} c{j + 1}")
    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title(f"matched pair {k + 1}\nr={similarity[i, j]:+.2f}", fontsize=9)
    ax.set_xlabel("time from stim onset (s)")
    ax.legend(fontsize=6)
axes[0].set_ylabel("time factor (z-scored)")
fig.suptitle("Cross-mouse time-factor matching (Hungarian assignment)", y=1.03)
fig.tight_layout()
fig.savefig("004_cross_mouse_matching.png", dpi=150, bbox_inches="tight")
print("saved 004_cross_mouse_matching.png")
