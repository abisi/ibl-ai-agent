"""Robustness check on rank selection: the pilot (001) only ran the
rank-selection sweep on one mouse (AB116, 1717 neurons). Here it's repeated
on 8 mice spanning the full neuron-count range in the cohort (65 to 2755
neurons), to see whether rank 4's "stable from rank 3 up" signature holds
consistently, or is a property of the one pilot mouse's particular size/
noise level.
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensortools as tt

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)

RANKS = range(1, 9)
N_REPLICATES = 5
MICE = [
    "MH007_20250202_165003", "AB120_20240811_143102", "MH036_20250515_111838",
    "AB142_20241128_113227", "AB139_20241119_115957", "MH035_20250514_142713",
    "AB157_20250412_154744", "AB130_20240902_123634",
]

sessions, units, trials = load_session_tables()
results = {}
t0 = time.time()
for session_id in MICE:
    units_session = units[units["session_id"] == session_id].reset_index(drop=True)
    trials_session = trials[trials["session_id"] == session_id]
    aligned = align_whisker_trials_first_hit(trials_session)
    spike_times_by_cluster = load_session_spike_times(session_id, units_session)
    tensor, time_bins = build_spike_tensor(spike_times_by_cluster, units_session, aligned)
    tensor_min, tensor_max = tensor.min(axis=(0, 2), keepdims=True), tensor.max(axis=(0, 2), keepdims=True)
    tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)

    ensemble = tt.Ensemble(fit_method="ncp_hals")
    ensemble.fit(tensor_norm, ranks=RANKS, replicates=N_REPLICATES)
    results[session_id] = {
        "n_neurons": tensor.shape[1],
        "error": [ensemble.objectives(r) for r in RANKS],  # normalized reconstruction error, directly
        "similarity": [ensemble.similarities(r) for r in RANKS],
    }
    print(f"{session_id} ({tensor.shape[1]} neurons) done, {time.time() - t0:.0f}s elapsed")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
cmap = plt.cm.viridis(np.linspace(0, 1, len(MICE)))
for color, session_id in zip(cmap, MICE):
    r = results[session_id]
    mean_err = [np.mean(e) for e in r["error"]]
    mean_sim = [np.mean(s) for s in r["similarity"]]
    axes[0].plot(list(RANKS), mean_err, color=color, lw=1.5, marker="o", ms=3, label=f"{session_id.split('_')[0]} (n={r['n_neurons']})")
    axes[1].plot(list(RANKS), mean_sim, color=color, lw=1.5, marker="o", ms=3)

axes[0].set_xlabel("model rank")
axes[0].set_ylabel("normalized reconstruction error")
axes[0].set_title("Reconstruction error vs. rank, 8 mice")
axes[1].set_xlabel("model rank")
axes[1].set_ylabel("replicate similarity")
axes[1].set_title("Replicate similarity vs. rank, 8 mice")
axes[0].legend(fontsize=6.5, loc="upper right")
axes[0].axvline(4, color="crimson", lw=1, ls="--", zorder=0)
axes[1].axvline(4, color="crimson", lw=1, ls="--", zorder=0)
fig.tight_layout()
fig.savefig("016_rank_robustness_single_mouse.png", dpi=150, bbox_inches="tight")
print("saved 016_rank_robustness_single_mouse.png")

rows = []
for session_id, r in results.items():
    for i, rank in enumerate(RANKS):
        rows.append({"session_id": session_id, "n_neurons": r["n_neurons"], "rank": rank,
                     "mean_error": np.mean(r["error"][i]), "mean_similarity": np.mean(r["similarity"][i])})
pd.DataFrame(rows).to_csv("../artifacts/rank_robustness_single_mouse.csv", index=False)
