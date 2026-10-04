"""Pilot: fit TCA on the single-mouse tensor from 000 and pick a candidate
rank range, timing the fit to estimate the cost of scaling to 68 mice.

Preprocessing note (deliberate deviation from the megamouse pipeline): TCA is
fit with `fit_method="ncp_hals"`, a *non-negative* CP solver, which requires
non-negative input regardless of tensortools's separate `nonneg` flag (see
`tensortools.Ensemble.__init__`). The reference megamouse pipeline mostly
preprocesses with `baseline_subtract_and_maxabs_normalize` (produces negative
values, baseline at zero) before calling ncp_hals for single-condition runs,
which is internally inconsistent -- but does use a non-negative
`minmax_normalize_per_neuron` for its "combined" condition. This pipeline
uses min-max normalization per neuron (non-negative, consistent with
ncp_hals) throughout, to avoid feeding negative values to a non-negative
solver.
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import tensortools as tt

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)

SESSION_ID = "AB116_20240724_102941"
RANKS = range(1, 9)
N_REPLICATES = 5  # pilot value; megamouse pipeline uses 20 for its pooled tensor

sessions, units, trials = load_session_tables()
units_session = units[units["session_id"] == SESSION_ID].reset_index(drop=True)
trials_session = trials[trials["session_id"] == SESSION_ID]
aligned = align_whisker_trials_first_hit(trials_session)
spike_times_by_cluster = load_session_spike_times(SESSION_ID, units_session)
tensor, time_bins = build_spike_tensor(spike_times_by_cluster, units_session, aligned)

# Min-max normalize each neuron to [0, 1] across all trials & time, so every
# neuron contributes on a comparable scale without introducing negative
# values (required for the non-negative ncp_hals solver used below).
tensor_min = tensor.min(axis=(0, 2), keepdims=True)
tensor_max = tensor.max(axis=(0, 2), keepdims=True)
tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)

t0 = time.time()
ensemble = tt.Ensemble(fit_method="ncp_hals")
ensemble.fit(tensor_norm, ranks=RANKS, replicates=N_REPLICATES)
fit_time = time.time() - t0

print(f"tensor shape: {tensor_norm.shape}")
print(f"fit time ({len(list(RANKS))} ranks x {N_REPLICATES} replicates): {fit_time:.1f}s")
print(f"estimated time for all 68 mice at this setting: {fit_time * 68 / 60:.1f} min")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
tt.plot_objective(ensemble, ax=axes[0])
axes[0].set_xlabel("model rank")
axes[0].set_ylabel("normalized reconstruction error")
tt.plot_similarity(ensemble, ax=axes[1])
axes[1].set_xlabel("model rank")
axes[1].set_ylabel("replicate similarity")
fig.suptitle(f"{SESSION_ID}: TCA rank selection ({N_REPLICATES} replicates/rank)")
fig.tight_layout()
fig.savefig("001_single_mouse_rank_selection.png", dpi=150, bbox_inches="tight")
print("saved 001_single_mouse_rank_selection.png")
