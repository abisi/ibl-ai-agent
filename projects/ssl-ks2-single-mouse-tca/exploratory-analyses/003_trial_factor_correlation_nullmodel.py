"""Pilot: fit a rank-4 TCA model on the pilot mouse, correlate each trial-mode
component against P(lick) and d-prime, and test each correlation against a
circular-shift null (stats_lib.circular_shift_test) instead of trusting a
naive parametric p-value -- both a TCA trial factor and a learning curve can
drift together over a session for unrelated reasons, and the circular shift
is the guard against calling that drift "learning-related".

Rank=4 chosen from 001's rank-selection plot: reconstruction error keeps
falling smoothly with no sharp elbow (typical for single-trial spike counts,
which are Poisson-noise-dominated), but replicate similarity is high and
stable from rank 3 up; 4 is a modest choice in that stable range, to be
revisited once this is run across all mice.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensortools as tt

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)
from behavior_lib import smoothed_plick, rolling_dprime
from stats_lib import circular_shift_test

SESSION_ID = "AB116_20240724_102941"
RANK = 4
N_SHIFTS = 2000
rng = np.random.default_rng(0)

sessions, units, trials = load_session_tables()
units_session = units[units["session_id"] == SESSION_ID].reset_index(drop=True)
trials_session = trials[trials["session_id"] == SESSION_ID]
aligned = align_whisker_trials_first_hit(trials_session)
spike_times_by_cluster = load_session_spike_times(SESSION_ID, units_session)
tensor, time_bins = build_spike_tensor(spike_times_by_cluster, units_session, aligned)

tensor_min = tensor.min(axis=(0, 2), keepdims=True)
tensor_max = tensor.max(axis=(0, 2), keepdims=True)
tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)

model = tt.ncp_hals(tensor_norm, rank=RANK, verbose=False)
trial_factors = model.factors[0]  # shape (n_trials, RANK)

behaviors = {
    "P(lick)": smoothed_plick(aligned, window=10),
    "d-prime": rolling_dprime(trials_session, aligned, window_trials=20),
}

fig, axes = plt.subplots(RANK, len(behaviors), figsize=(4 * len(behaviors), 2.2 * RANK), sharex="col")
results = []
for comp in range(RANK):
    for j, (beh_name, beh_trace) in enumerate(behaviors.items()):
        real_r, null_r, p = circular_shift_test(trial_factors[:, comp], beh_trace, n_shifts=N_SHIFTS, rng=rng)
        results.append({"component": comp + 1, "behavior": beh_name, "r": real_r, "p": p})

        ax = axes[comp, j]
        ax.hist(null_r, bins=40, color="0.7", label="circular-shift null")
        ax.axvline(real_r, color="crimson", lw=2, label="real r")
        ax.set_title(f"comp {comp + 1} x {beh_name}\nr={real_r:.2f}, p={p:.3f}", fontsize=9)
        if comp == RANK - 1:
            ax.set_xlabel("Pearson r")
        if j == 0:
            ax.set_ylabel("count (null)")

axes[0, 0].legend(fontsize=7)
fig.suptitle(f"{SESSION_ID}: trial-factor x behavior correlation, circular-shift null (n={N_SHIFTS})", y=1.01)
fig.tight_layout()
fig.savefig("003_trial_factor_correlation_nullmodel.png", dpi=150, bbox_inches="tight")
print("saved 003_trial_factor_correlation_nullmodel.png")

for r in results:
    print(f"component {r['component']:>2} x {r['behavior']:<8} r={r['r']:+.3f}  p={r['p']:.4f}")
