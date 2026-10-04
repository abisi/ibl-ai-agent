"""Stress test: was rank 4 too coarse for the megamouse fits specifically?
Refits the R+ pooled tensor (where the single-mouse analysis found the
"late activation" pattern in 17/37 mice -- the most favorable case for it
to show up) at rank 8 instead of 4, using the same reused pooled tensor
from 011 (no rebuild). If a late-stage component still doesn't appear even
with twice as many component "slots" available, that's much stronger
evidence the megamouse approach structurally can't see this pattern,
rather than it just needing more components.
"""
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensortools as tt

from component_semantics_lib import classify_component

RANK = 8
N_REPLICATES = 3
OUT_DIR = Path("../artifacts/megamouse")

d = np.load(OUT_DIR / "pooled_raw.npz", allow_pickle=True)
tensor, time_bins = d["tensor"], d["time_bins"]
reward_group = d["reward_group"]

mask = reward_group == 1  # R+ only -- most favorable case for recovering "late"
sub_tensor = tensor[:, mask, :]
print(f"R+ subset: {sub_tensor.shape[1]} neurons, fitting rank={RANK}")

tmin, tmax = sub_tensor.min(axis=(0, 2), keepdims=True), sub_tensor.max(axis=(0, 2), keepdims=True)
sub_norm = ((sub_tensor - tmin) / (tmax - tmin + 1e-10)).astype(np.float32)

t0 = time.time()
best_model, best_obj = None, np.inf
for rep in range(N_REPLICATES):
    model = tt.ncp_hals(sub_norm, rank=RANK, verbose=False, random_state=rep)
    print(f"  replicate {rep + 1}/{N_REPLICATES}: objective={model.obj:.4f}, {time.time() - t0:.0f}s elapsed")
    if model.obj < best_obj:
        best_obj, best_model = model.obj, model

trial_factors, neuron_factors, time_factors = best_model.factors
print(f"\nrank-{RANK} R+ megamouse, best objective={best_obj:.4f}, fit time={time.time() - t0:.0f}s")

labels = []
for comp in range(RANK):
    sign, stage, latency_ms = classify_component(time_factors[:, comp], time_bins)
    labels.append(f"c{comp + 1}: {stage} {sign} ({latency_ms:.0f}ms)")
    print(f"  {labels[-1]}")

n_late = sum(1 for l in labels if "late" in l)
print(f"\n{n_late}/{RANK} components classified as 'late' stage")

time_centers = (time_bins[:-1] + time_bins[1:]) / 2
fig, ax = plt.subplots(figsize=(6, 4.5))
for comp in range(RANK):
    ax.plot(time_centers, time_factors[:, comp], lw=1.6, label=labels[comp])
ax.axvline(0, color="0.5", ls="--", lw=0.8)
ax.set_xlabel("time from stim onset (s)")
ax.set_ylabel("time factor")
ax.set_title(f"R+ megamouse, rank={RANK} (vs. rank=4 used elsewhere)")
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig("017_megamouse_higher_rank.png", dpi=150, bbox_inches="tight")
print("saved 017_megamouse_higher_rank.png")

np.savez(OUT_DIR / "rplus_rank8.npz", trial_factors=trial_factors, neuron_factors=neuron_factors,
         time_factors=time_factors, time_bins=time_bins, objective=best_obj)
