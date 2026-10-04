"""Fit TCA on the pooled megamouse tensor from 011 -- rank 4, same as every
single-mouse fit (see the rank-validation note in the report: both this
project's own rank-selection curve and the reference pipeline's own saved
outputs converge on 4), same min-max-per-neuron normalization (non-negative,
required by the non-negative solver `ncp_hals`), several replicates so a
poor local optimum on this much larger fit doesn't go unnoticed.
"""
import time
from pathlib import Path

import numpy as np
import tensortools as tt

RANK = 4
N_REPLICATES = 5
OUT_DIR = Path("../artifacts/megamouse")

d = np.load(OUT_DIR / "pooled_raw.npz", allow_pickle=True)
tensor = d["tensor"]
print(f"pooled tensor: {tensor.shape}, {tensor.nbytes / 1e9:.2f} GB")

tensor_min = tensor.min(axis=(0, 2), keepdims=True)
tensor_max = tensor.max(axis=(0, 2), keepdims=True)
tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)
tensor_norm = tensor_norm.astype(np.float32)

t0 = time.time()
best_model, best_obj = None, np.inf
for rep in range(N_REPLICATES):
    model = tt.ncp_hals(tensor_norm, rank=RANK, verbose=False, random_state=rep)
    obj = model.obj
    print(f"  replicate {rep + 1}/{N_REPLICATES}: objective={obj:.4f}")
    if obj < best_obj:
        best_obj, best_model = obj, model
fit_time = time.time() - t0
print(f"fit time ({N_REPLICATES} replicates): {fit_time:.1f}s; best objective={best_obj:.4f}")

trial_factors, neuron_factors, time_factors = best_model.factors
np.savez(
    OUT_DIR / "megamouse_factors.npz",
    trial_factors=trial_factors, neuron_factors=neuron_factors, time_factors=time_factors,
    time_bins=d["time_bins"], area_custom=d["area_custom"], reward_group=d["reward_group"],
    mouse_id=d["mouse_id"], pooled_plick=d["pooled_plick"], pooled_dprime=d["pooled_dprime"],
    n_mice=int(d["n_mice"]), objective=best_obj,
)
print(f"saved {OUT_DIR / 'megamouse_factors.npz'}")
