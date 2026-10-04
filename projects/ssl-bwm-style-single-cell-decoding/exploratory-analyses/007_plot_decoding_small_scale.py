"""Diagnostic figure for the decoding small-scale checkpoint: imposter-null
balanced-accuracy distribution vs. the observed real-target score, for the
modality decode on AB080_20230622_152205.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

with open(HERE / "005_decode_modality_small_scale_results.json") as f:
    result = json.load(f)

null_scores = np.array(result["null_scores"])
observed = result["observed_balanced_accuracy"]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.hist(null_scores, bins=12, color="#7f7f7f", alpha=0.8, edgecolor="white", label=f"imposter null (n={len(null_scores)})")
ax.axvline(observed, color="#d62728", linewidth=2.5, label=f"observed = {observed:.3f}")
ax.axvline(0.5, color="black", linestyle=":", linewidth=1, label="chance = 0.5")
ax.set_xlabel("nested-CV balanced accuracy")
ax.set_ylabel("count (imposter sessions)")
ax.set_title(f"Decoding small-scale checkpoint: modality, {result['session_id']} ({result['day_stage']})\n"
             f"n_trials={result['n_trials']}, n_units={result['n_units']}, "
             f"n_runs={result['n_runs']}, p={result['p_value']:.4f}", fontsize=10)
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
out_path = HERE / "007_decoding_null_distribution.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Wrote {out_path}")
