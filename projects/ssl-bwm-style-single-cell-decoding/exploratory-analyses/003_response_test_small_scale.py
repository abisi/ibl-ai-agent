"""Small-scale checkpoint: Response single-cell test.
Tested factor = lick vs no-lick, in a pre-stimulus baseline window
[-200ms, -1ms] relative to start_time (the -1ms edge is the dead-zone-safe
baseline endpoint per ssl_artifact_dead_zone.md, not a fully locked window
width yet -- see question.md open item 3); stratified by modality x
t-1-rewarded (4 strata); block-aware shuffle keyed on the t-1-rewarded run
index. See `scripts/ssl_bwm_single_cell.py` for the shared driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np

from ssl_bwm_single_cell import run_small_scale_test
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATASET_ROOT = resolve_dataset_dir("ssl_ephys")
BASELINE_WINDOW = (-0.200, -0.010)  # locked by user, 2026-08-19
N_SHUF = 1000
RNG_SEED = 20260818
N_UNITS_SUBSET = 20

SESSIONS = [
    ("AB080_20230622_152205", "learning"),
    ("MH021_20250311_110321", "expert"),
]

if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    run_small_scale_test(
        test_name="response",
        dataset_root=DATASET_ROOT,
        sessions=SESSIONS,
        window=BASELINE_WINDOW,
        factor_col="response",
        strata1_col="is_whisker",
        strata2_col="t1_rewarded",
        block_aware=True,
        n_shuf=N_SHUF,
        n_units_subset=N_UNITS_SUBSET,
        rng=rng,
        out_dir=Path(__file__).resolve().parent,
    )
