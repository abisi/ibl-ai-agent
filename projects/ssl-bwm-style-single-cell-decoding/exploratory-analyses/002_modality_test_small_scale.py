"""Small-scale checkpoint (TODO.md step 3): Modality single-cell test.
Tested factor = whisker vs auditory trial_type, in the dead-zone-respecting
evoked window [4ms, 100ms] post start_time; stratified by response
(lick/no-lick) x t-1-rewarded (4 strata); block-aware shuffle keyed on the
t-1-rewarded run index. See question.md for the full design and
`scripts/ssl_bwm_single_cell.py` for the shared driver all 3 tests use.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np

from ssl_bwm_single_cell import run_small_scale_test
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATASET_ROOT = resolve_dataset_dir("ssl_ephys")
EVOKED_WINDOW = (0.005, 0.035)  # locked by user, 2026-08-19
N_SHUF = 1000  # reduced from BWM's 3000 per user decision, 2026-08-19
RNG_SEED = 20260818
N_UNITS_SUBSET = 20

SESSIONS = [
    ("AB080_20230622_152205", "learning"),
    ("MH021_20250311_110321", "expert"),
]

if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    run_small_scale_test(
        test_name="modality",
        dataset_root=DATASET_ROOT,
        sessions=SESSIONS,
        window=EVOKED_WINDOW,
        factor_col="is_whisker",
        strata1_col="response",
        strata2_col="t1_rewarded",
        block_aware=True,
        n_shuf=N_SHUF,
        n_units_subset=N_UNITS_SUBSET,
        rng=rng,
        out_dir=Path(__file__).resolve().parent,
    )
