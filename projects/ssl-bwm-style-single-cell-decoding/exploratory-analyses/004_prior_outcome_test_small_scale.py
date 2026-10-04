"""Small-scale checkpoint: Prior-outcome (block-analog) single-cell test.
Tested factor = t-1-rewarded vs not, in the same pre-stimulus baseline
window as the Response test [-200ms, -1ms] relative to start_time;
stratified by modality x response (4 strata); PLAIN (not block-aware)
shuffle, since t-1-rewarded is itself the tested factor here (mirrors BWM's
own `get_block`, which uses the plain `TwoNmannWhitneyUshuf` rather than the
time-aware variant for the block test itself). See
`scripts/ssl_bwm_single_cell.py` for the shared driver.
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
        test_name="prior_outcome",
        dataset_root=DATASET_ROOT,
        sessions=SESSIONS,
        window=BASELINE_WINDOW,
        factor_col="t1_rewarded",
        strata1_col="is_whisker",
        strata2_col="response",
        block_aware=False,
        n_shuf=N_SHUF,
        n_units_subset=N_UNITS_SUBSET,
        rng=rng,
        out_dir=Path(__file__).resolve().parent,
    )
