"""Confirmatory run: all 3 single-cell tests (Modality, Response,
Prior-outcome) across the full ephys cohort, both day-stages, at the locked
parameters from question.md (evoked [5ms,35ms], baseline [-200ms,-10ms],
nShuf=1000). **Updated 2026-08-19 (user decision)**: QC scope widened from
`bc_label=='good'` only (~16,426 units) to `bc_label in {'good','mua'}`
(~139,275 units, ~8.5x more) -- the original good-only run's outputs were
renamed to `*_good_only.parquet`/`*_good_only.png` for reference rather than
overwritten; area-level `MIN_UNITS_PER_AREA` stays at 5 per that same
decision. This is a multi-hour job -- run in the background.

**Fixed 2026-08-20**: a run was killed (not by this code -- externally)
partway through the `response` test, after ~15h, and only the already-fully-
completed `modality` test survived -- checkpointing previously only wrote
after an entire test finished, not per-session. Now checkpoints per session
via a `{test}_partial.parquet` file (`ssl_bwm_single_cell.run_full_test`'s
`resume_rows`/`on_session_done`), so a kill mid-test loses at most one
session's worth of work, not the whole test.

FDR correction (Benjamini-Hochberg, per question.md's locked decision) is
applied per (test, day_stage) -- i.e. one FDR family per test x learning/
expert, matching the day-stage separation policy in ssl_analysis_patterns.md
(learning and expert arms are never pooled for a single statistical family).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np
import pandas as pd

from ssl_bwm_single_cell import run_full_test
from ssl_bwm_stats_util import benjamini_hochberg
from ssl_bwm_trial_prep import list_whisker_training_ephys_sessions
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATASET_ROOT = resolve_dataset_dir("ssl_ephys")
EVOKED_WINDOW = (0.005, 0.035)
BASELINE_WINDOW = (-0.200, -0.010)
N_SHUF = 1000
RNG_SEED = 20260819
QC_VALUES = ("good", "mua")

OUT_DIR = Path(__file__).resolve().parent
RAW_OUT_PATH = OUT_DIR / "single_cell_results_raw.parquet"
FINAL_OUT_PATH = OUT_DIR / "single_cell_results.parquet"

TEST_SPECS = [
    dict(test_name="modality", window=EVOKED_WINDOW, factor_col="is_whisker",
         strata1_col="response", strata2_col="t1_rewarded", block_aware=True),
    dict(test_name="response", window=BASELINE_WINDOW, factor_col="response",
         strata1_col="is_whisker", strata2_col="t1_rewarded", block_aware=True),
    dict(test_name="prior_outcome", window=BASELINE_WINDOW, factor_col="t1_rewarded",
         strata1_col="is_whisker", strata2_col="response", block_aware=False),
]


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    sessions = list_whisker_training_ephys_sessions(sessions_tbl)
    n_learning = sum(1 for _, d in sessions if d == "learning")
    n_expert = sum(1 for _, d in sessions if d == "expert")
    print(f"{len(sessions)} whisker-training ephys sessions ({n_learning} learning, {n_expert} expert)", flush=True)

    all_dfs = []
    if RAW_OUT_PATH.exists():
        existing = pd.read_parquet(RAW_OUT_PATH)
        done_tests = set(existing["test"].unique())
        print(f"Resuming: found existing raw results for tests {sorted(done_tests)}", flush=True)
        all_dfs.append(existing)
    else:
        done_tests = set()

    t_start = time.time()
    for spec in TEST_SPECS:
        test_name = spec["test_name"]
        if test_name in done_tests:
            print(f"Skipping {test_name} (already in {RAW_OUT_PATH})", flush=True)
            continue

        partial_path = OUT_DIR / f"{test_name}_partial.parquet"
        resume_rows = pd.read_parquet(partial_path) if partial_path.exists() else None

        def checkpoint(df: pd.DataFrame, _path: Path = partial_path) -> None:
            df.to_parquet(_path, index=False)

        print(f"\n=== Running {test_name} across full cohort ===", flush=True)
        df = run_full_test(
            dataset_root=DATASET_ROOT, sessions=sessions, n_shuf=N_SHUF, rng=rng,
            qc_values=QC_VALUES, resume_rows=resume_rows, on_session_done=checkpoint, **spec,
        )
        all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(RAW_OUT_PATH, index=False)
        if partial_path.exists():
            partial_path.unlink()
        print(f"Checkpoint: wrote {len(combined)} rows to {RAW_OUT_PATH} "
              f"(total elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    raw = pd.read_parquet(RAW_OUT_PATH)

    # BH-FDR per (test, day_stage) family.
    raw["p_fdr"] = np.nan
    for (test_name, day_stage), grp in raw.groupby(["test", "day_stage"]):
        q = benjamini_hochberg(grp["p_raw"].to_numpy())
        raw.loc[grp.index, "p_fdr"] = q

    raw.to_parquet(FINAL_OUT_PATH, index=False)
    print(f"\nWrote {len(raw)} rows to {FINAL_OUT_PATH}")
    print(raw.groupby(["test", "day_stage"]).agg(
        n_units=("p_raw", "size"),
        n_raw_sig=("p_raw", lambda s: int((s < 0.05).sum())),
        n_fdr_sig=("p_fdr", lambda s: int((s < 0.05).sum())),
    ))
    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
