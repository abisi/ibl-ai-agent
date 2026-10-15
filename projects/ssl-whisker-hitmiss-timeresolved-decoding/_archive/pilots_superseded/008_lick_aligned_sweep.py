"""Checkpointed/resumable parallel sweep: lick_time-aligned modality
(whisker vs auditory) decode, restricted to lick_flag==1 trials, window
-500ms/+200ms relative to lick_time. Entire-dataset population scope, both
area schemes, both session-halves; `sys.argv[1]` selects day_stage
('learning' default, or 'expert'). Mirrors
`002_learning_stage_pilot_sweep.py`'s parallel/checkpoint structure.

**Binning + classifier changed 2026-09-10 (user decisions, same session as
the start_time-aligned sweep's changes)**: 50ms-wide sliding windows
stepped every 10ms (`sliding_bin_edges`, shared with the start_time-aligned
sweep) instead of the original 10ms-disjoint grid, and
`event_aligned_rates_for_trials` generalized from blanket-NaN-on-any-overlap
to per-trial two-piece dead-zone excision (recovers partial data for a
window much wider than the 5ms artifact -- see that function's docstring).
Classifier now goes through `_make_classifier`'s per-fold StandardScaler
(added in `scripts/ssl_timeresolved_decoding.py`, not re-touched here).
Prior (10ms-disjoint, unscaled) results preserved under `_unscaled`.

**Known confound** (checked 2026-09-10 on 30 sessions before the first run
of this sweep): whisker RT is ~120ms longer than auditory RT on average,
paired per session -- a modality decode near the lick may partly reflect
time-since-stimulus rather than pure modality content. Flag in any report.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

WINDOW = (-0.5, 0.2)
BIN_WIDTH = 0.05
STRIDE = 0.01
N_REPEATS = 5
N_WORKERS = 8
DAY_STAGE = sys.argv[1] if len(sys.argv) > 1 else "learning"  # 'learning' or 'expert'
OUT_DIR = Path(__file__).resolve().parent
_suffix = "" if DAY_STAGE == "learning" else f"_{DAY_STAGE}"
PARTIAL_PATH = OUT_DIR / f"008_lick_aligned_results_partial{_suffix}.parquet"
BIN_EDGES_PATH = OUT_DIR / f"008_bin_edges{_suffix}.json"

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _worker_init():
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


def process_one_session(args: tuple) -> list[dict]:
    session_id, subject_id, reward_group, learning_category, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH,
        area_units,
        areas_with_enough_units,
        data_sufficiency_ok,
        sliding_bin_edges,
        lick_aligned_bin_population_matrices,
        load_session_unit_spikes,
        prep_lick_aligned_trials,
        select_fixed_c,
        session_real_and_shuffled_curves,
        wide_window_matrix_from_bins,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)
    bin_edges = sliding_bin_edges(WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)
    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))

    trials = prep_lick_aligned_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
    if trials is None or len(trials) == 0:
        return [dict(session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                      learning_category=learning_category, area_col=None, area_value=None, half=None,
                      skipped_reason="no usable licked trials")]

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    rows = []
    for area_col in ("area_group", "area_acronym_custom"):
        areas = areas_with_enough_units(session_id, area_col, area_labels)
        for area_value in areas:
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            for half in ("first", "second"):
                half_trials = trials[trials["half"] == half]
                y = (half_trials["trial_type"] == "whisker_trial").to_numpy()
                ok, reason = data_sufficiency_ok(len(unit_ids), y)
                if not ok:
                    rows.append(dict(
                        session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                        learning_category=learning_category, area_col=area_col, area_value=area_value,
                        half=half, n_units=len(unit_ids), n_trials=len(y), skipped_reason=reason,
                    ))
                    continue
                event_time = half_trials["lick_time"].to_numpy()
                start_time = half_trials["start_time"].to_numpy()
                is_whisker = y.copy()
                matrices = lick_aligned_bin_population_matrices(unit_spikes, unit_ids, event_time, start_time, is_whisker, bin_edges)
                X_wide = wide_window_matrix_from_bins(matrices)
                C = select_fixed_c(X_wide, y, rng)
                real, surrogate = session_real_and_shuffled_curves(matrices, y, C, rng, n_repeats=N_REPEATS)
                mean_rt_whisker = float(half_trials.loc[is_whisker, "rt"].mean()) if is_whisker.any() else float("nan")
                mean_rt_auditory = float(half_trials.loc[~is_whisker, "rt"].mean()) if (~is_whisker).any() else float("nan")
                rows.append(dict(
                    session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                    learning_category=learning_category, area_col=area_col, area_value=area_value,
                    half=half, n_units=len(unit_ids), n_trials=len(y), C=C,
                    real_curve=real.tolist(), surrogate_curve=surrogate.tolist(),
                    peak_acc=float(np.nanmax(real)), skipped_reason=None,
                    mean_rt_whisker=mean_rt_whisker, mean_rt_auditory=mean_rt_auditory,
                ))
    return rows


def load_done_sessions() -> set[str]:
    if not PARTIAL_PATH.exists():
        return set()
    return set(pd.read_parquet(PARTIAL_PATH, columns=["session_id"])["session_id"].unique())


def append_rows(rows: list[dict]):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if PARTIAL_PATH.exists():
        existing = pd.read_parquet(PARTIAL_PATH)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(PARTIAL_PATH, index=False)


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list, sliding_bin_edges

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")

    hitmiss_sessions = hitmiss_session_list(sessions_tbl)
    learning_sessions = hitmiss_sessions[hitmiss_sessions.day_stage == DAY_STAGE].reset_index(drop=True)
    print(f"{DAY_STAGE}-stage sessions to process: {len(learning_sessions)}", flush=True)

    bin_edges = sliding_bin_edges(WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)
    BIN_EDGES_PATH.write_text(json.dumps(bin_edges))
    print(f"bin grid: {len(bin_edges)} bins", flush=True)

    done = load_done_sessions()
    print(f"already-done sessions (resume): {len(done)}", flush=True)

    todo = learning_sessions[~learning_sessions.session_id.isin(done)]
    print(f"launching {len(todo)} sessions across {N_WORKERS} worker processes", flush=True)

    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    tasks = [
        (r.session_id, r.subject_id, r.reward_group, r.learning_category, scripts_dir)
        for r in todo.itertuples()
    ]

    t_start = time.time()
    n_processed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(process_one_session, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            session_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"ERROR {session_id}: {e!r}", flush=True)
                continue
            append_rows(rows)
            n_processed += 1
            elapsed = time.time() - t_start
            remaining = len(tasks) - n_processed
            eta_min = (elapsed / n_processed) * remaining / 60.0 if n_processed else float("nan")
            n_computed_rows = sum(1 for r in rows if r.get("skipped_reason") is None)
            reward_group = next((t[2] for t in tasks if t[0] == session_id), "?")
            print(
                f"[{n_processed}/{len(tasks)}] {session_id} ({reward_group}): "
                f"{n_computed_rows} computed rows, {len(rows) - n_computed_rows} skipped -- "
                f"elapsed {elapsed/60:.1f}min, ETA remaining {eta_min:.1f}min",
                flush=True,
            )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
