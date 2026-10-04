"""Checkpointed/resumable sweep: decode modality (whisker vs auditory)
anchored to **start_time** (stimulus onset), 50ms/10ms-stride sliding
windows -- same pipeline/classifier as `002_learning_stage_pilot_sweep.py`
but target=modality and trial set=all active-epoch whisker+auditory trials
(`prep_modality_trials`, NOT restricted to licked trials -- unlike the
lick-aligned modality decoder, no lick event is needed to anchor this
window, so using the full trial set gives more data and answers a more
general question: does modality decode from the stimulus regardless of
behavioral response).

Both cohorts, both session-halves, both area schemes, entire-dataset scope
(learners-only computed post-hoc as a session filter, as elsewhere).
`sys.argv[1]` selects day_stage ('learning' default, or 'expert').

Purpose (user request, 2026-09-11): full comparison across halves/cohorts
for the stim-locked modality decode, complementing the existing
start_time-aligned lick_flag sweep and the lick_time-aligned modality
sweep already done.
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

WINDOW = (-0.2, 0.6)
BIN_WIDTH = 0.05
STRIDE = 0.01
N_REPEATS = 5
N_WORKERS = 8
DAY_STAGE = sys.argv[1] if len(sys.argv) > 1 else "learning"
OUT_DIR = Path(__file__).resolve().parent
_suffix = "" if DAY_STAGE == "learning" else f"_{DAY_STAGE}"
PARTIAL_PATH = OUT_DIR / f"021_modality_stimlocked_results_partial{_suffix}.parquet"
BIN_EDGES_PATH = OUT_DIR / f"021_bin_edges{_suffix}.json"

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
        sliding_bin_population_matrices,
        data_sufficiency_ok,
        sliding_bin_edges,
        load_session_unit_spikes,
        prep_modality_trials,
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

    trials = prep_modality_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
    if trials is None or len(trials) == 0:
        return [dict(session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                      learning_category=learning_category, area_col=None, area_value=None, half=None,
                      skipped_reason="no usable whisker/auditory trials")]

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
                start_time = half_trials["start_time"].to_numpy()
                is_whisker = y.copy()
                matrices = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges)
                X_wide = wide_window_matrix_from_bins(matrices)
                C = select_fixed_c(X_wide, y, rng)
                real, surrogate = session_real_and_shuffled_curves(matrices, y, C, rng, n_repeats=N_REPEATS)
                rows.append(dict(
                    session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                    learning_category=learning_category, area_col=area_col, area_value=area_value,
                    half=half, n_units=len(unit_ids), n_trials=len(y), C=C,
                    real_curve=real.tolist(), surrogate_curve=surrogate.tolist(),
                    peak_acc=float(np.nanmax(real)), skipped_reason=None,
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
    from ssl_timeresolved_decoding import sliding_bin_edges, hitmiss_session_list

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")

    hitmiss_sessions = hitmiss_session_list(sessions_tbl)
    todo_sessions = hitmiss_sessions[hitmiss_sessions.day_stage == DAY_STAGE].reset_index(drop=True)
    print(f"{DAY_STAGE}-stage sessions to process: {len(todo_sessions)}", flush=True)

    bin_edges = sliding_bin_edges(WINDOW, bin_width=BIN_WIDTH, stride=STRIDE)
    BIN_EDGES_PATH.write_text(json.dumps(bin_edges))
    print(f"bin grid: {len(bin_edges)} bins", flush=True)

    done = load_done_sessions()
    print(f"already-done sessions (resume): {len(done)}", flush=True)

    todo = todo_sessions[~todo_sessions.session_id.isin(done)]
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
