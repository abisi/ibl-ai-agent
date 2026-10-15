"""Checkpointed/resumable pilot sweep, learning-stage only, entire-dataset
population scope, both area parcellation schemes, both session-halves.
Purpose (per TODO.md): get a real full-scope timing estimate (the 001 demo's
16-348s/session was for one large area only) and produce real per-session
decode curves usable for the group-level significance tests once enough
sessions have accumulated.

**Decode target changed 2026-09-10 (user decision, mid-pilot)**: decodes raw
`lick_flag` directly (1 = licked, 0 = no lick), **the same definition for
both cohorts** -- not the cohort-corrected `rewarded` label used in the
first ~24-session run of this pilot (that run's results/logs are preserved
under the `_cohortcorrected_hit` suffix, not overwritten). Motivation
(`question.md`'s "Sharpened framing note"): decoding cohort-corrected hit is
mathematically identical to decoding `lick_flag` *within* a cohort (R+
hit=lick, R- hit=NOT lick -- balanced accuracy is invariant to that
complement), so comparing accuracy *across* cohorts under the
cohort-corrected label was comparing two differently-defined targets. Raw
`lick_flag` is the same target in both cohorts, so a cohort difference in
decodability is now interpretable as a difference in how well licking
itself is encoded, not an artifact of the label flip.

Parallelized across sessions (added 2026-09-10, machine has 32 logical
cores and was ~11-19% loaded when checked -- the original sequential
version left almost all of that idle). Each worker process is thread-capped
to 1 BLAS/OMP thread (same fix the sibling `ssl-bwm-style-single-cell-decoding`
project's TODO.md used to avoid oversubscription when running its own
concurrent jobs) so `N_WORKERS` processes don't each try to spawn their own
BLAS thread pool on top.

Writes one row per (session_id, area_col, area_value, half) to
`002_pilot_results_partial.parquet`, appended incrementally (main process
only -- workers return rows, they don't write) so an interruption loses at
most the in-progress sessions' rows. `real_curve`/`surrogate_curve` stored
as list columns (bin_edges are the same fixed grid for every row, saved
once in `002_bin_edges.json`).

**Binning changed 2026-09-10 (user decision)**: 50ms-wide sliding windows
stepped every 10ms (`sliding_bin_edges`/`sliding_bin_population_matrices`),
replacing the original 10ms-disjoint grid. Motivation: richer per-bin
features (5x more spikes per window) than 10ms disjoint bins allowed.
The mandatory whisker dead zone is handled by excising it from each
window's valid duration (`sliding_window_rates_for_trials`) rather than
dropping one global bin -- correctness verified on synthetic data (a
window fully straddling the dead zone recovers the correct outside-dead-
zone rate using the reduced valid duration) before this rerun. Prior
10ms-disjoint, unstandardized results are preserved under `_unscaled`;
this run also carries the standardization fix from the same session.
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
N_WORKERS = 8  # conservative given this is a shared machine; low load observed but not guaranteed to stay that way
DAY_STAGE = sys.argv[1] if len(sys.argv) > 1 else "learning"  # 'learning' or 'expert'
OUT_DIR = Path(__file__).resolve().parent
_suffix = "" if DAY_STAGE == "learning" else f"_{DAY_STAGE}"
PARTIAL_PATH = OUT_DIR / f"002_pilot_results_partial{_suffix}.parquet"
BIN_EDGES_PATH = OUT_DIR / f"002_bin_edges{_suffix}.json"

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _worker_init():
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


def process_one_session(args: tuple) -> list[dict]:
    """Runs in a worker process. Re-imports everything here (not at module
    scope) so the thread-limit env vars set in `_worker_init` take effect
    before numpy/sklearn ever spin up their BLAS thread pools."""
    session_id, subject_id, reward_group, learning_category, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811 (worker-local import, see docstring)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH,
        area_units,
        areas_with_enough_units,
        sliding_bin_population_matrices,
        data_sufficiency_ok,
        sliding_bin_edges,
        load_session_unit_spikes,
        prep_hitmiss_trials,
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

    trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
    if trials is None or len(trials) == 0:
        return [dict(session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                      learning_category=learning_category, area_col=None, area_value=None, half=None,
                      skipped_reason="no usable whisker trials")]

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    rows = []
    for area_col in ("area_group", "area_acronym_custom"):
        areas = areas_with_enough_units(session_id, area_col, area_labels)
        for area_value in areas:
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            for half in ("first", "second"):
                half_trials = trials[trials["half"] == half]
                y = half_trials["lick_flag"].to_numpy().astype(bool)  # raw lick_flag, same definition both cohorts
                ok, reason = data_sufficiency_ok(len(unit_ids), y)
                if not ok:
                    rows.append(dict(
                        session_id=session_id, subject_id=subject_id, reward_group=reward_group,
                        learning_category=learning_category, area_col=area_col, area_value=area_value,
                        half=half, n_units=len(unit_ids), n_trials=len(y), skipped_reason=reason,
                    ))
                    continue
                start_time = half_trials["start_time"].to_numpy()
                is_whisker = np.ones(len(half_trials), dtype=bool)
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
            except Exception as e:  # noqa: BLE001 -- record and keep going, don't kill the whole pool
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
