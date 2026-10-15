"""Smoke test for the lick_time-aligned modality (whisker vs auditory)
decode, restricted to lick_flag==1 trials (the only trials with a real
lick_time), window -500ms/+200ms relative to lick_time, 10ms bins,
per-trial dead-zone masking (see `event_aligned_rates_for_trials`).
Validates the pipeline end-to-end on 2 real sessions before any full sweep.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ssl_timeresolved_decoding import (
    AREA_LABELS_PATH,
    area_units,
    areas_with_enough_units,
    data_sufficiency_ok,
    lick_aligned_bin_edges,
    lick_aligned_bin_population_matrices,
    load_session_unit_spikes,
    prep_lick_aligned_trials,
    select_fixed_c,
    session_real_and_shuffled_curves,
    wide_window_matrix_from_bins,
)

WINDOW = (-0.5, 0.2)
SESSIONS = ["AB080_20230622_152205", "MH022_20250309_161625"]
rng = np.random.default_rng(3)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    bin_edges = lick_aligned_bin_edges(WINDOW)
    print(f"bin grid: {len(bin_edges)} bins, {bin_edges[0]} .. {bin_edges[-1]}")

    for session_id in SESSIONS:
        t0 = time.time()
        trials = prep_lick_aligned_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        if trials is None:
            print(f"{session_id}: SKIP no usable trials")
            continue
        print(f"\n{session_id}: {len(trials)} licked trials, modality counts:\n{trials.trial_type.value_counts()}")

        unit_spikes = load_session_unit_spikes(dataset_root, session_id)
        area = areas_with_enough_units(session_id, "area_group", area_labels)
        if not area:
            print("  SKIP: no areas with enough units")
            continue
        counts = area_labels[(area_labels.session_id == session_id) & (area_labels.bc_label.isin(("good", "mua")))].groupby("area_group")["cluster_id"].nunique()
        area_value = counts.loc[area].idxmax()
        unit_ids = area_units(session_id, "area_group", area_value, area_labels)
        print(f"  area={area_value!r} n_units={len(unit_ids)}")

        for half in ("first", "second"):
            half_trials = trials[trials["half"] == half]
            y = (half_trials["trial_type"] == "whisker_trial").to_numpy()
            ok, reason = data_sufficiency_ok(len(unit_ids), y)
            if not ok:
                print(f"  [{half}] SKIP: {reason}")
                continue
            event_time = half_trials["lick_time"].to_numpy()
            start_time = half_trials["start_time"].to_numpy()
            is_whisker = (half_trials["trial_type"] == "whisker_trial").to_numpy()

            t1 = time.time()
            matrices = lick_aligned_bin_population_matrices(unit_spikes, unit_ids, event_time, start_time, is_whisker, bin_edges)
            t_matrices = time.time() - t1

            n_nan_per_bin = [int(np.isnan(X).any(axis=1).sum()) for X in matrices]
            print(f"  [{half}] n_trials={len(y)} NaN-masked trials per bin: min={min(n_nan_per_bin)} max={max(n_nan_per_bin)} mean={np.mean(n_nan_per_bin):.1f}")

            t1 = time.time()
            X_wide = wide_window_matrix_from_bins(matrices)
            C = select_fixed_c(X_wide, y, rng)
            real, surrogate = session_real_and_shuffled_curves(matrices, y, C, rng, n_repeats=5)
            t_decode = time.time() - t1

            print(f"  [{half}] C={C:.4g} peak_acc={np.nanmax(real):.3f} n_valid_bins={np.sum(~np.isnan(real))}/{len(real)} "
                  f"| timing matrices={t_matrices:.1f}s decode={t_decode:.1f}s")

        print(f"  session wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
