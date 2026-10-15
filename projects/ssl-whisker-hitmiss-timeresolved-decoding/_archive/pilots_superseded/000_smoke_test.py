"""Small-scale smoke test for the time-resolved hit/miss decoding pipeline
(`scripts/ssl_timeresolved_decoding.py`). Goals (per TODO.md): verify no
bugs end-to-end, check class balance after the session-half split, get real
per-bin/per-session timing, sanity-check the accuracy-vs-time curve shape
and the dead-zone gap. Entire-dataset population scope only, both area
parcellation schemes, 1 learning/R+ session + 1 expert/R- session.

Not a confirmatory run: small `n_shuf`/`n_repeats` for timing purposes.
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
    bin_population_matrices,
    cluster_permutation_test,
    data_sufficiency_ok,
    dead_zone_offset_bin_edges,
    decode_curve,
    hitmiss_session_list,
    label_shuffle_null_curves,
    load_session_unit_spikes,
    prep_hitmiss_trials,
    select_fixed_c,
    wide_window_matrix_from_bins,
)

# NOTE (2026-09-10): this script's logged results/timings in TODO.md were
# produced *before* the `bin_population_matrices` signature changed to take
# a pre-loaded `unit_spikes_by_cluster` dict (session-level shard caching,
# added after this smoke test surfaced the per-call shard-reload cost) and
# *before* the group-level significance pivot (see
# `scripts/ssl_timeresolved_decoding.py`'s Group-level significance
# section) replaced the per-session label-shuffle null this script still
# demonstrates. Patched to keep running under the current API; the
# per-session null path is kept here only as a smoke-test reference for
# `label_shuffle_null_curves` itself, not the project's chosen significance
# method going forward -- see `001_group_level_demo.py`.

WINDOW = (-0.2, 0.6)  # seconds relative to start_time
N_SHUF_SMOKE = 30
N_REPEATS_SMOKE = 2
SESSIONS_TO_TEST = ["AB080_20230622_152205", "MH022_20250311_140628"]

OUT_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    hitmiss_sessions = hitmiss_session_list(sessions_tbl)
    print(f"hitmiss_session_list: {len(hitmiss_sessions)} usable sessions")
    print(hitmiss_sessions[hitmiss_sessions.session_id.isin(SESSIONS_TO_TEST)])

    bin_edges = dead_zone_offset_bin_edges(WINDOW)
    print(f"bin grid: {len(bin_edges)} bins of 10ms, dead-zone bin dropped")
    print("first 3 bins:", bin_edges[:3], " ... bin nearest 0:",
          [b for b in bin_edges if b[0] < 0.02 and b[1] > -0.02])

    all_results = []
    timings = []

    for session_id in SESSIONS_TO_TEST:
        t_session_start = time.time()
        row = hitmiss_sessions[hitmiss_sessions.session_id == session_id]
        if len(row) == 0:
            print(f"SKIP {session_id}: not in hitmiss_session_list (filtered out)")
            continue
        day_stage = row.iloc[0]["day_stage"]
        reward_group = row.iloc[0]["reward_group"]
        print(f"\n=== {session_id} ({day_stage}, {reward_group}) ===")

        trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        if trials is None or len(trials) == 0:
            print("SKIP: no usable whisker trials")
            continue
        print(f"whisker trials: {len(trials)}, hit rate {trials['rewarded'].mean():.2f}")
        print(trials.groupby("half")["rewarded"].agg(["size", "mean"]))

        unit_spikes = load_session_unit_spikes(dataset_root, session_id)

        for area_col in ("area_group", "area_acronym_custom"):
            areas = areas_with_enough_units(session_id, area_col, area_labels)
            print(f"[{area_col}] {len(areas)} areas with >=5 good+mua units")
            if not areas:
                continue
            # For the smoke test, just take the area with the most units
            # (cheapest way to get a real timing number without sweeping
            # every area yet).
            counts = area_labels[
                (area_labels.session_id == session_id) & (area_labels.bc_label.isin(("good", "mua")))
            ].groupby(area_col)["cluster_id"].nunique()
            area_value = counts.loc[areas].idxmax()
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            print(f"  testing area={area_value!r}, n_units={len(unit_ids)}")

            for half in ("first", "second"):
                half_trials = trials[trials["half"] == half]
                y = half_trials["rewarded"].to_numpy()
                ok, reason = data_sufficiency_ok(len(unit_ids), y)
                if not ok:
                    print(f"  [{half}] SKIP: {reason}")
                    continue

                t0 = time.time()
                start_time = half_trials["start_time"].to_numpy()
                is_whisker = np.ones(len(half_trials), dtype=bool)
                matrices = bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges)
                t_matrices = time.time() - t0

                # Fixed C from the wide summary window, built from the already-computed
                # per-bin matrices (avoids re-hitting the dead-zone-straddle ValueError
                # a direct full-window call would raise).
                t0 = time.time()
                X_wide = wide_window_matrix_from_bins(matrices)
                C = select_fixed_c(X_wide, y, rng)
                t_c = time.time() - t0

                t0 = time.time()
                curve = decode_curve(matrices, y, C, rng, n_repeats=N_REPEATS_SMOKE)
                t_curve = time.time() - t0

                t0 = time.time()
                null_curves = label_shuffle_null_curves(matrices, y, C, rng, n_shuf=N_SHUF_SMOKE, n_repeats=1)
                t_null = time.time() - t0

                cluster_res = cluster_permutation_test(curve, null_curves)

                print(
                    f"  [{half}] n_units={len(unit_ids)} n_trials={len(y)} C={C:.4g} "
                    f"peak_acc={np.nanmax(curve):.3f} cluster_p={cluster_res['p_value']:.3f} "
                    f"| timing: matrices={t_matrices:.2f}s C={t_c:.2f}s curve={t_curve:.2f}s null({N_SHUF_SMOKE})={t_null:.2f}s"
                )

                timings.append(
                    dict(
                        session_id=session_id, day_stage=day_stage, reward_group=reward_group,
                        area_col=area_col, area_value=area_value, half=half, n_units=len(unit_ids),
                        n_trials=len(y), n_bins=len(bin_edges), t_matrices=t_matrices, t_c=t_c,
                        t_curve_per_repeat=t_curve / N_REPEATS_SMOKE, t_null_per_shuf=t_null / N_SHUF_SMOKE,
                    )
                )
                all_results.append(
                    dict(
                        session_id=session_id, day_stage=day_stage, reward_group=reward_group,
                        area_col=area_col, area_value=area_value, half=half,
                        bin_centers=[(b[0] + b[1]) / 2 for b in bin_edges],
                        curve=curve.tolist(), cluster_p=cluster_res["p_value"],
                        observed_max_cluster_mass=cluster_res["observed_max_cluster_mass"],
                    )
                )

        print(f"session wall time: {time.time() - t_session_start:.1f}s")

    timings_df = pd.DataFrame(timings)
    timings_df.to_csv(OUT_DIR / "000_smoke_timings.csv", index=False)
    print("\n=== TIMING SUMMARY ===")
    print(timings_df)

    results_df = pd.DataFrame(all_results)
    results_df.to_json(OUT_DIR / "000_smoke_results.json", orient="records", indent=2)

    if len(timings_df):
        per_decode_repeat = timings_df["t_curve_per_repeat"].mean()
        per_null_shuf = timings_df["t_null_per_shuf"].mean()
        n_bins = timings_df["n_bins"].iloc[0]
        print(f"\nExtrapolation (n_bins={n_bins}):")
        print(f"  ~{per_decode_repeat * n_bins:.2f}s per (5-repeat) decode curve x per-repeat cost")
        print(f"  ~{per_null_shuf * n_bins:.2f}s per label-shuffle null draw (x n_shuf for full null)")


if __name__ == "__main__":
    main()
