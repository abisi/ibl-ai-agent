"""Real-data demo of the group-level significance pivot (user decision
2026-09-10, see `question.md`): fixed area (`area_group == 'Motor and
frontal areas'`, the most session-common area_group at >=5 good+mua units),
learning-stage only, 3 R+ + 3 R- sessions, both session-halves. Computes
real + one-surrogate decode curve per session (cheap -- no per-session
n_shuf loop), then the mouse-block sign-flip group-level null and cluster
permutation test per (cohort, half), plus the R+/R- group comparison.

Not a confirmatory run: n=3 sessions/cohort is far too small for a real
biological claim -- this validates the group-level machinery end-to-end on
real spike data (synthetic-data correctness already checked separately),
and gives a second, more realistic timing estimate (now dominated by
2x-decode instead of the old n_shuf-null) before scoping the full sweep.
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
    bin_population_matrices,
    cluster_permutation_test,
    data_sufficiency_ok,
    dead_zone_offset_bin_edges,
    group_mouseblock_permutation_null,
    hitmiss_session_list,
    load_session_unit_spikes,
    prep_hitmiss_trials,
    rplus_rminus_group_test,
    select_fixed_c,
    session_real_and_shuffled_curves,
    wide_window_matrix_from_bins,
)

WINDOW = (-0.2, 0.6)
AREA_COL, AREA_VALUE = "area_group", "Motor and frontal areas"
SESSIONS = {
    "R+": ["AB149_20241217_113458", "AB086_20231015_141742", "MH032_20250507_143741"],
    "R-": ["MH023_20250316_110814", "AB158_20250413_145012", "AB124_20240815_111810"],
}
N_REPEATS = 5
N_PERM = 2000

OUT_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(1)


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)
    hitmiss_sessions = hitmiss_session_list(sessions_tbl)

    bin_edges = dead_zone_offset_bin_edges(WINDOW)
    print(f"bin grid: {len(bin_edges)} bins")

    per_session = []  # subject_id, reward_group, half, real_curve, surrogate_curve, peak_acc
    timings = []

    for reward_group, session_ids in SESSIONS.items():
        for session_id in session_ids:
            t0 = time.time()
            row = hitmiss_sessions[hitmiss_sessions.session_id == session_id]
            assert len(row) == 1, f"{session_id} not in hitmiss_session_list"
            trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
            unit_ids = area_units(session_id, AREA_COL, AREA_VALUE, area_labels)
            unit_spikes = load_session_unit_spikes(dataset_root, session_id)
            subject_id = row.iloc[0]["subject_id"]

            for half in ("first", "second"):
                half_trials = trials[trials["half"] == half]
                y = half_trials["rewarded"].to_numpy()
                ok, reason = data_sufficiency_ok(len(unit_ids), y)
                if not ok:
                    print(f"{session_id} [{half}] SKIP: {reason}")
                    continue
                start_time = half_trials["start_time"].to_numpy()
                is_whisker = np.ones(len(half_trials), dtype=bool)
                matrices = bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges)
                X_wide = wide_window_matrix_from_bins(matrices)
                C = select_fixed_c(X_wide, y, rng)
                real, surrogate = session_real_and_shuffled_curves(matrices, y, C, rng, n_repeats=N_REPEATS)
                per_session.append(
                    dict(
                        session_id=session_id, subject_id=subject_id, reward_group=reward_group, half=half,
                        n_units=len(unit_ids), n_trials=len(y), real_curve=real, surrogate_curve=surrogate,
                        metric=float(np.nanmax(real)),
                    )
                )
            dt = time.time() - t0
            timings.append(dict(session_id=session_id, reward_group=reward_group, n_units=len(unit_ids), wall_s=dt))
            print(f"{session_id} ({reward_group}, n_units={len(unit_ids)}): {dt:.1f}s for both halves")

    timings_df = pd.DataFrame(timings)
    timings_df.to_csv(OUT_DIR / "001_group_demo_timings.csv", index=False)
    print("\n=== TIMING ===")
    print(timings_df)
    print(f"mean wall time per session (2 halves): {timings_df.wall_s.mean():.1f}s")

    # --- Group-level above-chance test, per (cohort, half) ---
    print("\n=== GROUP-LEVEL ABOVE-CHANCE (cluster permutation) ===")
    group_results = {}
    for reward_group in ("R+", "R-"):
        for half in ("first", "second"):
            recs = [r for r in per_session if r["reward_group"] == reward_group and r["half"] == half]
            if len(recs) < 2:
                print(f"{reward_group} {half}: too few sessions ({len(recs)}), skipped")
                continue
            obs, null_curves = group_mouseblock_permutation_null(recs, rng, n_perm=N_PERM)
            res = cluster_permutation_test(obs, null_curves)
            group_results[(reward_group, half)] = (obs, res)
            print(
                f"{reward_group} {half}: n_sessions={len(recs)}, peak_acc={np.nanmax(obs):.3f}, "
                f"cluster_p={res['p_value']:.4f}"
            )

    # --- R+ vs R- comparison, per half, on peak-accuracy summary metric ---
    print("\n=== R+ vs R- GROUP COMPARISON (peak accuracy) ===")
    summary_df = pd.DataFrame(per_session)
    for half in ("first", "second"):
        sub = summary_df[summary_df.half == half]
        if sub.reward_group.nunique() < 2:
            continue
        res = rplus_rminus_group_test(sub, rng, n_perm=N_PERM)
        print(f"{half}: {res}")

    np.savez(
        OUT_DIR / "001_group_demo_curves.npz",
        **{f"{rg}_{half}_obs": v[0] for (rg, half), v in group_results.items()},
    )


if __name__ == "__main__":
    main()
