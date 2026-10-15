"""Quick illustrative check (not a full sweep): decode modality
(whisker vs auditory) anchored to **start_time** instead of lick_time, on
the same licked-trial population the lick-aligned decoder used, for a
handful of sessions. Purpose: answer "how can modality be decoded well
before the lick" -- the stimulus (which fully determines modality) occurs
well before the lick (RT ~350-450ms), so a lot of the lick-aligned
window's "pre-lick" time is already well after stimulus onset. Plots the
stimulus-locked decode curve alongside the RT distribution for the same
trials, on the same time axis, to make this concrete.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ssl_timeresolved_decoding import (
    AREA_LABELS_PATH,
    area_units,
    data_sufficiency_ok,
    load_session_unit_spikes,
    prep_lick_aligned_trials,
    select_fixed_c,
    session_real_and_shuffled_curves,
    sliding_bin_edges,
    sliding_bin_population_matrices,
    wide_window_matrix_from_bins,
)

WINDOW = (-0.2, 0.6)
AREA_COL, AREA_VALUE = "area_group", "Motor and frontal areas"
SESSIONS = {
    "R+": ["AB080_20230622_152205", "AB086_20231015_141742", "MH032_20250507_143741"],
    "R-": ["MH023_20250316_110814", "AB158_20250413_145012", "AB124_20240815_111810"],
}
rng = np.random.default_rng(51)
OUT_DIR = Path(__file__).resolve().parent


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)
    bin_edges = sliding_bin_edges(WINDOW, bin_width=0.05, stride=0.01)
    bin_centers_ms = np.array([(b[0] + b[1]) / 2 * 1000 for b in bin_edges])

    curves = []
    all_rt = []
    for reward_group, session_ids in SESSIONS.items():
        for session_id in session_ids:
            t0 = time.time()
            trials = prep_lick_aligned_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
            if trials is None:
                continue
            unit_ids = area_units(session_id, AREA_COL, AREA_VALUE, area_labels)
            y = (trials["trial_type"] == "whisker_trial").to_numpy()
            ok, reason = data_sufficiency_ok(len(unit_ids), y)
            if not ok:
                print(f"{session_id}: SKIP {reason}")
                continue
            unit_spikes = load_session_unit_spikes(dataset_root, session_id)
            start_time = trials["start_time"].to_numpy()
            is_whisker = y.copy()
            matrices = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges)
            X_wide = wide_window_matrix_from_bins(matrices)
            C = select_fixed_c(X_wide, y, rng)
            real, _ = session_real_and_shuffled_curves(matrices, y, C, rng, n_repeats=5)
            curves.append(real)
            all_rt.extend((trials["rt"] * 1000).tolist())
            print(f"{session_id} ({reward_group}): n_units={len(unit_ids)} n_trials={len(y)} "
                  f"peak_acc={np.nanmax(real):.3f} ({time.time()-t0:.0f}s)")

    curves = np.stack(curves)
    mean_curve = np.nanmean(curves, axis=0)
    sem_curve = np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0])
    all_rt = np.array(all_rt)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=[2, 1])
    ax1.plot(bin_centers_ms, mean_curve, color="#2c5f5b", lw=2.2)
    ax1.fill_between(bin_centers_ms, mean_curve - sem_curve, mean_curve + sem_curve, color="#2c5f5b", alpha=0.25, lw=0)
    ax1.axhline(0.5, color="#888888", lw=1.2, linestyle=":", label="chance (0.5)")
    ax1.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.5, label="start_time (stimulus onset)")
    ax1.set_ylabel("balanced accuracy")
    ax1.set_title(f"Modality decode, {AREA_VALUE}, anchored to start_time (n={curves.shape[0]} sessions)")
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(all_rt, bins=40, color="#8b9793", edgecolor="none")
    median_rt = np.median(all_rt)
    ax2.axvline(median_rt, color="#c0392b", lw=1.5, linestyle="--", label=f"median RT = {median_rt:.0f}ms")
    ax2.axvline(0, color="#333333", lw=1, linestyle="-", alpha=0.5)
    ax2.set_xlabel("time from start_time (ms)")
    ax2.set_ylabel("trial count")
    ax2.set_title("Lick-time distribution for the same trials (= 0 point of the lick-aligned analysis)")
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "020_modality_stimlocked_vs_rt.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nsaved {out_path}")
    print(f"median RT: {median_rt:.0f}ms -- so lick-aligned bin at -{median_rt:.0f}ms (the lick itself) "
          f"corresponds to ~0ms here (stimulus onset), and lick-aligned bins from -500ms to -{median_rt:.0f}ms "
          f"correspond to roughly {-500+median_rt:.0f}ms to 0ms here (before the stimulus, on the median trial).")


if __name__ == "__main__":
    main()
