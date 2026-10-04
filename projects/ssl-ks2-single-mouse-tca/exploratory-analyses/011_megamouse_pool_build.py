"""Build the actual "megamouse" tensor -- pooling neurons from many mice
along the neuron axis, on a shared trial axis (trial index relative to each
mouse's own first active-context hit) -- the same approach as the reference
`tca_pipeline_bis.py`, applied here to ssl_ks2_ephys via this project's own
loader (tca_lib.py) instead of the M:-drive unit_table/trial_table format.

Concatenating along the neuron axis requires every pooled mouse to share
the exact same trial-axis length and meaning, so (unlike the single-mouse
pipeline, where a mouse with fewer available trials just gets a shorter,
still-valid tensor) only mice with the *full*, unclipped 81-trial window
(30 pre + hit + 50 post) are included here -- 56 of the 68 single-mouse
mice qualify; the other 12 had fewer whisker trials on one side of their
first hit than the window needs. This mirrors a real constraint of the
reference pipeline's own `np.concatenate(tensors, axis=1)` pooling step.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"M:\analysis\Axel_Bisi\brain_wide_analysis")
import allen_utils as allen  # noqa: E402

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)
from behavior_lib import smoothed_plick, rolling_dprime

TRIAL_WINDOW_PRE, TRIAL_WINDOW_POST = 30, 50
EXPECTED_N_TRIALS = TRIAL_WINDOW_PRE + TRIAL_WINDOW_POST + 1
OUT_DIR = Path("../artifacts/megamouse")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sessions, units, trials = load_session_tables()
mice = pd.read_csv("../artifacts/eligible_mice.csv")
cohort_meta = pd.read_parquet(
    "../../ssl-ks2-dataset-description/artifacts/session_metadata_extra.parquet"
)[["session_id", "reward_group"]]
mice = mice.merge(cohort_meta, on="session_id", how="left")

t_start = time.time()
tensors, area_custom_list, reward_group_list, mouse_id_list = [], [], [], []
plick_by_mouse, dprime_by_mouse = [], []
n_pooled = 0

for _, row in mice.iterrows():
    session_id = row["session_id"]
    units_session = units[units["session_id"] == session_id].reset_index(drop=True)
    trials_session = trials[trials["session_id"] == session_id]
    aligned = align_whisker_trials_first_hit(trials_session, TRIAL_WINDOW_PRE, TRIAL_WINDOW_POST)

    if len(aligned) != EXPECTED_N_TRIALS:
        continue  # clipped mouse -- can't share the pooled trial axis

    spike_times_by_cluster = load_session_spike_times(session_id, units_session)
    tensor, time_bins = build_spike_tensor(spike_times_by_cluster, units_session, aligned)  # raw counts, non-negative

    units_session = units_session.copy()
    units_session["ccf_atlas_acronym"] = units_session["ccf_atlas_acronym"].fillna(units_session["ccf_acronym"])
    units_session["ccf_atlas_parent_acronym"] = units_session["ccf_atlas_parent_acronym"].fillna(units_session["ccf_parent_acronym"])
    units_session = allen.create_area_custom_column(units_session)

    tensors.append(tensor)
    area_custom_list.append(units_session["area_acronym_custom"].to_numpy())
    reward_group_list.append(np.full(len(units_session), int(row["reward_group"])))
    mouse_id_list.append(np.full(len(units_session), session_id))
    plick_by_mouse.append(smoothed_plick(aligned, window=10))
    dprime_by_mouse.append(rolling_dprime(trials_session, aligned, window_trials=20))
    n_pooled += 1
    print(f"[{n_pooled}] {session_id}: {tensor.shape[1]} neurons")

pooled_tensor = np.concatenate(tensors, axis=1).astype(np.float32)  # (81, N_total_neurons, 31)
area_custom = np.concatenate(area_custom_list)
reward_group = np.concatenate(reward_group_list)
mouse_id = np.concatenate(mouse_id_list)

# Pooled behavior: mean P(lick) / d-prime across mice at each trial index --
# same quantity the reference pipeline's own "megamouse_aligned_perf" plot
# shows, since every mouse's trial index 0 means the same thing (its own
# first active-context hit).
pooled_plick = np.nanmean(np.stack(plick_by_mouse), axis=0)
pooled_dprime = np.nanmean(np.stack(dprime_by_mouse), axis=0)

print(f"\npooled {n_pooled} mice ({(reward_group == 1).sum()} R+ neurons, {(reward_group == 0).sum()} R- neurons)")
print(f"pooled tensor shape: {pooled_tensor.shape}, {pooled_tensor.nbytes / 1e9:.2f} GB (float32)")
print(f"build time: {(time.time() - t_start) / 60:.1f} min")

np.savez(
    OUT_DIR / "pooled_raw.npz",
    tensor=pooled_tensor, time_bins=time_bins,
    area_custom=area_custom, reward_group=reward_group, mouse_id=mouse_id,
    pooled_plick=pooled_plick, pooled_dprime=pooled_dprime,
    n_mice=n_pooled,
)
print(f"saved {OUT_DIR / 'pooled_raw.npz'}")
