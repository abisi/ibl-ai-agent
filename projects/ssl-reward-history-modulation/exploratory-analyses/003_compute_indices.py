"""Compute the 4 reward/choice-history modulation indices per unit.

For each session: trial-bucket membership (which trials are group A / t-1 hit,
vs group B / t-1 non-hit, for each of the 4 index categories) is the same for
every unit, so it's computed once per session from the trial table. Only the
firing-rate lookup is per-unit, via searchsorted into that unit's sorted spike
times -- no giant per-trial-per-unit table is ever materialized.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ibl_ai_agent.datasets.ssl_ks2_ephys import load_spike_shard
from pathlib import Path

sys.path.insert(0, "projects/ssl-reward-history-modulation/exploratory-analyses")
from reward_history_lib import RESPONSE_WINDOW_S, MIN_TRIALS_PER_BUCKET

SPIKES_DIR = Path("reports/datasets/ssl_ks2_ephys/1.0.0/spikes")

trials = pd.read_parquet("projects/ssl-reward-history-modulation/artifacts/trials_with_history.parquet")
units = pd.read_parquet("projects/ssl-reward-history-modulation/artifacts/units_with_area.parquet")

prev_hit_true = (trials["prev_hit"] == True).fillna(False)   # noqa: E712  (nullable-bool comparison, NA -> False on both sides)
prev_hit_false = (trials["prev_hit"] == False).fillna(False)  # noqa: E712

CATEGORIES = {
    "hit_stay": (trials["modality"] == "whisker") & (trials["hit"] == True),
    "nonhit_stay": (trials["modality"] == "whisker") & (trials["hit"] == False),
    "w2a_transition": (trials["modality"] == "auditory") & (trials["hit"] == True) & (trials["prev_modality"] == "whisker"),
    "a2w_transition": (trials["modality"] == "whisker") & (trials["hit"] == True) & (trials["prev_modality"] == "auditory"),
}


def unit_spike_time_lookup(shard):
    """Split shard['spike_times_seconds'] by unit. Returns dict cluster_id -> sorted spike times (s)."""
    order = np.argsort(shard["spike_clusters"], kind="stable")  # stable: preserves time order within each unit
    sorted_clusters = shard["spike_clusters"][order]
    sorted_times = shard["spike_times_seconds"][order]
    n_units = len(shard["cluster_ids"])
    bounds = np.searchsorted(sorted_clusters, np.arange(n_units + 1))
    return {
        int(shard["cluster_ids"][i]): sorted_times[bounds[i]:bounds[i + 1]]
        for i in range(n_units)
    }


def mean_firing_rate(unit_times, stim_onsets):
    """Mean firing rate (Hz) in RESPONSE_WINDOW_S after each stim onset."""
    if len(stim_onsets) == 0:
        return np.nan
    starts = np.searchsorted(unit_times, stim_onsets + RESPONSE_WINDOW_S[0])
    stops = np.searchsorted(unit_times, stim_onsets + RESPONSE_WINDOW_S[1])
    counts = stops - starts
    return counts.mean() / (RESPONSE_WINDOW_S[1] - RESPONSE_WINDOW_S[0])


rows = []
for session_id, session_units in units.groupby("session_id"):
    session_trials = trials[trials["session_id"] == session_id]
    shard = load_spike_shard(SPIKES_DIR / session_id)
    unit_times = unit_spike_time_lookup(shard)

    # Precompute, per category, group-A/group-B stim onset arrays (shared across all units in this session).
    group_stim_onsets = {}
    for cat, base_mask in CATEGORIES.items():
        m = base_mask[session_trials.index]
        group_stim_onsets[cat] = {
            "A": session_trials.loc[m & prev_hit_true[session_trials.index], "stim_onset"].to_numpy(),
            "B": session_trials.loc[m & prev_hit_false[session_trials.index], "stim_onset"].to_numpy(),
        }

    for _, unit in session_units.iterrows():
        times = unit_times.get(int(unit["cluster_id"]))
        if times is None or len(times) == 0:
            continue
        row = {"session_id": session_id, "cluster_id": unit["cluster_id"], "mouse_id": unit["mouse_id"],
               "reward_group": unit["reward_group"], "area_acronym_custom": unit["area_acronym_custom"],
               "bc_label": unit["bc_label"]}
        for cat, groups in group_stim_onsets.items():
            n_a, n_b = len(groups["A"]), len(groups["B"])
            row[f"{cat}_n_a"], row[f"{cat}_n_b"] = n_a, n_b
            if n_a < MIN_TRIALS_PER_BUCKET or n_b < MIN_TRIALS_PER_BUCKET:
                row[cat] = np.nan
                continue
            fr_a = mean_firing_rate(times, groups["A"])
            fr_b = mean_firing_rate(times, groups["B"])
            pooled_mean = (n_a * fr_a + n_b * fr_b) / (n_a + n_b)
            row[cat] = (fr_a - fr_b) / pooled_mean if pooled_mean > 0 else np.nan
        rows.append(row)

index_df = pd.DataFrame(rows)
print(f"{len(index_df)} units processed across {index_df['session_id'].nunique()} sessions")
for cat in CATEGORIES:
    n_valid = index_df[cat].notna().sum()
    print(f"  {cat}: {n_valid}/{len(index_df)} units with a defined index "
          f"(>= {MIN_TRIALS_PER_BUCKET} trials/bucket)")

index_df.to_parquet("projects/ssl-reward-history-modulation/artifacts/unit_indices.parquet", index=False)
