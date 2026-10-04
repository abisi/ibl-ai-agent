"""RHMI/EHMI example computation for one R+ and one R- session. See
../question.md for the locked metric definition. Whisker+auditory trials
pooled (no modality split), no_stim_trial excluded entirely.

index = (mean_FR(t | t-1 rewarded) - mean_FR(t | t-1 non-rewarded)) / mean_FR(all t-trials)

where `t` = correct trials (RHMI) or incorrect trials (EHMI), `t-1 rewarded`
= literal water delivery on the immediately preceding active trial (any
modality) -- NOT the same as `correct`, since R- whisker trials never
deliver water regardless of outcome.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from ssl_bwm_trial_prep import load_reward_group
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
from history_lib import DATASET_ROOT, EVOKED_WINDOW_S, MIN_TRIALS_PER_BUCKET, QC_VALUES, prep_trials, compute_index_for_unit

EXAMPLES = [
    ("MH030_20250501_151231", "R+"),
    ("MH023_20250316_110814", "R-"),
]


def main():
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "units.parquet")

    all_rows = []
    for session_id, expected_rg in EXAMPLES:
        subject_id = sessions_tbl.loc[sessions_tbl.session_id == session_id, "subject_id"].iloc[0]
        reward_group = load_reward_group(subject_id)
        assert reward_group == expected_rg, f"{session_id}: expected {expected_rg}, got {reward_group}"

        trials = prep_trials(session_id, sessions_tbl, trials_tbl, reward_group)
        start_time = trials["start_time"].to_numpy()
        correct = trials["correct"].to_numpy()
        t1_rewarded = trials["t1_reward_delivered"].to_numpy().astype(bool)

        n_correct, n_incorrect = int(correct.sum()), int((~correct).sum())
        n_t1_rew, n_t1_norew = int(t1_rewarded.sum()), int((~t1_rewarded).sum())
        print(f"\n=== {session_id} ({reward_group}), subject {subject_id} ===")
        print(f"  {len(trials)} candidate (whisker+auditory, active) trials after dropping first-trial-of-session")
        print(f"  correct={n_correct} incorrect={n_incorrect}  |  t-1 rewarded={n_t1_rew} t-1 non-rewarded={n_t1_norew}")

        sess_units = units_tbl[(units_tbl["session_id"] == session_id) & (units_tbl["bc_label"].isin(QC_VALUES))]
        print(f"  {len(sess_units)} good units")

        shard = load_spike_shard(DATASET_ROOT / "spikes" / session_id)
        spike_times_all = shard["spike_times_seconds"]
        spike_clusters_dense = shard["spike_clusters"]
        cluster_ids = shard["cluster_ids"]

        for _, urow in sess_units.iterrows():
            cid = urow["cluster_id"]
            dense_idx = np.where(cluster_ids == cid)[0]
            if len(dense_idx) == 0:
                continue
            unit_spikes = np.sort(spike_times_all[spike_clusters_dense == dense_idx[0]])

            rhmi = compute_index_for_unit(unit_spikes, start_time, correct, t1_rewarded, EVOKED_WINDOW_S)
            ehmi = compute_index_for_unit(unit_spikes, start_time, ~correct, t1_rewarded, EVOKED_WINDOW_S)

            all_rows.append({
                "session_id": session_id, "reward_group": reward_group, "cluster_id": int(cid),
                **{f"rhmi_{k}": v for k, v in rhmi.items()},
                **{f"ehmi_{k}": v for k, v in ehmi.items()},
            })

    df = pd.DataFrame(all_rows)
    out_path = Path("projects/ssl-history-modulation-indices/exploratory-analyses/example_unit_indices.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df)} unit rows to {out_path}")

    for session_id, rg in EXAMPLES:
        sub = df[df.session_id == session_id]
        print(f"\n--- {session_id} ({rg}) ---")
        for metric in ("rhmi", "ehmi"):
            valid = sub[f"{metric}_index"].notna()
            print(f"  {metric.upper()}: {valid.sum()}/{len(sub)} units with a defined index "
                  f"(>= {MIN_TRIALS_PER_BUCKET} trials/bucket); "
                  f"mean={sub.loc[valid, f'{metric}_index'].mean():.3f} "
                  f"median={sub.loc[valid, f'{metric}_index'].median():.3f}")
        print("  first 5 units, full numerator/denominator breakdown:")
        cols = ["cluster_id", "rhmi_n_a", "rhmi_n_b", "rhmi_fr_a", "rhmi_fr_b", "rhmi_denom_mean_fr", "rhmi_index",
                "ehmi_n_a", "ehmi_n_b", "ehmi_fr_a", "ehmi_fr_b", "ehmi_denom_mean_fr", "ehmi_index"]
        print(sub[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
