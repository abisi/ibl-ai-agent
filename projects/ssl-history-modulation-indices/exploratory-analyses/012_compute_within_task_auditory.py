"""Within-task (auditory-only) RHMI/EHMI, day 0 (learning-stage) sessions
and learner mice only -- per user request 2026-08-22, to check whether the
R+/R- distributional difference found on the mixed-modality metric
(009-011) is an artifact of the whisker Go/No-Go task-design asymmetry
between cohorts (R- whisker trials structurally never deliver water --
t-1-rewarded fraction averages 70% for R+ sessions vs 20% for R- on the
mixed metric) rather than a genuine cohort effect. Restricting both `t` and
`t-1` to auditory trials only removes that asymmetry: auditory contingency
(lick=reward) is identical in both cohorts, so this metric's formula is
cohort-agnostic by construction (see history_lib.prep_trials_auditory_only).

Feasibility note (checked before running): auditory hits are common
(median ~30-50/session) so RHMI should be well-powered, but auditory
misses are rare (median 3/session, ssl_task_semantics.md's documented
sparsity issue) -- EHMI (t=incorrect=auditory miss) may end up
underpowered/sparse here. Reported honestly either way, not forced.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from ssl_bwm_trial_prep import list_whisker_training_ephys_sessions, load_reward_group
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
from history_lib import DATASET_ROOT, EVOKED_WINDOW_S, QC_VALUES, prep_trials_auditory_only, compute_index_for_unit

REF_PATH = Path("reports/ssl_analysis/derived/mouse_reference.parquet")
OUT_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/within_task_auditory_day0_learners.parquet")


def main():
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "units.parquet")
    ref = pd.read_parquet(REF_PATH)[["subject_id", "learning_category", "exclude_ephys"]]

    sessions = [s for s in list_whisker_training_ephys_sessions(sessions_tbl) if s[1] == "learning"]
    print(f"{len(sessions)} day-0 (learning-stage) candidate sessions")

    rows = []
    t_start = time.time()
    n_skipped = 0
    for i, (session_id, day_stage) in enumerate(sessions, 1):
        subject_id = sessions_tbl.loc[sessions_tbl.session_id == session_id, "subject_id"].iloc[0]
        reward_group = load_reward_group(subject_id)
        if reward_group is None:
            n_skipped += 1
            continue
        ref_row = ref[ref.subject_id == subject_id]
        if len(ref_row) != 1:
            n_skipped += 1
            continue
        lc, excl = ref_row.iloc[0]["learning_category"], ref_row.iloc[0]["exclude_ephys"]
        if lc not in ("good", "moderate") or excl != 0:
            n_skipped += 1
            continue

        trials = prep_trials_auditory_only(session_id, sessions_tbl, trials_tbl)
        if len(trials) < 8:
            n_skipped += 1
            continue
        start_time = trials["start_time"].to_numpy()
        correct = trials["correct"].to_numpy()
        t1_rewarded = trials["t1_reward_delivered"].to_numpy().astype(bool)

        sess_units = units_tbl[(units_tbl["session_id"] == session_id) & (units_tbl["bc_label"].isin(QC_VALUES))]
        shard = load_spike_shard(DATASET_ROOT / "spikes" / session_id)
        spike_times_all = shard["spike_times_seconds"]
        spike_clusters_dense = shard["spike_clusters"]
        cluster_ids = shard["cluster_ids"]

        n_units = 0
        for _, urow in sess_units.iterrows():
            cid = urow["cluster_id"]
            dense_idx = np.where(cluster_ids == cid)[0]
            if len(dense_idx) == 0:
                continue
            unit_spikes = np.sort(spike_times_all[spike_clusters_dense == dense_idx[0]])

            rhmi = compute_index_for_unit(unit_spikes, start_time, correct, t1_rewarded, EVOKED_WINDOW_S)
            ehmi = compute_index_for_unit(unit_spikes, start_time, ~correct, t1_rewarded, EVOKED_WINDOW_S)

            rows.append({
                "session_id": session_id, "mouse_id": subject_id, "reward_group": reward_group,
                "day_stage": day_stage, "cluster_id": int(cid), "bc_label": urow["bc_label"],
                "n_aud_trials": len(trials), "n_aud_correct": int(correct.sum()), "n_aud_incorrect": int((~correct).sum()),
                **{f"rhmi_{k}": v for k, v in rhmi.items()},
                **{f"ehmi_{k}": v for k, v in ehmi.items()},
            })
            n_units += 1

        elapsed = time.time() - t_start
        print(f"[{i}/{len(sessions)}] {session_id} ({reward_group}): {len(trials)} auditory trials, "
              f"{n_units} units (total elapsed {elapsed/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n{n_skipped} sessions skipped (no reward_group / not learner / too few auditory trials)")
    print(f"Wrote {len(df)} unit rows ({df['session_id'].nunique()} sessions, {df['mouse_id'].nunique()} mice) to {OUT_PATH}")
    print(df.groupby("reward_group")["mouse_id"].nunique())
    print("\nRHMI/EHMI valid-index counts:")
    print(f"  RHMI: {df['rhmi_index'].notna().sum()}/{len(df)}")
    print(f"  EHMI: {df['ehmi_index'].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    main()
