"""RHMI/EHMI over the full whisker-training ephys cohort (95 sessions, per
ssl_bwm_trial_prep.list_whisker_training_ephys_sessions), QC = good+non-soma
bc_label (per user request 2026-08-21 -- narrower than the good+mua scope
used elsewhere in ssl-bwm-style-single-cell-decoding; non-soma is Bombcell's
third QC class here, mua is NOT included in this project's scope). See
../question.md for the metric definition.

Checkpoints (writes the partial parquet + reruns 005_stats.py and
006_plot_publication_figure.py) every CHECKPOINT_EVERY_MICE newly-completed
mice (per user request 2026-08-21), not every session -- a mouse can
contribute more than one session (expert-stage), so mouse count is tracked
separately from session count. Checkpoint failures (e.g. too little data yet
for the mixed model to fit) are caught and logged, not fatal to the main
loop.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from ssl_bwm_trial_prep import list_whisker_training_ephys_sessions, load_reward_group
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
from history_lib import DATASET_ROOT, EVOKED_WINDOW_S, prep_trials, compute_index_for_unit

QC_VALUES = ("good", "non-soma")
OUT_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/full_cohort_unit_indices.parquet")
CHECKPOINT_EVERY_MICE = 10
STATS_SCRIPT = Path("projects/ssl-history-modulation-indices/exploratory-analyses/005_stats.py")
PLOT_SCRIPT = Path("projects/ssl-history-modulation-indices/exploratory-analyses/006_plot_publication_figure.py")


def checkpoint_and_refresh(rows: list[dict], n_mice: int) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"  [checkpoint] {n_mice} mice, {len(df)} unit rows written to {OUT_PATH}", flush=True)
    for script in (STATS_SCRIPT, PLOT_SCRIPT):
        try:
            subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True, timeout=180)
            print(f"  [checkpoint] {script.name} refreshed", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"  [checkpoint] {script.name} FAILED (likely too little data yet): {e.stderr[-500:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  [checkpoint] {script.name} timed out, skipping this round", flush=True)


def main():
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "units.parquet")

    sessions = list_whisker_training_ephys_sessions(sessions_tbl)
    print(f"{len(sessions)} candidate whisker-training ephys sessions")

    rows = []
    t_start = time.time()
    n_skipped = 0
    seen_mice = set()
    last_checkpoint_n_mice = 0
    for i, (session_id, day_stage) in enumerate(sessions, 1):
        subject_id = sessions_tbl.loc[sessions_tbl.session_id == session_id, "subject_id"].iloc[0]
        reward_group = load_reward_group(subject_id)
        if reward_group is None:
            n_skipped += 1
            continue

        trials = prep_trials(session_id, sessions_tbl, trials_tbl, reward_group)
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
                **{f"rhmi_{k}": v for k, v in rhmi.items()},
                **{f"ehmi_{k}": v for k, v in ehmi.items()},
            })
            n_units += 1

        elapsed = time.time() - t_start
        seen_mice.add(subject_id)
        print(f"[{i}/{len(sessions)}] {session_id} ({reward_group}, {day_stage}): {n_units} units "
              f"(total elapsed {elapsed/60:.1f} min, {len(seen_mice)} mice seen so far)", flush=True)

        if len(seen_mice) - last_checkpoint_n_mice >= CHECKPOINT_EVERY_MICE:
            last_checkpoint_n_mice = len(seen_mice)
            checkpoint_and_refresh(rows, last_checkpoint_n_mice)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    checkpoint_and_refresh(rows, len(seen_mice))
    print(f"\n{n_skipped} sessions skipped (no reward_group / too few trials)")
    print(f"Wrote {len(df)} unit rows ({df['session_id'].nunique()} sessions, {df['mouse_id'].nunique()} mice) to {OUT_PATH}")
    print(df.groupby("reward_group")["mouse_id"].nunique())


if __name__ == "__main__":
    main()
