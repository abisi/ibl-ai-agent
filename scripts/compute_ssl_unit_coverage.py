"""Per-unit coverage-ratio computation for the SSL KS4 analysis, restricted to
sessions that have both ephys and passive_pre/passive_post epochs.

coverage_ratio = (last_spike_time - first_spike_time) / session_duration_s

Also records, per unit, whether it has >=1 spike inside passive_pre and
inside passive_post directly, as a literal cross-check of the "fires in both
passive epochs" criterion that coverage_ratio is used as a proxy for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
OUT_DIR = Path("reports/ssl_analysis/derived")
OUT_PATH = OUT_DIR / "unit_coverage.parquet"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = pd.read_parquet(DATASET_DIR / "metadata/sessions.parquet")
    epochs = pd.read_parquet(DATASET_DIR / "metadata/epochs.parquet")

    usable = sessions[(sessions["has_ephys"]) & (sessions["has_passive_epochs"])].copy()
    usable_ids = usable["session_id"].tolist()
    print(f"Usable sessions (has_ephys & has_passive_epochs): {len(usable_ids)}")

    epochs_u = epochs[epochs["session_id"].isin(usable_ids)]
    epoch_bounds = {
        session_id: {
            row["epoch_name"]: (row["start_time"], row["stop_time"])
            for _, row in grp.iterrows()
        }
        for session_id, grp in epochs_u.groupby("session_id")
    }

    all_rows = []
    for i, session_id in enumerate(usable_ids, 1):
        shard_path = DATASET_DIR / "spikes" / session_id
        try:
            shard = load_spike_shard(shard_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(usable_ids)}] {session_id}: FAILED to load shard ({exc})")
            continue

        cluster_ids = shard["cluster_ids"]
        spike_clusters_local = shard["spike_clusters"]
        spike_times = shard["spike_times_seconds"]

        df = pd.DataFrame({"local_cluster": spike_clusters_local, "t": spike_times})
        agg = df.groupby("local_cluster")["t"].agg(first_spike_time="min", last_spike_time="max", n_spikes="count")
        agg = agg.reindex(range(len(cluster_ids)))
        agg["cluster_id"] = cluster_ids

        bounds = epoch_bounds.get(session_id, {})
        pre = bounds.get("passive_pre")
        post = bounds.get("passive_post")

        if pre is not None:
            pre_mask = (df["t"] >= pre[0]) & (df["t"] < pre[1])
            n_pre = df.loc[pre_mask].groupby("local_cluster").size()
        else:
            n_pre = pd.Series(dtype=int)
        if post is not None:
            post_mask = (df["t"] >= post[0]) & (df["t"] < post[1])
            n_post = df.loc[post_mask].groupby("local_cluster").size()
        else:
            n_post = pd.Series(dtype=int)

        agg["n_spikes_passive_pre"] = agg.index.map(n_pre).fillna(0).astype(int)
        agg["n_spikes_passive_post"] = agg.index.map(n_post).fillna(0).astype(int)
        agg["session_id"] = session_id

        duration_s = float(usable.loc[usable["session_id"] == session_id, "duration_s"].iloc[0])
        agg["session_duration_s"] = duration_s
        agg["coverage_ratio"] = (agg["last_spike_time"] - agg["first_spike_time"]) / duration_s

        all_rows.append(agg.reset_index(drop=True))
        n_spikes_total = int(df.shape[0])
        print(f"[{i}/{len(usable_ids)}] {session_id}: {len(cluster_ids)} units, {n_spikes_total} spikes", flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    cols = [
        "session_id", "cluster_id", "n_spikes", "first_spike_time", "last_spike_time",
        "session_duration_s", "coverage_ratio", "n_spikes_passive_pre", "n_spikes_passive_post",
    ]
    result = result[cols]
    result.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(result)} unit-coverage rows to {OUT_PATH}")
    print(result["coverage_ratio"].describe())


if __name__ == "__main__":
    main()
