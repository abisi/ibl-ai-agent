"""Compute `quality_label` for the SSL (`ssl_ephys`) unit table, per
`skills/ssl-valid-data/SKILL.md`'s canonical pipeline (user request
2026-09-11: "it is not the bc_label used but the quality_label" -- this is
the reusable-across-all-SSL-projects, more-inclusive QC label that replaces
raw bombcell `bc_label` as the unit-inclusion filter).

Adapts the pipeline for this repo's compressed-dataset setting (no NWB
re-read):
  - bombcell metrics (nSpikes, percentageSpikesMissing_gaussian,
    fractionRPVs_estimatedTauR, maxDriftEstimate, presenceRatio,
    isolationDistance, Lratio) are already columns in `units.parquet` --
    used as-is.
  - `presence_ratio`/`coverage_ratio` (this pipeline's own additional
    metrics) are computed here directly from this project's own spike
    shards (`load_session_unit_spikes`), not re-derived from NWB --
    equivalent inputs (spike_times per unit, grouped per session), see
    `ssl_valid_data_pipeline.md`.
  - the drift-shift joint check (`drift_abs_r`/`drift_shift_test_pval`) is
    SKIPPED here -- those come from a separate NWB-keyed DREDge-CSV pipeline
    (`load_motion_dredge_shift_test_results`) not wired up in this repo.
    `classify_units_quality` no-ops that joint check with a warning when the
    columns are absent (documented, not a silent bug) -- flag this
    explicitly in any report using `quality_label` from this script.

Writes `reports/ssl_analysis/derived/unit_quality_labels.parquet`: one row
per (session_id, cluster_id) with `bc_label`, `presence_ratio`,
`coverage_ratio`, `quality_label`.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_PATH = Path("reports/ssl_analysis/derived/unit_quality_labels.parquet")
N_WORKERS = 8
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _worker_init():
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


def presence_and_coverage_for_session(unit_spikes: dict, cluster_ids: list, bin_size: float = 60.0) -> pd.DataFrame:
    """`unit_metrics_utils.compute_presence_coverage_metrics`'s formulas,
    applied directly to this session's own spike-shard dict (equivalent
    input to that function's `spike_times`/`session_id`-grouped columns,
    since we call this once per session already)."""
    non_empty = {cid: s for cid, s in unit_spikes.items() if len(s) > 0}
    if not non_empty:
        return pd.DataFrame({"cluster_id": cluster_ids, "presence_ratio": 0.0, "coverage_ratio": 0.0})

    rec_start = min(s.min() for s in non_empty.values())
    rec_end = max(s.max() for s in non_empty.values())
    n_bins = int(np.ceil((rec_end - rec_start) / bin_size)) if rec_end > rec_start else 0
    rec_dur = rec_end - rec_start

    rows = []
    for cid in cluster_ids:
        spikes = unit_spikes.get(cid, np.array([]))
        if len(spikes) < 2 or n_bins == 0:
            rows.append(dict(cluster_id=cid, presence_ratio=0.0, coverage_ratio=0.0))
            continue
        bin_idx = np.floor((spikes - rec_start) / bin_size).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        presence_ratio = float(len(np.unique(bin_idx))) / n_bins
        coverage_ratio = float((spikes.max() - spikes.min()) / rec_dur) if rec_dur > 0 else 0.0
        rows.append(dict(cluster_id=cid, presence_ratio=presence_ratio, coverage_ratio=coverage_ratio))
    return pd.DataFrame(rows)


def process_one_session(args: tuple) -> list[dict]:
    """Runs in a worker process. Re-imports here (not at module scope) so
    the thread-limit env vars set in `_worker_init` take effect first, and
    so each worker resolves the M-drive `unit_metrics_utils`/dataset paths
    independently (same pattern as the decode sweeps' `process_one_session`)."""
    session_id, cluster_ids, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import load_session_unit_spikes

    dataset_root = resolve_dataset_dir("ssl_ephys")
    try:
        unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    except Exception as e:  # noqa: BLE001
        return [dict(session_id=session_id, cluster_id=cid, presence_ratio=None, coverage_ratio=None,
                      error=repr(e)) for cid in cluster_ids]
    pc = presence_and_coverage_for_session(unit_spikes, cluster_ids)
    pc["session_id"] = session_id
    pc["error"] = None
    return pc.to_dict("records")


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\neural_utils")
    import unit_metrics_utils

    dataset_root = resolve_dataset_dir("ssl_ephys")
    units = pd.read_parquet(dataset_root / "metadata" / "units.parquet")
    sessions = sorted(units["session_id"].unique())
    print(f"{len(sessions)} sessions, {len(units)} units total, {N_WORKERS} workers", flush=True)

    scripts_dir = str(Path(__file__).resolve().parent)
    tasks = [
        (sid, units.loc[units.session_id == sid, "cluster_id"].unique().tolist(), scripts_dir)
        for sid in sessions
    ]

    t_start = time.time()
    n_done = 0
    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(process_one_session, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            session_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"ERROR {session_id}: {e!r}", flush=True)
                continue
            all_rows.extend(rows)
            n_done += 1
            elapsed = time.time() - t_start
            eta_min = (elapsed / n_done) * (len(tasks) - n_done) / 60.0
            print(f"[{n_done}/{len(tasks)}] {session_id}: {len(rows)} units -- elapsed {elapsed/60:.1f}min, ETA {eta_min:.1f}min", flush=True)

    pc_all = pd.DataFrame(all_rows)
    n_errors = pc_all["error"].notna().sum()
    if n_errors:
        print(f"WARNING: {n_errors} units had spike-load errors (presence/coverage left NaN for them)", flush=True)
    pc_all = pc_all.drop(columns=["error"])

    merged = units.merge(pc_all, on=["session_id", "cluster_id"], how="left", validate="one_to_one")
    print(f"presence/coverage merged: {merged['presence_ratio'].notna().sum()}/{len(merged)} units matched", flush=True)

    classified = unit_metrics_utils.classify_units_quality(merged, label_col="quality_label")

    print("\nbc_label counts:")
    print(classified["bc_label"].value_counts())
    print("\nquality_label counts:")
    print(classified["quality_label"].value_counts())
    recovered = (classified["bc_label"] == "mua") & (classified["quality_label"] == "good")
    demoted = (classified["bc_label"] == "good") & (classified["quality_label"] == "mua")
    print(f"\nrecovered (bc_label=mua -> quality_label=good): {recovered.sum()}")
    print(f"demoted (bc_label=good -> quality_label=mua): {demoted.sum()}")

    out = classified[["session_id", "cluster_id", "bc_label", "presence_ratio", "coverage_ratio", "quality_label"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
