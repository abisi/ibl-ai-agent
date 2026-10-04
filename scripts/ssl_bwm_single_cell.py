"""Generic driver for the SSL BWM-style single-cell tests (Modality,
Response, Prior-outcome -- Outcome/feedback dropped, see question.md: reward
status is a deterministic function of (modality, response) in this task, so
a reward-vs-non-reward test conditioned on response is structurally
degenerate). All three tests share the same candidate-row population
(whisker_trial/auditory_trial rows of the active epoch, from
`ssl_bwm_trial_prep.prep_session`) and the same 2x2-stratified
condition-combined shuffle-test machinery
(`ssl_bwm_stats_util.combined_stratified_pvalue`); they differ only in which
of {is_whisker, response, t1_rewarded} is the tested factor vs. the two
stratifying dimensions, and in the event window and block_aware flag.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ssl_bwm_stats_util import combined_stratified_pvalue, benjamini_hochberg
from ssl_bwm_windows import unit_rates_for_trials
from ssl_bwm_trial_prep import prep_session
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard


def unit_test_pvalue(
    unit_spikes: np.ndarray,
    start_time: np.ndarray,
    is_whisker: np.ndarray,
    window: tuple[float, float],
    factor: np.ndarray,
    strata1: np.ndarray,
    strata2: np.ndarray,
    run_index: np.ndarray,
    block_aware: bool,
    n_shuf: int,
    rng: np.random.Generator,
) -> float:
    rates = unit_rates_for_trials(unit_spikes, start_time, is_whisker, window)
    valid = ~np.isnan(rates)
    strata = []
    for s1 in (False, True):
        for s2 in (False, True):
            m = valid & (strata1 == s1) & (strata2 == s2)
            entry = {"x": rates[m & factor], "y": rates[m & ~factor]}
            if block_aware:
                entry["bx"] = run_index[m & factor]
                entry["by"] = run_index[m & ~factor]
            strata.append(entry)
    return combined_stratified_pvalue(strata, n_shuf, block_aware, rng)


def run_small_scale_test(
    test_name: str,
    dataset_root: Path,
    sessions: list[tuple[str, str]],
    window: tuple[float, float],
    factor_col: str,
    strata1_col: str,
    strata2_col: str,
    block_aware: bool,
    n_shuf: int,
    n_units_subset: int,
    rng: np.random.Generator,
    out_dir: Path,
    qc_col: str = "bc_label",
    qc_values: tuple[str, ...] = ("good",),
) -> list[dict]:
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(dataset_root / "metadata" / "units.parquet")

    all_results = []
    for session_id, day_stage in sessions:
        print(f"\n=== [{test_name}] {session_id} ({day_stage}) ===")
        prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
        if prepped is None:
            print("  SKIP: no usable reward_group")
            continue
        trials = prepped["trials"]
        candidate = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
        # t-1 outcome needed here: prep_session keeps a session-initial whisker trial with t1_rewarded = NaN since
        # 2026-09-28 (first whisker trial never removed) -- exclude it for this t-1-dependent analysis only.
        candidate = candidate[candidate["t1_rewarded"].notna()].reset_index(drop=True)

        cols = {
            "is_whisker": (candidate["trial_type"] == "whisker_trial").to_numpy(),
            "response": candidate["lick_flag"].to_numpy().astype(bool),
            "t1_rewarded": candidate["t1_rewarded"].to_numpy().astype(bool),
        }
        run_index = candidate["run_index"].to_numpy()
        start_time = candidate["start_time"].to_numpy()
        is_whisker = cols["is_whisker"]
        factor, strata1, strata2 = cols[factor_col], cols[strata1_col], cols[strata2_col]

        print(f"  subject={prepped['subject_id']} reward_group={prepped['reward_group']} "
              f"n_candidate_rows={len(candidate)} whisker={int(is_whisker.sum())} auditory={int((~is_whisker).sum())} "
              f"n_{factor_col}_true={int(factor.sum())}")

        sess_units = units_tbl[(units_tbl["session_id"] == session_id) & (units_tbl[qc_col].isin(qc_values))]
        print(f"  good units: {len(sess_units)}; using first {n_units_subset} for timing check")
        sess_units_subset = sess_units.head(n_units_subset)

        shard = load_spike_shard(dataset_root / "spikes" / session_id)
        spike_times_all = shard["spike_times_seconds"]
        spike_clusters_dense = shard["spike_clusters"]
        cluster_ids = shard["cluster_ids"]

        t0 = time.time()
        pvals, cluster_id_list = [], []
        for _, urow in sess_units_subset.iterrows():
            cid = urow["cluster_id"]
            dense_idx = np.where(cluster_ids == cid)[0]
            if len(dense_idx) == 0:
                continue
            unit_spikes = np.sort(spike_times_all[spike_clusters_dense == dense_idx[0]])
            p = unit_test_pvalue(unit_spikes, start_time, is_whisker, window, factor, strata1, strata2, run_index, block_aware, n_shuf, rng)
            pvals.append(p)
            cluster_id_list.append(int(cid))

        elapsed = time.time() - t0
        n_done = len(pvals)
        per_unit = elapsed / max(n_done, 1)
        good_n = int(len(sess_units))
        print(f"  {n_done} units tested in {elapsed:.1f}s ({per_unit:.3f}s/unit) -> "
              f"est. full-session = {per_unit * good_n / 60:.1f} min ({good_n} good units)")

        pvals_arr = np.array(pvals)
        q = benjamini_hochberg(pvals_arr)
        print(f"  p-values: n={len(pvals_arr)} n_raw<0.05={int(np.nansum(pvals_arr < 0.05))} "
              f"n_FDR<0.05={int(np.nansum(q < 0.05))}")
        print(f"  {np.round(pvals_arr, 4)}")

        all_results.append({
            "test": test_name, "session_id": session_id, "day_stage": day_stage,
            "cluster_id": cluster_id_list, "p_raw": pvals_arr.tolist(), "p_fdr": q.tolist(),
            "elapsed_s": elapsed, "n_units": n_done, "good_units_total": good_n,
        })

    out_path = out_dir / f"{test_name}_small_scale_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")
    return all_results


def run_full_test(
    test_name: str,
    dataset_root: Path,
    sessions: list[tuple[str, str]],
    window: tuple[float, float],
    factor_col: str,
    strata1_col: str,
    strata2_col: str,
    block_aware: bool,
    n_shuf: int,
    rng: np.random.Generator,
    qc_col: str = "bc_label",
    qc_values: tuple[str, ...] = ("good",),
    resume_rows: pd.DataFrame | None = None,
    on_session_done=None,
) -> pd.DataFrame:
    """Same test as `run_small_scale_test`, but over ALL good units in every
    session (no subset cap), returned as one long DataFrame
    (test, session_id, day_stage, cluster_id, p_raw) for confirmatory-run
    aggregation. Prints per-session progress/timing as it goes -- this is a
    multi-hour job for the full cohort.

    Per-session checkpointing (added 2026-08-20 after a killed job lost
    ~15h of in-progress-test work that had only been checkpointed at the
    end of the whole test): pass `resume_rows` (a prior partial result for
    this exact test, e.g. loaded from a `{test}_partial.parquet`) to skip
    its already-covered sessions, and `on_session_done(rows_so_far_df)` to
    persist progress after every session, not just at the end."""
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(dataset_root / "metadata" / "units.parquet")

    done_session_ids = set(resume_rows["session_id"].unique()) if resume_rows is not None and len(resume_rows) else set()
    rows = resume_rows.to_dict("records") if resume_rows is not None and len(resume_rows) else []
    if done_session_ids:
        print(f"{test_name}: resuming, {len(done_session_ids)} sessions already done ({len(rows)} rows)", flush=True)

    t_start_all = time.time()
    for i, (session_id, day_stage) in enumerate(sessions, 1):
        if session_id in done_session_ids:
            continue
        t0 = time.time()
        prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
        if prepped is None:
            print(f"[{i}/{len(sessions)}] {test_name} {session_id}: SKIP (no usable reward_group)", flush=True)
            continue
        trials = prepped["trials"]
        candidate = trials[trials["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
        # t-1 outcome needed here: prep_session keeps a session-initial whisker trial with t1_rewarded = NaN since
        # 2026-09-28 (first whisker trial never removed) -- exclude it for this t-1-dependent analysis only.
        candidate = candidate[candidate["t1_rewarded"].notna()].reset_index(drop=True)
        if len(candidate) < 8:
            print(f"[{i}/{len(sessions)}] {test_name} {session_id}: SKIP (only {len(candidate)} candidate trials)", flush=True)
            continue

        cols = {
            "is_whisker": (candidate["trial_type"] == "whisker_trial").to_numpy(),
            "response": candidate["lick_flag"].to_numpy().astype(bool),
            "t1_rewarded": candidate["t1_rewarded"].to_numpy().astype(bool),
        }
        run_index = candidate["run_index"].to_numpy()
        start_time = candidate["start_time"].to_numpy()
        is_whisker = cols["is_whisker"]
        factor, strata1, strata2 = cols[factor_col], cols[strata1_col], cols[strata2_col]

        sess_units = units_tbl[(units_tbl["session_id"] == session_id) & (units_tbl[qc_col].isin(qc_values))]

        shard = load_spike_shard(dataset_root / "spikes" / session_id)
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
            p = unit_test_pvalue(unit_spikes, start_time, is_whisker, window, factor, strata1, strata2, run_index, block_aware, n_shuf, rng)
            rows.append({"test": test_name, "session_id": session_id, "day_stage": day_stage, "cluster_id": int(cid), "p_raw": p})
            n_units += 1

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start_all
        print(f"[{i}/{len(sessions)}] {test_name} {session_id} ({day_stage}): {n_units} units in {elapsed:.1f}s "
              f"(total elapsed {total_elapsed/60:.1f} min)", flush=True)

        if on_session_done is not None:
            on_session_done(pd.DataFrame(rows))

    return pd.DataFrame(rows)
