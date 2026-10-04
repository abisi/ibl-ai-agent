"""Scale the validated single-mouse pipeline (000-004) to all 68 eligible
mice (one ephys session per subject, chosen as the eligible session with the
most active-context whisker trials -- so each mouse contributes one
replicate, per exploration-confirmation SKILL.md's caution about repeated
sessions from the same subject not being fully independent).

For each mouse: build the tensor, fit a single rank-4 TCA model (rank chosen
from the 001 pilot), compute P(lick)/d-prime, and run the circular-shift
correlation test per component. Also attaches, per neuron, its Beryl-atlas
area (standard IBL region grouping, mapped from the dataset's raw
`ccf_acronym` via `iblatlas.regions.BrainRegions`, per
`brain_regions_qc.md`) and, per mouse, its reward-group cohort and
day_stage -- both pulled from
`projects/ssl-ks2-dataset-description/artifacts/session_metadata_extra.parquet`
(reward_group = `wh_reward` from the source NWB's experiment_description,
already extracted by that project; not present in ssl_ks2_ephys itself).

Per-mouse results are checkpointed to ../artifacts/per_mouse/<session_id>.npz
so a failed or interrupted run can resume without recomputing finished mice
(per AGENTS.md "save intermediate results").
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensortools as tt
from iblatlas.regions import BrainRegions

from tca_lib import (
    load_session_tables, load_session_spike_times,
    align_whisker_trials_first_hit, build_spike_tensor,
)
from behavior_lib import smoothed_plick, rolling_dprime
from stats_lib import circular_shift_test
from area_lib import acronym_to_beryl

RANK = 4
N_SHIFTS = 1000  # halved from the 003 pilot's 2000, to keep the full-cohort run fast
ARTIFACTS_DIR = Path("../artifacts/per_mouse")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

mice = pd.read_csv("../artifacts/eligible_mice.csv")
sessions, units, trials = load_session_tables()
cohort_meta = pd.read_parquet(
    "../../ssl-ks2-dataset-description/artifacts/session_metadata_extra.parquet"
)[["session_id", "reward_group", "day_stage"]]
mice = mice.merge(cohort_meta, on="session_id", how="left")
br = BrainRegions()

t_run_start = time.time()
for row_idx, row in mice.iterrows():
    session_id = row["session_id"]
    out_path = ARTIFACTS_DIR / f"{session_id}.npz"
    if out_path.exists():
        continue  # checkpoint: already done

    t0 = time.time()
    units_session = units[units["session_id"] == session_id].reset_index(drop=True)
    trials_session = trials[trials["session_id"] == session_id]
    aligned = align_whisker_trials_first_hit(trials_session)
    spike_times_by_cluster = load_session_spike_times(session_id, units_session)
    tensor, time_bins = build_spike_tensor(spike_times_by_cluster, units_session, aligned)

    tensor_min = tensor.min(axis=(0, 2), keepdims=True)
    tensor_max = tensor.max(axis=(0, 2), keepdims=True)
    tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-10)

    model = tt.ncp_hals(tensor_norm, rank=RANK, verbose=False)
    trial_factors, neuron_factors, time_factors = model.factors

    plick = smoothed_plick(aligned, window=10)
    dprime = rolling_dprime(trials_session, aligned, window_trials=20)

    rng = np.random.default_rng(hash(session_id) % (2**32))
    rows = []
    for comp in range(RANK):
        for beh_name, beh_trace in [("plick", plick), ("dprime", dprime)]:
            valid_n = int((~np.isnan(beh_trace)).sum())
            if valid_n < 10:
                rows.append((comp + 1, beh_name, np.nan, np.nan, valid_n))
                continue
            real_r, _, p = circular_shift_test(trial_factors[:, comp], beh_trace, n_shifts=N_SHIFTS, rng=rng)
            rows.append((comp + 1, beh_name, real_r, p, valid_n))
    corr_df = pd.DataFrame(rows, columns=["component", "behavior", "r", "p", "n_valid_trials"])

    # Beryl-atlas area per neuron (standard IBL region grouping), same row
    # order as neuron_factors -- see area_lib.py for why this can't just be
    # br.acronym2acronym (silently drops unmapped rows, breaking alignment).
    beryl_area, _ = acronym_to_beryl(units_session, br)

    np.savez(
        out_path,
        trial_factors=trial_factors, neuron_factors=neuron_factors, time_factors=time_factors,
        time_bins=time_bins, n_neurons=len(units_session), n_trials=len(aligned),
        beryl_area=beryl_area,
        reward_group=row["reward_group"] if pd.notna(row["reward_group"]) else -1,
        day_stage=str(row["day_stage"]),
        corr_component=corr_df["component"].to_numpy(),
        corr_behavior=corr_df["behavior"].to_numpy(),
        corr_r=corr_df["r"].to_numpy(), corr_p=corr_df["p"].to_numpy(),
        corr_n_valid=corr_df["n_valid_trials"].to_numpy(),
    )
    elapsed = time.time() - t0
    print(f"[{row_idx + 1}/{len(mice)}] {session_id}: n_neurons={len(units_session)}, "
          f"n_trials={len(aligned)}, reward_group={row['reward_group']}, {elapsed:.1f}s")

print(f"total run time: {(time.time() - t_run_start) / 60:.1f} min")
