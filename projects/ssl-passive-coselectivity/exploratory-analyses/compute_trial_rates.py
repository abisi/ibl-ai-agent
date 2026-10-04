"""Per-trial baseline and response-window firing rates for whisker/auditory
passive trials. Response window is 5-35ms post-stimulus-onset, per direct
user specification (2026-08-15) for this project's new ROC-based metrics --
a deliberate deviation from ssl-passive-sensory-selectivity's established
10-50ms/10-30ms windows (that project's windows belong to its own
index-based metric; this one is a new metric with its own window). Baseline
-60/-10ms retained from the established convention. The response window
starts at 5ms, just past the -1ms/+4ms whisker artifact dead zone minimum,
so no clipping is needed.

Also tags each session with day_stage (learning = day==0, expert = day>0),
parsed from session_description ("<type>_<day>", day either a plain int or
"+N" for expert days).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ssl-whisker-auditory-cohort-modulation" / "exploratory-analyses"))
sys.path.insert(0, "scripts")
from ssl_wm_lib import tag_passive_subepoch  # noqa: E402
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
ANALYSIS_UNITS = Path("reports/ssl_analysis/derived/analysis_units.parquet")
OUT_DIR = Path("projects/ssl-passive-coselectivity/artifacts")

BASELINE_WINDOW = (-0.060, -0.010)
RESPONSE_WINDOWS = {"w5_35": (0.005, 0.035)}


def parse_day_stage(desc: str | None) -> str | None:
    if desc is None or pd.isna(desc):
        return None
    m = re.search(r"_(\+?\d+)$", str(desc))
    if not m:
        return None
    day = int(m.group(1).replace("+", ""))
    return "learning" if day == 0 else "expert"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = pd.read_parquet(ANALYSIS_UNITS)
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    epochs = pd.read_parquet(DATASET_DIR / "metadata/epochs.parquet")
    sessions = pd.read_parquet(DATASET_DIR / "metadata/sessions.parquet")

    sessions = sessions.copy()
    sessions["day_stage"] = sessions["session_description"].apply(parse_day_stage)
    print(sessions[sessions["session_id"].isin(units["session_id"])].groupby("day_stage")["subject_id"].nunique())

    trials = trials[(trials["context"].astype(str) == "passive") & (trials["trial_type"].isin(["whisker_trial", "auditory_trial"]))]
    trials = tag_passive_subepoch(trials, epochs)
    trials = trials[trials["passive_epoch"].isin(["passive_pre", "passive_post"])].copy()
    trials = trials.merge(sessions[["session_id", "day_stage"]], on="session_id", how="left")
    trials = trials.dropna(subset=["day_stage"])
    print(f"Passive whisker/auditory trials, tagged pre/post + day_stage: {len(trials)}")
    print(trials.groupby(["day_stage", "trial_type", "passive_epoch"]).size())

    session_ids = sorted(units["session_id"].unique())
    all_rows: list[pd.DataFrame] = []

    for i, session_id in enumerate(session_ids, 1):
        sess_units = units[units["session_id"] == session_id]
        sess_trials = trials[trials["session_id"] == session_id]
        if sess_trials.empty or sess_units.empty:
            continue

        shard = load_spike_shard(DATASET_DIR / "spikes" / session_id)
        cluster_ids = shard["cluster_ids"]
        spike_clusters_local = shard["spike_clusters"]
        spike_times = shard["spike_times_seconds"]
        cluster_id_to_local = {int(cid): idx for idx, cid in enumerate(cluster_ids)}
        order = np.argsort(spike_clusters_local, kind="stable")
        sc_sorted = spike_clusters_local[order]
        st_sorted = spike_times[order]
        boundaries = np.searchsorted(sc_sorted, np.arange(len(cluster_ids) + 1))

        for trial_type in ("whisker_trial", "auditory_trial"):
            tt_trials = sess_trials[sess_trials["trial_type"] == trial_type]
            if tt_trials.empty:
                continue
            trial_starts = tt_trials["start_time"].to_numpy()
            passive_epoch = tt_trials["passive_epoch"].to_numpy()
            day_stage = tt_trials["day_stage"].to_numpy()
            trial_id = tt_trials["trial_id"].to_numpy() if "trial_id" in tt_trials.columns else np.arange(len(tt_trials))

            base_lo = trial_starts + BASELINE_WINDOW[0]
            base_hi = trial_starts + BASELINE_WINDOW[1]
            resp_bounds = {
                name: (trial_starts + w[0], trial_starts + w[1]) for name, w in RESPONSE_WINDOWS.items()
            }

            for _, unit_row in sess_units.iterrows():
                local_idx = cluster_id_to_local.get(int(unit_row["cluster_id"]))
                if local_idx is None:
                    continue
                lo, hi = boundaries[local_idx], boundaries[local_idx + 1]
                u_spikes = st_sorted[lo:hi]

                base_n = np.searchsorted(u_spikes, base_hi, side="left") - np.searchsorted(u_spikes, base_lo, side="left")
                base_rate = base_n / (BASELINE_WINDOW[1] - BASELINE_WINDOW[0])

                row = {
                    "session_id": session_id, "cluster_id": int(unit_row["cluster_id"]),
                    "mouse_id": unit_row["mouse_id"], "area_group": unit_row["area_group"],
                    "reward_group": unit_row["reward_group"], "trial_type": trial_type,
                    "passive_epoch": passive_epoch, "day_stage": day_stage, "trial_id": trial_id,
                    "baseline_rate_hz": base_rate,
                }
                for name, (lo_t, hi_t) in resp_bounds.items():
                    n = np.searchsorted(u_spikes, hi_t, side="left") - np.searchsorted(u_spikes, lo_t, side="left")
                    row[f"response_rate_hz_{name}"] = n / (RESPONSE_WINDOWS[name][1] - RESPONSE_WINDOWS[name][0])

                all_rows.append(pd.DataFrame(row))

        if i % 10 == 0 or i == len(session_ids):
            print(f"[{i}/{len(session_ids)}] {session_id} done", flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    out_path = OUT_DIR / "trial_rates.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nWrote {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
