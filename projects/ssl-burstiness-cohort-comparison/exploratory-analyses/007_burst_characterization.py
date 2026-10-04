"""Deeper burst-structure characterization (spikes/burst, burst duration,
inter-burst interval) on the validated small subset, requested as a
follow-up to see more detail on burst identification itself.
"""
from __future__ import annotations

import sys
import importlib
from pathlib import Path

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

burst_lib = importlib.import_module("000_burst_lib")
loader = importlib.import_module("001_small_subset_load")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    all_run_rows = []
    per_unit_rows = []

    for day_stage, files in [("learning", loader.LEARNING_FILES), ("expert", loader.EXPERT_FILES)]:
        ut, tt = loader.load_arm(day_stage, files)
        ut = loader.apply_mouse_filters(ut, ref_df)
        ut = ut[ut["quality_label"] != "non-soma"].copy()
        for session_id, sess_units in ut.groupby("session_id"):
            sess_trials = tt[tt.session_id == session_id]
            wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
            for t in sess_units.itertuples():
                spikes = np.sort(np.asarray(t.spike_times, dtype=float))
                clean = burst_lib.excise_whisker_dead_zone(spikes, wh_starts)
                runs = burst_lib.burst_runs(clean)
                for r in runs:
                    r["unit_uid"] = t.unit_uid
                    r["day_stage"] = day_stage
                    r["quality_label"] = t.quality_label
                all_run_rows.extend(runs)
                per_unit_rows.append({
                    "unit_uid": t.unit_uid, "day_stage": day_stage, "quality_label": t.quality_label,
                    "n_spikes_clean": len(clean), "n_bursts": len(runs),
                })

    runs_df = pd.DataFrame(all_run_rows)
    units_df = pd.DataFrame(per_unit_rows)
    runs_df.to_parquet(ARTIFACTS_DIR / "small_subset_burst_runs.parquet", index=False)

    print(f"Total bursts detected: {len(runs_df)} across {len(units_df)} units")
    print(f"\nSpikes per burst:\n{runs_df['n_spikes'].describe()}")
    print(f"\nBurst duration (ms):\n{(runs_df['duration_s']*1000).describe()}")
    units_df["bursts_per_1k_spikes"] = units_df["n_bursts"] / units_df["n_spikes_clean"] * 1000
    print(f"\nBursts per unit (units with >=200 spikes):\n{units_df[units_df.n_spikes_clean>=200]['n_bursts'].describe()}")

    # inter-burst interval: gap between consecutive bursts' start times, per unit
    ibi_rows = []
    for unit_uid, g in runs_df.groupby("unit_uid"):
        starts = np.sort(g["start_time"].to_numpy())
        if len(starts) >= 2:
            ibi_rows.extend(np.diff(starts))
    ibi = np.array(ibi_rows)
    print(f"\nInter-burst interval (s), n={len(ibi)}:\n{pd.Series(ibi).describe()}")

    # Two MUA units (very high, sustained firing rate: 25-44 Hz) dominate the
    # extreme tail (max 11,393 spikes / 26s in one "burst") -- a real,
    # expected limitation of pure-ISI burst detection on noisy/pooled MUA,
    # not a bug (checked directly: both quality_label=='mua'). Cap the
    # plotted range at the 99.9th percentile so the bulk of the (real,
    # classically-sized) distribution is visible; the long tail is reported
    # separately in text, not hidden.
    n_spikes_cap = int(runs_df["n_spikes"].quantile(0.999))
    dur_cap_ms = (runs_df["duration_s"] * 1000).quantile(0.999)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(runs_df["n_spikes"].clip(upper=n_spikes_cap), bins=np.arange(2, n_spikes_cap + 2) - 0.5, color="#4C72B0")
    axes[0].set_xlabel(f"spikes per burst (capped at p99.9={n_spikes_cap})"); axes[0].set_ylabel("count"); axes[0].set_yscale("log")
    axes[0].set_title(f"Spikes/burst (n={len(runs_df):,} bursts)")

    axes[1].hist((runs_df["duration_s"] * 1000).clip(upper=dur_cap_ms), bins=50, color="#4C72B0")
    axes[1].set_xlabel(f"burst duration (ms, capped at p99.9={dur_cap_ms:.0f})"); axes[1].set_title("Burst duration")

    axes[2].hist(np.log10(ibi[ibi>0]), bins=50, color="#4C72B0")
    axes[2].set_xlabel("inter-burst interval (log10 s)"); axes[2].set_title("Inter-burst interval")
    fig.suptitle("Burst structure characterization (small validated subset, 6 sessions)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "burst_structure_characterization.png", dpi=140, bbox_inches="tight")
    print(f"\nWrote {ARTIFACTS_DIR / 'burst_structure_characterization.png'}")
    print(f"\nExtreme tail (>p99.9, {n_spikes_cap} spikes): {(runs_df['n_spikes']>n_spikes_cap).sum()} bursts, "
          f"from {runs_df.loc[runs_df['n_spikes']>n_spikes_cap, 'unit_uid'].nunique()} units")


if __name__ == "__main__":
    main()
