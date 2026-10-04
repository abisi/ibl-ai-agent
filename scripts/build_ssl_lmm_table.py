"""Build the per-unit x stim_type wide table of post-minus-pre deltas for
baseline, evoked, and evoked-baseline-corrected firing rate, the input to
the LMM hierarchy. One row per (unit, trial_type).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DERIVED_DIR = Path("reports/ssl_analysis/derived")


def main() -> None:
    rates = pd.read_parquet(DERIVED_DIR / "unit_condition_rates.parquet")
    stim_counts = pd.read_parquet(DERIVED_DIR / "mouse_active_stim_counts.parquet")

    idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "trial_type"]
    wide = rates.pivot_table(index=idx_cols, columns="passive_epoch",
                              values=["baseline_rate_hz", "evoked_rate_hz", "evoked_corrected_hz"])
    wide.columns = [f"{val}__{epoch}" for val, epoch in wide.columns]
    wide = wide.reset_index()

    before = len(wide)
    wide = wide.dropna(subset=[c for c in wide.columns if c.startswith(("baseline_rate_hz__", "evoked_rate_hz__", "evoked_corrected_hz__"))])
    print(f"Dropped {before - len(wide)} unit-trialtype rows missing pre or post (of {before})")

    wide["delta_baseline"] = wide["baseline_rate_hz__passive_post"] - wide["baseline_rate_hz__passive_pre"]
    wide["delta_evoked"] = wide["evoked_rate_hz__passive_post"] - wide["evoked_rate_hz__passive_pre"]
    wide["delta_corrected"] = wide["evoked_corrected_hz__passive_post"] - wide["evoked_corrected_hz__passive_pre"]

    wide["n_stim"] = None
    is_wh = wide["trial_type"] == "whisker_trial"
    stim_by_mouse = stim_counts.set_index("mouse_id")
    wide.loc[is_wh, "n_stim"] = wide.loc[is_wh, "mouse_id"].map(stim_by_mouse["n_active_whisker_trial"])
    wide.loc[~is_wh, "n_stim"] = wide.loc[~is_wh, "mouse_id"].map(stim_by_mouse["n_active_auditory_trial"])
    wide["n_stim"] = wide["n_stim"].astype(float)

    n_missing_stim = wide["n_stim"].isna().sum()
    print(f"Rows missing n_stim (mouse not in active-count table): {n_missing_stim}")
    wide = wide.dropna(subset=["n_stim"])

    wide["stim_type"] = wide["trial_type"].map({"whisker_trial": "whisker", "auditory_trial": "auditory"})
    wide["reward_group"] = wide["reward_group"].astype("category")
    wide["stim_type"] = wide["stim_type"].astype("category")
    wide["area_group"] = wide["area_group"].astype("category")

    out_path = DERIVED_DIR / "lmm_table.parquet"
    wide.to_parquet(out_path, index=False)
    print(f"Wrote {len(wide)} rows to {out_path}")
    print(f"Mice: {wide['mouse_id'].nunique()}, units: {wide.groupby(['session_id','cluster_id']).ngroups}")
    print(wide.groupby(["stim_type", "reward_group"]).size())
    print()
    print(wide[["delta_baseline", "delta_evoked", "delta_corrected", "n_stim"]].describe())


if __name__ == "__main__":
    main()
