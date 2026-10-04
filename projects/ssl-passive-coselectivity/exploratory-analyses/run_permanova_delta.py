"""Refined quantity (per user, 2026-08-15): passive_pre precedes the reward
contingency entirely (only in effect during the active block, between
passive_pre and passive_post), so a real R+/R- difference in passive_pre
alone cannot be a consequence of the manipulation -- it can only reflect a
static confound (mouse-level variability correlated with cohort by chance),
not signal. The raw per-epoch PERMANOVAs (run_permanova_responsiveness.py,
run_permanova_selectivity.py) found exactly this pattern (significant in
passive_pre, null in passive_post) -- diagnostic of the confound, not the
scientific result.

Fix: test the CHANGE (delta = post - pre) instead of raw per-epoch levels,
per unit, by cohort. Differencing cancels any static per-mouse/per-unit
confound the same way ssl-whisker-auditory-cohort-modulation's Δ-based LMM
did throughout (never modeled raw pre or post levels separately). Same
PERMANOVA machinery (mouse-block permutation, main + per-area BH-FDR
post-hoc) applied to delta_whisker_auc/delta_auditory_auc (responsiveness)
and delta_selectivity_auc (selectivity, already computed in
compute_direction_change.py's selectivity_change_{w}.parquet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permanova import run_with_posthoc  # noqa: E402

ARTIFACTS_DIR = Path("projects/ssl-passive-coselectivity/artifacts")
WINDOW = "w5_35"


def build_delta_responsiveness() -> pd.DataFrame:
    auc = pd.read_parquet(ARTIFACTS_DIR / f"responsiveness_auc_wide_{WINDOW}.parquet")
    idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "day_stage"]
    wide = auc.pivot_table(index=idx_cols, columns="passive_epoch", values=["whisker_auc", "auditory_auc"], aggfunc="first")
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide = wide.dropna(subset=["whisker_auc__passive_pre", "whisker_auc__passive_post", "auditory_auc__passive_pre", "auditory_auc__passive_post"])
    wide["delta_whisker_auc"] = wide["whisker_auc__passive_post"] - wide["whisker_auc__passive_pre"]
    wide["delta_auditory_auc"] = wide["auditory_auc__passive_post"] - wide["auditory_auc__passive_pre"]
    return wide


def main() -> None:
    delta_resp = build_delta_responsiveness()
    delta_resp.to_parquet(ARTIFACTS_DIR / f"delta_responsiveness_auc_{WINDOW}.parquet", index=False)
    delta_sel = pd.read_parquet(ARTIFACTS_DIR / f"selectivity_change_{WINDOW}.parquet")

    all_main = []
    for day_stage in ["learning", "expert"]:
        sub = delta_resp[delta_resp["day_stage"] == day_stage]
        label = f"delta_responsiveness_{day_stage}"
        print(f"\n===== {label}: n_units={len(sub)}, n_mice={sub['mouse_id'].nunique()} =====")
        main_result, posthoc = run_with_posthoc(sub, ["delta_whisker_auc", "delta_auditory_auc"], area_col="area_group", label_prefix=label)
        all_main.append(main_result)
        posthoc.to_parquet(ARTIFACTS_DIR / f"permanova_delta_responsiveness_posthoc_{day_stage}.parquet", index=False)

        sub2 = delta_sel[delta_sel["day_stage"] == day_stage]
        label2 = f"delta_selectivity_{day_stage}"
        print(f"\n===== {label2}: n_units={len(sub2)}, n_mice={sub2['mouse_id'].nunique()} =====")
        main_result2, posthoc2 = run_with_posthoc(sub2, ["delta_selectivity_auc"], area_col="area_group", label_prefix=label2)
        all_main.append(main_result2)
        posthoc2.to_parquet(ARTIFACTS_DIR / f"permanova_delta_selectivity_posthoc_{day_stage}.parquet", index=False)

    with open(ARTIFACTS_DIR / "permanova_delta_main.json", "w") as f:
        json.dump(all_main, f, indent=2, default=str)
    print("\nWrote permanova_delta_main.json")


if __name__ == "__main__":
    main()
