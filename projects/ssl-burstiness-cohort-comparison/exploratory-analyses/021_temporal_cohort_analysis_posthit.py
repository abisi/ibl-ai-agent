"""Aggregate post-first-hit per-unit temporal bins to per-mouse
trajectories, plot R+ vs R- cohort-level curves, and test via mouse-block
PERMANOVA -- mirrors 014_temporal_cohort_analysis.py exactly, reading the
*_posthit parquets instead. No BH-FDR anywhere in this script (it never
used per-area post-hoc), so "no multiple-testing correction" doesn't change
anything here beyond what 014 already did.
"""
from __future__ import annotations

import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

temporal_cohort = importlib.import_module("014_temporal_cohort_analysis")
aggregate_to_mice = temporal_cohort.aggregate_to_mice
plot_cohort_trajectory = temporal_cohort.plot_cohort_trajectory
run_permanova_on_trajectory = temporal_cohort.run_permanova_on_trajectory

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
N_BINS = 20
ABS_BINS_FOR_TEST = 15


def main() -> None:
    norm_cols = [f"norm_bin_{i}" for i in range(N_BINS)]
    abs_cols = [f"abs_bin_{i}" for i in range(N_BINS)]
    norm_x = (np.arange(N_BINS) + 0.5) / N_BINS
    abs_x = (np.arange(N_BINS) + 0.5) * 3.0

    test_results = []
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    panel_i = 0
    for day_stage in ["learning", "expert"]:
        df = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_temporal_metrics_posthit.parquet")
        for scope_name, scope_filter in [
            ("all_mice", df["reward_group"].isin(["R+", "R-"])),
            ("learners_only", df["learning_category"].isin(["good", "moderate"])),
        ]:
            scoped = df[scope_filter]
            for tier_name, tier_filter in [
                ("good", scoped["quality_label"] == "good"),
                ("good_mua", scoped["quality_label"].isin(["good", "mua"])),
            ]:
                tiered = scoped[tier_filter]
                label = f"{day_stage}_{scope_name}_{tier_name}"
                per_mouse_norm = aggregate_to_mice(tiered, norm_cols)
                per_mouse_abs = aggregate_to_mice(tiered, abs_cols)

                ax = axes.flat[panel_i]
                plot_cohort_trajectory(per_mouse_norm, norm_cols, norm_x, "post-hit progression (fraction)", ax, f"{label}\n(normalized)")
                panel_i += 1
                ax = axes.flat[panel_i]
                plot_cohort_trajectory(per_mouse_abs, abs_cols, abs_x, "elapsed time post-hit (min)", ax, f"{label}\n(absolute)")
                panel_i += 1

                r_norm = run_permanova_on_trajectory(per_mouse_norm, norm_cols, f"{label}_normalized")
                r_abs = run_permanova_on_trajectory(per_mouse_abs, abs_cols[:ABS_BINS_FOR_TEST], f"{label}_absolute_0-45min")
                test_results.append(r_norm)
                test_results.append(r_abs)
                print(f"{label}: norm p={r_norm.get('p_value', 'NA')}, abs(0-45min) p={r_abs.get('p_value', 'NA')}")

    fig.suptitle("Post-first-hit burstiness across session, R+ vs R- (mean +/- SEM across mice)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "temporal_cohort_trajectories_posthit.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote {ARTIFACTS_DIR / 'temporal_cohort_trajectories_posthit.png'}")

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(ARTIFACTS_DIR / "temporal_permanova_summary_posthit.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
