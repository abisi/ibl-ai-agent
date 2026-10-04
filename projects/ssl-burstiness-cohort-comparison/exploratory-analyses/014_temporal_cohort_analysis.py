"""Aggregate the full-population per-unit temporal bins to per-mouse
trajectories, plot R+ vs R- cohort-level curves (normalized session
progression + absolute time), and test whether the joint trajectory shape
differs by cohort via mouse-block-permutation PERMANOVA (reusing
permanova.py, already fixed for performance/gating this session) -- treating
the 20 bins as multivariate features, exactly like burst_index_whisker/
auditory were treated as scalar features earlier, just higher-dimensional.

Note on method (corrects an earlier suggestion in conversation): a
circular/block-shift permutation (the TCA precedent in
ssl_analysis_patterns.md) is for testing whether a trial-mode loading
correlates with a *separate* time-varying signal, preserving each series'
own autocorrelation while destroying their alignment. That's not this
question -- "does the joint 20-bin trajectory differ by cohort" is a
between-mouse group-difference question on multivariate features, which is
exactly what mouse-block-permutation PERMANOVA already tests (same tool
used 24 times earlier in this project for burst_index/continuous_burstiness).
Using it here keeps the whole project on one consistent, already-validated
test rather than introducing a second permutation scheme for a similar
question.

Absolute-time bins: coverage drops for later bins (sessions shorter than
60min contribute no data past their own end) -- 79-93% of mice have data in
bins 0-14 (0-45min), dropping to 52-70% by bin 19. Plots show all 20 bins
(nanmean/nanSEM); the PERMANOVA test restricts to bins 0-14 (0-45min) to
avoid dropping too many mice via listwise deletion on always-NaN mice in bin
19-esque columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from permanova import permanova_euclidean

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
N_BINS = 20
ABS_BINS_FOR_TEST = 15  # 0-45min, where coverage is >=79%


def aggregate_to_mice(df: pd.DataFrame, bin_cols: list[str]) -> pd.DataFrame:
    """unit -> session (mean per bin) -> mouse (mean across that mouse's sessions)."""
    per_session = df.groupby(["session_id", "mouse_id", "reward_group"])[bin_cols].mean().reset_index()
    per_mouse = per_session.groupby(["mouse_id", "reward_group"])[bin_cols].mean().reset_index()
    return per_mouse


def plot_cohort_trajectory(per_mouse: pd.DataFrame, bin_cols: list[str], x_vals: np.ndarray,
                            xlabel: str, ax, title: str) -> None:
    for rg, color in [("R+", "#4C72B0"), ("R-", "#DD8452")]:
        sub = per_mouse[per_mouse.reward_group == rg]
        vals = sub[bin_cols].to_numpy(dtype=float)
        mean = np.nanmean(vals, axis=0)
        sem = np.nanstd(vals, axis=0) / np.sqrt(np.sum(~np.isnan(vals), axis=0))
        ax.plot(x_vals, mean, color=color, label=f"{rg} (n={len(sub)} mice)")
        ax.fill_between(x_vals, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean burstiness")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def run_permanova_on_trajectory(per_mouse: pd.DataFrame, bin_cols: list[str], label: str) -> dict:
    complete = per_mouse.dropna(subset=bin_cols)
    if complete["reward_group"].nunique() < 2 or complete.groupby("reward_group").size().min() < 2:
        return {"label": label, "n_mice": len(complete), "skipped": True}
    X = complete[bin_cols].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    f_obs, p_val, n, n_mice = permanova_euclidean(X, complete["reward_group"].to_numpy(), complete["mouse_id"].to_numpy())
    return {"label": label, "n_mice": n_mice, "pseudo_F": f_obs, "p_value": p_val, "skipped": False}


def main() -> None:
    norm_cols = [f"norm_bin_{i}" for i in range(N_BINS)]
    abs_cols = [f"abs_bin_{i}" for i in range(N_BINS)]
    norm_x = (np.arange(N_BINS) + 0.5) / N_BINS
    abs_x = (np.arange(N_BINS) + 0.5) * 3.0  # 3-min bins -> minutes

    test_results = []
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    panel_i = 0
    for day_stage in ["learning", "expert"]:
        df = pd.read_parquet(ARTIFACTS_DIR / f"full_{day_stage}_temporal_metrics.parquet")
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
                plot_cohort_trajectory(per_mouse_norm, norm_cols, norm_x, "session progression (fraction)", ax, f"{label}\n(normalized)")
                panel_i += 1
                ax = axes.flat[panel_i]
                plot_cohort_trajectory(per_mouse_abs, abs_cols, abs_x, "elapsed time (min)", ax, f"{label}\n(absolute)")
                panel_i += 1

                r_norm = run_permanova_on_trajectory(per_mouse_norm, norm_cols, f"{label}_normalized")
                r_abs = run_permanova_on_trajectory(per_mouse_abs, abs_cols[:ABS_BINS_FOR_TEST], f"{label}_absolute_0-45min")
                test_results.append(r_norm)
                test_results.append(r_abs)
                print(f"{label}: norm p={r_norm.get('p_value', 'NA')}, abs(0-45min) p={r_abs.get('p_value', 'NA')}")

    fig.suptitle("Burstiness across session, R+ vs R- (mean +/- SEM across mice)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "temporal_cohort_trajectories_full.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote {ARTIFACTS_DIR / 'temporal_cohort_trajectories_full.png'}")

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(ARTIFACTS_DIR / "temporal_permanova_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
