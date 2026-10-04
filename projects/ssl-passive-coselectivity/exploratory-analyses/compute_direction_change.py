"""Test whether neurons changing selectivity go in different directions in
R+ vs R-, at the area level (see question.md). Two complementary tests:

1. Flip-direction contingency: among units selectivity-significant
   (permutation p<0.05) in BOTH passive_pre and passive_post whose
   preference sign flips, per-area Fisher's exact test of R+ vs R- flip
   direction (whisker->auditory vs auditory->whisker).
2. Continuous shift: delta_selectivity_auc = auc(post) - auc(pre), all
   units, mouse-block permutation test of R+ vs R- difference in
   mouse-mean delta, per area.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ARTIFACTS_DIR = Path("projects/ssl-passive-coselectivity/artifacts")
WINDOWS = ["w5_35"]
N_PERM = 4999
SEED = 0
MIN_UNITS_FLIP = 3
MIN_MICE_PER_COHORT = 2


def mouse_block_permutation_test(mouse_values: pd.Series, mouse_groups: pd.Series, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = mouse_values.to_numpy()
    groups = mouse_groups.to_numpy()
    levels = np.unique(groups)
    if len(levels) != 2:
        return {"observed_diff_mean": np.nan, "p_value": np.nan, "n_mice": len(values)}
    g0, g1 = levels
    observed = values[groups == g1].mean() - values[groups == g0].mean()
    n1 = int((groups == g1).sum())
    n = len(values)
    idx = np.arange(n)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(idx)
        perm_diffs[i] = values[perm[:n1]].mean() - values[perm[n1:]].mean()
    p = float((np.abs(perm_diffs) >= np.abs(observed)).mean())
    return {"observed_diff_mean": float(observed), "p_value": p, "n_mice_g0": int((groups == g0).sum()), "n_mice_g1": int((groups == g1).sum())}


def main() -> None:
    sel = pd.read_parquet(ARTIFACTS_DIR / "selectivity_auc.parquet")
    all_results = {}

    for w in WINDOWS:
        idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "day_stage"]
        wide = sel.pivot_table(index=idx_cols, columns="passive_epoch", values=[f"selectivity_auc_{w}", f"selective_{w}"], aggfunc="first")
        wide.columns = [f"{a}__{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        wide = wide.dropna(subset=[f"selectivity_auc_{w}__passive_pre", f"selectivity_auc_{w}__passive_post"])

        wide["delta_selectivity_auc"] = wide[f"selectivity_auc_{w}__passive_post"] - wide[f"selectivity_auc_{w}__passive_pre"]
        wide["pref_pre"] = np.where(wide[f"selectivity_auc_{w}__passive_pre"] > 0.5, "whisker", "auditory")
        wide["pref_post"] = np.where(wide[f"selectivity_auc_{w}__passive_post"] > 0.5, "whisker", "auditory")

        # --- Test 1: flip-direction contingency ---
        double_sig = wide[(wide[f"selective_{w}__passive_pre"] == True) & (wide[f"selective_{w}__passive_post"] == True)].copy()  # noqa: E712
        flipped = double_sig[double_sig["pref_pre"] != double_sig["pref_post"]].copy()
        flipped["flip_direction"] = flipped["pref_pre"] + "_to_" + flipped["pref_post"]
        print(f"[{w}] double-significant units: {len(double_sig)}, flipped: {len(flipped)}")

        flip_results = []
        for area, group in flipped.groupby(["day_stage", "area_group"], observed=True):
            day_stage, area_name = area
            ct = pd.crosstab(group["reward_group"], group["flip_direction"])
            if ct.shape != (2, 2) or ct.to_numpy().sum() < MIN_UNITS_FLIP:
                continue
            odds, p = fisher_exact(ct.to_numpy())
            flip_results.append({"window": w, "day_stage": day_stage, "area_group": area_name, "n_flipped": int(ct.to_numpy().sum()), "odds_ratio": odds, "p_value": p, "table": ct.to_dict()})
        flip_df = pd.DataFrame(flip_results).sort_values("p_value").reset_index(drop=True)
        if len(flip_df):
            m = len(flip_df)
            flip_df["p_rank"] = np.arange(1, m + 1)
            flip_df["bh_threshold"] = flip_df["p_rank"] / m * 0.05
            flip_df["significant_fdr05"] = flip_df["p_value"] <= flip_df["bh_threshold"]
            if flip_df["significant_fdr05"].any():
                last_sig = flip_df["significant_fdr05"].to_numpy().nonzero()[0].max()
                flip_df.loc[:last_sig, "significant_fdr05"] = True
                flip_df.loc[last_sig + 1:, "significant_fdr05"] = False
        flip_df.to_parquet(ARTIFACTS_DIR / f"flip_direction_posthoc_{w}.parquet", index=False)
        print(f"[{w}] Flip-direction Fisher's exact, per (day_stage, area): {len(flip_df)} testable groups")
        if len(flip_df):
            print(flip_df[["day_stage", "area_group", "n_flipped", "odds_ratio", "p_value"]].sort_values("p_value").to_string(index=False))

        # --- Test 2: continuous shift, mouse-block permutation, per area ---
        shift_results = []
        for (day_stage, area), group in wide.groupby(["day_stage", "area_group"], observed=True):
            per_mouse = group.groupby(["mouse_id", "reward_group"])["delta_selectivity_auc"].mean().reset_index()
            n_mice_cohort = per_mouse["reward_group"].value_counts()
            if n_mice_cohort.get("R+", 0) < MIN_MICE_PER_COHORT or n_mice_cohort.get("R-", 0) < MIN_MICE_PER_COHORT:
                continue
            res = mouse_block_permutation_test(per_mouse["delta_selectivity_auc"], per_mouse["reward_group"], N_PERM, SEED)
            res.update({"window": w, "day_stage": day_stage, "area_group": area, "n_units": len(group)})
            shift_results.append(res)
        shift_df = pd.DataFrame(shift_results).sort_values("p_value") if shift_results else pd.DataFrame()
        if len(shift_df):
            m = len(shift_df)
            shift_df["p_rank"] = np.arange(1, m + 1)
            shift_df["bh_threshold"] = shift_df["p_rank"] / m * 0.05
            shift_df["significant_fdr05"] = shift_df["p_value"] <= shift_df["bh_threshold"]
            if shift_df["significant_fdr05"].any():
                last_sig = shift_df["significant_fdr05"].to_numpy().nonzero()[0].max()
                shift_df.loc[:last_sig, "significant_fdr05"] = True
                shift_df.loc[last_sig + 1:, "significant_fdr05"] = False
        shift_df.to_parquet(ARTIFACTS_DIR / f"delta_selectivity_posthoc_{w}.parquet", index=False)
        print(f"\n[{w}] Continuous-shift mouse-block permutation, per (day_stage, area): {len(shift_df)} testable groups")
        if len(shift_df):
            print(shift_df[["day_stage", "area_group", "n_units", "observed_diff_mean", "p_value", "significant_fdr05"]].to_string(index=False))

        wide.to_parquet(ARTIFACTS_DIR / f"selectivity_change_{w}.parquet", index=False)
        all_results[w] = {"n_double_sig": len(double_sig), "n_flipped": len(flipped)}

    with open(ARTIFACTS_DIR / "direction_change_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
