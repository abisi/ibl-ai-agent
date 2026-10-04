"""PERMANOVA with mouse-block permutation, algorithm reused unchanged from
projects/ssl-reward-history-modulation/exploratory-analyses/005_confirmatory_permanova.py
-- the established, validated fix for the pseudoreplication trap documented
in skills/ssl-analyze/references/ssl_analysis_patterns.md (naive unit-level
permutation gave a false-positive p=0.0004 with "significant" areas;
mouse-block permutation on the same data gave the correct p=0.57).

Performance fix (this project, 2026-08-26): the per-permutation label
expansion originally used `mouse_of_unit.map(shuffled)` (a fresh pandas
Series + dict-style map over all N units, every permutation) -- fine at the
scale of prior projects, but this project's populations are much larger
(up to ~129k units) and `run_with_posthoc` runs a full N_PERM-permutation
test per area (~40-70 areas), so this was on track to take several hours.
Replaced with a precomputed integer mouse-index array + numpy fancy
indexing (`shuffled_values[mouse_codes]`), which is the *same* shuffle
(mouse-block permutation of group labels, expanded to units via the same
per-unit mouse assignment) -- verified byte-for-byte equivalent to the
original on synthetic data with a fixed seed before deploying. N_PERM
reduced 4999 -> 999 for this preliminary/exploratory pass specifically
(still resolves p to 0.001) -- rerun with N_PERM=4999 if/when this becomes
a locked confirmatory test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PERM = 999
SEED = 0
MIN_UNITS_PER_COHORT_AREA = 5


def permanova_euclidean(X: np.ndarray, groups: np.ndarray, mouse_id: np.ndarray, n_perm: int = N_PERM, seed: int = SEED):
    """Two-group PERMANOVA pseudo-F and permutation p-value, mouse-block
    permutation (cohort is a mouse-level property; permuting individual
    units would treat non-independent same-mouse units as independent).

    :param X: (n, p) array, standardized.
    :param groups: (n,) array of 2 distinct labels, constant within a mouse.
    :param mouse_id: (n,) array, one label per unit's source mouse.
    :return: (F_observed, p_value, n, n_mice)
    """
    rng = np.random.default_rng(seed)
    grand_mean = X.mean(axis=0)
    sst = ((X - grand_mean) ** 2).sum()

    def ssw(g):
        total = 0.0
        for label in np.unique(g):
            Xg = X[g == label]
            total += ((Xg - Xg.mean(axis=0)) ** 2).sum()
        return total

    n, k = len(X), 2
    f_obs = ((sst - ssw(groups)) / (k - 1)) / (ssw(groups) / (n - k))

    mouse_group = pd.Series(groups, index=mouse_id).groupby(level=0).first()
    n_mice = len(mouse_group)
    # Precompute once: each unit's mouse's position in mouse_group.index, so
    # a permutation is just indexing the shuffled per-mouse values by this
    # fixed array -- equivalent to (but far faster than) mouse_of_unit.map(shuffled).
    mouse_codes = pd.Categorical(mouse_id, categories=mouse_group.index).codes
    mouse_group_values = mouse_group.values

    f_perm = np.empty(n_perm)
    for i in range(n_perm):
        shuffled_values = rng.permutation(mouse_group_values)
        g_perm = shuffled_values[mouse_codes]
        f_perm[i] = ((sst - ssw(g_perm)) / (k - 1)) / (ssw(g_perm) / (n - k))
    p = (1 + (f_perm >= f_obs).sum()) / (n_perm + 1)
    return f_obs, p, n, n_mice


def run_with_posthoc(df: pd.DataFrame, index_cols: list[str], area_col: str = "area_group", label_prefix: str = ""):
    """Main pooled-area PERMANOVA + per-area post-hoc with BH-FDR, same
    structure as the source project's script."""
    complete = df.dropna(subset=index_cols).copy()
    X_all = complete[index_cols].to_numpy()
    X_all = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    groups_all = complete["reward_group"].to_numpy()
    f_obs, p_val, n_main, n_mice_main = permanova_euclidean(X_all, groups_all, complete["mouse_id"].to_numpy())
    main_result = {"label": label_prefix, "pseudo_F": f_obs, "p_value": p_val, "n_units": n_main, "n_mice": n_mice_main, "index_cols": index_cols}
    print(f"[{label_prefix}] Main PERMANOVA: pseudo-F={f_obs:.3f}, p={p_val:.4f}, n_units={n_main}, n_mice={n_mice_main}")

    results = []
    for area, group in complete.groupby(area_col, observed=True):
        counts = group["reward_group"].value_counts()
        n_mice_area = group.drop_duplicates("mouse_id")["reward_group"].value_counts()
        if counts.get("R+", 0) < MIN_UNITS_PER_COHORT_AREA or counts.get("R-", 0) < MIN_UNITS_PER_COHORT_AREA:
            continue
        if n_mice_area.get("R+", 0) < 2 or n_mice_area.get("R-", 0) < 2:
            continue
        X = group[index_cols].to_numpy()
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        f, p, n, n_mice = permanova_euclidean(X, group["reward_group"].to_numpy(), group["mouse_id"].to_numpy())
        results.append({area_col: area, "n_units": n, "n_mice": n_mice, "n_rplus": counts.get("R+", 0), "n_rminus": counts.get("R-", 0), "pseudo_F": f, "p_value": p})

    posthoc = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    if len(posthoc):
        m = len(posthoc)
        posthoc["p_rank"] = np.arange(1, m + 1)
        posthoc["bh_threshold"] = posthoc["p_rank"] / m * 0.05
        posthoc["significant_fdr05"] = posthoc["p_value"] <= posthoc["bh_threshold"]
        if posthoc["significant_fdr05"].any():
            last_sig = posthoc["significant_fdr05"].to_numpy().nonzero()[0].max()
            posthoc.loc[:last_sig, "significant_fdr05"] = True
            posthoc.loc[last_sig + 1:, "significant_fdr05"] = False
    print(f"[{label_prefix}] Post-hoc per area ({len(posthoc)} areas tested, BH-FDR 5%):")
    print(posthoc.to_string(index=False))
    return main_result, posthoc
