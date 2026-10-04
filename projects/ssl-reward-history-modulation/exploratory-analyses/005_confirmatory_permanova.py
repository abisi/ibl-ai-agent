"""Locked confirmatory analysis, confirmation-set mice only.

PERMANOVA (Euclidean distance) on the standardized 3-index vector
(hit_stay, nonhit_stay, w2a_transition -- a2w_transition dropped, see
exploratory findings in 004: undefined for essentially all of R+) with
factor = cohort (reward_group), complete cases only. Post-hoc: same test
run within each area with enough coverage in both cohorts, BH-FDR corrected.

For Euclidean distances, PERMANOVA's pseudo-F is exactly the MANOVA-style
sum-of-squares ratio (McArdle & Anderson 2001) -- no NxN distance matrix
needed, so this scales fine to thousands of units.
"""
import numpy as np
import pandas as pd

INDEX_COLS = ["hit_stay", "nonhit_stay", "w2a_transition"]
N_PERM = 4999
SEED = 0
MIN_UNITS_PER_COHORT_AREA = 5


def permanova_euclidean(X, groups, mouse_id, n_perm=N_PERM, seed=SEED):
    """Two-group PERMANOVA pseudo-F and permutation p-value, with mouse-block
    permutation: cohort is a mouse-level property (every unit from a mouse
    shares one label), so the null must permute which *mice* get which
    cohort label (preserving the observed mouse-per-cohort counts) and keep
    each mouse's units together -- permuting individual units would treat
    ~700 non-independent units per mouse as if they were independent draws.

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

    mouse_of_unit = pd.Series(mouse_id)
    mouse_group = pd.Series(groups, index=mouse_id).groupby(level=0).first()  # one label per mouse
    n_mice = len(mouse_group)

    f_perm = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = pd.Series(rng.permutation(mouse_group.values), index=mouse_group.index)
        g_perm = mouse_of_unit.map(shuffled).to_numpy()
        f_perm[i] = ((sst - ssw(g_perm)) / (k - 1)) / (ssw(g_perm) / (n - k))
    p = (1 + (f_perm >= f_obs).sum()) / (n_perm + 1)
    return f_obs, p, n, n_mice


mouse_split = pd.read_csv("projects/ssl-reward-history-modulation/artifacts/mouse_split.csv")
index_df = pd.read_parquet("projects/ssl-reward-history-modulation/artifacts/unit_indices.parquet")
index_df = index_df.merge(mouse_split[["mouse_id", "split"]], on="mouse_id")

confirm = index_df[index_df["split"] == "confirmation"].dropna(subset=INDEX_COLS).copy()
print(f"Confirmation set, complete cases on {INDEX_COLS}: {len(confirm)} units, "
      f"{confirm['mouse_id'].nunique()} mice, cohort counts:\n{confirm['reward_group'].value_counts()}")

X_all = confirm[INDEX_COLS].to_numpy()
X_all = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)  # standardize on confirmation set only
groups_all = confirm["reward_group"].to_numpy()

f_obs, p_val, n_main, n_mice_main = permanova_euclidean(X_all, groups_all, confirm["mouse_id"].to_numpy())
print(f"\n=== Main PERMANOVA (cohort, all areas pooled, mouse-block permutation) ===\n"
      f"pseudo-F={f_obs:.3f}, p={p_val:.4f}, n_units={n_main}, n_mice={n_mice_main}")

# --- Post-hoc: per area ---
results = []
for area, group in confirm.groupby("area_acronym_custom"):
    counts = group["reward_group"].value_counts()
    n_mice_area = group.drop_duplicates("mouse_id")["reward_group"].value_counts()
    if counts.get("R+", 0) < MIN_UNITS_PER_COHORT_AREA or counts.get("R-", 0) < MIN_UNITS_PER_COHORT_AREA:
        continue
    if n_mice_area.get("R+", 0) < 2 or n_mice_area.get("R-", 0) < 2:
        continue  # need >=2 mice/cohort for the mouse-block permutation to be meaningful
    X = group[INDEX_COLS].to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    f, p, n, n_mice = permanova_euclidean(X, group["reward_group"].to_numpy(), group["mouse_id"].to_numpy())
    results.append({"area_acronym_custom": area, "n_units": n, "n_mice": n_mice,
                     "n_rplus": counts["R+"], "n_rminus": counts["R-"], "pseudo_F": f, "p_value": p})

posthoc = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
# Benjamini-Hochberg FDR
m = len(posthoc)
posthoc["p_rank"] = np.arange(1, m + 1)
posthoc["bh_threshold"] = posthoc["p_rank"] / m * 0.05
posthoc["significant_fdr05"] = posthoc["p_value"] <= posthoc["bh_threshold"]
# BH is only valid up to the largest rank where p <= threshold holds for all smaller ranks; enforce monotonicity
if posthoc["significant_fdr05"].any():
    last_sig = posthoc["significant_fdr05"].to_numpy().nonzero()[0].max()
    posthoc.loc[:last_sig, "significant_fdr05"] = True
    posthoc.loc[last_sig + 1:, "significant_fdr05"] = False

print(f"\n=== Post-hoc per area ({m} areas tested, BH-FDR 5%) ===")
print(posthoc.to_string(index=False))

posthoc.to_csv("projects/ssl-reward-history-modulation/artifacts/posthoc_area_permanova.csv", index=False)
with open("projects/ssl-reward-history-modulation/artifacts/main_permanova_result.txt", "w") as f:
    f.write(f"pseudo-F={f_obs:.4f}, p={p_val:.4f}, n_units={n_main}, n_mice={n_mice_main}, n_perm={N_PERM}, "
            f"indices={INDEX_COLS}, permutation=mouse-block\n")
