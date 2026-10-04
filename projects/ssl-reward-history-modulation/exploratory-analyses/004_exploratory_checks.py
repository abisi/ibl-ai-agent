"""Sanity-check the 4 indices on the exploration-set mice only, before
locking the confirmatory PERMANOVA. Checks: trial-count coverage per bucket,
index distributions by cohort, and area coverage.
"""
import pandas as pd
import matplotlib.pyplot as plt

CATEGORIES = ["hit_stay", "nonhit_stay", "w2a_transition", "a2w_transition"]

mouse_split = pd.read_csv("projects/ssl-reward-history-modulation/artifacts/mouse_split.csv")
index_df = pd.read_parquet("projects/ssl-reward-history-modulation/artifacts/unit_indices.parquet")
index_df = index_df.merge(mouse_split[["mouse_id", "split"]], on="mouse_id")

explore = index_df[index_df["split"] == "exploration"]
print(f"Exploration set: {len(explore)} units, {explore['mouse_id'].nunique()} mice")

# --- Trial-count coverage: min(n_a, n_b) per category, all units (not just exploration) ---
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, cat in zip(axes, CATEGORIES):
    min_n = index_df[[f"{cat}_n_a", f"{cat}_n_b"]].min(axis=1)
    ax.hist(min_n.clip(upper=50), bins=30)
    ax.axvline(5, color="r", linestyle="--", label="floor (5)")
    frac_valid = (min_n >= 5).mean()
    ax.set_title(f"{cat}\n{frac_valid:.0%} units pass floor")
    ax.set_xlabel("min(n_A, n_B) per unit (clipped at 50)")
fig.tight_layout()
fig.savefig("projects/ssl-reward-history-modulation/exploratory-analyses/004_trial_coverage.png", dpi=150)

# --- Index distributions by cohort (exploration set) ---
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, cat in zip(axes, CATEGORIES):
    for group, color in [("R+", "tab:blue"), ("R-", "tab:orange")]:
        vals = explore.loc[explore["reward_group"] == group, cat].dropna()
        ax.hist(vals, bins=25, alpha=0.5, label=f"{group} (n={len(vals)})", color=color, density=True)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_title(cat)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("projects/ssl-reward-history-modulation/exploratory-analyses/004_index_distributions.png", dpi=150)

print("\nPer-category summary (exploration set):")
for cat in CATEGORIES:
    g = explore.groupby("reward_group")[cat].agg(["count", "mean", "std"])
    print(f"\n{cat}:\n{g}")

print("\nAreas represented in exploration set:", sorted(explore["area_acronym_custom"].unique()))
print("n areas x cohort with >=5 units (exploration set):")
print(explore.dropna(subset=["hit_stay"]).groupby(["area_acronym_custom", "reward_group"]).size().unstack(fill_value=0))
