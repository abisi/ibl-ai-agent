"""Which brain areas contain the most neurons with large loadings on
components that track learning, and does that differ between reward-group
cohorts (R+ vs R-)?

"Tracks learning" = the component's trial-mode loading survived the
circular-shift null (p < 0.05) against P(lick) and/or d-prime, in that
mouse (see 003/005). This is deliberately uncorrected across the 68 mice x
4 components x 2 behaviors = 544 tests run in 005 -- there is no held-out
confirmation set to validate a stricter threshold against, per the
no-split decision in question.md, so this whole script is exploratory and
labelled as such throughout.

"Largest loading" = top decile of that mouse-and-component's neuron-mode
loadings (NCP factors are non-negative by construction, so this is just the
largest raw loadings -- there's no separate signed/unsigned distinction to
make here despite the user's "absolute weight" phrasing).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

ARTIFACTS_DIR = Path("../artifacts/per_mouse")
P_THRESHOLD = 0.05
TOP_QUANTILE = 0.90
UNASSIGNED_AREAS = {"void", "root"}

assign_df = pd.read_csv("../artifacts/consensus_assignment.csv")

rows = []
recorded_rows = []  # every recorded neuron, once per mouse (not per learning-component), for the enrichment denominator
for f in sorted(ARTIFACTS_DIR.glob("*.npz")):
    session_id = f.stem
    d = np.load(f, allow_pickle=True)
    neuron_factors = d["neuron_factors"]  # (n_neurons, RANK)
    beryl_area = d["beryl_area"]
    reward_group = int(d["reward_group"])

    for area in beryl_area:
        recorded_rows.append({"session_id": session_id, "reward_group": reward_group, "area": area})

    corr = pd.DataFrame({
        "component": d["corr_component"], "behavior": d["corr_behavior"],
        "r": d["corr_r"], "p": d["corr_p"],
    })
    # A component "tracks learning" in this mouse if it clears the null for
    # either behavioral variable.
    learning_components = corr[corr["p"] < P_THRESHOLD]["component"].unique()

    for comp in learning_components:
        comp = int(comp)
        loadings = neuron_factors[:, comp - 1]
        threshold = np.quantile(loadings, TOP_QUANTILE)
        top_mask = loadings >= threshold
        slot_row = assign_df[(assign_df["session_id"] == session_id) & (assign_df["original_component"] == comp)]
        slot_label = slot_row["slot_label"].iloc[0] if len(slot_row) else "unmatched"
        for area in beryl_area[top_mask]:
            rows.append({
                "session_id": session_id, "reward_group": reward_group,
                "component": comp, "slot_label": slot_label, "area": area,
            })

top_neurons_df = pd.DataFrame(rows)
top_neurons_df.to_csv("../artifacts/top_loading_neurons_learning_components.csv", index=False)
print(f"{top_neurons_df['session_id'].nunique()} mice contributed at least one learning-correlated "
      f"component; {len(top_neurons_df)} top-decile-loading neuron instances total")

area_counts = top_neurons_df[~top_neurons_df["area"].isin(UNASSIGNED_AREAS)]

# Denominator for enrichment: every recorded neuron (regardless of learning
# status), so a raw top-area ranking (Figure 1) can be checked against how
# much of that ranking just reflects which areas were recorded most --
# MOs and CP are common, heavily-sampled targets in this preparation
# regardless of any learning signal.
recorded_df = pd.DataFrame(recorded_rows)
recorded_df = recorded_df[~recorded_df["area"].isin(UNASSIGNED_AREAS)]
n_recorded_by_area = recorded_df["area"].value_counts()
n_recorded_by_area_cohort = recorded_df.groupby(["area", "reward_group"]).size().unstack(fill_value=0)

# --- Figure 1: top areas overall, across all learning-correlated components ---
overall_counts = area_counts["area"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(overall_counts.index[::-1], overall_counts.values[::-1], color="#1f6f78")
ax.set_xlabel("# top-decile-loading neurons\n(pooled across learning-correlated components, all mice)")
ax.set_title(f"Areas most represented among learning-tracking TCA components\n(p<{P_THRESHOLD}, top {int((1 - TOP_QUANTILE) * 100)}% loadings)")
fig.tight_layout()
fig.savefig("007_top_areas_overall.png", dpi=150, bbox_inches="tight")

# --- Figure 1b: same ranking, but enrichment-normalized (fraction of that
# area's OWN recorded neurons that are top-decile-loading on a
# learning-tracking component) -- adversarial check against Figure 1 just
# reflecting recording depth (MOs/CP are heavily-sampled targets in this
# preparation regardless of any learning signal). Restricted to areas with
# >=100 recorded neurons cohort-wide, so rarely-sampled areas don't produce
# noisy, extreme ratios.
well_sampled = n_recorded_by_area[n_recorded_by_area >= 100].index
enrichment = (area_counts["area"].value_counts().reindex(well_sampled).fillna(0) / n_recorded_by_area.reindex(well_sampled))
enrichment = enrichment.sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(enrichment.index[::-1], enrichment.values[::-1], color="#1f6f78")
ax.set_xlabel("fraction of that area's own recorded neurons\nthat are top-decile-loading on a learning-tracking component")
ax.set_title("Same ranking, normalized by recording depth per area\n(areas with >=100 recorded neurons cohort-wide)")
fig.tight_layout()
fig.savefig("007_top_areas_enrichment.png", dpi=150, bbox_inches="tight")
print("enrichment-normalized top areas:")
print(enrichment)

# --- Figure 2: top areas by consensus slot (semantic component identity) ---
slots = sorted(area_counts["slot_label"].unique())
fig, axes = plt.subplots(1, len(slots), figsize=(4.5 * len(slots), 5), sharex=False)
if len(slots) == 1:
    axes = [axes]
for ax, slot in zip(axes, slots):
    counts = area_counts[area_counts["slot_label"] == slot]["area"].value_counts().head(10)
    ax.barh(counts.index[::-1], counts.values[::-1], color="#1f6f78")
    ax.set_title(slot.split(": ")[1] if ": " in slot else slot, fontsize=9)
    ax.set_xlabel("# neurons")
fig.suptitle("Top areas among learning-tracking components, by consensus component identity", y=1.03)
fig.tight_layout()
fig.savefig("007_top_areas_by_slot.png", dpi=150, bbox_inches="tight")

# --- Figure 3: R+ vs R- comparison for the top areas, enrichment-normalized
# (fraction of that area's own recorded neurons *within that cohort* that
# are top-decile-loading & learning-tracking) -- controls for R+ and R-
# mice not necessarily sampling each area to the same depth, not just for
# unequal R+/R- mouse counts. ---
top_area_names = enrichment.index.tolist()
comparison = (
    area_counts[area_counts["area"].isin(top_area_names)]
    .groupby(["area", "reward_group"]).size().unstack(fill_value=0)
    .reindex(top_area_names, fill_value=0)
)
comparison_enrich = comparison / n_recorded_by_area_cohort.reindex(top_area_names)

fig, ax = plt.subplots(figsize=(7, 5.5))
y = np.arange(len(top_area_names))
width = 0.38
ax.barh(y + width / 2, comparison_enrich.get(1, pd.Series(0, index=top_area_names)), width, color="#2f7d6b", label="R+ (reward_group=1)")
ax.barh(y - width / 2, comparison_enrich.get(0, pd.Series(0, index=top_area_names)), width, color="#a8593a", label="R- (reward_group=0)")
ax.set_yticks(y)
ax.set_yticklabels(top_area_names)
ax.invert_yaxis()
ax.set_xlabel("fraction of that area's own recorded neurons, within cohort,\nthat are top-decile-loading & learning-tracking")
ax.set_title("Area enrichment for learning-tracking components, R+ vs R-")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("007_area_by_cohort.png", dpi=150, bbox_inches="tight")

# Omnibus test: does the *raw count* distribution (top areas vs. "other")
# differ between cohorts? Uses counts (not the enrichment ratio, which
# chi-square can't take directly) as a contingency table -- still an
# omnibus test over all top areas at once, per exploration-confirmation
# guidance to prefer tests of a general hypothesis when possible.
chi2, p_chi2, dof, _ = chi2_contingency(comparison)
print(f"omnibus chi-square test, area count distribution x cohort (top {len(top_area_names)} areas): "
      f"chi2={chi2:.1f}, dof={dof}, p={p_chi2:.4f}")

comparison.to_csv("../artifacts/area_by_cohort_counts.csv")
comparison_enrich.to_csv("../artifacts/area_by_cohort_enrichment.csv")
print("saved 007_top_areas_overall.png, 007_top_areas_by_slot.png, 007_area_by_cohort.png")
print("saved ../artifacts/top_loading_neurons_learning_components.csv, ../artifacts/area_by_cohort_counts.csv")
