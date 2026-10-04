"""Same analyses as the single-mouse pipeline (003 correlation test, 010
component semantics), run on the ONE shared megamouse decomposition from
012 instead of on 68 separate per-mouse decompositions -- so the two
approaches' outputs can be compared directly.

Differences from the single-mouse version, by construction rather than
choice: there is only one set of 4 components total (not 4 per mouse x 68
mice), so "proportions of neurons per type" here comes from one component
each having many neurons, not many mice each contributing to several
categories; and the trial-factor/behavior correlation test has one result
per component (not one per mouse x component).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, r"M:\analysis\Axel_Bisi\brain_wide_analysis")
import allen_utils as allen  # noqa: E402

from component_semantics_lib import classify_component, STAGE_ORDER, SIGN_ORDER  # noqa: E402
from stats_lib import circular_shift_test  # noqa: E402

RANK = 4
OUT_DIR = Path("../artifacts/megamouse")
STAGE_BASE_COLOR = {"sensory": "#1f6f78", "decision": "#c98a2c", "motor": "#b8433f", "late": "#5b4b8a"}
TYPE_ORDER = [f"{stage}\n{sign}" for stage in STAGE_ORDER for sign in SIGN_ORDER]
TYPE_COLORS = {f"{stage}\n{sign}": (STAGE_BASE_COLOR[stage] if sign == "activation" else STAGE_BASE_COLOR[stage] + "80") for stage in STAGE_ORDER for sign in SIGN_ORDER}

d = np.load(OUT_DIR / "megamouse_factors.npz", allow_pickle=True)
trial_factors, neuron_factors, time_factors = d["trial_factors"], d["neuron_factors"], d["time_factors"]
time_bins = d["time_bins"]
area_custom, reward_group = d["area_custom"], d["reward_group"]
n_mice = int(d["n_mice"])
print(f"megamouse: {neuron_factors.shape[0]} neurons pooled from {n_mice} mice, rank {RANK}")

# --- Tensor/behavior sanity check ---
time_centers = (time_bins[:-1] + time_bins[1:]) / 2
trial_idx = np.arange(-30, 51)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].plot(trial_idx, d["pooled_plick"], color="k", lw=1.6, label="P(lick), mean across mice")
ax2 = axes[0].twinx()
ax2.plot(trial_idx, d["pooled_dprime"], color="#1f6f78", lw=1.2, ls=":", label="d-prime")
axes[0].axvline(0, color="crimson", lw=1.5)
axes[0].set_xlabel("whisker trial id (0 = each mouse's own first hit)")
axes[0].set_ylabel("P(lick)")
ax2.set_ylabel("d-prime", color="#1f6f78")
axes[0].set_title(f"Megamouse behavior ({n_mice} mice pooled)")
axes[1].plot(time_centers, time_factors, lw=1.8)
axes[1].axvline(0, color="0.5", ls="--", lw=0.8)
axes[1].set_xlabel("time from stim onset (s)")
axes[1].set_ylabel("time factor")
axes[1].legend([f"comp {i+1}" for i in range(RANK)], fontsize=7)
axes[1].set_title("Megamouse time factors")
fig.tight_layout()
fig.savefig("013_megamouse_sanity.png", dpi=150, bbox_inches="tight")

# --- Trial-factor x behavior correlation (one result per component, not per mouse) ---
rng = np.random.default_rng(0)
corr_rows = []
fig, axes = plt.subplots(RANK, 2, figsize=(8, 2.2 * RANK), sharex="col")
for comp in range(RANK):
    for j, (name, trace) in enumerate([("P(lick)", d["pooled_plick"]), ("d-prime", d["pooled_dprime"])]):
        r, null_r, p = circular_shift_test(trial_factors[:, comp], trace, n_shifts=2000, rng=rng)
        corr_rows.append({"component": comp + 1, "behavior": name, "r": r, "p": p})
        ax = axes[comp, j]
        ax.hist(null_r, bins=40, color="0.7")
        ax.axvline(r, color="crimson", lw=2)
        ax.set_title(f"comp {comp + 1} x {name}: r={r:+.2f}, p={p:.3f}", fontsize=8)
fig.tight_layout()
fig.savefig("013_megamouse_correlation.png", dpi=150, bbox_inches="tight")
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv("../artifacts/megamouse/megamouse_correlations.csv", index=False)
print(corr_df.to_string(index=False))

# --- Component semantic classification (one label per component, not per mouse-component) ---
comp_labels = {}
for comp in range(RANK):
    sign, stage, latency_ms = classify_component(time_factors[:, comp], time_bins)
    comp_labels[comp] = (sign, stage)
    print(f"component {comp + 1}: {stage} {sign} (latency {latency_ms:.0f} ms)")

dominant = np.argmax(neuron_factors, axis=1)
neuron_type = np.array([f"{comp_labels[c][1]}\n{comp_labels[c][0]}" for c in dominant])
neurons_df = pd.DataFrame({"area_custom": area_custom, "reward_group": reward_group, "neuron_type": neuron_type})
neurons_df.to_csv("../artifacts/megamouse/megamouse_neuron_types.csv", index=False)


def stacked_bar(ax, counts_by_group, group_order, title):
    bottoms = np.zeros(len(group_order))
    for t in TYPE_ORDER:
        vals = np.array([counts_by_group[g].get(t, 0) for g in group_order])
        ax.bar(group_order, vals, bottom=bottoms, color=TYPE_COLORS[t], label=t.replace("\n", " "), edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("proportion of neurons")


# --- Global proportions (megamouse) ---
global_counts = neurons_df["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0)
fig, ax = plt.subplots(figsize=(5, 5))
stacked_bar(ax, {"megamouse": global_counts}, ["megamouse"], f"Neuron type proportions, megamouse ({n_mice} mice pooled)")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_xlim(-1, 1)
fig.tight_layout()
fig.savefig("013_megamouse_global_proportions.png", dpi=150, bbox_inches="tight")

# --- By cohort (splitting the SAME shared decomposition by each neuron's own reward_group) ---
rg_counts = {
    "R+": neurons_df[neurons_df.reward_group == 1]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
    "R-": neurons_df[neurons_df.reward_group == 0]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
}
fig, ax = plt.subplots(figsize=(5, 5))
stacked_bar(ax, rg_counts, ["R+", "R-"], "Megamouse neuron type proportions, by cohort")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("013_megamouse_proportions_by_cohort.png", dpi=150, bbox_inches="tight")

# --- By custom area group ---
area_to_group = allen.get_custom_area_groups_from_name()
neurons_df["area_group"] = neurons_df["area_custom"].map(area_to_group)
grouped = neurons_df.dropna(subset=["area_group"])
group_order = list(grouped["area_group"].value_counts().index)
area_counts = {g: grouped[grouped.area_group == g]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0) for g in group_order}
fig, ax = plt.subplots(figsize=(10, 5.5))
stacked_bar(ax, area_counts, group_order, f"Megamouse neuron type composition by custom area group ({n_mice} mice pooled)")
ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order, rotation=40, ha="right", fontsize=8)
ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
fig.savefig("013_megamouse_proportions_by_area_group.png", dpi=150, bbox_inches="tight")

print("saved 013_megamouse_sanity.png, 013_megamouse_correlation.png, 013_megamouse_global_proportions.png, "
      "013_megamouse_proportions_by_cohort.png, 013_megamouse_proportions_by_area_group.png")
