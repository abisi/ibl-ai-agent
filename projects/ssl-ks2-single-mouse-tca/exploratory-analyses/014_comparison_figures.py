"""Direct side-by-side comparison: single-mouse approach (68 separate
per-mouse TCA fits, each mouse's neurons classified by its own decomposition,
then pooled for these proportions) vs. megamouse approach (one shared TCA
fit on neurons pooled from 56 mice, classified by that one decomposition).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from component_semantics_lib import STAGE_ORDER, SIGN_ORDER

STAGE_BASE_COLOR = {"sensory": "#1f6f78", "decision": "#c98a2c", "motor": "#b8433f", "late": "#5b4b8a"}
TYPE_ORDER = [f"{stage}\n{sign}" for stage in STAGE_ORDER for sign in SIGN_ORDER]
TYPE_COLORS = {f"{stage}\n{sign}": (STAGE_BASE_COLOR[stage] if sign == "activation" else STAGE_BASE_COLOR[stage] + "80") for stage in STAGE_ORDER for sign in SIGN_ORDER}

single = pd.read_csv("../artifacts/neuron_component_types.csv")
mega = pd.read_csv("../artifacts/megamouse/megamouse_neuron_types.csv")


def stacked_bar(ax, counts_by_group, group_order, title):
    bottoms = np.zeros(len(group_order))
    for t in TYPE_ORDER:
        vals = np.array([counts_by_group[g].get(t, 0) for g in group_order])
        ax.bar(group_order, vals, bottom=bottoms, color=TYPE_COLORS[t], label=t.replace("\n", " "), edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax.set_title(title, fontsize=10)


# --- Global proportions: single-mouse-pooled vs megamouse ---
counts = {
    "single-mouse\n(68 fits, pooled)": single["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
    "megamouse\n(1 shared fit)": mega["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
}
fig, ax = plt.subplots(figsize=(5.5, 5.5))
stacked_bar(ax, counts, list(counts.keys()), "Neuron type proportions:\nsingle-mouse approach vs. megamouse")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("014_compare_global.png", dpi=150, bbox_inches="tight")

# --- By cohort, both approaches side by side ---
fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)
for ax, df, title in zip(axes, [single, mega], ["single-mouse approach", "megamouse"]):
    counts = {
        "R+": df[df.reward_group == 1]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
        "R-": df[df.reward_group == 0]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
    }
    stacked_bar(ax, counts, ["R+", "R-"], title)
axes[0].set_ylabel("proportion of neurons")
axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle("Cohort split: single-mouse approach vs. megamouse", y=1.02)
fig.tight_layout()
fig.savefig("014_compare_cohort.png", dpi=150, bbox_inches="tight")

print("saved 014_compare_global.png, 014_compare_cohort.png")
print("\nsingle-mouse global:")
print(single["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0))
print("\nmegamouse global:")
print(mega["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0))
