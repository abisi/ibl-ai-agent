"""For every neuron in every mouse: which component is its "favorite"
(dominant, largest loading), what does that component's time-factor shape
mean (sensory/decision/motor/late x activation/suppression -- see
component_semantics_lib.py), and where does that neuron live
(area_acronym_custom, Axel's own nomenclature, via allen_utils.py)?

Produces global proportions of neuron "type" across the whole dataset, the
same split by reward-group cohort, and the same again broken down by
Axel's custom area groups (allen_utils.get_custom_area_groups_from_name),
globally and per cohort.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, r"M:\analysis\Axel_Bisi\brain_wide_analysis")
import allen_utils as allen  # noqa: E402

from component_semantics_lib import classify_component, STAGE_ORDER, SIGN_ORDER  # noqa: E402

ARTIFACTS_DIR = Path("../artifacts/per_mouse")
RANK = 4

# A consistent color per (stage, sign): activation in a saturated shade,
# suppression in a lighter tint of the same hue, so the two share a family
# but stay visually distinguishable at a glance.
STAGE_BASE_COLOR = {"sensory": "#1f6f78", "decision": "#c98a2c", "motor": "#b8433f", "late": "#5b4b8a"}


def type_color(sign, stage):
    base = STAGE_BASE_COLOR[stage]
    return base if sign == "activation" else base + "80"  # 80 = ~50% alpha hex suffix


TYPE_ORDER = [f"{stage}\n{sign}" for stage in STAGE_ORDER for sign in SIGN_ORDER]
TYPE_COLORS = {f"{stage}\n{sign}": type_color(sign, stage) for stage in STAGE_ORDER for sign in SIGN_ORDER}

rows = []
component_class_rows = []
for f in sorted(ARTIFACTS_DIR.glob("*.npz")):
    session_id = f.stem
    d = np.load(f, allow_pickle=True)
    neuron_factors = d["neuron_factors"]  # (n_neurons, RANK)
    time_factors = d["time_factors"]  # (n_time_bins, RANK)
    time_bins = d["time_bins"]
    area_custom = d["area_custom"]
    reward_group = int(d["reward_group"])

    comp_class = {}
    for comp in range(RANK):
        sign, stage, latency_ms = classify_component(time_factors[:, comp], time_bins)
        comp_class[comp] = (sign, stage)
        component_class_rows.append({
            "session_id": session_id, "component": comp + 1, "sign": sign, "stage": stage, "latency_ms": latency_ms,
        })

    dominant = np.argmax(neuron_factors, axis=1)  # favorite component per neuron, 0-indexed
    for neuron_idx, comp in enumerate(dominant):
        sign, stage = comp_class[comp]
        rows.append({
            "session_id": session_id, "reward_group": reward_group,
            "area_custom": area_custom[neuron_idx], "dominant_component": comp + 1,
            "sign": sign, "stage": stage, "neuron_type": f"{stage}\n{sign}",
        })

neurons_df = pd.DataFrame(rows)
comp_class_df = pd.DataFrame(component_class_rows)
neurons_df.to_csv("../artifacts/neuron_component_types.csv", index=False)
comp_class_df.to_csv("../artifacts/component_classifications.csv", index=False)
print(f"{len(neurons_df)} neurons classified across {neurons_df['session_id'].nunique()} mice")
print(comp_class_df.groupby(["stage", "sign"]).size().reindex(
    pd.MultiIndex.from_product([STAGE_ORDER, SIGN_ORDER], names=["stage", "sign"]), fill_value=0
))


def stacked_bar(ax, counts_by_group, group_order, title):
    """counts_by_group: dict {group_label: Series indexed by TYPE_ORDER}"""
    bottoms = np.zeros(len(group_order))
    for t in TYPE_ORDER:
        vals = np.array([counts_by_group[g].get(t, 0) for g in group_order])
        ax.bar(group_order, vals, bottom=bottoms, color=TYPE_COLORS[t], label=t.replace("\n", " "), edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("proportion of neurons")


# --- Figure A: global proportions ---
global_counts = neurons_df["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0)
fig, ax = plt.subplots(figsize=(5, 5))
stacked_bar(ax, {"all mice": global_counts}, ["all mice"], "Neuron type proportions, whole dataset")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_xlim(-1, 1)
fig.tight_layout()
fig.savefig("010_global_proportions.png", dpi=150, bbox_inches="tight")

# --- Figure B: global proportions, by reward group ---
rg_counts = {
    "R+": neurons_df[neurons_df.reward_group == 1]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
    "R-": neurons_df[neurons_df.reward_group == 0]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0),
}
fig, ax = plt.subplots(figsize=(5, 5))
stacked_bar(ax, rg_counts, ["R+", "R-"], "Neuron type proportions, by cohort")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("010_proportions_by_cohort.png", dpi=150, bbox_inches="tight")

# --- Figure C: proportions per custom area group (pooled cohorts) ---
area_to_group = allen.get_custom_area_groups_from_name()
neurons_df["area_group"] = neurons_df["area_custom"].map(area_to_group)
grouped = neurons_df.dropna(subset=["area_group"])
group_order = [g for g in grouped["area_group"].value_counts().index]  # sorted by n neurons

area_counts = {g: grouped[grouped.area_group == g]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0) for g in group_order}
fig, ax = plt.subplots(figsize=(10, 5.5))
stacked_bar(ax, area_counts, group_order, "Neuron type composition by custom area group (both cohorts pooled)")
ax.set_xticklabels(group_order, rotation=40, ha="right", fontsize=8)
ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
fig.savefig("010_proportions_by_area_group.png", dpi=150, bbox_inches="tight")

# --- Figure D: proportions per custom area group, split R+ vs R- ---
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
for ax, rg, label in zip(axes, [1, 0], ["R+", "R-"]):
    sub = grouped[grouped.reward_group == rg]
    counts = {g: sub[sub.area_group == g]["neuron_type"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0) for g in group_order}
    stacked_bar(ax, counts, group_order, f"{label} (reward_group={rg})")
axes[1].set_xticklabels(group_order, rotation=40, ha="right", fontsize=8)
axes[0].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.suptitle("Neuron type composition by custom area group, R+ vs R-", y=1.0)
fig.tight_layout()
fig.savefig("010_proportions_by_area_group_cohort.png", dpi=150, bbox_inches="tight")

print("saved 010_global_proportions.png, 010_proportions_by_cohort.png, "
      "010_proportions_by_area_group.png, 010_proportions_by_area_group_cohort.png")
print(f"n neurons with no custom area group (excluded from C/D): {neurons_df['area_group'].isna().sum()} / {len(neurons_df)}")

# --- Figure E: the R+/R- "late activation" gap, tested at the correct
# replicate level (mouse, not neuron -- neurons within a mouse aren't
# independent, so the pooled-neuron stacked bars above are good for
# visualization but would badly overstate significance if used for a test;
# per exploration-confirmation SKILL.md, the replicate is the mouse). ---
from scipy.stats import mannwhitneyu

per_mouse_frac = neurons_df.groupby(["session_id", "reward_group"])["neuron_type"].apply(
    lambda s: (s == "late\nactivation").mean()
).reset_index(name="frac_late_activation")
per_mouse_frac.to_csv("../artifacts/per_mouse_late_activation_fraction.csv", index=False)

rplus_frac = per_mouse_frac[per_mouse_frac.reward_group == 1]["frac_late_activation"]
rminus_frac = per_mouse_frac[per_mouse_frac.reward_group == 0]["frac_late_activation"]
stat, p_mw = mannwhitneyu(rplus_frac, rminus_frac, alternative="two-sided")
print(f"\nmouse-level late-activation fraction: R+ median={rplus_frac.median():.3f} mean={rplus_frac.mean():.3f}, "
      f"R- median={rminus_frac.median():.3f} mean={rminus_frac.mean():.3f}, Mann-Whitney p={p_mw:.4f}")

fig, ax = plt.subplots(figsize=(4, 5))
rng = np.random.default_rng(0)
for i, (frac, color, label) in enumerate([(rminus_frac, "#a8593a", "R-"), (rplus_frac, "#2f7d6b", "R+")]):
    x_jitter = i + rng.uniform(-0.08, 0.08, size=len(frac))
    ax.scatter(x_jitter, frac, color=color, alpha=0.6, s=22, zorder=2)
    ax.hlines(frac.median(), i - 0.2, i + 0.2, color="k", lw=2, zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["R-", "R+"])
ax.set_ylabel("fraction of that mouse's neurons\nwhose favorite component is 'late activation'")
ax.set_title(f"Mouse-level replicate test\nMann-Whitney p={p_mw:.4f}", fontsize=10)
fig.tight_layout()
fig.savefig("010_late_activation_by_mouse.png", dpi=150, bbox_inches="tight")
print("saved 010_late_activation_by_mouse.png")
