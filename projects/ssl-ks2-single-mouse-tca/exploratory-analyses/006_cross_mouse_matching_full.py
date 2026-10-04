"""Consensus cross-mouse component matching across all 68 mice (elaborating
the 2-mouse pairwise pilot in 004), visualized across reward-group cohorts.

Loads every ../artifacts/per_mouse/<session_id>.npz from 005, runs iterative
Hungarian-assignment consensus matching (matching_lib.consensus_match) on
z-scored time factors, assigns each consensus "slot" a semantic label from
its shape (peak latency, sustained index), and checks whether matching
consistency (similarity to the consensus prototype) differs between R+ and
R- cohorts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matching_lib import consensus_match

RANK = 4
ARTIFACTS_DIR = Path("../artifacts/per_mouse")

time_factors_by_mouse = {}
reward_group_by_mouse = {}
day_stage_by_mouse = {}
time_bins = None
for f in sorted(ARTIFACTS_DIR.glob("*.npz")):
    d = np.load(f, allow_pickle=True)
    session_id = f.stem
    time_factors_by_mouse[session_id] = d["time_factors"]
    reward_group_by_mouse[session_id] = int(d["reward_group"])
    day_stage_by_mouse[session_id] = str(d["day_stage"])
    time_bins = d["time_bins"]  # same for every mouse (fixed time_window/bin_size)

print(f"loaded {len(time_factors_by_mouse)} mice")

prototypes, assignment, similarity = consensus_match(time_factors_by_mouse, rank=RANK, n_iter=8)

# Semantic label per consensus slot, derived from its own shape (not
# assigned by hand). First check whether the prototype peaks *before*
# stimulus onset (pre_mean > post_mean): that shape -- elevated at baseline,
# dropping after the stimulus -- is qualitatively different from a
# post-stimulus response and needs its own label rather than being
# mislabelled as an "early transient" just because argmax happens to fall
# in the baseline window. For genuine post-stimulus shapes, use peak
# latency (within the post-stimulus window only) and a sustained index
# (mean over the last third of the post-stimulus window / mean over the
# whole post-stimulus window; >1 means activity is still rising/high late
# in the window). NCP factors are non-negative, so no sign handling is
# needed anywhere here.
time_centers = (time_bins[:-1] + time_bins[1:]) / 2
pre_mask = time_centers < 0
post_mask = ~pre_mask
post_centers = time_centers[post_mask]
n_post = len(post_centers)
post_last_third = slice(2 * n_post // 3, n_post)
slot_labels = []
for k in range(RANK):
    proto = prototypes[:, k] - prototypes[:, k].min()  # shift to non-negative for latency/ratio readability
    pre_mean = proto[pre_mask].mean()
    post_mean = proto[post_mask].mean()
    if pre_mean > post_mean:
        shape_word = "pre-stimulus elevated (drops after stimulus)"
        detail = f"baseline/post ratio {pre_mean / (post_mean + 1e-9):.2f}"
    else:
        post_proto = proto[post_mask]
        peak_latency_ms = 1000 * post_centers[np.argmax(post_proto)]
        sustained_index = post_proto[post_last_third].mean() / (post_proto.mean() + 1e-9)
        if sustained_index > 1.2:
            shape_word = "sustained/late ramp"
        elif peak_latency_ms < 40:
            shape_word = "early sensory transient"
        else:
            shape_word = "mid-latency response"
        detail = f"peak {peak_latency_ms:.0f} ms, sustained idx {sustained_index:.2f}"
    slot_labels.append(f"slot {k + 1}: {shape_word} ({detail})")
    print(slot_labels[-1])

# Save the assignment table: for every (mouse, consensus slot), which of
# that mouse's original TCA components (1-indexed, matching 005's
# corr_component numbering) it is, and how well it matched the consensus
# shape -- this is the join key the area/cohort analysis (007) uses to
# label "component 2 of mouse X" as e.g. "slot 3: sustained/late".
rows = []
for sid in time_factors_by_mouse:
    for k in range(RANK):
        rows.append({
            "session_id": sid,
            "consensus_slot": k + 1,
            "slot_label": slot_labels[k],
            "original_component": int(assignment[sid][k]) + 1,  # 1-indexed to match 005's corr_component
            "similarity_to_prototype": float(similarity[sid][k]),
            "reward_group": reward_group_by_mouse[sid],
            "day_stage": day_stage_by_mouse[sid],
        })
assign_df = pd.DataFrame(rows)
assign_df.to_csv("../artifacts/consensus_assignment.csv", index=False)
print(f"saved ../artifacts/consensus_assignment.csv ({len(assign_df)} rows)")

# --- Figure 1: consensus prototypes ---
fig, axes = plt.subplots(1, RANK, figsize=(3.2 * RANK, 3), sharey=True)
for k in range(RANK):
    axes[k].plot(time_centers, prototypes[:, k], color="k", lw=2)
    axes[k].axvline(0, color="0.6", lw=0.8, ls="--")
    axes[k].set_title(slot_labels[k].split(": ")[1], fontsize=9)
    axes[k].set_xlabel("time from stim onset (s)")
axes[0].set_ylabel("consensus time factor\n(z-scored)")
fig.suptitle(f"Consensus component shapes across {len(time_factors_by_mouse)} mice", y=1.05)
fig.tight_layout()
fig.savefig("006_consensus_prototypes.png", dpi=150, bbox_inches="tight")

# --- Figure 2: every mouse's matched component overlaid per slot, colored by cohort ---
fig, axes = plt.subplots(1, RANK, figsize=(3.6 * RANK, 3.4), sharey=True)
cohort_colors = {1: "#2f7d6b", 0: "#a8593a", -1: "0.6"}
cohort_labels = {1: "R+ (reward_group=1)", 0: "R- (reward_group=0)", -1: "unknown"}
for k in range(RANK):
    ax = axes[k]
    seen_labels = set()
    for sid in time_factors_by_mouse:
        z = (time_factors_by_mouse[sid] - time_factors_by_mouse[sid].mean(axis=0)) / time_factors_by_mouse[sid].std(axis=0)
        comp = assignment[sid][k]
        rg = reward_group_by_mouse[sid]
        label = cohort_labels[rg] if rg not in seen_labels else None
        seen_labels.add(rg)
        ax.plot(time_centers, z[:, comp], color=cohort_colors[rg], alpha=0.35, lw=0.8, label=label)
    ax.plot(time_centers, prototypes[:, k], color="k", lw=2.2, label="consensus" if k == 0 else None)
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_title(slot_labels[k].split(": ")[1], fontsize=9)
    ax.set_xlabel("time from stim onset (s)")
axes[0].set_ylabel("time factor (z-scored)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Per-mouse matched components by consensus slot, colored by cohort", y=1.05)
fig.tight_layout()
fig.savefig("006_per_mouse_by_cohort.png", dpi=150, bbox_inches="tight")

# --- Figure 3: matching consistency (similarity to prototype) by slot and cohort ---
fig, ax = plt.subplots(figsize=(6, 4))
positions = []
data = []
colors = []
xticks = []
xticklabels = []
pos = 0
for k in range(RANK):
    for rg in [1, 0]:
        vals = assign_df[(assign_df["consensus_slot"] == k + 1) & (assign_df["reward_group"] == rg)]["similarity_to_prototype"]
        positions.append(pos)
        data.append(vals.to_numpy())
        colors.append(cohort_colors[rg])
        pos += 1
    xticks.append(pos - 1.5)
    xticklabels.append(f"slot {k + 1}")
    pos += 0.8
bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylabel("similarity to consensus prototype (r)")
ax.set_title("Matching consistency per consensus slot, by cohort")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=cohort_colors[1], label="R+"), mpatches.Patch(color=cohort_colors[0], label="R-")], fontsize=8)
fig.tight_layout()
fig.savefig("006_matching_consistency_by_cohort.png", dpi=150, bbox_inches="tight")

print("saved 006_consensus_prototypes.png, 006_per_mouse_by_cohort.png, 006_matching_consistency_by_cohort.png")
