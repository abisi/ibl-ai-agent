"""Generate the figures used in report/README.md."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from ibl_ai_agent.datasets.ssl_ks2_ephys import load_spike_shard

FIG_DIR = Path("projects/ssl-reward-history-modulation/report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

trials = pd.read_parquet("projects/ssl-reward-history-modulation/artifacts/trials_with_history.parquet")
index_df = pd.read_parquet("projects/ssl-reward-history-modulation/artifacts/unit_indices.parquet")

# =====================================================================
# Figure 1: metric-definition figure for hit_stay, one example unit
# =====================================================================
EXAMPLE_SESSION, EXAMPLE_CLUSTER = "AB159_20250409_135813", 4000256  # 11.7 Hz baseline, n_a=87, n_b=28
shard = load_spike_shard(Path("reports/datasets/ssl_ks2_ephys/1.0.0/spikes") / EXAMPLE_SESSION)
dense_idx = int(np.where(shard["cluster_ids"] == EXAMPLE_CLUSTER)[0][0])
unit_times = np.sort(shard["spike_times_seconds"][shard["spike_clusters"] == dense_idx])

session_trials = trials[trials["session_id"] == EXAMPLE_SESSION]
prev_hit_true = (session_trials["prev_hit"] == True).fillna(False)   # noqa: E712
prev_hit_false = (session_trials["prev_hit"] == False).fillna(False)  # noqa: E712
base = (session_trials["modality"] == "whisker") & (session_trials["hit"] == True)
group_a = session_trials.loc[base & prev_hit_true, "stim_onset"].to_numpy()   # t-1 was hit ("win-stay")
group_b = session_trials.loc[base & prev_hit_false, "stim_onset"].to_numpy()  # t-1 was non-hit

BIN_S, PRE_S, POST_S = 0.005, 0.10, 0.15
bin_edges = np.arange(-PRE_S, POST_S + BIN_S, BIN_S)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


def psth(stim_onsets):
    counts = np.zeros(len(bin_centers))
    for t in stim_onsets:
        rel = unit_times[(unit_times >= t - PRE_S) & (unit_times < t + POST_S)] - t
        counts += np.histogram(rel, bins=bin_edges)[0]
    return counts / len(stim_onsets) / BIN_S  # Hz


psth_a, psth_b = psth(group_a), psth(group_b)
row = index_df[(index_df["session_id"] == EXAMPLE_SESSION) & (index_df["cluster_id"] == EXAMPLE_CLUSTER)].iloc[0]

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(0.005, 0.050, color="gray", alpha=0.2, label="response window (5-50ms)")
ax.step(bin_centers, psth_a, where="mid", color="tab:blue", label=f"t-1 hit, group A (n={len(group_a)})")
ax.step(bin_centers, psth_b, where="mid", color="tab:red", label=f"t-1 non-hit, group B (n={len(group_b)})")
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("Time from whisker-hit stimulus onset (s)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title(f"hit_stay metric definition -- example unit\n"
             f"{EXAMPLE_SESSION}, cluster {EXAMPLE_CLUSTER} -- index = {row['hit_stay']:.2f}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_metric_definition_hitstay.png", dpi=150)
plt.close(fig)

# =====================================================================
# Figure 2: index distributions by cohort, full population, 3 locked indices
# =====================================================================
CATS = ["hit_stay", "nonhit_stay", "w2a_transition"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, cat in zip(axes, CATS):
    for group, color in [("R+", "tab:blue"), ("R-", "tab:orange")]:
        vals = index_df.loc[index_df["reward_group"] == group, cat].dropna()
        ax.hist(vals, bins=30, alpha=0.5, density=True, color=color, label=f"{group} (n={len(vals)})")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_title(cat)
    ax.set_xlabel("index value")
    ax.legend(fontsize=8)
fig.suptitle("Index distributions by cohort, full population (all 74 mice, per-unit)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_index_distributions_fullpop.png", dpi=150)
plt.close(fig)

# =====================================================================
# Figure 3: post-hoc area-level significance, all three test variants
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
titles = ["Locked confirmatory\n(3-index, 11 mice)", "Full population\n(3-index, 14 mice)",
          "Supplementary\n(2-index, 41 mice)"]
files = ["posthoc_area_permanova.csv", "posthoc_area_permanova_fullpop.csv", "posthoc_area_permanova_fullpop_2index.csv"]
for ax, title, fname in zip(axes, titles, files):
    df = pd.read_csv(f"projects/ssl-reward-history-modulation/artifacts/{fname}").sort_values("p_value")
    colors = ["tab:red" if sig else "tab:gray" for sig in df["significant_fdr05"]]
    ax.barh(df["area_acronym_custom"], -np.log10(df["p_value"]), color=colors)
    ax.axvline(-np.log10(0.05), color="k", linestyle="--", linewidth=0.8, label="uncorrected p=0.05")
    ax.set_xlabel("-log10(p)")
    ax.set_title(title, fontsize=9)
    ax.invert_yaxis()
axes[0].legend(fontsize=7)
fig.suptitle("Post-hoc per-area PERMANOVA (mouse-block permutation); red = BH-FDR 5% significant (none)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_posthoc_areas_all_variants.png", dpi=150)
plt.close(fig)

print("Saved figures to", FIG_DIR)
