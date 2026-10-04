"""Small-scale sanity check: OFC neuron activity as a function of reward index.

Question: what is the mean activity of orbitofrontal cortex (OFC) neurons as a
function of reward obtained? User clarified this means tracking activity across
the session as rewards accumulate (1st reward, 2nd reward, ...), not a
rewarded-vs-unrewarded contrast.

OFC = Beryl regions ORBl, ORBm, ORBvl. Rewarded trials (rewardVolume > 0) are
taken in chronological order within one session. Two group resolutions are used:
- QUANTILE_GROUPS (coarse, e.g. quintiles) for the visual PSTH overlay, since
  10ms bins need enough trials per group to average out spike noise and stay
  human-legible;
- GROUP_SIZE (fine, e.g. 10 trials) for a scalar evoked-FR-vs-reward-index trend,
  where noise across points is expected and informative about trial-to-trial
  variability.
Evoked FR = mean rate in RESPONSE_WINDOW minus mean rate in BASELINE_WINDOW
(both relative to feedback_times). See ../question.md for full explication.

This is a small-scale example-neuron test (one insertion, a few units spanning
its two present OFC subregions) to sanity-check the metric before any
population-level analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from brainbox.singlecell import calculate_peths

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard

EPHYS_DIR = resolve_dataset_dir("bwm_ephys")
PID = "5a34d971-1cb3-4f0e-8dfe-e51e2313a668"  # insertion with the most OFC units (226)
BIN_SIZE = 0.01  # seconds, per user request
PRE_TIME, POST_TIME = 0.2, 0.8  # seconds around feedback_times
GROUP_SIZE = 10  # rewarded trials per group, per user's suggested 5-10 range (scalar trend panel)
QUANTILE_GROUPS = 5  # coarser grouping for the visual PSTH overlay (quintiles of the session)
BASELINE_WINDOW = (-0.2, 0.0)  # seconds relative to feedback, for evoked-FR baseline
RESPONSE_WINDOW = (0.0, 0.3)  # seconds relative to feedback, for evoked-FR response

units = pd.read_parquet(EPHYS_DIR / "metadata" / "units.parquet")
trials = pd.read_parquet(EPHYS_DIR / "metadata" / "trials.parquet")

units_pid = units.loc[units["pid"] == PID].copy()
eid = units_pid["eid"].iloc[0]

trials_eid = trials.loc[(trials["eid"] == eid) & trials["bwm_include"]].sort_values("intervals_0")
rewarded = trials_eid.loc[trials_eid["rewardVolume"] > 0, "feedback_times"].dropna().to_numpy()

n_fine_groups = len(rewarded) // GROUP_SIZE
fine_groups = [rewarded[i * GROUP_SIZE:(i + 1) * GROUP_SIZE] for i in range(n_fine_groups)]

quantile_edges = np.linspace(0, len(rewarded), QUANTILE_GROUPS + 1).round().astype(int)
quantile_groups = [rewarded[quantile_edges[i]:quantile_edges[i + 1]] for i in range(QUANTILE_GROUPS)]

print(f"eid={eid}, n_rewarded={len(rewarded)}, n_fine_groups of {GROUP_SIZE}={n_fine_groups}, "
      f"n_quantile_groups={QUANTILE_GROUPS} (~{len(rewarded) // QUANTILE_GROUPS} trials each)")

shard = load_spike_shard(EPHYS_DIR / "spikes" / PID)
spike_times = np.asarray(shard["spike_times_seconds"], dtype=float)
spike_clusters_dense = np.asarray(shard["spike_clusters"], dtype=int)
cluster_ids = np.asarray(shard["cluster_ids"], dtype=int)
spike_cluster_ids = cluster_ids[spike_clusters_dense]

# Pick example units spanning both OFC subregions present on this insertion, at
# low/high firing rate within each, for a diverse but small example set.
ofc_pid = units_pid[units_pid["beryl_acronym"].isin(["ORBl", "ORBm", "ORBvl"])].copy()
example_rows = []
for region in sorted(ofc_pid["beryl_acronym"].unique()):
    region_units = ofc_pid.loc[ofc_pid["beryl_acronym"] == region].sort_values("firing_rate")
    example_rows.append(region_units.iloc[len(region_units) // 4])
    example_rows.append(region_units.iloc[3 * len(region_units) // 4])
example_units = pd.DataFrame(example_rows)
n_examples = len(example_units)

cmap = mpl.colormaps["viridis"].resampled(QUANTILE_GROUPS)

fig, axes = plt.subplots(n_examples, 2, figsize=(11, 2.4 * n_examples))
for row, (_, u) in enumerate(example_units.iterrows()):
    cluster_id = int(u["cluster_id"])
    ax_psth, ax_scalar = axes[row, 0], axes[row, 1]

    for g, align_times in enumerate(quantile_groups):
        peths, _ = calculate_peths(
            spike_times, spike_cluster_ids, [cluster_id], align_times,
            pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE, smoothing=0,
        )
        t = peths["tscale"]
        mean_fr = peths["means"][0]
        ax_psth.plot(t, mean_fr, color=cmap(g), lw=1.3, label=f"q{g + 1}")

    evoked_fr = np.full(n_fine_groups, np.nan)
    group_mean_trial_idx = np.arange(n_fine_groups) * GROUP_SIZE + GROUP_SIZE / 2
    for g, align_times in enumerate(fine_groups):
        peths, _ = calculate_peths(
            spike_times, spike_cluster_ids, [cluster_id], align_times,
            pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE, smoothing=0,
        )
        t = peths["tscale"]
        mean_fr = peths["means"][0]
        baseline = mean_fr[(t >= BASELINE_WINDOW[0]) & (t < BASELINE_WINDOW[1])].mean()
        response = mean_fr[(t >= RESPONSE_WINDOW[0]) & (t < RESPONSE_WINDOW[1])].mean()
        evoked_fr[g] = response - baseline

    ax_psth.axvline(0, color="k", lw=0.8, ls="--")
    ax_psth.set_ylabel("firing rate (sp/s)")
    ax_psth.set_title(
        f"{u['beryl_acronym']} cluster {cluster_id} (baseline FR={u['firing_rate']:.1f} sp/s)"
        f" -- PSTH per quintile (dark=early, light=late)",
        fontsize=8,
    )
    if row == 0:
        ax_psth.legend(loc="upper right", fontsize=6, ncol=QUANTILE_GROUPS)

    ax_scalar.plot(group_mean_trial_idx, evoked_fr, "o-", color="tab:purple", ms=4)
    ax_scalar.axhline(0, color="k", lw=0.5, ls=":")
    ax_scalar.set_ylabel(f"evoked FR (sp/s)\n[{RESPONSE_WINDOW[0]},{RESPONSE_WINDOW[1]}]s - baseline")
    ax_scalar.set_title(f"evoked response per {GROUP_SIZE}-reward group vs. reward index", fontsize=8)

axes[-1, 0].set_xlabel("time from feedback (s)")
axes[-1, 1].set_xlabel("rewarded-trial index within session")
fig.suptitle(
    f"Example OFC units, PID {PID[:8]} -- activity vs. reward index within session\n"
    f"10ms bins, groups of {GROUP_SIZE} consecutive rewarded trials",
    y=1.0,
)
fig.tight_layout()
fig.savefig("000_example_ofc_reward_psths.png", dpi=150, bbox_inches="tight")
print("saved 000_example_ofc_reward_psths.png")
