"""Population-level check: how general is the reward-index activity trend across OFC?

Question: mean activity of OFC neurons as a function of reward obtained (reward
index / session progression). The single-unit example (000_*) showed one OFC
neuron with a declining evoked-response trend across the session. This script
checks how common that pattern is across all OFC units in the EXPLORATION SET
only (see ../question.md for the approved subject-level split; the confirmation
set stays untouched).

For every OFC unit (Beryl ORBl/ORBm/ORBvl) from an exploration-set subject:
- take rewarded trials (rewardVolume > 0, bwm_include) in chronological order;
- compute per-trial evoked firing rate = response-window rate minus
  baseline-window rate (both relative to feedback_times), vectorized via
  np.searchsorted (no visual PSTH needed here, so no bin-size choice);
- correlate (Spearman) evoked FR against trial order index -> one rho per unit,
  our per-unit "reward-index trend" summary statistic.

This is exploratory: it estimates effect size and consistency to inform a later
locked confirmatory test, and is not itself a confirmatory claim.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard

EPHYS_DIR = resolve_dataset_dir("bwm_ephys")
BASELINE_WINDOW = (-0.2, 0.0)  # seconds relative to feedback_times
RESPONSE_WINDOW = (0.0, 0.3)  # seconds relative to feedback_times
N_DECILE_BINS = 10  # for the grand-average trend curve, as fraction of session

EXPLORATION_SUBJECTS = [
    "ZM_1897", "ZM_2240", "SWC_038", "ZFM-01936",
    "SWC_061", "DY_014", "KS094", "KS052",
]

units = pd.read_parquet(EPHYS_DIR / "metadata" / "units.parquet")
trials = pd.read_parquet(EPHYS_DIR / "metadata" / "trials.parquet")

ofc = units[units["beryl_acronym"].isin(["ORBl", "ORBm", "ORBvl"])]
ofc_exp = ofc[ofc["subject"].isin(EXPLORATION_SUBJECTS)].copy()
print(f"exploration-set OFC units: {len(ofc_exp)} across {ofc_exp['pid'].nunique()} insertions, "
      f"{ofc_exp['subject'].nunique()} subjects")


def evoked_fr_per_trial(spike_times, align_times):
    """Per-trial evoked firing rate (response minus baseline), vectorized.

    spike_times: sorted array of spike times (s) for one unit.
    align_times: array (n_trials,) of feedback times, chronologically sorted.
    Returns: array (n_trials,) of evoked FR (sp/s).
    """
    base_lo = np.searchsorted(spike_times, align_times + BASELINE_WINDOW[0])
    base_hi = np.searchsorted(spike_times, align_times + BASELINE_WINDOW[1])
    resp_lo = np.searchsorted(spike_times, align_times + RESPONSE_WINDOW[0])
    resp_hi = np.searchsorted(spike_times, align_times + RESPONSE_WINDOW[1])
    baseline_fr = (base_hi - base_lo) / (BASELINE_WINDOW[1] - BASELINE_WINDOW[0])
    response_fr = (resp_hi - resp_lo) / (RESPONSE_WINDOW[1] - RESPONSE_WINDOW[0])
    return response_fr - baseline_fr


records = []
decile_curves = []  # one row per unit: evoked FR z-scored, per decile of session

for pid, units_pid in ofc_exp.groupby("pid"):
    eid = units_pid["eid"].iloc[0]
    trials_eid = trials.loc[(trials["eid"] == eid) & trials["bwm_include"]].sort_values("intervals_0")
    rewarded = trials_eid.loc[trials_eid["rewardVolume"] > 0, "feedback_times"].dropna().to_numpy()
    trial_index = np.arange(len(rewarded))
    decile = np.minimum((trial_index / len(rewarded) * N_DECILE_BINS).astype(int), N_DECILE_BINS - 1)

    shard = load_spike_shard(EPHYS_DIR / "spikes" / pid)
    spike_times_all = np.asarray(shard["spike_times_seconds"], dtype=float)
    spike_clusters_dense = np.asarray(shard["spike_clusters"], dtype=int)
    cluster_ids = np.asarray(shard["cluster_ids"], dtype=int)
    spike_cluster_ids = cluster_ids[spike_clusters_dense]

    for _, u in units_pid.iterrows():
        mask = spike_cluster_ids == u["cluster_id"]
        unit_spike_times = spike_times_all[mask]
        evoked = evoked_fr_per_trial(unit_spike_times, rewarded)

        rho, pval = spearmanr(trial_index, evoked)
        records.append({
            "pid": pid, "eid": eid, "subject": u["subject"], "cluster_id": u["cluster_id"],
            "region": u["beryl_acronym"], "firing_rate": u["firing_rate"],
            "n_rewarded_trials": len(rewarded), "rho": rho, "pval": pval,
        })

        # z-score this unit's evoked FR (across its own trials) before averaging across
        # units with very different baseline firing rates.
        z = (evoked - evoked.mean()) / (evoked.std() + 1e-9)
        decile_means = np.array([z[decile == d].mean() if np.any(decile == d) else np.nan
                                  for d in range(N_DECILE_BINS)])
        decile_curves.append(decile_means)

results = pd.DataFrame(records)
decile_curves = np.array(decile_curves)
results.to_csv("../artifacts/001_ofc_reward_trend_per_unit.csv", index=False)

defined_mask = results["rho"].notna().to_numpy()
n_undefined = (~defined_mask).sum()
results = results.loc[defined_mask].copy()
decile_curves = decile_curves[defined_mask]
print(f"n_units analyzed: {len(results)} "
      f"({n_undefined} excluded: constant/undefined evoked FR, likely near-silent units)")

stat, pval_wilcoxon = wilcoxon(results["rho"])
print(f"Wilcoxon signed-rank test on per-unit rho vs 0: stat={stat:.1f}, p={pval_wilcoxon:.4g}")
print(f"median rho={results['rho'].median():.3f}, "
      f"fraction rho<0: {(results['rho'] < 0).mean():.2f}, "
      f"fraction |rho|>0.2: {(results['rho'].abs() > 0.2).mean():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: distribution of per-unit trend (rho), by region.
ax = axes[0]
for region, color in zip(["ORBl", "ORBm", "ORBvl"], ["tab:blue", "tab:orange", "tab:green"]):
    sub = results.loc[results["region"] == region, "rho"]
    if len(sub):
        ax.hist(sub, bins=25, range=(-0.6, 0.6), histtype="step", lw=1.6, color=color,
                label=f"{region} (n={len(sub)})")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axvline(results["rho"].median(), color="k", lw=1.2, label=f"median={results['rho'].median():.3f}")
ax.set_xlabel("per-unit Spearman rho (evoked FR vs. reward index)")
ax.set_ylabel("number of units")
ax.set_title(f"OFC exploration set (n={len(results)} units)\nWilcoxon p={pval_wilcoxon:.3g}")
ax.legend(fontsize=8)

# Panel 2: same distribution, faceted by subject (colored), to check no single subject dominates.
ax = axes[1]
subjects = sorted(results["subject"].unique())
positions = np.arange(len(subjects))
box_data = [results.loc[results["subject"] == s, "rho"].to_numpy() for s in subjects]
ax.boxplot(box_data, positions=positions, widths=0.6, showfliers=False)
for i, s in enumerate(subjects):
    y = results.loc[results["subject"] == s, "rho"]
    ax.scatter(np.full(len(y), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(y)),
               y, s=8, alpha=0.4, color="tab:gray")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xticks(positions)
ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("per-unit rho")
ax.set_title("per-unit trend by subject")

# Panel 3: grand-average population trend curve across the session (z-scored, pooled units).
ax = axes[2]
decile_x = (np.arange(N_DECILE_BINS) + 0.5) / N_DECILE_BINS
grand_mean = np.nanmean(decile_curves, axis=0)
grand_sem = np.nanstd(decile_curves, axis=0) / np.sqrt(np.sum(~np.isnan(decile_curves), axis=0))
ax.plot(decile_x, grand_mean, "o-", color="tab:purple")
ax.fill_between(decile_x, grand_mean - grand_sem, grand_mean + grand_sem, color="tab:purple", alpha=0.25)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("fraction of session (reward index / total rewards)")
ax.set_ylabel("z-scored evoked FR\n(mean +/- SEM across units)")
ax.set_title("grand-average trend across exploration-set OFC units")

fig.suptitle(
    "Population-level view: OFC evoked reward response vs. reward index within session\n"
    "(exploration set only, 8 subjects, 589 units)",
    y=1.03,
)
fig.tight_layout()
fig.savefig("001_population_ofc_reward_trend.png", dpi=150, bbox_inches="tight")
print("saved 001_population_ofc_reward_trend.png")
