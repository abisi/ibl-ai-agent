"""QA check: event-aligned neural activity, single-neuron and population-by-area.

Goal: sanity-check that the ssl_ephys NWB->compressed conversion preserved
correct spike times, trial events, and region labels, by verifying that
PSTHs aligned to stim_onset look like real, structured sensory/task responses
(not noise), for individual example neurons and for population averages
grouped by brain region.

Alignment event: trials.stim_onset (present and populated for all trial
types: auditory/whisker/no_stim), restricted to each session's own "active"
epoch window (per epochs.parquet). 10ms bins for single-neuron PSTHs;
population PSTHs use a coarser bin for a smoother group-average curve.

Region grouping: exact ccf_acronym, restricted to gray-matter regions (Allen
CCF acronyms starting with an uppercase letter) to exclude fiber tracts
('or', 'ar', 'scwm', 'ccb', ...) and missing labels ('nan'), then the top 6
regions by unit count per session.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainbox.singlecell import calculate_peths

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard

BIN_SIZE_SINGLE = 0.01  # seconds
BIN_SIZE_POP = 0.02
PRE_TIME, POST_TIME = 0.3, 0.8
N_TOP_REGIONS = 6
N_EXAMPLE_UNITS = 3

d = resolve_dataset_dir("ssl_ephys")
sessions = pd.read_parquet(d / "metadata/sessions.parquet")
units_all = pd.read_parquet(d / "metadata/units.parquet")
trials_all = pd.read_parquet(d / "metadata/trials.parquet")
epochs_all = pd.read_parquet(d / "metadata/epochs.parquet")

ephys_sessions = sessions.loc[sessions["has_ephys"], ["session_id", "subject_id"]].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Figure 1: population PSTH by region, one panel per session
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9))
axes1 = axes1.ravel()

# Figure 2: example single-neuron PSTHs (hit vs miss / catch), a few per session
fig2, axes2 = plt.subplots(N_EXAMPLE_UNITS, len(ephys_sessions), figsize=(4.2 * len(ephys_sessions), 3.0 * N_EXAMPLE_UNITS), sharex=True)

for col, (_, row) in enumerate(ephys_sessions.iterrows()):
    session_id, subject_id = row["session_id"], row["subject_id"]
    ax1 = axes1[col]

    active = epochs_all[(epochs_all["session_id"] == session_id) & (epochs_all["epoch_name"] == "active")]
    lo, hi = active["start_time"].iloc[0], active["stop_time"].iloc[0]
    t = trials_all[trials_all["session_id"] == session_id]
    t = t[(t["stim_onset"] >= lo) & (t["stim_onset"] <= hi)]

    shard = load_spike_shard(d / "spikes" / session_id)
    spike_times = shard["spike_times_seconds"]
    spike_clusters_dense = shard["spike_clusters"]
    cluster_ids = shard["cluster_ids"]
    spike_cluster_ids = cluster_ids[spike_clusters_dense]

    units = units_all[units_all["session_id"] == session_id].copy()
    is_gray_matter = units["ccf_acronym"].astype(str).str[0].str.isupper()
    region_counts = units.loc[is_gray_matter, "ccf_acronym"].value_counts()
    top_regions = region_counts.head(N_TOP_REGIONS).index.tolist()

    align_times = t["stim_onset"].dropna().to_numpy()
    cmap = plt.get_cmap("tab10")
    for i, region in enumerate(top_regions):
        region_cluster_ids = units.loc[units["ccf_acronym"] == region, "cluster_id"].to_numpy()
        peths, _ = calculate_peths(
            spike_times, spike_cluster_ids, region_cluster_ids, align_times,
            pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE_POP, smoothing=0.02,
        )
        tscale = peths["tscale"]
        z = (peths["means"] - peths["means"].mean(axis=1, keepdims=True)) / (peths["means"].std(axis=1, keepdims=True) + 1e-9)
        ax1.plot(tscale, z.mean(axis=0), color=cmap(i), lw=1.6, label=f"{region} (n={len(region_cluster_ids)})")

    ax1.axvline(0, color="k", lw=0.8, ls="--")
    ax1.set_title(f"{subject_id} ({session_id})\nn_trials={len(align_times)}", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_xlabel("time from stim onset (s)")
    ax1.set_ylabel("z-scored firing rate\n(mean across region units)")

    # --- example single-neuron PSTHs: hit vs miss for stim trials, or resp vs no-resp for catch
    is_stim = t["trial_type"].isin(["auditory_trial", "whisker_trial"])
    hit_times = t.loc[is_stim & (t["lick_flag"] == 1), "stim_onset"].dropna().to_numpy()
    miss_times = t.loc[is_stim & (t["lick_flag"] == 0), "stim_onset"].dropna().to_numpy()

    example_units = units.sort_values("firing_rate", ascending=False).iloc[
        np.linspace(0, min(len(units), 60) - 1, N_EXAMPLE_UNITS).round().astype(int)
    ]
    for row_i, (_, u) in enumerate(example_units.iterrows()):
        ax = axes2[row_i, col] if len(ephys_sessions) > 1 else axes2[row_i]
        cluster_id = u["cluster_id"]
        for align_times_ex, color, label in [(hit_times, "tab:green", "hit"), (miss_times, "tab:red", "miss")]:
            if len(align_times_ex) < 3:
                continue
            peths, _ = calculate_peths(
                spike_times, spike_cluster_ids, [cluster_id], align_times_ex,
                pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE_SINGLE, smoothing=0.03,
            )
            ax.plot(peths["tscale"], peths["means"][0], color=color, lw=1.3, label=label)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(f"{u['ccf_acronym']} clu {cluster_id} (FR={u['firing_rate']:.1f} sp/s)", fontsize=8)
        if row_i == 0:
            ax.set_title(f"{subject_id}\n{u['ccf_acronym']} clu {cluster_id} (FR={u['firing_rate']:.1f} sp/s)", fontsize=8)
            ax.legend(fontsize=7)

for ax in axes2[-1, :] if axes2.ndim > 1 else [axes2[-1]]:
    ax.set_xlabel("time from stim onset (s)")

fig1.suptitle("Population activity by brain region, aligned to stim onset (active epoch, z-scored)", y=1.0)
fig1.tight_layout()
fig1.savefig("001_population_psth_by_region.png", dpi=150, bbox_inches="tight")
print("saved 001_population_psth_by_region.png")

fig2.suptitle("Example single-neuron PSTHs: hit vs miss trials, aligned to stim onset", y=1.0)
fig2.tight_layout()
fig2.savefig("001_example_single_neuron_psths.png", dpi=150, bbox_inches="tight")
print("saved 001_example_single_neuron_psths.png")
