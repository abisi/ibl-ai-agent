"""QA check: single-session overview combining spikes, task events, epochs, and DLC.

Goal: visually confirm that all data streams compressed into ssl_ephys and
ssl_behavior share one consistent, correctly-aligned time axis for a given
session -- population spiking activity, trial/stimulus events, epoch
boundaries (passive_pre/active/passive_post), and continuous DLC tracking.

Two views per session:
1. Full-session overview: population firing-rate heatmap (units grouped by
   region), epoch shading, trial event rug, and two DLC traces (pupil area,
   jaw velocity), all on one shared full-session time axis.
2. Zoomed literal spike raster (individual dots) over a short window of a few
   example trials within the active epoch, with the same event/DLC overlays,
   to confirm alignment holds at fine time resolution too.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard
from ibl_ai_agent.datasets.ssl_behavior import load_tracking_shard

FULL_SESSION_BIN_S = 5.0
ZOOM_WINDOW_S = 60.0
ZOOM_N_UNITS = 120  # stratified subsample across units for a readable literal raster

eph_d = resolve_dataset_dir("ssl_ephys")
beh_d = resolve_dataset_dir("ssl_behavior")

sessions = pd.read_parquet(eph_d / "metadata/sessions.parquet")
units_all = pd.read_parquet(eph_d / "metadata/units.parquet")
trials_all = pd.read_parquet(eph_d / "metadata/trials.parquet")
epochs_all = pd.read_parquet(eph_d / "metadata/epochs.parquet")

EPOCH_COLORS = {"passive_pre": "#cfe8ff", "active": "#ffe8b3", "passive_post": "#d8f5d0"}
TRIAL_COLORS = {"auditory_trial": "tab:green", "whisker_trial": "tab:blue", "no_stim_trial": "tab:red"}

ephys_sessions = sessions.loc[sessions["has_ephys"], ["session_id", "subject_id"]].reset_index(drop=True)

for _, row in ephys_sessions.iterrows():
    session_id, subject_id = row["session_id"], row["subject_id"]
    print(f"session {session_id}")

    units = units_all[units_all["session_id"] == session_id].copy()
    units = units.sort_values("ccf_acronym").reset_index(drop=True)
    cluster_ids_sorted = units["cluster_id"].to_numpy()

    shard = load_spike_shard(eph_d / "spikes" / session_id)
    spike_times = shard["spike_times_seconds"]
    spike_clusters_dense = shard["spike_clusters"]
    cluster_ids_shard = shard["cluster_ids"]
    spike_cluster_ids = cluster_ids_shard[spike_clusters_dense]
    # map each spike's real cluster_id to its row position in `units` (region-sorted)
    cluster_to_row = {cid: i for i, cid in enumerate(cluster_ids_sorted)}
    spike_row = np.array([cluster_to_row.get(c, -1) for c in spike_cluster_ids])
    valid = spike_row >= 0

    session_end = float(max(spike_times.max(), trials_all.loc[trials_all["session_id"] == session_id, "stop_time"].max()))
    t_edges = np.arange(0, session_end + FULL_SESSION_BIN_S, FULL_SESSION_BIN_S)
    n_bins = len(t_edges) - 1
    n_units = len(units)

    heat = np.zeros((n_units, n_bins), dtype=np.float32)
    bin_idx = np.clip(np.searchsorted(t_edges, spike_times[valid], side="right") - 1, 0, n_bins - 1)
    np.add.at(heat, (spike_row[valid], bin_idx), 1)
    heat = heat / FULL_SESSION_BIN_S  # firing rate
    heat_z = (heat - heat.mean(axis=1, keepdims=True)) / (heat.std(axis=1, keepdims=True) + 1e-9)

    ep = epochs_all[epochs_all["session_id"] == session_id]
    t = trials_all[trials_all["session_id"] == session_id]

    beh_avail_path = beh_d / "tracking" / session_id
    has_tracking = beh_avail_path.exists()
    if has_tracking:
        tracking = load_tracking_shard(beh_avail_path)
        dlc_time = tracking["timing_group_0.timestamps_seconds"]
        pupil = tracking["timing_group_0.pupil_area"]
        jaw_vel = np.abs(tracking["timing_group_0.jaw_velocity"])

    # ---------------- Figure A: full-session overview ----------------
    n_rows = 4 if has_tracking else 2
    fig = plt.figure(figsize=(14, 2.2 * n_rows + 2))
    gs = GridSpec(n_rows, 1, height_ratios=[0.3, 3] + ([0.8, 0.8] if has_tracking else []), hspace=0.15)

    ax_epoch = fig.add_subplot(gs[0])
    for _, e in ep.iterrows():
        ax_epoch.axvspan(e["start_time"], e["stop_time"], color=EPOCH_COLORS.get(e["epoch_name"], "gray"), alpha=0.8)
        ax_epoch.text((e["start_time"] + e["stop_time"]) / 2, 0.5, e["epoch_name"], ha="center", va="center", fontsize=7)
    ax_epoch.set_xlim(0, session_end)
    ax_epoch.set_yticks([])
    ax_epoch.set_title(f"{subject_id} ({session_id}) -- full-session overview", fontsize=10)

    ax_heat = fig.add_subplot(gs[1], sharex=ax_epoch)
    ax_heat.imshow(heat_z, aspect="auto", extent=[0, session_end, n_units, 0], cmap="viridis", vmin=-1, vmax=3)
    # region boundary labels
    region_boundaries = units["ccf_acronym"].ne(units["ccf_acronym"].shift()).to_numpy().nonzero()[0]
    for b in region_boundaries:
        ax_heat.axhline(b, color="w", lw=0.3, alpha=0.5)
    for b in region_boundaries:
        ax_heat.text(-0.01 * session_end, b, units["ccf_acronym"].iloc[b], fontsize=5, ha="right", va="top")
    ax_heat.set_ylabel(f"units (n={n_units}, region-sorted)")
    for trial_type, color in TRIAL_COLORS.items():
        stim_times = t.loc[t["trial_type"] == trial_type, "stim_onset"].dropna().to_numpy()
        ax_heat.scatter(stim_times, np.full(len(stim_times), -n_units * 0.02), marker="|", color=color, s=15, clip_on=False)

    if has_tracking:
        ax_pupil = fig.add_subplot(gs[2], sharex=ax_epoch)
        ax_pupil.plot(dlc_time, pupil, color="purple", lw=0.5)
        ax_pupil.set_ylabel("pupil area\n(a.u.)", fontsize=8)

        ax_jaw = fig.add_subplot(gs[3], sharex=ax_epoch)
        ax_jaw.plot(dlc_time, jaw_vel, color="darkorange", lw=0.5)
        ax_jaw.set_ylabel("|jaw velocity|\n(a.u.)", fontsize=8)
        ax_jaw.set_xlabel("time (s)")
    else:
        ax_heat.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(f"002_overview_{session_id}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 002_overview_{session_id}.png")

    # ---------------- Figure B: zoomed literal raster ----------------
    active = ep[ep["epoch_name"] == "active"]
    zoom_start = float(active["start_time"].iloc[0]) + 120.0  # skip session start transient
    zoom_end = zoom_start + ZOOM_WINDOW_S

    zoom_unit_idx = np.linspace(0, n_units - 1, min(ZOOM_N_UNITS, n_units)).round().astype(int)
    zoom_rows_set = set(zoom_unit_idx.tolist())

    fig2 = plt.figure(figsize=(14, 8))
    gs2 = GridSpec(3, 1, height_ratios=[3, 0.4, 0.8], hspace=0.15)
    ax_raster = fig2.add_subplot(gs2[0])
    mask_zoom = valid & (spike_times >= zoom_start) & (spike_times < zoom_end) & np.isin(spike_row, zoom_unit_idx)
    row_map = {orig: i for i, orig in enumerate(sorted(zoom_rows_set))}
    y_vals = np.array([row_map[r] for r in spike_row[mask_zoom]])
    ax_raster.scatter(spike_times[mask_zoom], y_vals, marker="|", s=3, color="k")
    for trial_type, color in TRIAL_COLORS.items():
        stim_times = t.loc[t["trial_type"] == trial_type, "stim_onset"].dropna().to_numpy()
        stim_times = stim_times[(stim_times >= zoom_start) & (stim_times < zoom_end)]
        for st in stim_times:
            ax_raster.axvline(st, color=color, lw=1.0, alpha=0.6)
    lick_times = t["lick_time"].dropna().to_numpy()
    lick_times = lick_times[(lick_times >= zoom_start) & (lick_times < zoom_end)]
    ax_raster.scatter(lick_times, np.full(len(lick_times), -3), marker="v", color="magenta", s=20, clip_on=False, label="lick")
    ax_raster.set_xlim(zoom_start, zoom_end)
    ax_raster.set_ylabel(f"unit (subsample, n={len(zoom_rows_set)})")
    ax_raster.set_title(f"{subject_id} ({session_id}) -- zoomed raster, {ZOOM_WINDOW_S:.0f}s window in active epoch\nvertical lines = stim onset (green=auditory, blue=whisker, red=no-stim), magenta triangle = lick", fontsize=9)

    if has_tracking:
        ax_dlc = fig2.add_subplot(gs2[2], sharex=ax_raster)
        mask_dlc = (dlc_time >= zoom_start) & (dlc_time < zoom_end)
        ax_dlc.plot(dlc_time[mask_dlc], jaw_vel[mask_dlc], color="darkorange", lw=0.8, label="|jaw velocity|")
        ax_dlc2 = ax_dlc.twinx()
        ax_dlc2.plot(dlc_time[mask_dlc], pupil[mask_dlc], color="purple", lw=0.8, label="pupil area")
        ax_dlc.set_xlabel("time (s)")
        ax_dlc.set_ylabel("|jaw vel|", color="darkorange", fontsize=8)
        ax_dlc2.set_ylabel("pupil area", color="purple", fontsize=8)

    fig2.tight_layout()
    fig2.savefig(f"002_zoomraster_{session_id}.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved 002_zoomraster_{session_id}.png")
