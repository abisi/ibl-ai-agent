"""Intermediate validation report (small subset only) -- burst-detection
figures and example burst-coding neurons, for sign-off before the full-scale
run. See ../question.md and TODO.md.
"""
from __future__ import annotations

import base64
import importlib
import sys
from pathlib import Path

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

burst_lib = importlib.import_module("000_burst_lib")
loader = importlib.import_module("001_small_subset_load")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FIG_DIR = ARTIFACTS_DIR / "intermediate_figs"
FIG_DIR.mkdir(exist_ok=True)


def fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def get_spike_times(unit_table, unit_uid) -> np.ndarray:
    matches = unit_table[unit_table.unit_uid == unit_uid]
    assert len(matches) == 1, f"expected exactly 1 row for unit_uid={unit_uid}, got {len(matches)}"
    return np.sort(np.asarray(matches.iloc[0]["spike_times"], dtype=float))


def densest_window_start(spike_times: np.ndarray, weight: np.ndarray, window: float) -> float:
    """Start time of the `window`-second slice with the largest sum of
    `weight` over spikes it contains (e.g. weight=is_burst to find where a
    unit's bursts are concentrated, rather than an arbitrary fixed window
    that can land on a silent stretch for a low-rate unit)."""
    if len(spike_times) == 0:
        return 0.0
    counts = np.zeros(len(spike_times))
    for i, t0 in enumerate(spike_times):
        hi = np.searchsorted(spike_times, t0 + window, side="left")
        counts[i] = weight[i:hi].sum()
    return float(spike_times[np.argmax(counts)])


def fig_raster_burst_vs_nonburst(bursty_spikes, bursty_is_burst, quiet_spikes, quiet_is_burst, window=5.0):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=False)
    bursty_t0 = densest_window_start(bursty_spikes, bursty_is_burst.astype(float), window)
    quiet_t0 = densest_window_start(quiet_spikes, np.ones(len(quiet_spikes)), window)
    for ax, spikes, is_burst, title, t0 in [
        (axes[0], bursty_spikes, bursty_is_burst, "Example bursty unit", bursty_t0),
        (axes[1], quiet_spikes, quiet_is_burst, "Example non-bursty unit", quiet_t0),
    ]:
        m = (spikes >= t0) & (spikes < t0 + window)
        s, b = spikes[m], is_burst[m]
        ax.eventplot(s[~b], lineoffsets=0, colors="0.4", linelengths=0.8, label="non-burst")
        ax.eventplot(s[b], lineoffsets=0, colors="crimson", linelengths=0.8, label="burst")
        ax.set_title(f"{title} ({b.sum()}/{len(s)} spikes in burst over {window}s shown)")
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_isi_histogram(all_isis_ms):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(np.log10(0.1), np.log10(2000), 60)
    ax.hist(all_isis_ms, bins=bins, color="#4C72B0")
    ax.set_xscale("log")
    ax.axvline(10, color="crimson", linestyle="--", label="10ms intra-burst max")
    ax.axvline(15, color="darkorange", linestyle="--", label="15ms tail min")
    ax.set_xlabel("ISI (ms, log scale)")
    ax.set_ylabel("count (pooled example units)")
    ax.legend(fontsize=8)
    ax.set_title("ISI distribution with burst thresholds")
    fig.tight_layout()
    return fig


def fig_window_illustration(spikes, is_burst, start_times, n_trials=15):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(-0.001, 0.004, color="0.85", label="dead zone (excised)")
    ax.axvspan(0.005, 0.050, color="#CDE3F5", label="response 5-50ms")
    ax.axvspan(-0.200, -0.010, color="#F5DCC9", label="baseline -200/-10ms")
    for i, t0 in enumerate(start_times[:n_trials]):
        m = (spikes >= t0 - 0.25) & (spikes < t0 + 0.06)
        rel = (spikes[m] - t0)
        b = is_burst[m]
        ax.scatter(rel[~b], np.full(m.sum() - b.sum(), i), s=6, color="0.3")
        ax.scatter(rel[b], np.full(b.sum(), i), s=10, color="crimson")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("time relative to start_time (s)")
    ax.set_ylabel("trial #")
    ax.set_title("Peri-event windows (whisker trials): dead zone, response, baseline")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def fig_burst_coding_examples(examples: list[dict], label: str):
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, ex in zip(axes, examples):
        spikes, is_burst, start_times = ex["spikes"], ex["is_burst"], ex["start_times"]
        ax.axvspan(0.005, 0.050, color="#CDE3F5", zorder=0)
        ax.axvspan(-0.200, -0.010, color="#F5DCC9", zorder=0)
        for i, t0 in enumerate(start_times):
            m = (spikes >= t0 - 0.25) & (spikes < t0 + 0.06)
            rel = spikes[m] - t0
            b = is_burst[m]
            ax.scatter(rel[~b], np.full((~b).sum(), i), s=4, color="0.3")
            ax.scatter(rel[b], np.full(b.sum(), i), s=8, color="crimson")
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_title(f"{ex['mouse_id']}/{ex['cluster_id']}\nburst_index={ex['burst_index']:.2f}", fontsize=9)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("trial #")
    fig.suptitle(f"Example burst-coding units -- {label} trials (largest |burst_index|)")
    fig.tight_layout()
    return fig


def fig_sanity_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, title in [(axes[0], "burst_index_whisker", "whisker"), (axes[1], "burst_index_auditory", "auditory")]:
        for tier, color in [("good", "#4C72B0"), ("mua", "#C44E52")]:
            sub = df[df.quality_label == tier]
            ax.scatter(sub["continuous_burstiness"], sub[col], s=6, alpha=0.4, color=color, label=tier)
        ax.set_xlabel("continuous burstiness")
        ax.set_ylabel(f"burst_index ({title})")
        ax.legend(fontsize=8)
        ax.axhline(0, color="0.7", linewidth=0.7)
    fig.suptitle("Continuous burstiness vs. sensory-evoked burst index (not the same quantity)")
    fig.tight_layout()
    return fig


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "small_subset_unit_metrics.parquet")
    print(f"Loaded {len(df)} units")

    # Reload raw unit/trial tables for the same small subset (spike_times not in the metrics parquet).
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    units_by_arm, trials_by_arm = {}, {}
    for day_stage, files in [("learning", loader.LEARNING_FILES), ("expert", loader.EXPERT_FILES)]:
        ut, tt = loader.load_arm(day_stage, files)
        ut = loader.apply_mouse_filters(ut, ref_df)
        ut = ut[ut["quality_label"] != "non-soma"].copy()
        units_by_arm[day_stage] = ut
        trials_by_arm[day_stage] = tt

    figs_html = []

    # --- Fig 1+2: bursty vs non-bursty example, ISI histogram ---
    valid = df[df["n_spikes_clean"] > 200]
    bursty_row = valid.loc[valid["continuous_burstiness"].idxmax()]
    quiet_row = valid.loc[valid["continuous_burstiness"].idxmin()]
    ut = units_by_arm[bursty_row["day_stage"]]
    bursty_spikes = get_spike_times(ut, bursty_row.unit_uid)
    bursty_is_burst = burst_lib.label_bursts(bursty_spikes)
    ut2 = units_by_arm[quiet_row["day_stage"]]
    quiet_spikes = get_spike_times(ut2, quiet_row.unit_uid)
    quiet_is_burst = burst_lib.label_bursts(quiet_spikes)
    print(f"Bursty example: {bursty_row.unit_uid}, burstiness={bursty_row.continuous_burstiness:.3f}")
    print(f"Quiet example: {quiet_row.unit_uid}, burstiness={quiet_row.continuous_burstiness:.3f}")
    figs_html.append(("Burst-labeled spike trains", fig_to_b64(fig_raster_burst_vs_nonburst(bursty_spikes, bursty_is_burst, quiet_spikes, quiet_is_burst))))

    all_isis_ms = np.concatenate([np.diff(bursty_spikes), np.diff(quiet_spikes)]) * 1000
    figs_html.append(("ISI histogram", fig_to_b64(fig_isi_histogram(all_isis_ms))))

    # --- Fig 4: example burst-coding neurons (top |burst_index|) ---
    # A unit with only 1-2 spikes total in a window can hit |burst_index|==1.0
    # by construction (fB=0 convention) without any real bursting signal --
    # found by inspecting this exact "example" selection (min spike-count
    # floor added in response). Require a reasonable spike count in BOTH
    # windows before ranking by burst_index magnitude.
    MIN_WINDOW_SPIKES = 15
    whisker_examples_for_window_fig = None
    for metric, is_whisker, label in [("burst_index_whisker", True, "whisker"), ("burst_index_auditory", False, "auditory")]:
        well_sampled = df[(df[f"n_total_response_{label}"] >= MIN_WINDOW_SPIKES) & (df[f"n_total_baseline_{label}"] >= MIN_WINDOW_SPIKES)]
        print(f"  {label}: {len(well_sampled)}/{len(df)} units have >={MIN_WINDOW_SPIKES} spikes in BOTH windows")
        top = well_sampled.reindex(well_sampled[metric].abs().sort_values(ascending=False).index[:3])
        examples = []
        for _, r in top.iterrows():
            ut_r = units_by_arm[r.day_stage]
            tt_r = trials_by_arm[r.day_stage]
            spikes = get_spike_times(ut_r, r.unit_uid)
            sess_trials_r = tt_r[tt_r.session_id == r.session_id]
            wh_starts_r = np.sort(sess_trials_r.loc[sess_trials_r.trial_type == "whisker_trial", "start_time"].to_numpy(float))
            trial_col = "whisker_trial" if is_whisker else "auditory_trial"
            starts_r = np.sort(sess_trials_r.loc[sess_trials_r.trial_type == trial_col, "start_time"].to_numpy(float))
            clean_r = burst_lib.excise_whisker_dead_zone(spikes, wh_starts_r) if is_whisker else spikes
            is_burst_r = burst_lib.label_bursts(clean_r)
            print(f"  {label} example: {r.unit_uid}, n_trials={len(starts_r)}, burst_index={r[metric]:.3f}")
            examples.append({"spikes": clean_r, "is_burst": is_burst_r, "start_times": starts_r,
                              "mouse_id": r.mouse_id, "cluster_id": r.cluster_id, "burst_index": r[metric]})
        figs_html.append((f"Example burst-coding units ({label})", fig_to_b64(fig_burst_coding_examples(examples, label))))
        if is_whisker:
            whisker_examples_for_window_fig = examples

    # --- Fig 3: peri-event window illustration, reusing a confirmed-active
    # whisker burst-coding example above (the earlier choice -- the unit with
    # the single highest overall continuous_burstiness -- turned out to have
    # ~no spikes near whisker trials at all; a real finding about that unit,
    # not useful for a didactic window-definition figure).
    ex0 = whisker_examples_for_window_fig[0]
    figs_html.insert(2, ("Peri-event window illustration", fig_to_b64(
        fig_window_illustration(ex0["spikes"], ex0["is_burst"], ex0["start_times"])
    )))

    # --- Fig 5: sanity scatter ---
    figs_html.append(("Continuous burstiness vs burst index (sanity check)", fig_to_b64(fig_sanity_scatter(df))))

    # --- summary stats ---
    summary = df.groupby(["day_stage", "quality_label"]).agg(
        n_units=("continuous_burstiness", "size"),
        median_burstiness=("continuous_burstiness", "median"),
        median_burst_index_whisker=("burst_index_whisker", "median"),
        median_burst_index_auditory=("burst_index_auditory", "median"),
    ).reset_index()
    print(summary.to_string(index=False))

    html_parts = [
        "<title>Burstiness Pipeline Check</title>",
        "<meta charset='utf-8'>",
        "<body style='font-family:sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5'>",
        "<h1>Burstiness pipeline -- intermediate validation report</h1>",
        "<p>Small subset only (3 learning + 3 expert known-ephys sessions, one filtered out by mouse-inclusion "
        "criteria), for sign-off before the full-scale run. See <code>question.md</code>/<code>TODO.md</code>.</p>",
        "<h2>Issues found and fixed while building this checkpoint</h2>"
        "<ul>"
        "<li><b>cluster_id is not session-unique in this Path B unit table</b> (unlike the compressed-dataset "
        "builder, which offsets it per probe) &mdash; verified directly: session MH021_20250311_110321 has "
        "cluster_id 314 on 3 of its 4 probes. Unit lookups by (mouse_id, session_id, cluster_id) alone were "
        "silently grabbing whichever probe's unit came first. Fixed with an explicit "
        "<code>unit_uid = session_id::electrode_group::cluster_id</code> key used for every lookup from here on.</li>"
        "<li><b>Degenerate burst_index at low spike counts</b>: a unit with only 1-2 spikes total in a window can "
        "hit |burst_index|==1.0 purely from the fB=0 convention, with no real bursting signal. The example "
        "burst-coding units below require &gt;=15 spikes in *both* the response and baseline windows before "
        "ranking by |burst_index| &mdash; without that floor, all 6 \"top\" examples were single/double-spike "
        "artifacts. The sanity scatter (last figure) does <i>not</i> apply this floor, so the bands of points "
        "sitting exactly at burst_index=+-1.0 there are this same artifact, not real signal &mdash; the full-scale "
        "statistics will need a spike-count floor (or NaN-out) decision for these units, flagged here for your "
        "input rather than assumed.</li>"
        "<li>The single unit with the highest <i>continuous</i> burstiness in this subset (MH021, imec2_shank0, "
        "cluster 314) turned out to have ~no spikes near whisker trials at all when checked directly &mdash; a "
        "real property of that unit (tonic/continuous bursting unrelated to task events), not a bug. The "
        "peri-event window-definition figure below uses a different, confirmed-active unit instead.</li>"
        "</ul>",
        f"<h2>Unit counts (this subset)</h2>{summary.to_html(index=False)}",
    ]
    for title, src in figs_html:
        html_parts.append(f"<h2>{title}</h2><img src='{src}' style='max-width:100%'>")
    html_parts.append("</body>")

    out_path = ARTIFACTS_DIR / "intermediate_report.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
