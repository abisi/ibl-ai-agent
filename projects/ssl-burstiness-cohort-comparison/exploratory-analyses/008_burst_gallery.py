"""Extended gallery of example burst rasters: continuous (spontaneous,
densest-burst window) and sensory-evoked (peri-event, both signs of
burst_index), across quality tiers where available. Follow-up to the
intermediate checkpoint's single continuous example and 3-per-modality
sensory-evoked examples.
"""
from __future__ import annotations

import sys
import importlib
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
MIN_WINDOW_SPIKES = 15


def get_spike_times(unit_table, unit_uid) -> np.ndarray:
    matches = unit_table[unit_table.unit_uid == unit_uid]
    assert len(matches) == 1
    return np.sort(np.asarray(matches.iloc[0]["spike_times"], dtype=float))


def densest_window_start(spike_times, weight, window):
    if len(spike_times) == 0:
        return 0.0
    counts = np.zeros(len(spike_times))
    for i, t0 in enumerate(spike_times):
        hi = np.searchsorted(spike_times, t0 + window, side="left")
        counts[i] = weight[i:hi].sum()
    return float(spike_times[np.argmax(counts)])


def fig_continuous_gallery(examples: list[dict], window: float = 5.0) -> plt.Figure:
    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, ex in zip(axes, examples):
        spikes, is_burst = ex["spikes"], ex["is_burst"]
        t0 = densest_window_start(spikes, is_burst.astype(float), window)
        m = (spikes >= t0) & (spikes < t0 + window)
        s, b = spikes[m], is_burst[m]
        ax.eventplot(s[~b], lineoffsets=0, colors="0.4", linelengths=0.8)
        ax.eventplot(s[b], lineoffsets=0, colors="crimson", linelengths=0.8)
        ax.set_yticks([])
        ax.set_ylabel(f"{ex['unit_uid'].split('::')[-1]}\n({ex['quality_label']})\nburstiness={ex['burstiness']:.2f}",
                       rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("time (s), densest 5s burst window per unit")
    fig.suptitle("Continuous burstiness examples (grey=non-burst, red=burst spikes)")
    fig.tight_layout()
    return fig


def fig_evoked_gallery(examples: list[dict], label: str) -> plt.Figure:
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=False)
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
        sign = "+" if ex["burst_index"] >= 0 else "-"
        ax.set_title(f"{ex['unit_uid'].split('::')[0][:6]}/{ex['unit_uid'].split('::')[-1]} ({ex['quality_label']})\n"
                     f"burst_index={ex['burst_index']:.2f} ({sign})", fontsize=8)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("trial #")
    fig.suptitle(f"Sensory-evoked burst-index examples -- {label} trials (blue=response 5-50ms, orange=baseline -200/-10ms)")
    fig.tight_layout()
    return fig


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "small_subset_unit_metrics.parquet")
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")

    units_by_arm, trials_by_arm = {}, {}
    for day_stage, files in [("learning", loader.LEARNING_FILES), ("expert", loader.EXPERT_FILES)]:
        ut, tt = loader.load_arm(day_stage, files)
        ut = loader.apply_mouse_filters(ut, ref_df)
        ut = ut[ut["quality_label"] != "non-soma"].copy()
        units_by_arm[day_stage] = ut
        trials_by_arm[day_stage] = tt

    # --- Continuous: 6 examples spanning the burstiness range, both tiers ---
    well_sampled = df[df["n_spikes_clean"] >= 500]
    quantile_targets = well_sampled["continuous_burstiness"].quantile([0.1, 0.3, 0.5, 0.7, 0.9, 0.99]).to_numpy()
    picked_uids = set()
    cont_examples = []
    for q in quantile_targets:
        remaining = well_sampled[~well_sampled.unit_uid.isin(picked_uids)]
        idx = (remaining["continuous_burstiness"] - q).abs().idxmin()
        row = remaining.loc[idx]
        picked_uids.add(row.unit_uid)
        ut = units_by_arm[row.day_stage]
        tt = trials_by_arm[row.day_stage]
        spikes = get_spike_times(ut, row.unit_uid)
        sess_trials = tt[tt.session_id == row.session_id]
        wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
        burstiness_check, clean, is_burst = burst_lib.continuous_burstiness(spikes, wh_starts)
        assert abs(burstiness_check - row.continuous_burstiness) < 1e-9, "recomputed burstiness disagrees with stored metric"
        cont_examples.append({"unit_uid": row.unit_uid, "quality_label": row.quality_label,
                               "burstiness": row.continuous_burstiness, "spikes": clean, "is_burst": is_burst})
    fig1 = fig_continuous_gallery(cont_examples)
    fig1.savefig(ARTIFACTS_DIR / "gallery_continuous.png", dpi=140, bbox_inches="tight")
    print("Continuous examples:")
    for ex in cont_examples:
        print(f"  {ex['unit_uid']} ({ex['quality_label']}): burstiness={ex['burstiness']:.3f}, n_spikes={len(ex['spikes'])}")

    # --- Sensory-evoked: 3 positive + 2 negative per modality, well-sampled ---
    for metric, is_whisker, label in [("burst_index_whisker", True, "whisker"), ("burst_index_auditory", False, "auditory")]:
        well = df[(df[f"n_total_response_{label}"] >= MIN_WINDOW_SPIKES) & (df[f"n_total_baseline_{label}"] >= MIN_WINDOW_SPIKES)]
        top_pos = well.nlargest(3, metric)
        top_neg = well.nsmallest(2, metric)
        selection = pd.concat([top_pos, top_neg])
        examples = []
        for _, r in selection.iterrows():
            ut_r = units_by_arm[r.day_stage]
            tt_r = trials_by_arm[r.day_stage]
            spikes = get_spike_times(ut_r, r.unit_uid)
            sess_trials_r = tt_r[tt_r.session_id == r.session_id]
            wh_starts_r = np.sort(sess_trials_r.loc[sess_trials_r.trial_type == "whisker_trial", "start_time"].to_numpy(float))
            trial_col = "whisker_trial" if is_whisker else "auditory_trial"
            starts_r = np.sort(sess_trials_r.loc[sess_trials_r.trial_type == trial_col, "start_time"].to_numpy(float))
            clean_r = burst_lib.excise_whisker_dead_zone(spikes, wh_starts_r) if is_whisker else spikes
            is_burst_r = burst_lib.label_bursts(clean_r)
            examples.append({"unit_uid": r.unit_uid, "quality_label": r.quality_label, "spikes": clean_r,
                              "is_burst": is_burst_r, "start_times": starts_r, "burst_index": r[metric]})
        fig = fig_evoked_gallery(examples, label)
        fig.savefig(ARTIFACTS_DIR / f"gallery_evoked_{label}.png", dpi=140, bbox_inches="tight")
        print(f"\n{label} examples:")
        for ex in examples:
            print(f"  {ex['unit_uid']} ({ex['quality_label']}): burst_index={ex['burst_index']:.3f}")

    print(f"\nWrote gallery_continuous.png, gallery_evoked_whisker.png, gallery_evoked_auditory.png to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
