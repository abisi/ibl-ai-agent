"""Unit-level validation of the temporal-binning method behind
010_temporal_dynamics_preview.py: full-session raster (burst spikes marked)
with the 20 bin edges overlaid, paired with that unit's own per-bin
burstiness trace -- the actual data underlying the per-session aggregate
curves shown in the preview, requested before committing to a full-scale
rerun.
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
temporal = importlib.import_module("010_temporal_dynamics_preview")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
N_BINS = temporal.N_BINS

# One example per available (mouse, cohort, arm) combination in the small subset.
EXAMPLE_SESSIONS = ["AB080_20230622_152205", "AB085_20231005_152636", "MH021_20250311_110321"]


def pick_example_units(sess_units: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Low/median/high continuous-burstiness units among reasonably active
    ones -- shows the range of individual heterogeneity within one session,
    not just one representative unit."""
    candidates = sess_units[sess_units["n_spikes_clean"] >= 2000]
    if len(candidates) < 3:
        candidates = sess_units
    ordered = candidates.sort_values("continuous_burstiness").reset_index(drop=True)
    n = len(ordered)
    return [
        ("low", ordered.iloc[max(0, int(n * 0.15))]),
        ("median", ordered.iloc[n // 2]),
        ("high", ordered.iloc[min(n - 1, int(n * 0.85))]),
    ]


def main() -> None:
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    n_rows = len(EXAMPLE_SESSIONS) * 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.6 * n_rows),
                              gridspec_kw={"width_ratios": [3, 1.2]})

    row_i = 0
    for target_session in EXAMPLE_SESSIONS:
        day_stage = "learning" if target_session.startswith(("AB080", "AB085")) else "expert"
        files = loader.LEARNING_FILES if day_stage == "learning" else loader.EXPERT_FILES
        ut, tt = loader.load_arm(day_stage, files)
        ut = loader.apply_mouse_filters(ut, ref_df)
        ut = ut[ut["quality_label"].isin(["good", "mua"])].copy()
        sess_units = ut[ut.session_id == target_session].copy()
        sess_trials = tt[tt.session_id == target_session]
        wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
        t_start = float(sess_trials["start_time"].min())
        t_end = float(sess_trials["stop_time"].max())

        # compute n_spikes_clean + continuous_burstiness per unit to rank low/median/high
        n_spikes, burstiness_vals = [], []
        clean_cache = {}
        for t in sess_units.itertuples():
            spikes = np.sort(np.asarray(t.spike_times, dtype=float))
            _, clean, is_burst = burst_lib.continuous_burstiness(spikes, wh_starts)
            clean_cache[t.unit_uid] = (clean, is_burst)
            n_spikes.append(len(clean))
            burstiness_vals.append(is_burst.mean() if len(clean) else np.nan)
        sess_units["n_spikes_clean"] = n_spikes
        sess_units["continuous_burstiness"] = burstiness_vals

        for tier_name, example_row in pick_example_units(sess_units):
            clean, is_burst = clean_cache[example_row.unit_uid]
            bin_frac = temporal.normalized_bin_burstiness(clean, is_burst, t_start, t_end, n_bins=N_BINS)
            bin_edges = np.linspace(t_start, t_end, N_BINS + 1)

            ax_raster, ax_bar = axes[row_i]
            ax_raster.scatter((clean[~is_burst] - t_start) / 60, np.zeros((~is_burst).sum()), s=1, color="0.5", marker="|")
            ax_raster.scatter((clean[is_burst] - t_start) / 60, np.zeros(is_burst.sum()), s=4, color="crimson", marker="|")
            for e in bin_edges:
                ax_raster.axvline((e - t_start) / 60, color="0.85", linewidth=0.6, zorder=0)
            ax_raster.set_yticks([])
            ax_raster.set_xlabel("elapsed time (min)")
            ax_raster.set_title(f"{target_session} ({example_row.mouse_id}, {example_row.reward_group}) -- "
                                 f"unit {example_row.unit_uid.split('::')[-1]} ({example_row.quality_label}), "
                                 f"tier={tier_name}, n={len(clean)} spikes, burstiness={is_burst.mean():.3f}", fontsize=9)

            ax_bar.bar(np.arange(N_BINS), bin_frac, color="#4C72B0" if example_row.reward_group == "R+" else "#DD8452")
            ax_bar.set_xlabel("bin (session progression)")
            ax_bar.set_ylabel("burstiness")
            ax_bar.set_title(f"per-bin burstiness ({tier_name})", fontsize=9)
            row_i += 1

    fig.suptitle("Temporal-binning method validation: full-session raster + resulting per-bin burstiness", y=1.002)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = ARTIFACTS_DIR / "temporal_binning_unit_examples.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
