"""Preview (small validated subset only, 5 sessions after mouse filters) of
how burstiness evolves across a session, both normalized to each session's
own progression (0-1) and in absolute elapsed time -- requested as a
follow-up. Explicitly a preview: this subset has only ~2-3 mice per cohort
per arm, nowhere near enough for a real cohort comparison -- the point here
is to validate the method and see whether it's worth the expensive
full-population reprocessing (each arm took ~15-80min for scalar metrics
alone; per-bin metrics will cost more).

Session span: [min(trials.start_time), max(trials.stop_time)] per session
(task span, not raw spike extent -- avoids being skewed by a stray very
early/late spike).
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
N_BINS = 20
ABS_BIN_WIDTH_S = 180.0  # 3-minute absolute bins
ABS_N_BINS = 20  # covers up to 60 minutes


def normalized_bin_burstiness(clean_spikes, is_burst, t_start, t_end, n_bins=N_BINS):
    edges = np.linspace(t_start, t_end, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, clean_spikes, side="right") - 1, 0, n_bins - 1)
    n_total = np.bincount(idx, minlength=n_bins)
    n_burst = np.bincount(idx[is_burst], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_total > 0, n_burst / np.maximum(n_total, 1), np.nan)
    return frac


def absolute_bin_burstiness(clean_spikes, is_burst, t_start, t_end, bin_width=ABS_BIN_WIDTH_S, n_bins=ABS_N_BINS):
    rel = clean_spikes - t_start
    idx = np.floor(rel / bin_width).astype(int)
    valid = (idx >= 0) & (idx < n_bins)
    frac = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = valid & (idx == b)
        if b * bin_width > (t_end - t_start):
            continue  # session ended before this bin started
        n_total = m.sum()
        if n_total > 0:
            frac[b] = is_burst[m].sum() / n_total
    return frac


def main() -> None:
    ref_df = pd.read_excel(loader.REF_XLSX, sheet_name="Sheet1")
    rows_norm, rows_abs = [], []

    for day_stage, files in [("learning", loader.LEARNING_FILES), ("expert", loader.EXPERT_FILES)]:
        ut, tt = loader.load_arm(day_stage, files)
        ut = loader.apply_mouse_filters(ut, ref_df)
        ut = ut[ut["quality_label"].isin(["good", "mua"])].copy()
        for session_id, sess_units in ut.groupby("session_id"):
            sess_trials = tt[tt.session_id == session_id]
            wh_starts = np.sort(sess_trials.loc[sess_trials.trial_type == "whisker_trial", "start_time"].to_numpy(float))
            t_start = float(sess_trials["start_time"].min())
            t_end = float(sess_trials["stop_time"].max())
            mouse_id = sess_units["mouse_id"].iloc[0]
            reward_group = sess_units["reward_group"].iloc[0]
            print(f"  {session_id} ({mouse_id}, {reward_group}): span {t_end-t_start:.0f}s, {len(sess_units)} units")

            unit_norm, unit_abs = [], []
            for t in sess_units.itertuples():
                spikes = np.sort(np.asarray(t.spike_times, dtype=float))
                _, clean, is_burst = burst_lib.continuous_burstiness(spikes, wh_starts)
                if len(clean) < 20:
                    continue
                unit_norm.append(normalized_bin_burstiness(clean, is_burst, t_start, t_end))
                unit_abs.append(absolute_bin_burstiness(clean, is_burst, t_start, t_end))
            if not unit_norm:
                continue
            session_norm = np.nanmean(np.vstack(unit_norm), axis=0)
            session_abs = np.nanmean(np.vstack(unit_abs), axis=0)
            rows_norm.append({"session_id": session_id, "mouse_id": mouse_id, "day_stage": day_stage,
                               "reward_group": reward_group, **{f"bin_{i}": session_norm[i] for i in range(N_BINS)}})
            rows_abs.append({"session_id": session_id, "mouse_id": mouse_id, "day_stage": day_stage,
                              "reward_group": reward_group, **{f"bin_{i}": session_abs[i] for i in range(ABS_N_BINS)}})

    norm_df = pd.DataFrame(rows_norm)
    abs_df = pd.DataFrame(rows_abs)
    norm_df.to_parquet(ARTIFACTS_DIR / "temporal_preview_normalized.parquet", index=False)
    abs_df.to_parquet(ARTIFACTS_DIR / "temporal_preview_absolute.parquet", index=False)
    print(f"\n{len(norm_df)} sessions with usable data")
    print(norm_df[["session_id", "mouse_id", "day_stage", "reward_group"]])

    # Plot: one line per session (not averaged across cohort -- too few
    # sessions here for that to mean anything), colored by cohort.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row_i, (df, bin_cols, xlabel, n_bins) in enumerate([
        (norm_df, [f"bin_{i}" for i in range(N_BINS)], "session progression (fraction)", N_BINS),
        (abs_df, [f"bin_{i}" for i in range(ABS_N_BINS)], "elapsed time (min)", ABS_N_BINS),
    ]):
        for col_i, day_stage in enumerate(["learning", "expert"]):
            ax = axes[row_i, col_i]
            sub = df[df.day_stage == day_stage]
            x = (np.arange(n_bins) + 0.5) / n_bins if row_i == 0 else (np.arange(n_bins) + 0.5) * ABS_BIN_WIDTH_S / 60
            for _, r in sub.iterrows():
                color = "#4C72B0" if r.reward_group == "R+" else "#DD8452"
                ax.plot(x, r[bin_cols].to_numpy(dtype=float), color=color, alpha=0.7,
                        label=f"{r.mouse_id} ({r.reward_group})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("mean burstiness")
            ax.set_title(f"{day_stage} -- {'normalized' if row_i==0 else 'absolute'}")
            ax.legend(fontsize=7)
    fig.suptitle("Burstiness across session (PREVIEW, small subset: 3 learning + 2 expert sessions only)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "temporal_dynamics_preview.png", dpi=140, bbox_inches="tight")
    print(f"\nWrote {ARTIFACTS_DIR / 'temporal_dynamics_preview.png'}")


if __name__ == "__main__":
    main()
