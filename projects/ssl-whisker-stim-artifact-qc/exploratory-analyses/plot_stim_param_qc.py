"""Whisker stimulus parameter QC: amplitude/strength/duration distributions
(global, per-session, per-mouse) and start_time / whisker_stim_time /
stim_onset timing-relationship check.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
ANALYSIS_UNITS = Path("reports/ssl_analysis/derived/analysis_units.parquet")
FIG_DIR = Path("projects/ssl-whisker-stim-artifact-qc/report/figures")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = pd.read_parquet(DATASET_DIR / "metadata/trials.parquet")
    sessions = pd.read_parquet(DATASET_DIR / "metadata/sessions.parquet")
    units = pd.read_parquet(ANALYSIS_UNITS)
    analyzed_sessions = set(units["session_id"].unique())

    wh = trials[(trials["trial_type"] == "whisker_trial") & (trials["session_id"].isin(analyzed_sessions))].copy()
    wh = wh.merge(sessions[["session_id", "subject_id"]], on="session_id", how="left")
    print(f"Whisker trials in the analyzed session set: {len(wh)}")

    # --- timing relationship ---
    for col in ("whisker_stim_time", "stim_onset"):
        if col in wh.columns:
            offset = wh[col] - wh["start_time"]
            print(f"{col} - start_time: n_nan={offset.isna().sum()}, "
                  f"mean={offset.mean():.6f}s, std={offset.std():.6f}s, "
                  f"min={offset.min():.6f}s, max={offset.max():.6f}s")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ("whisker_stim_amplitude", "whisker_stim_strength", "whisker_stim_duration")):
        if col not in wh.columns:
            ax.set_title(f"{col} (missing)")
            continue
        vals = wh[col].dropna()
        ax.hist(vals, bins=60, color="tab:brown", alpha=0.8)
        ax.set_title(f"{col}\n(n={len(vals)}, n_missing={wh[col].isna().sum()})")
        ax.set_xlabel(col)
    fig.suptitle("Whisker stimulus parameter distributions (all analyzed sessions pooled)")
    fig.tight_layout()
    out_path = FIG_DIR / "stim_param_distributions_global.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # --- per-mouse consistency: amplitude ---
    if "whisker_stim_amplitude" in wh.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        mice = sorted(wh["subject_id"].dropna().unique())
        data_by_mouse = [wh.loc[wh["subject_id"] == m, "whisker_stim_amplitude"].dropna() for m in mice]
        ax.boxplot(data_by_mouse, positions=range(len(mice)), showfliers=False)
        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(mice, rotation=90, fontsize=6)
        ax.set_ylabel("whisker_stim_amplitude")
        ax.set_title("Whisker stimulus amplitude per mouse (pooled across that mouse's sessions)")
        fig.tight_layout()
        out_path = FIG_DIR / "stim_amplitude_per_mouse.png"
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")

        # per-session, ordered within mouse (drift across sessions?)
        sess_summary = wh.groupby(["subject_id", "session_id"])["whisker_stim_amplitude"].agg(["mean", "std", "count"]).reset_index()
        sess_summary = sess_summary.sort_values(["subject_id", "session_id"])
        n_multi_session_mice = sess_summary.groupby("subject_id").size()
        print(f"\nMice with >1 analyzed session: {(n_multi_session_mice > 1).sum()} / {len(n_multi_session_mice)}")
        within_mouse_cv = wh.groupby("subject_id")["whisker_stim_amplitude"].agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)
        print("Within-mouse coefficient of variation (amplitude), describe:")
        print(within_mouse_cv.describe())

    wh.to_parquet("projects/ssl-whisker-stim-artifact-qc/artifacts/whisker_trials_analyzed.parquet", index=False)


if __name__ == "__main__":
    main()
