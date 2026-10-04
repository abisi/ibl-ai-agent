"""Per-session stability of the raw (uncorrected) whisker-trial artifact
signature: is the dropout-then-burst pattern at native 1ms resolution
consistent across sessions/probes, or does its timing/presence vary?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PSTH_DIR = Path("reports/ssl_analysis/derived/psth")
FIG_DIR = Path("projects/ssl-whisker-stim-artifact-qc/report/figures")
FINE_BIN_S = 0.001


def load_pooled_per_session():
    fine_counts_total = None
    index_total = None
    for epoch in ("passive_pre", "passive_post"):
        npz = np.load(PSTH_DIR / f"whisker_trial__{epoch}.npz")
        idx = pd.read_parquet(PSTH_DIR / f"whisker_trial__{epoch}_index.parquet")
        fc = npz["fine_counts"]
        bc = npz["bin_centers"]
        if fine_counts_total is None:
            fine_counts_total = fc
            index_total = idx.copy()
        else:
            fine_counts_total = np.concatenate([fine_counts_total, fc], axis=0)
            index_total = pd.concat([index_total, idx], ignore_index=True)
    return fine_counts_total, index_total, bc


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fine_counts, index, bin_centers = load_pooled_per_session()

    dz_mask = (bin_centers >= -0.010) & (bin_centers <= 0.015)
    dz_centers = bin_centers[dz_mask]

    sessions = sorted(index["session_id"].unique())
    rows = []
    for session_id in sessions:
        mask = (index["session_id"] == session_id).to_numpy()
        n_trials = index.loc[mask, "n_trials"].sum()
        if n_trials == 0:
            continue
        summed = fine_counts[mask][:, dz_mask].sum(axis=0)
        rate = summed / (n_trials * FINE_BIN_S)
        baseline = rate[dz_centers < -0.005].mean()
        dropout_idx = np.argmin(rate)
        burst_idx = np.argmax(rate)
        rows.append({
            "session_id": session_id, "n_trials": n_trials, "baseline_hz": baseline,
            "dropout_time_ms": dz_centers[dropout_idx] * 1000, "dropout_rate_hz": rate[dropout_idx],
            "burst_time_ms": dz_centers[burst_idx] * 1000, "burst_rate_hz": rate[burst_idx],
        })

    df = pd.DataFrame(rows)
    df.to_csv("projects/ssl-whisker-stim-artifact-qc/artifacts/per_session_artifact_timing.csv", index=False)
    print(f"{len(df)} sessions with whisker trials in the analyzed set")
    print(df[["baseline_hz", "dropout_time_ms", "dropout_rate_hz", "burst_time_ms", "burst_rate_hz"]].describe())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["dropout_time_ms"], bins=np.arange(-10.5, 15.5, 1), color="tab:red", alpha=0.7)
    axes[0].set_xlabel("time of minimum rate (ms from start_time)")
    axes[0].set_title("Dropout timing across sessions")
    axes[1].hist(df["burst_time_ms"], bins=np.arange(-10.5, 15.5, 1), color="tab:orange", alpha=0.7)
    axes[1].set_xlabel("time of maximum rate (ms from start_time)")
    axes[1].set_title("Burst timing across sessions")
    for ax in axes:
        ax.axvspan(-10, 5, color="grey", alpha=0.15, label="ssl_artifact_dead_zone.md (-10/+5ms)")
        ax.axvspan(-1, 4, color="tab:red", alpha=0.15, label="actual correction window (-1/+4ms)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Per-session artifact timing stability (n={len(df)} sessions, raw/uncorrected)")
    fig.tight_layout()
    out_path = FIG_DIR / "artifact_timing_stability_by_session.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    for label, lo, hi in (("documented ssl_artifact_dead_zone.md window (-10/+5ms)", -10, 5),
                          ("actual correction window (-1/+4ms)", -1, 4)):
        n_dropout_outside = ((df["dropout_time_ms"] < lo) | (df["dropout_time_ms"] > hi)).sum()
        n_burst_outside = ((df["burst_time_ms"] < lo) | (df["burst_time_ms"] > hi)).sum()
        print(f"\nAgainst {label}:")
        print(f"  Sessions with dropout minimum outside: {n_dropout_outside}/{len(df)}")
        print(f"  Sessions with burst maximum outside: {n_burst_outside}/{len(df)}")


if __name__ == "__main__":
    main()
