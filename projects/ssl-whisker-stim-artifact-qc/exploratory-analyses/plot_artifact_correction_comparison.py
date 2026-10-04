"""Average whisker-trial neural activity around stimulus start_time, shown
two ways on the same axes:
1. No artifact correction: every detected spike counted as real, including
   inside the -10ms/+5ms dead zone (i.e. what you would see if you naively
   trusted the spike sorter's output through the artifact).
2. With artifact correction: real spikes inside the dead zone are excluded
   (as in the confirmatory analyses) and replaced, for display only, with a
   linearly-interpolated-lambda Poisson process (ssl_artifact_dead_zone.md's
   documented cosmetic procedure).

Both series are one population-average rate curve pooling ALL retained
whisker trials (passive_pre + passive_post together, all mice/areas/
cohorts) -- this project is about the artifact itself, not the cohort
science, so trial-type/epoch splits are deliberately collapsed here.

Reuses the fine (1ms) per-unit spike-count histograms already computed for
the ssl-whisker-auditory-cohort-modulation project
(reports/ssl_analysis/derived/psth/whisker_trial__passive_{pre,post}.npz):
those histograms count every detected spike in every 1ms bin regardless of
the dead zone (the exclusion happens only at the rate-conversion step), so
the "no correction" curve does not require re-reading spike shards.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PSTH_DIR = Path("reports/ssl_analysis/derived/psth")
FIG_DIR = Path("projects/ssl-whisker-stim-artifact-qc/report/figures")

OUTPUT_BIN_S = 0.010
STRIDE_S = 0.002
FINE_BIN_S = 0.001
N_PER_WINDOW = round(OUTPUT_BIN_S / FINE_BIN_S)
STRIDE_FINE = round(STRIDE_S / FINE_BIN_S)
DEAD_ZONE = (-0.010, 0.005)
RNG_SEED = 20260814


def load_and_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool passive_pre + passive_post whisker-trial fine histograms across
    all retained units into one grand total, weighted by trial count."""
    fine_counts_total = None
    n_trials_total = 0.0
    bin_centers = None
    for epoch in ("passive_pre", "passive_post"):
        npz = np.load(PSTH_DIR / f"whisker_trial__{epoch}.npz")
        fc = npz["fine_counts"]
        nt = npz["n_trials"]
        bc = npz["bin_centers"]
        summed = fc.sum(axis=0)
        fine_counts_total = summed if fine_counts_total is None else fine_counts_total + summed
        n_trials_total += nt.sum()
        bin_centers = bc
    return fine_counts_total, n_trials_total, bin_centers, np.full_like(bin_centers, FINE_BIN_S)


def sliding_rate(fine_counts_sum: np.ndarray, n_trials_total: float, valid_seconds_per_bin: np.ndarray, bin_centers: np.ndarray):
    n_fine = len(bin_centers)
    starts = np.arange(0, n_fine - N_PER_WINDOW + 1, STRIDE_FINE)
    counts_cs = np.concatenate([[0], np.cumsum(fine_counts_sum)])
    valid_cs = np.concatenate([[0], np.cumsum(valid_seconds_per_bin)])
    win_counts = counts_cs[starts + N_PER_WINDOW] - counts_cs[starts]
    win_valid = (valid_cs[starts + N_PER_WINDOW] - valid_cs[starts]) * n_trials_total
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(win_valid > 0, win_counts / win_valid, np.nan)
    centers = np.array([bin_centers[s:s + N_PER_WINDOW].mean() for s in starts])
    return centers, rate


def poisson_fill(centers: np.ndarray, rate: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    filled = rate.copy()
    nan_mask = np.isnan(rate)
    if not nan_mask.any():
        return filled
    valid_idx = np.where(~nan_mask)[0]
    lam_interp = np.interp(np.arange(len(rate)), valid_idx, rate[valid_idx])
    for i in np.where(nan_mask)[0]:
        filled[i] = rng.poisson(max(lam_interp[i], 0.1))
    return filled


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fine_counts, n_trials_total, bin_centers, ones_valid = load_and_pool()
    print(f"Pooled whisker trials (passive_pre + passive_post): {n_trials_total:.0f}")

    # Native 1ms resolution throughout this figure -- the 10ms/2ms sliding
    # window used for the confirmatory PSTHs smooths out the artifact's own
    # fine structure (verified: a 10ms-smoothed version of this curve shows
    # no dead-zone anomaly at all, mean 4.83 vs 5.09 Hz baseline, because the
    # true anomaly is only ~1-4ms wide and gets averaged away).
    rate_raw = fine_counts / (n_trials_total * FINE_BIN_S)
    centers_raw = bin_centers

    valid_seconds_corrected = np.full_like(bin_centers, FINE_BIN_S)
    dz_start, dz_end = DEAD_ZONE
    overlap_start = np.maximum(bin_centers - FINE_BIN_S / 2, dz_start)
    overlap_end = np.minimum(bin_centers + FINE_BIN_S / 2, dz_end)
    overlap = np.clip(overlap_end - overlap_start, 0, None)
    valid_seconds_corrected = np.clip(valid_seconds_corrected - overlap, 0, None)
    # floating-point subtraction at the dead-zone boundary can leave a tiny
    # non-zero residual (e.g. 1e-19) instead of exact 0; dividing by that
    # explodes to ~1e13 Hz and silently wrecks the plot scale. Require at
    # least half a fine bin of real valid duration, not just "> 0".
    min_valid_s = FINE_BIN_S * 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_corr = np.where(valid_seconds_corrected > min_valid_s, fine_counts / (n_trials_total * valid_seconds_corrected), np.nan)
    rate_corr_filled = poisson_fill(bin_centers, rate_corr, seed=RNG_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, xlim, title in ((axes[0], (-50, 100), "Wide view"), (axes[1], (-15, 15), "Zoomed on the dead zone")):
        ax.axvspan(DEAD_ZONE[0] * 1000, DEAD_ZONE[1] * 1000, color="grey", alpha=0.2, label="documented dead zone (-10/+5ms)")
        ax.plot(centers_raw * 1000, rate_raw, color="tab:red", linewidth=1.2, marker="o" if title.startswith("Zoomed") else None, markersize=3, label="no artifact correction (raw)")
        ax.plot(bin_centers * 1000, rate_corr_filled, color="tab:blue", linewidth=1.2, label="with artifact correction\n(dead-zone excluded, Poisson-filled)")
        ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
        ax.set_xlim(*xlim)
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("population average firing rate (Hz)")
        ax.set_title(title)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Whisker-trial average neural activity, native 1ms resolution: raw vs artifact-corrected\n(n={n_trials_total:.0f} trials pooled, passive_pre+post, all retained units)")
    fig.tight_layout()
    out_path = FIG_DIR / "artifact_correction_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # numeric summary of the raw artifact signature at native resolution
    baseline_mask = (centers_raw >= -0.030) & (centers_raw < -0.003)
    baseline_rate = np.nanmean(rate_raw[baseline_mask])
    print(f"\nStable pre-stimulus baseline (-30 to -3ms): {baseline_rate:.2f} Hz")
    dz_mask = (centers_raw >= DEAD_ZONE[0]) & (centers_raw <= DEAD_ZONE[1])
    for c, r in zip(centers_raw[dz_mask], rate_raw[dz_mask]):
        flag = "  <-- >2x baseline" if r > 2 * baseline_rate else ("  <-- <0.5x baseline" if r < 0.5 * baseline_rate else "")
        print(f"  {c*1000:+6.1f}ms  {r:7.2f} Hz{flag}")


if __name__ == "__main__":
    main()
