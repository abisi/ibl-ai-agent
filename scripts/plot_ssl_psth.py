"""PSTH figures from the fine (1ms) histograms in reports/ssl_analysis/derived/psth/:
1. Main cohort-level grid: rows=reward_group, cols=[whisker, auditory, difference],
   passive_pre vs passive_post overlaid, pooled across all retained units.
2. Supplementary per-area figure: one line per mouse, faceted by area_group x
   [whisker, auditory, difference], colored by reward_group.

10ms bins / 2ms stride, built from the 1ms fine histograms via a moving sum.
Whisker-trial panels: real data excludes the -10ms/+5ms dead zone (duration
goes to 0 there -> gap in the rate curve); a cosmetic Poisson-process fill
(reports/ssl_analysis/derived/psth dead-zone bins) is drawn as a shaded,
labeled band, per ssl_artifact_dead_zone.md -- for display only, never used
in any of the earlier quantitative computations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PSTH_DIR = Path("reports/ssl_analysis/derived/psth")
FIG_DIR = Path("reports/ssl_analysis/figures")

OUTPUT_BIN_S = 0.010
STRIDE_S = 0.002
FINE_BIN_S = 0.001
N_PER_WINDOW = round(OUTPUT_BIN_S / FINE_BIN_S)
STRIDE_FINE = round(STRIDE_S / FINE_BIN_S)
DEAD_ZONE = (-0.010, 0.005)
RNG_SEED = 20260814


def load_condition(trial_type: str, passive_epoch: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    npz = np.load(PSTH_DIR / f"{trial_type}__{passive_epoch}.npz")
    index = pd.read_parquet(PSTH_DIR / f"{trial_type}__{passive_epoch}_index.parquet")
    return npz["fine_counts"], index, npz["valid_seconds_per_bin"], npz["bin_centers"]


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


def group_rate(fine_counts: np.ndarray, index: pd.DataFrame, mask: np.ndarray, valid_seconds_per_bin: np.ndarray, bin_centers: np.ndarray):
    if mask.sum() == 0:
        return None, None
    summed = fine_counts[mask].sum(axis=0)
    n_trials_total = index.loc[mask, "n_trials"].sum()
    return sliding_rate(summed, n_trials_total, valid_seconds_per_bin, bin_centers)


def poisson_fill_for_display(centers: np.ndarray, rate: np.ndarray, seed: int) -> np.ndarray:
    """Fill NaN gaps (dead-zone output bins) with a Poisson-consistent smooth
    interpolation for display only: lambda is linearly interpolated between
    the flanking real rates (not nearest-neighbor, which would plateau at
    whichever side has the higher rate and could look like a spurious sharp
    peak -- misleading given the real onset transient right after the gap).
    Purely cosmetic; never used in any statistic."""
    rng = np.random.default_rng(seed)
    filled = rate.copy()
    nan_mask = np.isnan(rate)
    if not nan_mask.any():
        return filled
    valid_idx = np.where(~nan_mask)[0]
    if valid_idx.size == 0:
        return filled
    lam_interp = np.interp(np.arange(len(rate)), valid_idx, rate[valid_idx])
    for i in np.where(nan_mask)[0]:
        lam = max(lam_interp[i], 0.1)
        filled[i] = rng.poisson(lam)
    return filled


def plot_area_grid(data: dict) -> None:
    """Supplementary figure: rows = area_group, cols = [whisker, auditory,
    difference], pre/post mean curves per area (thin grey lines = individual
    mice, per the "each mouse, each area" request -- a full per-mouse x
    per-area facet grid (65 x ~12) is impractical as one static figure, so
    mice are shown as overlaid individual curves within each area panel
    rather than as separate panels)."""
    _, index0, _, _ = data[("whisker_trial", "passive_pre")]
    area_groups = [a for a in index0["area_group"].dropna().unique()]
    area_counts = index0[index0["area_group"].notna()].groupby("area_group").size().sort_values(ascending=False)
    area_groups = [a for a in area_counts.index if area_counts[a] >= 200]

    fig, axes = plt.subplots(len(area_groups), 3, figsize=(15, 3 * len(area_groups)), sharex=True)
    col_specs = [("whisker_trial", "whisker"), ("auditory_trial", "auditory"), (None, "difference")]

    for row, area in enumerate(area_groups):
        for col, (trial_type, label) in enumerate(col_specs):
            ax = axes[row, col]
            for epoch, color in (("passive_pre", "tab:blue"), ("passive_post", "tab:orange")):
                if trial_type is not None:
                    fine_counts, index, valid_s, centers_fine = data[(trial_type, epoch)]
                    mice = index.loc[index["area_group"] == area, "mouse_id"].unique()
                    for mouse in mice:
                        m_mask = ((index["area_group"] == area) & (index["mouse_id"] == mouse)).to_numpy()
                        c, r = group_rate(fine_counts, index, m_mask, valid_s, centers_fine)
                        if r is not None:
                            ax.plot(c * 1000, r, color=color, alpha=0.10, linewidth=0.6)
                    mask = (index["area_group"] == area).to_numpy()
                    centers, rate = group_rate(fine_counts, index, mask, valid_s, centers_fine)
                    is_whisker = trial_type == "whisker_trial"
                else:
                    fc_w, idx_w, vs_w, cf_w = data[("whisker_trial", epoch)]
                    fc_a, idx_a, vs_a, cf_a = data[("auditory_trial", epoch)]
                    mask_w = (idx_w["area_group"] == area).to_numpy()
                    mask_a = (idx_a["area_group"] == area).to_numpy()
                    c_w, r_w = group_rate(fc_w, idx_w, mask_w, vs_w, cf_w)
                    c_a, r_a = group_rate(fc_a, idx_a, mask_a, vs_a, cf_a)
                    centers, rate = c_w, r_w - r_a
                    is_whisker = False
                if rate is None:
                    continue
                plot_rate = poisson_fill_for_display(centers, rate, seed=RNG_SEED) if is_whisker else rate
                if is_whisker:
                    ax.axvspan(DEAD_ZONE[0], DEAD_ZONE[1], color="grey", alpha=0.2, label="_nolegend_")
                ax.plot(centers * 1000, plot_rate, color=color, linewidth=1.3, label=epoch)
            ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{area}\nHz")
            if row == len(area_groups) - 1:
                ax.set_xlabel("time from start_time (ms)")

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-area PSTHs (thin lines = individual mice, thick = area mean; areas with >=200 retained units)")
    fig.tight_layout()
    out_path = FIG_DIR / "psth_area_grid.png"
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for trial_type in ("whisker_trial", "auditory_trial"):
        for passive_epoch in ("passive_pre", "passive_post"):
            data[(trial_type, passive_epoch)] = load_condition(trial_type, passive_epoch)

    reward_groups = ["R+", "R-"]
    col_specs = [("whisker_trial", "whisker"), ("auditory_trial", "auditory"), (None, "difference")]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for row, rg in enumerate(reward_groups):
        for col, (trial_type, label) in enumerate(col_specs):
            ax = axes[row, col]
            for epoch, color, ls in (("passive_pre", "tab:blue", "-"), ("passive_post", "tab:orange", "-")):
                if trial_type is not None:
                    fine_counts, index, valid_s, centers_fine = data[(trial_type, epoch)]
                    mask = (index["reward_group"] == rg).to_numpy()
                    centers, rate = group_rate(fine_counts, index, mask, valid_s, centers_fine)
                    is_whisker = trial_type == "whisker_trial"
                else:
                    fc_w, idx_w, vs_w, cf_w = data[("whisker_trial", epoch)]
                    fc_a, idx_a, vs_a, cf_a = data[("auditory_trial", epoch)]
                    mask_w = (idx_w["reward_group"] == rg).to_numpy()
                    mask_a = (idx_a["reward_group"] == rg).to_numpy()
                    c_w, r_w = group_rate(fc_w, idx_w, mask_w, vs_w, cf_w)
                    c_a, r_a = group_rate(fc_a, idx_a, mask_a, vs_a, cf_a)
                    centers, rate = c_w, r_w - r_a
                    is_whisker = False  # difference curve: don't re-apply cosmetic fill logic on subtraction artifacts

                if rate is None:
                    continue
                plot_rate = rate.copy()
                if is_whisker:
                    dz_mask = (centers > DEAD_ZONE[0] - OUTPUT_BIN_S / 2) & (centers < DEAD_ZONE[1] + OUTPUT_BIN_S / 2) & np.isnan(rate)
                    plot_rate = poisson_fill_for_display(centers, rate, seed=RNG_SEED)
                    ax.axvspan(DEAD_ZONE[0], DEAD_ZONE[1], color="grey", alpha=0.25, label="_nolegend_")
                ax.plot(centers * 1000, plot_rate, color=color, linestyle=ls, label=epoch, linewidth=1.3)

            ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{rg}\nfiring rate (Hz)")
            if row == 1:
                ax.set_xlabel("time from start_time (ms)")

    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("SSL passive-epoch PSTHs by cohort (grey band = whisker artifact dead zone,\nsynthetic/cosmetic fill only, not used in any statistic)")
    fig.tight_layout()
    out_path = FIG_DIR / "psth_cohort_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    plot_area_grid(data)


if __name__ == "__main__":
    main()
