"""Scatter plots of whisker vs auditory post-minus-pre delta (evoked,
baseline-corrected), global and per area_group, at both single-neuron level
(density, given ~100k points) and mouse level (per-mouse means, N~65)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED_DIR = Path("reports/ssl_analysis/derived")
FIG_DIR = Path("reports/ssl_analysis/figures")
LIMIT = 60  # symmetric axis limit (Hz) for readability; a handful of outliers get clipped by the axis, not the data


def wide_deltas() -> pd.DataFrame:
    lmm_table = pd.read_parquet(DERIVED_DIR / "lmm_table.parquet")
    idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group"]
    wide = lmm_table.pivot_table(index=idx_cols, columns="stim_type", values="delta_corrected").reset_index()
    return wide.dropna(subset=["whisker", "auditory"])


def add_ref_lines(ax) -> None:
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    lims = (-LIMIT, LIMIT)
    ax.plot(lims, lims, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def plot_global(wide: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    hb = ax.hexbin(wide["auditory"], wide["whisker"], gridsize=80, cmap="viridis", mincnt=1, extent=(-LIMIT, LIMIT, -LIMIT, LIMIT))
    fig.colorbar(hb, ax=ax, label="n units")
    add_ref_lines(ax)
    ax.set_xlabel("auditory delta (Hz)")
    ax.set_ylabel("whisker delta (Hz)")
    ax.set_title(f"Single-neuron level (n={len(wide)})")

    ax = axes[1]
    per_mouse = wide.groupby(["mouse_id", "reward_group"])[["whisker", "auditory"]].mean().reset_index()
    for rg, color in (("R+", "tab:red"), ("R-", "tab:blue")):
        sub = per_mouse[per_mouse["reward_group"] == rg]
        ax.scatter(sub["auditory"], sub["whisker"], color=color, label=rg, s=25, alpha=0.8)
    add_ref_lines(ax)
    ax.set_xlabel("auditory delta (Hz)")
    ax.set_ylabel("whisker delta (Hz)")
    ax.set_title(f"Mouse level (n={len(per_mouse)})")
    ax.legend(title="reward_group")

    fig.suptitle("Whisker vs auditory evoked, baseline-corrected delta (post-pre) -- global")
    fig.tight_layout()
    out_path = FIG_DIR / "scatter_global.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_per_area(wide: pd.DataFrame) -> None:
    counts = wide[wide["area_group"].notna()].groupby("area_group").size().sort_values(ascending=False)
    areas = [a for a in counts.index if counts[a] >= 200]

    n = len(areas)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for i, area in enumerate(areas):
        ax = axes[i // ncols, i % ncols]
        sub = wide[wide["area_group"] == area]
        ax.hexbin(sub["auditory"], sub["whisker"], gridsize=50, cmap="viridis", mincnt=1, extent=(-LIMIT, LIMIT, -LIMIT, LIMIT))
        add_ref_lines(ax)
        ax.set_title(f"{area} (n={len(sub)})", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("whisker delta (Hz)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("auditory delta (Hz)")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle("Whisker vs auditory evoked, baseline-corrected delta -- per area (single-neuron level, density)")
    fig.tight_layout()
    out_path = FIG_DIR / "scatter_per_area.png"
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    wide = wide_deltas()
    plot_global(wide)
    plot_per_area(wide)


if __name__ == "__main__":
    main()
