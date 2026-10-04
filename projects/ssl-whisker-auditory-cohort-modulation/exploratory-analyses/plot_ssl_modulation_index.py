"""Modulation-index distribution figures:
1. Global MI distribution by cohort (violin + per-mouse median points), with
   the mouse-block permutation test result annotated.
2. MI distribution by area_group x cohort in a single figure.
"""

from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED_DIR = Path("reports/ssl_analysis/derived")
RESULTS_DIR = Path("reports/ssl_analysis/results")
FIG_DIR = Path("reports/ssl_analysis/figures")


def plot_global(df: pd.DataFrame) -> None:
    with open(RESULTS_DIR / "modulation_index_results.json") as f:
        results = json.load(f)
    test = results["cohort_permutation_test"]

    fig, ax = plt.subplots(figsize=(6, 5))
    groups = ["R+", "R-"]
    data_by_group = [df.loc[df["reward_group"] == g, "modulation_index"].to_numpy() for g in groups]
    parts = ax.violinplot(data_by_group, showmedians=True)
    for pc, color in zip(parts["bodies"], ["tab:red", "tab:blue"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.4)

    per_mouse = df.groupby(["mouse_id", "reward_group"])["modulation_index"].median().reset_index()
    for i, g in enumerate(groups, start=1):
        pts = per_mouse.loc[per_mouse["reward_group"] == g, "modulation_index"]
        jitter = np.random.default_rng(20260814).uniform(-0.06, 0.06, size=len(pts))
        ax.scatter(np.full(len(pts), i) + jitter, pts, color="black", s=14, alpha=0.7, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("modulation index (whisker vs auditory |delta|)")
    ax.set_title(
        f"Global MI by cohort (points = per-mouse median)\n"
        f"mouse-block permutation test: diff(median)={test['observed_diff_median']:.3f}, p={test['p_value']:.3f}"
    )
    fig.tight_layout()
    out_path = FIG_DIR / "modulation_index_global.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_by_area(df: pd.DataFrame) -> None:
    counts = df[df["area_group"].notna()].groupby("area_group").size().sort_values(ascending=False)
    areas = [a for a in counts.index if counts[a] >= 200]

    fig, ax = plt.subplots(figsize=(max(10, len(areas) * 1.1), 5.5))
    positions = []
    labels = []
    pos = 0
    for area in areas:
        for g, color in (("R+", "tab:red"), ("R-", "tab:blue")):
            vals = df.loc[(df["area_group"] == area) & (df["reward_group"] == g), "modulation_index"]
            if len(vals) < 5:
                pos += 1
                continue
            bp = ax.boxplot(vals, positions=[pos], widths=0.7, patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.5)
            positions.append(pos)
            pos += 1
        pos += 0.6
        labels.append(area)

    tick_positions = [i * 2.6 + 0.5 for i in range(len(areas))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("modulation index")
    ax.set_title("Modulation index by area_group x cohort (red=R+, blue=R-)")
    fig.tight_layout()
    out_path = FIG_DIR / "modulation_index_by_area.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DERIVED_DIR / "modulation_index.parquet")
    plot_global(df)
    plot_by_area(df)


if __name__ == "__main__":
    main()
