"""Effect plots (barplot + per-mouse points) for each analysis window, and
the stim_type x cohort interaction figure for the global model. Uses
per-mouse mean deltas (the same aggregation level the LMM's random effect
respects) as the overlaid points, and the grand mean across mice per cell
as the bar height -- descriptive, not re-deriving model-predicted marginal
means (documented as a simplification).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED_DIR = Path("reports/ssl_analysis/derived")
FIG_DIR = Path("reports/ssl_analysis/figures")

OUTCOMES = [("delta_baseline", "Baseline (-1s to 0)"), ("delta_evoked", "Evoked (5-45ms)"), ("delta_corrected", "Evoked, baseline-corrected")]


def per_mouse_means(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return df.groupby(["mouse_id", "stim_type", "reward_group"])[outcome].mean().reset_index()


def plot_effects_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, (outcome, title) in zip(axes, OUTCOMES):
        pm = per_mouse_means(df, outcome)
        cells = pm.groupby(["stim_type", "reward_group"])[outcome]
        means = cells.mean()
        sems = cells.sem()

        combos = [("whisker", "R+"), ("whisker", "R-"), ("auditory", "R+"), ("auditory", "R-")]
        x = np.arange(len(combos))
        bar_means = [means.get(c, np.nan) for c in combos]
        bar_sems = [sems.get(c, np.nan) for c in combos]
        colors = ["tab:blue", "tab:red", "tab:blue", "tab:red"]
        ax.bar(x, bar_means, yerr=bar_sems, color=colors, alpha=0.5, capsize=4)

        for i, combo in enumerate(combos):
            pts = pm[(pm["stim_type"] == combo[0]) & (pm["reward_group"] == combo[1])][outcome]
            jitter = np.random.default_rng(20260814 + i).uniform(-0.12, 0.12, size=len(pts))
            ax.scatter(np.full(len(pts), x[i]) + jitter, pts, color="black", s=10, alpha=0.6, zorder=3)

        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n{r}" for s, r in combos])
        ax.set_title(title)
        ax.set_ylabel("post - pre firing rate (Hz), per-mouse mean")

    fig.suptitle("Per-mouse post-minus-pre effects by stim type x cohort (bars = mean +/- SEM across mice, points = per-mouse means)")
    fig.tight_layout()
    out_path = FIG_DIR / "lmm_effects_barplots.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_interaction(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    outcome = "delta_evoked"
    pm = per_mouse_means(df, outcome)
    cells = pm.groupby(["stim_type", "reward_group"])[outcome]
    means = cells.mean()
    sems = cells.sem()

    for rg, color in (("R+", "tab:red"), ("R-", "tab:blue")):
        y = [means.get(("whisker", rg), np.nan), means.get(("auditory", rg), np.nan)]
        yerr = [sems.get(("whisker", rg), np.nan), sems.get(("auditory", rg), np.nan)]
        ax.errorbar([0, 1], y, yerr=yerr, marker="o", color=color, label=rg, capsize=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["whisker", "auditory"])
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("post - pre evoked firing rate (Hz)\nper-mouse mean +/- SEM")
    ax.set_title("Global model: stim_type x reward_group interaction")
    ax.legend(title="reward_group")
    fig.tight_layout()
    out_path = FIG_DIR / "lmm_interaction_global.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DERIVED_DIR / "lmm_table.parquet")
    plot_effects_grid(df)
    plot_interaction(df)


if __name__ == "__main__":
    main()
