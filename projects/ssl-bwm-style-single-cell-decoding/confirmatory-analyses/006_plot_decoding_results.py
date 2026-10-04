"""Summary figures for the completed session-level decoding run
(`decoding_results.parquet`, 190 rows: 95 sessions x 2 targets, good+mua QC
scope). Each session is one imposter-null decode; this plots observed vs.
null accuracy per session, faceted by target x day_stage, plus a p-value
summary panel in the same style as `006_plot_single_cell_small_scale.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "decoding_results.parquet"

TARGET_TITLES = {"modality": "Modality (whisker vs auditory)", "response": "Response (lick vs no-lick)"}


def plot_observed_vs_null(df: pd.DataFrame) -> None:
    targets = ["modality", "response"]
    day_stages = ["learning", "expert"]
    fig, axes = plt.subplots(len(day_stages), len(targets), figsize=(6.5 * len(targets), 5.5 * len(day_stages)), squeeze=False)

    for row_i, day_stage in enumerate(day_stages):
        for col_i, target in enumerate(targets):
            ax = axes[row_i][col_i]
            sub = df[(df["day_stage"] == day_stage) & (df["target"] == target)]
            if sub.empty:
                ax.set_visible(False)
                continue
            sig = sub["p_value"] < 0.05
            colors = np.where(sig, "#d62728", "#7f7f7f")
            ax.scatter(sub["null_mean"], sub["observed_balanced_accuracy"], c=colors,
                       s=30 + 0.02 * sub["n_units"], alpha=0.85, edgecolor="black", linewidth=0.3)
            lims = [0.4, 1.0]
            ax.plot(lims, lims, color="black", linestyle=":", linewidth=1, label="chance (y=x)")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("imposter-null mean accuracy")
            ax.set_ylabel("observed accuracy")
            n_sig = int(sig.sum())
            ax.set_title(f"{TARGET_TITLES[target]}\n{day_stage}: {n_sig}/{len(sub)} sessions significant (p<0.05)", fontsize=10)
            ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle("Session-level decoding: observed vs. imposter-null accuracy\n"
                 "red = significant session; point size ~ n_units; each point = one session", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = HERE / "006_decoding_observed_vs_null.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_accuracy_distribution(df: pd.DataFrame) -> None:
    targets = ["modality", "response"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)

    for ax, target in zip(axes, targets):
        sub = df[df["target"] == target]
        for i, day_stage in enumerate(["learning", "expert"]):
            vals = sub[sub["day_stage"] == day_stage]["observed_balanced_accuracy"]
            sig = sub[sub["day_stage"] == day_stage]["p_value"] < 0.05
            x = i + rng.uniform(-0.12, 0.12, size=len(vals))
            colors = np.where(sig, "#d62728", "#7f7f7f")
            ax.scatter(x, vals, c=colors, s=30, alpha=0.85, edgecolor="none")
            ax.scatter([i], [vals.median()], marker="_", color="black", s=400, linewidth=2, zorder=5)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["learning", "expert"])
        ax.set_title(TARGET_TITLES[target], fontsize=11)
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel("observed nested-CV balanced accuracy")
    fig.suptitle("Session-level decoding accuracy by day-stage\nred = significant (p<0.05) vs imposter null; black dash = median", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = HERE / "006_decoding_accuracy_by_day_stage.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    df = pd.read_parquet(RESULTS_PATH)
    print(f"Loaded {len(df)} rows from {RESULTS_PATH}")
    plot_observed_vs_null(df)
    plot_accuracy_distribution(df)


if __name__ == "__main__":
    main()
