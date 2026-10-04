"""Area-level decoding summary figures, same two-level-significance design
as `003_area_aggregation_and_figures.py` (per-area binomial test of the
fraction of significant cells vs. the global rate, then area-level BH-FDR),
applied to `decoding_per_area_results.parquet`'s per-(session, area, target)
decodes instead of per-unit p-values -- each (session, area) decode's own
imposter-null p-value stands in for a "unit", so the "fraction significant"
per area is the fraction of that area's session-level decodes with
p_value < 0.05 (imposter null p, not FDR-corrected per decode -- there is no
natural per-decode multiple-comparisons family the way there is for
per-neuron tests, since each decode already IS the test unit here).

**Fixed 2026-08-19**: the first version of this script pooled learning and
expert sessions together when computing each area's global rate and
fraction-significant, inconsistent with `003_*` (which correctly separates
day_stage) and with this project's own day-stage-separation policy
(ssl_analysis_patterns.md). Now grouped by (parcellation, target, day_stage)
throughout, same as the single-cell aggregation. Note the expert arm has
only 16 sessions vs. learning's 79, so far fewer (session, area) cells will
clear MIN_CELLS_PER_AREA there -- expect sparser expert-arm area coverage,
not a bug.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ssl_bwm_stats_util import benjamini_hochberg

HERE = Path(__file__).resolve().parent
DECODING_PATH = HERE / "decoding_per_area_results.parquet"

MIN_CELLS_PER_AREA = 3  # minimum number of (session, area) decodes to trust an area's fraction
TARGET_TITLES = {"modality": "Decode: modality (whisker vs auditory)", "response": "Decode: response (lick vs no-lick)"}
PARCELLATIONS = ["area_acronym_custom", "area_group"]
MAX_AREAS_PLOTTED = 30


def build_area_table(decoding: pd.DataFrame) -> pd.DataFrame:
    decoding = decoding.dropna(subset=["p_value"])
    rows = []
    for (parcellation, target, day_stage), grp in decoding.groupby(["parcellation", "target", "day_stage"]):
        global_rate = (grp["p_value"] < 0.05).mean()
        area_rows = []
        for area, agrp in grp.groupby("area"):
            n_cells = len(agrp)
            if n_cells < MIN_CELLS_PER_AREA:
                continue
            n_sig = int((agrp["p_value"] < 0.05).sum())
            frac_sig = n_sig / n_cells
            mean_acc = agrp["observed_balanced_accuracy"].mean()
            bt = binomtest(n_sig, n_cells, p=max(global_rate, 1e-9), alternative="greater")
            area_rows.append({
                "parcellation": parcellation, "target": target, "day_stage": day_stage, "area": area,
                "n_cells": n_cells, "n_sig": n_sig, "fraction_sig": frac_sig,
                "mean_balanced_accuracy": mean_acc, "global_rate": global_rate, "binom_p": bt.pvalue,
            })
        if not area_rows:
            continue
        area_df = pd.DataFrame(area_rows)
        area_df["area_p_fdr"] = benjamini_hochberg(area_df["binom_p"].to_numpy())
        area_df["area_significant"] = area_df["area_p_fdr"] < 0.05
        rows.append(area_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_area_summary(area_table: pd.DataFrame, parcellation: str) -> None:
    targets = sorted(area_table["target"].unique())
    day_stages = sorted(area_table["day_stage"].unique())
    fig, axes = plt.subplots(len(day_stages), len(targets), figsize=(6.5 * len(targets), 5.5 * len(day_stages)), squeeze=False)

    for row_i, day_stage in enumerate(day_stages):
        for col_i, target in enumerate(targets):
            ax = axes[row_i][col_i]
            sub = area_table[(area_table["parcellation"] == parcellation) & (area_table["target"] == target)
                              & (area_table["day_stage"] == day_stage)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub = sub.sort_values("mean_balanced_accuracy", ascending=True).tail(MAX_AREAS_PLOTTED)
            y = np.arange(len(sub))
            colors = np.where(sub["area_significant"], "#d62728", "#7f7f7f")
            sizes = 20 + 5 * sub["n_cells"]
            ax.scatter(sub["mean_balanced_accuracy"], y, c=colors, s=sizes, edgecolor="black", linewidth=0.4, zorder=3)
            ax.axvline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
            ax.set_yticks(y)
            ax.set_yticklabels(sub["area"], fontsize=7)
            ax.set_xlabel("mean nested-CV balanced accuracy across sessions")
            ax.set_title(f"{TARGET_TITLES[target]}\n{day_stage}", fontsize=10)
            if col_i == 0:
                ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.suptitle(f"Area-level decoding summary -- parcellation: {parcellation}\n"
                 f"red = area significant (fraction of significant session-decodes vs global rate "
                 f"within that day-stage, binomial + area BH-FDR<0.05); point size ~ n_cells; "
                 f"min {MIN_CELLS_PER_AREA} session-decodes/area", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = HERE / f"005_decoding_area_summary_{parcellation}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    decoding = pd.read_parquet(DECODING_PATH)
    table = build_area_table(decoding)
    if table.empty:
        print("No area table produced -- check decoding_per_area_results.parquet contents / thresholds.")
        return

    for parcellation in PARCELLATIONS:
        if parcellation not in table["parcellation"].unique():
            continue
        plot_area_summary(table, parcellation)
        print(f"\n=== {parcellation}: significant areas per target x day_stage ===")
        for (target, day_stage), grp in table[table["parcellation"] == parcellation].groupby(["target", "day_stage"]):
            sig = grp[grp["area_significant"]].sort_values("mean_balanced_accuracy", ascending=False)
            print(f"{target} / {day_stage}: {len(sig)} significant areas (of {len(grp)} tested)")
            if len(sig):
                print(sig[["area", "n_cells", "fraction_sig", "mean_balanced_accuracy", "area_p_fdr"]].to_string(index=False))

    out_path = HERE / "decoding_area_significance.parquet"
    table.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
