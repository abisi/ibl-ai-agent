"""Area-level aggregation and BWM-style "which area significantly encodes
this variable" summary figures, at BOTH area parcellation levels available
in `reports/ssl_analysis/derived/unit_area_labels.parquet`:
  - `area_acronym_custom`: fine-grained Allen custom labels (allen_utils).
  - `area_group`: coarse Allen custom large-groups.

Two-level significance, mirroring how BWM-style papers report region-level
significance from a per-neuron test:
  1. Unit-level: **raw (uncorrected) p<0.05** per unit is the "candidate"
     flag feeding into the area aggregation -- see "Why raw, not FDR" below.
  2. Area-level: for each area (with >= MIN_UNITS_PER_AREA units in that
     test x day_stage x parcellation cell), a one-sided exact binomial test
     of "this area's fraction of nominally-significant units exceeds the
     test x day_stage global fraction", then a BH-FDR pass across areas
     within that (test, day_stage, parcellation) family. An area is called
     "significant" at area_p_fdr < 0.05 -- this area-level FDR is the real
     multiple-comparisons control in this design.

**Why raw, not FDR-corrected, p-values feed the area step (fixed 2026-08-19,
user decision)**: at the locked `nShuf=1000` for the per-unit shuffle test,
the smallest achievable p-value is `1/(1+1000)=0.001`. Unit-level BH-FDR
across the ~2,400-14,000 units tested per (test, day_stage) family requires
even the single most-significant unit to reach p<=~3.6e-6 (learning) or
p<=~2.1e-5 (expert) to survive -- both far below the 0.001 floor, so **zero**
units can ever survive unit-level FDR at this nShuf/cohort-size combination,
regardless of true effect size (verified: 0/49,278 units in the actual run).
This mirrors BWM's own single_cell_stats example scripts, which also report
only raw per-neuron p-values with no unit-level correction -- the rigorous
correction belongs at the area level, where `m` is only ~15-40 (well above
what a 0.001 p-value floor can support) and the binomial test itself isn't
permutation-floor-limited at all (exact test on counts). `p_fdr` from
`001_run_all_single_cell_tests.py` is still loaded/available in the merged
table for transparency, just not used as the area-step's "significant" flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ssl_bwm_stats_util import benjamini_hochberg

HERE = Path(__file__).resolve().parent
SINGLE_CELL_PATH = HERE / "single_cell_results.parquet"
AREA_LABELS_PATH = Path("reports/ssl_analysis/derived/unit_area_labels.parquet")

MIN_UNITS_PER_AREA = 5  # matches bwm_paper_repo.md's own "min 5 units per region" convention
TEST_ORDER = ["modality", "response", "prior_outcome"]
TEST_TITLES = {"modality": "Modality (whisker vs auditory)", "response": "Response (lick vs no-lick)",
               "prior_outcome": "Prior-outcome (t-1 rewarded vs not)"}
PARCELLATIONS = ["area_acronym_custom", "area_group"]
MAX_AREAS_PLOTTED = 30  # for the fine-grained parcellation, keep figures legible


def build_area_table(results: pd.DataFrame, area_labels: pd.DataFrame, parcellation: str) -> pd.DataFrame:
    df = results.merge(
        area_labels[["session_id", "cluster_id", parcellation]].drop_duplicates(),
        on=["session_id", "cluster_id"], how="left",
    )
    df = df.dropna(subset=[parcellation])

    rows = []
    for (test_name, day_stage), grp in df.groupby(["test", "day_stage"]):
        global_rate = (grp["p_raw"] < 0.05).mean()
        area_rows = []
        for area, agrp in grp.groupby(parcellation):
            n_units = len(agrp)
            if n_units < MIN_UNITS_PER_AREA:
                continue
            n_sig = int((agrp["p_raw"] < 0.05).sum())
            frac_sig = n_sig / n_units
            bt = binomtest(n_sig, n_units, p=max(global_rate, 1e-9), alternative="greater")
            area_rows.append({
                "test": test_name, "day_stage": day_stage, "parcellation": parcellation,
                "area": area, "n_units": n_units, "n_raw_sig": n_sig, "fraction_sig": frac_sig,
                "global_rate": global_rate, "binom_p": bt.pvalue,
            })
        if not area_rows:
            continue
        area_df = pd.DataFrame(area_rows)
        area_df["area_p_fdr"] = benjamini_hochberg(area_df["binom_p"].to_numpy())
        area_df["area_significant"] = area_df["area_p_fdr"] < 0.05
        rows.append(area_df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_area_summary(area_table: pd.DataFrame, parcellation: str) -> None:
    day_stages = sorted(area_table["day_stage"].unique())
    fig, axes = plt.subplots(len(day_stages), len(TEST_ORDER), figsize=(5.2 * len(TEST_ORDER), 4.5 * len(day_stages)), squeeze=False)

    for row_i, day_stage in enumerate(day_stages):
        for col_i, test_name in enumerate(TEST_ORDER):
            ax = axes[row_i][col_i]
            sub = area_table[(area_table["day_stage"] == day_stage) & (area_table["test"] == test_name)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub = sub.sort_values("fraction_sig", ascending=True).tail(MAX_AREAS_PLOTTED)
            y = np.arange(len(sub))
            colors = np.where(sub["area_significant"], "#d62728", "#7f7f7f")
            sizes = 20 + 4 * np.sqrt(sub["n_units"])
            ax.scatter(sub["fraction_sig"], y, c=colors, s=sizes, edgecolor="black", linewidth=0.4, zorder=3)
            global_rate = sub["global_rate"].iloc[0]
            ax.axvline(global_rate, color="gray", linestyle="--", linewidth=1, label=f"global rate={global_rate:.2f}")
            ax.set_yticks(y)
            ax.set_yticklabels(sub["area"], fontsize=7)
            ax.set_xlabel("fraction of units with raw p<0.05")
            ax.set_xlim(-0.02, max(0.5, sub["fraction_sig"].max() * 1.15))
            ax.set_title(f"{TEST_TITLES[test_name]}\n{day_stage}", fontsize=9)
            if col_i == 0:
                ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.suptitle(f"Area-level single-cell encoding summary -- parcellation: {parcellation}\n"
                 f"x-axis = fraction of units with raw (uncorrected) p<0.05; "
                 f"red = area significant (binomial vs global rate, area-level BH-FDR<0.05 -- "
                 f"the real multiple-comparisons control here, see script docstring); "
                 f"point size ~ sqrt(n_units); min {MIN_UNITS_PER_AREA} units/area", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = HERE / f"003_area_summary_{parcellation}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    results = pd.read_parquet(SINGLE_CELL_PATH)
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    all_tables = []
    for parcellation in PARCELLATIONS:
        table = build_area_table(results, area_labels, parcellation)
        if table.empty:
            print(f"No area table produced for {parcellation} (check unit-count threshold / join coverage)")
            continue
        all_tables.append(table)
        plot_area_summary(table, parcellation)

        print(f"\n=== {parcellation}: significant areas per test x day_stage ===")
        for (test_name, day_stage), grp in table.groupby(["test", "day_stage"]):
            sig = grp[grp["area_significant"]].sort_values("fraction_sig", ascending=False)
            print(f"{test_name} / {day_stage}: {len(sig)} significant areas "
                  f"(of {len(grp)} tested, min {MIN_UNITS_PER_AREA} units/area)")
            if len(sig):
                print(sig[["area", "n_units", "n_raw_sig", "fraction_sig", "area_p_fdr"]].to_string(index=False))

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        out_path = HERE / "area_significance.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
