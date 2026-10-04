"""Per Axel's 2026-08-31 follow-up: (1) drop significance-dot styling,
uniform dots for all points; (2) share the y-axis across the 3 variant
columns within each area-pair row (still with visible tick labels on every
panel); (3) new lag-0 summary heatmaps (pairs x conditions, one panel per
variant). Pure replot from the already-saved variant_A/B/C/results.csv --
no new compute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
VARIANT_DIRS = {v: ARTIFACTS_DIR / f"variant_{v}" for v in ["A", "B", "C"]}
AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1)]
N_DIMS_MAX = 20


def square(ax):
    ax.set_box_aspect(1)


def short(area: str) -> str:
    return area.replace(" areas", "").replace(" and pallidum", "")


def main() -> None:
    all_dfs = {v: pd.read_csv(VARIANT_DIRS[v] / "results.csv") for v in ["A", "B", "C"]}
    dims = np.arange(1, N_DIMS_MAX + 1)

    # ================= Figures: uniform dots, y-axis shared per row (across variants) =================
    for tier_name in ["good", "good_mua"]:
        for trial_type, lick_flag in CONDITIONS:
            for kind, ylabel_suffix in [("raw", ""), ("excess", " (observed - shuffle null)")]:
                fig, axes = plt.subplots(3, 3, figsize=(19, 20))
                for row, (area_a, area_b) in enumerate(AREA_PAIRS):
                    row_aggs = {}
                    for col, variant in enumerate(["A", "B", "C"]):
                        df = all_dfs[variant]
                        sub = df[(df.quality_tier == tier_name) & (df.trial_type == trial_type)
                                 & (df.lick_flag == lick_flag) & (df.area_a == area_a) & (df.area_b == area_b)]
                        aggs = {}
                        for cohort in ["R+", "R-"]:
                            coh = sub[sub.reward_group == cohort].copy()
                            if coh.empty:
                                continue
                            coh["excess"] = coh["observed_corr"] - coh["shuffle_null_mean"]
                            value_col = "observed_corr" if kind == "raw" else "excess"
                            aggs[cohort] = coh.groupby("dimension")[value_col].agg(["mean", "sem"]).reindex(dims)
                        row_aggs[variant] = aggs

                    # shared y-limits across this row's 3 variant panels
                    all_vals = []
                    for variant, aggs in row_aggs.items():
                        for cohort, agg in aggs.items():
                            all_vals.append((agg["mean"] - agg["sem"]).dropna())
                            all_vals.append((agg["mean"] + agg["sem"]).dropna())
                    y_lo = min(v.min() for v in all_vals if len(v))
                    y_hi = max(v.max() for v in all_vals if len(v))
                    pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.01
                    if kind == "excess":
                        y_lo = min(y_lo, 0)

                    for col, variant in enumerate(["A", "B", "C"]):
                        ax = axes[row, col]
                        for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                            if cohort not in row_aggs[variant]:
                                continue
                            agg = row_aggs[variant][cohort]
                            ax.plot(dims, agg["mean"], color=color, linewidth=1, alpha=0.7)
                            ax.fill_between(dims, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=color, alpha=0.13)
                            ax.scatter(dims, agg["mean"], s=45, color=color, alpha=0.85, edgecolor="none", zorder=5)
                        if kind == "excess":
                            ax.axhline(0, color="black", linewidth=0.8)
                        ax.set_ylim(y_lo - pad, y_hi + pad)
                        ax.tick_params(labelleft=True)
                        ax.set_title(f"{short(area_a)} vs {short(area_b)}, variant {variant}", fontsize=10)
                        ax.set_xlabel("canonical dimension")
                        ax.set_ylabel(f"canonical correlation{ylabel_suffix}")
                        square(ax)
                fig.suptitle(f"{tier_name}, {trial_type} (lick_flag={lick_flag}), {kind} correlation, variants A/B/C "
                             f"as subplots (5ms bins, dead zone excluded; y-axis shared within each row for direct "
                             f"cross-variant comparison)", fontsize=13)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                out_name = f"variants_ABC_{kind}_{tier_name}_{trial_type}_lick{lick_flag}.png"
                fig.savefig(ARTIFACTS_DIR / out_name, dpi=110)
                plt.close(fig)
                print(f"Wrote {out_name}")

    # ================= Lag-0 summary heatmaps: pairs x conditions, one panel per variant =================
    for tier_name in ["good", "good_mua"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        for col, variant in enumerate(["A", "B", "C"]):
            df = all_dfs[variant]
            d1 = df[(df.quality_tier == tier_name) & (df.dimension == 1)]
            mat = np.full((len(AREA_PAIRS), len(CONDITIONS)), np.nan)
            for i, (area_a, area_b) in enumerate(AREA_PAIRS):
                for j, (trial_type, lick_flag) in enumerate(CONDITIONS):
                    cell = d1[(d1.area_a == area_a) & (d1.area_b == area_b)
                              & (d1.trial_type == trial_type) & (d1.lick_flag == lick_flag)]
                    if len(cell):
                        mat[i, j] = cell["observed_corr"].mean()

            ax = axes[col]
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
            ax.set_xticks(range(len(CONDITIONS)))
            ax.set_xticklabels([f"{tt.replace('_trial','')}\nlick={lf}" for tt, lf in CONDITIONS], fontsize=8)
            ax.set_yticks(range(len(AREA_PAIRS)))
            ax.set_yticklabels([f"{short(a)} vs\n{short(b)}" for a, b in AREA_PAIRS], fontsize=8)
            for i in range(len(AREA_PAIRS)):
                for j in range(len(CONDITIONS)):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                                color="white" if mat[i, j] < np.nanmean(mat) else "black")
            ax.set_title(f"variant {variant}")
            fig.colorbar(im, ax=ax, fraction=0.046, label="dim-1 canonical correlation")
        fig.suptitle(f"Lag-0 canonical correlation summary (dim 1, mean across sessions x cohorts), "
                     f"quality tier = {tier_name}")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / f"heatmap_lag0_summary_{tier_name}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote heatmap_lag0_summary_{tier_name}.png")


if __name__ == "__main__":
    main()
