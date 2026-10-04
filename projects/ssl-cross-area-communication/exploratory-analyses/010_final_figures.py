"""Consolidated final figure set under the current methodology (partial_CCA
library, 5ms bins, tier-specific unit floors, refit-per-shuffle null, 3 area
pairs) -- pure replotting from `new_rules_results.csv` (009's output), no
new compute needed since that run already covers all 3 pairs x 2 tiers x
6 sessions (both cohorts) with the corrected shuffle null.

Per Axel's 2026-08-31 request:
- For every correlation-vs-dimension plot, also show the excess correlation
  ABOVE the shuffle null (observed - null), not just the raw value with a
  null band alongside it.
- Replace star-marker significance annotation with marker size/saturation:
  significant dimensions (>=2/3 of that line's sessions individually beat
  their own 500-shuffle null at p<0.05) get large, fully-saturated dots;
  non-significant dimensions get small, desaturated dots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
N_DIMS_MAX = 20
SIG_FRAC_THRESHOLD = 2 / 3
SIG_MARKER = dict(size=140, alpha=1.0)
NONSIG_MARKER = dict(size=35, alpha=0.35)


def short(area: str) -> str:
    return area.replace(" areas", "").replace(" and pallidum", "").replace(" and frontal", "-frontal")


def sig_scatter(ax, x, y, color, is_sig):
    """Plot points with size/saturation encoding significance instead of stars:
    significant = large + fully saturated; non-significant = small + faded."""
    sizes = np.where(is_sig, SIG_MARKER["size"], NONSIG_MARKER["size"])
    alphas = np.where(is_sig, SIG_MARKER["alpha"], NONSIG_MARKER["alpha"])
    rgb = to_rgb(color)
    for xi, yi, si, ai in zip(x, y, sizes, alphas):
        ax.scatter(xi, yi, s=si, color=rgb, alpha=ai, edgecolor="black" if ai > 0.9 else "none",
                   linewidth=0.8, zorder=5)


def square(ax):
    ax.set_box_aspect(1)


def per_session_significance(df: pd.DataFrame) -> pd.Series:
    """Per (dimension,), fraction of rows (sessions) with p<0.05."""
    return df.groupby("dimension")["p_value"].apply(lambda p: (p < 0.05).mean())


def main() -> None:
    r = pd.read_csv(ARTIFACTS_DIR / "new_rules_results.csv")
    dims = np.arange(1, N_DIMS_MAX + 1)

    for tier_name in ["good", "good_mua"]:
        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        for col, (area_a, area_b) in enumerate(AREA_PAIRS):
            pair_df = r[(r.quality_tier == tier_name) & (r.trial_type == "whisker_trial")
                        & (r.area_a == area_a) & (r.area_b == area_b)]
            pair_label = f"{short(area_a)} vs {short(area_b)}"

            ax_raw, ax_excess = axes[0, col], axes[1, col]
            for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                coh = pair_df[pair_df.reward_group == cohort].copy()
                coh["excess"] = coh["observed_corr"] - coh["shuffle_null_mean"]

                obs = coh.groupby("dimension")["observed_corr"].agg(["mean", "sem"]).reindex(dims)
                null_m = coh.groupby("dimension")["shuffle_null_mean"].mean().reindex(dims)
                null_s = coh.groupby("dimension")["shuffle_null_sem"].mean().reindex(dims)
                exc = coh.groupby("dimension")["excess"].agg(["mean", "sem"]).reindex(dims)
                frac_sig = per_session_significance(coh).reindex(dims).fillna(0)
                is_sig = (frac_sig >= SIG_FRAC_THRESHOLD).to_numpy()

                ax_raw.plot(dims, obs["mean"], color=color, linewidth=1, alpha=0.6, zorder=2)
                ax_raw.fill_between(dims, obs["mean"] - obs["sem"], obs["mean"] + obs["sem"], color=color, alpha=0.12, zorder=1)
                sig_scatter(ax_raw, dims, obs["mean"].to_numpy(), color, is_sig)
                ax_raw.fill_between(dims, null_m - null_s, null_m + null_s, color=color, alpha=0.10, hatch="//",
                                     label=f"{cohort} shuffle null (mean+/-SEM)")
                ax_raw.plot(dims, null_m, color=color, linestyle="--", linewidth=1, alpha=0.5)

                ax_excess.axhline(0, color="black", linewidth=0.8)
                ax_excess.plot(dims, exc["mean"], color=color, linewidth=1, alpha=0.6, zorder=2,
                               label=f"{cohort} (observed - shuffle null)")
                ax_excess.fill_between(dims, exc["mean"] - exc["sem"], exc["mean"] + exc["sem"], color=color, alpha=0.15, zorder=1)
                sig_scatter(ax_excess, dims, exc["mean"].to_numpy(), color, is_sig)

            ax_raw.set_title(f"{pair_label}\nraw correlation")
            ax_excess.set_title(f"{pair_label}\ncorrelation ABOVE shuffle null")
            for ax in (ax_raw, ax_excess):
                ax.set_xlabel("canonical dimension")
                square(ax)
            ax_raw.set_ylabel("canonical correlation\n(mean +/- SEM across sessions)")
            ax_excess.set_ylabel("observed - shuffle null\n(mean +/- SEM across sessions)")
            ax_raw.legend(fontsize=6.5)
            ax_excess.legend(fontsize=6.5)

        fig.suptitle(f"Correlation vs. canonical dimension, all 3 area pairs, R+ vs R-, quality tier = {tier_name}\n"
                     f"(variant A, whisker trials, 5ms bins, dead zone excluded; large/dark dots = "
                     f">={SIG_FRAC_THRESHOLD:.0%} of that line's sessions individually beat their own "
                     f"500-shuffle trial-identity null at p<0.05; small/faded = not significant)")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / f"final_01_corr_and_excess_by_cohort_{tier_name}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote final_01_corr_and_excess_by_cohort_{tier_name}.png")

    # ================= Pooled-across-cohort area comparison: raw + excess =================
    for tier_name in ["good", "good_mua"]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        sub = r[(r.quality_tier == tier_name) & (r.trial_type == "whisker_trial")]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for (area_a, area_b), color in zip(AREA_PAIRS, colors):
            pair_df = sub[(sub.area_a == area_a) & (sub.area_b == area_b)].copy()
            pair_df["excess"] = pair_df["observed_corr"] - pair_df["shuffle_null_mean"]
            label = f"{short(area_a)} vs {short(area_b)}"

            obs = pair_df.groupby("dimension")["observed_corr"].agg(["mean", "sem"]).reindex(dims)
            exc = pair_df.groupby("dimension")["excess"].agg(["mean", "sem"]).reindex(dims)
            frac_sig = per_session_significance(pair_df).reindex(dims).fillna(0)
            is_sig = (frac_sig >= SIG_FRAC_THRESHOLD).to_numpy()

            axes[0].plot(dims, obs["mean"], color=color, linewidth=1, alpha=0.6, label=label)
            axes[0].fill_between(dims, obs["mean"] - obs["sem"], obs["mean"] + obs["sem"], color=color, alpha=0.12)
            sig_scatter(axes[0], dims, obs["mean"].to_numpy(), color, is_sig)

            axes[1].plot(dims, exc["mean"], color=color, linewidth=1, alpha=0.6, label=label)
            axes[1].fill_between(dims, exc["mean"] - exc["sem"], exc["mean"] + exc["sem"], color=color, alpha=0.15)
            sig_scatter(axes[1], dims, exc["mean"].to_numpy(), color, is_sig)

        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Raw correlation"); axes[1].set_title("Correlation ABOVE shuffle null")
        for ax in axes:
            ax.set_xlabel("canonical dimension")
            ax.legend(fontsize=8)
            square(ax)
        axes[0].set_ylabel("canonical correlation\n(mean +/- SEM across sessions x cohorts)")
        axes[1].set_ylabel("observed - shuffle null\n(mean +/- SEM across sessions x cohorts)")
        fig.suptitle(f"Canonical correlation across area pairs and dimensions, quality tier = {tier_name}\n"
                     f"(variant A, whisker trials, 5ms bins; large/dark dots = "
                     f">={SIG_FRAC_THRESHOLD:.0%} of sessions individually significant, p<0.05)")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / f"final_02_areas_pooled_corr_and_excess_{tier_name}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote final_02_areas_pooled_corr_and_excess_{tier_name}.png")


if __name__ == "__main__":
    main()
