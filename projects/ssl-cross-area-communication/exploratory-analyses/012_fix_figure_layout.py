"""Pure replot fix: 011's suptitle overlapped the top-row subplot titles in
the 3x3 variant-comparison figures (tight_layout didn't reserve room for a
2-line suptitle above a 3x3 grid). No compute changes -- reads the already-
saved variant_A/B/C/results.csv and variant_significance_tests.csv directly.
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
VARIANT_DIRS = {v: ARTIFACTS_DIR / f"variant_{v}" for v in ["A", "B", "C"]}
AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1)]
N_DIMS_MAX = 20


def square(ax):
    ax.set_box_aspect(1)


def sig_scatter(ax, x, y, color, is_sig):
    for xi, yi, sig in zip(x, y, is_sig):
        if sig:
            ax.scatter(xi, yi, s=140, color=to_rgb(color), alpha=1.0, edgecolor="black", linewidth=0.8, zorder=5)
        else:
            ax.scatter(xi, yi, s=35, color=to_rgb(color), alpha=0.35, zorder=5)


def main() -> None:
    all_dfs = {v: pd.read_csv(VARIANT_DIRS[v] / "results.csv") for v in ["A", "B", "C"]}
    sig_df = pd.read_csv(ARTIFACTS_DIR / "variant_significance_tests.csv")
    dims = np.arange(1, N_DIMS_MAX + 1)

    for tier_name in ["good", "good_mua"]:
        for trial_type, lick_flag in CONDITIONS:
            for kind, ylabel_suffix in [("raw", ""), ("excess", " (observed - shuffle null)")]:
                fig, axes = plt.subplots(3, 3, figsize=(19, 20))
                for row, (area_a, area_b) in enumerate(AREA_PAIRS):
                    for col, variant in enumerate(["A", "B", "C"]):
                        ax = axes[row, col]
                        df = all_dfs[variant]
                        sub = df[(df.quality_tier == tier_name) & (df.trial_type == trial_type)
                                 & (df.lick_flag == lick_flag) & (df.area_a == area_a) & (df.area_b == area_b)]
                        sig_sub = sig_df[(sig_df.quality_tier == tier_name) & (sig_df.trial_type == trial_type)
                                         & (sig_df.lick_flag == lick_flag) & (sig_df.area_a == area_a)
                                         & (sig_df.area_b == area_b) & (sig_df.variant == variant)]
                        sig_by_dim = sig_sub.set_index("dimension")
                        is_sig = np.array([
                            d in sig_by_dim.index and sig_by_dim.loc[d, "wilcoxon_p"] < 0.05
                            and sig_by_dim.loc[d, "paired_ttest_p"] < 0.05
                            for d in dims
                        ])

                        for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                            coh = sub[sub.reward_group == cohort].copy()
                            if coh.empty:
                                continue
                            coh["excess"] = coh["observed_corr"] - coh["shuffle_null_mean"]
                            value_col = "observed_corr" if kind == "raw" else "excess"
                            agg = coh.groupby("dimension")[value_col].agg(["mean", "sem"]).reindex(dims)

                            ax.plot(dims, agg["mean"], color=color, linewidth=1, alpha=0.6)
                            ax.fill_between(dims, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=color, alpha=0.13)
                            sig_scatter(ax, dims, agg["mean"].to_numpy(), color, is_sig)
                        if kind == "excess":
                            ax.axhline(0, color="black", linewidth=0.8)
                        a_short = area_a.replace(" areas", "").replace(" and pallidum", "")
                        b_short = area_b.replace(" areas", "").replace(" and pallidum", "")
                        ax.set_title(f"{a_short} vs {b_short}, variant {variant}", fontsize=10)
                        ax.set_xlabel("canonical dimension")
                        ax.set_ylabel(f"canonical correlation{ylabel_suffix}")
                        square(ax)
                fig.suptitle(f"{tier_name}, {trial_type} (lick_flag={lick_flag}), {kind} correlation, variants A/B/C "
                             f"as subplots\n(5ms bins, dead zone excluded; large/dark dots = paired Wilcoxon AND "
                             f"paired t-test both p<0.05, observed vs. each session's own shuffle null)", fontsize=13)
                fig.tight_layout(rect=[0, 0, 1, 0.94])
                out_name = f"variants_ABC_{kind}_{tier_name}_{trial_type}_lick{lick_flag}.png"
                fig.savefig(ARTIFACTS_DIR / out_name, dpi=110)
                plt.close(fig)
                print(f"Wrote {out_name}")


if __name__ == "__main__":
    main()
