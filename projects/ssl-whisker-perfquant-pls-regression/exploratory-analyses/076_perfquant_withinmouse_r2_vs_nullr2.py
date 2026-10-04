"""`071`'s within-mouse condition x cohort test, but plotting real R2
against null R2 directly (user correction 2026-09-23: "I am talking
about the perfquant 071 figure" -- clarifying that "R2 over null R2"
meant redoing `071`'s own figure with R2, not the unrelated N-subsample
control `074`/`075`).

Pure post-hoc on `058`/`070`'s already-computed CSVs -- no new fits,
same data `071` used, same per-mouse aggregation (sessions averaged
within mouse; real averaging only at expert stage). Unlike `071`'s bar
chart (which only plotted the above-null Pearson delta), this shows
**test R2 and null R2 as paired bars side by side**, per (day_stage,
cohort, target) cell -- so the null's own level is visible, not just the
real-minus-null summary. Significance stars reuse `071`'s own Wilcoxon
q-values (on above_null_r2, recomputed here identically for
self-containment).

**2026-09-23 follow-up ("Also shown individual mouse lines")**: each
mouse's own (test R2, null R2) pair is now overlaid as a jittered point
pair connected by a thin line -- same paired-line convention as `046`'s
`_paired_panel` -- on top of the (now semi-transparent) bars, so the
per-mouse spread and how consistently test beats null WITHIN a mouse
(not just on average) is directly visible, including the expert-R- cells
where n=6 makes that spread the whole story.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COHORTS = ["R+", "R-"]
STAGES = ["learning", "expert"]
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def _stars(q):
    if np.isnan(q):
        return "n/a"
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "n.s."))


def main():
    d_learning = pd.read_csv(OUT_DIR / "058_perfquant_pls_nulldist_fullpool.csv")
    d_learning["day_stage"] = "learning"
    d_expert = pd.read_csv(OUT_DIR / "070_perfquant_expert_pls_nulldist_fullpool.csv")
    d_expert["day_stage"] = "expert"
    df = pd.concat([d_learning, d_expert], ignore_index=True)

    per_mouse = (df.groupby(["mouse", "reward_group", "day_stage", "target"])
                 .agg(test_r2=("test_r2", "mean"), null_r2=("null_r2_mean", "mean"),
                      above_null_r2=("above_null_r2", "mean")).reset_index())

    records = []
    for stage in STAGES:
        for cohort in COHORTS:
            for target in TARGETS:
                g = per_mouse[(per_mouse.day_stage == stage) & (per_mouse.reward_group == cohort)
                               & (per_mouse.target == target)]
                rec = dict(day_stage=stage, cohort=cohort, target=target, n_mice=len(g))
                for col in ("test_r2", "null_r2", "above_null_r2"):
                    vals = g[col].dropna().values
                    rec[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                    rec[f"{col}_sem"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                vals = g["above_null_r2"].dropna().values
                if len(vals) >= 5:
                    try:
                        rec["p"] = wilcoxon(vals).pvalue
                    except ValueError:
                        rec["p"] = np.nan
                else:
                    rec["p"] = np.nan
                records.append(rec)
    result_df = pd.DataFrame(records)
    mask = result_df["p"].notna()
    q = np.full(len(result_df), np.nan)
    q[mask.values] = bh_fdr(result_df.loc[mask, "p"].values)
    result_df["q"] = q
    result_df.to_csv(OUT_DIR / "076_perfquant_withinmouse_r2_vs_nullr2.csv", index=False)

    print("=== within-mouse test R2 vs null R2, by day_stage x cohort x target ===")
    for stage in STAGES:
        for cohort in COHORTS:
            sub = result_df[(result_df.day_stage == stage) & (result_df.cohort == cohort)]
            print(f"-- {stage} / {cohort} --")
            for _, row in sub.iterrows():
                print(f"    {row['target']:<20} n_mice={int(row['n_mice']):>3}  "
                      f"test_r2={row['test_r2_mean']:+.3f}  null_r2={row['null_r2_mean']:+.3f}  "
                      f"above_null={row['above_null_r2_mean']:+.3f}  p={row['p']:.4g} q={row['q']:.4g} {_stars(row['q'])}")

    # --- Figure: grouped bars + individual mouse paired lines, rows=day_stage, cols=target, x=cohort ---
    fig, axes = plt.subplots(len(STAGES), len(TARGETS), figsize=(4.4 * len(TARGETS), 3.8 * len(STAGES)),
                              constrained_layout=True, squeeze=False)
    bar_w = 0.32
    rng = np.random.default_rng(0)
    for row_i, stage in enumerate(STAGES):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            x = np.arange(len(COHORTS))
            for i, cohort in enumerate(COHORTS):
                row = result_df[(result_df.day_stage == stage) & (result_df.cohort == cohort)
                                 & (result_df.target == target)].iloc[0]
                color = COHORT_COLOR[cohort]
                ax.bar(x[i] - bar_w / 2, row["test_r2_mean"], yerr=row["test_r2_sem"], width=bar_w,
                       color=color, alpha=0.45, capsize=3, zorder=1, label="test R2 (mean)" if i == 0 else None)
                ax.bar(x[i] + bar_w / 2, row["null_r2_mean"], yerr=row["null_r2_sem"], width=bar_w,
                       color=color, alpha=0.18, capsize=3, hatch="//", zorder=1,
                       label="null R2 (mean)" if i == 0 else None)

                # individual mice: jittered (test, null) pair connected by a thin line -- 046's
                # _paired_panel convention -- plotted on top of the (now lighter) mean bars.
                mice = per_mouse[(per_mouse.day_stage == stage) & (per_mouse.reward_group == cohort)
                                  & (per_mouse.target == target)]
                jitter = rng.uniform(-0.06, 0.06, size=len(mice))
                x_test = x[i] - bar_w / 2 + jitter
                x_null = x[i] + bar_w / 2 + jitter
                for xt, xn, tv, nv in zip(x_test, x_null, mice["test_r2"], mice["null_r2"]):
                    ax.plot([xt, xn], [tv, nv], color="black", lw=0.6, alpha=0.35, zorder=2)
                ax.scatter(x_test, mice["test_r2"], s=14, color=color, edgecolors="black", linewidths=0.4,
                           zorder=3, label="per-mouse test R2" if i == 0 else None)
                ax.scatter(x_null, mice["null_r2"], s=14, color="white", edgecolors=color, linewidths=1.0,
                           zorder=3, label="per-mouse null R2" if i == 0 else None)

                star = _stars(row["q"])
                ytop = max(mice["test_r2"].max(), mice["null_r2"].max(),
                           row["test_r2_mean"] + row["test_r2_sem"], row["null_r2_mean"] + row["null_r2_sem"])
                ax.text(x[i], ytop + 0.015, f"{star}\nn={int(row['n_mice'])}", ha="center", va="bottom", fontsize=7)
            ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
            ax.set_xticks(x)
            ax.set_xticklabels(COHORTS, fontsize=9)
            ax.set_title(f"{stage} -- {target.replace('_curve','')}", fontsize=9.5)
            if col_i == 0:
                ax.set_ylabel("R2 (per-mouse points + mean +/- SEM bars)", fontsize=8.5)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6, frameon=False, loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Within-mouse test R2 vs null R2, by condition x cohort, individual mice shown\n"
                 "(* = BH-FDR q<0.05 on above-null R2, Wilcoxon)", fontsize=11.5)
    fig_path = OUT_DIR / "076_perfquant_withinmouse_r2_vs_nullr2_bars.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nsaved {fig_path.name}")
    print("DONE_076")


if __name__ == "__main__":
    main()
