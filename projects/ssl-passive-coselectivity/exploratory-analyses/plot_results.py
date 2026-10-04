"""Figures for the passive co-selectivity project: bi-responsive fraction
(global + per area), selectivity AUC distributions (global + per area),
direction-change summary. Learning (day==0) arm is the headline; expert
(day>0) arm is reported in the same figures where sample size allows, at
reduced visual weight, given its much smaller N (12 subjects vs 66).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("projects/ssl-passive-coselectivity/artifacts")
FIG_DIR = Path("projects/ssl-passive-coselectivity/report/figures")
WINDOW = "w5_35"
MIN_UNITS_PER_AREA = 200


def plot_bi_responsive_global() -> None:
    per_mouse = pd.read_parquet(ARTIFACTS_DIR / f"bi_responsive_per_mouse_{WINDOW}.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, day_stage in zip(axes, ["learning", "expert"]):
        sub = per_mouse[per_mouse["day_stage"] == day_stage]
        combos = [("passive_pre", "R+"), ("passive_pre", "R-"), ("passive_post", "R+"), ("passive_post", "R-")]
        x = np.arange(len(combos))
        means = [sub[(sub.passive_epoch == e) & (sub.reward_group == r)][f"bi_responsive_{WINDOW}"].mean() for e, r in combos]
        sems = [sub[(sub.passive_epoch == e) & (sub.reward_group == r)][f"bi_responsive_{WINDOW}"].sem() for e, r in combos]
        colors = ["tab:blue", "tab:red"] * 2
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.6, capsize=4)
        for i, (e, r) in enumerate(combos):
            pts = sub[(sub.passive_epoch == e) & (sub.reward_group == r)][f"bi_responsive_{WINDOW}"]
            jitter = np.random.default_rng(1).uniform(-0.1, 0.1, size=len(pts))
            ax.scatter(np.full(len(pts), i) + jitter, pts, color="black", s=12, alpha=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e.split('_')[1]}\n{r}" for e, r in combos])
        ax.set_title(f"{day_stage} (n={sub['mouse_id'].nunique()} mice)")
        ax.set_ylabel("bi-responsive fraction (per mouse)")
    fig.suptitle(f"Fraction of units responsive to both whisker and auditory (response window {WINDOW})")
    fig.tight_layout()
    out = FIG_DIR / "bi_responsive_fraction_global.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_bi_responsive_by_area() -> None:
    per_mouse_area = pd.read_parquet(ARTIFACTS_DIR / f"bi_responsive_per_mouse_area_{WINDOW}.parquet")
    sub = per_mouse_area[per_mouse_area["day_stage"] == "learning"]
    areas = sub["area_group"].dropna().unique()
    area_counts = sub.groupby("area_group").size().sort_values(ascending=False)
    areas = [a for a in area_counts.index]

    fig, axes = plt.subplots(1, len(areas), figsize=(2.2 * len(areas), 4.5), sharey=True)
    for ax, area in zip(np.atleast_1d(axes), areas):
        a = sub[sub["area_group"] == area]
        combos = [("passive_pre", "R+"), ("passive_pre", "R-"), ("passive_post", "R+"), ("passive_post", "R-")]
        x = np.arange(len(combos))
        means = [a[(a.passive_epoch == e) & (a.reward_group == r)][f"bi_responsive_{WINDOW}"].mean() for e, r in combos]
        colors = ["tab:blue", "tab:red"] * 2
        ax.bar(x, means, color=colors, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["pre\nR+", "pre\nR-", "post\nR+", "post\nR-"], fontsize=7)
        ax.set_title(area, fontsize=8)
    axes[0].set_ylabel("bi-responsive fraction")
    fig.suptitle(f"Bi-responsive fraction by area (learning arm, response window {WINDOW})")
    fig.tight_layout()
    out = FIG_DIR / "bi_responsive_fraction_by_area.png"
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def plot_selectivity_distributions() -> None:
    sel = pd.read_parquet(ARTIFACTS_DIR / "selectivity_auc.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, day_stage in zip(axes, ["learning", "expert"]):
        sub = sel[sel["day_stage"] == day_stage]
        groups = [("passive_pre", "R+"), ("passive_pre", "R-"), ("passive_post", "R+"), ("passive_post", "R-")]
        data = [sub[(sub.passive_epoch == e) & (sub.reward_group == r)][f"selectivity_auc_{WINDOW}"].dropna() for e, r in groups]
        parts = ax.violinplot(data, showmedians=True)
        for pc, (e, r) in zip(parts["bodies"], groups):
            pc.set_facecolor("tab:red" if r == "R+" else "tab:blue")
            pc.set_alpha(0.5)
        ax.axhline(0.5, color="grey", linewidth=0.6)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([f"{e.split('_')[1]}\n{r}" for e, r in groups])
        ax.set_title(f"{day_stage} (n={len(sub)} unit-epochs)")
    axes[0].set_ylabel(f"selectivity AUC ({WINDOW})\n(>0.5 = whisker-preferring)")
    fig.suptitle("Selectivity AUC distributions (baseline-corrected whisker vs auditory)")
    fig.tight_layout()
    out = FIG_DIR / "selectivity_auc_distributions.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def plot_direction_change_summary() -> None:
    change = pd.read_parquet(ARTIFACTS_DIR / f"selectivity_change_{WINDOW}.parquet")
    learning = change[change["day_stage"] == "learning"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for rg, color in (("R+", "tab:red"), ("R-", "tab:blue")):
        vals = learning[learning["reward_group"] == rg]["delta_selectivity_auc"].dropna()
        ax.hist(vals, bins=40, alpha=0.5, color=color, label=rg, density=True)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("delta selectivity AUC (post - pre)")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Learning arm: distribution of selectivity change")

    ax = axes[1]
    flip = pd.read_parquet(ARTIFACTS_DIR / f"flip_direction_posthoc_{WINDOW}.parquet")
    if len(flip):
        flip_learning = flip[flip["day_stage"] == "learning"].sort_values("p_value")
        ax.barh(flip_learning["area_group"], -np.log10(flip_learning["p_value"]), color="tab:purple", alpha=0.7)
        ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8, label="p=0.05")
        ax.set_xlabel("-log10(p) Fisher's exact (flip direction x cohort)")
        ax.legend()
    ax.set_title("Flip-direction test by area (learning arm)")
    fig.tight_layout()
    out = FIG_DIR / "direction_change_summary.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_bi_responsive_global()
    plot_bi_responsive_by_area()
    plot_selectivity_distributions()
    plot_direction_change_summary()


if __name__ == "__main__":
    main()
