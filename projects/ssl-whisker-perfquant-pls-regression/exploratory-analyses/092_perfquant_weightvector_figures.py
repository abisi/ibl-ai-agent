"""Full-pool weight-vector figures (user: "For the weight vector, show
other visualization and do on all data" -- follow-up to `090`'s 12-session
pilot strip plot). Pure post-hoc on `091`'s already-computed full-pool
(88-session) cosine CSV and coef_ cache, no new compute.

Three views of the same underlying question -- do target pairs share a
population-level neuron-weighting, or are they distinct -- each showing
something the others don't:

1. **Summary heatmap**: 3x3 mean-cosine matrix across all 88 sessions,
   same visual language as `090`'s specificity matrix, for direct
   side-by-side comparison. One number per pair, fastest to read.
2. **Distribution, split by cohort**: violin + strip of per-session cosine
   per target pair, full n=88 (up from the pilot's n=12), colored by
   reward_group (R+/R-, this project's standard cohort colors) -- shows
   the SHAPE of the distribution (multimodal? skewed? cohort-separated?)
   that a single mean number in (1) can't.
3. **Neuron-level loadings scatter**: for each target pair, three example
   sessions (lowest, median, highest cosine for that pair) plotted as
   weight_A[neuron] vs weight_B[neuron], one point per neuron -- shows
   WHAT a given cosine value actually looks like at the neuron level
   (tight diagonal line vs. scattered cloud vs. anti-diagonal), which
   niether the scalar heatmap nor the distribution plot can.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
TARGET_SHORT = {"whisker_curve": "whisker", "falsealarm_curve": "falsealarm", "performance_curve": "performance"}
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
PAIRS = [("whisker_curve", "falsealarm_curve"), ("whisker_curve", "performance_curve"),
         ("falsealarm_curve", "performance_curve")]


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


def stars(q):
    if np.isnan(q):
        return ""
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else ""))


def main():
    cos_df = pd.read_csv(OUT_DIR / "091_perfquant_weightvector_fullpool_cosine.csv")
    with open(OUT_DIR / "091_perfquant_weightvector_fullpool_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    by_sid = {r["session_id"]: r for r in cache}
    n_sessions = cos_df["session_id"].nunique()

    stats_rows = []
    for ta, tb in PAIRS:
        vals = cos_df[(cos_df.target_a == ta) & (cos_df.target_b == tb)]["cosine"].to_numpy()
        wp = wilcoxon(vals).pvalue
        tp = ttest_1samp(vals, 0.0).pvalue
        stats_rows.append(dict(target_a=ta, target_b=tb, n=len(vals), mean=vals.mean(), median=np.median(vals),
                                wilcoxon_p=wp, ttest_p=tp))
    stats_df = pd.DataFrame(stats_rows)
    stats_df["wilcoxon_q"] = bh_fdr(stats_df["wilcoxon_p"].to_numpy())
    stats_df["ttest_q"] = bh_fdr(stats_df["ttest_p"].to_numpy())
    stats_df.to_csv(OUT_DIR / "092_perfquant_weightvector_fullpool_stats.csv", index=False)
    print(stats_df.to_string(index=False))

    # ================= Figure 1: summary heatmap =================
    mat = np.full((3, 3), np.nan)
    qmat = np.full((3, 3), np.nan)
    for _, row in stats_df.iterrows():
        i, j = TARGETS.index(row["target_a"]), TARGETS.index(row["target_b"])
        mat[i, j] = mat[j, i] = row["mean"]
        qmat[i, j] = qmat[j, i] = row["wilcoxon_q"]
    np.fill_diagonal(mat, 1.0)

    fig1, ax1 = plt.subplots(figsize=(7, 5.6), constrained_layout=True)
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            if i == j:
                txt = "1.000\n(self)"
            else:
                txt = f"{mat[i, j]:+.3f}\n{stars(qmat[i, j])}"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=11,
                     color="white" if abs(mat[i, j]) > 0.55 else "black", fontweight="bold" if i == j else "normal")
    ax1.set_xticks(range(3)); ax1.set_xticklabels([TARGET_SHORT[t] for t in TARGETS], fontsize=10)
    ax1.set_yticks(range(3)); ax1.set_yticklabels([TARGET_SHORT[t] for t in TARGETS], fontsize=10)
    plt.colorbar(im, ax=ax1, label="mean PLS weight-vector cosine similarity", shrink=0.85)
    ax1.set_title(f"Weight-vector cosine similarity\nfull pool (n={n_sessions} sessions), "
                  f"'*' = BH-FDR q<0.05 vs 0 (Wilcoxon)", fontsize=10.5)
    fig1_path = OUT_DIR / "092_perfquant_weightvector_matrix.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
    fig1.savefig(fig1_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig1_path.name} (+ .pdf)")

    # ================= Figure 2: distribution, violin + strip, split by cohort =================
    fig2, axes2 = plt.subplots(1, 3, figsize=(4.6 * 3, 5), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (ta, tb) in zip(axes2, PAIRS):
        sub = cos_df[(cos_df.target_a == ta) & (cos_df.target_b == tb)]
        vals = sub["cosine"].to_numpy()
        parts = ax.violinplot([vals], positions=[0], showmeans=False, showextrema=False, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#cccccc"); pc.set_alpha(0.5)
        for cohort in ("R+", "R-"):
            csub = sub[sub.reward_group == cohort]
            jitter = rng.uniform(-0.15, 0.15, size=len(csub))
            ax.scatter(jitter, csub["cosine"], s=16, color=COHORT_COLOR[cohort], alpha=0.65,
                       edgecolors="none", label=f"{cohort} (n={len(csub)})", zorder=3)
        mean, sem = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(0, mean, yerr=sem, fmt="D", color="#222222", markersize=8, capsize=4, zorder=4)
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        row = stats_df[(stats_df.target_a == ta) & (stats_df.target_b == tb)].iloc[0]
        ax.set_title(f"{TARGET_SHORT[ta]} vs {TARGET_SHORT[tb]}  (n={row['n']})\n"
                     f"mean={row['mean']:+.3f}, median={row['median']:+.3f}, q={row['wilcoxon_q']:.2g} {stars(row['wilcoxon_q'])}",
                     fontsize=9.5)
        ax.set_xticks([]); ax.set_xlim(-0.6, 0.6); ax.set_ylim(-1, 1)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if ax is axes2[0]:
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    axes2[0].set_ylabel("PLS weight-vector cosine similarity\n(per session)", fontsize=9.5)
    fig2.suptitle("Distribution of weight-vector cosine similarity, full pool, by cohort", fontsize=11)
    fig2_path = OUT_DIR / "092_perfquant_weightvector_distribution.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    fig2.savefig(fig2_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig2_path.name} (+ .pdf)")

    # ================= Figure 3: neuron-level loadings scatter, low/median/high example per pair =================
    fig3, axes3 = plt.subplots(3, 3, figsize=(4.0 * 3, 4.0 * 3), constrained_layout=True)
    for row_i, (ta, tb) in enumerate(PAIRS):
        sub = cos_df[(cos_df.target_a == ta) & (cos_df.target_b == tb)].sort_values("cosine").reset_index(drop=True)
        med_idx = (sub["cosine"] - sub["cosine"].median()).abs().idxmin()
        picks = [("lowest", sub.iloc[0]), ("median", sub.loc[med_idx]), ("highest", sub.iloc[-1])]
        for col_i, (label, row) in enumerate(picks):
            ax = axes3[row_i][col_i]
            r = by_sid[row["session_id"]]
            i_a, i_b = TARGETS.index(ta), TARGETS.index(tb)
            wa, wb = r["coef_stack"][i_a], r["coef_stack"][i_b]
            ax.scatter(wa, wb, s=6, color="#333333", alpha=0.35, edgecolors="none")
            lim = max(np.abs(wa).max(), np.abs(wb).max()) * 1.1
            ax.plot([-lim, lim], [-lim, lim], color="#bbbbbb", lw=0.8, linestyle="--", zorder=0)
            ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
            ax.axvline(0, color="#dddddd", lw=0.6, zorder=0)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_title(f"{label} cosine: {row['cosine']:+.3f}\n{row['mouse']} (n_units={r['n_units']})", fontsize=9)
            ax.set_xlabel(f"{TARGET_SHORT[ta]} weight", fontsize=8)
            ax.set_ylabel(f"{TARGET_SHORT[tb]} weight", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    fig3.suptitle("Neuron-level PLS weight loadings, one point per neuron -- lowest/median/highest cosine example per pair",
                  fontsize=11)
    fig3_path = OUT_DIR / "092_perfquant_weightvector_loadings_scatter.png"
    fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
    fig3.savefig(fig3_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {fig3_path.name} (+ .pdf)")
    print("DONE_092")


if __name__ == "__main__":
    main()
