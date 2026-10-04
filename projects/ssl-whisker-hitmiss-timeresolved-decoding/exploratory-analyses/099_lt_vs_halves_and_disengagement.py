"""Two comparisons for the learning_trial split (user request 2026-09-24):

(A) LT split vs split-halves, SAME pipeline (095 with SSL_SPLIT_MODE=half:
    split at the median decoded whisker trial instead of learning_trial;
    same sessions, trials, disengagement drop, windows, size-matching and
    session-level shift null). Per session and window:
      d_lt   = (acc_post - null_post) - (acc_pre - null_pre), matched, LT split
      d_half = same for the halves split
    Figure: per window, scatter d_half vs d_lt (identity line) per cohort,
    and paired pre->post means for both splits side by side. Stats: per
    cohort, Wilcoxon + paired t on d_lt - d_half; Spearman d_lt vs d_half;
    per-split pre->post Wilcoxon + paired t (same as 096) for reference.
    NB not the same as the original 024-based halves figures (time-resolved
    causal 50ms bins, unmatched, different estimator) -- this is the
    like-for-like comparison.

(B) Disengagement sensitivity: the 8 flagged sessions re-run WITHOUT the
    drop (SSL_NO_DISENGAGE=1); table of d_lt with vs without, per window.

Outputs: figures/whole_brain/learning/099_hitmiss_lt_vs_halves_whole_brain_<scope>.png,
         099_lt_vs_halves_stats.csv, 099_disengagement_sensitivity.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
WINDOWS = {"sensory": "sensory 5-50ms", "baseline": "baseline -200..-10ms", "sensory_minus_base": "sensory minus baseline"}
SCOPES = {"all": None, "learners": {"good", "moderate"}}
KEYS = ["session_id", "window"]


def load(path: str) -> pd.DataFrame:
    d = pd.read_parquet(OUT_DIR / path)
    d = d[d.skipped_reason.isna()].copy()
    for ep in ("pre", "post"):
        d[f"v_{ep}"] = d[f"acc_{ep}_matched"] - d[f"nullmean_{ep}_matched"]
    d["d"] = d.v_post - d.v_pre
    return d


def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    return float(wilcoxon(a[ok], b[ok]).pvalue), float(ttest_rel(a[ok], b[ok]).pvalue), int(ok.sum())


def main():
    lt = load("095_lt_split_shiftnull_whole_brain.parquet")
    half = load("095_lt_split_shiftnull_halfsplit_whole_brain.parquet")
    cols = KEYS + ["v_pre", "v_post", "d", "matched_n_hit", "matched_n_miss", "n_pre"]
    m = lt[KEYS + ["reward_group", "learning_category", "lt_source", "learning_trial"] + cols[2:]].merge(
        half[cols], on=KEYS, suffixes=("_lt", "_half"))
    m.to_csv(OUT_DIR / "099_lt_vs_halves_persession.csv", index=False)

    stats = []
    for scope, cats in SCOPES.items():
        ms = m if cats is None else m[m.learning_category.isin(cats)]
        fig, axes = plt.subplots(2, len(WINDOWS), figsize=(4.6 * len(WINDOWS), 9), constrained_layout=True)
        for c, (win, wlab) in enumerate(WINDOWS.items()):
            w = ms[ms.window == win]
            ax = axes[0, c]
            lim = np.nanmax(np.abs(w[["d_lt", "d_half"]].to_numpy())) * 1.08
            lines = []
            for cohort in ("R+", "R-"):
                g = w[w.reward_group == cohort]
                ax.scatter(g.d_half, g.d_lt, s=22, color=COHORT_COLOR[cohort], alpha=0.7, lw=0, label=cohort)
                pw, pt, n = paired(g.d_lt.to_numpy(), g.d_half.to_numpy())
                r = spearmanr(g.d_lt, g.d_half)
                lines.append(f"{cohort}: n={n}, LT-half W p={pw:.2g}, t p={pt:.2g}; rho={r.statistic:.2f} (p={r.pvalue:.2g})")
                stats.append(dict(scope=scope, window=win, reward_group=cohort, n=n, mean_d_lt=g.d_lt.mean(),
                                  mean_d_half=g.d_half.mean(), p_wilcoxon_lt_vs_half=pw, p_t_lt_vs_half=pt,
                                  spearman_rho=r.statistic, spearman_p=r.pvalue,
                                  p_wilcoxon_prepost_lt=paired(g.v_pre_lt.to_numpy(), g.v_post_lt.to_numpy())[0],
                                  p_wilcoxon_prepost_half=paired(g.v_pre_half.to_numpy(), g.v_post_half.to_numpy())[0]))
            ax.plot([-lim, lim], [-lim, lim], color="#888888", lw=1, ls="--")
            ax.axhline(0, color="#cccccc", lw=0.8)
            ax.axvline(0, color="#cccccc", lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_box_aspect(1)
            ax.set_xlabel("change at HALVES split (post - pre, acc - shift null)", fontsize=8.5)
            ax.set_ylabel("change at LEARNING-TRIAL split", fontsize=8.5)
            ax.set_title(f"{wlab}\n" + "\n".join(lines), fontsize=7.5)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)

            ax = axes[1, c]
            for k, cohort in enumerate(("R+", "R-")):
                g = w[w.reward_group == cohort]
                for j, (split, ls) in enumerate((("lt", "-"), ("half", "--"))):
                    pre, post = g[f"v_pre_{split}"], g[f"v_post_{split}"]
                    x0 = k * 2.6 + j * 1.15
                    ax.errorbar([x0, x0 + 0.8], [pre.mean(), post.mean()], yerr=[pre.sem(), post.sem()],
                                color=COHORT_COLOR[cohort], ls=ls, marker="o", capsize=3, lw=2)
                    pw, _, _ = paired(pre.to_numpy(), post.to_numpy())
                    ax.text(x0 + 0.4, 0.03, f"p={pw:.2g}", ha="center", fontsize=7,
                            transform=ax.get_xaxis_transform())
            ax.set_xticks([0, 0.8, 1.15, 1.95, 2.6, 3.4, 3.75, 4.55],
                          ["pre", "post", "1st", "2nd", "pre", "post", "1st", "2nd"], fontsize=7.5)
            ax.axhline(0, color="#888888", lw=1, ls=":")
            ax.set_ylabel("matched acc - shift null", fontsize=8.5)
            ax.set_title(f"{wlab}: LT split (solid) vs halves (dashed)\nR+ left, R- right; p = paired Wilcoxon",
                         fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Learning-trial split vs split-halves, identical pipeline (095: size-matched, shift null, disengagement "
                     f"dropped) -- whole brain -- scope: {scope}", fontsize=11)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"099_hitmiss_lt_vs_halves_whole_brain_{scope}.png", dpi=250,
                           bbox_inches="tight")
        plt.close(fig)
    st = pd.DataFrame(stats)
    st.to_csv(OUT_DIR / "099_lt_vs_halves_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(st.round(4).to_string())

    nd_path = OUT_DIR / "095_lt_split_shiftnull_nodisengagedrop_whole_brain_subset.parquet"
    if nd_path.exists():
        nd = load(nd_path.name)
        s = lt[lt.session_id.isin(nd.session_id)][KEYS + ["reward_group", "v_pre", "v_post", "d", "post_hit", "post_miss"]].merge(
            nd[KEYS + ["v_pre", "v_post", "d", "post_hit", "post_miss"]], on=KEYS, suffixes=("_drop", "_nodrop"))
        s.to_csv(OUT_DIR / "099_disengagement_sensitivity.csv", index=False)
        print("\ndisengagement sensitivity (8 flagged sessions; rows skipped in both are absent):")
        print(s.round(3).to_string())


if __name__ == "__main__":
    main()
