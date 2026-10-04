"""100 -- Pre/post learning-trial hit/miss decoding (095 pipeline: size-
matched, session-level shift null, disengagement dropped, whole brain) with
the STORED learning trial vs the re-identified ones from the side project
`ssl-learning-trial-identification` (artifacts/006_learning_trials_final.csv):
  stored               : H5 learning_trial (40-sample MCMC curve, 5-run rule)
  lt_recommended       : L6 joint whisker+FA Bayesian change point (learners only)
  lt_recommended_broad : L6, else L7 half-way rule on smooth curves
Per version x window x cohort x scope: pre and post (matched acc - shift
null), paired Wilcoxon + t (pre vs post), R+ vs R- MWU + Welch on the change.
Figure per scope: rows = window, cols = LT version; cohorts overlaid,
pre/post mean +- SEM with per-session lines.
Outputs: figures/whole_brain/learning/100_hitmiss_newlt_whole_brain_<scope>.png,
         100_new_learning_trial_stats.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
WINDOWS = {"sensory": "sensory 5-50ms", "baseline": "baseline -200..-10ms", "sensory_minus_base": "sensory minus baseline"}
SCOPES = {"all": None, "learners": {"good", "moderate"}}
VERSIONS = {"stored": "095_lt_split_shiftnull_whole_brain.parquet",
            "lt_recommended": "095_lt_split_shiftnull_lt-lt_recommended_whole_brain.parquet",
            "lt_recommended_broad": "095_lt_split_shiftnull_lt-lt_recommended_broad_whole_brain.parquet"}
FLAGGED = {"learner_floor10", "expert_floor10", "learner_fallback10"}


def load(path):
    d = pd.read_parquet(OUT_DIR / path)
    print(f"{path}: {d.session_id.nunique()} sessions, decoded {d[d.skipped_reason.isna()].session_id.nunique()}")
    d = d[d.skipped_reason.isna()].copy()
    for ep in ("pre", "post"):
        d[f"v_{ep}"] = d[f"acc_{ep}_matched"] - d[f"nullmean_{ep}_matched"]
    d["d"] = d.v_post - d.v_pre
    return d


def main():
    data = {k: load(v) for k, v in VERSIONS.items() if (OUT_DIR / v).exists()}
    stats = []
    for scope, cats in SCOPES.items():
        fig, axes = plt.subplots(len(WINDOWS), len(data), figsize=(4.3 * len(data), 4.0 * len(WINDOWS)),
                                 constrained_layout=True, squeeze=False)
        for c, (ver, d) in enumerate(data.items()):
            ds = d if cats is None else d[d.learning_category.isin(cats)]
            for r, (win, wlab) in enumerate(WINDOWS.items()):
                ax = axes[r, c]
                w = ds[ds.window == win]
                lines = []
                for k, cohort in enumerate(("R+", "R-")):
                    g = w[w.reward_group == cohort]
                    x = np.array([0, 1]) + k * 0.08
                    for row in g.itertuples():
                        ax.plot(x, [row.v_pre, row.v_post], color=COHORT_COLOR[cohort], alpha=0.18, lw=0.7)
                    ax.errorbar(x, [g.v_pre.mean(), g.v_post.mean()], yerr=[g.v_pre.sem(), g.v_post.sem()],
                                color=COHORT_COLOR[cohort], lw=2.5, marker="o", capsize=3)
                    ok = g.dropna(subset=["v_pre", "v_post"])
                    pw = wilcoxon(ok.v_pre, ok.v_post).pvalue if len(ok) >= 3 else np.nan
                    pt = ttest_rel(ok.v_pre, ok.v_post).pvalue if len(ok) >= 3 else np.nan
                    lines.append(f"{cohort} n={len(ok)}: {ok.v_pre.mean():+.3f}->{ok.v_post.mean():+.3f} W p={pw:.2g} t p={pt:.2g}")
                    stats.append(dict(version=ver, scope=scope, window=win, reward_group=cohort, n=len(ok),
                                      mean_pre=ok.v_pre.mean(), mean_post=ok.v_post.mean(), mean_change=ok.d.mean(),
                                      median_n_pre=ok.n_pre.median(), p_wilcoxon=pw, p_t=pt))
                a, b = w[w.reward_group == "R+"].d.dropna(), w[w.reward_group == "R-"].d.dropna()
                if len(a) > 1 and len(b) > 1:
                    pm, pwe = mannwhitneyu(a, b).pvalue, ttest_ind(a, b, equal_var=False).pvalue
                    lines.append(f"R+ vs R- change: MW p={pm:.2g}, Welch p={pwe:.2g}")
                    stats.append(dict(version=ver, scope=scope, window=win, reward_group="R+ vs R-", n=len(a) + len(b),
                                      p_mannwhitney=pm, p_welch=pwe))
                ax.axhline(0, color="#888888", lw=1, ls=":")
                ax.set_xticks([0.04, 1.04], ["pre", "post"])
                ax.set_xlim(-0.3, 1.35)
                ax.set_box_aspect(1)
                ax.set_title(f"{ver} | {wlab}\n" + "\n".join(lines), fontsize=7)
                if c == 0:
                    ax.set_ylabel("matched acc - shift null", fontsize=8.5)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Hit/miss decoding pre vs post learning trial: stored vs re-identified learning trials (whole brain, "
                     f"095 pipeline) -- scope: {scope}", fontsize=11)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"100_hitmiss_newlt_whole_brain_{scope}.png", dpi=250,
                           bbox_inches="tight")
        plt.close(fig)
    st = pd.DataFrame(stats)
    st.to_csv(OUT_DIR / "100_new_learning_trial_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(st[st.scope == "all"].round(4).to_string())


if __name__ == "__main__":
    main()
