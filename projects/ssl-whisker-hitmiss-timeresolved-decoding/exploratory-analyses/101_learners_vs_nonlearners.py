"""101 -- Hit/miss decoding change for learners vs non-learners under the new
learning-trial definition (user request 2026-09-25: "compare the same effect
for these non-learners according to the new separation, although what to
compare to, mid-session splits?"), and stored vs new learning trial with
the disengaged trials KEPT ("Keep disengagement trials").

All runs: 095 pipeline (whole brain, size-matched balanced accuracy minus the
session-level shift null, three windows), SSL_NO_DISENGAGE=1.
  learners     = sessions with `lt_lenient_clean` (ssl-learning-trial-
                 identification/artifacts/007_learning_trials_v2.csv)
  non-learners = all other decodable sessions (no identifiable, clean
                 learning event: high_from_start / no_clear_transition /
                 never_licked, or failing the clean-separation gate)
Change = (post - null) - (pre - null), matched.
Groups compared (per cohort x window x scope):
  L@LT    learners split at their new learning trial
  L@half  the same learners split at the session midpoint (median whisker trial)
  NL@half non-learners split at the midpoint -- the only split they have
Tests: pre vs post within each group (paired Wilcoxon + t); L@LT vs L@half
(paired); L@LT vs NL@half and L@half vs NL@half (MWU + Welch).
Figure 1: 101_hitmiss_learners_vs_nonlearners_whole_brain_<scope>.png
Figure 2: 101_hitmiss_stored_vs_newlt_nodrop_whole_brain_<scope>.png (stored vs
          new LT, both with disengaged trials kept; cohorts overlaid)
Stats: 101_learners_vs_nonlearners_stats.csv, 101_stored_vs_newlt_nodrop_stats.csv
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
LT_TABLE = OUT_DIR.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "007_learning_trials_v2.csv"
F_LT = "095_lt_split_shiftnull_nodisengagedrop_lt-lt_lenient_clean_whole_brain.parquet"
F_HALF = "095_lt_split_shiftnull_halfsplit_nodisengagedrop_whole_brain.parquet"
F_STORED = "095_lt_split_shiftnull_nodisengagedrop_whole_brain.parquet"


def load(fname):
    d = pd.read_parquet(OUT_DIR / fname)
    d = d[d.skipped_reason.isna()].copy()
    for ep in ("pre", "post"):
        d[f"v_{ep}"] = d[f"acc_{ep}_matched"] - d[f"nullmean_{ep}_matched"]
    d["d"] = d.v_post - d.v_pre
    return d


def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan, np.nan
    return float(wilcoxon(a[ok], b[ok]).pvalue), float(ttest_rel(a[ok], b[ok]).pvalue)


def unpaired(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return float(mannwhitneyu(a, b).pvalue), float(ttest_ind(a, b, equal_var=False).pvalue)


def learners_vs_nonlearners(lt_tab):
    lt, half = load(F_LT), load(F_HALF)
    learners = set(lt_tab.loc[lt_tab.lt_lenient_clean.notna(), "session_id"])
    cat = lt_tab.set_index("session_id")
    half["group"] = np.where(half.session_id.isin(learners), "learner", "non-learner")
    half["nl_category"] = half.session_id.map(lambda s: cat.L6_category.get(s, "no_curve"))
    stats = []
    for scope, cats in SCOPES.items():
        fig, axes = plt.subplots(len(WINDOWS), 2, figsize=(10, 4.3 * len(WINDOWS)), constrained_layout=True)
        for r, win in enumerate(WINDOWS):
            for c, rg in enumerate(("R+", "R-")):
                ax = axes[r, c]
                sel = lambda d: d[(d.window == win) & (d.reward_group == rg) & (True if cats is None else d.learning_category.isin(cats))]  # noqa: E731
                L_lt = sel(lt)
                h = sel(half)
                L_h, NL_h = h[h.group == "learner"], h[h.group == "non-learner"]
                m = L_lt[["session_id", "d", "v_pre", "v_post"]].merge(L_h[["session_id", "d"]], on="session_id",
                                                                         suffixes=("_lt", "_half"), how="left")
                rng = np.random.default_rng(r * 2 + c)
                groups = [("L@LT", L_lt.d.to_numpy()), ("L@half", L_h.d.to_numpy()), ("NL@half", NL_h.d.to_numpy())]
                for k, (lab, v) in enumerate(groups):
                    ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=COHORT_COLOR[rg],
                               alpha=0.35 if k == 2 else 0.7, lw=0)
                    ax.errorbar(k + 0.28, np.nanmean(v), yerr=np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))), color="k",
                                marker="o", capsize=3)
                for row in m.dropna().itertuples():
                    ax.plot([0, 1], [row.d_lt, row.d_half], color=COHORT_COLOR[rg], alpha=0.15, lw=0.7)
                p_lt = paired(L_lt.v_pre.to_numpy(), L_lt.v_post.to_numpy())
                p_lh = paired(L_h.v_pre.to_numpy(), L_h.v_post.to_numpy())
                p_nl = paired(NL_h.v_pre.to_numpy(), NL_h.v_post.to_numpy())
                p_lt_vs_lh = paired(m.d_lt.to_numpy(), m.d_half.to_numpy())
                p_lt_vs_nl = unpaired(L_lt.d.to_numpy(), NL_h.d.to_numpy())
                p_lh_vs_nl = unpaired(L_h.d.to_numpy(), NL_h.d.to_numpy())
                ax.axhline(0, color="#888888", lw=1, ls=":")
                ax.set_xticks([0, 1, 2], [f"learners\n@new LT\n(n={len(L_lt)})", f"learners\n@midpoint\n(n={len(L_h)})",
                                          f"non-learners\n@midpoint\n(n={len(NL_h)})"], fontsize=8)
                ax.set_ylabel("change post - pre (matched acc - shift null)", fontsize=8)
                ax.set_title(f"{rg} | {WINDOWS[win]}\npre vs post W p: L@LT {p_lt[0]:.2g}, L@half {p_lh[0]:.2g}, NL@half "
                             f"{p_nl[0]:.2g}\nL@LT vs L@half W p={p_lt_vs_lh[0]:.2g}; L@LT vs NL MW p={p_lt_vs_nl[0]:.2g}; "
                             f"L@half vs NL MW p={p_lh_vs_nl[0]:.2g}", fontsize=7.5)
                ax.spines[["top", "right"]].set_visible(False)
                stats.append(dict(scope=scope, window=win, reward_group=rg, n_L=len(L_lt), n_NL=len(NL_h),
                                  mean_d_L_lt=L_lt.d.mean(), mean_d_L_half=L_h.d.mean(), mean_d_NL_half=NL_h.d.mean(),
                                  p_prepost_L_lt_w=p_lt[0], p_prepost_L_lt_t=p_lt[1],
                                  p_prepost_L_half_w=p_lh[0], p_prepost_L_half_t=p_lh[1],
                                  p_prepost_NL_half_w=p_nl[0], p_prepost_NL_half_t=p_nl[1],
                                  p_Llt_vs_Lhalf_w=p_lt_vs_lh[0], p_Llt_vs_Lhalf_t=p_lt_vs_lh[1],
                                  p_Llt_vs_NLhalf_mw=p_lt_vs_nl[0], p_Llt_vs_NLhalf_welch=p_lt_vs_nl[1],
                                  p_Lhalf_vs_NLhalf_mw=p_lh_vs_nl[0], p_Lhalf_vs_NLhalf_welch=p_lh_vs_nl[1],
                                  nl_categories=NL_h.nl_category.value_counts().to_dict()))
        fig.suptitle(f"Learners (new clean LT) vs non-learners: change in hit/miss decoding. Learners split at the new LT and "
                     f"at the session midpoint; non-learners at the midpoint (disengaged trials kept) -- scope: {scope}",
                     fontsize=10)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"101_hitmiss_learners_vs_nonlearners_whole_brain_{scope}.png",
                           dpi=250, bbox_inches="tight")
        plt.close(fig)
    st = pd.DataFrame(stats)
    st.to_csv(OUT_DIR / "101_learners_vs_nonlearners_stats.csv", index=False)
    return st


def stored_vs_new():
    data = {"stored LT": load(F_STORED), "new LT (lenient, clean)": load(F_LT)}
    stats = []
    for scope, cats in SCOPES.items():
        fig, axes = plt.subplots(len(WINDOWS), 2, figsize=(9, 4.0 * len(WINDOWS)), constrained_layout=True)
        for c, (ver, d) in enumerate(data.items()):
            ds = d if cats is None else d[d.learning_category.isin(cats)]
            for r, (win, wlab) in enumerate(WINDOWS.items()):
                ax = axes[r, c]
                w = ds[ds.window == win]
                lines = []
                for k, rg in enumerate(("R+", "R-")):
                    g = w[w.reward_group == rg]
                    x = np.array([0, 1]) + k * 0.08
                    for row in g.itertuples():
                        ax.plot(x, [row.v_pre, row.v_post], color=COHORT_COLOR[rg], alpha=0.15, lw=0.7)
                    ax.errorbar(x, [g.v_pre.mean(), g.v_post.mean()], yerr=[g.v_pre.sem(), g.v_post.sem()],
                                color=COHORT_COLOR[rg], lw=2.5, marker="o", capsize=3)
                    pw, pt = paired(g.v_pre.to_numpy(), g.v_post.to_numpy())
                    lines.append(f"{rg} n={len(g)}: {g.v_pre.mean():+.3f}->{g.v_post.mean():+.3f} W p={pw:.2g} t p={pt:.2g}")
                    stats.append(dict(version=ver, scope=scope, window=win, reward_group=rg, n=len(g), mean_pre=g.v_pre.mean(),
                                      mean_post=g.v_post.mean(), p_w=pw, p_t=pt))
                pm, pwe = unpaired(w[w.reward_group == "R+"].d.to_numpy(), w[w.reward_group == "R-"].d.to_numpy())
                lines.append(f"R+ vs R- change: MW p={pm:.2g}, Welch p={pwe:.2g}")
                stats.append(dict(version=ver, scope=scope, window=win, reward_group="R+ vs R-", p_mw=pm, p_welch=pwe))
                ax.axhline(0, color="#888888", lw=1, ls=":")
                ax.set_xticks([0.04, 1.04], ["pre", "post"])
                ax.set_box_aspect(1)
                ax.set_title(f"{ver} | {wlab}\n" + "\n".join(lines), fontsize=7)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Stored vs new learning trial, disengaged trials kept (whole brain) -- scope: {scope}", fontsize=10)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"101_hitmiss_stored_vs_newlt_nodrop_whole_brain_{scope}.png",
                           dpi=250, bbox_inches="tight")
        plt.close(fig)
    st = pd.DataFrame(stats)
    st.to_csv(OUT_DIR / "101_stored_vs_newlt_nodrop_stats.csv", index=False)
    return st


def main():
    pd.set_option("display.width", 250)
    lt_tab = pd.read_csv(LT_TABLE)
    s1 = learners_vs_nonlearners(lt_tab)
    print(s1[s1.scope == "all"].drop(columns=["nl_categories"]).round(4).to_string())
    print(s1[(s1.scope == "all") & (s1.window == "sensory")][["reward_group", "nl_categories"]].to_string())
    if (OUT_DIR / F_STORED).exists():
        s2 = stored_vs_new()
        print(s2[s2.scope == "all"].round(4).to_string())


if __name__ == "__main__":
    main()
