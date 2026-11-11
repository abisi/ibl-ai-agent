"""Panel i preview (user 2026-11-09): data only, no learning curves. Reads the panel's CSV (Fig1_aligned_curves.csv,
kind == 'data') of a snapshot; two rows: per trial, and bins of 5 trials (per-mouse mean, then mean +- s.e.m. over mice).
Also: what the shared penalty (glm_v6) changes against glm_v5 (penalty chosen per fold), per session.
Output: Figures/_explore/panel_i_data_only_<date>.png and glm_v6_vs_v5_<date>.csv (printed summary).
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_aligned_data_only.py --date 2026-11-09
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
from common import CACHE_DIR, FIGURES_ROOT, load_spec, snapshot_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("fig1_plot", FIG_DIR / "plot.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
X = "trial_from_first_whisker_hit"
TT = ["whisker_trial", "no_stim_trial", "auditory_trial"]
TITLE = {"whisker_trial": "Whisker", "no_stim_trial": "False alarm", "auditory_trial": "Auditory"}


def panel_i(date):
    """Per-mouse values per trial (top) and per 5-trial bin (bottom; whisker trial 0, a hit by definition, left out of
    the bins, user 2026-11-09). Bars: cohort clusters (Figure 1 cluster_test: Welch t per x, cluster mass, 1,000
    permutations of cohort labels, p < 0.05). Bins also: Mann-Whitney per bin (stars, uncorrected) and Welch."""
    _c = importlib.util.spec_from_file_location("fig1_compute", FIG_DIR / "compute.py")
    C = importlib.util.module_from_spec(_c)
    _c.loader.exec_module(C)
    df = pd.read_csv(snapshot_dir(load_spec(FIG_DIR), date) / "data" / "Fig1_aligned_curves.csv")
    df["bin"] = np.floor(df[X] / 5).astype(int)  # bin 0 = trials 0-4, bin -1 = trials -5..-1
    rows, fig = [], plt.figure(figsize=(7.2, 6.2))
    axes = fig.subplots(3, 3, sharey=True)
    for k, tt in enumerate(TT):
        for row, (kind, col) in enumerate((("data", X), ("data", "bin"), ("curve", X))):
            a = axes[row, k]
            g = df[(df.kind == kind) & (df.trial_type == tt)]
            if col == "bin" and tt == "whisker_trial":
                g = g[g[X] > 0]
            pm = g.groupby(["mouse_id", "cohort", col]).p_lick.mean().reset_index()
            xpos = (lambda v: v * 5 + 2) if col == "bin" else (lambda v: v)
            for c in P.COHORTS:
                m = pm[pm.cohort == c].groupby(col).p_lick
                if kind == "curve":  # exact learning curves (as panel i), same alignment: first whisker hit
                    a.plot(m.mean().index, m.mean(), color=P.group_color(tt, c), lw=0.8)
                    a.fill_between(m.mean().index, m.mean() - m.sem(), m.mean() + m.sem(), color=P.group_color(tt, c),
                                   alpha=0.25, lw=0)
                    continue
                a.errorbar(xpos(m.mean().index) + (0.3 if c == "R-" else -0.3), m.mean(), m.sem(), color=P.group_color(tt, c),
                           marker="o", ms=2 if col == "bin" else 1.2, lw=0.6 if col == "bin" else 0, elinewidth=0.5,
                           capsize=0)
            cl = C.cluster_test(pm, col, "p_lick")
            for c in cl:
                if c["p"] < 0.05:
                    a.plot([xpos(c["start"]) - 0.5, xpos(c["end"]) + 0.5], [1.04, 1.04], color="k", lw=1.2,
                           solid_capstyle="butt", clip_on=False)
                rows.append({"trial_type": tt, "kind": kind, "x": col, **c})
            if col == "bin":
                for b, gb in pm.groupby("bin"):
                    r, m_ = gb[gb.cohort == "R+"].p_lick, gb[gb.cohort == "R-"].p_lick
                    if min(len(r), len(m_)) < 5:
                        continue
                    mw = stats.mannwhitneyu(r, m_).pvalue
                    rows.append({"trial_type": tt, "x": "bin_test", "start": b * 5, "end": b * 5 + 4, "n_R+": len(r),
                                 "n_R-": len(m_), "mannwhitney_p": mw,
                                 "welch_p": stats.ttest_ind(r, m_, equal_var=False).pvalue})
                    if mw < 0.05:
                        a.text(xpos(b), 0.93, P.stars(mw), ha="center", fontsize=6)
            a.axvline(0, color="0.6", lw=0.4, ls=":")
            a.set_xlim(-5.5, 30.5)
            a.set_ylim(-0.02, 1.02)
            a.set_title(f"{TITLE[tt]}, " + ("learning curve" if kind == "curve" else "data per trial" if col == X
                                             else "data, bins of 5 trials"), fontsize=7, pad=8)
            a.set_xlabel("Trials from 1st W hit", fontsize=6)
            a.tick_params(labelsize=6)
    for r in range(3):
        axes[r, 0].set_ylabel("P(lick)", fontsize=7)
    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "Bars: cluster-based permutation test R+ vs R- (p < 0.05). Stars: Mann-Whitney per bin, uncorrected. "
             "Whisker bins exclude trial 0. All rows aligned at the first whisker hit (lick).", ha="center", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    out = FIGURES_ROOT / "_explore" / f"panel_i_data_only_{date}.png"
    fig.savefig(out, dpi=200)
    st = pd.DataFrame(rows)
    st.to_csv(FIGURES_ROOT / "_explore" / f"panel_i_data_only_stats_{date}.csv", index=False)
    print(out)
    print(st.round(4).to_string())


def glm_compare(date):
    def load(v):
        rows = [pd.read_csv(f).assign(session_id=f.name.split("_", 1)[1].replace("_results_metrics.csv", ""),
                                      mouse_id=f.parent.name)
                for f in (CACHE_DIR / v).glob("*/*_results_metrics.csv")]
        return pd.concat(rows)[["mouse_id", "session_id", "test_auc", "test_balanced_accuracy", "test_ll_per_trial",
                                "test_pseudo_r2"]]
    v5, v6 = load("glm_v5"), load("glm_v6")
    coh = pd.read_csv(CACHE_DIR / "glm_v6" / "all_sessions_summary.csv")[["session_id", "cohort"]]
    m = v6.merge(v5, on=["mouse_id", "session_id"], suffixes=("_v6", "_v5")).merge(coh, on="session_id")
    m.to_csv(FIGURES_ROOT / "_explore" / f"glm_v6_vs_v5_{date}.csv", index=False)
    for c, g in [("all", m)] + list(m.groupby("cohort")):
        print(f"\n{c} (n = {len(g)} sessions)")
        for k in ("test_balanced_accuracy", "test_auc", "test_ll_per_trial", "test_pseudo_r2"):
            d = g[f"{k}_v6"] - g[f"{k}_v5"]
            print(f"  {k:24s} v5 {g[f'{k}_v5'].mean():.3f}  v6 {g[f'{k}_v6'].mean():.3f}  diff {d.mean():+.4f} "
                  f"(median {d.median():+.4f}), v6 better in {np.mean(d > 0):.0%}, Wilcoxon p={stats.wilcoxon(d).pvalue:.2g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--glm", action="store_true", help="also compare glm_v6 with glm_v5")
    a = ap.parse_args()
    panel_i(a.date)
    if a.glm:
        glm_compare(a.date)
