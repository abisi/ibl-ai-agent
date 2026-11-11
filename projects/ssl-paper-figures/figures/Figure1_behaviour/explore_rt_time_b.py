"""Exploration (user 2026-11-10): alternatives for view B of explore_rt_time ("within-mouse change"); referenced to the
mouse's own median, the cohorts start at different values. Reads the per-hit table of explore_rt_time
(Figures/_explore/rt_time_<date>_hits.csv; no recompute). Whisker hits (row 1), auditory hits (row 2), first 30 hits,
learners, whisker day 0.

Columns (per mouse, then mean +- s.e.m. over mice; per-hit values smoothed with a centred 5-hit running mean per mouse,
except the binned column):
  1  raw reaction time (no reference);
  2  minus the mouse's median over its first 30 hits (as view B);
  3  minus the mouse's mean over its first 30 hits (centred: the mean over hits is 0 for every mouse);
  4  minus the mouse's median of its first 5 hits (change from the start: every mouse starts near 0);
  5  percent change from the median of the first 5 hits;
  6  within-mouse z-score over the first 30 hits;
  7  column 4 in bins of 5 hits (no smoothing), the first bin is 0 by construction.
Statistics (user 2026-11-10: "add stats"; a mixed model was unstable across references, so per-mouse summaries):
  slope    per mouse, least-squares slope of the (unsmoothed) value on hit number, per 10 hits; within cohort Wilcoxon
           signed-rank vs 0 (and one-sample t); R+ vs R- Mann-Whitney and Welch (columns 1-4 share it: per-mouse offsets
           do not change a slope; 5-6 rescale each mouse);
  late     columns 4, 5, 7 (change from the start): per mouse, mean value over hits 21-30; within cohort Wilcoxon vs 0;
           R+ vs R- Mann-Whitney and Welch;
  bars     time-resolved, on the plotted per-mouse values (positions with >= 8 mice per cohort): coloured bars, within-cohort
           sign-flip cluster test vs 0, columns 4, 5, 7, after the reference hits (where a cohort has moved from its
           start); black bar, R+ vs R- cluster permutation test (compute.cluster_test); 1,000 permutations, cluster P < 0.05.
Output: Figures/_explore/rt_time_b_options_<date>.{pdf,png} and _stats.csv (slope / late tests) + _clusters.csv.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_time_b.py [--date 2026-11-10]
"""

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common import FIGURES_ROOT, two_group_tests  # noqa: E402
from explore_rt_time import N_HITS, TT, pstr  # noqa: E402
from paperstyle import MM, light, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
START = 5  # hits defining the start reference
LATE = (21, 30)
MIN_MICE = 8
FROM_START = {"start", "pct"}
OPTIONS = [("raw", "Raw", "Reaction time (ms)", True), ("med", "− own median", "Δ RT (ms)", True),
           ("mean", "− own mean", "Δ RT (ms)", True), ("start", f"− first {START} hits", "Δ RT (ms)", True),
           ("pct", f"% of first {START} hits", "Δ RT (%)", True), ("z", "z-score in mouse", "RT (z)", True),
           ("start", f"− first {START} hits, bins", "Δ RT (ms)", False)]


def references(d):
    """Per-hit columns of the options for one trial type (rows: licked trials with a reaction time)."""
    g = d.groupby("mouse_id").rt_ms
    s0 = d.mouse_id.map(d[d.hit_number <= START].groupby("mouse_id").rt_ms.median())
    return d.assign(raw=d.rt_ms, med=d.rt_ms - g.transform("median"), mean=d.rt_ms - g.transform("mean"),
                    start=d.rt_ms - s0, pct=100 * (d.rt_ms - s0) / s0,
                    z=(d.rt_ms - g.transform("mean")) / g.transform("std"))


def per_mouse_values(dc, col, smooth):
    """[mouse, x] values as plotted: 5-hit running mean per hit, or the median per bin of 5 hits (x = bin centre)."""
    if smooth:
        v = dc.groupby("mouse_id")[col].transform(lambda x: x.rolling(5, center=True, min_periods=3).mean())
        return dc.assign(x=dc.hit_number, v=v).pivot_table(index="mouse_id", columns="x", values="v")
    return dc.assign(x=(dc.hit_number - 1) // 5 * 5 + 3).pivot_table(index="mouse_id", columns="x", values=col,
                                                                       aggfunc="median")


def sign_flip_clusters(wide, n_perm=1000, alpha=0.05, seed=0):
    """One-sample cluster test vs 0 along the columns of wide [mouse, x]: t per x (nan-aware), clusters of contiguous
    x with p < alpha and one sign, mass = sum of t; null = largest |mass| after flipping the sign of whole mice."""
    data, xs = wide.to_numpy(float), wide.columns.to_numpy()

    def tval(d):
        n = np.sum(np.isfinite(d), 0)
        with np.errstate(all="ignore"):
            t = np.nanmean(d, 0) / (np.nanstd(d, 0, ddof=1) / np.sqrt(n))
        return t, 2 * stats.t.sf(np.abs(t), np.maximum(n - 1, 1))

    def clusters(t, p):
        out, i = [], 0
        while i < len(t):
            if not (np.isfinite(p[i]) and p[i] < alpha):
                i += 1
                continue
            j = i
            while j + 1 < len(t) and np.isfinite(p[j + 1]) and p[j + 1] < alpha and np.sign(t[j + 1]) == np.sign(t[i]):
                j += 1
            out.append((i, j, float(np.nansum(t[i:j + 1]))))
            i = j + 1
        return out

    obs = clusters(*tval(data))
    rng = np.random.default_rng(seed)
    null = np.array([max([abs(m) for *_, m in clusters(*tval(data * rng.choice([-1, 1], (len(data), 1))))] or [0])
                     for _ in range(n_perm)])
    return [{"start": float(xs[i]), "end": float(xs[j]), "mass": m, "p": float((1 + np.sum(null >= abs(m))) / (n_perm + 1))}
            for i, j, m in obs]


def mouse_summaries(dc, col):
    """Per mouse: least-squares slope of col on hit number (per 10 hits) and mean of col over the late hits."""
    out = []
    for (m, c), g in dc.dropna(subset=[col]).groupby(["mouse_id", "cohort"]):
        out.append(dict(mouse_id=m, cohort=c, slope=10 * np.polyfit(g.hit_number, g[col], 1)[0] if len(g) >= 10 else np.nan,
                        late=g[g.hit_number.between(*LATE)][col].mean()))
    return pd.DataFrame(out)


def summary_tests(s, tt, name, var):
    a, b = s[s.cohort == "R+"][var].dropna(), s[s.cohort == "R-"][var].dropna()
    two = two_group_tests(a, b)
    return dict(trial_type=tt, option=name, variable=var, rplus_mean=a.mean(), rminus_mean=b.mean(), rplus_n=len(a),
                rminus_n=len(b), rplus_wilcoxon_p=stats.wilcoxon(a).pvalue, rminus_wilcoxon_p=stats.wilcoxon(b).pvalue,
                rplus_t_p=stats.ttest_1samp(a, 0).pvalue, rminus_t_p=stats.ttest_1samp(b, 0).pvalue,
                mannwhitney_p=two["mannwhitney_p"], welch_p=two["welch_p"])


def bar(ax, cl, step, y, color):
    ax.plot([cl["start"] - step / 2, cl["end"] + step / 2], [y, y], color=color, lw=1, transform=ax.get_xaxis_transform(),
            clip_on=False, solid_capstyle="butt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(datetime.date.today()))
    a = ap.parse_args()
    from compute import cluster_test
    from plot import COHORTS, group_color
    set_nature_rc()
    hits = pd.read_csv(OUT / f"rt_time_{a.date}_hits.csv")
    hits = hits[(hits.hit_number <= N_HITS)].dropna(subset=["rt_ms"])
    tests, clus = [], []
    fig, axs = plt.subplots(2, len(OPTIONS), figsize=(183 * MM, 82 * MM))
    for r, (tt, title) in enumerate(TT):
        d = references(hits[hits.trial_type == tt])
        for k, (col, name, ylab, smooth) in enumerate(OPTIONS):
            ax, step, wide, summ = axs[r, k], (1 if smooth else 5), {}, {}
            for i, c in enumerate(COHORTS):
                dc, color = d[d.cohort == c], group_color(tt, c)
                w = per_mouse_values(dc, col, smooth)
                wide[c] = w.loc[:, w.notna().sum() >= MIN_MICE]
                m, e = w.mean(), w.sem()
                if smooth:
                    ax.fill_between(m.index, m - e, m + e, color=light(color, 0.7), lw=0)
                    ax.plot(m.index, m, color=color, lw=0.7)
                else:
                    ax.errorbar(m.index + (i - 0.5) * 0.8, m, e, color=color, marker="o", ms=1.8, lw=0.7, elinewidth=0.6,
                                capsize=0)
                summ[c] = mouse_summaries(dc, col)
                txt = f"{c} slope {summ[c].slope.mean():+.2g}/10 {pstr(stats.wilcoxon(summ[c].slope.dropna()).pvalue)}"
                if col in FROM_START:
                    txt += f"; late {summ[c].late.mean():+.2g} {pstr(stats.wilcoxon(summ[c].late.dropna()).pvalue)}"
                    after = wide[c].loc[:, wide[c].columns > (START if smooth else 3)]  # after the reference hits
                    for cl in sign_flip_clusters(after):
                        clus.append(dict(trial_type=tt, option=name, test=f"{c} vs 0", **cl))
                        if cl["p"] < 0.05:
                            bar(ax, cl, step, 1.04 + 0.04 * i, color)
                ax.text(0.03, 0.98 - 0.09 * i, txt, transform=ax.transAxes, fontsize=3.5, color=color, va="top")
            both = pd.concat(wide, names=["cohort"]).stack().rename("v").reset_index()
            both = both[both.groupby("x").cohort.transform("nunique") == 2]
            for cl in cluster_test(both, "x", "v"):
                clus.append(dict(trial_type=tt, option=name, test="R+ vs R-", **cl))
                if cl["p"] < 0.05:
                    bar(ax, cl, step, 1.0, "k")
            s = pd.concat(summ.values())
            tests += [summary_tests(s, tt, name, v) for v in (("slope", "late") if col in FROM_START else ("slope",))]
            sl = tests[-2] if col in FROM_START else tests[-1]
            txt = f"R+ vs R- slope MW {sl['mannwhitney_p']:.2g}, W {sl['welch_p']:.2g}"
            if col in FROM_START:
                txt += f"\nlate MW {tests[-1]['mannwhitney_p']:.2g}, W {tests[-1]['welch_p']:.2g}"
            ax.text(0.03, 0.80, txt, transform=ax.transAxes, fontsize=3.5, va="top")
            if col != "raw":
                ax.axhline(0, color="0.6", lw=0.4, ls=":")
            ax.set_xlim(0, N_HITS + 1)
            ax.set_xticks([1, 10, 20, 30])
            ax.tick_params(labelsize=4.5)
            ax.set_ylabel(ylab if k == 0 or ylab != OPTIONS[k - 1][2] else "", fontsize=5, labelpad=1)
            if r == 0:
                ax.set_title(name, fontsize=5.5, pad=8)
            if r == 1:
                ax.set_xlabel("Hit number", fontsize=5)
        axs[r, 0].set_ylabel(f"{title.split(' (')[0]}\nReaction time (ms)", fontsize=5, labelpad=1)
    fig.suptitle("View B alternatives: reference for each mouse's reaction time over its first 30 hits (whisker day 0, "
                 "learners; mean ± s.e.m. over mice)\ntext: per-mouse slope per 10 hits and late change (hits 21-30), "
                 "Wilcoxon vs 0; R+ vs R- Mann-Whitney (MW) and Welch (W); bars: coloured, cohort cluster vs 0 (from-start "
                 "columns); black, R+ vs R- cluster (P < 0.05)", fontsize=5)
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    stem = OUT / f"rt_time_b_options_{a.date}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)
    pd.DataFrame(tests).to_csv(f"{stem}_stats.csv", index=False)
    pd.DataFrame(clus).to_csv(f"{stem}_clusters.csv", index=False)
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(tests).round(4).to_string())
        print(pd.DataFrame(clus).query("p < 0.05").round(4).to_string())
    print(stem)


if __name__ == "__main__":
    main()
