"""105 -- Full comparison with the cohort-specific learning trials (ssl-learning-trial-identification 020;
sigma = 1, whisker-direction gate), user request 2026-09-25: show raw decoding in each stage, the decoding
difference, the difference relative to the shift null, and the comparison with placebo splits.

Comparisons (columns): A around the learning trial, learners R+ vs R- (unpaired); B/C learners, learning
trial vs midpoint, R+ / R- (paired); D around the midpoint, learners vs non-learners (R+, R-, merged).
Metrics (rows):
  1 raw matched balanced accuracy, pre and post                       (095 runs)
  2 raw difference post - pre                                        (095)
  3 difference relative to the session-level shift null               (095)
  4 placebo percentile: fraction of the session's placebo-split changes (|k - split| >= EXCLUDE_NEAR,
    same estimator, 104) below the change at the split; 0.5 = no different from arbitrary splits
Tests: vs 0 (rows 2-3) or vs 0.5 (row 4) with Wilcoxon + t; between groups Mann-Whitney + Welch (unpaired)
or Wilcoxon + paired t (paired); row 1 pre vs post Wilcoxon.
Inputs: 095_lt_split_shiftnull_nodisengagedrop_lt-lt_eval_whole_brain.parquet (LT split),
        095_lt_split_shiftnull_halfsplit_nodisengagedrop_whole_brain.parquet (midpoint),
        104_placebo_all_whole_brain.parquet, ssl-learning-trial-identification/artifacts/020_lt_eval.csv
Outputs: figures/whole_brain/learning/105_hitmiss_cohortrule_full_<window>.png, 105_cohortrule_full_stats.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, ttest_rel, wilcoxon

OUT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("q034", OUT / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(q034)
LT_TABLE = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "020_lt_eval.csv"
F_LT = "095_lt_split_shiftnull_nodisengagedrop_lt-lt_eval_whole_brain.parquet"
F_MID = "095_lt_split_shiftnull_halfsplit_nodisengagedrop_whole_brain.parquet"
F_PLC = "104_placebo_all_whole_brain.parquet"
WINDOWS = {"sensory": "sensory 5-50ms", "baseline": "baseline -200..-10ms", "sensory_minus_base": "sensory minus baseline"}
COL = {"R+": "#00B400", "R-": "#C800C8", "merged": "#2c5f5b"}
EXCLUDE_NEAR = 5
# Minimum pre-learning epoch (added 2026-09-25): learners whose LT split has fewer than MIN_PRE hits or misses
# before the LT are dropped (as learners) from every comparison; 0 = no extra filter (decoding already needs 2).
import os  # noqa: E402
MIN_PRE = int(os.environ.get("SSL_MIN_PRE_PER_CLASS", "0"))
SUF = f"_minpre{MIN_PRE}" if MIN_PRE else ""
METRICS = [("raw", "raw matched balanced accuracy\n(pre = open, post = filled)"), ("draw", "raw change post - pre"),
           ("dnull", "change relative to shift null"), ("pct", "placebo percentile\n(0.5 = like arbitrary splits)")]


def load095(f, tag):
    d = pd.read_parquet(OUT / f)
    d = d[d.skipped_reason.isna()]
    out = d[["session_id", "window"]].copy()
    out[f"pre_{tag}"], out[f"post_{tag}"] = d.acc_pre_matched.values, d.acc_post_matched.values
    out[f"draw_{tag}"] = (d.acc_post_matched - d.acc_pre_matched).values
    out[f"dnull_{tag}"] = ((d.acc_post_matched - d.nullmean_post_matched) - (d.acc_pre_matched - d.nullmean_pre_matched)).values
    return out


def percentiles():
    p = OUT / F_PLC
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d = d[d.skipped_reason.isna()]
    rows = []
    for (sid, w), g in d.groupby(["session_id", "window"]):
        pl = g[g.kind == "placebo"]
        r = dict(session_id=sid, window=w)
        for kind, tag in (("lt", "lt"), ("midpoint", "mid")):
            s = g[g.kind == kind]
            if s.empty:
                continue
            k, v = s.split_k.iloc[0], s.delta.iloc[0]
            ref = pl[(pl.split_k - k).abs() >= EXCLUDE_NEAR].delta.to_numpy()
            if len(ref) >= 3:
                r[f"pct_{tag}"] = (np.sum(ref < v) + 0.5 * np.sum(ref == v)) / len(ref)
        rows.append(r)
    return pd.DataFrame(rows)


def t1(x, mu):
    x = x[~np.isnan(x)]
    return (np.nan, np.nan) if len(x) < 3 else (wilcoxon(x - mu).pvalue, ttest_1samp(x, mu).pvalue)


def tp(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return (np.nan, np.nan) if ok.sum() < 3 else (wilcoxon(a[ok], b[ok]).pvalue, ttest_rel(a[ok], b[ok]).pvalue)


def tu(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    return (np.nan, np.nan) if min(len(a), len(b)) < 2 else (mannwhitneyu(a, b).pvalue, ttest_ind(a, b, equal_var=False).pvalue)


def main():
    tab = pd.read_csv(LT_TABLE)[["session_id", "reward_group", "group"]]
    m = load095(F_LT, "lt").merge(load095(F_MID, "mid"), on=["session_id", "window"], how="outer")
    pc = percentiles()
    if pc is not None:
        m = m.merge(pc, on=["session_id", "window"], how="left")
    for c in ("pct_lt", "pct_mid"):
        if c not in m:
            m[c] = np.nan
    m = m.merge(tab, on="session_id")
    for c in [c for c in m.columns if c.endswith("_lt")]:
        m.loc[m.group != "learner", c] = np.nan
    if MIN_PRE:
        raw = pd.read_parquet(OUT / F_LT)
        raw = raw[raw.skipped_reason.isna()].drop_duplicates("session_id")
        ok = set(raw[(raw.pre_hit >= MIN_PRE) & (raw.pre_miss >= MIN_PRE)].session_id)
        drop = (m.group == "learner") & ~m.session_id.isin(ok)
        print(f"MIN_PRE={MIN_PRE}: dropping {m[drop].session_id.nunique()} learner sessions")
        m = m[~drop]
    m.to_csv(OUT / f"105_cohortrule_full_persession{SUF}.csv", index=False)

    def v(g, metric, tag):
        return g[f"{metric}_{tag}"].to_numpy(dtype=float)

    stats = []
    for w, wl in WINDOWS.items():
        g = m[m.window == w]
        L = g[g.group == "learner"]
        cols = [
            ("A. around the LEARNING TRIAL\nlearners: R+ vs R-", "unpaired",
             [("R+ L", "R+", L[L.reward_group == "R+"], "lt"), ("R- L", "R-", L[L.reward_group == "R-"], "lt")]),
            ("B. R+ learners:\nlearning trial vs midpoint", "paired",
             [("LT", "R+", L[L.reward_group == "R+"], "lt"), ("mid", "R+", L[L.reward_group == "R+"], "mid")]),
            ("C. R- learners:\nlearning trial vs midpoint", "paired",
             [("LT", "R-", L[L.reward_group == "R-"], "lt"), ("mid", "R-", L[L.reward_group == "R-"], "mid")]),
            ("D. around the MIDPOINT\nlearners vs non-learners", "unpaired3",
             [(f"{rg} L", rg, gg[gg.group == "learner"], "mid") for rg, gg in (("R+", g[g.reward_group == "R+"]),)]
             + [("R+ NL", "R+", g[(g.reward_group == "R+") & (g.group != "learner")], "mid")]
             + [("R- L", "R-", g[(g.reward_group == "R-") & (g.group == "learner")], "mid"),
                ("R- NL", "R-", g[(g.reward_group == "R-") & (g.group != "learner")], "mid"),
                ("all L", "merged", g[g.group == "learner"], "mid"), ("all NL", "merged", g[g.group != "learner"], "mid")]),
        ]
        fig, axes = plt.subplots(len(METRICS), len(cols), figsize=(5.2 * len(cols), 3.7 * len(METRICS)), constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[1, 1, 1, 2.2]))
        for r, (metric, mlab) in enumerate(METRICS):
            for c, (ctitle, design, groups) in enumerate(cols):
                ax = axes[r, c]
                xs = []
                for i, (lab, rg, gg, tag) in enumerate(groups):
                    x0 = i * (1.0 if design != "unpaired3" else 1.0) + (0.5 * (i // 2) if design == "unpaired3" else 0)
                    xs.append((x0, lab))
                    col = COL[rg]
                    if metric == "raw":
                        pre, post = v(gg, "pre", tag), v(gg, "post", tag)
                        ok = ~(np.isnan(pre) | np.isnan(post))
                        for a_, b_ in zip(pre[ok], post[ok]):
                            ax.plot([x0 - 0.15, x0 + 0.15], [a_, b_], color=col, alpha=0.15, lw=0.7)
                        ax.errorbar([x0 - 0.15, x0 + 0.15], [np.nanmean(pre), np.nanmean(post)],
                                    yerr=[np.nanstd(pre) / np.sqrt(ok.sum() or 1), np.nanstd(post) / np.sqrt(ok.sum() or 1)],
                                    color="k", capsize=2, lw=1.5)
                        ax.scatter([x0 - 0.15], [np.nanmean(pre)], s=36, facecolor="white", edgecolor=col, zorder=5)
                        ax.scatter([x0 + 0.15], [np.nanmean(post)], s=36, color=col, zorder=5)
                        pw = tp(pre, post)[0]
                        ax.text(x0, 0.02, f"n={ok.sum()}\np={pw:.2g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
                        stats.append(dict(window=w, column=ctitle.replace("\n", " "), group=lab, metric="pre vs post",
                                          n=int(ok.sum()), mean_pre=np.nanmean(pre), mean_post=np.nanmean(post), p_nonparam=pw,
                                          p_param=tp(pre, post)[1]))
                    else:
                        vals = v(gg, metric, tag)
                        vv = vals[~np.isnan(vals)]
                        ax.scatter(x0 + np.random.default_rng(i).uniform(-0.12, 0.12, len(vv)), vv, s=12, color=col, alpha=0.6, lw=0)
                        if len(vv):
                            ax.errorbar(x0 + 0.25, vv.mean(), yerr=vv.std() / np.sqrt(len(vv)), color="k", marker="o", capsize=3)
                        mu = 0.5 if metric == "pct" else 0.0
                        pw, pt = t1(vals, mu)
                        ax.text(x0, 0.02, f"n={len(vv)}\np={pw:.2g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
                        stats.append(dict(window=w, column=ctitle.replace("\n", " "), group=lab, metric=metric, n=len(vv),
                                          mean=vv.mean() if len(vv) else np.nan, ref=mu, p_nonparam=pw, p_param=pt))
                if metric != "raw":
                    ax.axhline(0.5 if metric == "pct" else 0, color="#888888", ls=":", lw=1)
                    pairs = [(0, 1)] if design != "unpaired3" else [(0, 1), (2, 3), (4, 5)]
                    txt = []
                    for a_i, b_i in pairs:
                        la, _, ga, ta = groups[a_i]
                        lb, _, gb, tb = groups[b_i]
                        A, B = v(ga, metric, ta), v(gb, metric, tb)
                        pn, pp = tp(A, B) if design == "paired" else tu(A, B)
                        txt.append(f"{la} vs {lb}: {'W' if design == 'paired' else 'MW'} p={pn:.2g}, "
                                   f"{'t' if design == 'paired' else 'Welch'} p={pp:.2g}")
                        stats.append(dict(window=w, column=ctitle.replace("\n", " "), group=f"{la} vs {lb}", metric=metric,
                                          test="paired" if design == "paired" else "unpaired", p_nonparam=pn, p_param=pp))
                    ax.set_title(("" if r else ctitle + "\n") + "\n".join(txt), fontsize=7.2)
                else:
                    ax.set_title(ctitle + "\n(p = pre vs post, paired Wilcoxon)", fontsize=8)
                ax.set_xticks([x for x, _ in xs], [lab for _, lab in xs], fontsize=7.5)
                ax.spines[["top", "right"]].set_visible(False)
                if c == 0:
                    ax.set_ylabel(mlab, fontsize=8.5)
            if metric == "pct":
                for a_ in axes[r]:
                    a_.set_ylim(-0.05, 1.05)
        fig.suptitle(f"Hit/miss decoding, whole brain, {wl} -- cohort-specific learning trials (sigma = 1, whisker gate), disengaged "
                     f"trials kept{f'; learners with >= {MIN_PRE} pre-LT hits and misses' if MIN_PRE else ''}. Rows: raw accuracy, raw change, change vs shift null, placebo percentile. Non-learners include "
                     f"fast non-lickers.", fontsize=11)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"105_hitmiss_cohortrule_full_{w}{SUF}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    st = pd.DataFrame(stats)
    st.to_csv(OUT / f"105_cohortrule_full_stats{SUF}.csv", index=False)
    print(m[m.window == "sensory"].groupby(["reward_group", "group"]).agg(n_lt=("dnull_lt", "count"), n_mid=("dnull_mid", "count"),
                                                                           n_pct_lt=("pct_lt", "count"), n_pct_mid=("pct_mid", "count")))


if __name__ == "__main__":
    main()
