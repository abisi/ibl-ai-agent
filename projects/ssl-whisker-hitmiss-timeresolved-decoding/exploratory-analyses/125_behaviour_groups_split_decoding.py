"""125 -- Pre/post decoding change by BEHAVIOUR GROUP, with the SAME split rule for every mouse (user 2026-10-01: "forced split
for everyone" and "mid-session split for everyone"; compare groups with the same statistic, one row per mouse, behaviour
group x cohort, pairwise Mann-Whitney AND Welch).
Neural data: 122 (whole brain, A1-trimmed, separate size-matched decoders before / after a split; splits every 4 whisker
trials + the session-half split): hit vs miss 5-50 / 5-100 ms, whisker vs auditory -100..0 ms pre-lick.
Designs:
  forced  split at each session's L6x change point (034: joint whisker + FA model with optional lapse; posterior median of the
          learned-segment start), used whether or not the session passes the learner criteria; 122 split nearest to it
          (within 2 whisker trials);
  mid     the session-half split of the decoded trials (122 "half").
Statistic per mouse: EXCESS = (post - pre accuracy at the split) - mean (post - pre) over the session's placebo splits
>= 10 whisker trials away (placebo = drift reference). Also the percentile among placebo splits.
Behaviour groups (036 rule: L6x, log10 BF > 0.3, P(whisker > FA) > 0.9, no start gates):
  step learner     learner with log10 BF >= 1;
  gradual learner  learner with 0.3 < log10 BF < 1 (weaker evidence for a single step);
  high from start  R+ non-learner already discriminating in the first 20 whisker trials (005 L6 category);
  never licked     R- non-learner whose whisker licking never exceeded FA (005 L6 category);
  no transition    any other non-learner.
Stats (uncorrected, mouse = unit): per group vs 0 (Wilcoxon AND one-sample t); within cohort, every pair of groups
(Mann-Whitney AND Welch); two-way OLS excess ~ group3 * cohort (group3 = step / gradual / non-learner, type-II ANOVA).
Continuous: excess vs max slope of whisker - FA (124 score), Spearman, per cohort.
Outputs: figures/125_behaviour_groups_split.{pdf,png,svg}; 125_per_mouse.csv, 125_group_tests.csv, 125_anova.csv
Run (haas): python 125_behaviour_groups_split_decoding.py
"""

from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
COL = {"R+": "#00B400", "R-": "#C800C8"}
GROUPS = ["step learner", "gradual learner", "high from start", "never licked", "no transition"]
GCOL = {"step learner": "#1a9850", "gradual learner": "#a6d96a", "high from start": "#fdae61", "never licked": "#9e9ac8",
        "no transition": "#bdbdbd"}
MEASURES = [("hitmiss", "5-50ms", "hit/miss 5-50 ms"), ("hitmiss", "5-100ms", "hit/miss 5-100 ms"),
            ("modality_lick", "-100-0ms", "whisker vs auditory, pre-lick")]
EXCLUDE_NEAR = 10
FS = 6.5


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def groups():
    R = pd.read_csv(LTP / "036_joint_cp_relaxed.csv")
    R = R[(R.version == "L6x") & (R.bf_th == 0.3) & (R.p_th == 0.9)].set_index("session_id")
    g = np.where(R.learner & (R.log10_bf >= 1), "step learner",
                 np.where(R.learner, "gradual learner",
                          np.where(R.L6_category == "high_from_start", "high from start",
                                   np.where(R.L6_category == "never_licked", "never licked", "no transition"))))
    out = R[["mouse_id", "reward_group", "learning_category", "log10_bf", "LT", "learner"]].copy()
    out["group"] = g
    out["group3"] = np.where(out.learner & (out.log10_bf >= 1), "step", np.where(out.learner, "gradual", "non-learner"))
    return out


def neural(G):
    d = pd.read_parquet(OUT / f"122_lt_placebo_whole_brain{os.environ.get('SSL_PLACEBO_TAG', '')}.parquet")
    d = d[d.skipped_reason.isna()]
    rows = []
    for (sid, dec, w), g in d.groupby(["session_id", "decoding", "window"]):
        if sid not in G.index:
            continue
        pl_all = g[g.split_k >= 0]
        for design in ("forced", "mid"):
            if design == "forced":
                k = G.loc[sid, "LT"]
                if not np.isfinite(k) or not len(pl_all):
                    continue
                dist = np.abs(pl_all.split_k.to_numpy() - k)
                i = int(np.argmin(dist))
                if dist[i] > 2:
                    continue
                real = pl_all.delta_matched.to_numpy()[i]
                far = dist >= EXCLUDE_NEAR
            else:
                h = g[g.real_for == "half"]
                if not len(h):
                    continue
                real = h.delta_matched.iloc[0]
                far = np.abs(pl_all.n_pre.to_numpy() - h.n_pre.iloc[0]) >= EXCLUDE_NEAR
            pl = pl_all.delta_matched.to_numpy()[far]
            pl = pl[np.isfinite(pl)]
            if len(pl) < 5 or not np.isfinite(real):
                continue
            rows.append(dict(session_id=sid, decoding=dec, window=w, design=design, delta=real,
                             placebo_mean=pl.mean(), excess=real - pl.mean(),
                             pct=float(np.mean(pl < real) + 0.5 * np.mean(pl == real))))
    N = pd.DataFrame(rows)
    return N.merge(G.reset_index(), on="session_id")


def tests(N):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows, an = [], []
    for (dec, w, design), g in N.groupby(["decoding", "window", "design"]):
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            for gr in GROUPS:
                v = x[x.group == gr].excess.to_numpy()
                if len(v) >= 3:
                    rows.append(dict(decoding=dec, window=w, design=design, cohort=rg, test="vs 0", a=gr, b="", n_a=len(v),
                                     n_b=0, mean_a=v.mean(), mean_b=np.nan,
                                     p_nonparam=stats.wilcoxon(v).pvalue if np.any(v != 0) and len(v) >= 5 else np.nan,
                                     p_param=stats.ttest_1samp(v, 0).pvalue))
            for a, b in combinations(GROUPS, 2):
                va, vb = x[x.group == a].excess.to_numpy(), x[x.group == b].excess.to_numpy()
                if min(len(va), len(vb)) >= 3:
                    rows.append(dict(decoding=dec, window=w, design=design, cohort=rg, test="pair", a=a, b=b, n_a=len(va),
                                     n_b=len(vb), mean_a=va.mean(), mean_b=vb.mean(),
                                     p_nonparam=stats.mannwhitneyu(va, vb).pvalue,
                                     p_param=stats.ttest_ind(va, vb, equal_var=False).pvalue))
        m = smf.ols("excess ~ C(group3) * C(reward_group)", data=g).fit()
        a = anova_lm(m, typ=2)
        for term in a.index[:-1]:
            an.append(dict(decoding=dec, window=w, design=design, term=term, F=a.loc[term, "F"], p=a.loc[term, "PR(>F)"],
                           n=len(g)))
    return pd.DataFrame(rows), pd.DataFrame(an)


def figure(N, T, A):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    fig, axes = plt.subplots(len(MEASURES), 2, figsize=(8.27, 7.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, wspace=0.25, hspace=0.75)
    rng = np.random.default_rng(0)
    for r, (dec, w, lab) in enumerate(MEASURES):
        for c, design in enumerate(("forced", "mid")):
            ax = axes[r, c]
            g = N[(N.decoding == dec) & (N.window == w) & (N.design == design)]
            xt, xl = [], []
            x0 = 0
            for rg in ("R+", "R-"):
                for gr in GROUPS:
                    v = g[(g.reward_group == rg) & (g.group == gr)].excess.to_numpy()
                    if not len(v):
                        continue
                    ax.scatter(x0 + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=GCOL[gr], edgecolor=COL[rg], lw=0.5,
                               zorder=2)
                    if len(v) > 1:
                        ax.errorbar(x0, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="_", color="k", ms=10,
                                    elinewidth=1, capsize=0, zorder=3)
                    t = T[(T.decoding == dec) & (T.window == w) & (T.design == design) & (T.cohort == rg)
                          & (T.test == "vs 0") & (T.a == gr)]
                    if len(t):
                        ax.text(x0, 1.0, f"W {pf(t.p_nonparam.iloc[0])}\nt {pf(t.p_param.iloc[0])}",
                                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 1.5)
                    xt.append(x0)
                    xl.append(f"{gr.replace(' ', chr(10))}\n(n={len(v)})")
                    x0 += 1
                x0 += 0.6
            ax.axhline(0, color="0.6", lw=0.6, ls=":")
            ax.set_xticks(xt)
            ax.set_xticklabels(xl, fontsize=FS - 1.5)
            ax.set_ylabel("Δacc at split − placebo mean", fontsize=FS)
            a = A[(A.decoding == dec) & (A.window == w) & (A.design == design)].set_index("term")
            txt = " · ".join(f"{t.replace('C(group3)', 'group').replace('C(reward_group)', 'cohort')} p {pf(a.loc[t, 'p'])}"
                             for t in a.index)
            ax.set_title(f"{lab} — {'forced split (L6x change point)' if design == 'forced' else 'mid-session split'}\n"
                         f"two-way OLS (step/gradual/non-learner × cohort): {txt}", fontsize=FS, pad=14)
            ax.text(0.25, -0.33, "R+", transform=ax.transAxes, color=COL["R+"], fontsize=FS + 1, fontweight="bold")
            ax.text(0.75, -0.33, "R−", transform=ax.transAxes, color=COL["R-"], fontsize=FS + 1, fontweight="bold")
    fig.suptitle("Decoding change at the split vs the session's placebo splits, by behaviour group (one dot per mouse; "
                 "W = Wilcoxon, t = one-sample t vs 0; uncorrected). Groups: L6x rule (log10 BF > 0.3, P(w > FA) > 0.9); step = "
                 "log10 BF ≥ 1.", fontsize=FS + 0.5, y=0.985)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"125_behaviour_groups_split.{ext}", dpi=250)
    plt.close(fig)


def main():
    G = groups()
    N = neural(G)
    sc = pd.read_csv(OUT / "124_per_session.csv")[["session_id", "max_slope"]]
    N = N.merge(sc, on="session_id", how="left")
    N.to_csv(OUT / "125_per_mouse.csv", index=False)
    T, A = tests(N)
    T.to_csv(OUT / "125_group_tests.csv", index=False)
    A.to_csv(OUT / "125_anova.csv", index=False)
    figure(N, T, A)
    pd.set_option("display.width", 220)
    print(N.drop_duplicates(["session_id"]).groupby(["reward_group", "group"]).size().to_string())
    print(A.round(3).to_string(index=False))
    sig = T[(T.p_nonparam < 0.05) | (T.p_param < 0.05)]
    print(sig.round(3).to_string(index=False))
    for (dec, w, design), g in N.groupby(["decoding", "window", "design"]):
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg].dropna(subset=["max_slope"])
            if len(x) > 5:
                rho, p = stats.spearmanr(x.max_slope, x.excess)
                print(f"{dec} {w} {design} {rg}: excess vs max slope rho {rho:.2f} p {p:.3f} n {len(x)}")


if __name__ == "__main__":
    main()
